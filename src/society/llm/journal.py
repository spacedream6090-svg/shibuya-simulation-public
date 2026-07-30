"""LLM 入出力ジャーナル(第71バッチ 2026-07-31)= **プロンプト全文の永続化**。

なぜ要るか
----------
現行の永続化は 2 本しか無く、どちらもプロンプト全文を残さない:
  - `llm_cache.jsonl` = `{key, response}` だけ(key は sha256 なので**逆写像できない**)
  - L1b(`l1b_llm.parquet`)= `{llm_call_id, agent_id, purpose, step, cached}` のメタだけ
つまり「そのエージェントは**何を見せられて**その応答を返したのか」が、ランが終わると
永久に失われる。本選 10 日ランの観察対象(語彙の専門化・噂の分岐・規範の成立)を事後に
検証するには、入力全文が要る。これを埋めるのが本モジュール
(docs/plans/source/dual-mode-requirements.md §2.3-1: 「LLM 出力全文の保存は必須、
world snapshot 間隔は設定可能。容量が逼迫した場合に削るのは snapshot 側」)。

設計
----
- **観測専用**: シム本体はジャーナルを**読まない**。REPLAY(cache_mode=replay)が使うのは
  従来どおり `llm_cache.jsonl`(content-addressed キャッシュ)であって、このジャーナルではない。
  読む経路を作らない限り、ジャーナルは決定論にも k 呼数にも影響しえない(R1)。
- **append-only + multi-member gzip**: `flush_records` 件たまるごとに「完結した gzip メンバ」を
  1 個追記する。gzip は連結メンバをそのまま読めるので、追記のたびに
  ファイルは常に有効な .gz のままで、かつ**メンバ境界=安全な切り詰め点**になる
  (checkpoint からの resume で巻き戻せる)。1 メンバに複数レコードを詰めるのは、
  プロンプトの大半が共通テンプレートで、deflate の 32KB 窓が効くから(実測は下記 §容量)。
- **サイドカー index**: `<name>.index.json` = `{"records": N, "bytes": B}`。
  flush のたびに原子的に更新する。これがあるので resume 時に本体 .gz を読み直さずに
  O(1) で「どこまでが確定分か」が判る(10 日ランの .gz を数え直すのは非現実的)。
  クラッシュで中途半端なメンバが残っていたら、次に開いたときに B へ切り詰めて修復する。
- **resume で二重記録しない**: checkpoint.save が `mark()`(= flush してから records/bytes を
  採る)を保存し、checkpoint.load が `rewind(mark)` でファイルをそこまで切り詰める。
  checkpoint 以降に走ってクラッシュした分の記録は捨てられ、再走した分が同じ seq から
  書き直される(= seq は巻き戻らず、同じ呼び出しが 2 行に増えない)。
  ObserverLogger の part/flush_segment と同じ「checkpoint が真の境界」という流儀。
- **同じ out_dir を作り直したときは追記になる**(上書きしない)。`llm_cache.jsonl` と同じ
  「ディレクトリ単位で append-only」の意味論に揃えてある。resume は必ず既存 out_dir への
  再構築を伴うので、構築時点では「これが resume なのか作り直しなのか」を知りようがなく、
  ここで消すと resume で記録が消える。したがって「ジャーナル行数 == そのランの llm.calls」が
  成り立つのは fresh な out_dir のとき(何回走らせたかは run_manifest.json の history で判る)。

容量(実測は docs/log/devlog に記録。tests/test_llm_journal.py が回帰を見る)
- 1 レコード = seq/key/rng_key/backend/params/cached + prompt 全文 + response 全文。
"""
from __future__ import annotations

import gzip
import json
import os
import zlib
from pathlib import Path
from typing import Any, Iterator

INDEX_SUFFIX = ".index.json"


class LlmJournal:
    """1 バックエンド(= 1 CachedLLM)ぶんの入出力ジャーナル。

    router 配線では子ごとに 1 本(`llm_journal.<child-name>.jsonl.gz`)。
    """

    def __init__(self, path: str | Path, *, flush_records: int = 128) -> None:
        self.path = Path(path)
        self.index_path = self.path.with_name(self.path.name + INDEX_SUFFIX)
        self.flush_records = max(1, int(flush_records))
        self._buf: list[str] = []
        self.seq = 0            # 次に振る seq(= 記録済み総数。バッファ込み)
        self._written = 0       # ディスク上に確定したレコード数
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._adopt_existing()

    # ------------------------------------------------------------------ 既存分
    def _adopt_existing(self) -> None:
        """既に同名ファイルがあれば「確定分の続き」から書けるよう状態を合わせる。

        index があればそれを信じて bytes へ切り詰める(クラッシュで中途半端なメンバが
        残っている場合の修復)。index が無い旧ファイルは寛容リーダで数え直す。
        """
        if not self.path.exists():
            return
        size = self.path.stat().st_size
        idx = _read_index(self.index_path)
        if idx is not None and 0 <= int(idx["bytes"]) <= size:
            if int(idx["bytes"]) < size:
                os.truncate(self.path, int(idx["bytes"]))
            self.seq = self._written = int(idx["records"])
            return
        # index 無し/壊れている: 実ファイルを数える(旧ファイル互換の保険。通常は通らない)
        self.seq = self._written = sum(1 for _ in iter_records(self.path))
        self._write_index()

    # ------------------------------------------------------------------ 記録
    def record(self, *, key: str, rng_key: str, prompt: str, response: str,
               temperature: float, max_tokens: int, think: bool,
               cached: bool, backend: str) -> None:
        row = {
            "seq": self.seq,
            "key": key,
            "rng_key": rng_key,
            "backend": backend,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "think": bool(think),
            "cached": bool(cached),
            "prompt": prompt,
            "response": response,
        }
        self._buf.append(json.dumps(row, ensure_ascii=False))
        self.seq += 1
        if len(self._buf) >= self.flush_records:
            self.flush()

    # ------------------------------------------------------------------ 永続化
    def flush(self) -> None:
        """バッファを「完結した gzip メンバ 1 個」として追記する(= 常に有効な .gz)。"""
        if not self._buf:
            return
        blob = ("\n".join(self._buf) + "\n").encode("utf-8")
        with open(self.path, "ab") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
                gz.write(blob)
        self._written += len(self._buf)
        self._buf.clear()
        self._write_index()

    def close(self) -> None:
        self.flush()

    def _write_index(self) -> None:
        size = self.path.stat().st_size if self.path.exists() else 0
        tmp = self.index_path.with_name(self.index_path.name + ".tmp")
        tmp.write_text(json.dumps({"records": self._written, "bytes": size}),
                       encoding="utf-8")
        os.replace(tmp, self.index_path)

    # -------------------------------------------------- checkpoint 連携(resume)
    def mark(self) -> dict:
        """checkpoint に載せる「確定点」。flush してから採るので bytes と records が対応する。"""
        self.flush()
        return {"records": int(self._written),
                "bytes": int(self.path.stat().st_size if self.path.exists() else 0)}

    def rewind(self, mark: dict | None) -> None:
        """checkpoint の確定点まで巻き戻す(= 再走ぶんが二重記録にならない)。"""
        if not mark:
            return
        records = int(mark.get("records", 0))
        nbytes = int(mark.get("bytes", 0))
        self._buf.clear()
        if self.path.exists() and self.path.stat().st_size > nbytes:
            os.truncate(self.path, nbytes)
        self.seq = self._written = records
        self._write_index()

    # ------------------------------------------------------------------ 参照
    @property
    def n_records(self) -> int:
        """記録したレコード総数(バッファ込み)。"""
        return self.seq


def _read_index(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or "records" not in data or "bytes" not in data:
        return None
    try:
        return {"records": int(data["records"]), "bytes": int(data["bytes"])}
    except (TypeError, ValueError):
        return None


def iter_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """ジャーナルを読む(解析・検証用。**シム本体からは呼ばない**)。

    末尾がクラッシュで切れている(= 最後の gzip メンバが不完全)場合でも、そこまでの
    レコードを返して静かに終わる。壊れた 1 メンバのために全記録を失わないため。
    """
    p = Path(path)
    if not p.exists():
        return
    try:
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except ValueError:          # 途中で切れた行(最終メンバの尻尾)
                    return
    except (OSError, EOFError, gzip.BadGzipFile, zlib.error):   # 不完全メンバ
        return
