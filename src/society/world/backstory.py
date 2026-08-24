"""ペルソナ過去情報(サイドカー)の遅延読み。**既定 OFF = 1 バイトも読まない**。

役割分担(2026-08-24 ユーザー決定)
------------------------------------
- **生成レーン**(`scripts/build_persona_backstory.py`)… 事前に 100 万人ぶんの過去情報を
  作って `data/persona_backstory_v2/` へ層別 JSONL(gz 可)で置く。1 行 =
  ``{"pid": …, "backstory": …, …}``(他の欄は自由。本 module は 2 欄しか読まない)。
- **エンジン側**(この module + `Simulation.build_pool_agent` + `deliberate.build_prompt`)
  … プール個体の実体化時に pid で辞書引きして `agent.backstory` を据え、プロンプトの
  自己紹介節の直後へ 1 節として差し込む。**生成物は 1 バイトも書き換えない**。

なぜ**プール record と分ける**のか(サイドカーである理由)
----------------------------------------------------------
`PoolStore` は「presence 判定に要るスリム記述子だけを常駐させ、full record は present 個体
ぶんだけ都度読む」という RAM 設計(`world/pool.py` の docstring)で成り立っている。過去情報を
プール record 本体へ混ぜると (a) 736MB のシャードを作り直すことになり (b) `PoolStore.get` が
返す record が太って `build_agent` の入口(`entry`)の意味論まで変わる。サイドカーなら
**プール生成物にも record の中身にも触れずに**足せる。

RAM(100 万件)
--------------
pid(12B 級の str)+ 本文(数百 B)の dict で概ね 150-300MB。層別に**最初に必要になった層
だけ**読む(遅延)ので、L1 だけの小さなランは L1 ぶんしか載らない。

R1 ドクトリン
-------------
- `pool.backstory_dir` が空(既定)なら `Simulation` は本 module を**一度も呼ばない**
  (`sim._backstory is None`)= 属性が 1 つも生えない = L1 バイト一致。
- サイドカーに無い pid は**無音で骨格のみ**(例外にしない)。件数はログ 1 行に出す。
- 乱数を 1 本も引かない。LLM 呼数も 1 つも動かない(変わるのはプロンプト文字列だけ)。
"""
from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path

log = logging.getLogger("society.backstory")

#: サイドカー 1 行のキー(= 生成レーンと共有する唯一の契約。他の欄は読まない)。
PID_KEY = "pid"
TEXT_KEY = "backstory"

#: 受け入れる拡張子(gz 圧縮と素の JSONL の両方。生成レーンの都合に依存しない)。
SUFFIXES: tuple[str, ...] = (".jsonl.gz", ".jsonl")


class BackstoryStore:
    """pid → 過去情報の遅延辞書(層別ファイルを「最初に要る時」に 1 度だけ読む)。

    受け入れる置き方(生成レーンの流儀に合わせて 4 通りを見る。先に当たったものを使う):
      ``<root>/L4.jsonl.gz`` / ``<root>/L4.jsonl`` / ``<root>/L4/*.jsonl.gz`` /
      ``<root>/L4/*.jsonl``。層別に割られていないサイドカー(``<root>/*.jsonl.gz`` など)
      なら、最初の 1 回で**ルート全体**を読んで以後は読み直さない。
    """

    def __init__(self, root):
        self.root = Path(root)
        self._map: dict[str, str] = {}
        self._loaded: set[str] = set()      # 読み終えた層("" = ルート一括を読んだ印)
        self._seen: set[str] = set()        # 既に読んだファイル(二度読みしない)
        self._miss_logged: set[str] = set()  # 欠損ログを出した層(ログは層ごとに 1 行)
        self.n_hit = 0
        self.n_miss = 0

    # ---- ファイル探索(決定論=同じ root なら常に同じ並び)------------------------- #
    def _files_for(self, layer: str) -> list[Path]:
        pats: list[str] = []
        for suf in SUFFIXES:
            if layer:
                pats += [f"{layer}{suf}", f"{layer}/*{suf}"]
            else:
                pats += [f"*{suf}", f"*/*{suf}"]
        out: list[Path] = []
        seen: set[str] = set()
        for pat in pats:
            for path in sorted(self.root.glob(pat)):
                key = str(path)
                if key not in seen:
                    seen.add(key)
                    out.append(path)
        return out

    def _read(self, files) -> int:
        """JSONL(gz 可)を読んで辞書へ足す。**壊れた行・読めないファイルは飛ばす**。

        一度読んだファイルは二度読まない(層別ファイルが欠けている層の照会で
        ルート一括へ退避したとき、既読の層を読み直さないため)。
        """
        n = 0
        for path in files:
            key = str(path)
            if key in self._seen:
                continue
            self._seen.add(key)
            try:
                opener = gzip.open if path.name.endswith(".gz") else open
                with opener(path, "rt", encoding="utf-8") as f:      # type: ignore[operator]
                    for raw in f:
                        line = raw.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except ValueError:
                            continue        # 生成途中の切れた行 = 黙って飛ばす(落とさない)
                        pid = str(rec.get(PID_KEY, "") or "")
                        text = str(rec.get(TEXT_KEY, "") or "").strip()
                        if pid and text:
                            self._map[pid] = text
                            n += 1
            except OSError as exc:          # noqa: BLE001(読めなくてもランは続ける)
                log.warning("backstory サイドカーが読めない: %s (%s)", path, exc)
        return n

    def _ensure(self, layer: str) -> None:
        """その層を(まだなら)読む。**層につき高々 1 回**しか探索しない。

        層別ファイルが見つからない層は「割られていないサイドカー」か「まだ生成されて
        いない層」のどちらかなので、ルート全体を退路にする(既読ファイルは `_read` が
        飛ばすので、実際に読むのは未読ぶんだけ)。★"" の印だけで早期 return しない:
        それをやると、退路を 1 度通ったあとに**実在する別の層**が読めなくなる。
        """
        key = str(layer or "")
        if key in self._loaded:
            return
        files = self._files_for(key) if key else []
        if not files:
            files = self._files_for("")     # 退路(ルート全体)
            self._loaded.add("")            # ルート一括を走査した印
        n = self._read(files)
        self._loaded.add(key)
        log.info("backstory 読込: layer=%s files=%d 新規=%d 累計=%d",
                 key or "(all)", len(files), n, len(self._map))

    # ---- 取り出し(欠損は無音で "" = 骨格のみ)------------------------------------ #
    def get(self, pid: str, layer: str = "") -> str:
        """pid の過去情報。無ければ **""**(例外にしない = 骨格ペルソナのまま進む)。"""
        if not pid:
            return ""
        self._ensure(str(layer or ""))
        text = self._map.get(str(pid), "")
        if text:
            self.n_hit += 1
            return text
        self.n_miss += 1
        key = str(layer or "")
        if key not in self._miss_logged:
            # ログは**層ごとに 1 行**(25 万人の入場で溢れさせない)。既定の logging 設定でも
            # stderr に出るよう WARNING にする(サイドカーの取りこぼしは設定ミスでありうる)。
            # 実数は `stats()` の n_miss が持つ(ランを止めない = 骨格ペルソナで続行)。
            self._miss_logged.add(key)
            log.warning("backstory: サイドカーに無い pid がある(layer=%s 例=%s)。"
                        "その個体は骨格ペルソナのまま進む(件数は stats().n_miss)",
                        key or "(all)", pid)
        return ""

    def stats(self) -> dict:
        """検収・観測用のカウンタ(世界は 1 ビットも読まない)。"""
        return {"n_hit": int(self.n_hit), "n_miss": int(self.n_miss),
                "n_loaded": len(self._map),
                "layers": sorted(self._loaded)}

    def __len__(self) -> int:
        return len(self._map)

    def __repr__(self):
        return (f"<BackstoryStore root={self.root} loaded={len(self._map)}"
                f" hit={self.n_hit} miss={self.n_miss}>")


def store_of(pool_cfg, repo_root=None) -> "BackstoryStore | None":
    """`pool.backstory_dir` から store を作る。**空(既定)は None = 完全 no-op**。

    相対パスはリポジトリルート基準(`pool.dir` と同じ規約)。ディレクトリが無い場合も
    None を返さず store は作る(中身が 0 件 = 全員が骨格のみ)ほうが「設定したのに黙って
    効かない」を避けられる —— が、**存在しないパスは設定ミス**なので警告 1 行を出す。
    """
    raw = ""
    if pool_cfg is not None:
        try:
            raw = str(pool_cfg.get("backstory_dir", "") or "").strip()
        except (AttributeError, TypeError):  # 旧 config / スタブ
            raw = ""
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        if repo_root is None:
            from ..config import REPO_ROOT as repo_root       # noqa: N813
        path = Path(repo_root) / path
    if not path.is_dir():
        log.warning("pool.backstory_dir が存在しない: %s(全員が骨格ペルソナのまま進む)",
                    path)
    return BackstoryStore(path)
