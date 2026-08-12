#!/usr/bin/env python
"""ラン成果物の日次バックアップ(**確定分のみ**・増分tar・BagIt 式 sha256 マニフェスト)。

`docs/plans/finals-reliability-plan.md` §3(転送・保管)の主線を 1 スクリプトにしたもの。
本選 10 日ランの一次データ(L1 part 群・checkpoint・サイドカー)を、**走行中のランを
一切邪魔せずに**別ディスク/ネットワーク先へ増分コピーする。

    # 日次(タスクスケジューラ/cron から)
    python scripts/backup_run.py --run-dir runs/finals --dest D:/backup/shibuya
    # ネットワーク先(UNC もそのまま渡せる)
    python scripts/backup_run.py --run-dir runs/finals --dest //nas/share/shibuya
    # 転送先での照合(restore drill の前段。3-2-1-1-0 の「0」)
    python scripts/backup_run.py --verify --dest D:/backup/shibuya --run-id finals

設計の要点(なぜこうなっているか)
--------------------------------
1. **確定分のみを取る**。part の flush は非原子的(`observer/logger.py` は最終名へ直接
   `pq.write_table`)なので、書きかけの part は **footer が閉じていない不正 parquet** で
   あり得る。よって part は
     (a) parquet の完結判定(先頭 magic・末尾 magic・footer 長の整合。pyarrow があれば
         footer の実パースまで)を通り、かつ
     (b) **最新 checkpoint の mtime 以前**に書かれたもの(= checkpoint 境界の内側)
   だけを対象にする。(b) が要るのは footer の都合ではなく **watchdog の巻き戻し**の都合:
   `scripts/watchdog.py` の `_restore_from_backup` は run-dir の part を消して世代から
   復元するため、checkpoint より先の part は**同じ名前で中身が変わりうる**。checkpoint
   境界の内側なら不変(append-only)と見なせる。
   なお `summary.json` があるラン(= 完走済み)は書き手が居ないので (b) を課さない。
2. **走行中のランに触らない**。読み取りは Windows で `FILE_SHARE_DELETE` を立てて開く
   (`_open_shared`。出典は `scripts/live_viewer.py` — 素の `open()` だと finalize の
   `part.unlink()` が WinError 32 で失敗し**シム本体が落ちる**)。書き込みは dest 側だけ。
3. **copy 系のみ**。dest からは何も消さない。元(run-dir)で消えたファイルはバックアップに
   残り続ける(削除を伝播させない = GitLab 事故の教訓)。
4. **増分**。前回の state(size + mtime_ns)と一致するファイルは読みもしない。差分が
   1 件も無ければ **dest には 1 バイトも書かない**(冪等)。
5. **再圧縮しない**。part は zstd 済み parquet・checkpoint と journal は gzip 済みなので、
   増分 tar は store モード(`mode="w|"`)。tar 化の目的は圧縮ではなく**小ファイル多数の
   転送セッション削減**(リサーチ §3-5)。
6. **BagIt 式マニフェスト**(RFC 8493 準拠ではなく「式」)。増分 tar 1 本が
   `bagit.txt` + `bag-info.txt` + `manifest-sha256.txt` + `data/<相対パス>` を含む
   **転送単位ごとのバッグ**になり、dest 直下には累積マニフェストも置く。

dest のレイアウト
-----------------
    <dest>/<run_id>/
      bagit.txt                     … BagIt-Version / エンコーディング
      bag-info.txt                  … Payload-Oxum(累積バイト数.件数)・Bagging-Date ほか
      manifest-sha256.txt           … **累積**マニフェスト("<sha256>  data/<相対パス>")
      backup_state.json             … 増分の状態(size/mtime_ns/sha256/どの増分に居るか)
      increments/inc-0001-<ts>.tar  … --layout tar(既定)。1 本が 1 バッグ
      data/<相対パス>               … --layout tree(展開済みコピー。restore drill 向き)

終了コード: 0=正常 / 1=問題あり(コピー失敗・照合エラー)。運用スクリプトから拾える。
標準ライブラリのみで動く(pyarrow があれば footer 検証を厚くするが、無くても動く)。
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import io
import json
import os
import sys
import tarfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

for _s in (sys.stdout, sys.stderr):              # Windows コンソール(cp932)対策
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SCHEMA = 1
PARQUET_MAGIC = b"PAR1"
STATE_NAME = "backup_state.json"
MANIFEST_NAME = "manifest-sha256.txt"
BAGIT_NAME = "bagit.txt"
BAGINFO_NAME = "bag-info.txt"
PAYLOAD_DIR = "data"
INCREMENTS_DIR = "increments"
CHUNK = 1 << 20

CKPT_PREFIXES = ("ckpt-", "dormant-")            # 同 step の 2 つで 1 世代(pool サイドカー)
CKPT_SUFFIX = ".pkl.gz"

#: 既定で対象外にするもの(派生物・巨大ログ・書きかけ)。--exclude で足せる。
DEFAULT_EXCLUDE = ("*.tmp", "run.out.log", "*.html")
#: run-dir 直下で辿るサブディレクトリ。ほかは派生物(analysis/ panel/ …)として飛ばす。
TRAVERSE_DIRS = ("checkpoint",)


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _ts_tag() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


# --------------------------------------------------------------------------- #
# 非侵襲な読み口(出典: scripts/live_viewer.py `_open_shared`)
# --------------------------------------------------------------------------- #
def _open_shared(path: Path):
    """**読んでいる最中でも相手が消せる**ハンドルで開く(Windows 必須)。

    `scripts/live_viewer.py` から逐語で持ってきた(依存の向きを保つための写し。
    あちらは viewer 専用モジュールで、バックアップ側が import すると
    「観測層の道具がバックアップに依存する」向きが生まれてしまう)。
    なぜ要るか: Python の `open()` は Windows で FILE_SHARE_DELETE を立てないため、
    バックアップが part を開いている間に `observer/logger.py` の finalize が
    `part.unlink()` すると **PermissionError [WinError 32] でシム本体が落ちる**。
    POSIX は open 済みファイルの unlink が元から成功するので素の open で同義。
    """
    if os.name != "nt":
        return open(path, "rb")
    import ctypes
    import msvcrt
    from ctypes import wintypes

    GENERIC_READ = 0x80000000
    SHARE_ALL = 0x01 | 0x02 | 0x04            # READ | WRITE | DELETE
    OPEN_EXISTING = 3
    FILE_ATTRIBUTE_NORMAL = 0x80
    INVALID_HANDLE = ctypes.c_void_p(-1).value

    create = ctypes.windll.kernel32.CreateFileW
    create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                       wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create.restype = wintypes.HANDLE
    handle = create(str(path), GENERIC_READ, SHARE_ALL, None,
                    OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, None)
    if not handle or handle == INVALID_HANDLE:
        err = ctypes.get_last_error() or ctypes.windll.kernel32.GetLastError()
        raise FileNotFoundError(err, f"CreateFileW 失敗(WinError {err})", str(path))
    fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    return os.fdopen(fd, "rb")


# --------------------------------------------------------------------------- #
# parquet の完結判定(書きかけを掴まないための唯一の機械判定)
# --------------------------------------------------------------------------- #
def is_complete_parquet(path: Path) -> bool:
    """先頭 magic・末尾 magic・footer 長の整合だけを見る軽い判定(pyarrow 不要)。

    parquet は footer をファイル末尾に書いて閉じる形式なので、**書きかけ = footer 不在**が
    機械検出できる(`scripts/live_viewer.py` と同じ判定式)。
    """
    try:
        size = path.stat().st_size
        if size < 12:
            return False
        with _open_shared(path) as f:
            if f.read(4) != PARQUET_MAGIC:
                return False
            f.seek(-8, os.SEEK_END)
            tail = f.read(8)
    except OSError:
        return False
    if len(tail) != 8 or tail[4:] != PARQUET_MAGIC:
        return False
    footer_len = int.from_bytes(tail[:4], "little")
    return 0 < footer_len <= size - 12


def _pyarrow_footer_ok(path: Path) -> bool | None:
    """pyarrow で footer を**実パース**する(1 行も読まない)。pyarrow 不在なら None。"""
    try:
        import pyarrow.parquet as pq
    except Exception:
        return None
    try:
        with _open_shared(path) as fh:
            md = pq.ParquetFile(fh).metadata
        return md is not None
    except Exception:
        return False


def parquet_ok(path: Path, mode: str = "auto") -> bool:
    """完結判定。mode: auto(magic→可能なら pyarrow) / magic / pyarrow。"""
    if not is_complete_parquet(path):
        return False
    if mode == "magic":
        return True
    deep = _pyarrow_footer_ok(path)
    if deep is None:                              # pyarrow 不在
        return mode != "pyarrow"                  # pyarrow 明示指定なら「検証できない=不採用」
    return bool(deep)


# --------------------------------------------------------------------------- #
# 対象の選別
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Item:
    rel: str                                      # run-dir からの相対パス(posix)
    path: Path
    category: str                                 # part|canonical|checkpoint|meta
    size: int
    mtime_ns: int


@dataclass(frozen=True)
class Skip:
    rel: str
    reason: str


def _is_part(name: str) -> bool:
    return ".part-" in name and name.endswith(".parquet")


def _excluded(name: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(name, pat) for pat in patterns)


def _ckpt_step(name: str) -> int | None:
    for pre in CKPT_PREFIXES:
        if name.startswith(pre) and name.endswith(CKPT_SUFFIX):
            try:
                return int(name[len(pre):-len(CKPT_SUFFIX)])
            except ValueError:
                return None
    return None


def checkpoint_boundary(run_dir: Path) -> tuple[int | None, float | None]:
    """最新 checkpoint の (step, mtime)。無ければ (None, None)。"""
    d = run_dir / "checkpoint"
    if not d.is_dir():
        return (None, None)
    best: tuple[int, float] | None = None
    for p in d.iterdir():
        if not p.is_file() or not p.name.startswith("ckpt-"):
            continue
        step = _ckpt_step(p.name)
        if step is None:
            continue
        try:
            mt = p.stat().st_mtime
        except OSError:
            continue
        if best is None or step > best[0]:
            best = (step, mt)
    return best if best else (None, None)


def select_payload(run_dir: Path, *, ckpt_generations: int = 2,
                   min_age_sec: float = 300.0, now: float | None = None,
                   require_boundary: bool = True,
                   exclude: tuple[str, ...] = (),
                   footer_mode: str = "auto") -> tuple[list[Item], list[Skip]]:
    """バックアップして良い「確定分」を返す。第 2 返値はスキップとその理由。

    - `*.part-*.parquet` … footer 完結 + checkpoint 境界の内側(完走ランは境界不問)
    - その他 `*.parquet`(canonical) … footer 完結(finalize は tmp→replace の原子的書き)
    - `checkpoint/` … **直近 `ckpt_generations` 世代**の `ckpt-` と `dormant-`(pool サイドカー)。
      どちらも tmp→os.replace の原子的書きなので、最終名で存在すれば完結している。
    - それ以外の直下ファイル(summary.json / run_manifest.json / config.yaml / サイドカー /
      llm_journal.jsonl.gz / llm_cache.jsonl …) … そのまま同梱。追記中でも「その時点までの
      前半」は有効(journal の確定点は checkpoint 側に記録されている)。
    - `analysis/` `panel/` などの派生ディレクトリと `DEFAULT_EXCLUDE` は対象外。
    """
    now = time.time() if now is None else now
    pats = tuple(DEFAULT_EXCLUDE) + tuple(exclude)
    items: list[Item] = []
    skips: list[Skip] = []
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run-dir が無い: {run_dir}")

    run_complete = (run_dir / "summary.json").exists()
    _, ckpt_mtime = checkpoint_boundary(run_dir)

    def _stat(p: Path):
        try:
            st = p.stat()
        except OSError:
            return None
        return st

    for p in sorted(run_dir.iterdir(), key=lambda q: q.name):
        rel = p.name
        if p.is_dir():
            if p.name not in TRAVERSE_DIRS:
                skips.append(Skip(rel + "/", "derived-dir(派生物なので対象外)"))
            continue
        if not p.is_file():
            continue
        if _excluded(p.name, pats):
            skips.append(Skip(rel, "excluded(既定除外 or --exclude)"))
            continue
        st = _stat(p)
        if st is None:
            skips.append(Skip(rel, "vanished(選別中に消えた)"))
            continue
        if p.name.endswith(".parquet"):
            if not parquet_ok(p, footer_mode):
                skips.append(Skip(rel, "parquet-footer-incomplete(書き込み中とみなす)"))
                continue
            if _is_part(p.name):
                if require_boundary and not run_complete:
                    if ckpt_mtime is None:
                        if (now - st.st_mtime) < min_age_sec:
                            skips.append(Skip(rel, "no-checkpoint-yet(静止待ち)"))
                            continue
                    elif st.st_mtime > ckpt_mtime:
                        skips.append(Skip(rel, "after-checkpoint-boundary(未確定)"))
                        continue
                items.append(Item(rel, p, "part", st.st_size, st.st_mtime_ns))
            else:
                items.append(Item(rel, p, "canonical", st.st_size, st.st_mtime_ns))
            continue
        items.append(Item(rel, p, "meta", st.st_size, st.st_mtime_ns))

    # ---- checkpoint(直近世代のみ)----
    ck_dir = run_dir / "checkpoint"
    if ck_dir.is_dir() and ckpt_generations > 0:
        by_step: dict[int, list[Path]] = {}
        for p in sorted(ck_dir.iterdir(), key=lambda q: q.name):
            if p.is_dir():
                skips.append(Skip(f"checkpoint/{p.name}/", "checkpoint-subdir(corrupt 隔離等)"))
                continue
            if _excluded(p.name, pats):
                skips.append(Skip(f"checkpoint/{p.name}", "excluded(既定除外 or --exclude)"))
                continue
            step = _ckpt_step(p.name)
            if step is None:
                skips.append(Skip(f"checkpoint/{p.name}", "not-a-checkpoint(命名が想定外)"))
                continue
            by_step.setdefault(step, []).append(p)
        keep = sorted(by_step)[-ckpt_generations:]
        for step in sorted(by_step):
            for p in by_step[step]:
                rel = f"checkpoint/{p.name}"
                if step not in keep:
                    skips.append(Skip(rel, "older-generation(直近世代のみ転送)"))
                    continue
                st = _stat(p)
                if st is None:
                    skips.append(Skip(rel, "vanished(選別中に消えた)"))
                    continue
                items.append(Item(rel, p, "checkpoint", st.st_size, st.st_mtime_ns))
    return items, skips


# --------------------------------------------------------------------------- #
# ハッシュ・コピー(1 パスで読み・書き・ハッシュを同時に済ませる)
# --------------------------------------------------------------------------- #
class _HashingWriter:
    """書いたバイト列の sha256 を同時に取る薄いラッパ(tar 自体のハッシュを 0 パスで)。

    `io.RawIOBase` を継承しないのは、GC の `__del__` が閉じ済みの下位ハンドルへ
    flush して "Exception ignored" を撒くのを避けるため。tarfile の stream モードは
    `write` しか使わない(fileobj を渡した場合 close もしない)。
    """

    def __init__(self, fh):
        self._fh = fh
        self._h = hashlib.sha256()
        self._n = 0

    def write(self, b) -> int:
        self._h.update(b)
        self._n += len(b)
        return self._fh.write(b)

    def tell(self) -> int:
        return self._n

    def flush(self) -> None:
        self._fh.flush()

    def close(self) -> None:                      # ラップ元は呼び出し側が閉じる
        pass

    @property
    def hexdigest(self) -> str:
        return self._h.hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with _open_shared(path) as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _sha256_stream(fh, size: int | None = None) -> tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    while True:
        want = CHUNK if size is None else min(CHUNK, size - n)
        if want <= 0:
            break
        b = fh.read(want)
        if not b:
            break
        h.update(b)
        n += len(b)
    return h.hexdigest(), n


def _copy_exact(src: Path, dst: Path, size: int) -> str:
    """src の先頭 size バイトを dst へ(tmp→replace)。書いたバイトの sha256 を返す。

    「先頭 size バイトだけ」なのは追記中ファイル(llm_cache.jsonl 等)への配慮:
    選別時の size で切ると **一貫した前半**が取れ、マニフェストのハッシュも
    「実際に置いたバイト列」と一致する(伸びた分は mtime 変化で次回まるごと再取得)。
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    h = hashlib.sha256()
    n = 0
    with _open_shared(src) as fin, open(tmp, "wb") as fout:
        while n < size:
            b = fin.read(min(CHUNK, size - n))
            if not b:
                break
            fout.write(b)
            h.update(b)
            n += len(b)
        fout.flush()
        os.fsync(fout.fileno())
    if n != size:
        tmp.unlink(missing_ok=True)
        raise IOError(f"短く読めた({n}/{size} bytes): {src}")
    os.replace(tmp, dst)
    return h.hexdigest()


def _add_to_tar(tar: tarfile.TarFile, arcname: str, src: Path,
                size: int, mtime_ns: int) -> str:
    """tar へ 1 メンバ追加(store)。書いたバイトの sha256 を返す。"""
    info = tarfile.TarInfo(arcname)
    info.size = size
    info.mtime = int(mtime_ns // 1_000_000_000)
    info.mode = 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    h = hashlib.sha256()

    class _Tee:                                   # tarfile は read() しか使わない
        def __init__(self, fh):
            self._fh = fh
            self.n = 0

        def read(self, n=-1):
            b = self._fh.read(CHUNK if n is None or n < 0 else n)
            h.update(b)
            self.n += len(b)
            return b

    with _open_shared(src) as fin:
        tee = _Tee(fin)
        tar.addfile(info, tee)
        if tee.n != size:
            raise IOError(f"短く読めた({tee.n}/{size} bytes): {src}")
    return h.hexdigest()


def _add_bytes_to_tar(tar: tarfile.TarFile, arcname: str, blob: bytes) -> None:
    info = tarfile.TarInfo(arcname)
    info.size = len(blob)
    info.mtime = int(time.time())
    info.mode = 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    tar.addfile(info, io.BytesIO(blob))


# --------------------------------------------------------------------------- #
# バッグ(dest 側)の読み書き
# --------------------------------------------------------------------------- #
def _empty_state(run_id: str, source: Path) -> dict:
    return {"schema": SCHEMA, "run_id": run_id, "source": str(source),
            "created": _now_iso(), "updated": _now_iso(),
            "increments": [], "files": {}}


def load_state(bag: Path) -> dict:
    p = bag / STATE_NAME
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"{p} が壊れている(手で直すか退避してから再実行): {e}")
    if int(data.get("schema", 0)) != SCHEMA:
        raise ValueError(f"{p}: 未知の schema={data.get('schema')}")
    data.setdefault("files", {})
    data.setdefault("increments", [])
    return data


def _write_atomic(path: Path, blob: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(blob)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def manifest_text(entries: dict[str, dict]) -> str:
    """BagIt 式 manifest-sha256.txt("<sha256>  data/<相対パス>" を路順に)。"""
    lines = [f"{entries[rel]['sha256']}  {PAYLOAD_DIR}/{rel}" for rel in sorted(entries)]
    return "\n".join(lines) + ("\n" if lines else "")


def parse_manifest(text: str) -> dict[str, str]:
    """manifest-sha256.txt → {相対パス(data/ を除いた): sha256}。"""
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        digest, rel = parts[0], parts[1].strip()
        if rel.startswith(PAYLOAD_DIR + "/"):
            rel = rel[len(PAYLOAD_DIR) + 1:]
        out[rel] = digest
    return out


def bagit_text() -> str:
    return "BagIt-Version: 1.0\nTag-File-Character-Encoding: UTF-8\n"


def baginfo_text(run_id: str, source: Path, entries: dict[str, dict],
                 note: str = "") -> str:
    total = sum(int(e["size"]) for e in entries.values())
    lines = [
        f"Bagging-Date: {datetime.now().strftime('%Y-%m-%d')}",
        "Bag-Software-Agent: scripts/backup_run.py (shibuya-simulation)",
        f"External-Identifier: {run_id}",
        f"Source-Directory: {source}",
        f"Payload-Oxum: {total}.{len(entries)}",
    ]
    if note:
        lines.append(f"Internal-Sender-Description: {note}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# バックアップ本体
# --------------------------------------------------------------------------- #
def backup(run_dir: Path, dest: Path, *, run_id: str | None = None,
           layout: str = "tar", ckpt_generations: int = 2,
           min_age_sec: float = 300.0, require_boundary: bool = True,
           exclude: tuple[str, ...] = (), footer_mode: str = "auto",
           dry_run: bool = False, verify_written: bool = False,
           quiet: bool = False) -> dict:
    """確定分の増分を dest へ。**dest からは何も消さない**。差分ゼロなら何も書かない。"""
    t0 = time.monotonic()
    run_dir = Path(run_dir)
    rid = run_id or run_dir.name
    bag = Path(dest) / rid
    state = load_state(bag) or _empty_state(rid, run_dir)

    items, skips = select_payload(
        run_dir, ckpt_generations=ckpt_generations, min_age_sec=min_age_sec,
        require_boundary=require_boundary, exclude=exclude, footer_mode=footer_mode)

    known: dict[str, dict] = state["files"]
    todo: list[Item] = []
    for it in items:
        prev = known.get(it.rel)
        if prev and int(prev["size"]) == it.size and int(prev["mtime_ns"]) == it.mtime_ns:
            continue                                   # 不変 = 読みもしない(増分の要)
        todo.append(it)

    selected = {it.rel for it in items}
    missing_at_source = [rel for rel in sorted(known)
                         if rel not in selected and not (run_dir / rel).exists()]

    report = {
        "run_id": rid, "source": str(run_dir), "dest": str(bag), "layout": layout,
        "selected": len(items), "transferred": 0, "bytes": 0,
        "increment": None, "zero_diff": not todo, "dry_run": bool(dry_run),
        "skipped": [{"path": s.rel, "reason": s.reason} for s in skips],
        "changed_existing": [], "missing_at_source": missing_at_source,
        "errors": [], "elapsed_sec": 0.0,
    }

    def _say(msg: str) -> None:
        if not quiet:
            print(f"[backup_run] {msg}", flush=True)

    for s in skips:                                    # 正直に全部出す(黙って落とさない)
        _say(f"skip {s.rel}: {s.reason}")
    for rel in missing_at_source:
        _say(f"NOTE 元から消えている(バックアップには残す=削除を伝播しない): {rel}")

    if not todo:
        report["elapsed_sec"] = round(time.monotonic() - t0, 3)
        _say(f"差分ゼロ({len(items)} 件は転送済み)。dest には何も書かない。")
        return report
    if dry_run:
        report["transferred"] = len(todo)
        report["bytes"] = sum(it.size for it in todo)
        report["elapsed_sec"] = round(time.monotonic() - t0, 3)
        for it in todo:
            _say(f"would transfer [{it.category}] {it.rel} ({it.size} B)")
        return report

    inc_id = len(state["increments"]) + 1
    inc_name = f"inc-{inc_id:04d}-{_ts_tag()}.tar"
    new_entries: dict[str, dict] = {}
    written_bytes = 0

    if layout in ("tar", "both"):
        inc_dir = bag / INCREMENTS_DIR
        inc_dir.mkdir(parents=True, exist_ok=True)
        tar_path = inc_dir / inc_name
        tmp_path = inc_dir / (inc_name + ".tmp")
        try:
            with open(tmp_path, "wb") as raw:
                hw = _HashingWriter(raw)
                # store モード(再圧縮しない)・PAX(長い名前と UTF-8 に強い)
                with tarfile.open(fileobj=hw, mode="w|",
                                  format=tarfile.PAX_FORMAT) as tar:
                    for it in todo:
                        digest = _add_to_tar(tar, f"{PAYLOAD_DIR}/{it.rel}",
                                             it.path, it.size, it.mtime_ns)
                        new_entries[it.rel] = {
                            "size": it.size, "mtime_ns": it.mtime_ns,
                            "sha256": digest, "category": it.category,
                            "increment": inc_id, "backed_up": _now_iso(),
                        }
                        written_bytes += it.size
                    # マニフェストは全メンバのハッシュが出そろってから末尾に置く
                    _add_bytes_to_tar(tar, MANIFEST_NAME,
                                      manifest_text(new_entries).encode("utf-8"))
                    _add_bytes_to_tar(tar, BAGIT_NAME, bagit_text().encode("utf-8"))
                    _add_bytes_to_tar(
                        tar, BAGINFO_NAME,
                        baginfo_text(rid, run_dir, new_entries,
                                     note=f"increment {inc_id}").encode("utf-8"))
                raw.flush()
                os.fsync(raw.fileno())
                tar_sha = hw.hexdigest
            os.replace(tmp_path, tar_path)
        except Exception as e:
            Path(tmp_path).unlink(missing_ok=True)
            report["errors"].append(f"increment tar 作成に失敗: {e}")
            report["elapsed_sec"] = round(time.monotonic() - t0, 3)
            _say(f"ERROR increment tar 作成に失敗: {e}")
            return report
        _write_atomic(inc_dir / (inc_name + ".sha256"),
                      f"{tar_sha}  {inc_name}\n".encode("utf-8"))
        if verify_written:
            bad = _verify_tar(tar_path, new_entries)
            if bad:
                # 壊れた増分を「転送済み」として state に刻むと**次回に再送されなくなる**。
                # tar は証拠として残し、state とマニフェストは更新せずに引き返す。
                report["errors"].extend(bad)
                report["elapsed_sec"] = round(time.monotonic() - t0, 3)
                _say(f"ERROR 書いた増分の照合に失敗({len(bad)} 件)。state は更新しない"
                     f"(次回に再送される): {tar_path.name}")
                return report
        state["increments"].append({
            "id": inc_id, "name": inc_name, "created": _now_iso(),
            "files": len(new_entries), "bytes": written_bytes, "sha256": tar_sha,
        })

    if layout in ("tree", "both"):
        for it in todo:
            dst = bag / PAYLOAD_DIR / it.rel
            try:
                digest = _copy_exact(it.path, dst, it.size)
            except Exception as e:
                report["errors"].append(f"copy 失敗 {it.rel}: {e}")
                _say(f"ERROR copy 失敗 {it.rel}: {e}")
                continue
            ent = new_entries.get(it.rel)
            if ent and ent["sha256"] != digest:
                report["errors"].append(
                    f"tar と tree で sha256 不一致 {it.rel}(コピー中に変化した疑い)")
            if ent is None:
                new_entries[it.rel] = {
                    "size": it.size, "mtime_ns": it.mtime_ns, "sha256": digest,
                    "category": it.category, "increment": None,
                    "backed_up": _now_iso(),
                }
                written_bytes += it.size

    for rel, ent in new_entries.items():
        prev = known.get(rel)
        if prev and prev.get("sha256") != ent["sha256"]:
            report["changed_existing"].append(rel)
            _say(f"WARN 既にバックアップ済みのパスが**中身ごと**変わった: {rel} "
                 "(watchdog の巻き戻し等を疑う。旧版は古い増分に残っている)")
        known[rel] = ent

    state["updated"] = _now_iso()
    _write_atomic(bag / MANIFEST_NAME, manifest_text(known).encode("utf-8"))
    _write_atomic(bag / BAGIT_NAME, bagit_text().encode("utf-8"))
    _write_atomic(bag / BAGINFO_NAME, baginfo_text(rid, run_dir, known).encode("utf-8"))
    _write_atomic(bag / STATE_NAME,
                  json.dumps(state, ensure_ascii=False, indent=2).encode("utf-8"))

    report["transferred"] = len(new_entries)
    report["bytes"] = written_bytes
    report["increment"] = inc_name if layout in ("tar", "both") else None
    report["elapsed_sec"] = round(time.monotonic() - t0, 3)
    _say(f"転送 {len(new_entries)} 件 / {written_bytes} B → {bag}"
         + (f" [{inc_name}]" if report["increment"] else ""))
    return report


# --------------------------------------------------------------------------- #
# 照合(--verify)
# --------------------------------------------------------------------------- #
def _verify_tar(tar_path: Path, want: dict[str, dict]) -> list[str]:
    """tar 内 `data/<rel>` のハッシュを want と突き合わせる。問題の一覧を返す。"""
    problems: list[str] = []
    seen: set[str] = set()
    try:
        with tarfile.open(tar_path, mode="r|") as tar:      # 逐次読み(全展開しない)
            for info in tar:
                if not info.isfile() or not info.name.startswith(PAYLOAD_DIR + "/"):
                    continue
                rel = info.name[len(PAYLOAD_DIR) + 1:]
                if rel not in want:
                    continue
                fh = tar.extractfile(info)
                if fh is None:
                    problems.append(f"{tar_path.name}: メンバを読めない {rel}")
                    continue
                digest, n = _sha256_stream(fh)
                seen.add(rel)
                if digest != want[rel]["sha256"]:
                    problems.append(f"{tar_path.name}: sha256 不一致 {rel}")
                elif n != int(want[rel]["size"]):
                    problems.append(f"{tar_path.name}: サイズ不一致 {rel}")
    except Exception as e:
        problems.append(f"{tar_path.name}: tar を読めない: {e}")
        return problems
    for rel in want:
        if rel not in seen:
            problems.append(f"{tar_path.name}: メンバが無い {rel}")
    return problems


def verify(dest: Path, run_id: str, *, quiet: bool = False) -> dict:
    """バックアップ側だけで完結する照合(restore drill の前段)。

    累積マニフェスト × backup_state.json を突き合わせ、実体(tree の data/ か
    増分 tar のメンバ)を **実際に読んで** sha256 を再計算する。
    """
    t0 = time.monotonic()
    bag = Path(dest) / run_id
    out = {"run_id": run_id, "bag": str(bag), "checked": 0, "ok": 0,
           "problems": [], "elapsed_sec": 0.0}

    def _say(msg: str) -> None:
        if not quiet:
            print(f"[backup_run:verify] {msg}", flush=True)

    if not bag.is_dir():
        out["problems"].append(f"バッグが無い: {bag}")
        return out
    try:
        state = load_state(bag)
    except ValueError as e:
        out["problems"].append(str(e))
        return out
    if not state:
        out["problems"].append(f"{STATE_NAME} が無い(このバッグは本スクリプト製ではない)")
        return out

    man_path = bag / MANIFEST_NAME
    if not man_path.exists():
        out["problems"].append(f"{MANIFEST_NAME} が無い")
        return out
    manifest = parse_manifest(man_path.read_text(encoding="utf-8"))
    files: dict[str, dict] = state["files"]
    for rel, digest in manifest.items():
        if rel not in files:
            out["problems"].append(f"マニフェストにあるが state に無い: {rel}")
        elif files[rel]["sha256"] != digest:
            out["problems"].append(f"マニフェストと state で sha256 が食い違う: {rel}")
    for rel in files:
        if rel not in manifest:
            out["problems"].append(f"state にあるがマニフェストに無い: {rel}")

    # 増分 tar 自体のハッシュ(転送中の破損はここで出る)
    for inc in state["increments"]:
        p = bag / INCREMENTS_DIR / inc["name"]
        if not p.exists():
            out["problems"].append(f"増分 tar が無い: {inc['name']}")
            continue
        if inc.get("sha256") and sha256_file(p) != inc["sha256"]:
            out["problems"].append(f"増分 tar の sha256 不一致: {inc['name']}")

    # 実体の照合(tree 優先。無ければ所属増分の tar を読む)
    by_inc: dict[int, dict[str, dict]] = {}
    for rel, ent in files.items():
        tree = bag / PAYLOAD_DIR / rel
        if tree.exists():
            out["checked"] += 1
            if sha256_file(tree) == ent["sha256"]:
                out["ok"] += 1
            else:
                out["problems"].append(f"data/ の sha256 不一致: {rel}")
            continue
        inc_id = ent.get("increment")
        if inc_id is None:
            out["problems"].append(f"実体が見つからない(tree にも増分にも): {rel}")
            continue
        by_inc.setdefault(int(inc_id), {})[rel] = ent
    for inc_id, want in sorted(by_inc.items()):
        rec = next((i for i in state["increments"] if int(i["id"]) == inc_id), None)
        if rec is None:
            out["problems"].append(f"増分 {inc_id} の記録が無い")
            continue
        p = bag / INCREMENTS_DIR / rec["name"]
        out["checked"] += len(want)
        problems = _verify_tar(p, want) if p.exists() else [f"増分 tar が無い: {rec['name']}"]
        out["ok"] += len(want) - len(problems)
        out["problems"].extend(problems)

    out["elapsed_sec"] = round(time.monotonic() - t0, 3)
    for pb in out["problems"]:
        _say(f"NG {pb}")
    _say(f"照合 {out['ok']}/{out['checked']} OK・問題 {len(out['problems'])} 件")
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="backup_run",
        description="ラン成果物の増分バックアップ(確定分のみ・BagIt 式マニフェスト)")
    p.add_argument("--run-dir", default=None,
                   help="バックアップ元の run ディレクトリ(--verify 時は不要)")
    p.add_argument("--dest", required=True,
                   help="退避先のルート(ローカルパス/外付け/UNC。<dest>/<run_id>/ を作る)")
    p.add_argument("--run-id", default=None,
                   help="バッグ名(既定: run-dir の名前)")
    p.add_argument("--layout", choices=("tar", "tree", "both"), default="tar",
                   help="tar=増分tarのみ(既定・転送向き) / tree=展開コピー(解析をそのまま"
                        "掛けられる) / both=両方")
    p.add_argument("--ckpt-generations", type=int, default=2,
                   help="転送する checkpoint 世代数(既定2。ckpt- と dormant- を同 step で対に)")
    p.add_argument("--min-age-sec", type=float, default=300.0,
                   help="checkpoint がまだ 1 つも無いときに part を確定とみなす静止秒(既定300)")
    p.add_argument("--no-checkpoint-boundary", action="store_true",
                   help="checkpoint 境界の条件を外す(footer 検証のみ)。巻き戻しで part が"
                        "書き換わる可能性を承知の上で使うこと")
    p.add_argument("--exclude", action="append", default=[],
                   help="除外する glob(複数可)。既定除外: " + " ".join(DEFAULT_EXCLUDE))
    p.add_argument("--footer-mode", choices=("auto", "magic", "pyarrow"), default="auto",
                   help="parquet 完結判定の厚み(既定 auto = magic + 可能なら pyarrow)")
    p.add_argument("--dry-run", action="store_true", help="何も書かずに転送予定を出す")
    p.add_argument("--verify-written", action="store_true",
                   help="書いた増分 tar を読み直して照合(実転送後の --verify が本命)")
    p.add_argument("--verify", action="store_true",
                   help="照合モード(既存バックアップの sha256 を全件再計算)")
    p.add_argument("--json", action="store_true", help="レポートを JSON で標準出力へ")
    p.add_argument("--quiet", action="store_true", help="人間向けの行を出さない")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    ns = parse_args(list(sys.argv[1:] if argv is None else argv))
    if ns.verify:
        rid = ns.run_id or (Path(ns.run_dir).name if ns.run_dir else None)
        if not rid:
            print("--verify には --run-id か --run-dir が要る", file=sys.stderr)
            return 2
        rep = verify(Path(ns.dest), rid, quiet=ns.quiet or ns.json)
        if ns.json:
            print(json.dumps(rep, ensure_ascii=False, indent=2))
        return 1 if rep["problems"] else 0

    if not ns.run_dir:
        print("--run-dir が要る(--verify 以外)", file=sys.stderr)
        return 2
    try:
        rep = backup(Path(ns.run_dir), Path(ns.dest), run_id=ns.run_id,
                     layout=ns.layout, ckpt_generations=ns.ckpt_generations,
                     min_age_sec=ns.min_age_sec,
                     require_boundary=not ns.no_checkpoint_boundary,
                     exclude=tuple(ns.exclude), footer_mode=ns.footer_mode,
                     dry_run=ns.dry_run, verify_written=ns.verify_written,
                     quiet=ns.quiet or ns.json)
    except (FileNotFoundError, ValueError) as e:   # run-dir 不在・state 破損など
        print(f"[backup_run] ERROR {e}", file=sys.stderr)
        return 2
    if ns.json:
        print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 1 if rep["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
