#!/usr/bin/env python
"""追いかけ再生(chase playback)= ラン実行中の part parquet を**読むだけ**で動くライブ風画面。

設計正典: docs/research/dt-integration-deep.md §7.3(P6)/ docs/plans/dt-integration-plan.md §3。

    python scripts/live_viewer.py runs/prod1 [--interval 60]

何をするか
----------
実行中のラン out_dir を定期的に覗き、`ObserverLogger.flush_segment()` が書いた
`l1_events.part-NNNN.parquet` / `l2_metrics.part-NNNN.parquet` を**増分**で読み、
「いま何日目の何時を走っているか」を地図・時計・イベントティッカー・L2 スパークラインに
描いた自己完結 HTML(`<run-dir>/_live/live.html`)を生成し続ける。ブラウザは開きっぱなしで
よく、データだけが数分遅れで更新される。

**シミュレーション本体には read も write も一切しない。**
  - 読むのは L1/L2 の part(+ config.yaml / agents.json / summary.json)だけ。
  - 書くのは `<run-dir>/_live/`(`--out-dir` で run dir の外へも出せる)だけ。
    ファイル名は `live.html` / `live_data.js` で、シム側が glob する `*.part-*.parquet` にも
    `checkpoint/` にも一切かからない(サブディレクトリなので非再帰 glob にも入らない)。
  - このプロセスが落ちても・途中から参加しても・二重起動しても、ランは何も変わらない。
  - ★**「読むだけ」でもシムを壊しうる箇所が 1 つだけあった**(実測で踏んだ):Windows の
    `open()` は `FILE_SHARE_DELETE` を立てないため、こちらが part を開いている最中に
    `logger._finalize_stream` の `p.unlink()` が `PermissionError [WinError 32]` で失敗し、
    **ラン本体が finalize で落ちる**。part を読むときは必ず `_open_shared()` を使う。

part の完結判定(唯一の規律。書きかけを読まないこと)
--------------------------------------------------
parquet は `PAR1 …データ… フッタ フッタ長(int32,LE) PAR1` の順で書かれ、**末尾 magic は
最後に書かれる**。`flush_segment` は `pq.write_table` = open→write→close の一括なので、

    「先頭 4B が PAR1、末尾 4B が PAR1、末尾手前の footer 長が整合」ならその part は書き終わっている
    (以後サイズが変わることはない。次の flush は part N+1 という別ファイルへ行く)

が成立する。これを `is_complete_parquet()` として単一の判定にした。deep §7.3 が提案した
「次の part が現れたら直前を確定」規則はこれより弱い(1 flush 分ぶん余計に遅れる)ので、
**フッタ判定を主・index 規則を従**にしている。従の使い道は 1 つだけ:「より新しい part が
既に存在するのに N が完結していない」= クラッシュで書きかけのまま残った屍と判定して、
警告つきで**恒久スキップ**し先へ進む(そこで止まると以後ずっと追いかけられなくなるため)。

L1 の読み方(10日ラン=1 part 数百万行でも重くならないための実装上の判断)
-----------------------------------------------------------------------
- **payload 列を読まない**。位置の状態遷移(屋内/屋外/就寝)は kind だけで決まるので、
  L1 で最も重い payload 列は「ティッカーに出す見せ場イベント(全体の数%)」に限って読む。
- 位置は `viz/make_viewer.py::build_data` と**同じイベント意味論**(move_segment/arrive/
  speak/reflect で x,y 更新、enter/exit_building で屋内/路上、exit/enter_area、sleep/wake)。
  ただし w は make_viewer の `1000+bIdx*100+floor` を使わず 4 値の状態コードへ畳む
  (0=路上 1=屋内 -1=範囲外 -2=就寝)。ライブ画面は点を打つだけでフロアビューを持たない=
  建物 index も階も要らず、payload を読まないで済むという上の判断とセットの帰結。

画面の更新方式(判断の理由は本ファイル末尾 `_HTML` の注記と報告を参照)
--------------------------------------------------------------------
静的 HTML を**1回だけ**書き、データは `live_data.js`(JSONP: `LIVE_DATA({...})`)として
毎回上書きする。ブラウザ側は `<script src>` を動的に挿し直して取り込む(`fetch` は file:// で
CORS 不可・`script src` は可、という第76バッチの知見をそのまま踏襲)。全 HTML 再生成+
meta refresh だと 10日間で 1万回超のフルリロードになり、地図の pan/zoom も毎回失われる。

正直な限界
----------
- part が出るのは `observer.checkpoint_every > 0` か `observer.flush_every_steps > 0` の
  ランだけ(`Simulation.run`)。どちらも 0 のランは追いかけられない(画面に明示して出す)。
- 追いかけ位置は常に「最後に完結した part の末尾 step」= 遅延は flush 間隔が下限。
- finalize で part は削除され canonical へ結合される。ラン終了後は L1 canonical(巨大)を
  読み直さず、L2 canonical の最終行だけを読んで「完了」表示に切り替える(通常ビューアへ誘導)。
  そのため**最後の flush 以降のぶんは地図に出ない**(その事実を画面の注記に出す)。
- crash → checkpoint から resume したランでは、`ObserverLogger._next_seg` が既存 part の
  最大 index+1 から採番するため、**既に読んだ step を含む part が新しい index で現れる**。
  追いかけ位置(step)は max で単調にしているので巻き戻らないが、位置は数 poll のあいだ
  古い step の再生になる(ランが追いつけば自然に解消する)。
- L2 の系列は finalize 後に canonical で完結させるので、地図の追いかけ位置より先の step に
  なることがある。どの step の値かは画面に出す(黙って混ぜない)。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

for _s in (sys.stdout, sys.stderr):              # Windows コンソール(cp932)対策
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_HERE_DIR = str(Path(__file__).resolve().parent)
if _HERE_DIR not in sys.path:    # 同ディレクトリの run_dt を import
    sys.path.insert(0, _HERE_DIR)

import run_dt                    # noqa: E402  (W2-3: ランの Δt の単一の源)

# W2-3: **ラン依存**(run.dt_min)。既定は正準 Δt=10(society.engine.clock と同一)で、
# 実際の値は read_run_config が run dir から読み ChaseState が self.step_minutes に持つ。
STEP_MINUTES = run_dt.CANON_DT_MIN
DEFAULT_START_MIN = 7 * 60       # sim_min 列も config も無いときの最終退避(make_viewer と同値)
PARQUET_MAGIC = b"PAR1"
DEFAULT_INTERVAL = 45.0
DEFAULT_OUT_SUBDIR = "_live"

# 位置の状態コード(make_viewer の w を畳んだもの)
W_ROAD, W_INDOOR, W_OUTSIDE, W_SLEEP = 0, 1, -1, -2

# 位置に効くイベント(payload 不要)。make_viewer.build_data の分岐と同じ意味論。
# floor_move は「屋内で階が変わる」だけ=平面の点は動かないので意図的に含めない。
POS_KINDS = {
    "move_segment": "xy", "arrive": "xy", "speak": "xy", "reflect": "xy",
    "enter_building": "in", "exit_building": "out",
    "exit_area": "gone", "enter_area": "back",
    "sleep_start": "sleep", "wake_up": "wake",
}

# ティッカーに出す「見せ場」イベント。payload を読むのはこの集合だけ。
HIGHLIGHT_KINDS = {
    "vocab_coin", "label_coin", "label_adopt", "place_label_bind",
    "joint_invite", "joint_activity",
    "undefined_action", "free_action",
    "belief_update", "belief_transmit", "belief_verify",
    "institution", "institution_rule", "proposal", "proposal_passed",
    "group_found", "group_join",
    "venture_open", "venture_close", "event_host",
    "flyer_post", "world_event", "scenario_shock",
    "relation_tier", "relation_break", "relation_dormant", "relation_rekindle",
    "labor_action", "candidacy", "election_result", "ordinance_vote",
    "move_home", "long_goal", "chance_event", "fallback",
}

# ティッカー本文に使う payload キーの優先順(最初に見つかったものを短く出す)
_TEXT_KEYS = ("text", "word", "title", "name", "what", "goal", "action",
              "norm_text", "label", "type", "kind", "verdict", "tier",
              "claim", "topic", "place", "reason", "demand", "output")

# スパークラインに出す L2 列の候補(存在するものを先頭から --series-max 本)
SERIES_PREF = [
    ("llm_fallback_rate", "LLM fallback率", "rate"),
    ("joint_accept_rate", "共同行動 承諾率", "rate"),
    ("echo_utterance_rate", "エコー(反復)率", "rate"),
    ("belief_distance_mean", "信念距離 平均", "num"),
    ("undefined_action_rate", "未定義行動率", "rate"),
    ("distinct_vocab_in_use", "使用中の語彙", "int"),
    ("total_adoptions", "語の採用 累計", "int"),
    ("n_moving", "移動中", "int"),
    ("n_inside_buildings", "屋内", "int"),
    ("n_sleeping", "就寝", "int"),
    ("opinion_var", "意見の分散", "num"),
    ("mean_money", "所持金 平均", "num"),
]


# ====================================================================== 非侵襲な読み口
def _open_shared(path: Path):
    """**読んでいる最中でも相手が消せる**ハンドルで開く(Windows 必須。読み取り専用の要)。

    なぜ必要か(実測で踏んだ事故):
      Python の `open()` は Windows で `FILE_SHARE_DELETE` を立てない。したがって live_viewer が
      part を開いている間に `logger._finalize_stream` の `p.unlink()` が
      `PermissionError [WinError 32]` を食らい、**シミュレーション本体が finalize で落ちる**。
      「観測はシムに一切影響しない」というドクトリンが、素直に読むだけで破れる。
      → CreateFileW で SHARE_READ|SHARE_WRITE|**SHARE_DELETE** を立てて開く。シム側の unlink は
        成功し(削除保留)、こちらのハンドルは閉じるまで有効に読み続けられる。
    POSIX は open 済みファイルの unlink が元から成功するので、素の open で同じ意味になる。
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


# ====================================================================== part 完結判定
def part_index(path: Path) -> int:
    """`<stem>.part-NNNN.parquet` の NNNN。読めなければ -1。"""
    name = path.name
    i = name.find(".part-")
    if i < 0:
        return -1
    try:
        return int(name[i + 6:].split(".")[0])
    except ValueError:
        return -1


def list_parts(run_dir: Path, stem: str) -> list[tuple[int, Path]]:
    """`<stem>.part-*.parquet` を index 昇順で。"""
    out = [(part_index(p), p) for p in run_dir.glob(f"{stem}.part-*.parquet")]
    return sorted([(i, p) for i, p in out if i >= 0])


def is_complete_parquet(path: Path) -> bool:
    """書きかけの parquet を読まないための完結判定(モジュール docstring の規律)。

    先頭 magic・末尾 magic・footer 長の整合を見るだけ(pyarrow を起動しない軽い判定)。
    存在しない/短すぎる/読めない場合は False(=まだ読まない)。
    ここも `_open_shared` で開く(この一瞬の read すらシムの unlink を弾いてしまうため)。
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


# ====================================================================== 小道具
def _fmt_tod(minute_of_day: int) -> str:
    m = int(minute_of_day) % 1440
    return f"{m // 60:02d}:{m % 60:02d}"


def _fmt_dur(sec: float | None) -> str:
    if sec is None:
        return "—"
    sec = max(0.0, float(sec))
    if sec < 90:
        return f"{sec:.0f}秒"
    if sec < 5400:
        return f"{sec / 60:.1f}分"
    return f"{sec / 3600:.1f}時間"


def _short(v, n: int = 48) -> str:
    s = str(v).replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _parse_tod(v) -> int | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return int(v) % 1440
    s = str(v).strip()
    if ":" in s:
        try:
            h, m = s.split(":")[:2]
            return (int(h) * 60 + int(m)) % 1440
        except ValueError:
            return None
    try:
        return int(s) % 1440
    except ValueError:
        return None


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def read_run_config(run_dir: Path) -> dict:
    """run dir の config.yaml(無ければリポ既定 conf/config.yaml)から必要な 3 つだけ。

    `save_config` は `observer.checkpoint_every>0` のとき run 開始時に呼ばれるが、
    `flush_every_steps` だけを有効にしたランでは finalize まで書かれない。その場合は
    リポ既定へ退避する(地図の実体はどのランでも同じ既定を指すのが通例)。
    """
    src = run_dir / "config.yaml"
    cfg = _load_yaml(src) if src.exists() else {}
    fallback = False
    if not cfg:
        cfg = _load_yaml(REPO_ROOT / "conf" / "config.yaml")
        fallback = True
    run = cfg.get("run") or {}
    world = cfg.get("world") or {}
    obs = cfg.get("observer") or {}
    map_rel = world.get("map") or "data/shibuya_osm.json"
    map_path = Path(map_rel)
    if not map_path.is_absolute():
        map_path = REPO_ROOT / map_path
    return {
        "map_path": map_path,
        "n_steps": int(run.get("n_steps") or 0) or None,
        "n_agents": int(run.get("n_agents") or 0) or None,
        "start_min": _parse_tod(run.get("start_tod")),
        # W2-3: 1 step の分数(Δt=10 なら 10 = 従来と 1 ビットも変わらない)。
        # run dir に無ければ run_dt が正準 10 を仮定し stderr に告知する(黙って仮定しない)。
        "dt_min": run_dt.dt_min_of(run_dir),
        "checkpoint_every": int(obs.get("checkpoint_every") or 0),
        "flush_every_steps": int(obs.get("flush_every_steps") or 0),
        "config_fallback": fallback,
    }


def build_background(map_path: Path) -> dict:
    """地図 JSON → 背景ジオメトリ(道路・建物・線路)。0.1m 丸めで HTML に 1 回だけ埋める。"""
    try:
        city = json.loads(Path(map_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"roads": [], "blds": [], "rails": [], "bbox": [-500, -500, 500, 500],
                "attr": "", "name": ""}

    def _poly(pts):
        return [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in pts]

    roads = [_poly(e["geometry"]) for e in city.get("edges", []) if e.get("geometry")]
    blds = [_poly(b["footprint"]) for b in city.get("buildings", []) if b.get("footprint")]
    rails = [_poly(r["geometry"]) for r in city.get("railways", []) if r.get("geometry")]
    # bbox は道路+建物だけから取る(線路は郊外まで伸びていて実測 bbox が 8km 級になり、
    # 街のスケールが潰れるため。線路は描画はするが枠には効かせない)。
    core = roads + blds or rails
    xs = [p[0] for g in core for p in g]
    ys = [p[1] for g in core for p in g]
    bbox = [min(xs), min(ys), max(xs), max(ys)] if xs else [-500, -500, 500, 500]
    meta = city.get("meta") or {}
    return {"roads": roads, "blds": blds, "rails": rails, "bbox": bbox,
            "attr": str(meta.get("attribution") or ""),
            "name": str(meta.get("name") or "")}


# ====================================================================== 追いかけ状態
class ChaseState:
    """完結済み part を増分で取り込み、画面用データを組み立てる(読み取り専用)。"""

    def __init__(self, run_dir: Path, trail_steps: int = 3, ticker_max: int = 40,
                 series_points: int = 240, series_max: int = 6,
                 max_dots: int = 0) -> None:
        self.run_dir = Path(run_dir)
        self.trail_steps = max(1, int(trail_steps))
        self.ticker_max = max(1, int(ticker_max))
        self.series_points = max(8, int(series_points))
        self.series_max = max(1, int(series_max))
        self.max_dots = int(max_dots)          # >0 で描画点を決定論間引き(巨大ラン向け)

        self.cfg = read_run_config(self.run_dir)
        self.next_l1 = 0
        self.next_l2 = 0
        self.skipped: set[int] = set()
        self.warnings: list[str] = []

        self.pos: dict[int, list] = {}                 # agent_id -> [x, y, w]
        self.trail: deque = deque(maxlen=self.trail_steps)   # [(step, {aid: (x, y)})]
        self.ticker: deque = deque(maxlen=self.ticker_max)
        self.counts: Counter = Counter()
        self.series: dict[str, list] = {}
        self.last_step: int | None = None
        self.start_min: int | None = self.cfg["start_min"]
        self.step_minutes: int = int(self.cfg["dt_min"])   # W2-3: このランの Δt [分]
        self.parts_read = 0
        self.rows_read = 0
        self.last_part_mtime: float | None = None
        self.last_part_name: str | None = None
        self.finished = False
        self.final_step: int | None = None      # 終了後に L2 canonical から拾う真の最終 step
        self._names: dict[int, str] | None = None
        self._l2_last: dict | None = None

    # ---------------------------------------------------------------- 名簿
    def _names_map(self) -> dict[int, str]:
        if self._names is None:
            path = self.run_dir / "agents.json"
            try:
                rows = json.loads(path.read_text(encoding="utf-8"))
                self._names = {int(r["id"]): str(r.get("name") or f"agent{r['id']}")
                               for r in rows}
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                return {}            # まだ書かれていない = 次の poll で読み直す
        return self._names

    # ---------------------------------------------------------------- part 選別
    def _ready_parts(self, stem: str, next_idx: int) -> tuple[list[tuple[int, Path]], int]:
        """次に読むべき完結済み part 群と、未完結で待っている数。

        index 昇順に見て、完結していれば採用。完結していないが**より新しい part が既にある**
        なら「クラッシュで書きかけのまま残った屍」と判定して恒久スキップ(先へ進む)。
        """
        parts = [(i, p) for i, p in list_parts(self.run_dir, stem) if i >= next_idx]
        ready: list[tuple[int, Path]] = []
        pending = 0
        max_idx = max((i for i, _ in parts), default=-1)
        for i, p in parts:
            if i in self.skipped:
                continue
            if is_complete_parquet(p):
                ready.append((i, p))
            elif i < max_idx:
                self.skipped.add(i)
                self.warnings.append(
                    f"{p.name} は不完全なまま新しい part が現れた"
                    f"(書きかけの残骸と判断してスキップ)")
            else:
                pending += 1
                break                     # 書き込み中の最新 part は読まない(唯一の規律)
        return ready, pending

    # ---------------------------------------------------------------- 安全な読み
    def _safe(self, path: Path, fn, what: str):
        """part を読む操作をまとめて包む。**完結判定と読み取りの間に消える**のが正常系。

        finalize は part 群を canonical へ結合してから **unlink する**(`logger._finalize_stream`)。
        追いかけ側はそのレースに必ず遭う(実測: フルスイート並列実行で FileNotFoundError)。
        消えた = そのぶんは canonical に入った、というだけなので**黙って先へ進む**のが正しい。
        観測プロセスがランの都合で落ちてはいけない(読み取り専用の原則の裏返し)。
        """
        try:
            return fn(path)
        except FileNotFoundError:
            self.warnings.append(
                f"{path.name} は読む前に消えた(finalize が canonical へ結合・{what})")
        except Exception as exc:              # 壊れた part でも監視は続ける(捨てて先へ)
            self.warnings.append(
                f"{path.name} を読めない({what}: {type(exc).__name__}: {exc})")
        return None

    # ---------------------------------------------------------------- L1 取り込み
    def _ingest_l1(self, path: Path) -> None:
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        def _read(p: Path):
            """1 ハンドルで schema と 2 つの列集合を読む(開き直さない=レース窓を最小化)。"""
            with _open_shared(p) as fh:
                pf = pq.ParquetFile(fh)
                have = set(pf.schema_arrow.names)
                if not {"step", "agent_id", "kind", "x", "y"} <= have:
                    return have, None, None
                base = [c for c in ("step", "sim_min", "agent_id", "kind", "x", "y")
                        if c in have]
                hi_cols = [c for c in ("step", "sim_min", "agent_id", "kind", "payload")
                           if c in have]
                return have, pf.read(columns=base), pf.read(columns=hi_cols)

        got = self._safe(path, _read, "l1")
        if got is None:
            return
        have, tbl, hi_all = got
        if tbl is None:
            self.warnings.append(f"{path.name}: L1 の必須列が無い(読み飛ばす)")
            return
        if tbl.num_rows == 0:
            return
        self.rows_read += tbl.num_rows
        kind_col = tbl.column("kind")

        # --- 壁時計の原点(make_viewer._derive_start_min と同じ不変量) ---
        if self.start_min is None and "sim_min" in have:
            sm = tbl.column("sim_min")[0].as_py()
            st = tbl.column("step")[0].as_py()
            if sm is not None and st is not None:
                self.start_min = int(sm) - int(st) * self.step_minutes

        # --- 位置(payload は読まない) ---
        mask = pc.and_(pc.is_in(kind_col, value_set=_pa_array(list(POS_KINDS))),
                       pc.greater_equal(tbl.column("agent_id"), 0))
        pos_tbl = tbl.filter(mask)
        max_step = int(pc.max(tbl.column("step")).as_py())
        trail_from = max_step - self.trail_steps + 1
        if pos_tbl.num_rows:
            steps = pos_tbl.column("step").to_pylist()
            aids = pos_tbl.column("agent_id").to_pylist()
            xs = pos_tbl.column("x").to_pylist()
            ys = pos_tbl.column("y").to_pylist()
            kinds = pos_tbl.column("kind").to_pylist()
            prev_step = None
            for st, aid, k, x, y in zip(steps, aids, kinds, xs, ys):
                if prev_step is not None and st != prev_step and prev_step >= trail_from:
                    self._push_trail(prev_step)
                prev_step = st
                cur = self.pos.get(aid)
                if cur is None:
                    cur = [0.0, 0.0, W_ROAD]
                    self.pos[aid] = cur
                op = POS_KINDS[k]
                if op == "xy":
                    cur[0], cur[1] = round(float(x), 1), round(float(y), 1)
                elif op == "in":
                    cur[0], cur[1], cur[2] = round(float(x), 1), round(float(y), 1), W_INDOOR
                elif op == "out":
                    cur[0], cur[1], cur[2] = round(float(x), 1), round(float(y), 1), W_ROAD
                elif op == "back":
                    cur[0], cur[1], cur[2] = round(float(x), 1), round(float(y), 1), W_ROAD
                elif op == "gone":
                    cur[2] = W_OUTSIDE
                elif op == "sleep":
                    cur[2] = W_SLEEP
                elif op == "wake":
                    if cur[2] == W_SLEEP:
                        cur[2] = W_ROAD
            if prev_step is not None and prev_step >= trail_from:
                self._push_trail(prev_step)

        # --- 見せ場イベント(ここだけ payload を読む) ---
        hi_mask = pc.is_in(kind_col, value_set=_pa_array(sorted(HIGHLIGHT_KINDS)))
        if (pc.sum(pc.cast(hi_mask, "int64")).as_py() or 0) > 0:
            hi = hi_all.filter(hi_mask)
            names = self._names_map()
            for r in hi.to_pylist():
                kind = r["kind"]
                self.counts[kind] += 1
                aid = int(r["agent_id"])
                sm = r.get("sim_min")
                sm = int(sm) if sm is not None else self._sim_min(int(r["step"]))
                self.ticker.append({
                    "s": int(r["step"]),
                    "d": sm // 1440,
                    "t": _fmt_tod(sm),
                    "a": ("世界" if aid < 0 else names.get(aid, f"agent{aid}")),
                    "k": kind,
                    "x": _payload_text(r.get("payload")),
                })

        self.last_step = max_step if self.last_step is None else max(self.last_step, max_step)
        self.parts_read += 1
        try:
            self.last_part_mtime = path.stat().st_mtime
            self.last_part_name = path.name
        except OSError:
            pass

    def _push_trail(self, step: int) -> None:
        if self.trail and self.trail[-1][0] == step:
            return
        self.trail.append((step, {a: (v[0], v[1]) for a, v in self.pos.items()}))

    # ---------------------------------------------------------------- L2 取り込み
    def _ingest_l2(self, path: Path) -> None:
        import pyarrow.parquet as pq

        def _read(p: Path):
            with _open_shared(p) as fh:
                pf = pq.ParquetFile(fh)
                head = pf.schema_arrow.names
                cols = ["step"] + [k for k, _, _ in SERIES_PREF
                                   if k in head and k != "step"]
                if "step" not in head or len(cols) <= 1:
                    return None, None
                return cols, pf.read(columns=cols)

        got = self._safe(path, _read, "l2")
        if got is None:
            return
        cols, tbl = got
        if tbl is None:
            return
        rows = tbl.to_pylist()
        for r in rows:
            st = r.get("step")
            if st is None:
                continue
            for k in cols[1:]:
                v = r.get(k)
                if v is None:
                    continue
                self.series.setdefault(k, []).append([int(st), float(v)])
        if rows:
            self._l2_last = {k: rows[-1].get(k) for k in cols}

    # ---------------------------------------------------------------- poll
    def poll(self) -> dict:
        """新しい完結 part を取り込み、画面用データ dict を返す。"""
        self.warnings = []
        l1_ready, l1_pending = self._ready_parts("l1_events", self.next_l1)
        for i, p in l1_ready:
            self._ingest_l1(p)
            self.next_l1 = i + 1
        l2_ready, _ = self._ready_parts("l2_metrics", self.next_l2)
        for i, p in l2_ready:
            self._ingest_l2(p)
            self.next_l2 = i + 1

        summary = self.run_dir / "summary.json"
        if summary.exists() and not self.finished:
            self.finished = True
            self._on_finish()
        return self._data(l1_pending)

    def _on_finish(self) -> None:
        """finalize 後の後始末。canonical L1(巨大)は読まない — L2 の最終行だけ拾う。"""
        import pyarrow.parquet as pq
        path = self.run_dir / "l2_metrics.parquet"
        if not path.exists():
            return
        try:
            with _open_shared(path) as fh:        # canonical も同じ非侵襲な開き方で統一
                pf = pq.ParquetFile(fh)
                head = pf.schema_arrow.names
                cols = ["step"] + [k for k, _, _ in SERIES_PREF
                                   if k in head and k != "step"]
                if len(cols) <= 1:
                    return
                tbl = pf.read(columns=cols)
            if tbl.num_rows == 0:
                return
            last = tbl.slice(tbl.num_rows - 1, 1).to_pylist()[0]
            self._l2_last = last
            self.final_step = int(last.get("step") or 0)
            # L2 canonical は小さい(1 step 1 行)ので、追いかけ中に読めなかった末尾ぶんだけ
            # 系列に足して完結させる。既読 step より後だけを足すので二重計上しない。
            # **L1 canonical は読まない**(10日ランで巨大 = 監視プロセスが食い潰す)。
            seen = {k: (v[-1][0] if v else -1) for k, v in self.series.items()}
            for r in tbl.to_pylist():
                st = r.get("step")
                if st is None:
                    continue
                for k in cols[1:]:
                    v = r.get(k)
                    if v is None or int(st) <= seen.get(k, -1):
                        continue
                    self.series.setdefault(k, []).append([int(st), float(v)])
            if self.last_step is None:
                self.last_step = self.final_step
        except Exception as exc:                      # 読めなくても画面は出す
            self.warnings.append(f"l2_metrics.parquet を読めない: {exc}")

    # ---------------------------------------------------------------- 出力データ
    def _sim_min(self, step: int) -> int:
        base = self.start_min if self.start_min is not None else DEFAULT_START_MIN
        return int(base) + int(step) * self.step_minutes

    def _dot_ids(self) -> list[int]:
        ids = sorted(self.pos)
        if self.max_dots and len(ids) > self.max_dots:
            stride = len(ids) / float(self.max_dots)        # 決定論間引き(乱数ゼロ)
            ids = [ids[int(i * stride)] for i in range(self.max_dots)]
        return ids

    def _series_out(self) -> list[dict]:
        out = []
        for key, label, fmt in SERIES_PREF:
            pts = self.series.get(key)
            if not pts:
                continue
            step = max(1, len(pts) // self.series_points)
            thin = pts[::step]
            if thin and thin[-1] != pts[-1]:
                thin.append(pts[-1])
            vals = [v for _, v in thin]
            out.append({"key": key, "label": label, "fmt": fmt, "pts": thin,
                        "last": pts[-1][1], "min": min(vals), "max": max(vals)})
            if len(out) >= self.series_max:
                break
        return out

    def _status(self, pending: int) -> tuple[str, str]:
        if self.finished:
            msg = "ラン終了(summary.json 検出)= 追いかけ完了"
            if (self.final_step is not None and self.last_step is not None
                    and self.final_step > self.last_step):
                msg += (f"。地図は step {self.last_step} 止まり"
                        f"(最後の flush ぶん step {self.last_step + 1}〜{self.final_step} は"
                        f" finalize で canonical へ結合され part が消えたため)。全体は通常ビューアで")
            return "finished", msg
        if self.last_step is None:
            if (self.cfg["checkpoint_every"] <= 0 and self.cfg["flush_every_steps"] <= 0
                    and (self.run_dir / "config.yaml").exists()
                    and not self.cfg["config_fallback"]):
                return ("no_parts",
                        "このランは observer.checkpoint_every=0 かつ flush_every_steps=0 = "
                        "part を書かない設定なので追いかけられない(終了後に "
                        "viz/make_viewer.py を使う)")
            return "waiting", "最初の part(flush)を待っている"
        return "chasing", f"追いかけ中(未完結の part: {pending})"

    def _data(self, pending: int) -> dict:
        now = time.time()
        ids = self._dot_ids()
        pos = [[self.pos[a][0], self.pos[a][1], self.pos[a][2]] for a in ids]
        trail = []
        for step, snap in list(self.trail):
            if self.last_step is not None and step >= self.last_step:
                continue
            trail.append([list(snap[a]) if a in snap else None for a in ids])
        stat = Counter(v[2] for v in self.pos.values())
        step = self.last_step
        sim_min = self._sim_min(step) if step is not None else None
        n_steps = self.cfg["n_steps"]
        status, note = self._status(pending)
        lag = (now - self.last_part_mtime) if self.last_part_mtime else None
        return {
            "gen": 0,                                  # 呼び出し側が上書き
            "run": self.run_dir.name,
            "run_dir": str(self.run_dir),
            "wrote_at": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S"),
            "wrote_epoch": now,
            "status": status,
            "note": note,
            "finished": self.finished,
            "warnings": list(self.warnings),
            "chase": {
                "step": step,
                "n_steps": n_steps,
                "day": (sim_min // 1440) if sim_min is not None else None,
                "tod": _fmt_tod(sim_min) if sim_min is not None else "--:--",
                "sim_min": sim_min,
                "progress": (round(min(1.0, (step + 1) / n_steps), 4)
                             if (step is not None and n_steps) else None),
            },
            "lag": {
                "seconds": (round(lag, 1) if lag is not None else None),
                "text": _fmt_dur(lag),
                "src": self.last_part_name,
                "parts_read": self.parts_read,
                "parts_pending": pending,
                "rows_read": self.rows_read,
            },
            "ids": ids,
            "pos": pos,
            "trail": trail,
            "stat": {"n": len(self.pos), "road": stat.get(W_ROAD, 0),
                     "indoor": stat.get(W_INDOOR, 0),
                     "outside": stat.get(W_OUTSIDE, 0),
                     "sleep": stat.get(W_SLEEP, 0)},
            "ticker": list(self.ticker)[::-1],
            "counts": dict(sorted(self.counts.items(), key=lambda kv: (-kv[1], kv[0]))),
            "series": self._series_out(),
            # L2 は finalize 後に canonical で完結させるため、地図の追いかけ位置より先へ行くことが
            # ある(その差は画面に出す=どの step の値かを黙って混ぜない)。
            "series_step": max((v[-1][0] for v in self.series.values() if v), default=None),
            "l2_last": self._l2_last,
            "viewer_cmd": f"python viz/make_viewer.py {self.run_dir.as_posix()}",
        }


def _pa_array(values):
    import pyarrow as pa
    return pa.array(values, pa.string())


def _payload_text(payload) -> str:
    """ティッカー1行の本文。payload の代表キー 1 個を短く出すだけ(要約も推測もしない)。"""
    if not payload:
        return ""
    try:
        p = json.loads(payload)
    except (json.JSONDecodeError, TypeError):
        return _short(payload)
    if not isinstance(p, dict):
        return _short(p)
    for k in _TEXT_KEYS:
        if k in p and p[k] not in (None, "", [], {}):
            return _short(p[k])
    for k in sorted(p):
        if p[k] not in (None, "", [], {}):
            return _short(f"{k}={p[k]}")
    return ""


# ====================================================================== 出力
def write_data(out_dir: Path, data: dict) -> Path:
    """live_data.js(JSONP)を原子的に置き換える。ブラウザが掴んでいても壊れた JS を見せない。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "live_data.js"
    tmp = out_dir / f"live_data.js.{os.getpid()}.tmp"      # 二重起動でも tmp が衝突しない
    body = "LIVE_DATA(" + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ");\n"
    tmp.write_text(body, encoding="utf-8")
    for attempt in range(6):                 # Windows: 読み取り中は置換が弾かれることがある
        try:
            os.replace(tmp, path)
            return path
        except OSError:
            if attempt == 5:
                path.write_text(body, encoding="utf-8")     # 最後の手段(非原子)
                try:
                    tmp.unlink()
                except OSError:
                    pass
                return path
            time.sleep(0.25)
    return path


def _json_for_script(text: str) -> str:
    """`<script type=application/json>` 内に安全に埋める(viz/make_viewer3d と同じ流儀)。"""
    return text.replace("</", "<\\/")


def render_html(bg: dict, run_name: str, interval: float,
                refresh: str = "js") -> str:
    html = _HTML
    html = html.replace("__BG_JSON__",
                        _json_for_script(json.dumps(bg, ensure_ascii=False,
                                                    separators=(",", ":"))))
    html = html.replace("__RUN_NAME__", _esc_html(run_name))
    html = html.replace("__INTERVAL_MS__", str(int(max(2.0, interval) * 1000)))
    meta = ('<meta http-equiv="refresh" content="%d">' % int(max(5.0, interval))
            if refresh == "meta" else "")
    return html.replace("__META_REFRESH__", meta)


def _esc_html(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


# ====================================================================== CLI
def _resolve_run_dir(arg: str) -> Path:
    """絶対パス→そのまま。相対は cwd 起点を優先し、無ければリポ直下起点(watchdog_llm と同流儀)。"""
    p = Path(arg)
    if p.is_absolute() or p.exists():
        return p
    return REPO_ROOT / arg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="追いかけ再生: 実行中のランの part parquet を読むだけのライブ風画面")
    ap.add_argument("run_dir", help="ラン出力ディレクトリ(例 runs/prod1)")
    ap.add_argument("--interval", type=float, default=DEFAULT_INTERVAL,
                    help=f"ポーリング間隔[秒](既定 {DEFAULT_INTERVAL:g})")
    ap.add_argument("--out-dir", default=None,
                    help=f"HTML の出力先(既定 <run-dir>/{DEFAULT_OUT_SUBDIR})")
    ap.add_argument("--trail", type=int, default=3, help="残像の step 数(既定3)")
    ap.add_argument("--ticker", type=int, default=40, help="ティッカー保持件数(既定40)")
    ap.add_argument("--series-max", type=int, default=6,
                    help="スパークラインの本数上限(既定6)")
    ap.add_argument("--max-dots", type=int, default=0,
                    help=">0 で描画点を決定論間引き(巨大ラン向け。既定0=全点)")
    ap.add_argument("--refresh", choices=("js", "meta"), default="js",
                    help="更新方式: js=データだけ差し替え(既定)/ meta=ページ全体を再読込")
    ap.add_argument("--once", action="store_true", help="1 回だけ生成して終了")
    ap.add_argument("--max-polls", type=int, default=0, help=">0 で回数上限")
    ap.add_argument("--timeout-min", type=float, default=0.0,
                    help=">0 で総時間上限[分]")
    ap.add_argument("--keep-watching", action="store_true",
                    help="ラン終了(summary.json)後も監視を続ける")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    run_dir = _resolve_run_dir(args.run_dir)
    if not run_dir.is_dir():
        print(f"[live] ラン dir が無い: {run_dir}", file=sys.stderr)
        return 2
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / DEFAULT_OUT_SUBDIR
    if not Path(out_dir).is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir = Path(out_dir)

    state = ChaseState(run_dir, trail_steps=args.trail, ticker_max=args.ticker,
                       series_max=args.series_max, max_dots=args.max_dots)
    bg = build_background(state.cfg["map_path"])
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "live.html"
    html_path.write_text(render_html(bg, run_dir.name, args.interval, args.refresh),
                         encoding="utf-8")
    if not args.quiet:
        print(f"[live] 追いかけ再生: {run_dir}")
        print(f"[live] 画面: {html_path}  (ブラウザで開いたままにする)")
        print(f"[live] 間隔: {args.interval:g}s  更新方式: {args.refresh}")

    t0 = time.time()
    gen = 0
    while True:
        gen += 1
        try:
            data = state.poll()
        except Exception as exc:                       # 観測側の失敗でランは止めない
            print(f"[live] poll 失敗(継続する): {exc!r}", file=sys.stderr)
            time.sleep(min(args.interval, 10.0))
            continue
        data["gen"] = gen
        data["interval"] = args.interval
        write_data(out_dir, data)
        if not args.quiet:
            c = data["chase"]
            print(f"[live] #{gen} {data['status']} step={c['step']} "
                  f"day{c['day']} {c['tod']} 遅延={data['lag']['text']} "
                  f"parts={data['lag']['parts_read']} ticker={len(data['ticker'])}")
            for w in data["warnings"]:
                print(f"[live] WARN {w}", file=sys.stderr)
        if args.once:
            break
        if state.finished and not args.keep_watching:
            if not args.quiet:
                print(f"[live] 追いかけ完了。通常ビューア: {data['viewer_cmd']}")
            break
        if args.max_polls and gen >= args.max_polls:
            break
        if args.timeout_min and (time.time() - t0) >= args.timeout_min * 60.0:
            break
        time.sleep(max(0.05, args.interval))
    return 0


# ====================================================================== HTML
# 更新方式の判断(10日間つけっぱなしに強いのはどちらか):
#   案A 毎回 HTML 全体を再生成 + meta refresh …… 10日×60秒 = 1.4万回のフルリロード。
#        200KB の背景ジオメトリを毎回パースし直し、pan/zoom も毎回失われる。
#   案B 静的 HTML(1回)+ JSONP データを差し替え …… ページは一度も再読込されない。
#        背景は最初の1回だけ、以後は数十〜数百KB のデータだけが飛ぶ。pan/zoom は保たれる。
#   → **案B を採用**(--refresh meta で案A 相当も出せるようにはしてある)。
#   file:// で動かす必要があるため取り込みは `fetch` ではなく `<script src>`(第76バッチの
#   知見: file:// では fetch が CORS で不可・script src は可)。同名ファイルを差し替えるので
#   クエリ `?v=<gen>&t=<epoch ms>` でキャッシュを外す(file:// でもクエリは無視されて
#   同じ実体が読まれる = 実体1つ・URL は毎回別)。挿した script 要素は実行後に必ず外す
#   (10日で1.4万ノード溜めないため)。
_HTML = r"""<!doctype html>
<html lang="ja"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>追いかけ再生 — __RUN_NAME__</title>
__META_REFRESH__
<style>
:root{--bg:#0e1116;--panel:#161b22;--line:#232b36;--fg:#e6edf3;--dim:#8b949e;
      --road:#2b3644;--bld:#1b2430;--rail:#3a2f4a;
      --road_a:#ffb454;--indoor_a:#4aa3ff;--sleep_a:#a78bfa;--out_a:#5b6673;--ok:#3fb950;--warn:#d29922}
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);
  font-family:"Yu Gothic UI","Hiragino Kaku Gothic ProN",Meiryo,system-ui,sans-serif}
body{display:flex;flex-direction:column}
header{display:flex;align-items:center;gap:14px;padding:8px 14px;background:var(--panel);
  border-bottom:1px solid var(--line);flex-wrap:wrap;flex:none}
header h1{font-size:15px;margin:0;font-weight:600;letter-spacing:.02em}
.badge{font-size:12px;padding:2px 8px;border-radius:10px;background:#21262d;color:var(--dim);
  border:1px solid var(--line);white-space:nowrap}
.badge b{color:var(--fg);font-weight:600}
.badge.live{color:var(--ok);border-color:#1f4d2b}
.badge.warn{color:var(--warn);border-color:#4d3d0f}
#clock{font-size:22px;font-weight:700;font-variant-numeric:tabular-nums;letter-spacing:.03em}
#prog{flex:1;min-width:120px;height:6px;background:#21262d;border-radius:3px;overflow:hidden}
#prog>i{display:block;height:100%;width:0;background:linear-gradient(90deg,#1f6feb,#3fb950)}
main{display:flex;flex:1;min-height:0}
#mapwrap{position:relative;flex:1;min-width:0}
canvas{display:block;width:100%;height:100%;cursor:grab}
canvas.drag{cursor:grabbing}
#legend{position:absolute;left:10px;bottom:10px;background:rgba(13,17,23,.82);border:1px solid var(--line);
  border-radius:8px;padding:7px 10px;font-size:11px;color:var(--dim);line-height:1.7}
#legend i{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px;vertical-align:1px}
#attr{position:absolute;right:10px;bottom:10px;font-size:10px;color:#59636e}
aside{width:370px;flex:none;border-left:1px solid var(--line);background:var(--panel);
  display:flex;flex-direction:column;min-height:0}
aside section{border-bottom:1px solid var(--line);padding:9px 12px}
aside h2{font-size:11px;color:var(--dim);margin:0 0 7px;font-weight:600;letter-spacing:.08em}
#sparks{display:grid;grid-template-columns:1fr 1fr;gap:7px}
.sp{background:#0d1117;border:1px solid var(--line);border-radius:6px;padding:5px 6px}
.sp .l{font-size:10px;color:var(--dim);display:flex;justify-content:space-between;gap:4px}
.sp .v{font-size:14px;font-weight:700;font-variant-numeric:tabular-nums}
.sp svg{display:block;width:100%;height:24px}
#tickwrap{flex:1;overflow:auto;padding:0}
#tick{list-style:none;margin:0;padding:0;font-size:12px}
#tick li{display:flex;gap:7px;padding:5px 12px;border-bottom:1px solid #1b212a;align-items:baseline}
#tick .tm{color:var(--dim);font-variant-numeric:tabular-nums;white-space:nowrap;font-size:11px}
#tick .kd{color:#79c0ff;white-space:nowrap;font-size:11px}
#tick .nm{color:#d2a8ff;white-space:nowrap;max-width:88px;overflow:hidden;text-overflow:ellipsis}
#tick .tx{color:var(--fg);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
#note{padding:7px 12px;font-size:11px;color:var(--dim);border-bottom:1px solid var(--line)}
#note.bad{color:var(--warn)}
#fin{display:none;padding:9px 12px;background:#12261a;border-bottom:1px solid #1f4d2b;font-size:12px}
#fin code{display:block;margin-top:5px;background:#0d1117;border:1px solid var(--line);border-radius:5px;
  padding:5px 7px;font-size:11px;color:#7ee787;word-break:break-all}
#stat{display:flex;gap:12px;font-size:11px;color:var(--dim);flex-wrap:wrap}
#stat b{color:var(--fg);font-variant-numeric:tabular-nums}
</style></head><body>
<header>
  <h1>追いかけ再生 <span style="color:var(--dim);font-weight:400">__RUN_NAME__</span></h1>
  <div id="clock">--日 --:--</div>
  <div id="prog"><i></i></div>
  <span class="badge" id="b-step">step —</span>
  <span class="badge" id="b-lag">遅延 —</span>
  <span class="badge" id="b-gen">更新 —</span>
</header>
<main>
  <div id="mapwrap">
    <canvas id="map"></canvas>
    <div id="legend">
      <i style="background:var(--road_a)"></i>路上 <i style="background:var(--indoor_a)"></i>屋内
      <i style="background:var(--sleep_a)"></i>就寝 <i style="background:var(--out_a)"></i>範囲外
      <div style="margin-top:3px">ホイール=拡大 / ドラッグ=移動 / ダブルクリック=全体</div>
    </div>
    <div id="attr"></div>
  </div>
  <aside>
    <div id="note">データ待ち…</div>
    <div id="fin"><b>追いかけ完了</b> — 通常ビューアで全体を見る:<code id="fincmd"></code></div>
    <section><h2>いま街にいる人</h2><div id="stat"></div></section>
    <section><h2 id="sparkh">L2 系列</h2><div id="sparks"></div></section>
    <section style="flex:1;display:flex;flex-direction:column;min-height:0;padding:9px 0 0">
      <h2 style="padding:0 12px">出来事(新しい順)</h2>
      <div id="tickwrap"><ul id="tick"></ul></div>
    </section>
  </aside>
</main>
<script type="application/json" id="bg-data">__BG_JSON__</script>
<script>
"use strict";
var BG = JSON.parse(document.getElementById("bg-data").textContent);
var INTERVAL = __INTERVAL_MS__;
var CV = document.getElementById("map"), CX = CV.getContext("2d");
var VIEW = {s:1, ox:0, oy:0, init:false};
var BGC = null, BGKEY = "";
var LAST = null, DPR = Math.min(2, window.devicePixelRatio || 1);

function esc(s){ return String(s==null?"":s).replace(/[&<>"]/g, function(c){
  return {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]; }); }

function size(){
  var w = CV.clientWidth, h = CV.clientHeight;
  if(CV.width !== Math.round(w*DPR) || CV.height !== Math.round(h*DPR)){
    CV.width = Math.round(w*DPR); CV.height = Math.round(h*DPR); BGC = null;
  }
  return [w, h];
}
function fit(){
  var wh = size(), w = wh[0], h = wh[1], b = BG.bbox;
  var bw = Math.max(1, b[2]-b[0]), bh = Math.max(1, b[3]-b[1]);
  VIEW.s = Math.min(w/(bw*1.06), h/(bh*1.06));
  VIEW.ox = 0; VIEW.oy = 0; VIEW.init = true; BGC = null;
}
function cx(){ return (BG.bbox[0]+BG.bbox[2])/2; }
function cy(){ return (BG.bbox[1]+BG.bbox[3])/2; }
function sx(x, w){ return (x-cx())*VIEW.s + w/2 + VIEW.ox; }
function sy(y, h){ return -(y-cy())*VIEW.s + h/2 + VIEW.oy; }

function drawBG(w, h){
  var key = w+"x"+h+"@"+VIEW.s.toFixed(5)+","+VIEW.ox.toFixed(1)+","+VIEW.oy.toFixed(1);
  if(BGC && BGKEY === key) return;
  BGC = document.createElement("canvas");
  BGC.width = Math.round(w*DPR); BGC.height = Math.round(h*DPR);
  var g = BGC.getContext("2d"); g.scale(DPR, DPR);
  g.fillStyle = (getComputedStyle(document.documentElement)
    .getPropertyValue("--bg") || "").trim() || "#0e1116";
  g.fillRect(0, 0, w, h);
  var i, j, p;
  g.fillStyle = "#1b2430"; g.strokeStyle = "#222d3a"; g.lineWidth = 0.6;
  for(i=0;i<BG.blds.length;i++){ p = BG.blds[i]; if(p.length<3) continue;
    g.beginPath(); g.moveTo(sx(p[0][0],w), sy(p[0][1],h));
    for(j=1;j<p.length;j++) g.lineTo(sx(p[j][0],w), sy(p[j][1],h));
    g.closePath(); g.fill(); g.stroke(); }
  g.strokeStyle = "#2b3644"; g.lineWidth = 1.1;
  for(i=0;i<BG.roads.length;i++){ p = BG.roads[i]; if(p.length<2) continue;
    g.beginPath(); g.moveTo(sx(p[0][0],w), sy(p[0][1],h));
    for(j=1;j<p.length;j++) g.lineTo(sx(p[j][0],w), sy(p[j][1],h)); g.stroke(); }
  g.strokeStyle = "#3a2f4a"; g.lineWidth = 1.6;
  for(i=0;i<BG.rails.length;i++){ p = BG.rails[i]; if(p.length<2) continue;
    g.beginPath(); g.moveTo(sx(p[0][0],w), sy(p[0][1],h));
    for(j=1;j<p.length;j++) g.lineTo(sx(p[j][0],w), sy(p[j][1],h)); g.stroke(); }
  BGKEY = key;
}
var COLOR = {0:"#ffb454", 1:"#4aa3ff", "-1":"#5b6673", "-2":"#a78bfa"};
function draw(){
  if(!VIEW.init) fit();
  var wh = size(), w = wh[0], h = wh[1];
  drawBG(w, h);
  CX.setTransform(1,0,0,1,0,0); CX.scale(DPR, DPR);
  CX.clearRect(0,0,w,h); CX.drawImage(BGC, 0, 0, w, h);
  if(!LAST) return;
  var t, i, p, n;
  var tr = LAST.trail || [];
  for(t=0;t<tr.length;t++){
    CX.globalAlpha = 0.10 + 0.16*(t+1)/tr.length;
    CX.fillStyle = "#ffffff";
    for(i=0;i<tr[t].length;i++){ p = tr[t][i]; if(!p) continue;
      CX.fillRect(sx(p[0],w)-0.8, sy(p[1],h)-0.8, 1.6, 1.6); }
  }
  CX.globalAlpha = 1;
  var pos = LAST.pos || [], r = Math.max(1.6, Math.min(3.4, 1.2 + VIEW.s*1.4));
  for(i=0;i<pos.length;i++){
    p = pos[i]; if(p[2] === -1) continue;
    CX.fillStyle = COLOR[p[2]] || "#ffffff";
    CX.beginPath(); CX.arc(sx(p[0],w), sy(p[1],h), r, 0, 6.2832); CX.fill();
  }
}
function spark(s){
  var pts = s.pts || [], i, lo = s.min, hi = s.max, d = (hi-lo) || 1;
  var W = 100, H = 24, out = [];
  for(i=0;i<pts.length;i++){
    var x = pts.length<2 ? W : (i/(pts.length-1))*W;
    var y = H - 2 - ((pts[i][1]-lo)/d)*(H-4);
    out.push(x.toFixed(1)+","+y.toFixed(1));
  }
  var v = s.last;
  var txt = s.fmt==="rate" ? (v*100).toFixed(1)+"%"
          : s.fmt==="int" ? Math.round(v).toLocaleString()
          : (Math.abs(v)>=100 ? v.toFixed(0) : v.toFixed(3));
  return '<div class="sp"><div class="l"><span>'+esc(s.label)+'</span></div>'
    + '<div class="v">'+esc(txt)+'</div>'
    + '<svg viewBox="0 0 '+W+' '+H+'" preserveAspectRatio="none">'
    + '<polyline fill="none" stroke="#58a6ff" stroke-width="1.2" points="'+out.join(" ")+'"/>'
    + '</svg></div>';
}
function render(d){
  var c = d.chase || {};
  document.getElementById("clock").textContent =
    (c.day==null ? "--日" : (c.day+1)+"日目") + " " + (c.tod || "--:--");
  document.querySelector("#prog>i").style.width =
    ((c.progress==null ? 0 : c.progress*100).toFixed(1)) + "%";
  document.getElementById("b-step").innerHTML =
    "step <b>" + (c.step==null?"—":c.step) + "</b>" + (c.n_steps? " / "+c.n_steps : "");
  var bl = document.getElementById("b-lag");
  bl.innerHTML = "遅延 <b>" + esc((d.lag||{}).text || "—") + "</b>";
  bl.className = "badge" + (d.status==="chasing" ? " live" : (d.status==="no_parts" ? " warn" : ""));
  /* 生成側(live_viewer プロセス)が止まっていないか = ブラウザ側で毎回測る。
     データが同じ gen のまま古びていくのを黙って表示し続けない。 */
  var age = Date.now()/1000 - (d.wrote_epoch || 0);
  var iv = (d.interval || INTERVAL/1000);
  var bg2 = document.getElementById("b-gen");
  bg2.innerHTML = "更新 <b>#" + d.gen + "</b> " + esc(d.wrote_at||"")
    + (age > 3*iv ? " <b>(生成が止まっている?)</b>" : "");
  bg2.className = "badge" + (age > 3*iv ? " warn" : "");
  var note = document.getElementById("note");
  note.textContent = (d.note||"") + ((d.warnings&&d.warnings.length) ? " / " + d.warnings.join(" / ") : "");
  note.className = (d.status==="no_parts" || (d.warnings&&d.warnings.length)) ? "bad" : "";
  var fin = document.getElementById("fin");
  fin.style.display = d.finished ? "block" : "none";
  document.getElementById("fincmd").textContent = d.viewer_cmd || "";
  var st = d.stat || {};
  document.getElementById("stat").innerHTML =
    "全体 <b>"+(st.n||0)+"</b>　路上 <b>"+(st.road||0)+"</b>　屋内 <b>"+(st.indoor||0)
    +"</b>　就寝 <b>"+(st.sleep||0)+"</b>　範囲外 <b>"+(st.outside||0)+"</b>";
  var sp = d.series || [], html = "", i;
  document.getElementById("sparkh").textContent =
    "L2 系列" + (d.series_step==null ? "" : "(step " + d.series_step + " 時点)");
  for(i=0;i<sp.length;i++) html += spark(sp[i]);
  document.getElementById("sparks").innerHTML = html || '<div class="sp"><div class="l">L2 待ち</div></div>';
  var tk = d.ticker || [], li = "";
  for(i=0;i<tk.length;i++){
    var e = tk[i];
    li += '<li><span class="tm">'+(e.d+1)+'d '+esc(e.t)+'</span>'
       + '<span class="kd">'+esc(e.k)+'</span>'
       + '<span class="nm">'+esc(e.a)+'</span>'
       + '<span class="tx">'+esc(e.x)+'</span></li>';
  }
  document.getElementById("tick").innerHTML = li
    || '<li><span class="tm">—</span><span class="tx">まだ出来事がない</span></li>';
  document.getElementById("attr").textContent = BG.attr || "";
  draw();
}
/* JSONP 受け口: live_data.js の末尾がこれを呼ぶ */
function LIVE_DATA(d){
  LAST = d;
  try { render(d); } catch(err){ console.error(err); }
  var cur = document.currentScript;
  if(cur && cur.parentNode) cur.parentNode.removeChild(cur);   /* ノードを溜めない */
  if(d.finished) return;                                       /* 完了したら止める */
  schedule(d.interval ? d.interval*1000 : INTERVAL);
}
var _timer = null;
function schedule(ms){
  if(_timer) clearTimeout(_timer);
  _timer = setTimeout(poll, Math.max(2000, ms||INTERVAL));
}
function poll(){
  var s = document.createElement("script");
  s.src = "live_data.js?v=" + (LAST ? LAST.gen : 0) + "&t=" + Date.now();
  s.async = true;
  s.onerror = function(){
    if(s.parentNode) s.parentNode.removeChild(s);
    schedule(INTERVAL);          /* まだ書かれていない/差し替え中 = 次の周期で再試行 */
  };
  document.head.appendChild(s);
}
/* --- 操作 --- */
var drag = null;
CV.addEventListener("mousedown", function(e){ drag = [e.clientX, e.clientY, VIEW.ox, VIEW.oy];
  CV.classList.add("drag"); });
window.addEventListener("mouseup", function(){ drag = null; CV.classList.remove("drag"); });
window.addEventListener("mousemove", function(e){
  if(!drag) return;
  VIEW.ox = drag[2] + (e.clientX-drag[0]); VIEW.oy = drag[3] + (e.clientY-drag[1]); draw(); });
CV.addEventListener("wheel", function(e){
  e.preventDefault();
  var k = Math.exp(-e.deltaY*0.0015), r = CV.getBoundingClientRect();
  var mx = e.clientX-r.left-r.width/2-VIEW.ox, my = e.clientY-r.top-r.height/2-VIEW.oy;
  VIEW.s *= k; VIEW.ox -= mx*(k-1); VIEW.oy -= my*(k-1); draw();
}, {passive:false});
CV.addEventListener("dblclick", function(){ fit(); draw(); });
window.addEventListener("resize", function(){ BGC = null; draw(); });
fit(); draw(); poll();
</script>
</body></html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
