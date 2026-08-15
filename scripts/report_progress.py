#!/usr/bin/env python
"""進捗報告サイドカー = 走行中のランを**読むだけ**で Discord へ途中経過を出す 1 プロセス。

設計正典: docs/plans/progress-reporting-plan.md(レーン 1 + 2 = 最小構成)。

    # 1 サイクルだけ(タスクスケジューラ / cron 向け。既定)
    python scripts/report_progress.py runs/finals --dry-run
    # 常駐(tmux / systemd 向け)
    python scripts/report_progress.py runs/finals --interval 900
    # 途中取り出しだけ(投稿しない)
    python scripts/report_progress.py runs/finals --extract --day 3

何をするか(3 系統)
--------------------
* **H ハートビート**: 1 通を `?wait=true` で立てて以後 PATCH で編集し続ける。チャンネルを
  汚さずに「生きている・どこまで進んだ」だけを更新する。
* **D 日次ダイジェスト**: **シミュ日の境界**で 1 通ずつ新規投稿(embed)。前日比つき。
  `--extract` と同じ機構で `rollup.html` を作って添付する(`--no-attach` で止まる)。
* **A アラート**: 状態が**遷移したときだけ**。クールダウン / ヒステリシス / 毎時上限つき。
  抑制した件数は次の日次ダイジェストに**必ず数を出す**(silent cap 禁止)。

どのランにも使える(汎用性)
----------------------------
入力は run-dir だけ。診断ラン・リハーサルラン・本選ラン・**ローカル PC へ日次 pull した
コピー**(`backup_run.py --dest` の出力)のどれを `--run-dir` に指しても同じ機構で動く。
GPU 機から `discord.com:443` へ出られない場合は、pull 先を指してローカル PC 側から投稿する
(報告が 1 日遅れになるだけで機構は同一)。

**ラン本体に一切触れない**(R1 ドクトリン ⑥「観測がシムを変えない」)
--------------------------------------------------------------------
* 別プロセス。signal も送らない・watchdog の子にもしない・シム本体には read も write もしない。
* 読むのは `status.json` / `l2_metrics.part-*.parquet` / ファイルの mtime と個数だけ。
  **L1 は 1 バイトも読まない**(最小構成の要。L2 は 1 step 1 行 = 全部読んでも 1,440 行)。
* 書くのは `<run>/_progress/` 配下だけ(`--out-dir` で run-dir の外へも出せる)。
  **run-dir 直下には 1 バイトも書かない**(logger の `_next_seg`・watchdog の `PART_GLOB`・
  backup_run の走査は全て非再帰なので、サブディレクトリは構造的に無害)。
* ★**「読むだけ」でもシムを壊しうる箇所が 1 つだけある**(第77バッチの実測事故): Windows の
  素の `open()` は `FILE_SHARE_DELETE` を立てないため、こちらが part を開いている最中に
  `logger._finalize_stream` の `p.unlink()` が `PermissionError [WinError 32]` で失敗し、
  **ラン本体が finalize で落ちる**。part を開くときは必ず `live_viewer._open_shared` を使う
  (`l1_stream.py` / `detect_regression.py` / `backup_run.py` と同じ借り方)。
* あらゆる例外を握りつぶし `<out>/reporter.log` に 1 行残すだけ。**終了コードは常に 0**。

正確さのための 3 つの掟(見落とすとバグる)
------------------------------------------
1. **L2 は毎回全部読み直し、step で dedupe(後勝ち)する。** カーソルを持たないので、
   resume で part 番号が振り直されても(`logger._next_seg` は既存 part の max+1 から採番)
   **構造的に二重計上が起きない**。240 part × 6 行 = 1,440 行しかないので費用も無い。
2. **書きかけを読まない。** `is_complete_parquet`(先頭 magic + 末尾 magic + footer 長)を
   通った part だけ。未完結の最新 part は待つ。
3. **「確報」と「速報」を分ける。** watchdog の `_restore_from_backup` は run-dir の part を
   消して checkpoint 世代から復元するため、**最新 checkpoint の mtime より後に書かれた part は
   同じ名前で中身が変わりうる**。したがって:
     - 画面・ハートビート・日次ダイジェストの本文 = **速報**(境界外も読む。「暫定」と明記)
     - `digest.json` に残す数字 = **確報**(checkpoint mtime 以前の part のみ)
   同じキーに確報と速報を混ぜない(`digest.json` は `confirmed` / `provisional` の 2 節)。

セキュリティ(webhook URL は認証情報そのもの)
----------------------------------------------
* **環境変数のみ**(既定 `SHIBUYA_DISCORD_WEBHOOK`)。`--webhook-url` のような CLI 引数は
  **意図的に実装しない**(引数はシェル履歴・タスクスケジューラの XML・`ps` に残る)。
* ログ・標準出力・例外文字列に URL(id も token も)を出さない。投稿系の例外は自前の
  メッセージへ詰め替えてから記録する(`_redact` が最後の網)。
* **404 を受けたら以後の投稿を恒久停止**(公式が「404 の webhook を叩き続けると一時制限」と明記)。
* 投稿本文に**絶対パス**を入れない(`status.json` の `run_dir` / `backup_dir` はユーザー名 =
  個人情報を含む)。出すのは**ラン名だけ**。最終ペイロードは `_scrub` が機械的に走査する。
* `allowed_mentions` は常に「何も ping しない」形を明示する(既定は本文中の `@everyone` が解釈される)。
* **エージェントの発話・DM・SNS 本文・内省本文は 1 文字も出さない**(ETHICS.md §2-4。
  `--quotes on` は承認済みの決定により**実装しない**)。`llm_journal` は開かない。

依存
----
標準ライブラリのみ + `pyarrow`(関数内 import・不在でも落ちない)。`requests` は環境に
無いので `urllib.request` + 手書き multipart で完結させる(本選前に依存を増やさない)。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

for _s in (sys.stdout, sys.stderr):              # Windows コンソール(cp932)対策
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_HERE_DIR = str(Path(__file__).resolve().parent)
if _HERE_DIR not in sys.path:
    sys.path.insert(0, _HERE_DIR)

# 非侵襲な読み口の**単一の源**は live_viewer.py。l1_stream / detect_regression と同じ借り方
# (backup_run は「観測層への依存の向き」を避けるため逐語コピーしている。こちらは報告側 =
#  観測層に依存してよい向きなので import する)。
from live_viewer import (                        # noqa: E402
    _open_shared, is_complete_parquet, list_parts, read_run_config,
)

# --------------------------------------------------------------------------- #
# 定数
# --------------------------------------------------------------------------- #
WEBHOOK_ENV_DEFAULT = "SHIBUYA_DISCORD_WEBHOOK"
OUT_SUBDIR = "_progress"
STATE_NAME = "reporter_state.json"
LOG_NAME = "reporter.log"
STATE_VERSION = 1

BATCH_ROWS = 131_072
MINUTES_PER_DAY = 1440

# Discord の上限(docs.discord.com/developers/resources/message)。焼き込むのは**上限だけ**で、
# レート制限の具体値は焼き込まない(公式に per-webhook の値が無い = ヘッダに従う)。
MAX_CONTENT = 2000
MAX_EMBED_DESC = 4096
MAX_EMBED_TITLE = 256
MAX_FIELD_NAME = 256
MAX_FIELD_VALUE = 1024
MAX_FOOTER = 2048
MAX_EMBED_TOTAL = 6000
MAX_FIELDS = 25
# 添付は公式既定 10 MiB だが webhook は 8MB のまま 413 を返す履歴がある(discord-api-docs
# #6058)。**8MB 以下で設計すれば全変種で安全**。
MAX_ATTACH_BYTES = 8 * 1024 * 1024

HTTP_TIMEOUT_SEC = 20.0
MAX_RETRY = 3

COLOR_OK = 0x2E86DE        # 青
COLOR_WARN = 0xF0B429      # 黄
COLOR_BAD = 0xE74C3C       # 赤
COLOR_DONE = 0x27AE60      # 緑

FICTION_NOTE = "※ 本シミュレーションの人物・組織・発言はすべて架空です"

# 日次ダイジェスト「街のようす」。(L2 列, 表示名, 形式, 集計)。**存在する列だけ**出す
# (P7: 読めなかった指標は捏造せず「—」= 測れなかったと出す)。
CITY_METRICS = [
    ("n_moving",             "移動中",         "int",  "mean"),
    ("n_inside_buildings",   "屋内",           "int",  "mean"),
    ("n_sleeping",           "就寝",           "int",  "mean"),
    ("n_working",            "勤務中",         "int",  "mean"),
    ("n_outside",            "範囲外",         "int",  "mean"),
    ("mean_money",           "所持金 平均",    "yen",  "mean"),
    ("opinion_var",          "意見の分散",     "num",  "mean"),
    ("status_gini",          "地位ジニ",       "num",  "mean"),
    ("undefined_action_rate", "未定義行動率",  "rate", "mean"),
    ("echo_utterance_rate",  "エコー(反復)率", "rate", "mean"),
]
SOCIETY_METRICS = [
    ("distinct_vocab_in_use", "使用中の語彙",  "int",  "last"),
    ("total_adoptions",      "語の採用 累計",  "int",  "last"),
    ("n_groups",             "グループ",       "int",  "last"),
    ("n_ventures",           "ベンチャー",     "int",  "last"),
    ("n_proposals",          "議案",           "int",  "last"),
    ("n_institutions",       "制度",           "int",  "last"),
    ("n_events_hosted",      "催し 累計",      "int",  "last"),
    ("n_sns_posts",          "SNS 投稿",       "int",  "mean"),
]
HEALTH_METRICS = [
    ("llm_calls_total",      "LLM 呼数",       "int",  "last"),
    ("llm_fallback_rate",    "fallback 率",    "rate", "mean"),
    ("llm_cache_hit_rate",   "cache ヒット率", "rate", "mean"),
]
ALL_METRICS = CITY_METRICS + SOCIETY_METRICS + HEALTH_METRICS
REPORT_COLUMNS = tuple(k for k, _, _, _ in ALL_METRICS)

# 退行判定の閾値(docs/research/regression-signals.md §3 の実測で確立した値)。
# detect_regression は「既定値をコードに埋め込まない」方針で必須引数にしているので、
# **呼ぶ側が明示する**(値は出力にも記録する)。
REG_ALPHA = 0.05
REG_MIN_REL_SLOPE = 0.02

# 重大度(数値が大きいほど重い)
SEV_INFO, SEV_WARN, SEV_CRIT = 0, 1, 2


# --------------------------------------------------------------------------- #
# 秘匿・小道具
# --------------------------------------------------------------------------- #
_WEBHOOK_RE = re.compile(
    r"https?://(?:\w+\.)?discord(?:app)?\.com/api(?:/v\d+)?/webhooks/\S*",
    re.IGNORECASE)
# 絶対パス(= ユーザー名 = 個人情報)。投稿本文とログの最後の網。
_ABSPATH_RE = re.compile(
    r"(?:[A-Za-z]:[\\/][^\s\"'<>|]*)"                      # C:\... / C:/...
    r"|(?:\\\\[^\s\"'<>|]+)"                               # UNC
    r"|(?:/(?:home|Users|root|mnt|media|var|opt|tmp)/[^\s\"'<>|]*)")


def _redact(text, url: str | None = None) -> str:
    """webhook URL(と絶対パス)を潰した文字列。**ログ・stdout に出す前に必ず通す**。"""
    s = "" if text is None else str(text)
    if url:
        s = s.replace(url, "<webhook>")
        tail = url.rstrip("/").rsplit("/", 2)
        for piece in tail[1:]:                     # id / token を単体で漏らさない
            if len(piece) >= 8:
                s = s.replace(piece, "<redacted>")
    s = _WEBHOOK_RE.sub("<webhook>", s)
    home = os.path.expanduser("~")
    if home and len(home) > 3:
        s = s.replace(home, "<home>")
    return _ABSPATH_RE.sub("<path>", s)


def _scrub(obj, url: str | None = None):
    """投稿ペイロードを再帰的に走査して秘匿を適用する(機械の最終防壁)。"""
    if isinstance(obj, str):
        return _redact(obj, url)
    if isinstance(obj, dict):
        return {k: _scrub(v, url) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_scrub(v, url) for v in obj]
    return obj


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _fmt_tod(minute_of_day) -> str:
    m = int(minute_of_day) % MINUTES_PER_DAY
    return f"{m // 60:02d}:{m % 60:02d}"


def _fmt_dur(sec) -> str:
    if sec is None:
        return "—"
    sec = max(0.0, float(sec))
    if sec < 90:
        return f"{sec:.0f}秒"
    if sec < 5400:
        return f"{sec / 60:.1f}分"
    if sec < 86400 * 2:
        return f"{sec / 3600:.1f}時間"
    return f"{sec / 86400:.1f}日"


def _fmt_val(v, kind: str) -> str:
    """P7: None は「—」= 測れなかった(0 で埋めない)。"""
    if v is None:
        return "—"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    if v != v:                                     # NaN
        return "—"
    if kind == "int":
        return f"{v:,.0f}"
    if kind == "yen":
        return f"¥{v:,.0f}"
    if kind == "rate":
        return f"{v * 100:.1f}%"
    return f"{v:,.4g}"


def _fmt_delta(cur, prev, kind: str) -> str:
    """前日比。**絶対値だけだと人は異常に気づけない**ので必ず添える。"""
    if cur is None or prev is None:
        return ""
    try:
        cur, prev = float(cur), float(prev)
    except (TypeError, ValueError):
        return ""
    if cur != cur or prev != prev:
        return ""
    d = cur - prev
    if kind == "rate":
        return f" ({d * 100:+.1f}pt)"
    if kind == "int":
        return f" ({d:+,.0f})"
    if prev == 0:
        return f" ({d:+,.4g})"
    return f" ({d / abs(prev) * 100:+.1f}%)"


def _bar(frac: float | None, width: int = 20) -> str:
    if frac is None:
        return "░" * width
    f = min(1.0, max(0.0, float(frac)))
    n = int(round(f * width))
    return "█" * n + "░" * (width - n)


def _clip(s: str, n: int) -> str:
    s = "" if s is None else str(s)
    return s if len(s) <= n else s[: max(0, n - 1)] + "…"


def _write_atomic(path: Path, text: str) -> None:
    """tmp → os.replace(同一 FS で原子的)。pid つき tmp = 二重起動でも壊れない。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class Reporter:
    """ログの口。**ここを通らない出力を作らない**(秘匿の単一の関門)。"""

    def __init__(self, out_dir: Path, url: str | None = None, echo: bool = True):
        self.out_dir = Path(out_dir)
        self.url = url
        self.echo = echo

    def log(self, msg: str) -> None:
        line = f"[{_now_iso()}] {_redact(msg, self.url)}"
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            with (self.out_dir / LOG_NAME).open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:                          # ログすら書けなくても続行(P5)
            pass
        if self.echo:
            try:
                print(f"[report] {line}", flush=True)
            except Exception:
                pass


# --------------------------------------------------------------------------- #
# 読み取り(status.json / ファイルシステム / L2)
# --------------------------------------------------------------------------- #
def read_status(run_dir: Path) -> dict | None:
    """watchdog の `status.json`。tmp→os.replace の原子的書き込みなので途中を読むことがない。

    無い / 壊れている場合は None(= 「測れなかった」。捏造しない)。
    """
    path = Path(run_dir) / "status.json"
    try:
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    # ★ 絶対パス(= ユーザー名)は読み込んだ時点で捨てる。投稿経路に載せないための第一関門。
    data.pop("run_dir", None)
    data.pop("backup_dir", None)
    disk = data.get("disk")
    if isinstance(disk, dict):
        disk.pop("path", None)
        for t in (disk.get("targets") or []):
            if isinstance(t, dict):
                t.pop("path", None)
    return data


def scan_filesystem(run_dir: Path) -> dict:
    """part / checkpoint / 空き容量を**stat だけ**で見る(中身は開かない)。"""
    run_dir = Path(run_dir)
    out = {
        "l1_parts": 0, "l2_parts": 0,
        "latest_part_mtime": None, "latest_part_name": None,
        "ckpt_count": 0, "ckpt_step": None, "ckpt_mtime": None, "ckpt_name": None,
        "free_gb": None, "finalized": False, "notes": [],
    }
    try:
        out["finalized"] = (run_dir / "summary.json").is_file()
    except Exception:
        pass
    for stem, key in (("l1_events", "l1_parts"), ("l2_metrics", "l2_parts")):
        try:
            parts = list_parts(run_dir, stem)
        except Exception:
            parts = []
        out[key] = len(parts)
        for _, p in parts:
            try:
                mt = p.stat().st_mtime
            except OSError:
                continue
            if out["latest_part_mtime"] is None or mt > out["latest_part_mtime"]:
                out["latest_part_mtime"] = mt
                out["latest_part_name"] = p.name
    try:
        ck_dir = run_dir / "checkpoint"
        cands = []
        if ck_dir.is_dir():
            for p in ck_dir.glob("ckpt-*.pkl.gz"):
                try:
                    step = int(p.name.split("-")[1].split(".")[0])
                except (IndexError, ValueError):
                    continue
                cands.append((step, p))
        out["ckpt_count"] = len(cands)
        if cands:
            step, p = max(cands, key=lambda t: t[0])
            out["ckpt_step"] = step
            out["ckpt_name"] = p.name
            out["ckpt_mtime"] = p.stat().st_mtime
    except Exception:
        pass
    try:
        import shutil
        out["free_gb"] = round(shutil.disk_usage(str(run_dir)).free / 1e9, 2)
    except Exception:
        out["free_gb"] = None
    return out


def _read_part_rows(path: Path, columns=None) -> list[dict]:
    """完結 part を batch 単位で読む(part 全体を Python 化しない)。開くのは `_open_shared`。

    ★開けたハンドルは相手が unlink しても有効(削除保留)なので、`FileNotFoundError` は
    **開く瞬間**に限られる = そこまでに状態は 1 バイトも動いていない。
    """
    import pyarrow.parquet as pq                   # 任意 import(不在でも呼び出し側が処理)

    rows: list[dict] = []
    with _open_shared(path) as fh:
        pf = pq.ParquetFile(fh)
        names = list(pf.schema_arrow.names)
        if "step" not in names:
            return []
        cols = None
        if columns is not None:
            cols = ["step"] + [c for c in columns if c in names and c != "step"]
        for batch in pf.iter_batches(columns=cols, batch_size=BATCH_ROWS):
            if batch.num_rows:
                rows.extend(batch.to_pylist())
    return rows


def read_l2(run_dir: Path, columns=None, confirm_before: float | None = None) -> dict:
    """完結 L2 part を**全部読み直して step で dedupe(後勝ち)**する。

    カーソルを持たないので resume の part 振り直しで二重計上しない(掟 1)。
    `confirm_before`(= 最新 checkpoint の mtime)以前に書かれた part の step だけを
    「確報」として印を付ける(掟 3)。finalize 済み(canonical あり)なら全部が確報。
    """
    run_dir = Path(run_dir)
    res = {"rows": [], "confirmed_steps": set(), "confirmed_step": None,
           "parts_read": [], "parts_pending": [], "notes": [], "available": True}
    try:
        import pyarrow  # noqa: F401
    except Exception:
        res["available"] = False
        res["notes"].append("pyarrow 不在のため L2 を読めない")
        return res

    by_step: dict[int, dict] = {}
    confirmed: set[int] = set()
    try:
        parts = list_parts(run_dir, "l2_metrics")
    except Exception as exc:
        res["notes"].append(f"part の走査に失敗: {type(exc).__name__}")
        parts = []

    sources: list[tuple[Path, bool]] = []
    for _, p in parts:
        try:
            complete = is_complete_parquet(p)
        except Exception:
            complete = False
        if not complete:
            res["parts_pending"].append(p.name)
            continue
        try:
            mt = p.stat().st_mtime
        except OSError:
            res["parts_pending"].append(p.name)
            continue
        sources.append((p, confirm_before is not None and mt <= confirm_before))
    canonical = run_dir / "l2_metrics.parquet"
    if canonical.is_file():
        try:
            if is_complete_parquet(canonical):
                sources.append((canonical, True))   # finalize 済み = 巻き戻らない
        except Exception:
            pass

    for path, is_conf in sources:
        try:
            rows = _read_part_rows(path, columns)
        except FileNotFoundError:
            res["notes"].append(f"読む直前に消えた: {path.name}")
            continue
        except Exception as exc:
            res["notes"].append(f"{path.name} を読めない: {type(exc).__name__}")
            continue
        res["parts_read"].append(path.name)
        for r in rows:
            st = r.get("step")
            if st is None:
                continue
            st = int(st)
            by_step[st] = r                        # ★後勝ち = resume 後の値を正とする
            if is_conf:
                confirmed.add(st)
            else:
                confirmed.discard(st)              # 巻き戻り得る値へ差し替わった

    res["rows"] = [by_step[s] for s in sorted(by_step)]
    res["confirmed_steps"] = confirmed
    res["confirmed_step"] = max(confirmed) if confirmed else None
    return res


# --------------------------------------------------------------------------- #
# 日境界・集計
# --------------------------------------------------------------------------- #
def day_of(step: int, start_min: int, dt_min: int) -> int:
    """day index。`analyze_structure` / `make_viewer --daily-rollup` と**同定義**(0 始まり)。"""
    return (int(start_min) + int(step) * int(dt_min)) // MINUTES_PER_DAY


def aggregate_day(rows: list[dict], day: int, start_min: int, dt_min: int) -> dict:
    """その day の行から metric ごとに mean / last / n を作る。値が無い列は None。"""
    sel = [r for r in rows if day_of(r.get("step", 0), start_min, dt_min) == day]
    out = {"day": day, "n_rows": len(sel),
           "step_min": None, "step_max": None, "metrics": {}}
    if not sel:
        return out
    steps = [int(r["step"]) for r in sel if r.get("step") is not None]
    if steps:
        out["step_min"], out["step_max"] = min(steps), max(steps)
    for key, _label, _kind, _agg in ALL_METRICS:
        vals = []
        last = None
        for r in sel:
            v = r.get(key)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            if v != v:                             # NaN
                continue
            vals.append(float(v))
            last = float(v)
        if not vals:
            continue
        out["metrics"][key] = {"mean": sum(vals) / len(vals), "last": last,
                               "n": len(vals)}
    return out


def _pick(agg: dict, key: str, how: str):
    m = (agg.get("metrics") or {}).get(key)
    return None if m is None else m.get(how)


def completed_days(rows: list[dict], start_min: int, dt_min: int) -> list[int]:
    """**終わった**シミュ日の一覧。最後に観測した day は「まだ途中」なので含めない。"""
    days = sorted({day_of(r["step"], start_min, dt_min)
                   for r in rows if r.get("step") is not None})
    return days[:-1] if len(days) >= 2 else []


# --------------------------------------------------------------------------- #
# スナップショット(1 サイクルで見る全て)
# --------------------------------------------------------------------------- #
def build_snapshot(cfg, prev_state: dict | None = None) -> dict:
    """status + fs + L2 を 1 つの小さな dict へ。**ここから先はファイルを触らない**。"""
    run_dir = cfg.run_dir
    conf = {}
    try:
        conf = read_run_config(run_dir)
    except Exception:
        conf = {}
    dt_min = int(conf.get("dt_min") or 10)
    start_min = conf.get("start_min")
    start_min = 7 * 60 if start_min is None else int(start_min)
    n_steps = conf.get("n_steps")

    status = read_status(run_dir)
    fs = scan_filesystem(run_dir)
    l2 = read_l2(run_dir, columns=REPORT_COLUMNS, confirm_before=fs.get("ckpt_mtime"))

    rows = l2["rows"]
    last_step = int(rows[-1]["step"]) if rows else None
    if last_step is None and status:
        lp = status.get("last_progress") or {}
        last_step = lp.get("step")
    days = sorted({day_of(r["step"], start_min, dt_min)
                   for r in rows if r.get("step") is not None})
    last_day = day_of(int(n_steps) - 1, start_min, dt_min) if n_steps else None

    now = time.time()
    snap = {
        "now": now,
        "run_name": run_dir.name,                  # ★投稿に出すのはラン名だけ
        "dt_min": dt_min, "start_min": start_min, "n_steps": n_steps,
        "n_agents": conf.get("n_agents"),
        "last_step": last_step,
        "sim_min": (start_min + last_step * dt_min) if last_step is not None else None,
        "cur_day": days[-1] if days else None,
        "last_day": last_day,
        "days_seen": days,
        "completed_days": completed_days(rows, start_min, dt_min),
        "state": (status or {}).get("state"),
        "restarts": (status or {}).get("restarts"),
        "max_restarts": (status or {}).get("max_restarts"),
        "status_updated": (status or {}).get("updated"),
        "llm_health": (status or {}).get("llm_health"),
        "disk": (status or {}).get("disk"),
        "last_backup_step": (status or {}).get("last_backup_step"),
        "has_status": status is not None,
        "fs": fs,
        "l2_rows": rows,
        "l2_notes": l2["notes"],
        "l2_available": l2["available"],
        "confirmed_step": l2["confirmed_step"],
        "parts_pending": l2["parts_pending"],
        "regression": None,
    }
    snap["progress"] = (last_step / n_steps) if (last_step is not None and n_steps) else None
    lag = None
    if fs.get("latest_part_mtime"):
        lag = now - float(fs["latest_part_mtime"])
    if fs.get("ckpt_mtime"):
        lag = min(lag, now - float(fs["ckpt_mtime"])) if lag is not None \
            else now - float(fs["ckpt_mtime"])
    snap["lag_sec"] = lag
    snap["sec_per_step"] = _rate_from_history(prev_state, last_step, now)
    return snap


def _rate_from_history(prev_state: dict | None, step, now: float):
    """進捗履歴(state に残す)から 1 step の壁時計秒を推定する。足りなければ None。"""
    if not prev_state or step is None:
        return None
    hist = prev_state.get("progress_history") or []
    pts = [h for h in hist if isinstance(h, list) and len(h) == 2 and h[1] is not None]
    if not pts:
        return None
    t0, s0 = float(pts[0][0]), int(pts[0][1])
    if step <= s0 or now <= t0:
        return None
    return (now - t0) / (step - s0)


# --------------------------------------------------------------------------- #
# アラート(症状ベース・遷移でのみ発火)
# --------------------------------------------------------------------------- #
def _alert_slot(state: dict, key: str) -> dict:
    return state.setdefault("alerts", {}).setdefault(
        key, {"level": None, "cand": None, "cand_n": 0, "last_post": 0.0})


def _transition(state: dict, key: str, level, need: int = 1) -> tuple[bool, object]:
    """level が変わったか。`need` サイクル連続で同じ候補を見てから確定する(スパイク弾き)。"""
    slot = _alert_slot(state, key)
    prev = slot.get("level")
    if level == prev:
        slot["cand"], slot["cand_n"] = None, 0
        return False, prev
    if slot.get("cand") == level:
        slot["cand_n"] = int(slot.get("cand_n") or 0) + 1
    else:
        slot["cand"], slot["cand_n"] = level, 1
    if slot["cand_n"] < max(1, need):
        return False, prev
    slot["level"] = level
    slot["cand"], slot["cand_n"] = None, 0
    return True, prev


def evaluate_alerts(snap: dict, state: dict, cfg) -> list[dict]:
    """§7.1 の A1-A6。**状態が遷移したときだけ**候補を返す(抑制は post 側)。"""
    out: list[dict] = []

    def emit(key, level, prev, sev, title, body, action=""):
        out.append({"key": key, "level": level, "prev": prev, "severity": sev,
                    "title": title, "body": body, "action": action})

    # A1 ラン状態
    st = snap.get("state")
    if snap.get("has_status"):
        changed, prev = _transition(state, "state", st)
        if changed and not (prev is None and st in (None, "running")):
            sev = {"failed": SEV_CRIT, "restarting": SEV_WARN,
                   "done": SEV_INFO, "running": SEV_INFO}.get(st, SEV_WARN)
            act = {"failed": "即対応(ログと最新 checkpoint を確認 → 再開判断)",
                   "restarting": "2 回連続なら原因調査(同じ step で落ちていないか)",
                   "done": "finalize の完了と成果物の転送を確認",
                   "running": "復帰(対応不要)"}.get(st, "status.json を確認")
            emit("state", st, prev, sev, f"ラン状態: {prev} → {st}",
                 f"再起動 {snap.get('restarts')} / {snap.get('max_restarts')} 回", act)

    # A2 進捗停止(watchdog の --stall-min 既定 20 分より**後**に鳴らす = 二重に騒がない)
    lag = snap.get("lag_sec")
    if lag is not None and not snap["fs"].get("finalized") and st != "done":
        level = "stalled" if lag > cfg.stall_min * 60 else "ok"
        changed, prev = _transition(state, "stall", level, need=2)
        if changed and not (prev is None and level == "ok"):
            if level == "stalled":
                emit("stall", level, prev, SEV_CRIT, "進捗が止まっている",
                     f"最新 part / checkpoint が {_fmt_dur(lag)} 更新されていない"
                     f"(閾値 {cfg.stall_min} 分)",
                     "watchdog のログとプロセス生存を確認(watchdog が先に再起動を試みる)")
            else:
                emit("stall", level, prev, SEV_INFO, "進捗が再開した",
                     f"最新の更新から {_fmt_dur(lag)}", "対応不要")

    # A3 ディスク
    disk = snap.get("disk") or {}
    dstate = disk.get("state")
    if dstate:
        changed, prev = _transition(state, "disk", dstate)
        if changed and not (prev is None and dstate == "ok"):
            sev = {"critical": SEV_CRIT, "warn": SEV_WARN}.get(dstate, SEV_INFO)
            emit("disk", dstate, prev, sev, f"ディスク: {prev} → {dstate}",
                 f"空き {disk.get('free_gb')} GB "
                 f"(warn<{disk.get('warn_gb')} / crit<{disk.get('crit_gb')})",
                 "★checkpoint / dormant は剪定禁止(ops/finals-compute-checklist.md E0)。"
                 "落とす順序は indoor_tracks → llm_journal → 人間の判断")

    # A4 LLM fallback(ヒステリシス: 上抜け 0.20 / 下抜け 0.15)
    rate = ((snap.get("llm_health") or {}).get("llm_fallback_rate"))
    if rate is not None:
        cur = _alert_slot(state, "fallback").get("level")
        if float(rate) > cfg.fallback_warn:
            level = "high"
        elif float(rate) < cfg.fallback_clear:
            level = "ok"
        else:
            level = cur if cur is not None else "ok"
        changed, prev = _transition(state, "fallback", level, need=2)
        if changed and not (prev is None and level == "ok"):
            sev = SEV_WARN if level == "high" else SEV_INFO
            emit("fallback", level, prev, sev,
                 f"LLM fallback 率: {prev} → {level}",
                 f"{float(rate) * 100:.1f}%(発火 {cfg.fallback_warn * 100:.0f}% / "
                 f"復帰 {cfg.fallback_clear * 100:.0f}%)",
                 "モデル / プロンプト / バックエンドの点検(ランは止めない)")

    # A5 退行判定
    reg = snap.get("regression") or {}
    verdict = reg.get("verdict")
    if verdict:
        changed, prev = _transition(state, "regression", verdict)
        if changed and not (prev is None and verdict != "REGRESSION"):
            sev = SEV_WARN if verdict == "REGRESSION" else SEV_INFO
            emit("regression", verdict, prev, sev,
                 f"退行判定: {prev} → {verdict}",
                 f"flagged={reg.get('flagged')} n_rows={reg.get('n_rows')}",
                 "事前登録の判断へ(**止めない**)")

    # A6 再起動回数
    rs, mx = snap.get("restarts"), snap.get("max_restarts")
    if isinstance(rs, int) and isinstance(mx, int) and mx > 0:
        frac = rs / mx
        band = "80%" if frac >= 0.8 else ("50%" if frac >= 0.5 else "ok")
        changed, prev = _transition(state, "restarts", band)
        if changed and not (prev is None and band == "ok"):
            sev = SEV_CRIT if band == "80%" else (SEV_WARN if band == "50%" else SEV_INFO)
            emit("restarts", band, prev, sev, f"再起動回数が {band} 帯へ",
                 f"{rs} / {mx} 回",
                 "上限に達すると watchdog は諦める。原因の特定を優先")
    return out


# --------------------------------------------------------------------------- #
# embed の組み立て
# --------------------------------------------------------------------------- #
def _clamp_embed(embed: dict) -> dict:
    """Discord の上限へ機械的に収める(title 256 / desc 4096 / field 1024 / 合計 6000)。"""
    e = dict(embed)
    if "title" in e:
        e["title"] = _clip(e["title"], MAX_EMBED_TITLE)
    if "description" in e:
        e["description"] = _clip(e["description"], MAX_EMBED_DESC)
    fields = []
    for f in (e.get("fields") or [])[:MAX_FIELDS]:
        fields.append({"name": _clip(f.get("name", "—"), MAX_FIELD_NAME),
                       "value": _clip(f.get("value", "—") or "—", MAX_FIELD_VALUE),
                       "inline": bool(f.get("inline", False))})
    if fields:
        e["fields"] = fields
    elif "fields" in e:
        e.pop("fields")
    if isinstance(e.get("footer"), dict):
        e["footer"] = {"text": _clip(e["footer"].get("text", ""), MAX_FOOTER)}

    def _total(x):
        n = len(x.get("title", "")) + len(x.get("description", ""))
        n += len((x.get("footer") or {}).get("text", ""))
        for f in x.get("fields") or []:
            n += len(f["name"]) + len(f["value"])
        return n

    while _total(e) > MAX_EMBED_TOTAL and e.get("fields"):
        e["fields"] = e["fields"][:-1]
        if not e["fields"]:
            e.pop("fields")
    if _total(e) > MAX_EMBED_TOTAL and e.get("description"):
        over = _total(e) - MAX_EMBED_TOTAL
        e["description"] = _clip(e["description"], max(0, len(e["description"]) - over))
    return e


def _progress_line(snap: dict) -> str:
    step, n = snap.get("last_step"), snap.get("n_steps")
    frac = snap.get("progress")
    head = f"`{_bar(frac)}` {step if step is not None else '—'} / {n or '?'} step"
    if frac is not None:
        head += f"({frac * 100:.1f}%)"
    sm = snap.get("sim_min")
    if sm is not None:
        head += f"\nシム内 day {sm // MINUTES_PER_DAY} {_fmt_tod(sm)}"
        if snap.get("last_day") is not None:
            head += f"(最終 day {snap['last_day']})"
    return head


def _eta_line(snap: dict) -> str:
    sps, step, n = snap.get("sec_per_step"), snap.get("last_step"), snap.get("n_steps")
    if not sps or step is None or not n or step >= n:
        return "残り時間 —(壁時計の実測が溜まるまで出さない)"
    remain = (n - step) * sps
    dt, start = snap["dt_min"], snap["start_min"]
    cur_day = day_of(step, start, dt)
    next_day_step = ((cur_day + 1) * MINUTES_PER_DAY - start + dt - 1) // dt
    to_day = max(0, (next_day_step - step)) * sps
    return (f"1 step ≈ {_fmt_dur(sps)} / 次の日次まで ≈ {_fmt_dur(to_day)} / "
            f"完了まで ≈ {_fmt_dur(remain)}")


def _severity_color(snap: dict) -> int:
    st = snap.get("state")
    if st == "failed":
        return COLOR_BAD
    if st == "done":
        return COLOR_DONE
    disk = (snap.get("disk") or {}).get("state")
    if disk == "critical":
        return COLOR_BAD
    if st == "restarting" or disk == "warn":
        return COLOR_WARN
    lag = snap.get("lag_sec")
    if lag is not None and lag > 45 * 60 and not snap["fs"].get("finalized"):
        return COLOR_WARN
    return COLOR_OK


def _footer_text(snap: dict, extra: str = "") -> str:
    conf = snap.get("confirmed_step")
    bits = [FICTION_NOTE,
            f"数字は暫定(速報)/ 確報は step {conf if conf is not None else '—'} まで"]
    if extra:
        bits.append(extra)
    return " / ".join(bits)


def build_heartbeat(snap: dict, state: dict) -> dict:
    """1 通を編集し続ける生存確認。チャンネルを汚さない。"""
    fs = snap["fs"]
    disk_free = (snap.get("disk") or {}).get("free_gb")
    if disk_free is None:
        disk_free = fs.get("free_gb")
    lines = [
        f"状態 **{snap.get('state') or '不明(status.json なし)'}**"
        f"・再起動 {snap.get('restarts') if snap.get('restarts') is not None else '—'}"
        f" / {snap.get('max_restarts') if snap.get('max_restarts') is not None else '—'} 回",
        f"最終更新 {_fmt_dur(snap.get('lag_sec'))}前"
        f"(part L1 {fs['l1_parts']} / L2 {fs['l2_parts']}本"
        f"・checkpoint {fs['ckpt_count']}世代 step {fs.get('ckpt_step') or '—'})",
        f"空き容量 {disk_free if disk_free is not None else '—'} GB",
        _eta_line(snap),
    ]
    if snap.get("parts_pending"):
        lines.append(f"書きかけ part {len(snap['parts_pending'])} 本は読んでいない(待機中)")
    if not snap.get("l2_available"):
        lines.append("L2 を読めない: " + "・".join(snap.get("l2_notes") or ["理由不明"]))
    embed = {
        "title": f"渋谷シム 進捗 — {snap['run_name']}",
        "description": _progress_line(snap) + "\n" + "\n".join(lines),
        "color": _severity_color(snap),
        "footer": {"text": _footer_text(snap, f"更新 {_now_iso()}")},
    }
    return {"embeds": [_clamp_embed(embed)], "allowed_mentions": {"parse": []}}


def _metric_block(rows_def, agg: dict, prev: dict | None) -> str:
    out = []
    for key, label, kind, how in rows_def:
        cur = _pick(agg, key, how)
        if cur is None:
            continue
        pv = _pick(prev, key, how) if prev else None
        out.append(f"{label} **{_fmt_val(cur, kind)}**{_fmt_delta(cur, pv, kind)}")
    return "\n".join(out) if out else "—(この日の L2 に該当列が無い)"


def build_daily_digest(snap: dict, day: int, state: dict,
                       attach_name: str | None = None) -> dict:
    """シミュ日の境界で 1 通。**前日比を必ず添える**(絶対値だけでは異常に気づけない)。"""
    rows = snap["l2_rows"]
    dt, start = snap["dt_min"], snap["start_min"]
    agg = aggregate_day(rows, day, start, dt)
    prev = aggregate_day(rows, day - 1, start, dt) if day > 0 else None
    if prev is not None and prev.get("n_rows", 0) == 0:
        prev = None

    desc = [f"シミュ **day {day}** が終わりました"
            f"(day は 0 始まり・このランの最終 day は "
            f"{snap.get('last_day') if snap.get('last_day') is not None else '?'})",
            "── 投稿時点の進捗 ──",
            _progress_line(snap),
            f"状態 **{snap.get('state') or '不明'}**"
            f"・再起動 {snap.get('restarts') if snap.get('restarts') is not None else '—'} 回"
            f"・最終 checkpoint step {snap['fs'].get('ckpt_step') or '—'}"]
    if agg["n_rows"]:
        desc.append(f"この日の L2 行数 {agg['n_rows']}(step {agg['step_min']}–{agg['step_max']})")
    else:
        desc.append("この日の L2 行が 1 つも読めなかった(= 測れなかった)")

    fields = [
        {"name": "街のようす(その日の平均・前日比)",
         "value": _metric_block(CITY_METRICS, agg, prev)},
        {"name": "社会(その日の最終値・前日比)",
         "value": _metric_block(SOCIETY_METRICS, agg, prev)},
    ]
    health = _metric_block(HEALTH_METRICS, agg, prev)
    dsk = snap.get("disk") or {}
    health += (f"\nディスク {dsk.get('free_gb', snap['fs'].get('free_gb'))} GB"
               f"({dsk.get('state') or 'unknown'})"
               f"・バックアップ確定 step {snap.get('last_backup_step') or '—'}")
    reg = snap.get("regression")
    if reg:
        health += f"\n退行判定 {reg.get('verdict')}(flagged={reg.get('flagged')})"
    fields.append({"name": "装置の健康", "value": health})

    suppressed = int(state.get("suppressed") or 0)
    if suppressed:
        fields.append({"name": "抑制したアラート",
                       "value": f"毎時上限 / クールダウンで **{suppressed} 件**を送らなかった"
                                "(silent cap 禁止のため件数だけ出す)"})
    if snap.get("l2_notes"):
        fields.append({"name": "読み取りの注記",
                       "value": "・".join(snap["l2_notes"][:8])})
    if attach_name:
        fields.append({"name": "添付", "value": f"{attach_name}(日次ロールアップ・自己完結)"})

    embed = {
        "title": f"渋谷シム — day {day} 終了({snap['run_name']})",
        "description": "\n".join(desc),
        "color": _severity_color(snap),
        "fields": fields,
        "footer": {"text": _footer_text(snap)},
    }
    return {"embeds": [_clamp_embed(embed)], "allowed_mentions": {"parse": []}}


def build_alert(alert: dict, snap: dict) -> dict:
    color = {SEV_CRIT: COLOR_BAD, SEV_WARN: COLOR_WARN}.get(alert["severity"], COLOR_OK)
    mark = {SEV_CRIT: "🔴", SEV_WARN: "🟡"}.get(alert["severity"], "🔵")
    sm = snap.get("sim_min")
    desc = [alert["body"],
            f"step {snap.get('last_step') if snap.get('last_step') is not None else '—'}"
            f" / {snap.get('n_steps') or '?'}"
            f"・シム内 day {sm // MINUTES_PER_DAY if sm is not None else '—'}"
            f" {_fmt_tod(sm) if sm is not None else ''}"]
    if alert.get("action"):
        desc.append(f"**推奨アクション**: {alert['action']}")
    embed = {
        "title": f"{mark} {snap['run_name']}: {alert['title']}",
        "description": "\n".join(desc),
        "color": color,
        "footer": {"text": _footer_text(snap)},
    }
    return {"embeds": [_clamp_embed(embed)], "allowed_mentions": {"parse": []}}


# --------------------------------------------------------------------------- #
# Discord(stdlib のみ。手書き multipart)
# --------------------------------------------------------------------------- #
class PostError(Exception):
    """**URL を含まない**投稿失敗(例外の repr に URL が載る事故を構造的に防ぐ)。"""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def _safe_filename(name: str) -> str:
    return re.sub(r'[^\w.\-]+', "_", str(name))[:80] or "file.bin"


def build_multipart(payload: dict, files: list[tuple[str, bytes]],
                    boundary: str | None = None) -> tuple[str, bytes]:
    """`payload_json` + `files[N]` の multipart/form-data を手で組む(stdlib に encoder が無い)。

    embed からは `attachment://<name>` で参照できる。境界は本体に現れないことを機械で確認する。
    """
    body_parts: list[bytes] = []
    blob = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    for _ in range(8):
        b = boundary or ("shibuya" + uuid.uuid4().hex)
        marker = ("--" + b).encode("utf-8")
        if marker not in blob and not any(marker in data for _, data in files):
            boundary = b
            break
        boundary = None
    if boundary is None:                           # 事実上ありえないが黙って壊さない
        raise PostError("multipart の境界を決められなかった")
    sep = f"--{boundary}\r\n".encode("utf-8")

    body_parts.append(sep)
    body_parts.append(b'Content-Disposition: form-data; name="payload_json"\r\n')
    body_parts.append(b"Content-Type: application/json\r\n\r\n")
    body_parts.append(blob)
    body_parts.append(b"\r\n")
    for i, (fname, data) in enumerate(files):
        safe = _safe_filename(fname)
        body_parts.append(sep)
        body_parts.append(
            f'Content-Disposition: form-data; name="files[{i}]"; filename="{safe}"\r\n'
            .encode("utf-8"))
        body_parts.append(b"Content-Type: application/octet-stream\r\n\r\n")
        body_parts.append(data)
        body_parts.append(b"\r\n")
    body_parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    return f"multipart/form-data; boundary={boundary}", b"".join(body_parts)


def _retry_after_sec(headers, body_text: str) -> float:
    """429 の待ち時間。**API v8+ の `retry_after` は秒**(v6 のミリ秒と混同しない)。"""
    for key in ("Retry-After", "retry-after"):
        try:
            v = headers.get(key) if headers is not None else None
        except Exception:
            v = None
        if v is not None:
            try:
                return max(0.0, float(v))
            except (TypeError, ValueError):
                pass
    try:
        j = json.loads(body_text or "{}")
        if isinstance(j, dict) and j.get("retry_after") is not None:
            return max(0.0, float(j["retry_after"]))
    except Exception:
        pass
    return 1.0


class DiscordSink:
    """投稿の口。`--dry-run` では 1 バイトも送らずローカルへ書く。"""

    def __init__(self, url: str | None, reporter: Reporter, dry_dir: Path | None = None,
                 sleep=time.sleep, opener=None):
        self.url = url
        self.rep = reporter
        self.dry_dir = dry_dir
        self.sleep = sleep
        self.opener = opener                       # テストが差し替える HTTP 実行器
        self.disabled = False                      # 404 後は恒久停止
        self.sent = 0
        self.dry_files: list[Path] = []

    # ---------------------------------------------------------------- 低レベル
    def _http(self, method: str, url: str, body: bytes, content_type: str):
        """(status, headers, text)。**例外は必ず PostError へ詰め替える**(URL 秘匿)。"""
        if self.opener is not None:
            return self.opener(method, url, body, content_type)
        import urllib.error
        import urllib.request
        req = urllib.request.Request(url, data=body, method=method)
        req.add_header("Content-Type", content_type)
        req.add_header("User-Agent",
                       "shibuya-report-progress/1.0 (+sidecar; read-only)")
        try:
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SEC) as resp:
                return resp.status, resp.headers, resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:      # 4xx/5xx はここ
            try:
                text = exc.read().decode("utf-8", "replace")
            except Exception:
                text = ""
            return exc.code, exc.headers, text
        except Exception as exc:                   # URLError / timeout / socket
            raise PostError(f"送信できない({type(exc).__name__})") from None

    def _send(self, method: str, url: str, payload: dict,
              files: list[tuple[str, bytes]] | None) -> dict | None:
        """429(Retry-After 秒)/ 5xx(指数バックオフ)を最大 3 回。404 は恒久停止。"""
        if files:
            ctype, body = build_multipart(payload, files)
        else:
            ctype = "application/json"
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        delay = 1.0
        for attempt in range(MAX_RETRY):
            status, headers, text = self._http(method, url, body, ctype)
            if 200 <= status < 300:
                self.sent += 1
                try:
                    return json.loads(text) if text.strip() else {}
                except Exception:
                    return {}
            if status == 404:
                raise PostError("404: webhook が存在しない(以後の投稿を恒久停止する)", 404)
            if status == 429:
                wait = _retry_after_sec(headers, text)
                self.rep.log(f"429 レート制限: {wait:.2f} 秒待って再試行"
                             f"({attempt + 1}/{MAX_RETRY})")
                self.sleep(wait)
                continue
            if 500 <= status < 600:
                self.rep.log(f"{status} サーバ側エラー: {delay:.1f} 秒後に再試行"
                             f"({attempt + 1}/{MAX_RETRY})")
                self.sleep(delay)
                delay *= 2
                continue
            raise PostError(f"HTTP {status}(本文の先頭: "
                            f"{_clip(text, 120)})", status)
        raise PostError(f"再試行 {MAX_RETRY} 回でも送れなかった")

    # ---------------------------------------------------------------- 高レベル
    def _dry_write(self, kind: str, payload: dict,
                   files: list[tuple[str, bytes]] | None) -> dict:
        rec = {"kind": kind, "written": _now_iso(), "payload": payload,
               "files": [{"filename": _safe_filename(n), "bytes": len(d)}
                         for n, d in (files or [])]}
        if self.dry_dir is not None:
            self.dry_dir.mkdir(parents=True, exist_ok=True)
            n = len(list(self.dry_dir.glob("*.json")))
            path = self.dry_dir / f"{n:04d}-{_safe_filename(kind)}.json"
            _write_atomic(path, json.dumps(rec, ensure_ascii=False, indent=2))
            self.dry_files.append(path)
        try:
            print(f"--- [dry-run] {kind} ---")
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            if files:
                print(f"    添付: {[ (n, len(d)) for n, d in files ]}")
        except Exception:
            pass
        return {"id": f"dry-{kind}"}

    def post(self, kind: str, payload: dict, files=None, wait: bool = False) -> dict | None:
        """新規メッセージ。`wait=True` で `?wait=true`(message.id を得るため)。"""
        payload = _scrub(payload, self.url)
        if self.dry_dir is not None or not self.url:
            return self._dry_write(kind, payload, files)
        if self.disabled:
            return None
        url = self.url + ("?wait=true" if wait else "")
        try:
            return self._send("POST", url, payload, files)
        except PostError as exc:
            if exc.status == 404:
                self.disabled = True
            self.rep.log(f"投稿失敗({kind}): {exc}")
            return None

    def patch(self, kind: str, message_id: str, payload: dict, files=None) -> bool:
        """ハートビートの編集。message が消えていれば False(呼び出し側が立て直す)。"""
        payload = _scrub(payload, self.url)
        if self.dry_dir is not None or not self.url:
            self._dry_write(f"{kind}-edit", payload, files)
            return True
        if self.disabled:
            return False
        url = f"{self.url}/messages/{message_id}"
        try:
            self._send("PATCH", url, payload, files)
            return True
        except PostError as exc:
            if exc.status == 404:                  # ★ message が消えただけ(webhook は生きている)
                self.rep.log("ハートビートのメッセージが見つからない → 立て直す")
                return False
            self.rep.log(f"編集失敗({kind}): {exc}")
            return False


# --------------------------------------------------------------------------- #
# 状態(原子的保存)
# --------------------------------------------------------------------------- #
def load_state(out_dir: Path) -> dict:
    path = Path(out_dir) / STATE_NAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and int(data.get("version") or 0) == STATE_VERSION:
            data.setdefault("alerts", {})
            data.setdefault("posted_days", [])
            data.setdefault("post_times", [])
            data.setdefault("progress_history", [])
            return data
    except Exception:
        pass
    return {"version": STATE_VERSION, "alerts": {}, "posted_days": [],
            "post_times": [], "progress_history": [], "suppressed": 0,
            "heartbeat_id": None, "last_post_ts": 0.0, "cycles": 0,
            "disabled": False}


def save_state(out_dir: Path, state: dict) -> None:
    try:
        _write_atomic(Path(out_dir) / STATE_NAME,
                      json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True))
    except Exception:
        pass                                       # 状態が残せなくてもランには影響しない


def _cap_ok(state: dict, now: float, cfg) -> bool:
    """毎時上限は**アラートにだけ**掛ける(日次ダイジェスト = 全 10 通 と ハートビート =
    1 通の編集 は主たる成果物なので数えない)。超過分は握り潰さず件数を日次に出す。"""
    times = [t for t in (state.get("post_times") or []) if now - float(t) < 3600.0]
    state["post_times"] = times
    return len(times) < max(1, cfg.max_posts_per_hour)


def _note_post(state: dict, now: float, is_alert: bool = False) -> None:
    if is_alert:
        state.setdefault("post_times", []).append(now)
    state["last_post_ts"] = now                    # A7(投稿の途絶)は種類を問わず見る


# --------------------------------------------------------------------------- #
# 取り出し(レーン 2)
# --------------------------------------------------------------------------- #
def _write_shims(day_dir: Path, start_min: int, dt_min: int) -> list[str]:
    """`make_viewer --daily-rollup` が真の start_min / Δt を復元するための最小の足場。

    ★なぜ要るか: `build_rollup_data` は Δt を `run_dt`(config.yaml → run_manifest.json →
    正準 10)から、`start_min` を `l1_events.parquet` の**先頭 row group**から復元する。
    抽出ディレクトリにはどちらも無いので、放っておくと **黙って 07:00 / Δt=10 を仮定**する。
    `--daily-rollup` の経路は `--start-tod` を見ない(main が rollup 分岐へ値を渡していない)
    ので、**viz/make_viewer.py を 1 バイトも触らずに真値を渡す唯一の手段**が、この 2 つの
    最小ファイルを置くこと。合成物であることは `_SHIMS.txt` と digest.json の provenance に
    明記する(黙って 07:00 にしない = 計画 §6.2 の要求)。
    """
    made: list[str] = []
    try:
        _write_atomic(day_dir / "config.yaml",
                      f"# report_progress.py が置いた最小の足場(合成物)。\n"
                      f"# 真のラン設定は run-dir 側の config.yaml。\n"
                      f"run:\n  dt_min: {int(dt_min)}\n")
        made.append("config.yaml")
    except Exception:
        pass
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        tbl = pa.table({"step": pa.array([0], pa.int32()),
                        "sim_min": pa.array([int(start_min)], pa.int32())})
        pq.write_table(tbl, day_dir / "l1_events.parquet")
        made.append("l1_events.parquet")
    except Exception:
        pass
    if made:
        try:
            _write_atomic(day_dir / "_SHIMS.txt",
                          "この dir の以下は report_progress.py が置いた**合成物**であり、\n"
                          "ランの一次データではない(make_viewer --daily-rollup に\n"
                          "start_min / dt_min の真値を渡すためだけの足場):\n  - "
                          + "\n  - ".join(made)
                          + f"\n\nstart_min={start_min} dt_min={dt_min}\n"
                            "l1_events.parquet は step/sim_min の 1 行だけで、L1 の中身は\n"
                            "1 バイトも含まない(レポーターは L1 を読まない)。\n")
        except Exception:
            pass
    return made


def extract_day(snap: dict, day: int, out_dir: Path, rep: Reporter,
                make_rollup: bool = True, timeout: float = 300.0) -> dict:
    """`<out>/day-NN/` に L2 結合 + digest.json + summary.json(最小) + rollup.html。

    **確報と速報を混ぜない**: digest.json は `confirmed`(checkpoint mtime 以前の part 由来)と
    `provisional`(境界外も含む)の 2 節に分ける。
    """
    day_dir = Path(out_dir) / f"day-{day:02d}"
    day_dir.mkdir(parents=True, exist_ok=True)
    res = {"dir": day_dir, "rollup": None, "l2_rows": 0, "notes": []}
    dt, start = snap["dt_min"], snap["start_min"]
    run_dir = snap["_run_dir"]

    # 1) 全列の L2 を読み直して結合(報告用スナップショットは列を絞ってあるため)
    full = read_l2(run_dir, columns=None, confirm_before=snap["fs"].get("ckpt_mtime"))
    rows = [r for r in full["rows"]
            if r.get("step") is not None and day_of(r["step"], start, dt) <= day]
    res["l2_rows"] = len(rows)
    if rows:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            keys = sorted({k for r in rows for k in r})
            tbl = pa.table({k: [r.get(k) for r in rows] for k in keys})
            pq.write_table(tbl, day_dir / "l2_metrics.parquet", compression="zstd")
        except Exception as exc:
            res["notes"].append(f"l2_metrics.parquet を書けない: {type(exc).__name__}")
    else:
        res["notes"].append("この day までの L2 行が 1 つも無い")

    # 2) ビューアが読む最小の summary(本物は finalize でしか出ない)
    try:
        _write_atomic(day_dir / "summary.json", json.dumps(
            {"n_agents": snap.get("n_agents"), "n_steps": snap.get("n_steps"),
             "_note": "report_progress.py が置いた最小版(本物は finalize 後の run-dir 側)。"
                      "値の出所は run-dir の config.yaml(run.n_agents / run.n_steps)"},
            ensure_ascii=False, indent=2))
    except Exception:
        pass

    shims = _write_shims(day_dir, start, dt)

    # 3) digest.json(確報 / 速報を分けて残す)
    conf_steps = full["confirmed_steps"]
    conf_rows = [r for r in rows if int(r["step"]) in conf_steps]
    digest = {
        "schema": 1,
        "run_name": snap["run_name"],
        "day": day,
        "generated": _now_iso(),
        "provenance": {
            "start_min": start, "dt_min": dt, "n_steps": snap.get("n_steps"),
            "parts_read": full["parts_read"],
            "parts_pending": full["parts_pending"],
            "confirmed_step": full["confirmed_step"],
            "checkpoint_step": snap["fs"].get("ckpt_step"),
            "checkpoint_name": snap["fs"].get("ckpt_name"),
            "l2_rows_total": len(rows),
            "shims": shims,
            "notes": full["notes"] + res["notes"],
        },
        "confirmed": {
            "definition": "最新 checkpoint の mtime 以前に書かれた part 由来の step のみ"
                          "(watchdog の巻き戻しで中身が変わらない範囲)",
            "day": aggregate_day(conf_rows, day, start, dt),
            "prev_day": aggregate_day(conf_rows, day - 1, start, dt) if day > 0 else None,
        },
        "provisional": {
            "definition": "checkpoint 境界より先の part も含む速報値(巻き戻しで変わりうる)",
            "day": aggregate_day(rows, day, start, dt),
            "prev_day": aggregate_day(rows, day - 1, start, dt) if day > 0 else None,
        },
    }
    try:
        _write_atomic(day_dir / "digest.json",
                      json.dumps(digest, ensure_ascii=False, indent=2, sort_keys=True))
    except Exception as exc:
        res["notes"].append(f"digest.json を書けない: {type(exc).__name__}")

    # 4) rollup.html(**新規ビューアコードは 0 行**。既存 make_viewer をサブプロセスで)
    if make_rollup and (day_dir / "l2_metrics.parquet").is_file():
        try:
            cmd = [sys.executable, str(REPO_ROOT / "viz" / "make_viewer.py"),
                   str(day_dir), "--daily-rollup"]
            env = dict(os.environ)
            env["PYTHONIOENCODING"] = "utf-8"
            proc = subprocess.run(cmd, capture_output=True, timeout=timeout,
                                  cwd=str(REPO_ROOT), env=env)
            html = day_dir / "rollup.html"
            if proc.returncode == 0 and html.is_file():
                res["rollup"] = html
            else:
                res["notes"].append(
                    f"make_viewer --daily-rollup 失敗 rc={proc.returncode}: "
                    f"{_clip(proc.stderr.decode('utf-8', 'replace'), 200)}")
        except Exception as exc:
            res["notes"].append(f"rollup を作れない: {type(exc).__name__}")
    for n in res["notes"]:
        rep.log(f"extract day {day}: {n}")
    return res


def maybe_regression(run_dir: Path, rep: Reporter, cfg, timeout: float = 300.0):
    """`detect_regression.py --quick` を**サブプロセス**で。失敗しても None を返すだけ。"""
    try:
        cmd = [sys.executable, str(REPO_ROOT / "scripts" / "detect_regression.py"),
               str(run_dir), "--quick",
               "--alpha", str(cfg.reg_alpha),
               "--min-rel-slope", str(cfg.reg_min_rel_slope)]
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout,
                              cwd=str(REPO_ROOT), env=env)
        line = (proc.stdout or b"").decode("utf-8", "replace").strip().splitlines()
        for s in reversed(line):
            if s.strip().startswith("{"):
                out = json.loads(s)
                out.pop("run", None)               # 絶対パスを持ち込まない
                return out
    except Exception as exc:
        rep.log(f"退行判定を実行できない: {type(exc).__name__}")
    return None


# --------------------------------------------------------------------------- #
# 1 サイクル
# --------------------------------------------------------------------------- #
def run_cycle(cfg, sink: DiscordSink, rep: Reporter, state: dict) -> dict:
    """読む → 組み立てる → **最後にまとめて投稿**(読み取りが投稿の失敗に巻き込まれない)。"""
    snap = build_snapshot(cfg, state)
    snap["_run_dir"] = cfg.run_dir
    now = snap["now"]
    state["cycles"] = int(state.get("cycles") or 0) + 1

    pending_days = [d for d in snap["completed_days"]
                    if d not in set(state.get("posted_days") or [])]
    if pending_days and cfg.with_regression:
        snap["regression"] = maybe_regression(cfg.run_dir, rep, cfg)

    alerts = evaluate_alerts(snap, state, cfg) if not cfg.no_alerts else []

    gap_note = ""
    last_post = float(state.get("last_post_ts") or 0.0)
    if last_post and (now - last_post) > cfg.gap_min * 60:
        gap_note = (f"⚠ 前回の投稿から **{_fmt_dur(now - last_post)}ぶり**"
                    "(レポーター停止 or 送信失敗が続いた可能性)")

    posted = {"alerts": 0, "digests": 0, "heartbeat": False, "suppressed": 0}

    # ---- A アラート(毎時上限。重大(SEV_CRIT)は cooldown も上限も跨いで必ず通す)
    # ★抑制は「遅らせる」ではなく「送らずに数える」。遷移の記録(slot["level"])は済んで
    #   いるので同じ遷移が後から蘇ることはない = 通知疲れを作らない代わりに、抑制した
    #   件数を次の日次ダイジェストへ必ず出す(silent cap 禁止)。
    for a in alerts:
        slot = _alert_slot(state, a["key"])
        cool = (now - float(slot.get("last_post") or 0.0)) < cfg.cooldown_min * 60
        if a["severity"] < SEV_CRIT and (cool or not _cap_ok(state, now, cfg)):
            state["suppressed"] = int(state.get("suppressed") or 0) + 1
            posted["suppressed"] += 1
            rep.log(f"アラート抑制({a['key']}→{a['level']}): "
                    f"{'クールダウン' if cool else '毎時上限'}")
            continue
        payload = build_alert(a, snap)
        if gap_note:
            payload["content"] = _clip(gap_note, MAX_CONTENT)
            gap_note = ""
        if sink.post(f"alert-{a['key']}", payload) is not None:
            slot["last_post"] = now
            _note_post(state, now, is_alert=True)
            posted["alerts"] += 1

    # ---- D 日次ダイジェスト(シミュ日の境界。上限の対象外 = 主たる成果物)
    for day in pending_days[: cfg.max_digests_per_cycle]:
        files = None
        attach_name = None
        if not cfg.no_attach:
            try:
                ex = extract_day(snap, day, cfg.out_dir, rep,
                                 make_rollup=True)
                if ex["rollup"] is not None:
                    data = ex["rollup"].read_bytes()
                    if len(data) <= MAX_ATTACH_BYTES:
                        attach_name = f"rollup-day{day:02d}.html"
                        files = [(attach_name, data)]
                    else:
                        rep.log(f"rollup が {len(data)} バイト = 添付上限超過のため送らない")
            except Exception as exc:
                rep.log(f"day {day} の取り出しに失敗: {type(exc).__name__}")
        payload = build_daily_digest(snap, day, state, attach_name)
        if gap_note:
            payload["content"] = _clip(gap_note, MAX_CONTENT)
            gap_note = ""
        if sink.post(f"daily-day{day:02d}", payload, files=files) is not None:
            state.setdefault("posted_days", []).append(day)
            state["suppressed"] = 0                # 件数を出したので台帳を空にする
            _note_post(state, now)
            posted["digests"] += 1

    # ---- H ハートビート(1 通を編集し続ける)
    if not cfg.daily_only:
        due = (now - float(state.get("heartbeat_ts") or 0.0)) >= cfg.heartbeat_min * 60
        if due or posted["digests"] or posted["alerts"]:
            payload = build_heartbeat(snap, state)
            if gap_note:                           # 他に何も出さなかった場合の受け皿
                payload["content"] = _clip(gap_note, MAX_CONTENT)
                gap_note = ""
            mid = state.get("heartbeat_id")
            ok = False
            if mid:
                ok = sink.patch("heartbeat", mid, payload)
                if not ok:
                    state["heartbeat_id"] = None
            if not ok:
                res = sink.post("heartbeat", payload, wait=True)
                if res is not None:
                    state["heartbeat_id"] = res.get("id")
                    ok = True
            if ok:
                state["heartbeat_ts"] = now
                # ★編集の成功も「Discord へ届いている」証拠 = A7(投稿の途絶)の分母に入れる。
                #   ここを落とすと、ハートビートが正常に更新され続けていても 2 時間後に
                #   「N 時間ぶり」の偽アラートが出る。
                _note_post(state, now)
                posted["heartbeat"] = True

    # ---- 進捗履歴(ETA の材料。12 点まで)
    hist = [h for h in (state.get("progress_history") or [])
            if isinstance(h, list) and len(h) == 2]
    hist.append([now, snap.get("last_step")])
    state["progress_history"] = hist[-12:]
    state["disabled"] = bool(sink.disabled)
    save_state(cfg.out_dir, state)

    rep.log(f"cycle #{state['cycles']}: step={snap.get('last_step')} "
            f"day={snap.get('cur_day')} state={snap.get('state')} "
            f"alerts={posted['alerts']} digests={posted['digests']} "
            f"suppressed={posted['suppressed']} hb={posted['heartbeat']}")
    return {"snapshot": snap, "posted": posted, "alerts": alerts}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="走行中のランを読むだけで Discord へ途中経過を出すサイドカー"
                    "(docs/plans/progress-reporting-plan.md レーン1+2)。")
    ap.add_argument("run_dir_pos", nargs="?", default=None, metavar="RUN_DIR",
                    help="ランのディレクトリ(--run-dir と同じ)")
    ap.add_argument("--run-dir", default=None,
                    help="ランのディレクトリ。診断 / リハーサル / 本選 / ローカルへ pull した"
                         "コピーのどれでもよい")
    ap.add_argument("--out-dir", default=None,
                    help=f"出力先(既定 <run-dir>/{OUT_SUBDIR})")
    ap.add_argument("--interval", type=float, default=0.0,
                    help="常駐する場合の間隔[秒](既定 0 = 1 サイクルで終了)")
    ap.add_argument("--once", action="store_true", help="1 サイクルで終了(既定と同じ)")
    ap.add_argument("--dry-run", action="store_true",
                    help="投稿せず、送る本文を <out>/dryrun/*.json と標準出力へ"
                         "(★初回はこれで確認する)")
    ap.add_argument("--webhook-env", default=WEBHOOK_ENV_DEFAULT,
                    help=f"webhook URL を持つ環境変数名(既定 {WEBHOOK_ENV_DEFAULT})。"
                         "★URL そのものを引数で受け取ることは意図的にしない")
    ap.add_argument("--daily-only", action="store_true", help="ハートビートを出さない")
    ap.add_argument("--no-alerts", action="store_true", help="アラートを出さない")
    ap.add_argument("--no-attach", action="store_true", help="rollup.html を添付しない")
    ap.add_argument("--heartbeat-min", type=float, default=10.0,
                    help="ハートビートの最短更新間隔[分](既定 10)")
    ap.add_argument("--stall-min", type=float, default=45.0,
                    help="進捗停止とみなす分(既定 45 = part 間隔上限 41 分 + 余裕)")
    ap.add_argument("--fallback-warn", type=float, default=0.20,
                    help="LLM fallback 率の発火閾値(既定 0.20)")
    ap.add_argument("--fallback-clear", type=float, default=0.15,
                    help="LLM fallback 率の復帰閾値(既定 0.15 = ヒステリシス)")
    ap.add_argument("--cooldown-min", type=float, default=30.0,
                    help="同一アラートの再送禁止時間[分](既定 30)")
    ap.add_argument("--max-posts-per-hour", type=int, default=6,
                    help="アラートの毎時上限(既定 6。超過分は件数を日次に出す)")
    ap.add_argument("--gap-min", type=float, default=120.0,
                    help="この分数ぶん投稿できていなければ復帰時に明記する(既定 120)")
    ap.add_argument("--max-digests-per-cycle", type=int, default=3,
                    help="1 サイクルで出す日次ダイジェストの上限(既定 3)")
    ap.add_argument("--with-regression", action="store_true",
                    help="日次のとき detect_regression.py --quick を実行して判定を載せる")
    ap.add_argument("--reg-alpha", type=float, default=REG_ALPHA,
                    help=f"退行判定の有意水準(既定 {REG_ALPHA})")
    ap.add_argument("--reg-min-rel-slope", type=float, default=REG_MIN_REL_SLOPE,
                    help=f"退行判定の効果量下限(既定 {REG_MIN_REL_SLOPE})")
    ap.add_argument("--extract", action="store_true",
                    help="取り出しのみ(投稿しない)。--day と併用")
    ap.add_argument("--day", type=int, default=None,
                    help="取り出す day(0 始まり = make_viewer --daily-rollup と同定義)")
    ap.add_argument("--quotes", default="off",
                    help="エージェント発話の引用。**off のみ実装**(ETHICS.md §2-4 の帰結)")
    ap.add_argument("--quiet", action="store_true", help="標準出力へのログを抑える")
    return ap


class Config:
    """CLI を 1 つの読み取り専用オブジェクトへ(テストが直接組み立てられる形)。"""

    def __init__(self, args, run_dir: Path, out_dir: Path):
        self.run_dir = run_dir
        self.out_dir = out_dir
        for k in ("interval", "dry_run", "webhook_env", "daily_only", "no_alerts",
                  "no_attach", "heartbeat_min", "stall_min", "fallback_warn",
                  "fallback_clear", "cooldown_min", "max_posts_per_hour", "gap_min",
                  "max_digests_per_cycle", "with_regression", "reg_alpha",
                  "reg_min_rel_slope", "quiet"):
            setattr(self, k, getattr(args, k))


def main(argv=None) -> int:
    """**終了コードは常に 0**(観測がランを殺してはならない)。"""
    try:
        args = build_parser().parse_args(argv)
    except SystemExit:                             # 使い方の誤りでも 0 で帰る
        return 0
    run_dir = Path(args.run_dir or args.run_dir_pos or ".")
    if not run_dir.is_absolute():
        run_dir = (Path.cwd() / run_dir).resolve()
    out_dir = Path(args.out_dir) if args.out_dir else (run_dir / OUT_SUBDIR)
    cfg = Config(args, run_dir, out_dir)

    url = None
    if not args.dry_run:
        url = os.environ.get(args.webhook_env) or None
    rep = Reporter(out_dir, url, echo=not args.quiet)

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        rep.log(f"出力先を作れない: {type(exc).__name__}")
        return 0
    if not run_dir.is_dir():
        rep.log(f"run-dir が無い: {run_dir.name}")
        return 0
    if str(args.quotes).lower() != "off":
        rep.log("--quotes on は意図的に未実装(ETHICS.md §2-4)。off として続行する")

    dry_dir = (out_dir / "dryrun") if args.dry_run else None
    if not args.dry_run and not url:
        rep.log(f"環境変数 {args.webhook_env} が未設定 → 投稿せず本文を "
                f"{OUT_SUBDIR}/dryrun へ書く(--dry-run と同じ挙動)")
        dry_dir = out_dir / "dryrun"

    state = load_state(out_dir)
    sink = DiscordSink(url, rep, dry_dir=dry_dir)
    if state.get("disabled"):
        sink.disabled = True
        rep.log("以前に 404 を受けているため投稿は恒久停止中(環境変数を差し替えて "
                "reporter_state.json の disabled を false に戻すと再開する)")

    # ---- 取り出しのみ
    if args.extract:
        try:
            snap = build_snapshot(cfg, state)
            snap["_run_dir"] = run_dir
            day = args.day
            if day is None:
                day = (snap["completed_days"] or [snap.get("cur_day") or 0])[-1]
            res = extract_day(snap, int(day), out_dir, rep, make_rollup=True)
            rep.log(f"extract day {day}: rows={res['l2_rows']} "
                    f"rollup={'yes' if res['rollup'] else 'no'} → "
                    f"{OUT_SUBDIR}/day-{int(day):02d}/")
        except Exception as exc:
            rep.log(f"取り出しで例外: {type(exc).__name__}: {exc}")
        return 0

    interval = 0.0 if args.once else float(args.interval or 0.0)
    while True:
        try:
            run_cycle(cfg, sink, rep, state)
        except Exception as exc:                   # ★ここから外へ例外を出さない(P5)
            rep.log(f"サイクルで例外: {type(exc).__name__}: {exc}")
        if interval <= 0:
            return 0
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            rep.log("中断(Ctrl-C)")
            return 0


if __name__ == "__main__":
    sys.exit(main())
