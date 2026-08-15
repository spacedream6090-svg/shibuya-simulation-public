"""進捗報告サイドカー(scripts/report_progress.py)。**観測がシムを変えないこと**が最上位の検収。

検収の柱(docs/plans/progress-reporting-plan.md §10):
  R 読み取り専用 — run-dir 直下に 1 バイトも書かない / 非侵襲な読み口(`_open_shared`)を
    live_viewer と**同一オブジェクト**で使っている / 読んでいる最中に part を消されても落ちない。
  P part 規律 — 書きかけ(切り詰めた)part を読まない・読む直前に消えた part で例外を出さない。
  D dedupe — resume 模擬(part 番号の振り直し + step 重複)で日次集計が二重計上しない。
    確報/速報の境界が checkpoint mtime で切れている。
  A アラート — 遷移でのみ発火・クールダウン・ヒステリシス・毎時上限と抑制件数・復帰通知。
  N ネットワーク — 実 Discord を叩かない。multipart のバイト組み立て・429(Retry-After 秒)・
    404 恒久停止・タイムアウトで終了コード 0・上限文字数の切り詰め。
  S セキュリティ — webhook URL がログ/stdout/例外文字列のどこにも現れない・絶対パスが
    投稿本文に現れない・発話本文が 1 文字も入らない・allowed_mentions が抑止形。
  V ビューア — 抽出 dir で make_viewer --daily-rollup が実際に走り rollup.html が生える。
    自己完結(外部 URL ゼロ)。start_min が 07:00 フォールバックでなく真値になっている。
  E 欠測 — status.json 無し/壊れ・列欠け・part ゼロ・pyarrow 不在で落ちず・捏造しない。
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ★ live_viewer を**先に**通常 import して sys.modules に載せる。report_progress の
# `from live_viewer import _open_shared, ...` が同じモジュールオブジェクトを掴むので、
# 「非侵襲な読み口を自前で書き直していない」ことを同一性で検査できる(test_r3)。
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
import live_viewer as LV                                        # noqa: E402

RP = _load("report_progress", "scripts/report_progress.py")

FAKE_WEBHOOK = ("https://discord.com/api/webhooks/1234567890123456789/"
                "abcdefGHIJKLmnopQRSTuvwxYZ0123456789_secret_token_xyz")

L2_COLS = ["n_moving", "n_inside_buildings", "n_sleeping", "mean_money",
           "opinion_var", "distinct_vocab_in_use", "total_adoptions", "n_groups",
           "llm_calls_total", "llm_fallback_rate", "llm_cache_hit_rate"]


# ============================================================ 合成ラン
def _row(step: int, bump: float = 0.0) -> dict:
    return {
        "step": step,
        "n_moving": 30 + step % 7 + int(bump),
        "n_inside_buildings": 60 + step % 5,
        "n_sleeping": 20 + step % 3,
        "mean_money": 180000.0 - step * 10 + bump,
        "opinion_var": 0.27 + step * 1e-5,
        "distinct_vocab_in_use": 1000 + step,
        "total_adoptions": 40 * step,
        "n_groups": 60 + step // 100,
        "llm_calls_total": 5000 * (step + 1),
        "llm_fallback_rate": 0.03,
        "llm_cache_hit_rate": 0.22,
    }


def _write_l2_part(run_dir: Path, idx: int, rows: list[dict]) -> Path:
    keys = ["step"] + L2_COLS
    path = run_dir / f"l2_metrics.part-{idx:04d}.parquet"
    pq.write_table(pa.table({k: [r.get(k) for r in rows] for k in keys}), path)
    return path


def _write_config(run_dir: Path, n_steps: int, start_tod: str = "07:00",
                  dt_min: int = 10) -> None:
    (run_dir / "config.yaml").write_text(
        f"run:\n  dt_min: {dt_min}\n  n_steps: {n_steps}\n  n_agents: 100\n"
        f"  start_tod: '{start_tod}'\n"
        "observer:\n  flush_every_steps: 6\n  checkpoint_every: 72\n",
        encoding="utf-8")


def _write_status(run_dir: Path, **over) -> None:
    data = {"state": "running", "restarts": 0, "max_restarts": 10,
            "last_progress": {"checkpoint": "ckpt-000072.pkl.gz", "step": 72,
                              "mtime": "2026-08-15T12:00:00+09:00"},
            "last_backup_step": 72,
            "run_dir": str(Path.home() / "runs" / "finals"),
            "backup_dir": str(Path.home() / "backup"),
            "pid": 4242, "updated": "2026-08-15T12:00:05+09:00",
            "llm_health": {"step": 100, "llm_calls_total": 500000,
                           "llm_fallback_rate": 0.03, "llm_cache_hit_rate": 0.22,
                           "source": "l2_metrics.part-0016.parquet"},
            "disk": {"label": "run", "path": str(Path.home()), "state": "ok",
                     "free_gb": 1420.5, "warn_gb": 200.0, "crit_gb": 50.0}}
    data.update(over)
    (run_dir / "status.json").write_text(json.dumps(data, ensure_ascii=False),
                                         encoding="utf-8")


def _make_run(tmp_path: Path, name: str = "r", n_steps: int = 300,
              flush: int = 6, start_tod: str = "07:00", ckpt_steps=(72, 144),
              with_status: bool = True) -> Path:
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_config(run_dir, n_steps, start_tod)
    for i, base in enumerate(range(0, n_steps, flush)):
        _write_l2_part(run_dir, i, [_row(s) for s in range(base, min(base + flush, n_steps))])
    ck = run_dir / "checkpoint"
    ck.mkdir(exist_ok=True)
    for s in ckpt_steps:
        (ck / f"ckpt-{s:06d}.pkl.gz").write_bytes(b"\x1f\x8b" + b"0" * 16)
    if with_status:
        _write_status(run_dir)
    return run_dir


def _cfg(run_dir: Path, out_dir: Path, **over):
    """CLI を通さずに Config 相当を組む(テストが直接つまみを回せる形)。"""
    args = RP.build_parser().parse_args([str(run_dir)])
    for k, v in over.items():
        setattr(args, k, v)
    return RP.Config(args, run_dir, out_dir)


class _Sink(RP.DiscordSink):
    """1 バイトも送らない sink(投稿内容だけ集める)。"""

    def __init__(self, rep, url=None):
        super().__init__(url, rep, dry_dir=None)
        self.posts = []
        self.patches = []

    def post(self, kind, payload, files=None, wait=False):
        self.posts.append({"kind": kind, "payload": RP._scrub(payload, self.url),
                           "files": files or []})
        return {"id": f"msg-{len(self.posts)}"}

    def patch(self, kind, message_id, payload, files=None):
        self.patches.append({"kind": kind, "id": message_id,
                             "payload": RP._scrub(payload, self.url)})
        return True


def _rep(out_dir: Path, url=None):
    return RP.Reporter(out_dir, url, echo=False)


def _cycle(run_dir: Path, out_dir: Path, sink=None, state=None, **over):
    rep = _rep(out_dir)
    sink = sink or _Sink(rep)
    state = state if state is not None else RP.load_state(out_dir)
    cfg = _cfg(run_dir, out_dir, **over)
    res = RP.run_cycle(cfg, sink, rep, state)
    return res, sink, state


# ============================================================ R 読み取り専用
def test_r1_writes_nothing_into_run_dir_root(tmp_path):
    """run-dir 直下に増えるのは `_progress/` だけ(1 バイトも直下へ書かない)。"""
    run_dir = _make_run(tmp_path)
    before = {p.name for p in run_dir.iterdir()}
    _cycle(run_dir, run_dir / "_progress", no_attach=True)
    after = {p.name for p in run_dir.iterdir()}
    assert after - before == {"_progress"}, f"run-dir 直下が汚れた: {after - before}"


def test_r2_out_dir_can_live_outside_run_dir(tmp_path):
    """--out-dir で run-dir の外へ出せば run-dir は 1 エントリも増えない。"""
    run_dir = _make_run(tmp_path)
    out = tmp_path / "elsewhere"
    before = {p.name for p in run_dir.iterdir()}
    _cycle(run_dir, out, no_attach=True)
    assert {p.name for p in run_dir.iterdir()} == before
    assert (out / RP.STATE_NAME).is_file()


def test_r3_open_shared_is_the_same_object_as_live_viewer(tmp_path):
    """非侵襲な読み口を**自前で書き直していない**(単一の源 = live_viewer)。"""
    assert RP._open_shared is LV._open_shared
    assert RP.is_complete_parquet is LV.is_complete_parquet
    assert RP.list_parts is LV.list_parts


def test_r4_part_deleted_while_open_does_not_break(tmp_path):
    """開いている最中に part を消されても(Windows の共有削除)読み取りは完走する。"""
    run_dir = _make_run(tmp_path, n_steps=60)
    part = run_dir / "l2_metrics.part-0000.parquet"
    with RP._open_shared(part) as fh:
        part.unlink()                        # ★素の open だとここが WinError 32 で落ちる
        assert fh.read(4) == b"PAR1"
    res = RP.read_l2(run_dir, columns=RP.REPORT_COLUMNS)
    assert res["rows"], "残りの part は読めていなければならない"


@pytest.mark.xdist_group("subprocess_viewer")
def test_r6_finalize_unlink_succeeds_while_reporter_reads(tmp_path):
    """★第77バッチの事故の再現固定: レポーターが読んでいる最中の `part.unlink()` が成功する。

    素の `open()`(FILE_SHARE_DELETE なし)で読んでいると、ここで
    `PermissionError [WinError 32]` が出て **シム本体の finalize が落ちる**。
    """
    import threading

    run_dir = _make_run(tmp_path, n_steps=900, flush=6, with_status=False)
    errors: list = []
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            try:
                RP.read_l2(run_dir, columns=RP.REPORT_COLUMNS)
            except Exception as exc:               # 読む側が落ちてもいけない
                errors.append(exc)

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    time.sleep(0.15)
    try:
        for p in sorted(run_dir.glob("l2_metrics.part-*.parquet")):
            p.unlink()                             # ← logger._finalize_stream 相当
    finally:
        stop.set()
        t.join(timeout=10)
    assert not errors, errors[:3]


def test_r5_reporter_never_touches_l1(tmp_path):
    """最小構成は L1 を 1 バイトも開かない(kind 列すら読まない)。"""
    run_dir = _make_run(tmp_path, n_steps=60)
    l1 = run_dir / "l1_events.part-0000.parquet"
    l1.write_bytes(b"NOT A PARQUET FILE AT ALL")   # 開いたら必ず壊れる餌
    res, _, _ = _cycle(run_dir, tmp_path / "o", no_attach=True)
    assert res["snapshot"]["fs"]["l1_parts"] == 1     # 個数は数える(stat だけ)
    assert res["snapshot"]["l2_rows"], "L2 は読めている"


# ============================================================ P part 規律
def test_p1_truncated_part_is_not_read(tmp_path):
    """切り詰めた(= 書き込み中の)part は取り込まない。"""
    run_dir = _make_run(tmp_path, n_steps=60)
    last = sorted(run_dir.glob("l2_metrics.part-*.parquet"))[-1]
    data = last.read_bytes()
    last.write_bytes(data[: len(data) // 2])
    res = RP.read_l2(run_dir, columns=RP.REPORT_COLUMNS)
    assert last.name in res["parts_pending"]
    assert max(r["step"] for r in res["rows"]) < 54


def test_p2_missing_part_race_is_swallowed(tmp_path, monkeypatch):
    """読む直前に消えた part(finalize レース)で例外を出さない。"""
    run_dir = _make_run(tmp_path, n_steps=30)
    real = RP._read_part_rows
    calls = {"n": 0}

    def flaky(path, columns=None):
        calls["n"] += 1
        if calls["n"] == 2:
            raise FileNotFoundError(2, "gone", str(path))
        return real(path, columns)

    monkeypatch.setattr(RP, "_read_part_rows", flaky)
    res = RP.read_l2(run_dir, columns=RP.REPORT_COLUMNS)
    assert any("消えた" in n for n in res["notes"])
    assert res["rows"], "残りは読めている"


def test_p3_no_parts_at_all(tmp_path):
    """part が 1 本も無くても落ちず・捏造しない。"""
    run_dir = tmp_path / "empty"
    run_dir.mkdir()
    _write_config(run_dir, 100)
    res = RP.read_l2(run_dir, columns=RP.REPORT_COLUMNS)
    assert res["rows"] == [] and res["confirmed_step"] is None


# ============================================================ D dedupe / 確報
def test_d1_resume_renumbered_parts_do_not_double_count(tmp_path):
    """resume 模擬: 同じ step を含む part が新しい index で現れても二重計上しない。"""
    run_dir = _make_run(tmp_path, n_steps=144)
    n_before = len(RP.read_l2(run_dir, columns=RP.REPORT_COLUMNS)["rows"])
    # crash → checkpoint step 72 から resume。step 72.. を新 index で書き直す。
    idx = 100
    for base in range(72, 144, 6):
        _write_l2_part(run_dir, idx, [_row(s, bump=1000.0) for s in range(base, base + 6)])
        idx += 1
    res = RP.read_l2(run_dir, columns=RP.REPORT_COLUMNS)
    assert len(res["rows"]) == n_before == 144, "行数が増えてはならない(step で dedupe)"
    steps = [r["step"] for r in res["rows"]]
    assert steps == sorted(set(steps)) == list(range(144))
    after = {r["step"]: r for r in res["rows"]}
    assert after[100]["mean_money"] == pytest.approx(180000.0 - 1000 + 1000.0), \
        "★後勝ち: resume 後に書かれた値が正でなければならない"
    assert after[10]["mean_money"] == pytest.approx(180000.0 - 100)


def test_d2_daily_aggregate_is_not_double_counted_after_resume(tmp_path):
    """日次集計そのものが resume で二重にならない(n_rows が 1 日 = 144 行のまま)。"""
    run_dir = _make_run(tmp_path, n_steps=300)
    idx = 200
    for base in range(150, 300, 6):
        _write_l2_part(run_dir, idx, [_row(s) for s in range(base, base + 6)])
        idx += 1
    rows = RP.read_l2(run_dir, columns=RP.REPORT_COLUMNS)["rows"]
    agg = RP.aggregate_day(rows, 1, 7 * 60, 10)
    assert agg["n_rows"] == 144


def test_d3_confirmed_boundary_is_checkpoint_mtime(tmp_path):
    """確報 = 最新 checkpoint の mtime 以前に書かれた part だけ(境界より先は速報)。"""
    run_dir = _make_run(tmp_path, n_steps=60)
    parts = sorted(run_dir.glob("l2_metrics.part-*.parquet"))
    base = time.time() - 10000
    for i, p in enumerate(parts):
        os.utime(p, (base + i, base + i))
    ck = run_dir / "checkpoint" / "ckpt-000072.pkl.gz"
    boundary = base + 5.5                                   # 先頭 6 本だけが境界の内側
    os.utime(ck, (boundary, boundary))
    res = RP.read_l2(run_dir, columns=RP.REPORT_COLUMNS, confirm_before=boundary)
    assert res["confirmed_step"] == 35, res["confirmed_step"]
    assert max(r["step"] for r in res["rows"]) == 59, "速報は境界より先も読む"


def test_d4_finalized_canonical_is_confirmed(tmp_path):
    """finalize 済み(canonical あり)なら巻き戻らない = 全部が確報。"""
    run_dir = _make_run(tmp_path, n_steps=60)
    rows = [_row(s) for s in range(60)]
    pq.write_table(pa.table({k: [r[k] for r in rows] for k in ["step"] + L2_COLS}),
                   run_dir / "l2_metrics.parquet")
    res = RP.read_l2(run_dir, columns=RP.REPORT_COLUMNS, confirm_before=None)
    assert res["confirmed_step"] == 59


def test_d5_day_boundary_matches_make_viewer(tmp_path):
    """日境界の定義が make_viewer --daily-rollup / analyze_structure と同じ。"""
    assert RP.day_of(0, 7 * 60, 10) == 0
    assert RP.day_of(101, 7 * 60, 10) == 0
    assert RP.day_of(102, 7 * 60, 10) == 1          # 07:00 + 102*10 分 = 翌 00:00
    assert RP.completed_days([{"step": s} for s in range(0, 150)], 7 * 60, 10) == [0]


# ============================================================ A アラート
def _alert_cycle(run_dir, out_dir, state, sink, **over):
    rep = _rep(out_dir)
    cfg = _cfg(run_dir, out_dir, no_attach=True, daily_only=True, **over)
    return RP.run_cycle(cfg, sink, rep, state)


def test_a1_fires_only_on_transition(tmp_path):
    """同じ状態が続く間は鳴らさない(遷移でのみ発火)。"""
    run_dir = _make_run(tmp_path, n_steps=60)
    out = tmp_path / "o"
    state = RP.load_state(out)
    sink = _Sink(_rep(out))
    _alert_cycle(run_dir, out, state, sink)                     # 初回 running = 鳴らさない
    assert [p for p in sink.posts if p["kind"].startswith("alert")] == []
    _write_status(run_dir, state="failed")
    _alert_cycle(run_dir, out, state, sink)
    keys = [p["kind"] for p in sink.posts if p["kind"].startswith("alert")]
    assert keys == ["alert-state"]
    _alert_cycle(run_dir, out, state, sink)                     # 同じ failed = 沈黙
    assert len([p for p in sink.posts if p["kind"].startswith("alert")]) == 1


def test_a2_cooldown_suppresses_and_counts(tmp_path):
    """クールダウン中は再送しない。抑制した件数を数える(silent cap 禁止)。"""
    run_dir = _make_run(tmp_path, n_steps=60)
    out = tmp_path / "o"
    state = RP.load_state(out)
    sink = _Sink(_rep(out))
    _write_status(run_dir, disk={"state": "warn", "free_gb": 10.0,
                                 "warn_gb": 200.0, "crit_gb": 5.0})
    _alert_cycle(run_dir, out, state, sink)
    assert any(p["kind"] == "alert-disk" for p in sink.posts)
    _write_status(run_dir, disk={"state": "ok", "free_gb": 900.0,
                                 "warn_gb": 200.0, "crit_gb": 5.0})
    _alert_cycle(run_dir, out, state, sink)                     # 30 分以内 = 抑制
    assert len([p for p in sink.posts if p["kind"] == "alert-disk"]) == 1
    assert state["suppressed"] == 1


def test_a3_critical_bypasses_cooldown(tmp_path):
    """重大(critical / failed)はクールダウンを跨いで必ず通す。"""
    run_dir = _make_run(tmp_path, n_steps=60)
    out = tmp_path / "o"
    state = RP.load_state(out)
    sink = _Sink(_rep(out))
    _write_status(run_dir, disk={"state": "warn", "free_gb": 10.0,
                                 "warn_gb": 200.0, "crit_gb": 5.0})
    _alert_cycle(run_dir, out, state, sink)
    _write_status(run_dir, disk={"state": "critical", "free_gb": 1.0,
                                 "warn_gb": 200.0, "crit_gb": 5.0})
    _alert_cycle(run_dir, out, state, sink)
    assert len([p for p in sink.posts if p["kind"] == "alert-disk"]) == 2


def test_a4_hysteresis_uses_two_thresholds(tmp_path):
    """fallback 率は 0.20 で発火・0.15 で復帰(帯の中では状態を保つ)。2 サイクル連続が必要。"""
    run_dir = _make_run(tmp_path, n_steps=60)
    out = tmp_path / "o"
    state = RP.load_state(out)
    sink = _Sink(_rep(out))
    hi = {"step": 1, "llm_fallback_rate": 0.30, "llm_calls_total": 1,
          "llm_cache_hit_rate": 0.2, "source": "x"}
    mid = dict(hi, llm_fallback_rate=0.17)
    lo = dict(hi, llm_fallback_rate=0.05)
    _write_status(run_dir, llm_health=lo)
    _alert_cycle(run_dir, out, state, sink)                     # ok を記録(無音)
    _write_status(run_dir, llm_health=hi)
    _alert_cycle(run_dir, out, state, sink)                     # 1 サイクル目 = まだ鳴らない
    assert not [p for p in sink.posts if p["kind"] == "alert-fallback"]
    _alert_cycle(run_dir, out, state, sink)                     # 2 サイクル連続 = 発火
    assert len([p for p in sink.posts if p["kind"] == "alert-fallback"]) == 1
    _write_status(run_dir, llm_health=mid)                      # 帯の中 = 復帰しない
    _alert_cycle(run_dir, out, state, sink)
    _alert_cycle(run_dir, out, state, sink)
    assert RP._alert_slot(state, "fallback")["level"] == "high"
    _write_status(run_dir, llm_health=lo)                       # 下抜け = 復帰
    _alert_cycle(run_dir, out, state, sink, cooldown_min=0.0)
    _alert_cycle(run_dir, out, state, sink, cooldown_min=0.0)
    assert RP._alert_slot(state, "fallback")["level"] == "ok"
    assert len([p for p in sink.posts if p["kind"] == "alert-fallback"]) == 2


def test_a5_hourly_cap_counts_suppressed(tmp_path):
    """毎時上限を超えたら送らずに数え、次の日次ダイジェストへ件数を出す。"""
    run_dir = _make_run(tmp_path, n_steps=300)
    out = tmp_path / "o"
    state = RP.load_state(out)
    state["posted_days"] = [0, 1]                    # 日次は出し終えた状態にしておく
    state["post_times"] = [time.time()] * 6                     # 既に上限
    sink = _Sink(_rep(out))
    _write_status(run_dir, restarts=5, max_restarts=10)          # 50% 帯 = WARN
    _alert_cycle(run_dir, out, state, sink, cooldown_min=0.0)
    assert not [p for p in sink.posts if p["kind"].startswith("alert")]
    assert state["suppressed"] >= 1
    state["posted_days"] = [0]                       # day 1 の日次をこれから出す
    res, sink2, _ = _cycle(run_dir, out, state=state, no_attach=True)
    digest = [p for p in sink2.posts if p["kind"].startswith("daily")]
    assert digest, "日次ダイジェストが出ていない"
    text = json.dumps(digest[0]["payload"], ensure_ascii=False)
    assert "抑制したアラート" in text and "件" in text
    assert state["suppressed"] == 0, "件数を出したら台帳を空にする"


def test_a6_stall_alert_and_recovery(tmp_path):
    """進捗停止 → 復帰の 2 通が出る(2 サイクル連続でのみ確定)。"""
    run_dir = _make_run(tmp_path, n_steps=60)
    out = tmp_path / "o"
    state = RP.load_state(out)
    sink = _Sink(_rep(out))
    old = time.time() - 3 * 3600
    for p in list(run_dir.glob("*.parquet")) + list((run_dir / "checkpoint").iterdir()):
        os.utime(p, (old, old))
    for _ in range(2):
        _alert_cycle(run_dir, out, state, sink, stall_min=45.0)
    assert RP._alert_slot(state, "stall")["level"] == "stalled"
    now = time.time()
    for p in run_dir.glob("*.parquet"):
        os.utime(p, (now, now))
    for _ in range(2):
        _alert_cycle(run_dir, out, state, sink, stall_min=45.0, cooldown_min=0.0)
    assert RP._alert_slot(state, "stall")["level"] == "ok"
    stalls = [p for p in sink.posts if p["kind"] == "alert-stall"]
    assert len(stalls) == 2, [p["payload"]["embeds"][0]["title"] for p in stalls]


def test_a7_gap_note_on_return(tmp_path):
    """長時間投稿できていなかった後の 1 通目に「N 時間ぶり」を明記する(監視の監視)。"""
    run_dir = _make_run(tmp_path, n_steps=300)
    out = tmp_path / "o"
    state = RP.load_state(out)
    state["last_post_ts"] = time.time() - 5 * 3600
    sink = _Sink(_rep(out))
    _cycle(run_dir, out, sink=sink, state=state, no_attach=True)
    first = sink.posts[0]
    assert "ぶり" in (first["payload"].get("content") or "")
    # 日次もアラートも無いサイクルでは、ハートビートが受け皿になる
    state2 = RP.load_state(tmp_path / "o2")
    state2["posted_days"] = [0, 1]
    state2["last_post_ts"] = time.time() - 5 * 3600
    sink2 = _Sink(_rep(out))
    _cycle(run_dir, out, sink=sink2, state=state2, no_attach=True)
    assert [p["kind"] for p in sink2.posts] == ["heartbeat"]
    assert "ぶり" in (sink2.posts[0]["payload"].get("content") or "")


def test_a8_heartbeat_edit_counts_as_a_successful_post(tmp_path):
    """★ハートビートの**編集**成功も「届いている」証拠(偽の「N 時間ぶり」を出さない)。"""
    run_dir = _make_run(tmp_path, n_steps=60)
    out = tmp_path / "o"
    state = RP.load_state(out)
    sink = _Sink(_rep(out))
    _cycle(run_dir, out, sink=sink, state=state, no_attach=True)     # POST(立てる)
    assert state["heartbeat_id"]
    state["heartbeat_ts"] = 0.0
    state["last_post_ts"] = 0.0                                      # 途絶したことにする
    _cycle(run_dir, out, sink=sink, state=state, no_attach=True)     # PATCH(編集)
    assert sink.patches, "編集経路を通っていない"
    assert state["last_post_ts"] > 0.0, "編集が A7 の分母に入っていない"


# ============================================================ N ネットワーク
def test_n1_dry_run_sends_nothing(tmp_path, monkeypatch):
    """--dry-run は 1 バイトも送らず、送る本文をローカルへ書く。"""
    run_dir = _make_run(tmp_path, n_steps=300)
    monkeypatch.setenv(RP.WEBHOOK_ENV_DEFAULT, FAKE_WEBHOOK)

    def boom(*a, **k):
        raise AssertionError("dry-run なのにネットワークへ出た")

    monkeypatch.setattr(RP.DiscordSink, "_http", boom)
    rc = RP.main([str(run_dir), "--dry-run", "--quiet", "--no-attach"])
    assert rc == 0
    files = sorted((run_dir / "_progress" / "dryrun").glob("*.json"))
    assert files, "dry-run の成果物が無い"
    kinds = {json.loads(f.read_text(encoding="utf-8"))["kind"] for f in files}
    assert any(k.startswith("daily") for k in kinds)
    assert "heartbeat" in kinds


def test_n2_multipart_is_byte_correct():
    """multipart の境界・payload_json・files[0] がバイト単位で正しい。"""
    payload = {"embeds": [{"title": "t"}], "allowed_mentions": {"parse": []}}
    ctype, body = RP.build_multipart(payload, [("rollup.html", b"<html>x</html>")],
                                     boundary="BOUND")
    assert ctype == "multipart/form-data; boundary=BOUND"
    text = body.decode("utf-8")
    assert text.startswith("--BOUND\r\n")
    assert text.endswith("--BOUND--\r\n")
    assert 'Content-Disposition: form-data; name="payload_json"' in text
    assert 'name="files[0]"; filename="rollup.html"' in text
    assert json.dumps(payload, ensure_ascii=False) in text
    assert "<html>x</html>" in text
    assert text.count("--BOUND\r\n") == 2               # payload_json と files[0]


def test_n3_multipart_regenerates_boundary_on_collision():
    """境界が本体に現れる場合は決して壊れた multipart を作らない。"""
    ctype, body = RP.build_multipart({"content": "--shibuyaX"},
                                     [("a.bin", b"\x00\x01")])
    b = ctype.split("boundary=")[1]
    assert body.count(f"--{b}".encode()) == 3           # 開始 2 + 終端 1


def test_n4_retry_after_is_seconds(tmp_path):
    """429 は `Retry-After`(**秒**・小数)に従う。v6 のミリ秒と混同しない。"""
    out = tmp_path / "o"
    rep = _rep(out, FAKE_WEBHOOK)
    slept = []
    calls = {"n": 0}

    def opener(method, url, body, ctype):
        calls["n"] += 1
        if calls["n"] == 1:
            return 429, {"Retry-After": "1.25"}, '{"retry_after": 1.25}'
        return 200, {}, '{"id": "42"}'

    sink = RP.DiscordSink(FAKE_WEBHOOK, rep, sleep=slept.append, opener=opener)
    res = sink.post("x", {"content": "hi"})
    assert res == {"id": "42"} and slept == [1.25] and calls["n"] == 2


def test_n5_404_stops_permanently(tmp_path):
    """404 を受けたら以後の投稿を恒久停止する(叩き続けると一時制限される)。"""
    out = tmp_path / "o"
    rep = _rep(out, FAKE_WEBHOOK)
    calls = {"n": 0}

    def opener(method, url, body, ctype):
        calls["n"] += 1
        return 404, {}, '{"message": "Unknown Webhook", "code": 10015}'

    sink = RP.DiscordSink(FAKE_WEBHOOK, rep, sleep=lambda s: None, opener=opener)
    assert sink.post("x", {"content": "hi"}) is None
    assert sink.disabled is True
    assert sink.post("y", {"content": "hi"}) is None
    assert calls["n"] == 1, "恒久停止後に叩いてはならない"


def test_n6_message_404_relaunches_heartbeat(tmp_path):
    """メッセージだけ消えた場合(PATCH の 404)は webhook を殺さず立て直す。"""
    out = tmp_path / "o"
    rep = _rep(out, FAKE_WEBHOOK)
    seen = []

    def opener(method, url, body, ctype):
        seen.append(method)
        if method == "PATCH":
            return 404, {}, '{"message": "Unknown Message"}'
        return 200, {}, '{"id": "99"}'

    sink = RP.DiscordSink(FAKE_WEBHOOK, rep, sleep=lambda s: None, opener=opener)
    assert sink.patch("heartbeat", "1", {"content": "x"}) is False
    assert sink.disabled is False
    assert sink.post("heartbeat", {"content": "x"}, wait=True) == {"id": "99"}


def test_n7_network_failure_never_escapes(tmp_path, monkeypatch):
    """接続失敗・タイムアウトで例外を外へ出さず、終了コードは 0。"""
    run_dir = _make_run(tmp_path, n_steps=300)
    monkeypatch.setenv(RP.WEBHOOK_ENV_DEFAULT, FAKE_WEBHOOK)

    def boom(*a, **k):
        raise TimeoutError("The read operation timed out")

    monkeypatch.setattr(RP.DiscordSink, "_http", boom)
    assert RP.main([str(run_dir), "--quiet", "--no-attach"]) == 0


def test_n8_embed_limits_are_enforced():
    """title 256 / description 4096 / field value 1024 / 全体 6000 の切り詰めが働く。"""
    e = RP._clamp_embed({"title": "あ" * 400, "description": "い" * 5000,
                         "fields": [{"name": "n" * 400, "value": "v" * 4000}
                                    for _ in range(30)],
                         "footer": {"text": "f" * 3000}})
    assert len(e["title"]) <= RP.MAX_EMBED_TITLE
    assert len(e["description"]) <= RP.MAX_EMBED_DESC
    assert len(e.get("fields", [])) <= RP.MAX_FIELDS
    for f in e.get("fields", []):
        assert len(f["name"]) <= RP.MAX_FIELD_NAME
        assert len(f["value"]) <= RP.MAX_FIELD_VALUE
    assert len(e["footer"]["text"]) <= RP.MAX_FOOTER
    total = (len(e["title"]) + len(e["description"]) + len(e["footer"]["text"])
             + sum(len(f["name"]) + len(f["value"]) for f in e.get("fields", [])))
    assert total <= RP.MAX_EMBED_TOTAL


def test_n9_real_payloads_are_within_limits(tmp_path):
    """実際に組む 3 系統のペイロードが全て Discord の上限内に収まる。"""
    run_dir = _make_run(tmp_path, n_steps=300)
    out = tmp_path / "o"
    res, sink, state = _cycle(run_dir, out, no_attach=True)
    for post in sink.posts + sink.patches:
        for e in post["payload"]["embeds"]:
            total = (len(e.get("title", "")) + len(e.get("description", ""))
                     + len((e.get("footer") or {}).get("text", ""))
                     + sum(len(f["name"]) + len(f["value"])
                           for f in e.get("fields", [])))
            assert total <= RP.MAX_EMBED_TOTAL
        assert len(post["payload"].get("content") or "") <= RP.MAX_CONTENT


# ============================================================ S セキュリティ
def _all_text(*objs) -> str:
    return "\n".join(json.dumps(o, ensure_ascii=False, default=str) for o in objs)


def test_s1_webhook_url_never_appears_anywhere(tmp_path, capsys, monkeypatch):
    """URL(id も token も)がログ・stdout・例外文字列のどこにも現れない。"""
    run_dir = _make_run(tmp_path, n_steps=300)
    monkeypatch.setenv(RP.WEBHOOK_ENV_DEFAULT, FAKE_WEBHOOK)
    import urllib.error

    def boom(*a, **k):
        # ★典型事故: 例外の repr に URL が載る
        raise urllib.error.URLError(f"connect failed for {FAKE_WEBHOOK}")

    monkeypatch.setattr(RP.DiscordSink, "_http", boom)
    assert RP.main([str(run_dir), "--no-attach"]) == 0
    captured = capsys.readouterr()
    log = (run_dir / "_progress" / RP.LOG_NAME).read_text(encoding="utf-8")
    token = FAKE_WEBHOOK.rsplit("/", 1)[-1]
    wid = FAKE_WEBHOOK.rsplit("/", 2)[-2]
    for blob, where in ((log, "reporter.log"), (captured.out, "stdout"),
                        (captured.err, "stderr")):
        assert FAKE_WEBHOOK not in blob, where
        assert token not in blob, where
        assert wid not in blob, where
        assert "discord.com/api/webhooks" not in blob, where


def test_s2_state_file_has_no_webhook(tmp_path, monkeypatch):
    """状態ファイルにも URL を残さない。"""
    run_dir = _make_run(tmp_path, n_steps=300)
    monkeypatch.setenv(RP.WEBHOOK_ENV_DEFAULT, FAKE_WEBHOOK)
    monkeypatch.setattr(RP.DiscordSink, "_http",
                        lambda *a, **k: (200, {}, '{"id": "1"}'))
    RP.main([str(run_dir), "--quiet", "--no-attach"])
    text = (run_dir / "_progress" / RP.STATE_NAME).read_text(encoding="utf-8")
    assert FAKE_WEBHOOK not in text and "webhooks" not in text


def test_s3_no_absolute_paths_in_posts(tmp_path):
    """投稿本文に絶対パス(= ユーザー名 = 個人情報)が 1 つも出ない。"""
    run_dir = _make_run(tmp_path, n_steps=300)
    res, sink, state = _cycle(run_dir, tmp_path / "o", no_attach=True)
    blob = _all_text(*[p["payload"] for p in sink.posts + sink.patches])
    assert not RP._ABSPATH_RE.search(blob), RP._ABSPATH_RE.search(blob).group(0)
    assert str(tmp_path) not in blob
    assert os.path.expanduser("~") not in blob


def test_s4_status_run_dir_is_dropped_at_read(tmp_path):
    """status.json の run_dir / backup_dir / disk.path は読み込んだ時点で捨てる。"""
    run_dir = _make_run(tmp_path, n_steps=60)
    st = RP.read_status(run_dir)
    assert "run_dir" not in st and "backup_dir" not in st
    assert "path" not in st["disk"]


def test_s5_allowed_mentions_suppresses_everything(tmp_path):
    """全メッセージに「何も ping しない」allowed_mentions が付く。"""
    run_dir = _make_run(tmp_path, n_steps=300)
    res, sink, state = _cycle(run_dir, tmp_path / "o", no_attach=True)
    assert sink.posts
    for p in sink.posts + sink.patches:
        assert p["payload"]["allowed_mentions"] == {"parse": []}


def test_s6_everyone_in_run_name_cannot_ping(tmp_path):
    """`@everyone` を含む文字列が混ざっても allowed_mentions で必ず抑止される。"""
    run_dir = _make_run(tmp_path, name="@everyone-run", n_steps=300)
    res, sink, state = _cycle(run_dir, tmp_path / "o", no_attach=True)
    blob = _all_text(*[p["payload"] for p in sink.posts])
    assert "@everyone" in blob                      # 文字としては出るが
    for p in sink.posts:
        assert p["payload"]["allowed_mentions"] == {"parse": []}   # ping はしない


def test_s7_no_utterance_text_in_posts(tmp_path):
    """--quotes off 既定: 発話・DM・SNS 本文が 1 文字も入らない(L1 を開かないので構造的)。"""
    run_dir = _make_run(tmp_path, n_steps=300)
    secret = "これはエージェントの発話本文である"
    pq.write_table(pa.table({"step": [0], "payload": [secret]}),
                   run_dir / "l1_events.part-0000.parquet")
    res, sink, state = _cycle(run_dir, tmp_path / "o", no_attach=True)
    blob = _all_text(*[p["payload"] for p in sink.posts + sink.patches])
    assert secret not in blob
    assert "--quotes" not in RP.build_daily_digest.__doc__.replace("--quotes", "")


def test_s8_quotes_on_is_not_implemented(tmp_path):
    """--quotes on は意図的に未実装(ETHICS.md §2-4)。落ちずに off として続行する。"""
    run_dir = _make_run(tmp_path, n_steps=60)
    assert RP.main([str(run_dir), "--dry-run", "--quiet", "--quotes", "on",
                    "--no-attach"]) == 0
    log = (run_dir / "_progress" / RP.LOG_NAME).read_text(encoding="utf-8")
    assert "未実装" in log


def test_s9_no_webhook_url_cli_flag():
    """★URL を CLI 引数で受け取る口を作らない(ps / シェル履歴 / タスク XML に残るため)。"""
    opts = {s for a in RP.build_parser()._actions for s in (a.option_strings or [])}
    assert "--webhook-env" in opts
    for bad in ("--webhook-url", "--webhook", "--url", "--token"):
        assert bad not in opts, f"URL を引数で受け取る口があってはならない: {bad}"
    text = (REPO_ROOT / "scripts" / "report_progress.py").read_text(encoding="utf-8")
    assert "discord.com/api/webhooks/1" not in text, "リポジトリに実 URL を書かない"


def test_s10_fiction_note_on_every_message(tmp_path):
    """フィクション注記が全メッセージのフッタに入る(ETHICS §2-1,2)。"""
    run_dir = _make_run(tmp_path, n_steps=300)
    res, sink, state = _cycle(run_dir, tmp_path / "o", no_attach=True)
    for p in sink.posts + sink.patches:
        for e in p["payload"]["embeds"]:
            assert RP.FICTION_NOTE in e["footer"]["text"]


# ============================================================ V ビューア
@pytest.mark.xdist_group("subprocess_viewer")
def test_v1_rollup_html_is_generated_and_self_contained(tmp_path):
    """抽出 dir で make_viewer.py --daily-rollup が実際に走り rollup.html が生える。"""
    run_dir = _make_run(tmp_path, n_steps=300)
    out = tmp_path / "o"
    rep = _rep(out)
    cfg = _cfg(run_dir, out)
    snap = RP.build_snapshot(cfg, None)
    snap["_run_dir"] = run_dir
    res = RP.extract_day(snap, 0, out, rep, make_rollup=True)
    assert res["rollup"] is not None, res["notes"]
    html = res["rollup"].read_text(encoding="utf-8")
    assert "__DATA__" not in html and "__RUN__" not in html
    assert not re.findall(r"(?:src|href)=[\"']https?://", html), "外部 URL がある"
    assert (res["dir"] / "l2_metrics.parquet").is_file()
    assert (res["dir"] / "digest.json").is_file()
    summary = json.loads((res["dir"] / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_agents"] == 100 and summary["n_steps"] == 300


@pytest.mark.xdist_group("subprocess_viewer")
def test_v2_start_min_is_the_true_value_not_the_0700_fallback(tmp_path):
    """★start_min が 07:00 フォールバックでなく**ランの真値**になっている。"""
    run_dir = _make_run(tmp_path, n_steps=300, start_tod="05:30")
    out = tmp_path / "o"
    rep = _rep(out)
    cfg = _cfg(run_dir, out)
    snap = RP.build_snapshot(cfg, None)
    snap["_run_dir"] = run_dir
    assert snap["start_min"] == 5 * 60 + 30
    res = RP.extract_day(snap, 0, out, rep, make_rollup=True)
    html = res["rollup"].read_text(encoding="utf-8")
    m = re.search(r'"startMin":\s*(\d+)', html)
    assert m and int(m.group(1)) == 330, f"07:00 へ落ちている: {m and m.group(1)}"
    digest = json.loads((res["dir"] / "digest.json").read_text(encoding="utf-8"))
    assert digest["provenance"]["start_min"] == 330
    assert "l1_events.parquet" in digest["provenance"]["shims"]
    assert (res["dir"] / "_SHIMS.txt").is_file(), "合成物であることを明記していない"


@pytest.mark.xdist_group("subprocess_viewer")
def test_v3_extract_cli_writes_only_under_out_dir(tmp_path):
    """--extract は投稿せず、出力は <out>/day-NN/ の下だけ。"""
    run_dir = _make_run(tmp_path, n_steps=300)
    before = {p.name for p in run_dir.iterdir()}
    assert RP.main([str(run_dir), "--extract", "--day", "1", "--quiet"]) == 0
    assert {p.name for p in run_dir.iterdir()} - before == {"_progress"}
    day = run_dir / "_progress" / "day-01"
    assert (day / "digest.json").is_file()
    assert not list((run_dir / "_progress").glob("dryrun/*.json"))


def test_v4_digest_json_separates_confirmed_and_provisional(tmp_path):
    """digest.json は確報と速報を**別の節**に分ける(同じキーに混ぜない)。"""
    run_dir = _make_run(tmp_path, n_steps=300)
    out = tmp_path / "o"
    rep = _rep(out)
    cfg = _cfg(run_dir, out)
    snap = RP.build_snapshot(cfg, None)
    snap["_run_dir"] = run_dir
    res = RP.extract_day(snap, 0, out, rep, make_rollup=False)
    d = json.loads((res["dir"] / "digest.json").read_text(encoding="utf-8"))
    assert set(d) >= {"confirmed", "provisional", "provenance"}
    assert d["confirmed"]["definition"] != d["provisional"]["definition"]
    assert d["provenance"]["parts_read"]
    assert d["provenance"]["checkpoint_step"] == 144


# ============================================================ E 欠測
def test_e1_missing_status_json(tmp_path):
    """status.json が無くても落ちず・捏造しない。"""
    run_dir = _make_run(tmp_path, n_steps=300, with_status=False)
    res, sink, state = _cycle(run_dir, tmp_path / "o", no_attach=True)
    snap = res["snapshot"]
    assert snap["state"] is None and snap["has_status"] is False
    blob = _all_text(*[p["payload"] for p in sink.posts + sink.patches])
    assert "不明" in blob


def test_e2_corrupt_status_json(tmp_path):
    """壊れた status.json は None 扱い(例外を出さない)。"""
    run_dir = _make_run(tmp_path, n_steps=60)
    (run_dir / "status.json").write_text("{ not json", encoding="utf-8")
    assert RP.read_status(run_dir) is None


def test_e3_missing_columns_show_as_unmeasured(tmp_path):
    """L2 に列が無い指標は 0 で埋めず「—」= 測れなかったと出す。"""
    agg = {"day": 0, "n_rows": 3, "metrics": {"n_moving": {"mean": 12.0, "last": 12.0}}}
    text = RP._metric_block(RP.CITY_METRICS, agg, None)
    assert "移動中" in text and "屋内" not in text
    assert RP._fmt_val(None, "int") == "—"
    assert RP._fmt_delta(None, 1.0, "int") == ""


def test_e4_pyarrow_absent(tmp_path, monkeypatch):
    """pyarrow 不在でも落ちず、「読めなかった」と出す。"""
    run_dir = _make_run(tmp_path, n_steps=60)
    real_import = __import__

    def no_pyarrow(name, *a, **k):
        if name.startswith("pyarrow"):
            raise ImportError("no pyarrow")
        return real_import(name, *a, **k)

    monkeypatch.setattr("builtins.__import__", no_pyarrow)
    res = RP.read_l2(run_dir, columns=RP.REPORT_COLUMNS)
    assert res["available"] is False and res["rows"] == []
    assert any("pyarrow" in n for n in res["notes"])


def test_e5_run_dir_does_not_exist(tmp_path):
    """存在しない run-dir でも終了コード 0。"""
    assert RP.main([str(tmp_path / "nope"), "--dry-run", "--quiet"]) == 0


def test_e6_exception_inside_cycle_is_swallowed(tmp_path, monkeypatch):
    """サイクル内で何が起きても例外を外へ出さない(P5)。"""
    run_dir = _make_run(tmp_path, n_steps=60)

    def boom(*a, **k):
        raise RuntimeError("内部で爆発")

    monkeypatch.setattr(RP, "build_snapshot", boom)
    assert RP.main([str(run_dir), "--dry-run", "--quiet"]) == 0
    log = (run_dir / "_progress" / RP.LOG_NAME).read_text(encoding="utf-8")
    assert "サイクルで例外" in log


def test_e7_state_survives_round_trip_and_bad_file(tmp_path):
    """状態は原子的に保存され、壊れていれば既定へ戻る(二重起動でも壊れない)。"""
    out = tmp_path / "o"
    out.mkdir()
    state = RP.load_state(out)
    state["posted_days"] = [0, 1]
    RP.save_state(out, state)
    assert RP.load_state(out)["posted_days"] == [0, 1]
    (out / RP.STATE_NAME).write_text("garbage", encoding="utf-8")
    assert RP.load_state(out)["posted_days"] == []


def test_e8_days_are_posted_once(tmp_path):
    """同じシミュ日を二度投稿しない(状態を跨いで冪等)。"""
    run_dir = _make_run(tmp_path, n_steps=300)
    out = tmp_path / "o"
    state = RP.load_state(out)
    _, sink1, _ = _cycle(run_dir, out, state=state, no_attach=True)
    n1 = len([p for p in sink1.posts if p["kind"].startswith("daily")])
    _, sink2, _ = _cycle(run_dir, out, state=state, no_attach=True)
    n2 = len([p for p in sink2.posts if p["kind"].startswith("daily")])
    # n_steps=300 / start 07:00 → day0(step 0-101)と day1(102-245)が完了・day2 は途中
    assert n1 == 2 and n2 == 0
    assert sorted(state["posted_days"]) == [0, 1]


def test_e9_heartbeat_is_edited_not_reposted(tmp_path):
    """ハートビートは 1 通を編集し続ける(チャンネルを汚さない)。"""
    run_dir = _make_run(tmp_path, n_steps=300)
    out = tmp_path / "o"
    state = RP.load_state(out)
    sink = _Sink(_rep(out))
    _cycle(run_dir, out, sink=sink, state=state, no_attach=True)
    hb_posts = [p for p in sink.posts if p["kind"] == "heartbeat"]
    assert len(hb_posts) == 1
    state["heartbeat_ts"] = 0.0                     # 間隔を跨いだことにする
    _cycle(run_dir, out, sink=sink, state=state, no_attach=True)
    assert len([p for p in sink.posts if p["kind"] == "heartbeat"]) == 1
    assert len(sink.patches) == 1
