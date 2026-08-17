"""レーンG = 集合イベントの内生観測(案A 事後検知器 + 案B 意図収束サイドカー)。

正典: docs/research/emergent-events-and-narrative-ui.md §3(案A / 案B)。
実装: ``src/society/observer/gathering.py`` + ``scripts/detect_gatherings.py``。

**共通の受入条件**(R1 = 世界を 1 バイトも動かさない):
  (a) 既定 OFF ではオブジェクトも生えず 1 ファイルも書かない
  (b) **ON でも L1/L1b/L2/L3 はバイト一致**(新 kind を 1 つも足していないので、
      OFF↔ON の比較で L1 が変わったら即座に落ちる)
  (c) 事後スクリプトはランを 1 バイトも書き換えない(入力の mtime/バイトを固定)
  (d) 決定論(同一入力 → 同一出力・2 回走らせてバイト一致)
全経路 mock(実 LLM 不使用)。
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

import detect_gatherings as DG                          # noqa: E402
from society import registry as R                       # noqa: E402
from society.config import load_config                  # noqa: E402
from society.engine.simulation import Simulation        # noqa: E402
from society.observer import gathering as OG            # noqa: E402
from society.observer import schema as OS               # noqa: E402

ON = {"observer.gathering_intent.enabled": "true"}
#: 集合が実際に生まれる構成(イベント開催 + 予定抽出 + 朝の計画)。
WORLD = {"schedule.enabled": "true", "planning.day_plan.enabled": "true"}


def _dots(name, n_steps, n_agents, **ov):
    dot = ["run.seed=42", f"run.n_agents={n_agents}", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock"]
    return dot + [f"{k}={v}" for k, v in ov.items()]


def _sim(tmp_path, name, n_steps=24, n_agents=12, **ov):
    return Simulation(load_config(_dots(name, n_steps, n_agents, **ov)),
                      out_dir=tmp_path / name)


def _run(tmp_path, name, n_steps=180, n_agents=30, **ov):
    sim = _sim(tmp_path, name, n_steps, n_agents, **ov)
    sim.run()
    return sim.out_dir


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Agent:
    """観測層が読む属性だけを持つ最小スタブ(観測は sim.agents の既存欄しか読まない)。"""

    def __init__(self, aid, *, schedule=None, dayplan=None, known=(),
                 attending=None):
        self.id = aid
        self.schedule = list(schedule or ())
        self._dayplan = dayplan
        self._known_events = set(known)
        self._attending_event = attending


class _Clock:
    def __init__(self, dt=10, start=0):
        self.dt, self.start = dt, start

    def sim_min(self, step):
        return self.start + int(step) * self.dt


class _Tools:
    def __init__(self, events):
        self.events = dict(events)


class _Sim:
    def __init__(self, agents, events=None, clock=None):
        self.agents = list(agents)
        self.tools = _Tools(events or {})
        self.clock = clock or _Clock()


# =========================================================================== #
# 0. 宣言と契約(既定 OFF・レジストリ・凍結・新 kind ゼロ)
# =========================================================================== #
def test_toggle_is_declared_in_the_registry():
    feats = {f.id: f for f in R.FEATURES}
    fid = "observer.gathering_intent.enabled"
    assert fid in feats, f"{fid} が registry に宣言されていない"
    f = feats[fid]
    assert f.off_value is False
    assert f.affects_k is False          # generate() の呼び出し点を 1 つも足さない
    assert f.repro_tier == "strict"      # LLM の自由文を 1 バイトも読まない
    assert f.fingerprint_risk == "none"  # プロンプトを 1 バイトも変えない


def test_no_undeclared_toggles():
    assert R.undeclared_toggles(load_config()) == []


def test_defaults_are_off():
    cfg = load_config()
    assert OG.cfg_of_config(cfg)["enabled"] is False
    assert OG.cfg_of_config({})["enabled"] is False       # 旧 config でも OFF へ落ちる


def test_build_cfg_accepts_dictconfig_not_only_dict():
    """★roster.py が踏んだ穴の再発防止: DictConfig は dict の部分型ではない。"""
    from omegaconf import OmegaConf
    raw = OmegaConf.create({"enabled": True, "slot_min": 60, "capture_min": 300,
                            "repeat_min": 0, "min_intent": 5, "sample_cap": 3})
    got = OG.build_cfg(raw)
    assert got == {"enabled": True, "slot_min": 60, "capture_min": 300,
                   "repeat_min": 0, "min_intent": 5, "sample_cap": 3}


def test_module_adds_no_new_l1_kind():
    """★本層は ``register_event_kind`` を 1 度も**呼ばない**(ON でも L1 の kind 集合が不変)。

    判定は AST(呼び出し式)で行う: 文字列検索だと docstring で設計意図を説明した
    だけで落ちてしまい、「説明を書くと失敗するテスト」になってしまう。
    """
    import ast
    src = (_ROOT / "src" / "society" / "observer" / "gathering.py").read_text(
        encoding="utf-8")
    calls = {getattr(n.func, "id", None) or getattr(n.func, "attr", None)
             for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call)}
    assert "register_event_kind" not in calls, \
        "案B はサイドカーのみ = 新 L1 kind を足さない設計"
    for kind in OS.EVENT_KINDS:
        assert not kind.startswith("gathering"), f"新 kind が生えている: {kind}"
        assert not kind.startswith("assembly_"), f"新 kind が生えている: {kind}"


def test_module_is_not_in_the_frozen_spec():
    """凍結 14 ファイル(metrics_spec)に 1 本も触れていない = spec hash 無風。"""
    from society.observer import metrics_spec as MS
    for rel in ("src/society/observer/gathering.py",
                "scripts/detect_gatherings.py"):
        assert rel not in MS.SPEC_FILES


def test_off_creates_no_object_and_no_file(tmp_path):
    sim = _sim(tmp_path, "g_off", n_steps=2, n_agents=6)
    assert sim.gathering_sc is None
    sim.run()
    assert not (sim.out_dir / "gathering_intent.parquet").exists()


# =========================================================================== #
# 1. 観測不変性(受入 b): ON でも L1 はバイト一致
# =========================================================================== #
@pytest.mark.slow
def test_on_leaves_every_log_byte_identical(tmp_path):
    a = _run(tmp_path, "inv_off", n_steps=120, n_agents=24, **WORLD)
    b = _run(tmp_path, "inv_on", n_steps=120, n_agents=24, **WORLD, **ON)
    for name in ("l1_events.parquet", "l1b_llm.parquet",
                 "l2_metrics.parquet", "l3_snapshots.parquet"):
        assert _sha(a / name) == _sha(b / name), \
            f"{name} が ON/OFF で変わった(観測がシムを動かしている)"
    assert (b / "gathering_intent.parquet").exists(), "ON なのに 1 枚も出ていない"
    assert not (a / "gathering_intent.parquet").exists()


@pytest.mark.slow
def test_on_is_deterministic_across_two_runs(tmp_path):
    a = _run(tmp_path, "det_a", n_steps=120, n_agents=24, **WORLD, **ON)
    b = _run(tmp_path, "det_b", n_steps=120, n_agents=24, **WORLD, **ON)
    ta = pq.read_table(a / "gathering_intent.parquet")
    tb = pq.read_table(b / "gathering_intent.parquet")
    assert ta.to_pydict() == tb.to_pydict()


# =========================================================================== #
# 2. 撮影スケジュール(純関数)= resume==straight の根拠
# =========================================================================== #
def test_last_scheduled_is_a_pure_function_of_time(tmp_path):
    sc = OG.GatheringIntent(tmp_path, capture_min=600, repeat_min=180)
    assert sc.last_scheduled(0) is None            # 初回より前は「まだ無い」
    assert sc.last_scheduled(599) is None
    assert sc.last_scheduled(600) == 600
    assert sc.last_scheduled(779) == 600
    assert sc.last_scheduled(780) == 780
    # 翌日の 0:00 は「前日最後の撮影(600+4*180=1320)」が直近
    assert sc.last_scheduled(1440) == 1320
    assert sc.last_scheduled(1440 + 600) == 1440 + 600


def test_last_scheduled_once_per_day_when_repeat_is_zero(tmp_path):
    sc = OG.GatheringIntent(tmp_path, capture_min=600, repeat_min=0)
    assert sc.last_scheduled(600) == 600
    assert sc.last_scheduled(1439) == 600
    assert sc.last_scheduled(1440 + 599) == 600
    assert sc.last_scheduled(1440 + 600) == 1440 + 600


def test_resume_at_reproduces_the_straight_run_state(tmp_path):
    """resume の据え直しと走行中の判定が**同じ純関数**を通る(行集合が食い違わない)。"""
    sim = _Sim([_Agent(1)])
    straight = OG.GatheringIntent(tmp_path / "s", capture_min=600, repeat_min=180)
    for smin in range(0, 2000, 10):
        straight.on_step(sim, smin // 10, smin)
    resumed = OG.GatheringIntent(tmp_path / "r", capture_min=600, repeat_min=180)
    resumed.resume_at(1490)
    for smin in range(1500, 2000, 10):
        resumed.on_step(sim, smin // 10, smin)
    assert resumed._last_target == straight._last_target


def test_capture_fires_once_per_scheduled_time(tmp_path):
    plan = {"day": 0, "blocks": [{"start": 660, "node": "nX", "place": "food"}]}
    sim = _Sim([_Agent(1, dayplan=plan), _Agent(2, dayplan=plan)])
    sc = OG.GatheringIntent(tmp_path, capture_min=600, repeat_min=180,
                            min_intent=2)
    fired = [sc.on_step(sim, m // 10, m) for m in range(590, 800, 10)]
    assert sum(1 for f in fired if f) == 2, "600 と 780 の 2 回だけ撮る"
    assert sc._n_captures == 2


# =========================================================================== #
# 3. セル集計(案B の中身)
# =========================================================================== #
def test_plan_block_node_is_captured_because_l1_cannot_carry_it(tmp_path):
    """★本サイドカー最大の存在理由: 計画ブロックの**解決済みノード**は L1 に出ない。"""
    plan = {"day": 3, "blocks": [{"start": 19 * 60, "node": "nHub",
                                  "place": "food", "act": "meal"}]}
    sim = _Sim([_Agent(i, dayplan=plan) for i in (5, 2, 9)])
    cells = OG._Cells()
    OG.collect(sim, cells, 30)
    rows = cells.rows(3, 600, 30, 2, 32)
    assert len(rows) == 1
    (cap_day, cap_min, day, wbin, kind, place, n_intent, n_appt, n_plan,
     n_event, event_ids, lead_min, sample_ids) = rows[0]
    assert (day, wbin, kind, place) == (3, 38, OG.KIND_NODE, "nHub")
    assert (n_intent, n_plan, n_appt, n_event) == (3, 3, 0, 0)
    assert json.loads(sample_ids) == [2, 5, 9]        # id 昇順(走査順は漏れない)
    assert lead_min == 3 * 1440 + 19 * 60 - 600


def test_unresolved_plan_block_falls_back_to_category_not_a_guess(tmp_path):
    plan = {"day": 0, "blocks": [{"start": 600, "node": None, "place": "street"}]}
    sim = _Sim([_Agent(1, dayplan=plan), _Agent(2, dayplan=plan)])
    cells = OG._Cells()
    OG.collect(sim, cells, 30)
    rows = cells.rows(0, 600, 30, 2, 32)
    assert rows[0][4] == OG.KIND_CATEGORY and rows[0][5] == "street"


def test_appointment_rows_use_the_raw_place_label(tmp_path):
    appt = {"day": 1, "when": "19:00", "what": "会う", "place": "公園",
            "with": [2]}
    sim = _Sim([_Agent(1, schedule=[appt]), _Agent(2, schedule=[appt])])
    cells = OG._Cells()
    OG.collect(sim, cells, 30)
    rows = cells.rows(0, 600, 30, 2, 32)
    assert len(rows) == 1
    assert rows[0][2:7] == (1, 38, OG.KIND_LABEL, "公園", 2)


def test_appointment_without_a_clock_lands_in_the_unknown_bin(tmp_path):
    appt = {"day": 1, "when": "", "place": "", "with": []}
    sim = _Sim([_Agent(1, schedule=[appt]), _Agent(2, schedule=[appt])])
    cells = OG._Cells()
    OG.collect(sim, cells, 30)
    rows = cells.rows(0, 600, 30, 2, 32)
    assert rows[0][3] == -1, "時刻不明は -1(0 時と偽らない)"


def test_band_words_are_resolved_through_schedule_module():
    """時間帯語 → 分の表は society/schedule.py が単一の源(本 module に語を書かない)。"""
    from society import schedule as S
    for word, want in S._BAND_MIN.items():
        if not word:
            continue
        assert OG.when_minutes(word) == want
    assert OG.when_minutes("") is None
    assert OG.when_minutes("07:30") == 450


def test_known_events_are_the_denominator_l1_never_records(tmp_path):
    """★イベントの**認知集合**は L1 に出ない(出るのは開催宣言と到着だけ)。"""
    events = {7: {"id": 7, "node": "nSquare", "start_step": 90, "ended": False}}
    sim = _Sim([_Agent(1, known=[7]), _Agent(2, known=[7]),
                _Agent(3, attending=7), _Agent(4)],
               events=events, clock=_Clock(dt=10, start=0))
    cells = OG._Cells()
    OG.collect(sim, cells, 30)
    rows = cells.rows(0, 600, 30, 2, 32)
    assert len(rows) == 1
    assert rows[0][4:10] == (OG.KIND_NODE, "nSquare", 3, 0, 0, 3)
    assert json.loads(rows[0][10]) == [7]
    assert rows[0][3] == 900 // 30                    # start_step 90 → 900 分 → bin30


def test_ended_events_are_skipped(tmp_path):
    events = {7: {"id": 7, "node": "n1", "start_step": 90, "ended": True}}
    sim = _Sim([_Agent(1, known=[7]), _Agent(2, known=[7])], events=events)
    cells = OG._Cells()
    OG.collect(sim, cells, 30)
    assert cells.rows(0, 600, 30, 2, 32) == []


def test_min_intent_filters_and_sample_cap_truncates_only_the_sample(tmp_path):
    plan = {"day": 0, "blocks": [{"start": 600, "node": "nA", "place": "food"}]}
    sim = _Sim([_Agent(i, dayplan=plan) for i in range(10)])
    cells = OG._Cells()
    OG.collect(sim, cells, 30)
    assert cells.rows(0, 600, 30, 11, 32) == [], "min_intent 未満は行にしない"
    row = cells.rows(0, 600, 30, 2, 3)[0]
    assert row[6] == 10, "n_intent は切り詰めない(臨界まであと何人が読めなくなる)"
    assert json.loads(row[12]) == [0, 1, 2], "sample だけが cap で切れる"


def test_wrapping_plan_block_lands_on_the_next_day(tmp_path):
    """DPH-C の日跨ぎブロック(start>=1440)は翌日のセルへ落ちる。"""
    plan = {"day": 4, "blocks": [{"start": 1440 + 60, "node": "nN", "place": "food"}]}
    sim = _Sim([_Agent(1, dayplan=plan), _Agent(2, dayplan=plan)])
    cells = OG._Cells()
    OG.collect(sim, cells, 30)
    row = cells.rows(4, 600, 30, 2, 32)[0]
    assert (row[2], row[3]) == (5, 2)


def test_collect_writes_nothing_back_to_the_world(tmp_path):
    """★観測層は個体へ属性を 1 つも書かない(checkpoint のバイト列を動かさない)。"""
    plan = {"day": 0, "blocks": [{"start": 600, "node": "nA", "place": "food"}]}
    agents = [_Agent(1, dayplan=plan), _Agent(2, dayplan=plan)]
    before = [dict(vars(a)) for a in agents]
    plan_snapshot = json.dumps(plan, sort_keys=True)
    sim = _Sim(agents)
    OG.collect(sim, OG._Cells(), 30)
    assert [dict(vars(a)) for a in agents] == before
    assert json.dumps(plan, sort_keys=True) == plan_snapshot


# =========================================================================== #
# 4. サイドカーの入出力(part / finalize / 空でも 1 枚)
# =========================================================================== #
def test_empty_capture_still_writes_a_file_to_distinguish_off(tmp_path):
    """「ON だが 0 セル」と「OFF」はファイルの有無で区別できなければならない。"""
    sim = _Sim([_Agent(1)])
    sc = OG.GatheringIntent(tmp_path, capture_min=0, repeat_min=0)
    sc.on_step(sim, 0, 0)
    path = sc.finalize()
    assert path is not None and path.exists()
    assert pq.read_table(path).num_rows == 0


def test_flush_segment_and_finalize_round_trip(tmp_path):
    plan = {"day": 0, "blocks": [{"start": 600, "node": "nA", "place": "food"}]}
    sim = _Sim([_Agent(1, dayplan=plan), _Agent(2, dayplan=plan)])
    sc = OG.GatheringIntent(tmp_path, capture_min=0, repeat_min=180)
    sc.on_step(sim, 0, 0)
    sc.flush_segment()
    sc.on_step(sim, 18, 180)
    path = sc.finalize()
    tbl = pq.read_table(path)
    assert tbl.num_rows == 2                       # 2 回の撮影 × 1 セル
    assert sorted(tbl.column("cap_min").to_pylist()) == [0, 180]
    assert not list(Path(tmp_path).glob("gathering_intent.part-*.parquet"))


@pytest.mark.slow
def test_resume_equals_straight_for_the_sidecar(tmp_path):
    """★分割実行の行集合が一気通しと一致する(tests/test_resume.py と同じ様式)。

    撮影は日境界ではなく ``capture_min`` 起点なので、resume の据え直しを誤ると
    「同じ時刻をもう一度撮る」「撮り損ねる」のどちらかが起きる。
    """
    from society.engine import checkpoint, scheduler

    # ★2 日ぶん回す: day0 は朝の計画予約 step がラン開始(7:00)より前なので計画が
    #   1 件も立たず、意図セルがほぼ出ない(= 分割点の前後に行が揃わない)。
    total, split, n = 288, 200, 24
    # min_intent=1 = 「1 人の予定でも 1 行」= 行が必ず出る構成にする(空同士の一致で
    # 通ってしまうテストは何も守らないため)。
    ov = dict(WORLD, **ON, **{"observer.gathering_intent.min_intent": "1"})
    straight = _run(tmp_path, "rs_straight", n_steps=total, n_agents=n, **ov)
    d = tmp_path / "rs_resumed"
    ev = {"observer.checkpoint_every": split}
    sim1 = Simulation(load_config(_dots("rs_resumed", split, n, **ov, **ev)),
                      out_dir=d)
    for step in range(split):
        scheduler.run_step(sim1, step)
    checkpoint.save(sim1, split, d / "checkpoint" / f"ckpt-{split:06d}.pkl.gz")
    sim1.logger.flush_segment()
    sim1.gathering_sc.flush_segment()
    sim2 = Simulation(load_config(_dots("rs_resumed", total, n, **ov, **ev)),
                      out_dir=d)
    sim2.run(resume_from=d)
    a = pq.read_table(straight / "gathering_intent.parquet").to_pylist()
    b = pq.read_table(d / "gathering_intent.parquet").to_pylist()
    # ★空同士の一致で通ってしまわないよう、分割点の**前後どちらにも**撮影がある
    #   構成であることを先に固定する(そうでないと本テストは何も守らない)。
    caps = {r["cap_min"] for r in a}
    split_min = int(sim2.clock.sim_min(split))
    assert any(c < split_min for c in caps) and any(c >= split_min for c in caps), \
        f"分割点 {split_min} を跨ぐ撮影が無い構成では resume を検証できない: {sorted(caps)}"
    assert a == b, "resume != straight(撮影時刻の据え直しが誤っている)"


@pytest.mark.slow
def test_streaming_finalize_conf_reaches_the_sidecar(tmp_path):
    """W4-E: 1 つの conf キーが本サイドカーにも届く(配り漏れの検出)。"""
    sim = _sim(tmp_path, "wire", n_steps=1, n_agents=6, **ON,
               **{"observer.finalize.streaming": "true",
                  "observer.finalize.row_group_rows": "7"})
    assert sim.gathering_sc is not None
    assert sim.gathering_sc.streaming_finalize is True
    assert sim.gathering_sc.finalize_row_group_rows == 7


# =========================================================================== #
# 5. 案A 事後検知器(合成 L1 で検出できることを固定する)
# =========================================================================== #
def _synth_run(tmp_path, name, rows, *, dt_min=10):
    """最小の L1 parquet を書く(``measure`` の列定義に合わせる)。"""
    import pyarrow as pa
    out = tmp_path / name
    out.mkdir(parents=True, exist_ok=True)
    cols = {
        "step": pa.array([r[0] for r in rows], pa.int32()),
        "sim_min": pa.array([r[1] for r in rows], pa.int32()),
        "agent_id": pa.array([r[2] for r in rows], pa.int32()),
        "kind": pa.array([r[3] for r in rows], pa.string()),
        "x": pa.array([0.0] * len(rows), pa.float64()),
        "y": pa.array([0.0] * len(rows), pa.float64()),
        "payload": pa.array([json.dumps(r[4], ensure_ascii=False) for r in rows],
                            pa.string()),
    }
    pq.write_table(pa.table(cols), out / "l1_events.parquet")
    (out / "config.yaml").write_text(
        f"run:\n  dt_min: {dt_min}\n  start_min: 0\n", encoding="utf-8")
    return out


def _timeline(n_normal, n_burst, burst_min, node="nPark", other="nHome"):
    """2 日ぶんの合成 L1: 同じ場所・同じ時間帯に普段は n_normal 人、1 日目だけ n_burst 人。"""
    rows = []
    for day in (0, 1):
        for m in range(0, 1440, 10):
            step = (day * 1440 + m) // 10
            smin = day * 1440 + m
            n = (n_burst if (day == 1 and burst_min <= m < burst_min + 60)
                 else n_normal)
            for aid in range(n):
                rows.append((step, smin, aid, "arrive", {"node": node}))
            for aid in range(n, n_burst):
                rows.append((step, smin, aid, "arrive", {"node": other}))
    return rows


def test_detector_finds_a_burst_against_the_daily_baseline(tmp_path):
    run = _synth_run(tmp_path, "burst", _timeline(2, 12, 20 * 60))
    res = DG.analyze(str(run), slot_min=30, min_n=6, ratio=3.0, dur=1)
    assert res["coverage"]["baseline_src"] == "cross_day"
    got = res["gatherings"]
    assert got, "明白な集合(2人 → 12人)を検出できていない"
    top = got[0]
    assert top["node"] == "nPark"
    assert top["n_peak"] == 12
    assert top["day"] == 1 and top["clock"] == "20:00"
    assert top["via"] == "none", "誰も企画していない集合は via=none になる"
    assert top["n_members"] == 12


def test_detector_is_silent_without_a_burst(tmp_path):
    run = _synth_run(tmp_path, "flat", _timeline(12, 12, 20 * 60))
    res = DG.analyze(str(run), slot_min=30, min_n=6, ratio=3.0, dur=1)
    assert res["gatherings"] == [], "定常の混雑を集合と呼んではいけない"


def test_detector_ignores_agents_in_transit_and_asleep(tmp_path):
    """移動中(move_segment)と就寝中は同時在場に数えない(通り過ぎ=集合ではない)。"""
    rows = []
    for m in range(0, 1440, 10):
        for aid in range(20):
            rows.append((m // 10, m, aid, "arrive", {"node": "nPark"}))
            rows.append((m // 10, m, aid,
                         "move_segment" if aid % 2 else "sleep_start", {}))
    run = _synth_run(tmp_path, "ghost", rows)
    res = DG.analyze(str(run), slot_min=30, min_n=1, ratio=1.0, dur=1)
    assert res["coverage"]["occupied_cells"] == 0, \
        "移動中/就寝中を在場に数えている"


def test_detector_attributes_a_hosted_event(tmp_path):
    rows = _timeline(2, 10, 20 * 60)
    rows.append((0, 0, 0, "event_host",
                 {"event_id": 1, "node": "nPark", "start_step": 120,
                  "title": "t"}))
    for aid in range(10):                              # 20:00 = step 2040/… に到着
        rows.append(((1440 + 20 * 60) // 10, 1440 + 20 * 60, aid, "event_attend",
                     {"event_id": 1, "host": 0}))
    run = _synth_run(tmp_path, "hosted", rows)
    res = DG.analyze(str(run), slot_min=30, min_n=6, ratio=3.0, dur=1)
    assert res["gatherings"][0]["via"] == "event"
    assert res["gatherings"][0]["refs"]["event_ids"] == [1]


def test_detector_output_is_deterministic(tmp_path):
    run = _synth_run(tmp_path, "det", _timeline(2, 12, 20 * 60))
    a = DG.analyze(str(run), slot_min=30, min_n=6, ratio=3.0, dur=1)
    b = DG.analyze(str(run), slot_min=30, min_n=6, ratio=3.0, dur=1)
    assert json.dumps(a, sort_keys=True, ensure_ascii=False) == \
        json.dumps(b, sort_keys=True, ensure_ascii=False)


def test_script_never_writes_back_into_the_run(tmp_path):
    """受入 (c): 事後スクリプトは入力を 1 バイトも書き換えない。"""
    run = _synth_run(tmp_path, "ro", _timeline(2, 12, 20 * 60))
    l1 = run / "l1_events.parquet"
    before = (_sha(l1), l1.stat().st_mtime_ns)
    out = tmp_path / "report"
    DG.main(["--run", str(run), "--min-n", "6", "--ratio", "3.0",
             "--out", str(out), "--quiet"])
    assert (_sha(l1), l1.stat().st_mtime_ns) == before
    assert (out / "gatherings.json").exists()
    assert (out / "gatherings_report.md").exists()
    assert not (run / "gatherings.json").exists()


def test_slot_min_must_divide_the_day(tmp_path):
    run = _synth_run(tmp_path, "badslot", _timeline(2, 12, 20 * 60))
    with pytest.raises(SystemExit):
        DG.analyze(str(run), slot_min=7)


# =========================================================================== #
# 6. 意図 → 実現 / 不発(案A × 案B の突合)
# =========================================================================== #
def _cell(day, wbin, place, n, kind="node"):
    return {"day": day, "when_bin": wbin, "place_kind": kind, "place": place,
            "n_intent": n, "n_appointment": 0, "n_plan": n, "n_event": 0,
            "event_ids": [], "sample_ids": [], "src": "sidecar"}


def test_unrealized_cells_carry_how_many_more_were_needed(tmp_path):
    run = _synth_run(tmp_path, "gap", _timeline(2, 12, 20 * 60))
    res = DG.analyze(str(run), slot_min=30, min_n=6, ratio=3.0, dur=1)
    occ = None                                          # analyze は occ を返さない
    # 突合そのものを純関数として直接叩く(analyze の内部と同じ引数)。
    mats = DG.scan_pass1(str(run), 30, 10)
    occ = mats["occ"]
    baseline, _mean, _src = DG.build_baselines(occ, 48)
    got = DG.reconcile([_cell(0, 40, "nPark", 9)], [], occ, baseline, 48,
                       min_n=6, min_intent=2, ratio=3.0, floor=0.5, slack=1)
    assert len(got) == 1
    row = got[0]
    assert row["realized"] is False
    assert row["n_intent"] == 9 and row["n_present"] == 2
    assert row["short_by"] == 4, "臨界 6 人に対し 2 人 = あと 4 人"
    assert row["fail_reason"] == "headcount"


def test_realized_cells_carry_the_arrival_rate(tmp_path):
    run = _synth_run(tmp_path, "hit", _timeline(2, 12, 20 * 60))
    mats = DG.scan_pass1(str(run), 30, 10)
    occ = mats["occ"]
    baseline, _m, _s = DG.build_baselines(occ, 48)
    gs = DG.detect_gatherings(occ, baseline, 48, min_n=6, ratio=3.0,
                              ratio_end=1.8, dur=1, floor=0.5)
    DG.scan_participants(str(run), gs, 30, 200)
    got = DG.reconcile([_cell(1, 40, "nPark", 24)], gs, occ, baseline, 48,
                       min_n=6, min_intent=2, ratio=3.0, floor=0.5, slack=1)
    assert got[0]["realized"] is True
    assert got[0]["arrival_rate"] == round(12 / 24, 3)


def test_non_node_cells_are_never_matched(tmp_path):
    """場所名/カテゴリのセルはノードへ束縛していないので突合しない(推測しない)。"""
    run = _synth_run(tmp_path, "nonnode", _timeline(2, 12, 20 * 60))
    mats = DG.scan_pass1(str(run), 30, 10)
    baseline, _m, _s = DG.build_baselines(mats["occ"], 48)
    cells = [_cell(1, 40, "公園", 9, kind="label"),
             _cell(1, 40, "street", 9, kind="category")]
    assert DG.reconcile(cells, [], mats["occ"], baseline, 48, min_n=6,
                        min_intent=2, ratio=3.0, floor=0.5) == []


def test_merge_cells_keeps_the_peak_intent():
    """1 日数回撮るので同じセルが複数回出る → n_intent 最大の行を採る。"""
    a = [_cell(1, 40, "nA", 3), _cell(1, 40, "nA", 9)]
    b = [_cell(1, 41, "nB", 4)]
    got = DG.merge_cells(a, b)
    assert len(got) == 2
    assert [c["n_intent"] for c in got if c["place"] == "nA"] == [9]


# =========================================================================== #
# 7. 種の系譜(伝播木を transmission から再構成できる範囲だけ)
# =========================================================================== #
def test_genealogy_reconstructs_the_transmission_tree(tmp_path):
    rows = _timeline(2, 12, 20 * 60)
    # 0 → 1 → 2 の 2 hop + 0 → 3 の枝(集合の前に届く)
    for step, frm, to in ((10, 0, 1), (20, 1, 2), (30, 0, 3)):
        rows.append((step, step * 10, to, "transmission",
                     {"item_id": "w-1", "from": frm, "channel": "face"}))
    run = _synth_run(tmp_path, "tree", rows)
    res = DG.analyze(str(run), slot_min=30, min_n=6, ratio=3.0, dur=1,
                     genealogy_top=5)
    assert res["genealogy"], "参加者を結ぶ item の伝播木が出ていない"
    item = res["genealogy"][0]["items"][0]
    assert item["item_id"] == "w-1"
    assert item["holders_in_gathering"] == 3
    assert item["tree_depth"] == 2
    assert item["channels"] == ["face"]


def test_genealogy_skips_items_held_by_a_single_participant(tmp_path):
    rows = _timeline(2, 12, 20 * 60)
    rows.append((10, 100, 1, "transmission",
                 {"item_id": "w-solo", "from": 0, "channel": "sns"}))
    run = _synth_run(tmp_path, "solo", rows)
    res = DG.analyze(str(run), slot_min=30, min_n=6, ratio=3.0, dur=1)
    for entry in res["genealogy"]:
        assert all(it["item_id"] != "w-solo" for it in entry["items"])


def test_tree_depth_tolerates_cycles():
    assert DG._tree_depth({1: 2, 2: 1}) <= 2           # 閉路で無限ループしない
    assert DG._tree_depth({}) == 0


# =========================================================================== #
# 8. 端から端まで(mock ラン → サイドカー → 検知器 → 突合)
# =========================================================================== #
@pytest.mark.slow
def test_end_to_end_on_a_mock_run(tmp_path):
    run = _run(tmp_path, "e2e", n_steps=288, n_agents=40, **WORLD, **ON)
    assert (run / "gathering_intent.parquet").exists()
    res = DG.analyze(str(run), slot_min=30, min_n=4, ratio=2.0, dur=1)
    assert res["coverage"]["intent_source"] == "sidecar"
    assert res["coverage"]["sidecar_rows"] > 0
    assert res["gatherings"], "mock 2 日ランで 1 件も集合が検出されない"
    assert res["intent_summary"]["n_cells_node"] > 0
    # 検出された集合には必ず参加者と帰属が付く(空欄を作らない)
    for g in res["gatherings"]:
        assert g["n_members"] >= 1
        assert g["via"] in ("event", "appointment", "plan", "joint",
                            "congestion", "none")
    report = DG.build_report(res)
    assert "検出された集合" in report and "意図 → 実現 / 不発" in report
