"""因果台帳 IF-F(observer.causality。既定 OFF)のテスト。

正典
  - 分類表と設計 src/society/observer/causality.py
  - 事後解析 scripts/analyze_causality.py
  - house style は tests/test_llm_link.py(IF-A の provlink)に倣う

守るもの(検収基準の順)
  (1) **網羅**: 登録された全 kind に cause_type がある。捏造の kind は落ちる。
  (2) **既定 OFF = バイト一致**: 既定 conf が false・ゴールデン L1 一致・
      L1 parquet の列が変更前の 9 列のまま(2 列が生えない)。
  (3) **ON**: 3 列が生え、全イベントが cause_type を持ち、行為イベントの actor_id が
      行為者と一致し、患者 index の種(hear)は payload から行為者へ戻る。決定論。
  (4) **事後解析**: 行列の合計 = 件数・突き合わせ 0 件。刻印列が無いランでも表だけで動く。
  (5) **患者 override のキー名**が実装と一致する(合成 Event で 1 種ずつ確かめる)。
  (6) **装置の同一性 W2**: 世界プロセスの phase 本体が装置スコープを開き、対象の kind に
      device_id が載る。id は名簿の閉リストの中だけ。刻んでよい cause_type は
      device/schedule/boundary に閉じている(agent / physics / natural には刻まない)。
      OFF ではスコープを開く経路が構造的に存在しない。
  (7) **装置 id の被覆 W3**: 窓を開けられない場所(1 行ごとに装置の個体が違う /
      窓の周りが個体の行為)に per-emit の刻印を通し、実測で -1 質量の最大種だった
      traffic_flow・無人 serve・org_output・dwell_decision と、制度移転
      (tax / wage / 銀行)が装置 id を名乗る。★スタッフが応対した serve と
      乗務員の居た dwell_decision には**付けない**(あれは個体の行為)。
      刻んでも 9 列の L1 は OFF と完全一致(観測がシムを変えない)。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))          # scripts/ は package ではない
sys.path.insert(0, str(_ROOT / "src"))

import analyze_causality as ac                       # noqa: E402

from society import registry as R                    # noqa: E402
from society.config import load_config               # noqa: E402
from society.engine.simulation import Simulation     # noqa: E402
from society.observer import causality as C          # noqa: E402
from society.observer.schema import EVENT_KINDS, Event  # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "data" / "golden_baseline_l1.json"

# test_scenario.py:45 / test_llm_link.py:38 と同じ「意図的な既定挙動追加」の中立化
_GOLDEN_NEUTRAL = {"economy.wages.自営": 0, "planning.enabled": "false",
                   "transit_ride.taxi.enabled": "false",
                   "transit_ride.bus.enabled": "false", "rules.enabled": "false"}

CAUSALITY_ON = {"observer.causality.enabled": "true"}

#: 変更前の L1 parquet の列(ここが 1 つでも動いたら既存ランと突き合わせられなくなる)。
PRECHANGE_L1_COLUMNS = ["step", "sim_min", "agent_id", "kind", "x", "y",
                        "payload", "rng_stream", "llm_call_id"]

#: ON のランで足りる 3 列(順序も固定)。device_id は W2 で足した装置の同一性。
NEW_L1_COLUMNS = ["cause_type", "actor_id", "device_id"]


def _cfg(name, n_steps=24, n_agents=12, **ov):
    dot = ["run.seed=42", f"run.n_agents={n_agents}", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock", "observer.snapshot_every=24"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def _sim(tmp_path, name, n_steps=24, n_agents=12, **ov):
    return Simulation(_cfg(name, n_steps, n_agents, **ov), out_dir=tmp_path / name)


def _l1(sim):
    """ゴールデン比較の形(step / agent_id / kind / payload)。"""
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _l1c(sim):
    """決定論比較の形(新 3 列まで含める)。"""
    return [[e.step, e.agent_id, e.kind, e.cause_type, e.actor_id, e.device_id,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _kind(sim, kind):
    return [e for e in sim.logger.events if e.kind == kind]


#: 「機能が実際に火を噴く」小ラン用のトグル束(全 kind を出すためではなく、
#:  行為系 / 装置系 / 自然系 / 境界系が最低 1 種ずつ混ざる状態を作るため)。
RICH = {"economy.enabled": "true", "commerce.enabled": "true",
        "weather.enabled": "true", "health.enabled": "true",
        "relations.enabled": "true", "freedom.open_actions": "true",
        "government.enabled": "true", "work.service.enabled": "true"}


# --------------------------------------------------------------------------- #
# (1) 網羅(検収基準 1)
# --------------------------------------------------------------------------- #
def _import_every_society_module() -> dict[str, str]:
    """``society.*`` を全て import して**材料側の register_event_kind を全部走らせる**。

    kind の登録は observer/schema.py だけではない: `src/society/devices.py` のように
    「自分の module で宣言する」流儀の feature module がある(gate_pass / device_load /
    signal_summary)。module 名を手で並べると必ず古びるので、走査で拾う。
    import に失敗した module は握り潰さず戻り値で返す(呼び出し側が assert する)。
    """
    import importlib
    import pkgutil

    import society
    failed: dict[str, str] = {}
    for mod in pkgutil.walk_packages(society.__path__, "society."):
        try:
            importlib.import_module(mod.name)
        except Exception as ex:                      # noqa: BLE001(理由ごと返す)
            failed[mod.name] = repr(ex)
    return failed


def test_every_registered_kind_is_classified():
    """登録された全 kind に cause_type がある(材料側 module を読み込んだ後で数える)。"""
    failed = _import_every_society_module()
    assert failed == {}, f"import できない module がある(網羅を保証できない): {failed}"
    missing = sorted(set(EVENT_KINDS) - set(C.CAUSE_OF_KIND))
    assert missing == [], f"cause_type が無い kind: {missing}"


def test_no_stale_entries_in_the_table():
    """表に「もう存在しない kind」が残っていない(死んだ宣言の検出)。"""
    _import_every_society_module()
    stale = sorted(set(C.CAUSE_OF_KIND) - set(EVENT_KINDS))
    assert stale == [], f"EVENT_KINDS に無い kind が表に残っている: {stale}"


def test_every_value_is_a_known_cause_type():
    bad = {k: v for k, v in C.CAUSE_OF_KIND.items() if v not in C.CAUSE_TYPES}
    assert bad == {}, f"未知の cause_type: {bad}"
    assert set(C.PROJECTION_4WAY) == set(C.CAUSE_TYPES)
    assert set(C.PROJECTION_4WAY.values()) == {
        C.AGENT, C.DEVICE, C.NATURAL, C.BOUNDARY}, "4 分類への射影が崩れた"
    assert C.PROJECTION_4WAY[C.SCHEDULE] == C.DEVICE      # DEVS δ_int(暦)
    assert C.PROJECTION_4WAY[C.PHYSICS] == C.DEVICE       # DEVS δ_int(状態)
    assert C.TICK_CAUSES == frozenset({C.PHYSICS, C.SCHEDULE})


def test_fabricated_kind_is_rejected():
    """捏造の kind は黙って unknown にならず KeyError で落ちる。"""
    assert "totally_made_up_kind" not in C.CAUSE_OF_KIND
    with pytest.raises(KeyError):
        C.cause_of("totally_made_up_kind")


def test_overrides_only_reference_registered_kinds():
    unknown = sorted(set(C.PATIENT_ACTOR_OVERRIDES) - set(EVENT_KINDS))
    assert unknown == [], f"未登録 kind の override: {unknown}"


# --------------------------------------------------------------------------- #
# (2) 既定 OFF = バイト一致(検収基準 2)
# --------------------------------------------------------------------------- #
def test_shipped_default_is_off():
    cfg = load_config()
    assert bool(cfg.observer.causality.enabled) is False
    assert C.cfg_of_config(cfg) is False


def test_registry_declares_the_toggle():
    feat = {f.id: f for f in R.FEATURES}["observer.causality.enabled"]
    assert feat.repro_tier == "strict" and feat.affects_k is False
    assert feat.fingerprint_risk == "none"
    assert R.undeclared_toggles(load_config()) == []


def test_off_matches_golden(tmp_path):
    """既定 OFF は変更前ゴールデンと一字一句一致(IF-F の seam が完全な no-op)。"""
    golden = json.load(open(GOLDEN, encoding="utf-8"))
    sim = _sim(tmp_path, "cz_golden", n_steps=144, n_agents=15, **_GOLDEN_NEUTRAL)
    sim.run()
    assert _l1(sim) == golden, "IF-F の seam がゴールデンを動かしている"


def test_off_leaves_the_three_fields_none(tmp_path):
    sim = _sim(tmp_path, "cz_off_fields", **RICH)
    sim.run()
    assert sim.logger.causality_on is False
    assert sim.logger._cause is None
    assert sim.logger.cause_device() is None, "OFF なのに装置スコープが開いた"
    assert all(e.cause_type is None and e.actor_id is None and e.device_id is None
               for e in sim.logger.events), "OFF なのに刻印された"


def test_off_parquet_schema_is_the_prechange_column_list(tmp_path):
    """既定ランの L1 parquet の列が変更前と完全一致(2 列が生えない)。"""
    sim = _sim(tmp_path, "cz_off_pq", **RICH)
    sim.run()
    names = pq.read_schema(tmp_path / "cz_off_pq" / "l1_events.parquet").names
    assert list(names) == PRECHANGE_L1_COLUMNS


def test_off_matches_pure_default(tmp_path):
    """明示 OFF と純粋既定が L1 完全一致(新 2 列まで含めて)。"""
    pure = _sim(tmp_path, "cz_pure")
    pure.run()
    off = _sim(tmp_path, "cz_explicit_off",
               **{"observer.causality.enabled": "false"})
    off.run()
    assert _l1c(pure) == _l1c(off)


# --------------------------------------------------------------------------- #
# (3) ON(検収基準 3)
# --------------------------------------------------------------------------- #
def test_on_adds_exactly_three_columns(tmp_path):
    sim = _sim(tmp_path, "cz_on_pq", **CAUSALITY_ON, **RICH)
    sim.run()
    names = list(pq.read_schema(tmp_path / "cz_on_pq" / "l1_events.parquet").names)
    assert names == PRECHANGE_L1_COLUMNS + NEW_L1_COLUMNS


def test_on_fills_cause_type_for_every_event(tmp_path):
    sim = _sim(tmp_path, "cz_on_fill", **CAUSALITY_ON, **RICH)
    sim.run()
    assert sim.logger.events, "テスト前提が崩れた(イベントが 1 件も無い)"
    for e in sim.logger.events:
        assert e.cause_type in C.CAUSE_TYPES, f"{e.kind}: {e.cause_type}"
        assert e.cause_type == C.CAUSE_OF_KIND[e.kind], \
            f"{e.kind}: 刻印 {e.cause_type} が表 {C.CAUSE_OF_KIND[e.kind]} と食い違う"


def test_on_actor_is_the_acting_agent_for_agent_actions(tmp_path):
    """行為イベント(speak / spend / move_segment …)の actor_id = その個体。"""
    sim = _sim(tmp_path, "cz_on_actor", **CAUSALITY_ON, **RICH)
    sim.run()
    seen = 0
    for e in sim.logger.events:
        if e.kind in ("speak", "spend", "move_segment", "arrive", "route_start"):
            seen += 1
            assert e.actor_id == e.agent_id, f"{e.kind}: actor が行為者と違う"
            assert e.cause_type == C.AGENT
    assert seen > 0, "テスト前提が崩れた(行為イベントが 1 件も無い)"


def test_on_hear_actor_is_the_speaker(tmp_path):
    """患者 index の種は payload から**本当の行為者**へ戻る(agent_id は聞き手のまま)。"""
    sim = _sim(tmp_path, "cz_on_hear", **CAUSALITY_ON, **RICH)
    sim.run()
    hears = _kind(sim, "hear")
    assert hears, "テスト前提が崩れた(聴取が 1 件も無い)"
    for e in hears:
        assert e.actor_id == e.payload["speaker"] != e.agent_id
        assert e.cause_type == C.AGENT


def test_on_world_events_have_no_actor(tmp_path):
    """agent_id=-1 の世界イベントは actor_id=None(=-1 を id として数えない)。"""
    sim = _sim(tmp_path, "cz_on_world", **CAUSALITY_ON, **RICH)
    sim.run()
    world = [e for e in sim.logger.events if e.agent_id < 0]
    assert world, "テスト前提が崩れた(世界イベントが 1 件も無い)"
    for e in world:
        assert e.actor_id is None, f"{e.kind}: -1 を行為者として数えた"


def test_on_is_deterministic_across_two_runs(tmp_path):
    a = _sim(tmp_path, "cz_det_a", **CAUSALITY_ON, **RICH)
    a.run()
    b = _sim(tmp_path, "cz_det_b", **CAUSALITY_ON, **RICH)
    b.run()
    assert _l1c(a) == _l1c(b), "causality ON の決定論が崩れている"


def test_on_does_not_change_the_world(tmp_path):
    """刻んだだけ = 既存 9 列のイベント列が OFF と完全一致(観測がシムを変えない)。"""
    off = _sim(tmp_path, "cz_world_off", **RICH)
    off.run()
    on = _sim(tmp_path, "cz_world_on", **CAUSALITY_ON, **RICH)
    on.run()
    assert _l1(off) == _l1(on)
    assert on.llm.calls == off.llm.calls > 0, "LLM 呼数が動いた(affects_k=False 違反)"


def _spy_apply(sim, agent, on_seen):
    """_apply の内側でスコープの中身を覗き、**例外を投げて** finally を試す。"""
    from society.engine import scheduler
    inner = scheduler._apply_action

    def _boom(s, a, action, step, sim_min):
        on_seen.append(s.logger._cause)
        raise RuntimeError("boom")

    scheduler._apply_action = _boom
    try:
        with pytest.raises(RuntimeError):
            scheduler._apply(sim, agent, {"type": "stay"}, 0, 0)
    finally:
        scheduler._apply_action = inner


def test_apply_scope_is_opened_and_closed(tmp_path):
    """_apply の因果スコープが開き、**例外が出ても** finally で必ず閉じる。"""
    sim = _sim(tmp_path, "cz_scope", n_steps=1, **CAUSALITY_ON)
    agent = sim.agents[0]
    assert sim.logger._cause is None
    seen: list = []
    _spy_apply(sim, agent, seen)
    assert seen == [(C.AGENT, agent.id)], "行為スコープが開いていない"
    assert sim.logger._cause is None, "例外で行為スコープが開きっぱなしになった"


def test_scope_never_opens_when_off(tmp_path):
    """OFF ではスコープを開く経路が構造的に存在しない。"""
    sim = _sim(tmp_path, "cz_scope_off", n_steps=1)
    seen: list = []
    _spy_apply(sim, sim.agents[0], seen)
    assert seen == [None]
    assert sim.logger._cause is None and sim.logger.causality_on is False
    assert all(e.cause_type is None for e in sim.logger.events)


# --------------------------------------------------------------------------- #
# (4) 事後解析(検収基準 4)
# --------------------------------------------------------------------------- #
def _run_dir(tmp_path, name, **ov):
    sim = _sim(tmp_path, name, **ov)
    sim.run()
    return tmp_path / name, sim


def test_analyze_on_run_matches_the_engine(tmp_path):
    """ON のラン: 行列の合計 = 件数 / 突き合わせ 0 件。"""
    d, sim = _run_dir(tmp_path, "cz_an_on", **CAUSALITY_ON, **RICH)
    rep = ac.analyze(d)
    assert rep["engine_columns"] is True
    assert rep["n_events"] == len(sim.logger.events)
    total = sum(sum(row.values()) for row in rep["matrix"].values())
    assert total == rep["n_events"], "行列の合計が件数と合わない"
    for kind, row in rep["matrix"].items():
        assert sum(row.values()) == rep["kinds"][kind]["n"]
    assert rep["unclassified_kinds"] == {}
    cv = rep["cross_validation"]
    assert cv["applicable"] is True
    assert cv["n_cause_mismatch"] == 0 and cv["n_actor_mismatch"] == 0, \
        f"表とエンジンが食い違う: {cv['by_kind']}"


def test_analyze_off_run_classifies_from_the_table_alone(tmp_path):
    """刻印列が無いランでも同じ分類ができる(既存ランを捨てない)。"""
    d, sim = _run_dir(tmp_path, "cz_an_off", **RICH)
    rep = ac.analyze(d)
    assert rep["engine_columns"] is False
    assert rep["cross_validation"]["applicable"] is False
    assert rep["n_events"] == len(sim.logger.events)
    assert rep["unclassified_kinds"] == {}
    per = rep["per_cause"]
    assert per[C.AGENT] > 0 and sum(per.values()) == rep["n_events"]


def test_analyze_agrees_between_on_and_off_runs(tmp_path):
    """同 seed の ON / OFF ランで **kind ごとの分類と帰属率が一致**する。

    = 「刻印はあってもなくても同じ答えになる」= 既存ランに遡って測れることの実測。
    """
    d_on, _ = _run_dir(tmp_path, "cz_ab_on", **CAUSALITY_ON, **RICH)
    d_off, _ = _run_dir(tmp_path, "cz_ab_off", **RICH)
    a, b = ac.analyze(d_on), ac.analyze(d_off)
    assert set(a["kinds"]) == set(b["kinds"])
    for kind in a["kinds"]:
        assert a["kinds"][kind]["cause_type"] == b["kinds"][kind]["cause_type"]
        assert a["kinds"][kind]["n"] == b["kinds"][kind]["n"]
        assert a["kinds"][kind]["n_actor"] == b["kinds"][kind]["n_actor"]


def test_analyze_aggregate_excludes_tick_kinds(tmp_path):
    """集計は physics / schedule を**明示的に除いた** 1 行だけ(単一スカラーを作らない)。"""
    d, _ = _run_dir(tmp_path, "cz_an_ag", **CAUSALITY_ON, **RICH)
    rep = ac.analyze(d)
    ag = rep["aggregate"]
    assert set(ag["excluded_causes"]) == set(C.TICK_CAUSES)
    ticks = sum(v["n"] for v in rep["kinds"].values()
                if v["cause_type"] in C.TICK_CAUSES)
    assert ticks > 0, "テスト前提が崩れた(tick 系が 1 件も無い)"
    assert ag["n_events"] == rep["n_events"] - ticks
    assert 0.0 <= ag["actor_rate"] <= 1.0
    assert ag["n_events_all"] == rep["n_events"]


def test_analyze_ranks_unattributed_mass(tmp_path):
    d, _ = _run_dir(tmp_path, "cz_an_rank", **CAUSALITY_ON, **RICH)
    rep = ac.analyze(d)
    rank = rep["unattributed_ranking"]
    assert rank, "テスト前提が崩れた(agent_id=-1 の種が 1 つも無い)"
    assert rank == sorted(rank, key=lambda r: (-r["n_unattributed"], r["kind"]))
    for r in rank:
        assert r["n_unattributed"] >= r["n_neg_agent"]


def test_analyze_cli_and_markdown(tmp_path):
    d, _ = _run_dir(tmp_path, "cz_cli", **CAUSALITY_ON, **RICH)
    out, js = tmp_path / "cz.md", tmp_path / "cz.json"
    assert ac.main([str(d), "--out", str(out), "--json", str(js)]) == 0
    md = out.read_text(encoding="utf-8")
    assert "# 因果の帰属" in md and "kind × cause_type 行列" in md
    rep = json.loads(js.read_text(encoding="utf-8"))
    assert rep["n_events"] > 0 and rep["cross_validation"]["n_cause_mismatch"] == 0


def _write_l1(dir_path: Path, rows: list[dict], *, with_engine_cols: bool,
              with_device_col: bool = False):
    """合成 L1 を 1 枚書く(実ランでは作れない縁のケースを固定するため)。"""
    import pyarrow as pa
    dir_path.mkdir(parents=True, exist_ok=True)
    cols = {
        "step":        pa.array([r["step"] for r in rows], pa.int32()),
        "sim_min":     pa.array([r["step"] * 10 for r in rows], pa.int32()),
        "agent_id":    pa.array([r["agent_id"] for r in rows], pa.int32()),
        "kind":        pa.array([r["kind"] for r in rows], pa.string()),
        "x":           pa.array([0.0] * len(rows), pa.float32()),
        "y":           pa.array([0.0] * len(rows), pa.float32()),
        "payload":     pa.array([json.dumps(r.get("payload") or {}) for r in rows],
                                pa.string()),
        "rng_stream":  pa.array([""] * len(rows), pa.string()),
        "llm_call_id": pa.array([None] * len(rows), pa.string()),
    }
    if with_engine_cols:
        cols["cause_type"] = pa.array([r.get("cause_type") for r in rows], pa.string())
        cols["actor_id"] = pa.array([r.get("actor_id") for r in rows], pa.int32())
    if with_device_col:
        cols["device_id"] = pa.array([r.get("device_id") for r in rows], pa.string())
    pq.write_table(pa.table(cols), dir_path / "l1_events.parquet", compression="zstd")


def test_analyze_warns_loudly_about_unclassified_kinds(tmp_path):
    """データに表に無い kind が出たら警告セクションと終了コード 1 で知らせる。"""
    d = tmp_path / "cz_unknown"
    _write_l1(d, [{"step": 0, "agent_id": 1, "kind": "speak"},
                  {"step": 0, "agent_id": 2, "kind": "brand_new_kind"},
                  {"step": 1, "agent_id": 2, "kind": "brand_new_kind"}],
              with_engine_cols=False)
    rep = ac.analyze(d)
    assert rep["unclassified_kinds"] == {"brand_new_kind": 2}
    assert rep["kinds"]["brand_new_kind"]["cause_type"] is None
    assert rep["per_cause"]["?"] == 2, "未分類は `?` として正直に数える"
    md = ac.render_markdown(rep)
    assert "分類表に無い kind" in md and "警告" in md and "brand_new_kind" in md
    assert ac.main([str(d), "--out", str(tmp_path / "u.md")]) == 1


def test_analyze_ignores_unstamped_rows_of_a_half_migrated_run(tmp_path):
    """列はあるのに値が null の行(途中で ON にしたラン)を刻印扱いにしない。

    OFF の part と ON の part を unify_schemas で結合すると必ずこの形になる。
    ここを刻印扱いにすると突き合わせが偽陽性で埋まる。
    """
    d = tmp_path / "cz_half"
    _write_l1(d, [
        {"step": 0, "agent_id": 1, "kind": "speak"},                 # 未刻印(null)
        {"step": 1, "agent_id": 1, "kind": "speak",
         "cause_type": C.AGENT, "actor_id": 1},                      # 刻印済み
        {"step": 1, "agent_id": 4, "kind": "hear", "payload": {"speaker": 1},
         "cause_type": C.AGENT, "actor_id": 1},
    ], with_engine_cols=True)
    rep = ac.analyze(d)
    assert rep["engine_columns"] is True
    assert rep["cross_validation"]["n_cause_mismatch"] == 0
    assert rep["cross_validation"]["n_actor_mismatch"] == 0
    # 未刻印の行も表で分類され、行為者が復元されている(捨てない)
    assert rep["kinds"]["speak"]["n"] == 2 and rep["kinds"]["speak"]["n_actor"] == 2
    assert rep["matrix"]["speak"] == {C.AGENT: 2}
    assert rep["kinds"]["hear"]["n_actor"] == 1


def test_analyze_raises_on_a_run_without_l1(tmp_path):
    empty = tmp_path / "no_such_run"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        ac.analyze(empty)


# --------------------------------------------------------------------------- #
# (5) 患者 override のキー名(検収基準 5)
# --------------------------------------------------------------------------- #
def _ev(kind, agent_id, payload):
    return Event(step=0, sim_min=0, agent_id=agent_id, kind=kind,
                 x=0.0, y=0.0, payload=payload)


#: (kind, payload, 期待する行為者)。**実装側の 1 行を読んで確かめた対応**:
#:   hear          scheduler.py:3151  agent_id=hearer   / payload["speaker"]
#:   opinion_shift scheduler.py:1346 + conversation.py:206  agent_id=listener / ["source"]
#:   transmission  observer/provenance.py:45  agent_id=to_agent / ["from"]
#:   deliver       delivery.py:398    agent_id=courier  / ["courier"](現状は恒等)
_OVERRIDE_CASES = [
    ("hear",          {"speaker": 7, "items": []},          7),
    ("opinion_shift", {"source": 3, "old": 0.1, "new": 0.2}, 3),
    ("transmission",  {"item_id": "vocab-1", "from": 5},     5),
    ("deliver",       {"courier": 9, "fare": 100.0},         9),
]


@pytest.mark.parametrize("kind,payload,expected", _OVERRIDE_CASES)
def test_patient_override_recovers_the_actor(kind, payload, expected):
    e = _ev(kind, 99, payload)                    # agent_id は患者(受け手)
    assert C.actor_of(e.kind, e.agent_id, e.payload) == expected


def test_override_cases_cover_the_whole_table():
    """表に載せた種は 1 つ残らずここで検証されている(片方だけ増えたら落ちる)。"""
    assert {k for k, _p, _e in _OVERRIDE_CASES} == set(C.PATIENT_ACTOR_OVERRIDES)


def test_callable_override_rule_is_supported(monkeypatch):
    """override は文字列キーだけでなく callable でも書ける(payload の形が入り組んだ種の逃げ道)。"""
    rule = C._from_key("speaker")
    assert rule.payload_key == "speaker"
    monkeypatch.setitem(C.PATIENT_ACTOR_OVERRIDES, "sns_like", rule)
    assert C.actor_of("sns_like", 1, {"speaker": 42}) == 42
    assert C.actor_of("sns_like", 1, {}) == 1        # 取れなければ agent_id へ後退


def test_override_falls_back_to_agent_id_when_the_key_is_missing():
    """payload にキーが無い/壊れているときは agent_id へ後退(偽の値を作らない)。"""
    assert C.actor_of("hear", 4, {}) == 4
    assert C.actor_of("hear", 4, {"speaker": None}) == 4
    assert C.actor_of("hear", 4, {"speaker": "誰か"}) == 4
    assert C.actor_of("hear", -1, {}) is None


def test_actor_of_folds_world_events_to_none():
    assert C.actor_of("weather", -1, {}) is None
    assert C.actor_of("speak", 0, {}) == 0            # id 0 は有効(偽の None にしない)
    assert C.actor_of("speak", True, {}) is None      # bool は id ではない


def test_non_patient_kinds_keep_their_agent_id():
    """agent_id が既に行為者の種を上書きしていない(serve / enforcement / venture_sale)。"""
    assert "serve" not in C.PATIENT_ACTOR_OVERRIDES
    assert C.actor_of("serve", 12, {"customer": 3}) == 12
    assert C.actor_of("serve", -1, {"customer": 3}) is None   # スタッフ不在
    assert C.actor_of("enforcement", 5, {"target": 8, "officer": 5}) == 5
    assert C.actor_of("detention", 5, {"target": 8, "officer": 5}) == 5
    assert C.actor_of("venture_sale", 2, {"buyer": 6}) == 2
    assert C.actor_of("crime", 1, {"victim": 4}) == 1
    # gossip_spread の payload["sources"] は**人数(int)**であって id 列ではない
    assert "gossip_spread" not in C.PATIENT_ACTOR_OVERRIDES
    assert C.actor_of("gossip_spread", 11, {"target": 2, "sources": 3}) == 11


def test_summarize_folds_kind_counts():
    got = C.summarize({"speak": 3, "weather": 1, "wage": 2, "unknown_kind": 9})
    assert got[C.AGENT] == 3 and got[C.NATURAL] == 1 and got[C.DEVICE] == 2
    assert sum(got.values()) == 6, "未登録 kind を黙って数えた"


# --------------------------------------------------------------------------- #
# (6) 装置の同一性 device_id(検収基準 6・W2)
# --------------------------------------------------------------------------- #
#: 世界プロセスの装置スコープが実際に火を噴く小ラン。
#:   commerce:hours   … 営業時間の開閉遷移(shop_state)
#:   logistics:goods  … (s,S) レビュー + 補充到着(stock_low / delivery_trip / restock)
#:   gov:main         … 行政会計の締め(public_budget)
#: restock_hour=12 は「その日のレビューを 48 step の窓に入れる」ためだけの実験条件で、
#: 発注点を上げてあるのは 12 人の小ランでも在庫が発注点を割るようにするため。
PROCESS = {"economy.enabled": "true", "commerce.enabled": "true",
           "commerce.inventory.enabled": "true",
           "commerce.inventory.restock_hour": 12,
           "commerce.inventory.reorder_point.food": 39,
           "commerce.inventory.reorder_point.cafe": 39,
           "commerce.inventory.reorder_point.shop": 29,
           "commerce.inventory.reorder_point.nightlife": 29,
           "government.enabled": "true"}

#: 装置スコープが刻むはずの種 → 期待する device_id(**実測で確かめた対応**)。
_EXPECTED_DEVICE_OF_KIND = {
    "shop_state": "commerce:hours",
    "stock_low": "logistics:goods",
    "delivery_trip": "logistics:goods",
    "restock": "logistics:goods",
    "public_budget": "gov:main",
}


def _process_run(tmp_path, name, **ov):
    sim = _sim(tmp_path, name, n_steps=48, **{**CAUSALITY_ON, **PROCESS, **ov})
    sim.run()
    return sim


def test_device_id_catalogue_is_a_closed_list():
    """装置 id の名簿は 1 箇所(society/devices.py)。形も接頭辞も閉じている。"""
    from society import devices as D
    assert set(D.PROCESS_DEVICE_IDS) == {
        D.DEV_BANK_MAIN, D.DEV_COMMERCE_HOURS, D.DEV_COMMERCE_PRICING,
        D.DEV_GOV_MAIN, D.DEV_GOV_PAYROLL, D.DEV_GOV_TAX,
        D.DEV_LOGISTICS_GOODS, D.DEV_OPERATOR_INFRA, D.DEV_OPERATOR_TRANSIT,
        D.DEV_TRAFFIC_AMBIENT, D.DEV_TRAFFIC_OD}
    assert list(D.PROCESS_DEVICE_IDS) == sorted(D.PROCESS_DEVICE_IDS), "並びが不定"
    for did in D.PROCESS_DEVICE_IDS:
        assert D.device_id_is_known(did), did
        assert did.count(":") == 1 and did.split(":")[0] in D.DEVICE_ID_PREFIXES
    # 実体を持つ装置の動的 id も同じ形(接頭辞は名簿の中)
    assert D.device_id_is_known(D.faregate_device_id("n123"))
    assert D.device_id_is_known(D.signal_device_id(42))
    assert D.device_id_is_known(D.transit_operator_device_id())
    # W3 で足した動的 id の族(":" の右が地図 / 台帳から来るので列挙できない)
    assert D.DYNAMIC_DEVICE_PREFIXES == frozenset(
        {"faregate", "signal", "pos", "org", "train_op"})
    assert D.device_id_is_known(D.pos_device_id("n42")) and \
        D.pos_device_id("n42") == "pos:n42"
    assert D.device_id_is_known(D.org_device_id("org_7")) and \
        D.org_device_id("org_7") == "org:org_7"
    assert D.traffic_device_id("ambient") == D.DEV_TRAFFIC_AMBIENT
    assert D.traffic_device_id("od") == D.DEV_TRAFFIC_OD
    # ★接頭辞の名簿は**導出**である(手で並べた 2 つ目の名簿を作らない = 腐らない)
    assert D.DEVICE_ID_PREFIXES == (
        {d.split(":")[0] for d in D.PROCESS_DEVICE_IDS} | D.DYNAMIC_DEVICE_PREFIXES)
    # 名簿に無い形は**検出できる**(捏造の検出器。禁止ではない)
    assert D.device_id_is_known("mystery:1") is False
    assert D.device_id_is_known("gov") is False and D.device_id_is_known("") is False


def _known_ids(sim):
    """ランに現れた device_id(名簿の形をしていることまで確かめる)。"""
    from society import devices as D
    ids = {e.device_id for e in sim.logger.events if e.device_id is not None}
    assert all(D.device_id_is_known(i) for i in ids), sorted(ids)
    return ids


def _in_catalogue(device_id) -> bool:
    """静的名簿の id か、動的 id の族(接頭辞が閉リストの中)か。"""
    from society import devices as D
    return (device_id in set(D.PROCESS_DEVICE_IDS)
            or str(device_id).partition(":")[0] in D.DYNAMIC_DEVICE_PREFIXES)


def test_on_process_scopes_stamp_their_device_id(tmp_path):
    """世界プロセスの phase 本体が開いた窓の中の種に device_id が載る。"""
    sim = _process_run(tmp_path, "cz_dev_stamp")
    seen = {}
    for e in sim.logger.events:
        if e.kind in _EXPECTED_DEVICE_OF_KIND:
            assert e.device_id == _EXPECTED_DEVICE_OF_KIND[e.kind], \
                f"{e.kind}: device_id={e.device_id}"
            seen[e.kind] = seen.get(e.kind, 0) + 1
    assert set(seen) == set(_EXPECTED_DEVICE_OF_KIND), \
        f"検収の空回り(出ていない種がある): {sorted(set(_EXPECTED_DEVICE_OF_KIND) - set(seen))}"


def test_on_stamped_ids_are_all_in_the_catalogue(tmp_path):
    """ランに現れた device_id は 1 つ残らず名簿の中(未知の id を作っていない)。"""
    sim = _process_run(tmp_path, "cz_dev_closed")
    ids = _known_ids(sim)
    assert ids, "検収の空回り(1 件も刻まれていない)"
    assert all(_in_catalogue(i) for i in ids), \
        f"名簿にも動的 id の族にも無い id が出た: {sorted(i for i in ids if not _in_catalogue(i))}"


def test_on_device_causes_have_no_actor(tmp_path):
    """世界プロセスの装置が起こした行に**偽の行為者**が入っていない。"""
    sim = _process_run(tmp_path, "cz_dev_actor")
    stamped = [e for e in sim.logger.events
               if e.device_id is not None and e.kind in _EXPECTED_DEVICE_OF_KIND]
    assert stamped, "検収の空回り"
    for e in stamped:
        assert e.agent_id == -1, f"{e.kind}: 前提(世界イベント)が崩れた"
        assert e.actor_id is None, f"{e.kind}: 装置起因なのに行為者 id が入っている"
        assert e.cause_type in C.DEVICE_STAMPABLE


def test_on_scope_does_not_stamp_agent_or_physics_events(tmp_path):
    """★窓の中で出ても agent / physics / natural には刻まない(DEVICE_STAMPABLE)。

    これが無いと、装置の処理に連なって出る個体側の出来事(state_update など)まで
    「その装置が起こした」ことになり、帰属が構造的に汚染される。
    """
    sim = _process_run(tmp_path, "cz_dev_discipline")
    assert C.DEVICE_STAMPABLE == frozenset({C.DEVICE, C.SCHEDULE, C.BOUNDARY})
    for e in sim.logger.events:
        if e.cause_type in (C.AGENT, C.PHYSICS, C.NATURAL):
            assert e.device_id is None, f"{e.kind}({e.cause_type})に装置 id が付いた"


def test_on_device_scope_is_closed_even_on_exception(tmp_path):
    """装置スコープは with(= try/finally)なので例外でも必ず閉じる。入れ子も戻す。"""
    from society import devices as D
    sim = _sim(tmp_path, "cz_dev_scope", n_steps=1, **CAUSALITY_ON)
    assert sim.logger.cause_device() is None
    with pytest.raises(RuntimeError):
        with D.cause_scope(sim, D.DEV_GOV_MAIN):
            assert sim.logger.cause_device() == D.DEV_GOV_MAIN
            with D.cause_scope(sim, D.DEV_LOGISTICS_GOODS):
                assert sim.logger.cause_device() == D.DEV_LOGISTICS_GOODS
            assert sim.logger.cause_device() == D.DEV_GOV_MAIN, "入れ子から戻っていない"
            raise RuntimeError("boom")
    assert sim.logger.cause_device() is None, "例外で装置スコープが開きっぱなし"


def test_off_device_scope_is_a_no_op_singleton(tmp_path):
    """OFF ではスコープが**何もしない singleton**(割り当てゼロ・logger 不触)。"""
    from society import devices as D
    sim = _sim(tmp_path, "cz_dev_off_scope", n_steps=1)
    scope = D.cause_scope(sim, D.DEV_GOV_MAIN)
    assert scope is D.NO_SCOPE
    with scope:
        assert sim.logger.cause_device() is None
    assert sim.logger.cause_device() is None


def test_on_device_stamps_do_not_change_the_world(tmp_path):
    """device_id を刻んでも既存 9 列のイベント列は OFF と完全一致(観測がシムを変えない)。"""
    off = _sim(tmp_path, "cz_dev_w_off", n_steps=48, **PROCESS)
    off.run()
    on = _process_run(tmp_path, "cz_dev_w_on")
    assert _l1(off) == _l1(on)
    assert on.llm.calls == off.llm.calls > 0


def test_on_device_stamping_is_deterministic(tmp_path):
    a = _process_run(tmp_path, "cz_dev_det_a")
    b = _process_run(tmp_path, "cz_dev_det_b")
    assert _l1c(a) == _l1c(b)


def test_analyze_reports_device_breakdown(tmp_path):
    """事後解析が device_id 別の内訳を出し、突き合わせは 0 件のまま。"""
    from society import devices as D
    sim = _process_run(tmp_path, "cz_dev_an")
    rep = ac.analyze(tmp_path / "cz_dev_an")
    dev = rep["devices"]
    assert rep["device_column"] is True and dev["applicable"] is True
    assert dev["n_stamped"] == sum(1 for e in sim.logger.events
                                   if e.device_id is not None) > 0
    assert dev["unknown_device_ids"] == {}
    assert all(_in_catalogue(i) for i in dev["per_device"])
    assert dev["by_kind"]["public_budget"] == {D.DEV_GOV_MAIN: 3}
    assert sum(dev["per_device"].values()) == dev["n_stamped"]
    cv = rep["cross_validation"]
    assert cv["n_cause_mismatch"] == 0 and cv["n_actor_mismatch"] == 0
    md = ac.render_markdown(rep)
    assert "装置の同一性" in md and D.DEV_GOV_MAIN in md
    assert ac.main([str(tmp_path / "cz_dev_an"), "--out", str(tmp_path / "d.md")]) == 0


def test_analyze_warns_about_device_ids_outside_the_catalogue(tmp_path):
    """名簿に無い装置 id は警告 + 終了コード 1(表とデータのずれの検出器)。"""
    d = tmp_path / "cz_dev_unknown"
    _write_l1(d, [{"step": 0, "agent_id": -1, "kind": "restock",
                   "cause_type": C.DEVICE, "device_id": "mystery:9"},
                  {"step": 0, "agent_id": -1, "kind": "restock",
                   "cause_type": C.DEVICE, "device_id": "logistics:goods"}],
              with_engine_cols=True, with_device_col=True)
    rep = ac.analyze(d)
    assert rep["devices"]["unknown_device_ids"] == {"mystery:9": 1}
    assert rep["devices"]["per_device"] == {"logistics:goods": 1, "mystery:9": 1}
    md = ac.render_markdown(rep)
    assert "名簿に無い装置 id" in md and "mystery:9" in md
    assert ac.main([str(d), "--out", str(tmp_path / "u2.md")]) == 1


def test_analyze_still_works_on_a_w1_run_without_the_device_column(tmp_path):
    """W1 の ON ラン(2 列だけ)を『刻印なし』に落とさない(過去ランを捨てない)。"""
    d = tmp_path / "cz_dev_w1"
    _write_l1(d, [{"step": 0, "agent_id": 1, "kind": "speak",
                   "cause_type": C.AGENT, "actor_id": 1}],
              with_engine_cols=True)
    rep = ac.analyze(d)
    assert rep["engine_columns"] is True            # 突き合わせは生きている
    assert rep["device_column"] is False
    assert rep["devices"]["applicable"] is False and rep["devices"]["per_device"] == {}
    assert rep["cross_validation"]["n_cause_mismatch"] == 0
    md = ac.render_markdown(rep)
    assert "device_id` 列が無い" in md


# --------------------------------------------------------------------------- #
# (7) 装置 id の被覆(検収基準 7・W3)= 窓を開けられない場所の per-emit 刻印
# --------------------------------------------------------------------------- #
#: 経済・商業・行政・銀行・接客までひととおり点火する小ラン(48 step)。
#: 実測(probe)で traffic_flow 48 / serve 13 / org_output 8 / tax 39 /
#: price_change 13 / interest_paid 4 が出ることを確かめた組み合わせ。
W3 = {"agents.personas_file": "data/personas_80.json",
      "organizations.enabled": "true",
      "economy.enabled": "true", "economy.fixed_cost_daily": 300,
      "economy.accounts.enabled": "true", "economy.accounts.payday_dom": 1,
      "economy.bank.enabled": "true", "economy.bank.deposit_rate": 3.65,
      "commerce.enabled": "true", "government.enabled": "true",
      "work.service.enabled": "true"}


def _w3_run(tmp_path, name, n_steps=48, **ov):
    sim = _sim(tmp_path, name, n_steps=n_steps, **{**CAUSALITY_ON, **W3, **ov})
    sim.run()
    return sim


def _serve_sim(tmp_path, name, *, staffed: bool):
    """work.service の接客 seam を**合成配置**で 1 件だけ踏む(tests/test_work_service.py と同型)。

    フルランでは「無人の応対」しか出ないので、スタッフが応対した行と無人の行を
    同じ土俵で比べるにはこの合成配置が要る(スタッフ有無だけが違う 2 本)。
    """
    sim = _sim(tmp_path, name, n_steps=1, n_agents=25,
               **{**CAUSALITY_ON, "work.service.enabled": "true"})
    node = sim.city.pois_by_cat("food")[0]["node"]
    x, y = sim.city.node_xy(node)
    for a in sim.agents:                        # 誰も勤務中スタッフにならない状態から始める
        a.work_start_min = -1
    customer = sim.agents[0]
    customer.node, customer.x, customer.y = node, x, y
    if staffed:
        staff = sim.agents[1]
        staff.node = staff.work_node = node
        staff.x, staff.y = x, y
        staff.work_start_min, staff.work_end_min = 0, 1440
    since = len(sim.logger.events)
    sim.logger.log(Event(step=3, sim_min=700, agent_id=customer.id, kind="spend",
                         x=x, y=y, payload={"amount": 900.0, "balance": 9100.0,
                                            "cat": "food"}))
    from society.engine import scheduler as _sched
    _sched._phase_work_service(sim, 3, 700, since)
    return sim, node


# ---- 7-1. 背景交通(-1 質量の最大種)------------------------------------------ #
def test_on_traffic_flow_names_its_generator(tmp_path):
    """★実測で最大の『行為者不明』だった traffic_flow が発生器の名を持つ。"""
    from society import devices as D
    sim = _sim(tmp_path, "cz_w3_traffic", n_steps=12, **CAUSALITY_ON)
    sim.run()
    rows = _kind(sim, "traffic_flow")
    assert rows, "検収の空回り(背景交通が 1 件も出ていない)"
    for e in rows:
        assert e.agent_id == -1 and e.actor_id is None   # 行為者は今も居ない(正しい)
        assert e.cause_type == C.BOUNDARY
        assert e.device_id == D.DEV_TRAFFIC_AMBIENT == "traffic:ambient"
    assert sim.traffic.mode == "ambient"                  # 既定モードの実測
    assert D.traffic_device_id("od") == "traffic:od", "od は別の装置として名乗る"


# ---- 7-2. 無人の応対だけが店頭装置(pos)を名乗る ------------------------------ #
def test_on_unstaffed_serve_names_the_point_of_sale(tmp_path):
    from society import devices as D
    sim, node = _serve_sim(tmp_path, "cz_w3_pos_un", staffed=False)
    rows = _kind(sim, "serve")
    assert len(rows) == 1
    e = rows[0]
    assert e.agent_id == -1 and e.payload["unstaffed"] is True
    assert e.device_id == D.pos_device_id(node) == f"pos:{node}"
    assert e.cause_type == C.DEVICE and e.actor_id is None


def test_on_staffed_serve_stays_an_agent_act_with_no_device_id(tmp_path):
    """★スタッフが応対した行に装置 id は**付けない**(個体の行為を装置に化けさせない)。"""
    sim, _node = _serve_sim(tmp_path, "cz_w3_pos_staffed", staffed=True)
    rows = _kind(sim, "serve")
    assert len(rows) == 1
    e = rows[0]
    assert e.agent_id == sim.agents[1].id and "unstaffed" not in e.payload
    assert e.device_id is None, "スタッフの応対に店頭装置の id が付いた"
    assert e.actor_id == sim.agents[1].id      # 行為者はそのスタッフのまま


# ---- 7-3. 会社の日次産出 ------------------------------------------------------ #
def test_on_org_output_names_the_org(tmp_path):
    """職場キー経路(by_org OFF)。id は payload の org と 1 対 1 = 限界の開示。"""
    from society import devices as D
    sim = _w3_run(tmp_path, "cz_w3_org")
    rows = _kind(sim, "org_output")
    assert rows, "検収の空回り(org_output が出ていない)"
    for e in rows:
        assert e.agent_id == -1
        assert e.device_id == D.org_device_id(e.payload["org"])
        assert e.device_id.startswith("org:")


# ---- 7-4. 値付け(窓を開けてはいけない場所の実例)------------------------------ #
def test_on_price_change_names_the_pricing_device_and_spend_stays_agent(tmp_path):
    """★店の値付けだけに id が付き、**隣の spend(客の行為)には付かない**。"""
    from society import devices as D
    sim = _w3_run(tmp_path, "cz_w3_price")
    prices = _kind(sim, "price_change")
    assert prices, "検収の空回り(価格変動が起きていない)"
    for e in prices:
        assert e.device_id == D.DEV_COMMERCE_PRICING == "commerce:pricing"
        assert e.cause_type == C.DEVICE
    spends = _kind(sim, "spend")
    assert spends and all(e.device_id is None and e.cause_type == C.AGENT
                          for e in spends), "客の消費に店の装置 id が付いた"
    # 同じ step・同じ個体で隣り合っている(= 窓を開けていたら必ず巻き込む配置)
    keys = {(e.step, e.agent_id) for e in prices}
    assert keys & {(e.step, e.agent_id) for e in spends}, "隣接の検収が空回り"


# ---- 7-5. 制度移転(行政・銀行)----------------------------------------------- #
def test_on_tax_names_the_tax_office(tmp_path):
    """agent_id は納税者(患者)。徴収したのは徴税制度 = gov:tax。"""
    from society import devices as D
    sim = _w3_run(tmp_path, "cz_w3_tax")
    rows = _kind(sim, "tax")
    assert rows, "検収の空回り(税が 1 件も徴収されていない)"
    for e in rows:
        assert e.agent_id >= 0 and e.device_id == D.DEV_GOV_TAX == "gov:tax"


def test_on_civil_wage_names_the_payroll_and_ordinary_wage_stays_unstamped(tmp_path):
    """公務員給与だけが gov:payroll。雇い主が emit 点に無い賃金は**無印のまま**。"""
    from society import devices as D
    from society.engine import scheduler as _sched
    sim = _sim(tmp_path, "cz_w3_payroll", n_steps=1,
               **{**CAUSALITY_ON, "economy.enabled": "true",
                  "government.enabled": "true"})
    for a in sim.agents[:2]:
        a.occupation = "区職員"
    _sched._phase_government(sim, 0, 0)             # 基準日(記録なし)
    _sched._phase_government(sim, 1, 1440)          # 日境界 → ペイロール
    civil = [e for e in _kind(sim, "wage") if e.payload.get("source") == "civil"]
    assert civil, "検収の空回り(公務員給与が出ていない)"
    for e in civil:
        assert e.device_id == D.DEV_GOV_PAYROLL == "gov:payroll"
    # 本業/バイトの賃金は雇い主が emit 点に無い = 刻まない(欠測を偽の id で埋めない)
    plain = _sim(tmp_path, "cz_w3_wage_plain", n_steps=24,
                 **{**CAUSALITY_ON, "economy.enabled": "true"})
    plain.run()
    ordinary = [e for e in _kind(plain, "wage")
                if e.payload.get("source") != "civil"]
    assert ordinary, "検収の空回り(賃金が 1 件も出ていない)"
    assert all(e.device_id is None for e in ordinary)


def test_on_bank_rows_name_the_bank(tmp_path):
    """融資・返済・利息はすべて bank:main(agent_id は借り手 / 預金者 = 患者)。"""
    from society import devices as D
    from society.engine import scheduler as _sched
    sim = _sim(tmp_path, "cz_w3_bank", n_steps=1,
               **{**CAUSALITY_ON, "economy.enabled": "true",
                  "economy.accounts.enabled": "true",
                  "economy.bank.enabled": "true",
                  "economy.bank.deposit_rate": 3.65})
    a = next(x for x in sim.agents if not x.visitor)
    a.period_income, a.account, a.money, a.arrears_days = 300000.0, 1000.0, 500.0, 0
    assert _sched._maybe_loan(sim, a, 20000.0, 0, 600) > 0.0, "検収の空回り(融資が下りない)"
    sim._bank_day = -1
    a.account = 100000.0
    _sched._phase_bank_day(sim, 1, 1440 * 40)
    rows = _kind(sim, "loan_grant") + _kind(sim, "loan_repay")
    assert len(rows) >= 2, "検収の空回り(融資 / 返済が揃っていない)"
    for e in rows:
        assert e.agent_id == a.id and e.device_id == D.DEV_BANK_MAIN == "bank:main"
    # 預金利息もフルランで同じ id を名乗る
    full = _w3_run(tmp_path, "cz_w3_interest")
    itr = _kind(full, "interest_paid")
    assert itr and all(e.device_id == D.DEV_BANK_MAIN for e in itr)


# ---- 7-6. 乗務員不在のドア閉判断(payload → 列)-------------------------------- #
def test_on_unstaffed_dwell_decision_names_the_train_operator(tmp_path):
    """``payload["operator"]`` と**同じ文字列**が device_id 列に出る(後方互換つき)。"""
    from society import transit_staff as TS
    sim = _sim(tmp_path, "cz_w3_dwell", n_steps=1, n_agents=20,
               **{**CAUSALITY_ON, "transit_staff.enabled": "true",
                  "env.feedback.enabled": "true",
                  "env.feedback.log_every_steps": "1",
                  "env.feedback.transit.platform_threshold": "1"})
    station = sim.city.station_node
    for a in sim.agents:                                 # 合成高負荷(乗務員は 1 人も居ない)
        a.loc, a.node, a.sleeping, a.route = "street", station, False, []
        a.x, a.y = sim.city.node_xy(station)
    sim_min = next(m for m in range(0, 1440, 10) if sim.transit.has_service(m))
    assert TS.on_duty_crew(sim, sim_min) is None
    TS.phase(sim, 0, sim_min, len(sim.logger.events))
    rows = _kind(sim, "dwell_decision")
    assert len(rows) == 1 and rows[0].agent_id == -1
    e = rows[0]
    assert e.device_id == TS.operator_device_id(station) == e.payload["operator"]
    assert e.device_id.startswith("train_op:")


# ---- 7-7. 見送った種(関係のダイナミクスは装置ではない)------------------------ #
def test_on_relational_kinds_are_deliberately_unstamped(tmp_path):
    """★``reputation_update`` / ``relation_tier`` / ``rent`` に装置 id を作らない。

    どれも cause_type=device だが「それを管理する機関」が世界に居ない(関係は 2 人の
    あいだの状態・家主は rest-of-world)。id を作れば存在しない制度を捏造することになる。
    理由は observer/causality.py の見送り表に書いてある。
    """
    sim = _w3_run(tmp_path, "cz_w3_skip", **{"relations.enabled": "true"})
    seen = set()
    for e in sim.logger.events:
        if e.kind in ("reputation_update", "relation_tier", "relation_break",
                      "partner_formed", "rent"):
            seen.add(e.kind)
            assert e.device_id is None, f"{e.kind} に装置 id が付いた(制度の捏造)"
    assert seen, "検収の空回り(見送り対象の種が 1 件も出ていない)"
    src = (_ROOT / "src" / "society" / "observer" / "causality.py").read_text(
        encoding="utf-8")
    assert "関係のダイナミクスであって装置ではない" in src, "見送りの理由が書かれていない"
    assert "rest-of-world" in src


# ---- 7-8. 観測がシムを変えない / 決定論 / 名簿の閉包 --------------------------- #
def test_on_w3_stamps_do_not_change_the_world(tmp_path):
    """刻んでも既存 9 列の L1 は OFF と完全一致・LLM 呼数も同一。"""
    off = _sim(tmp_path, "cz_w3_w_off", n_steps=48, **W3)
    off.run()
    on = _w3_run(tmp_path, "cz_w3_w_on")
    assert _l1(off) == _l1(on)
    assert on.llm.calls == off.llm.calls > 0


def test_on_w3_is_deterministic(tmp_path):
    a = _w3_run(tmp_path, "cz_w3_det_a")
    b = _w3_run(tmp_path, "cz_w3_det_b")
    assert _l1c(a) == _l1c(b)


def test_on_w3_ids_are_all_in_the_catalogue(tmp_path):
    """W3 の点火セットでも未知 id は 1 つも出ない(名簿が閉じている)。"""
    sim = _w3_run(tmp_path, "cz_w3_closed")
    ids = _known_ids(sim)
    assert ids, "検収の空回り"
    assert all(_in_catalogue(i) for i in ids), sorted(ids)
    # W3 で狙った族が実際に出ている(空回りの検出)
    heads = {i.partition(":")[0] for i in ids}
    assert {"traffic", "pos", "org", "gov", "commerce"} <= heads, sorted(heads)


def test_on_w3_never_stamps_agent_physics_or_natural(tmp_path):
    """per-emit の刻印でも DEVICE_STAMPABLE の外には 1 件も付かない。"""
    sim = _w3_run(tmp_path, "cz_w3_discipline")
    for e in sim.logger.events:
        if e.cause_type in (C.AGENT, C.PHYSICS, C.NATURAL):
            assert e.device_id is None, f"{e.kind}({e.cause_type})に装置 id が付いた"


def test_analyze_device_breakdown_includes_the_new_ids(tmp_path):
    """事後解析: 突き合わせ 0 件のまま、W3 の装置が内訳に出る。"""
    from society import devices as D
    sim = _sim(tmp_path, "cz_w3_an", n_steps=48, **{**CAUSALITY_ON, **W3})
    sim.run()
    rep = ac.analyze(tmp_path / "cz_w3_an")
    dev = rep["devices"]
    assert dev["applicable"] is True and dev["unknown_device_ids"] == {}
    assert dev["n_stamped"] == sum(1 for e in sim.logger.events
                                   if e.device_id is not None) > 0
    assert dev["by_kind"]["traffic_flow"] == {
        D.DEV_TRAFFIC_AMBIENT: len(_kind(sim, "traffic_flow"))}
    assert dev["by_kind"]["tax"] == {D.DEV_GOV_TAX: len(_kind(sim, "tax"))}
    assert set(dev["by_kind"]["serve"]) == {
        D.pos_device_id(e.payload["node"]) for e in _kind(sim, "serve")}
    cv = rep["cross_validation"]
    assert cv["n_cause_mismatch"] == 0 and cv["n_actor_mismatch"] == 0
    # 接頭辞ロールアップ(W3 で足した節)と「まだ名乗れていない種」の一覧
    assert dev["per_prefix"][D.TRAFFIC_PREFIX] == len(_kind(sim, "traffic_flow"))
    # ★見送った/届いていない種が「まだ名乗れていない」側に正直に出る(wage の
    #   本業経路 = 雇い主が emit 点に無い。埋めずに残したことがレポートに現れる)
    assert "wage" in dev["stampable_without_device"]
    assert set(dev["stampable_without_device"]) & set(dev["by_kind"]) == set()
    md = ac.render_markdown(rep)
    assert D.DEV_TRAFFIC_AMBIENT in md and "装置種(接頭辞)別" in md
    assert ac.main([str(tmp_path / "cz_w3_an"), "--out", str(tmp_path / "w3.md")]) == 0
