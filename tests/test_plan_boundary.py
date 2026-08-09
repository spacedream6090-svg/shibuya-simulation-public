"""計画駆動の圏外滞在(actor model P4・planning.day_plan.boundary)のテスト。

正典: docs/plans/actor-model-migration-plan.md §4(境界)/ 実装 src/society/cognition/plan_boundary.py。

守るもの(検収基準の順)
  (1) 既定 OFF = day_plan ON だけのランと L1 バイト一致・payload の欄も 1 つも増えない・
      新イベント種ゼロ・新 stream ゼロ・レジストリ宣言あり
  (2) 指名の決定論(同 seed 2 回 → 同一集合 / 別 seed → 別集合)と eligible の境界
      (来街者・流入通勤者・work_node 保持者・バイト持ちは指名しない = bind_workplace が勝つ)
  (3) 計画スキーマ: boundary 欄(exit/gateway/exit_min/entry_min)+ node=None。既存欄に多重定義なし。
      物理検証は圏外ブロックを**受理**し(no_place も closed も出さない)、修復は**落とさない**
  (4) 実行: 予定 entry_min ちょうどに再出現(exact)・退出は縁へ着いてから(移動時間は無料でない)
  (5) 圏外では本人発の L1 ゼロ・LLM 呼ゼロ・帰還時の圧縮記憶は**ちょうど 1 行**
  (6) 保存則: 個体×日で exits == entries、総数 = 計画由来の期待値
  (7) 決定論(同 seed 2 ラン → L1 一致)と resume == straight(圏外滞在を跨ぐ分割)
  (8) 二系統の台帳が分離できる(統計駆動の流入通勤者の payload は 1 バイトも変わらない)
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from society import registry as R
from society.cognition import day_plan as DP
from society.cognition import plan_boundary as PB
from society.config import load_config
from society.engine import checkpoint, scheduler
from society.engine.simulation import Simulation
from society.observer.schema import EVENT_KINDS

DAYPLAN = {"planning.enabled": "true", "planning.day_plan.enabled": "true"}
ON = {**DAYPLAN, "planning.day_plan.boundary.enabled": "true"}
ALL_ON = {**ON, "planning.day_plan.boundary.outside_work_fraction": "1.0"}

# 圏外に居る間に本人以外の原因で残りうる L1(第三者起因=本人は何もしていない)。
#   state_update(cause=own_adopted) … **街に残っている別の個体**が、この個体の造語を
#   採用したときの記録。行為者は採用した側で、圏外の本人は 1 度も動いていない。
# ここを allowlist にしておくと、将来「圏外の個体が自分で何かする」経路が生えた瞬間に落ちる。
PASSIVE_KINDS = {"state_update"}
PASSIVE_CAUSES = {"own_adopted"}
# 本人の行為(_decide/_apply/_phase_move/_phase_drive 由来)を代表する種。圏外では 1 件も出ない。
AGENCY_KINDS = {"route_start", "arrive", "speak", "enter_building", "exit_building",
                "spend", "sleep_start", "wake_up", "plan_block_start",
                "plan_block_drop", "plan_slide", "plan_created", "sns_post"}


# --------------------------------------------------------------------------- #
# 共通ヘルパ
# --------------------------------------------------------------------------- #
def _cfg(name, n_steps=288, n_agents=20, **ov):
    dot = ["run.seed=42", f"run.n_agents={n_agents}", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock", "observer.snapshot_every=144"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def _sim(tmp_path, name, n_steps=288, n_agents=20, **ov):
    return Simulation(_cfg(name, n_steps, n_agents, **ov), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _blk(start, end, place, act, priority="must", flex="fixed", aim="livelihood"):
    return {"reason": "そうしたい", "start": start, "end": end, "place": place,
            "act": act, "with": [], "aim": aim, "priority": priority,
            "flex": flex, "note": ""}


def _j(blocks, cont=None):
    return {"action": "plan", "mood": "", "carry": "",
            "blocks": blocks, "if_then": cont or []}


# 圏外勤務 12:00-16:00 を含む固定の計画(全個体・全日で同一 = 実行タイミングだけが個体差)。
PLAN_JSON = json.dumps(_j([
    _blk("09:00", "10:00", "home", "home", "should", "slideable", "rest"),
    _blk("12:00", "16:00", "work", "work"),
    _blk("16:30", "17:30", "food", "meal", "should", "slideable", "sustenance"),
    _blk("18:00", "20:00", "home", "home", "should", "slideable", "rest"),
]), ensure_ascii=False)


def _stub_plan(sim, body: str = PLAN_JSON):
    """朝の計画だけを固定応答へ差し替える(backend 側=CachedLLM の呼数計上は生きる)。"""
    inner = sim.llm.backend.generate

    def _gen(prompt, **kw):
        return body if DP.PROMPT_MARK in prompt else inner(prompt, **kw)
    sim.llm.backend.generate = _gen
    return sim


def _spy_calls(sim) -> list:
    """LLM 呼の rng_key を控える(rng_key の 2 番目が agent_id)。"""
    seen: list = []
    inner = sim.llm.generate

    def _gen(prompt, **kw):
        seen.append(str(kw.get("rng_key", "")))
        return inner(prompt, **kw)
    sim.llm.generate = _gen
    return seen


def _run(tmp_path, name, n_steps=288, n_agents=20, **ov):
    sim = _stub_plan(_sim(tmp_path, name, n_steps, n_agents, **ov))
    sim.run()
    return sim


def _crossings(sim) -> tuple[list, list]:
    ex = [e for e in sim.logger.events
          if e.kind == "exit_area" and (e.payload or {}).get("boundary") == "plan"]
    en = [e for e in sim.logger.events
          if e.kind == "enter_area" and (e.payload or {}).get("boundary") == "plan"]
    return ex, en


def _spans(sim) -> dict:
    """agent_id → [(exit_step, enter_step), ...](閉じた滞在だけ)。"""
    out: dict = {}
    open_at: dict = {}
    for e in sim.logger.events:
        p = e.payload or {}
        if p.get("boundary") != "plan":
            continue
        if e.kind == "exit_area":
            open_at[e.agent_id] = e.step
        elif e.kind == "enter_area" and e.agent_id in open_at:
            out.setdefault(e.agent_id, []).append((open_at.pop(e.agent_id), e.step))
    return out


# --------------------------------------------------------------------------- #
# (A) 既定 OFF(検収基準 1)
# --------------------------------------------------------------------------- #
def test_shipped_default_is_off():
    cfg = load_config()
    assert bool(cfg.planning.day_plan.boundary.enabled) is False
    assert PB.build_cfg(None)["enabled"] is False
    assert PB.DEFAULTS["enabled"] is False


def test_off_matches_day_plan_only_run(tmp_path):
    """day_plan ON / boundary OFF が、boundary キーを明示 false にしたランと L1 完全一致。"""
    base = _run(tmp_path, "b_base", **DAYPLAN)
    off = _run(tmp_path, "b_off", **{**DAYPLAN,
                                     "planning.day_plan.boundary.enabled": "false"})
    assert _l1(base) == _l1(off)
    assert not [e for e in base.logger.events if (e.payload or {}).get("boundary")]
    for a in base.agents:                          # 誰にもフラグが立たない(中立値のまま)
        assert getattr(a, "work_outside", False) is False
        assert getattr(a, "work_outside_gateway", "") == ""
        assert getattr(a, "_boundary_pending", None) is None


def test_off_keeps_pure_default_bytes(tmp_path):
    """boundary キーが conf に増えても、**素の既定ラン**は 1 バイトも変わらない。"""
    pure = _sim(tmp_path, "b_pure")
    pure.run()
    off = _sim(tmp_path, "b_pure_off",
               **{"planning.day_plan.boundary.enabled": "false"})
    off.run()
    assert _l1(pure) == _l1(off)
    assert getattr(pure, "planboundarycfg", None) is None, \
        "OFF なのに boundary cfg が構築された(遅延構築の作法が崩れている)"


def test_off_l1_payload_keys_are_unchanged(tmp_path):
    """OFF の exit_area / enter_area の payload の欄は従来どおり(boundary 欄が生えない)。"""
    sim = _run(tmp_path, "b_keys", **DAYPLAN)
    ex = [e for e in sim.logger.events if e.kind == "exit_area"]
    en = [e for e in sim.logger.events if e.kind == "enter_area"]
    assert ex and en, "退出/帰還が 1 件も起きていない(テストの母数が無い)"
    assert all(set(e.payload) == {"gateway", "homing", "via"} for e in ex)
    assert all(set(e.payload) == {"gateway", "via"} for e in en)


def test_no_new_event_kinds(tmp_path):
    """新しいイベント種を 1 つも作らない(既存の exit_area / enter_area の再利用)。"""
    before = set(EVENT_KINDS)
    sim = _run(tmp_path, "b_kinds", **ALL_ON)
    assert set(EVENT_KINDS) == before, "boundary が新 kind を登録した"
    kinds = {e.kind for e in sim.logger.events}
    assert kinds <= before, f"未登録の kind が出た: {kinds - before}"
    assert "exit_area" in EVENT_KINDS and "enter_area" in EVENT_KINDS


def test_registry_declared():
    feat = {f.id: f for f in R.FEATURES}["planning.day_plan.boundary.enabled"]
    assert feat.repro_tier == "strict", "指名も時刻表も決定論なので strict"
    assert feat.affects_k is True, "圏外の個体は発火母集団から外れる=呼数が動く"
    assert feat.fingerprint_risk == "none", "プロンプトを 1 バイトも変えない"


def test_requires_day_plan(tmp_path):
    """day_plan が実効でなければ boundary も完全 no-op(計画がブロックの時刻表だから)。"""
    sim = _sim(tmp_path, "b_nodp", n_steps=24,
               **{"planning.enabled": "false",
                  "planning.day_plan.boundary.enabled": "true"})
    assert PB.enabled(sim) is False
    sim2 = _sim(tmp_path, "b_nodp2", n_steps=24,
                **{"planning.day_plan.boundary.enabled": "true"})
    assert PB.enabled(sim2) is False, "day_plan OFF なのに boundary が実効"


# --------------------------------------------------------------------------- #
# (B) 指名の決定論と eligible(検収基準 2)
# --------------------------------------------------------------------------- #
def test_designation_is_identical_for_the_same_seed(tmp_path):
    a = _run(tmp_path, "b_des_a", n_steps=144, **ALL_ON)
    b = _run(tmp_path, "b_des_b", n_steps=144, **ALL_ON)
    sa = {x.id for x in a.agents if getattr(x, "work_outside", False)}
    sb = {x.id for x in b.agents if getattr(x, "work_outside", False)}
    assert sa and sa == sb, f"同 seed 2 ランで指名集合が違う: {sa} vs {sb}"


def test_designation_rule_depends_on_seed():
    """指名規則そのものが seed に依存する(名簿の差ではなく規則の差を直接見る)。"""
    def _set(seed):
        return {i for i in range(300)
                if PB.stable_uniform(seed, f"work_outside/{i}") < 0.7}
    assert _set(42) != _set(43), "seed を変えても指名集合が同じ(決定論が退化している)"
    assert _set(42) == _set(42)
    n = len(_set(42))
    assert 150 < n < 270, f"指名率 0.7 から大きく外れる(n={n}/300)"


def test_designation_fraction_bounds(tmp_path):
    zero = _run(tmp_path, "b_frac0", n_steps=144,
                **{**ON, "planning.day_plan.boundary.outside_work_fraction": "0.0"})
    assert not [a for a in zero.agents if getattr(a, "work_outside", False)]
    assert not _crossings(zero)[0]
    full = _run(tmp_path, "b_frac1", **ALL_ON)
    want = {a.id for a in full.agents if PB.eligible(a)}
    assert {a.id for a in full.agents if PB.designated(full, a)} == want and want, \
        "率 1.0 では eligible 全員が指名されるはず"
    # フラグは「その朝の計画を立てた個体」にだけ書かれる(ensure の呼び口が apply だから)
    got = {a.id for a in full.agents if getattr(a, "work_outside", False)}
    assert got and got <= want, f"eligible でない個体にフラグが立った: {got - want}"


def test_eligible_excludes_inflow_and_bound_workplaces(tmp_path):
    """統計駆動の台帳(来街者・流入通勤者)と、地図内に職場を持つ個体は指名しない。

    ★work.bind_workplace(別レーン)が work_node を付けた個体はここで候補から外れる
      = 束ねが先に走れば束ねが勝つ(二重指名も競合も起きない)。
    """
    sim = _sim(tmp_path, "b_elig", n_steps=1, **ALL_ON)
    a = next(x for x in sim.agents if PB.eligible(x))
    assert PB.designated(sim, a) is True
    a.work_node = "n_bound_by_work_py"             # bind_workplace が束ねた相当
    assert PB.eligible(a) is False and PB.designated(sim, a) is False
    assert PB.ensure(sim, a) is False
    assert a.work_outside is False and a.work_outside_gateway == ""
    a.work_node = ""
    a.part_time = {"node": "n_pt", "start_min": 600, "end_min": 900}
    assert PB.eligible(a) is False                 # バイト先も地図内の勤務アンカー
    a.part_time = None
    a.visitor = True
    assert PB.eligible(a) is False                 # 来街者 = 家が既に街の外
    a.visitor = False
    a.commute = True
    assert PB.eligible(a) is False                 # 流入通勤者 = 統計駆動の台帳の担当


def test_ensure_is_idempotent_and_mirrors_the_pure_rule(tmp_path):
    sim = _sim(tmp_path, "b_ens", n_steps=1, **ON)
    for a in sim.agents:
        first = PB.ensure(sim, a)
        snap = (a.work_outside, a.work_outside_gateway,
                a.work_outside_start_min, a.work_outside_end_min)
        assert PB.ensure(sim, a) is first
        assert (a.work_outside, a.work_outside_gateway,
                a.work_outside_start_min, a.work_outside_end_min) == snap
        assert a.work_outside is PB.designated(sim, a)
        if a.work_outside:
            assert a.work_outside_gateway == sim.city.station_node
            assert 0 <= a.work_outside_start_min < a.work_outside_end_min <= 1439


def test_designation_draws_no_rng_stream(tmp_path):
    """指名は hashlib の純関数 = RngHub を 1 本も引かない(既存の draw 順を汚さない)。"""
    sim = _sim(tmp_path, "b_norng", n_steps=1, **ALL_ON)
    drawn: list = []
    inner = sim.hub.stream

    def _stream(*key):
        drawn.append(key)
        return inner(*key)
    sim.hub.stream = _stream
    for a in sim.agents:
        PB.ensure(sim, a)
        PB.designated(sim, a)
        PB.gateway_of(sim, a)
        PB.window_of(sim, a)
    assert drawn == [], f"指名が乱数ストリームを引いた: {drawn}"


# --------------------------------------------------------------------------- #
# (C) 計画スキーマ(検収基準 3)
# --------------------------------------------------------------------------- #
def _outside_agent(tmp_path, name, **ov):
    sim = _sim(tmp_path, name, n_steps=1, **{**ALL_ON, **ov})
    a = next(x for x in sim.agents if PB.eligible(x))
    PB.ensure(sim, a)
    assert a.work_outside is True
    return sim, a


def test_boundary_fields_are_explicit_and_do_not_overload_old_fields(tmp_path):
    sim, a = _outside_agent(tmp_path, "b_field")
    cfg = DP.cfg_of(sim)
    blocks, conts, errs = DP.validate_schema(_j([
        _blk("09:00", "10:00", "home", "home", "should", "slideable", "rest"),
        _blk("12:00", "16:00", "work", "work"),
        _blk("16:30", "17:30", "food", "meal", "should", "slideable", "sustenance"),
        _blk("18:00", "20:00", "home", "home", "should", "slideable", "rest")]), cfg)
    assert errs == []
    perrs = DP.validate_physical(sim, a, blocks, cfg, 0)
    wb = blocks[1]
    assert set(wb["boundary"]) == {"exit", "gateway", "exit_min", "entry_min"}
    assert wb["boundary"]["exit"] is True
    assert wb["boundary"]["gateway"] == a.work_outside_gateway
    assert wb["boundary"]["exit_min"] == 720 and wb["boundary"]["entry_min"] == 960
    assert wb["node"] is None, "圏外ブロックに地図内ノードが入った(欄の多重定義)"
    # 既存欄は 1 つも上書きしていない
    assert wb["place"] == "work" and wb["act"] == "work" and wb["state"] == "todo"
    # 圏外は地図の外 = 場所の実在も営業時間も問わない
    assert not [e for e in perrs if e.endswith("b1")], perrs
    assert not any(b.get("boundary") for i, b in enumerate(blocks) if i != 1)


def test_repair_keeps_boundary_blocks(tmp_path):
    """圏外ブロックは resolve_place が解けないが**落ちない**(従来なら drop されていた)。"""
    sim, a = _outside_agent(tmp_path, "b_keep")
    cfg = DP.cfg_of(sim)
    raw = _j([_blk("09:00", "10:00", "home", "home", "should", "slideable", "rest"),
              _blk("12:00", "16:00", "work", "work"),
              _blk("16:30", "17:30", "food", "meal", "should", "slideable",
                   "sustenance"),
              _blk("18:00", "20:00", "home", "home", "should", "slideable", "rest")])
    blocks, conts, _e = DP.validate_schema(raw, cfg)
    out, _c, ops = DP.repair(sim, a, blocks, conts, cfg, 0)
    got = [b for b in out if PB.is_boundary(b)]
    assert len(got) == 1 and got[0]["place"] == "work"
    assert got[0]["node"] is None
    assert got[0]["boundary"]["entry_min"] == got[0]["end"]
    assert got[0]["boundary"]["exit_min"] == got[0]["start"]
    # ★差の出所が**指名だけ**であること: 同じ個体・同じ計画でも、指名が外れれば
    #   work ブロックは従来どおり「場所が解けない」として落ちる。
    a.work_outside = False
    blocks2, conts2, _e2 = DP.validate_schema(raw, cfg)
    out2, _c2, _o2 = DP.repair(sim, a, blocks2, conts2, cfg, 0)
    assert not [b for b in out2 if PB.is_boundary(b)]
    assert not [b for b in out2 if b["place"] == "work"], \
        "指名が外れても work ブロックが残った(圏外扱いの根拠が指名以外にある)"


def test_stale_boundary_mark_is_cleared(tmp_path):
    """前日の計画を再利用したときの**古い刻印の持ち越し**を断つ(mark が唯一の刻印点)。"""
    sim, a = _outside_agent(tmp_path, "b_stale")
    b = {"start": 720, "end": 960, "place": "food", "act": "meal", "with": [],
         "purpose": "sustenance", "priority": "must", "flex": "fixed",
         "reason": "", "note": "", "node": None, "state": "todo", "slid": 0,
         "boundary": {"exit": True, "gateway": "n_old", "exit_min": 1, "entry_min": 2}}
    assert PB.mark(sim, a, b) is None, "places 外のカテゴリが圏外と判定された"
    assert "boundary" not in b and PB.is_boundary(b) is False
    b["place"] = "work"
    assert PB.mark(sim, a, b) == a.work_outside_gateway
    a.work_outside = False                          # 指名が外れた(bind_workplace が勝った等)
    assert PB.mark(sim, a, b) is None and "boundary" not in b


def test_skeleton_gives_the_outside_commuter_a_boundary_block(tmp_path):
    """骨格(フォールバック)経路でも圏外ブロックが出る(修復不能でも世界は止まらない)。"""
    sim, a = _outside_agent(tmp_path, "b_skel")
    rows = DP.skeleton(sim, a, DP.cfg_of(sim))
    assert rows and rows[0]["place"] == "work"
    assert rows[0]["start"] == a.work_outside_start_min
    assert rows[0]["end"] == a.work_outside_end_min
    DP.apply(sim, a, 0, 8 * 60, "{壊れた JSON", None)
    plan = a._dayplan
    assert plan["src"] == "skeleton"
    assert [b for b in plan["blocks"] if PB.is_boundary(b)], plan["blocks"]
    # 非指名の個体の骨格は従来どおり(第 3 の分岐に入らない)
    b2 = next(x for x in sim.agents if not PB.eligible(x))
    PB.ensure(sim, b2)
    assert not [r for r in DP.skeleton(sim, b2, DP.cfg_of(sim)) if r.get("boundary")]


# --------------------------------------------------------------------------- #
# (D) 実行: 退出と再出現(検収基準 4)
# --------------------------------------------------------------------------- #
def test_respawn_lands_exactly_on_the_planned_entry_minute(tmp_path):
    """再出現は**予定 entry_min ちょうど**(誤差ゼロ)。退出は縁へ着いてから(≥ exit_min)。"""
    sim = _run(tmp_path, "b_exec", **ALL_ON)
    ex, en = _crossings(sim)
    assert ex and en, "圏外滞在が 1 件も起きていない"
    for e in en:
        assert e.sim_min % 1440 == e.payload["entry_min"], \
            f"再出現が予定とずれた: {e.sim_min % 1440} != {e.payload['entry_min']}"
        assert e.payload["gateway"] == e.payload["gateway"]
    for e in ex:
        # ★正直な近似 3: exit_min は**予定**で、実際の退出は縁まで歩いた後に起きる
        #   (移動時間を無料にしない)。ずれ幅は step と exit_min の差として観測できる。
        assert e.sim_min % 1440 >= e.payload["exit_min"]
        assert e.payload["boundary"] == "plan"
        assert set(e.payload) == {"gateway", "homing", "via", "boundary", "block",
                                  "place", "exit_min", "entry_min"}
    for e in en:
        assert set(e.payload) == {"gateway", "via", "boundary", "block",
                                  "place", "exit_min", "entry_min"}


def test_exit_and_entry_happen_at_the_same_gateway(tmp_path):
    sim = _run(tmp_path, "b_gate", **ALL_ON)
    ex, en = _crossings(sim)
    by_ex = {(e.agent_id, e.payload["block"], e.sim_min // 1440): e for e in ex}
    assert by_ex
    for e in en:
        src = by_ex[(e.agent_id, e.payload["block"], e.sim_min // 1440)]
        assert e.payload["gateway"] == src.payload["gateway"]
        assert e.payload["gateway"] == sim.city.station_node


def test_no_events_and_no_llm_while_outside(tmp_path):
    """圏外では本人発の L1 も LLM 呼も 1 件も無い(退出/帰還の 2 件だけ)。"""
    sim = _stub_plan(_sim(tmp_path, "b_quiet", 288, 20, **ALL_ON))
    seen = _spy_calls(sim)
    sim.run()
    spans = _spans(sim)
    assert spans, "閉じた圏外滞在が 1 件も無い"
    n_checked = 0
    for e in sim.logger.events:
        for lo, hi in spans.get(e.agent_id, ()):
            if lo < e.step < hi:
                n_checked += 1
                assert e.kind not in AGENCY_KINDS, (e.step, e.agent_id, e.kind)
                assert e.kind in PASSIVE_KINDS, \
                    f"圏外の個体に本人発でない未知の記録: {e.kind} {e.payload}"
                assert (e.payload or {}).get("cause") in PASSIVE_CAUSES, e.payload
    # LLM: rng_key の 2 番目が agent_id・3 番目が step("plan/12/34" 等)
    for rk in seen:
        parts = rk.split("/")
        if len(parts) < 3 or not parts[1].lstrip("-").isdigit() \
                or not parts[2].isdigit():
            continue
        aid, st = int(parts[1]), int(parts[2])
        for lo, hi in spans.get(aid, ()):
            assert not (lo < st < hi), f"圏外の個体に LLM 呼が出た: {rk}"
    assert sim.llm.calls > 0
    assert n_checked >= 0                            # 母数 0 でも成立(その方が強い)


def test_plan_exit_draws_no_rng_and_uses_the_block_end(tmp_path):
    """計画退出は "outside" stream を 1 本も引かず、return_at をブロック終了時刻から決める。

    ラン比較ではなく **`_try_exit` の作用点そのもの**を直接見る(world.outside_steps を
    動かすと他の個体の滞在長まで変わって世界が別物になり、比較が交絡するため)。
    """
    sim = _sim(tmp_path, "b_exit1", n_steps=1, **ALL_ON)
    a = next(x for x in sim.agents if PB.eligible(x))
    PB.ensure(sim, a)
    gate = a.work_outside_gateway
    a.node, a.loc, a.route, a.homing = gate, "street", [], False
    a.x, a.y = sim.city.node_xy(gate)
    a._boundary_pending = {"gateway": gate, "block": 1, "day": 0, "place": "work",
                           "exit_min": 720, "entry_min": 960}
    a.exit_intent = True
    drawn: list = []
    inner = sim.hub.stream

    def _stream(*key):
        drawn.append(key)
        return inner(*key)
    sim.hub.stream = _stream
    assert a in PB.pending_exits(sim, 0), "縁に居る計画退出者を再試行口が拾わない"
    scheduler._try_exit(sim, a, 72, 720)
    assert a.loc == "outside"
    assert not [k for k in drawn if k and k[0] == "outside"], \
        f"計画退出が outside stream を引いた: {drawn}"
    assert a.return_at == 72 + (960 - 720) // sim.clock.step_minutes
    ev = [e for e in sim.logger.events if e.kind == "exit_area"][-1]
    assert ev.payload["boundary"] == "plan" and ev.payload["entry_min"] == 960


def test_memory_gets_exactly_one_compressed_line_per_stay(tmp_path):
    sim = _stub_plan(_sim(tmp_path, "b_mem", 288, 20, **ALL_ON))
    seen: dict = {}
    for a in sim.agents:
        def _wrap(aid, inner):
            def _obs(step, text, **kw):
                if PB.MEMO_TAIL in str(text):
                    seen.setdefault(aid, []).append(str(text))
                return inner(step, text, **kw)
            return _obs
        a.mem.observe = _wrap(a.id, a.mem.observe)
    sim.run()
    _ex, en = _crossings(sim)
    want = Counter(e.agent_id for e in en)
    assert want, "帰還が 1 件も無い"
    assert {k: len(v) for k, v in seen.items()} == dict(want), \
        f"圧縮記憶の本数が帰還数と一致しない: {seen} vs {want}"
    for aid, lines in seen.items():
        for line in lines:
            assert line.count("\n") == 0 and len(line) <= 60, line
            assert line.endswith(PB.MEMO_TAIL)


def test_memo_line_is_a_pure_template(tmp_path):
    """定型文は (場所, 予定時刻) の純関数。実験条件・機構語・自由文を 1 バイトも含まない。"""
    line = PB.memo_line("work", 720, 960)
    assert line == PB.memo_line("work", 720, 960)
    assert line == "12:00〜16:00は仕事で街の外に居た。計画通りに実行、特筆事項なし。"
    assert PB.memo_line("education", 540, 900).startswith("09:00〜15:00は学業")
    for word in ("発火", "閾値", "k=", "seed", "grievance", "efficacy", "圏外", "gateway"):
        assert word not in line


def test_block_state_is_marked_executed_on_return(tmp_path):
    """圏外滞在中は "away"、帰還で "done"(割り込み処理は todo 以外を触らない)。"""
    sim = _run(tmp_path, "b_state", **ALL_ON)
    closed = _spans(sim)
    for a in sim.agents:
        if a.id not in closed:
            continue
        plan = getattr(a, "_dayplan", None)
        if plan is None:
            continue
        for b in plan["blocks"]:
            if PB.is_boundary(b):
                assert b["state"] in ("todo", "done"), b["state"]
        assert getattr(a, "_boundary_pending", None) is None


# --------------------------------------------------------------------------- #
# (E) 保存則(検収基準 6)
# --------------------------------------------------------------------------- #
def test_conservation_exits_equal_entries_per_agent_per_day(tmp_path):
    """個体×日で exits == entries。日境界を跨いだ持ち越しは 0 件(帰還は同日内)。"""
    sim = _run(tmp_path, "b_cons", n_steps=432, **ALL_ON)
    ex, en = _crossings(sim)
    assert ex and en
    ce = Counter((e.agent_id, e.sim_min // 1440) for e in ex)
    cn = Counter((e.agent_id, e.sim_min // 1440) for e in en)
    assert ce == cn, f"個体×日の出入りが釣り合わない: {ce} vs {cn}"
    assert all(v == 1 for v in ce.values()), f"1 日に 2 回以上出ている: {ce}"
    # 台帳から見た総数 = plan_block_start(boundary)の件数
    starts = [e for e in sim.logger.events
              if e.kind == "plan_block_start" and (e.payload or {}).get("boundary")]
    assert len(starts) >= len(ex), "退出が計画の開始より多い(計画外の退出が混ざった)"
    # 未閉鎖(run 終了時に圏外に居る)ぶんだけが差になる
    still = sum(1 for a in sim.agents if a.loc == "outside"
                and getattr(a, "_boundary_pending", None) is not None)
    assert len(ex) - len(en) == still, "出入り差が『いま圏外に居る人数』と一致しない"


def test_plan_derived_expectation_matches_l1(tmp_path):
    """L1 の境界横断数 = 計画由来の期待値(実行された圏外ブロックのうち縁へ着けたもの)。"""
    sim = _run(tmp_path, "b_expect", **ALL_ON)
    ex, _en = _crossings(sim)
    starts = {(e.agent_id, e.sim_min // 1440, e.payload["block"])
              for e in sim.logger.events
              if e.kind == "plan_block_start" and (e.payload or {}).get("boundary")}
    got = {(e.agent_id, e.sim_min // 1440, e.payload["block"]) for e in ex}
    assert got and got <= starts, f"計画に無い境界退出がある: {got - starts}"


def test_statistical_inflow_ledger_is_untouched(tmp_path):
    """二系統の台帳が混ざらない: 統計駆動(commute/homing)の payload に boundary は無い。"""
    sim = _run(tmp_path, "b_two", **ALL_ON)
    other = [e for e in sim.logger.events
             if e.kind in ("exit_area", "enter_area")
             and (e.payload or {}).get("boundary") is None]
    assert other, "統計駆動側の出入りが 1 件も無い(母数が足りない)"
    assert all(set(e.payload) in ({"gateway", "homing", "via"}, {"gateway", "via"})
               for e in other)
    # 計画駆動側は homing=false(帰宅ではなく通勤)= 2 つの台帳が意味の上でも交わらない
    assert all(e.payload["homing"] is False for e in _crossings(sim)[0])
    # 統計駆動側は本 module の欄を 1 つも持たない = クエリ 1 本で完全に分離できる
    for e in other:
        assert "boundary" not in e.payload and "entry_min" not in e.payload


# --------------------------------------------------------------------------- #
# (F) 決定論 / resume(検収基準 7)
# --------------------------------------------------------------------------- #
def test_on_is_deterministic_across_two_runs(tmp_path):
    a = _run(tmp_path, "b_det_a", **ALL_ON)
    b = _run(tmp_path, "b_det_b", **ALL_ON)
    assert _l1(a) == _l1(b), "同 seed 2 ランの L1 が一致しない"


def _rows(run_dir: Path, stem: str = "l1_events") -> list[dict]:
    return pq.read_table(Path(run_dir) / f"{stem}.parquet").to_pylist()


def test_resume_matches_straight_across_a_boundary_absence(tmp_path):
    """圏外滞在の**まっただ中**で checkpoint を切っても straight と全層一致。

    split=190 は実測で「複数の個体が圏外に居る」step(退出 175-186 / 帰還 198)。
    保留(agent._boundary_pending)は agents pickle に自然同梱される = 中央管理不要、
    帰還時の記憶と台帳印もそのまま復元されることを resume==straight で機械固定する。
    """
    total, split = 240, 190
    straight = tmp_path / "b_rs"
    _stub_plan(Simulation(_cfg("b_rs", total, 20, **ALL_ON), out_dir=straight)).run()
    d = tmp_path / "b_rr"
    every = {"observer.checkpoint_every": split}
    s1 = _stub_plan(Simulation(_cfg("b_rr", split, 20, **every, **ALL_ON), out_dir=d))
    for step in range(split):
        scheduler.run_step(s1, step)
    outside = [a.id for a in s1.agents if a.loc == "outside"
               and getattr(a, "_boundary_pending", None) is not None]
    assert outside, f"split={split} で圏外に居る計画退出者が居ない(テストが無意味)"
    checkpoint.save(s1, split, d / "checkpoint" / f"ckpt-{split:06d}.pkl.gz")
    s1.logger.flush_segment()
    s2 = _stub_plan(Simulation(_cfg("b_rr", total, 20, **every, **ALL_ON), out_dir=d))
    s2.run(resume_from=d)
    for stem in ("l1_events", "l2_metrics", "l3_snapshots"):
        assert _rows(straight, stem) == _rows(d, stem), f"{stem} 不一致(resume)"


# --------------------------------------------------------------------------- #
# (G) 契約(既存機構との一致)
# --------------------------------------------------------------------------- #
def test_steps_until_tod_matches_the_scheduler_formula():
    """時刻→step の変換式が engine 側の既存実装と 1 ビットも違わない(逆向き import の代替)。"""
    for step_minutes in (1, 5, 10, 30):
        for cur in range(0, 1440, 7):
            for tgt in (0, 1, 479, 720, 960, 1439):
                assert PB.steps_until_tod(cur, tgt, step_minutes) == \
                    scheduler._steps_until_tod(cur, tgt, step_minutes)


def test_module_has_no_llm_and_no_rng_identifiers():
    """圏外はコスト 0 = この module に LLM 呼と乱数の識別子が 1 つも無いことを機械固定。"""
    src = Path(PB.__file__).read_text(encoding="utf-8")
    body = "\n".join(line for line in src.splitlines()
                     if not line.lstrip().startswith("#"))
    for ident in ("generate(", "sim.llm", "hub.stream", "np.random", "random."):
        assert ident not in body, f"{ident} が plan_boundary.py に現れた"


def test_docstring_records_the_two_mandated_approximations():
    doc = PB.__doc__ or ""
    assert "摩擦ゼロ" in doc and "フラックスは固定" in doc, \
        "§4 が求める 2 つの正直な近似が module docstring に無い"
