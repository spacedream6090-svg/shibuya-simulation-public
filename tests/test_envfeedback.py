"""環境フィードバック(エージェント → 環境)最小 3 規則 第84バッチ(env.feedback)のテスト。

正典: docs/plans/source/cognition-design-record.md §4 / cognition-physics-plan.md §4 第84行。

守るもの(重要度順):
  (1) **既定 OFF は 1 バイトも変えない**: 明示 OFF と純粋既定の L1 完全一致・状態(_envfb)不在・
      env_feedback 0 件・乱数 draw 総数一致・filter_open が同一オブジェクトを返す。
  (2) **T5 非発散**(§6.2): 人工的な高負荷でも遅延が上限内に収束し、負荷を切れば回復運転項で
      0 へ戻る。個体は max_hold_steps を超えて待たされない(必ず通す安全弁)= 駅が詰まらない。
  (3) 3 規則それぞれの発火条件・減衰・上限・L1 の追跡可能性(§4.3「どの集約量が・どの閾値を・
      どれだけ超えたか」)。
  (4) ON の再現性: 同 seed 2 ラン一致・resume==straight(_envfb の checkpoint 中央管理)。
  (5) 環が閉じたことの観測: 第80 の ext.transit_delay が ON ランで初めて非ゼロ分散を持つ。
  (6) 第67 の予約フィールド world.mod.gate_capacity が実際に消費される。
検証は mock のみ(実 LLM 禁止)。
"""
from __future__ import annotations

import json

import pytest
import pyarrow.parquet as pq

from society import envfeedback as envfb
from society import commerce as _commerce
from society import registry as R
from society import timeconv as T
from society.config import load_config
from society.engine import checkpoint, scheduler
from society.engine.simulation import Simulation
from society.observer.schema import Event

# 20 体の mock で実際に 3 規則が発火する縮約(既定の閾値は本番規模向けなので小規模では効かない)。
_ON = {"env.feedback.enabled": "true",
       "env.feedback.log_every_steps": "1",
       "env.feedback.transit.platform_threshold": "1",
       "env.feedback.gate.capacity_per_min": "0.05",
       "env.feedback.poi.capacity": "1"}


def _sim(tmp_path, name, n=20, steps=1, **ov):
    dot = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
           "run.seed=42", "model.backend=mock", "observer.snapshot_every=144"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _kind(sim, kind):
    return [e for e in sim.logger.events if e.kind == kind]


def _rules(sim, rule):
    return [e for e in sim.logger.events
            if e.kind == envfb.EVENT_KIND and e.payload.get("rule") == rule]


class _CountingHub:
    """全 stream の draw を数えるプロキシ(OFF の乱数消費不変を直接固定する)。"""

    def __init__(self, inner):
        self._inner = inner
        self.draws = 0

    def stream(self, *key):
        g = self._inner.stream(*key)
        outer = self

        class _W:
            def __getattr__(self, name):
                attr = getattr(g, name)
                if name in ("random", "integers", "choice", "normal", "shuffle",
                            "permutation", "uniform"):
                    def _wrapped(*a, **k):
                        outer.draws += 1
                        return attr(*a, **k)
                    return _wrapped
                return attr

        return _W()

    def key_name(self, *key):
        return self._inner.key_name(*key)

    @property
    def master_seed(self):
        return self._inner.master_seed


def _station_crowd(sim, n=None):
    """合成高負荷: 指定人数を駅ノードへ物理的に置く(起きている・街内)。"""
    station = sim.city.station_node
    agents = sim.agents if n is None else sim.agents[:n]
    for a in agents:
        a.loc = "street"
        a.node = station
        a.sleeping = False
        a.route = []
    return station


# =========================================================================== #
# (1) 既定 OFF = 1 バイトも変えない
# =========================================================================== #
def test_off_matches_pure_default(tmp_path):
    """明示 OFF と純粋既定が L1 完全一致(144 step)。_envfb 不在・env_feedback 0 件。"""
    pure = _sim(tmp_path, "pure", steps=144)
    pure.run()
    off = _sim(tmp_path, "expl_off", steps=144, **{"env.feedback.enabled": "false"})
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(環境FB の seam が no-op でない)"
    assert len(pure.logger.llm_calls) == len(off.logger.llm_calls) > 0, "LLM 呼数が変わった"
    assert getattr(off, "_envfb", None) is None, "OFF なのに _envfb が生えている"
    assert _kind(off, envfb.EVENT_KIND) == [], "OFF なのに env_feedback イベントが出ている"
    assert envfb.enabled(off) is False
    assert envfb.delay_min(off) == 0.0 and envfb.delay_flag(off) == 0.0
    assert envfb.blocked_nodes(off) == frozenset()
    assert envfb.pending_exits(off, 0) == ()


def test_off_draw_count_unchanged(tmp_path):
    """OFF は乱数 draw 総数も完全一致(専用 stream を含め 1 本も引かない)。"""
    base = _sim(tmp_path, "dr_base", steps=72)
    base.hub = _CountingHub(base.hub)
    base.run()
    off = _sim(tmp_path, "dr_off", steps=72, **{"env.feedback.enabled": "false"})
    off.hub = _CountingHub(off.hub)
    off.run()
    assert base.hub.draws == off.hub.draws > 0, \
        f"draw 数が変わった: {base.hub.draws} vs {off.hub.draws}"


def test_on_draws_no_random_numbers(tmp_path):
    """★ON でも乱数を 1 本も**足さない**(規則は完全決定論=repro_tier strict の根拠)。

    ON では物理位置が変わるので既存機構の draw 数は動きうるが、環境FB 自身は RngHub に
    触れない。ここでは「合成シナリオで update/hold を直接叩いても draw が 0」を固定する。
    """
    sim = _sim(tmp_path, "nodraw", n=20, steps=1, **_ON)
    sim.hub = _CountingHub(sim.hub)
    _station_crowd(sim)
    for s in range(20):
        envfb.update(sim, s, s * 10, -1)
        envfb.hold_exit(sim, sim.agents[0], s, s * 10)
        envfb.blocked_nodes(sim)
    assert sim.hub.draws == 0, f"環境FB が乱数を引いた: {sim.hub.draws}"


def test_filter_open_identity_when_everything_off(tmp_path):
    """commerce OFF かつ環境FB OFF では filter_open が**同一オブジェクト**を返す(既存契約)。"""
    sim = _sim(tmp_path, "fo_off")
    pois = sim.city.pois_by_cat("food")
    assert _commerce.filter_open(sim, pois, 12 * 60) is pois


# =========================================================================== #
# (2) T5 非発散(設計 §6.2)
# =========================================================================== #
def test_t5_delay_converges_within_bound_under_extreme_load(tmp_path):
    """★T5: 人工的な高負荷で遅延が**上限内に収束**する(発散しない)。

    不動点は減衰項から解析的に決まる: D* = dwell_cap_min/(1−γ)(かつ delay_cap_min 以下)。
    ここでは 3.0/(1−0.7)=10.0 分 < 上限 15.0 分 = **ハード上限ではなく回復運転項が効いて
    いる**ことを固定する(cap だけで押さえていたら『発散を止めた』とは言えない)。
    """
    sim = _sim(tmp_path, "t5", n=40, steps=1, **{**_ON,
                                                 "env.feedback.transit.platform_threshold": "0",
                                                 "env.feedback.transit.dwell_sec_per_pax": "60.0"})
    _station_crowd(sim)                      # 40 人 × 60 秒 = 40 分ぶんの注入圧(上限 3 分で頭打ち)
    tcfg = sim.envfbcfg["transit"]
    bound = min(tcfg["delay_cap_min"], tcfg["dwell_cap_min"] / (1.0 - tcfg["recovery"]))
    assert bound == pytest.approx(10.0) and bound < tcfg["delay_cap_min"]
    trace = []
    for s in range(80):
        envfb.update(sim, s, s * 10, -1)
        trace.append(envfb.delay_min(sim))
    assert max(trace) <= bound + 1e-9, f"上限 {bound} を超えた: max={max(trace)}"
    assert all(b >= a - 1e-12 for a, b in zip(trace, trace[1:])), "単調増加でない(振動している)"
    assert abs(trace[-1] - trace[-2]) < 1e-9, f"収束していない: {trace[-3:]}"
    assert trace[-1] == pytest.approx(bound, abs=1e-9), \
        f"不動点に達していない: {trace[-1]} vs {bound}"

    # 負荷を切る → 回復運転項 γ<1 で幾何級数的に 0 へ戻る(減衰項が実在することの直接検証)
    for a in sim.agents:
        a.loc = "outside"
    tail = []
    for s in range(80, 200):
        envfb.update(sim, s, s * 10, -1)
        tail.append(envfb.delay_min(sim))
    assert all(b <= a + 1e-12 for a, b in zip(tail, tail[1:])), "回復中に増えている"
    assert tail[0] == trace[-1] * tcfg["recovery"], "1 step の減衰が γ 倍でない"
    assert tail[-1] == 0.0, f"遅延が 0 に戻らない: {tail[-1]}"


def test_t5_individual_wait_is_bounded_and_always_released(tmp_path):
    """★T5(個体側): 遅延・規制が続いても 1 個体が待たされる合計は max_hold_steps まで。

    これが「駅に人が溜まり続けて二度と捌けない」発散を原理的に止める安全弁。
    """
    sim = _sim(tmp_path, "t5b", n=10, steps=1, **{**_ON,
                                                  "env.feedback.transit.platform_threshold": "0",
                                                  "env.feedback.transit.dwell_sec_per_pax": "60.0"})
    _station_crowd(sim)
    for s in range(30):                       # 遅延を上限まで積む
        envfb.update(sim, s, s * 10, -1)
    assert envfb.delay_min(sim) > 0.0
    cap = sim.envfbcfg["transit"]["max_hold_steps"]
    agent = sim.agents[0]
    held = 0
    for s in range(30, 60):
        envfb.update(sim, s, s * 10, -1)      # 遅延は上限で持続したまま
        if envfb.hold_exit(sim, agent, s, s * 10):
            held += 1
        else:
            break
    else:                                     # pragma: no cover - 落ちたら発散している
        raise AssertionError("待ちが解除されない(発散)")
    assert int(getattr(agent, "_env_exit_held", 0)) <= cap, "待ちの合計が上限を超えた"
    assert held >= 1, "遅延中なのに一度も待たされていない(テスト前提が崩れた)"


def test_t5_station_drains_in_a_full_run(tmp_path):
    """★T5(結合): 全員を駅へ集めた高負荷ランで、駅の滞在人数が最後には捌けている。

    規則1(遅延で待たせる)と規則2(入場規制)を同時に効かせても、上限と解除で必ず流れる。
    """
    sim = _sim(tmp_path, "t5c", n=30, steps=60, **{**_ON,
                                                   "env.feedback.transit.platform_threshold": "0",
                                                   "env.feedback.transit.dwell_sec_per_pax": "60.0"})
    station = _station_crowd(sim)
    for a in sim.agents:
        a.exit_intent = True
        a.stay_until = 0
    peak = 0
    for step in range(60):
        scheduler.run_step(sim, step)
        n = sum(1 for a in sim.agents if a.loc != "outside" and a.node == station)
        peak = max(peak, n)
        assert envfb.delay_min(sim) <= sim.envfbcfg["transit"]["delay_cap_min"] + 1e-9
    left = sum(1 for a in sim.agents if a.loc != "outside" and a.node == station)
    assert peak >= 5, "テスト前提が崩れた(駅に人が溜まっていない)"
    assert left < peak, f"駅が捌けていない(発散): peak={peak} left={left}"
    assert _rules(sim, envfb.RULE_TRANSIT), "遅延規則が一度も発火していない"


# =========================================================================== #
# (3) 規則1: ホーム密度 → 停車時間延長 → 遅延伝播
# =========================================================================== #
def test_rule1_threshold_and_payload_traceability(tmp_path):
    """閾値を超えた分だけ注入され、L1 から『どの集約量が・どの閾値を・どれだけ超えたか』
    が辿れる(§4.3)。閾値以下では遅延が生まれない。"""
    sim = _sim(tmp_path, "r1", n=20, steps=1,
               **{**_ON, "env.feedback.transit.platform_threshold": "5",
                  "env.feedback.gate.enabled": "false",
                  "env.feedback.poi.enabled": "false"})
    _station_crowd(sim, n=5)                    # 閾値ちょうど=超過なし
    for a in sim.agents[5:]:
        a.loc = "outside"
    envfb.update(sim, 0, 0, -1)
    assert envfb.delay_min(sim) == 0.0, "閾値以下で遅延が生まれている"
    assert _rules(sim, envfb.RULE_TRANSIT) == []

    _station_crowd(sim, n=20)                   # 20 人 = 15 人の超過
    envfb.update(sim, 1, 10, -1)
    ev = _rules(sim, envfb.RULE_TRANSIT)
    assert len(ev) == 1
    p = ev[0].payload
    assert p["metric"] == "platform_n" and p["value"] == 20 and p["threshold"] == 5
    assert p["excess"] == 15 and p["node"] == sim.city.station_node
    assert p["injected_min"] == round(15 * 0.8 / 60.0, 3)
    assert p["delay_min"] == p["injected_min"] and p["prev_min"] == 0.0
    assert ev[0].agent_id == -1, "世界イベントは agent_id=-1"


def test_rule1_delays_the_return_from_outside(tmp_path):
    """遅延中は駅経由の帰還が遅れる(到着時刻に反映される)。上限は max_hold_steps。"""
    sim = _sim(tmp_path, "r1b", n=10, steps=1,
               **{**_ON, "env.feedback.transit.platform_threshold": "0",
                  "env.feedback.transit.dwell_sec_per_pax": "60.0"})
    _station_crowd(sim)
    for s in range(30):
        envfb.update(sim, s, s * 10, -1)
    agent = sim.agents[0]
    agent.loc = "outside"
    agent.return_gateway = sim.city.station_node
    # 実際の呼び出し側(_phase_wake_and_returns)は False を返した step に帰着させるので、
    # 1 トリップぶん = False が返るまでを数える。
    waits = 0
    for s in range(40, 60):
        if not envfb.hold_return(sim, agent, s, s * 10):
            break
        waits += 1
    else:                                     # pragma: no cover - 落ちたら発散している
        raise AssertionError("帰還の待ちが解除されない(発散)")
    assert 0 < waits <= sim.envfbcfg["transit"]["max_hold_steps"], \
        f"帰還の待ちが上限内でない: {waits}"
    ev = [e for e in sim.logger.events
          if e.kind == envfb.EVENT_KIND and e.payload.get("effect") == "return_wait"]
    assert ev and ev[0].agent_id == agent.id


# =========================================================================== #
# (4) 規則2: 改札スループット飽和 → 入場規制
# =========================================================================== #
def _inject_arrivals(sim, step, n):
    """駅ノードへの到着 L1 を n 件注入し、注入前の index を返す。"""
    idx = len(sim.logger.events)
    for i in range(n):
        sim.logger.log(Event(step=step, sim_min=step * 10, agent_id=i, kind="arrive",
                             x=0.0, y=0.0,
                             payload={"node": sim.city.station_node, "name": "",
                                      "first_visit": False}))
    return idx


def test_rule2_onset_hold_clear_and_cooldown(tmp_path):
    """流入 > 改札容量で規制が発動し、hold_steps で必ず解除され、cooldown 中は再発動しない。"""
    sim = _sim(tmp_path, "r2", n=20, steps=1,
               **{**_ON, "env.feedback.transit.enabled": "false",
                  "env.feedback.poi.enabled": "false",
                  "env.feedback.gate.capacity_per_min": "0.3",   # cap = 3 人/step(Δt=10)
                  "env.feedback.gate.hold_steps": "2",
                  "env.feedback.gate.cooldown_steps": "3"})
    for a in sim.agents:
        a.loc = "outside"
    idx = _inject_arrivals(sim, 0, 3)                       # 容量ちょうど=発動しない
    envfb.update(sim, 0, 0, idx)
    assert not envfb.gate_active(sim, 0)
    idx = _inject_arrivals(sim, 1, 4)                       # 容量超過=発動
    envfb.update(sim, 1, 10, idx)
    assert envfb.gate_active(sim, 1)
    onset = _rules(sim, envfb.RULE_GATE)
    assert len(onset) == 1 and onset[0].payload["phase"] == "onset"
    assert onset[0].payload["value"] == 4 and onset[0].payload["threshold"] == 3.0
    assert onset[0].payload["until_step"] == 3

    for s in (2,):                                          # 規制中(解除 step 未満)
        envfb.update(sim, s, s * 10, len(sim.logger.events))
        assert envfb.gate_active(sim, s)
    idx = _inject_arrivals(sim, 3, 10)                      # 解除 step。流入超過でも cooldown 中
    envfb.update(sim, 3, 30, idx)
    assert not envfb.gate_active(sim, 3), "hold_steps を過ぎても解除されない"
    clears = [e for e in _rules(sim, envfb.RULE_GATE) if e.payload.get("phase") == "clear"]
    assert len(clears) == 1, "解除が記録されていない"
    for s in (4, 5):
        idx = _inject_arrivals(sim, s, 10)
        envfb.update(sim, s, s * 10, idx)
        assert not envfb.gate_active(sim, s), f"cooldown 中に再発動した(step={s})"
    idx = _inject_arrivals(sim, 6, 10)                      # cooldown 明け=再発動できる
    envfb.update(sim, 6, 60, idx)
    assert envfb.gate_active(sim, 6)


def test_rule2_consumes_world_mod_gate_capacity(tmp_path):
    """★第67 の予約フィールド world.mod.gate_capacity を**ここで初めて消費**する。

    係数 0.5 で容量が半分になり、summary の reserved 記録が consumed=true になる。
    """
    prof = tmp_path / "gate_profile.yaml"
    prof.write_text("name: gate_half\ngate_capacity:\n  station: 0.5\n", encoding="utf-8")
    ov = {**_ON, "world.mod.enabled": "true", "world.mod.profile": str(prof).replace("\\", "/"),
          "env.feedback.gate.capacity_per_min": "0.4"}       # cap = 4 → 係数 0.5 で 2
    sim = _sim(tmp_path, "gc", n=8, steps=1, **ov)
    assert envfb.gate_scale(sim) == 0.5
    for a in sim.agents:
        a.loc = "outside"
    idx = _inject_arrivals(sim, 0, 3)                        # 3 > 2 = 半分になった容量を超える
    envfb.update(sim, 0, 0, idx)
    assert envfb.gate_active(sim, 0), "gate_capacity 係数が効いていない"
    assert _rules(sim, envfb.RULE_GATE)[0].payload["threshold"] == 2.0
    rsv = sim.worldmod.summary()["reserved_not_consumed"]["gate_capacity"]
    assert rsv["consumed"] is True and rsv["consumer"] == "env.feedback.gate"

    # 環境FB OFF なら第67 の契約(未消費)のまま
    off = _sim(tmp_path, "gc_off", n=8, steps=1,
               **{"world.mod.enabled": "true",
                  "world.mod.profile": str(prof).replace("\\", "/")})
    assert off.worldmod.summary()["reserved_not_consumed"]["gate_capacity"]["consumed"] is False


# =========================================================================== #
# (5) 規則3: POI 占有 > 容量 → 待ち行列 → 他 POI へ流出
# =========================================================================== #
def test_rule3_marks_expires_and_is_bounded(tmp_path):
    """容量超過ノードが待ち行列として記録され、hold_steps で自動失効し、max_nodes で頭打ち。"""
    sim = _sim(tmp_path, "r3", n=20, steps=1,
               **{**_ON, "env.feedback.transit.enabled": "false",
                  "env.feedback.gate.enabled": "false",
                  "env.feedback.poi.capacity": "2",
                  "env.feedback.poi.hold_steps": "3",
                  "env.feedback.poi.max_nodes": "2"})
    nodes = sorted({str(p["node"]) for p in sim.city.poi_list})[:4]
    for i, a in enumerate(sim.agents[:12]):                  # 各ノードに 3 人ずつ(容量 2 超過)
        a.loc = "street"
        a.sleeping = False
        a.node = nodes[i // 3]
    for a in sim.agents[12:]:
        a.loc = "outside"
    envfb.update(sim, 0, 0, -1)
    blocked = envfb.blocked_nodes(sim)
    assert len(blocked) == 2, f"max_nodes の上限が効いていない: {sorted(blocked)}"
    assert blocked == frozenset(nodes[:2]), "並びが決定論でない(ノード id 昇順のはず)"
    ev = _rules(sim, envfb.RULE_POI)
    assert len(ev) == 2
    assert ev[0].payload["metric"] == "occupancy" and ev[0].payload["value"] == 3
    assert ev[0].payload["threshold"] == 2 and ev[0].payload["excess"] == 1
    assert ev[0].payload["until_step"] == 3
    for a in sim.agents[:12]:                                # 混雑を解消
        a.loc = "outside"
    envfb.update(sim, 1, 10, -1)
    assert len(envfb.blocked_nodes(sim)) == 2, "hold_steps 以前に解除された"
    envfb.update(sim, 3, 30, -1)
    assert envfb.blocked_nodes(sim) == frozenset(), "hold_steps を過ぎても失効しない"


def test_rule3_filter_open_excludes_but_never_empties(tmp_path):
    """混雑ノードは行き先候補から外れる(既存 filter_open 経路)。全滅するときは外さない。"""
    sim = _sim(tmp_path, "r3b", n=8, steps=1, **_ON)
    envfb.state(sim)
    foods = sim.city.pois_by_cat("food")
    assert len(foods) >= 2, "テスト前提(food POI が 2 件以上)が崩れた"
    sim._envfb["poi_hold"] = {str(foods[0]["node"]): 99}
    kept = _commerce.filter_open(sim, foods, 12 * 60)
    assert all(p["node"] != foods[0]["node"] for p in kept), "混雑ノードが候補に残っている"
    assert len(kept) < len(foods)
    # ★安全弁: 候補が全滅する除外はしない(行き先を失って世界が固まるのを防ぐ)
    sim._envfb["poi_hold"] = {str(p["node"]): 99 for p in foods}
    assert _commerce.filter_open(sim, foods, 12 * 60) == foods


# =========================================================================== #
# (6) ON の再現性(同 seed 2 ラン一致 / resume==straight)
# =========================================================================== #
def test_on_runs_and_two_runs_are_identical(tmp_path):
    """ON で 288 step 完走し、同 seed の 2 ランが L1 完全一致(決定論)。"""
    a = _sim(tmp_path, "on_a", n=20, steps=288, **_ON)
    a.run()
    b = _sim(tmp_path, "on_b", n=20, steps=288, **_ON)
    b.run()
    assert _l1(a) == _l1(b), "同 seed の 2 ランが不一致(決定論が壊れている)"
    assert _kind(a, envfb.EVENT_KIND), "ON なのに env_feedback が 1 件も出ていない"
    assert getattr(a, "_envfb", None) is not None


def test_resume_matches_straight(tmp_path):
    """ON の resume==straight(_envfb の checkpoint 中央管理)。"""
    def _cfg(name, n_steps, **extra):
        dot = ["run.seed=42", "run.n_agents=20", f"run.n_steps={n_steps}",
               f"run.name={name}", "model.backend=mock"]
        dot += [f"{k}={v}" for k, v in {**_ON, **extra}.items()]
        return load_config(dot)

    straight_dir = tmp_path / "e_straight"
    straight = Simulation(_cfg("e_straight", 120), out_dir=straight_dir)
    straight.run()
    assert _kind(straight, envfb.EVENT_KIND), "テスト前提が崩れた(環境イベントが出ていない)"

    d = tmp_path / "e_resumed"
    split, total = 75, 120
    sim1 = Simulation(_cfg("e_resumed", split,
                           **{"observer.checkpoint_every": split}), out_dir=d)
    for step in range(split):
        scheduler.run_step(sim1, step)
    checkpoint.save(sim1, split, d / "checkpoint" / f"ckpt-{split:06d}.pkl.gz")
    sim1.logger.flush_segment()
    sim2 = Simulation(_cfg("e_resumed", total,
                           **{"observer.checkpoint_every": split}), out_dir=d)
    sim2.run(resume_from=d)
    for stem in ("l1_events", "l2_metrics", "l3_snapshots"):
        sa = pq.read_table(straight_dir / f"{stem}.parquet").to_pylist()
        sb = pq.read_table(d / f"{stem}.parquet").to_pylist()
        assert sa == sb, f"{stem} 不一致(環境FB resume)"
    # 直接検証: _envfb が round-trip で復元される(空回り防止)
    sim3 = Simulation(_cfg("e_inspect", split,
                           **{"observer.checkpoint_every": split}),
                      out_dir=tmp_path / "e_inspect")
    checkpoint.load(sim3, d / "checkpoint" / f"ckpt-{split:06d}.pkl.gz")
    assert sim3._envfb == sim1._envfb is not None
    assert sim3._envfb["n_transit"] + sim3._envfb["n_gate"] + sim3._envfb["n_poi"] > 0


# =========================================================================== #
# (7) 環が閉じたことの観測(第80 チャンネルとの接続)
# =========================================================================== #
def test_channel_transit_delay_becomes_non_constant(tmp_path):
    """★第80 の ext.transit_delay が **ON ランで初めて**非ゼロの分散を持つ。

    OFF ランでは災害機構が無効なので常に 0(定数チャンネル=σ=0)。ON では
    『エージェントの行動が作った遅延』で 0/1 が切り替わる = 環が閉じたことの観測側の証拠。
    """
    ch = {"cognition.channels.enabled": "true"}
    off = _sim(tmp_path, "ch_off", n=20, steps=144, **ch)
    off.run()
    col_off = pq.read_table(tmp_path / "ch_off" / "channels.parquet") \
        .column("ext_transit_delay").to_pylist()
    assert set(col_off) == {0.0}, "OFF で ext.transit_delay が定数 0 でない(前提の崩れ)"

    on = _sim(tmp_path, "ch_on", n=20, steps=144,
              **{**_ON, **ch, "env.feedback.transit.platform_threshold": "0",
                 "env.feedback.transit.dwell_sec_per_pax": "60.0",
                 "env.feedback.transit.flag_min": "0.5"})
    _station_crowd(on)                      # 合成の初期高負荷(散った後は回復運転で 0 へ戻る)
    on.run()
    col_on = pq.read_table(tmp_path / "ch_on" / "channels.parquet") \
        .column("ext_transit_delay").to_pylist()
    assert set(col_on) == {0.0, 1.0}, \
        f"ON でも ext.transit_delay が定数のまま: {sorted(set(col_on))}"


def test_causal_chain_is_traceable_from_l1_alone(tmp_path):
    """★環の閉じ: L1 だけで 行動 → 密度超過 → 環境イベント → 個体の待ち が辿れる(§4.3)。

    合成するのは**初期配置だけ**で、以降はエンジン本体(scheduler.run_step)が回す:
      駅に人が居る(行動) → ホーム密度が閾値超過 → env_feedback(transit_dwell) →
      遅延 → 駅へ来た個体が改札を通れず待つ(env_feedback effect=exit_wait)。
    """
    sim = _sim(tmp_path, "chain", n=24, steps=8,
               **{**_ON, "env.feedback.transit.platform_threshold": "0",
                  "env.feedback.transit.dwell_sec_per_pax": "60.0",
                  "env.feedback.gate.enabled": "false",
                  "env.feedback.poi.enabled": "false"})
    station = sim.city.station_node
    crowd, traveller = sim.agents[:-1], sim.agents[-1]
    for a in crowd:                          # 駅に滞留する群れ(=ホーム密度の源)
        a.loc = "street"
        a.node = station
        a.sleeping = False
        a.route = []
        a.exit_intent = False
        a.stay_until = 999
    nb = next(iter(sim.city.graph.neighbors(station)))   # 隣ノードから駅へ歩いてくる個体
    traveller.loc = "street"
    traveller.sleeping = False
    traveller.node = nb
    traveller.x, traveller.y = sim.city.node_xy(nb)
    traveller.route = []                     # step 0 はまだ発たない(遅延が先に立つ)
    traveller.exit_intent = False
    traveller.stay_until = 999
    scheduler.run_step(sim, 0)               # ← ここで密度超過 → 遅延が立つ
    assert envfb.delay_min(sim) > 0.0, "テスト前提が崩れた(遅延が立っていない)"
    traveller.node = nb
    traveller.x, traveller.y = sim.city.node_xy(nb)
    traveller.route = [station]
    traveller.edge_offset = 0.0
    traveller.trip_mode = "walk"
    traveller.exit_intent = True
    traveller.stay_until = 0
    for step in range(1, 8):
        scheduler.run_step(sim, step)

    dwell = _rules(sim, envfb.RULE_TRANSIT)
    assert dwell, "密度超過の環境イベントが無い"
    # (a) 集約量・閾値・超過分が全部載っている(§4.3「どの物理量をどれだけ動かしたか」)
    p = dwell[0].payload
    assert {"metric", "value", "threshold", "excess", "delay_min", "standing",
            "exchange", "injected_min", "recovery"} <= set(p)
    assert p["value"] > p["threshold"] and p["excess"] == p["value"] - p["threshold"]
    assert p["node"] == station and p["standing"] >= len(crowd) - 2
    # (b) その遅延を受けて個体が待たされた記録が**後の step に**ある
    waits = [e for e in sim.logger.events
             if e.kind == envfb.EVENT_KIND and e.payload.get("effect") == "exit_wait"]
    assert waits, "遅延が個体の行動に効いていない(環が閉じていない)"
    first_dwell = min(e.step for e in dwell if e.payload["delay_min"] > 0)
    assert min(e.step for e in waits) > first_dwell, "待ちが遅延より前に起きている(因果が逆)"
    assert all(w.payload["delay_min"] > 0 or w.payload["gate_until"] > w.step
               for w in waits), "待ちの理由(遅延 or 規制)が payload から辿れない"
    assert waits[0].agent_id == traveller.id
    # (c) 新しいイベント種は 1 つも出ていない(登録済みの語彙だけで世界が記述されている)
    kinds = {e.kind for e in sim.logger.events}
    assert not (kinds - _known_kinds()), f"未登録・新規のイベント種が出ている: {kinds}"


def _known_kinds() -> set:
    from society.observer.schema import EVENT_KINDS
    return set(EVENT_KINDS)


def test_no_new_prompt_vocabulary_static():
    """no-fingerprint(静的): 環境FB は**プロンプト構築層に一切現れない**。

    遅延・規制・待ちは「動けない / 候補に無い / 混雑の不快感(既存語彙)」という世界の事実
    としてしか現れない。プロンプトを組む層が本 module を読んでいないことを構造で固定する。
    """
    import inspect
    from pathlib import Path
    from society.cognition import deliberate, planning, reflection, routine
    from society import envfeedback as mod
    for m in (deliberate, planning, reflection, routine):
        src = Path(inspect.getfile(m)).read_text(encoding="utf-8")
        assert "envfeedback" not in src, \
            f"{m.__name__} が環境FB を参照している(プロンプトへ漏れる経路)"
    src = Path(inspect.getfile(mod)).read_text(encoding="utf-8")
    for banned in ("remember(", "prompt", "遅延しています", "規制中"):
        assert banned not in src.split('"""')[2], \
            f"envfeedback.py がプロンプト/記憶へ文言を書いている: {banned}"


# =========================================================================== #
# (8) 宣言(registry / timeconv)
# =========================================================================== #
def test_registry_declares_env_feedback_toggles():
    """第72 の制約: 新 conf トグルはレジストリ宣言必須(未宣言検出は test_registry_modes)。"""
    for fid in ("env.feedback.enabled", "env.feedback.transit.enabled",
                "env.feedback.gate.enabled", "env.feedback.poi.enabled"):
        f = R.BY_ID.get(fid)
        assert f is not None, f"{fid} がレジストリに宣言されていない"
        assert f.repro_tier == "strict", f"{fid}: 乱数も LLM も使わないので strict のはず"
        assert f.affects_k is False, f"{fid}: generate() の呼び出し点を足さない"
        assert f.fingerprint_risk == "possible", \
            f"{fid}: 当人が観測できる世界の事実が ON でだけ起きる=possible"


def test_timeconv_classifies_env_feedback_constants():
    """Δt 不変化(第79): 減衰項・持続長・毎分レートが分類テーブルに載っている。"""
    assert T.classify("env.feedback.transit.recovery")[0] == T.KEEP, \
        "回復運転項は毎 step の残存割合=KEEP(線形にすると Δt で意味が変わる)"
    assert T.classify("env.feedback.transit.dwell_sec_per_pax")[0] == T.RATE
    assert T.classify("env.feedback.transit.dwell_cap_min")[0] == T.RATE, \
        "1 step ぶんの注入の天井は注入量と同じスケールで動かす"
    for key in ("env.feedback.transit.max_hold_steps", "env.feedback.gate.hold_steps",
                "env.feedback.gate.cooldown_steps", "env.feedback.poi.hold_steps"):
        assert T.classify(key)[0] == T.STEPS, f"{key} は [step] なので STEPS"
    for key in ("env.feedback.gate.capacity_per_min", "env.feedback.poi.capacity",
                "env.feedback.transit.platform_threshold",
                "env.feedback.transit.delay_cap_min"):
        assert T.classify(key)[0] == T.INVARIANT, f"{key} は Δt 非依存"


def test_build_cfg_defaults_are_off_and_typed():
    """conf 正準化: 既定 OFF・型強制・下限クランプ。"""
    cfg = envfb.build_cfg(None)
    assert cfg["enabled"] is False
    assert cfg["transit"]["enabled"] is True and cfg["gate"]["enabled"] is True
    assert 0.0 < cfg["transit"]["recovery"] < 1.0, "回復運転項は 0<γ<1 でなければ発散する"
    assert envfb.build_cfg({"log_every_steps": 0})["log_every_steps"] == 1
    assert envfb.build_cfg({"gate": {"hold_steps": 0}})["gate"]["hold_steps"] == 1
    assert envfb.build_cfg({"transit": {"max_hold_steps": -3}})["transit"]["max_hold_steps"] == 0
