"""好奇心/退屈の内発ドライブ = P2 スライス S5 のテスト。

設計: docs/plans/p2-interstitial-design.md §1 S5 / docs/research/interstitial-life.md §4.4-a。
仕組み: 「同じ場所への長居・単調な入力の継続」で退屈ゲージ(cognition/drive.py の boredom_*、
silence ゲージと同じ作法)が溜まり、閾値超えで LLM なしの内発的探索(近傍の未訪問/低頻度 POI へ
移動)を発火する(cognition/routine.py の _maybe_boredom_explore)。行き先は S4 Gumbel 機構が
あれば再利用、なければ決定論近傍選択。新 stream は "boredom" 1本のみ。

方針(既存 R1 の鉄則を継承):
- OFF(既定): 純粋既定と L1 バイト一致(golden_baseline_l1.json は tests/test_scenario.py が別途担保)。
  boredom_explore 0 件・agent に _boredom 等の実行時フィールドを一切生やさない・stream "boredom" を
  一度も引かない。
- ON(mock ≤24step): (a)長居する個体で boredom_explore が出る。(b)同 seed 2 回で L1 完全一致(決定論)。
  (c)予算飽和下で ON と OFF の LLM 呼数が一致(=追加 LLM 呼ゼロ)+ 呼数が k 非依存。
  (d)ゲージの蓄積→発火→リセット/cooldown の単体テスト。(e)探索先が近傍かつ現在地と異なる。
検証は mock のみ(実LLM 禁止・≤24step。OFF バイト一致確認のみ 144step)。
"""
from __future__ import annotations

import json

from society.cognition import drive, routine
from society.config import load_config
from society.engine.simulation import Simulation
from society.observer.schema import EVENT_KINDS

# 発火を確実に観測するための強めのパラメータ(長居 1〜2step で閾値超え)。
_ON = {"drive.boredom.enabled": "true",
       "drive.boredom.accrual": "0.6", "drive.boredom.threshold": "0.2",
       "drive.boredom.fire_prob": "1.0", "drive.boredom.cooldown_steps": "1",
       "drive.boredom.radius_m": "1000"}
_OFF = {"drive.boredom.enabled": "false"}


def _sim(tmp_path, name, n=40, steps=24, seed=42, **ov):
    dot = [f"run.seed={seed}", f"run.n_agents={n}", f"run.n_steps={steps}",
           f"run.name={name}"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _kind(sim, kind):
    return [e for e in sim.logger.events if e.kind == kind]


# ------------------------------------------------------------ schema 登録
def test_boredom_kind_registered():
    assert "boredom_explore" in EVENT_KINDS


# ------------------------------------------------------------ OFF: 純粋既定と一致 + 無副作用
def test_off_matches_pure_default(tmp_path):
    """明示 OFF と純粋既定が L1 完全一致(24step)。OFF では実行時フィールドも作られない。"""
    pure = _sim(tmp_path, "pure")
    pure.run()
    off = _sim(tmp_path, "bore_off", **_OFF)
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(S5 seam が no-op でない)"
    assert not any(e.kind == "boredom_explore" for e in off.logger.events)
    for a in pure.agents:
        assert not hasattr(a, "_boredom"), "OFF なのに _boredom が生えている"
        assert not hasattr(a, "_bore_node"), "OFF なのに _bore_node が生えている"


def test_off_matches_pure_default_long(tmp_path):
    """ゴールデンと同じ 144step 尺でも OFF は純粋既定とバイト一致(no-op の確度を上げる)。"""
    pure = _sim(tmp_path, "purel", n=12, steps=144)
    pure.run()
    off = _sim(tmp_path, "offl", n=12, steps=144, **_OFF)
    off.run()
    assert _l1(pure) == _l1(off)


# ------------------------------------------------------------ (a) 長居する個体で探索が出る
def test_on_emits_boredom_explore(tmp_path):
    """ON の mock 24step で boredom_explore が出る(payload の型も固定)。"""
    sim = _sim(tmp_path, "emit", n=200, **_ON)
    sim.run()
    evs = _kind(sim, "boredom_explore")
    assert evs, "ON なのに boredom_explore が1件も出ていない(長居→探索が発火せず)"
    assert set(evs[0].payload) == {"from", "to_kind", "gauge"}
    assert all(e.payload["gauge"] >= 0.0 for e in evs)


# ------------------------------------------------------------ (b) 決定論(同 seed 2回で一致)
def test_on_deterministic(tmp_path):
    a = _sim(tmp_path, "det_a", n=80, **_ON)
    a.run()
    b = _sim(tmp_path, "det_b", n=80, **_ON)
    b.run()
    assert _l1(a) == _l1(b), "ON の決定論が崩れている(専用 stream 以外の乱数漏れ?)"


def test_on_differs_from_off(tmp_path):
    """ON は OFF と L1 が異なる(=内発的探索が実際に注入され行動が非決定化している)。"""
    on = _sim(tmp_path, "diff_on", n=80, **_ON)
    on.run()
    off = _sim(tmp_path, "diff_off", n=80, **_OFF)
    off.run()
    assert _l1(on) != _l1(off)


# ------------------------------------------------------------ (c) 追加 LLM 呼ゼロ(予算飽和下)
def test_call_count_invariant_under_saturated_budget(tmp_path):
    """予算飽和下では ON と OFF の LLM 呼数が一致(S5 は generate() を1本も足さない=R1)。

    S5 は行き先を非決定化して co-presence を再配分し novel_place で drive を押し上げうるが、
    予算を飽和させれば 総発火数 = Σ min(申請者数, 予算) となり「誰が発火するか」が変わっても
    「何本発火するか」は変わらない(test_stochastic_exec と同型の検査)。"""
    ov = {"lod.max_llm_per_step": 2}
    on = _sim(tmp_path, "cc_on", n=100, **{**_ON, **ov})
    on.run()
    off = _sim(tmp_path, "cc_off", n=100, **{**_OFF, **ov})
    off.run()
    assert len(on.logger.llm_calls) == len(off.logger.llm_calls) > 0, \
        f"予算飽和下で呼数が不一致: on={len(on.logger.llm_calls)} off={len(off.logger.llm_calls)}"


def test_call_count_k_invariant(tmp_path):
    """真の R1「呼数 k 非依存」: ON のまま k(writeback)を free/off に振っても呼数が不変
    (S5 は物理量[現在地・移動中か・訪問回数]のみを入力にし k 由来量を一切見ない)。"""
    ov = {"lod.max_llm_per_step": 2}
    free = _sim(tmp_path, "k_free", n=100, **{**_ON, **ov, "k.writeback": "free"})
    free.run()
    off = _sim(tmp_path, "k_off", n=100, **{**_ON, **ov, "k.writeback": "off"})
    off.run()
    assert len(free.logger.llm_calls) == len(off.logger.llm_calls), \
        f"ON の呼数が k で乖離(R1 違反): free={len(free.logger.llm_calls)} off={len(off.logger.llm_calls)}"


# ------------------------------------------------------------ (d) ゲージの蓄積→発火→リセット/cooldown
def test_gauge_accrue_fire_reset_cooldown():
    """退屈ゲージの単体機構(drive.boredom_tick/ready/fire)。同じ場所への長居で蓄積、
    移動・新奇で減衰、閾値超え+cooldown 明けで発火候補、発火でリセット+cooldown。"""
    class A:
        pass

    a = A()
    a.node = "n1"
    a.route = None
    cfg = drive.build_cfg({"boredom": {
        "enabled": True, "accrual": 0.3, "decay": 0.1, "novelty_relief": 0.2,
        "threshold": 0.6, "cooldown_steps": 5}})

    drive.boredom_tick(a, cfg, 0)                # 初回=観測開始(蓄積しない)
    assert a._boredom == 0.0
    assert not drive.boredom_ready(a, cfg, 0)

    for s in range(1, 4):                        # 同じ場所への長居 → 蓄積
        drive.boredom_tick(a, cfg, s)
    assert abs(a._boredom - 0.9) < 1e-9          # 0.3 × 3
    assert drive.boredom_ready(a, cfg, 4)        # 0.9 >= 0.6

    drive.boredom_fire(a, cfg, 4)                # 発火 → リセット + cooldown
    assert a._boredom == 0.0
    assert a._bore_cooldown == 9
    assert not drive.boredom_ready(a, cfg, 5)    # ゲージ0 かつ cooldown 中

    for s in range(5, 9):                        # 再蓄積するが cooldown 中は ready にしない
        drive.boredom_tick(a, cfg, s)
    assert a._boredom >= 0.6
    assert not drive.boredom_ready(a, cfg, 8), "cooldown 中に発火候補になった"
    assert drive.boredom_ready(a, cfg, 9), "cooldown 明けで発火候補にならない"

    before = a._boredom                          # 場所移動(新奇)→ 減衰
    a.node = "n2"
    drive.boredom_tick(a, cfg, 9)
    assert a._boredom < before, "新しい場所に着いても減衰しない"


def test_gauge_disabled_is_noop():
    """OFF(enabled=False)は agent に一切触れない(byte-match 保証の単体版)。"""
    class A:
        pass

    a = A()
    a.node = "n1"
    a.route = None
    cfg = drive.build_cfg({})                     # boredom 未指定=既定 OFF
    assert not cfg["boredom"]["enabled"]
    drive.boredom_tick(a, cfg, 0)
    drive.boredom_tick(a, cfg, 1)
    assert not hasattr(a, "_boredom")
    assert not hasattr(a, "_bore_node")
    assert drive.boredom_ready(a, cfg, 2) is False


# ------------------------------------------------------------ (e) 探索先が近傍かつ現在地と異なる
def test_explore_destination_is_nearby_and_different(tmp_path):
    """_boredom_destination は近傍(radius_m)集合の中から現在地と異なるノードを返す。"""
    sim = _sim(tmp_path, "dest", n=40, steps=1, **{**_ON, "drive.boredom.radius_m": "500"})
    bc = sim.drivecfg["boredom"]
    scfg = routine._stochastic_cfg(sim)          # S4 OFF → None(=決定論近傍選択)
    assert scfg is None
    for a in sim.agents:
        nearby = set(routine._nearby_pois(a, sim, a.node, bc["radius_m"]))
        if not nearby:
            continue
        rng = sim.hub.stream("boredom", a.id, 0)
        dest = routine._boredom_destination(a, sim, scfg, bc, rng)
        assert dest is not None
        assert dest != a.node, "探索先が現在地と同じ"
        assert dest in nearby, "探索先が近傍集合の外にある"
        return
    assert False, "近傍 POI を持つ個体が見つからない(radius/map を確認)"


def test_explore_prefers_unvisited(tmp_path):
    """近傍に未訪問 POI があれば最優先で選ぶ(脱馴化の入口=novel_place 接続の下地)。"""
    sim = _sim(tmp_path, "pref", n=40, steps=1, **{**_ON, "drive.boredom.radius_m": "800"})
    bc = sim.drivecfg["boredom"]
    for a in sim.agents:
        nearby = list(dict.fromkeys(routine._nearby_pois(a, sim, a.node, bc["radius_m"])))
        if len(nearby) < 2:
            continue
        for n in nearby:                          # 近傍を全訪問済みにし
            a.visits[n] = 5
        target = nearby[-1]                       # 1つだけ未訪問へ戻す
        a.visits[target] = 0
        rng = sim.hub.stream("boredom", a.id, 0)
        dest = routine._boredom_destination(a, sim, None, bc, rng)
        assert dest == target, "近傍の未訪問 POI を最優先していない"
        return
    assert False, "候補2つ以上の個体が見つからない"


# ------------------------------------------------------------ Gumbel(S4)機構の再利用
def test_explore_reuses_gumbel_when_s4_on(tmp_path):
    """S4(routine.stochastic + gumbel)が ON なら退屈探索は Gumbel 機構を再利用して確率選択する
    (行き先が複数出る)。新 stream は "boredom" のみ(gumbel 乱数もこの stream から引く)。"""
    sim = _sim(tmp_path, "gum", n=40, steps=1,
               **{**_ON, "drive.boredom.radius_m": "1500",
                  "routine.stochastic.enabled": "true",
                  "routine.stochastic.gumbel.enabled": "true"})
    bc = sim.drivecfg["boredom"]
    scfg = routine._stochastic_cfg(sim)
    assert scfg is not None and scfg["gumbel_enabled"]
    for a in sim.agents:
        nearby = list(dict.fromkeys(routine._nearby_pois(a, sim, a.node, bc["radius_m"])))
        if len(nearby) < 4:
            continue
        got = {routine._boredom_destination(
                   a, sim, scfg, bc, sim.hub.stream("boredom", a.id, s))
               for s in range(60)}
        got.discard(None)
        assert all(d in set(nearby) and d != a.node for d in got)
        assert len(got) >= 2, f"Gumbel 再利用で行き先が多様化しない: {got}"
        return
    assert False, "候補4つ以上の個体が見つからない"
