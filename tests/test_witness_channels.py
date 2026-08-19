"""WIT-1: 目撃走査の作り直し + チャネル×注意ゲートの検収(2026-08-20)。

なぜこのテストが要るか
----------------------
`src/society/truth_ledger.py` は `observer/metrics_spec.py` の凍結 `SPEC_FILES` 14 本の
1 つで、通常は 1 バイトも触らない。WIT-1 は **2026-08-20 にユーザーが明示承認した凍結解除**
であり(正典 docs/plans/witness-channel-attention-plan.md §5)、W3-1(2026-08-07・
tests/test_w3_frozen_analyzers.py 冒頭)と同じプロトコルを踏む:

  (1) **逐語温存との同値**: 変更前の `_witness_pass` / `_set_belief` を本ファイル内に
      `_legacy_*` として逐語で持ち、合成配置に対する **L1 イベント列と信念状態が完全一致**
      することを機械証明する(判定式に 1 ミリも触っていないという主張の唯一の証拠)。
  (2) **OFF = 現行挙動**: `beliefs.channels.enabled` 既定 false で実ランの L1 が一致。
  (3) **ON = 決定論**: 同 seed 2 ラン一致(乱数 stream を 1 本も足していない)。

`SPEC_FILES` のリスト(14 本)から `truth_ledger.py` を外す改竄は禁止で、ここでも固定する。
実 LLM は使わない(mock / 合成配置だけで完結する)。
"""
from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import numpy as np
import pytest

from society import attention as ATT
from society import truth_ledger as TL
from society.config import load_config
from society.engine.simulation import Simulation
from society.observer import metrics_spec as MS
from society.observer.schema import Event

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src" / "society" / "truth_ledger.py"


# =========================================================================== #
# 合成ハーネス(実 city / 実 logger を使わずに走査だけを取り出す)
# =========================================================================== #
class _Agent:
    __slots__ = ("id", "x", "y", "sleeping", "loc", "_fact_beliefs")

    def __init__(self, aid, x, y, sleeping, loc):
        self.id = aid
        self.x = x
        self.y = y
        self.sleeping = sleeping
        self.loc = loc
        self._fact_beliefs = None


class _Logger:
    """`Event.payload` を書き足す実 logger(setdefault)の性質だけ真似た記録係。"""

    def __init__(self):
        self.events: list[Event] = []

    def log(self, event: Event) -> None:
        event.payload.setdefault("_seq", len(self.events))
        self.events.append(event)


class _Sim:
    def __init__(self, agents):
        self.agents = agents
        self.agent_by_id = {int(a.id): a for a in agents}
        self.logger = _Logger()
        self._tl_facts = {}
        self._tl_stats = {"facts": 0, "witness": 0, "transmit": 0,
                          "verify_ok": 0, "verify_try": 0}


def _make_agents(rng, n, box, *, shuffle_ids=False):
    xs = rng.random(n) * box
    ys = rng.random(n) * box
    sleep = rng.random(n) < 0.18
    outside = rng.random(n) < 0.22
    ids = list(range(n))
    if shuffle_ids:                       # id 順 != sim.agents の並び でも壊れないこと
        ids = [int(v) for v in rng.permutation(n)]
    return [_Agent(ids[i], float(xs[i]), float(ys[i]), bool(sleep[i]),
                   "outside" if outside[i] else "street") for i in range(n)]


def _make_facts(rng, n_facts, box, radii, *, kinds=("event_host",), window=6,
                n_agents=0, actor_rate=0.0):
    facts = {}
    for i in range(n_facts):
        actor = -1
        if actor_rate and rng.random() < actor_rate and n_agents:
            actor = int(rng.integers(0, n_agents))
        fid = f"f{i:05d}"
        facts[fid] = {
            "id": fid, "kind": str(rng.choice(kinds)), "step": 0, "sim_min": 0,
            "node": None, "place": None,
            "x": float(rng.random() * box), "y": float(rng.random() * box),
            "value": 1.0, "actor": actor, "topics": [],
            "radius_m": float(rng.choice(radii)), "window": window,
            "canary": f"c{i}"}
    return facts


# --- 変更前の実装の逐語温存(ここが「触っていない」の証拠) ------------------- #
def _legacy_set_belief(sim, cfg, agent, fact, *, value, conf, src, parent, hop,
                       cause, verified, step, sim_min):
    bels = TL.beliefs_of(agent)
    rec = {"value": round(float(value), 6), "conf": round(float(conf), 6),
           "src": src, "from": int(parent), "step": int(step), "hop": int(hop),
           "verified": verified, "kind": fact["kind"]}
    bels[fact["id"]] = rec
    cap = int(cfg["max_beliefs_per_agent"])
    if cap > 0 and len(bels) > cap:
        drop = sorted(bels, key=lambda k: (bels[k]["step"], k))[:len(bels) - cap]
        for k in drop:
            del bels[k]
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=int(agent.id),
                         kind="belief_update", x=agent.x, y=agent.y,
                         payload={"fact": fact["id"], "fact_kind": fact["kind"],
                                  "value": rec["value"], "conf": rec["conf"],
                                  "src": src, "from": int(parent),
                                  "hop": int(hop), "cause": cause,
                                  "verified": verified}))
    return rec


def _legacy_witness_pass(sim, cfg, step, sim_min):
    facts = TL.facts_of(sim)
    if not facts:
        return
    stats = TL._stats(sim)
    open_ids = [fid for fid, f in facts.items()
                if f["step"] <= step < f["step"] + max(1, int(f["window"]))]
    if not open_ids:
        return
    for fid in open_ids:
        fact = facts[fid]
        radius = float(fact["radius_m"])
        fx, fy = float(fact["x"]), float(fact["y"])
        for agent in sim.agents:
            if fid in (getattr(agent, "_fact_beliefs", None) or ()):
                continue
            is_actor = (fact["actor"] >= 0 and int(agent.id) == fact["actor"])
            if not is_actor:
                if not TL._present(agent):
                    continue
                if radius > 0.0 and math.hypot(agent.x - fx, agent.y - fy) > radius:
                    continue
            _legacy_set_belief(sim, cfg, agent, fact, value=fact["value"],
                               conf=1.0, src="direct", parent=-1, hop=0,
                               cause="witness", verified="witness",
                               step=step, sim_min=sim_min)
            stats["witness"] += 1


def _l1(sim):
    return [(e.step, e.sim_min, e.agent_id, e.kind, e.x, e.y,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True))
            for e in sim.logger.events]


def _state(sim):
    return {int(a.id): dict(getattr(a, "_fact_beliefs", None) or {})
            for a in sim.agents}


def _both(agents_spec, facts, cfg, steps=(0,)):
    """同じ入力に対して 旧 / 新 を走らせ、(L1, 信念状態) を返す。"""
    out = []
    for pas in (_legacy_witness_pass, TL._witness_pass):
        agents = [_Agent(a.id, a.x, a.y, a.sleeping, a.loc) for a in agents_spec]
        sim = _Sim(agents)
        sim._tl_facts = {k: dict(v) for k, v in facts.items()}
        for st in steps:
            pas(sim, cfg, st, st * 10)
        out.append((_l1(sim), _state(sim), dict(sim._tl_stats)))
    return out


def _cfg_plain(**over):
    cfg = TL.build_cfg(None)
    cfg.update(over)
    return cfg


# =========================================================================== #
# 1) 同値スイート: 新走査 == 逐語温存(半径 × セル幅 × 配置)
# =========================================================================== #
@pytest.mark.parametrize("cell", [60.0, 120.0, 250.0])
def test_new_scan_is_identical_to_the_legacy_predicate(cell, monkeypatch):
    """★ランダム配置 × 半径 {5,60,300,1000} で **L1 も信念状態も完全一致**。"""
    monkeypatch.setattr(TL, "_CELL_M", cell)
    rng = np.random.default_rng(20260820)
    cfg = _cfg_plain()
    n_facts_total = 0
    for trial in range(24):
        n = int(rng.integers(200, 1200))
        box = float(rng.choice([200.0, 900.0, 3000.0]))
        agents = _make_agents(rng, n, box, shuffle_ids=bool(trial % 4 == 3))
        facts = _make_facts(rng, int(rng.integers(20, 60)), box,
                            [5.0, 60.0, 300.0, 1000.0],
                            n_agents=n, actor_rate=0.5)
        (l1a, sa, sta), (l1b, sb, stb) = _both(agents, facts, cfg)
        assert l1a == l1b, f"L1 が食い違った(cell={cell}, trial={trial})"
        assert sa == sb and sta == stb
        n_facts_total += len(facts)
    assert n_facts_total >= 800
    assert any(sa.values()), "1 件も目撃が起きていない(前提が崩れた)"


def test_equivalence_over_many_facts_and_windows(monkeypatch):
    """★数千 fact 試行 + 複数 step(窓)でも一致(既知スキップ・cap 落ちを含む)。"""
    monkeypatch.setattr(TL, "_CELL_M", 120.0)
    rng = np.random.default_rng(7)
    cfg = _cfg_plain(max_beliefs_per_agent=8)      # cap を小さくして退避経路も踏む
    total = 0
    for _ in range(40):
        n = int(rng.integers(150, 600))
        box = float(rng.choice([150.0, 600.0, 2500.0]))
        agents = _make_agents(rng, n, box)
        facts = _make_facts(rng, 60, box, [0.0, 5.0, 60.0, 300.0, 1000.0],
                            n_agents=n, actor_rate=0.3)
        (l1a, sa, sta), (l1b, sb, stb) = _both(agents, facts, cfg,
                                               steps=(0, 1, 2))
        assert l1a == l1b and sa == sb and sta == stb
        total += len(facts) * 3
    assert total >= 5000, f"試行数が足りない: {total}"


def test_city_wide_facts_skip_distance_entirely(monkeypatch):
    """radius<=0(市域全体)は距離計算をしないが、通過集合は現行と一致する。"""
    monkeypatch.setattr(TL, "_CELL_M", 120.0)
    rng = np.random.default_rng(99)
    agents = _make_agents(rng, 400, 1000.0)
    facts = _make_facts(rng, 6, 1000.0, [0.0, -1.0], n_agents=400, actor_rate=1.0)
    (l1a, sa, _x), (l1b, sb, _y) = _both(agents, facts, _cfg_plain())
    assert l1a == l1b and sa == sb
    present = sum(1 for a in agents if TL._present(a))
    assert len({e[2] for e in l1a}) >= present, "在場者全員に届いていない"


def test_no_present_agents_still_reaches_the_actor(monkeypatch):
    """全員が不在(就寝/圏外)でも当事者だけは通る(present/半径のバイパス)。"""
    monkeypatch.setattr(TL, "_CELL_M", 120.0)
    agents = [_Agent(i, 10.0 * i, 0.0, True, "outside") for i in range(5)]
    facts = {"f00000": {"id": "f00000", "kind": "crime", "step": 0, "sim_min": 0,
                        "node": None, "place": None, "x": 9000.0, "y": 9000.0,
                        "value": 1.0, "actor": 3, "topics": [],
                        "radius_m": 60.0, "window": 6, "canary": "c"}}
    (l1a, sa, _x), (l1b, sb, _y) = _both(agents, facts, _cfg_plain())
    assert l1a == l1b and sa == sb
    assert [e[2] for e in l1b] == [3]


# =========================================================================== #
# 2) 境界帯: 半径ちょうど ±1ulp でフォールバック(厳密再判定)が効く
# =========================================================================== #
def _nudge(v: float, n: int) -> float:
    for _ in range(abs(n)):
        v = math.nextafter(v, math.inf if n > 0 else -math.inf)
    return v


def test_boundary_band_falls_back_to_the_exact_predicate():
    """半径境界の ±1ulp 近傍で、二乗距離では決まらず現行式で決まることを固定する。"""
    radius = 60.0
    t = radius * radius
    xs, ys, want = [], [], []
    n_edge = 0
    for k in range(-3, 4):
        for ang in (0.0, 0.37, 1.1, 2.9, 4.4):
            d = _nudge(radius, k)
            x = d * math.cos(ang)
            y = d * math.sin(ang)
            xs.append(x)
            ys.append(y)
            want.append(not (math.hypot(x, y) > radius))
            s = x * x + y * y
            if not (s <= t * (1.0 - TL._BAND)) and not (s >= t * (1.0 + TL._BAND)):
                n_edge += 1
    ax = np.array(xs, dtype=np.float64)
    ay = np.array(ys, dtype=np.float64)
    cand = np.arange(len(xs), dtype=np.int64)
    got = TL._inside(ax, ay, cand, 0.0, 0.0, radius).tolist()
    assert got == [i for i, w in enumerate(want) if w]
    assert n_edge > 0, "境界帯に 1 点も落ちていない(フォールバックが試されていない)"


def test_band_is_wide_enough_on_adversarial_points():
    """境界に貼り付けた点でも、帯の外で下した判定が現行式と食い違わない。"""
    rng = np.random.default_rng(4242)
    radius = 60.0
    t = radius * radius
    th = rng.random(200_000) * (2.0 * math.pi)
    f = 1.0 + (rng.random(200_000) * 2.0 - 1.0) * 1e-15
    dx = np.cos(th) * radius * f
    dy = np.sin(th) * radius * f
    s = dx * dx + dy * dy
    hard_in = s <= t * (1.0 - TL._BAND)
    hard_out = s >= t * (1.0 + TL._BAND)
    exact = np.fromiter((not (math.hypot(a, b) > radius)
                         for a, b in zip(dx.tolist(), dy.tolist())),
                        dtype=bool, count=len(dx))
    assert not np.any(hard_in & ~exact), "帯の内で『内側』と決めたが現行式は外側"
    assert not np.any(hard_out & exact), "帯の外で『外側』と決めたが現行式は内側"


# =========================================================================== #
# 3) 順序: 出力は常に昇順・L1 の並びが不変
# =========================================================================== #
def test_candidate_indices_are_always_ascending():
    rng = np.random.default_rng(11)
    for _ in range(30):
        n = int(rng.integers(50, 800))
        box = float(rng.choice([300.0, 1500.0]))
        xs = rng.random(n) * box
        ys = rng.random(n) * box
        idx = np.arange(n, dtype=np.int64)
        grid = TL._build_grid(xs, ys, idx, 120.0)
        for _f in range(20):
            fx, fy = float(rng.random() * box), float(rng.random() * box)
            r = float(rng.choice([5.0, 60.0, 300.0]))
            near = TL._grid_near(grid, fx, fy, r)
            if near is None:
                continue
            arr = np.asarray(near)
            assert np.array_equal(arr, np.sort(arr)), "近傍セル候補が昇順でない"
            assert len(set(arr.tolist())) == len(arr), "候補に重複がある"
            got = TL._inside(xs, ys, near, fx, fy, r)
            assert np.array_equal(got, np.sort(got))


def test_l1_order_is_fact_order_then_agent_order(monkeypatch):
    """L1 は「fact 生成順 × sim.agents の並び」= 現行と同じ並びで出る。"""
    monkeypatch.setattr(TL, "_CELL_M", 120.0)
    rng = np.random.default_rng(5)
    agents = _make_agents(rng, 500, 800.0)
    facts = _make_facts(rng, 12, 800.0, [60.0, 300.0], n_agents=500,
                        actor_rate=0.5)
    (l1a, _sa, _x), (l1b, _sb, _y) = _both(agents, facts, _cfg_plain())
    assert l1a == l1b
    order = {int(a.id): i for i, a in enumerate(agents)}
    seen_fact: list[str] = []
    per_fact: dict[str, list[int]] = {}
    for _st, _sm, aid, _k, _x2, _y2, payload in l1b:
        fid = json.loads(payload)["fact"]
        if fid not in per_fact:
            per_fact[fid] = []
            seen_fact.append(fid)
        per_fact[fid].append(order[int(aid)])
    assert seen_fact == sorted(seen_fact), "fact の並びが生成順から崩れた"
    for fid, seq in per_fact.items():
        assert seq == sorted(seq), f"{fid} の受信者が sim.agents の並びでない"


def test_payload_and_rec_are_not_shared_between_agents(monkeypatch):
    """雛形の複製: 受信者ごとに別 dict(logger の書き足しが隣へ漏れない)。"""
    monkeypatch.setattr(TL, "_CELL_M", 120.0)
    agents = [_Agent(i, 0.0, 0.0, False, "street") for i in range(5)]
    sim = _Sim(agents)
    sim._tl_facts = _make_facts(np.random.default_rng(1), 1, 1.0, [1000.0])
    sim._tl_facts["f00000"]["x"] = 0.0
    sim._tl_facts["f00000"]["y"] = 0.0
    TL._witness_pass(sim, _cfg_plain(), 0, 0)
    evs = sim.logger.events
    assert len(evs) == 5
    assert len({id(e.payload) for e in evs}) == 5, "payload を共有している"
    assert [e.payload["_seq"] for e in evs] == [0, 1, 2, 3, 4], \
        "logger の書き足しが前の個体の値で埋まっている"
    recs = [a._fact_beliefs["f00000"] for a in agents]
    assert len({id(r) for r in recs}) == 5, "信念 rec を共有している"


# =========================================================================== #
# 4) conf: 既定 OFF = 現行挙動
# =========================================================================== #
def test_channels_default_is_off_and_inert():
    cfg = load_config()
    assert bool(cfg.beliefs.channels.enabled) is False
    built = TL.build_cfg(None)
    assert built["channels"]["enabled"] is False
    assert built["channels"]["ambient_k"] == 2
    # 出荷 conf の表と DEFAULTS の表が一致(片方だけ直したら気づける)
    shipped = TL.build_cfg(cfg.beliefs)["channels"]["kinds"]
    assert shipped == TL._CHANNEL_DEFAULTS["kinds"]
    assert "disaster" not in shipped, "放送系にゲートが付いている"


def test_channels_off_changes_nothing_in_the_scan(monkeypatch):
    """channels 省略 / false のどちらでも、通過集合が現行と 1 件も変わらない。"""
    monkeypatch.setattr(TL, "_CELL_M", 120.0)
    rng = np.random.default_rng(31337)
    agents = _make_agents(rng, 600, 1200.0)
    kinds = ("crime", "flyer_post", "price_change", "disaster")
    facts = _make_facts(rng, 40, 1200.0, [60.0], kinds=kinds, n_agents=600,
                        actor_rate=0.3)
    base = _cfg_plain()
    off = TL.build_cfg({"channels": {"enabled": False}})
    off["max_beliefs_per_agent"] = base["max_beliefs_per_agent"]
    (l1a, sa, _x), (l1b, sb, _y) = _both(agents, facts, base)
    assert l1a == l1b and sa == sb
    (_l1c, _sc, _z), (l1d, sd, _w) = _both(agents, facts, off)
    assert l1d == l1b and sd == sb, "channels: {enabled: false} で挙動が動いた"


# =========================================================================== #
# 5) ON: 半径上書き + 通過ゲート
# =========================================================================== #
_ON = {"enabled": True, "ambient_k": 2,
       "kinds": {"flyer_post": {"radius_m": 15.0, "p_notice": 1.0},
                 "crime": {"radius_m": 40.0, "p_notice_day": 1.0,
                           "p_notice_night": 0.0}}}


def test_radius_override_narrows_reach(monkeypatch):
    """ON の半径上書きが効く(通過集合が現行の部分集合になる)。"""
    monkeypatch.setattr(TL, "_CELL_M", 120.0)
    rng = np.random.default_rng(808)
    agents = _make_agents(rng, 800, 400.0)
    facts = _make_facts(rng, 20, 400.0, [60.0], kinds=("flyer_post",))
    base = _cfg_plain()
    on = TL.build_cfg({"channels": _ON})
    (l1_old, _s1, _a), (_l1_new, _s2, _b) = _both(agents, facts, base)
    (_x, _y, _c), (l1_on, _s3, _d) = _both(agents, facts, on)
    old_pairs = {(json.loads(r[6])["fact"], r[2]) for r in l1_old}
    on_pairs = {(json.loads(r[6])["fact"], r[2]) for r in l1_on}
    assert on_pairs < old_pairs, "半径 15m が 60m の真部分集合になっていない"
    assert on_pairs, "上書き後に 1 件も通っていない(狭すぎ=前提が崩れた)"


def test_day_night_split_for_scene_facts(monkeypatch):
    """昼(6:00-18:00)と夜で通過率が切り替わる。"""
    monkeypatch.setattr(TL, "_CELL_M", 120.0)
    rng = np.random.default_rng(606)
    agents = _make_agents(rng, 400, 200.0)
    facts = _make_facts(rng, 8, 200.0, [60.0], kinds=("crime",))
    on = TL.build_cfg({"channels": _ON})
    spec = on["channels"]["kinds"]["crime"]
    assert TL._p_notice(spec, 9 * 60) == 1.0
    assert TL._p_notice(spec, 21 * 60) == 0.0
    assert TL._p_notice(spec, 6 * 60) == 1.0 and TL._p_notice(spec, 18 * 60) == 0.0
    assert TL._p_notice(spec, 0) == 0.0 and TL._p_notice(spec, 3 * 1440 + 700) == 1.0

    # 実際の pass でも昼=全通過 / 夜=0 件(当事者は居ない配置)になる
    def _run(sim_min):
        agents2 = [_Agent(a.id, a.x, a.y, a.sleeping, a.loc) for a in agents]
        sim = _Sim(agents2)
        sim._tl_facts = {k: dict(v) for k, v in facts.items()}
        TL._witness_pass(sim, on, 0, sim_min)
        return len(sim.logger.events)
    assert _run(9 * 60) > 0
    assert _run(21 * 60) == 0, "夜の通過率 0.0 なのに目撃が起きた"


def test_hash_gate_converges_to_p_notice(monkeypatch):
    """★大数の通過率が p_notice に収束する(±数%)。"""
    monkeypatch.setattr(TL, "_CELL_M", 120.0)
    for p in (0.30, 0.55):
        hits = sum(1 for f in range(200) for a in range(200)
                   if TL._u01(TL._WIT_SALT, f"f{f:05d}", a) < p)
        rate = hits / (200 * 200)
        assert abs(rate - p) < 0.02, f"p={p} に対して通過率 {rate:.4f}"


def test_gate_is_drawn_once_per_fact_and_agent(monkeypatch):
    """抽選キーに step を入れない = 窓の間ずっと同じ判定(気づく機会は 1 回)。"""
    monkeypatch.setattr(TL, "_CELL_M", 120.0)
    rng = np.random.default_rng(21)
    agents = _make_agents(rng, 300, 100.0)
    for a in agents:                              # 全員が在場・全 fact の半径内
        a.sleeping = False
        a.loc = "street"
    facts = _make_facts(rng, 4, 100.0, [1000.0], kinds=("flyer_post",))
    on = TL.build_cfg({"channels": {"enabled": True, "kinds": {
        "flyer_post": {"radius_m": 1000.0, "p_notice": 0.5}}}})
    sim = _Sim([_Agent(a.id, a.x, a.y, a.sleeping, a.loc) for a in agents])
    sim._tl_facts = {k: dict(v) for k, v in facts.items()}
    counts = []
    for st in range(6):
        n0 = len(sim.logger.events)
        TL._witness_pass(sim, on, st, st * 10)
        counts.append(len(sim.logger.events) - n0)
    assert counts[0] > 0
    assert counts[1:] == [0] * 5, \
        f"窓の 2 step 目以降にも新規目撃が出た(step が鍵に混ざっている): {counts}"
    got = {int(e.agent_id) for e in sim.logger.events
           if json.loads(json.dumps(e.payload))["fact"] == "f00000"}
    want = {int(a.id) for a in agents
            if TL._u01(TL._WIT_SALT, "f00000", int(a.id)) < 0.5}
    assert got == want


def test_actor_is_never_gated(monkeypatch):
    """当事者はゲートを受けない(気づく気づかないの話ではない)。"""
    monkeypatch.setattr(TL, "_CELL_M", 120.0)
    agents = [_Agent(i, 0.0, 0.0, False, "street") for i in range(64)]
    on = TL.build_cfg({"channels": {"enabled": True, "kinds": {
        "flyer_post": {"radius_m": 15.0, "p_notice": 0.0}}}})
    facts = {"f00000": {"id": "f00000", "kind": "flyer_post", "step": 0,
                        "sim_min": 0, "node": None, "place": None,
                        "x": 0.0, "y": 0.0, "value": 1.0, "actor": 7,
                        "topics": [], "radius_m": 60.0, "window": 6,
                        "canary": "c"}}
    sim = _Sim(agents)
    sim._tl_facts = facts
    TL._witness_pass(sim, on, 0, 0)
    assert [int(e.agent_id) for e in sim.logger.events] == [7]


def test_kinds_not_listed_are_untouched(monkeypatch):
    """表に載らない種(放送系)は半径上書きもゲートも受けない。"""
    monkeypatch.setattr(TL, "_CELL_M", 120.0)
    rng = np.random.default_rng(77)
    agents = _make_agents(rng, 300, 500.0)
    facts = _make_facts(rng, 10, 500.0, [0.0], kinds=("disaster",))
    base = _cfg_plain()
    on = TL.build_cfg({"channels": {"enabled": True, "kinds": {
        "flyer_post": {"radius_m": 15.0, "p_notice": 0.0}}}})
    (l1_old, _a, _b), (_c, _d, _e) = _both(agents, facts, base)
    (_f, _g, _h), (l1_on, _i, _j) = _both(agents, facts, on)
    assert l1_on == l1_old, "表に無い種の挙動が動いた"


# =========================================================================== #
# 6) 実ラン(mock): OFF バイト一致 / ON 決定論 / ON で認知が 100% から離れる
# =========================================================================== #
def _cfg(name, n_steps, n_agents, **ov):
    dot = ["run.seed=42", f"run.n_agents={n_agents}", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def _run(tmp_path, name, n_steps=48, n_agents=30, **ov):
    sim = Simulation(_cfg(name, n_steps, n_agents, **ov), out_dir=tmp_path / name)
    sim.run()
    return sim


def _rows(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def test_off_run_is_identical_to_the_current_world(tmp_path):
    """★channels 省略 / false のどちらでも L1 が現行と一致(既定 OFF = バイト一致)。"""
    base = _run(tmp_path, "wit_base", **{"beliefs.enabled": "true"})
    off = _run(tmp_path, "wit_off", **{"beliefs.enabled": "true",
                                       "beliefs.channels.enabled": "false"})
    assert _rows(off) == _rows(base)
    assert [r for r in _rows(base) if r[2] == "belief_update"], \
        "目撃/伝聞が 1 件も出ていない(前提が崩れた)"


def test_on_is_deterministic_across_runs(tmp_path):
    ov = {"beliefs.enabled": "true", "beliefs.channels.enabled": "true"}
    a = _run(tmp_path, "wit_on_a", **ov)
    b = _run(tmp_path, "wit_on_b", **ov)
    assert _rows(a) == _rows(b), "ON が非決定(乱数を引いている疑い)"


def test_on_reduces_witness_volume(tmp_path):
    """ON は目撃を減らす(全員が認知する世界から離れる)= 効果が実在する。"""
    ov = {"beliefs.enabled": "true"}
    base = _run(tmp_path, "wit_vol_off", n_steps=144, n_agents=40, **ov)
    on = _run(tmp_path, "wit_vol_on", n_steps=144, n_agents=40,
              **{**ov, "beliefs.channels.enabled": "true"})
    n_off = sum(1 for e in base.logger.events
                if e.kind == "belief_update"
                and e.payload.get("cause") == "witness")
    n_on = sum(1 for e in on.logger.events
               if e.kind == "belief_update"
               and e.payload.get("cause") == "witness")
    assert n_off > 0 and n_on < n_off, f"ON で目撃が減っていない: {n_on} vs {n_off}"


def test_on_resume_matches_a_straight_run(tmp_path):
    """ON の checkpoint 再開が通しランと一致(ゲートは step を鍵に入れない = 再開に強い)。"""
    import pyarrow.parquet as pq
    ov = {"beliefs.enabled": "true", "beliefs.channels.enabled": "true",
          "observer.checkpoint_every": 24}
    straight = _run(tmp_path, "wit_rs_straight", n_steps=48, n_agents=30, **ov)
    part = tmp_path / "wit_rs_part"
    Simulation(_cfg("wit_rs_part", 24, 30, **ov), out_dir=part).run()
    resumed = Simulation(_cfg("wit_rs_part", 48, 30, **ov), out_dir=part)
    resumed.run(resume_from=part)
    assert (pq.read_table(tmp_path / "wit_rs_straight" / "l1_events.parquet")
            .to_pylist()
            == pq.read_table(part / "l1_events.parquet").to_pylist()), \
        "resume != straight(L1)"

    def _bel(sim):
        return {a.id: dict(getattr(a, "_fact_beliefs", None) or {})
                for a in sim.agents}
    assert _bel(straight) == _bel(resumed)
    assert any(_bel(straight).values()), "信念が 1 件も無い(前提が崩れた)"


def test_no_undeclared_toggles_after_the_new_key():
    from society import registry as R
    assert "beliefs.channels.enabled" in R.BY_ID
    assert R.BY_ID["beliefs.channels.enabled"].affects_k is False
    assert R.BY_ID["beliefs.channels.enabled"].fingerprint_risk == "none"
    assert R.undeclared_toggles(load_config()) == []


# =========================================================================== #
# 7) np.hypot 禁止ピン + 凍結の状態
# =========================================================================== #
def test_numpy_hypot_is_never_used():
    """★`np.hypot` は math.hypot とビット不一致の入力がある = 使用禁止を機械固定する。

    散文だけでなく **実際に評価される識別子**でも見る(numpy 側の hypot を属性でも
    関数名でも掴んでいないこと)。
    """
    src = _SRC.read_text(encoding="utf-8")
    assert "np.hypot" not in src and "numpy.hypot" not in src
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "hypot":
            assert isinstance(node.value, ast.Name) and node.value.id == "math", \
                "math 以外の hypot を呼んでいる"
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
                for a in n.names}
    assert "hypot" not in imported, "hypot を from-import で持ち込んでいる"


def test_truth_ledger_is_still_in_the_frozen_list():
    """★承認は『中身を直してよい』であって『凍結から外してよい』ではない。"""
    assert "src/society/truth_ledger.py" in MS.SPEC_FILES
    assert len(MS.SPEC_FILES) == 14
    assert MS.compute()["missing"] == []


def test_local_u01_matches_the_shared_one():
    """`_u01` を凍結ファイル内に複製した判断の担保: 値が attention 側と同型であること。"""
    for parts in (("f00001", 0), ("f00042", 12345), ("fzzz", -1)):
        assert TL._u01("wit", *parts) == ATT._u01("wit", *parts)
    assert 0.0 <= TL._u01("wit", "f", 0) < 1.0


def test_module_draws_no_random_stream():
    """乱数 stream を 1 本も引いていない(決定論ハッシュだけ)。"""
    tree = ast.parse(_SRC.read_text(encoding="utf-8"))
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert "rng" not in attrs and "random" not in attrs
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert "random" not in names
