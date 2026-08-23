"""第153 ホットパス 4 点(C2 畳み込み / 細格子門キー / physics `_admit` / party 連れ探索)。

正典: docs/log/devlog.md 第153。250k v6c の py-spy 実測(step0-1)への処置:
  C2 31.7%(`_bounded_pick` 26.6%)・物理 18.8%(**`_admit` 11.3%**)・
  `__eq__ (<string>:4)` 10.9%・`form_parties` 10.4%。

守るもの(検収基準の順)
  (1) ★**返り値・イベント列が 1 ビットも変わらない**。4 点とも「同じ答えをより速く出す」
      だけの層1(純粋な実装最適化)で、世界の力学には触らない。
      - C2: 旧実装(第152 の逐語コピー)と総当たりで同一集合・同一順序・同一オブジェクト。
      - 細格子門: 門をどこに置いても答えは同一(粗格子/細格子が同値だから)。
      - physics: `_drop_recs` ≡ 逐次 `list.remove`。実ランで L1 バイト一致。
      - party: 連れ選抜が素朴な全走査と総当たりで一致。
  (2) 新 conf キー `world.perception_fine_gate` の契約列挙ピン(既定 = 現行同値 /
      finals = 500 / registry 宣言 / scheduler 配線 / 凍結ファイル不触)。
  (3) 乱数を 1 粒も足さない・LLM 呼数不変(mock ランでの L1 一致がまとめて示す)。
検証は mock のみ(実 LLM 禁止・≤24 step)。

第154 追補(§(5)): 第153 が潰しきれなかった `_admit` の残存 O(W²)。
  占有判定の相手が「呼び出し時点の在場者(`_admit_blocked`)」と「この呼び出しで
  入った個体(`added`)」の 2 つに割れておらず、生の `members` を待ち行列ぶん
  舐めていた(空ゾーン × 待ち 3,000 で 451 ms・信号の赤明けごとに再発)。
  処置も同じ層1(相手の集合は 1 体も変わらない = 答えは 1 ビットも変わらない)。
"""
from __future__ import annotations

import json
import random

import numpy as np
import pytest
from omegaconf import OmegaConf

from society import party as PARTY
from society import physics as PH
from society import registry as R
from society.config import load_config
from society.engine.simulation import Simulation
from society.world import perception as P

from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_FINALS = _REPO / "conf" / "finals_observe.yaml"


# --------------------------------------------------------------------------- #
# 共通ヘルパ
# --------------------------------------------------------------------------- #
class _A:
    """最小 duck-type(test_hearer_cap / test_index_cell と同型)。"""

    __slots__ = ("id", "x", "y", "loc", "building", "floor", "sleeping")

    def __init__(self, i, x, y, *, building=None, floor=0, loc="street",
                 sleeping=False):
        self.id = int(i)
        self.x = float(x)
        self.y = float(y)
        self.loc = loc
        self.building = building
        self.floor = floor
        self.sleeping = sleeping


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _ids(hs):
    return [h.id for h in hs]


# --------------------------------------------------------------------------- #
# (1) C2: `_bounded_pick` / `_bounded_rank` の畳み込みは値を変えない
#     旧実装 = HEAD 02b79e3(第152)の逐語コピー。
# --------------------------------------------------------------------------- #
def _rank_pre153(dx_a, dy_a, ids, flat, keep, n, cap):
    if cap <= 0 or n <= cap:
        sel = keep[np.argsort(ids[keep], kind="stable")]
        return [flat[int(i)] for i in sel]
    d2 = dx_a[keep] * dx_a[keep] + dy_a[keep] * dy_a[keep]
    part = np.argpartition(d2, cap - 1)[:cap]
    thresh = float(d2[part].max())
    cand = np.flatnonzero(d2 <= thresh)
    order = np.lexsort((ids[keep][cand], d2[cand]))
    chosen = keep[cand[order[:cap]]]
    chosen = chosen[np.argsort(ids[chosen], kind="stable")]
    return [flat[int(i)] for i in chosen]


def _pick_pre153(nb, sx, sy, sid, radius, cap, box_prefilter=True):
    xs, ys, ids, flat = nb
    dx_a = xs - sx
    dy_a = ys - sy
    if not box_prefilter:
        keep = np.flatnonzero((np.hypot(dx_a, dy_a) <= radius) & (ids != sid))
        n = int(keep.size)
        if n == 0:
            return []
        return _rank_pre153(dx_a, dy_a, ids, flat, keep, n, cap)
    box = np.flatnonzero((np.abs(dx_a) <= radius) & (np.abs(dy_a) <= radius)
                         & (ids != sid))
    if box.size == 0:
        return []
    keep = box[np.hypot(dx_a[box], dy_a[box]) <= radius]
    n = int(keep.size)
    if n == 0:
        return []
    return _rank_pre153(dx_a, dy_a, ids, flat, keep, n, cap)


def _nb_of(pts):
    n = len(pts)
    return (np.fromiter((p[1] for p in pts), np.float64, n),
            np.fromiter((p[2] for p in pts), np.float64, n),
            np.fromiter((p[0] for p in pts), np.int64, n),
            [_A(*p) for p in pts])


def test_bounded_pick_matches_the_pre153_implementation():
    """★総当たり: 格子配置(距離同値が大量に出る)を含めて旧実装と完全一致。

    「同一オブジェクトが同一順序で返る」ところまで見る(下流は個体参照を使う)。
    """
    rnd = random.Random(20260823)
    for _ in range(700):
        n = rnd.randint(1, 40)
        lattice = rnd.random() < 0.55
        pts = []
        for i in range(n):
            if lattice:                       # 整数格子 = 距離二乗の同値が頻出
                pts.append((i, float(rnd.randint(-6, 6)), float(rnd.randint(-6, 6))))
            else:
                pts.append((i, rnd.uniform(-8, 8), rnd.uniform(-8, 8)))
        nb = _nb_of(pts)
        sid = rnd.randrange(n)
        sx, sy = (pts[sid][1], pts[sid][2]) if rnd.random() < 0.7 else \
                 (rnd.uniform(-8, 8), rnd.uniform(-8, 8))
        for radius in (0.0, 1.0, 3.0, 5.0, 1e9):
            for cap in (0, 1, 2, 5, 15, 1000):
                for boxf in (True, False):
                    want = _pick_pre153(nb, sx, sy, sid, radius, cap, boxf)
                    got = P._bounded_pick(nb, sx, sy, sid, radius, cap, boxf)
                    assert len(want) == len(got), (radius, cap, boxf)
                    assert all(u is v for u, v in zip(want, got)), (radius, cap, boxf)


def test_bounded_rank_threshold_is_the_cap_th_order_statistic():
    """畳み込みの肝: `argpartition→max` と `partition[cap-1]` は同じ実数(演算を挟まない)。"""
    rnd = np.random.default_rng(3)
    for n in (5, 17, 64, 513):
        for cap in (1, 2, 3, 7):
            if n <= cap:
                continue
            for _ in range(20):
                d2 = np.round(rnd.random(n) * 8.0, 2)     # 同値を意図的に増やす
                old = float(d2[np.argpartition(d2, cap - 1)[:cap]].max())
                new = float(np.partition(d2, cap - 1)[cap - 1])
                assert old == new, (n, cap)


def _count_pre153(idx, speaker):
    """第152 の `PerceptIndex._count`(ベクトル化枝)逐語コピー。"""
    import math
    ctx = P._context(speaker)
    if ctx[0] == "outside":
        return 0
    cx, cy = idx._cell_xy(speaker.x, speaker.y)
    radius = idx.radius
    sx, sy, sid = speaker.x, speaker.y, speaker.id
    nb = idx._count_arrays(ctx, cx, cy)
    if nb is None:
        return 0
    xs, ys, ids = nb
    lo, hi = P._radius_band(radius)
    dxa = xs - float(sx)
    dya = ys - float(sy)
    box = np.flatnonzero((np.abs(dxa) <= hi) & (np.abs(dya) <= hi)
                         & (ids != int(sid)))
    if box.size == 0:
        return 0
    h = np.hypot(dxa[box], dya[box])
    total = int(np.count_nonzero(h <= lo))
    if total != int(np.count_nonzero(h <= hi)):
        for j in np.flatnonzero((h > lo) & (h <= hi)):
            k = int(box[int(j)])
            if math.hypot(float(dxa[k]), float(dya[k])) <= radius:
                total += 1
    return total


def test_count_hearers_matches_the_pre153_implementation():
    """`_count` の外接正方形の畳み込みも値を変えない(旧実装・`len(hearers_of)` と三つ巴)。

    ★第153 の調査結果: v6c 火炎図の `__eq__ (<string>:4)` は count 経路ではなく
      physics `_admit` の `waiting.remove` の子フレームだった(下の
      `test_record_dict_equality_really_walks_the_agent_dataclass` が根拠)。
      count 経路は dataclass の等値比較を 1 度も起こさない。
    """
    rnd = random.Random(153)
    for _ in range(250):
        n = rnd.randint(1, 90)
        lattice = rnd.random() < 0.5
        ags = [_A(i,
                  float(rnd.randint(-60, 60)) if lattice else rnd.uniform(-60, 60),
                  float(rnd.randint(-60, 60)) if lattice else rnd.uniform(-60, 60))
               for i in range(n)]
        radius = rnd.choice([1.0, 5.0, 12.5, 40.0])
        idx = P.build_index(ags, radius)
        probes = list(ags) + [_A(9999, rnd.uniform(-60, 60), rnd.uniform(-60, 60))]
        for s in probes:
            want = _count_pre153(idx, s)
            assert idx._count(s, None) == want
            assert len(P.hearers_of(s, ags, radius)) == want


def test_count_hearers_calls_no_dataclass_eq():
    """★調査ピン: count 経路は `Agent.__eq__`(dataclass 生成)を 1 度も呼ばない。"""
    from society.agents.agent import Agent

    calls = {"n": 0}
    orig = Agent.__dict__["__eq__"]

    def _counting_eq(self, other):
        calls["n"] += 1
        return orig(self, other)

    agents = [Agent(id=i, name=f"a{i}", age=30, occupation="x", persona="p",
                    traits={}, states={}, x=float(i % 30) * 4.0,
                    y=float(i // 30) * 4.0)
              for i in range(300)]
    idx = P.build_index(agents, 40.0)
    Agent.__eq__ = _counting_eq
    try:
        total = sum(P.count_hearers(a, idx, 40.0) for a in agents)
    finally:
        Agent.__eq__ = orig
    assert total > 0, "テスト前提が崩れた(誰も聞こえていない)"
    assert calls["n"] == 0, "count 経路が dataclass __eq__ を呼んでいる"


def test_self_is_still_excluded_after_the_box_filter():
    """自分自身の除外を外接正方形の**後**へ移しても、話者は 1 度も返らない。"""
    pts = [(i, float(i % 5), float(i // 5)) for i in range(25)]
    nb = _nb_of(pts)
    for sid in range(25):
        sx, sy = pts[sid][1], pts[sid][2]
        for cap in (0, 3, 100):
            got = P._bounded_pick(nb, sx, sy, sid, 10.0, cap)
            assert sid not in _ids(got)


# --------------------------------------------------------------------------- #
# (2) 細格子の門(`world.perception_fine_gate`)
# --------------------------------------------------------------------------- #
def _dense(n_side=26, pitch=1.6):
    return [_A(i, (i % n_side) * pitch, (i // n_side) * pitch)
            for i in range(n_side * n_side)]


def test_gate_value_never_changes_any_answer():
    """★門をどこに置いても返り値は同一(粗格子/細格子が同値だから = 速さだけのつまみ)。"""
    ags = _dense()
    ref = P.build_index(ags, 40.0)                       # 粗格子のみ(現行経路)
    for gate in (0.0, 1.0, 500.0, 2000.0, 1e18):
        idx = P.build_index(ags, 40.0, cell_m=5.0, fine_gain=gate)
        for r in (5.0, 10.0, 20.0):
            for cap in (0, 15):
                for a in ags[::7]:
                    assert _ids(idx.hearers(a, cap, r)) == _ids(ref.hearers(a, cap, r)), \
                        (gate, r, cap, a.id)


def test_gate_actually_moves_the_boundary():
    """門を下げると細格子へ回る近傍が広がる(= つまみが効いている)ことの構造ピン。"""
    ags = _dense()
    hi = P.build_index(ags, 40.0, cell_m=5.0, fine_gain=1e18)
    for a in ags[:30]:
        hi.hearers(a, 15, 5.0)
    assert hi._fcells is None, "門を無限大にしたのに細格子が建った"
    lo = P.build_index(ags, 40.0, cell_m=5.0, fine_gain=0.0)
    for a in ags[:30]:
        lo.hearers(a, 15, 5.0)
    assert lo._fcells is not None, "門を 0 にしたのに細格子を 1 度も使っていない"


def test_default_index_gate_is_the_module_constant():
    idx = P.build_index([_A(0, 0, 0)], 40.0, cell_m=5.0)
    assert idx.fine_gain is None, "既定は None(= 実装既定に従う)でなければならない"


def test_conf_default_is_the_implementation_default():
    cfg = load_config()
    assert float(cfg.world.perception_fine_gate) == P._FINE_GATE_CONF_DEFAULT

    class _S:
        pass
    sim = _S()
    sim.cfg = cfg
    assert P.fine_gate_of(sim) is None, "既定値の明示は『実装既定に従う』でなければならない"


def test_fine_gate_of_reads_an_explicit_value():
    class _S:
        pass
    sim = _S()
    sim.cfg = load_config(["world.perception_fine_gate=500"])
    assert P.fine_gate_of(sim) == 500.0
    assert P.fine_gate_of(sim) == 500.0          # キャッシュしても同じ


def test_finals_profile_declares_the_gate():
    fin = OmegaConf.load(_FINALS)
    assert float(fin.world.perception_fine_gate) == 500.0
    # 門は細格子キーが立っているランでしか読まれない(前提の確認)。
    assert float(fin.world.perception_cell_m) > 0.0


def test_registry_declares_the_gate():
    f = R.BY_ID.get("world.perception_fine_gate")
    assert f is not None, "world.perception_fine_gate がレジストリ未宣言"
    assert f.repro_tier == "strict"
    assert f.fingerprint_risk == "none"
    assert f.affects_k is False              # 層1 = LLM 呼の発生点も本数も変わらない
    assert f.off_value == 2000
    assert R.undeclared_toggles(load_config()) == []


def _sim(tmp_path, name, n=24, steps=12, **ov):
    dot = ["run.seed=42", f"run.n_agents={n}", f"run.n_steps={steps}",
           f"run.name={name}", "model.backend=mock"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def test_scheduler_builds_the_index_with_the_configured_gate(tmp_path):
    sim = _sim(tmp_path, "gate_wired", steps=2,
               **{"world.perception_cell_m": 5.0,
                  "world.perception_fine_gate": 500})
    sim.run()
    assert sim.percept_index.fine_gain == 500.0
    off = _sim(tmp_path, "gate_wired_off", steps=2,
               **{"world.perception_cell_m": 5.0})
    off.run()
    assert off.percept_index.fine_gain is None


def test_gate_run_is_l1_identical(tmp_path):
    """★門を動かした実ランが既定ランと L1 バイト一致(世界は 1 ビットも動かない)。"""
    ov = {"world.speech_levels.enabled": "true", "world.c2_neighbors_max": 15,
          "conversation.enabled": "true", "world.perception_cell_m": 5.0}
    base = _sim(tmp_path, "gate_l1_base", **ov)
    base.run()
    low = _sim(tmp_path, "gate_l1_low", **{**ov, "world.perception_fine_gate": 1})
    low.run()
    assert _l1(base) == _l1(low)


# --------------------------------------------------------------------------- #
# (3) physics `_admit` / `_drop_recs`
# --------------------------------------------------------------------------- #
def _naive_drop(lst, drop):
    """第153 以前の作法(= 1 件ずつ `list.remove`)。テストの独立オラクル。"""
    for rec in drop:
        lst.remove(rec)


def test_drop_recs_matches_sequential_remove():
    """★総当たり: 落とす件数・順序・位置をランダムに振っても逐次 remove と同一の残り。"""
    rnd = random.Random(153)
    for _ in range(3000):
        n = rnd.randint(0, 30)
        recs = [{"agent": _A(i, i, 0), "k": i} for i in range(n)]
        k = rnd.randint(0, n)
        drop = rnd.sample(recs, k)
        a, b = list(recs), list(recs)
        _naive_drop(a, drop)
        PH._drop_recs(b, drop)
        assert len(a) == len(b)
        assert all(u is v for u, v in zip(a, b))


def test_drop_recs_falls_back_when_identity_is_absent():
    """同一性で見つからない要素があれば従来の `remove`(等値)へ後退する。"""
    r0 = {"agent": _A(0, 0, 0), "k": 0}
    r1 = {"agent": _A(1, 1, 0), "k": 1}
    lst = [r0, r1]
    twin = dict(r0)                          # 別オブジェクトだが == で一致する双子
    PH._drop_recs(lst, [twin])
    assert lst == [r1], "等値の要素が落ちていない(後退経路が働いていない)"


def test_drop_recs_is_a_noop_for_an_empty_drop():
    lst = [{"agent": _A(0, 0, 0)}]
    before = list(lst)
    PH._drop_recs(lst, [])
    assert lst == before


def test_record_dict_equality_really_walks_the_agent_dataclass():
    """根拠のピン: 流入レコードの `==` は `Agent.__eq__`(dataclass 生成)を必ず呼ぶ。

    これが `list.remove` の中で 1 要素ごとに走っていたのが `__eq__ (<string>:4)` 10.9%
    の正体。`_drop_recs` はこの比較を 1 度も起こさない。
    """
    from society.agents.agent import Agent

    calls = {"n": 0}
    orig = Agent.__dict__["__eq__"]
    assert orig.__code__.co_filename == "<string>", "dataclass 生成の __eq__ ではない"

    def _counting_eq(self, other):
        calls["n"] += 1
        return orig(self, other)

    def _mk(i):
        return Agent(id=i, name=f"a{i}", age=30, occupation="x", persona="p",
                     traits={}, states={})

    recs = [{"agent": _mk(i), "k": i} for i in range(8)]
    Agent.__eq__ = _counting_eq
    try:
        a = list(recs)
        _naive_drop(a, recs[4:])            # 手前 4 件を等値比較で舐める
        assert calls["n"] > 0, "前提が崩れた(dict の等値比較が Agent を見ていない)"
        calls["n"] = 0
        b = list(recs)
        PH._drop_recs(b, recs[4:])
        assert calls["n"] == 0, "_drop_recs が dataclass __eq__ を呼んでいる"
        assert len(a) == len(b) and all(u is v for u, v in zip(a, b))
    finally:
        Agent.__eq__ = orig


_ZONE_R = 25.0
_POLY = [[-_ZONE_R, -_ZONE_R], [_ZONE_R, -_ZONE_R], [_ZONE_R, _ZONE_R],
         [-_ZONE_R, _ZONE_R]]


def _zone_spec(zid="z1", engine="orca", **ov):
    z = {"id": zid, "engine": engine, "dt_sub": 0.05, "polygon": list(_POLY)}
    z.update(ov)
    return z


def _phys_sim(tmp_path, name, n=30, steps=8):
    dot = ["run.seed=42", f"run.n_agents={n}", f"run.n_steps={steps}",
           f"run.name={name}", "model.backend=mock", "observer.snapshot_every=144"]
    cfg = load_config(dot)
    OmegaConf.update(cfg, "physics.zones_enabled", True, force_add=True)
    cfg.physics.zones = [_zone_spec()]
    return Simulation(cfg, out_dir=tmp_path / name)


def _admit_pre153(sim, zone, waiting, members, signal, sim_sec, t_in_step,
                  step, sim_min, st) -> bool:
    """第152 の `_admit` 逐語コピー(独立オラクル)。"""
    from society.observer.schema import Event
    if signal is not None and not signal.can_cross(sim_sec):
        return False
    gap = float(zone.gate["min_gap_m"])
    any_admitted = False
    queue = list(waiting)
    blocked = PH._admit_blocked(queue, members, gap)
    added: list = []
    for qi, rec in enumerate(queue):
        px, py = rec["pos"]
        free = not (blocked is not None and blocked[qi])
        if free and blocked is None:
            for other in members:
                ox, oy = other["pos"]
                need = rec["radius"] + other["radius"] + gap
                if (px - ox) ** 2 + (py - oy) ** 2 < need * need:
                    free = False
                    break
        if free and added:
            for other in added:
                ox, oy = other["pos"]
                need = rec["radius"] + other["radius"] + gap
                if (px - ox) ** 2 + (py - oy) ** 2 < need * need:
                    free = False
                    break
        if not free:
            continue
        added.append(rec)
        rec["waiting"] = False
        rec["seen_inside"] = zone.contains(px, py)
        waiting.remove(rec)
        members.append(rec)
        members.sort(key=lambda r: int(r["agent"].id))
        any_admitted = True
        st["enter_total"] += 1
        agent = rec["agent"]
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                             kind="zone_gate", x=agent.x, y=agent.y,
                             payload={"zone": zone.id, "gate": rec["gate"],
                                      "dir": "enter", "engine": zone.engine,
                                      "v0": round(rec["v0"], 3),
                                      "speed": round(float(np.hypot(*rec["vel"])), 3),
                                      "span_m": round(float(rec["span_m"]), 1),
                                      "wait_s": round(float(t_in_step)
                                                      + float(rec["wait_steps"])
                                                      * float(sim.clock.step_seconds), 2),
                                      "waited_steps": int(rec["wait_steps"])}))
    return any_admitted


def test_admit_matches_the_pre153_reference_in_a_real_run(tmp_path, monkeypatch):
    """★実ラン A/B: `_admit` を第152 の逐語コピーへ差し替えても L1 がバイト一致。"""
    new = _phys_sim(tmp_path, "admit_new")
    new.run()
    assert [e for e in new.logger.events if e.kind == "zone_gate"], \
        "テスト前提が崩れた(ゾーンを誰も通らない)"

    monkeypatch.setattr(PH, "_admit", _admit_pre153)
    monkeypatch.setattr(PH, "_drop_recs", _naive_drop)
    old = _phys_sim(tmp_path, "admit_old")
    old.run()
    assert _l1(new) == _l1(old)


def test_run_zone_drop_paths_match_sequential_remove(tmp_path, monkeypatch):
    """`_drop_recs` を逐次 remove へ差し替えた実ランと L1 バイト一致(退場・強制返却も込み)。"""
    new = _phys_sim(tmp_path, "drop_new", steps=10)
    new.run()
    monkeypatch.setattr(PH, "_drop_recs", _naive_drop)
    old = _phys_sim(tmp_path, "drop_old", steps=10)
    old.run()
    assert _l1(new) == _l1(old)


def test_admit_still_keeps_members_in_id_order(tmp_path):
    """整列をループの外へ出しても在場者は agent.id 昇順のまま(id は一意 = 同じ全順序)。"""
    dot = ["run.seed=42", "run.n_agents=12", "run.n_steps=1", "run.name=admit_order",
           "model.backend=mock", "observer.snapshot_every=144"]
    cfg = load_config(dot)
    OmegaConf.update(cfg, "physics.zones_enabled", True, force_add=True)
    cfg.physics.zones = [_zone_spec(gate={"min_gap_m": 0.0, "band_m": 3.0,
                                          "max_hold_steps": 3, "max_zone_steps": 6,
                                          "handover_jump_max_m": 20.0})]
    sim = Simulation(cfg, out_dir=None)
    zone = sim.physcfg["zones"][0]
    st = PH._new_state()
    recs = []
    for k, agent in enumerate(reversed(sim.agents[:6])):     # わざと id 降順で並べる
        agent.x, agent.y = 100.0 * k, 0.0
        nxt = sorted(sim.city.graph.neighbors(agent.node))[0]
        rec = dict(PH._admit_record(sim, zone, agent, [agent.node, nxt], [], 0))
        rec["pos"] = (100.0 * k, 0.0)
        agent._phys_zone = zone.id
        recs.append(rec)
    waiting, members = list(recs), []
    assert PH._admit(sim, zone, waiting, members, None, 0.0, 0.0, 0, 0, st) is True
    assert not waiting, "全員入れるはずの配置で待ちが残った"
    assert [int(r["agent"].id) for r in members] == \
        sorted(int(r["agent"].id) for r in members)


# --------------------------------------------------------------------------- #
# (4) party: 連れ探索の head カーソル
# --------------------------------------------------------------------------- #
def _companions_naive(sizes, max_p):
    """第152 以前の全走査(独立オラクル)。(代表, 連れ) の列を返す。"""
    ids = list(range(len(sizes)))
    assigned: set = set()
    out = []
    for lid in ids:
        if lid in assigned:
            continue
        want = min(sizes[lid], max_p)
        if want < 2:
            continue
        members = [lid]
        for c in ids:
            if len(members) >= want:
                break
            if c == lid or c in assigned:
                continue
            members.append(c)
        if len(members) < 2:
            continue
        out.append(tuple(members))
        assigned.update(members)
    return out


def _companions_cursor(sizes, max_p):
    """実装と同じ head カーソル(party.form_parties の内側と同一の規則)。"""
    ids = list(range(len(sizes)))
    n_v = len(ids)
    assigned: set = set()
    out = []
    head = 0
    for lid in ids:
        if lid in assigned:
            continue
        want = min(sizes[lid], max_p)
        if want < 2:
            continue
        members = [lid]
        while head < n_v and ids[head] in assigned:
            head += 1
        for j in range(head, n_v):
            if len(members) >= want:
                break
            c = ids[j]
            if c == lid or c in assigned:
                continue
            members.append(c)
        if len(members) < 2:
            continue
        out.append(tuple(members))
        assigned.update(members)
    return out


def test_party_cursor_matches_the_full_scan_exhaustively():
    """★網羅: n<=6 の全 party_size 組み合わせ × 上限 3 種で全走査と一致。"""
    from itertools import product
    for n in range(0, 7):
        for sizes in product(range(1, 5), repeat=n):
            for max_p in (2, 3, 5):
                assert _companions_naive(list(sizes), max_p) == \
                    _companions_cursor(list(sizes), max_p), (sizes, max_p)


def test_party_cursor_matches_the_full_scan_on_random_scenes():
    rnd = random.Random(45)
    for _ in range(3000):
        n = rnd.randint(0, 60)
        sizes = [rnd.randint(1, 7) for _ in range(n)]
        max_p = rnd.choice([2, 3, 5, 8])
        assert _companions_naive(sizes, max_p) == _companions_cursor(sizes, max_p)


def _party_sim(tmp_path, name, n=40, steps=1):
    dot = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
           "observer.snapshot_every=144", "party.enabled=true", "party.roam_bias=1.0"]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


@pytest.mark.parametrize("size", [2, 3, 5])
def test_form_parties_groups_match_the_full_scan(tmp_path, size):
    """実 sim の form_parties が出す組が、素朴な全走査の答えと一致する。"""
    sim = _party_sim(tmp_path, f"party_scan_{size}")
    for k, a in enumerate(sim.agents):
        a.visitor = True
        a.party_size = size if (k % 3) else 1        # 単独者を混ぜる(head が滞る形)
    PARTY.form_parties(sim, step=0, sim_min=0)
    groups = sorted(tuple(sorted(e.payload["with"]))
                    for e in sim.logger.events
                    if e.kind == "joint_activity" and e.payload.get("type") == "party")
    visitors = sorted((a for a in sim.agents if a.visitor), key=lambda a: a.id)
    order = [v.id for v in visitors]
    sizes = [size if (k % 3) else 1 for k in range(len(sim.agents))]
    sizes = [sizes[sim.agents.index(v)] for v in visitors]
    want = sorted(tuple(sorted(order[i] for i in g))
                  for g in _companions_naive(sizes, 5))
    assert groups == want


def test_party_run_is_deterministic_and_repeatable(tmp_path):
    a = _party_sim(tmp_path, "party_det_a", steps=6)
    a.run()
    b = _party_sim(tmp_path, "party_det_b", steps=6)
    b.run()
    assert _l1(a) == _l1(b)


# --------------------------------------------------------------------------- #
# (5) 第154 physics `_admit` に残っていた O(W²)(入場済みの総当たり)
#
# 第153 は `waiting.remove` を潰したが、占有判定そのものには 2 つの総当たりが
# 残っていた:
#   (a) `blocked is None`(在場者 ≤ 48 = **ゾーンが空**)のとき、生の `members` を
#       待ち行列ぶん舐める。`members` は入場のたびに伸びるので O(W²)。
#   (b) `added`(この呼び出しで入った個体)の逐次リスト走査。これは `blocked` の
#       有無に依らず走るので、(a) を消しても O(W²) は残る。
#   おまけに `blocked is None` のときは `added ⊆ members` なので (b) は (a) の
#   **完全な冗長**だった。
# 処置: (a) は呼び出し時点のスナップショット `base` に、(b) は `_AdmitCells` の
#       増分セル法に。相手の集合は 1 体も変わらない = 答えは 1 ビットも変わらない。
# 実測(空ゾーン × 待ち 3,000): 広場配置 452 ms → 7.9 ms(57x)/散開配置
# 1,408 ms → 13.8 ms(102x)/縁石配置 26.9 ms → 4.4 ms(6.1x)。
# 「1 体も入れない」呼び出し(赤の間)の固定費は不変(4.29 ms → 4.23 ms)。
# --------------------------------------------------------------------------- #
def _admit_pre154(sim, zone, waiting, members, signal, sim_sec, t_in_step,
                  step, sim_min, st) -> bool:
    """第153(HEAD 2eb065f)の `_admit` 逐語コピー(独立オラクル)。"""
    from society.observer.schema import Event
    if signal is not None and not signal.can_cross(sim_sec):
        return False
    gap = float(zone.gate["min_gap_m"])
    any_admitted = False
    queue = list(waiting)
    blocked = PH._admit_blocked(queue, members, gap)
    added: list = []
    for qi, rec in enumerate(queue):
        px, py = rec["pos"]
        free = not (blocked is not None and blocked[qi])
        if free and blocked is None:
            for other in members:
                ox, oy = other["pos"]
                need = rec["radius"] + other["radius"] + gap
                if (px - ox) ** 2 + (py - oy) ** 2 < need * need:
                    free = False
                    break
        if free and added:
            for other in added:
                ox, oy = other["pos"]
                need = rec["radius"] + other["radius"] + gap
                if (px - ox) ** 2 + (py - oy) ** 2 < need * need:
                    free = False
                    break
        if not free:
            continue
        added.append(rec)
        rec["waiting"] = False
        rec["seen_inside"] = zone.contains(px, py)
        members.append(rec)
        any_admitted = True
        st["enter_total"] += 1
        agent = rec["agent"]
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                             kind="zone_gate", x=agent.x, y=agent.y,
                             payload={"zone": zone.id, "gate": rec["gate"],
                                      "dir": "enter", "engine": zone.engine,
                                      "v0": round(rec["v0"], 3),
                                      "speed": round(float(np.hypot(*rec["vel"])), 3),
                                      "span_m": round(float(rec["span_m"]), 1),
                                      "wait_s": round(float(t_in_step)
                                                      + float(rec["wait_steps"])
                                                      * float(sim.clock.step_seconds), 2),
                                      "waited_steps": int(rec["wait_steps"])}))
    if any_admitted:
        members.sort(key=lambda r: int(r["agent"].id))
        PH._drop_recs(waiting, added)
    return any_admitted


# ── `_admit` を単体で回すための最小スタブ(sim/zone は使う口しか要らない)──── #
class _StubLogger:
    def __init__(self):
        self.rows: list = []

    def log(self, ev):
        self.rows.append((ev.step, ev.sim_min, ev.agent_id, ev.kind, ev.x, ev.y,
                          json.dumps(ev.payload, ensure_ascii=False,
                                     sort_keys=True)))


class _StubClock:
    step_seconds = 600.0


class _StubSim:
    def __init__(self):
        self.logger = _StubLogger()
        self.clock = _StubClock()


class _StubZone:
    def __init__(self, gap, half=25.0):
        self.id = "z1"
        self.engine = "orca"
        self.gate = {"min_gap_m": gap}
        self.half = half

    def contains(self, x, y):
        return abs(x) <= self.half and abs(y) <= self.half


def _wrec(i, x, y, r):
    """`_admit_record` と同じ形の流入レコード(必要なキーだけ)。"""
    return {"agent": _A(i, x, y), "zone": "z1", "path": [0, 1], "rest": [],
            "gate": "g0", "exit_xy": (0.0, 0.0), "span_m": 30.0, "wp": 1,
            "wp_xy": (0.0, 0.0), "pos": (x, y), "vel": (1.0, 0.0),
            "dir0": (1.0, 0.0), "seg_dir": (1.0, 0.0), "v0": 1.3, "radius": r,
            "waiting": True, "seen_inside": False, "wait_steps": i % 4,
            "step_n": 0, "elapsed_s": 0.0}


def _admit_outcome(fn, wrecs, mrecs, gap):
    """入場集合・順序・座標・待ち残り・件数・イベント列 をまとめて返す。"""
    sim, zone, st = _StubSim(), _StubZone(gap), {"enter_total": 0}
    waiting = [dict(r) for r in wrecs]
    members = [dict(r) for r in mrecs]
    ret = fn(sim, zone, waiting, members, None, 0.0, 0.0, 0, 0, st)
    return (ret,
            [(r["agent"].id, r["pos"], r["radius"], r["waiting"],
              r["seen_inside"]) for r in members],
            [(r["agent"].id, r["waiting"], r["seen_inside"]) for r in waiting],
            st["enter_total"], sim.logger.rows)


def _admit_scene(rnd, n_w, n_m, spread, mode):
    """(待ち, 在場) のレコード列。mode で配置の性質を変える。"""
    def _pos():
        if mode == "uniform":
            return (rnd.uniform(-spread, spread), rnd.uniform(-spread, spread))
        if mode == "lattice":                  # 距離同値・体表接触が大量に出る格子
            return (0.35 * rnd.randint(-12, 12), 0.35 * rnd.randint(-12, 12))
        if mode == "cellline":                 # ★セル境界(1.5·reach ≒ 1.2 m)直撃
            eps = rnd.choice([0.0, 1e-16, -1e-16, 1e-12, -1e-12, 5e-9, -5e-9])
            return (1.2 * rnd.randint(-8, 8) + eps,
                    1.2 * rnd.randint(-8, 8) + eps)
        if mode == "dup":                      # 完全同一座標の山
            return (rnd.choice([0.0, 0.5, 1.0]), rnd.choice([0.0, 0.5, 1.0]))
        raise AssertionError(mode)

    ids = rnd.sample(range(100000), n_w + n_m)
    recs = [_wrec(ids[k], *_pos(), rnd.uniform(0.25, 0.35))
            for k in range(n_w + n_m)]
    for r in recs[n_w:]:
        r["waiting"] = False
    return recs[:n_w], recs[n_w:]


_ADMIT_MODES = ("uniform", "lattice", "cellline", "dup")


@pytest.mark.parametrize("cell_min,cell_work", [
    (24, 1),            # 実装の既定
    (-1, 0),            # ★格子を最初の 1 体から強制 ON(セル法だけを見る)
    (10 ** 9, 10 ** 9),  # 格子を完全 OFF(素の総当たり = 冗長ループ除去だけを見る)
    (4, 2),             # 切替点をずらす
])
def test_admit_matches_the_pre154_reference_on_random_scenes(cell_min, cell_work,
                                                             monkeypatch):
    """★同値証明: 第153 逐語オラクルと入場集合・順序・座標・イベント列まで完全一致。

    切替閾値をどこに置いても答えが同じ = 「速い経路」と「素の経路」が同値である
    ことの直接の証拠(第153 の `world.perception_fine_gate` と同じ作法)。
    """
    monkeypatch.setattr(PH, "_ADMIT_CELL_MIN", cell_min)
    monkeypatch.setattr(PH, "_ADMIT_CELL_WORK", cell_work)
    rnd = random.Random(20260823)
    for k in range(400):
        mode = _ADMIT_MODES[k % len(_ADMIT_MODES)]
        n_w = rnd.choice([0, 1, 2, 3, 7, 25, 26, 60, 120, 300])
        n_m = rnd.choice([0, 1, 5, 47, 48, 49, 90, 250])
        spread = rnd.choice([0.5, 2.0, 8.0, 40.0])
        gap = rnd.choice([0.0, 0.1, 0.5, 2.0])
        w0, m0 = _admit_scene(rnd, n_w, n_m, spread, mode)
        assert _admit_outcome(PH._admit, w0, m0, gap) == \
            _admit_outcome(_admit_pre154, w0, m0, gap), \
            (k, mode, n_w, n_m, spread, gap)


def test_admit_cells_hit_matches_the_full_scan():
    """`_AdmitCells.hit` ≡ `for other in added:` の総当たり(同じ bool)。"""
    rnd = random.Random(154)
    for _ in range(400):
        gap = rnd.choice([0.0, 0.1, 0.5])
        pts = [(rnd.uniform(-3, 3), rnd.uniform(-3, 3), rnd.uniform(0.25, 0.35))
               for _ in range(rnd.randint(1, 60))]
        cells = PH._AdmitCells.build(
            [{"radius": p[2], "pos": (p[0], p[1])} for p in pts], gap)
        assert cells is not None
        for p in pts:
            cells.add(*p)
        for _ in range(20):
            px, py = rnd.uniform(-3.5, 3.5), rnd.uniform(-3.5, 3.5)
            r = rnd.uniform(0.25, 0.35)
            want = any((px - ox) ** 2 + (py - oy) ** 2
                       < (r + orad + gap) * (r + orad + gap)
                       for ox, oy, orad in pts)
            assert cells.hit(px, py, r) is want


def test_admit_cells_build_seeds_the_already_admitted():
    """遅延構築の backfill: `seed` に渡した個体が最初の問い合わせから見えている。"""
    seed = [_wrec(1, 0.0, 0.0, 0.30), _wrec(2, 10.0, 0.0, 0.30)]
    cells = PH._AdmitCells.build(seed, 0.1, seed)
    assert cells.hit(0.2, 0.0, 0.30) is True
    assert cells.hit(10.2, 0.0, 0.30) is True
    assert cells.hit(5.0, 0.0, 0.30) is False


def test_admit_cells_ring_covers_every_pair_within_reach():
    """★幾何の要: セル辺 = 1.5·reach なら `floor` の差は高々 1 = 3×3 で取りこぼし無し。

    座標の桁を 10⁻³〜10¹² まで振っても差は 1 を超えない(余裕 0.333 に対して
    丸め誤差は 10⁻¹⁶ 桁)。ここが破れると `hit` が偽陰性を返しうる。
    """
    import math
    rnd = random.Random(1154)
    worst = 0
    for mag in (1e-3, 1.0, 1e3, 1e6, 1e9, 1e12):
        for _ in range(20000):
            r_max = rnd.uniform(0.25, 0.35)
            gap = rnd.choice([0.0, 0.1, 0.5, 2.0])
            reach = r_max + r_max + gap
            cell = reach * PH._ADMIT_CELL_SCALE
            px = rnd.uniform(-mag, mag)
            ox = px + rnd.uniform(-reach, reach)
            if abs(px - ox) > reach:
                continue
            worst = max(worst, abs(math.floor(px / cell) - math.floor(ox / cell)))
    assert worst <= 1, f"3×3 リングでは足りない(floor 差 {worst})"


def test_admit_cells_build_refuses_degenerate_input():
    """前提が崩れる入力では格子を作らない(= 素の総当たりへ後退 = 従来と同じ答え)。"""
    ok = [{"radius": 0.3, "pos": (0.0, 0.0)}]
    assert PH._AdmitCells.build(ok, 0.1) is not None
    assert PH._AdmitCells.build([], 0.1) is None                     # 空の待ち
    assert PH._AdmitCells.build(ok, -0.1) is None                    # 負の間隙
    assert PH._AdmitCells.build([{"radius": 0.0, "pos": (0.0, 0.0)}], 0.0) is None
    for bad in ((float("nan"), 0.0), (float("inf"), 0.0), (0.0, float("nan"))):
        assert PH._AdmitCells.build([{"radius": 0.3, "pos": bad}], 0.1) is None
    assert PH._AdmitCells.build([{"radius": -0.3, "pos": (0.0, 0.0)}], 0.1) is None
    # 極小セル × 天文学的座標(`math.floor(px/cell)` が OverflowError を投げる形)
    assert PH._AdmitCells.build([{"radius": 0.0, "pos": (1e308, 0.0)},
                                 {"radius": 5e-324, "pos": (0.0, 0.0)}], 0.0) is None


def test_admit_falls_back_verbatim_on_degenerate_positions():
    """非有限座標が混じっても第153 と同じ答え(格子は組まれず素の総当たりになる)。"""
    rnd = random.Random(9)
    w = [_wrec(i, rnd.uniform(-2, 2), rnd.uniform(-2, 2), 0.30) for i in range(60)]
    w[7]["pos"] = (float("nan"), 0.0)
    w[31]["pos"] = (0.0, float("inf"))
    assert _admit_outcome(PH._admit, w, [], 0.1) == \
        _admit_outcome(_admit_pre154, w, [], 0.1)


def test_admit_work_per_agent_is_bounded_not_quadratic():
    """★計算量のピン: 1 体あたりに見る相手の数が待ち行列長に依らず定数で収まる。

    セル法が見るのは 3×3 セル。入場済みは互いに `need`(≒0.6-0.8 m)以上離れて
    いるので、辺 1.5·reach のセル 9 個に入りうる点数は**充填限界で頭打ち**になる
    (待ち W を 5 倍にしても 1 体あたりの候補数は増えない)。第153 は 1 体あたり
    「それまでに入場した全員」= W/2 相当を舐めていた。
    """
    import math
    seen = {"cand": 0}
    orig_hit = PH._AdmitCells.hit

    def _counting_hit(self, px, py, radius):
        cx = math.floor(px / self.cell)
        cy = math.floor(py / self.cell)
        for ix in (cx - 1, cx, cx + 1):
            for iy in (cy - 1, cy, cy + 1):
                seen["cand"] += len(self.bins.get((ix, iy), ()))
        return orig_hit(self, px, py, radius)

    rnd = random.Random(77)
    per = {}
    PH._AdmitCells.hit = _counting_hit
    try:
        for n_w in (600, 3000):
            seen["cand"] = 0
            w = [_wrec(i, rnd.uniform(-40, 40), rnd.uniform(-20, 20),
                       rnd.uniform(0.25, 0.35)) for i in range(n_w)]
            out = _admit_outcome(PH._admit, w, [], 0.1)
            assert out[3] > 200, "テスト前提が崩れた(ほとんど入場していない)"
            per[n_w] = seen["cand"] / n_w
    finally:
        PH._AdmitCells.hit = orig_hit
    # 充填限界: 半径 0.3 + 間隙 0.1 の点が辺 1.2 m のセル 9 個(3.6 m 角)に
    # 入りうる数の上界は (3.6/0.6)² = 36。
    for n_w, v in per.items():
        assert v <= 40.0, f"1 体あたりの候補が多すぎる(W={n_w}: {v:.1f})"


def test_admit_matches_the_pre154_reference_in_a_real_run(tmp_path, monkeypatch):
    """★実ラン A/B: `_admit` を第153 の逐語コピーへ差し替えても L1 がバイト一致。"""
    new = _phys_sim(tmp_path, "admit154_new")
    new.run()
    assert [e for e in new.logger.events if e.kind == "zone_gate"], \
        "テスト前提が崩れた(ゾーンを誰も通らない)"

    monkeypatch.setattr(PH, "_admit", _admit_pre154)
    old = _phys_sim(tmp_path, "admit154_old")
    old.run()
    assert _l1(new) == _l1(old)


def test_admit_real_run_is_l1_identical_with_the_grid_forced_on(tmp_path,
                                                                monkeypatch):
    """小さな実ランでは格子が組まれないので、強制 ON でも L1 が変わらないことを見る。"""
    base = _phys_sim(tmp_path, "admit154_base")
    base.run()
    monkeypatch.setattr(PH, "_ADMIT_CELL_MIN", -1)
    monkeypatch.setattr(PH, "_ADMIT_CELL_WORK", 0)
    forced = _phys_sim(tmp_path, "admit154_forced")
    forced.run()
    assert _l1(base) == _l1(forced)


class _CountingRec(dict):
    """`rec["pos"]` の読み出し回数を数える流入レコード(冗長走査の検出器)。"""

    reads = [0]

    def __getitem__(self, key):
        if key == "pos":
            _CountingRec.reads[0] += 1
        return dict.__getitem__(self, key)


def test_admit_never_rescans_the_admitted_twice(monkeypatch):
    """★冗長ループの除去: 入場済みを 2 度舐めない(`base` は呼び出し時点の写し)。

    第153 は `blocked is None`(在場者 ≤ `_ADMIT_HASH_MIN`)のとき、入場のたびに
    伸びる生の `members` を舐めた**後で** `added` をもう一度舐めていた。
    `added ⊆ members` なので後者は完全な冗長で、W 体が全員入れる配置では
    `rec["pos"]` の読み出しにちょうど W·(W−1)/2 回ぶん現れる。
    格子は切って(素の経路どうしで)比べる = 冗長の除去だけを見る。
    """
    monkeypatch.setattr(PH, "_ADMIT_CELL_MIN", 10 ** 9)
    monkeypatch.setattr(PH, "_ADMIT_CELL_WORK", 10 ** 9)
    n_w, n_m = 40, 10
    # 全員が互いに十分離れた配置 = 誰も弾かれない(走査が最後まで走る)
    w = [_wrec(i, 100.0 * i, 0.0, 0.30) for i in range(n_w)]
    m = [_wrec(1000 + j, -100.0 * (j + 1), 0.0, 0.30) for j in range(n_m)]
    for r in m:
        r["waiting"] = False
    reads = {}
    for tag, fn in (("new", PH._admit), ("old", _admit_pre154)):
        sim, zone, st = _StubSim(), _StubZone(0.1), {"enter_total": 0}
        waiting = [_CountingRec(r) for r in w]
        members = [_CountingRec(r) for r in m]
        _CountingRec.reads[0] = 0
        fn(sim, zone, waiting, members, None, 0.0, 0.0, 0, 0, st)
        assert st["enter_total"] == n_w, "テスト前提が崩れた(全員入れる配置のはず)"
        reads[tag] = _CountingRec.reads[0]
    assert reads["old"] - reads["new"] == n_w * (n_w - 1) // 2, reads


# --------------------------------------------------------------------------- #
# (6) 凍結ファイル不触
# --------------------------------------------------------------------------- #
def test_touched_files_are_not_frozen():
    from society.observer import metrics_spec as MS
    for rel in ("src/society/world/perception.py",
                "src/society/engine/scheduler.py",
                "src/society/physics.py",
                "src/society/party.py",
                "src/society/registry.py"):
        assert rel not in MS.SPEC_FILES, f"凍結ファイルを触っている: {rel}"
