"""step 時間監査の是正 A3 / A4 / A7 / C1(第155 レーン2)のテスト。

正典: docs/plans/step-time-audit.md §3 ランク表 / §4 適用順 /
      src/society/physics.py の「step 時間監査の是正」節 / conf/config.yaml `physics_levers`

守るもの(R1 の鉄則)
--------------------
- **A3 / A4 は出力バイト同一**(conf キーを持たない = 常時有効の同値変換)。
  A3 は「engine が持つ半径配列は列内包の結果と同一で、以後書き換えられない」ことを
  機械証明する。A4 は **A4 以前の呼び出し順序を復元した対照ラン**と L1 バイト一致を照合する。
- **A7 / C1 は既定 = 現行と 1 バイト同一**(鍵を書いても書かなくても L1 も continuity も同じ)。
  ON は世界(C1)/ 診断値(A7)を変えるが、**新しい状態は 1 つも足さない** = resume==straight。
- 決定論: 距離判定も間引きも座標・カウンタの純関数。"physics" stream の消費本数は不変。
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from omegaconf import OmegaConf

from society import physics as P
from society import registry as R
from society.config import load_config
from society.engine import checkpoint, scheduler
from society.engine.simulation import Simulation
from society.world import orca_core as _orca
from society.world import sfm_core as _sfm
from society.world import zones

REPO = Path(__file__).resolve().parents[1]

_ZONE_R = 25.0
_POLY = [[-_ZONE_R, -_ZONE_R], [_ZONE_R, -_ZONE_R], [_ZONE_R, _ZONE_R], [-_ZONE_R, _ZONE_R]]

#: 赤 = 入場不可の窓を必ず含む信号(cycle 20 s / green 4 s / flash 0 s = 80% が赤)。
#: ★offset 5 s が要る: 1 step = 600 s はサイクル 20 s の整数倍なので、offset 0 だと
#:   **どの step も φ=0(青)から始まり**、待機列が最初のサブステップで空になってしまう
#:   (= 赤のサブステップが 1 つも生えず A4 の効きが観測できない)。
_SIGNAL = {"mode": "explicit", "cycle_s": 20.0, "green_s": 4.0, "flash_s": 0.0,
           "offset_s": 5.0}
#: 入口の占有余裕を広く採り、**在場者と待機列が同時に存在する**状態を作る
#: (= 旧実装が赤のサブステップで捨てていた `_writeback` が実際に生える条件)。
_GATE_TIGHT = {"min_gap_m": 8.0}


def _zone(zid="z1", engine="orca", **ov):
    z = {"id": zid, "engine": engine, "dt_sub": 0.05, "polygon": list(_POLY)}
    z.update(ov)
    return z


def _cfg(name, n=30, steps=8, zone_specs=None, drop_levers=False, **ov):
    dot = ["run.seed=42", f"run.n_agents={n}", f"run.n_steps={steps}",
           f"run.name={name}", "model.backend=mock", "observer.snapshot_every=144"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    cfg = load_config(dot)
    if zone_specs is not None:
        OmegaConf.update(cfg, "physics.zones_enabled", True, force_add=True)
        cfg.physics.zones = list(zone_specs)
    if drop_levers:                       # 鍵が**存在しない**conf(= 第155 以前の conf)
        cfg.pop("physics_levers", None)
    return cfg


def _sim(tmp_path, name, **kw):
    return Simulation(_cfg(name, **kw), out_dir=tmp_path / name)


def _run(tmp_path, name, **kw):
    sim = _sim(tmp_path, name, **kw)
    sim.run()
    return sim


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _kind(sim, kind):
    return [e for e in sim.logger.events if e.kind == kind]


#: 物理 ON の共通シナリオ(ゾーンを実際に誰かが通る最小構成)。
_ON = dict(n=30, steps=8, zone_specs=[_zone()])
_ON_SIG = dict(n=30, steps=8,
               zone_specs=[_zone(signal=dict(_SIGNAL), gate=dict(_GATE_TIGHT))])
#: 所有が**必ず step 境界を跨ぐ**条件(resume 検収用。tests/test_physics_zones.py と同じ手)。
#: mock の小規模ランでは既定のままだと 1 step でゾーンを抜けてしまい検収が空回りする。
_ON_HOLD = dict(n=30, steps=12, zone_specs=[_zone(max_sub_steps=400)])


# =========================================================================== #
# (0) conf の正準化・レジストリ宣言
# =========================================================================== #
_DEFAULT_LEVERS = {"ownership_mode": "", "ownership_max_dist_m": 0.0,
                   "min_gap_every": 1, "_mode": ""}


def test_lever_defaults_are_the_current_world(tmp_path):
    sim = _sim(tmp_path, "lv_def", **_ON)
    assert P._levers(sim) == _DEFAULT_LEVERS
    assert P._own_mode(sim) == ""
    # 既定は「分岐を 1 度も通らない」= 有界化そのものが組み立てられない
    assert P._own_bound(sim, ((0.0, 0.0),)) == (None, None)


def test_missing_block_falls_back_to_the_same_defaults(tmp_path):
    """鍵が conf に**存在しない**ラン(第155 以前の config.yaml)でも既定へ落ちる。"""
    sim = Simulation(_cfg("lv_drop", **_ON, drop_levers=True),
                     out_dir=tmp_path / "lv_drop")
    assert P._levers(sim) == _DEFAULT_LEVERS


def test_distance_key_alone_still_means_euclid(tmp_path):
    """`ownership_mode` を書かずに距離だけ書いた conf は euclid(後方互換)。"""
    sim = _sim(tmp_path, "lv_compat", **_ON,
               **{"physics_levers.ownership_max_dist_m": 40.0})
    assert P._own_mode(sim) == "euclid"
    assert P._own_bound(sim, ((0.0, 0.0),))[0] == 1600.0


def test_bad_lever_values_are_rejected(tmp_path):
    import pytest

    bad = (("physics_levers.ownership_max_dist_m", -1.0),
           ("physics_levers.min_gap_every", 0),
           ("physics_levers.ownership_mode", "nearest"),
           # euclid と宣言したのにしきいが無い = 黙って無制限にしない
           ("physics_levers.ownership_mode", "euclid"))
    for i, (key, val) in enumerate(bad):
        sim = _sim(tmp_path, f"lv_bad{i}", **_ON, **{key: val})
        with pytest.raises(ValueError):
            P._levers(sim)


def test_registry_declares_every_lever():
    ids = {f.id: f for f in R.FEATURES}
    for key, off in (("physics_levers.ownership_mode", ""),
                     ("physics_levers.ownership_max_dist_m", 0.0),
                     ("physics_levers.min_gap_every", 1)):
        assert key in ids, f"レジストリ未宣言: {key}"
        f = ids[key]
        assert f.repro_tier == "strict" and f.affects_k is False
        assert f.fingerprint_risk == "none"
        assert f.off_value == off, f"{key}: off 値が現行既定と違う"


def test_registered_ids_exist_in_the_shipped_config():
    cfg = OmegaConf.to_container(load_config(), resolve=True)
    for key in ("physics_levers.ownership_mode",
                "physics_levers.ownership_max_dist_m",
                "physics_levers.min_gap_every"):
        assert R._select(cfg, key) is not None, f"conf に無い id を宣言している: {key}"


# =========================================================================== #
# (1) A3 — `engine.radius` は列内包の結果と同一で、以後 1 度も書き換わらない
# =========================================================================== #
def _members(n, seed=0):
    rng = np.random.default_rng(seed)
    return [{"radius": float(0.25 + 0.1 * rng.random())} for _ in range(n)]


def test_engine_radius_equals_the_comprehension_and_never_mutates():
    """両エンジンとも構築時に自分のコピーを持ち、step を回しても書き換えない。"""
    recs = _members(40, seed=3)
    want = np.array([r["radius"] for r in recs], dtype=np.float64)
    n = len(recs)
    pos = np.stack([np.arange(n) * 1.5, np.zeros(n)], axis=1).astype(np.float64)
    vel = np.zeros((n, 2))
    goal = pos + np.array([50.0, 0.0])
    v0 = np.full(n, 1.2)
    for eng in (_orca.OrcaCrowd(pos, vel, goal, v0, want.copy(), walls=(),
                                neighbor_cap=12, tau=2.0, tau_obst=2.0,
                                neighbor_dist=10.0, wall_range=2.0,
                                v_max_factor=1.3, arrive_radius=1.0,
                                pref_noise=0.0, rng=None, radius_margin=0.05,
                                separation_iters=16),
                _sfm.Crowd(pos, vel, goal, v0, radius=want.copy(), rng=None,
                           noise=0.0, arrive_radius=1.0, walls=None,
                           wall_range=2.0, neighbor_cap=12, v_max_factor=1.3)):
        assert np.array_equal(eng.radius, want), type(eng).__name__
        for _ in range(20):
            eng.step(0.05)
        assert np.array_equal(eng.radius, want), \
            f"{type(eng).__name__} が radius を書き換えた(A3 の前提が崩れた)"


def test_accumulate_reads_the_engine_radius_in_member_order(tmp_path, monkeypatch):
    """実ランの全 `_accumulate` 呼び出しで `engine.radius` == 列内包 であること。"""
    sim = _sim(tmp_path, "a3_live", **_ON)
    orig = P._accumulate
    seen = {"n": 0}

    def spy(zone, members, engine, prev_pos, prev_vel, dt, gate_xy, cont, st,
            pcfg, dt_acc=None):
        want = np.array([r["radius"] for r in members], dtype=np.float64)
        assert np.array_equal(engine.radius, want), "members と engine がずれた"
        seen["n"] += 1
        return orig(zone, members, engine, prev_pos, prev_vel, dt, gate_xy, cont,
                    st, pcfg, dt_acc)

    monkeypatch.setattr(P, "_accumulate", spy)
    sim.run()
    assert seen["n"] > 0, "テスト前提が崩れた(1 サブステップも積分していない)"


# =========================================================================== #
# (2) A4 — 信号ゲートの前倒しは**呼び出し順序の変更だけ**(出力バイト同一)
# =========================================================================== #
def _legacy_admit_order(monkeypatch):
    """A4 **以前**の呼び出し順序を復元する(前倒しゲートだけ素通しにする)。

    前倒しゲートと `_admit` 内の判定は同じ `signal.can_cross` を呼ぶので、
    「`_admit` の中で呼ばれたときだけ本物を返す」ようにすれば

        if waiting:                       # ← 前倒しゲートが常に True = 無いのと同じ
            _writeback(...)               # ← 赤でも走る = 旧実装
            if _admit(...):               # ← ここは本物の信号判定

    という旧コードと**厳密に同じ**分岐になる。装置層(devices)は既定 OFF なので
    `SignalGate.can_cross` の呼び手は物理層しか居ない。
    """
    real_cc = _sfm.SignalGate.can_cross
    real_admit = P._admit
    flag = {"in_admit": False}

    def cc(self, sim_sec):
        return real_cc(self, sim_sec) if flag["in_admit"] else True

    def admit(*a, **kw):
        flag["in_admit"] = True
        try:
            return real_admit(*a, **kw)
        finally:
            flag["in_admit"] = False

    monkeypatch.setattr(_sfm.SignalGate, "can_cross", cc)
    monkeypatch.setattr(P, "_admit", admit)


def _count_calls(monkeypatch):
    """`_writeback` / `_admit` の呼び出し回数を数える(A4 が実際に効いたかの計器)。"""
    box = {"writeback": 0, "admit": 0}
    wb, ad = P._writeback, P._admit

    def wb_spy(members, engine):
        box["writeback"] += 1
        return wb(members, engine)

    def ad_spy(*a, **kw):
        box["admit"] += 1
        return ad(*a, **kw)

    monkeypatch.setattr(P, "_writeback", wb_spy)
    monkeypatch.setattr(P, "_admit", ad_spy)
    return box


def test_red_admit_is_a_pure_no_op():
    """赤の `_admit` は False を返し、待機列も在場者も st も 1 バイト変えない。"""
    sig = _sfm.SignalGate(cycle_s=20.0, green_s=4.0, flash_s=0.0, offset_s=0.0)
    assert not sig.can_cross(10.0), "テスト前提が崩れた(10 s は赤のはず)"
    zone = zones.build_cfg({"zones_enabled": True, "zones": [_zone()]},
                           REPO)["zones"][0]
    waiting = [{"pos": (0.0, 0.0), "radius": 0.3, "waiting": True}]
    members: list = []
    st = P._new_state()
    before = (copy.deepcopy(waiting), copy.deepcopy(members), copy.deepcopy(st))
    got = P._admit(None, zone, waiting, members, sig, 10.0, 10.0, 0, 0, st)
    assert got is False
    assert (waiting, members, st) == before


def test_signal_pre_gate_is_byte_identical_to_the_legacy_order(tmp_path,
                                                               monkeypatch):
    """A4 あり / A4 以前の順序 で L1・continuity・scalars が完全一致する。"""
    legacy = _sim(tmp_path, "a4_legacy", **_ON_SIG)
    _legacy_admit_order(monkeypatch)
    n_legacy = _count_calls(monkeypatch)
    legacy.run()
    got_legacy = (_l1(legacy), P.continuity(legacy), P.scalars(legacy))
    monkeypatch.undo()

    now = _sim(tmp_path, "a4_now", **_ON_SIG)
    n_now = _count_calls(monkeypatch)
    now.run()
    assert _kind(now, "zone_gate"), "テスト前提が崩れた(ゾーンを誰も通らない)"
    assert _l1(now) == got_legacy[0], "A4 が L1 を変えた"
    assert P.continuity(now) == got_legacy[1], "A4 が continuity を変えた"
    assert P.scalars(now) == got_legacy[2], "A4 が L2 スカラーを変えた"
    # 効いていること: 赤のサブステップでは `_admit` を 1 度も呼ばない
    # (`_writeback` は在場者ゼロのサブステップでは元々呼ばれないので `<=`)。
    assert n_now["admit"] < n_legacy["admit"], \
        f"A4 が効いていない(_admit {n_now['admit']} vs {n_legacy['admit']})"
    assert n_now["writeback"] <= n_legacy["writeback"]


# =========================================================================== #
# (3) A7 — min_gap の間引き(既定 1 = 1 バイト同一 / n>1 は上界)
# =========================================================================== #
def test_min_gap_every_default_leaves_no_trace(tmp_path):
    """既定 1 では `_phys_state` に鍵を 1 つも生やさない = checkpoint blob も同形。"""
    plain = _run(tmp_path, "mg_plain", **_ON)
    explicit = _run(tmp_path, "mg_expl", **_ON,
                    **{"physics_levers.min_gap_every": 1,
                       "physics_levers.ownership_max_dist_m": 0.0})
    st = P.state_of(plain)
    assert "min_gap_every" not in st and "min_gap_i" not in st
    assert sorted(st) == sorted(P._new_state()), "既定で状態の形が変わった"
    assert _l1(plain) == _l1(explicit)
    assert P.continuity(plain) == P.continuity(explicit)


def test_min_gap_every_thins_only_the_diagnostic(tmp_path):
    """n>1 は L1 も力学も 1 バイト変えず、`min_gap_m` だけが**上界**へ緩む。"""
    off = _run(tmp_path, "mg_off", **_ON)
    on = _run(tmp_path, "mg_on", **_ON, **{"physics_levers.min_gap_every": 5})
    assert _l1(off) == _l1(on), "A7 が L1 を変えた(診断専用のはず)"
    assert P.scalars(off) == P.scalars(on)
    a, b = P.continuity(off), P.continuity(on)
    thin = b.pop("min_gap_m")
    full = a.pop("min_gap_m")
    assert a == b, "min_gap_m 以外の continuity が動いた"
    if full is not None:
        assert thin is not None and thin >= full, (full, thin)
    # 間引きが実際に起きた(カウンタが立っている)
    assert P.state_of(on).get("min_gap_i", 0) > 0


def test_min_gap_every_survives_resume(tmp_path):
    """ON でも resume==straight(間引きカウンタは `_phys_state` で搬送される)。"""
    kw = dict(_ON_HOLD, steps=12, **{"physics_levers.min_gap_every": 3})
    straight, resumed, pairs = _resume_pair(tmp_path, "mg_res", kw)
    for sa, sb, stem in pairs:
        assert sa == sb, f"{stem} 不一致(A7 ON の resume)"
    # 間引きカウンタは `_phys_state` に載る = 診断値も straight と完全一致する
    assert P.continuity(resumed) == P.continuity(straight)
    assert P.state_of(resumed)["min_gap_i"] == P.state_of(straight)["min_gap_i"]


# =========================================================================== #
# (4) C1 — 所有の距離有界化
# =========================================================================== #
def test_near_gates_is_the_plain_euclidean_test():
    gates = ((0.0, 0.0), (100.0, 0.0))
    d2, box = 50.0 ** 2, (-50.0, 150.0, -50.0, 50.0)
    assert P._near_gates(0.0, 49.9, gates, d2, box) is True
    assert P._near_gates(0.0, 50.0, gates, d2, box) is True     # 境界は圏内(<=)
    assert P._near_gates(0.0, 50.1, gates, d2, box) is False
    assert P._near_gates(100.0, 10.0, gates, d2, box) is True   # 最寄りゲートで判定
    assert P._near_gates(50.0, 0.0, gates, d2, box) is True      # ちょうど 50 m = 圏内
    assert P._near_gates(50.0, 5.0, gates, d2, box) is False      # どちらからも 50 m 超
    assert P._near_gates(1e6, 0.0, gates, d2, box) is False     # 外接矩形で即落ち


def test_own_bound_box_is_the_gate_hull_plus_x(tmp_path):
    sim = _sim(tmp_path, "c1_box", **_ON, **{"physics_levers.ownership_max_dist_m": 40.0})
    d2, box = P._own_bound(sim, ((-5.0, 2.0), (7.0, -3.0)))
    assert d2 == 1600.0
    assert box == (-45.0, 47.0, -43.0, 42.0)
    # ゲート 0 件のゾーンは有界化しない(route_span も原理的に None しか返さない)
    assert P._own_bound(sim, ()) == (None, None)


def test_ownership_default_is_byte_identical(tmp_path):
    """既定 0.0(= 無制限)は鍵を書かないランと L1 バイト一致。"""
    plain = Simulation(_cfg("c1_drop", **_ON, drop_levers=True),
                       out_dir=tmp_path / "c1_drop")
    plain.run()
    explicit = _run(tmp_path, "c1_expl", **_ON,
                    **{"physics_levers.ownership_max_dist_m": 0.0})
    assert _kind(plain, "zone_gate"), "テスト前提が崩れた(ゾーンを誰も通らない)"
    assert _l1(plain) == _l1(explicit)
    assert P.continuity(plain) == P.continuity(explicit)
    assert P.scalars(plain) == P.scalars(explicit)


def test_ownership_bound_shrinks_the_owned_set_and_keeps_everyone_moving(tmp_path):
    """X を絞ると所有が減り、圏外の個体は**従来どおりグラフ移動**で進む。"""
    wide = _run(tmp_path, "c1_wide", **_ON)
    tight = _run(tmp_path, "c1_tight", **_ON,
                 **{"physics_levers.ownership_max_dist_m": 1.0})
    n_wide = len([e for e in _kind(wide, "zone_gate")
                  if e.payload.get("dir") == "enter"])
    n_tight = len([e for e in _kind(tight, "zone_gate")
                   if e.payload.get("dir") == "enter"])
    assert n_wide > 0
    assert n_tight < n_wide, (n_wide, n_tight)
    # 圏外の個体は従来どおりグラフ移動で進む(所有されないぶん move が減らない)
    assert len(_kind(tight, "move_segment")) >= len(_kind(wide, "move_segment"))
    # 所有はこのゾーンのものしか無い(所有権が壊れていない)
    for a in tight.agents:
        z = getattr(a, "_phys_zone", None)
        assert z is None or z == "z1"


def test_ownership_bound_is_deterministic_and_adds_no_stream(tmp_path):
    """同 seed 2 ランが完全一致し、**新しい named stream を 1 本も足さない**。

    ★ON では所有が減るぶん `"physics"` stream の**取り出し回数**は減る(ゾーンが空の
      step では取り出し自体が起きない)。RngHub はステートレスで stream はキーの純関数
      なので、取り出さないことが他の stream の draw 順を動かすことは無い。ここで固定
      するのは「stream の**名前の集合**が増えない」ことである。
    """
    class _CountingHub:
        def __init__(self, inner):
            self._inner = inner
            self.master_seed = inner.master_seed
            self.names: set = set()
            self.n_phys = 0

        def stream(self, *key):
            self.names.add(key[0] if key else None)
            if key and key[0] == P.STREAM:
                self.n_phys += 1
            return self._inner.stream(*key)

        def key_name(self, *key):
            return self._inner.key_name(*key)

    def run(name, **ov):
        sim = _sim(tmp_path, name, **_ON, **ov)
        sim.hub = _CountingHub(sim.hub)
        sim.run()
        return sim

    a = run("c1_det_a", **{"physics_levers.ownership_max_dist_m": 12.0})
    b = run("c1_det_b", **{"physics_levers.ownership_max_dist_m": 12.0})
    off = run("c1_det_off")
    assert _l1(a) == _l1(b), "同 seed 2 ランが一致しない"
    assert P.continuity(a) == P.continuity(b)
    assert a.hub.n_phys == b.hub.n_phys
    assert a.hub.names <= off.hub.names, "新しい named stream が生えた"
    assert a.hub.n_phys <= off.hub.n_phys


def test_ownership_bound_adds_no_carried_state(tmp_path):
    """新しい状態を 1 つも足さない(`_phys_state` の形も `_phys_*` 属性の集合も同じ)。"""
    off = _run(tmp_path, "c1_state_off", **_ON)
    on = _run(tmp_path, "c1_state_on", **_ON,
              **{"physics_levers.ownership_max_dist_m": 12.0})
    assert sorted(P.state_of(on)) == sorted(P.state_of(off))
    fields = {f"_phys_{f}" for f in P._FIELDS} | {"_phys_zone"}
    for sim in (off, on):
        for a in sim.agents:
            extra = {k for k in vars(a) if k.startswith("_phys_")} - fields
            assert extra <= {"_phys_body", "_phys_used_step", "_phys_used_s"}, extra


def _resume_pair(tmp_path, name, kw):
    """straight ラン / ゾーン滞在中で分割した resume ランを回して `(straight, resumed,
    [(L1a, L1b, 名前), …])` を返す。

    分割点の採り方も比較の仕方も tests/test_physics_zones.py の resume 検収と同じ形。
    """
    total = int(kw["steps"])
    sdir = tmp_path / (name + "_straight")
    straight = Simulation(_cfg(name + "_straight", **kw), out_dir=sdir)
    straight.run()
    assert _kind(straight, "zone_gate"), "テスト前提が崩れた(ゾーンを誰も通らない)"
    d = tmp_path / (name + "_resumed")
    sim1 = Simulation(_cfg(name + "_resumed", **kw), out_dir=d)
    split, owned = 0, []
    for step in range(total - 1):
        scheduler.run_step(sim1, step)
        split = step + 1
        owned = [a.id for a in sim1.agents if P.owned(a)]
        if split >= 3 and owned:
            break
    assert owned, "分割点でゾーン滞在中の個体が居ない(検収の空回り)"
    checkpoint.save(sim1, split, d / "checkpoint" / f"ckpt-{split:06d}.pkl.gz")
    sim1.logger.flush_segment()
    sim2 = Simulation(_cfg(name + "_resumed", **kw), out_dir=d)
    sim2.run(resume_from=d)
    out = []
    for stem in ("l1_events", "l2_metrics", "l3_snapshots"):
        out.append((pq.read_table(sdir / f"{stem}.parquet").to_pylist(),
                    pq.read_table(d / f"{stem}.parquet").to_pylist(), stem))
    return straight, sim2, out


def test_ownership_bound_resume_equals_straight(tmp_path):
    kw = dict(_ON_HOLD, steps=12,
              **{"physics_levers.ownership_max_dist_m": 12.0})
    straight, resumed, pairs = _resume_pair(tmp_path, "c1_res", kw)
    for sa, sb, stem in pairs:
        assert sa == sb, f"{stem} 不一致(C1 ON の resume)"
    assert P.continuity(resumed) == P.continuity(straight)


# =========================================================================== #
# (5) C1 route_arrival — 経路到達版(捕捉率を落とさない有界化)
# =========================================================================== #
_RA = {"physics_levers.ownership_mode": "route_arrival"}


def test_route_arrival_captures_essentially_every_crossing(tmp_path):
    """euclid は取りこぼすが route_arrival は「この step で届く個体」を全部拾う。"""
    off = _run(tmp_path, "ra_off", **_ON)
    euc = _run(tmp_path, "ra_euclid", **_ON,
               **{"physics_levers.ownership_mode": "euclid",
                  "physics_levers.ownership_max_dist_m": 50.0})
    ra = _run(tmp_path, "ra_on", **_ON, **_RA)

    def enters(sim):
        return len([e for e in _kind(sim, "zone_gate")
                    if e.payload.get("dir") == "enter"])

    n_off, n_euc, n_ra = enters(off), enters(euc), enters(ra)
    assert n_off > 0
    assert n_euc < n_off, (n_euc, n_off)          # 素朴版は取りこぼす(第155 実測の再現)
    assert n_ra >= 0.8 * n_off, (n_ra, n_off)     # 経路到達版はほぼ全部拾う
    assert n_ra > n_euc, (n_ra, n_euc)


def test_route_arrival_enters_at_the_gate_node(tmp_path):
    """入場位置は**ゲートノードの座標そのもの**(手前区間は物理で積分しない)。"""
    sim = _run(tmp_path, "ra_gate", **_ON, **_RA)
    zone = sim.physcfg["zones"][0]
    gates = {(round(x, 6), round(y, 6))
             for x, y in (sim.city.node_xy(n) for n in P._gate_nodes(sim, zone))}
    at_gate = already_in = 0
    for e in _kind(sim, "zone_gate"):
        if e.payload.get("dir") != "enter":
            continue
        if (round(e.x, 6), round(e.y, 6)) in gates:
            at_gate += 1
        else:
            # 唯一の例外: 現在ノードが**既にゾーン内**の個体(手前区間が無いので
            # 従来どおり現在座標から所有する。`_gate_arrival` の gate_i=-1 の枝)。
            already_in += 1
            assert zone.contains(e.x, e.y), (e.x, e.y)
    assert at_gate > 0, "テスト前提が崩れた(ゲート入場が 1 件も無い)"
    assert at_gate > already_in, (at_gate, already_in)


def test_route_arrival_never_admits_before_the_arrival_time(tmp_path, monkeypatch):
    """`_admit` は到着時刻より前の個体を**1 度も見ない**(pending で構造的に保証)。"""
    sim = _sim(tmp_path, "ra_time", **_ON, **_RA)
    real = P._admit
    seen = {"calls": 0, "queued": 0}

    def spy(s, zone, waiting, members, signal, sim_sec, t_in_step, *a, **kw):
        seen["calls"] += 1
        for rec in waiting:
            seen["queued"] += 1
            assert float(rec.get("arrive_s", 0.0)) <= t_in_step + 1e-12, \
                (rec.get("arrive_s"), t_in_step)
        return real(s, zone, waiting, members, signal, sim_sec, t_in_step, *a, **kw)

    monkeypatch.setattr(P, "_admit", spy)
    sim.run()
    assert seen["calls"] > 0 and seen["queued"] > 0


def test_route_arrival_keeps_dwell_to_the_zone_crossing(tmp_path):
    """`dwell_s` は**ゾーン滞在**だけ(ゲートまでの徒歩は含めない)。

    無制限モードでは所有が数百 m 手前から始まるので dwell は step 秒に張り付くが、
    route_arrival ではゾーンを横切る実時間になる。
    """
    off = _run(tmp_path, "ra_dwell_off", **_ON)
    ra = _run(tmp_path, "ra_dwell_on", **_ON, **_RA)

    def dwells(sim):
        return [float(e.payload["dwell_s"]) for e in _kind(sim, "zone_gate")
                if e.payload.get("dir") == "exit"]

    d_off, d_ra = dwells(off), dwells(ra)
    assert d_off and d_ra
    step_s = float(off.clock.step_seconds)
    assert max(d_ra) < 0.5 * step_s, max(d_ra)
    assert sum(d_ra) / len(d_ra) < sum(d_off) / len(d_off)


def test_route_arrival_charges_the_walk_to_the_move_budget(tmp_path):
    """ゲートまでの徒歩は `_phys_used_s`(= `_phase_move` の予算)へ課金される。

    `budget_scale` は [0,1] に収まり、`used_s >= dwell 相当`(= 積分秒 + 徒歩秒)である
    こと = 同じ時間を 2 度歩かない(二重移動の防止)。
    """
    sim = _sim(tmp_path, "ra_budget", **_ON, **_RA)
    step_s = float(sim.clock.step_seconds)
    seen = {"n": 0, "walk": 0}
    real = P._release

    def spy(s, zone, agent, step, sim_min, st, **kw):
        out = real(s, zone, agent, step, sim_min, st, **kw)
        used = float(getattr(agent, "_phys_used_s", 0.0))
        assert 0.0 <= used <= step_s + 1e-9, used
        assert 0.0 <= P.budget_scale(s, agent, step) <= 1.0
        if kw.get("rec") is not None:
            assert used + 1e-9 >= float(kw.get("elapsed_s", 0.0) or 0.0) - step_s
        seen["n"] += 1
        if used > 0.0:
            seen["walk"] += 1
        return out

    P._release = spy
    try:
        sim.run()
    finally:
        P._release = real
    assert seen["n"] > 0 and seen["walk"] > 0


def test_route_arrival_adds_no_persisted_state(tmp_path):
    """到着秒は **step ローカル**で agent へ焼かない = checkpoint の形が変わらない。"""
    off = _run(tmp_path, "ra_state_off", **_ON)
    ra = _run(tmp_path, "ra_state_on", **_ON, **_RA)
    assert sorted(P.state_of(ra)) == sorted(P.state_of(off))
    fields = {f"_phys_{f}" for f in P._FIELDS} | {"_phys_zone"}
    for a in ra.agents:
        assert not hasattr(a, "_phys_arrive_s"), "到着秒が agent へ焼かれている"
        extra = {k for k in vars(a) if k.startswith("_phys_")} - fields
        assert extra <= {"_phys_body", "_phys_used_step", "_phys_used_s"}, extra


def test_route_arrival_is_deterministic_and_adds_no_stream(tmp_path):
    class _CountingHub:
        def __init__(self, inner):
            self._inner = inner
            self.master_seed = inner.master_seed
            self.names: set = set()

        def stream(self, *key):
            self.names.add(key[0] if key else None)
            return self._inner.stream(*key)

        def key_name(self, *key):
            return self._inner.key_name(*key)

    def run(name, **ov):
        sim = _sim(tmp_path, name, **_ON, **ov)
        sim.hub = _CountingHub(sim.hub)
        sim.run()
        return sim

    a = run("ra_det_a", **_RA)
    b = run("ra_det_b", **_RA)
    off = run("ra_det_off")
    assert _l1(a) == _l1(b), "同 seed 2 ランが一致しない"
    assert P.continuity(a) == P.continuity(b)
    assert a.hub.names <= off.hub.names, "新しい named stream が生えた"
    assert _l1(a) != _l1(off), "ON なのに世界が変わっていない"


def test_route_arrival_resume_equals_straight(tmp_path):
    kw = dict(_ON_HOLD, steps=12, **_RA)
    straight, resumed, pairs = _resume_pair(tmp_path, "ra_res", kw)
    for sa, sb, stem in pairs:
        assert sa == sb, f"{stem} 不一致(route_arrival の resume)"
    assert P.continuity(resumed) == P.continuity(straight)
    assert P.scalars(resumed) == P.scalars(straight)


def test_route_arrival_respects_the_integration_horizon(tmp_path):
    """`max_sub_steps` で積分時間が step 秒に足りないときは、その秒数までしか拾わない。

    (拾ってしまうと到着時刻が積分区間の外に落ちて、待機列へ入れないまま step が終わる)
    """
    sim = _run(tmp_path, "ra_horizon", **_ON_HOLD, **_RA)
    zone = sim.physcfg["zones"][0]
    horizon = min(int(zone.max_sub_steps),
                  max(1, round(float(sim.clock.step_seconds) / zone.dt_sub))) * zone.dt_sub
    for e in _kind(sim, "zone_gate"):
        if e.payload.get("dir") == "enter":
            assert float(e.payload["wait_s"]) <= horizon + float(
                sim.clock.step_seconds) * int(e.payload["waited_steps"]) + 1e-9


# =========================================================================== #
# (6) 全レバー既定 = 物理 ON のランが第155 以前と 1 バイト同一
# =========================================================================== #
def test_all_defaults_keep_a_physics_on_run_byte_identical(tmp_path):
    plain = Simulation(_cfg("all_drop", **_ON_SIG, drop_levers=True),
                       out_dir=tmp_path / "all_drop")
    plain.run()
    explicit = _run(tmp_path, "all_expl", **_ON_SIG,
                    **{"physics_levers.ownership_mode": "''",
                       "physics_levers.ownership_max_dist_m": 0.0,
                       "physics_levers.min_gap_every": 1})
    assert _kind(plain, "zone_gate"), "テスト前提が崩れた(ゾーンを誰も通らない)"
    assert _l1(plain) == _l1(explicit)
    assert P.continuity(plain) == P.continuity(explicit)
    assert P.scalars(plain) == P.scalars(explicit)
