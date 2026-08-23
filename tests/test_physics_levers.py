"""物理 2 レバー(第154)— 密度適応 dt(B)と近傍 cap の引き下げ(D)のテスト。

正典: src/society/physics.py の「物理 2 レバー」節 / conf/config.yaml `physics.adaptive_dt`
      + `physics.neighbor_cap` / docs/research/crowd-attention-physics.md(Ballerini 位相的近傍)

R1 の鉄則:
- **既定 OFF は 1 バイト同一**。`adaptive_dt.enabled=false` かつ `neighbor_cap=0` のとき、
  物理 ON のランは新キーを書かないランと L1 バイト一致・continuity 完全一致であること。
- ON は世界を変える(opt-in)。変えてよいのは軌跡だけで、**積分時間の総量は保存する**
  (Σ dt_eff = n_sub × dt_sub。端数は最終塊で吸収 = 1 秒も失わない)。
- 決定論: 係数は在場者数(決定論量)の純関数。同 seed 2 ラン一致・乱数消費ゼロ本。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from society import physics as P
from society import registry as R
from society.config import load_config
from society.engine import scheduler
from society.engine.simulation import Simulation
from society.world import zones

REPO = Path(__file__).resolve().parents[1]

_ZONE_R = 25.0
_POLY = [[-_ZONE_R, -_ZONE_R], [_ZONE_R, -_ZONE_R], [_ZONE_R, _ZONE_R], [-_ZONE_R, _ZONE_R]]


def _zone(zid="z1", engine="orca", **ov):
    z = {"id": zid, "engine": engine, "dt_sub": 0.05, "polygon": list(_POLY)}
    z.update(ov)
    return z


def _cfg(name, n=30, steps=8, zone_specs=None, **ov):
    dot = ["run.seed=42", f"run.n_agents={n}", f"run.n_steps={steps}",
           f"run.name={name}", "model.backend=mock", "observer.snapshot_every=144"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    cfg = load_config(dot)
    if zone_specs is not None:
        OmegaConf.update(cfg, "physics.zones_enabled", True, force_add=True)
        cfg.physics.zones = list(zone_specs)
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


# =========================================================================== #
# (1) conf の正準化とバリデーション
# =========================================================================== #
def test_defaults_are_off_and_identical_to_the_current_world():
    cfg = zones.build_cfg({}, REPO)
    assert cfg["adaptive_dt"]["enabled"] is False
    assert cfg["neighbor_cap"] == 0
    assert cfg["cognitive"]["neighbors"] == 12, "既定で cog の k が動いた"
    # 既定ではゾーン宣言の cap がそのまま(= 現行と 1 バイト同一)
    z = zones.build_cfg({"zones_enabled": True, "zones": [_zone()]}, REPO)["zones"][0]
    assert z.neighbor_cap == 12


def test_thresholds_are_canonicalised_to_ascending_order():
    """記入順に依らない決定論(N 昇順へ正準化する)。"""
    cfg = zones.build_cfg({"adaptive_dt": {"enabled": True,
                                           "thresholds": [[2000, 4.0], [500, 2.0]]}},
                          REPO)
    assert cfg["adaptive_dt"]["thresholds"] == ((500, 2.0), (2000, 4.0))


def test_bad_declarations_are_rejected():
    for bad in ([[500, 0.5]],            # 係数 < 1 = dt を細かくする = 目的に反する
                [[-1, 2.0]],             # 負の人数
                [[500]],                 # 形が違う
                [[500, 2.0, 1]]):
        try:
            zones.build_cfg({"adaptive_dt": {"enabled": True, "thresholds": bad}}, REPO)
            raise AssertionError(f"不正な thresholds {bad} が素通りした")
        except ValueError:
            pass
    try:
        zones.build_cfg({"neighbor_cap": -1}, REPO)
        raise AssertionError("負の neighbor_cap が素通りした")
    except ValueError:
        pass
    try:
        zones.build_cfg({"adaptive_dt": {"nope": 1}}, REPO)
        raise AssertionError("未知の adaptive_dt キーが素通りした")
    except KeyError:
        pass


def test_neighbor_cap_is_a_single_point_of_truncation():
    """cap は 1 点で絞る: 全ゾーンの neighbor_cap と 認知的近傍の k の**両方**。

    認知的近傍 ON では `neighbor_cap` が 1 度も読まれない(sfm_core._repulsion_cognitive /
    orca_core.step はどちらも cog_neighbors を使う)ので、揃えないと
    「cap を下げたのに何も起きない」= 二重 cap の罠になる。
    """
    cfg = zones.build_cfg({"zones_enabled": True, "neighbor_cap": 7,
                           "cognitive": {"enabled": True},
                           "zones": [_zone(), _zone("z2", engine="sfm",
                                                    polygon=[[200, 200], [240, 200],
                                                             [240, 240], [200, 240]])]},
                          REPO)
    assert [z.neighbor_cap for z in cfg["zones"]] == [7, 7]
    assert cfg["cognitive"]["neighbors"] == 7
    # 既に k の方が小さいときは**絞り込みの向きだけ**(min)= 緩めない
    tight = zones.build_cfg({"neighbor_cap": 7,
                             "cognitive": {"enabled": True, "neighbors": 5}}, REPO)
    assert tight["cognitive"]["neighbors"] == 5
    # 認知的近傍 OFF のときは cog 側に触らない(死んだ値を書き換えない)
    off = zones.build_cfg({"neighbor_cap": 7, "cognitive": {"enabled": False}}, REPO)
    assert off["cognitive"]["neighbors"] == 12
    # ゾーン宣言より conf の cap が優先(全ゾーン 1 点)
    ov = zones.build_cfg({"zones_enabled": True, "neighbor_cap": 6,
                          "zones": [_zone(neighbor_cap=20)]}, REPO)
    assert ov["zones"][0].neighbor_cap == 6


def test_separation_iters_defaults_to_the_zone_declaration():
    """既定 0 = ゾーン宣言(64)のまま = 現行と 1 バイト同一。"""
    cfg = zones.build_cfg({}, REPO)
    assert cfg["separation_iters"] == 0
    z = zones.build_cfg({"zones_enabled": True, "zones": [_zone()]}, REPO)["zones"][0]
    assert z.orca["separation_iters"] == zones.ZONE_DEFAULTS["orca"]["separation_iters"]
    assert z.orca["separation_iters"] == 64
    # 宣言側で明示した値も既定 0 では尊重される
    ov = zones.build_cfg({"zones_enabled": True,
                          "zones": [_zone(orca={"separation_iters": 8})]}, REPO)
    assert ov["zones"][0].orca["separation_iters"] == 8


def test_separation_iters_overrides_every_orca_zone():
    cfg = zones.build_cfg({"zones_enabled": True, "separation_iters": 24,
                           "zones": [_zone(),
                                     _zone("z2", orca={"separation_iters": 8},
                                           polygon=[[200, 200], [240, 200],
                                                    [240, 240], [200, 240]])]},
                          REPO)
    assert [z.orca["separation_iters"] for z in cfg["zones"]] == [24, 24]
    # 他の orca パラメータは 1 つも動かさない(上書きは 1 キーだけ)
    assert cfg["zones"][0].orca["separation_iters"] == 24
    for k in ("tau", "tau_obst", "neighbor_dist_m", "wall_range_m", "pref_noise",
              "radius_margin_m"):
        assert cfg["zones"][0].orca[k] == zones.ZONE_DEFAULTS["orca"][k]
    try:
        zones.build_cfg({"separation_iters": -1}, REPO)
        raise AssertionError("負の separation_iters が素通りした")
    except ValueError:
        pass


def test_separation_iters_reaches_the_engine_and_binds(tmp_path):
    """conf の 1 行がエンジンの反復上限まで届き、密集では実際に押し戻しが変わる。"""
    sim = _sim(tmp_path, "sep_end", **_ON, **{"physics.separation_iters": 24})
    assert all(z.orca["separation_iters"] == 24 for z in sim.physcfg["zones"])

    from society.world import orca_core

    def gap(sep):
        pos, vel, goal, v0, rad = _packed(n=36, pitch=0.45)
        c = orca_core.OrcaCrowd(pos.copy(), vel.copy(), goal, v0, rad,
                                neighbor_cap=12, pref_noise=0.0, rng=None,
                                separation_iters=sep)
        c.step(0.1)
        return c.min_gap(), c.last_sep_iters

    g64, i64 = gap(64)
    g8, i8 = gap(8)
    assert i64 > i8, (i64, i8)
    assert g64 > g8, "上限を削っても残留めり込みが 1 ビットも変わらない"


def test_cog_kwargs_carry_the_lowered_cap():
    class _S:
        physcfg = zones.build_cfg({"neighbor_cap": 7,
                                   "cognitive": {"enabled": True}}, REPO)
    assert P._cog_kwargs(_S())["cog_neighbors"] == 7
    class _Off:
        physcfg = zones.build_cfg({"neighbor_cap": 7}, REPO)
    assert P._cog_kwargs(_Off()) == {}, "cognitive OFF なのに引数が生えた"


# =========================================================================== #
# (2) 係数選択と会計(純関数)
# =========================================================================== #
def test_dt_factor_is_a_deterministic_step_function():
    th = ((500, 2.0), (2000, 4.0))
    assert P._dt_factor(th, 0) == 1.0
    assert P._dt_factor(th, 500) == 1.0, "境界は『超えたら』= 500 ちょうどは 1.0"
    assert P._dt_factor(th, 501) == 2.0
    assert P._dt_factor(th, 2000) == 2.0
    assert P._dt_factor(th, 2001) == 4.0
    assert P._dt_factor((), 10 ** 6) == 1.0, "閾値ゼロ件は常に係数 1"


def test_adaptive_of_is_none_unless_really_enabled():
    class _S:
        physcfg = {}
    assert P._adaptive_of(_S()) is None
    _S.physcfg = zones.build_cfg({}, REPO)
    assert P._adaptive_of(_S()) is None, "既定 OFF で有効化された"
    _S.physcfg = zones.build_cfg({"adaptive_dt": {"enabled": True,
                                                  "thresholds": []}}, REPO)
    assert P._adaptive_of(_S()) is None, "閾値ゼロ件は OFF と同じ扱い"
    _S.physcfg = zones.build_cfg({"adaptive_dt": {"enabled": True}}, REPO)
    th, every = P._adaptive_of(_S())
    assert th == ((500, 2.0), (2000, 4.0)) and every == 20


def test_engines_filter_selects_which_zones_get_the_coarse_dt():
    """`engines` を書いたゾーンだけが粗い dt を使う(他は dt_sub のまま = 同一)。

    分ける根拠はベンチ実測: 係数 2(dt 0.2)で SFM は重なり −0.171 m・壁貫通 −0.105 m を
    出すが、ORCA は重なりゼロを維持する(速度層 + 位置層の二層構成)。
    """
    cfg = zones.build_cfg({"zones_enabled": True,
                           "adaptive_dt": {"enabled": True, "engines": ["orca"]},
                           "zones": [_zone("z_orca", engine="orca"),
                                     _zone("z_sfm", engine="sfm",
                                           polygon=[[200, 200], [240, 200],
                                                    [240, 240], [200, 240]])]},
                          REPO)

    class _S:
        physcfg = cfg
    by_id = {z.id: z for z in cfg["zones"]}
    assert P._adaptive_of(_S(), by_id["z_orca"]) is not None
    assert P._adaptive_of(_S(), by_id["z_sfm"]) is None, "SFM ゾーンに粗い dt が掛かった"
    # 空(既定)= 全エンジン
    both = zones.build_cfg({"zones_enabled": True,
                            "adaptive_dt": {"enabled": True},
                            "zones": [_zone("z_sfm", engine="sfm")]}, REPO)

    class _B:
        physcfg = both
    assert P._adaptive_of(_B(), both["zones"][0]) is not None
    try:
        zones.build_cfg({"adaptive_dt": {"enabled": True, "engines": ["rvo3"]}}, REPO)
        raise AssertionError("未知 engine が素通りした")
    except ValueError:
        pass


def test_next_dt_never_emits_a_dust_substep():
    """浮動小数の塵で **dt≈0 のサブステップ**が生えない(実測された事故の回帰テスト)。

    0.2 を 100 回足すと 20.0 に 3.6e-15 だけ届かないので、素直に書くと最後に
    dt=3.6e-15 のサブステップが 1 回生える。ORCA はその 1 回で
    「時間 dt の遮断円へ射影する」枝の半径が 1/dt で発散し、速度が v_max を大きく
    超えた(ベンチ実測: 平均速さ 0.57 → **53 m/s**・accel 分位が最終ビンへ張り付き)。
    """
    total, dt = 20.0, 0.1
    for factor in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
        t, out = 0.0, []
        for _ in range(100000):
            de, t = P.next_dt(t, total, dt, factor)
            if de <= 0.0:
                break
            out.append(de)
            if t >= total:
                break
        assert abs(sum(out) - total) < 1e-9, (factor, sum(out))
        assert t == total, (factor, t)
        assert min(out) > dt * 1e-3, (factor, min(out))
        assert max(out) <= dt * factor * (1.0 + 1e-6) + 1e-12, (factor, max(out))


def test_used_s_is_bit_identical_when_dt_is_fixed_and_exact_when_not():
    """滞在秒の会計: OFF は従来の `step_n × dt`、ON は累積列の差分。"""
    assert P._used_s({"step_n": 4}, 0.05, None) == 4 * 0.05
    assert P._used_s({"step_n": 0}, 0.05, None) == 0.0
    # 累積列(dt が 0.1 / 0.2 / 0.3 / 0.4 と変わった 4 回ぶん)
    cum = [0.1, 0.30000000000000004, 0.6000000000000001, 1.0000000000000002]
    assert P._used_s({"step_n": 4}, 0.05, cum) == cum[-1]      # 頭から居た個体
    assert P._used_s({"step_n": 2}, 0.05, cum) == cum[3] - cum[1]   # 途中入場
    assert P._used_s({"step_n": 0}, 0.05, cum) == 0.0          # 1 度も積分していない


def test_reversal_rate_uses_measured_seconds_when_dt_varies():
    """適応 dt の下では反転率の分母を実 dt の積算(`sec`)から採る。"""
    class _S:
        pass
    sim = _S()
    st = P._new_state()
    c = st["cont"]
    c["dt_sub"] = 0.05
    c["interior"].update(n=10, samples=10, flip=2)
    out_fixed = P.continuity(_with(sim, st))["interior_reversal_rate"]
    assert abs(out_fixed - 2.0 / (10 * 0.05)) < 1e-12
    c["interior"]["sec"] = 4.0            # 実測 4 秒ぶんのサンプルだった
    out_adapt = P.continuity(_with(sim, st))["interior_reversal_rate"]
    assert abs(out_adapt - 2.0 / 4.0) < 1e-12


def _with(sim, st):
    sim._phys_state = st
    return sim


# =========================================================================== #
# (3) 既定 OFF = 1 バイト同一(物理 ON のランで新キーを書いても書かなくても同じ)
# =========================================================================== #
_ON = dict(n=30, steps=8, zone_specs=[_zone()])


def test_lever_defaults_do_not_change_a_physics_on_run(tmp_path):
    plain = _run(tmp_path, "lev_plain", **_ON)
    explicit = _run(tmp_path, "lev_explicit", **_ON,
                    **{"physics.adaptive_dt.enabled": "false",
                       "physics.neighbor_cap": 0,
                       "physics.separation_iters": 0})
    assert _kind(plain, "zone_gate"), "テスト前提が崩れた(ゾーンを誰も通らない)"
    assert _l1(plain) == _l1(explicit)
    assert P.continuity(plain) == P.continuity(explicit)


def test_describe_stays_empty_until_a_lever_is_on(tmp_path):
    off = _sim(tmp_path, "desc_off", **_ON)
    assert "adaptive_dt" not in P.calib_describe(off)
    assert "neighbor_cap" not in P.calib_describe(off)
    assert "separation_iters" not in P.calib_describe(off)
    on = _sim(tmp_path, "desc_on", **_ON,
              **{"physics.adaptive_dt.enabled": "true", "physics.neighbor_cap": 7,
                 "physics.separation_iters": 24})
    d = P.calib_describe(on)
    assert d["neighbor_cap"] == 7
    assert d["separation_iters"] == 24
    assert d["adaptive_dt"]["thresholds"] == [[500, 2.0], [2000, 4.0]]


# =========================================================================== #
# (4) ON: 積分時間の総量が保存される(会計の中心契約)
# =========================================================================== #
def _per_step_dt(tmp_path, name, monkeypatch, steps=10, **ov):
    """各 step の「積分した dt の列」を採る(ゾーンは 1 つ = step ごとに 1 本)。"""
    zspec = _zone(max_sub_steps=400)      # 20 秒ぶん = 所有が必ず step 境界を跨ぐ
    sim = _sim(tmp_path, name, n=30, steps=steps, zone_specs=[zspec], **ov)
    bucket: list[float] = []
    orig = P._accumulate

    def spy(zone, members, engine, prev_pos, prev_vel, dt, gate_xy, cont, st,
            pcfg, dt_acc=None):
        bucket.append(float(dt))
        return orig(zone, members, engine, prev_pos, prev_vel, dt, gate_xy, cont,
                    st, pcfg, dt_acc)
    monkeypatch.setattr(P, "_accumulate", spy)      # ★差し替えは 1 回だけ(入れ子にしない)
    out = []
    for step in range(steps):
        bucket.clear()
        scheduler.run_step(sim, step)
        out.append(list(bucket))
    return sim, out, 400 * 0.05


def test_fixed_dt_step_is_covered_exactly(tmp_path, monkeypatch):
    """OFF: 1 サブステップ = dt_sub 固定で、満員の step はちょうど総量ぶん積む。"""
    _, per_step, total_s = _per_step_dt(tmp_path, "acct_off", monkeypatch)
    assert any(per_step), "ゾーンを誰も通らない(検収の空回り)"
    for b in per_step:
        assert all(x == 0.05 for x in b)
        assert sum(b) <= total_s + 1e-12
    assert max(sum(b) for b in per_step) == total_s


def test_adaptive_dt_covers_the_same_total_and_absorbs_the_remainder(tmp_path,
                                                                     monkeypatch):
    """ON: 塊は dt×係数、**端数は最終塊で吸収**して総量は 1 秒も失われない。

    係数 3.0 は 400 サブステップ(20.0 s)を割り切らない(20/0.15 = 133.33…)ので、
    最終塊に端数が必ず残る = 端数処理そのものが検収対象になる。
    ★許容 1e-9: 会計の基準は `_run_zone` の**左結合の逐次和 t** で、これは最終塊で
      total_s へ厳密に着地する。一方この検収が使う `sum()` は CPython の補償総和
      (Neumaier)なので「数学的な真の和」を返し、両者は丸めの分だけ食い違う
      (実測 1.4e-14 s = 0.014 ピコ秒)。**失っている時間ではなく足し方の違い**である。
    """
    _, per_step, total_s = _per_step_dt(
        tmp_path, "acct_on", monkeypatch,
        **{"physics.adaptive_dt.enabled": "true",
           "physics.adaptive_dt.thresholds": "[[1,3.0]]",
           "physics.adaptive_dt.recheck_every": 20})
    full = [b for b in per_step if b and abs(sum(b) - total_s) < 1e-9]
    assert full, "step 全長を積んだ step が 1 つも無い(検収の空回り)"
    for b in per_step:
        assert sum(b) <= total_s + 1e-9, "未来の時間を先食いしている"
    b = full[0]
    assert abs(sum(b) - total_s) < 1e-9, "会計が総量へ着地していない"
    assert max(b) == 0.05 * 3.0, "係数 3 が効いていない"
    assert b[-1] < 0.05 * 3.0, "端数が最終塊で吸収されていない"
    assert len(b) < 400, "サブステップ数が減っていない(レバーの目的そのもの)"


def test_adaptive_dt_cuts_substeps_and_is_deterministic(tmp_path):
    """ON はサブステップ数を実際に減らし、同 seed 2 ランは完全一致する。"""
    kw = {"physics.adaptive_dt.enabled": "true",
          "physics.adaptive_dt.thresholds": "[[1,4.0]]"}
    off = _run(tmp_path, "sub_off", **_ON)
    a = _run(tmp_path, "sub_on_a", **_ON, **kw)
    b = _run(tmp_path, "sub_on_b", **_ON, **kw)
    assert _l1(a) == _l1(b), "同 seed 2 ランが一致しない(決定論が壊れた)"
    assert P.continuity(a) == P.continuity(b)
    on_sub = P.continuity(a)["sub_steps_total"]
    off_sub = P.continuity(off)["sub_steps_total"]
    assert on_sub < off_sub, (on_sub, off_sub)
    assert _l1(off) != _l1(a), "ON なのに世界が変わっていない"


def test_adaptive_dt_draws_no_rng(tmp_path):
    """係数選択は決定論量の純関数 = "physics" stream の消費本数が変わらない。"""
    class _CountingHub:
        def __init__(self, inner):
            self._inner = inner
            self.master_seed = inner.master_seed
            self.n = 0

        def stream(self, *key):
            if key and key[0] == P.STREAM:
                self.n += 1
            return self._inner.stream(*key)

        def key_name(self, *key):
            return self._inner.key_name(*key)

    def run(name, **ov):
        sim = _sim(tmp_path, name, **_ON, **ov)
        sim.hub = _CountingHub(sim.hub)
        sim.run()
        return sim.hub.n

    assert run("rng_on", **{"physics.adaptive_dt.enabled": "true",
                            "physics.adaptive_dt.thresholds": "[[1,4.0]]"}) \
        == run("rng_off")


# =========================================================================== #
# (5) ON: 破綻していないこと(品質統計)
# =========================================================================== #
def test_levers_on_keep_the_breakage_statistics_sane(tmp_path):
    """ON でも「重なりゼロ・瞬間移動なし・跳び上限内」は保たれる。

    ★jump の上限は **v_max × dt_eff**(= 適応 dt では係数ぶん緩む)。これは前進 Euler +
      v_max クリップからの機械的上限であって、緩めた値を後から正当化したものではない。
    """
    sim = _run(tmp_path, "qual_on", n=40, steps=10, zone_specs=[_zone()],
               **{"physics.adaptive_dt.enabled": "true",
                  "physics.adaptive_dt.thresholds": "[[1,4.0]]",
                  "physics.neighbor_cap": 7})
    c = P.continuity(sim)
    assert c["interior_samples"] > 0
    v_max = 1.4 * 1.3
    assert c["jump_max_m"] <= v_max * (0.05 * 4.0) + 1e-9, c["jump_max_m"]
    assert c["min_gap_m"] is None or c["min_gap_m"] >= 0.0, c["min_gap_m"]
    assert c["handover_jump_max_m"] <= zones.GATE_DEFAULTS["handover_jump_max_m"]
    assert not any(e.payload.get("far") for e in _kind(sim, "zone_gate"))
    # 滞在秒(会計の出口)が負にも NaN にもならない
    dwell = [e.payload["dwell_s"] for e in _kind(sim, "zone_gate")
             if e.payload["dir"] == "exit"]
    assert dwell and all(0.0 <= d < 10 ** 6 for d in dwell), dwell[:5]


def test_neighbor_cap_reaches_every_zone_end_to_end(tmp_path):
    """conf の 1 行が全ゾーンのエンジン引数まで届く(配線の end-to-end)。

    ★この mock ゾーン(50×50 m に 30 体)では cap は **binding しない**(半径 2 m の
      斥力カットオフ内に 7 体以上が入らない)= 軌跡は変わらない。cap が効いていることの
      証明は下の密集シナリオ(`test_neighbor_cap_changes_the_dynamics_when_it_binds`)で
      別に取る。ここで検収するのは配線だけ、と分けて書いておく。
    """
    sim = _sim(tmp_path, "cap_end", **_ON, **{"physics.neighbor_cap": 7})
    assert all(z.neighbor_cap == 7 for z in sim.physcfg["zones"])
    low = _run(tmp_path, "cap_low", **_ON, **{"physics.neighbor_cap": 7})
    assert _kind(low, "zone_gate")
    c = P.continuity(low)
    assert c["min_gap_m"] is None or c["min_gap_m"] >= 0.0, c["min_gap_m"]


def _packed(n=25, pitch=0.8, cross=False):
    """密集した格子配置(乱数ゼロ)。半径 2 m の斥力カットオフ内に 8 体以上が入る。

    `cross=True` は 4 方向の交差流(= スクランブルの縮図)。ORCA では**同じ向きに
    流れる群**の遠い近傍は既に満たされた制約しか作らないので、cap の効きを見るには
    向きを散らす必要がある(= 交差流こそ ORCA を置いた理由そのもの)。
    """
    side = int(round(n ** 0.5))
    xy = np.array([[(i % side) * pitch, (i // side) * pitch] for i in range(n)],
                  dtype=np.float64)
    v0 = np.full(n, 1.2)
    radius = np.full(n, 0.25)
    dirs = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    e = dirs[np.arange(n) % 4] if cross else np.tile([1.0, 0.0], (n, 1))
    return xy, e * v0[:, None], xy + e * 50.0, v0, radius


def test_neighbor_cap_changes_the_dynamics_when_it_binds():
    """cap は「飾り」ではない: 密集では 12 と 7 で SFM の力も ORCA の速度も変わる。"""
    from society.world import orca_core, sfm_core

    def sfm_force(cap):
        pos, vel, goal, v0, rad = _packed()
        c = sfm_core.Crowd(pos.copy(), vel.copy(), goal, v0, radius=rad,
                           neighbor_cap=cap)
        return c.forces()

    def orca_vel(cap):
        pos, vel, goal, v0, rad = _packed(cross=True)
        c = orca_core.OrcaCrowd(pos.copy(), vel.copy(), goal, v0, rad,
                                neighbor_cap=cap, pref_noise=0.0, rng=None)
        c.step(0.1)
        return c.vel.copy()

    assert not np.allclose(sfm_force(12), sfm_force(7)), "SFM で cap が効いていない"
    assert not np.allclose(orca_vel(12), orca_vel(7)), "ORCA で cap が効いていない"
    # 認知的近傍 ON でも k が効く(finals はこちらの経路を通る)
    def cog_force(k):
        pos, vel, goal, v0, rad = _packed()
        c = sfm_core.Crowd(pos.copy(), vel.copy(), goal, v0, radius=rad,
                           neighbor_cap=12, cognitive=True, cog_neighbors=k)
        return c.forces()

    assert not np.allclose(cog_force(12), cog_force(7)), \
        "認知的近傍の k が効いていない(= physics.neighbor_cap が死ぬ経路)"


# =========================================================================== #
# (6) registry 宣言
# =========================================================================== #
def test_registry_declares_both_levers():
    ids = {f.id: f for f in R.FEATURES}
    for fid in ("physics.adaptive_dt.enabled", "physics.neighbor_cap",
                "physics.separation_iters"):
        f = ids[fid]
        assert f.repro_tier == "strict"
        assert f.affects_k is False
        assert f.fingerprint_risk == "none"
        assert f.description.strip()
    assert ids["physics.neighbor_cap"].off_value == 0
    assert ids["physics.separation_iters"].off_value == 0
    cfg = OmegaConf.to_container(load_config(), resolve=True)
    assert R._select(cfg, "physics.adaptive_dt.enabled") is False
    assert R._select(cfg, "physics.neighbor_cap") == 0
    assert R._select(cfg, "physics.separation_iters") == 0
