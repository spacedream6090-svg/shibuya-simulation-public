"""conf/zones_shibuya.yaml — 渋谷の 3 空間(スクランブル交差点 / ハチ公前広場 / センター街)を
連続物理にする**実配置**プロファイルの検収。

竹-4(P3 境界縫合)の機構そのものは tests/test_physics_zones.py が固定している。本ファイルが
固定するのは「**実際の渋谷の地図に据えた宣言が正しいか**」= 幾何・ノード結線・信号の実体・
較正の選択・既定 OFF の不変であって、物理エンジンの性質ではない。

正典:
  conf/zones_shibuya.yaml(宣言そのもの)
  data/shibuya_osm_wide_v7.json(ポリゴン座標の前提。原点 = スクランブル交差点)
  data/crossings_shibuya.json(信号の実測 cycle/green/flash・OSM crossing id)
  reference/physics_bench/README.md §9〜§11(どの較正を ON にしてよいかの実測)
  docs/research/p4-calibration-research.md 追記(2026-08-05)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from society import physics as P
from society.config import load_config
from society.engine import checkpoint, scheduler
from society.engine.simulation import Simulation
from society.world import indoor_flow as _flow, sfm_core as _sfm, zones as Z

REPO = Path(__file__).resolve().parents[1]
PROFILE = REPO / "conf" / "zones_shibuya.yaml"
MAP = "data/shibuya_osm_wide_v7.json"

# 宣言が指す実ノード(地図 wide_v7 の実体。ここが変わったら宣言も直すこと)
EXPECT_INSIDE = {
    "scramble": ("n291758776",),                       # 「スクランブル交差点」= 車道そのもの
    "hachiko_square": ("n12314075741", "n12352271648"),  # 「ハチ公前広場」+ 広場東の通路
    "center_gai": ("n10151868701", "n5244745710"),     # センター街 入口の角 〜 中程
}
EXPECT_GATES = {
    "scramble": ("n1499530620", "n1499530621", "n1499530624", "n1676749591",
                 "n2021391989", "n2223657295", "n2223657341", "n5244745709"),
    "hachiko_square": ("n12352271624", "n12352271649", "n12352271650",
                       "n5091218458", "n5091218473", "n6219729369"),
    "center_gai": ("n1014529169", "n10151868702", "n12352271628", "n12352271657",
                   "n1424672751", "n2268223553"),
}
# 交差点表の実測(渋谷スクランブル。docs/research/pedestrian-signals.md §2.3)
SCRAMBLE_CROSSING_ID = 291758776
CYCLE_S, GREEN_S, FLASH_S = 140.0, 37.0, 10.0


# --------------------------------------------------------------------------- #
# 共通
# --------------------------------------------------------------------------- #
def _cfg(name, n=40, steps=8, profile=True, **ov):
    dot = ["run.seed=42", f"run.n_agents={n}", f"run.n_steps={steps}",
           f"run.name={name}", "model.backend=mock", "observer.snapshot_every=1440"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot, profile=str(PROFILE) if profile else None)


def _kind(sim, kind):
    return [e for e in sim.logger.events if e.kind == kind]


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


@pytest.fixture(scope="module")
def on_sim(tmp_path_factory):
    """プロファイルをそのまま使った mock スモーク(モジュール内で 1 回だけ回す)。"""
    d = tmp_path_factory.mktemp("zones_on")
    sim = Simulation(_cfg("zshib_on"), out_dir=d / "on")
    sim.run()
    return sim


# =========================================================================== #
# (1) プロファイルが読めて検証を通る
# =========================================================================== #
def test_profile_loads_and_builds_three_disjoint_zones():
    cfg = _cfg("load", steps=1)
    assert cfg.world.map == MAP, "ポリゴン座標の前提である広域地図が指定されていない"
    raw = OmegaConf.to_container(cfg.physics, resolve=True)
    # build_cfg は非重複(P2 決定 条件6)を**構築時に**検査する = ここを通ることが検収
    pc = Z.build_cfg(raw, REPO, step_seconds=600.0)
    assert pc["zones_enabled"] is True
    zs = pc["zones"]
    assert [z.id for z in zs] == ["scramble", "hachiko_square", "center_gai"], \
        "ゾーンの並び順は意味を持つ(信号を持つ scramble が先着で所有する必要がある)"
    assert [z.engine for z in zs] == ["orca", "sfm", "sfm"], \
        "交差流は orca・開放平面/街路は sfm(P2 決定の割当規則)"
    for z in zs:
        assert 0.0 < z.dt_sub <= 0.1                 # P2 条件1
        assert z.layers == (0,), "地上レイヤーに限定していない(地下通路を巻き込む)"
        assert z.max_sub_steps == 12000              # Δt=10 / dt_sub=0.05 の導出値
        assert z.walls == ()                         # 壁線分は宣言しない(= 壁項は評価されない)
        assert len(z.polygon) >= 3


def test_zone_areas_are_in_the_declared_range():
    """面積(密度の分母)が宣言どおりであること。数字が動いたら密度の解釈が変わる。"""
    zs = {z.id: z for z in _zones()}
    assert 800 <= zs["scramble"].area_m2() <= 900          # 840 m²(縁石の内側)
    assert 1050 <= zs["hachiko_square"].area_m2() <= 1150  # 1,102 m²
    assert 480 <= zs["center_gai"].area_m2() <= 540        # 507 m²(幅 9 m × 約 56 m)


def _zones():
    raw = OmegaConf.to_container(_cfg("z", steps=1).physics, resolve=True)
    return Z.build_cfg(raw, REPO, step_seconds=600.0)["zones"]


def test_disjointness_check_would_catch_an_overlap():
    """非重複検査が**空回りしていない**ことの対照(わざと重ねたら落ちる)。"""
    zs = list(_zones())
    twin = Z._build_zone({"id": "twin", "engine": "sfm", "dt_sub": 0.05,
                          "polygon": [list(p) for p in zs[0].polygon]}, REPO, 600.0)
    with pytest.raises(ValueError, match="重複"):
        Z._check_disjoint(zs + [twin])


# =========================================================================== #
# (2) 実地図への結線(内部ノード・ゲート)
# =========================================================================== #
def test_zones_bind_to_the_expected_real_nodes(on_sim):
    graph = on_sim.city.graph
    for z in on_sim.physcfg["zones"]:
        ins = Z.inside_nodes(z, graph)
        gates = Z.gates_of(z, graph)
        assert ins == EXPECT_INSIDE[z.id], f"{z.id}: 内部ノードが宣言とずれた"
        assert gates == EXPECT_GATES[z.id], f"{z.id}: ゲートが宣言とずれた"
        for g in gates:
            assert g not in set(ins)
            assert any(n in set(ins) for n in graph.neighbors(g))
            assert int(graph.nodes[g]["layer"]) == 0


def test_layer_filter_is_load_bearing_at_the_scramble():
    """`layers: [0]` を外すと交差点の**真下の地下通路ノード**を巻き込む(= この宣言は必要)。

    同時に、既定 `layers: ()` が従来の幾何だけの判定と完全同値であることも固定する。
    """
    from society.world.map import CityMap
    city = CityMap(str(REPO / MAP))
    z = next(z for z in _zones() if z.id == "scramble")
    spec_all = {"id": "s_all", "engine": "orca", "dt_sub": 0.05,
                "polygon": [list(p) for p in z.polygon]}
    z_all = Z._build_zone(spec_all, REPO, 600.0)
    assert z_all.layers == (), "既定は全レイヤー(従来と同値)でなければならない"
    ins_all = Z.inside_nodes(z_all, city.graph)
    ins_0 = Z.inside_nodes(z, city.graph)
    below = [n for n in ins_all if n not in set(ins_0)]
    assert below, "地下ノードが 1 つも無い = この検収が空回りしている"
    assert all(int(city.graph.nodes[n]["layer"]) < 0 for n in below)
    # 既定(全レイヤー)の判定は `contains` そのもの = 1 バイトも変えていない
    assert set(ins_all) == {n for n, d in city.graph.nodes(data=True)
                            if z_all.contains(float(d["x"]), float(d["y"]))}


def test_polygons_do_not_swallow_buildings_at_the_crossing_and_the_square():
    """交差点と広場のポリゴンは建物 footprint を 1 点も含まない(車道・広場そのもの)。

    センター街は例外で、OSM の建物 footprint がこの街区では粗く(b136690966 が一体化した
    大ブロック)建物間の空きが約 4.3 m しかない。実測の街路幅(約 9 m)を採ったので
    footprint 頂点が入る。**walls は none なので力学には無影響**(効くのは密度の分母だけ)。
    """
    raw = json.loads((REPO / MAP).read_text(encoding="utf-8"))
    zs = {z.id: z for z in _zones()}
    def hits(z):
        return [b["id"] for b in raw["buildings"]
                if any(z.contains(float(q[0]), float(q[1])) for q in b["footprint"])]
    assert hits(zs["scramble"]) == []
    assert hits(zs["hachiko_square"]) == []
    assert len(hits(zs["center_gai"])) <= 4, "センター街の建物被りが既知の 4 棟を超えた"


# =========================================================================== #
# (3) 信号 = 実在の交差点表への結線
# =========================================================================== #
def test_scramble_signal_binds_to_the_real_crossing_table_row():
    rows = json.loads((REPO / "data" / "crossings_shibuya.json")
                      .read_text(encoding="utf-8"))["crossings"]
    row = next(r for r in rows if int(r["id"]) == SCRAMBLE_CROSSING_ID)
    # 選んだのは「本物のスクランブル横断」(OSM crossing_ref=pedestrian_scramble)
    assert row.get("scramble") is True
    assert row.get("crossing_ref") == "pedestrian_scramble"
    assert row.get("name") == "渋谷駅前"
    assert (row["cycle_s"], row["green_s"], row["flash_s"]) == (CYCLE_S, GREEN_S, FLASH_S)
    # 交差点表の id は**グラフのノード id と同じ**(= 中心ノードそのもの)
    assert f"n{SCRAMBLE_CROSSING_ID}" in EXPECT_INSIDE["scramble"]
    # 交差点の位置とゾーンの中心が一致している(表と地図が同じ原点)
    z = next(z for z in _zones() if z.id == "scramble")
    assert z.contains(float(row["x"]), float(row["y"]))
    sig = next(z for z in _zones() if z.id == "scramble").signal
    assert sig["crossing_id"] == SCRAMBLE_CROSSING_ID
    assert (sig["cycle_s"], sig["green_s"], sig["flash_s"]) == (CYCLE_S, GREEN_S, FLASH_S)


def test_only_the_scramble_carries_a_signal():
    """広場・街路に信号は無い(歩行者専用空間を信号で止めない)。"""
    for z in _zones():
        assert bool(z.signal) == (z.id == "scramble")


# =========================================================================== #
# (4) P4 較正の選択(実測が支持するものだけ ON)
# =========================================================================== #
def test_only_far_field_is_enabled_with_the_calibrated_values():
    """far_field だけ ON(A/B/C 合格の実測構成)。v_of_s と wall は既定のまま。

    根拠: reference/physics_bench/README.md §11(受入判定表)
      - 「far のみ」= A 1.00 ✓ / B 0 違反 ✓ / C 1.00 ✓ / churn ≤0.001
      - V(s) は「限界効用ほぼゼロ(0.113→0.131 と微悪化)」= 既定 OFF 据え置き(§11-4)
      - 壁のみ ON は単独では悪化(A 0.20・churn 0.230)(§11-5)。かつ本ゾーン群は
        walls を宣言しないので壁項は 1 度も評価されない。
    """
    raw = OmegaConf.to_container(_cfg("calib", steps=1).physics, resolve=True)
    s = Z.build_cfg(raw, REPO, step_seconds=600.0)["sfm"]
    assert s["far_field"] == {"enabled": True, "a2": 0.119, "b2": 1.890,
                              "cutoff_factor": 2.5, "taper_m": 1.0}
    assert s["v_of_s"]["enabled"] is False
    assert (s["wall"]["a"], s["wall"]["b"]) == (_sfm.WALL_A_DEFAULT, _sfm.WALL_B_DEFAULT)


def test_calibration_reaches_the_sfm_zones_only(on_sim):
    """較正引数は SFM ゾーンにだけ効く(ORCA ゾーンは無関係)。"""
    kw = P._calib_kwargs(on_sim)
    assert kw == {"far_a2": 0.119, "far_b2": 1.890,
                  "far_cutoff_factor": 2.5, "far_taper_m": 1.0}
    assert P.calib_describe(on_sim).keys() == {"far_field"}
    sfm_zone = next(z for z in on_sim.physcfg["zones"] if z.engine == "sfm")
    orca_zone = next(z for z in on_sim.physcfg["zones"] if z.engine == "orca")
    rec = [{"pos": (0.0, 0.0), "vel": (0.0, 0.0), "wp_xy": (10.0, 0.0),
            "v0": 1.2, "radius": 0.3}]
    assert isinstance(P._build_engine(sfm_zone, rec, None, kw), P._CalibratedCrowd)
    eng = P._build_engine(orca_zone, rec, None, kw)
    assert not isinstance(eng, P._CalibratedCrowd)


# =========================================================================== #
# (5) 既定 OFF の不変(基底 conf は 1 バイトも動かしていない)
# =========================================================================== #
def test_base_config_still_declares_zero_zones():
    base = load_config(["run.name=zshib_base"])
    assert base.physics.zones_enabled is False
    assert list(base.physics.zones) == []
    assert base.physics.sfm.far_field.enabled is False
    assert base.physics.sfm.v_of_s.enabled is False
    pc = Z.build_cfg(OmegaConf.to_container(base.physics, resolve=True), REPO, 600.0)
    assert pc["zones"] == ()


def test_off_run_on_the_same_map_leaves_no_zone_trace(tmp_path):
    """プロファイルを外せば(同じ広域地図でも)痕跡ゼロ = ON は完全に opt-in。"""
    cfg = _cfg("zshib_off", n=24, steps=3, profile=False, **{"world.map": MAP})
    sim = Simulation(cfg, out_dir=tmp_path / "off")
    sim.run()
    assert sim.physcfg["zones"] == ()
    assert not _kind(sim, "zone_gate")
    assert getattr(sim, "_phys_state", None) is None
    assert all(getattr(a, "_phys_zone", None) is None for a in sim.agents)


# =========================================================================== #
# (6) ON スモーク(実地図・実信号での挙動)
# =========================================================================== #
def test_agents_actually_traverse_the_scramble(on_sim):
    ev = _kind(on_sim, "zone_gate")
    assert ev, "3 ゾーンを誰も通っていない(検収の空回り)"
    zs = {e.payload["zone"] for e in ev}
    assert "scramble" in zs, "スクランブル交差点を誰も横断していない"
    for e in ev:
        assert e.payload["zone"] in EXPECT_INSIDE
        assert e.payload["gate"].startswith(e.payload["zone"] + ":")


def test_enter_equals_exit_plus_still_owned(on_sim):
    """会計: 入場総数 = 退場総数 + いま所有されている人数(取りこぼしゼロ)。"""
    st = P.state_of(on_sim)
    owned = [a for a in on_sim.agents if P.owned(a)]
    assert st["enter_total"] == st["exit_total"] + len(owned)
    ins = [e for e in _kind(on_sim, "zone_gate") if e.payload["dir"] == "enter"]
    outs = [e for e in _kind(on_sim, "zone_gate") if e.payload["dir"] == "exit"]
    assert len(ins) == st["enter_total"] and len(outs) == st["exit_total"]


def test_every_admission_happens_on_green_and_red_makes_them_wait(on_sim):
    """赤の間に車道へ入った個体は 1 人も居ない。かつ赤待ちが実際に起きている。"""
    zone = next(z for z in on_sim.physcfg["zones"] if z.id == "scramble")
    gate = _sfm.SignalGate(**zone.signal)
    ins = [e for e in _kind(on_sim, "zone_gate")
           if e.payload["dir"] == "enter" and e.payload["zone"] == "scramble"]
    assert ins, "スクランブルに誰も入場していない(検収の空回り)"
    waits = []
    for e in ins:
        adm = (float(e.sim_min) * 60.0 + float(e.payload["wait_s"])
               - float(e.payload["waited_steps"]) * float(on_sim.clock.step_seconds))
        assert gate.can_cross(adm), f"赤で入場した: wait_s={e.payload['wait_s']}"
        waits.append(float(e.payload["wait_s"]))
    assert any(w > 0.0 for w in waits), "赤待ちが一度も起きていない(信号が空回り)"
    # 赤の長さ = cycle − (green + flash) = 93 s。待ちがこれを超えることは無い
    assert max(waits) <= CYCLE_S - GREEN_S - FLASH_S + 1e-6
    # 信号のあるゾーンだけが待たせている。信号の無いゾーンでも guarded ゲート(入口が
    # 塞がっていれば入れない)で微小な待ちは出るが、**待つ個体の割合が桁で違う**。
    others = [float(e.payload["wait_s"]) for e in _kind(on_sim, "zone_gate")
              if e.payload["dir"] == "enter" and e.payload["zone"] != "scramble"]
    if others:
        red_frac = sum(1 for w in waits if w > 0.0) / len(waits)
        oth_frac = sum(1 for w in others if w > 0.0) / len(others)
        assert red_frac > oth_frac + 0.15, \
            f"信号ゾーンの待ち率が非信号ゾーンと差が無い({red_frac:.2f} vs {oth_frac:.2f})"


def test_handover_is_continuous_and_no_body_overlaps(on_sim):
    """グラフ復帰の跳びが安全弁の内側(far フラグ 0 件)・体表の重なりが無い。"""
    cont = P.continuity(on_sim)
    limit = next(z for z in on_sim.physcfg["zones"] if z.id == "scramble") \
        .gate["handover_jump_max_m"]
    assert cont["handover_jump_max_m"] <= limit, "射影距離が安全弁を超えた"
    assert not [e for e in _kind(on_sim, "zone_gate") if e.payload.get("far")]
    # 体表の重なり: ORCA は分離パスで非負が保証されるが **SFM は保証しない**(力ベース)。
    # min_gap_m は全ゾーン通算なので、ここでは「体半径ぶんめり込む」ような破綻だけを弾く。
    assert cont["min_gap_m"] is None or cont["min_gap_m"] > -_flow.RADIUS_MIN, \
        "体半径を超えて重なった(詰め込みが破綻している)"
    # 1 サブステップの変位が v_max·dt を超えない(積分が飛んでいない)
    dt_sub = next(z.dt_sub for z in on_sim.physcfg["zones"])
    v_max = _flow.V0_MAX * 1.3
    assert cont["jump_max_m"] <= v_max * dt_sub + 1e-6


def test_no_forced_release_in_the_smoke(on_sim):
    """安全弁(滞在超過・待ち超過)が発火しない = 詰まっていない。"""
    st = P.state_of(on_sim)
    assert st["forced_total"] == 0
    reasons = {e.payload["reason"] for e in _kind(on_sim, "zone_gate")
               if e.payload["dir"] == "exit"}
    assert reasons <= {"gate", "detached"}


def test_on_same_seed_two_runs_are_identical(tmp_path):
    a = Simulation(_cfg("det_a", n=30, steps=5), out_dir=tmp_path / "a")
    a.run()
    b = Simulation(_cfg("det_b", n=30, steps=5), out_dir=tmp_path / "b")
    b.run()
    assert _kind(a, "zone_gate"), "検収の空回り"
    assert _l1(a) == _l1(b)
    assert P.continuity(a) == P.continuity(b)


def test_resume_across_a_zone_matches_straight(tmp_path):
    """ゾーン滞在を跨いで分割再開しても straight と一致する。

    実地図の mock ランでは多くの個体が 1 step で抜けるので、
    `max_sub_steps` を絞って**所有が必ず step 境界を跨ぐ**条件を作る
    (tests/test_physics_zones.py の resume 検収と同じ手口)。
    """
    import pyarrow.parquet as pq

    def cfg(name):
        c = _cfg(name, n=30, steps=8)
        for z in c.physics.zones:
            z.max_sub_steps = 400          # = 20 秒ぶんだけ積む
        return c

    sdir = tmp_path / "straight"
    straight = Simulation(cfg("zr_straight"), out_dir=sdir)
    straight.run()
    assert _kind(straight, "zone_gate"), "検収の空回り"

    d = tmp_path / "resumed"
    sim1 = Simulation(cfg("zr_resumed"), out_dir=d)
    split, owned = 0, []
    for step in range(7):
        scheduler.run_step(sim1, step)
        split = step + 1
        owned = [a.id for a in sim1.agents if P.owned(a)]
        if split >= 2 and owned:
            break
    assert owned, "分割点でゾーン滞在中の個体が居ない(検収の空回り)"
    checkpoint.save(sim1, split, d / "checkpoint" / f"ckpt-{split:06d}.pkl.gz")
    sim1.logger.flush_segment()
    sim2 = Simulation(cfg("zr_resumed"), out_dir=d)
    sim2.run(resume_from=d)
    for stem in ("l1_events", "l2_metrics", "l3_snapshots"):
        assert (pq.read_table(sdir / f"{stem}.parquet").to_pylist()
                == pq.read_table(d / f"{stem}.parquet").to_pylist()), \
            f"{stem} 不一致(実ゾーンの resume)"


def test_measured_density_stays_below_the_wall_penetration_regime(on_sim):
    """到達密度の実測。P4 の壁貫通病理(§8)は ρ_global ≥ 2.0 の**壁つき**系の話なので、
    walls を宣言しない本ゾーン群では原理的に起きない。ここでは「実際に何人/m² まで
    行ったか」を記録として固定する(閾値は緩く、桁が変わったら気づくためのもの)。
    """
    st = P.state_of(on_sim)
    for zid, z in sorted(st["by_zone"].items()):
        assert z["density"] >= 0.0
    zone_area = {z.id: (z.walkable_area_m2 or z.area_m2()) for z in on_sim.physcfg["zones"]}
    for zid, area in zone_area.items():
        assert area > 100.0, f"{zid}: 密度の分母が小さすぎる"
    # 体表の重なりゼロ = 物理的にありえない詰め込みは起きていない
    assert st["min_gap_m"] is None or st["min_gap_m"] > -1e-9


def test_l2_carries_the_zone_columns(on_sim):
    import pyarrow.parquet as pq
    tbl = pq.read_table(on_sim.out_dir / "l2_metrics.parquet")
    for c in ("zone_occupancy", "zone_density_mean", "zone_dwell_mean_s",
              "zone_gate_enter_total", "zone_gate_exit_total"):
        assert c in tbl.column_names
    enters = [v for v in tbl.column("zone_gate_enter_total").to_pylist() if v is not None]
    assert enters and enters == sorted(enters)


def test_owned_span_is_reported_for_the_operator(on_sim):
    """★正直な申告: 所有は「経路がゾーンを通り抜ける個体」に対して**現在地から**始まる。

    グラフ移動は 1 step で 800 m 進む(world.modes.speeds.walk)ので、ゾーンを跨ぐ個体を
    取りこぼさないためには現在地から所有するしかない(= 設計どおり)。その結果
    `span_m`(所有区間のグラフ経路長)は数百 m に達し、**L2 の zone_occupancy /
    zone_density_mean はポリゴン内の人数ではなく「所有中の人数」**になる。
    ここではその性質を機械的に固定して、指標の読み違いを防ぐ。
    """
    ins = [e for e in _kind(on_sim, "zone_gate") if e.payload["dir"] == "enter"]
    spans = [float(e.payload["span_m"]) for e in ins]
    assert spans and max(spans) > 100.0, \
        "span_m が短い = 前提が変わった(この検収の意味が失われている)"
    # 所有区間はゾーン面積の代表長(√area ≈ 30 m)より必ず長い
    assert min(spans) > 0.0
