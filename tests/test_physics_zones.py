"""竹-4 = P3 境界縫合(physics.zones / orca_core / ゲート / 物理→知覚)のテスト。

正典: docs/plans/source/physics-instructions.md Part P3 の受け入れ基準
      docs/research/physics-engine-selection.md ★P2 決定(2026-08-02)条件 1〜6

方針(R1 の鉄則を継承):
- OFF(既定): 純粋既定と L1 バイト一致・`agent._phys_*` が 1 つも生えない・zone_gate 0 件・
  L2 に列なし・`sim._phys_state` 不在・"physics" stream を 1 本も引かない
  (ゴールデンそのものは tests/test_scenario.py が固定)。
- ON: 同 seed 2 ラン一致・workers 数非依存・resume==straight・k=free/off で LLM 呼数一致・
  no-fingerprint(**同一の位置系列なら**プロンプト文字列不変)。
- 境界連続性(jump_max / accel_p99 / reversal_rate)と ORCA の重なり(min_gap ≥ 0)は
  **ベンチ実測値を上限の根拠に**閾値化する。
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from omegaconf import OmegaConf

from society import physics as P
from society import registry as R
from society.cognition import perception_contract as contract
from society.config import load_config
from society.engine import checkpoint, scheduler
from society.engine.simulation import Simulation
from society.world import orca_core, sfm_core as _sfm, zones

REPO = Path(__file__).resolve().parents[1]

# ── ベンチ実測(reference/physics_bench/out/results.json・dt=0.1・幅3m 通路・200体)──
#   guarded · ORCA: gate accel_p99 5.15 / interior 2.18 / gate 反転 0.123 / 反転 0.122
#   blind   · ORCA: gate accel_p99 7.17 / interior 2.12 / gate 反転 0.246
#   guarded · SFM : gate accel_p99 9.34 / interior 5.72 / gate 反転 0.695
# 加速度は dt に反比例するので、**同じ Δv でも dt=0.05 なら 2 倍の値になる**。
# 比較可能な量にするため「1 サブステップあたりの Δv = accel × dt」で上限を置く。
BENCH_DT = 0.1
BENCH_GATE_DV = 5.15 * BENCH_DT        # 0.515 m/s(guarded ORCA のゲート帯)
BENCH_INT_DV = 5.72 * BENCH_DT         # 0.572 m/s(guarded SFM の内部=両候補の悪い方)
BENCH_BLIND_REVERSAL = 0.246           # blind ORCA のゲート帯反転率(guarded はこれ未満のはず)

_ZONE_R = 25.0
_POLY = [[-_ZONE_R, -_ZONE_R], [_ZONE_R, -_ZONE_R], [_ZONE_R, _ZONE_R], [-_ZONE_R, _ZONE_R]]


def _zone(zid="z1", engine="orca", **ov):
    z = {"id": zid, "engine": engine, "dt_sub": 0.05, "polygon": list(_POLY)}
    z.update(ov)
    return z


def _cfg(name, n=20, steps=6, zone_specs=None, **ov):
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


class _CountingHub:
    """"physics" stream の draw 消費を数えるプロキシ(test_dunbar と同型)。"""

    def __init__(self, inner):
        self._inner = inner
        self.master_seed = inner.master_seed
        self.physics_streams = 0

    def stream(self, *key):
        if key and key[0] == P.STREAM:
            self.physics_streams += 1
        return self._inner.stream(*key)

    def key_name(self, *key):
        return self._inner.key_name(*key)


class _FixedLLM:
    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.hits = 0

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        self.calls += 1
        return self.response, str(self.calls), False


# =========================================================================== #
# (1) 既定 OFF = 完全 no-op
# =========================================================================== #
def test_default_off_is_byte_identical_and_leaves_no_trace(tmp_path):
    """既定(zones_enabled=false / zones=[])は純粋既定と L1 バイト一致・痕跡ゼロ。"""
    base = _run(tmp_path, "off_base", steps=6)
    # 同一 config を 2 回組んでも同じ(= 物理層が何も足していない)
    again = _run(tmp_path, "off_again", steps=6)
    assert _l1(base) == _l1(again)
    assert base.physcfg["zones"] == ()
    assert not P.enabled(base)
    assert getattr(base, "_phys_state", None) is None, "OFF で状態が生えている"
    assert not _kind(base, "zone_gate"), "OFF で zone_gate が出ている"
    for a in base.agents:
        assert getattr(a, "_phys_zone", None) is None
        assert not P.owned(a)
        assert P.body_of(a) is None


def test_default_off_draws_no_physics_stream(tmp_path):
    """OFF では "physics" stream を 1 本も派生しない(既存 draw 順に無風)。"""
    sim = _sim(tmp_path, "off_stream", steps=6)
    sim.hub = _CountingHub(sim.hub)
    sim.run()
    assert sim.hub.physics_streams == 0


def test_default_off_has_no_l2_columns(tmp_path):
    """OFF の L2 にゾーン列が 1 つも無い(第75 dunbar と同構造)。"""
    sim = _run(tmp_path, "off_l2", steps=4)
    cols = pq.read_table(sim.out_dir / "l2_metrics.parquet").column_names
    for c in ("zone_occupancy", "zone_density_mean", "zone_dwell_mean_s",
              "zone_gate_enter_total", "zone_gate_exit_total"):
        assert c not in cols, f"OFF なのに L2 列 {c} が生えている"


def test_perception_body_is_missing_when_off(tmp_path):
    """OFF では body の物理 3 欄が None のまま(欠測を 0 で埋めない)。"""
    sim = _sim(tmp_path, "off_body", steps=1)
    scheduler.run_step(sim, 0)
    agent = sim.agents[0]
    material = scheduler._gather_material(sim, agent, "solo", 0, 0)
    percept = scheduler.build_perception(sim, agent, material)
    assert percept.body["blocked"] is None
    assert percept.body["contact"] is None
    assert percept.body["local_density"] is None


# =========================================================================== #
# (2) 宣言(conf)の検証
# =========================================================================== #
def test_build_cfg_rejects_bad_declarations():
    root = REPO
    # 未知キー
    try:
        zones.build_cfg({"nope": 1}, root)
        raise AssertionError("未知キーが素通りした")
    except KeyError:
        pass
    # dt_sub は P2 条件1 の (0, 0.1]
    for bad in (0.0, 0.2):
        try:
            zones.build_cfg({"zones_enabled": True,
                             "zones": [_zone(dt_sub=bad)]}, root)
            raise AssertionError(f"dt_sub={bad} が素通りした")
        except ValueError:
            pass
    # ポリゴンは 3 点以上
    try:
        zones.build_cfg({"zones_enabled": True,
                         "zones": [_zone(polygon=[[0, 0], [1, 1]])]}, root)
        raise AssertionError("2 点ポリゴンが素通りした")
    except ValueError:
        pass
    # エンジンは sfm|orca
    try:
        zones.build_cfg({"zones_enabled": True, "zones": [_zone(engine="rvo3")]}, root)
        raise AssertionError("未知 engine が素通りした")
    except ValueError:
        pass


def test_max_sub_steps_tracks_dt_through_the_clock(tmp_path):
    """第99(物理見積 残①): `max_sub_steps` 未指定のゾーンは上限を **Δt から導く**。

    OBS-U2 §1.2 B5 / R7 は「12000 は 600s/0.05s の直書きで Δt に追随しない。Δt>10 では
    physics.py:331 の min() が積分を黙って打ち切る」と指摘していた。Simulation は
    `clock.step_seconds` を build_cfg へ渡すので、宣言が書いていなければ上限が追随する。
    **Δt=10 では厳密に 12000 = 従来値**(= 既定ランは 1 ビットも変わらない)。
    """
    def cap(name, **ov):
        sim = _sim(tmp_path, name, n=6, steps=1, zone_specs=[_zone()], **ov)
        return int(sim.physcfg["zones"][0].max_sub_steps)

    assert cap("cap_dt10") == 12000, "既定 Δt=10 で従来値 12000 から動いた"
    assert cap("cap_dt20", **{"run.dt_min": 20}) == 24000     # 旧: 12000 で打ち切り
    assert cap("cap_dt1", **{"run.dt_min": 1}) == 1200
    # 明示宣言は Δt に関わらず尊重される(上限を意図的に絞る使い方 = resume テストの前提)
    sim = _sim(tmp_path, "cap_explicit", n=6, steps=1,
               zone_specs=[_zone(max_sub_steps=400)], **{"run.dt_min": 20})
    assert int(sim.physcfg["zones"][0].max_sub_steps) == 400
    # 導出は physics.py:331 の min() を binding させない = step 全長ぶん積める
    z = sim.physcfg["zones"][0]
    assert zones.derive_max_sub_steps(z.dt_sub, 1200.0) == 24000


def test_overlapping_zones_are_rejected():
    """ゾーン非重複(P2 決定 条件6)。重複宣言はランを始めさせない。"""
    a = _zone("a")
    b = _zone("b", polygon=[[0, 0], [40, 0], [40, 40], [0, 40]])
    try:
        zones.build_cfg({"zones_enabled": True, "zones": [a, b]}, REPO)
        raise AssertionError("重複ゾーンが素通りした")
    except ValueError as exc:
        assert "重複" in str(exc)
    # 離れていれば通る
    far = _zone("b", polygon=[[500, 500], [540, 500], [540, 540], [500, 540]])
    cfg = zones.build_cfg({"zones_enabled": True, "zones": [a, far]}, REPO)
    assert len(cfg["zones"]) == 2


def test_zones_enabled_false_ignores_declarations():
    """zones_enabled=false なら宣言があってもゾーン 0 件(二重の安全弁)。"""
    cfg = zones.build_cfg({"zones_enabled": False, "zones": [_zone()]}, REPO)
    assert cfg["zones"] == ()


def test_point_in_polygon_is_correct_and_boundary_inclusive():
    poly = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))
    bbox = zones.polygon_bbox(poly)
    assert zones.point_in(poly, 5.0, 5.0, bbox)
    assert not zones.point_in(poly, 15.0, 5.0, bbox)
    assert not zones.point_in(poly, -0.001, 5.0, bbox)
    assert zones.point_in(poly, 0.0, 5.0, bbox), "境界上は内側扱い"
    assert zones.point_in(poly, 10.0, 10.0, bbox), "頂点は内側扱い"
    assert abs(zones.polygon_area(poly) - 100.0) < 1e-9
    # 凹多角形(L 字)でも ray casting が効く
    ell = ((0.0, 0.0), (10.0, 0.0), (10.0, 4.0), (4.0, 4.0), (4.0, 10.0), (0.0, 10.0))
    bb = zones.polygon_bbox(ell)
    assert zones.point_in(ell, 1.0, 9.0, bb)
    assert not zones.point_in(ell, 9.0, 9.0, bb)


def test_gates_are_outside_and_adjacent_to_inside(tmp_path):
    """ゲート = ゾーン外にあってゾーン内ノードに隣接するノード(id 昇順・決定論)。"""
    sim = _sim(tmp_path, "gates", steps=1, zone_specs=[_zone()])
    zone = sim.physcfg["zones"][0]
    graph = sim.city.graph
    ins = set(zones.inside_nodes(zone, graph))
    gates = zones.gates_of(zone, graph)
    assert ins and gates, "テスト前提が崩れた(ゾーンに道路が無い)"
    assert list(gates) == sorted(gates)
    for g in gates:
        assert g not in ins, "ゲートがゾーンの内側にある"
        assert any(n in ins for n in graph.neighbors(g))
    assert zones.gates_of(zone, graph) == gates, "同一入力で結果が揺れる"


def test_route_span_skips_routes_that_end_inside(tmp_path):
    """目的地がゾーン内の経路は所有しない(到着処理はグラフ側の責務=正直な適用範囲)。"""
    sim = _sim(tmp_path, "span", steps=1, zone_specs=[_zone()])
    zone = sim.physcfg["zones"][0]
    graph = sim.city.graph
    ins = sorted(zones.inside_nodes(zone, graph))
    gates = zones.gates_of(zone, graph)
    inside = ins[0]
    gate = next(g for g in gates if inside in set(graph.neighbors(g)))
    # 目的地がゾーン内 → None
    assert zones.route_span(zone, graph, gate, [inside]) is None
    # 通り抜ける → (path, rest)
    out_node = next((n for n in graph.neighbors(inside) if n not in set(ins)), None)
    if out_node is not None:
        span = zones.route_span(zone, graph, gate, [inside, out_node, gate])
        assert span is not None
        path, rest = span
        assert path[0] == gate and path[-1] == out_node
        assert rest == [gate]


# =========================================================================== #
# (3) ORCA コア(移植の忠実さ + P2 決定 条件3 の重なり対策)
# =========================================================================== #
def _reference_orca():
    """reference/physics_bench/orca_min.py を **その場所から** 読み込む(reference は不変)。"""
    path = REPO / "reference" / "physics_bench" / "orca_min.py"
    spec = importlib.util.spec_from_file_location("_ref_orca_min", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _orca_case(cls, **extra):
    n = 12
    ang = np.arange(n) * (2 * math.pi / n)
    pos = np.stack([10.0 * np.cos(ang), 10.0 * np.sin(ang)], axis=1)
    goal = -pos
    vel = np.zeros((n, 2))
    v0 = np.linspace(1.0, 1.4, n)
    radius = np.linspace(0.25, 0.35, n)
    return cls(pos, vel, goal, v0, radius, neighbor_cap=6, **extra)


def test_orca_core_matches_reference_bit_for_bit():
    """昇格版は margin=0 / 分離パス無効なら reference の移植元と **バイト一致**。

    = 「重なり対策で数式そのものを変えていない」ことの機械的な証明。
    """
    ref = _reference_orca()
    a = _orca_case(ref.OrcaCrowd)
    b = _orca_case(orca_core.OrcaCrowd, radius_margin=0.0, separation_iters=0)
    for _ in range(120):
        a.step(0.05)
        b.step(0.05)
    assert a.pos.tobytes() == b.pos.tobytes()
    assert a.vel.tobytes() == b.vel.tobytes()


def test_orca_core_is_deterministic_with_and_without_noise():
    rng = np.random.default_rng(7)
    a = _orca_case(orca_core.OrcaCrowd, pref_noise=0.05, rng=np.random.default_rng(7))
    b = _orca_case(orca_core.OrcaCrowd, pref_noise=0.05, rng=np.random.default_rng(7))
    for _ in range(60):
        a.step(0.05)
        b.step(0.05)
    assert a.pos.tobytes() == b.pos.tobytes()
    # noise=0 なら Generator を 1 度も消費しない
    used = np.random.default_rng(7)
    c = _orca_case(orca_core.OrcaCrowd, pref_noise=0.0, rng=used)
    for _ in range(20):
        c.step(0.05)
    assert used.bit_generator.state == rng.bit_generator.state


def test_separation_pass_resolves_overlap_deterministically():
    """P2 決定 条件3(b): 事後分離パスの後は **min_gap >= 0**(決定論のまま)。

    入力は ORCA の離散化で実際に起きる程度の重なり(ベンチ実測の最深 −0.207 m 相当)を
    16 体の格子配置に与えたもの。既定の反復上限で収束し、体表間が負でなくなること。
    """
    rng = np.random.default_rng(1)
    n = 16
    pos = (np.stack([np.arange(n) % 4 * 0.45, np.arange(n) // 4 * 0.45], axis=1)
           + rng.normal(0.0, 0.02, size=(n, 2)))
    radius = np.full(n, 0.3)
    g0 = orca_core.min_gap(pos, radius)
    assert -0.25 < g0 < 0.0, f"テスト前提が崩れた(重なりの深さ {g0})"
    out, iters, moved = orca_core.separate_positions(pos, radius)   # 既定 64 反復
    assert orca_core.min_gap(out, radius) >= 0.0, "分離後も重なっている"
    assert iters > 0 and moved > 0.0
    out2, iters2, moved2 = orca_core.separate_positions(pos, radius)
    assert out.tobytes() == out2.tobytes() and iters == iters2 and moved == moved2
    # 重なっていない入力は 1 mm も動かさない
    clean = np.array([[0.0, 0.0], [5.0, 0.0]])
    same, it0, mv0 = orca_core.separate_positions(clean, np.full(2, 0.3))
    assert same.tobytes() == clean.tobytes() and it0 == 0 and mv0 == 0.0


def test_separation_pass_reports_truncation_on_impossible_packing():
    """正直な限界: 幾何的に無理な密集は上限で打ち切る(残差は iters==max_iters で判る)。"""
    rng = np.random.default_rng(0)
    pos = rng.normal(0.0, 0.25, size=(24, 2))       # 半径 0.25 の塊に 24 体
    radius = np.full(24, 0.3)
    out, iters, _moved = orca_core.separate_positions(pos, radius, max_iters=8)
    assert iters == 8, "打ち切りが検出できない"
    # 打ち切っても **必ず改善する**(悪化しない)
    assert orca_core.min_gap(out, radius) > orca_core.min_gap(pos, radius)
    # 反復を増やせば収束する
    out2, iters2, _ = orca_core.separate_positions(pos, radius, max_iters=512)
    assert orca_core.min_gap(out2, radius) >= 0.0


def test_radius_margin_and_separation_keep_orca_gap_nonnegative():
    """半径マージン + 分離パスを入れた既定構成で、積分中ずっと重なりが出ない。"""
    crowd = _orca_case(orca_core.OrcaCrowd, pref_noise=0.05,
                       rng=np.random.default_rng(3))
    worst = float("inf")
    for _ in range(200):
        crowd.step(0.05)
        worst = min(worst, crowd.min_gap())
    assert worst >= 0.0, f"重なりが残った: min_gap={worst}"


# =========================================================================== #
# (4) ON: 決定論・resume・呼数・workers
# =========================================================================== #
_ON = dict(n=30, steps=8, zone_specs=[_zone()])


def test_on_same_seed_two_runs_are_identical(tmp_path):
    a = _run(tmp_path, "on_a", **_ON)
    b = _run(tmp_path, "on_b", **_ON)
    assert _kind(a, "zone_gate"), "テスト前提が崩れた(ゾーンを誰も通らない)"
    assert _l1(a) == _l1(b)
    assert P.continuity(a) == P.continuity(b)


def test_on_is_independent_of_worker_count(tmp_path):
    a = _run(tmp_path, "on_w1", **_ON,
             **{"engine.batch_llm.enabled": "true", "engine.batch_llm.workers": 1})
    b = _run(tmp_path, "on_w4", **_ON,
             **{"engine.batch_llm.enabled": "true", "engine.batch_llm.workers": 4})
    assert _l1(a) == _l1(b)


def test_on_changes_the_world_relative_to_off(tmp_path):
    """ON は **世界を変える**(opt-in の世界変更機能であることの明示的な固定)。"""
    off = _run(tmp_path, "cmp_off", n=30, steps=8)
    on = _run(tmp_path, "cmp_on", **_ON)
    assert _l1(off) != _l1(on)
    assert not _kind(off, "zone_gate") and _kind(on, "zone_gate")


def test_on_resume_matches_straight(tmp_path):
    """resume==straight(ゾーン滞在を跨ぐ分割再開。_phys_state の中央管理 + agents pickle)。

    ★ 実測では既定の 1 万体規模でない mock ランだと、ほとんどの個体が 1 step のうちに
      ゾーンを抜けてしまい「step 境界を跨ぐ所有」が起きない(= 検収が空回りする)。
      サブステップ上限 `max_sub_steps` を 400(= 20 秒ぶん)へ絞って、**所有が必ず
      step 境界を跨ぐ**条件を作る。物理の現実性ではなく状態の往復を検収するための設定。
    """
    zspec = _zone(max_sub_steps=400)

    def cfg(name, n_steps, **extra):
        return _cfg(name, n=30, steps=n_steps, zone_specs=[zspec], **extra)

    total = 12
    sdir = tmp_path / "r_straight"
    straight = Simulation(cfg("r_straight", total), out_dir=sdir)
    straight.run()
    assert _kind(straight, "zone_gate")
    d = tmp_path / "r_resumed"
    sim1 = Simulation(cfg("r_resumed", total), out_dir=d)
    # **ゾーン滞在中に分割する**(所有の往復を検収したいので、空でない step を選ぶ)
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
    sim2 = Simulation(cfg("r_resumed", total), out_dir=d)
    sim2.run(resume_from=d)
    for stem in ("l1_events", "l2_metrics", "l3_snapshots"):
        sa = pq.read_table(sdir / f"{stem}.parquet").to_pylist()
        sb = pq.read_table(d / f"{stem}.parquet").to_pylist()
        assert sa == sb, f"{stem} 不一致(physics resume)"


def test_on_llm_call_count_is_k_invariant(tmp_path):
    """物理 ON のまま compute_matched 下で k=free と k=off の generate 呼数が完全一致(R1)。"""
    def run(name, writeback):
        sim = _sim(tmp_path, name, n=30, steps=10, zone_specs=[_zone()],
                   **{"controls.mode": "compute_matched", "k.writeback": writeback})
        sim.llm = _FixedLLM(json.dumps({"action": "speak", "text": "x"},
                                       ensure_ascii=False))
        sim.run()
        return sim
    free = run("k_free", "free")
    off = run("k_off", "off")
    assert free.llm.calls == off.llm.calls > 0


def test_on_l2_columns_appear(tmp_path):
    sim = _run(tmp_path, "on_l2", **_ON)
    tbl = pq.read_table(sim.out_dir / "l2_metrics.parquet")
    for c in ("zone_occupancy", "zone_density_mean",
              "zone_gate_enter_total", "zone_gate_exit_total"):
        assert c in tbl.column_names, f"ON なのに L2 列 {c} が無い"
    enters = [v for v in tbl.column("zone_gate_enter_total").to_pylist() if v is not None]
    assert enters and enters == sorted(enters), "累計列が単調非減少でない(resume 安全性)"


# =========================================================================== #
# (5) 境界プロトコル(guarded)と排他所有
# =========================================================================== #
def test_gate_events_pair_up_and_carry_the_contract(tmp_path):
    sim = _run(tmp_path, "gate_ev", **_ON)
    ev = _kind(sim, "zone_gate")
    ins = [e for e in ev if e.payload["dir"] == "enter"]
    outs = [e for e in ev if e.payload["dir"] == "exit"]
    assert ins and outs
    for e in ins:
        assert e.payload["zone"] == "z1"
        assert e.payload["gate"].startswith("z1:")
        assert e.payload["engine"] == "orca"
        # payload は 3 桁丸めなので丸め誤差ぶんの余裕を見る
        assert e.payload["speed"] <= e.payload["v0"] * 1.3 + 2e-3, \
            "流入初速がグラフ速度の連続引き継ぎ範囲を超えている"
    for e in outs:
        assert e.payload["dwell_s"] >= 0.0
        assert "jump_m" in e.payload
    # 出入りは同数以下(step 内で終わらなかった個体は次 step へ持ち越す)
    assert len(outs) <= len(ins)


def test_owned_agents_do_not_move_on_the_graph(tmp_path):
    """排他所有: その step に物理が所有した個体は、同じ step の move_segment に現れない。

    多くの個体は 1 step のうちにゾーンを抜けるので「step 末に所有が残っている」ことは
    当てにできない。所有の証拠は zone_gate(enter)の側から採る。
    """
    sim = _sim(tmp_path, "excl", n=30, steps=8, zone_specs=[_zone()])
    seen = 0
    for step in range(8):
        before = len(sim.logger.events)
        scheduler.run_step(sim, step)
        new = sim.logger.events[before:]
        entered = {e.agent_id for e in new
                   if e.kind == "zone_gate" and e.payload["dir"] == "enter"}
        owned = {a.id for a in sim.agents if P.owned(a)}
        moved = {e.agent_id for e in new if e.kind == "move_segment"}
        # step 末まで所有が続いた個体は、その step のグラフ移動を 1 件も出していない
        assert not (owned & moved), f"所有下の個体がグラフ移動した: {owned & moved}"
        seen += len(entered)
    assert seen > 0, "所有が一度も起きていない(検収の空回り)"


def test_budget_scale_prevents_double_movement(tmp_path):
    """二重移動の防止(実測で見つけた不具合の回帰固定)。

    step の途中でゾーンを抜けた個体は、その step のグラフ予算を **物理で使った秒数ぶん**
    削られる。OFF(属性が生えない)なら常に 1.0 = 従来の予算と 1 バイトも変わらない。
    """
    sim = _sim(tmp_path, "budget", n=10, steps=1, zone_specs=[_zone()])
    agent = sim.agents[0]
    assert P.budget_scale(sim, agent, 3) == 1.0        # OFF/未使用は素通り
    agent._phys_used_step, agent._phys_used_s = 3, 150.0
    assert abs(P.budget_scale(sim, agent, 3) - 0.75) < 1e-12
    assert P.budget_scale(sim, agent, 4) == 1.0        # 別 step には効かない
    agent._phys_used_s = 10_000.0                      # 使い切りは 0(負にしない)
    assert P.budget_scale(sim, agent, 3) == 0.0


def test_agent_belongs_to_at_most_one_zone(tmp_path):
    """ゾーン間の直接移籍は存在しない(P2 決定 条件6)。"""
    z2 = _zone("z2", engine="sfm",
               polygon=[[80, -140], [180, -140], [180, -40], [80, -40]])
    sim = _sim(tmp_path, "twozone", n=30, steps=8, zone_specs=[_zone(), z2])
    for step in range(8):
        scheduler.run_step(sim, step)
        for a in sim.agents:
            z = getattr(a, "_phys_zone", None)
            assert z in (None, "z1", "z2")
    ev = _kind(sim, "zone_gate")
    # 各個体について enter → exit → enter … の交互列(2 ゾーン同時所有なら崩れる)
    by_agent: dict[int, list[str]] = {}
    for e in sorted(ev, key=lambda e: (e.step, e.agent_id)):
        by_agent.setdefault(e.agent_id, []).append(e.payload["dir"])
    for aid, seq in by_agent.items():
        for a, b in zip(seq, seq[1:]):
            assert a != b, f"agent {aid} のゲート列が交互でない: {seq}"


def test_guarded_gate_blocks_occupied_entry(tmp_path):
    """guarded: 入口が塞がっていれば入れない(= 待たせる)。

    実ランで同一 step に 2 人以上の流入候補が同じ入口へ来るのは稀なので、
    ゲート規則そのものを `_admit` へ直接与えて固定する(id 昇順・先着だけ入場)。
    """
    sim = _sim(tmp_path, "guard", n=10, steps=1, zone_specs=[
        _zone(gate={"min_gap_m": 5.0, "band_m": 3.0, "max_hold_steps": 3,
                    "max_zone_steps": 6, "handover_jump_max_m": 20.0})])
    zone = sim.physcfg["zones"][0]
    st = P._new_state()
    a, b = sim.agents[0], sim.agents[1]
    recs = []
    for agent in (a, b):
        agent.x, agent.y = 0.0, 0.0            # まったく同じ位置に 2 人
        nxt = sorted(sim.city.graph.neighbors(agent.node))[0]
        rec = dict(P._admit_record(sim, zone, agent, [agent.node, nxt], [], 0))
        rec["pos"] = (0.0, 0.0)
        agent._phys_zone = zone.id
        recs.append(rec)
    waiting, members = list(recs), []
    assert P._admit(sim, zone, waiting, members, None, 0.0, 0.0, 0, 0, st) is True
    assert len(members) == 1 and len(waiting) == 1, "占有ガードが効いていない"
    assert members[0]["agent"].id < waiting[0]["agent"].id, "入場順が id 昇順でない"
    # 入口が空けば次が入れる
    members[0]["pos"] = (100.0, 100.0)
    assert P._admit(sim, zone, waiting, members, None, 0.0, 0.0, 0, 0, st) is True
    assert not waiting and len(members) == 2
    assert st["enter_total"] == 2


def test_zone_release_restores_a_consistent_graph_state(tmp_path):
    """流出時にグラフ状態(node/route/edge_offset/xy)が整合して戻る。"""
    sim = _sim(tmp_path, "release", n=30, steps=8, zone_specs=[_zone()])
    for step in range(8):
        scheduler.run_step(sim, step)
        for a in sim.agents:
            if P.owned(a) or not a.route or a.loc != "street":
                continue
            assert sim.city.graph.has_edge(a.node, a.route[0]), \
                "復帰したグラフ状態が地図と矛盾している"
            assert 0.0 <= a.edge_offset <= sim.city.edge_length(a.node, a.route[0]) + 1e-6


# =========================================================================== #
# (6) 境界連続性(ベンチ実測値を上限の根拠に)
# =========================================================================== #
def test_boundary_continuity_within_bench_bounds(tmp_path):
    sim = _run(tmp_path, "cont", n=40, steps=10, zone_specs=[_zone()])
    c = P.continuity(sim)
    assert c["gate_samples"] > 0 and c["interior_samples"] > 0
    dt = 0.05
    # 1) 瞬間移動が無い: 1 サブステップ変位 <= v_max·dt(理論上限。ベンチと同じ機械的証明)
    v_max = 1.4 * 1.3
    assert c["jump_max_m"] <= v_max * dt + 1e-9, c["jump_max_m"]
    # 2) 急停止・急加速: Δv = accel·dt がベンチ(guarded)の水準を超えない
    assert c["gate_accel_p99"] * dt <= BENCH_GATE_DV, c["gate_accel_p99"]
    assert c["interior_accel_p99"] * dt <= BENCH_INT_DV, c["interior_accel_p99"]
    # 3) 振動: guarded は blind(ベンチ最悪値)を超えない
    assert c["gate_reversal_rate"] <= BENCH_BLIND_REVERSAL, c["gate_reversal_rate"]
    assert c["interior_reversal_rate"] <= BENCH_BLIND_REVERSAL, c["interior_reversal_rate"]
    # 4) グラフ復帰の跳び: 宣言した上限(既定 20m)以内
    assert c["handover_jump_max_m"] <= zones.GATE_DEFAULTS["handover_jump_max_m"], \
        c["handover_jump_max_m"]
    # 5) 重なり: ORCA ゾーンでも体表間 >= 0(条件3 の検収)
    assert c["min_gap_m"] is None or c["min_gap_m"] >= 0.0, c["min_gap_m"]
    assert not any(e.payload.get("far") for e in _kind(sim, "zone_gate")), \
        "handover の跳びが上限を超えた個体がある"


def test_reversal_metric_actually_counts_flips(tmp_path):
    """反転率が「常に 0 を返すだけの飾り」でないことを、合成の速度列で固定する。"""
    zone = zones.build_cfg({"zones_enabled": True, "zones": [_zone()]},
                           REPO)["zones"][0]
    cont = P._new_cont()
    st = P._new_state()
    rec = {"seg_dir": (1.0, 0.0), "sign": 0, "speed_sum": 0.0, "speed_n": 0,
           "step_n": 0, "contact_n": 0, "dens_sum": 0.0, "radius": 0.3, "v0": 1.2}

    class _E:
        pos = np.array([[0.0, 0.0]])
        goal = np.array([[10.0, 0.0]])

        def __init__(self):
            self.vel = np.array([[1.0, 0.0]])
    eng = _E()
    pcfg = {"density_radius_m": 2.0, "contact_gap_m": 0.05}
    prev = np.zeros((1, 2))
    for vx in (1.0, -1.0, 1.0, -1.0):
        eng.vel = np.array([[vx, 0.0]])
        P._accumulate(zone, [rec], eng, prev, prev, 0.05, (), cont, st, pcfg)
    assert cont["interior"]["flip"] == 3, cont["interior"]["flip"]


# =========================================================================== #
# (7) 物理 → 知覚(no-fingerprint)
# =========================================================================== #
def test_physics_body_is_measured_and_never_reaches_the_prompt(tmp_path):
    """body の 3 欄が実測で埋まり、かつ **プロンプト文字列は 1 バイトも変わらない**。"""
    sim = _sim(tmp_path, "body_on", n=30, steps=8, zone_specs=[_zone()],
               **{"cognition.contract.enabled": "true"})
    for step in range(8):
        scheduler.run_step(sim, step)
    agent = next((a for a in sim.agents if P.body_of(a)), None)
    assert agent is not None, "ゾーンを通った個体の身体観測が 1 件も無い"
    body = P.body_of(agent)
    assert 0.0 <= body["blocked"] <= 1.0
    assert 0.0 <= body["contact"] <= 1.0
    assert body["local_density"] >= 0.0
    material = scheduler._gather_material(sim, agent, "solo", 8, 80)
    percept = scheduler.build_perception(sim, agent, material)
    assert percept.body["blocked"] == body["blocked"]
    # 契約の中心: prompt_kwargs は material と完全に等しい(body は出ない)。
    # material は _gather_material の生の出力なので、_llm_speak が後から足す 7 欄
    # (interstitial_digest / watch_section / revision_line / engaged_section=第87 /
    #  reject_line=IF-B / snc_section=第117 SNC v2 / attention_section=第142 ATT層B)
    # は比較の対象外にする。
    kw = percept.prompt_kwargs()
    assert {k: v for k, v in kw.items() if k in material} == material
    assert set(kw) - set(material) == {"interstitial_digest", "watch_section",
                                       "revision_line", "engaged_section",
                                       "reject_line", "snc_section",
                                       "attention_section"}
    for f in contract.NON_PROMPT_FIELDS:
        assert f not in contract.PROMPT_KEYWORDS
    assert "blocked" not in percept.text_blob()


def test_same_position_series_gives_identical_prompts(tmp_path):
    """no-fingerprint: **同一の位置系列なら**物理 ON/OFF でプロンプト文字列が一致する。

    物理で位置が変われば見える景色が変わるのは正当な世界変化なので、
    「ON/OFF で L1 が一致する」ことは求めない(それは世界を変えない機能の基準)。
    ここでは「同じ位置・同じ材料から組んだプロンプトが、物理層の有無で変わらない」
    ことを固定する = 物理の存在自体はプロンプトから読み取れない。
    """
    on = _sim(tmp_path, "fp_on", n=20, steps=4, zone_specs=[_zone()],
              **{"cognition.contract.enabled": "true"})
    off = _sim(tmp_path, "fp_off", n=20, steps=4,
               **{"cognition.contract.enabled": "true"})
    for step in range(4):
        scheduler.run_step(on, step)
        scheduler.run_step(off, step)
    # off 側の位置系列を on 側へ写して(= 同一位置系列にして)プロンプトを組む
    for a_on, a_off in zip(sorted(on.agents, key=lambda a: a.id),
                           sorted(off.agents, key=lambda a: a.id)):
        a_on.x, a_on.y, a_on.node = a_off.x, a_off.y, a_off.node
        a_on.building, a_on.floor, a_on.loc = a_off.building, a_off.floor, a_off.loc
    from society.cognition import deliberate
    n_checked = 0
    for a_on, a_off in zip(sorted(on.agents, key=lambda a: a.id),
                           sorted(off.agents, key=lambda a: a.id)):
        m_on = scheduler._gather_material(on, a_on, "solo", 4, 40)
        m_off = scheduler._gather_material(off, a_off, "solo", 4, 40)
        if m_on != m_off:
            continue                     # 記憶・関係など物理以外の差(この検収の対象外)
        p_on = deliberate.build_prompt(
            a_on, **scheduler.build_perception(on, a_on, m_on).prompt_kwargs())
        p_off = deliberate.build_prompt(
            a_off, **scheduler.build_perception(off, a_off, m_off).prompt_kwargs())
        assert p_on == p_off, "物理層の有無でプロンプト文字列が変わった(fingerprint)"
        n_checked += 1
    assert n_checked > 0, "比較できる個体が 1 人も居ない(検収の空回り)"


def test_crowd_channel_override_is_wired_but_off_by_default(tmp_path):
    """ext.crowd 系への差し込みは **配線だけ**(既定 OFF は引数をそのまま返す)。"""
    sim = _sim(tmp_path, "chan", n=20, steps=2, zone_specs=[_zone()])
    agent = sim.agents[0]
    agent._phys_body = {"blocked": 0.5, "contact": 0.1, "local_density": 0.25}
    assert sim.physcfg["perception"]["channels"] is False
    assert P.crowd_override(sim, agent, 3.0) == 3.0
    sim.physcfg["perception"]["channels"] = True
    got = P.crowd_override(sim, agent, 3.0)
    assert abs(got - 0.25 * math.pi * 4.0) < 1e-9


# =========================================================================== #
# (8) 信号ゲート(既存の交差点表との結線)
# =========================================================================== #
def test_signal_gate_limits_admission_to_green(tmp_path):
    """信号のあるゾーンでは赤の間に入場が確定しない(縁石に溜まって青で一斉横断)。"""
    spec = _zone("zsig", signal={"mode": "explicit", "cycle_s": 140.0,
                                 "green_s": 37.0, "flash_s": 10.0, "offset_s": 0.0})
    sim = _sim(tmp_path, "signal", n=40, steps=8, zone_specs=[spec])
    zone = sim.physcfg["zones"][0]
    assert zone.signal["cycle_s"] == 140.0
    for step in range(8):
        scheduler.run_step(sim, step)
    ins = [e for e in _kind(sim, "zone_gate") if e.payload["dir"] == "enter"]
    assert ins, "信号ありゾーンに誰も入れていない(検収の空回り)"
    gate = _sfm.SignalGate(**zone.signal)
    # **入場が確定した瞬間は必ず青(+青点滅)**。赤で入れた個体が 1 人も居ない。
    for e in ins:
        adm_sec = float(e.sim_min) * 60.0 + float(e.payload["wait_s"])             - float(e.payload["waited_steps"]) * float(sim.clock.step_seconds)
        assert gate.can_cross(adm_sec),             f"赤の間に入場した(信号ゲートが効いていない): wait_s={e.payload['wait_s']}"
    assert any(e.payload["wait_s"] > 0.0 for e in ins),         "赤待ちが一度も起きていない(検収の空回り)"


def test_signal_table_lookup_reads_the_existing_crossing_file():
    """既存の交差点表(実測 cycle/green/flash)から結線できる。"""
    path = REPO / "data" / "crossings_shibuya.json"
    rows = json.loads(path.read_text(encoding="utf-8"))["crossings"]
    row = next(r for r in rows if r.get("signal") and r.get("cycle_s"))
    got = zones.load_signal({"mode": "table", "path": "data/crossings_shibuya.json",
                             "crossing_id": int(row["id"]), "offset_s": 0.0}, REPO)
    assert got["cycle_s"] == float(row["cycle_s"])
    assert got["green_s"] == float(row["green_s"])


def test_walls_can_be_loaded_inline_and_from_layered_json():
    inline = zones.load_walls({"mode": "inline", "segments": [[0, 0, 1, 0], [1, 0, 1, 1]]},
                              REPO)
    assert inline == (((0.0, 0.0), (1.0, 0.0)), ((1.0, 0.0), (1.0, 1.0)))
    path = REPO / "data" / "plateau" / "ubld_lod4.json"
    if not path.exists():                       # data/plateau は gitignore 圏
        return
    all_segs = zones.load_walls({"mode": "layered_json",
                                 "path": "data/plateau/ubld_lod4.json"}, REPO)
    one = zones.load_walls({"mode": "layered_json",
                            "path": "data/plateau/ubld_lod4.json", "layers": [0]}, REPO)
    assert len(all_segs) > len(one) > 0
    assert all(len(s) == 2 and len(s[0]) == 2 for s in one)


# =========================================================================== #
# (9) レジストリ宣言
# =========================================================================== #
def test_registry_declares_physics_zones():
    ids = {f.id: f for f in R.FEATURES}
    f = ids["physics.zones_enabled"]
    assert f.repro_tier == "strict"
    assert f.affects_k is False
    assert f.fingerprint_risk == "none"
    assert ids["physics.perception.channels"].repro_tier == "strict"


# =========================================================================== #
# (12) 竹-4 持ち越し②(第86バッチ保守 M-4): 所有中のノード基準同席が古くならない
# =========================================================================== #
def _three_node_path(sim, agent):
    """agent.node から始まる連結 3 ノード(n0-n1-n2)を地図から採る。"""
    g = sim.city.graph
    n0 = agent.node
    for n1 in sorted(g.neighbors(n0)):
        for n2 in sorted(g.neighbors(n1)):
            if n2 != n0:
                return [n0, n1, n2]
    raise AssertionError("3 ノードの連結経路が地図から採れない")


def test_owned_agent_node_advances_across_interior_nodes(tmp_path):
    """ゾーン内で経路ノードを跨いだ時点で `agent.node` が進む。

    入れる前は入場ゲートのノードに固定されたままで、ノード基準の同席
    (cognition/channels._place_key = ("node", agent.node) → ext.crowd_local)が
    「実際には先を歩いている個体」を入口に居ることにして数えていた。
    退場時の射影復元(_release)と同じ「直前に通過したノード」意味論に揃える。
    """
    from society.cognition import channels as channels_mod

    sim = _sim(tmp_path, "nodetrack", n=6, steps=1, zone_specs=[_zone()])
    zone = sim.physcfg["zones"][0]
    agent = sim.agents[0]
    path = _three_node_path(sim, agent)
    agent.x, agent.y = sim.city.node_xy(path[0])
    rec = P._admit_record(sim, zone, agent, path, [], 0)
    rec["waiting"] = False
    agent._phys_zone = zone.id
    members = [rec]
    engine = P._build_engine(zone, members, None)

    assert agent.node == path[0]                    # 入場時は入口ノード
    assert channels_mod._place_key(agent) == ("node", path[0])

    # 中間ノード n1 に到達した状態で通過点前進を回す
    engine.pos[0, 0], engine.pos[0, 1] = sim.city.node_xy(path[1])
    released = P._advance_and_collect(sim, zone, members, engine)
    assert rec["wp"] == 2, "通過点が前進していない(テスト前提が崩れた)"
    assert agent.node == path[1], "所有中に agent.node が進んでいない(竹-4 持ち越し②)"
    assert channels_mod._place_key(agent) == ("node", path[1])
    assert not released                              # まだ経路の途中

    # 終端 n2 に到達 → 退場が確定し、_release がグラフ状態を整合させる
    engine.pos[0, 0], engine.pos[0, 1] = sim.city.node_xy(path[2])
    P._writeback(members, engine)
    released = P._advance_and_collect(sim, zone, members, engine)
    assert released == members
    P._release(sim, zone, agent, 0, 0, P._new_state(), rec=rec)
    assert not P.owned(agent)
    if agent.route:                                  # 復帰後は node/route が地図と整合
        assert sim.city.graph.has_edge(agent.node, agent.route[0])


def test_owned_node_update_is_deterministic_and_draws_no_rng(tmp_path):
    """同 seed 2 ランで zone_gate 列と最終ノードが一致(node 追従は決定論の純関数)。"""
    def once(name):
        sim = _run(tmp_path, name, n=24, steps=6, zone_specs=[_zone()])
        gates = [[e.step, e.agent_id, e.payload["dir"], e.payload["zone"]]
                 for e in _kind(sim, "zone_gate")]
        return gates, {a.id: a.node for a in sim.agents}

    g1, n1 = once("nd1")
    g2, n2 = once("nd2")
    assert g1 == g2 and n1 == n2
    assert g1, "ゾーンを誰も通っていない(テスト前提が崩れた)"
