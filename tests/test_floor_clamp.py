"""3D-U0(sim 側 floor クランプ `world.floor_clamp`・既定 OFF)のテスト。

正典
  - ``PENDING.md`` 3D-U0「sim 側 floor クランプ(建物階数超の floor が通る = L1 が変わる
    修正。表示側は修正済み)」
  - ``docs/plans/highfidelity-3d-physics-plan.md`` §1 / §4(ユーザー判断 3D-U0 =
    「conf トグル既定 OFF で実装し観察ランで ON」)
  - 表示側の正典実装 = ``scripts/export_3d.py::encode_indoor_w``
    (``tests/test_export3d.py::test_encode_indoor_w_clamps_floor_to_levels``)

守るもの(検収基準の順)
  ① **シムとビューアで別の値にならない**: 純関数 ``floors.clamp_floor`` の出力が
     表示側 ``encode_indoor_w`` の階部分と格子上で完全一致する
  ② OFF(既定)= ゴールデン L1 バイト一致・明示 false と既定が一致・
     ``clamp`` は受け取った値をそのまま返す(型も含めて素通し)
  ③ ON: 全 step・全エージェントの屋内 floor が ``[1, max(1, min(levels, 99))]`` に入る
     (L1 の floor 付きイベント + ラン終了時の状態の両方)。かつ**空振りでない**
     (OFF では実データ由来の逸脱が実際に出ている)
  ④ ON は決定論(同 seed 2 走で L1 一致・乱数を 1 つも引かない)
  ⑤ ON で resume == straight
  ⑥ レジストリ宣言(未宣言トグル 0)・既定 false
  ⑦ ★``agent.floor`` の代入点の網羅を AST で機械固定(将来の代入点追加で漏れたら落ちる)
"""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from society import registry as R
from society.config import load_config
from society.engine import checkpoint, scheduler
from society.engine.simulation import Simulation
from society.world import floors as F

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN = Path(__file__).resolve().parent / "data" / "golden_baseline_l1.json"

# test_traces.py:45 / test_rumors.py と同じ「意図的な既定挙動追加」の中立化
_GOLDEN_NEUTRAL = {"economy.wages.自営": 0, "planning.enabled": "false",
                   "transit_ride.taxi.enabled": "false",
                   "transit_ride.bus.enabled": "false", "rules.enabled": "false"}

OFF = {"world.floor_clamp.enabled": "false"}
ON = {"world.floor_clamp.enabled": "true"}

# 実データ(data/shibuya_osm.json)由来の逸脱が確実に出る規模。seed 42 の 40 体名簿に
# 「2 階建ての建物に floor=6 の職場」を持つ個体が居る(調査実測)。
_SMOKE = {"n_agents": 40, "n_steps": 24}


def _load_export3d():
    """表示側の正典(scripts/export_3d.py)を読み込む(tests/test_export3d.py と同じ作法)。"""
    spec = importlib.util.spec_from_file_location(
        "export_3d_for_floor_clamp", REPO_ROOT / "scripts" / "export_3d.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cfg(name, n_steps=24, n_agents=40, **ov):
    dot = ["run.seed=42", f"run.n_agents={n_agents}", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def _sim(tmp_path, name, n_steps=24, n_agents=40, **ov):
    return Simulation(_cfg(name, n_steps, n_agents, **ov), out_dir=tmp_path / name)


def _run(tmp_path, name, n_steps=24, n_agents=40, **ov):
    sim = _sim(tmp_path, name, n_steps, n_agents, **ov)
    sim.run()
    return sim


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _hi(sim, bld_id: str) -> int:
    """その建物で妥当な最上階 = max(1, min(levels, 99))(表示側 encode_indoor_w と同じ)。"""
    lv = sim.city.building(bld_id).get("levels")
    return max(1, min(int(lv or 1), F.W_FLOOR_MAX))


def _out_of_range(sim) -> list:
    """L1 の floor 付きイベント + ラン終了時の状態から、範囲外の (どこ, 誰, 何) を集める。"""
    bad = []
    for e in sim.logger.events:
        p = e.payload or {}
        b, f = p.get("building"), p.get("floor")
        if not isinstance(b, str) or f is None or not sim.city.has_building(b):
            continue
        if not (1 <= int(f) <= _hi(sim, b)):
            bad.append((e.step, e.agent_id, e.kind, b, int(f), _hi(sim, b)))
    for a in sim.agents:
        if a.building and sim.city.has_building(a.building):
            if not (1 <= int(a.floor) <= _hi(sim, a.building)):
                bad.append(("final", a.id, "state", a.building,
                            int(a.floor), _hi(sim, a.building)))
    return bad


# =========================================================================== #
# (A) 規則の同一性 = シムとビューアで別の値にならない(検収基準 ①)
# =========================================================================== #
def test_rule_matches_viewer_encode_indoor_w():
    """``clamp_floor`` の出力が表示側 ``encode_indoor_w`` の階部分と格子上で完全一致。"""
    e3d = _load_export3d()
    assert F.W_FLOOR_MAX == e3d.W_FLOOR_MAX, "2 桁枠の上限がビューアとずれている"
    bld_idx = {"b": 0}
    for levels in (0, 1, 2, 3, 10, 47, 99, 100, 200):
        for floor in (-99, -2, -1, 0, 1, 2, 3, 7, 10, 47, 98, 99, 100, 12345):
            w = e3d.encode_indoor_w(bld_idx, [levels], "b", floor, {})
            assert F.clamp_floor(floor, levels) == (w - 1000) % 100, (
                f"levels={levels} floor={floor} で sim とビューアの値が食い違う")


def test_clamp_floor_pure_rules():
    """規則の言語化: 地下・0 → 1F / levels 超え → levels / 欠測 levels → 1 / 冪等。"""
    assert F.clamp_floor(6, 2) == 2                 # 階数超え(実データ 20 件)
    assert F.clamp_floor(2, 2) == 2                 # 境界はそのまま
    assert F.clamp_floor(1, 47) == 1
    assert F.clamp_floor(0, 5) == 1                 # 0(屋外の規約)→ 1F
    assert F.clamp_floor(-1, 5) == 1                # 地下 POI(実データ 31 件)→ 1F
    assert F.clamp_floor(-2, 5) == 1
    assert F.clamp_floor(3, None) == 1              # levels 欠測 → 1(捏造しない)
    assert F.clamp_floor(3, 0) == 1
    assert F.clamp_floor(150, 200) == 99            # 2 桁枠の上限
    assert F.clamp_floor("x", 5) == 1               # int にできない値 → 1
    for levels in (1, 3, 47, 200):
        for floor in (-5, 0, 1, 4, 300):
            once = F.clamp_floor(floor, levels)
            assert F.clamp_floor(once, levels) == once, "冪等でない"


# =========================================================================== #
# (B) OFF = 既定(検収基準 ②)
# =========================================================================== #
def test_default_is_off():
    cfg = load_config()
    assert cfg.world.floor_clamp.enabled is False


def test_off_returns_value_untouched():
    """OFF は受け取った値をそのまま返す(int 化もしない = 素通し)。"""
    class _Sim:
        cfg = {"world": {"floor_clamp": {"enabled": False}}}

    sim = _Sim()
    assert F.enabled(sim) is False
    for v in (6, 0, -1, "7", None):
        assert F.clamp(sim, {"levels": 2}, v) is v


def test_on_flag_is_read_from_cfg():
    class _Sim:
        cfg = {"world": {"floor_clamp": {"enabled": True}}}

    sim = _Sim()
    assert F.enabled(sim) is True
    assert F.clamp(sim, {"levels": 2}, 6) == 2


def test_clamp_without_building_is_noop():
    """建物が解決できないときはクランプしない(存在しない建物の階数を捏造しない)。"""
    class _City:
        def has_building(self, bid):
            return False

    class _Sim:
        cfg = {"world": {"floor_clamp": {"enabled": True}}}
        city = _City()

    sim = _Sim()
    assert F.clamp(sim, "居ない建物", 42) == 42
    assert F.clamp(sim, "", 42) == 42


def test_off_matches_default_l1(tmp_path):
    """明示 false と既定(キー既定 false)が L1 バイト一致。"""
    a = _run(tmp_path, "fc_base", **_SMOKE)
    b = _run(tmp_path, "fc_off", **_SMOKE, **OFF)
    assert _l1(a) == _l1(b)


def test_off_matches_golden_l1(tmp_path):
    """OFF(既定)は変更前ゴールデン L1 と一字一句一致(seam が no-op であること)。"""
    golden = json.load(open(GOLDEN, encoding="utf-8"))
    sim = _run(tmp_path, "fc_golden", n_steps=144, n_agents=15, **_GOLDEN_NEUTRAL)
    assert _l1(sim) == golden, "floor_clamp の seam がゴールデンを動かしている"


# =========================================================================== #
# (C) ON = 全 floor が建物の実階数に収まる(検収基準 ③)
# =========================================================================== #
def test_off_run_actually_has_out_of_range_floor(tmp_path):
    """空振り検収の防止: OFF では実データ由来の逸脱が現に L1 に出ている。"""
    sim = _run(tmp_path, "fc_bug", **_SMOKE)
    bad = _out_of_range(sim)
    assert bad, "テストが空振り(この規模では逸脱が出ない = 前提が変わった)"
    kinds = {b[2] for b in bad}
    assert "enter_building" in kinds


def test_on_run_keeps_every_floor_in_building_range(tmp_path):
    """ON: 全 step・全エージェントの屋内 floor が [1, max(1,min(levels,99))] に入る。"""
    sim = _run(tmp_path, "fc_on", **_SMOKE, **ON)
    assert _out_of_range(sim) == []
    # クランプが実際に効いた = OFF と L1 が違う(no-op ではない)
    off = _run(tmp_path, "fc_on_ref", **_SMOKE)
    assert _l1(sim) != _l1(off)


def test_on_run_keeps_floor_in_range_with_natural_start(tmp_path):
    """natural_start(simulation.py の着席)経路でも範囲内(代入点の網羅)。"""
    sim = _run(tmp_path, "fc_on_ns", **_SMOKE, **ON,
               **{"run.start_tod": "00:00", "run.natural_start": "true"})
    assert _out_of_range(sim) == []


# =========================================================================== #
# (D) 決定論(検収基準 ④)
# =========================================================================== #
def test_on_is_deterministic_across_two_runs(tmp_path):
    a = _run(tmp_path, "fc_det_a", **_SMOKE, **ON)
    b = _run(tmp_path, "fc_det_b", **_SMOKE, **ON)
    assert _l1(a) == _l1(b), "floor_clamp ON の決定論が崩れている"


def test_on_does_not_change_enter_building_floor_draw(tmp_path):
    """クランプは既存の抽選を消費しない: 階抽選 stream から引いた値そのものは ON/OFF で同一。

    ``enter_building`` の抽選分岐は 1..levels から引くので**クランプは常に恒等**であり、
    ON でも抽選値がそのまま採用される(= 新しい draw を足していない)ことを、
    同一 (agent, step) の抽選再現で確かめる。
    """
    sim = _run(tmp_path, "fc_draw", **_SMOKE, **ON)
    checked = 0
    for e in sim.logger.events:
        p = e.payload or {}
        if e.kind != "enter_building" or "levels" not in p:
            continue
        rng = sim.hub.stream("floor", e.agent_id, e.step)
        drawn = int(rng.integers(1, int(p["levels"]) + 1))
        assert 1 <= int(p["floor"]) <= max(1, int(p["levels"]))
        assert drawn == F.clamp_floor(drawn, p["levels"]), \
            "抽選分岐でクランプが恒等でない(既定経路の意味が変わっている)"
        checked += 1
    assert checked > 0, "テストが空振り"


# =========================================================================== #
# (E) resume == straight(検収基準 ⑤)
# =========================================================================== #
def test_resume_matches_straight(tmp_path):
    """floor_clamp ON で resume==straight(parquet バイト一致)。"""
    ov = {**ON, "run.start_tod": "00:00", "run.natural_start": "true"}
    split, total = 72, 144
    straight_dir = tmp_path / "fc_straight"
    straight = Simulation(_cfg("fc_straight", total, 20, **ov), out_dir=straight_dir)
    straight.run()

    d = tmp_path / "fc_resumed"
    sim1 = Simulation(_cfg("fc_resumed", split, 20, **ov,
                           **{"observer.checkpoint_every": split}), out_dir=d)
    for step in range(split):
        scheduler.run_step(sim1, step)
    checkpoint.save(sim1, split, d / "checkpoint" / f"ckpt-{split:06d}.pkl.gz")
    sim1.logger.flush_segment()
    sim2 = Simulation(_cfg("fc_resumed", total, 20, **ov,
                           **{"observer.checkpoint_every": split}), out_dir=d)
    sim2.run(resume_from=d)
    for stem in ("l1_events", "l2_metrics", "l3_snapshots"):
        sa = pq.read_table(straight_dir / f"{stem}.parquet").to_pylist()
        sb = pq.read_table(d / f"{stem}.parquet").to_pylist()
        assert sa == sb, f"{stem} 不一致(floor_clamp resume)"
    assert _out_of_range(sim2) == []


# =========================================================================== #
# (F) レジストリ(検収基準 ⑥)
# =========================================================================== #
def test_registry_declares_floor_clamp():
    feat = [f for f in R.FEATURES if f.id == "world.floor_clamp.enabled"]
    assert len(feat) == 1, "レジストリに宣言が無い / 重複している"
    f = feat[0]
    assert f.repro_tier == "strict"          # 再現性の道具 = verify でも落とさない
    assert f.affects_k is False
    assert f.fingerprint_risk == "none"


def test_no_undeclared_toggles():
    assert R.undeclared_toggles(load_config()) == []


# =========================================================================== #
# (G) ★代入点の網羅を AST で機械固定(検収基準 ⑦)
# =========================================================================== #
@pytest.mark.parametrize("rel", ["src/society/engine/scheduler.py",
                                 "src/society/engine/simulation.py"])
def test_every_floor_assignment_goes_through_clamp_or_constant(rel):
    """``<obj>.floor = X`` の右辺は **clamp 呼び出し**か**定数**でなければならない。

    定数が許されるのは ``= 0``(退館 = 屋外の規約)と ``= 1``(lodging チェックイン。
    1 はどんな建物でも妥当)だけ。将来 floor の代入点が増えたとき、クランプを
    通し忘れたらここで落ちる = 「表示側とだけ規則が一致している」状態への逆戻りを防ぐ。
    """
    tree = ast.parse((REPO_ROOT / rel).read_text(encoding="utf-8"))
    seen = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            if not (isinstance(tgt, ast.Attribute) and tgt.attr == "floor"):
                continue
            seen += 1
            v = node.value
            if isinstance(v, ast.Constant):
                assert v.value in (0, 1), f"{rel}: floor へ想定外の定数 {v.value!r}"
                continue
            assert isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) \
                and v.func.attr == "clamp", \
                (f"{rel}:{node.lineno} の floor 代入がクランプを通っていない"
                 f"(3D-U0: 代入点は floors.clamp に集約する)")
    assert seen > 0, f"{rel} に floor の代入が見つからない(テストが空振り)"


def test_floor_is_assigned_only_in_the_two_known_files():
    """``.floor`` の代入は engine の 2 ファイルにしか無い(上の網羅テストの前提の固定)。

    別モジュールが屋内位置の階を書き始めたらここで落ちる = クランプの網羅が
    静かに破れることを防ぐ(``home_floor`` / ``work_floor`` は名簿の値なので対象外)。
    """
    allowed = {"scheduler.py", "simulation.py"}
    offenders = []
    for path in sorted((REPO_ROOT / "src" / "society").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for tgt in node.targets:
                if isinstance(tgt, ast.Attribute) and tgt.attr == "floor" \
                        and path.name not in allowed:
                    offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == [], f"未クランプの floor 代入点: {offenders}"


def test_clamp_module_has_no_randomness():
    """クランプ機構に乱数・時刻の識別子が 1 つも無い(純関数であることの静的固定)。"""
    src = (REPO_ROOT / "src" / "society" / "world" / "floors.py").read_text(
        encoding="utf-8")
    tree = ast.parse(src)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for bad in ("rng", "random", "integers", "uniform", "stream", "hub", "shuffle"):
        assert bad not in names, f"floors.py に乱数由来の識別子 {bad} がある"
