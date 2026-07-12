"""行動心理プラグイン(config 着脱式・既定 OFF)のテスト = Wave δ Batch P。

ユーザー方針: 文献の数値を固定実装せず、config トグルで着脱し回しながら変える。
4プラグイン(SDT 動機 / 集団効力感 / Lynch 都市イメージ / Searle 制度化)を独立に検証。

鉄則の担保:
- 全 OFF(既定)は既存ゴールデン(test_scenario)が守る。ここでは default==explicit-off の
  L1 バイト一致と、各プラグイン ON で L1 が変わる(=実際に効く)ことを確認する。
- 各プラグイン単独 ON で 20人×48step mock が完走+同 seed で決定論。
"""
from __future__ import annotations

import json
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from society.config import load_config
from society.engine.simulation import Simulation


# --------------------------------------------------------------------------- #
def _sim(tmp_path, name: str, n: int = 6, steps: int = 1, **ov) -> Simulation:
    dotlist = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
               "observer.snapshot_every=1"]
    dotlist += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dotlist), out_dir=tmp_path / name)


def _l1(sim) -> list:
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _run_l1(tmp_path, name: str, n: int = 15, steps: int = 72, **ov) -> list:
    sim = _sim(tmp_path, name, n=n, steps=steps, **ov)
    sim.run()
    return _l1(sim)


# ============================================================ 全 OFF 不変(共通鉄則)
def test_config_declares_all_plugins_off():
    cfg = load_config([])
    for plug in ("sdt", "collective", "lynch", "searle"):
        assert cfg.psych[plug].enabled is False, f"{plug} が既定 OFF でない"


def test_all_plugins_off_is_byte_identical(tmp_path):
    """default(=全 OFF)と明示 off の L1 が完全一致(config 追加+全 seam が no-op)。"""
    base = _run_l1(tmp_path, "off_default")
    explicit = _run_l1(tmp_path, "off_explicit", **{
        "psych.sdt.enabled": "false", "psych.collective.enabled": "false",
        "psych.lynch.enabled": "false", "psych.searle.enabled": "false"})
    assert base == explicit


@pytest.mark.parametrize("flag", ["psych.sdt.enabled", "psych.lynch.enabled"])
def test_generic_plugin_on_changes_stream(tmp_path, flag):
    """SDT・Lynch は任意の run で常時効く(欲求倍率/行き先)= ON で L1 が変わる。

    集団効力感・Searle は**イベント駆動**(提案成立・グループ成功が起きて初めて L1 に
    差が出る)ため、汎用 run では OFF と一致するのが正しい挙動 → それぞれ合成テスト
    (test_collective_group_success_raises_efficacy / test_searle_institution_...)で検証する。
    """
    base = _run_l1(tmp_path, "base_" + flag.split(".")[1])
    on = _run_l1(tmp_path, "on_" + flag.split(".")[1], **{flag: "true"})
    assert base != on, f"{flag} を ON にしても L1 が変わらない(効いていない)"


# ============================================================ 1. SDT 動機
def test_sdt_drive_mods_multiplier():
    """drive.add が drive_mods の不透明倍率を scale に掛ける(reason 文字列しか見ない)。"""
    from society.cognition import drive
    cfg = drive.build_cfg({})

    class A:
        pass

    a = A(); a.drive = 0.0; a._drive_reasons = {}
    a.drive_mods = {"novel_place": 1.5}
    drive.add(a, "novel_place", cfg)
    assert abs(a.drive - drive.DEFAULT_WEIGHTS["novel_place"] * 1.5) < 1e-9
    # 倍率 dict に無い reason は 1.0(現行どおり)
    b = A(); b.drive = 0.0; b._drive_reasons = {}
    b.drive_mods = {"novel_place": 1.5}
    drive.add(b, "sns", cfg)
    assert abs(b.drive - drive.DEFAULT_WEIGHTS["sns"]) < 1e-9
    # drive_mods 属性が無いオブジェクトでも安全(既定 OFF 経路)
    c = A(); c.drive = 0.0; c._drive_reasons = {}
    drive.add(c, "novel_place", cfg)
    assert abs(c.drive - drive.DEFAULT_WEIGHTS["novel_place"]) < 1e-9


def test_sdt_params_present_vary_and_logged(tmp_path):
    """SDT ON: 全員に drive_mods が付き個人差があり、traits.json に sdt が出る。"""
    sim = _sim(tmp_path, "sdt_params", n=20, steps=1,
               **{"psych.sdt.enabled": "true"})
    mods = [a.drive_mods for a in sim.agents]
    assert all(m is not None for m in mods)
    assert any(m["novel_place"] != mods[0]["novel_place"] for m in mods), "個人差なし"
    sim.run()
    traits = json.loads((sim.out_dir / "traits.json").read_text(encoding="utf-8"))
    assert "sdt" in traits["0"] and "novel_place" in traits["0"]["sdt"]


# ============================================================ 2. 集団効力感
def test_collective_group_success_raises_efficacy(tmp_path):
    """合成: グループ内提案の成立 → メンバー全員の collective_efficacy が上がる。"""
    sim = _sim(tmp_path, "col_succ", n=8, steps=1,
               **{"psych.collective.enabled": "true"})
    a0, a1 = sim.agents[0], sim.agents[1]
    sim.tools.groups[0] = {"id": 0, "founder": a0.id, "name": "g",
                           "members": {a0.id, a1.id}}
    sim.tools.member_of[a0.id].add(0)
    sim.tools.member_of[a1.id].add(0)
    before0 = a0.states["collective_efficacy"]
    before1 = a1.states["collective_efficacy"]
    sim.tools.apply(sim, a0, {"type": "propose", "text": "P"}, 0, 0)
    for a in (sim.agents[1], sim.agents[2]):    # 露出 → 署名(閾値 0.25*8=2)
        a.adopted.add("P")
    sim.tools.phase(sim, 1, 10)
    assert sim.tools.proposals[0]["passed"]
    assert a0.states["collective_efficacy"] > before0
    assert a1.states["collective_efficacy"] > before1


def test_collective_ingroup_boost_amplifies_heard():
    """on_ingroup_heard: boost>1 の方が heard_valence 効果が大きく、boost=1.0 は基準と同一。"""
    from society.factors import update
    from society.observer.logger import ObserverLogger
    mags = update.build_magnitudes(None)
    logger = ObserverLogger(Path(tempfile.mkdtemp()))

    def fresh():
        class A:
            pass
        a = A(); a.id = 0; a.x = 0.0; a.y = 0.0
        a.states = {"grievance": 0.2, "efficacy": 0.5}
        return a

    a = fresh()
    d_plain = update.on_ingroup_heard(a, -0.5, 1.0, mags=mags, step=0,
                                      sim_min=0, logger=logger)
    b = fresh()
    d_base = update.on_heard_valence(b, -0.5, mags=mags, step=0,
                                     sim_min=0, logger=logger)
    assert d_plain == d_base                     # boost=1.0 は基準と完全同一(OFF 不変の核)
    c = fresh()
    d_boost = update.on_ingroup_heard(c, -0.5, 1.4, mags=mags, step=0,
                                      sim_min=0, logger=logger)
    assert abs(d_boost) > abs(d_plain) > 0


def test_collective_aggregate_column_gated(tmp_path):
    """mean_collective_efficacy は ON 時のみ L2 列に出る(OFF は None で列なし=L2 不変)。"""
    from society.observer.aggregate import collect
    on = _sim(tmp_path, "col_agg_on", n=6, **{"psych.collective.enabled": "true"})
    assert "mean_collective_efficacy" in collect(on)
    off = _sim(tmp_path, "col_agg_off", n=6)
    assert "mean_collective_efficacy" not in collect(off)
    assert "n_institutions" not in collect(off)


# ============================================================ 3. Lynch 都市イメージ
def test_lynch_destination_favors_landmarks():
    """_lynch_destination が landmark_score の高いノードへ強く寄り、現在地は除外する。"""
    import types

    from society.cognition import routine
    sim = types.SimpleNamespace(
        landmark_score={"a": 1.0, "b": 1.0, "hi": 100.0},
        dests=["a", "b", "hi"])
    agent = types.SimpleNamespace(node="a")
    rng = np.random.default_rng(0)
    counts = Counter(routine._lynch_destination(agent, sim, rng)
                     for _ in range(400))
    assert counts["hi"] > counts["b"], counts
    assert "a" not in counts                     # 現在地は候補から除外


def test_lynch_prompt_injection_optional(tmp_path):
    """build_prompt: familiar_places / institutions を渡した時だけ1行足す(None で不変)。"""
    from society.cognition import deliberate
    ag = _sim(tmp_path, "lyn_prompt", n=2).agents[0]
    on = deliberate.build_prompt(ag, place_name="渋谷", surprise="solo",
                                 nearby_names=[], sim_min=0, step=0,
                                 familiar_places=["渋谷駅", "ハチ公前"],
                                 institutions=["渋谷に緑を増やそう"])
    assert "馴染みの場所: 渋谷駅、ハチ公前" in on
    assert "この街の取り決め: 渋谷に緑を増やそう" in on
    off = deliberate.build_prompt(ag, place_name="渋谷", surprise="solo",
                                  nearby_names=[], sim_min=0, step=0)
    assert "馴染みの場所" not in off and "この街の取り決め" not in off


def test_lynch_biases_arrivals_and_measure_columns(tmp_path):
    """ON の到着分布は landmark に寄る(統計的に緩く)+ measure が観測列を足す。"""
    from society.observer import measure
    common = {"planning.enabled": "false", "world.meal_prob": 0.0}  # 自由行き先を支配的に
    on = _sim(tmp_path, "lyn_on", n=25, steps=96,
              **{"psych.lynch.enabled": "true", **common})
    on.run()
    off = _sim(tmp_path, "lyn_off", n=25, steps=96, **common)
    off.run()
    assert (on.out_dir / "landmark_score.json").exists()
    lm = json.loads((on.out_dir / "landmark_score.json").read_text(encoding="utf-8"))

    def mean_focus(run):
        events = measure.load_events(str(run.out_dir))
        meta = measure.load_agents(str(run.out_dir))
        feats = measure.agent_features(events, meta, landmark_score=lm)
        vals = [f["landmark_focus"] for f in feats
                if f.get("cognitive_map_breadth", 0) > 0]
        return sum(vals) / len(vals) if vals else 0.0

    assert mean_focus(on) > mean_focus(off), "Lynch ON で landmark 集中が上がっていない"
    # landmark_score を渡さない既定呼び出しでは観測列を足さない(後方互換)
    events = measure.load_events(str(on.out_dir))
    meta = measure.load_agents(str(on.out_dir))
    assert all("cognitive_map_breadth" not in f
               for f in measure.agent_features(events, meta))


# ============================================================ 4. Searle 制度化
def test_searle_institution_registered_and_prompt(tmp_path):
    """提案成立 → institution 登録 + institution イベント + プロンプト行 + 発起人効果。"""
    sim = _sim(tmp_path, "searle", n=8, steps=1,
               **{"psych.searle.enabled": "true"})
    author = sim.agents[0]
    own_before = author.states["ownership"]
    text = "渋谷に緑を増やそう"
    sim.tools.apply(sim, author, {"type": "propose", "text": text}, 0, 0)
    for a in (sim.agents[1], sim.agents[2]):
        a.adopted.add(text)
    sim.tools.phase(sim, 1, 10)
    assert sim.tools.proposals[0]["passed"]
    assert len(sim.tools.institutions) == 1
    inst = sim.tools.institutions[0]
    assert inst["created_by"] == author.id and inst["norm_text"] == text
    assert any(e.kind == "institution" for e in sim.logger.events)
    # 発起人効果 = 既存 on_own_adopted の再利用(ownership 上昇)
    assert author.states["ownership"] > own_before
    # 全員平等・最新2件のプロンプト行
    assert sim.tools.institution_lines() == [text]


def test_searle_aggregate_column_gated(tmp_path):
    """n_institutions は Searle ON 時のみ L2 列に出る(OFF は列なし=L2 不変)。"""
    from society.observer.aggregate import collect
    on = _sim(tmp_path, "sea_agg_on", n=6, **{"psych.searle.enabled": "true"})
    assert "n_institutions" in collect(on)
    off = _sim(tmp_path, "sea_agg_off", n=6)
    assert "n_institutions" not in collect(off)


# ============================================================ 各プラグイン単独 ON で完走+決定論
@pytest.mark.parametrize("flag", ["psych.sdt.enabled", "psych.collective.enabled",
                                  "psych.lynch.enabled", "psych.searle.enabled"])
def test_each_plugin_runs_20x48_deterministic(tmp_path, flag):
    key = flag.split(".")[1]
    a = _run_l1(tmp_path, f"run1_{key}", n=20, steps=48, **{flag: "true"})
    b = _run_l1(tmp_path, f"run2_{key}", n=20, steps=48, **{flag: "true"})
    assert a == b, f"{flag} ON が同 seed で非決定論"
    assert len(a) > 50, f"{flag} ON が完走していない"
