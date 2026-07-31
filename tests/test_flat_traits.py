"""初期個体差ゼロ対照 + 初期フレーム共変量(第74バッチ IDEA④)のテスト。

受入基準:
- **既定 OFF は 1 バイトも変えない**: L1 バイト一致・LLM 呼数一致・summary のキー構成も同じ。
- ON で **全個体の traits が同一の定数**になる(名簿経路・手続き生成経路の両方)。
  traits から決まる発火個体差(drive_threshold / fire_weight)も潰れる(include_derived)。
- ON/OFF で **build_agent の乱数消費本数が完全に一致**する(CRN ペアの共分散を守る)。
  ★ラン全体の LLM 呼数は一致しない(閾値が変われば誰が発火するかが変わる= 処置そのもの)。
- 初期フレーム共変量は **完全な事後処理**: days=0 は summary にキーを足さず、
  days>0 でも L1/L2/LLM 呼数は 1 バイトも動かない。
- conf/experiments/zero_traits.yaml が run_experiment のマニフェストとして読める。

検証は mock のみ(実 LLM 禁止)。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from society.agents.persona import build_agent
from society.config import load_config
from society.engine.simulation import Simulation
from society.factors.registry import (TRAITS, drive_params, flat_drive_params,
                                      flat_traits, sample_traits)
from society.observer import initial_frame as IF

REPO_ROOT = Path(__file__).resolve().parents[1]

_ON = {"experiment.flat_traits.enabled": "true"}


def _sim(tmp_path, name, n=12, steps=24, **ov):
    dot = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
           "run.seed=42", "model.backend=mock", "observer.snapshot_every=144",
           f"run.out_dir={(tmp_path / 'runs').as_posix()}"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


class _FakeCity:
    """build_agent が触る最小面(POI 無し・住宅無し=後退経路)。"""
    residential_buildings: list = []
    station_node = "n0"

    def pois_by_cat(self, cat):
        return []

    def building(self, bid):
        return {"levels": 1}


class _CountingRng:
    """np.random.Generator のメソッド呼び出し回数を数えるプロキシ。"""

    def __init__(self, g):
        self._g, self.n = g, 0

    def __getattr__(self, attr):
        target = getattr(self._g, attr)
        if not callable(target):
            return target

        def wrapped(*a, **k):
            self.n += 1
            return target(*a, **k)
        return wrapped


def _entry(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    tr = sample_traits(rng)
    thr, fw = drive_params(tr, rng)
    return {"gender": "女", "name": f"名簿{seed}", "age": 30,
            "occupation": "会社員", "traits": tr,
            "drive_threshold": thr, "fire_weight": fw,
            "bedtime_min": 1380, "sleep_steps": 42,
            "has_bicycle": False, "has_car": False, "persona": "説明文"}


def _build(entry, flat, seed=7):
    rng = _CountingRng(np.random.default_rng(seed))
    a = build_agent(0, rng, ["n0", "n1"], _FakeCity(), entry=entry, flat=flat)
    return a, rng.n


# --------------------------------------------------------------------------- #
# 1) 純関数(手計算一致)
# --------------------------------------------------------------------------- #
def test_flat_traits_helper():
    tr = flat_traits(value=0.5)
    assert set(tr) == set(TRAITS) and set(tr.values()) == {0.5}
    tr2 = flat_traits(["a", "b"], 0.25)
    assert tr2 == {"a": 0.25, "b": 0.25}


def test_flat_drive_params_is_the_center_of_drive_params():
    """定数版は drive_params の分布中心(乱数ゼロ)。中央 trait なら 0.60 / 0.50。"""
    thr, fw = flat_drive_params(flat_traits(value=0.5))
    assert (round(thr, 10), round(fw, 10)) == (0.60, 0.50)
    # nfc が高いほど閾値が下がる / locus が高いほど重みが上がる(写像の向きを固定)
    hi = flat_drive_params({"nfc": 1.0, "internal_locus": 1.0})
    assert hi[0] < thr and hi[1] > fw


# --------------------------------------------------------------------------- #
# 2) build_agent(唯一の実装点)
# --------------------------------------------------------------------------- #
def test_off_is_byte_identical_at_build_agent():
    base, n0 = _build(_entry(1), None)
    off, n1 = _build(_entry(1), {"enabled": False, "value": 0.5,
                                 "include_derived": True})
    assert base.traits == off.traits and n0 == n1
    assert (base.drive_threshold, base.fire_weight) == (off.drive_threshold,
                                                        off.fire_weight)


def test_on_flattens_traits_and_derived_roster_path():
    e1, e2 = _entry(1), _entry(2)
    assert e1["traits"] != e2["traits"], "前提: 名簿の traits は個体で違う"
    cfg = {"enabled": True, "value": 0.5, "include_derived": True}
    a1, _ = _build(e1, cfg)
    a2, _ = _build(e2, cfg)
    assert a1.traits == a2.traits == flat_traits(value=0.5)
    assert a1.drive_threshold == a2.drive_threshold
    assert a1.fire_weight == a2.fire_weight
    # 意見感受性(traits→写像)も同じ traits から決まるので一致する
    assert a1.opinion_susceptibility == a2.opinion_susceptibility


def test_include_derived_false_leaves_threshold_individual():
    cfg = {"enabled": True, "value": 0.5, "include_derived": False}
    a1, _ = _build(_entry(1), cfg)
    a2, _ = _build(_entry(2), cfg)
    assert a1.traits == a2.traits, "traits は潰れる"
    assert a1.drive_threshold != a2.drive_threshold, "名簿の発火個体差は残る(不完全な対照)"


def test_draw_count_identical_on_off_both_paths():
    """乱数消費本数が ON/OFF で完全一致(= CRN ペアの共分散が保たれる)。"""
    on = {"enabled": True, "value": 0.5, "include_derived": True}
    off = {"enabled": False, "value": 0.5, "include_derived": True}
    _, n_on = _build(_entry(3), on)
    _, n_off = _build(_entry(3), off)
    assert n_on == n_off > 0, f"名簿経路の draw 数が変わった: {n_on} vs {n_off}"
    # 手続き生成経路(entry=None: sample_traits / drive_params を実際に引く)
    _, p_on = _build(None, on)
    _, p_off = _build(None, off)
    assert p_on == p_off > n_on, f"手続き経路の draw 数が変わった: {p_on} vs {p_off}"


def test_procedural_path_flattens_too():
    on = {"enabled": True, "value": 0.5, "include_derived": True}
    a1, _ = _build(None, on, seed=11)
    a2, _ = _build(None, on, seed=23)
    assert a1.traits == a2.traits == flat_traits(value=0.5)
    assert a1.drive_threshold == a2.drive_threshold


def test_custom_value():
    on = {"enabled": True, "value": 0.8, "include_derived": True}
    a, _ = _build(_entry(1), on)
    assert set(a.traits.values()) == {0.8}


# --------------------------------------------------------------------------- #
# 3) シム配線(OFF は不変・ON は全員同一)
# --------------------------------------------------------------------------- #
def test_sim_off_identical_to_default(tmp_path):
    base = _sim(tmp_path, "ft_base")
    base.run()
    off = _sim(tmp_path, "ft_off", **{"experiment.flat_traits.enabled": "false"})
    off.run()
    assert _l1(base) == _l1(off)
    assert base.llm.calls == off.llm.calls


def test_sim_on_all_agents_share_traits(tmp_path):
    sim = _sim(tmp_path, "ft_on", **_ON)
    sim.run()
    uniq = {tuple(sorted(a.traits.items())) for a in sim.agents}
    assert len(uniq) == 1, "flat_traits ON で traits が個体差を残している"
    assert len({a.drive_threshold for a in sim.agents}) == 1
    assert len({a.fire_weight for a in sim.agents}) == 1
    # traits.json(解析側 r2_traits の入力)にも定数が落ちている
    tj = json.loads((sim.out_dir / "traits.json").read_text(encoding="utf-8"))
    vals = {tuple(sorted((k, v) for k, v in rec.items() if k in TRAITS))
            for rec in tj.values()}
    assert len(vals) == 1


def test_sim_on_runs_and_reports_call_delta(tmp_path):
    """★正直な限界の固定: ON/OFF で LLM 呼数は一致しない(= 発火個体が変わる)。

    一致を要求してしまうと『閾値を潰したのに誰も影響を受けていない』ことになり、
    対照条件として無意味になる。ここでは **完走すること** と **呼が出ていること**
    だけを固定し、呼数差は zero_traits.yaml の controls.mode=compute_matched と
    レポートの併記で扱う。
    """
    off = _sim(tmp_path, "ft_d_off", steps=48)
    off.run()
    on = _sim(tmp_path, "ft_d_on", steps=48, **_ON)
    on.run()
    assert on.llm.calls > 0 and off.llm.calls > 0
    assert len(on.agents) == len(off.agents)


def test_sim_on_is_deterministic(tmp_path):
    a = _sim(tmp_path, "ft_det_a", steps=24, **_ON)
    a.run()
    b = _sim(tmp_path, "ft_det_b", steps=24, **_ON)
    b.run()
    assert _l1(a) == _l1(b)


# --------------------------------------------------------------------------- #
# 4) 初期フレーム共変量(完全な事後処理)
# --------------------------------------------------------------------------- #
def test_initial_frame_off_adds_no_key(tmp_path):
    sim = _sim(tmp_path, "if_off", steps=24)
    s = sim.run()
    assert "initial_frame" not in s
    saved = json.loads((sim.out_dir / "summary.json").read_text(encoding="utf-8"))
    assert "initial_frame" not in saved


def test_initial_frame_on_does_not_touch_l1_or_calls(tmp_path):
    off = _sim(tmp_path, "if_a", steps=48)
    off.run()
    on = _sim(tmp_path, "if_b", steps=48, **{"observer.initial_frame.days": 1})
    s = on.run()
    assert _l1(off) == _l1(on), "事後処理が L1 を動かしてはならない"
    assert off.llm.calls == on.llm.calls
    frame = s["initial_frame"]
    assert frame["days"] == 1 and frame["window_steps"] == 144
    assert frame["n_events"] > 0 and frame["n_agents_active"] > 0
    assert 0.0 <= sum(frame["kind_share"].values()) <= 1.0 + 1e-9
    assert frame["norm_marker_rate"] == 0.0, "mock の発話に規範マーカーは無い"
    assert "mean_grievance" in frame["l2_mean"]


def test_initial_frame_is_deterministic(tmp_path):
    a = _sim(tmp_path, "if_d1", steps=48, **{"observer.initial_frame.days": 1})
    b = _sim(tmp_path, "if_d2", steps=48, **{"observer.initial_frame.days": 1})
    assert a.run()["initial_frame"] == b.run()["initial_frame"]


def test_initial_frame_window_truncates_by_days(tmp_path):
    sim = _sim(tmp_path, "if_win", steps=200,
               **{"observer.initial_frame.days": 1})
    s1 = sim.run()
    full = IF.summarize(str(sim.out_dir), 2, 20, {"definite": [], "agreement": []})
    assert s1["initial_frame"]["n_events"] < full["n_events"], \
        "窓が広がればイベント数は増える(窓の切り出しが効いている)"
    assert IF.summarize(str(sim.out_dir), 0, 20, None) is None


# --------------------------------------------------------------------------- #
# 5) 実験マニフェスト(zero_traits.yaml)
# --------------------------------------------------------------------------- #
def test_zero_traits_manifest_dry_run():
    path = REPO_ROOT / "conf" / "experiments" / "zero_traits.yaml"
    assert path.exists()
    r = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "run_experiment.py"),
         str(path), "--dry-run"],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stderr
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import run_experiment as rx
    man = rx.load_manifest(path)
    assert len(man["conditions"]) == 4
    names = [c["name"] for c in man["conditions"]]
    assert names == ["koff_flatoff", "koff_flaton", "kfree_flatoff", "kfree_flaton"]
    # CRN: 全条件が同じ seed 列を使う / 4 セル × seed 数 の行列になる
    jobs = rx.expand_matrix(man)
    assert len(jobs) == 4 * len(man["seeds"])
    # 対照の座は experiment.flat_traits.enabled だけ(他の差は k のみ)
    flat = {c["name"]: c["overrides"]["experiment.flat_traits.enabled"]
            for c in man["conditions"]}
    assert flat == {"koff_flatoff": False, "koff_flaton": True,
                    "kfree_flatoff": False, "kfree_flaton": True}
    assert man["common"]["controls.mode"] == "compute_matched"
