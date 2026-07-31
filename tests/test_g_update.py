"""第82バッチ: 感度 g / 閾値 θ の更新則 + 初期値条件 F/N/P のテスト。

正典: docs/plans/source/cognition-design-record.md §2.5(g の更新則)・§2.6(θ の恒常性)・
      §2.7(初期値の実験条件化)・§8(観測量)・§9(% で測らない)。

守るもの(検収基準の順)
  (1) 既定 OFF = サイドカー不在・新 kind ゼロ件・g 状態がエージェントに生えない
  (2) ★更新則の 3 性質: **慣れ**(反復で g 低下)/ **感作**(結果を伴う反復で g 上昇)/
      **引き戻し**(λ で g⁰ へ)。純関数のユニットテストで固定する
  (3) ★θ の恒常性が**日オーダー**で動く(1 日の途中では 1 度も動かない)
  (4) F/N/P: 3 条件で g(0) 分布が意図どおり / ノイズの draw 数が条件間で不変(CRN)
  (5) 決定論: 同 seed 2 ラン一致・resume==straight
  (6) 宣言: registry(strict / affects_k=False)・timeconv・manifest
  (7) scripts/analyze_g.py が分散分解を出す
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from society import registry as R
from society import timeconv as T
from society.cognition import fire as F
from society.cognition import plasticity as P
from society.config import load_config
from society.engine import checkpoint, scheduler
from society.engine.simulation import Simulation
from society.factors import registry as FR

REPO_ROOT = Path(__file__).resolve().parents[1]
FIRE = {"cognition.fire.enabled": "true"}
GON = {**FIRE, "cognition.g_update.enabled": "true"}
NEW_KINDS = {"cog_theta"}


def _cfg(name: str, n_steps: int = 24, n_agents: int = 24, **ov):
    dot = ["run.seed=42", f"run.n_agents={n_agents}", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def _run(tmp_path, name: str, n_steps: int = 24, n_agents: int = 24, **ov):
    out = tmp_path / name
    sim = Simulation(_cfg(name, n_steps, n_agents, **ov), out_dir=out)
    sim.run()
    return sim, out


def _l1(sim, skip=frozenset()):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events if e.kind not in skip]


def _rows(out_dir: Path, stem: str):
    path = out_dir / f"{stem}.parquet"
    return pq.read_table(path).to_pylist() if path.exists() else None


class _CountingHub:
    def __init__(self, inner):
        self._inner = inner
        self.counts: dict[str, int] = {}

    def stream(self, *key):
        name = str(key[0]) if key else ""
        self.counts[name] = self.counts.get(name, 0) + 1
        return self._inner.stream(*key)

    def __getattr__(self, item):
        return getattr(self._inner, item)


# --------------------------------------------------------------------------- #
# (A) 既定 OFF(検収基準 1)
# --------------------------------------------------------------------------- #
def test_shipped_default_is_off():
    assert bool(load_config().cognition.g_update.enabled) is False


def test_off_leaves_no_trace(tmp_path):
    sim, out = _run(tmp_path, "g_off", 12, 12, **FIRE)
    assert P.enabled(sim) is False
    assert sim.cognition_g_sc is None
    assert not (out / "cognition_g.parquet").exists()
    assert not [e for e in sim.logger.events if e.kind in NEW_KINDS]
    for agent in sim.agents:
        assert getattr(agent, "_fire_g", None) is None
    man = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert "g_update" not in man["cognition"]


def test_g_update_needs_fire_on(tmp_path):
    sim = Simulation(_cfg("g_nofire", 1, 4, **{"cognition.g_update.enabled": "true"}),
                     out_dir=tmp_path / "g_nofire")
    assert P.enabled(sim) is False


def test_theta_is_unchanged_when_off(tmp_path):
    """OFF では θ が第81 と同じ値(個体倍率が掛からない)。"""
    sim = Simulation(_cfg("g_theta_off", 1, 4, **FIRE), out_dir=tmp_path / "g_theta_off")
    agent = sim.agents[0]
    assert F.theta_of(sim, F.WALKING, agent) == F.theta_of(sim, F.WALKING)


# --------------------------------------------------------------------------- #
# (B) ★更新則の 3 性質(検収基準 2)。設計 §2.5 / Groves & Thompson 1970
# --------------------------------------------------------------------------- #
_KW = {"eta": 0.1, "rho": 0.5, "lam": 0.05, "lo": 0.0, "hi": 5.0}


def test_habituation_lowers_g_under_mere_repetition():
    """★慣れ: **単なる反復**(誤差は来るが結果は伴わない)で g は下がる。"""
    g, g0 = 1.0, 1.0
    seq = [g]
    for _ in range(20):
        g = P.g_step(g, g0, r=0.0, ebar=1.0, **_KW)     # ē>0, r=0 = 反復するだけ
        seq.append(g)
    assert seq[-1] < seq[0], "反復しても g が下がらない(慣れの項が効いていない)"
    assert all(b <= a for a, b in zip(seq, seq[1:])), "単調に下がっていない"


def test_sensitization_raises_g_when_the_repetition_has_consequences():
    """★感作: **結果を伴う反復**(r>0)では同じ ē でも g が上がる。

    設計 §2.5「**単なる反復は慣れ、結果を伴う反復は感作**。したがって更新を生の予測誤差
    だけで駆動してはならない」。慣れと感作の**差が r だけ**であることを固定する。
    """
    habituated = P.g_step(1.0, 1.0, r=0.0, ebar=1.0, **_KW)
    sensitized = P.g_step(1.0, 1.0, r=2.0, ebar=1.0, **_KW)
    assert habituated < 1.0 < sensitized, \
        "同じ反復量なのに『結果を伴う』ほうが上がっていない"
    # r = ρ·ē のちょうど釣り合う点では引き戻しだけが残る(g=g0 なら不動点)
    assert P.g_step(1.0, 1.0, r=0.5, ebar=1.0, **_KW) == pytest.approx(1.0)


def test_pullback_returns_g_to_the_persona_baseline():
    """★引き戻し: r=0・ē=0 なら g は g⁰ へ幾何収束する(性格の永続性)。"""
    g, g0 = 3.0, 1.0
    for _ in range(300):
        g = P.g_step(g, g0, r=0.0, ebar=0.0, **_KW)
    assert g == pytest.approx(g0, abs=1e-3)
    # 下から近づく場合も同じ
    g = 0.2
    for _ in range(300):
        g = P.g_step(g, g0, r=0.0, ebar=0.0, **_KW)
    assert g == pytest.approx(g0, abs=1e-3)


def test_without_pullback_history_overwrites_personality():
    """λ=0 だと g⁰ へ戻らない(設計 §2.5「この項がないと全員収束する」の反証)。"""
    kw = dict(_KW, lam=0.0)
    g = 3.0
    for _ in range(300):
        g = P.g_step(g, 1.0, r=0.0, ebar=0.0, **kw)
    assert g == pytest.approx(3.0), "λ=0 なのに基準値へ引き戻されている"


def test_g_is_bounded():
    assert P.g_step(10.0, 1.0, r=100.0, ebar=0.0, **_KW) <= _KW["hi"]
    assert P.g_step(0.0, 0.0, r=-100.0, ebar=0.0, **_KW) >= _KW["lo"]


def _drive_plasticity(sim, agent, *, outcome_seq, contrib, n_events, period=30):
    """`ensure → on_event → observe_tick` の**実 API だけ**で履歴を進めるヘルパ。

    `outcome_seq(sim_min)` が「うまくいっている度合い」を返す(= 世界の代わり)。
    純関数テストではなく**配線を通した**慣れ/感作の検査に使う。
    """
    P.ensure(sim, agent)
    calls = {"n": 0}

    def _fake_outcome(a):
        return float(outcome_seq(calls["n"]))

    orig = P._outcome
    P._outcome = _fake_outcome
    try:
        for i in range(n_events):
            calls["n"] = i
            sim_min = i * period
            P.observe_tick(sim, agent, {k: 1.0 for k in contrib}, sim_min)
            P.on_event(sim, agent, dict(contrib), sim_min, F.SALIENCE)
    finally:
        P._outcome = orig


def test_sensitization_shows_up_through_the_real_wiring(tmp_path):
    """★配線込みのオラクル: **結果が伴う**発火を繰り返すと g が g⁰ を超えて上がる。

    純関数の性質(上)とは別に、`on_event`(窓を開く)→ `observe_tick`(窓を閉じて
    credit を積む)→ 次の `on_event`(credit を消費)の**受け渡しが実際に成立して
    いる**ことを固定する。ここが切れていると感作の項は永久に 0 のままになる
    (実装中に実際に踏んだ退行: 窓を 1 本しか持たないと次の発火が前の窓を上書きする)。
    """
    sim = Simulation(_cfg("sens", 1, 6, **GON,
                          **{"cognition.g_update.rho": 0.0}),   # 慣れを止めて感作だけを見る
                     out_dir=tmp_path / "sens")
    agent = sim.agents[0]
    P.ensure(sim, agent)
    idx = sorted(agent._fire_g)[0]
    g0 = agent._fire_g0[idx]
    # 発火のたびに「うまくいっている度合い」が単調に上がる世界
    _drive_plasticity(sim, agent, outcome_seq=lambda i: 0.5 + 0.02 * i,
                      contrib={idx: 2.0}, n_events=40)
    assert agent._fire_g[idx] > g0, "結果を伴う反復で g が g⁰ を超えていない(感作が死んでいる)"


def test_habituation_shows_up_through_the_real_wiring(tmp_path):
    """★配線込み: **結果を伴わない**反復では g が g⁰ を下回る(慣れ)。"""
    sim = Simulation(_cfg("habi", 1, 6, **GON), out_dir=tmp_path / "habi")
    agent = sim.agents[0]
    P.ensure(sim, agent)
    idx = sorted(agent._fire_g)[0]
    g0 = agent._fire_g0[idx]
    _drive_plasticity(sim, agent, outcome_seq=lambda _i: 0.5,   # 何も変わらない世界
                      contrib={idx: 2.0}, n_events=40)
    assert agent._fire_g[idx] < g0, "単なる反復で g が下がっていない(慣れが効いていない)"


def test_credit_windows_are_not_overwritten_by_later_events(tmp_path):
    """★退行検知: 発火間隔 < 窓 でも、開いた窓が**全部**満期を迎えること。"""
    sim = Simulation(_cfg("elig", 1, 6, **GON,
                          **{"cognition.g_update.r_window_min": 60}),
                     out_dir=tmp_path / "elig")
    agent = sim.agents[0]
    P.ensure(sim, agent)
    idx = sorted(agent._fire_g)[0]
    for i in range(4):                     # 10 分間隔 = 窓(60 分)より短い
        P.on_event(sim, agent, {idx: 1.0}, i * 10, F.SALIENCE)
    assert len(agent._fire_pending) == 4, "後の発火が前の credit 窓を上書きしている"
    P.observe_tick(sim, agent, {}, 200)    # 全部満期
    assert agent._fire_pending == []


def test_pending_windows_are_bounded(tmp_path):
    sim = Simulation(_cfg("cap", 1, 6, **GON,
                          **{"cognition.g_update.max_pending": 3,
                             "cognition.g_update.r_window_min": 10_000}),
                     out_dir=tmp_path / "cap")
    agent = sim.agents[0]
    P.ensure(sim, agent)
    idx = sorted(agent._fire_g)[0]
    for i in range(20):
        P.on_event(sim, agent, {idx: 1.0}, i, F.SALIENCE)
    assert len(agent._fire_pending) == 3


def test_ebar_halflife_is_dt_invariant():
    """ē の EMA は**分**の半減期で持つ(Δt を変えても同じ実時間の平均になる)。"""
    a10 = P.ema_alpha(10.0, 120.0)
    a5 = P.ema_alpha(5.0, 120.0)
    # Δt=5 を 2 回適用 == Δt=10 を 1 回適用(残存率で比較)
    assert (1 - a5) ** 2 == pytest.approx(1 - a10)


# --------------------------------------------------------------------------- #
# (C) ★θ の恒常性は日オーダー(検収基準 3)。設計 §2.6
# --------------------------------------------------------------------------- #
def test_theta_step_moves_toward_the_target():
    up = P.theta_step(1.0, fbar=30.0, target=8.0, mu=0.02, lo=0.25, hi=4.0)
    down = P.theta_step(1.0, fbar=1.0, target=8.0, mu=0.02, lo=0.25, hi=4.0)
    assert up > 1.0 > down, "発火過多で閾値が上がり、過少で下がること"
    assert P.theta_step(1.0, 8.0, 8.0, mu=0.02, lo=0.25, hi=4.0) == pytest.approx(1.0)
    assert P.theta_step(1.0, 1e6, 8.0, mu=0.02, lo=0.25, hi=4.0) == 4.0   # 上限


def test_theta_does_not_move_within_a_day(tmp_path):
    """★「μ の時定数は日オーダー」= 1 日の途中では θ が 1 度も動かない。

    設計 §2.6:「短くすると発火率が定数に張り付き、閾値方式を選んだ意味(発火数の
    変動を観測する)が消える」。日内で動かないことがその実装上の意味。
    """
    d = tmp_path / "th_day"
    sim = Simulation(_cfg("th_day", 100, 20, **GON), out_dir=d)
    seen: list[float] = []
    for step in range(100):               # 100 step × 10 分 = 1000 分 < 1 日
        scheduler.run_step(sim, step)
        seen.append(float(sim.agents[0]._fire_theta_m))
    assert len(set(seen)) == 1, f"1 日の途中で θ 倍率が動いた: {sorted(set(seen))[:5]}"
    assert not [e for e in sim.logger.events if e.kind == "cog_theta"], \
        "日境界を跨いでいないのに恒常性が適用された"


def test_theta_moves_at_the_day_boundary(tmp_path):
    """日境界を跨げば θ が 1 回だけ動く(全員ぶん cog_theta が 1 件ずつ出る)。"""
    sim, _out = _run(tmp_path, "th_bnd", 288, 20, **GON)
    events = [e for e in sim.logger.events if e.kind == "cog_theta"]
    assert events, "日境界を跨いだのに恒常性が 1 度も適用されていない"
    by_day = {}
    for e in events:
        by_day.setdefault(e.payload["day"], set()).add(e.agent_id)
    for day, ids in by_day.items():
        assert len(ids) == len(sim.agents), f"day={day} で一部の個体しか更新されていない"
    assert all(e.payload["theta_mult"] > 0 for e in events)


def test_theta_counts_only_the_firings_it_gates(tmp_path):
    """f̄ は **θ が門番をしている驚き発火**だけを数える(周期発火は θ で減らせない)。"""
    sim = Simulation(_cfg("th_cnt", 1, 6, **GON), out_dir=tmp_path / "th_cnt")
    agent = sim.agents[0]
    P.ensure(sim, agent)
    P.on_event(sim, agent, {}, 0, F.PERIODIC)
    P.on_event(sim, agent, {}, 0, F.SOCIAL)
    assert agent._fire_day_n == 0
    P.on_event(sim, agent, {}, 0, F.SALIENCE)
    assert agent._fire_day_n == 1


# --------------------------------------------------------------------------- #
# (D) ★初期値条件 F / N / P(検収基準 4)。設計 §2.7
# --------------------------------------------------------------------------- #
def _g0_matrix(sim) -> list[list[float]]:
    return [[v for _k, v in sorted(a._fire_g_init.items())]
            for a in sim.agents if getattr(a, "_fire_g", None) is not None]


def _prime(tmp_path, name, mode, n_agents=24, **ov):
    """1 step だけ回して g(0) を確定させる。"""
    sim = Simulation(_cfg(name, 1, n_agents, **GON,
                          **{"experiment.g_init.mode": mode, **ov}),
                     out_dir=tmp_path / name)
    scheduler.run_step(sim, 0)
    return sim


def test_flat_condition_gives_every_agent_the_same_g0(tmp_path):
    """条件 F: 全員 g⁰ 同一(純粋対照)。"""
    sim = _prime(tmp_path, "gi_F", "flat")
    rows = _g0_matrix(sim)
    assert rows
    assert all(r == rows[0] for r in rows), "flat なのに個体差がある"
    assert len(set(rows[0])) == 1, "flat なのにチャンネル間で差がある"


def test_noise_condition_adds_heterogeneity_without_persona(tmp_path):
    """条件 N: フラット + ノイズ(**異質性のみ**)。"""
    flat = _g0_matrix(_prime(tmp_path, "gi_F2", "flat"))
    noise = _g0_matrix(_prime(tmp_path, "gi_N", "noise"))
    import statistics as st
    v_flat = st.pvariance([r[0] for r in flat])
    v_noise = st.pvariance([r[0] for r in noise])
    assert v_flat == 0.0 and v_noise > 0.0, "noise 条件で個体差が生まれていない"
    assert st.mean([x for r in noise for x in r]) == pytest.approx(
        st.mean([x for r in flat for x in r]), rel=0.25), \
        "ノイズが平均をずらしている(異質性だけを足す条件ではない)"


def test_persona_condition_is_heterogeneous_and_channel_structured(tmp_path):
    """条件 P: ペルソナ由来(個体差 + **源ごとの配分**が付く)。"""
    sim = _prime(tmp_path, "gi_P", "persona")
    rows = _g0_matrix(sim)
    import statistics as st
    assert st.pvariance([r[0] for r in rows]) > 0.0, "persona なのに個体差がない"
    assert any(len(set(r)) > 1 for r in rows), \
        "persona なのに全チャンネル同値(源ごとの配分が効いていない)"


def test_sigma0_scales_the_noise(tmp_path):
    import statistics as st
    small = _g0_matrix(_prime(tmp_path, "gi_s1", "noise",
                              **{"experiment.g_init.sigma0": 0.05}))
    big = _g0_matrix(_prime(tmp_path, "gi_s2", "noise",
                            **{"experiment.g_init.sigma0": 0.50}))
    assert st.pvariance([r[0] for r in big]) > st.pvariance([r[0] for r in small])


def test_noise_draw_count_is_identical_across_conditions(tmp_path):
    """★CRN: ノイズの draw 数が 3 条件で不変(条件差から乱数消費の交絡を消す)。"""
    counts = {}
    for mode in P.G_INIT_MODES:
        sim = Simulation(_cfg(f"crn_{mode}", 6, 16, **GON,
                              **{"experiment.g_init.mode": mode}),
                         out_dir=tmp_path / f"crn_{mode}")
        sim.hub = _CountingHub(sim.hub)
        sim.run()
        counts[mode] = sim.hub.counts.get(P.NOISE_STREAM, 0)
    assert counts["persona"] == counts["flat"] == counts["noise"] > 0, \
        f"条件間で g_init stream の消費が違う: {counts}"


def test_unknown_init_mode_is_rejected():
    with pytest.raises(ValueError):
        P.build_init_cfg({"mode": "telepathy"})


def test_persona_mapping_is_deterministic_and_zero_random():
    """ペルソナ →(g⁰, η, λ, θ₀)の写像は決定論・乱数ゼロ(factors 層の純関数)。"""
    traits = {"nfc": 0.8, "risk_tolerance": 0.3, "internal_locus": 0.6}
    a = FR.cognition_params(traits)
    b = FR.cognition_params(dict(traits))
    assert a == b
    assert set(a) >= {"g0", "eta", "lam", "theta0",
                      "bias_external", "bias_body", "bias_prediction"}
    assert FR.flat_cognition_params(0.5) == FR.cognition_params(
        {"nfc": 0.5, "risk_tolerance": 0.5, "internal_locus": 0.5})


# --------------------------------------------------------------------------- #
# (E) 決定論・resume(検収基準 5)
# --------------------------------------------------------------------------- #
def test_same_seed_two_runs_are_identical(tmp_path):
    a, _oa = _run(tmp_path, "g_det_a", 36, 24, **GON)
    b, _ob = _run(tmp_path, "g_det_b", 36, 24, **GON)
    assert _l1(a) == _l1(b)
    assert [sorted(x._fire_g.items()) for x in a.agents] == \
           [sorted(x._fire_g.items()) for x in b.agents]


def test_resume_matches_straight(tmp_path):
    straight = tmp_path / "gs"
    Simulation(_cfg("gs", 180, 20, **GON), out_dir=straight).run()

    d = tmp_path / "gr"
    every = {"observer.checkpoint_every": 90}
    sim1 = Simulation(_cfg("gr", 90, 20, **GON, **every), out_dir=d)
    for step in range(90):
        scheduler.run_step(sim1, step)
    checkpoint.save(sim1, 90, d / "checkpoint" / "ckpt-000090.pkl.gz")
    sim1.logger.flush_segment()
    sim2 = Simulation(_cfg("gr", 180, 20, **GON, **every), out_dir=d)
    sim2.run(resume_from=d)

    assert _rows(straight, "l1_events") == _rows(d, "l1_events"), "resume≠straight (L1)"
    assert _rows(straight, "l2_metrics") == _rows(d, "l2_metrics"), "resume≠straight (L2)"


def test_checkpoint_carries_the_day_boundary(tmp_path):
    """日境界を保存しないと resume 直後に『初日扱い』へ戻り 1 日分の恒常性が飛ぶ。"""
    d = tmp_path / "gck"
    sim = Simulation(_cfg("gck", 150, 12, **GON), out_dir=d)
    for step in range(150):
        scheduler.run_step(sim, step)
    saved = sim._g_day
    assert saved >= 0
    path = checkpoint.save(sim, 150, d / "checkpoint" / "ckpt-000150.pkl.gz")
    sim2 = Simulation(_cfg("gck", 300, 12, **GON), out_dir=tmp_path / "gck2")
    checkpoint.load(sim2, path)
    assert sim2._g_day == saved


# --------------------------------------------------------------------------- #
# (F) 軌跡サイドカー(設計 §2.7「g_i(0) と g の全軌跡をログする」/ §8)
# --------------------------------------------------------------------------- #
def test_sidecar_records_the_whole_trajectory(tmp_path):
    sim, out = _run(tmp_path, "g_sc", 36, 12, **GON)
    table = pq.read_table(out / "cognition_g.parquet")
    assert table.num_rows == 36 * 12
    names = set(table.column_names)
    assert {"theta_mult", "fbar", "fired_today"} <= names
    assert any(c.startswith("g_") for c in names)
    assert any(c.startswith("g0_") for c in names)
    rows = table.to_pylist()
    # g(0) 列は全 step で不変(初期値の凍結)
    per_agent: dict[int, set] = {}
    for r in rows:
        key = tuple(v for k, v in sorted(r.items()) if k.startswith("g0_"))
        per_agent.setdefault(r["agent_id"], set()).add(key)
    assert all(len(v) == 1 for v in per_agent.values()), "g(0) が途中で書き換わった"


def test_sidecar_can_be_switched_off(tmp_path):
    _sim, out = _run(tmp_path, "g_sc_off", 12, 8, **GON,
                     **{"cognition.g_update.log_every_steps": 0})
    assert not (out / "cognition_g.parquet").exists()


def test_g_variance_actually_moves_in_a_run(tmp_path):
    """★§8「g ベクトルの時間発展。分散の拡大 = 役割分化の一次証拠」が観測できる形で出る。"""
    import statistics as st
    sim, _out = _run(tmp_path, "g_move", 288, 30, **GON)
    dgs = [[g - g0 for (_k, g), (_k0, g0)
            in zip(sorted(a._fire_g.items()), sorted(a._fire_g_init.items()))]
           for a in sim.agents if getattr(a, "_fire_g", None) is not None]
    flat = [x for row in dgs for x in row]
    assert any(abs(x) > 1e-6 for x in flat), "g が 1 度も動いていない"
    assert st.pvariance(flat) > 0.0, "Δg に個体差が生まれていない"


# --------------------------------------------------------------------------- #
# (G) 宣言(検収基準 6)
# --------------------------------------------------------------------------- #
def test_feature_is_declared_in_the_registry():
    feature = {f.id: f for f in R.FEATURES}.get("cognition.g_update.enabled")
    assert feature is not None
    assert feature.repro_tier == "strict", "LLM は g/θ に一切関与しない(§2.8)"
    assert feature.affects_k is False, "generate() の呼び出し点を足しも減らしもしない"
    assert feature.fingerprint_risk == "none", "プロンプトを 1 バイトも変えない"


def test_new_conf_keys_are_classified_in_timeconv():
    for key in ("cognition.g_update.eta_scale", "cognition.g_update.lam_scale",
                "cognition.g_update.rho", "cognition.g_update.ebar_halflife_min",
                "cognition.g_update.r_window_min", "cognition.g_update.r_gain",
                "cognition.g_update.max_pending",
                "cognition.g_update.g_min", "cognition.g_update.g_max",
                "cognition.g_update.theta_mu",
                "cognition.g_update.theta_target_per_day",
                "cognition.g_update.fbar_weight",
                "cognition.g_update.theta_min_mult",
                "cognition.g_update.theta_max_mult",
                "cognition.g_update.log_every_steps",
                "experiment.g_init.flat_value", "experiment.g_init.sigma0"):
        assert T.covers(key), f"Δt 分類テーブルに宣言が無い: {key}"
        assert T.classify(key)[0] == T.INVARIANT


def test_manifest_declares_the_update_rule(tmp_path):
    _sim, out = _run(tmp_path, "g_man", 12, 8, **GON)
    man = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    g = man["cognition"]["g_update"]
    assert g["rule"].startswith("g <- g + eta*(r - rho*ebar)")
    assert "day_boundary_once" in g["theta_update"]
    assert g["g_init"]["mode"] == "persona"
    assert g["g_init"]["stream"] == P.NOISE_STREAM
    # ★却下せず未実装であることを正直に宣言する(設計 §2.5 の将来拡張)
    assert g["attention_contagion"].startswith("not_implemented")
    assert man["cognition"]["fire"]["g_policy"].startswith("history_update")


def test_attention_contagion_is_not_implemented():
    """設計 §2.5 の将来拡張(注意の伝染)は**コードが存在しない**(既定 OFF ですらない)。"""
    src = (REPO_ROOT / "src" / "society" / "cognition").rglob("*.py")
    for path in src:
        text = path.read_text(encoding="utf-8")
        assert "contagion_gain" not in text and "g_contagion" not in text


# --------------------------------------------------------------------------- #
# (H) 解析スクリプト(検収基準 7)
# --------------------------------------------------------------------------- #
def test_analyze_g_produces_a_variance_decomposition(tmp_path):
    _sim_p, out_p = _run(tmp_path, "an_P", 150, 20, **GON,
                         **{"experiment.g_init.mode": "persona"})
    _sim_n, out_n = _run(tmp_path, "an_N", 150, 20, **GON,
                         **{"experiment.g_init.mode": "noise"})
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "analyze_g.py"),
         str(out_p), "--control", str(out_n)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(REPO_ROOT))
    assert proc.returncode == 0, proc.stderr[-2000:]
    doc = json.loads((out_p / "g_trajectory.json").read_text(encoding="utf-8"))
    assert len(doc["runs"]) == 2
    cm = doc["runs"][0]["decomposition"]["channel_mean"]
    # Var(g_end) = Var(g0) + Var(Δg) + 2·Cov(g0, Δg) の恒等式が数値で閉じている
    assert abs(cm["identity_residual"]) < 1e-9
    assert cm["share_born"] is not None and cm["share_emergent"] is not None
    assert cm["log_change_sd"] is not None      # §9: % ではなく対数変化
    assert doc["runs"][0]["mode"] == "persona"
    assert doc["runs"][1]["mode"] == "noise"
    assert doc["comparison"]["reading"], "P vs N の読み方が出力されていない"
    assert (out_p / "g_trajectory_report.md").exists()


def test_analyze_g_handles_the_flat_condition(tmp_path):
    """条件 F は Var(g0)=0 = 順位相関が定義できない(None で落ちずに出力できること)。"""
    _sim, out = _run(tmp_path, "an_F", 60, 16, **GON,
                     **{"experiment.g_init.mode": "flat"})
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "analyze_g.py"), str(out)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(REPO_ROOT))
    assert proc.returncode == 0, proc.stderr[-2000:]
    doc = json.loads((out / "g_trajectory.json").read_text(encoding="utf-8"))
    cm = doc["runs"][0]["decomposition"]["channel_mean"]
    assert cm["var_g0"] == 0.0
    assert cm["spearman_g0_gend"] is None
    assert cm["share_emergent"] == pytest.approx(1.0)
