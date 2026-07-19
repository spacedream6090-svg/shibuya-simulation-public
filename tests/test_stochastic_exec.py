"""確率的実行(行動のゆらぎ)= P2 スライス S4 のテスト。

設計: docs/plans/p2-interstitial-design.md §1 S4 / docs/research/interstitial-life.md §4.1
(実証3層 L1 骨格 motif / L2 時刻ジッター / L3 逸脱[寄り道・中断] + Gumbel ロジット化)。
方針(既存 R1 の鉄則を継承):
- OFF(既定): L1 が純粋既定とバイト一致・detour/interrupt 0 件・agent に _motif 等の実行時
  フィールドを一切生やさない(ゴールデン golden_baseline_l1.json は tests/test_scenario.py が別途担保)。
  新 stream(motif/jitter_time/detour/interrupt/gumbel)を一度も引かない。
- ON(mock ≤24step): (a)detour/interrupt イベントが出る。(b)同 seed 2 回で L1 完全一致(決定論)。
  (c)予算飽和下で ON と OFF の LLM 呼数が一致(=追加 LLM 呼ゼロ)+ 呼数が k 非依存。
  (d)fixed アンカー(勤務・通勤)はジッター/中断で動かない。(e)Gumbel ON で行き先分布が多様化
  (同一状況の反復で複数の行き先・温度で多様性が変わる)。(f)逸脱率が較正ターゲットの方向にある
  (日次逸脱 25〜35% / per-move 寄り道 10〜20% の"レンジ"。厳密較正は後日)。
検証は mock のみ(実LLM 禁止・≤24step。OFF バイト一致確認のみ 144step)。
"""
from __future__ import annotations

import json

from society.cognition import routine
from society.config import load_config
from society.engine.simulation import Simulation
from society.observer.schema import EVENT_KINDS

_ON = {"routine.stochastic.enabled": "true"}
_OFF = {"routine.stochastic.enabled": "false"}
_GUMBEL = {"routine.stochastic.enabled": "true",
           "routine.stochastic.gumbel.enabled": "true"}


def _sim(tmp_path, name, n=40, steps=24, seed=42, **ov):
    dot = [f"run.seed={seed}", f"run.n_agents={n}", f"run.n_steps={steps}",
           f"run.name={name}"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _kind(sim, kind):
    return [e for e in sim.logger.events if e.kind == kind]


# ------------------------------------------------------------ schema 登録
def test_detour_interrupt_kinds_registered():
    assert "detour" in EVENT_KINDS and "interrupt" in EVENT_KINDS


# ------------------------------------------------------------ OFF: 純粋既定と一致 + 無副作用
def test_off_matches_pure_default(tmp_path):
    """明示 OFF と純粋既定が L1 完全一致(24step)。OFF では実行時フィールドも作られない。"""
    pure = _sim(tmp_path, "pure")
    pure.run()
    off = _sim(tmp_path, "expl_off", **_OFF)
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(S4 seam が no-op でない)"
    assert not any(e.kind in ("detour", "interrupt") for e in off.logger.events)
    for a in pure.agents:
        assert not hasattr(a, "_motif"), "OFF なのに _motif が生えている"
        assert not hasattr(a, "_motif_day"), "OFF なのに _motif_day が生えている"


def test_off_matches_pure_default_long(tmp_path):
    """ゴールデンと同じ 144step 尺でも OFF は純粋既定とバイト一致(no-op の確度を上げる)。"""
    pure = _sim(tmp_path, "purel", n=12, steps=144)
    pure.run()
    off = _sim(tmp_path, "offl", n=12, steps=144, **_OFF)
    off.run()
    assert _l1(pure) == _l1(off)


# ------------------------------------------------------------ (a) detour/interrupt が出る
def test_on_emits_detour_and_interrupt(tmp_path):
    """ON の mock 24step で detour と interrupt が両方出る(payload の型も固定)。"""
    sim = _sim(tmp_path, "emit", n=150,
               **{**_ON, "routine.stochastic.interrupt_prob.discretionary": "0.3"})
    sim.run()
    det, itr = _kind(sim, "detour"), _kind(sim, "interrupt")
    assert det, "ON なのに detour が1件も出ていない"
    assert itr, "ON なのに interrupt が1件も出ていない"
    assert set(det[0].payload) == {"act", "to", "from_dest"}
    assert set(itr[0].payload) == {"act"}


# ------------------------------------------------------------ (b) 決定論(同 seed 2回で一致)
def test_on_deterministic(tmp_path):
    a = _sim(tmp_path, "det_a", n=60, **_ON)
    a.run()
    b = _sim(tmp_path, "det_b", n=60, **_ON)
    b.run()
    assert _l1(a) == _l1(b), "ON の決定論が崩れている(専用 stream 以外の乱数漏れ?)"


def test_on_differs_from_off(tmp_path):
    """ON は OFF と L1 が異なる(=ゆらぎが実際に注入され行動が非決定化している)。"""
    on = _sim(tmp_path, "diff_on", n=60, **_ON)
    on.run()
    off = _sim(tmp_path, "diff_off", n=60, **_OFF)
    off.run()
    assert _l1(on) != _l1(off)


# ------------------------------------------------------------ (c) 追加 LLM 呼ゼロ(予算飽和下)
def test_call_count_invariant_under_saturated_budget(tmp_path):
    """予算飽和下では ON と OFF の LLM 呼数が一致(S4 は generate() を1本も足さない=R1)。

    S4 は行き先/滞在を非決定化して co-presence を再配分しうるが、予算を飽和させれば
    総発火数 = Σ min(申請者数, 予算) となり「誰が発火するか」が変わっても「何本発火するか」は
    変わらない(tests/test_conversation_c2.py と同型の検査)。"""
    ov = {"lod.max_llm_per_step": 2}
    on = _sim(tmp_path, "cc_on", n=100, **{**_ON, **ov})
    on.run()
    off = _sim(tmp_path, "cc_off", n=100, **{**_OFF, **ov})
    off.run()
    assert len(on.logger.llm_calls) == len(off.logger.llm_calls) > 0, \
        f"予算飽和下で呼数が不一致: on={len(on.logger.llm_calls)} off={len(off.logger.llm_calls)}"


def test_call_count_k_invariant(tmp_path):
    """真の R1「呼数 k 非依存」: ON のまま k(writeback)を free/off に振っても呼数が不変
    (S4 は物理量・時刻・活動種別のみを入力にし k 由来量を一切見ない)。"""
    ov = {"lod.max_llm_per_step": 2}
    free = _sim(tmp_path, "k_free", n=100, **{**_ON, **ov, "k.writeback": "free"})
    free.run()
    off = _sim(tmp_path, "k_off", n=100, **{**_ON, **ov, "k.writeback": "off"})
    off.run()
    assert len(free.logger.llm_calls) == len(off.logger.llm_calls), \
        f"ON の呼数が k で乖離(R1 違反): free={len(free.logger.llm_calls)} off={len(off.logger.llm_calls)}"


# ------------------------------------------------------------ (d) fixed アンカーが動かない
def test_fixed_anchor_not_jittered(tmp_path):
    """L2/L3: 勤務・通勤(fixed アンカー=mandatory)は継続ジッターも中断も受けない。
    flexible 活動(eating 等)は継続がゆらぐ。"""
    sim = _sim(tmp_path, "anchor", n=6, steps=1,
               **{**_ON, "routine.stochastic.interrupt_prob.discretionary": "0.9"})
    scfg = routine._stochastic_cfg(sim)
    assert scfg is not None
    a = sim.agents[0]
    # 継続ジッター: fixed(working/commuting)は不変・flexible(eating)は揺らぐ
    for act in ("working", "commuting"):
        assert all(routine._jitter_stay(a, sim, 3, s, scfg, activity=act) == 3
                   for s in range(40)), f"fixed アンカー {act} の stay がジッターされた"
    flex = {routine._jitter_stay(a, sim, 3, s, scfg, activity="eating")
            for s in range(40)}
    assert len(flex) > 1, "flexible 活動(eating)の継続が全くゆらいでいない"
    # 中断: working(fixed)は常に中断しない・discretionary は中断しうる
    a.activity = "working"
    assert not any(routine._maybe_interrupt(a, sim, s, 600, scfg)
                   for s in range(40)), "fixed アンカー working が中断された"
    a.activity = ""                     # discretionary(自由行動)
    assert any(routine._maybe_interrupt(a, sim, s, 600, scfg)
               for s in range(40)), "discretionary が一度も中断しない(interrupt 無効?)"


# ------------------------------------------------------------ (e) Gumbel で行き先が多様化
def test_gumbel_diversifies_destinations(tmp_path):
    """Gumbel ON: 同一状況(=同じ訪問履歴)の反復で複数の行き先が出る。温度が高い活動
    (discretionary)ほど多様、低い活動(mandatory)は「いつもの行き先」に集中する。"""
    sim = _sim(tmp_path, "gumbel", n=6, steps=1, **_GUMBEL)
    scfg = routine._stochastic_cfg(sim)
    assert scfg is not None and scfg["gumbel_enabled"]
    a = sim.agents[0]
    dests = list(sim.dests)
    assert len(dests) >= 4, "テスト用の行き先候補が少なすぎる"
    a.node = dests[-1]                  # 現在地(候補から除かれる)
    a.visits[dests[0]] = 20             # 支配的な「いつもの行き先」(他は 0)
    hi = {routine._gumbel_destination(a, dests, sim, s, scfg, activity="")
          for s in range(80)}           # discretionary=高温
    lo = {routine._gumbel_destination(a, dests, sim, s, scfg, what="work")
          for s in range(80)}           # mandatory=低温
    assert len(hi) >= 3, f"高温でも行き先が多様化しない: {hi}"
    assert len(lo) < len(hi), f"低温が高温より多様(温度が効いていない): lo={lo} hi={hi}"
    assert dests[0] in lo, "低温で支配的な『いつもの行き先』が選ばれていない"


# ------------------------------------------------------------ (f) 逸脱率が較正ターゲットの方向
def test_deviation_rates_in_calibration_direction(tmp_path):
    """ON の mock 24step で「逸脱を含む人の割合」と「per-move 寄り道率」を数え、
    interstitial-life §4.1 の較正ターゲット(日次逸脱 25〜35% / per-move 10〜20%)の
    方向(=0 でも全員でもない現実的レンジ)にあることを確認する。厳密較正は後日・レンジで可。"""
    n = 150
    sim = _sim(tmp_path, "calib", n=n, steps=24, **_ON)
    sim.run()
    moves = len(_kind(sim, "route_start"))         # 移動の分母(全 move の起点)
    det, itr = _kind(sim, "detour"), _kind(sim, "interrupt")
    assert moves > 0 and det and itr
    deviators = {e.agent_id for e in det} | {e.agent_id for e in itr}
    dev_frac = len(deviators) / n
    per_move_detour = len(det) / moves
    # per-move 寄り道率: 全 route_start が母数(通勤等の非対象 move で希釈されるため 10〜20% より低め)。
    assert 0.01 <= per_move_detour <= 0.30, \
        f"per-move 寄り道率が方向外: {per_move_detour:.3f}(target 0.10〜0.20 方向)"
    # 逸脱を含む人の割合(24step=4時間の部分日。0 でも全員でもない現実的レンジ)。
    assert 0.10 <= dev_frac <= 0.98, \
        f"逸脱を含む人の割合が方向外: {dev_frac:.3f}(events det={len(det)} itr={len(itr)})"
