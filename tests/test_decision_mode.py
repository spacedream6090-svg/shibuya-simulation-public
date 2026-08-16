"""V3 決定モード印字(observer.decision_mode)の検収。

正典: docs/plans/external-audit-triage.md §3.2 V3 /
      実装 src/society/observer/decision_mode.py / 解析 scripts/decision_modes.py

守るもの(検収基準の順)
  (1) **既定 OFF = 1 バイトも動かない**: 純粋既定と L1 完全一致・summary にキーなし・
      state 不在・新 L1 kind 0 件。
  (2) **ON でも世界不変**: 観測 ON / OFF で L1 が**完全一致**(この機能は L1 に 1 行も
      足さないので、starvation のように「観測イベントを除いた残り」を取る必要すら無い)・
      LLM 呼数一致・正準状態(乱数消費と世界状態)一致 = 観測がシムを変えないことの機械証明。
  (3) **不変式 points == llm + reuse + rule**(日ごと・全期間)。分母 points が
      「在場覚醒の個体数 × step」と一致すること。llm_calls が l1b_llm の行と一致すること。
  (4) 各モードが実際に立つこと: 再利用(policy_cache)・計画駆動(day_plan)・
      パース不成立(用途つき)・予算枯渇(DPH-B tiers)・定型返答(engaged)・勾留。
  (5) **batch_llm ON/OFF の両経路で同じ記録**が出る(第129 の deliberate 一括化)。
  (6) resume == straight(累積タリーが checkpoint を跨いで一致)。
  (7) レジストリ宣言・conf 既定値・解析スクリプトの疎通。
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from society import registry as R
from society.cognition import deliberate as DEL
from society.config import load_config
from society.engine import checkpoint, scheduler
from society.engine.simulation import Simulation
from society.observer import decision_mode as DM
from society.observer import state_hash
from society.observer.schema import EVENT_KINDS

_REPO = Path(__file__).resolve().parents[1]

ON = {"observer.decision_mode.enabled": "true"}
DAYPLAN = {"planning.day_plan.enabled": "true"}
CACHE = {"cognition.policy_cache.enabled": "true"}


# --------------------------------------------------------------------------- #
# 共通ヘルパ
# --------------------------------------------------------------------------- #
def _cfg(name, n_steps=144, n_agents=15, **ov):
    dot = ["run.seed=42", f"run.n_agents={n_agents}", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock", "observer.snapshot_every=144"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def _sim(tmp_path, name, n_steps=144, n_agents=15, **ov):
    return Simulation(_cfg(name, n_steps, n_agents, **ov), out_dir=tmp_path / name)


def _run(tmp_path, name, n_steps=144, n_agents=15, **ov):
    sim = _sim(tmp_path, name, n_steps, n_agents, **ov)
    sim.run()
    return sim


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _prov(sim) -> dict:
    p = DM.provenance(sim)
    assert p is not None, "ON なのに provenance が None"
    return p


# =========================================================================== #
# (1) 既定 OFF = 1 バイトも動かない
# =========================================================================== #
def test_off_is_a_complete_noop(tmp_path):
    """既定では state も summary キーも生えず、L1 は純粋既定と完全一致。"""
    base = _run(tmp_path, "dm_base")
    assert DM.enabled(base) is False
    assert getattr(base, "_decision_mode_state", None) is None
    assert DM.provenance(base) is None
    summary = json.loads((tmp_path / "dm_base" / "summary.json")
                         .read_text(encoding="utf-8"))
    assert "decision_mode" not in summary, "OFF なのに summary にキーが出た"


def test_module_registers_no_new_event_kind():
    """本 module は L1 の kind を 1 つも足さない(出口は summary のキー 1 つだけ)。

    ★これが starvation(新 kind 3 種)との決定的な違いで、schema/causality への
      登録が要らない根拠でもある(第115 教訓の「新 kind は両方へ登録」の対象外)。
    """
    src = (_REPO / "src" / "society" / "observer" / "decision_mode.py") \
        .read_text(encoding="utf-8")
    assert "register_event_kind" not in src, \
        "decision_mode が L1 kind を登録している(設計に反する)"
    assert "logger.log(" not in src, "decision_mode が L1 へ書いている(設計に反する)"


# =========================================================================== #
# (2) ON でも世界は 1 バイトも動かない
# =========================================================================== #
@pytest.mark.parametrize("extra", [{}, DAYPLAN, {**DAYPLAN, **CACHE}])
def test_on_does_not_change_the_world(tmp_path, extra):
    """観測 ON / OFF で L1・LLM 呼数・正準状態(乱数消費込み)が完全一致。"""
    name = "dm_inv" + str(abs(hash(tuple(sorted(extra.items())))) % 9973)
    off = _run(tmp_path, name + "_off", 144, 20, **extra)
    on = _run(tmp_path, name + "_on", 144, 20, **extra, **ON)

    assert _l1(off) == _l1(on), "ON で L1 が変わった(観測がシムを変えている)"
    assert len(off.logger.llm_calls) == len(on.logger.llm_calls), "LLM 呼数が変わった"
    assert (state_hash.canonical_state(off, 144)
            == state_hash.canonical_state(on, 144)), "正準状態が変わった"


def test_on_adds_no_new_l1_kind(tmp_path):
    off = _run(tmp_path, "dm_kind_off", 72, 12)
    on = _run(tmp_path, "dm_kind_on", 72, 12, **ON)
    assert (Counter(e.kind for e in off.logger.events)
            == Counter(e.kind for e in on.logger.events))
    # 念のため: 語彙表に決定モード由来の語が紛れ込んでいない
    assert "decision_mode" not in EVENT_KINDS


# =========================================================================== #
# (3) 不変式と分母の正しさ
# =========================================================================== #
def test_invariant_points_equals_llm_plus_reuse_plus_rule(tmp_path):
    sim = _run(tmp_path, "dm_inv2", 288, 30, **DAYPLAN, **CACHE, **ON)
    prov = _prov(sim)
    for label, cell in [("total", prov["total"])] + list(prov["by_day"].items()):
        assert cell["residual"] == 0, f"day={label} で不変式が破れた: {cell}"
        assert cell["points"] == cell["llm"] + cell["reuse"] + cell["rule"]
        assert cell["llm"] >= 0 and cell["rule"] >= 0
    # 日別の合計が全期間と一致する(ロールアップの正しさ)
    assert sum(c["points"] for c in prov["by_day"].values()) == prov["total"]["points"]
    assert sum(c["rule"] for c in prov["by_day"].values()) == prov["total"]["rule"]


@pytest.mark.parametrize("batch", [False, True])
def test_points_equals_the_number_of_decide_calls(tmp_path, monkeypatch, batch):
    """分母 = `_decide_g` が回された回数と**厳密に一致**する(逐次・一括の両経路)。

    ★これが V3 の核心。この数はこれまでどこにも記録されておらず、L1 からも
      l1b からも復元できなかった(= LLM 被覆率の分母が原理的に無かった)。
      `_decide_g` は逐次経路(`_decide`)と一括発行経路(`_decide_rounds`)の
      **両方が通る唯一の入口**なので、ここを数えれば決定の総数が取れる。
    """
    n_calls = [0]
    real = scheduler._decide_g

    def counting(sim, agent, step, sim_min):
        n_calls[0] += 1
        return real(sim, agent, step, sim_min)

    monkeypatch.setattr(scheduler, "_decide_g", counting)
    ov = {"engine.batch_llm.enabled": "true"} if batch else {}
    sim = _run(tmp_path, f"dm_points_{int(batch)}", 72, 20, **ov, **ON)
    prov = _prov(sim)
    assert n_calls[0] > 0, "決定が 1 度も走っていない(前提が崩れた)"
    assert prov["total"]["points"] == n_calls[0], \
        f"points {prov['total']['points']} != _decide 呼び出し {n_calls[0]}"


def test_points_matches_active_predicate_step_by_step(tmp_path, monkeypatch):
    """`active` の要素数(= scheduler が数えた値)と points の増分が step ごとに一致。

    `active` は run_step の**途中**(起床・帰還・在場ローテーションの後)で組まれるので、
    step の頭で同じ述語を当てても一致しない。記録の入口を覗いて実際の n を捕まえる。
    """
    seen: list[int] = []
    real = DM.note_points

    def spy(sim, sim_min, n):
        seen.append(int(n))
        return real(sim, sim_min, n)

    monkeypatch.setattr(scheduler.decmode_mod, "note_points", spy)
    sim = _sim(tmp_path, "dm_points2", 12, 20, **ON)
    for step in range(6):
        before = _prov(sim)["total"]["points"]
        scheduler.run_step(sim, step)
        after = _prov(sim)["total"]["points"]
        assert after - before == seen[-1], \
            f"step={step}: points の増分 {after - before} != active {seen[-1]}"
        assert 0 <= seen[-1] <= len(sim.agents)


def test_llm_calls_match_l1b_rows(tmp_path):
    """記録した熟慮呼が l1b_llm の行と 1:1(log_llm_call と同じ位置で数えている)。"""
    sim = _run(tmp_path, "dm_l1b", 288, 25, **DAYPLAN, **ON)
    prov = _prov(sim)
    recorded = Counter()
    for k, v in prov["total"]["llm_calls"].items():
        recorded[k] = v
    actual = Counter(c["purpose"] for c in sim.logger.llm_calls
                     if c["purpose"] in recorded)
    assert recorded == actual, f"熟慮呼の用途別内訳が l1b と食い違う: {recorded} vs {actual}"
    # 計画・内省は熟慮レーンではないので **数えていない**(混ぜていないことの確認)
    assert "plan" not in prov["total"]["llm_calls"]
    assert "reflect" not in prov["total"]["llm_calls"]


# =========================================================================== #
# (4) 各モードが実際に立つ
# =========================================================================== #
def test_reuse_is_counted_and_matches_policy_reuse_events(tmp_path):
    sim = _run(tmp_path, "dm_reuse", 288, 30, **CACHE, **ON)
    prov = _prov(sim)
    l1 = Counter(e.payload.get("kind") for e in sim.logger.events
                 if e.kind == "policy_reuse")
    assert l1, "方針キャッシュ ON なのに再利用が 1 件も起きていない(前提が崩れた)"
    assert prov["total"]["reuse_by_kind"] == {k: v for k, v in sorted(l1.items())}
    assert prov["total"]["reuse"] == l1.get("deliberate", 0), \
        "熟慮レーンの分子は kind=deliberate だけ(plan の再利用を混ぜない)"


def test_plan_driven_actions_carry_the_plan_provenance(tmp_path):
    """朝の計画のブロックが決めた step は `plan:<計画の来歴>` として rule の中に立つ。"""
    sim = _run(tmp_path, "dm_plan", 288, 30, **DAYPLAN, **ON)
    srcs = _prov(sim)["total"]["rule_by_src"]
    assert "habit" in srcs
    plan_keys = [k for k in srcs if k.startswith("plan:")]
    assert plan_keys, f"計画駆動の行動が 1 件も立っていない: {srcs}"
    assert all(k.split(":", 1)[1] in ("llm", "skeleton", "prev_day", "")
               for k in plan_keys), f"未知の計画来歴: {plan_keys}"
    # 計画駆動は必ず rule の部分集合(合計を超えない)
    assert sum(srcs.values()) == _prov(sim)["total"]["rule"]


def test_plan_provenance_slot_never_leaks_across_steps(tmp_path):
    """一時スロットは step 境界で必ず空になる(別の決定へ出所が漏れない)。"""
    sim = _sim(tmp_path, "dm_leak", 12, 15, **DAYPLAN, **ON)
    for step in range(4):
        sim._decision_mode_pending = "plan:bogus"          # 対が崩れた状態を作る
        scheduler.run_step(sim, step)
        assert getattr(sim, "_decision_mode_pending", None) is None
    assert "plan:bogus" not in _prov(sim)["total"]["rule_by_src"]


def test_day_plan_off_never_produces_plan_src(tmp_path):
    sim = _run(tmp_path, "dm_noplan", 144, 20, **ON)
    srcs = _prov(sim)["total"]["rule_by_src"]
    assert set(srcs) == {"habit"}, f"day_plan OFF なのに計画駆動が立った: {srcs}"


def test_unparsed_is_counted_with_its_trigger(tmp_path, monkeypatch):
    """壊れた応答は用途つきで数え、その決定は rule(*_unparsed)へ落ちる。

    `fallback{reason:"parse_error"}` には trigger が載らないので、**この内訳は
    既存 L1 からは復元できない**(V3 が埋めた 4 点のうちの 1 つ)。
    """
    monkeypatch.setattr(DEL, "parse_action", lambda _resp: None)
    sim = _run(tmp_path, "dm_broken", 144, 20, **ON)
    prov = _prov(sim)
    assert prov["total"]["llm_unparsed"], "壊した応答が 1 件も数えられていない"
    assert prov["total"]["llm_unparsed"] == prov["total"]["llm_calls"], \
        "全応答を壊したのに一部しか不成立になっていない"
    assert prov["total"]["llm"] == 0, "全滅なのに LLM が決めた決定が残っている"
    reasons = prov["total"]["rule_by_reason"]
    assert {"reply_unparsed", "fire_unparsed"} & set(reasons), \
        f"後退の理由が記録されていない: {reasons}"
    assert prov["total"]["residual"] == 0


def test_reply_starvation_is_counted(tmp_path):
    """予算切れで返事が落ちた決定は reply_starved として立つ。

    ★DPH-B(lod.budget.tiers)ON では返事に予約枠があり 100% 通るので、飢餓を
      起こすのは **tiers OFF + 小さい cap**(= DPH-O が実測した現行構成)である。
    """
    ov = {"lod.max_llm_per_step": "4", "observer.starvation.enabled": "true"}
    sim = _run(tmp_path, "dm_starve", 120, 30, **ov, **ON)
    reasons = _prov(sim)["total"]["rule_by_reason"]
    dropped = sum(1 for e in sim.logger.events if e.kind == "reply_dropped")
    assert dropped > 0, "予算 1 でも返事が 1 件も落ちていない(前提が崩れた)"
    assert reasons.get("reply_starved", 0) > 0, f"reply_starved が立たない: {reasons}"
    # 落ちた返事のうち、そのまま rule で終わった決定だけが reply_starved になる
    # (その後に発火権があれば LLM が決めうるので <= が正しい関係)。
    assert reasons["reply_starved"] <= dropped


def test_template_reply_is_counted_as_rule(tmp_path):
    """第87 engaged の定型返答は LLM を 1 本も呼ばない = rule(template)。

    自然発生を待つと構成依存になるので、test_engaged と同じく `_decide` を直接 1 回
    回して「初対面からの接触 = 定型で流す」枝だけを踏む。
    """
    ov = {"cognition.fire.enabled": "true", "cognition.engaged.enabled": "true"}
    sim = _sim(tmp_path, "dm_tmpl", 1, 15, **ov, **ON)
    a, b = sim.agents[0], sim.agents[1]
    a._reply_to = (b.id, "こんにちは")
    before = sim.llm.calls
    action = scheduler._decide(sim, a, 0, 0)
    assert action["type"] == "speak", "定型返答の枝を踏んでいない(前提が崩れた)"
    assert sim.llm.calls == before, "テンプレ応答なのに LLM を呼んだ"
    reasons = _prov(sim)["total"]["rule_by_reason"]
    assert reasons == {"template": 1}, f"template として数えていない: {reasons}"


def test_detained_agents_are_counted_as_rule(tmp_path):
    """勾留中(既定 0=立たない)も決定点であり、rule(detained)として数える。"""
    sim = _sim(tmp_path, "dm_detain", 12, 20, **ON)
    for a in sim.agents[:5]:
        a.detained_until = 999
    for step in range(6):
        scheduler.run_step(sim, step)
    reasons = _prov(sim)["total"]["rule_by_reason"]
    assert reasons.get("detained", 0) > 0, f"detained が立たない: {reasons}"
    assert _prov(sim)["total"]["residual"] == 0


# =========================================================================== #
# (5) batch_llm ON/OFF の両経路で同じ記録
# =========================================================================== #
def test_batch_and_serial_paths_record_the_same(tmp_path):
    """第129 の熟慮一括化(engine.batch_llm)は決定モードの記録を変えない。

    一括発行は `_decide_g` を**そのまま**回して応答適用だけを遅らせる設計なので、
    記録も逐列と同じ位置で 1 度ずつ起きるはずである — それを機械で固定する。
    """
    serial = _run(tmp_path, "dm_serial", 144, 25, **DAYPLAN, **ON)
    batch = _run(tmp_path, "dm_batch", 144, 25, **DAYPLAN, **ON,
                 **{"engine.batch_llm.enabled": "true",
                    "engine.batch_llm.workers": "4"})
    assert _l1(serial) == _l1(batch), "batch で L1 が変わった(前提が崩れた)"
    assert _prov(serial) == _prov(batch), "batch で決定モードの記録が変わった"


# =========================================================================== #
# (6) resume == straight
# =========================================================================== #
def test_resume_matches_straight(tmp_path):
    """★方針キャッシュは**入れない**: `cognition.policy_cache` は設計上 checkpoint 対象外
    (揮発でよい)なので resume 後にキャッシュが空になり、再利用が LLM へ振り替わる。
    これは PENDING §3 の既知の未決事項であって決定モードの記録側の問題ではない
    (実際 `rule` は resume でも完全一致し、`llm` と `reuse` の間だけが動く)。
    """
    ov = {**DAYPLAN, **ON, "observer.checkpoint_every": "20"}
    straight = tmp_path / "dm_straight"
    s0 = Simulation(_cfg("dm_straight", 40, 20, **ov), out_dir=straight)
    s0.run()

    resumed = tmp_path / "dm_resumed"
    s1 = Simulation(_cfg("dm_resumed", 20, 20, **ov), out_dir=resumed)
    for step in range(20):
        scheduler.run_step(s1, step)
    checkpoint.save(s1, 20, resumed / "checkpoint" / "ckpt-000020.pkl.gz")
    s1.logger.flush_segment()
    s2 = Simulation(_cfg("dm_resumed", 40, 20, **ov), out_dir=resumed)
    s2.run(resume_from=resumed)

    a = json.loads((straight / "summary.json").read_text(encoding="utf-8"))
    b = json.loads((resumed / "summary.json").read_text(encoding="utf-8"))
    assert a["decision_mode"] == b["decision_mode"], \
        "決定モードの累積タリーが resume で straight と食い違う"


# =========================================================================== #
# (7) 宣言・conf・解析スクリプト
# =========================================================================== #
def test_toggle_is_declared_in_the_registry():
    ids = {f.id for f in R.FEATURES}
    assert "observer.decision_mode.enabled" in ids, "レジストリ未宣言"
    f = next(x for x in R.FEATURES if x.id == "observer.decision_mode.enabled")
    assert f.repro_tier == "strict"
    assert f.affects_k is False, "記録専用なのに affects_k=True になっている"
    assert f.fingerprint_risk == "none"


def test_shipped_conf_default_is_off():
    import yaml                                                # noqa: PLC0415
    doc = yaml.safe_load((_REPO / "conf" / "config.yaml").read_text(encoding="utf-8"))
    assert doc["observer"]["decision_mode"]["enabled"] is False


def test_analysis_script_runs_and_reports_three_lanes(tmp_path):
    """scripts/decision_modes.py が run-dir を読み、3 レーンの表を出す。"""
    import sys                                                 # noqa: PLC0415
    sys.path.insert(0, str(_REPO / "scripts"))
    import decision_modes as DMS                               # noqa: PLC0415

    out = tmp_path / "dm_script"
    Simulation(_cfg("dm_script", 288, 30, **DAYPLAN, **CACHE, **ON),
               out_dir=out).run()
    rep = DMS.analyze(out)
    assert rep["have_decision_mode"] is True
    assert rep["residual_total"] == 0
    lanes = {ln["lane"] for row in rep["days"] for ln in row["lanes"]}
    assert "deliberate" in lanes, "熟慮レーンが出ていない"
    assert {"plan", "reflect"} & lanes, f"既存 L1 由来のレーンが出ていない: {lanes}"
    md = DMS.render_markdown(rep)
    assert "決定モードの内訳" in md and "deliberate" in md
    assert DMS.main([str(out)]) == 0


def test_analysis_script_degrades_without_the_summary_block(tmp_path):
    """OFF で回したランでも落ちない(計画・内省の 2 レーンだけを出す)。"""
    import sys                                                 # noqa: PLC0415
    sys.path.insert(0, str(_REPO / "scripts"))
    import decision_modes as DMS                               # noqa: PLC0415

    out = tmp_path / "dm_script_off"
    Simulation(_cfg("dm_script_off", 288, 20, **DAYPLAN), out_dir=out).run()
    rep = DMS.analyze(out)
    assert rep["have_decision_mode"] is False
    assert rep["residual_total"] == 0
    md = DMS.render_markdown(rep)
    assert "observer.decision_mode OFF" in md
