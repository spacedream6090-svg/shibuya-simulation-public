"""IF-B(拒否通知の段階 conf 化 `cognition.rejection_notify`)のテスト。

正典
  - docs/research/llm-world-interface-audit.md **§2-C**(拒否3層と通知の現状=無音拒否の全リスト)
  - docs/research/if-lane-research.md **§3**(Inner Monologue の feedback 3 種 /
    **Ovsiankina 効果**+ Masicampo & Baumeister 2011。**Zeigarnik は 2025 メタ分析で否定**)
  - docs/plans/if-sv-p4-plan.md 「IF-B」行

守るもの(検収基準の順)
  ① silent(既定)= ゴールデン L1 バイト一致・記憶も L1 も**増分ゼロ**・属性も state も
     生えない・summary にキーなし・プロンプト 1 バイト不変
  ② memory = 対象 8 種すべてで定型文が記憶へ入り L1 `action_reject` が 1 件出る。
     **文面に数字も実験条件語も含まれない**機械検査 + 「同じ拒否は全条件で同一バイト列」
  ③ engaged = 認知イベントの前倒しが起きる / `cognition.engaged` OFF では memory へ縮退
  ④ plan_exception の失敗理由: 注入は**状況行 1 行だけ** + **降格**
     (再計画が実ったら消える / 日を跨いだら消える)
  ⑤ ON ランの同 seed 2 ラン一致
  ⑥ resume == straight
  ⑦ k=free/off で LLM 呼数一致(通知は generate() の呼び出しサイトを 1 つも作らない)
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pyarrow.parquet as pq

from society import registry as R
from society import reject as RJ
from society import truth_ledger as TL
from society.cognition import day_plan as DP
from society.cognition import deliberate
from society.cognition import engaged as EG
from society.config import load_config
from society.engine import checkpoint, scheduler
from society.engine.simulation import Simulation
from society.observer.schema import EVENT_KINDS

GOLDEN = Path(__file__).resolve().parent / "data" / "golden_baseline_l1.json"

# test_scenario.py:45 と同じ「意図的な既定挙動追加」の中立化(ゴールデン比較用)
_GOLDEN_NEUTRAL = {"economy.wages.自営": 0, "planning.enabled": "false",
                   "transit_ride.taxi.enabled": "false",
                   "transit_ride.bus.enabled": "false", "rules.enabled": "false"}

SILENT = {"cognition.rejection_notify": "silent"}
MEMORY = {"cognition.rejection_notify": "memory"}
ENGAGED = {"cognition.rejection_notify": "engaged"}

FIRE_ON = {"cognition.fire.enabled": "true"}
ENGAGED_ON = {**FIRE_ON, "cognition.engaged.enabled": "true"}
DAYPLAN_ON = {"planning.day_plan.enabled": "true"}
BELIEFS_ON = {"beliefs.enabled": "true", "beliefs.verify_actions": "true"}

# 監査 §2-C が「記録なし・記憶なし」とした無音拒否の**全リスト**(本バッチの対象)。
TARGETS: tuple[tuple[str, str], ...] = (
    ("open_venture", "no_money"),          # 所持金不足で出店(tools._open_venture)
    ("move_home", "no_money"),             # 敷金不足で転居
    ("move_home", "no_room"),              # 空き住戸なし
    ("propose_partnership", "absent"),     # 交際申込の相手が近傍にいない
    ("move_to", "unreachable"),            # 経路が張れない(len(path)<2)
    ("verify", "no_target"),               # 確かめる対象が無い(従来 L1 のみ)
    ("verify", "no_witness"),              # 目撃者が近くにいない(同上)
    ("verify", "no_channel"),              # 調べる手立てが無い(同上)
)


# --------------------------------------------------------------------------- #
# 共通ヘルパ(test_llm_link.py / test_engaged.py と同型)
# --------------------------------------------------------------------------- #
def _cfg(name, n_steps=144, n_agents=15, **ov):
    dot = ["run.seed=42", f"run.n_agents={n_agents}", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock", "observer.snapshot_every=144"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def _sim(tmp_path, name, n_steps=144, n_agents=15, **ov):
    return Simulation(_cfg(name, n_steps, n_agents, **ov), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _l1x(sim):
    return [[e.step, e.agent_id, e.kind, e.llm_call_id,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _kind(sim, kind):
    return [e for e in sim.logger.events if e.kind == kind]


def _memories(agent) -> list[str]:
    return [ep.text for ep in agent.mem.buffer]


def _reject_memories(agent) -> list[str]:
    return [ep.text for ep in agent.mem.buffer if ep.kind == "reject"]


class _FixedLLM:
    """固定応答スタブ(test_day_plan / test_llm_link と同型)。"""

    name = "fixed"

    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.hits = 0

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        self.calls += 1
        return self.response, str(self.calls), False


# ---- 対象 8 種を**実際の裁定サイト**で 1 件ずつ起こすドライバ ------------------- #
def _seed_belief(sim, agent) -> None:
    """検証対象の fact を 1 件だけ台帳へ置き、当人に未検証の信念を持たせる。"""
    facts = TL.facts_of(sim)
    fid = "f00000"
    facts[fid] = {"id": fid, "kind": "venture_open", "step": 0, "sim_min": 0,
                  "node": agent.node, "place": "どこか",
                  "x": float(agent.x), "y": float(agent.y), "value": 1.0,
                  "actor": -1, "topics": ["とある店"], "radius_m": 60.0,
                  "window": 6, "canary": TL.register_canary(fid)}
    TL.beliefs_of(agent)[fid] = {"value": 1.0, "conf": 0.5, "src": "agent",
                                 "from": -1, "step": 0, "hop": 1,
                                 "verified": None}


def _trigger(sim, agent, kind: str, reason: str, monkeypatch=None) -> None:
    """(kind, reason) の拒否を**裁定側の実コード**で 1 回起こす。"""
    if kind == "open_venture":
        agent.money = 0.0
        sim.tools._open_venture(sim, agent, {"name": "店", "offer": ""}, 0, 0)
    elif kind == "move_home" and reason == "no_money":
        agent.money = 0.0
        scheduler._apply_p2(sim, agent, {"type": "move_home", "area": ""}, 0, 0)
    elif kind == "move_home" and reason == "no_room":
        agent.money = 10 ** 9
        monkeypatch.setattr(scheduler.freedom_p2_mod, "pick_home",
                            lambda *a, **k: None)
        scheduler._apply_p2(sim, agent, {"type": "move_home", "area": ""}, 0, 0)
    elif kind == "propose_partnership":
        agent.partner_id = None
        scheduler._apply_p2(sim, agent,
                            {"type": "propose_partnership",
                             "to": "この街にいない誰か"}, 0, 0)
    elif kind == "move_to":
        # 同一ノードへの move_to = router が 1 ノードの経路しか返せない(len(path)<2)
        scheduler._apply(sim, agent, {"type": "move_to", "dest": agent.node}, 0, 0)
    elif kind == "verify":
        # ★裁定 module(truth_ledger.py)は凍結 14 ファイルなので触っていない。
        #   通知は engine の唯一のディスパッチャ経由で入るので、テストも _apply を通す。
        how = {"no_target": "go", "no_witness": "ask", "no_channel": "net"}[reason]
        if reason != "no_target":
            _seed_belief(sim, agent)
        scheduler._apply(sim, agent,
                         {"type": "verify", "how": how, "about": ""}, 0, 0)
    else:                                          # pragma: no cover(取りこぼし検出)
        raise AssertionError(f"未対応の拒否種: {kind}/{reason}")


def _trigger_cfg(kind: str, reason: str) -> dict:
    """その拒否を起こすのに要る conf(前提機構だけ・通知水準は呼び手が足す)。"""
    if kind == "move_home":
        return {"freedom.p2.move_home": "true",
                "freedom.p2.deposit": (0 if reason == "no_room" else 10 ** 9)}
    if kind == "propose_partnership":
        return {"freedom.p2.partnership": "true"}
    if kind == "verify":
        ov = dict(BELIEFS_ON)
        if reason == "no_channel":
            ov["net.enabled"] = "false"
        return ov
    return {}


# --------------------------------------------------------------------------- #
# (A) 出荷既定・宣言(検収基準 ①の前段)
# --------------------------------------------------------------------------- #
def test_shipped_default_is_silent():
    cfg = load_config()
    assert str(cfg.cognition.rejection_notify) == RJ.SILENT
    assert RJ.LEVELS == ("silent", "memory", "engaged")


def test_registry_and_schema_declared():
    feat = {f.id: f for f in R.FEATURES}["cognition.rejection_notify"]
    assert feat.repro_tier == "strict"
    assert feat.affects_k is True          # engaged 水準が発火点を動かす(正直な宣言)
    assert feat.fingerprint_risk == "possible"
    assert feat.off_value == RJ.SILENT
    assert "action_reject" in EVENT_KINDS


def test_vocabulary_tables_are_complete_and_finite():
    """語彙表と列挙が 1 対 1(片方だけ増えたら落ちる)= 自由文が通知経路へ入らない構造。"""
    assert set(RJ.ACT_JP) == set(RJ.KINDS)
    assert set(RJ.REASON_JP) == set(RJ.REASONS)
    assert set(RJ.PLAN_REASON_JP) == set(RJ.PLAN_REASONS)
    assert set(RJ.PLAN_ACT_JP) == set(DP.ACTS), \
        "計画の活動語が day_plan.ACTS と食い違う(語彙の単一源が崩れた)"
    # 監査 §2-C の対象一覧が語彙に収まっている(取りこぼし検出)
    for kind, reason in TARGETS:
        assert kind in RJ.KINDS and reason in RJ.REASONS
    assert {k for k, _r in TARGETS} == set(RJ.KINDS)
    assert {r for _k, r in TARGETS} == set(RJ.REASONS)


def test_scope_exclusions_are_documented():
    """対象外 3 件の理由が docstring に明記されている(仕様の所在を固定する)。"""
    text = (Path(RJ.__file__)).read_text(encoding="utf-8")
    for token in ("envfeedback", "選択**前**のフィルタ", "config ゲート層"):
        assert token in text, f"対象外の理由 {token} が docstring に無い"


def test_frozen_metric_spec_files_are_untouched():
    """凍結 14 ファイル(metrics_spec.SPEC_FILES)に本バッチの痕跡が 1 つも無い。

    verify の通知は engine 側で outcome を読む方式にしてあるので、裁定 module
    ``truth_ledger.py`` は 1 バイトも変わらない = ``metrics_spec_hash`` は無風。
    """
    from society.observer import metrics_spec as MS
    root = Path(__file__).resolve().parents[1]
    assert len(MS.SPEC_FILES) == 14
    for rel in MS.SPEC_FILES:
        text = (root / rel).read_text(encoding="utf-8")
        assert "reject" not in text and "rejection_notify" not in text, \
            f"凍結ファイル {rel} に IF-B の痕跡がある(metrics_spec_hash が動く)"


def test_module_creates_no_llm_call_site():
    """通知は generate() の呼び出しサイトも乱数 stream も 1 つも作らない(静的検査)。

    散文(docstring / コメント)は対象外 — **実際に評価される識別子**だけを見る。
    """
    tree = ast.parse((Path(RJ.__file__)).read_text(encoding="utf-8"))
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert "generate" not in attrs and "generate" not in names
    assert "llm" not in attrs, "通知が LLM を触っている"
    assert "stream" not in attrs, "通知が乱数 stream を引いている(決定論違反)"


# --------------------------------------------------------------------------- #
# (B) silent = 現行と 1 バイトも変わらない(検収基準 ①)
# --------------------------------------------------------------------------- #
def test_silent_matches_golden(tmp_path):
    golden = json.load(open(GOLDEN, encoding="utf-8"))
    sim = _sim(tmp_path, "rj_golden", **_GOLDEN_NEUTRAL)
    sim.run()
    assert _l1(sim) == golden, "IF-B の seam がゴールデンを動かしている"


def test_silent_matches_pure_default(tmp_path):
    pure = _sim(tmp_path, "rj_pure")
    pure.run()
    off = _sim(tmp_path, "rj_off", **SILENT)
    off.run()
    assert _l1x(pure) == _l1x(off)


def test_silent_emits_no_event_no_memory_no_state(tmp_path, monkeypatch):
    """silent では対象 8 種を全部起こしても L1 も記憶も state も 1 も動かない。"""
    for kind, reason in TARGETS:
        sim = _sim(tmp_path, f"rj_s_{kind}_{reason}", n_steps=1,
                   **SILENT, **_trigger_cfg(kind, reason))
        a = sim.agents[0]
        n_events, n_mem = len(sim.logger.events), len(a.mem.buffer)
        _trigger(sim, a, kind, reason, monkeypatch)
        new = sim.logger.events[n_events:]
        assert not [e for e in new if e.kind == "action_reject"], \
            f"silent なのに action_reject が出た({kind}/{reason})"
        assert len(a.mem.buffer) == n_mem, \
            f"silent なのに記憶が増えた({kind}/{reason})"
        assert getattr(sim, "_reject_state", None) is None
        assert getattr(a, RJ.WHY_KEY, None) is None


def test_silent_notify_is_a_pure_noop(tmp_path):
    sim = _sim(tmp_path, "rj_noop", n_steps=1)
    a = sim.agents[0]
    before = (len(sim.logger.events), len(a.mem.buffer))
    assert RJ.notify(sim, a, "open_venture", "no_money", 0, 0) is False
    assert (len(sim.logger.events), len(a.mem.buffer)) == before
    assert RJ.enabled(sim) is False
    RJ.note_why(sim, a, "meal", "late")
    assert getattr(a, RJ.WHY_KEY, None) is None, "silent なのに属性が生えた"
    assert RJ.why_line(sim, a) is None


def test_silent_summary_has_no_key(tmp_path):
    sim = _sim(tmp_path, "rj_sum_off", n_steps=24, n_agents=10)
    sim.run()
    summary = json.loads((tmp_path / "rj_sum_off" / "summary.json")
                         .read_text(encoding="utf-8"))
    assert "rejection_notify" not in summary
    assert RJ.provenance(sim) is None


def test_silent_prompts_are_byte_identical(tmp_path):
    """通知が無い世界ではプロンプトが純粋既定と 1 バイトも変わらない。"""
    def _prompts(name, **ov):
        sim = _sim(tmp_path, name, n_steps=72, n_agents=12, **ov)
        seen: list = []
        inner = sim.llm.generate

        def _gen(prompt, **kw):
            seen.append(prompt)
            return inner(prompt, **kw)
        sim.llm.generate = _gen
        sim.run()
        return seen

    base = _prompts("rj_p_base")
    off = _prompts("rj_p_off", **SILENT)
    assert base and base == off


# --------------------------------------------------------------------------- #
# (C) memory 水準(検収基準 ②)
# --------------------------------------------------------------------------- #
def test_every_target_notifies_at_memory_level(tmp_path, monkeypatch):
    """対象 8 種すべてで定型文が記憶に入り、L1 action_reject が 1 件出る。"""
    for kind, reason in TARGETS:
        sim = _sim(tmp_path, f"rj_m_{kind}_{reason}", n_steps=1,
                   **MEMORY, **_trigger_cfg(kind, reason))
        a = sim.agents[0]
        n_events = len(sim.logger.events)
        _trigger(sim, a, kind, reason, monkeypatch)
        got = [e for e in sim.logger.events[n_events:] if e.kind == "action_reject"]
        assert len(got) == 1, f"{kind}/{reason}: action_reject が {len(got)} 件"
        assert got[0].payload == {"kind": kind, "reason": reason}
        assert got[0].agent_id == a.id
        want = RJ.sentence(kind, reason)
        assert want and want in _reject_memories(a), \
            f"{kind}/{reason}: 定型文が記憶に入っていない"


def test_sentences_have_no_digits_and_no_condition_words():
    """定型文は 2 要素だけ = 数字・時刻・実験条件語・機構語を 1 文字も含まない。"""
    banned = ("silent", "memory", "engaged", "ablate", "placebo", "seed",
              "実験", "条件", "閾値", "発火", "驚き", "エピソード", "モデル",
              "シミュレーション", "エージェント", "円", "分", "歩")
    for kind in RJ.KINDS:
        for reason in RJ.REASONS:
            s = RJ.sentence(kind, reason)
            if not s:
                continue
            assert not re.search(r"[0-9０-９]", s), f"{kind}/{reason}: 数字が入った"
            for w in banned:
                assert w not in s, f"{kind}/{reason}: 禁止語 {w} が入った"
    for act in RJ.PLAN_ACT_JP:
        for reason in RJ.PLAN_REASONS:
            s = RJ.plan_sentence(act, reason)
            assert s and not re.search(r"[0-9０-９]", s)


def test_sentence_is_a_pure_function_of_kind_and_reason(tmp_path, monkeypatch):
    """同じ拒否は**全実験条件で同一バイト列**(no-fingerprint の機械固定)。"""
    variants = (
        {},
        {"k.writeback": "off"},
        {"k.writeback": "sham"},
        {"ablate.propagation_off": "true"},
        {"ablate.cognitive_tier": "small"},
        {"controls.mode": "compute_matched"},
        {**ENGAGED_ON},
    )
    seen: dict[tuple[str, str], set[str]] = {}
    for i, extra in enumerate(variants):
        for kind, reason in TARGETS:
            sim = _sim(tmp_path, f"rj_pf{i}_{kind}_{reason}", n_steps=1,
                       **MEMORY, **_trigger_cfg(kind, reason), **extra)
            a = sim.agents[0]
            _trigger(sim, a, kind, reason, monkeypatch)
            texts = _reject_memories(a)
            assert len(texts) == 1
            seen.setdefault((kind, reason), set()).add(texts[0])
    for key, texts in seen.items():
        assert len(texts) == 1, f"{key}: 条件によって文面が変わった {texts}"


def test_notification_does_not_rescue_the_action(tmp_path):
    """通知は「知らせる」だけで世界を動かさない(拒否そのものは従来どおり成立する)。"""
    for level in (SILENT, MEMORY, ENGAGED):
        sim = _sim(tmp_path, f"rj_norescue_{level['cognition.rejection_notify']}",
                   n_steps=1, **level)
        a = sim.agents[0]
        a.money = 0.0
        sim.tools._open_venture(sim, a, {"name": "店", "offer": ""}, 0, 0)
        assert a.id not in sim.tools.ventures, "拒否されたのに出店できている"
        assert a.money == 0.0


# --------------------------------------------------------------------------- #
# (D) engaged 水準(検収基準 ③)
# --------------------------------------------------------------------------- #
def test_engaged_level_advances_the_cognitive_event(tmp_path):
    """engaged 水準は既存 note_plan_exception と同じ流儀で「今」へ前倒しする。"""
    sim = _sim(tmp_path, "rj_adv", n_steps=1, **ENGAGED, **ENGAGED_ON)
    a = sim.agents[0]
    sim.cogq.schedule(a.id, 9999, "periodic")
    a.money = 0.0
    sim.tools._open_venture(sim, a, {"name": "店", "offer": ""}, 0, 240)
    assert int(getattr(a, "_plan_exception", -1)) == 240, "前倒しの印が立っていない"
    rec = sim.cogq.pending[int(a.id)]
    assert rec["at"] == 240 and rec["reason"] == "internal", \
        "認知イベントが「今」へ前倒しされていない"
    assert sim._reject_state["advanced"] == 1
    assert sim._reject_state["degraded"] == 0


def test_engaged_level_degrades_to_memory_without_engaged(tmp_path):
    """cognition.engaged OFF のランでは memory へ縮退する(隠さず数える)。"""
    sim = _sim(tmp_path, "rj_deg", n_steps=1, **ENGAGED)      # engaged/fire は OFF
    a = sim.agents[0]
    a.money = 0.0
    sim.tools._open_venture(sim, a, {"name": "店", "offer": ""}, 0, 240)
    assert _reject_memories(a), "縮退しても記憶 1 行は入る(= memory と同一)"
    assert not hasattr(a, "_plan_exception"), "engaged OFF なのに前倒しが起きた"
    assert sim._reject_state["advanced"] == 0
    assert sim._reject_state["degraded"] == 1
    assert RJ.provenance(sim)["effective_level"] == RJ.MEMORY


def test_engaged_level_equals_memory_when_engaged_is_off(tmp_path):
    """縮退の実測: engaged 水準と memory 水準が L1 まで完全一致(engaged OFF のラン)。"""
    m = _sim(tmp_path, "rj_eqm", n_steps=72, n_agents=12, **MEMORY)
    m.run()
    e = _sim(tmp_path, "rj_eqe", n_steps=72, n_agents=12, **ENGAGED)
    e.run()
    assert _l1x(m) == _l1x(e), "engaged OFF のランで水準差が世界に出た(縮退が不完全)"


# --------------------------------------------------------------------------- #
# (E) plan_exception の失敗理由(検収基準 ④)
# --------------------------------------------------------------------------- #
def _block(start=8 * 60, end=9 * 60, place="food", act="meal", node="n", **extra):
    row = {"start": start, "end": end, "place": place, "act": act, "with": [],
           "purpose": DP.ACT_PURPOSE[act], "priority": "must",
           "flex": "slideable", "reason": "", "note": "", "node": node,
           "state": "todo", "slid": 0}
    row.update(extra)
    return row


def _plan(agent, blocks, day=0):
    agent._dayplan = {"schema": DP.SCHEMA, "day": day, "version": 1,
                      "model": "mock", "mood": "", "carry": "", "src": "llm",
                      "blocks": blocks, "cont": []}
    return agent._dayplan


def _why_sim(tmp_path, name, **ov):
    return _sim(tmp_path, name, n_steps=1, **ENGAGED, **ENGAGED_ON,
                **DAYPLAN_ON, **ov)


def test_replan_records_the_failure_reason(tmp_path):
    sim = _why_sim(tmp_path, "rj_why")
    a = sim.agents[0]
    b = _block()
    plan = _plan(a, [b])
    DP._replan(sim, a, plan, DP.cfg_of(sim), 0, 12 * 60, [b])
    assert RJ.why_of(a) == ("meal", "late")
    assert sim._reject_state["why_set"] == 1


def test_replan_classifies_a_missed_fixed_block(tmp_path):
    sim = _why_sim(tmp_path, "rj_why_missed")
    a = sim.agents[0]
    b = _block(act="work", flex="fixed", start=6 * 60, end=7 * 60)
    plan = _plan(a, [b])
    DP._replan(sim, a, plan, DP.cfg_of(sim), 0, 12 * 60, [b])
    assert RJ.why_of(a) == ("work", "missed")


def test_why_is_injected_as_exactly_one_situation_line(tmp_path):
    """注入は**状況行 1 行だけ**。REPLAN エピソード中以外では 1 バイトも足さない。"""
    sim = _why_sim(tmp_path, "rj_inject")
    a = sim.agents[0]
    RJ.note_why(sim, a, "meal", "late")
    assert RJ.why_line(sim, a) is None, "エピソード外なのに 1 行出た"
    a._engaged = EG._new_episode(EG.TALK, EG.T_SOCIAL, 0, 0, None, "mock")
    assert RJ.why_line(sim, a) is None, "会話エピソードに再計画の 1 行が出た"
    a._engaged = EG._new_episode(EG.REPLAN, EG.T_EXCEPTION, 0, 0, None, "mock")
    line = RJ.why_line(sim, a)
    assert line == "予定していた食事が時間の都合でできなかった。"
    kwargs = dict(place_name="どこか", surprise="solo", nearby_names=[],
                  sim_min=0, step=0)
    base = deliberate.build_prompt(a, **kwargs)
    with_line = deliberate.build_prompt(a, **kwargs, reject_line=line)
    assert with_line.count(line) == 1
    assert len(with_line.splitlines()) == len(base.splitlines()) + 1, \
        "注入が 1 行を超えた"


def test_why_is_demoted_when_the_replacement_block_starts(tmp_path):
    """降格(Masicampo & Baumeister): 代替ブロックが動き出したら失敗理由は消える。"""
    sim = _why_sim(tmp_path, "rj_demote")
    a = sim.agents[0]
    RJ.note_why(sim, a, "meal", "late")
    assert RJ.why_of(a) is not None
    node = next(n for n in sorted(sim.dests) if n != a.node)
    _plan(a, [_block(8 * 60, 9 * 60, place="food", node=node)])
    rng = sim.hub.stream("decide", a.id, 0)
    out = DP.plan_action(a, sim, 8 * 60 + 5, 0, rng)
    assert out is not None, "テスト前提が崩れた(代替ブロックが動いていない)"
    assert RJ.why_of(a) is None, "再計画が実ったのに失敗理由が残っている"
    assert sim._reject_state["why_cleared"] == 1


def test_why_is_not_carried_over_the_day_boundary(tmp_path):
    """未解決のまま日を跨いだら消す(朝の計画生成が日境界の単一作用点)。"""
    sim = _why_sim(tmp_path, "rj_carry")
    a = sim.agents[0]
    RJ.note_why(sim, a, "meal", "late")
    DP.apply(sim, a, 0, 1440 + 8 * 60, "{壊れた", "cid-1")   # 翌朝の計画生成
    assert RJ.why_of(a) is None, "失敗理由が翌日へ持ち越された"


def test_why_never_stored_when_silent(tmp_path):
    sim = _sim(tmp_path, "rj_why_silent", n_steps=1, **SILENT, **ENGAGED_ON,
               **DAYPLAN_ON)
    a = sim.agents[0]
    b = _block()
    DP._replan(sim, a, _plan(a, [b]), DP.cfg_of(sim), 0, 12 * 60, [b])
    assert getattr(a, RJ.WHY_KEY, None) is None
    assert int(getattr(a, "_plan_exception", -1)) == 12 * 60, \
        "silent が既存の plan_exception を壊した"


# --------------------------------------------------------------------------- #
# (F) ON ラン: 決定論・resume・k(検収基準 ⑤⑥⑦)
# --------------------------------------------------------------------------- #
# 常に move_home を主張させ、敷金を払えないので毎回 no_money で拒否される世界。
_MOVE_HOME = json.dumps({"action": "move_home", "area": ""}, ensure_ascii=False)
_P2_ON = {"freedom.p2.move_home": "true", "freedom.p2.deposit": 10 ** 9}
_ON_RUN = {**MEMORY, **_P2_ON}


def _fixed_run(tmp_path, name, n_steps=144, n_agents=15, **ov):
    sim = _sim(tmp_path, name, n_steps=n_steps, n_agents=n_agents, **ov)
    sim.llm = _FixedLLM(_MOVE_HOME)
    sim.run()
    return sim


def test_on_actually_fires_in_a_full_run(tmp_path):
    """テスト前提の確認: ON ランで実際に action_reject が立つ(空振り検収を防ぐ)。"""
    sim = _fixed_run(tmp_path, "rj_run", **_ON_RUN)
    got = _kind(sim, "action_reject")
    assert got, "ON ランで拒否が 1 件も起きていない(テストが空振り)"
    assert {e.payload["reason"] for e in got} == {"no_money"}
    summary = json.loads((tmp_path / "rj_run" / "summary.json")
                         .read_text(encoding="utf-8"))
    assert summary["rejection_notify"]["notified"] == len(got)
    assert summary["rejection_notify"]["level"] == "memory"


def test_on_is_deterministic_across_two_runs(tmp_path):
    a = _fixed_run(tmp_path, "rj_det_a", **_ON_RUN)
    b = _fixed_run(tmp_path, "rj_det_b", **_ON_RUN)
    assert _l1x(a) == _l1x(b), "通知 ON の決定論が崩れている"


def test_on_engaged_level_run_is_deterministic(tmp_path):
    ov = {**_P2_ON, **ENGAGED, **ENGAGED_ON}
    a = _fixed_run(tmp_path, "rj_dete_a", n_steps=72, n_agents=12, **ov)
    b = _fixed_run(tmp_path, "rj_dete_b", n_steps=72, n_agents=12, **ov)
    assert _l1x(a) == _l1x(b), "engaged 水準の決定論が崩れている"


def test_llm_call_count_is_k_invariant(tmp_path):
    """compute_matched 下で k=free と k=off の generate 呼数が完全一致(R1)。"""
    def _run(name, writeback):
        return _fixed_run(tmp_path, name, n_steps=110, n_agents=15,
                          **_ON_RUN, **{"controls.mode": "compute_matched",
                                        "k.writeback": writeback})
    free = _run("rj_k_free", "free")
    off = _run("rj_k_off", "off")
    assert free.llm.calls == off.llm.calls > 0, \
        f"呼数が k に依存(R1 違反): free={free.llm.calls} off={off.llm.calls}"


def test_engaged_level_llm_call_count_is_k_invariant(tmp_path):
    """engaged 水準(fire 経由の前倒しあり)でも呼数は k に依存しない。"""
    def _run(name, writeback):
        return _fixed_run(tmp_path, name, n_steps=110, n_agents=15,
                          **_P2_ON, **ENGAGED, **ENGAGED_ON,
                          **{"controls.mode": "compute_matched",
                             "k.writeback": writeback})
    free = _run("rj_ke_free", "free")
    off = _run("rj_ke_off", "off")
    assert free.llm.calls == off.llm.calls > 0


def test_resume_matches_straight(tmp_path):
    """通知 ON(engaged 水準 + fire/engaged ON)で resume==straight(parquet バイト一致)。

    ★ここだけ ``_FixedLLM`` を使わない: 固定スタブは call_id を 1 から振り直すので
      resume 側の ``llm_call_id`` 列が必ずずれる(機能とは無関係のスタブの性質)。
      mock なら P2 の menu 経由で拒否が自然に立つ(下の assert が空振りを止める)。
    """
    ov = {**_P2_ON, **ENGAGED, **ENGAGED_ON,
          "run.start_tod": "00:00", "run.natural_start": "true"}
    split, total = 60, 120
    straight_dir = tmp_path / "rj_straight"
    straight = Simulation(_cfg("rj_straight", total, 15, **ov),
                          out_dir=straight_dir)
    straight.run()
    assert _kind(straight, "action_reject"), "テスト前提が崩れた(拒否ゼロ)"

    d = tmp_path / "rj_resumed"
    sim1 = Simulation(_cfg("rj_resumed", split, 15, **ov,
                           **{"observer.checkpoint_every": split}), out_dir=d)
    for step in range(split):
        scheduler.run_step(sim1, step)
    checkpoint.save(sim1, split, d / "checkpoint" / f"ckpt-{split:06d}.pkl.gz")
    sim1.logger.flush_segment()
    sim2 = Simulation(_cfg("rj_resumed", total, 15, **ov,
                           **{"observer.checkpoint_every": split}), out_dir=d)
    sim2.run(resume_from=d)
    for stem in ("l1_events", "l2_metrics", "l3_snapshots"):
        sa = pq.read_table(straight_dir / f"{stem}.parquet").to_pylist()
        sb = pq.read_table(d / f"{stem}.parquet").to_pylist()
        assert sa == sb, f"{stem} 不一致(rejection_notify resume)"
    ja = json.loads((straight_dir / "summary.json").read_text(encoding="utf-8"))
    jb = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    assert ja["rejection_notify"] == jb["rejection_notify"], \
        "累積タリーが resume で straight と食い違う(中央管理の漏れ)"
