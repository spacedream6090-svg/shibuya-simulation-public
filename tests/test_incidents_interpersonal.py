"""H4(対人事件の収束化 ``incidents_interpersonal``)のテスト。

正典
  - docs/plans/body-incident-layer-plan.md **§3 対人行**(レート → **共在収束**
    = Birks / Groff の RAT。加害者候補 × 標的 × **監視者不在**。喧嘩 = 酒 × 密度 ×
    閉店放出。閉店 +1 時間の **+16%** が検証対象)
  - 同 **§6-3 ユーザー決定**「**H4はエージェントドリブンで設計**」
    = レート抽選の残滓を残さない = 「誰もいなければ起きない」を**構造で**保証する
  - 同 **§7**(分母の罠 = 犯罪は人口比でなく**共在機会比**で較正 / 管轄混同)

守るもの(検収基準の順)
  ① OFF(既定)= ゴールデン L1 バイト一致・キー不発生・state も属性も生えない
  ②★**エージェントドリブンの機械的保証**:
     (a) AST = 本 module 中の ``hub.stream`` 呼び出しは **1 か所だけ**・
         それは ``_pair_draw`` の中・その本体の**最初の文**は空ペアの早期 return
     (b) 共在が無い世界(知覚半径 0)では ``pair_prob=1.0`` でも事件 0 件・**抽選 0 回**
     (c) 監視の目(同席者・警察官・交番)が閾値以上なら**抽選そのものを行わない**
  ③ 既存 theft の世代交代: ON で ``diversity`` は窃盗を供給しない/
     **stream "crime" の消費列は 1 バイトも変わらない**
  ④ 通報層: 通報は目撃者/被害者の行為・傍観者効果(曖昧のみ人数減衰)・
     非緊急/誤通報が**内生**で出る・応答の無人マーカー
  ⑤ 定型文に数字も実験条件も無い(no-fingerprint の 2 段検査)
  ⑥ ON 同 seed 2 ラン一致 / resume == straight
  ⑦ LLM 呼数 ON/OFF 完全一致(FixedLLM)
  ⑧ 新 3 種が EVENT_KINDS と causality の両方に載る / registry 宣言
  ⑨ パターン検収(モック統計): **場所集中(ホットスポット)**と
     **反復被害(同一標的の再被害率 > 一様)**が創発する
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

from society import diversity as D
from society import incidents_interpersonal as M
from society import registry as R
from society.config import load_config
from society.engine import checkpoint, scheduler
from society.engine.simulation import Simulation
from society.observer import causality as C
from society.observer.schema import EVENT_KINDS

GOLDEN = Path(__file__).resolve().parent / "data" / "golden_baseline_l1.json"

# test_traces.py:45 / test_rumors.py と同じ「意図的な既定挙動追加」の中立化
_GOLDEN_NEUTRAL = {"economy.wages.自営": 0, "planning.enabled": "false",
                   "transit_ride.taxi.enabled": "false",
                   "transit_ride.bus.enabled": "false", "rules.enabled": "false"}

OFF = {"incidents_interpersonal.enabled": "false"}
ON = {"incidents_interpersonal.enabled": "true"}

#: 事件が実際に出る強さ(既定の較正値は「今日は何も起きない」が正しい = §7)。
#: パターン検収・通報層の検査はこの上書きの下で行う。
LOUD = {"incidents_interpersonal.enabled": "true",
        "incidents_interpersonal.theft.pair_prob": 0.25,
        "incidents_interpersonal.brawl.pair_prob": 0.05,
        "incidents_interpersonal.guardian.w_watcher": 0.02,
        "incidents_interpersonal.motive.money_low": 1000000.0}


# --------------------------------------------------------------------------- #
# 共通ヘルパ(test_traces.py と同型)
# --------------------------------------------------------------------------- #
def _cfg(name, n_steps=144, n_agents=20, **ov):
    dot = ["run.seed=42", f"run.n_agents={n_agents}", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock", "observer.snapshot_every=144"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def _sim(tmp_path, name, n_steps=144, n_agents=20, **ov):
    return Simulation(_cfg(name, n_steps, n_agents, **ov), out_dir=tmp_path / name)


def _run(tmp_path, name, n_steps=144, n_agents=20, **ov):
    sim = _sim(tmp_path, name, n_steps, n_agents, **ov)
    for step in range(n_steps):
        scheduler.run_step(sim, step)
    return sim


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _kind(sim, kind):
    return [e for e in sim.logger.events if e.kind == kind]


class _FixedLLM:
    """**プロンプト非依存**の巡回応答スタブ(test_traces / test_rumors と同型)。"""

    name = "fixed"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.hits = 0
        self.prompts: list[str] = []

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        out = self.responses[self.calls % len(self.responses)]
        self.calls += 1
        self.prompts.append(prompt)
        return out, str(self.calls), False


_TALK = [json.dumps({"action": "speak", "text": "こんにちは"}, ensure_ascii=False),
         json.dumps({"action": "stay"}, ensure_ascii=False)]


# --------------------------------------------------------------------------- #
# ① OFF = 現行と完全同値
# --------------------------------------------------------------------------- #
def _golden_run(tmp_path, name, **ov):
    """ゴールデン採取時と**同じ形**で回す(tests/test_scenario.py::_run と同一)。"""
    dot = ["run.seed=42", "run.n_agents=15", "run.n_steps=144", f"run.name={name}"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    sim = Simulation(load_config(dot), out_dir=tmp_path / name)
    sim.run()
    return sim


def test_off_matches_prechange_golden(tmp_path):
    """既定 OFF は変更前ゴールデン L1 と一字一句一致(seam が完全な no-op)。"""
    golden = json.load(open(GOLDEN, encoding="utf-8"))
    sim = _golden_run(tmp_path, "gold_off", **_GOLDEN_NEUTRAL)
    assert _l1(sim) == golden, "既定 OFF がゴールデンと不一致(H4 が no-op でない)"


def test_off_explicit_flag_matches_golden(tmp_path):
    """明示 OFF も同じ(トグルの読み取り自体が世界を触らない)。"""
    golden = json.load(open(GOLDEN, encoding="utf-8"))
    sim = _golden_run(tmp_path, "gold_off2", **dict(_GOLDEN_NEUTRAL, **OFF))
    assert _l1(sim) == golden


def test_off_grows_no_state_no_attribute_no_key(tmp_path):
    """OFF では sim に state が生えず・agent に属性が生えず・summary にキーが出ない。"""
    sim = _run(tmp_path, "off_state", n_steps=24, n_agents=12)
    assert getattr(sim, "_inc_state", None) is None
    assert getattr(sim, "_inc_koban_nodes", None) is None
    assert all(not hasattr(a, M.INTOX_KEY) for a in sim.agents)
    assert M.provenance(sim) is None


def test_off_emits_none_of_the_new_kinds(tmp_path):
    """OFF では新 3 種が 1 件も出ない。"""
    sim = _run(tmp_path, "off_kinds", n_steps=48, n_agents=15)
    for kind in (M.BRAWL, M.REPORT, M.RESPONSE):
        assert _kind(sim, kind) == []


# --------------------------------------------------------------------------- #
# ②(a) ★AST = 乱数を引く場所が 1 つだけ・そこは共在ペアを要求する
# --------------------------------------------------------------------------- #
def _module_ast():
    src = Path(M.__file__).read_text(encoding="utf-8")
    return ast.parse(src), src


def test_ast_single_stream_call_inside_pair_draw():
    """★本 module 中の ``hub.stream`` 呼び出しは **1 か所だけ**・``_pair_draw`` の中。

    これが「レート抽選の残滓を残さない」の機械的な担保である: 乱数を引く口が
    1 つしか無く、その口が**共在ペアの列**を引数に取ることを下のテストが固定する。
    """
    tree, _src = _module_ast()
    sites: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) \
                    and sub.func.attr == "stream":
                sites.append(node.name)
    assert sites == ["_pair_draw"], f"stream 呼び出しが想定外の場所にある: {sites}"


def test_ast_pair_draw_returns_early_on_empty_pairs():
    """★``_pair_draw`` の**最初の文**は「共在ペアが空なら return」でなければならない。

    = 共在が無ければ乱数は 1 本も引かれない(人口にも時間にも直接掛からない)。
    """
    tree, _src = _module_ast()
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_pair_draw")
    assert "pairs" in [a.arg for a in fn.args.args], "_pair_draw が pairs を取っていない"
    first = fn.body[1] if isinstance(fn.body[0], ast.Expr) else fn.body[0]
    assert isinstance(first, ast.If), "最初の文が if でない"
    assert isinstance(first.test, ast.UnaryOp) and isinstance(first.test.op, ast.Not)
    assert isinstance(first.test.operand, ast.Name) and first.test.operand.id == "pairs"
    assert len(first.body) == 1 and isinstance(first.body[0], ast.Return)


def test_ast_no_population_level_lottery():
    """★``sim.agents`` の走査が乱数と同じ関数に同居していないこと(レートの残滓の検知)。"""
    tree, _src = _module_ast()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        has_stream = any(isinstance(s, ast.Call) and isinstance(s.func, ast.Attribute)
                         and s.func.attr == "stream" for s in ast.walk(node))
        touches_agents = any(isinstance(s, ast.Attribute) and s.attr == "agents"
                             for s in ast.walk(node))
        assert not (has_stream and touches_agents), \
            f"{node.name} が「全個体の走査」と「抽選」を同居させている(レート抽選の残滓)"


# --------------------------------------------------------------------------- #
# ②(b)(c) ★共在が無ければ起きない / 監視の目が塞ぐ
# --------------------------------------------------------------------------- #
def test_no_copresence_no_incident_and_no_draw(tmp_path):
    """★知覚半径 0(= 誰とも共在しない)なら ``pair_prob=1.0`` でも事件 0 件・**抽選 0 回**。"""
    sim = _run(tmp_path, "isolated", n_steps=72, n_agents=20,
               **dict(LOUD, **{"world.perception_radius_m": 0.0,
                               "incidents_interpersonal.theft.pair_prob": 1.0,
                               "incidents_interpersonal.brawl.pair_prob": 1.0}))
    prov = M.provenance(sim)
    assert prov["candidates"] > 0, "候補が 0 では検査にならない(動機は成立している)"
    assert prov["pairs_seen"] == 0
    assert prov["draws"] == 0, "共在が無いのに抽選が走った(= レート抽選の残滓)"
    assert prov["incidents"] == 0
    assert _kind(sim, M.BRAWL) == []
    assert [e for e in _kind(sim, "crime")
            if (e.payload or {}).get("src") == "rat"] == []


def test_guardian_block_skips_the_draw_entirely(tmp_path):
    """★監視の目が閾値以上なら**抽選そのものを行わない**(事後フィルタではない)。

    ``guardian.block <= 0`` = 「常に監視されている」対照条件。``pair_prob=1.0`` でも
    ペアが 1 組も組まれず、抽選が 1 回も走らない = 監視が確率を掛け算で薄めるのではなく
    **制御フローを塞いでいる**ことの直接の証拠。
    """
    strict = dict(LOUD, **{"incidents_interpersonal.guardian.block": 0.0,
                           "incidents_interpersonal.theft.pair_prob": 1.0,
                           "incidents_interpersonal.brawl.pair_prob": 1.0})
    sim = _run(tmp_path, "guarded", n_steps=72, n_agents=25, **strict)
    prov = M.provenance(sim)
    assert prov["blocked_by_guardian"] > 0, "監視で塞がれたペアが 0(検査にならない)"
    assert prov["pairs_seen"] == 0 and prov["draws"] == 0
    assert prov["incidents"] == 0


def test_watched_pairs_never_become_incidents(tmp_path):
    """★同席者が 1 人でも居れば塞がる設定では、起きた事件の監視者数は必ず 0。"""
    strict = dict(LOUD, **{"incidents_interpersonal.guardian.w_watcher": 99.0,
                           "incidents_interpersonal.theft.pair_prob": 1.0,
                           "incidents_interpersonal.brawl.pair_prob": 1.0})
    sim = _run(tmp_path, "watched", n_steps=72, n_agents=25, **strict)
    prov = M.provenance(sim)
    assert prov["blocked_by_guardian"] > 0
    events = [e for e in _kind(sim, "crime")
              if (e.payload or {}).get("src") == "rat"] + _kind(sim, M.BRAWL)
    assert events, "事件が 0 件では検査にならない"
    assert all(int(e.payload["guardians"]) == 0 for e in events), \
        "監視者が居るのに事件が起きた(監視が事後フィルタになっている)"


def test_guardians_reduce_incidents_monotonically(tmp_path):
    """監視の重みを上げると事件は増えない(RAT の中心命題の単調性)。"""
    lo = M.provenance(_run(tmp_path, "g_lo", n_steps=72, n_agents=25, **LOUD))
    hi = M.provenance(_run(tmp_path, "g_hi", n_steps=72, n_agents=25,
                           **dict(LOUD, **{"incidents_interpersonal.guardian.w_watcher": 0.5})))
    assert hi["incidents"] <= lo["incidents"]
    # ★``blocked_by_guardian`` の**件数**は比べない: 事件が減ると世界の軌跡が変わり
    #   (所持金・位置)、そもそも走査されるペアの母数が変わるため。RAT の主張は
    #   「監視が増えれば事件が減る」であって「塞いだ回数が増える」ではない。
    assert hi["pairs_seen"] <= lo["pairs_seen"]


def test_police_presence_is_unconditional_block():
    """同ノードの警察官は無条件の抑止(``block`` を必ず超えるスコアを返す)。"""
    cfg = M.build_cfg({"enabled": True})

    class _Sim:                                    # 最小のダミー(traces OFF 相当)
        pass

    score = M.guardian_score(_Sim(), cfg, "n1", 0, True, frozenset())
    assert score > float(cfg["guardian"]["block"])


# --------------------------------------------------------------------------- #
# ② RAT の 3 要素が「実エージェント状態の純関数」であること
# --------------------------------------------------------------------------- #
class _A:
    """動機/適性の純関数だけを検査する最小ダミー(agent の必要フィールドのみ)。"""

    def __init__(self, **kw):
        self.id = kw.pop("id", 1)
        self.money = kw.pop("money", 5000.0)
        self.fatigue = kw.pop("fatigue", 0.0)
        self.has_phone = kw.pop("has_phone", True)
        self.arrears_days = kw.pop("arrears_days", 0)
        self.evicted = kw.pop("evicted", False)
        self.bankrupt_until = kw.pop("bankrupt_until", 0)
        self.states = kw.pop("states", {})
        for k, v in kw.items():
            setattr(self, k, v)


def test_motive_is_a_recoverable_state_not_a_label():
    """★動機は**一時的で回復可能**: 手持ちが戻れば 0 に戻る(人格ラベルではない)。"""
    cfg = M.build_cfg({"enabled": True})
    poor = _A(money=0.0)
    rich = _A(money=100000.0)
    assert M.motive_of(poor, 0, cfg) > M.motive_of(rich, 0, cfg)
    poor.money = 100000.0                          # 状態が戻る
    assert M.motive_of(poor, 0, cfg) == M.motive_of(rich, 0, cfg)


def test_motive_ignores_grievance_by_default():
    """★既定 ``w_grievance=0`` = 構成概念(不満)を発火判断に食わせない(R1)。"""
    cfg = M.build_cfg({"enabled": True})
    calm = _A(states={"grievance": 0.0})
    angry = _A(states={"grievance": 1.0})
    assert M.motive_of(calm, 0, cfg) == M.motive_of(angry, 0, cfg)
    on = M.build_cfg({"enabled": True, "motive": {"w_grievance": 0.5}})
    assert M.motive_of(angry, 0, on) > M.motive_of(calm, 0, on)


def test_suitability_reads_cash_phone_and_isolation():
    """標的の適性は VIVA の観測可能な代理から作る(traits は 1 つも読まない)。"""
    cfg = M.build_cfg({"enabled": True})
    poor_alone = M.suitability_of(_A(money=0.0, has_phone=False), 0, cfg, True)
    rich_alone = M.suitability_of(_A(money=100000.0), 0, cfg, True)
    rich_crowd = M.suitability_of(_A(money=100000.0), 0, cfg, False)
    assert rich_alone > rich_crowd > poor_alone


def test_intoxication_is_a_decaying_mark():
    """酩酊は減衰する一時的な印(``steps`` を過ぎれば 0 = 回復する)。"""
    cfg = M.build_cfg({"enabled": True})
    a = _A()
    setattr(a, M.INTOX_KEY, 10)
    assert M.intox_of(a, 0, cfg) > M.intox_of(a, 5, cfg) > 0.0
    assert M.intox_of(a, 10, cfg) == 0.0
    assert M.intox_of(_A(), 0, cfg) == 0.0         # 印が無ければ 0(属性を生やさない)


# --------------------------------------------------------------------------- #
# ③ 既存 theft の世代交代
# --------------------------------------------------------------------------- #
def test_supersede_flag_is_false_when_off():
    """★OFF では ``superseded_theft`` が必ず False = diversity の制御フローが不変。"""

    class _Sim:
        cfg = {}

    sim = _Sim()
    sim.incidentscfg = M.build_cfg({"enabled": False})
    assert M.superseded_theft(sim) is False
    sim.incidentscfg = M.build_cfg({"enabled": True})
    assert M.superseded_theft(sim) is True
    sim.incidentscfg = M.build_cfg({"enabled": True,
                                    "theft": {"supersede_diversity": False}})
    assert M.superseded_theft(sim) is False


def _crime_draws(sim):
    """``diversity`` が消費した stream "crime" の draw 列(世代交代で不変であること)。"""
    return [round(float(sim.hub.stream("crime", a.id, 0).random()), 12)
            for a in sorted(sim.agents, key=lambda x: x.id)]


def test_diversity_theft_is_superseded_but_stream_is_untouched(tmp_path):
    """ON で diversity は窃盗を供給しない。**stream "crime" の消費列は 1 バイトも変わらない**。"""
    base = {"society_diversity.enabled": "true",
            "society_diversity.crime_prob": 0.5,
            "society_diversity.nuisance_prob": 0.0}
    off = _run(tmp_path, "sup_off", n_steps=144, n_agents=25, **base)
    on = _run(tmp_path, "sup_on", n_steps=144, n_agents=25,
              **dict(base, **{"incidents_interpersonal.enabled": "true",
                              "incidents_interpersonal.theft.pair_prob": 0.0,
                              "incidents_interpersonal.brawl.pair_prob": 0.0}))
    legacy_off = [e for e in _kind(off, "crime")
                  if (e.payload or {}).get("src") != "rat"]
    legacy_on = [e for e in _kind(on, "crime")
                 if (e.payload or {}).get("src") != "rat"]
    assert legacy_off, "対照側で従来の窃盗が 0 件(検査にならない)"
    assert legacy_on == [], "世代交代したのに従来の窃盗が出ている(併存している)"
    # 抽選そのものは飛ばしていない = 同じキーの stream が同じ値を返す(消費列不変)
    assert _crime_draws(off) == _crime_draws(on)


def test_rat_theft_uses_the_existing_crime_kind_with_a_generation_marker(tmp_path):
    """RAT の窃盗は既存 ``crime`` 種で出す(痕跡・危険地帯・解析がそのまま効く)+ 世代の印。"""
    sim = _run(tmp_path, "rat_theft", n_steps=144, n_agents=25, **LOUD)
    rat = [e for e in _kind(sim, "crime") if (e.payload or {}).get("src") == "rat"]
    assert rat, "RAT の窃盗が 1 件も出ていない(LOUD の設定が効いていない)"
    p = rat[0].payload
    assert p["kind"] == "theft" and "victim" in p and "offender" in p
    assert p["offender"] == rat[0].agent_id        # agent_id = 加害者(既存の規約)


def test_rat_theft_moves_money_and_feeds_danger_history(tmp_path):
    """窃盗は現金を動かし、危険地帯の履歴(``diversity``)にも積まれる。"""
    sim = _run(tmp_path, "rat_money", n_steps=144, n_agents=25, **LOUD)
    rat = [e for e in _kind(sim, "crime") if (e.payload or {}).get("src") == "rat"]
    assert any(float(e.payload["amount"]) > 0.0 for e in rat)
    counter = getattr(sim, "_diversity_crime_nodes", None)
    assert counter is not None and sum(counter.values()) >= len(rat)


# --------------------------------------------------------------------------- #
# ★L1 は 1 行 + 前兆状態を同梱(内生性の機械検証)
# --------------------------------------------------------------------------- #
_PRECURSORS = ("guardians", "density", "intox", "closing", "motive", "suitability")


def test_incident_payload_carries_the_precursor_state(tmp_path):
    """★事件の L1 payload に前兆状態(監視者数・密度・飲酒・閉店フラグ)が必ず載る。"""
    sim = _run(tmp_path, "precursor", n_steps=144, n_agents=25, **LOUD)
    rat = [e for e in _kind(sim, "crime") if (e.payload or {}).get("src") == "rat"]
    assert rat
    for e in rat + _kind(sim, M.BRAWL):
        for key in _PRECURSORS:
            assert key in e.payload, f"{e.kind} の payload に {key} が無い"
        assert e.payload["guardians"] * 0.34 < 1.0   # 監視スコアが block 未満だった証跡


# --------------------------------------------------------------------------- #
# ④ 通報層
# --------------------------------------------------------------------------- #
def test_report_is_an_action_of_a_present_agent(tmp_path):
    """通報は**居合わせた個体の行為**(agent_id は実在の個体・-1 にならない)。"""
    sim = _run(tmp_path, "report", n_steps=144, n_agents=25, **LOUD)
    reports = _kind(sim, M.REPORT)
    assert reports, "通報が 1 件も出ていない"
    ids = {a.id for a in sim.agents}
    assert all(e.agent_id in ids for e in reports)
    assert all(e.payload["witnesses"] >= 0 for e in reports)


def test_bystander_effect_is_ambiguity_conditional():
    """★**個人**の通報確率は曖昧・軽微な状況だけ人数で減衰する(危険事態は落ちない)。"""
    cfg = M.build_cfg({"enabled": True})
    amb1 = M.report_p_ind(cfg, False, False, 1, False)
    amb9 = M.report_p_ind(cfg, False, False, 9, False)
    dan1 = M.report_p_ind(cfg, False, False, 1, True)
    dan9 = M.report_p_ind(cfg, False, False, 9, True)
    assert amb9 < amb1, "曖昧な状況で人数減衰が効いていない"
    assert dan9 == dan1, "危険事態で人数減衰が効いてしまっている"
    assert M.report_p_ind(cfg, True, False, 9, False) > amb9     # 被害者は傍観者ではない
    assert M.report_p_ind(cfg, False, True, 9, False) > amb9     # 関係者は倍率>1


def test_group_intervention_rate_matches_the_cctv_anchor():
    """★**集団**の「誰かが通報する」確率が実測アンカー(0.85-0.95・介入率 91%)の帯に乗る。

    個人の p を人数で落としても集団の確率は上がる = 実 CCTV 研究の所見そのもの。
    """
    cfg = M.build_cfg({"enabled": True})
    got = {}
    for n in (3, 5, 8):
        ps = [M.report_p_ind(cfg, False, False, n, False) for _ in range(n)]
        got[n] = M.report_p_group(ps)
    assert 0.80 <= got[3] <= 0.90, got
    assert 0.85 <= got[5] <= 0.95, got
    assert 0.90 <= got[8] <= 0.98, got
    assert got[3] < got[5] < got[8], "集団の確率が人数で上がっていない"


def test_nonurgent_and_unclear_reports_are_endogenous(tmp_path):
    """非緊急/誤通報は**捏造せず内生で出る**(事後に気づいた被害者・遠い目撃者)。"""
    sim = _run(tmp_path, "nonurgent", n_steps=144, n_agents=30, **LOUD)
    prov = M.provenance(sim)
    reports = _kind(sim, M.REPORT)
    assert reports
    assert all(isinstance(e.payload["urgent"], bool) for e in reports)
    assert 0.0 <= prov["nonurgent_rate"] <= 1.0
    assert prov["reports_nonurgent"] > 0, "非緊急通報が 1 件も出ない(内生になっていない)"
    for e in reports:                              # 特定できない回は加害者を書かない
        if e.payload["unclear"]:
            assert e.payload["offender"] is None
    # ★窃盗の被害者の自己通報は「事後に気づいた」= 非緊急かつ加害者を特定できない
    self_theft = [e for e in reports
                  if e.payload["self_report"] and e.payload["incident"] == "theft"]
    assert self_theft
    assert all((not e.payload["urgent"]) and e.payload["unclear"]
               and e.payload["offender"] is None for e in self_theft)


def test_brawl_participants_do_not_report_themselves(tmp_path):
    """★喧嘩の当事者は通報者にならない(もみ合っている)= 通報は目撃者の行為。"""
    sim = _run(tmp_path, "brawl_report", n_steps=432, n_agents=30,
               **dict(LOUD, **{"incidents_interpersonal.brawl.pair_prob": 0.5,
                               "incidents_interpersonal.theft.pair_prob": 0.0}))
    reports = [e for e in _kind(sim, M.REPORT)
               if e.payload["incident"] == M.BRAWL]
    if not reports:                                # 目撃者が居なければ通報は起きない(正しい)
        assert M.provenance(sim)["by_kind"].get(M.BRAWL, 0) >= 0
        return
    assert all(e.payload["self_report"] is False for e in reports)


def test_response_marks_unstaffed_honestly(tmp_path):
    """応答は当直が居なければ ``agent_id=-1`` + ``unstaffed=true``(欠測を偽らない)。"""
    sim = _run(tmp_path, "respond", n_steps=144, n_agents=25, **LOUD)
    res = _kind(sim, M.RESPONSE)
    assert res, "応答が 1 件も出ていない"
    for e in res:
        if e.payload["unstaffed"]:
            assert e.agent_id == -1
        else:
            assert e.agent_id >= 0 and e.payload["response_min"] is not None


def test_report_can_be_switched_off(tmp_path):
    """``report.enabled=false`` では事件だけが起き、通報も応答も 0 件。"""
    sim = _run(tmp_path, "noreport", n_steps=144, n_agents=25,
               **dict(LOUD, **{"incidents_interpersonal.report.enabled": "false"}))
    assert M.provenance(sim)["incidents"] > 0
    assert _kind(sim, M.REPORT) == [] and _kind(sim, M.RESPONSE) == []


def test_detention_is_off_by_default(tmp_path):
    """★既定 ``detain_steps=0`` = 既存の勾留 seam を 1 度も踏まない(呼数を動かさない)。"""
    sim = _run(tmp_path, "nodetain", n_steps=144, n_agents=25, **LOUD)
    assert M.provenance(sim)["detained"] == 0
    assert all(int(getattr(a, "detained_until", 0) or 0) == 0 for a in sim.agents)


# --------------------------------------------------------------------------- #
# ⑤ 定型文の no-fingerprint(2 段検査 = traces / rumors の流儀)
# --------------------------------------------------------------------------- #
def test_sentences_are_pure_and_carry_no_numbers():
    """記憶の 1 行は**出来事の種類だけの純関数**で、数字も記号も実験条件も含まない。"""
    for key, text in M.TEXTS.items():
        assert M.sentence(key) == text
        assert not any(ch.isdigit() for ch in text), f"{key} に数字がある"
        assert "{" not in text and "}" not in text
    assert M.sentence("そんな種は無い") == ""      # 語彙外は捏造しない


def test_sentences_are_identical_across_experiment_conditions():
    """全実験条件で同一バイト列(config も k も特性も読まない)。"""
    seen = set()
    for prob in (0.0, 0.5, 1.0):
        M.build_cfg({"enabled": True, "theft": {"pair_prob": prob}})
        seen.add(tuple(sorted((k, M.sentence(k)) for k in M.TEXTS)))
    assert len(seen) == 1


def test_memory_lines_reach_the_participants(tmp_path):
    """当事者・目撃者の記憶に定型 1 行が入る(プロンプトの**欄**は増やさない)。"""
    sim = _run(tmp_path, "memory", n_steps=144, n_agents=25, **LOUD)
    texts = {m.text for a in sim.agents
             for m in list(a.mem.buffer) + list(a.mem.episodes)}
    assert M.sentence("theft_victim") in texts or M.sentence("brawl") in texts
    assert M.sentence("witness") in texts or M.sentence("report") in texts


# --------------------------------------------------------------------------- #
# ⑥ 決定論 / resume
# --------------------------------------------------------------------------- #
def test_on_is_deterministic_across_two_runs(tmp_path):
    """ON でも同 seed 2 ランは L1 バイト一致。"""
    a = _run(tmp_path, "det_a", n_steps=72, n_agents=25, **LOUD)
    b = _run(tmp_path, "det_b", n_steps=72, n_agents=25, **LOUD)
    assert _l1(a) == _l1(b)
    assert M.provenance(a) == M.provenance(b)


def test_resume_equals_straight(tmp_path):
    """resume == straight(state を checkpoint が中央管理している)。"""
    straight = _run(tmp_path, "st", n_steps=72, n_agents=25, **LOUD)
    half = _sim(tmp_path, "rs", n_steps=72, n_agents=25, **LOUD)
    for step in range(36):
        scheduler.run_step(half, step)
    path = checkpoint.save(half, 35, tmp_path / "rs" / "ck.pkl.gz")
    rest = _sim(tmp_path, "rs2", n_steps=72, n_agents=25, **LOUD)
    checkpoint.load(rest, path)
    for step in range(36, 72):
        scheduler.run_step(rest, step)
    tail_a = [r for r in _l1(straight) if r[0] >= 36]
    tail_b = [r for r in _l1(rest) if r[0] >= 36]
    assert tail_a == tail_b
    prov_a, prov_b = M.provenance(straight), M.provenance(rest)
    for key in ("incidents", "reports", "responses", "by_kind", "victims_unique"):
        assert prov_a[key] == prov_b[key], f"resume で {key} が食い違う"


# --------------------------------------------------------------------------- #
# ⑦ LLM 呼数(ON/OFF 完全一致 = generate() の呼び出しサイトを 1 つも足していない)
# --------------------------------------------------------------------------- #
def _fixed_run(tmp_path, name, n_steps, n_agents, **ov):
    sim = _sim(tmp_path, name, n_steps, n_agents, **ov)
    llm = _FixedLLM(_TALK)
    sim.llm = llm
    for step in range(n_steps):
        scheduler.run_step(sim, step)
    return sim, llm


def test_llm_call_count_is_identical_on_and_off(tmp_path):
    """★LLM 呼数の増分ゼロ(プロンプト非依存スタブで ON/OFF を突き合わせる)。"""
    _a, llm_off = _fixed_run(tmp_path, "llm_off", 48, 20, **OFF)
    _b, llm_on = _fixed_run(tmp_path, "llm_on", 48, 20,
                            **dict(LOUD, **{"incidents_interpersonal.report.detain_steps": 0}))
    assert llm_on.calls == llm_off.calls, "H4 ON で LLM 呼数が動いた"


def test_module_has_no_generate_call_site():
    """本 module に ``generate(`` の呼び出しサイトが 1 つも無い(AST)。"""
    tree, _src = _module_ast()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr != "generate", "generate() の呼び出しサイトがある"


# --------------------------------------------------------------------------- #
# ⑧ 契約(EVENT_KINDS / causality / registry)
# --------------------------------------------------------------------------- #
def test_new_kinds_are_registered_and_classified():
    """新 3 種が EVENT_KINDS(材料側 registration)と causality の両方に載っている。"""
    for kind in (M.BRAWL, M.REPORT, M.RESPONSE):
        assert kind in EVENT_KINDS, f"{kind} が schema に未登録"
        assert C.CAUSE_OF_KIND[kind] == C.AGENT, f"{kind} が agent に分類されていない"
    assert C.CAUSE_OF_KIND[M.THEFT_KIND] == C.AGENT   # 既存種を再利用している


def test_registry_declares_every_toggle():
    """すべての bool トグルがレジストリに宣言済み(未宣言があれば fail)。"""
    undeclared = [k for k in R.undeclared_toggles(load_config())
                  if k.startswith("incidents_interpersonal")]
    assert undeclared == []
    feat = {f.id: f for f in R.FEATURES}["incidents_interpersonal.enabled"]
    assert feat.repro_tier == "strict" and feat.affects_k is False


def test_config_default_is_off():
    """出荷 config の既定は OFF(= 観察ランに勝手に混ざらない)。"""
    cfg = load_config()
    assert bool(cfg.incidents_interpersonal.enabled) is False
    assert M.build_cfg(cfg.get("incidents_interpersonal"))["enabled"] is False


# --------------------------------------------------------------------------- #
# ★路上生活者の尊厳規約(street_life の明文条件の継承)
# --------------------------------------------------------------------------- #
def test_rough_sleepers_never_appear_in_incidents(tmp_path):
    """★路上生活者の id が事件・通報の payload に 1 度も現れない(規約 2 の継承)。"""
    sim = _run(tmp_path, "dignity", n_steps=144, n_agents=30,
               **dict(LOUD, **{"street_life.enabled": "true"}))
    from society import street_life as SL
    excluded = SL.rough_sleeper_ids(sim)
    if not excluded:                               # 名簿に居なければ規約は自明に成立
        return
    for kind in ("crime", M.BRAWL, M.REPORT, M.RESPONSE):
        for e in _kind(sim, kind):
            if kind == "crime" and (e.payload or {}).get("src") != "rat":
                continue                           # 従来経路は diversity の担当
            assert int(e.agent_id) not in excluded
            for key in ("victim", "offender", "other", "caller"):
                val = (e.payload or {}).get(key)
                if isinstance(val, int):
                    assert val not in excluded


def test_module_source_has_no_stigma_words():
    """尊厳規約 1: 蔑称・スティグマ語を module の文字列に 1 つも書かない。"""
    _tree, src = _module_ast()
    for word in ("浮浪", "ホームレス", "乞食", "不審者", "犯罪者", "非行少年"):
        assert word not in src, f"禁止語 {word} が module に書かれている"


# --------------------------------------------------------------------------- #
# ⑨ パターン検収(モック統計。★seed 固定なので判定は決定論)
# --------------------------------------------------------------------------- #
def _pattern_sim(tmp_path):
    return _run(tmp_path, "pattern", n_steps=432, n_agents=30, **LOUD)


def test_hotspots_emerge_place_concentration(tmp_path):
    """★場所集中(ホットスポット)が創発する = ノード分布が一様より集中している。"""
    prov = M.provenance(_pattern_sim(tmp_path))
    n_inc, n_nodes = prov["incidents"], prov["nodes_unique"]
    assert n_inc >= 20, f"件数が少なすぎて統計にならない({n_inc})"
    assert n_nodes >= 2, "ノードが 1 つでは集中の検定にならない"
    uniform_hhi = 1.0 / n_nodes                    # 一様分布の HHI
    assert prov["node_hhi"] > uniform_hhi * 1.2, (
        f"場所集中が創発していない(HHI {prov['node_hhi']} vs 一様 {uniform_hhi})")


def test_repeat_victimization_emerges(tmp_path):
    """★反復被害(同一標的の再被害率 > 一様)が創発する。"""
    prov = M.provenance(_pattern_sim(tmp_path))
    n_inc, n_victims = prov["incidents"], prov["victims_unique"]
    assert n_inc >= 20
    assert n_victims >= 2
    assert prov["victim_repeat_rate"] > 0.0, "同じ標的が 2 度と狙われていない"
    uniform_hhi = 1.0 / n_victims
    assert prov["victim_hhi"] > uniform_hhi * 1.2, (
        f"反復被害が一様を超えていない(HHI {prov['victim_hhi']} vs 一様 {uniform_hhi})")


def test_closing_release_boosts_brawl_probability():
    """★閉店 +1 時間の弾性(+16%)が確率へ掛かる帯として実装されている。"""
    cfg = M.build_cfg({"enabled": True})
    close_min = cfg["intox"]["close_hour"] * 60
    assert M._closing_window(cfg, close_min) is True
    assert M._closing_window(cfg, close_min + 59) is True
    assert M._closing_window(cfg, close_min + 61) is False
    assert M._closing_window(cfg, close_min - 10) is False
    assert abs(float(cfg["brawl"]["closing_boost"]) - 0.16) < 1e-9


def test_brawl_requires_drink_density_and_night(tmp_path):
    """喧嘩は酒 × 密度 × 夜(閉店放出を含む)の連言で、密度条件を上げれば減る。"""
    lo = M.provenance(_run(tmp_path, "b_lo", n_steps=288, n_agents=30, **LOUD))
    hi = M.provenance(_run(tmp_path, "b_hi", n_steps=288, n_agents=30,
                           **dict(LOUD, **{"incidents_interpersonal.brawl.density_min": 99})))
    assert hi["by_kind"].get(M.BRAWL, 0) == 0
    assert hi["by_kind"].get(M.BRAWL, 0) <= lo["by_kind"].get(M.BRAWL, 0)


# --------------------------------------------------------------------------- #
# 較正(§7 の分母)
# --------------------------------------------------------------------------- #
def test_calibrated_default_is_quiet(tmp_path):
    """★既定の較正値では「今日は何も起きない」が正しい(§7 の災害映画化の防止)。"""
    sim = _run(tmp_path, "quiet", n_steps=144, n_agents=25, **ON)
    prov = M.provenance(sim)
    assert prov["pairs_seen"] > 0, "共在の機会そのものが 0(検査にならない)"
    assert prov["per_person_day"] <= 3e-4, (
        f"既定較正で事件が多すぎる(許容帯の上限 3e-4 超: {prov['per_person_day']})")


def test_provenance_reports_the_copresence_denominator(tmp_path):
    """★較正の分母は人口比ではなく**共在機会比**(§7)= 必ず summary に出す。"""
    sim = _run(tmp_path, "denom", n_steps=72, n_agents=25, **LOUD)
    prov = M.provenance(sim)
    for key in ("candidates", "pairs_seen", "weight_sum", "weight_by_kind",
                "draws", "per_pair_step", "per_person_day", "days_elapsed"):
        assert key in prov, f"provenance に {key} が無い"
    assert prov["draws"] <= prov["pairs_seen"]


def test_summary_key_absent_when_off(tmp_path):
    """OFF では summary にキー自体を出さない(既存ランと同形)。"""
    assert M.provenance(_run(tmp_path, "sum_off", n_steps=24, n_agents=12)) is None


# --------------------------------------------------------------------------- #
# diversity 側の最小結線が壊れていないこと
# --------------------------------------------------------------------------- #
def test_diversity_note_crime_node_is_public_and_safe(tmp_path):
    """``diversity.note_crime_node`` は公開口で、あちらが OFF でも安全に呼べる。"""
    sim = _sim(tmp_path, "note", n_steps=1, n_agents=5)
    D.note_crime_node(sim, "nodeA")
    D.note_crime_node(sim, "nodeA")
    assert sim._diversity_crime_nodes["nodeA"] == 2
    assert D.is_danger(sim, "nodeA") is False      # diversity OFF なら行き先は変わらない
