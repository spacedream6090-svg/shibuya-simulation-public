"""真偽台帳ミニマル 第73バッチ(Part B)のテスト。

検収基準(タスク 2026-07-31):
  (R1)    既定 OFF = 純粋既定と L1 完全一致(belief_* 0 件・L2 に belief_* 列なし・
          プロンプト不変・台帳サイドカーを作らない)。ON でも **k(LLM 呼数)が不変**。
  (漏洩)  3 点セット: ① 静的検査(cognition が台帳 module を import しない)
          ② 実行時アサーション(全プロンプトの関門 CachedLLM で canary 検査)
          ③ canary テスト(一意 canary を持つ fact を注入し、llm_journal の**全プロンプト**に
             一切現れない)。
  (台帳)  fact 種はデータ駆動。場所・発生 step・真値・直接目撃可能条件(半径×窓)を持つ。
  (信念)  mock ランで belief_update(目撃)と belief_transmit(伝聞)が実際に出る。
          伝播木(親ノード)が再構成でき、ホップで確信度が減衰し値が無情報 0.5 へ寄る。
  (検証)  固定応答 LLM で go / ask / net の 3 種が受理・記録される。OFF では無視される。
  (registry) beliefs.* が journal 等級で宣言済み・未宣言検出が緑。
  (resume) ON ランの checkpoint 再開で信念状態・台帳・L1 が一致。
検証は mock / 固定応答スタブのみ(実 LLM 禁止)。
"""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from society import registry as R
from society import truth_ledger as TL
from society.cognition import deliberate
from society.config import load_config
from society.engine import scheduler
from society.engine.simulation import Simulation
from society.llm.cache import CachedLLM
from society.llm.journal import iter_records

SRC = Path(__file__).resolve().parents[1] / "src" / "society"
BELIEF_COLS = ("belief_facts_total", "belief_distance_mean",
               "belief_verified_rate", "belief_hop_mean", "belief_holders_mean")


# --------------------------------------------------------------------------- #
# 共通ヘルパ
# --------------------------------------------------------------------------- #
def _cfg(name: str, n_steps: int = 144, n_agents: int = 40, **ov):
    dot = ["run.seed=42", f"run.n_agents={n_agents}", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def _run(tmp_path, name: str, n_steps: int = 144, n_agents: int = 40, **ov):
    out = tmp_path / name
    sim = Simulation(_cfg(name, n_steps, n_agents, **ov), out_dir=out)
    summary = sim.run()
    return sim, out, summary


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _events(sim, kind):
    return [e for e in sim.logger.events if e.kind == kind]


def _l2_cols(out_dir):
    return set(pq.read_table(out_dir / "l2_metrics.parquet").column_names)


# --------------------------------------------------------------------------- #
# (R1) 既定 OFF = 完全 no-op / ON でも k 不変
# --------------------------------------------------------------------------- #
def test_default_is_off():
    cfg = load_config()
    assert bool(cfg.beliefs.enabled) is False
    assert bool(cfg.beliefs.verify_actions) is False


def test_off_is_byte_identical_and_leaves_no_trace(tmp_path):
    """OFF(既定)は L1 バイト一致・belief_* 0 件・L2 に列なし・台帳サイドカー無し。"""
    base, out_b, _ = _run(tmp_path, "off_base", n_steps=48, n_agents=30)
    off, out_o, _ = _run(tmp_path, "off_expl", n_steps=48, n_agents=30,
                         **{"beliefs.enabled": "false"})
    assert _l1(off) == _l1(base), "beliefs=false を明示しただけで L1 が動いた"
    assert not [e for e in off.logger.events if e.kind.startswith("belief")]
    assert not (BELIEF_COLS[0] in _l2_cols(out_o)), "OFF なのに L2 に belief 列がある"
    assert not (out_o / TL.LEDGER_NAME).exists(), "OFF なのに台帳サイドカーが出ている"
    # 状態も生やさない(fact/belief/pending の属性が 1 つも付かない)
    assert getattr(off, "_tl_facts", None) is None
    assert all(getattr(a, "_fact_beliefs", None) is None for a in off.agents)


def test_on_adds_only_belief_events_and_k_is_invariant(tmp_path):
    """ON は L1 に belief_* を足すだけ(他イベントはバイト一致)。LLM 呼数は完全一致。"""
    off, _o, s_off = _run(tmp_path, "k_off", n_steps=48, n_agents=30)
    on, out_on, s_on = _run(tmp_path, "k_on", n_steps=48, n_agents=30,
                            **{"beliefs.enabled": "true"})
    assert s_on["llm_calls"] == s_off["llm_calls"] > 0, \
        f"beliefs ON で呼数が変わった(R1 違反): {s_on['llm_calls']} != {s_off['llm_calls']}"
    stripped = [r for r in _l1(on) if not r[2].startswith("belief")]
    assert stripped == _l1(off), "belief_* 以外の L1 が動いた(世界を歪めている)"
    assert BELIEF_COLS[0] in _l2_cols(out_on), "ON なのに L2 に belief 列が無い"
    assert set(BELIEF_COLS) <= _l2_cols(out_on)


def test_verify_actions_keeps_k_invariant(tmp_path):
    """verify_actions ON はプロンプトに 1 行足すが **LLM 呼数は増やさない**(R1)。"""
    _off, _o, s_off = _run(tmp_path, "vk_off", n_steps=48, n_agents=30)
    _on, _n, s_on = _run(tmp_path, "vk_on", n_steps=48, n_agents=30,
                         **{"beliefs.enabled": "true",
                            "beliefs.verify_actions": "true"})
    assert s_on["llm_calls"] == s_off["llm_calls"] > 0


def test_on_is_deterministic(tmp_path):
    a, _oa, _sa = _run(tmp_path, "det_a", n_steps=48, n_agents=30,
                       **{"beliefs.enabled": "true"})
    b, _ob, _sb = _run(tmp_path, "det_b", n_steps=48, n_agents=30,
                       **{"beliefs.enabled": "true"})
    assert _l1(a) == _l1(b), "ON が非決定(乱数を引いている疑い)"


# --------------------------------------------------------------------------- #
# (漏洩 ①) 静的検査: プロンプト構築層は台帳を知らない
# --------------------------------------------------------------------------- #
# cognition/ に現れてはならない識別子(台帳 module 名 + 台帳/信念の状態属性名 + canary)。
_LEDGER_TOKENS = ("truth_ledger", "_tl_facts", "_tl_keys", "_tl_pending",
                  "_fact_beliefs", "facts_of", "beliefs_of", "CANARY_PREFIX",
                  "register_canary")


def test_cognition_never_references_the_ledger():
    """静的検査: プロンプト構築層 cognition/ が台帳 module を import・参照しないこと。

    build_prompt が受け取るのは `verify_actions: bool` 1 個だけで、fact も belief も
    値も ID も渡らない = 構造的にプロンプトへ入りようがない。"""
    hits = []
    for path in sorted((SRC / "cognition").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for tok in _LEDGER_TOKENS:
            if tok in text:
                hits.append(f"{path.name}: {tok}")
    assert hits == [], f"cognition が真偽台帳に触れている(漏洩経路の芽): {hits}"


def test_ledger_module_lives_outside_checked_dirs():
    """台帳の実装は src/society 直下(no-fingerprint 契約の CHECKED_DIRS 外)に置く。"""
    assert (SRC / "truth_ledger.py").exists()
    for d in ("engine", "cognition", "actions", "labeling", "world"):
        assert not (SRC / d / "truth_ledger.py").exists()


def test_build_prompt_signature_takes_only_a_bool():
    """build_prompt の検証行動 seam は bool 1 個(オブジェクトを渡す口が無い)。"""
    import inspect
    sig = inspect.signature(deliberate.build_prompt)
    assert sig.parameters["verify_actions"].default is False
    assert not [p for p in sig.parameters
                if "fact" in p or "belief" in p or "ledger" in p]


# --------------------------------------------------------------------------- #
# (漏洩 ②) 実行時アサーション: 全プロンプトの関門で canary を検査する
# --------------------------------------------------------------------------- #
class _Stub:
    """rng_key を返す決定論スタブ。"""
    name = "stub"

    def __init__(self, response: str | None = None):
        self.response = response
        self.calls: list[str] = []

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        self.calls.append(rng_key)
        return self.response if self.response is not None else f"resp:{rng_key}"


def test_check_prompt_detects_canary():
    TL.reset_canaries()
    try:
        assert TL.armed() is False
        TL.check_prompt("台帳の値は一切入っていない普通のプロンプト")   # 未武装=素通り
        tok = TL.register_canary("f00042")
        assert TL.armed() is True
        TL.check_prompt("台帳の値は一切入っていない普通のプロンプト")   # 混入なし=素通り
        with pytest.raises(TL.LedgerLeak) as ei:
            TL.check_prompt(f"場所: どこか\n{tok}\n気分: ふつう")
        assert "f00042" in str(ei.value)
    finally:
        TL.reset_canaries()


def test_cached_llm_refuses_a_leaked_prompt(tmp_path):
    """CachedLLM(全 LLM 呼び出しが通る唯一の関門)が漏洩プロンプトを止める。"""
    TL.reset_canaries()
    try:
        tok = TL.register_canary("f00007")
        llm = CachedLLM(_Stub(), enabled=True, path=tmp_path / "c.jsonl")
        llm.generate("普通のプロンプト", rng_key="ok/1/0", temperature=0.7,
                     max_tokens=32)                       # 素通り
        with pytest.raises(TL.LedgerLeak):
            llm.generate(f"漏れた: {tok}", rng_key="ng/1/0", temperature=0.7,
                         max_tokens=32)
        with pytest.raises(TL.LedgerLeak):                # 一括発行経路も塞がっている
            llm.generate_many([{"prompt": f"x{tok}", "rng_key": "ng/2/0",
                                "temperature": 0.7, "max_tokens": 32}])
    finally:
        TL.reset_canaries()


# --------------------------------------------------------------------------- #
# (漏洩 ③) canary テスト: 実ランの**全プロンプト**に台帳が現れない
# --------------------------------------------------------------------------- #
def test_no_canary_in_any_prompt_of_a_real_run(tmp_path):
    """beliefs ON の mock ランで、llm_journal に残った全プロンプトを走査する(第71 の活用)。

    さらに「台帳に一意な canary を持つ fact を注入する」条件も満たすため、
    ラン後の台帳の canary を 1 つずつ照合する(接頭辞検査だけに頼らない)。"""
    sim, out, summary = _run(tmp_path, "canary", n_steps=144, n_agents=40,
                             **{"beliefs.enabled": "true",
                                "beliefs.verify_actions": "true"})
    facts = TL.facts_of(sim)
    assert facts, "fact が 1 件も積まれていない(テストの前提が崩れた)"
    recs = list(iter_records(out / "llm_journal.jsonl.gz"))
    assert len(recs) == summary["llm_calls"] > 0, "ジャーナルが全呼び出しを覆っていない"
    canaries = [f["canary"] for f in facts.values()]
    assert len(set(canaries)) == len(canaries), "canary が一意でない"
    for rec in recs:
        prompt = rec["prompt"]
        assert TL.CANARY_PREFIX not in prompt, "プロンプトに台帳 canary 接頭辞が混入"
        for tok in canaries:
            assert tok not in prompt, f"プロンプトに台帳 canary {tok} が混入"
    # 真値そのものも L1 には出ていない(サイドカーへ分離してある)
    ledger = json.loads((out / TL.LEDGER_NAME).read_text(encoding="utf-8"))
    assert ledger["n_facts"] == len(facts)
    assert all("value" in f and "canary" in f for f in ledger["facts"])


# --------------------------------------------------------------------------- #
# (台帳) fact 抽出 = データ駆動・場所と時刻と目撃条件を持つ
# --------------------------------------------------------------------------- #
def test_facts_have_place_step_value_and_witness_condition(tmp_path):
    sim, _out, _s = _run(tmp_path, "facts", n_steps=144, n_agents=40,
                         **{"beliefs.enabled": "true"})
    facts = TL.facts_of(sim)
    assert len(facts) >= 2, "fact が積まれていない"
    kinds = {f["kind"] for f in facts.values()}
    assert len(kinds) >= 2, f"fact 種が 1 種しか出ていない: {kinds}"
    for f in facts.values():
        assert f["id"] and isinstance(f["step"], int)
        assert isinstance(f["x"], float) and isinstance(f["y"], float)
        assert 0.0 <= f["value"] <= 1.0
        assert f["window"] >= 1 and isinstance(f["radius_m"], float)
        assert isinstance(f["topics"], list)


def _run_with_map(tmp_path, name, fact_kinds: dict, n_steps=144, n_agents=40):
    """fact_kinds を**丸ごと差し替え**て走らせる(dotlist は OmegaConf の仕様で
    マップが merge されるため、ここでは resolved config を直接置き換える)。"""
    cfg = _cfg(name, n_steps, n_agents, **{"beliefs.enabled": "true"})
    cfg.beliefs.fact_kinds = fact_kinds
    sim = Simulation(cfg, out_dir=tmp_path / name)
    sim.run()
    return sim


def test_fact_kinds_are_data_driven(tmp_path):
    """conf の fact_kinds マップだけで供給源が決まる(コードに固有名詞を書かない)。"""
    sim = _run_with_map(tmp_path, "dd", {"event_host": {"topic_key": "title"}})
    kinds = {f["kind"] for f in TL.facts_of(sim).values()}
    assert kinds == {"event_host"}, f"マップ外の種が混ざった: {kinds}"


def test_empty_fact_map_yields_no_facts(tmp_path):
    sim = _run_with_map(tmp_path, "empty", {}, n_steps=48, n_agents=30)
    assert TL.facts_of(sim) == {}
    assert not [e for e in sim.logger.events if e.kind.startswith("belief")]


# --------------------------------------------------------------------------- #
# (信念) mock ランで目撃と伝聞が実際に発火する / 伝播木が構成できる
# --------------------------------------------------------------------------- #
def test_witness_and_hearsay_fire_in_mock(tmp_path):
    """mock の発話語彙(場所名を含む)でも話題一致が成立し、伝聞が発火する。"""
    sim, _out, _s = _run(tmp_path, "fire", n_steps=144, n_agents=40,
                         **{"beliefs.enabled": "true"})
    ups = _events(sim, "belief_update")
    trs = _events(sim, "belief_transmit")
    assert ups, "belief_update が 1 件も出ていない"
    assert trs, "belief_transmit が 1 件も出ていない(mock で伝聞が発火しない設計になっている)"
    causes = {e.payload["cause"] for e in ups}
    assert "witness" in causes, "直接目撃が 1 件も無い"
    assert "transmit" in causes, "伝聞由来の信念が 1 件も無い"
    for e in trs:
        assert e.payload["to"], "受け手が空の belief_transmit"
        assert e.payload["hop"] >= 1
        assert e.payload["topic"], "話題一致の根拠(topic)が残っていない"


def test_propagation_tree_is_reconstructible(tmp_path):
    """親ノード(from)から伝播木が再構成でき、ホップが単調に深まる。"""
    sim, _out, _s = _run(tmp_path, "tree", n_steps=144, n_agents=40,
                         **{"beliefs.enabled": "true"})
    seen: dict[tuple[str, int], int] = {}          # (fact, agent) -> hop
    n_edges = 0
    for e in _events(sim, "belief_update"):
        fid = e.payload["fact"]
        hop = int(e.payload["hop"])
        parent = int(e.payload["from"])
        if e.payload["cause"] == "transmit":
            assert parent >= 0, "伝聞なのに親ノードが無い(伝播木が切れる)"
            phop = seen.get((fid, parent))
            assert phop is not None, "親が先にその fact の信念を持っていない"
            assert hop == phop + 1, f"ホップが親+1 でない: {hop} vs {phop}"
            n_edges += 1
        else:
            assert hop == 0 and parent == -1
        seen[(fid, int(e.agent_id))] = hop
    assert n_edges > 0, "伝播木の枝が 1 本も無い"


def test_hop_drift_and_confidence_decay(tmp_path):
    """唯一の変形モデル: ホップごとに値が無情報 0.5 へ寄り、確信度が減衰する。"""
    sim, _out, _s = _run(tmp_path, "drift", n_steps=144, n_agents=40,
                         **{"beliefs.enabled": "true"})
    cfg = TL.cfg_of(sim)
    facts = TL.facts_of(sim)
    checked = 0
    for agent in sim.agents:
        for fid, bel in (getattr(agent, "_fact_beliefs", None) or {}).items():
            if bel["hop"] == 0:
                assert bel["conf"] == pytest.approx(1.0) or bel["src"] == "net"
                assert bel["value"] == pytest.approx(facts[fid]["value"]) \
                    or bel["src"] == "net"
                continue
            expect = cfg["conf_decay"] ** bel["hop"]
            assert bel["conf"] <= 1.0 and bel["conf"] == pytest.approx(expect, abs=1e-6)
            true_v = facts[fid]["value"]
            assert abs(bel["value"] - 0.5) <= abs(true_v - 0.5) + 1e-9, \
                "ホップで無情報方向へ寄っていない"
            checked += 1
    assert checked > 0, "伝聞由来の信念が 1 件も無い(前提が崩れた)"


def test_topic_match_is_substring_and_deterministic():
    assert TL._matches("ハチ公前は今日も混んでる", ["ハチ公前"]) == "ハチ公前"
    assert TL._matches("今日はいい天気", ["ハチ公前"]) is None
    assert TL._matches("Aの会に行った", ["X", "Aの会"]) == "Aの会"
    assert TL._matches("", ["A"]) is None


# --------------------------------------------------------------------------- #
# (検証行動) go / ask / net の 3 種が受理・記録される
# --------------------------------------------------------------------------- #
def test_parse_action_accepts_verify_verbs():
    for verb in ("verify", "check", "confirm"):
        got = deliberate.parse_action(
            json.dumps({"action": verb, "about": "あの店", "how": "go"},
                       ensure_ascii=False))
        assert got == {"type": "verify", "about": "あの店", "how": "go"}
    assert {"verify", "check", "confirm"} <= deliberate.KNOWN_ACTIONS
    # 既知動詞は未定義行動レジスタ(第70)へ落ちない
    reason, _info = deliberate.classify_reject('{"action": "verify"}')
    assert reason != "unknown_action"


def test_header_line_only_when_verify_actions_on():
    off = deliberate._header("constrained", False, "街", False, False)
    on = deliberate._header("constrained", False, "街", False, True)
    assert off == deliberate._header("constrained", False, "街", False), \
        "既定 OFF でヘッダが変わった(ゴールデン破壊)"
    assert on != off and on.startswith(off.rstrip("\n").rsplit("\n", 0)[0][:10])
    assert '"action": "verify"' in on
    # 誘導語を入れない(中立提示の掟)
    for banned in ("べき", "おすすめ", "ぜひ", "重要", "しましょう", "大切"):
        assert banned not in deliberate._VERIFY_LINE, f"誘導語 {banned} が混入"


def _seed_fact(sim, *, fid="f00000", node=None, value=1.0, topics=("試験場",)):
    """テスト用に台帳へ fact を 1 件だけ差し込む(本体の抽出経路は別テストで検証済み)。"""
    node = node or sorted(sim.city.graph.nodes)[0]
    x, y = sim.city.node_xy(node)
    fact = {"id": fid, "kind": "event_host", "step": 0, "sim_min": 0,
            "node": node, "place": "試験場", "x": round(float(x), 2),
            "y": round(float(y), 2), "value": float(value), "actor": -1,
            "topics": list(topics), "radius_m": 60.0, "window": 6,
            "canary": TL.register_canary(fid)}
    TL.facts_of(sim)[fid] = fact
    return fact


def _hearsay(sim, agent, fact, *, conf=0.8, value=0.6, parent=-1, hop=1,
             verified=None):
    TL.beliefs_of(agent)[fact["id"]] = {
        "value": value, "conf": conf, "src": "agent", "from": parent,
        "step": 0, "hop": hop, "verified": verified, "kind": fact["kind"]}


def _verify_sim(tmp_path, name, **ov):
    sim = Simulation(_cfg(name, 1, 10, **{"beliefs.enabled": "true",
                                          "beliefs.verify_actions": "true", **ov}),
                     out_dir=tmp_path / name)
    return sim


def test_verify_net_is_accepted_and_recorded(tmp_path):
    sim = _verify_sim(tmp_path, "vnet")
    agent = sim.agents[0]
    fact = _seed_fact(sim, fid="fnet")
    _hearsay(sim, agent, fact)
    scheduler._apply(sim, agent,
                     {"type": "verify", "about": "試験場", "how": "net"}, 0, 0)
    ev = [e for e in sim.logger.events if e.kind == "belief_verify"]
    assert len(ev) == 1 and ev[0].payload["outcome"] == "ok"
    assert ev[0].payload["evidence"] == "net" and ev[0].payload["match"] == "topic"
    bel = TL.beliefs_of(agent)["fnet"]
    assert bel["verified"] == "net" and bel["value"] == pytest.approx(fact["value"])
    assert bel["conf"] == pytest.approx(TL.cfg_of(sim)["net_conf"])


def test_verify_ask_needs_a_witness(tmp_path):
    sim = _verify_sim(tmp_path, "vask")
    a, b = sim.agents[0], sim.agents[1]
    fact = _seed_fact(sim, fid="fask")
    _hearsay(sim, a, fact)
    scheduler._apply(sim, a, {"type": "verify", "about": "試験場", "how": "ask"}, 0, 0)
    ev = [e for e in sim.logger.events if e.kind == "belief_verify"]
    assert ev[-1].payload["outcome"] == "no_witness", "目撃者不在なのに確かめられた"
    # 目撃者を同席させると成立する
    b.x, b.y, b.loc, b.sleeping = a.x, a.y, "street", False
    TL.beliefs_of(b)[fact["id"]] = {"value": fact["value"], "conf": 1.0,
                                    "src": "direct", "from": -1, "step": 0,
                                    "hop": 0, "verified": "witness",
                                    "kind": fact["kind"]}
    scheduler._apply(sim, a, {"type": "verify", "about": "試験場", "how": "ask"}, 0, 0)
    ev = [e for e in sim.logger.events if e.kind == "belief_verify"]
    assert ev[-1].payload["outcome"] == "ok"
    assert ev[-1].payload["evidence"] == f"agent:{b.id}"
    assert TL.beliefs_of(a)["fask"]["verified"] == "ask"


def test_verify_go_on_site_and_pending(tmp_path):
    sim = _verify_sim(tmp_path, "vgo")
    a = sim.agents[0]
    fact = _seed_fact(sim, fid="fgo")
    _hearsay(sim, a, fact)
    a.x, a.y = fact["x"], fact["y"]                 # 現場に居る=即確認
    scheduler._apply(sim, a, {"type": "verify", "about": "試験場", "how": "go"}, 0, 0)
    ev = [e for e in sim.logger.events if e.kind == "belief_verify"]
    assert ev[-1].payload["outcome"] == "ok" and ev[-1].payload["evidence"] == "site"
    assert TL.beliefs_of(a)["fgo"]["verified"] == "site"
    assert TL.beliefs_of(a)["fgo"]["value"] == pytest.approx(fact["value"])
    # 遠方にいる別人は「保留」になり、到着すると確定する
    b = sim.agents[1]
    _hearsay(sim, b, fact)
    b.x, b.y = fact["x"] + 5000.0, fact["y"] + 5000.0
    b.sleeping = False
    b.activity = ""
    scheduler._apply(sim, b, {"type": "verify", "about": "試験場", "how": "go"}, 0, 0)
    ev = [e for e in sim.logger.events if e.kind == "belief_verify"]
    assert ev[-1].payload["outcome"].startswith("pending")
    assert getattr(b, "_tl_pending", None) and b._tl_pending["fact"] == "fgo"
    b.x, b.y = fact["x"], fact["y"]                 # 到着
    TL._pending_pass(sim, TL.cfg_of(sim), 1, 10)
    assert TL.beliefs_of(b)["fgo"]["verified"] == "site"
    assert getattr(b, "_tl_pending", None) is None


def test_verify_without_belief_is_recorded_as_no_target(tmp_path):
    sim = _verify_sim(tmp_path, "vno")
    scheduler._apply(sim, sim.agents[0],
                     {"type": "verify", "about": "なにか", "how": "net"}, 0, 0)
    ev = [e for e in sim.logger.events if e.kind == "belief_verify"]
    assert len(ev) == 1 and ev[0].payload["outcome"] == "no_target"


def test_verify_is_ignored_when_off(tmp_path):
    """beliefs OFF / verify_actions OFF では検証行動を静かに無視する(=wander 相当)。"""
    for ov in ({"beliefs.enabled": "false", "beliefs.verify_actions": "false"},
               {"beliefs.enabled": "true", "beliefs.verify_actions": "false"}):
        name = "vign" + str(abs(hash(tuple(sorted(ov.items())))) % 1000)
        sim = Simulation(_cfg(name, 1, 10, **ov), out_dir=tmp_path / name)
        scheduler._apply(sim, sim.agents[0],
                         {"type": "verify", "about": "x", "how": "net"}, 0, 0)
        assert not [e for e in sim.logger.events if e.kind == "belief_verify"]


def test_how_is_normalized_deterministically():
    for raw, want in (("go", "go"), ("GO", "go"), ("現場", "go"), (None, "go"),
                      ("ask", "ask"), ("人に聞く", "ask"), ("net", "net"),
                      ("ネットで調べる", "net"),
                      ("go(その場所へ行く)", "go")):
        assert TL.normalize_how(raw) == want, raw


# --------------------------------------------------------------------------- #
# (registry) 宣言済み(journal 等級)+ 未宣言検出が緑
# --------------------------------------------------------------------------- #
def test_registry_declares_beliefs_toggles():
    for fid in ("beliefs.enabled", "beliefs.verify_actions"):
        assert fid in R.BY_ID, f"{fid} がレジストリ未宣言"
        feat = R.BY_ID[fid]
        assert feat.repro_tier == "journal", \
            "伝聞の発火判定が LLM の自由文(発話)を消費するので journal が正しい"
        assert feat.affects_k is False
    assert R.BY_ID["beliefs.enabled"].fingerprint_risk == "none"
    assert R.BY_ID["beliefs.verify_actions"].fingerprint_risk == "possible"
    assert R.undeclared_toggles(load_config()) == []


def test_verify_mode_disables_beliefs():
    """verify モード(strict のみ許可)では journal 等級の beliefs が自動 OFF になる。"""
    cfg = load_config(["run.mode=verify", "beliefs.enabled=true",
                       "beliefs.verify_actions=true"])
    gated, rep = R.apply_mode(cfg)
    assert bool(gated.beliefs.enabled) is False
    assert bool(gated.beliefs.verify_actions) is False
    dropped = {d["id"] for d in rep["auto_disabled"]}
    assert {"beliefs.enabled", "beliefs.verify_actions"} <= dropped


# --------------------------------------------------------------------------- #
# (resume) checkpoint 再開で信念状態・台帳・L1 が一致
# --------------------------------------------------------------------------- #
def test_resume_matches_straight_run(tmp_path):
    ov = {"beliefs.enabled": "true", "observer.checkpoint_every": 24}
    straight, out_s, _ = _run(tmp_path, "rs_straight", n_steps=48, n_agents=30, **ov)
    part = tmp_path / "rs_part"
    Simulation(_cfg("rs_part", 24, 30, **ov), out_dir=part).run()
    resumed = Simulation(_cfg("rs_part", 48, 30, **ov), out_dir=part)
    resumed.run(resume_from=part)

    assert (pq.read_table(out_s / "l1_events.parquet").to_pylist()
            == pq.read_table(part / "l1_events.parquet").to_pylist()), \
        "resume != straight(L1)"
    # 台帳(fact_id の採番まで含めて)一致
    assert json.loads((out_s / TL.LEDGER_NAME).read_text(encoding="utf-8")) \
        == json.loads((part / TL.LEDGER_NAME).read_text(encoding="utf-8"))
    # 信念状態(agent×fact)が一致
    def _bel(sim):
        return {a.id: dict(getattr(a, "_fact_beliefs", None) or {})
                for a in sim.agents}
    assert _bel(straight) == _bel(resumed), "resume で信念状態が食い違った"
    assert any(_bel(straight).values()), "信念が 1 件も無い(前提が崩れた)"


# --------------------------------------------------------------------------- #
# (L2) 列の出方
# --------------------------------------------------------------------------- #
def test_l2_scalars_are_sane(tmp_path):
    sim, out, _s = _run(tmp_path, "l2", n_steps=144, n_agents=40,
                        **{"beliefs.enabled": "true"})
    sc = TL.scalars(sim)
    assert set(sc) == set(BELIEF_COLS)
    assert sc["belief_facts_total"] == len(TL.facts_of(sim)) > 0
    assert 0.0 <= sc["belief_distance_mean"] <= 1.0
    assert 0.0 <= sc["belief_verified_rate"] <= 1.0
    assert sc["belief_hop_mean"] >= 0.0 and sc["belief_holders_mean"] > 0.0
    rows = pq.read_table(out / "l2_metrics.parquet").to_pylist()
    assert rows[-1]["belief_facts_total"] == sc["belief_facts_total"]


# --------------------------------------------------------------------------- #
# (解析) scripts/analyze_beliefs.py が伝播木・距離・検証率を組める
# --------------------------------------------------------------------------- #
def _analyzer():
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_beliefs.py"
    spec = importlib.util.spec_from_file_location("analyze_beliefs", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_analyze_beliefs_builds_tree_and_metrics(tmp_path):
    _sim, out, _s = _run(tmp_path, "an", n_steps=288, n_agents=40,
                         **{"beliefs.enabled": "true"})
    mod = _analyzer()
    res = mod.analyze(str(out))
    assert res["ledger"]["n_facts"] > 0
    assert len(res["ledger"]["by_kind"]) >= 2
    assert res["beliefs"]["n_agent_fact_pairs"] > 0
    assert res["series"], "信念距離の時系列が空"
    assert res["tree_stats"]["n_edges"] > 0, "伝播木の枝が構成できていない"
    assert res["tree_stats"]["max_hop"] >= 1
    # 木の親子は必ず holders に含まれる(整合)
    for fid, edges in res["tree"].items():
        holders = set(res["holders"][fid])
        for e in edges:
            assert e["parent"] in holders and e["child"] in holders
    assert 0.0 <= res["beliefs"]["distance_mean_final"] <= 1.0
    md = tmp_path / "rep.md"
    mod.write_report(res, str(md))
    assert "伝播木" in md.read_text(encoding="utf-8")


class _FixedLLM:
    """CachedLLM を差し替える固定応答(呼数だけ数える)。第73: 3 種の検証行動を巡回で返す。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.hits = 0

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        out = self.responses[self.calls % len(self.responses)]
        self.calls += 1
        return out, str(self.calls), False


def test_verify_actions_end_to_end_with_fixed_llm(tmp_path):
    """ON + 固定応答 LLM: 3 種の検証行動が実ランの経路で受理・記録され、解析に載る。"""
    out = tmp_path / "e2e"
    sim = Simulation(_cfg("e2e", 144, 30, **{"beliefs.enabled": "true",
                                             "beliefs.verify_actions": "true"}),
                     out_dir=out)
    sim.llm = _FixedLLM([
        # 台帳の材料(fact 源)を作る行動 → 発話(伝聞の材料)→ 3 種の検証行動、の巡回
        json.dumps({"action": "host_event", "title": "試しの会",
                    "hours_later": 2}, ensure_ascii=False),
        json.dumps({"action": "speak", "text": "試しの会がひらかれるらしい"},
                   ensure_ascii=False),
        *[json.dumps({"action": "verify", "about": "試しの会", "how": h},
                     ensure_ascii=False) for h in ("go", "ask", "net")]])
    sim.run()
    ev = [e for e in sim.logger.events if e.kind == "belief_verify"]
    assert ev, "固定応答で検証行動が 1 件も受理されていない"
    hows = {e.payload["how"] for e in ev}
    assert hows == {"go", "ask", "net"}, f"3 種が揃っていない: {hows}"
    oks = [e for e in ev if e.payload["outcome"] == "ok"]
    assert oks, "検証が 1 件も成立していない"
    assert {e.payload.get("evidence") for e in oks} & {"net", "site"}
    # 検証済みフラグと根拠が信念側にも残る
    vers = {b["verified"] for a in sim.agents
            for b in (getattr(a, "_fact_beliefs", None) or {}).values()}
    assert vers & {"net", "site", "ask"}, f"検証済みフラグが立っていない: {vers}"
    # 解析が検証率を組める
    res = _analyzer().analyze(str(out))
    assert res["verification"]["n_actions"] == len(ev)
    assert set(res["verification"]["by_how"]) == {"go", "ask", "net"}
    assert res["verification"]["verifiers"], "検証を実行した個体が記録されていない"
