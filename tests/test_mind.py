"""心モデル固定 + 三層知能配置(第88バッチ・model.mind)のテスト。

正典: docs/plans/source/design-discussion-20260802.md §5 /
      docs/plans/dayplan-engaged-plan.md 第88。

守るもの(検収基準の順)
  (1) 既定 OFF = 純粋既定と L1 バイト一致・属性なし・L2 に列なし・manifest / summary に
      キーなし・agents.json 不変・LLM 呼数不変・**新 stream を 1 本も引かない**
  (2) 割当の決定論と比率(同 seed 同割当・重みどおり・conf の並び順に依らない・traits 非依存)
  (3) **同一個体の全「心」呼が同一モデル**(llm_journal 全走査)・機械的判断は既定バックエンド
  (4) キャッシュがモデル別に正しく分離される(cache キーにモデル名が入る既存構造の維持)
  (5) 三層(基底=ルールのみ / 思考=pool / 高解像度=大型モデル+夜内省+上限緩和)
  (6) 不変量: ON で呼数・呼び出しサイトが変わらない・k 不変・resume==straight・no-fingerprint
  (7) ログ(交絡の記録): manifest / summary / agents.json / L1 mind_assign
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from society import mind as MD
from society import registry as R
from society.cognition import engaged as EG
from society.config import load_config
from society.engine import checkpoint, scheduler
from society.engine.simulation import Simulation
from society.llm.journal import iter_records
from society.llm.mind_router import MindRouter
from society.observer.schema import EVENT_KINDS
from society.rng import RngHub

SRC = Path(__file__).resolve().parents[1] / "src" / "society"

# mock サブモデル 3 本(応答本文は同一・**名前だけ違う**= 経路の分離だけを検証できる)
POOL3 = ('[{backend: mock, name: "mock:a", weight: 1}, '
         '{backend: mock, name: "mock:b", weight: 2}, '
         '{backend: mock, name: "mock:c", weight: 1}]')
POOL2 = ('[{backend: mock, name: "mock:a", weight: 1}, '
         '{backend: mock, name: "mock:b", weight: 1}]')
ON = {"model.mind.enabled": "true", "model.mind.pool": POOL3}
ON2 = {"model.mind.enabled": "true", "model.mind.pool": POOL2}
HIGH = {**ON2, "model.mind.tiers.high.frac": 0.2,
        "model.mind.tiers.high.name": "mock:hi"}
L2_COLS = ("mind_models_present", "mind_high_agents")
MODEL_NAMES = ("mock:a", "mock:b", "mock:c", "mock:hi")


# --------------------------------------------------------------------------- #
# 共通ヘルパ
# --------------------------------------------------------------------------- #
def _cfg(name, n_steps=48, n_agents=20, **ov):
    dot = ["run.seed=42", f"run.n_agents={n_agents}", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock", "observer.snapshot_every=96"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def _sim(tmp_path, name, n_steps=48, n_agents=20, **ov):
    return Simulation(_cfg(name, n_steps, n_agents, **ov), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _journal_rows(out_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for p in sorted(Path(out_dir).glob("llm_journal*.jsonl.gz")):
        rows.extend(iter_records(p))
    return rows


class _CountingHub:
    """stream 派生を数えるプロキシ(test_engaged / test_fire と同型)。"""

    def __init__(self, inner):
        self._inner = inner
        self.counts: dict[str, int] = {}

    def stream(self, *key):
        name = str(key[0]) if key else ""
        self.counts[name] = self.counts.get(name, 0) + 1
        return self._inner.stream(*key)

    def __getattr__(self, item):
        return getattr(self._inner, item)


class _FixedLLM:
    """内容非結合の固定応答(k 不変の切り分け用。CachedLLM と同じ 3 つ組を返す)。"""

    def __init__(self, response: str, name: str = "fixed"):
        self.response = response
        self.name = name
        self.calls = 0
        self.hits = 0

    def generate(self, prompt, **kw):
        self.calls += 1
        return self.response, str(self.calls), False


def _fix_children(sim, response: str) -> None:
    """MindRouter の全ての子(default 込み)を _FixedLLM へ差し替える(dispatch は残す)。"""
    sim.llm.default = _FixedLLM(response, "fixed:default")
    for mid in list(sim.llm.children):
        sim.llm.children[mid] = _FixedLLM(response, f"fixed:{mid}")


# --------------------------------------------------------------------------- #
# (A) 既定 OFF(検収基準 1)
# --------------------------------------------------------------------------- #
def test_shipped_default_is_off():
    cfg = load_config()
    assert bool(cfg.model.mind.enabled) is False
    assert MD.build_cfg(None)["enabled"] is False
    assert MD.build_cfg(None)["pool"] == []


def test_off_matches_pure_default(tmp_path):
    """明示 OFF と純粋既定が L1 完全一致。属性なし・L2 に列なし・agents.json 不変。"""
    pure = _sim(tmp_path, "pure")
    pure.run()
    off = _sim(tmp_path, "expl_off", **{"model.mind.enabled": "false"})
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(mind seam が no-op でない)"
    assert all(getattr(a, "mind", None) is None for a in off.agents), \
        "OFF なのに agent.mind 属性が生えた"
    cols = pq.read_table(tmp_path / "expl_off" / "l2_metrics.parquet").column_names
    for c in L2_COLS:
        assert c not in cols, f"OFF なのに L2 に {c} 列"
    agents_json = json.loads((tmp_path / "expl_off" / "agents.json")
                             .read_text(encoding="utf-8"))
    assert not any("mind_model" in r or "mind_tier" in r for r in agents_json)


def test_off_has_no_manifest_or_summary_key(tmp_path):
    off = _sim(tmp_path, "prov_off", n_steps=2)
    off.run()
    man = json.loads((tmp_path / "prov_off" / "run_manifest.json")
                     .read_text(encoding="utf-8"))
    summary = json.loads((tmp_path / "prov_off" / "summary.json")
                         .read_text(encoding="utf-8"))
    assert "mind" not in man, "OFF なのに manifest に mind キー"
    assert "mind" not in summary, "OFF なのに summary に mind キー"
    assert MD.provenance(off) is None and MD.scalars(off) is None


def test_off_emits_no_mind_assign_and_no_router(tmp_path):
    off = _sim(tmp_path, "off_ev")
    assert type(off.llm).__name__ == "CachedLLM", \
        "OFF なのに解決層が被さっている(既定の配線が変わった)"
    off.run()
    assert not [e for e in off.logger.events if e.kind == "mind_assign"]


def test_off_llm_call_count_unchanged(tmp_path):
    a = _sim(tmp_path, "cc_pure")
    a.run()
    b = _sim(tmp_path, "cc_off", **{"model.mind.enabled": "false"})
    b.run()
    assert a.llm.calls == b.llm.calls > 0


def test_off_draws_no_new_stream(tmp_path):
    """既定 OFF では mind 由来の stream("mind_model" / "mind_tier")を 1 本も引かない。"""
    sim = _sim(tmp_path, "off_stream")
    sim.hub = _CountingHub(sim.hub)
    sim.run()
    assert "mind_model" not in sim.hub.counts
    assert "mind_tier" not in sim.hub.counts


def test_event_kind_and_registry_declared():
    assert "mind_assign" in EVENT_KINDS
    feat = {f.id: f for f in R.FEATURES}["model.mind.enabled"]
    assert feat.repro_tier == "journal"
    assert feat.affects_k is True, \
        "affects_k の宣言(高解像度層が夜内省の選抜を引き継ぐ)が落ちた"
    assert feat.fingerprint_risk == "none", \
        "mind はプロンプトを 1 バイトも変えない(モデル名も層名も出ない)"


def test_no_undeclared_toggles():
    assert R.undeclared_toggles(load_config()) == []


# --------------------------------------------------------------------------- #
# (B) 割当の決定論と比率(検収基準 2)
# --------------------------------------------------------------------------- #
_CFG2 = MD.build_cfg({"enabled": True, "pool": [
    {"backend": "mock", "name": "a", "weight": 1},
    {"backend": "mock", "name": "b", "weight": 3}]})


def test_assignment_is_a_pure_function_of_seed_and_id():
    hub = RngHub(42)
    first = [MD.derive(_CFG2, hub, i) for i in range(200)]
    again = [MD.derive(_CFG2, RngHub(42), i) for i in range(200)]
    assert first == again, "同 seed・同 id で割当が揺れた(誕生時固定が純関数でない)"
    other = [MD.derive(_CFG2, RngHub(7), i) for i in range(200)]
    assert first != other, "seed を変えても割当が同じ(stream を引いていない疑い)"


def test_assignment_ratio_matches_weights():
    """重み 1:3 の割当比が統計誤差の内側(n=20,000 で ±0.01)。"""
    hub = RngHub(42)
    got = Counter(MD.derive(_CFG2, hub, i)[0] for i in range(20_000))
    share_a = got["a"] / 20_000
    assert abs(share_a - 0.25) < 0.01, f"重みどおりに割り当たっていない: {got}"


def test_choose_model_is_independent_of_conf_order():
    """conf の並び順を変えても同じ一様値なら同じモデル(name 昇順の累積分割)。"""
    pool = _CFG2["pool"]
    rev = list(reversed(pool))
    for u in (0.0, 0.1, 0.2499, 0.25, 0.6, 0.999):
        assert MD.choose_model(pool, u) == MD.choose_model(rev, u)


def test_high_tier_frac_is_uniform_and_traits_independent():
    cfg = MD.build_cfg({"enabled": True, "pool": [{"backend": "mock", "name": "a"}],
                        "tiers": {"high": {"frac": 0.05, "name": "hi"}}})
    hub = RngHub(42)
    tiers = Counter(MD.derive(cfg, hub, i)[1] for i in range(20_000))
    assert abs(tiers[MD.TIER_HIGH] / 20_000 - 0.05) < 0.005
    # traits を渡しても既定(uniform)では結果が 1 つも動かない = k* と直交
    weird = {"nfc": 0.99, "risk_tolerance": 0.01}
    assert [MD.derive(cfg, RngHub(42), i) for i in range(500)] == \
           [MD.derive(cfg, RngHub(42), i, weird) for i in range(500)]


def test_assignment_ignores_traits_in_a_real_run(tmp_path):
    """flat_traits(全個体同一 traits)ランでも割当が変わらない(traits 非参照の実証)。"""
    a = _sim(tmp_path, "tr_normal", n_steps=1, **ON)
    b = _sim(tmp_path, "tr_flat", n_steps=1, **ON,
             **{"experiment.flat_traits.enabled": "true"})
    assert [x.mind["model"] for x in a.agents] == [x.mind["model"] for x in b.agents]


def test_config_validation():
    with pytest.raises(ValueError):                     # pool なしで ON
        MD.build_cfg({"enabled": True})
    with pytest.raises(ValueError):                     # name 重複
        MD.build_cfg({"enabled": True, "pool": [
            {"backend": "mock", "name": "a"}, {"backend": "mock", "name": "a"}]})
    with pytest.raises(ValueError):                     # name なし
        MD.build_cfg({"enabled": True, "pool": [{"backend": "mock"}]})
    with pytest.raises(ValueError):                     # 未知の選抜方式
        MD.build_cfg({"enabled": True, "pool": [{"backend": "mock", "name": "a"}],
                      "tiers": {"high": {"select": "psychic"}}})


# --------------------------------------------------------------------------- #
# (C) 心の呼び出しが同一モデルへ(検収基準 3)
# --------------------------------------------------------------------------- #
def test_all_mind_calls_of_one_agent_go_to_one_model(tmp_path):
    """**llm_journal 全走査**: 各個体の「心」の呼びが 1 つのモデルに閉じている。"""
    sim = _sim(tmp_path, "one_model", n_steps=288, n_agents=25, **ON)
    sim.run()
    by_agent: dict[int, set[str]] = defaultdict(set)
    purposes: set[str] = set()
    for rec in _journal_rows(tmp_path / "one_model"):
        parts = str(rec["rng_key"]).split("/")
        purposes.add(parts[0])
        if len(parts) >= 2 and parts[0] in MD.MIND_PURPOSES:
            by_agent[int(parts[1])].add(str(rec["backend"]))
    assert by_agent, "テスト前提が崩れた(心の呼びが 1 本も無い)"
    assert len(purposes & set(MD.MIND_PURPOSES)) >= 2, \
        f"心の呼び種が 1 つしか出ていない(検証が薄い): {purposes}"
    bad = {a: v for a, v in by_agent.items() if len(v) > 1}
    assert not bad, f"同一個体の心の呼びが複数モデルへ散った: {bad}"
    # ジャーナルの backend と誕生時固定の属性が一致する
    for agent in sim.agents:
        seen = by_agent.get(int(agent.id))
        if seen:
            assert seen == {agent.mind["model"]}, \
                f"agent {agent.id}: journal={seen} attr={agent.mind['model']}"


def test_non_mind_purposes_go_to_the_shared_default(tmp_path):
    """機械的判断・対照系列(agent_id を持つが心でない呼び)は既定バックエンドが受ける。"""
    sim = _sim(tmp_path, "null_default", n_steps=48, n_agents=12, **ON,
               **{"controls.mode": "null_series", "controls.null_calls": 1})
    before = sim.llm.default.calls
    sim.run()
    assert sim.llm.default.calls > before, "null 系列が既定バックエンドへ行っていない"
    for rec in _journal_rows(tmp_path / "null_default"):
        if str(rec["rng_key"]).startswith("null/"):
            assert rec["backend"] == "mock", \
                "対照系列(内容非結合)が心のモデルへ流れた"


def test_router_dispatch_rules():
    """解決層の dispatch 規則(単体)。purpose と agent_id の 2 条件だけを見る。"""
    router = MindRouter(_FixedLLM("d", "default"),
                        {"m1": _FixedLLM("1", "m1"), "m2": _FixedLLM("2", "m2")},
                        resolve=lambda aid: "m1" if int(aid) % 2 == 0 else "m2",
                        purposes=MD.MIND_PURPOSES)
    assert router.model_for("deliberate/2/10") == "m1"
    assert router.model_for("reflect/3/10") == "m2"
    assert router.model_for("null/2/10/0") is None      # 心でない purpose → default
    assert router.model_for("deliberate") is None       # agent_id なし → default
    assert router.model_for("gov//10") is None
    assert router.name == "mind", "解決層の name はモデル非依存でなければならない"


def test_router_generate_many_preserves_request_order():
    a, b, d = _FixedLLM("A", "a"), _FixedLLM("B", "b"), _FixedLLM("D", "d")
    for child in (a, b, d):                              # CachedLLM 互換の一括発行
        child.generate_many = (
            lambda reqs, workers=1, _c=child: [_c.generate(r["prompt"]) for r in reqs])
    router = MindRouter(d, {"m1": a, "m2": b},
                        resolve=lambda aid: "m1" if int(aid) % 2 == 0 else "m2",
                        purposes=MD.MIND_PURPOSES)
    reqs = [{"prompt": "p", "rng_key": f"deliberate/{i}/0"} for i in range(6)]
    reqs.append({"prompt": "p", "rng_key": "null/9/0/0"})
    out = router.generate_many(reqs, workers=1)
    assert [o[0] for o in out] == ["A", "B", "A", "B", "A", "B", "D"]
    assert router.calls == 7


# --------------------------------------------------------------------------- #
# (D) キャッシュのモデル別分離(検収基準 4)
# --------------------------------------------------------------------------- #
def test_cache_is_separated_by_model(tmp_path):
    """同一プロンプトでも**モデルが違えば別キー**(cache キーにモデル名が入る既存構造)。"""
    sim = _sim(tmp_path, "cache_sep", n_steps=1, n_agents=6, **ON)
    a, b = sim.llm.children["mock:a"], sim.llm.children["mock:b"]
    kw = {"temperature": 0.0, "max_tokens": 16}
    r1 = a.generate("同じ質問", rng_key="deliberate/1/0", **kw)
    r2 = b.generate("同じ質問", rng_key="deliberate/2/0", **kw)
    assert r1[2] is False and r2[2] is False, "別モデルなのにキャッシュを共有した"
    assert r1[1] != r2[1], "cache キーがモデル間で衝突している(D13 違反)"
    r3 = a.generate("同じ質問", rng_key="deliberate/1/1", **kw)
    assert r3[2] is True, "同一モデル・同一プロンプトでキャッシュに当たらない"


def test_cache_and_journal_files_are_per_model(tmp_path):
    sim = _sim(tmp_path, "cache_files", n_steps=48, n_agents=15, **ON)
    sim.run()
    out = tmp_path / "cache_files"
    caches = {p.name for p in out.glob("llm_cache*.jsonl")}
    journals = {p.name for p in out.glob("llm_journal*.jsonl.gz")}
    for mid in ("mock_a", "mock_b", "mock_c"):
        assert f"llm_cache.{mid}.jsonl" in caches, f"{mid} のキャッシュファイルが無い"
        assert f"llm_journal.{mid}.jsonl.gz" in journals


# --------------------------------------------------------------------------- #
# (E) 三層知能(検収基準 5)
# --------------------------------------------------------------------------- #
def test_high_tier_uses_its_own_model(tmp_path):
    sim = _sim(tmp_path, "hi_model", n_steps=1, n_agents=60, **HIGH)
    high = [a for a in sim.agents if a.mind["tier"] == MD.TIER_HIGH]
    think = [a for a in sim.agents if a.mind["tier"] == MD.TIER_THINK]
    assert high and think, "層が片方しか出ていない(検証にならない)"
    assert {a.mind["model"] for a in high} == {"mock:hi"}
    assert "mock:hi" not in {a.mind["model"] for a in think}
    assert "mock:hi" in sim.llm.children, "高解像度層の大型モデルが子に居ない"


def test_high_tier_selection_is_deterministic(tmp_path):
    a = _sim(tmp_path, "hi_det1", n_steps=1, n_agents=60, **HIGH)
    b = _sim(tmp_path, "hi_det2", n_steps=1, n_agents=60, **HIGH)
    assert [x.mind["tier"] for x in a.agents] == [x.mind["tier"] for x in b.agents]


def test_high_tier_takes_over_reflect_selection(tmp_path):
    """夜内省(§8 突入 5)の対象が高解像度層に一致する(第87 reflect_frac から権威が移る)。"""
    sim = _sim(tmp_path, "hi_reflect", n_steps=1, n_agents=60, **HIGH,
               **{"cognition.engaged.reflect_frac": 1.0})
    picked = {a.id for a in sim.agents if EG.high_res(a, sim.engagedcfg)}
    high = {a.id for a in sim.agents if a.mind["tier"] == MD.TIER_HIGH}
    assert picked == high, "reflect_frac=1.0 でも高解像度層だけが選ばれるべき"


def test_high_reflect_false_keeps_the_legacy_selection(tmp_path):
    """接続を切る(high.reflect=false)と第87 の reflect_frac がそのまま権威に戻る。"""
    sim = _sim(tmp_path, "hi_noreflect", n_steps=1, n_agents=60, **HIGH,
               **{"model.mind.tiers.high.reflect": "false",
                  "cognition.engaged.reflect_frac": 1.0})
    assert all(EG.high_res(a, sim.engagedcfg) for a in sim.agents)
    assert all(MD.reflect_override(a) is None for a in sim.agents)


def test_cap_mult_relaxes_the_thinking_cap(tmp_path):
    """思考頻度の上限緩和(§5)。既定 1.0 では 1 も変わらない。"""
    sim = _sim(tmp_path, "cap", n_steps=1, n_agents=60, **HIGH,
               **{"model.mind.tiers.high.cap_mult": 2.0})
    cfg = sim.engagedcfg
    high = next(a for a in sim.agents if a.mind["tier"] == MD.TIER_HIGH)
    think = next(a for a in sim.agents if a.mind["tier"] == MD.TIER_THINK)
    assert EG.turn_cap_of(EG.TALK, cfg, high) == 2 * EG.turn_cap_of(EG.TALK, cfg, think)
    assert EG.turn_cap_of(EG.REPLAN, cfg, high) == \
        2 * EG.turn_cap_of(EG.REPLAN, cfg, think)
    # agent を渡さない従来呼び出し・mind OFF の個体では素の値
    assert EG.turn_cap_of(EG.TALK, cfg) == int(cfg["turn_cap"])
    assert MD.cap_mult(object()) == 1.0


def test_base_tier_is_rule_only(tmp_path):
    """基底層(経路選択・定型購買・習慣)は **LLM を 1 本も呼ばない**(§5 の確認事項)。"""
    for name in ("mobility.py", "commerce.py", "goods.py", "cognition/routine.py"):
        text = (SRC / name).read_text(encoding="utf-8")
        assert "llm.generate" not in text, \
            f"{name} が LLM を呼んでいる(機械的判断はルール層のはず)"


def test_traits_selection_is_declared_as_ablation_only(tmp_path):
    """traits 依存選抜は分離宣言(アブレーション専用)で、manifest に警告が残る。"""
    sim = _sim(tmp_path, "sel_traits", n_steps=1, n_agents=60, **HIGH,
               **{"model.mind.tiers.high.select": "traits",
                  "model.mind.tiers.high.select_trait": "nfc"})
    prov = MD.provenance(sim)
    assert prov["tiers"]["high"]["select"] == "traits"
    assert "warning" in prov, "意図的な交絡であることの宣言が manifest に無い"
    high = [a for a in sim.agents if a.mind["tier"] == MD.TIER_HIGH]
    assert high, "traits 選抜で 1 人も選ばれていない"
    assert min(a.traits["nfc"] for a in high) >= 0.8 - 1e-9, \
        "traits 選抜が上位 frac になっていない"
    # 既定(uniform)では 1 つも trait を読まない
    base = _sim(tmp_path, "sel_uniform", n_steps=1, n_agents=60, **HIGH)
    assert MD.provenance(base)["tiers"]["high"]["select"] == "uniform"
    assert "warning" not in MD.provenance(base)


# --------------------------------------------------------------------------- #
# (F) 不変量(検収基準 6)
# --------------------------------------------------------------------------- #
def test_on_does_not_change_llm_call_count(tmp_path):
    """ON でも generate() の**本数**は 1 本も変わらない(解決層は行き先だけを変える)。"""
    off = _sim(tmp_path, "n_off", n_steps=288, n_agents=20)
    off.run()
    on = _sim(tmp_path, "n_on", n_steps=288, n_agents=20, **ON)
    on.run()
    assert off.llm.calls == on.llm.calls > 0, \
        f"mind ON で呼数が動いた: {off.llm.calls} vs {on.llm.calls}"


def test_on_adds_no_new_llm_call_site(tmp_path):
    base = _sim(tmp_path, "site_base", n_steps=288, n_agents=20)
    base.run()
    on = _sim(tmp_path, "site_on", n_steps=288, n_agents=20, **ON)
    on.run()
    assert {c["purpose"] for c in on.logger.llm_calls} == \
           {c["purpose"] for c in base.logger.llm_calls}


def test_llm_call_count_k_invariant(tmp_path):
    """compute_matched の下で k=free と k=off の呼数・割当が完全一致する(R1)。"""
    def _run(name, writeback):
        sim = _sim(tmp_path, name, n_steps=144, n_agents=25, **ON,
                   **{"controls.mode": "compute_matched", "k.writeback": writeback})
        _fix_children(sim, json.dumps({"action": "speak", "text": "x"},
                                      ensure_ascii=False))
        sim.run()
        return sim
    free = _run("k_free", "free")
    off = _run("k_off", "off")
    assert free.llm.calls == off.llm.calls > 0, \
        f"mind の呼数が k に依存(R1 違反): {free.llm.calls} vs {off.llm.calls}"
    assert [a.mind["model"] for a in free.agents] == \
           [a.mind["model"] for a in off.agents], "割当が k に依存している"
    per_free = {m: c.calls for m, c in free.llm.children.items()}
    per_off = {m: c.calls for m, c in off.llm.children.items()}
    assert per_free == per_off, f"モデル別の呼数が k に依存: {per_free} vs {per_off}"


def test_no_fingerprint_model_names_absent_from_prompts(tmp_path):
    """モデル名・層名がプロンプト全文に 1 文字も出ない(fingerprint_risk=none の実証)。"""
    sim = _sim(tmp_path, "nofp", n_steps=288, n_agents=20, **HIGH)
    sim.run()
    rows = _journal_rows(tmp_path / "nofp")
    assert rows, "ジャーナルが空(検証にならない)"
    blob = "\n".join(str(r["prompt"]) for r in rows)
    for word in MODEL_NAMES + ("mind_model", "mind_tier", "high_res", "cap_mult"):
        assert word not in blob, f"プロンプトに機構語 '{word}' が漏れた"


def test_same_seed_two_runs_match(tmp_path):
    a = _sim(tmp_path, "seed_a", n_steps=144, n_agents=20, **ON)
    a.run()
    b = _sim(tmp_path, "seed_b", n_steps=144, n_agents=20, **ON)
    b.run()
    assert _l1(a) == _l1(b)
    assert [x.mind for x in a.agents] == [x.mind for x in b.agents]


def test_resume_matches_straight(tmp_path):
    """mid-day resume が straight と L1/L2/L3 一致(割当の保持 + mind_assign を二重に出さない)。"""
    ov = {**ON2, "model.mind.tiers.high.frac": 0.15,
          "model.mind.tiers.high.name": "mock:hi",
          "cognition.fire.enabled": "true", "cognition.engaged.enabled": "true",
          "planning.day_plan.enabled": "true"}
    split, total, n = 48, 96, 25
    straight_dir = tmp_path / "md_straight"
    straight = Simulation(_cfg("md_straight", total, n, **ov), out_dir=straight_dir)
    straight.run()
    assert [e for e in straight.logger.events if e.kind == "mind_assign"]

    d = tmp_path / "md_resumed"
    ckpt = {**ov, "observer.checkpoint_every": 48}
    sim1 = Simulation(_cfg("md_resumed", split, n, **ckpt), out_dir=d)
    for step in range(split):
        scheduler.run_step(sim1, step)
    checkpoint.save(sim1, split, d / "checkpoint" / f"ckpt-{split:06d}.pkl.gz")
    sim1.logger.flush_segment()
    sim2 = Simulation(_cfg("md_resumed", total, n, **ckpt), out_dir=d)
    sim2.run(resume_from=d)
    for stem in ("l1_events", "l2_metrics", "l3_snapshots"):
        sa = pq.read_table(straight_dir / f"{stem}.parquet").to_pylist()
        sb = pq.read_table(d / f"{stem}.parquet").to_pylist()
        assert sa == sb, f"{stem} 不一致(mind resume)"

    # 直接検証: 割当と「記録済み id」が round-trip で復元される(空回り防止)
    sim3 = Simulation(_cfg("md_inspect", split, n, **ckpt), out_dir=tmp_path / "md_ins")
    checkpoint.load(sim3, d / "checkpoint" / f"ckpt-{split:06d}.pkl.gz")
    assert sim3._mind_logged == sim1._mind_logged and sim3._mind_logged
    assert [a.mind for a in sim3.agents] == [a.mind for a in sim1.agents]
    assert sim3._mind_binding == {int(a.id): a.mind["model"] for a in sim1.agents}


# --------------------------------------------------------------------------- #
# (G) ログ = 交絡の記録(検収基準 7)
# --------------------------------------------------------------------------- #
def test_binding_is_logged_in_all_three_places(tmp_path):
    """§5「agent_id と model_id の対応は必ずログに残す」= L1 / agents.json / manifest。"""
    sim = _sim(tmp_path, "log3", n_steps=48, n_agents=20, **HIGH)
    sim.run()
    out = tmp_path / "log3"
    events = [e for e in sim.logger.events if e.kind == "mind_assign"]
    assert len(events) == len(sim.agents), "全個体ぶんの mind_assign が無い"
    assert all(e.step == 0 for e in events), "誕生時(step0)に出ていない"
    from_l1 = {e.agent_id: (e.payload["model"], e.payload["tier"]) for e in events}
    assert from_l1 == {a.id: (a.mind["model"], a.mind["tier"]) for a in sim.agents}

    agents_json = json.loads((out / "agents.json").read_text(encoding="utf-8"))
    from_json = {r["id"]: (r["mind_model"], r["mind_tier"]) for r in agents_json}
    assert from_json == from_l1, "agents.json の対応が L1 と食い違う"

    man = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert man["mind"]["n_bound"] == len(sim.agents)
    assert set(man["mind"]["by_model"]) >= {m for m, _t in from_l1.values()}
    assert "confound_note" in man["mind"], "交絡の注記が manifest に無い"


def test_summary_by_model_merges_calls_episodes_and_repairs(tmp_path):
    """summary.mind.by_model が呼数・エピソード数・修復率を統合する(第86/87 との整合)。"""
    ov = {**ON, "cognition.fire.enabled": "true",
          "cognition.engaged.enabled": "true", "planning.day_plan.enabled": "true"}
    sim = _sim(tmp_path, "sum_merge", n_steps=288, n_agents=30, **ov)
    sim.run()
    summary = json.loads((tmp_path / "sum_merge" / "summary.json")
                         .read_text(encoding="utf-8"))
    by_model = summary["mind"]["by_model"]
    assert set(by_model) <= set(MODEL_NAMES)
    assert sum(r.get("calls", 0) for r in by_model.values()) > 0
    assert sum(r.get("agents", 0) for r in by_model.values()) == len(sim.agents)
    assert sum(r.get("episodes", 0) for r in by_model.values()) == \
        summary["engaged"]["episodes"], "エピソード数がモデル別集計と合わない"
    assert sum(r.get("plans", 0) for r in by_model.values()) == \
        summary["day_plan"]["plans"], "計画数がモデル別集計と合わない"
    assert all("repair_rate" in r for r in by_model.values() if r.get("plans"))


def test_episode_and_plan_model_id_use_the_fixed_binding(tmp_path):
    """L1 のエピソード/計画イベントの model 欄が **固定割当**になっている(暫定の backend 名でない)。"""
    ov = {**ON, "cognition.fire.enabled": "true",
          "cognition.engaged.enabled": "true", "planning.day_plan.enabled": "true"}
    sim = _sim(tmp_path, "mid_fixed", n_steps=288, n_agents=30, **ov)
    sim.run()
    bound = {a.id: a.mind["model"] for a in sim.agents}
    checked = 0
    for e in sim.logger.events:
        if e.kind in ("episode_start", "episode_end", "plan_created"):
            assert e.payload.get("model") == bound[e.agent_id], \
                f"{e.kind} の model 欄が固定割当と違う(agent {e.agent_id})"
            assert e.payload["model"] != "mock", "backend 名のままになっている"
            checked += 1
    assert checked > 0, "テスト前提が崩れた(エピソード/計画イベントが 1 件も無い)"


def test_l2_columns_present_when_on(tmp_path):
    sim = _sim(tmp_path, "l2_on", n_steps=48, n_agents=20, **HIGH)
    sim.run()
    table = pq.read_table(tmp_path / "l2_on" / "l2_metrics.parquet")
    cols = table.column_names
    for c in L2_COLS:
        assert c in cols, f"ON なのに L2 に {c} 列が無い"
    n_high = sum(1 for a in sim.agents if a.mind["tier"] == MD.TIER_HIGH)
    assert table.column("mind_high_agents").to_pylist()[-1] == float(n_high)
    assert table.column("mind_models_present").to_pylist()[-1] == \
        float(len({a.mind["model"] for a in sim.agents}))


def test_model_of_id_reconstructs_binding_without_agents(tmp_path):
    """解決層は agent が居なくても純関数で同じモデルに戻る(resume / pool 再入場の根拠)。"""
    sim = _sim(tmp_path, "rebuild", n_steps=1, n_agents=20, **ON)
    want = {a.id: a.mind["model"] for a in sim.agents}
    sim._mind_binding = {}
    sim.agent_by_id = {}
    got = {aid: MD.model_of_id(sim, aid) for aid in want}
    assert got == want
    assert MD.model_of_id(sim, "not-an-int") is None
