"""H3(遺失物ループ = 完全内生の事件 ``lost_property``)のテスト。

正典
  - docs/plans/body-incident-layer-plan.md **§3 の「遺失物」行**(様式 = **完全内生**(本命)。
    落とす(移動/飲酒/混雑で増)→ 拾う → **届ける/無視/着服**(全て行為)→ 交番/駅事務室 →
    報労金 5-20% / 3 か月時効。検証 = **品目別返還率**(傘 1% / 財布 60-80% / 携帯 87%)・
    都実測 = 拾得 454 万件/年)
  - 同 **§4 chance_event の溶解**(★拾得金は必ず誰かの drop から = 貨幣保存則と整合)
  - 同 **§3 の共在判定の一本化**(``sim.percept_index`` に集約 = O(n²) 回避)/
    **L1 は 1 行 + 前兆状態同梱**

守るもの(検収基準の順)
  ① OFF(既定)= ゴールデン L1 バイト一致・キー不発生・state も属性も生えない・所持金不変
  ② ON: 落とす → 気づく → 拾う → 届ける/着服 → 返還/時効 の全段が実際に立つ
  ③ ★**貨幣保存**: Σ(所持金) + 遺失物の中の現金 + 失効累計 = 一定(= §4 の核心)
  ④ ★**品目別返還率**が実測帯に入る(傘 0.5-3% / 財布 60-80% / 携帯 80-92%)
  ⑤ 報労金は 5-20%(遺失物法 28 条)かつ **持ち主 → 拾得者の直接授受**(総額不変)
  ⑥ 着服は ``crime`` を出さず payload の offense / guardians(RAT)で宣言する
  ⑦ ON 同 seed 2 ラン一致 / resume == straight(進行中の遺失物が checkpoint に載る)
  ⑧ LLM 呼数 ON/OFF 完全一致(generate() の呼び出しサイトを 1 つも足さない)
  ⑨ ★静的検査: 新 stream は "lost_drop" 1 本だけ・全対全スキャンを足していない
  ⑩ 凍結 14 ファイル / chance.py を 1 バイトも触っていない
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from society import chance as chance_mod
from society import economy_sfc as SFC
from society import lost_property as L
from society import registry as R
from society.config import load_config
from society.engine import checkpoint, scheduler
from society.engine.simulation import Simulation
from society.observer import causality as C
from society.observer.schema import EVENT_KINDS
from society.world.perception import build_index

GOLDEN = Path(__file__).resolve().parent / "data" / "golden_baseline_l1.json"

# test_traces.py:45 / test_rumors.py:45 と同じ「意図的な既定挙動追加」の中立化
_GOLDEN_NEUTRAL = {"economy.wages.自営": 0, "planning.enabled": "false",
                   "transit_ride.taxi.enabled": "false",
                   "transit_ride.bus.enabled": "false", "rules.enabled": "false"}

OFF = {"lost_property.enabled": "false"}
ON = {"lost_property.enabled": "true"}

#: 落下率を上げて短いランでも全段が立つようにする(**較正値そのものは触らない** =
#: 既定は module の BASE_DAILY のまま。ここで上げるのは「機構が動くか」を見るため)。
LOUD = dict(ON, **{"lost_property.base_daily.umbrella": "1.2",
                   "lost_property.base_daily.wallet": "1.2",
                   "lost_property.base_daily.phone": "1.2",
                   "lost_property.base_daily.other": "1.2",
                   "lost_property.max_events_per_step": "64"})

LOST_KINDS = ("lost_drop", "lost_notice", "lost_pickup", "lost_turnin",
              "lost_return", "lost_keep", "lost_expire")


# --------------------------------------------------------------------------- #
# 共通ヘルパ(test_traces.py と同型)
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


class _FixedLLM:
    """**プロンプト非依存**の巡回応答スタブ(test_traces / test_rumors と同型)。"""

    name = "fixed"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.hits = 0

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        out = self.responses[self.calls % len(self.responses)]
        self.calls += 1
        return out, str(self.calls), False


def _huddle(sim, n=20):
    """先頭 n 体を同じ場所に集める(落下地点の共在を作る = 拾得が起きる条件)。"""
    base = sim.agents[0]
    for a in sim.agents[1:n]:
        a.x, a.y, a.node, a.loc = base.x, base.y, base.node, base.loc
        a.building, a.floor, a.sleeping = base.building, base.floor, False


def _drive(sim, steps, sim_min0=420):
    """``L.phase`` だけを回す隔離ドライバ(賃金・消費など他の金の経路を混ぜない)。

    ★貨幣保存の検査は本 module の外の経路(wage / spend / home_refill …)が同時に動くと
      成立しないので、**位置確定 → 索引 → 本 module** の最小ループだけを回す。
    """
    for step in range(steps):
        sim.percept_index = build_index(
            sim.agents, float(sim.cfg.world.perception_radius_m))
        L.phase(sim, step, sim_min0 + step * 10)


# =========================================================================== #
# (A) 出荷既定・宣言(検収基準 ①の前段)
# =========================================================================== #
def test_shipped_default_is_off():
    cfg = load_config()
    assert bool(cfg.lost_property.enabled) is False
    # 較正値が config と実装で食い違っていない(較正の単一の源)
    assert dict(cfg.lost_property.base_daily) == L.BASE_DAILY
    assert dict(cfg.lost_property.notice_prob) == L.NOTICE_PROB
    assert dict(cfg.lost_property.claim_prob) == L.CLAIM_PROB
    assert dict(cfg.lost_property.pickup_prob) == L.PICKUP_PROB
    assert float(cfg.lost_property.reward_lo) == 0.05    # 遺失物法 28 条
    assert float(cfg.lost_property.reward_hi) == 0.20
    assert int(cfg.lost_property.statute_steps) == 90 * 144   # 同 7 条 = 3 か月


def test_registry_and_schema_declared():
    feat = {f.id: f for f in R.FEATURES}["lost_property.enabled"]
    assert feat.repro_tier == "strict"
    assert feat.affects_k is False       # generate() の呼び出しサイトを 1 つも足さない
    assert feat.fingerprint_risk == "possible"
    assert feat.off_value is False
    for kind in LOST_KINDS:
        assert kind in EVENT_KINDS, f"{kind} が L1 スキーマに未登録"
        assert kind in C.CAUSE_OF_KIND, f"{kind} が因果台帳に未分類"
    # 因果の型: 落とすのは身体・拾う/届ける/着服は行為・時効は暦
    assert C.CAUSE_OF_KIND["lost_drop"] == C.PHYSICS
    for kind in ("lost_notice", "lost_pickup", "lost_turnin", "lost_keep",
                 "lost_return"):
        assert C.CAUSE_OF_KIND[kind] == C.AGENT, kind
    assert C.CAUSE_OF_KIND["lost_expire"] == C.SCHEDULE


def test_item_table_is_finite_and_every_item_has_every_parameter():
    """品目表が有限で、全品目に全パラメータが揃っている(綴り違いで永久に 0 件を防ぐ)。"""
    assert set(L.ITEMS) == set(L.ITEM_WORD)
    for item in L.ITEMS:
        assert L.ITEM_WORD[item].strip()
        for blk in (L.BASE_DAILY, L.NOTICE_PROB, L.CLAIM_PROB, L.PICKUP_PROB,
                    L.NOTICE_DELAY, L.ITEM_VALUE):
            assert item in blk, f"{item} が {blk} に無い"
    # 品目別の遅れが単一値に退化していない(携帯は早い・傘は遅い = 計画書の要求)
    assert L.NOTICE_DELAY["phone"] < L.NOTICE_DELAY["wallet"] < L.NOTICE_DELAY["umbrella"]
    assert len(set(L.NOTICE_DELAY.values())) >= 3


def test_unknown_config_values_degrade_to_defaults():
    cfg = L.build_cfg({"enabled": True,
                       "base_daily": {"phone": 0.5, "存在しない品目": 9.9},
                       "notice_delay": {"phone": -3},
                       "reward_lo": 0.30, "reward_hi": 0.10,
                       "turnin_prob": -1.0})
    assert cfg["base_daily"]["phone"] == 0.5
    assert "存在しない品目" not in cfg["base_daily"]      # 表に無い品目は黙って捨てる
    assert cfg["base_daily"]["wallet"] == L.BASE_DAILY["wallet"]   # 既定へ重ねる
    assert cfg["notice_delay"]["phone"] == 0             # 負値は 0 へ
    assert (cfg["reward_lo"], cfg["reward_hi"]) == (0.10, 0.30)    # 逆順は入れ替え
    assert cfg["turnin_prob"] == 0.0


def test_sentence_is_a_pure_function_without_numbers():
    """記憶の 1 行は (出来事, 品目) の純関数で、数字・実験条件・機構語が 1 文字も無い。"""
    for tpl in L.TEXT.values():
        assert not any(ch.isdigit() for ch in tpl), tpl
    for word in L.ITEM_WORD.values():
        assert not any(ch.isdigit() for ch in word), word
    for event in L.TEXT:
        for item in L.ITEMS:
            line = L.sentence(event, item)
            assert line and not any(ch.isdigit() for ch in line)
            for banned in ("円", "%", "確率", "config", "step", "seed"):
                assert banned not in line, (event, item, banned)
    assert L.sentence("存在しない出来事", "phone") == ""    # 語彙外は捏造しない
    assert L.sentence("pickup", "存在しない品目") == ""


# --------------------------------------------------------------------------- #
# (A-2) ★静的検査(検収基準 ⑨⑩)
# --------------------------------------------------------------------------- #
def test_module_adds_exactly_one_stream_and_no_llm():
    """新乱数は "lost_drop" 1 本だけ・generate() の呼び出しサイトはゼロ。"""
    src = Path(L.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert "generate" not in (names | attrs), "LLM を撃っている"
    assert "llm" not in attrs
    # stream 名は 1 種類だけ(AST の呼び出し引数から実際に拾う)
    used = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "stream" and node.args
                and isinstance(node.args[0], ast.Constant)):
            used.add(node.args[0].value)
    assert used == {"lost_drop"}, f"新 stream が 1 本ではない: {sorted(used)}"


def test_module_never_scans_all_pairs():
    """共在は ``sim.percept_index`` 経由だけ(新しい全対全スキャンを足していない)。

    ``sim.agents`` の走査は 1 step 1 回の在場者列挙(``diversity.tick_crime`` と同じ前例)
    までは許すが、**入れ子の二重走査**(agent × agent)は 1 つも無い。
    """
    tree = ast.parse(Path(L.__file__).read_text(encoding="utf-8"))
    src = Path(L.__file__).read_text(encoding="utf-8")
    assert "percept_index" in src, "唯一の共在索引を使っていない"
    for outer in ast.walk(tree):
        if not isinstance(outer, ast.For):
            continue
        if "agents" not in ast.dump(outer.iter):
            continue
        for inner in ast.walk(outer):
            if inner is outer or not isinstance(inner, ast.For):
                continue
            assert "agents" not in ast.dump(inner.iter), "全対全スキャンを足している"


def test_frozen_metric_spec_files_are_untouched():
    """凍結 14 ファイル(metrics_spec.SPEC_FILES)に本レーンの痕跡が 1 つも無い。"""
    from society.observer import metrics_spec as MS
    root = Path(__file__).resolve().parents[1]
    assert len(MS.SPEC_FILES) == 14
    for rel in MS.SPEC_FILES:
        text = (root / rel).read_text(encoding="utf-8")
        assert "lost_property" not in text and "lost_drop" not in text, \
            f"凍結ファイル {rel} に H3 の痕跡がある(metrics_spec_hash が動く)"


def test_chance_module_is_untouched():
    """★chance.py は 1 バイトも触らない(溶解は H3+H4 完了後の別バッチ = 計画書 §5)。"""
    text = Path(chance_mod.__file__).read_text(encoding="utf-8")
    assert "lost_property" not in text and "lost_drop" not in text


# =========================================================================== #
# (B) OFF = バイト一致(検収基準 ①)
# =========================================================================== #
def test_off_matches_golden(tmp_path):
    golden = json.load(open(GOLDEN, encoding="utf-8"))
    sim = _sim(tmp_path, "lp_golden", **_GOLDEN_NEUTRAL)
    sim.run()
    assert _l1(sim) == golden, "H3 の seam がゴールデンを動かしている"


def test_off_matches_pure_default(tmp_path):
    pure = _sim(tmp_path, "lp_pure", n_steps=72, n_agents=12)
    pure.run()
    off = _sim(tmp_path, "lp_off", n_steps=72, n_agents=12, **OFF)
    off.run()
    assert _l1x(pure) == _l1x(off)


def test_off_emits_nothing_and_grows_no_state(tmp_path):
    """OFF では phase を直接叩いても L1 も state も所持金も 1 も動かない。"""
    sim = _sim(tmp_path, "lp_off_noop", n_steps=1)
    money = [float(a.money) for a in sim.agents]
    n_events = len(sim.logger.events)
    for step in range(20):
        L.phase(sim, step, 420 + step * 10)
    assert not [e for e in sim.logger.events[n_events:] if e.kind.startswith("lost_")]
    assert getattr(sim, "_lost_state", None) is None
    assert L.provenance(sim) is None
    assert [float(a.money) for a in sim.agents] == money


def test_off_summary_has_no_key(tmp_path):
    sim = _sim(tmp_path, "lp_sum_off", n_steps=24, n_agents=10)
    sim.run()
    js = json.loads((tmp_path / "lp_sum_off" / "summary.json").read_text(encoding="utf-8"))
    assert "lost_property" not in js


# =========================================================================== #
# (C) ON: 全段が実際に立つ(検収基準 ②)
# =========================================================================== #
@pytest.fixture(scope="module")
def loud(tmp_path_factory):
    """落下率を上げた隔離ラン(全段が立つ材料)。``L.phase`` だけを回す。"""
    d = tmp_path_factory.mktemp("lp_loud")
    sim = Simulation(_cfg("lp_loud", 2, 40, **dict(
        LOUD, **{"lost_property.abandon_steps": "20",
                 "lost_property.statute_steps": "40"})), out_dir=d / "lp_loud")
    _huddle(sim, 20)
    total0 = sum(float(a.money) for a in sim.agents)
    _drive(sim, 200)
    return sim, total0


def test_every_stage_of_the_loop_fires(loud):
    sim, _ = loud
    for kind in LOST_KINDS:
        assert _kind(sim, kind), f"{kind} が 1 件も立っていない"


def test_drop_payload_carries_the_precursor_state(loud):
    """★L1 は 1 行 + **前兆状態**(混雑・飲酒・品目)同梱(計画書 §3)。"""
    sim, _ = loud
    for e in _kind(sim, "lost_drop"):
        p = e.payload
        assert set(p) >= {"item", "node", "crowd", "drinking", "moving", "rain"}
        assert p["item"] in L.ITEMS
        assert isinstance(p["crowd"], int) and p["crowd"] >= 1
        assert isinstance(p["drinking"], bool) and isinstance(p["rain"], bool)
    # 財布だけが現金を載せる(所持金から分離した実額)
    with_cash = {e.payload["item"] for e in _kind(sim, "lost_drop")
                 if "cash" in e.payload}
    assert with_cash <= {"wallet"}


def test_pickup_and_keep_carry_the_guardian_count(loud):
    """★着服は crime を出さず、payload の offense / guardians(RAT の語彙)で宣言する。"""
    sim, _ = loud
    assert not _kind(sim, "crime"), "diversity の較正済み crime 系列に混線している"
    for e in _kind(sim, "lost_keep"):
        assert e.payload["offense"] == "占有離脱物横領"
        assert isinstance(e.payload["guardians"], int)
    for e in _kind(sim, "lost_pickup"):
        assert isinstance(e.payload["guardians"], int)
        assert e.payload["lying_steps"] >= 1, "落ちた step のうちに拾われている"


def test_turnin_and_notice_name_a_post(loud):
    """届出・届け先が記録される(移動は伴わない簡約だが、行為のイベントは必須)。"""
    sim, _ = loud
    posts = set(L._post_nodes(sim))
    for e in _kind(sim, "lost_turnin") + _kind(sim, "lost_notice"):
        assert "post" in e.payload
        if posts:
            assert e.payload["post"] in posts


def test_memory_lines_are_the_template_only(loud):
    """当事者の記憶に入るのは定型 1 行だけ(数字・金額が 1 文字も無い)。"""
    sim, _ = loud
    templates = {L.sentence(ev, it) for ev in L.TEXT for it in L.ITEMS}
    seen = 0
    for a in sim.agents:
        for ep in list(a.mem.buffer) + list(a.mem.episodes):
            text = str(ep.text)
            if "拾" in text or "失く" in text or "落とし物" in text:
                assert text in templates, text
                seen += 1
    assert seen > 0, "記憶に 1 行も入っていない(テスト前提が崩れた)"


# =========================================================================== #
# (D) ★貨幣保存(検収基準 ③ = §4 の核心)
# =========================================================================== #
def test_money_is_conserved_across_the_whole_loop(loud):
    """Σ(所持金) + 遺失物の中の現金 + 失効累計 = **一定**。

    ★これが「拾得金は必ず誰かの drop から」の機械的な意味である。金が湧く経路が
      1 つでもあれば(= chance.windfall のように受け取り手だけが居れば)ここが破れる。
    """
    sim, total0 = loud
    st = sim._lost_state
    total = (sum(float(a.money) for a in sim.agents)
             + float(st["hold"]) + float(st["lapsed"]))
    assert abs(total - total0) < 1e-6, f"貨幣が湧いた/消えた: {total - total0}"
    assert st["hold"] >= 0.0 and st["lapsed"] >= 0.0


def test_conservation_holds_at_every_step(tmp_path):
    """毎 step 検査(累積の相殺で偶然合っているのではないことの確認)。"""
    sim = Simulation(_cfg("lp_cons", 2, 30, **dict(
        LOUD, **{"lost_property.abandon_steps": "12"})), out_dir=tmp_path / "lp_cons")
    _huddle(sim, 15)
    total0 = sum(float(a.money) for a in sim.agents)
    worst = 0.0
    for step in range(120):
        sim.percept_index = build_index(
            sim.agents, float(sim.cfg.world.perception_radius_m))
        L.phase(sim, step, 420 + step * 10)
        st = sim._lost_state
        worst = max(worst, abs(sum(float(a.money) for a in sim.agents)
                               + st["hold"] + st["lapsed"] - total0))
    assert worst < 1e-6, f"step 途中で貨幣保存が破れた: {worst}"


def test_wallet_cash_is_separated_from_the_owner_at_drop(tmp_path):
    """財布の現金は落とした瞬間に所持金から**分離**される(持ち主はもう使えない)。"""
    sim = _sim(tmp_path, "lp_sep", n_steps=1, n_agents=6, **ON)
    cfg = L.cfg_of(sim)
    st = L._state(sim)
    owner = sim.agents[0]
    owner.money = 20_000.0
    L._drop(sim, cfg, st, owner, L.WALLET, 0, 420, 3, False, False, True)
    cash = float(list(st["items"].values())[0]["cash"])
    assert cash == pytest.approx(20_000.0 * cfg["wallet_cash_frac"], abs=0.1)
    assert float(owner.money) == pytest.approx(20_000.0 - cash, abs=1e-6)
    assert float(st["hold"]) == pytest.approx(cash, abs=1e-6)


def test_lapsed_cash_is_recorded_as_a_non_transaction(tmp_path):
    """誰にも拾われずに失われた現金は **K5(取引でない資産変動)** へ落ちる。"""
    assert "lost_property" in SFC.K5_KINDS
    assert not (set(SFC.K5_KINDS) & set(SFC.CHANNELS)), "RoW と名前空間が衝突している"
    sim = _sim(tmp_path, "lp_k5", n_steps=1, n_agents=6,
               **dict(ON, **{"economy.org_accounting.enabled": "true",
                             "lost_property.abandon_steps": "1"}))
    cfg = L.cfg_of(sim)
    st = L._state(sim)
    owner = sim.agents[0]
    owner.money = 10_000.0
    before = SFC.total_money(sim)
    L._drop(sim, cfg, st, owner, L.WALLET, 0, 420, 1, False, False, False)
    assert SFC.total_money(sim) == pytest.approx(before, abs=1e-6), \
        "落とした瞬間に街から金が消えた(city_total に遺失物の在庫が入っていない)"
    sim.percept_index = build_index(sim.agents, 1.0)
    L._phase_resolve(sim, cfg, st, 10, 520, 8)
    assert float(st["lapsed"]) > 0.0
    assert SFC.k5_total(sim) == pytest.approx(float(st["lapsed"]), abs=1e-6)
    assert SFC.total_money(sim) == pytest.approx(before, abs=1e-6), \
        "失効で閉じた不変量(city + RoW + K5)が破れた"


# =========================================================================== #
# (E) ★品目別返還率の帯検証(検収基準 ④ = 計画書 §3 の検証ターゲット)
# =========================================================================== #
#: 東京都遺失物センターの実測(返還率 = 返還数 / **拾得受理数**)。
RETURN_BANDS: dict[str, tuple[float, float]] = {
    "umbrella": (0.005, 0.030),      # ≈1%
    "wallet":   (0.60, 0.80),        # 60-80%
    "phone":    (0.80, 0.92),        # 87%
}


class _Owner:
    """較正検証用の合成個体(traits の分布は factors.registry.sample_traits と同じ)。"""

    def __init__(self, aid: int, traits: dict, states: dict):
        self.id = aid
        self.traits = traits
        self.states = states


def _synthetic_owners(n: int) -> list[_Owner]:
    import numpy as np
    from society.factors.registry import sample_traits
    rng = np.random.default_rng(20260812)
    out = []
    for i in range(n):
        out.append(_Owner(i, sample_traits(rng),
                          {"grievance": float(rng.uniform(0.0, 0.4)),
                           "ownership": float(rng.uniform(0.0, 0.4))}))
    return out


def test_return_rate_by_item_matches_the_measured_bands():
    """★**届けられた物のうち持ち主に返る割合**が品目別の実測帯に入る。

    返還は 2 段の連言(持ち主が気づく × 引き取りに来る)で、どちらも**乱数を引かず**
    決定論ジッタと性格の関数で決まる。1000 件の合成拾得で実現率を測る。
    """
    cfg = L.build_cfg({"enabled": True})
    owners = _synthetic_owners(1000)
    got: dict[str, float] = {}
    for item, (lo, hi) in RETURN_BANDS.items():
        returned = 0
        for k, owner in enumerate(owners):
            iid = k + 1
            if L._jitter(0, "notice", owner.id, iid) >= L.notice_p(cfg, owner, item):
                continue
            if L._jitter(0, "claim", owner.id, iid) < L.claim_p(cfg, owner, item):
                returned += 1
        rate = returned / len(owners)
        got[item] = rate
        assert lo <= rate <= hi, (
            f"{item} の返還率 {rate:.3f} が実測帯 [{lo}, {hi}] の外"
            f"(実測値一覧: {got})")
    # 品目差が支配的であること(= 「取りに行く価値があるか」は物で決まる)
    assert got["umbrella"] < got["wallet"] < got["phone"]


def test_pickup_and_turnin_rates_are_in_a_plausible_range():
    """拾得率・届出率も退化していない(0% / 100% に張り付かない)。"""
    cfg = L.build_cfg({"enabled": True})
    finders = _synthetic_owners(1000)
    for item in L.ITEMS:
        picked = sum(1 for k, f in enumerate(finders)
                     if L._jitter(0, "pickup", f.id, k + 1) < L.pickup_p(cfg, f, item))
        assert 0.05 < picked / len(finders) < 0.98, (item, picked)
    turned = sum(1 for k, f in enumerate(finders)
                 if L._jitter(0, "turnin", f.id, k + 1)
                 < L.turnin_p(cfg, f, "wallet", 20_000.0, 0))
    assert 0.55 < turned / len(finders) < 0.90, turned


def test_the_choice_functions_read_personality_and_guardians():
    """届ける/着服の選択が **性格・規範状態・金額・監視者** に反応する(RAT の語彙)。"""
    cfg = L.build_cfg({"enabled": True})
    calm = _Owner(1, {"risk_tolerance": 0.1, "internal_locus": 0.9, "nfc": 0.5},
                  {"grievance": 0.0, "ownership": 0.3})
    wild = _Owner(2, {"risk_tolerance": 0.9, "internal_locus": 0.1, "nfc": 0.5},
                  {"grievance": 0.4, "ownership": 0.3})
    assert L.turnin_p(cfg, calm, "wallet", 1_000.0, 0) > \
        L.turnin_p(cfg, wild, "wallet", 1_000.0, 0), "性格が効いていない"
    assert L.turnin_p(cfg, wild, "wallet", 1_000.0, 0) > \
        L.turnin_p(cfg, wild, "wallet", 50_000.0, 0), "金額(誘惑)が効いていない"
    assert L.turnin_p(cfg, wild, "wallet", 1_000.0, 3) > \
        L.turnin_p(cfg, wild, "wallet", 1_000.0, 0), "監視者(guardianship)が効いていない"
    # 気づく/引き取るは品目が支配的
    assert L.claim_p(cfg, calm, "umbrella") < L.claim_p(cfg, wild, "phone")


def test_baseline_drop_rate_matches_the_tokyo_anchor():
    """★較正の出どころ: 都実測 454 万件/年 → 渋谷来街者スケール ≈2.9e-3 件/人日。

    基底(共変量なし)の合計は 2.5e-3 件/人日で、共変量込みの実現値が目標帯に乗る。
    """
    total = sum(L.BASE_DAILY.values())
    assert 1.5e-3 <= total <= 3.0e-3, total
    # 品目構成が警視庁の受理内訳の桁(その他が大半・傘と携帯が数 %)
    assert L.BASE_DAILY["other"] / total > 0.6
    for item in ("umbrella", "wallet", "phone"):
        assert 0.02 < L.BASE_DAILY[item] / total < 0.20, item


class _FakeSim:
    """``_drop_mults`` が読むのは node のカテゴリ集合だけ(キャッシュを直に据える)。"""

    def __init__(self, cats):
        self._lost_node_cats = {"n": frozenset(cats)}


class _Carrier:
    def __init__(self, building):
        self.building = building
        self.node = "n"


def test_realized_drop_rate_matches_the_tokyo_anchor():
    """★**共変量込みの実現落下率**が実測アンカー(≈2.9e-3 件/人日)の帯に乗る。

    モンテカルロで数えると 1 ラン数件のオーダー(ポアソン雑音で無意味)になるので、
    **実コードの倍率関数 ``_drop_mults`` を代表的な曝露分布で期待値化**して測る
    (較正が壊れたら必ず落ち、乱数の運では落ちない)。曝露は渋谷の来街者像:
    移動中 40% / 混雑ノード 20% / 飲酒帯の店内 5% / 雨天日 30% / 店内滞在 20%。
    """
    cfg = L.build_cfg({"enabled": True})
    p_move, p_crowd, p_drink, p_rain, p_indoor = 0.40, 0.20, 0.05, 0.30, 0.20
    sim_in = _FakeSim({"food"})
    sim_out = _FakeSim(set())
    indoor_cats = frozenset(cfg["indoor_cats"])
    total = 0.0
    for moving, w1 in ((True, p_move), (False, 1 - p_move)):
        for crowd, w2 in ((True, p_crowd), (False, 1 - p_crowd)):
            for drinking, w3 in ((True, p_drink), (False, 1 - p_drink)):
                for rain, w4 in ((True, p_rain), (False, 1 - p_rain)):
                    for indoor, w5 in ((True, p_indoor), (False, 1 - p_indoor)):
                        mult = L._drop_mults(
                            sim_in if indoor else sim_out, cfg,
                            _Carrier("b1" if indoor else None),
                            int(cfg["crowd_threshold"]) if crowd else 1,
                            rain, drinking, moving, indoor_cats)
                        w = w1 * w2 * w3 * w4 * w5
                        total += w * sum(cfg["base_daily"][it] * mult[k]
                                         for k, it in enumerate(L.ITEMS))
    assert 2.0e-3 <= total <= 5.0e-3, (
        f"実現落下率 {total:.5f} 件/人日が較正帯の外(実測アンカー ≈2.9e-3)")


def test_drop_hazard_is_actually_drawn_from_the_new_stream(tmp_path):
    """落下は新 stream "lost_drop" からの draw で決まる(既定 OFF では 1 粒も引かない)。"""
    sim = _sim(tmp_path, "lp_stream", n_steps=1, n_agents=8, **ON)
    cfg = L.cfg_of(sim)
    st = L._state(sim)
    # 基底率を 1.0/人日 に上げると、同じ step でも stream の値によって当たり外れが割れる
    for item in L.ITEMS:
        cfg["base_daily"][item] = 0.9
    sim.percept_index = build_index(sim.agents, 1.0)
    L._phase_drop(sim, cfg, st, 0, 420, 32)
    drawn = [float(sim.hub.stream("lost_drop", int(a.id), 0).random())
             for a in sim.agents]
    assert len(set(round(d, 9) for d in drawn)) > 1, "個体別キーになっていない"
    assert st["drops"], "新 stream 経路で 1 件も落ちていない"


# =========================================================================== #
# (F) 報労金(検収基準 ⑤)
# =========================================================================== #
def test_reward_rate_is_within_the_statutory_band():
    """遺失物法 28 条の 5-20% を決定論ジッタが一様に埋める(帯の外に出ない)。"""
    cfg = L.build_cfg({"enabled": True})
    rates = [L.reward_rate(cfg, 0, i, i * 7 + 1) for i in range(2000)]
    assert min(rates) >= 0.05 and max(rates) <= 0.20
    assert 0.11 < sum(rates) / len(rates) < 0.14, "帯の中で偏っている"


def test_reward_is_a_direct_transfer_from_owner_to_finder(tmp_path):
    """報労金は **持ち主 → 拾得者の直接授受**(2 人の合計は 1 円も変わらない)。"""
    sim = _sim(tmp_path, "lp_reward", n_steps=1, n_agents=6, **ON)
    cfg = L.cfg_of(sim)
    st = L._state(sim)
    owner, finder = sim.agents[0], sim.agents[1]
    owner.money, finder.money = 100_000.0, 5_000.0
    L._drop(sim, cfg, st, owner, L.WALLET, 0, 420, 2, False, False, True)
    rec = list(st["items"].values())[0]
    rec["noticed"], rec["reported"], rec["turn_step"] = True, True, 1
    pair0 = float(owner.money) + float(finder.money) + float(st["hold"])
    L._return(sim, cfg, st, rec, owner, finder, 5, 470, )
    assert float(owner.money) + float(finder.money) == pytest.approx(pair0, abs=1e-6)
    ev = _kind(sim, "lost_return")[-1]
    paid = float(ev.payload["amount"])
    value = float(cfg["item_value"]["wallet"]) + float(rec["cash"])
    assert 0.05 * value - 0.6 <= paid <= 0.20 * value + 0.6
    assert float(ev.payload["cash"]) == pytest.approx(float(rec["cash"]), abs=0.05)


def test_keep_moves_the_cash_to_the_finder(tmp_path):
    """着服では中の現金が拾得者へ渡る(= 拾得金は必ず誰かの drop から)。"""
    sim = _sim(tmp_path, "lp_keep", n_steps=1, n_agents=6, **ON)
    cfg = L.cfg_of(sim)
    st = L._state(sim)
    owner, finder = sim.agents[0], sim.agents[1]
    owner.money, finder.money = 40_000.0, 0.0
    L._drop(sim, cfg, st, owner, L.WALLET, 0, 420, 1, False, False, False)
    rec = list(st["items"].values())[0]
    cash = float(rec["cash"])
    L._keep(sim, cfg, st, rec, finder, 0, 3, 450)
    assert float(finder.money) == pytest.approx(cash, abs=1e-6)
    assert float(st["hold"]) == pytest.approx(0.0, abs=1e-6)
    ev = _kind(sim, "lost_keep")[-1]
    assert ev.payload["guardians"] == 0            # 監視者不在(RAT)
    assert float(ev.payload["amount"]) == pytest.approx(cash, abs=0.05)


def test_statute_expiry_gives_the_item_to_the_finder(tmp_path):
    """3 か月時効(遺失物法 7 条)で拾得者が取得する。"""
    sim = _sim(tmp_path, "lp_statute", n_steps=1, n_agents=6,
               **dict(ON, **{"lost_property.statute_steps": "5"}))
    cfg = L.cfg_of(sim)
    st = L._state(sim)
    owner, finder = sim.agents[0], sim.agents[1]
    owner.money, finder.money = 30_000.0, 0.0
    L._drop(sim, cfg, st, owner, L.WALLET, 0, 420, 1, False, False, False)
    rec = list(st["items"].values())[0]
    cash = float(rec["cash"])
    rec["status"], rec["finder"], rec["turn_step"] = L.TURNED_IN, int(finder.id), 1
    rec["noticed"] = True                          # 気づいたが引き取りに来なかった
    sim.percept_index = build_index(sim.agents, 1.0)
    L._phase_resolve(sim, cfg, st, 20, 620, 8)
    ev = _kind(sim, "lost_expire")[-1]
    assert ev.payload["to"] == "finder"
    assert float(finder.money) == pytest.approx(cash, abs=1e-6)
    assert st["items"] == {}                       # 決着した物はストアから外れる


# =========================================================================== #
# (G) 決定論・resume・LLM 呼数(検収基準 ⑦⑧)
# =========================================================================== #
def test_on_is_deterministic_across_two_runs(tmp_path):
    a = Simulation(_cfg("lp_det_a", 144, 20, **LOUD), out_dir=tmp_path / "a")
    a.run()
    b = Simulation(_cfg("lp_det_b", 144, 20, **LOUD), out_dir=tmp_path / "b")
    b.run()
    ea = [x for x in _l1(a) if x[2].startswith("lost_")]
    eb = [x for x in _l1(b) if x[2].startswith("lost_")]
    assert ea and ea == eb
    assert [round(float(x.money), 6) for x in a.agents] == \
        [round(float(x.money), 6) for x in b.agents]


def test_llm_call_count_is_identical_on_and_off(tmp_path):
    """generate() の呼び出しサイトを 1 つも足さない(k 非依存)。"""
    acts = [json.dumps({"action": "stay"}, ensure_ascii=False),
            json.dumps({"action": "speak", "text": "こんにちは"}, ensure_ascii=False)]
    counts = []
    for name, ov in (("lp_k_off", OFF), ("lp_k_on", LOUD)):
        sim = Simulation(_cfg(name, 72, 12, **ov), out_dir=tmp_path / name)
        sim.llm = _FixedLLM(acts)
        sim.run()
        counts.append(sim.llm.calls)
    assert counts[0] == counts[1], f"LLM 呼数が ON/OFF で違う: {counts}"


def test_resume_matches_straight(tmp_path):
    """遺失物 ON で resume==straight(進行中の物と現金の在庫が checkpoint に載る)。"""
    ov = {**LOUD, "run.start_tod": "00:00", "run.natural_start": "true"}
    split, total = 144, 288
    straight_dir = tmp_path / "lp_straight"
    straight = Simulation(_cfg("lp_straight", total, 20, **ov), out_dir=straight_dir)
    straight.run()
    assert _kind(straight, "lost_drop"), "テスト前提が崩れた(落下ゼロ)"

    d = tmp_path / "lp_resumed"
    sim1 = Simulation(_cfg("lp_resumed", split, 20, **ov,
                           **{"observer.checkpoint_every": split}), out_dir=d)
    for step in range(split):
        scheduler.run_step(sim1, step)
    checkpoint.save(sim1, split, d / "checkpoint" / f"ckpt-{split:06d}.pkl.gz")
    sim1.logger.flush_segment()
    sim2 = Simulation(_cfg("lp_resumed", total, 20, **ov,
                           **{"observer.checkpoint_every": split}), out_dir=d)
    sim2.run(resume_from=d)
    for stem in ("l1_events", "l2_metrics", "l3_snapshots"):
        sa = pq.read_table(straight_dir / f"{stem}.parquet").to_pylist()
        sb = pq.read_table(d / f"{stem}.parquet").to_pylist()
        assert sa == sb, f"{stem} 不一致(lost_property resume)"
    ja = json.loads((straight_dir / "summary.json").read_text(encoding="utf-8"))
    jb = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    assert ja["lost_property"] == jb["lost_property"], \
        "遺失物のタリー/在庫が resume で straight と食い違う(中央管理の漏れ)"
    assert straight._lost_state["items"] == sim2._lost_state["items"]
    assert straight._lost_state["hold"] == pytest.approx(sim2._lost_state["hold"])


# =========================================================================== #
# (H) 観測(summary)
# =========================================================================== #
def test_summary_publishes_the_return_rate_with_its_denominator(tmp_path):
    sim = Simulation(_cfg("lp_sum", 144, 25, **LOUD), out_dir=tmp_path / "lp_sum")
    sim.run()
    js = json.loads((tmp_path / "lp_sum" / "summary.json").read_text(encoding="utf-8"))
    prov = js["lost_property"]
    assert prov["schema"] == L.SCHEMA
    assert set(prov["items"]) == set(L.ITEMS)
    for key in ("drops", "notices", "pickups", "passed_by", "turnins", "returns",
                "keeps", "expired", "lapses", "return_rate", "turnin_rate",
                "cash_held", "cash_lapsed", "reward_paid", "kept_cash"):
        assert key in prov, key
    assert sum(prov["drops"].values()) > 0
    for item, rate in prov["return_rate"].items():
        assert 0.0 <= rate <= 1.0
        assert prov["turnins"].get(item, 0) > 0, "分母ゼロの率が出ている"
