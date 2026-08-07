"""IF-E2 案B: org の会計主体化 + rest-of-world(``economy.org_accounting``・既定 OFF)の検証。

正典: docs/research/ifE2-org-accounting-research.md §4-3(実装への確定ディレクティブ 6 項)/
      §5(バッチ分解 E2-1〜3)。実装: src/society/economy_sfc.py。
house style は tests/test_traces.py / tests/test_rumors.py(既定 OFF レーンの流儀)に倣う。

確かめること(受入 8 点)
------------------------
① **OFF = 現行と 1 バイトも変わらない**(golden L1 バイト一致・キー不発生・state 不発生)
② ON の記帳: 賃金 → org 預金が減る / 消費 → org 預金が増える / 不足 → 自動当座借越 + L1
③ **スカラー総マネー保存**(ABCredit.jl ``test/stock_flow_consistency.jl`` と同型の
   ``isapprox(init_money, tot_money, atol)`` を 100 step 以上回す)
④ RoW チャネル分類の網羅(**未分類種の検知** = IF-E の監視装置との接続)
⑤ 検査①(Caiani の貨幣ストック保存)が org / 銀行 / RoW / 行政 の**全部門**で PASS する
⑥ ON・同 seed の 2 ラン が完全一致(決定論 = 乱数 stream を 1 本も引かない)
⑦ **resume == straight**(残高群が checkpoint の runtime で中央管理されている)
⑧ 税 / 利息 / 供託金の対称化が ON のときだけ起きる(OFF は現行挙動とバイト一致)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))          # scripts/ は package ではない
sys.path.insert(0, str(_ROOT / "src"))

import analyze_accounting as aa                      # noqa: E402

from society import b2b as B2BMOD                    # noqa: E402
from society import economy_sfc as SFC               # noqa: E402
from society.config import load_config               # noqa: E402
from society.engine import checkpoint, scheduler     # noqa: E402
from society.engine.simulation import Simulation     # noqa: E402
from society.rules import apply_bonus                # noqa: E402

GOLDEN = _ROOT / "tests" / "data" / "golden_baseline_l1.json"

# test_rumors.py:45 / test_traces.py:48 と同じ「意図的な既定挙動追加」の中立化
_GOLDEN_NEUTRAL = {"economy.wages.自営": 0, "planning.enabled": "false",
                   "transit_ride.taxi.enabled": "false",
                   "transit_ride.bus.enabled": "false", "rules.enabled": "false"}

OFF = {"economy.org_accounting.enabled": "false"}
ON = {"economy.org_accounting.enabled": "true"}

#: 経済まわりを一通り点火する小ラン(test_accounting.py の econ_run と同じ点火セット)。
ECON = {
    "agents.personas_file": "data/personas_80.json",
    "organizations.enabled": "true",
    "work.service.enabled": "true", "work.service.ledger.enabled": "true",
    "work.service.indoor_fields": "true",
    "government.enabled": "true",
    "economy.fixed_cost_daily": 300, "economy.visitor_refresh": "true",
}
#: 口座 E5 + 銀行 E-W1 まで点火する変種(家賃・利息・ATM・カードを踏む)。
ACCOUNTS = dict(ECON, **{
    "economy.accounts.enabled": "true", "economy.accounts.payday_dom": 1,
    "economy.bank.enabled": "true", "economy.bank.deposit_rate": 3.65,
})

# ---- 第98バッチ IF-E2 UNCOVERED: 残り 4 種の点火セット ----
#: 窃盗(diversity.tick_crime)。既定ランでは 1 件も起きないので明示的に点火する。
CRIME = {"society_diversity.enabled": "true", "society_diversity.crime_prob": 0.05}
#: 偶発の臨時収入 / 紛失(chance.tick_day)。encounter は金を動かさないので weight=0 で外す。
CHANCE = {"chance.enabled": "true", "chance.daily_rate": 1.0,
          "chance.events.encounter.weight": 0}
#: B2B 卸→小売(b2b.fulfill)。在庫(goods)ON が前提。
B2B = {"commerce.inventory.enabled": "true", "commerce.inventory.b2b.enabled": "true"}


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


def _run(tmp_path, name, n_steps=288, n_agents=40, **ov):
    sim = _sim(tmp_path, name, n_steps, n_agents, **ov)
    sim.run()
    return sim


# ===========================================================================
# ① OFF = 現行と 1 バイトも変わらない
# ===========================================================================
def test_off_matches_golden(tmp_path):
    golden = json.load(open(GOLDEN, encoding="utf-8"))
    sim = _sim(tmp_path, "sfc_golden", **_GOLDEN_NEUTRAL)
    sim.run()
    assert _l1(sim) == golden, "IF-E2 の seam がゴールデンを動かしている"


def test_off_matches_pure_default(tmp_path):
    pure = _sim(tmp_path, "sfc_pure")
    pure.run()
    off = _sim(tmp_path, "sfc_off", **OFF)
    off.run()
    assert _l1x(pure) == _l1x(off)


def test_off_emits_nothing_and_grows_no_state(tmp_path):
    """OFF では経済を全部点火しても state も新 L1 種も payload キーも 1 つも生えない。"""
    sim = _run(tmp_path, "sfc_off_noop", n_steps=144, **ECON)
    assert getattr(sim, "_sfc_state", None) is None, "OFF なのに会計 state が生えた"
    assert SFC.state_of(sim) is None
    assert SFC.provenance(sim) is None
    assert not _kind(sim, "org_overdraft") and not _kind(sim, "row_flow")
    for e in sim.logger.events:
        p = e.payload or {}
        assert "payee" not in p and "payer" not in p, f"OFF なのに相手方キー: {e.kind} {p}"
    assert not (tmp_path / "sfc_off_noop" / "finance.parquet").exists()


def test_off_summary_has_no_key(tmp_path):
    _run(tmp_path, "sfc_sum_off", n_steps=144, **ECON)
    summary = json.loads((tmp_path / "sfc_sum_off" / "summary.json")
                         .read_text(encoding="utf-8"))
    assert "org_accounting" not in summary


def test_off_prompts_are_byte_identical(tmp_path):
    """no-fingerprint: ON でもプロンプトが 1 バイトも変わらない(残高は観測層だけの概念)。"""
    def _prompts(name, **ov):
        sim = _sim(tmp_path, name, n_steps=144, n_agents=15, **ov)
        seen: list = []
        inner = sim.llm.generate

        def _gen(prompt, **kw):
            seen.append(prompt)
            return inner(prompt, **kw)
        sim.llm.generate = _gen
        sim.run()
        return seen

    base = _prompts("sfc_p_base", **ECON)
    on = _prompts("sfc_p_on", **dict(ECON, **ON))
    assert base and base == on, "org 残高がプロンプトへ漏れている(no-fingerprint 違反)"


def test_module_never_touches_prompt_or_rng(tmp_path):
    """機械固定: economy_sfc.py は乱数 stream も generate() も 1 つも呼ばない。"""
    src = Path(SFC.__file__).read_text(encoding="utf-8")
    body = src.split('"""', 2)[2]                   # module docstring を除いた本体
    for forbidden in ("hub.stream", "random(", "uniform(", "integers(",
                      "generate(", "remember(", "build_prompt"):
        assert forbidden not in body, f"economy_sfc.py が {forbidden} を含む"


# ===========================================================================
# ② ON の記帳: 賃金 → 減 / 消費 → 増 / 不足 → 当座借越
# ===========================================================================
def test_on_initial_balance_is_sigma_times_monthly_wage(tmp_path):
    """§3-a: 初期預金 = σ × (配属者の日給合計 × month_days)。乱数ゼロ・決定論。"""
    sim = _sim(tmp_path, "sfc_init", n_steps=1, n_agents=40, **dict(ECON, **ON))
    sim.run()
    st = SFC.state_of(sim)
    assert st is not None and st["init"]
    daily: dict[str, float] = {}
    for a in sim.agents:
        oid = getattr(a, "org_id", None)
        if oid and getattr(a, "org_role", "") != "学生":
            daily[str(oid)] = daily.get(str(oid), 0.0) + float(a.wage or 0.0)
    cfg = SFC.cfg_of(sim)
    factor = cfg["sigma"] * cfg["month_days"]
    for oid, d in daily.items():
        if d > 0:
            assert st["org"][oid] == pytest.approx(d * factor, rel=1e-9), oid
    # 期首資本は RoW → org の期首フローとして必ず両建てで記帳される(Cities:Skylines の反面教師)
    assert st["row"]["initial_capital"]["in"] == pytest.approx(sum(st["org"].values()))


def test_on_wage_debits_the_employer_org(tmp_path):
    """賃金 gross がその org の預金から出る(payload の payer が org_id になる)。"""
    sim = _run(tmp_path, "sfc_wage", n_steps=288, **dict(ECON, **ON))
    st = SFC.state_of(sim)
    paid: dict[str, float] = {}
    for e in _kind(sim, "wage"):
        payer = str((e.payload or {}).get("payer") or "")
        assert payer, "ON なのに wage に payer が無い"
        if payer in st["org"]:
            paid[payer] = paid.get(payer, 0.0) + float(
                e.payload.get("gross", e.payload["amount"]))
    assert paid, "org が払った賃金が 1 件も無い(ランが点火していない)"
    assert st["wage_out"] == pytest.approx(sum(paid.values()), rel=1e-9)


def test_on_spend_credits_the_resolved_payee(tmp_path):
    """消費のその場で受け手が確定し、org なら**実支払−消費税**が預金に入る。"""
    sim = _run(tmp_path, "sfc_spend", n_steps=288, **dict(ECON, **ON))
    st = SFC.state_of(sim)
    payees = [str((e.payload or {}).get("payee") or "") for e in _kind(sim, "spend")]
    assert payees and all(payees), "ON なのに spend に payee が無い"
    assert any(p in st["org"] for p in payees), "org へ着地した消費が 1 件も無い"
    assert any(p.startswith("row:") for p in payees), "RoW へ落ちた消費が無い(開示が空)"
    assert st["revenue_in"] > 0.0
    # 受け手解決の段は全部で 4 通りしかなく、件数の合計 = spend 件数
    assert sum(st["payee"].values()) == len(payees)


def test_on_overdraft_is_allowed_and_logged(tmp_path):
    """§4-3 ディレクティブ 2: 不足でも払う(負残高許容)+ L1 org_overdraft。破綻処理は無い。"""
    # 初期預金をゼロにし輸出も切ると、最初の賃金支払で必ず当座借越に落ちる。
    sim = _run(tmp_path, "sfc_od", n_steps=288,
               **dict(ECON, **ON, **{"economy.org_accounting.sigma": 0.0,
                                     "economy.org_accounting.export_production": "false"}))
    st = SFC.state_of(sim)
    assert st["n_overdraft"] > 0, "当座借越が 1 度も起きていない(条件が弱い)"
    ev = _kind(sim, "org_overdraft")
    assert ev, "当座借越の L1 が出ていない"
    assert {"org", "amount", "balance", "shortfall", "reason"} <= set(ev[0].payload)
    assert ev[0].payload["balance"] < 0.0
    assert any(v < 0.0 for v in st["org"].values()), "負残高が許容されていない"
    # 破綻処理は入れない(Poledna の AND 条件は将来の拡張として docstring 宣言のみ)
    assert not _kind(sim, "org_bankrupt")


def test_on_export_revenue_reaches_only_non_customer_facing_orgs(tmp_path):
    """§3-c (ii): 域内に客が居ない org(office/education)だけが輸出代金を受け取る。"""
    sim = _run(tmp_path, "sfc_exp", n_steps=288, **dict(ECON, **ON))
    idx = SFC._book_index(sim)["export"]
    got = {str(e.payload["org"]) for e in _kind(sim, "production")
           if "revenue" in (e.payload or {})}
    assert got, "輸出代金を受け取った org が 1 社も無い"
    assert got <= idx, "接客業種に輸出代金が入っている(二重計上)"


def test_payee_index_on_the_real_11k_book_matches_the_researched_uniqueness():
    """研究文書 §3-c の全数測定を**解決器そのもの**で機械固定する(本番台帳 11,010 社)。

    node 鍵だけでは原理的にほぼ決まらず(4.5%)、``floor`` を鍵に入れた瞬間に一意率が
    跳ね上がる(56.8%)。= **indoor が org 単位の売上帰属の事実上の前提**であることの根拠。
    決まらない分は RoW へ落ちる(捏造しない)。"""
    from society import organizations as ORGS
    book_path = _ROOT / "data" / "organizations_shibuya_wide11k.json"
    if not book_path.exists():                      # 台帳が無い環境ではスキップ
        pytest.skip("11k 台帳が無い")
    book = ORGS.load_book({"book": str(book_path)})

    class _S:                                       # sim の最小スタブ(索引は book だけを読む)
        orgs = book
    idx = SFC._book_index(_S())
    n = len(book)
    assert n >= 11_000
    n_floor_keys, n_node_keys = set(), set()
    for org in book.values():
        wp = org.get("workplace_poi") or {}
        if wp.get("cat"):
            if wp.get("building"):
                n_floor_keys.add((str(wp["building"]), int(wp.get("floor") or 0),
                                  str(wp["cat"])))
            if wp.get("node"):
                n_node_keys.add((str(wp["node"]), str(wp["cat"])))
    # §3-c 実測: (building, floor, POI種別) は 7,585 鍵中 56.8% が一意 / (node, POI種別) は 4.5%
    assert 0.50 <= len(idx["floor"]) / len(n_floor_keys) <= 0.62, \
        f"floor 鍵の一意率が想定外: {len(idx['floor'])}/{len(n_floor_keys)}"
    assert len(idx["node"]) / len(n_node_keys) < 0.10, \
        f"node 鍵だけで決まりすぎている(測定と矛盾): {len(idx['node'])}/{len(n_node_keys)}"
    # 域内に客が居ない org(office/education)= 全体の 3 割強(§3-c の 36.5%)
    assert 0.30 <= len(idx["export"]) / n <= 0.40


# ===========================================================================
# ③ スカラー総マネー保存(ABCredit.jl 流の atol 検査)
# ===========================================================================
@pytest.mark.parametrize("tag,extra", [("plain", ECON), ("accounts", ACCOUNTS)])
def test_total_money_is_conserved_every_step(tmp_path, tag, extra):
    """**閉じた不変量**: Σ(全主体残高) + RoW 累積 = 一定。100 step 以上を毎 step 検査する。

    ABCredit.jl(イタリア中銀)の test/stock_flow_consistency.jl と同型
    (``isapprox(init_money, tot_money, atol=1e-5)`` を 100 step 回す)。許容は絶対誤差ではなく
    **総額に対する相対**で置く(残高は円で 1e8 オーダー・浮動小数の丸めで厳密ゼロは不成立)。"""
    sim = _sim(tmp_path, f"sfc_cons_{tag}", n_steps=288, n_agents=40,
               **dict(extra, **ON))
    seen: list[float] = []
    orig = scheduler.run_step
    try:
        def patched(s, step):
            orig(s, step)
            seen.append(SFC.total_money(s))
        scheduler.run_step = patched
        sim.run()
    finally:
        scheduler.run_step = orig
    assert len(seen) == 288
    spread = max(seen) - min(seen)
    scale = max(abs(seen[0]), 1.0)
    assert spread / scale < 1e-12, (
        f"総マネーが保存していない: 振れ幅 {spread} / 総額 {seen[0]}")
    # 検査が空虚でないこと: 実際に金が動いている(RoW も両建てで動いている)。
    # ★口座 ON の 2 日ランでは月給まとめの給料日が来ないので org 発の賃金は 0 になりうる
    #   (日割り支給が payday へ繰り延べられるため)。そこは or で正直に許す。
    st = SFC.state_of(sim)
    assert st["revenue_in"] > 0 and (st["wage_out"] > 0 or st["export_in"] > 0)
    assert SFC.row_net(sim) != 0.0
    assert sum(v["in"] for v in st["row"].values()) > 0
    assert sum(v["out"] for v in st["row"].values()) > 0


def test_floor_clamped_spend_stays_conserved(tmp_path):
    """所持金 < 名目価格(床クリップ)でも不変量が壊れない。**実支払**を基準に配るため。

    ★受け手が org / RoW のときは RoW を経由させてはいけない(経由させると総マネーが
      クリップ差額ぶん減る)。この経路は実ランでほぼ発火しないので、直に踏んで固定する。"""
    sim = _sim(tmp_path, "sfc_clamp", n_steps=1, n_agents=15, **dict(ECON, **ON))
    sim.run()
    a = sim.agents[0]
    a.money, a.account = 100.0, 0.0
    before = SFC.total_money(sim)
    n0 = len(sim.logger.events)
    scheduler._spend(sim, a, 5000.0, "food", 0, 0)      # 100 円しか払えない
    assert a.money == 0.0
    ev = [e for e in sim.logger.events[n0:] if e.kind == "spend"][0]
    assert ev.payload["paid"] == pytest.approx(100.0), "実支払が開示されていない"
    assert SFC.total_money(sim) == pytest.approx(before, abs=1e-6), \
        "床クリップで総マネーが動いた"


def test_conservation_detector_is_armed(tmp_path):
    """検出器が生きている: 世界から 1 円抜けば不変量が壊れることを示す(検査の空虚化防止)。"""
    sim = _sim(tmp_path, "sfc_armed", n_steps=24, n_agents=15, **dict(ECON, **ON))
    sim.run()
    before = SFC.total_money(sim)
    sim.agents[0].money -= 1.0                     # 相手方の無い出金 = 貨幣の消滅
    assert SFC.total_money(sim) == pytest.approx(before - 1.0)


# ===========================================================================
# ④ RoW チャネル分類の網羅(IF-E の監視装置との接続)
# ===========================================================================
def test_row_channels_are_declared_and_closed():
    """RoW の窓口は宣言済みの有限表に閉じる(「何でも吸収する RoW」の防止 = §4-1)。"""
    assert set(SFC.CHANNELS) == set(SFC.CHANNELS_IN) | set(SFC.CHANNELS_OUT)
    assert len(SFC.CHANNELS) == len(set(SFC.CHANNELS)), "チャネル名が重複している"
    assert not (set(SFC.CHANNELS_IN) & set(SFC.CHANNELS_OUT)), "向きが両属のチャネルがある"


def test_money_kinds_are_partitioned_into_covered_and_declared_uncovered():
    """**IF-E の監視装置との接続**: analyze_accounting が知る金の経路は、本 module が
    「接続した」か「接続できていない(理由つき)」のどちらかに必ず分類されている。

    新しい金の経路が実装されると analyze 側の MONEY_KINDS へ足すことになり、そのとき
    本テストが落ちる = 会計の網羅性が構造的に守られる(第58 の掟と同じ流儀)。"""
    known = SFC.COVERED_KINDS | set(SFC.UNCOVERED_KINDS)
    missing = set(aa.MONEY_KINDS) - known
    assert not missing, (
        f"分類されていない金の経路: {sorted(missing)}。"
        "economy_sfc.COVERED_KINDS(接続済み)か UNCOVERED_KINDS(理由つきで未接続)へ足すこと")
    assert not (SFC.COVERED_KINDS & set(SFC.UNCOVERED_KINDS)), "両方に入っている種がある"
    for kind, why in SFC.UNCOVERED_KINDS.items():
        assert why.strip(), f"{kind}: 未接続の理由が空(ゼロと偽らない義務)"


def test_row_channel_breakdown_is_published(tmp_path):
    """§4-1 (c): RoW 累積は summary / L1 / サイドカーの一級市民として公表される。"""
    sim = _run(tmp_path, "sfc_row", n_steps=288, **dict(ECON, **ON))
    prov = SFC.provenance(sim)
    assert prov is not None
    assert prov["row_channels"], "チャネル内訳が空"
    assert set(prov["row_channels"]) <= set(SFC.CHANNELS), "未宣言のチャネルが出た"
    assert prov["row_in_total"] > 0 and prov["row_out_total"] > 0
    # 域外依存の代表 3 経路(輸出代金・来街者の財布=輸出・帰属不能な消費)が実測で立つ
    for ch in ("export_production", "visitor_refill", "unknown_payee"):
        assert ch in prov["row_channels"], ch
    flows = _kind(sim, "row_flow")
    assert flows, "L1 に日次の域外収支が出ていない"
    # 第99(IF-E2 残③): K5 も L1 の一級市民になった(それまで finance.parquet 専用だった)
    assert {"day", "channels", "in_total", "out_total", "net",
            "k5_total"} <= set(flows[0].payload)
    summary = json.loads((tmp_path / "sfc_row" / "summary.json").read_text(encoding="utf-8"))
    assert summary["org_accounting"]["row_net"] == prov["row_net"]
    assert "uncovered_kinds_declared" in summary["org_accounting"], "未接続の宣言が消えている"
    # 第98バッチ IF-E2 UNCOVERED: 4 種を接続したので**空**。ただしキーは監視装置として残す
    #(「装置は残したまま値をゼロにする」)。
    assert summary["org_accounting"]["uncovered_kinds_declared"] == {}


def test_finance_sidecar_has_the_declared_schema(tmp_path):
    """L2 サイドカー finance.parquet(日次 1 行)= 検査①を全部門で成立させる唯一の観測点。"""
    _run(tmp_path, "sfc_side", n_steps=288, **dict(ECON, **ON))
    t = pq.read_table(tmp_path / "sfc_side" / "finance.parquet")
    assert tuple(t.column_names) == SFC.COLUMNS
    rows = t.to_pylist()
    assert len(rows) >= 3, "日次行が足りない(期首アンカー + 日境界 + 最終)"
    assert rows[0]["step"] == 0, "期首アンカーが無い(窓の基準点)"
    assert rows[-1]["step"] == 288, "最終行が n_steps でない(窓のオフバイワン)"
    assert rows[-1]["org_count"] > 0 and rows[-1]["org_balance"] != 0.0


# ===========================================================================
# ⑤ 検査①(Caiani の貨幣ストック保存)が全部門で成立する
# ===========================================================================
@pytest.fixture(scope="module")
def acct_on(tmp_path_factory):
    """mock 小ラン(288 step = 2日・40体)を ON で走らせ analyze_accounting にかける。"""
    out = tmp_path_factory.mktemp("sfc_aa")
    dot = ["run.seed=20260805", "run.n_agents=40", "run.n_steps=288",
           "run.name=sfc_aa", "model.backend=mock", "observer.snapshot_every=12"]
    dot += [f"{k}={v}" for k, v in dict(ACCOUNTS, **ON).items()]
    d = out / "sfc_aa"
    Simulation(load_config(dot), out_dir=d).run()
    return aa.analyze(aa.load_run(d), rel_tol=1e-4)


def test_measured_leak_is_closed(acct_on):
    """受入条件 (i)(iii): 漏れ比率 < 1%(丸めのみ)・void 行/列の絶対額 = 0。"""
    t = acct_on["totals"]
    assert t["leak_share"] < 0.01, f"漏れが残っている: {t['leak_share']*100:.2f}%"
    assert t["money_created"] == 0.0 and t["money_destroyed"] == 0.0
    assert acct_on["leak_families"] == [], f"漏れの族が残っている: {acct_on['leak_families']}"
    v = acct_on["sector_sums"][aa.VOID]
    assert v["inflow"] == 0.0 and v["outflow"] == 0.0, "void 行・列が空でない"


def test_measured_check1_passes_for_every_balance_holding_sector(acct_on):
    """受入条件 (ii): **残高を持つ全部門**で検査①(期首+流入−流出=期末)が成立する。

    venture(屋台)は売上が即座に店主=家計へ抜ける**通過部門**で世界に残高そのものが無く、
    void は「相手方が存在しない金」の検出器なので、この 2 つだけは原理的に残高を持たない
    (= 案B 後も observable にはならない。ゼロと偽らずそう表示する)。"""
    for sec in (aa.HOUSEHOLD, aa.ORG, aa.BANK, aa.GOVERNMENT, aa.EXTERNAL):
        c = acct_on["checks"][sec]
        assert c.get("observable"), f"{sec} の残高が観測できない: {c.get('reason')}"
        assert c["pass"], (f"{sec} の貨幣ストック保存が破れた: "
                          f"残差={c['total_residual']} 相対={c['relative_residual']}")
    for sec in (aa.VENTURE, aa.VOID):
        assert not acct_on["checks"][sec].get("observable")
    # 行政は public_budget 由来の既存検査に加え、供託(escrow)を含む finance 由来でも成立する
    gv = acct_on["checks"][aa.GOVERNMENT]["external_finance"]
    assert gv["observable"] and gv["pass"], f"行政(finance 由来)が破れた: {gv}"
    assert gv["open_balance"] > 0.0, "行政の期首残高が 0(遅延構築が期首基準に入っていない)"


def test_measured_classifier_and_monitors_stay_green(acct_on):
    assert acct_on["classifier_consistency"]["pass"]
    assert acct_on["unclassified_money_kinds"] == {}, (
        "会計検査が見ていない金の経路が増えた(MONEY_KINDS へ追加すること)")


def test_measured_payee_revenue_replaces_serve_matching(acct_on):
    """受け手は支払のその場で確定するので serve→spend 突合が要らない(220/222 不在問題の解消)。"""
    g = acct_on["revenue_gap"]
    assert g["n_payee_orgs"] > 0 and g["payee_revenue_total"] > 0.0
    # revenue_est(日給×margin の推定値)との乖離は**残る**。ゼロと偽らない。
    assert g["payee_ratio"] > 1.0
    assert acct_on["finance"]["n_rows"] >= 3


def test_measured_markdown_renders(acct_on):
    md = aa.render_markdown(acct_on)
    assert "IF-E2 案B" in md and "rest-of-world" in md
    assert "org_accounting=True" in md


# ===========================================================================
# ⑥ 決定論(同 seed の 2 ラン一致)
# ===========================================================================
def test_on_is_deterministic(tmp_path):
    a = _run(tmp_path, "sfc_det_a", n_steps=144, **dict(ECON, **ON))
    b = _run(tmp_path, "sfc_det_b", n_steps=144, **dict(ECON, **ON))
    # run.name が違うので L1 の内容だけを比べる(step/agent/kind/payload)
    assert _l1(a) == _l1(b)
    assert SFC.state_of(a)["org"] == SFC.state_of(b)["org"]
    assert SFC.state_of(a)["row"] == SFC.state_of(b)["row"]


def test_on_does_not_change_llm_call_count(tmp_path):
    """affects_k=False の機械固定: ON/OFF で generate() の呼数が 1 も動かない。"""
    def _n(name, **ov):
        sim = _sim(tmp_path, name, n_steps=144, n_agents=15, **ov)
        calls = []
        inner = sim.llm.generate

        def _gen(prompt, **kw):
            calls.append(1)
            return inner(prompt, **kw)
        sim.llm.generate = _gen
        sim.run()
        return len(calls)

    assert _n("sfc_k_off", **ECON) == _n("sfc_k_on", **dict(ECON, **ON))


# ===========================================================================
# ⑦ resume == straight(残高群の checkpoint 中央管理)
# ===========================================================================
def _rows(run_dir: Path, stem: str = "l1_events") -> list[dict]:
    return pq.read_table(Path(run_dir) / f"{stem}.parquet").to_pylist()


def _resume_cfg(name, n_steps, split, **ov):
    dot = ["run.seed=42", "run.n_agents=25", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock", "observer.snapshot_every=144",
           f"observer.checkpoint_every={split}"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def test_resume_matches_straight_with_org_accounting_on(tmp_path):
    """mid-day resume でも org 預金 / RoW 累積 / 行政・銀行残高が straight と一致する。"""
    total, split = 288, 150
    kw = dict(ACCOUNTS, **ON)
    st_dir = tmp_path / "sfc_straight"
    straight = Simulation(_resume_cfg("sfc_straight", total, split, **kw), out_dir=st_dir)
    straight.run()

    rs_dir = tmp_path / "sfc_resumed"
    sim1 = Simulation(_resume_cfg("sfc_resumed", split, split, **kw), out_dir=rs_dir)
    for step in range(split):
        scheduler.run_step(sim1, step)
    checkpoint.save(sim1, split, rs_dir / "checkpoint" / f"ckpt-{split:06d}.pkl.gz")
    sim1.logger.flush_segment()
    sim1.finance_sc.flush_segment()               # Simulation.run が checkpoint と対で行う分
    sim2 = Simulation(_resume_cfg("sfc_resumed", total, split, **kw), out_dir=rs_dir)
    sim2.run(resume_from=rs_dir)

    assert _rows(st_dir) == _rows(rs_dir), "resume の L1 が straight と不一致"
    a, b = SFC.state_of(straight), SFC.state_of(sim2)
    assert a["org"] == b["org"], "org 預金が resume で食い違う"
    assert a["row"] == b["row"], "RoW 累積が resume で食い違う"
    assert a["escrow"] == b["escrow"]
    assert _rows(st_dir, "finance") == _rows(rs_dir, "finance")


def test_checkpoint_carries_the_non_agent_balances(tmp_path):
    """前提工事(§3-0): 行政 / 銀行 / VC の残高が checkpoint に入り resume で戻る。

    これが無いと「org 預金は残るのに行政は初期値へ戻る」という非対称なバグになる
    (IF-E2 以前から存在した欠陥で、tests/test_resume.py は検査していなかった)。"""
    sim = _sim(tmp_path, "sfc_ckpt", n_steps=200, n_agents=25, **dict(ACCOUNTS, **ON))
    for step in range(200):
        scheduler.run_step(sim, step)
    path = checkpoint.save(sim, 200, tmp_path / "ck" / "ckpt.pkl.gz")
    gov_before = dict(sim.government.balance)
    bank_before = float(sim.bank.capital)
    assert any(v != 0.0 for v in gov_before.values())

    fresh = _sim(tmp_path, "sfc_ckpt2", n_steps=200, n_agents=25, **dict(ACCOUNTS, **ON))
    checkpoint.load(fresh, path)
    assert dict(fresh.government.balance) == gov_before, "行政残高が resume で戻らない"
    assert float(fresh.bank.capital) == bank_before, "銀行 capital が resume で戻らない"
    assert SFC.state_of(fresh)["org"] == SFC.state_of(sim)["org"]


# ===========================================================================
# ⑧ 税 / 利息 / 供託金の対称化が ON のときだけ起きる
# ===========================================================================
def test_interest_is_symmetric_only_when_on(tmp_path):
    """§3-e: interest_paid は OFF では Bank.capital を減らさない(=貨幣創出のまま)。
    ON では 1 行で対称化する。挙動差は ON のときだけ起きる。"""
    off = _run(tmp_path, "sfc_int_off", n_steps=288, **ACCOUNTS)
    on = _run(tmp_path, "sfc_int_on", n_steps=288, **dict(ACCOUNTS, **ON))
    paid = sum(float(e.payload["amount"]) for e in _kind(on, "interest_paid"))
    assert paid > 0.0, "銀行 ON が点火していない"
    assert getattr(off, "bank", None) is None or float(off.bank.capital) >= 0.0
    assert float(on.bank.capital) < 0.0, "ON で Bank.capital が利息ぶん減っていない"


def test_escrow_holds_the_proposal_deposit_only_when_on(tmp_path):
    """§3-e: 供託金は行政の**預り金**(balance とは別勘定)。OFF では受け手が居ない。"""
    kw = dict(ECON, **{"tools.enabled": "true", "assembly.deposit": 5000})
    on = _sim(tmp_path, "sfc_esc_on", n_steps=24, n_agents=15, **dict(kw, **ON))
    on.run()
    st = SFC.state_of(on)
    assert st is not None and st["escrow"] >= 0.0
    # 直接口を叩いて往復がゼロ和になることを固定する(供託は小ランでは滅多に起きない)
    SFC.escrow_in(on, 1000.0)
    assert st["escrow"] == pytest.approx(1000.0)
    SFC.escrow_out(on, 1000.0)
    assert st["escrow"] == pytest.approx(0.0)
    off = _sim(tmp_path, "sfc_esc_off", n_steps=1, n_agents=15, **kw)
    off.run()
    assert SFC.escrow_in(off, 1000.0) is None, "OFF で預り金が動いた"
    assert getattr(off, "_sfc_state", None) is None


def test_off_and_on_differ_only_where_declared(tmp_path):
    """ON/OFF の L1 差分は **新 2 種 + 相手方キー + 実支払キー**だけに閉じる(挙動の局在)。"""
    off = _run(tmp_path, "sfc_diff_off", n_steps=144, n_agents=25, **ECON)
    on = _run(tmp_path, "sfc_diff_on", n_steps=144, n_agents=25, **dict(ECON, **ON))
    new_kinds = {"org_overdraft", "row_flow"}
    a = [e for e in off.logger.events]
    b = [e for e in on.logger.events if e.kind not in new_kinds]
    assert len(a) == len(b), "新 2 種以外のイベント数が変わった(発火が動いた)"
    allowed = {"payer", "payee", "paid", "revenue"}
    for x, y in zip(a, b):
        assert (x.step, x.agent_id, x.kind) == (y.step, y.agent_id, y.kind)
        extra = set(y.payload or {}) - set(x.payload or {})
        assert extra <= allowed, f"{y.kind}: 宣言外のキーが増えた {sorted(extra)}"
        for k in set(x.payload or {}):
            if k in ("balance", "account", "cash"):
                continue                    # 残高そのものは ON でも変わらない想定だが
            assert (x.payload or {})[k] == (y.payload or {})[k], f"{y.kind}.{k} が変わった"


# ===========================================================================
# ⑨ 第98バッチ IF-E2 UNCOVERED — 残り 4 種の接続
#    (rule_bonus = 区の歳出 / crime = 非取引 K5 / chance_event = RoW / b2b = org 預金間の実移転)
# ===========================================================================
def test_uncovered_kinds_is_empty_and_every_money_kind_is_covered():
    """★受入の核: 接続できていない金の経路が **0 本**になった(辞書自体は装置として残す)。"""
    assert SFC.UNCOVERED_KINDS == {}, "宣言だけ残して実装が戻っている"
    assert set(aa.MONEY_KINDS) <= set(SFC.COVERED_KINDS)
    for k in ("rule_bonus", "crime", "chance_event", "b2b_trade"):
        assert k in SFC.COVERED_KINDS, k
    # K5 は RoW と**別勘定**(SNA 2008 §3.98: 窃盗は取引ではない)。名前空間も分ける。
    assert "theft" in SFC.K5_KINDS
    assert not (set(SFC.K5_KINDS) & set(SFC.CHANNELS))


def test_rule_bonus_is_funded_by_the_ward_budget_only_when_on(tmp_path):
    """① rule_bonus = 区(ward)予算からの移転。支給額もタイミングも 1 円も変えない。"""
    kw = dict(ECON, **{"rules.enabled": "true"})
    for name, ov, funded in (("sfc_bonus_off", kw, False),
                             ("sfc_bonus_on", dict(kw, **ON), True)):
        sim = _sim(tmp_path, name, n_steps=1, n_agents=15, **ov)
        sim.run()                       # step 先頭の _sfc_arm が行政/銀行/VC を実体化する
        gov = sim.government
        rec = sim.rulebook.enact({"type": "bonus", "behavior": "park", "amount": 500},
                                 name="公園", proposer=0, step=0, day=0)
        agent = sim.agents[0]
        before_money, before_ward = agent.money, gov.balance["ward"]
        before_total = SFC.total_money(sim) if funded else 0.0
        paid = apply_bonus(sim.rulebook, sim, agent, "park", 0, 0)
        ev = _kind(sim, "rule_bonus")[-1]
        # 動力学(支給額・累計・payload の金額)は ON/OFF で完全に同じ
        assert paid == 500.0 and agent.money == before_money + 500.0
        assert rec["spent"] == 500.0 and ev.payload["amount"] == 500.0
        if not funded:
            assert "payer" not in ev.payload
            assert gov.balance["ward"] == before_ward, "OFF なのに区予算が減った"
            assert getattr(sim, "_sfc_state", None) is None
            continue
        assert ev.payload["payer"] == "government"
        assert gov.balance["ward"] == pytest.approx(before_ward - 500.0)
        assert SFC.total_money(sim) == pytest.approx(before_total)
        assert SFC.state_of(sim)["bonus_out"] == pytest.approx(500.0)
        # ★議会の予算否決(exec_ratio)を掛けてはいけない: 支給は既に満額で済んでいるので、
        #   執行率で目減りさせると差額が無から生まれる(on_rule_bonus の docstring)。
        gov.exec_ratio = 0.5
        again = SFC.total_money(sim)
        apply_bonus(sim.rulebook, sim, agent, "park", 0, 0)
        assert gov.balance["ward"] == pytest.approx(before_ward - 1000.0)
        assert SFC.total_money(sim) == pytest.approx(again)


def test_theft_is_an_other_change_in_volume_not_a_transaction(tmp_path):
    """② 窃盗 = **取引ではない**(SNA 2008 §3.98)ので RoW ではなく K5 勘定へ。

    加害者へは 1 円も入らない(= K5 が持っているのは「世界がまだ記録していない受け取り側」)。"""
    sim = _sim(tmp_path, "sfc_k5", n_steps=1, n_agents=15, **dict(ECON, **ON))
    sim.run()
    victim = sim.agents[0]
    victim.money = 5000.0
    base, row0 = SFC.total_money(sim), SFC.row_net(sim)
    cash0 = sum(a.money for a in sim.agents)
    orgs0 = dict(SFC.state_of(sim)["org"])

    victim.money -= 3000.0                      # diversity.py 側の実減額(動力学)
    token = SFC.on_theft(sim, 3000.0)           # 会計側の分類だけ

    assert token == "k5:theft"
    assert SFC.k5_total(sim) == pytest.approx(3000.0)
    assert SFC.state_of(sim)["k5"] == {"theft": pytest.approx(3000.0)}
    assert SFC.total_money(sim) == pytest.approx(base), "K5 込みの総マネーが動いた"
    assert SFC.row_net(sim) == pytest.approx(row0), "窃盗を RoW(取引の相手方)へ落としている"
    assert sum(a.money for a in sim.agents) == pytest.approx(cash0 - 3000.0)
    assert SFC.state_of(sim)["org"] == orgs0, "加害者側(org)へ入金してしまっている"


def test_crime_run_classifies_every_theft_into_k5(tmp_path):
    """②(実ラン): 窃盗が実際に点火し、全件が K5 へ分類され、累計が payload と一致する。"""
    sim = _run(tmp_path, "sfc_crime", n_steps=288, n_agents=40,
               **dict(ECON, **ON, **CRIME))
    thefts = [e for e in _kind(sim, "crime")
              if e.payload.get("kind") == "theft" and e.payload.get("amount", 0.0) > 0.0]
    assert thefts, "窃盗が 1 件も起きていない(点火条件が弱い)"
    assert all(e.payload.get("payee") == "k5:theft" for e in thefts)
    st = SFC.state_of(sim)
    logged = sum(float(e.payload["amount"]) for e in thefts)
    # payload は 0.1 円丸め・会計は無限精度(1 件あたり最大 0.05 円の差)
    assert st["k5"]["theft"] == pytest.approx(logged, abs=0.05 * len(thefts) + 0.1)
    assert "theft" not in st["row"], "K5 が RoW チャネルに混ざっている"


# ===========================================================================
# ⑩ 第99 IF-E2 残③ — K5 の日次 L1 出力(`_emit` の 1 キー)
#    それまで K5 は finance.parquet の k5_other 列にしか出ておらず、L1 だけを読む解析からは
#    総マネー保存の第 3 項が見えなかった。粒度は in_total/out_total と同じ**累積**。
# ===========================================================================
def test_k5_is_published_in_the_daily_l1_row_flow(tmp_path):
    """③ K5 累積が row_flow の 1 キーとして出る。値は finance.parquet の k5_other と同一。"""
    sim = _run(tmp_path, "sfc_k5_l1", n_steps=288, n_agents=40,
               **dict(ECON, **ON, **CRIME))
    flows = _kind(sim, "row_flow")
    assert flows, "L1 に日次の域外収支が出ていない"
    seq = [e.payload["k5_total"] for e in flows]
    assert all(isinstance(v, float) for v in seq)
    # K5 は積むだけ(取り崩す経路が無い)= 単調非減少 = 差分で日次フローが採れる
    assert seq == sorted(seq), f"K5 累積が減っている: {seq}"
    assert seq[-1] > 0.0, "窃盗が起きているのに L1 の K5 が 0 のまま"
    assert seq[-1] == pytest.approx(round(SFC.k5_total(sim), 1))
    # ★同一の式であることの機械固定: サイドカー(6 桁丸め)を L1 の桁(1 桁)へ落とすと一致
    col = pq.read_table(tmp_path / "sfc_k5_l1" / "finance.parquet") \
            .column("k5_other").to_pylist()
    assert len(col) == len(seq), "row_flow とサイドカー行が 1:1 で対応していない"
    assert [round(float(v), 1) for v in col] == seq
    # 総マネー保存の 3 項が **L1 と summary だけ**で閉じる(K5 が L1 に出た目的そのもの)
    prov = SFC.provenance(sim)
    assert prov["k5_total"] == pytest.approx(seq[-1])
    assert sum(prov["k5_kinds"].values()) == pytest.approx(prov["k5_total"], abs=0.05)


def test_k5_key_is_zero_when_nothing_non_transactional_happened(tmp_path):
    """K5 の源(窃盗)が点火していないランでは 0.0 が出る(キーを欠測にしない)。"""
    sim = _run(tmp_path, "sfc_k5_zero", n_steps=288, n_agents=25, **dict(ECON, **ON))
    flows = _kind(sim, "row_flow")
    assert flows
    assert not _kind(sim, "crime"), "テスト前提が崩れた(窃盗が点火している)"
    assert all(e.payload["k5_total"] == 0.0 for e in flows)


def test_k5_l1_key_does_not_exist_when_org_accounting_is_off(tmp_path):
    """OFF ゲート配下: row_flow 自体が出ない = 新キーも 1 つも生えない。"""
    sim = _run(tmp_path, "sfc_k5_off", n_steps=288, n_agents=25, **dict(ECON, **CRIME))
    assert not _kind(sim, "row_flow")
    assert not any("k5_total" in (e.payload or {}) for e in sim.logger.events)


def test_chance_event_is_classified_into_row_channels(tmp_path):
    """③ 偶発の臨時収入 / 紛失 = 外生 = RoW の 2 チャネル(向きが両属のチャネルは作らない)。"""
    sim = _run(tmp_path, "sfc_chance", n_steps=288, n_agents=25,
               **dict(ECON, **ON, **CHANCE))
    ev = _kind(sim, "chance_event")
    wf = [e for e in ev if e.payload["type"] == "windfall"]
    ls = [e for e in ev if e.payload["type"] == "loss"]
    assert wf and ls, "臨時収入/紛失が点火していない"
    assert all(e.payload["payer"] == "row:chance_windfall" for e in wf)
    assert all(e.payload["payee"] == "row:chance_loss" for e in ls)
    row = SFC.state_of(sim)["row"]
    assert row["chance_windfall"]["in"] > 0.0 and row["chance_loss"]["out"] > 0.0
    assert row["chance_windfall"]["out"] == 0.0 and row["chance_loss"]["in"] == 0.0
    assert "chance_windfall" in SFC.CHANNELS_IN and "chance_loss" in SFC.CHANNELS_OUT


def _b2b_sim(tmp_path, name, **ov):
    sim = _sim(tmp_path, name, n_steps=1, n_agents=15, **ov)
    sim.run()
    return sim


def test_b2b_moves_org_deposits_only_when_on(tmp_path):
    """④ b2b = 帳簿だけ → **org 預金間の実移転**。b2b 自身の動力学(在庫・成否)は不変。"""
    kw = dict(ECON, **B2B)
    off = _b2b_sim(tmp_path, "sfc_b2b_off", **kw)
    on = _b2b_sim(tmp_path, "sfc_b2b_on", **dict(kw, **ON))
    assert B2BMOD.enabled(on) and B2BMOD.enabled(off), "b2b が点火していない"

    node = next((n for (n, poi) in sorted(SFC._book_index(on)["node"]) if poi == "shop"),
                None)
    if node is None:
        pytest.skip("台帳に (node, shop) の一意鍵が無い")
    seller = B2BMOD.wholesale_for(on, node, "shop")
    if seller is None:
        pytest.skip("卸 org が居ない台帳")
    for sim in (off, on):                       # 在庫を同じだけ積む(仕入れ成立の条件)
        B2BMOD._state(sim)["stock"][str(seller["id"])] = 100

    bal0 = dict(SFC.state_of(on)["org"])
    row0, total0 = SFC.row_net(on), SFC.total_money(on)
    assert B2BMOD.fulfill(on, node, "shop", 2, 0, 0) is True
    assert B2BMOD.fulfill(off, node, "shop", 2, 0, 0) is True

    p_on = [e for e in on.logger.events if e.kind == "b2b_trade"][-1].payload
    p_off = [e for e in off.logger.events if e.kind == "b2b_trade"][-1].payload
    amt, payer, payee = float(p_on["amount"]), p_on["payer"], p_on["payee"]
    bal1 = SFC.state_of(on)["org"]

    assert payee == str(seller["id"])
    assert SFC.total_money(on) == pytest.approx(total0), "b2b 移転で総マネーが動いた"
    if payer == payee:                          # 自己取引=純額ゼロ(残高を動かさない)
        assert bal1 == bal0
    elif payer.startswith("row:"):              # 買い手の小売 POI を特定できない=域外資本の店
        assert payer == "row:b2b_buyer_unknown"
        assert bal1[payee] == pytest.approx(bal0.get(payee, 0.0) + amt)
        assert SFC.row_net(on) == pytest.approx(row0 - amt)
    else:
        assert bal1[payer] == pytest.approx(bal0.get(payer, 0.0) - amt)
        assert bal1[payee] == pytest.approx(bal0.get(payee, 0.0) + amt)

    # OFF = 現行どおり(相手方キーなし・state 不発生)。b2b の帳簿/在庫は ON/OFF で完全一致
    assert "payer" not in p_off and "payee" not in p_off
    assert getattr(off, "_sfc_state", None) is None
    assert {k: v for k, v in p_on.items() if k not in ("payer", "payee")} == p_off
    assert off._b2b == on._b2b, "org 会計 ON で b2b の在庫/帳簿が動いた(動力学不変の違反)"


def test_uncovered_sites_change_nothing_but_the_relationship_keys(tmp_path):
    """★動力学不変の機械固定: 4 種を点火した ON/OFF で L1 の差分が相手方キーだけに閉じる。"""
    kw = dict(ECON, **CRIME, **CHANCE)
    off = _run(tmp_path, "sfc_unc_off", n_steps=144, n_agents=25, **kw)
    on = _run(tmp_path, "sfc_unc_on", n_steps=144, n_agents=25, **dict(kw, **ON))
    new_kinds = {"org_overdraft", "row_flow"}
    a = list(off.logger.events)
    b = [e for e in on.logger.events if e.kind not in new_kinds]
    assert len(a) == len(b), "新 2 種以外のイベント数が変わった(発火が動いた)"
    allowed = {"payer", "payee", "paid", "revenue"}
    fired: set[str] = set()
    for x, y in zip(a, b):
        assert (x.step, x.agent_id, x.kind) == (y.step, y.agent_id, y.kind)
        extra = set(y.payload or {}) - set(x.payload or {})
        assert extra <= allowed, f"{y.kind}: 宣言外のキーが増えた {sorted(extra)}"
        if y.kind in ("crime", "chance_event"):
            fired.add(y.kind)
        for k in set(x.payload or {}):
            if k in ("balance", "account", "cash"):
                continue
            assert (x.payload or {})[k] == (y.payload or {})[k], f"{y.kind}.{k} が変わった"
    assert fired == {"crime", "chance_event"}, f"点火していない種がある: {fired}"


def test_analyze_maps_the_four_kinds_to_known_sectors_not_void():
    """解析側の追随: トークンが有れば void に落ちない / 無ければ従来どおり(OFF ラン互換)。"""
    # OFF(トークン無し)= 第95バッチまでの分類のまま
    assert aa.flows_for("rule_bonus", {"amount": 100.0}, {})[0].src == aa.VOID
    assert aa.flows_for("crime", {"kind": "theft", "amount": 100.0}, {})[0].dst == aa.VOID
    assert aa.flows_for("b2b_trade", {"amount": 600.0}, {})[0].src == aa.ORG
    # ON(トークンあり)= 既知の部門へ
    f = aa.flows_for("rule_bonus", {"amount": 100.0, "payer": "government"}, {})[0]
    assert (f.src, f.dst) == (aa.GOVERNMENT, aa.HOUSEHOLD)
    f = aa.flows_for("crime", {"kind": "theft", "amount": 100.0,
                               "payee": "k5:theft"}, {})[0]
    assert (f.src, f.dst) == (aa.HOUSEHOLD, aa.OTHER), "窃盗を RoW へ落としている"
    assert aa.party_sector("k5:theft") == aa.OTHER != aa.EXTERNAL
    f = aa.flows_for("chance_event", {"type": "windfall", "amount": 50.0,
                                      "payer": "row:chance_windfall"}, {})[0]
    assert (f.src, f.dst) == (aa.EXTERNAL, aa.HOUSEHOLD)
    f = aa.flows_for("chance_event", {"type": "loss", "amount": 50.0,
                                      "payee": "row:chance_loss"}, {})[0]
    assert (f.src, f.dst) == (aa.HOUSEHOLD, aa.EXTERNAL)
    f = aa.flows_for("b2b_trade", {"amount": 600.0, "payee": "co_x",
                                   "payer": "row:b2b_buyer_unknown"}, {})[0]
    assert (f.src, f.dst) == (aa.EXTERNAL, aa.ORG)
    # 新部門は SECTORS の一級市民(行列・検査表・不可観測理由のすべてに載る)
    assert aa.OTHER in aa.SECTORS and aa.SECTOR_JA[aa.OTHER]
    assert aa.OTHER in aa.UNOBSERVABLE_REASON and aa.OTHER in aa.FINANCE_BALANCE
    assert aa.leak_family("theft:other_volume") == "theft"


@pytest.fixture(scope="module")
def acct_uncovered(tmp_path_factory):
    """4 種のうち**ランで点火できる 2 種**(窃盗・偶発事)を足した mock 小ランの解析。"""
    out = tmp_path_factory.mktemp("sfc_unc")
    dot = ["run.seed=20260807", "run.n_agents=40", "run.n_steps=288",
           "run.name=sfc_unc", "model.backend=mock", "observer.snapshot_every=12"]
    dot += [f"{k}={v}" for k, v in dict(ACCOUNTS, **ON, **CRIME, **CHANCE).items()]
    d = out / "sfc_unc"
    Simulation(load_config(dot), out_dir=d).run()
    return aa.analyze(aa.load_run(d), rel_tol=1e-4)


def test_measured_uncovered_kinds_leave_no_void_flow(acct_uncovered):
    """★受入(実測): 窃盗・偶発事が点火しても漏れ 0・検査①が K5 部門を含めて全部門 PASS。"""
    rep = acct_uncovered
    assert rep["event_counts"].get("crime", 0) > 0, "窃盗が点火していない(検査が空虚)"
    assert rep["event_counts"].get("chance_event", 0) > 0
    assert rep["leak_families"] == [], f"漏れの族が残っている: {rep['leak_families']}"
    assert rep["totals"]["leak_share"] < 0.01
    for sec in (aa.HOUSEHOLD, aa.ORG, aa.BANK, aa.EXTERNAL, aa.OTHER):
        c = rep["checks"][sec]
        assert c.get("observable"), f"{sec} の残高が観測できない: {c.get('reason')}"
        assert c["pass"], f"{sec} の貨幣ストック保存が破れた: {c}"
    assert rep["checks"][aa.OTHER]["close_balance"] > 0.0, "K5 が積まれていない"
    assert rep["classifier_consistency"]["pass"]
    assert rep["unclassified_money_kinds"] == {}
    assert "k5_other" in rep["finance"]["close"], "サイドカーに K5 列が無い"
