"""部門別会計行列 + 貨幣保存の検査(IF-E フェーズ1 = 検査。第95バッチ 2026-08-05)。

正典: docs/research/if-lane-research.md §5(Caiani, Godin, Caverzasi, Gallegati, Kinsella &
      Stiglitz 2016, *JEDC* 69:375-408 の検査法2本)/ docs/research/llm-world-interface-audit.md §3
      (経済会計の断絶: revenue_est = 日給×margin で客の spend と非接続)/ docs/plans/if-sv-p4-plan.md IF-E。

Caiani et al. (2016) §5.2 Validation の2検査をそのまま移植する:

  検査① **貨幣ストックの保存** — 「各部門の期首残高 + 流入 − 流出 = 期末残高」。
        本シムで残高が**観測できる**部門は 家計(L3 スナップショットの money+account)と
        行政(public_budget の balance)だけ。残りは残高そのものが世界に無いので
        「不可観測」と正直に開示する(ゼロと偽らない)。
  検査② **取引フロー行列のゼロ和** — 「あらゆる時点で行列のすべての行と列がゼロに合計される」。
        本シムのイベントは**片側記入**(客が払った金の受け取り手が世界に居ない)なので、
        本スクリプトは相手方を分類器で明示的に割り当て、相手が存在しない金を専用部門
        `void`(未接続)へ落とす。**void 行・列の絶対額 = 貨幣の創出/消滅 = 漏れ(leakage)**。

Caiani et al. 本文が「大規模で複雑な AB モデルで実装中の漏れ(leakage)が生じることは珍しくない」と
明言しているとおり、これは本シム固有の不手際ではなく分野で標準的に検査される既知の失敗モードである。

読み取り専用
------------
本スクリプトは runs/<name>/ を読むだけで、シム本体(src/society)を一切呼ばない・変更しない。
乱数を引かず LLM も呼ばない(R1 適合が自明)。依存は 標準ライブラリ + pyarrow のみ(pandas 禁止)。

部門(6+1)
----------
Caiani et al. の行列も**部門別**(家計・消費財企業・資本財企業・銀行・政府・中央銀行)であり、
25万体でエージェント別 N×N 行列は不可能。本シムは以下の 6 部門 + 未接続枠 1 で足りる。

  household  家計   = エージェント(住民+来街者)。現金 money + 口座 account。
  org        企業   = 組織台帳(organizations / org_ledger)。
  venture    個人事業 = 屋台(tools.ventures)。売上は店主(家計)へ抜ける通過部門。
  bank       銀行   = 預金利息・融資・VC ファンド。
  government 行政   = 区(ward)/都(metro)/国(nation)の予算。
  external   街外   = **設計上明示的に**街の外(来街者の財布補充・偶発事の窓口)。
  void       未接続 = 相手方が世界に存在しない金。**これが漏れの正体**。

使い方
------
    python scripts/analyze_accounting.py runs/night_llm_100a3d
    python scripts/analyze_accounting.py runs/foo --out runs/foo/accounting.md --json runs/foo/accounting.json
    python scripts/analyze_accounting.py runs/foo --rel-tol 1e-4      # 検査①の相対許容誤差

実測(2026-08-05 第95バッチ・フェーズ1の結論)
-----------------------------------------------
| ラン | 総フロー | 漏れ比率 | 漏れの族の内訳 |
|---|---:|---:|---|
| mock 40体2日(organizations ON) | 2,504,269 | **96.0 %** | org 56.4 / wage 21.5 / spend 12.8 / venture_setup 5.0 / tax 4.3 |
| 同(accounts+bank ON)          | 1,911,566 | **90.8 %** | org 78.2 / spend 17.0 / wage 2.5 / tax 1.8 / rent 0.4 |
| night_llm_100a3d(実LLM・org OFF)| 1,626,893 | **71.8 %** | spend 74.3 / wage 16.8 / tax 8.9 |

一方で **家計の貨幣ストックは保存している**(相対残差 4.0e-07 / 8.9e-18 / 3.3e-07 = 0.1円丸めの
ノイズのみ)。つまり「個体の財布の中では金は消えていない」が、「財布から出た金の受け取り手が
世界に存在しない」。これが本シムの漏れの正体である。

**フェーズ2(接続)は本バッチでは実装しない。** 判断の根拠は上の実測3点:
  (1) 漏れは `revenue_est` の断絶に**一意に帰着しない**。organizations OFF のランでは org 族は
      0% で、漏れの 74% は「客の払った金を受け取る店舗が会計主体でない」ことに由来する。
  (2) org ON のランでも org 族 56% に対し他4族が 44% を占める(独立した経路が5本ある)。
  (3) 決定的な点として、`revenue_actual` 列を org_ledger に足しても**残高は1円も動かない**
      (org は残高そのものを持たない)。すなわち計画の受入条件「金の保存則が接続後に改善する
      ことをテストで固定」は列の追加では原理的に満たせない。保存を閉じるには org へ残高を
      新設し、支払不能時の規則(倒産/借入)まで決める必要があり、これは設計判断を要する。

**選択肢(数値つき。採否はユーザー判断)**
  - 案A 観測のみ: `org_ledger.revenue_actual` を追加(本スクリプトの突合と同じ規則)。
    保存の改善 = 0 円。乖離(下表)を運用中に可視化できるだけ。工数=小。
  - 案B org を会計主体化: org に残高を新設し、①org 帰属店舗の spend を入金 ②賃金を残高から
    支出 ③枯渇時の規則を決める。閉じる漏れ = org 族 + wage 族 + spend 族の org 帰属分
    (mock org ON ラン換算で漏れの 56.4 + 21.5 + α %)。新状態=checkpoint/resume/golden への
    波及あり。設計判断が必要。
  - 案C 部門宣言: 家主・仕入先・非 org 店舗を **rest-of-world(街外)部門**として設計上明示し、
    「街 + 街外」で保存が閉じると宣言する(Caiani の外国部門と同型)。シム改変ゼロ・
    本スクリプトの写像を変えるだけ。ただし「街の中で経済が回っていない」事実は変わらない。

正直な限界(出力にも印字する)
------------------------------
- **許容誤差はゼロにできない。** L1 payload も L3 スナップショットも 0.1 円単位に丸められており、
  実際の残高更新は丸めない値で行われる(例 interest_paid は payload が小数2桁、加算は無限精度)。
  よって判定は**絶対誤差ではなく相対誤差**(総フローに対する比)で置き、閾値は引数化する
  (if-lane-research.md §5-3 の設計示唆 4)。
- 部門別残高が観測できるのは household と government のみ。bank.capital / venture 在庫 /
  org の内部留保は L1・L3 のどこにも出ないので検査①は「不可観測」と表示する。
- 相手方の割り当ては**本スクリプトの分類器**が行う(世界のイベントは片側記入)。分類器が漏らした
  経路は検査①の残差として必ず現れる。検査①=分類器の網羅性の独立検証、という関係になっている。
- pool.enabled=true(日次 presence 選抜)のランでは在場エージェント集合が入れ替わるため、
  L3 の合計残高が個体の出入りで跳ぶ。この分は `pool_rotation` として分離表示する(漏れではない)。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# --------------------------------------------------------------------------- #
# 部門
# --------------------------------------------------------------------------- #
HOUSEHOLD = "household"
ORG = "org"
VENTURE = "venture"
BANK = "bank"
GOVERNMENT = "government"
EXTERNAL = "external"
VOID = "void"

SECTORS: tuple[str, ...] = (HOUSEHOLD, ORG, VENTURE, BANK, GOVERNMENT, EXTERNAL, VOID)

SECTOR_JA = {
    HOUSEHOLD: "家計", ORG: "企業(org)", VENTURE: "個人事業", BANK: "銀行",
    GOVERNMENT: "行政", EXTERNAL: "街外", VOID: "未接続(漏れ)",
}

#: 残高が観測できない理由(正直開示用)。
UNOBSERVABLE_REASON = {
    ORG: "org_ledger は revenue_est/wage_paid の集計のみで残高列を持たない",
    VENTURE: "屋台は sales_total(累計売上)だけを持ち残高を持たない=通過部門",
    BANK: "Bank.capital / VCFund は L1・L3 のどこにも出力されない",
    EXTERNAL: "街の外=定義上の無限の源/シンク(残高の概念を置いていない)",
    VOID: "相手方が存在しない=残高の器そのものが無い(これが漏れの定義)",
}

#: 金額を運ぶ L1 イベント種(payload に金が乗る種のみ。schema.py の 189 種から抽出)。
MONEY_KINDS: frozenset[str] = frozenset({
    "spend", "wage", "withdraw", "rent", "interest_paid", "loan_grant", "loan_repay",
    "venture_sale", "venture_open", "deposit", "candidacy", "reward", "rule_bonus",
    "civic_service", "chance_event", "crime", "bankruptcy", "vc_investment",
    "enforcement", "tax", "b2b_trade", "move_home",
})

#: 金額は payload に出るが**別のイベントで既に会計済み**の種(二重計上の禁止リスト)。
#: これらは _spend / _pay_wage を経由するので spend / wage 側でだけ数える。
DERIVED_MONEY_KINDS: frozenset[str] = frozenset({
    "service_use",      # services.charge_service → _spend(cat=service)
    "medical_visit",    # scheduler:4278 → _spend(cat="medical")
    "lodging_checkin",  # scheduler:4453 → _spend(cat="lodging")
    "order", "deliver",  # delivery.tick → _spend(注文者) + _pay_wage(配達員 source="gig")
    "free_action",      # scheduler:3711 → _spend(cat=f"free_{category}")
    "ride",             # _charge_ride: ride を記録した直後に _spend(cat=taxi/bus)
    "public_budget",    # 行政の残高観測(フローではない)
    "eviction",         # 立退き: arrears(滞納**残高**)を載せるが金は動かない
})


# --------------------------------------------------------------------------- #
# 1. フロー分類器 — 誰から誰へいくら(検査②の材料)
# --------------------------------------------------------------------------- #
class Flow:
    """部門間フロー1本(src → dst に amount)。tag は漏れの内訳キー。"""

    __slots__ = ("src", "dst", "amount", "tag", "step", "kind")

    def __init__(self, src: str, dst: str, amount: float, tag: str,
                 step: int = -1, kind: str = ""):
        self.src, self.dst, self.amount, self.tag = src, dst, float(amount), tag
        self.step, self.kind = int(step), str(kind)

    def __repr__(self) -> str:                                   # pragma: no cover
        return f"Flow({self.src}->{self.dst}, {self.amount}, {self.tag!r})"


class Move:
    """家計(個体)の残高移動1本。cash=現金 / acct=口座。合計 total が検査①の入力。"""

    __slots__ = ("agent_id", "cash", "acct", "kind")

    def __init__(self, agent_id: int, cash: float = 0.0, acct: float = 0.0,
                 kind: str = ""):
        self.agent_id, self.cash, self.acct, self.kind = int(agent_id), float(cash), float(acct), str(kind)

    @property
    def total(self) -> float:
        return self.cash + self.acct

    def __repr__(self) -> str:                                   # pragma: no cover
        return f"Move({self.agent_id}, cash={self.cash}, acct={self.acct}, {self.kind!r})"


def _f(payload: dict, key: str, default: float = 0.0) -> float:
    try:
        v = payload.get(key, default)
        return float(v) if v is not None else float(default)
    except (TypeError, ValueError):
        return float(default)


def spend_destination(cat: str) -> tuple[str, str]:
    """消費 spend の受け取り部門と漏れタグ。venture だけが世界に受け皿を持つ。

    戻り値 (dst, tag)。dst==VOID は「客が払った金を受け取る主体が世界に居ない」= 漏れ。"""
    c = str(cat or "")
    if c == "venture":
        return VENTURE, "venture_purchase"
    return VOID, f"spend:{c}" if c else "spend:?"


#: wage の source → 支払い部門(と漏れタグ)。
#: gig/salary/severance/None は**どの主体の残高も減らさずに**家計へ入金される=貨幣の創出。
WAGE_SOURCE_SECTOR: dict[str, tuple[str, str]] = {
    "civil":       (GOVERNMENT, "wage:civil"),        # gov.expense(fund_level, gross) で実際に歳出
    "home_refill": (EXTERNAL,   "wage:home_refill"),  # 来街者が街の外の家で財布を補充=設計上の外生
    "gig":         (VOID,       "wage:gig"),          # 自営の日銭。客が居ない
    "salary":      (VOID,       "wage:salary"),       # 月給。org の残高は減らない
    "severance":   (VOID,       "wage:severance"),    # 退職金。org の残高は減らない
}
_WAGE_DEFAULT = (VOID, "wage:work")                   # source なし = 本業/バイトの勤務完遂


def flows_for(kind: str, payload: dict, ctx: dict) -> list[Flow]:
    """1イベント → 部門間フローのリスト(0本以上)。

    ctx は run 全体の文脈: {"government_on": bool, "tax_src": {(step,agent,base_key): sector}}。
    tax の相手方(消費税=販売側 / 源泉税=雇用側)は build() の突合で先に解決しておく。"""
    p = payload or {}
    out: list[Flow] = []
    if kind == "spend":
        amt = _f(p, "amount")
        dst, tag = spend_destination(p.get("cat"))
        if amt:
            out.append(Flow(HOUSEHOLD, dst, amt, tag))
    elif kind == "wage":
        amt = _f(p, "amount")                      # 手取り(行政 ON 時は gross > amount)
        src, tag = WAGE_SOURCE_SECTOR.get(str(p.get("source") or ""), _WAGE_DEFAULT)
        if amt:
            out.append(Flow(src, HOUSEHOLD, amt, tag))
    elif kind == "withdraw":
        pass                                        # 現金⇄口座=家計内部の資産振替(行列に出さない)
    elif kind == "rent":
        paid = _f(p, "paid")
        if paid:
            out.append(Flow(HOUSEHOLD, VOID, paid, "rent:no_landlord"))
    elif kind == "interest_paid":
        amt = _f(p, "amount")
        if amt:
            out.append(Flow(BANK, HOUSEHOLD, amt, "interest"))
    elif kind == "loan_grant":
        amt = _f(p, "amount")
        if amt:
            out.append(Flow(BANK, HOUSEHOLD, amt, "loan_grant"))
    elif kind == "loan_repay":
        amt = _f(p, "amount")
        if amt:
            out.append(Flow(HOUSEHOLD, BANK, amt, "loan_repay"))
    elif kind == "venture_sale":
        amt, div = _f(p, "amount"), _f(p, "dividend")
        if amt - div:
            out.append(Flow(VENTURE, HOUSEHOLD, amt - div, "venture_income"))
        if div:
            out.append(Flow(VENTURE, BANK, div, "vc_dividend"))
    elif kind == "venture_open":
        cost = _f(p, "cost")
        if cost:
            out.append(Flow(HOUSEHOLD, VOID, cost, "venture_setup_cost"))
    elif kind == "move_home":
        dep = _f(p, "deposit")
        if dep:
            out.append(Flow(HOUSEHOLD, VOID, dep, "move_home:deposit_no_landlord"))
    elif kind == "vc_investment":
        amt = _f(p, "amount")
        if amt:
            out.append(Flow(BANK, HOUSEHOLD, amt, "vc_investment"))
    elif kind == "deposit":
        amt, phase = _f(p, "amount"), str(p.get("phase") or "")
        if amt and phase == "paid":                 # 供託=誰の残高でもない預かり金(escrow は未実装)
            out.append(Flow(HOUSEHOLD, VOID, amt, "deposit:escrow_in"))
        elif amt and phase == "refund":
            out.append(Flow(VOID, HOUSEHOLD, amt, "deposit:escrow_out"))
        elif amt and phase == "forfeit":
            out.append(Flow(VOID, GOVERNMENT, amt, "deposit:forfeit"))
        # phase == "insufficient" は金が動かない(拠出不能)
    elif kind == "candidacy":
        dep = _f(p, "deposit")
        if dep:
            out.append(Flow(HOUSEHOLD, VOID, dep, "candidacy:escrow_in"))
    elif kind == "reward":
        amt = _f(p, "amount")
        if amt:
            out.append(Flow(EXTERNAL, HOUSEHOLD, amt, "reward"))   # D9 ablation の外生報酬
    elif kind == "rule_bonus":
        amt = _f(p, "amount")
        if amt:                                     # 区が「発行」するが行政予算は減らない
            out.append(Flow(VOID, HOUSEHOLD, amt, "rule_bonus:unfunded"))
    elif kind == "civic_service":
        amt = _f(p, "amount")
        if amt:
            out.append(Flow(GOVERNMENT, HOUSEHOLD, amt, "civic_benefit"))
    elif kind == "chance_event":
        amt, t = _f(p, "amount"), str(p.get("type") or "")
        if amt and t == "windfall":
            out.append(Flow(EXTERNAL, HOUSEHOLD, amt, "chance:windfall"))
        elif amt and t == "loss":
            out.append(Flow(HOUSEHOLD, EXTERNAL, amt, "chance:loss"))
    elif kind == "crime":
        amt = _f(p, "amount")
        if amt and str(p.get("kind") or "") == "theft":
            out.append(Flow(HOUSEHOLD, VOID, amt, "theft:destroyed"))   # 加害者は受け取らない
    elif kind == "bankruptcy":
        seized = _f(p, "seized")
        if seized:
            out.append(Flow(HOUSEHOLD, VOID, seized, "bankruptcy:seizure"))
    elif kind == "enforcement":
        pen = _f(p, "penalty")
        if pen:
            dst = GOVERNMENT if ctx.get("government_on") else VOID
            out.append(Flow(HOUSEHOLD, dst, pen, "enforcement:fine"))
    elif kind == "tax":
        amt = _f(p, "amount")
        src = ctx.get("tax_src_default", VOID)
        if amt:
            src = ctx.get("tax_src", {}).get(id(payload), src)
            if src != GOVERNMENT:                   # 公務員給与の源泉税は行政の内部振替=行列に出さない
                out.append(Flow(src, GOVERNMENT, amt, f"tax:{p.get('tax')}"))
    elif kind == "b2b_trade":
        amt = _f(p, "amount")
        if amt:
            out.append(Flow(ORG, ORG, amt, "b2b_trade"))           # 部門内(卸→小売)=行/列で相殺
    return out


# --------------------------------------------------------------------------- #
# 2. 家計側の残高移動(検査①の材料)
# --------------------------------------------------------------------------- #
def move_for(kind: str, agent_id: int, payload: dict, accounts_on: bool) -> list[Move]:
    """1イベント → 家計個体の残高移動(現金/口座)。**合計 total が検査①の唯一の入力**。

    accounts ON のとき現金/口座の内訳が payload から一意に決まらない種(venture_open /
    candidacy / bankruptcy)は cash に寄せて記録する。**total は常に正しい**ので
    検査①(=合計残高の保存)は内訳の曖昧さに影響されない。"""
    p = payload or {}
    a = int(agent_id)
    if kind == "spend":
        amt = _f(p, "amount")
        if accounts_on and str(p.get("src") or "") == "card":
            return [Move(a, acct=-amt, kind=kind)]
        return [Move(a, cash=-amt, kind=kind)]
    if kind == "wage":
        amt = _f(p, "amount")                        # 手取り(税は既に控除済み)
        if str(p.get("to") or "") == "account":
            return [Move(a, acct=amt, kind=kind)]
        return [Move(a, cash=amt, kind=kind)]
    if kind == "withdraw":
        amt = _f(p, "amount")
        return [Move(a, cash=amt, acct=-amt, kind=kind)]
    if kind == "rent":
        return [Move(a, acct=-_f(p, "paid"), kind=kind)]
    if kind == "interest_paid":
        return [Move(a, acct=_f(p, "amount"), kind=kind)]
    if kind == "loan_grant":
        return [Move(a, acct=_f(p, "amount"), kind=kind)]
    if kind == "loan_repay":
        return [Move(a, acct=-_f(p, "amount"), kind=kind)]
    if kind == "vc_investment":
        return [Move(a, acct=_f(p, "amount"), kind=kind)]
    if kind == "venture_sale":
        net = _f(p, "amount") - _f(p, "dividend")    # 店主(=イベントの agent_id)の取り分
        return [Move(a, acct=net, kind=kind)] if accounts_on else [Move(a, cash=net, kind=kind)]
    if kind == "venture_open":
        return [Move(a, cash=-_f(p, "cost"), kind=kind)]
    if kind == "candidacy":
        return [Move(a, cash=-_f(p, "deposit"), kind=kind)]
    if kind == "deposit":
        amt, phase = _f(p, "amount"), str(p.get("phase") or "")
        if phase == "paid":
            return [Move(a, cash=-amt, kind=kind)]
        if phase == "refund":
            return [Move(a, cash=amt, kind=kind)]
        return []                                    # forfeit / insufficient は本人の残高を動かさない
    if kind in ("reward", "rule_bonus", "civic_service"):
        return [Move(a, cash=_f(p, "amount"), kind=kind)]
    if kind == "chance_event":
        amt, t = _f(p, "amount"), str(p.get("type") or "")
        if t == "windfall":
            return [Move(a, cash=amt, kind=kind)]
        if t == "loss":
            return [Move(a, cash=-amt, kind=kind)]
        return []
    if kind == "crime":                              # agent_id は加害者。減るのは被害者の現金
        if str(p.get("kind") or "") != "theft":
            return []
        v = p.get("victim")
        return [Move(int(v), cash=-_f(p, "amount"), kind=kind)] if v is not None else []
    if kind == "enforcement":                        # agent_id は警察官。減るのは target の現金
        t = p.get("target")
        return [Move(int(t), cash=-_f(p, "penalty"), kind=kind)] if t is not None else []
    if kind == "bankruptcy":
        return [Move(a, cash=-_f(p, "seized"), kind=kind)]
    if kind == "move_home":
        return [Move(a, cash=-_f(p, "deposit"), kind=kind)]
    return []


#: move_for が家計の残高を動かす種(move_home は L1 種に「金額」列が無いが敷金で現金が減る)。
HOUSEHOLD_KINDS: frozenset[str] = frozenset({
    "spend", "wage", "withdraw", "rent", "interest_paid", "loan_grant", "loan_repay",
    "vc_investment", "venture_sale", "venture_open", "candidacy", "deposit", "reward",
    "rule_bonus", "civic_service", "chance_event", "crime", "enforcement", "bankruptcy",
    "move_home",
})

#: 検査に読み込むイベント種(フロー用 ∪ 家計残高用)。
SCAN_KINDS: frozenset[str] = MONEY_KINDS | HOUSEHOLD_KINDS

#: payload に「金らしい」数値キーがあるか判定するための鍵語(新しい金の経路の検知に使う)。
MONEY_KEYS: tuple[str, ...] = (
    "amount", "cost", "price", "fee", "fare", "deposit", "penalty", "seized",
    "paid", "balance", "salary", "wage", "debt", "revenue", "expense", "fine",
)


def leak_family(tag: str) -> str:
    """漏れタグ → 経路の族(`spend:food` → `spend`)。

    消費カテゴリの増減(`spend:cafe` の新設など)は**新しい金の経路ではない**ので族へ畳む。
    族が増えたときだけ「金の経路が増えた」と判定できる=監視テストの粒度。"""
    return str(tag).split(":", 1)[0]


def unclassified_money_kinds(events: list[dict]) -> dict[str, dict]:
    """**金の経路の追加を検知する装置**。

    payload に金らしい数値キーを持つのに MONEY_KINDS にも DERIVED_MONEY_KINDS にも
    入っていないイベント種を洗い出す。新しい送金経路が実装されると必ずここに現れるので、
    テストで「空である」ことを固定すれば、会計検査の網羅性が構造的に守られる。

    events は L1 の全イベント(kind, payload)。戻り値 {kind: {"n":件数, "keys":[鍵語]}}。"""
    known = MONEY_KINDS | DERIVED_MONEY_KINDS
    out: dict[str, dict] = {}
    for e in events:
        kind = e.get("kind")
        if kind in known:
            continue
        p = e.get("payload") or {}
        if not isinstance(p, dict):
            continue
        hits = sorted(k for k, v in p.items()
                      if k in MONEY_KEYS and isinstance(v, (int, float))
                      and not isinstance(v, bool) and float(v) != 0.0)
        if not hits:
            continue
        rec = out.setdefault(kind, {"n": 0, "keys": set()})
        rec["n"] += 1
        rec["keys"].update(hits)
    return {k: {"n": v["n"], "keys": sorted(v["keys"])} for k, v in sorted(out.items())}


# --------------------------------------------------------------------------- #
# 3. 行列の組み立て
# --------------------------------------------------------------------------- #
def build_matrix(flows: list[Flow]) -> dict[tuple[str, str], float]:
    """フロー列 → 部門×部門の金額行列 {(src, dst): amount}。"""
    m: dict[tuple[str, str], float] = defaultdict(float)
    for fl in flows:
        m[(fl.src, fl.dst)] += fl.amount
    return dict(m)


def row_col_sums(matrix: dict[tuple[str, str], float]) -> dict[str, dict[str, float]]:
    """各部門の 流出(row)/ 流入(col)/ 純額(col-row)。

    Caiani の「行と列がゼロに合計される」は本スクリプトでは
    「**各部門の純額 = その部門の残高変化**」として検査①で判定する。
    行列そのものは二重記入で作っているので全部門の純額の総和は構造的に 0(自明)。"""
    out = {s: {"outflow": 0.0, "inflow": 0.0, "net": 0.0} for s in SECTORS}
    for (src, dst), amt in matrix.items():
        out[src]["outflow"] += amt
        out[dst]["inflow"] += amt
    for s in SECTORS:
        out[s]["net"] = out[s]["inflow"] - out[s]["outflow"]
    return out


def leak_breakdown(flows: list[Flow]) -> dict[str, dict[str, float]]:
    """void を相手にしたフローの内訳 {tag: {"created":…, "destroyed":…, "count":…}}。

    created  = void → 実在部門(何もないところから貨幣が湧いた)
    destroyed = 実在部門 → void(貨幣が消えた)"""
    acc: dict[str, dict[str, float]] = defaultdict(
        lambda: {"created": 0.0, "destroyed": 0.0, "count": 0.0})
    for fl in flows:
        if fl.src == VOID and fl.dst == VOID:
            continue
        if fl.src == VOID:
            acc[fl.tag]["created"] += fl.amount
            acc[fl.tag]["count"] += 1
        elif fl.dst == VOID:
            acc[fl.tag]["destroyed"] += fl.amount
            acc[fl.tag]["count"] += 1
    return {k: dict(v) for k, v in acc.items()}


# --------------------------------------------------------------------------- #
# 4. 検査① — 貨幣ストックの保存
# --------------------------------------------------------------------------- #
def conserve_household(stock_by_step: dict[int, dict[int, float]],
                       moves_by_step: dict[int, list[Move]],
                       rel_tol: float) -> dict:
    """家計の検査①: L3 スナップショット間で Δ(money+account) = Σ move.total か。

    stock_by_step: {step: {agent_id: money+account}}(L3 由来)
    moves_by_step: {step: [Move, ...]}(L1 由来。**その step の処理で動いた分**)

    L1 の step s のイベントは step s のスナップショット(step 末に取る)に**反映済み**なので、
    窓 (s0, s1] の流量は step が s0 < step <= s1 のイベント。"""
    steps = sorted(stock_by_step)
    windows = []
    tot_abs_flow = 0.0
    tot_resid = 0.0
    for s0, s1 in zip(steps, steps[1:]):
        a0, a1 = stock_by_step[s0], stock_by_step[s1]
        common = a0.keys() & a1.keys()
        d_common = sum(a1[i] - a0[i] for i in common)
        rot_in = sum(a1[i] for i in a1.keys() - a0.keys())     # 途中参加(pool)
        rot_out = sum(a0[i] for i in a0.keys() - a1.keys())    # 途中退出(pool)
        flow = 0.0
        by_kind: dict[str, float] = defaultdict(float)
        for st in range(s0 + 1, s1 + 1):
            for mv in moves_by_step.get(st, ()):
                if mv.agent_id in common:
                    flow += mv.total
                    by_kind[mv.kind] += mv.total
        resid = d_common - flow
        tot_abs_flow += sum(abs(v) for v in by_kind.values())
        tot_resid += resid
        windows.append({
            "from_step": s0, "to_step": s1, "n_common": len(common),
            "delta_stock": round(d_common, 3), "flow": round(flow, 3),
            "residual": round(resid, 3),
            "pool_in": round(rot_in, 3), "pool_out": round(rot_out, 3),
            "kinds": sorted(by_kind),
        })
    rel = abs(tot_resid) / tot_abs_flow if tot_abs_flow > 0 else 0.0
    return {
        "sector": HOUSEHOLD, "observable": True, "n_windows": len(windows),
        "total_abs_flow": round(tot_abs_flow, 3),
        "total_residual": round(tot_resid, 3),
        "relative_residual": rel,
        "pass": rel <= rel_tol,
        "rel_tol": rel_tol,
        "windows": windows,
    }


def conserve_government(budget_rows: list[dict], flows: list[Flow],
                        rel_tol: float) -> dict:
    """行政の検査①。2段構えで見る。

    (a) **内部整合**: public_budget の (revenue, expense, balance) が level ごとに
        「前回残高 + 歳入 − 歳出 = 今回残高」を満たすか(行政モジュールの自己申告の検算)。
    (b) **外部整合(本命)**: 行政の残高変化が、**本スクリプトが L1 から独立に組んだ
        フロー行列の行政純額**と一致するか。(a) が通っても (b) が破れていれば、
        行政が世界の他部門と繋がっていない金を動かしていることになる。

    注意(正直な限界): public_budget は日境界フェーズの先頭で「前日ぶん」を締めるので、
    締め step のうち締めより後に走るフェーズ(公務員ペイロール等)は次の記録に入る。
    (b) は締め step を排他境界(step < S)として集計するが、同一 step 内のフェーズ順の
    ずれは残差として現れうる。判定はこの粒度での近似である。"""
    per_level: dict[str, list[dict]] = defaultdict(list)
    for r in budget_rows:
        per_level[str(r.get("level"))].append(r)
    levels: dict[str, list[dict]] = {}
    tot_abs, tot_res = 0.0, 0.0
    for lv in sorted(per_level):
        rows = sorted(per_level[lv], key=lambda r: int(r.get("step", 0)))
        recs, prev = [], None
        for r in rows:
            rev, exp, bal = _f(r, "revenue"), _f(r, "expense"), _f(r, "balance")
            if prev is not None:
                resid = bal - (prev + rev - exp)
                recs.append({"step": r.get("step"), "residual": round(resid, 3),
                             "revenue": rev, "expense": exp, "balance": bal})
                tot_abs += abs(rev) + abs(exp)
                tot_res += resid
            prev = bal
        levels[lv] = recs
    rel = abs(tot_res) / tot_abs if tot_abs > 0 else 0.0

    steps = sorted({int(r.get("step", 0)) for r in budget_rows})
    ext: dict = {"applicable": False}
    if len(steps) >= 2:
        s_first, s_last = steps[0], steps[-1]
        bal0 = sum(_f(r, "balance") for r in budget_rows if int(r.get("step", 0)) == s_first)
        bal1 = sum(_f(r, "balance") for r in budget_rows if int(r.get("step", 0)) == s_last)
        inflow = sum(f.amount for f in flows
                     if f.dst == GOVERNMENT and f.src != GOVERNMENT and s_first <= f.step < s_last)
        outflow = sum(f.amount for f in flows
                      if f.src == GOVERNMENT and f.dst != GOVERNMENT and s_first <= f.step < s_last)
        resid = (bal1 - bal0) - (inflow - outflow)
        denom = inflow + outflow
        ext = {"applicable": True, "from_step": s_first, "to_step": s_last,
               "open_balance": round(bal0, 1), "close_balance": round(bal1, 1),
               "matrix_inflow": round(inflow, 1), "matrix_outflow": round(outflow, 1),
               "residual": round(resid, 1),
               "relative_residual": (abs(resid) / denom) if denom else 0.0}
        ext["pass"] = ext["relative_residual"] <= max(rel_tol, 1e-3)

    return {"sector": GOVERNMENT, "observable": True,
            "total_abs_flow": round(tot_abs, 3), "total_residual": round(tot_res, 3),
            "relative_residual": rel, "pass": rel <= rel_tol, "rel_tol": rel_tol,
            "levels": levels, "external": ext}


def classifier_consistency(flows: list[Flow], moves: list[Move]) -> dict:
    """分類器の自己整合: フロー行列の**家計純額**と、家計個体の残高移動の合計が一致するか。

    flows_for()(検査②側)と move_for()(検査①側)は別々に書かれた2つの写像なので、
    片方に経路の書き漏れがあれば必ずここで差が出る。**分類器のバグ検出器**であって
    シムのバグ検出器ではない(シム側の漏れは void 列に出る)。"""
    net_matrix = (sum(f.amount for f in flows if f.dst == HOUSEHOLD and f.src != HOUSEHOLD)
                  - sum(f.amount for f in flows if f.src == HOUSEHOLD and f.dst != HOUSEHOLD))
    net_moves = sum(m.total for m in moves)
    diff = net_matrix - net_moves
    scale = max(abs(net_matrix), abs(net_moves), 1.0)
    return {"matrix_net": round(net_matrix, 3), "moves_net": round(net_moves, 3),
            "diff": round(diff, 3), "relative": abs(diff) / scale,
            "pass": abs(diff) <= 1e-6 * scale + 1e-6}


# --------------------------------------------------------------------------- #
# 5. revenue_est 乖離(既知の断絶を別枠で明示)
# --------------------------------------------------------------------------- #
UNKNOWN_ORG = "__unknown__"


def match_serve_to_spend(serve_rows: list[dict], spend_rows: list[dict]) -> dict:
    """serve イベントを、それを生んだ客の spend へ厳密に突き合わせる。

    serve は scheduler が「客の spend と同一 work_node の勤務中スタッフ」へ機械的に帰属させた
    イベントなので、対応する spend は必ず存在する。突き合わせ鍵は2通り:

      - **スタッフ有り**: payload に `customer` が入る → (step, customer, cat) で一意。
      - **スタッフ不在**(`unstaffed`): payload に customer が**入らない**(agent_id=-1)。
        座標も使えない — serve フェーズは `_phase_jitter`(路上の微移動)の**後**に走るので
        serve の (x, y) は spend 時点の座標と一致しない(実測で確認)。そこで
        **同 step・同 cat の spend を L1 記録順に消費する**。serve は spend を記録順に走査して
        発火するので、この対応は順序で一意に決まる。

    同一 spend に複数 serve が立ちうる(max_serve_per_event>1)ので、spend 側を使用済みに
    してから加算する(二重計上の禁止)。org_id が無い/多義 node は `__unknown__` へ落として
    正直に別立てする(第58 の掟=多義 node は unknown と開示)。"""
    by_key: dict[tuple[int, int, str], list[int]] = defaultdict(list)
    queue: dict[tuple[int, str], list[int]] = defaultdict(list)
    for i, sp in enumerate(spend_rows):
        st, cat = int(sp["step"]), str(sp["cat"])
        by_key[(st, int(sp["agent_id"]), cat)].append(i)
        queue[(st, cat)].append(i)

    used: set[int] = set()
    by_org: dict[str, float] = defaultdict(float)
    n_matched = n_unmatched = n_dup = 0
    n_unstaffed = 0
    for s in serve_rows:
        p = s.get("payload") or {}
        st, cat = int(s["step"]), str(p.get("cat") or "")
        if p.get("unstaffed"):
            n_unstaffed += 1
        cust = p.get("customer")
        cand = by_key.get((st, int(cust), cat), []) if cust is not None else queue.get((st, cat), [])
        pick = next((i for i in cand if i not in used), None)
        if pick is None:
            if cand:
                n_dup += 1            # 同一 spend への2本目以降の serve(二重計上しない)
            else:
                n_unmatched += 1
            continue
        used.add(pick)
        n_matched += 1
        oid = p.get("org_id")
        by_org[str(oid) if oid else UNKNOWN_ORG] += float(spend_rows[pick]["amount"])
    return {"by_org": dict(by_org), "n_matched": n_matched, "n_unmatched": n_unmatched,
            "n_dup_serve": n_dup, "n_serve": len(serve_rows), "n_unstaffed": n_unstaffed}


def revenue_gap(ledger_rows: list[dict], serve_rows: list[dict],
                spend_rows: list[dict]) -> dict:
    """org_ledger.revenue_est の合計 vs org 帰属店舗で実際に発生した spend の合計。

    ledger_rows: org_ledger.parquet の行 [{day, org_id, revenue_est, wage_paid, serve_count, ...}]
    serve_rows : L1 の serve イベント / spend_rows: L1 の spend(step, agent_id, x, y, cat, amount)。"""
    est_by_org: dict[str, float] = defaultdict(float)
    wage_by_org: dict[str, float] = defaultdict(float)
    for r in ledger_rows:
        est_by_org[str(r.get("org_id"))] += _f(r, "revenue_est")
        wage_by_org[str(r.get("org_id"))] += _f(r, "wage_paid")

    mt = match_serve_to_spend(serve_rows, spend_rows)
    actual_by_org = mt["by_org"]

    orgs = sorted(set(est_by_org) | set(actual_by_org))
    rows = []
    for oid in orgs:
        est, act = est_by_org.get(oid, 0.0), actual_by_org.get(oid, 0.0)
        rows.append({"org_id": oid, "revenue_est": round(est, 1),
                     "spend_actual": round(act, 1), "gap": round(est - act, 1),
                     "ratio": (est / act) if act else None,
                     "wage_paid": round(wage_by_org.get(oid, 0.0), 1)})
    tot_est = sum(est_by_org.values())
    tot_act = sum(actual_by_org.values())
    known_act = tot_act - actual_by_org.get(UNKNOWN_ORG, 0.0)
    return {
        "rows": rows,
        "total_revenue_est": round(tot_est, 1),
        "total_spend_actual": round(tot_act, 1),
        "known_org_spend": round(known_act, 1),
        "gap": round(tot_est - known_act, 1),
        "ratio": (tot_est / known_act) if known_act else None,
        "unknown_org_spend": round(actual_by_org.get(UNKNOWN_ORG, 0.0), 1),
        "n_serve": mt["n_serve"], "n_serve_matched": mt["n_matched"],
        "n_serve_unstaffed": mt["n_unstaffed"],
        "n_serve_without_spend": mt["n_unmatched"],
        "n_serve_dedup_dropped": mt["n_dup_serve"],
        "total_wage_paid": round(sum(wage_by_org.values()), 1),
    }


# --------------------------------------------------------------------------- #
# 6. ラン読み込み
# --------------------------------------------------------------------------- #
def _read_config(run_dir: Path) -> dict:
    """runs/<name>/config.yaml から検査に必要なトグルだけを取り出す(omegaconf 依存を避ける)。"""
    path = run_dir / "config.yaml"
    out = {"accounts_on": False, "government_on": False, "pool_on": False,
           "org_ledger_on": False, "indoor_fields_on": False}
    if not path.exists():
        return out
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    # 素朴なインデント追跡(依存を増やさないための最小パーサ。キーの重複は経路で解決)。
    stack: list[tuple[int, str]] = []
    for raw in text:
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip())
        line = raw.strip()
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key, val = key.strip(), val.split("#")[0].strip()
        while stack and stack[-1][0] >= indent:
            stack.pop()
        path_key = ".".join([k for _, k in stack] + [key])
        stack.append((indent, key))
        if val in ("true", "false"):
            b = (val == "true")
            if path_key == "economy.accounts.enabled":
                out["accounts_on"] = b
            elif path_key == "government.enabled":
                out["government_on"] = b
            elif path_key == "agents.pool.enabled" or path_key == "pool.enabled":
                out["pool_on"] = b
            elif path_key == "work.service.ledger.enabled":
                out["org_ledger_on"] = b
            elif path_key == "work.service.indoor_fields":
                out["indoor_fields_on"] = b
    return out


def load_run(run_dir: Path) -> dict:
    """ラン1本を読み、検査に必要な素材だけをメモリに載せる(payload の JSON パースは対象種のみ)。"""
    import pyarrow.parquet as pq

    run_dir = Path(run_dir)
    cfg = _read_config(run_dir)

    events: list[dict] = []
    serve_rows: list[dict] = []
    budget_rows: list[dict] = []
    other_money: list[dict] = []       # 未分類の金の経路の検知用(安価な部分文字列で事前ふるい)
    needles = tuple(f'"{k}"' for k in MONEY_KEYS)
    want = SCAN_KINDS | {"serve", "public_budget"}
    pf = pq.ParquetFile(run_dir / "l1_events.parquet")
    for batch in pf.iter_batches(batch_size=200_000,
                                 columns=["step", "agent_id", "kind", "x", "y", "payload"]):
        d = batch.to_pydict()
        for st, aid, kind, x, y, pl in zip(d["step"], d["agent_id"], d["kind"],
                                           d["x"], d["y"], d["payload"]):
            if kind not in want:
                if kind not in DERIVED_MONEY_KINDS and pl and any(n in pl for n in needles):
                    try:
                        other_money.append({"kind": kind, "payload": json.loads(pl)})
                    except (TypeError, ValueError):
                        pass
                continue
            try:
                p = json.loads(pl) if pl else {}
            except (TypeError, ValueError):
                p = {}
            rec = {"step": int(st), "agent_id": int(aid), "kind": kind, "payload": p,
                   "x": float(x or 0.0), "y": float(y or 0.0)}
            if kind == "serve":
                serve_rows.append(rec)
            elif kind == "public_budget":
                budget_rows.append({"step": int(st), **p})
            else:
                events.append(rec)

    stock_by_step: dict[int, dict[int, float]] = {}
    snap_path = run_dir / "l3_snapshots.parquet"
    if snap_path.exists():
        t = pq.read_table(snap_path)
        for st, state in zip(t.column("step").to_pylist(), t.column("state").to_pylist()):
            try:
                agents = json.loads(state).get("agents", [])
            except (TypeError, ValueError):
                continue
            stock_by_step[int(st)] = {
                int(a["id"]): float(a.get("money", 0.0)) + float(a.get("account", 0.0) or 0.0)
                for a in agents}

    ledger_rows: list[dict] = []
    led_path = run_dir / "org_ledger.parquet"
    if led_path.exists():
        ledger_rows = pq.read_table(led_path).to_pylist()

    return {"cfg": cfg, "events": events, "serve": serve_rows,
            "budget": budget_rows, "stock": stock_by_step, "ledger": ledger_rows,
            "other_money": other_money, "run_dir": str(run_dir)}


# --------------------------------------------------------------------------- #
# 7. 解析本体
# --------------------------------------------------------------------------- #
def resolve_tax_sources(events: list[dict]) -> dict[int, str]:
    """tax イベントの支払い元部門を突き合わせで解決する。

    - consumption: 同 (step, agent) の spend で base==amount のものの**受け取り部門**が納税者
      (内税=価格に含まれる。客はもう払っており、remit するのは販売側)。
    - income/resident: 同 (step, agent) の wage で gross==base のものの**支払い部門**。
      公務員給与(source=civil)なら行政の内部振替なのでフローを出さない(GOVERNMENT を返す)。

    戻り値は id(payload) → 部門。payload dict の同一性で引く(build 内で1回だけ使う)。"""
    spend_ix: dict[tuple[int, int], list[dict]] = defaultdict(list)
    wage_ix: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for e in events:
        if e["kind"] == "spend":
            spend_ix[(e["step"], e["agent_id"])].append(e["payload"])
        elif e["kind"] == "wage":
            wage_ix[(e["step"], e["agent_id"])].append(e["payload"])
    out: dict[int, str] = {}
    for e in events:
        if e["kind"] != "tax":
            continue
        p = e["payload"]
        base = _f(p, "base")
        which = str(p.get("tax") or "")
        key = (e["step"], e["agent_id"])
        if which == "consumption":
            for sp in spend_ix.get(key, ()):
                if abs(_f(sp, "amount") - base) < 0.051:
                    out[id(p)] = spend_destination(sp.get("cat"))[0]
                    break
        elif which in ("income", "resident"):
            for wg in wage_ix.get(key, ()):
                if abs(_f(wg, "gross") - base) < 0.051:
                    out[id(p)] = WAGE_SOURCE_SECTOR.get(
                        str(wg.get("source") or ""), _WAGE_DEFAULT)[0]
                    break
    return out


def analyze(run: dict, rel_tol: float = 1e-6) -> dict:
    """ラン素材 → 検査①②+revenue_est 乖離のレポート辞書。"""
    events, cfg = run["events"], run["cfg"]
    ctx = {"government_on": cfg["government_on"],
           "tax_src": resolve_tax_sources(events),
           "tax_src_default": VOID}

    flows: list[Flow] = []
    flows_by_day: dict[int, list[Flow]] = defaultdict(list)
    moves_by_step: dict[int, list[Move]] = defaultdict(list)
    kind_counts: dict[str, int] = defaultdict(int)
    for e in events:
        kind, p, st = e["kind"], e["payload"], e["step"]
        kind_counts[kind] += 1
        for fl in flows_for(kind, p, ctx):
            fl.step, fl.kind = st, kind
            flows.append(fl)
            flows_by_day[st // 144].append(fl)
        if kind in HOUSEHOLD_KINDS:
            moves_by_step[st].extend(move_for(kind, e["agent_id"], p, cfg["accounts_on"]))

    # 企業(org)部門: org_ledger の revenue_est / wage_paid は**どの主体の残高とも接続していない**。
    # 行列に明示的に載せることで、org 行・列のどれだけが未接続かが1目で見える。
    for r in run["ledger"]:
        day = int(r.get("day", 0) or 0)
        est, wage = _f(r, "revenue_est"), _f(r, "wage_paid")
        if est:
            fl = Flow(VOID, ORG, est, "org:revenue_est_unfunded", day * 144, "org_ledger")
            flows.append(fl)
            flows_by_day[day].append(fl)
        if wage:
            fl = Flow(ORG, VOID, wage, "org:wage_paid_unfunded", day * 144, "org_ledger")
            flows.append(fl)
            flows_by_day[day].append(fl)

    matrix = build_matrix(flows)
    sums = row_col_sums(matrix)
    leaks = leak_breakdown(flows)

    daily = []
    for day in sorted(flows_by_day):
        m = build_matrix(flows_by_day[day])
        s = row_col_sums(m)
        lk = leak_breakdown(flows_by_day[day])
        daily.append({
            "day": day,
            "total_flow": round(sum(m.values()), 1),
            "created": round(sum(v["created"] for v in lk.values()), 1),
            "destroyed": round(sum(v["destroyed"] for v in lk.values()), 1),
            "net": {k: round(v["net"], 1) for k, v in s.items()},
        })

    checks = {}
    if run["stock"]:
        checks[HOUSEHOLD] = conserve_household(run["stock"], moves_by_step, rel_tol)
    else:
        checks[HOUSEHOLD] = {"sector": HOUSEHOLD, "observable": False,
                             "reason": "l3_snapshots.parquet が無い"}
    if run["budget"]:
        checks[GOVERNMENT] = conserve_government(run["budget"], flows, rel_tol)
    else:
        checks[GOVERNMENT] = {"sector": GOVERNMENT, "observable": False,
                              "reason": "public_budget イベントが 0 件(government OFF)"}
    for s in (ORG, VENTURE, BANK, EXTERNAL, VOID):
        checks[s] = {"sector": s, "observable": False, "reason": UNOBSERVABLE_REASON[s]}

    spend_rows = [{"step": e["step"], "agent_id": e["agent_id"], "x": e["x"], "y": e["y"],
                   "cat": str(e["payload"].get("cat") or ""), "amount": _f(e["payload"], "amount")}
                  for e in events if e["kind"] == "spend"]
    gap = revenue_gap(run["ledger"], run["serve"], spend_rows)

    total_flow = sum(matrix.values())
    created = sum(v["created"] for v in leaks.values())
    destroyed = sum(v["destroyed"] for v in leaks.values())
    return {
        "run_dir": run["run_dir"], "config": cfg, "rel_tol": rel_tol,
        "event_counts": dict(sorted(kind_counts.items())),
        "matrix": {f"{a}->{b}": round(v, 1) for (a, b), v in sorted(matrix.items())},
        "sector_sums": {k: {kk: round(vv, 1) for kk, vv in v.items()} for k, v in sums.items()},
        "leaks": {k: {"created": round(v["created"], 1),
                      "destroyed": round(v["destroyed"], 1),
                      "count": int(v["count"])} for k, v in sorted(leaks.items())},
        "totals": {"total_flow": round(total_flow, 1),
                   "money_created": round(created, 1),
                   "money_destroyed": round(destroyed, 1),
                   "leak_share": (created + destroyed) / total_flow if total_flow else 0.0},
        "daily": daily,
        "checks": checks,
        "leak_families": sorted({leak_family(t) for t in leaks}),
        "unclassified_money_kinds": unclassified_money_kinds(run.get("other_money", [])),
        "classifier_consistency": classifier_consistency(
            flows, [m for mv in moves_by_step.values() for m in mv]),
        "revenue_gap": gap,
    }


# --------------------------------------------------------------------------- #
# 8. Markdown 出力
# --------------------------------------------------------------------------- #
def _num(v: float) -> str:
    return f"{v:,.1f}"


def render_markdown(rep: dict) -> str:
    L: list[str] = []
    cfg = rep["config"]
    L.append("# 部門別会計行列と貨幣保存の検査(IF-E フェーズ1)")
    L.append("")
    L.append(f"- ラン: `{rep['run_dir']}`")
    L.append(f"- トグル: accounts={cfg['accounts_on']} / government={cfg['government_on']} / "
             f"org_ledger={cfg['org_ledger_on']} / serve.indoor_fields={cfg['indoor_fields_on']}")
    L.append(f"- 相対許容誤差 rel_tol = {rep['rel_tol']:g}")
    L.append("- 検査法の出典: Caiani et al. (2016) *JEDC* 69:375-408 §5.2 / Godley & Lavoie (2007)")
    L.append("")

    L.append("## 0. 総額")
    t = rep["totals"]
    L.append("")
    L.append("| 量 | 金額 |")
    L.append("|---|---:|")
    L.append(f"| 総フロー(部門間の記帳合計) | {_num(t['total_flow'])} |")
    L.append(f"| **貨幣の創出**(void → 実在部門) | {_num(t['money_created'])} |")
    L.append(f"| **貨幣の消滅**(実在部門 → void) | {_num(t['money_destroyed'])} |")
    L.append(f"| 漏れ比率 (創出+消滅)/総フロー | {t['leak_share']*100:.2f} % |")
    L.append("")

    L.append("## 1. 検査① 貨幣ストックの保存(部門別)")
    L.append("")
    L.append("> 「各部門の期首残高 + 流入 − 流出 = 期末残高」。残高が世界に実在し観測できる部門だけが判定可能。")
    L.append("")
    L.append("| 部門 | 残高観測 | 総流量 | 残差 | 相対残差 | 判定 |")
    L.append("|---|---|---:|---:|---:|---|")
    for s in SECTORS:
        c = rep["checks"][s]
        if not c.get("observable"):
            L.append(f"| {SECTOR_JA[s]} | ✗ 不可 | — | — | — | {c.get('reason','')} |")
            continue
        L.append(f"| {SECTOR_JA[s]} | ✓ | {_num(c['total_abs_flow'])} | "
                 f"{_num(c['total_residual'])} | {c['relative_residual']:.3e} | "
                 f"{'PASS' if c['pass'] else 'FAIL'} |")
    L.append("")

    hh = rep["checks"][HOUSEHOLD]
    if hh.get("observable"):
        bad = [w for w in hh["windows"] if abs(w["residual"]) > 0.5]
        L.append(f"家計の窓数 = {hh['n_windows']}(L3 スナップショット間隔)。"
                 f"残差 |r|>0.5 円の窓 = {len(bad)} 個。")
        if bad:
            L.append("")
            L.append("| 窓(step) | 対象人数 | Δ残高 | Σフロー | 残差 | pool 入 | pool 出 | 窓内のイベント種 |")
            L.append("|---|---:|---:|---:|---:|---:|---:|---|")
            for w in bad[:25]:
                L.append(f"| {w['from_step']}→{w['to_step']} | {w['n_common']} | "
                         f"{_num(w['delta_stock'])} | {_num(w['flow'])} | {_num(w['residual'])} | "
                         f"{_num(w['pool_in'])} | {_num(w['pool_out'])} | {', '.join(w['kinds'])} |")
            if len(bad) > 25:
                L.append(f"| … | | | | | | | 残り {len(bad)-25} 窓は JSON 側 |")
        L.append("")

    gv = rep["checks"][GOVERNMENT]
    if gv.get("observable"):
        n_bad = sum(1 for lv in gv["levels"].values() for r in lv if abs(r["residual"]) > 0.5)
        L.append(f"行政(a) 内部整合: public_budget の日次記録 "
                 f"{sum(len(v) for v in gv['levels'].values())} 本のうち |残差|>0.5 は {n_bad} 本。")
        ext = gv.get("external") or {}
        if ext.get("applicable"):
            L.append("")
            L.append(f"行政(b) 外部整合(step {ext['from_step']}→{ext['to_step']}): "
                     f"期首 {_num(ext['open_balance'])} + 流入 {_num(ext['matrix_inflow'])} "
                     f"− 流出 {_num(ext['matrix_outflow'])} vs 期末 {_num(ext['close_balance'])} "
                     f"→ 残差 {_num(ext['residual'])}(相対 {ext['relative_residual']:.3e}・"
                     f"{'PASS' if ext['pass'] else 'FAIL'})")
        L.append("")

    cc = rep["classifier_consistency"]
    L.append(f"分類器の自己整合(検査②の家計純額 {_num(cc['matrix_net'])} vs 検査①の残高移動合計 "
             f"{_num(cc['moves_net'])}): 差 {_num(cc['diff'])} → "
             f"{'PASS' if cc['pass'] else 'FAIL(分類器のどちらかに経路の書き漏れ)'}")
    L.append("")

    L.append("## 2. 検査② 部門別フロー行列(行=支払い側 / 列=受け取り側)")
    L.append("")
    L.append("> Caiani の「行と列がゼロに合計される」。本スクリプトは相手方を分類器で明示割り当てするので")
    L.append("> 行列自体は二重記入で構造的にゼロ和になる。**実質のゼロ和違反は `void` 行・列の絶対額**")
    L.append("> = 相手方が世界に存在しない金 = 貨幣の創出/消滅。")
    L.append("")
    hdr = "| from \\ to | " + " | ".join(SECTOR_JA[s] for s in SECTORS) + " | 流出計 |"
    L.append(hdr)
    L.append("|---|" + "---:|" * (len(SECTORS) + 1))
    m = {tuple(k.split("->")): v for k, v in rep["matrix"].items()}
    for a in SECTORS:
        cells = []
        for b in SECTORS:
            v = m.get((a, b), 0.0)
            cells.append(_num(v) if v else "·")
        L.append(f"| **{SECTOR_JA[a]}** | " + " | ".join(cells) + " | "
                 + _num(rep["sector_sums"][a]["outflow"]) + " |")
    L.append("| **流入計** | " + " | ".join(_num(rep["sector_sums"][b]["inflow"])
                                            for b in SECTORS) + " | |")
    L.append("")
    L.append("| 部門 | 流入 | 流出 | 純額(流入−流出) |")
    L.append("|---|---:|---:|---:|")
    for s in SECTORS:
        v = rep["sector_sums"][s]
        L.append(f"| {SECTOR_JA[s]} | {_num(v['inflow'])} | {_num(v['outflow'])} | {_num(v['net'])} |")
    L.append("")

    L.append("## 3. 漏れの内訳(発生イベント種別)")
    L.append("")
    L.append("| タグ | 件数 | 貨幣の創出 | 貨幣の消滅 |")
    L.append("|---|---:|---:|---:|")
    for tag, v in sorted(rep["leaks"].items(),
                         key=lambda kv: -(kv[1]["created"] + kv[1]["destroyed"])):
        L.append(f"| `{tag}` | {v['count']} | {_num(v['created'])} | {_num(v['destroyed'])} |")
    L.append("")
    L.append(f"漏れの経路の族: {', '.join('`%s`' % f for f in rep['leak_families'])}")
    L.append("")
    unc = rep["unclassified_money_kinds"]
    if unc:
        L.append("**未分類の金額キーを持つイベント種**(= 会計検査が見ていない金の経路の候補):")
        L.append("")
        L.append("| kind | 件数 | 金額らしい鍵 |")
        L.append("|---|---:|---|")
        for k, v in unc.items():
            L.append(f"| `{k}` | {v['n']} | {', '.join(v['keys'])} |")
    else:
        L.append("未分類の金額キーを持つイベント種: **なし**(既知の金の経路で閉じている)。")
    L.append("")

    L.append("## 4. 既知の断絶: revenue_est vs 実際の spend")
    L.append("")
    g = rep["revenue_gap"]
    L.append("> `revenue_est = 日給 × revenue_margin`(scheduler.py `_log_org_output`)は")
    L.append("> **客が払った金と一切接続していない**。org 帰属店舗で実際に発生した spend と比較する。")
    L.append("")
    L.append("| 量 | 金額 |")
    L.append("|---|---:|")
    L.append(f"| org_ledger.revenue_est 合計 | {_num(g['total_revenue_est'])} |")
    L.append(f"| serve で spend へ突き合った金額の合計 | {_num(g['total_spend_actual'])} |")
    L.append(f"| うち **org が特定できた**分(乖離の分母) | {_num(g['known_org_spend'])} |")
    L.append(f"| うち org 不明(スタッフ不在×多義 node) | {_num(g['unknown_org_spend'])} |")
    L.append(f"| **乖離**(revenue_est − org特定 spend) | {_num(g['gap'])} |")
    L.append(f"| 比 est/actual | {('%.2f' % g['ratio']) if g['ratio'] else '—'} |")
    L.append(f"| org_ledger.wage_paid 合計 | {_num(g['total_wage_paid'])} |")
    L.append(f"| serve 件数 / うちスタッフ不在 | {g['n_serve']} / {g['n_serve_unstaffed']} |")
    L.append(f"| serve→spend 突合 成功 / 失敗 / 重複除去 | "
             f"{g['n_serve_matched']} / {g['n_serve_without_spend']} / {g['n_serve_dedup_dropped']} |")
    L.append("")
    if g["rows"]:
        L.append("| org_id | revenue_est | spend 実測 | 乖離 | wage_paid |")
        L.append("|---|---:|---:|---:|---:|")
        for r in sorted(g["rows"], key=lambda r: -abs(r["gap"]))[:30]:
            L.append(f"| `{r['org_id']}` | {_num(r['revenue_est'])} | {_num(r['spend_actual'])} | "
                     f"{_num(r['gap'])} | {_num(r['wage_paid'])} |")
        L.append("")
    else:
        L.append("*(org_ledger.parquet が無い = work.service.ledger.enabled=false。乖離表は空。)*")
        L.append("")

    L.append("## 5. 日次推移")
    L.append("")
    L.append("| 日 | 総フロー | 創出 | 消滅 | 家計 純額 | 行政 純額 |")
    L.append("|---:|---:|---:|---:|---:|---:|")
    for d in rep["daily"]:
        L.append(f"| {d['day']} | {_num(d['total_flow'])} | {_num(d['created'])} | "
                 f"{_num(d['destroyed'])} | {_num(d['net'][HOUSEHOLD])} | "
                 f"{_num(d['net'][GOVERNMENT])} |")
    L.append("")
    L.append("---")
    L.append("")
    L.append("*本スクリプトは読み取り専用(シム本体ゼロタッチ・乱数0・LLM呼0)。*")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# 9. CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir", help="runs/<name>")
    ap.add_argument("--out", help="Markdown の出力先(省略時は標準出力)")
    ap.add_argument("--json", dest="json_out", help="JSON の出力先")
    ap.add_argument("--rel-tol", type=float, default=1e-6,
                    help="検査①の相対許容誤差(既定 1e-6。丸めがあるので絶対ゼロは不成立)")
    args = ap.parse_args(argv)

    run = load_run(Path(args.run_dir))
    rep = analyze(run, rel_tol=args.rel_tol)
    md = render_markdown(rep)
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        print(md)
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(rep, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote {args.json_out}")
    hh = rep["checks"][HOUSEHOLD]
    return 0 if (not hh.get("observable") or hh.get("pass")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
