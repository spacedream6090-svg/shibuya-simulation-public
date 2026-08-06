"""IF-E2 案B — org の会計主体化 + rest-of-world(渋谷域外)部門(``economy.org_accounting``・**既定 OFF**)。

正典
----
- ``docs/research/ifE2-org-accounting-research.md`` **§4-3 実装への確定ディレクティブ**(6項)と
  **§5 バッチ分解 E2-1〜3**。設計詳細は §3-a〜g、金の経路の完全目録は §3-0。
- ``scripts/analyze_accounting.py`` の docstring(IF-E フェーズ1 の実測 = 漏れ 71.8〜96.0%・
  部門定義 6+1・突合鍵)/ ``tests/test_accounting.py``(既存の検査 39 本)。
- ``docs/plans/if-sv-p4-plan.md`` の IF-E 行。

何を解く問題か
--------------
IF-E(第95バッチ)の実測で、**家計の財布の中では金は消えていない**(相対残差 4.0e-07)が、
**「財布から出た金の受け取り手が世界に存在しない」**ことが判った(漏れ 71.8〜96.0%)。
原因は単純で、org(会社)が**残高そのものを持っていない**ことにある。org_ledger は
``revenue_est``(日給×margin)と ``wage_paid`` を**集計するだけ**で、どの主体の残高とも
接続していない(``revenue_actual`` 列を足しても残高は 1 円も動かない = 案A が原理的に
受入条件を満たせなかった理由)。

本 module は org に**スカラー預金 1 本**を与え、賃金をそこから払い、消費をそこへ入れる。
受け手が街の中に居ない金は **rest-of-world(RoW)= 渋谷域外**という**明示部門**へ
チャネル別に落とす。これで「Σ(全主体残高) + RoW 累積 = 一定」という**閉じた不変量**が立つ。

文献的根拠(要点だけ。詳細は研究文書)
--------------------------------------
- **スカラー預金 1 本で足りる**: Lengnick 2013(LEN)は Dawid & Delli Gatti (2018) の比較表で
  *Stock-flow consistent = Y* に分類され、同章脚注 70 が「労働のみで生産する企業なら、
  流動性 M は賃金支払のみを賄う: M = wN」と明示している。Caiani et al. (2016) の企業が持つ
  金融ストックも預金と借入の 2 つだけで、貸借対照表行列は**事後の検査装置**である。
- **賃金は預金から払う**: Caiani et al. (2016) §3.1 「企業は予想賃金支払額の一定割合 σ を
  予備的動機で預金として保有したい」。初期残高 = σ × 月次賃金支払(σ=1)。
- **支払不能は自動当座借越が最頻**(Poledna / Mark-0 / CATS)。最も洗練された Poledna は
  **AND 条件**(D<0 は許容 = 有利子当座借越。**D<0 かつ E<0 の同時成立でのみ破綻**)。
  ★本 module は**破綻処理を入れない**(残高が負に振れるのを許すだけ)。理由は 2 つ:
  (i) IF-E の実測で ``revenue_est`` は「org 帰属店舗で実際に起きた spend」の **30.9 倍** =
  域内消費は域内賃金を賄うには桁で足りず、倒産機構を既定にすると初日から全社が落ちる。
  (ii) 本選前は「観測する」ことが目的で、倒産の残余処理は新しい漏れを生む。
  **将来の拡張**として Poledna の AND 条件(``deposit<0 and equity<0`` の同時成立)を
  ここに宣言しておく(equity を持たせるのは E2 の範囲外)。
- **RoW は国際統計標準そのもの**: SNA 2008 §26.2 は RoW を「あたかも国内のもう一つの部門で
  あるかのように」記帳するとし、§26.5 は「RoW が受け取った財・サービスを内部で何に使うかは
  記録しない」と明言、§26.6 は RoW 勘定にバランス項目を置かない。= **行動方程式なし・
  残高制約なしの明示部門**。地域 SAM でも "Rest of Economy" は第 4 の制度部門として
  最初から組み込まれている。
- **ただし「何でも吸収する RoW」はゼロ和検査を空虚に通す**。Zezza (2026) OPENSIMPLEST は
  「residual RoW column が会計構造に無いこと」を欠陥として名指しし、Cities: Skylines の
  outside connections は「輸出入が市の予算に計上されない」= 金が閉じていない反面教師である。
  そこで本 module は (a) 吸収を**チャネル別に分類**し (b) **Σ(全主体残高)+RoW 累積 = 一定**を
  テストで固定し (c) RoW 累積を summary / L1 / サイドカーの一級市民として公表する。

R1 ドクトリン
-------------
- 既定 ``economy.org_accounting.enabled: false`` では、本 module の関数はすべて
  **即 return** し、**残高も state も L1 も payload キーも 1 つも生えない**
  (= ゴールデン L1 バイト一致。第95 IF-C / 第96 IF-D と同じ「OFF ではキー自体を作らない」流儀)。
- **乱数 stream を 1 本も引かない**(初期化も受け手解決も完全決定論。台帳は事前計算データ)。
- **generate() の呼び出しサイトを 1 つも作らない**(LLM 追加呼ゼロ = k 完全不変)。
- **no-fingerprint**: org の残高・RoW 依存度・当座借越は**エージェントのプロンプトに一切出さない**
  (観測層だけの概念)。★「自社の資金繰りを経営者エージェントの判断に効かせる」のは
  **将来の拡張**であり、E2 ではやらない(やると ``affects_k`` / ``fingerprint_risk`` の
  再評価が要る)。
- **25 万スケール**: org 残高は ``dict[str, float]``(11,010 エントリ ≒ 1MB 未満)。
  1 取引あたりの追加コストは**静的索引の dict 参照 1 回**= **O(1)/取引**。
  **毎 step の全エージェント走査を 1 つも足さない**(部門行列は部門²=定数)。
- **resume == straight**: 残高群(org / RoW / escrow)は ``sim._sfc_state`` 1 本に集約し、
  ``checkpoint.py`` の ``runtime`` が中央管理する。同時に既存欠陥(``Government`` /
  ``Bank`` / ``VCFund`` が checkpoint に無く resume で残高が初期値へ戻る)も塞ぐ。

受け手解決の設計判断(研究文書 §4-3 ディレクティブ 3 からの**明示的な逸脱**)
-----------------------------------------------------------------------------
ディレクティブ 3 は「スタッフ経由が主」としているが、本実装は**台帳の静的索引だけ**を使う。
理由は 3 つで、いずれも研究文書自身の実測に基づく:
  1. **R1 の計算量規約**。スタッフ経由の解決は「その step に自社 work_node に居る在勤者」の
     索引が要り、**毎 step O(エージェント数) の走査**になる。R1 は「25 万スケール = O(取引数)
     のみ」を要求しており、台帳索引なら **O(1)/取引**(索引は起動時に 1 回だけ構築)。
  2. **スタッフ経由は実測で機能していない**。IF-E の実測で ``serve`` の **220/222 がスタッフ不在**。
     さらに研究文書 §3-c の全数測定で、現行の「node が一意なら org を付ける」規則は
     11,010 社 / 1,008 ノード(平均 10.9 社/node)では**一意に決まるノードが 1 つも無い = 0.0%**。
  3. **台帳索引の方が精度が高い**。同 §3-c の実測で ``(building, floor, POI種別)`` を鍵にすると
     **56.8%(接客業種 58.8%)が一意**、候補数の平均 1.5。
「多義 node は RoW 帰属で正直開示」というディレクティブの**核心部分は完全に守る**
(解決できなかった消費は ``unknown_payee`` チャネルで RoW へ落ち、段別の解決件数を公表する)。

正直な限界(4 件・出力にも残す)
--------------------------------
- **カバーしていない金の経路がある**(``UNCOVERED_KINDS``)。``rule_bonus``(rules.py)・
  ``crime``(diversity.py)・``chance_event``(chance.py)は本バッチの変更範囲外のファイルで
  残高を動かすので、**ON でもそれらが点火するランでは不変量が破れる**。
  一覧と理由は ``summary.org_accounting.uncovered_kinds_declared`` に必ず刻む(ゼロと偽らない)。
  実際に発火したかの検出は**解析側**が担う — 相手方の無い金は
  ``scripts/analyze_accounting.py`` の ``leak_families`` に void フローとして必ず現れる
  (「装置は残したまま値をゼロにする」= 検査の網羅性を将来にわたって守る唯一の方法)。
- **消費税の按分**。内税(価格に含まれる)なので受け手 org には ``実支払 − 消費税`` が入る。
  一方 ``venture``(屋台)は既存実装が売上**全額**を店主へ渡すため、消費税ぶんが世界に
  無いところから出る。この穴は ``tax_gap`` チャネルとして RoW が埋める(**改名して隠さない**)。
- **床クリップの差額**。``_spend`` は残高を 0 未満にしないので、名目より実際の支払が
  少ないことがある。org / RoW が受け手のときは**実支払**を基準に配るので不変量は保たれる
  (「客が払えなかったぶん店の売上が減る」= 現実的だが名目とはズレる)。屋台だけは既存実装が
  名目を店主へ渡すので差額を ``clamp_gap`` チャネルで RoW が埋める。差額が出た spend には
  payload に ``paid``(実支払)が載るので、解析側はそれを使って厳密に突き合わせられる。
- **RoW は残高制約を持たない**(SNA 2008 §26.6 のとおり)。したがって「域外が破産する」ことは
  無い。域外依存度は ``row_in`` の大きさとして**観測するだけ**である。
"""
from __future__ import annotations

import json
from pathlib import Path

from .observer.schema import Event

SCHEMA = 1

# --------------------------------------------------------------------------- #
# RoW のチャネル(§4-1 の要求: 吸収をチャネル別に分類する。「何でも吸収する RoW」を防ぐ)
#
#   direction は「RoW から見た」向きではなく「街から見た」向き:
#     in  … RoW → 街(街に金が入る = 輸出代金・域外本社からの送金・外生ショック)
#     out … 街 → RoW(街から金が出る = 域外への支払・帰属不能な消費)
# --------------------------------------------------------------------------- #
#: RoW → 街(輸出・域外からの入金)
CHANNELS_IN: tuple[str, ...] = (
    "initial_capital",     # 期首: 創業時に域外から持ち込まれた org の初期預金(§3-a)
    "visitor_refill",      # 来街者の財布補充 = **サービスの輸出**(IRTS 2008 §4.21 / SNA §9.80)
    "export_gig",          # 自営の日銭 = 域外クライアントからの入金(街の中に客が居ない)
    "export_production",   # office/education 系 org の輸出代金(§3-c: 全 org の 36.5%・従業者の 51.5%)
    "wage_nonorg",         # 域外/未帰属の雇用主が払う賃金(バイト先・org 未配属者)
    "tax_gap",             # 屋台売上の内税を売り手が留保しない既存挙動の穴埋め(正直開示)
    "clamp_gap",           # 名目 > 実支払(残高床クリップ)の差額(正直開示)
    "shock",               # 外生ショック(D9 の reward 等。ablation 装置であることを明記)
)
#: 街 → RoW(域外への支払・帰属不能)
CHANNELS_OUT: tuple[str, ...] = (
    "unknown_payee",       # 受け手 org を特定できなかった消費 = 域外資本の店(**正直開示**)
    "utilities",           # 固定費(光熱費・サブスク)= 域外のインフラ事業者
    "transport",           # taxi / bus = 域外の運輸事業者
    "rent_landlord",       # 家賃 = 不在家主(域外の不動産所有者)
    "deposit_landlord",    # 転居敷金 = 同上
    "procurement",         # 出店費用 = 域外の内装業者・仕入れ
    "profit_remit",        # 利益送金(将来の拡張。現状は常に 0)
    "fine_no_authority",   # government OFF 時の罰金(徴収主体が街に居ない)
    "seizure",             # 破産の資産圧縮(債権者が街に居ない)
)
CHANNELS: tuple[str, ...] = CHANNELS_IN + CHANNELS_OUT

#: 本 module が残高に**接続した** L1 の金額運搬種(analyze_accounting.MONEY_KINDS の部分集合)。
COVERED_KINDS: frozenset[str] = frozenset({
    "spend", "wage", "withdraw", "rent", "interest_paid", "loan_grant", "loan_repay",
    "venture_sale", "venture_open", "deposit", "candidacy", "reward",
    "civic_service", "bankruptcy", "vc_investment", "enforcement", "tax", "move_home",
    "production",   # ON のときだけ payload に revenue(RoW → org の輸出代金)が載る
})
#: **接続できていない**金額運搬種と、その理由(ゼロと偽らない)。IF-E の監視装置と対で持つ。
#: いずれも本バッチの変更範囲外のファイルで残高を動かすため、ON でも不変量の外側に残る。
UNCOVERED_KINDS: dict[str, str] = {
    "rule_bonus": "rules.py が財源なしで支給する(変更範囲外。行政歳出化は将来)",
    "crime": "diversity.py の窃盗は被害者から減るだけで加害者が受け取らない(変更範囲外)",
    "chance_event": "chance.py の臨時収入/紛失は外生(変更範囲外。RoW 化は将来)",
    "b2b_trade": "b2b.py は帳簿 dict のみで残高を動かさない(不変量には現れない)",
}

# --------------------------------------------------------------------------- #
# 消費カテゴリ → 台帳の POI 種別(受け手解決の鍵の一部)/ 直行 RoW チャネル
# --------------------------------------------------------------------------- #
CAT_TO_POI: dict[str, str] = {
    "food": "food", "cafe": "food", "nightlife": "food",
    "shop": "shop",
    "lodging": "service", "leisure": "service", "medical": "service",
    "service": "service",
}
#: 街の中に受け手が居ないことが**構造的に確定**している消費(索引を引かず直行する)。
CAT_TO_ROW: dict[str, str] = {
    "fixed_cost": "utilities",     # 光熱費・サブスク = 域外のインフラ事業者
    "taxi": "transport", "bus": "transport",
}
#: 台帳で「域内の消費から売上を得られない」POI 種別(= 輸出でしか賄えない。§3-c)
EXPORT_POI_CATS: tuple[str, ...] = ("office", "education")

DEFAULTS = {
    "enabled": False,
    # Caiani et al. (2016): 目標預金 = σ × 予想賃金支払。σ=1(= 月次賃金支払 1 期分)。
    "sigma": 1.0,
    "month_days": 30,          # 「月次賃金支払」を日給の何日ぶんとみなすか(暦は day%30)
    "min_initial": 0.0,        # 初期残高の下限(配属 0 の org を守る。0=下限なし)
    # office/education 系 org の輸出代金(既存 revenue_est の式・発火点は 1 つも変えず、
    # 相手方を void → RoW に付け替えるだけ)。研究文書 §3-c の (ii)。
    "export_production": True,
    "sidecar": True,           # finance.parquet(日次 1 行)を書くか
}

_STEM = "finance"


# --------------------------------------------------------------------------- #
# cfg 正準化(traces.build_cfg / rumors.build_cfg と同型: dict / OmegaConf 両対応)
# --------------------------------------------------------------------------- #
def _to_plain(raw):
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(raw):
            return OmegaConf.to_container(raw, resolve=True)
    except Exception:                                  # noqa: BLE001(旧 config 互換)
        pass
    return raw


def build_cfg(raw) -> dict:
    """conf の ``economy.org_accounting`` ブロックを正準化(既定 OFF=現行挙動と完全同一)。

    dotlist 上書きは文字列で入り得るため型強制する(``traces.build_cfg`` と同じ作法)。"""
    raw = dict(_to_plain(raw) or {})
    cfg = dict(DEFAULTS)
    for k, v in raw.items():
        if k in ("enabled", "export_production", "sidecar"):
            cfg[k] = (v if isinstance(v, bool) else str(v).strip().lower() == "true")
        elif k == "month_days":
            cfg[k] = max(1, int(v))
        elif k in ("sigma", "min_initial"):
            cfg[k] = max(0.0, float(v))
    return cfg


def cfg_of(sim) -> dict:
    """org 会計設定(初回のみ ``sim.cfg.economy.org_accounting`` から遅延構築してキャッシュ)。

    simulation.py を編集せずに済ませる据え付け(``_gov`` / ``traces.cfg_of`` と同型)。
    キャッシュ属性 ``sim.sfccfg`` は L1/L2/L3/乱数に一切現れない = OFF のバイト一致を壊さない。"""
    c = getattr(sim, "sfccfg", None)
    if c is None:
        try:
            raw = (sim.cfg.get("economy", None) or {}).get("org_accounting", None)
        except Exception:                              # noqa: BLE001(旧 config 互換)
            raw = None
        c = build_cfg(raw)
        sim.sfccfg = c
    return c


def enabled(sim) -> bool:
    """org 会計 + RoW 部門が有効か。**既定 OFF = 本 module は 1 バイトも世界に触らない**。"""
    return bool(cfg_of(sim)["enabled"])


# --------------------------------------------------------------------------- #
# 状態(ON 経路でのみ生やす。checkpoint.py が runtime["sfc_state"] で中央管理する)
# --------------------------------------------------------------------------- #
def _state(sim) -> dict:
    st = getattr(sim, "_sfc_state", None)
    if st is None:
        st = {
            "schema": SCHEMA,
            "init": False,              # org 初期残高を配ったか(resume では checkpoint から復元)
            "day": -1,                  # 日次締めの進行
            "org": {},                  # org_id -> 預金残高(スカラー 1 本)
            "row": {},                  # channel -> {"in": …, "out": …}(累積)
            "row_prev": {},             # 前日の累積(日次差分 = row_flow イベント)
            "escrow": 0.0,              # 行政の預り金(議案供託・立候補供託)
            "wage_out": 0.0,            # org が払った賃金の累計
            "revenue_in": 0.0,          # org が受け取った売上の累計
            "export_in": 0.0,           # org が受け取った輸出代金の累計
            "n_overdraft": 0,           # 当座借越に落ちた回数(残高 >=0 → <0 の遷移)
            "n_shortfall": 0,           # 残高不足のまま払った回数(遷移でないものも含む)
            "overdraft_total": 0.0,     # 遷移時の負残高の合計
            "payee": {"floor": 0, "node": 0, "row": 0, "venture": 0},   # 受け手解決の段別件数
        }
        sim._sfc_state = st
    return st


def state_of(sim):
    """ON のときだけ state を返す(OFF は None = checkpoint も summary もキーを作らない)。"""
    return getattr(sim, "_sfc_state", None)


def _row(st: dict, channel: str) -> dict:
    return st["row"].setdefault(str(channel), {"in": 0.0, "out": 0.0})


def row_in(sim, channel: str, amount: float) -> str:
    """RoW → 街(輸出代金・域外からの入金)。戻り値 = payload に載せるトークン。"""
    amt = float(amount)
    if amt:
        _row(_state(sim), channel)["in"] += amt
    return f"row:{channel}"


def row_out(sim, channel: str, amount: float) -> str:
    """街 → RoW(域外への支払・帰属不能)。戻り値 = payload に載せるトークン。"""
    amt = float(amount)
    if amt:
        _row(_state(sim), channel)["out"] += amt
    return f"row:{channel}"


# --------------------------------------------------------------------------- #
# org 残高(スカラー預金 1 本 = Lengnick 2013 の M = wN)
# --------------------------------------------------------------------------- #
def org_balance(sim, org_id) -> float:
    return float(_state(sim)["org"].get(str(org_id), 0.0))


def _is_org(sim, org_id) -> bool:
    """台帳に実在する org か(career の転職先も台帳由来なので book だけで足りる)。"""
    if not org_id:
        return False
    book = getattr(sim, "orgs", None) or {}
    return str(org_id) in book


def credit_org(sim, org_id, amount: float) -> None:
    st = _state(sim)
    key = str(org_id)
    st["org"][key] = st["org"].get(key, 0.0) + float(amount)


def debit_org(sim, org_id, amount: float, step: int, sim_min: int,
              reason: str = "wage") -> None:
    """org の預金から支払う。**不足でも払う(自動当座借越 = 残高が負に振れるのを許す)**。

    Poledna / Mark-0 / CATS で最頻出の支払不能規約。破綻処理は入れない(docstring 冒頭の
    「将来の拡張」宣言を参照)。残高 >=0 から <0 へ落ちた瞬間だけ L1 ``org_overdraft`` を出す
    (毎回出すと L1 が膨れるため。不足のまま払った回数は ``n_shortfall`` に積む)。"""
    st = _state(sim)
    key = str(org_id)
    before = st["org"].get(key, 0.0)
    amt = float(amount)
    after = before - amt
    st["org"][key] = after
    if before < amt:
        st["n_shortfall"] += 1
    if before >= 0.0 > after:
        st["n_overdraft"] += 1
        st["overdraft_total"] += -after
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=-1,
                             kind="org_overdraft", x=0.0, y=0.0,
                             payload={"org": key, "amount": round(amt, 1),
                                      "balance": round(after, 1),
                                      "shortfall": round(-after, 1),
                                      "reason": str(reason)}))


# --------------------------------------------------------------------------- #
# 初期化(§3-a: 初期残高 = σ × 月次賃金支払。乱数ゼロ・決定論)
# --------------------------------------------------------------------------- #
def arm(sim, step: int, sim_min: int) -> None:
    """step 先頭の遅延初期化(OFF は即 return = 完全 no-op)。

    org の初期預金は「**実際に配属された** agent の日給合計 × month_days × σ」で与える
    (Caiani の σ×予想賃金支払の離散版)。台帳の名目従業者数ではなく実配属数を使うので、
    40 体 mock でも 25 万体本番でも自動でスケールする。**乱数を 1 粒も引かない**。
    この初期資本は「創業時に域外から持ち込まれた資本」= RoW → org の期首フローとして
    明示的に記帳する(§3-a)。こうすると t=0 でも不変量が成立する。"""
    if not enabled(sim):
        return
    st = _state(sim)
    if st["init"]:
        return
    st["init"] = True
    book = getattr(sim, "orgs", None) or {}
    if not book:
        return
    cfg = cfg_of(sim)
    daily: dict[str, float] = {}
    for a in sim.agents:                               # id 昇順(sim.agents の生成順)= 決定論
        oid = getattr(a, "org_id", None)
        if not oid or str(oid) not in book:
            continue
        if getattr(a, "org_role", "") == "学生":       # 学生は賃金の受け手でない
            continue
        daily[str(oid)] = daily.get(str(oid), 0.0) + float(getattr(a, "wage", 0.0) or 0.0)
    factor = float(cfg["sigma"]) * int(cfg["month_days"])
    floor = float(cfg["min_initial"])
    total = 0.0
    for oid in sorted(book):                           # 台帳 id 昇順 = 決定論
        bal = max(daily.get(oid, 0.0) * factor, floor)
        if bal <= 0.0:
            continue
        st["org"][oid] = st["org"].get(oid, 0.0) + bal
        total += bal
    if total:
        row_in(sim, "initial_capital", total)


# --------------------------------------------------------------------------- #
# 受け手解決(台帳の**静的索引**だけを使う = O(1)/取引・毎 step 走査ゼロ)
# --------------------------------------------------------------------------- #
def _book_index(sim) -> dict:
    """台帳 → 受け手索引(起動時に 1 回だけ構築してキャッシュ。乱数ゼロ・決定論)。

    ``floor``: (building, floor, POI種別) → **一意なときだけ** org_id(多義は入れない)
    ``node`` : (node, POI種別)            → 同上
    研究文書 §3-c の全数測定: 11,010 社で floor 鍵は 56.8%(接客業種 58.8%)が一意、
    node 鍵は 4.5% しか一意にならない(= floor 鍵が主・node 鍵は indoor OFF の縮退経路)。"""
    idx = getattr(sim, "_sfc_book_idx", None)
    if idx is not None:
        return idx
    book = getattr(sim, "orgs", None) or {}
    by_floor: dict[tuple, list] = {}
    by_node: dict[tuple, list] = {}
    for oid in sorted(book):                           # id 昇順 = 決定論
        wp = book[oid].get("workplace_poi") or {}
        cat = str(wp.get("cat") or "")
        if not cat:
            continue
        bld, flr = wp.get("building"), wp.get("floor")
        if bld:
            by_floor.setdefault((str(bld), int(flr or 0), cat), []).append(oid)
        node = wp.get("node")
        if node:
            by_node.setdefault((str(node), cat), []).append(oid)
    idx = {"floor": {k: v[0] for k, v in by_floor.items() if len(v) == 1},
           "node": {k: v[0] for k, v in by_node.items() if len(v) == 1},
           "export": {oid for oid in book
                      if str((book[oid].get("workplace_poi") or {}).get("cat") or "")
                      in EXPORT_POI_CATS}}
    sim._sfc_book_idx = idx
    return idx


def resolve_payee(sim, agent, cat: str) -> tuple[str | None, str]:
    """消費 ``cat`` の受け手 org を決定論で解決する。戻り値 ``(org_id | None, 段)``。

    段は 3 つ(研究文書 §3-c の c-1 / c-3 / c-4。**c-2 の候補配分は採らない** =
    多義は迷わず RoW へ落として「特定できなかった」ことを正直に開示する):
      ``floor`` … (building, floor, POI種別) が台帳で一意 → その org
      ``node``  … (node, POI種別) が台帳で一意 → その org(indoor OFF ランの縮退経路)
      ``row``   … 決まらない → **RoW(域外資本の店)**。保存を無条件に閉じる最後の受け皿
    """
    poi = CAT_TO_POI.get(str(cat) or "")
    if poi is None and str(cat).startswith("free_"):   # services.py の自由行動(free_*)
        poi = "service"
    if poi is None:
        return None, "row"
    idx = _book_index(sim)
    bld = str(getattr(agent, "building", "") or "")
    if bld:
        oid = idx["floor"].get((bld, int(getattr(agent, "floor", 0) or 0), poi))
        if oid:
            return oid, "floor"
    oid = idx["node"].get((str(getattr(agent, "node", "") or ""), poi))
    if oid:
        return oid, "node"
    return None, "row"


# --------------------------------------------------------------------------- #
# 記帳の口(scheduler / tools はこの 4 つしか呼ばない)
# --------------------------------------------------------------------------- #
def _consumption_tax(sim, nominal: float, cat: str) -> float:
    """内税の消費税額(**読むだけ**・行政の残高は動かさない)。行政 OFF なら 0。"""
    gov = getattr(sim, "government", None)
    if gov is None or not gov.cfg.get("enabled"):
        return 0.0
    national, local, _rate = gov.consumption_tax(float(nominal), str(cat))
    return float(national) + float(local)


def on_spend(sim, agent, nominal: float, actual: float, cat: str,
             step: int, sim_min: int) -> str | None:
    """消費の受け手へ入金する(``_spend`` の唯一の呼び出し点)。戻り値 = payload の ``payee``。

    内税なので受け手が受け取るのは **実支払 − 消費税**(行政は既存経路で消費税を歳入計上済み)。
    家計 −実支払 / 行政 +消費税 / 受け手 +(実支払−消費税) で合計 0 = 不変量が閉じる。"""
    if not enabled(sim):
        return None
    amt = float(actual)
    tax = _consumption_tax(sim, nominal, cat)
    st = _state(sim)
    if str(cat) == "venture":                          # 屋台は既存の保存経路(店主=家計へ全額)
        st["payee"]["venture"] += 1
        # 屋台だけは受け手(店主)が tools 側で**名目**を受け取るので、内税ぶんと
        # 床クリップぶんが世界に無いところから出る。その穴は RoW が埋める(改名して隠さない)。
        gap = float(nominal) - amt
        if gap > 0.0:
            row_in(sim, "clamp_gap", gap)
        if tax:                                        # 売り手が内税を留保しない既存挙動の穴
            row_in(sim, "tax_gap", tax)
        return "venture"
    # ★org / RoW が受け手のときは **実支払** を基準に配るので穴は開かない
    #   (家計 −実支払 / 行政 +消費税 / 受け手 +(実支払−消費税) の合計がちょうど 0)。
    net = amt - tax
    ch = CAT_TO_ROW.get(str(cat))
    if ch is not None:                                 # 構造的に街の外(光熱費・運輸)
        st["payee"]["row"] += 1
        return row_out(sim, ch, net)
    oid, stage = resolve_payee(sim, agent, cat)
    if oid is None:
        st["payee"]["row"] += 1
        return row_out(sim, "unknown_payee", net)
    st["payee"][stage] += 1
    credit_org(sim, oid, net)
    st["revenue_in"] += net
    return oid


def on_wage(sim, agent, gross: float, source: str | None, payer_org,
            step: int, sim_min: int) -> str | None:
    """賃金の支払側を引き落とす(``_pay_wage`` の唯一の呼び出し点)。戻り値 = payload の ``payer``。

    ``payer_org`` は呼び出し元が確定した雇用主(本業の勤務完遂・月給まとめ・退職金)。
    台帳に無い/未配属なら RoW が払う(域外の雇用主 = Eurostat の地域家計所得統計で
    ブリュッセルの域内発生所得の約 62% が域外へ出るのと同型の、公表される通常の地域統計)。"""
    if not enabled(sim):
        return None
    amt = float(gross)
    if amt <= 0.0:
        return None
    if _is_org(sim, payer_org):
        debit_org(sim, payer_org, amt, step, sim_min, reason=str(source or "wage"))
        _state(sim)["wage_out"] += amt
        return str(payer_org)
    src = str(source or "")
    if src == "home_refill":
        return row_in(sim, "visitor_refill", amt)      # 来街者の財布補充 = 輸出
    if src == "gig":
        return row_in(sim, "export_gig", amt)          # 自営の日銭 = 域外クライアント
    return row_in(sim, "wage_nonorg", amt)             # 域外/未帰属の雇用主


def on_production(sim, org, d_rev: float) -> float:
    """勤務完遂 → 産出のたびに、**域内に客が居ない org** へ輸出代金を入れる(§3-c の (ii))。

    既存の ``revenue_est = 日給 × revenue_margin`` の式も発火条件も 1 つも変えず、
    相手方を void → RoW に付け替えるだけ。対象は台帳の POI 種別が office / education の org
    (全 org の 36.5%・従業者の 51.5% = 域内のどの消費カテゴリからも 1 円も受け取れない)。
    接客業種は域内 ``spend`` の受け手なので**二重計上しない**。
    戻り値 = 実際に入金した額(0.0 = 入金なし = ``production`` payload は従来とバイト一致)。"""
    if not enabled(sim) or not cfg_of(sim)["export_production"]:
        return 0.0
    rev = float(d_rev)
    if rev <= 0.0:
        return 0.0
    oid = str(org.get("id"))
    if oid not in _book_index(sim)["export"]:
        return 0.0
    credit_org(sim, oid, rev)
    st = _state(sim)
    st["export_in"] += rev
    row_in(sim, "export_production", rev)
    return rev


def escrow_in(sim, amount: float) -> str | None:
    """供託金の払込(行政の預り金へ)。``Government.balance`` とは別勘定(§3-e)。"""
    if not enabled(sim):
        return None
    _state(sim)["escrow"] += float(amount)
    return "escrow"


def escrow_out(sim, amount: float, *, to_government: bool = False) -> str | None:
    """供託金の返還(``to_government=False``)/ 没収(``True`` = 行政の歳入へ振替)。"""
    if not enabled(sim):
        return None
    _state(sim)["escrow"] -= float(amount)
    return "escrow"


# --------------------------------------------------------------------------- #
# 不変量(§4-1 (b): Σ(全主体残高) + RoW 累積 = 一定。ABCredit.jl 流のスカラー総マネー保存)
# --------------------------------------------------------------------------- #
def city_total(sim) -> float:
    """街の中に**実在する**全残高の合計(家計 + org + 行政 + 供託 + 銀行 + VC)。"""
    total = 0.0
    for a in sim.agents:
        total += float(a.money) + float(getattr(a, "account", 0.0) or 0.0)
    st = state_of(sim)
    if st is not None:
        total += sum(st["org"].values())
        total += float(st["escrow"])
    gov = getattr(sim, "government", None)
    if gov is not None:
        total += sum(gov.balance.values())
    bank = getattr(sim, "bank", None)
    if bank is not None:
        total += float(bank.capital)
    vc = getattr(sim, "vc_fund", None)
    if vc is not None:
        total += float(vc.balance)
    return total


def row_net(sim) -> float:
    """RoW が**正味で吸収した**額(街 → RoW − RoW → 街)。街の外に残高は無い(SNA §26.6)。"""
    st = state_of(sim)
    if st is None:
        return 0.0
    return sum(v["out"] - v["in"] for v in st["row"].values())


def total_money(sim) -> float:
    """**閉じた不変量**: 街の全残高 + RoW 累積。これが一定であることをテストで固定する
    (ABCredit.jl ``test/stock_flow_consistency.jl`` の ``isapprox(init, tot, atol)`` と同型)。"""
    return city_total(sim) + row_net(sim)


# --------------------------------------------------------------------------- #
# 日次締め(L1 ``row_flow`` + finance.parquet サイドカー)
# --------------------------------------------------------------------------- #
def _row_totals(st: dict) -> tuple[float, float]:
    return (sum(v["in"] for v in st["row"].values()),
            sum(v["out"] for v in st["row"].values()))


def _finance_row(sim, st: dict, day: int, step: int) -> tuple:
    orgs = st["org"]
    r_in, r_out = _row_totals(st)
    gov = getattr(sim, "government", None)
    bank = getattr(sim, "bank", None)
    vc = getattr(sim, "vc_fund", None)
    hh = sum(float(a.money) + float(getattr(a, "account", 0.0) or 0.0) for a in sim.agents)
    return (int(day), int(step),
            round(sum(orgs.values()), 6), len(orgs),
            sum(1 for v in orgs.values() if v < 0.0),
            round(min(orgs.values()) if orgs else 0.0, 6),
            round(float(bank.capital) if bank is not None else 0.0, 6),
            round(float(vc.balance) if vc is not None else 0.0, 6),
            round(sum(gov.balance.values()) if gov is not None else 0.0, 6),
            round(float(st["escrow"]), 6),
            round(hh, 6), round(r_in, 6), round(r_out, 6),
            json.dumps({k: {"in": round(v["in"], 3), "out": round(v["out"], 3)}
                        for k, v in sorted(st["row"].items())},
                       ensure_ascii=False, sort_keys=True))


def _emit(sim, st: dict, day: int, step: int, sim_min: int,
          row_step: int | None = None) -> None:
    """L1 ``row_flow``(当日の域外収支)+ サイドカー 1 行。日次 = 高頻度にしない。

    ``row_step`` はサイドカー行に書く step(既定 = L1 の step)。日境界の行は
    ``day_roll`` = **step 先頭**で採られるので「その step の活動より前の残高」を意味し、
    窓 [s0, s1) の排他境界がそのまま成立する。最終行だけは全 step を走り終えた後に採るので
    ``n_steps`` を書く(そうしないと最後の step のフローが窓から落ちる = オフバイワン)。"""
    prev = st.get("row_prev") or {}
    delta = {}
    for ch, v in sorted(st["row"].items()):
        p = prev.get(ch) or {"in": 0.0, "out": 0.0}
        d_in, d_out = v["in"] - p["in"], v["out"] - p["out"]
        if d_in or d_out:
            delta[ch] = {"in": round(d_in, 1), "out": round(d_out, 1)}
    r_in, r_out = _row_totals(st)
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=-1,
                         kind="row_flow", x=0.0, y=0.0,
                         payload={"day": int(day), "channels": delta,
                                  "in_total": round(r_in, 1),
                                  "out_total": round(r_out, 1),
                                  "net": round(r_out - r_in, 1),
                                  "org_balance": round(sum(st["org"].values()), 1)}))
    st["row_prev"] = {k: dict(v) for k, v in st["row"].items()}
    sc = getattr(sim, "finance_sc", None)
    if sc is not None:
        sc.add_rows([_finance_row(sim, st, day,
                                  step if row_step is None else row_step)])


def day_roll(sim, step: int, sim_min: int) -> None:
    """日次境界: 前日の域外収支を締める(``_phase_org_ledger_roll`` と同じ「早期」位置に置く)。

    最初の armed step では**期首の残高アンカー**(day=当日・差分ゼロ)を 1 行出す
    (``Government.daily`` の初回アンカーと同じ理由 = 窓の差分検査の基準点になる)。"""
    if not enabled(sim):
        return
    st = _state(sim)
    day = sim_min // 1440
    prev = st["day"]
    if prev < 0:
        st["day"] = day
        _emit(sim, st, day, step, sim_min)             # 期首アンカー
        return
    if day != prev:
        _emit(sim, st, prev, step, sim_min)
        st["day"] = day


def finalize(sim) -> None:
    """run 終了時: 最終日の域外収支を締める(``finalize_org_day`` と同流儀)。既定 OFF=no-op。"""
    if not enabled(sim):
        return
    st = state_of(sim)
    if st is None or st["day"] < 0:
        return
    n = int(sim.cfg.run.n_steps)
    step = max(0, n - 1)
    _emit(sim, st, st["day"], step, sim.clock.sim_min(step), row_step=n)


# --------------------------------------------------------------------------- #
# summary(既定 OFF = キーそのものが出ない)
# --------------------------------------------------------------------------- #
def provenance(sim) -> dict | None:
    st = state_of(sim)
    if st is None or not enabled(sim):
        return None
    orgs = st["org"]
    r_in, r_out = _row_totals(st)
    return {
        "schema": SCHEMA,
        "n_orgs": len(orgs),
        "org_balance_total": round(sum(orgs.values()), 1),
        "org_balance_min": round(min(orgs.values()), 1) if orgs else 0.0,
        "n_org_negative": sum(1 for v in orgs.values() if v < 0.0),
        "wage_out_total": round(st["wage_out"], 1),
        "revenue_in_total": round(st["revenue_in"], 1),
        "export_in_total": round(st["export_in"], 1),
        "n_overdraft": int(st["n_overdraft"]),
        "n_shortfall": int(st["n_shortfall"]),
        "overdraft_total": round(st["overdraft_total"], 1),
        "payee_stages": dict(st["payee"]),
        "row_in_total": round(r_in, 1),
        "row_out_total": round(r_out, 1),
        "row_net": round(r_out - r_in, 1),
        "row_channels": {k: {"in": round(v["in"], 1), "out": round(v["out"], 1)}
                         for k, v in sorted(st["row"].items())},
        "escrow": round(float(st["escrow"]), 1),
        "city_total": round(city_total(sim), 1),
        "total_money": round(total_money(sim), 1),
        # ★ゼロと偽らない: 接続できていない金の経路(変更範囲外のファイルが残高を動かす)を
        #   ラン結果に必ず刻む。**実際に発火したかの検出は解析側**が行う
        #   (scripts/analyze_accounting.py の leak_families に void フローとして必ず現れる)。
        "uncovered_kinds_declared": dict(sorted(UNCOVERED_KINDS.items())),
    }


# --------------------------------------------------------------------------- #
# finance.parquet(L2 サイドカー・日次 1 行。observer/org_ledger.OrgLedger と同型)
#
# ★記録と動力学の分離(indoor_tracks.py の設計原則③): 動力学は add_rows / flush_segment /
#   finalize しか呼ばず、本クラスの内部バッファ(.rows)を読む向きは存在しない。
# --------------------------------------------------------------------------- #
COLUMNS: tuple[str, ...] = (
    "day", "step", "org_balance", "org_count", "org_negative", "org_min",
    "bank_capital", "vc_balance", "gov_balance", "escrow", "household_balance",
    "row_in", "row_out", "row_channels",
)


class FinanceLedger:
    """部門別残高の日次サイドカー(``finance.parquet``)。検査①を org/bank/RoW/行政で成立させる。"""

    def __init__(self, out_dir):
        self.out_dir = Path(out_dir)
        self.rows: list[tuple] = []
        self._n_flushed = 0
        self._seg = self._next_seg()
        self._resumed = False

    def add_rows(self, rows: list) -> None:
        if rows:
            self.rows.extend(rows)

    def _table(self, rows: list):
        import pyarrow as pa
        return pa.table({
            "day":               pa.array([r[0] for r in rows], pa.int32()),
            "step":              pa.array([r[1] for r in rows], pa.int32()),
            "org_balance":       pa.array([r[2] for r in rows], pa.float64()),
            "org_count":         pa.array([r[3] for r in rows], pa.int32()),
            "org_negative":      pa.array([r[4] for r in rows], pa.int32()),
            "org_min":           pa.array([r[5] for r in rows], pa.float64()),
            "bank_capital":      pa.array([r[6] for r in rows], pa.float64()),
            "vc_balance":        pa.array([r[7] for r in rows], pa.float64()),
            "gov_balance":       pa.array([r[8] for r in rows], pa.float64()),
            "escrow":            pa.array([r[9] for r in rows], pa.float64()),
            "household_balance": pa.array([r[10] for r in rows], pa.float64()),
            "row_in":            pa.array([r[11] for r in rows], pa.float64()),
            "row_out":           pa.array([r[12] for r in rows], pa.float64()),
            "row_channels":      pa.array([r[13] for r in rows], pa.string()),
        })

    def _next_seg(self) -> int:
        mx = -1
        if self.out_dir.is_dir():
            for p in self.out_dir.glob(f"{_STEM}.part-*.parquet"):
                try:
                    mx = max(mx, int(p.name[len(f"{_STEM}.part-"):].split(".")[0]))
                except ValueError:
                    pass
        return mx + 1

    def flush_segment(self) -> None:
        import pyarrow.parquet as pq
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if self.rows:
            pq.write_table(self._table(self.rows),
                           self.out_dir / f"{_STEM}.part-{self._seg:04d}.parquet",
                           compression="zstd")
            self._n_flushed += len(self.rows)
            self.rows = []
        self._seg += 1

    def finalize(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        return self._finalize_stream(self._table(self.rows) if self.rows else None)

    def _finalize_stream(self, table):
        import pyarrow as pa
        import pyarrow.parquet as pq
        parts = sorted(self.out_dir.glob(f"{_STEM}.part-*.parquet"))
        canonical = self.out_dir / f"{_STEM}.parquet"
        if not parts:
            if table is None:
                return None
            pq.write_table(table, canonical, compression="zstd")
            return canonical
        tables = []
        if self._resumed and canonical.exists():
            tables.append(pq.read_table(canonical))
        tables += [pq.read_table(p) for p in parts]
        if table is not None and table.num_rows > 0:
            tables.append(table)
        combined = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
        pq.write_table(combined, canonical, compression="zstd")
        for p in parts:
            p.unlink()
        return canonical
