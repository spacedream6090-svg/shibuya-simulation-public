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

第98バッチ IF-E2 UNCOVERED — 残り 4 種の接続(``UNCOVERED_KINDS`` が空になった)
------------------------------------------------------------------------------
IF-E2 案B の時点では ``rule_bonus`` / ``crime`` / ``chance_event`` / ``b2b_trade`` の 4 種が
「変更範囲外のファイルが残高を動かす」ため不変量の外側に残っていた(**それらが点火するランでは
ON でも不変量が破れる**と正直に宣言していた)。本バッチでその 4 本を閉じる。**いずれも ON のときだけ・
金額もタイミングも既存の動力学を 1 つも変えない(分類の追加だけ)**:

- ``rule_bonus``(rules.py)= **行政歳出**。制度DSL の bonus は「区が発行する」建て付けなので、
  区(ward)予算からの移転として記帳する。**区の残高が負に振れるのは許容**(``debit_org`` の
  自動当座借越と同じ思想 = 払えないことを隠さずに記録する)。
  ★``Government.expense`` ではなく残高を直に動かす。理由は ``on_rule_bonus`` の docstring。
- ``crime``(diversity.py の窃盗)= **非取引**。SNA 2008 §3.98(逐語):
  *"If thefts, or acts of violence (including war), involve significant redistributions, or
  destructions, of assets, it is necessary to take them into account. As explained below,
  **they are treated as other flows, not as transactions**."* §3.96 が「相互合意という取引の特徴を
  満たす違法行為は合法行為と同じに扱う」と定める裏返しで、**窃盗は相互合意が無いので取引ではない**。
  したがって RoW(=取引の相手方)へ落としてはならず、**「その他の資産変動勘定」**(SNA 2008
  第12章 *The other changes in assets accounts*・K.5 = other changes in volume n.e.c.)に相当する
  専用バケツ ``K5``(``K5_KINDS``)へ分類する。総マネー保存は
  **Σ(全主体残高) + RoW 累積 + K5 累積 = 一定**へ拡張される。
  ★**加害者が受け取る動力学変更はしない**(現行実装は被害者から減るだけで加害者は受け取らない)。
    SNA では窃盗は「被害者の資産 −・加害者の資産 +」の再分配なので、本バケツが持っているのは
    **世界がまだ記録していない受け取り側**である。加害者へ入金するかは**将来の判断**として
    ここに宣言しておく(やると加害者の消費余力が増える = 挙動変化を伴うので独立トグルが要る)。
- ``chance_event``(chance.py の臨時収入 / 紛失)= **外生 = RoW**。拾得・還付・当選小口も財布の
  紛失も街の外との資金移動なので、``chance_windfall``(RoW → 街)/ ``chance_loss``(街 → RoW)の
  2 チャネルへ分類する(向きが両属のチャネルは作らない = ``CHANNELS_IN``/``OUT`` の規約)。
- ``b2b_trade``(b2b.py の卸→小売)= **org 預金間の実移転**。従来は帳簿 dict(``sim._b2b``)だけが
  動き残高は 1 円も動かなかった。ON では買い手 org の預金 → 売り手 org の預金へ実際に移す
  (= 案B の「範囲内の取引は個人と企業・組織を観察できるようにする」の完成)。買い手の小売 POI は
  ``(node, POI種別)`` の台帳索引で解決し、決まらなければ**域外資本の店**とみなして
  ``b2b_buyer_unknown``(RoW → 街)から支払う(``unknown_payee`` の裏返し・正直開示)。
  **b2b の動力学(在庫・仕入れ成否・トリップ)は 1 バイトも変えない**。

``UNCOVERED_KINDS`` は**空 dict のまま残す**(``summary`` の ``uncovered_kinds_declared`` も同様)。
「装置は残したまま値をゼロにする」= 新しい金の経路が実装された瞬間に
``tests/test_org_accounting.py`` の網羅テスト(``COVERED_KINDS ∪ UNCOVERED_KINDS ⊇
analyze_accounting.MONEY_KINDS``)が落ちる = 会計検査の網羅性を将来にわたって守る唯一の方法。

正直な限界(出力にも残す)
--------------------------
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

from .observer.finalize import FinalizeStreamMixin
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
    "chance_windfall",     # 偶発の臨時収入(拾得・還付・当選小口)= 外生(chance.py)
    "b2b_buyer_unknown",   # 買い手の小売 POI を台帳で特定できなかった仕入れ = 域外資本の店
    "subsidy_no_authority",  # 行政が未構築の世界での rule_bonus(支給主体が街に居ない)
    # H2 医療: 保険給付の 7 割は**保険者(協会けんぽ・国保)から医療機関へ**入る。保険者は
    # 街の中に居ないので新部門を作らず、**名前のある RoW チャネル**として正直に開示する
    # (計画書 §6-4 でユーザー承認済み。tax_gap / clamp_gap と同じ流儀)。
    "insurance_reimbursement",
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
    "chance_loss",         # 偶発の紛失・急な出費 = 外生の流出(chance.py)
    # H2 医療: 救急搬送の公費。★支出主体は区(本シムの government)だが、**救急を運行するのは
    # 東京消防庁 = 都**であって街の中に運行主体の org は 1 つも存在しない(計画書 §7 の
    # 「管轄混同」への回答)。したがって受け手は街の外 = 本チャネルへ落とす。
    "ems_operation",
    # 所有権レイヤー O3(相続。src/society/assets.py): **相続人が 1 人も居ない**死者の遺産は
    # 民法 959 条により国庫へ帰属する。国は街の中に居ない(区でも都でもない)ので、
    # 新部門を作らず**名前のある RoW チャネル**として正直に開示する(insurance_reimbursement /
    # ems_operation と同じ流儀)。★これが無いと「死者の財布」に凍結されていた金が
    # 相続を入れた瞬間に世界から消え、閉じた不変量 city + RoW + K5 が破れる。
    "inheritance_escheat",
)
CHANNELS: tuple[str, ...] = CHANNELS_IN + CHANNELS_OUT

#: **その他の資産変動**(SNA 2008 第12章 *The other changes in assets accounts* / K.5 =
#: other changes in volume n.e.c.)へ落とす非取引の種。RoW(= 取引の相手方)とは**別勘定**。
#: §3.98 が「窃盗は取引ではなく other flows」と定めるので、ここに来るものは**取引ではない**。
K5_KINDS: tuple[str, ...] = (
    "theft",               # 窃盗: 被害者から減るが加害者は受け取らない(受け取り側が未記録)
    "lost_property",       # 遺失: 誰にも拾われないまま失われた現金(受け取り手が世界に居ない)
)

#: 本 module が残高に**接続した** L1 の金額運搬種(analyze_accounting.MONEY_KINDS の部分集合)。
COVERED_KINDS: frozenset[str] = frozenset({
    "spend", "wage", "withdraw", "rent", "interest_paid", "loan_grant", "loan_repay",
    "venture_sale", "venture_open", "deposit", "candidacy", "reward",
    "civic_service", "bankruptcy", "vc_investment", "enforcement", "tax", "move_home",
    "production",   # ON のときだけ payload に revenue(RoW → org の輸出代金)が載る
    # ---- 第98バッチ IF-E2 UNCOVERED(module docstring の該当節を参照)----
    "rule_bonus",    # 区(ward)予算からの移転
    "crime",         # 窃盗 = 非取引 → K5(その他の資産変動)
    "chance_event",  # 臨時収入/紛失 = 外生 → RoW の chance_windfall / chance_loss
    "b2b_trade",     # 卸→小売 = org 預金間の実移転(買い手不明なら RoW から)
    # ---- H2 医療(身体と事件のレイヤー。既定 OFF)----
    "ems_transport",  # 救急搬送の公費 = 区(ward)の歳出 → RoW(ems_operation)
    "medical_bill",   # 保険給付 7 割 = RoW(insurance_reimbursement)→ 医療機関 org
    # ---- H3 遺失物(第109バッチ D2 で解析側の分類器へ接続。既定 OFF)----
    # 4 種とも ``on_lost_hold`` / ``on_lost_release`` / ``on_lost_lapse`` で本 module の
    # ``lost`` バケツを経由済み(= 残高に接続されている)。街の外(RoW)は 1 度も通らない。
    "lost_drop",      # 落下 = 財布の中の現金が所持金から遺失物バケツへ(``on_lost_hold``)
    "lost_return",    # 返還 = 遺失物バケツ → 落とし主 + 報労金(落とし主 → 拾得者)
    "lost_keep",      # 着服 = 遺失物バケツ → 拾得者(受け手が特定できるので K5 ではない)
    "lost_expire",    # 時効取得 = バケツ → 拾得者 / 失効 = K5(``on_lost_lapse``)
})
#: **接続できていない**金額運搬種と、その理由(ゼロと偽らない)。IF-E の監視装置と対で持つ。
#: ★第98バッチで**空になった**。辞書と ``summary.org_accounting.uncovered_kinds_declared`` は
#: 空のまま**残す**(装置は残したまま値をゼロにする)。新しい金の経路が
#: ``analyze_accounting.MONEY_KINDS`` へ足された瞬間に網羅テストが落ち、ここか
#: ``COVERED_KINDS`` への追加を強制する = 会計検査の網羅性が構造的に守られる。
UNCOVERED_KINDS: dict[str, str] = {}

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
    # ---- レーン乙 D1: 初期預金の母集団(既定 False = 現行と完全同一)----------------
    #  False = day0 の**在場者だけ**の日給合計(pool ON では 100 万人中 25 万人 = 1/4 の
    #          さらに一部しか数えず、org の期首預金が構造的に過少 → 初日から当座借越)。
    #  True  = **束ねられた名簿全体**(attach_record 宇宙 = record に org_id を持つ全ペルソナ)の
    #          日給合計。台帳と wage_plan だけで決まる純関数なので在場に依存しない
    #          (= 世界のアルゴリズムが org の体力を決めない)。pool OFF では
    #          sim.agents が名簿そのものなので True でも False でも同じ値になる。
    "seed_from_pool_roster": False,
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
        if k in ("enabled", "export_production", "sidecar", "seed_from_pool_roster"):
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
            "k5": {},                   # 非取引の資産変動(SNA K.5)。kind -> 累積額(第98)
            "bonus_out": 0.0,           # 区予算から出した制度DSL bonus の累計(第98)
            "b2b_transfer": 0.0,        # org 預金間で実移転した b2b 仕入れの累計(第98)
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
# K5 = その他の資産変動(SNA 2008 第12章。**取引ではない**フローの受け皿)
#
#   RoW と分けるのが要点: RoW は「取引の相手方が街の外に居る」ことを表す部門で、
#   K5 は「そもそも取引ではない(相互合意が無い)」ことを表す**勘定**である(SNA §3.96/§3.98)。
#   混ぜると「窃盗は輸入だった」という嘘になる。
# --------------------------------------------------------------------------- #
def _k5(st: dict) -> dict:
    # 旧 checkpoint(第97バッチ)からの resume でもキーが生える(setdefault = 互換の作法)
    return st.setdefault("k5", {})


def k5_out(sim, kind: str, amount: float) -> str:
    """街の残高から**取引でない**理由で消えた額を K5 へ分類する。戻り値 = payload のトークン。"""
    amt = float(amount)
    if amt:
        k = _k5(_state(sim))
        k[str(kind)] = k.get(str(kind), 0.0) + amt
    return f"k5:{kind}"


def k5_total(sim) -> float:
    st = state_of(sim)
    if st is None:
        return 0.0
    return sum((st.get("k5") or {}).values())


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
    daily = (_daily_wage_by_org_from_roster(sim, book)
             if (cfg["seed_from_pool_roster"] and getattr(sim, "_pool", None) is not None)
             else None)
    if daily is None:
        daily = {}
        for a in sim.agents:                           # id 昇順(sim.agents の生成順)= 決定論
            oid = getattr(a, "org_id", None)
            if not oid or str(oid) not in book:
                continue
            if getattr(a, "org_role", "") == "学生":   # 学生は賃金の受け手でない
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


def _daily_wage_by_org_from_roster(sim, book: dict) -> dict:
    """org → 日給合計を**名簿全体**(pool の全 record)から組む(レーン乙 D1)。決定論・乱数ゼロ。

    ★なぜ在場者ではいけないか: 初期預金は「創業時にその会社が用意した運転資金」であって、
      **たまたま day0 に街に居た人数**で決まってよいものではない(ユーザー原理:
      世界のアルゴリズムがエージェント量を決めない)。pool ON では在場 cap が 1/4 なので、
      現状は期首から構造的に過少 = 初日の給与支払いで当座借越に落ちる org が出る。
    ★母集団 = ``organizations.attach_record`` が配属を認める record(= ``org_id`` が台帳に
      在る)そのもの。日給は ``economy.wage_plan``(record の occupation/role/org と
      **pool の pid** だけで決まる純関数)で、在場個体に付く ``agent.wage`` と**同じ値**に
      なる(``assign_wage_plan`` が使うキーが ``agent.pool_pid`` = record["id"] だから)。
    ★走査は ``PoolStore._build_index`` と同じシャードのストリーム読み 1 周(全 record を
      RAM に載せない)。step 0 の先頭で 1 度だけ。"""
    from . import economy as _econ
    pool = sim._pool
    wcfg = (getattr(sim, "economy", None) or {}).get("wage_profile") or {}
    wage_on = bool(wcfg.get("enabled"))
    econ = getattr(sim, "economy", None) or {}
    daily: dict[str, float] = {}
    for sh in pool.meta.get("shards", []):
        with open(pool.dir / sh["file"], "rb") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                rec = json.loads(line)
                oid = rec.get("org_id")
                if not oid or str(oid) not in book:
                    continue
                role = str(rec.get("role", "") or "")
                if role == "学生":                     # 学生は賃金の受け手でない(現行と同判定)
                    continue
                occ = str(rec.get("occupation", "") or "")
                if wage_on:
                    if not _econ.wage_eligible(occ, role):
                        continue
                    w = _econ.wage_plan(occupation=occ, role=role,
                                        org=book[str(oid)], org_id=oid,
                                        key=str(rec["id"]), cfg=wcfg)["daily"]
                else:
                    w = _econ.wage_amount(occ, True, econ)
                daily[str(oid)] = daily.get(str(oid), 0.0) + float(w or 0.0)
    return daily


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


def resolve_payee_at_node(sim, node: str, cat: str) -> str | None:
    """**場所だけ**から受け手 org を解決する(b2b の買い手 = 小売 POI。客の個体が居ない経路)。

    b2b の仕入れは ``(node, cat)`` の小売 POI が買い手で、``agent.building`` / ``floor`` が
    存在しない(客ではなく店が買う)。したがって使える鍵は ``(node, POI種別)`` だけ =
    ``resolve_payee`` の縮退経路と同じ精度(研究文書 §3-c の実測で一意率 4.5%)。
    決まらなければ None(= 呼び出し側が RoW = 域外資本の店として扱う)。"""
    poi = CAT_TO_POI.get(str(cat) or "")
    if poi is None:
        return None
    return _book_index(sim)["node"].get((str(node), poi))


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
             step: int, sim_min: int, payee_node: str | None = None) -> str | None:
    """消費の受け手へ入金する(``_spend`` の唯一の呼び出し点)。戻り値 = payload の ``payee``。

    内税なので受け手が受け取るのは **実支払 − 消費税**(行政は既存経路で消費税を歳入計上済み)。
    家計 −実支払 / 行政 +消費税 / 受け手 +(実支払−消費税) で合計 0 = 不変量が閉じる。

    ★``payee_node``(H2 医療。既定 None = 従来と 1 バイトも変わらない): 受け手が
      **払った人の居場所では決まらない**消費のための口。医療がその唯一の例で、中等症の受診は
      自宅や路上で発火するため ``resolve_payee``(building/floor → node)がどうしても当たらず、
      医療費が黙って ``unknown_payee`` へ漏れていた(IF-E の監査発見)。受診先・搬送先を
      知っているのは医療側なので、そこから場所を渡してもらって ``resolve_payee_at_node`` で
      引く(b2b の買い手解決と**同じ鍵**= 精度も同じ = 決まらなければ従来どおり RoW)。"""
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
    if payee_node:                                     # 場所が渡された消費(医療)
        oid, stage = resolve_payee_at_node(sim, str(payee_node), cat), "node"
    else:
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
# 遺失物の中の現金(H3 ``lost_property``。**既定 OFF の機構からしか呼ばれない**)
#
# 落とした財布の中の現金は、拾われるまで **街の中に在るが誰の残高でもない**。供託金
# (``escrow`` = 行政の預り金)と構造は同じ「主体を持たない域内残高」だが、勘定は分ける:
# 供託は行政が預かっている金で、遺失物の現金は**まだ誰も預かっていない**。混ぜると
# finance サイドカーの ``escrow`` 列の意味(議案/立候補供託の残高)が壊れる。
#
# ★これを city_total に足す理由: 足さないと「財布を落とした瞬間に街から金が消え、
#   拾われた瞬間に湧く」ことになり、閉じた不変量 Σ(全主体残高)+RoW+K5 が破れる。
#   拾得金が必ず誰かの drop から出ることを**会計でも**保証するのが H3 の核心
#   (計画書 §4「★拾得金は必ず誰かの drop から = 貨幣保存則(IF-E ゼロ和検査)と整合」)。
# --------------------------------------------------------------------------- #
def _lost(st: dict) -> float:
    # 旧 checkpoint(第97/98バッチ)からの resume でもキーが生える(setdefault = 互換の作法)
    return float(st.setdefault("lost", 0.0))


def on_lost_hold(sim, amount: float) -> str | None:
    """財布の現金が所持金から分離された(街の中の「遺失物」へ移った)。"""
    if not enabled(sim):
        return None
    st = _state(sim)
    st["lost"] = _lost(st) + float(amount)
    return "lost_property"


def on_lost_release(sim, amount: float) -> str | None:
    """遺失物の中の現金が誰かの財布へ入った(返還 / 着服 / 時効取得)。"""
    if not enabled(sim):
        return None
    st = _state(sim)
    st["lost"] = _lost(st) - float(amount)
    return "lost_property"


def on_lost_lapse(sim, amount: float) -> str | None:
    """誰にも拾われないまま失われた現金 = **K5(取引でない資産変動)**。

    SNA 2008 §3.98 の窃盗と同じ理由で RoW(取引の相手方)へ落としてはならない
    (受け取り手が世界に存在しない)。``on_theft`` と同じ扱いにする。"""
    if not enabled(sim):
        return None
    st = _state(sim)
    st["lost"] = _lost(st) - float(amount)
    return k5_out(sim, "lost_property", float(amount))


# --------------------------------------------------------------------------- #
# 第98バッチ IF-E2 UNCOVERED — 残り 4 種の記帳口(いずれも既定 OFF で即 return)
#
# 共通規約: **金額もタイミングも既存の動力学を 1 つも変えない**。呼び出し元は支給/被害/仕入れを
#   従来どおり済ませてから、その額を本節の関数へ渡して**分類だけ**を足す(payload に 1 語増える)。
# --------------------------------------------------------------------------- #
def on_rule_bonus(sim, amount: float) -> str | None:
    """制度DSL の bonus 支給を**区(ward)の歳出**として記帳する(``rules.apply_bonus`` の唯一の口)。

    ★``Government.expense`` を使わない理由: あちらは議会の予算承認フック(``exec_ratio``)で
      区の歳出額に執行率を掛ける。本関数が呼ばれる時点で **agent は既に満額を受け取っている**ので、
      執行率で目減りさせると差額が無から生まれて不変量が破れる。``exec_ratio`` は「これから出す
      歳出」を絞る装置であって、**事後の記帳**に掛けるものではない(掛けたい場合は支給額そのものを
      絞るべきで、それは rules.py 側の設計変更 = 本バッチの範囲外)。

    区の残高が負に振れるのは許容する(``debit_org`` の自動当座借越と同じ思想 = 払えないことを
    隠さずに記録する。行政の破綻処理は世界に無い)。行政がまだ実体化していない世界
    (ON では ``scheduler._sfc_arm`` が step 先頭で必ず作る)では支給主体が街に居ないので RoW。"""
    if not enabled(sim):
        return None
    amt = float(amount)
    if amt <= 0.0:
        return None
    gov = getattr(sim, "government", None)
    if gov is None:
        return row_in(sim, "subsidy_no_authority", amt)
    gov.balance["ward"] -= amt
    gov.day_exp["ward"] += amt              # public_budget の内部整合(残高=前+歳入−歳出)を保つ
    _state(sim)["bonus_out"] += amt
    return "government"


def on_ems_transport(sim, amount: float):
    """救急搬送の**公費**を記帳する(H2 ``medical.py`` からの唯一の口)。戻り値 ``(payer, payee)``。

    ``on_rule_bonus`` と同型の「区(ward)の歳出」だが、**受け手が違う**: bonus は家計が受け取る
    (だから相手方は街の中)のに対し、救急を運行するのは東京消防庁 = **都**であって、街の中に
    運行主体の org は 1 つも存在しない。したがって受け手は RoW の ``ems_operation`` に置く
    (区は都区財政調整を通じて消防費を負担する側なので、支出主体=区・受け手=街の外が実態に近い)。
    区 −額 / RoW +額 = 総マネー保存 ``city + RoW + K5`` はそのまま閉じる。

    行政がまだ実体化していない世界では**街の残高が 1 円も動かない**ので None を返す
    (= 街の外どうしの取引。``ems_transport`` の payload には payer/payee が載らず、
    解析側もフローを立てない = 動いていない金を動いたことにしない)。"""
    if not enabled(sim):
        return None
    amt = float(amount)
    if amt <= 0.0:
        return None
    gov = getattr(sim, "government", None)
    if gov is None:
        return None
    gov.balance["ward"] -= amt
    gov.day_exp["ward"] += amt              # public_budget の内部整合(残高=前+歳入−歳出)を保つ
    st = _state(sim)
    st["ems_out"] = float(st.get("ems_out", 0.0)) + amt
    return "government", row_out(sim, "ems_operation", amt)


def on_insurance(sim, node: str, cat: str, amount: float):
    """医療保険の給付(**街の外の保険者 → 医療機関 org**)。戻り値 ``(payer, payee)``。

    受け手は ``(node, POI種別)`` の台帳索引で引く(``resolve_payee_at_node`` = b2b の買い手と
    同じ鍵)。**決まらなければ None**: 保険者も医療機関も街の外に居るので、街の残高は 1 円も
    動かない(RoW → RoW の取引を RoW の累積へ足すと、域外依存度の指標が水増しになる)。
    その分は medical 側の provenance ``insurance_to_row`` に出るので、黙って消えることはない。"""
    if not enabled(sim):
        return None
    amt = float(amount)
    if amt <= 0.0:
        return None
    oid = resolve_payee_at_node(sim, str(node), str(cat))
    if oid is None:
        return None
    credit_org(sim, oid, amt)
    st = _state(sim)
    st["insurance_in"] = float(st.get("insurance_in", 0.0)) + amt
    return row_in(sim, "insurance_reimbursement", amt), str(oid)


def on_theft(sim, amount: float) -> str | None:
    """窃盗で被害者から減った額を **K5(その他の資産変動)** へ分類する(``diversity.tick_crime``)。

    SNA 2008 §3.98 のとおり窃盗は**取引ではない**ので、RoW(取引の相手方)へ落としてはならない。
    ``amount`` は**丸める前の実減少額**(床クリップ後)を渡すこと(payload の丸め値ではない)。
    ★加害者へ入金する動力学変更は**しない**(module docstring の「将来の判断」宣言)。"""
    if not enabled(sim):
        return None
    return k5_out(sim, "theft", float(amount))     # 0 でもトークンは返す(row_* と同じ流儀)


def on_inheritance_escheat(sim, amount: float) -> str | None:
    """★相続人不存在の遺産 = **国庫へ帰属**(民法 959 条)。所有権レイヤー O3 の唯一の口。

    ``on_theft`` と違って**取引である**(法定の移転であって、相互合意の欠如ではない)ので
    K5 ではなく RoW へ落とす。受け手の国は街の中に居ないため ``inheritance_escheat`` チャネル。
    相続人が居る回は家計 → 家計の内部移転で街の総額が 1 円も動かないので、ここは呼ばれない。

    ``amount`` は**丸める前の実移動額**(現金 + 口座残高)を渡すこと。IF-E2 が OFF のランでは
    即 return して ``sim._sfc_state`` を生やさない(相手の「OFF ではキー自体を作らない」規約を
    こちらから壊さない = ``on_lost_lapse`` と同じ作法)。"""
    if not enabled(sim):
        return None
    amt = float(amount)
    if amt <= 0.0:
        return None
    return row_out(sim, "inheritance_escheat", amt)


def on_chance(sim, kind: str, amount: float) -> str | None:
    """偶発の臨時収入 / 紛失を RoW の外生チャネルへ分類する(``chance._apply_money``)。

    ``kind`` は "windfall"(RoW → 街)/ "loss"(街 → RoW)。``amount`` は**丸める前の実変化量**
    (loss が手持ちで頭打ちになった場合は実損)。戻り値は windfall なら payload の ``payer``、
    loss なら ``payee`` に載せるトークン。"""
    if not enabled(sim):
        return None
    amt = float(amount)
    if str(kind) == "windfall":
        return row_in(sim, "chance_windfall", amt)
    return row_out(sim, "chance_loss", amt)


def on_b2b_trade(sim, node: str, cat: str, seller_org, amount: float,
                 step: int, sim_min: int) -> tuple[str, str] | None:
    """卸→小売の仕入れを **org 預金間の実移転**にする(``b2b.fulfill`` の唯一の口)。

    戻り値 ``(payer, payee)`` = payload に載せる買い手 / 売り手のトークン(OFF は None)。
    買い手は小売 POI ``(node, cat)`` を台帳索引で解決する。決まらなければ**域外資本の店**と
    みなして ``b2b_buyer_unknown``(RoW → 街)から払う(``unknown_payee`` の裏返し)。
    買い手と売り手が同一 org に解決したときは**残高を動かさない**(自己取引 = 純額ゼロ。
    動かすと当座借越の L1 が偽って立つ)。**b2b の在庫・仕入れ成否・トリップには一切触れない**。"""
    if not enabled(sim):
        return None
    amt = float(amount)
    seller = str(seller_org)
    if amt <= 0.0 or not _is_org(sim, seller):
        return None
    buyer = resolve_payee_at_node(sim, node, cat)
    if buyer == seller:                                # 自己取引(純額ゼロ)
        return seller, seller
    if buyer is None:
        payer = row_in(sim, "b2b_buyer_unknown", amt)  # 域外資本の店が払う(正直開示)
    else:
        debit_org(sim, buyer, amt, step, sim_min, reason="b2b")
        payer = buyer
    credit_org(sim, seller, amt)
    _state(sim)["b2b_transfer"] += amt
    return payer, seller


# --------------------------------------------------------------------------- #
# 不変量(§4-1 (b): Σ(全主体残高) + RoW 累積 = 一定。ABCredit.jl 流のスカラー総マネー保存)
# --------------------------------------------------------------------------- #
def city_total(sim) -> float:
    """街の中に**実在する**全残高の合計(家計 + org + 行政 + 供託 + 遺失物 + 銀行 + VC)。

    ``lost``(H3 遺失物の中の現金)は「街の中に在るが誰の残高でもない」金。既定 OFF の
    ``lost_property`` が動かないランでは常に 0.0 = 従来と完全同一(``.get`` で旧 ckpt 互換)。
    """
    total = 0.0
    for a in sim.agents:
        total += float(a.money) + float(getattr(a, "account", 0.0) or 0.0)
    st = state_of(sim)
    if st is not None:
        total += sum(st["org"].values())
        total += float(st["escrow"])
        total += float(st.get("lost", 0.0))
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
    """**閉じた不変量**: 街の全残高 + RoW 累積 + K5 累積。一定であることをテストで固定する
    (ABCredit.jl ``test/stock_flow_consistency.jl`` の ``isapprox(init, tot, atol)`` と同型)。

    第98バッチで **K5(取引でない資産変動。SNA 2008 第12章)** の項が増えた。RoW と足し合わせて
    1 本にしないのは、「街の外の取引相手」と「そもそも取引でないもの」を混ぜないため(§3.98)。"""
    return city_total(sim) + row_net(sim) + k5_total(sim)


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
                       ensure_ascii=False, sort_keys=True),
            round(sum((st.get("k5") or {}).values()), 6))


def _emit(sim, st: dict, day: int, step: int, sim_min: int,
          row_step: int | None = None) -> None:
    """L1 ``row_flow``(当日の域外収支)+ サイドカー 1 行。日次 = 高頻度にしない。

    ``row_step`` はサイドカー行に書く step(既定 = L1 の step)。日境界の行は
    ``day_roll`` = **step 先頭**で採られるので「その step の活動より前の残高」を意味し、
    窓 [s0, s1) の排他境界がそのまま成立する。最終行だけは全 step を走り終えた後に採るので
    ``n_steps`` を書く(そうしないと最後の step のフローが窓から落ちる = オフバイワン)。

    IF-E2 残③(第99): K5(取引でない資産変動。SNA 2008 §3.98)は今まで
    ``finance.parquet`` の ``k5_other`` 列**にしか**出ていなかった = L1 だけを見る解析
    (l1_stream / detect_regression / 外部の読み手)から K5 が見えず、総マネー保存
    ``city + RoW + K5`` の第 3 項を L1 単独では閉じられなかった。ここで **1 キーだけ**
    足す。粒度は ``in_total`` / ``out_total`` と同じ**累積**にする(内訳は provenance の
    ``k5_kinds``・日次差分は連続する 2 行の引き算で採れる = 新しい指標を定義しない)。"""
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
                                  "org_balance": round(sum(st["org"].values()), 1),
                                  # 第99: K5 累積。`_finance_row` の `k5_other` 列と
                                  # **同一の式**(丸め桁だけ L1 の他キーに合わせて 1 桁)。
                                  # float() は空 dict の sum が int 0 になるのを防ぐ
                                  # (L1 は JSON バイト比較の対象 = 型が揺れてはいけない)。
                                  "k5_total": round(
                                      float(sum((st.get("k5") or {}).values())), 1)}))
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
        # 第98バッチ: 取引でない資産変動(SNA K.5)と、新たに接続した 3 経路の累計。
        "k5_total": round(k5_total(sim), 1),
        "k5_kinds": {k: round(v, 1) for k, v in sorted((st.get("k5") or {}).items())},
        "bonus_out_total": round(float(st.get("bonus_out", 0.0)), 1),
        "b2b_transfer_total": round(float(st.get("b2b_transfer", 0.0)), 1),
        # H2 医療の三本足のうち、**街の残高を動かした 2 本**の累計(既定 OFF は 0.0)。
        "ems_out_total": round(float(st.get("ems_out", 0.0)), 1),
        "insurance_in_total": round(float(st.get("insurance_in", 0.0)), 1),
        "escrow": round(float(st["escrow"]), 1),
        # H3 遺失物: 街の中に在るが誰の残高でもない現金(既定 OFF のランでは常に 0.0)
        "lost_property_held": round(float(st.get("lost", 0.0)), 1),
        "city_total": round(city_total(sim), 1),
        "total_money": round(total_money(sim), 1),
        # ★接続できていない金の経路の宣言。第98バッチで**空になった**が、キーは後方互換と
        #   監視装置として残す(装置は残したまま値をゼロにする)。新しい金の経路が生えたら
        #   ここか COVERED_KINDS への追加が強制され、解析側の leak_families にも現れる。
        "uncovered_kinds_declared": dict(sorted(UNCOVERED_KINDS.items())),
    }


# --------------------------------------------------------------------------- #
# finance.parquet(L2 サイドカー・日次 1 行。observer/org_ledger.OrgLedger と同型)
#
# ★記録と動力学の分離(indoor_tracks.py の設計原則③): 動力学は add_rows / flush_segment /
#   finalize しか呼ばず、本クラスの内部バッファ(.rows)を読む向きは存在しない。
# ★W4-E(第99バッチ): part 群 → canonical の結合は observer/finalize.py の
#   FinalizeStreamMixin **1 本**が実装する(同型 finalize の二重実装をやめた)。conf
#   `observer.finalize.streaming`(既定 false)が L1 と同じ 1 つの判断で本サイドカーにも効く。
# --------------------------------------------------------------------------- #
COLUMNS: tuple[str, ...] = (
    "day", "step", "org_balance", "org_count", "org_negative", "org_min",
    "bank_capital", "vc_balance", "gov_balance", "escrow", "household_balance",
    "row_in", "row_out", "row_channels",
    # 第98バッチ: K5(取引でない資産変動。窃盗の未記録な受け取り側)の累計。
    # 末尾に足す(既存 14 列は順序含め不変 = 会社UI/解析の契約を壊さない)。
    "k5_other",
)


class FinanceLedger(FinalizeStreamMixin):
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
            "k5_other":          pa.array([r[14] for r in rows], pa.float64()),
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

    # 結合(既定の concat 経路 / streaming 経路)は FinalizeStreamMixin が唯一の実装。
    # ここに自前の複製は置かない(W4-E)。
    def finalize(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        return self._finalize_stream(_STEM,
                                     self._table(self.rows) if self.rows else None)
