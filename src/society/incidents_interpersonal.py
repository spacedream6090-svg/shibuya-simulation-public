"""対人事件の収束化 H4(``incidents_interpersonal``・**既定 OFF**)。

正典
----
- docs/plans/body-incident-layer-plan.md **§3 の「対人」行**
  (レート → **共在収束**(Birks / Groff の RAT = Routine Activity Theory)。
   加害者候補(動機状態)× 標的 × **監視者不在** が同じ場所に**揃ったときだけ**事件になる。
   喧嘩 = **酒 × 密度 × 閉店放出**の関数。閉店 +1 時間の **+16%** が検証対象の実証弾性)
- 同 **§6-3 ユーザー決定**(原文)「**H4はエージェントドリブンで設計**」
  = レート抽選の残滓を残さない。事件は「**誰もいなければ起きない**」を**構造で**保証する。
- 同 **§3 の通報層**(110 / 119 という**通報という行為**を挟む。誤通報・非緊急通報も行為 =
  110 番の **16% が非緊急**という現実ごと再現できる)
- 同 **§7 の分母の罠**(犯罪は人口比ではなく**共在機会比**で較正する。区の遭遇率は夜間人口
  分母なので、来街ピークで事件が過剰になる)/ **管轄混同**(渋谷署 ≠ 渋谷区 = 3 署)

何を解く問題か
--------------
現行の窃盗(``diversity.tick_crime``)は「**1 人 1 step あたり ``crime_prob``**」の抽選である。
つまり**人口と時間に直接掛かるレート**で、共在は「当たったあとで被害者を探す」ための
後付けの条件でしかない(``_pick_victim`` が None なら ``continue`` = 抽選は既に消費済み)。
これは 3 つの意味で現実と食い違う:

1. **孤立した個体からも乱数は引かれる**。世界に誰も居なくても「窃盗の抽選」は毎 step 走る。
2. **監視は事後のフィルタ**である。近傍の警察官は抑止するが、それは「加害者候補を消す」
   だけで、「監視の目が事件の確率を下げる」という RAT の中心命題は表現されていない。
3. **標的の適性が存在しない**。被害者は同ノードの**最小 id** で、財布を持っているかも
   酩酊しているかも独りかも見ていない。

本 module は事件を **(加害者候補 × 標的) という共在ペアの上の条件付き確率**として作り直す。
基底レートは**ペアの上にしか存在しない**ので、共在が無ければ乱数は 1 本も引かれない
(= 「誰もいなければ起きない」が config の値ではなく**制御フローの構造**で保証される)。

★エージェントドリブンの機械的保証(テストが固定する 3 点)
----------------------------------------------------------
1. **乱数を引く関数は :func:`_pair_draw` **ただ 1 つ**で、その本体の先頭は
   ``if not pairs: return None``**(= 共在ペアが空なら乱数は引かれない)。
   ``tests/test_incidents_interpersonal.py`` が **AST で**「本 module 中の
   ``hub.stream`` 呼び出しは 1 か所だけ・それは ``_pair_draw`` の中・
   ``_pair_draw`` の最初の文は空ペアの早期 return」を機械固定する。
2. **知覚半径を 0 にした世界(= 誰とも共在しない)では、``pair_prob=1.0`` でも
   事件 0 件・乱数抽選 0 回**(``provenance()`` の ``draws`` が 0)。
3. ``phase`` の中で ``sim.agents`` を走るのは**動機の計算(候補の絞り込み)まで**で、
   事件の生成は ``sim.percept_index``(唯一の共在索引)から得たペア列の上でしか起きない。

RAT の 3 要素(すべて**既存の agent 状態と世界状態の純関数**)
--------------------------------------------------------------
| 要素 | 何から作るか | ★禁止したこと |
|---|---|---|
| **動機**(加害者候補)| 手持ちの現金の少なさ / 家賃滞納・立退き・破産(= 一時的で**回復可能**な経済的困窮)/ 酩酊 / 疲労 | **新しい人格ラベルを付けない**(属性も「傾向」パラメータも作らない)= 状態が戻れば候補から外れる |
| **標的** | 携行する現金 / 携帯の有無 / 酩酊 / **孤立**(同伴者が居ない) | 特性(traits)を 1 つも読まない |
| **監視者** | 同席者の数(``sim.percept_index``)/ **交番の近接** / ``traces`` の摘発痕跡 / 同ノードの警察官 | 監視が閾値以上なら**抽選そのものを行わない**(事後フィルタにしない) |

★**不満(grievance)の重みは既定 0.0**。動機に構成概念を入れると「内面状態が発火判断に
入る」= 本リポの R1 ドクトリン(``diversity`` / ``health`` の module docstring が明記)に
抵触するので、seam は開けるが**既定では 1 ミリも効かせない**。疲労(fatigue)は
``arousal`` と同じ内部 transient(states 監査集合の外)なので既定で小さく効かせている。

既存 theft の世代交代(``diversity.tick_crime``)
------------------------------------------------
本 module が ON のとき、``diversity`` は**窃盗の枝を供給しない**
(``diversity.superseded_theft`` = ``street_life`` ON のとき ``nuisance_kinds_for`` が
「客引き」を供給しなくなるのと**同じ作法・同じ理由**: 併存ではなく**置き換え**にしないと、
L1 の ``crime`` 件数が 2 つの機構の和になって RAT の効き目が測れない)。
★**乱数の消費列は 1 バイトも変えない**: ``diversity`` は従来どおり 1 人 1 step に
``crime`` stream を 1 回引き、窃盗の枝に入ったときだけ**何もせずに次へ行く**
(抽選そのものを飛ばすと迷惑行為の判定まで動いて OFF/ON の比較にならない)。
生成された窃盗は従来と**同じ L1 種 ``crime``**(payload ``kind="theft"``)で出す
= ``traces`` の trouble 源・``diversity`` の危険地帯・``causality`` の分類・既存の解析
スクリプトがそのまま効く。世代の別は payload の ``"src": "rat"`` で分離できる
(``rumors`` が item_id 接頭辞で世代を分離したのと同じ流儀)。

★L1 は 1 行 + **前兆状態を同梱**(内生性の機械検証)
----------------------------------------------------
``crime`` / ``brawl`` の payload に、発火した瞬間の **監視者数・密度・酩酊・閉店フラグ・
動機・標的適性** を必ず載せる。「この事件は外から注入されたのではなく、その場の状態から
生まれた」ことを**解析側だけで**検証できるようにするため(乱数注入ゼロの AST 固定と対に
なる観測側の担保 = IF-C / IF-D の流儀)。

通報層(110 番 = **目撃者/被害者の行為**)
-------------------------------------------
- 通報は :data:`REPORT` = ``crime_report``(cause_type ``agent``)で、**新しい乱数を 1 本も
  引かない完全決定論**(誰が居合わせたかは既に物理が決めている)。
- **傍観者効果は「個人」に掛け、判定は「集団」で行う**(計画書 §1 の 2 つの実測値を
  同時に満たす唯一の入れ方): 個人の通報確率は曖昧・軽微な状況でだけ ``n^-α``
  (α ≈ 0.4)で落ち、危険事態では落ちない(α=0)。**集団**の「誰かが通報する」確率は
  ``1 − Π(1−p_i)`` の解析式で、既定値では n=3 で 0.83 / n=5 で 0.90 / n=8 で 0.94
  = 実 CCTV 研究の**集団介入率 91%**の帯に乗る(個人の p が人数で落ちても集団の
  確率は上がる、という研究の所見をそのまま構造にした)。
  ★**喧嘩の当事者は通報者にならない**(その step の行動枠をもみ合いに使い切っている)ので、
  傍観者効果が実際に効く場面(= 負傷の無いもみ合いを、居合わせた人数が多いほど誰も
  通報しない)が世界の中にできる。
- **非緊急通報は捏造しない**: 緊急なのは「危険事態(負傷)」か「**進行中を目撃した**通報」
  だけで、**財布が無いことに事後に気づいた被害者**の通報は ``urgent=false`` になる。
  実測の **110 番の 16% が非緊急** はレートで注入せず、この**出どころの違い**から出す。
- **誰が加害者かを名乗れるか**も内生: 窃盗の被害者は取られる瞬間を見ていないので特定できず、
  現場から離れた目撃者(``clear_radius_m`` 超)も特定できない。どちらも ``unclear=true`` で
  通報し、payload の ``offender`` は ``null`` になる(欠測を偽の値で埋めない)。
- 応答(:data:`RESPONSE` = ``police_response``)は**本 module 内の簡約実装**。
  ``city_ops``(交番配置)は**読まない**(別レーン所有)。当直の警察官が世界に居なければ
  ``unstaffed=true`` の正直な無人マーカーを出す(``ems_dispatch`` / ``dwell_decision`` と
  同じ作法)。★**警察官を動かさない**(``work_node`` の差し替えは ``city_ops`` の作法で、
  ここでやると 2 つの機構が同じ属性を奪い合う)。``detain_steps`` > 0 のときだけ既存の
  勾留 seam(``agent.detained_until``)を使うが、**既定 0**(勾留は発火権を止めるので
  LLM 呼数が動く = 既定では 1 も動かさない)。

較正(**実測してから置いた値**。手で決め打った数字は 1 つも無い)
--------------------------------------------------------------
現実のアンカー(計画書 §3・§7):

  - 刑法犯 **10.5 件/日**(渋谷**区**)。★管轄混同に注意 = 渋谷署 ≠ 渋谷区(区内は 3 署)
    なので、署の統計をそのまま区の分母に掛けると 3 倍ずれる。ここでは**区**で揃えた。
  - 昼間人口 **54 万** → 刑法犯 ≈ **1.9e-5 件/人日**。
  - 既存の窃盗の実測較正値 = **2.8e-5 件/人日**(渋谷区の認知件数 ÷ 昼間人口。
    ``docs/calibration/calibration-20260709.md`` が ``crime_prob 2.0e-6`` を導いた分母と同じ)。
    許容帯は同文書の **1e-5 〜 3e-4 件/人日**(観測性のため 10 倍まで許容)。

★**分母は人口比ではなく共在機会比**(§7「分母の罠」): ``pair_prob`` は
  「共在ペア 1 組 1 step あたり」の条件付き確率なので、**同じ値でも人口密度が上がれば
  1 人あたりの件数は増える**(共在の機会が増えるから = これは欠陥ではなく RAT そのもの)。
  実測(mock・144 step):

      n_agents=40  → 共在重み和 theft 40.8 / brawl 4.6
      n_agents=80  → 同        theft 161.3 / brawl 17.1
      n_agents=100 → 同        theft 251.4 / brawl 27.9   ← **production の規模**

  既定値は **production の規模(100 体)で theft ≈ 2.4e-5・brawl ≈ 0.4e-5 = 合計 2.8e-5
  件/人日**になるように置いた(= 既存 theft のアンカーと同水準・世代交代で発生量が動かない):

      theft.pair_prob = 2.4e-5 × 100 / 251.4 ≈ **9.5e-6**
      brawl.pair_prob = 0.4e-5 × 100 /  27.9 ≈ **1.4e-5**

  ★別の規模で回すときは ``summary.incidents_interpersonal`` の ``weight_by_kind`` /
    ``per_person_day`` / ``per_pair_step`` を見て置き直すこと(値をここに固定した理由と
    測り方が判るように、重み和の実測をこの docstring に残してある)。
  ★**「今日は何も起きない」が正しい日が多い**(§7 の災害映画化の防止): 100 体 1 日の
    期待値は窃盗 0.0024 件で、ほとんどの日は 0 件になる。パターン検収(場所集中・反復被害)は
    ``pair_prob`` を大きく上書きした専用ランで行う(テストがそうしている)。

R1 ドクトリン
-------------
- 既定 ``incidents_interpersonal.enabled: false`` では :func:`phase` が即 return し、
  **L1 に 1 件も出ず・agent に属性が生えず・sim に state が生えず・プロンプトが 1 バイトも
  変わらず・新 stream を 1 本も引かない**(= ゴールデン L1 バイト一致)。
- **generate() の呼び出しサイトを 1 つも作らない**(LLM 追加呼ゼロ = k 非依存)。
  加害も通報も**動機状態の決定論的な純関数**で、出来事は既存の記憶機構
  (``agent.remember``)へ定型 1 行が入るだけ = プロンプトの**欄**は 1 つも増えない
  (``street_life`` / ``city_ops`` と同じ線引き = ``perception_contract`` の追随不要)。
- 乱数は**新 named stream ``"incident_pair"`` 1 本だけ**(既存の draw 順に挿入しない)。
- no-fingerprint: 記憶の 1 行は :func:`sentence` = **出来事の種類だけの純関数**で、
  金額・件数・監視者数・config・実験条件・機構語を 1 文字も含まない。

★路上生活者の尊厳規約(``street_life`` の明文条件をそのまま継承)
-----------------------------------------------------------------
本 module は ``street_life.rough_sleeper_ids`` を**加害者・標的・目撃者・通報者の
すべてから除外**する(``diversity`` の ``_street_life_excluded`` と同一の集合・同一の理由)。
= L1 の ``crime`` / ``brawl`` / ``crime_report`` の payload に彼らの id が 1 度も現れない。
これは現実の主張ではなく「**この機構と結びつけない**」という設計判断である。

正直な限界(6 件)
------------------
1. **酩酊は「夜間店舗に居た痕跡」の代理**である。血中濃度も飲酒量も作っていない
   (H1 レーンが急性アルコールの重症度を持つまでは、``intox.steps`` step の減衰する印だけ)。
2. **喧嘩の負傷は記憶 1 行で終わる**(H1 が OFF のとき)。H1 が ``health.on_injury`` を
   生やしたら**自動的に**そちらへ繋がる(``getattr`` の soft 依存 = あちらのファイルを
   1 バイトも触らずに待つ)。
3. **警察官は現場へ動かない**(上記)。``police_response`` の ``response_min`` は距離から
   計算した**モデル値**で、実際に何 step で着いたかとは別物である。
4. **プール退場で酩酊の印を失う**: ``world/pool.py`` の dehydrate は明示列挙した可変状態
   しか運ばないので、街を出て戻った個体の ``_inc_intox_until`` は落ちる
   (``rumors`` の ``_rumors`` と同じ制約。pool.py を触らずに直せる方法が無い)。
5. **標的の選択は「加害者から見た適性の重み付き」であって、加害者の探索行動ではない**。
   「良い獲物を探して歩く」動きは作っていない(移動は既存 routine の所有)。
6. **1 step の遅れ**: 事件は共在索引が張られた直後に起きるので、その step の思考に
   1 行が乗るのは「同 step」だが、**索引が張られた時点の位置**で判定する
   (``_apply`` 中の移動は捉えない = ``rumors`` の目撃者判定と同じ粒度)。
7. **ペア確率の合成は和で近似している**(:func:`_pair_draw`)。厳密には
   ``1 − Π(1−p_i)`` だが、較正値では ``p_i ≈ 1e-5`` なので差は無視できる
   (相対誤差は ``Σp`` のオーダー)。``pair_prob`` をテストのように大きく上書きすると
   ``Σp`` が 1 を超えて**飽和**する(必ず 1 件起きる)= 較正帯の外では確率の意味が
   保たれないことを正直に書いておく。上限クリップは**入れない**(黙って歪めるより、
   ``weight_sum`` と ``per_pair_step`` を見て気づける方が良い)。
"""
from __future__ import annotations

import hashlib

from . import economy_sfc as sfc_mod
from . import traces as traces_mod
from .factors import update as factor_update
from .observer.schema import Event, register_event_kind
from .world.perception import hearers_of

SCHEMA = 1

# --------------------------------------------------------------------------- #
# L1 イベント種(**材料側 registration**: city_ops.py:167 / street_life.py / devices.py と
# 同じ流儀 = observer/schema.py 本体を 1 バイトも触らずに自分の種を自分で登録する)。
#
# ★``crime``(窃盗)は**新設しない**: 既存種の世代交代なので同じ種で出す
#   (traces の trouble 源・危険地帯・causality・既存解析がそのまま効く)。
# ★payload に自由文は 1 つも入らない(node は世界の識別子・残りは id と件数と真偽値)。
# --------------------------------------------------------------------------- #
BRAWL = "brawl"
REPORT = "crime_report"
RESPONSE = "police_response"

register_event_kind(BRAWL, "★喧嘩(酒 × 密度 × 閉店放出の共在から発火。双方向 = "
                           "加害と被害が対称)。agent_id = id の小さい側"
                           "{other, node, severity, guardians, density, intox, "
                           "closing, motive, suitability}")
register_event_kind(REPORT, "★110 番通報(**目撃者/被害者の行為**。誤通報・非緊急も行為)。"
                            "agent_id = 通報者{incident, node, victim, offender, "
                            "self_report, witnesses, urgent, unclear, delay_min}")
register_event_kind(RESPONSE, "★通報に応えた警察官の臨場(因果の親は crime_report)。"
                              "agent_id = 警察官(当直不在なら -1 かつ "
                              "payload.unstaffed=true){node, caller, incident, "
                              "response_min, unstaffed, detained}")

# 窃盗が使う既存種(**登録しない = 既に schema.py が持っている**)
THEFT_KIND = "crime"

#: 本 module が出す事件の種類(観測・provenance の並び)
INCIDENT_KINDS: tuple[str, ...] = ("theft", BRAWL)

# 警察官(G3 執行の公務員)。``diversity.POLICE_OCCS`` と同一(循環 import を避けて再掲)。
POLICE_OCCS: tuple[str, ...] = ("警察官",)

# --------------------------------------------------------------------------- #
# 記憶の 1 行(**出来事の種類だけの純関数**。数字・金額・監視者数・config・実験条件・
# 機構語は 1 文字も入らない)= traces.sentence / street_life.NOTICE_TEXT / city_ops の
# COLLAPSE_TEXT と同じ no-fingerprint の定型文規約。
# --------------------------------------------------------------------------- #
TEXTS: dict[str, str] = {
    "theft_victim":   "気づいたら持ち物が無くなっていた。",
    "theft_offender": "その場の勢いで、他人の持ち物を持ち去ってしまった。",
    "brawl":          "路上で他の人ともみ合いになった。",
    "injury":         "もみ合いで体を痛めた。",
    "witness":        "目の前で揉め事が起きるのを見た。",
    "report":         "その場から警察に通報した。",
    "response":       "通報を受けて現場へ向かった。",
}


def sentence(kind: str) -> str:
    """記憶の 1 行。**出来事の種類だけの純関数**(語彙外は空文字 = 捏造しない)。"""
    return TEXTS.get(str(kind), "")


# --------------------------------------------------------------------------- #
# 既定値(**すべて OFF 側 = 現行と完全同値**)
#
# ★``pair_prob`` は「**共在ペア 1 組 1 step あたり**の条件付き確率」であって、
#   人口にも時間にも直接掛からない。較正の物差しは §7 の「分母の罠」に従い
#   **共在機会比**である(下の「較正」節を参照)。
# --------------------------------------------------------------------------- #
DEFAULTS: dict = {
    "enabled": False,
    "max_events_per_step": 16,      # 1 step に出す本 module の L1 件数の上限(安全弁)
    "max_pairs_per_offender": 8,    # 1 候補が同 step に見るペア数の上限(O(n·k) の頭)
    # ---- 動機(加害者候補。**一時的で回復可能な状態だけ**)----
    "motive": {
        "money_low": 1500.0,        # 手持ちがこれ未満 = 金欠(円)
        "w_money": 0.50,
        "w_distress": 0.30,         # 家賃滞納 / 立退き / 破産(既存の一時状態)
        "w_intox": 0.40,
        "w_fatigue": 0.20,
        "w_grievance": 0.0,         # ★既定 0(構成概念を発火判断に食わせない)
        "min": 0.05,                # これ未満は候補にしない(前置フィルタ)
    },
    # ---- 標的の適性(VIVA: 価値・慣性・可視性・接近可能性の観測可能な代理)----
    "target": {
        "cash_ref": 8000.0,         # これだけ持っていれば価値項が飽和(円)
        "w_cash": 0.60,
        "w_phone": 0.20,
        "w_intox": 0.40,
        "w_alone": 0.40,            # 同伴者が居ない(孤立)
        "min": 0.05,
    },
    # ---- 監視(guardianship)----
    "guardian": {
        "w_watcher": 0.34,          # 同席者 1 人あたりの監視スコア(3 人で block に届く)
        "w_koban": 1.00,            # 交番の近くに居る
        "w_trace": 0.50,            # traces の摘発痕跡 1.0 あたり
        "block": 1.00,              # このスコア以上で**抽選そのものを行わない**
        "koban_radius_m": 60.0,
        "koban_keywords": ("交番", "警察署"),   # 一般名詞のみ(ブランド名・地名は書かない)
        "koban_cats": ("service", "police"),
        "police_blocks": True,      # 同ノードの警察官は無条件で抑止
    },
    # ---- 酩酊(夜間店舗に居た痕跡の減衰する印。血中濃度は作らない)----
    "intox": {
        "steps": 12,                # 印の寿命[step](既定 Δt=10 分なら 2 時間)
        "cats": ("nightlife",),
        "open_hour": 18,            # 夜間店舗の窓(commerce の既定表と同値)
        "close_hour": 5,
    },
    # ---- 窃盗(既存 theft の世代交代)----
    "theft": {
        "enabled": True,
        "pair_prob": 9.5e-06,       # ★較正値(module docstring の「較正」節)
        "amount": 3000.0,           # 被害額(diversity.theft_amount と同値)
        "grievance": 0.0,           # 被害者の不満増分(factors 経由。0=金銭のみ)
        "supersede_diversity": True,
    },
    # ---- 喧嘩(酒 × 密度 × 閉店放出)----
    "brawl": {
        "enabled": True,
        "pair_prob": 1.4e-05,       # ★較正値(module docstring の「較正」節)
        "density_min": 3,           # 同席者がこの人数以上のときだけ起きる(密度項)
        "require_intox": True,      # 双方または片方が酩酊していること(酒項)
        "closing_window_min": 60,   # 閉店から何分を「放出」とみなすか
        "closing_boost": 0.16,      # ★閉店 +1 時間の実証弾性 +16%(検証対象)
        "night_only": True,         # 夜間店舗の窓の中でだけ起きる
        "injury_density": 5,        # 密度がこれ以上 かつ 双方酩酊 → S2(中等症)
        "grievance": 0.0,
    },
    # ---- 通報層(110 番。**新しい乱数を 1 本も引かない**)----
    "report": {
        "enabled": True,
        "victim_base": 1.00,        # 被害者本人の通報確率(傍観者ではないので人数減衰なし)
        "witness_base": 0.70,       # 目撃者 1 人あたりの通報確率(人数減衰の基点)
        "kin_bonus": 0.30,          # 家族・同居人・恋人は倍率 > 1(関係者)
        "threshold": 0.50,          # **集団**の通報確率がこれ以上なら通報が起きる(決定論)
        "alpha_ambiguous": 0.40,    # 曖昧・軽微な状況の**個人**の人数減衰(Darley & Latané)
        "alpha_danger": 0.0,        # 危険事態は人数で落ちない(メタ分析)
        "clear_radius_m": 12.0,     # これより遠い目撃者は加害者を特定できない(誤通報の源)
        "response": True,
        "speed_m_per_min": 300.0,   # response_min の換算(モデル値。実走行ではない)
        "detain_steps": 0,          # >0 で既存の勾留 seam を使う(★既定 0 = 呼数不変)
        "ward_calls_per_day": 0.0,  # ★参照値のみ(判定には使わない。0 = 未設定)
    },
}

_TOP_INT = ("max_events_per_step", "max_pairs_per_offender")
_BLOCK_SPECS: dict[str, tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = {
    # ブロック名: (float キー, int キー, bool キー)  ※残りは語リスト
    "motive": (("money_low", "w_money", "w_distress", "w_intox", "w_fatigue",
                "w_grievance", "min"), (), ()),
    "target": (("cash_ref", "w_cash", "w_phone", "w_intox", "w_alone", "min"),
               (), ()),
    "guardian": (("w_watcher", "w_koban", "w_trace", "block", "koban_radius_m"),
                 (), ("police_blocks",)),
    "intox": ((), ("steps", "open_hour", "close_hour"), ()),
    "theft": (("pair_prob", "amount", "grievance"), (),
              ("enabled", "supersede_diversity")),
    "brawl": (("pair_prob", "closing_boost", "grievance"),
              ("density_min", "closing_window_min", "injury_density"),
              ("enabled", "require_intox", "night_only")),
    "report": (("victim_base", "witness_base", "kin_bonus", "threshold",
                "alpha_ambiguous", "alpha_danger", "clear_radius_m",
                "speed_m_per_min", "ward_calls_per_day"),
               ("detain_steps",), ("enabled", "response")),
}


# --------------------------------------------------------------------------- #
# cfg 正準化(city_ops.build_cfg / traces.build_cfg と同型: dict / OmegaConf 両対応・
# dotlist の文字列を型強制・未知キーは黙って捨てる = 捏造しない)
# --------------------------------------------------------------------------- #
def _to_plain(raw):
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(raw):
            return OmegaConf.to_container(raw, resolve=True)
    except Exception:                              # noqa: BLE001(旧 config 互換)
        pass
    return raw


def _words(raw, fallback) -> tuple:
    got = _to_plain(raw)
    if isinstance(got, str):
        got = [got]
    out = tuple(str(x).strip() for x in (got or ()) if str(x).strip())
    return out or tuple(fallback)


def _block(raw, defaults: dict, floats: tuple, ints: tuple, bools: tuple) -> dict:
    out = dict(defaults)
    for key, value in dict(_to_plain(raw) or {}).items():
        if key not in out:
            continue                               # 未知キーは捨てる
        if key in bools:
            out[key] = bool(value)
        elif key in floats:
            out[key] = max(0.0, float(value))
        elif key in ints:
            out[key] = int(value)
        else:
            out[key] = _words(value, defaults[key])
    return out


def build_cfg(raw) -> dict:
    """conf の ``incidents_interpersonal`` ブロックを正準化(既定 OFF=現行と完全同一)。"""
    raw = dict(_to_plain(raw) or {})
    cfg: dict = {"enabled": bool(raw.get("enabled", False))}
    for key in _TOP_INT:
        cfg[key] = max(0, int(raw.get(key, DEFAULTS[key])))
    for name, (floats, ints, bools) in _BLOCK_SPECS.items():
        cfg[name] = _block(raw.get(name), DEFAULTS[name], floats, ints, bools)
    cfg["intox"]["steps"] = max(0, int(cfg["intox"]["steps"]))
    cfg["brawl"]["density_min"] = max(0, int(cfg["brawl"]["density_min"]))
    cfg["report"]["detain_steps"] = max(0, int(cfg["report"]["detain_steps"]))
    return cfg


def cfg_of(sim) -> dict:
    """事件設定(初回のみ ``sim.cfg.incidents_interpersonal`` から遅延構築してキャッシュ)。

    simulation.py は別レーン所有なので cfg は本 module が遅延構築する
    (``traces.cfg_of`` / ``city_ops.cfg_of`` と同型)。キャッシュ属性 ``sim.incidentscfg``
    は L1/L2/L3/乱数に一切現れない = 既定 OFF のバイト一致を壊さない。
    """
    got = getattr(sim, "incidentscfg", None)
    if got is None:
        try:
            raw = sim.cfg.get("incidents_interpersonal", None)
        except Exception:                          # noqa: BLE001(旧 config 互換)
            raw = None
        got = build_cfg(raw)
        sim.incidentscfg = got
    return got


def enabled(sim) -> bool:
    """対人事件の収束化(H4)が有効か。既定 OFF=新経路を一切通さない(バイト一致)。"""
    return bool(cfg_of(sim)["enabled"])


def superseded_theft(sim) -> bool:
    """``diversity.tick_crime`` が窃盗の枝を**供給しない**か(世代交代のスイッチ)。

    ★既定 OFF では必ず False = あちらの制御フローが 1 バイトも変わらない
      (``street_life.enabled`` が False のとき ``nuisance_kinds_for`` が既定を
      そのまま返すのと同じ形)。
    """
    if not enabled(sim):
        return False
    cfg = cfg_of(sim)
    return bool(cfg["theft"]["enabled"] and cfg["theft"]["supersede_diversity"])


# --------------------------------------------------------------------------- #
# 決定論の補助
# --------------------------------------------------------------------------- #
def _stable_hash(value: str) -> int:
    """プロセス非依存の安定ハッシュ(rng.py / city_ops.py と同流儀)。"""
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _in_hours(open_hour: int, close_hour: int, sim_min: int) -> bool:
    """分 of day が夜間店舗の窓の中か(``commerce.is_open_window`` と同じ式・時刻の純関数)。"""
    o, c = int(open_hour) % 24, int(close_hour) % 24
    h = (int(sim_min) % 1440) // 60
    if o == c:
        return True
    return (o <= h < c) if o < c else (h >= o or h < c)


def _excluded(sim) -> frozenset:
    """路上生活者の尊厳規約(``diversity._street_life_excluded`` と**同一の集合**)。

    ★``street_life`` が OFF(既定)なら必ず空集合 = 下のフィルタが完全 no-op。
    ★「路上生活者は事件を起こさない」という現実の主張ではなく、**この機構と
      結びつけない**という設計判断である(結びつければシムがスティグマを再生産する)。
      除外は加害者・標的・目撃者・通報者の**すべて**に掛かる。
    """
    from . import street_life as street_life_mod
    return street_life_mod.rough_sleeper_ids(sim)


def _koban_nodes(sim, cfg: dict) -> frozenset:
    """交番・警察署の**近くのノード**の集合(地図の純関数・乱数ゼロ・1 度だけキャッシュ)。

    ★``city_ops.police_posts`` を**呼ばない**: あちらは別レーン所有で、呼ぶと
      ``sim.cityopscfg`` を本 module の ON 経路で作ってしまう(所有権の混線)。
      同じ判定(カテゴリ ∧ 一般名詞のキーワード)をここで独立に持つ = 10 行の重複を
      払って結合を作らない(``diversity.POLICE_OCCS`` の再掲と同じ判断)。
    """
    cache = getattr(sim, "_inc_koban_nodes", None)
    if cache is not None:
        return cache
    gcfg = cfg["guardian"]
    cats = frozenset(gcfg["koban_cats"])
    words = tuple(gcfg["koban_keywords"])
    posts: list[str] = []
    for poi in getattr(sim.city, "poi_list", ()) or ():
        if str(poi.get("cat") or "") not in cats:
            continue
        name = str(poi.get("name") or "")
        if not any(w in name for w in words):
            continue
        node = str(poi.get("node") or "")
        if node:
            posts.append(node)
    radius = float(gcfg["koban_radius_m"])
    near: set[str] = set()
    if posts:
        pxy = []
        for node in sorted(set(posts)):
            try:
                pxy.append(sim.city.node_xy(node))
            except Exception:                      # noqa: BLE001(未知ノード)
                continue
        for node in sorted(sim.city.graph.nodes):
            try:
                nx, ny = sim.city.node_xy(node)
            except Exception:                      # noqa: BLE001
                continue
            for px, py in pxy:
                if (nx - px) ** 2 + (ny - py) ** 2 <= radius * radius:
                    near.add(node)
                    break
    out = frozenset(near)
    sim._inc_koban_nodes = out
    return out


# --------------------------------------------------------------------------- #
# 酩酊 = 夜間店舗に居た痕跡の減衰する印(**一時的で回復する状態**)
# --------------------------------------------------------------------------- #
INTOX_KEY = "_inc_intox_until"


def _mark_intox(sim, cfg: dict, agent, step: int, sim_min: int) -> None:
    """夜間店舗の窓の中で夜間店舗の場所に居る個体に、減衰する酩酊の印を押す(決定論)。"""
    icfg = cfg["intox"]
    ttl = int(icfg["steps"])
    if ttl <= 0:
        return
    if not _in_hours(icfg["open_hour"], icfg["close_hour"], sim_min):
        return
    cats = frozenset(icfg["cats"])
    try:
        pois = sim.city.pois_at_node(agent.node)
    except Exception:                              # noqa: BLE001(未知ノード)
        return
    if not any(str(p.get("cat") or "") in cats for p in pois):
        return
    setattr(agent, INTOX_KEY, int(step) + ttl)


def intox_of(agent, step: int, cfg: dict) -> float:
    """酩酊の度合い [0,1](印の残り寿命の線形減衰)。印が無ければ 0.0(属性を生やさない)。"""
    until = int(getattr(agent, INTOX_KEY, 0) or 0)
    ttl = int(cfg["intox"]["steps"])
    if ttl <= 0 or until <= int(step):
        return 0.0
    return _clip01((until - int(step)) / float(ttl))


# --------------------------------------------------------------------------- #
# RAT の 3 要素(**すべて既存 agent 状態の純関数**。乱数も LLM も traits も読まない)
# --------------------------------------------------------------------------- #
def motive_of(agent, step: int, cfg: dict) -> float:
    """加害者候補の**動機状態** [0,1]。

    ★人格ラベルではない: 手持ちが戻れば・酔いが醒めれば・滞納が解ければ 0 に戻る。
    ★``w_grievance`` は既定 0.0(構成概念を発火判断に食わせない = R1)。
    """
    m = cfg["motive"]
    low = float(m["money_low"])
    money = float(getattr(agent, "money", 0.0))
    w_money = _clip01((low - money) / low) if low > 0.0 else 0.0
    distress = 1.0 if (int(getattr(agent, "arrears_days", 0)) > 0
                       or bool(getattr(agent, "evicted", False))
                       or int(getattr(agent, "bankrupt_until", 0)) > int(step)) else 0.0
    intox = intox_of(agent, step, cfg)
    fatigue = _clip01(float(getattr(agent, "fatigue", 0.0)))
    griev = 0.0
    if float(m["w_grievance"]) > 0.0:              # 既定 0 = states を 1 度も読まない
        griev = _clip01(float((getattr(agent, "states", None) or {})
                              .get("grievance", 0.0)))
    return _clip01(float(m["w_money"]) * w_money
                   + float(m["w_distress"]) * distress
                   + float(m["w_intox"]) * intox
                   + float(m["w_fatigue"]) * fatigue
                   + float(m["w_grievance"]) * griev)


def suitability_of(agent, step: int, cfg: dict, alone: bool) -> float:
    """標的の**適性** [0,1](VIVA の観測可能な代理: 現金・携帯・酩酊・孤立)。"""
    t = cfg["target"]
    ref = float(t["cash_ref"])
    cash = _clip01(float(getattr(agent, "money", 0.0)) / ref) if ref > 0.0 else 0.0
    phone = 1.0 if bool(getattr(agent, "has_phone", False)) else 0.0
    intox = intox_of(agent, step, cfg)
    return _clip01(float(t["w_cash"]) * cash
                   + float(t["w_phone"]) * phone
                   + float(t["w_intox"]) * intox
                   + float(t["w_alone"]) * (1.0 if alone else 0.0))


def guardian_score(sim, cfg: dict, node: str, n_watchers: int,
                   police_here: bool, koban: frozenset) -> float:
    """**監視の目**のスコア。``block`` 以上なら事件は起きない(抽選そのものを行わない)。

    材料は 4 つとも観測可能な世界状態: 同席者の数・同ノードの警察官・交番の近接・
    ``traces``(IF-D)の摘発痕跡。★``traces`` が OFF のランでは
    ``strength_of`` が 0.0 を返すだけ(あちらの state も cfg も作らない読み取り)。
    """
    g = cfg["guardian"]
    if police_here and bool(g["police_blocks"]):
        return float(g["block"]) + 1.0             # 無条件で抑止(G3 執行との接続)
    score = float(g["w_watcher"]) * max(0, int(n_watchers))
    if node in koban:
        score += float(g["w_koban"])
    trace = float(traces_mod.strength_of(sim, node, "enforcement"))
    if trace > 0.0:
        score += float(g["w_trace"]) * trace
    return score


# --------------------------------------------------------------------------- #
# state(sim 側。**OFF では 1 つも生えない**)
# --------------------------------------------------------------------------- #
def _state(sim) -> dict:
    st = getattr(sim, "_inc_state", None)
    if st is None:
        st = {"schema": SCHEMA, "candidates": 0, "pairs_seen": 0,
              "blocked_by_guardian": 0, "draws": 0, "incidents": 0,
              "last_step": -1, "dropped": 0, "weight_sum": 0.0,
              "weight_by_kind": {}, "by_kind": {},
              "reports": 0, "reports_nonurgent": 0, "reports_unclear": 0,
              "reports_self": 0, "responses": 0, "responses_unstaffed": 0,
              "detained": 0, "injuries": 0, "closing_incidents": 0,
              "victims": {}, "nodes": {}}
        sim._inc_state = st
    return st


def _bump(table: dict, key, n: int = 1) -> None:
    table[str(key)] = int(table.get(str(key), 0)) + int(n)


# --------------------------------------------------------------------------- #
# ★★★ 乱数を引く**唯一の**関数 ★★★
#
# ここが本 module の「エージェントドリブン」の機械的な保証点である:
#   - 引数 ``pairs`` は**共在ペアの列**であり、本体の**最初の文**が空なら return する
#     = 共在が無ければ乱数は 1 本も引かれない(人口にも時間にも直接掛からない)。
#   - 抽選は「ペアごとの条件付き確率の合成」で、選ばれるのも**ペア**である
#     (「当たってから被害者を探す」現行 theft の順序を逆転させた)。
#   - stream は新 named stream ``"incident_pair"``(既存 draw 順に挿入しない)。
#
# tests/test_incidents_interpersonal.py が AST で
#   「本 module 中の hub.stream 呼び出しは 1 か所・この関数の中・先頭は空ペア return」
# を機械固定する。
# --------------------------------------------------------------------------- #
def _pair_draw(sim, offender, pairs: list, step: int):
    """共在ペア列から 1 組を選ぶ(選ばれなければ None)。**空なら乱数を引かない**。"""
    if not pairs:
        return None
    total = 0.0
    for _t, p, _ctx in pairs:
        total += float(p)
    if total <= 0.0:
        return None
    r = float(sim.hub.stream("incident_pair", int(offender.id), int(step)).random())
    if r >= total:
        return None
    acc = 0.0
    for item in pairs:
        acc += float(item[1])
        if r < acc:
            return item
    return pairs[-1]


# --------------------------------------------------------------------------- #
# 事件の適用(窃盗 / 喧嘩)
# --------------------------------------------------------------------------- #
def _pre(ctx: dict) -> dict:
    """L1 に同梱する**前兆状態**(内生性の機械検証。数値は payload だけに出る)。"""
    return {"guardians": int(ctx["guardians"]), "density": int(ctx["density"]),
            "intox": round(float(ctx["intox"]), 3),
            "closing": bool(ctx["closing"]),
            "motive": round(float(ctx["motive"]), 3),
            "suitability": round(float(ctx["suitability"]), 3)}


def _apply_theft(sim, cfg: dict, st: dict, offender, victim, ctx: dict,
                 step: int, sim_min: int) -> dict:
    """窃盗 1 件(既存 ``crime`` 種で出す = 世代交代)。金の扱いは現行と完全に同型。"""
    tcfg = cfg["theft"]
    amount = float(tcfg["amount"])
    stolen = min(float(victim.money), amount)
    victim.money = max(0.0, float(victim.money) - amount)
    payload = {"kind": "theft", "victim": int(victim.id), "offender": int(offender.id),
               "amount": round(stolen, 1), "src": "rat", "node": str(offender.node)}
    payload.update(_pre(ctx))
    # IF-E2(既定 OFF=キーなし): 窃盗は相互合意が無いので**取引ではない**(SNA 2008 §3.98)。
    # 現行 diversity.tick_crime と**同じ関数・同じ引数の意味**(丸める前の実減少額)で渡す。
    payee = sfc_mod.on_theft(sim, stolen)
    if payee is not None:
        payload["payee"] = payee
    sim.logger.log(Event(step=int(step), sim_min=int(sim_min),
                         agent_id=int(offender.id), kind=THEFT_KIND,
                         x=offender.x, y=offender.y, payload=payload))
    factor_update.on_crime(victim, float(tcfg["grievance"]), cause="crime",
                           step=int(step), sim_min=int(sim_min), logger=sim.logger)
    victim.remember(sentence("theft_victim"))
    offender.remember(sentence("theft_offender"))
    return payload


def _apply_brawl(sim, cfg: dict, st: dict, a, b, ctx: dict,
                 step: int, sim_min: int) -> dict:
    """喧嘩 1 件(**双方向** = 加害と被害が対称。agent_id は id の小さい側)。"""
    bcfg = cfg["brawl"]
    first, second = (a, b) if int(a.id) <= int(b.id) else (b, a)
    both_drunk = (intox_of(a, step, cfg) > 0.0 and intox_of(b, step, cfg) > 0.0)
    severity = 2 if (both_drunk and int(ctx["density"]) >= int(bcfg["injury_density"])) else 1
    payload = {"other": int(second.id), "node": str(first.node),
               "severity": int(severity)}
    payload.update(_pre(ctx))
    sim.logger.log(Event(step=int(step), sim_min=int(sim_min),
                         agent_id=int(first.id), kind=BRAWL,
                         x=first.x, y=first.y, payload=payload))
    for who in (first, second):
        factor_update.on_crime(who, float(bcfg["grievance"]), cause="nuisance",
                               step=int(step), sim_min=int(sim_min), logger=sim.logger)
        who.remember(sentence("brawl"))
        if severity >= 2:
            _injure(sim, who, severity, step, sim_min)
            st["injuries"] += 1
    return payload


def _injure(sim, agent, severity: int, step: int, sim_min: int) -> None:
    """負傷(S1-S2)。**H1 レーンへの soft 依存**: あちらが ``health.on_injury`` を
    生やしたら自動で繋がり、無ければ記憶 1 行だけで終わる(health.py は 1 バイトも触らない)。
    """
    from . import health as health_mod
    hook = getattr(health_mod, "on_injury", None)
    if callable(hook):
        try:
            hook(sim, agent, int(severity), step=int(step), sim_min=int(sim_min))
        except Exception:                          # noqa: BLE001(あちらの契約が違っても壊さない)
            pass
    agent.remember(sentence("injury"))


# --------------------------------------------------------------------------- #
# 通報層(110 番 = 目撃者/被害者の**行為**。新しい乱数を 1 本も引かない完全決定論)
# --------------------------------------------------------------------------- #
def _kin(a, b) -> bool:
    """関係者(家族・同居人・恋人)か = 通報意思の倍率 > 1 の根拠(既存の台帳を読むだけ)。"""
    if int(getattr(a, "partner_id", -1) or -1) == int(b.id):
        return True
    mates = getattr(a, "housemates", None) or ()
    if int(b.id) in {int(x) for x in mates}:
        return True
    ha, hb = getattr(a, "household_id", None), getattr(b, "household_id", None)
    return bool(ha is not None and ha == hb)


def report_p_ind(cfg: dict, is_victim: bool, kin: bool, n_witnesses: int,
                 danger: bool) -> float:
    """**個人**が通報する確率 [0,1](決定論の純関数)。

    ★Darley & Latané 型の人数減衰は「曖昧で軽微」な状況にだけ掛ける
      (``alpha_ambiguous`` ≈ 0.4)。危険事態では ``alpha_danger`` = 0 = 人数で落ちない
      (メタ分析。介入が規範)。**掛かるのは個人の確率であって集団の確率ではない**
      — この区別が計画書 §1 の「集団レベル P(誰かが行動)≈0.85-0.95」と整合する
      唯一の入れ方である(個人の p を n^-α で落としても、下の :func:`report_p_group`
      は n が増えるほど上がる = 実 CCTV 研究の所見そのもの)。
    ★被害者は**傍観者ではない**ので人数減衰を掛けない。
    """
    r = cfg["report"]
    base = float(r["victim_base"]) if is_victim else float(r["witness_base"])
    if kin:
        base += float(r["kin_bonus"])
    base = _clip01(base)
    if is_victim:
        return base
    alpha = float(r["alpha_danger"]) if danger else float(r["alpha_ambiguous"])
    n = max(1, int(n_witnesses))
    return _clip01(base * (n ** (-alpha)) if alpha > 0.0 else base)


def report_p_group(probs) -> float:
    """**集団**の「誰かが通報する」確率 = ``1 − Π(1−p_i)``(解析式・乱数ゼロ)。

    ★これが実測アンカー(集団レベル 0.85-0.95・実 CCTV 研究の介入率 91%)と
      突き合わせる量である。個人の p を人数で落としても、居合わせた人が増えれば
      集団の確率は上がる(既定値では n=3 で 0.83・n=5 で 0.90・n=8 で 0.94)。
    """
    q = 1.0
    for p in probs:
        q *= (1.0 - _clip01(float(p)))
    return _clip01(1.0 - q)


def _delay_min(reporter, step: int) -> int:
    """通報までの時間[分]。実測分布(46% <1 分 / 29% 1-5 分 / 25% >5 分)の
    決定論写像((通報者 id, step) の安定ハッシュ = ``city_ops`` の倒れ判定と同流儀)。"""
    h = _stable_hash(f"report/{int(reporter.id)}/{int(step)}") % 100
    if h < 46:
        return 0
    if h < 75:
        return 1 + (h % 5)
    return 6 + (h % 10)


def _report_and_respond(sim, cfg: dict, st: dict, bud: list, kind: str,
                        victim, offender, witnesses: list, node: str,
                        danger: bool, self_eligible: bool,
                        step: int, sim_min: int) -> None:
    """事件 1 件に対する通報(と応答)。通報者が居なければ**1 件も出ない**。

    ★``self_eligible``: 窃盗の被害者は自分で通報できる(財布が無いことに気づく)が、
      喧嘩の**当事者は通報者にならない**(その step の行動枠をもみ合いに使い切っている)。
      この線引きがあるので Darley & Latané 型の人数減衰に**実際に効く場面**ができる:
      「負傷の無いもみ合い(曖昧・軽微)を、居合わせた人数が多いほど誰も通報しない」。
    """
    rcfg = cfg["report"]
    if not rcfg["enabled"]:
        return
    thr = float(rcfg["threshold"])
    clear = float(rcfg["clear_radius_m"])
    n_w = len(witnesses)
    cands: list = []
    if victim is not None and self_eligible:
        cands.append((victim, True))
    cands.extend((w, False) for w in witnesses)
    scored: list = []
    for who, is_victim in cands:
        kin = (not is_victim and victim is not None and _kin(who, victim))
        scored.append((report_p_ind(cfg, is_victim, kin, n_w, danger),
                       int(who.id), who, is_victim))
    if not scored:                                  # ★居合わせた人が居なければ通報も無い
        return
    if report_p_group(p for p, _i, _w, _v in scored) < thr:
        return                                      # 誰も動かない(傍観者効果が勝った回)
    # 通報するのは**最も動機の強い 1 人**(同点は id 昇順 = 決定論)
    best = min(scored, key=lambda t: (-t[0], t[1]))
    _p, _id, reporter, is_victim = best
    # ★**誰が加害者かを名乗れるか**は「見えたか」で決まる(欠測を偽の値で埋めない):
    #   - 喧嘩の目撃者/当事者は相手を見ている = 特定できる
    #   - 窃盗の被害者は**取られる瞬間を見ていない**(事後に気づく)= 特定できない
    #   - 現場から ``clear_radius_m`` より遠い目撃者も特定できない
    if offender is None:
        identified = False
    elif str(kind) == BRAWL:
        identified = True
    else:
        identified = bool(not is_victim
                          and _dist(reporter, offender) <= clear)
    unclear = not identified
    # ★緊急/非緊急も内生: 危険事態(負傷)か、**進行中を目撃した**通報だけが緊急。
    #   被害者が事後に気づいて通報する窃盗は非緊急(現実の 110 番の 16% が非緊急という
    #   構造をレートで注入せず、通報の出どころの違いから出す)。
    urgent = bool(danger or (not is_victim and identified))
    payload = {"incident": str(kind), "node": str(node),
               "victim": (int(victim.id) if victim is not None else None),
               "offender": (None if (unclear or offender is None)
                            else int(offender.id)),
               "self_report": bool(is_victim), "witnesses": int(n_w),
               "urgent": bool(urgent), "unclear": bool(unclear),
               "delay_min": int(_delay_min(reporter, step))}
    if not _log(sim, st, bud, Event(step=int(step), sim_min=int(sim_min),
                                    agent_id=int(reporter.id), kind=REPORT,
                                    x=reporter.x, y=reporter.y, payload=payload)):
        return
    st["reports"] += 1
    if not urgent:
        st["reports_nonurgent"] += 1
    if unclear:
        st["reports_unclear"] += 1
    if is_victim:
        st["reports_self"] += 1
    reporter.remember(sentence("report"))
    if rcfg["response"]:
        _respond(sim, cfg, st, bud, reporter, offender, kind, node, step, sim_min)


def _dist(a, b) -> float:
    dx, dy = float(a.x) - float(b.x), float(a.y) - float(b.y)
    return (dx * dx + dy * dy) ** 0.5


def _respond(sim, cfg: dict, st: dict, bud: list, caller, offender, kind: str,
             node: str, step: int, sim_min: int) -> None:
    """通報への応答(**本 module 内の簡約実装**)。当直不在なら正直な無人マーカー。"""
    rcfg = cfg["report"]
    officer = None
    best = None
    for a in sim.agents:
        if a.occupation not in POLICE_OCCS or a.loc == "outside" or a.sleeping:
            continue
        d = _dist(a, caller)
        key = (d, int(a.id))
        if best is None or key < best:
            best, officer = key, a
    speed = float(rcfg["speed_m_per_min"]) or 1.0
    payload = {"node": str(node), "caller": int(caller.id), "incident": str(kind),
               "response_min": (round(best[0] / speed, 1) if best is not None else None),
               "unstaffed": bool(officer is None), "detained": False}
    detain = int(rcfg["detain_steps"])
    if officer is not None and offender is not None and detain > 0 \
            and offender.node == officer.node:
        # 既存の勾留 seam(制度深化2)を使う。**既定 detain_steps=0 では 1 度も通らない**
        # (勾留は発火権を止める = LLM 呼数が動くため、既定では 1 も動かさない)。
        offender.detained_until = int(step) + detain
        payload["detained"] = True
        st["detained"] += 1
    if not _log(sim, st, bud, Event(step=int(step), sim_min=int(sim_min),
                                    agent_id=(int(officer.id) if officer is not None else -1),
                                    kind=RESPONSE,
                                    x=(officer.x if officer is not None else caller.x),
                                    y=(officer.y if officer is not None else caller.y),
                                    payload=payload)):
        return
    st["responses"] += 1
    if officer is None:
        st["responses_unstaffed"] += 1
    else:
        officer.remember(sentence("response"))


# --------------------------------------------------------------------------- #
# 単一作用点(scheduler が共在索引を張った直後に 1 回だけ呼ぶ)
# --------------------------------------------------------------------------- #
def _log(sim, st: dict, bud: list, ev) -> bool:
    """L1 件数の上限(暴走時の膨張対策。超過分は**捨てて数える**= city_ops と同流儀)。"""
    if bud[0] <= 0:
        st["dropped"] += 1
        return False
    bud[0] -= 1
    sim.logger.log(ev)
    return True


def phase(sim, step: int, sim_min: int) -> None:
    """毎 step の単一作用点。**既定 OFF は即 return**(乱数も state も属性も 0)。

    処理の順(すべて決定論・id 昇順):
      1. 酩酊の印(夜間店舗に居る個体)を押す = **世界の観測から作る一時状態**
      2. 動機のある候補を絞る(``sim.agents`` を走るのは**ここまで**)
      3. 候補ごとに ``sim.percept_index`` から共在者を取り、
         (標的適性 × 監視の減衰)で**ペアの条件付き確率**を組む
      4. :func:`_pair_draw` で 1 組を選ぶ(**空なら乱数を引かない**)
      5. 事件を適用し、通報(と応答)を回す
    """
    if not enabled(sim):
        return
    cfg = cfg_of(sim)
    st = _state(sim)
    st["last_step"] = max(int(st["last_step"]), int(step))
    bud = [int(cfg["max_events_per_step"])]
    excluded = _excluded(sim)                      # 尊厳規約(street_life OFF なら空集合)
    koban = _koban_nodes(sim, cfg)
    theft_on = bool(cfg["theft"]["enabled"])
    brawl_on = bool(cfg["brawl"]["enabled"])
    if not theft_on and not brawl_on:
        return
    icfg = cfg["intox"]
    night = _in_hours(icfg["open_hour"], icfg["close_hour"], sim_min)
    closing = _closing_window(cfg, sim_min)
    radius = float(sim.cfg.world.perception_radius_m)
    index = getattr(sim, "percept_index", None)
    scan = index if index is not None else sim.agents
    # ---- 1. 酩酊の印 + 2. 動機のある候補(sim.agents を走るのはここまで)------------- #
    candidates: list = []
    for a in sorted(sim.agents, key=lambda x: int(x.id)):
        if a.loc == "outside" or a.sleeping:
            continue
        if int(a.id) in excluded:                  # 尊厳規約: 加害者候補にしない
            continue
        _mark_intox(sim, cfg, a, step, sim_min)
        if a.occupation in POLICE_OCCS:            # 警察官は加害しない(現行と同じ線引き)
            continue
        if int(step) < int(getattr(a, "detained_until", 0) or 0):
            continue
        if motive_of(a, step, cfg) >= float(cfg["motive"]["min"]):
            candidates.append(a)
    st["candidates"] += len(candidates)
    police_nodes = {a.node for a in sim.agents
                    if a.occupation in POLICE_OCCS and a.loc != "outside"
                    and not a.sleeping}
    cap = int(cfg["max_pairs_per_offender"])
    bcfg = cfg["brawl"]
    theft_p = float(cfg["theft"]["pair_prob"])
    # ★閉店 +1 時間の実証弾性 +16%(§3)= 帯の中でだけ喧嘩の条件付き確率に掛かる。
    brawl_p = float(bcfg["pair_prob"])
    if closing:
        brawl_p *= 1.0 + float(bcfg["closing_boost"])
    for offender in candidates:
        # ---- 3. 共在ペア(**唯一の共在索引** sim.percept_index から)------------------ #
        others = [o for o in hearers_of(offender, scan, radius)
                  if int(o.id) not in excluded
                  and o.occupation not in POLICE_OCCS]
        if not others:
            continue                               # ★共在が無ければ以降に乱数は無い
        density = len(others)
        police_here = offender.node in police_nodes
        motive = motive_of(offender, step, cfg)
        off_intox = intox_of(offender, step, cfg)
        pairs: list = []
        for target in others[:max(0, cap)] if cap > 0 else others:
            n_watchers = max(0, density - 1)       # 標的以外の同席者 = 監視の目
            g = guardian_score(sim, cfg, offender.node, n_watchers, police_here, koban)
            block = float(cfg["guardian"]["block"])
            # ★``block <= 0`` は「常に監視されている」対照条件(事件が 1 件も起きない世界)。
            if block <= 0.0 or g >= block:
                st["blocked_by_guardian"] += 1
                continue                           # ★監視が閾値以上 = **抽選そのものを行わない**
            atten = 1.0 - (g / block)
            alone = (density <= 1)
            suit = suitability_of(target, step, cfg, alone)
            if suit < float(cfg["target"]["min"]):
                continue
            ctx = {"guardians": n_watchers, "density": density,
                   "intox": max(off_intox, intox_of(target, step, cfg)),
                   "closing": closing, "motive": motive, "suitability": suit}
            if theft_on and theft_p > 0.0:
                p = theft_p * motive * suit * atten
                if p > 0.0:
                    pairs.append((target, p, dict(ctx, kind="theft")))
            # ★``night or closing``: 閉店放出の帯(05:00〜)は夜間店舗の窓の**外**なので、
            #   窓だけで絞ると +16% の弾性が原理的に測れない(実装中に気づいた点)。
            if brawl_on and brawl_p > 0.0 and density >= int(bcfg["density_min"]) \
                    and ((not bcfg["night_only"]) or night or closing):
                t_intox = intox_of(target, step, cfg)
                if (not bcfg["require_intox"]) or off_intox > 0.0 or t_intox > 0.0:
                    p = brawl_p * motive * max(off_intox, t_intox) * atten
                    if p > 0.0:
                        pairs.append((target, p, dict(ctx, kind=BRAWL)))
        st["pairs_seen"] += len(pairs)
        for _t, _p, _c in pairs:                   # 較正の分母を種別に残す(§7 の物差し)
            st["weight_sum"] += float(_p)
            key = str(_c["kind"])
            st["weight_by_kind"][key] = float(st["weight_by_kind"].get(key, 0.0)) + float(_p)
        # ---- 4. 抽選(**空なら乱数を引かない** = 本 module の唯一の stream)------------ #
        if pairs:
            st["draws"] += 1
        picked = _pair_draw(sim, offender, pairs, step)
        if picked is None:
            continue
        target, _p, ctx = picked
        kind = str(ctx["kind"])
        # ---- 5. 事件 + 通報 --------------------------------------------------------- #
        witnesses = [o for o in others if int(o.id) != int(target.id)]
        if kind == "theft":
            if bud[0] <= 0:
                st["dropped"] += 1
                continue
            bud[0] -= 1
            _apply_theft(sim, cfg, st, offender, target, ctx, step, sim_min)
            victim, danger = target, False
        else:
            if bud[0] <= 0:
                st["dropped"] += 1
                continue
            bud[0] -= 1
            pay = _apply_brawl(sim, cfg, st, offender, target, ctx, step, sim_min)
            victim = target
            danger = int(pay["severity"]) >= 2
        st["incidents"] += 1
        _bump(st["by_kind"], kind)
        _bump(st["victims"], int(target.id))
        _bump(st["nodes"], str(offender.node))
        if closing:
            st["closing_incidents"] += 1
        _note_crime_node(sim, offender.node)
        for w in witnesses:
            w.remember(sentence("witness"))
        # ★喧嘩の当事者は通報者にならない(もみ合っている)= 目撃者の傍観者効果が効く場面
        _report_and_respond(sim, cfg, st, bud, kind, victim, offender,
                            witnesses, str(offender.node), danger,
                            kind == "theft", step, sim_min)


def _closing_window(cfg: dict, sim_min: int) -> bool:
    """夜間店舗の**閉店から ``closing_window_min`` 分**の帯か(時刻の純関数)。

    ★閉店 +1 時間で喧嘩が +16% という実証弾性(§3)の作用点。帯の中では
      ``brawl.pair_prob`` に ``(1 + closing_boost)`` が掛かる。
    """
    win = int(cfg["brawl"]["closing_window_min"])
    if win <= 0:
        return False
    close_min = (int(cfg["intox"]["close_hour"]) % 24) * 60
    m = int(sim_min) % 1440
    d = (m - close_min) % 1440
    return 0 <= d < win


def _note_crime_node(sim, node: str) -> None:
    """危険地帯(``diversity`` の治安回避)へ 1 件積む。**あちらが OFF でも無害**。"""
    from . import diversity as diversity_mod
    diversity_mod.note_crime_node(sim, node)


# --------------------------------------------------------------------------- #
# 観測(summary.json の "incidents_interpersonal" キー。**OFF ではキー自体を出さない**)
# --------------------------------------------------------------------------- #
def provenance(sim) -> dict | None:
    """``summary.json`` の ``incidents_interpersonal`` キー(既定 OFF は None)。

    ★較正の分母を**共在機会比**で出す(§7「分母の罠」): 人口あたりの件数だけでなく
      ``pairs_seen`` / ``weight_sum`` / ``draws`` を並べる = 来街ピークで事件が過剰に
      なっていないかを事後に判定できる。
    ★パターンの検収量(場所集中 = ``node_top_share`` / ``node_hhi``、反復被害 =
      ``victim_repeat_rate`` / ``victim_hhi``)もここに出す(判定式は解析側/テスト)。
    """
    if not enabled(sim):
        return None
    cfg = cfg_of(sim)
    out: dict = {"schema": SCHEMA,
                 "theft_pair_prob": float(cfg["theft"]["pair_prob"]),
                 "brawl_pair_prob": float(cfg["brawl"]["pair_prob"]),
                 "closing_boost": float(cfg["brawl"]["closing_boost"]),
                 "guardian_block": float(cfg["guardian"]["block"])}
    st = getattr(sim, "_inc_state", None)
    if st is None:                                 # ON だが 1 度も phase が走っていない
        out.update({"candidates": 0, "pairs_seen": 0, "blocked_by_guardian": 0,
                    "draws": 0, "incidents": 0, "dropped": 0, "weight_sum": 0.0,
                    "weight_by_kind": {}, "by_kind": {}, "closing_incidents": 0,
                    "injuries": 0, "reports": 0, "reports_nonurgent": 0,
                    "reports_unclear": 0, "reports_self": 0, "responses": 0,
                    "responses_unstaffed": 0, "detained": 0,
                    "victims_unique": 0, "nodes_unique": 0,
                    "victim_repeat_rate": 0.0, "victim_hhi": 0.0, "node_hhi": 0.0,
                    "node_top_share": 0.0, "nonurgent_rate": 0.0,
                    "days_elapsed": 0.0, "per_person_day": 0.0, "per_pair_step": 0.0})
        return out
    victims = st["victims"]
    nodes = st["nodes"]
    n_inc = int(st["incidents"])
    out.update({
        "candidates": int(st["candidates"]), "pairs_seen": int(st["pairs_seen"]),
        "blocked_by_guardian": int(st["blocked_by_guardian"]),
        "draws": int(st["draws"]), "incidents": n_inc,
        "dropped": int(st["dropped"]),
        "weight_sum": round(float(st["weight_sum"]), 9),
        "weight_by_kind": {k: round(float(v), 9)
                           for k, v in sorted(st["weight_by_kind"].items())},
        "by_kind": {k: int(v) for k, v in sorted(st["by_kind"].items())},
        "closing_incidents": int(st["closing_incidents"]),
        "injuries": int(st["injuries"]),
        "reports": int(st["reports"]),
        "reports_nonurgent": int(st["reports_nonurgent"]),
        "reports_unclear": int(st["reports_unclear"]),
        "reports_self": int(st["reports_self"]),
        "responses": int(st["responses"]),
        "responses_unstaffed": int(st["responses_unstaffed"]),
        "detained": int(st["detained"]),
        "victims_unique": len(victims), "nodes_unique": len(nodes),
        "victim_repeat_rate": (round(1.0 - len(victims) / n_inc, 4) if n_inc else 0.0),
        "victim_hhi": _hhi(victims), "node_hhi": _hhi(nodes),
        "node_top_share": (round(max(int(v) for v in nodes.values()) / n_inc, 4)
                           if nodes and n_inc else 0.0),
        "nonurgent_rate": (round(int(st["reports_nonurgent"]) / int(st["reports"]), 4)
                           if st["reports"] else 0.0),
    })
    # 較正の分母(§7): 人口比は**参考**で、共在機会比(pairs_seen / weight_sum)が本命。
    dt = float(getattr(getattr(sim, "clock", None), "step_minutes", 10) or 10)
    days = max(1.0 / 144.0, (int(st["last_step"]) + 1) * dt / 1440.0)
    n_agents = max(1, len(sim.agents))
    out["days_elapsed"] = round(days, 4)
    out["per_person_day"] = round(n_inc / (n_agents * days), 8)
    out["per_pair_step"] = (round(n_inc / int(st["pairs_seen"]), 8)
                            if int(st["pairs_seen"]) else 0.0)
    return out


def _hhi(table: dict) -> float:
    """ハーフィンダール指数 Σ(share²)。一様なら 1/K に近づく(集中で大きくなる)。"""
    total = sum(int(v) for v in table.values())
    if total <= 0:
        return 0.0
    return round(sum((int(v) / total) ** 2 for v in table.values()), 6)
