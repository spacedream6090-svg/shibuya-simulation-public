"""遺失物ループ H3 = **完全内生の事件**(``lost_property``・**既定 OFF**)。

正典
----
- ``docs/plans/body-incident-layer-plan.md`` **§3 の「遺失物」行**(様式 = **完全内生**(本命)。
  落とす(移動/飲酒/混雑で増)→ 拾う → **届ける/無視/着服**(全て行為)→ 交番/駅事務室 →
  報労金 5-20% / 3 か月時効。検証 = **品目別返還率**(傘 1% / 財布 60-80% / 携帯 87%)・
  都実測 = 拾得 454 万件/年)
- 同 **§4 chance_event の溶解**(windfall → 「拾得者になる」。★**拾得金は必ず誰かの drop から**
  = 貨幣保存則(IF-E ゼロ和検査)と整合)
- 同 **§6-2 ユーザー決定**(原文): 「あらゆることに運は見出せると思うし、それは世界の
  アルゴリズムがそうであったからではなくて、人の行動によってそのほとんどが引き起こされて
  いると思う。なので運をコードで表現、作成するのではなくて、エージェントの行動であったり
  によってたまたま発生するようにしたい」
- 同 **§3 の共在判定の一本化**(収束判定は唯一の索引持ち ``sim.percept_index`` に集約 =
  O(n²) 回避)/ **L1 は 1 行 + 前兆状態同梱**

何を解く問題か
--------------
現行の ``chance.py`` は「windfall(臨時収入 = 拾得・還付・当選小口)」を **カタログからの
抽選**で作る。拾ったお金がどこから来たのかは世界のどこにも書かれていない = 誰の財布からも
出ていない金が increments する。これは §0 の三層因果(physics / agent / device)の外に
「運」というカテゴリを 1 つ残していることの帰結である。

本 module は同じ現象を**構造で**作る。**拾得金は必ず誰かの drop から出る**:

    落とす(physics: 移動・飲酒・混雑のハザード)
      → 気づく(agent: 遺失届)          ／  拾う(agent: 共在した誰かの選択)
                                          → 届ける(agent)/ 無視(agent)/ 着服(agent = 犯罪)
      → 返還(owner → finder に報労金)  ／  3 か月時効で拾得者取得

「拾った人が得をする」も「財布を失くして損をする」も、**別々の抽選ではなく 1 つの出来事の
両側**になる。これが §4 の「chance 溶解の第一歩」であり、貨幣保存(IF-E)と整合する唯一の形。

較正(すべて実測アンカーつき。数字の出どころをコメントに残す義務)
------------------------------------------------------------------
- **落とす頻度**: 警視庁(東京都)の拾得物受理件数 ≈ **454 万件/年** ≈ 1.24 万件/日。
  東京都の昼間人口 ≈ 1,600 万人 → **届出 ≈ 7.8e-4 件/人日**。届出は「拾われて **かつ**
  届けられた」ものなので、落とした総数はこれより多い:

      落下数 = 届出数 ÷ (P(拾われる) × P(届けられる|拾われた))
             ≈ 7.8e-4 ÷ (0.55 × 0.75) ≈ 1.9e-3 件/人日

  渋谷は来街者密度の高い盛り場(§7「分母の罠」= 区の夜間人口ではなく**共在機会**で較正する)
  なので ×1.5 → **≈ 2.9e-3 件/人日**を全品目合計の目標に置く。``BASE_DAILY`` は共変量を
  1 つも掛けない**基底**(合計 2.0e-3)で、渋谷の来街者像の曝露分布
  (移動中 40% / 混雑 20% / 飲酒帯の店内 5% / 雨天日 30% / 店内滞在 20%)で期待値化すると
  **実現 2.92e-3 件/人日**になる = 目標に一致する(``tests`` が実コードの倍率関数で固定)。
- **品目構成**: 警視庁の受理内訳の桁(傘 ≈ 28 万本/年・携帯 ≈ 36 万台/年 = それぞれ全体の
  6% / 8% 程度で、証明書類・衣類等の「その他」が大半)に合わせて
  傘 6% / 携帯 8% / 財布 8% / その他 78% とした。
- **品目別返還率(本レーンの検証ターゲット)**: 東京都遺失物センターの公表値
  = 携帯電話 ≈ **87%** / 財布(現金入り)≈ **60-80%** / 傘 ≈ **1%**。
  本実装ではこれを **「届けられた物のうち持ち主に返った割合」** と定義する(実測の定義と同じ:
  分母は拾得受理件数)。返還は 2 段の連言 ``P(持ち主が気づく) × P(持ち主が引き取りに来る)``
  で作り、``NOTICE_PROB`` × ``CLAIM_PROB`` を較正した。個体差は ``internal_locus`` の
  倍率が乗るので**実現値は名目の積より少し下がる**。2000 体の合成個体での実測
  (``tests/test_lost_property.py`` が帯で固定する):

      ============ ============== ============ ==========
      品目          名目 (P_n×P_c)  実現返還率    実測アンカー
      ============ ============== ============ ==========
      傘            0.35 × 0.03     **0.0115**   ≈ 0.01
      財布          0.98 × 0.73     **0.702**    0.60-0.80
      携帯          0.99 × 0.92     **0.875**    0.87
      その他        0.60 × 0.36     0.230        (実測アンカーなし)
      ============ ============== ============ ==========
- **報労金 5-20%**: 遺失物法第 28 条(物件の返還を受ける者は、その価格の 100 分の 5 以上
  100 分の 20 以下に相当する額の報労金を拾得者に支払わなければならない)。
- **3 か月時効**: 遺失物法第 7 条(公告から 3 か月以内に所有者が判明しないときは
  拾得者が所有権を取得する)。既定 ``statute_steps`` = 90 日 × 144 step。

R1 ドクトリン
-------------
- 既定 ``lost_property.enabled: false`` では ``phase`` が即 return し、**L1 に 1 件も出ず・
  記憶に 1 行も入らず・agent に属性が生えず・sim に state が生えず・プロンプトが 1 バイトも
  変わらず・乱数 stream を 1 本も引かない**(= ゴールデン L1 バイト一致。第95 IF-C /
  第96 IF-D / 第97 IF-E2 と同じ「OFF ではキー自体を作らない」流儀)。
- **新乱数は新 named stream 1 本だけ**: ``"lost_drop"``((agent.id, step) 個体別キー =
  編成順非依存)。**落とすかどうか**だけが確率事象で、それ以外の分岐
  (気づく / 拾う / 届ける / 着服 / 引き取る)は **1 粒も乱数を引かない**。
- **generate() の呼び出しサイトを 1 つも作らない**(LLM 追加呼ゼロ = k 非依存)。
  当事者に届くのは ``agent.remember`` の**定型 1 行**だけで、既存の記憶欄に並ぶ
  (プロンプトの**節**は 1 つも増えない = traces / street_life と同じ帯域)。
- **no-fingerprint**: 記憶の 1 行は ``(出来事, 品目語)`` の純関数(``TEXT`` / ``ITEM_WORD``)で、
  金額・件数・確率・config・実験条件・k・機構語は 1 文字も入らない。
  全実験条件で同一バイト列になることを tests が機械固定する。
- **共在は ``sim.percept_index`` だけを使う**(§3「共在判定の一本化」)。落下地点に**探針**
  (``_Probe`` = 既存の知覚規約 ``hearers_of`` が要求する duck-type)を立てて近傍 9 セルだけを
  引く = **新しい全対全スキャンを 1 つも足さない**。
- **25 万スケール**: 1 step のコストは (a) 在場者 1 人 1 draw(``diversity.tick_crime`` と
  同じ前例)と (b) 落ちている物 1 件あたり索引 1 引き(``max_per_step`` で上限)。
  毎 step の全対全走査はゼロ。

決定論の作り方(**乱数ではなく純関数**で個体差を作る)
------------------------------------------------------
落とす以外の分岐は「性格(traits)・規範状態(states)・金額・監視者の有無」の決定論関数で
決める(計画書 §3 の要求)。ただし性格だけを閾値で切ると (i) ``flat_traits`` 実験条件で
全員同一 → 0% か 100% に退化し、(ii) 較正が trait 分布の形に張り付いて脆くなる。そこで

    通過する ⟺ jitter(個体, 物, 用途) < p_base(品目) × 倍率(traits, states, 金額, 監視者)

という**ハザード型**にした。``jitter`` は ``(seed, agent_id, item_id, 用途)`` の
**sha256 純関数**(= ``city_ops._collapse_gate`` のハッシュゲートと同じ前例)で、
**乱数 stream を 1 本も消費しない**。これで

  - 全体の通過率は ``p_base`` に張り付く(= 実測帯に較正できる)
  - 個体差は traits / states / 金額 / 監視者が**乗算で**作る(= 性格が効く)
  - resume を跨いでも・編成順が変わっても同じ答えになる(= 純関数)

の 3 つが同時に立つ。

着服を「犯罪」としてどう記録するか(**意図的な逸脱**)
------------------------------------------------------
計画書は着服を「犯罪イベントとして記録」と書いているが、本 module は既存の ``crime``
イベントを**出さない**。理由は 2 つ:
  1. ``crime`` は ``diversity.tick_crime`` が窃盗レート 2.8e-5/人日 で**実測較正済み**の
     系列で、そこに別機構の件数を混ぜると条例・執行の効き目が測れなくなる
     (``street_life`` が客引きを ``nuisance_kinds`` から**外した**のと同じ線引き)。
  2. ``traces`` の ``trouble`` 痕跡は ``crime`` / ``nuisance`` を源にしているので、
     着服を ``crime`` で出すと「場所の性格」に混線する。
代わりに ``lost_keep`` の payload に ``offense``(占有離脱物横領)と ``guardians``
(**監視者数** = RAT の語彙)を載せる。「犯罪である」ことは種別ではなく payload で宣言する。

正直な限界(5 件)
------------------
- ~~**解析側の分類はまだ入れていない**~~ → **第109バッチ D2 で接続した**。
  ``scripts/analyze_accounting.py`` の ``MONEY_KINDS`` / ``HOUSEHOLD_KINDS`` / 部門フロー
  分類器に ``lost_return`` / ``lost_keep`` / ``lost_expire``(と家計残高側の ``lost_drop``)が
  入り、``unclassified_money_kinds`` は鳴り止んだ。積み残しだった設計判断
  =「遺失物という主体なしバケツをどの部門に置くか」は **家計部門の通過勘定**(供託金
  ``escrow`` を行政の内訳として扱うのと同じ流儀)と決めた: 落下は家計内部の資産振替なので
  フローを立てず、返還/報労金/着服/時効取得は家計 → 家計、失効だけが K5 へ出る。
  ★残る限界: ``lost`` バケツの残高は ``finance.parquet`` の列に無い(provenance にしか
  出ない)ので、**finance 版の検査①**では「まだバケツに残っている現金」が残差になる。
  L3 由来の ``conserve_household`` は 4 種すべてを数えるので完全に閉じている。
- **交番へ移動しない**: 届出も届け出も「最寄りの交番に届いた」と**記録するだけ**で、
  拾得者も遺失者もその場から動かない(計画書が明示的に許した簡約)。移動を伴わせると
  経路探索・滞在の予定を書き換えることになり、既存の行動列に手を入れてしまう。
- **落とし物は見えない**: 落ちている物はプロンプトに 1 バイトも出ない(拾得は決定論の
  判定で、個体が「見た」ことにはしていない)。「気づいて拾う」の知覚側は将来の拡張。
- **1 step の遅れ**: 落下の判定は位置確定後に走るので、その step のうちに拾われることは
  ない(最短で次 step)。
- **3 か月時効はほぼ発火しない**: 10 日ランでは ``statute_steps`` に届かないので、
  時効取得は既定では観測されない(テストは短い時効値で機構を固定する)。
- **街を出た当事者には金が動かない**(レーン甲 2026-08-13 で塞いだ保存則の穴): 返還も
  報労金も時効取得も**在場者にしか**起こらない。``sim.agent_by_id`` は退場者(プール回転で
  脱水された個体)も返すので、以前はそこへ ``money +=`` していた = 書いた値が hydrate で
  捨てられ ``hold`` からだけ引かれて**現金が世界から消えていた**。いまは
  ``sim.present_agent`` で解決し、持ち主 / 拾得者が街に居ない間は物を交番に留め置く
  (時効の瞬間に拾得者が不在なら**失効**= 遺失物法 36 条)。総額はどの分岐でも不変。

**絶対に触らない範囲(明文規約)**
--------------------------------
1. ``observer/metrics_spec.py`` の**凍結 14 ファイル**は 1 バイトも変更しない。
2. ``chance.py`` は 1 バイトも変更しない。本 module は windfall の**構造的な代替**だが、
   溶解(chance の退役)は H3 + H4 完了後の別バッチの仕事である(計画書 §5 の依存)。
3. ``diversity.py`` / ``city_ops.py`` / ``health.py`` / ``agents/agent.py`` は読むだけ。
"""
from __future__ import annotations

import hashlib

from . import economy_sfc as sfc_mod
from .observer.schema import Event, register_event_kind
from .world.perception import hearers_of

SCHEMA = 1

# --------------------------------------------------------------------------- #
# L1 イベント種(**材料側 registration**: city_ops.py:167 / devices.py / traces.py と
# 同じ流儀 = observer/schema.py 本体を触らずに自分の種を自分で登録する)。
# ★payload に自由文は 1 つも入らない(品目は下の有限表の語・場所は世界の識別子・
#   残りは件数と真偽値)。★**前兆状態**(混雑 crowd / 飲酒 drinking / 品目 item)を
#   発火時の値のまま同梱する = 「内生である」ことを解析側で機械検証できる(計画書 §3)。
# --------------------------------------------------------------------------- #
register_event_kind("lost_drop", "落とし物が発生した(移動・飲酒・混雑のハザード)。"
                                 "agent_id = 落とした本人"
                                 "{item, node, crowd, drinking, moving, rain, cash}")
register_event_kind("lost_notice", "紛失に気づいて交番へ遺失届を出した。agent_id = 落とし主"
                                   "{item, post, delay_steps}")
register_event_kind("lost_pickup", "落ちている物を拾った。agent_id = 拾得者"
                                   "{item, node, owner, guardians, lying_steps}")
register_event_kind("lost_turnin", "拾った物を交番/駅事務室へ届けた。agent_id = 拾得者"
                                   "{item, node, post, owner, guardians}")
register_event_kind("lost_return", "届いた物が持ち主へ返還された(報労金つき)。"
                                   "agent_id = 落とし主"
                                   "{item, finder, amount, cash, held_steps}")
register_event_kind("lost_keep", "★拾った物をそのまま自分のものにした(占有離脱物横領)。"
                                 "agent_id = 拾得者"
                                 "{item, owner, amount, guardians, offense}")
register_event_kind("lost_expire", "時効(3 か月)で拾得者の物になった / 誰にも拾われず失われた。"
                                   "agent_id = 拾得者(誰も拾わなかった回は -1)"
                                   "{item, owner, to, amount, age_steps}")


# --------------------------------------------------------------------------- #
# 品目の有限表(**ここに無い品目は存在しない**)
#
#   word … 記憶の 1 行に埋まる定型語。**品目の種類だけ**を指す一般名詞で、
#          固有名・ブランド名・数値は入らない(no-fingerprint の定型文規約)。
# --------------------------------------------------------------------------- #
UMBRELLA = "umbrella"
WALLET = "wallet"
PHONE = "phone"
OTHER = "other"
ITEMS: tuple[str, ...] = (UMBRELLA, WALLET, PHONE, OTHER)

ITEM_WORD: dict[str, str] = {
    UMBRELLA: "傘",
    WALLET: "財布",
    PHONE: "携帯電話",
    OTHER: "持ち物",
}

# 記憶の 1 行(**(出来事, 品目語) だけの純関数**。金額・件数・確率・時刻・実験条件は 1 つも入らない)
TEXT: dict[str, str] = {
    "notice": "{word}を失くしたことに気づいて、交番に届け出た。",
    "pickup": "落ちていた{word}を拾った。",
    "turnin": "拾った{word}を交番に届けた。",
    "keep":   "拾った{word}を、そのまま自分のものにした。",
    "return": "失くした{word}が手元に戻ってきた。",
    "reward": "届けた落とし物の持ち主から、お礼を受け取った。",
    "expire": "届けた落とし物が、そのまま自分のものになった。",
}

# 落ちている物の状態(有限)
LYING = "lying"          # 路上/屋内に落ちている(拾得の対象)
HELD = "held"            # 拾得者が持っている(この step のうちに届ける/着服が決まる)
TURNED_IN = "turnedin"   # 交番に届いている(持ち主の引き取り待ち / 時効待ち)


# --------------------------------------------------------------------------- #
# 較正済みの既定値(**すべて module docstring の「較正」節に出どころがある**)
# --------------------------------------------------------------------------- #
#: 共変量を 1 つも掛けない基底の落下率 [件/人日]。合計 2.0e-3 → 曝露分布で期待値化すると
#: **実現 2.92e-3**(= 都実測から導いた目標)。実現値の構成比は警視庁の受理内訳の桁に一致する
#: (傘 7.4% / 携帯 8.7% / 財布 8.7% / その他 75.1%)。
BASE_DAILY: dict[str, float] = {
    UMBRELLA: 0.8e-4,
    WALLET:   1.6e-4,
    PHONE:    1.6e-4,
    OTHER:    1.6e-3,
}

#: 持ち主が「失くした」ことに気づくまでの遅れ [step](Δt=10 分)。
#: 携帯は身体化された道具なので即座に気づき(10 分)、傘は次に雨が降るまで気づかない(3 時間)。
NOTICE_DELAY: dict[str, int] = {PHONE: 1, WALLET: 3, OTHER: 9, UMBRELLA: 18}

#: 持ち主が紛失に気づく確率(= 返還率の第 1 段)。
NOTICE_PROB: dict[str, float] = {PHONE: 0.99, WALLET: 0.98, OTHER: 0.60, UMBRELLA: 0.35}

#: 気づいた持ち主が交番へ引き取りに行く確率(= 返還率の第 2 段)。
#: ``NOTICE_PROB × CLAIM_PROB`` が東京都遺失物センターの品目別返還率に一致する。
CLAIM_PROB: dict[str, float] = {PHONE: 0.92, WALLET: 0.73, OTHER: 0.36, UMBRELLA: 0.03}

#: 落ちている物を通りかかった人が拾う確率(素通り = 1 − これ)。
PICKUP_PROB: dict[str, float] = {PHONE: 0.80, WALLET: 0.85, OTHER: 0.45, UMBRELLA: 0.25}

#: 拾った人が交番へ届ける確率(1 − これ = 着服)。日本の財布返還実験の水準(≈8 割)。
TURNIN_PROB: float = 0.78

#: 品目の名目価格 [円](報労金の算定基礎)。財布は「本体 + 中の現金」。
ITEM_VALUE: dict[str, float] = {
    UMBRELLA: 1_500.0, WALLET: 3_000.0, PHONE: 60_000.0, OTHER: 3_000.0}

DEFAULTS: dict = {
    "enabled": False,
    # ---- ① 落とす(唯一の確率事象。新 stream "lost_drop")----
    "base_daily": dict(BASE_DAILY),
    "move_mult": 1.8,          # 移動中(経路を持っている)= 遺失の主戦場
    "crowd_mult": 1.5,         # 混雑ノード(頭数が閾値以上)
    "crowd_threshold": 8,      # 同一ノードの在場者数(これ以上を「混雑」と呼ぶ)
    "drink_mult": 3.0,         # 飲酒(夜間の nightlife ノードに居る)= 財布・携帯のみ
    "rain_mult": 5.0,          # 雨/雪の日 = 傘のみ
    "dry_mult": 0.2,           # 雨でない日の傘(持ち歩いていないので落としようがない)
    "indoor_mult": 2.0,        # 店内(shop/food/cafe/nightlife の建物内)= 傘の置き忘れ
    "drink_start_min": 18 * 60,   # 飲酒帯の始まり(分 of day)
    "drink_end_min": 5 * 60,      # 同・終わり(翌朝。日をまたぐ帯)
    "drink_cats": ("nightlife",),
    "indoor_cats": ("shop", "food", "cafe", "nightlife"),
    # ---- ② 気づく / 引き取る ----
    "notice_delay": dict(NOTICE_DELAY),
    "notice_prob": dict(NOTICE_PROB),
    "claim_prob": dict(CLAIM_PROB),
    # ---- ③ 拾う / 届ける / 着服 ----
    "pickup_prob": dict(PICKUP_PROB),
    "turnin_prob": TURNIN_PROB,
    "temptation": 0.25,        # 高額なほど着服に傾く強さ(0=金額に無感応)
    "guardian_mult": 1.20,     # 監視者(同席者)が居るときの届ける倍率(RAT)
    # ---- ④ 金 ----
    "wallet_cash_frac": 0.35,  # 財布に入れて持ち歩く現金の割合(所持金比)
    "wallet_cash_max": 50_000.0,
    "item_value": dict(ITEM_VALUE),
    "reward_lo": 0.05,         # 遺失物法 28 条: 5% 以上
    "reward_hi": 0.20,         #               20% 以下
    # ---- ⑤ 時間 ----
    "statute_steps": 90 * 144,  # 3 か月時効(遺失物法 7 条)
    "abandon_steps": 3 * 144,   # 誰にも拾われないまま消える(路上の物の寿命)
    # ---- ⑥ 安全弁 ----
    "max_events_per_step": 24,  # 1 step に出す本 module の L1 件数の上限
    "max_items": 4096,          # 同時に存在する未解決の遺失物の上限
}

_FLOAT_KEYS = ("move_mult", "crowd_mult", "drink_mult", "rain_mult", "dry_mult",
               "indoor_mult", "turnin_prob", "temptation", "guardian_mult",
               "wallet_cash_frac", "wallet_cash_max", "reward_lo", "reward_hi")
_INT_KEYS = ("crowd_threshold", "drink_start_min", "drink_end_min",
             "statute_steps", "abandon_steps", "max_events_per_step", "max_items")
_ITEM_FLOAT_BLOCKS = ("base_daily", "notice_prob", "claim_prob", "pickup_prob",
                      "item_value")
_ITEM_INT_BLOCKS = ("notice_delay",)
_TUPLE_KEYS = ("drink_cats", "indoor_cats")


# --------------------------------------------------------------------------- #
# cfg 正準化(traces.build_cfg / rumors.build_cfg と同型: dict / OmegaConf 両対応)
# --------------------------------------------------------------------------- #
def _to_plain(raw):
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(raw):
            return OmegaConf.to_container(raw, resolve=True)
    except Exception:                              # noqa: BLE001(旧 config 互換)
        pass
    return raw


def build_cfg(raw) -> dict:
    """conf の ``lost_property`` ブロックを型強制つきで正準化(既定 OFF=現行挙動と完全同一)。

    dotlist 上書きは文字列で入り得るので型強制する(``traces.build_cfg`` と同じ作法)。
    品目別の辞書は**既定へ重ねる**(``kinds`` のような「置き換え」ではない): 品目の集合は
    有限表 ``ITEMS`` で閉じており、config で品目を増減させる意味がないため。表に無い品目名は
    **黙って捨てる**(存在しない品目の較正値を作らない = 捏造しない)。
    """
    raw = dict(_to_plain(raw) or {})
    cfg = dict(DEFAULTS)
    for blk in _ITEM_FLOAT_BLOCKS + _ITEM_INT_BLOCKS:
        cfg[blk] = dict(DEFAULTS[blk])
    for key, val in raw.items():
        if key == "enabled":
            cfg["enabled"] = bool(val)
        elif key in _ITEM_FLOAT_BLOCKS:
            for item, v in dict(_to_plain(val) or {}).items():
                if str(item) in ITEMS:
                    cfg[key][str(item)] = max(0.0, float(v))
        elif key in _ITEM_INT_BLOCKS:
            for item, v in dict(_to_plain(val) or {}).items():
                if str(item) in ITEMS:
                    cfg[key][str(item)] = max(0, int(v))
        elif key in _FLOAT_KEYS:
            cfg[key] = max(0.0, float(val))
        elif key in _INT_KEYS:
            cfg[key] = max(0, int(val))
        elif key in _TUPLE_KEYS:
            cfg[key] = tuple(str(s) for s in (_to_plain(val) or ()))
    # 退化の防止: 報労金の帯は lo <= hi(逆順に書かれたら入れ替える = 5-20% の帯を守る)
    if cfg["reward_hi"] < cfg["reward_lo"]:
        cfg["reward_lo"], cfg["reward_hi"] = cfg["reward_hi"], cfg["reward_lo"]
    return cfg


def cfg_of(sim) -> dict:
    """遺失物設定(初回のみ ``sim.cfg.lost_property`` から遅延構築してキャッシュ)。

    simulation.py は読み取り専用なので cfg は本 module が遅延構築する
    (``traces.cfg_of`` / ``rumors.cfg_of`` と同型)。キャッシュ属性 ``sim.lostcfg`` は
    L1/L2/L3/乱数に一切現れない = 既定 OFF のバイト一致を壊さない。
    """
    c = getattr(sim, "lostcfg", None)
    if c is None:
        try:
            raw = sim.cfg.get("lost_property", None)
        except Exception:                          # noqa: BLE001(旧 config 互換)
            raw = None
        c = build_cfg(raw)
        sim.lostcfg = c
    return c


def enabled(sim) -> bool:
    return bool(cfg_of(sim)["enabled"])


# --------------------------------------------------------------------------- #
# 定型文(純関数。sim も agent も config も金額も見ない = 全実験条件で同一バイト列)
# --------------------------------------------------------------------------- #
def sentence(event: str, item: str) -> str:
    """記憶の 1 行。**(出来事, 品目) だけの純関数**。

    埋まるのは有限表の定型語だけで、金額・件数・確率・階層名・時刻・実験条件・機構語は
    1 つも入らない(tests が「テンプレート側にも品目語側にも数字が無い」ことを機械検査する)。
    語彙外は空文字 = 行にしない(捏造しない)。
    """
    tpl = TEXT.get(str(event))
    if tpl is None:
        return ""
    if "{word}" not in tpl:                        # 品目を言わない行(reward / expire)
        return tpl
    word = ITEM_WORD.get(str(item))
    return "" if word is None else tpl.format(word=word)


# --------------------------------------------------------------------------- #
# 決定論ジッタ(**乱数 stream を 1 本も消費しない純関数**)
#
# ``city_ops._collapse_gate`` のハッシュゲートと同じ前例。(master_seed, 個体, 物, 用途)の
# sha256 から [0,1) の値を作る。同 seed・同編成なら resume を跨いでも同じ値になる。
# --------------------------------------------------------------------------- #
def _jitter(seed: int, *parts) -> float:
    blob = "/".join([str(int(seed))] + [str(p) for p in parts])
    digest = hashlib.sha256(blob.encode("utf-8")).digest()[:8]
    return int.from_bytes(digest, "big") / float(1 << 64)


def _seed_of(sim) -> int:
    try:
        return int(sim.hub.master_seed)
    except Exception:                              # noqa: BLE001(hub 不在の単体テスト)
        return 0


def _trait(agent, name: str, default: float = 0.5) -> float:
    try:
        return float((getattr(agent, "traits", None) or {}).get(name, default))
    except Exception:                              # noqa: BLE001
        return default


def _state_of(agent, name: str, default: float = 0.0) -> float:
    try:
        return float((getattr(agent, "states", None) or {}).get(name, default))
    except Exception:                              # noqa: BLE001
        return default


def _clip01(v: float) -> float:
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else float(v))


# --------------------------------------------------------------------------- #
# 遺失物ストア(ON 経路でのみ生やす。checkpoint.py が runtime で中央管理する)
#
#   items[item_id] = {kind, owner, node, x, y, loc, building, floor, cash,
#                     step, status, noticed, finder, turn_step}
#   hold   … いま「世界のどこかに在るが誰の残高でもない」現金の合計(貨幣保存の第 3 項)
#   lapsed … 誰にも拾われないまま失われた現金の累計(SNA K.5 = 取引でない資産変動)
# --------------------------------------------------------------------------- #
def _state(sim) -> dict:
    st = getattr(sim, "_lost_state", None)
    if st is None:
        st = {"schema": SCHEMA, "seq": 0, "items": {},
              "hold": 0.0, "lapsed": 0.0, "reward_paid": 0.0, "kept_cash": 0.0,
              "drops": {}, "notices": {}, "pickups": {}, "passed": {},
              "turnins": {}, "returns": {}, "keeps": {}, "expired": {},
              "lapses": {}}
        sim._lost_state = st
    return st


def _bump(table: dict, key: str) -> None:
    table[key] = int(table.get(key, 0)) + 1


# --------------------------------------------------------------------------- #
# 世界を読むだけのヘルパ(既存機構は 1 バイトも書き換えない)
# --------------------------------------------------------------------------- #
def _post_nodes(sim) -> list[str]:
    """交番・警察署・駅事務室のノード列(**読むだけ**・起動時 1 回でキャッシュ)。

    ``city_ops._police_posts`` と同じ判定(POI カテゴリ ∩ 一般名詞キーワード)だが、
    あちらを import すると city_ops の cfg(既定 OFF)に依存してしまうので、
    **読み取り専用の独立実装**を置く(city_ops.py は 1 バイトも触らない)。
    語は一般名詞のみ(施設種そのものを指すので誤爆しない)。
    """
    cache = getattr(sim, "_lost_post_nodes", None)
    if cache is not None:
        return cache
    out: set[str] = set()
    city = getattr(sim, "city", None)
    for poi in (getattr(city, "poi_list", None) or ()):
        if str(poi.get("cat") or "") not in ("service", "police", "transport"):
            continue
        name = str(poi.get("name") or "")
        if not any(w in name for w in ("交番", "警察署", "駅事務室")):
            continue
        node = str(poi.get("node") or "")
        if node:
            out.add(node)
    posts = sorted(out)
    sim._lost_post_nodes = posts
    return posts


def _nearest_post(sim, node: str) -> str:
    """最寄りの届け先ノード(**移動はしない**= 記録上の届け先)。

    交番が地図に 1 件も無ければ空文字(= 届け先が世界に存在しないことを隠さない)。
    距離は直線距離で、経路探索は 1 回も呼ばない(既存の移動計画に一切触れないため)。
    """
    posts = _post_nodes(sim)
    if not posts:
        return ""
    city = getattr(sim, "city", None)
    try:
        x0, y0 = city.node_xy(str(node))
    except Exception:                              # noqa: BLE001(未知ノード)
        return posts[0]
    best, best_d = posts[0], None
    for p in posts:                                # 交番は数十件 = O(1) 相当
        try:
            x1, y1 = city.node_xy(p)
        except Exception:                          # noqa: BLE001
            continue
        d = (x1 - x0) ** 2 + (y1 - y0) ** 2
        if best_d is None or d < best_d:
            best, best_d = p, d
    return best


def _node_cats(sim, node: str) -> frozenset:
    """そのノードの POI カテゴリ集合(**読むだけ**・ノード単位でキャッシュ)。"""
    cache = getattr(sim, "_lost_node_cats", None)
    if cache is None:
        cache = {}
        sim._lost_node_cats = cache
    got = cache.get(node)
    if got is None:
        city = getattr(sim, "city", None)
        try:
            got = frozenset(str(p.get("cat") or "")
                            for p in city.pois_at_node(str(node)))
        except Exception:                          # noqa: BLE001(未知ノード)
            got = frozenset()
        cache[node] = got
    return got


def _rain_today(sim) -> bool:
    """今日は雨/雪か(既存の天気を**読むだけ**。天気 OFF なら False)。"""
    from .weather import _BAD_CONDS
    w = getattr(sim, "today_weather", None)
    return bool(w) and str(w.get("cond")) in _BAD_CONDS


def _in_drink_band(cfg: dict, sim_min: int) -> bool:
    """飲酒帯(18:00-05:00 = 日をまたぐ帯)に入っているか。"""
    tod = int(sim_min) % 1440
    lo, hi = int(cfg["drink_start_min"]), int(cfg["drink_end_min"])
    return (tod >= lo or tod < hi) if lo > hi else (lo <= tod < hi)


# --------------------------------------------------------------------------- #
# 探針(落下地点の共在を ``sim.percept_index`` から引くための duck-type)
#
# ★``hearers_of`` は「話者」に (id, x, y, loc, building, floor, sleeping) を要求する。
#   落とし物は agent ではないので、落下時の位置を凍らせた**探針**を立てて索引を引く。
#   これで **新しい全対全スキャンを 1 つも足さずに** co-location を得る(計画書 §3)。
# --------------------------------------------------------------------------- #
class _Probe:
    __slots__ = ("id", "x", "y", "loc", "building", "floor", "sleeping")

    def __init__(self, rec: dict):
        self.id = -1                               # どの agent とも衝突しない番兵
        self.x = float(rec["x"])
        self.y = float(rec["y"])
        self.loc = str(rec["loc"])
        self.building = rec["building"]
        self.floor = int(rec["floor"])
        self.sleeping = False


def _bystanders(sim, rec: dict) -> list:
    """落下地点に居合わせている agent(**agent_id 昇順**の決定論・乱数ゼロ)。

    索引が未構築(= 位置確定前に呼ばれた単体テスト)のときだけ live 走査へ後退する
    (``traces._bystanders`` と同じ「索引があれば索引・無ければ列」の作法)。
    """
    index = getattr(sim, "percept_index", None)
    radius = float(sim.cfg.world.perception_radius_m)
    return hearers_of(_Probe(rec), index if index is not None else sim.agents, radius)


# --------------------------------------------------------------------------- #
# 決定論の分岐(**乱数を 1 粒も引かない**。ハザード型 = p_base × 倍率 と純関数ジッタの比較)
# --------------------------------------------------------------------------- #
def notice_p(cfg: dict, owner, item: str) -> float:
    """持ち主が紛失に気づく確率。品目の基底 × 内的統制(自分の持ち物を把握している度合い)。"""
    base = float(cfg["notice_prob"].get(item, 0.5))
    return _clip01(base * (0.75 + 0.5 * _trait(owner, "internal_locus")))


def claim_p(cfg: dict, owner, item: str) -> float:
    """気づいた持ち主が交番へ引き取りに行く確率。品目の基底 × 内的統制。

    ★品目差が支配的(傘 3% / 携帯 89%)なのは実測どおり = 「取りに行く価値があるか」は
      性格ではなく物で決まる。性格は縁を作るだけ(±25%)。
    """
    base = float(cfg["claim_prob"].get(item, 0.5))
    return _clip01(base * (0.75 + 0.5 * _trait(owner, "internal_locus")))


def pickup_p(cfg: dict, finder, item: str) -> float:
    """通りかかった人が拾う確率。品目の基底 × 当事者意識(ownership)。

    ★拾う = 手間をかけて他人事に関わること。当事者意識(``states["ownership"]``)が
      高いほど拾う(Latané-Darley の「責任の分散」の裏返しを規範状態で表現する)。
    """
    base = float(cfg["pickup_prob"].get(item, 0.5))
    return _clip01(base * (0.85 + 0.6 * _state_of(finder, "ownership", 0.1)))


def turnin_p(cfg: dict, finder, item: str, value: float, guardians: int) -> float:
    """拾った人が届ける確率(1 − これ = 着服)。**性格・規範状態・金額・監視者**の純関数。

    - ``risk_tolerance`` が高いほど着服に傾く(捕まる可能性を織り込まない)
    - ``grievance``(不満)が高いほど着服に傾く(社会への負債感の反転)
    - ``internal_locus`` が高いほど届ける(自分の行いが結果を作るという信念)
    - **金額が大きいほど着服に傾く**(``temptation``)
    - **監視者(同席者)が居ると届ける**(RAT の guardianship。``guardian_mult``)
    """
    base = float(cfg["turnin_prob"])
    risk = _trait(finder, "risk_tolerance")
    locus = _trait(finder, "internal_locus")
    grief = _state_of(finder, "grievance", 0.1)
    mult = (1.0 - 0.35 * (risk - 0.5)) * (1.0 + 0.25 * (locus - 0.5))
    mult *= (1.0 - 0.30 * _clip01(grief))
    vmax = max(1.0, float(cfg["wallet_cash_max"]))
    mult *= (1.0 - float(cfg["temptation"]) * _clip01(float(value) / vmax))
    if int(guardians) > 0:
        mult *= float(cfg["guardian_mult"])
    return _clip01(base * mult)


def reward_rate(cfg: dict, seed: int, owner_id: int, item_id: int) -> float:
    """報労金の率。遺失物法 28 条の帯 [5%, 20%] を決定論ジッタで一様に埋める。"""
    lo, hi = float(cfg["reward_lo"]), float(cfg["reward_hi"])
    return lo + (hi - lo) * _jitter(seed, "reward", int(owner_id), int(item_id))


# --------------------------------------------------------------------------- #
# ① 落とす(唯一の確率事象。新 stream "lost_drop")
# --------------------------------------------------------------------------- #
def _drop_mults(sim, cfg: dict, agent, node_n: int, rain: bool,
                drinking: bool, moving: bool, indoor_cats: frozenset) -> tuple:
    """品目別の共変量倍率(**品目ごとに効く共変量が違う**のが計画書の要求)。

    戻り値は ``ITEMS`` と同じ順の tuple(1 step に在場者ぶん呼ばれるので dict を作らない)。
    """
    crowd = node_n >= int(cfg["crowd_threshold"])
    base = float(cfg["move_mult"]) if moving else 1.0
    # 傘 = 雨天日 + 店舗滞在(置き忘れ)。降っていない日はそもそも持ち歩かない。
    umb = base * (float(cfg["rain_mult"]) if rain else float(cfg["dry_mult"]))
    if getattr(agent, "building", None) and (
            _node_cats(sim, getattr(agent, "node", "")) & indoor_cats):
        umb *= float(cfg["indoor_mult"])
    # 財布・携帯 = 飲酒 + 混雑(ポケットから落ちる・置き忘れる)
    val = base
    if drinking:
        val *= float(cfg["drink_mult"])
    if crowd:
        val *= float(cfg["crowd_mult"])
        umb *= 1.2
    oth = base * (1.2 if crowd else 1.0)
    return (umb, val, val, oth)                    # = (UMBRELLA, WALLET, PHONE, OTHER)


def _wallet_cash(cfg: dict, owner) -> float:
    """財布に入れて持ち歩いている現金(**所持金から分離する額**)。決定論。"""
    money = max(0.0, float(getattr(owner, "money", 0.0) or 0.0))
    cash = min(money * float(cfg["wallet_cash_frac"]), float(cfg["wallet_cash_max"]))
    return round(min(cash, money), 1)


def _drop(sim, cfg: dict, st: dict, agent, item: str, step: int, sim_min: int,
          node_n: int, rain: bool, drinking: bool, moving: bool) -> None:
    """落下 1 件: 物を世界に置き、財布なら**現金を所持金から分離**する(貨幣保存)。"""
    cash = _wallet_cash(cfg, agent) if item == WALLET else 0.0
    if cash > 0.0:
        agent.money = round(max(0.0, float(agent.money) - cash), 6)
        st["hold"] = round(float(st["hold"]) + cash, 6)
        sfc_mod.on_lost_hold(sim, cash)            # IF-E: 街の中に在るが誰の残高でもない金
    st["seq"] = int(st["seq"]) + 1
    iid = int(st["seq"])
    st["items"][iid] = {
        "id": iid, "item": str(item), "owner": int(agent.id),
        "node": str(getattr(agent, "node", "") or ""),
        "x": float(agent.x), "y": float(agent.y),
        "loc": str(getattr(agent, "loc", "street") or "street"),
        "building": getattr(agent, "building", None),
        "floor": int(getattr(agent, "floor", 0) or 0),
        "cash": float(cash), "step": int(step), "status": LYING,
        "noticed": False, "finder": -1, "turn_step": -1,
    }
    _bump(st["drops"], str(item))
    payload = {"item": str(item), "node": str(getattr(agent, "node", "") or ""),
               "crowd": int(node_n), "drinking": bool(drinking),
               "moving": bool(moving), "rain": bool(rain)}
    if cash > 0.0:
        payload["cash"] = round(cash, 1)
    sim.logger.log(Event(step=int(step), sim_min=int(sim_min), agent_id=int(agent.id),
                         kind="lost_drop", x=agent.x, y=agent.y, payload=payload))


def _phase_drop(sim, cfg: dict, st: dict, step: int, sim_min: int, budget: int) -> int:
    """在場者 1 人 1 draw の落下ハザード(``diversity.tick_crime`` と同じ前例)。

    1 人 1 step に stream "lost_drop" から**ちょうど 1 回**引き、品目の累積帯に振り分ける
    (相互排他 = 同じ step に 2 つ落とすことはない)。
    """
    if budget <= 0 or len(st["items"]) >= int(cfg["max_items"]):
        return budget
    spd = max(1, int(getattr(getattr(sim, "clock", None), "steps_per_day", 144)))
    rain = _rain_today(sim)
    drink_band = _in_drink_band(cfg, sim_min)
    drink_cats = frozenset(cfg["drink_cats"])
    indoor_cats = frozenset(cfg["indoor_cats"])
    # 基底の per-step ハザード(品目順・共変量なし)。1 step に 1 回だけ作る。
    spd_base = tuple(float(cfg["base_daily"].get(it, 0.0)) for it in ITEMS)
    at_node: dict[str, int] = {}
    active = []
    for a in sim.agents:
        if a.loc == "outside" or a.sleeping:
            continue
        at_node[a.node] = at_node.get(a.node, 0) + 1
        active.append(a)
    for agent in active:                           # sim.agents 由来 = id 昇順 = 決定論
        if budget <= 0 or len(st["items"]) >= int(cfg["max_items"]):
            break
        node_n = at_node.get(agent.node, 1)
        moving = bool(getattr(agent, "route", None)) or bool(getattr(agent, "dest", None))
        drinking = bool(drink_band and (_node_cats(sim, agent.node) & drink_cats))
        mults = _drop_mults(sim, cfg, agent, node_n, rain, drinking, moving,
                            indoor_cats)
        r = float(sim.hub.stream("lost_drop", int(agent.id), int(step)).random())
        acc = 0.0
        for k, item in enumerate(ITEMS):           # 固定順 = 決定論
            acc += spd_base[k] * mults[k] / float(spd)
            if r < acc:
                _drop(sim, cfg, st, agent, item, step, sim_min,
                      node_n, rain, drinking, moving)
                budget -= 1
                break
    return budget


# --------------------------------------------------------------------------- #
# ② 気づく(遅延つき)= 遺失届
# --------------------------------------------------------------------------- #
def _phase_notice(sim, cfg: dict, st: dict, step: int, sim_min: int, budget: int) -> int:
    seed = _seed_of(sim)
    for iid in sorted(st["items"]):                # id 昇順 = 決定論
        if budget <= 0:
            break
        rec = st["items"][iid]
        if rec["noticed"]:
            continue
        item = str(rec["item"])
        if int(step) - int(rec["step"]) < int(cfg["notice_delay"].get(item, 6)):
            continue
        rec["noticed"] = True                      # 「判定は 1 回だけ」= 遅延到達の時点で確定
        # ★在場述語で解決する(``agent_by_id`` は退場者も返す = 幽霊の記憶に書いてしまう)。
        owner = sim.present_agent(int(rec["owner"]))
        if owner is None:                          # プール回転で街から居なくなった
            continue
        if _jitter(seed, "notice", int(owner.id), iid) >= notice_p(cfg, owner, item):
            continue                               # 失くしたことに気づかない(傘の 65%)
        rec["reported"] = True
        _bump(st["notices"], item)
        owner.remember(sentence("notice", item))
        budget -= 1
        sim.logger.log(Event(step=int(step), sim_min=int(sim_min), agent_id=int(owner.id),
                             kind="lost_notice", x=owner.x, y=owner.y,
                             payload={"item": item,
                                      "post": _nearest_post(sim, str(rec["node"])),
                                      "delay_steps": int(step) - int(rec["step"])}))
    return budget


# --------------------------------------------------------------------------- #
# ③ 拾う(共在)→ ④ 分岐(届ける / 着服)
# --------------------------------------------------------------------------- #
def _pay(src, dst, amount: float) -> float:
    """個体間の直接授受(**街の中の家計 → 家計**= 総額は 1 円も増減しない)。

    払い手の手持ちを超える額は払えない(床クリップ)。返すのは**実際に動いた額**なので、
    payload と残高の増減が必ず一致する(``chance._apply_money`` と同じ正直さの作法)。

    ★**呼ぶ側の責務**: src / dst はどちらも**在場者**(``sim.present_agent`` で解決した実体)
      でなければならない。退場者(脱水済み)を渡すとその側の書き込みが hydrate で捨てられ、
      片側だけ動いた分だけ街の総額がずれる。
    """
    amt = min(max(0.0, float(amount)), max(0.0, float(getattr(src, "money", 0.0) or 0.0)))
    if amt <= 0.0:
        return 0.0
    src.money = round(float(src.money) - amt, 6)
    dst.money = round(float(dst.money) + amt, 6)
    return amt


def _value_of(cfg: dict, rec: dict) -> float:
    """報労金の算定基礎(財布は本体 + 中の現金)。"""
    base = float(cfg["item_value"].get(str(rec["item"]), 0.0))
    return base + float(rec["cash"])


def _keep(sim, cfg: dict, st: dict, rec: dict, finder, guardians: int,
          step: int, sim_min: int) -> None:
    """★着服(占有離脱物横領)。中の現金は拾得者の手に渡る = **必ず誰かの drop から**。

    ``crime`` イベントは出さない(module docstring「意図的な逸脱」節)。犯罪であることは
    payload の ``offense`` で宣言し、**監視者不在**を ``guardians`` = 0 として残す
    (RAT と同じ語彙 = 解析側が「監視が無かったから起きた」を機械検証できる)。
    """
    item = str(rec["item"])
    cash = float(rec["cash"])
    if cash > 0.0:
        finder.money = round(float(finder.money) + cash, 6)
        st["hold"] = round(float(st["hold"]) - cash, 6)
        st["kept_cash"] = round(float(st["kept_cash"]) + cash, 6)
        sfc_mod.on_lost_release(sim, cash)
    rec["status"] = "kept"
    _bump(st["keeps"], item)
    finder.remember(sentence("keep", item))
    sim.logger.log(Event(step=int(step), sim_min=int(sim_min), agent_id=int(finder.id),
                         kind="lost_keep", x=finder.x, y=finder.y,
                         payload={"item": item, "owner": int(rec["owner"]),
                                  "amount": round(cash, 1),
                                  "guardians": int(guardians),
                                  "offense": "占有離脱物横領"}))


def _turnin(sim, cfg: dict, st: dict, rec: dict, finder, guardians: int,
            step: int, sim_min: int) -> None:
    """交番/駅事務室へ届ける(**移動は伴わない簡約**だが、行為のイベントは必ず出す)。"""
    item = str(rec["item"])
    rec["status"] = TURNED_IN
    rec["turn_step"] = int(step)
    _bump(st["turnins"], item)
    finder.remember(sentence("turnin", item))
    sim.logger.log(Event(step=int(step), sim_min=int(sim_min), agent_id=int(finder.id),
                         kind="lost_turnin", x=finder.x, y=finder.y,
                         payload={"item": item, "node": str(rec["node"]),
                                  "post": _nearest_post(sim, str(rec["node"])),
                                  "owner": int(rec["owner"]),
                                  "guardians": int(guardians)}))


def _phase_pickup(sim, cfg: dict, st: dict, step: int, sim_min: int, budget: int) -> int:
    """落ちている物 → 共在者から拾得者を決め、その場で届ける/着服を決める。"""
    seed = _seed_of(sim)
    for iid in sorted(st["items"]):                # id 昇順 = 決定論
        if budget <= 0:
            break
        rec = st["items"][iid]
        if rec["status"] != LYING or int(rec["step"]) >= int(step):
            continue                               # 落ちた step のうちには拾われない
        if budget < 2:
            break     # ★拾得と分岐(届ける/着服)は**不可分**: 途中で予算が尽きると物が
                      #   「拾われたまま宙に浮き」中の現金が誰の手にも渡らない = 貨幣が漏れる。
                      #   2 件ぶんの枠が無いときは、この step では拾わせない(次 step へ持ち越す)。
        near = _bystanders(sim, rec)
        if not near:
            continue
        item = str(rec["item"])
        finder = None
        for cand in near:                          # hearers_of は id 昇順 = 決定論
            if int(cand.id) == int(rec["owner"]):
                continue                           # 落とし主本人は「拾得者」ではない
            if _jitter(seed, "pickup", int(cand.id), iid) < pickup_p(cfg, cand, item):
                finder = cand
                break
            _bump(st["passed"], item)              # 素通り(無視)も観測量として数える
        if finder is None:
            continue
        # 監視者 = 拾得の瞬間に居合わせた**拾得者以外**の人数(RAT の guardianship)
        guardians = sum(1 for a in near if int(a.id) != int(finder.id))
        rec["status"] = HELD
        rec["finder"] = int(finder.id)
        _bump(st["pickups"], item)
        finder.remember(sentence("pickup", item))
        sim.logger.log(Event(step=int(step), sim_min=int(sim_min),
                             agent_id=int(finder.id), kind="lost_pickup",
                             x=finder.x, y=finder.y,
                             payload={"item": item, "node": str(rec["node"]),
                                      "owner": int(rec["owner"]),
                                      "guardians": int(guardians),
                                      "lying_steps": int(step) - int(rec["step"])}))
        budget -= 1
        value = _value_of(cfg, rec)
        if _jitter(seed, "turnin", int(finder.id), iid) < turnin_p(
                cfg, finder, item, value, guardians):
            _turnin(sim, cfg, st, rec, finder, guardians, step, sim_min)
        else:
            _keep(sim, cfg, st, rec, finder, guardians, step, sim_min)
        budget -= 1
    return budget


# --------------------------------------------------------------------------- #
# ⑤ 返還(報労金 5-20%)/ 時効 / 失効
# --------------------------------------------------------------------------- #
def _return(sim, cfg: dict, st: dict, rec: dict, owner, finder,
            step: int, sim_min: int) -> None:
    """返還 + 報労金。**現金は持ち主へ戻り、報労金は持ち主 → 拾得者の直接授受**。

    金は 1 円も湧かない: 中の現金は世界の「遺失物」バケツ(``hold``)から持ち主の財布へ
    戻るだけ、報労金は持ち主の財布から拾得者の財布へ動くだけ(IF-E のゼロ和検査と整合)。

    ★``owner`` / ``finder`` は**在場者に限る**(``_phase_resolve`` が ``sim.present_agent``
      で解決してから呼ぶ)。どちらかが街を出ている間、物は交番に留め置かれ時効まで待つ。
    """
    item = str(rec["item"])
    cash = float(rec["cash"])
    if cash > 0.0:
        owner.money = round(float(owner.money) + cash, 6)
        st["hold"] = round(float(st["hold"]) - cash, 6)
        sfc_mod.on_lost_release(sim, cash)
    rate = reward_rate(cfg, _seed_of(sim), int(owner.id), int(rec["id"]))
    paid = _pay(owner, finder, round(_value_of(cfg, rec) * rate, 1))
    st["reward_paid"] = round(float(st["reward_paid"]) + paid, 6)
    rec["status"] = "returned"
    _bump(st["returns"], item)
    owner.remember(sentence("return", item))
    if paid > 0.0:
        finder.remember(sentence("reward", item))
    sim.logger.log(Event(step=int(step), sim_min=int(sim_min), agent_id=int(owner.id),
                         kind="lost_return", x=owner.x, y=owner.y,
                         payload={"item": item, "finder": int(finder.id),
                                  "amount": round(paid, 1), "cash": round(cash, 1),
                                  "held_steps": int(step) - int(rec["turn_step"])}))


def _expire(sim, cfg: dict, st: dict, rec: dict, to_finder, step: int,
            sim_min: int) -> None:
    """時効取得(拾得者へ)/ 失効(誰にも拾われないまま世界から消える)。

    後者の現金は**取引ではない資産変動**(SNA 2008 第12章 K.5)として記帳する:
    受け取り手が世界に存在しないので RoW(取引の相手方)へ落としてはならない。

    ★``to_finder`` は**在場者か None** のいずれか(呼び出し側が ``sim.present_agent`` で
      解決する)。時効の瞬間に拾得者が街に居ない回は None = 失効側へ倒す(遺失物法 36 条:
      引き取らなければ所有権を失う)。退場者の財布へ書くと hold からだけ引かれて現金が消える。
    """
    item = str(rec["item"])
    cash = float(rec["cash"])
    if cash > 0.0:
        st["hold"] = round(float(st["hold"]) - cash, 6)
        if to_finder is not None:
            to_finder.money = round(float(to_finder.money) + cash, 6)
            sfc_mod.on_lost_release(sim, cash)
        else:
            st["lapsed"] = round(float(st["lapsed"]) + cash, 6)
            sfc_mod.on_lost_lapse(sim, cash)
    rec["status"] = "expired"
    if to_finder is not None:
        _bump(st["expired"], item)
        to_finder.remember(sentence("expire", item))
    else:
        _bump(st["lapses"], item)
    aid = int(to_finder.id) if to_finder is not None else -1
    x = float(to_finder.x) if to_finder is not None else float(rec["x"])
    y = float(to_finder.y) if to_finder is not None else float(rec["y"])
    sim.logger.log(Event(step=int(step), sim_min=int(sim_min), agent_id=aid,
                         kind="lost_expire", x=x, y=y,
                         payload={"item": item, "owner": int(rec["owner"]),
                                  "to": "finder" if to_finder is not None else "void",
                                  "amount": round(cash, 1),
                                  "age_steps": int(step) - int(rec["step"])}))


def _phase_resolve(sim, cfg: dict, st: dict, step: int, sim_min: int,
                   budget: int) -> int:
    """届いた物の引き取り / 時効 / 路上の物の失効。"""
    seed = _seed_of(sim)
    done: list[int] = []
    for iid in sorted(st["items"]):                # id 昇順 = 決定論
        rec = st["items"][iid]
        status = str(rec["status"])
        if status == HELD:
            continue     # ★決着前の物は絶対に捨てない(中の現金の行き先が未定 = 捨てると漏れる)。
                         #   ``_phase_pickup`` が拾得と同じ step のうちに必ず決着させるので、
                         #   ここに来ることは構造上ないが、防波堤として明示しておく。
        if status not in (LYING, TURNED_IN):
            done.append(iid)
            continue
        if budget <= 0:
            continue
        if status == TURNED_IN:
            # ★**在場者にしか金は動かせない**: ``agent_by_id`` は退場者(脱水済み)も返すので、
            #   そこへ ``money +=`` すると次の hydrate で書いた値が捨てられ、hold からは
            #   引かれたまま = **現金が世界から消える**。在場述語で解決する。
            owner = sim.present_agent(int(rec["owner"]))
            finder = sim.present_agent(int(rec["finder"]))
            item = str(rec["item"])
            if (owner is not None and finder is not None and rec["noticed"]
                    and rec.get("reported")
                    and _jitter(seed, "claim", int(owner.id), iid)
                    < claim_p(cfg, owner, item)):
                _return(sim, cfg, st, rec, owner, finder, step, sim_min)
                budget -= 1
                done.append(iid)
            elif int(step) - int(rec["turn_step"]) >= int(cfg["statute_steps"]):
                # 時効。拾得者が街に居なければ**誰にも渡さず失効**(= to_finder=None)。
                # 遺失物法 36 条(拾得者が所有権取得後 2 か月以内に物件を引き取らないときは
                # 所有権を失い、物件は都道府県に帰属する)= 街の外へ出るのと同じ扱い。
                # 中の現金は lapsed バケツへ落ちるので**総額は 1 円も動かない**。
                _expire(sim, cfg, st, rec, finder, step, sim_min)
                budget -= 1
                done.append(iid)
        elif int(step) - int(rec["step"]) >= int(cfg["abandon_steps"]):
            _expire(sim, cfg, st, rec, None, step, sim_min)
            budget -= 1
            done.append(iid)
    for iid in done:                               # 決着した物はストアから外す(無限成長の防止)
        st["items"].pop(iid, None)
    return budget


# --------------------------------------------------------------------------- #
# 単一作用点(scheduler が呼ぶのはこの 1 本だけ)
# --------------------------------------------------------------------------- #
def phase(sim, step: int, sim_min: int) -> None:
    """落とす → 気づく → 拾う → 届ける/着服 → 返還/時効。**既定 OFF は即 return**。

    ★呼び出し位置は ``sim.percept_index`` 構築の**直後**(位置が確定している)。
      落下地点の共在をこの step の確定した co-location で判定するため
      (``_phase_enforcement`` / ``_phase_diversity`` / ``street_life`` と同じ線引き)。
    ★動かす世界状態は **所持金と記憶だけ**(位置・関係・drive・opinion は 1 つも触らない)。
    """
    if not enabled(sim):
        return
    cfg = cfg_of(sim)
    st = _state(sim)
    budget = int(cfg["max_events_per_step"])
    budget = _phase_resolve(sim, cfg, st, step, sim_min, budget)
    budget = _phase_pickup(sim, cfg, st, step, sim_min, budget)
    budget = _phase_notice(sim, cfg, st, step, sim_min, budget)
    _phase_drop(sim, cfg, st, step, sim_min, budget)


# --------------------------------------------------------------------------- #
# 観測記録(summary.json の "lost_property" キー。**OFF ではキー自体を出さない**)
# --------------------------------------------------------------------------- #
def _rate(num: dict, den: dict, item: str) -> float | None:
    d = int(den.get(item, 0))
    return None if d <= 0 else round(int(num.get(item, 0)) / d, 4)


def provenance(sim) -> dict | None:
    """``summary.json`` の ``lost_property`` キー(既定 OFF は None = キー自体を出さない)。

    ★**品目別返還率**(= 本レーンの検証ターゲット)を分母つきで残す。分母は実測の定義に
      合わせて「届けられた件数」(``turnins``)。実測帯は 傘 ≈1% / 財布 60-80% / 携帯 87%。
    ★貨幣保存の第 3 項(``hold`` = 世界に在るが誰の残高でもない現金)も一級市民として出す。
    """
    if not enabled(sim):
        return None
    st = getattr(sim, "_lost_state", None)
    cfg = cfg_of(sim)
    out: dict = {"schema": SCHEMA,
                 "items": list(ITEMS),
                 "base_daily": {k: float(cfg["base_daily"][k]) for k in ITEMS},
                 "statute_steps": int(cfg["statute_steps"])}
    if st is None:                                 # ON だが 1 件も落ちていない
        out.update({"drops": {}, "notices": {}, "pickups": {}, "passed_by": {},
                    "turnins": {}, "returns": {}, "keeps": {}, "expired": {},
                    "lapses": {}, "return_rate": {}, "turnin_rate": {},
                    "live_items": 0, "cash_held": 0.0, "cash_lapsed": 0.0,
                    "reward_paid": 0.0, "kept_cash": 0.0})
        return out
    out.update({
        "drops": {k: int(v) for k, v in sorted(st["drops"].items())},
        "notices": {k: int(v) for k, v in sorted(st["notices"].items())},
        "pickups": {k: int(v) for k, v in sorted(st["pickups"].items())},
        "passed_by": {k: int(v) for k, v in sorted(st["passed"].items())},
        "turnins": {k: int(v) for k, v in sorted(st["turnins"].items())},
        "returns": {k: int(v) for k, v in sorted(st["returns"].items())},
        "keeps": {k: int(v) for k, v in sorted(st["keeps"].items())},
        "expired": {k: int(v) for k, v in sorted(st["expired"].items())},
        "lapses": {k: int(v) for k, v in sorted(st["lapses"].items())},
        # 返還率 = 返還 / **届けられた件数**(実測 = 拾得受理件数が分母。定義を揃える)
        "return_rate": {k: r for k in ITEMS
                        if (r := _rate(st["returns"], st["turnins"], k)) is not None},
        # 届出率 = 届けられた / 拾われた(1 − これ = 着服率)
        "turnin_rate": {k: r for k in ITEMS
                        if (r := _rate(st["turnins"], st["pickups"], k)) is not None},
        "live_items": len(st["items"]),
        "cash_held": round(float(st["hold"]), 1),
        "cash_lapsed": round(float(st["lapsed"]), 1),
        "reward_paid": round(float(st["reward_paid"]), 1),
        "kept_cash": round(float(st["kept_cash"]), 1),
    })
    return out


def cash_in_flight(sim) -> float:
    """いま遺失物の中に在る現金(**読むだけ**・state も属性も生やさない)。

    貨幣保存の検査に使う第 3 項: ``Σ agent.money + cash_in_flight + 失効累計`` が一定。
    """
    st = getattr(sim, "_lost_state", None)
    return 0.0 if st is None else float(st["hold"])
