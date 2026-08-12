"""都市運営 = **見えない労働力**(Wave 4 III-4 ``city_ops``・既定 OFF)。

何を解く問題か
--------------
街は「そこで暮らす人」と「そこへ遊びに来る人」だけでは回らない。現実の渋谷を成り立たせて
いるのは、ほとんどの人が名前を知らないまま毎日その仕事の結果だけを受け取っている層である:

- **交番**に立つ警察官。地図には既に警察施設の POI があるのに、名簿の警察官 140 人は
  **どこにも束ねられていなかった**(執行機構 ``POLICE_OCCS`` は「同じノードに居る警察官」を
  探すのに、警察官はランダムな職場に立っていた)。
- **ごみ収集**。区の家庭ごみは可燃 2 回/週の**地区別の曜日表**で回り、繁華街は朝 **07:30**
  までに出し切る。事業系ごみは民間業者が**深夜**に回る。どちらもシムには 1 人も居なかった。
- **ビルの夜間清掃**。この街最大のサービス業でありながら、夜が構造的に空だったので
  (Wave 4 III-1 が開けた)働く場所も時間も表現できなかった。
- **納品**。物流①(``goods``)の補充トリップは ``agent_id=-1`` の世界イベントで、
  **運転席に誰も座っていなかった**。現実の納品は 04:00〜11:00 に荷捌き場へ集中する。
- **救急**。区内の救急出動は実測 **1 日 65 件**(約 22 分に 1 件)。世界にはその出動も、
  出動を呼ぶ「その場に居合わせた誰か」も居なかった。

本 module はこの層を **tier0(LLM 追加呼ゼロ・乱数 stream ゼロ)**で載せる。

★**救急はスケジュールではなく行為の連鎖である**(ユーザー指定の設計原則)
------------------------------------------------------------------------
初版の設計は「1 日 n 件を決定論の時刻表で撃つ」だったが、これは
「**世界の出来事はすべて誰かの行為に対応していなければならない**」という本プロジェクトの
原則に反する(無主体の確率イベント = ``diversity`` の nuisance を街路の生業へ作り替えた
Wave 4 III-3 と同じ轍)。そこで救急は **3 段の因果連鎖**として作り直した:

    ① ``collapse``     … **既存の健康機構**(``health.py`` の ``sick`` / ``fatigue``)が
                          既に成立させている状態から、その日その個体が倒れる
                          (= 身体の出来事 = cause_type ``physics``)
    ② ``ems_call``     … **近くに居合わせた誰か**が通報する(= 行為 = cause_type ``agent``)。
                          誰も居なければ本人が自分で呼ぶ(``self_call``)
    ③ ``ems_dispatch`` … 当直の救急隊が**その通報に応えて**現場へ向かう
                          (= 行為。因果の親は ②。時刻表ではない)

★**health.py には 1 バイトも書いていない**(あちらは別レーンの所有)。本 module は
  ``sick`` / ``sick_until`` / ``fatigue`` を**読むだけ**で、倒れる判定は
  (agent id, 経過日)の安定ハッシュという決定論の純関数である。したがって
  **健康機構が OFF(または ``onset_prob=0``)のランでは救急の連鎖は 1 件も起きない**。
  これは取りこぼしではなく正直な依存関係で、``ems.require_sick=false`` にすれば
  疲労ゲージ(既存)を代替の引き金にできる。

正直な限界(救急。**後続の設計判断が要る箇所**)
------------------------------------------------
1. **患者の医学的な状態は 1 つも作っていない**。倒れた個体は既存の睡眠機構で
   ``treatment_steps`` のあいだその場に留まり、そのあと**元の状態のまま**起きる
   (``sick`` は ``health`` 側の日程どおりに治り、本 module は治さないし悪くもしない)。
   入院・搬送先・重症度・死は**作らない**。理由: それらは「健康とは何か」の意味論を
   決める話で、既存の ``illness`` が単一の真偽値しか持たない現状では、ここで決め打つと
   健康機構の後続設計を縛ってしまう。★申し送りは**果たされた**: 重症度階層は
   ``health.severity``(H1)が持ち、``_collapse_gate`` は「S3/S4 への遷移」という
   **既存状態の読み取り**に置き換わった。搬送先・入院・医療費は ``medical.py``(H2)が
   持ち、本 module は「出動が決まった」ことを 1 行で伝えるだけである(所有の分離)。
2. **回復は既存の ``wake_up`` で観測される**。専用の ``ems_recover`` 種は作らない
   (出来事を 2 度数えないため)。``wake_up`` の ``slept_steps`` は本人の通常の睡眠長を
   載せる既存仕様なので、倒れて起きた回のその欄は**意味を持たない**(正直な既知の粗さ)。
3. **救急隊は「持ち場が現場へ移る」形で駆けつける**。専用の移動機構は作らず、
   ``work_node`` を現場へ差し替えて既存 routine に歩かせる(``transit_staff.bind`` /
   ``street_life.bind`` と同じ作法)。したがって ``response_min`` は**距離から計算した
   モデル値**であって、実際に何 step で着いたかとは別物である(到着率は provenance に出る)。

正直な限界(救急以外。**実測して判ったこと**を含む)
----------------------------------------------------
4. **深夜に働く役割は「名簿の就寝時刻」に負ける**。深夜の事業系ごみ収集(02:00〜05:00)は
   夜間開放 ON でも、名簿(L5 pool)の就寝時刻が一般住民と同じ生成器(22:00〜01:50 就寝)
   から来るため、その時刻に起きている作業員が居なければ **1 巡回も立たない**
   (実測: 既定名簿 + 夜間開放 ON の 1 日スモークで深夜班 2 人・巡回 0 件)。
   ★これは本 module の欠陥ではなく**名簿側の前提**である(``world.night_economy`` の
   registry 宣言が「本トグルを ON にしても台帳/プールを再生成しない限り夜勤者は 1 人も
   居ない」と明記しているのと同じ構造)。稼働率は provenance の ``rounds_by_kind`` に
   ``ward`` / ``commercial`` 別で出るので、0 のまま黙って通り過ぎることはない。
5. **持ち場に着けなければ何も起きない**(``street_life`` の限界 1 と同じ)。夜間清掃は
   「担当ビルの中に居る」ことが条件なので、routine が別の用事を優先した個体はその晩
   1 件も出さない。救急隊の現着率は provenance の ``ems_arrived`` に出る。
6. **車両の実体を作っていない**。納品ドライバーは同じ step に複数の補充トリップを
   持ちうるし、収集車も救急車もオブジェクトとしては存在しない
   (``transit_staff`` が列車エンティティを持てないと宣言したのと同じ線引き)。
7. **地区は区の収集区割りではない**。地図 bbox の格子分割で作った序数の地区であって、
   実在の町丁目でも収集ブロックでもない(再現したのは**曜日表という構造**である)。
8. **resume で巡回・清掃・倒れの印が切れる**: 状態は agent の動的属性なので checkpoint の
   対象外(第59 assets / 第96 traces / Wave 4 III-3 と同じ流儀)。持ち場の割当は地図と
   名簿順からの決定論なので resume 後も同じ(``bind`` は冪等)。

制度的背景(実装が参照した現実の構造。ブランド名・実在個人名は 1 つも書かない)
--------------------------------------------------------------------------------
- **交番**: 地図の警察施設 POI(``service`` カテゴリで名称に「交番」「警察署」を含む)へ
  名簿順で 3 直を回す。★**全員は持ち場に付けない**: 名簿(L5)の警察官には
  「巡回ユニット」という post が実在するので、``patrol_every`` 人に 1 人は持ち場を
  持たないまま街を回る(既存の執行機構がその個体を街中で使う)。
- **家庭ごみ**: 可燃 2 回/週・**地区別の曜日表**(月木 / 火金 / 水土)・朝の収集・
  繁華街の期限 **07:30**。地区は**地図の格子分割**で作る(区の実際の収集区割りデータは
  持っていないので、持っているふりをしない = 再現したのは**構造**であって区割りではない)。
- **事業系ごみ**: 民間業者が **02:00〜05:00** に飲食・夜間店舗の密なノードを回る。
  ★**夜間開放(``world.night_economy``)が ON でないと成立しない**: 深夜に起きている
  作業員が世界に居ないからである。OFF のときは深夜班を**束ねない**(黙って空回りさせない)。
- **ビル夜間清掃**: 22:00〜05:00(日跨ぎ)。``office`` 建物へ束ねる。日跨ぎ勤務窓は
  夜間開放 ON のときだけ表現できる(``work_wraps``)ので、OFF では 22:00〜24:00 へ畳む。
- **納品**: 04:00〜11:00 に荷捌き場へ集中。物流①の補充トリップに**運転手を割り当てる**
  (時刻も数量も 1 も変えない = 変えたのは「誰が運んだか」だけ)。
- **救急**: 区内 1 日 65 件(約 22 分に 1 件)。★本 module はこの数字を**判定に使わない**
  (上記のとおり出動は通報から起きる)。conf には**較正の物差し**として置き、
  provenance が「実測の通報件数/日 vs 参照値 × 規模比」を並べて出す。

★**自販機の補充は本波では作らない**(スキップの明示。**v8 を実査した上での判断**)
--------------------------------------------------------------------------------
現実の補充は「1 人が 1 日に 10〜30 台を回る」という明確な労働の形を持つが、
**地図に自販機の POI が 1 件も無い**。当初の見立ては「v7 に ``subcat`` が無いから」
だったが、実際に v8(``data/shibuya_osm_wide_v8.json``)を数えたところ
``subcat`` は通っている(convenience 93 / love_hotel 63 / karaoke 17 / club 17 /
sauna 4 / net_cafe 1 / hospital 7 …)のに、**自販機は v8 でも 0 件**だった
(OSM の ``amenity=vending_machine`` を地図生成器が拾っていない)。
無い POI に補充へ行かせると**行き先を捏造する**ことになるので作らない。
必要になったら地図生成器側で自販機を通すのが先で、そのとき本 module に
``restock_vending`` 系の役割を足せば成立する(前方互換のためのメモ = 実装はしない)。
★同じ実査で判ったこと(**救急の後続設計への申し送り**): v8 には ``hospital`` subcat が
  7 件ある。搬送先を持つ救急(重症度 → 搬送 → 入院)を作るときの受け皿はもう地図側に居る。

R1 ドクトリン
-------------
- 既定 ``city_ops.enabled: false`` では ``phase`` / ``bind`` / ``assign_delivery_driver``
  が即 return し、**L1 に 1 件も出ず・agent に属性が生えず・sim に state が生えず・
  プロンプトが 1 バイトも変わらない**(= ゴールデン L1 バイト一致)。
- **乱数 stream を 1 本も引かない**(全経路が id・step・sim_min・地図・暦の純関数。
  倒れる判定も (id, 日) の安定ハッシュ = ``goods.pick_item`` と同じ流儀)。
  tests/test_city_ops.py が AST でこれを機械固定する。
- **generate() の呼び出しサイトを 1 つも作らない**(LLM 追加呼ゼロ = k 非依存)。
  出来事は**既存の記憶機構**(``agent.remember``)へ定型 1 行が入るだけで、
  プロンプトの**欄**は 1 つも増えない(``street_life`` と同じ線引き)。
- 持ち場への**移動は既存の routine が行う**。``bind`` が ``work_node`` /
  ``work_start_min`` / ``work_end_min`` を与えるだけで「出勤する」動きは 1 行も書かない。

既存機構との関係(重複させない)
--------------------------------
| 既存                                  | 本 module との関係 |
|---|---|
| ``scheduler._phase_enforcement``      | 警察官を**交番へ立たせるだけ**。摘発の判定・罰金・grievance は 1 バイトも触らない(あちらが同ノードの警察官を探す条件が、現実的な配置の下で評価されるようになる) |
| ``street_life`` の条例パトロール      | 同上(警察官の位置が変わるだけ)。本 module は客引き機構に 1 行も触らない |
| ``goods.review_and_order``            | 補充トリップの**時刻・数量・(s,S) 判定は不変**。``agent_id`` と payload の ``driver`` / ``unstaffed`` だけを足す(``transit_staff`` の無人マーカーと同じ正直さの作法) |
| ``delivery``(宅配④)                 | **触らない**。あちらの配達員は既にアクター化済み(``_select_courier``)で、本 module が作るのは**納品**(店へ商品を運ぶ側)= 別の職能 |
| ``health``(illness / fatigue)        | **読むだけ**。倒れる・治すの判定は本 module に閉じ、``sick`` を書き換えない |
| ``night``(夜間開放)                  | 深夜班・夜間清掃の**前提**として読むだけ。OFF では該当の班を束ねない |
"""
from __future__ import annotations

import hashlib

from .observer.schema import Event, register_event_kind

SCHEMA = 1

# --------------------------------------------------------------------------- #
# L1 イベント種(**材料側 registration**: devices.py / transit_staff.py /
# street_life.py と同じ流儀 = observer/schema.py 本体を触らずに自分の種を自分で登録する)。
# ★payload に自由文は 1 つも入らない(district は本 module が作る序数名・node/building は
#   世界の識別子・残りは件数と時刻)。
# --------------------------------------------------------------------------- #
register_event_kind("city_ops_bound", "都市運営の担い手を持ち場へ束ねた統計。**起動時 1 回**"
                                      "(step 0 のみ){police, patrol, waste, waste_night,"
                                      " driver, cleaner, ems, posts, districts, notes}")
register_event_kind("waste_collection", "ごみ収集の 1 巡回(1 班 1 日 1 件)。"
                                        "agent_id = 収集作業員"
                                        "{district, kind, nodes, stops, deadline_min}")
register_event_kind("night_cleaning", "ビルの夜間清掃(1 棟 1 晩 1 件)。agent_id = 清掃員"
                                      "{building, node, start_min}")
register_event_kind("collapse", "その場で倒れた(既存の健康状態からの決定論。"
                                "agent_id = 本人){node, source, until_step}")
register_event_kind("ems_call", "救急への通報(**この行為が出動の原因**)。agent_id = 通報者"
                                "{patient, node, self_call, dist_m}")
register_event_kind("ems_dispatch", "通報に応えた救急隊の出動。agent_id = 隊員"
                                    "(当直不在なら -1 かつ payload.unstaffed=true)"
                                    "{node, patient, caller, response_min, crew, unstaffed}")


# --------------------------------------------------------------------------- #
# 役割(= persona の occupation。scripts/build_persona_pool.py の _L5_ROLES と同じ語)
# --------------------------------------------------------------------------- #
WASTE = "清掃作業員"
DRIVER = "納品ドライバー"
NIGHT_CLEANER = "夜間清掃員"
EMS_CREW = "救急隊員"

# 既存名簿にある職(新設しない)。警察官 = 交番へ束ねる対象・消防士 = 救急の担い手の候補。
POLICE_OCCS = ("警察官",)
FIREFIGHTER = "消防士"

#: 本 module が新設する L5 役割(名簿の occupation 判定に使う)
CITY_OPS_OCCS: tuple[str, ...] = (WASTE, DRIVER, NIGHT_CLEANER, EMS_CREW)

# --------------------------------------------------------------------------- #
# 記憶の 1 行(**役割だけの純関数**。数字・金額・config・実験条件・機構語は 1 つも入らない)
# = traces.sentence / street_life.NOTICE_TEXT と同じ no-fingerprint の定型文規約。
# --------------------------------------------------------------------------- #
COLLAPSE_TEXT = "急に具合が悪くなり、その場から動けなくなった。"
CALL_TEXT = "目の前で人が倒れたので、救急に通報した。"
SELF_CALL_TEXT = "自分で救急に通報した。"
DISPATCH_TEXT = "通報を受けて現場へ向かった。"
WASTE_TEXT = "担当の地区を回ってごみを集めた。"
CLEAN_TEXT = "夜のビルを清掃した。"

# 収集の曜日表(0=月 … 6=日)。可燃 2 回/週を地区ごとに 3 組で回す(月木 / 火金 / 水土)。
_WEEKDAY_PAIRS = ((0, 3), (1, 4), (2, 5))

DEFAULTS: dict = {
    "enabled": False,
    "max_events_per_step": 24,     # 1 step に出す本 module の L1 件数の上限(暴走の安全弁)
    # ---- ① 交番配置 ----
    "police": {
        "enabled": True,
        "occupations": (POLICE_OCCS[0],),
        # 地図の警察施設が入っているカテゴリ。v7/v8 とも build_map が amenity=police を
        # ``service`` へ潰すので既定はそれ。``police`` は将来の地図が独立カテゴリを
        # 通したときのための前方互換(今のどの地図にも 1 件も無い = 完全な no-op)。
        "poi_cats": ("service", "police"),
        "name_keywords": ("交番", "警察署"),   # 一般名詞のみ(ブランド名・地名は書かない)
        "first_open": 0,               # 最初の直の始まり[分 of day]
        "shift_hours": 8,
        "n_shifts": 3,                 # 24 時間を 3 直で回す(交番の実態)
        "patrol_every": 4,             # 名簿順で n 人に 1 人は持ち場を持たない巡回ユニット
    },
    # ---- ② ごみ収集 ----
    "waste": {
        "enabled": True,
        "occupations": (WASTE,),
        "grid_cols": 3,                # 地区分割の格子(地図 bbox を等分)
        "grid_rows": 2,
        "stops_per_round": 6,          # 1 巡回で回るノード数
        "ward_start_min": 5 * 60,      # 区の家庭ごみ収集の開始
        "ward_deadline_min": 7 * 60 + 30,   # 繁華街の収集期限 07:30
        "commercial_enabled": True,
        "commercial_start_min": 2 * 60,     # 事業系(民間)= 深夜 02:00
        "commercial_end_min": 5 * 60,
        "commercial_every": 2,         # 名簿順で n 人に 1 人が深夜班
        "commercial_cats": ("nightlife", "food"),
        "require_night": True,         # 夜間開放 OFF では深夜班を束ねない(正直な依存)
    },
    # ---- ③ 納品の運転手化 ----
    "driver": {
        "enabled": True,
        "occupations": (DRIVER,),
        "start_min": 4 * 60,           # 荷捌きの集中帯 04:00
        "end_min": 11 * 60,            # 〜11:00
        "stops_per_round": 8,
        "stop_cats": ("shop", "food", "cafe", "nightlife"),
    },
    # ---- ④ ビル夜間清掃 ----
    "night_cleaning": {
        "enabled": True,
        "occupations": (NIGHT_CLEANER, "夜間清掃"),   # L5 新役割 + 台帳(build_orgs)の夜勤ロール
        "start_min": 22 * 60,
        "end_min": 5 * 60,             # 日跨ぎ(夜間開放 OFF では 24:00 へ畳む)
        # ★1 人 1 棟(``bind`` が建物 id 昇順で循環割当)。「1 人が複数棟を回る」形は
        #   作らない: 巡回の途中経過を表現する機構(収集班の停留列)を清掃にも持ち込むと、
        #   建物への入退館が毎晩複数回起きて L1 が膨らむ。必要になってから足す。
        "max_per_night": 16,           # 1 晩に出す night_cleaning の上限
    },
    # ---- ⑤ 救急(**通報から始まる行為の連鎖**)----
    "ems": {
        "enabled": True,
        "crew_occupations": (EMS_CREW, FIREFIGHTER),
        "first_open": 0,
        "shift_hours": 8,
        "n_shifts": 3,
        # 倒れる判定: 既存の健康状態が成立している個体のうち、1 万人に n 人/日(決定論)。
        "collapse_per_10k": 400,
        "require_sick": True,          # false = 疲労ゲージも引き金にする
        "collapse_fatigue": 0.95,      # require_sick=false のときの疲労側の閾値
        "call_radius_m": 40.0,         # 通報者を探す半径(既定 = 世界の知覚半径と同水準)
        "treatment_steps": 3,          # その場に留まる長さ [step]
        "on_scene_steps": 3,           # 救急隊が現場に留まる長さ [step]
        "speed_m_per_min": 300.0,      # response_min の換算(モデル値。実走行ではない)
        "max_calls_per_day": 8,        # 1 日に出す連鎖の上限(L1 の安全弁)
        "ward_calls_per_day": 65.0,    # ★参照値のみ(実測 65 件/日)。判定には使わない
        # ---- 傍観者モデル(**重症度 H1 が ON のときだけ効く**。OFF では最近傍が必ず通報する)
        #  実測との整合: 集団としての P(誰かが動く) は 0.85〜0.95(実 CCTV 研究で介入は規範・91%)。
        #  Darley-Latané 型の人数減衰は**曖昧で軽微な状況にだけ**効かせ、危険事態は減衰なし
        #  (メタ分析: 危険が明白なほど傍観者効果は消え、むしろ人数が増えると助かる)。
        #    P(誰かが動く) = 1 − (1 − solo)^(n^(1−alpha))
        # solo=0.65 は「n=3〜5 で集団 P が 0.85〜0.95 に載る」ことから逆算した値
        # (tests/test_health_severity.py が帯の内側にあることを機械固定する)。
        "bystander_solo_prob": 0.65,      # 1 人しか居合わせないときの介入率
        "bystander_alpha_ambiguous": 0.4,  # 曖昧・軽微(見かけが中等症以下)の責任分散
        "bystander_alpha_danger": 0.0,     # 危険が明白(見かけが重症以上)= 減衰なし
        "bystander_related_mult": 1.5,     # 家族・同居人・同僚が居合わせたときの倍率
        "bystander_max_prob": 0.98,
        # ---- 通報までの所要時間の分布(実測 46% 1分未満 / 29% 1〜5分 / 25% 5分超)
        #  ★正直な限界: 既定 Δt=10 分ではどのビンも同じ 1 刻みに丸まるため、**発火時刻は
        #    動かさず観測量として payload に載せる**(より細かい Δt で意味を持つ)。
        "call_delay_fast_min": 0.5,
        "call_delay_mid_min": 3.0,
        "call_delay_slow_min": 8.0,
        "call_delay_fast_share": 0.46,
        "call_delay_mid_share": 0.29,
    },
}

_TOP_INT = ("max_events_per_step",)


# --------------------------------------------------------------------------- #
# cfg 正準化(transit_staff.build_cfg / street_life.build_cfg と同型:
#   dict / OmegaConf 両対応・dotlist の文字列を型強制・未知キーは黙って捨てる)
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
    """語リストの正準化(空白除去・空要素落とし・**宣言順を保つ**)。空なら既定へ戻す。"""
    got = _to_plain(raw)
    if isinstance(got, str):
        got = [got]
    out = tuple(str(x).strip() for x in (got or ()) if str(x).strip())
    return out or tuple(fallback)


def _block(raw, defaults: dict, words_keys: tuple, floats: tuple,
           bools: tuple) -> dict:
    """サブブロック 1 つの正準化(既定を複製し、来たキーだけ型強制して上書きする)。"""
    out = dict(defaults)
    got = dict(_to_plain(raw) or {})
    for key, value in got.items():
        if key not in out:
            continue                               # 未知キーは捨てる(捏造しない)
        if key in bools:
            out[key] = bool(value)
        elif key in words_keys:
            out[key] = _words(value, defaults[key])
        elif key in floats:
            out[key] = max(0.0, float(value))
        else:
            out[key] = int(value)
    return out


def build_cfg(raw) -> dict:
    """conf の ``city_ops`` ブロックを型強制つきで正準化(既定 OFF=現行挙動と完全同一)。"""
    raw = dict(_to_plain(raw) or {})
    cfg: dict = {"enabled": bool(raw.get("enabled", False))}
    for key in _TOP_INT:
        cfg[key] = max(0, int(raw.get(key, DEFAULTS[key])))
    cfg["police"] = _block(raw.get("police"), DEFAULTS["police"],
                           ("occupations", "poi_cats", "name_keywords"), (),
                           ("enabled",))
    cfg["waste"] = _block(raw.get("waste"), DEFAULTS["waste"],
                          ("occupations", "commercial_cats"), (),
                          ("enabled", "commercial_enabled", "require_night"))
    cfg["driver"] = _block(raw.get("driver"), DEFAULTS["driver"],
                           ("occupations", "stop_cats"), (), ("enabled",))
    cfg["night_cleaning"] = _block(raw.get("night_cleaning"),
                                   DEFAULTS["night_cleaning"], ("occupations",), (),
                                   ("enabled",))
    cfg["ems"] = _block(raw.get("ems"), DEFAULTS["ems"], ("crew_occupations",),
                        ("call_radius_m", "collapse_fatigue", "speed_m_per_min",
                         "ward_calls_per_day", "bystander_solo_prob",
                         "bystander_alpha_ambiguous", "bystander_alpha_danger",
                         "bystander_related_mult", "bystander_max_prob",
                         "call_delay_fast_min", "call_delay_mid_min",
                         "call_delay_slow_min", "call_delay_fast_share",
                         "call_delay_mid_share"), ("enabled", "require_sick"))
    # 0 除算・退化の防止(刻みと区画は 1 以上)
    for blk, keys in (("police", ("shift_hours", "n_shifts", "patrol_every")),
                      ("waste", ("grid_cols", "grid_rows", "stops_per_round",
                                 "commercial_every")),
                      ("driver", ("stops_per_round",)),
                      ("ems", ("shift_hours", "n_shifts"))):
        for key in keys:
            cfg[blk][key] = max(1, int(cfg[blk][key]))
    return cfg


def cfg_of(sim) -> dict:
    """都市運営の設定(初回のみ ``sim.cfg.city_ops`` から遅延構築してキャッシュ)。

    simulation.py は別レーン所有なので cfg は本 module が遅延構築する
    (``traces.cfg_of`` / ``street_life.cfg_of`` と同型)。キャッシュ属性 ``sim.cityopscfg``
    は L1/L2/L3/乱数に一切現れない = 既定 OFF のバイト一致を壊さない。
    """
    got = getattr(sim, "cityopscfg", None)
    if got is None:
        try:
            raw = sim.cfg.get("city_ops", None)
        except Exception:                          # noqa: BLE001(旧 config 互換)
            raw = None
        got = build_cfg(raw)
        sim.cityopscfg = got
    return got


def enabled(sim) -> bool:
    """都市運営層(Wave 4 III-4)が有効か。既定 OFF=新経路を一切通さない(バイト一致)。"""
    return bool(cfg_of(sim)["enabled"])


# --------------------------------------------------------------------------- #
# 決定論の補助(**乱数を 1 本も引かない**ための道具)
# --------------------------------------------------------------------------- #
def _stable_hash(value: str) -> int:
    """プロセス非依存の安定ハッシュ(rng.py / goods.py と同流儀=決定論リプレイの土台)。"""
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def duty_window(first_open: int, shift_hours: int, n_shifts: int,
                index: int) -> tuple[int, int]:
    """当直の窓 ``(open, close)``[分 of day]。**名簿順 index の純関数**(乱数ゼロ)。

    ``transit_staff.duty_window`` と同じ規約(日を跨ぐ直は 24:00 で切る = 既存の
    ``routine.in_work_window`` が ``start <= 分 of day < end`` しか表現できないため)。
    """
    n = max(1, int(n_shifts))
    hours = max(1, int(shift_hours))
    open_min = int(first_open) + (int(index) % n) * hours * 60
    if open_min >= 1440:
        open_min = int(first_open)
    return open_min, min(1440, open_min + hours * 60)


def _in_window(start: int, end: int, sim_min: int) -> bool:
    """分 of day が [start, end) か(start > end は日跨ぎの帯)。"""
    m = int(sim_min) % 1440
    s, e = int(start), int(end)
    return (s <= m < e) if s <= e else (m >= s or m < e)


def _night_on(sim) -> bool:
    """夜間開放(Wave 4 III-1)が有効か。**読むだけ**(あちらの state を触らない)。"""
    ncfg = getattr(sim, "nightcfg", None)
    return bool(ncfg and ncfg["enabled"])


def _midnight_shift_on(sim) -> bool:
    """日跨ぎ勤務窓が表現できるか(夜間清掃の窓を畳むかの判断材料)。"""
    ncfg = getattr(sim, "nightcfg", None)
    return bool(ncfg and ncfg["enabled"] and ncfg["midnight_shift"])


def _weekday(sim, sim_min: int) -> int:
    """曜日(0=月 … 6=日)。暦が無い設定でも経過日から決定論で決める(純関数)。

    ★暦(``world.calendar``)が **有効なとき**だけ実日付の曜日を使う。無効なランでは
      「暦が世界に存在しない」ので実日付を持ち出さず、経過日の剰余(初日=月)で回す
      (``routine.in_work_window`` が ``cal.enabled`` を見るのと同じ線引き)。
    """
    cal = getattr(sim, "calendarcfg", None)
    if cal and cal.get("enabled"):
        try:
            from .world import calendar as _calendar
            return int(_calendar.weekday_of(cal, int(sim_min)))
        except Exception:                          # noqa: BLE001(旧 config 互換)
            pass
    return (int(sim_min) // 1440) % 7


def _npois(sim, node: str, cats: tuple = ()) -> int:
    pois = sim.city.pois_at_node(node)
    if not cats:
        return len(pois)
    return sum(1 for p in pois if str(p.get("cat") or "") in cats)


# --------------------------------------------------------------------------- #
# 地区 = **地図の格子分割**(区の実際の収集区割りは持っていないので作らない)
# --------------------------------------------------------------------------- #
def districts(sim) -> list[dict]:
    """地区の一覧(``[{name, nodes, ward_stops, night_stops}]``)。地図の純関数・乱数ゼロ。

    地図の bbox を ``grid_cols × grid_rows`` で等分し、ノードを 1 つ以上含む区画だけを
    (行, 列) 昇順で並べて ``d0, d1, …`` と序数で名づける。★**地名を 1 つも書かない**
    (``street_life.posts`` と同じ理由 = ETHICS §2 と、どの地図でも動くようにするため)。

    各地区の収集経路は「ごみが出る場所」= POI の多いノードから ``stops_per_round`` 件
    (件数降順 → ノード名昇順のタイブレーク = 決定論)。深夜の事業系は
    ``commercial_cats``(飲食・夜間店舗)の POI があるノードだけを同じ規則で並べる。
    """
    cache = getattr(sim, "_co_districts", None)
    if cache is not None:
        return cache
    wcfg = cfg_of(sim)["waste"]
    cols, rows = int(wcfg["grid_cols"]), int(wcfg["grid_rows"])
    nodes = sorted(sim.city.graph.nodes)
    if not nodes:
        sim._co_districts = []
        return sim._co_districts
    xy = {n: sim.city.node_xy(n) for n in nodes}
    xs = [p[0] for p in xy.values()]
    ys = [p[1] for p in xy.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(1e-9, max_x - min_x)
    span_y = max(1e-9, max_y - min_y)
    cells: dict[tuple[int, int], list[str]] = {}
    for node in nodes:
        px, py = xy[node]
        col = min(cols - 1, int((px - min_x) / span_x * cols))
        row = min(rows - 1, int((py - min_y) / span_y * rows))
        cells.setdefault((row, col), []).append(node)
    stops = int(wcfg["stops_per_round"])
    cats = tuple(wcfg["commercial_cats"])
    out: list[dict] = []
    for index, key in enumerate(sorted(cells)):
        group = cells[key]
        ranked = sorted(group, key=lambda n: (-_npois(sim, n), n))
        night = [n for n in ranked if _npois(sim, n, cats) > 0]
        night.sort(key=lambda n: (-_npois(sim, n, cats), n))
        out.append({"name": f"d{index}", "nodes": tuple(group),
                    "ward_stops": tuple(ranked[:stops]),
                    "night_stops": tuple(night[:stops])})
    sim._co_districts = out
    return out


def police_posts(sim) -> list[str]:
    """交番・警察署のノード列(**地図の純関数**・乱数ゼロ・sim に 1 度だけキャッシュ)。

    判定は「``poi_cats`` のカテゴリ」かつ「名称に ``name_keywords`` の語を含む」の連言。
    ★名称キーワードを使うのは地図側に警察カテゴリが無いためで、語は**一般名詞のみ**
      (「交番」「警察署」)。``night.poi_subcat`` が一般名詞マッチの誤爆を避けて既定を
      空にしたのとは事情が違い、この 2 語は施設種そのものを指すので誤爆しない
      (カテゴリで絞ってあるのでバス停「〜交番前」= ``office`` は入らない)。
    """
    cache = getattr(sim, "_co_police_posts", None)
    if cache is not None:
        return cache
    pcfg = cfg_of(sim)["police"]
    cats = frozenset(pcfg["poi_cats"])
    words = tuple(pcfg["name_keywords"])
    seen: dict[str, str] = {}
    for poi in sim.city.poi_list:
        if str(poi.get("cat") or "") not in cats:
            continue
        name = str(poi.get("name") or "")
        if not any(w in name for w in words):
            continue
        seen[str(poi.get("id") or "")] = str(poi.get("node") or "")
    out = sorted({node for node in seen.values() if node})
    sim._co_police_posts = out
    return out


def _commercial_nodes(sim, cats: tuple, limit: int) -> list[str]:
    """商業ノード(納品先)の列。POI 件数降順 → ノード名昇順(決定論)。"""
    nodes = sorted(sim.city.graph.nodes)
    scored = [(n, _npois(sim, n, cats)) for n in nodes]
    scored = [t for t in scored if t[1] > 0]
    scored.sort(key=lambda t: (-t[1], t[0]))
    return [n for n, _c in scored[:max(1, int(limit))]]


def _office_buildings(sim) -> list[dict]:
    """夜間清掃の対象(``office`` 建物)。建物 id 昇順(決定論)。"""
    cache = getattr(sim, "_co_offices", None)
    if cache is None:
        cache = sorted((b for b in sim.city.buildings
                        if str(b.get("kind") or "") == "office"),
                       key=lambda b: str(b["id"]))
        sim._co_offices = cache
    return cache


def _ems_base(sim) -> str:
    """救急隊の待機拠点。★地図に消防・救急の POI カテゴリが**無い**ので捏造しない:
    公共施設(``public`` 建物)のうち駅に最も近い入口を代理に使い、無ければ駅ノード。"""
    cache = getattr(sim, "_co_ems_base", None)
    if cache is not None:
        return cache
    station = str(getattr(sim.city, "station_node", "") or "")
    pubs = [b for b in sim.city.buildings if str(b.get("kind") or "") == "public"]
    base = ""
    if pubs and station:
        sx, sy = sim.city.node_xy(station)

        def _key(bld):
            bx, by = sim.city.node_xy(bld["entrance"])
            return ((bx - sx) ** 2 + (by - sy) ** 2, str(bld["id"]))
        base = str(min(pubs, key=_key)["entrance"])
    if not base:
        base = station or (sorted(sim.city.graph.nodes)[0]
                           if sim.city.graph.nodes else "")
    sim._co_ems_base = base
    return base


# --------------------------------------------------------------------------- #
# state(sim 側。**OFF では 1 つも生えない**)
# --------------------------------------------------------------------------- #
def _state(sim) -> dict:
    st = getattr(sim, "_co_state", None)
    if st is None:
        st = {"schema": SCHEMA, "bound": False, "by_kind": {}, "dropped": 0,
              "rounds": 0, "cleanings": 0, "collapses": 0, "calls": 0,
              "self_calls": 0, "no_call": 0, "dispatches": 0,
              "dispatch_unstaffed": 0,
              "trips_staffed": 0, "trips_unstaffed": 0, "rounds_by_kind": {},
              "ems_arrived": 0, "calls_by_day": {}, "cleaned_night": {},
              "notes": []}
        sim._co_state = st
    return st


def _bump(table: dict, key: str, n: int = 1) -> None:
    table[key] = int(table.get(key, 0)) + int(n)


class _Budget:
    """1 step に出す L1 件数の上限(暴走時の L1 膨張対策。超過分は**捨てて数える**)。"""

    def __init__(self, sim, st: dict, cap: int):
        self.sim = sim
        self.st = st
        self.left = int(cap)

    def log(self, event) -> bool:
        if self.left <= 0:
            self.st["dropped"] += 1
            return False
        self.left -= 1
        self.sim.logger.log(event)
        _bump(self.st["by_kind"], event.kind)
        return True


# --------------------------------------------------------------------------- #
# 束ね(起動時 1 回・**冪等**・乱数ゼロ・resume 安全)
# --------------------------------------------------------------------------- #
def _roster(sim, occupations) -> list:
    """その職の個体(**agent id 昇順**)。名簿順が当直表と持ち場割当の唯一の入力。"""
    occs = frozenset(occupations)
    return [a for a in sorted(sim.agents, key=lambda z: int(z.id))
            if str(getattr(a, "occupation", "")) in occs
            or str(getattr(a, "role", "")) in occs]


def _stand_on_street(agent, node: str, open_min: int, close_min: int) -> None:
    """路面の持ち場へ立たせる(建物内勤務にしない)。移動は既存 routine が行う。"""
    agent.work_node = str(node)
    agent.work_building = ""
    agent.work_floor = 0
    agent.work_start_min = int(open_min)
    agent.work_end_min = int(close_min)


def bind(sim) -> dict:
    """都市運営の担い手を持ち場へ束ねる(決定論・乱数ゼロ・**冪等**・resume 安全)。

    ``work.bind_workplace``(別レーン所有)は地図に対応 POI カテゴリが無い職業を
    ``no_category`` として**正直に束ねられないまま**残す。本関数はその後に走り、
    ゲートが ON のときだけ「都市運営の持ち場」を与える(あちらの統計の意味は変えない)。

    - 並びは **agent id 昇順**(当直表・地区割当・持ち場の循環割当もこの並びで決まる)。
    - 勤務窓は**必ず**役割の帯で上書きする(``street_life.bind`` と同じ判断)。理由:
      これらの職業は**職業そのものが時間帯**で、名簿側の一般的な窓(9〜17 時)を
      残すと深夜の収集班が昼に出勤する。
    - 交番の警察官だけは**既存の窓を尊重しない代わりに持ち場を全員には与えない**
      (``patrol_every`` 人に 1 人は名簿の「巡回ユニット」のまま街を回る)。
    """
    out = {"police": 0, "patrol": 0, "waste": 0, "waste_night": 0, "driver": 0,
           "cleaner": 0, "ems": 0, "posts": 0, "districts": 0, "notes": []}
    if not enabled(sim):
        return out
    cfg = cfg_of(sim)
    notes: list[str] = out["notes"]

    # ---- ① 交番配置 ------------------------------------------------------- #
    pcfg = cfg["police"]
    if pcfg["enabled"]:
        posts = police_posts(sim)
        out["posts"] = len(posts)
        if not posts:
            notes.append("no_police_poi")
        else:
            every = max(1, int(pcfg["patrol_every"]))
            for index, agent in enumerate(_roster(sim, pcfg["occupations"])):
                if index % every == every - 1:
                    # 巡回ユニット(名簿 L5 の post に実在する): 持ち場を与えない。
                    agent.city_ops_patrol = True
                    out["patrol"] += 1
                    continue
                open_min, close_min = duty_window(
                    pcfg["first_open"], pcfg["shift_hours"], pcfg["n_shifts"], index)
                _stand_on_street(agent, posts[index % len(posts)], open_min, close_min)
                agent.city_ops_post = str(posts[index % len(posts)])
                out["police"] += 1

    # ---- ② ごみ収集 ------------------------------------------------------- #
    wcfg = cfg["waste"]
    if wcfg["enabled"]:
        groups = districts(sim)
        out["districts"] = len(groups)
        night_ok = (wcfg["commercial_enabled"]
                    and (_night_on(sim) or not wcfg["require_night"]))
        if wcfg["commercial_enabled"] and not night_ok:
            # ★正直な依存関係: 深夜に起きている作業員が世界に居ないので束ねない。
            notes.append("commercial_waste_needs_night_economy")
        for index, agent in enumerate(_roster(sim, wcfg["occupations"])):
            if not groups:
                break
            group = groups[index % len(groups)]
            agent.city_ops_district = str(group["name"])
            agent.city_ops_district_idx = int(index % len(groups))
            night = night_ok and (index % max(1, int(wcfg["commercial_every"]))
                                  == max(1, int(wcfg["commercial_every"])) - 1)
            agent.city_ops_waste_night = bool(night)
            if night:
                start, end = wcfg["commercial_start_min"], wcfg["commercial_end_min"]
                stops = group["night_stops"] or group["ward_stops"]
                out["waste_night"] += 1
            else:
                start, end = wcfg["ward_start_min"], wcfg["ward_deadline_min"]
                stops = group["ward_stops"]
                out["waste"] += 1
            if not stops:
                continue
            _stand_on_street(agent, stops[0], start, end)

    # ---- ③ 納品の運転手化 -------------------------------------------------- #
    dcfg = cfg["driver"]
    if dcfg["enabled"]:
        stops = _commercial_nodes(sim, tuple(dcfg["stop_cats"]),
                                  int(dcfg["stops_per_round"]))
        if not stops:
            notes.append("no_commercial_node")
        for index, agent in enumerate(_roster(sim, dcfg["occupations"])):
            if not stops:
                break
            agent.city_ops_driver = True
            _stand_on_street(agent, stops[index % len(stops)],
                             dcfg["start_min"], dcfg["end_min"])
            out["driver"] += 1

    # ---- ④ ビル夜間清掃 ---------------------------------------------------- #
    ncfg = cfg["night_cleaning"]
    if ncfg["enabled"]:
        blds = _office_buildings(sim)
        if not blds:
            notes.append("no_office_building")
        wraps = _midnight_shift_on(sim)
        if not wraps:
            # ★日跨ぎ勤務窓は夜間開放 ON でしか表現できない(work._window の丸め)。
            #   OFF では 22:00〜24:00 へ畳む = 夜の前半だけ働く正直な縮退。
            notes.append("night_cleaning_window_clamped")
        for index, agent in enumerate(_roster(sim, ncfg["occupations"])):
            if not blds:
                break
            bld = blds[index % len(blds)]
            agent.work_building = str(bld["id"])
            agent.work_node = str(bld["entrance"])
            agent.work_floor = 1
            agent.work_start_min = int(ncfg["start_min"])
            agent.work_end_min = (int(ncfg["end_min"]) if wraps else 1440)
            if wraps:
                agent.work_wraps = True
            agent.city_ops_clean_building = str(bld["id"])
            out["cleaner"] += 1

    # ---- ⑤ 救急 ------------------------------------------------------------ #
    ecfg = cfg["ems"]
    if ecfg["enabled"]:
        base = _ems_base(sim)
        if not base:
            notes.append("no_ems_base")
        for index, agent in enumerate(_roster(sim, ecfg["crew_occupations"])):
            if not base:
                break
            open_min, close_min = duty_window(
                ecfg["first_open"], ecfg["shift_hours"], ecfg["n_shifts"], index)
            _stand_on_street(agent, base, open_min, close_min)
            agent.city_ops_ems_base = str(base)
            out["ems"] += 1
    return out


def _ensure_bound(sim, step: int = -1, sim_min: int = 0) -> None:
    """束ねを 1 回だけ行う(冪等)。``phase`` と ``assign_delivery_driver`` の共通入口。

    ★``goods.review_and_order``(``_phase_goods``)は本 module の ``phase`` より**前**に
      走るので、納品ドライバーの問い合わせが先に来ることがある。そこで束ねは
      「最初に必要になった側」が行う(どちらから来ても同じ結果 = 冪等)。
    """
    st = _state(sim)
    if st["bound"]:
        return
    st["bound"] = True
    st["bind"] = bind(sim)
    if int(step) == 0:
        stat = dict(st["bind"])
        stat["notes"] = list(stat.get("notes") or [])
        sim.logger.log(Event(step=int(step), sim_min=int(sim_min), agent_id=-1,
                             kind="city_ops_bound", x=0.0, y=0.0, payload=stat))
        _bump(st["by_kind"], "city_ops_bound")


# =========================================================================== #
# ② ごみ収集(巡回)
# =========================================================================== #
def _collect_day(wcfg: dict, district_idx: int, weekday: int) -> bool:
    """その地区がその曜日に収集日か(可燃 2 回/週の地区別曜日表。純関数)。"""
    pair = _WEEKDAY_PAIRS[int(district_idx) % len(_WEEKDAY_PAIRS)]
    return int(weekday) in pair


def _waste_round(sim, cfg: dict, st: dict, bud: _Budget, agent,
                 step: int, sim_min: int) -> None:
    """収集作業員 1 人の 1 step(巡回の開始で ``waste_collection`` を 1 件)。

    経路は地区の停留ノード列を**時間で割った決定論の巡回**(``work_node`` を進めるだけ =
    移動は既存 routine が行う)。深夜の事業系班は夜間開放 ON のときだけ束ねられている。
    """
    wcfg = cfg["waste"]
    groups = districts(sim)
    if not groups:
        return
    idx = int(getattr(agent, "city_ops_district_idx", 0)) % len(groups)
    group = groups[idx]
    night = bool(getattr(agent, "city_ops_waste_night", False))
    if night:
        start, end = int(wcfg["commercial_start_min"]), int(wcfg["commercial_end_min"])
        stops = group["night_stops"] or group["ward_stops"]
        kind = "commercial"
    else:
        start, end = int(wcfg["ward_start_min"]), int(wcfg["ward_deadline_min"])
        stops = group["ward_stops"]
        kind = "ward"
    if not stops or not _in_window(start, end, sim_min):
        return
    # ★事業系(深夜)は曜日を選ばない(民間の契約回収は毎日)。区の家庭ごみだけが曜日表。
    if not night and not _collect_day(wcfg, idx, _weekday(sim, sim_min)):
        return
    # 巡回位置 = 帯を停留数で割った決定論(乱数ゼロ・Δt 非依存)
    span = (end - start) if start <= end else (1440 - start + end)
    elapsed = (int(sim_min) % 1440) - start
    if elapsed < 0:
        elapsed += 1440
    slot = min(len(stops) - 1, int(elapsed * len(stops) // max(1, span)))
    agent.work_node = str(stops[slot])
    day = int(sim_min) // 1440
    key = f"{day}:{kind}"
    if getattr(agent, "city_ops_round", None) == key:
        return                                     # 1 班 1 日 1 件(L1 の throttle)
    agent.city_ops_round = key
    bud.log(Event(step=step, sim_min=sim_min, agent_id=int(agent.id),
                  kind="waste_collection", x=agent.x, y=agent.y,
                  payload={"district": str(group["name"]), "kind": kind,
                           "nodes": [str(n) for n in stops], "stops": len(stops),
                           "deadline_min": int(end)}))
    st["rounds"] += 1
    _bump(st["rounds_by_kind"], kind)
    agent.remember(WASTE_TEXT)


# =========================================================================== #
# ③ 納品の運転手化(``goods.review_and_order`` からの唯一の口)
# =========================================================================== #
def assign_delivery_driver(sim, node: str, step: int, sim_min: int):
    """補充トリップの運転手(居なければ ``None``)。**OFF では必ず ``None``**。

    ★``goods`` 側の呼び出しは「``None`` なら 1 バイトも変えない」形にしてあるので、
      既定 OFF では ``delivery_trip`` の payload も ``agent_id=-1`` も従来どおり
      (= ゴールデン L1 バイト一致)。
    ★選定は決定論: 当直中(勤務窓の中・起きている・街の中)の納品ドライバーのうち
      **納品先に最も近い個体**(同距離は id 昇順)。1 step に何本トリップが出ても
      同じ運転手が複数を持ちうる(車両の実体を作っていないため = 正直な近似)。
    ★世界状態は 1 つも動かさない(``work_node`` すら触らない)= 記録の帰属だけ。
    """
    if not enabled(sim):
        return None
    cfg = cfg_of(sim)
    if not cfg["driver"]["enabled"]:
        return None
    _ensure_bound(sim, step, sim_min)
    dcfg = cfg["driver"]
    if not _in_window(dcfg["start_min"], dcfg["end_min"], sim_min):
        return None
    occs = frozenset(dcfg["occupations"])
    try:
        tx, ty = sim.city.node_xy(str(node))
    except Exception:                              # noqa: BLE001(未知ノードの保険)
        tx, ty = 0.0, 0.0
    best = None
    best_key = None
    for agent in sim.agents:
        if str(getattr(agent, "occupation", "")) not in occs:
            continue
        if agent.loc == "outside" or agent.sleeping:
            continue
        dx, dy = float(agent.x) - tx, float(agent.y) - ty
        key = (dx * dx + dy * dy, int(agent.id))
        if best_key is None or key < best_key:
            best_key, best = key, agent
    return best


def note_delivery_trip(sim, staffed: bool) -> None:
    """補充トリップ 1 本の帰属を数える(``goods`` からの観測専用の口。OFF は no-op)。"""
    if not enabled(sim):
        return
    st = _state(sim)
    st["trips_staffed" if staffed else "trips_unstaffed"] += 1


# =========================================================================== #
# ④ ビル夜間清掃
# =========================================================================== #
def _night_cleaning(sim, cfg: dict, st: dict, bud: _Budget, agent,
                    step: int, sim_min: int) -> None:
    """夜間清掃員 1 人の 1 step(担当ビルに居る夜だけ ``night_cleaning`` を 1 件)。"""
    ncfg = cfg["night_cleaning"]
    end = int(ncfg["end_min"]) if _midnight_shift_on(sim) else 1440
    if not _in_window(int(ncfg["start_min"]), end, sim_min):
        return
    bld = str(getattr(agent, "city_ops_clean_building", "") or "")
    if not bld or str(getattr(agent, "building", "")) != bld:
        return                                     # 担当ビルに**居る**ことが条件
    # 夜の識別子(22:00 以降は「その日の夜」、00:00〜05:00 は「前日の夜」)
    minute = int(sim_min) % 1440
    night_id = int(sim_min) // 1440 - (1 if minute < int(ncfg["start_min"]) else 0)
    seen = st["cleaned_night"].setdefault(night_id, {})
    if bld in seen:
        return                                     # ★1 棟 1 晩 1 件(L1 の throttle)
    if len(seen) >= int(ncfg["max_per_night"]):
        return                                     # ★1 晩あたりの上限(L1 の安全弁)
    seen[bld] = 1
    bud.log(Event(step=step, sim_min=sim_min, agent_id=int(agent.id),
                  kind="night_cleaning", x=agent.x, y=agent.y,
                  payload={"building": bld, "node": str(agent.node),
                           "start_min": int(ncfg["start_min"])}))
    st["cleanings"] += 1
    agent.remember(CLEAN_TEXT)


# =========================================================================== #
# ⑤ 救急 = **通報から始まる行為の連鎖**(collapse → ems_call → ems_dispatch)
# =========================================================================== #
def _health_mod():
    """健康 module の遅延 import(循環と重い import の回避。``_routine`` と同じ作法)。"""
    from . import health as _health
    return _health


def _severity_on(sim) -> bool:
    """重症度の状態機械(レーン H1)が有効か = **倒れる引き金が本物になっているか**。"""
    return bool(_health_mod().severity_on(sim))


def _collapse_gate(agent, ecfg: dict, day: int, steps_per_day: int,
                   step_of_day: int, step: int = -1, sev_on: bool = False) -> str:
    """その個体がこの step に倒れるか。

    ★**世代交代**(レーン H1・2026-08-10): 本 module の docstring が最初から申し送っていた
      とおり、``sev_on``(= health.severity.enabled)のときは下の暫定ハッシュゲートを捨て、
      「**重症度が S3/S4 へ遷移したこと**そのもの」を引き金にする。判定は健康側の純関数
      ``health.collapse_source`` の読み取りだけで、本 module は健康状態へ 1 バイトも書かない。
      severity OFF のランは従来どおり下のハッシュゲート(既存挙動と完全同一)。

    以下は **severity OFF のときの暫定ゲート**の説明(据え置き)。

    戻り値は引き金の名前(``"illness"`` / ``"fatigue"``)。倒れないなら空文字。
    ★乱数を 1 本も引かない: 「その日倒れるか」も「その日の何 step 目か」も
      (agent id, 経過日)の安定ハッシュから決まる(``goods.pick_item`` と同じ流儀)。
    ★**健康機構が世界に成立させた状態しか読まない**(``sick`` / ``fatigue``)ので、
      健康 OFF のランでは 1 件も起きない = 正直な依存関係。
    ★既知の小さな粗さ: 「その日」は ``sim_min // 1440``(暦の日)で数え、「その日の何 step 目か」は
      ``step % steps_per_day`` で数える。開始時刻が 00:00 でないラン(既定は 07:00)では
      **初日だけ**この 2 つがずれ、初日の後半に当たる slot が来ない = その slot の個体は
      初日に倒れない。2 日目以降は 1 日ぶんの step が slot を過不足なく 1 周するので、
      同じ個体が同じ日に 2 回倒れることも、抜けることも無い(``city_ops_down`` の印が
      治療中の再発火を止めるのとは別に、構造として重複しない)。
    """
    if sev_on:
        return _health_mod().collapse_source(agent, int(step))
    source = ""
    if bool(getattr(agent, "sick", False)):
        source = "illness"
    elif not ecfg["require_sick"]:
        if float(getattr(agent, "fatigue", 0.0)) >= float(ecfg["collapse_fatigue"]):
            source = "fatigue"
    if not source:
        return ""
    aid = int(agent.id)
    if _stable_hash(f"collapse/{aid}/{day}") % 10000 >= int(ecfg["collapse_per_10k"]):
        return ""
    slot = _stable_hash(f"collapse_slot/{aid}/{day}") % max(1, int(steps_per_day))
    return source if int(step_of_day) == int(slot) else ""


def _caller_for(sim, patient, radius_m: float):
    """通報者 = 患者の近くに居合わせた個体(**最小 id で先勝ち**・同距離は id 昇順)。

    条件: 街の中に居て・起きていて・自分も倒れていない他人。居なければ ``None``
    (呼び出し側が「本人が自分で呼ぶ」へ落とす)。乱数ゼロ・状態を 1 バイトも触らない。
    """
    r2 = float(radius_m) * float(radius_m)
    best = None
    best_key = None
    for other in sim.agents:
        if int(other.id) == int(patient.id):
            continue
        if other.loc == "outside" or other.sleeping:
            continue
        if getattr(other, "city_ops_down", False):
            continue
        dx = float(other.x) - float(patient.x)
        dy = float(other.y) - float(patient.y)
        d2 = dx * dx + dy * dy
        if d2 > r2:
            continue
        key = (d2, int(other.id))
        if best_key is None or key < best_key:
            best_key, best = key, other
    return best


def _bystanders(sim, patient, radius_m: float) -> list:
    """患者に居合わせた個体(近い順・同距離は id 昇順)。``_caller_for`` と同じ絞り込み。"""
    r2 = float(radius_m) * float(radius_m)
    found = []
    for other in sim.agents:
        if int(other.id) == int(patient.id):
            continue
        if other.loc == "outside" or other.sleeping:
            continue
        if getattr(other, "city_ops_down", False):
            continue
        dx = float(other.x) - float(patient.x)
        dy = float(other.y) - float(patient.y)
        d2 = dx * dx + dy * dy
        if d2 <= r2:
            found.append((d2, int(other.id), other))
    found.sort(key=lambda t: (t[0], t[1]))
    return [t[2] for t in found]


def _is_related(patient, other) -> bool:
    """関係者(同居人・恋人・同僚)か = 傍観者効果を弱める側の共変量。"""
    oid = int(other.id)
    if oid in {int(x) for x in (getattr(patient, "housemates", None) or ())}:
        return True
    if getattr(patient, "partner_id", None) is not None \
            and int(patient.partner_id) == oid:
        return True
    work = str(getattr(patient, "work_node", "") or "")
    return bool(work) and work == str(getattr(other, "work_node", "") or "")


def _call_delay_min(ecfg: dict, patient, day: int, step: int) -> float:
    """通報までの所要時間[分]を実測分布(46/29/25%)から決定論で引く。

    ★これは**観測量**である: 既定 Δt=10 分ではどのビンも同じ 1 刻みに丸まるので
      発火時刻は動かさない(より細かい Δt で意味を持つ。正直な既知の粗さ)。
    """
    u = (_stable_hash(f"ems_delay/{int(patient.id)}/{int(day)}/{int(step)}")
         % 10000) / 10000.0
    fast = float(ecfg["call_delay_fast_share"])
    if u < fast:
        return float(ecfg["call_delay_fast_min"])
    if u < fast + float(ecfg["call_delay_mid_share"]):
        return float(ecfg["call_delay_mid_min"])
    return float(ecfg["call_delay_slow_min"])


def _bystander_caller(sim, ecfg: dict, patient, day: int, step: int):
    """居合わせた集団から通報者を決める(**決定論・乱数ゼロ**の傍観者モデル)。

    戻り値 ``(caller, n_bystanders, group_prob)``。``caller`` が ``None`` なら
    「誰も動かず、本人も呼べなかった」= 通報そのものが起きない(目撃されない倒れ)。

    較正(計画書 §1):
      - 集団としての P(誰かが動く) = 1 − (1 − solo)^(n^(1−alpha)) を 0.85〜0.95 帯に置く。
      - alpha は**状況の曖昧さ**で切り替える: 見かけが重症以上なら 0(責任分散なし・
        人数が増えるほど誰かが動く)、中等症以下なら 0.4(Darley-Latané 型の減衰)。
      - 家族・同居人・同僚が居合わせれば個人の介入率に倍率がかかる。
      - 誰も動かなかったとき、**本人に意識があれば**(= 心停止でなければ)自分で呼ぶ。
        心停止で誰も居合わせなければ通報そのものが起きない(目撃されなかった倒れ)。
    """
    health = _health_mod()
    cands = _bystanders(sim, patient, float(ecfg["call_radius_m"]))
    apparent = int(health.apparent_severity(patient))
    danger = apparent >= health.S_SEVERE
    alpha = float(ecfg["bystander_alpha_danger"] if danger
                  else ecfg["bystander_alpha_ambiguous"])
    solo = float(ecfg["bystander_solo_prob"])
    if cands and any(_is_related(patient, o) for o in cands):
        solo = min(float(ecfg["bystander_max_prob"]),
                   solo * float(ecfg["bystander_related_mult"]))
    n = len(cands)
    if n <= 0:
        group = 0.0
    else:
        eff = float(n) ** (1.0 - alpha)
        group = min(float(ecfg["bystander_max_prob"]), 1.0 - (1.0 - solo) ** eff)
    acts = n > 0 and (_stable_hash(f"bystander/{int(patient.id)}/{int(day)}/{int(step)}")
                      % 10000) < int(round(group * 10000.0))
    if acts:
        # 動くのは**関係者が居ればその最近傍**、居なければ単純な最近傍(id 昇順で一意)
        for other in cands:
            if _is_related(patient, other):
                return other, n, group
        return cands[0], n, group
    if apparent < health.S_ARREST:                 # 意識がある = 本人が自分で呼ぶ
        return patient, n, group
    return None, n, group


def _on_duty_crew(sim, ecfg: dict, node: str, sim_min: int):
    """通報に応えられる当直の救急隊(現場に最も近い 1 人。同距離は id 昇順)。

    条件は ``transit_staff.on_duty_crew`` と同じ 3 つ(その職 / 街の中に居る /
    勤務窓の中)。既に別の現場へ出ている隊員は除く(二重出動しない)。
    """
    occs = frozenset(ecfg["crew_occupations"])
    from .cognition import routine as _routine     # 遅延 import(循環と重い import の回避)
    cal = getattr(sim, "calendarcfg", None)
    try:
        tx, ty = sim.city.node_xy(str(node))
    except Exception:                              # noqa: BLE001(未知ノードの保険)
        return None
    best = None
    best_key = None
    for agent in sim.agents:
        if str(getattr(agent, "occupation", "")) not in occs:
            continue
        if agent.loc == "outside" or agent.sleeping:
            continue
        if getattr(agent, "city_ops_ems_until", -1) >= 0:
            continue                               # 出動中(二重出動しない)
        if not _routine.in_work_window(agent, int(sim_min), cal):
            continue
        dx, dy = float(agent.x) - tx, float(agent.y) - ty
        key = (dx * dx + dy * dy, int(agent.id))
        if best_key is None or key < best_key:
            best_key, best = key, agent
    return best


def request_ems(sim, patient, caller=None, source: str = "", *,
                step: int = -1, sim_min: int = 0) -> dict:
    """通報に応える当直を割り当てる**公開シーム**(救急の実体を持つのは本 module だけ)。

    戻り値 ``{"crew", "response_min", "unstaffed", "node"}``。``crew`` が None のときは
    ``unstaffed=True``(救急が OFF / 当直が全員出動中 = **ひっ迫**)。

    ★**L1 を 1 件も出さない**: 出来事の記録は呼び出し側が自分の予算(``_Budget``)で行う。
      そうしないと呼び出し側ごとに違う payload 規約(source 欄など)を本 module が
      知らなければならなくなる。ここが引き受けるのは「誰が応えるか」の**選定と印**だけである。
    ★``crew`` に付ける印(``city_ops_ems_until`` / ``city_ops_ems_home``)は本 module のもので、
      持ち場へ戻す処理は ``_ems_restore`` が従来どおり行う(復帰経路を二重に作らない)。
    ★乱数を 1 本も引かない(選定は距離と id の純関数)。
    ★他レーン(身体・環境事件)がこの関数を通ってくるので、``_on_duty_crew`` の private を
      直接呼ばせない = 選定規則が 1 箇所に閉じる(綴り替えで黙って 0 件になるのを防ぐ)。
    """
    node = str(getattr(patient, "node", "") or "")
    out: dict = {"crew": None, "response_min": None, "unstaffed": True, "node": node}
    if not enabled(sim):
        return out
    ecfg = cfg_of(sim)["ems"]
    if not ecfg["enabled"]:
        return out
    crew = _on_duty_crew(sim, ecfg, node, sim_min)
    if crew is None:
        return out                                 # 当直不在 = ひっ迫(呼び出し側が記録する)
    dist = ((float(crew.x) - float(patient.x)) ** 2
            + (float(crew.y) - float(patient.y)) ** 2) ** 0.5
    crew.city_ops_ems_home = str(getattr(crew, "work_node", "") or "")
    crew.city_ops_ems_until = int(step) + max(1, int(ecfg["on_scene_steps"]))
    crew.work_node = node                          # 持ち場が現場へ移る(移動は routine)
    out["crew"] = crew
    out["response_min"] = round(dist / max(1.0, float(ecfg["speed_m_per_min"])), 1)
    out["unstaffed"] = False
    return out


def _medical_mod():
    """医療の受け皿(H2)の遅延 import。**あちらが OFF なら全関数が即 return する**。"""
    from . import medical as _medical
    return _medical


def _ems_chain(sim, cfg: dict, st: dict, bud: _Budget, step: int,
               sim_min: int) -> None:
    """救急の 3 段(倒れる → 通報 → 出動)。**すべて既存状態の決定論**・乱数ゼロ。

    ★レーン H1 が ON のときは ① の引き金が「重症度 S3/S4 への遷移」に、② の通報者選定が
      **較正済みの傍観者モデル**(集団 P≒0.85〜0.95・危険事態は責任分散なし)に替わる。
      どちらも読むだけ・乱数ゼロという本 module の性質は変わらない。
    """
    ecfg = cfg["ems"]
    day = int(sim_min) // 1440
    steps_per_day = max(1, int(sim.clock.steps_per_day))
    step_of_day = int(step) % steps_per_day
    done = int(st["calls_by_day"].get(day, 0))
    cap = int(ecfg["max_calls_per_day"])
    if done >= cap:
        return
    sev_on = _severity_on(sim)
    radius = float(ecfg["call_radius_m"])
    treat = max(1, int(ecfg["treatment_steps"]))
    # ★走査は 1 パスで**候補を絞ってから**並べる(25 万体で毎 step 全体を sort しない)。
    #   絞り込みは属性の読み取りだけ = 世界を 1 バイトも触らない。
    candidates = []
    for agent in sim.agents:
        if agent.loc == "outside" or agent.sleeping:
            continue
        if getattr(agent, "city_ops_down", False):
            continue
        source = _collapse_gate(agent, ecfg, day, steps_per_day, step_of_day,
                                step, sev_on)
        if source:
            candidates.append((int(agent.id), source, agent))
    for _aid, source, patient in sorted(candidates, key=lambda t: t[0]):
        if done >= cap:
            break
        # ---- ① 倒れる(身体の出来事)。既存の睡眠機構でその場に留まる ---------- #
        patient.city_ops_down = True
        patient.city_ops_down_until = int(step) + treat
        patient.route = []
        patient.sleeping = True
        patient.sleep_until = int(step) + treat
        patient.stay_until = int(step)
        patient.activity = ""
        patient._pending_activity = ""
        patient._pending_stay = 0
        patient._ride_pending = None
        payload = {"node": str(patient.node), "source": source,
                   "until_step": int(step) + treat}
        if sev_on:                                 # 見かけの重症度(前兆状態の同梱)
            payload["sev"] = int(_health_mod().apparent_severity(patient))
        bud.log(Event(step=step, sim_min=sim_min, agent_id=int(patient.id),
                      kind="collapse", x=patient.x, y=patient.y, payload=payload))
        st["collapses"] += 1
        patient.remember(COLLAPSE_TEXT)
        # ---- ② 通報(**この行為が出動の原因**)------------------------------- #
        call_extra: dict = {}
        if sev_on:
            caller, n_by, group_p = _bystander_caller(sim, ecfg, patient, day, step)
            if caller is None:
                # ★誰も動かず本人も呼べない = **目撃されなかった倒れ**。出動は起きない。
                #   黙って通り過ぎず、無通報として数える(正直な稼働率の観測)。
                st["no_call"] = int(st.get("no_call", 0)) + 1
                continue
            self_call = int(caller.id) == int(patient.id)
            call_extra = {"bystanders": int(n_by),
                          "group_prob": round(float(group_p), 3),
                          "delay_min": _call_delay_min(ecfg, patient, day, step)}
        else:
            caller = _caller_for(sim, patient, radius)
            self_call = caller is None
            if self_call:
                caller = patient
        if self_call:
            st["self_calls"] += 1
        dist_m = 0.0 if self_call else round(
            ((float(caller.x) - float(patient.x)) ** 2
             + (float(caller.y) - float(patient.y)) ** 2) ** 0.5, 1)
        call_payload = {"patient": int(patient.id), "node": str(patient.node),
                        "self_call": bool(self_call), "dist_m": dist_m}
        call_payload.update(call_extra)
        bud.log(Event(step=step, sim_min=sim_min, agent_id=int(caller.id),
                      kind="ems_call", x=caller.x, y=caller.y,
                      payload=call_payload))
        st["calls"] += 1
        done += 1
        st["calls_by_day"][day] = done
        caller.remember(SELF_CALL_TEXT if self_call else CALL_TEXT)
        # ---- ③ 出動(通報に応える行為)-------------------------------------- #
        #  ★選定と印は**公開シーム** request_ems に閉じる(他レーンと同じ 1 本の口を通る)。
        answer = request_ems(sim, patient, caller, "collapse", step=step,
                             sim_min=sim_min)
        crew = answer["crew"]
        if crew is None:
            # ★当直が居ない = **誰も応えなかった**。救急を運行する装置は世界に存在しない
            #   ので装置 id は作らず、無人であることだけを正直に記録する
            #   (``transit_staff`` の unstaffed と同じ作法)。
            bud.log(Event(step=step, sim_min=sim_min, agent_id=-1,
                          kind="ems_dispatch", x=patient.x, y=patient.y,
                          payload={"node": str(patient.node),
                                   "patient": int(patient.id),
                                   "caller": int(caller.id), "crew": -1,
                                   "response_min": None, "unstaffed": True}))
            st["dispatch_unstaffed"] += 1
            continue
        disp_payload = {"node": str(patient.node), "patient": int(patient.id),
                        "caller": int(caller.id), "crew": int(crew.id),
                        "response_min": answer["response_min"], "unstaffed": False}
        if sev_on:
            # ★**重症度は搬送先で確定する**(東京実測: 搬送の 52.8% が軽症)。通報時点の
            #   見かけ(apparent)と確定(confirmed)を分けて両方載せる = 「通報は不確実性から
            #   生まれる」の観測可能な再現。確定を刻むのは健康側(本 module は書かない)。
            health = _health_mod()
            disp_payload["apparent"] = int(health.apparent_severity(patient))
            disp_payload["confirmed"] = int(health.note_confirmed(sim, patient, step))
        bud.log(Event(step=step, sim_min=sim_min, agent_id=int(crew.id),
                      kind="ems_dispatch", x=crew.x, y=crew.y,
                      payload=disp_payload))
        st["dispatches"] += 1
        crew.remember(DISPATCH_TEXT)
        # ---- ④ 搬送(医療の受け皿 H2。**あちらが OFF なら 1 バイトも動かない**)------ #
        #  ★本 module は搬送先も入院も持たない(限界 1 の宣言どおり)。患者を病院へ運ぶのは
        #    医療側の意味論なので、ここは「出動が決まった」ことを伝えるだけである
        #    (健康状態を 1 バイトも書かないのと同じ線引き = 所有の分離)。
        _medical_mod().on_ems_dispatch(sim, patient, crew, step, sim_min)


def _ems_restore(sim, st: dict, step: int) -> None:
    """治療期間・現着期間の終了で状態を元へ戻す(**round trip の後半**)。

    - 患者: 既存の睡眠機構が ``sleep_until`` で起こす(``wake_up``)。ここでは
      「倒れている」印だけを落とす = **健康状態は 1 バイトも書き換えない**
      (``sick`` は ``health`` 側の日程どおりに治る)。
    - 救急隊: 持ち場を待機拠点へ戻す。現場に着けていたかを数える(正直な稼働率)。
    """
    for agent in sim.agents:
        if getattr(agent, "city_ops_down", False) \
                and int(step) >= int(getattr(agent, "city_ops_down_until", 0)):
            agent.city_ops_down = False
        until = int(getattr(agent, "city_ops_ems_until", -1))
        if until >= 0 and int(step) >= until:
            home = str(getattr(agent, "city_ops_ems_home", "") or "")
            if str(agent.node) == str(getattr(agent, "work_node", "")):
                st["ems_arrived"] += 1
            if home:
                agent.work_node = home
            agent.city_ops_ems_until = -1


# =========================================================================== #
# 単一作用点(scheduler の唯一のフック。位置確定後に呼ぶ)
# =========================================================================== #
def phase(sim, step: int, sim_min: int) -> None:
    """毎 step: 交番 / ごみ収集 / 納品 / 夜間清掃 / 救急。**既定 OFF は即 return**。

    ★世界に対して触るのは (a) L1、(b) 記憶の定型 1 行、(c) 持ち場(``work_node`` と
      勤務窓)、(d) 倒れた個体の在席(既存の睡眠機構)だけ。所持金・関係・drive・
      opinion・k のどれにも触らない。
    ★乱数を 1 本も引かない(全部 id・step・sim_min・地図・暦の純関数)。
    """
    if not enabled(sim):
        return
    cfg = cfg_of(sim)
    st = _state(sim)
    _ensure_bound(sim, step, sim_min)
    bud = _Budget(sim, st, int(cfg["max_events_per_step"]))
    _ems_restore(sim, st, step)

    waste_on = bool(cfg["waste"]["enabled"])
    clean_on = bool(cfg["night_cleaning"]["enabled"])
    if waste_on or clean_on:
        waste_occs = frozenset(cfg["waste"]["occupations"])
        clean_occs = frozenset(cfg["night_cleaning"]["occupations"])
        # ★1 パスで候補を絞ってから id 昇順に並べる(25 万体で毎 step 全体を sort しない)。
        duty = []
        for agent in sim.agents:
            if agent.loc == "outside" or agent.sleeping:
                continue
            occ = str(getattr(agent, "occupation", ""))
            if waste_on and occ in waste_occs:
                duty.append((int(agent.id), "waste", agent))
            elif clean_on and (occ in clean_occs
                               or str(getattr(agent, "role", "")) in clean_occs):
                duty.append((int(agent.id), "clean", agent))
        for _aid, which, agent in sorted(duty, key=lambda t: t[0]):
            if which == "waste":
                _waste_round(sim, cfg, st, bud, agent, step, sim_min)
            else:
                _night_cleaning(sim, cfg, st, bud, agent, step, sim_min)

    if cfg["ems"]["enabled"]:
        _ems_chain(sim, cfg, st, bud, step, sim_min)


# --------------------------------------------------------------------------- #
# 観測タリー(**OFF では None = 何も出さない**)
#
# ★第109バッチ D2 で **``summary.json`` の ``city_ops`` キーへ配線済み**
#   (``engine/simulation.py`` の finalize 末尾。Wave 4 の観測タリー群と一括で入れた)。
#   OFF では本関数が None を返し、summary はキー自体を持たない = 既存ラン無風。
#   ``run_manifest.json``(ラン開始時に書かれる)へは今も載せていない: 本タリーは
#   走り切ってからでないと値が確定しないため(``ablate`` の provenance と同じ線引き)。
# --------------------------------------------------------------------------- #
def provenance(sim) -> dict | None:
    """都市運営の観測タリー(既定 OFF は None = 何も返さない)。

    ★救急の**参照値との突き合わせ**をここで正直に出す: 実測の通報件数/日と、
      区の実測(``ward_calls_per_day`` = 65 件/日)を人口比で割り戻した参照値を並べる。
      規模比が判らない(人口が 0)ランでは参照値を出さない(捏造しない)。
    """
    if not enabled(sim):
        return None
    cfg = cfg_of(sim)
    out: dict = {"schema": SCHEMA, "roles": list(CITY_OPS_OCCS),
                 "police_posts": len(police_posts(sim)),
                 "districts": len(districts(sim))}
    st = getattr(sim, "_co_state", None)
    if st is None:
        out.update({"rounds": 0, "cleanings": 0, "collapses": 0, "calls": 0,
                    "dispatches": 0, "by_kind": {}, "dropped": 0})
        return out
    days = max(1, len(st["calls_by_day"]))
    out.update({"bind": dict(st.get("bind") or {}),
                "rounds": int(st["rounds"]),
                # ★深夜の事業系が 0 件でも黙らない: 深夜に起きている作業員が名簿に
                #   居ないランではここが 0 のまま残る(= 正直な稼働率の観測)。
                "rounds_by_kind": {k: int(v)
                                   for k, v in sorted(st["rounds_by_kind"].items())},
                "cleanings": int(st["cleanings"]),
                "collapses": int(st["collapses"]), "calls": int(st["calls"]),
                "self_calls": int(st["self_calls"]),
                # ★傍観者モデル ON のとき「誰も動かず本人も呼べなかった」件数
                #   (目撃されなかった倒れ)。severity OFF では常に 0。
                "no_call": int(st.get("no_call", 0)),
                "dispatches": int(st["dispatches"]),
                "dispatch_unstaffed": int(st["dispatch_unstaffed"]),
                "ems_arrived": int(st["ems_arrived"]),
                "trips_staffed": int(st["trips_staffed"]),
                "trips_unstaffed": int(st["trips_unstaffed"]),
                "calls_per_day": round(float(st["calls"]) / days, 2),
                "dropped": int(st["dropped"]),
                "by_kind": {k: int(v) for k, v in sorted(st["by_kind"].items())}})
    n_agents = len(getattr(sim, "agents", ()) or ())
    if n_agents > 0:
        # 区の実測 65 件/日 は区の全人口(約 23 万人)に対する数字。本ランの規模へ
        # 単純に線形で割り戻した**物差し**であって、目標値でも検定でもない。
        out["ems_reference_per_day"] = round(
            float(cfg["ems"]["ward_calls_per_day"]) * n_agents / 230000.0, 3)
    return out
