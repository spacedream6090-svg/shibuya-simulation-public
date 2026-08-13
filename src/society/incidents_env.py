"""事件レイヤー H5 = **環境側の 3 族**(火災 / 交通 / 群集。``incidents_env``・**既定 OFF**)。

正典
----
- ``docs/plans/body-incident-layer-plan.md`` §3(6 族オントロジーと実装様式)・§6(ユーザー決定)
  - 火災 = 「薄いレート + 完全アクター連鎖」。**0.66 件/日**・**重度分布 ぼや175:全焼0 を
    頻度と同時に較正**(§7「事件インフレ = 災害映画化」の防止)
  - 交通 = 「曝露の積」。P(事故) ∝ 歩行者流 × 車流(横断部)・**被害者は実在の横断中エージェント**
  - 群集 = 「**生成しない = 状態が事件**」。SFM ゾーンの密度が 4/6/13 人/m² を跨ぐこと自体を
    incident 化(near-miss 含む)・**人工的な群集注入は禁忌**・雑踏警備は緩和アクター(読み取りのみ)
- ``docs/plans/body-incident-layer-plan.md`` §0(三層の因果: 身体 physics / 行為 agent /
  応答 agent・device)

何を解く問題か
--------------
街には「誰の思考とも関係なく起きるが、**起きたあとは必ず誰かの行為で処理される**」出来事が
ある。火が出れば誰かが見つけて通報し、消防が出場して消す。横断中に車と接触すれば居合わせた
誰かが救急を呼ぶ。人が集まりすぎれば、それは誰かが「集めた」のではなく**状態そのものが事件**
である。本 module はこの 3 族を、それぞれ**違う様式**で載せる(1 つの確率抽選器に畳まない)。

    火災 … 薄いレート(建物ハザード)→ **第一発見者 → 119 → 出場 → 鎮火** の完全アクター連鎖
    交通 … **曝露の積**(その step その node に歩行者と車が同時に居た量)だけが引き金
    群集 … **抽選をしない**。既に物理層が測っている密度が閾値を跨いだことを記録するだけ

★**停電・漏水は実装しない**(skip の明示)
------------------------------------------
東京の SAIDI(1 需要家あたり年間停電時間)は **約 13 分/年**である。10 日ランに引き直すと
期待値は 0.4 秒で、「起きない」が正しい答えになる。断水も同様(計画断水を除けば年単位の
希少事象)。**起きないものを起こす機構を書くと、それは較正ではなく演出になる**ので作らない。
供給事業者の装置 id(``devices.DEV_OPERATOR_INFRA``)は既に名簿にあるので、必要になった
時点でそこへ δ_ext を足せば成立する(前方互換のためのメモ = 実装はしない)。

較正(**すべて公表値・実測ベース。管轄と区の別を明記する**)
------------------------------------------------------------
計画 §7 の「**管轄混同**(渋谷署 ≠ 渋谷区・3 署/消防も同様 = 3 倍ズレの源)」がここでの
最大の罠なので、換算を 2 段に割って **conf のキーとして外に出す**:

    ① アンカー(管轄スケールの実測)      … ``jurisdiction_per_day``
    ② 管轄 → 区スケールの換算係数        … ``jurisdiction_to_ward``(**既定 1.0**)
    ③ 区 → この世界(地図の範囲)の規模比 … ``area_share``(既定 0 = 地図 bbox から自動計算)

★②の既定を 1.0 にした理由(**少なく見積もる方向へ倒す**):
  火災 0.66 件/日 は年 241 件に当たる。渋谷区は 2 つの消防署が分担するので素朴には
  区 = 0.66 × 2 = 1.32 件/日 になるが、これは東京消防庁管内の総火災件数(年約 4,000 件)を
  23 区へ按分した水準(区あたり年 60〜200 件)と 2〜8 倍食い違う。つまり
  「アンカー × 署数」は**上振れの上限側**であって実測ではない。同じことが交通(渋谷署管内
  328 件/年 = 0.9 件/日・渋谷区は 3 署)にも言える。どちらを採っても嘘にならない書き方は
  「**アンカーをそのまま区スケールとして扱い、上限側は conf で試せるようにする**」なので、
  既定は 1.0(= 過大評価を避ける向き。observer/causality.py の原則 3 と同じ倒し方)。
  上限側を測りたいランは ``jurisdiction_to_ward: 2.0``(火災)/ ``3.0``(交通)を書く。

★③は**地図のメタデータから計算する**(発明した定数を置かない): ``city.meta["bbox"]`` の
  緯度経度矩形の面積 ÷ 渋谷区の面積 15.11 km²。現行地図 v7 では約 3.8 km² / 15.11 km² ≈ 0.25。
  正直な限界: 駅周辺は区の平均より建物・活動密度が高いので面積按分は**過少**に出る。逆に
  この地図の建物 7,210 棟の 4 分の 3 は ``house?``(小規模住家)で火災荷重は低い。
  2 つの偏りは逆向きで、どちらが勝つかは判らない(片方だけを補正して精度を装わない)。

重度分布(**頻度と同時に較正する**= 計画 §7 の要求)
----------------------------------------------------
火災: **ぼや 175 : 全焼 0**(同じ年報)。残余(241 − 175 = 66)を部分焼 : 半焼 = 3 : 1 に
割った。★アンカーとして固定なのは「**ぼやが 7 割強**」と「**全焼はゼロ**」の 2 点だけで、
中間の割り方は仮定であることをここに明記する(``severity_weights`` で差し替えられる)。
全焼の重みが 0 = **既定では 1 件も起きない**。これが「災害映画化の防止」の実体である。

交通: 人身事故のうち **軽傷 92 : 重傷 7 : 死亡 0**。死は **H1(身体レイヤー)の管轄**なので
本 module では重みを 0 にしてある(計画 §6-1 = mortality は H1 トグル配下・尊厳規約)。
さらに人身事故のうち**人対車両**は約 22%(``pedestrian_share``)= 残りは車対車で、
本シムのエージェントは巻き込まれない(車は背景交通 = エージェントではない)。

R1 ドクトリン
-------------
- 既定 ``incidents_env.enabled: false`` では ``phase`` が即 return し、**L1 に 1 件も出ず・
  sim に state が生えず・agent に属性が生えず・乱数 stream を 1 本も引かず・プロンプトが
  1 バイトも変わらない**(= ゴールデン L1 バイト一致。第96 traces / Wave 4 III-4 と同じ
  「OFF ではキー自体を作らない」流儀)。
- **generate() の呼び出しサイトを 1 つも作らない**(LLM 追加呼ゼロ = k 非依存)。通報も
  避難も**決定論の純関数**で、出来事は既存の記憶機構(``agent.remember``)へ定型 1 行が
  入るだけ = プロンプトの**欄**は 1 つも増えない(``street_life`` / ``city_ops`` と同じ線引き)。
- 新しい乱数は**新しい named stream だけ**(``incident_fire`` / ``incident_traffic``)。
  群集は**乱数を 1 本も引かない**(状態の閾値跨ぎ = 完全決定論)。
- L1 は **1 行 + 前兆状態を payload 同梱**(密度・器具種・曝露の積)= 「内生である」ことを
  解析側で機械検証できる(計画 §3 末尾の要求)。

EMS 連鎖への接続(**``city_ops.py`` を 1 バイトも編集していない**)
-------------------------------------------------------------------
負傷者が出たときは、既存の救急連鎖に**既存の kind と既存の読み取り専用ヘルパだけ**で乗る:

  1. ``injury``(本 module が登録する新種・cause_type=physics)= 身体の事実を 1 行残す
  2. ``ems_call``(**city_ops が登録済みの既存種**)= 居合わせた誰かの**通報という行為**
  3. ``ems_dispatch``(同上)= 当直の応答。隊員の選定は ``city_ops`` の**公開シーム**
     ``request_ems(sim, patient, caller, source)`` を**呼ぶだけ**で、選定規則を二重定義しない。
     出動中の印(``city_ops_ems_until`` / ``city_ops_ems_home``)を付けるのもシームの側なので、
     持ち場へ戻す処理は ``city_ops._ems_restore`` が従来どおり行う(復帰経路を新設しない)。

★したがって **救急の実体は city_ops.ems が ON のときだけ動く**(正直な依存関係)。OFF の
  ランでは ``ems_dispatch`` を ``unstaffed=true`` で 1 行だけ残す(``transit_staff`` の無人
  マーカーと同じ作法 = 黙って落とさない)。
★申し送りは**果たされた**(H2 レーン 2026-08-10): 初版は private ヘルパ ``_on_duty_crew`` を
  直接呼んでいたが、``city_ops`` 側に公開シーム ``request_ems`` が生えたのでそちらへ移した
  (挙動は同値 = 同じ選定・同じ印・同じ応答時間)。**シームが消えたら黙って 0 件になる**のを
  防ぐため、tests/test_incidents_env.py が「``city_ops.request_ems`` が実在すること」を機械固定する。
★負傷は身体の状態機械へも載る(H1 の公開 API ``health.on_injury``)。渡すのは重度 1 語の
  **見立て**(``INJURY_HINT``)だけで、重症度も療養日数も決めるのは健康側である。

正直な限界(6 件)
------------------
1. **火は建物から広がらない**(延焼を作らない)。1 件の出火は 1 棟に閉じる。ぼやが 7 割強で
   全焼 0 という重度分布の下では延焼は起きないのが実態で、延焼を書くと重度分布の較正と
   二重になる(同じ現象を 2 つの機構で数えることになる)。
2. **消防車・救急車の実体を作っていない**(``city_ops`` の限界 6 と同じ線引き)。出場は
   「持ち場が現場へ移る」形で、``response_min`` は距離から計算した**モデル値**であって
   実走行時間ではない。実測アンカー(現着 9 分)は payload の ``reference_min`` に併記する。
3. **交通事故の相手は個体化されていない**。背景交通(``world.traffic``)は「エージェントでは
   ない通過車両」なので、加害車両の運転者は世界に存在しない。したがって事故の cause_type は
   ``device``(背景交通という発生器の出来事)で、``device_id=traffic:<mode>`` を刻む。
   **世界に居ない運転者を捏造しない**ための線引きである。
4. **曝露の積は「街全体の車流 × 横断中の歩行者数」**であって、node 別の積ではない。
   理由は ``vehicle_flow`` の docstring に実測つきで書いた(ambient の車は 1 step で
   経路を走り切るので、**step 末に「その node に居る車」という量が世界に残っていない**)。
   したがって事故が起きる**場所**は歩行者側だけが決めており、車の側の場所情報は無い。
   さらに ``hazard_per_exposure`` は**本シムの標本規模を吸収する係数**である(60 体の
   ランは実際の街頭人口の 1 万分の 1 なので、曝露の積も同じだけ小さい)。**人数・地図・
   車の設定を変えると件数が動く**ので、provenance が「実測件数/日 vs アンカー」を必ず
   並べて出す(``city_ops.ems_reference_per_day`` と同じ流儀)。
   ★既定値 ``3.0e-7`` の出どころ(**実測**): 本 repo の参照ラン(既定地図
   ``data/shibuya_osm.json``・60 体・144 step = 1 日・``world.traffic`` 既定)で
   曝露の積の総和を測ると **144,916 /日**(曝露が立った step は 118/144)だった。
   そこへアンカー **0.9 × 0.22 × area_share(0.0476)= 0.0094 件/日**を割り戻すと
   ``0.0094 / (144916 × 0.22) = 2.95e-7`` になる。丸めて 3.0e-7 を既定にした。
   ★同じ係数を本番想定の広域地図(``shibuya_osm_wide_v7``・60 体)で測ると、曝露は
   1 step あたり 2,102(小地図の 1.7 倍)に増える一方でアンカーは area_share が
   0.0476 → 0.2519 と 5.3 倍になるため、**実測の期待値がアンカーの約 1/3 に落ちる**。
   **人数・地図が変われば合わなくなることを承知の上で置いた暫定値**であり、
   本番規模(wide 地図・数千体)での再較正は未了(残課題)。
5. **群集は物理ゾーンが ON のランでしか起きない**(``physics.zones_enabled``)。密度を測って
   いるのは物理層だけで、それが無いランで密度を推定すると**測っていない量を捏造する**ことに
   なる。conf/zones_shibuya.yaml を重ねたランだけが群集の観測対象である。
6. **群集の緩和アクター(雑踏警備)は読み取りだけ**。近傍の警備員・警察官の頭数を payload に
   残すが、密度を下げる作用は書いていない(誘導の物理は P4 の担当で、ここで近似すると
   物理層の較正と食い違う)。

**絶対に触らない範囲(明文規約)**
--------------------------------
1. ``observer/metrics_spec.py`` の**凍結 14 ファイル**は 1 バイトも変更しない。
2. ``src/society/city_ops.py`` / ``health.py`` / ``chance.py`` / ``agents/agent.py`` は
   1 バイトも変更しない(本 module は読むだけ)。
3. ``observer/schema.py`` は触らない(新 kind は**材料側 registration** = 本 module が
   自分で ``register_event_kind`` する。``city_ops.py:167`` の前例と同型)。
"""
from __future__ import annotations

import hashlib
import math

from . import devices as _devices
from .observer.schema import Event, register_event_kind

SCHEMA = 1

# --------------------------------------------------------------------------- #
# L1 イベント種の**材料側**登録(traces.py / rumors.py / city_ops.py と同じ流儀。
# observer/schema.py には 1 バイトも書かない)。
# ★payload に自由文は 1 つも入らない(建物 id / ノード / ゾーン id は世界の識別子、
#   残りは有限語彙の分類名・件数・密度・時刻)。
# --------------------------------------------------------------------------- #
register_event_kind(
    "fire_start",
    "出火(建物ハザードの薄いレート)。agent_id=-1(誰の行為でもない)"
    "{building, node, use, levels, appliance, attributed, severity, temp_hi,"
    " burn_steps}")
register_event_kind(
    "fire_report",
    "119 番通報(**この行為が出場の原因**)。agent_id = 第一発見者"
    "{fire_id, building, node, inside, dist_m}")
register_event_kind(
    "fire_dispatch",
    "通報に応えた消防の出場。agent_id = 隊員(当直不在なら -1 かつ unstaffed=true)"
    "{fire_id, node, crew, response_min, reference_min, unstaffed}")
register_event_kind(
    "fire_out",
    "鎮火。agent_id = 隊員(無人で燃え尽きた回は -1){fire_id, node, severity,"
    " burn_steps, suppressed, injured}")
register_event_kind(
    "traffic_accident",
    "横断部の人対車両の接触(曝露の積が引き金)。agent_id = 被害者"
    "{victim, node, ped_n, veh_n, exposure, severity, injured, signalized}")
register_event_kind(
    "crowd_density_incident",
    "群集密度が閾値を跨いだこと**そのもの**(生成しない = 状態が事件)。agent_id=-1"
    "{zone, level, threshold, density, occupancy, guards, near_miss}")
register_event_kind(
    "injury",
    "事件による負傷(身体の事実。救急連鎖の入口)。agent_id = 負傷者"
    "{victim, source, node, severity}")

# --------------------------------------------------------------------------- #
# 記憶の 1 行(**役割・出来事の種類だけの純関数**。数字・金額・config・実験条件・
# 機構語・ゾーン名は 1 つも入らない)= traces.sentence / city_ops.COLLAPSE_TEXT と
# 同じ no-fingerprint の定型文規約。
# --------------------------------------------------------------------------- #
FIRE_REPORT_TEXT = "煙と火が上がっているのを見つけて、消防に通報した。"
FIRE_DISPATCH_TEXT = "通報を受けて火災の現場へ向かった。"
FIRE_OUT_TEXT = "火を消し止めた。"
ACCIDENT_TEXT = "横断中に車と接触した。"
ACCIDENT_CALL_TEXT = "目の前で人が車とぶつかったので、救急に通報した。"
SELF_CALL_TEXT = "自分で救急に通報した。"

# --------------------------------------------------------------------------- #
# 有限語彙(payload に出る分類名。**この表に無い語は 1 つも出ない**)
# --------------------------------------------------------------------------- #
#: 建物の用途(火災荷重の共変量)。地図の ``kind`` と POI カテゴリから決まる純関数。
USES: tuple[str, ...] = ("eating", "retail", "office", "lodging", "public",
                         "residential", "other")

#: 出火の器具帰属(**73% は器具使用に帰属可能**という実測の再現)。
#: ``none`` = 器具に帰属できない残り(たばこ・放火・不明の族)。
APPLIANCES: tuple[str, ...] = ("kitchen", "heating", "electrical", "none")

#: 火災の重度(**ぼや 175 : 全焼 0** のアンカーを持つ 4 段)。
FIRE_SEVERITIES: tuple[str, ...] = ("boya", "partial", "half", "total")

#: 人身事故の重度(死亡は H1 の管轄なので既定重み 0)。
CRASH_SEVERITIES: tuple[str, ...] = ("light", "serious", "fatal")

#: 事件の重度 → **身体側へ渡す見立て**(1 = 軽症 / 2 = 中等症)。H1 の ``on_injury`` は
#: この 1 語だけを受け取り、重症度・療養日数・sick-role をあちらで決める(身体の意味論を
#: 2 箇所に書かない)。★死亡・重体は H1 の管轄なのでここから作らない(見立ては 2 で頭打ち)。
INJURY_HINT: dict[str, int] = {
    "boya": 1, "partial": 1, "half": 2, "total": 2,     # 火災の重度
    "light": 1, "serious": 2, "fatal": 2,               # 人身事故の重度
}

#: 群集密度の 3 閾値[人/m²]。4 = 歩行が困難になる水準(near-miss)/
#: 6 = 群集事故が起こりうる水準 / 13 = 群集雪崩の水準。
CROWD_LEVELS: tuple[float, ...] = (4.0, 6.0, 13.0)

#: 雑踏警備の担い手の**既定値**(**読み取りだけ**。密度を下げる作用は書かない)。
#: ★第109バッチ D2: conf キー ``incidents_env.crowd.guard_occupations`` の既定として使う
#:   (src のハードコードをやめた)。値を変えていないので既定ランは完全同値。
GUARD_OCCS: tuple[str, ...] = ("警備員", "警察官")

#: 消防の担い手の**既定値**(既存名簿の職。新設しない)。
#: ★第109バッチ D2: conf キー ``incidents_env.fire.occupations`` の既定として使う。
#: ★正直な現状(第108 縦煙の実測): ペルソナプール(``data/persona_pool``)の名簿に
#:   「消防士」は **1 人も居ない**(``scripts/build_persona_pool.py`` の ``_L5_ROLES`` に
#:   行が無い)。したがって既定のままでは出場者が見つからず ``fire_unstaffed`` が立つ。
#:   これは**不具合ではなく正直なマーカー**である(居ない職を別の職で埋めない)。
#:   救急の担い手だけは ``city_ops.ems.crew_occupations`` が「救急隊員 + 消防士」の
#:   2 語を引くので、救急隊員(L5 に 6 人)側で連鎖が成立する。
FIRE_OCCS: tuple[str, ...] = ("消防士",)

#: 渋谷区の面積[km²](公表値)。規模比 ③ の分母。
WARD_AREA_KM2 = 15.11

#: 緯度 1 度 ≒ 111.32 km(bbox から面積を出すための定数)。
KM_PER_DEG = 111.32

DEFAULTS: dict = {
    "enabled": False,
    "max_events_per_step": 24,     # 1 step に出す本 module の L1 件数の上限(安全弁)
    # 区 → この世界の規模比。0 = 地図 bbox から自動計算(発明した定数を置かない)
    "area_share": 0.0,
    # ---- ① 火災 ----
    "fire": {
        "enabled": True,
        # 出場する当直を名簿から探すときの職の語(**conf で差し替え可**)。
        # 既定 = FIRE_OCCS。★名簿に居ない語しか無いランでは fire_unstaffed が立つ
        #(居ない職を別の職で埋めない = 正直な無人マーカー)。
        "occupations": FIRE_OCCS,
        "jurisdiction_per_day": 0.66,   # ★アンカー(消防署管轄スケールの実測)
        "jurisdiction_to_ward": 1.0,    # ★管轄 → 区。既定 1.0 = 少なく見積もる向き
        # 重度分布のアンカー: ぼや 175 : 全焼 0(残余 66 を部分焼:半焼 = 3:1 と仮定)
        "severity_weights": {"boya": 175.0, "partial": 50.0, "half": 16.0,
                             "total": 0.0},
        # 器具帰属(合計 73% が器具・27% が非器具)。conf で差し替え可
        "appliance_weights": {"kitchen": 0.34, "heating": 0.20,
                              "electrical": 0.19, "none": 0.27},
        # 用途別の火災荷重の相対重み(飲食は火気を扱うので高い)
        "use_weights": {"eating": 4.0, "retail": 2.0, "office": 1.5,
                        "lodging": 2.0, "public": 1.0, "residential": 1.0,
                        "other": 0.5},
        "level_gain": 0.05,             # 階数 1 につき荷重 +5%(規模の代理)
        "cold_temp_c": 20.0,            # この気温を下回るほど暖房起因が増える
        "cold_gain": 0.5,               # 最も寒い日の出火レート倍率の上乗せ
        "discover_radius_m": 40.0,      # 第一発見者を探す半径(世界の知覚半径と同水準)
        "response_reference_min": 9.0,  # ★実測アンカー(通報から現着まで 9 分)
        "speed_m_per_min": 300.0,       # response_min の換算(モデル値。実走行ではない)
        "on_scene_steps": 3,            # 隊が現場に留まる長さ[step]
        "burn_steps": {"boya": 1, "partial": 2, "half": 3, "total": 4},
        "injury_severities": ["half", "total"],   # 負傷者が出る重度
        "max_per_day": 8,               # 1 日に出す出火の上限(L1 の安全弁)
    },
    # ---- ② 交通(曝露の積)----
    "traffic": {
        "enabled": True,
        "jurisdiction_per_day": 0.9,    # ★アンカー(警察署管内 328 件/年)
        "jurisdiction_to_ward": 1.0,    # ★管轄 → 区。既定 1.0(渋谷区は 3 署)
        "pedestrian_share": 0.22,       # 人身事故のうち**人対車両**の割合
        # 曝露 1 単位(横断中の歩行者 1 人 × その step に走った車 1 台)あたりのハザード。
        # ★既定値の根拠はモジュール docstring の限界 4 を参照(参照ランからの逆算)。
        "hazard_per_exposure": 3.0e-7,
        "severity_weights": {"light": 92.0, "serious": 7.0, "fatal": 0.0},
        "injury_severities": ["light", "serious", "fatal"],
        "call_radius_m": 40.0,          # 通報者を探す半径
        "signalized_only": False,       # true = 信号のある交差点だけを横断部とみなす
        "max_per_day": 8,               # 1 日に出す事故の上限(L1 の安全弁)
    },
    # ---- ③ 群集(生成しない = 状態が事件)----
    "crowd": {
        "enabled": True,
        "levels": list(CROWD_LEVELS),   # 4 / 6 / 13 人/m²
        "hysteresis": 0.85,             # 閾値 × この係数を下回ったら「跨ぎ」を再武装
        # 雑踏警備として数える職の語(**conf で差し替え可**)。既定 = GUARD_OCCS。
        # ★payload の guards に出るだけ = 密度も行動も 1 ミリも動かさない(読み取り専用)。
        "guard_occupations": GUARD_OCCS,
        "guard_radius_m": 50.0,         # 雑踏警備の頭数を数える半径(**読むだけ**)
        "max_per_day": 24,              # 1 日に出す群集 incident の上限(安全弁)
    },
}

_TOP_INT = ("max_events_per_step",)


# --------------------------------------------------------------------------- #
# cfg 正準化(city_ops.build_cfg / traces.build_cfg と同型:
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


def _weights(raw, defaults: dict, allowed: tuple) -> dict:
    """重み表の正準化(**表に無い語は捨てる**・負値は 0・空なら既定へ戻す)。"""
    out = dict(defaults)
    got = dict(_to_plain(raw) or {})
    for key, val in got.items():
        if str(key) in allowed:
            try:
                out[str(key)] = max(0.0, float(val))
            except (TypeError, ValueError):
                continue
    if sum(out.values()) <= 0.0:                   # 全部 0 = 退化 → 既定へ戻す
        return dict(defaults)
    return out


def _words(raw, fallback, allowed: tuple = ()) -> tuple:
    """語リストの正準化(**宣言順を保つ**・許可語だけ・空なら既定へ戻す)。"""
    got = _to_plain(raw)
    if isinstance(got, str):
        got = [got]
    out = tuple(str(x).strip() for x in (got or ()) if str(x).strip()
                and (not allowed or str(x).strip() in allowed))
    return out or tuple(fallback)


def _block(raw, defaults: dict, floats: tuple, bools: tuple,
           skip: tuple = ()) -> dict:
    """サブブロック 1 つの正準化(既定を複製し、来たキーだけ型強制して上書きする)。"""
    out = dict(defaults)
    got = dict(_to_plain(raw) or {})
    for key, value in got.items():
        if key not in out or key in skip:
            continue                               # 未知キーは捨てる(捏造しない)
        if key in bools:
            out[key] = bool(value)
        elif key in floats:
            try:
                out[key] = max(0.0, float(value))
            except (TypeError, ValueError):
                continue
        else:
            try:
                out[key] = int(value)
            except (TypeError, ValueError):
                continue
    return out


def build_cfg(raw) -> dict:
    """conf の ``incidents_env`` ブロックを型強制つきで正準化(既定 OFF=現行と完全同一)。"""
    raw = dict(_to_plain(raw) or {})
    cfg: dict = {"enabled": bool(raw.get("enabled", False))}
    for key in _TOP_INT:
        cfg[key] = max(0, int(raw.get(key, DEFAULTS[key])))
    try:
        cfg["area_share"] = max(0.0, float(raw.get("area_share",
                                                   DEFAULTS["area_share"])))
    except (TypeError, ValueError):
        cfg["area_share"] = 0.0

    # ---- ① 火災 ----
    fdef = DEFAULTS["fire"]
    fire = _block(raw.get("fire"), fdef,
                  ("jurisdiction_per_day", "jurisdiction_to_ward", "level_gain",
                   "cold_temp_c", "cold_gain", "discover_radius_m",
                   "response_reference_min", "speed_m_per_min"),
                  ("enabled",),
                  skip=("severity_weights", "appliance_weights", "use_weights",
                        "burn_steps", "injury_severities", "occupations"))
    got = dict(_to_plain(raw.get("fire")) or {})
    # 職の語は**許可語の表を持たない**(名簿の語彙は conf/台帳の側にあるので、ここで
    # 有限表に閉じると新しい名簿の語を書けなくなる)。空なら既定へ戻す = _words の規約。
    fire["occupations"] = _words(got.get("occupations"), fdef["occupations"])
    fire["severity_weights"] = _weights(got.get("severity_weights"),
                                        fdef["severity_weights"], FIRE_SEVERITIES)
    fire["appliance_weights"] = _weights(got.get("appliance_weights"),
                                         fdef["appliance_weights"], APPLIANCES)
    fire["use_weights"] = _weights(got.get("use_weights"), fdef["use_weights"], USES)
    burn = dict(fdef["burn_steps"])
    for key, val in dict(_to_plain(got.get("burn_steps")) or {}).items():
        if str(key) in FIRE_SEVERITIES:
            try:
                burn[str(key)] = max(1, int(val))
            except (TypeError, ValueError):
                continue
    fire["burn_steps"] = burn
    fire["injury_severities"] = _words(got.get("injury_severities"),
                                       fdef["injury_severities"], FIRE_SEVERITIES)
    fire["on_scene_steps"] = max(1, int(fire["on_scene_steps"]))
    fire["max_per_day"] = max(0, int(fire["max_per_day"]))
    cfg["fire"] = fire

    # ---- ② 交通 ----
    tdef = DEFAULTS["traffic"]
    trf = _block(raw.get("traffic"), tdef,
                 ("jurisdiction_per_day", "jurisdiction_to_ward",
                  "pedestrian_share", "hazard_per_exposure", "call_radius_m"),
                 ("enabled", "signalized_only"),
                 skip=("severity_weights", "injury_severities"))
    got = dict(_to_plain(raw.get("traffic")) or {})
    trf["severity_weights"] = _weights(got.get("severity_weights"),
                                       tdef["severity_weights"], CRASH_SEVERITIES)
    trf["injury_severities"] = _words(got.get("injury_severities"),
                                      tdef["injury_severities"], CRASH_SEVERITIES)
    trf["max_per_day"] = max(0, int(trf["max_per_day"]))
    cfg["traffic"] = trf

    # ---- ③ 群集 ----
    cdef = DEFAULTS["crowd"]
    crowd = _block(raw.get("crowd"), cdef, ("hysteresis", "guard_radius_m"),
                   ("enabled",), skip=("levels", "guard_occupations"))
    got = dict(_to_plain(raw.get("crowd")) or {})
    crowd["guard_occupations"] = _words(got.get("guard_occupations"),
                                        cdef["guard_occupations"])
    levels = []
    for val in (_to_plain(got.get("levels")) or cdef["levels"]):
        try:
            lv = float(val)
        except (TypeError, ValueError):
            continue
        if lv > 0.0:
            levels.append(lv)
    crowd["levels"] = sorted(set(levels)) or list(CROWD_LEVELS)
    crowd["hysteresis"] = min(1.0, max(0.0, float(crowd["hysteresis"])))
    crowd["max_per_day"] = max(0, int(crowd["max_per_day"]))
    cfg["crowd"] = crowd
    return cfg


def cfg_of(sim) -> dict:
    """設定(初回のみ ``sim.cfg.incidents_env`` から遅延構築してキャッシュ)。

    simulation.py は別レーン所有なので cfg は本 module が遅延構築する
    (``traces.cfg_of`` / ``city_ops.cfg_of`` と同型)。キャッシュ属性 ``sim.incenvcfg``
    は L1/L2/L3/乱数に一切現れない = 既定 OFF のバイト一致を壊さない。
    """
    got = getattr(sim, "incenvcfg", None)
    if got is None:
        try:
            raw = sim.cfg.get("incidents_env", None)
        except Exception:                          # noqa: BLE001(旧 config 互換)
            raw = None
        got = build_cfg(raw)
        sim.incenvcfg = got
    return got


def enabled(sim) -> bool:
    """環境事件層が有効か。既定 OFF=新経路を一切通さない(バイト一致)。"""
    return bool(cfg_of(sim)["enabled"])


# --------------------------------------------------------------------------- #
# 決定論の補助
# --------------------------------------------------------------------------- #
def _stable_hash(value: str) -> int:
    """プロセス非依存の安定ハッシュ(rng.py / city_ops.py と同流儀)。"""
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def _pick(names: tuple, weights: dict, u: float) -> str:
    """重み表からの決定論的な選択(``u`` ∈ [0,1) は呼び出し側が引いた 1 個の乱数)。

    ★並びは ``names`` の宣言順に固定する(dict の反復順に依存しない = 決定論)。
    """
    total = sum(max(0.0, float(weights.get(n, 0.0))) for n in names)
    if total <= 0.0:
        return names[0]
    acc = 0.0
    target = float(u) * total
    for name in names:
        acc += max(0.0, float(weights.get(name, 0.0)))
        if target < acc:
            return name
    return names[-1]


def area_share(sim) -> float:
    """区 → この世界の規模比(地図 bbox の面積 ÷ 渋谷区 15.11 km²)。

    ★**発明した定数を置かない**: 分子は地図のメタデータ(``city.meta["bbox"]`` =
      緯度経度の矩形)から計算する。bbox が無い地図では 0.0 を返し、火災・交通の
      レートも 0 になる(= 規模比が判らないランでは件数を捏造しない)。
    ★conf の ``area_share`` が正なら**そちらが勝つ**(合成地図でのテスト・感度分析用)。
    """
    over = float(cfg_of(sim)["area_share"])
    if over > 0.0:
        return over
    cached = getattr(sim, "_incenv_share", None)
    if cached is not None:
        return cached
    share = 0.0
    try:
        bbox = list((getattr(sim.city, "meta", None) or {}).get("bbox") or ())
    except Exception:                              # noqa: BLE001(合成地図の保険)
        bbox = []
    if len(bbox) == 4:
        lat0, lon0, lat1, lon1 = (float(v) for v in bbox)
        dlat_km = abs(lat1 - lat0) * KM_PER_DEG
        mid = math.radians((lat0 + lat1) * 0.5)
        dlon_km = abs(lon1 - lon0) * KM_PER_DEG * math.cos(mid)
        share = max(0.0, dlat_km * dlon_km / WARD_AREA_KM2)
    sim._incenv_share = share
    return share


def _in_area(agent) -> bool:
    """世界の中に居て起きているか(通報者・被害者・隊員の共通条件)。"""
    return agent.loc != "outside" and not agent.sleeping


# --------------------------------------------------------------------------- #
# state(sim 側。**OFF では 1 つも生えない**。checkpoint.py が中央管理する)
# --------------------------------------------------------------------------- #
def _state(sim) -> dict:
    st = getattr(sim, "_incenv_state", None)
    if st is None:
        st = {"schema": SCHEMA, "by_kind": {}, "dropped": 0,
              # 火災
              "fires": 0, "fires_by_severity": {}, "fires_by_appliance": {},
              "fire_reports": 0, "fire_undiscovered": 0, "fire_dispatches": 0,
              "fire_unstaffed": 0, "fires_by_day": {}, "live_fires": {},
              "next_fire_id": 1,
              # 交通
              "crashes": 0, "crashes_by_severity": {}, "crashes_by_day": {},
              "exposure_sum": 0.0, "exposure_steps": 0,
              # 群集
              "crowd_events": 0, "crowd_by_level": {}, "crowd_armed": {},
              "crowd_by_day": {}, "density_max": 0.0,
              # 救急連鎖
              "injuries": 0, "ems_calls": 0, "ems_dispatches": 0,
              "ems_unstaffed": 0, "notes": []}
        sim._incenv_state = st
    return st


def _bump(table: dict, key: str, n: int = 1) -> None:
    table[key] = int(table.get(key, 0)) + int(n)


class _Budget:
    """1 step に出す L1 件数の上限(暴走時の L1 膨張対策。超過分は**捨てて数える**)。

    ``city_ops._Budget`` と同型(同じ安全弁を 2 つの書き方で持たない)。
    """

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


# =========================================================================== #
# 救急連鎖への接続(**city_ops を編集せず・既存 kind と読み取り専用ヘルパだけで乗る**)
# =========================================================================== #
def _ems_enabled(sim) -> bool:
    """救急の実体(当直・出動)が世界に在るか。**読むだけ**(city_ops を書き換えない)。"""
    from . import city_ops as _city_ops
    if not _city_ops.enabled(sim):
        return False
    return bool(_city_ops.cfg_of(sim)["ems"]["enabled"])


def _ems_answer(sim, victim, step: int, sim_min: int) -> dict:
    """通報への応答(``city_ops`` の**公開シーム** ``request_ems`` をそのまま使う)。

    ★選定規則を二重定義しない。シームが将来 city_ops から消えたら**黙って 0 件**に
      なるので、tests/test_incidents_env.py が実在を機械固定する。
    ★出動中の印(``city_ops_ems_until`` / ``city_ops_ems_home``)を付けるのもシームの側で、
      持ち場へ戻す処理は ``city_ops._ems_restore`` が従来どおり行う(復帰経路を新設しない)。
    """
    empty = {"crew": None, "response_min": None, "unstaffed": True}
    from . import city_ops as _city_ops
    seam = getattr(_city_ops, "request_ems", None)
    if seam is None:                               # 綴り替え・削除に対する保険
        return empty
    try:
        return seam(sim, victim, None, "injury", step=int(step), sim_min=int(sim_min))
    except Exception:                              # noqa: BLE001(city_ops 側の変更に対する保険)
        return empty


def _note_injury(sim, victim, severity: str, source: str, step: int,
                 sim_min: int) -> None:
    """負傷を**身体の状態機械へ載せる**(H1 の公開 API ``health.on_injury``)。

    ★あちらが OFF(重症度 OFF)のときは 1 バイトも動かない = 本 module の従来挙動と同値。
      見立て(``INJURY_HINT``)だけを渡し、重症度・療養日数・sick-role・記憶は
      **健康側が決める**(身体の意味論を 2 箇所に書かない)。
    """
    from . import health as _health
    hook = getattr(_health, "on_injury", None)
    if not callable(hook):
        return
    try:
        hook(sim, victim, INJURY_HINT.get(str(severity), 1), str(source),
             step=int(step), sim_min=int(sim_min))
    except Exception:                              # noqa: BLE001(あちらの契約が違っても壊さない)
        pass


def _nearest_bystander(sim, victim, radius_m: float):
    """負傷者の近くに居合わせた個体(**最小 id で先勝ち**・同距離は id 昇順)。

    ``city_ops._caller_for`` と同じ規約(街の中・起きている・他人)。あちらの private を
    呼ばないのは、あちらが ``city_ops_down``(救急連鎖の内部印)を見るためで、火災・
    事故の現場ではその印は立っていない = 条件が 1 つ余る。
    """
    r2 = float(radius_m) * float(radius_m)
    best = None
    best_key = None
    for other in sim.agents:
        if int(other.id) == int(victim.id) or not _in_area(other):
            continue
        dx = float(other.x) - float(victim.x)
        dy = float(other.y) - float(victim.y)
        d2 = dx * dx + dy * dy
        if d2 > r2:
            continue
        key = (d2, int(other.id))
        if best_key is None or key < best_key:
            best_key, best = key, other
    return best


def _ems_chain(sim, cfg: dict, st: dict, bud: _Budget, victim, source: str,
               severity: str, radius_m: float, step: int, sim_min: int) -> None:
    """負傷 → 通報 → 出動(**既存 kind ``injury`` / ``ems_call`` / ``ems_dispatch``**)。

    ★身体の状態(``sick`` / 重症度 / 在席)は **1 バイトも書かない**。書くのは H1 の
      公開 API ``health.on_injury`` で、本 module が渡すのは重度 1 語の**見立て**だけである
      (身体の意味論を 2 つのレーンに書かないための線引き)。あちらが OFF なら従来どおり
      「負傷したという世界の事実」と「通報という行為」だけが残る。
    """
    node = str(getattr(victim, "node", "") or "")
    bud.log(Event(step=step, sim_min=sim_min, agent_id=int(victim.id),
                  kind="injury", x=victim.x, y=victim.y,
                  payload={"victim": int(victim.id), "source": str(source),
                           "node": node, "severity": str(severity)}))
    st["injuries"] += 1
    # 身体の状態機械へ載せる(H1 が OFF のランでは 1 バイトも動かない = 従来と同値)。
    _note_injury(sim, victim, str(severity), str(source), step, sim_min)
    caller = _nearest_bystander(sim, victim, radius_m)
    self_call = caller is None
    if self_call:
        caller = victim
    dist_m = 0.0 if self_call else round(
        math.hypot(float(caller.x) - float(victim.x),
                   float(caller.y) - float(victim.y)), 1)
    bud.log(Event(step=step, sim_min=sim_min, agent_id=int(caller.id),
                  kind="ems_call", x=caller.x, y=caller.y,
                  payload={"patient": int(victim.id), "node": node,
                           "self_call": bool(self_call), "dist_m": dist_m,
                           "source": str(source)}))
    st["ems_calls"] += 1
    caller.remember(SELF_CALL_TEXT if self_call else ACCIDENT_CALL_TEXT)
    answer = _ems_answer(sim, victim, step, sim_min) if _ems_enabled(sim) \
        else {"crew": None, "response_min": None}
    crew = answer["crew"]
    if crew is None:
        # ★救急の実体が世界に無い(city_ops.ems が OFF)か、当直が居ない。
        #   **装置 id は与えない**(救急を運行する装置は世界に存在しない = city_ops と同じ線引き)。
        bud.log(Event(step=step, sim_min=sim_min, agent_id=-1,
                      kind="ems_dispatch", x=victim.x, y=victim.y,
                      payload={"node": node, "patient": int(victim.id),
                               "caller": int(caller.id), "crew": -1,
                               "response_min": None, "unstaffed": True,
                               "source": str(source)}))
        st["ems_unstaffed"] += 1
        return
    # ★出動中の印(``city_ops_ems_until`` / ``city_ops_ems_home``)と応答時間はシームが
    #   付けた = 選定規則も復帰経路も本 module には 1 行も無い。
    bud.log(Event(step=step, sim_min=sim_min, agent_id=int(crew.id),
                  kind="ems_dispatch", x=crew.x, y=crew.y,
                  payload={"node": node, "patient": int(victim.id),
                           "caller": int(caller.id), "crew": int(crew.id),
                           "response_min": answer["response_min"],
                           "unstaffed": False, "source": str(source)}))
    st["ems_dispatches"] += 1


# =========================================================================== #
# ① 火災 = 薄いレート + 完全アクター連鎖
# =========================================================================== #
def _building_use(sim, bld: dict) -> str:
    """建物の用途(**地図の純関数**)。POI カテゴリ優先 → 建物 kind → other。"""
    cats = {str(p.get("cat") or "") for p in sim.city.pois_in_building(str(bld["id"]))}
    if "food" in cats or "cafe" in cats or "nightlife" in cats:
        return "eating"
    kind = str(bld.get("kind") or "")
    if kind == "retail" or "shop" in cats:
        return "retail"
    if kind == "office" or "office" in cats:
        return "office"
    if kind == "hotel" or "hotel" in cats:
        return "lodging"
    if kind in ("public", "station"):
        return "public"
    if kind in ("residential", "house?"):
        return "residential"
    return "other"


def fire_targets(sim) -> list[dict]:
    """出火しうる建物の表(``[{id, node, use, levels, weight}]``。地図の純関数・乱数ゼロ)。

    重みは **用途 × 規模**(``use_weights`` × (1 + level_gain × (levels − 1)))。
    並びは建物 id 昇順(= 決定論)。sim に 1 度だけキャッシュする。
    """
    cache = getattr(sim, "_incenv_fire_targets", None)
    if cache is not None:
        return cache
    fcfg = cfg_of(sim)["fire"]
    gain = float(fcfg["level_gain"])
    out: list[dict] = []
    for bld in sorted(sim.city.buildings, key=lambda b: str(b["id"])):
        node = str(bld.get("entrance") or "")
        if not node:
            continue
        use = _building_use(sim, bld)
        levels = max(1, int(bld.get("levels") or 1))
        weight = float(fcfg["use_weights"].get(use, 1.0)) * (1.0 + gain * (levels - 1))
        if weight <= 0.0:
            continue
        out.append({"id": str(bld["id"]), "node": node, "use": use,
                    "levels": levels, "weight": weight})
    sim._incenv_fire_targets = out
    return out


def _cold_factor(sim, fcfg: dict) -> float:
    """季節の共変量(**気温が低いほど暖房起因が増える**)。天気が無いランでは 1.0。"""
    weather = getattr(sim, "today_weather", None)
    if not weather:
        return 1.0
    try:
        temp = float(weather.get("temp_hi"))
    except (TypeError, ValueError):
        return 1.0
    base = float(fcfg["cold_temp_c"])
    if base <= 0.0 or temp >= base:
        return 1.0
    chill = min(1.0, (base - temp) / base)
    return 1.0 + float(fcfg["cold_gain"]) * chill


def _appliance_weights(fcfg: dict, use: str, cold: float) -> dict:
    """器具帰属の重み(**用途と季節の共変量**を掛けた表)。

    - ``kitchen`` は火気を扱う用途(eating)でだけ増える。
    - ``heating`` は寒い日ほど増える(``cold`` は ``_cold_factor`` の戻り値)。
    - ``none``(たばこ・放火・不明の族)は共変量で動かさない = **73% の内訳だけが動く**。
    """
    base = dict(fcfg["appliance_weights"])
    out = dict(base)
    out["kitchen"] = base["kitchen"] * (3.0 if use == "eating" else 0.6)
    out["heating"] = base["heating"] * cold
    return out


def _first_discoverer(sim, node: str, bld_id: str, radius_m: float):
    """第一発見者(**建物の中に居る者が優先**・次に半径内の最も近い者・同距離は id 昇順)。

    共在の判定は **``sim.percept_index`` を持たない経路でも成立する素の距離**で行う
    (索引は step 前半の位置で張られており、火災は step 末に判定するため)。
    """
    try:
        tx, ty = sim.city.node_xy(str(node))
    except Exception:                              # noqa: BLE001(未知ノードの保険)
        return None, False
    r2 = float(radius_m) * float(radius_m)
    inside_best = None
    inside_key = None
    near_best = None
    near_key = None
    for agent in sim.agents:
        if not _in_area(agent):
            continue
        if str(getattr(agent, "building", "") or "") == str(bld_id):
            key = int(agent.id)
            if inside_key is None or key < inside_key:
                inside_key, inside_best = key, agent
            continue
        d2 = (float(agent.x) - tx) ** 2 + (float(agent.y) - ty) ** 2
        if d2 > r2:
            continue
        key = (d2, int(agent.id))
        if near_key is None or key < near_key:
            near_key, near_best = key, agent
    if inside_best is not None:
        return inside_best, True
    return near_best, False


def _fire_crew(sim, node: str, sim_min: int):
    """出場できる当直の消防隊(現場に最も近い 1 人。同距離は id 昇順)。

    条件は ``city_ops._on_duty_crew`` と同じ 3 つ(その職 / 街の中に居る / 勤務窓の中)に
    「別の火災へ出場していない」を足したもの。★救急の出動中(``city_ops_ems_until``)も
    除く = **同じ人を 2 つの現場へ送らない**(消防と救急は同じ名簿を共有する)。
    """
    from .cognition import routine as _routine
    cal = getattr(sim, "calendarcfg", None)
    # ★第109バッチ D2: src のハードコードをやめて conf キーから引く
    #(既定 = FIRE_OCCS なので無指定のランは 1 バイトも変わらない)。
    occs = frozenset(cfg_of(sim)["fire"]["occupations"])
    try:
        tx, ty = sim.city.node_xy(str(node))
    except Exception:                              # noqa: BLE001(未知ノードの保険)
        return None
    best = None
    best_key = None
    for agent in sim.agents:
        if str(getattr(agent, "occupation", "")) not in occs or not _in_area(agent):
            continue
        if int(getattr(agent, "incenv_fire_until", -1)) >= 0:
            continue                               # 別の火災に出場中
        if int(getattr(agent, "city_ops_ems_until", -1)) >= 0:
            continue                               # 救急の出動中(名簿を共有している)
        if not _routine.in_work_window(agent, int(sim_min), cal):
            continue
        d2 = (float(agent.x) - tx) ** 2 + (float(agent.y) - ty) ** 2
        key = (d2, int(agent.id))
        if best_key is None or key < best_key:
            best_key, best = key, agent
    return best


def _fire_tick(sim, cfg: dict, st: dict, bud: _Budget, step: int,
               sim_min: int) -> None:
    """出火 → 第一発見者 → 119 → 出場(**通報が無ければ出場も無い**)。"""
    fcfg = cfg["fire"]
    day = int(sim_min) // 1440
    done = int(st["fires_by_day"].get(day, 0))
    if done >= int(fcfg["max_per_day"]):
        return
    share = area_share(sim)
    if share <= 0.0:
        return                                     # 規模比が判らない = 件数を捏造しない
    targets = fire_targets(sim)
    if not targets:
        return
    steps_per_day = max(1, int(sim.clock.steps_per_day))
    cold = _cold_factor(sim, fcfg)
    lam = (float(fcfg["jurisdiction_per_day"]) * float(fcfg["jurisdiction_to_ward"])
           * share * cold / steps_per_day)
    if lam <= 0.0:
        return
    rng = sim.hub.stream("incident_fire", step)
    n_start = int(rng.poisson(lam))
    if n_start <= 0:
        return
    total_w = sum(t["weight"] for t in targets)
    for _ in range(n_start):
        if done >= int(fcfg["max_per_day"]):
            break
        # ---- ① 出火(建物ハザード。用途 × 規模の重みで 1 棟を選ぶ)-------------- #
        target = _weighted_target(targets, total_w, float(rng.random()))
        severity = _pick(FIRE_SEVERITIES, fcfg["severity_weights"],
                         float(rng.random()))
        appliance = _pick(APPLIANCES,
                          _appliance_weights(fcfg, target["use"], cold),
                          float(rng.random()))
        burn = max(1, int(fcfg["burn_steps"].get(severity, 1)))
        fire_id = int(st["next_fire_id"])
        st["next_fire_id"] = fire_id + 1
        weather = getattr(sim, "today_weather", None) or {}
        payload = {"fire_id": fire_id, "building": target["id"],
                   "node": target["node"], "use": target["use"],
                   "levels": int(target["levels"]), "appliance": appliance,
                   "attributed": appliance != "none", "severity": severity,
                   "burn_steps": burn}
        try:
            payload["temp_hi"] = int(weather.get("temp_hi"))
        except (TypeError, ValueError):
            payload["temp_hi"] = None
        bud.log(Event(step=step, sim_min=sim_min, agent_id=-1, kind="fire_start",
                      x=0.0, y=0.0, payload=payload))
        st["fires"] += 1
        done += 1
        st["fires_by_day"][day] = done
        _bump(st["fires_by_severity"], severity)
        _bump(st["fires_by_appliance"], appliance)
        # ---- ② 第一発見者 → 119 通報(**この行為が出場の原因**)---------------- #
        finder, inside = _first_discoverer(sim, target["node"], target["id"],
                                           float(fcfg["discover_radius_m"]))
        record = {"fire_id": fire_id, "node": target["node"],
                  "building": target["id"], "severity": severity,
                  "burn_steps": burn, "out_step": int(step) + burn, "crew": -1,
                  "suppressed": False, "reported": finder is not None}
        st["live_fires"][str(fire_id)] = record
        if finder is None:
            # ★誰も見つけなかった火は**通報されない**(出場も起きない)。
            #   「行為が無ければ応答も無い」= 本計画の三層因果そのもの。
            st["fire_undiscovered"] += 1
            continue
        try:
            fx, fy = sim.city.node_xy(target["node"])
        except Exception:                          # noqa: BLE001
            fx, fy = float(finder.x), float(finder.y)
        bud.log(Event(step=step, sim_min=sim_min, agent_id=int(finder.id),
                      kind="fire_report", x=finder.x, y=finder.y,
                      payload={"fire_id": fire_id, "building": target["id"],
                               "node": target["node"], "inside": bool(inside),
                               "dist_m": round(math.hypot(float(finder.x) - fx,
                                                          float(finder.y) - fy), 1)}))
        st["fire_reports"] += 1
        finder.remember(FIRE_REPORT_TEXT)
        # ---- ③ 出場(通報に応える行為)---------------------------------------- #
        crew = _fire_crew(sim, target["node"], sim_min)
        if crew is None:
            bud.log(Event(step=step, sim_min=sim_min, agent_id=-1,
                          kind="fire_dispatch", x=fx, y=fy,
                          payload={"fire_id": fire_id, "node": target["node"],
                                   "crew": -1, "response_min": None,
                                   "reference_min": float(
                                       fcfg["response_reference_min"]),
                                   "unstaffed": True}))
            st["fire_unstaffed"] += 1
            continue
        dist = math.hypot(float(crew.x) - fx, float(crew.y) - fy)
        response_min = round(dist / max(1.0, float(fcfg["speed_m_per_min"])), 1)
        crew.incenv_fire_home = str(getattr(crew, "work_node", "") or "")
        crew.incenv_fire_until = int(step) + int(fcfg["on_scene_steps"])
        crew.work_node = target["node"]
        record["crew"] = int(crew.id)
        record["suppressed"] = True
        bud.log(Event(step=step, sim_min=sim_min, agent_id=int(crew.id),
                      kind="fire_dispatch", x=crew.x, y=crew.y,
                      payload={"fire_id": fire_id, "node": target["node"],
                               "crew": int(crew.id), "response_min": response_min,
                               "reference_min": float(
                                   fcfg["response_reference_min"]),
                               "unstaffed": False}))
        st["fire_dispatches"] += 1
        crew.remember(FIRE_DISPATCH_TEXT)


def _weighted_target(targets: list, total_w: float, u: float) -> dict:
    """重み付き 1 件の選択(**並びは targets の順 = 建物 id 昇順**で決定論)。"""
    if total_w <= 0.0:
        return targets[0]
    acc = 0.0
    goal = float(u) * total_w
    for target in targets:
        acc += float(target["weight"])
        if goal < acc:
            return target
    return targets[-1]


def _fire_restore(sim, cfg: dict, st: dict, bud: _Budget, step: int,
                  sim_min: int) -> None:
    """鎮火(``fire_out``)と隊員の持ち場戻し。**負傷者はここで救急連鎖へ渡す**。"""
    fcfg = cfg["fire"]
    injury_kinds = frozenset(fcfg["injury_severities"])
    for key in sorted(st["live_fires"], key=lambda k: int(k)):
        rec = st["live_fires"][key]
        if int(step) < int(rec["out_step"]):
            continue
        node = str(rec["node"])
        crew_id = int(rec["crew"])
        # ★出場隊員は**在場者に限る**(``agent_by_id`` は退場者も返す = 幽霊)。退場者を掴むと
        #   fire_out の座標が脱水時点の**古い位置**になり、記憶の 1 行も hydrate で捨てられる。
        #   居ない回は crew=None = ノード座標で記録し agent_id=-1(= 誰が消したか名乗れない)。
        crew = sim.present_agent(crew_id) if crew_id >= 0 else None
        try:
            fx, fy = sim.city.node_xy(node)
        except Exception:                          # noqa: BLE001
            fx, fy = 0.0, 0.0
        # 負傷者 = その建物の中に居る個体のうち id 最小(**実在の在館者だけ**)
        victim = None
        if str(rec["severity"]) in injury_kinds:
            for agent in sim.agents:
                if (_in_area(agent)
                        and str(getattr(agent, "building", "") or "")
                        == str(rec["building"])):
                    if victim is None or int(agent.id) < int(victim.id):
                        victim = agent
        bud.log(Event(step=step, sim_min=sim_min,
                      agent_id=(crew_id if crew is not None else -1),
                      kind="fire_out",
                      x=(crew.x if crew is not None else fx),
                      y=(crew.y if crew is not None else fy),
                      payload={"fire_id": int(key), "node": node,
                               "severity": str(rec["severity"]),
                               "burn_steps": int(rec.get("burn_steps", 1)),
                               "suppressed": bool(rec["suppressed"]),
                               "injured": (1 if victim is not None else 0)}))
        if crew is not None:
            crew.remember(FIRE_OUT_TEXT)
        del st["live_fires"][key]
        if victim is not None:
            _ems_chain(sim, cfg, st, bud, victim, "fire", str(rec["severity"]),
                       float(fcfg["discover_radius_m"]), step, sim_min)
    # 隊員の持ち場戻し(``city_ops._ems_restore`` と同型。印は本 module のもの)
    for agent in sim.agents:
        until = int(getattr(agent, "incenv_fire_until", -1))
        if until >= 0 and int(step) >= until:
            home = str(getattr(agent, "incenv_fire_home", "") or "")
            if home:
                agent.work_node = home
            agent.incenv_fire_until = -1


# =========================================================================== #
# ② 交通 = 曝露の積(歩行者流 × 車流)
# =========================================================================== #
def vehicle_flow(sim) -> int:
    """この step に街の道路網を走った車の台数(**背景交通を読むだけ**)。

    ★なぜ node 別ではなく街全体なのか(**実測して判ったこと**):
      ``world/traffic.py`` の ambient モードは 1 step(600 秒)で車を**経路の端から端まで**
      走らせ、走り終えた車をその step のうちに捨てる(``self.cars = alive`` に残るのは
      経路が余った車だけ)。地図が 1〜2 km 四方で車速が数 km/step なので、実測では
      **step 末の ``cars`` は常に空**だった(60 step のラン全体で在庫 0 台)。つまり
      「その step にその node に居た車」という量は**そもそも保持されていない**。
      持っていない量を推定して node へ按分すると、それは較正ではなく捏造になる。
      そこで車流は ``traffic.last_n``(= その step に軌跡を残した車の台数)という
      **実際に保持されている街全体の量**を使い、場所の情報は歩行者側だけが持つ。
    """
    traffic = getattr(sim, "traffic", None)
    if traffic is None or not getattr(traffic, "enabled", False):
        return 0
    return max(0, int(getattr(traffic, "last_n", 0) or 0))


def road_nodes(sim) -> frozenset:
    """車が通れるエッジに接している node(= **横断部**の候補)。地図の純関数・乱数ゼロ。

    ``world/map.DRIVABLE``(車が通れる道路種の唯一の定義)をそのまま読む
    = 「どこが車道か」の判定を 2 つ目の表として持たない。
    """
    cached = getattr(sim, "_incenv_road_nodes", None)
    if cached is not None:
        return cached
    from .world.map import DRIVABLE
    out: set[str] = set()
    for u, v, data in sim.city.graph.edges(data=True):
        if str(data.get("klass") or "") in DRIVABLE:
            out.add(str(u))
            out.add(str(v))
    got = frozenset(out)
    sim._incenv_road_nodes = got
    return got


def _signalized_nodes(sim) -> frozenset:
    """信号のある交差点ノード(**物理ゾーンの信号宣言から読むだけ**。無ければ空)。

    ★``crossing_id`` → node id の対応は ``world/traffic.crossing_id_of_node`` の**逆**
      (どちらも同じ OSM node id から作られており、グラフ側だけ接頭辞 "n" が付く)。
      座標の近傍探索は使わない = 地図に無い同一性を捏造しない(あちらと同じ線引き)。
    """
    cached = getattr(sim, "_incenv_signal_nodes", None)
    if cached is not None:
        return cached
    nodes: set[str] = set()
    try:
        for zone in sim.physcfg["zones"]:
            sig = getattr(zone, "signal", None)
            if not sig:
                continue
            cid = dict(sig).get("crossing_id")
            if cid is not None:
                nodes.add(f"n{int(cid)}")          # 交差点表の id = グラフの node id
    except Exception:                              # noqa: BLE001(physics OFF / 旧 cfg)
        nodes = set()
    out = frozenset(nodes)
    sim._incenv_signal_nodes = out
    return out


def _traffic_tick(sim, cfg: dict, st: dict, bud: _Budget, step: int,
                  sim_min: int) -> None:
    """横断中の個体と背景交通の**曝露の積**だけを引き金にする(抽象的被害者は作らない)。"""
    tcfg = cfg["traffic"]
    day = int(sim_min) // 1440
    done = int(st["crashes_by_day"].get(day, 0))
    if done >= int(tcfg["max_per_day"]):
        return
    veh = vehicle_flow(sim)
    if veh <= 0:
        return                                     # 車が 1 台も走っていない = 曝露ゼロ
    signalized = _signalized_nodes(sim)
    only_signal = bool(tcfg["signalized_only"])
    roads = road_nodes(sim)
    crossers = []
    for agent in sim.agents:
        # **横断中** = 路上に居て・起きていて・まだ経路が残っていて(= 移動中)、
        # いま**車道に接する node**に居る(信号縛りなら信号のある交差点だけ)
        if agent.loc != "street" or agent.sleeping or not agent.route:
            continue
        node = str(agent.node)
        if node not in (signalized if only_signal else roads):
            continue
        crossers.append(agent)
    if not crossers:
        return
    crossers.sort(key=lambda a: int(a.id))         # ★id 昇順 = 決定論
    exposure = float(len(crossers)) * float(veh)   # ★**曝露の積**(歩行者流 × 車流)
    st["exposure_sum"] = float(st["exposure_sum"]) + exposure
    st["exposure_steps"] = int(st["exposure_steps"]) + 1
    lam = (exposure * float(tcfg["hazard_per_exposure"])
           * float(tcfg["jurisdiction_to_ward"]) * float(tcfg["pedestrian_share"]))
    if lam <= 0.0:
        return
    rng = sim.hub.stream("incident_traffic", step)
    n_crash = int(rng.poisson(lam))
    if n_crash <= 0:
        return
    injury_kinds = frozenset(tcfg["injury_severities"])
    for _ in range(n_crash):
        if done >= int(tcfg["max_per_day"]):
            break
        # ---- 被害者は**その step に実在した横断中の個体**(抽象的被害者を作らない)---- #
        victim = crossers[int(rng.integers(len(crossers)))]
        node = str(victim.node)
        severity = _pick(CRASH_SEVERITIES, tcfg["severity_weights"],
                         float(rng.random()))
        event = Event(step=step, sim_min=sim_min, agent_id=int(victim.id),
                      kind="traffic_accident", x=victim.x, y=victim.y,
                      payload={"victim": int(victim.id), "node": node,
                               "ped_n": len(crossers), "veh_n": int(veh),
                               "exposure": round(exposure, 3),
                               "severity": severity,
                               "injured": int(severity in injury_kinds),
                               "signalized": bool(node in signalized)})
        # ★加害車両は背景交通(エージェントではない)= 行を出した発生器の同一性だけを刻む。
        #   causality OFF(既定)では Event を 1 バイトも触らない(devices.stamp の契約)。
        _devices.stamp(sim, event,
                       _devices.traffic_device_id(str(getattr(sim.traffic, "mode",
                                                              "ambient"))))
        bud.log(event)
        st["crashes"] += 1
        done += 1
        st["crashes_by_day"][day] = done
        _bump(st["crashes_by_severity"], severity)
        victim.remember(ACCIDENT_TEXT)
        if severity in injury_kinds:
            _ems_chain(sim, cfg, st, bud, victim, "traffic", severity,
                       float(tcfg["call_radius_m"]), step, sim_min)


# =========================================================================== #
# ③ 群集 = **生成しない**。状態(密度)の閾値跨ぎそのものが事件
# =========================================================================== #
def _guards_near(sim, zone_id: str, radius_m: float) -> int:
    """雑踏警備の頭数(**読むだけ**。密度を 1 ミリも下げない)。

    ゾーンの座標を持たないので、警備員・警察官のうち「その step にそのゾーンの
    在場者と同じ node 群に居る者」ではなく、**街全体の当直数**を数える近似にする…
    のではなく、ゾーン多角形の重心からの半径で数える(物理ゾーンは多角形を持つ)。
    重心が取れないゾーンでは 0(数えられないものを数えたことにしない)。
    """
    try:
        zones = sim.physcfg["zones"]
    except Exception:                              # noqa: BLE001(physics OFF / 旧 cfg)
        return 0
    poly = None
    for zone in zones:
        if str(getattr(zone, "id", "")) == str(zone_id):
            poly = list(getattr(zone, "polygon", ()) or ())
            break
    if not poly:
        return 0
    cx = sum(float(p[0]) for p in poly) / len(poly)
    cy = sum(float(p[1]) for p in poly) / len(poly)
    r2 = float(radius_m) * float(radius_m)
    # ★第109バッチ D2: conf キー(既定 = GUARD_OCCS)。無指定は従来と完全同値。
    occs = frozenset(cfg_of(sim)["crowd"]["guard_occupations"])
    n = 0
    for agent in sim.agents:
        if not _in_area(agent):
            continue
        if str(getattr(agent, "occupation", "")) not in occs:
            continue
        if (float(agent.x) - cx) ** 2 + (float(agent.y) - cy) ** 2 <= r2:
            n += 1
    return n


def _crowd_tick(sim, cfg: dict, st: dict, bud: _Budget, step: int,
                sim_min: int) -> None:
    """物理層が測った密度が閾値を跨いだことを記録する(**乱数ゼロ・状態を動かさない**)。

    ★観測は状態を動かさない: 読むのは ``sim._phys_state["by_zone"]`` だけで、
      ゾーンの占有も個体の位置も 1 バイトも触らない(traces の「観測では痕跡を
      強化しない」と同じ線引き)。
    ★1 ゾーン 1 水準につき **1 エピソード 1 件**(ヒステリシス付き)= 閾値の上を
      うろつくあいだ毎 step 出さない。
    """
    ccfg = cfg["crowd"]
    day = int(sim_min) // 1440
    done = int(st["crowd_by_day"].get(day, 0))
    if done >= int(ccfg["max_per_day"]):
        return
    phys = getattr(sim, "_phys_state", None)
    if not phys:
        return                                     # 物理ゾーンが無いランでは何も測っていない
    by_zone = phys.get("by_zone") or {}
    if not by_zone:
        return
    levels = list(ccfg["levels"])
    hyst = float(ccfg["hysteresis"])
    armed = st["crowd_armed"]
    for zone_id in sorted(by_zone):
        stat = by_zone[zone_id] or {}
        try:
            density = float(stat.get("density", 0.0))
        except (TypeError, ValueError):
            continue
        st["density_max"] = max(float(st["density_max"]), density)
        for index, threshold in enumerate(levels):
            key = f"{zone_id}/{index}"
            fired = bool(armed.get(key, False))
            if density < float(threshold) * hyst:
                armed[key] = False                 # 下振れ = 次の跨ぎを再武装
                continue
            if density < float(threshold) or fired:
                continue
            armed[key] = True
            if done >= int(ccfg["max_per_day"]):
                break
            level = index + 1
            bud.log(Event(step=step, sim_min=sim_min, agent_id=-1,
                          kind="crowd_density_incident", x=0.0, y=0.0,
                          payload={"zone": str(zone_id), "level": level,
                                   "threshold": float(threshold),
                                   "density": round(density, 4),
                                   "occupancy": int(stat.get("occupancy", 0)),
                                   "guards": _guards_near(
                                       sim, zone_id, float(ccfg["guard_radius_m"])),
                                   "near_miss": bool(level == 1)}))
            st["crowd_events"] += 1
            done += 1
            st["crowd_by_day"][day] = done
            _bump(st["crowd_by_level"], str(level))


# =========================================================================== #
# 単一作用点(scheduler の唯一のフック。step 末 = 位置が確定した後に呼ぶ)
# =========================================================================== #
def phase(sim, step: int, sim_min: int) -> None:
    """毎 step: 火災 / 交通 / 群集。**既定 OFF は即 return**(乱数も引かない)。

    ★世界に対して触るのは (a) L1、(b) 記憶の定型 1 行、(c) 出場した隊員の持ち場
      (``work_node``)だけ。所持金・関係・drive・opinion・健康状態・k のどれにも触らない。
    ★``city_ops`` / ``health`` / ``chance`` / ``agent`` の状態は**読むだけ**。
    """
    if not enabled(sim):
        return
    cfg = cfg_of(sim)
    st = _state(sim)
    bud = _Budget(sim, st, int(cfg["max_events_per_step"]))
    if cfg["fire"]["enabled"]:
        _fire_restore(sim, cfg, st, bud, step, sim_min)
        _fire_tick(sim, cfg, st, bud, step, sim_min)
    if cfg["traffic"]["enabled"]:
        _traffic_tick(sim, cfg, st, bud, step, sim_min)
    if cfg["crowd"]["enabled"]:
        _crowd_tick(sim, cfg, st, bud, step, sim_min)


# --------------------------------------------------------------------------- #
# 観測タリー(**OFF では None = 何も出さない**)
# --------------------------------------------------------------------------- #
def _roster_n(sim, occupations) -> int:
    """名簿にその職が何人居るか(**読むだけ**・在場/勤務は問わない純粋な頭数)。

    ``fire_unstaffed`` が「較正の話」なのか「名簿にその職が居ない話」なのかを
    ランの成果物だけで切り分けるための分母(``city_ops._roster`` の数え上げ版)。
    """
    occs = frozenset(occupations)
    return sum(1 for a in getattr(sim, "agents", ()) or ()
               if str(getattr(a, "occupation", "")) in occs
               or str(getattr(a, "role", "")) in occs)


def provenance(sim) -> dict | None:
    """観測タリー(既定 OFF は None)。**実測 vs アンカー**を必ず並べて出す。

    ★``city_ops.provenance`` の ``ems_reference_per_day`` と同じ流儀: 較正が合っている
      かどうかを、ランの成果物そのものが自己申告する(黙って外れたまま通り過ぎない)。
    ★第109バッチ D2: ``summary.json`` の ``incidents_env`` キーへ配線した。同時に
      **担い手の語と名簿の実人数**(``fire_occupations`` / ``fire_roster`` /
      ``guard_occupations`` / ``guard_roster``)を出す。``fire_unstaffed`` が立ったとき
      「較正が外れたのか、名簿にその職が 1 人も居ないのか」を成果物だけで判別できる
      ようにするため(第108 縦煙で消防士 0 人が L1 経由でしか判らなかったことへの対応)。
    """
    if not enabled(sim):
        return None
    cfg = cfg_of(sim)
    share = area_share(sim)
    out: dict = {"schema": SCHEMA, "area_share": round(share, 5),
                 "ward_area_km2": WARD_AREA_KM2}
    fcfg, tcfg = cfg["fire"], cfg["traffic"]
    out["fire_occupations"] = list(fcfg["occupations"])
    out["guard_occupations"] = list(cfg["crowd"]["guard_occupations"])
    out["fire_roster"] = _roster_n(sim, fcfg["occupations"])
    out["guard_roster"] = _roster_n(sim, cfg["crowd"]["guard_occupations"])
    out["fire_reference_per_day"] = round(
        float(fcfg["jurisdiction_per_day"]) * float(fcfg["jurisdiction_to_ward"])
        * share, 4)
    out["traffic_reference_per_day"] = round(
        float(tcfg["jurisdiction_per_day"]) * float(tcfg["jurisdiction_to_ward"])
        * float(tcfg["pedestrian_share"]) * share, 4)
    st = getattr(sim, "_incenv_state", None)
    if st is None:                                 # ON だが 1 度も回っていない
        out.update({"fires": 0, "crashes": 0, "crowd_events": 0, "injuries": 0,
                    "by_kind": {}, "dropped": 0})
        return out
    days = max(1, len(st["fires_by_day"]) or 1)
    exposure_steps = max(1, int(st["exposure_steps"]))
    out.update({
        "fires": int(st["fires"]),
        "fires_per_day": round(float(st["fires"]) / days, 3),
        "fires_by_severity": {k: int(v)
                              for k, v in sorted(st["fires_by_severity"].items())},
        "fires_by_appliance": {k: int(v)
                               for k, v in sorted(st["fires_by_appliance"].items())},
        # ★器具帰属率(実測)= アンカー 73% との突き合わせに使う分子/分母
        "fire_attributed_rate": (
            round(1.0 - float(st["fires_by_appliance"].get("none", 0))
                  / float(st["fires"]), 3) if st["fires"] else None),
        "fire_reports": int(st["fire_reports"]),
        "fire_undiscovered": int(st["fire_undiscovered"]),
        "fire_dispatches": int(st["fire_dispatches"]),
        "fire_unstaffed": int(st["fire_unstaffed"]),
        "crashes": int(st["crashes"]),
        "crashes_by_severity": {k: int(v) for k, v
                                in sorted(st["crashes_by_severity"].items())},
        "exposure_mean": round(float(st["exposure_sum"]) / exposure_steps, 4),
        "crowd_events": int(st["crowd_events"]),
        "crowd_by_level": {k: int(v) for k, v in sorted(st["crowd_by_level"].items())},
        "density_max": round(float(st["density_max"]), 4),
        "injuries": int(st["injuries"]), "ems_calls": int(st["ems_calls"]),
        "ems_dispatches": int(st["ems_dispatches"]),
        "ems_unstaffed": int(st["ems_unstaffed"]),
        "dropped": int(st["dropped"]),
        "by_kind": {k: int(v) for k, v in sorted(st["by_kind"].items())}})
    return out
