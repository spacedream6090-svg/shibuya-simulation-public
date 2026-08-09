"""街の顔 = 路上の生業と条例(Wave 4 III-3 ``street_life``・**既定 OFF**)。

何を解く問題か
--------------
本シムの路上には「店」と「通行人」しか居ない。現実の渋谷の路上でいちばん目に入るのは
**そこで生計を立てている人**である: ティッシュを配る人、駅前で歌う人、昼に出る
キッチンカー、夜の路地の占い師、ハチ公前の街頭演説、駅前の街頭募金、そして
**客引き**と、それを追う**条例パトロール**。加えて、渋谷区は 23 区で最も
**路上生活者**が多い区でもある(区の昼間概数調査で 74 人)。

これらは全部「街の見え方」を作っているのに、シムの中では 1 人も居なかった。本 module は
その層を **tier0(LLM 追加呼ゼロ・乱数ゼロ)**で載せる。

制度的背景(実装が参照した現実の構造。ブランド名・実在個人名は 1 つも書かない)
--------------------------------------------------------------------------------
- **客引き / スカウト**: 渋谷区の条例(2025-04 改正)で区内全域が禁止対象になり、
  **過料 5 万円**・立入調査・氏名公表が入った。実態は「取り締まると散り、居なくなると
  戻る」いたちごっこ。啓発区域は**駅からおよそ 700 m**。
  ★条例は**雇い主(店)も対象**にしている = 客引きは独立した職業ではなく、
  **夜の店に雇われている従業員の副業務**である。したがって本 module は
  「客引き」という L5 役割を**作らない**。夜間店舗(nightlife)に職場を持つ既存の
  L2 従業者の**決定論的な部分集合**が、夜の時間帯に持ち場の前で客引きに立つ。
- **ティッシュ配り**: 道路使用許可があれば合法。駅出口・商店街入口、通勤帯と夕方。
- **ストリートミュージシャン**: 路上演奏の許可は事実上下りない = 黙認と退去の循環。
  夜(18〜23 時)に駅前・商店街入口・広場。
- **キッチンカー**: 許可された区画で昼(11〜14 時)。1 日に数台。
- **路上占い師**: 夜の裏通りに数人。
- **街頭演説**: 駅前広場。選挙期に集中し、人だかりを作る。
- **街頭募金**: 駅前広場、日中、散発。
- **路上生活者**: 区の昼間概数調査 74 人(23 区最多)、夜間推計 150〜200 人。

★**尊厳規約**(ユーザー承認済みの明文条件。実装・テスト・プロンプトの全部に掛かる)
--------------------------------------------------------------------------------
1. 語彙は **路上生活者** のみ。蔑称・スティグマ語は module の文字列に 1 つも書かない
   (tests/test_street_life.py が禁止語リストで機械検査する)。
2. **犯罪・迷惑行為の機構に一切接続しない**。``diversity.tick_crime`` の加害者候補・
   被害者候補・周囲候補の**すべて**から除外する(= L1 の ``crime`` / ``nuisance`` の
   payload に路上生活者の id が 1 度も現れない。テストが機械固定する)。
   これは「路上生活者は犯罪をしない」という現実の主張ではなく、**この機構と結びつけない**
   という設計判断である(結びつければシムがスティグマを再生産するため)。
3. 記憶・行動・発火の権利は他のエージェントと**完全に同一**。属性を 1 つも削らない。
4. **区の支援事業が世界に存在する**(路上支援員の巡回相談 = ``outreach_contact``)。
   接触が積み重なれば既存の住居機構(``home_building`` / ``home_node`` / ``home_floor``)
   経由で住居へ移行しうる(``shelter_move``)。
5. 夜の居場所は**公園のそば・地下通路(高架下相当)**、日中は街を移動する。雨天は
   屋根のある場所へ移る(既存の天気を**読むだけ**)。

R1 ドクトリン
-------------
- 既定 ``street_life.enabled: false`` では ``phase`` / ``bind`` が即 return し、
  **L1 に 1 件も出ず・agent に属性が生えず・sim に state が生えず・プロンプトが
  1 バイトも変わらない**(= ゴールデン L1 バイト一致)。
- **乱数 stream を 1 本も引かない**(全経路が id・step・sim_min・地図の純関数)。
  traces.py と同じ「演算だけで決まる」流儀にした = 新 stream の追加もゼロ。
- **generate() の呼び出しサイトを 1 つも作らない**(LLM 追加呼ゼロ = k 非依存)。
  路上の出来事は**既存の記憶機構**(``agent.remember``)へ 1 行入るだけで、
  プロンプトの**欄**は 1 つも増えない(= ``perception_contract`` の追随不要。
  第96 IF-D が trace_line で欄を足したのとは違い、本 module は既存の記憶欄に乗る)。
- 持ち場への**移動は既存の routine が行う**。``bind`` が
  ``work_node`` / ``work_start_min`` / ``work_end_min`` を与えるだけで、
  「出勤する」動きは 1 行も書いていない(``transit_staff.bind`` と同じ作法)。

実装中に**実測して直した** 2 件(初版の設計が世界の中で成立しなかった箇所)
------------------------------------------------------------------------
1. **多帯の役割は勤務窓を union にした**。朝夕のティッシュ配り・朝夕の街頭演説・
   昼夜の巡回相談は帯を 2 つ持つが、既存の勤務窓は 1 本しか持てない。初版が
   ``bands[0]`` だけを窓にしたところ**夕方の帯が一度も立たなかった**(実測)。
   union にした代償として帯と帯のあいだも持ち場に居ることになるが、その時間は
   ``_in_band`` が外すので**出来事は 1 件も出ない**(居るだけ)。
2. **巡回相談は「回る」行為として作り直した**(``roam`` + ``outreach_radius_m``)。
   初版は「持ち場に立ち・同ノードの人に声をかける」形だったが、路上生活者は
   日中は街を移動し夜の居場所に戻るとすぐ眠るため、接触が**実測 0 件**だった
   (支援員と路上生活者が同じ帯に街に居ても最近接 121 m)。これは
   「区の支援事業が世界に存在しない」のと同じ状態なので、(a) 持ち場一致を条件から外し、
   (b) 判定を 150 m の**一帯**にした。現実の巡回相談(公園・地下通路を回る)の形にも合う。

正直な限界(5 件)
------------------
- **持ち場に着けなければ何も起きない**。出来事は「その agent が持ち場ノードに居る」
  ことが条件で、routine が別の用事を優先すればその日はセッションが立たない
  (実測の稼働率は ``summary.street_life.sessions`` に出る)。
- **resume でセッション/クールダウンが切れる**: 状態は agent の動的属性なので
  checkpoint の対象外(第59 assets / 第96 traces と同じ流儀)。持ち場の割当は
  地図からの決定論なので resume 後も同じ(``bind`` は冪等)。
- **客引きの受け手に「店へ行く」強制はしない**。受諾は記憶 1 行 + L1 の計数までで、
  行き先の書き換えはしない(LLM の自然な反応に委ねる = 街頭広告 ``ad_exposure`` と同じ線引き)。
- **キッチンカーの売上は agent 間の移転**で閉じる(買い手 → 売り手)。org 会計
  (economy_sfc)には接続しない = 金は 1 円も湧かず消えない。街頭募金だけは
  **街の外の団体**へ出るので、SFC ON のとき RoW へ落とす(``row_out("donation")``)。
- **選挙期の判定をしない**。街頭演説は時間帯だけで立つ(選挙暦との結合は
  ``tools`` の立候補機構が別レーン所有のため、読むだけの結合も入れていない)。

既存機構との関係(重複させない)
--------------------------------
| 既存                          | 本 module との関係 |
|---|---|
| ``diversity`` の ``nuisance`` 種 "客引き" | ★**本 module ON のとき供給を止める**(``diversity.nuisance_kinds_for``)。同じ出来事を「無主体の確率イベント」と「雇われた人の行為」で二重に出さない |
| ``_phase_enforcement``(警察官の摘発)     | 罰金の作法(所持金−・区の歳入・payee トークン)を**そのまま踏襲**。ルール DSL の prohibit 経路とは独立(条例は世界の定数として本 module が持つ) |
| ``street.ad_exposure``                    | 「路上で何かに接触した」の先例。**行動転換を押し込まない**中立提示の線引きを踏襲 |
| ``traces``(痕跡)                          | 本 module の L1 は traces の源にはしていない(源の表 ``TRACE_SPECS`` は別レーン所有) |
| ``mobility``(転居)                        | 住居移行は同じ状態欄(``home_building`` 等)を書くだけ。転居機構の関数は呼ばない(あちらは日次フェーズ所有) |
"""
from __future__ import annotations

from .observer.schema import Event, register_event_kind
from .weather import _BAD_CONDS

SCHEMA = 1

# --------------------------------------------------------------------------- #
# L1 イベント種(**材料側 registration**: devices.py / transit_staff.py と同じ流儀。
# observer/schema.py 本体を触らずに自分の種を自分で登録する)。
# ★payload に自由文は 1 つも入らない(node は世界の識別子・role は有限語彙・
#   残りは件数と金額)。
# --------------------------------------------------------------------------- #
register_event_kind("street_performance", "路上での演奏セッション(1 セッション 1 件)"
                                          "{node, role, duration_steps, listeners}")
register_event_kind("street_speech",      "街頭演説のセッション(1 セッション 1 件)"
                                          "{node, duration_steps, listeners}")
register_event_kind("tissue_offer",       "ティッシュ配布のセッション(1 セッション 1 件)"
                                          "{node, offered, taken}")
register_event_kind("stall_sale",         "キッチンカーの販売(1 セッション 1 件・"
                                          "買い手→売り手の移転){node, sold, amount}")
register_event_kind("donation",           "街頭募金(1 セッション 1 件・街の外の団体へ)"
                                          "{node, donors, amount, payee?}")
register_event_kind("fortune_reading",    "路上占いのセッション(1 セッション 1 件)"
                                          "{node, clients, amount}")
register_event_kind("touting",            "★客引き(条例の禁止行為)のセッション"
                                          "{node, targets, accepted, in_zone}")
register_event_kind("touting_warning",    "★客引きへの警告(条例パトロールの第 1 段)"
                                          "{officer, target, node, count}")
register_event_kind("touting_fine",       "★客引きへの過料(条例パトロールの第 2 段)"
                                          "{officer, target, node, amount, payee?}")
register_event_kind("street_disperse",    "★警察官の接近で路上の持ち場を移した"
                                          "(いたちごっこ){role, from, to, officer}")
register_event_kind("outreach_contact",   "区の支援員による路上生活者への巡回相談"
                                          "{worker, target, node, count}")
register_event_kind("shelter_move",       "支援の積み重ねで住居へ移行した"
                                          "{target, building, contacts}")


# --------------------------------------------------------------------------- #
# 役割(= persona の occupation。scripts/build_persona_pool.py の _L5_ROLES と同じ語)
# --------------------------------------------------------------------------- #
TISSUE = "ティッシュ配り"
MUSICIAN = "ストリートミュージシャン"
KITCHEN = "キッチンカー営業者"
FORTUNE = "路上占い師"
SPEECH = "街頭演説者"
FUNDRAISER = "募金スタッフ"
ROUGH = "路上生活者"
OUTREACH = "路上支援員"

# 持ち場の種類(**地図から決定論導出する**。地名・ブランド名を 1 つも書かない)
P_STATION = "station"          # 駅前(駅ノード + その周辺の名前つきノード)
P_PLAZA = "plaza"              # 広場・人だかりのできる地点(POI 密度の高いノード)
P_NIGHT = "nightlife"          # 夜間店舗のあるノード(客引きの現場)
P_SIDE = "sidestreet"          # 裏通り(名前つきで POI の少ないノード)
P_REST = "rest"                # 公園・憩いの場(leisure / attraction POI のノード)
P_COVERED = "covered"          # 屋根のある場所(地下通路 = layer<0 のノード)

# 時間帯は**分 of day**(Δt 非依存の INVARIANT)。日を跨ぐ帯は (start > end) で表す。
ROLE_SPECS: dict[str, dict] = {
    # 通勤帯と夕方。駅出口・商店街入口(現実の配布は道路使用許可つきで駅前に集中)
    TISSUE:     {"post": P_STATION, "bands": ((7 * 60, 10 * 60), (17 * 60, 20 * 60))},
    # 夜。許可は事実上下りず、黙認と退去の循環(= 警察官接近で持ち場を移す)
    MUSICIAN:   {"post": P_PLAZA,   "bands": ((18 * 60, 23 * 60),)},
    # 昼。許可された区画で 11〜14 時
    KITCHEN:    {"post": P_PLAZA,   "bands": ((11 * 60, 14 * 60),)},
    # 夜の裏通り
    FORTUNE:    {"post": P_SIDE,    "bands": ((20 * 60, 24 * 60),)},
    # 朝夕の駅前(人の流れが最大になる帯)
    SPEECH:     {"post": P_STATION, "bands": ((8 * 60, 9 * 60), (17 * 60, 19 * 60))},
    # 日中の駅前広場
    FUNDRAISER: {"post": P_STATION, "bands": ((10 * 60, 17 * 60),)},
    # 区の巡回相談。★**唯一の roam 役割**(持ち場に立つのではなく**回る**)。
    #   ・**昼と夜の 2 帯**にするのは、路上生活者が日中は街を移動していて夜の居場所に
    #     戻るのが遅いから(昼の帯だけでは相談が成立しない = 実測で確認した)。
    #   ・持ち場ノードに居ることを条件にしない(``roam``)のは、そもそも「巡回」だから。
    #     持ち場一致を条件にした初版では、夜の居場所に戻った路上生活者がすぐ就寝するため
    #     接触が**実測 0 件**になった(= 支援事業が世界に存在しないのと同じ状態)。
    #     持ち場(rest ノード)は routine が向かう先として残し、**道中で出会った人**にも
    #     声をかける。これは現実の巡回相談(公園・地下通路を回る)の形にも合う。
    OUTREACH:   {"post": P_REST, "roam": True,
                 "bands": ((14 * 60, 16 * 60), (19 * 60, 23 * 60))},
    # 路上生活者は「持ち場で働く」役割ではない = 勤務窓を与えない(下の bind を参照)。
    # 夜の居場所だけを与え、日中は街を移動する(既存の routine のまま)。
    ROUGH:      {"post": P_REST,    "bands": ()},
}

# 働く役割(勤務窓を与える = 既存 routine が持ち場へ歩かせる)
WORKING_ROLES: tuple[str, ...] = (TISSUE, MUSICIAN, KITCHEN, FORTUNE,
                                  SPEECH, FUNDRAISER, OUTREACH)
# 全役割(名簿の occupation 判定に使う)
STREET_OCCS: tuple[str, ...] = tuple(ROLE_SPECS)

# 警察官(執行を担う公務員)。scheduler.POLICE_OCCS / diversity.POLICE_OCCS と同一
# (循環 import を避けるためここでも定義する。src/society 直下の流儀)。
POLICE_OCCS = ("警察官",)

# 警察官の接近で持ち場を移す役割(黙認と退去の循環が現実に起きている 2 種だけ)
DISPERSIBLE: frozenset = frozenset({MUSICIAN})

# --------------------------------------------------------------------------- #
# 記憶の 1 行(**役割だけの純関数**。数字・金額・config・実験条件・機構語は 1 つも入らない)
#
# ★これは traces.sentence と同じ「no-fingerprint の定型文」規約。同じ役割からは
#   全実験条件で同一バイト列になる(tests が機械固定する)。
# ★路上生活者に関する文は**中立語のみ**(尊厳規約 1)。
# --------------------------------------------------------------------------- #
NOTICE_TEXT: dict[str, str] = {
    MUSICIAN:   "路上で演奏している人がいた。",
    SPEECH:     "駅前で演説をしている人がいた。",
    TISSUE:     "路上でティッシュを受け取った。",
    KITCHEN:    "路上のキッチンカーで昼食を買った。",
    FORTUNE:    "路上の占い師に見てもらった。",
    FUNDRAISER: "街頭の募金に応じた。",
}
TOUT_TEXT = "路上で店の呼び込みに声をかけられた。"
TOUT_WARN_TEXT = "路上の呼び込みで警察官に注意された。"
TOUT_FINE_TEXT = "路上の呼び込みで条例違反の過料を科された。"
OUTREACH_TEXT = "区の支援員から声をかけられた。"
OUTREACH_WORKER_TEXT = "巡回で路上生活の相談に応じた。"
SHELTER_TEXT = "支援を受けて住まいが決まった。"


DEFAULTS = {
    "enabled": False,
    # ---- 持ち場(地図からの決定論導出)----
    "posts_per_role": 4,          # 1 役割に用意する持ち場ノード数(id 昇順で循環割当)
    "notice_radius_m": 25.0,      # 路上の出来事に気づく距離(既存 knowledge radius と別系統)
    "notice_cap": 8,              # 1 セッションで記憶 1 行が入る人数の上限(暴走の安全弁)
    # ---- 活動(**確率ではなく決定論の刻み**: 何人に 1 人が応じるか)----
    "tissue_accept_every": 4,     # 通行人の 4 人に 1 人が受け取る
    "stall_price": 900.0,         # キッチンカーの単価[円]
    "stall_buy_every": 6,
    "fortune_fee": 2000.0,        # 路上占いの料金[円]
    "fortune_every": 12,
    "donation_amount": 300.0,     # 街頭募金の 1 口[円]
    "donation_every": 8,
    # ---- 客引き(条例の対象)----
    "tout_every": 4,              # 夜間店舗の従業者の 4 人に 1 人が客引きに立つ(決定論)
    "tout_accept_every": 10,
    "tout_bands": [19 * 60, 24 * 60],   # 客引きの時間帯[分 of day]
    "zone_radius_m": 700.0,       # 条例の啓発区域(駅からおよそ 700 m)
    # ---- 条例パトロール ----
    "fine_amount": 50000.0,       # 過料 5 万円(2025-04 改正)
    "fine_grievance": 0.0,        # 過料を受けた不満(factors 経由。0.0=接続しない)
    "warn_before_fine": 1,        # 何回目の警告までは過料にしないか(1=2 回目で過料)
    "cooldown_steps": 6,          # 警告/退去のあとに活動を止める step 数(いたちごっこ)
    # ---- 路上生活者(尊厳規約)----
    "rough_money_cap": 3000.0,    # 手持ちの上限[円](名簿の既定値を ON 時に丸める)
    # ★巡回相談が届く範囲[m]。**notice_radius_m(25m)とは別の値**にする理由:
    #   路上の生業は「その場に居合わせた人」に届く出来事(25m = 路上の視聴距離)だが、
    #   巡回相談は**その一帯を回る**行為で、公園や地下通路という**範囲**を対象にする。
    #   初版で 25m を使い回したところ接触が**実測 0 件**になった(支援員 2 人と路上生活者
    #   2 人が同じ帯に街に居ても最近接 121m)= 支援事業が世界に存在しないのと同じ状態。
    "outreach_radius_m": 150.0,
    "outreach_period_days": 3,    # 巡回相談の周期[日]
    "outreach_shelter_after": 3,  # 何回の相談で住居へ移行するか(0=移行なし)
    # ---- L1 安全弁 ----
    "max_events_per_step": 32,    # 1 step に出す本 module の L1 件数の上限
}

_INT_KEYS = ("posts_per_role", "notice_cap", "tissue_accept_every", "stall_buy_every",
             "fortune_every", "donation_every", "tout_every", "tout_accept_every",
             "warn_before_fine", "cooldown_steps", "outreach_period_days",
             "outreach_shelter_after", "max_events_per_step")
_FLOAT_KEYS = ("notice_radius_m", "stall_price", "fortune_fee", "donation_amount",
               "zone_radius_m", "fine_amount", "fine_grievance", "rough_money_cap",
               "outreach_radius_m")


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
    """conf の ``street_life`` ブロックを型強制つきで正準化(既定 OFF=現行挙動と完全同一)。

    dotlist 上書きは文字列で入り得るため型強制する(``traces.build_cfg`` と同じ作法)。
    未知キーは黙って捨てる。``tout_bands`` は [開始, 終了] の分 of day 2 要素。
    """
    raw = dict(_to_plain(raw) or {})
    cfg = dict(DEFAULTS)
    cfg["tout_bands"] = list(DEFAULTS["tout_bands"])
    for k, v in raw.items():
        if k == "enabled":
            cfg["enabled"] = bool(v)
        elif k == "tout_bands":
            got = [int(x) for x in (_to_plain(v) or [])]
            if len(got) == 2:
                cfg["tout_bands"] = got
        elif k in _INT_KEYS:
            cfg[k] = max(0, int(v))
        elif k in _FLOAT_KEYS:
            cfg[k] = max(0.0, float(v))
    for k in ("posts_per_role", "tissue_accept_every", "stall_buy_every",
              "fortune_every", "donation_every", "tout_every", "tout_accept_every"):
        cfg[k] = max(1, int(cfg[k]))               # 0 除算の防止(刻みは 1 以上)
    return cfg


def cfg_of(sim) -> dict:
    """路上の生業の設定(初回のみ ``sim.cfg.street_life`` から遅延構築してキャッシュ)。

    simulation.py は別レーン所有なので cfg は本 module が遅延構築する
    (``traces.cfg_of`` / ``rumors.cfg_of`` と同型)。キャッシュ属性 ``sim.streetlifecfg``
    は L1/L2/L3/乱数に一切現れない = 既定 OFF のバイト一致を壊さない。
    """
    c = getattr(sim, "streetlifecfg", None)
    if c is None:
        try:
            raw = sim.cfg.get("street_life", None)
        except Exception:                          # noqa: BLE001(旧 config 互換)
            raw = None
        c = build_cfg(raw)
        sim.streetlifecfg = c
    return c


def enabled(sim) -> bool:
    """路上の生業(Wave 4 III-3)が有効か。既定 OFF=新経路を一切通さない(バイト一致)。"""
    return bool(cfg_of(sim)["enabled"])


# --------------------------------------------------------------------------- #
# 持ち場の解決(**地図からの決定論導出**。conf にも code にも地名を書かない)
#
# ★ ``street.resolve_slots`` は「名前で引く」流儀だが、本 module は名前を 1 つも
#   持たない(ETHICS: 生成文にブランド名・店名を出さない)。代わりに
#   **地図の構造**(駅ノード・POI 密度・POI カテゴリ・レイヤー)から決定論で選ぶので、
#   どの地図・どの envpack でも動き、固有名詞は 1 つも現れない。
# --------------------------------------------------------------------------- #
def _dests(sim) -> list[str]:
    d = getattr(sim, "dests", None)
    if d:
        return list(d)
    return sorted(sim.city.graph.nodes)


def _rank_by_pois(sim, nodes: list[str], cats: tuple[str, ...] | None,
                  reverse: bool) -> list[str]:
    """POI 件数で並べたノード列(件数の降順/昇順 → node 名昇順のタイブレーク=決定論)。"""
    scored = []
    for n in nodes:
        pois = sim.city.pois_at_node(n)
        if cats is None:
            c = len(pois)
        else:
            c = sum(1 for p in pois if str(p.get("cat") or "") in cats)
        scored.append((c, str(n)))
    scored.sort(key=lambda t: (-t[0] if reverse else t[0], t[1]))
    return [n for _c, n in scored]


def posts(sim, kind: str) -> list[str]:
    """持ち場の種類 → ノード列(**地図の純関数**・乱数ゼロ・sim に 1 度だけキャッシュ)。

    - ``station``  … 駅ノードを先頭に、POI 密度の高い名前つきノードを続ける
    - ``plaza``    … POI 密度の高いノード(人だかりのできる地点)
    - ``nightlife``… 夜間店舗 POI のあるノード(客引きの現場)
    - ``sidestreet``… 名前つきで POI の**少ない**ノード(裏通り)
    - ``rest``     … leisure / attraction POI のノード(公園・憩いの場)。無ければ
                     地下通路(layer<0)へ後退し、それも無ければ POI の少ないノード
    - ``covered``  … 地下通路(layer<0)= 雨をしのげる場所。無ければ ``rest``
    候補が 1 つも無い地図では空リストを返す(= その役割は何もしない)。
    """
    cache = getattr(sim, "_sl_posts", None)
    if cache is None:
        cache = {}
        sim._sl_posts = cache
    got = cache.get(kind)
    if got is not None:
        return got
    cfg = cfg_of(sim)
    top = max(1, int(cfg["posts_per_role"]))
    nodes = _dests(sim)
    graph = sim.city.graph
    named = [n for n in nodes if graph.nodes[n].get("name")]
    station = getattr(sim.city, "station_node", None)
    if kind == P_STATION:
        out = ([str(station)] if station else []) + \
            [n for n in _rank_by_pois(sim, named or nodes, None, True)
             if n != station]
    elif kind == P_PLAZA:
        out = _rank_by_pois(sim, nodes, None, True)
    elif kind == P_NIGHT:
        out = [n for n in _rank_by_pois(sim, nodes, ("nightlife",), True)
               if any(str(p.get("cat") or "") == "nightlife"
                      for p in sim.city.pois_at_node(n))]
    elif kind == P_SIDE:
        out = _rank_by_pois(sim, named or nodes, None, False)
    elif kind == P_COVERED:
        out = [n for n in nodes if sim.city.node_layer(n) < 0]
        if not out:
            return posts(sim, P_REST)
    else:                                          # P_REST
        out = [n for n in _rank_by_pois(sim, nodes, ("leisure", "attraction"), True)
               if any(str(p.get("cat") or "") in ("leisure", "attraction")
                      for p in sim.city.pois_at_node(n))]
        if not out:
            out = [n for n in nodes if sim.city.node_layer(n) < 0]
        if not out:
            out = _rank_by_pois(sim, nodes, None, False)
    out = out[:top]
    cache[kind] = out
    return out


def _station_xy(sim):
    st = getattr(sim.city, "station_node", None)
    return sim.city.node_xy(st) if st else None


def in_zone(sim, node: str) -> bool:
    """条例の啓発区域(駅からおよそ 700 m)の中か。駅ノードが無い地図では常に True
    (= 区域の概念が成立しないので区別しない)。"""
    xy = _station_xy(sim)
    if xy is None:
        return True
    r = float(cfg_of(sim)["zone_radius_m"])
    x, y = sim.city.node_xy(node)
    return (x - xy[0]) ** 2 + (y - xy[1]) ** 2 <= r * r


# --------------------------------------------------------------------------- #
# 束ね(起動時 1 回・**冪等**・乱数ゼロ・resume 安全)
# --------------------------------------------------------------------------- #
def _rain(sim) -> bool:
    """今日は雨/雪か(既存の天気を**読むだけ**。天気 OFF なら False)。"""
    w = getattr(sim, "today_weather", None)
    return bool(w) and str(w.get("cond")) in _BAD_CONDS


def _is_tout(sim, agent, cfg: dict) -> bool:
    """この従業者が客引きに立つか(**決定論の部分集合**・乱数ゼロ)。

    条件: (a) 職場ノードが**夜間店舗のあるノード**である(= 夜の店に雇われている)、
    (b) id が ``tout_every`` の刻みに当たる。★条例が雇い主も対象にしている構造に合わせ、
    「客引き」という独立の職業は作らない(module docstring)。
    """
    wn = str(getattr(agent, "work_node", "") or "")
    if not wn:
        return False
    if str(getattr(agent, "occupation", "")) in STREET_OCCS:
        return False                               # 路上の役割は客引きを兼ねない
    if not any(str(p.get("cat") or "") == "nightlife"
               for p in sim.city.pois_at_node(wn)):
        return False
    return int(agent.id) % max(1, int(cfg["tout_every"])) == 0


def bind(sim) -> dict:
    """路上の役割を持ち場へ束ねる(決定論・乱数ゼロ・**冪等**・resume 安全)。

    - 並びは **agent id 昇順**(持ち場の循環割当もこの並びで決まる)。
    - 働く役割(``WORKING_ROLES``)には ``work_node`` と勤務窓を与える
      = **移動は既存 routine が行う**(``transit_staff.bind`` と同じ作法)。
      持ち場は路面(``work_building`` は空)= 路上の生業だから。
    - **路上生活者**には勤務窓を与えない。夜の居場所(``home_node``)だけを与え、
      建物を持たない(``home_building=""``)= 既存の ``go_to_bed`` がその場で眠らせる
      (立退き中の路上就寝と同じ既存経路。新しい睡眠機構は 1 行も作らない)。
      手持ちは ``rough_money_cap`` で丸める(名簿の既定レンジは住民向けのため)。
    - **客引き**(夜間店舗の従業者の決定論部分集合)は職場を路面に落とす
      = 店の前の路上に立つ(現実の客引きの位置)。勤務窓は既存のまま触らない。
    """
    out = {"n_role": 0, "n_rough": 0, "n_tout": 0, "posts": {}}
    if not enabled(sim):
        return out
    cfg = cfg_of(sim)
    agents = sorted(sim.agents, key=lambda a: int(a.id))
    per_role: dict[str, int] = {}
    for a in agents:
        occ = str(getattr(a, "occupation", ""))
        spec = ROLE_SPECS.get(occ)
        if spec is None:
            continue
        pool = posts(sim, spec["post"])
        if not pool:
            continue
        idx = per_role.get(occ, 0)
        per_role[occ] = idx + 1
        node = pool[idx % len(pool)]
        if occ == ROUGH:
            # 尊厳規約 5: 夜の居場所 = 公園のそば / 地下通路。雨天は屋根のある場所。
            a.street_sleep_node = str(node)
            a.home_node = str(node)
            a.home_building = ""
            a.home_floor = 0
            a.money = min(float(a.money), float(cfg["rough_money_cap"]))
            a.outreach_count = int(getattr(a, "outreach_count", 0))
            out["n_rough"] += 1
            continue
        a.street_post = str(node)
        a.street_post_idx = int(idx)
        if str(getattr(a, "work_node", "") or "") != str(node):
            a.work_node = str(node)
        a.work_building = ""                       # 路上の生業(建物内勤務にしない)
        a.work_floor = 0
        # ★勤務窓は**必ず**役割の帯で上書きする(``transit_staff.bind`` が既存窓を残すのとは
        #   逆の判断)。理由: 路上の生業は**職業そのものが時間帯**であって、名簿側の一般的な
        #   勤務窓(9〜17 時)を残すと夜の演奏者が昼に持ち場へ立つ。名簿(L5 pool)経由の
        #   個体は職場 POI を持たない = work_start_min が -1 なので、上書きしても実害は無い。
        # ★2 帯以上の役割(朝夕のティッシュ配り・朝夕の街頭演説・昼夜の巡回相談)は
        #   **union の範囲**を勤務窓にする。既存の勤務窓は 1 本しか持てないので、
        #   bands[0] だけを使うと夕方の帯が一度も立たない(実測で確認した)。
        #   正直な近似: 帯と帯のあいだも持ち場に居ることになるが、その時間は
        #   ``_in_band`` が外すので**出来事は 1 件も出ない**(居るだけ)。
        if spec["bands"]:
            a.work_start_min = int(spec["bands"][0][0])
            a.work_end_min = int(min(1440, max(b[1] for b in spec["bands"])))
        out["n_role"] += 1
    # 客引き = 夜間店舗の従業者の決定論部分集合(独立した職業を作らない)
    for a in agents:
        if _is_tout(sim, a, cfg):
            a.street_tout = True
            a.work_building = ""                   # 店の前の路上に立つ
            a.work_floor = 0
            out["n_tout"] += 1
    out["posts"] = {k: len(posts(sim, k))
                    for k in (P_STATION, P_PLAZA, P_NIGHT, P_SIDE, P_REST, P_COVERED)}
    return out


def rough_sleeper_ids(sim) -> frozenset:
    """路上生活者の id 集合(**尊厳規約 2**: 犯罪・迷惑機構から除外するための唯一の口)。

    ★OFF では**必ず空集合**を返す(= ``diversity`` 側のフィルタが完全 no-op になり、
      ゴールデン L1 バイト一致が壊れない)。sim に state を生やさない読み取り専用。
    """
    if not enabled(sim):
        return frozenset()
    cached = getattr(sim, "_sl_rough_ids", None)
    if cached is not None:
        return cached
    ids = frozenset(int(a.id) for a in sim.agents
                    if str(getattr(a, "occupation", "")) == ROUGH)
    sim._sl_rough_ids = ids
    return ids


# --------------------------------------------------------------------------- #
# 時間帯・セッション
# --------------------------------------------------------------------------- #
def _in_band(bands, sim_min: int) -> int:
    """今の分 of day がどの帯か(帯の index。外なら -1)。start>end は日跨ぎの帯。"""
    mod = int(sim_min) % 1440
    for i, (s, e) in enumerate(bands):
        s, e = int(s), int(e)
        if s <= e:
            if s <= mod < e:
                return i
        elif mod >= s or mod < e:                  # 日跨ぎ
            return i
    return -1


def _band_steps(sim, band) -> int:
    """帯の長さ[step](記録用。Δt を読むだけで世界は動かさない)。"""
    s, e = int(band[0]), int(band[1])
    span = (e - s) if s <= e else (1440 - s + e)
    return max(1, span // max(1, int(sim.clock.step_minutes)))


def _state(sim) -> dict:
    st = getattr(sim, "_sl_state", None)
    if st is None:
        st = {"schema": SCHEMA, "bound": False, "sessions": 0, "by_kind": {},
              "notices": 0, "warnings": 0, "fines": 0, "fine_total": 0.0,
              "disperses": 0, "contacts": 0, "shelters": 0, "dropped": 0}
        sim._sl_state = st
    return st


def _bump(table: dict, key: str, n: int = 1) -> None:
    table[key] = int(table.get(key, 0)) + int(n)


class _Budget:
    """1 step に出す L1 件数の上限(暴走時の L1 膨張対策。超過分は**捨てる**)。"""

    def __init__(self, sim, st: dict, cap: int):
        self.sim = sim
        self.st = st
        self.left = int(cap)

    def log(self, ev) -> bool:
        if self.left <= 0:
            self.st["dropped"] += 1
            return False
        self.left -= 1
        self.sim.logger.log(ev)
        _bump(self.st["by_kind"], ev.kind)
        return True


# --------------------------------------------------------------------------- #
# 通行人(**既存の位置データを読むだけ**。知覚機構も prompt も触らない)
# --------------------------------------------------------------------------- #
def _passersby(sim, actor, radius: float, cap: int) -> list:
    """持ち場の周りに居る通行人(id 昇順・上限つき)。

    路上に居て・起きていて・街の中に居る他人だけ。**公務員も路上の役割も除かない**
    (誰でも通行人になる)が、行為者自身は除く。乱数ゼロ・状態を 1 バイトも触らない。
    """
    r2 = float(radius) * float(radius)
    out = []
    # ★``sim.percept_index``(知覚半径の空間索引)は**使わない**: 路上の視聴距離
    #   (notice_radius_m)は知覚半径とは別系統の量で、索引は知覚半径のセル幅で
    #   張られている。別の半径で引くと索引の意味が壊れるので素直に走査する
    #   (対象は持ち場に立っている数十人だけなので O(n) の外側ループは小さい)。
    for o in sim.agents:
        if int(o.id) == int(actor.id):
            continue
        if o.loc == "outside" or o.sleeping or o.building:
            continue
        dx = float(o.x) - float(actor.x)
        dy = float(o.y) - float(actor.y)
        if dx * dx + dy * dy > r2:
            continue
        out.append(o)
        if len(out) >= cap * 4:                    # 走査の早期打ち切り(id 昇順で十分)
            break
    out.sort(key=lambda o: int(o.id))
    return out[:cap]


def _notice(agent, text: str) -> None:
    """路上の出来事に気づいた 1 行を**既存の記憶機構**へ入れる(プロンプトの欄は増えない)。"""
    agent.remember(text)


# --------------------------------------------------------------------------- #
# 役割ごとの 1 セッション(**セッション単位**で L1 に 1 件だけ = 毎 step ではない)
# --------------------------------------------------------------------------- #
def _session_key(day: int, band_idx: int, node: str) -> str:
    return f"{int(day)}:{int(band_idx)}:{node}"


def _run_session(sim, cfg: dict, st: dict, bud: _Budget, a, occ: str,
                 band_idx: int, band, step: int, sim_min: int) -> None:
    """持ち場に立っている 1 人の 1 step 分の作用(セッションの開始で L1 を 1 件出す)。"""
    day = int(sim_min) // 1440
    key = _session_key(day, band_idx, str(a.node))
    fresh = getattr(a, "street_session", None) != key
    if fresh:
        a.street_session = key
        a.street_seen = set()
    seen = getattr(a, "street_seen", None)
    if seen is None:
        seen = set()
        a.street_seen = seen
    cap = max(0, int(cfg["notice_cap"]) - len(seen))
    crowd = _passersby(sim, a, float(cfg["notice_radius_m"]), cap) if cap else []
    fresh_crowd = [o for o in crowd if int(o.id) not in seen]

    if occ in (MUSICIAN, SPEECH):
        # 演奏・演説: その場に居合わせた人の記憶に 1 行(行動は押し込まない)
        for o in fresh_crowd:
            seen.add(int(o.id))
            _notice(o, NOTICE_TEXT[occ])
        st["notices"] += len(fresh_crowd)
        if fresh:
            kind = "street_performance" if occ == MUSICIAN else "street_speech"
            payload = {"node": str(a.node), "duration_steps": _band_steps(sim, band),
                       "listeners": len(fresh_crowd)}
            if occ == MUSICIAN:
                payload["role"] = occ
            bud.log(Event(step=step, sim_min=sim_min, agent_id=int(a.id),
                          kind=kind, x=a.x, y=a.y, payload=payload))
            st["sessions"] += 1
        return

    if occ == TISSUE:
        every = int(cfg["tissue_accept_every"])
        taken = [o for o in fresh_crowd if (int(o.id) + step) % every == 0]
        for o in fresh_crowd:
            seen.add(int(o.id))
        for o in taken:
            _notice(o, NOTICE_TEXT[TISSUE])
        st["notices"] += len(taken)
        if fresh:
            bud.log(Event(step=step, sim_min=sim_min, agent_id=int(a.id),
                          kind="tissue_offer", x=a.x, y=a.y,
                          payload={"node": str(a.node), "offered": len(fresh_crowd),
                                   "taken": len(taken)}))
            st["sessions"] += 1
        return

    if occ in (KITCHEN, FORTUNE, FUNDRAISER):
        if occ == KITCHEN:
            every, price = int(cfg["stall_buy_every"]), float(cfg["stall_price"])
        elif occ == FORTUNE:
            every, price = int(cfg["fortune_every"]), float(cfg["fortune_fee"])
        else:
            every, price = int(cfg["donation_every"]), float(cfg["donation_amount"])
        buyers = [o for o in fresh_crowd
                  if (int(o.id) + step) % every == 0 and float(o.money) >= price]
        for o in fresh_crowd:
            seen.add(int(o.id))
        total = 0.0
        for o in buyers:
            o.money = float(o.money) - price
            total += price
            _notice(o, NOTICE_TEXT[occ])
            if occ != FUNDRAISER:                  # 売上は売り手へ(街の中で保存)
                a.money = float(a.money) + price
        st["notices"] += len(buyers)
        if not fresh:
            return
        payload = {"node": str(a.node), "amount": round(total, 1)}
        if occ == KITCHEN:
            payload["sold"] = len(buyers)
            kind = "stall_sale"
        elif occ == FORTUNE:
            payload["clients"] = len(buyers)
            kind = "fortune_reading"
        else:
            payload["donors"] = len(buyers)
            kind = "donation"
            # 街頭募金は**街の外の団体**へ出る = SFC ON では RoW(取引の相手方)。
            # OFF ではキー自体を出さない(既存の payee 規約と同じ流儀)。
            from . import economy_sfc as sfc_mod
            if sfc_mod.enabled(sim) and total > 0.0:
                payload["payee"] = sfc_mod.row_out(sim, "donation", total)
        bud.log(Event(step=step, sim_min=sim_min, agent_id=int(a.id),
                      kind=kind, x=a.x, y=a.y, payload=payload))
        st["sessions"] += 1
        return


# --------------------------------------------------------------------------- #
# 客引き × 条例パトロール(いたちごっこ)
# --------------------------------------------------------------------------- #
def _gov_on(sim):
    """行政が実体化していて有効なら Government を返す(**構築はしない**=副作用ゼロ)。"""
    gov = getattr(sim, "government", None)
    if gov is None:
        return None
    try:
        return gov if bool(gov.cfg["enabled"]) else None
    except Exception:                              # noqa: BLE001(旧 config 互換)
        return None


def _tout_session(sim, cfg: dict, st: dict, bud: _Budget, a,
                  step: int, sim_min: int) -> None:
    """客引き 1 人の 1 step(セッション開始で ``touting`` を 1 件)。"""
    day = int(sim_min) // 1440
    key = _session_key(day, 9, str(a.node))        # 帯 index 9 = 客引き専用の名前空間
    fresh = getattr(a, "street_session", None) != key
    if fresh:
        a.street_session = key
        a.street_seen = set()
    seen = getattr(a, "street_seen", None)
    if seen is None:
        seen = set()
        a.street_seen = seen
    cap = max(0, int(cfg["notice_cap"]) - len(seen))
    crowd = _passersby(sim, a, float(cfg["notice_radius_m"]), cap) if cap else []
    fresh_crowd = [o for o in crowd if int(o.id) not in seen]
    every = int(cfg["tout_accept_every"])
    accepted = [o for o in fresh_crowd if (int(o.id) + step) % every == 0]
    for o in fresh_crowd:
        seen.add(int(o.id))
    for o in accepted:
        _notice(o, TOUT_TEXT)                      # ★行き先は書き換えない(中立提示)
    st["notices"] += len(accepted)
    if fresh:
        bud.log(Event(step=step, sim_min=sim_min, agent_id=int(a.id),
                      kind="touting", x=a.x, y=a.y,
                      payload={"node": str(a.node), "targets": len(fresh_crowd),
                               "accepted": len(accepted),
                               "in_zone": bool(in_zone(sim, a.node))}))
        st["sessions"] += 1


def _patrol(sim, cfg: dict, st: dict, bud: _Budget, a, officer,
            step: int, sim_min: int) -> None:
    """条例パトロール: 警告 → 再犯で過料 5 万円 → クールダウン(決定論)。

    罰金の作法は ``scheduler._phase_enforcement`` を**そのまま踏襲**する
    (所持金の範囲で徴収・行政 ON なら区の歳入・SFC ON なら payee トークン)。
    """
    n = int(getattr(a, "tout_warnings", 0)) + 1
    a.tout_warnings = n
    a.street_cooldown_until = int(step) + int(cfg["cooldown_steps"])
    a.street_session = None                        # セッションを切る(次は仕切り直し)
    if n <= int(cfg["warn_before_fine"]):
        bud.log(Event(step=step, sim_min=sim_min, agent_id=int(officer.id),
                      kind="touting_warning", x=a.x, y=a.y,
                      payload={"officer": int(officer.id), "target": int(a.id),
                               "node": str(a.node), "count": n}))
        st["warnings"] += 1
        a.remember(TOUT_WARN_TEXT)
        return
    fine = float(cfg["fine_amount"])
    penalty = min(float(a.money), fine)
    a.money = max(0.0, float(a.money) - fine)
    gov = _gov_on(sim)
    if gov is not None and penalty > 0:
        gov.collect("ward", penalty)
    payload = {"officer": int(officer.id), "target": int(a.id), "node": str(a.node),
               "amount": round(penalty, 1)}
    from . import economy_sfc as sfc_mod
    if sfc_mod.enabled(sim) and penalty > 0:
        payload["payee"] = ("government" if gov is not None else
                            sfc_mod.row_out(sim, "fine_no_authority", penalty))
    bud.log(Event(step=step, sim_min=sim_min, agent_id=int(officer.id),
                  kind="touting_fine", x=a.x, y=a.y, payload=payload))
    st["fines"] += 1
    st["fine_total"] += float(penalty)
    griev = float(cfg["fine_grievance"])
    if griev > 0.0:
        from .factors import update as factor_update
        factor_update.on_enforcement(a, griev, step=step, sim_min=sim_min,
                                     logger=sim.logger)
    a.remember(TOUT_FINE_TEXT)


def _disperse(sim, cfg: dict, st: dict, bud: _Budget, a, officer,
              step: int, sim_min: int) -> None:
    """警察官の接近で持ち場を次へ移す(黙認と退去の循環。決定論・乱数ゼロ)。"""
    spec = ROLE_SPECS.get(str(getattr(a, "occupation", "")))
    pool = posts(sim, spec["post"]) if spec else []
    if len(pool) < 2:
        return
    idx = (int(getattr(a, "street_post_idx", 0)) + 1) % len(pool)
    old = str(getattr(a, "street_post", "") or a.node)
    a.street_post_idx = idx
    a.street_post = str(pool[idx])
    a.work_node = str(pool[idx])
    a.street_cooldown_until = int(step) + int(cfg["cooldown_steps"])
    a.street_session = None
    bud.log(Event(step=step, sim_min=sim_min, agent_id=int(a.id),
                  kind="street_disperse", x=a.x, y=a.y,
                  payload={"role": str(a.occupation), "from": old,
                           "to": str(pool[idx]), "officer": int(officer.id)}))
    st["disperses"] += 1


# --------------------------------------------------------------------------- #
# 路上生活者の支援(尊厳規約 4/5)
# --------------------------------------------------------------------------- #
def _outreach(sim, cfg: dict, st: dict, bud: _Budget, worker,
              step: int, sim_min: int) -> None:
    """区の支援員の巡回相談(近くに居る路上生活者へ 1 人 1 日 1 回まで・決定論)。

    ★同ノードではなく ``outreach_radius_m`` の近さで判定する: 夜の居場所は「公園のそば」
      「地下通路」という**範囲**であって 1 点ではないので、ノード完全一致にすると
      相談が原理的にほとんど成立しない(実測で確認した = DEFAULTS の注記)。
    """
    day = int(sim_min) // 1440
    period = max(1, int(cfg["outreach_period_days"]))
    if day % period != 0:
        return
    r = float(cfg["outreach_radius_m"])
    r2 = r * r
    for o in sorted(sim.agents, key=lambda z: int(z.id)):
        if str(getattr(o, "occupation", "")) != ROUGH:
            continue
        if o.loc == "outside" or o.sleeping:
            continue
        if str(o.node) != str(worker.node):
            dx, dy = float(o.x) - float(worker.x), float(o.y) - float(worker.y)
            if dx * dx + dy * dy > r2:
                continue
        if int(getattr(o, "outreach_day", -1)) == day:
            continue
        o.outreach_day = day
        n = int(getattr(o, "outreach_count", 0)) + 1
        o.outreach_count = n
        bud.log(Event(step=step, sim_min=sim_min, agent_id=int(worker.id),
                      kind="outreach_contact", x=o.x, y=o.y,
                      payload={"worker": int(worker.id), "target": int(o.id),
                               "node": str(o.node), "count": n}))
        st["contacts"] += 1
        o.remember(OUTREACH_TEXT)
        worker.remember(OUTREACH_WORKER_TEXT)
        after = int(cfg["outreach_shelter_after"])
        if after > 0 and n >= after:
            _shelter(sim, st, bud, o, n, step, sim_min)


def _shelter(sim, st: dict, bud: _Budget, agent, contacts: int,
             step: int, sim_min: int) -> None:
    """住居への移行(**既存の住居状態欄をそのまま書く**= mobility の関数は呼ばない)。

    住宅建物は ``city.residential_buildings`` から id 昇順の決定論選択(乱数ゼロ)。
    以後は既存の ``go_to_bed`` が建物へ入って眠らせる = 特別扱いが 1 つも残らない。
    """
    blds = getattr(sim.city, "residential_buildings", None) or []
    if not blds:
        return
    bld = blds[int(agent.id) % len(blds)]
    agent.home_building = str(bld["id"])
    agent.home_node = str(bld["entrance"])
    agent.home_floor = 1
    agent.street_sleep_node = ""
    bud.log(Event(step=step, sim_min=sim_min, agent_id=int(agent.id),
                  kind="shelter_move", x=agent.x, y=agent.y,
                  payload={"target": int(agent.id), "building": str(bld["id"]),
                           "contacts": int(contacts)}))
    st["shelters"] += 1
    agent.remember(SHELTER_TEXT)


def _weather_shift(sim, cfg: dict) -> None:
    """雨/雪の日は屋根のある場所(地下通路)へ夜の居場所を移す(既存の天気を読むだけ)。

    晴れの日は元の居場所へ戻す。乱数ゼロ・L1 に 1 件も出さない(位置の帰結は
    既存の帰宅/就寝イベントに現れる)。
    """
    covered = posts(sim, P_COVERED)
    rain = _rain(sim)
    for a in sim.agents:
        if str(getattr(a, "occupation", "")) != ROUGH:
            continue
        base = str(getattr(a, "street_sleep_node", "") or "")
        if not base:
            continue                               # 住居へ移行済み = 触らない
        if rain and covered:
            a.home_node = str(covered[int(a.id) % len(covered)])
        else:
            a.home_node = base
        a.home_building = ""


# --------------------------------------------------------------------------- #
# 単一作用点(scheduler の唯一のフック。位置確定後に呼ぶ)
# --------------------------------------------------------------------------- #
def phase(sim, step: int, sim_min: int) -> None:
    """毎 step: 路上の生業 + 条例パトロール + 支援。**既定 OFF は即 return**。

    ★世界に対して触るのは (a) L1、(b) 記憶 1 行、(c) 金の移転(買い手↔売り手・過料)、
      (d) 持ち場と夜の居場所の欄 だけ。drive・opinion・関係・k のどれにも触らない。
    ★乱数を 1 本も引かない(全部 id・step・sim_min・地図の純関数)。
    """
    if not enabled(sim):
        return
    cfg = cfg_of(sim)
    st = _state(sim)
    if not st["bound"]:
        st["bound"] = True
        st["bind"] = bind(sim)
    _weather_shift(sim, cfg)
    bud = _Budget(sim, st, int(cfg["max_events_per_step"]))

    # 近傍(同ノード)の警察官(最小 id で先勝ち)= 執行・退去の判定材料。
    # ★路上の役割・客引きが 1 人も立っていない step では走査ごと省く(25 万体対策)。
    actors = []
    tout_band = _in_band((tuple(cfg["tout_bands"]),), sim_min) >= 0
    for a in sim.agents:
        occ = str(getattr(a, "occupation", ""))
        if a.loc == "outside" or a.sleeping or a.building:
            continue
        if int(getattr(a, "street_cooldown_until", -1)) > int(step):
            continue
        spec = ROLE_SPECS.get(occ)
        if spec is not None and spec["bands"]:
            bi = _in_band(spec["bands"], sim_min)
            # 持ち場に立つ役割は持ち場でだけ働く。roam 役割(巡回相談)は道中でも働く。
            at_post = (spec.get("roam", False)
                       or str(a.node) == str(getattr(a, "street_post", "")))
            if bi >= 0 and at_post:
                actors.append((a, occ, bi, spec["bands"][bi]))
        elif (tout_band and getattr(a, "street_tout", False)
                and str(a.node) == str(getattr(a, "work_node", "") or "\0")
                and in_zone(sim, a.node)):
            # ★客引きは**店の前**でしか立たない(職場ノード = 雇い主の店のあるノード)。
            #   街のどこでも湧く「無主体の迷惑行為」にしない = diversity の nuisance との違い。
            actors.append((a, "", -1, None))
    officer_at: dict[str, object] = {}
    if actors:
        for a in sim.agents:
            if (str(getattr(a, "occupation", "")) in POLICE_OCCS
                    and a.loc != "outside" and not a.sleeping
                    and str(a.node) not in officer_at):
                officer_at[str(a.node)] = a

    for a, occ, bi, band in sorted(actors, key=lambda t: int(t[0].id)):
        officer = officer_at.get(str(a.node))
        if occ == "":                              # 客引き
            if officer is not None:
                _patrol(sim, cfg, st, bud, a, officer, step, sim_min)
                continue
            _tout_session(sim, cfg, st, bud, a, step, sim_min)
            continue
        if officer is not None and occ in DISPERSIBLE:
            _disperse(sim, cfg, st, bud, a, officer, step, sim_min)
            continue
        if occ == OUTREACH:
            _outreach(sim, cfg, st, bud, a, step, sim_min)
            continue
        _run_session(sim, cfg, st, bud, a, occ, bi, band, step, sim_min)


# --------------------------------------------------------------------------- #
# 観測記録(summary.json の "street_life" キー。**OFF ではキー自体を出さない**)
# --------------------------------------------------------------------------- #
def provenance(sim) -> dict | None:
    """``summary.json`` の ``street_life`` キー(既定 OFF は None = キー自体を出さない)。"""
    if not enabled(sim):
        return None
    cfg = cfg_of(sim)
    out: dict = {"schema": SCHEMA, "roles": list(STREET_OCCS),
                 "zone_radius_m": float(cfg["zone_radius_m"]),
                 "fine_amount": float(cfg["fine_amount"])}
    st = getattr(sim, "_sl_state", None)
    if st is None:
        out.update({"sessions": 0, "notices": 0, "warnings": 0, "fines": 0,
                    "disperses": 0, "contacts": 0, "shelters": 0,
                    "dropped": 0, "by_kind": {}})
        return out
    out.update({"sessions": int(st["sessions"]), "notices": int(st["notices"]),
                "warnings": int(st["warnings"]), "fines": int(st["fines"]),
                "fine_total": round(float(st["fine_total"]), 1),
                "disperses": int(st["disperses"]), "contacts": int(st["contacts"]),
                "shelters": int(st["shelters"]), "dropped": int(st["dropped"]),
                "by_kind": {k: int(v) for k, v in sorted(st["by_kind"].items())},
                "bind": dict(st.get("bind") or {})})
    return out
