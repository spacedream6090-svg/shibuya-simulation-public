"""因果台帳 — 「その出来事は**何が**起こしたのか」の唯一の分類源(第100バッチ IF-F)。

正典
----
- docs/research/llm-world-interface-audit.md §2(誰が何を起こしたかが L1 から復元できない)
- docs/research/if-lane-research.md §4(W3C PROV-DM の ``wasAssociatedWith`` / ``wasAttributedTo``)
- 本レーンの設計文書「世界イベントに cause_type + actor_id を持たせ、**実測の帰属率**を
  移行の進捗指標にする」

何を解く問題か
--------------
L1 の 194 種は **step / agent_id / kind / payload** しか持たない。``agent_id`` は
「この行の主語」ではなく「**この行が誰の視点で記録されたか**」であり、種によって
意味が違う:

  - ``speak`` … agent_id = 話した本人(= 行為者)
  - ``hear``  … agent_id = **聞いた側**(行為者は payload["speaker"])
  - ``wage``  … agent_id = 受け取った本人(行為者は雇用制度 = 誰でもない)
  - ``weather`` … agent_id = -1(そもそも行為者が居ない)

この 4 つが同じ 1 列に同居しているため、「エージェントが起こした変化」と
「装置が起こした変化」を事後に分離できない。分離できないと
**『世界を変えた人』の測定そのものが成立しない**(agent_id を主語として数えると
賃金の支給が「その人の起こした変化」に化ける)。

本 module は**分類そのものを 1 箇所に集める**。エンジン側(logger の刻印)も
事後解析側(``scripts/analyze_causality.py``)も、この表だけを読む。二重定義を作らない。

語彙(6 語)と設計文書の 4 分類への射影
--------------------------------------
設計文書の 4 分類は **agent / device / natural / boundary** である。本 module は
device をさらに 3 つに割って 6 語で持ち、``PROJECTION_4WAY`` で 4 語へ畳む:

  ============ ============ ==================================================
  本 module    4 分類        意味
  ============ ============ ==================================================
  ``agent``    agent        ある個体が**選んで行った**ことの帰結
  ``device``   device       制度・自動処理・ルールエンジンが世界状態に反応して発火
  ``schedule`` device       暦・時計だけで決まる制度(DEVS の δ_int = 外部入力なしの
                            内部遷移。年中行事・大型連休の群集)
  ``physics``  device       身体・空間・内部状態の連続力学(これも外部入力なしの
                            δ_int だが、駆動するのは暦ではなく**状態**)
  ``natural``  natural      外生(天気・災害)。世界の外から来る入力
  ``boundary`` boundary     **観測の枠の外**の統計・実験者の介入・名簿の配布
  ============ ============ ==================================================

``schedule`` と ``physics`` を device の下位に置くのは DEVS の内部遷移関数
δ_int の分解にあたる(「時計で進む」と「状態で進む」)。4 分類だけだと
「1 日 1 回必ず出る暦イベント」と「毎 step 出る身体更新」が同じ箱に入り、
帰属率の分母が tick で埋まって指標が意味を失う。だから 6 語で持ち、
集計側では ``physics`` / ``schedule`` を**明示的に除いた**集計を併記する。

分類の原則(迷ったときのタイブレーク)
--------------------------------------
1. **agent** = その step に**ある個体が選択した行為**が無ければこの行は存在しない。
2. **device** = 個体の行動が作った世界状態に**ルールが反応**して発火した
   (個体が選んだのは状態であって、この出来事ではない)。
3. 同じ kind が両方から出る場合は **device 側に寄せる**(agent と偽ると帰属率が
   過大評価される。少なく見積もる方向へ倒す)。
4. ``unknown`` クラスは**作らない**。分からないものは最も近い語を選び、
   直下にその理由を 1 行で書く(後から検証できる形で残す)。

正直な限界
----------
- 分類は **kind 単位**である。``deposit`` のように phase によって行為者が変わる種は
  1 語に畳むほかない(payload の値で分岐すると表が payload スキーマに依存し、
  過去ランで壊れる)。畳んだ種は下の表のコメントに明記した。
- ``actor_of`` が返すのは「**この出来事に結びつく最良の個体 id**」であって、
  道義的な責任者ではない。``rule_expired`` の agent_id はルールの提案者だが、
  失効させたのは期限であってその人ではない(cause_type=device がそれを言っている)。
- 本 module は**読むだけ**の純関数の集まりで、乱数も LLM も世界状態も触らない。
"""
from __future__ import annotations

from typing import Any, Callable

# --------------------------------------------------------------------------- #
# 語彙
# --------------------------------------------------------------------------- #
AGENT = "agent"
DEVICE = "device"
SCHEDULE = "schedule"
PHYSICS = "physics"
NATURAL = "natural"
BOUNDARY = "boundary"

#: 本 module が使う 6 語(この順序で表・レポートを並べる)。
CAUSE_TYPES: tuple[str, ...] = (AGENT, DEVICE, SCHEDULE, PHYSICS, NATURAL, BOUNDARY)

#: 設計文書の 4 分類への射影(schedule / physics は device の下位 = DEVS δ_int)。
PROJECTION_4WAY: dict[str, str] = {
    AGENT: AGENT, DEVICE: DEVICE, SCHEDULE: DEVICE, PHYSICS: DEVICE,
    NATURAL: NATURAL, BOUNDARY: BOUNDARY,
}

#: 帰属率の分母から外す語。毎 step / 毎日必ず出る tick 系で、ここを混ぜると
#: 「エージェント帰属率」が個体数と step 数の関数になってしまい指標として死ぬ。
TICK_CAUSES: frozenset[str] = frozenset({PHYSICS, SCHEDULE})

#: **装置スコープが ``device_id`` を刻んでよい cause_type**(IF-F W2)。
#:
#: 装置スコープ(``society/devices.cause_scope``)は「いまこの装置の処理を回している」
#: という窓でしかない。窓の中で出た行が全てその装置の作った出来事だとは限らないので、
#: 刻んでよい語を**明示的に閉じる**:
#:
#:   - ``device`` / ``schedule`` … その装置の処理そのもの(刻む)
#:   - ``boundary``              … その装置が出した**観測要約**(device_load /
#:                                 signal_summary / public_budget 系。世界を動かさない
#:                                 記録だが、出したのは確かにその装置なので刻む)
#:   - ``agent``   … **別の主体の行為**。装置の窓の中で出ても行為者は個体である(刻まない)
#:   - ``physics`` … 個体の身体・空間の内部力学(grievance の state_update 等)。
#:                   装置の処理が引き金でも、その出来事を作ったのは身体である(刻まない)
#:   - ``natural`` … 世界の外から来た入力(災害そのもの)。装置は起こしていない(刻まない)
#:
#: ★この線引きが無いと、交通事業者スコープの中で出る grievance の ``state_update`` まで
#:   「事業者が起こした出来事」に化ける(実測: 遅延 1 回で街内人数ぶん出る)。
DEVICE_STAMPABLE: frozenset[str] = frozenset({DEVICE, SCHEDULE, BOUNDARY})


# --------------------------------------------------------------------------- #
# kind → cause_type(**register_event_kind された全種を網羅する**。件数は書かない
#   = 種が増えるたびに直す嘘の数字を作らない。抜け・余りはどちらも
#   tests/test_causality.py の網羅テストが落とす。数え方は observer/schema.py の
#   EVENT_KINDS を **society.* を全 import してから**見る = 材料側登録
#   (src/society/devices.py 等)も必ず入る)
# --------------------------------------------------------------------------- #
CAUSE_OF_KIND: dict[str, str] = {
    # ---- 移動・在場(行為者自身の身体の移動)----------------------------------- #
    "route_start":   AGENT,
    "move_segment":  AGENT,
    "arrive":        AGENT,
    "stay":          AGENT,
    "enter_building": AGENT,
    "exit_building": AGENT,
    "floor_move":    AGENT,
    "enter_area":    AGENT,
    "exit_area":     AGENT,
    "detour":        AGENT,          # 寄り道 = 経路の選び直し(確率的実行 L3)
    "interrupt":     AGENT,          # 活動の途中離脱
    "boredom_explore": AGENT,        # 退屈ドライブの内発的探索(行き先は個体が選ぶ)
    "tourist_visit": AGENT,
    "ride":          AGENT,
    "taxi_unmatched": DEVICE,        # 配車装置が捕まえられなかった(個体は乗ろうとしただけ)
    "sleep_start":   AGENT,          # 自宅へ帰って就寝する = 選択の帰結
    "wake_up":       PHYSICS,        # 起床は概日リズム側(選べない)
    "space_move":    PHYSICS,        # 屋内区画遷移 = markov 回遊(個体の選択ではない)
    "zone_gate":     PHYSICS,        # 物理ゾーンの境界通過(SFM の積分結果)

    # ---- 発話・情報(行為は話者側。聞き手側は PATIENT_ACTOR_OVERRIDES で戻す)---- #
    "speak":         AGENT,
    "hear":          AGENT,          # 行為者 = payload["speaker"](agent_id は聞き手)
    "dm":            AGENT,
    "conversation":  AGENT,          # 構造化会話 C2(1 会話 1 件・agent_id = 開始側)
    "sns_post":      AGENT,
    "sns_read":      AGENT,
    "sns_like":      AGENT,
    "sns_reshare":   AGENT,
    "news_read":     AGENT,
    "search":        AGENT,
    "media_use":     AGENT,
    "transmission":  AGENT,          # 行為者 = payload["from"](agent_id は受け手)
    "opinion_shift": AGENT,          # 行為者 = payload["source"](agent_id は動かされた側)
    "label_coin":    AGENT,
    "label_adopt":   AGENT,          # 聞き手がその語を自分のものにした = 個体の採用
    "vocab_coin":    AGENT,
    "vocab_use":     AGENT,
    "place_label_bind": AGENT,       # 造語を発生ノードへ束縛(最初の coin のみ)
    "appointment":   AGENT,          # 会話からの予定記入(agent_id = 言い出した側)
    "appointment_kept": DEVICE,      # 遵守判定は事後の自動照合(個体は何もしていない)

    # ---- 思考・内面(LLM 呼とその後始末)-------------------------------------- #
    "llm_deliberate": AGENT,
    "reflect":       AGENT,
    "memory_recall": AGENT,          # 内省の agentic pull = 個体が思い出そうとした
    "memory_fail":   PHYSICS,        # 活性化が閾値未達 = 記憶の減衰力学(選べない)
    "day_plan":      AGENT,
    "plan_created":  AGENT,          # LLM が出した計画そのもの
    "day_plan_compiled": DEVICE,     # 型スキーマ → アンカー先置き → 充填(決定論のコンパイラ)
    "plan_repair":   DEVICE,         # 決定的ルールによる修復
    "plan_fallback": DEVICE,         # 修復不能 → 前日/骨組みへ後退
    "plan_block_start": DEVICE,      # ルールエンジンが計画ブロックを無料で実行
    "plan_block_drop": DEVICE,
    "plan_slide":    DEVICE,
    "plan_replan":   DEVICE,
    "plan_cont_fire": DEVICE,        # contingency の決定論評価
    "policy_reuse":  DEVICE,         # 方針キャッシュの再利用(LLM を撃たない側の判断)
    "fallback":      AGENT,          # LLM 出力が使えなかった = その個体の思考の記録
    "undefined_action": AGENT,       # enum 外の行動主張 = 行動空間の外へ出ようとした痕跡
    "watch_spec":    AGENT,          # 監視仕様の受理結果(読めなかったことも含めて個体の出力)
    "llm_null":      BOUNDARY,       # null 系列対照 = 実験者が撃つダミー呼(世界に作用しない)
    "cog_fire":      PHYSICS,        # 認知イベントの期限到来(θ の内部力学)
    "cog_event":     PHYSICS,
    "cog_theta":     PHYSICS,        # θ の恒常性(日境界で 1 回)
    "episode_start": PHYSICS,        # ENGAGED 状態機械の遷移(状態で進む δ_int)
    "episode_end":   PHYSICS,
    "episode_closing": AGENT,        # 別れ挨拶は**発話**なので行為
    "engaged_template": AGENT,       # 定型でもその個体の会話ターン(LLM 呼はゼロ)
    "sign_memory":   PHYSICS,        # 中断内容の兆しメモリ(状態の書き込み)
    "reflection_trigger": PHYSICS,   # 日内衝撃ゲージの閾値超え
    "drive_request": PHYSICS,        # 欲求ゲージの発火申請(身体側の要求)
    "state_update":  PHYSICS,
    "affect_update": PHYSICS,
    "emotion_label": PHYSICS,
    "belief_update": PHYSICS,        # 信念の更新(伝聞/検証の**結果**としての内部状態)
    "belief_transmit": AGENT,        # 伝聞の送出 = 発話に乗せた行為
    "belief_verify": AGENT,          # 検証行動(現場go/当事者ask/ネットnet)
    "worldview":     PHYSICS,        # 主観的世界モデルの日次スナップ
    "long_goal":     PHYSICS,        # 長期目標の設定/変化(内面の日次力学)
    "pov_image":     BOUNDARY,       # 顕著時の POV 撮影 = 観測装置(世界状態を動かさない)

    # ---- 経済(制度が動かす金。個体が選ぶのは「使うこと」だけ)------------------ #
    "spend":         AGENT,
    "wage":          DEVICE,
    "withdraw":      DEVICE,         # 現金不足時の自動引き出し(ATM 近似)
    "rent":          DEVICE,
    "tax":           DEVICE,
    "interest_paid": DEVICE,
    "loan_grant":    DEVICE,
    "loan_repay":    DEVICE,
    "vc_investment": DEVICE,
    "deposit":       DEVICE,         # 供託の拠出/返還/没収。phase=paid だけは個体発だが
                                     #  種としては行政の預り金機構なので device へ畳む(原則3)
    "candidacy":     AGENT,          # 自発的立候補(供託金を払うのは本人の選択)
    "reward":        DEVICE,         # D9 ablation の採用報酬(自動支給)
    "rule_bonus":    DEVICE,
    "civic_service": DEVICE,
    "public_budget": DEVICE,
    "bankruptcy":    DEVICE,
    "eviction":      DEVICE,
    "org_output":    DEVICE,
    "org_overdraft": DEVICE,
    "row_flow":      BOUNDARY,       # 域外(rest-of-world)との日次収支 = 枠の外の統計
    "production":    AGENT,          # 職場での産出(勤務している個体の労働)
    "study":         AGENT,
    "serve":         DEVICE,         # 客の消費に応対を帰属させる自動処理(不在なら agent_id=-1)
    "service_use":   AGENT,          # 受給側は個体の選択
    "order":         AGENT,
    "deliver":       AGENT,          # 配達員が届けた(agent_id = 配達員 = payload["courier"])
    "delivery_trip": DEVICE,
    "restock":       DEVICE,
    "stock_low":     DEVICE,
    "stock_out":     DEVICE,
    "b2b_trade":     DEVICE,
    "price_change":  DEVICE,
    "shop_state":    DEVICE,
    "venture_open":  AGENT,
    "venture_sale":  DEVICE,         # 通行人の到着で自動発生する売上(店主は選んでいない)
    "venture_close": DEVICE,         # 売上ゼロ継続による自動閉店
    "venture_permit": DEVICE,
    "venture_fulltime": DEVICE,      # 売上条件を満たした自動の起業転換
    "job_search":    AGENT,          # 求職 tool の発火(career 選択由来)
    "job_change":    DEVICE,         # 決定論マッチの結果としての配属変更
    "unemployment":  DEVICE,
    "labor_action":  AGENT,
    "medical_visit": AGENT,
    "lodging_checkin": AGENT,
    "lodging_checkout": DEVICE,      # 滞在終了は日数で自動
    "move_home":     AGENT,          # 生活の自己決定 P2(敷金を払って移る)
    "relocate":      DEVICE,         # 内生的な世帯転居(通勤逼迫/家賃逼迫のルール)
    "move_in":       DEVICE,
    "move_out":      DEVICE,

    # ---- 制度・統治 ------------------------------------------------------------ #
    "proposal":      AGENT,
    "proposal_support": AGENT,
    "proposal_passed": DEVICE,       # 署名が閾値へ到達したことによる自動成立
    "proposal_review": DEVICE,
    "vote_cast":     AGENT,
    "vote_result":   DEVICE,
    "ordinance_vote": DEVICE,
    "council_budget": DEVICE,
    "council_elected": BOUNDARY,     # 意見最近傍の**決定論選挙**(誰も投票していない)
    "election_result": BOUNDARY,     # 開票結果(名簿制。投票行動を伴わない条件がある)
    "institution":   DEVICE,
    "institution_rule": DEVICE,
    "rule_expired":  DEVICE,
    "rule_repealed": DEVICE,
    "rule_weekly_fire": DEVICE,
    "norm_digest":   BOUNDARY,       # 昨日の街の客観カウント = 枠の外の統計
    "enforcement":   AGENT,          # 執行するのは公務員(= 個体)
    "detention":     AGENT,          # 同上(agent_id = officer)
    "action_reject": DEVICE,         # 裁定が行為を拒否した = ルール側の応答

    # ---- 社会関係 -------------------------------------------------------------- #
    "relation_tier": DEVICE,         # 親密度の閾値到達
    "relation_break": DEVICE,
    "relation_dormant": DEVICE,      # 認知枠(Dunbar)の上限超過
    "relation_rekindle": DEVICE,
    "reputation_update": DEVICE,
    "partner_formed": DEVICE,        # 相互の親密度が閾値に達したことによる自動成立
    "partnership_declined": AGENT,   # 申し込んだ個体の行為の帰結(不成立の通知)
    "life_event":    DEVICE,
    "joint_activity": DEVICE,        # 同席の**検出**(誘い自体は joint_invite)
    "joint_invite":  AGENT,
    "group_found":   AGENT,
    "group_join":    AGENT,
    "event_host":    AGENT,
    "event_attend":  AGENT,
    "flyer_post":    AGENT,
    "flyer_view":    AGENT,
    "flyer_expire":  DEVICE,
    "gossip_seed":   DEVICE,         # 負イベント当事者の悪評化(タグ付け機構)
    "gossip_spread": DEVICE,         # 閾値到達による採用(complex contagion)
    "gossip_fade":   DEVICE,
    "rumor_born":    DEVICE,         # 構造化イベントから噂 Item が生える(源は自由文を読まない)
    "rumor_stifle":  DEVICE,         # Maki-Thompson の stifler 化(語った回数の閾値)
    "trace_mark":    DEVICE,         # 行為の副産物としての痕跡集約
    "friend_graph_built": BOUNDARY,  # 起動時 1 回の初期グラフ生成統計

    # ---- 情報環境・広告 -------------------------------------------------------- #
    "feed_rank":     DEVICE,
    "viral_cascade": DEVICE,
    "misinfo":       DEVICE,
    "ad_campaign":   BOUNDARY,       # 枠 × 周期の改定(街の設定であって出来事ではない)
    "ad_exposure":   DEVICE,

    # ---- 身体・健康 ------------------------------------------------------------ #
    "illness":       PHYSICS,
    "health_update": PHYSICS,

    # ---- 都市・環境・自然 ------------------------------------------------------ #
    "weather":       NATURAL,
    "disaster":      NATURAL,        # ★自然事象は**災害そのもの**だけ(外生入力)
    # ★運休・遅延・障害の告知は「自然の続き」ではなく**事業者(装置)の判断**である
    #   (IF-F W2 で実体化した: 台風が線路を止めるのではなく、台風という外部入力を見た
    #    運行事業者が止めると決める = DEVS の δ_ext)。分類は W1 の時点で既に device
    #    だったので**表は 1 バイトも変えていない**。W2 で足したのは device_id 側の
    #    同一性(operator:transit / operator:infra)であって、語の付け替えではない。
    "transit_delay": DEVICE,         # 運行事業者の遅延/運休の判断(device_id=operator:transit)
    "infra_outage":  DEVICE,         # 供給事業者の障害告知(device_id=operator:infra)
    "env_feedback":  DEVICE,         # 集約物理量の閾値超過でコード側が発火
    "crowd_surge":   SCHEDULE,       # 暦(大型行事)で決まる群集
    "annual_event":  SCHEDULE,
    "crime":         AGENT,          # agent_id = 加害者(victim は payload)
    "nuisance":      AGENT,
    "traffic_flow":  BOUNDARY,       # 背景交通 = エージェントではない通過車両の統計
    "world_event":   BOUNDARY,       # シナリオイベント(実験者の注入)
    "scenario_shock": BOUNDARY,
    "chance_event":  BOUNDARY,       # 臨時収入/紛失 = 街の外から来る金(RoW)。
                                     #  ★ユーザー判断待ち: 「偶然の出会い(encounter)」だけは
                                     #    個体間の出来事なので agent とも読める。kind を割らずに
                                     #    1 語へ畳む方針を優先し、現状は boundary で固定してある。
    "presence_change": BOUNDARY,     # 日次ローテーションの在場入替(名簿の配布)
    "workplace_bound": BOUNDARY,     # 起動時の職場束ね直し統計
    "mind_assign":   BOUNDARY,       # 誕生時のモデル固定(実験条件の割り当て)
    "spark_roster":  BOUNDARY,       # 火種介入 t=0 名簿(実験者の介入)
    "free_action":   AGENT,

    # ---- 装置(改札・歩行者信号)------------------------------------------------ #
    # ★これらは **材料側 registration**(`src/society/devices.py` が自分の module で
    #   register_event_kind する流儀)。observer/schema.py を読むだけでは見つからないので、
    #   網羅テストは `society.*` を全 import してから EVENT_KINDS を数えている。
    "gate_pass":     DEVICE,         # 改札装置が待たせた通過(待ちが出た回だけ記録)
    "device_load":   BOUNDARY,       # 装置 1 台 1 時間の負荷**要約**(世界を動かさない観測)
    "signal_summary": BOUNDARY,      # 信号の周期パラメータの開示(同上)

    # ---- 鉄道の当直(駅員・車掌)。材料側 registration = src/society/transit_staff.py ---- #
    "dwell_decision": DEVICE,        # ドア閉(停車時間延長)の判断。当直が居れば
                                     #  agent_id = 乗務員だが、**居ない回は agent_id=-1** で
                                     #  同じ kind が出る(payload.unstaffed)。判断そのものは
                                     #  ホーム負荷の決定論則なので原則3(迷ったら device)を適用。
                                     #  乗務員が居た回の行為者は actor_of が agent_id から戻す。
    "transit_staff_bound": BOUNDARY,  # 起動時 1 回の持ち場束ね統計(workplace_bound と同型)
}


# --------------------------------------------------------------------------- #
# agent_id が**受け手**である種 — 本当の行為者を payload から戻す
# --------------------------------------------------------------------------- #
def _from_key(key: str) -> Callable[[dict], Any]:
    """payload の 1 キーをそのまま行為者 id として読む取り出し器。"""
    def _get(payload: dict) -> Any:
        return (payload or {}).get(key)
    _get.payload_key = key            # テストと解析が「どのキーを見たか」を言えるように
    return _get


#: kind → 行為者を payload から取り出す関数(値は文字列キー or callable)。
#:
#: **ここに載る種は「agent_id が患者(patient)である」ことを確認済みのものだけ**。
#: 実装側の 1 行を読んで確かめた結果を下にそのまま残す(推測で足さない)。
#:
#:   hear           scheduler.py:3151  agent_id=hearer.id      / payload["speaker"]  ← 話者
#:   opinion_shift  scheduler.py:1346 + conversation.py:206
#:                                     agent_id=listener.id    / payload["source"]   ← 説得側
#:   transmission   observer/provenance.py:45
#:                                     agent_id=to_agent       / payload["from"]     ← 送り手
#:   deliver        delivery.py:398    agent_id=cid            / payload["courier"]
#:                  ★現在の実装では **両者は同一値**(cid をそのまま payload にも載せている)。
#:                    恒等でも表に載せるのは、配達員が居ない回に agent_id が -1 になる設計で
#:                    「行為者の正典は payload 側」と決めておくため(片方だけ変わっても壊れない)。
#:
#: **確認したが載せなかった種**(agent_id が既に行為者なので上書きしてはいけない):
#:   serve        scheduler.py:1912 agent_id=s.id(勤務中スタッフ)/ :1927 agent_id=-1(不在)。
#:                payload["customer"] は**客** = 患者。行為者は agent_id のままが正しい。
#:   enforcement  scheduler.py:3918/3979 agent_id=officer.id。payload["target"] は被執行者。
#:   detention    scheduler.py:4008     agent_id=officer.id。payload["target"] は被拘束者。
#:   venture_sale tools.py:1694        agent_id=owner(店主)。payload["buyer"] は買い手。
#:                売上そのものは通行人の到着で自動発生する(cause_type=device)ので、
#:                買い手を「行為者」に昇格させると device の定義と矛盾する。
#:   crime        diversity.py:308     agent_id=加害者。payload["victim"] は被害者。
#:   gossip_spread gossip.py:252       agent_id=採用した本人。payload["sources"] は
#:                **相異なる情報源の人数(int)**であって id 列ではない = 行為者は復元できない。
#:                欠測を偽の値で埋めないため、ここには載せない。
PATIENT_ACTOR_OVERRIDES: dict[str, Any] = {
    "hear":          "speaker",
    "opinion_shift": "source",
    "transmission":  "from",
    "deliver":       "courier",
}


def cause_of(kind: str) -> str:
    """kind → cause_type。**未登録なら KeyError**(黙って unknown を作らない)。"""
    try:
        return CAUSE_OF_KIND[kind]
    except KeyError:
        raise KeyError(
            f"未分類のイベント種類 '{kind}'。observer/causality.py の CAUSE_OF_KIND へ"
            " 1 行足すこと(分類の単一の源。unknown クラスは作らない)。") from None


def actor_of(kind: str, agent_id, payload: dict | None = None) -> int | None:
    """その出来事に結びつく**最良の行為者 id**。復元できなければ None。

    優先順位は ``PATIENT_ACTOR_OVERRIDES`` → ``agent_id``。負値(世界イベントの -1)は
    「行為者が居ない」の意味なので **None** に畳む(-1 を id として数えない)。
    """
    rule = PATIENT_ACTOR_OVERRIDES.get(kind)
    if rule is not None:
        raw = rule(payload) if callable(rule) else (payload or {}).get(rule)
        aid = as_actor_id(raw)
        if aid is not None:
            return aid
    return as_actor_id(agent_id)


def as_actor_id(raw) -> int | None:
    """id らしきものを非負 int へ。負値・None・非数は None(= 帰属先なし)。"""
    if raw is None or isinstance(raw, bool):
        return None
    try:
        v = int(raw)
    except (TypeError, ValueError):
        return None
    return v if v >= 0 else None


def summarize(counts: dict[str, int]) -> dict[str, int]:
    """{kind: 件数} → {cause_type: 件数}。未登録 kind は数えない(呼び出し側が別途警告する)。"""
    out = {c: 0 for c in CAUSE_TYPES}
    for kind, n in counts.items():
        c = CAUSE_OF_KIND.get(kind)
        if c is not None:
            out[c] += int(n)
    return out


# --------------------------------------------------------------------------- #
# エンジン側の配線(observer.causality.enabled。**既定 OFF**)
# --------------------------------------------------------------------------- #
def cfg_of_config(cfg) -> bool:
    """conf から実効値を 1 度だけ読む(旧 config 互換のため例外は False へ倒す)。"""
    try:
        block = (cfg.get("observer", None) or {}).get("causality", None)
        return bool(block and block.get("enabled", False))
    except Exception:                              # noqa: BLE001(旧 config 互換)
        return False


def apply_cfg(logger, on: bool) -> None:
    """1 つの出力層へ設定を配る。**OFF では属性を 1 つも触らない**。

    finalize.apply_cfg と同じ流儀(early return)。既定ランは ObserverLogger の
    初期値のまま = L1 parquet のスキーマも刻印経路もバイト一致であることが、
    この 1 行で構造的に保証される。
    """
    if not on:
        return
    logger.causality_on = True


def enabled(sim) -> bool:
    """observer.causality.enabled が実効か(初回のみ読んでキャッシュ)。

    キャッシュ属性 ``_causality_on`` は L1/L2/乱数・checkpoint のいずれにも出ない
    (provlink.enabled と同じ流儀)。
    """
    on = getattr(sim, "_causality_on", None)
    if on is None:
        on = cfg_of_config(getattr(sim, "cfg", None) or {})
        sim._causality_on = on
    return on
