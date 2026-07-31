"""イベントの正準スキーマ + 種類レジストリ(D12)。

新しいログ種類の追加 = `register_event_kind("kind名", "説明")` を1行呼ぶだけ。
固定列は変えない(詳細は payload に入れる)ので、過去データとの互換は常に保たれる。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

EVENT_KINDS: dict[str, str] = {}


def register_event_kind(kind: str, description: str) -> None:
    EVENT_KINDS[kind] = description


# ---- 既定のイベント種類(D12 で確定した最初のセット) ----
register_event_kind("route_start",   "目的地への経路探索を開始 {dest, path, dist_m}")
register_event_kind("move_segment",  "経路に沿った1stepの前進 {from_xy, to_xy, dist_m, edge}")
register_event_kind("arrive",        "目的地に到着 {node}")
register_event_kind("stay",          "滞在(その場に留まる){node}")
register_event_kind("speak",         "発話 {text, item_ids, hearers}")
register_event_kind("hear",          "聴取 {speaker, item_ids}")
register_event_kind("label_coin",    "ラベル/新語の創出 {item_id, text}")
register_event_kind("label_adopt",   "ラベルの採用(閾値到達){item_id}")
register_event_kind("vocab_coin",    "★新規・専門語彙の誕生(ユーザー指定ログ){item_id, text}")
register_event_kind("vocab_use",     "★語彙の使用(ユーザー指定ログ){item_id}")
register_event_kind("transmission",  "★伝播系譜: item が from→to へ伝わった(ユーザー指定ログ){item_id, from, channel}")
register_event_kind("reflect",       "ソロ内省(k の実装部位){mode, written_back, belief}")
register_event_kind("state_update",  "state 更新(OPEN#2 seam){name, old, new, cause}")
register_event_kind("fallback",      "LLM 失敗→routine 代替(D16){reason}")
register_event_kind("llm_deliberate","LLM 熟考が発火 {trigger}")
register_event_kind("enter_building","建物に入る {building, name, floor}")
register_event_kind("exit_building", "建物から出る {building}")
register_event_kind("floor_move",    "建物内で階を移動 {building, floor}")
register_event_kind("exit_area",     "シミュ範囲外へ退出 {gateway, via}")
register_event_kind("enter_area",    "範囲外から帰還 {gateway, via}")
register_event_kind("sleep_start",   "自宅で就寝(個別時刻){until_step, building}")
register_event_kind("wake_up",       "起床 {slept_steps}")
register_event_kind("traffic_flow",  "背景交通(通過車両)の1step分の軌跡 {n, total, segs}")
register_event_kind("sns_post",      "SNS に投稿 {text, items}")
register_event_kind("sns_read",      "SNS タイムラインを閲覧 {n_posts, authors}")
register_event_kind("news_read",     "ニュースアプリを閲覧 {titles}")
register_event_kind("dm",            "1対1メッセージ送信(受信側は transmission/hear 相当){to, text}")
register_event_kind("search",        "検索エンジンで調べる {query}")
register_event_kind("world_event",   "シナリオイベント発生(公式発表など){title, word}")
register_event_kind("scenario_shock", "摂動シナリオの発動・解除(#8){kind, at, phase, params}")
register_event_kind("drive_request", "欲求発火の申請(Phase A){drive, threshold, lottery, granted, reason}")
register_event_kind("wage",          "賃金の支給(本業の勤務完遂・バイトのシフト完遂・月給まとめ・日銭){amount, balance, to, account?, source?}")
register_event_kind("spend",         "消費(食事・買い物・nightlife・taxi・bus){amount, balance, cat, src?, account?}")
# ---- 口座(銀行)概念 E5(既定 OFF。ユーザー決定 2026-07-06)----
register_event_kind("withdraw",      "現金不足→口座から自動引き出し(ATM近似・移動なし){amount, cash, account}")
register_event_kind("rent",          "家賃の口座引き落とし(給料日翌日・不足は翌日繰越){amount, paid, carry, account, phase}")
# ---- 朝の一日計画 / 交通機関(ユーザー要望 2026-07-06)----
register_event_kind("day_plan",      "朝の一日計画(LLM。ルールベース行動の土台){n, plan}")
register_event_kind("ride",          "交通機関の乗車(タクシー/簡易バス/実ダイヤ静的表){mode, fare, from, to, wait_s?, ride_s?, delay_s?, shared?}")
# ---- SUMO ライブ連成タクシー v-Ride-1(transit_ride.live ON のみ・既定 OFF=0件。設計: docs/research/sumo-live-transit.md)----
register_event_kind("taxi_unmatched", "SUMO ライブ配車が捕まらない(未配車=満車/max_wait 超過→徒歩フォールバック){from, to}")
register_event_kind("free_action",   "開放行動(第17バッチ: LLM の自由記述行動+価値4軸の観測){what, category, tags, match, report, minutes, cost, dest, sat}")
# ---- 生活の自己決定 P2(D3 棚卸し。freedom.p2.*。既定 全 OFF)----
register_event_kind("move_home",       "住居移転(#6: 空き住戸へ転居。敷金=現金障壁){from, to, deposit}")
register_event_kind("partnership_declined", "交際の申込が不成立(#9: closeness が形成閾値未満){to, closeness}")
# ---- 主観的世界モデル(第20バッチ 2026-07-12: 期待・可制御性・規範予期の日次観測。既定 OFF)----
register_event_kind("worldview",     "主観的世界モデルの日次スナップ(agent: ctrl/期待規模/期待誤差、街: norm_rate){ctrl?, expect_n?, err_mean?, err_n?, norm_rate?, pioneer_1d?}")
# ---- 街頭広告 OOH(第18バッチ 2026-07-11: 街路の環境情報。既定 OFF)----
register_event_kind("ad_campaign",   "街頭広告キャンペーンの改定(枠×周期・世界イベント agent_id=-1){slot, campaign, target, cat, period}")
register_event_kind("ad_exposure",   "街頭広告の視認(接触→来店→購買ファネルの起点){campaign, target, slot, cat, n_seen}")
# ---- ツール(世界を変える affordance。中立提示・自然使用の観察。R1/R4 準拠) ----
register_event_kind("event_host",    "イベントを開催宣言 {event_id, title, node, start_step}")
register_event_kind("event_attend",  "イベントに参加(会場到着){event_id, host, title}")
register_event_kind("flyer_post",    "掲示・ビラを貼る {author, node, text, items}")
register_event_kind("flyer_view",    "掲示・ビラを閲覧(通行人・ユニーク){author, node}")
register_event_kind("flyer_expire",  "掲示・ビラの撤去(ttl 失効・上限超過){author, node, reason}")
register_event_kind("group_found",   "コミュニティを結成 {group_id, name, purpose, founder}")
register_event_kind("group_join",    "コミュニティに加入 {group_id, name, founder}")
register_event_kind("proposal",      "提案・署名運動を起こす {proposal_id, text}")
register_event_kind("proposal_support", "提案に賛同=署名(露出2回){proposal_id, author}")
register_event_kind("proposal_passed", "提案が成立(署名が閾値到達){proposal_id, text, supporters}")
register_event_kind("venture_open",  "出店(屋台を開業){name, offer, node, cost, balance}")
register_event_kind("venture_sale",  "出店の売上(通行人の購入){amount, balance, buyer}")
register_event_kind("venture_close", "出店の閉店(売上ゼロ継続){name, node}")
# ---- 行動心理プラグイン(既定 OFF。Searle 制度化)----
register_event_kind("institution",   "成立提案の制度化(status function 樹立。Searle){name, norm_text, created_by}")
# ---- 制度DSL(ホワイトリスト型ルールの自動制定。既定 ON。ユーザー構想 2026-07-06) ----
register_event_kind("institution_rule", "成立提案が実効ルールとして自動制定(制度DSL){rule_id, type, name, proposer, rule}")
register_event_kind("rule_bonus",       "制度DSL bonus ルールによる支給(park/event_attend/flyer_view){rule_id, behavior, amount, balance}")
register_event_kind("rule_expired",     "制度DSL ルールの失効(duration 期限切れ / max_active 上限超過){rule_id, type, reason}")
register_event_kind("rule_weekly_fire", "制度DSL weekly_event の期日発火(定期ニュース+場所ブースト){rule_id, title, place, node}")
# ---- 再帰性(第9バッチ 2026-07-07。規範・現状の監視→知覚→フィードバック→改変。既定 OFF) ----
register_event_kind("rule_repealed",    "再帰性: repeal 提案の成立による既存ルールの廃止 {rule_id, type, name, proposer, repealed_by}")
register_event_kind("norm_digest",      "再帰性: 昨日の街の動きの日次ダイジェスト(客観カウント){day, proposals, passed, enacted, repealed, enforced, active_rules}")
# ---- 実験プロトコル核(D7 対照 / D9 過正当化 / agentic pull。既定 off で現状不変) ----
register_event_kind("llm_null",      "null 系列対照: 内容非結合のダミー LLM 呼び出し(D7 R1){i}")
register_event_kind("memory_recall", "内省の agentic pull: 思い出したいことのクエリと件数 {query, n_hits}")
register_event_kind("reward",        "採用報酬(D9 過正当化 ablation。既定off){amount, balance}")
# ---- 意見力学(FJ #16)+ SNS 反応(#14)。非LLM・決定論(いいね/リシェアは新streamで抽選)----
register_event_kind("opinion_shift", "意見の更新(Friedkin-Johnsen。source が聞き手の意見を動かした){source, old, new}")
register_event_kind("sns_like",      "SNS 投稿へのいいね {post_id, author}")
register_event_kind("sns_reshare",   "SNS 投稿のリシェア(RT。フォロワーへ自然再配信){post_id, author}")
# ---- 第6バッチ(ユーザー要望 2026-07-06: 行政・税/娯楽メディア/職場・学校の実態)----
register_event_kind("tax",           "税の徴収(所得税・住民税・消費税など){tax, amount, to, base?, balance?}")
register_event_kind("civic_service", "行政サービスの利用・給付(区/都/国){service, level, amount?, detail?}")
register_event_kind("public_budget", "行政主体の会計集計(歳入・歳出・残高){level, revenue, expense, balance}")
register_event_kind("media_use",     "娯楽メディアの利用(TV/動画/ゲーム。タイトルは架空){medium, title?, steps, at}")
register_event_kind("study",         "学校の授業・学習(教科・講義){org, subject, role}")
register_event_kind("production",    "職場での産出(財・サービスの実態){org, output, kind}")
# ---- 第7バッチ(ユーザー要望 2026-07-07: 日付・天気)----
register_event_kind("weather",       "その日の天気(日次・世界イベント agent_id=-1){date, weekday, cond, temp_hi, holiday?}"
                                     "。weather.mode=generated/table では観測用に "
                                     "{temp_lo, temp_hi_c, temp_lo_c, precip_mm, humid, wbgt, source} を併記"
                                     "(既定 synthetic では従来どおり cond/temp_hi のみ)")
# ---- 長期予定・スケジュール帳(第7バッチ 2026-07-07。会話からの自動記入。既定 OFF)----
register_event_kind("appointment",   "予定の設定(会話からの自動記入){day, when, what, place, with}")
register_event_kind("appointment_kept", "予定の遵守(予定時間帯に該当場所に居た。任意観測){day, when, what, place}")
# ---- 感情・興味・注意ハブ(affect。既定 OFF。設計: docs/lit/neuroscience__emotion-interest-attention.md)----
register_event_kind("affect_update", "覚醒度arousalの更新(感情・興味・注意ハブ){old,new,cause}")
# ---- 現実ギャップ実装 Wave G2(社会関係の質。既定 OFF。設計: docs/design-candidates/gap-implementation-plan.md)----
register_event_kind("relation_tier",   "関係の深化段階の変化(知人→友人→親友等){other, tier, count}")
register_event_kind("relation_break",  "関係の断絶・悪化(ネガ交流・長期不在){other, from_tier, to_tier, cause}")
register_event_kind("reputation_update", "評判・信頼スコアの更新(口コミ・語の採用で伝播){old, new, cause}")
# ---- 現実ギャップ実装 Wave G3(制度改変の3ルート: 職域・民主・執行。既定 OFF)----
register_event_kind("labor_action",  "労働争議の提起(職場同僚への集合行為){org, demand, initiator}")
register_event_kind("vote_cast",     "提案への投票(民主的ルート){proposal_id, vote}")
register_event_kind("vote_result",   "投票による提案の可決・否決 {proposal_id, yes, no, passed}")
# ---- 制度深化(第9バッチ 2026-07-07。多段制定=審議/パブコメ・供託金。既定 OFF)----
register_event_kind("proposal_review", "提案の審議段階(パブコメ窓の開始/可決送り/否決){proposal_id, phase, supporters?, opposed?}")
register_event_kind("deposit",         "供託金(propose の受理拠出・返還・没収・拠出不能){proposal_id?, amount, phase, balance?}")
# ---- 制度深化 第2弾(第10バッチ 2026-07-08。勾留・営業許可。既定 OFF)----
register_event_kind("detention",       "執行時の拘束=勾留の最小形(数stepの行動停止){target, officer, rule_id, steps}")
register_event_kind("venture_permit",  "出店の営業許可(却下 or 許可=開業待ち){name, outcome, open_step?, reason?}")
# ---- 出来事誘発の深い内省(第12バッチ 2026-07-08。既定 OFF)----
register_event_kind("reflection_trigger", "日内衝撃ゲージの閾値超え=深い内省の予約(侵入的段階の開始){gauge, day, due_day}")
# ---- 制度深化 第3弾(2026-07-08。代表制議会・立退き・破産サイクル。既定 OFF)----
register_event_kind("council_elected", "代表制議会の改選(意見最近傍の決定論選挙。世界イベント agent_id=-1){term, day, members, n_candidates}")
register_event_kind("eviction",        "家賃滞納の立退き/完済の再入居(住居サイクル){phase, arrears?, days?}")
register_event_kind("bankruptcy",      "自己破産=免責+自由財産以外の圧縮+制限期間{debt, seized, keep, until_step, venture_closed}")
register_event_kind("enforcement",   "条例・制度DSLルールの執行(公務員=警察官による){rule_id, officer, target, penalty}")
# ---- 現実ギャップ実装 Wave G4(文化カレンダー・群集。既定 OFF)----
register_event_kind("annual_event",  "年中行事の発生(正月・ハロウィン・年末など。世界イベント agent_id=-1){name, date}")
register_event_kind("crowd_surge",   "大規模群集の発生・観測(スクランブル等への集中){node, level, event}")
# ---- 現実ギャップ実装 Wave G5(キャリア転換。既定 OFF)----
register_event_kind("job_change",    "転職・配属変更(職場が変わった){from_org, to_org, cause}")
register_event_kind("unemployment",  "失業・求職(職を失う/職を得る){state, org?}")
register_event_kind("venture_fulltime", "起業転換(本業を辞め出店を本業にする){name, node}")
# ---- 現実ギャップ実装 Wave G6(情報環境の非対称。既定 OFF)----
register_event_kind("feed_rank",     "TL推薦の並べ替え(意見整合バイアス=エコーチェンバー){agent, boosted, filtered}")
register_event_kind("viral_cascade", "バイラル拡散(インフルエンサー非対称のリシェア連鎖){post_id, author, reach}")
register_event_kind("misinfo",       "誤情報・訂正・炎上(フェイクの拡散と打ち消し){post_id, kind}")
# ---- 現実ギャップ実装 後続波 H1(健康・疲労・病気。既定 OFF)----
register_event_kind("illness",       "病気の発症・回復(欠勤・外出抑制){state, kind?}")
register_event_kind("medical_visit", "医療機関の受診(移動・消費){node?, cost}")
register_event_kind("health_update", "疲労・メンタルの状態更新(内部 transient){name, old, new, cause}")
# ---- 現実ギャップ実装 後続波 H2(世帯・家族・恋愛。既定 OFF)----
register_event_kind("partner_formed", "恋愛・パートナー関係の成立(相互の強い親密度から){other}")
register_event_kind("life_event",     "ライフイベント(結婚・同居・別れなど){kind, other?}")
# ---- 内部可動性 第60バッチ(b)(転居・同棲・求職の内生化。既定 OFF=0件。実装 src/society/mobility.py)----
register_event_kind("relocate",  "内生的な世帯転居(職場変更後の通勤逼迫 or 家賃逼迫。世帯単位){from, to, reason, n}")
register_event_kind("move_in",   "同棲開始=後から来た側が相手宅へ転居し世帯併合(bond→N日+closeness){pair, from, to, reason}")
register_event_kind("move_out",  "別離に伴う転出=後から来た側が新居へ移り世帯分離(unbond 時){pair, from, to, reason}")
register_event_kind("job_search","求職 tool の発火(career選択由来化。決定論マッチ→switch_org/rehire){outcome, from_org?, to_org?}")
# ---- 現実ギャップ実装 後続波 H3(商業ダイナミクス。既定 OFF)----
register_event_kind("shop_state",    "店舗の開店・閉店(営業時間){poi, state}")
register_event_kind("price_change",  "動的価格の変動(需給・セール){poi?, cat, ratio}")
register_event_kind("stock_out",     "品切れ・行列(需要集中の資源希少化){poi, cat}")
# ---- 現実ギャップ実装 後続波 H4(都市・環境ショック。既定 OFF)----
register_event_kind("disaster",      "災害ショック(台風・地震・大雪。世界イベント agent_id=-1){kind, phase}")
register_event_kind("transit_delay", "交通の遅延・運休(電車・バス){line?, kind}")
register_event_kind("infra_outage",  "インフラ障害(停電・通信断・断水){kind, phase}")
# ---- 現実ギャップ実装 後続波 H5(観光・多言語・犯罪・治安。既定 OFF)----
register_event_kind("crime",         "犯罪・被害(窃盗など。被害者/加害者){kind, victim?, offender?, amount?}")
register_event_kind("nuisance",      "迷惑行為(客引き・ナンパ・騒ぎ){kind, node}")
register_event_kind("tourist_visit", "観光客のランドマーク回遊(訪日客・回遊型){node}")
# ---- 現実ギャップ実装 後続波 H6(内面本格版: 離散感情・長期目標・趣味。既定 OFF)----
register_event_kind("emotion_label", "離散感情ラベルの変化(core affect→怒/哀/喜/怖/驚等){label}")
register_event_kind("long_goal",     "長期目標・人生設計の設定/変化(keystone の駆動源){goal}")
# ---- 宿泊・ホテル滞在(第9バッチ 2026-07-07。来街者・観光客の夜間宿泊。既定 OFF)----
register_event_kind("lodging_checkin",  "ホテルへのチェックイン(宿泊費の支払い・夜間滞在){poi, node, price, nights}")
register_event_kind("lodging_checkout", "ホテルからのチェックアウト(滞在終了・活動再開){poi, nights_stayed}")
# ---- 第37バッチ(2026-07-19。選挙現実化 S1・経済深化 E・記憶 M。全て既定 OFF)----
register_event_kind("candidacy",       "議会選挙への自発的立候補(供託金納付・告示期間中のみ){day, deposit, age}")
register_event_kind("election_result", "選挙の開票結果(SNTV。世界イベント agent_id=-1){day, votes, elected, forfeited}")
register_event_kind("council_budget",  "議会の予算承認・否決(government接続){term, day, approved, amount}")
register_event_kind("ordinance_vote",  "議会の条例議決(住民署名→審議→採決の終端){rule_id?, passed, yes, no}")
register_event_kind("loan_grant",      "銀行融資の実行(与信スコア審査通過){amount, rate, term_days, score}")
register_event_kind("loan_repay",      "融資の返済(定期返済・完済・延滞){amount, remaining, status}")
register_event_kind("interest_paid",   "預金利息の付与(日次/週次){amount, balance}")
register_event_kind("vc_investment",   "ベンチャー出資の実行(トラクション/ネットワーク審査){venture, amount, equity}")
register_event_kind("memory_fail",     "想起の失敗(活性化が閾値未達=思い出せない){query, activation, tau}")
# ---- 会話3層 C2(P2 S3。構造化会話=LLMなし。既定 OFF。設計: docs/plans/p2-interstitial-design.md §1 S3)----
register_event_kind("conversation",    "構造化会話 C2(実文なし・機械効果のみ。1会話1件){with, topic, tone, outcome, acts}")
# ---- 日課計画フレームワーク(P2 S1。planning.framework ON のみ・1日1件。設計: docs/research/daily-plan-framework.md)----
register_event_kind("day_plan_compiled", "日課計画のコンパイル要約(型スキーマ→アンカー先置き→充填){n, cats, anchors, path}")
# ---- 確率的実行 L3(P2 S4。routine.stochastic ON のみ・既定 OFF=0件。設計: docs/plans/p2-interstitial-design.md §1 S4)----
register_event_kind("detour",        "確率的実行 L3: 寄り道=目的地手前で近傍 POI に立ち寄り(P2 S4){act, to, from_dest}")
register_event_kind("interrupt",     "確率的実行 L3: 活動途中の中断=離脱(P2 S4){act}")
# ---- 好奇心/退屈の内発ドライブ(P2 S5。drive.boredom ON のみ・既定 OFF=0件。設計: docs/plans/p2-interstitial-design.md §1 S5)----
register_event_kind("boredom_explore", "好奇心/退屈ドライブの内発的探索=近傍の未訪問/低頻度 POI へ移動(LLM なし。P2 S5){from, to_kind, gauge}")
# ---- 行動方針キャッシュ(P2 S7。cognition.policy_cache ON のみ・既定 OFF=0件。設計: docs/research/interstitial-life.md §4.4/§7)----
register_event_kind("policy_reuse", "行動方針キャッシュの再利用=LLM 呼をスキップ(P2 S7){kind, relax, saved}")
# ---- 日次ローテーション/presence(W2 P3。pool.enabled ON のみ・既定 OFF=0件。日境界で1件。
#      agent_id=-1 の世界イベント。設計: docs/plans/w2-execution-plan.md §4 P3 / persona-pool.md §5)----
register_event_kind("presence_change", "日次ローテーションの在場入替(日境界で1件){day, n_enter, n_exit, n_present}")
# ---- エージェント視覚 v1(pov.enabled ON のみ・既定 OFF=0件。顕著性ゲート→POVレンダ→画像ストア。
#      専用 stream "pov_salience"・追加LLM呼ゼロ。設計: docs/research/agent-vision.md §4 v1)----
register_event_kind("pov_image", "顕著時のPOV画像を撮影・保存(サイドカー画像への参照キー){ref, w, h, trigger, vlm?}")
# ---- L2 業務の実体(work.service ON のみ・既定 OFF=0件。決定論・LLM/乱数ゼロ。
#      設計: docs/research/l2-work-reality.md)----
register_event_kind("serve",      "接客業務の帰属(客の消費と同一 work_node の勤務中スタッフに応対を帰属。不在時は agent_id=-1 の記録){cat, label, customer, node, unstaffed?, org_id?, floor?}")
register_event_kind("org_output", "オフィス系職場の日次産出集計(頭数×role重み or ミクロ在席分。会社が『何かを作っている』の最小観測形){org, output, n, kind, basis?, day?}")
# ---- 職場束ね直し(work.bind_workplace ON のみ・既定 OFF=0件。起動時1回・世界イベント agent_id=-1。
#      pool 経路の L2/L3 を台帳 workplace_poi へ org_id 束ね=serve/org_output の網羅率を上げる)----
register_event_kind("workplace_bound", "起動時の職場束ね直しの coverage 統計(day0 present の pool 個体){n_total, n_unbound_before, n_bound, n_unbound_after}")
# ---- 物流の実体化 スライス①+②(commerce.inventory ON のみ・既定 OFF=0件。決定論・LLM/乱数ゼロ。
#      実装 src/society/goods.py。設計: docs/research/economy-goods-services.md §7 ①②)----
register_event_kind("delivery_trip", "補充の配送トリップ(depot=最寄りゲートウェイ→店。(s,S)発注。世界イベント agent_id=-1){from, to, cat, qty, eta}")
register_event_kind("restock",       "補充トリップの到着=在庫が上限 S へ回復(封鎖時は失敗=不発){poi, cat, qty, from}")
register_event_kind("stock_low",     "在庫僅少=発注点 s 以下(補充発注のトリガ。世界イベント agent_id=-1){poi, cat, level}")
# ★ stock_out は commerce 既存 kind を再利用(意味を在館数の代理→実在庫の枯渇に拡張。src="inventory")
# ---- サービスの実体化 スライス③(services ON のみ・既定 OFF=0件。決定論・新 stream "service" のみ。
#      実装 src/society/services.py。設計: docs/research/economy-goods-services.md §7 ③)----
register_event_kind("service_use", "サービスの受給=滞在+課金+効果(理美容/クリニック/塾/ジム/クリーニング等。需要側=供給側 serve と対){node, service, cost, poi}")
# ---- B2B 卸→小売の仕入れ スライス⑤(commerce.inventory.b2b ON のみ・既定 OFF=0件。決定論・乱数ゼロ。
#      実装 src/society/b2b.py。設計: docs/research/economy-goods-services.md §7 ⑤)----
register_event_kind("b2b_trade", "会社間取引=卸 org→小売 POI の仕入れ(org 間で金+物が移転。世界イベント agent_id=-1){from_org, to_poi, cat, qty, amount, from_node}")
# ---- 宅配・フードデリバリー スライス④(delivery ON のみ・既定 OFF=0件。決定論・新 stream "delivery" のみ・
#      LLM 呼ゼロ。実装 src/society/delivery.py。設計 docs/research/economy-goods-services.md §7 ④)----
register_event_kind("order",   "宅配の注文(在宅/職場滞在で外食の代替。① 在庫を1引き店で受取){poi, item, fee, cat, to_node}")
register_event_kind("deliver", "宅配の配送完了=到着で受給+課金(配達員に gig 収入。agent_id=配達員 or -1){courier, from_poi, to_node, item, fare, cat}")
# ---- 関係性の再現 第44バッチ(共同行動エンジン S-R3 + 世帯の夕食共食 S-R1。既定 OFF=0件。
#      決定論・LLM/generate ゼロ増。実装 src/society/joint.py + household.py。設計: docs/research/relationships-activities.md §4)----
register_event_kind("joint_activity", "共同行動の成立=同伴グループ/世帯が同一POI・home で2人以上同席(1グループ1日1回){type, with, place, tier}")
# ---- 関係性パッケージ完結 第45バッチ(友人グラフ S-R2 / 職場会食 S-R4 / 来街者 party S-R5。既定 OFF=0件。
#      決定論・LLM/generate ゼロ増。実装 src/society/friends.py + joint.py + party.py。設計: docs/research/relationships-activities.md §4)----
register_event_kind("friend_graph_built", "初期友人グラフの生成統計(homophily+所属+Dunbar。起動時1回・世界イベント agent_id=-1){n_edges, mean_degree}")
# ★ S-R4(職場の会食・飲み会)は joint_activity{type:colleague_lunch|colleague_dinner} を再利用(schema変更なし)。
# ★ S-R5(来街者 party)は joint_activity{type:party} を再利用(schema変更なし)。
# ---- 火種介入の実験条件化 spark treatment(第53バッチ 2026-07-23。spark.enabled ON のみ・既定 OFF=0件。
#      決定論・LLM/generate ゼロ増・新 stream "spark" のみ。実装 src/society/spark.py。事後トレーサ
#      scripts/trace_spark.py。設計: docs/plans/finals-day1-decisions.md D13)----
register_event_kind("spark_roster", "火種介入の t=0 名簿(選抜 ids/pids・3メニュー ON/OFF・較正 params。起動時1回・世界イベント agent_id=-1){ids, pids, n, menus, params}")
# ---- 生活の偶発イベント層(第54バッチ 2026-07-23。chance.enabled ON のみ・既定 OFF=0件。決定論・新 stream
#      "chance" のみ・LLM/generate ゼロ増。実装 src/society/chance.py。運/実力分解 audit_uncertainty.LUCK_KINDS
#      に 1 行接続=監査+ΔR² に自動反映。設計: docs/research/uncertainty-audit.md §4)----
register_event_kind("chance_event", "生活の偶発事(臨時収入 windfall / 財布紛失 loss / 偶然の出会い encounter){type, amount?, balance?, other?}")
# ---- 屋内エンジン配線 B3(indoor.enabled ON のみ・既定 OFF=0件。決定論・新 stream "indoor"/"indoor_meet" のみ。
#      フロア内の区画遷移=step 粒度の屋内ミクロ移動。秒スケール軌跡は L1 に入れず tracks サイドカーへ)----
register_event_kind("space_move", "屋内フロア内の区画遷移(markov 回遊/会議/階間到着){building, floor, from_zone, to_zone, from_type, to_type, offset_s, kind}")
# ---- 負の評判(悪評)の内生伝播 第61バッチ スライス(c)(gossip.enabled ON のみ・既定 OFF=0件。決定論・
#      新 stream "gossip" のみ・LLM/generate ゼロ増。実装 src/society/gossip.py。設計 devlog Entry 50)----
register_event_kind("gossip_seed",   "悪評タグの発生=既存の負イベント当事者が悪評化(匿名タグ=何をしたかは L1 参照){n, cause}")
register_event_kind("gossip_spread", "悪評タグの伝播採用=独立した複数の知人から聞いて採用(complex contagion){target, sources}")
register_event_kind("gossip_fade",   "悪評タグの忘却=知る者が悪評を忘れる(烙印にしない=噂の風化){target}")
# ---- 承諾/拒否判断の内生化 第62バッチ フェーズ1(relations.endogenous_accept ON のみ・既定 OFF=0件。
#      全決定論・LLM/generate ゼロ増・新 stream なし(既存 "joint" の draw を再利用=always-draw)。
#      実装 src/society/relations_endo.py + joint.py。設計正典 docs/plans/endogenous-relations-plan.md §2)----
# ★ 第64バッチ フェーズ3(relations.endogenous_invite ON 時のみ): payload に source
#    (plan_with/dialog_cue/weak_tie/closeness/housemate/colleague=誘い先の選抜経路)が付く。
#    accept 単独 ON の payload は従来とバイト一致のまま(source キー自体が無い)。
register_event_kind("joint_invite",  "共同行動の誘い1件の承諾判定内訳(誘い手視点。1日1誘い1件){invitee, verdict, basis, p_calib, p_final, accepted, source?}")
# ---- 場所の意味づけ最小版 D1 第69バッチ(labeling.place_binding ON のみ・既定 OFF=0件。
#      全決定論・乱数ゼロ・LLM/generate ゼロ増・新 stream なし。造語を発生ノードへ束縛した
#      瞬間を 1 件だけ記録する(最初の coin のみ・上書きなし)。実装 src/society/labeling/labels.py。
#      設計 docs/plans/twin-physics-vision-affordance-plan.md §2 レーン1 D1)----
register_event_kind("place_label_bind", "造語がその発生ノードへ束縛された(語ごとに最初の1回のみ){word, node}")
# ---- 未定義行動レジスタ 第70バッチ IDEA②(freedom.undefined_register ON のみ・既定 OFF=0件。
#      全決定論・乱数ゼロ・LLM/generate ゼロ増・プロンプト 1 バイト不変=パース後の振り分けのみ)。
#      「JSON は読めて "action" もあるが、既知の動詞のどれでもない」出力 = 与えられた行動空間の
#      外へ出ようとした痕跡。従来はこれが fallback{reason:"parse_error"} = JSON 崩れと同じ箱に
#      入って内容ごと捨てられていた。ON のときは fallback **ではなく**こちらへ振り分ける(分子は排他)。
#      実装 src/society/cognition/deliberate.classify_reject + engine/scheduler._log_reject。
#      ★正直な限界: mock は enum 内しか返さないので実発火は実 LLM ランに限られる。
register_event_kind("undefined_action", "enum 外だが構文としては行動を主張している LLM 出力(行動空間の拡張の痕跡){action, keys, text, trigger}")
# ★ 沈黙の第一級化(freedom.explicit_nothing ON のみ)は既存 kind "stay" を再利用する
#   (payload に reason:"chosen_nothing" が付く。既定 OFF では stay イベントは 1 件も出ない)。
# ---- 真偽台帳ミニマル 第73バッチ Part B(beliefs.enabled ON のみ・既定 OFF=0件。全決定論・
#      乱数ゼロ(新 stream なし)・LLM/generate ゼロ増・プロンプトは verify_actions の 1 行のみ。
#      実装 src/society/truth_ledger.py。設計 docs/plans/source/dual-mode-instruments.md Part B)。
#      ★ 真値(ground truth)そのものは L1 に出さない: 台帳は runs/<name>/beliefs_ledger.json
#        (サイドカー)に分離してある。ここに出るのは**エージェントが持っている信念**だけ。
register_event_kind("belief_update",   "信念の獲得・更新(親ノードつき=伝播木が事後に構成できる)"
                                       "{fact, fact_kind, value, conf, src, from, hop, cause, verified}")
register_event_kind("belief_transmit", "伝聞の送出(話題が一致した発話に乗って聞き手へ伝わった)"
                                       "{fact, fact_kind, to, channel, hop, topic, conf}")
register_event_kind("belief_verify",   "検証行動(現場go/当事者ask/ネットnet)の実行と根拠"
                                       "{fact, fact_kind, how, about, match, outcome, evidence?}")
# ---- ダンバー認知枠(関係の維持コスト)第75バッチ IDEA⑤(relations.dunbar ON のみ・既定 OFF=0件。
#      全決定論・乱数ゼロ・LLM/generate ゼロ増・新 stream なし。実装 src/society/dunbar.py。
#      設計 docs/research/hackathon1-analysis/ideas-shortlist.md §2 ⑦)。
#      休眠は**削除ではない**: 台帳エントリ・count・記憶はそのまま残り、休眠前の closeness を
#      退避して closeness=0(=tier 0)にする。再会は退避値 × rekindle_discount で復元する。
register_event_kind("relation_dormant",  "関係の休眠=認知枠の上限超過で最弱の紐帯を不活性化"
                                         "(退避であって削除ではない){other, closeness, from_tier, gap_days}")
register_event_kind("relation_rekindle", "休眠関係の再会=再接触で割引つきに再活性化"
                                         "{other, before, closeness, tier, gap_days}")
# ---- 認知イベントキュー(閾値発火)第81バッチ(cognition.fire ON のみ・既定 OFF=0件)。
#      P0-4「認知イベントの発火時刻・理由・トリガー元をすべて記録し、事後にスケジュール列
#      そのものを解析対象にできること(思考頻度の分布自体が観測量になります)」。
#      実装 src/society/cognition/fire.py。**LLM へ渡す文面は 1 バイトも変わらない**
#      (発火理由はここにしか出ない=エージェント側から機構の ON/OFF が読めない)。
register_event_kind("cog_fire",  "認知イベントが期限を迎えた(スケジュール列そのもの)"
                                 "{reason, due, ctx, s, theta, contrib, interrupt, active, granted}")
register_event_kind("cog_event", "既存機構が担う第一級の認知イベント(会話返答/夜内省/朝計画)"
                                 "{reason, via, ctx}")
# ---- 監視仕様 + g/θ 更新則 第82バッチ(cognition.watch / cognition.g_update ON のみ)。
#      設計 §2.2(watch spec)・§2.5(g)・§2.6(θ の恒常性)。
#      watch_spec は**読めなかったことも記録する**(構造化出力の遵守率が観測量)。
register_event_kind("watch_spec", "監視仕様(期待値 ô + 名前付きトリガ)の受理結果"
                                  "{status, clamped, n_expect?, n_trigger?, names?}")
register_event_kind("cog_theta",  "閾値 θ の恒常性を日境界で 1 回適用した"
                                  "{day, fbar, fired, theta_mult}")


@dataclass
class Event:
    step: int
    sim_min: int
    agent_id: int
    kind: str
    x: float
    y: float
    payload: dict[str, Any] = field(default_factory=dict)
    rng_stream: str = ""
    llm_call_id: str | None = None
