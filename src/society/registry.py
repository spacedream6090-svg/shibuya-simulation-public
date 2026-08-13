"""機能レジストリ + ランモード(第72バッチ 2026-07-31)。

正典: docs/plans/dual-mode-observe-verify-plan.md 第72行 /
      docs/plans/source/dual-mode-instruments.md Part A。

何を解く問題か
--------------
本リポジトリは「再現性を担保できない実装も積極的に入れる」方針(本選の観察ランは
面白いものを全部載せる)を採る。その代わり、**載せたものを対照実験のときに確実に
外せる**必要がある。そこで各機能トグルへ「再現性等級 repro_tier」を宣言させ、
ランのモード(run.mode)で機械的に取捨する。

  repro_tier
    strict  … 決定論(+用途別 named stream の乱数)だけで挙動が決まる層。
              LLM の**自由文出力をデータとして読まない**。プロンプトに1行足すだけの
              層・観測専用の層もここに入る(世界の因果は seed から再構成できる)。
    journal … 機能の本体が **LLM の自由文出力を消費する**(自由記述の行動をパースする・
              造語文字列を世界状態にする・LLM 呼を足す/減らす/差し替える)。
              単体では非決定だが llm_cache / llm_journal があれば事後に再生できる。
    none    … 記録しても再生できない(実時刻・外部プロセス/API・非決定並列)。
              **現状ほぼ存在しない**のが本リポジトリの実態で、正直にそう登録してある
              (該当は SUMO ライブ連成タクシーの2件だけ)。

  affects_k … ON/OFF で **LLM 呼び出しの発生箇所・本数が直接変わる**か。
              ★正直な限界: 位置・ゲージ経由で発火数が間接的に動く機能(健康・商業・
              災害など「co-location 変化系」)は **False** にしてある。ここを True に
              すると本リポジトリのほぼ全機能が True になり属性として役に立たないため、
              「generate() の呼び出し点を足す/減らす/予算を変えるか」に限定した。
              間接効果は既存の compute_matched 対照でしか切り分けられない。

  fingerprint_risk … エージェント側から実験条件が観測されうるか。
              none=物理・会計のみ / possible=プロンプトに文言や欄が増減する層
              (差分に気づく余地が原理的にある)/ known=当人にとって不自然な事象が
              観測できることが判っている層(t=0 の資本注入・偽内省)。

ランモード(conf: run.mode。既定 "none" = **現行動作そのまま**)
--------------------------------------------------------------
  none    … 何もしない(1バイトも変えない)。既存ラン・golden・過去 config との互換。
  observe … 全等級を許可(制限なし)。宣言を manifest に明示するだけ。本選の観察ラン。
  journal … repro_tier=none の機能を自動 OFF。事後再生可能な範囲での観察。
  verify  … strict のみ許可(journal / none を自動 OFF)。対照実験・アブレーション用。

自動 OFF は **黙って起きない**: 目立つ警告ログ(logging WARNING= 既定でも stderr に
出る)+ run_manifest.json の run_mode / features.enabled / features.auto_disabled。

R1 との関係
-----------
- 既定 run.mode=none では本 module は **一切 config を書き換えない**(同一オブジェクトを
  返す)。golden L1 バイト一致は無風。
- 書き換えるときも「シム構築の最初の1箇所」(Simulation.__init__ 冒頭)で resolved config
  のキーを off 値へ落とすだけ。落とした後の config が save_config でランへ保存されるので、
  「そのランが実際に何を有効にしていたか」は事後にも config.yaml と manifest の両方から判る。

未宣言検出
----------
tests/test_registry_modes.py が conf/config.yaml の **真偽値リーフ全部**を走査し、
本レジストリにも ALLOWLIST にも無いキーがあれば fail する(= 新機能を足すと
repro_tier の宣言を忘れられない)。ALLOWLIST は「機能でない bool 設定キー」専用で、
理由をコメントとして必ず添える。
"""
from __future__ import annotations

import copy
import json
import logging
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

log = logging.getLogger("society.registry")

TIERS = ("strict", "journal", "none")
RISKS = ("none", "possible", "known")
MODES = ("none", "observe", "journal", "verify")

# モードが許す最上位の等級(none モードと observe は全許可)
MODE_MAX_TIER = {"none": "none", "observe": "none",
                 "journal": "journal", "verify": "strict"}

_TIER_RANK = {"strict": 0, "journal": 1, "none": 2}


@dataclass(frozen=True)
class Feature:
    """機能トグル1件の宣言。id は conf のドットパス(= 実在するキーでなければならない)。"""
    id: str
    repro_tier: str
    affects_k: bool
    fingerprint_risk: str
    description: str
    off_value: object = False      # 自動 OFF 時に書き込む値(bool 以外のトグル用)


def _f(fid: str, tier: str, affects_k: bool, risk: str, desc: str,
       off_value: object = False) -> Feature:
    return Feature(fid, tier, affects_k, risk, desc, off_value)


# --------------------------------------------------------------------------- #
# レジストリ本体(conf/config.yaml の宣言順に近い並び)
# --------------------------------------------------------------------------- #
FEATURES: tuple[Feature, ...] = (
    # ---- run / k / controls / reflection ----
    _f("run.natural_start", "strict", False, "none",
       "開始時刻が就寝帯に入る居住者を就寝状態・自宅で着席させる(初日コールドスタート改善)"),
    _f("run.dt_min", "strict", True, "none",
       "中央 Δt(1 step の分数。既定 10=正準=golden の世界)。timeconv.py の分類テーブルに従い"
       "レート/確率/step 長を config ロードの 1 箇所で変換する。10 のときはテーブルを走査せず"
       "config を 1 バイトも触らない。Δt≠10 は乱数消費列が変わる別世界(統計量の同オーダーで検証)。"
       "★affects_k=True(第94バッチ OBS-U2 で False から訂正): affects_k の定義は"
       "「generate() の呼び出し点を足す/減らす/**予算を変える**か」であり、Δt は RATE 類の"
       "lod.max_llm_per_step を 300→30 に変える(1 日 cap 総量 43,200 は保存)。さらに Δt=10 では"
       "CogQueue が周期発火で潰していた salience 割込みが解放され、呼数は実測 ×2.2〜2.4 になる",
       off_value=10),
    _f("k.writeback", "strict", False, "possible",
       "D7 の主実験条件(free/degraded/sham/off)。内省の書き戻し自由度。sham は当人の記憶と"
       "矛盾する belief が入りうる=原理的に観測可能", off_value="off"),
    _f("controls.mode", "strict", True, "none",
       "D7 対照系列(null_series=内容非結合のダミー呼を足す / compute_matched=計算量一定)",
       off_value="none"),
    _f("reflection.deep.enabled", "journal", False, "possible",
       "出来事誘発の深い内省。LLM 内省の自由文から自己像(self_model)を作る"),
    _f("reflection.implicit_self.enabled", "strict", False, "possible",
       "無意識層。行動カウントのベースライン逸脱から決定論で『最近の自分』1行を日次更新"),

    # ---- model(LLM 側の装置。cache/journal は再現性の道具なので strict)----
    _f("model.cache", "strict", False, "none",
       "応答キャッシュ(D13)。再現性の実体そのもの=verify でも落としてはならない"),
    _f("model.journal", "strict", False, "none",
       "LLM 入出力ジャーナル(第71)。記録専用でシムは読まない"),
    _f("model.cache_mode", "strict", False, "none",
       "free(既定)/ replay(キャッシュミスで即例外=再現ランの fail-fast)", off_value="free"),
    _f("model.reflect_think", "strict", False, "none",
       "内省を思考モードで回す。LLM のパラメータ(キャッシュキーに含まれる)=分岐は生まない"),

    # ---- world ----
    _f("world.scenario", "strict", False, "possible",
       "摂動カタログ(shock_closure=区間封鎖 / shock_event=定型ニュース注入)",
       off_value="baseline"),
    _f("world.elevation.enabled", "strict", False, "none",
       "標高 z 列をイベント payload へ付ける(表示・観測専用)"),
    _f("world.heights.enabled", "strict", False, "none",
       "建物へ実高さ属性を付与(PLATEAU 実測表。現状は属性付与のみ)"),
    _f("world.floor_clamp.enabled", "strict", False, "none",
       "屋内位置の階を建物の実階数へ丸める(3D-U0)。規則は表示側 export_3d.encode_indoor_w と"
       "同一 = [1, max(1, min(levels, 99))]。POI 由来の階数超え floor が L1 に通るのを止める。"
       "ON は L1 が変わる(floor 値と、同一階 co-location 経由でプロンプト loc も)"),
    _f("world.mod.enabled", "strict", False, "possible",
       "環境改変条件(エッジ封鎖・速度係数・営業時間)をワールド構築時に一度だけ適用"),
    _f("world.traffic.enabled", "strict", False, "none",
       "背景交通(エージェント以外の通過車両)の生成"),
    _f("world.vision.enabled", "strict", False, "none",
       "擬似視覚 C: 壁による視線遮蔽で可聴・交流を絞る"),
    _f("world.vision.outdoor", "strict", False, "none",
       "屋外も建物フットプリントで遮蔽する(vision の重い版)"),
    _f("world.scene_desc.enabled", "strict", False, "possible",
       "構造化シーン記述を発火プロンプトへ注入(方向つき視界・注視対象・垂直関係)"),
    _f("world.scene_desc.vertical", "strict", False, "possible",
       "シーン記述の垂直関係(レイヤ/標高)1行を出すか"),
    _f("world.calendar.enabled", "strict", False, "possible",
       "暦(日付・曜日・祝日)。全プロンプトへ日付1行を注入する"),
    _f("world.calendar.weekday_work", "strict", False, "none",
       "本業勤務・登校を平日だけに絞る"),
    # 装置(device)層 = 目標を持たないアクター(actor model P2。実装 src/society/devices.py)。
    # strict の根拠: 装置は**決定論的な応答関数**であり乱数 stream を 1 本も引かない
    #   (設計契約。module に乱数の識別子が存在しないことを tests/test_devices.py が AST 固定)。
    # affects_k=False: generate() の呼び出しサイトを 1 つも足さず・減らさない。
    # fingerprint_risk=none: **プロンプトへ 1 バイトも足さない**(待たされたことは
    #   「動けない」という世界の事実としてしか現れない=新語彙も新欄も無い)。
    _f("world.devices.enabled", "strict", False, "none",
       "装置層の親トグル。装置が状態を変えてよいのは (a) 宣言済みのスケジュール/物理"
       "[DEVS δ_int] か (b) エージェントの行為への応答[DEVS δ_ext] のときだけ、という"
       "契約を型と監査証跡で与える。ON では装置名簿(id → 装置)が sim に生え、"
       "改札 / 歩行者信号がその契約の下で動く。既定 OFF は名簿を組まない(L1 バイト一致)"),
    _f("world.devices.faregate.enabled", "strict", False, "none",
       "改札装置(通路配列 + FIFO 待ち行列)。通路数 × 処理間隔(既定 60 人/分/通路 = "
       "日本の駅改札の**実測系公表値**。公称値は使わない)で個体ごとの待ち時間を出し、"
       "整備窓(= 1 step)の定員を超えた個体だけ退出遷移が遅れる。ON のとき "
       "env.feedback 規則2(改札スループット飽和 → 入場規制)は**評価されない** = "
       "同じ物理(改札の処理能力飽和)の二重計上を防ぐ細粒度側の優先。既定 OFF は "
       "gate_pass 0 件・agent に属性が生えない"),
    _f("world.devices.signal_events", "strict", False, "none",
       "歩行者信号装置の観測(signal_summary を **1 交差点 1 日に最大 1 件**)。"
       "信号サイクル 140 秒 < 1 step 600 秒 なので**サイクルごとのイベントは出さない**"
       "(出すと L1 が信号で溢れる)。sfm_core.SignalGate の時刻計算は 1 バイトも"
       "変更していない = 足したのは交差点表由来の安定 device_id とこの要約だけ"),
    _f("world.devices.transit_operator", "strict", False, "none",
       "運行事業者(TransitOperator)= 運休を**決める**装置。自然事象は台風そのものだけで、"
       "『今日は止める』は外部入力への応答(DEVS δ_ext)である。ON にすると disaster.py の"
       "運休判定が装置の δ_ext を**素通し**で通る(計算は 1 演算も変えない = 運休する日は"
       "ビット同値)。得られるのは DEVS 監査(自発的遷移ゼロ)と装置名簿への登録であって、"
       "挙動の変化ではない。L1 の device_id は observer.causality 側の仕事で独立に効く"),
    # 駅到着のパルス量子化(actor model Wave 3。実装 engine/simulation.py の流入初期化)。
    # strict の根拠: 時刻表(静的データ)からの**純関数**で、乱数 stream を 1 本も引かず
    #   LLM も呼ばない(同じダイヤ・同じ名簿なら何度組んでも同じ到着表)。
    # affects_k=False: generate() の呼び出し点を足しも減らしもしない(到着 step が動くだけ)。
    # fingerprint_risk=none: プロンプトへ 1 バイトも足さない(新しい欄も語彙も無い)。
    _f("world.inflow_pulse.enabled", "strict", False, "none",
       "駅ゲートウェイの流入通勤者の到着時刻を**時刻表の到着へスナップ**する(境界の研究="
       "駅からの流入は平滑ではなく列車ごとの塊 platoon である。平滑な流入では改札の待ち行列も"
       "ホームの滞留も原理的に立たず、改札装置 world.devices.faregate が観測すべき現象が消える)。"
       "規則 = 引かれた時刻以上で最も早い到着 / 始発前は始発へ / 終電後は最終到着へ、同着は"
       "路線名の昇順。二峰分布の draw 自体は 1 ビットも変えない(量子化だけが新しい)。"
       "縁ゲートウェイは徒歩流入なので対象外。★分解能の限界: ダイヤは路線ごとの"
       "(始発, 終電, 運転間隔)の 3 値しか無く 1 本ごとの実発車時刻を保持していないため、"
       "返るのは等間隔の**再構成**である(GTFS 経路も中央値 headway へ畳んでいる)。"
       "既定 OFF は到着表を組まない = 1 体も触らない(L1 バイト一致)"),
    # 歩車信号の同一化(実装 src/society/world/traffic.py)。★**挙動変更トグル**。
    # strict の根拠: 静的データ(交差点表)だけの純関数で、乱数 stream を 1 本も引かず
    #   LLM も呼ばない(同じ地図なら何度組んでも同じ表 = 完全な決定論)。
    # affects_k=False: generate() の呼び出し点を足しも減らしもしない。
    # fingerprint_risk=none: プロンプトへ 1 バイトも足さない(車の走行にしか出ない)。
    _f("world.traffic.signal_from_crossings", "strict", False, "none",
       "od モードの**車側**信号を、歩行者側と同じ交差点表(data/crossings_shibuya.json)"
       "から導く。既定 OFF では同じ交差点なのに車(赤率 = 0.35+0.20×sha256(node)・周期は"
       "conf の全交差点共通の定数)と歩行者(実測 140/37/10)が**統計的に独立**に動いている。"
       "ON では OSM node id の完全一致で結合できたノード(実測 69 本中 23 本)だけ、"
       "赤率 r = 1 − (green_s+flash_s)/cycle_s(140/37/10 → 0.664)とその交差点自身の周期を"
       "使う。仮定は『車の青窓 ≈ 歩行者の横断可能窓』の 1 つだけで、方向別スプリット等は"
       "データが無いので主張しない。★これは観測ではなく**走行が変わる挙動変更**であり、"
       "主張できるのは『同一交差点の車と歩行者が同じ 1 台の装置になる』ことに限られる。"
       "結合できないノードは従来の hash 値のまま(欠測を埋めない)"),

    # ---- 駅員・車掌 = 持ち場を持つアクター(actor model P3a。実装 src/society/transit_staff.py)----
    # strict の根拠: 当直の選定(最小 agent id)も合成当直表(名簿順 index の純関数)も
    #   停車時間の演算も**完全決定論**で、乱数 stream を 1 本も引かない
    #   (module に乱数の識別子が存在しないことを tests/test_transit_staff.py が AST 固定)。
    # affects_k=False: generate() の呼び出しサイトを 1 つも足さず・減らさない。
    # fingerprint_risk=none: **プロンプトへ 1 バイトも足さない**(遅延は従来どおり
    #   「動けない」という世界の事実としてしか個体に届く経路が無い)。
    _f("transit_staff.enabled", "strict", False, "none",
       "駅員・車掌アクター層の親トグル。監査事実『電車は実体ではなく(has_service は"
       "時刻表の述語)、遅延を作っていたのは主体なしの近似(env.feedback 規則1)と"
       "日次サイコロ(disaster)だけ』の穴を塞ぐ。ON では駅員/車掌/電車運転士 が"
       "駅ノードの持ち場へ束ねられ(work.bind_workplace が no_category で束ねられずに"
       "残していた層)、**当直の車掌のドア閉判断**が L1 dwell_decision として世界に残る。"
       "既定 OFF は束ね 0 件・dwell_decision 0 件・属性も生えない(L1 バイト一致)"),
    _f("transit_staff.bind.enabled", "strict", False, "none",
       "駅員・乗務員を駅ノードの持ち場へ束ねる(決定論・冪等・resume 安全)。合成当直表は"
       "名簿順(agent id 昇順)index の純関数で、既に勤務窓を持つ個体の窓は上書きしない。"
       "false にすると『判断だけ動かして束ねない』対照が作れる"),
    _f("transit_staff.bind.rebind_daily", "strict", False, "none",
       "束ねの**日次追随**(親 transit_staff.bind.enabled 配下)。既定 false は従来どおり"
       "起動時 1 回きり = 駅員・乗務員は名簿(L5)の duty 層なので 100万プールの在場"
       "ローテーションで毎日入れ替わり、途中入場した個体が駅に立たないまま日が進む"
       "(= ドア閉判断 on_duty_crew が unstaffed へ倒れていく)。true にすると日境界に"
       "bind を通し直す(既に駅へ立っている個体は n_kept として素通し = 冪等・"
       "L1 の transit_staff_bound は宣言どおり step0 の 1 件のまま・乱数ゼロ)。"
       "street_life.rebind_daily / city_ops.rebind_daily と同型(第110 レーン PRES-C)"),
    _f("transit_staff.dwell.enabled", "strict", False, "none",
       "車掌のドア閉判断(ホーム負荷 → 停車時間延長 → delay_min)。ON のとき "
       "env.feedback 規則1 は**評価されない** = 同じ物理(停車時間 → 遅延)の二重計上を"
       "防ぐ主体つき側の優先(改札装置が規則2 を置き換えるのと同じ線引き)。式・閾値・"
       "上限・回復運転項は規則1 と**同一の関数**を呼ぶので、既定サブフラグでは"
       "『誰が決めるか』だけが変わり数値は 1 ビットも変わらない"),
    _f("transit_staff.dwell.require_service", "strict", False, "none",
       "運行が無い step(終電後・運休)は停車時間を足さない(閉めるドアが無い)。"
       "規則1 は運行の有無を見ずに注入していたので、ここだけが意図的な差である。"
       "false にすると規則1 と全 step で完全同値になる"),
    _f("transit_staff.dwell.calibrated", "strict", False, "none",
       "停車時間の係数を東京の実測則(Palmqvist/Tomii/Ochiai 2020: **+15 人/車両 ≒ "
       "+1 秒**・通勤列車の総遅延の約 9 割が 5 分以下の停車時間超過)へ差し替える。"
       "既定 false = env.feedback.transit.dwell_sec_per_pax(legacy の既定係数)のまま"
       "= 既存の数値を 1 も動かさない。差し替わるのは**係数だけ**で、式・上限・"
       "回復運転項は共有のまま"),

    # ---- ラッシュ時の車内 = シミュレートされた空間(Wave 4 II-1。実装 transit_interior.py)----
    # strict の根拠: 車両選択(誤差項なしの argmax)・座席抽選・区画割当・車内行動の
    #   いずれも **(run.seed, agent id, 日, 到着分) の安定ハッシュ**の純関数で、乱数
    #   stream を 1 本も引かない(module に stream / hub / random の識別子が存在しない
    #   ことを tests/test_transit_interior.py が AST 固定)。
    # affects_k=False: **generate() の呼び出しサイトを 1 つも足さない**。乗車中の個体は
    #   loc=="outside" なので scheduler の active 母集団に構造的に入らない
    #   (= 車内で LLM は 1 本も撃たれない)。会話経路も 1 本も作らない。
    # fingerprint_risk=possible: 降車後の記憶に定型 1 行が増える(traces と同じ等級)。
    _f("transit_interior.enabled", "strict", False, "possible",
       "ラッシュ時の**車内**を空間にする親トグル。監査事実『駅からの流入は瞬間移動で、"
       "混雑率 111〜167% の車内に居た十数分が世界に存在しない』を塞ぐ。ON では "
       "world.inflow_pulse が作った (路線, 到着分) の群 = **同じ 1 本の列車の乗客**へ "
       "車両(編成・定員・座席・扉数は路線別=conf の表。★銀座線だけ 16 m 3 扉)と "
       "区画(ドア脇 / 座席前 / 通路)を決定論で割り当て、降車で train_ride / "
       "train_copresence / train_patrol を残す。★**会話は 1 本も増やさない**(車内は"
       "静かな同席 = スマホ 70〜84%・見知らぬ他人との会話はほぼゼロ・騒々しい会話は"
       "迷惑行為の第 1 位 51.8%)。同席は記録するだけで、そこから何かが起きるかは既存の"
       "関係機構が街の中で決める(familiar strangers は通勤の規則性から自然に生まれる)。"
       "★world.inflow_pulse.enabled が OFF のときは本トグルを ON にしても動かない"
       "(列車ごとの塊が立たないので『列車の形をしていない列車』になるため)。"
       "既定 OFF は train_* 0 件・属性も記憶も state も生えない(L1 バイト一致)"),
    _f("transit_interior.include_plan_returnees", "strict", False, "possible",
       "計画駆動の圏外帰還者(planning.day_plan.boundary)も駅から帰るなら列車に乗せる。"
       "★帰還時刻そのものは **1 ビットも動かさない**(その時刻に着く列車の名前を"
       "ダイヤから**読むだけ**で後付けする)。false にすると流入通勤者だけが車内を持つ"
       "対照が作れる", True),
    _f("transit_interior.memory.enabled", "strict", False, "possible",
       "降車後に読む圧縮記憶 1 行(乗車 1 回につき 1 行だけ)。文面は (路線名, 混雑段, "
       "行動) だけの純関数で、数字・実験条件・k・機構語を 1 バイトも含まない"
       "(混雑は『空いていた/混んでいた/ぎゅうぎゅうだった』の**段の語**でしか出さない = "
       "個体は混雑率という統計量を知らない)。false = 車内の物理だけ回して"
       "プロンプトへ 1 バイトも足さない対照", True),
    _f("transit_interior.copresence.enabled", "strict", False, "none",
       "同じ車両に居合わせた対を L1 へ記録する(**1 日 1 対 1 件**・総量上限つき)。"
       "familiar strangers(顔見知りの他人)の**素材**であって、ここから関係も会話も"
       "起こさない(PNAS 2013 の枠組み = 通勤の規則性から自然に生まれるものを"
       "こちらから出会わせない)。プロンプトには 1 バイトも入らない", True),
    _f("transit_interior.copresence.adjacent_only", "strict", False, "none",
       "同一/隣接区画の対だけを記録する(= 相互作用の可能性がある距離に限る)。"
       "既定 false = 同一車両の全対を記録し、隣接かどうかは payload の "
       "zone_adjacent 欄で残す(解析側で切れる)"),
    _f("transit_interior.conductor.enabled", "strict", False, "none",
       "車掌の車内巡回(受け持ち列車の車両を patrol_interval_steps ごとに 1 つ進む)と、"
       "その車両の乗客との同席記録。★transit_staff.enabled も ON でないと動かない"
       "(乗務員という個体が居ないため)。車内負荷を停車時間へ足すかは "
       "conductor.dwell_load_weight(既定 0.0 = 観測のみ)が別に決める", True),

    # ---- indoor ----
    _f("indoor.enabled", "strict", False, "none",
       "空間レイヤ核(全建物・全階の間取りと区画用途型を決定論保持)"),
    _f("indoor.markov.enabled", "strict", False, "none",
       "フロア内の区画滞在→遷移(幾何 dwell)"),
    _f("indoor.sfm.enabled", "strict", False, "none",
       "区画内の微視軌跡(Social Force Model)"),
    _f("indoor.meeting.enabled", "strict", False, "none",
       "同一職場グループを同時刻に meeting 区画へ集める会合"),
    _f("indoor.encounter.pairing", "strict", False, "none",
       "前 step の屋内遭遇ペアを対面会話の返答相手選択で優先(相手選択のみ=呼数不変)"),
    _f("indoor.tracks.enabled", "strict", False, "none",
       "秒スケール軌跡+遭遇の記録サイドカー(parquet。記録専用)"),
    _f("indoor.los.enabled", "strict", False, "possible",
       "屋内の同席リストを実座標距離+間仕切り壁 LOS で絞る(知覚の区画粒度化)"),

    # ---- physics(竹-4 = P3 境界縫合。既定 OFF)----
    _f("physics.zones_enabled", "strict", False, "none",
       "局所物理ゾーン(P3 境界縫合)。ゾーンのポリゴンを通り抜ける経路の個体を step の頭で"
       "物理が所有し(グラフ移動を止め)、dt_sub 刻みで実際に歩かせて、ゾーン外へ出たら経路へ"
       "射影して返す。エンジンはゾーン別(sfm|orca。P2 決定=多方向交差流だけ orca)。"
       "★ON は世界を変える(位置・到着時刻が物理で決まる)opt-in 機能。"
       "repro_tier=strict: 固定 dt・agent.id 昇順・集約順序固定・乱数は用途別 stream "
       "\"physics\" のみ(ORCA pref_noise / SFM ξ。0 なら 1 本も引かない)。"
       "affects_k=false: generate() の呼び出し点を 1 つも足さない/減らさない。"
       "混雑で滞在が伸びれば同席の相手・タイミングが変わり発火数が**間接的に**動くが、"
       "それは registry の規約どおり False 側(位置・ゲージ経由の間接効果)。"
       "fingerprint_risk=none: プロンプトの語彙・欄が 1 つも増減しない"
       "(第85 契約の body 欄は prompt_kwargs() に出ない構造)。"
       "物理で位置が変われば見える景色が変わるのは**正当な世界変化**であって実験条件の漏洩ではない"),
    _f("physics.perception.channels", "strict", False, "none",
       "発火チャンネル ext.crowd_local の値を物理の実測近傍人数へ差し替える(**配線のみ**)。"
       "ON にすると第80 の σ 較正がやり直しになるため既定 OFF で据え置く"),
    # ---- physics 較正(P4-2 / P4-3。既定 OFF = 現行挙動と数値的に恒等)----
    _f("physics.sfm.far_field.enabled", "strict", False, "none",
       "ゾーン SFM に**長距離 social 項**(VISSIM 製品版の 2 項構造のうち本リポジトリに"
       "欠けていた側)を足す。f2_ij = m·a2·exp((r_i+r_j−d)/b2)·w(φ)·n_ij。"
       "カットオフ規則(cutoff_factor·b2 + C¹ テーパー)は a2 とセットでしか意味を持たない。"
       "repro_tier=strict: 位置と半径だけの純関数で乱数も LLM も 1 本も増えない。"
       "affects_k=false: generate() の呼び出し点を足さない/減らさない。"
       "fingerprint_risk=none: プロンプトの語彙・欄が 1 つも増減しない(物理層)。"
       "★既定 OFF では `_CalibratedCrowd` を構築すらせず sfm_core.Crowd を"
       "従来の引数のまま作る = golden L1 バイト一致"),
    _f("physics.sfm.v_of_s.enabled", "strict", False, "none",
       "ゾーン SFM の**駆動項の希望速度だけ**を Tordeux 型 V(s)=min{v0,max{0,(s−l)/T}} へ"
       "差し替える(s = 進行方向の最近前方者までの中心間距離)。方向モデル・斥力・v_max は"
       "SFM のまま = JuPedSim の Collision-Free Speed Model と同じ発想の最小適用。"
       "repro_tier=strict / affects_k=false / fingerprint_risk=none は far_field と同じ理由。"
       "★既定 OFF では V(s) を 1 度も評価しない = golden L1 バイト一致"),

    # ---- economy ----
    _f("economy.enabled", "strict", False, "none",
       "経済 v0(賃金・消費・バイト・金銭圧力の心理接続)"),
    _f("economy.visitor_refresh", "strict", False, "none",
       "来街者が帰宅から戻るたび手持ちを補充(長期ランの恒久破綻対策)"),
    _f("economy.accounts.enabled", "strict", False, "none",
       "口座(銀行)概念。月給・家賃引落・カード/現金・ATM・立退き/破産サイクル"),
    # ---- 賃金多様性 WAGE(第112バッチ 2026-08-13。ユーザー要求)----
    # strict の根拠: 金額・支給形態・給料日・賞与のすべてが (台帳, 安定キー) の blake2b 純関数で、
    #   **乱数 stream を 1 本も引かない**。LLM の自由文は 1 バイトも読まない。
    # affects_k=False: generate() の呼び出しサイトを 1 つも足さず・減らさず、プロンプトの節も
    #   1 つも増やさない(賃金額は観測層と会計にしか現れない)。
    # fingerprint_risk=none: プロンプトを 1 バイトも変えない。所持金は元々世界の観測量で、
    #   ON/OFF の別が当人に見える経路は存在しない。
    _f("economy.wage_profile.enabled", "strict", False, "none",
       "L2(域内従業者 224,240 人のうち 198,264 人)の本業日給が構造的に 0 円だった穴を塞ぐ。"
       "月額 = 産業(14種)× 職種群 × 規模帯(8区分)× 個体差(±15%・安定ハッシュ)で決め、"
       "日額 = 月額/所定労働日数 を既存 agent.wage に載せ替える。支給は新設の日次清算フェーズが"
       "唯一の点になり(_settle_work の建物退館依存を外す=路面の職場・夜勤でも賃金が出る)、"
       "月給者は給料日にまとめ・日給者は働いた日に受け取る。給料日は職場ごとに "
       "{10,15,20,25,月末} へ分散(25 日が最頻)。不在中に過ぎた給料日は再来街時に遡って支給。"
       "既定 OFF はプランを 1 つも作らず支給点も勤務日カウントも通らない(ゴールデン L1 バイト一致)"),
    _f("economy.wage_profile.calendar", "strict", False, "none",
       "給料日・賞与月を**実暦**(world.calendar の日付)で判定する。false は口座 E5 と同じ"
       "「run 開始日=1日」の 30 日周期。★実暦にしないと block_day%30+1 では 8/16 開始・"
       "10 日ランで 25 日に永遠に到達せず、月給者が 1 円も受け取れない(第111 で発見した既存バグ"
       "と同型)。既存 _phase_accounts_day の式は 1 バイトも触らずこの新経路の中だけで切り替える"),
    _f("economy.wage_profile.bonus.enabled", "strict", False, "none",
       "賞与(夏 6-7月 / 冬 12月)。支給する職場の割合も支給月数も**規模帯**で決まる"
       "(毎月勤労統計の実態)。支給日は給料日と重ならない日を選ぶ(同一 step に賃金を 2 本"
       "重ねない=源泉税の帰属突合が壊れない)。★本選の窓 8/16-8/26 では発火しない(8月に"
       "賞与は出ない=現実どおり)。機構とテストだけ本番投入可能にしておく"),

    # ---- IF-E2 案B: org の会計主体化 + rest-of-world(設計 src/society/economy_sfc.py /
    # docs/research/ifE2-org-accounting-research.md §4-3)----
    # strict の根拠: 初期残高(配属者の日給合計 × month_days × σ)も受け手解決(台帳の静的索引)も
    #   **完全決定論**で、乱数 stream を 1 本も引かない。LLM の自由文は 1 バイトも読まない。
    # affects_k=False: generate() の呼び出しサイトを 1 つも足さず・減らさず、プロンプトの節も
    #   1 つも増やさない(残高は観測層だけの概念)。
    # fingerprint_risk=none: **プロンプトを 1 バイトも変えない**。org の残高・当座借越・域外依存度が
    #   エージェントから観測できる経路は存在しない(記憶にも 1 行も入らない)。
    #   ★「自社の資金繰りを経営者エージェントの判断に効かせる」のは将来の拡張であり、
    #     やるときは affects_k / fingerprint_risk の再評価が要る。
    _f("economy.org_accounting.enabled", "strict", False, "none",
       "IF-E の実測(漏れ 71.8〜96.0% = 財布から出た金の受け取り手が世界に居ない)を閉じる。"
       "org へ**スカラー預金 1 本**を新設し(Lengnick 2013 の M=wN・Caiani et al. 2016 の "
       "σ×予想賃金支払で初期化)、賃金をそこから支払い(不足時は**自動当座借越**=負残高許容+"
       "L1 org_overdraft。破綻処理は入れない)、消費を台帳の静的索引で解決した受け手 org へ"
       "**実支払−消費税**として入金する。受け手が街に居ない金は rest-of-world(渋谷域外)へ"
       "**チャネル別に**落とす(SNA 2008 §26.2/26.5/26.6 = 行動方程式なし・残高制約なしの明示部門)。"
       "これで **Σ(全主体残高)+RoW 累積 = 一定**という閉じた不変量が立つ(ABCredit.jl 流の"
       "スカラー総マネー保存を tests で常設)。既定 OFF は残高・state・L1・payload キーの"
       "いずれも生えない(ゴールデン L1 バイト一致)"),
    _f("economy.org_accounting.export_production", "strict", False, "none",
       "域内のどの消費カテゴリからも 1 円も受け取れない org(台帳の POI 種別 office/education = "
       "全 org の 36.5%・従業者の 51.5%)へ、既存 revenue_est(日給×revenue_margin)を"
       "**RoW からの輸出代金**として入金する(式も発火点も 1 つも変えず相手方だけ void→RoW へ)。"
       "地域会計で標準の扱い(観光サテライト勘定の逆向き)。false=輸出なし=当座借越へ沈む"),
    _f("economy.org_accounting.seed_from_pool_roster", "strict", False, "none",
       "org の期首預金を **day0 在場者**でなく **束ねられた名簿全体**(record に org_id を持つ"
       "全ペルソナ)の日給合計から与える(レーン乙 D1)。pool ON では在場 cap が名簿の 1/4 なので、"
       "現行は『たまたま初日に街に居た人数』が会社の運転資金を決めており、期首から構造的に過少で"
       "初日の給与支払いで当座借越に落ちる org が出る。日給は economy.wage_plan(record と pid だけの"
       "純関数)で在場個体に付く agent.wage と同値。走査はシャードのストリーム読み 1 周・乱数ゼロ。"
       "pool OFF では母集団が一致するので値は変わらない"),
    _f("economy.org_accounting.sidecar", "strict", False, "none",
       "部門別残高の日次サイドカー finance.parquet(org/bank/VC/行政/供託/家計/RoW)を書く。"
       "analyze_accounting.py の検査①が org・銀行・行政・RoW でも成立するための観測点。"
       "記録と動力学の分離(動力学は add_rows/flush/finalize しか呼ばない)"),

    # ---- tools / rules / recursion(LLM が文字列と rule JSON を書き込む層)----
    _f("tools.enabled", "journal", False, "possible",
       "世界を変えるツール群(出店・提案・イベント・ビラ・グループ)。提案文/グループ名など"
       "LLM の自由文がそのまま世界状態になるため journal"),
    _f("tools.equip_all", "strict", False, "possible",
       "全発火プロンプトに『所持ツール』節を中立告知(提示のみ=呼数不変)"),
    _f("rules.enabled", "journal", False, "possible",
       "制度DSL。LLM が propose で書いた rule JSON を検証して実効ルールに制定する"),
    _f("rules.allow_declare", "journal", False, "possible",
       "宣言型 rule(権利創設など象徴的制度)を許可する"),
    _f("recursion.enabled", "strict", False, "possible",
       "再帰性(実効の取り決め・昨日の街の動きの知覚→フィードバック)のマスター"),
    _f("recursion.rules_in_prompt", "strict", False, "possible",
       "いま実効の取り決めを全発火プロンプトへ1行注入"),
    _f("recursion.digest_in_prompt", "strict", False, "possible",
       "昨日の街の動き(提案/成立/廃止/取締件数)を1行注入"),
    _f("recursion.allow_repeal", "journal", False, "possible",
       "LLM が書いた repeal 提案(既存の取り決めの廃止)を受理する"),
    _f("recursion.impact_news", "strict", False, "possible",
       "取り締まりが多発したルールを日次ニュース化する"),

    # ---- net / drive / conversation / planning / routine / prompts ----
    _f("net.enabled", "strict", False, "none",
       "インターネット層(SNS 閲覧・ニュース・フォロー・いいね/リシェア)"),
    _f("drive.boredom.enabled", "strict", False, "none",
       "退屈ゲージ(同じことの反復で発火閾値が下がる)"),
    _f("drive.drift.enabled", "strict", False, "none",
       "欲求パラメータの日次ドリフト"),
    _f("conversation.enabled", "strict", False, "none",
       "会話3層 C2/C3。LLM を1本も使わない構造化会話(Dialogue Act 遷移)で会話密度を作る"),
    _f("planning.enabled", "journal", True, "possible",
       "朝の一日計画。LLM に予定 2〜5 件を立てさせ自由文の what/intent を行動の土台にする"),
    _f("planning.framework.enabled", "journal", False, "possible",
       "計画を型スキーマ+コンパイラへ拡張(LLM の計画出力を構造化して読む)"),
    # 第86バッチ day_plan v1。journal の根拠: 機能の本体が **LLM の自由文/構造化出力を
    # データとして消費する**(ブロック列がそのまま日中の行動になる)。
    # ★affects_k=True を正直に宣言する: 既定(cognition.fire OFF)では朝 1 呼のまま呼数は
    #   1 本も変わらない(再試行なし・修復とフォールバックは全て決定論)が、fire ON では
    #   must 危機の再計画が **内部発火 plan_exception を認知イベントキューへ前倒し登録**する
    #   = generate() の発生点そのものを動かす。間接経路ではなく明示的な作用点なので True。
    # fingerprint_risk=possible: 朝のプロンプトの計画タスクが別書式に差し替わる
    #   (当人から見て「今日の書き方が違う」= 差分に気づく余地が原理的にある)。
    _f("planning.day_plan.enabled", "journal", True, "possible",
       "day_plan v1(構造化された朝の計画)。ブロック 4〜8 + contingency ≤3 の列挙型スキーマ・"
       "スキーマ/物理検証→決定的修復→前日計画/既定ルーチンへのフォールバック・"
       "priority×flex による割り込み(could は LLM を呼ばず削り must 危機だけ再計画)"),
    # 第93バッチ IF-A: contingency(if_then)の消費。第86 は書かせて格納するだけで
    # **消費するコードが無かった**(監査 §5-1「デッドデータ」)ので、その穴を塞ぐ子トグル。
    # journal 等級の根拠: 適用する条件・対処が **LLM の構造化出力**(plan["cont"])に由来する
    #   = 親の day_plan.enabled と同じ理由(判定規則そのものは決定論)。
    # affects_k=False: generate() の呼び出し点を 1 つも足さない(評価も適用も完全決定論・
    #   乱数ゼロ)。行き先が変われば co-location 経由で発火数が間接的に動きうるのは
    #   commerce / health / disaster と同型で、レジストリ冒頭の方針どおり False 側。
    # fingerprint_risk=none: **プロンプトを 1 バイトも変えない**(if_then の書式は
    #   day_plan ON なら本トグルの ON/OFF と無関係に schema_prompt へ載る)。
    # actor model P4: 計画駆動の圏外滞在。指名(誰が圏外通勤者か)も時刻表(いつ出て
    #   いつ帰るか)も**決定論**((run.seed, agent.id) の純関数 + ブロックの start/end)
    #   なので strict 等級。LLM の自由文は 1 バイトも読まない(親の day_plan が組んだ
    #   ブロックの place/start/end という**構造化された欄だけ**を見る)。
    # affects_k=True を正直に宣言する: 圏外の個体は _phase_drive の母集団から外れるので
    #   **その間の発火機会がゼロになる** = generate() の総数が確実に動く(間接経路ではない)。
    # fingerprint_risk=none: プロンプトを 1 バイトも変えない(schema_prompt も朝の計画文も
    #   本トグルの ON/OFF と無関係)。圏外に居る間はそもそも 1 呼も発行しない。
    _f("planning.day_plan.boundary.enabled", "strict", True, "none",
       "計画駆動の圏外滞在(actor model P4 境界)。勤務先が街の外にある居住者を決定論で"
       "指名し、その計画の work ブロックに boundary 欄(exit/gateway/exit_min/entry_min)を"
       "刻む。**その欄がそのまま despawn/respawn の時刻表**であり境界機構は新設しない"
       "(退出=既存 _try_exit・帰還=既存 _phase_wake_and_returns・L1 も exit_area/enter_area を"
       "再利用し payload に boundary='plan' を足すだけ)。圏外ではイベント 0 件・乱数 0 本・"
       "LLM 0 呼で、帰還時に定型の圧縮記憶 1 行だけが残る。統計駆動の流入通勤者(commute)"
       "とは別台帳(payload の boundary 欄で機械的に分離できる)"),
    _f("planning.day_plan.use_contingency", "journal", False, "none",
       "day_plan の contingency(if_then)を実行時に消費する。ブロック実行の直前に "
       "CONDS 7 種を決定論評価し(前提機構が OFF の条件は常に不成立=捏造しない)、"
       "最初に成立した 1 件だけ THENS 5 種を適用して plan_cont_fire を 1 件記録する。"
       "1 ブロックにつき発火は高々 1 回(振動しない)"),
    _f("cognition.engaged.enabled", "journal", True, "possible",
       "engaged モード = AUTOPILOT / ENGAGED の 2 状態機械(第87バッチ。設計 §8)。"
       "第81 fire が答えた『いつ考えるか』(= 点)の上に『いつまで考えるか』(= 区間)を"
       "載せる層。突入 5 条件(S>θ_in / 社会的直接性 / 実行不能例外 / 欲求臨界 / 予定思考)と"
       "脱出 4 条件(解消 / 減衰=ヒステリシス θ_out<θ_in / ターン上限 / プリエンプト)。"
       "★affects_k=true: エピソードが既存発火を束ねるので LLM 呼数が変わる。ただし"
       "**generate() の呼び出しサイトは 1 つも増やさない**(呼ぶのは従来どおり reply / "
       "発話 / 内省の 3 経路だけ)。定型応答経路はむしろ呼数を減らす向きに効く。"
       "★journal 等級: 会話の終結を LLM の自由文出力(end 欄)から読むので事後再生に"
       "llm_cache/llm_journal が要る。"
       "★fingerprint_risk=possible を正直に宣言する: ON のとき**会話ターンのプロンプトにだけ**"
       "『切り上げるなら end と書いてよい』の 1 節が増える(= 終結の宣言路そのもので、"
       "§8 が要求した設計なので隠さない)。機構語(発火・驚き・閾値・エピソード・不応期・"
       "自動操縦)も実験条件語も因子名も 1 文字も出さない。"
       "★cognition.fire OFF では完全 no-op(watch / g_update と同じ前提関係)。"
       "★乱数 stream を 1 本も引かない(高解像度層の選定も _stable_hash の決定論)"),
    # 第94バッチ IF-B: 拒否通知の段階 conf 化(設計 src/society/reject.py /
    # docs/research/llm-world-interface-audit.md §2-C / if-lane-research.md §3)。
    # strict の根拠: 通知文は **(行為種, 理由コード) の純関数**で、LLM の自由文を
    #   1 バイトも読まない。判定材料は裁定側が既に持っている客観条件(所持金・空き住戸・
    #   近傍の有無・経路の有無)だけで、乱数 stream も 1 本も引かない。
    # ★affects_k=True を正直に宣言する: "engaged" 水準だけ、認知イベントを「今」へ前倒し
    #   する(既存 note_plan_exception と同じ作用点)= generate() の**発生点**が動く。
    #   "memory" 水準では 1 本も動かない。1 つの id で 3 水準を申告するので上位水準に合わせた
    #   (ablate.llm_off が strict × affects_k=True を取るのと同じ組み合わせ)。
    # fingerprint_risk=possible: memory / engaged では拒否の定型文 1 行が当人の記憶に入り、
    #   次のプロンプトの「直近の出来事」に載る(= 当人から見て差分に気づく余地がある)。
    #   ただし文面は有限語彙の純関数なので**実験条件は 1 文字も漏れない**。
    _f("cognition.rejection_notify", "strict", True, "possible",
       "無音だった裁定拒否 8 件(所持金不足の出店 / 敷金不足の転居 / 空き住戸なし / "
       "交際申込の相手不在 / 経路が張れない / verify の no_target・no_witness・no_channel)へ "
       "silent / memory / engaged の 3 水準の通知を入れる(Inner Monologue の "
       "なし・記述・介入に 1 対 1)。memory=定型文 1 行 + L1 action_reject、"
       "engaged=それに加えて認知イベントの前倒し(cognition.engaged OFF では memory へ縮退)。"
       "day_plan の plan_exception には失敗理由を 1 行だけ載せ、**再計画が実ったら降格**する"
       "(Masicampo & Baumeister 2011。Zeigarnik 効果は 2025 メタ分析で否定済みなので根拠にしない)。"
       "既定 silent は現行と 1 バイトも変わらない(イベント 0 件・記憶 0 行・プロンプト不変)",
       off_value="silent"),
    # 第95バッチ IF-C: 情報オブジェクトの一般化 = 噂(設計 src/society/rumors.py /
    # docs/research/if-lane-research.md §2 / llm-world-interface-audit.md §3)。
    # strict の根拠: 噂の**誕生**は L1 の有限種の構造化イベントだけを源にし、文面は
    #   (源イベント種, 場所名 or 人名) の純関数。源イベントの自由文(title / name)は
    #   1 バイトも読まない。**伝播**は既存の会話接触に相乗りするだけで、発話テキストを
    #   読みも書き換えもしない。**停止**(Maki-Thompson の stifler 化)は接触順だけで決まる
    #   完全決定論で、乱数 stream を 1 本も引かない。
    # ★affects_k=False: generate() の呼び出しサイトを 1 つも足さず・減らさず・プロンプトの
    #   **節**も 1 つも増やさない。増えるのは既存「直近の出来事」欄の中身だけ(flyer / news /
    #   dm と同じ帯域)で、そこ経由の間接的な発火数の揺れは本レジストリの規約どおり False。
    # fingerprint_risk=possible: 伝聞の定型文 1 行が聞き手の記憶に入り次のプロンプトに載る
    #   (当人から見て差分に気づく余地がある)。ただし文面は有限テンプレの純関数で、
    #   実験条件も機構語も数値も 1 文字も漏れない。**item_id は絶対にプロンプトへ入れない**。
    # ★正直な相互作用: ON のランでは凍結指標 c_transmission / n_transmission /
    #   transmission_novel_rate(measure.py / stream.py / echo.py = 凍結 14 ファイル)に
    #   噂の transmission が混ざる(あちらは item の kind を見ない)。既定 OFF では無風。
    _f("information.rumors.enabled", "strict", False, "possible",
       "Item.kind=\"rumor\" の宣言済み枠を実体化する。構造化イベント(既定 = event_host / "
       "venture_open / enforcement の公共 3 種)から定型文の噂 Item を起こし、当事者と"
       "同席者を初期 knower にする。伝播は既存の speak/dm に相乗りして transmission"
       "(= provenance の既存 API)に載り、聞き手の記憶に定型文 1 行が入る"
       "(**発話テキストは 1 バイトも変えない = 別チャネルの伝聞**)。"
       "停止規則 stifle=mt(既定)は Maki & Thompson 1973 = 既に知っている相手へ語ったら"
       "以後語らない(= 噂が止まる唯一の理由。理論的極限は最終 ignorant≈20%)。"
       "指標は OASIS に揃えた scale / depth / max_breadth(解析 scripts/analyze_rumors.py)。"
       "既定 OFF は Item 0 件・イベント 0 件・記憶 0 行・プロンプト不変(L1 バイト一致)"),
    # 第96バッチ IF-D: 痕跡 = 場所のイベント履歴(設計 src/society/traces.py /
    # docs/research/if-lane-research.md §1 / llm-world-interface-audit.md §3)。
    # strict の根拠: 集約(強度加算+上限クリップ)も蒸発(半減期の決定論減衰・日境界 1 回)も
    #   完全決定論で、**乱数 stream を 1 本も引かない**。源は L1 の有限種の構造化イベントだけで、
    #   源イベントの自由文(title / name / kind)は 1 バイトも読まない。プロンプトへ出るのは
    #   (痕跡種) の純関数である定型 1 行だけ。
    # ★affects_k=False: generate() の呼び出しサイトを 1 つも足さず・減らさず、プロンプトの
    #   **節**も増やさない(増えるのは既存の 1 行欄の族に並ぶ 1 行だけ = place_label_line /
    #   crowd_line と同じ帯域)。そこ経由の間接的な発火数の揺れは本レジストリの規約どおり False。
    # fingerprint_risk=possible: 痕跡の定型 1 行がその node に居る者のプロンプトに載る
    #   (当人から見て差分に気づく余地がある)。ただし文面は有限テンプレの純関数で、
    #   **強度・件数・階層名・実験条件・機構語は 1 文字も出ない**。
    # ★propagation(拡散)は**コードとして存在しない**(Parunak の propagation factor = 0 として
    #   文献的に正当。node グラフ上の拡散は近傍列挙順依存で golden を壊しやすく、25 万体で
    #   毎 step 掃引が乗り、「隣へ滲む」は情報オブジェクト IF-C の担当だから)。
    _f("world.traces.enabled", "strict", False, "possible",
       "監査 §3 の穴(場所のイベント履歴 = L1 に座標付きで残るがシム内から読む経路ゼロ)を"
       "閉じ、node を Heylighen の言う medium にする。行為の副産物として node × 痕跡種の"
       "強度が増え(集約)、日境界に半減期で薄れ(蒸発)、**その場に居合わせなかった者**の"
       "プロンプトへ定型 1 行(最強 1 件のみ)が入る。TTL は 3 階層 = transient(揉め事・"
       "半減期 ~3 時間)/ daily(出来事・摘発・144 step = 既存 flyer と同水準)/ "
       "persistent(場所に定着した性格・減衰なし)。単一 TTL を全種に使うのは文献に反する"
       "(Heylighen: 減衰速度は情報が陳腐化する速度に合わせる。古い痕跡は無関係ではなく誤誘導)。"
       "**演算は集約と蒸発の 2 つだけで拡散は無い**(Parunak factor 0)。"
       "既定 OFF は trace_mark 0 件・state なし・プロンプト不変(L1 バイト一致)"),
    # H3(身体と事件のレイヤー): 遺失物ループ = **完全内生の事件**
    # (設計 src/society/lost_property.py / docs/plans/body-incident-layer-plan.md §3・§4)。
    # strict の根拠: 新 stream は "lost_drop" 1 本だけ((agent, step) 個体別キー = 編成順非依存)で、
    #   それ以外の分岐(気づく/拾う/届ける/着服/引き取る)は **乱数を 1 粒も引かない**
    #   決定論(性格 traits・規範状態 states・金額・監視者の純関数 × sha256 ジッタ)。
    #   既定 OFF では stream も引かない = 乱数消費不変。
    # ★affects_k=False: generate() の呼び出しサイトを 1 つも足さず、プロンプトの**節**も
    #   増やさない(増えるのは既存の記憶欄に並ぶ定型 1 行だけ = street_life と同じ帯域)。
    # fingerprint_risk=possible: 記憶の定型 1 行が当事者のプロンプトに載り、所持金が動く
    #   (= FixedLLM で ON≠OFF になりうる。career / crowd / chance と同型)。ただし文面は
    #   (出来事, 品目) の純関数で、金額・件数・確率・実験条件・機構語は 1 文字も出ない。
    _f("lost_property.enabled", "strict", False, "possible",
       "計画書 §3 の「遺失物」= **完全内生の事件**(本命)。落とす(移動・飲酒・混雑の共変量つき"
       "ハザード)→ 気づく(品目別の遅れ + 遺失届)→ 拾う(落下地点の共在 = sim.percept_index)"
       "→ **届ける / 無視 / 着服**(すべて行為)→ 交番/駅事務室 → 返還 + 報労金 5-20%"
       "(遺失物法 28 条・持ち主 → 拾得者の直接授受)/ 3 か月時効で拾得者取得(同 7 条)。"
       "較正 = 都実測 拾得 454 万件/年を渋谷来街者スケールへ換算(≈2.9e-3 件/人日)。"
       "検証ターゲット = **品目別返還率**(傘 ≈1% / 財布 60-80% / 携帯 87% = 東京都遺失物"
       "センター実測)。★これが §4 の chance_event 溶解の第一歩 = **拾得金は必ず誰かの"
       "drop から**出ることを構造で保証する(貨幣保存 = IF-E ゼロ和検査と整合)。"
       "着服は crime を出さず payload の offense / guardians(RAT の監視者数)で宣言する"
       "(diversity の較正済み窃盗系列と混線させないため)。"
       "既定 OFF は lost_* 0 件・state なし・所持金不変・プロンプト不変(L1 バイト一致)"),
    # 所有権レイヤー O1(登記簿)+ O3(相続)(設計 src/society/assets.py /
    # docs/plans/ownership-layer-plan.md §1・§2・§5)。
    # strict の根拠: 新 stream は "asset_alloc" 1 本だけ(**起動時 1 回の初期配賦**で、住戸
    #   1 戸に 1 draw)。家主 org の選定・相続の分配・売却の買い手は **乱数を 1 粒も引かない**
    #   (sha256 の純関数ジッタと id 昇順の決定論)。LLM の自由文を 1 バイトも読まない。
    # ★affects_k=False: generate() の呼び出しサイトを 1 つも足さず、プロンプトの**節**も
    #   **行**も 1 つも増やさない(第 1 段は観測層に閉じる = 当事者の記憶にも 1 行も入らない)。
    # fingerprint_risk=none: エージェント側から観測できる差分が原理的に無い。ON で動く世界状態は
    #   **死者の遺産が相続人へ移る現金だけ**で、生きている個体の残高・位置・記憶・プロンプトは
    #   1 バイトも変わらない(所有は完全に台帳の中に閉じている)。
    _f("world.assets.enabled", "strict", False, "none",
       "計画書 §1「所有されるもの」を世界の第一級オブジェクトにする層。**タグ付きレコードを"
       "登記簿に置く**ハイブリッド(ユーザーの「物にタグ」案 × Torrens/LADM の中央台帳)= "
       "権利行 (asset_id, party_id, right_kind, since) + 双方向索引(by_owner / by_asset)。"
       "O1 の対象は不動産(住戸 =(建物, 階))と車両(has_car の個体昇格)で、権利種は own 1 種"
       "(lease / permit / lien / custody は conf 宣言だけの将来枠 = O4)。"
       "初期配賦は**渋谷区の持ち家住宅率 ≈32%**(住宅・土地統計調査)で居住者 own、賃貸住戸は"
       "**域内の不動産 org**(組織台帳の industry_key=RE × sector_detail 賃貸仲介/管理・開発/PM)"
       "と RoW(域外家主)のミックス(法人所有 ≈2 割 = ユーザー決定 §5-1)。"
       "★O3 相続 = 死亡時に故人の現金 + 資産行を世帯(housemates)へ移す。相続人不存在の遺産は"
       "民法 959 条どおり国庫 = RoW の inheritance_escheat チャネルへ出す(第107 の観測盲点"
       "「死者の財布に凍結」の解消)。検証は**資産保存則**"
       "(Σ 所有者別保有数 + K5 − RoW 生成 = 初期ストック)+ 未分類移転種ゼロ。"
       "既定 OFF は台帳オブジェクトを作らず inheritance / asset_transfer 0 件・"
       "死者の財布は凍結のまま・プロンプト不変(L1 バイト一致)"),
    _f("world.assets.dwellings", "strict", False, "none",
       "住戸(=(住宅系建物, 階))だけを個別に切る子トグル(親 world.assets.enabled 配下)。"
       "false にすると台帳は車両だけになる(不動産の寄与を切る対照条件)", off_value=True),
    _f("world.assets.vehicles", "strict", False, "none",
       "車両(has_car の個体昇格)だけを個別に切る子トグル(親配下)", off_value=True),
    # Wave 4 III-3: 街の顔 = 路上の生業と条例(設計 src/society/street_life.py)。
    # strict の根拠: **乱数 stream を 1 本も引かない**(持ち場は地図の純関数・受諾は id の
    #   剰余・パトロールは同ノード判定)。LLM の自由文を 1 バイトも読まない。
    # ★affects_k=False: generate() の呼び出しサイトを 1 つも足さず・減らさず、プロンプトの
    #   **節**も増やさない(路上の出来事は既存の記憶欄に定型 1 行が入るだけ = perception_contract
    #   の追随不要)。位置・所持金経由で発火数が間接的に動くのは本レジストリの規約どおり False。
    # fingerprint_risk=possible: 路上の出来事の定型 1 行が当事者の記憶に載る(当人から見て
    #   差分に気づく余地がある)。文面は**役割だけの純関数**で、金額・件数・config・実験条件は
    #   1 文字も出ない。
    _f("street_life.enabled", "strict", False, "possible",
       "路上の生業 8 役割(ティッシュ配り/ストリートミュージシャン/キッチンカー営業者/"
       "路上占い師/街頭演説者/募金スタッフ/路上生活者/路上支援員)を持ち場と時間帯の"
       "日課で回し、渋谷区の条例(客引き禁止・過料5万円・啓発区域700m)を警告→過料→"
       "クールダウンのいたちごっことして実装する。★客引きは独立した職業にせず、"
       "夜間店舗の従業者の**決定論的な部分集合**が行う副業務にする(条例が雇い主も"
       "対象にしている構造に合わせた)。ON のとき society_diversity の迷惑行為「客引き」は"
       "供給を止める(併存ではなく置き換え=条例の効き目が測れるようにする)。"
       "★路上生活者には尊厳規約が掛かる: 中立語のみ・犯罪/迷惑機構から完全除外・"
       "権利は他と同一・区の支援事業(巡回相談→住居移行)が世界に存在する。"
       "既定 OFF は新 12 種の L1 0 件・state なし・プロンプト不変(L1 バイト一致)"),
    _f("street_life.rebind_daily", "strict", False, "none",
       "役割バインドの**日次追随**(親 street_life.enabled 配下)。既定 false は従来どおり"
       "起動時 1 回きり = 100万プールの在場ローテーションで途中入場した個体が担い手に"
       "ならず、**担い手が日を追って痩せる**(第109 の実測発見。10日ランで顕著)。"
       "true にすると日境界(_phase_pool_rotation の後)に bind を通し直し、退場者が"
       "空けた**枠**を当日の在場者で埋める(選抜規則は起動時と同一の決定論=乱数ゼロ・"
       "generate() 増分ゼロ・既に束ねた在場者は 1 バイトも触らない=冪等)。"
       "L1 の新種は 1 つも増えず、日ごとの担い手数は summary.street_life.bind_by_day に出る"),
    # Wave 4 III-1: 夜間開放 = 夜(00:00-05:00)を表現可能にする(設計 src/society/night.py)。
    # strict の根拠: **乱数 stream を 1 本も引かない**(避難先は (距離, node id, poi id) の
    #   純関数・営業時間は時刻の純関数・勤務窓は台帳値の読み替え)。LLM の自由文を 1 バイトも
    #   読まない。始発判定は既存の運行述語 transit.has_service をそのまま使う。
    # ★affects_k=False: generate() の呼び出しサイトを 1 つも足さず・減らさない。避難した個体に
    #   内省(reflect_step)を**立てない**のは意図的で、立てると夜 1 回ぶん呼数が増えるため。
    #   位置(夜に街へ残る)経由で発火数が間接的に動くのは本レジストリの規約どおり False。
    # fingerprint_risk=none: プロンプトを 1 バイトも変えない(新しい欄も行も足さない)。
    #   増えるのは L1 の night_refuge 1 種と、夜に街へ居る個体が居るという世界の状態だけ。
    _f("world.night_economy.enabled", "strict", False, "none",
       "監査が指摘した「夜が構造的に空」を 3 点で閉じる: (1) 日跨ぎシフト(work._window の"
       "『閉<=開 → open+8h』の丸めを外し 22:00→06:00 を表現可能にする。routine.in_work_window は"
       "agent.work_wraps が立った個体だけ円環判定へ落ちる)(2) POI 単位/24 時間の営業時間"
       "(cat に加え subcat を引き当てる。コンビニ 24h 等。**現行地図 v7 は subcat を持たないので"
       "その分は no-op = 地図 v8 待ちであることを正直に宣言する**)(3) 終電後の避難先"
       "(縁のゲートウェイまで歩いて世界から出る代わりに、徒歩圏の営業中 POI へ移り始発まで"
       "街に留まる。実在の渋谷: 終電 ~24:40 / 始発 04:34 / ネットカフェ・サウナが受け皿)。"
       "夜勤者そのものは生成器側(scripts/build_orgs.py --night-shifts + build_persona_pool の"
       "L2 継承)が作るので、**本トグルを ON にしても台帳/プールを再生成しない限り夜勤者は"
       "1 人も居ない**(運用上の前提)。"
       "既定 OFF は night_refuge 0 件・営業時間表 不変・勤務窓の丸め 従来どおり(L1 バイト一致)"),
    _f("world.night_economy.midnight_shift", "strict", False, "none",
       "日跨ぎシフトだけを個別に切る子トグル(親 enabled 配下)。false にすると営業時間と"
       "避難先だけが効き、work._window は従来の丸め(閉<=開 → open+8h)を文字どおり通す"),
    _f("world.night_economy.refuge.enabled", "strict", False, "none",
       "終電後の避難先だけを個別に切る子トグル(親 enabled 配下)。false にすると"
       "終電後の homing は従来どおり縁のゲートウェイへ歩いて退出する"),
    # Wave 4 III-4: 都市運営 = 見えない労働力(設計 src/society/city_ops.py)。
    # strict の根拠: **乱数 stream を 1 本も引かない**(持ち場は地図の純関数・当直表は名簿順の
    #   剰余・収集の曜日表は暦の純関数・倒れる判定は (agent id, 経過日) の安定ハッシュ)。
    #   LLM の自由文を 1 バイトも読まない。
    # ★affects_k=False: generate() の呼び出しサイトを 1 つも足さず・減らさず、プロンプトの
    #   **節**も増やさない(出来事は既存の記憶欄に定型 1 行が入るだけ)。倒れた個体は既存の
    #   睡眠機構でその場に留まるが、朝の計画は 1 日 1 回のガード(plan_day)に守られるので
    #   呼数は増えない。位置経由で発火数が間接的に動くのは本レジストリの規約どおり False。
    # fingerprint_risk=possible: 出来事の定型 1 行が当事者の記憶に載る(街路の生業と同等級)。
    _f("city_ops.enabled", "strict", False, "possible",
       "街を成り立たせているのに世界に 1 人も居なかった層を載せる: (1) 交番配置"
       "(地図の警察施設 POI へ名簿の警察官を 3 直で束ねる。patrol_every 人に 1 人は"
       "名簿どおり『巡回ユニット』のまま持ち場を持たない)(2) ごみ収集(可燃 2 回/週の"
       "**地区別曜日表**・繁華街の期限 07:30・事業系は民間業者が深夜 02:00-05:00。"
       "地区は地図の格子分割=区の実際の収集区割りは持っていないので作らない)"
       "(3) 納品の運転手化(物流① delivery_trip の agent_id が当直の納品ドライバーになる。"
       "**時刻・数量・(s,S) 判定は 1 も変えない**=変えたのは誰が運んだかだけ。不在なら"
       "unstaffed=true の正直な無人マーカー)(4) ビルの夜間清掃(office 建物へ束ね 22:00-05:00)"
       "(5) 救急(**時刻表ではなく行為の連鎖**: 既存の健康状態から倒れる collapse → "
       "近くに居合わせた誰かが通報する ems_call → 当直が応える ems_dispatch)。"
       "★自販機の補充は作らない(地図 v7 に自販機 POI が無く、無い行き先を捏造しないため)。"
       "既定 OFF は新 6 種の L1 0 件・state なし・プロンプト不変(L1 バイト一致)"),
    _f("city_ops.rebind_daily", "strict", False, "none",
       "役割バインドの**日次追随**(親 city_ops.enabled 配下)。street_life.rebind_daily と"
       "同型・同じ理由(第109 の実測発見「10日ランで担い手が痩せる」)。対象は交番の警官・"
       "収集班・納品ドライバー・夜間清掃員・**救急隊**の 5 群で、痩せると ems_dispatch が"
       "unstaffed へ倒れる。★既に枠を持つ在場者を触らないのは最適化ではなく正しさ"
       "(出動中の救急隊を日境界に待機拠点へ引き戻さない)。L1 の city_ops_bound は"
       "宣言どおり step0 の 1 件のままで、日次の数字は summary.city_ops.bind_by_day に出る"),
    _f("city_ops.police.enabled", "strict", False, "none",
       "交番配置だけを個別に切る子トグル(親 city_ops.enabled 配下)。false では警察官の"
       "持ち場を従来どおり work.bind_workplace の結果のままにする"),
    _f("city_ops.waste.enabled", "strict", False, "none",
       "ごみ収集だけを個別に切る子トグル(親配下)。false では収集班を束ねず"
       "waste_collection が 0 件になる"),
    _f("city_ops.waste.commercial_enabled", "strict", False, "none",
       "事業系(民間業者の深夜 02:00-05:00 収集)だけを個別に切る孫トグル。"
       "false では全班が区の家庭ごみ(朝)だけを回る"),
    _f("city_ops.waste.require_night", "strict", False, "none",
       "深夜班を『夜間開放(world.night_economy)ON のときだけ束ねる』かの指定。"
       "既定 true = 深夜に起きている作業員が世界に居ないランでは黙って空回りさせない"
       "(束ねない理由は L1 city_ops_bound の notes に残す)。false にすると夜間開放 OFF でも"
       "深夜の窓を与える(実際に起きているかは名簿の就寝時刻しだい=正直な近似)",
       off_value=True),
    _f("city_ops.driver.enabled", "strict", False, "none",
       "納品の運転手化だけを個別に切る子トグル(親配下)。false では delivery_trip が"
       "従来どおり agent_id=-1 の世界イベントのまま(payload も従来の 5 キーのまま)"),
    _f("city_ops.night_cleaning.enabled", "strict", False, "none",
       "ビル夜間清掃だけを個別に切る子トグル(親配下)。false では清掃員を束ねず"
       "night_cleaning が 0 件になる"),
    _f("city_ops.ems.enabled", "strict", False, "none",
       "救急の行為連鎖(collapse → ems_call → ems_dispatch)だけを個別に切る子トグル"),
    _f("city_ops.ems.require_sick", "strict", False, "none",
       "倒れる引き金を『既存の病気(health の sick)』に限るかの指定。既定 true = 健康機構が"
       "OFF(または onset_prob=0)のランでは救急の連鎖が 1 件も起きない(正直な依存関係)。"
       "false にすると既存の疲労ゲージ(fatigue>=collapse_fatigue)も引き金になる",
       off_value=True),
    # ---- 事件レイヤー H5(環境側 3 族。設計 src/society/incidents_env.py)---- #
    # strict の根拠: 乱数は**新しい named stream 2 本だけ**("incident_fire" /
    #   "incident_traffic")で、既存の draw 順に 1 本も挿さない。群集は乱数ゼロ
    #   (物理層が測った密度の閾値跨ぎ = 完全決定論)。LLM の自由文を 1 バイトも読まない。
    # affects_k=False: generate() の呼び出しサイトを 1 つも足さず・減らさず、プロンプトの
    #   **節**も増やさない(出来事は既存の記憶欄に定型 1 行が入るだけ = street_life / city_ops
    #   と同じ線引き)。位置経由の間接的な発火数の揺れは本レジストリの規約どおり False。
    # fingerprint_risk=possible: 出来事の定型 1 行が当事者の記憶に載る(city_ops と同等級)。
    _f("incidents_env.enabled", "strict", False, "possible",
       "事件レイヤーの環境側 3 族を載せる: (1) 火災(建物ハザードの薄いレート → "
       "**第一発見者 → 119 → 出場 → 鎮火** の完全アクター連鎖。誰も見つけなければ通報も"
       "出場も起きない)(2) 交通(P(事故) ∝ 歩行者流 × 車流 = **曝露の積**だけが引き金で、"
       "被害者は実在の横断中エージェント。加害車両は背景交通なので device_id=traffic:<mode>)"
       "(3) 群集(**生成しない = 状態が事件**。SFM ゾーンの密度が 4/6/13 人/m² を跨ぐこと"
       "自体を incident 化。人工的な群集注入は禁忌・雑踏警備は読み取りのみ)。"
       "★較正は 3 段(管轄アンカー → 区への換算 → 地図 bbox からの規模比)で、**重度分布を"
       "頻度と同時に較正する**(火災 ぼや175:全焼0 / 事故 軽傷92:重傷7:死亡0)= 災害映画化の防止。"
       "★停電・漏水は**作らない**(東京 SAIDI 年 13 分 = 起きないが正解)。"
       "既定 OFF は新 7 種の L1 0 件・state なし・stream 未使用・プロンプト不変(L1 バイト一致)"),
    _f("incidents_env.fire.enabled", "strict", False, "none",
       "火災だけを個別に切る子トグル(親 incidents_env.enabled 配下)。false では"
       "fire_start / fire_report / fire_dispatch / fire_out が 0 件になる",
       off_value=True),
    _f("incidents_env.traffic.enabled", "strict", False, "none",
       "交通事故だけを個別に切る子トグル(親配下)。false では traffic_accident が 0 件。"
       "★ON でも world.traffic が OFF のランでは車が 1 台も走っていない = 曝露ゼロ = 0 件",
       off_value=True),
    _f("incidents_env.traffic.signalized_only", "strict", False, "none",
       "横断部を『信号のある交差点だけ』に限るかの指定。既定 false = 車と歩行者が同じ"
       "ノードに居ること自体を曝露とみなす(信号表を持たない地図でも成立させるため)"),
    _f("incidents_env.crowd.enabled", "strict", False, "none",
       "群集の状態 incident だけを個別に切る子トグル(親配下)。false では"
       "crowd_density_incident が 0 件。★ON でも physics.zones_enabled が OFF のランでは"
       "密度を**誰も測っていない**ので 0 件(測っていない量を推定して埋めない)",
       off_value=True),
    # ---- 設備 = 摩耗する装置(設計 src/society/facility_devices.py)---- #
    # strict の根拠: 乱数は**新 stream 1 本だけ**("incident_facility")で、1 step に
    #   最大 2 draw(発生数 + 台の選択)= 台数非依存。**装置自身は 1 本も引かない**
    #   (devices.py の設計契約どおり。故障は世界からの外部入力として δ_ext へ渡す)。
    # affects_k=False / fingerprint_risk=possible(閉じ込め・通報の定型 1 行が記憶に載る)。
    _f("world.facilities.enabled", "strict", False, "possible",
       "昇降設備(エレベーター/エスカレーター)を DEVS の装置として世界に置く: "
       "wear clock(利用 δ_ext + 待機 δ_int)+ **月 1 回の保守で摩耗リセット** → 故障 → "
       "**閉じ込め → インターホン通報(行為) → 対応者の出動(行為) → 復旧**。"
       "較正アンカー = EV 閉じ込め **全国およそ 1 万件/年** ÷ 全国設置台数 ≒ 78 万台 = "
       "3.5e-5 件/台/日(地図 v7 の該当 405 棟で 0.014 件/日 = 10 日ランの期待値 0.14 件 = "
       "**何も起きない日が正しい**側の較正)。★停電・漏水は作らない(SAIDI 年 13 分)。"
       "既定 OFF は装置名簿を組まず facility_* 0 件・state なし・L1 バイト一致"),
    _f("world.facilities.elevator.enabled", "strict", False, "none",
       "エレベーターだけを個別に切る子トグル(親 world.facilities.enabled 配下)",
       off_value=True),
    _f("world.facilities.elevator.traps", "strict", False, "none",
       "エレベーターの故障で**閉じ込め**が起きるかの指定(既定 true = 現実どおり)。"
       "false にすると停止するだけになる(閉じ込めの寄与を切る対照条件)", off_value=True),
    _f("world.facilities.escalator.enabled", "strict", False, "none",
       "エスカレーターだけを個別に切る子トグル(親配下)。★故障率は**推定**であって"
       "アンカーではない(閉じ込めが構造上起きないので統計そのものが存在しない)",
       off_value=True),
    _f("world.facilities.escalator.traps", "strict", False, "none",
       "エスカレーターの故障で閉じ込めが起きるかの指定。**既定 false = 構造上起きない**。"
       "true にするのは現実に反するので、対照条件としてのみ意味を持つ"),
    # H4: 対人事件の収束化(設計 src/society/incidents_interpersonal.py /
    # docs/plans/body-incident-layer-plan.md §3 対人行 + §6-3 ユーザー決定
    # 「H4はエージェントドリブンで設計」= レート抽選の残滓を残さない)。
    # strict の根拠: 乱数は**新 named stream "incident_pair" 1 本だけ**で、それを引くのは
    #   _pair_draw ただ 1 つ。その本体の先頭は「共在ペアが空なら return」= **共在が無ければ
    #   乱数は 1 本も引かれない**(tests が AST で機械固定)。動機・標的適性・監視スコア・
    #   通報意思はすべて既存 agent 状態と世界状態の純関数で、LLM の自由文を 1 バイトも読まない。
    # ★affects_k=False: generate() の呼び出しサイトを 1 つも足さず・減らさず、プロンプトの
    #   **節**も増やさない(出来事は既存の記憶欄に定型 1 行が入るだけ = street_life / city_ops と
    #   同じ線引き = perception_contract の追随不要)。所持金・位置経由で発火数が間接的に動くのは
    #   本レジストリの規約どおり False。★report.detain_steps を既定 0 にしてあるのは、勾留が
    #   発火権を止める(= 呼数が動く)ため = 既定 ON セットでは 1 も動かさないという判断。
    # fingerprint_risk=possible: 出来事の定型 1 行が当事者・目撃者の記憶に載る(路上の生業と
    #   同等級)。文面は**出来事の種類だけの純関数**で、金額・件数・監視者数・config・
    #   実験条件は 1 文字も出ない(数値は L1 payload 側にだけ載る)。
    _f("incidents_interpersonal.enabled", "strict", False, "possible",
       "対人事件(窃盗・喧嘩)を「1人1step のレート抽選」から **共在ペアの上の条件付き確率**へ"
       "世代交代させる(Birks / Groff の RAT)。加害者候補(動機状態=金欠・滞納/立退き/破産・"
       "酩酊・疲労。**すべて一時的で回復可能** = 新しい人格ラベルを作らない)× 標的"
       "(携行現金・携帯・酩酊・孤立 = VIVA の観測可能な代理)× **監視者不在**"
       "(同席者数 + 交番近接 + traces の摘発痕跡 + 同ノードの警察官。閾値以上なら"
       "**抽選そのものを行わない**)が揃ったときだけ発火する。喧嘩は **酒 x 密度 x 閉店放出**の"
       "関数で、閉店 +1 時間の実証弾性 **+16%**(closing_boost)が検証対象。"
       "通報層 crime_report は**目撃者/被害者の行為**で、曖昧・軽微な状況にだけ人数減衰"
       "(Darley & Latane)が掛かり危険事態では掛からない = 非緊急通報・誤通報が**内生で**出る"
       "(実測 110 番の 16% が非緊急と突き合わせられる)。ON のとき society_diversity の窃盗は"
       "供給を止める(併存ではなく置き換え=RAT の効き目が測れるようにする。**乱数の消費列は"
       "1 バイトも変えない**)。既定 OFF は新 3 種の L1 0 件・crime も従来経路のまま・"
       "state なし・プロンプト不変(L1 バイト一致)"),
    _f("incidents_interpersonal.theft.supersede_diversity", "strict", False, "none",
       "窃盗の世代交代だけを個別に切る子トグル(親 incidents_interpersonal.enabled 配下)。"
       "false にすると RAT の窃盗と society_diversity のレート窃盗が**併存**する"
       "(L1 の crime 件数が 2 機構の和になるので、比較実験のときだけ使うこと)",
       off_value=True),
    _f("incidents_interpersonal.theft.enabled", "strict", False, "none",
       "窃盗(RAT の共在収束版)だけを個別に切る子トグル(親配下)。false では crime が 0 件"),
    _f("incidents_interpersonal.brawl.enabled", "strict", False, "none",
       "喧嘩(酒 x 密度 x 閉店放出)だけを個別に切る子トグル(親配下)。false では brawl が 0 件"),
    _f("incidents_interpersonal.brawl.require_intox", "strict", False, "none",
       "喧嘩の発火に『片方以上が酩酊していること』を課すかの指定(親配下)。既定 true = "
       "酒の項が無ければ喧嘩は起きない。false にすると密度と閉店放出だけで起きる対照条件になる",
       off_value=True),
    _f("incidents_interpersonal.brawl.night_only", "strict", False, "none",
       "喧嘩を『夜間店舗の窓 or 閉店放出の帯』に限るかの指定(親配下)。既定 true。"
       "false にすると昼間の混雑でも喧嘩が起きる対照条件になる",
       off_value=True),
    _f("incidents_interpersonal.guardian.police_blocks", "strict", False, "none",
       "同ノードの警察官を無条件の抑止にするかの指定(親配下)。既定 true = 既存 G3 執行の"
       "『近傍の警察官が犯罪を抑止する』線引きをそのまま引き継ぐ",
       off_value=True),
    _f("incidents_interpersonal.report.enabled", "strict", False, "none",
       "通報層(110 番 = crime_report → police_response)だけを個別に切る子トグル(親配下)。"
       "false では事件だけが起きて誰も通報しない世界になる"),
    _f("incidents_interpersonal.report.response", "strict", False, "none",
       "通報への臨場(police_response)を出すかの指定(親 report.enabled 配下)。"
       "false では通報だけが記録され、応答の有無は世界に現れない",
       off_value=True),
    # 第88バッチ 心モデル固定 + 三層知能配置(設計 §5)。
    # journal 等級の根拠: 個体ごとに**別のモデル**が自由文を書く。事後に再生するには
    #   モデル別の llm_cache / llm_journal(子ごとに 1 本)が要る = 単体では非決定。
    # ★affects_k=True を正直に宣言する: 解決層そのものは「どのモデルが答えるか」しか
    #   変えない(generate() の呼び出しサイトは 1 つも増減しない)が、**高解像度層が夜内省の
    #   選抜を第87 reflect_frac から引き継ぐ**ため、engaged ON のランでは ON/OFF で
    #   エピソード集合が変わり、結果として呼数が動く。間接ではなく明示的な作用点。
    # fingerprint_risk=none: プロンプトを 1 バイトも変えない(モデル名も層名もプロンプトに
    #   出ない)。エージェントから見て「自分がどのモデルか」を観測する経路は存在しない。
    _f("model.mind.enabled", "journal", True, "none",
       "心モデル固定(1 体 1 モデルを誕生時に固定)+ 三層知能配置。会話・思考・価値判断は"
       "その個体の固定モデルへ、機械的判断(経路・定型購買)は従来どおりルール層(LLM ゼロ)。"
       "割当は専用 stream(mind_model / mind_tier)の決定論=既存 draw 順に無風・traits 非参照。"
       "高解像度層(1〜5%)は大型モデル + 夜内省の対象 + 思考頻度の上限緩和。"
       "★agent_id→model_id の対応を manifest / summary / agents.json / L1 mind_assign に"
       "必ず残す(モデルと人格の交絡の記録=設計 §5 の明示要求)"),
    _f("routine.stochastic.enabled", "strict", False, "none",
       "確率的実行=行動のゆらぎ(骨格 motif・時刻ジッター・寄り道・中断)"),
    _f("routine.stochastic.gumbel.enabled", "strict", False, "none",
       "自由時間の行き先選択を Gumbel-max の確率選択にする"),
    _f("prompts.variety_hint", "strict", False, "possible",
       "状況文に『決まり文句で始めない』注意書きを1行足す"),
    _f("prompts.reflect_variety", "strict", False, "possible",
       "内省プロンプトの belief 説明を個体×日で決定論ローテーションする"),
    _f("prompts.interstitial.enabled", "strict", False, "possible",
       "前回発火以降の客観的な出来事の機械ダイジェストを1行注入(ナラティブ補間)"),

    # ---- transit_ride(live のみ外部プロセス=none)----
    _f("transit_ride.taxi.enabled", "strict", False, "none",
       "タクシー乗車(遠距離+所持金で低確率。専用 stream)"),
    _f("transit_ride.bus.enabled", "strict", False, "none",
       "簡易バス(合成路線での短縮移動+運賃)"),
    _f("transit_ride.bus_table.enabled", "strict", False, "none",
       "実バスダイヤの静的表(GTFS 由来のファイル)で待ち時間・区間所要を近似"),
    _f("transit_ride.live.enabled", "none", False, "none",
       "SUMO ライブ連成タクシー配車(TraCI)。**外部プロセス**の応答に到着 step が依存し、"
       "その応答は我々のジャーナルに残らない=事後再生の手段が無い"),
    _f("transit_ride.live.shared.enabled", "none", False, "none",
       "SUMO ライブ配車の相乗り・並行配車(live と同じ外部プロセス依存)"),

    # ---- lod / pool / engine / cognition ----
    _f("lod.n_proportional.enabled", "strict", True, "none",
       "LLM 発火の step 上限を人数比例 ceil(density×N)へ置換する予算制御"),
    _f("lod.input_res.enabled", "strict", False, "possible",
       "入力解像度 LOD(知覚・記憶・フィードの注入件数を水準別に振る)"),
    _f("pool.enabled", "strict", True, "none",
       "ペルソナプールの日次ローテーション(在場者の決定論選択=当日の思考対象人数が変わる)"),
    _f("pool.tier_quota.enabled", "strict", True, "none",
       "presence の層別クォータ(DP-U3 案A)。present_cap の充足を『優先順に埋めて途中 break』"
       "から『quota[t] = round(cap × 当日資格者[t] / Σ 当日資格者)』へ替え、溢れた層だけ従来と"
       "同じ当日ランキングで切る。端数は整数演算の最大剰余法(同値は層の固定順)= **追加する"
       "乱数はゼロ**・k 非依存・trait 非依存・resume 不変。"
       "★affects_k=true: 当日 present な個体の集合そのものが変わる(= 誰が LLM を呼ぶかが変わる)"
       "ため pool.enabled と同じ扱いにする。pool.enabled が OFF なら 1 バイトも効かない子トグル"
       "だが、ON 時は在場の層構成を作り替えるので独立に宣言する。"
       "既定 OFF では quota 経路を 1 度も通らない=現行の層優先と選抜集合が完全一致"),
    _f("pool.presence.mode", "strict", True, "none",
       "在場の内生化 A2(PRES)。quota(既定)= 現行の present_cap 充足経路 / emergent = "
       "**cap を一切見ない**(quota_by_ratio も通らない)= 当日の資格者がそのまま在場者。"
       "ユーザー原理『世界のアルゴリズムでエージェントの量が決まらない』の実装であり、"
       "縮退線は cap ではなく build_persona_pool --fraction(標本設計)側へ移る。"
       "★affects_k=true: 在場集合そのものが変わる(= 誰が LLM を呼ぶかが変わる)。"
       "★乱数は**減る**(溢れた層の presence_rank draw が 1 本も出ない)。"
       "pool.enabled が OFF なら 1 バイトも効かない子トグル",
       off_value="quota"),
    _f("pool.presence.habit.enabled", "strict", True, "none",
       "在場の内生化 A1(PRES)。stochastic 層の『visit_rate × 当日乱数』(= 今日来るかを"
       "世界のコインが決める)を、個体固定の位相 θ を持つ**習慣カレンダー**(Bresenham 列: "
       "present ⟺ floor(θ+r·(d+1)) > floor(θ+r·d))へ置き換える。"
       "★1 日あたりの在場確率は Bernoulli(visit_rate) と**厳密に一致**する(θ~U[0,1) より"
       "区間長 = r。period=round(1/r) のような丸め誤差が入らない)= 既存の visit_rate 較正は"
       "1 人も動かない。個人の長期レートも floor(θ+D·r)−floor(θ) で厳密保存。"
       "★乱数は**減る**: 当日 draw(stream 'presence')が消え、個体固定 stream "
       "'presence_habit' の 2 draw(位相・天候耐性)になる。k 非依存・trait 非依存・"
       "resume 不変は不変。子トグル weekday / weather は ALLOWLIST(本トグルが親)。"
       "pool.enabled が OFF なら 1 バイトも効かない子トグル"),
    _f("engine.batch_llm.enabled", "journal", False, "none",
       "LLM 一括発行(未命中のみ並行発行→id 順 apply)。LLM 発行経路そのものを差し替える"),
    _f("cognition.policy_cache.enabled", "journal", True, "none",
       "方針キャッシュ。過去の LLM 出力を物理量キーで再利用して呼をスキップする"),
    _f("cognition.channels.enabled", "strict", False, "none",
       "観測チャンネル o_c(t)(第80バッチ。驚き駆動発火の入力側の定義)。外界(混雑/遭遇/受信発話/"
       "掲示/遅延/天候)・身体(各ゲージ)・予測不成立(第81 まで枠のみ)の 14 本を step ごとに"
       "決定論計算し、観測サイドカー channels.parquet へ書くだけ。**読み取り専用・乱数ゼロ・"
       "LLM ゼロ・プロンプト不変**なので ON でも L1/L2/L3 はバイト一致(サイドカーが 1 本増える)。"
       "σ_c は scripts/measure_sigma.py がこの parquet から実測し data/calib/sigma_c.json へ凍結する"),
    _f("cognition.fire.enabled", "journal", True, "none",
       "閾値発火+認知イベントキュー+同期バリア(第81バッチ)。『1 step=10分』が兼ねていた"
       "世界の進行と思考の頻度を分離し、(発火時刻, agent_id) の全順序キューが『いつ考えるか』を"
       "決める。発火源4種=periodic(文脈別の基本周期)/salience(S=Σg·|o−ô|/σ が θ 超え)/"
       "internal(ニーズ閾値の上向き横断)/social(会話返答・夜内省・朝計画の第一級化)。"
       "★affects_k=true を正直に宣言する: LLM 呼の**発生点そのもの**(_phase_drive の発火権配布)を"
       "差し替えるため。旧 k 非依存則は OFF のときに従来どおり成立し、ON では新不変量"
       "『認知イベントのスケジュール列が同一なら世界状態が同一』(P0-4)に置き換わる。"
       "★journal 等級: 発火数が変わる=LLM の自由文出力の量そのものが変わるので、"
       "事後再生には llm_cache/llm_journal が要る。"
       "★fingerprint_risk=none: プロンプト文面・欄構成を 1 バイトも変えない(発火理由は L1 の"
       "cog_fire にしか出ず、LLM へ渡す『きっかけ』は従来どおり drive.top_reason の既存語彙)。"
       "既定 OFF では本機能のモジュールが 1 度も呼ばれない=L1/L2/L3・乱数・呼数バイト一致"),
    _f("cognition.watch.enabled", "journal", True, "possible",
       "監視仕様 watch spec(第82バッチ。設計 §2.2)。発火のたびに LLM が『次にどうなると"
       "思うか』(期待値 ô)と『こうなったら気づきたい』(名前付きトリガ)を出力し、"
       "次の発火まで有効になる(**有効期限は持たせない**=時間軸が裏口から入るため)。"
       "受け側はホワイトリスト(記号・演算子)+ 数値クランプで検証し、**不正出力なら"
       "前回仕様を維持**する(捏造の既定値を作らない)。感度 g は含めない(§2.4)。"
       "★affects_k=true: ô が変われば S が変わり驚き発火の本数が変わる。"
       "★journal 等級: LLM の自由文出力(watch 節)を世界状態として消費する。"
       "★fingerprint_risk=possible を正直に宣言する: (1) watch 節そのものが全発火"
       "プロンプトに出る、(2) model-revision の中立 1 行は**驚き発火のときだけ**出る"
       "(= その思考が予測外れに起因することを本人が知りうる)。(2) は計画書 §6-3 が"
       "要求した設計そのもの(別種の認知モード)なので隠さず宣言する。なお発火源の語彙"
       "(periodic/salience/…)と実験条件語はプロンプトに 1 文字も出ない。"
       "★チャンネル id はプロンプトに出さない(不透明記号 c01… + 中立ラベルのみ)= "
       "factors 層のキーが LLM へ漏れない(no-fingerprint 契約の維持)"),
    _f("cognition.g_update.enabled", "strict", False, "none",
       "感度 g と閾値 θ の更新則(第82バッチ。設計 §2.5/§2.6)。"
       "g ← g + η(r − ρ·ē) − λ(g − g⁰): ē=最近の平均誤差(**慣れ**)・r=結果に効いた"
       "度合い(**感作**)・λ=ペルソナ基準への引き戻し(これが無いと履歴が性格を上書きして"
       "全員収束する)。θ ← θ + μ(f̄ − f*) は**日境界 1 回**(§2.6「μ の時定数は日オーダー」)。"
       "**LLM は一切関与しない**(§2.8「自分の過敏さは自分では設定できない」)ので strict。"
       "★affects_k=False: 呼び出し点を足しも減らしもしない(g が S を通じて発火判定を"
       "動かすのは fire.enabled の affects_k=true が既に宣言している間接経路)。"
       "★プロンプトを 1 バイトも変えない(g/θ は L1 と g/θ 軌跡サイドカーにしか出ない)。"
       "初期値条件は experiment.g_init(F/N/P)で切り替える"),
    _f("cognition.contract.enabled", "strict", False, "none",
       "Perception / Intent 契約(第85バッチ。physics-instructions.md Part P1)。"
       "エージェントコアと世界の結合面を 2 つのデータ型に限定する層で、物理エンジン(P3)は"
       "この契約の**裏側**に入る。Perception=1 回の思考が受け取る世界情報の唯一の型"
       "(視覚/聴覚/身体/内部 + 第80 channels 由来の salience。**生成は世界側の責務**="
       "engine/scheduler.py::build_perception 1 本)。Intent=思考の出力の唯一の型"
       "(移動目標・急ぎ度・回避傾向・対話意図・滞在意図。**経路の各点・速度の時系列は"
       "含めない**= 実行は世界側)。"
       "★repro_tier=strict: 通り道を変えるだけで乱数も LLM 呼も 1 本も足さない。"
       "★affects_k=False: 呼び出し点を 1 つも増減しない。"
       "★fingerprint_risk=none: **プロンプト文字列が 1 バイトも変わらない**ことが契約の"
       "定義そのもの(prompt_kwargs() が従来の材料 dict と完全に等しい dict を返す)。"
       "したがって ON/OFF で L1/L2/L3・draw・呼数が一致する(P1(3)『この時点では挙動は"
       "変わらない』)。既定 OFF は契約 module の関数が 1 度も呼ばれない従来経路"),

    # ---- freedom(LLM の自由記述をそのまま行動にする層=全部 journal)----
    _f("freedom.open_actions", "journal", False, "possible",
       "開放行動 'do' をメニューに追加し LLM の自由記述をそのまま行動として実行する"),
    _f("freedom.undefined_register", "journal", False, "none",
       "enum 外の行動主張を undefined_action として記録(パース後の振り分けのみ=プロンプト不変)"),
    _f("freedom.explicit_nothing", "journal", False, "possible",
       "『何もしない』を選択肢として明示し chosen_nothing を記録する"),
    _f("freedom.p2.move_home", "journal", False, "possible",
       "生活の自己決定: 空き住戸への転居(中立メニュー提示+決定論裁定)"),
    _f("freedom.p2.buy", "journal", False, "possible",
       "生活の自己決定: 発火時の消費意思"),
    _f("freedom.p2.study", "journal", False, "possible",
       "生活の自己決定: 学校/図書館での聴講"),
    _f("freedom.p2.partnership", "journal", False, "possible",
       "生活の自己決定: 交際の申込/別れ"),
    _f("freedom.p2.deviance", "journal", False, "possible",
       "生活の自己決定: 軽微な逸脱(無許可出店)"),

    # ---- beliefs(真偽台帳 第73バッチ Part B)----
    # journal 等級の根拠: 伝聞の発火判定が **LLM の自由文(発話テキスト)を消費する**
    # (話題の一致判定の入力が speak/dm の本文)。判定規則そのものは決定論で、入力は
    # L1 と llm_journal に残るので事後に再生できる = journal(none ではない)。
    _f("beliefs.enabled", "journal", False, "none",
       "真偽台帳(世界の事実を決定論で fact 化)+信念状態+伝播木。台帳はエージェント不可視"
       "(プロンプトに 1 バイトも足さない=fingerprint_risk は none)"),
    _f("beliefs.verify_actions", "journal", False, "possible",
       "『確かめる』(現場go/人にask/ネットnet)を行動空間へ追加する。全発火プロンプトへ"
       "中立な 1 行(fact に依存しない固定文字列)を足すので possible"),

    # ---- worldview / ontology ----
    _f("worldview.enabled", "strict", False, "possible",
       "主観的世界モデル(期待・可制御性・記述規範を状態から決定論導出)"),
    _f("worldview.expect_line", "strict", False, "possible",
       "『いつもと違う』1行をプロンプトへ"),
    _f("worldview.ctrl_line", "strict", False, "possible",
       "手応え/無力の自然文1行をプロンプトへ"),
    _f("worldview.norm_line", "strict", False, "possible",
       "記述規範1行(全員共通)をプロンプトへ"),
    _f("worldview.snapshot", "strict", False, "none",
       "日次 worldview イベント(観測専用)"),
    _f("ontology.enabled", "strict", False, "possible",
       "群のオントロジー(文化圏×経験の世界観共有群を決定論割当)"),
    _f("ontology.drill.enabled", "strict", False, "possible",
       "文化圏群の訓練経験1行をプロンプトへ注入"),
    _f("ontology.axes.companions.day_varying", "strict", False, "none",
       "同行者構成の割当に day を混ぜる(日単位で変わりうる)"),

    # ---- ads / crowd / pov / sns_geo ----
    _f("ads.enabled", "strict", False, "possible",
       "街路の環境情報(大型ビジョン・広告面)の知覚"),
    _f("crowd_visual.enabled", "strict", False, "possible",
       "群衆の視覚情報1行(同席者の決定論要約=記述的規範)"),
    _f("pov.enabled", "strict", False, "none",
       "顕著性駆動の POV 画像(CPU 決定論レンダ。text LLM 呼数に非依存)"),
    _f("pov.salience.first_visit", "strict", False, "none",
       "初訪問ノードを POV 発火候補にする"),
    _f("pov.salience.world_event", "strict", False, "none",
       "世界イベントの現場を POV 発火候補にする"),
    _f("pov.store.enabled", "strict", False, "none",
       "POV 画像を runs/<name>/pov/ に保存する(記録専用)"),
    _f("sns_geo.enabled", "strict", False, "none",
       "SNS/DM 伝播に両者の物理距離 dist_m を記録する(payload にキーを足すだけ)"),

    # ---- labeling / rewards / memory ----
    _f("labeling.mode", "journal", False, "possible",
       "造語の様式。open は LLM が自由に語を作れる(constrained=語彙制約)",
       off_value="constrained"),
    _f("labeling.place_binding.enabled", "journal", False, "none",
       "熟慮の coin_label(LLM が作った語の文字列)を発生ノードへ束縛する"),
    _f("labeling.place_binding.prompt_line", "journal", False, "possible",
       "束縛された造語を『この場所の呼ばれ方』として熟慮プロンプトへ1行注入"),
    # 第74バッチ IDEA③: 観測専用(L1 と発話テキストを読むだけ・世界へ戻さない)。
    # LLM の自由文(発話本文)を**入力に使う**が、出力は L2 の 2 列と解析だけで
    # 世界の因果に一切入らない。入力は L1 に残っており事後に同じ判定を再現できる=strict。
    _f("labeling.norm_stage.enabled", "strict", False, "none",
       "規範化ステージ検出器の L2 2列(観測専用。プロンプト・イベント・乱数は 1 バイトも動かない)"),
    _f("rewards.enabled", "strict", False, "possible",
       "造語が採用されるたび創作者に金銭報酬(D9。既定 off=過正当化の交絡排除)"),
    _f("memory.agentic_pull", "journal", True, "possible",
       "発火・内省で能動的に記憶検索する。**LLM 呼を1本足す**(recall ラウンド)"),

    # ---- relations / friend_graph / hierarchy / gossip / joint / party ----
    _f("relations.enabled", "strict", False, "possible",
       "社会関係の質(closeness/tier/評判)。増減は発話の感情価(決定論の語彙判定)から"),
    _f("relations.faction", "strict", False, "possible",
       "同じグループ所属を『同じ仲間』1行に反映する"),
    _f("relations.endogenous_accept.enabled", "journal", False, "possible",
       "共同行動の承諾/拒否を前日の発話・計画の**自由文**から構造化抽出して決める"),
    _f("relations.endogenous_accept.conflict_veto", "strict", False, "none",
       "当日予定と時間帯が衝突するときは確率でなく確定拒否(予定帳簿=決定論)"),
    _f("relations.endogenous_invite.enabled", "journal", False, "possible",
       "誘う相手の選択を前日の計画・発話の自由文から内生化する"),
    _f("relations.endogenous_quality.enabled", "journal", False, "possible",
       "交流の質(closeness 増分)を発話内容から内生化する"),
    # 第75バッチ IDEA⑤(ダンバー認知枠)。strict の根拠: 判定材料は closeness 台帳(数値)と
    # 同居人名簿と config だけで、**LLM の自由文出力を一切読まない**(第65 quality が journal
    # なのは発話本文を読むから=対照的)。乱数も 1 本も引かない(順序は closeness/id の全順序)。
    # fingerprint_risk=possible: 休眠で tier が 0 に落ちるとプロンプトの間柄行が
    # 「○○とは友人」→「○○とはN回話した仲」へ変わる(差分に気づく余地が原理的にある)。
    _f("relations.dunbar.enabled", "strict", False, "possible",
       "ダンバー認知枠(関係の維持コスト)。活性関係の総数が上限を超えたら最も弱い紐帯を"
       "休眠(退避=削除でない)にし、再接触で割引つきに再活性化する"),
    _f("relations.dunbar.protect_housemates", "strict", False, "none",
       "同居人との関係を休眠の対象から外す(日々の共在で維持される紐帯を認知枠の淘汰から保護。"
       "枠は消費するが落ちない)"),
    _f("friend_graph.enabled", "strict", False, "none",
       "現実的な友人ネットワークの生成(決定論+専用 stream)"),
    _f("hierarchy.enabled", "strict", False, "possible",
       "社会的ヒエラルキー(地位・信用・名声)の集計と反映"),
    _f("gossip.enabled", "strict", False, "possible",
       "負の評判の内生伝播。悪評タグは**内容を持たない匿名タグ**で LLM を使わない"),
    _f("joint.enabled", "strict", False, "possible",
       "共同行動エンジン(友人・同居人の同席編成。generate を1本も足さない)"),
    _f("party.enabled", "strict", False, "possible",
       "来街者 party の実体化(連れとして同席させる)"),
    _f("spark.enabled", "strict", False, "known",
       "火種介入の実験条件化。t=0 に資本・関係・集会アンカーを注入する=**当人から観測可能**"),
    _f("spark.menus.relations.enabled", "strict", False, "known",
       "火種メニュー(a) 初期関係の束を注入する"),
    _f("spark.menus.capital.enabled", "strict", False, "known",
       "火種メニュー(b) 初期資本・在庫を注入する"),
    _f("spark.menus.anchor.enabled", "strict", False, "known",
       "火種メニュー(c) 集会アンカーへ自由時間を寄せる"),

    # ---- lens(観測専用。世界状態を変えないので strict)----
    _f("lens.enabled", "strict", False, "none",
       "観測レンズのマスター(価値4軸・3M欲望・信用)"),
    _f("lens.value4.enabled", "strict", False, "none",
       "価値4軸の列・タブ(観測専用。LLM の自己申告値を読むが世界へは戻さない)"),
    _f("lens.motives.enabled", "strict", False, "none",
       "3M 欲望の列・ビュー(観測専用)"),
    _f("lens.trust.enabled", "strict", False, "none",
       "信用内訳の可視化(観測専用。要 hierarchy)"),
    _f("lens.deviation.enabled", "strict", False, "none",
       "ペルソナ逸脱率レンズ(観測専用)"),
    _f("lens.structure.enabled", "strict", False, "none",
       "社会構造の内生変動レンズ(観測専用)"),
    _f("lens.assets.enabled", "strict", False, "none",
       "資産分布レンズ(観測専用)"),

    # ---- psych / opinion / observer ----
    _f("psych.sdt.enabled", "strict", False, "none",
       "自己決定理論プラグイン(内発/外発で欲求ゲージ入力を個人別に倍率化)"),
    _f("psych.collective.enabled", "strict", False, "none",
       "集団効力感・社会的アイデンティティのプラグイン"),
    _f("psych.lynch.enabled", "strict", False, "none",
       "都市イメージ(landmark 重み付き行き先+観測レンズ)"),
    _f("psych.searle.enabled", "strict", False, "possible",
       "制度化(成立提案を制度としてプロンプト注入)"),
    _f("opinion.enabled", "strict", False, "none",
       "意見力学(Friedkin-Johnsen)。入力は発話から決定論で採った感情価スカラーのみ"),
    _f("observer.run_manifest", "strict", False, "none",
       "run_manifest.json を書く(記録専用・シムは読まない)"),
    # 第93バッチ IF-1(行為 → 思考の来歴リンク)。設計 src/society/provlink.py /
    # docs/research/if-lane-research.md §4(W3C PROV-DM の wasInformedBy + prov:role)。
    # strict の根拠: L1 の**既存固定列 llm_call_id と payload の 1 キー**を埋めるだけの
    #   記録専用層。LLM の自由文を新たに読む経路を 1 つも足さず、乱数も 1 本も引かない。
    # affects_k=False: generate() の呼び出し点を足しも減らしもしない(既に撃った呼の id を
    #   行為イベントへ写すだけ)。
    # fingerprint_risk=none: プロンプトを 1 バイトも変えない(id は L1 にしか出ない)。
    _f("observer.llm_link", "strict", False, "none",
       "行為イベントへ (llm_call_id, role) の対を刻む(PROV の wasInformedBy 辺 1 本)。"
       "role の語彙は l1b_llm の purpose と同一で新語彙を作らない。刻むのは**行為者自身が"
       "出したイベントだけ**(聞き手側の hear/opinion_shift は別主体の activity)。"
       "ON のときだけ、熟慮経路に迷い込んだ plan/recall/reflect を "
       "fallback{reason:misrouted_action} として計上する(従来は無音の no-op=観測の穴)。"
       "DAG はシム側に持たせない(捕集と組み立ての分離)= 観測がシムを変えない"),
    # 第100バッチ IF-F(因果台帳)。分類表と実装 src/society/observer/causality.py。
    # strict の根拠: L1 に**記録専用の 2 列**を足すだけの層。分類は kind 単位の静的表で、
    #   LLM の自由文を新たに読む経路を 1 つも足さず、乱数も 1 本も引かない。
    # affects_k=False: generate() の呼び出し点を足しも減らしもしない。
    # fingerprint_risk=none: プロンプトを 1 バイトも変えない(2 列は L1 にしか出ない)。
    _f("observer.causality.enabled", "strict", False, "none",
       "L1 の各イベントへ cause_type(agent/device/schedule/physics/natural/boundary)と "
       "actor_id を刻む。agent_id は『この行が誰の視点で記録されたか』であって行為者では"
       "ないため(hear=聞いた側 / wage=受け取った側 / weather=-1)、そのままでは"
       "『エージェントが起こした変化』と『装置が起こした変化』を事後に分離できない。"
       "agent_id が患者である種は payload から行為者を戻す(hear→speaker / "
       "opinion_shift→source / transmission→from)。復元できないものは null で正直に"
       "開示する。分類は静的表なので事後解析は**この列が無い既存ランでも**同じ分類ができ、"
       "列があるランでは表とエンジンの刻印を突き合わせて食い違いを報告する。"
       "W2 で 3 列目 device_id(装置の同一性。faregate:<駅> / signal:<交差点> / "
       "logistics:goods / commerce:hours / gov:main / operator:transit)を足した: "
       "actor_id は int(個体)なので装置は名乗れず、装置起因の行が『行為者不明』に"
       "落ちていた。刻んでよい cause_type は device/schedule/boundary に閉じてある"
       "(装置の窓の中で出ても agent/physics/natural はその装置の作った出来事ではない)"),
    _f("observer.llm_health.enabled", "strict", False, "none",
       "L2 に LLM 健全性 KPI 3列を足す(観測専用・累積カウンタ)"),
    # 第98バッチ W2-6: finalize のメモリ有界化(設計 src/society/observer/logger.py)。
    # ★repro_tier=strict の根拠(**バイト列が変わる機能なので特に正直に書く**):
    #   本トグルが変えるのは「既に確定した記録を **どの物理レイアウトで書き出すか**」だけで、
    #   世界の因果には 1 バイトも触れない。乱数 stream を 1 本も引かず、LLM を 1 本も呼ばず、
    #   プロンプトを 1 バイトも変えず、**シムは書いた parquet を二度と読まない**。
    #   ON/OFF で L1/L2/L3 の **行の内容・行順・スキーマは完全同値**であり、変わるのは
    #   parquet の row-group の切れ目に由来するファイルのバイト列だけである。
    #   registry の 3 等級は「世界の因果を事後に再構成できるか」の分類(journal=LLM の自由文を
    #   消費する / none=実時刻・外部プロセス・非決定並列)であって、**記録ファイルの
    #   バイト同一性の分類ではない**。したがって journal / none のどちらにも該当せず strict。
    #   ただし「同一 seed の 2 ランで parquet を sha256 比較する」型の検査は ON/OFF を
    #   跨いで成立しない(行比較なら成立する)。この 1 点だけは正直に申告しておく。
    # affects_k=False: generate() の呼び出し点を 1 つも足さない/減らさない(finalize だけの層)。
    # fingerprint_risk=none: プロンプトを 1 バイトも変えない。エージェントから観測する経路は無い。
    _f("observer.finalize.streaming", "strict", False, "none",
       "finalize(part → canonical の結合)を **全 part 同時載せ**から**逐次書き**へ替える。"
       "既定 OFF は全 part を read_table して concat_tables するため、part 化で走行中の RAM を"
       "解放しても**ランのピークメモリが L1 の総量で決まる**(在場 25万 × 10 日 = 42.7 GB・"
       "40.6 億イベントの外挿 = 解析以前に『ランの書き終わり』で落ちる。W2-2 の発見)。"
       "ON は part を 1 つずつ row-group 単位で読み ParquetWriter へ流すので、ピークは"
       "row-group 1 個ぶん(observer.finalize.row_group_rows)だけで part 数にも L1 総量にも"
       "依存しない。★観察ランでは ON を推奨する。"
       "出力は OFF と行の内容・行順・スキーマが完全同値(**バイト列は row-group の切れ目が"
       "変わるので不一致**)。part が 1 つも無いランは ON でも従来経路に落ちる=バイト一致。"
       "副産物として一時ファイル + os.replace により merge 中のクラッシュに強い。"
       "★W4-E(第99バッチ)で対象は L1/L1b/L2/L3 に加え **サイドカー全部**"
       "(indoor_tracks_samples / indoor_tracks_contacts / org_ledger / finance / channels /"
       " cognition_g)になった。実装は observer/finalize.py の FinalizeStreamMixin 1 本で、"
       "**この 1 キーの判断で全ファイルが同じモードになる**(サイドカー用の別キーは作らない)"),
    _f("observer.echo.enabled", "strict", False, "none",
       "L2 にエコー/自己反復 5列を足す(観測専用・常設)"),
    # 第91バッチ: 退行シグナル監視(設計 §3)。L1 を読むだけで世界も乱数も 1 バイト触らない=strict。
    _f("observer.regression.enabled", "strict", False, "none",
       "L2 に退行シグナル列(行動分散・訪問エントロピー・語彙エントロピー/n-gram 重複率・"
       "発火率の張り付き)を足す(観測専用・既定 OFF=列なし)"),
    # 第78バッチ: 状態ハッシュチェーン(verify 用)。記録専用でシムは読まない=strict。
    _f("observer.state_hash.enabled", "strict", False, "none",
       "各 step の world state を正準シリアライズ→sha256→前 step と連鎖させて "
       "state_hash.jsonl に書く(記録専用・シムは読まない・T1/T6 の判定装置)"),
    # ---- A13 日次入場者名簿サイドカー(レーン丙 2026-08-13)----
    # strict の根拠: 観測側だけで閉じる追記(sim.agents の既存属性を読むだけ)。
    # affects_k=False: generate() の呼び出し点を 1 つも足さない。
    # fingerprint_risk=none: プロンプトを 1 バイトも変えない(当人から観測できる差分ゼロ)。
    _f("observer.roster_daily.enabled", "strict", False, "none",
       "日境界に「まだ名簿に載せていない在場者」を roster.parquet へ 1 行ずつ追記する"
       "(A13)。traits.json は day0 の在場者・agents.json はその瞬間の在場者しか写さないため、"
       "プール回転で day1 以降に入場する個体(finals 構成で 20.9 万人)の素性がどこにも残らず、"
       "事後解析(measure の R² 回帰・deviation・復元実験)の分母から丸ごと落ちていた。"
       "1 個体は生涯 1 行(入場した日)= 再来街で 2 行目は作らない。trait 名はコードに書かず"
       "traits_json の 1 列に丸ごと入れる(no-fingerprint 契約)。既定 OFF ではオブジェクトを"
       "作らず 1 ファイルも書かない。★規模: 在場25万×10日で約46万行 = +100MB 前後"),

    # ---- ablate(第78バッチ: アブレーション 4 種。対照条件のスイッチだけ)----
    # 全て既定 OFF。repro_tier=strict の根拠は「機構を**外す**方向の決定論ゲート」で
    # あること(LLM の自由文を新たに読む経路を 1 つも足さない)。
    _f("ablate.llm_off", "strict", True, "none",
       "LLM を 1 本も呼ばず既存のルール層(routine.decide=ニーズ充足 + POI 選好)だけで回す。"
       "計画・内省も撃たない。**呼数 k は 0 になる**ので affects_k=True。"
       "プロンプトが 1 つも組まれない=当人から観測できる差分が存在しない=risk none"),
    _f("ablate.propagation_off", "strict", False, "known",
       "発話は通常どおり生成する(呼の発生箇所は不変)が、内容が他エージェントの文脈へ"
       "一切入らない。専門化スコアの帰無モデル。**合成文は注入しない**方針のため"
       "『周りは居るが誰も何も言わない街』として原理的に観測されうる=risk known"),
    _f("ablate.cognitive_tier", "strict", True, "possible",
       "llm.fleet の割当を強制下位へ(rule/small/mid/full)。rule は LLM を使わないルール層"
       "=呼数が変わるので affects_k=True。small/mid はモデルが変わり応答の質感が変わる"
       "=possible。fleet 非使用ランでは small/mid は縮退して full と同一(manifest に明示)",
       off_value="full"),
    _f("ablate.shuffle_partners", "strict", False, "possible",
       "対話の返答権/宛先を関係グラフでなく同席者からの一様乱択にする(専用 stream・"
       "always-draw)。聞き手集合は不変=会話量・呼数は不変。関係の近い相手が隣に居ても"
       "返答が来ない=間柄行との不整合に気づく余地が原理的にある=possible"),

    # ---- ablate プラセボ L1(第89バッチ: 呼数・書式・乱数消費は同一のまま中身だけ壊す)----
    # 正典 docs/plans/source/design-discussion-20260802.md §1。3 種とも
    #   affects_k=False … プロンプト文字列を書き換えるだけで generate() の呼び出し点は不変
    #   repro_tier=strict … LLM の自由文を新たに読む経路をひとつも足さない(決定論の書換)
    #   fingerprint_risk=known … **中身を壊すのが定義**なので当人から観測できる差分が必ず出る。
    #                            隠す方法は存在しない(隠せたらプラセボとして無効)。
    _f("ablate.context_shuffle", "strict", False, "known",
       "他者由来の文脈節(語彙・記憶・間柄・同席者・TL・直前のやりとり・返答の相手発話)を、"
       "同一ラン内の別エージェントの**同種**の節と決定論的に入れ替える(専用 stream)。"
       "ペルソナと自分の状態は保持=「文脈が正しいこと」だけを壊す。節の位置・区切り・"
       "項目数は保つので書式は不変。見覚えのない出来事が自分の文脈として出る=risk known"),
    _f("ablate.persona_swap", "strict", False, "known",
       "プロンプトのペルソナ節を別エージェントのペルソナと**対合**(A↔B の相互交換)で"
       "入れ替える。全単射なので人口のペルソナ分布は保存される。世界状態・文脈は正しいまま"
       "=「人格と行動の結びつき」だけを壊す。自己紹介と自分の記憶・関係が食い違う=risk known"),
    _f("ablate.context_sever", "strict", False, "known",
       "文脈節を中立プレースホルダ('…')へ置換する(項目数・区切りは保つ)。第78 "
       "propagation_off が**送り手側**で節ごと消すのに対し、こちらは世界状態を一切変えずに"
       "**受け手側**で中身だけ潰す『文脈遮断の完全版』。節はあるが中身が無い=risk known"),

    # ---- ablate プロンプト言い換え(第92バッチ SV-U1 B4 = サーベイ S-16)----
    #   affects_k=False … 凍結表の文字列置換だけで generate() の呼び出し点は不変
    #                     (プラセボ 3 種と同じ構造。実測 k は応答経由で数 % 動く=間接効果)
    #   repro_tier=strict … LLM の自由文を新たに読む経路をひとつも足さない(決定論の表引き・
    #                       乱数ゼロ)。言い換え表は事前にオフラインで作った凍結資産。
    #   fingerprint_risk=known … プロンプト文字列が変わる以上「観測が世界を変えない」は
    #                            成立しない。隠す方法は存在しない(隠せたら検査として無効)。
    _f("ablate.prompt_paraphrase", "strict", False, "known",
       "プロンプトの言い回しを意味等価な凍結ルックアップ表で差し替える"
       "(''=OFF / v1=語彙 / v2=文体 / v3=語彙+文体 / v4=語彙+文体+統語限定)。"
       "表は data/prompt_paraphrase_sets.json のみが源で LLM 生成は禁止、sha256 を manifest へ。"
       "JSON キーと行動語彙は変えない=パース互換。作用点は build_prompt 末尾 1 箇所で"
       "呼び出しサイトは増減ゼロ。**主要結論の符号が言い回しに依存しないか**を測る"
       "(Sclar et al. 2024 / Ye et al. 2026 TRAILS)。プラセボ 3 種・propagation_off とは相互排他",
       off_value=""),

    # ---- experiment(第74バッチ IDEA④: 対照セルの宣言。決定論=strict)----
    _f("experiment.flat_traits.enabled", "strict", False, "none",
       "初期個体差ゼロ対照。全個体の traits を定数化して R²(k) の分母を実験的に消す"
       "(ペルソナ文は不変・乱数消費本数も不変。プロンプトに条件を示す語は 1 つも出ない)"),
    _f("experiment.flat_traits.include_derived", "strict", False, "none",
       "flat_traits ON のとき traits 由来の drive_threshold / fire_weight も定数写像へ潰す"
       "(false は名簿の個体差が発火閾値経由で残る=不完全な対照)"),

    # ---- government / media / organizations / work ----
    _f("government.enabled", "strict", False, "none",
       "行政(区/都/国の税・給付・公務員給与)"),
    _f("media.enabled", "strict", False, "none",
       "娯楽メディア(TV/動画/ゲームの在宅娯楽と気分修復)"),
    _f("media.prompt_context", "strict", False, "possible",
       "直近視聴タイトル(架空)1行を発火文脈へ入れる"),
    _f("organizations.enabled", "strict", False, "none",
       "組織(職場・学校)台帳の配属"),
    _f("organizations.commute_to_poi", "strict", False, "none",
       "配属者の職場/学校を台帳の実 POI へ束ねる"),
    _f("work.service.enabled", "strict", False, "none",
       "L2 域内従業者の業務の実体(接客・供給)"),
    _f("work.service.record_unstaffed", "strict", False, "none",
       "スタッフ不在の消費を agent_id=-1 の記録として残す(観測のみ)"),
    _f("work.service.digest", "strict", False, "possible",
       "スタッフの S2 ダイジェストに業務要約を1行供給する"),
    _f("work.service.indoor_fields", "strict", False, "none",
       "serve イベントに org_id/floor を付ける"),
    _f("work.service.office.enabled", "strict", False, "none",
       "オフィス系の日次 org_output を出す"),
    _f("work.service.office.by_org", "strict", False, "none",
       "org_output を org_id 単位に分解する(同居複数社)"),
    _f("work.service.ledger.enabled", "strict", False, "none",
       "会社台帳(1日1行/社)を書く(記録専用)"),
    _f("work.bind_workplace.enabled", "strict", False, "none",
       "未束の個体の work_node を実 POI へ束ねる(coverage 拡大)"),
    _f("work.bind_workplace.rebind_bound", "strict", False, "none",
       "既に work_node を持つ個体も束ね直す"),
    _f("work.bind_workplace.poi_match_fallback", "strict", False, "none",
       "台帳ノードが地図に無い時に POI カテゴリ+安定ハッシュで決定論マッチ"),

    # ---- needs / affect / inner_life / weather / schedule ----
    _f("needs.enabled", "strict", False, "none",
       "欲求プロファイルの個人差(5次元潜在プロファイル)"),
    _f("affect.enabled", "strict", False, "possible",
       "感情・興味・注意ハブ(感情価/覚醒度と注意配分)"),
    _f("inner_life.enabled", "strict", False, "possible",
       "内面本格版のマスター(感情・目標・趣味)"),
    _f("inner_life.emotion.enabled", "strict", False, "possible",
       "内面: 感情状態の保持と更新"),
    _f("inner_life.goals.enabled", "strict", False, "possible",
       "内面: 長期目標の保持"),
    _f("inner_life.goals.inject_prompt", "strict", False, "possible",
       "『長期的な目標: ◯◯』を発火プロンプトへ注入"),
    _f("inner_life.hobbies.enabled", "strict", False, "possible",
       "内面: 趣味・関心の保持"),
    _f("inner_life.hobbies.inject_prompt", "strict", False, "possible",
       "『趣味・関心: ◯◯』を発火プロンプトへ注入"),
    _f("weather.enabled", "strict", False, "possible",
       "天気(気温・降水・降雪)。暦からの決定論+専用 stream"),
    # 第80バッチ W2(天候の生成型実データ化)。strict の根拠: generated は
    # 「静的ファイル(較正済みパラメータ)+ マスターシード」の純関数で、ラン中の
    # ネットワーク呼び出しはゼロ・LLM の自由文を 1 バイトも読まない。table は乱数すら
    # 引かない完全決定論。fingerprint_risk=possible: 気温の分布が変わり、猛暑日が
    # 連続しうる(既定 synthetic では 36℃以上が構造的に出ない)=プロンプトの天気1行が
    # 変わる。off_value="synthetic" = 現行動作。
    _f("weather.mode", "strict", False, "possible",
       "天候の決め方(synthetic=現行合成 / generated=較正済み確率生成器 / table=凍結実測の"
       "実日付引き)。較正済みでない月・凍結範囲外の日付は synthetic へ自動フォールバックし、"
       "件数と理由を summary.json と run_manifest.json の weather ブロックに残す",
       off_value="synthetic"),
    _f("weather.extra_prompt_fields", "strict", False, "possible",
       "天気1行に最低気温・湿度・降水・猛暑日・暑さ指数(推定)を追記する。mode=synthetic "
       "では無視する(既定モードのプロンプト文言は 1 バイトも変えない)"),
    _f("schedule.enabled", "journal", False, "possible",
       "長期予定・スケジュール帳。**発話テキスト(LLM 出力)から予定を抽出**して帳簿に積む"),
    _f("schedule.inject_prompt", "journal", False, "possible",
       "近い予定をプロンプトへ1行注入する(抽出元が LLM 発話なので journal)"),

    # ---- institution_routes / career / annual_events ----
    _f("institution_routes.assembly.enabled", "strict", False, "possible",
       "制度改変ルート: 議会(名簿制の議員)"),
    _f("institution_routes.vote.enabled", "strict", False, "possible",
       "制度改変ルート: 署名モードを投票モードに置換"),
    _f("institution_routes.deliberation.enabled", "strict", False, "possible",
       "制度改変ルート: 否決可能な熟議の摩擦"),
    _f("institution_routes.labor.enabled", "strict", False, "possible",
       "制度改変ルート: 労働(団体交渉等)"),
    _f("institution_routes.enforcement.enabled", "strict", False, "possible",
       "制度改変ルート: 取り締まり(警察官による執行)"),
    _f("career.enabled", "strict", False, "none",
       "失業・転職・再就職・起業転換・解雇規制(日次確率)"),
    _f("career.by_choice.enabled", "journal", False, "possible",
       "求職 tool を行動空間に足し、**LLM の選択**を転職の起点にする(呼数は不変)"),
    _f("annual_events.enabled", "strict", False, "possible",
       "年中行事(暦の行事名と群集フラグ)"),
    _f("annual_events.ambient", "strict", False, "none",
       "群集密度の雰囲気推定を crowd_surge payload に併記する"),

    # ---- info_env / health / household / housing ----
    _f("info_env.enabled", "strict", False, "none",
       "情報環境の非対称のマスター"),
    _f("info_env.recommendation.enabled", "strict", False, "none",
       "意見整合バイアスでタイムラインを選別(エコーチェンバー)"),
    _f("info_env.influence.enabled", "strict", False, "none",
       "高フォロワー投稿の reach を加重(バイラル)"),
    _f("info_env.misinfo.enabled", "strict", False, "possible",
       "誤情報・訂正・炎上のダイナミクス(誤情報タグは専用 stream の抽選=内容非依存)"),
    _f("health.enabled", "strict", False, "possible",
       "健康(疲労・病気・欠勤・受診・引きこもり)"),
    # 身体と事件のレイヤー H1(2026-08-10。正典 docs/plans/body-incident-layer-plan.md §1)。
    # strict の根拠: 発症は 5 つの独立ハザード × frailty を **新 named stream "health_onset"**
    #   から引くだけ(1 個体 1 日あたり固定長 12 本 = 分岐で消費順が動かない)。発火・進行・
    #   回復・転帰・傍観者の通報者選定はすべて決定論の純関数で、LLM の自由文を 1 バイトも
    #   読まない。frailty も (seed, agent.id) の安定ハッシュ = seed から再構成できる。
    # affects_k=False: generate() の呼び出し点を 1 つも足さない。欠勤・在宅・倒れ・死で
    #   co-location が変わり発火数が間接的に動きうるのは親トグル health と同型で、
    #   レジストリ冒頭の方針どおり False 側(間接効果は compute_matched 対照で切り分ける)。
    # fingerprint_risk=possible: プロンプトの**欄**は 1 つも増えないが、定型 1 行の記憶
    #   (「暑さで気分が悪くなった」等)が ON のときだけ入り、当人にとって観測可能な世界の
    #   事実(倒れた人が居る・同僚が居なくなった)が起きる = 親 health と同じ等級。
    # ★死(mortality)はユーザー決定 §6-1 により**本トグル配下**に含める(別オプトインに
    #   しない)。量は較正で担保する: OHCA 生存退院 ≒10% / 搬送の死亡 ≒1.3%。
    _f("health.severity.enabled", "strict", False, "possible",
       "重症度の状態機械(S0 健康→S1 軽症→S2 中等症→S3 重症→S4 心停止→{回復, 死亡})。"
       "単一の真偽値 sick を置き換え、発症チャネル 5 種(急病/熱中症/急性アルコール/"
       "外傷/心停止)を独立ハザード × frailty(年齢帯の通院者率由来)で引く。"
       "sick-role(軽症の presenteeism・中等症の受診/停留)と、救急の引き金の世代交代"
       "(city_ops の暫定ハッシュゲート → S3/S4 への遷移そのもの)・傍観者モデルの較正・"
       "搬送先での重症度確定・死(境界 despawn と同型の永続退場)を含む。"
       "既定 false では単一真偽値の現行動作と完全同値"),
    # ---- 医療の受け皿 H2(設計 src/society/medical.py /
    # docs/plans/body-incident-layer-plan.md §2)。H1(重症度)と city_ops(救急連鎖)の続き。
    # strict の根拠: **乱数 stream を 1 本も引かない**(搬送先は地図の純関数・在院長は
    #   確定重症度の純関数・金額は config の純関数)。LLM の自由文を 1 バイトも読まない。
    # affects_k=False: generate() の呼び出しサイトを 1 つも足さず、プロンプトの**節**も
    #   増やさない(出来事は既存の記憶欄に定型 1 行が入るだけ)。在院で物理位置が変わり
    #   対面 co-location が動くのは本レジストリの規約どおり False(位置経由の間接効果)。
    # fingerprint_risk=possible: 出来事の定型 1 行が当事者の記憶に載る(H1 と同等級)。
    _f("medical.enabled", "strict", False, "possible",
       "医療の受け皿(搬送・入院・医療費)。(1) 搬送先 = 地図の subcat=hospital(v8 は 7 件。"
       "**subcat を持たない地図では 0 件 = 搬送が起きない**= 名前から病院を推測しない)"
       "(2) 入院 = 確定重症度で在院長が決まる(軽症=数時間 / 中等症=数日 / 重症=長期。"
       "物理は宿泊 lodging と同型の『非自宅建物で N 泊』)(3) 金の三本足 = 救急搬送の公費"
       "(区の歳出 → RoW ems_operation。運行主体の都は街の外)/ 患者 3 割(既存 spend "
       "cat=medical。**受け手を受診先ノードで解決する是正**を含む)/ 保険 7 割"
       "(RoW チャネル insurance_reimbursement)。★治療は転帰を変えない(死は H1 が発症時に"
       "決める設計判断を再開封しない)。既定 OFF は新 4 種の L1 0 件・state なし・"
       "agent に属性が生えず・プロンプト不変(L1 バイト一致)"),
    _f("medical.transport.enabled", "strict", False, "none",
       "救急搬送(現場 → 病院)だけを個別に切る子トグル(親 medical.enabled 配下)。"
       "false では ems_transport / hospital_admit が 0 件になる(倒れた個体はその場に留まる)"),
    _f("medical.transport.hold_crew", "strict", False, "none",
       "搬送のあいだ救急隊を現場復帰させないかの指定(親配下)。既定 true = 搬送中の隊は"
       "次の通報に応えられない(救急車ひっ迫が観測できる)。false では現着時間だけで戻る",
       off_value=True),
    _f("medical.clinic.enabled", "strict", False, "none",
       "自力受診(中等症)の受診先解決だけを個別に切る子トグル(親配下)。false では"
       "医療費の受け手が従来どおり支払者の居場所から解決される(= 多くが RoW へ漏れる)"),
    _f("medical.admit.enabled", "strict", False, "none",
       "入院だけを個別に切る子トグル(親配下)。false では搬送されても在院せずその場で目覚める"),
    _f("medical.money.enabled", "strict", False, "none",
       "金の三本足(公費・自己負担・保険給付)だけを個別に切る子トグル(親配下)。"
       "false では搬送・入院という**物理と状態**だけが起き、会計は 1 円も動かない"),
    _f("household.enabled", "strict", False, "possible",
       "世帯・家族・恋愛のマスター"),
    _f("household.realistic", "strict", False, "none",
       "地理×年齢の整合束ね+続柄+規模重みで現実的な世帯を作る"),
    _f("household.family_dinner.enabled", "strict", False, "none",
       "家族の夕食共食"),
    _f("household.cohabit.enabled", "strict", False, "none",
       "同棲=世帯の物理再編"),
    _f("household.pool_bind.enabled", "strict", False, "none",
       "世帯を **pool 名簿の決定論的な分割**として持ち、入場のたびに冪等に着席させる"
       "(レーン乙 ブロック3)。build_households は起動時 1 回しか走らないため、プール回転で"
       "day1 以降に入場する個体は世帯を永久に持てず、housemates/household_id が退避にも"
       "載っていなかったので『同居しているという事実』が回転のたびに壊れていた。"
       "ON では住居も世帯 id から決定論で決まる(メンバは在場日が違っても同じ家に住む)。"
       "乱数 stream は 1 本も引かない(全て安定ハッシュ・run.seed 非依存)。"
       "既定 OFF=1 行も通らない=home 割当も属性も完全に不変"),
    _f("housing.relocation.enabled", "strict", False, "none",
       "内生的な転居(構造固着を運営者介入なしに動かす)"),

    # ---- commerce / services / delivery / disaster / chance / lodging / diversity ----
    _f("commerce.enabled", "strict", False, "possible",
       "商業のダイナミクス(営業時間・動的価格・品切れ)"),
    _f("commerce.inventory.enabled", "strict", False, "none",
       "実在庫・日次補充トリップ・商品実体"),
    _f("commerce.inventory.b2b.enabled", "strict", False, "none",
       "供給網の内生化(組織間の B2B 取引)"),
    _f("services.enabled", "strict", False, "possible",
       "サービスの実体化(service/education POI を経済的に起こす)"),
    _f("services.self_dev.enabled", "strict", False, "none",
       "自助努力(塾=技能/ジム=体力の反復累積と skill→賃金経路)"),
    _f("delivery.enabled", "strict", False, "none",
       "宅配・フードデリバリー(注文→受取→配送→課金)"),
    _f("disaster.enabled", "strict", False, "possible",
       "都市・環境インフラのショック(外生災害)"),
    _f("disaster.suspend_transit", "strict", False, "none",
       "災害中に電車を運休にする"),
    _f("chance.enabled", "strict", False, "possible",
       "生活の偶発イベント層(偶然の出会い等。専用 stream)"),
    _f("lodging.enabled", "strict", False, "none",
       "宿泊・ホテル滞在(hotel POI があるときだけ実効)"),
    _f("society_diversity.enabled", "strict", False, "possible",
       "観光・多言語・犯罪・治安"),
    _f("society_diversity.avoid_danger", "strict", False, "none",
       "危険地帯を自由時間の行き先から避ける"),
    # ---- 環境フィードバック(エージェント → 環境。第84バッチ・設計 §4)----
    _f("env.feedback.enabled", "strict", False, "possible",
       "環境フィードバック最小 3 規則(第84バッチ。設計 §4)。これまで一方向だった"
       "『環境 → エージェント』に逆向きの矢印を 1 本通し、**行動 → 集約物理量 → 閾値超過 →"
       "環境イベント → 知覚 → 行動**の環を閉じる。規則は §4.2 の最小構成 3 本のみ"
       "(ホーム密度→停車時間延長→遅延伝播 / 改札飽和→入場規制 / POI 占有>容量→待ち行列→"
       "他 POI へ流出)。★エージェントは環境イベントを宣言できない(§4.1)= 発火判定は"
       "すべてコード側の集約量で、LLM 呼び出し 0 本・乱数 0 本の完全決定論=strict。"
       "★affects_k=false: generate() の呼び出し点を 1 つも足さない。遅延・規制で"
       "co-location が変わり発火数が間接的に動きうるのは commerce/disaster/health と同型で、"
       "ここを true にすると本リポジトリのほぼ全機能が true になる(レジストリ冒頭の方針)。"
       "★fingerprint_risk=possible: プロンプトの文面・欄は 1 バイトも変えないが、"
       "『駅から出られない』『行きたかった店が候補に無い』という**当人にとって観測可能な"
       "世界の事実**が ON のときだけ起きる(commerce の閉店除外と同じ等級)。"
       "★発散対策(§4.4)は 3 規則すべてに減衰項+上限を入れて T5 で実測固定してある。"
       "既定 OFF では envfeedback.py が 1 度も状態を作らない=L1/L2/L3・乱数・呼数バイト一致"),
    _f("env.feedback.transit.enabled", "strict", False, "possible",
       "規則1: ホーム負荷(駅ノードの同時滞在人数 + その step の乗降人数)が閾値を超えると"
       "停車時間が延び、遅延が回復運転項 γ<1 で次の発車へ持ち越される。遅延は駅からの"
       "退出/帰還を待たせ、第80 の観測チャンネル ext.transit_delay を初めて非ゼロ分散にする"),
    _f("env.feedback.gate.enabled", "strict", False, "possible",
       "規則2: 改札への流入レートが処理能力(world.mod.gate_capacity の係数を掛けた"
       "capacity_per_min×Δt)を超えると入場規制が発動する。**第67バッチの予約フィールド"
       "gate_capacity をここで初めて消費する**(summary の consumed が true になる)"),
    _f("env.feedback.poi.enabled", "strict", False, "possible",
       "規則3: POI ノードの占有が容量を超えると待ち行列イベントを出し、そのノードを"
       "一定 step のあいだ行き先候補から外す(既存の commerce.filter_open 経路を使う="
       "新しい選択ヒューリスティックを足さない)"),
)

BY_ID: dict[str, Feature] = {f.id: f for f in FEATURES}

# 「機能トグルではない bool 設定キー」= 未宣言検出の対象外。**理由を必ず書くこと**。
ALLOWLIST: dict[str, str] = {
    # conf には無く、実行時に scripts/run.py が dotlist で足す実行制御フラグ。
    # 「seed を OS エントロピーから採るよう要求したか」の記録であって世界の機能ではない。
    "run.seed_auto": "実行時の seed 採取要求フラグ(conf 非搭載・世界の挙動を変えない)",
    # 第82バッチ: watch 機能**内部**の形の指定であって独立した機能トグルではない
    # (cognition.watch.enabled が OFF なら 1 バイトも効かない)。ON のときこの行を
    # 落とすと「watch 節はあるが驚き発火の 1 行が無い」対照条件になる=解析用の内部レバー。
    "cognition.watch.model_revision":
        "watch 内部の提示形の指定(watch.enabled が親トグル。単独では何も起きない)",
    # 第87バッチ: engaged 機能**内部**の形の指定であって独立した機能トグルではない
    # (cognition.engaged.enabled が OFF なら 1 バイトも効かない)。どちらも
    # 「その規則を切ったらどうなるか」を見るための解析用の内部レバー。
    "cognition.engaged.wrapup":
        "ターン上限で切り上げターンを挟むかの指定(engaged.enabled が親トグル)",
    "cognition.engaged.sign_memory":
        "プリエンプト時に兆しメモリを書くかの指定(engaged.enabled が親トグル)",
    # 第88バッチ: 高解像度層**内部**の形の指定であって独立した機能トグルではない
    # (model.mind.enabled が OFF なら 1 バイトも効かない。frac=0 でも効かない)。
    # 「夜内省の選抜を第87 reflect_frac から高解像度層へ移すか」の内部レバー。
    "model.mind.tiers.high.reflect":
        "夜内省の対象を高解像度層にするかの指定(model.mind.enabled が親トグル)",
    # 第91バッチ: 退行シグナル監視**内部**の形の指定であって独立した機能トグルではない
    # (observer.regression.enabled が OFF なら列自体が生えない)。第87 の申し送り
    # (機構由来の定型応答を語彙指標から外す)を切って影響を見るための解析用レバー。
    "observer.regression.exclude_template":
        "定型応答を語彙エントロピー/n-gram から除外するかの指定"
        "(observer.regression.enabled が親トグル。観測列の定義であって世界の機能ではない)",
    # 第109バッチ PRES-A1: 習慣カレンダー**内部**の共変量の on/off であって独立した機能
    # トグルではない(pool.presence.habit.enabled が OFF なら 1 バイトも効かない)。
    # 「曜日/天候の共変量を切ったらどうなるか」を見るための解析用の内部レバー。
    "pool.presence.habit.weekday":
        "目的別の曜日プロファイルを掛けるかの指定(pool.presence.habit.enabled が親トグル。"
        "プロファイルは週平均 1.0 に正規化済み = 週次の総来街量は ON/OFF で不変)",
    "pool.presence.habit.weather":
        "雨/雪の目的別弾性を掛けるかの指定(pool.presence.habit.enabled が親トグル。"
        "weather.mode=generated 以外では ON でも不活性)",
}


# --------------------------------------------------------------------------- #
# 走査ヘルパ
# --------------------------------------------------------------------------- #
def flatten_bools(cfg) -> dict[str, bool]:
    """config(DictConfig / dict)の**真偽値リーフ全部**をドット記法で平坦化する。

    observer/manifest.py の collect_toggles(「全スイッチ状態」)と同一定義。
    定義を1箇所に保つため manifest 側はここへ委譲する。
    """
    out: dict[str, bool] = {}

    def walk(node, prefix: str) -> None:
        if isinstance(node, dict):
            for k in sorted(node.keys()):
                walk(node[k], f"{prefix}.{k}" if prefix else str(k))
        elif isinstance(node, bool):
            out[prefix] = node

    walk(_container(cfg), "")
    return out


def _container(cfg):
    from omegaconf import OmegaConf
    if OmegaConf.is_config(cfg):
        return OmegaConf.to_container(cfg, resolve=True)
    return cfg


def _select(cfg, dotted: str):
    """cfg(DictConfig / dict)からドットパスで値を取る。無ければ None。"""
    node = cfg
    for part in dotted.split("."):
        if node is None:
            return None
        try:
            node = node.get(part, None) if hasattr(node, "get") else None
        except Exception:                      # noqa: BLE001 (struct モード等の保険)
            return None
    return node


def _update(cfg, dotted: str, value) -> None:
    """cfg(DictConfig / dict)のドットパスへ値を書く(既存キーのみ)。"""
    from omegaconf import OmegaConf
    if OmegaConf.is_config(cfg):
        OmegaConf.update(cfg, dotted, value)
        return
    node = cfg
    parts = dotted.split(".")
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value


def undeclared_toggles(cfg) -> list[str]:
    """レジストリにも ALLOWLIST にも無い bool 型設定キーを返す(空ならレジストリ網羅)。"""
    return sorted(k for k in flatten_bools(cfg)
                  if k not in BY_ID and k not in ALLOWLIST)


def tier_rank(tier: str) -> int:
    return _TIER_RANK[tier]


def mode_of(cfg) -> str:
    mode = _select(cfg, "run.mode")
    mode = "none" if mode is None else str(mode).strip().lower()
    if mode not in MODES:
        raise ValueError(
            f"run.mode='{mode}' は未知(有効値: {', '.join(MODES)})。"
            "既定 none = 現行動作。")
    return mode


def is_enabled(feature: Feature, value) -> bool:
    """その機能が『いま有効』か。bool は True、非 bool は off_value 以外を有効とみなす。"""
    if value is None:
        return False
    if isinstance(value, bool):
        return bool(value)
    return value != feature.off_value


def _entry(f: Feature) -> dict:
    return {"id": f.id, "repro_tier": f.repro_tier, "affects_k": f.affects_k,
            "fingerprint_risk": f.fingerprint_risk}


@lru_cache(maxsize=1)
def _shipped_defaults() -> dict:
    """出荷時 conf/config.yaml の resolved container(『明示 ON』判定の基準)。"""
    from omegaconf import OmegaConf
    from .config import DEFAULT_CONFIG
    try:
        return OmegaConf.to_container(OmegaConf.load(DEFAULT_CONFIG), resolve=True)
    except Exception:                          # noqa: BLE001 (配布形態で conf が無い場合)
        return {}


# --------------------------------------------------------------------------- #
# 報告 / ゲーティング
# --------------------------------------------------------------------------- #
def describe(cfg, auto_disabled: list[dict] | None = None) -> dict:
    """いまの config が有効にしている機能の一覧と等級内訳(**副作用なし**)。"""
    mode = mode_of(cfg)
    max_tier = MODE_MAX_TIER[mode]
    enabled: list[dict] = []
    counts = {t: 0 for t in TIERS}
    for f in FEATURES:
        val = _select(cfg, f.id)
        if is_enabled(f, val):
            enabled.append(_entry(f))
            counts[f.repro_tier] += 1
    return {
        "run_mode": mode,
        "max_tier": max_tier,
        "registry_size": len(FEATURES),
        "counts": counts,
        "enabled": enabled,
        "auto_disabled": list(auto_disabled or []),
        "undeclared": undeclared_toggles(cfg),
    }


def _warn_auto_disabled(mode: str, max_tier: str, dropped: list[dict]) -> None:
    """自動 OFF を**目立つ形で**告知する(黙って落とさない)。"""
    explicit = [d for d in dropped if d["explicit"]]
    lines = ["",
             "=" * 72,
             f"[run.mode={mode}] 再現性等級 '{max_tier}' を超える機能 {len(dropped)} 件を"
             "自動 OFF にした",
             "=" * 72]
    for d in dropped:
        mark = "  ★明示 ON を落とした" if d["explicit"] else ""
        lines.append(f"  - {d['id']}  (tier={d['repro_tier']}){mark}")
    if explicit:
        lines.append("")
        lines.append(f"  ★ {len(explicit)} 件は出荷既定では OFF = **実験者が明示的に ON にした**"
                     "機能。モード指定と設定が矛盾している。")
    lines.append("  一覧は run_manifest.json の features.auto_disabled にも記録した。")
    lines.append("=" * 72)
    log.warning("\n".join(lines))


def apply_mode(cfg):
    """run.mode に従い、許される等級を超える機能を OFF にした config と報告を返す。

    返り値 (cfg, report)。**mode=none / observe では cfg を一切触らず同一オブジェクトを返す**
    (= 既定ランは 1 バイトも変わらない)。落とす場合のみ deepcopy してから書き換えるので、
    呼び出し側が渡した config オブジェクトは変更されない。
    """
    mode = mode_of(cfg)
    max_tier = MODE_MAX_TIER[mode]
    limit = _TIER_RANK[max_tier]

    targets: list[Feature] = []
    for f in FEATURES:
        if _TIER_RANK[f.repro_tier] <= limit:
            continue
        if is_enabled(f, _select(cfg, f.id)):
            targets.append(f)

    if not targets:
        return cfg, describe(cfg)

    defaults = _shipped_defaults()
    cfg = copy.deepcopy(cfg)
    dropped: list[dict] = []
    for f in targets:
        was = _select(cfg, f.id)
        _update(cfg, f.id, f.off_value)
        shipped = _select(defaults, f.id)
        dropped.append({
            "id": f.id, "repro_tier": f.repro_tier, "affects_k": f.affects_k,
            "fingerprint_risk": f.fingerprint_risk,
            "was": was, "now": f.off_value,
            # 出荷既定が OFF なのに ON だった = 実験者が明示的に立てた
            "explicit": not is_enabled(f, shipped),
            "reason": f"run.mode={mode} は repro_tier<= {max_tier} のみ許可",
        })
    _warn_auto_disabled(mode, max_tier, dropped)
    return cfg, describe(cfg, auto_disabled=dropped)


# --------------------------------------------------------------------------- #
# ラン間比較ガード(解析側)
# --------------------------------------------------------------------------- #
MANIFEST_NAME = "run_manifest.json"


def read_manifest(run_dir) -> dict | None:
    path = Path(run_dir) / MANIFEST_NAME
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def run_signature(run_dir) -> dict:
    """ラン1本の比較署名 {name, run_mode, tiers, known}。manifest が無い旧ランは known=False。"""
    name = os.path.basename(os.path.normpath(str(run_dir)))
    man = read_manifest(run_dir)
    if not isinstance(man, dict):
        return {"name": name, "dir": str(run_dir), "run_mode": "unknown",
                "tiers": [], "known": False}
    feats = man.get("features") or {}
    mode = str(man.get("run_mode") or (man.get("run") or {}).get("mode") or "unknown")
    tiers = sorted({str(e.get("repro_tier")) for e in (feats.get("enabled") or [])
                    if e.get("repro_tier")})
    return {"name": name, "dir": str(run_dir), "run_mode": mode,
            "tiers": tiers, "known": bool(feats) or mode != "unknown"}


def check_runs(run_dirs, allow_mismatch: bool = False) -> dict:
    """複数ランの run_mode / 有効機能の等級集合を照合する(読み出しのみ)。

    返り値:
      ok        … 混在なし(または allow_mismatch=True で続行を許可)
      mismatch  … 混在を検出したか
      refuse    … 既定(allow_mismatch=False)で比較を拒否すべきか
      messages  … 人間向けの説明行(呼び出し側が print する)
      signatures/unknown … 明細
    """
    sigs = [run_signature(d) for d in run_dirs]
    known = [s for s in sigs if s["known"]]
    unknown = [s for s in sigs if not s["known"]]
    modes = sorted({s["run_mode"] for s in known})
    tiersets = sorted({",".join(s["tiers"]) for s in known})
    mismatch = len(modes) > 1 or len(tiersets) > 1

    messages: list[str] = []
    if unknown:
        messages.append(
            f"⚠ run_manifest.json が無いラン {len(unknown)} 本は等級**不明**として扱う"
            f"(第72バッチ以前のラン): {', '.join(s['name'] for s in unknown)}")
    if mismatch:
        detail = "; ".join(f"{s['name']}: mode={s['run_mode']} tiers=[{','.join(s['tiers']) or '-'}]"
                           for s in known)
        messages.append(
            "✖ 再現性等級の異なるラン同士を比較しようとしている(observe ランと verify ランを"
            f"無自覚に並べるのが最も危険な誤読)。{detail}")
        if allow_mismatch:
            messages.append(
                "⚠ --allow-tier-mismatch が指定されたので続行する。"
                "**等級混在の比較であることをレポートに明記すること**。")
        else:
            messages.append(
                "→ 意図した比較なら --allow-tier-mismatch を付けて再実行する。")
    return {"ok": (not mismatch) or allow_mismatch, "mismatch": mismatch,
            "refuse": mismatch and not allow_mismatch,
            "messages": messages, "signatures": sigs,
            "modes": modes, "tiersets": tiersets,
            "unknown": [s["name"] for s in unknown]}


def _safe_echo(msg: str) -> None:
    """print の cp932 安全版。Windows の非TTY stdout(cp932)では警告文の ⚠/✖ が
    UnicodeEncodeError になり**解析スクリプトごと落ちる**(小粒C検収で実測)。
    本文は変えず、表示できない文字だけ置換して必ず出力する。"""
    try:
        print(msg)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(msg.encode(enc, errors="replace").decode(enc, errors="replace"))


def guard_or_die(run_dirs, allow_mismatch: bool = False, echo=_safe_echo) -> dict:
    """check_runs を実行し、拒否条件なら SystemExit で止める(解析スクリプト用の口)。

    - 同一モード・同一等級集合 … **無言**(何も出力しない)
    - manifest 欠落 … 警告のみで続行(過去ラン互換)
    - 等級混在 … 既定は SystemExit / allow_mismatch=True なら警告つき続行
    """
    rep = check_runs(run_dirs, allow_mismatch=allow_mismatch)
    for m in rep["messages"]:
        echo(m)
    if rep["refuse"]:
        raise SystemExit(
            "[tier-guard] 再現性等級が混在するラン群の比較を拒否した"
            "(--allow-tier-mismatch で明示的に続行できる)")
    return rep
