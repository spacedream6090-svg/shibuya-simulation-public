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
    _f("run.dt_min", "strict", False, "none",
       "中央 Δt(1 step の分数。既定 10=正準=golden の世界)。timeconv.py の分類テーブルに従い"
       "レート/確率/step 長を config ロードの 1 箇所で変換する。10 のときはテーブルを走査せず"
       "config を 1 バイトも触らない。Δt≠10 は乱数消費列が変わる別世界(統計量の同オーダーで検証)",
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

    # ---- economy ----
    _f("economy.enabled", "strict", False, "none",
       "経済 v0(賃金・消費・バイト・金銭圧力の心理接続)"),
    _f("economy.visitor_refresh", "strict", False, "none",
       "来街者が帰宅から戻るたび手持ちを補充(長期ランの恒久破綻対策)"),
    _f("economy.accounts.enabled", "strict", False, "none",
       "口座(銀行)概念。月給・家賃引落・カード/現金・ATM・立退き/破産サイクル"),

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
    _f("observer.llm_health.enabled", "strict", False, "none",
       "L2 に LLM 健全性 KPI 3列を足す(観測専用・累積カウンタ)"),
    _f("observer.echo.enabled", "strict", False, "none",
       "L2 にエコー/自己反復 5列を足す(観測専用・常設)"),
    # 第78バッチ: 状態ハッシュチェーン(verify 用)。記録専用でシムは読まない=strict。
    _f("observer.state_hash.enabled", "strict", False, "none",
       "各 step の world state を正準シリアライズ→sha256→前 step と連鎖させて "
       "state_hash.jsonl に書く(記録専用・シムは読まない・T1/T6 の判定装置)"),

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
    _f("household.enabled", "strict", False, "possible",
       "世帯・家族・恋愛のマスター"),
    _f("household.realistic", "strict", False, "none",
       "地理×年齢の整合束ね+続柄+規模重みで現実的な世帯を作る"),
    _f("household.family_dinner.enabled", "strict", False, "none",
       "家族の夕食共食"),
    _f("household.cohabit.enabled", "strict", False, "none",
       "同棲=世帯の物理再編"),
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


def guard_or_die(run_dirs, allow_mismatch: bool = False, echo=print) -> dict:
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
