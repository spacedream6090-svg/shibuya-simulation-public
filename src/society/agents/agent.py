"""エージェント = 状態の容器。行動ロジックは cognition/、更新則は factors/ に置く。"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .memory import MemoryStore


@dataclass
class Agent:
    id: int
    name: str
    age: int
    occupation: str
    persona: str                      # LLM に渡す自己紹介文(構成概念は含めない)
    traits: dict[str, float]
    states: dict[str, float]
    gender: str = "?"
    visitor: bool = False             # 来街者(夜は範囲外の自宅へ帰る)。渋谷昼間人口の~56%
    # ---- 流入通勤者(commuter = visitor の下位型。区外→朝流入・勤務・夕退出)----
    #  ユーザー要望 2026-07-06 / docs/research/shibuya-inflow.md(昼間就業者の~74%が区外流入)
    commute: bool = False             # 通勤・通学流入者か(visitor かつ職場が範囲内)
    arrival_min: int = -1             # 朝の到着時刻(分 of day、二峰分布から。-1=非commuter)
    commute_gateway: str = "station"  # 到着/帰宅の出入口種別("station" | "edge")
    residence_line: str = ""          # 居住=沿線タグ(話題接地用メモ。構成概念ではない)
    # ---- 空間 ----
    node: str = ""                    # 現在ノード(移動中は直前ノード)
    x: float = 0.0
    y: float = 0.0
    route: list[str] = field(default_factory=list)   # 残りの経路(次ノードから)
    edge_offset: float = 0.0          # 現在エッジ上の進捗(m)
    dest: str | None = None
    stay_until: int = 0               # この step まで滞在
    visits: Counter = field(default_factory=Counter)  # EPR の preferential return 用
    # ---- 移動手段(自由度: 徒歩/自転車/車)----
    has_bicycle: bool = False
    has_car: bool = False
    trip_mode: str = "walk"           # 現在の移動のモード
    # ---- 所在(street | outside)+ 建物内 ----
    loc: str = "street"
    building: str | None = None       # 建物 id(屋内のとき)
    floor: int = 0                    # 何階にいるか
    exit_intent: bool = False         # ゲートウェイ到着後に範囲外へ出る意図
    return_at: int = 0                # 範囲外からの帰還予定 step
    return_gateway: str = ""
    # ---- 家・睡眠(個別タイミング → 内省=記憶整理 LLM も自然に分散)----
    home_node: str = ""
    home_building: str = ""           # 実在の住宅・マンション建物 id(v3)
    home_floor: int = 1
    bedtime_min: int = 1320           # 就寝時刻(分 of day、個体差あり)
    sleep_steps: int = 42             # 睡眠の長さ(step)
    sleeping: bool = False
    sleep_until: int = 0
    homing: bool = False              # 帰宅移動中
    reflect_step: int = -1            # 就寝直後に1回の内省(記憶整理)
    # ---- 職場・活動(v3: POI ベースの日課)----
    work_name: str = ""               # 職場の実名(POI 名)。"" = 決まった職場なし
    work_node: str = ""
    work_building: str = ""           # 職場が入る建物 id("" = 路面)
    work_floor: int = 0
    work_start_min: int = -1          # -1 = 職場なし
    work_end_min: int = 0
    has_phone: bool = True            # スマホ(SNS/検索/ニュース/DM/地図)
    activity: str = ""                # working / eating / shopping / ...(表示・プロンプト用)
    said: list[str] = field(default_factory=list)  # 自分の直近発話(反復抑制用)
    # ---- 朝の一日計画(ユーザー要望 2026-07-06。routine の行き先の土台)----
    day_plan: list = field(default_factory=list)   # [{when, what, place, done}]。翌朝は上書き
    plan_step: int = -1               # この step で計画を生成する(起床/帰還の直後 step)
    plan_day: int = -1                # 計画を立てた日(1日1回に制限)
    # ---- 長期予定・スケジュール帳(第7バッチ 2026-07-07。会話由来の自動記入。既定 空)----
    #  各 = {day, when, what, place, with, src_step}。schedule 有効時のみ書き込まれる(OFF=空)。
    schedule: list = field(default_factory=list)
    # ---- 経済 v0(賃金+消費+バイト。ユーザー決定 2026-07-05)----
    money: float = 0.0                # 手持ちの現金(円)。職業別に persona 生成時決定
    wage: float = 0.0                 # 本業の勤務完遂で支給する日給(0=無給・学生・無職)
    part_time: dict | None = None     # バイト先(POI node/building・シフト曜日/時間帯)。None=なし
    # ---- 口座(銀行)概念 E5(既定 OFF。ユーザー決定 2026-07-06。100日=月次資金繰り)----
    #  ON 時のみ意味を持つ(OFF 時は全て 0 のまま=挙動バイト一致)。
    account: float = 0.0              # 口座残高(円)。賃金・バイト・日銭・売上の入金先
    work_days: int = 0                # 給料日までに完遂した本業勤務日数(月給まとめ支給の元)
    last_salary: float = 0.0          # 直近の月給支給額(観測用)
    period_income: float = 0.0        # 家賃計算の元: 前回家賃日からの入金累計(=月収相当)
    rent_due: float = 0.0             # 家賃の未払い残(残高不足→翌日繰越)
    # ---- 記憶・言語 ----
    heard_counts: Counter = field(default_factory=Counter)  # item_id -> 聞いた回数
    adopted: set[str] = field(default_factory=set)           # 採用した item_id
    beliefs: list[str] = field(default_factory=list)         # 内省の書き戻し先(k の作用点)
    mem: MemoryStore = field(default_factory=MemoryStore)    # 記憶 v2(3層)
    now_step: int = 0                 # 現在 step(remember の時刻付けに使用)
    # ---- 欲求駆動発火(Phase A)----
    drive: float = 0.0                # 欲求ゲージ(0-1)。出来事で溜まり思考で減る
    # ---- 感情・興味・注意ハブ(affect、既定 OFF。設計: docs/lit/neuroscience__emotion-interest-attention.md)----
    #  arousal = core affect の第2軸(覚醒度)。drive ゲージと並列の内部 transient(states 監査集合に
    #  入れない=R²(k) を汚さない)。既定 baseline 0.2。affect OFF 時は誰も読まない=バイト一致。
    arousal: float = 0.2
    drive_mods: dict | None = None    # SDT 動機プラグイン(既定 OFF): reason→倍率。None=OFF
    drive_threshold: float = 0.55     # 申請閾値(個人差、factors で traits から写像)
    fire_weight: float = 0.5          # 申請時の発火確率(個人の重み)
    refractory_until: int = 0         # 不応期(発火直後は考え込まない)
    # ---- 内省しやすさの時間ドリフト E2(既定 OFF。設計: docs/research/reflection-drift.md)----
    #  実効閾値 = clip(drive_threshold + theta_drift, 0.30, 0.85)。fire_weight は不変。
    #  OFF 時は theta_drift=0・drift_p=None のまま(=drive_threshold と恒等=バイト一致)。
    theta_drift: float = 0.0          # 緩変数(馴化+/鋭敏化−/自発回復→0)。0=中立
    drift_p: dict | None = None       # 個人別ドリフト率(traits→factors 写像)。None=OFF
    # ---- 対面会話(確定発火 + 返答保証。ユーザー仕様 2026-07-05)----
    _reply_to: tuple[int, str] | None = None  # (話者id, 発話文): 話しかけられた→必ず返答
    conv_turns_left: int = 3          # 今の会話で返答できる残り回数(初期値=cfg)
    conv_cooldown_until: int = 0      # 会話クールダウン明けの step(0=会話可)
    # ---- 意見力学(Friedkin-Johnsen。#16、非LLM 決定論更新。プロンプト非注入)----
    opinion: float = 0.0              # 現在の意見(-1..1)。初期値=anchor
    opinion_anchor: float = 0.0       # 固定アンカー=初期意見(FJ の stubbornness 源)
    opinion_susceptibility: float = 0.5  # 社会的影響の受けやすさ s(traits→写像)
    # ---- 健康・疲労・病気・メンタル(後続波 H1、既定 OFF。src/society/health.py)----
    #  すべて中立値(OFF 不変)。health.enabled=false ではどれも変化せず=挙動バイト一致。
    #  fatigue = drive/arousal と並列の内部 transient(states 監査集合に入れない=R²(k) を汚さない)。
    fatigue: float = 0.0              # 疲労ゲージ(0..1)。活動で+、睡眠で回復(既定 gain 0=不変)
    sick: bool = False                # 病気で在宅療養中(欠勤・外出抑制)。OFF/健康時は False
    sick_until: int = -1              # この step まで病気(-1=非病気)
    withdrawn: bool = False           # 引きこもり傾向(慢性高 grievance→自由時間の外出抑制)
    chronic_days: int = 0             # 高 grievance の連続日数(メンタル判定用のカウンタ)
    # ---- 世帯・家族・恋愛(後続波 H2、既定 OFF。src/society/household.py)----
    #  すべて中立(OFF 不変)。household.enabled=false ではどれも設定されず=挙動バイト一致。
    household_id: str | None = None   # 所属世帯 id(同一世帯は home を共有)。None=世帯なし=個人
    household_kind: str = ""          # "family"(家族)/ "roommate"(同居人)。""=世帯なし
    household_role: str = ""          # 続柄(現実化 S-R1: 夫/妻/親/子/同居人)。""=未割当(既定・非現実化)
    housemates: list = field(default_factory=list)  # 同居者の id(自分を除く)。既定は空
    partner_id: int | None = None     # 恋人の id(相互の強い親密度から決定論成立)。None=独身
    # ---- 共同行動エンジン(関係性の再現 S-R3、既定 OFF。src/society/joint.py)----
    #  中立(OFF 不変)。joint.enabled=false では設定されず=挙動バイト一致。日次境界で当日分を編成し
    #  クリアする。{poi, band, activity, activity_tag, tier}。None=当日の共同行動なし。
    joint_today: dict | None = None
    # ---- 宿泊・ホテル滞在(後続波 Wave L、既定 OFF。src/society/lodging.py)----
    #  すべて中立(OFF 不変)。lodging.enabled=false ではどれも設定されず=挙動バイト一致。visitor が
    #  夜に範囲外退出する代わりにホテルへ泊まる物理機構(移動→チェックイン→就寝→checkout_hour に退館)。
    lodging: bool = False             # ホテルにチェックイン中(夜間滞在=就寝中)。OFF/非宿泊時は False
    lodging_intent: bool = False      # ホテルへ移動中(到着でチェックイン)。到着で False に落ちる
    lodging_node: str = ""            # 宿泊先ホテルの最寄りノード(移動先/到着判定)
    lodging_poi: str = ""             # 宿泊先ホテル名(lodging_checkin/checkout イベント用)
    lodging_building: str | None = None  # ホテル建物 id(あれば屋内滞在)。None=路面ノード泊
    lodging_nights: int = 0           # 連泊数(max_nights 判定用)。帰宅=退出でリセット
    # ---- 勾留(制度深化2 第10バッチ、既定 OFF=0 のまま。執行時の数stepの行動停止)----
    detained_until: int = 0           # この step まで拘束(会話・発火・移動なし)。0=非拘束
    # ---- 立退き・破産サイクル(制度深化3 2026-07-08、既定 OFF=すべて中立値のまま)----
    #  口座 E5(economy.accounts)ON かつ eviction_days/bankruptcy_days > 0 のときだけ動く。
    arrears_days: int = 0             # 家賃滞納(rent_due>0)の連続日数(暦日カウント)
    evicted: bool = False             # 立退き中(自宅建物に入れない=路上就寝。完済で再入居)
    bankrupt_until: int = 0           # この step まで破産の制限期間(出店不可)。0=非破産
    # ---- 反射=自己モデル(第11バッチ 2026-07-08、既定 OFF=None のまま。Generative Agents の
    #  reflection に倣う: N日ごとの深い内省で自己像を生成し、以後のプロンプトへ1行注入して
    #  再帰的に更新する。reflection.self_model_days=0 では一切書かれない=バイト一致)----
    self_model: dict | None = None    # {"self": 自己像一文, "ties": 大事な関係一文, "day": 更新日}
    # ---- 社会的地位(第11バッチ・ヒエラルキー、既定 OFF=0 のまま。src/society/status.py)----
    #  客観カウント(評判・フォロワー・資産順位・制度実績・商い・主催)から日次で決定論算出する
    #  合成地位スコア(0..1)。hierarchy.enabled=false では誰も書かない=バイト一致。
    #  states 監査集合には入れない(arousal/fatigue と同じ内部量=R²(k) を汚さない)。
    status: float = 0.0
    # ---- 出来事誘発の深い内省+無意識の自己認識層(第12バッチ 2026-07-08、既定 OFF=中立値)----
    #  文献検証: docs/research/deep-reflection-triggers.md / self-concept-identity.md。
    #  reflection.deep / reflection.implicit_self が OFF のときは誰も書かない=バイト一致。
    impact_today: float = 0.0         # 日内衝撃ゲージ(信念との乖離の近似。ネガ非対称加重)
    impact_neg_today: float = 0.0     # 生のネガ量(grievance↑・efficacy↓)。無意識層の材料
    impact_pos_today: float = 0.0     # 生のポジ量(grievance↓・efficacy↑)
    reflect_p: dict | None = None     # 個人パラメータ(閾値・非対称・遅延。traits→registry 写像)
    deep_due_day: int = -1            # 深い内省の予約日(閾値超え日+incubation)。-1=予約なし
    deep_cooldown_until_day: int = -1  # この日まで再予約しない(反芻の抑制)
    implicit_self: str = ""           # 無意識層「最近の自分」1行(揮発的な作動自己。日次更新)
    behav_today: dict | None = None   # 当日の行動・経験カウント(reason→回数)。None=OFF
    behav_ema: dict | None = None     # 行動ベースライン EMA(+感情価 _neg/_pos)。None=OFF

    def remember(self, text: str, kind: str = "event",
                 importance_bonus: float = 0.0) -> None:
        self.mem.observe(self.now_step, text, kind=kind,
                         importance_bonus=importance_bonus)
