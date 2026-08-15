"""Δt(1 step の分数)の中央管理と定数の毎分レート化(第79バッチ)。

正典: docs/plans/source/cognition-design-record.md §5.2-5.3 /
      docs/plans/cognition-physics-plan.md §4 第79行。

何を解く問題か
--------------
本リポジトリの定数は **1 step = 10 分**を暗黙の前提にして直書きされている
(`eta_m_per_step`、`decay: 0.02 /step`、`cooldown_steps: 6` …)。この状態では
Δt を変えた瞬間に世界の挙動が全部ずれ、較正も壊れる。そこで

  1. 「Δt=10 分」という仮定を **conf の 1 キー `run.dt_min`** に集約し、
  2. 各定数が Δt に対してどう振る舞うかを **本 module のテーブル 1 枚**で宣言し、
  3. 変換は **config ロードの 1 箇所**(config.load_config)で一括適用する。

これで Δt が実験パラメータになる(Δt 掃引が条件になる)。

分類(ここが本 module の本体)
------------------------------
ABM の time-step 正規化で最も多い事故は「per-step 確率を Δt に線形スケールする」こと。
確率は線形では合成できない(p=0.35 を Δt=1/10 にして 0.035 にすると、10 step 通した
成功確率は 1−0.965^10 = 0.30 ≠ 0.35)。次元で分けて扱う:

  RATE     … 単位 [量/step]。**線形**。 v' = v × (Δt/10)
              例: 移動速度 m/step、毎 step のゲージ加算、毎 step の疲労蓄積。
  PROB     … 単位 [無次元・毎 step の Bernoulli 成功率 or 毎 step の乗算減衰率]。
              **べき変換**。 p' = 1 − (1−p)^(Δt/10)
              (= ハザード λ = −ln(1−p)/10 を一定に保つ変換。Δt→0 で p'→λ·Δt に一致し、
               Δt を粗くしても p' が 1 を超えない。**線形スケールは不可**)
              「毎 step x *= (1−d)」の d もこの族(1 step の損失割合)。
  KEEP     … 単位 [無次元・毎 step の残存割合](PROB の裏返し)。 k' = k^(Δt/10)
              例: memory の recency_decay 0.9983/step。
  STEPS    … 単位 [step](持続長・窓・クールダウン・TTL)。**逆比例**。
              n' = max(1, round(n × 10/Δt))。ただし n=0 は 0(=無効の意味)を保つ。
              ★これは指示された 3 分類に無い第4の類。実査で 20 件超あり、
                「カウントだから不変」で放置すると Δt=5 で滞在時間が半分になる。
  INVARIANT… Δt に依存しない量(件数・閾値・金額・per-day / per-hour のレート・
              分 of day の時刻帯・物理秒)。**理由を必ず添える**(下記 TABLE の第3要素)。

R1 ドクトリンとの関係
---------------------
- 既定 `run.dt_min == 10` では `apply_dt()` は **config を 1 バイトも書き換えない**
  (テーブルを走査すらせず同一オブジェクトを返す)。恒等式になる変換でも浮動小数の
  同値性を信用しない、というのが本 module の設計上の要請(golden が絶対の検収)。
- Δt≠10 は「別の世界」= 乱数消費列が変わることを許容する(同一性は要求しない)。
  要求するのは「1 日あたりの統計量が同オーダーに収まる」ことだけ(T3 の緩い版)。
"""
from __future__ import annotations

# --------------------------------------------------------------------------- #
# 正準 Δt(golden の世界)
# --------------------------------------------------------------------------- #
CANON_DT_MIN = 10
MINUTES_PER_DAY = 1440

# 分類ラベル
RATE = "rate"
PROB = "prob"
KEEP = "keep"
STEPS = "steps"
INVARIANT = "invariant"

CLASSES = (RATE, PROB, KEEP, STEPS, INVARIANT)


# --------------------------------------------------------------------------- #
# 変換ヘルパ(純関数。Δt=10 は厳密な恒等)
# --------------------------------------------------------------------------- #
def ratio(dt: int | float) -> float:
    """Δt / 10。1 step が正準の何倍の時間か。"""
    return float(dt) / float(CANON_DT_MIN)


def scale_rate(v: float, dt: int | float) -> float:
    """[量/step] を線形にスケールする。"""
    if int(dt) == CANON_DT_MIN:
        return v
    return float(v) * ratio(dt)


def scale_prob(p: float, dt: int | float) -> float:
    """[毎 step の Bernoulli 成功率 / 乗算減衰率] をべき変換する。

    p' = 1 − (1−p)^(Δt/10)。ハザードを保つ = 10 分あたりの「少なくとも1回起きる確率」が不変。
    """
    if int(dt) == CANON_DT_MIN:
        return p
    p = float(p)
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    return 1.0 - (1.0 - p) ** ratio(dt)


def scale_keep(k: float, dt: int | float) -> float:
    """[毎 step の残存割合] をべき変換する。k' = k^(Δt/10)。"""
    if int(dt) == CANON_DT_MIN:
        return k
    k = float(k)
    if k <= 0.0:
        return 0.0
    return k ** ratio(dt)


def scale_steps(n: int | float, dt: int | float) -> int:
    """[step] で表された持続長・窓・TTL を逆比例でスケールする。

    n=0 は「無効/即時」の意味で使われているので 0 のまま(1 に繰り上げない)。
    """
    if int(dt) == CANON_DT_MIN:
        return int(n)
    n = int(round(float(n)))
    if n == 0:
        return 0
    scaled = int(round(n * float(CANON_DT_MIN) / float(dt)))
    return max(1, scaled) if n > 0 else min(-1, scaled)


def steps_per_day(dt: int | float) -> int:
    """1 日の step 数(Δt=10 で 144)。"""
    return max(1, int(round(MINUTES_PER_DAY / float(dt))))


def steps_per_hour(dt: int | float) -> int:
    """1 時間の step 数(Δt=10 で 6)。"""
    return max(1, int(round(60.0 / float(dt))))


def step_seconds(dt: int | float) -> float:
    """1 step の秒数(Δt=10 で 600.0)。"""
    return float(dt) * 60.0


# --------------------------------------------------------------------------- #
# 分類テーブル(= この作業の成果物本体。散在させない)
#
#   key … conf のドットパス。`*` は 1 セグメントのワイルドカード。
#   cls … 上の分類ラベル。
#   why … INVARIANT は理由必須(なぜ Δt で動かさないか)。他は意味の説明。
#
# ★網羅性の担保: tests/test_timeconv.py が conf/config.yaml を「1 step 前提」を
#   示唆する正規表現(per_step / /step / _steps / 10分 …)で走査し、ヒットした
#   キーが全て本テーブルに載っていることを検査する。載っていなければ CI が落ちる。
# --------------------------------------------------------------------------- #
TABLE: tuple[tuple[str, str, str], ...] = (
    # ---- run ----
    ("run.n_steps", INVARIANT,
     "ランの長さ=実験者が指定する量。自動換算すると計算コストが黙って倍増するため"
     "触らない(Δt を変えるときは呼び出し側が n_steps も換算する)"),

    # ---- 移動速度(m/step)= 典型的な RATE ----
    ("world.modes.speeds.*", RATE, "移動距離 m/step"),
    ("world.traffic.free_speed_min", RATE, "od 自由流速度の下限 m/step"),
    ("world.traffic.free_speed_max", RATE, "od 自由流速度の上限 m/step"),
    ("delivery.eta_m_per_step", RATE, "配送トリップの走行速度 m/step"),

    # ---- 毎 step の Bernoulli(行き先バイアス・発生確率)----
    ("world.exit_prob", PROB, "暇なとき範囲外へ出かける確率/step"),
    ("world.building_enter_prob", PROB, "入口に居るとき建物に入る確率/step(exit と同じ draw)"),
    ("world.meal_prob", PROB, "食事帯に飲食店へ行く確率/step"),
    ("rules.boost_prob", PROB, "weekly_event 発火日に余暇候補へ選ぶ確率/step"),
    ("net.browse_prob", PROB, "SNS タイムラインを見る確率/step"),
    ("net.news_prob", PROB, "ニュースアプリを見る確率/step"),
    ("conversation.c2.meet_prob", PROB, "同席1対あたり毎 step の会話成立確率"),
    ("routine.stochastic.interrupt_prob.*", PROB, "滞在中の活動を中断する確率/step"),
    ("media.start_prob", PROB, "在宅×帯の 1 step あたり視聴開始確率"),
    # ---- 在宅覚醒 HOME_AWAKE(β9)。ハザードの切片だけが「1 step あたり」の量 ----
    #  p_sleep = sigmoid(logit(p0) + b1*概日 + b2*疲労 + b3*翌日早出 − b4*没入)。
    #  切片を**確率 p0** で持たせてあるのは、まさにこの PROB 変換を通すため
    #  (対数オッズのまま conf に置くと既存 4 分類のどれにも乗らず、Δt≠10 で静かに壊れる)。
    ("daily.home_awake.hazard.p0", PROB,
     "就寝ハザードの切片 = 就寝時刻ちょうど・他項ゼロのときの 1 step あたり就寝確率。"
     "コード側は logit(p0) を切片に使うので、この変換で 10 分あたりのハザードが保たれる"),
    ("annual_events.crowd_bias", PROB, "群集日に集会ノードへ寄せる確率/step"),
    ("household.date_bias", PROB, "自由時間にデート先へ寄せる確率/step"),
    ("household.family_dinner.prob", PROB, "夕食帯の home 収束バイアス/step"),
    ("party.roam_bias", PROB, "回遊帯に共有 POI へ寄せる確率/step"),
    ("spark.menus.anchor.bias", PROB, "集会アンカーへ寄せる確率/step"),
    ("society_diversity.tourist_bias", PROB, "ランドマークへ回遊する確率/step"),
    ("society_diversity.crime_prob", PROB, "1人1step あたりの窃盗発生確率"),
    ("society_diversity.nuisance_prob", PROB, "1人1step あたりの迷惑行為発生確率"),
    ("delivery.order_rate", PROB, "食事帯の滞在 step あたりの注文確率"),
    ("pov.salience.prob_cap", PROB, "撮影の確率上限/step"),
    ("drive.boredom.fire_prob", PROB, "閾値超え時に探索へ動く確率/step"),

    # ---- 毎 step の乗算減衰(数学的には PROB と同族)----
    ("drive.decay", PROB, "ゲージの自然減衰/step(drive *= 1−decay)"),
    ("drive.boredom.decay", PROB, "退屈ゲージの減衰率/step"),
    ("drive.drift.recovery_rate", PROB, "theta_drift を毎 step 0 へ引く割合"),
    ("affect.arousal_decay", PROB, "arousal を毎 step baseline へ戻す割合"),

    # ---- 毎 step の加算(ゲージ・疲労)----
    ("drive.weights.silence", RATE,
     "沈黙=毎 step 加算されるゲージ入力(他の weights は出来事1件あたり=不変)"),
    ("drive.boredom.accrual", RATE, "長居 1step あたりのゲージ蓄積"),
    ("health.fatigue_gain_work", RATE, "勤務中 1step あたりの疲労蓄積"),
    ("health.fatigue_gain_move", RATE, "移動中 1step あたりの疲労蓄積"),
    ("health.fatigue_recovery", RATE, "睡眠中 1step あたりの回復量"),
    ("spark.menus.anchor.decay", RATE,
     "bias × exp(−decay×step) の指数係数(exp の肩が step 単位なので線形)"),

    # ---- 毎 step の予算 ----
    ("lod.max_llm_per_step", RATE, "LLM 発火の step 上限(1日あたりの総量を保つ)"),
    # DPH-B 二層予算の FIFO 繰り越し上限。**実時間の長さ**(既定 18 step = 3 時間 @Δt10)
    # として意味を持つ量なので STEPS = 実時間を保つ側へ倒す。ここを RATE にすると
    # Δt=1 で「18 分待ったら骨格へ落とす」になり、朝の計画がほぼ全部骨格へ落ちる。
    # 兄弟キー reply_share / life_share は **cap に対する割合**(無次元)なので
    # timeconv の棚卸し grep(`_steps` / `_min` / `per_step` 等)に掛からず分類不要。
    ("lod.budget.tiers.max_defer_steps", STEPS,
     "朝の計画/夜の内省を予算不足で繰り越してよい上限 [step](超過で骨格へ後退)"),
    ("env.feedback.transit.dwell_sec_per_pax", RATE,
     "超過1人あたりの停車時間延長[秒]。**1 step ぶんの注入量**として使うので、Δt が伸びれば"
     "その step に含まれる発車回数が増える=線形にスケールする(第84バッチ)"),
    ("env.feedback.transit.dwell_cap_min", RATE,
     "1 step ぶんの注入の上限[分]。上の注入量そのものの天井なので**同じ次元=同じスケール**で"
     "動かす(不変にすると Δt を細かくしたとき天井だけが相対的に高くなる。第84バッチ)"),
    ("env.feedback.transit.recovery", KEEP,
     "回復運転項 γ<1。**毎 step 遅延に掛ける残存割合**なので Δt でべき変換する"
     "(γ' = γ^(Δt/10))。線形にすると Δt を細かくしたとき遅延が消えなくなる(第84バッチ)"),

    # ---- step 単位の持続長・窓・TTL・クールダウン(逆比例)----
    ("world.outside_steps", STEPS, "範囲外滞在の長さ [step]"),
    # ★§1.2 B1(第94バッチ OBS-U2): scenario の引数は基底 conf が `{}` なので
    #   _iter_targets(実在キーのみ走査)にも棚卸し正規表現にも掛からない = 唯一の
    #   「宣言しないと永久に見つからない」穴だった。ラン側が dotlist/profile で
    #   world.scenario_params={at_step: 36, duration_steps: 18} を渡した瞬間に
    #   実在キーになるので、ここに載せておけば apply_dt が拾う。at_step は**時点**だが
    #   逆比例で正しい(Δt=10 の step 36 = Δt=1 の step 360 = 同じ実時刻)。
    #   基底に実在しない唯一の宣言なので、tests/test_timeconv.py の
    #   test_every_table_key_exists_in_shipped_config が理由つきで除外している。
    ("world.scenario_params.at_step", STEPS,
     "scenario 発動時点 [step]。実時刻を保つため逆比例(scale_steps は 0 を 0 のまま残す)"),
    ("world.scenario_params.duration_steps", STEPS,
     "scenario の継続時間 [step]。実時間の長さを保つため逆比例"),
    ("indoor.markov.dwell_steps", STEPS, "平均滞在 step(遷移確率 = 1/dwell)"),
    ("world.devices.faregate.max_hold_steps", STEPS,
     "改札待ちで持ち越せる上限 [step]。実時間の安全弁を保つため逆比例"),
    ("transit_staff.dwell.log_every_steps", STEPS,
     "dwell_decision の記録周期 [step]。実時間の記録密度を保つため逆比例"
     "(Δt=1 で不変のままだと L1 行数が10倍に膨れる)"),
    # ---- ラッシュ時の車内(Wave 4 II-1)。**時間らしいキーは 3 種**に分かれる ----
    ("transit_interior.conductor.patrol_interval_steps", STEPS,
     "車掌が 1 両進む間隔 [step]。巡回の**実時間の速さ**を保つため逆比例"
     "(Δt=1 で不変のままだと巡回が 10 倍速くなり、1 編成を舐める実時間が 1/10 になる)"),
    ("transit_interior.comfort.fatigue_per_step", RATE,
     "1 step・超過密度 1.0 あたりの疲労蓄積 [/step]。毎 step の加算量なので線形"
     "(health.fatigue_gain_* と同族)"),
    ("transit_interior.ride_minutes", INVARIANT,
     "乗車の長さ [**分**]。step ではなく実時間の分なので Δt に依らない"
     "(step への換算はコード側の ride_steps が max(1, round(分/Δt)) で行う"
     "= A1/A9 と同じ『分を持って step へ換算する』側の作法)"),
    ("transit_interior.congestion_scale.windows", INVARIANT,
     "時間帯の混雑倍率 [[開始分, 終了分, 倍率]]。**分 of day の時刻帯**と無次元倍率で"
     "step ではない(indoor.meeting.window_min と同じ族)"),
    ("transit_interior.copresence.max_pairs_per_day", INVARIANT,
     "1 日 1 人あたりに記録する同席の対の上限 [件/人/day]。per-day のレートなので"
     "Δt 非依存(1 日の実時間の長さは Δt を変えても 1440 分のまま)"),
    ("transit_interior.copresence.max_pairs_per_car", INVARIANT,
     "1 乗車 1 人あたりに控える同席相手の上限 [件]。件数であって時間量ではない"),
    ("transit_interior.comfort.fatigue_max", INVARIANT,
     "1 **乗車**あたりに乗せる疲労の上限。出来事(乗車)単位の総量であって"
     "per-step のレートではない(freedom.sat_step と同じ族)"),
    ("tools.permit_steps", STEPS, "許可待ちの長さ [step](0=待ちなし)"),
    ("tools.event_duration_steps", STEPS, "イベント開催時間 [step]"),
    ("tools.flyer_ttl_steps", STEPS, "チラシの寿命 [step]"),
    ("drive.refractory_steps", STEPS, "発火後の不応期 [step]"),
    ("drive.conv_cooldown_steps", STEPS, "会話後のクールダウン [step]"),
    ("drive.boredom.cooldown_steps", STEPS, "探索発火後の不応期 [step]"),
    ("drive.boredom.stay_steps", STEPS, "探索先での滞在 [step]"),
    ("conversation.c2.cooldown_steps", STEPS, "C2 再突入までの間隔 [step]"),
    ("transit_ride.bus.headway_steps", STEPS, "便の間隔 [step]"),
    ("ads.cooldown_steps", STEPS, "同一広告の再接触の最短間隔 [step]"),
    ("ads.recall_steps", STEPS, "プロンプト注入の想起窓 [step]"),
    ("observer.snapshot_every", STEPS, "L3 スナップショットの間隔 [step](実時間の粒度を保つ)"),
    ("observer.echo.window_steps", STEPS, "エコー計測の rolling 窓 [step]"),
    # 第91バッチ 退行シグナル監視(観測専用)。窓は実時間を保ち、飽和閾値は
    # 「発火数/step」なので 1 日あたりの総量が保たれるようにスケールする(lod.max_llm_per_step と同型)。
    ("observer.regression.window_steps", STEPS, "退行シグナルの rolling 窓 [step]"),
    ("observer.regression.fire_sat_per_step", RATE,
     "発火率が飽和に張り付いたとみなす閾値 [発火/step](1日あたりの総量を保つ)"),
    ("media.min_steps", STEPS, "視聴セッション長の下限 [step]"),
    ("media.max_steps", STEPS, "視聴セッション長の上限 [step]"),
    ("institution_routes.enforcement.detention_steps", STEPS, "拘束の長さ [step]"),
    # 第95バッチ IF-C(噂)。TTL は実時間で同じ長さを保ち、生成上限は「件/step」なので
    # 1 日あたりの総量が保たれるようにスケールする(lod.max_llm_per_step と同型)。
    ("information.rumors.forget_steps", STEPS,
     "噂の忘却 TTL [step](0=忘れない=無効の意味なので 0 のまま保たれる)"),
    ("information.rumors.max_per_step", RATE,
     "1 step に生む噂の上限 [件/step](1日あたりの総量を保つ安全弁)"),
    # 第96バッチ IF-D(痕跡)。半減期は実時間で同じ長さを保つ(0=減衰しない=無効の意味なので
    # 0 のまま保たれる)。集約上限は「件/step」なので 1 日あたりの総量が保たれるようにする。
    ("world.traces.half_life_steps.*", STEPS,
     "痕跡の蒸発の半減期 [step](0=減衰しない persistent の意味なので 0 のまま保たれる)"),
    ("world.traces.max_per_step", RATE,
     "1 step に集約する痕跡の上限 [件/step](1日あたりの総量を保つ安全弁)"),
    # Wave 4 III-3(路上の生業と条例)。クールダウンは実時間で同じ長さを保ち、L1 の
    # 安全弁は「件/step」なので 1 日あたりの総量が保たれるようにする(traces と同型)。
    ("street_life.cooldown_steps", STEPS,
     "警告/退去のあと客引き・演奏を止める長さ [step](いたちごっこの周期)"),
    ("street_life.max_events_per_step", RATE,
     "1 step に出す路上イベントの上限 [件/step](1日あたりの総量を保つ安全弁)"),
    # Wave 4 III-4(都市運営)。L1 の安全弁は「件/step」= 1 日あたりの総量を保つ(traces と同型)。
    # 治療・現着の長さは**実時間で同じ長さ**を保つ(逆比例)。
    # ★これ以外の city_ops のキーは全部 Δt 非依存(分 of day の時刻帯・件数・距離・
    #   曜日・m/分の速度)なので TABLE に載せない = 棚卸し正規表現にも掛からない綴りである。
    ("city_ops.max_events_per_step", RATE,
     "1 step に出す都市運営イベントの上限 [件/step](1日あたりの総量を保つ安全弁)"),
    ("city_ops.ems.treatment_steps", STEPS,
     "倒れた個体がその場に留まる長さ [step]。実時間で同じ長さを保つ"),
    ("city_ops.ems.on_scene_steps", STEPS,
     "救急隊が現場に留まる長さ [step]。実時間で同じ長さを保つ"),
    # H5(事件レイヤーの環境側 3 族: 火災 / 交通 / 群集)。L1 の安全弁は「件/step」=
    # 1 日あたりの総量を保つ(traces と同型)。燃焼と現場滞在の長さは**実時間で同じ長さ**。
    # ★曝露ハザードは「曝露 1 単位 × 1 step あたりの確率」なので **RATE**: Δt を細かく
    #   すると同じ瞬時曝露が step 数ぶん繰り返されるため、線形に割らないと件数が Δt に
    #   比例して増える(1 日あたりの期待件数を保つ変換)。
    # ★これ以外の incidents_env のキーは全部 Δt 非依存(件/日のレート・重み・分 of day・
    #   距離 m・気温 ℃・m/分の速度・人/m² の密度閾値・面積比)なので TABLE に載せない。
    ("incidents_env.max_events_per_step", RATE,
     "1 step に出す環境事件イベントの上限 [件/step](1日あたりの総量を保つ安全弁)"),
    ("incidents_env.fire.on_scene_steps", STEPS,
     "消防隊が現場に留まる長さ [step]。実時間で同じ長さを保つ"),
    ("incidents_env.fire.burn_steps.*", STEPS,
     "重度別の燃焼の長さ [step]。実時間で同じ長さを保つ(鎮火までの実時間は Δt に依らない)"),
    ("incidents_env.traffic.hazard_per_exposure", RATE,
     "曝露 1 単位あたりの事故ハザード [件/(曝露·step)]。1 日あたりの期待件数を保つ"),
    # H5(設備 = 摩耗する装置)。L1 の安全弁は「件/step」= 1 日あたりの総量を保つ。
    # ★これ以外の world.facilities のキーは Δt 非依存: 故障率は **件/台/日**、保守周期は
    #   **日**、復旧は**分**、摩耗は**利用 1 回あたり / 日あたり**(module 側が分へ割る)、
    #   速度は m/分 = いずれも実時間の単位である。
    ("world.facilities.max_events_per_step", RATE,
     "1 step に出す設備イベントの上限 [件/step](1日あたりの総量を保つ安全弁)"),
    # H3(遺失物ループ)。時効・寿命・気づくまでの遅れはいずれも**実時間で同じ長さ**を保つ
    # (3 か月時効は遺失物法 7 条の実時間・「携帯は 10 分で気づく」も実時間の主張)。
    # 落下率は **件/人日** なので Δt に依存しない(module 側で steps_per_day で割る)=
    # 分類は INVARIANT。確率・倍率・金額・分 of day の時刻帯も同様に Δt 非依存。
    ("lost_property.statute_steps", STEPS,
     "遺失物法 7 条の 3 か月時効 [step]。実時間で同じ長さを保つ"),
    ("lost_property.abandon_steps", STEPS,
     "誰にも拾われないまま路上の物が消えるまで [step]。実時間で同じ長さを保つ"),
    ("lost_property.notice_delay.*", STEPS,
     "持ち主が紛失に気づくまでの遅れ [step](品目別)。実時間で同じ長さを保つ"),
    ("lost_property.max_events_per_step", RATE,
     "1 step に出す遺失物イベントの上限 [件/step](1日あたりの総量を保つ安全弁)"),
    ("lost_property.base_daily.*", INVARIANT,
     "落下率は **件/人日**(Δt 非依存)。per-step 化は module 側で steps_per_day で割る"),
    # 所有権レイヤー O1+O3。L1 の安全弁だけが「件/step」= 1 日あたりの総量を保つ。
    # ★これ以外の world.assets のキーは Δt 非依存: 持ち家率・法人家主比率は**割合**、
    #   家主 org 数と台帳行数の上限は**個数**で、いずれも実時間の単位を持たない。
    ("world.assets.max_events_per_step", RATE,
     "1 step に出す所有権イベント(相続・権利移転)の上限 [件/step](1日あたりの総量を保つ安全弁)"),
    # H4(対人 = RAT収束)。酩酊の残存と勾留の長さは**実時間で同じ長さ**を保つ。
    # ★これ以外の incidents_interpersonal のキーは Δt 非依存(ペア条件付き確率・重み・
    #   閾値・分 of day の時刻帯)なので TABLE に載せない。
    ("incidents_interpersonal.intox.steps", STEPS,
     "酩酊マーカーの残存 [step]。実時間で同じ長さを保つ"),
    ("incidents_interpersonal.report.detain_steps", STEPS,
     "勾留の長さ [step](既定 0=勾留 seam を踏まない)。実時間で同じ長さを保つ"),
    ("incidents_interpersonal.max_events_per_step", RATE,
     "1 step に出す対人事件イベントの上限 [件/step](1日あたりの総量を保つ安全弁)"),
    ("beliefs.witness_window", STEPS, "目撃可能な窓 [step]"),
    ("beliefs.fact_ttl_steps", STEPS, "fact の鮮度 [step]"),
    ("beliefs.verify_deadline_steps", STEPS, "現場確認の有効期限 [step]"),
    ("commerce.inventory.lead_time_steps", STEPS, "発注→到着のリードタイム [step]"),
    ("services.free_steps_ref", STEPS, "daily_rate→per-step 化の基準自由 step 数"),
    ("services.services.*.stay", STEPS, "サービス滞在 [step]"),
    ("delivery.min_eta_steps", STEPS, "配送 ETA の下限 [step]"),
    ("delivery.max_eta_steps", STEPS, "配送 ETA の上限 [step]"),
    # ---- 環境フィードバック(第84バッチ)。持続長・上限は実時間で保つ ----
    ("env.feedback.transit.max_hold_steps", STEPS,
     "1 回の退出/帰還で待たせる合計の上限 [step]。実時間で同じ長さを保つ(発散対策の安全弁)"),
    ("env.feedback.gate.hold_steps", STEPS, "入場規制の継続 [step]"),
    ("env.feedback.gate.cooldown_steps", STEPS, "規制解除後に再発動しない期間 [step]"),
    ("env.feedback.poi.hold_steps", STEPS, "混雑 POI を行き先候補から外す長さ [step]"),

    # ---- 不変(理由つき)。棚卸し grep のヒットを取りこぼさないための宣言 ----
    ("envpack.media.video", INVARIANT,
     "番組名の文字列リスト(『10分でわかる都市伝説』)。数値ではない"),
    ("world.traffic.max_log", INVARIANT, "可視化用のログ件数上限(世界の量ではない)"),
    ("world.traffic.max_cars_log", INVARIANT, "同上"),
    ("world.traffic.signal_cycle_s", INVARIANT, "信号周期 [秒]。物理時間なので Δt と無関係"),
    ("world.traffic.cars_per_day", INVARIANT, "台/日。per-step 化はコード側が steps_per_day で行う"),
    ("world.traffic.od_cars_per_day", INVARIANT, "同上"),
    ("world.inflow_pulse.enabled", INVARIANT,
     "駅到着のパルス量子化の ON/OFF(真偽値)。スナップ先は時刻表の**分 of day**であって "
     "step ではないので Δt に依存しない(分 → step の換算は既存の clock.min_to_steps / "
     "_steps_until_tod が Δt 込みで行う=A9/A1 の経路をそのまま通る)"),
    ("indoor.meeting.window_min", INVARIANT, "分 of day の時刻帯。step ではない"),
    # ---- 在宅覚醒 HOME_AWAKE(β9)の残り = すべて Δt 非依存(理由つき)----
    ("daily.home_awake.enabled", INVARIANT, "機構トグル(量ではない)"),
    ("daily.home_awake.lead_min", INVARIANT,
     "帰宅トリガの前倒し[分]。実時間の長さなので Δt を変えても同じ分数を意味する"),
    ("daily.home_awake.max_awake_min", INVARIANT,
     "帰宅後に起きていられる上限[分]。実時間の長さ(コード側は step 差×step_minutes で"
     "分へ直して比較する)"),
    ("daily.home_awake.early_start_min", INVARIANT,
     "「翌日早出」の判定境界[分 of day]= 時刻。step ではない"),
    ("daily.home_awake.family_talk_boost", INVARIANT,
     "在宅活動の重み表にかける無次元倍率(1 回の抽選にかかる比であって毎 step の量ではない)"),
    ("daily.home_awake.engaged_acts", INVARIANT, "没入とみなす活動ラベルの集合(語彙)"),
    ("daily.home_awake.hazard.b1", INVARIANT,
     "概日項の傾き[対数オッズ/**時間**]。step ではなく実時間あたりの勾配なので Δt 非依存"),
    ("daily.home_awake.hazard.b2", INVARIANT,
     "疲労項の対数オッズ寄与 = ハザード比の対数。切片(p0)が Δt 変換を吸収するので"
     "比のほうは Δt 非依存(小さい p では logit 差 ≒ log ハザード比)"),
    ("daily.home_awake.hazard.b3", INVARIANT, "翌日早出項。b2 と同じ理由で Δt 非依存"),
    ("daily.home_awake.hazard.b4", INVARIANT, "在宅活動への没入項。b2 と同じ理由で Δt 非依存"),
    # Wave 4 III-1(夜間開放)。**どれも step 量ではない**ので不変で正しい:
    #   hours は「分 of day の時刻帯」(indoor.meeting.window_min と同族)、max_stay_min は
    #   分(実時間。step 換算は呼び出し側の clock.min_to_steps が Δt 込みで行う = A1 の経路)、
    #   max_dist_m は距離。
    ("world.night_economy.hours.*", INVARIANT,
     "営業時間 [開店時, 閉店時](24h 表記の**時**)。分 of day の時刻帯なので step ではない"),
    ("world.night_economy.refuge.max_stay_min", INVARIANT,
     "避難先の滞在上限 [分]。実時間の量で、step 換算は clock.min_to_steps が行う"),
    ("world.night_economy.refuge.max_dist_m", INVARIANT,
     "避難先を探す半径 [m]。距離であって時間ではない"),
    ("indoor.sfm.dt", INVARIANT, "屋内 SFM の物理積分刻み [秒]。社会レイヤー Δt と非同期(設計 §3.1)"),
    ("freedom.sat_step", INVARIANT, "自由行動 1 回あたりの充足量(出来事単位。step ではない)"),
    ("worldview.ctrl_step", INVARIANT, "介入 1 回あたりの更新量(出来事単位)"),
    ("memory.relations_max", INVARIANT, "関係台帳の件数上限"),
    ("observer.flush_every_steps", INVARIANT,
     "L1 part 書き出しの I/O 頻度。世界の因果に触れないので実験者の指定どおりにする"),
    ("observer.checkpoint_every", INVARIANT,
     "完全状態を保存する I/O 頻度(第94バッチ OBS-U2 で宣言を補完。棚卸し正規表現に"
     "掛からない綴りだったため未宣言のままだった)。flush_every_steps と同じ族=世界の"
     "因果に触れない運用設定で、実験者が『何 step ごとに落ちても良いか』を指定する量。"
     "★STEPS にすると Δt を変えるたび resume の刻みが黙って動く(既存の Δt=5 resume "
     "テストの前提も崩れる)ので、明示的に不変とする"),
    ("observer.state_hash.interval", INVARIANT, "状態ハッシュの採取頻度。観測装置の設定"),
    ("cognition.channels.every_steps", INVARIANT,
     "観測チャンネル o_c(t) の採取間隔(第80バッチ)。世界の因果に一切触れない観測装置の設定で、"
     "σ_c を測る母集団の粒度は実験者が指定する量。state_hash.interval と同じ扱い"
     "(snapshot_every のように実時間粒度を保つ量ではない: Δt を変えたら『何 step ごとに"
     "採るか』も実験者が測りたい粒度に合わせて指定し直す)"),
    # ---- 閾値発火(第81バッチ)。★周期系の扱いに注意: **単位が分であって step ではない** ----
    #  P0 の要点は「認知イベントは世界 tick から独立」。基本周期は較正テーブル(分)から来て
    #  実時間の分でスケジュールされるので、Δt を変えても**思考の間隔は実時間で不変**でなければ
    #  ならない(そうでないと「Δt を細かくしても総発火数は変わらない」という驚き駆動の主利点=
    #  設計 §5.1 が崩れる)。したがって周期系は STEPS ではなく INVARIANT。
    ("cognition.fire.period_override_min", INVARIANT,
     "基本周期の上書き値[分]。step ではなく実時間の分なので Δt に依らず同じ間隔になる"
     "(=これが『認知時間を世界 tick から分離する』ことの実装上の意味)"),
    ("cognition.fire.period_scale", INVARIANT, "基本周期[分]への無次元倍率"),
    ("cognition.fire.period_cv_scale", INVARIANT, "変動係数への無次元倍率(0=ばらつきなし)"),
    ("cognition.fire.sleep_period_mult", INVARIANT, "睡眠中の周期の無次元倍率"),
    ("cognition.fire.theta_scale", INVARIANT,
     "発火閾値 θ の無次元倍率。θ の単位は σ の本数(時間量ではない)"),
    ("cognition.fire.max_contrib", INVARIANT,
     "L1 に残す S 寄与内訳の件数。観測装置の設定で世界の因果に触れない"),
    # ---- 監視仕様 watch(第82バッチ)。すべて**σ の本数・件数・文字数**= 時間量ではない ----
    ("cognition.watch.clamp_sigmas", INVARIANT,
     "期待値 ô のクランプ幅。単位は σ の本数(時間量ではない)"),
    ("cognition.watch.max_triggers", INVARIANT, "同時に持てる名前付きトリガの本数"),
    ("cognition.watch.trigger_weight", INVARIANT,
     "トリガ 1 本の重み w_ij。単位は σ の本数(時間量ではない)"),
    ("cognition.watch.name_max", INVARIANT, "トリガ名の最大文字数"),
    ("cognition.watch.belief_revision", INVARIANT,
     "model-revision 1 回あたり確信度に掛ける係数(step ではなくイベント単位)"),
    ("cognition.watch.belief_max_facts", INVARIANT, "1 回の見直しで触る信念の件数上限"),
    # ---- g/θ 更新則(第82バッチ)。★時間量は 2 つだけ(半減期と窓)で、どちらも**分**で
    #      持つので Δt 非依存。更新そのものは「認知イベント 1 回につき 1 度」「日境界 1 回」
    #      なので step 数に依存しない(= 設計 §5.1 の『Δt を細かくしても思考の総量は
    #      変わらない』が g/θ にもそのまま効く)。
    ("cognition.g_update.eta_scale", INVARIANT,
     "可塑性 η の無次元倍率。更新は認知イベント 1 回につき 1 度=step 単位ではない"),
    ("cognition.g_update.lam_scale", INVARIANT, "引き戻し λ の無次元倍率(同上)"),
    ("cognition.g_update.rho", INVARIANT, "慣れの重み(誤差 1σ あたり。イベント単位)"),
    ("cognition.g_update.ebar_halflife_min", INVARIANT,
     "ē の半減期[分]。**分で持つ**ので Δt を変えても同じ実時間の平均になる"
     "(実装は α = 1 − 2^(−Δt/T½) で毎 tick の係数へ変換する)"),
    ("cognition.g_update.r_window_min", INVARIANT,
     "発火から結果を測るまでの窓[分]。step ではなく実時間なので Δt に依らない"),
    ("cognition.g_update.r_gain", INVARIANT, "結果スカラ差分 → g への無次元倍率"),
    ("cognition.g_update.max_pending", INVARIANT,
     "同時に開ける credit 窓の本数。件数であって時間量ではない"),
    ("cognition.g_update.g_min", INVARIANT, "感度 g の下限(無次元)"),
    ("cognition.g_update.g_max", INVARIANT, "感度 g の上限(無次元)"),
    ("cognition.g_update.theta_mu", INVARIANT,
     "恒常性の利得 μ。**日境界 1 回**しか効かないので Δt に依らない(設計 §2.6)"),
    ("cognition.g_update.theta_target_per_day", INVARIANT,
     "目標発火率 f*[件/日/人]。per-day レートなので Δt 非依存"),
    ("cognition.g_update.fbar_weight", INVARIANT, "f̄ の日次 EMA 重み(1 日 1 回の更新)"),
    ("cognition.g_update.theta_min_mult", INVARIANT, "θ 個体倍率の下限(無次元)"),
    ("cognition.g_update.theta_max_mult", INVARIANT, "θ 個体倍率の上限(無次元)"),
    ("cognition.g_update.log_every_steps", INVARIANT,
     "g/θ 軌跡サイドカーの採取間隔。観測装置の設定で世界の因果に触れない"
     "(cognition.channels.every_steps と同じ扱い)"),
    # ---- 環境フィードバック(第84バッチ)の不変量。人数・物理分は Δt に依存しない ----
    ("env.feedback.log_every_steps", INVARIANT,
     "環境イベントの記録間引き。観測装置の設定(cognition.channels.every_steps と同じ扱い)"),
    ("env.feedback.transit.platform_threshold", INVARIANT, "ホーム密度の閾値[人](頭数)"),
    ("env.feedback.transit.delay_cap_min", INVARIANT,
     "遅延の絶対上限[分]。**世界の物理時間**(15 分の遅れは Δt に関わらず 15 分)なので不変"),
    ("env.feedback.transit.flag_min", INVARIANT, "観測チャンネルを 1 にする遅延[分](物理時間)"),
    ("env.feedback.gate.capacity_per_min", INVARIANT,
     "改札の処理能力[人/分]。**毎分レート**なので Δt 非依存(コード側が Δt を掛ける)"),
    # ---- day_plan v1(第86バッチ)。計画の時間量は全て **世界の物理時間[分]** で持つ ----
    # (step で持つと Δt を変えたとき「9 時から 1 時間」の意味が変わってしまう)。
    ("planning.day_plan.walk_m_per_min", INVARIANT,
     "移動時間見積りの徒歩速度[m/分]。**毎分レート**なので Δt 非依存"
     "(world.modes.speeds.walk は m/step なので RATE 側。こちらは分あたりの物理速度)"),
    ("planning.day_plan.round_min", INVARIANT,
     "計画時刻の丸め幅[分]。人が予定を書くときの粒度であって step ではない"),
    ("planning.day_plan.min_dur_min", INVARIANT, "ブロックの最小継続[分](物理時間)"),
    ("planning.day_plan.max_dur_min", INVARIANT, "ブロックの最大継続[分](物理時間)"),
    ("planning.day_plan.grace_min", INVARIANT,
     "開始予定からの猶予[分]。『20 分遅れたら諦める』は Δt に依らず 20 分"),
    ("planning.day_plan.transfer_min", INVARIANT, "移動/準備の最小時間[分](物理時間)"),
    ("planning.day_plan.max_slide_min", INVARIANT, "累積ずらしの上限[分](物理時間)"),
    ("planning.day_plan.day_end_min", INVARIANT, "計画が収まるべき終端。分 of day の時刻"),
    # DPH-C 日跨ぎブロック。wrap ON のときの終端で、1800 = 翌 06:00 という**時刻**。
    # day_end_min と同じ理由で Δt に依らない(棚卸し grep には掛からないが、姉妹キーが
    # 分類済みなのに片方だけ表に無いと「見落としたのか意図なのか」が後から判らないので置く)。
    ("planning.day_plan.wrap_end_min", INVARIANT,
     "日跨ぎ計画の終端。分 of day を 1440 超へ延ばした時刻(1800 = 翌 06:00)"),
    # ---- 計画駆動の圏外滞在(actor model P4)。時刻・時間量は全て**物理時間[分]**で持つ ----
    # (start_min / start_spread_min / start_grid_min / hours_min は棚卸し正規表現に
    #  掛からない = 分の量として自明。ここに宣言が要るのは step を名に持つ 1 件だけ)。
    ("planning.day_plan.boundary.min_outside_steps", INVARIANT,
     "退出が遅れて帰還予定時刻を過ぎた場合の**最小滞在 tick 数**。物理的な滞在長ではなく"
     "『滞在を空にしない』という構造的な下限(既定 1 = 少なくとも 1 tick は外に居る)"
     "なので Δt を掛けてはいけない — 掛けると Δt=1 で下限だけが 10 倍に伸びる"),
    # ---- engaged モード(第87バッチ)。★時間量は**すべてシミュ内の分**で持つ ----
    # エピソードは「区間」なので step で持つと Δt を変えたとき『30 分の不応期』の意味が
    # 変わってしまう。ターン上限・試行上限は**件数**、比率・倍率は**無次元**なので不変。
    ("cognition.engaged.theta_out_ratio", INVARIANT,
     "θ_out / θ_in の比(無次元)。ヒステリシス幅であって時間量ではない"),
    ("cognition.engaged.turn_cap", INVARIANT,
     "会話のターン上限[回]。**やりとりの回数**であって step 数ではない"
     "(Δt を細かくしても『12 往復で切り上げる』の意味は変わらない)"),
    ("cognition.engaged.replan_cap", INVARIANT, "再計画の試行上限[回](件数)"),
    ("cognition.engaged.refractory_min", INVARIANT,
     "不応期[分]。**実時間の分**なので Δt に依らず 30 分は 30 分"
     "(drive.refractory_steps が step 単位なのと対照的=認知時間の分離の実装上の意味)"),
    ("cognition.engaged.refractory_mult", INVARIANT, "不応期中の θ_in 倍率(無次元)"),
    ("cognition.engaged.min_stay_min", INVARIANT,
     "エピソードの最短滞在[分]。物理時間なので Δt 非依存(Δt を細かくしたときに"
     "初めて効く dithering 対策)"),
    ("cognition.engaged.talk_idle_min", INVARIANT,
     "会話が生きているとみなす無音の上限[分](物理時間)"),
    ("cognition.engaged.familiar_closeness", INVARIANT, "親密度の閾値(無次元)"),
    ("cognition.engaged.familiar_contacts", INVARIANT, "接触回数の閾値(件数)"),
    ("cognition.engaged.reflect_frac", INVARIANT,
     "夜内省をエピソード化する人口割合(無次元。per-day レートですらない)"),
    ("env.feedback.poi.capacity", INVARIANT, "POI ノードの収容人数[人](頭数)"),
    ("env.feedback.poi.max_nodes", INVARIANT, "同時に除外できるノード数の上限(件数)"),
    ("experiment.g_init.flat_value", INVARIANT, "条件 F/N の trait 定数(無次元)"),
    ("experiment.g_init.sigma0", INVARIANT, "条件 N のノイズ相対分散 σ₀(無次元)"),
    ("beliefs.fact_kinds.event_host", INVARIANT, "fact 種の写像定義(数値ではない)"),
    ("drive.fail_decay", INVARIANT, "抽選落ち 1 回あたりの減衰(出来事単位)"),
    ("drive.fire_reset", INVARIANT, "発火 1 回あたりのリセット係数(出来事単位)"),
    ("drive.drift.habit_rate", INVARIANT, "出来事 1 件あたりの馴化(step ではなくイベント単位)"),
    ("drive.drift.sens_rate", INVARIANT, "出来事 1 件あたりの鋭敏化(同上)"),
    ("drive.boredom.novelty_relief", INVARIANT, "新奇到達 1 回あたりの追加減衰(出来事単位)"),
    ("net.like_prob", INVARIANT, "閲覧した投稿 1 件あたりの反応確率(出来事単位)"),
    ("net.reshare_prob", INVARIANT, "同上"),
    ("lodging.prob", INVARIANT, "夜の退出前 1 回の判定(1日1回)"),
    ("indoor.meeting.prob", INVARIANT, "会合の日次発生確率(1日1回判定)"),
    ("transit_ride.taxi.prob", INVARIANT, "移動 1 回あたりのタクシー選択確率(トリップ単位)"),
    ("tools.permit_deny_prob", INVARIANT, "申請 1 件あたりの却下確率"),
    ("routine.stochastic.novelty_prob", INVARIANT, "1 日 1 回の motif 抽選"),
    ("routine.stochastic.detour_prob.*", INVARIANT, "移動 1 回あたりの寄り道確率(per-move)"),
    ("conversation.c2.daily_cap", INVARIANT, "1 日あたりの上限件数"),
    ("media.max_sessions_per_day", INVARIANT, "1 日あたりの上限回数"),
    ("delivery.max_per_day", INVARIANT, "1 日あたりの上限件数"),
    ("joint.daily_rate", INVARIANT, "1人1日あたりの発生確率(日次判定)"),
    ("chance.daily_rate", INVARIANT, "1人1日あたりの発生確率(日次判定)"),
    ("services.services.*.daily_rate", INVARIANT,
     "1日あたりの利用率。per-step 化は free_steps_ref(STEPS 側)が担うので二重変換しない"),
    ("services.self_dev.decay", INVARIANT, "skill/fitness の日次減衰"),
    # 天候(第7バッチ / 第80バッチ W2)は **日次確定**(1 日 1 回 sim に載る)。
    # rain_grievance は「その日1回の加算量」、mode は文字列、生成器のパラメータは
    # 日別値の較正なので、いずれも Δt を変えても意味が変わらない。
    ("weather.rain_grievance", INVARIANT, "悪天候の日に 1 日 1 回だけ加える不快感の量(日次)"),
    ("weather.mode", INVARIANT, "天候の決め方(文字列)。生成も表引きも日別値=Δt 非依存"),
    ("weather.extra_prompt_fields", INVARIANT, "プロンプト1行の内容(真偽値)。時間量ではない"),
    ("relations.decay_per_day", INVARIANT, "日次の風化"),
    ("relations.rep_decay_per_day", INVARIANT, "日次の風化"),
    ("gossip.decay_prob", INVARIANT, "悪評を 1 日で忘れる確率(日次)"),
    ("gossip.seed_prob", INVARIANT, "(agent,day) 単位の日次判定"),
    ("career.layoff_prob", INVARIANT, "日次確率"),
    ("career.switch_prob", INVARIANT, "日次確率"),
    ("career.rehire_prob", INVARIANT, "日次確率"),
    ("health.onset_prob", INVARIANT, "日次発症確率"),
    ("health.medical_prob", INVARIANT, "発症 1 回あたりの受診確率"),
    # 重症度の状態機械(H1)。転帰の待ち時間だけが step 単位で、残りは日次レート・割合・
    # 気温閾値・年齢帯係数 = Δt 非依存(実時刻を保つため転帰の 2 本だけ逆比例させる)。
    ("health.severity.arrest_outcome_steps", STEPS, "心停止の転帰が確定するまでの長さ"),
    ("health.severity.severe_outcome_steps", STEPS, "致死性の重症の転帰が確定するまでの長さ"),
    # H2 医療(搬送・入院)。**実時間で同じ長さを保つ 2 本**だけが step 単位で、残りは
    # 日数・割合・金額・半径 m = Δt 非依存(下の医療ワイルドカードで INVARIANT へ畳む)。
    ("medical.transport.steps", STEPS,
     "現場 → 病院の搬送にかかる長さ [step]。実時間で同じ長さを保つ"),
    ("medical.admit.mild_steps", STEPS,
     "軽症の在院の長さ [step](数時間で帰宅)。実時間で同じ長さを保つ"),
    ("medical.max_events_per_step", RATE,
     "1 step に出す医療イベントの上限 [件/step](1 日あたりの総量を保つ安全弁)"),
    ("medical.*", INVARIANT,
     "トグル = Δt 非依存(上限件数と搬送・在院の step 長は上で個別に分類済み)"),
    ("medical.*.*", INVARIANT,
     "在院日数・自己負担割合・金額・半径 m・1 日あたりの上限件数・POI 語彙 = Δt 非依存"),
    ("health.severity.*", INVARIANT,
     "日次ハザード・重症度分布の割合・WBGT 閾値/勾配・年齢帯係数・金額 = Δt 非依存"
     "(発症は『その日その個体が発症するか』の判定で、毎 step の Bernoulli ではない)"),
    ("housing.relocation.job_prob", INVARIANT, "日次転居確率"),
    ("housing.relocation.rent_prob", INVARIANT, "日次転居確率"),
    ("disaster.onset_prob", INVARIANT, "日次発生確率"),
    ("disaster.delay_prob", INVARIANT, "日次確率"),
    ("disaster.suspend_prob", INVARIANT, "遅延日 1 件あたりの確率"),
    ("disaster.outage_prob", INVARIANT, "日次確率"),
    ("info_env.misinfo.rate", INVARIANT, "新規投稿 1 件あたりの割合"),
    ("info_env.misinfo.correction_prob", INVARIANT, "誤情報 1 件あたりの訂正確率"),
    ("freedom.sat_decay", INVARIANT, "日次の中立への回帰率"),
)

# 検索用インデックス(ワイルドカードは展開せずパターンのまま持つ)
_BY_KEY: dict[str, tuple[str, str]] = {k: (c, w) for k, c, w in TABLE}


def classify(key: str) -> tuple[str, str] | None:
    """conf のドットパス → (分類, 理由)。テーブルに無ければ None。"""
    hit = _BY_KEY.get(key)
    if hit is not None:
        return hit
    parts = key.split(".")
    for pat, cls, why in TABLE:
        if "*" not in pat:
            continue
        pp = pat.split(".")
        if len(pp) != len(parts):
            continue
        if all(a == "*" or a == b for a, b in zip(pp, parts)):
            return (cls, why)
    return None


def covers(key: str) -> bool:
    """そのキーがテーブルで説明済みか(変換済み or 不変と判断済み)。"""
    return classify(key) is not None


# --------------------------------------------------------------------------- #
# config への一括適用(唯一の作用点)
# --------------------------------------------------------------------------- #
_APPLY = {RATE: scale_rate, PROB: scale_prob, KEEP: scale_keep}


def dt_of(cfg) -> int:
    """config から Δt [分] を取り出す(未搭載の旧 config は正準 10)。"""
    try:
        run = cfg.get("run", None) if hasattr(cfg, "get") else None
    except Exception:                              # noqa: BLE001
        run = None
    if run is None:
        return CANON_DT_MIN
    try:
        v = run.get("dt_min", None)
    except Exception:                              # noqa: BLE001
        return CANON_DT_MIN
    if v is None:
        return CANON_DT_MIN
    dt = int(v)
    if dt <= 0:
        raise ValueError(f"run.dt_min は正の整数でなければならない: {dt}")
    if MINUTES_PER_DAY % dt != 0:
        raise ValueError(
            f"run.dt_min={dt} は 1440 の約数でなければならない"
            "(日境界が step 境界に落ちないと日次機構が壊れる)")
    return dt


def _iter_targets(cfg, pattern: str):
    """ワイルドカード込みのパターン → **実在する**具体キー(ドットパス)の列。

    `*` は 1 セグメントの任意キー。存在しない枝は静かに落とす(旧 config 互換)。
    """
    parts = pattern.split(".")

    def children(node):
        try:
            return list(node.keys())
        except Exception:                          # noqa: BLE001 (リーフ/非 mapping)
            return []

    def get(node, key):
        try:
            return node[key] if key in node else None
        except Exception:                          # noqa: BLE001
            return None

    def walk(node, idx, acc):
        if idx == len(parts):
            yield ".".join(acc)
            return
        if node is None:
            return
        seg = parts[idx]
        if seg == "*":
            for k in children(node):
                yield from walk(get(node, k), idx + 1, acc + [str(k)])
            return
        if seg not in children(node):
            return
        yield from walk(get(node, seg), idx + 1, acc + [seg])

    yield from walk(cfg, 0, [])


def apply_dt(cfg):
    """`run.dt_min` に従って分類テーブルの各定数を変換した config を返す。

    **Δt=10(既定)なら config を 1 バイトも触らない**(同一オブジェクトをそのまま返す)。
    変換が恒等になる場合も浮動小数の同値性を信用しないための構造上の分岐であり、
    golden L1 バイト一致の担保そのもの。
    """
    from omegaconf import OmegaConf
    dt = dt_of(cfg)
    if dt == CANON_DT_MIN:
        return cfg                                  # ★唯一の既定パス: 完全 no-op
    for pattern, cls, _why in TABLE:
        fn = _APPLY.get(cls)
        if fn is None and cls != STEPS:
            continue                                # INVARIANT は触らない
        for key in _iter_targets(cfg, pattern):
            val = OmegaConf.select(cfg, key)
            if val is None:
                continue
            if isinstance(val, bool):
                continue
            if isinstance(val, (list, tuple)) or OmegaConf.is_list(val):
                seq = list(val)
                if not all(isinstance(x, (int, float)) and not isinstance(x, bool)
                           for x in seq):
                    continue
                new = ([scale_steps(x, dt) for x in seq] if cls == STEPS
                       else [_keep_type(x, fn(float(x), dt)) for x in seq])
                OmegaConf.update(cfg, key, new)
                continue
            if not isinstance(val, (int, float)):
                continue
            new_v = (scale_steps(val, dt) if cls == STEPS
                     else _keep_type(val, fn(float(val), dt)))
            OmegaConf.update(cfg, key, new_v)
    return cfg


def _keep_type(old, new: float):
    """元が int のキー(件数っぽい予算・速度)は int のまま返す(型が変わると下流の
    int() 前提や config スナップショットの見た目が壊れるため)。0 は 0 のまま。"""
    if isinstance(old, bool) or not isinstance(old, int):
        return new
    if old == 0:
        return 0
    r = int(round(new))
    return max(1, r) if old > 0 else min(-1, r)
