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

    # ---- step 単位の持続長・窓・TTL・クールダウン(逆比例)----
    ("world.outside_steps", STEPS, "範囲外滞在の長さ [step]"),
    ("indoor.markov.dwell_steps", STEPS, "平均滞在 step(遷移確率 = 1/dwell)"),
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
    ("media.min_steps", STEPS, "視聴セッション長の下限 [step]"),
    ("media.max_steps", STEPS, "視聴セッション長の上限 [step]"),
    ("institution_routes.enforcement.detention_steps", STEPS, "拘束の長さ [step]"),
    ("beliefs.witness_window", STEPS, "目撃可能な窓 [step]"),
    ("beliefs.fact_ttl_steps", STEPS, "fact の鮮度 [step]"),
    ("beliefs.verify_deadline_steps", STEPS, "現場確認の有効期限 [step]"),
    ("commerce.inventory.lead_time_steps", STEPS, "発注→到着のリードタイム [step]"),
    ("services.free_steps_ref", STEPS, "daily_rate→per-step 化の基準自由 step 数"),
    ("services.services.*.stay", STEPS, "サービス滞在 [step]"),
    ("delivery.min_eta_steps", STEPS, "配送 ETA の下限 [step]"),
    ("delivery.max_eta_steps", STEPS, "配送 ETA の上限 [step]"),

    # ---- 不変(理由つき)。棚卸し grep のヒットを取りこぼさないための宣言 ----
    ("envpack.media.video", INVARIANT,
     "番組名の文字列リスト(『10分でわかる都市伝説』)。数値ではない"),
    ("world.traffic.max_log", INVARIANT, "可視化用のログ件数上限(世界の量ではない)"),
    ("world.traffic.max_cars_log", INVARIANT, "同上"),
    ("world.traffic.signal_cycle_s", INVARIANT, "信号周期 [秒]。物理時間なので Δt と無関係"),
    ("world.traffic.cars_per_day", INVARIANT, "台/日。per-step 化はコード側が steps_per_day で行う"),
    ("world.traffic.od_cars_per_day", INVARIANT, "同上"),
    ("indoor.meeting.window_min", INVARIANT, "分 of day の時刻帯。step ではない"),
    ("indoor.sfm.dt", INVARIANT, "屋内 SFM の物理積分刻み [秒]。社会レイヤー Δt と非同期(設計 §3.1)"),
    ("freedom.sat_step", INVARIANT, "自由行動 1 回あたりの充足量(出来事単位。step ではない)"),
    ("worldview.ctrl_step", INVARIANT, "介入 1 回あたりの更新量(出来事単位)"),
    ("memory.relations_max", INVARIANT, "関係台帳の件数上限"),
    ("observer.flush_every_steps", INVARIANT,
     "L1 part 書き出しの I/O 頻度。世界の因果に触れないので実験者の指定どおりにする"),
    ("observer.state_hash.interval", INVARIANT, "状態ハッシュの採取頻度。観測装置の設定"),
    ("cognition.channels.every_steps", INVARIANT,
     "観測チャンネル o_c(t) の採取間隔(第80バッチ)。世界の因果に一切触れない観測装置の設定で、"
     "σ_c を測る母集団の粒度は実験者が指定する量。state_hash.interval と同じ扱い"
     "(snapshot_every のように実時間粒度を保つ量ではない: Δt を変えたら『何 step ごとに"
     "採るか』も実験者が測りたい粒度に合わせて指定し直す)"),
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
