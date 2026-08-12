"""日次プレゼンス純関数(W2 P3 / ローテーション機構)。

正典: docs/plans/w2-execution-plan.md §4 P3 / docs/plans/persona-pool.md §5「presence 純関数」・§9。

100万ペルソナプール(P5)から「今日この街にいる人(present)」を**決定論**で選ぶ。
選択は `hub.stream("presence", persona_id, day)` ベースの層別純関数であり、

  - **k 非依存**: 経験→内部状態のゲイン k を一切読まない。
  - **trait 非依存**: 個体の trait/beliefs を読まない。入力は暦(day/weekday)+ ペルソナ固有の
    presence 属性(層・visit_rate・cadence・duty)+ 専用 stream のみ。
  - **resume 不変**: day のみが可変入力。他者の実行順・生存状態に依存しない(RngHub がステートレス)。

→ k 掃引の全条件で「同じ人が同じ日に present」(共通乱数)= presence が k* を交絡しない(R1)。

日内の出入り(朝流入→夕退出)は既存の visitor 入退場機構が担う。ここは**日次選択のみ**を行う。
在場実測曲線(data/jinryu/shibuya_concurrent_144step_curve.csv)は v1 では使わない(将来拡張点)。

層別規則(P5 の presence キー):
  - resident       (L1 住民・L5 議員): 毎日 present。
  - duty           (L5 役割=駅員/運転士/警察/配信者): 役割で常時(当番。duty_pattern.days)。
  - workday_shift  (L2 域内従業者): 平日 + シフト暦。
  - cadence        (L3 定期来街=学生/常連): 通学日・週次常連の暦。
  - stochastic     (L4 非定期来街): visit_rate で抽選(re-visit 性 = 同一人物が周期で再訪)。

日次の総在場数は `present_cap` を上限に**層優先**で充足する:
  resident/duty/workday > cadence > stochastic。上限で溢れた層のみ、当日固有の決定論ランキングで
  部分集合を採る(→ 日ごとに顔ぶれが入れ替わる=ローテーション)。上位層は溢れない限り毎日不変。

層別クォータ(DP-U3 案A・conf `pool.tier_quota.enabled`・**既定 OFF = 上の層優先と完全一致**)
------------------------------------------------------------------------------------------
正典: docs/plans/proposal-dp-u3-observe-250k.md §1.2「案 A の中身」。

層優先は「優先順に埋めて途中で break」なので、上位層が cap を食い切ると**下位層が構造的に
全滅**する。第91 の実測(名簿 100万・cap 25万・平日)では resident 30,034 + duty 986 で残
218,980 を workday_shift(253,702)が埋め切り、**cadence / stochastic が 1 人も入らない**
= 25万人の街に来街者がゼロ、という破綻が起きた。

ON にすると cap を層ごとのクォータに分配する:

    quota[t] = round(present_cap × eligible[t] / Σ eligible)     # 当日資格者の比率を保存
    present  = ∪_t rank_t(eligible[t])[:quota[t]]                # 溢れた層だけ当日ランキングで切る

- **乱数追加ゼロ**: 端数処理は整数演算の最大剰余法(同値タイブレーク = 層の固定順)。
  層内の並び順・選抜規則は層優先のときと同一(既存 `presence_rank` stream)。
- k 非依存・trait 非依存・resume 不変は不変(入力は当日の資格者数と層だけ)。
- ★ON では resident も比率で切られる(cap < 資格者総数のとき)。「上位層は毎日不変」という
  層優先の性質は ON では成り立たない。これは構成比を保存することの代償であり設計どおり。
- 週末の総在場が現実の 28% しかない問題(提案書 §1.1)は**名簿側の較正問題**であって
  クォータでは直らない(クォータは比率を保存するだけで人を増やさない)。

在場の内生化(PRES-A1 / A2。conf `pool.presence`・**既定 = 現行と 1 バイト一致**)
--------------------------------------------------------------------------------
正典: docs/plans/presence-endogenization-plan.md §4(ユーザー承認 2026-08-12)/
      docs/research/presence-endogenization.md §4.2。

ユーザー原理:「範囲内にいる人の数もこちらが指定するものではない…世界のアルゴリズムに
よって結果が左右されたり、エージェントの量が決まったりすることはなく」。この物差しで測ると
現行機構に残る違反は 2 点だけだった(リサーチ §2.4):

  (a) `present_cap` の当日ランキング切り … 落ちる理由が**その人の行動に帰属しない**
  (b) stochastic 層の `visit_rate × 当日乱数` … レートは個人属性だが「今日行くか」は世界のコイン

本 module はその 2 点をそれぞれ **新モード**として実装する(既定 OFF = 現行経路のまま):

`pool.presence.habit.enabled`(A1 = (b) の解消。既定 false)
  非 revisit の Bernoulli を「個体固定の**習慣カレンダー**」へ置き換える。式は
  Bresenham(Beatty 列)そのもの:

      present(i, day) ⟺ floor(θ_i + C_i(day+1)) > floor(θ_i + C_i(day))

  θ_i = 個体固定の位相 ∈ [0,1)(stream "presence_habit"、**day を渡さない**)。
  C_i(d) = その人の来街レートの累積(profile を掛けない素の形なら C_i(d) = d·r_i)。

  ★**分布保存の証明**(tests/test_presence_habit.py が機械固定):
    θ_i ~ U[0,1) なので frac(θ_i + C_i(day)) も U[0,1) 上一様 → 区間
    (C_i(day), C_i(day+1)] に整数が入る確率は区間長そのもの = r_i。つまり
    **1 日あたりの在場確率が Bernoulli(visit_rate) と厳密に一致する**(丸めない・
    period=round(1/r) のような丸め誤差が入らない)。しかも D 日の来街回数は
    floor(θ+D·r) − floor(θ) ∈ {floor(D·r), ceil(D·r)} = 個人の長期レートも厳密保存。
  ★乱数は**減る**: 当日 draw(stream "presence")が消え、個体固定 stream の 2 draw
    (位相 θ・天候耐性 tol)になる。k 非依存・trait 非依存・resume 不変は不変
    (入力は day / weekday / 個体属性のみ)。

`pool.presence.habit.weekday`(A1② 曜日。habit ON のときだけ効く)
  目的(`visit_purpose`)別の曜日プロファイルを累積 C_i に掛ける。プロファイルは
  **週平均 1.0 に機械正規化**してあるので、7 日ぶんの期待在場は Σ r_i·7 のまま
  = 週次の総来街量(既存の visit_rate 較正)を 1 人も動かさず、**週内の配分だけ**が動く。

`pool.presence.habit.weather`(A1② 天候。habit ON かつ weather.mode=generated のみ)
  雨/雪の日は「その目的で来る予定だった人のうち、雨に弱い人」が出控える。判定は
  個体固定の耐性 tol_i ∈ [0,1) と目的別弾性 e の比較(tol_i < e で出控え)= 母集団の
  減少量はちょうど e·N。★synthetic / table / weather OFF では **不活性**(rain=None)。
  H1 の熱中症(WBGT)と同型の「generated 限定・それ以外は正直に何もしない」規約。

`pool.presence.mode: quota | emergent`(A2 = (a) の解消。既定 "quota")
  emergent では `present_cap` を**一切見ない**(quota_by_ratio 経路も通らない)。
  当日の資格者が**そのまま**在場者になる = 総在場は個人の予定の合算として創発する。
  縮退線は cap ではなく `--fraction`(= プール規模 = 世界の人口という事実)側へ移す
  (MATSim 系の標本設計と同じ作法。リサーチ §3.4)。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# presence キー → 優先度 tier(小さいほど先に充足=cap で落とされにくい)。
# resident/duty/workday は「毎日概ね同じ顔」(回転が薄い)。cadence/stochastic が回転の主層。
_TIER = {"resident": 0, "duty": 1, "workday_shift": 2, "cadence": 3, "stochastic": 4}
_N_TIERS = 5

_WEEKDAYS = frozenset({0, 1, 2, 3, 4})     # Mon..Fri(day % 7 で day0=Monday とする)


@dataclass
class PresenceRec:
    """presence 純関数が読む**スリムな**ペルソナ記述子(full record は読まない)。

    PoolStore が P5 シャードから抽出する(pool.py::_slim)。ここに beliefs/traits/k は入れない。
    """
    pid: str                    # ペルソナ id(例 "L4_00012345")= agent.id の安定源
    key: str                    # presence キー(resident/duty/workday_shift/cadence/stochastic)
    work_days: str = ""         # workday_shift: "mon-fri" | "all" | ...
    cadence: str = ""           # cadence: "school_day" | "weekly_N"
    visit_rate: float = 0.0     # stochastic: 来訪確率/日(0.003〜0.06)
    revisit: bool = False       # stochastic: 再訪する個体か(周期で同一人物が戻る)
    duty_days: str = "all"      # duty: duty_pattern.days("all" | ...)
    visit_purpose: str = ""     # stochastic: 来街目的(A1② の曜日/天候共変量のキー。既定は無指定)


# ---- 曜日仕様の語彙(A1③ mon-sat バグ修正)------------------------------------------
# ★第109 レーン PRES-Pool で直したバグ: 旧実装は `mon-fri` / `school_day` 以外を**全部**
#   平日扱いへ後退させていた(旧 :84 の `return weekday in _WEEKDAYS`)。ところが台帳
#   (build_orgs)は小売・飲食に **`mon-sat`** を書いており、v8 プールの L2 には
#   `work_days="mon-sat"` が 44,486 人居る(リサーチ §2.3 実測)。この後退規則のせいで
#   **名簿が土曜出勤と宣言している 44,486 人が土曜に失格**していた = 名簿の意図と機構のズレ。
#   これは新機能ではなく**バグ修正**なので、pool ON のランでは既定挙動が変わる(土曜の
#   workday_shift が増える)。pool OFF(golden)は presence を 1 度も通らないので無風。
#: 曜日名 → index(0=Mon..6=Sun)。範囲記法 "a-b" の両端に使う。
_DAY_INDEX = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
#: 範囲記法でない固定語彙(名簿・台帳・conf が実際に書いている語)。
_DAYS_ALIAS: dict[str, frozenset[int]] = {
    "all": frozenset(range(7)),
    "everyday": frozenset(range(7)),
    "daily": frozenset(range(7)),
    "weekday": _WEEKDAYS,          # L5 duty_pattern("ティッシュ配り" 等)が使う語
    "weekdays": _WEEKDAYS,
    "school_day": _WEEKDAYS,       # cadence(通学日)
    "weekend": frozenset({5, 6}),
    "weekends": frozenset({5, 6}),
    "holiday": frozenset({5, 6}),
}


def _parse_days(spec: str) -> frozenset[int] | None:
    """曜日仕様 → 出勤曜日の集合。**解釈できない仕様は None**(呼び出し側が後退する)。

    受理する形(名簿・台帳・conf が実際に書いている形だけ。推測で語彙を増やさない):
      - 固定語 `_DAYS_ALIAS`("all" / "weekday" / "school_day" / "weekend" …)
      - 範囲記法 `mon-fri` / `mon-sat` / `tue-sun` / `sat-sun`(両端含む・折り返し可)
      - カンマ列 `mon,wed,fri`(範囲と混在可: `mon-fri,sun`)
    """
    key = str(spec or "").strip().lower()
    if not key:
        return None
    got: set[int] = set()
    for token in key.split(","):
        token = token.strip()
        if not token:
            continue
        alias = _DAYS_ALIAS.get(token)
        if alias is not None:
            got |= set(alias)
            continue
        if "-" in token:
            lo_s, _, hi_s = token.partition("-")
            lo, hi = _DAY_INDEX.get(lo_s.strip()), _DAY_INDEX.get(hi_s.strip())
            if lo is None or hi is None:
                return None
            # 折り返し(例 "sat-sun" は 5,6 / "sun-tue" は 6,0,1)も素直に展開する。
            got |= {(lo + i) % 7 for i in range((hi - lo) % 7 + 1)}
            continue
        one = _DAY_INDEX.get(token)
        if one is None:
            return None
        got.add(one)
    return frozenset(got) or None


def _days_match(spec: str, weekday: int) -> bool:
    """暦の曜日仕様と当日 weekday の一致(**解釈できない仕様だけ**平日扱いに後退)。

    ★A1③: `mon-sat` 等の範囲記法を名簿の意図どおり解釈する(旧実装はここで失格させていた)。
    後退規則そのものは残す(未知の語を黙って「毎日出勤」にしない方が安全側)。
    """
    if not spec or spec == "all":
        return True
    days = _parse_days(spec)
    if days is None:
        return weekday in _WEEKDAYS          # 解釈できない仕様 = 従来どおり平日扱い
    return int(weekday) in days


def _weekly_days(hub, pid: str, week: int, n: int) -> set[int]:
    """常連(weekly_N): その週に来る n 曜日を決定論で確定(週固有 stream)。"""
    n = max(1, min(int(n), 7))
    g = hub.stream("presence_cadence", pid, int(week))
    return {int(x) for x in g.choice(7, size=n, replace=False)}


def _revisit_present(hub, pid: str, day: int) -> bool:
    """再訪個体の周期プレゼンス。period/phase は**day 非依存**の固定値(同一人物が規則的に戻る)。

    → 「二度目の来街で前回の場所を思い出す観光客」「行きつけの常連」の観測が自然に立ち上がる。
    """
    g = hub.stream("presence_revisit", pid)      # day を渡さない=個体固定の周期
    period = 3 + int(g.integers(0, 8))           # 3..10 日周期
    phase = int(g.integers(0, period))
    return (day - phase) % period == 0


# =========================================================================== #
# A1① ② 習慣カレンダー(conf `pool.presence.habit`。既定 OFF = 1 バイトも通らない)
# =========================================================================== #
#: 目的別の曜日プロファイル(index 0=Mon..6=Sun)。**生の形**をここに書き、
#: `build_presence_cfg` が週平均 1.0 へ機械正規化する(= 週次の総来街量は不変)。
#:
#: ★正直な素性: 係数の**向き**だけが実測アンカー由来で、**水準は設計値**である。
#:   - 娯楽・観光・買物の週末集中 … モバイル空間統計の対象地区分析(平日勤務人口はピーク時
#:     約 2,000 人減で未回復・**休日来訪は 2019 水準まで回復**)= 平日と週末で別の母集団が
#:     動く実証。docs/research/presence-endogenization.md §3.2。
#:   - 飲食の金曜ピーク … 東京都繁華街利用実態調査(本地区は平日≒休日の稀有な街)と
#:     夜間経済(night_economy)の営業時間帯。
#:   - 業務来訪の平日集中・通院の日曜落ち … PT 調査の目的別トリップ(業務・通院等)と
#:     診療所の日曜休診という制度事実。
#: ★水準そのものを実測へ寄せるのは**別のレバー**(リサーチ §5-3「変更は 1 レバーずつ・
#:   jinryu 曲線との照合を毎回挟む」)。ここは「週内の配分」だけを担当する。
_WEEKDAY_PROFILE_RAW: dict[str, tuple[float, ...]] = {
    #                    Mon   Tue   Wed   Thu   Fri   Sat   Sun
    "観光・見物":        (0.80, 0.80, 0.80, 0.80, 1.15, 1.60, 1.50),
    "エンタメ・イベント": (0.80, 0.80, 0.80, 0.85, 1.20, 1.60, 1.45),
    "買い物":            (0.85, 0.85, 0.85, 0.90, 1.05, 1.45, 1.40),
    "飲食":              (0.90, 0.90, 0.90, 0.95, 1.30, 1.40, 0.95),
    "友人と会う":        (0.85, 0.85, 0.85, 0.90, 1.20, 1.45, 1.30),
    "ビジネス来訪":      (1.35, 1.35, 1.35, 1.35, 1.30, 0.15, 0.05),
    "通院・用事":        (1.30, 1.30, 1.30, 1.30, 1.20, 0.70, 0.10),
}
_FLAT_PROFILE = (1.0,) * 7          # 目的が無い / 未知の目的 = 共変量を掛けない(正直側)

#: 雨・雪の日の目的別弾性(その目的で来る予定だった人のうち出控える割合)。
#: 実測レンジ(docs/research/presence-endogenization.md §3.5):
#:   降雨の歩行者数 −5% 〜 −27%(複数研究)/ 降雨 5mm/h で**週末**トリップ −29%。
#: → 屋外性の高い目的ほど上限側、必要用事(通院・業務)は 0(天候で減らない)。
_RAIN_ELASTICITY: dict[str, float] = {
    "観光・見物": 0.27,             # 見物 = 屋外そのもの(実測レンジの上端)
    "エンタメ・イベント": 0.15,
    "友人と会う": 0.10,
    "買い物": 0.10,                 # 雨×小売は「路面店減・モール増」の付け替え = 純減は小さい
    "飲食": 0.08,
    "ビジネス来訪": 0.0,            # 業務トリップは天候で消えない
    "通院・用事": 0.0,              # 必要用事は天候で消えない
}
#: 週末は雨弾性が大きい(実測: 週末 −29% > 平日)。掛けたうえで実測上限で頭打ちにする。
_RAIN_WEEKEND_GAIN = 1.5
_RAIN_MAX = 0.29
#: ★雪(`weather._BAD_CONDS` のもう一方)の弾性は実測が無いので**雨と同値**に置く。
#:   別係数を発明しないための明示(黙って 0 にも 2 倍にもしない)。

PRESENCE_MODES = ("quota", "emergent")


def _normalized(raw) -> tuple[float, ...]:
    """曜日プロファイルを**週平均 1.0**へ正規化(= 週次の期待在場を 1 人も動かさない)。"""
    vals = [max(0.0, float(v)) for v in raw][:7]
    vals += [0.0] * (7 - len(vals))
    total = sum(vals)
    if total <= 0.0:
        return _FLAT_PROFILE
    scale = 7.0 / total
    return tuple(v * scale for v in vals)


def build_presence_cfg(raw: dict | None) -> dict:
    """conf `pool.presence` ブロックの正準化(既定 = 現行経路と完全一致)。

    返り値は `present_for_day` の `habit` / `emergent` 引数へそのまま渡せる形。
    """
    raw = dict(raw or {})
    mode = str(raw.get("mode", "quota")).strip().lower()
    if mode not in PRESENCE_MODES:
        raise ValueError(
            f"pool.presence.mode='{mode}' は未知(有効値: {', '.join(PRESENCE_MODES)})。"
            "既定 quota = 現行の present_cap 充足経路。")
    h = dict(raw.get("habit", {}) or {})
    return {
        "mode": mode,
        "emergent": mode == "emergent",
        "habit": {
            "enabled": bool(h.get("enabled", False)),
            "weekday": bool(h.get("weekday", True)),
            "weather": bool(h.get("weather", True)),
            "profile": {k: _normalized(v) for k, v in _WEEKDAY_PROFILE_RAW.items()},
            "rain": dict(_RAIN_ELASTICITY),
            "rain_gain": _RAIN_WEEKEND_GAIN,
            "rain_max": _RAIN_MAX,
        },
    }


def habit_on(habit: dict | None) -> bool:
    return bool(habit and habit.get("enabled"))


def _cum_rate(rate: float, day: int, weekday: int,
              profile: tuple[float, ...]) -> tuple[float, float]:
    """day までの累積来街レート (day 開始時, day 終了時) を **O(1)** で返す。

    プロファイルは週平均 1.0(= 7 日ぶんの和がちょうど 7.0)なので、丸ごとの週は
    `7·rate` で飛ばせる。端数の日だけ day0 の曜日から順に足す。
    """
    if profile is _FLAT_PROFILE:
        return rate * day, rate * (day + 1)
    w0 = (int(weekday) - int(day)) % 7            # day0 の曜日(呼び出し側の暦から逆算)
    rest = int(day) % 7
    partial = 0.0
    for j in range(rest):
        partial += profile[(w0 + j) % 7]
    lo = rate * (7.0 * (int(day) // 7) + partial)
    return lo, lo + rate * profile[int(weekday) % 7]


def _rain_elasticity(purpose: str, weekday: int, habit: dict) -> float:
    """雨/雪の日にその目的の人が出控える割合(母集団の減少量そのもの)。"""
    base = float(habit["rain"].get(purpose, 0.0))
    if base <= 0.0:
        return 0.0
    if int(weekday) >= 5:                          # 週末は弾性が大きい(実測)
        base *= float(habit["rain_gain"])
    return min(base, float(habit["rain_max"]))


def _habit_present(hub, rec: PresenceRec, day: int, weekday: int,
                   habit: dict, rain: bool | None) -> bool:
    """A1①②: 個体固定の習慣カレンダー(Bresenham)+ 目的×曜日×天候。

    「今日は行く日だから居る」が個人の属性(レート・位相・目的・雨耐性)だけで決まる。
    """
    rate = float(rec.visit_rate)
    if rate <= 0.0:
        return False
    if rate >= 1.0:
        return True
    g = hub.stream("presence_habit", rec.pid)      # ★day を渡さない = 個体固定の習慣
    theta = float(g.random())                      # 位相 θ ∈ [0,1)
    tol = float(g.random())                        # 天候耐性(この人が雨で出控えるか)
    profile = _FLAT_PROFILE
    if habit["weekday"]:
        profile = habit["profile"].get(rec.visit_purpose, _FLAT_PROFILE)
    lo, hi = _cum_rate(rate, int(day), int(weekday), profile)
    if math.floor(theta + hi) <= math.floor(theta + lo):
        return False                               # 今日は「来る日」ではない
    if rain and habit["weather"]:
        if tol < _rain_elasticity(rec.visit_purpose, weekday, habit):
            return False                           # 雨に弱い人が出控えた
    return True


def _eligible(rec: PresenceRec, day: int, weekday: int, hub,
              habit: dict | None = None, rain: bool | None = None) -> bool:
    """当日 present 資格があるか(層別・純関数・k/trait 非参照)。

    habit=None(既定)では stochastic 層が従来の Bernoulli を通る = 1 バイトも変わらない。
    """
    key = rec.key
    if key == "resident":
        return True
    if key == "duty":
        return _days_match(rec.duty_days, weekday)
    if key == "workday_shift":
        return _days_match(rec.work_days, weekday)
    if key == "cadence":
        cad = rec.cadence or "school_day"
        if cad.startswith("weekly_"):
            try:
                n = int(cad.split("_", 1)[1])
            except (ValueError, IndexError):
                n = 2
            return weekday in _weekly_days(hub, rec.pid, day // 7, n)
        return _days_match(cad, weekday)
    if key == "stochastic":
        if rec.revisit and _revisit_present(hub, rec.pid, day):
            return True
        if habit_on(habit):
            return _habit_present(hub, rec, day, weekday, habit, rain)
        return float(hub.stream("presence", rec.pid, int(day)).random()) < rec.visit_rate
    return False


def _ranked(ids: list[str], day: int, hub) -> list[str]:
    """層内の当日ランキング(既存規則。溢れた層の部分集合を採る順序)。"""
    return sorted(
        ids,
        key=lambda pid: (float(hub.stream("presence_rank", pid, int(day)).random()), pid))


def quota_by_ratio(sizes: list[int], cap: int) -> list[int]:
    """層別クォータ(DP-U3 案A)。**整数演算のみ・乱数ゼロ・同入力同出力**。

    quota[t] = round(cap × sizes[t] / Σ sizes) を**最大剰余法**で合計 = min(cap, Σ sizes) に
    正規化する。剰余が同値なら層 index(= 層名の固定順)の小さい方を優先=タイブレークも決定論。
    資格者数がクォータ未満になる層は**全員だけ**入れ、余った枠は残りの層へ同じ規則で再配分する
    (毎周 1 人以上配るので必ず停止する)。cap ≧ Σ sizes なら全層まるごと(= 切らない)。
    """
    n = len(sizes)
    quota = [0] * n
    room = [max(0, int(s)) for s in sizes]          # 各層の未割当の資格者
    remaining = max(0, int(cap))
    while remaining > 0:
        active = [t for t in range(n) if room[t] > 0]
        if not active:
            break                                    # 資格者を配り切った(cap > Σ eligible)
        total = sum(room[t] for t in active)
        if remaining >= total:                       # 全層とも取り分 ≧ 資格者 → 全員入れて終了
            for t in active:
                quota[t] += room[t]
                room[t] = 0
            break
        base = [(remaining * room[t]) // total for t in active]
        order = sorted(range(len(active)),           # 端数の大きい層から 1 ずつ(同値は層順)
                       key=lambda i: (-((remaining * room[active[i]]) % total), active[i]))
        for i in order[:remaining - sum(base)]:
            base[i] += 1
        moved = 0
        for i, t in enumerate(active):
            give = min(base[i], room[t])             # 資格者不足の層は全員だけ(残枠は次周で再配分)
            quota[t] += give
            room[t] -= give
            moved += give
        if moved <= 0:
            break                                    # 前進しない(到達しない)= 無限ループ防止
        remaining -= moved
    return quota


def present_for_day(recs, day: int, present_cap: int, hub, weekday: int,
                    tier_quota: bool = False, *,
                    habit: dict | None = None, emergent: bool = False,
                    rain: bool | None = None) -> list[str]:
    """day の present ペルソナ id を present_cap まで選ぶ(決定論・純関数)。

    引数:
      recs: PresenceRec の反復可能(PoolStore.presence_records())。
      day: 日インデックス(0 起点)。
      present_cap: 日次総在場の上限。
      hub: RngHub(stream 専用。k/step 非参照)。
      weekday: 当日の曜日(0=Mon..6=Sun)。呼び出し側が暦から計算して渡す。
      tier_quota: False(既定)= 従来の層優先 break。True = 層別クォータ(module docstring)。
      habit: A1 習慣カレンダーの設定(`build_presence_cfg()["habit"]`)。
             None / enabled=False(既定)= stochastic 層は従来の Bernoulli = バイト一致。
      emergent: A2。True で **present_cap を一切見ない**(資格者 = 在場者)。
      rain: 当日が雨/雪か。None(既定)= 天候共変量は**不活性**
            (weather.mode=generated 以外・weather OFF・暦 OFF はここに落ちる)。

    返り: present の pid の**ソート済み**リスト(同 (seed, day) で常に同一集合・同一順)。
    """
    cap = int(present_cap)
    tiers: list[list[str]] = [[] for _ in range(_N_TIERS)]
    for rec in recs:
        if _eligible(rec, day, weekday, hub, habit, rain):
            tiers[_TIER.get(rec.key, _N_TIERS - 1)].append(rec.pid)

    present: list[str] = []
    if emergent:
        # A2 cap 撤去: 当日の資格者が**そのまま**在場者(切らない・ランキングを引かない)。
        # → 総在場は個人の予定の合算として創発する。cap も quota_by_ratio も通らない。
        for ids in tiers:
            present.extend(ids)
        present.sort()
        return present
    if tier_quota:
        # 層別クォータ: cap を当日資格者の比率で分配し、溢れた層だけ当日ランキングで切る。
        quota = quota_by_ratio([len(ids) for ids in tiers], cap)
        for tier in range(_N_TIERS):
            ids, q = tiers[tier], quota[tier]
            if not ids or q <= 0:
                continue
            present.extend(ids if len(ids) <= q else _ranked(ids, day, hub)[:q])
        present.sort()
        return present

    for tier in range(_N_TIERS):
        ids = tiers[tier]
        if not ids:
            continue
        room = cap - len(present)
        if room <= 0:
            break
        if len(ids) <= room:
            present.extend(ids)
        else:
            # cap で溢れる層: 当日固有の決定論ランキングで部分集合を採る(=日ごとに入れ替わる)。
            present.extend(_ranked(ids, day, hub)[:room])
            break
    present.sort()
    return present
