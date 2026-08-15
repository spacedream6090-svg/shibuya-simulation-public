"""内省の現実的タイミング RFX-A / 発火文脈の観測 RFX-O(第116バッチ 2026-08-15・既定 OFF)。

正典: docs/plans/reflection-leisure-plan.md §3(RFX-A)・§3.2 の疑似コード。

何を解く問題か(同書 §0 / §2.2 の実測)
--------------------------------------
**設計は既にリポジトリの中に在り、来街者にだけ適用されている。** 来街者は「帰路の電車で
今日を内省」する(`exit_area{homing}` が `reflect_step` を立てる)ので、100 日ランで
**来街者の内省の 35.6% が 17-20 時**に発火している。一方 **居住者の内省は 100.0% が 22-06 時**。
理由は設計判断ではなく、居住者の帰宅が「入館 → 即就寝」の 1 アクションに畳まれていて
**"帰路" という状態が存在しない**から。⇒ 本レーンは新機能ではなく**内部不整合の解消**である。

文献(同書 §1)
--------------
- 内省は「外向きの注意を要求されていない時間」全般。心のさまよいは**起床時間の 46.9%**
  (Killingsworth & Gilbert 2010 *Science* 330:932)、レビューの推奨レンジ 25-50%
  (Smallwood & Schooler 2015)。
- ★**空白時間は内省を生まない**: Baird ら 2012 *Psychol Sci* 23:1117 で
  **休息は「休憩なし」と差ゼロ**、要求の**低い**課題だけが孵化効果を出した。
  Wilson ら 2014 *Science* 345:75 では男性の 67% が「ただ考える」より電気ショックを選んだ。
  ⇒ 発火条件に入れるべきは**歩行・在宅メディア**であって「立ち止まって何もしていない」ではない。
- ★**活動名のホワイトリストという方式そのものは文献の支持が弱い**(K&G 2010 補遺:
  「活動の種類は心のさまよいの個人間分散の **3.5%** しか説明しない」)。効いているのは
  **作業記憶の負荷**である。下の述語は本来その代理でしかない —— 正直に書いておく。
  本シムで負荷に最も近い既存量は `activity ∈ {working, commuting}` と勤務窓なので、
  それで近似している(day_plan の `cat3` を直接読むほうが忠実、というのは同書 §3.2 の推奨)。
- 「一人」ゲートは文献の支持が明確で**活動に依らず効く**(Nguyen, Ryan & Deci 2018 研究 2)。
  ただし「**選んで**一人になったか」が価値を決める(同 研究 4)という含意は本シムでは
  表現できない(孤独は移動の副産物であって選択ではない)= **未実装と正直に申告する**。
- ★現行の就寝前テンプレは**良眠者ではなく不眠者の型**(Lemyre ら 2020 *Sleep Med Rev*
  50:101253 = 1,730 本の系統的レビュー: 正常な入眠移行は「感覚的イメジャリ・高次認知の
  脱活性化」で、**計画や問題解決に関わる就寝前思考は不眠者の特徴**)。
  → `sleep_task_rewrite` トグルで言い換えを用意する(**プロンプト変更 = 挙動変化**なので
  必ずトグル配下。既定 OFF はバイト一致)。

設計の骨子 — 「発火点を足さない。予約の"満期"を状態が決める」
--------------------------------------------------------------
    現行:  就寝イベント ─arm→ reflect_step = 就寝step+1 → 必ずその step で発火
    RFX-A: 起床 ─arm→ 予約 ─┬ 務め終了後、最初の「内省的瞬間」で発火(早期発火)
                             └ 来なければ 就寝step+1 で発火(= 現行と同一)

**1 予約 = 1 発火の機械保証**: 早期発火は `reflect_suppress_arm` を 1 にし、次に来る
予約イベント(就寝 / 宿泊チェックイン / 帰路退出)がそれを消費して**1 回だけ見送る**。
⇒ 早期発火 1 回が将来の予約 1 回を厳密に相殺する。二重発火は原理的に起こらない。

★正直な限界(呼数が一致しない唯一のケース。同書 §3.3): 夕方に早期発火した個体が
**その夜に眠らないままランが終わる**場合、OFF ではその内省はランの外だったのに ON では
発生する。**ラン境界効果として ±1 日ぶん**の差が出うる。

R1
--
- **既定 `mode: "sleep"`** = 本 module のすべての関数が恒等 / no-op = 現行とバイト一致。
- **乱数ゼロ**: 発火判定に乱数を 1 本も引かない(全述語が決定論)。`resume == straight` は無風。
- **新 state は int 2 本**(`reflect_moment_day` / `reflect_suppress_arm`)。
- **DPH-B(`lod.budget.tiers`)ON が前提**(同書 §3.5): 予算外の内省が飽和帯へ移るので、
  二層予算の life レーンが無いと一般呼(social/reply)の granted 率を押し下げる。
  finals では既に `lod.budget.tiers.enabled: true`。
"""
from __future__ import annotations

from ..world.perception import hearers_of
from . import routine

#: 発火文脈のタグ語彙(RFX-O)。`sleep` は現行どおりの就寝時発火。
CONTEXTS: tuple[str, ...] = ("home", "media", "walk", "transit", "sleep")

MODES: tuple[str, ...] = ("sleep", "reflective_moment")

DEFAULTS: dict = {
    "mode": "sleep",              # ★既定 = 現行と完全同値
    "evening_floor_hour": 18,     # 務めの終わりが無い個体(無職・学生・来街者)の窓の下限
    "context_tag": False,         # RFX-O: reflect payload に when / context を足す(観測のみ)
    "sleep_task_rewrite": False,  # 就寝前テンプレを良眠者型へ言い換える(プロンプト変更)
}


def build_cfg(raw) -> dict:
    """conf の `reflection.timing` ブロックを型強制つきで正準化する(既定 = 現行同値)。"""
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(raw):
            raw = OmegaConf.to_container(raw, resolve=True)
    except Exception:                              # noqa: BLE001
        pass
    raw = dict(raw or {})
    mode = str(raw.get("mode", DEFAULTS["mode"]))
    if mode not in MODES:
        raise ValueError(f"reflection.timing.mode は {MODES} のいずれか: {mode!r}")
    return {
        "mode": mode,
        "evening_floor_hour": int(raw.get("evening_floor_hour",
                                          DEFAULTS["evening_floor_hour"])),
        "context_tag": bool(raw.get("context_tag", DEFAULTS["context_tag"])),
        "sleep_task_rewrite": bool(raw.get("sleep_task_rewrite",
                                           DEFAULTS["sleep_task_rewrite"])),
    }


def cfg_of(sim) -> dict:
    """設定を返す(初回のみ `sim.cfg` から遅延構築してキャッシュ)。"""
    cfg = getattr(sim, "_rfxcfg", None)
    if cfg is None:
        cfg = build_cfg(((sim.cfg.get("reflection", {}) or {}).get("timing", {})))
        sim._rfxcfg = cfg
    return cfg


def moment_mode(sim) -> bool:
    return cfg_of(sim)["mode"] == "reflective_moment"


def context_tag_on(sim) -> bool:
    """RFX-O(発火文脈の観測)。**payload にキーが増える = L1 が変わる**ので既定 OFF。"""
    return bool(cfg_of(sim)["context_tag"])


# --------------------------------------------------------------------------- #
# 予約(arm)— 既存 3 箇所の `agent.reflect_step = step + 1` の唯一の置換点
# --------------------------------------------------------------------------- #
def arm(sim, agent, step: int) -> None:
    """就寝 / 宿泊チェックイン / 帰路退出の予約。**既定 mode では現行と 1 ビットも変わらない**。

    reflective_moment では、直前に早期発火していたら**この 1 回だけ見送る**
    (= 1 予約 1 発火の機械保証)。見送りフラグはそこで消費されるので、次の予約は通る。
    """
    if cfg_of(sim)["mode"] == "reflective_moment" \
            and int(getattr(agent, "reflect_suppress_arm", 0)):
        agent.reflect_suppress_arm = 0             # 消費(次の予約は通常どおり立つ)
        agent.reflect_moment_day = -1
        return
    agent.reflect_step = step + 1
    agent._reflect_when = None                     # 就寝発火は文脈タグ無し(= "sleep")


def on_wake(sim, agent, sim_min: int) -> None:
    """起床時に「今日ぶんの早期発火」を 1 枚だけ armed にする(既定 mode では no-op)。

    ★ここが「予約の起点を務めの終わりへ前倒しする」の実体。就寝イベントで arm すると
    遅すぎる(就寝は内省的瞬間の**後**に来る)ので、起床時に予約だけ置いておき、
    満期(= 内省的瞬間)が来たら発火する。
    """
    if cfg_of(sim)["mode"] != "reflective_moment":
        return
    agent.reflect_moment_day = int(sim_min) // 1440


# --------------------------------------------------------------------------- #
# 窓の始まり(務めの終わり)— 決定論・乱数ゼロ・既存の個体定数のみから導出
# --------------------------------------------------------------------------- #
def window_from_min(agent, cfg: dict) -> int:
    """その個体の「務めの終わり」= 早期発火の窓の始まり(分 of day)。

    `max(work_end_min, part_time.end_min, evening_floor)`。
    ★日跨ぎ勤務(`work_wraps` = 夜勤 22:00→06:00)は円環で解く: その個体の「一日の終わり」は
      朝なので、夕方の下限を当てず **work_end_min をそのまま窓の始まり**にする
      (夜勤明けの午前が本人の"帰路"に相当する)。
    """
    floor = int(cfg["evening_floor_hour"]) * 60
    we = int(getattr(agent, "work_end_min", -1) or -1)
    if getattr(agent, "work_wraps", False) and we >= 0:
        return we % 1440
    ends = [floor]
    if we >= 0:
        ends.append(we)
    pt = getattr(agent, "part_time", None)
    if pt:
        ends.append(int(pt.get("end_min", 0)))
    return max(ends) % 1440


# --------------------------------------------------------------------------- #
# 発火(満期判定)
# --------------------------------------------------------------------------- #
def _low_load_context(agent) -> str | None:
    """低負荷かつ**占有されている**文脈(§1.4: 空白時間ではなく単純作業中)。O(1)・乱数ゼロ。

    ★`transit` は語彙に在るが**現行の実装では到達しない**: 車内層(transit_interior)は
      `loc == "outside"` の個体にしか付かず、圏外の個体は下の述語で除外されるため。
      来街者はそもそも `exit_area{homing}` で既に帰路内省を持っている(= 本レーンが
      解こうとしている不整合の"正しい側")。**在るふりをせず、そう書いておく。**
    """
    if getattr(agent, "activity", "") == "media":
        return "media"                             # 在宅メディア(受動的・低負荷)
    if agent.route and getattr(agent, "trip_mode", "") == "walk":
        return "walk"                              # 一人で歩いている(Oppezzo 2014 / Mason 2007)
    if routine._at_home(agent):
        return "home"                              # 在宅(★立ち止まった在宅も含む = §1.4 の限界)
    return None


def moment_context(sim, agent, step: int, sim_min: int, cfg: dict,
                   cal: dict | None) -> str | None:
    """いま「内省的瞬間」か。該当すれば文脈タグ、そうでなければ None。**乱数ゼロ**。

    述語はすべて既存(`hearers_of` / `activity` / 勤務窓 / `sleeping` / `_at_home`)。
    ★順序は計画書の疑似コードから **O(1) の判定を前に寄せてある**(結果は同一で、
      空間索引を引く回数だけが減る)。
    """
    day = int(sim_min) // 1440
    if int(getattr(agent, "reflect_moment_day", -1)) != day:
        return None                                # 今日ぶんは未 arm or 消化済み
    if int(getattr(agent, "reflect_step", -1)) >= 0:
        return None                                # 既に予約済み(二重発火の防止)
    if agent.sleeping or agent.loc == "outside":
        return None
    if getattr(agent, "activity", "") in ("working", "commuting"):
        return None
    if int(getattr(agent, "detained_until", 0)) > step:
        return None
    if getattr(agent, "sick", False) or int(getattr(agent, "severity", 0)) >= 2:
        return None
    if (sim_min % 1440) < window_from_min(agent, cfg):
        return None                                # 務めが終わっていない
    if routine.in_work_window(agent, sim_min, cal) \
            or routine.in_part_time_window(agent, sim_min):
        return None
    ctx = _low_load_context(agent)
    if ctx is None:
        return None
    # ---- 一人 or 同居者のみ(最後 = 空間索引を引く回数を最小化)----
    idx = getattr(sim, "percept_index", None)
    company = hearers_of(agent, idx if idx is not None else sim.agents,
                         float(sim.cfg.world.perception_radius_m))
    if company:
        mates = set(getattr(agent, "housemates", ()) or ())
        if not all(int(o.id) in mates for o in company):
            return None
    return ctx


def arm_moments(sim, step: int, sim_min: int) -> None:
    """step 末: 内省的瞬間を迎えた個体の予約を**この step で満期にする**(既定 mode は即 return)。

    既存の内省ループ(`_reflect_due`)の直前に置く。新しい全個体走査を 1 本足すが、
    **LLM 呼は 1 本も足さない**(立てた予約は必ず後で立つはずだった予約の前倒しで、
    `reflect_suppress_arm` がその夜の予約を 1 回見送らせる)。
    """
    cfg = cfg_of(sim)
    if cfg["mode"] != "reflective_moment":
        return
    cal = getattr(sim, "calendarcfg", None)
    for agent in sim.agents:
        ctx = moment_context(sim, agent, step, sim_min, cfg, cal)
        if ctx is None:
            continue
        agent.reflect_step = step                  # この step の内省ループが拾う
        agent.reflect_moment_day = -1              # 今日ぶんは消化
        agent.reflect_suppress_arm = 1             # 次の予約(就寝/宿泊/退出)を 1 回見送る
        agent._reflect_when = ctx


def when_of(agent):
    """この個体の今回の内省が「早期発火」ならその文脈タグ、就寝発火なら None。"""
    return getattr(agent, "_reflect_when", None)
