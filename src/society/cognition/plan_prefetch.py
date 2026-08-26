"""夜間計画プリフェッチ PPF(``planning.day_plan.prefetch``・**既定 OFF**)。

正典
----
- ユーザー要望(2026-08-25)「計画は夜のうちに作って処理を分散」
- 実測の背景(``docs/log/devlog.md`` 2026-08-25 夜「リサーチ3レーン」レーンB = 24c 土曜
  前半 12h・L1 2,555 万行): 朝の計画需要は**二峰一斉型**で、起床の 54% が 04:40 の
  1 step に 34,019 人・**全需要の 92% が 04:40-09:59** に集中する。二層予算(DPH-B)の
  FIFO 繰り越しは、その山を約 10 時間ぶん滞留させて捌く(半日実測: 滞留 79,621 件)。
  一方、シミュ内 00:10-04:30 の夜間 step では life レーンがほぼ空で回っている。

何を解く問題か
--------------
**計画の前提はシミュ内 0 時に確定している。**
  - 当日の日付・天気 … ``scheduler._phase_calendar_weather``(日境界で 1 回・sim へ保持)
  - 当日の予定       … ``scheduler._phase_schedule_gc``(日境界で過去予定を失効)
  - 計画の日スタンプ … ``day_plan.apply`` が ``plan["day"] = sim_min // 1440``
したがって **0 時を過ぎていれば「その日の計画」を就寝中に先行生成しても前提は正しい**。
本 module は「いつ撃つか」だけを前へずらす。生成そのもの(プロンプト構築・検証・修復・
適用・簿記)は**起床時とまったく同じ既存経路**(``planning.make_plan``)を通る。

設計(v1 = 就寝中コホート限定・低リスク)
------------------------------------------
1. 窓 … シミュ内 ``[start_min, end_min)``(既定 00:10-05:50)。0 時ちょうどを避けるのは
   上の日境界処理より**後**に入るため。終端を 05:50 まで伸ばすのは、一斉起床 04:40 帯の
   後もしばらく回し切るための保険(まだ寝ている人が居る限り前倒しの余地がある)。
2. 候補 … **毎 step 再計算**する純粋な述語(下記 ``pick``)。新しい checkpoint 状態は
   1 つも作らない(= 分割ランの搬送はゼロ = ``resume == straight`` が構造的に成り立つ)。
3. 順序 … ``(sleep_until, id)`` 昇順 = **早く起きる人の計画から作る**(決定論の全順序)。
4. 予算 … 既存の ``sim.budget.take("plan")``(life レーン)が取れる範囲だけ。
   取れなくなったらその step は静かに終える = **純粋な前倒し**であって、
   取り零しの新しいモードは 1 つも作らない(起きたときに従来どおり予約される)。
   ★``lod.budget.tiers``(DPH-B)が OFF のとき、起床時の計画は**元から予算外**で撃たれる
   (``_phase_planning`` の素のループは ``take`` を通さない)。プリフェッチは tiers の
   ON/OFF に依らず**必ず ``take`` を通してから撃つ** = 予算の外へ 1 本も出さない。
5. 二重生成の防止 … 既存の ``_schedule_plan`` の ``plan_day == today`` ガードだけで足りる。
   プリフェッチは生成前に ``agent.plan_day = today`` を書く = 起床時の予約が立たない。
6. 就寝スキップの反転は**プリフェッチが選んだ個体に対してだけ**起きる。
   ``_phase_planning`` / ``_phase_planning_tiered`` の sleeping/outside スキップは
   1 バイトも触っていない(プリフェッチは別の入り口から同じ下流を呼ぶ)。

R1(既定 OFF = 現行と 1 バイト同一)
-----------------------------------
- ``enabled: false`` では ``pick`` も走らず、印(``sim._ppf_mark``)が立たず、
  L1 の payload に欄が 1 つも増えず、**乱数 stream を 1 本も引かない**。
  (``sim.planprefetchcfg`` は ``dayplancfg`` / ``planboundarycfg`` と同じ**設定の
  遅延構築キャッシュ**で、checkpoint にも L1 にも L2 にも出ない。)
- ON でも **generate() の呼び出し点を 1 つも足さない**: 1 個体 1 暦日あたりの計画は
  高々 1 本のまま(既存の ``plan_day == today`` ガードは 1 バイトも触っていない)で、
  予算の総量(``max_per_step``)も 1 も動かさない。
  ★正直に言う: それでも **ラン全体の総呼数は ON/OFF で一致しない**。計画が立つ時刻が
  変われば当日の行動が変わり、世界の軌道そのものが分岐するからである(実測 20 体 ×
  3 日 mock: plan 53 → 56 本・「ON だけに在る (個体,日)」6 件と「OFF だけに在る」3 件が
  両方向に出る)。加えてラン境界では前倒しのぶん ON が先行する(RFX-A と同型の
  ±1 日効果)。**保存されるのは「1 個体 1 暦日 ≤ 1 本」という構造の側**である。
- 新しい checkpoint 状態はゼロ。書くのは ``agent.plan_day`` と計画そのもの
  (どちらも既存の agents pickle に自然同梱)。

正直な近似・限界(隠さない)
----------------------------
(a) **記憶参照の時刻が前へずれる**。``agents/memory.py`` の ``retrieve`` は step を鍵に
    ACT-R のノイズを引き、``ep.refs`` へ**現在 step を書き込む**副作用がある
    (memory.py:371-388)。先行生成では起床時より早い step で参照が刻まれる =
    以後の基礎活性化が僅かに違う軌道に乗る。観察ランで許容する近似として明記する。
(b) **行間ダイジェストの消費点が前へずれる**。``scheduler._isl_take`` は破壊的に
    バッファを空にする(scheduler.py:2036-2041)。先行生成が消費すると起床時ぶんが
    空になるが、二重生成は起きないので**同じ 1 回の消費が早い時刻に移る**だけである。
(c) **場所は前倒ししても同じ**。``day_plan.apply`` は適用時点の ``agent.node/x/y`` を読むが、
    就寝中の個体は自宅に居て、起床時も同じ自宅から計画を立てる = v1 では追加の手当て不要。
    (例外: 入院・宿泊で ``sleeping`` になっている個体は自宅ではない。そこは起床時に
     読む位置と同じなので、やはり「前倒しでずれる」性質のものではない。)
(d) **内省の完了は近似**である。リポジトリには「その個体がその日の内省を終えた」という
    印が**存在しない**(``mem.day_summaries`` は日付を持たない直近 7 件のリスト、
    ``sim._reflect_day`` は無意識層 EMA の**世界側**の日カウンタで個体別ではない)。
    そこで「内省の予約が残っていない」= ``reflect_step < step`` で近似する。
    ``reflect_step`` は予約時に **step+1** が入り、発火で -1 に戻る(reflection.py:471-477)。
    DPH-B の繰り越しも ``reflect_step = step + 1`` を立て直すので、繰り越し中の個体は
    この述語で自然に除外される。書き戻し ``k.writeback == "off"`` のランでは
    ``begin_reflect`` が予約を消費しないまま返るので値が**過去の step のまま**残るが、
    そのときも ``< step`` は真 = 「もう発火しない予約」を正しく無視できる。
(e) **motif の抽選 step が変わる**。``planning.apply_plan_response`` は
    ``routine.maybe_roll_motif(sim, agent, step)`` をその場の step で引くので、
    ``routine.stochastic`` ON のランでは前倒しで別の目が出る(既定 OFF では no-op)。
(f) **L1 の印は day_plan v1 のときだけ**。``prefetch: true`` を足すのは ``plan_created``
    (day_plan v1 の経路)であって、旧経路の ``day_plan`` イベントには足さない。
    ``planning.day_plan.enabled: false`` のランでプリフェッチを ON にすると
    「前倒しで撃った」ことが L1 からは判らない(summary/L2 にも列を足していない)。
(g) **一括発行(``engine.batch_llm``)と組む**(第160 で修正)。v1 はここを組んでおらず、
    ``batch_llm.enabled: true, workers: 64`` のランでもプリフェッチだけが逐次経路
    (``planning.make_plan``)を 1 呼ずつ回していた(本番実測 8/26: vLLM Running 0-1・
    0.19 呼/s・step 1 の 2,025 件で約 3 時間 = 「夜のうちに量産する」設計目的が死ぬ)。
    現在は ``scheduler._plan_prefetch_batched`` が朝計画(``_phase_planning_batched``)と
    同じ 3 段(選抜順に build → 未命中だけ並行発行 → 同じ順に apply)を通る。
    ``batch_llm`` OFF のランは下の逐次ループのまま = **1 バイト同一**。
"""
from __future__ import annotations

import heapq

DEFAULTS = {
    "enabled": False,     # true=夜間(日付確定後)に就寝中エージェントの当日計画を先行生成
    "start_min": 10,      # シミュ内 00:10 から(0 時の日境界処理・日付/天気確定の後)
    "end_min": 350,       # 05:50 まで(一斉起床 04:40 帯の後も回し切る保険)
}
_BOOL_KEYS = ("enabled",)
_INT_KEYS = ("start_min", "end_min")


def _to_plain(raw):
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(raw):
            return OmegaConf.to_container(raw, resolve=True)
    except Exception:                              # noqa: BLE001 (omegaconf 不在でも動く)
        pass
    return raw


def build_cfg(raw) -> dict:
    """conf の ``planning.day_plan.prefetch`` を正準化する(plan_boundary と同型)。"""
    raw = dict(_to_plain(raw) or {})
    cfg = dict(DEFAULTS)
    for k, v in raw.items():
        if k not in DEFAULTS:
            continue
        if k in _BOOL_KEYS:
            cfg[k] = bool(v)
        elif k in _INT_KEYS:
            cfg[k] = int(v)
    cfg["start_min"] = min(1440, max(0, cfg["start_min"]))
    # 終端は必ず始端以上(逆転していたら「窓なし」= 1 件も撃たない、に潰す)
    cfg["end_min"] = min(1440, max(cfg["start_min"], cfg["end_min"]))
    return cfg


def cfg_of(sim) -> dict:
    """設定(初回のみ遅延構築してキャッシュ)。キャッシュ属性は L1/L2/乱数に出ない。"""
    c = getattr(sim, "planprefetchcfg", None)
    if c is None:
        try:
            dp = (sim.cfg.get("planning", None) or {}).get("day_plan", None) or {}
            raw = dp.get("prefetch", None)
        except Exception:                          # noqa: BLE001 (旧 config 互換)
            raw = None
        c = build_cfg(raw)
        sim.planprefetchcfg = c
    return c


def enabled(sim) -> bool:
    """夜間プリフェッチが実効か。**``planning.enabled`` が前提**(朝の計画呼びの前倒し)。

    ★``planning.day_plan.enabled`` は前提にしない: 前倒しするのは
      ``planning.make_plan`` という**共通の入り口**であって、その先が day_plan v1 か
      旧経路かを本 module は問わない(旧経路でも「夜のうちに立てる」は成立する)。
      conf のキーが ``planning.day_plan`` 配下に居るのは近傍キーとの並びのためである。
    """
    pc = getattr(sim, "planningcfg", None)
    if not (pc and pc.get("enabled")):
        return False
    return bool(cfg_of(sim)["enabled"])


def in_window(cfg: dict, sim_min: int) -> bool:
    """その step が夜間の窓 ``[start_min, end_min)`` に入っているか(当日内の分で見る)。"""
    tod = int(sim_min) % 1440
    return int(cfg["start_min"]) <= tod < int(cfg["end_min"])


# --------------------------------------------------------------------------- #
# 候補(**毎 step 再計算するステートレスな述語**)
# --------------------------------------------------------------------------- #
def is_candidate(agent, step: int, today: int) -> bool:
    """この個体の当日計画を「いま」先行生成してよいか。**決定論・乱数ゼロ**。

    条件(すべて既存フィールドだけを読む):
      1. その日の計画をまだ持っていない(``plan_day != today``)
      2. 通常の予約が 1 つも無い(``plan_step == -1``)= 起床済みの人と競合しない
      3. 就寝中(``sleeping``)= 前倒しできる時間が実際に残っている
      4. 街の中に居る(``loc != "outside"``)= 圏外の個体は既存経路でも撃たない
      5. 当日の内省の予約が残っていない(``reflect_step < step``。module docstring (d))
    """
    if int(getattr(agent, "plan_day", -1)) == int(today):
        return False
    if int(getattr(agent, "plan_step", -1)) != -1:
        return False
    if not getattr(agent, "sleeping", False):
        return False
    if getattr(agent, "loc", "") == "outside":
        return False
    return int(getattr(agent, "reflect_step", -1)) < int(step)


def _order_key(agent):
    """全順序 = (起床予定 step, agent_id) 昇順 = 「早く起きる人の計画から作る」。"""
    return (int(getattr(agent, "sleep_until", 0)), int(agent.id))


def pick(sim, step: int, sim_min: int, limit: int) -> list:
    """この step に先行生成する個体を最大 ``limit`` 人、決定論の全順序で選ぶ。

    ``heapq.nsmallest`` は ``sorted(..., key=…)[:limit]`` と**厳密に同値**(標準ライブラリの
    契約)で、25 万体を毎 step 全ソートしないための最適化にすぎない = 選抜結果は不変。
    """
    if limit <= 0:
        return []
    today = int(sim_min) // 1440
    return heapq.nsmallest(
        limit, (a for a in sim.agents if is_candidate(a, step, today)),
        key=_order_key)


def budget_room(sim) -> int:
    """この step にまだ許可されうる LLM 呼の**上限**(総量の硬い上限からの残り)。

    レーン別の残り枠までは見ない(見ても ``take`` が最終判定なので結果は変わらない)。
    ここは「候補を何人まで並べれば足りるか」を決めるためだけの数である。
    """
    budget = getattr(sim, "budget", None)
    if budget is None:
        return 0
    return max(0, int(budget.max_per_step) - int(budget.used))


# --------------------------------------------------------------------------- #
# L1 の印(**ON のときだけ payload に 1 欄増える**)
# --------------------------------------------------------------------------- #
def mark_begin(sim) -> None:
    """以降に確定する計画は「夜間の先行生成ぶん」である、と印を立てる。"""
    sim._ppf_mark = True


def mark_end(sim) -> None:
    """印を降ろす(1 個体ぶんの生成が終わるたびに必ず降ろす)。"""
    sim._ppf_mark = False


def marked(sim) -> bool:
    """いま確定しようとしている計画が先行生成ぶんか。**既定 OFF では属性すら生えない**。"""
    return bool(getattr(sim, "_ppf_mark", False))
