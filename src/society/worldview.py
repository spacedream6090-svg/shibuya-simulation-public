"""主観的世界モデル(第20バッチ 2026-07-12。ユーザー要望)。

「各エージェントには世界を各々の解釈で捉えた世界があり、その中で仮説を立て検証しながら
世界がどのようなものかを知り、実際に行動して変えていく」— この主観的世界モデルを
3成分で実装し、日次で観測可能にする層。

  C1 期待形成(expectations): 場所×時間帯の人出の期待を経験から EMA 学習する
     (=個人が持つ検証可能な仮説)。発火時に「期待と実際の差」が大きいときだけ1行注入。
     期待誤差が日々縮むこと=「世界を知っていく」の定量化(予測地図: Stachenfeld SR)。
  C2 可制御性(controllability): 「自分が動けば世界は応えるか」(Bandura outcome
     expectancy / Seligman 学習性無力感)。**全員 0.5 から出発**し、自分の働きかけへの
     世界の応答(提案の可決/否決・出店の売上/閉店・ビラの閲覧・主催イベントへの参加・
     許可の却下)だけで分岐する=生得でなく純経験の経路依存(k* の問いに接地)。
     高低が閾値を超えたときだけ自然文1行を注入。
  C6 規範予期(norms): 「この街では新しいことを始める人がいるか」の記述的規範
     (Bicchieri / Cialdini descriptive norm)を直近数日の開拓的行動(出店・提案・主催・
     結成)から決定論集計し、全員共通の1行として注入=世界改変を試みる/抑える社会側の
     閾値。個人の規範感受性・発火変調(norm_gain)は将来の seam(本版は観測と提示のみ)。

観測: 日次 L1 "worldview" イベント(agent ごと: controllability・期待テーブル規模・
  当日の平均期待誤差)+ 街レベルの規範率(agent_id=-1)。後処理
  scripts/analyze_worldview.py が仮説検証ループ(誤差収束)・無力/効力の分岐・
  解釈の分岐を分析する。

接地: docs/research/world-models.md §5-B C1/C2/C6・§3(理論)。
決定論: **乱数ゼロ**(EMA・カウント・固定閾値のみ。既存 stream の draw 順に無影響)。
R1 呼数不変: generate() を1本も足さない(発火者のプロンプト内容のみ)。
no-fingerprint: 注入は自然文(mood_text 方式)。因子語・構成概念名は書かない。
既定 OFF(enabled=false)= 状態・イベント・プロンプトとも既定と完全一致(バイト一致)。
制約: C2 の応答走査は logger を絶対位置(total_n_events 基準)で増分走査する。
  ~~checkpoint の flush_segment で退避された分は走査済み扱い~~ → **第98バッチ W4-A で解消**:
  flush の直前に absorb_before_flush() が未走査区間を carry へ吸い上げるので、分割 straight
  (checkpoint_every / flush_every_steps)も plain と一致する。

resume(第98バッチ W4-A で解消)
--------------------------------
走査位置 _wv_pos は**プロセス内 logger カウンタ由来**なので resume で 0 に戻る。走査は
**日境界にしか走らない**ため、「直前の日境界 〜 checkpoint」区間のイベントは resume 後の
新 logger に存在せず、そのままでは C2 応答も C6 開拓行動数もまるごと落ちていた(日カウンタ
_wv_day を保存しても埋まらない=**イベント本体に相当する情報**が要る)。
そこで checkpoint_state()/restore_checkpoint() が**未走査区間の射影**だけを運ぶ:
  - C2: (対象 agent_id, 肯定か, 強い応答か) の三つ組列(_wv_carry)。日次上限(ctrl_daily_*)
    を保存時に先取り適用するので **要素数は高々 n_agents×(strong+weak 上限)= 有界**。
    上限超過分を捨てても消費時に必ず弾かれる=同値(応答の解決可否は agent_id ごとに一意)。
  - C6: 未走査区間の開拓行動数(_wv_pioneer_carry。int 1 個)。
checkpoint_state() は **1 バイトも sim を書き換えない純粋な読み取り**なので、straight ラン
(checkpoint を挟む分割 straight を含む)の挙動は完全に不変。
"""
from __future__ import annotations

from collections import deque

from .observer.schema import Event

DEFAULTS = {
    "enabled": False,
    # ---- C1 期待形成 ----
    "crowd_alpha": 0.2,       # 期待 EMA の学習率
    "expect_line": True,      # 「いつもと違う」1行の注入
    "gap_min": 3.0,           # 期待との人数差がこれ以上で「いつもと違う」と感じる
    # ---- C2 可制御性 ----
    "ctrl_line": True,        # 自然文1行の注入
    "ctrl_step": 0.15,        # 応答1件あたりの更新幅(限界効用逓減)
    "weak_scale": 0.3,        # 弱い応答(売上・閲覧・参加)の倍率
    "ctrl_hi": 0.7,           # これ以上で「手応え」文
    "ctrl_lo": 0.3,           # これ以下で「無力」文
    # 応答の日次計上上限(感覚更新の生理的上限。応答の洪水でも1日に動く量を抑え
    # 全員が天井に張り付く飽和を防ぐ=mock 30日で全員0.9台に達した実測への対策)。
    "ctrl_daily_strong": 2,
    "ctrl_daily_weak": 3,
    # ---- C6 規範予期 ----
    "norm_line": True,        # 記述規範1行(全員共通)
    "norm_window_days": 7,    # 開拓的行動の集計窓(日)
    "norm_hi": 0.02,          # 1人日あたりの率がこれ以上で「珍しくない」
    # ---- 観測 ----
    "snapshot": True,         # 日次 worldview イベント
}

# 開拓的行動(C6 記述規範の分子)= 世界に新しい構造を作る行為。
_PIONEER_KINDS = ("venture_open", "proposal", "event_host", "group_found")

# C2: 世界の応答シグナル(kind → 判定)。strong=±1.0 / weak=±weak_scale。
_CTRL_CLIP = (0.02, 0.98)


def build_cfg(raw: dict | None) -> dict:
    """conf の worldview ブロックを正準化(既定 OFF=現行挙動と完全同一)。"""
    raw = dict(raw or {})
    cfg = dict(DEFAULTS)
    cfg.update({k: raw[k] for k in raw if k in DEFAULTS})
    cfg["enabled"] = bool(cfg["enabled"])
    for k in ("expect_line", "ctrl_line", "norm_line", "snapshot"):
        cfg[k] = bool(cfg[k])
    for k in ("crowd_alpha", "gap_min", "ctrl_step", "weak_scale",
              "ctrl_hi", "ctrl_lo", "norm_hi"):
        cfg[k] = float(cfg[k])
    for k in ("norm_window_days", "ctrl_daily_strong", "ctrl_daily_weak"):
        cfg[k] = int(cfg[k])
    return cfg


# ------------------------------------------------------------------ 毎step
def _place_key(agent) -> str:
    return agent.building or agent.node


def phase(sim, step: int, sim_min: int) -> None:
    """毎step: 日次境界処理→在街エージェントの人出観測→期待 EMA 更新。

    scheduler が位置確定後(percept_index 構築直後)に呼ぶ。OFF は即 return
    (状態・イベント・乱数とも既定と不変)。走査は id 昇順=決定論。
    """
    cfg = getattr(sim, "worldviewcfg", None)
    if not cfg or not cfg["enabled"]:
        return
    day = sim_min // 1440
    if day != getattr(sim, "_wv_day", -1):
        _on_day(sim, day, step, sim_min, cfg)
    # 場所ごとの人出(同じ建物 or 同じノード)を1パスで数える。
    counts: dict[str, int] = {}
    awake = [a for a in sim.agents if a.loc != "outside" and not a.sleeping]
    for a in awake:
        k = _place_key(a)
        counts[k] = counts.get(k, 0) + 1
    sim._wv_counts = counts
    band = (sim_min // 60) % 24 // 4                 # 4時間帯 ×6
    alpha = cfg["crowd_alpha"]
    for a in awake:                                   # id 昇順(sim.agents 順)
        exp = getattr(a, "wv_expect", None)
        if exp is None:
            exp = {}
            a.wv_expect = exp
            a._wv_err_sum = 0.0
            a._wv_err_n = 0
        key = (_place_key(a), band)
        actual = float(counts[_place_key(a)])
        cur = exp.get(key)
        if cur is None:
            exp[key] = actual                         # 初見=仮説の形成
        else:
            a._wv_err_sum += abs(actual - cur)        # 仮説の検証(誤差の記録)
            a._wv_err_n += 1
            exp[key] = cur + alpha * (actual - cur)   # 仮説の更新


# ------------------------------------------------------------------ 日次境界
def _ctrl_apply(agent, positive: bool, scale: float, cfg: dict) -> None:
    c = getattr(agent, "controllability", 0.5)
    step_ = cfg["ctrl_step"] * scale
    c = c + step_ * (1.0 - c) if positive else c - step_ * c
    agent.controllability = min(_CTRL_CLIP[1], max(_CTRL_CLIP[0], c))


def _response_of(e):
    """イベント → (対象 agent_id, 肯定か, 強い応答か)。応答でなければ None(純関数・状態不変)。

    _scan_responses(消費側)と checkpoint_state(保存側)が**同じ写像**を使うための単一の源。
    対象 id は解決前の生値(payload 由来で None / 未知 id もありうる)を返し、解決は消費側が行う。"""
    k = e.kind
    if k == "proposal_passed":
        return (e.agent_id, True, True)
    if k == "vote_result":
        return (e.agent_id, bool(e.payload.get("passed")), True)
    if k == "venture_permit":
        if str(e.payload.get("outcome")) == "denied":
            return (e.agent_id, False, True)
        return None
    if k == "venture_close":
        return (e.agent_id, False, True)
    if k == "venture_sale":
        return (e.agent_id, True, False)
    if k == "flyer_view":
        return (e.payload.get("author"), True, False)
    if k == "event_attend":
        return (e.payload.get("host"), True, False)
    if k == "proposal_support":
        return (e.payload.get("author"), True, False)
    return None


def _scan_responses(sim, cfg: dict) -> None:
    """前回位置から新しいイベントを1回ずつ走査し、C2 を世界の応答で更新する(R9)。

    絶対位置 = logger.total_n_events() 基準。flush_segment で退避された分は
    走査済み扱い(標準ランでは退避は起きない)。resume で運ばれた未走査分
    (_wv_carry)は**時間順で先**なので、新 logger の走査より前に消費する
    (日次キャップ quota も carry から順に食う=straight と完全同一)。
    """
    log = sim.logger
    flushed = getattr(log, "_n_flushed", 0)
    pos_abs = getattr(sim, "_wv_pos", 0)
    start = max(0, pos_abs - flushed)
    events = log.events
    by_id = sim.agent_by_id if hasattr(sim, "agent_by_id") else {
        a.id: a for a in sim.agents}
    ws = cfg["weak_scale"]
    # 日次計上キャップ(この走査=1日ぶん)。応答の洪水でも1日に動く量を抑える。
    quota: dict[tuple[int, bool], int] = {}
    cap = {True: cfg["ctrl_daily_strong"], False: cfg["ctrl_daily_weak"]}

    def _apply(aid, positive: bool, strong: bool) -> None:
        agent = by_id.get(aid)
        if agent is None:                  # 未知 id / payload 欠落 = 計上もキャップ消費もしない
            return
        key = (agent.id, strong)
        used = quota.get(key, 0)
        if used >= cap[strong]:
            return
        quota[key] = used + 1
        _ctrl_apply(agent, positive, 1.0 if strong else ws, cfg)

    for aid, positive, strong in (getattr(sim, "_wv_carry", None) or ()):
        _apply(aid, positive, strong)      # resume で運ばれた「日境界〜checkpoint」区間
    sim._wv_carry = []
    for e in events[start:]:
        r = _response_of(e)
        if r is not None:
            _apply(r[0], r[1], r[2])
    sim._wv_pos = flushed + len(events)


def _on_day(sim, day: int, step: int, sim_min: int, cfg: dict) -> None:
    """日次境界: C2 応答走査 → C6 規範窓の更新 → worldview スナップ → 誤差リセット。"""
    first = getattr(sim, "_wv_day", -1) < 0
    sim._wv_day = day
    if not hasattr(sim, "_wv_pioneer"):
        sim._wv_pioneer = deque(maxlen=max(1, cfg["norm_window_days"]))
        sim._wv_pos = 0
    # 直近イベントの走査(C2 応答)と、同じ区間の開拓的行動カウント(C6)。
    log = sim.logger
    flushed = getattr(log, "_n_flushed", 0)
    start = max(0, getattr(sim, "_wv_pos", 0) - flushed)
    pioneer = (int(getattr(sim, "_wv_pioneer_carry", 0))     # resume で運ばれた未走査分
               + sum(1 for e in log.events[start:] if e.kind in _PIONEER_KINDS))
    sim._wv_pioneer_carry = 0
    _scan_responses(sim, cfg)                         # ここで _wv_pos が進む
    sim._wv_pioneer.append(pioneer)
    if first:
        return                                        # 起動直後(day確定のみ)
    if not cfg["snapshot"]:
        _reset_err(sim)
        return
    n = max(1, len(sim.agents))
    rate = (sum(sim._wv_pioneer)
            / max(1, len(sim._wv_pioneer)) / n)       # 1人日あたり開拓行動率
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=-1,
                         kind="worldview", x=0.0, y=0.0,
                         payload={"norm_rate": round(rate, 5),
                                  "pioneer_1d": pioneer}))
    for a in sim.agents:                              # id 昇順=決定論
        err_n = getattr(a, "_wv_err_n", 0)
        err = (getattr(a, "_wv_err_sum", 0.0) / err_n) if err_n else None
        sim.logger.log(Event(
            step=step, sim_min=sim_min, agent_id=a.id, kind="worldview",
            x=a.x, y=a.y,
            payload={"ctrl": round(getattr(a, "controllability", 0.5), 4),
                     "expect_n": len(getattr(a, "wv_expect", {}) or {}),
                     "err_mean": (round(err, 3) if err is not None else None),
                     "err_n": err_n}))
    _reset_err(sim)


def _reset_err(sim) -> None:
    for a in sim.agents:
        if getattr(a, "wv_expect", None) is not None:
            a._wv_err_sum = 0.0
            a._wv_err_n = 0


# ------------------------------------------------------------------ checkpoint
def checkpoint_state(sim) -> dict | None:
    """未走査区間(直前の日境界 〜 いま)の C2 応答射影と C6 開拓行動数を返す(OFF は None)。

    ★**純粋な読み取り**(sim も logger も 1 バイトも書き換えない)= straight ラン不変。
    engine/checkpoint.py が save 時に1回呼ぶ(ablate.placebo_state と同じ口の切り方)。

    有界性: 三つ組は (対象 id, strong) ごとに日次上限(ctrl_daily_strong/weak)まで**しか**
    積まない。上限超過分は消費側の quota で必ず弾かれるので、捨てても結果は同値
    (応答の解決可否 by_id.get(id) は id ごとに一意=同 id の要素は全て解決するか全て
    しないかのどちらか。したがって「先頭 cap 件」が straight で実際に適用される要素と一致)。
    """
    cfg = getattr(sim, "worldviewcfg", None)
    if not cfg or not cfg["enabled"]:
        return None
    log = getattr(sim, "logger", None)
    if log is None:
        return None
    flushed = getattr(log, "_n_flushed", 0)
    start = max(0, int(getattr(sim, "_wv_pos", 0)) - flushed)
    tail = log.events[start:]
    cap = {True: int(cfg["ctrl_daily_strong"]), False: int(cfg["ctrl_daily_weak"])}
    quota: dict[tuple, int] = {}
    pending: list = []

    def _push(aid, positive: bool, strong: bool) -> None:
        key = (aid, bool(strong))
        used = quota.get(key, 0)
        if used >= cap[bool(strong)]:
            return
        quota[key] = used + 1
        pending.append((aid, bool(positive), bool(strong)))

    for aid, positive, strong in (getattr(sim, "_wv_carry", None) or ()):
        _push(aid, positive, strong)       # 前回 resume ぶんも取りこぼさない(時間順で先)
    for e in tail:
        r = _response_of(e)
        if r is not None:
            _push(r[0], r[1], r[2])
    pioneer = (int(getattr(sim, "_wv_pioneer_carry", 0))
               + sum(1 for e in tail if e.kind in _PIONEER_KINDS))
    return {"pending": pending, "pioneer": pioneer}


def absorb_before_flush(sim) -> None:
    """L1 バッファが flush_segment で空になる**前**に、未走査区間を carry へ吸い上げる(OFF は no-op)。

    走査は日境界にしか走らないので、日の途中で flush が入ると [_wv_pos, flush 点) のイベントが
    バッファから消えて二度と読めない(本 module 冒頭が「退避された分は走査済み扱い」と註記して
    いた**既存の欠落**。実測: worldview ON・checkpoint_every=40 の分割 straight は plain と
    L1 が 3 行食い違っていた=第98 W4-A で確認)。checkpoint_state と同じ有界射影で先に吸って
    おけば、日境界の走査が「carry + flush 後のイベント」で丸ごと 1 日分になり、分割 straight が
    plain と一致し、resume も同じ土俵に乗る。
    ★flush を 1 度も挟まないラン(checkpoint_every=0 かつ flush_every_steps=0 = 既定)では
      本関数は 1 度も呼ばれない = 既定ランは 1 バイトも変わらない。
    """
    st = checkpoint_state(sim)
    if st is None:                                    # worldview OFF
        return
    sim._wv_carry = list(st["pending"])
    sim._wv_pioneer_carry = int(st["pioneer"])
    log = sim.logger
    sim._wv_pos = getattr(log, "_n_flushed", 0) + len(log.events)


def restore_checkpoint(sim, state: dict | None) -> None:
    """checkpoint_state() の返り値を sim へ戻す(None / 旧 checkpoint は no-op=従来挙動)。"""
    if not state:
        return
    sim._wv_carry = [(aid, bool(p), bool(s))
                     for aid, p, s in (state.get("pending") or ())]
    sim._wv_pioneer_carry = int(state.get("pioneer", 0) or 0)


# ------------------------------------------------------------------ プロンプト行
def expect_line(sim, agent, sim_min: int) -> str | None:
    """C1: いまの場所の「期待と実際の差」が大きいときだけ1行(トークン節約)。"""
    cfg = getattr(sim, "worldviewcfg", None)
    if not cfg or not cfg["enabled"] or not cfg["expect_line"]:
        return None
    exp = getattr(agent, "wv_expect", None)
    counts = getattr(sim, "_wv_counts", None)
    if not exp or not counts:
        return None
    band = (sim_min // 60) % 24 // 4
    key = (_place_key(agent), band)
    cur = exp.get(key)
    if cur is None:
        return None
    actual = float(counts.get(_place_key(agent), 0))
    gap = actual - cur
    if abs(gap) < cfg["gap_min"]:
        return None
    if gap > 0:
        return "この場所、いつものこの時間より人が多い気がする。"
    return "この場所、いつものこの時間より人が少ない気がする。"


def ctrl_line(agent, cfg: dict | None) -> str | None:
    """C2: 可制御性が閾値を超えたときだけ自然文1行(mood_text 方式・因子語なし)。"""
    if not cfg or not cfg["enabled"] or not cfg["ctrl_line"]:
        return None
    c = getattr(agent, "controllability", None)
    if c is None:
        return None
    if c >= cfg["ctrl_hi"]:
        return "最近、自分が動けば何かが変わる、という手応えを感じている。"
    if c <= cfg["ctrl_lo"]:
        return "最近、自分が何をしても世界は変わらない気がしている。"
    return None


def norm_line(sim, cfg: dict | None) -> str | None:
    """C6: 「新しいことを始める人がいる街か」の記述規範1行(全員共通・k非依存)。"""
    if not cfg or not cfg["enabled"] or not cfg["norm_line"]:
        return None
    win = getattr(sim, "_wv_pioneer", None)
    if not win or len(win) < 2:                       # 集計が育つまで沈黙
        return None
    n = max(1, len(sim.agents))
    rate = sum(win) / len(win) / n
    if rate >= cfg["norm_hi"]:
        return "この街では、新しく何かを始める人(出店・提案・イベント)は珍しくない。"
    if sum(win) == 0:
        return "この街で、新しく何かを始める人はほとんど見かけない。"
    return None
