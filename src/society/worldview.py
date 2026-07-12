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
  checkpoint の flush_segment で退避された分は走査済み扱い(標準ランは影響なし)。
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


def _scan_responses(sim, cfg: dict) -> None:
    """前回位置から新しいイベントを1回ずつ走査し、C2 を世界の応答で更新する(R9)。

    絶対位置 = logger.total_n_events() 基準。flush_segment で退避された分は
    走査済み扱い(標準ランでは退避は起きない)。
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

    def _apply(agent, positive: bool, strong: bool) -> None:
        key = (agent.id, strong)
        used = quota.get(key, 0)
        if used >= cap[strong]:
            return
        quota[key] = used + 1
        _ctrl_apply(agent, positive, 1.0 if strong else ws, cfg)

    for e in events[start:]:
        k = e.kind
        if k == "proposal_passed":
            a = by_id.get(e.agent_id)
            if a is not None:
                _apply(a, True, True)
        elif k == "vote_result":
            a = by_id.get(e.agent_id)
            if a is not None:
                _apply(a, bool(e.payload.get("passed")), True)
        elif k == "venture_permit":
            if str(e.payload.get("outcome")) == "denied":
                a = by_id.get(e.agent_id)
                if a is not None:
                    _apply(a, False, True)
        elif k == "venture_close":
            a = by_id.get(e.agent_id)
            if a is not None:
                _apply(a, False, True)
        elif k == "venture_sale":
            a = by_id.get(e.agent_id)
            if a is not None:
                _apply(a, True, False)
        elif k == "flyer_view":
            a = by_id.get(e.payload.get("author"))
            if a is not None:
                _apply(a, True, False)
        elif k == "event_attend":
            a = by_id.get(e.payload.get("host"))
            if a is not None:
                _apply(a, True, False)
        elif k == "proposal_support":
            a = by_id.get(e.payload.get("author"))
            if a is not None:
                _apply(a, True, False)
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
    pioneer = sum(1 for e in log.events[start:] if e.kind in _PIONEER_KINDS)
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
