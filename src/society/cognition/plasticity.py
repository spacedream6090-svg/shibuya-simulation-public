"""感度 g と閾値 θ の更新則 + 初期値条件 F/N/P(第82バッチ 2026-08-01・**既定 OFF**)。

正典
----
- docs/plans/source/cognition-design-record.md **§2.5**(g の更新則)・**§2.6**(θ の恒常性)・
  **§2.7**(初期値の実験条件化)・§2.8(責務分界)・§8(観測量)・§9(% で測らない)
- docs/plans/cognition-physics-plan.md §4 第82行

何を解く問題か
--------------
第81バッチの g は **全 usable チャンネルで 1.0 固定**、θ は較正テーブルの仮値だった。
設計 §2.5/§2.6 が要求するのは履歴で動く量である。

    g_ic ← g_ic + η_ic·(r_ic − ρ·ē_ic) − λ_ic·(g_ic − g⁰_ic)      … §2.5
    θ_i  ← θ_i + μ·(f̄_i − f*)                                      … §2.6

| 記号 | 意味 | 効果 |
|---|---|---|
| `ē_ic` | そのチャンネルの最近の平均誤差 | **慣れ**(下げる) |
| `r_ic` | そのチャンネルの誤差が結果に効いた度合い | **感作**(上げる) |
| `λ_ic` | ペルソナ基準値への引き戻し | 性格の永続性 |
| `η_ic` | 可塑性 | 動きやすさ |

**LLM はここに一切関与しない**(§2.8: 感度 g は**コード**。「自分の過敏さは自分では
設定できない」)。本 module は乱数も **初期化の 1 箇所以外では引かない**。

二重過程説との対応(実装前リサーチの帰結)
------------------------------------------
この式は Groves & Thompson (1970) の **dual-process theory**(*Habituation: A
dual-process theory*, Psychological Review 77(5):419-450)そのものの形をしている。
同説は「反応の減衰(habituation)と増強(sensitization)は**別々の過程**で、両者の
代数和が観測される行動になる」とし、**弱い/反復的な刺激では慣れが優位・強い(有意な)
刺激では感作が優位**になるとした(Thompson 2009, *Habituation: A history*,
Neurobiol Learn Mem 92(2):127-134 が経緯をまとめている)。

  * 本実装の `−ρ·ē`(繰り返し来る誤差ほど鈍る)= habituation の項
  * 本実装の `+r`(結果に効いた誤差だけ効く)= sensitization の項

設計 §2.5 の「**単なる反復は慣れ、結果を伴う反復は感作**。したがって更新を生の予測誤差
だけで駆動してはならない」は、この二重過程の要請と一致する。生誤差だけで駆動すると
慣れの一過程しか無くなり、注意が単調に平坦化する。

`g` を「予測誤差の重み」と読むと、予測処理(predictive coding)の **precision 学習**
とも同型である。予測誤差ユニットの利得(= 期待精度)を経験で調整することが注意の実装で
あるという定式化(Feldman & Friston 2010; Kanai et al. 2015; Smout et al. 2019 PLoS
Biol 17(2):e2006812 「注意は予測誤差の神経表現を増強する」)に対応する。**ただし本実装は
自由エネルギー最小化を解いていない**(精度の厳密な逆分散推定ではなく、上記 2 項の
ヒューリスティックな加算である)。この点は誇張しない。

r の実装(credit assignment)
-----------------------------
設計 §2.5:「発火後の行動によるニーズ充足・ゴール進捗の**改善分**を、その発火の `S` への
**寄与比**で各チャンネルに按分する」。本実装:

  1. 認知イベントの瞬間に、その時点の S 寄与比 `share_c = contrib_c / Σcontrib` と
     「うまくいっている度合い」の基準値を控える(`_fire_pending` に 1 本追加)。
  2. `r_window_min` 分だけ経ったら、同じスカラの**差分**を取り `r_c = Δ · share_c` を
     チャンネル別の credit へ積む。
  3. 次の認知イベントの更新式がその credit を消費して 0 に戻す。

★窓は**複数同時に開く**(適格性トレース)。発火間隔(10〜30 分)が窓(60 分)より短い
  ので、窓を 1 本しか持たない実装だと次の発火が前の窓を上書きし、**感作の項が事実上
  死ぬ**(実測: 11,000 件の認知イベントに対し満期を迎えた窓が 181 件しかなかった)。

「うまくいっている度合い」は `factors.registry.outcome_scalar` が返す 1 本のスカラ
(**因子名を知ってよいのは factors 層だけ** = no-fingerprint 契約。本 module は差分の
数値しか見ない)。★正直な限界: ゴール進捗そのものは本リポジトリに数値量として存在しない
ため、r は「state ベクトルの改善分」の代理である(来歴に `r_source` として明記する)。

Δt 不変性(第79バッチとの整合)
--------------------------------
- g の更新は **認知イベント 1 回につき 1 度**(step ごとではない)。総イベント数は Δt に
  依存しないので、η/λ/ρ は Δt 非依存の無次元量になる。
- ē の EMA だけは毎 tick 更新なので、**半減期を「分」で持ち** `α = 1 − 2^(−Δt/T½)` と
  変換する。これで Δt を変えても同じ実時間の平均になる。
- θ の恒常性は **日境界 1 回**(設計 §2.6「μ の時定数は日オーダー」)。第75 dunbar の
  日境界 1 回適用と同じ流儀。

初期値条件 F / N / P(§2.7)
-----------------------------
| 条件 | `experiment.g_init.mode` | 内容 |
|---|---|---|
| F | `flat`   | 全員 g⁰ 同一(純粋対照) |
| N | `noise`  | フラット + 専用 stream のノイズ(**異質性のみ**の効果) |
| P | `persona`| ペルソナ由来(既定。**ペルソナ整合的な異質性**の効果) |

「**フラット初期値は中立ではない**」(§2.7)ので初期値そのものを実験条件として記録する。
`P ≈ N` ならペルソナの中身は効いておらず異質性だけが効いている、という分離が
`scripts/analyze_g.py` の分散分解で事後にできる。

★ **CRN(共通乱数)**: ノイズは **3 条件すべてで同じ本数だけ引く**(使うのは noise 条件
だけ)。第74 flat_traits が「sample_traits を従来どおり引いてから値を捨てる」ことで
draw 順を保ったのと同型で、条件間の比較から「乱数消費列の差」という交絡を消す。

将来拡張は**実装しない**
------------------------
設計 §2.5 の「会話相手の g に引かれる項(注意の伝染)」は **未実装**(既定 OFF ですらなく
コードが存在しない)。§10 の未決事項のままにしてある。
"""
from __future__ import annotations

import math

from ..observer.schema import Event

SCHEMA = 1

# 初期値条件(§2.7 の F / N / P)。
PERSONA, FLAT, NOISE = "persona", "flat", "noise"
G_INIT_MODES: tuple[str, ...] = (PERSONA, FLAT, NOISE)

# ノイズ専用 stream(R1: 用途別 named stream)。
NOISE_STREAM = "g_init"


# --------------------------------------------------------------------------- #
# cfg 正準化(既定 OFF)
# --------------------------------------------------------------------------- #
def build_cfg(raw: dict | None) -> dict:
    """conf の `cognition.g_update` ブロックを型強制つきで正準化する(既定 OFF)。"""
    raw = dict(raw or {})
    return {
        "enabled": bool(raw.get("enabled", False)),
        # η・λ の全体倍率(個体値は factors 層のペルソナ写像が決める)。
        "eta_scale": max(0.0, float(raw.get("eta_scale", 1.0))),
        "lam_scale": max(0.0, float(raw.get("lam_scale", 1.0))),
        # ρ: 慣れの重み(平均誤差 1σ あたりどれだけ鈍らせるか)。
        "rho": max(0.0, float(raw.get("rho", 0.10))),
        # ē の半減期[分](**分で持つ**= Δt 非依存)。
        "ebar_halflife_min": max(1.0, float(raw.get("ebar_halflife_min", 120.0))),
        # 発火から結果を測るまでの窓[分]。
        "r_window_min": max(1, int(raw.get("r_window_min", 60) or 1)),
        # r の効き幅(state スカラの差分 [-1,1] を g の単位へ写す倍率)。
        "r_gain": max(0.0, float(raw.get("r_gain", 4.0))),
        # 同時に開いていられる credit 窓の本数(適格性トレースの上限。有界化のため)。
        "max_pending": max(0, int(raw.get("max_pending", 12) or 0)),
        "g_min": max(0.0, float(raw.get("g_min", 0.05))),
        "g_max": max(0.0, float(raw.get("g_max", 5.0))),
        # ---- θ の恒常性(§2.6。**日境界 1 回**= 時定数は日オーダー)----
        "theta_mu": max(0.0, float(raw.get("theta_mu", 0.02))),
        # f*: 目標**驚き発火**率[件/日/人]。★仮値(較正は第83のパイロット)。
        #  ★θ が門番をしているのは salience(驚き)だけなので、恒常性の制御量も
        #    salience の本数にする。周期発火は較正テーブルの周期が決めており θ では
        #    下げられない = 総発火数を目標にすると到達不能な目標を追い続けてしまう。
        "theta_target_per_day": max(0.0, float(raw.get("theta_target_per_day", 8.0))),
        # f̄ の日次 EMA 重み(1.0=前日そのもの。小さいほど時定数が長い)。
        "fbar_weight": min(1.0, max(0.0, float(raw.get("fbar_weight", 0.5)))),
        "theta_min_mult": max(0.0, float(raw.get("theta_min_mult", 0.25))),
        "theta_max_mult": max(0.0, float(raw.get("theta_max_mult", 4.0))),
        # g/θ 軌跡サイドカーの採取間隔(0=書かない)。観測装置の設定。
        "log_every_steps": max(0, int(raw.get("log_every_steps", 1) or 0)),
    }


def build_init_cfg(raw: dict | None) -> dict:
    """conf の `experiment.g_init` ブロックを正準化する(既定 = 条件 P)。"""
    raw = dict(raw or {})
    mode = str(raw.get("mode", PERSONA) or PERSONA)
    if mode not in G_INIT_MODES:
        raise ValueError(f"experiment.g_init.mode が未知: {mode!r}"
                         f"(有効値: {', '.join(G_INIT_MODES)})")
    return {
        "mode": mode,
        # 条件 F/N の土台になる trait 定数(flat_traits と同じ既定 0.5)。
        "flat_value": float(raw.get("flat_value", 0.5)),
        # σ₀: ノイズ分散(§2.7「σ₀ 自体をパラメータ化する」= σ₀ 掃引は k 掃引より安い)。
        "sigma0": max(0.0, float(raw.get("sigma0", 0.30))),
    }


def enabled(sim) -> bool:
    """g/θ 更新が有効か。**発火機構(fire)が ON であることが前提**。"""
    cfg = getattr(sim, "gcfg", None)
    if not (cfg and cfg["enabled"]):
        return False
    from . import fire as _fire
    return _fire.enabled(sim)


# --------------------------------------------------------------------------- #
# 更新則(**純関数**。ユニットテストが慣れ/感作/引き戻しの 3 性質をここで固定する)
# --------------------------------------------------------------------------- #
def g_step(g: float, g0: float, r: float, ebar: float, *,
           eta: float, rho: float, lam: float,
           lo: float = 0.0, hi: float = 5.0) -> float:
    """設計 §2.5 の 1 ステップ。

        g ← g + η·(r − ρ·ē) − λ·(g − g⁰)

    3 性質(テストが固定する):
      慣れ   … r=0 で ē>0 なら g は下がる
      感作   … r>0 が ρ·ē を上回れば g は上がる
      引き戻し… r=0, ē=0 なら g は g⁰ へ幾何収束する(λ>0)
    """
    nxt = g + eta * (r - rho * ebar) - lam * (g - g0)
    return float(min(hi, max(lo, nxt)))


def theta_step(mult: float, fbar: float, target: float, *,
               mu: float, lo: float, hi: float) -> float:
    """設計 §2.6 の恒常性。θ の**個体倍率**へ適用する(文脈別の較正比は壊さない)。

        m ← m + μ·(f̄ − f*)

    発火が多すぎた個体は閾値が上がり、少なすぎた個体は下がる。**日境界 1 回**しか
    呼ばないので時定数は日オーダーになる(§2.6「短くすると発火率が定数に張り付き、
    閾値方式を選んだ意味が消える」)。
    """
    return float(min(hi, max(lo, mult + mu * (float(fbar) - float(target)))))


def ema_alpha(dt_min: float, halflife_min: float) -> float:
    """半減期[分] → 1 tick あたりの EMA 係数(Δt 非依存化)。"""
    if halflife_min <= 0.0:
        return 1.0
    return float(1.0 - math.pow(0.5, float(dt_min) / float(halflife_min)))


# --------------------------------------------------------------------------- #
# 個体の初期化(F / N / P)
# --------------------------------------------------------------------------- #
def _persona_params(sim, agent) -> dict[str, float]:
    """ペルソナ → (g⁰, η, λ, θ₀) の決定論写像。条件 F/N では trait 定数版を使う。

    ★因子名を知ってよいのは factors 層だけ。本 module は返り値の数値しか見ない。
    """
    from ..factors import registry as _reg
    if sim.ginitcfg["mode"] == PERSONA:
        return _reg.cognition_params(getattr(agent, "traits", None) or {})
    return _reg.flat_cognition_params(float(sim.ginitcfg["flat_value"]))


def ensure(sim, agent) -> None:
    """個体の g/θ 状態を 1 度だけ据える(pool 経路で途中入場した個体もここで拾う)。

    ★乱数は「ノイズ専用 stream から usable チャンネル数だけ」引く。**3 条件すべてで
    同じ本数を引き**、使うのは noise 条件だけ(CRN)。stream キーは agent.id のみで
    step を含まないので、いつ初期化されても同じ値になる(pool 途中入場でも不変)。
    """
    if getattr(agent, "_fire_g", None) is not None:
        return
    from . import fire as _fire
    from . import channels as _channels
    usable = _fire.usable_channels(sim)
    params = _persona_params(sim, agent)
    icfg = sim.ginitcfg

    rng = sim.hub.stream(NOISE_STREAM, int(agent.id))
    sigma0 = float(icfg["sigma0"])
    g0: dict[int, float] = {}
    for idx, cid, _sigma in usable:
        noise = float(rng.normal(0.0, 1.0))        # ★常に引く(CRN)
        source = _channels.BY_ID[cid].source if cid in _channels.BY_ID else ""
        bias = float(params.get(f"bias_{source}", 1.0))
        if icfg["mode"] == NOISE:
            base = params["g0"] * (1.0 + sigma0 * noise)
        elif icfg["mode"] == FLAT:
            base = params["g0"]
        else:                                      # 条件 P: ペルソナ整合的な異質性
            base = params["g0"] * bias
        g0[idx] = float(min(sim.gcfg["g_max"], max(sim.gcfg["g_min"], base)))

    agent._fire_g0 = dict(g0)
    agent._fire_g = dict(g0)                       # g(0) = g⁰(履歴はここから動かす)
    agent._fire_g_init = dict(g0)                  # ★ g(0) を保存(§2.7「g_i(0) をログする」)
    agent._fire_eta = float(params["eta"]) * float(sim.gcfg["eta_scale"])
    agent._fire_lam = float(params["lam"]) * float(sim.gcfg["lam_scale"])
    agent._fire_theta_m = float(params["theta0"]) if icfg["mode"] == PERSONA else 1.0
    agent._fire_ebar = {idx: 0.0 for idx in g0}
    agent._fire_credit = {}
    agent._fire_pending = []                       # 開いている credit 窓(適格性トレース)
    agent._fire_day_n = 0
    agent._fire_fbar = float(sim.gcfg["theta_target_per_day"])


def g_of(sim, agent) -> dict[int, float] | None:
    """S の重みとして使う g(OFF なら None = fire.py 側で 1.0 固定に後退)。"""
    if not enabled(sim):
        return None
    ensure(sim, agent)
    return agent._fire_g


def theta_mult(sim, agent) -> float:
    """θ の個体倍率(OFF なら 1.0 = 第81 と同一)。"""
    if not enabled(sim):
        return 1.0
    ensure(sim, agent)
    return float(agent._fire_theta_m)


# --------------------------------------------------------------------------- #
# 毎 tick: 慣れ ē の更新 + credit の回収
# --------------------------------------------------------------------------- #
def observe_tick(sim, agent, errors: dict[int, float], sim_min: int) -> None:
    """凍結観測から求めた σ 正規化誤差で ē を更新し、窓の来た credit を回収する。

    `errors` は `{観測タプル内の位置: |o−ô|/σ}`(**g を掛ける前**の生の偏差)。
    寄与が 0 のチャンネルも 0 として EMA に入れる(= 何も起きない時間が慣れを解く)。
    """
    if not enabled(sim):
        return
    ensure(sim, agent)
    alpha = ema_alpha(float(getattr(sim, "dt_min", 10)),
                      float(sim.gcfg["ebar_halflife_min"]))
    ebar = agent._fire_ebar
    for idx in ebar:
        ebar[idx] = float((1.0 - alpha) * ebar[idx] + alpha * float(errors.get(idx, 0.0)))
    _resolve_credit(sim, agent, sim_min)


def _outcome(agent) -> float:
    from ..factors import registry as _reg
    return float(_reg.outcome_scalar(getattr(agent, "states", None) or {}))


def _resolve_credit(sim, agent, sim_min: int) -> None:
    """窓が閉じた発火について r を確定し、チャンネル別 credit へ積む。

    ★窓は**複数同時に開いている**(発火の間隔 ≈ 10〜30 分 < 窓 60 分なので、1 本しか
      持たない実装だと次の発火が前の窓を上書きして **r がほぼ常に 0 になる**。実測で
      11,000 件の認知イベントに対して窓が閉じたのは 181 件しかなく、感作の項が事実上
      死んでいた)。適格性トレース(eligibility trace)の標準どおり、閉じていない窓は
      並行に保持して**それぞれ独立に**満期を迎えさせる。
    """
    pend = getattr(agent, "_fire_pending", None) or []
    if not pend:
        return
    window = int(sim.gcfg["r_window_min"])
    now = int(sim_min)
    credit = agent._fire_credit
    keep = []
    value = None
    for rec in pend:                               # 作成順 = 決定論
        if now - int(rec["at"]) < window:
            keep.append(rec)
            continue
        if value is None:
            value = _outcome(agent)                # 1 tick に 1 度だけ読む
        delta = (value - float(rec["base"])) * float(sim.gcfg["r_gain"])
        for idx, share in rec["shares"].items():
            credit[idx] = float(credit.get(idx, 0.0) + delta * float(share))
    agent._fire_pending = keep


# --------------------------------------------------------------------------- #
# 認知イベント: g の更新 + 次の credit 窓の開始
# --------------------------------------------------------------------------- #
def on_event(sim, agent, contrib_by_idx: dict[int, float], sim_min: int,
             reason: str = "") -> None:
    """認知イベント 1 回につき 1 度、更新式を適用する(Δt 非依存)。

    `contrib_by_idx` はその瞬間の S 寄与 `g·|o−ô|/σ`(チャンネル位置 → 値)。
    寄与が全部 0(= 予測どおりの世界)なら按分先が無いので credit 窓は開かない
    (**credit assignment は「S に効いたチャンネル」にしか流れない**)。

    `reason` が θ の門番する発火源(= 驚き)のときだけ日次カウンタを進める。
    """
    if not enabled(sim):
        return
    ensure(sim, agent)
    cfg = sim.gcfg
    credit = agent._fire_credit
    ebar = agent._fire_ebar
    g, g0 = agent._fire_g, agent._fire_g0
    eta, lam = float(agent._fire_eta), float(agent._fire_lam)
    for idx in sorted(g):                          # 走査順を辞書順に固定(決定論)
        g[idx] = g_step(g[idx], g0[idx], float(credit.get(idx, 0.0)),
                        float(ebar.get(idx, 0.0)),
                        eta=eta, rho=float(cfg["rho"]), lam=lam,
                        lo=float(cfg["g_min"]), hi=float(cfg["g_max"]))
    agent._fire_credit = {}                        # 消費した credit は空にする

    total = sum(v for v in contrib_by_idx.values() if v > 0.0)
    if total > 0.0:
        pend = list(getattr(agent, "_fire_pending", None) or [])
        pend.append({
            "at": int(sim_min), "base": _outcome(agent),
            "shares": {idx: float(v) / total
                       for idx, v in sorted(contrib_by_idx.items()) if v > 0.0},
        })
        # 有界にする(窓 ÷ 最短発火間隔 の上限 + 余裕。古いものから捨てる=決定論)。
        cap = int(cfg["max_pending"])
        agent._fire_pending = pend[-cap:] if cap > 0 else []
    from . import fire as _fire
    if reason == _fire.SALIENCE:               # θ の恒常性が制御できるのはここだけ
        agent._fire_day_n = int(getattr(agent, "_fire_day_n", 0)) + 1


# --------------------------------------------------------------------------- #
# 日境界: θ の恒常性(§2.6。**日オーダーの時定数**)
# --------------------------------------------------------------------------- #
def day_boundary(sim, step: int, sim_min: int) -> bool:
    """暦日が変わったら θ の恒常性を 1 回だけ適用する(変わっていなければ False)。"""
    if not enabled(sim):
        return False
    day = int(sim_min) // 1440
    if int(getattr(sim, "_g_day", -1)) == day:
        return False
    first = int(getattr(sim, "_g_day", -1)) < 0
    sim._g_day = day
    if first:
        return False                               # 初日は「前日の発火率」が存在しない
    cfg = sim.gcfg
    target = float(cfg["theta_target_per_day"])
    weight = float(cfg["fbar_weight"])
    for agent in sim.agents:
        if getattr(agent, "_fire_g", None) is None:
            continue
        today = float(getattr(agent, "_fire_day_n", 0))
        fbar = float(getattr(agent, "_fire_fbar", target))
        fbar = (1.0 - weight) * fbar + weight * today
        agent._fire_fbar = fbar
        agent._fire_day_n = 0
        agent._fire_theta_m = theta_step(
            float(agent._fire_theta_m), fbar, target,
            mu=float(cfg["theta_mu"]), lo=float(cfg["theta_min_mult"]),
            hi=float(cfg["theta_max_mult"]))
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=int(agent.id),
                             kind="cog_theta", x=agent.x, y=agent.y,
                             payload={"day": day, "fbar": round(fbar, 4),
                                      "fired": int(today),
                                      "theta_mult": round(agent._fire_theta_m, 5)}))
    return True


# --------------------------------------------------------------------------- #
# 軌跡サイドカー(§8「g ベクトルの時間発展 = 分散の拡大が役割分化の一次証拠」)
# --------------------------------------------------------------------------- #
def columns(sim) -> tuple[str, ...]:
    """サイドカーの値列名(g_<channel> … + θ 倍率 + 発火率 EMA)。"""
    from . import fire as _fire
    from . import channels as _channels
    cols = [f"g_{_channels.BY_ID[cid].column}" for _i, cid, _s in _fire.usable_channels(sim)]
    cols += [f"g0_{_channels.BY_ID[cid].column}"
             for _i, cid, _s in _fire.usable_channels(sim)]
    return tuple(cols + ["theta_mult", "fbar", "fired_today"])


def due(sim, step: int) -> bool:
    every = int(sim.gcfg["log_every_steps"]) if enabled(sim) else 0
    return bool(every) and int(step) % every == 0


def rows(sim, step: int, sim_min: int) -> list[tuple]:
    """1 step 分の g/θ 軌跡行(**読むだけ**。世界には書き戻さない)。"""
    from . import fire as _fire
    order = [i for i, _cid, _s in _fire.usable_channels(sim)]
    out: list[tuple] = []
    for agent in sim.agents:
        g = getattr(agent, "_fire_g", None)
        if g is None:
            continue
        g0 = agent._fire_g_init
        values = [float(g.get(i, 0.0)) for i in order]
        values += [float(g0.get(i, 0.0)) for i in order]
        values += [float(agent._fire_theta_m), float(agent._fire_fbar),
                   float(agent._fire_day_n)]
        out.append((int(step), int(sim_min), int(agent.id), *values))
    return out


# --------------------------------------------------------------------------- #
# 来歴(run_manifest.json 用)
# --------------------------------------------------------------------------- #
def provenance(sim) -> dict | None:
    """g/θ 更新則の宣言。OFF(既定)では None = 既存 manifest と同形。"""
    if not enabled(sim):
        return None
    cfg, icfg = sim.gcfg, sim.ginitcfg
    return {
        "schema": SCHEMA,
        "rule": "g <- g + eta*(r - rho*ebar) - lam*(g - g0)",
        "theta_rule": "theta_mult <- theta_mult + mu*(fbar - f_target)  [daily]",
        "eta_scale": cfg["eta_scale"], "lam_scale": cfg["lam_scale"],
        "rho": cfg["rho"], "ebar_halflife_min": cfg["ebar_halflife_min"],
        "r_window_min": cfg["r_window_min"], "r_gain": cfg["r_gain"],
        "max_pending": cfg["max_pending"],
        "g_bounds": [cfg["g_min"], cfg["g_max"]],
        "theta_mu": cfg["theta_mu"],
        "theta_target_per_day": cfg["theta_target_per_day"],
        "theta_update": "day_boundary_once(時定数は日オーダー=設計 §2.6)",
        # ★正直な宣言: r はゴール進捗そのものではなく state ベクトルの改善分の代理
        "r_source": "factors.outcome_scalar の差分(ゴール進捗の数値量は未実装)",
        "g_init": {"mode": icfg["mode"], "flat_value": icfg["flat_value"],
                   "sigma0": icfg["sigma0"], "stream": NOISE_STREAM,
                   "crn": "3 条件すべてで同じ本数を引く(使うのは noise だけ)"},
        "attention_contagion": "not_implemented(設計 §2.5 の将来拡張。§10 の未決のまま)",
    }
