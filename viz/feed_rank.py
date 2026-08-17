"""イベントフィードのランキング(純関数・乱数ゼロ・観測専用)。

正典: docs/research/emergent-events-and-narrative-ui.md §4-1 / §4-4 と
      docs/plans/expressiveness-and-events-plan.md レーン F1。

    score(e) = w₁·importance + w₂·rarity + w₃·magnitude_z + w₄·chain + w₅·first − dedup

各項の意味と「なぜその形か」:

importance  レジストリ(viz/notable_events.KIND_REGISTRY)の事前値 1..5 を [0,1] へ。
            人手が与える唯一の事前情報。

rarity      **自己較正**の希少度 = log₂(total/count(kind)) / log₂(total) ∈ (0,1]。
            memo §2-3 の実測が示すとおり kind 頻度は「配管:物語 = 1000:1」で歪む
            (transmission 5,453 件 vs council_elected 1 件)。固定表では 60 体スモークと
            25 万本選で意味が変わってしまうので、**そのラン自身の分布**で較正する。
            希少度は 1 kind 1 値(件数の逆数)なので、同じ importance なら
            「めったに起きない kind」が必ず上に来る。

magnitude_z レジストリ mag(reach / supporters / amount / debt …)の **kind 内 z 値**。
            kind をまたいで生の値を比べると単位が違う(円 vs 人)ので必ず kind 内で正規化。
            σ=0(全部同じ値)や mag 無しの kind は 0。負の側は切り上げ(小さい方は減点しない)。

chain       連鎖長・影響半径の代理。payload の数値(reach / supporters / hop / n / knowers …)を
            log で潰して [0,1]。ライブでは系譜を辿れないので代理量にする(memo §4-1)。

first       ラン内初出現ボーナス。「最初の partner_formed は 2 件目より物語価値が高い」。
            kind 初出 = 満点、storyline(語・ペア・組織)初出 = 部分点。出現済み集合を
            持つだけの決定論。

dedup       ストーリー折りたたみ。同じ storyline 鍵(item_id / event_id / ペア / org …)が
            既に何件出たかで減点。notable_events の PER_KIND_CAP(kind 単位の間引き)より
            物語の連続性が保たれる(同じ話題の続報だけが薄くなる)。

ペーシング(L4D 式)は **表示制御**であってスコアではない。掲載枠は「上位 keep_ratio 位」の
順位の枠として与え、大事件が連発した直後だけ **大事件の枠を一時的に狭める**(= 見えている
列の構成比が日常イベント側へ寄る)。静かな時間帯は枠が満枠へ戻る。緊張と緩和の演出は
ビューアの仕事であり、シムには 1 バイトも触れない(詳細は pace() の docstring)。

全て決定論(同じ入力 → 同じ出力・乱数ゼロ・時刻依存ゼロ)。
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path


def _load_registry():
    """viz/notable_events.py を場所非依存で読む(make_viewer3d._load_notable と同じ流儀)。"""
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "notable_events_registry", here / "notable_events.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


NB = _load_registry()

# ------------------------------------------------------------------ 既定パラメータ
# 重み(memo §4-1 の w₁..w₅)。合計 3.0 に揃えてあるので score は概ね [0, 3] に収まる。
DEFAULT_WEIGHTS: dict[str, float] = {
    "importance": 1.0,
    "rarity":     1.0,
    "magnitude":  0.45,
    "chain":      0.35,
    "first":      0.20,
    "dedup":      1.0,
}

# chain の代理に使う payload 数値キー(先頭優先)。
CHAIN_KEYS = ("reach", "supporters", "knowers", "hop", "n", "n_attendees",
              "members", "adopted_n", "votes", "yes")
CHAIN_SCALE = 60.0                 # log1p(v)/log1p(CHAIN_SCALE) を 1 で頭打ち

# dedup: 同一 storyline の 1 件目は 0、以降 1 件ごとに減点(上限つき)。
DEDUP_STEP = 0.34
DEDUP_MAX = 1.35

# ペーシング(L4D 式の「強度推定」)。掲載枠は**絶対スコア閾値でなく順位の枠**にする:
# ランごとにスコア分布の形が違い(60 体スモークと 25 万本選)、しかも同 kind の大量
# イベントは dedup 上限でスコアが 1 点に潰れて団子になる。閾値方式だと団子が雪崩れて
# 掲載数が制御不能になる(実測で 3 倍に跳ねた)。詳細は pace() の docstring。
PACE_KEEP_RATIO = 0.40             # 静かな時に通す割合(= 順位の枠)
PACE_GAIN = 0.55                   # 強度が「大事件の枠」を狭める強さ
PACE_HALF_LIFE_STEPS = 18.0        # 強度の半減期(Δt=10 なら 3 時間)
PACE_ALWAYS_IMPORTANCE = 5         # この重要度以上は無条件掲載(memo §4-4 の S 層)
PACE_BIG_IMPORTANCE = 4            # これ以上を「大事件」= 連発時に枠が狭まる側

DEFAULT_MAX_EVENTS = 1200          # フィード全体の上限(超過分は caps に記録)


# ------------------------------------------------------------------ 内部ヘルパ
def _num(v) -> float | None:
    if isinstance(v, bool) or v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and abs(f) != math.inf else None


def _chain_value(payload: dict) -> float:
    for k in CHAIN_KEYS:
        f = _num(payload.get(k))
        if f is not None and f > 0:
            return f
    for k in CHAIN_KEYS:                       # list 値(members/knowers 等)は長さを使う
        v = payload.get(k)
        if isinstance(v, list) and v:
            return float(len(v))
    return 0.0


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


# ------------------------------------------------------------------ 1) 素材化
def collect(events, *, step_minutes: int = 10, start_min: int = 420) -> list[dict]:
    """L1 行の列 → レジストリ対象 kind だけの素材 dict 列(時刻順・決定論)。

    events: {step, sim_min, agent_id, kind, x, y, payload(JSON文字列 or dict)} の列。
    """
    out: list[dict] = []
    for e in events:
        kind = e.get("kind")
        spec = NB.KIND_REGISTRY.get(kind)
        if spec is None:
            continue
        payload = NB._as_payload(e.get("payload"))
        try:
            x = round(float(e.get("x", 0.0) or 0.0), 1)
            y = round(float(e.get("y", 0.0) or 0.0), 1)
        except (TypeError, ValueError):
            x = y = 0.0
        step = int(e.get("step", 0) or 0)
        sm = e.get("sim_min")
        sm = int(sm) if sm is not None else start_min + step * step_minutes
        aid = int(e.get("agent_id", -1))
        out.append({
            "step": step,
            "sim_min": sm,
            "kind": kind,
            "label": spec["label"],
            "icon": spec["icon"],
            "importance": spec["importance"],
            "agent_id": aid,
            "x": x,
            "y": y,
            "has_pos": not (x == 0.0 and y == 0.0),
            "text": NB.summarize(kind, payload),
            "story": NB.storyline_key(kind, aid, payload),
            "_mag": NB.feed_magnitude(kind, payload),
            "_chain": _chain_value(payload),
        })
    out.sort(key=lambda r: (r["step"], -r["importance"], r["kind"], r["agent_id"]))
    return out


# ------------------------------------------------------------------ 2) スコア
def score_events(recs: list[dict], *, weights: dict | None = None) -> list[dict]:
    """素材列にスコア(と内訳)を書き込んで返す。**ラン自身の分布で自己較正**する。"""
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)
    total = len(recs)
    if total == 0:
        return []

    # --- 希少度: kind 件数 → log₂(total/count)/log₂(total) ∈ (0,1] ---
    counts: dict[str, int] = {}
    for r in recs:
        counts[r["kind"]] = counts.get(r["kind"], 0) + 1
    log_total = math.log2(total) if total > 1 else 0.0
    rarity: dict[str, float] = {}
    for k, c in counts.items():
        raw = math.log2(total / c)
        rarity[k] = _clamp01(raw / log_total) if log_total > 0 else 0.0

    # --- magnitude: kind 内の平均/標準偏差(母集団 σ) ---
    mags: dict[str, list[float]] = {}
    for r in recs:
        if r["_mag"] is not None:
            mags.setdefault(r["kind"], []).append(r["_mag"])
    stat: dict[str, tuple[float, float]] = {}
    for k, vals in mags.items():
        mu = sum(vals) / len(vals)
        var = sum((v - mu) ** 2 for v in vals) / len(vals)
        stat[k] = (mu, math.sqrt(var))

    seen_kind: set[str] = set()
    seen_story: set[str] = set()
    story_n: dict[str, int] = {}
    chain_den = math.log1p(CHAIN_SCALE)

    for r in recs:
        kind = r["kind"]
        imp_n = (r["importance"] - 1) / 4.0
        rar_n = rarity.get(kind, 0.0)
        mag_n = 0.0
        if r["_mag"] is not None and kind in stat:
            mu, sd = stat[kind]
            if sd > 0:
                mag_n = _clamp01((r["_mag"] - mu) / (3.0 * sd))
        chain_n = _clamp01(math.log1p(max(0.0, r["_chain"])) / chain_den)
        story = r["story"]
        first_n = 0.0
        if kind not in seen_kind:
            first_n = 1.0
        elif story is not None and story not in seen_story:
            first_n = 0.6
        prior = story_n.get(story, 0) if story is not None else 0
        dedup_n = min(DEDUP_MAX, DEDUP_STEP * prior)

        r["score"] = round(
            w["importance"] * imp_n + w["rarity"] * rar_n
            + w["magnitude"] * mag_n + w["chain"] * chain_n
            + w["first"] * first_n - w["dedup"] * dedup_n, 6)
        r["parts"] = {"imp": round(imp_n, 4), "rar": round(rar_n, 4),
                      "mag": round(mag_n, 4), "chn": round(chain_n, 4),
                      "fst": round(first_n, 4), "ded": round(-dedup_n, 4)}
        r["dup"] = prior

        seen_kind.add(kind)
        if story is not None:
            seen_story.add(story)
            story_n[story] = prior + 1
    return recs


# ------------------------------------------------------------------ 3) ペーシング
def pace(recs: list[dict], *, keep_ratio: float = PACE_KEEP_RATIO,
         gain: float = PACE_GAIN, half_life: float = PACE_HALF_LIFE_STEPS,
         budget: float | None = None,
         always_importance: int = PACE_ALWAYS_IMPORTANCE,
         big_importance: int = PACE_BIG_IMPORTANCE) -> tuple[list[dict], list[dict]]:
    """時刻順に走査し、掲載/非掲載へ振り分ける(L4D 式の緊張と緩和・決定論)。

    仕組みは 2 点:

    ① 掲載枠は絶対スコア閾値でなく **順位の枠**(上位 keep_ratio 位まで)。
       スコア閾値を採らなかったのは実測上の理由がある: 同じ kind の大量イベントは
       dedup の上限で **スコアが 1 点に潰れて団子になる**(free_action 413 件、
       relation_tier 200 件…)。閾値をわずかに動かすだけで団子が丸ごと雪崩れ込み、
       掲載数が 3 倍に跳ねて制御にならない。順位の枠なら団子があっても掲載数が決まる
       (同点は「早い方=より初出に近い方」を優先する決定論のタイブレーク)。

    ② 強度が上がると **大事件(importance ≥ big_importance)の枠だけが狭まる**。
       日常側の枠は動かさないので、賑やかな直後は「大事件が入りにくく、日常はそのまま」
       = 見えている列の構成比が日常側へ寄る(plan レーン F1「大事件連発後は掲載閾値を
       一時上げ日常イベントを混ぜる」)。日常側の枠も**広げる**対称版は、上と同じ団子の
       せいで暴れるため採らない。

    強度の増分は **そのランで期待される掲載レート**で正規化する(budget = 半減期の
    あいだに載る想定本数)。期待レートちょうどで走ると強度は 0.5 前後に落ち着き、
    ラン規模(3 日 60 体 / 30 日 25 万)に依らず同じ意味で効く。

    シムには触れない**表示制御**であり、スコアそのものは書き換えない。
    """
    if not recs:
        return [], []
    hi = max(r["score"] for r in recs)
    lo = min(r["score"] for r in recs)
    span = (hi - lo) or 1.0
    for r in recs:
        r["_norm"] = (r["score"] - lo) / span
    ratio = max(0.0, min(1.0, keep_ratio))
    # 順位(同点は step 昇順 → kind → agent_id の決定論タイブレーク)
    for i, r in enumerate(sorted(recs, key=lambda x: (-x["score"], x["step"],
                                                      x["kind"], x["agent_id"]))):
        r["_rank"] = i
    n_slots = int(math.ceil(ratio * len(recs)))
    n_steps = max(r["step"] for r in recs) + 1
    if budget is None:                               # 半減期のあいだに載る想定本数
        budget = max(2.0, ratio * len(recs) * half_life / max(1, n_steps))
    keep: list[dict] = []
    drop: list[dict] = []
    intensity = 0.0
    last_step = None
    for r in sorted(recs, key=lambda x: (x["step"], -x["score"], x["kind"])):
        if last_step is not None and r["step"] > last_step and half_life > 0:
            intensity *= 0.5 ** ((r["step"] - last_step) / half_life)
        last_step = r["step"]
        # 大事件は連発すると枠が狭まる。日常は基準のまま(= 構成比が日常側へ寄る)。
        slots = n_slots * (1.0 - gain * intensity) \
            if r["importance"] >= big_importance else float(n_slots)
        r["pace"] = round(slots, 3)
        if r["importance"] >= always_importance or r["_rank"] < slots:
            keep.append(r)
            if budget > 0:
                intensity = min(1.0, intensity + r["_norm"] / budget)
        else:
            drop.append(r)
    return keep, drop


# ------------------------------------------------------------------ 4) 入口
def build_feed(events, *, step_minutes: int = 10, start_min: int = 420,
               weights: dict | None = None,
               max_events: int = DEFAULT_MAX_EVENTS,
               pacing: bool = True) -> dict:
    """L1 行の列 → ビューアへ渡すフィード dict(決定論)。

    戻り値:
      kinds   {kind: {label, icon, imp}}  出現した kind の表示辞書
      events  掲載イベント(時刻順)。各行 {s:step, m:sim_min, k:kind, a:agent_id,
              x,y:位置, p:位置あり, t:要約, i:重要度, sc:スコア, g:storyline鍵, n:折りたたみ件数}
      caps    {"paced": 非掲載件数, "capped": 上限で落ちた件数, "n_total": 素材総数}
      stats   {"n_total","n_kept","kind_counts","rarity"}
    """
    recs = collect(events, step_minutes=step_minutes, start_min=start_min)
    n_total = len(recs)
    if n_total == 0:
        return {"kinds": {}, "events": [], "caps": {"paced": 0, "capped": 0, "n_total": 0},
                "stats": {"n_total": 0, "n_kept": 0, "kind_counts": {}, "rarity": {}}}
    score_events(recs, weights=weights)
    if pacing:
        # 掲載の目標本数を上限に合わせて自己較正する(= 大きなランほど順位の枠を狭める)。
        # そうしないと「上限で機械的に切る」段で時間方向の広がりが壊れる(高スコア kind に
        # 偏った塊だけが残る)。ペーシングの段で絞れば時系列の疎密は保たれる。
        ratio = PACE_KEEP_RATIO
        if max_events:
            ratio = min(ratio, max_events / float(n_total))
        keep, drop = pace(recs, keep_ratio=ratio)
    else:
        keep, drop = list(recs), []

    # 上限(silent cap 禁止: 落ちた件数は caps に必ず出す)。スコア降順で残す。
    capped = 0
    if max_events and len(keep) > max_events:
        keep_sorted = sorted(keep, key=lambda r: (-r["score"], r["step"], r["kind"]))
        cut = keep_sorted[max_events:]
        keep = sorted(keep_sorted[:max_events],
                      key=lambda r: (r["step"], -r["score"], r["kind"]))
        drop = drop + cut
        capped = len(cut)
    else:
        keep = sorted(keep, key=lambda r: (r["step"], -r["score"], r["kind"]))

    # 折りたたみバッジ: 落ちた行は「同じ storyline で最後に掲載された行」に件数を足す
    # (「+12件」= この話題であと何件起きたか)。1 物語 1 カード + 件数バッジ(memo §4-1)。
    fold: dict[str, dict] = {}
    for r in keep:
        r["fold"] = 0
        if r["story"] is not None:
            fold[r["story"]] = r                     # 同 storyline の最後の掲載行
    for r in sorted(drop, key=lambda x: (x["step"], x["kind"])):
        tgt = fold.get(r["story"]) if r["story"] is not None else None
        if tgt is not None:
            tgt["fold"] += 1

    kinds = {}
    counts: dict[str, int] = {}
    for r in recs:
        counts[r["kind"]] = counts.get(r["kind"], 0) + 1
    for r in keep:
        kinds.setdefault(r["kind"], {"label": r["label"], "icon": r["icon"],
                                     "imp": r["importance"]})
    out_events = [{
        "s": r["step"], "m": r["sim_min"], "k": r["kind"], "a": r["agent_id"],
        "x": r["x"], "y": r["y"], "p": bool(r["has_pos"]), "t": r["text"],
        "i": r["importance"], "sc": round(r["score"], 4),
        "g": r["story"], "n": r["fold"],
    } for r in keep]
    return {
        "kinds": kinds,
        "events": out_events,
        "caps": {"paced": len(drop) - capped, "capped": capped, "n_total": n_total},
        "stats": {"n_total": n_total, "n_kept": len(out_events),
                  "kind_counts": dict(sorted(counts.items())),
                  "rarity": {k: round(math.log2(n_total / c)
                                      / (math.log2(n_total) if n_total > 1 else 1.0), 4)
                             for k, c in sorted(counts.items())}},
    }
