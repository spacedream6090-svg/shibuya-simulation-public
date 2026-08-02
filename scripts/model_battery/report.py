#!/usr/bin/env python
"""レポート生成 — モデル×層プロファイル表 + プラセボ健全性判定(第90バッチ)。

使い方:
    python scripts/model_battery/report.py --raw data/battery/raw \\
        --reference data/battery/reference --out data/battery \\
        [--d-ratio-floor 0.6]

思想(合否表ではなく**役割決めの地図**):
  - 層スコアは「人間らしさの総合点」ではない。各層の下位指標を平均しただけの
    **相対比較用の要約**であり、下位指標を必ず併記する(隠さない)。
  - オラクル(人間参照統計)が無い下位指標は**絶対評価しない**。参照が無ければ
    その項目を落として平均する(欠測は "ref_missing" として明示)。
  - D 層(分散と裾=生命線)の判定線 `--d-ratio-floor` は**引数必須**。
    未指定なら分散比を出すだけで合否は出さない(事後に閾値をいじれないようにする)。
  - プラセボが全層で最下位に沈まなければ、その層の指標が何も測れていないという
    ことなので、健全性 FAIL として正直に出す。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                              # noqa: BLE001
        pass

from model_battery import metrics as M                          # noqa: E402
from model_battery import reference as R                         # noqa: E402
from model_battery import stimuli as S                           # noqa: E402

REPORT_SCHEMA = "model_battery.report/1"
REPO_ROOT = _HERE.parents[1]
SLOT_MIN = 15
N_SLOTS = 1440 // SLOT_MIN

# 参照統計の id(data/battery/reference/*.json の "id")
REF_TIME_USE = "estat_shakai_seikatsu_time_use"
REF_DIALOGUE = "jp_conversation_corpus_descriptive"


# ---------------------------------------------------------------- 読み込み
def load_raw(raw_dir: Path) -> dict[str, dict]:
    """data/battery/raw/<slug>/ を読む。slug -> {manifest, tests:{test:[rec]}}"""
    out: dict[str, dict] = {}
    if not raw_dir.is_dir():
        return out
    for d in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        man_path = d / "manifest.json"
        if not man_path.exists():
            continue
        man = json.loads(man_path.read_text(encoding="utf-8"))
        tests: dict[str, list[dict]] = {}
        for jl in sorted(d.glob("*.jsonl")):
            recs = []
            for line in jl.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    recs.append(json.loads(line))
            tests[jl.stem] = recs
        out[d.name] = {"manifest": man, "tests": tests, "dir": str(d)}
    return out


def _ok(recs):
    return [r for r in recs if not r.get("error")]


def _mean(xs, default=None):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else default


# ---------------------------------------------------------------- A 層
HOURLY_KEY = "hourly_activity_rate_weekday"
BUDGET_KEY = "week_average_minutes"


def hourly_jsd(parsed: dict, ref: dict, daytype: str) -> float | None:
    """時刻別活動分布の JSD(参照 = 社会生活基本調査 第17-1表 の時間帯別行動者率)。

    参照表は20分類のうち一部の行動しか転記していないので、**表外は残余1ビン**に
    まとめて比較する(モデル側も同じ畳み方をする=公平)。未計画スロットは母数外。
    """
    s = (ref.get("series") or {}).get(HOURLY_KEY)
    if not isinstance(s, dict) or not s.get("values"):
        return None
    cats = list(s.get("categories") or [])
    mat = s["values"]
    if len(mat) != 24 or not cats:
        return None
    slots_list = [p["slots"] for p in parsed.values() if p["daytype"] == daytype]
    if not slots_list:
        return None
    per_hour = N_SLOTS // 24
    vals = []
    for h in range(24):
        counts = [0.0] * (len(cats) + 1)
        for slots in slots_list:
            for k in range(per_hour):
                a = slots[h * per_hour + k]
                if a is None:
                    continue
                counts[cats.index(a) if a in cats else len(cats)] += 1.0
        if sum(counts) <= 0:
            continue
        row = [float(x) for x in mat[h]]
        row.append(max(0.0, 100.0 - sum(row)))       # 表外の行動(残余)
        vals.append(M.jensen_shannon(counts, row))
    return sum(vals) / len(vals) if vals else None


def budget_jsd(parsed: dict, ref: dict) -> float | None:
    """1日の時間配分(分)の JSD。参照は週全体平均なので全曜日をまとめて比較する。

    参照ベクトルは20分類のうち公表要約表に載っている項目だけを持ち、残りは null。
    null の位置はモデル側も含めて**残余1ビン**に畳む(捏造で埋めない)。
    """
    vals = R.series_values(ref, BUDGET_KEY)
    if not vals or len(vals) != len(M.ACTIVITY_CATEGORIES):
        return None
    known = [i for i, v in enumerate(vals) if v is not None]
    if not known:
        return None
    budgets = [p["budget"] for p in parsed.values()]
    if not budgets:
        return None
    agg = [sum(b[i] for b in budgets) / len(budgets)
           for i in range(len(M.ACTIVITY_CATEGORIES))]
    ref_vec = [float(vals[i]) for i in known]
    ref_vec.append(max(0.0, 1440.0 - sum(ref_vec)))
    mod_vec = [agg[i] for i in known]
    mod_vec.append(max(0.0, sum(agg) - sum(mod_vec)))
    if sum(mod_vec) <= 0:
        return None
    return M.jensen_shannon(mod_vec, ref_vec)


def score_a(recs: list[dict], ref: dict | None) -> dict:
    """A層=個体行動。1日スケジュールの妥当性・ペルソナ/曜日感度・参照統計との JSD。"""
    good = _ok(recs)
    sub: dict = {"n": len(recs), "errors": len(recs) - len(good)}
    if not good:
        return {**sub, "score": None, "note": "有効応答なし"}

    parsed: dict[str, dict] = {}
    n_schema_ok = 0
    labeled_num, labeled_den = 0, 0
    for r in good:
        obj = M.extract_json(r.get("response", ""))
        blocks = (obj or {}).get("blocks")
        if not isinstance(blocks, list) or not blocks:
            continue
        blocks = [b for b in blocks if isinstance(b, dict)]
        # ★activity キーを落として start/end だけ返すモデルが実在する(実測)。
        # 未ラベルを黙って "その他" に畳むと被覆率も JSD も嘘になるので、
        # ラベル率を独立の観測量として出し、半分未満なら解釈不能として弾く。
        labeled = sum(1 for b in blocks if str(b.get("activity", "")).strip())
        labeled_den += len(blocks)
        labeled_num += labeled
        if not blocks or labeled / len(blocks) < 0.5:
            continue
        slots, conflicts = M.schedule_to_slots(blocks, slot_minutes=SLOT_MIN)
        if M.slot_coverage(slots) <= 0.0:
            continue
        n_schema_ok += 1
        parsed[r["case_id"]] = {
            "slots": slots, "conflicts": conflicts,
            "coverage": M.slot_coverage(slots),
            "budget": M.time_budget(slots, slot_minutes=SLOT_MIN),
            "wake": M.wake_time(slots, slot_minutes=SLOT_MIN),
            "sleep": M.sleep_time(slots, slot_minutes=SLOT_MIN),
            "commute": M.first_activity_time(slots, "通勤・通学",
                                             slot_minutes=SLOT_MIN),
            "persona": r["meta"].get("persona"),
            "daytype": r["meta"].get("daytype"),
        }
    sub["schema_ok"] = n_schema_ok / len(good)
    sub["activity_labeled"] = (labeled_num / labeled_den) if labeled_den else 0.0
    if not parsed:
        return {**sub, "score": None,
                "note": "スケジュールを1件も解釈できない"
                        f"(ブロックの行動ラベル率 {sub['activity_labeled']:.2f})"}

    sub["coverage"] = _mean([p["coverage"] for p in parsed.values()], 0.0)
    sub["conflict_free"] = M.clamp01(
        1.0 - _mean([p["conflicts"] / N_SLOTS for p in parsed.values()], 0.0))

    # ペルソナ感度: 同じ曜日種別で、ペルソナ間にスロットの差がどれだけあるか
    def _slot_diff(a, b) -> float:
        n = min(len(a), len(b))
        return sum(1 for i in range(n) if a[i] != b[i]) / n if n else 0.0

    per_day: dict[str, list] = {}
    per_persona: dict[str, dict] = {}
    for p in parsed.values():
        per_day.setdefault(p["daytype"], []).append(p["slots"])
        per_persona.setdefault(p["persona"], {})[p["daytype"]] = p["slots"]
    psens = []
    for slots_list in per_day.values():
        for i in range(len(slots_list)):
            for j in range(i + 1, len(slots_list)):
                psens.append(_slot_diff(slots_list[i], slots_list[j]))
    sub["persona_sensitivity"] = _mean(psens, 0.0)
    dsens = [_slot_diff(v["平日"], v["休日"]) for v in per_persona.values()
             if "平日" in v and "休日" in v]
    sub["daytype_sensitivity"] = _mean(dsens, 0.0)

    # 参照統計との比較(無ければ絶対評価しない)
    comps = [sub["schema_ok"], sub["activity_labeled"], sub["coverage"],
             sub["conflict_free"], sub["persona_sensitivity"],
             sub["daytype_sensitivity"]]
    sub["ref_used"] = False
    if ref is not None:
        hj = hourly_jsd(parsed, ref, "平日")
        if hj is not None:
            sub["hourly_jsd_weekday"] = hj      # ★A層の本命=時刻別活動分布の JSD
            comps.append(1.0 - hj)
            sub["ref_used"] = True
        bj = budget_jsd(parsed, ref)
        if bj is not None:
            sub["budget_jsd_week"] = bj
            comps.append(1.0 - bj)
            sub["ref_used"] = True
        # 起床/就寝/通勤の分位差(分)。参照は点推定(平均)なので分位ごとの差を並べる。
        gaps = []
        for name, key in (("wake", "wake_time"), ("sleep", "sleep_time"),
                          ("commute", "commute_start")):
            rv = R.series_values(ref, key)
            if not isinstance(rv, dict):
                continue
            for daytype, spec in rv.items():
                point = (spec or {}).get("mean")
                if point is None:
                    continue
                vals = [p[name] for p in parsed.values()
                        if p["daytype"] == daytype and p[name] is not None]
                if not vals:
                    continue
                g = M.quantile_gap(vals, [point] * 3)
                sub[f"{name}_gap_{daytype}"] = {k: round(v, 1)
                                                for k, v in g.items()}
                gaps.append(g["mean_abs"])
        if gaps:
            # 120分ずれで 0 点、一致で 1 点(較正対象の粗い写像であることを明記)
            sub["timing_fit"] = M.clamp01(1.0 - _mean(gaps, 0.0) / 120.0)
            comps.append(sub["timing_fit"])
            sub["ref_used"] = True
    if not sub["ref_used"]:
        sub["note"] = "参照統計なし: 絶対評価(JSD/分位差)は未算出=相対比較のみ"
    sub["score"] = _mean(comps, None)
    return sub


# ---------------------------------------------------------------- B 層
def _b_direction_ok(check: str, obj: dict) -> bool | None:
    """摂動ごとの方向妥当性。invite は「正しい向き」が無いので None(母数外)。"""
    change = bool(obj.get("change"))
    act = M.normalize_activity(obj.get("new_activity", ""))
    try:
        delta = float(obj.get("delta_minutes", 0) or 0)
    except (TypeError, ValueError):
        delta = 0.0
    if check == "rain":
        return change and act in M.INDOOR_CATEGORIES
    if check == "delay":
        return change and 10.0 <= abs(delta) <= 180.0
    if check == "closure":
        return change
    if check == "invite":
        return None
    return None


def score_b(recs: list[dict]) -> dict:
    good = _ok(recs)
    sub: dict = {"n": len(recs), "errors": len(recs) - len(good)}
    if not good:
        return {**sub, "score": None, "note": "有効応答なし"}
    parsed = []
    for r in good:
        obj = M.extract_json(r.get("response", ""))
        if not isinstance(obj, dict) or "change" not in obj:
            continue
        parsed.append({"obj": obj, "meta": r["meta"], "case": r["case_id"]})
    sub["schema_ok"] = len(parsed) / len(good)
    if not parsed:
        return {**sub, "score": None, "note": "摂動応答を1件も解釈できない"}

    checks = [(p, _b_direction_ok(p["meta"].get("check", ""), p["obj"]))
              for p in parsed]
    checkable = [v for _, v in checks if v is not None]
    sub["direction_ok"] = (sum(1 for v in checkable if v) / len(checkable)
                           if checkable else None)
    sub["direction_n"] = len(checkable)
    per_pert: dict[str, list] = {}
    for p in parsed:
        per_pert.setdefault(p["meta"].get("perturbation", "?"), []).append(p)
    sub["by_perturbation"] = {}
    spreads = []
    for kid, items in sorted(per_pert.items()):
        sigs = {(bool(i["obj"].get("change")),
                 M.normalize_activity(i["obj"].get("new_activity", "")))
                for i in items}
        spread = (len(sigs) - 1) / (len(items) - 1) if len(items) > 1 else 0.0
        spreads.append(spread)
        vals = [_b_direction_ok(i["meta"].get("check", ""), i["obj"])
                for i in items]
        vals = [v for v in vals if v is not None]
        sub["by_perturbation"][kid] = {
            "n": len(items),
            "change_rate": sum(1 for i in items
                               if i["obj"].get("change")) / len(items),
            "distinct_responses": len(sigs),
            "direction_ok": (sum(1 for v in vals if v) / len(vals)
                             if vals else None),
        }
    sub["persona_spread"] = _mean(spreads, 0.0)
    inv = sub["by_perturbation"].get("invite")
    if inv:
        # 迎合(sycophancy)診断: 誘いに常に乗る/常に断るなら 1.0 or 0.0
        sub["invite_change_rate"] = inv["change_rate"]
    comps = [sub["schema_ok"], sub["persona_spread"]]
    if sub["direction_ok"] is not None:
        comps.append(sub["direction_ok"])
    sub["score"] = _mean(comps, None)
    return sub


# ---------------------------------------------------------------- C 層
def _says(recs: list[dict]) -> tuple[list[str], float]:
    """{"say": ...} 封筒から発話本文を取り出す。返り値 = (本文列, スキーマ遵守率)。

    封筒が壊れていた応答は**捨てる**(生 JSON を発話として数えると語彙統計が汚れる)。
    捨てた分は schema_ok に現れる。
    """
    ok = 0
    out: list[str] = []
    for r in recs:
        raw = r.get("response", "")
        obj = M.extract_json(raw)
        v = obj.get(S.SAY_KEY) if isinstance(obj, dict) else None
        if isinstance(v, str) and v.strip():
            ok += 1
            out.append(M.norm_text(v))
    return out, (ok / len(recs) if recs else 0.0)


def score_c(recs: list[dict], ref: dict | None) -> dict:
    good = _ok(recs)
    sub: dict = {"n": len(recs), "errors": len(recs) - len(good)}
    utt, schema_ok = _says(good)
    sub["schema_ok"] = schema_ok
    sub["utterances"] = len(utt)
    if len(utt) < 2:
        return {**sub, "score": None, "note": "発話が足りない"}

    lens = M.utterance_lengths(utt)
    sub["len_mean"] = _mean(lens, 0.0)
    sub["len_median"] = statistics.median(lens)
    sub["len_p10"] = M.quantile(lens, 0.10)
    sub["len_p90"] = M.quantile(lens, 0.90)
    sub["len_cv"] = M.cv(lens)
    sub["backchannel_rate"] = M.backchannel_rate(utt)
    sub["disagreement_rate"] = M.disagreement_rate(utt)
    sub["topic_persistence"] = M.topic_persistence(utt)
    sub["vocab_entropy"] = M.vocab_entropy(utt)
    sub["vocab_richness"] = M.distinct_n(utt, 2)
    sub["non_repetition"] = 1.0 - M.repetition_rate(utt, 4)
    sub["len_dispersion"] = M.clamp01(sub["len_cv"])
    # ★妥当性ゲート: 日本語で発話できていないなら「会話統計」は会話を測っていない。
    # 思考モデルが英語の前置きを本文に書くと発話長・語彙多様度が見かけ上よくなる
    # (実測で踏んだ)。平均に足すのではなく**係数**にして、他の指標の意味を殺す。
    sub["latin_ratio"] = M.latin_ratio(utt)
    sub["japanese_ratio"] = 1.0 - sub["latin_ratio"]

    comps = [sub["schema_ok"], sub["len_dispersion"], sub["vocab_richness"],
             sub["non_repetition"]]
    sub["ref_used"] = False
    if ref is not None:
        for key, mine, tol in (("mean_utterance_chars", sub["len_mean"], None),
                               ("backchannel_rate", sub["backchannel_rate"], 1.0),
                               ("disagreement_rate", sub["disagreement_rate"], 1.0)):
            rv = R.series_values(ref, key)
            if rv is None:
                continue
            rv = float(rv)
            denom = float(tol) if tol else max(abs(rv), 1e-9)
            fit = M.clamp01(1.0 - abs(mine - rv) / denom)
            sub[f"{key}_ref"] = rv
            sub[f"{key}_fit"] = fit
            comps.append(fit)
            sub["ref_used"] = True
    if not sub["ref_used"]:
        sub["note"] = "会話コーパスの参照統計なし: 絶対比較は未算出=相対比較のみ"
    base = _mean(comps, None)
    sub["score_before_gate"] = base
    sub["score"] = None if base is None else base * sub["japanese_ratio"]
    return sub


# ---------------------------------------------------------------- D 層
def score_d(recs: list[dict]) -> dict:
    """D層=分散と裾。生命線。同一プロンプト×n サンプルの散らばりだけを測る。"""
    good = _ok(recs)
    sub: dict = {"n": len(recs), "errors": len(recs) - len(good)}
    choices, whys = [], []
    for r in good:
        obj = M.extract_json(r.get("response", ""))
        if not isinstance(obj, dict):
            continue
        ch = M.norm_text(str(obj.get("choice", "")))
        if not ch:
            continue
        # 表記ゆれは正典の選択肢に部分一致で寄せる(寄らなければ生の文字列のまま)
        for cand in S.D_CHOICES:
            if cand in ch or ch in cand:
                ch = cand
                break
        choices.append(ch)
        whys.append(M.norm_text(str(obj.get("why", ""))))
    sub["parsed"] = len(choices)
    sub["schema_ok"] = len(choices) / len(good) if good else 0.0
    if len(choices) < 2:
        return {**sub, "score": None, "note": "サンプルが足りない"}

    counts_map: dict[str, int] = {}
    for c in choices:
        counts_map[c] = counts_map.get(c, 0) + 1
    k = len(S.D_CHOICES)
    counts = [counts_map.get(c, 0) for c in S.D_CHOICES]
    extra = sum(v for c, v in counts_map.items() if c not in S.D_CHOICES)
    if extra:
        counts.append(extra)
        k += 1
    sub["choice_counts"] = dict(sorted(counts_map.items()))
    sub["distinct_choices"] = len(counts_map)
    sub["choice_entropy"] = M.normalized_entropy(counts, k=k)
    gini_max = 1.0 - 1.0 / k
    sub["choice_gini"] = M.gini_simpson(counts)
    sub["choice_gini_norm"] = (sub["choice_gini"] / gini_max
                               if gini_max > 0 else 0.0)
    sub["lexical_diversity"] = M.mean_pairwise_distance(whys, 2)
    sub["score"] = _mean([sub["choice_entropy"], sub["choice_gini_norm"],
                          sub["lexical_diversity"]], None)
    return sub


# ---------------------------------------------------------------- E 層
def score_e(mono: list[dict], plan: list[dict]) -> dict:
    sub: dict = {"n_mono": len(mono), "n_plan": len(plan)}
    texts, schema_ok = _says(_ok(mono))
    sub["schema_ok"] = schema_ok
    sub["mono_used"] = len(texts)
    comps = [schema_ok]
    if len(texts) >= 2:
        sub["novelty"] = M.distinct_n(texts, 4)
        wins = M.windows(texts, 3)
        if len(wins) >= 2:
            first, last = wins[0], wins[-1]
            sub["late_novelty"] = M.distinct_n(last, 4)
            h0, h1 = M.vocab_entropy(first), M.vocab_entropy(last)
            sub["vocab_entropy_first"] = h0
            sub["vocab_entropy_last"] = h1
            sub["vocab_retention"] = M.clamp01(h1 / h0) if h0 > 0 else 0.0
            sub["repetition_by_window"] = [M.repetition_rate(w, 4) for w in wins]
            sub["repetition_slope"] = M.slope(sub["repetition_by_window"])
            comps += [sub["late_novelty"], sub["vocab_retention"]]
        comps.append(sub["novelty"])
        sub["contradictions"] = M.contradiction_count(texts)
        sub["integrity"] = 1.0 / (1.0 + sub["contradictions"])
        comps.append(sub["integrity"])
    plan_texts = [M.norm_text(r.get("response", "")) for r in _ok(plan)]
    plan_texts = [t for t in plan_texts if t]
    if len(plan_texts) >= 2:
        sub["plan_diversity"] = M.mean_pairwise_distance(plan_texts, 3)
        comps.append(sub["plan_diversity"])
    # C 層と同じ妥当性ゲート(独り言が日本語でないなら語彙収縮を測っても意味がない)
    sub["latin_ratio"] = M.latin_ratio(texts) if texts else 0.0
    sub["japanese_ratio"] = 1.0 - sub["latin_ratio"] if texts else 0.0
    base = _mean(comps, None) if comps else None
    sub["score_before_gate"] = base
    sub["score"] = None if base is None else base * sub["japanese_ratio"]
    return sub


# ---------------------------------------------------------------- 束ね
LAYER_LABEL = {"A": "A 個体行動", "B": "B 摂動応答", "C": "C 会話統計",
               "D": "D 分散と裾", "E": "E 長期退行"}


def profile_model(entry: dict, refs: dict[str, dict]) -> dict:
    tests = entry["tests"]
    ref_time = refs.get(REF_TIME_USE)
    ref_dial = refs.get(REF_DIALOGUE)
    layers: dict[str, dict] = {}
    if S.A_TEST in tests:
        layers["A"] = score_a(tests[S.A_TEST], ref_time)
    if S.B_TEST in tests:
        layers["B"] = score_b(tests[S.B_TEST])
    if S.C_TEST in tests:
        layers["C"] = score_c(tests[S.C_TEST], ref_dial)
    if S.D_TEST in tests:
        layers["D"] = score_d(tests[S.D_TEST])
    if S.E_MONO_TEST in tests or S.E_PLAN_TEST in tests:
        layers["E"] = score_e(tests.get(S.E_MONO_TEST, []),
                              tests.get(S.E_PLAN_TEST, []))
    man = entry["manifest"]
    return {
        "model": man.get("model", {}),
        "config": man.get("config", {}),
        "calls": man.get("calls"),
        "errors": man.get("errors"),
        "elapsed_s": man.get("elapsed_s"),
        "determinism_probe": man.get("determinism_probe"),
        "digests": man.get("digests"),
        "layers": layers,
    }


def add_variance_ratio(profiles: dict[str, dict], ref_d: float | None) -> None:
    """D 層の分散比。人間参照があれば絶対、無ければ**モデル間相対**(最良=1.0)。"""
    scores = {slug: p["layers"].get("D", {}).get("score")
              for slug, p in profiles.items()}
    vals = [v for v in scores.values() if v is not None]
    if not vals:
        return
    if ref_d is not None and ref_d > 0:
        base, mode = float(ref_d), "absolute_vs_human_reference"
    else:
        base = max(vals) if max(vals) > 0 else None
        mode = "relative_to_best_model"
    for slug, p in profiles.items():
        d = p["layers"].get("D")
        if d is None or d.get("score") is None:
            continue
        d["variance_ratio_mode"] = mode
        d["variance_ratio"] = (M.variance_ratio(d["score"], base)
                               if base else None)


def placebo_sanity(profiles: dict[str, dict]) -> dict:
    """プラセボが全層で最下位に沈むか。沈まない層があれば FAIL(=指標が測れていない)。"""
    placebos = [s for s, p in profiles.items()
                if p["model"].get("backend") == "placebo"]
    others = [s for s in profiles if s not in placebos]
    if not placebos:
        return {"verdict": "INCONCLUSIVE",
                "reason": "プラセボが含まれていない(健全性を確認できない)"}
    if not others:
        return {"verdict": "INCONCLUSIVE", "reason": "対照となる実モデルがいない"}
    per_layer = {}
    for layer in S.LAYER_IDS:
        pv = [profiles[s]["layers"].get(layer, {}).get("score") for s in placebos]
        ov = [profiles[s]["layers"].get(layer, {}).get("score") for s in others]
        pv = [v for v in pv if v is not None]
        ov = [v for v in ov if v is not None]
        if not pv or not ov:
            per_layer[layer] = {"status": "SKIP", "reason": "スコア欠測"}
            continue
        ok = max(pv) < min(ov)
        per_layer[layer] = {
            "status": "PASS" if ok else "FAIL",
            "placebo_max": max(pv), "others_min": min(ov),
            "margin": min(ov) - max(pv),
        }
    checked = [v for v in per_layer.values() if v["status"] in ("PASS", "FAIL")]
    if not checked:
        verdict = "INCONCLUSIVE"
    elif all(v["status"] == "PASS" for v in checked):
        verdict = "PASS"
    else:
        verdict = "FAIL"
    return {"verdict": verdict, "by_layer": per_layer,
            "placebos": placebos, "others": others}


def shortlist_verdict(profiles: dict[str, dict], floor: float | None) -> dict:
    """正典 §4 の判定線: 「プラセボに差をつけ、D 層分散比が下限超え」のモデルが2本以上。"""
    if floor is None:
        return {"verdict": "NOT_JUDGED",
                "reason": "--d-ratio-floor 未指定(閾値は較正対象なのでハードコードしない)"}
    placebos = {s for s, p in profiles.items()
                if p["model"].get("backend") == "placebo"}
    p_scores = {layer: max(
        [profiles[s]["layers"].get(layer, {}).get("score") or 0.0
         for s in placebos] or [0.0]) for layer in S.LAYER_IDS}
    passed = []
    for slug, p in profiles.items():
        if slug in placebos:
            continue
        d = p["layers"].get("D") or {}
        ratio = d.get("variance_ratio")
        beats = all(
            (p["layers"].get(layer, {}).get("score") is None)
            or (p["layers"][layer]["score"] > p_scores[layer])
            for layer in S.LAYER_IDS)
        if ratio is not None and ratio >= floor and beats:
            passed.append({"model": slug, "d_ratio": ratio})
    return {"verdict": "GO" if len(passed) >= 2 else "NO_GO",
            "floor": floor, "passed": passed,
            "rule": "全層でプラセボ超え かつ D 層分散比 >= floor のモデルが2本以上"}


# ---------------------------------------------------------------- 出力
def _fmt(v, nd=3):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def render_markdown(report: dict) -> str:
    L: list[str] = []
    A = L.append
    A("# モデル人間らしさテストバッテリー — プロファイル表")
    A("")
    A(f"- 生成: `scripts/model_battery/report.py`(schema `{REPORT_SCHEMA}`)")
    A(f"- 対象: {len(report['profiles'])} モデル")
    A("- **これは合否表ではない**。各層の下位指標を平均した相対比較用の要約であり、"
      "fleet 内の役割(会話/思考/価値判断を誰に任せるか)を決めるための地図である。")
    A("")
    A("## 1. モデル×層 プロファイル")
    A("")
    A("| モデル | 呼数 | err | A 個体行動 | B 摂動応答 | C 会話統計 | D 分散と裾 | "
      "E 長期退行 | D分散比 | 再現率 |")
    A("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for slug, p in report["profiles"].items():
        cells = [_fmt(p["layers"].get(x, {}).get("score")) for x in S.LAYER_IDS]
        dr = p["layers"].get("D", {}).get("variance_ratio")
        det = (p.get("determinism_probe") or {}).get("rate")
        A(f"| `{p['model'].get('name', slug)}` | {p.get('calls')} | "
          f"{p.get('errors')} | " + " | ".join(cells) +
          f" | {_fmt(dr)} | {_fmt(det, 2)} |")
    A("")
    mode = next((p["layers"]["D"].get("variance_ratio_mode")
                 for p in report["profiles"].values()
                 if p["layers"].get("D", {}).get("variance_ratio_mode")), None)
    if mode:
        A(f"- D分散比の基準: `{mode}`"
          + ("(人間参照が無いのでモデル間相対=最良モデルを 1.0 とした値)"
             if mode == "relative_to_best_model" else ""))
    A("- 再現率 = `determinism_probe`(先頭 K 呼を同一シードで再送した応答一致率)。"
      "ハーネス層の決定論とは別物で、**モデル/サーバ側の再現性**を表す。")
    A("")

    A("## 2. プラセボ健全性")
    A("")
    ps = report["placebo_sanity"]
    A(f"**{ps['verdict']}** — {ps.get('reason', '')}")
    if "by_layer" in ps:
        A("")
        A("| 層 | 判定 | プラセボ最大 | 実モデル最小 | 差 |")
        A("|---|---|--:|--:|--:|")
        for layer, v in ps["by_layer"].items():
            A(f"| {LAYER_LABEL[layer]} | {v['status']} | "
              f"{_fmt(v.get('placebo_max'))} | {_fmt(v.get('others_min'))} | "
              f"{_fmt(v.get('margin'))} |")
        A("")
        A("- FAIL の層は「その指標が人間らしさを測れていない」ことを意味する"
          "(モデルが悪いのではなく指標が悪い)。")
    A("")

    A("## 3. 判定線(正典 §4)")
    A("")
    sv = report["shortlist"]
    A(f"**{sv['verdict']}** — {sv.get('rule', sv.get('reason', ''))}")
    if sv.get("floor") is not None:
        A(f"- D 層分散比の下限 = {sv['floor']}(★引数 `--d-ratio-floor` で与えた"
          "較正対象の値。コードにハードコードしていない)")
        for x in sv.get("passed", []):
            A(f"  - 通過: `{x['model']}` (D 分散比 {x['d_ratio']:.3f})")
    A("")

    A("## 4. 層ごとの下位指標")
    for layer in S.LAYER_IDS:
        rows = {slug: p["layers"].get(layer) for slug, p in
                report["profiles"].items() if p["layers"].get(layer)}
        if not rows:
            continue
        keys: list[str] = []
        for v in rows.values():
            for k in v:
                if k not in keys and isinstance(v[k], (int, float, type(None))) \
                        and k not in ("score",):
                    keys.append(k)
        A("")
        A(f"### {LAYER_LABEL[layer]}")
        A("")
        A("| 指標 | " + " | ".join(f"`{s}`" for s in rows) + " |")
        A("|---|" + "--:|" * len(rows))
        A("| **score** | " + " | ".join(_fmt(v.get("score")) for v in rows.values())
          + " |")
        for k in keys:
            A(f"| {k} | " + " | ".join(_fmt(v.get(k)) for v in rows.values()) + " |")
        notes = {s: v.get("note") for s, v in rows.items() if v.get("note")}
        for s, n in notes.items():
            A(f"- `{s}`: {n}")
    A("")

    A("## 5. 参照統計の出典")
    A("")
    if report["references"]:
        for line in report["references"]:
            A(line)
    else:
        A("- なし(参照統計が読み込まれていない = 絶対評価は行われていない)")
    A("")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="バッテリー結果のレポート生成")
    ap.add_argument("--raw", default=str(REPO_ROOT / "data" / "battery" / "raw"))
    ap.add_argument("--reference",
                    default=str(REPO_ROOT / "data" / "battery" / "reference"))
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "battery"))
    ap.add_argument("--d-ratio-floor", type=float, default=None,
                    help="D 層分散比の下限(★較正対象。未指定なら合否を出さない)")
    ap.add_argument("--stdout", action="store_true", help="Markdown を標準出力にも出す")
    args = ap.parse_args(argv)

    raw = load_raw(Path(args.raw))
    if not raw:
        print(f"[report] raw が空: {args.raw}", file=sys.stderr)
        return 1
    try:
        refs = R.load_dir(Path(args.reference))
    except R.ReferenceError as exc:
        print(f"[report] 参照統計が掟に反している: {exc}", file=sys.stderr)
        return 2

    profiles = {slug: profile_model(entry, refs)
                for slug, entry in sorted(raw.items())}
    ref_d = None
    rd = refs.get(REF_DIALOGUE)
    if rd is not None:
        ref_d = R.series_values(rd, "human_choice_entropy")
    add_variance_ratio(profiles, ref_d)

    report = {
        "schema": REPORT_SCHEMA,
        "profiles": profiles,
        "placebo_sanity": placebo_sanity(profiles),
        "shortlist": shortlist_verdict(profiles, args.d_ratio_floor),
        "references": R.attribution_lines(refs.values()),
        "reference_ids": sorted(refs),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8")
    md = render_markdown(report)
    (out / "report.md").write_text(md, encoding="utf-8")
    print(f"[report] {out/'report.json'} / {out/'report.md'} を書いた",
          file=sys.stderr)
    print(f"[report] プラセボ健全性={report['placebo_sanity']['verdict']} "
          f"判定線={report['shortlist']['verdict']}", file=sys.stderr)
    if args.stdout:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
