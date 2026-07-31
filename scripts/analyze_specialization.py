#!/usr/bin/env python
"""語彙の専門化スコア(第78バッチ Part C1)= 凍結 3 指標の最後の 1 つ。

    python scripts/analyze_specialization.py runs/day10 \
        --control runs/day10_propoff --min-delta 0.02 --alpha 0.05

設計正典: docs/plans/source/dual-mode-instruments.md Part C /
          docs/plans/dual-mode-observe-verify-plan.md §2 第78行。

なぜ「差分でしか主張しない」のか
--------------------------------
信念距離(第73)や円環の閉じ(第74)と違い、**語彙の専門化にはオラクルが存在しない**。
「専門的に見える」は観察結果でしかなく、ペルソナのプロンプトに職業語が書いてあるだけでも
それらしいスコアが出る。したがって帰無モデルを自前で作る:

    ablate.propagation_off = 発話は通常どおり生成するが**他エージェントの文脈に入らない**

この対照でスコアが落ちなければ、その専門化は相互作用ではなく**プロンプト由来**である。
本スクリプトは **単独ランのスコアに合否を付けない**(--control が無ければ verdict は
NO_CONTROL = 判定不能)。閾値は事前登録(U-10)と同じ流儀で **引数必須**。

3 つの下位スコア(すべて L1 だけから決定論計算・乱数ゼロ・stdlib のみ)
----------------------------------------------------------------------
① topic_narrowing  … **トピッククラスタ内の語彙の狭まり**
     トピック = 高頻度トークン t を持つ発話の集合(t がアンカー。過度に一般的な
     トークンは df 比 --max-df で除外)。各クラスタで **前半窓 / 後半窓** の TTR
     (異なりトークン数 ÷ トークン数)を **同じトークン予算**で切り出して比べ、
     early − late を「狭まり」とする(正 = 語彙が狭まった)。
     ★トークン予算を揃えるのは TTR が長さに強く依存するため(Herdan/Heaps の問題)。

② role_differentiation … **役割による語彙の分化**
     agents.json の occupation で発話を層別し、役割ごとのトークン分布を作る。
     全ペアの **Jensen-Shannon divergence**(log2 = 値域 [0,1])の平均。
     1 に近いほど役割ごとに使う語が違う。

③ vocab_persistence … **特定語彙の時間持続**
     「定着した」トークン(出現 >= --min-count かつ相異なる話者 >= --min-users)に
     ついて、初出日から run 末までの日ビンのうち**実際に出現した日ビンの割合**。
     1 に近いほど、いったん現れた語が使われ続けている。

  composite = 3 つの単純平均(**便宜値**。判定は下位スコアの差分も併記して読む)。

トークン化について(正直な限界)
--------------------------------
形態素解析は使わず **文字 n-gram**(既定 n=2)。observer/echo.py の自己類似度と同じ流儀で、
外部辞書に依存せず決定論・再現可能であることを優先した。したがって「語」ではなく
「文字の並び」を数えている。日本語の語彙分化の代理指標であり、語そのものの計量ではない。

出力
----
  <out>/specialization.json  … 全スコアと内訳(sort_keys=True でバイト同一)
  <out>/specialization_report.md … 日本語の要約。**2 節に分けて**出す:
     §A 実装健全性の検証(golden 一致・k 非依存・fingerprint・等級・指標ハッシュ)
     §B 現象の由来の検証(アブレーション差分)
   この 2 つは種類の違う検証なので同じ表に混ぜない(Part G の要求)。
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Windows cp932 対策(コンソール出力の文字化け・例外を防ぐ)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                        # noqa: BLE001
    pass

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from society import registry as R                         # noqa: E402
from society.observer import measure as m                 # noqa: E402
from society.observer import metrics_spec as MS           # noqa: E402
from society.observer import state_hash as SH             # noqa: E402

# 発話とみなすイベント種(observer/echo.py・norms.py と同一定義)
UTTERANCE_KINDS = ("speak", "sns_post", "dm")
STEPS_PER_DAY = 144

# トークン化から落とす文字(空白・句読点・括弧・記号)。**語彙リテラルは 1 語も置かない**。
_DROP = set(" \t\n　、。「」『』()()[]【】,.!?！?…・:;:;\"'‘’“”-—~〜/\\")

NOTE = ("文字 n-gram による代理指標であって語の計量ではない。"
        "単独ランのスコアに合否は付けない(対照との差分でのみ主張する)。"
        "実装健全性の検証(§A)と現象の由来の検証(§B)は種類が違うので混ぜて読まないこと。")


# --------------------------------------------------------------------------- #
# トークン化(決定論・stdlib のみ)
# --------------------------------------------------------------------------- #
def tokens_of(text: str, n: int) -> list[str]:
    """文字 n-gram(記号・空白を落としてから連続 n 文字を全て取る)。"""
    s = "".join(ch for ch in str(text) if ch not in _DROP)
    if len(s) < n:
        return [s] if s else []
    return [s[i:i + n] for i in range(len(s) - n + 1)]


# --------------------------------------------------------------------------- #
# L1 読み出し
# --------------------------------------------------------------------------- #
def load_utterances(run_dir: str, ngram: int) -> tuple[list[dict], int]:
    """(発話レコード列, ランの最終 step)。レコード = {step, agent, tokens}。"""
    recs: list[dict] = []
    last = -1
    for e in m.stream_events(run_dir, ["step", "agent_id", "kind", "payload"]):
        s = int(e["step"])
        last = max(last, s)
        if e["kind"] not in UTTERANCE_KINDS:
            continue
        aid = e["agent_id"]
        if not isinstance(aid, int) or aid < 0:
            continue
        text = (e.get("payload") or {}).get("text") or ""
        toks = tokens_of(text, ngram)
        if toks:
            recs.append({"step": s, "agent": aid, "tokens": toks})
    return recs, last


def load_roles(run_dir: str) -> dict[int, str]:
    """agents.json の occupation(無ければ空 dict = ② は算出不能として None を返す)。"""
    path = os.path.join(run_dir, "agents.json")
    try:
        with open(path, encoding="utf-8") as fh:
            rows = json.load(fh)
    except (OSError, ValueError):
        return {}
    out = {}
    for r in rows:
        try:
            out[int(r["id"])] = str(r.get("occupation") or "")
        except (KeyError, TypeError, ValueError):
            continue
    return out


# --------------------------------------------------------------------------- #
# ① トピッククラスタ内の語彙の狭まり
# --------------------------------------------------------------------------- #
def topic_narrowing(recs, last_step: int, *, topics: int, max_df: float,
                    budget: int, min_cluster: int) -> dict:
    """アンカートークンごとのクラスタで TTR(前半窓)− TTR(後半窓)の平均。"""
    if not recs:
        return {"value": None, "n_clusters": 0, "reason": "発話 0 件"}
    n_utt = len(recs)
    df = Counter()
    for r in recs:
        df.update(set(r["tokens"]))
    # 過度に一般的なトークン(df 比 > max_df)はトピックのアンカーにしない
    cands = [(t, c) for t, c in df.items() if c / n_utt <= max_df]
    cands.sort(key=lambda kv: (-kv[1], kv[0]))          # 頻度降順 → 文字列昇順(決定論)
    anchors = [t for t, _ in cands[:max(1, int(topics))]]
    mid = last_step / 2.0
    per: list[dict] = []
    for t in anchors:
        early = [r for r in recs if t in r["tokens"] and r["step"] <= mid]
        late = [r for r in recs if t in r["tokens"] and r["step"] > mid]
        if len(early) < min_cluster or len(late) < min_cluster:
            continue
        e_ttr = _ttr(early, budget)
        l_ttr = _ttr(late, budget)
        if e_ttr is None or l_ttr is None:
            continue
        per.append({"anchor": t, "n_early": len(early), "n_late": len(late),
                    "ttr_early": round(e_ttr, 6), "ttr_late": round(l_ttr, 6),
                    "narrowing": round(e_ttr - l_ttr, 6)})
    if not per:
        return {"value": None, "n_clusters": 0,
                "reason": f"前後窓とも {min_cluster} 件以上のクラスタが無い"}
    per.sort(key=lambda d: (-d["narrowing"], d["anchor"]))
    val = sum(d["narrowing"] for d in per) / len(per)
    return {"value": round(val, 6), "n_clusters": len(per),
            "anchors": per[:20], "reason": ""}


def _ttr(records, budget: int):
    """同じトークン予算で切った TTR(異なり ÷ 総数)。予算に満たなければ None。"""
    toks: list[str] = []
    for r in records:                                    # step 昇順(読み出し順)で詰める
        toks.extend(r["tokens"])
        if len(toks) >= budget:
            break
    if len(toks) < budget:
        return None
    toks = toks[:budget]
    return len(set(toks)) / float(budget)


# --------------------------------------------------------------------------- #
# ② 役割による語彙の分化(平均 Jensen-Shannon divergence)
# --------------------------------------------------------------------------- #
def role_differentiation(recs, roles: dict, *, min_tokens: int) -> dict:
    if not roles:
        return {"value": None, "n_roles": 0, "reason": "agents.json が読めない"}
    by_role: dict[str, Counter] = defaultdict(Counter)
    for r in recs:
        role = roles.get(r["agent"])
        if role:
            by_role[role].update(r["tokens"])
    kept = {k: v for k, v in by_role.items() if sum(v.values()) >= min_tokens}
    names = sorted(kept)
    if len(names) < 2:
        return {"value": None, "n_roles": len(names),
                "reason": f"トークン {min_tokens} 以上の役割が 2 群に満たない"}
    pairs: list[dict] = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pairs.append({"a": a, "b": b,
                          "jsd": round(_jsd(kept[a], kept[b]), 6)})
    val = sum(p["jsd"] for p in pairs) / len(pairs)
    return {"value": round(val, 6), "n_roles": len(names), "roles": names,
            "n_pairs": len(pairs), "pairs": sorted(
                pairs, key=lambda d: (-d["jsd"], d["a"], d["b"]))[:20],
            "reason": ""}


def _jsd(ca: Counter, cb: Counter) -> float:
    """Jensen-Shannon divergence(log2 = 値域 [0,1])。順序非依存・決定論。"""
    na, nb = float(sum(ca.values())), float(sum(cb.values()))
    keys = sorted(set(ca) | set(cb))                     # 反復順を固定(決定論)
    out = 0.0
    for k in keys:
        p = ca.get(k, 0) / na
        q = cb.get(k, 0) / nb
        mval = 0.5 * (p + q)
        if p > 0:
            out += 0.5 * p * math.log2(p / mval)
        if q > 0:
            out += 0.5 * q * math.log2(q / mval)
    return max(0.0, min(1.0, out))


# --------------------------------------------------------------------------- #
# ③ 特定語彙の時間持続
# --------------------------------------------------------------------------- #
def vocab_persistence(recs, last_step: int, *, min_count: int, min_users: int,
                      bin_steps: int) -> dict:
    if not recs:
        return {"value": None, "n_tokens": 0, "reason": "発話 0 件"}
    count = Counter()
    users: dict[str, set] = defaultdict(set)
    bins: dict[str, set] = defaultdict(set)
    for r in recs:
        b = int(r["step"]) // int(bin_steps)
        for t in set(r["tokens"]):
            count[t] += 1
            users[t].add(r["agent"])
            bins[t].add(b)
    last_bin = int(last_step) // int(bin_steps)
    per: list[dict] = []
    for t in sorted(count):                              # 反復順を固定(決定論)
        if count[t] < min_count or len(users[t]) < min_users:
            continue
        first = min(bins[t])
        span = last_bin - first + 1
        per.append({"token": t, "count": count[t], "users": len(users[t]),
                    "first_bin": first, "span": span, "seen": len(bins[t]),
                    "persistence": round(len(bins[t]) / float(span), 6)})
    if not per:
        return {"value": None, "n_tokens": 0,
                "reason": f"count>={min_count} かつ users>={min_users} のトークンが無い"}
    per.sort(key=lambda d: (-d["persistence"], -d["count"], d["token"]))
    val = sum(d["persistence"] for d in per) / len(per)
    return {"value": round(val, 6), "n_tokens": len(per), "tokens": per[:20],
            "reason": ""}


# --------------------------------------------------------------------------- #
# ラン 1 本のスコア
# --------------------------------------------------------------------------- #
def score_run(run_dir: str, args) -> dict:
    recs, last = load_utterances(run_dir, args.ngram)
    roles = load_roles(run_dir)
    a = topic_narrowing(recs, last, topics=args.topics, max_df=args.max_df,
                        budget=args.ttr_budget, min_cluster=args.min_cluster)
    b = role_differentiation(recs, roles, min_tokens=args.min_role_tokens)
    c = vocab_persistence(recs, last, min_count=args.min_count,
                          min_users=args.min_users, bin_steps=args.bin_steps)
    vals = [x["value"] for x in (a, b, c) if x["value"] is not None]
    composite = round(sum(vals) / len(vals), 6) if len(vals) == 3 else None
    return {
        "run": os.path.basename(os.path.normpath(run_dir)),
        "dir": str(run_dir),
        "n_utterances": len(recs), "last_step": last,
        "topic_narrowing": a, "role_differentiation": b, "vocab_persistence": c,
        "composite": composite,
        "composite_partial": (round(sum(vals) / len(vals), 6) if vals else None),
        "manifest": _manifest_facts(run_dir),
        "summary": _summary_facts(run_dir),
    }


def _manifest_facts(run_dir: str) -> dict:
    man = R.read_manifest(run_dir) or {}
    feats = man.get("features") or {}
    risks = sorted({str(e.get("fingerprint_risk"))
                    for e in (feats.get("enabled") or [])})
    return {
        "present": bool(man),
        "run_mode": man.get("run_mode"),
        "git_sha": (man.get("git") or {}).get("sha"),
        "config_determinism_sha256": man.get("config_determinism_sha256"),
        "metrics_spec_hash": man.get("metrics_spec_hash"),
        "ablate": man.get("ablate"),
        "tiers": sorted({str(e.get("repro_tier"))
                         for e in (feats.get("enabled") or [])}),
        "fingerprint_risks": risks,
        "undeclared": feats.get("undeclared") or [],
    }


def _summary_facts(run_dir: str) -> dict:
    try:
        with open(os.path.join(run_dir, "summary.json"), encoding="utf-8") as fh:
            s = json.load(fh)
    except (OSError, ValueError):
        return {}
    return {k: s.get(k) for k in ("llm_calls", "llm_cache_hits", "n_events",
                                  "n_agents", "n_steps", "n_items")}


def _state_hash_facts(run_dir: str) -> dict:
    path = os.path.join(run_dir, SH.FILENAME)
    if not os.path.exists(path):
        return {"present": False}
    try:
        rep = SH.verify_chain(SH.read_chain(path))
    except ValueError as exc:
        return {"present": True, "ok": False, "reason": str(exc)}
    return {"present": True, **rep}


# --------------------------------------------------------------------------- #
# 条件(= ラン群)の集約と差分
# --------------------------------------------------------------------------- #
def _mean(vals):
    v = [x for x in vals if x is not None]
    return round(sum(v) / len(v), 6) if v else None


def condition(dirs: list[str], args, label: str) -> dict:
    runs = [score_run(d, args) for d in dirs]
    keys = ("topic_narrowing", "role_differentiation", "vocab_persistence")
    return {
        "label": label, "n_runs": len(runs),
        "means": {k: _mean([r[k]["value"] for r in runs]) for k in keys},
        "composite": _mean([r["composite"] for r in runs]),
        "runs": runs,
    }


def compare(treat: dict, ctrl: dict, *, min_delta: float, alpha: float) -> dict:
    keys = ("topic_narrowing", "role_differentiation", "vocab_persistence")
    deltas = {}
    for k in keys:
        t, c = treat["means"][k], ctrl["means"][k]
        deltas[k] = (None if (t is None or c is None) else round(t - c, 6))
    dt, dc = treat["composite"], ctrl["composite"]
    d_comp = None if (dt is None or dc is None) else round(dt - dc, 6)
    perm = _perm_test([r["composite"] for r in ctrl["runs"]],
                      [r["composite"] for r in treat["runs"]])
    if d_comp is None:
        verdict, why = "INCONCLUSIVE", "composite が算出できない条件がある(下位スコアの欠測)"
    elif d_comp < min_delta:
        verdict, why = ("NOT_INTERACTION_DRIVEN",
                        f"差 {d_comp:+.6f} が閾値 {min_delta} 未満 = 伝播を止めても"
                        "専門化が落ちない = ペルソナのプロンプト由来を否定できない")
    elif perm["p"] is not None and perm["p"] >= alpha:
        verdict, why = ("UNDERPOWERED",
                        f"差 {d_comp:+.6f} は閾値以上だが p={perm['p']} >= {alpha}"
                        f"(ラン数 treat={treat['n_runs']} / ctrl={ctrl['n_runs']})"
                        "= ラン数不足で偶然と区別できない")
    else:
        verdict, why = ("INTERACTION_DRIVEN",
                        f"差 {d_comp:+.6f} >= {min_delta}"
                        + (f" かつ p={perm['p']} < {alpha}" if perm["p"] is not None
                           else "(検定はラン 1 本ずつのため未実施)"))
    return {"deltas": deltas, "delta_composite": d_comp, "perm": perm,
            "thresholds": {"min_delta": min_delta, "alpha": alpha},
            "verdict": verdict, "why": why}


def _perm_test(ctrl_vals, treat_vals) -> dict:
    """2 標本ラベル置換(全列挙・決定論)。片側でも両側でもなく **両側**。

    analyze_norms.two_sample_perm と同じ作法だが、条件あたりのラン数が小さい前提なので
    numpy を使わず全列挙だけにしてある(組合せが大きいときは p=None=検定しない)。
    """
    a = [float(v) for v in ctrl_vals if v is not None]
    b = [float(v) for v in treat_vals if v is not None]
    base = {"n_ctrl": len(a), "n_treat": len(b), "method": "none", "p": None}
    if len(a) < 2 or len(b) < 2:
        base["method"] = "skipped_n<2"       # ★ラン 1 本ずつでは検定できない(正直に)
        return base
    vals = a + b
    n, na = len(vals), len(a)
    if math.comb(n, na) > 200000:
        base["method"] = "skipped_too_many"
        return base
    obs = sum(b) / len(b) - sum(a) / len(a)
    from itertools import combinations
    total = cnt = 0
    s_all = sum(vals)
    for pick in combinations(range(n), na):
        sa = sum(vals[i] for i in pick)
        d = (s_all - sa) / (n - na) - sa / na
        total += 1
        if abs(d) >= abs(obs) - 1e-12:
            cnt += 1
    return {**base, "method": "exhaustive", "n_perm": total,
            "p": round(cnt / total, 6), "min_p": round(2.0 / total, 6),
            "observed_diff": round(obs, 6)}


# --------------------------------------------------------------------------- #
# レポート(§A 実装健全性 / §B 現象の由来 を**別セクション**で出す)
# --------------------------------------------------------------------------- #
def _soundness(treat: dict, ctrl: dict | None, all_dirs: list[str]) -> dict:
    """§A: 記録済みの成果物(manifest / summary / state_hash)から機械的に採る事実。"""
    mans = [r["manifest"] for cond in ([treat] + ([ctrl] if ctrl else []))
            for r in cond["runs"]]
    spec = sorted({m0.get("metrics_spec_hash") for m0 in mans if m0.get("present")})
    live = MS.compute()["metrics_spec_hash"]
    kt = [r["summary"].get("llm_calls") for r in treat["runs"]]
    kc = ([r["summary"].get("llm_calls") for r in ctrl["runs"]] if ctrl else [])
    tier = R.check_runs(all_dirs, allow_mismatch=True)
    return {
        "metrics_spec_hash_live": live,
        "metrics_spec_hash_in_runs": spec,
        "metrics_spec_consistent": (len(spec) == 1 and spec[0] == live),
        "llm_calls_treat": kt, "llm_calls_control": kc,
        "llm_calls_delta": (None if not (kt and kc) or kt[0] is None or kc[0] is None
                            else _mean(kc) - _mean(kt)),
        "run_modes": sorted({m0.get("run_mode") for m0 in mans}),
        "repro_tiers": sorted({t for m0 in mans for t in (m0.get("tiers") or [])}),
        "fingerprint_risks": sorted({t for m0 in mans
                                     for t in (m0.get("fingerprint_risks") or [])}),
        "undeclared_toggles": sorted({t for m0 in mans
                                      for t in (m0.get("undeclared") or [])}),
        "tier_mismatch": bool(tier["mismatch"]),
        "tier_messages": tier["messages"],
        "state_hash": {os.path.basename(os.path.normpath(d)): _state_hash_facts(d)
                       for d in all_dirs},
    }


def _dash(value) -> str:
    """欠測は "-"(0 と偽らない)。"""
    return "-" if value is None else str(value)


def _spec_cell(S: dict) -> str:
    hs = S["metrics_spec_hash_in_runs"]
    return ", ".join("`" + h[:16] + "…`" for h in hs) or "(manifest なし)"


def _spec_verdict(S: dict) -> str:
    return ("✔ 一致(事後の指標いじり無し)" if S["metrics_spec_consistent"]
            else "✖ **不一致**(ラン後に指標コードが変わっている)")


def render(res: dict) -> str:
    L = ["# 語彙の専門化スコア(Part C1・第78バッチ)", "",
         f"> {NOTE}", ""]
    S = res["soundness"]
    L += ["## §A 実装健全性の検証(**現象の主張ではない**)", "",
          "記録済みの成果物(run_manifest.json / summary.json / state_hash.jsonl)から"
          "機械的に採った事実。ここが崩れているなら §B は読む価値がない。", "",
          "| 項目 | 値 |", "|---|---|",
          f"| 指標定義ハッシュ(実行時) | `{S['metrics_spec_hash_live'][:16]}…` |",
          f"| ラン記録の指標定義ハッシュ | {_spec_cell(S)} |",
          f"| 指標定義の一致 | {_spec_verdict(S)} |",
          f"| ランモード | {', '.join(str(x) for x in S['run_modes'])} |",
          f"| 有効機能の再現性等級 | {', '.join(S['repro_tiers']) or '-'} |",
          f"| 等級混在 | {'✖ 混在' if S['tier_mismatch'] else '✔ なし'} |",
          f"| fingerprint_risk の内訳 | {', '.join(S['fingerprint_risks']) or '-'} |",
          f"| 未宣言トグル | {', '.join(S['undeclared_toggles']) or '✔ なし'} |",
          f"| LLM 呼数(treatment) | {S['llm_calls_treat']} |",
          f"| LLM 呼数(control) | {S['llm_calls_control'] or '-'} |",
          f"| 呼数の差(control − treatment) | {_dash(S['llm_calls_delta'])} |",
          ""]
    for name, sh in S["state_hash"].items():
        if sh.get("present"):
            ok = "✔ 整合" if sh.get("ok") else f"✖ **{sh.get('reason')}**"
            L.append(f"- 状態ハッシュチェーン `{name}`: {ok}(記録 {sh.get('n')} 件)")
    if any(v.get("present") for v in S["state_hash"].values()):
        L.append("")
    L += ["> golden L1 バイト一致と k 非依存の**本体**は tests/ のゲート(pytest)が担保する。"
          "ここに出るのは『そのランがどの等級・どの指標定義で回ったか』の来歴であって、"
          "ゲートの再実行ではない。", ""]

    L += ["## §B 現象の由来の検証(アブレーション差分)", ""]
    for cond in [res["treatment"]] + ([res["control"]] if res.get("control") else []):
        L += [f"### {cond['label']}(ラン {cond['n_runs']} 本)", "",
              "| 指標 | 値 |", "|---|---|",
              f"| ① トピック内の語彙の狭まり | {cond['means']['topic_narrowing']} |",
              f"| ② 役割による語彙の分化 | {cond['means']['role_differentiation']} |",
              f"| ③ 特定語彙の時間持続 | {cond['means']['vocab_persistence']} |",
              f"| composite(単純平均) | {cond['composite']} |", ""]
        for r in cond["runs"]:
            miss = [k for k in ("topic_narrowing", "role_differentiation",
                                "vocab_persistence") if r[k]["value"] is None]
            if miss:
                why = "; ".join(f"{k}: {r[k]['reason']}" for k in miss)
                L.append(f"- ⚠ `{r['run']}` は {', '.join(miss)} が算出不能({why})")
        L.append("")
    cmp_ = res.get("comparison")
    if cmp_ is None:
        L += ["### 判定", "",
              "**NO_CONTROL = 判定不能。** `--control`(ablate.propagation_off のラン)が"
              "指定されていない。語彙の専門化にはオラクルが無いため、**単独ランのスコアに"
              "合否は付けない**(Part C の設計原則)。", ""]
    else:
        d = cmp_["deltas"]
        L += ["### 判定(treatment − control)", "",
              "| 指標 | 差分 |", "|---|---|",
              f"| ① トピック内の語彙の狭まり | {d['topic_narrowing']} |",
              f"| ② 役割による語彙の分化 | {d['role_differentiation']} |",
              f"| ③ 特定語彙の時間持続 | {d['vocab_persistence']} |",
              f"| **composite** | **{cmp_['delta_composite']}** |", "",
              f"- 事前登録の閾値: `--min-delta {cmp_['thresholds']['min_delta']}` / "
              f"`--alpha {cmp_['thresholds']['alpha']}`",
              f"- 置換検定: method={cmp_['perm']['method']} p={cmp_['perm']['p']}"
              f"(ctrl {cmp_['perm']['n_ctrl']} 本 / treat {cmp_['perm']['n_treat']} 本)",
              "", f"### **{cmp_['verdict']}**", "", cmp_["why"], ""]
        if cmp_["perm"]["method"].startswith("skipped"):
            L.append("> ⚠ 条件あたり 2 本未満のため統計検定を実施していない。"
                     "差分の符号だけを見て有意性を語らないこと。")
        L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
def _expand(patterns) -> list[str]:
    out: list[str] = []
    for p in patterns:
        hits = sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p]
        for h in hits:
            if os.path.isdir(h) and h not in out:
                out.append(h)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="語彙の専門化スコア(Part C1)。判定は propagation_off 対照との差分のみ。")
    ap.add_argument("run_dirs", nargs="+", help="treatment のラン(glob 可)")
    ap.add_argument("--control", nargs="*", default=None,
                    help="対照ラン(ablate.propagation_off=true。glob 可)")
    # ---- 事前登録の判定閾値(**必須**。既定値をコードに埋め込まない)----
    ap.add_argument("--min-delta", type=float, required=True,
                    help="[必須] composite の差(treatment − control)の下限")
    ap.add_argument("--alpha", type=float, required=True,
                    help="[必須] 置換検定の有意水準(条件あたり 2 本以上のときのみ使用)")
    # ---- 計算パラメータ(既定あり。値は出力に必ず記録する)----
    ap.add_argument("--ngram", type=int, default=2, help="文字 n-gram の n(既定 2)")
    ap.add_argument("--topics", type=int, default=20, help="①のアンカー数(既定 20)")
    ap.add_argument("--max-df", type=float, default=0.5,
                    help="①のアンカーから外す df 比の上限(既定 0.5)")
    ap.add_argument("--ttr-budget", type=int, default=200,
                    help="①の TTR を測るトークン予算(既定 200)")
    ap.add_argument("--min-cluster", type=int, default=3,
                    help="①で前後窓それぞれに必要な最小発話数(既定 3)")
    ap.add_argument("--min-role-tokens", type=int, default=100,
                    help="②で役割を採用する最小トークン数(既定 100)")
    ap.add_argument("--min-count", type=int, default=5, help="③の最小出現数(既定 5)")
    ap.add_argument("--min-users", type=int, default=2, help="③の最小話者数(既定 2)")
    ap.add_argument("--bin-steps", type=int, default=STEPS_PER_DAY,
                    help=f"③の日ビン幅 step(既定 {STEPS_PER_DAY}=1日)")
    ap.add_argument("--out", default=None, help="出力先ディレクトリ(既定: 先頭ラン)")
    ap.add_argument("--allow-tier-mismatch", action="store_true",
                    help="再現性等級が異なるラン同士の比較を明示的に許可する")
    args = ap.parse_args()

    treat_dirs = _expand(args.run_dirs)
    ctrl_dirs = _expand(args.control or [])
    if not treat_dirs:
        raise SystemExit(f"treatment のランが見つからない: {args.run_dirs}")
    R.guard_or_die(treat_dirs + ctrl_dirs, allow_mismatch=args.allow_tier_mismatch)

    treat = condition(treat_dirs, args, "treatment(通常ラン)")
    ctrl = (condition(ctrl_dirs, args, "control(ablate.propagation_off)")
            if ctrl_dirs else None)
    res = {
        "schema": 1,
        "params": {k: getattr(args, k) for k in
                   ("ngram", "topics", "max_df", "ttr_budget", "min_cluster",
                    "min_role_tokens", "min_count", "min_users", "bin_steps",
                    "min_delta", "alpha")},
        "note": NOTE,
        "treatment": treat,
        "control": ctrl,
        "comparison": (compare(treat, ctrl, min_delta=args.min_delta,
                               alpha=args.alpha) if ctrl else None),
    }
    res["soundness"] = _soundness(treat, ctrl, treat_dirs + ctrl_dirs)

    out_dir = Path(args.out or treat_dirs[0])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "specialization.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8")
    md = render(res)
    (out_dir / "specialization_report.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"[written] {out_dir / 'specialization.json'}")
    print(f"[written] {out_dir / 'specialization_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
