"""規範化ステージ + コホートタグ + 下方因果の解析(第74バッチ 2026-07-31)。

正典: docs/research/hackathon1-analysis/ideas-shortlist.md §2 ②(IDEA③ 規範化ステージ)/
      docs/plans/source/dual-mode-instruments.md Part E(コホートタグ・規範候補・下方因果)。

何を出すか
----------
(1) **規範化ステージ表**(語 × 4 段階)
    S1 初出 / S2 他者引用 / S3 定冠詞化 / S4 制度化 の到達 step と到達させた個体。
    **coiner(命名者)と institutionalizer(制度化者)を別役割**として突き合わせ、
    「別人である割合」を出す(beyond-badminton の実例= 命名と形式化が別ペルソナ)。
(2) **規範候補と成立 step**(Part E(2))
    「ステージ >= --norm-stage」かつ「相異なる利用者 >= --norm-threshold 人」を
    **初めて同時に満たした step** を成立 step とする。
(3) **コホートタグ + 下方因果**(Part E(1)(3))
    各エージェントの初 presence step(= L1 初出 step)で
      pre  … 規範成立**前**に初参入(規範の形成に参加しうた群)
      post … 規範成立**後**に初参入(形成に一切参加していない群)
    に分け、**参入初日の行動分布**を比較する。post 群の行動が規範側へ寄っていれば
    「円環が閉じた(規範が個体を拘束し返した)」と判定する材料になる。
    検定は **2 標本ラベル置換**(第63バッチ analyze_endo_treatment.py の流儀:
    小 n は全列挙・大 n は MC で p=(b+1)/(n+1)= Phipson & Smyth 2010)。
    複数ランをまとめるときは **ラン内で層別置換**する(Farine 2022: 置換の単位を
    誤ると過大有意になる)。

事前登録(U-10)
---------------
判定線は **引数必須**で、既定値をコードに埋め込まない:
    --norm-stage      規範候補とみなす最小ステージ(1..4)
    --norm-threshold  その語を使った相異なるエージェント数の下限
どちらも未指定ならエラーで止まる(事後に閾値をいじった疑いを構造的に断つ)。

使い方
------
    python scripts/analyze_norms.py runs/day10 --norm-stage 3 --norm-threshold 5
    python scripts/analyze_norms.py "runs/endo14_*" --norm-stage 2 --norm-threshold 3 \
        --out runs/norms_report.json

正直な限界(出力にも印字する)
------------------------------
- S3/S4 は表層文字列の一致でしか判定しない(過小検出=下限)。mock ランでは
  発話テンプレートに該当表現が無いため **S3/S4 は 0 件が正常**。
- `pool.enabled=false`(既定)のランは全員が step 0 から在場するため **コホートが 1 群**
  しかできない。下方因果の群間比較が成立するのは日次 presence 選抜(pool ON)のランだけ。
- 「行動分布」はイベント種の構成比であって主観的意味づけではない。
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:                # 同ディレクトリの run_dt を import
    sys.path.insert(0, _HERE)

import numpy as np                                          # noqa: E402
from omegaconf import OmegaConf                             # noqa: E402

import run_dt                                               # noqa: E402  (W2-3)
from society import registry as R                           # noqa: E402
from society.observer import measure as m                   # noqa: E402
from society.observer import norms as N                     # noqa: E402
from society.observer import stream as S                    # noqa: E402

MC_ITER = 10000          # analyze_endo_treatment.py と同値(MC の反復数)
EXHAUST_MAX = 200000     # 全列挙する組合せ数の上限(超えたら MC へ)
# W2-3: **ラン依存**(run.dt_min)。この定数は正準 Δt=10 の 144 で、下位関数の既定引数
# としてだけ残る(= 渡さなければ従来と完全同値)。実際の値は入口 `analyze_run` が
# `run_dt.steps_per_day(run_dir)` で解決して配る。「参入初日」は暦ではなく
# **初出 step から 1 日ぶんの step** なので、Δt が変われば step 数も変わる。
STEPS_PER_DAY = run_dt.CANON_STEPS_PER_DAY

# 参入初日の行動分布を比較する既定のイベント種(存在するものだけ使う)。
FIRST_DAY_KINDS = ("speak", "sns_post", "dm", "hear", "stay", "route_start",
                   "arrive", "move_segment", "vocab_use", "label_coin",
                   "label_adopt", "transmission", "llm_deliberate", "reflect")


# --------------------------------------------------------------------------- #
# マーカー(検出語彙)の解決 — コードには 1 語も持たない
# --------------------------------------------------------------------------- #
def load_markers(run_dir: str) -> tuple[dict, str]:
    """ランの config.yaml → labeling.norm_stage.markers。無ければ出荷 conf へ後退。"""
    for path, src in ((os.path.join(run_dir, "config.yaml"), "run"),
                      (str(REPO_ROOT / "conf" / "config.yaml"), "shipped")):
        if not os.path.exists(path):
            continue
        try:
            cfg = OmegaConf.load(path)
        except Exception:                                    # noqa: BLE001
            continue
        blk = N.cfg_of_config(cfg)
        if blk["markers"]["definite"] or blk["markers"]["agreement"]:
            return blk, src
    return N.cfg_from_block(None), "empty"


# --------------------------------------------------------------------------- #
# 2 標本ラベル置換検定(analyze_endo_treatment.sign_flip_test と同じ作法)
# --------------------------------------------------------------------------- #
def two_sample_perm(a, b, n_mc: int = MC_ITER, rng_seed: int = 0,
                    strata=None) -> dict:
    """群 a / b の平均差の両側 p(決定論)。

    strata を渡すと **層(=ラン)内でだけ**ラベルを入れ替える(Farine 2022:
    ラン間の差を群間差と取り違えないための層別置換)。strata は a, b と同じ長さの
    ラベル列を連結したもの(a 側 → b 側の順)。
    小 n(組合せ数 <= EXHAUST_MAX・層なし)は全列挙で厳密。それ以外は MC で
    p=(b+1)/(n_mc+1)(p が 0 にならない不偏側)。
    """
    xa = [float(v) for v in a if v is not None and not _isnan(v)]
    xb = [float(v) for v in b if v is not None and not _isnan(v)]
    base = {"n_a": len(xa), "n_b": len(xb),
            "mean_a": round(float(np.mean(xa)), 6) if xa else None,
            "mean_b": round(float(np.mean(xb)), 6) if xb else None}
    if not xa or not xb:
        return {**base, "diff": None, "p": None, "method": "none"}
    obs = float(np.mean(xb)) - float(np.mean(xa))
    base["diff"] = round(obs, 6)
    vals = np.asarray(xa + xb, dtype=float)
    n, na = len(vals), len(xa)
    if float(np.max(vals)) - float(np.min(vals)) < 1e-15:
        return {**base, "p": 1.0, "method": "degenerate_constant"}
    if strata is None and math.comb(n, na) <= EXHAUST_MAX:
        from itertools import combinations
        total = cnt = 0
        idx = range(n)
        s_all = float(vals.sum())
        for pick in combinations(idx, na):
            sa = float(vals[list(pick)].sum())
            d = (s_all - sa) / (n - na) - sa / na
            total += 1
            if abs(d) >= abs(obs) - 1e-12:
                cnt += 1
        return {**base, "p": round(cnt / total, 6), "method": "exhaustive",
                "n_perm": total, "min_p": round(2.0 / total, 6)}
    rng = np.random.default_rng(int(rng_seed))
    labels = np.array([0] * na + [1] * (n - na))
    if strata is not None:
        strata = np.asarray(list(strata))
        groups = [np.flatnonzero(strata == s) for s in sorted(set(strata.tolist()))]
    cnt = 0
    for _ in range(n_mc):
        perm = labels.copy()
        if strata is None:
            rng.shuffle(perm)
        else:
            for g in groups:
                sub = perm[g].copy()
                rng.shuffle(sub)
                perm[g] = sub
        ga, gb = vals[perm == 0], vals[perm == 1]
        if len(ga) == 0 or len(gb) == 0:
            continue
        if abs(float(gb.mean()) - float(ga.mean())) >= abs(obs) - 1e-12:
            cnt += 1
    return {**base, "p": round((cnt + 1) / (n_mc + 1), 6),
            "method": "monte_carlo_stratified" if strata is not None else "monte_carlo",
            "n_perm": n_mc, "min_p": round(1.0 / (n_mc + 1), 6)}


def _isnan(v) -> bool:
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return False


# --------------------------------------------------------------------------- #
# ラン 1 本の解析
# --------------------------------------------------------------------------- #
def _read_initial_frame(run_dir: str):
    """summary.json の initial_frame(第74バッチ IDEA④)。無ければ None。"""
    path = os.path.join(run_dir, "summary.json")
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh).get("initial_frame")
    except (OSError, ValueError, AttributeError):
        return None


def first_day_features(run_dir: str, first_step: dict, words: list,
                       t_norm: int | None,
                       spd: int = STEPS_PER_DAY) -> tuple[list[dict], int]:
    """個体ごとの **参入初日**(初出 step から 1 日)の行動特徴を 1 パスで集める。

    返り値は (レコード列, ランの最終 step)。ランの終端に掛かって初日が丸ごと観測でき
    なかった個体には `full_day=False` を立てる(絶対件数の比較は歪むため)。

    W2-3: `spd` = 1 日の step 数。既定は正準 144 なので **渡さなければ従来と完全同値**。
    """
    counts: dict[int, dict] = {}
    utt: dict[int, list] = {}
    last_step = -1
    for e in m.stream_events(run_dir, ["step", "agent_id", "kind", "payload"]):
        s = int(e["step"])
        if s > last_step:
            last_step = s
        aid = e["agent_id"]
        if not isinstance(aid, int) or aid < 0:
            continue
        f0 = first_step.get(aid)
        if f0 is None or s < f0 or s >= f0 + spd:
            continue
        row = counts.setdefault(aid, {"agent": aid, "first_step": f0,
                                      "n_events": 0, "kinds": {}})
        row["n_events"] += 1
        row["kinds"][e["kind"]] = row["kinds"].get(e["kind"], 0) + 1
        if e["kind"] in N.UTTERANCE_KINDS:
            text = e["payload"].get("text")
            if isinstance(text, str) and text:
                utt.setdefault(aid, []).append(text)
    out: list[dict] = []
    for aid in sorted(counts):
        row = counts[aid]
        texts = utt.get(aid, [])
        n_utt = len(texts)
        hit = sum(1 for t in texts if any(w in t for w in words)) if words else 0
        observed = min(last_step + 1, row["first_step"] + spd) \
            - row["first_step"]
        rec = {
            "agent": aid, "first_step": row["first_step"],
            "first_day": row["first_step"] // spd,
            "observed_steps": observed, "full_day": observed >= spd,
            "n_events": row["n_events"], "n_utterances": n_utt,
            "norm_word_use_rate": (round(hit / n_utt, 6) if n_utt else None),
            "norm_word_use_any": (1.0 if hit else 0.0) if n_utt else None,
            "cohort": (None if t_norm is None else
                       ("post" if row["first_step"] >= t_norm else "pre")),
        }
        for k in FIRST_DAY_KINDS:
            rec[f"share_{k}"] = (round(row["kinds"].get(k, 0) / row["n_events"], 6)
                                 if row["n_events"] else 0.0)
        out.append(rec)
    return out, last_step


def analyze_run(run_dir: str, min_stage: int, min_agents: int,
                require_full_day: bool = False) -> dict:
    """ラン 1 本 → 規範化ステージ・規範候補・コホート・個体別初日特徴。"""
    mcfg, msrc = load_markers(run_dir)
    res = S.norm_stages(run_dir, mcfg["markers"], mcfg["min_word_len"],
                        mcfg["max_gap"], usage=True)
    cands = N.norm_candidates(res, min_stage, min_agents, res.get("usage"))
    t_norm = cands[0]["established_step"] if cands else None
    words = [c["word"] for c in cands]
    first = S.first_presence(run_dir)
    # W2-3: 「参入初日 = 初出 step から 1 日」の 1 日はこのランの run.dt_min で決まる
    # (Δt=10 なら 144 = 従来と同値・Δt=1 なら 1440)。
    feats, last_step = first_day_features(run_dir, first, words, t_norm,
                                          run_dt.steps_per_day(run_dir))
    n_truncated = sum(1 for r in feats if not r["full_day"])
    if require_full_day:
        feats = [r for r in feats if r["full_day"]]
    cohorts = {"pre": [r for r in feats if r["cohort"] == "pre"],
               "post": [r for r in feats if r["cohort"] == "post"]}
    return {
        "dir": run_dir, "name": os.path.basename(os.path.normpath(run_dir)),
        "marker_source": msrc,
        # 初期フレーム共変量(observer.initial_frame.days>0 のランだけ入っている)。
        # ラン単位の層別軸として使えるよう、解析結果へそのまま持ち上げる。
        "initial_frame": _read_initial_frame(run_dir),
        "stages": res["summary"],
        "words": res["words"],
        "candidates": cands,
        "t_norm": t_norm,
        "n_agents": len(first),
        "n_first_steps_distinct": len(set(first.values())),
        "last_step": last_step,
        "n_truncated_first_day": n_truncated,
        "require_full_day": bool(require_full_day),
        "cohort_sizes": {k: len(v) for k, v in cohorts.items()},
        "truncated_by_cohort": {
            k: sum(1 for r in v if not r["full_day"]) for k, v in cohorts.items()},
        "features": feats,
    }


# --------------------------------------------------------------------------- #
# 下方因果(コホート間の初日行動分布の比較)
# --------------------------------------------------------------------------- #
def _metric_names(runs: list[dict]) -> list[str]:
    names = {"norm_word_use_rate", "norm_word_use_any", "n_events", "n_utterances"}
    for r in runs:
        for rec in r["features"]:
            names.update(k for k in rec if k.startswith("share_"))
    return sorted(names)


def downward_causation(runs: list[dict], n_mc: int = MC_ITER,
                       rng_seed: int = 0) -> dict:
    """pre / post コホートの初日行動分布を比較する(全ラン層別プール)。"""
    rows_a, rows_b, strata = [], [], []
    for r in runs:
        for rec in r["features"]:
            if rec["cohort"] == "pre":
                rows_a.append((r["name"], rec))
            elif rec["cohort"] == "post":
                rows_b.append((r["name"], rec))
    n_runs_used = len({n for n, _ in rows_a} & {n for n, _ in rows_b})
    tests: dict[str, dict] = {}
    if rows_a and rows_b:
        multi = n_runs_used > 1
        for metric in _metric_names(runs):
            a = [rec.get(metric) for _, rec in rows_a]
            b = [rec.get(metric) for _, rec in rows_b]
            st = ([n for n, _ in rows_a] + [n for n, _ in rows_b]) if multi else None
            tests[metric] = two_sample_perm(a, b, n_mc=n_mc, rng_seed=rng_seed,
                                            strata=st)
    return {
        "n_pre": len(rows_a), "n_post": len(rows_b),
        "n_runs_with_both_cohorts": n_runs_used,
        "stratified": n_runs_used > 1,
        "tests": tests,
        "usable": bool(rows_a and rows_b),
    }


# --------------------------------------------------------------------------- #
# レポート
# --------------------------------------------------------------------------- #
def _f(v, nd=4):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def render(result: dict) -> str:
    L: list[str] = []
    p = result["params"]
    L.append("=" * 76)
    L.append("規範化ステージ + コホート + 下方因果(第74バッチ IDEA③ / Part E)")
    L.append("=" * 76)
    L.append(f"ラン数: {len(result['runs'])} / 事前登録の判定線: "
             f"stage>={p['norm_stage']} かつ 相異なる利用者>={p['norm_threshold']} 人")
    L.append("")
    L.append("-- (1) 規範化ステージ(ラン別)" + "-" * 44)
    L.append(f"{'run':<22}{'語数':>5}{'S2':>5}{'S3':>5}{'S4':>5}"
             f"{'max':>5}{'→引用':>8}{'→合意':>8}{'別人率':>8}")
    for r in result["runs"]:
        s = r["stages"]
        L.append(f"{r['name'][:22]:<22}{s['n_words']:>5}{s['n_stage2']:>5}"
                 f"{s['n_stage3']:>5}{s['n_stage4']:>5}{s['max_stage']:>5}"
                 f"{_f(s['steps_to_quote_median'], 1):>8}"
                 f"{_f(s['steps_to_agreement_median'], 1):>8}"
                 f"{_f(s['institutionalizer_distinct_share']):>8}")
    L.append("")
    L.append("-- (2) 規範候補と成立 step" + "-" * 49)
    any_c = False
    for r in result["runs"]:
        for c in r["candidates"][:10]:
            any_c = True
            L.append(f"  {r['name'][:18]:<18} 「{c['word']}」 stage={c['stage']} "
                     f"利用者={c['n_users']} 成立step={c['established_step']} "
                     f"命名者={c['coiner']} 制度化者={c['institutionalizer']}")
    if not any_c:
        L.append("  (該当なし)判定線を満たす語が 1 つも無い。")
        L.append("  ★mock ランでは発話テンプレートに定冠詞相当・合意参照の表現が無いため")
        L.append("    S3/S4 は 0 件が正常。stage>=2 で回すか実 LLM ランで再実行すること。")
    L.append("")
    L.append("-- (3) コホート(参入初日)" + "-" * 49)
    for r in result["runs"]:
        L.append(f"  {r['name'][:22]:<22} 個体={r['n_agents']:>4} "
                 f"初出step種類={r['n_first_steps_distinct']:>4} "
                 f"pre={r['cohort_sizes']['pre']:>4} post={r['cohort_sizes']['post']:>4} "
                 f"t_norm={r['t_norm']} "
                 f"初日欠落={r['truncated_by_cohort']}")
    frames = [r for r in result["runs"] if r["initial_frame"]]
    if frames:
        L.append("")
        L.append("-- (3b) 初期フレーム共変量(層別軸。observer.initial_frame.days>0 のラン)" + "-" * 4)
        for r in frames:
            f = r["initial_frame"]
            L.append(f"  {r['name'][:22]:<22} 窓={f['days']}日 発話={f['n_utterances']:>5} "
                     f"造語={f['n_coin']:>4} マーカー率={_f(f['norm_marker_rate'])} "
                     f"L2平均={ {k: _f(v, 3) for k, v in f['l2_mean'].items()} }")
    dc = result["downward"]
    L.append("")
    L.append("-- (4) 下方因果(post − pre の平均差・2標本ラベル置換)" + "-" * 21)
    if not dc["usable"]:
        L.append("  × 群間比較ができない(pre/post のどちらかが空)。")
        L.append("    原因は次のどちらか: (a) 規範候補が成立していない")
        L.append("    (b) pool.enabled=false で全員が step 0 参入=コホートが 1 群しか無い。")
        L.append("    → Part E の下方因果は **日次 presence 選抜(pool ON)のラン**でのみ成立する。")
    else:
        L.append(f"  n_pre={dc['n_pre']} n_post={dc['n_post']} "
                 f"層別置換={'あり(ラン内)' if dc['stratified'] else 'なし(単一ラン)'}")
        L.append(f"  {'metric':<26}{'pre':>10}{'post':>10}{'diff':>10}{'p':>10}  method")
        for k in sorted(dc["tests"]):
            t = dc["tests"][k]
            L.append(f"  {k:<26}{_f(t['mean_a']):>10}{_f(t['mean_b']):>10}"
                     f"{_f(t['diff']):>10}{_f(t['p']):>10}  {t['method']}")
        L.append("")
        L.append("  読み方: norm_word_use_rate の diff>0 かつ p 小 = 成立後に参入した個体が")
        L.append("  形成に参加していないのに規範語を使っている = **円環が閉じている**証拠。")
    L.append("")
    L.append("-- 正直な限界" + "-" * 62)
    L.append("  - S3/S4 は表層文字列の一致のみ(言い換え・婉曲な参照は捉えない=過小検出)。")
    L.append("  - 検出語彙は conf の labeling.norm_stage.markers。表を替えれば結果は変わる")
    L.append(f"    (このランの取得元: {sorted({r['marker_source'] for r in result['runs']})})。")
    L.append("  - 行動分布はイベント種の構成比であって主観的な意味づけではない。")
    L.append("  - 小標本では置換検定の p に下限がある(min_p を各行の method 横で確認すること)。")
    L.append("  - ランの終端に掛かった個体は初日が丸ごと観測できていない(上の『初日欠落』)。")
    L.append("    絶対件数(n_events / n_utterances)の群間差はその分だけ歪む。")
    L.append("    --require-full-day で落とせるが、標本が減って検出力も落ちる。")
    L.append("  - ★交絡の警告: 『成立後に参入した個体ほど規範語を使う』は、規範の拘束力ではなく")
    L.append("    **単に語がもう存在していたから**でも起きる(語の可用性)。因果の主張には")
    L.append("    語の可用性を揃えた対照(同じ語彙量・別の成立時刻)が要る。mock ランの数値は")
    L.append("    配線の確認であって下方因果の証拠ではない。")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
def main() -> None:
    # Windows の既定コンソール(cp932)へ出せない字が 1 つでもあると print が例外で落ちる。
    # レポートは人間向けの読み物なので、**落とすのではなく置換**して必ず最後まで出す。
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:                                       # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pattern", help="ラン dir か glob パターン(例 'runs/endo14_*')")
    ap.add_argument("--norm-stage", type=int, required=True,
                    help="★必須(事前登録): 規範候補とみなす最小ステージ 1..4")
    ap.add_argument("--norm-threshold", type=int, required=True,
                    help="★必須(事前登録): その語を使った相異なるエージェント数の下限")
    ap.add_argument("--mc", type=int, default=MC_ITER, help="MC 置換の反復数")
    ap.add_argument("--seed", type=int, default=0, help="置換の乱数 seed(決定論)")
    ap.add_argument("--require-full-day", action="store_true",
                    help="参入初日をランの終端で切られた個体を解析から落とす")
    ap.add_argument("--out", default=None, help="JSON の書き出し先")
    ap.add_argument("--allow-tier-mismatch", action="store_true",
                    help="再現性等級が混在するラン群の比較を許可する(第72バッチのガード)")
    args = ap.parse_args()

    if not 1 <= args.norm_stage <= 4:
        raise SystemExit("--norm-stage は 1..4 のいずれか")
    if args.norm_threshold < 1:
        raise SystemExit("--norm-threshold は 1 以上")

    dirs = sorted(d for d in glob.glob(args.pattern) if os.path.isdir(d))
    if not dirs:
        raise SystemExit(f"ランが見つからない: {args.pattern}")
    R.guard_or_die(dirs, allow_mismatch=args.allow_tier_mismatch)

    runs = [analyze_run(d, args.norm_stage, args.norm_threshold,
                        require_full_day=args.require_full_day) for d in dirs]
    result = {
        "params": {"norm_stage": args.norm_stage,
                   "norm_threshold": args.norm_threshold,
                   "mc": args.mc, "seed": args.seed,
                   # W2-3: 直書き 144 ではなく**ラン由来**(Δt=10 なら 144 = 従来と同値)。
                   # 複数ランを束ねるときは先頭ランの Δt を出す(スカラー 1 個という
                   # 出力の形を変えないため)。Δt が混在するラン群の比較はそもそも
                   # 「1 日」の意味が揃わないので、ラン別の窓は各 analyze_run が
                   # 自分の Δt で正しく取っている。
                   "steps_per_day": run_dt.steps_per_day(dirs[0])},
        "runs": runs,
        "downward": downward_causation(runs, n_mc=args.mc, rng_seed=args.seed),
    }
    print(render(result))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
        print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
