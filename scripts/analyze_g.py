#!/usr/bin/env python
"""感度 g の軌跡解析(第82バッチ)= 「生まれつきか、環境からの創発か」の分散分解。

    python scripts/analyze_g.py runs/day10_P --control runs/day10_N runs/day10_F

設計正典: docs/plans/source/cognition-design-record.md **§2.7**(初期値の実験条件化)・
          **§8**(観測量)・**§9**(測定指標の扱い)。

何を出すのか
------------
§2.7:「**中核研究課題との接続**: 生まれつき = `g(0)`、創発 = `Δg`。最終的な影響力を
『初期 g の分散』と『履歴で動いた分』に**分散分解**すれば、`world-changer は生まれつきか
環境からの創発か` が数値で答えられる形になる。k 掃引より計算コストが低い」。

本スクリプトは `cognition_g.parquet`(g_update ON のランが必ず出すサイドカー)だけを
読んで、次の 3 つを決定論・乱数ゼロで出す。

① **分散分解**  Var(g_end) = Var(g0) + Var(Δg) + 2·Cov(g0, Δg)
   `share_born` = Var(g0)/Var(g_end)、`share_emergent` = Var(Δg)/Var(g_end)。
   共分散項を無視しない(負の共分散 = 「初期が高い個体ほど下がる」= 平均回帰)。
   ★これは**分解であって因果推論ではない**。「生まれつきで決まっている」の証明ではなく
     「終端のばらつきのうち初期条件で説明できる割合」に過ぎない(下の限界を参照)。

② **対数変化と順位変化**(§9「エージェントの成長・衰退を **% で測らない**。初期値が
   小さい個体で発散するため」):
   - `log_change` = log(g_end / g_0) の分布(対称で加法的)
   - `spearman` = g(0) 順位と g(end) 順位の順位相関(頑健で読み手に伝わる)
   - `resid_sd` = g_end を g0 で回帰した**残差**の SD(= 初期条件で説明できない成長)
   **単一スカラに潰さない**(§9)ので、チャンネル別と全体の両方を出す。

③ **条件間比較 P vs N vs F**(§2.7「**条件 N が鍵**」):
   `--control` に別条件のランを渡すと、g(0) の分散・Δg の分散・順位入れ替わりを並べる。
   **P ≈ N ならペルソナの中身は効いておらず異質性だけが効いている**(どちらに転んでも
   主張になる)。本スクリプトは**並べるだけで合否は付けない**(閾値は事前登録の領分)。

正直な限界(必ず読むこと)
--------------------------
- **これは g の分解であって「影響力」の分解ではない。** 最終的な影響力(語彙の伝播・
  関係次数・組織の成立)へ接続するには、g_end と各影響力指標の関係を別途測る必要がある。
  本スクリプトはその手前の「注意の配分がどれだけ動いたか」までしか言わない。
- **Var(g0) は実験条件が直接決めている量である**(F なら 0、N なら σ₀、P ならペルソナ写像)。
  したがって `share_born` を単独ランで読んでも「生まれつきの寄与」にはならない。
  **意味を持つのは条件間の比較だけ**(だから --control がある)。
- Δg は g_update の η/λ/ρ に強く依存する。パラメータ較正(第83)前の値は暫定である。
- 全 usable チャンネルを対象にする。σ=0 で除外されたチャンネルはそもそも列に出ない。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pyarrow.parquet as pq

SCHEMA = 1
STEM = "cognition_g"


# --------------------------------------------------------------------------- #
# 統計(stdlib だけ。numpy に依存しない = 解析環境を選ばない)
# --------------------------------------------------------------------------- #
def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _var(xs: list[float]) -> float:
    """母分散(N で割る)。個体の集合そのものが母集団なので不偏補正はしない。"""
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)


def _cov(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    mx, my = _mean(xs), _mean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / len(xs)


def _ranks(xs: list[float]) -> list[float]:
    """同順位は平均順位(標準的な Spearman の tie 処理)。"""
    order = sorted(range(len(xs)), key=lambda i: (xs[i], i))
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    vx, vy = _var(rx), _var(ry)
    if vx <= 0.0 or vy <= 0.0:
        return None                       # 定数列(条件 F の g(0) など)= 順位が定義できない
    return _cov(rx, ry) / math.sqrt(vx * vy)


def _resid_sd(g0: list[float], gend: list[float]) -> float | None:
    """g_end を g0 で単回帰した残差の SD(§9「初期条件で説明できない成長」)。"""
    v0 = _var(g0)
    if len(g0) < 3:
        return None
    if v0 <= 0.0:                          # 条件 F: 回帰の説明変数が定数 = 残差 = 素の SD
        return math.sqrt(_var(gend))
    beta = _cov(g0, gend) / v0
    alpha = _mean(gend) - beta * _mean(g0)
    res = [y - (alpha + beta * x) for x, y in zip(g0, gend)]
    return math.sqrt(_var(res))


def _log_change(g0: float, gend: float) -> float | None:
    """§9: 対数変化 log(final/initial)。0 以下は定義できないので None。"""
    if g0 <= 0.0 or gend <= 0.0:
        return None
    return math.log(gend / g0)


# --------------------------------------------------------------------------- #
# 読み込み
# --------------------------------------------------------------------------- #
def load_run(run_dir: Path) -> dict:
    """`cognition_g.parquet` から `{agent_id: {ch: (g0, g_end)}}` を作る。

    g(0) は列 `g0_*`(初期化時に凍結した g_i(0) をそのまま毎行に持っている)、
    g(end) は各個体の**最終行**の `g_*`。行の走査は step 昇順で決定論。
    """
    path = Path(run_dir) / f"{STEM}.parquet"
    if not path.exists():
        raise SystemExit(
            f"g/θ 軌跡サイドカーが無い: {path}\n"
            f"  cognition.g_update.enabled=true かつ log_every_steps>0 のランが必要")
    table = pq.read_table(path)
    names = list(table.column_names)
    g_cols = [c for c in names if c.startswith("g_")]
    g0_cols = [c for c in names if c.startswith("g0_")]
    chans = [c[2:] for c in g_cols]
    if [c[3:] for c in g0_cols] != chans:
        raise SystemExit(f"列の対応が壊れている: {g_cols} / {g0_cols}")
    rows = table.to_pylist()
    if not rows:
        raise SystemExit(f"サイドカーが空: {path}")
    first: dict[int, dict] = {}
    last: dict[int, dict] = {}
    steps: list[int] = []
    for row in sorted(rows, key=lambda r: (int(r["step"]), int(r["agent_id"]))):
        aid = int(row["agent_id"])
        steps.append(int(row["step"]))
        if aid not in first:
            first[aid] = row
        last[aid] = row
    return {
        "dir": str(run_dir), "channels": chans, "n_agents": len(last),
        "step_first": min(steps), "step_last": max(steps),
        "g0": {aid: [float(first[aid][f"g0_{c}"]) for c in chans] for aid in last},
        "gend": {aid: [float(last[aid][f"g_{c}"]) for c in chans] for aid in last},
        "theta_end": [float(last[aid]["theta_mult"]) for aid in sorted(last)],
        "fbar_end": [float(last[aid]["fbar"]) for aid in sorted(last)],
        "mode": _mode_of(run_dir),
    }


def _mode_of(run_dir: Path) -> str:
    """run_manifest.json から初期値条件(F/N/P)を読む(無ければ "unknown")。"""
    man = Path(run_dir) / "run_manifest.json"
    if not man.exists():
        return "unknown"
    try:
        doc = json.loads(man.read_text(encoding="utf-8"))
        return str(doc["cognition"]["g_update"]["g_init"]["mode"])
    except (KeyError, TypeError, ValueError, OSError):
        return "unknown"


# --------------------------------------------------------------------------- #
# 分散分解(§2.7 / §8)
# --------------------------------------------------------------------------- #
def decompose(run: dict) -> dict:
    """チャンネル別 + 全体(チャンネル平均 g)の分散分解と §9 の指標。"""
    aids = sorted(run["gend"])
    chans = run["channels"]
    per_channel = {}
    for j, ch in enumerate(chans):
        g0 = [run["g0"][a][j] for a in aids]
        ge = [run["gend"][a][j] for a in aids]
        per_channel[ch] = _one(g0, ge)
    # 全体 = 個体ごとのチャンネル平均 g(**単一スカラに潰さない**ので併記であって代表値ではない)
    g0m = [_mean(run["g0"][a]) for a in aids]
    gem = [_mean(run["gend"][a]) for a in aids]
    return {"per_channel": per_channel, "channel_mean": _one(g0m, gem),
            "n_agents": len(aids)}


def _one(g0: list[float], gend: list[float]) -> dict:
    dg = [b - a for a, b in zip(g0, gend)]
    v0, vd, vend = _var(g0), _var(dg), _var(gend)
    cov = _cov(g0, dg)
    logs = [x for x in (_log_change(a, b) for a, b in zip(g0, gend)) if x is not None]
    return {
        "var_g0": v0, "var_dg": vd, "cov_g0_dg": cov, "var_gend": vend,
        # Var(g_end) = Var(g0) + Var(Δg) + 2·Cov(g0, Δg)。恒等式が成り立つことを出力で確かめられる。
        "identity_residual": vend - (v0 + vd + 2.0 * cov),
        "share_born": (v0 / vend) if vend > 0 else None,
        "share_emergent": (vd / vend) if vend > 0 else None,
        "share_cov": (2.0 * cov / vend) if vend > 0 else None,
        "mean_g0": _mean(g0), "mean_gend": _mean(gend), "mean_dg": _mean(dg),
        # §9: % ではなく対数変化・順位・残差で測る
        "log_change_mean": _mean(logs) if logs else None,
        "log_change_sd": math.sqrt(_var(logs)) if len(logs) > 1 else None,
        "spearman_g0_gend": _spearman(g0, gend),
        "resid_sd": _resid_sd(g0, gend),
    }


def compare(runs: list[dict], results: list[dict]) -> dict:
    """条件間の並置(§2.7「P ≈ N ならペルソナの中身は効いていない」)。

    ★合否は付けない。閾値つきの判定は事前登録の領分(analyze_specialization と同じ流儀)。
    """
    rows = []
    for run, res in zip(runs, results):
        cm = res["channel_mean"]
        rows.append({
            "dir": run["dir"], "mode": run["mode"], "n_agents": run["n_agents"],
            "var_g0": cm["var_g0"], "var_dg": cm["var_dg"],
            "share_born": cm["share_born"], "share_emergent": cm["share_emergent"],
            "spearman_g0_gend": cm["spearman_g0_gend"],
            "log_change_sd": cm["log_change_sd"], "resid_sd": cm["resid_sd"],
        })
    by_mode = {r["mode"]: r for r in rows}
    note = None
    if "persona" in by_mode and "noise" in by_mode:
        p, n = by_mode["persona"], by_mode["noise"]
        note = ("P と N の Var(Δg) / 順位相関を比べる。近ければ『ペルソナの中身ではなく"
                "異質性そのものが効いている』読み、離れていれば『ペルソナ整合的な異質性が"
                "効いている』読み。**この 1 対だけで断定しない**(seed 複数本が必要)。")
        note += (f" [Var(Δg) P={p['var_dg']:.6g} / N={n['var_dg']:.6g}]")
    return {"rows": rows, "reading": note}


# --------------------------------------------------------------------------- #
# 出力
# --------------------------------------------------------------------------- #
def render(res: dict) -> str:
    out = ["# 感度 g の軌跡: 生まれつき(g(0))と創発(Δg)の分散分解", ""]
    for run in res["runs"]:
        cm = run["decomposition"]["channel_mean"]
        out += [f"## {run['dir']}  (条件 = {run['mode']} / n={run['n_agents']} / "
                f"step {run['step_first']}..{run['step_last']})", ""]
        out.append("| 指標 | 値 |")
        out.append("|---|---|")
        for key in ("var_g0", "var_dg", "cov_g0_dg", "var_gend",
                    "share_born", "share_emergent", "share_cov",
                    "log_change_mean", "log_change_sd",
                    "spearman_g0_gend", "resid_sd"):
            v = cm[key]
            out.append(f"| {key} | {'—' if v is None else f'{v:.6g}'} |")
        out.append("")
        out.append("チャンネル別 Var(Δg)(§8「分散の拡大 = 役割分化の一次証拠」):")
        out.append("")
        out.append("| channel | var_g0 | var_dg | spearman | log_change_sd |")
        out.append("|---|---|---|---|---|")
        for ch, cell in sorted(run["decomposition"]["per_channel"].items()):
            def _f(x):
                return "n/a" if x is None else f"{x:.5g}"
            out.append(f"| {ch} | {_f(cell['var_g0'])} | {_f(cell['var_dg'])} | "
                       f"{_f(cell['spearman_g0_gend'])} | {_f(cell['log_change_sd'])} |")
        out.append("")
    if res.get("comparison") and res["comparison"].get("reading"):
        out += ["## 条件間の並置(F / N / P)", "",
                res["comparison"]["reading"], ""]
        out.append("| dir | mode | var_g0 | var_dg | share_emergent | spearman |")
        out.append("|---|---|---|---|---|---|")
        for r in res["comparison"]["rows"]:
            def _f(x):
                return "n/a" if x is None else f"{x:.5g}"
            out.append(f"| {r['dir']} | {r['mode']} | {_f(r['var_g0'])} | "
                       f"{_f(r['var_dg'])} | {_f(r['share_emergent'])} | "
                       f"{_f(r['spearman_g0_gend'])} |")
        out.append("")
    out += ["## 限界(必ず併記する)", "",
            "- これは **g の分解であって影響力の分解ではない**。",
            "- `Var(g0)` は実験条件が直接決める量なので、**単独ランの share_born は"
            "『生まれつきの寄与』を意味しない**。意味を持つのは条件間の比較だけ。",
            "- Δg は η/λ/ρ に依存する。第83の較正前の値は暫定。", ""]
    return "\n".join(out)


def main() -> int:
    # Windows の既定コンソール(cp932)でも表を印字できるようにする。
    # ファイル出力は常に UTF-8 なので、置換が起きるのは端末表示だけ。
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):        # 端末以外へリダイレクト時
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+", help="解析するランの出力ディレクトリ")
    ap.add_argument("--control", nargs="*", default=[],
                    help="比較する別条件のラン(F/N/P の並置に使う)")
    ap.add_argument("--out", default=None, help="出力先(既定: 先頭ラン)")
    args = ap.parse_args()

    dirs = [Path(d) for d in list(args.run_dirs) + list(args.control or [])]
    runs = [load_run(d) for d in dirs]
    results = [decompose(r) for r in runs]
    res = {
        "schema": SCHEMA,
        "runs": [{"dir": r["dir"], "mode": r["mode"], "n_agents": r["n_agents"],
                  "channels": r["channels"], "step_first": r["step_first"],
                  "step_last": r["step_last"], "decomposition": d}
                 for r, d in zip(runs, results)],
        "comparison": compare(runs, results) if len(runs) > 1 else None,
    }
    out_dir = Path(args.out or dirs[0])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "g_trajectory.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    md = render(res)
    (out_dir / "g_trajectory_report.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"[written] {out_dir / 'g_trajectory.json'}")
    print(f"[written] {out_dir / 'g_trajectory_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
