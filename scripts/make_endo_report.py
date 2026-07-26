#!/usr/bin/env python
"""第63バッチ フェーズ2: endogenous_accept treatment 比較の条件比較ビュー(スタンドアロン HTML)。

    python scripts/make_endo_report.py runs/_endo_treatment
    python scripts/make_endo_report.py runs/_endo_treatment --out runs/_endo_treatment/endo_report.html

analyze_endo_treatment.py の出力(endo_treatment.json)だけを読み、自己完結 HTML を生成する
(外部 CDN・JS ライブラリなし=analyze_flows_grid.py / viz/make_viewer と同じ方針。既存
make_viewer には触らない=旧ランのバイト同一を守る)。描画は素の SVG。

内容:
  1. 仮説判定チップ(H1/H2/H3・承諾率乖離ゲート・フェーズ3 GO/NO-GO)+検出力の正直な注記
  2. ペア差スロープチャート(セル×指標): 同一 seed の OFF→ON を線で結ぶ(CRN ペアの向きが
     ひと目で分かる)+ 平均差・sign-flip p(主検定)・block p(副検定)の注記
  3. 時系列重ね描き(k 水準ごと): OFF 平均(破線・橙)vs ON 平均(実線・青)の日次系列
色は Okabe-Ito(色覚多様性セーフ・analyze_sweep と同一パレット)+ 線種の冗長符号化
(破線/実線)=色だけに頼らない。ホバーは SVG <title>(ネイティブツールチップ)。
決定論: 入力 JSON が同じなら出力 HTML はバイト同一(乱数・時刻を書かない)。
依存: 標準ライブラリのみ(pandas/duckdb 禁止どころか pyarrow も不要=JSON を読むだけ)。
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Okabe-Ito(analyze_sweep.PALETTE と同値。ON=青 / OFF=橙 + 線種で冗長符号化)
C_ON = "#0072B2"
C_OFF = "#E69F00"
C_INK = "#e6edf3"
C_DIM = "#9aa4b2"
C_LINE = "#30363d"
C_GOOD = "#009E73"
C_BAD = "#D55E00"

STRUCT_METRICS = ("edge_churn_rate", "community_change_rate",
                  "centrality_turnover", "rank_tau_prev_day", "stagnant_days")
SERIES_METRICS = ("edge_churn_rate", "community_change_rate",
                  "centrality_turnover", "rank_tau_prev_day")
METRIC_JA = {
    "edge_churn_rate": "edge 組み替え率",
    "community_change_rate": "コミュニティ変化率",
    "centrality_turnover": "中心性 turnover",
    "rank_tau_prev_day": "順位 τ(前日比)",
    "stagnant_days": "固着延べ日数",
}


def _f(v, nd=4):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}".rstrip("0").rstrip(".") if v == v else "—"
    return str(v)


def _esc(s) -> str:
    return html.escape(str(s), quote=True)


def _scale(vals: list[float], lo_px: float, hi_px: float):
    """値→px の線形スケール(値域ゼロは中央固定)。返り値は関数。"""
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        pad = abs(hi) * 0.5 + 1e-6
        lo, hi = lo - pad, hi + pad
    span = hi - lo
    lo -= span * 0.08
    hi += span * 0.08

    def f(v: float) -> float:
        return lo_px + (hi_px - lo_px) * (v - lo) / (hi - lo)
    return f, lo, hi


# --------------------------------------------------------------------------- #
# SVG: ペア差スロープチャート(1 セル=1 チャート。x= OFF/ON の2位置)
# --------------------------------------------------------------------------- #
def slope_chart(k: str, metric: str, per_seed: dict, test: dict,
                block: dict | None) -> str:
    W, H = 300, 190
    PL, PR, PT, PB = 46, 14, 26, 30
    pts = [(s, v["off"], v["on"]) for s, v in sorted(per_seed.items(),
                                                    key=lambda x: int(x[0]))
           if v["off"] is not None and v["on"] is not None]
    if not pts:
        return (f'<svg class="ch" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
                f'<text x="{W/2}" y="{H/2}" fill="{C_DIM}" text-anchor="middle" '
                f'font-size="11">ペアなし({_esc(k)})</text></svg>')
    vals = [v for _s, o, n in pts for v in (o, n)]
    sy, _lo, _hi = _scale(vals, H - PB, PT)
    x0, x1 = PL + 34, W - PR - 34
    el = [f'<svg class="ch" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
          f'role="img" aria-label="{_esc(METRIC_JA.get(metric, metric))} {_esc(k)}">']
    # 軸・目盛(recessive)
    for frac in (0.0, 0.5, 1.0):
        yv = _lo + (_hi - _lo) * frac
        yp = sy(yv)
        el.append(f'<line x1="{PL}" y1="{yp:.1f}" x2="{W-PR}" y2="{yp:.1f}" '
                  f'stroke="{C_LINE}" stroke-width="1"/>')
        el.append(f'<text x="{PL-4}" y="{yp+3.5:.1f}" fill="{C_DIM}" '
                  f'font-size="9" text-anchor="end">{_f(yv, 3)}</text>')
    el.append(f'<text x="{x0}" y="{H-10}" fill="{C_DIM}" font-size="10" '
              f'text-anchor="middle">OFF</text>')
    el.append(f'<text x="{x1}" y="{H-10}" fill="{C_DIM}" font-size="10" '
              f'text-anchor="middle">ON</text>')
    for s, o, n in pts:
        yo, yn = sy(o), sy(n)
        col = C_GOOD if n > o else (C_BAD if n < o else C_DIM)
        tip = f"seed {s}: OFF {_f(o)} → ON {_f(n)}(差 {_f(n - o)})"
        el.append(f'<g><title>{_esc(tip)}</title>'
                  f'<line x1="{x0}" y1="{yo:.1f}" x2="{x1}" y2="{yn:.1f}" '
                  f'stroke="{col}" stroke-width="2" stroke-opacity="0.85"/>'
                  f'<circle cx="{x0}" cy="{yo:.1f}" r="4" fill="{C_OFF}"/>'
                  f'<circle cx="{x1}" cy="{yn:.1f}" r="4" fill="{C_ON}"/></g>')
    md = test.get("mean_diff")
    p = test.get("p")
    bp = (block or {}).get("p")
    lab = f"k={k}  Δ={_f(md)}  p={_f(p)}" + (f"  block p={_f(bp)}" if bp is not None else "")
    el.append(f'<text x="{PL}" y="14" fill="{C_INK}" font-size="10">{_esc(lab)}</text>')
    el.append("</svg>")
    return "".join(el)


# --------------------------------------------------------------------------- #
# SVG: 時系列重ね描き(k 水準ごと: OFF 平均=破線橙 / ON 平均=実線青)
# --------------------------------------------------------------------------- #
def _mean_series(runs: dict, k: str, endo: str, metric: str) -> list:
    series = [r["series"][metric] for r in runs.values()
              if r["k"] == k and r["endo"] == endo and metric in r["series"]]
    if not series:
        return []
    T = max(len(s) for s in series)
    out = []
    for t in range(T):
        vs = [s[t] for s in series if t < len(s) and s[t] is not None]
        out.append(round(sum(vs) / len(vs), 6) if vs else None)
    return out


def series_chart(k: str, metric: str, off_s: list, on_s: list) -> str:
    W, H = 300, 170
    PL, PR, PT, PB = 46, 12, 22, 26
    all_vals = [v for s in (off_s, on_s) for v in s if v is not None]
    if not all_vals:
        return ""
    T = max(len(off_s), len(on_s))
    sy, _lo, _hi = _scale(all_vals, H - PB, PT)

    def sx(t: float) -> float:
        return PL + (W - PL - PR) * (t / max(T - 1, 1))

    el = [f'<svg class="ch" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
          f'role="img" aria-label="daily {_esc(metric)} {_esc(k)}">']
    for frac in (0.0, 0.5, 1.0):
        yv = _lo + (_hi - _lo) * frac
        yp = sy(yv)
        el.append(f'<line x1="{PL}" y1="{yp:.1f}" x2="{W-PR}" y2="{yp:.1f}" '
                  f'stroke="{C_LINE}" stroke-width="1"/>')
        el.append(f'<text x="{PL-4}" y="{yp+3.5:.1f}" fill="{C_DIM}" '
                  f'font-size="9" text-anchor="end">{_f(yv, 3)}</text>')
    for d in range(T):
        if d % max(1, T // 7) == 0:
            el.append(f'<text x="{sx(d):.1f}" y="{H-8}" fill="{C_DIM}" '
                      f'font-size="9" text-anchor="middle">D{d}</text>')

    def path(series: list, color: str, dash: str, name: str) -> str:
        segs, cur = [], []
        for t, v in enumerate(series):
            if v is None:
                if cur:
                    segs.append(cur)
                cur = []
            else:
                cur.append((sx(t), sy(v), t, v))
        if cur:
            segs.append(cur)
        out = []
        for seg in segs:
            dstr = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y, _t, _v in seg)
            out.append(f'<path d="{dstr}" fill="none" stroke="{color}" '
                       f'stroke-width="2"{dash}/>')
            for x, y, t, v in seg:
                out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" '
                           f'fill="{color}"><title>'
                           f'{_esc(f"{name} D{t}: {_f(v)}")}</title></circle>')
        return "".join(out)

    el.append(path(off_s, C_OFF, ' stroke-dasharray="5,4"', "OFF"))
    el.append(path(on_s, C_ON, "", "ON"))
    el.append(f'<text x="{PL}" y="13" fill="{C_INK}" font-size="10">k={_esc(k)}</text>')
    el.append("</svg>")
    return "".join(el)


# --------------------------------------------------------------------------- #
# HTML 本体
# --------------------------------------------------------------------------- #
def build_html(result: dict) -> str:
    runs = result["runs"]
    pairs = result["pairs"]
    tests = result["tests"]
    kpi = result["kpi"]
    jd = result["judgment"]
    ks = result["k_order"]

    def chip(label: str, ok: bool | None, detail: str = "") -> str:
        cls = "ok" if ok else "ng"
        mark = "成立" if ok else "不成立"
        return (f'<span class="chip {cls}" title="{_esc(detail)}">'
                f'{_esc(label)}: {mark}</span>')

    parts = []
    parts.append(f"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>endogenous_accept 条件比較 {_esc(result.get('experiment', ''))}</title>
<style>
  :root{{--bg:#0d1117;--panel:#161b22;--ink:{C_INK};--dim:{C_DIM};--line:{C_LINE};}}
  *{{box-sizing:border-box;margin:0;}}
  body{{background:var(--bg);color:var(--ink);
    font-family:system-ui,-apple-system,"Hiragino Sans","Yu Gothic UI","Segoe UI",sans-serif;}}
  header{{padding:12px 16px;border-bottom:1px solid var(--line);}}
  h1{{font-size:15px;font-weight:600;}} h2{{font-size:13px;margin:14px 0 8px;}}
  h3{{font-size:12px;color:var(--dim);margin:10px 0 6px;font-weight:600;}}
  .sub{{color:var(--dim);font-size:12px;margin-top:3px;}}
  main{{padding:14px 16px;max-width:1400px;}}
  .card{{background:var(--panel);border:1px solid var(--line);border-radius:10px;
    padding:12px 14px;margin-bottom:14px;}}
  .chip{{display:inline-block;border-radius:999px;padding:3px 10px;font-size:12px;
    margin:2px 6px 2px 0;border:1px solid var(--line);}}
  .chip.ok{{background:rgba(0,158,115,.15);color:{C_GOOD};}}
  .chip.ng{{background:rgba(213,94,0,.15);color:{C_BAD};}}
  .row{{display:flex;flex-wrap:wrap;gap:10px;}}
  .ch{{background:#0a0e14;border:1px solid var(--line);border-radius:8px;}}
  table{{border-collapse:collapse;font-size:12px;margin:6px 0;}}
  th,td{{border:1px solid var(--line);padding:4px 8px;text-align:right;}}
  th{{color:var(--dim);font-weight:600;}} td:first-child,th:first-child{{text-align:left;}}
  .legend{{font-size:11px;color:var(--dim);margin:4px 0 8px;}}
  .legend .sw{{display:inline-block;width:18px;height:0;border-top:2px solid;
    vertical-align:middle;margin:0 4px 0 10px;}}
  .note{{font-size:11px;color:var(--dim);margin-top:6px;}}
</style></head><body>
<header><h1>endogenous_accept treatment 比較(6セル・CRN seedペア)</h1>
<div class="sub">{_esc(result.get('experiment', ''))} — ラン {len(runs)} 本 / k 水準 {_esc(', '.join(ks))} /
生成: scripts/make_endo_report.py(入力 endo_treatment.json)</div></header><main>""")

    # 1. 判定チップ
    gg = jd["accept_gap_gate"]
    parts.append('<div class="card"><h2>仮説の機械判定(計画書§3)</h2>')
    for h in ("H1", "H2", "H3"):
        parts.append(chip(h, jd[h]["pass"], jd[h]["hypothesis"]))
    parts.append(chip(f"承諾率乖離 ±{gg['threshold_pp']:.0f}pp", gg["pass"],
                      "ON セルの joint_accept_calib_gap ゲート"))
    go = jd["phase3_go"]
    parts.append(f'<span class="chip {"ok" if go else "ng"}">フェーズ3: '
                 f'{"GO" if go else "NO-GO"}</span>')
    parts.append(f'<div class="note">⚠ {_esc(jd["power_note"])}</div></div>')

    # 2. ペア差スロープチャート
    parts.append('<div class="card"><h2>ペア差(ON−OFF・同一 seed を線で結ぶ)</h2>'
                 '<div class="legend">● <span style="color:#E69F00">OFF</span>'
                 f'<span class="sw" style="border-color:{C_OFF}"></span>'
                 '→ ● <span style="color:#0072B2">ON</span>'
                 f'<span class="sw" style="border-color:{C_ON}"></span>'
                 f'/ 線色: <span style="color:{C_GOOD}">増</span>・'
                 f'<span style="color:{C_BAD}">減</span>。Δ=平均差、p=sign-flip(両側)</div>')
    for metric in STRUCT_METRICS:
        parts.append(f'<h3>{_esc(METRIC_JA.get(metric, metric))} ({_esc(metric)})</h3>'
                     '<div class="row">')
        for k in ks:
            per = pairs.get(k, {}).get(metric, {})
            t = tests["pair"].get(k, {}).get(metric, {})
            b = tests.get("block", {}).get(k, {}).get(metric)
            parts.append(slope_chart(k, metric, per, t, b))
        parts.append("</div>")
    parts.append("</div>")

    # 3. 時系列重ね描き
    parts.append('<div class="card"><h2>日次時系列の重ね描き(seed 平均)</h2>'
                 f'<div class="legend"><span class="sw" style="border-color:{C_OFF};'
                 'border-top-style:dashed"></span>OFF(破線・橙)'
                 f'<span class="sw" style="border-color:{C_ON}"></span>ON(実線・青)</div>')
    for metric in SERIES_METRICS:
        parts.append(f'<h3>{_esc(METRIC_JA.get(metric, metric))} ({_esc(metric)})</h3>'
                     '<div class="row">')
        for k in ks:
            off_s = _mean_series(runs, k, "off", metric)
            on_s = _mean_series(runs, k, "on", metric)
            svg = series_chart(k, metric, off_s, on_s)
            if svg:
                parts.append(svg)
        parts.append("</div>")
    parts.append("</div>")

    # 4. 検定・KPI テーブル(表ビュー=色に依存しない読み口)
    parts.append('<div class="card"><h2>検定テーブル(主=sign-flip / 副=block)</h2>'
                 '<table><tr><th>種</th><th>範囲</th><th>指標</th><th>n</th>'
                 '<th>平均差</th><th>+/−</th><th>p</th><th>法</th></tr>')
    for kind in ("pair", "interaction", "block"):
        for scope in sorted(tests.get(kind, {})):
            for metric in sorted(tests[kind][scope]):
                t = tests[kind][scope][metric]
                parts.append(
                    f'<tr><td>{kind}</td><td>{_esc(scope)}</td>'
                    f'<td>{_esc(metric)}</td>'
                    f'<td>{t.get("n", t.get("n_obs", 0)) or 0}</td>'
                    f'<td>{_f(t.get("mean_diff"))}</td>'
                    f'<td>{t.get("n_pos", "—")}/{t.get("n_neg", "—")}</td>'
                    f'<td>{_f(t.get("p"))}</td>'
                    f'<td>{_esc(t.get("method", ""))}</td></tr>')
    parts.append("</table>")

    parts.append('<h2>フェーズ1 KPI(ON セルのみ)</h2>'
                 '<table><tr><th>k</th><th>accept_rate</th><th>endo_share</th>'
                 '<th>calib_gap</th><th>fulfill_rate</th></tr>')
    for k in ks:
        kv = kpi.get(k, {})
        parts.append(f'<tr><td>{_esc(k)}</td>'
                     f'<td>{_f(kv.get("joint_accept_rate"))}</td>'
                     f'<td>{_f(kv.get("joint_endo_share"))}</td>'
                     f'<td>{_f(kv.get("joint_accept_calib_gap"))}</td>'
                     f'<td>{_f(kv.get("joint_fulfill_rate"))}</td></tr>')
    parts.append('</table><div class="note">OFF セルは joint_invite を記録しない'
                 '(較正抽選のまま)ため KPI は ON 水準の報告。mock では fallback 支配'
                 '(endo_share ≈ 0.03)=実 LLM で初めて basis 分布が本格化する。</div></div>')

    parts.append("</main></body></html>")
    return "".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="endogenous_accept treatment 比較の自己完結 HTML レポート")
    ap.add_argument("results_dir",
                    help="analyze_endo_treatment.py の出力ディレクトリ(endo_treatment.json を含む)")
    ap.add_argument("--out", default=None,
                    help="出力 HTML パス(既定 <results_dir>/endo_report.html)")
    args = ap.parse_args()
    src = os.path.join(args.results_dir, "endo_treatment.json")
    if not os.path.isfile(src):
        raise SystemExit(f"not found: {src}(先に analyze_endo_treatment.py を実行)")
    with open(src, encoding="utf-8") as fh:
        result = json.load(fh)
    out = args.out or os.path.join(args.results_dir, "endo_report.html")
    html_text = build_html(result)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(html_text)
    print(f"[endo-report] {len(result.get('runs', {}))} runs -> {out} "
          f"({os.path.getsize(out) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
