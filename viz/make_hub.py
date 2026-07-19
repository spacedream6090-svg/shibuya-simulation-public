#!/usr/bin/env python
"""可視化統合ハブ — ラン直下の既存成果物を 1 枚のタブ型 HTML にまとめる。

使い方:
    python viz/make_hub.py runs/<name>            -> runs/<name>/hub.html
    python viz/make_hub.py runs/<name> --out X    -> 任意の出力先

ラン直下の成果物(3D/2D ビューア・ダッシュボード・ヒートマップ・OD・群集デモ・
要約 md)を自動検出し、上部タブで切り替えられる自己完結 HTML を書く。
  - 依存ゼロ・自己完結 CSS(外部 CDN/JS/フォント無し)
  - タブ本体は <iframe src="同フォルダ相対パス">。file:// でも同フォルダの
    iframe は開ける(fetch は使わない)。巨大ファイル対策で src は初回表示時に
    遅延セットする(data-src)。
  - summary_ja.md は事前に簡易 HTML(見出し/箇条書き/太字/水平線)へ変換して埋め込み
  - 見た目は既存 3D ビューアのダークテーマ(#0a0e14 系)に合わせる

依存: 標準ライブラリのみ(pandas/duckdb 不使用)。
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

# ダークテーマ(viz/make_viewer3d.py と同系統)
BG = "#0a0e14"
PANEL_BG = "rgba(18,22,30,.82)"
FG = "#e6e9ee"
MUTED = "#9aa4b2"
ACCENT = "#3b82f6"
BORDER = "rgba(255,255,255,.09)"
FONT = '-apple-system,"Segoe UI","Hiragino Kaku Gothic ProN",Meiryo,sans-serif'

# 検出候補: (タブ名, [相対パス候補(先勝ち)], 種別)  種別: "iframe" | "md"
CANDIDATES = [
    ("3Dビュー", ["viewer3d.html", "viewer3d_lite.html"], "iframe"),
    ("2Dビュー", ["viewer.html"], "iframe"),
    ("ダッシュボード", ["dashboard.html"], "iframe"),
    ("ヒートマップ", ["heatmap.html"], "iframe"),
    ("OD人流", ["od_flowmap.html"], "iframe"),
    ("群集デモ", ["panel/crowd_demo.html"], "iframe"),
    ("要約", ["panel/summary_ja.md"], "md"),
]


# ── 簡易 Markdown → HTML(見出し/箇条書き/太字/水平線のみ・依存ゼロ) ──────
def md_to_html(md: str) -> str:
    """summary_ja.md 相当の簡易 md を安全な HTML 断片へ。

    対応: # 見出し(h1-h3)、- 箇条書き、**太字**、`コード`、--- 水平線、段落。
    それ以外はエスケープしたプレーン段落として扱う。
    """
    def inline(text: str) -> str:
        # 先に HTML エスケープ → その後にマーカーを復元(XSS 防止)
        text = html.escape(text)
        text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
        text = re.sub(r"`([^`]+?)`", r"<code>\1</code>", text)
        return text

    lines = md.splitlines()
    out: list[str] = []
    in_list = False

    def close_list():
        nonlocal in_list
        if in_list:
            out.append("</ul>")
            in_list = False

    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            close_list()
            continue
        if re.match(r"^-{3,}$", stripped):        # 水平線 ---
            close_list()
            out.append("<hr>")
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)  # 見出し
        if m:
            close_list()
            level = min(len(m.group(1)), 3)
            out.append(f"<h{level}>{inline(m.group(2))}</h{level}>")
            continue
        m = re.match(r"^[-*]\s+(.*)$", stripped)      # 箇条書き
        if m:
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{inline(m.group(1))}</li>")
            continue
        close_list()
        out.append(f"<p>{inline(stripped)}</p>")     # 段落
    close_list()
    return "\n".join(out)


# ── 検出 ──────────────────────────────────────────────────────────────────
def detect_artifacts(run_dir: Path) -> list[dict]:
    """存在する成果物のみを検出順に返す。

    各要素: {"label", "rel"(相対パス), "kind"("iframe"|"md")}。
    """
    found = []
    for label, rels, kind in CANDIDATES:
        for rel in rels:
            if (run_dir / rel).exists():
                found.append({"label": label, "rel": rel, "kind": kind})
                break
    return found


def read_summary(run_dir: Path) -> dict:
    """summary.json を読む(無ければ空 dict)。"""
    path = run_dir / "summary.json"
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _fmt_int(v) -> str:
    try:
        return f"{int(v):,}"
    except (TypeError, ValueError):
        return "-"


# ── HTML 組み立て ─────────────────────────────────────────────────────────
def build_hub_html(run_dir: Path, run_name: str) -> tuple[str, list[dict]]:
    """hub.html の文字列と検出結果を返す。"""
    artifacts = detect_artifacts(run_dir)
    summary = read_summary(run_dir)

    # ヘッダのラン概要
    n_steps = summary.get("n_steps")
    days = None
    if isinstance(n_steps, int) and n_steps > 0:
        days = max(1, round(n_steps / 144))
    stats = [
        ("エージェント", _fmt_int(summary.get("n_agents"))),
        ("ステップ", _fmt_int(n_steps)),
        ("推定日数", str(days) if days else "-"),
        ("イベント", _fmt_int(summary.get("n_events"))),
        ("イベント種別", _fmt_int(len(summary.get("event_kinds", {})) or None)),
        ("LLM 呼数", _fmt_int(summary.get("llm_calls"))),
    ]
    stat_html = "".join(
        f'<div class="stat"><div class="v">{html.escape(v)}</div>'
        f'<div class="k">{html.escape(k)}</div></div>'
        for k, v in stats
    )

    # タブとパネル
    tab_btns = []
    panels = []
    for i, a in enumerate(artifacts):
        active = " active" if i == 0 else ""
        label = html.escape(a["label"])
        tab_btns.append(
            f'<button class="tab{active}" data-panel="panel-{i}" '
            f'onclick="showPanel({i})">{label}</button>'
        )
        if a["kind"] == "iframe":
            rel = html.escape(a["rel"], quote=True)
            # 巨大ファイル対策: active タブのみ即 src、他は data-src で遅延
            src_attr = f'src="{rel}"' if i == 0 else f'data-src="{rel}"'
            panels.append(
                f'<div class="panel{active}" id="panel-{i}">'
                f'<iframe {src_attr} loading="lazy"></iframe></div>'
            )
        else:  # md
            try:
                md_text = (run_dir / a["rel"]).read_text(encoding="utf-8")
            except OSError:
                md_text = ""
            body = md_to_html(md_text)
            panels.append(
                f'<div class="panel{active}" id="panel-{i}">'
                f'<div class="md">{body}</div></div>'
            )

    if not artifacts:
        panels.append('<div class="panel active" id="panel-empty">'
                      '<div class="md"><p>検出できる成果物がありません。</p></div></div>')

    tabs_html = "\n".join(tab_btns) if tab_btns else '<span class="muted">タブなし</span>'
    panels_html = "\n".join(panels)

    doc = f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>可視化ハブ — {html.escape(run_name)}</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  html,body {{ width:100%; height:100%; overflow:hidden; background:{BG};
    color:{FG}; font-family:{FONT}; }}
  #wrap {{ display:flex; flex-direction:column; height:100vh; }}
  header {{ background:{PANEL_BG}; border-bottom:1px solid {BORDER};
    padding:10px 16px; display:flex; align-items:center; gap:18px; flex-wrap:wrap; }}
  header h1 {{ font-size:15px; font-weight:600; letter-spacing:.02em; }}
  header h1 .run {{ color:{MUTED}; font-weight:400; margin-left:6px; font-size:13px; }}
  .stats {{ display:flex; gap:16px; flex-wrap:wrap; margin-left:auto; }}
  .stat {{ text-align:right; }}
  .stat .v {{ font-size:15px; font-weight:600; font-variant-numeric:tabular-nums;
    line-height:1.1; }}
  .stat .k {{ font-size:10px; color:{MUTED}; letter-spacing:.06em; text-transform:uppercase; }}
  nav {{ background:{PANEL_BG}; border-bottom:1px solid {BORDER}; padding:0 12px;
    display:flex; gap:4px; flex-wrap:wrap; }}
  .tab {{ background:transparent; color:{MUTED}; border:none; border-bottom:2px solid transparent;
    padding:10px 16px; cursor:pointer; font-size:13px; font-family:inherit; }}
  .tab:hover {{ color:{FG}; }}
  .tab.active {{ color:{FG}; border-bottom-color:{ACCENT}; }}
  main {{ flex:1; position:relative; min-height:0; background:{BG}; }}
  .panel {{ position:absolute; inset:0; display:none; }}
  .panel.active {{ display:block; }}
  iframe {{ width:100%; height:100%; border:none; background:#fff; }}
  .md {{ height:100%; overflow-y:auto; padding:28px 36px; max-width:900px; margin:0 auto;
    line-height:1.7; font-size:14px; }}
  .md h1 {{ font-size:20px; margin:6px 0 14px; }}
  .md h2 {{ font-size:16px; margin:20px 0 8px; color:{ACCENT}; }}
  .md h3 {{ font-size:14px; margin:16px 0 6px; }}
  .md ul {{ margin:8px 0 8px 22px; }}
  .md li {{ margin:3px 0; }}
  .md p {{ margin:8px 0; color:#cfd5de; }}
  .md hr {{ border:none; border-top:1px solid {BORDER}; margin:18px 0; }}
  .md code {{ background:rgba(255,255,255,.08); padding:1px 5px; border-radius:4px;
    font-size:12.5px; }}
  .md strong {{ color:{FG}; }}
  .muted {{ color:{MUTED}; padding:10px 8px; font-size:13px; }}
</style>
</head>
<body>
<div id="wrap">
  <header>
    <h1>可視化ハブ<span class="run">{html.escape(run_name)}</span></h1>
    <div class="stats">{stat_html}</div>
  </header>
  <nav>{tabs_html}</nav>
  <main>
{panels_html}
  </main>
</div>
<script>
  function showPanel(i) {{
    var tabs = document.querySelectorAll('.tab');
    var panels = document.querySelectorAll('.panel');
    for (var t = 0; t < tabs.length; t++) tabs[t].classList.remove('active');
    for (var p = 0; p < panels.length; p++) panels[p].classList.remove('active');
    if (tabs[i]) tabs[i].classList.add('active');
    var panel = document.getElementById('panel-' + i);
    if (panel) {{
      panel.classList.add('active');
      // 遅延ロード: 初回表示時に data-src を src へ移す(巨大 iframe 対策)
      var frame = panel.querySelector('iframe[data-src]');
      if (frame) {{ frame.src = frame.getAttribute('data-src'); frame.removeAttribute('data-src'); }}
    }}
  }}
</script>
</body>
</html>
"""
    return doc, artifacts


def make_hub(run_dir: Path, out: Path | None = None) -> tuple[Path, list[dict]]:
    """run_dir の成果物から hub.html を生成して書き出す。"""
    run_dir = run_dir if run_dir.is_absolute() else (_ROOT / run_dir)
    if not run_dir.is_dir():
        raise NotADirectoryError(f"ラン dir が見つからない: {run_dir}")
    run_name = run_dir.name
    doc, artifacts = build_hub_html(run_dir, run_name)
    out_path = out if out is not None else (run_dir / "hub.html")
    out_path.write_text(doc, encoding="utf-8")
    return out_path, artifacts


# ── CLI ───────────────────────────────────────────────────────────────────
def main(argv: list[str]) -> int:
    # Windows コンソール(cp932)対策: 非 cp932 文字の print で死なない。
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description="ラン成果物を 1 枚のタブ型 HTML に統合する")
    ap.add_argument("run_dir", type=str, help="runs/<name>")
    ap.add_argument("--out", type=str, default=None, help="出力先(既定 <run_dir>/hub.html)")
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    out = Path(args.out) if args.out else None
    out_path, artifacts = make_hub(run_dir, out)

    # 検出結果を stdout へ
    print(f"[hub] {run_dir}")
    detected_labels = {a["label"] for a in artifacts}
    for label, rels, _kind in CANDIDATES:
        hit = next((a for a in artifacts if a["label"] == label), None)
        if hit:
            print(f"  [検出] {label:<10} -> {hit['rel']}")
        else:
            print(f"  [なし] {label:<10}    ({' / '.join(rels)})")
    print(f"[hub] タブ {len(artifacts)} 個 -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
