#!/usr/bin/env python
"""⑧ LLM 自然言語要約(分析スイート W4)。確定 KPI 表を制約付き LLM で日本語要約する。

    python scripts/summarize_run.py runs/<name>                         # 既定 = mock(決定論)
    python scripts/summarize_run.py runs/<name> --backend ollama --model qwen3:8b
    python scripts/summarize_run.py runs/<name> --backend anthropic --model claude-haiku-4-5

docs/plans/analytics-suite.md(W4)・docs/research/analytics-methods.md(⑧)の方針に沿う
**L1/派生成果物の読み出し専用**スクリプト。判定は 2 段構成:

  第1段(決定論・LLM 不使用): 確定 KPI 表の組み立て。
    - summary.json(体数・日数・イベント種別数・llm_calls 等)と panel/ 配下の parquet
      (存在するものだけ)から**決定論的に選べる代表値のみ**を収集する。
    - → runs/<name>/panel/kpi_tables.json に書き出す(= 要約の唯一の情報源 = 単一真実)。
    - 存在しない表はスキップし「データ不足」として記録する(値は一切捏造しない)。
    - md はパースしない(parquet/json を一次とする)。

  第2段: 制約プロンプトによる LLM 要約 + 数値照合ガード。
    - プロンプト: 「以下の表の数値**のみ**を使い、表に無い数値・比較・因果を作らない。日本語で簡潔に」
      + kpi_tables.json の内容。
    - **数値照合ガード**: 生成文から数値を正規表現で抽出 → kpi の数値集合と正規化後 exact-match。
      不一致数値が 1 つでもあれば破棄 → 再生成(--retries 既定 2)。数値を 1 つも引用しない要約も
      KPI 表の要約として無効なので破棄する(空要約 → フォールバック。実測でも mock はここに落ちる)。
      全滅なら**決定論フォールバック**(表の数値をそのまま並べたテンプレ要約)に切替え、その旨を明記。
    - 忠実性スコア = 一致数値数 / 抽出数値数 を summary_ja.md 末尾と kpi_tables.json に記録。
    - 出力: runs/<name>/panel/summary_ja.md。既存の決定論テンプレ .md 群(heatmap_report.md 等)は
      一次真実として**置換しない**(横に置くだけ)。

====================================================================
R4 遵守(measurement circularity への防壁)— judge.py と同じ憲法
--------------------------------------------------------------------
* 本要約は**読み手向けの projection** であり、`judge.py` と同じく**シミュ本体へ一切逆流しない**。
  本ファイルは scripts/ 配下・L1/派生成果物の**読み取り専用下流**で、出力は run dir の
  **panel/ のみ**(summary_ja.md と kpi_tables.json)。シミュ入力・conf・src へは何も書かない。
  よって要約の内容がエージェント状態・k・行動へ影響することは構造的に不可能。
* LLM は**言語化のみ**を担い、数値は Python が確定させた KPI 表からしか使わせない(制約生成)。
  生成文の数値は原表と exact-match で後検証する(hallucination = 数値矛盾を機械的に弾く)。
* API キーは**環境変数名のみ**を受け取り、値をコード・ログ・エラー文字列に一切出さない
  (llm/* バックエンドと同じ規律)。モデル名・温度はハードコードせず引数で受ける。
====================================================================
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))

# Windows コンソール(cp932)で日本語や記号を print しても落ちないように(ファイル出力は常に UTF-8)。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

STEPS_PER_DAY = 144                    # 本シミュ固定の 1 日あたりステップ数(analyze_flows_grid と共通)

# panel/ に「あることが期待される」派生表(無ければデータ不足として記録)。
_EXPECTED_PANEL = ["panel.parquet", "heatmap_grid.parquet",
                   "od_matrix.parquet", "network_ts.parquet"]


# --------------------------------------------------------------------------- #
# 数値の正規化(ガード用)— 抽出トークンと KPI 値を同じ正準形に落として exact-match する。
# --------------------------------------------------------------------------- #
# 「1,234」「45.6」「0.72」「45.6%」等に対応(整数・小数・カンマ区切り・末尾%)。
# 負号は KPI 側に持たない(代表値は非負のカウント/率)ので扱わない=範囲表記の "-" 誤検出を避ける。
_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?%?")


def canon_num(token: str) -> str:
    """数値トークンを正準文字列へ。カンマ除去・末尾%除去・小数末尾0除去・整数先頭0除去。

    例: "1,234"→"1234" / "45.60%"→"45.6" / "0.720"→"0.72" / "007"→"7" / "0"→"0"。
    """
    s = str(token).strip().rstrip("%").replace(",", "")
    if s.startswith("+"):
        s = s[1:]
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    neg = s.startswith("-")
    core = s[1:] if neg else s
    if core.isdigit():
        core = core.lstrip("0") or "0"
        s = ("-" + core) if neg else core
    return s


def render_num(v) -> str:
    """KPI 数値を表示文字列へ(整数はカンマ区切り・小数は末尾0を落とす)。

    プロンプト・フォールバック・許可集合の三者で**同じ関数**を使うことで、
    「表に出した数値」と「ガードが許可する数値」を一致させる(canon_num で正規化して比較)。
    """
    if isinstance(v, bool):                       # bool は int 扱いにしない(想定外を弾く)
        return str(v)
    if isinstance(v, float):
        r = round(v, 6)
        if r == int(r):
            v = int(r)
        else:
            return f"{r:.6f}".rstrip("0").rstrip(".")
    return f"{int(v):,}"


def extract_numbers(text: str) -> list[str]:
    """生成文から数値トークンを抽出し正準化して返す(空トークンは除外)。"""
    out = [canon_num(t) for t in _NUM_RE.findall(text or "")]
    return [x for x in out if x != ""]


def allowed_number_set(kpi: dict) -> set[str]:
    """KPI 表に載っている全数値(rows[].num)の正準形集合 = ガードが許可する数値。"""
    allowed: set[str] = set()
    for t in kpi.get("tables", []):
        for r in t.get("rows", []):
            allowed.add(canon_num(render_num(r["num"])))
    return allowed


def check_faithfulness(text: str, allowed: set[str]) -> dict:
    """生成文の数値忠実性を検査する。

    返り値: ok(合格か)・score(一致数値数/抽出数値数)・n_extracted・n_matched・
            unmatched(許可集合に無い数値の例)。
    合格条件: 抽出数値が 1 つ以上あり、その**全て**が KPI 表の数値に一致すること
             (= 表に無い数値が 0 個 かつ 空要約でない)。
    """
    extracted = extract_numbers(text)
    matched = [e for e in extracted if e in allowed]
    unmatched = [e for e in extracted if e not in allowed]
    n_ex = len(extracted)
    n_ma = len(matched)
    score = (n_ma / n_ex) if n_ex else 0.0
    ok = (n_ex >= 1) and (len(unmatched) == 0)
    return {"ok": ok, "score": round(score, 4), "n_extracted": n_ex,
            "n_matched": n_ma, "unmatched": unmatched[:8]}


# --------------------------------------------------------------------------- #
# 第1段: 確定 KPI 表の組み立て(決定論・LLM 不使用)
# --------------------------------------------------------------------------- #
def _to_num(v):
    """parquet/JSON から来た値を int/float へ(bool や None・非数は None)。"""
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, int):
        return int(v)
    if isinstance(v, float):
        return float(v)
    return None


def _overview_rows(summary: dict) -> list[dict]:
    """summary.json → ラン概要の代表値行(存在するキーだけ・決定論)。"""
    rows: list[dict] = []
    n_steps = summary.get("n_steps")

    def _add(label, key, unit=None, value=None):
        v = value if value is not None else summary.get(key)
        n = _to_num(v)
        if n is not None:
            rows.append({"label": label, "num": n, "unit": unit})

    _add("エージェント数", "n_agents", "体")
    if isinstance(n_steps, int):                  # 日数は n_steps // STEPS_PER_DAY の決定論的派生。
        # ラベルに数字を入れない(フォールバック本文へ表外数値が混入しガードを汚すのを避ける)。
        rows.append({"label": "シミュレーション日数(ステップ数から換算)",
                     "num": n_steps // STEPS_PER_DAY, "unit": "日"})
    _add("総ステップ数", "n_steps")
    _add("総イベント数", "n_events", "件")
    ek = summary.get("event_kinds")
    if isinstance(ek, dict):
        rows.append({"label": "イベント種別数", "num": len(ek), "unit": "種"})
    _add("LLM 呼び出し回数", "llm_calls", "回")
    _add("LLM キャッシュヒット", "llm_cache_hits", "回")
    _add("造語アイテム数", "n_items", "個")
    _add("伝播回数", "n_transmissions", "回")
    _add("総採用数", "total_adoptions", "回")
    return rows


def _top_event_rows(summary: dict, top: int = 5) -> list[dict]:
    """イベント種別を件数降順(タイは名前昇順)で上位 top 件 = 決定論的な代表値。"""
    ek = summary.get("event_kinds")
    if not isinstance(ek, dict) or not ek:
        return []
    ordered = sorted(ek.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
    return [{"label": str(k), "num": int(v), "unit": "件"} for k, v in ordered[:top]]


def _read_parquet(path: str):
    """parquet を列 dict + schema で返す(pyarrow・純 Python。pandas 非依存)。"""
    tab = pq.read_table(path)
    return tab.to_pydict(), tab.schema


def _numeric_columns(schema) -> list[str]:
    return [f.name for f in schema
            if pa.types.is_integer(f.type) or pa.types.is_floating(f.type)]


def _extract_heatmap_grid(cols: dict, schema) -> list[dict]:
    """heatmap_grid.parquet(cell_x, cell_y, hour_bin, pass_count, present_count,
    unique_agents)の curated 代表値。座標は数値としてガードに載せない(範囲/符号の誤検出回避)。"""
    n = len(cols.get("cell_x", []))
    pass_tot = sum(int(v) for v in cols.get("pass_count", []) if v is not None)
    pres_tot = sum(int(v) for v in cols.get("present_count", []) if v is not None)
    uniq_max = max((int(v) for v in cols.get("unique_agents", []) if v is not None),
                   default=0)
    present_by_cell: dict = defaultdict(int)
    for i in range(n):
        c = (cols["cell_x"][i], cols["cell_y"][i])
        present_by_cell[c] += int(cols["present_count"][i] or 0)
    top_present = max((v for v in present_by_cell.values()), default=0)
    return [
        {"label": "集計セル×時間帯レコード数", "num": n, "unit": "行"},
        {"label": "通過(pass_count)総数", "num": pass_tot, "unit": "件"},
        {"label": "在圏観測(present_count)総数", "num": pres_tot, "unit": "件"},
        {"label": "最大同時 distinct 体数", "num": uniq_max, "unit": "体"},
        {"label": "最混雑セルの在圏観測合計", "num": top_present, "unit": "件"},
    ]


def _extract_generic(cols: dict, schema) -> list[dict]:
    """未知スキーマの parquet の代表値: レコード数 + 数値列(先頭6列)の先頭/末尾値。
    trips/present_count/pass_count/trips 等の量的列があれば合計・最大も足す(いずれも決定論)。"""
    ncols = _numeric_columns(schema)
    n = len(cols[schema.names[0]]) if schema.names else 0
    rows: list[dict] = [{"label": "レコード数", "num": n, "unit": "行"}]
    _MAG = {"trips", "present_count", "pass_count", "n_transmissions",
            "n_edges", "n_nodes"}
    for c in ncols[:6]:
        vals = [v for v in cols[c] if v is not None]
        if not vals:
            continue
        first = _to_num(cols[c][0])
        last = _to_num(cols[c][-1])
        if first is not None:
            rows.append({"label": f"{c} 先頭値", "num": first, "unit": None})
        if last is not None and last != first:
            rows.append({"label": f"{c} 末尾値", "num": last, "unit": None})
        if c in _MAG:
            nums = [_to_num(v) for v in vals if _to_num(v) is not None]
            if nums:
                rows.append({"label": f"{c} 合計", "num": sum(nums), "unit": None})
                rows.append({"label": f"{c} 最大", "num": max(nums), "unit": None})
    return rows


# ファイル名 → 専用抽出器(未登録は generic)
_PANEL_EXTRACTORS = {"heatmap_grid.parquet": _extract_heatmap_grid}

_PANEL_TITLES = {
    "panel.parquet": "エージェント日次パネル",
    "heatmap_grid.parquet": "人流ヒートマップ格子",
    "od_matrix.parquet": "OD 行列",
    "network_ts.parquet": "社会ネットワーク時系列",
}


def _extract_panel_table(path: str) -> list[dict]:
    cols, schema = _read_parquet(path)
    fname = os.path.basename(path)
    extractor = _PANEL_EXTRACTORS.get(fname, _extract_generic)
    return extractor(cols, schema)


def collect_kpi(run_dir: str, write: bool = True) -> dict:
    """第1段: run dir から確定 KPI 表を集め kpi_tables.json を書く。返り値 = kpi dict。

    誠実性: 存在しない表は「データ不足」として missing に記録し、値は一切捏造しない。
    決定論: ソート順・キー選択は固定。壁時計時刻は出力に含めない(同入力→同出力)。
    """
    run_dir = os.path.abspath(run_dir)
    run_name = os.path.basename(os.path.normpath(run_dir))
    tables: list[dict] = []
    missing: list[dict] = []

    # --- summary.json(体数・日数・イベント種別数・llm_calls 等) ---
    spath = os.path.join(run_dir, "summary.json")
    if os.path.exists(spath):
        try:
            summary = json.loads(Path(spath).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            summary = {}
        ov = _overview_rows(summary)
        if ov:
            tables.append({"id": "overview", "title": "ラン概要",
                           "source": "summary.json", "rows": ov})
        tev = _top_event_rows(summary)
        if tev:
            tables.append({"id": "top_event_kinds",
                           "title": "主なイベント種別(件数の多い順)",
                           "source": "summary.json", "rows": tev})
    else:
        missing.append({"source": "summary.json", "note": "データ不足(未生成)"})

    # --- panel/ 配下の parquet(存在するものだけ) ---
    panel_dir = os.path.join(run_dir, "panel")
    present_extra: list[str] = []
    if os.path.isdir(panel_dir):
        present_extra = sorted(
            os.path.basename(p) for p in Path(panel_dir).glob("*.parquet"))
    # 期待表: あれば抽出、無ければデータ不足
    for fname in _EXPECTED_PANEL:
        ppath = os.path.join(panel_dir, fname)
        src = f"panel/{fname}"
        if os.path.exists(ppath):
            try:
                rows = _extract_panel_table(ppath)
            except Exception as exc:              # 破損等は捏造せずデータ不足扱い
                missing.append({"source": src,
                                "note": f"データ不足(読み取り失敗: {type(exc).__name__})"})
                continue
            tables.append({"id": fname.replace(".parquet", ""),
                           "title": _PANEL_TITLES.get(fname, fname),
                           "source": src, "rows": rows})
        else:
            missing.append({"source": src, "note": "データ不足(未生成)"})
    # 期待外だが存在する parquet(例: worldview.parquet)も generic で取り込む
    for fname in present_extra:
        if fname in _EXPECTED_PANEL:
            continue
        ppath = os.path.join(panel_dir, fname)
        try:
            cols, schema = _read_parquet(ppath)
            rows = _extract_generic(cols, schema)
        except Exception:
            continue
        tables.append({"id": fname.replace(".parquet", ""),
                       "title": _PANEL_TITLES.get(fname, fname),
                       "source": f"panel/{fname}", "rows": rows})

    kpi = {
        "run": run_name,
        "run_dir": run_dir,
        "generator": "scripts/summarize_run.py",
        "tables": tables,
        "missing": missing,
        "n_tables": len(tables),
    }
    kpi["allowed_numbers"] = sorted(allowed_number_set(kpi))
    kpi["n_numbers"] = len(kpi["allowed_numbers"])

    if write:
        os.makedirs(panel_dir, exist_ok=True)
        with open(os.path.join(panel_dir, "kpi_tables.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(kpi, fh, ensure_ascii=False, indent=1)
    return kpi


# --------------------------------------------------------------------------- #
# 第2段: 制約プロンプト + 数値照合ガード + 決定論フォールバック
# --------------------------------------------------------------------------- #
_PROMPT_HEADER = (
    "あなたはシミュレーション結果の要約を書くアシスタントです。\n"
    "以下の KPI 表に**実際に載っている数値だけ**を使い、日本語で簡潔に(3〜6文で)要約してください。\n"
    "厳守事項:\n"
    "・表に無い数値を書かない。表の数値を足し算・割り算・概算・丸めをしない(表記のまま書き写す)。\n"
    "・表に無い比較(増えた/減った/最多 等)・因果・推測・評価を書かない。\n"
    "・数値は表の値をそのまま使う(単位は付けてよい)。事実の列挙に留める。\n\n"
    "=== KPI表 ===\n"
)
_PROMPT_FOOTER = "\n=== ここまで ===\n\n要約:"


def _render_tables(kpi: dict) -> str:
    L: list[str] = []
    for t in kpi.get("tables", []):
        L.append(f"# {t['title']}(出典: {t['source']})")
        for r in t["rows"]:
            unit = r.get("unit") or ""
            L.append(f"- {r['label']}: {render_num(r['num'])}{unit}")
        L.append("")
    if kpi.get("missing"):
        L.append("# データ不足(未生成・要約対象外の表)")
        for mm in kpi["missing"]:
            L.append(f"- {mm['source']}: {mm['note']}")
    return "\n".join(L)


def build_prompt(kpi: dict) -> str:
    return _PROMPT_HEADER + _render_tables(kpi) + _PROMPT_FOOTER


def build_fallback_summary(kpi: dict) -> str:
    """決定論フォールバック: 表の数値をそのまま並べたテンプレ要約(= 数値忠実性 1.0)。

    固定文には数字を含めない(タイトル・ラベルとも数字なし)ので、抽出される数値は
    KPI 表の値だけ = ガードを必ず通る(全滅時の安全網)。
    """
    L = ["(自動生成・決定論フォールバック: LLM 生成が数値照合ガードを通過しなかったため、"
         "KPI 表の数値をそのまま提示します。)", ""]
    for t in kpi.get("tables", []):
        parts = [f"{r['label']} {render_num(r['num'])}{r.get('unit') or ''}"
                 for r in t["rows"]]
        if parts:
            L.append(f"【{t['title']}】" + "、".join(parts) + "。")
    if kpi.get("missing"):
        L.append("【データ不足】"
                 + "、".join(m["source"] for m in kpi["missing"])
                 + " は未生成のため要約対象外。")
    return "\n".join(L)


def _is_backend_error(text: str) -> bool:
    """llm/* バックエンドの失敗文字列(__ollama_error__ 等)を検出する。"""
    t = (text or "")[:40]
    return t.startswith("__") and "_error__" in t


def summarize_run(run_dir: str, backend, *, backend_name: str = "mock",
                  model: str | None = None, temperature: float = 0.2,
                  max_tokens: int = 512, retries: int = 2) -> dict:
    """第1段 + 第2段を実行し runs/<name>/panel/summary_ja.md を書く。返り値 = 実行情報 dict。

    backend: generate(prompt, *, rng_key, temperature, max_tokens, think) -> str を持つ
             オブジェクト(society.llm.* を軽量に直接インスタンス化したもの or テスト注入の偽物)。
    """
    kpi = collect_kpi(run_dir, write=True)
    allowed = allowed_number_set(kpi)
    prompt = build_prompt(kpi)
    run_name = kpi["run"]

    attempts: list[dict] = []
    chosen = None
    body = None
    n_try = max(0, int(retries)) + 1              # 初回 + リトライ回数
    for attempt in range(n_try):
        # rng_key に attempt を混ぜる: mock は attempt ごとに決定論、実 LLM は温度で変動。
        rng_key = f"summarize/{run_name}/{attempt}"
        text = backend.generate(prompt, rng_key=rng_key, temperature=temperature,
                                max_tokens=max_tokens, think=False)
        if _is_backend_error(text):
            attempts.append({"attempt": attempt, "ok": False, "error": True,
                             "score": 0.0, "n_extracted": 0, "n_matched": 0})
            continue
        chk = check_faithfulness(text, allowed)
        attempts.append({"attempt": attempt, "ok": chk["ok"], "error": False,
                         "score": chk["score"], "n_extracted": chk["n_extracted"],
                         "n_matched": chk["n_matched"], "unmatched": chk["unmatched"]})
        if chk["ok"]:
            body, chosen = text, chk
            fallback = False
            used_attempt = attempt
            break

    if chosen is None:                            # 全滅 → 決定論フォールバック
        body = build_fallback_summary(kpi)
        chosen = check_faithfulness(body, allowed)
        fallback = True
        used_attempt = None

    # summary_ja.md(本文 + メタ脚注)。脚注の数字は本文の忠実性検査には含めない。
    md = _render_summary_md(kpi, body, chosen, fallback, used_attempt,
                            backend_name, model)
    panel_dir = os.path.join(kpi["run_dir"], "panel")
    os.makedirs(panel_dir, exist_ok=True)
    with open(os.path.join(panel_dir, "summary_ja.md"), "w",
              encoding="utf-8") as fh:
        fh.write(md)

    # kpi_tables.json に忠実性を追記(単一真実に生成結果の来歴を残す)
    kpi["faithfulness"] = {
        "backend": backend_name, "model": (model if backend_name != "mock" else "mock"),
        "fallback": fallback, "used_attempt": used_attempt,
        "score": chosen["score"], "n_extracted": chosen["n_extracted"],
        "n_matched": chosen["n_matched"], "retries": int(retries),
        "attempts": attempts,
    }
    with open(os.path.join(panel_dir, "kpi_tables.json"), "w",
              encoding="utf-8") as fh:
        json.dump(kpi, fh, ensure_ascii=False, indent=1)

    return {
        "run": run_name, "run_dir": kpi["run_dir"],
        "backend": backend_name, "model": model,
        "n_tables": kpi["n_tables"], "n_numbers": kpi["n_numbers"],
        "missing": [m["source"] for m in kpi["missing"]],
        "fallback": fallback, "used_attempt": used_attempt,
        "faithfulness": chosen["score"], "n_extracted": chosen["n_extracted"],
        "n_matched": chosen["n_matched"],
        "kpi_tables": os.path.join(panel_dir, "kpi_tables.json"),
        "summary_md": os.path.join(panel_dir, "summary_ja.md"),
        "attempts": attempts,
    }


def _render_summary_md(kpi, body, chk, fallback, used_attempt,
                       backend_name, model) -> str:
    L: list[str] = [f"# ラン要約(LLM 自然言語要約 ⑧)— {kpi['run']}\n"]
    L.append(body.rstrip())
    L.append("")
    L.append("---")
    L.append(f"- 生成バックエンド: `{backend_name}` / モデル: "
             f"`{model if (model and backend_name != 'mock') else 'mock'}`")
    if fallback:
        L.append("- 経路: **決定論フォールバック**(LLM 生成が数値照合ガードを通過しなかったため、"
                 "KPI 表の数値をそのまま並べた)")
    else:
        L.append(f"- 経路: LLM 生成(第 {used_attempt + 1} 回で合格)")
    L.append(f"- 忠実性スコア(数値一致率 = 一致数値数 / 抽出数値数): "
             f"{chk['score']:.3f}({chk['n_matched']}/{chk['n_extracted']})")
    L.append("- 数値照合ガード: 生成文の数値を正規表現抽出し、kpi_tables.json の数値集合と"
             "正規化後 exact-match。表に無い数値が 1 つでもあれば破棄・再生成、全滅で決定論フォールバック。")
    L.append("- 一次真実: 本要約は読み手向けの projection。既存の決定論テンプレ .md "
             "(heatmap_report.md 等)を置換しない。KPI の単一真実は panel/kpi_tables.json。")
    L.append("- R4 防壁: 出力は run dir の panel/ のみ。シミュ本体・conf・src へは書き込まない"
             "(読み取り専用下流)。")
    L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# バックエンド構築(society.llm.* を軽量に直接インスタンス化・Simulation は import しない)
# --------------------------------------------------------------------------- #
def make_backend(name: str, model: str | None = None, *, base_url: str | None = None,
                 api_key_env: str | None = None, host: str | None = None,
                 timeout_s: float = 120.0, master_seed: int = 0):
    """--backend 名から生成メソッド互換のバックエンドを作る。モデル名・キーはハードコードしない。"""
    if name == "mock":
        from society.llm.mock import MockBackend
        from society.rng import RngHub
        return MockBackend(RngHub(master_seed))
    if name == "ollama":
        if not model:
            raise SystemExit("--backend ollama には --model が必要です")
        from society.llm.ollama import OllamaBackend
        return OllamaBackend(model, host=host or "http://localhost:11434",
                             timeout_s=timeout_s)
    if name == "openai_compat":
        if not model:
            raise SystemExit("--backend openai_compat には --model が必要です")
        from society.llm.openai_compat import OpenAICompatBackend
        return OpenAICompatBackend(model, base_url=base_url or "https://api.openai.com/v1",
                                   api_key_env=api_key_env or "OPENAI_API_KEY",
                                   timeout_s=timeout_s)
    if name == "anthropic":
        if not model:
            raise SystemExit("--backend anthropic には --model が必要です")
        from society.llm.anthropic import AnthropicBackend
        return AnthropicBackend(model, api_key_env=api_key_env or "ANTHROPIC_API_KEY",
                                base_url=base_url or "https://api.anthropic.com",
                                timeout_s=timeout_s)
    raise SystemExit(f"未対応 backend: {name}(mock | ollama | openai_compat | anthropic)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="⑧ LLM 自然言語要約(確定 KPI 表を制約生成 + 数値照合ガード)")
    ap.add_argument("run_dir", help="例: runs/wv_llm_7d")
    ap.add_argument("--backend", default="mock",
                    choices=["mock", "ollama", "openai_compat", "anthropic"])
    ap.add_argument("--model", default=None, help="モデル名(mock 以外は必須)")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--retries", type=int, default=2, help="ガード不通過時の再生成回数(既定2)")
    ap.add_argument("--base-url", default=None,
                    help="openai_compat/anthropic のベース URL(既定は各本家)")
    ap.add_argument("--api-key-env", default=None,
                    help="API キーを入れる環境変数名(値は渡さない)")
    ap.add_argument("--host", default=None, help="ollama のホスト(既定 http://localhost:11434)")
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()
    if not os.path.isdir(args.run_dir):
        raise SystemExit(f"not a directory: {args.run_dir}")

    backend = make_backend(args.backend, args.model, base_url=args.base_url,
                           api_key_env=args.api_key_env, host=args.host,
                           timeout_s=args.timeout)
    info = summarize_run(args.run_dir, backend, backend_name=args.backend,
                         model=args.model, temperature=args.temperature,
                         max_tokens=args.max_tokens, retries=args.retries)
    path = "決定論フォールバック" if info["fallback"] else f"LLM第{info['used_attempt']+1}回"
    print(f"[summarize_run] {info['run']}  backend={info['backend']}  "
          f"表={info['n_tables']}  数値={info['n_numbers']}  経路={path}")
    print(f"  忠実性={info['faithfulness']:.3f} "
          f"({info['n_matched']}/{info['n_extracted']})  "
          f"データ不足={info['missing']}")
    print(f"  -> {info['kpi_tables']}\n  -> {info['summary_md']}")


if __name__ == "__main__":
    main()
