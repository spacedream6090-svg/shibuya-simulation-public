"""scripts/bench.py の LOD 計測・reflect 出力上限分析の純関数テスト(第24バッチ A1/A5)。

実 LLM や実行環境に依存しない部分だけを検証する(ollama 実走・parquet 突合は別途手動)。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_bench():
    """scripts/bench.py をパッケージ外モジュールとして読み込む。"""
    spec = importlib.util.spec_from_file_location(
        "bench_mod", REPO_ROOT / "scripts" / "bench.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bench = _load_bench()


# ---- 純関数: percentile / summarize ----
def test_percentile_nearest_rank():
    xs = [10, 20, 30, 40, 50]
    assert bench._percentile(xs, 50) == 30
    assert bench._percentile(xs, 0) == 10
    assert bench._percentile(xs, 100) == 50
    assert bench._percentile([], 50) is None


def test_summarize_shape_and_empty():
    s = bench._summarize([100, 200, 300, 400])
    assert s["n"] == 4
    assert s["max"] == 400
    assert s["p50"] in (200, 300)      # 最近傍順位法
    assert s["mean"] == 250.0
    empty = bench._summarize([])
    assert empty["n"] == 0 and empty["max"] is None


# ---- 純関数: token 概算 / 推奨上限 ----
def test_estimate_tokens_ratio():
    # 180 文字・1.8 字/tok = 100 tok
    assert abs(bench._estimate_tokens("あ" * 180, 1.8) - 100.0) < 1e-6


def test_recommend_max_tokens_headroom_and_rounding():
    # max=459 文字, 1.8 字/tok → 255tok, ×1.5=382.5 → 64刻み切上げ=384
    rec = bench._recommend_max_tokens(459, 1.8, safety=1.5, floor=256)
    assert rec % 64 == 0
    assert rec >= 382
    # floor が効く(極端に短い観測でも下限を割らない)
    assert bench._recommend_max_tokens(10, 1.8, floor=256) == 256


# ---- fallback 判定 ----
def test_is_fallback():
    assert bench._is_fallback("") is True
    assert bench._is_fallback("   ") is True
    assert bench._is_fallback("__ollama_error__: boom") is True
    assert bench._is_fallback("__vllm_error__: HTTP 500") is True
    assert bench._is_fallback('{"action":"speak","text":"やあ"}') is False


# ---- LOD 集計: 真の eval_count 経路 ----
def test_aggregate_lod_real_tokens():
    recs = [
        # 2呼: それぞれ 100tok を 0.5s(=200 tok/s)。本文 180 字 → 実測 1.8 字/tok
        {"model": "qwen3:4b", "purpose": "speak", "text": "あ" * 180,
         "chars": 180, "fallback": False, "eval_count": 100,
         "eval_duration_ns": 500_000_000, "prompt_eval_count": 300, "wall_s": 0.6},
        {"model": "qwen3:4b", "purpose": "speak", "text": "い" * 180,
         "chars": 180, "fallback": False, "eval_count": 100,
         "eval_duration_ns": 500_000_000, "prompt_eval_count": 300, "wall_s": 0.6},
    ]
    out = bench._aggregate_lod(recs, chars_per_token=2.0)
    assert len(out) == 1
    row = out[0]
    assert row["calls"] == 2 and row["fallback"] == 0
    assert row["fallback_rate"] == 0.0
    assert row["decode_tok_s"] == 200.0        # 200 tok / 1.0 s
    assert row["mean_gen_tok"] == 100.0
    assert row["chars_per_tok_meas"] == 1.8    # 360 字 / 200 tok
    assert row["est_only"] is False
    assert row["mean_prompt_tok"] == 300.0


# ---- LOD 集計: fallback をレートに含め本文集計から除く ----
def test_aggregate_lod_fallback_excluded():
    recs = [
        {"model": "m", "purpose": "reflect", "text": "ok" * 50, "chars": 100,
         "fallback": False, "eval_count": 50, "eval_duration_ns": 250_000_000,
         "prompt_eval_count": 10, "wall_s": 0.3},
        {"model": "m", "purpose": "reflect", "text": "__ollama_error__: x",
         "chars": 19, "fallback": True, "eval_count": None,
         "eval_duration_ns": None, "prompt_eval_count": None, "wall_s": None},
    ]
    out = bench._aggregate_lod(recs, chars_per_token=1.8)
    row = out[0]
    assert row["calls"] == 2 and row["fallback"] == 1
    assert row["fallback_rate"] == 0.5
    assert row["mean_resp_chars"] == 100.0     # エラー呼は本文集計から除外
    assert row["max_resp_chars"] == 100


# ---- LOD 集計: eval_count が無い backend は概算経路 ----
def test_aggregate_lod_estimate_path():
    recs = [
        {"model": "vllm/x", "purpose": "plan", "text": "z" * 360, "chars": 360,
         "fallback": False, "eval_count": None, "eval_duration_ns": None,
         "prompt_eval_count": None, "wall_s": 1.0},
    ]
    out = bench._aggregate_lod(recs, chars_per_token=1.8)
    row = out[0]
    assert row["est_only"] is True
    assert row["chars_per_tok_meas"] is None
    # 360 字 / 1.8 = 200 tok を 1.0 s → 200 tok/s(概算)
    assert row["decode_tok_s"] == 200.0


# ---- reflect 分析: 複数 run のマージ ----
def test_merge_run_stats():
    per_run = [
        {"raw_by_purpose": {"reflect": [200, 300]}, "belief": [30], "summary": [40],
         "speak_text": [90], "plan_n": [5], "n_salient": [1],
         "reflect_total": 2, "reflect_written": 2},
        {"raw_by_purpose": {"reflect": [250], "plan": [400]}, "belief": [50],
         "summary": [], "speak_text": [100, 110], "plan_n": [4], "n_salient": [2],
         "reflect_total": 1, "reflect_written": 1},
    ]
    m = bench._merge_run_stats(per_run)
    assert sorted(m["raw"]["reflect"]) == [200, 250, 300]
    assert m["raw"]["plan"] == [400]
    assert m["belief"] == [30, 50]
    assert m["reflect_total"] == 3 and m["reflect_written"] == 3
    assert m["speak"] == [90, 100, 110]


# ---- cache 索引: key[:16] → response(壊れ行はスキップ)----
def test_load_cache_index(tmp_path):
    run = tmp_path
    key = "a" * 64
    (run / "llm_cache.jsonl").write_text(
        '{"key": "%s", "response": "hello"}\n'
        'this is not json\n'
        '{"key": "", "response": "skip-empty-key"}\n' % key,
        encoding="utf-8")
    idx = bench._load_cache_index(run)
    assert idx["a" * 16] == "hello"
    assert len(idx) == 1              # 壊れ行と空キーは無視


# ---- markdown 表がキー欠損でも落ちない ----
def test_lod_markdown_renders():
    rows = [{"model": "m", "purpose": "speak", "calls": 3}]
    txt = bench._lod_markdown(rows)
    assert "model" in txt and "purpose" in txt and "tok/s" in txt


def test_scaling_markdown_backcompat():
    rows = [{"agents": 10, "steps": 60, "wall_s": 1.2}]
    txt = bench._markdown_table(rows)
    assert "agents" in txt and "ms/step" in txt
