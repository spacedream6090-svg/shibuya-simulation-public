"""scripts/summarize_run.py の最小テスト(合成 run dir・偽 LLM 注入・決定論)。

分析スイート W4(⑧ LLM 自然言語要約)の 2 段構成を検証する:
  * 第1段: 合成 summary.json + 小さな parquet から KPI を決定論的に収集する。
  * ガード: 数値抽出の正規表現 / 忠実な文→合格・忠実性1.0 /
    表に無い数値を含む文→破棄→リトライ→決定論フォールバック(偽 LLM 注入)。
  * mock バックエンド経路が決定論(同入力2回で同出力)。
実 LLM・ollama・API は一切呼ばない(mock/注入で完結)。
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

_REPO = Path(__file__).resolve().parents[1]

# scripts/summarize_run.py をモジュールとして読み込む(scripts はパッケージではない)
_spec = importlib.util.spec_from_file_location(
    "summarize_run", _REPO / "scripts" / "summarize_run.py")
sr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sr)


# --------------------------------------------------------------------------- #
# 合成 run dir(summary.json + 小さな heatmap_grid.parquet)
# --------------------------------------------------------------------------- #
def _write_run(dir_: Path) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    summary = {
        "n_agents": 20,
        "n_steps": 1008,                       # 1008 ÷ 144 = 7 日
        "n_events": 30007,
        "event_kinds": {"state_update": 3721, "affect_update": 2885,
                        "move_segment": 2888, "speak": 1328, "arrive": 1684,
                        "dm": 62},
        "llm_calls": 1734,
        "llm_cache_hits": 0,
        "n_items": 0,
        "n_transmissions": 0,
        "total_adoptions": 0,
    }
    (dir_ / "summary.json").write_text(json.dumps(summary, ensure_ascii=False),
                                       encoding="utf-8")
    panel = dir_ / "panel"
    panel.mkdir(parents=True, exist_ok=True)
    # heatmap_grid.parquet: 3 セル×時間帯レコード(analyze_flows_grid と同じ6列)
    tbl = pa.table({
        "cell_x": pa.array([1, 1, -10], pa.int64()),
        "cell_y": pa.array([-3, -3, -5], pa.int64()),
        "hour_bin": pa.array([18, 19, 9], pa.int64()),
        "pass_count": pa.array([10, 20, 5], pa.int64()),
        "present_count": pa.array([16, 20, 4], pa.int64()),
        "unique_agents": pa.array([5, 5, 3], pa.int64()),
    })
    pq.write_table(tbl, panel / "heatmap_grid.parquet")


# --------------------------------------------------------------------------- #
# 偽 LLM(注入用)。generate の系列を順に返す = リトライ/フォールバック経路の検証。
# --------------------------------------------------------------------------- #
class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        r = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return r


# --------------------------------------------------------------------------- #
# 第1段: KPI 収集
# --------------------------------------------------------------------------- #
def test_collect_kpi_from_synthetic_run(tmp_path):
    run = tmp_path / "run"
    _write_run(run)
    kpi = sr.collect_kpi(str(run), write=True)

    # kpi_tables.json が panel/ に書かれる(単一真実)
    kp = run / "panel" / "kpi_tables.json"
    assert kp.exists()
    saved = json.loads(kp.read_text(encoding="utf-8"))
    assert saved["run"] == "run"

    titles = {t["title"]: t for t in kpi["tables"]}
    assert "ラン概要" in titles
    assert "主なイベント種別(件数の多い順)" in titles
    assert "人流ヒートマップ格子" in titles

    # ラン概要の代表値(決定論)
    ov = {r["label"]: r["num"] for r in titles["ラン概要"]["rows"]}
    assert ov["エージェント数"] == 20
    assert ov["シミュレーション日数(ステップ数から換算)"] == 7
    assert ov["総イベント数"] == 30007
    assert ov["イベント種別数"] == 6
    assert ov["LLM 呼び出し回数"] == 1734

    # イベント種別は件数降順(タイは名前昇順): move_segment(2888) > affect_update(2885)
    ev = [r["label"] for r in titles["主なイベント種別(件数の多い順)"]["rows"]]
    assert ev[0] == "state_update"
    assert ev.index("move_segment") < ev.index("affect_update")

    # heatmap 代表値: present 総数=16+20+4=40 / pass 総数=35 / 最混雑セル(1,-3)=36
    hm = {r["label"]: r["num"] for r in titles["人流ヒートマップ格子"]["rows"]}
    assert hm["在圏観測(present_count)総数"] == 40
    assert hm["通過(pass_count)総数"] == 35
    assert hm["最混雑セルの在圏観測合計"] == 36
    assert hm["集計セル×時間帯レコード数"] == 3

    # 未生成の期待表は「データ不足」(捏造しない)
    missing = {m["source"] for m in kpi["missing"]}
    assert "panel/od_matrix.parquet" in missing
    assert "panel/network_ts.parquet" in missing
    assert all("データ不足" in m["note"] for m in kpi["missing"])


def test_collect_kpi_is_deterministic(tmp_path):
    run = tmp_path / "run"
    _write_run(run)
    a = sr.collect_kpi(str(run), write=False)
    b = sr.collect_kpi(str(run), write=False)
    assert json.dumps(a, ensure_ascii=False, sort_keys=True) == \
        json.dumps(b, ensure_ascii=False, sort_keys=True)


def test_missing_summary_is_data_shortage(tmp_path):
    run = tmp_path / "bare"
    run.mkdir()
    kpi = sr.collect_kpi(str(run), write=False)
    srcs = {m["source"] for m in kpi["missing"]}
    assert "summary.json" in srcs
    # 表が 1 つも無くても壊れない
    assert kpi["n_tables"] == 0


# --------------------------------------------------------------------------- #
# 数値抽出の正規表現 / 正規化
# --------------------------------------------------------------------------- #
def test_number_extraction_regex():
    text = "総イベント30,007件で、混雑率は45.6%、指標は0.72、呼び出し1734回。"
    got = set(sr.extract_numbers(text))
    assert {"30007", "45.6", "0.72", "1734"} <= got


def test_canon_num_normalization():
    assert sr.canon_num("1,234") == "1234"
    assert sr.canon_num("45.60%") == "45.6"
    assert sr.canon_num("0.720") == "0.72"
    assert sr.canon_num("007") == "7"
    assert sr.canon_num("0") == "0"


# --------------------------------------------------------------------------- #
# ガード: 忠実な文→合格 / 表に無い数値→破棄→リトライ→フォールバック
# --------------------------------------------------------------------------- #
def test_faithful_summary_passes_with_score_1(tmp_path):
    run = tmp_path / "run"
    _write_run(run)
    # KPI に載っている数値だけを使う忠実な要約(20体・7日・30,007件・1,734回)
    faithful = ("本ランはエージェント20体・7日間で、総イベント30,007件・"
                "LLM呼び出し1,734回。人流の在圏観測は40件。")
    fake = FakeLLM([faithful])
    info = sr.summarize_run(str(run), fake, backend_name="stub", model="stub-1")

    assert info["fallback"] is False
    assert info["used_attempt"] == 0
    assert info["faithfulness"] == 1.0
    assert fake.calls == 1                        # 初回で合格 = リトライ無し
    assert (run / "panel" / "summary_ja.md").exists()


def test_hallucinated_numbers_reject_then_fallback(tmp_path):
    run = tmp_path / "run"
    _write_run(run)
    # 表に無い数値(9999・12345)を含む捏造文を毎回返す → 全滅 → 決定論フォールバック
    fabricated = "このランでは9999体が12345回行動し、混雑は88.8%だった。"
    fake = FakeLLM([fabricated, fabricated, fabricated])
    info = sr.summarize_run(str(run), fake, backend_name="stub",
                            model="stub-1", retries=2)

    assert info["fallback"] is True
    # 初回 + リトライ2 = 3 回生成し、全て破棄
    assert fake.calls == 3
    assert all(a["ok"] is False for a in info["attempts"])
    # フォールバックは表の数値だけ = 忠実性 1.0
    assert info["faithfulness"] == 1.0
    md = (run / "panel" / "summary_ja.md").read_text(encoding="utf-8")
    assert "決定論フォールバック" in md
    # kpi_tables.json に忠実性の来歴が残る
    saved = json.loads((run / "panel" / "kpi_tables.json").read_text(encoding="utf-8"))
    assert saved["faithfulness"]["fallback"] is True
    assert len(saved["faithfulness"]["attempts"]) == 3


def test_empty_numberless_summary_is_rejected(tmp_path):
    run = tmp_path / "run"
    _write_run(run)
    # 数値を 1 つも引用しない要約は KPI 表の要約として無効 → フォールバック
    numberless = "このランはいろいろな出来事があって、とても賑やかな一日でした。"
    fake = FakeLLM([numberless, numberless, numberless])
    info = sr.summarize_run(str(run), fake, backend_name="stub", retries=2)
    assert info["fallback"] is True


def test_backend_error_string_falls_back(tmp_path):
    run = tmp_path / "run"
    _write_run(run)
    fake = FakeLLM(["__api_error__: boom", "__api_error__: boom",
                    "__api_error__: boom"])
    info = sr.summarize_run(str(run), fake, backend_name="stub", retries=2)
    assert info["fallback"] is True
    assert all(a.get("error") for a in info["attempts"])


# --------------------------------------------------------------------------- #
# mock バックエンド経路が決定論(同入力2回で同出力)
# --------------------------------------------------------------------------- #
def test_mock_backend_is_deterministic(tmp_path):
    run1 = tmp_path / "r1"
    run2 = tmp_path / "r2"
    _write_run(run1)
    _write_run(run2)
    be1 = sr.make_backend("mock")
    be2 = sr.make_backend("mock")
    i1 = sr.summarize_run(str(run1), be1, backend_name="mock")
    i2 = sr.summarize_run(str(run2), be2, backend_name="mock")
    # mock はガードで弾かれフォールバックに落ちる(= それが正しい挙動)
    assert i1["fallback"] is True and i2["fallback"] is True
    # 本文(脚注前)が完全一致 = 決定論
    md1 = (run1 / "panel" / "summary_ja.md").read_text(encoding="utf-8")
    md2 = (run2 / "panel" / "summary_ja.md").read_text(encoding="utf-8")
    body1 = md1.split("---", 1)[0].replace("r1", "RUN")
    body2 = md2.split("---", 1)[0].replace("r2", "RUN")
    assert body1 == body2
