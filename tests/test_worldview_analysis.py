"""第20バッチ: 主観的世界モデルの後処理 scripts/analyze_worldview.py のテスト。

対象(読み出し専用スクリプト):
  scripts/analyze_worldview.py … L1 "worldview" イベント + agents.json を読む後処理。
    世界解釈パネル・仮説検証ループ・可制御性の分岐・解釈の分岐・世界観クラスタ・カード。

house style は tests/test_pipeline.py / tests/test_worldview.py に倣い、Simulation +
load_config + tmp_path の mock 短ラン(20体, worldview.enabled=true)を生成してから
後処理をかける。決定論(2回実行で出力一致)と OFF ラン(worldview 0件)のスキップ分岐を検証する。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pyarrow.parquet as pq

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))       # scripts/ は package ではないので path 追加
sys.path.insert(0, str(_ROOT / "src"))

from society.config import load_config             # noqa: E402
from society.engine.simulation import Simulation   # noqa: E402

import analyze_worldview as aw                      # noqa: E402


def _run(tmp_path, name, n=20, steps=432, seed=0, **ov) -> str:
    """mock 短ラン(既定 20体×3日=432step)を tmp_path に生成し run_dir を返す。"""
    dot = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
           f"run.seed={seed}", "observer.snapshot_every=144"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    Simulation(load_config(dot), out_dir=tmp_path / name).run()
    return str(tmp_path / name)


# --------------------------------------------------------------------- ON ラン
def test_on_run_panel_and_report(tmp_path):
    """worldview ON: パネル行数>0・必須列・レポート生成・主要セクションが出る。"""
    rd = _run(tmp_path, "wv_on", **{"worldview.enabled": "true"})
    info = aw.analyze(rd)

    assert info["off_run"] is False
    assert info["n_panel_rows"] > 0, "パネルが空(worldview 行が拾えていない)"

    # パネル parquet の列と行数
    pf = Path(rd) / "panel" / "worldview.parquet"
    assert pf.exists(), "worldview.parquet が出力されていない"
    t = pq.read_table(pf).to_pydict()
    need = {"run", "agent_id", "day", "ctrl", "expect_n", "err_mean", "err_n",
            "norm_rate"}
    assert need <= set(t.keys()), f"欠落列: {need - set(t.keys())}"
    assert len(t["day"]) == info["n_panel_rows"] > 0
    # ctrl は全員 0.5 起点。少なくとも 1 行は数値が入っている。
    assert any(c is not None for c in t["ctrl"]), "ctrl が全欠損"

    # レポート生成 + 主要セクション見出し
    rp = Path(rd) / "worldview_report.md"
    assert rp.exists()
    md = info["md"]
    for head in ("# 世界解釈レポート", "## 1. 仮説検証ループ",
                 "## 2. 可制御性の分岐", "## 3. 解釈の分岐",
                 "## 4. 信念(belief)の世界観クラスタ", "## 5. 世界観カード",
                 "## 付録"):
        assert head in md, f"見出しが無い: {head}"
    assert "worldview OFF ラン" not in md, "ON ランなのに OFF 扱いされた"


def test_on_run_deterministic(tmp_path):
    """同一ランに 2 回 analyze → レポート md とパネル内容が完全一致(決定論)。"""
    rd = _run(tmp_path, "wv_det", **{"worldview.enabled": "true"})
    a = aw.analyze(rd, out_md=str(tmp_path / "r1.md"))
    b = aw.analyze(rd, out_md=str(tmp_path / "r2.md"))
    assert a["md"] == b["md"], "レポートが決定論的でない"
    assert a["n_clusters"] == b["n_clusters"]
    assert a["n_panel_rows"] == b["n_panel_rows"]
    # パネルも 2 回書いて同一(同じパスに上書き=内容一致)
    t1 = pq.read_table(Path(rd) / "panel" / "worldview.parquet").to_pydict()
    t2 = pq.read_table(Path(rd) / "panel" / "worldview.parquet").to_pydict()
    assert t1 == t2


def test_off_run_skips_main_sections(tmp_path):
    """worldview OFF(既定): worldview イベント 0 件 → OFF ラン明記・主要部スキップ。"""
    rd = _run(tmp_path, "wv_off")                 # enabled 指定なし=既定 OFF
    info = aw.analyze(rd)

    assert info["off_run"] is True
    assert info["n_panel_rows"] == 0
    assert info["n_clusters"] == 0
    md = info["md"]
    assert "worldview OFF ラン" in md, "OFF ランの明記が無い"
    # OFF では主要セクション本体を出さない(スキップ分岐)
    assert "## 2. 可制御性の分岐" not in md
    # パネル parquet は作らない
    assert not (Path(rd) / "panel" / "worldview.parquet").exists()


# --------------------------------------------------------------------- 単体
def test_cluster_texts_deterministic():
    """belief クラスタリングは決定論(ソート順+閾値)で、類似文をまとめる。"""
    texts = ["渋谷での経験から、自分は少し考えが変わった気がする。",
             "原宿での経験から、自分は少し考えが変わった気がする。",
             "世界は自分の力では何一つ動かせないと痛感した。"]
    a1, reps1 = aw.cluster_texts(texts, aw._CLUSTER_THRESHOLD)
    a2, reps2 = aw.cluster_texts(list(reversed(texts)), aw._CLUSTER_THRESHOLD)
    assert a1 == a2 and reps1 == reps2, "入力順に依存している(非決定論)"
    # 定型2文は同クラスタ・毛色の違う1文は別クラスタ
    assert a1[texts[0]] == a1[texts[1]]
    assert a1[texts[2]] != a1[texts[0]]
    assert len(reps1) == 2


def test_interpretation_divergence_synthetic():
    """合成イベント: 同じ共有事象に対する発話 valence の個体差が測れる。"""
    def ev(kind, aid, sim_min, payload):
        return {"kind": kind, "agent_id": aid, "step": sim_min // 10,
                "sim_min": sim_min, "x": 0.0, "y": 0.0, "rng_stream": "",
                "llm_call_id": None, "payload": payload}
    t = 5000
    events = [
        ev("proposal_passed", 3, t, {"proposal_id": 1, "text": "x", "supporters": 5}),
        # 事象前(t-1440..t): 中立
        ev("speak", 1, t - 100, {"text": "ふつうの一日だった", "hearers": []}),
        ev("speak", 2, t - 100, {"text": "とくに何もない", "hearers": []}),
        # 事象後(t..t+1440): agent1 はポジ・agent2 はネガ
        ev("speak", 1, t + 100, {"text": "本当に最高で嬉しい、素晴らしい一日", "hearers": []}),
        ev("sns_post", 2, t + 200, {"text": "最悪だ、ひどい、つらい", "items": []}),
    ]
    res = aw.interpretation_analysis(events)
    assert res["has_events"] is True
    row = res["events"][0]
    assert row["kind"] == "提案成立"
    assert row["n_agents"] == 2
    # 最ポジ/最ネガが分かれ、発話例が付く
    assert row["most_pos"]["agent_id"] == 1
    assert row["most_neg"]["agent_id"] == 2
    assert row["most_pos"]["delta"] > row["most_neg"]["delta"]
    assert row["most_pos"]["text"] and row["most_neg"]["text"]
