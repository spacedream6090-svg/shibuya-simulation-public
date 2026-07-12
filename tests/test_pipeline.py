"""データパイプライン + 商業レポートのテスト(第18バッチ②)。

対象スクリプト(すべて読み出し専用の後処理):
  scripts/build_panel.py      … stage0 検証 + stage1 tidy パネル(agent_day / run_day)
  scripts/commercial_report.py … 商業4観点レポート(人流商圏/広告/消費売上/トレンドROI)
  scripts/panel_stats.py       … stage2 シード横断 CI + ペア比較

house style は tests/test_free_action.py に倣い、Simulation + load_config + tmp_path の
mock 短ラン(20体×2日=288step)を各テスト内で生成してから後処理をかける。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pyarrow.parquet as pq

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))       # scripts/ は package ではないので path 追加
sys.path.insert(0, str(_ROOT / "src"))

from society.config import load_config             # noqa: E402
from society.engine.simulation import Simulation   # noqa: E402

import build_panel                                 # noqa: E402
import commercial_report                           # noqa: E402
import panel_stats                                 # noqa: E402


def _run(tmp_path, name, n=20, steps=288, seed=0, **ov) -> str:
    """mock 短ラン(既定 20体×2日)を tmp_path に生成し run_dir パスを返す。"""
    dot = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
           f"run.seed={seed}", "observer.snapshot_every=144"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    Simulation(load_config(dot), out_dir=tmp_path / name).run()
    return str(tmp_path / name)


# --------------------------------------------------------------------- build_panel
def test_build_panel_shapes_columns_warmup(tmp_path):
    """行数=agents×days・必須列・warmup フラグ(warmup_days=1 で両分岐が出る)。"""
    rd = _run(tmp_path, "mk")
    info = build_panel.build(rd, warmup_days=1)

    ad = pq.read_table(Path(rd) / "panel" / "agent_day.parquet")
    assert info["n_days"] == 2, "288step は step基準で 2 日のはず"
    assert ad.num_rows == info["n_agents"] * info["n_days"], "行数≠agents×days"

    need = {"run", "agent_id", "name", "age", "occupation", "visitor", "day",
            "weekday", "warmup", "sleep_h", "work_h", "wage_income",
            "spend_total", "spend_food", "n_speak", "n_heard", "n_sns_post",
            "n_sns_react_recv", "move_dist_m", "n_enter_building", "n_poi_visit",
            "wealth_end"}
    assert need <= set(ad.column_names), f"欠落列: {need - set(ad.column_names)}"

    d = ad.to_pydict()
    # warmup フラグ: day<warmup_days(=1)のみ True。両値が出る(day0=True, day1=False)。
    for i in range(ad.num_rows):
        assert d["warmup"][i] == (d["day"][i] < 1)
    assert any(d["warmup"]) and not all(d["warmup"]), "warmup が両分岐で出ていない"

    # run_day: 較正指標 + 多様性・ジニの日次列
    rday = pq.read_table(Path(rd) / "panel" / "run_day.parquet")
    assert rday.num_rows == info["n_days"]
    for c in ("engel", "spend_pp", "wage_pp", "sleep_h_median", "status_gini",
              "wealth_gini", "n_distinct_items_cum"):
        assert c in rday.column_names

    # stage0 検証サマリが出力されている
    assert os.path.exists(Path(rd) / "panel" / "validation.json")
    assert info["validation"]["n_events"] > 0
    assert info["validation"]["unknown_kinds"] == [], "schema 未登録 kind が出た"


def test_build_panel_deterministic(tmp_path):
    """同一入力 → 2回ビルドの DataFrame 内容一致(pyarrow to_pydict 比較)。"""
    rd = _run(tmp_path, "det")
    build_panel.build(rd, warmup_days=1)                       # 既定 out=rd/panel
    b2 = Path(rd) / "panel_b"
    build_panel.build(rd, warmup_days=1, out_dir=str(b2))
    for fn in ("agent_day.parquet", "run_day.parquet"):
        t1 = pq.read_table(Path(rd) / "panel" / fn).to_pydict()
        t2 = pq.read_table(b2 / fn).to_pydict()
        assert t1 == t2, f"{fn} が決定論的でない"


# ---------------------------------------------------------------- commercial_report
def test_commercial_report_generates_and_no_ads(tmp_path):
    """エラーなく生成・「広告なしラン」分岐・footfall/dwell/消費に実数が入る。"""
    rd = _run(tmp_path, "cm")
    out_md = str(tmp_path / "cm" / "commercial_report.md")
    info = commercial_report.report(rd, out_md=out_md)

    assert os.path.exists(out_md)
    md = info["md"]
    # 広告なしラン分岐(ad_exposure は mock/旧ランでは必ず0件)
    assert info["has_ads"] is False
    assert "広告なしラン" in md

    # 人流: footfall 実数・dwell 実数
    assert info["foot"]["footfall"], "footfall が空(実数なし)"
    assert info["foot"]["dwell"]["n"] > 0, "dwell(退出観測)が 0"
    assert info["foot"]["dwell"]["median"] is not None

    # 消費: spend 実数・カテゴリ別売上に実額が入る(店舗帰属/客単価は在館中 spend が
    # 無いと 0=「データ不足」表示になり得るので実数要件はカテゴリ売上で確認する)
    assert info["sales"]["n_spend_events"] > 0, "spend が 0"
    assert info["sales"]["cat_total"], "カテゴリ別売上が空"
    assert sum(info["sales"]["cat_total"].values()) > 0, "カテゴリ別売上が実額 0"
    assert "## 3. 消費・売上" in md

    # 補助 parquet(panel/commercial_*.parquet)
    assert (Path(rd) / "panel" / "commercial_footfall.parquet").exists()
    assert (Path(rd) / "panel" / "commercial_sales.parquet").exists()


def test_ads_conversion_requires_target_building():
    """広告転換の照合: 無関係な建物への入館は転換に数えない・target 建物のみ転換・
    非接触対照とリフトが出る・転換時 spend は target 在館分のみ(純関数の単体)。"""
    def ev(kind, aid, step, payload):
        return {"kind": kind, "agent_id": aid, "step": step,
                "sim_min": 420 + step * 10, "x": 0.0, "y": 0.0,
                "rng_stream": "", "llm_call_id": None, "payload": payload}

    def ad(aid, step, tgt="テスト店"):
        return ev("ad_exposure", aid, step,
                  {"campaign": "c1", "target": tgt, "slot": "s",
                   "cat": "shop", "n_seen": 1})

    events = [
        # agent 0: 接触後、無関係な建物 bX にだけ入館 → 転換に数えてはならない
        ad(0, 1),
        ev("enter_building", 0, 5, {"building": "bX"}),
        ev("exit_building", 0, 8, {"building": "bX"}),
        # agent 1: 接触後、target 建物 bT に入館+在館 spend 500円 → 転換1
        ad(1, 2),
        ev("enter_building", 1, 6, {"building": "bT"}),
        ev("spend", 1, 7, {"amount": 500.0, "cat": "shop"}),
        ev("exit_building", 1, 9, {"building": "bT"}),
        # agent 2: 非接触で bT に入館 → 対照側の来店(転換ではない)
        ev("enter_building", 2, 10, {"building": "bT"}),
        ev("exit_building", 2, 12, {"building": "bT"}),
        # agent 3: 非接触・入館なし
    ]
    mp = {"bname": {"bT": "テスト店", "bX": "無関係ビル"}, "bcat": {},
          "node_xy": {}, "poi_building": {"テスト店": {"bT"}}, "resolved": True}
    agents = [{"id": i} for i in range(4)]

    sessions = commercial_report.build_sessions(events)
    ads = commercial_report.analyze_ads(events, sessions, mp, agents)

    assert ads["has_ads"] is True
    # 分母=接触×target ペア2(agent0, agent1)。転換=agent1 のみ(bX 入館は数えない)
    assert ads["conversion_targets"] == 2
    assert ads["conversions"] == 1, "無関係な建物への入館が転換に数えられている"
    assert abs(ads["conversion_rate"] - 0.5) < 1e-9
    # 非接触対照= c1 非接触 {2,3} のうち bT 来店 {2} → 50%。リフト= 50−50 = 0pt
    assert abs(ads["control_rate"] - 0.5) < 1e-9
    assert abs(ads["lift"] - 0.0) < 1e-9
    # 転換時 spend は target 在館分のみ(agent1 の 500円)
    assert ads["post_visit_spend"]["n"] == 1
    assert abs(ads["post_visit_spend"]["mean"] - 500.0) < 1e-9
    # 解決不可 target なし
    assert ads["targets_unresolved"] == []


def test_ads_unresolved_target_excluded():
    """地図で解決できない target は分母から除外され、件数が明記される。"""
    def ev(kind, aid, step, payload):
        return {"kind": kind, "agent_id": aid, "step": step,
                "sim_min": 420 + step * 10, "x": 0.0, "y": 0.0,
                "rng_stream": "", "llm_call_id": None, "payload": payload}
    events = [
        ev("ad_exposure", 0, 1, {"campaign": "c1", "target": "存在しない店",
                                 "slot": "s", "cat": "shop", "n_seen": 1}),
        ev("enter_building", 0, 5, {"building": "bX"}),
    ]
    mp = {"bname": {"bX": "無関係ビル"}, "bcat": {}, "node_xy": {},
          "poi_building": {}, "resolved": True}
    sessions = commercial_report.build_sessions(events)
    ads = commercial_report.analyze_ads(events, sessions, mp, [{"id": 0}])
    assert ads["conversion_targets"] == 0, "解決不可 target が分母に入っている"
    assert ads["targets_unresolved"] == ["存在しない店"]
    assert ads["excluded_unresolved_pairs"] == 1


# -------------------------------------------------------------------- panel_stats
def test_panel_stats_ci_two_seeds(tmp_path):
    """2ラン(seed違い)でシード横断 CI 表が出る。"""
    a = _run(tmp_path, "cond_s1", seed=1)
    b = _run(tmp_path, "cond_s2", seed=2)
    out = tmp_path / "exp"
    info = panel_stats.run_stats([a, b], str(out))

    assert (out / "panel_stats.parquet").exists()
    assert (out / "panel_combined.parquet").exists()
    assert "95%CI" in info["md"]

    t = pq.read_table(out / "panel_stats.parquet").to_pydict()
    # 全ラン横断(__all__)で n=2 の CI 行が少なくとも1つある
    all_n2 = [i for i in range(len(t["metric"]))
              if t["condition"][i] == "__all__" and t["n_runs"][i] == 2]
    assert all_n2, "n=2 の CI 行が無い"
    # CI 幅が算出されている(下限<上限)行がある
    assert any(t["ci_lo"][i] is not None and t["ci_hi"][i] is not None
               and t["ci_lo"][i] <= t["ci_hi"][i] for i in all_n2)
