"""①人流ヒートマップ + ③混雑 LOS の純関数テスト(第30バッチ・分析スイート W1)。

scripts/analyze_flows_grid.py の空間ビニング・LOS 格付け・unique 集計・建物 centroid
フォールバックを、合成イベントで検証する(シミュ実行不要=決定論の軽量テスト)。
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "src"))

import analyze_flows_grid as fg   # noqa: E402


def _ev(kind, aid, sim_min, x, y, payload=None, step=None):
    """合成 L1 イベント(payload は dict をそのまま渡す=aggregate の期待形)。"""
    if step is None:
        step = sim_min // 10          # 適当な step(day 計算用・10分刻み)
    return {"kind": kind, "agent_id": aid, "sim_min": sim_min, "step": step,
            "x": x, "y": y, "payload": payload or {}}


# --------------------------------------------------------------- ビニング(cell/hour)
def test_cell_and_hour_binning():
    # cell_of: 25m メッシュ・負座標・辺長違い
    assert fg.cell_of(10, 10, 25) == (0, 0)
    assert fg.cell_of(60, 10, 25) == (2, 0)      # 60//25 = 2
    assert fg.cell_of(-5, -5, 25) == (-1, -1)    # floor(-0.2) = -1
    assert fg.cell_of(60, 10, 50) == (1, 0)      # 辺長 50 で 60//50 = 1
    # hour_bin_of: 既定 1h / 2h ビン / 日跨ぎは同一時刻ビンへ
    assert fg.hour_bin_of(480, 1) == 8           # 08:00
    assert fg.hour_bin_of(1439, 1) == 23
    assert fg.hour_bin_of(1440, 1) == 0          # 翌日 00:00 → bin 0
    assert fg.hour_bin_of(480, 2) == 4           # 2h ビン → 480//120


def test_pass_and_present_counts():
    """move_segment は触れた全ユニークセルへ pass +1、arrive は在圏 present +1。"""
    events = [
        # 2セルを跨ぐ移動(pts=[(10,10),(60,10)] → cells (0,0),(1,0),(2,0))
        _ev("move_segment", 0, 480, 10, 10, {"pts": [[10, 10], [60, 10]]}),
        # 同時間帯・同セルに到着(present)
        _ev("arrive", 0, 480, 10, 10, {"node": "n1"}),
    ]
    agg = fg.aggregate(events, cell_m=25, time_bin_h=1)
    cols = fg.grid_rows(agg)
    idx = {(cols["cell_x"][i], cols["cell_y"][i], cols["hour_bin"][i]): i
           for i in range(len(cols["cell_x"]))}
    # 通過は 3 セル(from/mid/to+端点のユニーク)に 1 ずつ
    assert cols["pass_count"][idx[(0, 0, 8)]] == 1
    assert cols["pass_count"][idx[(1, 0, 8)]] == 1
    assert cols["pass_count"][idx[(2, 0, 8)]] == 1
    # 在圏は arrive のセルのみ
    assert cols["present_count"][idx[(0, 0, 8)]] == 1
    assert cols["present_count"][idx[(2, 0, 8)]] == 0
    # 出力列は仕様どおり6列
    assert set(cols) == {"cell_x", "cell_y", "hour_bin", "pass_count",
                         "present_count", "unique_agents"}


def test_unique_agents_dedup():
    """同一セル×時間帯の distinct 体数(同一 agent の重複は1体)。"""
    events = [
        _ev("arrive", 0, 480, 10, 10, {"node": "n1"}),
        _ev("arrive", 1, 485, 12, 12, {"node": "n1"}),   # 同セル(0,0)・別 agent
        _ev("arrive", 0, 489, 11, 11, {"node": "n1"}),   # 同 agent 0 の再到着
    ]
    agg = fg.aggregate(events, cell_m=25, time_bin_h=1)
    cols = fg.grid_rows(agg)
    i = [j for j in range(len(cols["cell_x"]))
         if (cols["cell_x"][j], cols["cell_y"][j], cols["hour_bin"][j]) == (0, 0, 8)][0]
    assert cols["present_count"][i] == 3          # 到着イベント3件
    assert cols["unique_agents"][i] == 2          # distinct 体は {0,1}


def test_los_grading_boundaries():
    """Fruin LOS(人/m²)の格付け(境界含む)。閾値=面積(m²/ped)の逆数。"""
    assert fg.los_grade(0.20) == "A"              # < 1/3.2 = 0.3125
    assert fg.los_grade(1.0 / 3.2) == "B"         # 境界 0.3125 は B 側
    assert fg.los_grade(0.35) == "B"
    assert fg.los_grade(0.50) == "C"
    assert fg.los_grade(0.90) == "D"
    assert fg.los_grade(1.50) == "E"
    assert fg.los_grade(2.0) == "F"               # 境界 2.0(=1/0.5)は F
    assert fg.los_grade(3.0) == "F"


def test_building_centroid_and_fallback():
    """在館は建物 centroid セルへ。centroid 不明の建物は event (x,y) へフォールバック。"""
    events = [
        # b1 の入館: door(x,y)=(999,999) でも centroid (10,10)→cell(0,0) に載せる
        _ev("enter_building", 0, 480, 999, 999, {"building": "b1"}),
        # bX は centroid 表に無い → event (60,10)→cell(2,0) にフォールバック
        _ev("enter_building", 1, 480, 60, 10, {"building": "bX"}),
    ]
    centroids = {"b1": (10.0, 10.0)}
    agg = fg.aggregate(events, cell_m=25, time_bin_h=1, centroids=centroids)
    present = agg["present"]
    assert present[(0, 0, 8)] == 1                # centroid セル
    assert present[(2, 0, 8)] == 1                # フォールバック(door 座標)
    assert (39, 39, 8) not in present             # door(999,999)には載らない


def test_los_table_and_crowd_surge_capture():
    """LOS ランキング(密度=present/n_days/面積)と crowd_surge の取り込み。"""
    events = [
        # 同セルへ多数の在圏観測を集めて高密度(cell面積=625m²)
        *[_ev("arrive", a, 480, 10, 10, {"node": "n1"}) for a in range(1000)],
        _ev("crowd_surge", -1, 480, 10, 10,
            {"node": "scramble", "level": 1000, "event": "halloween"}),
    ]
    agg = fg.aggregate(events, cell_m=25, time_bin_h=1)
    cols = fg.grid_rows(agg)
    rows = fg.los_table(cols, cell_m=25, n_days=1)
    assert rows, "LOS 行が空"
    top = rows[0]
    # 1000 obs / 1日 / 625 m² = 1.6 人/m² → LOS E(1.111..2.0)
    assert abs(top["density_ppm2"] - 1000 / 625) < 1e-9
    assert top["los"] == "E"
    # crowd_surge が発生セルに取り込まれている
    assert len(agg["surges"]) == 1
    s = agg["surges"][0]
    assert (s["cell_x"], s["cell_y"], s["hour_bin"]) == (0, 0, 8)
    assert s["level"] == 1000
