"""変更1: 滞在の短縮 + マイクロ移動(ユーザー: 「人は数十分同じ場所にとどまらない」)。

路上・非睡眠のエージェント(勤務・バイト中を除く)が「同一座標に4step以上連続滞在」
しないこと(位置微調整で完全静止が解消される)を集計で確認する。勤務・バイト中は
仕様どおり動かさない(ユーザー: work と睡眠は変えない)ので集計から除く。
"""
from __future__ import annotations

import json

from society.config import load_config
from society.engine.simulation import Simulation


def _is_working(meta: dict, node: str, sim_min: int) -> bool:
    """路面の職場/バイト先で勤務中か(=位置を動かさない対象)。"""
    m = sim_min % 1440
    ws, we = meta["work_start_min"], meta["work_end_min"]
    if ws >= 0 and ws <= m < we and node == meta["work_node"]:
        return True
    pt = meta["part_time"]
    if pt and node == pt["node"]:
        day = (sim_min // 1440) % 7
        if day in pt["days"] and pt["start_min"] <= m < pt["end_min"]:
            return True
    return False


def _max_static_runs(sim) -> dict[int, int]:
    """agent_id -> 路上・非睡眠・非勤務での「同一座標が連続した最大 step 数」。"""
    meta = {a.id: {"work_node": a.work_node, "work_start_min": a.work_start_min,
                   "work_end_min": a.work_end_min, "part_time": a.part_time}
            for a in sim.agents}
    per_agent: dict[int, list[tuple[int, tuple]]] = {}
    for snap in sim.logger.snapshots:
        state = json.loads(snap["state"])
        step = snap["step"]
        sim_min = sim.clock.sim_min(step)
        for a in state["agents"]:
            if a["loc"] != "street" or a["sleeping"] or a["building"]:
                continue
            if _is_working(meta[a["id"]], a["node"], sim_min):
                continue
            per_agent.setdefault(a["id"], []).append((step, (a["x"], a["y"])))
    out: dict[int, int] = {}
    for aid, series in per_agent.items():
        series.sort()
        best = cur = 1
        for i in range(1, len(series)):
            contiguous = series[i][0] == series[i - 1][0] + 1
            if contiguous and series[i][1] == series[i - 1][1]:
                cur += 1
                best = max(best, cur)
            else:
                cur = 1
        out[aid] = best
    return out


def test_street_agents_do_not_stand_still(tmp_path):
    cfg = load_config(["run.seed=42", "run.n_agents=40", "run.n_steps=48",
                       "run.name=stay", "observer.snapshot_every=1"])
    sim = Simulation(cfg, out_dir=tmp_path / "stay")
    sim.run()

    runs = _max_static_runs(sim)
    assert runs, "路上・非睡眠のスナップショットが取れていない"
    static = [aid for aid, best in runs.items() if best >= 4]
    # 位置微調整で、路上滞在(非勤務)の完全静止は起こらない。
    assert not static, f"同一座標に4step以上留まる路上エージェントがいる: {static}"
