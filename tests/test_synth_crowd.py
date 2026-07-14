"""案a(オフライン合成)の純関数テスト — SFM コア + スクランブル領域合成。

シミュ実行は不要。合成 move_segment とごく小さな l1_events.parquet を tmp に作り、
viz/sfm.py(SFM コア)と scripts/synth_crowd.py(領域抽出・窓合成・出力)を検証する
(test_analyze_od / test_flows_grid と同流儀の決定論・軽量テスト)。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "viz"))
sys.path.insert(0, str(_ROOT / "scripts"))

import sfm  # noqa: E402
from sfm import Crowd  # noqa: E402
import synth_crowd as sc  # noqa: E402


# ══════════════════════════════════ SFM コア単体 ══════════════════════════════
def test_single_agent_relaxes_to_desired_speed():
    """1 体が希望速度へ τ で緩和する(離散 Euler の厳密値と一致)。"""
    v0 = 1.34
    dt = 0.1
    c = Crowd(pos=[[0.0, 0.0]], vel=[[0.0, 0.0]], goal=[[1000.0, 0.0]],
              v0=[v0], radius=[0.3], noise=0.0)
    # t=τ(=5 ステップ)後の離散 Euler 期待値: v = v0(1 − (1 − dt/τ)^k)
    k = int(round(sfm.TAU_DEFAULT / dt))
    for _ in range(k):
        c.step(dt)
    sp = float(np.linalg.norm(c.vel[0]))
    expected = v0 * (1.0 - (1.0 - dt / sfm.TAU_DEFAULT) ** k)
    assert sp == pytest.approx(expected, rel=1e-6)
    # 連続解の目安 0.632·v0 の近傍でもある
    assert 0.55 * v0 < sp < 0.75 * v0
    # 十分時間が経てば v0 に漸近(1% 以内)
    for _ in range(200):
        c.step(dt)
    sp2 = float(np.linalg.norm(c.vel[0]))
    assert sp2 == pytest.approx(v0, rel=0.01)


def test_two_agents_head_on_pass_without_jamming():
    """対向 2 体がすれ違える: 最接近距離 > 0(固まらず両者とも通過)。"""
    v0 = 1.34
    # 完全な正面衝突コースを避けるため横に僅かにずらす(現実の歩行者も同様)
    c = Crowd(pos=[[-5.0, 0.15], [5.0, -0.15]],
              vel=[[v0, 0.0], [-v0, 0.0]],
              goal=[[10.0, 0.15], [-10.0, -0.15]],
              v0=[v0, v0], radius=[0.3, 0.3], noise=0.0)
    min_d = 1e9
    for _ in range(120):
        c.step(0.1)
        d = float(np.linalg.norm(c.pos[0] - c.pos[1]))
        min_d = min(min_d, d)
    assert min_d > 0.0                      # 一点に潰れない
    assert min_d > 0.5 * (0.3 + 0.3)        # 半径和の半分以上は離れてすれ違う
    # 固まらない = 両者とも中心を越えて反対側へ進む
    assert c.pos[0, 0] > 3.0
    assert c.pos[1, 0] < -3.0


def test_sfm_deterministic_same_conditions():
    """同条件 2 回で軌跡が完全一致(決定論)。"""
    def run():
        c = Crowd(pos=[[-5.0, 0.15], [5.0, -0.15]],
                  vel=[[1.34, 0.0], [-1.34, 0.0]],
                  goal=[[10.0, 0.15], [-10.0, -0.15]],
                  v0=[1.34, 1.34], radius=[0.3, 0.3], noise=0.0)
        traj = []
        for _ in range(80):
            c.step(0.1)
            traj.append(c.pos.copy())
        return np.array(traj)
    assert np.array_equal(run(), run())


def test_sfm_no_nan_and_speed_capped():
    """近接スポーンでも NaN が出ず、速さが v_max を超えない。"""
    v0 = 1.34
    c = Crowd(pos=[[0.0, 0.0], [0.1, 0.0]],        # ほぼ重なり
              vel=[[0.0, 0.0], [0.0, 0.0]],
              goal=[[100.0, 0.0], [-100.0, 0.0]],
              v0=[v0, v0], radius=[0.3, 0.3], noise=0.0)
    for _ in range(50):
        c.step(0.1)
        assert not np.isnan(c.pos).any()
        speed = np.linalg.norm(c.vel, axis=1)
        assert np.all(speed <= sfm.V_MAX_FACTOR * v0 + 1e-9)


# ══════════════════════════════ 領域抽出(circle_crossing) ══════════════════════
def test_region_extraction_through():
    """円を貫通する直線区間の入口/出口/弧長/所要時間。"""
    cr = sc.circle_crossing([[-100.0, 0.0], [100.0, 0.0]], 0.0, 0.0, 40.0)
    assert cr is not None
    assert cr["entry"] == pytest.approx((-40.0, 0.0))
    assert cr["exit"] == pytest.approx((40.0, 0.0))
    assert cr["portion_dist"] == pytest.approx(80.0)
    # 総弧長 200m を 600s で辿る前提 → 80m 区間は 240s
    assert cr["portion_time"] == pytest.approx(80.0 / 200.0 * 600.0)
    # 入口/出口の弧長割合(=step内時刻の根拠): 中心貫通は 0.3/0.7
    assert cr["f_in"] == pytest.approx((100.0 - 40.0) / 200.0)
    assert cr["f_out"] == pytest.approx((100.0 + 40.0) / 200.0)


def test_region_extraction_miss_returns_none():
    """円をかすりもしない区間は None(通過に数えない)。"""
    assert sc.circle_crossing([[-100.0, 300.0], [100.0, 300.0]], 0.0, 0.0, 40.0) is None


def test_region_extraction_starts_inside():
    """円内(中心)から外へ出る区間は端点が入口になる。"""
    cr = sc.circle_crossing([[0.0, 0.0], [100.0, 0.0]], 0.0, 0.0, 40.0)
    assert cr is not None
    assert cr["entry"] == pytest.approx((0.0, 0.0))
    assert cr["exit"] == pytest.approx((40.0, 0.0))


# ═══════════════════════════ 合成の end-to-end(tmp parquet) ═══════════════════
def _write_l1(tmp: Path, rows: list) -> Path:
    """rows=[(step, agent_id, kind, payload_dict)] から最小 l1_events.parquet を作る。"""
    import pyarrow as pa
    import pyarrow.parquet as pq
    run = tmp / "run"
    run.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "step": pa.array([r[0] for r in rows], type=pa.int32()),
        "agent_id": pa.array([r[1] for r in rows], type=pa.int32()),
        "kind": pa.array([r[2] for r in rows], type=pa.string()),
        "payload": pa.array([json.dumps(r[3]) if r[3] is not None else None
                             for r in rows], type=pa.string()),
    })
    pq.write_table(table, run / "l1_events.parquet")
    return run


def _seg(pts, mode="walk"):
    return {"mode": mode, "pts": pts, "dist_m": 0.0, "congestion": 1.0}


def test_synth_end_to_end_extraction_and_degeneracy(tmp_path):
    """領域抽出の正否 + 通過0窓の縮退 + NaN なし を一括検証。"""
    rows = [
        # step0: 貫通(walk)= 合成対象
        (0, 1, "move_segment", _seg([[-100.0, 0.0], [100.0, 0.0]])),
        # step0: 外れ(walk)= 通過に数えない
        (0, 2, "move_segment", _seg([[-100.0, 300.0], [100.0, 300.0]])),
        # step1: move_segment なし → データ不足窓
        (1, 3, "arrive", None),
        # step2: 貫通だが car → 歩行者群集ではないので除外 → データ不足窓
        (2, 4, "move_segment", _seg([[-100.0, 0.0], [100.0, 0.0]], mode="car")),
    ]
    run = _write_l1(tmp_path, rows)
    out = tmp_path / "panel" / "crowd_tracks.parquet"
    res = sc.synth(run, 0.0, 0.0, 40.0, 0, 2, 0.1, 0.5, out)

    # step0 だけ合成、step1/step2 はデータ不足
    assert res["n_windows"] == 1
    assert res["n_empty"] == 2
    assert res["total_crossers"] == 1
    assert res["n_rows"] > 0
    assert out.exists()
    assert res["html"].exists()

    # NaN なし
    import pyarrow.parquet as pq
    t = pq.read_table(out)
    for col in ("t_s", "x", "y"):
        arr = np.asarray(t[col])
        assert not np.isnan(arr).any()
    assert set(t.schema.names) == {"window", "t_s", "agent_id", "x", "y"}
    # 縮退窓(step1,2)は samples を持たない = window 列に現れない
    assert set(np.asarray(t["window"]).tolist()) == {0}


def test_synth_all_empty_produces_zero_rows(tmp_path):
    """範囲内に通過が全く無ければ 0 行 parquet・全窓データ不足。"""
    rows = [(0, 1, "move_segment", _seg([[-100.0, 300.0], [100.0, 300.0]]))]  # 外れのみ
    run = _write_l1(tmp_path, rows)
    out = tmp_path / "panel" / "crowd_tracks.parquet"
    res = sc.synth(run, 0.0, 0.0, 40.0, 0, 0, 0.1, 0.5, out)
    assert res["n_windows"] == 0
    assert res["n_empty"] == 1
    assert res["n_rows"] == 0


def test_synth_parquet_byte_identical_same_seed(tmp_path):
    """同 seed(=同ラン名・同窓)の 2 回合成で crowd_tracks.parquet が byte 一致。"""
    rows = [
        (0, 1, "move_segment", _seg([[-100.0, 5.0], [100.0, -5.0]])),
        (0, 2, "move_segment", _seg([[-100.0, -8.0], [100.0, 8.0]])),
        (0, 3, "move_segment", _seg([[0.0, -100.0], [0.0, 100.0]])),
    ]
    run = _write_l1(tmp_path, rows)
    out1 = tmp_path / "a" / "crowd_tracks.parquet"
    out2 = tmp_path / "b" / "crowd_tracks.parquet"
    sc.synth(run, 0.0, 0.0, 40.0, 0, 0, 0.1, 0.5, out1)
    sc.synth(run, 0.0, 0.0, 40.0, 0, 0, 0.1, 0.5, out2)
    assert out1.read_bytes() == out2.read_bytes()


def test_simulate_window_deterministic():
    """simulate_window: 同一入力で軌跡サンプルが完全一致。"""
    crossers = [
        {"agent_id": 7, "entry": (-40.0, 1.0), "exit": (40.0, -1.0),
         "v0": 1.2, "portion_time": 66.0, "portion_dist": 80.0, "t_enter": 30.0},
        {"agent_id": 3, "entry": (0.0, -40.0), "exit": (0.0, 40.0),
         "v0": 1.0, "portion_time": 80.0, "portion_dist": 80.0, "t_enter": 120.0},
    ]
    a = sc.simulate_window("runX", 5, crossers, 0.1, 0.5)
    b = sc.simulate_window("runX", 5, crossers, 0.1, 0.5)
    assert a["samples"] == b["samples"]
    assert a["seed"] == b["seed"]
    # 平均所要時間の保存: 各体の portion_time はそのまま統計に載る
    assert a["meso_seconds"] == pytest.approx(66.0 + 80.0)


def test_desired_speed_preserves_meso_time():
    """v0 = 区間距離 / メゾ所要時間 が厳密に成り立つ(平均所要時間の保存)。"""
    cr = sc.circle_crossing([[-100.0, 10.0], [100.0, -10.0]], 0.0, 0.0, 40.0)
    assert cr is not None
    v0 = cr["portion_dist"] / cr["portion_time"]
    assert v0 == pytest.approx(cr["portion_dist"] / cr["portion_time"])
    # 弧長一定速度なので v0 は総弧長/600s(=メゾ実効速度)に一致
    total = np.hypot(200.0, 20.0)
    assert v0 == pytest.approx(total / sc.STEP_SECONDS, rel=1e-9)
