"""歩行者信号(PED-SIG / 第38バッチ)の純関数テスト — 信号ゲート + サイドカー。

シミュ実行は不要。viz/sfm.py(SignalGate)・scripts/synth_crowd.py(信号ゲート合成)・
scripts/build_crossings.py(横断サイドカー抽出)を決定論・軽量に検証する
(test_synth_crowd と同流儀)。検収:
  (A) 位相計算の決定論(サイクル境界・青点滅・スクランブル同相)
  (B) ゲート ON で赤=停止線 curb 滞留・青=解放(SFM 小規模合成)
  (C) 信号パラメータ未指定 = 既存出力バイト一致(gate=None は現行経路を一切変えない)
  (D) サイドカー読み込み + ゲート構築
  (E) build_crossings 抽出(信号有無・スクランブル・サブタグ素通し)
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

from sfm import SignalGate  # noqa: E402
import synth_crowd as sc  # noqa: E402
import build_crossings as bc  # noqa: E402


# ══════════════════════════════ (A) 位相計算の決定論 ══════════════════════════
def test_phase_cycle_boundary():
    """位相はサイクル境界で 0 へ巻き戻る(絶対時刻の純関数)。"""
    g = SignalGate(cycle_s=140, green_s=37, flash_s=10, offset_s=0.0)
    assert g.phase(0.0) == pytest.approx(0.0)
    assert g.phase(140.0) == pytest.approx(0.0)       # 1 周期後 = 境界で 0
    assert g.phase(145.0) == pytest.approx(5.0)
    assert g.phase(280.0) == pytest.approx(0.0)       # 2 周期後
    # offset は位相を平行移動
    g2 = SignalGate(140, 37, 10, offset_s=100.0)
    assert g2.phase(40.0) == pytest.approx(0.0)       # (40+100)%140 = 0


def test_green_flash_red_windows():
    """青(<37)・青点滅([37,47))・赤([47,140))の窓を境界含めて判定。"""
    g = SignalGate(cycle_s=140, green_s=37, flash_s=10, offset_s=0.0)
    # 青
    assert g.is_green(0.0) and g.can_cross(0.0)
    assert g.is_green(36.9) and g.can_cross(36.9)
    # 青点滅: is_green は False だが can_cross は True(渡り切り許容)
    assert not g.is_green(40.0)
    assert g.is_flashing(40.0)
    assert g.can_cross(40.0)
    assert g.can_cross(46.9)
    # 赤: どちらも False(境界 47 は赤)
    assert not g.can_cross(47.0)
    assert not g.is_green(100.0)
    assert not g.can_cross(100.0)
    assert not g.is_flashing(100.0)


def test_phase_deterministic():
    """同一絶対時刻 → 同一 bool(乱数ゼロ=決定論)。"""
    g = SignalGate(140, 37, 10, offset_s=13.0)
    for t in (0.0, 12.3, 47.0, 139.9, 1000.5):
        assert g.can_cross(t) == g.can_cross(t)
        assert g.phase(t) == g.phase(t)


def test_scramble_same_phase():
    """スクランブル=全方向同相(同 offset 0 → 全 gate が同時に青)。"""
    a = SignalGate.scramble(140, 37, 10)
    b = SignalGate.scramble(140, 37, 10)
    assert a.offset_s == 0.0
    for t in np.linspace(0, 300, 61):
        assert a.can_cross(t) == b.can_cross(t)     # 位相が一致=一斉横断


def test_gate_validation():
    """不正パラメータは弾く(cycle<=0 / 青が周期超過)。"""
    with pytest.raises(ValueError):
        SignalGate(cycle_s=0, green_s=1)
    with pytest.raises(ValueError):
        SignalGate(cycle_s=100, green_s=101)
    with pytest.raises(ValueError):
        SignalGate(cycle_s=100, green_s=60, flash_s=50)   # 60+50 > 100


# ══════════════════ (B) SFM 信号ゲート: 赤で curb 滞留・青で解放 ═════════════════
def _crossers():
    return [
        {"agent_id": 1, "entry": (-30.0, 0.0), "exit": (30.0, 0.0),
         "v0": 1.2, "portion_time": 50.0, "portion_dist": 60.0, "t_enter": 0.0},
        {"agent_id": 2, "entry": (0.0, -30.0), "exit": (0.0, 30.0),
         "v0": 1.2, "portion_time": 50.0, "portion_dist": 60.0, "t_enter": 0.0},
    ]


def _samples_by_agent(res):
    by: dict[int, list] = {}
    for (t, a, x, y) in res["samples"]:
        by.setdefault(a, []).append((t, x, y))
    return by


def test_gate_red_holds_at_curb_green_releases():
    """赤(t<30)は停止線(=entry)で滞留・青(t>=30)で解放して横断する。"""
    # offset=30・cycle=60・green=30 → t∈[0,30) 赤 / t∈[30,60) 青(window 0=base_sec 0)
    gate = SignalGate(cycle_s=60, green_s=30, flash_s=0, offset_s=30.0)
    res = sc.simulate_window("gt", 0, _crossers(), 0.1, 0.5, gate=gate)
    assert res.get("gated") is True
    assert "green_frac" in res                          # 追加キーのみ
    by = _samples_by_agent(res)
    ent = {1: (-30.0, 0.0), 2: (0.0, -30.0)}

    for aid, (ex, ey) in ent.items():
        pts = by[aid]
        # 赤の間(t<=25)は entry(curb)に凍結=1mm も動かない
        red = [(x, y) for (t, x, y) in pts if t <= 25.0]
        assert red, "赤フレームのサンプルが無い"
        for (x, y) in red:
            assert (x, y) == pytest.approx((ex, ey), abs=1e-6)
        # 青で解放後(t>=45)は entry から明確に離れて横断している
        late = [(x, y) for (t, x, y) in pts if t >= 45.0]
        assert late, "青フレームのサンプルが無い"
        assert any(np.hypot(x - ex, y - ey) > 3.0 for (x, y) in late), \
            "青になっても横断していない(解放されていない)"


def test_gate_release_is_committed():
    """一度青で解放されたら次の赤で途中停止せず渡り切る(committed crossing)。"""
    gate = SignalGate(cycle_s=60, green_s=30, flash_s=0, offset_s=30.0)
    res = sc.simulate_window("gt2", 0, _crossers(), 0.1, 0.5, gate=gate)
    by = _samples_by_agent(res)
    for aid, pts in by.items():
        xs = [np.hypot(x, y) for (t, x, y) in pts if t >= 30.0]
        # 解放後の中心距離は概ね単調に減ってから増える(横断=entry 側→exit 側)…
        # 少なくとも最終サンプルは exit 近傍(中心を越えて反対側)へ到達している
        last_t, last_x, last_y = pts[-1]
        assert np.hypot(last_x, last_y) > 20.0          # 反対 curb 近くまで到達


def test_gate_deterministic_samples():
    """ゲート ON でも同一入力 → samples 完全一致(rng を追加で引かない=決定論)。"""
    gate = SignalGate(60, 30, 0, offset_s=30.0)
    a = sc.simulate_window("gt3", 0, _crossers(), 0.1, 0.5, gate=gate)
    b = sc.simulate_window("gt3", 0, _crossers(), 0.1, 0.5, gate=gate)
    assert a["samples"] == b["samples"]
    assert a["green_frac"] == b["green_frac"]


# ══════════════════ (C) 信号パラメータ未指定 = 既存出力バイト一致 ══════════════════
def test_gate_none_equals_default_path():
    """gate=None は gate 引数なし(現行呼び出し)と完全一致=現行経路を一切変えない。"""
    crossers = _crossers()
    default = sc.simulate_window("same", 0, crossers, 0.1, 0.5)          # 従来 5 引数
    none = sc.simulate_window("same", 0, crossers, 0.1, 0.5, gate=None)   # 明示 None
    assert default["samples"] == none["samples"]
    assert default == none
    # 未指定経路には gated / green_frac が付かない(契約キー不変)
    assert "gated" not in default
    assert "green_frac" not in default


def test_synth_gate_none_byte_identical(tmp_path):
    """synth(gate=None) の parquet が gate 有りと別・None 同士は byte 一致(既存挙動維持)。"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    run = tmp_path / "run"
    run.mkdir(parents=True, exist_ok=True)
    rows = [
        (0, 1, "move_segment", {"mode": "walk",
         "pts": [[-100.0, 3.0], [100.0, -3.0]], "dist_m": 0.0, "congestion": 1.0}),
        (0, 2, "move_segment", {"mode": "walk",
         "pts": [[0.0, -100.0], [0.0, 100.0]], "dist_m": 0.0, "congestion": 1.0}),
    ]
    table = pa.table({
        "step": pa.array([r[0] for r in rows], type=pa.int32()),
        "agent_id": pa.array([r[1] for r in rows], type=pa.int32()),
        "kind": pa.array([r[2] for r in rows], type=pa.string()),
        "payload": pa.array([json.dumps(r[3]) for r in rows], type=pa.string()),
    })
    pq.write_table(table, run / "l1_events.parquet")

    o1 = tmp_path / "a" / "crowd_tracks.parquet"
    o2 = tmp_path / "b" / "crowd_tracks.parquet"
    sc.synth(run, 0.0, 0.0, 40.0, 0, 0, 0.1, 0.5, o1)                # 既定=gate なし
    sc.synth(run, 0.0, 0.0, 40.0, 0, 0, 0.1, 0.5, o2, gate=None)     # 明示 None
    assert o1.read_bytes() == o2.read_bytes()

    # ゲート有りは別出力(実際に効いている)
    o3 = tmp_path / "c" / "crowd_tracks.parquet"
    gate = SignalGate(60, 30, 0, offset_s=30.0)
    sc.synth(run, 0.0, 0.0, 40.0, 0, 0, 0.1, 0.5, o3, gate=gate)
    assert o3.read_bytes() != o1.read_bytes()


# ══════════════════════════ (D) サイドカー読み込み + ゲート構築 ══════════════════════
def _sidecar_dict():
    return {
        "meta": {"name": "crossings_shibuya",
                 "signal_defaults": {"cycle_s": 140.0, "green_s": 37.0,
                                     "flash_s": 10.0, "red_s": 93.0}},
        "crossings": [
            {"id": 1, "x": 2.0, "y": -1.0, "signal": True, "scramble": True,
             "cycle_s": 140.0, "green_s": 37.0, "flash_s": 10.0, "name": "渋谷駅前"},
            {"id": 2, "x": 300.0, "y": 12.0, "signal": True,
             "cycle_s": 140.0, "green_s": 37.0, "flash_s": 10.0},
            {"id": 3, "x": 5.0, "y": 5.0, "signal": False},   # 信号なし横断
        ],
    }


def test_load_sidecar_roundtrip(tmp_path):
    """サイドカー JSON を書いて読み戻す(load_crossings_sidecar)。"""
    p = tmp_path / "crossings.json"
    p.write_text(json.dumps(_sidecar_dict(), ensure_ascii=False), encoding="utf-8")
    d = sc.load_crossings_sidecar(p)
    assert d is not None
    assert d["meta"]["name"] == "crossings_shibuya"
    assert len(d["crossings"]) == 3
    # 不在 → None(呼び出し側は従来挙動へ)
    assert sc.load_crossings_sidecar(tmp_path / "nope.json") is None


def test_gate_from_sidecar_picks_scramble():
    """中心近傍の信号横断からゲートを作る。スクランブル優先・周期はサイドカー由来。"""
    d = _sidecar_dict()
    gate = sc.gate_from_sidecar(d, 0.0, 0.0, 40.0)
    assert isinstance(gate, SignalGate)
    assert gate.cycle_s == pytest.approx(140.0)
    assert gate.green_s == pytest.approx(37.0)
    assert gate.flash_s == pytest.approx(10.0)
    assert gate.offset_s == 0.0                          # スクランブル=同相


def test_gate_from_sidecar_none_when_no_signal_in_region():
    """領域内に信号付き横断が無ければ None(=信号ゲートなしで続行)。"""
    d = _sidecar_dict()
    # 半径 40m の領域には id=1(信号) と id=3(信号なし)。id=3 だけ拾う中心を選ぶ:
    # 遠方(1000,1000)には何も無い → None
    assert sc.gate_from_sidecar(d, 1000.0, 1000.0, 40.0) is None
    # 信号なし横断(id=3)のみが入る半径でも None(signal=False は対象外)
    only_unsig = {"meta": d["meta"], "crossings": [d["crossings"][2]]}
    assert sc.gate_from_sidecar(only_unsig, 5.0, 5.0, 10.0) is None


# ══════════════════════════ (E) build_crossings 抽出(P0 純関数) ══════════════════
def _raw_nodes():
    # スクランブル交差点(原点)近傍の信号横断・スクランブル・遠方の信号なし横断
    return {"elements": [
        # 信号付き(crossing=traffic_signals)+ サブタグ
        {"type": "node", "id": 10, "lat": 35.6595, "lon": 139.70062,
         "tags": {"highway": "crossing", "crossing": "traffic_signals",
                  "name": "渋谷駅前", "button_operated": "no",
                  "tactile_paving": "yes", "traffic_signals:sound": "yes"}},
        # スクランブル(crossing_ref=pedestrian_scramble・信号タグ無しでも信号扱い)
        {"type": "node", "id": 11, "lat": 35.65952, "lon": 139.70064,
         "tags": {"highway": "crossing", "crossing_ref": "pedestrian_scramble"}},
        # 信号なし横断(marked)= signal False・周期を付けない
        {"type": "node", "id": 12, "lat": 35.6620, "lon": 139.7040,
         "tags": {"highway": "crossing", "crossing": "marked"}},
        # 横断でないノード(無視)
        {"type": "node", "id": 13, "lat": 35.6600, "lon": 139.7000,
         "tags": {"amenity": "cafe"}},
    ]}


def test_build_crossings_signal_and_scramble():
    """信号有無・スクランブル判定・サブタグ素通し・周期付与を検証。"""
    feat = bc.build_crossings(_raw_nodes(),
                              (35.656, 139.695, 35.6625, 139.706))
    m = feat["meta"]
    assert m["n_crossings"] == 3           # cafe は除外
    assert m["n_signalized"] == 2          # id=10(traffic_signals) + id=11(scramble)
    assert m["n_scramble"] == 2            # 両方原点近傍(< 45m)で信号=スクランブル
    by_id = {c["id"]: c for c in feat["crossings"]}
    # 信号付きにだけ周期が付く
    assert by_id[10]["signal"] and by_id[10]["cycle_s"] == 140.0
    assert by_id[11]["signal"] and by_id[11].get("scramble") is True
    assert by_id[12]["signal"] is False and "cycle_s" not in by_id[12]
    # サブタグ素通し(: は _ に正規化)
    assert by_id[10]["button_operated"] == "no"
    assert by_id[10]["tactile_paving"] == "yes"
    assert by_id[10]["traffic_signals_sound"] == "yes"
    # 名前・位置(原点近傍=0 付近)
    assert by_id[10]["name"] == "渋谷駅前"
    assert abs(by_id[10]["x"]) < 5 and abs(by_id[10]["y"]) < 5
    # 決定論の出典表示(ODbL)
    assert "OpenStreetMap" in m["source"]


def test_build_crossings_scramble_detection_paths():
    """スクランブル判定の 2 経路: OSM 権威タグ(距離不問)と 原点近傍フォールバック(無タグ)。"""
    raw = {"elements": [
        # (a) 遠方 + pedestrian_scramble タグ → スクランブル(権威タグは距離不問)
        {"type": "node", "id": 20, "lat": 35.6620, "lon": 139.7040,
         "tags": {"highway": "crossing", "crossing_ref": "pedestrian_scramble"}},
        # (b) 原点近傍 + 信号(タグ無し)→ フォールバックでスクランブル
        {"type": "node", "id": 21, "lat": 35.65952, "lon": 139.70064,
         "tags": {"highway": "crossing", "crossing": "traffic_signals"}},
        # (c) 遠方 + 信号(scramble タグ無し)→ 信号だがスクランブルではない
        {"type": "node", "id": 22, "lat": 35.6620, "lon": 139.7042,
         "tags": {"highway": "crossing", "crossing": "traffic_signals"}},
    ]}
    feat = bc.build_crossings(raw, (35.656, 139.695, 35.6625, 139.706))
    by_id = {c["id"]: c for c in feat["crossings"]}
    assert by_id[20]["signal"] and by_id[20].get("scramble") is True   # (a) 権威タグ
    assert by_id[21]["signal"] and by_id[21].get("scramble") is True   # (b) 近傍フォールバック
    assert by_id[22]["signal"] and by_id[22].get("scramble") is not True  # (c) 遠方無タグ


def test_real_sidecar_if_present():
    """実生成サイドカー(data/crossings_shibuya.json)が在れば構造を軽く検証。"""
    p = _ROOT / "data" / "crossings_shibuya.json"
    if not p.exists():
        pytest.skip("crossings_shibuya.json 未生成(ネットワーク要・build_crossings.py)")
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["meta"]["n_signalized"] > 0
    assert d["meta"]["n_scramble"] > 0
    assert any(c.get("scramble") for c in d["crossings"])
    # 領域(原点±40m)からスクランブルゲートが作れる
    gate = sc.gate_from_sidecar(d, 0.0, 0.0, 40.0)
    assert isinstance(gate, SignalGate)
