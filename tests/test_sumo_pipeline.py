"""SUMO v0 オフライン合成パイプラインの純関数テスト(第37バッチ Track S2 / SUMO-R v0)。

scripts/sumo_pipeline.py の座標写像・fcd 10 分窓集約・TAZ/OD XML 整形・SUMO 不在時の
check ガイドを、合成データ + subprocess モックで検証する。**SUMO 不在でも通る設計**
(test_analyze_od / test_flows_grid と同流儀=決定論の軽量テスト・シミュ実行不要)。
"""
from __future__ import annotations

import io
import sys
import types
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

import sumo_pipeline as sp   # noqa: E402

ORIGIN = (35.65950, 139.70062)   # スクランブル交差点(全地図共通の局所原点)


# ─────────────────────────────────────────── 経緯度 ↔ local-m 往復
def test_project_inv_project_roundtrip():
    # 原点はちょうど (0,0)
    x0, y0 = sp.project(ORIGIN[0], ORIGIN[1], ORIGIN)
    assert abs(x0) < 1e-9 and abs(y0) < 1e-9
    # いくつかの local-m 点を経緯度へ→戻して一致(mm 精度)
    for (x, y) in [(0.0, 0.0), (996.0, -361.6), (-922.9, 184.8), (250.5, 999.9)]:
        lat, lon = sp.inv_project(x, y, ORIGIN)
        x2, y2 = sp.project(lat, lon, ORIGIN)
        assert abs(x2 - x) < 1e-3 and abs(y2 - y) < 1e-3
    # 北向きに +110.54m ≒ +0.001 度緯度
    lat, lon = sp.inv_project(0.0, 110.54, ORIGIN)
    assert abs(lat - (ORIGIN[0] + 0.001)) < 1e-9


# ─────────────────────────────────────────── ゾーン重心 → 経緯度
def test_zone_centroid_lonlat():
    node_xy = {"n123": (996.0, -361.6)}
    # 地区セル D:cx:cy → セル中心((cx+0.5)*m,(cy+0.5)*m)
    lon, lat = sp.zone_centroid_lonlat("D:2:-1", 100.0, node_xy, ORIGIN)
    x, y = sp.project(lat, lon, ORIGIN)
    assert abs(x - 250.0) < 1e-3 and abs(y - (-50.0)) < 1e-3
    # 域外ゲートウェイ G:node → 地図ノード座標
    lon, lat = sp.zone_centroid_lonlat("G:n123", 100.0, node_xy, ORIGIN)
    x, y = sp.project(lat, lon, ORIGIN)
    assert abs(x - 996.0) < 1e-2 and abs(y - (-361.6)) < 1e-2
    # 未知ゲートウェイ・不正 id は None(捏造しない)
    assert sp.zone_centroid_lonlat("G:nZZZ", 100.0, node_xy, ORIGIN) is None
    assert sp.zone_centroid_lonlat("X:foo", 100.0, node_xy, ORIGIN) is None


def test_taz_id_sanitization():
    assert sp.taz_id("D:-10:-1") == "D_-10_-1"
    assert sp.taz_id("G:n123") == "G_n123"


# ─────────────────────────────────────────── fcd 解析
def test_parse_fcd_stream():
    xml = (
        '<fcd-export>\n'
        f'  <timestep time="0.00">\n'
        f'    <vehicle id="v0" x="{ORIGIN[1]}" y="{ORIGIN[0]}" speed="0"/>\n'
        f'    <vehicle id="v1" x="{ORIGIN[1]}" y="{ORIGIN[0]}" speed="1"/>\n'
        f'  </timestep>\n'
        f'  <timestep time="600.50">\n'
        f'    <vehicle id="v0" x="139.701" y="35.660" speed="3"/>\n'
        f'  </timestep>\n'
        '</fcd-export>\n')
    recs = sp.parse_fcd_stream(io.StringIO(xml))
    assert len(recs) == 3
    assert recs[0][0] == 0.0 and recs[0][1] == "v0"
    assert abs(recs[0][2] - ORIGIN[1]) < 1e-9 and abs(recs[0][3] - ORIGIN[0]) < 1e-9
    assert recs[2][0] == 600.5 and recs[2][1] == "v0"


# ─────────────────────────────────────────── 10 分窓集約(数値検証)
def test_aggregate_fcd_window_binning_and_coords():
    # local-m の狙い点を経緯度へ変換して fcd 相当の geo レコードを作る
    def geo(t, veh, x, y):
        lat, lon = sp.inv_project(x, y, ORIGIN)
        return (t, veh, lon, lat)

    records = [
        geo(0.0, "a", 10.0, 20.0),      # window 0
        geo(300.0, "a", 40.0, 20.0),    # window 0(同一車両・同一窓=1 polyline)
        geo(599.9, "b", -5.0, -5.0),    # window 0
        geo(600.0, "a", 100.0, 100.0),  # window 1
        geo(1200.0, "c", 0.0, 0.0),     # window 2
    ]
    par, segs = sp.aggregate_fcd(records, ORIGIN, step_seconds=600.0, max_seg_pts=24)

    # 窓番号 = t // 600
    assert set(segs.keys()) == {0, 1, 2}
    # window0: 車両 a(2 点)と b(1 点)→ 2 polyline
    assert segs[0][0] == [[10.0, 20.0], [40.0, 20.0]]      # a: 時刻順の 2 点
    assert segs[0][1] == [[-5.0, -5.0], [-5.0, -5.0]]      # b: 1 点は同点複製
    # window1: a の 1 点
    assert segs[1] == [[[100.0, 100.0], [100.0, 100.0]]]
    # parquet 行数 = 全サンプル点数(a:2 + b:1 + a:1 + c:1 = 5)
    assert len(par["step"]) == 5
    assert par["step"] == [0, 0, 0, 1, 2]
    assert par["veh_id"] == ["a", "a", "b", "a", "c"]
    # 座標往復(1 桁丸め内)
    assert abs(par["x"][0] - 10.0) < 0.1 and abs(par["y"][0] - 20.0) < 0.1


def test_downsample_keeps_endpoints():
    pts = [(i, float(i), 0.0) for i in range(10)]
    assert sp._downsample(pts, 20) == pts           # k>=len はそのまま
    ds = sp._downsample(pts, 4)
    assert len(ds) == 4
    assert ds[0] == pts[0] and ds[-1] == pts[-1]    # 先頭・末尾を含む


def test_aggregate_fcd_deterministic():
    def geo(t, veh, x, y):
        lat, lon = sp.inv_project(x, y, ORIGIN)
        return (t, veh, lon, lat)
    recs = [geo(0, "b", 1, 1), geo(0, "a", 2, 2), geo(10, "a", 3, 3)]
    p1, s1 = sp.aggregate_fcd(recs, ORIGIN)
    p2, s2 = sp.aggregate_fcd(list(reversed(recs)), ORIGIN)
    # 入力順に依らず (window, veh_id) 昇順で安定=同一出力
    assert p1 == p2 and s1 == s2
    assert p1["veh_id"][0] == "a"    # a が b より先(辞書順)


# ─────────────────────────────────────────── TAZ / tazRelation XML
def test_build_taz_xml():
    xml = sp.build_taz_xml({"G:n1": ["e1", "e2"], "D:0:0": ["e3"]})
    assert '<taz id="D_0_0" edges="e3"/>' in xml
    assert '<taz id="G_n1" edges="e1 e2"/>' in xml
    # ソート固定(D_0_0 が G_n1 より前)
    assert xml.index("D_0_0") < xml.index("G_n1")


def test_build_tazrel_xml_drops_unmapped_and_groups_by_hour():
    rows = [
        {"origin": "D:0:0", "dest": "D:1:1", "hour_bin": 8, "trips": 5},
        {"origin": "D:0:0", "dest": "D:1:1", "hour_bin": 8, "trips": 3},
        {"origin": "D:0:0", "dest": "G:xOUT", "hour_bin": 9, "trips": 7},  # dest 未写像→落とす
    ]
    kept = {"D:0:0", "D:1:1"}
    xml, stats = sp.build_tazrel_xml(rows, kept)
    assert '<interval id="h8" begin="28800" end="32400">' in xml
    assert '<tazRelation from="D_0_0" to="D_1_1" count="5"/>' in xml
    assert "G_xOUT" not in xml                     # 未写像ゾーンは出さない
    assert stats["n_rel_kept"] == 2 and stats["n_rel_dropped"] == 1
    assert stats["trips_kept"] == 8 and stats["trips_dropped"] == 7
    assert stats["hours"] == [8]


def test_build_segs_json_shape():
    segs = {0: [[[1.0, 2.0], [3.0, 4.0]]], 2: [[[0.0, 0.0], [0.0, 0.0]]]}
    import json
    data = json.loads(sp.build_segs_json(segs, {"run": "t"}))
    assert data["step_seconds"] == 600
    assert [s["step"] for s in data["steps"]] == [0, 2]     # ソート固定
    assert data["steps"][0]["n"] == 1
    assert data["steps"][0]["segs"] == [[[1.0, 2.0], [3.0, 4.0]]]


# ─────────────────────────────────────────── check(SUMO 不在→ガイド / pip 回復 / 既存)
class _FakeProc:
    def __init__(self, rc, out="", err=""):
        self.returncode, self.stdout, self.stderr = rc, out, err


def test_sumo_version_parses(monkeypatch):
    monkeypatch.setattr(sp, "find_bin", lambda name: "sumo" if name == "sumo" else None)
    runner = lambda *a, **k: _FakeProc(0, out="Eclipse SUMO sumo Version 1.27.1\n")
    assert sp.sumo_version(runner=runner) == "1.27.1"
    # 失敗コードは None(捏造しない)
    assert sp.sumo_version(runner=lambda *a, **k: _FakeProc(1)) is None


def test_sumo_version_none_when_absent(monkeypatch):
    monkeypatch.setattr(sp, "find_bin", lambda name: None)
    called = {"n": 0}
    def runner(*a, **k):
        called["n"] += 1
        return _FakeProc(0, out="x")
    assert sp.sumo_version(runner=runner) is None
    assert called["n"] == 0        # バイナリ不在なら実行しない


def test_ensure_sumo_present():
    logs = []
    res = sp.ensure_sumo(version_fn=lambda: "1.27.1",
                         pip_fn=lambda: (_ for _ in ()).throw(AssertionError("pip 不要")),
                         log=logs.append)
    assert res == {"ok": True, "how": "present", "version": "1.27.1"}


def test_ensure_sumo_guides_when_absent_and_pip_fails():
    logs = []
    res = sp.ensure_sumo(version_fn=lambda: None, pip_fn=lambda: False,
                         log=logs.append)
    assert res["ok"] is False and res["how"] == "guide" and res["version"] is None
    blob = "\n".join(logs)
    assert "sumo-setup.md" in blob        # 導入手順を案内
    assert "winget" in blob and "pip install eclipse-sumo" in blob
    assert "1.27" not in blob or "eclipse-sumo" in blob   # 版を捏造していない


def test_ensure_sumo_recovers_after_pip():
    state = {"installed": False}
    def version_fn():
        return "1.27.1" if state["installed"] else None
    def pip_fn():
        state["installed"] = True
        return True
    res = sp.ensure_sumo(version_fn=version_fn, pip_fn=pip_fn, log=lambda *_: None)
    assert res == {"ok": True, "how": "pip", "version": "1.27.1"}


# ─────────────────────────────────────────── Overpass 手順出力(勝手に DL しない)
def test_osm_fetch_instructions_no_download():
    txt = sp.osm_fetch_instructions((35.65, 139.69, 35.66, 139.71),
                                    Path("runs/x/sumo/road.osm.xml"))
    assert "osmGet.py" in txt and "overpass-api.de" in txt
    assert "--fetch-osm" in txt          # 明示取得の導線を案内
    # bbox が osmGet 形式(W,S,E,N)で載る
    assert "139.69,35.65,139.71,35.66" in txt
