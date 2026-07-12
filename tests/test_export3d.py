"""scripts/export_3d.py の検証(バッチE / 3D 可視化)。

- ear clipping の凹多角形ケース単体テスト
- 小 mock ラン(10人×24step, ラン名 e_check)→ export → scene/tracks の必須キー・件数 assert
- buildings.glb のマジックバイト・チャンク構造検証
"""
from __future__ import annotations

import importlib.util
import json
import struct
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_export3d():
    spec = importlib.util.spec_from_file_location(
        "export_3d", REPO_ROOT / "scripts" / "export_3d.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


E3D = _load_export3d()


# ----------------------------------------------------------------- ear clipping
def _tri_area(a, b, c):
    return abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / 2.0


def test_earclip_convex_square():
    ring = [(0, 0), (2, 0), (2, 2), (0, 2)]
    tris = E3D.triangulate(ring)
    assert len(tris) == 2
    total = sum(_tri_area(ring[i], ring[j], ring[k]) for i, j, k in tris)
    assert total == pytest.approx(4.0)


def test_earclip_concave_L():
    # 凹(reflex 頂点 (2,2))の L 字。面積 12、三角形は n-2=4 枚で過不足なく覆う。
    ring = [(0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)]
    tris = E3D.triangulate(ring)
    assert len(tris) == len(ring) - 2 == 4
    total = sum(_tri_area(ring[i], ring[j], ring[k]) for i, j, k in tris)
    assert total == pytest.approx(12.0)
    # 生成された各三角形は元頂点のみを参照する
    for i, j, k in tris:
        assert {i, j, k} <= set(range(len(ring)))


def test_earclip_closed_ring_and_cw_input():
    # 末尾が先頭と重複(閉環)かつ CW 入力でも正しく処理される
    ring_cw = [(0, 0), (0, 4), (2, 4), (2, 2), (4, 2), (4, 0), (0, 0)]
    tris = E3D.triangulate(ring_cw)
    assert len(tris) == 4  # 6 実頂点 - 2
    total = sum(_tri_area(ring_cw[i], ring_cw[j], ring_cw[k]) for i, j, k in tris)
    assert total == pytest.approx(12.0)


# ----------------------------------------------------------------- glb
def test_build_glb_structure():
    blds = [
        {"footprint": [[0, 0], [10, 0], [10, 10], [0, 10]],
         "height": 7.0, "base": 0.0, "kind": "office"},
        {"footprint": [[0, 0], [4, 0], [4, 2], [2, 2], [2, 4], [0, 4]],  # 凹
         "height": 3.5, "base": -7.0, "kind": "retail"},
    ]
    glb = E3D.build_glb(blds)
    magic, ver, length = struct.unpack("<4sII", glb[:12])
    assert magic == b"glTF"
    assert ver == 2
    assert length == len(glb)
    off = 12
    c0len, c0type = struct.unpack("<I4s", glb[off:off + 8]); off += 8
    assert c0type == b"JSON"
    gltf = json.loads(glb[off:off + c0len]); off += c0len
    c1len, c1type = struct.unpack("<I4s", glb[off:off + 8]); off += 8
    assert c1type == b"BIN\x00"
    assert off + c1len == len(glb)
    # 参照整合性
    assert gltf["asset"]["version"] == "2.0"
    assert gltf["buffers"][0]["byteLength"] == c1len
    acc = {a["type"]: a for a in gltf["accessors"]}
    assert "VEC3" in acc and "SCALAR" in acc
    # POSITION に min/max がある(glTF 必須)
    pos_acc = gltf["accessors"][gltf["meshes"][0]["primitives"][0]["attributes"]["POSITION"]]
    assert "min" in pos_acc and "max" in pos_acc


# ----------------------------------------------------------------- mock run
def _synthetic_map() -> dict:
    return {
        "meta": {"version": 3, "name": "mock", "attribution": "test",
                 "origin_latlon": [35.6595, 139.70062],
                 "bbox": [35.656, 139.695, 35.6625, 139.706], "crs": "local-m"},
        "nodes": [{"id": "n1", "name": "テスト交差点", "x": 0, "y": 0}],
        "edges": [
            {"u": "n1", "v": "n2", "klass": "primary", "layer": 0,
             "geometry": [[0, 0], [50, 0], [50, 50]], "length": 100},
            {"u": "n2", "v": "n3", "klass": "footway", "layer": 0,
             "geometry": [[50, 50], [80, 60]], "length": 32},
        ],
        "buildings": [
            {"id": "bA", "name": "モックタワー", "kind": "office", "levels": 10,
             "below": 2, "cx": 5, "cy": 5,
             "footprint": [[0, 0], [10, 0], [10, 10], [0, 10]]},
            {"id": "bB", "name": "L字ビル", "kind": "retail", "levels": 3,
             "cx": 22, "cy": 22,
             "footprint": [[20, 20], [24, 20], [24, 22], [22, 22], [22, 24], [20, 24]]},
            {"id": "bC", "kind": "house?", "levels": 2, "cx": 40, "cy": 40,
             "footprint": [[38, 38], [42, 38], [42, 42], [38, 42]]},
        ],
        "railways": [
            {"name": "モック地下鉄", "kind": "subway",
             "geometry": [[-30, -30], [0, 0], [30, 30]]},
            {"name": "モックJR", "kind": "rail",
             "geometry": [[-40, 40], [40, 40]]},
        ],
        "pois": [
            {"id": "p1", "name": "モックカフェ", "cat": "cafe", "x": 5, "y": 5,
             "building": "bA", "floor": 1},
            {"id": "p2", "name": "無所属POI", "cat": "office", "x": 60, "y": 60},
        ],
        "car_gateways": [],
    }


def _mock_events():
    """10 agents × 24 step の代表イベント。建物 ID は _synthetic_map と一致。"""
    rows = []

    def add(step, aid, kind, x, y, payload):
        rows.append({"step": step, "sim_min": 7 * 60 + step * 10, "agent_id": aid,
                     "kind": kind, "x": float(x), "y": float(y),
                     "payload": json.dumps(payload, ensure_ascii=False),
                     "rng_stream": "", "llm_call_id": ""})

    # 全 10 agent が step0 で路上
    for a in range(10):
        add(0, a, "arrive", a * 5, 0, {"name": "路上", "node": "n1"})
    # agent0: move_segment → enter_building → floor_move → exit_building
    add(1, 0, "move_segment", 0, 0, {"mode": "walk",
        "pts": [[0, 0], [5, 0], [10, 0], [10, 10]]})
    add(2, 0, "enter_building", 5, 5, {"building": "bA", "floor": 3})
    add(3, 0, "floor_move", 5, 5, {"building": "bA", "floor": 7})
    add(5, 0, "exit_building", 5, 5, {"building": "bA"})
    # agent1: 車移動 + traffic_flow
    add(1, 1, "move_segment", 20, 20, {"mode": "car",
        "pts": [[10, 0], [20, 10], [30, 20]]})
    # agent2: 範囲外 → 復帰
    add(4, 2, "exit_area", -100, -100, {"gateway": "n1", "via": "walk"})
    add(10, 2, "enter_area", -100, -100, {"gateway": "n1", "via": "walk"})
    # agent3: 睡眠 → 起床
    add(18, 3, "sleep_start", 3, 3, {"building": "bC", "until_step": 23})
    add(23, 3, "wake_up", 3, 3, {"slept_steps": 5})
    # agent4: speak
    add(6, 4, "speak", 15, 15, {"text": "テスト発話", "hearers": [], "items": []})
    # traffic_flow(agent_id=-1)
    for s in range(24):
        add(s, -1, "traffic_flow", 0, 0,
            {"n": 3, "segs": [[[0, 0], [10, 5]], [[10, 5], [20, 10]]]})
    return rows


def _write_mock_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "e_check"
    run_dir.mkdir()
    # map
    map_path = tmp_path / "mock_map.json"
    map_path.write_text(json.dumps(_synthetic_map(), ensure_ascii=False), encoding="utf-8")
    # config.yaml
    (run_dir / "config.yaml").write_text(
        f"world:\n  map: {map_path.as_posix()}\n", encoding="utf-8")
    # agents.json
    agents = [{"id": i, "name": f"住民{i}", "age": 20 + i, "gender": "男" if i % 2 else "女",
               "occupation": "会社員", "visitor": (i % 3 == 0),
               "has_bicycle": False, "has_car": (i == 1)} for i in range(10)]
    (run_dir / "agents.json").write_text(json.dumps(agents, ensure_ascii=False),
                                         encoding="utf-8")
    # l1_events.parquet
    rows = _mock_events()
    cols = {k: [r[k] for r in rows] for k in rows[0]}
    schema = pa.schema([
        ("step", pa.int32()), ("sim_min", pa.int32()), ("agent_id", pa.int32()),
        ("kind", pa.string()), ("x", pa.float32()), ("y", pa.float32()),
        ("payload", pa.string()), ("rng_stream", pa.string()), ("llm_call_id", pa.string())])
    pq.write_table(pa.table(cols, schema=schema), run_dir / "l1_events.parquet")
    return run_dir, map_path


def test_export_run_mock(tmp_path):
    run_dir, map_path = _write_mock_run(tmp_path)
    res = E3D.export_run(run_dir, map_path)

    scene = json.loads(res["scene"].read_text(encoding="utf-8"))
    tracks = json.loads(res["tracks"].read_text(encoding="utf-8"))

    # scene 必須キー
    for k in ("meta", "buildings", "roads", "rails", "pois"):
        assert k in scene, k
    assert scene["meta"]["crs"] == "local-m"
    assert scene["meta"]["floor_height"] == 3.5
    assert scene["meta"]["step_minutes"] == 10
    assert len(scene["buildings"]) == 3
    # 建物: 閉環・height/base
    bA = next(b for b in scene["buildings"] if b["id"] == "bA")
    assert bA["footprint"][0] == bA["footprint"][-1]  # 閉環に補正されている
    assert bA["height"] == pytest.approx(10 * 3.5)
    assert bA["base"] == pytest.approx(-2 * 3.5)
    # rails: subway は z=-8
    subway = next(r for r in scene["rails"] if r["kind"] == "subway")
    assert subway["z"] == -8.0
    assert len(scene["roads"]) == 2
    # 名前つき POI のみ
    assert all(p["name"] for p in scene["pois"])

    # tracks 必須キー
    for k in ("meta", "agents", "ids", "positions", "moves", "traffic", "sim_min"):
        assert k in tracks, k
    assert tracks["meta"]["nSteps"] == 24
    assert len(tracks["positions"]) == 24
    assert len(tracks["sim_min"]) == 24
    assert len(tracks["agents"]) == 10
    assert len(tracks["ids"]) == 10
    # 各 step の位置配列は 10 エージェント分
    assert all(len(p) == 10 for p in tracks["positions"])
    # agent0 は step2 で屋内(w >= 1000)、その後 exit で 0 に戻る
    w_step2 = tracks["positions"][2][0][2]
    assert w_step2 >= 1000
    assert tracks["positions"][5][0][2] == 0
    # agent2 は step4 以降 exit_area で w=-1
    assert tracks["positions"][4][2][2] == -1
    # agent3 は sleep 中 w=-2
    assert tracks["positions"][18][3][2] == -2
    # traffic は各 step 存在
    assert all("segs" in t for t in tracks["traffic"])
    # move: agent0 step1 は walk(mode 0)のポリライン
    assert tracks["moves"][1][0][0] == 0
    assert len(tracks["moves"][1][0][1]) >= 2
    # visitor フラグが伝播
    assert tracks["agents"][0]["visitor"] is True

    # glb 検証
    glb = res["glb"].read_bytes()
    magic, ver, length = struct.unpack("<4sII", glb[:12])
    assert magic == b"glTF" and ver == 2 and length == len(glb)
