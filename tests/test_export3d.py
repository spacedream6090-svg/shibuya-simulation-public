"""scripts/export_3d.py の検証(バッチE / 3D 可視化)。

- ear clipping の凹多角形ケース単体テスト
- 小 mock ラン(10人×24step, ラン名 e_check)→ export → scene/tracks の必須キー・件数 assert
- buildings.glb のマジックバイト・チャンク構造検証
"""
from __future__ import annotations

import base64
import importlib.util
import json
import struct
from pathlib import Path

import numpy as np
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


# ================================================================ --rich-tracks
def _write_events_run(tmp_path: Path, name: str, events: list, n_agents: int):
    """任意イベント列で mock ラン一式(map/config/agents/parquet)を書く。"""
    run_dir = tmp_path / name
    run_dir.mkdir()
    map_path = tmp_path / f"{name}_map.json"
    map_path.write_text(json.dumps(_synthetic_map(), ensure_ascii=False), encoding="utf-8")
    (run_dir / "config.yaml").write_text(
        f"world:\n  map: {map_path.as_posix()}\n", encoding="utf-8")
    agents = [{"id": i, "name": f"住民{i}", "age": 20 + i, "gender": "男",
               "occupation": "会社員", "visitor": False,
               "has_bicycle": (i == 5), "has_car": (i in (0, 1))} for i in range(n_agents)]
    (run_dir / "agents.json").write_text(json.dumps(agents, ensure_ascii=False),
                                         encoding="utf-8")
    cols = {k: [r[k] for r in events] for k in events[0]}
    schema = pa.schema([
        ("step", pa.int32()), ("sim_min", pa.int32()), ("agent_id", pa.int32()),
        ("kind", pa.string()), ("x", pa.float32()), ("y", pa.float32()),
        ("payload", pa.string()), ("rng_stream", pa.string()), ("llm_call_id", pa.string())])
    pq.write_table(pa.table(cols, schema=schema), run_dir / "l1_events.parquet")
    return run_dir, map_path


def _rich_events():
    """taxi ride(car / walk 記録)・train/walk exit を含む代表イベント。"""
    rows = []

    def add(step, aid, kind, x, y, payload):
        rows.append({"step": step, "sim_min": 7 * 60 + step * 10, "agent_id": aid,
                     "kind": kind, "x": float(x), "y": float(y),
                     "payload": json.dumps(payload, ensure_ascii=False),
                     "rng_stream": "", "llm_call_id": ""})

    for a in range(6):
        add(0, a, "arrive", a * 5, 0, {"name": "路上"})
    # agent0: タクシー乗車(car 区間)+ ride(taxi) 同一 step → rich で mode 3
    add(2, 0, "move_segment", 30, 40, {"mode": "car", "pts": [[0, 0], [15, 20], [30, 40]]})
    add(2, 0, "ride", 30, 40, {"mode": "taxi", "from": "n1", "to": "n2", "fare": 800.0})
    # agent1: 自家用車(ride 無し)→ rich でも mode 2 のまま
    add(2, 1, "move_segment", 10, 10, {"mode": "car", "pts": [[0, 0], [10, 10]]})
    # agent2: taxi ride だが移動が walk 記録 → mode=car のみ振替なので walk(0) のまま
    add(2, 2, "move_segment", 5, 5, {"mode": "walk", "pts": [[0, 0], [5, 5]]})
    add(2, 2, "ride", 5, 5, {"mode": "taxi", "from": "n1", "to": "n2", "fare": 700.0})
    # agent3: 電車で圏外(via=train)→ rich で w=-3
    add(3, 3, "exit_area", 200, 200, {"gateway": "n1", "via": "train", "homing": False})
    # agent4: 徒歩で圏外(via=walk)→ w=-1(両方)
    add(3, 4, "exit_area", 300, 300, {"gateway": "n1", "via": "walk", "homing": False})
    # agent5: 自転車 → mode 1
    add(1, 5, "move_segment", 8, 0, {"mode": "bicycle", "pts": [[0, 0], [8, 0]]})
    return rows


def test_rich_tracks_default_byte_identical(tmp_path):
    """--rich-tracks OFF: 同一入力で 2 回実行し scene/tracks/glb がバイト同一(既定不変)。"""
    run_dir, map_path = _write_events_run(tmp_path, "e_rich0", _rich_events(), 6)
    res1 = E3D.export_run(run_dir, map_path, rich_tracks=False)
    b1 = {k: res1[k].read_bytes() for k in ("scene", "tracks", "glb")}
    res2 = E3D.export_run(run_dir, map_path, rich_tracks=False)
    b2 = {k: res2[k].read_bytes() for k in ("scene", "tracks", "glb")}
    for k in b1:
        assert b1[k] == b2[k], f"{k} not byte-identical across runs"

    tracks = json.loads(b1["tracks"].decode("utf-8"))
    idx = {a: i for i, a in enumerate(tracks["ids"])}
    # 既定: タクシー car 区間は mode 2 のまま・train 圏外は w=-1・legend 無し
    assert tracks["moves"][2][idx[0]][0] == 2
    assert tracks["positions"][3][idx[3]][2] == -1
    assert "mode_legend" not in tracks["meta"]
    assert "away_train" not in tracks["meta"]


def test_rich_tracks_reassigns_taxi_and_train(tmp_path):
    run_dir, map_path = _write_events_run(tmp_path, "e_rich1", _rich_events(), 6)
    res = E3D.export_run(run_dir, map_path, rich_tracks=True)
    tracks = json.loads(res["tracks"].read_text(encoding="utf-8"))
    idx = {a: i for i, a in enumerate(tracks["ids"])}
    # agent0: ride(taxi)同一 step の car 区間 → mode 3
    assert tracks["moves"][2][idx[0]][0] == 3
    # agent1: ride 無しの自家用車 → mode 2 のまま
    assert tracks["moves"][2][idx[1]][0] == 2
    # agent2: taxi ride だが walk 記録 → walk(0) のまま(car のみ振替)
    assert tracks["moves"][2][idx[2]][0] == 0
    # agent5: bicycle → mode 1
    assert tracks["moves"][1][idx[5]][0] == 1
    # agent3: 電車で圏外 → w=-3
    assert tracks["positions"][3][idx[3]][2] == -3
    # agent4: 徒歩で圏外 → w=-1
    assert tracks["positions"][3][idx[4]][2] == -1
    # meta legend
    assert tracks["meta"]["mode_legend"]["3"] == "タクシー"
    assert tracks["meta"]["away_train"] == -3


# ================================================================ terrain / extras
def _make_terrain(a=0.3, b=-0.2, d=12.0, x0=-5.0, y0=3.0, cell=10.0, nx=8, ny=6):
    """平面 z = a*x + b*y + d の地表グリッド(双一次補間で厳密再現できる)。"""
    xs = x0 + np.arange(nx) * cell
    ys = y0 + np.arange(ny) * cell
    H = np.zeros((ny, nx), dtype=np.float64)
    for iy in range(ny):
        for ix in range(nx):
            H[iy, ix] = a * xs[ix] + b * ys[iy] + d
    return {"x0": x0, "y0": y0, "cell_m": cell, "nx": nx, "ny": ny, "H": H,
            "_plane": (a, b, d)}


def test_terrain_web_quant_roundtrip():
    terr = _make_terrain()
    web = E3D.build_terrain_web(terr)
    assert web["quant"] == 0.1
    assert web["nx"] == terr["nx"] and web["ny"] == terr["ny"]
    assert web["x0"] == terr["x0"] and web["cell_m"] == terr["cell_m"]
    raw = base64.b64decode(web["heights_b64"])
    q = np.frombuffer(raw, dtype="<i2")
    assert q.size == terr["nx"] * terr["ny"]
    # row-major (ny, nx) で復元し量子化ラウンドトリップ(誤差 < 0.05m)
    H_back = q.reshape(terr["ny"], terr["nx"]).astype(np.float64) * 0.1
    assert np.max(np.abs(H_back - terr["H"])) < 0.05


def test_terrain_gz_bilinear_exact_on_plane():
    terr = _make_terrain()
    a, b, d = terr["_plane"]
    # 平面 z=ax+by+d は双一次補間で厳密再現 → 量子化(0.1m 丸め)後も真値と半量子内で一致。
    for (x, y) in [(0.0, 5.0), (12.3, 8.7), (23.4, 33.3), (-2.0, 40.0), (50.0, 3.0),
                   (13.0, 8.0), (33.0, 41.0)]:
        gz = E3D._sample_terrain_gz(terr, x, y)
        assert abs(gz - (a * x + b * y + d)) <= 0.05 + 1e-9
    # 1桁で割り切れる点は丸め後も厳密一致(補間の正しさを直接確認)
    for (x, y, expect) in [(10.0, 5.0, 14.0), (13.0, 8.0, 14.3), (33.0, 41.0, 13.7)]:
        assert E3D._sample_terrain_gz(terr, x, y) == pytest.approx(expect, abs=1e-9)


def test_extras_web_roundtrip():
    V_u = np.array([[0, 0, -5], [10, 0, -5], [10, 10, -5], [0, 10, -5]], dtype=np.float64)
    F_u = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    V_b = np.array([[1.23, 2.34, 3.45], [4.56, 5.67, 6.78], [7.89, 8.9, 9.01]], dtype=np.float64)
    F_b = np.array([[0, 1, 2]], dtype=np.int32)
    web = E3D.build_extras_web({"ubld": (V_u, F_u), "brid": (V_b, F_b)})
    assert set(web) == {"ubld", "brid"}
    assert web["ubld"]["n_vertices"] == 4 and web["ubld"]["n_triangles"] == 2
    assert web["brid"]["n_vertices"] == 3 and web["brid"]["n_triangles"] == 1
    # 位置量子化ラウンドトリップ(int16 × 0.05m)
    q = np.frombuffer(base64.b64decode(web["brid"]["positions_b64"]), dtype="<i2")
    V_back = q.reshape(-1, 3).astype(np.float64) * 0.05
    assert np.max(np.abs(V_back - V_b)) < 0.05
    # index は <u4
    iu = np.frombuffer(base64.b64decode(web["ubld"]["indices_b64"]), dtype="<u4")
    assert iu.tolist() == [0, 1, 2, 0, 2, 3]


def _write_min_plateau(pdir: Path):
    """最小 plateau 一式(1 建物 bA を実測メッシュ化)を書く。"""
    pdir.mkdir(parents=True, exist_ok=True)
    V = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 35]], dtype=np.float32)
    F = np.array([[0, 1, 2]], dtype=np.int32)
    off = np.array([0, 1], dtype=np.int32)
    np.savez(pdir / "plateau_mesh.npz", V=V, F=F, building_offsets=off)
    (pdir / "plateau_index.json").write_text(json.dumps({
        "ground0_source": "test",
        "buildings": [{"gml_id": "g0", "height": 35.0, "base": 0.0,
                       "footprint": [[0, 0], [10, 0], [10, 10]], "n_tris": 1, "lod": 2}],
    }), encoding="utf-8")
    (pdir / "plateau_match.json").write_text(
        json.dumps({"matches": {"bA": {"gml_id": "g0"}}}), encoding="utf-8")


def test_export_plateau_terrain_extras_integration(tmp_path):
    """--plateau + terrain.npz/json + extras.npz の一式書き出し(gz/terrain_web/extras)。"""
    run_dir, map_path = _write_events_run(tmp_path, "e_pl", _rich_events(), 6)
    pdir = tmp_path / "plateau"
    _write_min_plateau(pdir)
    # terrain(平面)を保存
    terr = _make_terrain(nx=40, ny=40, x0=-50.0, y0=-50.0, cell=5.0)
    a, b, d = terr["_plane"]
    (pdir / "terrain.json").write_text(json.dumps({
        "x0": terr["x0"], "y0": terr["y0"], "cell_m": terr["cell_m"],
        "nx": terr["nx"], "ny": terr["ny"]}), encoding="utf-8")
    np.savez(pdir / "terrain.npz", heights=terr["H"].astype(np.float32))
    # extras(地下街 1 面・橋 1 面)
    np.savez(pdir / "extras.npz",
             ubld_V=np.array([[0, 0, -6], [4, 0, -6], [4, 4, -6]], dtype=np.float32),
             ubld_F=np.array([[0, 1, 2]], dtype=np.int32),
             brid_V=np.array([[0, 0, 8], [4, 0, 8], [4, 4, 8]], dtype=np.float32),
             brid_F=np.array([[0, 1, 2]], dtype=np.int32))

    res = E3D.export_run(run_dir, map_path, plateau_dir=pdir, rich_tracks=True)
    assert "terrain_web" in res and "plateau_web" in res

    scene = json.loads(res["scene"].read_text(encoding="utf-8"))
    # 全建物に gz が付与され、**footprint 頂点の最小地表高**と一致(A-6)。
    # 重心1点だと斜面で山側の壁が地面に埋まるため、最小値=足元の最も低い点を基準にする。
    for bl in scene["buildings"]:
        assert "gz" in bl
        want = min(round(a * p[0] + b * p[1] + d, 1) for p in bl["footprint"])
        assert bl["gz"] == pytest.approx(want, abs=1e-9)
        # 斜面なので重心高より必ず低い(等しいのは水平地形のときだけ)
        assert bl["gz"] <= round(a * bl["cx"] + b * bl["cy"] + d, 1) + 1e-9

    tw = json.loads(res["terrain_web"].read_text(encoding="utf-8"))
    assert tw["quant"] == 0.1 and tw["nx"] == 40 and tw["ny"] == 40

    web = json.loads(res["plateau_web"].read_text(encoding="utf-8"))
    assert "extras" in web and set(web["extras"]) == {"ubld", "brid"}
    assert web["extras"]["ubld"]["n_triangles"] == 1


# ================================================================ 屋内 w の是正(B-1/B-2)
def test_encode_indoor_w_clamps_floor_to_levels():
    """floor は 1..min(levels,99) にクランプ。負値・0・levels 超え・桁あふれを是正する
    (w=1000+bIdx*100+floor は floor が 2 桁を超えると別建物を指してしまう)。"""
    bld_idx = {"bA": 0, "bB": 1}
    lv = [10, 200]                      # bB は levels>99(桁あふれ源)
    st = {}
    assert E3D.encode_indoor_w(bld_idx, lv, "bA", 3, st) == 1000 + 0 * 100 + 3
    assert E3D.encode_indoor_w(bld_idx, lv, "bA", 10, st) == 1000 + 0 * 100 + 10
    assert E3D.encode_indoor_w(bld_idx, lv, "bA", 25, st) == 1000 + 0 * 100 + 10   # >levels
    assert E3D.encode_indoor_w(bld_idx, lv, "bA", 0, st) == 1000 + 0 * 100 + 1
    assert E3D.encode_indoor_w(bld_idx, lv, "bA", -3, st) == 1000 + 0 * 100 + 1
    assert E3D.encode_indoor_w(bld_idx, lv, "bB", 150, st) == 1000 + 1 * 100 + 99  # 2桁上限
    assert st["clamped"] == 4
    assert "unknown" not in st
    # クランプ後は必ず「その建物の枠(100 未満)」に収まる = 別建物を指さない
    for f in (-99, 0, 1, 7, 99, 100, 12345):
        w = E3D.encode_indoor_w(bld_idx, lv, "bB", f, {})
        assert 1 <= (w - 1000) % 100 <= 99
        assert (w - 1000) // 100 == 1


def test_encode_indoor_w_unknown_building_falls_back_to_street():
    """未知の建物名は idx0(無関係な建物)ではなく w=0(路上)へ退避し、件数を数える。"""
    st = {}
    assert E3D.encode_indoor_w({"bA": 0}, [5], "存在しないビル", 2, st) == 0
    assert E3D.encode_indoor_w({"bA": 0}, [5], None, 2, st) == 0
    assert st["unknown"] == 2
    assert st["unknown_names"]["存在しないビル"] == 1
    assert "clamped" not in st


def test_reconstruct_tracks_clamps_and_reports(tmp_path, capsys):
    """イベント経路の通し: levels 超え floor はクランプ、未知建物は w=0、報告が出る。"""
    rows = []

    def add(step, aid, kind, x, y, payload):
        rows.append({"step": step, "sim_min": 7 * 60 + step * 10, "agent_id": aid,
                     "kind": kind, "x": float(x), "y": float(y),
                     "payload": json.dumps(payload, ensure_ascii=False),
                     "rng_stream": "", "llm_call_id": ""})

    for a in range(3):
        add(0, a, "arrive", 0, 0, {"name": "路上"})
    add(1, 0, "enter_building", 5, 5, {"building": "bB", "floor": 99})   # levels=3
    add(2, 1, "enter_building", 5, 5, {"building": "無い建物", "floor": 2})
    add(3, 2, "enter_building", 5, 5, {"building": "bA", "floor": -1})   # 負値
    buildings = _synthetic_map()["buildings"]
    agents = [{"id": i, "name": f"a{i}", "occupation": "?", "visitor": False}
              for i in range(3)]
    tracks = E3D.reconstruct_tracks(rows, buildings, agents)
    bidx = {b["id"]: i for i, b in enumerate(buildings)}
    lv = {b["id"]: int(b.get("levels", 2)) for b in buildings}
    w0 = tracks["positions"][1][0][2]
    assert (w0 - 1000) // 100 == bidx["bB"]
    assert (w0 - 1000) % 100 == lv["bB"]            # 99 → levels(=3)
    assert tracks["positions"][2][1][2] == 0        # 未知建物 → 路上
    w2 = tracks["positions"][3][2][2]
    assert (w2 - 1000) // 100 == bidx["bA"] and (w2 - 1000) % 100 == 1   # 負値 → 1F
    out = capsys.readouterr().out
    assert "floor クランプ" in out and "未知建物" in out


def test_scene_depth_and_plateau_floor_height(tmp_path):
    """scene.json: 押出し深さ depth=(levels+below)*FH(地下階建物の屋上沈み対策)と、
    PLATEAU 実測高で上書きした建物の実効階高 floorH=height/levels。"""
    run_dir, map_path = _write_events_run(tmp_path, "e_fh", _rich_events(), 6)
    # depth は plateau 無しでも常に出る(押出しの唯一の権威)
    res0 = E3D.export_run(run_dir, map_path)
    scene0 = json.loads(res0["scene"].read_text(encoding="utf-8"))
    bA0 = next(b for b in scene0["buildings"] if b["id"] == "bA")   # levels=10, below=2
    assert bA0["depth"] == pytest.approx((10 + 2) * 3.5)
    assert bA0["base"] + bA0["depth"] == pytest.approx(bA0["height"])
    assert "floorH" not in bA0                      # 実測高が無ければ出さない

    pdir = tmp_path / "plateau_fh"
    _write_min_plateau(pdir)                        # bA を実測 35.0m に上書き
    res = E3D.export_run(run_dir, map_path, plateau_dir=pdir)
    scene = json.loads(res["scene"].read_text(encoding="utf-8"))
    bA = next(b for b in scene["buildings"] if b["id"] == "bA")
    assert bA["plateau"] == 1
    assert bA["height"] == pytest.approx(35.0)
    assert bA["levels"] == 10                       # levels は据え置き(sim 側の意味を保つ)
    assert bA["floorH"] == pytest.approx(3.5)       # 35.0 / 10
    assert bA["depth"] == pytest.approx(35.0 - bA["base"])
    # 最上階の床(gz + (levels-1)*floorH)は屋根(gz + height)を超えない
    assert (bA["levels"] - 1) * bA["floorH"] <= bA["height"]
    bB = next(b for b in scene["buildings"] if b["id"] == "bB")     # 非照合
    assert "floorH" not in bB and "plateau" not in bB


def test_export_plateau_without_terrain_extras_no_new_keys(tmp_path):
    """--plateau だが terrain/extras 無し: gz 無し・terrain_web 無し・extras キー無し。"""
    run_dir, map_path = _write_events_run(tmp_path, "e_pl2", _rich_events(), 6)
    pdir = tmp_path / "plateau2"
    _write_min_plateau(pdir)
    res = E3D.export_run(run_dir, map_path, plateau_dir=pdir)
    assert "terrain_web" not in res
    scene = json.loads(res["scene"].read_text(encoding="utf-8"))
    assert all("gz" not in bl for bl in scene["buildings"])
    web = json.loads(res["plateau_web"].read_text(encoding="utf-8"))
    assert "extras" not in web
