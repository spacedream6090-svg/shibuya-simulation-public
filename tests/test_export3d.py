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


# ================================================================ 竹-4 持ち越し①(M-4)
# 物理ゾーン(physics.zones)に所有されている間、その個体はグラフ移動をしないので
# move_segment が 1 件も出ず、位置が入口で固まったまま出口で瞬間移動する。
# zone_gate(enter/exit。どちらも位置を持つ)で挟んで直線補間する(w=0=路上扱い)。
def _zone_events():
    """agent0 が step2 でゾーンへ入り step6 で出る(その間 move_segment ゼロ)。
    agent1 は同じ区間を普通に歩く(ゾーンに入らない=補間の対象外)。"""
    rows = []

    def add(step, aid, kind, x, y, payload):
        rows.append({"step": step, "sim_min": 7 * 60 + step * 10, "agent_id": aid,
                     "kind": kind, "x": float(x), "y": float(y),
                     "payload": json.dumps(payload, ensure_ascii=False),
                     "rng_stream": "", "llm_call_id": ""})

    for a in range(2):
        add(0, a, "arrive", 0, 0, {"name": "路上"})
    add(2, 0, "zone_gate", 0, 0, {"zone": "z1", "gate": "g0", "dir": "enter",
                                  "engine": "orca", "wait_s": 0.0})
    add(6, 0, "zone_gate", 80, 40, {"zone": "z1", "gate": "g1", "dir": "exit",
                                    "reason": "gate", "dwell_s": 41.0, "jump_m": 0.2})
    add(1, 1, "move_segment", 10, 5, {"mode": "walk", "pts": [[0, 0], [10, 5]]})
    return rows


def test_zone_gate_gap_is_linearly_interpolated(tmp_path):
    """所有中(step3〜5)の位置が enter→exit の直線で埋まり、実績が meta に出る。"""
    run_dir, map_path = _write_events_run(tmp_path, "e_zone", _zone_events(), 2)
    res = E3D.export_run(run_dir, map_path)
    tracks = json.loads(res["tracks"].read_text(encoding="utf-8"))
    idx = {a: i for i, a in enumerate(tracks["ids"])}
    i0 = idx[0]
    # 入場 step は入口位置、退場 step は出口位置
    assert tracks["positions"][2][i0] == [0.0, 0.0, 0]
    assert tracks["positions"][6][i0] == [80.0, 40.0, 0]
    # 途中は 1/4, 2/4, 3/4 の直線内分(w=0=路上)
    assert tracks["positions"][3][i0] == [20.0, 10.0, 0]
    assert tracks["positions"][4][i0] == [40.0, 20.0, 0]
    assert tracks["positions"][5][i0] == [60.0, 30.0, 0]
    # 補間の実績(黙って埋めない)
    zi = tracks["meta"]["zone_interp"]
    assert zi == {"method": "linear", "segments": 1, "frames": 3, "unclosed": 0}
    # ゾーンに入らなかった個体は従来どおり(補間は他人に波及しない)
    assert tracks["positions"][3][idx[1]] == [10.0, 5.0, 0]


def test_zone_gate_unclosed_segment_is_reported_not_guessed(tmp_path):
    """ラン終端で所有中だった区間は**埋めずに件数だけ**出す(出口位置を捏造しない)。"""
    rows = [r for r in _zone_events() if not (r["kind"] == "zone_gate"
                                              and "exit" in r["payload"])]
    rows.append({"step": 6, "sim_min": 7 * 60 + 60, "agent_id": 1,   # ラン長を保つ
                 "kind": "arrive", "x": 10.0, "y": 5.0,
                 "payload": json.dumps({"name": "路上"}, ensure_ascii=False),
                 "rng_stream": "", "llm_call_id": ""})
    run_dir, map_path = _write_events_run(tmp_path, "e_zone_open", rows, 2)
    tracks = json.loads(E3D.export_run(run_dir, map_path)["tracks"]
                        .read_text(encoding="utf-8"))
    i0 = {a: i for i, a in enumerate(tracks["ids"])}[0]
    zi = tracks["meta"]["zone_interp"]
    assert zi["segments"] == 0 and zi["frames"] == 0 and zi["unclosed"] == 1
    for step in (3, 4, 5):                       # 入口で固まったまま(正直な欠測)
        assert tracks["positions"][step][i0] == [0.0, 0.0, 0]


def test_runs_without_zones_are_byte_identical(tmp_path):
    """zone_gate が 1 件も無いラン(= 物理 OFF の既存ラン)は出力が従来と同一。

    zone_interp キーが生えないこと + 2 回実行のバイト一致で「追加専用」を固定する。
    """
    run_dir, map_path = _write_events_run(tmp_path, "e_nozone", _rich_events(), 6)
    b1 = E3D.export_run(run_dir, map_path)["tracks"].read_bytes()
    b2 = E3D.export_run(run_dir, map_path)["tracks"].read_bytes()
    assert b1 == b2
    assert "zone_interp" not in json.loads(b1.decode("utf-8"))["meta"]


def test_zone_gate_interpolation_respects_step_stride(tmp_path):
    """step_stride で間引いても、出力されるフレームだけが正しく埋まる。"""
    run_dir, map_path = _write_events_run(tmp_path, "e_zone_lod", _zone_events(), 2)
    import pyarrow.parquet as _pq
    events = _pq.read_table(run_dir / "l1_events.parquet").to_pylist()
    city = json.loads(map_path.read_text(encoding="utf-8"))
    tracks = E3D.reconstruct_tracks(events, city.get("buildings", []), [],
                                    step_stride=2)
    i0 = {a: i for i, a in enumerate(tracks["ids"])}[0]
    # 出力フレームは step 0,2,4,6(frame = step//2)。埋まるのは step4 の 1 点だけ
    assert tracks["positions"][1][i0] == [0.0, 0.0, 0]      # step2 = 入場
    assert tracks["positions"][2][i0] == [40.0, 20.0, 0]    # step4 = 中点(補間)
    assert tracks["positions"][3][i0] == [80.0, 40.0, 0]    # step6 = 退場
    assert tracks["meta"]["zone_interp"]["frames"] == 1


# ==================================================== 小粒E: tracks 再構成のストリーミング化
# 検収条件 = **出力バイト同一**。リファクタ前(2026-08-06 時点)の reconstruct_tracks 本体を
# ここへ**逐語で**並置し、新しい 3 経路
#   (a) reconstruct_tracks(events)            … list 経路(既存 API)
#   (b) reconstruct_tracks(load_track_events) … --low-mem 第1段(間引き済み list)
#   (c) reconstruct_tracks_streaming(path)    … --low-mem 第2段(イベント列を持たない)
# の JSON バイト列がすべて旧実装と一致することを固定する。
# ★このコピーはテスト内でのみ参照する(本体からは import されない)。
from collections import defaultdict          # noqa: E402  (旧実装の逐語コピーが使う)

encode_indoor_w = E3D.encode_indoor_w
DEFAULT_START_MIN = E3D.DEFAULT_START_MIN
STEP_MINUTES = E3D.STEP_MINUTES
FLOOR_HEIGHT = E3D.FLOOR_HEIGHT


def _legacy_reconstruct_tracks(events: list, buildings: list, agents_meta: list,
                               sample_agents: int | None = None,
                               step_stride: int = 1, rich_tracks: bool = False,
                               n_steps_override: int | None = None,
                               step_min_override: dict | None = None,
                               agent_ids_override: list | None = None) -> dict:
    """リファクタ前の scripts/export_3d.py::reconstruct_tracks 本体(逐語)。"""
    n_steps = (max((e["step"] for e in events), default=-1) + 1
               if n_steps_override is None else int(n_steps_override))
    bld_idx = {b["id"]: i for i, b in enumerate(buildings)}
    bld_levels = [max(1, int(b.get("levels", 2) or 2)) for b in buildings]
    w_stats: dict = {}                 # B-1: floor クランプ / 未知建物の件数(黙って落とさない)

    agent_ids = (sorted({e["agent_id"] for e in events if e["agent_id"] >= 0})
                 if agent_ids_override is None else list(agent_ids_override))
    if agents_meta:
        idx = {a["id"]: i for i, a in enumerate(agents_meta)}
        agent_ids = [a["id"] for a in agents_meta]
    else:
        idx = {aid: i for i, aid in enumerate(agent_ids)}
        agents_meta = [{"id": a, "name": f"agent{a}", "occupation": "?",
                        "visitor": False} for a in agent_ids]
    # LOD: 先頭 N エージェントに間引く(idx/agent_ids/agents_meta を一貫して縮小)
    if sample_agents and sample_agents > 0 and sample_agents < len(agents_meta):
        agents_meta = agents_meta[:sample_agents]
        agent_ids = [a["id"] for a in agents_meta]
        idx = {a["id"]: i for i, a in enumerate(agents_meta)}
    stride = max(1, int(step_stride or 1))

    by_step: dict[int, list[dict]] = defaultdict(list)
    step_min: dict[int, int] = ({} if step_min_override is None
                                else {int(k): int(v) for k, v in step_min_override.items()})
    for e in events:
        by_step[e["step"]].append(e)
        if step_min_override is not None:
            continue
        sm = e.get("sim_min")
        if sm is not None:
            step_min.setdefault(e["step"], int(sm))

    mode_code = {"walk": 0, "bicycle": 1, "car": 2}
    # --rich-tracks 用: taxi 乗車(ride イベント, payload mode=="taxi")の (agent_id, step) 集合。
    # 対応付け: ride と同一 agent×step の move_segment(mode=car)がタクシー乗車区間。
    # 実データ runs/demo_event_200a3d では ride 79件すべてが同一 step の move_segment 終点と
    # (x,y)一致し、うち mode=car は 33件(残り46件は sim が walk で記録=car ではないため非対象)。
    # 既定 OFF では空集合=振替なし=現行と完全同一。
    taxi_steps: set = set()
    if rich_tracks:
        for e in events:
            if e["kind"] == "ride":
                pr = json.loads(e["payload"]) if e.get("payload") else {}
                if pr.get("mode") == "taxi":
                    taxi_steps.add((e["agent_id"], e["step"]))
    positions, moves, traffic = [], [], []
    cur = [[0.0, 0.0, 0] for _ in agent_ids]
    # 竹-4 持ち越し①: ゾーン所有中(zone_gate enter → exit)の位置の穴。
    #   zone_open[i] = (enter step, (x, y)) / zone_fills = 埋める区間
    zone_open: dict[int, tuple[int, tuple[float, float]]] = {}
    zone_fills: list[tuple[int, int, tuple[float, float], int, tuple[float, float]]] = []
    for step in range(n_steps):
        mv = [None] * len(agent_ids)
        tr_step = {"n": 0, "segs": []}
        for e in by_step.get(step, []):
            p = json.loads(e["payload"]) if e.get("payload") else {}
            kind = e["kind"]
            if kind == "traffic_flow":
                segs = [[[round(float(a), 1), round(float(b), 1)] for a, b in seg]
                        for seg in p.get("segs", [])]
                tr_step = {"n": p.get("n", 0), "segs": segs}
                continue
            if e["agent_id"] not in idx:
                continue
            i = idx[e["agent_id"]]
            if kind in ("move_segment", "arrive", "speak", "reflect"):
                cur[i][0], cur[i][1] = round(float(e["x"]), 1), round(float(e["y"]), 1)
            if kind == "move_segment" and p.get("pts"):
                pts = [[round(float(a), 1), round(float(b), 1)] for a, b in p["pts"]]
                code = mode_code.get(p.get("mode", "walk"), 0)
                # rich_tracks: 同一 agent×step に taxi ride がある car 区間(2)→タクシー(3)。
                # 既定は taxi_steps 空・rich_tracks False なので code 不変=現行と同一。
                if rich_tracks and code == 2 and (e["agent_id"], step) in taxi_steps:
                    code = 3
                mv[i] = [code, pts]
            elif kind == "enter_building":
                cur[i] = [round(float(e["x"]), 1), round(float(e["y"]), 1),
                          encode_indoor_w(bld_idx, bld_levels,
                                          p.get("building"), p.get("floor", 1), w_stats)]
            elif kind == "floor_move":
                cur[i][2] = encode_indoor_w(bld_idx, bld_levels,
                                            p.get("building"), p.get("floor", 1), w_stats)
            elif kind == "exit_building":
                cur[i] = [round(float(e["x"]), 1), round(float(e["y"]), 1), 0]
            elif kind == "exit_area":
                # rich_tracks: 退出手段が電車(payload via=="train")なら -3(電車で圏外)。
                # exit_area payload は via("train"/"walk")を権威情報として持つ(city nodes に
                # 駅種別はほぼ無く=3499中1件、近傍判定より確実)。既定 OFF=従来どおり -1。
                cur[i][2] = -3 if (rich_tracks and p.get("via") == "train") else -1
            elif kind == "enter_area":
                cur[i] = [round(float(e["x"]), 1), round(float(e["y"]), 1), 0]
            elif kind == "sleep_start":
                cur[i][2] = -2
            elif kind == "wake_up":
                cur[i][2] = 0 if cur[i][2] == -2 else cur[i][2]
            elif kind == "zone_gate":
                # 物理ゾーンの流入/流出。所有中は move_segment が出ないので、この 2 点で
                # 区間を挟んで**直線補間**する(後段の post-pass。w=0=路上扱い)。
                # 順序: physics.phase() は _phase_move の**直前**なので、同 step の
                # move_segment は zone_gate より後に来る = 退場した step の最終位置は
                # move_segment 側が正しく上書きする。
                xy = (round(float(e["x"]), 1), round(float(e["y"]), 1))
                cur[i] = [xy[0], xy[1], 0]
                if p.get("dir") == "enter":
                    zone_open[i] = (step, xy)
                else:                                  # "exit"(reason は問わない)
                    opened = zone_open.pop(i, None)
                    if opened is not None and step - opened[0] >= 2:
                        zone_fills.append((i, opened[0], opened[1], step, xy))
        if step % stride == 0:                     # LOD: K step ごとに1フレームだけ出力
            positions.append([list(p) for p in cur])
            moves.append(mv)
            traffic.append(tr_step)

    # ---- 竹-4 持ち越し①: ゾーン所有中の位置を enter→exit の直線で埋める(post-pass)----
    # なぜ post-pass か: 埋める先(exit の位置)は未来の行なので、前向き 1 パスでは書けない。
    # ★正直な限界: これは**実軌跡ではなく直線近似**である。物理(SFM/ORCA)は dt_sub=0.05s で
    #   局所回避しながら曲がって歩くが、その軌跡は L1 に一切残っていない(記録すると 1 個体
    #   1 step あたり最大 12000 点)。実軌跡の記録は将来課題。ここで埋めるのは
    #   「ゾーンに入った人が入口で固まって見え、出口で瞬間移動する」表示上の欠落だけである。
    n_interp_frames = 0
    for i, s0, (x0, y0), s1, (x1, y1) in zone_fills:
        span = s1 - s0
        for s in range(s0 + 1, s1):
            if s % stride:
                continue                       # 出力されないフレームは埋めない
            f = s // stride
            if f >= len(positions):
                continue
            t = (s - s0) / span
            positions[f][i] = [round(x0 + (x1 - x0) * t, 1),
                               round(y0 + (y1 - y0) * t, 1), 0]
            n_interp_frames += 1

    start_min = step_min.get(0, DEFAULT_START_MIN)
    emitted = list(range(0, n_steps, stride))
    sim_min = [step_min.get(s, start_min + s * STEP_MINUTES) for s in emitted]

    agents_slim = [{
        "id": a["id"], "name": a.get("name", f"agent{a['id']}"),
        "visitor": bool(a.get("visitor", False)),
        "occupation": a.get("occupation", "?"),
        "age": a.get("age", 0), "gender": a.get("gender", "?"),
        "has_car": bool(a.get("has_car", False)),
        "has_bicycle": bool(a.get("has_bicycle", False)),
    } for a in agents_meta]

    meta = {"nSteps": len(positions), "step_minutes": STEP_MINUTES,
            "start_min": start_min, "floor_height": FLOOR_HEIGHT}
    if stride > 1:                                  # 追加専用: 全量時は出さない=現行と同一
        meta["step_stride"] = stride
    if rich_tracks:                                 # 追加専用: 既定 OFF=現行と同一
        meta["mode_legend"] = {"0": "徒歩", "1": "自転車", "2": "車", "3": "タクシー"}
        meta["away_train"] = -3
    # 竹-4 持ち越し①: 補間の実績(追加専用キー。ゾーン 0 件のランでは出さない=既存出力と同一)
    if zone_fills or zone_open:
        meta["zone_interp"] = {"method": "linear", "segments": len(zone_fills),
                               "frames": n_interp_frames,
                               "unclosed": len(zone_open)}
        print(f"  [tracks] ゾーン所有中の位置を直線補間: 区間 {len(zone_fills)} 本 /"
              f" フレーム {n_interp_frames} 点"
              + (f" / 未閉 {len(zone_open)} 件(ラン終端で所有中)" if zone_open else "")
              + " ※実軌跡ではなく enter→exit の直線近似")
    if w_stats.get("clamped") or w_stats.get("unknown"):     # silent cap 禁止
        top = sorted(w_stats.get("unknown_names", {}).items(),
                     key=lambda kv: (-kv[1], str(kv[0])))[:3]
        print(f"  [tracks] 屋内 w の是正: floor クランプ {w_stats.get('clamped', 0)}件"
              f" / 未知建物 {w_stats.get('unknown', 0)}件は路上(w=0)へ退避"
              + (f"  上位={top}" if top else ""))
    return {
        "meta": meta,
        "agents": agents_slim,
        "ids": agent_ids,
        "positions": positions,
        "moves": moves,
        "traffic": traffic,
        "sim_min": sim_min,
    }


def _tracks_blob(tracks) -> str:
    return json.dumps(tracks, ensure_ascii=False, separators=(",", ":"))


def _stream_events():
    """ride(タクシー)+ zone_gate + traffic_flow を含む、全分岐を踏むイベント列。"""
    rows = _mock_events()

    def add(step, aid, kind, x, y, payload):
        rows.append({"step": step, "sim_min": 7 * 60 + step * 10, "agent_id": aid,
                     "kind": kind, "x": float(x), "y": float(y),
                     "payload": json.dumps(payload, ensure_ascii=False),
                     "rng_stream": "", "llm_call_id": ""})

    # agent1: 同一 step の ride(taxi)+ car の move_segment = --rich-tracks の振替対象
    add(7, 1, "ride", 30, 20, {"mode": "taxi"})
    add(7, 1, "move_segment", 30, 20, {"mode": "car", "pts": [[20, 10], [30, 20]]})
    # agent5: 物理ゾーンの通過(zone_gate enter → exit。間の位置は直線補間される)
    add(9, 5, "zone_gate", 0, 0, {"zone": "z1", "gate": "g0", "dir": "enter"})
    add(14, 5, "zone_gate", 80, 40, {"zone": "z1", "gate": "g1", "dir": "exit",
                                     "reason": "gate"})
    # agent6: 電車で圏外(--rich-tracks の w=-3)
    add(11, 6, "exit_area", -100, -100, {"gateway": "n1", "via": "train"})
    # 対象外 kind(tracks に一切影響しないことの担保)
    add(12, 7, "hear", 1, 1, {"speaker": 3})
    add(13, 8, "spend", 1, 1, {"amount": 100})
    rows.sort(key=lambda r: r["step"])          # L1 と同じ step 非減少
    return rows


def _write_stream_run(tmp_path, rows, name="e_stream", row_group_size=None):
    run_dir = tmp_path / name
    run_dir.mkdir()
    map_path = tmp_path / f"{name}_map.json"
    map_path.write_text(json.dumps(_synthetic_map(), ensure_ascii=False), encoding="utf-8")
    (run_dir / "config.yaml").write_text(
        f"world:\n  map: {map_path.as_posix()}\n", encoding="utf-8")
    agents = [{"id": i, "name": f"住民{i}", "age": 20 + i, "gender": "男" if i % 2 else "女",
               "occupation": "会社員", "visitor": (i % 3 == 0),
               "has_bicycle": False, "has_car": (i == 1)} for i in range(10)]
    (run_dir / "agents.json").write_text(json.dumps(agents, ensure_ascii=False),
                                         encoding="utf-8")
    cols = {k: [r[k] for r in rows] for k in rows[0]}
    schema = pa.schema([
        ("step", pa.int32()), ("sim_min", pa.int32()), ("agent_id", pa.int32()),
        ("kind", pa.string()), ("x", pa.float32()), ("y", pa.float32()),
        ("payload", pa.string()), ("rng_stream", pa.string()), ("llm_call_id", pa.string())])
    kw = {"row_group_size": row_group_size} if row_group_size else {}
    pq.write_table(pa.table(cols, schema=schema), run_dir / "l1_events.parquet", **kw)
    return run_dir, map_path


@pytest.mark.parametrize("kw", [
    {},
    {"rich_tracks": True},
    {"step_stride": 3},
    {"sample_agents": 4},
    {"sample_agents": 4, "step_stride": 2, "rich_tracks": True},
])
def test_streaming_reconstruct_is_byte_identical_to_legacy(tmp_path, kw):
    """旧実装 vs (a) list 経路 / (b) --low-mem 第1段 / (c) streaming がすべてバイト同一。"""
    rows = _stream_events()
    # row group を 7 行刻みに割る = ストリーミングが必ず複数 chunk をまたぐ
    run_dir, map_path = _write_stream_run(tmp_path, rows, row_group_size=7)
    city = json.loads(map_path.read_text(encoding="utf-8"))
    buildings = city["buildings"]
    agents_meta = json.loads((run_dir / "agents.json").read_text(encoding="utf-8"))
    l1 = run_dir / "l1_events.parquet"

    all_events = pq.read_table(l1).to_pylist()
    legacy = _tracks_blob(_legacy_reconstruct_tracks(all_events, buildings, agents_meta, **kw))
    a = _tracks_blob(E3D.reconstruct_tracks(all_events, buildings, agents_meta, **kw))
    thin, ov = E3D.load_track_events(l1)
    b = _tracks_blob(E3D.reconstruct_tracks(thin, buildings, agents_meta, **ov, **kw))
    c = _tracks_blob(E3D.reconstruct_tracks_streaming(l1, buildings, agents_meta, **kw))
    assert a == legacy, "list 経路が旧実装と食い違う"
    assert b == legacy, "--low-mem 第1段(間引き list)が旧実装と食い違う"
    assert c == legacy, "streaming 経路が旧実装と食い違う"


def test_streaming_never_materializes_all_events(tmp_path):
    """streaming 経路は「全イベントの list」を 1 度も作らない(chunk 単位で回る)。"""
    rows = _stream_events()
    run_dir, _ = _write_stream_run(tmp_path, rows, row_group_size=5)
    l1 = run_dir / "l1_events.parquet"
    chunks = list(E3D.iter_track_events(l1))
    assert len(chunks) >= 2, "row group が 1 つしかなく chunk 分割を検証できていない"
    # chunk を連結すると load_track_events の list と**同じ順・同じ内容**
    flat = [e for ch in chunks for e in ch]
    thin, _ = E3D.load_track_events(l1)
    assert flat == thin
    # step 単位のバケツは step 昇順で、各バケツは 1 step だけを含む
    groups = list(E3D._group_by_step(iter(chunks)))
    assert [g[0] for g in groups] == sorted({e["step"] for e in thin})
    for step, evs in groups:
        assert all(e["step"] == step for e in evs)


def test_scan_track_meta_matches_load_track_events(tmp_path):
    """pass1(列だけ走査)の override 3 点は全読み版と完全一致し、step 昇順を検出する。"""
    rows = _stream_events()
    run_dir, _ = _write_stream_run(tmp_path, rows, row_group_size=6)
    l1 = run_dir / "l1_events.parquet"
    _thin, ov = E3D.load_track_events(l1)
    meta = E3D.scan_track_meta(l1)
    assert meta.pop("step_sorted") is True
    assert meta == ov


def test_streaming_falls_back_when_l1_is_not_step_sorted(tmp_path, capsys):
    """step 非減少でない L1 では退避経路に落ち、結果は全保持経路と同一のまま。"""
    rows = _stream_events()
    rows = rows[12:] + rows[:12]                 # わざと step 順を壊す
    run_dir, map_path = _write_stream_run(tmp_path, rows, name="e_unsorted", row_group_size=6)
    city = json.loads(map_path.read_text(encoding="utf-8"))
    buildings = city["buildings"]
    agents_meta = json.loads((run_dir / "agents.json").read_text(encoding="utf-8"))
    l1 = run_dir / "l1_events.parquet"
    assert E3D.scan_track_meta(l1)["step_sorted"] is False
    thin, ov = E3D.load_track_events(l1)
    want = _tracks_blob(E3D.reconstruct_tracks(thin, buildings, agents_meta, **ov))
    got = _tracks_blob(E3D.reconstruct_tracks_streaming(l1, buildings, agents_meta))
    assert got == want
    assert "退避" in capsys.readouterr().out


def test_export_run_low_mem_outputs_are_byte_identical(tmp_path):
    """export_run の --low-mem あり/なしで生成物が 1 バイトも変わらない。"""
    rows = _stream_events()
    a_dir, map_path = _write_stream_run(tmp_path, rows, name="e_full", row_group_size=7)
    b_dir, _ = _write_stream_run(tmp_path, rows, name="e_low", row_group_size=7)
    ra = E3D.export_run(a_dir, map_path)
    rb = E3D.export_run(b_dir, map_path, low_mem=True)
    for key in ("scene", "tracks", "glb"):
        assert ra[key].read_bytes() == rb[key].read_bytes(), key


def test_dump_json_stream_is_byte_identical_to_dumps(tmp_path):
    """逐次書き出しは json.dumps + write_text と 1 バイトも変わらない。"""
    obj = {"meta": {"nSteps": 3, "start_min": 420, "floor_height": 3.5},
           "日本語キー": ["渋谷", "スクランブル", None, True, False],
           "floats": [0.1, -0.0, 1e-7, 1234567.891, 1 / 3],
           "ints": [0, -1, 2 ** 62],
           "nested": [[[1, 2], [3, 4]], {"a": {"b": {"c": []}}}],
           "empty": [{}, [], "", 0],
           # 非 str キー(dumps はキーを文字列化する)= 手で開かず dumps へ落ちる経路
           "intkeys": {1: "a", 2: "b"},
           "escapes": "tab\t nl\n quote\" backslash\\ ctrl\x01"}
    for ensure_ascii in (False, True):
        want = json.dumps(obj, ensure_ascii=ensure_ascii, separators=(",", ":"))
        p = tmp_path / f"a_{int(ensure_ascii)}.json"
        q = tmp_path / f"b_{int(ensure_ascii)}.json"
        p.write_text(want, encoding="utf-8")
        E3D.dump_json_stream(q, obj, ensure_ascii=ensure_ascii, separators=(",", ":"))
        assert q.read_bytes() == p.read_bytes()
        # チャンク境界を跨いでも同じ(1 文字ずつ書き出しても連結は同じ)
        r = tmp_path / f"c_{int(ensure_ascii)}.json"
        E3D.dump_json_stream(r, obj, ensure_ascii=ensure_ascii,
                             separators=(",", ":"), chunk_chars=1)
        assert r.read_bytes() == p.read_bytes()

