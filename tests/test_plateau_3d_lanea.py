"""高精細 3D レーンA(データパイプライン)の検証。

対象:
  - `scripts/det_npz.py`               決定論的 npz/JSON 書き出し
  - `tools/tiles3d_extract.py`         A-1 松: 3D Tiles(b3dm/Draco/WebP)→ 量子化タイル
  - `scripts/plateau_tran_extract.py`  A-2 竹-1: tran LOD3 → 歩行可能面 + 被覆マップ
  - `scripts/plateau_ubld_extract.py`  A-3 竹-2: ubld LOD4.1 → 壁線分 + 層 + ゲート

検証の柱:
  1. **座標変換が `scripts/plateau_extract.py` と厳密に同一**(定数を写した先が食い違わない)
  2. 量子化の往復誤差が量子化幅の半分以下
  3. 壁線分 dedup の不変量(重複は潰れる・別物は残る・入力順に依存しない)
  4. 被覆マップの自己整合(頂点セルは必ず被覆・膨張は単調)
  5. **決定論**(同じ入力で 2 回書けばバイト同一)
  6. 成果物が存在する場合の整合性(存在しなければ skip)
"""
from __future__ import annotations

import importlib.util
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


DET = _load("det_npz_t", "scripts/det_npz.py")
PEX = _load("plateau_extract_t", "scripts/plateau_extract.py")
TRAN = _load("plateau_tran_extract_t", "scripts/plateau_tran_extract.py")
UBLD = _load("plateau_ubld_extract_t", "scripts/plateau_ubld_extract.py")
T3D = _load("tiles3d_extract_t", "tools/tiles3d_extract.py")

DATA = REPO_ROOT / "data" / "plateau"
SAMPLE_LATLON = [
    (35.65950, 139.70062),      # 原点
    (35.66645113688206, 139.7100199766684),
    (35.65260195270707, 139.68931725223695),
    (35.6600, 139.6950),
    (35.6630, 139.7080),
]


# ===================================================== 1. 座標変換の一致
@pytest.mark.parametrize("mod", [TRAN, UBLD, T3D],
                         ids=["tran", "ubld", "tiles3d"])
@pytest.mark.parametrize("lat,lon", SAMPLE_LATLON)
def test_latlon_to_local_matches_plateau_extract(mod, lat, lon):
    """新規 3 モジュールの局所座標系が plateau_extract と **ビット一致** すること。
    定数を写す方式なので、写し間違いをここで殺す。"""
    lat0, lon0 = PEX.DEFAULT_ORIGIN
    ref = PEX.latlon_to_local(lat, lon, lat0, lon0)
    got = mod.latlon_to_local(lat, lon, lat0, lon0)
    assert float(got[0]) == ref[0]
    assert float(got[1]) == ref[1]


@pytest.mark.parametrize("mod", [TRAN, UBLD, T3D],
                         ids=["tran", "ubld", "tiles3d"])
def test_local_constants_match(mod):
    assert mod.M_PER_DEG_LAT == PEX.M_PER_DEG_LAT
    assert mod.M_PER_DEG_LON_EQ == PEX.M_PER_DEG_LON_EQ
    assert tuple(mod.DEFAULT_ORIGIN) == tuple(PEX.DEFAULT_ORIGIN)


def test_tran_local_to_latlon_roundtrip():
    for lat, lon in SAMPLE_LATLON:
        x, y = TRAN.latlon_to_local(lat, lon)
        lat2, lon2 = TRAN.local_to_latlon(x, y)
        assert abs(lat2 - lat) < 1e-12
        assert abs(lon2 - lon) < 1e-12


def test_tran_meshcode_matches_plateau_extract():
    for lat, lon in SAMPLE_LATLON:
        assert TRAN.meshcode3(lat, lon) == PEX.meshcode3(lat, lon)


# ===================================================== 2. ECEF / タイル座標
def test_ecef_roundtrip_sub_micrometre():
    """geodetic → ECEF → geodetic が µm 精度で戻ること(固定反復回数でも収束済み)。"""
    lats = np.array([35.6, 35.65950, 35.7])
    lons = np.array([139.6, 139.70062, 139.8])
    hs = np.array([-50.0, 36.79, 350.0])
    x, y, z = T3D.geodetic_to_ecef(lats, lons, hs)
    lat2, lon2, h2 = T3D.ecef_to_geodetic(x, y, z)
    assert np.max(np.abs(lat2 - lats)) * T3D.M_PER_DEG_LAT < 1e-6
    assert np.max(np.abs(lon2 - lons)) * T3D.M_PER_DEG_LON_EQ < 1e-6
    assert np.max(np.abs(h2 - hs)) < 1e-6


def test_tile_positions_to_local_matches_plateau_extract_formula():
    """3D Tiles 経路(RTC → ECEF → 測地 → local)が、CityGML 経路の
    latlon_to_local + (標高 - ground0) と一致すること。ここが松案の座標系の要。"""
    lat0, lon0 = T3D.DEFAULT_ORIGIN
    ground0, geoid = 15.18, T3D.GEOID_M
    targets = [(35.6600, 139.6950, 40.0), (35.6640, 139.7090, 12.5),
               (35.65950, 139.70062, 15.18)]
    # 目標点の楕円体高 = 標高 + geoid
    lat = np.array([t[0] for t in targets])
    lon = np.array([t[1] for t in targets])
    elev = np.array([t[2] for t in targets])
    ex, ey, ez = T3D.geodetic_to_ecef(lat, lon, elev + geoid)
    ecef = np.column_stack([ex, ey, ez])
    rtc = ecef.mean(axis=0)
    rel = ecef - rtc
    # ECEF → glTF(Y-up): (X, Y, Z) → (X, Z, -Y)  ( gltf_ypup_to_ecef の逆 )
    P = np.column_stack([rel[:, 0], rel[:, 2], -rel[:, 1]])
    L = T3D.tile_positions_to_local(P, rtc, (lat0, lon0), ground0, geoid)
    for i, (la, lo, el) in enumerate(targets):
        e_ref, n_ref = PEX.latlon_to_local(la, lo, lat0, lon0)
        assert abs(L[i, 0] - e_ref) < 1e-6
        assert abs(L[i, 1] - n_ref) < 1e-6
        assert abs(L[i, 2] - (el - ground0)) < 1e-6


# ===================================================== 3. 量子化の往復
def test_quantize_xyz_roundtrip_and_dtype():
    rng = np.random.default_rng(7)
    V = rng.uniform(-800.0, 800.0, size=(5000, 3))
    Q, org, dt = T3D.quantize_xyz(V)
    assert dt == "int16" and Q.dtype == np.int16
    back = T3D.dequantize_xyz(Q, org)
    assert np.max(np.abs(back - V)) <= T3D.PLATEAU_QUANT / 2 + 1e-12
    # 量子化格子はグローバル 0.05 m 格子(origin_q が整数)
    assert org.dtype.kind == "i"
    assert np.allclose(np.round(back / T3D.PLATEAU_QUANT), back / T3D.PLATEAU_QUANT)


def test_quantize_xyz_falls_back_to_int32_when_wide():
    """int16 相対幅(65535 単位 = 3276 m)を超えるタイルは int32 に落ちる。
    実データでは 148 枚中 3 枚(ルート近傍の粗い LOD タイル)が該当する。"""
    V = np.array([[-2500.0, 0.0, 0.0], [2500.0, 0.0, 0.0]])
    Q, org, dt = T3D.quantize_xyz(V)
    assert dt == "int32" and Q.dtype == np.int32
    back = T3D.dequantize_xyz(Q, org)
    assert np.max(np.abs(back - V)) <= T3D.PLATEAU_QUANT / 2 + 1e-12


def test_quantize_uv_roundtrip():
    rng = np.random.default_rng(11)
    UV = rng.uniform(0.0, 1.0, size=(4000, 2)).astype(np.float32)
    Q, oor = T3D.quantize_uv(UV)
    assert oor == 0 and Q.dtype == np.uint16
    back = Q.astype(np.float64) / 65535.0
    assert np.max(np.abs(back - UV)) <= 0.5 / 65535.0 + 1e-7


def test_quantize_uv_clamps_out_of_range():
    UV = np.array([[-0.5, 1.5], [0.25, 0.75]], dtype=np.float32)
    Q, oor = T3D.quantize_uv(UV)
    assert oor == 1
    assert Q[0, 0] == 0 and Q[0, 1] == 65535


def test_tran_quantize_xy_roundtrip_and_int32_fallback():
    rng = np.random.default_rng(3)
    V = rng.uniform(-700.0, 700.0, size=(3000, 2))
    Q, org, dt = TRAN.quantize_xy(V)
    assert dt == "int16"
    back = (Q.astype(np.int64) + org) * TRAN.QUANT
    assert np.max(np.abs(back - V)) <= TRAN.QUANT / 2 + 1e-12
    # 実 bbox は 32,649/32,767 単位まで使っており余裕が 6 m しかない = int32 経路が要る
    W = rng.uniform(-1100.0, 900.0, size=(3000, 2))
    Q2, org2, dt2 = TRAN.quantize_xy(W)
    assert dt2 == "int32"
    back2 = (Q2.astype(np.int64) + org2) * TRAN.QUANT
    assert np.max(np.abs(back2 - W)) <= TRAN.QUANT / 2 + 1e-12


def test_ubld_quantize_xyz_roundtrip():
    rng = np.random.default_rng(5)
    V = rng.uniform(-250.0, 250.0, size=(2000, 3))
    Q, org = UBLD.quantize_xyz(V)
    back = (Q.astype(np.int64) + org) * UBLD.QUANT
    assert np.max(np.abs(back - V)) <= UBLD.QUANT / 2 + 1e-12


# ===================================================== 4. b3dm / glTF / tileset
def _make_b3dm(feature_table, batch_table, glb):
    ftj = json.dumps(feature_table).encode()
    btj = json.dumps(batch_table).encode()
    ftj += b" " * ((8 - len(ftj) % 8) % 8)
    btj += b" " * ((8 - len(btj) % 8) % 8)
    total = 28 + len(ftj) + len(btj) + len(glb)
    head = struct.pack("<4sIIIIII", b"b3dm", 1, total, len(ftj), 0, len(btj), 0)
    return head + ftj + btj + glb


def _make_glb(js, binc=b""):
    jb = json.dumps(js).encode()
    jb += b" " * ((4 - len(jb) % 4) % 4)
    binc = binc + b"\x00" * ((4 - len(binc) % 4) % 4)
    total = 12 + 8 + len(jb) + (8 + len(binc) if binc else 0)
    out = struct.pack("<4sII", b"glTF", 2, total)
    out += struct.pack("<II", len(jb), 0x4E4F534A) + jb
    if binc:
        out += struct.pack("<II", len(binc), 0x004E4942) + binc
    return out


def test_b3dm_and_glb_split_roundtrip():
    js = {"asset": {"version": "2.0"},
          "extensions": {"CESIUM_RTC": {"center": [1.0, 2.0, 3.0]}}}
    glb = _make_glb(js, b"\x01\x02\x03\x04")
    raw = _make_b3dm({"BATCH_LENGTH": 2}, {"gml_id": ["a", "b"]}, glb)
    ft, _fb, bt, _bb, got = T3D.b3dm_split(raw)
    assert ft["BATCH_LENGTH"] == 2
    assert bt["gml_id"] == ["a", "b"]
    assert got == glb
    js2, bin2 = T3D.glb_split(got)
    assert js2["extensions"]["CESIUM_RTC"]["center"] == [1.0, 2.0, 3.0]
    assert bin2[:4] == b"\x01\x02\x03\x04"


def test_b3dm_split_rejects_non_b3dm():
    with pytest.raises(ValueError):
        T3D.b3dm_split(b"XXXX" + b"\x00" * 40)


def test_tileset_walk_select_and_shadow():
    """REPLACE 階層で bbox と交差するノードだけ選び、同一 gml_id は最深タイルを残す。"""
    def reg(s, w, n, e):
        # 3D Tiles の region は [west, south, east, north, hmin, hmax](ラジアン)
        return [math.radians(w), math.radians(s), math.radians(e),
                math.radians(n), 0.0, 100.0]
    root = {
        "boundingVolume": {"region": reg(35.0, 139.0, 36.0, 140.0)},
        "content": {"uri": "a.b3dm"}, "geometricError": 100,
        "children": [
            {"boundingVolume": {"region": reg(35.5, 139.5, 35.8, 139.8)},
             "content": {"uri": "b.b3dm"}, "geometricError": 10,
             "children": [
                 {"boundingVolume": {"region": reg(35.6, 139.6, 35.7, 139.7)},
                  "content": {"uri": "c.b3dm"}, "geometricError": 0}]},
            {"boundingVolume": {"region": reg(10.0, 10.0, 11.0, 11.0)},
             "content": {"uri": "far.b3dm"}, "geometricError": 0},
        ],
    }
    nodes = T3D.walk_tileset(root)
    assert [n["uri"] for n in nodes] == ["a.b3dm", "b.b3dm", "c.b3dm", "far.b3dm"]
    assert [n["depth"] for n in nodes] == [0, 1, 2, 1]
    sel = T3D.select_tiles(nodes, (35.6, 139.6, 35.7, 139.7))
    assert [nodes[i]["uri"] for i in sel] == ["a.b3dm", "b.b3dm", "c.b3dm"]
    assert T3D.ancestors(nodes, 2) == [1, 0]
    sh = T3D.shadow_map(nodes, sel, {0: ["G", "X"], 1: ["G"], 2: ["G", "Y"]})
    assert sh[2] == set()          # 最深 = 残す
    assert sh[1] == {0}            # 祖先の G は影
    assert sh[0] == {0}            # G は影 / X は唯一なので残る


# ===================================================== 5. 壁線分 dedup の不変量
def test_merge_segments_collapses_two_sided_duplicates():
    """厚み 0.2 m の壁の表裏 2 枚 + 高さ分割 3 枚 → 1 本になる。"""
    segs = []
    for off in (0.0, 0.2):            # 表裏
        for _ in range(3):            # 高さ分割
            segs.append((0.0, off, 10.0, off))
    uniq, mapping = UBLD.merge_segments(segs, tol=0.30, deg=8.0)
    assert len(uniq) == 1
    assert set(mapping) == {0}


def test_merge_segments_keeps_distinct_walls():
    segs = [(0.0, 0.0, 10.0, 0.0),          # 東西
            (0.0, 0.0, 0.0, 10.0),          # 南北(同じ始点・角度違い)
            (0.0, 5.0, 10.0, 5.0)]          # 平行だが 5 m 離れる
    uniq, mapping = UBLD.merge_segments(segs, tol=0.30, deg=8.0)
    assert len(uniq) == 3
    assert sorted(mapping) == [0, 1, 2]


def test_merge_segments_is_order_independent():
    rng = np.random.default_rng(19)
    base = [(float(x), 0.0, float(x) + 4.0, 0.0) for x in range(0, 40, 4)]
    segs = []
    for s in base:
        segs.append(s)
        segs.append((s[2], s[3] + 0.1, s[0], s[1] + 0.1))   # 裏面・逆向き
    a, _ = UBLD.merge_segments(list(segs), 0.30, 8.0)
    perm = list(rng.permutation(len(segs)))
    b, _ = UBLD.merge_segments([segs[i] for i in perm], 0.30, 8.0)
    assert a == b


def test_merge_segments_mapping_is_total():
    segs = [(0.0, 0.0, 3.0, 0.0), (0.05, 0.02, 3.02, 0.01), (7.0, 7.0, 9.0, 9.0)]
    uniq, mapping = UBLD.merge_segments(segs, 0.30, 8.0)
    assert len(mapping) == len(segs)
    assert all(0 <= m < len(uniq) for m in mapping)
    assert set(mapping) == set(range(len(uniq)))


def test_project_segment_and_is_vertical():
    wall = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0],
                     [10.0, 0.0, 3.0], [0.0, 0.0, 3.0]])
    assert UBLD.is_vertical(wall)
    s = UBLD.project_segment(wall)
    assert UBLD.segment_length(s) == pytest.approx(10.0)
    assert set(UBLD.canonical_segment(s)) == {0.0, 10.0}
    floor = np.array([[0.0, 0.0, 1.0], [4.0, 0.0, 1.0],
                      [4.0, 4.0, 1.0], [0.0, 4.0, 1.0]])
    assert not UBLD.is_vertical(floor)


def test_segment_angle_is_undirected():
    a = UBLD.segment_angle((0.0, 0.0, 1.0, 1.0))
    b = UBLD.segment_angle((1.0, 1.0, 0.0, 0.0))
    assert a == pytest.approx(b)
    assert 0.0 <= a < math.pi


# ===================================================== 6. 層分離
def test_layer_peaks_separates_levels_connected_by_a_ramp():
    """階段/スロープで z 連続につながっていても、床面積の山で層は分かれる
    (素朴なギャップ分割はここで 1 層に潰れる)。"""
    zs = []
    ws = []
    for level in (-12.0, -8.0, -4.0):
        zs.extend([level] * 40)
        ws.extend([25.0] * 40)
    for z in np.arange(-12.0, -4.0, 0.25):      # ランプ = 薄く連続
        zs.append(float(z))
        ws.append(0.5)
    peaks, hist = UBLD.layer_peaks(zs, ws, bin_m=0.5, min_sep_m=2.0, min_frac=0.03)
    assert len(peaks) == 3
    assert peaks == sorted(peaks)
    for p, exp in zip(peaks, (-12.0, -8.0, -4.0)):
        assert abs(p - exp) <= 0.5
    assert sum(h[1] for h in hist) == pytest.approx(sum(ws))
    # ギャップ分割は同じ入力で 1 層に潰れる(採用しなかった理由の実証)
    labels, layers = UBLD.cluster_z(zs, gap=1.5)
    assert len(layers) == 1


def test_assign_layer_picks_nearest():
    peaks = [-12.0, -8.0, -4.0]
    assert UBLD.assign_layer(-11.4, peaks) == 0
    assert UBLD.assign_layer(-7.0, peaks) == 1
    assert UBLD.assign_layer(0.0, peaks) == 2
    assert UBLD.assign_layer(0.0, []) == -1


# ===================================================== 7. 被覆マップの自己整合
def test_rasterize_covers_polygon_vertices_and_interior():
    poly = np.array([[1.0, 1.0], [9.0, 1.0], [9.0, 6.0], [1.0, 6.0]])
    m = TRAN.rasterize_polygons([poly], 0.0, 0.0, 1.0, 12, 10)
    # 頂点セルは必ず被覆(保守的ラスタ化の不変量)
    for x, y in poly:
        assert m[int(round(y)), int(round(x))] == 1
    assert m[3, 5] == 1                     # 内部
    assert m[9, 11] == 0                    # 明らかに外
    # 内部格子点の数は概ね面積(セル面積 1 m²)に一致
    assert m.sum() >= 8 * 5


def test_rasterize_thin_polygon_survives():
    """幅 0.4 m の細い歩道はセル中心判定だけだと落ちるが、輪郭ラスタ化で残る。"""
    poly = np.array([[0.2, 0.2], [9.8, 0.2], [9.8, 0.6], [0.2, 0.6]])
    m_edge = TRAN.rasterize_polygons([poly], 0.0, 0.0, 1.0, 12, 6, edges=True)
    m_pt = TRAN.rasterize_polygons([poly], 0.0, 0.0, 1.0, 12, 6, edges=False)
    assert m_pt.sum() == 0
    assert m_edge.sum() > 0


def test_dilate_mask_monotone_and_contains_original():
    m = np.zeros((11, 11), dtype=np.uint8)
    m[5, 5] = 1
    d1 = TRAN.dilate_mask(m, 1)
    d2 = TRAN.dilate_mask(m, 2)
    assert TRAN.dilate_mask(m, 0).sum() == 1
    assert d1.sum() == 9 and d2.sum() == 25
    assert np.all(d1 >= m) and np.all(d2 >= d1)


def test_tran_classify_function_codes():
    assert TRAN.classify_function("2000") == TRAN.CLASS_WALK
    assert TRAN.classify_function("2020") == TRAN.CLASS_WALK
    assert TRAN.classify_function("1000") == TRAN.CLASS_ROAD
    assert TRAN.classify_function("1020") == TRAN.CLASS_ROAD
    assert TRAN.classify_function("3000") == TRAN.CLASS_ISLAND
    assert TRAN.classify_function("7000") == TRAN.CLASS_OTHER
    assert TRAN.classify_function(None) == TRAN.CLASS_OTHER


def test_polygon_area_xy_square():
    P = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0], [0.0, 4.0]])
    assert TRAN.polygon_area_xy(P) == pytest.approx(12.0)


# ===================================================== 8. 決定論
def test_det_npz_is_byte_identical_across_writes(tmp_path):
    arrays = {"a": np.arange(1000, dtype=np.int16).reshape(-1, 2),
              "b": np.linspace(0, 1, 300).astype(np.float32),
              "names": np.asarray(["x", "yy", "zzz"])}
    p1 = DET.save_npz(tmp_path / "one.npz", arrays)
    p2 = DET.save_npz(tmp_path / "two.npz", arrays)
    assert p1.read_bytes() == p2.read_bytes()
    z = np.load(p1)
    assert np.array_equal(z["a"], arrays["a"])
    assert np.array_equal(z["names"], arrays["names"])
    assert list(z.files) == ["a", "b", "names"]


def test_det_npz_differs_when_content_differs(tmp_path):
    a = DET.save_npz(tmp_path / "a.npz", {"v": np.zeros(10, dtype=np.int16)})
    b = DET.save_npz(tmp_path / "b.npz", {"v": np.ones(10, dtype=np.int16)})
    assert a.read_bytes() != b.read_bytes()


def test_dump_json_is_deterministic_and_lf(tmp_path):
    obj = {"z": 1, "a": [1, 2, 3], "j": "日本語"}
    p1 = DET.dump_json(tmp_path / "a.json", obj, indent=1)
    p2 = DET.dump_json(tmp_path / "b.json", obj, indent=1)
    raw = p1.read_bytes()
    assert raw == p2.read_bytes()
    assert b"\r\n" not in raw
    assert raw.endswith(b"\n")
    assert json.loads(raw.decode("utf-8")) == obj


# ----------------------------------------------------- 合成 GML での end-to-end 決定論
def _synth_tran_gml(path):
    """最小の tran CityGML(歩道 1 枚 + 車道 1 枚 + 島 1 枚)。"""
    def ring(pts):
        return " ".join(f"{a} {b} {c}" for a, b, c in pts)
    lat0, lon0 = TRAN.DEFAULT_ORIGIN

    def ll(dx, dy, z=15.5):
        lat = lat0 + dy / TRAN.M_PER_DEG_LAT
        lon = lon0 + dx / (TRAN.M_PER_DEG_LON_EQ * math.cos(math.radians(lat0)))
        return (lat, lon, z)

    def area(tag, gid, code, pts):
        return f"""
  <core:cityObjectMember><tran:Road gml:id="road_{gid}">
    <tran:trafficArea><tran:{tag} gml:id="{gid}">
      <tran:function codeSpace="../../codelists/{tag}_function.xml">{code}</tran:function>
      <tran:lod3MultiSurface><gml:MultiSurface><gml:surfaceMember>
        <gml:Polygon><gml:exterior><gml:LinearRing>
          <gml:posList>{ring(pts)}</gml:posList>
        </gml:LinearRing></gml:exterior></gml:Polygon>
      </gml:surfaceMember></gml:MultiSurface></tran:lod3MultiSurface>
    </tran:{tag}></tran:trafficArea>
  </tran:Road></core:cityObjectMember>"""

    body = (area("TrafficArea", "w1", "2000",
                 [ll(0, 0), ll(20, 0), ll(20, 4), ll(0, 4), ll(0, 0)])
            + area("TrafficArea", "r1", "1000",
                   [ll(0, 4), ll(20, 4), ll(20, 14), ll(0, 14), ll(0, 4)])
            + area("AuxiliaryTrafficArea", "i1", "3000",
                   [ll(8, 6), ll(12, 6), ll(12, 8), ll(8, 8), ll(8, 6)]))
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<core:CityModel xmlns:core="http://www.opengis.net/citygml/2.0" '
        'xmlns:gml="http://www.opengis.net/gml" '
        'xmlns:tran="http://www.opengis.net/citygml/transportation/2.0">'
        + body + "\n</core:CityModel>\n")
    path.write_text(xml, encoding="utf-8")
    return path


def _synth_map(path):
    """sim_rect が読む最小の shibuya_osm.json(local-m 座標)。"""
    path.write_text(json.dumps({
        "meta": {"crs": "local-m"},
        "nodes": [{"id": 0, "x": -30.0, "y": -30.0}, {"id": 1, "x": 60.0, "y": 60.0}],
        "edges": [], "buildings": [], "pois": [],
    }), encoding="utf-8")
    return path


def test_tran_extract_end_to_end_is_deterministic(tmp_path):
    gdir = tmp_path / "tran"
    gdir.mkdir()
    # sim_rect で選ばれるメッシュコードに合わせたファイル名にする
    code = TRAN.meshcode3(*TRAN.DEFAULT_ORIGIN)
    _synth_tran_gml(gdir / f"{code}_tran_6697_op.gml")
    mp = _synth_map(tmp_path / "map.json")
    o1 = tmp_path / "out1"
    o2 = tmp_path / "out2"
    m1 = TRAN.run(gdir, mp, o1, buffer_m=20.0, cell_m=1.0, progress=False)
    m2 = TRAN.run(gdir, mp, o2, buffer_m=20.0, cell_m=1.0, progress=False)
    assert (o1 / "tran_lod3.npz").read_bytes() == (o2 / "tran_lod3.npz").read_bytes()
    assert (o1 / "tran_lod3.json").read_bytes() == (o2 / "tran_lod3.json").read_bytes()
    assert m1["n_polygons"] == 3
    assert m1["n_polygons_by_class"] == {"walk": 1, "road": 1, "island": 1, "other": 0}
    assert m1["area_m2_by_class"]["walk"] == pytest.approx(80.0, abs=0.5)
    assert m1["area_m2_by_class"]["road"] == pytest.approx(200.0, abs=1.0)
    assert m1["area_m2_by_class"]["island"] == pytest.approx(8.0, abs=0.5)
    assert m1["coverage"]["n_covered_all"] > 0
    assert m1["coverage"]["ratio_near"] >= m1["coverage"]["ratio_all"]
    z = np.load(o1 / "tran_lod3.npz")
    assert z["poly_offsets"][-1] == z["xy"].shape[0]
    assert z["cover_all"].shape == (m1["coverage"]["ny"], m1["coverage"]["nx"])
    assert int(z["cover_all"].sum()) == m1["coverage"]["n_covered_all"]
    assert int(z["cover_near"].sum()) == m1["coverage"]["n_covered_near"]


def _synth_ubld_gml(path):
    """最小の ubld CityGML(2 層 × 床 + 内壁の表裏 + 扉 + 閉鎖面)。"""
    lat0, lon0 = UBLD.DEFAULT_ORIGIN

    def ll(dx, dy, z):
        lat = lat0 + dy / UBLD.M_PER_DEG_LAT
        lon = lon0 + dx / (UBLD.M_PER_DEG_LON_EQ * math.cos(math.radians(lat0)))
        return f"{lat} {lon} {z}"

    def poly(pts):
        return ("<gml:surfaceMember><gml:Polygon><gml:exterior><gml:LinearRing>"
                f"<gml:posList>{' '.join(pts)}</gml:posList>"
                "</gml:LinearRing></gml:exterior></gml:Polygon></gml:surfaceMember>")

    def surf(tag, gid, polys, inner=""):
        return (f'<bldg:boundedBy><bldg:{tag} gml:id="{gid}">'
                "<bldg:lod4MultiSurface><gml:MultiSurface>"
                + "".join(polys) +
                "</gml:MultiSurface></bldg:lod4MultiSurface>"
                + inner + f"</bldg:{tag}></bldg:boundedBy>")

    parts = []
    # 2 層の床(z = 15.18-10 と 15.18-4 → local -10 / -4)
    for li, z in enumerate((5.18, 11.18)):
        parts.append(surf("FloorSurface", f"fl{li}", [
            poly([ll(0, 0, z), ll(10, 0, z), ll(10, 8, z)]),
            poly([ll(0, 0, z), ll(10, 8, z), ll(0, 8, z)])]))
    # 内壁: 同じ壁の表裏 2 枚 × 2 層
    for li, z in enumerate((5.18, 11.18)):
        for k, off in enumerate((0.0, 0.2)):
            parts.append(surf("InteriorWallSurface", f"iw{li}{k}", [
                poly([ll(0, off, z), ll(10, off, z), ll(10, off, z + 3)]),
                poly([ll(0, off, z), ll(10, off, z + 3), ll(0, off, z + 3)])]))
    # 扉(1 要素 = 2 ポリゴン)
    parts.append(surf("WallSurface", "w0",
                      [poly([ll(0, 8, 5.18), ll(10, 8, 5.18), ll(10, 8, 8.18)])],
                      inner=("<bldg:opening><bldg:Door gml:id=\"d0\">"
                             "<bldg:lod4MultiSurface><gml:MultiSurface>"
                             + poly([ll(4, 8, 5.18), ll(6, 8, 5.18), ll(6, 8, 7.18)])
                             + poly([ll(4, 8, 5.18), ll(6, 8, 7.18), ll(4, 8, 7.18)])
                             + "</gml:MultiSurface></bldg:lod4MultiSurface>"
                               "</bldg:Door></bldg:opening>")))
    # 閉鎖面
    parts.append(surf("ClosureSurface", "c0",
                      [poly([ll(10, 0, 5.18), ll(10, 8, 5.18), ll(10, 8, 8.18)])]))
    xml = ('<?xml version="1.0" encoding="UTF-8"?>\n'
           '<core:CityModel xmlns:core="http://www.opengis.net/citygml/2.0" '
           'xmlns:gml="http://www.opengis.net/gml" '
           'xmlns:bldg="http://www.opengis.net/citygml/building/2.0">'
           '<core:cityObjectMember><uro:UndergroundBuilding '
           'xmlns:uro="https://www.geospatial.jp/iur/uro/3.0" gml:id="ub0">'
           + "".join(parts) +
           "</uro:UndergroundBuilding></core:cityObjectMember></core:CityModel>\n")
    path.write_text(xml, encoding="utf-8")
    return path


def test_ubld_extract_end_to_end_is_deterministic(tmp_path):
    gml = _synth_ubld_gml(tmp_path / "synth_ubld.gml")
    o1 = tmp_path / "u1"
    o2 = tmp_path / "u2"
    m1 = UBLD.run(gml, o1, layer_sep=2.0, progress=False)
    m2 = UBLD.run(gml, o2, layer_sep=2.0, progress=False)
    assert (o1 / "ubld_lod4.json").read_bytes() == (o2 / "ubld_lod4.json").read_bytes()
    assert ((o1 / "ubld_lod4_mesh.npz").read_bytes()
            == (o2 / "ubld_lod4_mesh.npz").read_bytes())
    # 表裏 2 枚 × 高さ 2 分割 × 2 層 = 8 面 → **層ごとに** 1 本 = 2 本
    # (平面投影だけで潰すと 1 本になり、下の階の壁が消える = flat との差がその証拠)
    assert m1["walls"]["n_faces"] == 8
    assert (m1["walls"]["n_faces"]
            == m1["walls"]["n_non_vertical_faces"]
            + m1["walls"]["n_vertical_faces_below_min_length"]
            + m1["walls"]["n_segments_raw"])
    assert m1["walls"]["n_segments_unique"] == 2
    assert m1["walls"]["n_segments_unique_flat"] == 1
    assert m1["walls"]["n_segments_by_layer"] == [1, 1]
    assert sorted(s["layer"] for s in m1["wall_segments"]) == [0, 1]
    assert m1["doors"]["n_elements"] == 1
    assert m1["doors"]["n_polygons"] == 2
    assert m1["closures"]["n_segments_unique"] == 1
    assert len(m1["layers"]) == 2
    assert [round(ly["z"]) for ly in m1["layers"]] == [-10, -4]
    assert len(m1["wall_segments"]) == m1["walls"]["n_segments_unique"]
    assert len(m1["gates"]) == m1["doors"]["n_elements"]
    z = np.load(o1 / "ubld_lod4_mesh.npz")
    assert z["tri"].shape[0] == z["tri_kind"].shape[0]
    assert int(z["tri"].max()) < z["xyz"].shape[0]


# ===================================================== 9. 成果物の整合性(あれば)
def _load_json(p):
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.mark.skipif(not (DATA / "tiles_lod2" / "index.json").exists(),
                    reason="tiles_lod2 未生成(tools/tiles3d_extract.py を実行)")
def test_artifact_tiles_lod2_index_consistent():
    d = DATA / "tiles_lod2"
    idx = _load_json(d / "index.json")
    assert idx["schema"] == "plateau_tiles_lod2/1"
    assert idx["quant_scale"] == T3D.PLATEAU_QUANT
    assert idx["origin_latlon"] == list(PEX.DEFAULT_ORIGIN)
    assert idx["n_tiles"] == len(idx["tiles"])
    assert idx["n_batches"] - idx["n_shadowed_batches"] == idx["n_unique_gml_id"]
    tot_v = tot_t = 0
    for t in idx["tiles"]:
        assert (d / t["npz"]).exists()
        tot_v += t["n_vertices"]
        tot_t += t["n_triangles"]
        if t["atlas"]:
            assert (d / t["atlas"]).exists()
            assert (d / t["atlas"]).stat().st_size == t["atlas_bytes"]
    assert tot_v == idx["n_vertices"] and tot_t == idx["n_triangles"]


@pytest.mark.skipif(not (DATA / "tiles_lod2" / "index.json").exists(),
                    reason="tiles_lod2 未生成")
def test_artifact_tiles_lod2_geometry_in_bbox():
    """タイル npz の索引が範囲内で、頂点が渋谷 bbox 近傍にあること(座標系の取り違え検知)。"""
    d = DATA / "tiles_lod2"
    idx = _load_json(d / "index.json")
    for t in idx["tiles"][:12]:
        z = np.load(d / t["npz"])
        assert z["tri"].shape[0] == t["n_triangles"]
        assert z["xyz"].shape[0] == t["n_vertices"]
        assert int(z["tri"].max()) < z["xyz"].shape[0]
        assert z["prim_tri_offsets"][-1] == t["n_triangles"]
        xyz = T3D.dequantize_xyz(z["xyz"], z["origin_q"])
        assert np.all(np.abs(xyz[:, 0]) < 6000.0)
        assert np.all(np.abs(xyz[:, 1]) < 6000.0)
        assert np.all(xyz[:, 2] > -60.0) and np.all(xyz[:, 2] < 400.0)


@pytest.mark.skipif(not (DATA / "tran_lod3.json").exists(),
                    reason="tran_lod3 未生成(scripts/plateau_tran_extract.py を実行)")
def test_artifact_tran_lod3_consistent():
    meta = _load_json(DATA / "tran_lod3.json")
    z = np.load(DATA / "tran_lod3.npz")
    assert meta["schema"] == "plateau_tran_lod3/1"
    assert meta["origin_latlon"] == list(PEX.DEFAULT_ORIGIN)
    assert meta["n_polygons"] == z["poly_class"].shape[0]
    assert z["poly_offsets"].shape[0] == meta["n_polygons"] + 1
    assert int(z["poly_offsets"][-1]) == z["xy"].shape[0] == meta["n_vertices"]
    cov = meta["coverage"]
    assert z["cover_all"].shape == (cov["ny"], cov["nx"])
    assert int(z["cover_all"].sum()) == cov["n_covered_all"]
    assert int(z["cover_near"].sum()) == cov["n_covered_near"]
    # 歩行系マスクは全体マスクの部分集合
    assert np.all(z["cover_walk"] <= z["cover_all"])
    assert np.all(z["cover_all"] <= z["cover_near"])
    # 頂点セルは必ず被覆(保守的ラスタ化の不変量)
    xy = (z["xy"].astype(np.int64) + z["origin_q"]) * meta["quant_scale"]
    ii = np.round((xy[:, 0] - cov["x0"]) / cov["cell_m"]).astype(np.int64)
    jj = np.round((xy[:, 1] - cov["y0"]) / cov["cell_m"]).astype(np.int64)
    ok = (ii >= 0) & (ii < cov["nx"]) & (jj >= 0) & (jj < cov["ny"])
    hit = z["cover_all"][jj[ok], ii[ok]]
    # マスクは量子化前の座標で作り、npz は 0.05 m 量子化後なので、セル境界ちょうどに
    # 乗った頂点だけ隣セルへずれる(実測 58 / 238,998 = 0.024%)。1 セル近傍で必ず被覆。
    assert hit.mean() > 0.999
    for a, b in zip(jj[ok][hit == 0], ii[ok][hit == 0]):
        sub = z["cover_all"][max(0, a - 1):a + 2, max(0, b - 1):b + 2]
        assert sub.max() == 1


@pytest.mark.skipif(not (DATA / "ubld_lod4.json").exists(),
                    reason="ubld_lod4 未生成(scripts/plateau_ubld_extract.py を実行)")
def test_artifact_ubld_lod4_consistent():
    meta = _load_json(DATA / "ubld_lod4.json")
    assert meta["schema"] == "plateau_ubld_lod4/1"
    assert meta["origin_latlon"] == list(PEX.DEFAULT_ORIGIN)
    # PLATEAU 標準製品仕様の必須地物数(調査 docs/research/shibuya-3d-highfidelity.md §2.3)
    elems = meta["elements_in_file_by_tag"]
    assert elems["InteriorWallSurface"] == 1934
    assert elems["FloorSurface"] == 145
    assert elems["Door"] == 188
    assert elems["ClosureSurface"] == 218
    assert elems["IntBuildingInstallation"] == 5
    w = meta["walls"]
    assert w["n_elements_in_file"] == (w["n_elements_with_geometry"]
                                       + w["n_elements_opening_only"])
    assert w["n_segments_unique"] <= w["n_segments_raw"]
    # 面の行方が全部説明できる(黙って落ちた面がない)
    assert w["n_faces"] == (w["n_non_vertical_faces"]
                            + w["n_vertical_faces_below_min_length"]
                            + w["n_segments_raw"])
    # 層ごと dedup は平面のみの dedup 以上の本数になる(層をまたぐ潰しが起きない)
    assert w["n_segments_unique"] >= w["n_segments_unique_flat"]
    assert sum(w["n_segments_by_layer"]) == w["n_segments_unique"]
    assert len(meta["wall_segments"]) == w["n_segments_unique"]
    assert len(meta["gates"]) == meta["doors"]["n_elements"] == 188
    assert len(meta["closure_segments"]) == meta["closures"]["n_segments_unique"]
    assert len(meta["layers"]) >= 2
    zs = [ly["z"] for ly in meta["layers"]]
    assert zs == sorted(zs)
    for a, b in zip(zs, zs[1:]):
        assert b - a >= meta["params"]["layer_sep_m"] - 1e-9
    # 全ての壁/ゲートが層に割り当たっている
    assert all(0 <= s["layer"] < len(zs) for s in meta["wall_segments"])
    assert all(0 <= g["layer"] < len(zs) for g in meta["gates"])
    z = np.load(DATA / "ubld_lod4_mesh.npz")
    assert z["tri"].shape[0] == meta["mesh"]["n_triangles"]
    assert z["xyz"].shape[0] == meta["mesh"]["n_vertices"]
    assert int(z["tri"].max()) < z["xyz"].shape[0]
    assert int(z["tri_kind"].max()) < len(meta["mesh"]["kind_names"])
    xyz = (z["xyz"].astype(np.int64) + z["origin_q"]) * meta["params"]["quant_scale"]
    # 地下街は 332m×410m の一角・地表基準で概ね -15..+3 m
    assert np.all(np.abs(xyz[:, 0]) < 400.0) and np.all(np.abs(xyz[:, 1]) < 400.0)
    assert xyz[:, 2].min() > -20.0 and xyz[:, 2].max() < 10.0


@pytest.mark.skipif(not (DATA / "tiles_batch_attrs.json").exists(),
                    reason="tiles_batch_attrs 未生成")
def test_artifact_batch_attrs_align_with_tiles():
    attrs = _load_json(DATA / "tiles_batch_attrs.json")
    idx = _load_json(DATA / "tiles_lod2" / "index.json")
    assert attrs["schema"] == "plateau_tiles_batch_attrs/1"
    assert set(attrs["tiles"]) == {t["npz"] for t in idx["tiles"]}
    for t in idx["tiles"]:
        a = attrs["tiles"][t["npz"]]
        assert "attributes" not in a              # 入れ子重複キーは常に除外
        assert len(a["gml_id"]) == t["batch_length"]
        assert len(a["_shadowed_batches"]) == t["n_shadowed_batches"]
        assert all(0 <= b < t["batch_length"] for b in a["_shadowed_batches"])
