"""可視行列の事前計算 CLI(scripts/build_visibility.py・C0 第68バッチ)のテスト。

方針:
- **合成ミニ世界だけで検算する**(実地図の大規模実行はテスト外。ここは軽い単体のみ)。
  ミニ世界は「面が載る自建物 × 高いビル × 低いビル × 中くらいのビル」を東西の廊下として
  並べ、遮蔽・非遮蔽・部分可視の 3 通りを**手計算した値**と突き合わせる。
- 幾何の期待値はテスト側で独立に手計算して定数で書く(実装の出力をそのまま焼き付けない)。
- 疎格納(可視の行だけ)・再実行のバイト一致・meta の runtime キー隔離も固定する。
- シム本体には一切触らないスクリプトなので、L1 / ゴールデン / 乱数 stream への影響はゼロ
  (このテストも Simulation を起動しない)。

手計算の土台(scripts/build_visibility.py の docstring と同じ式):
    視線 z(s) = fz·(1 − s) + pz·s  (s = 面 → 視点の正規化水平距離, pz = 目線高)
    遮蔽 ⟺ top_z > z(s) ⟺ g = (top_z − pz·s)/(1 − s) > fz
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))          # scripts/ は package ではない
import build_visibility as bv                       # noqa: E402

EYE = 1.5
Z_BASE, Z_TOP = 20.0, 30.0
Z_MID = (Z_BASE + Z_TOP) / 2.0


# --------------------------------------------------------------------- 合成ミニ世界
def _rect(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


#   建物 id      : (矩形, 高さ[m])   ※東西(+X)方向の 3 本の廊下に並べる
_BUILDINGS = {
    "bhost":  (_rect(-10, -10, 0, 10), 35.0),    # 廊下A の面の自建物
    "btall":  (_rect(40, -10, 50, 10), 60.0),    # 廊下A: 高いビル(遮蔽する)
    "bhost2": (_rect(-10, 30, 0, 50), 35.0),     # 廊下B の面の自建物
    "blow":   (_rect(40, 30, 50, 50), 5.0),      # 廊下B: 低いビル(越えて見える)
    "bhost3": (_rect(-10, 70, 0, 90), 35.0),     # 廊下C の面の自建物
    "bmid":   (_rect(40, 70, 50, 90), 14.0),     # 廊下C: 中くらい(下端だけ隠れる)
}


def _write_world(tmp: Path, faces_extra=None, face_a_building="bhost",
                 face_d_building="bhost") -> tuple[Path, Path, Path]:
    """ミニ世界(地図 JSON・高さ表 JSON・面プロファイル YAML)を書いてパスを返す。"""
    buildings = []
    for bid, (fp, _h) in _BUILDINGS.items():
        buildings.append({"id": bid, "name": "", "levels": 1, "kind": "generic",
                          "footprint": fp, "entrance": "n1", "area": 100})
    world = {
        "meta": {"crs": "local-m"},
        "nodes": [{"id": "n1", "x": 0.0, "y": -60.0},
                  {"id": "n2", "x": 100.0, "y": -60.0}],
        "edges": [{"u": "n1", "v": "n2", "klass": "footway",
                   "geometry": [[0.0, -60.0], [100.0, -60.0]]}],
        "buildings": buildings, "pois": [],
    }
    map_p = tmp / "mini_map.json"
    map_p.write_text(json.dumps(world, ensure_ascii=False), encoding="utf-8")

    heights = {"meta": {"schema": "building-heights-1.0"},
               "heights": {bid: {"h": h, "src": "plateau"}
                           for bid, (_fp, h) in _BUILDINGS.items()}}
    h_p = tmp / "mini_heights.json"
    h_p.write_text(json.dumps(heights, ensure_ascii=False), encoding="utf-8")

    faces = [
        {"id": "fa", "slot": "A", "building": face_a_building, "kind": "large",
         "x": 0.0, "y": 0.0, "z_base": Z_BASE, "z_top": Z_TOP,
         "normal_deg": 90.0, "w_m": 10.0},
        {"id": "fb", "slot": "B", "building": "bhost2", "kind": "normal",
         "x": 0.0, "y": 40.0, "z_base": Z_BASE, "z_top": Z_TOP,
         "normal_deg": 90.0, "w_m": 10.0},
        {"id": "fc", "slot": "C", "building": "bhost3", "kind": "normal",
         "x": 0.0, "y": 80.0, "z_base": Z_BASE, "z_top": Z_TOP,
         "normal_deg": 90.0, "w_m": 10.0},
        # fd = 法線を書かない面(背面カリングなし)。自建物の遮蔽除外を裏側から観測するための面。
        {"id": "fd", "slot": "D", "kind": "normal",
         "x": 0.0, "y": 0.0, "z_base": Z_BASE, "z_top": Z_TOP, "w_m": 10.0},
    ]
    if face_d_building is not None:
        faces[3]["building"] = face_d_building
    if faces_extra is not None:
        faces = faces_extra
    f_p = tmp / "mini_faces.yaml"
    f_p.write_text(yaml.safe_dump({"meta": {"schema": "visibility-faces-1.0",
                                            "name": "mini"},
                                   "faces": faces}, allow_unicode=True),
                   encoding="utf-8")
    return map_p, h_p, f_p


def _run(tmp: Path, out_name: str, *extra, **kw) -> Path:
    """CLI を 1 回走らせて出力フォルダを返す。格子は x = -100+2·ix / y = -100+2·iy。"""
    map_p, h_p, f_p = _write_world(tmp, **kw)
    out = tmp / out_name
    rc = bv.main(["--faces", str(f_p), "--map", str(map_p), "--heights", str(h_p),
                  "--bbox", "-101", "-101", "101", "101", "--cell", "2.0",
                  "--eye", str(EYE), "--out", str(out), *extra])
    assert rc == 0
    return out


def _rows(out: Path) -> dict:
    """{(vp_x, vp_y, face_id): {"visible","dist_m","incidence_deg"}} に展開する。"""
    t = pq.read_table(out / "visibility_matrix.parquet").to_pydict()
    d = {}
    for i in range(len(t["face_id"])):
        d[(round(t["vp_x"][i], 3), round(t["vp_y"][i], 3), t["face_id"][i])] = {
            "visible": t["visible"][i], "dist_m": t["dist_m"][i],
            "incidence_deg": t["incidence_deg"][i],
            "vp_ix": t["vp_ix"][i], "vp_iy": t["vp_iy"][i]}
    return d


def _meta(out: Path) -> dict:
    return json.loads((out / "visibility_meta.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------- 面プロファイル
def test_load_faces_sorts_and_validates(tmp_path):
    _m, _h, f_p = _write_world(tmp_path)
    meta, faces = bv.load_faces(f_p)
    assert meta["schema"] == "visibility-faces-1.0"
    assert [f["id"] for f in faces] == ["fa", "fb", "fc", "fd"]      # id 昇順=決定論
    assert faces[3]["normal_deg"] is None                            # 法線は省略可
    assert faces[0]["kind"] == "large" and faces[0]["slot"] == "A"


@pytest.mark.parametrize("mutate, msg", [
    (lambda f: f[0].update({"kind": "huge"}), "kind"),
    (lambda f: f[0].update({"id": "fb"}), "重複"),
    (lambda f: f[0].update({"z_top": 1.0}), "z_top"),
    (lambda f: f[0].update({"colour": "red"}), "未知のキー"),
    (lambda f: f[0].pop("x"), "必須キー"),
])
def test_load_faces_rejects_bad_profiles(tmp_path, mutate, msg):
    faces = [
        {"id": "fa", "kind": "large", "x": 0.0, "y": 0.0,
         "z_base": 1.0, "z_top": 2.0},
        {"id": "fb", "kind": "normal", "x": 5.0, "y": 0.0,
         "z_base": 1.0, "z_top": 2.0},
    ]
    mutate(faces)
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump({"meta": {"schema": "visibility-faces-1.0"},
                                 "faces": faces}), encoding="utf-8")
    with pytest.raises(ValueError, match=msg):
        bv.load_faces(p)


def test_normal_vec_is_compass_bearing():
    """0=北(+Y)・90=東(+X)・時計回り。プロファイルの規約を固定する。"""
    for deg, (ex, ey) in ((0.0, (0.0, 1.0)), (90.0, (1.0, 0.0)),
                          (180.0, (0.0, -1.0)), (270.0, (-1.0, 0.0))):
        nx, ny = bv.normal_vec(deg)
        assert nx == pytest.approx(ex, abs=1e-12)
        assert ny == pytest.approx(ey, abs=1e-12)


# --------------------------------------------------------------------- LOS の核(純関数)
def test_max_block_g_matches_hand_computation():
    """壁 1 枚の G = (top_z − pz·s)/(1 − s) を手計算と一致させる。"""
    # 面 (0,0) → 視点 (100,0)。壁は x=40 の線分(y=−10..10)、高さ 12m、目線 1.5m。
    G = bv.max_block_g(np.array([100.0]), np.array([0.0]), np.array([EYE]),
                       0.0, 0.0,
                       np.array([40.0]), np.array([-10.0]),
                       np.array([40.0]), np.array([10.0]),
                       np.array([12.0]))
    s = 0.4
    assert G[0] == pytest.approx((12.0 - EYE * s) / (1.0 - s))
    # 交差しない向き(視点が壁の手前)→ −inf
    G2 = bv.max_block_g(np.array([20.0]), np.array([0.0]), np.array([EYE]),
                        0.0, 0.0,
                        np.array([40.0]), np.array([-10.0]),
                        np.array([40.0]), np.array([10.0]),
                        np.array([12.0]))
    assert G2[0] == -math.inf


# --------------------------------------------------------------------- 遮蔽の 3 通り
def test_tall_building_blocks_and_low_building_does_not(tmp_path):
    out = _run(tmp_path, "base")
    rows = _rows(out)

    # 廊下A: 高さ 60m のビル越し → 面 fa(z 20-30m)は見えない
    #   遠い辺 x=50 で s=0.5 → g=(60−1.5·0.5)/0.5=118.5 > 30 = z_top → 全サンプル遮蔽
    assert (100.0, 0.0, "fa") not in rows

    # 廊下B: 高さ 5m のビル越し → 面 fb は完全可視
    #   g_max = (5−1.5·0.5)/0.5 = 8.5 < 20 = z_base
    r = rows[(100.0, 40.0, "fb")]
    assert r["visible"] == pytest.approx(1.0)

    # 廊下C: 高さ 14m → 上端だけ見える = 部分可視 0.5
    #   g_max = (14−1.5·0.5)/0.5 = 26.5 → z_top(30)のみ可視・中心(25)と下端(20)は遮蔽
    assert 26.5 > Z_MID > Z_BASE and Z_TOP > 26.5
    assert rows[(100.0, 80.0, "fc")]["visible"] == pytest.approx(0.5)


def test_own_building_is_not_an_occluder(tmp_path):
    """面が載っている建物は遮蔽から外す(= 裏側からは自建物を透過して見える)。

    法線を書けば背面カリングで裏は落ちるので、この「透過」を観測できるのは法線の無い面だけ。
    building を外した対照ランでは同じ視点が自建物の西壁(35m)に遮蔽されて消える。
    """
    rows_on = _rows(_run(tmp_path, "own_on"))               # fd.building = bhost
    rows_off = _rows(_run(tmp_path, "own_off", face_d_building=None))

    # 視点 (−100, 0) は bhost(x∈[−10,0])の真西。fd は法線が無いので背面カリングされない。
    assert (-100.0, 0.0, "fd") in rows_on
    assert (-100.0, 0.0, "fd") not in rows_off              # 自建物を外すと西壁が遮る
    #   検算: 西壁 x=−10 は s=0.1 → g=(35−1.5·0.1)/0.9=38.7 > 30 = z_top → 全サンプル遮蔽
    assert (35.0 - EYE * 0.1) / 0.9 > Z_TOP
    # 法線を持つ面 fa は同じ視点で背面カリング(入射角 180° > 85°)により行が無い
    assert (-100.0, 0.0, "fa") not in rows_on


def test_distance_and_incidence_values(tmp_path):
    """dist_m(目線点→面中心の 3D 距離)と incidence_deg(法線からの水平角)の検算。"""
    rows = _rows(_run(tmp_path, "geom"))
    # 視点 (20,20) → 面 fa (0,0) 法線=東。遮蔽物なし(btall は x≥40)。
    r = rows[(20.0, 20.0, "fa")]
    assert r["visible"] == pytest.approx(1.0)
    expect_d = math.sqrt(20.0 ** 2 + 20.0 ** 2 + (Z_MID - EYE) ** 2)   # = 36.7729…
    assert r["dist_m"] == pytest.approx(round(expect_d, 2), abs=1e-3)
    assert r["incidence_deg"] == pytest.approx(45.0, abs=1e-3)         # (20,20) vs 法線 (1,0)
    # 正対(真東)なら 0°。距離も真横成分ゼロで検算できる。
    r0 = rows[(30.0, 0.0, "fa")]
    assert r0["incidence_deg"] == pytest.approx(0.0, abs=1e-3)
    assert r0["dist_m"] == pytest.approx(
        round(math.sqrt(30.0 ** 2 + (Z_MID - EYE) ** 2), 2), abs=1e-3)
    # 法線の無い面は null(欠測を 0 と偽らない)
    assert rows[(20.0, 20.0, "fd")]["incidence_deg"] is None


def test_back_face_culling_by_normal(tmp_path):
    """法線から --max-incidence-deg を超える視点は不可視(片面の掲出物)。"""
    wide = _rows(_run(tmp_path, "cull_wide"))                        # 既定 85°
    narrow = _rows(_run(tmp_path, "cull_narrow", "--max-incidence-deg", "60"))
    inc = math.degrees(math.acos(20.0 / math.hypot(20.0, 60.0)))     # = 71.565°
    assert wide[(20.0, 60.0, "fa")]["incidence_deg"] == pytest.approx(inc, abs=1e-2)
    assert (20.0, 60.0, "fa") not in narrow                          # 71.6° > 60°
    assert (30.0, 0.0, "fa") in narrow                               # 正対は残る
    assert (-100.0, 0.0, "fa") not in wide                           # 真裏は既定でも落ちる


# --------------------------------------------------------------------- 疎格納・件数
def test_sparse_storage_only_visible_rows(tmp_path):
    out = _run(tmp_path, "sparse")
    t = pq.read_table(out / "visibility_matrix.parquet").to_pydict()
    vis = np.array(t["visible"], dtype=float)
    assert vis.size > 0
    assert (vis > 0.0).all()                              # 0.0 の行は 1 本も無い
    assert set(np.unique(vis)) <= {0.5, 1.0}
    meta = _meta(out)
    n_pairs = meta["counts"]["n_pairs_total"]
    assert meta["counts"]["n_rows"] == vis.size < n_pairs  # 疎(全ペアより少ない)
    assert n_pairs == meta["grid"]["n_viewpoints"] * meta["counts"]["n_faces"]
    assert meta["counts"]["n_full"] + meta["counts"]["n_partial"] == vis.size
    # 建物内部のセルは視点から除外されている
    assert meta["grid"]["n_outdoor"] < meta["grid"]["n_cells"]
    assert meta["visibility"]["overall_rate"] == pytest.approx(vis.size / n_pairs, abs=1e-6)


def test_indoor_cells_are_excluded(tmp_path):
    """建物フットプリント内部の格子セルは視点にならない(屋外のみ)。"""
    out = _run(tmp_path, "indoor")
    rows = _rows(out)
    # 内部(境界セルの点in多角形の扱いに依らない範囲)に視点が 1 つも無いこと
    assert not any(-8.0 <= x <= -2.0 and -8.0 <= y <= 8.0 for x, y, _f in rows)
    meta = _meta(out)
    # 6 棟 × 10m×20m = 1200 m² / (2m)² = 300 セル前後が屋内(境界の丸めで多少ぶれる)
    n_indoor = meta["grid"]["n_cells"] - meta["grid"]["n_outdoor"]
    assert 200 <= n_indoor <= 400


def test_near_edge_option_restricts_viewpoints(tmp_path):
    """--near-edge-m は道路ポリライン近傍(=歩行可能面の近似)へ視点を絞る。"""
    base = _meta(_run(tmp_path, "ne_off"))
    near = _meta(_run(tmp_path, "ne_on", "--near-edge-m", "5"))
    assert near["grid"]["n_near_edge"] is not None
    assert 0 < near["grid"]["n_viewpoints"] < base["grid"]["n_viewpoints"]
    # ミニ世界の道は y=−60 の直線 1 本(x 0..100)なので、その帯だけが残る
    assert near["grid"]["n_viewpoints"] <= 6 * 51 + 10


# --------------------------------------------------------------------- 決定論
def test_rerun_is_byte_identical(tmp_path):
    """再実行で matrix はバイト一致・meta も runtime キーを除いて一致。"""
    a = _run(tmp_path, "det_a")
    b = _run(tmp_path, "det_b")
    ba = (a / "visibility_matrix.parquet").read_bytes()
    bb = (b / "visibility_matrix.parquet").read_bytes()
    assert ba == bb and len(ba) > 0
    ma, mb = _meta(a), _meta(b)
    assert set(ma) == set(mb) and "runtime" in ma
    ma.pop("runtime")
    mb.pop("runtime")
    assert json.dumps(ma, sort_keys=True) == json.dumps(mb, sort_keys=True)
    # runtime に隔離されているのは実行ごとに変わる値だけ
    assert set(_meta(a)["runtime"]) >= {"elapsed_sec", "matrix_bytes"}


def test_row_order_is_viewpoint_then_face(tmp_path):
    """行順 = (視点グリッド順, face_id 昇順)。下流のテーブル参照が前提にできる決定論。"""
    t = pq.read_table(_run(tmp_path, "order") / "visibility_matrix.parquet").to_pydict()
    key = [(t["vp_iy"][i], t["vp_ix"][i], t["face_id"][i])
           for i in range(len(t["face_id"]))]
    assert key == sorted(key)


def test_chunking_does_not_change_the_content(tmp_path):
    """--chunk は行グループの切り方を変えるだけで、行の内容は同一(分割処理の正しさ)。"""
    a = _rows(_run(tmp_path, "chunk_big", "--chunk", "100000"))
    b = _rows(_run(tmp_path, "chunk_small", "--chunk", "97"))
    assert a == b and len(a) > 100


# --------------------------------------------------------------------- 高さ・欠測の扱い
def test_without_heights_nothing_occludes(tmp_path):
    """--heights を渡さないと遮蔽物ゼロ(= 高さ欠測を 0m とも無限大とも解釈しない縮退)。"""
    map_p, _h, f_p = _write_world(tmp_path)
    out = tmp_path / "noh"
    assert bv.main(["--faces", str(f_p), "--map", str(map_p),
                    "--bbox", "-101", "-101", "101", "101", "--cell", "2.0",
                    "--eye", str(EYE), "--out", str(out)]) == 0
    meta = _meta(out)
    assert meta["occluders"]["n_with_height"] == 0 and meta["occluders"]["n_edges"] == 0
    rows = _rows(out)
    assert (100.0, 0.0, "fa") in rows                    # 高いビルが遮蔽物として存在しない
    assert rows[(100.0, 0.0, "fa")]["visible"] == pytest.approx(1.0)


def test_meta_records_vai_columns_and_limits(tmp_path):
    """VAI の幾何 3 変数を持ち、照明・滞留が将来列であることを meta に明記している。"""
    meta = _meta(_run(tmp_path, "vai"))
    assert "dist_m" in meta["columns"] and "incidence_deg" in meta["columns"]
    assert "lighting" in meta["columns"]["_future"] and "dwell" in meta["columns"]["_future"]
    assert meta["params"]["eye_m"] == pytest.approx(EYE)
    ids = [f["id"] for f in meta["faces"]]
    assert ids == sorted(ids)
    assert meta["faces"][0]["w_m"] == pytest.approx(10.0)   # サイズ変数は meta 側に持つ
