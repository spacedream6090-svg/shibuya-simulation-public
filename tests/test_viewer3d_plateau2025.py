"""A5: PLATEAU 2025年度版(V5)への 3D ビューア更新の検収。

**ネットワークは一切使わない**。生成 HTML の文字列検査と、書き出し payload の数値検査
(+ QuickJS があれば実際に注入された JS の実行)だけで固定する。

固定する契約:
- 出典表示: PLATEAU 由来の形状を描くランには、版(2025年度 / V5)・配信元(G空間情報
  センター `plateau-13113-shibuya-ku-2025`)・ライセンス(PLATEAU Site Policy)が
  **常時表示の独立要素**として入る(`#osmAttr` に相乗りしない = OSM トグルで消えない)。
- 道路 LOD3(udx/tran)は**新規トグルレイヤ**で、既定はチェック無し + mesh.visible=false
  (= 現行の見た目と同等)。
- 新引数を渡さない/None を渡すのは従来の生成物とバイト同一(旧ランを 1 バイトも変えない)。
- 道路 LOD3 は X線トグル・接近フェードと状態を共有しない(buildingMats に入れない)。
- payload の数値は data/plateau/tran_lod3.{json,npz} と一致する(量子化の往復)。

版の来歴: 2023年度版 → **2025年度版(標準製品仕様書 第5版 = V5)**。本リポの
data/plateau/* は 2025年度版 CityGML から抽出済みで、ビューアは 3D Tiles を
ネットワーク配信でストリーミングしない(file:// 単体で開ける自己完結が設計要件)。
"""
from __future__ import annotations

import base64
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data" / "plateau"


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MV3 = _load("make_viewer3d_p2025_t", "viz/make_viewer3d.py")


# --------------------------------------------------------------- 最小のシーン/軌跡
def _scene_json():
    return json.dumps({"meta": {"crs": "local-m", "step_minutes": 10, "floor_height": 3.5},
                       "buildings": [{"id": "b1", "kind": "retail", "name": "x",
                                      "footprint": [[0, 0], [10, 0], [10, 10], [0, 10]],
                                      "levels": 2, "below": 0, "height": 7.0, "base": 0.0,
                                      "depth": 7.0, "cx": 5.0, "cy": 5.0}],
                       "roads": [], "rails": [], "pois": []})


def _tracks_json():
    return json.dumps({"meta": {"nSteps": 2, "step_minutes": 10, "start_min": 420,
                                "floor_height": 3.5},
                       "agents": [{"id": 1, "name": "a", "visitor": False,
                                   "occupation": "x", "age": 30, "gender": "f",
                                   "has_car": False, "has_bicycle": False}],
                       "ids": [1], "positions": [[[0, 0, 0]], [[1, 1, 0]]],
                       "moves": [[None], [None]],
                       "traffic": [{"n": 0, "segs": []}, {"n": 0, "segs": []}],
                       "sim_min": [420, 430]})


_PW = json.dumps({"quant_scale": 0.05, "matched_ids": [], "n_vertices": 0,
                  "n_triangles": 0, "positions_b64": "", "indices_b64": "",
                  "colors_b64": ""})


def _tran3_stub(tris=(("walk", (0, 0, 0), (20, 0, 0), (20, 20, 0)),)):
    """量子化済み int16 の合成 payload(1 三角形 = 3 頂点 × (x,y,z))。"""
    names = ["walk", "road", "island", "other"]
    P, C = [], []
    for cls, *vs in tris:
        C.append(names.index(cls))
        for v in vs:
            P.append(list(v))
    return json.dumps({
        "schema": "plateau_tran_web/1", "quant_scale": 0.05,
        "class_names": names, "n_triangles": len(C), "n_vertices": len(P),
        "lod": "lod3MultiSurface",
        "positions_b64": base64.b64encode(
            np.asarray(P, dtype="<i2").tobytes()).decode("ascii"),
        "tri_class_b64": base64.b64encode(
            np.asarray(C, dtype="<u1").tobytes()).decode("ascii"),
    })


def _tran3_section(html: str) -> str:
    m = re.search(r"(// -+ 道路面 LOD3[\s\S]*?)\n// -+ ループ", html)
    assert m, "道路 LOD3 の注入ブロックが見つからない"
    return m.group(1)


def _code_only(js: str) -> str:
    """行コメントを落とす(「何に触らないか」を**コード**で見るため)。

    このブロックに文字列リテラル中の `//` は無い(URL を持たない)ので単純除去でよい。
    """
    assert "://" not in js, "URL が入ったらこの単純な除去は使えない"
    return "\n".join(re.sub(r"//.*$", "", ln) for ln in js.splitlines())


# ============================================================ 1. 後方互換(バイト同一)
def test_no_new_data_is_byte_identical():
    """新引数を渡さない = None を渡す = 従来の生成物と完全一致。"""
    a = MV3.build_html("t", _scene_json(), _tracks_json())
    b = MV3.build_html("t", _scene_json(), _tracks_json(), tran3_json=None, tran3_src=None)
    assert a == b
    for marker in ("lyTran3", "plateau-tran-data", "PLATEAU_TRAN",
                   "plateauAttr", "plateau-13113-shibuya-ku-2025"):
        assert marker not in a, f"PLATEAU 無しランに {marker} が混入している"


def test_generation_is_deterministic():
    t3 = _tran3_stub()
    a = MV3.build_html("t", _scene_json(), _tracks_json(), plateau_json=_PW, tran3_json=t3)
    b = MV3.build_html("t", _scene_json(), _tracks_json(), plateau_json=_PW, tran3_json=t3)
    assert a == b


# ============================================================ 2. 出典表示(帰属)
@pytest.mark.parametrize("kw", [
    {"plateau_json": _PW},
    {"plateau_src": "plateau_mesh.js"},
    {"plateau_src": "plateau_mesh.js", "plateau_tex_src": "plateau_tex.js"},
    {"tran3_json": _tran3_stub()},
    {"tran3_src": "plateau_tran.js"},
])
def test_credits_present_for_every_plateau_path(kw):
    html = MV3.build_html("t", _scene_json(), _tracks_json(), **kw)
    assert 'id="plateauAttr"' in html
    assert "出典: 国土交通省 都市局" in html                # 配信ページが求めるクレジット
    assert "Project PLATEAU" in html
    assert "渋谷区(2025年度)" in html                     # 版
    assert "第5版(V5)" in html                            # 製品仕様書の版
    assert "plateau-13113-shibuya-ku-2025" in html          # 配信元のデータセット ID
    assert "https://www.geospatial.jp/ckan/dataset/plateau-13113-shibuya-ku-2025" in html
    assert "PLATEAU Site Policy" in html
    assert "https://www.mlit.go.jp/plateau/site-policy/" in html


def test_credits_are_not_hidden_by_the_osm_toggle():
    """出典が OSM レイヤーのトグルで display:none にされない(独立要素であること)。"""
    html = MV3.build_html("t", _scene_json(), _tracks_json(), plateau_json=_PW)
    # applyLayers が display を触るのは #osmAttr だけ
    assert "document.getElementById('osmAttr').style.display" in html
    assert "getElementById('plateauAttr').style.display" not in html
    # 出典は #osmAttr の**外**にある(相乗りしていない)
    osm = re.search(r'<div id="osmAttr">.*?</div>', html, re.S)
    assert osm and "plateauAttr" not in osm.group(0)


def test_no_credits_without_plateau():
    html = MV3.build_html("t", _scene_json(), _tracks_json())
    assert "plateauAttr" not in html and "Project PLATEAU" not in html


def test_source_constants_are_the_2025_v5_edition():
    assert "2025" in MV3.PLATEAU_DATASET
    assert MV3.PLATEAU_DATASET_ID == "plateau-13113-shibuya-ku-2025"
    assert MV3.PLATEAU_DATASET_URL.endswith("plateau-13113-shibuya-ku-2025")
    assert "第5版" in MV3.PLATEAU_SPEC and "V5" in MV3.PLATEAU_SPEC
    assert MV3.PLATEAU_LICENSE_URL.startswith("https://www.mlit.go.jp/plateau/")
    # 旧 2023年度版は「来歴コメント」にだけ残し、生成 HTML には出さない
    html = MV3.build_html("t", _scene_json(), _tracks_json(), plateau_json=_PW)
    assert "shibuya-ku-2023" not in html


# ============================================================ 3. 道路 LOD3 レイヤー
def test_tran3_toggle_exists_and_defaults_off():
    html = MV3.build_html("t", _scene_json(), _tracks_json(), tran3_json=_tran3_stub())
    assert 'id="lyTran3"' in html
    assert "道路面 LOD3(車道/歩道)" in html
    # チェック無しで出る(既存の「道路」トグルは checked のまま)
    tog = re.search(r'<input type="checkbox" id="lyTran3"[^>]*>', html)
    assert tog and "checked" not in tog.group(0)
    assert '<input type="checkbox" id="lyRoad" checked>' in html
    # メッシュも作った直後は非表示
    assert "mesh.visible = false;" in _tran3_section(html)


def test_tran3_embedded_vs_sidecar():
    emb = MV3.build_html("t", _scene_json(), _tracks_json(), tran3_json=_tran3_stub())
    assert '<script type="application/json" id="plateau-tran-data">' in emb
    assert '<script src="plateau_tran.js">' not in emb
    side = MV3.build_html("t", _scene_json(), _tracks_json(), tran3_src="plateau_tran.js")
    assert '<script src="plateau_tran.js"></script>' in side
    assert 'id="plateau-tran-data"' not in side
    # どちらの読み口も JS 側に残っている(サイドカーを消しても落ちない)
    assert "typeof PLATEAU_TRAN !== 'undefined'" in _tran3_section(side)


def test_tran3_does_not_touch_xray_or_proximity_fade():
    """X線(buildingMats)・接近フェード(userData.plateau)と状態を共有しない。"""
    html = MV3.build_html("t", _scene_json(), _tracks_json(),
                          plateau_json=_PW, tran3_json=_tran3_stub())
    sec = _code_only(_tran3_section(html))
    assert "buildingMats" not in sec, "道路面が X 線トグルの対象に混ざっている"
    assert "userData.plateau" not in sec, "道路面が接近フェードの対象に混ざっている"
    assert "buildingMeshes" not in sec
    # 逆向き: 既存の X 線ハンドラは buildingMats しか触っていない
    assert ("for(const m of buildingMats){ m.opacity = o; m.needsUpdate = true; }") in html


def test_tran3_coexists_with_every_other_plateau_layer():
    """テクスチャ LOD2.2 / 地下街 LOD4.1 / 道路 LOD3 を同時に注入しても壊れない。"""
    html = MV3.build_html("t", _scene_json(), _tracks_json(),
                          plateau_src="plateau_mesh.js", plateau_tex_src="plateau_tex.js",
                          has_extras=True, has_ubld4=True, tran3_src="plateau_tran.js")
    for marker in ('id="lyTex"', 'id="lyUgai"', 'id="ub4Chips"', 'id="lyTran3"',
                   'id="plateauAttr"'):
        assert marker in html
    # 地下街 LOD4.1 の既定 OFF も維持(地表から箱が生えて見えるため)
    assert '<input type="checkbox" id="lyUgai"> 地下街' in html


def test_generated_variants_parse_as_javascript():
    esprima = pytest.importorskip("esprima")
    for kw in [{}, {"tran3_json": _tran3_stub()}, {"tran3_src": "plateau_tran.js"},
               {"plateau_json": _PW, "tran3_json": _tran3_stub()},
               {"plateau_src": "plateau_mesh.js", "plateau_tex_src": "plateau_tex.js",
                "has_extras": True, "has_ubld4": True, "tran3_src": "plateau_tran.js"}]:
        html = MV3.build_html("t", _scene_json(), _tracks_json(), **kw)
        body = re.findall(r"<script>(.*?)</script>", html, re.S)[-1]
        esprima.parseScript(body)


# ============================================================ 4. payload の数値契約
def _write_tran(d: Path, polys, quant=0.05, origin_q=(-100, -200), schema=None):
    """合成 tran_lod3.{json,npz}。polys = [(class_idx, [(x,y),...], (zmin,zmax)), ...]"""
    d.mkdir(parents=True, exist_ok=True)
    xy, offs, cls, zr = [], [0], [], []
    for c, verts, z in polys:
        xy += list(verts)
        offs.append(offs[-1] + len(verts))
        cls.append(c)
        zr.append(z)
    np.savez(d / "tran_lod3.npz",
             xy=np.asarray(xy, dtype=np.int16).reshape(-1, 2),
             origin_q=np.asarray(origin_q, dtype=np.int64),
             poly_offsets=np.asarray(offs, dtype=np.int32),
             poly_class=np.asarray(cls, dtype=np.uint8),
             poly_code=np.zeros(len(cls), dtype=np.uint16),
             poly_z=np.asarray(zr, dtype=np.float32).reshape(-1, 2))
    (d / "tran_lod3.json").write_text(json.dumps({
        "schema": schema or "plateau_tran_lod3/1", "quant_scale": quant,
        "origin_latlon": [35.6595, 139.70062], "ground0": 15.18,
        "lod": "lod3MultiSurface",
        "n_polygons_by_class": {"walk": sum(1 for c in cls if c == 0),
                                "road": sum(1 for c in cls if c == 1),
                                "island": sum(1 for c in cls if c == 2), "other": 0},
        "area_m2_by_class": {}, "attribution": "テスト",
    }, ensure_ascii=False), encoding="utf-8")
    return d


def test_tran3_absent_data_means_none(tmp_path):
    assert MV3._load_tran3(tmp_path / "nope") is None
    assert MV3.tran3_web_json(tmp_path / "nope") is None


def test_tran3_rejects_unknown_schema(tmp_path):
    d = _write_tran(tmp_path / "s", [(0, [(0, 0), (10, 0), (10, 10)], (1.0, 1.2))],
                    schema="plateau_tran_lod3/9")
    with pytest.raises(SystemExit, match="schema"):
        MV3._load_tran3(d)


def test_tran3_rejects_non_triangle_polygons(tmp_path):
    d = _write_tran(tmp_path / "q", [(1, [(0, 0), (10, 0), (10, 10), (0, 10)], (0.0, 0.0))])
    with pytest.raises(SystemExit, match="3 頂点でない"):
        MV3.build_tran3_web(MV3._load_tran3(d))


def test_tran3_quantization_round_trip(tmp_path):
    """絶対量子化 (xy + origin_q) と z の平板近似(帯の中点)が payload で復元できる。"""
    polys = [(0, [(0, 0), (10, 0), (10, 10)], (2.0, 2.2)),
             (1, [(20, 20), (40, 20), (40, 40)], (-1.0, -0.8))]
    d = _write_tran(tmp_path / "r", polys, origin_q=(-100, -200))
    web = MV3.build_tran3_web(MV3._load_tran3(d))
    assert web["schema"] == "plateau_tran_web/1"
    assert web["n_triangles"] == 2 and web["n_vertices"] == 6
    P = np.frombuffer(base64.b64decode(web["positions_b64"]), dtype="<i2").reshape(-1, 3)
    exp_xy = np.array([v for _, vs, _ in polys for v in vs]) + np.array([-100, -200])
    assert np.array_equal(P[:, :2], exp_xy)
    exp_z = np.repeat(np.round(np.array([(2.0 + 2.2) / 2, (-1.0 + -0.8) / 2]) / 0.05), 3)
    assert np.array_equal(P[:, 2], exp_z.astype(np.int16))
    C = np.frombuffer(base64.b64decode(web["tri_class_b64"]), dtype="<u1")
    assert C.tolist() == [0, 1]
    # 出典が payload にも同梱される(サイドカー単体でも由来が辿れる)
    assert web["dataset_url"] == MV3.PLATEAU_DATASET_URL
    assert "2025" in web["dataset"] and "V5" in web["spec"]


def test_tran3_rejects_int16_overflow(tmp_path):
    d = _write_tran(tmp_path / "o", [(0, [(0, 0), (1000, 0), (1000, 10)], (0.0, 0.0))],
                    origin_q=(32000, 0))       # 32000 + 1000 = 33000 > 32767
    with pytest.raises(SystemExit, match="int16"):
        MV3.build_tran3_web(MV3._load_tran3(d))


# ============================================================ 5. 実データ(あれば)
@pytest.mark.skipif(not ((DATA / "tran_lod3.json").exists()
                         and (DATA / "tran_lod3.npz").exists()),
                    reason="tran_lod3 未生成(scripts/plateau_tran_extract.py)")
def test_artifact_tran3_matches_extractor_meta():
    """実データ: payload の三角形数・種別内訳が抽出器のメタ宣言と一致する。"""
    meta = json.loads((DATA / "tran_lod3.json").read_text(encoding="utf-8"))
    web = MV3.build_tran3_web(MV3._load_tran3(DATA))
    assert web["n_triangles"] == meta["n_polygons"] == 79673
    assert web["n_vertices"] == meta["n_vertices"] == web["n_triangles"] * 3
    C = np.frombuffer(base64.b64decode(web["tri_class_b64"]), dtype="<u1")
    for i, name in enumerate(MV3.TRAN3_CLASS_NAMES):
        assert int((C == i).sum()) == meta["n_polygons_by_class"][name]
    P = np.frombuffer(base64.b64decode(web["positions_b64"]), dtype="<i2").reshape(-1, 3)
    assert len(P) == web["n_vertices"]
    z = np.load(DATA / "tran_lod3.npz")
    exp_xy = z["xy"].astype(np.int64) + z["origin_q"].astype(np.int64)
    assert np.array_equal(P[:, :2].astype(np.int64), exp_xy)
    # 2025年度版 CityGML の udx/tran を出所として明記している
    assert "2025" in meta["attribution"] and "tran" in meta["attribution"]
    assert meta["lod"] == "lod3MultiSurface"


@pytest.mark.skipif(not ((DATA / "tran_lod3.json").exists()
                         and (DATA / "tran_lod3.npz").exists()),
                    reason="tran_lod3 未生成")
def test_artifact_tran3_embed_size_is_bounded():
    """埋め込みは 3MB 未満(既定 OFF フラグでしか入らないが、上限は固定しておく)。"""
    text = MV3.tran3_web_json()
    assert text is not None and len(text) < 3_000_000, f"{len(text)} bytes"


# ============================================================ 6. QuickJS 実行
_JS_STUB = r"""
var B64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
function atob(s){ s = s.replace(/=+$/, ''); var out = '', bits = 0, acc = 0;
  for (var i=0;i<s.length;i++){ acc = (acc<<6) | B64.indexOf(s.charAt(i)); bits += 6;
    if (bits >= 8){ bits -= 8; out += String.fromCharCode((acc >> bits) & 0xFF); } }
  return out; }
var THREE = {
  DoubleSide: 2,
  BufferGeometry: function(){ this.attrs = {};
    this.setAttribute = function(n, a){ this.attrs[n] = a; };
    this.computeVertexNormals = function(){}; },
  BufferAttribute: function(a, n){ this.array = a; this.itemSize = n; },
  Float32BufferAttribute: function(a, n){ this.array = a; this.itemSize = n; },
  MeshLambertMaterial: function(o){ for (var k in o) this[k] = o[k]; },
  Mesh: function(g, m){ this.geometry = g; this.material = m; this.visible = true;
    this.renderOrder = 0; this.userData = {}; }
};
var scene = { add: function(){} };
var _checked = false;
var document = { querySelector: function(){ return null; },
  getElementById: function(id){ return (id === 'lyTran3')
    ? { checked: _checked, addEventListener: function(){} } : null; } };
var console = { info: function(){}, warn: function(){} };
"""


def _quickjs_run(ctx_setup, section, expr):
    quickjs = pytest.importorskip("quickjs")
    ctx = quickjs.Context()
    ctx.eval(_JS_STUB)
    ctx.eval(ctx_setup)
    ctx.eval(section)
    return json.loads(ctx.eval(expr))


_PROBE = ("(function(){ var by = {}; for (var i=0;i<tran3Meshes.length;i++){"
          " var m = tran3Meshes[i];"
          " by[m.userData.tran3] = m.geometry.attrs.position.array.length / 9; }"
          " return JSON.stringify({ n: tran3Meshes.length, by: by,"
          " vis: tran3Meshes.length ? tran3Meshes[0].visible : null,"
          " color: tran3Meshes.length ? tran3Meshes[0].material.color : null,"
          " order: tran3Meshes.length ? tran3Meshes[0].renderOrder : null }); })()")


def test_quickjs_tran3_buckets_by_class_and_starts_hidden():
    html = MV3.build_html("t", _scene_json(), _tracks_json(), tran3_src="plateau_tran.js")
    payload = _tran3_stub([
        ("walk", (0, 0, 0), (20, 0, 0), (20, 20, 0)),
        ("road", (0, 0, 0), (20, 0, 0), (20, 20, 0)),
        ("road", (5, 5, 0), (25, 5, 0), (25, 25, 0)),
        ("island", (1, 1, 0), (3, 1, 0), (3, 3, 0)),
    ])
    got = _quickjs_run("var PLATEAU_TRAN = " + payload + ";",
                       _tran3_section(html), _PROBE)
    assert got["n"] == 3                                  # walk / road / island
    assert got["by"] == {"walk": 1, "road": 2, "island": 1}
    assert got["vis"] is False                            # 既定 OFF
    assert got["order"] == -2                             # 地形(-3)の後・建物より先


def test_quickjs_tran3_decodes_positions_like_python():
    """local-m (e,n,up) -> three (e,up,-n) の写像が建物メッシュと同一規則。"""
    html = MV3.build_html("t", _scene_json(), _tracks_json(), tran3_src="plateau_tran.js")
    payload = _tran3_stub([("road", (10, 20, 30), (40, 50, 60), (-7, -8, -9))])
    got = _quickjs_run(
        "var PLATEAU_TRAN = " + payload + ";", _tran3_section(html),
        "(function(){ var a = tran3Meshes[0].geometry.attrs.position.array, o = [];"
        " for (var i=0;i<a.length;i++) o.push(a[i]); return JSON.stringify(o); })()")
    q = 0.05
    exp = []
    for x, y, z in ((10, 20, 30), (40, 50, 60), (-7, -8, -9)):
        exp += [x * q, z * q, -y * q]
    assert np.allclose(got, exp, atol=1e-6)


def test_quickjs_tran3_survives_missing_sidecar():
    """サイドカーを消して分離版だけ開いても例外を出さない(メッシュ 0 個で退避)。"""
    html = MV3.build_html("t", _scene_json(), _tracks_json(), tran3_src="plateau_tran.js")
    sec = _tran3_section(html)
    for setup in ("", "var PLATEAU_TRAN = null;",
                  'var PLATEAU_TRAN = {"quant_scale":0.05};'):
        got = _quickjs_run(setup, sec, "JSON.stringify({n: tran3Meshes.length})")
        assert got["n"] == 0
