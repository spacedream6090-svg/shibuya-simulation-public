"""テクスチャ付き LOD2.2(松 B-1)+ 地下街 LOD4.1 塗り分け(梅 B-2)の検収。

viewer 側は単一 HTML の JS なので、
  ① 書き出し側(scripts/export_3d.py)の数値契約を直接固定し、
  ② 生成 HTML に**実際に注入されたコード**を QuickJS で実行して数値を突き合わせる
の 2 段で検証する(ブラウザ実機は範囲外)。

固定する契約:
- サイドカー plateau_tex.js は batch_shadowed(refine=REPLACE の祖先重複)を落とす。
- サイドカーは決定論(同入力→バイト同一)。
- 既定(--plateau-tex なし)は plateau_tex.js を書かず、生成 HTML も従来とバイト同一。
- 埋め込み版 viewer3d.html にテクスチャは入らない(注記のみ)。
- テクスチャ版と無テクスチャ plateau_mesh は排他(置換)。
- plateau_web.ubld4 は面種別 kind と層 layer が三角形数と整合し、層は床面 z の最近傍。
- 地下街の地表クリップ(3 頂点とも地表+0.2m 超を落とす)は新メッシュでも維持。
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


E3D = _load("export_3d_tex_t", "scripts/export_3d.py")
MV3 = _load("make_viewer3d_tex_t", "viz/make_viewer3d.py")


# ===================================================== 合成タイル(レーンA npz 契約の最小形)
def _write_tile(d: Path, tid: int, *, n_prim_tex: int = 1, n_prim_flat: int = 1,
                shadowed: tuple = (), atlas: bytes | None = b"WEBPFAKE",
                xyz_dtype=np.int16) -> dict:
    """三角形 1 枚 = batch 1 個。プリミティブ 0=テクスチャ付き / 1=無地。"""
    d.mkdir(parents=True, exist_ok=True)
    n_tri = n_prim_tex + n_prim_flat
    V = []
    T = []
    B = []
    for k in range(n_tri):
        base = 3 * k
        V += [[100 * k, 10, 20], [100 * k + 40, 10, 20], [100 * k + 40, 50, 20]]
        T.append([base, base + 1, base + 2])
        B += [k, k, k]
    xyz = np.array(V, dtype=xyz_dtype)
    uv = (np.arange(len(V) * 2, dtype=np.uint16) * 7).reshape(-1, 2)
    npz = f"tile_{tid:03d}.npz"
    np.savez(d / npz,
             xyz=xyz, origin_q=np.array([-10, -20, -30], dtype=np.int64),
             uv=uv, tri=np.array(T, dtype=np.uint32),
             batch=np.array(B, dtype=np.uint16),
             prim_tri_offsets=np.array([0, n_prim_tex, n_tri], dtype=np.int32),
             prim_textured=np.array([1, 0], dtype=np.uint8),
             batch_shadowed=np.array(shadowed, dtype=np.uint16))
    rec = {"id": tid, "npz": npz, "atlas": None, "atlas_bytes": 0,
           "n_vertices": int(len(V)), "n_triangles": n_tri,
           "batch_length": n_tri, "n_shadowed_batches": len(shadowed)}
    if atlas is not None:
        (d / f"atlas_{tid:03d}.webp").write_bytes(atlas)
        rec["atlas"] = f"atlas_{tid:03d}.webp"
        rec["atlas_bytes"] = len(atlas)
    return rec


def _write_tiles_dir(d: Path, tiles: list) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    n_t = sum(t["n_triangles"] for t in tiles)
    (d / "index.json").write_text(json.dumps({
        "schema": "plateau_tiles_lod2/1", "quant_scale": 0.05, "uv_scale": 65535,
        "n_tiles": len(tiles), "n_triangles": n_t,
        "attribution": "test", "tiles": tiles}), encoding="utf-8")
    return d


def _b64(s: str, dtype: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype=dtype)


# ----------------------------------------------------------------- 1. shadowed 除外
def test_tex_sidecar_drops_shadowed_batches(tmp_path):
    d = tmp_path / "tiles"
    t0 = _write_tile(d, 0, n_prim_tex=2, n_prim_flat=2, shadowed=(1, 3))
    _write_tiles_dir(d, [t0])
    tex = E3D.build_plateau_tex(d)
    assert tex["n_triangles_dropped_shadowed"] == 2
    assert tex["n_triangles"] == 2
    # 生き残ったのは batch 0(テクスチャ付き)と batch 2(無地)
    assert tex["n_triangles_textured"] == 1 and tex["n_triangles_flat"] == 1
    rec = tex["tiles"][0]
    # 落とした batch の頂点は詰め直しで消える(4三角形×3頂点=12 → 6)
    assert rec["n_vertices"] == 6
    F = _b64(rec["indices_b64"], "<u2").reshape(-1, 3)
    assert int(F.max()) < rec["n_vertices"]
    P = _b64(rec["positions_b64"], "<i2").reshape(-1, 3)
    # batch0(x=0..40)と batch2(x=200..240)だけが残る
    assert sorted(set(P[:, 0].tolist())) == [0, 40, 200, 240]


def test_tex_tiles_fully_shadowed_are_omitted(tmp_path):
    d = tmp_path / "tiles"
    t0 = _write_tile(d, 0, n_prim_tex=1, n_prim_flat=1, shadowed=(0, 1))
    t1 = _write_tile(d, 1, n_prim_tex=1, n_prim_flat=1)
    _write_tiles_dir(d, [t0, t1])
    tex = E3D.build_plateau_tex(d)
    assert tex["n_tiles"] == 1 and tex["tiles"][0]["id"] == 1
    assert tex["n_triangles_dropped_shadowed"] == 2


# ----------------------------------------------------------------- 2. 決定論
def test_tex_sidecar_is_deterministic(tmp_path):
    d = tmp_path / "tiles"
    tiles = [_write_tile(d, i, n_prim_tex=2, n_prim_flat=1, shadowed=(i % 3,))
             for i in range(4)]
    _write_tiles_dir(d, tiles)
    a = E3D.build_plateau_tex(d)
    b = E3D.build_plateau_tex(d)
    ja = json.dumps(a, separators=(",", ":"))
    jb = json.dumps(b, separators=(",", ":"))
    assert ja == jb
    # ファイルへ 2 回書いてもバイト同一(サイドカーは ASCII のみ)
    p1, p2 = tmp_path / "a.js", tmp_path / "b.js"
    E3D.write_plateau_tex(p1, a)
    E3D.write_plateau_tex(p2, b)
    assert p1.read_bytes() == p2.read_bytes()
    assert p1.read_text(encoding="utf-8").isascii()
    assert p1.read_text(encoding="utf-8").startswith("PLATEAU_TEX = ")


# ----------------------------------------------------------------- 3. 量子化/型/UV
def test_tex_sidecar_roundtrip_dtypes_and_uv_range(tmp_path):
    d = tmp_path / "tiles"
    t0 = _write_tile(d, 0, n_prim_tex=1, n_prim_flat=1)
    t1 = _write_tile(d, 1, n_prim_tex=1, n_prim_flat=1, xyz_dtype=np.int32)
    _write_tiles_dir(d, [t0, t1])
    tex = E3D.build_plateau_tex(d)
    assert tex["quant_scale"] == E3D.PLATEAU_QUANT and tex["uv_scale"] == 65535
    for rec, want in zip(tex["tiles"], ("int16", "int32")):
        assert rec["xyz_dtype"] == want
        assert rec["idx_dtype"] == "uint16"          # 頂点 6 個 → uint16 で足りる
        dt = "<i4" if want == "int32" else "<i2"
        P = _b64(rec["positions_b64"], dt).reshape(-1, 3)
        UV = _b64(rec["uv_b64"], "<u2").reshape(-1, 2)
        assert len(P) == len(UV) == rec["n_vertices"]
        assert int(UV.max()) <= tex["uv_scale"]
        # 量子化解除 = (xyz + origin_q) * quant。原点は npz のものがそのまま出る。
        o = np.asarray(rec["origin_q"])
        assert o.tolist() == [-10, -20, -30]
        xyz_m = (P.astype(np.int64) + o) * tex["quant_scale"]
        assert xyz_m.shape == (rec["n_vertices"], 3)
        assert np.isfinite(xyz_m).all()


def test_tex_atlas_is_data_uri_only_when_textured_survives(tmp_path):
    d = tmp_path / "tiles"
    raw = bytes(range(64))
    t0 = _write_tile(d, 0, n_prim_tex=1, n_prim_flat=1, atlas=raw)
    # テクスチャ付き batch(=0)だけを影にすると、アトラスは出さない
    t1 = _write_tile(d, 1, n_prim_tex=1, n_prim_flat=1, atlas=raw, shadowed=(0,))
    t2 = _write_tile(d, 2, n_prim_tex=1, n_prim_flat=1, atlas=None)
    _write_tiles_dir(d, [t0, t1, t2])
    tex = E3D.build_plateau_tex(d)
    by = {t["id"]: t for t in tex["tiles"]}
    assert by[0]["atlas"].startswith("data:image/webp;base64,")
    assert base64.b64decode(by[0]["atlas"].split(",", 1)[1]) == raw
    assert by[1]["atlas"] is None and by[1]["n_tex"] == 0
    assert by[2]["atlas"] is None and by[2]["n_tex"] == 0
    # アトラスが無いタイルは全三角形が無地に回る
    assert by[2]["n_flat"] == 2
    assert tex["n_atlas"] == 1


# ----------------------------------------------------------------- 4. 既定不変
def _min_run(tmp_path: Path):
    """export_run を回せる最小のラン(l1/agents/map)。"""
    import pyarrow as pa
    import pyarrow.parquet as pq
    run = tmp_path / "r1"
    run.mkdir(parents=True, exist_ok=True)
    rows = [{"step": 0, "sim_min": 420, "agent_id": 0, "kind": "arrive",
             "x": 1.0, "y": 2.0, "payload": None},
            {"step": 1, "sim_min": 430, "agent_id": 0, "kind": "arrive",
             "x": 3.0, "y": 4.0, "payload": None}]
    tbl = pa.Table.from_pylist(rows, schema=pa.schema([
        ("step", pa.int64()), ("sim_min", pa.int64()), ("agent_id", pa.int64()),
        ("kind", pa.string()), ("x", pa.float64()), ("y", pa.float64()),
        ("payload", pa.string())]))
    pq.write_table(tbl, run / "l1_events.parquet")
    (run / "agents.json").write_text(json.dumps(
        [{"id": 0, "name": "a", "occupation": "x", "visitor": False}]), encoding="utf-8")
    mp = tmp_path / "map.json"
    mp.write_text(json.dumps({"meta": {"origin_latlon": [35.0, 139.0]},
                              "buildings": [{"id": "b1", "kind": "retail", "name": "n",
                                             "footprint": [[0, 0], [8, 0], [8, 8], [0, 8]],
                                             "levels": 2, "below": 0, "cx": 4, "cy": 4}],
                              "edges": [], "railways": [], "pois": []}), encoding="utf-8")
    return run, mp


def test_default_export_writes_no_tex_sidecar_and_is_stable(tmp_path):
    run, mp = _min_run(tmp_path)
    E3D.export_run(run, mp)
    first = {p.name: p.read_bytes() for p in (run / "scene3d").iterdir()}
    assert not (run / "plateau_tex.js").exists()
    E3D.export_run(run, mp)
    second = {p.name: p.read_bytes() for p in (run / "scene3d").iterdir()}
    assert first == second
    # --plateau-tex は --plateau(plateau_dir)とセット
    with pytest.raises(SystemExit):
        E3D.export_run(run, mp, plateau_tex=True)


# ----------------------------------------------------------------- 5. ubld4(梅)
def _write_ubld4(pdir: Path, layers=(-13.0, -7.0), quant=0.05):
    """合成 LOD4.1: 三角形 4 枚(kind: floor / interior_wall / door / installation)を
    2 層に分けて置く。"""
    pdir.mkdir(parents=True, exist_ok=True)
    zs = [layers[0], layers[0], layers[1], layers[1]]
    V, T, K = [], [], []
    kinds = ["floor", "interior_wall", "door", "installation"]
    names = ["wall", "interior_wall", "floor", "ceiling", "roof", "ground",
             "closure", "door", "window", "installation", "other"]
    for k, (z, kd) in enumerate(zip(zs, kinds)):
        q = int(round(z / quant))
        base = 3 * k
        V += [[0, 0, q], [40, 0, q], [40, 40, q]]
        T.append([base, base + 1, base + 2])
        K.append(names.index(kd))
    np.savez(pdir / "ubld_lod4_mesh.npz",
             xyz=np.array(V, dtype=np.int16),
             origin_q=np.array([0, 0, 0], dtype=np.int64),
             tri=np.array(T, dtype=np.uint32),
             tri_kind=np.array(K, dtype=np.uint8),
             kind_names=np.array(names))
    (pdir / "ubld_lod4.json").write_text(json.dumps({
        "schema": "plateau_ubld_lod4/1",
        "params": {"quant_scale": quant},
        "mesh": {"kind_names": names},
        "layers": [{"layer": i, "z": z} for i, z in enumerate(layers)],
    }), encoding="utf-8")


def test_ubld4_web_kind_and_layer_alignment(tmp_path):
    pdir = tmp_path / "p"
    _write_ubld4(pdir)
    u = E3D._load_ubld4(pdir)
    web = E3D.build_ubld4_web(u)
    assert web["n_triangles"] == 4 and web["n_vertices"] == 12
    assert web["quant_scale"] == E3D.PLATEAU_QUANT
    assert web["kind_names"][7] == "door" and web["kind_names"][9] == "installation"
    kind = _b64(web["tri_kind_b64"], "<u1")
    lay = _b64(web["tri_layer_b64"], "<u1")
    assert len(kind) == len(lay) == web["n_triangles"]
    # 層は床面 z ピークの最近傍(深い方=0)
    assert lay.tolist() == [0, 0, 1, 1]
    assert [ly["n_triangles"] for ly in web["layers"]] == [2, 2]
    assert sum(ly["n_triangles"] for ly in web["layers"]) == web["n_triangles"]
    # kind は npz の tri_kind そのまま(名前索引が意味を持つ)
    names = web["kind_names"]
    assert [names[i] for i in kind] == ["floor", "interior_wall", "door", "installation"]
    # 位置は絶対量子化 int16(origin_q を畳み込む)
    P = _b64(web["positions_b64"], "<i2").reshape(-1, 3)
    assert P[:, 2].min() * web["quant_scale"] == pytest.approx(-13.0)
    F = _b64(web["indices_b64"], "<u4").reshape(-1, 3)
    assert int(F.max()) < web["n_vertices"]


def test_ubld4_absent_means_no_key(tmp_path):
    pdir = tmp_path / "p_empty"
    pdir.mkdir()
    assert E3D._load_ubld4(pdir) is None


# ----------------------------------------------------------------- 6. HTML 注入
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


def test_viewer_html_byte_identical_without_new_data():
    """新引数を渡さない/None を渡すのは、どちらも従来の生成物と完全一致。"""
    a = MV3.build_html("t", _scene_json(), _tracks_json())
    b = MV3.build_html("t", _scene_json(), _tracks_json(),
                       has_ubld4=False, plateau_tex_src=None, tex_note=False)
    assert a == b
    assert "PLATEAU_TEX" not in a and "ubld4" not in a and "lyTex" not in a


def test_viewer_tex_replaces_plateau_mesh_and_boxes():
    html = MV3.build_html("t", _scene_json(), _tracks_json(),
                          plateau_src="plateau_mesh.js", plateau_tex_src="plateau_tex.js")
    assert '<script src="plateau_tex.js"></script>' in html
    assert 'id="lyTex"' in html
    # 排他: テクスチャがあれば無テクスチャ PLATEAU メッシュは作らない
    assert "if(!PLATEAU_DATA || PLATEAU_TEX_DATA) return;" in html
    # 押出し箱はテクスチャ OFF のときだけ出す
    assert "m.visible = L3('lyBld') && !TEX_ON" in html
    # 照合済み skip は tex 時に無効化(未照合の箱が実形状に刺さるのを防ぐ)
    assert ("const PLATEAU_SKIP = new Set((PLATEAU_DATA && !PLATEAU_TEX_DATA)"
            " ? PLATEAU_DATA.matched_ids : []);") in html


def test_viewer_embedded_gets_note_not_texture():
    html = MV3.build_html("t", _scene_json(), _tracks_json(),
                          plateau_json=json.dumps({"quant_scale": 0.05, "matched_ids": [],
                                                   "n_vertices": 0, "n_triangles": 0,
                                                   "positions_b64": "", "indices_b64": "",
                                                   "colors_b64": ""}),
                          tex_note=True)
    assert "PLATEAU_TEX" not in html
    assert "viewer3d_lite.html(+plateau_tex.js)で" in html


def test_viewer_ubld4_replaces_legacy_extras_box():
    html = MV3.build_html("t", _scene_json(), _tracks_json(),
                          plateau_src="plateau_mesh.js", has_extras=True, has_ubld4=True)
    assert "const u = null;" in html                    # 旧 extras.ubld の箱は描かない
    assert "meshOf(ex.ubld" not in html
    assert 'id="ub4Chips"' in html                      # 層チップ
    assert "ubld4Meshes" in html
    # 地表クリップの規則は維持
    assert "if(above === 3){ cut++; continue; }" in html


def test_generated_html_variants_parse_as_javascript():
    esprima = pytest.importorskip("esprima")
    pw = json.dumps({"quant_scale": 0.05, "matched_ids": [], "n_vertices": 0,
                     "n_triangles": 0, "positions_b64": "", "indices_b64": "",
                     "colors_b64": ""})
    variants = [
        {},
        {"plateau_json": pw},
        {"plateau_json": pw, "has_extras": True, "has_ubld4": True, "tex_note": True},
        {"plateau_src": "plateau_mesh.js", "plateau_tex_src": "plateau_tex.js"},
        {"plateau_src": "plateau_mesh.js", "plateau_tex_src": "plateau_tex.js",
         "has_extras": True, "has_ubld4": True},
    ]
    for kw in variants:
        html = MV3.build_html("t", _scene_json(), _tracks_json(), **kw)
        body = re.findall(r"<script>(.*?)</script>", html, re.S)[-1]
        esprima.parseScript(body)                       # 構文エラーなら例外


# ----------------------------------------------------------------- 7. QuickJS 実行
_JS_STUB = r"""
var B64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
function atob(s){ s = s.replace(/=+$/, ''); var out = '', bits = 0, acc = 0;
  for (var i=0;i<s.length;i++){ acc = (acc<<6) | B64.indexOf(s.charAt(i)); bits += 6;
    if (bits >= 8){ bits -= 8; out += String.fromCharCode((acc >> bits) & 0xFF); } }
  return out; }
var THREE = {
  DoubleSide: 2, sRGBEncoding: 3,
  BufferGeometry: function(){ this.attrs = {}; this.groups = [];
    this.setAttribute = function(n, a){ this.attrs[n] = a; };
    this.setIndex = function(a){ this.idx = a; };
    this.computeVertexNormals = function(){};
    this.addGroup = function(s, c, m){ this.groups.push([s, c, m]); }; },
  Float32BufferAttribute: function(a, n){ this.array = a; this.itemSize = n; },
  BufferAttribute: function(a, n){ this.array = a; this.itemSize = n; },
  Texture: function(img){ this.image = img; this.needsUpdate = false; },
  MeshLambertMaterial: function(o){ for (var k in o) this[k] = o[k]; },
  Mesh: function(g, m){ this.geometry = g; this.material = m; this.visible = true;
    this.renderOrder = 0; }
};
var NEUTRAL_BLD = 0xd0d4da;
var window = {};
var scene = { add: function(){} };
var document = { querySelector: function(){ return null; },
                 getElementById: function(){ return null; } };
function Image(){ this.src = ''; }
var _cut = 0;
var console = { info: function(a, n){ _cut = n; }, warn: function(){} };
"""


def _section(html: str, head: str) -> str:
    """生成 HTML から注入ブロックを切り出す(実際に出荷される JS をそのまま実行するため)。"""
    m = re.search(r"(// -+ " + head + r"[\s\S]*?)\n// -+ (テクスチャ付き|道路)", html)
    assert m, f"section not found: {head}"
    return m.group(1)


def test_quickjs_tex_decode_matches_python(tmp_path):
    quickjs = pytest.importorskip("quickjs")
    d = tmp_path / "tiles"
    t0 = _write_tile(d, 0, n_prim_tex=2, n_prim_flat=1)
    _write_tiles_dir(d, [t0])
    tex = E3D.build_plateau_tex(d)
    html = MV3.build_html("t", _scene_json(), _tracks_json(),
                          plateau_src="plateau_mesh.js", plateau_tex_src="plateau_tex.js")
    ctx = quickjs.Context()
    ctx.eval(_JS_STUB)
    ctx.eval("var PLATEAU_TEX_DATA = " + json.dumps(tex) + ";")
    ctx.eval(_section(html, r"テクスチャ付き PLATEAU LOD2\.2"))
    got = json.loads(ctx.eval(
        "(function(){ var g = texMeshes[0].geometry;"
        " function all(a){ var o=[]; for(var i=0;i<a.length;i++) o.push(a[i]); return o; }"
        " return JSON.stringify({ n: texMeshes.length, pos: all(g.attrs.position.array),"
        " uv: all(g.attrs.uv.array), idx: all(g.idx.array), groups: g.groups,"
        " mats: (texMeshes[0].material.length !== undefined)"
        " ? texMeshes[0].material.length : 1 }); })()"))
    rec = tex["tiles"][0]
    P = _b64(rec["positions_b64"], "<i2").reshape(-1, 3)
    o = np.asarray(rec["origin_q"])
    # local-m (e,n,up) -> three (e,up,-n)
    exp = np.column_stack([(P[:, 0] + o[0]) * tex["quant_scale"],
                           (P[:, 2] + o[2]) * tex["quant_scale"],
                           -(P[:, 1] + o[1]) * tex["quant_scale"]])
    assert np.allclose(np.array(got["pos"]).reshape(-1, 3), exp, atol=1e-4)
    UV = _b64(rec["uv_b64"], "<u2").astype(np.float64) / tex["uv_scale"]
    assert np.allclose(np.array(got["uv"]), UV, atol=1e-9)
    assert got["idx"] == _b64(rec["indices_b64"], "<u2").tolist()
    # テクスチャ群 → 無地群 の 2 グループ(索引数)
    assert got["groups"] == [[0, rec["n_tex"] * 3, 0],
                             [rec["n_tex"] * 3, rec["n_flat"] * 3, 1]]
    assert got["mats"] == 2


def test_quickjs_ubld4_layers_kinds_and_ground_clip(tmp_path):
    quickjs = pytest.importorskip("quickjs")
    pdir = tmp_path / "p"
    _write_ubld4(pdir)
    web = E3D.build_ubld4_web(E3D._load_ubld4(pdir))
    html = MV3.build_html("t", _scene_json(), _tracks_json(),
                          plateau_src="plateau_mesh.js", has_extras=True, has_ubld4=True)
    section = _section(html, r"地下街 LOD4\.1")

    def run(terrain_js: str):
        ctx = quickjs.Context()
        ctx.eval(_JS_STUB)
        ctx.eval(terrain_js)
        ctx.eval("var PLATEAU_DATA = { ubld4: " + json.dumps(web) + " };")
        ctx.eval(section)
        return json.loads(ctx.eval(
            "(function(){ var by = {}, mats = 0;"
            " for (var i=0;i<ubld4Meshes.length;i++){ var r = ubld4Meshes[i];"
            "   var n = r.mesh.geometry.attrs.position.array.length / 9;"
            "   by[r.layer] = (by[r.layer] || 0) + n; mats++; }"
            " return JSON.stringify({ meshes: mats, byLayer: by, cut: _cut }); })()"))

    # 地形なし=クリップ無し。層ごとの三角形数が書き出し側の宣言と一致する。
    flat = run("var TERRAIN = null; function groundAt(x, y){ return 0; }")
    assert flat["byLayer"] == {str(ly["layer"]): ly["n_triangles"] for ly in web["layers"]}
    assert flat["meshes"] == 4          # 4 種別 × 各 1 層
    # 地表 -8m: 浅い層(z=-7)の 2 枚だけが「3 頂点とも地表+0.2m 超」でクリップされる
    cut = run("var TERRAIN = {}; function groundAt(x, y){ return -8.0; }")
    assert cut["cut"] == 2
    assert cut["byLayer"] == {"0": 2}


# ===================================================== 実データ(あれば)の数値検証
@pytest.mark.skipif(not (DATA / "tiles_lod2" / "index.json").exists(),
                    reason="tiles_lod2 未生成(tools/tiles3d_extract.py)")
def test_artifact_tex_sidecar_accounting():
    """実タイル 148 枚: 影落とし除外の会計と gml_id の一意化(重複描画が消える証拠)。"""
    idx = json.loads((DATA / "tiles_lod2" / "index.json").read_text(encoding="utf-8"))
    tex = E3D.build_plateau_tex(DATA / "tiles_lod2")
    assert tex["n_triangles"] + tex["n_triangles_dropped_shadowed"] == idx["n_triangles"]
    ratio = tex["n_triangles_dropped_shadowed"] / idx["n_triangles"]
    assert 0.15 < ratio < 0.17, f"影落とし除外率 {ratio:.4f}(レーンA実測 15.70%)"
    attrs_p = DATA / "tiles_batch_attrs.json"
    if attrs_p.exists():
        attrs = json.loads(attrs_p.read_text(encoding="utf-8"))["tiles"]
        kept = [g for t in idx["tiles"]
                for b, g in enumerate(attrs[t["npz"]]["gml_id"])
                if b not in set(attrs[t["npz"]]["_shadowed_batches"])]
        assert len(kept) == len(set(kept)) == idx["n_unique_gml_id"]
    for rec in tex["tiles"]:
        UV = _b64(rec["uv_b64"], "<u2")
        assert int(UV.max()) <= tex["uv_scale"]
        it = "<u4" if rec["idx_dtype"] == "uint32" else "<u2"
        F = _b64(rec["indices_b64"], it)
        assert int(F.max()) < rec["n_vertices"]


@pytest.mark.skipif(not ((DATA / "ubld_lod4_mesh.npz").exists()
                         and (DATA / "terrain.npz").exists()),
                    reason="ubld_lod4_mesh / terrain 未生成")
def test_artifact_ubld4_z_datum_is_sound_vs_legacy_extras():
    """z 基準の検証(修正バッチが旧 extras で見つけた「頂点 45.4% が地表上」の追試)。

    新 LOD4.1 メッシュでは地表より上の頂点は数 % に落ちる。ゼロではない(地上への階段の
    天端)ため、ビューア側の地表クリップ規則は維持する — その根拠をここで固定する。"""
    meta = json.loads((DATA / "terrain.json").read_text(encoding="utf-8"))
    tz = np.load(DATA / "terrain.npz")
    H = np.asarray(tz["heights"] if "heights" in tz else tz[tz.files[0]], dtype=np.float64)

    def gat(x, y):
        nx, ny, c = int(meta["nx"]), int(meta["ny"]), float(meta["cell_m"])
        gx = (x - meta["x0"]) / c
        gy = (y - meta["y0"]) / c
        i0 = np.clip(np.floor(gx).astype(int), 0, nx - 2)
        j0 = np.clip(np.floor(gy).astype(int), 0, ny - 2)
        fx = np.clip(gx - i0, 0, 1)
        fy = np.clip(gy - j0, 0, 1)
        a = H[j0, i0] + (H[j0, i0 + 1] - H[j0, i0]) * fx
        b = H[j0 + 1, i0] + (H[j0 + 1, i0 + 1] - H[j0 + 1, i0]) * fx
        return a + (b - a) * fy

    u = E3D._load_ubld4(DATA)
    V = (u["xyz"].astype(np.int64) + u["origin_q"]) * u["quant"]
    frac_new = float((V[:, 2] > gat(V[:, 0], V[:, 1])).mean())
    assert frac_new < 0.05, f"新メッシュの地表上頂点 {frac_new:.4f}"
    ex = np.load(DATA / "extras.npz")
    Vo = np.asarray(ex["ubld_V"], dtype=np.float64)
    frac_old = float((Vo[:, 2] > gat(Vo[:, 0], Vo[:, 1])).mean())
    assert frac_old > 0.40, f"旧 extras の地表上頂点 {frac_old:.4f}"
    assert frac_new < frac_old / 10.0
    # 残る地上突出は最上層のみ(= 地上への階段)。深い層は 1 枚も地表を越えない。
    web = E3D.build_ubld4_web(u)
    F = _b64(web["indices_b64"], "<u4").reshape(-1, 3)
    lay = _b64(web["tri_layer_b64"], "<u1")
    P = _b64(web["positions_b64"], "<i2").reshape(-1, 3).astype(np.float64) * web["quant_scale"]
    zz = P[F][:, :, 2]
    gg = np.stack([gat(P[F[:, k], 0], P[F[:, k], 1]) for k in range(3)], 1)
    above = (zz > gg + 0.2).all(axis=1)
    top = int(max(ly["layer"] for ly in web["layers"]))
    assert set(np.unique(lay[above]).tolist()) <= {top}
