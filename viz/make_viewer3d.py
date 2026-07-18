"""Web3D ビューア生成(バッチE / 3D 可視化)。

使い方:  python viz/make_viewer3d.py runs/<name>
生成物:  runs/<name>/viewer3d.html
  自己完結の単一 HTML。three.js(r128, MIT)本体・OrbitControls・シーンデータを埋め込み。
  ブラウザで開けば即グリグリ(OrbitControls)+ 再生 + 昼夜 + クリックで人物情報。

データは runs/<name>/scene3d/{scene.json,tracks.json} を読む。無ければ export_3d を実行して生成。
移動補間は 2D ビューア(viz/make_viewer.py)の posAt/alongPath を移植し、同じ滑らかさを再現。
座標系: scene/tracks は local-m Z-up。three.js では (east, up, -north) に写像(Y-up)。
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR = REPO_ROOT / "viz" / "vendor"


def _load_export3d():
    spec = importlib.util.spec_from_file_location(
        "export_3d", REPO_ROOT / "scripts" / "export_3d.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ensure_scene(run_dir: Path) -> tuple[str, str, str | None]:
    scene_dir = run_dir / "scene3d"
    scene_p = scene_dir / "scene.json"
    tracks_p = scene_dir / "tracks.json"
    if not (scene_p.exists() and tracks_p.exists()):
        _load_export3d().export_run(run_dir)
    # PLATEAU 実形状(export_3d --plateau の成果物)。無ければ None=従来とバイト同一。
    pw_p = scene_dir / "plateau_web.json"
    plateau = pw_p.read_text(encoding="utf-8") if pw_p.exists() else None
    return (scene_p.read_text(encoding="utf-8"),
            tracks_p.read_text(encoding="utf-8"), plateau)


def _read_vendor() -> tuple[str, str, str]:
    three = VENDOR / "three.min.js"
    orbit = VENDOR / "OrbitControls.js"
    lic = VENDOR / "LICENSE"
    missing = [p.name for p in (three, orbit, lic) if not p.exists()]
    if missing:
        raise SystemExit(
            f"[make_viewer3d] vendor 未取得: {missing}. "
            "viz/vendor/ に three.min.js / OrbitControls.js / LICENSE が必要です。")
    return (three.read_text(encoding="utf-8"),
            orbit.read_text(encoding="utf-8"),
            lic.read_text(encoding="utf-8"))


def _json_for_script(text: str) -> str:
    # <script type=application/json> 内に安全に埋め込む
    return text.replace("</", "<\\/")


def build_html(run_name: str, scene_json: str, tracks_json: str,
               plateau_json: str | None = None,
               plateau_src: str | None = None) -> str:
    three_js, orbit_js, lic = _read_vendor()
    html = _TEMPLATE
    html = html.replace("__RUN_NAME__", run_name)
    html = html.replace("__THREE_LICENSE__", lic.strip())
    html = html.replace("__THREE_JS__", three_js)
    html = html.replace("__ORBIT_JS__", orbit_js)
    html = html.replace("__SCENE_JSON__", _json_for_script(scene_json))
    html = html.replace("__TRACKS_JSON__", _json_for_script(tracks_json))
    if plateau_json is not None or plateau_src is not None:
        html = _inject_plateau(html, plateau_json, plateau_src)
    return html


def _replace_once(html: str, old: str, new: str, what: str) -> str:
    """テンプレート内アンカーの一意置換。ズレたら黙って壊れず即エラー。"""
    n = html.count(old)
    if n != 1:
        raise SystemExit(f"[make_viewer3d] PLATEAU 注入アンカー '{what}' が {n} 箇所"
                         "(期待=1)。テンプレート変更時は _inject_plateau を追随させる。")
    return html.replace(old, new)


def _inject_plateau(html: str, plateau_json: str | None,
                    plateau_src: str | None) -> str:
    """PLATEAU 実形状メッシュの描画を後付け注入する(データ無し時は呼ばれない=
    既定出力は従来とバイト同一)。plateau_src 指定時は埋め込みの代わりに
    <script src> サイドカー参照(file:// でも動く分離版=JSONP 方式)。"""
    if plateau_src is not None:
        data_tag = f'<script src="{plateau_src}"></script>'
    else:
        data_tag = ('<script type="application/json" id="plateau-data">'
                    + _json_for_script(plateau_json) + "</script>")
    # 1) データブロック(scene-data の直前)
    anchor_data = '<script type="application/json" id="scene-data">'
    html = _replace_once(html, anchor_data, data_tag + "\n" + anchor_data, "data")
    # 2) PLATEAU_DATA / PLATEAU_SKIP 宣言(buildBuildings の直前)
    anchor_decl = "(function buildBuildings(){"
    html = _replace_once(html, anchor_decl, _PLATEAU_DECL + anchor_decl, "decl")
    # 3) 照合済み建物は押出しをスキップ
    anchor_skip = "for(const b of SCENE.buildings){\n    const fp = b.footprint;"
    html = _replace_once(
        html, anchor_skip,
        "for(const b of SCENE.buildings){\n"
        "    if(PLATEAU_SKIP.has(b.id)) continue;\n"
        "    const fp = b.footprint;", "skip")
    # 4) 実形状メッシュの構築(道路セクションの直前)
    anchor_build = "// ---------- 道路(全ポリラインを 1 本の LineSegments に統合)"
    html = _replace_once(html, anchor_build, _PLATEAU_BUILD + anchor_build, "build")
    return html


_PLATEAU_DECL = r"""const PLATEAU_DATA = (()=>{ try {
  const el = document.getElementById('plateau-data');
  if(el) return JSON.parse(el.textContent);
  if(typeof PLATEAU_MESH !== 'undefined') return PLATEAU_MESH;
} catch(e) { console.warn('PLATEAU data parse failed', e); } return null; })();
const PLATEAU_SKIP = new Set(PLATEAU_DATA ? PLATEAU_DATA.matched_ids : []);
"""

_PLATEAU_BUILD = r"""// ---------- PLATEAU 実形状建物(照合済み建物の実測メッシュ・出典は data.attribution)
(function buildPlateau(){
  if(!PLATEAU_DATA) return;
  const b64 = s => { const bin = atob(s); const u = new Uint8Array(bin.length);
    for(let i=0;i<bin.length;i++) u[i] = bin.charCodeAt(i); return u; };
  const q = PLATEAU_DATA.quant_scale;
  const pi = new Int16Array(b64(PLATEAU_DATA.positions_b64).buffer);
  const pos = new Float32Array(pi.length);
  for(let i=0;i<pi.length;i+=3){                 // local-m (e,n,up) -> three (e,up,-n)
    pos[i]   =  pi[i]   * q;
    pos[i+1] =  pi[i+2] * q;
    pos[i+2] = -pi[i+1] * q;
  }
  const idx = new Uint32Array(b64(PLATEAU_DATA.indices_b64).buffer);
  const cu = b64(PLATEAU_DATA.colors_b64);
  const col = new Float32Array(cu.length);
  for(let i=0;i<cu.length;i++) col[i] = cu[i] / 255;
  const bg = new THREE.BufferGeometry();
  bg.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  bg.setAttribute('color', new THREE.BufferAttribute(col, 3));
  bg.setIndex(new THREE.BufferAttribute(idx, 1));
  bg.computeVertexNormals();
  const mat = new THREE.MeshLambertMaterial({ vertexColors:true,
    transparent:true, opacity:1.0 });
  buildingMats.push(mat);
  const mesh = new THREE.Mesh(bg, mat);
  buildingMeshes.push(mesh); scene.add(mesh);
  try { const s = document.querySelector('#hud .sub');
    if(s && PLATEAU_DATA.attribution) s.textContent += ' / ' + PLATEAU_DATA.attribution;
  } catch(e){}
})();

"""


def _strip_traffic(tracks_json: str) -> str:
    # 長期ラン(例 100日=14400step)は背景交通の軌跡だけで数百MBになりブラウザで開けない。
    # --no-traffic で traffic を空にする(人物・建物は従来どおり)。JS は traffic[s] 不在で車0台。
    tracks = json.loads(tracks_json)
    tracks["traffic"] = []
    return json.dumps(tracks, ensure_ascii=False)


def main(argv: list) -> int:
    args = [a for a in argv if not a.startswith("--")]
    flags = {a for a in argv if a.startswith("--")}
    if not args:
        print(__doc__)
        return 1
    run_dir = Path(args[0]).resolve()
    scene_json, tracks_json, plateau_json = _ensure_scene(run_dir)
    if "--no-traffic" in flags:
        tracks_json = _strip_traffic(tracks_json)
    html = build_html(run_dir.name, scene_json, tracks_json,
                      plateau_json=plateau_json)
    out = run_dir / "viewer3d.html"
    out.write_text(html, encoding="utf-8")
    mb = out.stat().st_size / 1024 / 1024
    print(f"  {out}  ({mb:.2f} MB)")
    if plateau_json is not None:
        if mb > 80:
            print(f"  [warn] 埋め込み版が {mb:.1f} MB > 80MB ゲート。"
                  " LOD 簡略化か分離版の利用を検討。")
        # 分離版: 軽量 HTML + サイドカー(同フォルダに置けば file:// で動く)
        side = run_dir / "plateau_mesh.js"
        side.write_text("PLATEAU_MESH = " + plateau_json + ";",
                        encoding="utf-8")
        lite = build_html(run_dir.name, scene_json, tracks_json,
                          plateau_src="plateau_mesh.js")
        lite_p = run_dir / "viewer3d_lite.html"
        lite_p.write_text(lite, encoding="utf-8")
        print(f"  {lite_p}  ({lite_p.stat().st_size/1024/1024:.2f} MB)"
              f"  + {side.name}  ({side.stat().st_size/1024/1024:.2f} MB)")
    return 0


# ============================================================ HTML テンプレート
_TEMPLATE = r"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>渋谷シミュ 3D — __RUN_NAME__</title>
<!-- three.js r128 (MIT). ライセンス全文:
__THREE_LICENSE__
-->
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  html,body { width:100%; height:100%; overflow:hidden; background:#0a0e14;
    font-family:-apple-system,"Segoe UI","Hiragino Kaku Gothic ProN",Meiryo,sans-serif; }
  #app { position:fixed; inset:0; }
  canvas { display:block; }
  .panel { position:fixed; background:rgba(18,22,30,.82); color:#e6e9ee;
    border:1px solid rgba(255,255,255,.09); border-radius:10px; backdrop-filter:blur(6px);
    padding:10px 12px; font-size:13px; }
  #hud { top:12px; left:12px; max-width:280px; }
  #hud h1 { font-size:14px; font-weight:600; margin-bottom:4px; letter-spacing:.02em; }
  #hud .sub { color:#9aa4b2; font-size:11px; margin-bottom:8px; }
  #clock { font-variant-numeric:tabular-nums; font-size:22px; font-weight:600; margin:2px 0 6px; }
  #ctrl { bottom:12px; left:50%; transform:translateX(-50%); display:flex; gap:10px;
    align-items:center; flex-wrap:wrap; max-width:94vw; }
  #ctrl button, .seg button { background:rgba(255,255,255,.07); color:#e6e9ee;
    border:1px solid rgba(255,255,255,.12); border-radius:7px; padding:6px 12px;
    cursor:pointer; font-size:13px; }
  #ctrl button:hover, .seg button:hover { background:rgba(255,255,255,.14); }
  .seg button.on { background:#3b82f6; border-color:#3b82f6; color:#fff; }
  #timeline { width:min(420px,60vw); accent-color:#3b82f6; }
  #rightStack { position:fixed; top:12px; right:12px; display:flex; flex-direction:column;
    gap:10px; align-items:stretch; width:222px; max-height:calc(100vh - 24px); overflow-y:auto; z-index:5; }
  #layers { position:relative; }
  #layers .hdr { font-size:10.5px; letter-spacing:.12em; text-transform:uppercase; color:#9aa4b2;
    cursor:pointer; margin-bottom:8px; user-select:none; }
  #layers label.chk { display:flex; align-items:center; gap:7px; padding:2px 0; cursor:pointer; }
  #layers input[type=checkbox]{ accent-color:#3b82f6; }
  #layers .op { display:flex; align-items:center; gap:6px; margin:3px 0 8px 22px; color:#9aa4b2; font-size:11px; }
  #layers .op input[type=range]{ width:88px; accent-color:#3b82f6; }
  #layers .op span { min-width:32px; text-align:right; font-variant-numeric:tabular-nums; }
  #legend { position:relative; font-size:12px; }
  #legend .row { display:flex; align-items:center; gap:7px; margin:3px 0; }
  #legend .sw { width:12px; height:12px; border-radius:3px; display:inline-block; }
  #osmAttr { position:fixed; left:10px; bottom:8px; font-size:10px; color:#8a93a3; opacity:.78;
    text-shadow:0 1px 2px rgba(0,0,0,.6); z-index:3; pointer-events:none; }
  #info { right:12px; bottom:12px; min-width:200px; max-width:280px; display:none; }
  #info h2 { font-size:14px; margin-bottom:6px; }
  #info .k { color:#9aa4b2; }
  label.chk { display:inline-flex; align-items:center; gap:5px; font-size:12px; color:#c7cdd6; }
  select { background:rgba(255,255,255,.07); color:#e6e9ee; border:1px solid rgba(255,255,255,.12);
    border-radius:6px; padding:4px 6px; font-size:12px; }
  .hint { color:#7c8698; font-size:11px; margin-top:6px; }
</style>
</head>
<body>
<div id="app"></div>

<div id="hud" class="panel">
  <h1>渋谷シミュレーション 3D</h1>
  <div class="sub">run: __RUN_NAME__ ・ OSM 押出し都市</div>
  <div id="clock">--:--</div>
  <div class="seg" id="colorSeg">
    色分け:
    <button data-c="visitor" class="on">来訪/居住</button>
    <button data-c="occ">職業</button>
  </div>
  <div style="margin-top:8px; display:flex; flex-direction:column; gap:4px;">
    <label class="chk"><input type="checkbox" id="xray"> 建物を半透明(屋内の人を見る)</label>
  </div>
  <div class="hint">ドラッグ=回転 / ホイール=ズーム / 右ドラッグ=平行移動<br>人物クリックで詳細</div>
</div>

<div id="rightStack">
  <div id="layers" class="panel">
    <div class="hdr" id="lyHdr">レイヤー ▾</div>
    <div id="lyBody">
      <label class="chk"><input type="checkbox" id="lyOsm" checked> OSM地図(地面)</label>
      <div class="op">不透明度 <input type="range" id="osmOp" min="0" max="1" step="0.05" value="0.9"><span id="osmOpV">90%</span></div>
      <label class="chk"><input type="checkbox" id="lyBld" checked> 建物</label>
      <label class="chk"><input type="checkbox" id="lyAgent" checked> エージェント</label>
      <label class="chk"><input type="checkbox" id="lyCars" checked> 車</label>
      <label class="chk"><input type="checkbox" id="lyRoad" checked> 道路</label>
      <label class="chk"><input type="checkbox" id="lyRail" checked> 電車・線路</label>
      <label class="chk"><input type="checkbox" id="lyUnder" checked> 地下(地下鉄)</label>
      <label class="chk"><input type="checkbox" id="lyLabels" checked> ラベル(建物名)</label>
      <label class="chk"><input type="checkbox" id="lyDayNight" checked> 昼夜ライティング</label>
    </div>
  </div>
  <div id="legend" class="panel"></div>
</div>
<div id="osmAttr">© OpenStreetMap contributors</div>

<div id="info" class="panel">
  <h2 id="info-name">—</h2>
  <div><span class="k">属性:</span> <span id="info-role">—</span></div>
  <div><span class="k">現在:</span> <span id="info-loc">—</span></div>
  <button id="info-close" style="margin-top:8px; width:100%;">閉じる</button>
</div>

<div id="ctrl" class="panel">
  <button id="play">▶︎</button>
  <input type="range" id="timeline" min="0" max="100" value="0" step="0.01">
  <select id="speed">
    <option value="0.5">0.5×</option>
    <option value="1" selected>1×</option>
    <option value="2">2×</option>
    <option value="4">4×</option>
    <option value="8">8×</option>
  </select>
  <button id="reset">視点リセット</button>
</div>

<script type="application/json" id="scene-data">__SCENE_JSON__</script>
<script type="application/json" id="tracks-data">__TRACKS_JSON__</script>
<script>__THREE_JS__</script>
<script>__ORBIT_JS__</script>
<script>
"use strict";
const SCENE  = JSON.parse(document.getElementById('scene-data').textContent);
const TRACKS = JSON.parse(document.getElementById('tracks-data').textContent);
const FH = (SCENE.meta && SCENE.meta.floor_height) || 3.5;
const NS = TRACKS.meta.nSteps;
const STEP_MIN = TRACKS.meta.step_minutes || 10;

const KIND_COLOR = {
  station:0xE86A33, retail:0x3BA89C, office:0x5B8FD6, residential:0x7AB86A,
  hotel:0xC96AB0, public:0x9B7FD4, generic:0xB8BEC7, "house?":0xA99E92 };

// ---------- 座標写像: world (east=x, north=y, up=z) -> three.js (x, up, -north)
function V(x, y, z){ return new THREE.Vector3(x, z, -y); }

// ---------- Three 基本
const app = document.getElementById('app');
const renderer = new THREE.WebGLRenderer({ antialias:true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = false;
app.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0e14);
scene.fog = new THREE.Fog(0x0a0e14, 900, 2600);

const camera = new THREE.PerspectiveCamera(52, window.innerWidth/window.innerHeight, 1, 8000);
const CAM0 = new THREE.Vector3(360, 420, 520);
camera.position.copy(CAM0);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.target.set(0, 20, 0);
controls.maxPolarAngle = Math.PI * 0.56;   // 地平線少し下まで(半透明地面越しに地下鉄)
controls.maxDistance = 3000;
controls.minDistance = 30;

// ---------- ライト(昼夜で更新)
const sun = new THREE.DirectionalLight(0xffffff, 1.0);
scene.add(sun);
scene.add(sun.target);
const hemi = new THREE.HemisphereLight(0xbfd4ff, 0x2a2f3a, 0.6);
scene.add(hemi);
const ambient = new THREE.AmbientLight(0xffffff, 0.25);
scene.add(ambient);

// ---------- 地面
let gridHelper = null;
{
  const g = new THREE.PlaneGeometry(6000, 6000);
  g.rotateX(-Math.PI/2);
  const m = new THREE.MeshLambertMaterial({ color:0x141922,
    transparent:true, opacity:0.55, depthWrite:false });
  const ground = new THREE.Mesh(g, m);
  ground.position.y = -0.05;
  ground.renderOrder = -2;
  scene.add(ground);
  gridHelper = new THREE.GridHelper(2400, 48, 0x2a3242, 0x1c2330);
  scene.add(gridHelper);
}

// ---------- OSM 地図の地面(実行時にラスタタイルを取得→キャンバス合成→1テクスチャ)
// 2D ビューア(viz/make_viewer.py)と同じ mercator 近似・同じタイル配信元を用い、建物と整合させる。
const OSM = { mesh:null, opacity:0.9, loaded:false, load:null, ensureLoaded:null };
(function buildOsmGround(){
  const origin = SCENE.meta && SCENE.meta.origin_latlon;
  if(!origin || origin.length < 2) return;            // 地理原点なし → OSM 地面は作らない
  const LAT0 = origin[0], LON0 = origin[1];
  // 1) シーンの世界範囲(建物 footprint + 道路)。線路は郊外の遠方駅まで伸びて
  //    範囲を数 km に膨張させ地図を粗くするため、2D ビューアの視野基準(道路網)に合わせ除外。
  let x0=Infinity, x1=-Infinity, y0=Infinity, y1=-Infinity;
  const acc = (x,y)=>{ if(x<x0)x0=x; if(x>x1)x1=x; if(y<y0)y0=y; if(y>y1)y1=y; };
  for(const b of SCENE.buildings){ if(b.footprint) for(const p of b.footprint) acc(p[0],p[1]); }
  for(const r of SCENE.roads){ for(const p of r.g) acc(p[0],p[1]); }
  if(!isFinite(x0)) return;
  const mgx=(x1-x0)*0.06+50, mgy=(y1-y0)*0.06+50;     // 余白
  x0-=mgx; x1+=mgx; y0-=mgy; y1+=mgy;
  const extent = Math.max(x1-x0, y1-y0);
  // 2) ズーム: 1辺 6〜8 枚を狙って選ぶ(mercator の m/px は緯度依存)
  const mppAt = z => 156543.03392*Math.cos(LAT0*Math.PI/180)/Math.pow(2,z);
  let z = 14;
  for(let zz=18; zz>=14; zz--){ if(extent/(256*mppAt(zz)) <= 8){ z=zz; break; } }
  z = Math.max(14, Math.min(18, z));
  const ppm = 1/mppAt(z);                              // px/m(2D の ppm と同義)
  function mercPx(lat,lon){ const W=256*Math.pow(2,z);
    const sx=(lon+180)/360*W; const s=Math.sin(lat*Math.PI/180);
    const sy=(0.5-Math.log((1+s)/(1-s))/(4*Math.PI))*W; return [sx,sy]; }
  const [mx0,my0] = mercPx(LAT0,LON0);
  // 世界(m)→ mercator px: X = mx0 + wx*ppm ; Y = my0 - wy*ppm(北で Y 減少)
  const tx0=Math.floor((mx0+x0*ppm)/256), tx1=Math.floor((mx0+x1*ppm)/256);
  const ty0=Math.floor((my0-y1*ppm)/256), ty1=Math.floor((my0-y0*ppm)/256);  // y1(北)=上端
  const nx=tx1-tx0+1, ny=ty1-ty0+1, nTiles=nx*ny;
  if(nTiles<1 || nTiles>96) return;                    // 安全弁(異常/過大は無地地面のまま)
  // 3) タイルグリッド全体を 1 枚のキャンバスに合成
  const canvas = document.createElement('canvas'); canvas.width=nx*256; canvas.height=ny*256;
  const g2d = canvas.getContext('2d');
  const tex = new THREE.CanvasTexture(canvas);
  tex.minFilter = THREE.LinearFilter; tex.magFilter = THREE.LinearFilter;
  tex.generateMipmaps = false; tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
  try { const maxA = renderer.capabilities.getMaxAnisotropy(); tex.anisotropy = Math.min(8, maxA||1); } catch(e){}
  // 4) 地面平面(タイルグリッドが覆う世界矩形にぴったり合わせる)
  const wxL=(tx0*256-mx0)/ppm, wxR=((tx1+1)*256-mx0)/ppm;   // 西→東
  const wyT=-(ty0*256-my0)/ppm, wyB=-((ty1+1)*256-my0)/ppm; // 北(大)→南(小)
  const geo = new THREE.PlaneGeometry(wxR-wxL, wyT-wyB);
  geo.rotateX(-Math.PI/2);   // 平面 +Y → three -Z(世界北)/ +X → three +X(世界東)
  const mat = new THREE.MeshBasicMaterial({ map:tex, transparent:true,
    opacity:OSM.opacity, depthWrite:false });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.set((wxL+wxR)/2, 0.02, -(wyT+wyB)/2);       // three z = -世界y
  mesh.renderOrder = -1; mesh.visible = false;
  scene.add(mesh); OSM.mesh = mesh;
  // 5) タイル取得(非同期・失敗は無視=無地フォールバック)。初回有効化時のみ発火。
  OSM.load = function(){ if(OSM.loaded) return; OSM.loaded = true;
    for(let ty=ty0; ty<=ty1; ty++) for(let tx=tx0; tx<=tx1; tx++){
      const img = new Image(); img.crossOrigin='anonymous';
      const dx=(tx-tx0)*256, dy=(ty-ty0)*256;
      img.onload = ()=>{ try{ g2d.drawImage(img, dx, dy, 256, 256); tex.needsUpdate=true; }catch(e){} };
      img.onerror = ()=>{};
      img.src = `https://tile.openstreetmap.org/${z}/${tx}/${ty}.png`;
    } };
  OSM.ensureLoaded = OSM.load;
})();

// ---------- 建物(kind ごとにジオメトリを統合 = 少ないドローコール)
const buildingMats = [];
const buildingMeshes = [];
(function buildBuildings(){
  const rot = new THREE.Matrix4().makeRotationX(-Math.PI/2); // (e,n,up)->(e,up,-n)
  const acc = {}; // kind -> {pos:[],norm:[]}
  for(const b of SCENE.buildings){
    const fp = b.footprint;
    if(!fp || fp.length < 3) continue;
    const shape = new THREE.Shape();
    shape.moveTo(fp[0][0], fp[0][1]);
    for(let i=1;i<fp.length;i++) shape.lineTo(fp[i][0], fp[i][1]);
    const h = Math.max(b.height || FH, 1.0);
    let geo;
    try { geo = new THREE.ExtrudeGeometry(shape, { depth:h, bevelEnabled:false, steps:1 }); }
    catch(e){ continue; }
    geo.translate(0, 0, b.base || 0);
    geo.applyMatrix4(rot);
    const kind = KIND_COLOR[b.kind] !== undefined ? b.kind : 'generic';
    const a = acc[kind] || (acc[kind] = { pos:[], norm:[] });
    const p = geo.attributes.position.array, n = geo.attributes.normal.array;
    const idx = geo.index ? geo.index.array : null;
    if(idx){ for(let i=0;i<idx.length;i++){ const v=idx[i]*3;
      a.pos.push(p[v],p[v+1],p[v+2]); a.norm.push(n[v],n[v+1],n[v+2]); } }
    else { for(let i=0;i<p.length;i++){ a.pos.push(p[i]); a.norm.push(n[i]); } }
    geo.dispose();
  }
  for(const kind in acc){
    const a = acc[kind];
    const bg = new THREE.BufferGeometry();
    bg.setAttribute('position', new THREE.Float32BufferAttribute(a.pos, 3));
    bg.setAttribute('normal',   new THREE.Float32BufferAttribute(a.norm, 3));
    const mat = new THREE.MeshLambertMaterial({ color:KIND_COLOR[kind],
      transparent:true, opacity:1.0 });
    buildingMats.push(mat);
    const mesh = new THREE.Mesh(bg, mat);
    buildingMeshes.push(mesh); scene.add(mesh);
  }
})();

// ---------- 道路(全ポリラインを 1 本の LineSegments に統合)
let roadObj = null;
(function buildRoads(){
  const pos = [];
  for(const r of SCENE.roads){
    const g = r.g;
    for(let i=1;i<g.length;i++){
      pos.push(g[i-1][0], 0.4, -g[i-1][1]);
      pos.push(g[i][0],   0.4, -g[i][1]);
    }
  }
  const bg = new THREE.BufferGeometry();
  bg.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  roadObj = new THREE.LineSegments(bg, new THREE.LineBasicMaterial({ color:0x4a5364 }));
  scene.add(roadObj);
})();

// ---------- 線路(地上=線 / 地下鉄=半透明チューブ)
const railSurfaceObjs = [], subwayObjs = [];
(function buildRails(){
  for(const r of SCENE.rails){
    const g = r.g;
    if(g.length < 2) continue;
    if(r.kind === 'subway'){
      const pts = g.map(p => V(p[0], p[1], r.z));
      const curve = new THREE.CatmullRomCurve3(pts);
      const seg = Math.min(Math.max(g.length*4, 8), 200);
      const tube = new THREE.TubeGeometry(curve, seg, 3.2, 8, false);
      const mat = new THREE.MeshLambertMaterial({ color:0x8a5cf0,
        transparent:true, opacity:0.35 });
      const m = new THREE.Mesh(tube, mat); m.userData.rail = true;
      subwayObjs.push(m); scene.add(m);
    } else {
      const pos = [];
      for(const p of g){ pos.push(p[0], (r.z||0)+0.6, -p[1]); }
      const bg = new THREE.BufferGeometry();
      bg.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
      const m = new THREE.Line(bg, new THREE.LineBasicMaterial({ color:0x9aa4b2 }));
      m.userData.rail = true; railSurfaceObjs.push(m); scene.add(m);
    }
  }
})();

// ---------- 建物ラベル(名前つき・大規模のみ、うっすら)
const labelSprites = [];
(function buildLabels(){
  const cand = SCENE.buildings.filter(b => b.name &&
    (b.levels >= 7 || (b.kind==='station'||b.kind==='retail') && b.levels >= 5));
  cand.sort((a,b)=> (b.levels||0)-(a.levels||0));
  for(const b of cand.slice(0, 40)){
    const cv = document.createElement('canvas'); cv.width = 256; cv.height = 64;
    const cx = cv.getContext('2d');
    cx.font = '600 30px "Hiragino Kaku Gothic ProN", Meiryo, sans-serif';
    cx.fillStyle = 'rgba(255,255,255,.92)';
    cx.strokeStyle = 'rgba(0,0,0,.65)'; cx.lineWidth = 5;
    cx.textAlign = 'center'; cx.textBaseline = 'middle';
    const nm = b.name.length > 10 ? b.name.slice(0,10)+'…' : b.name;
    cx.strokeText(nm, 128, 34); cx.fillText(nm, 128, 34);
    const tex = new THREE.CanvasTexture(cv); tex.anisotropy = 4;
    const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map:tex,
      transparent:true, opacity:0.9, depthTest:true }));
    sp.position.copy(V(b.cx, b.cy, (b.height||FH) + 10));
    sp.scale.set(64, 16, 1);
    labelSprites.push(sp); scene.add(sp);
  }
})();

// ---------- エージェント(InstancedMesh のカプセル)
function capsuleGeometry(r, h){
  const pts = [], N = 6;
  for(let i=0;i<=N;i++){ const a=-Math.PI/2 + (Math.PI/2)*(i/N);
    pts.push(new THREE.Vector2(Math.cos(a)*r, -h/2 + Math.sin(a)*r)); }
  for(let i=0;i<=N;i++){ const a=(Math.PI/2)*(i/N);
    pts.push(new THREE.Vector2(Math.cos(a)*r, h/2 + Math.sin(a)*r)); }
  return new THREE.LatheGeometry(pts, 10);
}
const NA = TRACKS.ids.length;
const agentGeo = capsuleGeometry(2.6, 5.0);
const agentMat = new THREE.MeshLambertMaterial({ color:0xffffff });
const agents = new THREE.InstancedMesh(agentGeo, agentMat, NA);
agents.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
agents.count = NA;
scene.add(agents);
// per-instance color(setColorAt が instanceColor を生成する)
const _col = new THREE.Color();

// ---------- 車(traffic segs)= InstancedMesh の箱
const CAR_CAP = 220;
const carGeo = new THREE.BoxGeometry(4.5, 2.2, 2.4);
const carMat = new THREE.MeshLambertMaterial({ color:0xf2b134 });
const cars = new THREE.InstancedMesh(carGeo, carMat, CAR_CAP);
cars.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
scene.add(cars);

// ---------- 補間ロジック(2D ビューア posAt/alongPath 移植)
function pathLen(pts){ let l=0; for(let i=1;i<pts.length;i++)
  l += Math.hypot(pts[i][0]-pts[i-1][0], pts[i][1]-pts[i-1][1]); return l; }
function alongPath(pts, f){
  const total = pathLen(pts); if(total===0) return pts[0];
  let target = total*f;
  for(let i=1;i<pts.length;i++){ const seg=Math.hypot(pts[i][0]-pts[i-1][0], pts[i][1]-pts[i-1][1]);
    if(target<=seg){ const g = seg? target/seg : 0;
      return [pts[i-1][0]+(pts[i][0]-pts[i-1][0])*g, pts[i-1][1]+(pts[i][1]-pts[i-1][1])*g]; }
    target -= seg; }
  return pts[pts.length-1];
}
function posAt(t){ const s0=Math.floor(t), f=t-s0;
  const P = TRACKS.positions, M = TRACKS.moves;
  return TRACKS.ids.map((_, i)=>{
    const w = P[s0][i][2];
    if(w!==0) return [P[s0][i][0], P[s0][i][1], w];
    const nm = (s0+1<NS)? M[s0+1][i] : null;
    if(nm){ const p = alongPath(nm[1], f); return [p[0], p[1], 0]; }
    const a = P[s0][i], b = P[Math.min(s0+1,NS-1)][i];
    if(b[2]!==0) return [a[0], a[1], 0];
    return [a[0]+(b[0]-a[0])*f, a[1]+(b[1]-a[1])*f, 0]; });
}
// w -> up(高さ)
function upOf(w){
  if(w >= 1000){ const floor = (w-1000) % 100; return Math.max((floor-0.5)*FH, 0.8); }
  return 1.2; // 路上
}

// ---------- 色分け
const OCC_HUE = {};
function agentColor(i){
  const a = TRACKS.agents[i] || {};
  if(colorMode === 'visitor')
    return a.visitor ? 0xffb454 : 0x54a0ff;   // 来訪=橙 / 居住=青
  // occ
  const occ = a.occupation || '?';
  if(!(occ in OCC_HUE)){ let hsum=0; for(const ch of occ) hsum=(hsum*31+ch.charCodeAt(0))%360; OCC_HUE[occ]=hsum; }
  return _col.setHSL(OCC_HUE[occ]/360, 0.62, 0.6).getHex();
}
function refreshColors(){
  for(let i=0;i<NA;i++){ _col.setHex(agentColor(i));
    agents.setColorAt(i, _col); }
  agents.instanceColor.needsUpdate = true;
}

// ---------- 昼夜(sim_min から太陽と空)
function simMinAt(t){ const s0=Math.floor(t), f=t-s0;
  const a = TRACKS.sim_min[s0];
  const b = TRACKS.sim_min[Math.min(s0+1,NS-1)];
  return a + (b-a)*f; }
const C_DAY = new THREE.Color(0x9dc3ec), C_DUSK = new THREE.Color(0xdd7a45),
      C_NIGHT = new THREE.Color(0x0a0e18);
const _sky = new THREE.Color(), _sun = new THREE.Color();
function updateSky(min){
  const day = ((min % 1440) + 1440) % 1440;
  const ang = (day/1440)*Math.PI*2 - Math.PI/2;   // 6:00 で日の出付近
  const el = Math.sin(ang);                         // -1..1(正午で最大)
  // 太陽方向(東→天頂→西)。方位は時間で回す。
  const az = (day/1440)*Math.PI*2;
  const sx = Math.cos(az)*Math.max(0.15, Math.abs(el)+0.2);
  const sz = Math.sin(az)*Math.max(0.15, Math.abs(el)+0.2);
  sun.position.set(sx*600, Math.max(el,0.02)*700 + 40, sz*600);
  sun.target.position.set(0,0,0);
  const daylight = Math.max(0, el);
  const twilight = Math.max(0, 1 - Math.abs(el)*3.2);  // 地平線付近で最大
  // 空色: 夜→昼、地平線で夕焼け
  _sky.copy(C_NIGHT).lerp(C_DAY, Math.min(1, daylight*1.6));
  _sky.lerp(C_DUSK, twilight*0.55);
  scene.background.copy(_sky);
  scene.fog.color.copy(_sky);
  // 太陽色: 正午=白, 地平線=橙
  _sun.copy(C_DUSK).lerp(new THREE.Color(0xffffff), Math.min(1, daylight*1.8));
  sun.color.copy(_sun);
  sun.intensity = 0.25 + daylight*1.15;
  hemi.intensity = 0.25 + daylight*0.5;
  ambient.intensity = 0.18 + daylight*0.15;
}

// ---------- 毎フレームのエージェント/車の配置
const _m = new THREE.Matrix4(), _q = new THREE.Quaternion(),
      _p = new THREE.Vector3(), _s = new THREE.Vector3(1,1,1), _hide = new THREE.Vector3(0,0,0);
function placeAgents(t){
  const pos = posAt(t);
  for(let i=0;i<NA;i++){
    const [x,y,w] = pos[i];
    if(w === -1 || w === -2){  // 範囲外・睡眠は隠す
      _m.compose(_p.set(0,-9999,0), _q, _hide); agents.setMatrixAt(i,_m); continue; }
    _p.set(x, upOf(w), -y);
    _m.compose(_p, _q, _s); agents.setMatrixAt(i, _m);
  }
  agents.instanceMatrix.needsUpdate = true;
}
function placeCars(t){
  const s0 = Math.floor(t), f = t - s0;
  const segs = (TRACKS.traffic[s0] && TRACKS.traffic[s0].segs) || [];
  const n = Math.min(segs.length, CAR_CAP);
  for(let i=0;i<CAR_CAP;i++){
    if(i < n){ const p = alongPath(segs[i], f);
      _p.set(p[0], 0.9, -p[1]); _m.compose(_p, _q, _s); }
    else { _m.compose(_p.set(0,-9999,0), _q, _hide); }
    cars.setMatrixAt(i, _m);
  }
  cars.count = CAR_CAP;
  cars.instanceMatrix.needsUpdate = true;
}

// ---------- 再生状態
let t = 0, playing = true, speed = 1, colorMode = 'visitor', dayNight = true;
const timeline = document.getElementById('timeline');
timeline.max = String(NS - 1);
const clockEl = document.getElementById('clock');
function fmtClock(min){ const d=((min%1440)+1440)%1440;
  const day = Math.floor(((min)/1440));
  const h=String(Math.floor(d/60)).padStart(2,'0'), m=String(Math.floor(d%60)).padStart(2,'0');
  return `Day ${day+1}  ${h}:${m}`; }

document.getElementById('play').onclick = (e)=>{ playing=!playing;
  e.target.textContent = playing? '❚❚' : '▶︎'; };
document.getElementById('speed').onchange = (e)=>{ speed = parseFloat(e.target.value); };
timeline.oninput = (e)=>{ t = parseFloat(e.target.value); playing=false;
  document.getElementById('play').textContent='▶︎'; };
document.getElementById('reset').onclick = ()=>{ camera.position.copy(CAM0);
  controls.target.set(0,20,0); };
document.getElementById('xray').onchange = (e)=>{
  const o = e.target.checked ? 0.28 : 1.0;
  for(const m of buildingMats){ m.opacity = o; m.needsUpdate = true; }
  saveSettings(); };
document.querySelectorAll('#colorSeg button').forEach(btn=>{
  btn.onclick = ()=>{ document.querySelectorAll('#colorSeg button').forEach(b=>b.classList.remove('on'));
    btn.classList.add('on'); colorMode = btn.dataset.c; refreshColors(); buildLegend(); saveSettings(); };
});

// ---------- レイヤーパネル / OSM 地面 / 昼夜 の配線 + localStorage
const L3 = id => document.getElementById(id).checked;
const osmOp = document.getElementById('osmOp'), osmOpV = document.getElementById('osmOpV');
function applyLayers(){
  buildingMeshes.forEach(m=> m.visible = L3('lyBld'));
  agents.visible = L3('lyAgent');
  cars.visible = L3('lyCars');
  if(roadObj) roadObj.visible = L3('lyRoad');
  railSurfaceObjs.forEach(o=> o.visible = L3('lyRail'));
  subwayObjs.forEach(o=> o.visible = L3('lyUnder'));
  labelSprites.forEach(s=> s.visible = L3('lyLabels'));
  setDayNight(L3('lyDayNight'));
  const osmOn = L3('lyOsm');
  if(OSM.mesh){ OSM.mesh.visible = osmOn; if(osmOn && OSM.ensureLoaded) OSM.ensureLoaded(); }
  if(gridHelper) gridHelper.visible = !osmOn;
  document.getElementById('osmAttr').style.display = osmOn ? 'block' : 'none';
}
['lyBld','lyAgent','lyCars','lyRoad','lyRail','lyUnder','lyLabels','lyDayNight','lyOsm'].forEach(id=>{
  const el = document.getElementById(id);
  if(el) el.onchange = ()=>{ applyLayers(); saveSettings(); };
});
osmOp.oninput = ()=>{ OSM.opacity = Number(osmOp.value);
  if(OSM.mesh) OSM.mesh.material.opacity = OSM.opacity;
  osmOpV.textContent = Math.round(OSM.opacity*100)+'%'; saveSettings(); };
document.getElementById('lyHdr').onclick = ()=>{ const b = document.getElementById('lyBody');
  const off = b.style.display === 'none'; b.style.display = off ? 'block' : 'none';
  document.getElementById('lyHdr').textContent = off ? 'レイヤー ▾' : 'レイヤー ▸'; };

// 昼夜ライティング OFF = 均一な昼光で全体を見やすく固定
function freezeDaylight(){
  sun.position.set(420, 720, 260); sun.target.position.set(0,0,0);
  sun.color.set(0xffffff); sun.intensity = 1.1;
  hemi.intensity = 0.78; ambient.intensity = 0.5;
  const c = new THREE.Color(0x223047); scene.background.copy(c); scene.fog.color.copy(c);
}
function setDayNight(on){ dayNight = on; if(!on) freezeDaylight(); }

// 設定を localStorage にラン別保持(再訪時に維持)
const LS_KEY = 'shibuya3d:__RUN_NAME__';
function saveSettings(){ try{ localStorage.setItem(LS_KEY, JSON.stringify({
  lyBld:L3('lyBld'), lyAgent:L3('lyAgent'), lyCars:L3('lyCars'), lyRoad:L3('lyRoad'),
  lyRail:L3('lyRail'), lyUnder:L3('lyUnder'), lyLabels:L3('lyLabels'),
  lyDayNight:L3('lyDayNight'), lyOsm:L3('lyOsm'), osmOp:OSM.opacity,
  xray:document.getElementById('xray').checked, colorMode })); }catch(e){} }
function loadSettings(){ try{ const s = JSON.parse(localStorage.getItem(LS_KEY) || 'null'); if(!s) return;
  ['lyBld','lyAgent','lyCars','lyRoad','lyRail','lyUnder','lyLabels','lyDayNight','lyOsm','xray'].forEach(k=>{
    const el = document.getElementById(k); if(el && typeof s[k] === 'boolean') el.checked = s[k]; });
  if(typeof s.osmOp === 'number'){ OSM.opacity = s.osmOp; osmOp.value = s.osmOp; }
  if(s.colorMode){ colorMode = s.colorMode;
    document.querySelectorAll('#colorSeg button').forEach(b=> b.classList.toggle('on', b.dataset.c===colorMode)); }
}catch(e){} }

// ---------- 凡例
function buildLegend(){
  const el = document.getElementById('legend'); let h = '';
  if(colorMode === 'visitor'){
    h += `<div class="row"><span class="sw" style="background:#54a0ff"></span>居住者</div>`;
    h += `<div class="row"><span class="sw" style="background:#ffb454"></span>来訪者</div>`;
  } else {
    const occs = [...new Set(TRACKS.agents.map(a=>a.occupation))].slice(0, 12);
    for(const o of occs){ const c = '#'+_col.setHex(agentColorByOcc(o)).getHexString();
      h += `<div class="row"><span class="sw" style="background:${c}"></span>${o}</div>`; }
  }
  h += `<div class="row" style="margin-top:6px"><span class="sw" style="background:#f2b134"></span>車</div>`;
  el.innerHTML = h;
}
function agentColorByOcc(occ){
  if(!(occ in OCC_HUE)){ let hsum=0; for(const ch of occ) hsum=(hsum*31+ch.charCodeAt(0))%360; OCC_HUE[occ]=hsum; }
  return _col.setHSL(OCC_HUE[occ]/360, 0.62, 0.6).getHex();
}

// ---------- クリックで人物情報(Raycaster)
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
let downXY = null;
renderer.domElement.addEventListener('pointerdown', e=>{ downXY=[e.clientX,e.clientY]; });
renderer.domElement.addEventListener('pointerup', e=>{
  if(!downXY) return;
  if(Math.hypot(e.clientX-downXY[0], e.clientY-downXY[1]) > 5) return; // ドラッグは無視
  mouse.x = (e.clientX/window.innerWidth)*2 - 1;
  mouse.y = -(e.clientY/window.innerHeight)*2 + 1;
  raycaster.setFromCamera(mouse, camera);
  const hit = raycaster.intersectObject(agents);
  if(hit.length){ showInfo(hit[0].instanceId); }
});
document.getElementById('info-close').onclick = ()=>{
  document.getElementById('info').style.display='none'; };
function bldNameAt(w){ if(w < 1000) return null;
  const bi = Math.floor((w-1000)/100); const b = SCENE.buildings[bi];
  return b ? (b.name || b.kind) : null; }
function showInfo(i){
  const a = TRACKS.agents[i]; if(!a) return;
  const cur = posAt(t)[i]; const w = cur[2];
  document.getElementById('info-name').textContent = a.name || ('agent'+a.id);
  document.getElementById('info-role').textContent =
    `${a.occupation||'?'} ・ ${a.gender||'?'} ${a.age||''} ・ ${a.visitor?'来訪者':'居住者'}`;
  let loc = '路上';
  if(w >= 1000){ const floor=(w-1000)%100; const nm=bldNameAt(w);
    loc = `${nm||'建物'} ${floor}F`; }
  else if(w === -1) loc = '範囲外';
  else if(w === -2) loc = '睡眠中';
  document.getElementById('info-loc').textContent = loc;
  document.getElementById('info').style.display = 'block';
}

// ---------- ループ
loadSettings();
{ const o = document.getElementById('xray').checked ? 0.28 : 1.0;
  for(const m of buildingMats){ m.opacity = o; } }
if(OSM.mesh) OSM.mesh.material.opacity = OSM.opacity;
osmOpV.textContent = Math.round(OSM.opacity*100)+'%';
applyLayers();
saveSettings();
refreshColors();
buildLegend();
let last = performance.now();
function animate(now){
  requestAnimationFrame(animate);
  const dt = Math.min((now - last)/1000, 0.1); last = now;
  if(playing){ t += dt * speed * 1.2;   // 1x で ≈1.2 step/秒
    if(t >= NS-1){ t = 0; } }
  if(t > NS-1) t = NS-1;
  timeline.value = String(t);
  const min = simMinAt(t);
  if(dayNight) updateSky(min);
  clockEl.textContent = fmtClock(min);
  placeAgents(t);
  placeCars(t);
  controls.update();
  renderer.render(scene, camera);
}
requestAnimationFrame(animate);

window.addEventListener('resize', ()=>{
  camera.aspect = window.innerWidth/window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
