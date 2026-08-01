"""Web3D ビューア生成(バッチE / 3D 可視化)。

使い方:  python viz/make_viewer3d.py runs/<name> [--no-traffic] [--tracks-binary]
生成物:  runs/<name>/viewer3d.html
  自己完結の単一 HTML。three.js(r128, MIT)本体・OrbitControls・シーンデータを埋め込み。
  ブラウザで開けば即グリグリ(OrbitControls)+ 再生 + 昼夜 + クリックで人物情報。

  --tracks-binary : 軌跡を JSON 埋め込みではなく量子化バイナリのチャンク遅延ロードにする
    (P0 / docs/research/dt-integration-deep.md §2)。HTML には軽いヘッダ(meta/agents/ids/
    sim_min + チャンク表)だけを埋め、位置・移動・交通は runs/<name>/tracks_bin/chunk_NNNN.js
    を再生位置に応じて `<script src>` で動的に取り込む(既存 plateau_mesh.js と同じ file://
    で動く方式)。**HTML サイズが軌跡の長さにほぼ依存しなくなる**のが本質で、
    1万体×1日で 90.4MB(80MB ゲート超過)→ 約 26MB、10日でもほぼ同じに収まる。
    既定 OFF=このフラグ無しでは生成 HTML は従来とバイト同一。

  分離版(viewer3d_lite.html)は、ラン直下に plateau_tex.js(export_3d --plateau-tex 産)が
  あればテクスチャ付き PLATEAU LOD2.2 を描く(無テクスチャ plateau_mesh とは排他=置換)。
  埋め込み版 viewer3d.html にはテクスチャを入れない(アトラス 1/2 は 80MB ゲート超過)。
  plateau_web.json に ubld4 があれば地下街 LOD4.1 を面種別で塗り分ける(既定 OFF のまま)。
  テクスチャ経路の 80MB ゲート対策(標準経路・docs/plans/plateau-3d.md のサイズ表):
    ① plateau_mesh.js から統合メッシュ配列を落とす(tex が置換するので読まれない)
    ② scene3d/tracks.bin があれば**分離版だけ**軌跡をチャンク遅延ロードにする
       (埋め込み版は「単一ファイルで完結」を保つため触らない)

データは runs/<name>/scene3d/{scene.json,tracks.json} を読む。無ければ export_3d を実行して生成。
tracks.json が無く tracks.bin だけのラン(export_3d --no-tracks-json)も従来どおり開ける。
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


def _load_tracks_bin():
    """scripts/tracks_bin.py を場所非依存で読み込む(_load_export3d と同じ流儀)。"""
    spec = importlib.util.spec_from_file_location(
        "tracks_bin", REPO_ROOT / "scripts" / "tracks_bin.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_notable():
    """viz/notable_events.py を場所非依存で読み込む(_load_export3d と同じ流儀)。"""
    spec = importlib.util.spec_from_file_location(
        "notable_events", REPO_ROOT / "viz" / "notable_events.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ensure_notable(run_dir: Path, tracks_json: str) -> str | None:
    """顕著イベントを l1_events.parquet(scene/tracks と同一入力源)から抽出。

    notable が 0 件 / parquet 不在なら None を返す = 注入されない = 従来とバイト同一。
    成果は scene3d/notable_events.json に書き出し(透明性)、注入用テキストを返す。
    抽出は決定論(乱数なし)。間引き件数は stdout に明記(silent cap 禁止)。
    """
    ev_p = run_dir / "l1_events.parquet"
    if not ev_p.exists():
        return None
    try:
        meta = json.loads(tracks_json).get("meta", {})
    except Exception:
        meta = {}
    n_steps = int(meta.get("nSteps", 0) or 0)
    stride = int(meta.get("step_stride", 1) or 1)
    nb = _load_notable()
    try:
        data = nb.extract_from_run(ev_p, n_steps=n_steps, stride=stride)
    except Exception as e:                       # 抽出失敗は静かに従来動作へ退避
        print(f"  [notable] 抽出をスキップ({type(e).__name__}: {e})")
        return None
    if not data.get("events"):
        return None
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    try:
        out_p = run_dir / "scene3d" / "notable_events.json"
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(text, encoding="utf-8")
    except Exception:
        pass
    # kind 別件数と間引きの明示(silent cap 禁止)
    caps = data.get("caps", {})
    print(f"  [notable] {data['n_kept']}件を採用(候補 {data['n_total']}件)")
    for kind in sorted(caps, key=lambda k: (-caps[k]["kept"], k)):
        c = caps[kind]
        cap_note = f"  ← 間引き {c['dropped']}件" if c["dropped"] else ""
        print(f"    {kind:18s} {c['kept']:4d}/{c['total']:<5d}{cap_note}")
    return text


def _ensure_scene(run_dir: Path,
                  tracks_binary: bool = False) -> tuple[str, str, str | None, str | None]:
    """scene/tracks のテキストを返す。tracks_binary=True のときは tracks.json の代わりに
    tracks_meta.json(= tracks.bin の JSON ヘッダと同一文字列)を返す。"""
    scene_dir = run_dir / "scene3d"
    scene_p = scene_dir / "scene.json"
    tracks_p = scene_dir / "tracks.json"
    bin_p = scene_dir / "tracks.bin"
    meta_p = scene_dir / "tracks_meta.json"
    if tracks_binary:
        if not (scene_p.exists() and bin_p.exists() and meta_p.exists()):
            _load_export3d().export_run(run_dir, tracks_binary=True)
        tracks_text = meta_p.read_text(encoding="utf-8")
    else:
        if not (scene_p.exists() and (tracks_p.exists() or bin_p.exists())):
            _load_export3d().export_run(run_dir)
        if tracks_p.exists():
            tracks_text = tracks_p.read_text(encoding="utf-8")
        else:                       # tracks.bin だけのラン: 復号して従来経路へ流す
            tracks_text = json.dumps(
                _load_tracks_bin().decode_tracks(bin_p.read_bytes()),
                ensure_ascii=False, separators=(",", ":"))
    # PLATEAU 実形状(export_3d --plateau の成果物)。無ければ None=従来とバイト同一。
    pw_p = scene_dir / "plateau_web.json"
    plateau = pw_p.read_text(encoding="utf-8") if pw_p.exists() else None
    # 地形(並行生成の terrain_web.json)。無ければ None=地形なし=完全従来動作。
    tw_p = scene_dir / "terrain_web.json"
    terrain = tw_p.read_text(encoding="utf-8") if tw_p.exists() else None
    return (scene_p.read_text(encoding="utf-8"), tracks_text, plateau, terrain)


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
               plateau_src: str | None = None,
               terrain_json: str | None = None,
               has_extras: bool = False,
               mode_legend: dict | None = None,
               notable_json: str | None = None,
               indoor_json: str | None = None,
               tracks_binary: bool = False,
               chunk_dir: str = "tracks_bin",
               has_ubld4: bool = False,
               plateau_tex_src: str | None = None,
               tex_note: bool = False) -> str:
    three_js, orbit_js, lic = _read_vendor()
    html = _TEMPLATE
    html = html.replace("__RUN_NAME__", run_name)
    html = html.replace("__THREE_LICENSE__", lic.strip())
    html = html.replace("__THREE_JS__", three_js)
    html = html.replace("__ORBIT_JS__", orbit_js)
    html = html.replace("__SCENE_JSON__", _json_for_script(scene_json))
    html = html.replace("__TRACKS_JSON__", _json_for_script(tracks_json))
    # 以降は「データ存在時のみ注入」。無ければ一切触らない=データ無しラン同士はバイト同一。
    if plateau_json is not None or plateau_src is not None:
        html = _inject_plateau(html, plateau_json, plateau_src)
    if has_extras or has_ubld4:         # plateau_web.extras(地下街/歩道橋)
        html = _inject_extras(html)
    if has_ubld4:                       # plateau_web.ubld4(地下街 LOD4.1・面種別+層)
        html = _inject_ubld4(html)
    if plateau_tex_src is not None:     # plateau_tex.js(テクスチャ付き LOD2.2・分離版のみ)
        html = _inject_plateau_tex(html, plateau_tex_src)
    elif tex_note:                      # 埋め込み版は注記だけ(80MB ゲート)
        html = _inject_tex_note(html)
    if terrain_json is not None:        # terrain_web.json(地形起伏+接地)
        html = _inject_terrain(html, terrain_json)
    if mode_legend:                     # tracks.meta.mode_legend(移動手段)
        html = _inject_modes(html)
    if notable_json is not None:        # notable_events.json(顕著イベントパネル)
        html = _inject_notable(html, notable_json)
    if indoor_json is not None:         # indoor overlay(フロア板+接近フェード+実座標エージェント)
        html = _inject_indoor(html, indoor_json)
    if tracks_binary:                   # 軌跡バイナリ(P0): JSON 埋め込み → チャンク遅延ロード
        html = _inject_tracks_binary(html, chunk_dir)
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
  // DoubleSide: PLATEAU 由来メッシュは面の巻きが局所的に不整合なことがある
  // (three.js は裏面の法線を自動反転して陰影も正しく出す)
  // 既定は無彩色(vertexColors=false + NEUTRAL)。「分類色」ON で applyBuildingPalette が頂点色へ。
  const mat = new THREE.MeshLambertMaterial({ vertexColors:false, color:NEUTRAL_BLD,
    transparent:true, opacity:1.0, side:THREE.DoubleSide });
  mat.userData.plateau = true;
  buildingMats.push(mat);
  const mesh = new THREE.Mesh(bg, mat);
  buildingMeshes.push(mesh); scene.add(mesh);
  try { const s = document.querySelector('#hud .sub');
    if(s && PLATEAU_DATA.attribution) s.textContent += ' / ' + PLATEAU_DATA.attribution;
  } catch(e){}
})();

"""


# ============================================================ テクスチャ LOD2.2(松 / B-1)
def _inject_plateau_tex(html: str, tex_src: str) -> str:
    """テクスチャ付き LOD2.2(plateau_tex.js サイドカー)の描画を注入する。

    **_inject_plateau の後に呼ぶ**(_PLATEAU_DECL の PLATEAU_SKIP 行を書き換えるため)。
    分離版(viewer3d_lite.html)専用: 埋め込み版はアトラス 1/2 で 80MB ゲートを超えるため
    テクスチャを持たない(注記のみ・_inject_tex_note)。
    ①サイドカー参照 ②宣言の差し替え ③無テクスチャ PLATEAU の停止(排他)
    ④押出し箱の出し分け ⑤トグル ⑥タイル構築+配線 の 6 箇所を一意置換。"""
    # ① サイドカー(plateau_mesh.js の後・scene-data の前)
    anchor_data = '<script type="application/json" id="scene-data">'
    html = _replace_once(html, anchor_data,
                         f'<script src="{tex_src}"></script>\n' + anchor_data, "tex-data")
    # ② PLATEAU_SKIP の意味を切り替える(テクスチャ版は bbox 内の全建物を実形状で持つので、
    #    照合済みだけ skip すると未照合の押出し箱が実形状に突き刺さる)
    anchor_skip = "const PLATEAU_SKIP = new Set(PLATEAU_DATA ? PLATEAU_DATA.matched_ids : []);"
    html = _replace_once(html, anchor_skip, _TEX_DECL, "tex-decl")
    # ③ 無テクスチャ PLATEAU メッシュとは排他(tex があればそちらが置換する)。
    #    positions_b64 の有無も見るのは、tex 経路の plateau_mesh.js が統合メッシュ配列を
    #    載せない(縮退策①)ため。サイドカーだけ残して plateau_tex.js を消しても
    #    例外を出さず押出し箱へ退避する。
    anchor_off = "  if(!PLATEAU_DATA) return;"
    html = _replace_once(html, anchor_off,
                         "  if(!PLATEAU_DATA || PLATEAU_TEX_DATA"
                         " || !PLATEAU_DATA.positions_b64) return;"
                         "   // テクスチャ版と排他(置換)", "tex-plateau-off")
    # ④ 押出し箱はテクスチャ OFF のときの代替表示に回す
    anchor_vis = "  buildingMeshes.forEach(m=> m.visible = L3('lyBld'));"
    html = _replace_once(html, anchor_vis,
                         "  buildingMeshes.forEach(m=> m.visible = L3('lyBld') && !TEX_ON);",
                         "tex-bldvis")
    # ⑤ レイヤーパネルのトグル
    anchor_panel = '      <label class="chk"><input type="checkbox" id="lyBld" checked> 建物</label>'
    html = _replace_once(html, anchor_panel, anchor_panel + "\n" + _TEX_TOGGLE, "tex-panel")
    # ⑥ 構築(道路の直前)+ 配線(ループの直前)
    anchor_build = "// ---------- 道路(全ポリラインを 1 本の LineSegments に統合)"
    html = _replace_once(html, anchor_build, _TEX_BUILD + anchor_build, "tex-build")
    anchor_wire = "// ---------- ループ"
    html = _replace_once(html, anchor_wire, _TEX_WIRE + anchor_wire, "tex-wire")
    return html


# 分離版サイドカーのうち、テクスチャ経路では **一度も読まれない** 統合メッシュ配列。
# buildPlateau が tex 在時に即 return する(_inject_plateau_tex ③)ので落として安全。
_TEX_DEAD_KEYS = ("positions_b64", "indices_b64", "colors_b64")


def _slim_plateau_for_tex(plateau_json: str) -> str:
    """縮退策①: テクスチャ版の plateau_mesh.js から統合メッシュ配列を外す。

    残すもの = matched_ids / extras(歩道橋)/ ubld4(地下街)/ 出典 など、
    テクスチャ経路でも実際に読まれるキー。**tex が無い経路では呼ばない**ので、
    従来の plateau_mesh.js は 1 バイトも変わらない。
    再直列化は export_3d と同じ separators なので、落としたキー以外は原文と同じ並び。"""
    data = json.loads(plateau_json)
    dropped = [k for k in _TEX_DEAD_KEYS if k in data]
    if not dropped:
        return plateau_json
    slim = {k: v for k, v in data.items() if k not in _TEX_DEAD_KEYS}
    # 何をなぜ落としたかをサイドカー自身に書き残す(黙って欠けさせない)
    slim["merged_mesh_omitted"] = "plateau_tex.js supersedes it (" + ",".join(dropped) + ")"
    return json.dumps(slim, separators=(",", ":"))


def _inject_tex_note(html: str) -> str:
    """埋め込み版へ「テクスチャは分離版で」の注記だけ出す(データは入れない)。"""
    anchor_panel = '      <label class="chk"><input type="checkbox" id="lyBld" checked> 建物</label>'
    note = ('      <div class="op" style="margin-left:22px">テクスチャ表示は'
            ' viewer3d_lite.html(+plateau_tex.js)で</div>')
    return _replace_once(html, anchor_panel, anchor_panel + "\n" + note, "tex-note")


_TEX_DECL = r"""const PLATEAU_TEX_DATA = (typeof PLATEAU_TEX !== 'undefined') ? PLATEAU_TEX : null;
let TEX_ON = !!PLATEAU_TEX_DATA;
// テクスチャ版は bbox 内の全建物を実形状で持つ。照合済みだけ押出しを止めると未照合の箱が
// 実形状に突き刺さるので、tex がある時は skip を空にし「箱の集合」をまるごと
// テクスチャ OFF 時の代替表示に回す(applyLayers が TEX_ON で出し分ける)。
const PLATEAU_SKIP = new Set((PLATEAU_DATA && !PLATEAU_TEX_DATA) ? PLATEAU_DATA.matched_ids : []);"""

_TEX_TOGGLE = ('      <label class="chk"><input type="checkbox" id="lyTex" checked>'
               ' テクスチャ(PLATEAU LOD2.2)</label>')

_TEX_BUILD = r"""// ---------- テクスチャ付き PLATEAU LOD2.2(plateau_tex.js)
// 1 タイル = 1 ジオメトリ。テクスチャ付き三角形(前半)と無地の三角形(後半)を
// 2 グループに分け、マテリアル配列 [map 付き, 無彩色] で描く(タイルあたり最大 2 ドローコール)。
// UV は glTF 規約(原点=画像左上)のまま使い、テクスチャ側で flipY=false にする
// (three.js の GLTFLoader と同じ扱い)。
const texMeshes = [];
(function buildPlateauTex(){
  if(!PLATEAU_TEX_DATA) return;
  const D = PLATEAU_TEX_DATA;
  const q = D.quant_scale || 0.05, US = D.uv_scale || 65535;
  const b64 = s => { const bin=atob(s); const u=new Uint8Array(bin.length);
    for(let i=0;i<bin.length;i++) u[i]=bin.charCodeAt(i); return u; };
  const flatMat = new THREE.MeshLambertMaterial({ color:NEUTRAL_BLD, side:THREE.DoubleSide });
  for(const t of D.tiles){
    const pb = b64(t.positions_b64).buffer;
    const pi = (t.xyz_dtype === 'int32') ? new Int32Array(pb) : new Int16Array(pb);
    const o0 = t.origin_q[0], o1 = t.origin_q[1], o2 = t.origin_q[2];
    const pos = new Float32Array(pi.length);
    for(let i=0;i<pi.length;i+=3){       // 量子化解除 → local-m (e,n,up) → three (e,up,-n)
      pos[i]   =  (pi[i]   + o0) * q;
      pos[i+1] =  (pi[i+2] + o2) * q;
      pos[i+2] = -(pi[i+1] + o1) * q;
    }
    const uq = new Uint16Array(b64(t.uv_b64).buffer);
    const uv = new Float32Array(uq.length);
    for(let i=0;i<uq.length;i++) uv[i] = uq[i] / US;
    const ib = b64(t.indices_b64).buffer;
    const idx = (t.idx_dtype === 'uint32') ? new Uint32Array(ib) : new Uint16Array(ib);
    const bg = new THREE.BufferGeometry();
    bg.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    bg.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
    bg.setIndex(new THREE.BufferAttribute(idx, 1));
    bg.computeVertexNormals();
    let mesh;
    if(t.n_tex > 0 && t.atlas){
      const img = new Image();
      const tex = new THREE.Texture(img);
      tex.flipY = false;                 // glTF 規約の UV をそのまま使う
      tex.encoding = THREE.sRGBEncoding; // renderer.outputEncoding=sRGB に合わせる
      tex.anisotropy = 4;
      img.onload = ()=>{ tex.needsUpdate = true; };
      img.src = t.atlas;                 // data:image/webp;base64,...(file:// でも読める)
      const texMat = new THREE.MeshLambertMaterial({ map:tex, side:THREE.DoubleSide });
      if(t.n_flat > 0){
        bg.addGroup(0, t.n_tex*3, 0);
        bg.addGroup(t.n_tex*3, t.n_flat*3, 1);
        mesh = new THREE.Mesh(bg, [texMat, flatMat]);
      } else {
        mesh = new THREE.Mesh(bg, texMat);
      }
    } else {
      mesh = new THREE.Mesh(bg, flatMat);
    }
    texMeshes.push(mesh); scene.add(mesh);
  }
  try { const s = document.querySelector('#hud .sub');
    if(s && D.attribution) s.textContent += ' / ' + D.attribution; } catch(e){}
})();

"""

_TEX_WIRE = r"""// テクスチャ層の配線(ON=実形状テクスチャ / OFF=従来の押出し箱)
(function wireTex(){
  if(!PLATEAU_TEX_DATA) return;
  const el = document.getElementById('lyTex');
  function applyTex(){
    TEX_ON = el ? el.checked : true;
    texMeshes.forEach(o=> o.visible = TEX_ON);
    applyLayers();                       // 押出し箱の可視は TEX_ON で決まる
  }
  if(el) el.onchange = ()=>{ applyTex(); saveSettings(); };
  applyTex();
})();

"""


# ============================================================ 地形(terrain_web.json)
def _inject_terrain(html: str, terrain_json: str) -> str:
    """terrain_web.json 注入。TERRAIN を張り地形メッシュ生成+OSM をドレープ。
    無注入時は groundAt≡0 なので道路/エージェントの座標は従来と数値一致(接地は no-op)。"""
    data_tag = ('<script type="application/json" id="terrain-data">'
                + _json_for_script(terrain_json) + "</script>")
    anchor_data = '<script type="application/json" id="scene-data">'
    html = _replace_once(html, anchor_data, data_tag + "\n" + anchor_data, "terrain-data")
    # buildOsmGround の直後(= buildBuildings の直前)。ここで TERRAIN を張れば
    # 後続の buildRoads / placeAgents が地表高を拾い、OSM.mesh も既に在るのでドレープできる。
    anchor_setup = "// ---------- 建物(kind ごとにジオメトリを統合 = 少ないドローコール)"
    html = _replace_once(html, anchor_setup, _TERRAIN_SETUP + anchor_setup, "terrain-setup")
    return html


_TERRAIN_SETUP = r"""// ---------- 地形起伏(terrain_web.json)。TERRAIN を張り、地形サーフェスと OSM ドレープを作る。
(function setupTerrain(){
  const el = document.getElementById('terrain-data'); if(!el) return;
  let T; try { T = JSON.parse(el.textContent); } catch(e){ console.warn('terrain parse failed', e); return; }
  const b64 = s => { const bin = atob(s); const u = new Uint8Array(bin.length);
    for(let i=0;i<bin.length;i++) u[i] = bin.charCodeAt(i); return u; };
  const H = new Int16Array(b64(T.heights_b64).buffer);
  TERRAIN = { x0:T.x0, y0:T.y0, cell:T.cell_m, nx:T.nx, ny:T.ny, quant:(T.quant||0.1), H:H };
  const nx=T.nx, ny=T.ny, cell=T.cell_m;
  const cx = T.x0 + (nx-1)*cell/2, cy = T.y0 + (ny-1)*cell/2;
  // 起伏サーフェス(OSM オフでも地形が見えるニュートラルな地面)
  const geo = new THREE.PlaneGeometry((nx-1)*cell, (ny-1)*cell, nx-1, ny-1);
  geo.rotateX(-Math.PI/2);
  const pa = geo.attributes.position;
  for(let k=0;k<pa.count;k++){ const wx=cx+pa.getX(k), wy=cy-pa.getZ(k); pa.setY(k, groundAt(wx,wy)); }
  pa.needsUpdate = true; geo.computeVertexNormals();
  const tmesh = new THREE.Mesh(geo, new THREE.MeshLambertMaterial({ color:0x4a5468 }));
  tmesh.position.set(cx, 0, -cy); tmesh.renderOrder = -3;
  window.terrainMesh = tmesh; scene.add(tmesh);
  if(flatGround) flatGround.visible = false;   // 平面地面は起伏へ置換
  if(gridHelper) gridHelper.visible = false;   // 地形があるとグリッドは地中/地上で乱れるだけ
  // OSM 地図テクスチャを **地形サーフェスそのもの**(同一 geometry)へ平面 UV 投影で貼る。
  // 旧実装は「地形とは別のドレープ平面」を 240 セグメント上限で作っていたため、実効間隔
  // 14.5m ≫ 地形格子 2m となり、面積の 1/3 が不透明な地形サーフェスの下に潜って
  // 地図が溶けて見えた。頂点を共有すれば交差は構造的に起こり得ない(残るのは同一平面の
  // z-fight だけ = polygonOffset + depthWrite:false + renderOrder で決着する)。
  try { if(OSM.mesh && OSM.mesh.geometry.parameters){
    const p = OSM.mesh.geometry.parameters;
    const ox0 = OSM.mesh.position.x - p.width/2,   ox1 = OSM.mesh.position.x + p.width/2;
    const oy0 = -OSM.mesh.position.z - p.height/2, oy1 = -OSM.mesh.position.z + p.height/2;
    const uv = geo.attributes.uv;
    for(let k=0;k<pa.count;k++){ const wx=cx+pa.getX(k), wy=cy-pa.getZ(k);
      uv.setXY(k, (wx-ox0)/(ox1-ox0), (wy-oy0)/(oy1-oy0)); }   // 地図矩形の外は uv∉[0,1]
    uv.needsUpdate = true;
    const dmat = OSM.mesh.material;      // 既存マテリアル(map/opacity/配線)をそのまま使う
    dmat.polygonOffset = true; dmat.polygonOffsetFactor = -1; dmat.polygonOffsetUnits = -1;
    const dm = new THREE.Mesh(geo, dmat);
    dm.position.copy(tmesh.position); dm.renderOrder = -1; dm.visible = OSM.mesh.visible;
    scene.remove(OSM.mesh); OSM.mesh.geometry.dispose();
    scene.add(dm); OSM.mesh = dm;        // 以後 applyLayers/不透明度スライダはそのまま効く
  } } catch(e){ console.warn('OSM drape failed', e); }
})();

"""


# ============================================================ 地下街/歩道橋(extras)
def _inject_extras(html: str) -> str:
    """plateau_web.extras(ubld=地下街・brid=歩道橋)を注入。実行時 PLATEAU_DATA.extras
    が無ければメッシュ 0 個。パネルにトグル「地下街」「歩道橋」を追加し applyExtras で配線。"""
    anchor_panel = ('      <label class="chk"><input type="checkbox" id="lyLabels" checked>'
                    ' ラベル(建物名)</label>')
    # 地下街は既定 OFF: ubld の z 上端は +2.79m あり、地形の 81.7% がそれより低いため
    # ON のままだと「地面から半透明の箱が生えている」絵になる(A-2)。
    panel_add = ('      <label class="chk"><input type="checkbox" id="lyUgai">'
                 ' 地下街</label>\n'
                 '      <label class="chk"><input type="checkbox" id="lyBridge" checked>'
                 ' 歩道橋</label>\n')
    html = _replace_once(html, anchor_panel, panel_add + anchor_panel, "extras-panel")
    anchor_build = "// ---------- 道路(全ポリラインを 1 本の LineSegments に統合)"
    html = _replace_once(html, anchor_build, _EXTRAS_BUILD + anchor_build, "extras-build")
    anchor_wire = "// ---------- ループ"
    html = _replace_once(html, anchor_wire, _EXTRAS_WIRE + anchor_wire, "extras-wire")
    return html


_EXTRAS_BUILD = r"""// ---------- 地下街(半透明)/ 歩道橋(不透明)= plateau_web.extras(int16×0.05m・<u4)
const ugaiMeshes = [], bridgeMeshes = [];
(function buildExtras(){
  if(!(typeof PLATEAU_DATA !== 'undefined' && PLATEAU_DATA && PLATEAU_DATA.extras)) return;
  const ex = PLATEAU_DATA.extras;
  const b64 = s => { const bin=atob(s); const u=new Uint8Array(bin.length);
    for(let i=0;i<bin.length;i++) u[i]=bin.charCodeAt(i); return u; };
  function meshOf(part, color, opacity, clipGround){
    if(!part || !part.positions_b64 || !part.indices_b64) return null;
    const q = part.quant_scale || 0.05;
    const pi = new Int16Array(b64(part.positions_b64).buffer);
    const pos = new Float32Array(pi.length);
    for(let i=0;i<pi.length;i+=3){ pos[i]=pi[i]*q; pos[i+1]=pi[i+2]*q; pos[i+2]=-pi[i+1]*q; }
    let idx = new Uint32Array(b64(part.indices_b64).buffer);
    // 地表クリップ(A-2): 3 頂点すべてが地表より上にある三角形を落とす。地下構造物の
    // 天板が地形を突き抜けて「地面から生えて」見えるのを、描画順ではなく形状で止める。
    if(clipGround && TERRAIN){
      const keep = []; let cut = 0;
      for(let i=0;i<idx.length;i+=3){
        let above = 0;
        for(let k=0;k<3;k++){ const v = idx[i+k]*3;
          if(pos[v+1] > groundAt(pos[v], -pos[v+2]) + 0.2) above++; }
        if(above === 3){ cut++; continue; }
        keep.push(idx[i], idx[i+1], idx[i+2]);
      }
      if(cut){ console.info('extras: 地表より上の三角形を', cut, '枚クリップ');
        idx = new Uint32Array(keep); }
      if(idx.length === 0) return null;
    }
    const bg = new THREE.BufferGeometry();
    bg.setAttribute('position', new THREE.BufferAttribute(pos,3));
    bg.setIndex(new THREE.BufferAttribute(idx,1)); bg.computeVertexNormals();
    const mat = new THREE.MeshLambertMaterial({ color:color, side:THREE.DoubleSide,
      transparent:(opacity<1.0), opacity:opacity, depthWrite:(opacity>=1.0) });
    return new THREE.Mesh(bg, mat);
  }
  const u = meshOf(ex.ubld, 0x6f7fa8, 0.35, true);   // 地下街=地下色・半透明・地表クリップ
  if(u){ u.renderOrder = -4; ugaiMeshes.push(u); scene.add(u); }  // 地形より先に描く
  const br = meshOf(ex.brid, 0xb8bec7, 1.0, false);   // 歩道橋=無彩色・不透明(地上構造物)
  if(br){ bridgeMeshes.push(br); scene.add(br); }
})();

"""

# ============================================================ 地下街 LOD4.1(梅 / B-2)
def _inject_ubld4(html: str) -> str:
    """plateau_web.ubld4(面種別 kind + 層タグ付きメッシュ)を注入する。

    _inject_extras の**後**に呼ぶ(旧 extras.ubld の箱表示行を潰すため)。ubld4 が無い
    plateau_web(= 旧ラン)では呼ばれない → 生成 HTML は従来とバイト同一。
    ①層チップ ②旧 ubld 箱の停止 ③面種別メッシュ構築+配線 の 3 箇所を一意置換。"""
    anchor_panel = ('      <label class="chk"><input type="checkbox" id="lyUgai">'
                    ' 地下街</label>\n')
    html = _replace_once(html, anchor_panel, anchor_panel + _UBLD4_PANEL, "ubld4-panel")
    anchor_old = ("  const u = meshOf(ex.ubld, 0x6f7fa8, 0.35, true);"
                  "   // 地下街=地下色・半透明・地表クリップ")
    html = _replace_once(
        html, anchor_old,
        "  const u = null;   // ubld4(LOD4.1 面種別メッシュ)が在るので旧 extras.ubld の箱は描かない",
        "ubld4-old-off")
    anchor_build = "// ---------- 道路(全ポリラインを 1 本の LineSegments に統合)"
    html = _replace_once(html, anchor_build, _UBLD4_BUILD + anchor_build, "ubld4-build")
    return html


_UBLD4_PANEL = ('      <div class="op seg" id="ub4Chips" style="flex-wrap:wrap; gap:4px;">'
                '</div>\n')

_UBLD4_BUILD = r"""// ---------- 地下街 LOD4.1(plateau_web.ubld4)= 面種別の塗り分け + 層別表示
// 床/地面=不透明・内壁/壁/仕切=半透明・扉/窓=強調色・階段等(installation)=別色・
// 天井/屋根=ごく薄い蓋。層(レーンA が床面 z のピークで分離した 4 層)ごとに出し分ける。
const ubld4Meshes = [];                     // [{layer, mesh}]
(function buildUbld4(){
  const U = (typeof PLATEAU_DATA!=='undefined' && PLATEAU_DATA) ? PLATEAU_DATA.ubld4 : null;
  if(!U) return;
  const b64 = s => { const bin=atob(s); const u=new Uint8Array(bin.length);
    for(let i=0;i<bin.length;i++) u[i]=bin.charCodeAt(i); return u; };
  const q = U.quant_scale || 0.05;
  const pi = new Int16Array(b64(U.positions_b64).buffer);
  const pos = new Float32Array(pi.length);
  for(let i=0;i<pi.length;i+=3){          // local-m (e,n,up) -> three (e,up,-n)
    pos[i] = pi[i]*q; pos[i+1] = pi[i+2]*q; pos[i+2] = -pi[i+1]*q; }
  const idx = new Uint32Array(b64(U.indices_b64).buffer);
  const kind = b64(U.tri_kind_b64), lay = b64(U.tri_layer_b64);
  const names = U.kind_names || [];
  const GRP = [ {key:'floor', color:0x9aa7c4, op:1.00},
                {key:'wall',  color:0x6f7fa8, op:0.32},
                {key:'door',  color:0xffb454, op:0.95},
                {key:'inst',  color:0x3ba89c, op:0.80},
                {key:'lid',   color:0x5a6580, op:0.16} ];
  const OF = { floor:0, ground:0, interior_wall:1, wall:1, closure:1,
               door:2, window:2, installation:3, ceiling:4, roof:4, other:4 };
  // 地表クリップ(旧 extras と同一規則): 3 頂点すべてが地表より上の三角形は落とす。
  // 新データの z 基準は健全(地表より上の頂点 2.85% / 旧 47.37%)が、残る地上突出=
  // 地上への階段の天端が「地面から生えて」見えるため規則は維持する。
  const buckets = new Map(); let cut = 0;
  const nt = (idx.length/3)|0;
  for(let ti=0; ti<nt; ti++){
    const a=idx[ti*3]*3, b=idx[ti*3+1]*3, c=idx[ti*3+2]*3;
    if(TERRAIN){ let above = 0;
      if(pos[a+1] > groundAt(pos[a], -pos[a+2]) + 0.2) above++;
      if(pos[b+1] > groundAt(pos[b], -pos[b+2]) + 0.2) above++;
      if(pos[c+1] > groundAt(pos[c], -pos[c+2]) + 0.2) above++;
      if(above === 3){ cut++; continue; } }
    const nm = names[kind[ti]];
    const gi = (OF[nm] !== undefined) ? OF[nm] : 4;
    const key = lay[ti]*8 + gi;
    let arr = buckets.get(key); if(!arr){ arr = []; buckets.set(key, arr); }
    arr.push(pos[a],pos[a+1],pos[a+2], pos[b],pos[b+1],pos[b+2], pos[c],pos[c+1],pos[c+2]);
  }
  if(cut) console.info('ubld4: 地表より上の三角形を', cut, '枚クリップ');
  const mats = GRP.map(g => new THREE.MeshLambertMaterial({ color:g.color,
    side:THREE.DoubleSide, transparent:(g.op<1.0), opacity:g.op, depthWrite:(g.op>=1.0) }));
  for(const key of [...buckets.keys()].sort((a,b)=>a-b)){
    const arr = buckets.get(key); if(!arr.length) continue;
    const bg = new THREE.BufferGeometry();
    bg.setAttribute('position', new THREE.Float32BufferAttribute(arr, 3));
    bg.computeVertexNormals();                       // 非索引=フラット法線(建築面向き)
    const mesh = new THREE.Mesh(bg, mats[key % 8]);
    mesh.renderOrder = -4;                           // 地形(-3)より先に描く
    ubld4Meshes.push({ layer:(key/8)|0, mesh }); scene.add(mesh);
  }
  // ---- 層チップ(全部 / B1..Bn)+ 地下街トグル連動 ----
  const chips = document.getElementById('ub4Chips');
  const LY = U.layers || [];
  let selLayer = -1;
  function applyUbld4(){
    const eu = document.getElementById('lyUgai');
    const on = eu ? eu.checked : false;
    for(const r of ubld4Meshes) r.mesh.visible = on && (selLayer < 0 || r.layer === selLayer);
    // カットアウェイ: 地下は不透明な地表の下にあるので、ON の間だけ地表を薄くし
    // OSM ドレープを退避する(OFF で applyLayers の状態へ戻す)。
    const tm = window.terrainMesh;
    if(tm){ tm.material.transparent = on; tm.material.opacity = on ? 0.28 : 1.0;
      tm.material.depthWrite = !on; tm.material.needsUpdate = true; }
    if(typeof OSM !== 'undefined' && OSM.mesh){
      const eo = document.getElementById('lyOsm');
      OSM.mesh.visible = on ? false : (eo ? eo.checked : true); }
  }
  if(chips){
    let h = '<span style="color:#9aa4b2">層</span>';
    h += '<button data-l="-1" class="on" style="padding:2px 6px;font-size:11px">全</button>';
    for(let i=LY.length-1; i>=0; i--)     // 浅い層から B1, B2, ...
      h += '<button data-l="'+i+'" style="padding:2px 6px;font-size:11px" title="z='
        + LY[i].z + 'm / ' + LY[i].n_triangles + '面">B' + (LY.length-i) + '</button>';
    chips.innerHTML = h;
    chips.querySelectorAll('button').forEach(btn=>{ btn.onclick = ()=>{
      selLayer = parseInt(btn.dataset.l, 10);
      chips.querySelectorAll('button').forEach(b2=> b2.classList.remove('on'));
      btn.classList.add('on'); applyUbld4(); }; });
  }
  const eu0 = document.getElementById('lyUgai');
  if(eu0) eu0.addEventListener('change', applyUbld4);   // _EXTRAS_WIRE の onchange と共存
  applyUbld4();
  try { const s = document.querySelector('#hud .sub');
    if(s && PLATEAU_DATA.attribution && s.textContent.indexOf('LOD4.1') < 0)
      s.textContent += ' / 地下街 LOD4.1'; } catch(e){}
})();

"""


_EXTRAS_WIRE = r"""// 地下街/歩道橋トグルの配線(applyLayers を触らず独立関数で)
(function wireExtras(){
  function applyExtras(){
    const eu=document.getElementById('lyUgai'), eb=document.getElementById('lyBridge');
    if(eu) ugaiMeshes.forEach(o=> o.visible = eu.checked);
    if(eb) bridgeMeshes.forEach(o=> o.visible = eb.checked);
  }
  ['lyUgai','lyBridge'].forEach(id=>{ const el=document.getElementById(id);
    if(el) el.onchange = ()=>{ applyExtras(); }; });
  applyExtras();
})();
"""


# ============================================================ 移動手段(mode_legend)
def _inject_modes(html: str) -> str:
    """tracks.meta.mode_legend がある時のみ: 移動中エージェントを手段別に色/形分け
    (徒歩=カプセル現行 / 自転車=緑 / 車・タクシー=箱グリフ)。HUD に凡例+電車人数。"""
    anchor_hud = '  <div class="hint">ドラッグ=回転'
    hud_add = ('  <div id="modeLegend" style="margin-top:8px; font-size:11px;'
               ' line-height:1.6;"></div>\n')
    html = _replace_once(html, anchor_hud, hud_add + anchor_hud, "mode-hud")
    anchor_build = "// ---------- ループ"
    html = _replace_once(html, anchor_build, _MODES_BUILD + anchor_build, "mode-build")
    return html


_MODES_BUILD = r"""// ---------- 移動手段(feature5)。TRACKS.meta.mode_legend がある時だけ動く。
(function setupModes(){
  const LEG = (TRACKS.meta && TRACKS.meta.mode_legend) || null; if(!LEG) return;
  const MC = { 0:0x54a0ff, 1:0x22c55e, 2:0xf2b134, 3:0xe0559d };  // 徒歩/自転車/車/タクシー
  const veh = new THREE.InstancedMesh(new THREE.BoxGeometry(4.6,2.2,2.4),
    new THREE.MeshLambertMaterial({ color:0xffffff }), NA);
  veh.instanceMatrix.setUsage(THREE.DynamicDrawUsage); veh.count = NA; scene.add(veh);
  const _cc=new THREE.Color(), _mm=new THREE.Matrix4(), _pp=new THREE.Vector3(),
        _qq=new THREE.Quaternion(), _ss=new THREE.Vector3(1,1,1), _hh=new THREE.Vector3(0,0,0);
  function modeAt3(t,i){ const s0=Math.floor(t);
    const nm=(s0+1<NS)? TRACKS.moves[s0+1][i] : null; return nm? nm[0] : -1; }
  modeTick = function(t){
    const pos = posAt(t);
    for(let i=0;i<NA;i++){
      const w = pos[i][2], moving = (w===0), m = moving? modeAt3(t,i) : -1;
      if(moving && (m===2 || m===3)){                 // 車/タクシー: カプセルを隠し箱で描画
        _mm.compose(_pp.set(0,-9999,0),_qq,_hh); agents.setMatrixAt(i,_mm);
        const x=pos[i][0], y=pos[i][1];
        _pp.set(x, groundAt(x,y)+1.1, -y); _mm.compose(_pp,_qq,_ss); veh.setMatrixAt(i,_mm);
        _cc.setHex(MC[m]); veh.setColorAt(i,_cc);
      } else {                                         // 徒歩/自転車/静止: 箱を隠しカプセルを着色
        _mm.compose(_pp.set(0,-9999,0),_qq,_hh); veh.setMatrixAt(i,_mm);
        _cc.setHex(m===1 ? MC[1] : agentColor(i)); agents.setColorAt(i,_cc);
      }
    }
    agents.instanceMatrix.needsUpdate = true;
    if(agents.instanceColor) agents.instanceColor.needsUpdate = true;
    veh.instanceMatrix.needsUpdate = true;
    if(veh.instanceColor) veh.instanceColor.needsUpdate = true;
  };
  const legEl = document.getElementById('modeLegend'); let _hn = 0;
  hudTick = function(t){
    if(!legEl || (_hn++ % 15) !== 0) return;          // HUD は ~4回/秒に間引き
    const pos = posAt(t); let train=0; const cnt={0:0,1:0,2:0,3:0};
    for(let i=0;i<NA;i++){ const w=pos[i][2];
      if(w===-3){ train++; continue; }
      if(w===0){ const m=modeAt3(t,i); if(cnt[m]!==undefined) cnt[m]++; } }
    let h='<div style="color:#9aa4b2;margin-bottom:2px">移動手段(移動中)</div>';
    for(const k of Object.keys(LEG)){ const c='#'+_cc.setHex(MC[k]||0xffffff).getHexString();
      h+='<div><span style="display:inline-block;width:10px;height:10px;border-radius:2px;'
        +'background:'+c+';margin-right:6px;vertical-align:-1px"></span>'+LEG[k]+' '+(cnt[k]||0)+'</div>'; }
    if(train>0) h+='<div style="margin-top:3px;color:#c7cdd6">🚃 電車移動中 '+train+'人</div>';
    legEl.innerHTML = h;
  };
})();

"""


# ============================================================ 顕著イベント(notable)
def _inject_notable(html: str, notable_json: str) -> str:
    """顕著イベントパネルを後付け注入(データ無し時は呼ばれない=既定出力はバイト同一)。
    ①notable-data ②CSS ③右スタックのパネル ④リスト構築+クリック配線 の 4 箇所を一意置換。"""
    data_tag = ('<script type="application/json" id="notable-data">'
                + _json_for_script(notable_json) + "</script>")
    anchor_data = '<script type="application/json" id="scene-data">'
    html = _replace_once(html, anchor_data, data_tag + "\n" + anchor_data, "notable-data")
    anchor_css = "</style>"
    html = _replace_once(html, anchor_css, _NOTABLE_CSS + anchor_css, "notable-css")
    anchor_panel = '  <div id="legend" class="panel"></div>'
    html = _replace_once(html, anchor_panel, anchor_panel + "\n" + _NOTABLE_PANEL,
                         "notable-panel")
    anchor_build = "// ---------- ループ"
    html = _replace_once(html, anchor_build, _NOTABLE_BUILD + anchor_build, "notable-build")
    return html


_NOTABLE_CSS = r"""  #notable { position:relative; }
  #notable .hdr { font-size:10.5px; letter-spacing:.12em; text-transform:uppercase; color:#9aa4b2;
    cursor:pointer; margin-bottom:8px; user-select:none; display:flex; align-items:center; gap:6px; }
  #notable .hdr #ntCaret { margin-left:auto; }
  #notable #ntCount { color:#c7cdd6; letter-spacing:0; text-transform:none; }
  #notable #ntList { max-height:320px; overflow-y:auto; margin:-2px -4px; padding:0 2px; }
  #notable .nt-row { display:flex; align-items:center; gap:6px; padding:3px 4px; border-radius:5px;
    cursor:pointer; font-size:11px; line-height:1.35; }
  #notable .nt-row:hover { background:rgba(255,255,255,.08); }
  #notable .nt-row.on { background:rgba(59,130,246,.30); }
  #notable .nt-dot { width:8px; height:8px; border-radius:50%; flex:0 0 auto; }
  #notable .nt-t { color:#9aa4b2; font-variant-numeric:tabular-nums; flex:0 0 auto; }
  #notable .nt-k { color:#e6e9ee; flex:0 0 auto; font-weight:600; }
  #notable .nt-x { color:#c7cdd6; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
"""

_NOTABLE_PANEL = r"""  <div id="notable" class="panel">
    <div class="hdr" id="ntHdr">顕著イベント <span id="ntCount"></span><span id="ntCaret">&#9662;</span></div>
    <div id="ntBody"><div id="ntList"></div></div>
  </div>"""

_NOTABLE_BUILD = r"""// ========== 顕著イベント(notable-data 注入時のみ動く)。クリックで時間ジャンプ+カメラ移動。
(function setupNotable(){
  const el = document.getElementById('notable-data'); if(!el) return;
  let ND; try { ND = JSON.parse(el.textContent); } catch(e){ console.warn('notable parse failed', e); return; }
  const evs = (ND && ND.events) || []; if(!evs.length) return;
  const listEl = document.getElementById('ntList'); if(!listEl) return;
  const cntEl = document.getElementById('ntCount'); if(cntEl) cntEl.textContent = evs.length + '件';
  const IMP_COLOR = { 5:'#ef4444', 4:'#f59e0b', 3:'#eab308', 2:'#60a5fa', 1:'#9aa4b2' };
  const esc = s => String(s == null ? '' : s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  const playBtn = document.getElementById('play');
  function jump(ev){
    // ① 時間スライダをそのフレームへ(自動再生は止める)
    t = Math.max(0, Math.min(NS-1, ev.frame|0));
    playing = false; if(playBtn) playBtn.textContent = '▶︎';
    timeline.value = String(t);
    const min = simMinAt(t); clockEl.textContent = fmtClock(min);
    if(dayNight) updateSky(min);
    placeAgents(t); placeCars(t);
    // ② カメラをイベント発生位置へ(位置があるものだけ)
    if(ev.has_pos){
      const gx = ev.x, gy = ev.y, gz = groundAt(gx, gy);
      controls.target.set(gx, gz + 6, -gy);
      camera.position.set(gx + 95, gz + 135, -gy + 95);
      controls.update();
    }
  }
  let h = '';
  for(let i=0;i<evs.length;i++){
    const ev = evs[i];
    const c = IMP_COLOR[ev.importance] || '#9aa4b2';
    const txt = ev.text ? (' — ' + esc(ev.text)) : '';
    h += '<div class="nt-row" data-i="'+i+'" title="'+esc(ev.label)+esc(txt)+'">'
      + '<span class="nt-dot" style="background:'+c+'"></span>'
      + '<span class="nt-t">'+esc(fmtClock(ev.sim_min))+'</span>'
      + '<span class="nt-k">'+esc(ev.label)+'</span>'
      + '<span class="nt-x">'+txt+'</span></div>';
  }
  listEl.innerHTML = h;
  const rows = listEl.querySelectorAll('.nt-row');
  rows.forEach(row=>{ row.onclick = ()=>{
    jump(evs[Number(row.dataset.i)]);
    rows.forEach(r=> r.classList.remove('on')); row.classList.add('on'); }; });
  // ヘッダで折り畳み
  const hdr = document.getElementById('ntHdr'), caret = document.getElementById('ntCaret');
  if(hdr) hdr.onclick = ()=>{ const b = document.getElementById('ntBody');
    const off = b.style.display === 'none'; b.style.display = off ? 'block' : 'none';
    if(caret) caret.innerHTML = off ? '&#9662;' : '&#9656;'; };
})();

"""


# ============================================================ 屋内オーバレイ(B6)
def _has_space_move(ev_p: Path) -> bool:
    """l1_events.parquet に space_move が 1 件でもあるか(kind 列だけ読む)。"""
    if not ev_p.exists():
        return False
    try:
        import pyarrow.parquet as pq
        tbl = pq.read_table(ev_p, columns=["kind"])
        return "space_move" in set(tbl.column("kind").to_pylist())
    except Exception:
        return False


def _indoor_positions_from_samples(samples_p: Path, tracks_json: str):
    """indoor_tracks_samples.parquet の実座標(x,y)を viewer フレーム別に畳む。

    t_s → step = floor(t_s / (step_minutes*60))、step → frame は export の
    emitted=range(0,n,stride) と同型(stride=1 は恒等)。(frame, agentIdx) ごとに
    最大 t_s(その step 終端に最も近い)サンプルを採る。ids に無い agent は捨てる
    (--sample-agents 間引き整合)。返り: {"frames": {"<frame>": [[idx,x,y]...]}} or None。"""
    try:
        import pyarrow.parquet as pq
        tbl = pq.read_table(samples_p, columns=["agent_id", "t_s", "x", "y"])
    except Exception:
        return None
    try:
        tj = json.loads(tracks_json)
    except Exception:
        return None
    ids = tj.get("ids") or []
    id_to_idx = {int(a): i for i, a in enumerate(ids)}
    meta = tj.get("meta", {})
    n_frames = int(meta.get("nSteps", 0) or 0)
    stride = max(1, int(meta.get("step_stride", 1) or 1))
    step_secs = max(1, int(meta.get("step_minutes", 10) or 10)) * 60
    aid = tbl.column("agent_id").to_pylist()
    ts = tbl.column("t_s").to_pylist()
    xs = tbl.column("x").to_pylist()
    ys = tbl.column("y").to_pylist()
    best: dict = {}                                  # (frame, idx) -> (t_s, x, y)
    for a, t, x, y in zip(aid, ts, xs, ys):
        idx = id_to_idx.get(int(a))
        if idx is None:
            continue
        step = int(float(t) // step_secs)
        frame = step // stride if stride > 1 else step
        if n_frames and frame > n_frames - 1:
            frame = n_frames - 1
        if frame < 0:
            frame = 0
        k = (frame, idx)
        prev = best.get(k)
        if prev is None or float(t) > prev[0]:
            best[k] = (float(t), round(float(x), 1), round(float(y), 1))
    if not best:
        return None
    frames: dict = {}
    for (frame, idx), (_t, x, y) in best.items():
        frames.setdefault(frame, []).append([idx, x, y])
    out = {str(f): sorted(frames[f]) for f in sorted(frames)}
    return {"frames": out}


def _ensure_indoor(run_dir: Path, tracks_json: str) -> str | None:
    """屋内オーバレイの埋め込みデータを組み立てる(注入ゲート)。

    ゲート = 新ラン信号(space_move が L1 にある OR indoor_tracks_samples.parquet がある)。
    無ければ None = 一切注入しない = 旧ランは生成 HTML がバイト同一。有れば
    data/floor_layouts.json(フロア板用の間取り正典 spec)と、samples があれば実座標
    (フレーム別)を 1 つの JSON にまとめて返す。透明性のため scene3d へ書き出す。"""
    samples_p = run_dir / "indoor_tracks_samples.parquet"
    has_samples = samples_p.exists()
    if not (has_samples or _has_space_move(run_dir / "l1_events.parquet")):
        return None
    payload: dict = {}
    fl_p = REPO_ROOT / "data" / "floor_layouts.json"
    if fl_p.exists():
        try:
            blds = json.loads(fl_p.read_text(encoding="utf-8")).get("buildings")
            if isinstance(blds, list):
                payload["floor_layouts"] = {"buildings": blds}   # meta は viewer 不要=同梱しない
        except Exception:
            pass
    if has_samples:
        pos = _indoor_positions_from_samples(samples_p, tracks_json)
        if pos is not None:
            payload["positions"] = pos
    if not payload:
        return None
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    try:
        out_p = run_dir / "scene3d" / "indoor_overlay.json"
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(text, encoding="utf-8")
    except Exception:
        pass
    n_bld = len(payload.get("floor_layouts", {}).get("buildings", []))
    n_fr = len(payload.get("positions", {}).get("frames", {})) if "positions" in payload else 0
    print(f"  [indoor] フロア板 spec {n_bld}棟"
          f"{' + 実座標フレーム ' + str(n_fr) if 'positions' in payload else ''} を注入")
    return text


def _inject_indoor(html: str, indoor_json: str) -> str:
    """屋内オーバレイを後付け注入(データ無し時は呼ばれない=既定出力はバイト同一)。
    ①indoor-data ②レイヤートグル ③名寄せ+skip宣言 ④merged-mesh の skip 一行
    ⑤本体JS(floorLayout3d/フロア板/接近フェード/実座標) ⑥placeAgents の屋内分岐。"""
    data_tag = ('<script type="application/json" id="indoor-data">'
                + _json_for_script(indoor_json) + "</script>")
    anchor_data = '<script type="application/json" id="scene-data">'
    html = _replace_once(html, anchor_data, data_tag + "\n" + anchor_data, "indoor-data")
    # レイヤーパネルにトグル(昼夜の直後)
    anchor_toggle = ('      <label class="chk"><input type="checkbox" id="lyDayNight" checked>'
                     ' 昼夜ライティング</label>')
    html = _replace_once(html, anchor_toggle,
                         anchor_toggle + "\n" + _INDOOR_TOGGLE, "indoor-toggle")
    # 名寄せ+INDOOR_SKIP 宣言(buildBuildings の直前・PLATEAU_DECL と同じアンカー=共存)
    anchor_decl = "(function buildBuildings(){"
    html = _replace_once(html, anchor_decl, _INDOOR_DECL + anchor_decl, "indoor-decl")
    # merged-mesh から屋内建物を外す(PLATEAU skip の有無に依らず const fp 行の直前へ)
    anchor_skip = "    const fp = b.footprint;"
    html = _replace_once(
        html, anchor_skip,
        "    if(typeof INDOOR_SKIP!=='undefined' && INDOOR_SKIP.has(b.id)) continue;\n"
        + anchor_skip, "indoor-skip")
    # 本体(ループ直前=extras/modes/notable と同じアンカー)
    anchor_build = "// ---------- ループ"
    html = _replace_once(html, anchor_build, _INDOOR_MAIN + anchor_build, "indoor-main")
    # placeAgents: 屋内エージェントは実座標/推定区画へ
    anchor_place = ("    _p.set(x, footY(x, y, w) + AG_LIFT, -y);"
                    "   // 足元アンカー(屋外=地表 / 屋内=フロア面)")
    html = _replace_once(html, anchor_place, _INDOOR_PLACE, "indoor-place")
    return html


_INDOOR_TOGGLE = ('      <label class="chk"><input type="checkbox" id="lyIndoor" checked>'
                  ' 屋内(フロア板・接近フェード)</label>')

_INDOOR_PLACE = r"""    let _ax=x, _ay=y;                       // 屋内=実座標(samples)or 推定区画(floorLayout3d)
    if(w>=1000 && typeof window.__indoorXY==='function'){ const _r=window.__indoorXY(i, w, t);
      if(_r){ _ax=_r[0]; _ay=_r[1]; } }
    _p.set(_ax, footY(_ax, _ay, w) + AG_LIFT, -_ay);   // 足元アンカー(屋外=地表/屋内=フロア面)"""

# ---- ③ 名寄せ + skip(トップレベル・buildBuildings より前に実行)----
_INDOOR_DECL = r"""// ========== 屋内オーバレイ: floor_layouts の名寄せ(部分一致・双方向=sim indoor._match と同一)
const INDOOR = (()=>{ try { const el = document.getElementById('indoor-data');
  if(el) return JSON.parse(el.textContent); } catch(e){ console.warn('indoor parse failed', e); } return null; })();
const INDOOR_SPECS = (INDOOR && INDOOR.floor_layouts && INDOOR.floor_layouts.buildings) || [];
function _indoorSpecFor(b){
  const name = (b && (b.name || b.id)) || ''; if(!name) return null;
  for(const rec of INDOOR_SPECS){ for(const m of (rec.match || [])){
    if(m && (name.indexOf(m) >= 0 || m.indexOf(name) >= 0)) return rec; } }
  return null;
}
const INDOOR_BLD = [];                 // [{bi, b, spec}] 名寄せ済み屋内建物
const INDOOR_SKIP = new Set();         // merged-mesh から外す建物 id
if(INDOOR){ SCENE.buildings.forEach((b, bi)=>{ const rec = _indoorSpecFor(b);
  if(rec){ INDOOR_BLD.push({ bi, b, spec:rec }); INDOOR_SKIP.add(b.id); } }); }
"""

# ---- ⑤ 本体(IIFE。indoorXY は window へ公開=placeAgents から参照)----
_INDOOR_MAIN = r"""// ========== 屋内オーバレイ本体: floorLayout3d(n_override) / フロア板(LRU8) / 接近フェード / 実座標
(function setupIndoor(){
  if(!INDOOR) return;
  // ---- 決定論間取り(sim world.vision.building_layout と同一系列: FNV+mulberry32+重み列分割)----
  function _fnv(s){ let h=2166136261>>>0; s=''+s;
    for(let i=0;i<s.length;i++){ h^=s.charCodeAt(i); h=Math.imul(h,16777619); } return h>>>0; }
  function _mul32(seed){ let a=seed>>>0; return ()=>{ a=(a+0x6D2B79F5)>>>0;
    let t=Math.imul(a^(a>>>15),1|a); t=((t+Math.imul(t^(t>>>7),61|t))>>>0)^t;
    return ((t^(t>>>14))>>>0)/4294967296; }; }
  function _cols(a0,a1,n,rng){ if(n<=0) return []; const w=[]; let s=0;
    for(let i=0;i<n;i++){ const v=0.7+rng()*0.7; w.push(v); s+=v; }
    const out=[]; let p=a0; for(const v of w){ const seg=(a1-a0)*v/s; out.push([p,p+seg]); p+=seg; } return out; }
  // vision._POOL_LEN(区画候補プールの長さ)/ _kind_use。override/POI 無しのフォールバック経路のみ使う。
  const _POOL_LEN = { fashion:6, beauty:5, food:6, restaurant:6, lifestyle:6, office:5, shop:4,
    hall:3, theatre:3, hotel:3, station:4, park:3, nightlife:3, service:3, attraction:3, education:3, generic:3 };
  function _kindUse(kind){ return kind==='office'?'office':kind==='station'?'station':kind==='retail'?'shop':'generic'; }
  const _ZONE_HUE = { food:35, shop:210, nightlife:295, service:160, office:220, education:265,
    attraction:48, hotel:330, lifestyle:180, restaurant:20, fashion:200, beauty:320, hall:255,
    theatre:280, station:0, park:130, generic:210 };
  // spec(floor_layouts)由来の n_override(sim indoor._build と同一規則: shops > Σzone_mix > null)
  function _specFloor(b, f){ const rec=_indoorSpecFor(b); if(!rec) return null;
    for(const fl of (rec.floors||[])){ if((fl.f|0)===(f|0)) return fl; } return null; }
  function _nOverride(b, f){ const sp=_specFloor(b,f); if(!sp) return null;
    if(sp.shops) return sp.shops|0;
    if(sp.zone_mix){ let s=0; for(const k in sp.zone_mix) s+=(sp.zone_mix[k]|0); return s; }
    return null; }
  // building_layout(building, floor, n_override) の JS 移植(乱数消費順まで一致)
  function floorLayout3d(b, f){
    const fp=b.footprint; if(!fp || fp.length<3) return null;
    const xs=fp.map(p=>p[0]), ys=fp.map(p=>p[1]);
    const x0=Math.min.apply(null,xs), x1=Math.max.apply(null,xs);
    const y0=Math.min.apply(null,ys), y1=Math.max.apply(null,ys);
    const w=x1-x0, h=y1-y0; if(w<=0 || h<=0) return null;
    const cx=(x0+x1)/2, cy=(y0+y1)/2;
    const rng=_mul32((_fnv(b.id||b.name||'b')+(f+50)*2654435761)>>>0);
    const nov=_nOverride(b,f); let n;
    if(nov!=null){ n=Math.min(12, nov|0); }              // override 経路: cols 前に乱数を消費しない
    else {                                               // pool 経路(POI 無しの決定論生成)
      const use=_kindUse(b.kind); const poolN=_POOL_LEN[use]||_POOL_LEN.generic;
      n=2+Math.floor(rng()*Math.min(4, poolN-1));
      const used=new Set();
      for(let k=0;k<n;k++){ let g=0, idx;
        while(true){ idx=Math.floor(rng()*poolN); const cont=used.has(idx)&&(g<8); g++; if(!cont) break; }
        used.add(idx); } }
    if(n<=0) n=1;
    const horiz=w>=h, band=(horiz?h:w)*0.09, nA=Math.ceil(n/2), nB=n-nA;
    const zones=[]; let corridor;
    if(horiz){ const yc0=cy-band, yc1=cy+band;
      _cols(x0,x1,nA,rng).forEach(c=>zones.push([c[0],yc1,c[1],y1]));
      _cols(x0,x1,nB,rng).forEach(c=>zones.push([c[0],y0,c[1],yc0]));
      corridor=[x0,yc0,x1,yc1];
    } else { const xc0=cx-band, xc1=cx+band;
      _cols(y0,y1,nA,rng).forEach(c=>zones.push([xc1,c[0],x1,c[1]]));
      _cols(y0,y1,nB,rng).forEach(c=>zones.push([x0,c[0],xc0,c[1]]));
      corridor=[xc0,y0,xc1,y1]; }
    const cs=Math.min(w,h)*0.13;
    const core=[cx-cs/2, cy-cs/2, cx+cs/2, cy+cs/2];
    return { bbox:[x0,y0,x1,y1], corridor, core, zones, horiz, nA, nB };
  }
  const _layCache = new Map();
  function _layoutFor(bi, f){ const key=bi+':'+f; if(_layCache.has(key)) return _layCache.get(key);
    const b=SCENE.buildings[bi]; const lay=b? floorLayout3d(b, f) : null; _layCache.set(key, lay); return lay; }
  // ---- SpaceType 反映(zone_mix があれば面積降順+キー昇順で敷く=sim _types_from_mix と同型)----
  function _area(z){ return Math.abs((z[2]-z[0])*(z[3]-z[1])); }
  function _zoneCats(lay, b, f){
    const sp=_specFloor(b,f), zones=lay.zones, n=zones.length;
    if(sp && sp.zone_mix){
      const order=zones.map((z,i)=>i).sort((a,c)=> (_area(zones[c])-_area(zones[a])) || (a-c));
      const keys=Object.keys(sp.zone_mix).sort(); const seq=[];
      for(const k of keys){ const cnt=sp.zone_mix[k]|0; for(let j=0;j<Math.max(0,cnt);j++) seq.push(k); }
      const out=new Array(n); const fill=seq.length? seq[seq.length-1] : (sp.use||'generic');
      for(let pos=0;pos<order.length;pos++){ const zi=order[pos]; out[zi]= pos<seq.length? seq[pos] : fill; }
      return out;
    }
    const use=(sp&&sp.use) || _kindUse(b.kind);
    return new Array(n).fill(use);
  }
  // ---- 実座標(samples): frame -> (agentIdx -> [x,y]) ----
  const INDOOR_POS = new Map();
  if(INDOOR.positions && INDOOR.positions.frames){ const F=INDOOR.positions.frames;
    for(const k in F){ const m=new Map(); for(const e of F[k]) m.set(e[0], [e[1], e[2]]); INDOOR_POS.set(parseInt(k,10), m); } }
  // 屋内エージェント位置: 実座標優先→無ければ推定区画(2D _agentSpot と同じ hash 選択・微動なし)
  window.__indoorXY = function(i, w, tt){
    const s0=Math.floor(tt); const fm=INDOOR_POS.get(s0);
    if(fm){ const r=fm.get(i); if(r) return r; }
    const bi=Math.floor((w-1000)/100), f=((w-1000)%100);
    const b=SCENE.buildings[bi]; if(!b) return null;
    const lay=_layoutFor(bi, f); if(!lay || !lay.zones.length) return null;
    const id=TRACKS.ids[i];
    const z=lay.zones[_fnv('a'+id)%lay.zones.length];
    const rng=_mul32(_fnv('p'+id+':'+(b.id||b.name)+':'+f));
    const u=0.2+rng()*0.6, v=0.2+rng()*0.6;
    return [z[0]+(z[2]-z[0])*u, z[1]+(z[3]-z[1])*v];
  };
  // ---- 外殻シェル(屋内建物ごと・独自マテリアル=個別フェード可。PLATEAU 照合建物は作らない)----
  const indoorShells = [];               // {bi, b, mesh, mat}
  (function buildShells(){
    const rot=new THREE.Matrix4().makeRotationX(-Math.PI/2);
    for(const rec of INDOOR_BLD){ const b=rec.b;
      if(typeof PLATEAU_SKIP!=='undefined' && PLATEAU_SKIP.has(b.id)) continue;  // PLATEAU 実形状側で描画
      const sfp=b.footprint; if(!sfp || sfp.length<3) continue;
      const shape=new THREE.Shape(); shape.moveTo(sfp[0][0], sfp[0][1]);
      for(let i=1;i<sfp.length;i++) shape.lineTo(sfp[i][0], sfp[i][1]);
      const hh=Math.max(b.depth||b.height||FH, 1.0)+BLD_SKIRT; let geo;   // buildBuildings と同式
      try { geo=new THREE.ExtrudeGeometry(shape,{ depth:hh, bevelEnabled:false, steps:1 }); } catch(e){ continue; }
      geo.translate(0,0,(b.base||0)+(b.gz||0)-BLD_SKIRT); geo.applyMatrix4(rot);
      const kind=(KIND_COLOR[b.kind]!==undefined)? b.kind : 'generic';
      const mat=new THREE.MeshLambertMaterial({ color:NEUTRAL_BLD, transparent:true, opacity:1.0 });
      mat.userData.kindColor=KIND_COLOR[kind]; mat.userData.indoorShell=true;
      const mesh=new THREE.Mesh(geo, mat);
      buildingMats.push(mat); buildingMeshes.push(mesh);   // X線/分類色/lyBld に自動追従
      indoorShells.push({ bi:rec.bi, b, mesh, mat }); scene.add(mesh);
    }
  })();
  // ---- フロア板(canvas→texture)。遅延生成 + LRU8 ----
  const PLATE_CAP=8, PLATE_MAXPX=256, PLATE_FLOOR_CAP=24;
  const _plate=new Map(), _plateLRU=[];
  function _floorsToShow(b, bi){ const set=new Set();
    const rec=_indoorSpecFor(b); if(rec) for(const fl of (rec.floors||[])) set.add(fl.f|0);
    const s0=Math.floor(t), P=TRACKS.positions[s0]||[];   // 在館中の階も足す
    for(let i=0;i<P.length;i++){ const w=P[i][2];
      if(w>=1000 && Math.floor((w-1000)/100)===bi) set.add((w-1000)%100); }
    let arr=[...set].filter(f=> f!==0).sort((a,c)=>a-c);
    if(arr.length>PLATE_FLOOR_CAP) arr=arr.slice(0, PLATE_FLOOR_CAP); return arr; }
  function _makePlate(b, f, lay){
    const bb=lay.bbox, W=bb[2]-bb[0], H=bb[3]-bb[1]; if(W<=0||H<=0) return null;
    const asp=W/H; let cw, ch;
    if(asp>=1){ cw=PLATE_MAXPX; ch=Math.max(16, Math.round(PLATE_MAXPX/asp)); }
    else { ch=PLATE_MAXPX; cw=Math.max(16, Math.round(PLATE_MAXPX*asp)); }
    const cv=document.createElement('canvas'); cv.width=cw; cv.height=ch; const g=cv.getContext('2d');
    const tfx=(x,y)=>[ (x-bb[0])/W*cw, ch-(y-bb[1])/H*ch ];
    g.clearRect(0,0,cw,ch); g.fillStyle='rgba(20,26,34,0.72)'; g.fillRect(0,0,cw,ch);
    const cats=_zoneCats(lay, b, f);
    for(let zi=0; zi<lay.zones.length; zi++){ const z=lay.zones[zi];
      const hue=(_ZONE_HUE[cats[zi]]!==undefined)? _ZONE_HUE[cats[zi]] : 210;
      const a=tfx(z[0],z[1]), c=tfx(z[2],z[3]);
      const rx=Math.min(a[0],c[0]), ry=Math.min(a[1],c[1]), rw=Math.abs(c[0]-a[0]), rh=Math.abs(c[1]-a[1]);
      g.fillStyle='hsla('+hue+',48%,56%,0.86)'; g.fillRect(rx,ry,rw,rh);
      g.strokeStyle='rgba(12,16,22,0.55)'; g.lineWidth=1; g.strokeRect(rx,ry,rw,rh); }
    const co=lay.corridor, ca=tfx(co[0],co[1]), cc=tfx(co[2],co[3]);
    g.fillStyle='rgba(200,210,225,0.14)';
    g.fillRect(Math.min(ca[0],cc[0]),Math.min(ca[1],cc[1]),Math.abs(cc[0]-ca[0]),Math.abs(cc[1]-ca[1]));
    const cr=lay.core, ka=tfx(cr[0],cr[1]), kc=tfx(cr[2],cr[3]);
    g.fillStyle='rgba(40,52,68,0.95)';
    g.fillRect(Math.min(ka[0],kc[0]),Math.min(ka[1],kc[1]),Math.abs(kc[0]-ka[0]),Math.abs(kc[1]-ka[1]));
    const tex=new THREE.CanvasTexture(cv); tex.minFilter=THREE.LinearFilter; tex.magFilter=THREE.LinearFilter;
    tex.generateMipmaps=false;
    const pgeo=new THREE.PlaneGeometry(W, H); pgeo.rotateX(-Math.PI/2);
    const pmat=new THREE.MeshBasicMaterial({ map:tex, transparent:true, opacity:0.9,
      depthWrite:false, side:THREE.DoubleSide });
    const mesh=new THREE.Mesh(pgeo, pmat);
    // フロア板は建物の地上階床(gz)から実効階高で積む = upOf(w) と同じ式(B-3 整合)
    const yy=(b.gz||0) + Math.max((f-1),0)*floorHOf(b) + 0.15;
    mesh.position.set((bb[0]+bb[2])/2, yy, -(bb[1]+bb[3])/2); mesh.renderOrder=2;
    return mesh;
  }
  function _disposeGroup(grp){ scene.remove(grp); grp.traverse(o=>{
    if(o.material){ if(o.material.map) o.material.map.dispose(); o.material.dispose(); }
    if(o.geometry) o.geometry.dispose(); }); }
  function ensurePlates(bi){
    if(_plate.has(bi)){ const k=_plateLRU.indexOf(bi); if(k>=0){ _plateLRU.splice(k,1); _plateLRU.push(bi); } return _plate.get(bi); }
    const b=SCENE.buildings[bi]; if(!b) return null;
    const grp=new THREE.Group();
    for(const f of _floorsToShow(b, bi)){ const lay=_layoutFor(bi, f); if(!lay) continue;
      const m=_makePlate(b, f, lay); if(m) grp.add(m); }
    scene.add(grp); const cell={ group:grp }; _plate.set(bi, cell); _plateLRU.push(bi);
    while(_plateLRU.length>PLATE_CAP){ const ev=_plateLRU.shift(); const c=_plate.get(ev);
      if(c) _disposeGroup(c.group); _plate.delete(ev); }
    return cell;
  }
  function hideAllPlates(){ for(const c of _plate.values()) c.group.visible=false; }
  // ---- 接近フェード(外殻の不透明度=距離連動・X線と min 合成)+ フロア板の遅延表示 ----
  const FADE_NEAR=60, FADE_FAR=220, PLATE_DIST=260;
  function _bldDist(b){ const dx=camera.position.x - b.cx, dz=camera.position.z - (-b.cy); return Math.hypot(dx, dz); }
  function updateProximity(){
    if(!INDOOR) return;
    const onEl=document.getElementById('lyIndoor'); const active=onEl? onEl.checked : true;
    const xrayOn=document.getElementById('xray').checked; const base=xrayOn? 0.28 : 1.0;
    for(const sh of indoorShells){ let op=base;
      if(active){ const d=_bldDist(sh.b);
        const fr=Math.max(0, Math.min(1, (d-FADE_NEAR)/(FADE_FAR-FADE_NEAR)));
        op=Math.min(base, fr); }                         // どちらか透明なら透明(min 合成)
      sh.mat.opacity=op; sh.mat.needsUpdate=true; }
    // PLATEAU 実形状(1メッシュ=個別分離不可)は照合屋内建物への最短距離で一括フェード(妥協・報告済)
    if(typeof PLATEAU_DATA!=='undefined' && PLATEAU_DATA){ let dmin=Infinity;
      for(const rec of INDOOR_BLD){ if(PLATEAU_SKIP.has(rec.b.id)) dmin=Math.min(dmin, _bldDist(rec.b)); }
      if(dmin<Infinity){ const fr=Math.max(0, Math.min(1, (dmin-FADE_NEAR)/(FADE_FAR-FADE_NEAR)));
        const op=active? Math.min(base, fr) : base;
        for(const m of buildingMats){ if(m.userData.plateau){ m.opacity=op; m.needsUpdate=true; } } } }
    if(!active){ hideAllPlates(); return; }
    for(const rec of INDOOR_BLD){ if(_bldDist(rec.b)<PLATE_DIST){ const c=ensurePlates(rec.bi); if(c) c.group.visible=true; } }
    for(const bi of _plateLRU){ const b=SCENE.buildings[bi];
      if(b && _bldDist(b)>=PLATE_DIST){ const c=_plate.get(bi); if(c) c.group.visible=false; } }
  }
  controls.addEventListener('change', updateProximity);
  const _xr=document.getElementById('xray'); if(_xr) _xr.addEventListener('change', updateProximity);
  const _li=document.getElementById('lyIndoor'); if(_li) _li.addEventListener('change', updateProximity);
  updateProximity();                     // 初期化(遠景=不透明・板は未生成)
})();

"""


# ============================================================ 軌跡バイナリ(P0)
_TRACKS_BIN_LOADING = (
    '<div id="binload" style="position:fixed;left:50%;top:50%;'
    'transform:translate(-50%,-50%);z-index:60;padding:10px 16px;border-radius:8px;'
    'background:rgba(12,16,22,.84);color:#dfe7f2;font:13px/1.6 system-ui,sans-serif;'
    'display:none">軌跡データを読み込み中…</div>')

_TRACKS_BIN_ANCHOR = ("const TRACKS = JSON.parse("
                      "document.getElementById('tracks-data').textContent);")

_TRACKS_BIN_JS = r"""// ---------- 軌跡バイナリ(P0): チャンク遅延ロード。TRACKS は従来と同じ形の façade。
// tracks-data には tracks_meta.json(= tracks.bin の JSON ヘッダ)だけが入っている。
// 位置/移動/交通は chunk_NNNN.js を script 要素の動的挿入で取り込んで型付き配列にする
// (fetch は file:// で CORS に阻まれるため。既存 plateau_mesh.js と同じ src 参照方式)。
const TB = JSON.parse(document.getElementById('tracks-data').textContent);
const BIN = TB.binary;
const _CHUNK_DIR = "__CHUNK_DIR__";
const _NA = BIN.n_agents, _NSB = BIN.n_steps, _PC = BIN.pos_coords;
const _QU = BIN.quant, _OX = BIN.origin[0], _OY = BIN.origin[1];
const _PAL = BIN.state_palette;
const _NOTR = !!BIN.no_traffic;
const _CH = new Map(), _PEND = new Set(), _LRU = [];
const _KEEP = 4;                  // 常駐チャンク数(再生に必要なのは高々 2 = 現在と次)
const _PCACHE = new Map(), _MCACHE = new Map(), _TCACHE = new Map();
const _HOLD = (function(){ const a = new Array(_NA);
  for(let i=0;i<_NA;i++) a[i] = [0, 0, -1];      // 未ロード時は「範囲外」= 非表示
  return a; })();
const _MHOLD = new Array(_NA).fill(null);
const _THOLD = { n:0, segs:[] };

function _chunkOf(s){ const k = Math.floor(s / BIN.chunk_steps);
  return Math.max(0, Math.min(BIN.n_chunks - 1, k)); }
function _chunkFile(ci){ return _CHUNK_DIR + '/chunk_' + String(ci).padStart(4,'0') + '.js'; }
function _need(ci){
  if(ci < 0 || ci >= BIN.n_chunks || _CH.has(ci) || _PEND.has(ci)) return;
  _PEND.add(ci);
  const sc = document.createElement('script');
  sc.src = _chunkFile(ci); sc.async = true;
  sc.onerror = ()=>{ _PEND.delete(ci);
    console.error('tracks.bin: チャンク読込に失敗', sc.src); };
  document.head.appendChild(sc);
}
window.__TRACKS_CHUNK__ = function(ci, b64){
  const c = BIN.chunks[ci]; if(!c) return;
  if(_CH.has(ci)){ _PEND.delete(ci); return; }      // 二重取り込みで LRU が壊れないように
  const s = atob(b64), u = new Uint8Array(s.length);
  for(let i=0;i<s.length;i++) u[i] = s.charCodeAt(i);
  const buf = u.buffer, S = c.sec, ns = c.s1 - c.s0;
  const ST = (BIN.state_dtype === 'u2') ? Uint16Array : Uint32Array;
  _CH.set(ci, { s0:c.s0, s1:c.s1,
    pos:   new Int16Array(buf, S.pos, ns*_NA*_PC),
    st:    new ST(buf, S.state, ns*_NA),
    mvoff: new Uint32Array(buf, S.mvoff, ns+1),
    mvag:  new Uint32Array(buf, S.mvag, c.n_moves),
    mvmode:new Uint8Array(buf, S.mvmode, c.n_moves),
    mvpo:  new Uint32Array(buf, S.mvpo, c.n_moves+1),
    mvpts: new Int16Array(buf, S.mvpts, c.n_move_pts*2),
    troff: new Uint32Array(buf, S.troff, ns+1),
    trn:   new Int32Array(buf, S.trn, ns),
    trpo:  new Uint32Array(buf, S.trpo, c.n_segs+1),
    trpts: new Int16Array(buf, S.trpts, c.n_seg_pts*2) });
  _PEND.delete(ci); _LRU.push(ci);
  while(_LRU.length > _KEEP){ const old = _LRU.shift(); if(old !== ci) _CH.delete(old); }
  _PCACHE.clear(); _MCACHE.clear(); _TCACHE.clear();
};
function _trim(m){ while(m.size > 6){ m.delete(m.keys().next().value); } }
function _at(s){ const d = _CH.get(_chunkOf(s));
  return (d && s >= d.s0 && s < d.s1) ? d : null; }
function _posStep(s){
  if(_PCACHE.has(s)) return _PCACHE.get(s);
  const d = _at(s); if(!d) return _HOLD;
  const li = s - d.s0, b = li*_NA*_PC, sb = li*_NA, out = new Array(_NA);
  for(let i=0;i<_NA;i++){ const o = b + i*_PC;
    out[i] = [ d.pos[o]*_QU + _OX, d.pos[o+1]*_QU + _OY, _PAL[d.st[sb+i]] ]; }
  _PCACHE.set(s, out); _trim(_PCACHE); return out;
}
function _movesStep(s){
  if(_MCACHE.has(s)) return _MCACHE.get(s);
  const d = _at(s); if(!d) return _MHOLD;
  const li = s - d.s0, out = new Array(_NA).fill(null);
  for(let r = d.mvoff[li]; r < d.mvoff[li+1]; r++){
    const p0 = d.mvpo[r], p1 = d.mvpo[r+1], pts = new Array(p1-p0);
    for(let j=p0;j<p1;j++) pts[j-p0] = [ d.mvpts[j*2]*_QU + _OX, d.mvpts[j*2+1]*_QU + _OY ];
    out[d.mvag[r]] = [ d.mvmode[r], pts ];
  }
  _MCACHE.set(s, out); _trim(_MCACHE); return out;
}
function _trafficStep(s){
  if(_NOTR) return _THOLD;
  if(_TCACHE.has(s)) return _TCACHE.get(s);
  const d = _at(s); if(!d) return _THOLD;
  const li = s - d.s0, segs = [];
  for(let r = d.troff[li]; r < d.troff[li+1]; r++){
    const p0 = d.trpo[r], p1 = d.trpo[r+1], pts = new Array(p1-p0);
    for(let j=p0;j<p1;j++) pts[j-p0] = [ d.trpts[j*2]*_QU + _OX, d.trpts[j*2+1]*_QU + _OY ];
    segs.push(pts);
  }
  const o = { n: d.trn[li], segs: segs };
  _TCACHE.set(s, o); _trim(_TCACHE); return o;
}
function _series(fn){
  return new Proxy({}, {
    get(o, k){ if(k === 'length') return _NSB;
      if(typeof k !== 'string') return undefined;
      const s = +k; return Number.isInteger(s) ? fn(s) : undefined; },
    has(o, k){ const s = +k; return Number.isInteger(s) && s >= 0 && s < _NSB; } });
}
const TRACKS = { meta: TB.meta, agents: TB.agents, ids: TB.ids, sim_min: TB.sim_min,
  positions: _series(_posStep), moves: _series(_movesStep), traffic: _series(_trafficStep) };
function _clampStep(tt){ return Math.max(0, Math.min(_NSB-1, Math.floor(tt) || 0)); }
const TRACKS_LAZY = {
  ready(tt){ if(BIN.n_chunks === 0) return true;    // 0 step のランで待ち続けない
    const s = _clampStep(tt);
    return _at(s) !== null && _at(Math.min(s+1, _NSB-1)) !== null; },
  request(tt){ const s = _clampStep(tt);
    _need(_chunkOf(s)); _need(_chunkOf(Math.min(s+1, _NSB-1))); },
  prefetch(tt){ _need(_chunkOf(_clampStep(tt)) + 1); },
  status(){ return _CH.size + '/' + BIN.n_chunks; } };
function _showLoading(on){ const el = document.getElementById('binload');
  if(el) el.style.display = on ? 'block' : 'none'; }
_need(0); _need(1);
"""

_TRACKS_BIN_TICK_ANCHOR = "  if(playing){ t += dt * speed * 1.2;   // 1x で ≈1.2 step/秒"

_TRACKS_BIN_TICK = r"""  if(!TRACKS_LAZY.ready(t)){ TRACKS_LAZY.request(t); _showLoading(true);
    controls.update(); renderer.render(scene, camera); return; }
  _showLoading(false); TRACKS_LAZY.prefetch(t);
"""


def _inject_tracks_binary(html: str, chunk_dir: str) -> str:
    """TRACKS を「チャンク遅延ロードの façade」に差し替える(既定 OFF=呼ばれない)。
    ①ローディング表示 ②TRACKS 定義の置換 ③未ロード時に再生を止める animate ゲート。"""
    anchor_data = '<script type="application/json" id="scene-data">'
    html = _replace_once(html, anchor_data,
                         _TRACKS_BIN_LOADING + "\n" + anchor_data, "binload")
    html = _replace_once(html, _TRACKS_BIN_ANCHOR,
                         _TRACKS_BIN_JS.replace("__CHUNK_DIR__", chunk_dir), "tracks-bin")
    html = _replace_once(html, _TRACKS_BIN_TICK_ANCHOR,
                         _TRACKS_BIN_TICK + _TRACKS_BIN_TICK_ANCHOR, "tracks-bin-tick")
    return html


def _mark_no_traffic(tracks_meta_json: str) -> str:
    """--no-traffic をバイナリ経路で表現する(交通セクションは読まずに空を返させる)。"""
    meta = json.loads(tracks_meta_json)
    meta.setdefault("binary", {})["no_traffic"] = True
    return json.dumps(meta, ensure_ascii=False, separators=(",", ":"))


def _write_tracks_chunks(run_dir: Path, chunk_dir: str = "tracks_bin") -> tuple[int, int]:
    """scene3d/tracks.bin → runs/<name>/<chunk_dir>/chunk_NNNN.js(base64 の JSONP)。
    以前の生成物が残ると古いチャンクを掴むので、書く前に chunk_*.js を掃除する。"""
    blob = (run_dir / "scene3d" / "tracks.bin").read_bytes()
    out_dir = run_dir / chunk_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in sorted(out_dir.glob("chunk_*.js")):
        stale.unlink()
    total = 0
    parts = _load_tracks_bin().chunk_sidecars(blob)
    for name, text in parts:
        p = out_dir / name
        p.write_text(text, encoding="ascii")
        total += p.stat().st_size
    return len(parts), total


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
    tracks_binary = "--tracks-binary" in flags
    scene_json, tracks_json, plateau_json, terrain_json = _ensure_scene(run_dir, tracks_binary)
    if "--no-traffic" in flags:
        tracks_json = (_mark_no_traffic(tracks_json) if tracks_binary
                       else _strip_traffic(tracks_json))
    # データ存在フラグ(注入するか=バイト同一を崩すか の判定)。パースは 1 回だけ。
    has_extras = False
    has_ubld4 = False
    if plateau_json is not None:
        try:
            _pw = json.loads(plateau_json)
            has_extras = bool(_pw.get("extras"))
            has_ubld4 = bool(_pw.get("ubld4"))
        except Exception:
            has_extras = False
            has_ubld4 = False
    # テクスチャ付き LOD2.2 サイドカー(export_3d --plateau-tex 産)。分離版のみで使う。
    tex_p = run_dir / "plateau_tex.js"
    has_tex = tex_p.exists()
    # 縮退策②: tex がある時は**分離版だけ**軌跡をチャンク遅延ロードにする。
    # 埋め込み版 viewer3d.html は「単一ファイルで完結」が存在理由なので触らない
    # (--tracks-binary を明示した時だけ、従来どおり両方がバイナリ経路になる)。
    lite_binary = tracks_binary
    lite_tracks_json = tracks_json
    if has_tex and not tracks_binary:
        _bin_p = run_dir / "scene3d" / "tracks.bin"
        _meta_p = run_dir / "scene3d" / "tracks_meta.json"
        if _bin_p.exists() and _meta_p.exists():
            lite_tracks_json = _meta_p.read_text(encoding="utf-8")
            if "--no-traffic" in flags:
                lite_tracks_json = _mark_no_traffic(lite_tracks_json)
            lite_binary = True
    mode_legend = None
    try:
        mode_legend = json.loads(tracks_json).get("meta", {}).get("mode_legend")
    except Exception:
        mode_legend = None
    # 顕著イベント(scene/tracks と同一入力=l1_events.parquet から抽出。無ければ None=バイト同一)
    notable_json = _ensure_notable(run_dir, tracks_json)
    # 屋内オーバレイ(space_move / indoor_tracks サイドカー有=新ランのみ。無ければ None=バイト同一)
    indoor_json = _ensure_indoor(run_dir, tracks_json)
    html = build_html(run_dir.name, scene_json, tracks_json,
                      plateau_json=plateau_json, terrain_json=terrain_json,
                      has_extras=has_extras, mode_legend=mode_legend,
                      notable_json=notable_json, indoor_json=indoor_json,
                      tracks_binary=tracks_binary, has_ubld4=has_ubld4,
                      tex_note=has_tex)
    out = run_dir / "viewer3d.html"
    out.write_text(html, encoding="utf-8")
    mb = out.stat().st_size / 1024 / 1024
    print(f"  {out}  ({mb:.2f} MB)")
    if tracks_binary:
        n_chunks, chunk_bytes = _write_tracks_chunks(run_dir)
        bin_mb = (run_dir / "scene3d" / "tracks.bin").stat().st_size / 1024 / 1024
        print(f"  {run_dir / 'tracks_bin'}  ({n_chunks} チャンク・"
              f"{chunk_bytes/1024/1024:.2f} MB / tracks.bin {bin_mb:.2f} MB)"
              f"  ← 遅延ロード(HTML には常駐しない)")
    if plateau_json is not None:
        if mb > 80:
            print(f"  [warn] 埋め込み版が {mb:.1f} MB > 80MB ゲート。"
                  " --tracks-binary(軌跡のチャンク遅延ロード)"
                  "・LOD 簡略化・分離版の利用を検討。")
        # 分離版: 軽量 HTML + サイドカー(同フォルダに置けば file:// で動く)
        side = run_dir / "plateau_mesh.js"
        # 縮退策①: tex 経路では統合メッシュ配列を載せない(buildPlateau が読まない)。
        # tex が無ければ plateau_json をそのまま書く=従来とバイト同一。
        side.write_text("PLATEAU_MESH = "
                        + (_slim_plateau_for_tex(plateau_json) if has_tex else plateau_json)
                        + ";", encoding="utf-8")
        lite = build_html(run_dir.name, scene_json, lite_tracks_json,
                          plateau_src="plateau_mesh.js", terrain_json=terrain_json,
                          has_extras=has_extras, mode_legend=mode_legend,
                          notable_json=notable_json, indoor_json=indoor_json,
                          tracks_binary=lite_binary, has_ubld4=has_ubld4,
                          plateau_tex_src=("plateau_tex.js" if has_tex else None))
        lite_p = run_dir / "viewer3d_lite.html"
        lite_p.write_text(lite, encoding="utf-8")
        print(f"  {lite_p}  ({lite_p.stat().st_size/1024/1024:.2f} MB)"
              f"  + {side.name}  ({side.stat().st_size/1024/1024:.2f} MB)")
        chunk_bytes = 0
        if lite_binary and not tracks_binary:      # 分離版だけバイナリ = ここで書き出す
            n_chunks, chunk_bytes = _write_tracks_chunks(run_dir)
            print(f"  + {run_dir / 'tracks_bin'}  ({n_chunks} チャンク・"
                  f"{chunk_bytes/1024/1024:.2f} MB)  ← 分離版のみ遅延ロード")
        elif tracks_binary:
            chunk_bytes = sum(p.stat().st_size
                              for p in (run_dir / "tracks_bin").glob("chunk_*.js"))
        if has_tex:
            tex_mb = tex_p.stat().st_size / 1024 / 1024
            total = (lite_p.stat().st_size + side.stat().st_size
                     + tex_p.stat().st_size + chunk_bytes) / 1024 / 1024
            gate = "以内" if total <= 80 else "超過"
            print(f"  + {tex_p.name}  ({tex_mb:.2f} MB)"
                  f"  ← 分離版合計 {total:.2f} MB(80MB ゲート{gate}"
                  f"・埋め込み版にテクスチャは入れない)")
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
      <label class="chk"><input type="checkbox" id="lyClass"> 分類色(建物を用途で色分け)</label>
      <label class="chk"><input type="checkbox" id="lyAgent" checked> エージェント</label>
      <div class="op">大きさ <input type="range" id="agSize" min="1" max="6" step="0.5" value="2"><span id="agSizeV">×2.0</span></div>
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

// ---------- 地形(terrain_web.json 注入時のみ TERRAIN が入る。無ければ groundAt=0=完全従来動作)
// TERRAIN = {x0,y0,cell,nx,ny,quant,H:Int16Array}。格子(j,i)→世界(x0+i*cell, y0+j*cell)。
let TERRAIN = null;
function groundAt(x, y){          // 世界(x,y)での地表高[m]。地形なし=0。双一次補間。
  if(!TERRAIN) return 0;
  const nx = TERRAIN.nx, ny = TERRAIN.ny;
  let gx = (x - TERRAIN.x0) / TERRAIN.cell, gy = (y - TERRAIN.y0) / TERRAIN.cell;
  let i0 = Math.floor(gx), j0 = Math.floor(gy);
  if(i0 < 0) i0 = 0; else if(i0 > nx - 2) i0 = nx - 2;
  if(j0 < 0) j0 = 0; else if(j0 > ny - 2) j0 = ny - 2;
  const fx = Math.min(1, Math.max(0, gx - i0)), fy = Math.min(1, Math.max(0, gy - j0));
  const H = TERRAIN.H;
  const h00 = H[j0*nx+i0], h10 = H[j0*nx+i0+1], h01 = H[(j0+1)*nx+i0], h11 = H[(j0+1)*nx+i0+1];
  const a = h00 + (h10 - h00)*fx, b = h01 + (h11 - h01)*fx;
  return (a + (b - a)*fy) * TERRAIN.quant;
}
// フレーム毎フック(注入時のみ設定・データ無しでは常に null=従来動作)
let modeTick = null;    // feature5: 移動手段別の色/形の更新
let hudTick  = null;    // feature5: HUD の電車人数など

// ---------- Three 基本
const app = document.getElementById('app');
const renderer = new THREE.WebGLRenderer({ antialias:true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = false;
// 物理ベースに近い自然な階調(白飛び抑制)。ライト強度は updateSky/freezeDaylight で再バランス。
renderer.outputEncoding = THREE.sRGBEncoding;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.1;
app.appendChild(renderer.domElement);

const NEUTRAL_BLD = 0xd0d4da;   // 建物の無彩色(既定)。「分類色」トグル ON で kind 色/頂点色へ。
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0e14);
// 薄い Fog(遠景を空色に溶かす)。色は updateSky/freezeDaylight が空と連動させる。
scene.fog = new THREE.Fog(0x0a0e14, 1100, 3200);

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
let gridHelper = null, flatGround = null;
{
  const g = new THREE.PlaneGeometry(6000, 6000);
  g.rotateX(-Math.PI/2);
  const m = new THREE.MeshLambertMaterial({ color:0x141922,
    transparent:true, opacity:0.55, depthWrite:false });
  flatGround = new THREE.Mesh(g, m);
  flatGround.position.y = -0.05;
  flatGround.renderOrder = -2;
  scene.add(flatGround);
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
  //    外周 1px を **透明枠** として空けておく: 地形へ貼るとき地図矩形の外は uv∉[0,1] に
  //    なるが、ClampToEdge がこの透明枠を拾うので「地図の無い所は素の地形色」になる
  //    (枠が無いと端のタイル色が外側へ引き伸ばされる)。offset/repeat で内側を 0..1 に
  //    対応させるため、平面ドレープ(地形なしラン)の見た目は従来と同一。
  const canvas = document.createElement('canvas');
  canvas.width=nx*256+2; canvas.height=ny*256+2;
  const g2d = canvas.getContext('2d');
  const tex = new THREE.CanvasTexture(canvas);
  tex.encoding = THREE.sRGBEncoding;   // outputEncoding=sRGB 下で地図の色を正しく表示
  tex.minFilter = THREE.LinearFilter; tex.magFilter = THREE.LinearFilter;
  tex.generateMipmaps = false; tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.offset.set(1/canvas.width, 1/canvas.height);
  tex.repeat.set((nx*256)/canvas.width, (ny*256)/canvas.height);
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
      const dx=(tx-tx0)*256+1, dy=(ty-ty0)*256+1;   // +1 = 外周の透明枠を残す
      img.onload = ()=>{ try{ g2d.drawImage(img, dx, dy, 256, 256); tex.needsUpdate=true; }catch(e){} };
      img.onerror = ()=>{};
      img.src = `https://tile.openstreetmap.org/${z}/${tx}/${ty}.png`;
    } };
  OSM.ensureLoaded = OSM.load;
})();

// ---------- 建物(kind ごとにジオメトリを統合 = 少ないドローコール)
const BLD_SKIRT = 3.0;     // 押出しを下方へ延長する量[m](斜面での足元の隙間隠し・A-6)
// 建物の実効階高: PLATEAU 実測高がある建物は height/levels(=floorH)、無ければ既定 FH。
function floorHOf(b){ return (b && b.floorH) ? b.floorH : FH; }
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
    // 押出し深さ: base から depth 分。depth が無い旧 scene.json は従来どおり height。
    // 地下階を持つ建物は depth=(levels+below)*FH でないと屋上が below*FH 沈む(B-3)。
    // さらに SKIRT 分だけ下へ伸ばして、斜面での「足元の隙間」を隠す(A-6)。
    const h = Math.max(b.depth || b.height || FH, 1.0) + BLD_SKIRT;
    let geo;
    try { geo = new THREE.ExtrudeGeometry(shape, { depth:h, bevelEnabled:false, steps:1 }); }
    catch(e){ continue; }
    geo.translate(0, 0, (b.base || 0) + (b.gz || 0) - BLD_SKIRT);   // gz=地表高で地形に接地
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
    const mat = new THREE.MeshLambertMaterial({ color:NEUTRAL_BLD,
      transparent:true, opacity:1.0 });
    mat.userData.kindColor = KIND_COLOR[kind];   // 「分類色」トグルで復元する kind 色
    buildingMats.push(mat);
    const mesh = new THREE.Mesh(bg, mat);
    buildingMeshes.push(mesh); scene.add(mesh);
  }
})();

// 建物パレット: 既定=無彩色 / 「分類色」ON で kind 色(押出し)・頂点色(PLATEAU)へ
function applyBuildingPalette(useClass){
  for(const m of buildingMats){
    if(m.userData.plateau){                 // PLATEAU 実形状: 頂点色の on/off
      m.vertexColors = !!useClass;
      m.color.setHex(useClass ? 0xffffff : NEUTRAL_BLD);
    } else if(m.userData.kindColor !== undefined){
      m.color.setHex(useClass ? m.userData.kindColor : NEUTRAL_BLD);
    }
    m.needsUpdate = true;
  }
}

// ---------- 道路(全ポリラインを 1 本の LineSegments に統合)
let roadObj = null;
(function buildRoads(){
  const pos = [];
  // 地形がある時は「地形セル幅以下」に再分割してから接地する(A-5)。道路セグメントは
  // p90=45.5m あり、両端だけを地表に載せると起伏をまたぐ区間が空中/地中を貫いていた。
  const SUB = TERRAIN ? TERRAIN.cell : 0;
  for(const r of SCENE.roads){
    const g = r.g;
    for(let i=1;i<g.length;i++){
      const ax=g[i-1][0], ay=g[i-1][1], bx=g[i][0], by=g[i][1];
      let n = 1;
      if(SUB > 0){ n = Math.max(1, Math.ceil(Math.hypot(bx-ax, by-ay) / SUB)); }
      for(let s=0;s<n;s++){
        const t0=s/n, t1=(s+1)/n;
        const x0=ax+(bx-ax)*t0, y0=ay+(by-ay)*t0;
        const x1=ax+(bx-ax)*t1, y1=ay+(by-ay)*t1;
        pos.push(x0, groundAt(x0, y0) + 0.4, -y0);
        pos.push(x1, groundAt(x1, y1) + 0.4, -y1);
      }
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
      // 地上線路は地表に載る(A-4): 絶対 y=0.6 だと地形の 67% に埋没していた。
      // 地下鉄(上の分岐)は絶対 z=-8m が正しいのでそのまま。
      const pos = [];
      for(const p of g){ pos.push(p[0], groundAt(p[0], p[1]) + (r.z||0) + 0.6, -p[1]); }
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
    sp.position.copy(V(b.cx, b.cy, (b.gz||0) + (b.height||FH) + 10));   // A-7: gz 加算
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
// 人間比の素寸法(半径0.45m・全高1.8m=胴 0.9m + 半球 0.45m×2)を「1.0倍」とし、
// 表示倍率はスライダーで変える(既定 2.0 = 遠景での視認性優先)。旧実装は
// 半径2.6m・全高10.2m を **中心配置** していたため足元が 4m 埋まっていた(B-5)。
const AG_R = 0.45, AG_BODY = 0.9;
const AG_HALF = AG_BODY/2 + AG_R;        // 素寸法の半高 = 0.9m
let agScale = 2.0;                       // 表示倍率(UI スライダー)
let AG_LIFT = AG_HALF * agScale;         // 足元アンカー用の持ち上げ量
const NA = TRACKS.ids.length;
const agentGeo = capsuleGeometry(AG_R, AG_BODY);
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
// w -> 足元の接地面 y[m](屋内=そのフロアの床面 / 路上=地表)。
// 旧実装は「地表からの相対高」を返し placeAgents が groundAt に足していたため、
// 屋内でも足す基準が「エージェント位置の地表」= 建物基準面とズレていた(B-3)。
// いまは屋内は建物の地上階床(gz)から実効階高で積む絶対値を返す。
function upOf(w){
  if(w >= 1000){
    const bi = Math.floor((w-1000)/100);
    let floor = (w-1000) % 100;
    if(!(floor >= 0)) floor = 1;
    floor = Math.max(0, Math.min(99, floor));          // B-4: 表示側の上限ガード
    const b = SCENE.buildings[bi];
    // 基準面は「建物の地上階の床」= gz(地表高)。base(=-below*FH)は地下階の押出し
    // 下端であって階番号の原点ではない(sim の floor は 1=地上階)。base を足すと
    // 地下階を持つ建物で 1F の人が地下に沈むので使わない。
    if(b) return (b.gz||0) + Math.max(floor-1, 0) * floorHOf(b);
    return Math.max((floor-1)*FH, 0);                  // 建物不明(旧 tracks)の退避
  }
  return 0; // 路上 = 地表(呼び出し側が groundAt を使う)
}
// 足元の接地面(屋内=フロア面の絶対 y / 屋外=地形の高さ)
function footY(x, y, w){ return (w >= 1000) ? upOf(w) : groundAt(x, y); }

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
  // ACES + sRGB 下で白飛びしない強度に再バランス(hemi/ambient を厚めにして陰を持ち上げる)
  sun.intensity = 0.18 + daylight*0.95;
  hemi.intensity = 0.30 + daylight*0.45;
  ambient.intensity = 0.14 + daylight*0.12;
}

// ---------- 毎フレームのエージェント/車の配置
const _m = new THREE.Matrix4(), _q = new THREE.Quaternion(),
      _p = new THREE.Vector3(), _s = new THREE.Vector3(1,1,1), _hide = new THREE.Vector3(0,0,0);
const _sa = new THREE.Vector3(agScale, agScale, agScale);   // エージェント表示倍率
function placeAgents(t){
  const pos = posAt(t);
  for(let i=0;i<NA;i++){
    const [x,y,w] = pos[i];
    if(w === -1 || w === -2 || w === -3){  // 範囲外・睡眠・電車圏外は隠す
      _m.compose(_p.set(0,-9999,0), _q, _hide); agents.setMatrixAt(i,_m); continue; }
    _p.set(x, footY(x, y, w) + AG_LIFT, -y);   // 足元アンカー(屋外=地表 / 屋内=フロア面)
    _m.compose(_p, _q, _sa); agents.setMatrixAt(i, _m);
  }
  agents.instanceMatrix.needsUpdate = true;
}
// 表示倍率スライダー(人間比 1.0 倍を基準)。InstancedMesh のスケールで反映する。
function setAgentScale(v){
  agScale = Math.max(0.5, Math.min(8, Number(v) || 1));
  AG_LIFT = AG_HALF * agScale;
  _sa.set(agScale, agScale, agScale);
  const el = document.getElementById('agSizeV');
  if(el) el.textContent = '×' + agScale.toFixed(1);
  placeAgents(t);
}
function placeCars(t){
  const s0 = Math.floor(t), f = t - s0;
  const segs = (TRACKS.traffic[s0] && TRACKS.traffic[s0].segs) || [];
  const n = Math.min(segs.length, CAR_CAP);
  for(let i=0;i<CAR_CAP;i++){
    if(i < n){ const p = alongPath(segs[i], f);
      _p.set(p[0], groundAt(p[0], p[1]) + 0.9, -p[1]); _m.compose(_p, _q, _s); }
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
  applyBuildingPalette(L3('lyClass'));
  agents.visible = L3('lyAgent');
  cars.visible = L3('lyCars');
  if(roadObj) roadObj.visible = L3('lyRoad');
  railSurfaceObjs.forEach(o=> o.visible = L3('lyRail'));
  subwayObjs.forEach(o=> o.visible = L3('lyUnder'));
  labelSprites.forEach(s=> s.visible = L3('lyLabels'));
  setDayNight(L3('lyDayNight'));
  const osmOn = L3('lyOsm');
  if(OSM.mesh){ OSM.mesh.visible = osmOn; if(osmOn && OSM.ensureLoaded) OSM.ensureLoaded(); }
  if(gridHelper) gridHelper.visible = !osmOn && !TERRAIN;   // A-7: 地形注入時は常時非表示
  document.getElementById('osmAttr').style.display = osmOn ? 'block' : 'none';
}
['lyBld','lyClass','lyAgent','lyCars','lyRoad','lyRail','lyUnder','lyLabels','lyDayNight','lyOsm'].forEach(id=>{
  const el = document.getElementById(id);
  if(el) el.onchange = ()=>{ applyLayers(); saveSettings(); };
});
osmOp.oninput = ()=>{ OSM.opacity = Number(osmOp.value);
  if(OSM.mesh) OSM.mesh.material.opacity = OSM.opacity;
  osmOpV.textContent = Math.round(OSM.opacity*100)+'%'; saveSettings(); };
// エージェントの大きさ(人間比 1.0 = 半径0.45m・全高1.8m)
const agSize = document.getElementById('agSize');
if(agSize) agSize.oninput = ()=>{ setAgentScale(agSize.value); saveSettings(); };
document.getElementById('lyHdr').onclick = ()=>{ const b = document.getElementById('lyBody');
  const off = b.style.display === 'none'; b.style.display = off ? 'block' : 'none';
  document.getElementById('lyHdr').textContent = off ? 'レイヤー ▾' : 'レイヤー ▸'; };

// 昼夜ライティング OFF = 均一な昼光で全体を見やすく固定
function freezeDaylight(){
  sun.position.set(420, 720, 260); sun.target.position.set(0,0,0);
  sun.color.set(0xffffff); sun.intensity = 0.95;
  hemi.intensity = 0.55; ambient.intensity = 0.32;
  const c = new THREE.Color(0x223047); scene.background.copy(c); scene.fog.color.copy(c);
}
function setDayNight(on){ dayNight = on; if(!on) freezeDaylight(); }

// 設定を localStorage にラン別保持(再訪時に維持)
const LS_KEY = 'shibuya3d:__RUN_NAME__';
function saveSettings(){ try{ localStorage.setItem(LS_KEY, JSON.stringify({
  lyBld:L3('lyBld'), lyClass:L3('lyClass'), lyAgent:L3('lyAgent'), lyCars:L3('lyCars'), lyRoad:L3('lyRoad'),
  lyRail:L3('lyRail'), lyUnder:L3('lyUnder'), lyLabels:L3('lyLabels'),
  lyDayNight:L3('lyDayNight'), lyOsm:L3('lyOsm'), osmOp:OSM.opacity, agScale,
  xray:document.getElementById('xray').checked, colorMode })); }catch(e){} }
function loadSettings(){ try{ const s = JSON.parse(localStorage.getItem(LS_KEY) || 'null'); if(!s) return;
  ['lyBld','lyClass','lyAgent','lyCars','lyRoad','lyRail','lyUnder','lyLabels','lyDayNight','lyOsm','xray'].forEach(k=>{
    const el = document.getElementById(k); if(el && typeof s[k] === 'boolean') el.checked = s[k]; });
  if(typeof s.osmOp === 'number'){ OSM.opacity = s.osmOp; osmOp.value = s.osmOp; }
  if(typeof s.agScale === 'number' && agSize){ agSize.value = s.agScale; }
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
setAgentScale(agSize ? agSize.value : agScale);
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
  if(modeTick) modeTick(t);   // feature5: 移動手段別の色/形(注入時のみ)
  if(hudTick)  hudTick(t);    // feature5: HUD の電車人数など(注入時のみ)
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
