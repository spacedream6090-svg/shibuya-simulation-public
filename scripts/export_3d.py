"""ラン → 中立 3D シーン書き出し(バッチE / 3D 可視化)。

使い方:  python scripts/export_3d.py runs/<name> [--map data/shibuya_osm.json]
                                  [--sample-agents N] [--step-stride K]
                                  [--plateau] [--plateau-dir data/plateau]
                                  [--rich-tracks] [--low-mem]
                                  [--tracks-binary [--no-tracks-json] [--chunk-mb M]]
  --sample-agents N : tracks の対象を先頭 N エージェントに間引く(大規模ランの LOD 出力)。
  --step-stride K   : K step ごとに1フレームだけ出力する(時間ダウンサンプル)。
  --low-mem         : L1 を row group 単位でストリーム読みし、tracks 再構成が実際に使う
                      7 列 × 13 kind だけを Python へ持ち上げる(既定は 9 列全行を
                      to_pylist() で全展開)。**出力はバイト同一**(下記 load_track_events)。
                      1万体×1日(16.2M イベント・L1 167MiB)の実測で
                      プロセスピーク RSS 13.3GB → 2.0GB・所要 84.8s → 12.8s。
                      10日ランのように L1 が GB 級のときに使う。既定 OFF=従来経路そのまま。
                      ★第2段(小粒E): --low-mem は **イベント列自体を保持しない**
                      (reconstruct_tracks_streaming)。従来の --low-mem は「間引いた
                      イベント dict の全期間ぶん」を list で持っていた(10日ラン規模で
                      GB 級 = 台帳 PENDING §4)。いまは
                        pass1 = 列だけ走査して n_steps / step→sim_min / agent_id 集合を取り
                        pass2 = row group ごとに読んで **step 単位のバケツ 1 つ**だけ持つ
                      = イベント側は O(全期間) → O(row group + 1 step)。出力はバイト同一。
                      L1 が step 昇順でない場合は自動で従来の全保持経路へ退避する。
                      加えて tracks.json も **逐次書き出し**(dump_json_stream)になる
                      = dumps の戻り値(GB 級の 1 本の str)を作らない。バイト列は同一。
                      実測(wv_mock_30d・L1 22.6MB・tracks.json 82.8MB・実ピーク RSS):
                        既定     2235.6MB / 12.7s → 1985.5MB / 12.8s
                        --low-mem 1895.7MB /  9.5s → 1634.7MB /  8.9s
                      (tracks.json の SHA-256 は 4 通りすべて一致)
  --plateau         : PLATEAU 実形状(plateau_extract/match_plateau の成果物)で、照合済み建物を
                      実測メッシュに置換する(glb ハイブリッド+scene.json height 上書き+
                      viewer3d 用 plateau_web.json)。既定 OFF=従来出力とバイト同一。
                      terrain.npz/json があれば地表グリッド(terrain_web.json)と建物 gz、
                      extras.npz があれば地下街/橋メッシュ(plateau_web.extras)も書き出す。
                      ubld_lod4_mesh.npz(レーンA)があれば地下街 LOD4.1 の面種別つき
                      メッシュ(plateau_web.ubld4)も足す(旧 extras.ubld の箱を置換)。
  --plateau-tex     : テクスチャ付き LOD2.2(data/plateau/tiles_lod2/・レーンA産)を
                      **分離版サイドカー** runs/<name>/plateau_tex.js(JSONP)へ書き出す。
                      --plateau と併用必須。既定 OFF=このフラグ無しでは 1 バイトも変わらない。
                      埋め込み版 viewer3d.html には入れない(アトラス 1/2 で 80MB ゲート超過が
                      レーンA実測。分離版 viewer3d_lite.html + サイドカー 2 本が主経路)。
                      **--tracks-binary を自動で立てる**(標準経路)。分離版の軌跡を
                      チャンク遅延ロードへ逃がさないと 80MB ゲートに収まらないため
                      (実測は docs/plans/plateau-3d.md のサイズ表)。tracks.json も従来どおり
                      書くので、埋め込み版の自己完結性は変わらない。
  --rich-tracks     : tracks.json の移動手段を細分化(タクシー=mode 3・電車で圏外=w -3)。
                      既定 OFF=tracks.json はバイト同一。--plateau と併用可。
  --tracks-binary   : tracks を量子化型付きバイナリ(scene3d/tracks.bin + tracks_meta.json)でも
                      書き出す(P0 / docs/research/dt-integration-deep.md §2)。既定 OFF=
                      **既存出力とバイト同一**(このフラグ無しでは 1 バイトも変わらない)。
                      量子化は int16 × 0.05 m/unit(PLATEAU メッシュ埋め込みと同一・D-3)。
                      1万体×1日の実測で tracks.json 65.8MB → tracks.bin 19.1MB(1/3.4)。
  --no-tracks-json  : --tracks-binary 併用時のみ有効。tracks.json を書かない(10日ランで
                      数百 MB の JSON を作らずに済む)。読み手は tracks.bin を復号する。
  --chunk-mb M      : tracks.bin の遅延ロード用チャンク目標サイズ[MB](既定 8)。
  既定はいずれも全量・OFF=現行と完全同一。
生成物:  runs/<name>/scene3d/
  scene.json     — 静的シーン(建物押出し情報・道路・線路・POI)。座標系 local-m(X=east,Y=north,Z=up)。
  tracks.json    — 時系列(エージェント位置・移動ポリライン・車トラフィック・sim 時刻)。
  buildings.glb  — 建物押出しプリズムの glTF 2.0 バイナリ(numpy+stdlib 手書き生成、依存追加なし)。
  plateau_web.json — (--plateau 時のみ)照合建物の量子化メッシュ(int16×0.05m・base64)。
  tracks.bin / tracks_meta.json — (--tracks-binary 時のみ)量子化軌跡バイナリと、その JSON
                   ヘッダの同内容コピー(形式は scripts/tracks_bin.py の docstring)。
  ../plateau_tex.js — (--plateau-tex 時のみ・scene3d ではなくラン直下)テクスチャ付き
                   LOD2.2 のタイル別メッシュ+WebP アトラス data:URI(JSONP・分離版専用)。

設計方針: sim⇄viz 疎結合(docs/lit/viz__plateau-pipeline-overview.md)。
  本スクリプトは l1_events.parquet を読むだけ。sim 本体には非依存。
  座標は local-m のまま(2D ビューアの canvas 用 Y 反転は持ち込まない)。
  scene.json / tracks.json は Z-up。buildings.glb は glTF 標準の Y-up(-90°X 回転)で書き出す。

位置再構成ロジックは viz/make_viewer.py の 82-125 行(build_data 内)を移植・整理したもの。
make_viewer.py 自体は変更しない。
"""
from __future__ import annotations

import base64
import importlib.util
import json
import struct
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_tracks_bin():
    """scripts/tracks_bin.py を場所非依存で読み込む(本ファイル自体が importlib で
    読まれる前例に合わせる。scripts/ を sys.path に載せない)。"""
    spec = importlib.util.spec_from_file_location(
        "tracks_bin", Path(__file__).resolve().parent / "tracks_bin.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def _load_run_dt():
    """scripts/run_dt.py を場所非依存で読み込む(_load_tracks_bin と同じ流儀。
    本ファイルは scripts/ を sys.path に載せない設計なのでそれを守る)。"""
    spec = importlib.util.spec_from_file_location(
        "run_dt", Path(__file__).resolve().parent / "run_dt.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FLOOR_HEIGHT = 3.5
# W2-3: **ラン依存**(run.dt_min)。既定は正準 Δt=10 = 従来と 1 ビットも変わらない。
# 実際の値は export_run() が run dir から読んで build_scene / 再構成へ引数で配る。
STEP_MINUTES = 10
DEFAULT_START_MIN = 7 * 60  # make_viewer と同じ既定(7:00)

# 建物 kind → 頂点色 RGB(glb / three.js 共通の意図)
KIND_COLOR = {
    "station": (0xE8, 0x6A, 0x33),
    "retail": (0x3B, 0xA8, 0x9C),
    "office": (0x5B, 0x8F, 0xD6),
    "residential": (0x7A, 0xB8, 0x6A),
    "hotel": (0xC9, 0x6A, 0xB0),
    "public": (0x9B, 0x7F, 0xD4),
    "generic": (0xB8, 0xBE, 0xC7),
    "house?": (0xA9, 0x9E, 0x92),
}
DEFAULT_COLOR = (0xB8, 0xBE, 0xC7)


# ------------------------------------------------------------------ 設定/入出力
def _load_yaml(path: Path) -> dict:
    """config.yaml を読む。yaml が無い環境でも omegaconf で読めるようにフォールバック。"""
    try:
        import yaml
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except ModuleNotFoundError:
        from omegaconf import OmegaConf
        return OmegaConf.to_container(OmegaConf.load(str(path)), resolve=True)


def _resolve_map_path(run_dir: Path, override: Path | None) -> Path:
    if override is not None:
        return override if override.is_absolute() else (REPO_ROOT / override)
    cfg_path = run_dir / "config.yaml"
    if cfg_path.exists():
        cfg = _load_yaml(cfg_path)
        mp = Path(cfg.get("world", {}).get("map", "data/shibuya_osm.json"))
        return mp if mp.is_absolute() else (REPO_ROOT / mp)
    return REPO_ROOT / "data" / "shibuya_osm.json"


# ------------------------------------------------------------------ ear clipping
def _signed_area(ring: list) -> float:
    a = 0.0
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return a * 0.5


def _point_in_tri(p, a, b, c) -> bool:
    """p が三角形 abc の内部(辺上は除外)にあるか。"""
    (px, py), (ax, ay), (bx, by), (cx, cy) = p, a, b, c
    d1 = (px - bx) * (ay - by) - (ax - bx) * (py - by)
    d2 = (px - cx) * (by - cy) - (bx - cx) * (py - cy)
    d3 = (px - ax) * (cy - ay) - (cx - ax) * (py - ay)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def triangulate(ring: list) -> list:
    """凹多角形対応の ear clipping。ring は閉じない頂点列 [(x,y),...]。
    戻り値は ring への index 三つ組のリスト(CCW 巻き)。"""
    pts = [tuple(p) for p in ring]
    # 末尾が先頭と重複していれば除去
    if len(pts) > 1 and pts[-1] == pts[0]:
        pts = pts[:-1]
    n = len(pts)
    if n < 3:
        return []
    idx = list(range(n))
    if _signed_area(pts) < 0:  # CCW に正規化
        idx.reverse()
    tris: list = []
    guard = 0
    limit = 4 * n * n + 16
    while len(idx) > 3 and guard < limit:
        guard += 1
        m = len(idx)
        ear = False
        for k in range(m):
            i0, i1, i2 = idx[(k - 1) % m], idx[k], idx[(k + 1) % m]
            a, b, c = pts[i0], pts[i1], pts[i2]
            cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if cross <= 1e-12:  # reflex または退化
                continue
            clean = True
            for j in idx:
                if j in (i0, i1, i2):
                    continue
                if _point_in_tri(pts[j], a, b, c):
                    clean = False
                    break
            if clean:
                tris.append((i0, i1, i2))
                idx.pop(k)
                ear = True
                break
        if not ear:
            break  # 退化ポリゴン: これ以上分割できない
    if len(idx) == 3:
        tris.append((idx[0], idx[1], idx[2]))
    return tris


# ------------------------------------------------------------------ PLATEAU 実形状
PLATEAU_QUANT = 0.05  # viewer3d 埋め込み量子化 [m/単位](int16 で ±1638m まで表現可)
PLATEAU_ATTRIBUTION = ("建物形状: 国土交通省 Project PLATEAU "
                       "3D都市モデル(渋谷区 2025年度)を加工")


def _load_plateau(plateau_dir: Path) -> dict:
    """plateau_extract / match_plateau の成果物を読み、osm建物ID→実形状メッシュにする。

    building_offsets は F(三角形行列)の行オフセット(長さ=建物数+1・末尾=総行数)。
    照合表に居るが index に無い gml_id は黙ってスキップせず件数を報告する。"""
    idx_p = plateau_dir / "plateau_index.json"
    npz_p = plateau_dir / "plateau_mesh.npz"
    match_p = plateau_dir / "plateau_match.json"
    missing = [p.name for p in (idx_p, npz_p, match_p) if not p.exists()]
    if missing:
        raise SystemExit(
            f"[export_3d] --plateau: {plateau_dir} に {missing} が無い。"
            " 先に scripts/plateau_extract.py と scripts/match_plateau.py を実行する。")
    index = json.loads(idx_p.read_text(encoding="utf-8"))
    data = np.load(npz_p)
    V, F, off = data["V"], data["F"], data["building_offsets"]
    if int(off[-1]) != len(F):
        raise SystemExit("[export_3d] --plateau: building_offsets の末尾が F 行数と不一致")
    gml_pos = {b["gml_id"]: i for i, b in enumerate(index["buildings"])}
    matches = json.loads(match_p.read_text(encoding="utf-8"))["matches"]
    meshes: dict = {}
    heights: dict = {}
    dropped = 0
    for osm_id, m in matches.items():
        i = gml_pos.get(m["gml_id"])
        if i is None:
            dropped += 1
            continue
        Fb = F[int(off[i]):int(off[i + 1])]
        if len(Fb) == 0:
            dropped += 1
            continue
        vids, inv = np.unique(Fb, return_inverse=True)
        meshes[osm_id] = (V[vids].astype(np.float32),
                          inv.reshape(-1, 3).astype(np.int32))
        heights[osm_id] = float(index["buildings"][i]["height"])
    if dropped:
        print(f"  [plateau] 照合表のうち {dropped} 件は index/メッシュ不在でスキップ")
    return {"meshes": meshes, "heights": heights,
            "ground0_source": index.get("ground0_source", "?")}


def _vertex_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """面法線の頂点集約(面積重み)。PLATEAU 面は表面単位で頂点が独立しているため、
    集約してもほぼフラットシェーディング相当になる。"""
    n = np.zeros_like(V)
    tri = V[F]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for k in range(3):
        np.add.at(n, F[:, k], fn)
    L = np.linalg.norm(n, axis=1, keepdims=True)
    L[L == 0] = 1.0
    return (n / L).astype(np.float32)


def _append_mesh(V, F, color, pos, nrm, col, idxs):
    """実形状メッシュ(local-m Z-up)を glb 用配列へ追記する。"""
    n0 = len(pos)
    N = _vertex_normals(V, F)
    pos.extend(map(tuple, V))
    nrm.extend(map(tuple, N))
    col.extend([color] * len(V))
    idxs.extend((F.astype(np.int64) + n0).ravel().tolist())


def build_plateau_web(buildings: list, plateau: dict) -> dict:
    """viewer3d 埋め込み用の量子化メッシュ(int16×0.05m・base64・建物色は kind 準拠)。"""
    Vs, Fs, Cs, matched_ids = [], [], [], []
    n0 = 0
    for b in buildings:
        pm = plateau["meshes"].get(b["id"])
        if pm is None:
            continue
        V, F = pm
        color = KIND_COLOR.get(b.get("kind"), DEFAULT_COLOR)
        Vs.append(V)
        Fs.append(F.astype(np.int64) + n0)
        Cs.append(np.tile(np.array(color, dtype=np.uint8), (len(V), 1)))
        matched_ids.append(b["id"])
        n0 += len(V)
    if not Vs:
        raise SystemExit("[export_3d] --plateau: 照合済み建物が 0 件(match 対象の地図と"
                         " ランの地図が一致しているか確認)")
    V = np.vstack(Vs)
    F = np.vstack(Fs)
    C = np.vstack(Cs)
    Q = np.clip(np.round(V / PLATEAU_QUANT), -32767, 32767).astype("<i2")
    return {
        "quant_scale": PLATEAU_QUANT,
        "n_vertices": int(len(V)),
        "n_triangles": int(len(F)),
        "matched_ids": matched_ids,
        "positions_b64": base64.b64encode(Q.tobytes()).decode("ascii"),
        "indices_b64": base64.b64encode(F.astype("<u4").tobytes()).decode("ascii"),
        "colors_b64": base64.b64encode(C.tobytes()).decode("ascii"),
        "ground0_source": plateau.get("ground0_source", "?"),
        "attribution": PLATEAU_ATTRIBUTION,
    }


# ------------------------------------------------------------------ terrain / extras
TERRAIN_QUANT = 0.1   # terrain_web 量子化 [m/単位](int16 で ±3276.7m まで=標高十分)


def _load_terrain(plateau_dir: Path) -> dict | None:
    """地表グリッド(terrain.npz + terrain.json)を読む。無ければ None(従来どおり)。

    契約(並行エージェントの成果物と合わせる想定):
    - terrain.json : {"x0","y0","cell_m","nx","ny"} グリッド原点[local-m]・セル幅[m]・格子数。
    - terrain.npz  : 標高配列 H。key は "heights"(無ければ "z"、それも無ければ先頭配列)。
                     形状 (ny, nx)・row-major(H[iy][ix] = (x0+ix*cell_m, y0+iy*cell_m) の地表高[m])。
    """
    npz_p = plateau_dir / "terrain.npz"
    json_p = plateau_dir / "terrain.json"
    if not (npz_p.exists() and json_p.exists()):
        return None
    meta = json.loads(json_p.read_text(encoding="utf-8"))
    data = np.load(npz_p)
    if "heights" in data:
        H = data["heights"]
    elif "z" in data:
        H = data["z"]
    else:
        H = data[data.files[0]]
    H = np.ascontiguousarray(np.asarray(H, dtype=np.float64))
    if H.ndim != 2:
        raise SystemExit("[export_3d] terrain: 標高配列 H は 2次元(ny,nx)である必要がある")
    ny, nx = H.shape
    return {
        "x0": float(meta["x0"]), "y0": float(meta["y0"]),
        "cell_m": float(meta["cell_m"]),
        "nx": int(meta.get("nx", nx)), "ny": int(meta.get("ny", ny)),
        "H": H,
    }


def _sample_terrain_gz(terrain: dict, x: float, y: float) -> float:
    """双一次補間で (x,y) の地表高を返す(小数1桁)。格子外は端セルにクランプ(外挿しない)。
    平面グリッド(z=ax+by+c)では格子内で厳密に平面値を返す。"""
    H = terrain["H"]
    c = terrain["cell_m"]
    nx, ny = terrain["nx"], terrain["ny"]
    fx = (x - terrain["x0"]) / c
    fy = (y - terrain["y0"]) / c
    ix = int(np.floor(fx))
    iy = int(np.floor(fy))
    ix = min(max(ix, 0), max(nx - 2, 0))
    iy = min(max(iy, 0), max(ny - 2, 0))
    tx = min(max(fx - ix, 0.0), 1.0)
    ty = min(max(fy - iy, 0.0), 1.0)
    ix1 = min(ix + 1, nx - 1)
    iy1 = min(iy + 1, ny - 1)
    h00 = float(H[iy][ix]);  h10 = float(H[iy][ix1])
    h01 = float(H[iy1][ix]); h11 = float(H[iy1][ix1])
    h = (h00 * (1 - tx) * (1 - ty) + h10 * tx * (1 - ty)
         + h01 * (1 - tx) * ty + h11 * tx * ty)
    return round(h, 1)


def build_terrain_web(terrain: dict) -> dict:
    """viewer3d 用の量子化地表グリッド(int16×0.1m LE row-major・base64)。"""
    H = np.ascontiguousarray(terrain["H"], dtype=np.float64)
    Q = np.clip(np.round(H / TERRAIN_QUANT), -32768, 32767).astype("<i2")
    return {
        "x0": terrain["x0"], "y0": terrain["y0"],
        "cell_m": terrain["cell_m"],
        "nx": terrain["nx"], "ny": terrain["ny"],
        "quant": TERRAIN_QUANT,
        "heights_b64": base64.b64encode(np.ascontiguousarray(Q).tobytes()).decode("ascii"),
    }


def _load_extras(plateau_dir: Path) -> dict | None:
    """地下街(ubld)/橋(brid)の付帯メッシュ(extras.npz)を読む。無ければ None。

    契約: extras.npz は各カテゴリの頂点/面を "<cat>_V"(float (n,3))・"<cat>_F"(int (m,3)・
    カテゴリ内 0-based)で持つ。存在するカテゴリのみ採用(両キーが揃い非空のもの)。"""
    npz_p = plateau_dir / "extras.npz"
    if not npz_p.exists():
        return None
    data = np.load(npz_p)
    out: dict = {}
    for cat in ("ubld", "brid"):
        vk, fk = f"{cat}_V", f"{cat}_F"
        if vk in data and fk in data:
            V = np.asarray(data[vk], dtype=np.float64)
            F = np.asarray(data[fk])
            if len(V) and len(F):
                out[cat] = (V, F)
    return out or None


def build_extras_web(extras: dict) -> dict:
    """plateau_web.extras 用(既存メッシュと同じ int16×0.05m 量子化・<u4 index・建物色なし)。"""
    web: dict = {}
    for cat, (V, F) in extras.items():
        Q = np.clip(np.round(V / PLATEAU_QUANT), -32767, 32767).astype("<i2")
        web[cat] = {
            "positions_b64": base64.b64encode(np.ascontiguousarray(Q).tobytes()).decode("ascii"),
            "indices_b64": base64.b64encode(F.astype("<u4").tobytes()).decode("ascii"),
            "n_vertices": int(len(V)),
            "n_triangles": int(len(F)),
        }
    return web


# ---------------------------------------------------- 地下街 LOD4.1(梅 / レーンA産)
def _load_ubld4(plateau_dir: Path) -> dict | None:
    """ubld_lod4_mesh.npz + ubld_lod4.json(scripts/plateau_ubld_extract.py 産)を読む。

    契約(レーンA コミット 5ff56c4):
      npz: xyz(int16, (n,3) 量子化・origin_q オフセット付き) / origin_q(int64,(3,)) /
           tri(uint32,(m,3)) / tri_kind(uint8,(m,)) / kind_names(<U, (k,))
      json: params.quant_scale・mesh.kind_names・layers[{layer,z,...}](床面 z のピーク)
    無ければ None(= plateau_web に ubld4 キーを出さない=旧ランと同じ形)。"""
    npz_p = plateau_dir / "ubld_lod4_mesh.npz"
    json_p = plateau_dir / "ubld_lod4.json"
    if not (npz_p.exists() and json_p.exists()):
        return None
    meta = json.loads(json_p.read_text(encoding="utf-8"))
    data = np.load(npz_p)
    for key in ("xyz", "origin_q", "tri", "tri_kind"):
        if key not in data:
            raise SystemExit(f"[export_3d] ubld_lod4_mesh.npz に '{key}' が無い")
    quant = float(meta.get("params", {}).get("quant_scale", PLATEAU_QUANT))
    kinds = list(meta.get("mesh", {}).get("kind_names")
                 or [str(s) for s in data["kind_names"]])
    layers = [float(ly["z"]) for ly in meta.get("layers", [])]
    return {"xyz": np.asarray(data["xyz"]),
            "origin_q": np.asarray(data["origin_q"], dtype=np.int64),
            "tri": np.asarray(data["tri"], dtype=np.int64),
            "tri_kind": np.asarray(data["tri_kind"], dtype=np.uint8),
            "kind_names": kinds, "layer_z": layers, "quant": quant}


def build_ubld4_web(u: dict) -> dict:
    """plateau_web.ubld4 用。既存 extras と同じ「絶対 int16×0.05m」に直して base64 化し、
    面種別(tri_kind)と層(tri_layer)を付ける。

    層は「三角形重心 z に最も近い床面ピーク」で決める(レーンA の assign_layer と同一規則)。
    層分離そのものはレーンA が済ませた z ヒストグラムのピーク列をそのまま使う。"""
    Q = u["xyz"].astype(np.int64) + u["origin_q"]           # 絶対量子化値
    if Q.size and (int(np.abs(Q).max()) > 32767):
        raise SystemExit("[export_3d] ubld4: 絶対量子化値が int16 に収まらない"
                         f"(max={int(np.abs(Q).max())})")
    tri = u["tri"]
    zc = (Q[tri][:, :, 2].mean(axis=1) * u["quant"]) if len(tri) else np.zeros(0)
    peaks = np.asarray(u["layer_z"], dtype=np.float64)
    if peaks.size:
        tri_layer = np.argmin(np.abs(zc[:, None] - peaks[None, :]), axis=1).astype(np.uint8)
    else:
        tri_layer = np.zeros(len(tri), dtype=np.uint8)
    layers = [{"layer": i, "z": round(float(z), 3),
               "n_triangles": int((tri_layer == i).sum())}
              for i, z in enumerate(u["layer_z"])]
    return {
        "quant_scale": PLATEAU_QUANT,
        "n_vertices": int(len(Q)),
        "n_triangles": int(len(tri)),
        "kind_names": list(u["kind_names"]),
        "layers": layers,
        "positions_b64": base64.b64encode(
            np.ascontiguousarray(Q.astype("<i2")).tobytes()).decode("ascii"),
        "indices_b64": base64.b64encode(
            np.ascontiguousarray(tri.astype("<u4")).tobytes()).decode("ascii"),
        "tri_kind_b64": base64.b64encode(
            np.ascontiguousarray(u["tri_kind"].astype("<u1")).tobytes()).decode("ascii"),
        "tri_layer_b64": base64.b64encode(
            np.ascontiguousarray(tri_layer).tobytes()).decode("ascii"),
    }


# ------------------------------------------- テクスチャ付き LOD2.2(松 / レーンA産)
PLATEAU_TEX_ATTRIBUTION = ("テクスチャ付き建物: 国土交通省 Project PLATEAU "
                           "3D都市モデル(渋谷区 2025年度)3D Tiles を加工")


def _tex_triangle_flags(z, has_atlas: bool) -> np.ndarray:
    """三角形ごとの「テクスチャ付きプリミティブ由来か」フラグ。
    アトラスを持たないタイルは全て非テクスチャ扱い(UV は 0 で意味を持たない)。"""
    tri_n = int(z["tri"].shape[0])
    flags = np.zeros(tri_n, dtype=bool)
    if not has_atlas:
        return flags
    off = np.asarray(z["prim_tri_offsets"], dtype=np.int64)
    ptex = np.asarray(z["prim_textured"], dtype=np.int64)
    for p in range(len(ptex)):
        if ptex[p]:
            flags[off[p]:off[p + 1]] = True
    return flags


def build_plateau_tex(tiles_dir: Path) -> dict:
    """data/plateau/tiles_lod2/(レーンA A-1)→ 分離版サイドカー用の dict。

    設計:
    - **batch_shadowed を落とす**。この tileset は refine=REPLACE なので、bbox と交差する
      148 タイルには祖先タイルが持つ同一建物の低精細版が混ざる(686 batch)。そのまま描くと
      同じ建物が二重に出る。レーンA が npz に入れた batch 番号の三角形を捨てる(実測 15.70%)。
    - 残った三角形が参照する頂点だけに**詰め直す**(索引を uint16 に落とせる余地も作る)。
    - 三角形は「テクスチャ付きプリミティブ由来 → 非テクスチャ」の順に並べ替える。
      ビューアは 1 ジオメトリ 2 グループ(map 付き / 無彩色)で描く。
    - アトラス WebP は data:URI(file:// で fetch できないため。既存 plateau_mesh.js と同じ思想)。
    決定論: index.json のタイル順・np.unique の昇順・base64 のみ=同入力なら常にバイト同一。"""
    idx_p = tiles_dir / "index.json"
    if not idx_p.exists():
        raise SystemExit(f"[export_3d] --plateau-tex: {tiles_dir}/index.json が無い。"
                         " 先に tools/tiles3d_extract.py を実行する。")
    index = json.loads(idx_p.read_text(encoding="utf-8"))
    uv_scale = int(index.get("uv_scale", 65535))
    tiles_out: list = []
    n_v = n_tri_tex = n_tri_flat = n_drop = 0
    n_atlas = 0
    atlas_bytes = 0
    for t in index["tiles"]:
        z = np.load(tiles_dir / t["npz"])
        tri = np.asarray(z["tri"], dtype=np.int64)
        if len(tri) == 0:
            continue
        batch = np.asarray(z["batch"], dtype=np.int64)
        sh = (np.asarray(z["batch_shadowed"], dtype=np.int64)
              if "batch_shadowed" in z else np.zeros(0, dtype=np.int64))
        keep = np.ones(len(tri), dtype=bool)
        if sh.size:
            keep = ~np.isin(batch[tri[:, 0]], sh)
        n_drop += int((~keep).sum())
        atlas_name = t.get("atlas")
        raw_atlas = None
        if atlas_name:
            ap = tiles_dir / atlas_name
            if ap.exists():
                raw_atlas = ap.read_bytes()
        istex = _tex_triangle_flags(z, raw_atlas is not None)
        sel_tex = np.flatnonzero(keep & istex)
        sel_flat = np.flatnonzero(keep & ~istex)
        order = np.concatenate([sel_tex, sel_flat])
        if order.size == 0:
            continue
        kt = tri[order]
        used, inv = np.unique(kt, return_inverse=True)
        F = inv.reshape(-1, 3)
        xyz = np.asarray(z["xyz"])[used]
        uv = np.asarray(z["uv"], dtype=np.uint16)[used]
        xyz_dtype = "int32" if xyz.dtype == np.int32 else "int16"
        qs = "<i4" if xyz_dtype == "int32" else "<i2"
        idx_dtype = "uint16" if used.size <= 0xFFFF else "uint32"
        fs = "<u2" if idx_dtype == "uint16" else "<u4"
        rec = {
            "id": int(t["id"]),
            "origin_q": [int(v) for v in np.asarray(z["origin_q"]).tolist()],
            "xyz_dtype": xyz_dtype,
            "idx_dtype": idx_dtype,
            "n_vertices": int(used.size),
            "n_tex": int(sel_tex.size),
            "n_flat": int(sel_flat.size),
            "positions_b64": base64.b64encode(
                np.ascontiguousarray(xyz.astype(qs)).tobytes()).decode("ascii"),
            "uv_b64": base64.b64encode(
                np.ascontiguousarray(uv.astype("<u2")).tobytes()).decode("ascii"),
            "indices_b64": base64.b64encode(
                np.ascontiguousarray(F.astype(fs)).tobytes()).decode("ascii"),
            "atlas": (f"data:image/webp;base64,{base64.b64encode(raw_atlas).decode('ascii')}"
                      if (raw_atlas is not None and sel_tex.size) else None),
        }
        if rec["atlas"] is not None:
            n_atlas += 1
            atlas_bytes += len(raw_atlas)
        tiles_out.append(rec)
        n_v += int(used.size)
        n_tri_tex += int(sel_tex.size)
        n_tri_flat += int(sel_flat.size)
    return {
        "schema": "plateau_tex/1",
        "quant_scale": float(index.get("quant_scale", PLATEAU_QUANT)),
        "uv_scale": uv_scale,
        "n_tiles": len(tiles_out),
        "n_vertices": n_v,
        "n_triangles": n_tri_tex + n_tri_flat,
        "n_triangles_textured": n_tri_tex,
        "n_triangles_flat": n_tri_flat,
        "n_triangles_dropped_shadowed": n_drop,
        "n_atlas": n_atlas,
        "atlas_source_bytes": atlas_bytes,
        "attribution": PLATEAU_TEX_ATTRIBUTION,
        "tiles": tiles_out,
    }


def write_plateau_tex(path: Path, tex: dict) -> Path:
    """JSONP サイドカー(既存 plateau_mesh.js と同方式)。ASCII のみ=文字コード事故なし。"""
    path.write_text("PLATEAU_TEX = "
                    + json.dumps(tex, separators=(",", ":")) + ";", encoding="utf-8")
    return path


# ------------------------------------------------------------------ glb 生成
def _extrude_building(ring, base, top, color, pos, nrm, col, idxs):
    """1 建物の押出しメッシュを配列に追記(座標は local-m Z-up のまま渡す)。"""
    pts = [tuple(p) for p in ring]
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts = pts[:-1]
    n = len(pts)
    if n < 3:
        return
    tris = triangulate(pts)
    if not tris:
        return
    # CCW を保証してから壁の外向き法線を決める
    ccw = _signed_area(pts) >= 0
    base_v = len(pos)

    def add(x, y, z, nx, ny, nz):
        pos.append((x, y, z))
        nrm.append((nx, ny, nz))
        col.append(color)

    # bottom cap 頂点 (下向き法線)  [0 .. n-1]
    for (x, y) in pts:
        add(x, y, base, 0.0, 0.0, -1.0)
    # top cap 頂点 (上向き法線)      [n .. 2n-1]
    for (x, y) in pts:
        add(x, y, top, 0.0, 0.0, 1.0)
    for (i0, i1, i2) in tris:
        # top: そのまま(上から見て CCW → 上向き)
        idxs.extend([base_v + n + i0, base_v + n + i1, base_v + n + i2])
        # bottom: 巻き反転(下向き)
        idxs.extend([base_v + i0, base_v + i2, base_v + i1])
    # side walls (辺ごとに独立頂点=フラット法線)
    for e in range(n):
        p0 = pts[e]
        p1 = pts[(e + 1) % n]
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        L = (dx * dx + dy * dy) ** 0.5 or 1.0
        # CCW ポリゴンの外向き法線 = (dy,-dx)/L
        nx, ny = (dy / L, -dx / L) if ccw else (-dy / L, dx / L)
        v = len(pos)
        add(p0[0], p0[1], base, nx, ny, 0.0)   # 0 b0
        add(p1[0], p1[1], base, nx, ny, 0.0)   # 1 b1
        add(p1[0], p1[1], top, nx, ny, 0.0)    # 2 t1
        add(p0[0], p0[1], top, nx, ny, 0.0)    # 3 t0
        idxs.extend([v + 0, v + 1, v + 2, v + 0, v + 2, v + 3])


def _pad4(b: bytes, fill: bytes) -> bytes:
    r = (-len(b)) % 4
    return b + fill * r


def build_glb(buildings: list, plateau: dict | None = None) -> bytes:
    """建物リスト([{footprint,height,base,kind}]) から glTF 2.0 バイナリを生成。
    座標系変換: local-m (X=east,Y=north,Z=up) → glTF 標準 (x=east, y=up, z=-north)。
    plateau 指定時は照合済み建物を実形状メッシュに置換(未照合は従来押出し)。"""
    pos: list = []
    nrm: list = []
    col: list = []
    idxs: list = []
    for b in buildings:
        color = KIND_COLOR.get(b.get("kind"), DEFAULT_COLOR)
        pm = plateau["meshes"].get(b["id"]) if plateau else None
        if pm is not None:
            _append_mesh(pm[0], pm[1], color, pos, nrm, col, idxs)
            continue
        base = float(b.get("base", 0.0))
        # depth=(levels+below)*FH があればそれを使う(屋上が height に一致する)。
        # 無い呼び出し(旧 dict/テスト)は従来どおり height を深さとして扱う。
        top = base + float(b.get("depth", b.get("height", FLOOR_HEIGHT)))
        _extrude_building(b["footprint"], base, top, color, pos, nrm, col, idxs)

    if not pos:
        pos = [(0.0, 0.0, 0.0)]
        nrm = [(0.0, 0.0, 1.0)]
        col = [DEFAULT_COLOR]
        idxs = [0, 0, 0]

    P = np.array(pos, dtype=np.float32)                  # (N,3) Z-up
    Nn = np.array(nrm, dtype=np.float32)
    # Z-up → Y-up: (x, y, z) -> (x, z, -y)
    P = np.column_stack([P[:, 0], P[:, 2], -P[:, 1]]).astype(np.float32)
    Nn = np.column_stack([Nn[:, 0], Nn[:, 2], -Nn[:, 1]]).astype(np.float32)
    C = np.array([(r, g, bl, 255) for (r, g, bl) in col], dtype=np.uint8)
    I = np.array(idxs, dtype=np.uint32)

    pos_b = P.tobytes()
    nrm_b = Nn.tobytes()
    col_b = _pad4(C.tobytes(), b"\x00")
    idx_b = I.tobytes()

    o0 = 0
    o1 = o0 + len(pos_b)
    o2 = o1 + len(nrm_b)
    o3 = o2 + len(col_b)
    blob = pos_b + nrm_b + col_b + idx_b

    gltf = {
        "asset": {"version": "2.0", "generator": "shibuya-sim export_3d"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "shibuya_buildings"}],
        "materials": [{
            "name": "buildings", "doubleSided": True,
            "pbrMetallicRoughness": {
                "baseColorFactor": [1, 1, 1, 1],
                "metallicFactor": 0.0, "roughnessFactor": 0.9},
        }],
        "meshes": [{"primitives": [{
            "attributes": {"POSITION": 0, "NORMAL": 1, "COLOR_0": 2},
            "indices": 3, "material": 0}]}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(P),
             "type": "VEC3",
             "min": [float(P[:, 0].min()), float(P[:, 1].min()), float(P[:, 2].min())],
             "max": [float(P[:, 0].max()), float(P[:, 1].max()), float(P[:, 2].max())]},
            {"bufferView": 1, "componentType": 5126, "count": len(Nn), "type": "VEC3"},
            {"bufferView": 2, "componentType": 5121, "normalized": True,
             "count": len(C), "type": "VEC4"},
            {"bufferView": 3, "componentType": 5125, "count": len(I), "type": "SCALAR"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": o0, "byteLength": len(pos_b), "target": 34962},
            {"buffer": 0, "byteOffset": o1, "byteLength": len(nrm_b), "target": 34962},
            {"buffer": 0, "byteOffset": o2, "byteLength": len(col_b), "target": 34962},
            {"buffer": 0, "byteOffset": o3, "byteLength": len(idx_b), "target": 34963},
        ],
        "buffers": [{"byteLength": len(blob)}],
    }

    json_b = _pad4(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), b" ")
    bin_b = _pad4(blob, b"\x00")
    total = 12 + 8 + len(json_b) + 8 + len(bin_b)
    out = bytearray()
    out += struct.pack("<4sII", b"glTF", 2, total)
    out += struct.pack("<I4s", len(json_b), b"JSON")
    out += json_b
    out += struct.pack("<I4s", len(bin_b), b"BIN\x00")
    out += bin_b
    return bytes(out)


# ------------------------------------------------------------------ scene.json
def _close_ring(fp: list) -> list:
    ring = [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in fp]
    if len(ring) >= 1 and ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring


def build_scene(city: dict, buildings: list,
                step_minutes: int = STEP_MINUTES) -> dict:
    blds_out = []
    for b in buildings:
        levels = int(b.get("levels", 2) or 2)
        below = int(b.get("below", 0) or 0)
        blds_out.append({
            "id": b["id"],
            "kind": b.get("kind", "generic"),
            "name": b.get("name", ""),
            "footprint": _close_ring(b["footprint"]),
            "levels": levels,
            "below": below,
            "height": round(levels * FLOOR_HEIGHT, 2),
            "base": round(-below * FLOOR_HEIGHT, 2),
            # 押出し深さ = 地下階分も含めた全高。base(=-below*FH)から depth だけ押し出すと
            # 屋上が height に一致する。旧ビューアは depth に height を使っていたため、
            # 地下階を持つ建物は屋上が below*FH だけ沈んでいた(B-3 是正)。
            "depth": round((levels + below) * FLOOR_HEIGHT, 2),
            "cx": round(float(b.get("cx", 0.0)), 1),
            "cy": round(float(b.get("cy", 0.0)), 1),
        })

    roads = [{
        "klass": e.get("klass", "footway"),
        "layer": int(e.get("layer", 0) or 0),
        "g": [[round(float(x), 1), round(float(y), 1)] for x, y in e["geometry"]],
    } for e in city.get("edges", []) if e.get("geometry")]

    rails = [{
        "name": r.get("name", ""),
        "kind": r.get("kind", "rail"),
        "z": -8.0 if r.get("kind") == "subway" else 0.0,
        "g": [[round(float(x), 1), round(float(y), 1)] for x, y in r["geometry"]],
    } for r in city.get("railways", []) if r.get("geometry")]

    # 名前つき POI の上位のみ(建物ひも付けを優先)
    named = [p for p in city.get("pois", []) if p.get("name")]
    named.sort(key=lambda p: (0 if p.get("building") else 1, p.get("name", "")))
    pois = [{
        "x": round(float(p["x"]), 1), "y": round(float(p["y"]), 1),
        "name": p["name"], "cat": p.get("cat", ""),
        "floor": int(p.get("floor", 0) or 0),
        "building": p.get("building", ""),
    } for p in named[:250]]

    meta = city.get("meta", {})
    return {
        "meta": {
            "crs": "local-m",
            "axes": "X=east,Y=north,Z=up",
            "origin_latlon": meta.get("origin_latlon"),
            "bbox": meta.get("bbox"),
            "step_minutes": int(step_minutes),
            "floor_height": FLOOR_HEIGHT,
            "attribution": meta.get("attribution", ""),
            "source": meta.get("name", ""),
        },
        "buildings": blds_out,
        "roads": roads,
        "rails": rails,
        "pois": pois,
    }


# ------------------------------------------------------------------ L1 の省メモリ読み
# reconstruct_tracks が実際に「読む」列と kind。ここに無い列/kind は tracks.json に一切影響しない。
#   列: e["step"] / e["agent_id"] / e["kind"] / e["x"] / e["y"] / e["payload"] / e.get("sim_min")
#       (L1 の rng_stream・llm_call_id は本関数から参照されない)
#   kind: 下の集合以外は for ループの全分岐から外れるので cur/mv/tr_step を変えない。
#         "ride" は --rich-tracks のタクシー振替でのみ使うが、既定でも読み込みだけはしておく
#         (ride は本ループのどの分岐にも当たらないので既定出力は不変)。
TRACK_COLUMNS = ("step", "sim_min", "agent_id", "kind", "x", "y", "payload")
TRACK_KINDS = ("arrive", "enter_area", "enter_building", "exit_area", "exit_building",
               "floor_move", "move_segment", "reflect", "ride", "sleep_start",
               "speak", "traffic_flow", "wake_up",
               # 竹-4 持ち越し①(第86バッチ保守 M-4): 物理ゾーンに所有されている間は
               # move_segment が 1 件も出ない(グラフ移動をしないため)ので、この 2 点
               # (enter/exit の位置)が**ゾーン通過区間の唯一の手掛かり**になる。
               # 物理ゾーン OFF のランには 1 行も存在しない = 既存出力はバイト不変。
               "zone_gate")


def _new_meta_acc() -> dict:
    """「間引きで落とした行にしか無い情報」の集計器(row group 単位で畳む)。"""
    return {"max_step": -1, "step_min": {}, "agent_ids": set(),
            "sorted": True, "last_step": None}


def _feed_meta(tbl, acc: dict) -> None:
    """1 row group ぶんの列を numpy のまま畳む(Python オブジェクト化しない)。

    ここで採る 3 つは全件 to_pylist() 版と**同値**であることが --low-mem の
    バイト同一性の根拠:
      max_step   … `max(e["step"])`
      step_min   … step ごとの「ファイル順で最初の非 null sim_min」(setdefault と同義)
      agent_ids  … `{e["agent_id"] for e in events if e["agent_id"] >= 0}`
    併せて **step 非減少か**も見る(ストリーミング再構成の前提。破れていたら退避する)。
    """
    st = tbl.column("step").to_numpy(zero_copy_only=False)
    acc["max_step"] = max(acc["max_step"], int(st.max()))
    if acc["sorted"]:
        if acc["last_step"] is not None and int(st[0]) < acc["last_step"]:
            acc["sorted"] = False
        elif len(st) > 1 and not bool(np.all(st[1:] >= st[:-1])):
            acc["sorted"] = False
    acc["last_step"] = int(st[-1])
    # step ごとの「最初の非 null sim_min」= 全件版 step_min.setdefault(...) と同一
    sm = tbl.column("sim_min").to_numpy(zero_copy_only=False)
    ok = ~np.isnan(sm) if sm.dtype.kind == "f" else np.ones(len(sm), dtype=bool)
    if ok.any():
        st_ok, sm_ok = st[ok], sm[ok]
        uniq, first = np.unique(st_ok, return_index=True)   # return_index は最初の出現
        step_min = acc["step_min"]
        for s, j in zip(uniq.tolist(), first.tolist()):
            step_min.setdefault(int(s), int(sm_ok[j]))
    aid = np.unique(tbl.column("agent_id").to_numpy(zero_copy_only=False))
    acc["agent_ids"].update(int(a) for a in aid.tolist() if a >= 0)


def _meta_overrides(acc: dict) -> dict:
    return {"n_steps_override": acc["max_step"] + 1,
            "step_min_override": acc["step_min"],
            "agent_ids_override": sorted(acc["agent_ids"])}


def load_track_events(parquet_path: Path,
                      columns: tuple = TRACK_COLUMNS,
                      kinds: tuple = TRACK_KINDS) -> tuple[list, dict]:
    """L1 を row group 単位でストリーム読みし、tracks 再構成に要る行だけ Python へ持ち上げる。

    `pq.read_table(...).to_pylist()` は全行を dict 化するため L1 のサイズに比例して RAM を食う
    (実測 790 B/row = 16.2M イベントで約 12GB)。本関数は
      (a) 列を TRACK_COLUMNS に射影し、
      (b) 行を TRACK_KINDS に絞り(1万体1日ランでは 3.40% しか残らない)、
      (c) 落とした行にしか無い情報 —— n_steps・step ごとの sim_min・全 agent_id ——
          を Arrow/numpy 側(Python オブジェクト化せず)で先に集計して override として返す。
    (c) があるので **戻り値で reconstruct_tracks を呼ぶと全件 to_pylist() と tracks.json が
    バイト同一**になる。順序も row group 順 = ファイル順で保存される。

    ★注意(小粒E): 本関数は **絞ったイベントを全期間ぶん list で持つ**。10日ラン規模では
    これ自体が GB 級になるので、`reconstruct_tracks_streaming` が主経路になった
    (本関数は「L1 が step 昇順でない」ときの退避経路と、既存呼び出しの後方互換用)。

    戻り値: (events, overrides)。overrides は reconstruct_tracks の *_override 引数へそのまま渡す。
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet_path)
    value_set = pa.array(sorted(set(kinds)), type=pa.string())
    events: list = []
    acc = _new_meta_acc()
    for i in range(pf.metadata.num_row_groups):
        tbl = pf.read_row_group(i, columns=list(columns))
        if tbl.num_rows == 0:
            continue
        _feed_meta(tbl, acc)
        events.extend(tbl.filter(pc.is_in(tbl.column("kind"), value_set=value_set)).to_pylist())
        del tbl
    return events, _meta_overrides(acc)


def scan_track_meta(parquet_path: Path, columns: tuple = TRACK_COLUMNS) -> dict:
    """pass1: 列だけを走査して override 3 点 + `step_sorted` を返す(イベントを作らない)。

    メモリは O(row group + n_steps + n_agents)。`step` / `sim_min` / `agent_id` の
    3 列しか読まないので、pass2(本読み)より圧倒的に軽い。
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet_path)
    want = tuple(c for c in ("step", "sim_min", "agent_id") if c in columns)
    acc = _new_meta_acc()
    for i in range(pf.metadata.num_row_groups):
        tbl = pf.read_row_group(i, columns=list(want))
        if tbl.num_rows == 0:
            continue
        _feed_meta(tbl, acc)
        del tbl
    out = _meta_overrides(acc)
    out["step_sorted"] = bool(acc["sorted"])
    return out


def iter_track_events(parquet_path: Path,
                      columns: tuple = TRACK_COLUMNS,
                      kinds: tuple = TRACK_KINDS):
    """pass2: row group ごとに「tracks が読む 7 列 × 対象 kind」だけを dict 化して yield。

    yield 単位は **row group**(list[dict])。呼び出し側が使い終えれば GC される
    = 常駐は 1 row group ぶんだけ。行の順序は row group 順 = ファイル順で、
    `load_track_events` が返す list を同じ順に切っただけのもの(= 出力バイト同一の根拠)。
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet_path)
    value_set = pa.array(sorted(set(kinds)), type=pa.string())
    for i in range(pf.metadata.num_row_groups):
        tbl = pf.read_row_group(i, columns=list(columns))
        if tbl.num_rows == 0:
            continue
        chunk = tbl.filter(pc.is_in(tbl.column("kind"), value_set=value_set)).to_pylist()
        del tbl
        if chunk:
            yield chunk


def _group_by_step(chunks):
    """イベント chunk 列 → `(step, その step のイベント list)` を step 昇順で yield。

    L1 は step 非減少で追記される(`src/society/observer/stream.py` の前提と同じ)。
    連続する同 step をひとまとめにするだけなので、常駐は **1 step ぶん**。
    """
    cur = None
    buf: list = []
    for chunk in chunks:
        for e in chunk:
            s = e["step"]
            if s != cur:
                if buf:
                    yield cur, buf
                cur, buf = s, []
            buf.append(e)
    if buf:
        yield cur, buf


# ------------------------------------------------------------------ tracks.json
W_INDOOR_BASE = 1000        # w = 1000 + bIdx*100 + floor(1..99)
W_FLOOR_MAX = 99            # 1 建物あたり 2 桁 = 99 階まで(桁あふれは w が別建物を指す)


def encode_indoor_w(bld_idx: dict, bld_levels: list, name, floor, stats: dict) -> int:
    """屋内 w(1000 + bIdx*100 + floor)を**安全に**組む(B-1・表示側の防御)。

    - 未知の建物名(bld_idx に無い)は idx0 に黙って落とすと「無関係な建物の中に人が居る」
      表示になるため、w=0(路上)へ退避して件数を数える。
    - floor は 1..min(levels,99) にクランプ(負値・0・桁あふれ・levels 超えを是正)。
      sim 側(scheduler)の floor はここでは変えない=L1 は不変。表示だけを実在の階に収める。
    stats は {"unknown": n, "clamped": n, "unknown_names": {name: n}} を積む。"""
    bi = bld_idx.get(name)
    if bi is None:
        stats["unknown"] = stats.get("unknown", 0) + 1
        names = stats.setdefault("unknown_names", {})
        key = str(name)
        names[key] = names.get(key, 0) + 1
        return 0
    try:
        f = int(floor)
    except (TypeError, ValueError):
        f = 1
    lv = bld_levels[bi] if 0 <= bi < len(bld_levels) else W_FLOOR_MAX
    hi = max(1, min(int(lv or 1), W_FLOOR_MAX))
    f2 = max(1, min(f, hi))
    if f2 != f:
        stats["clamped"] = stats.get("clamped", 0) + 1
    return W_INDOOR_BASE + bi * 100 + f2


def reconstruct_tracks(events: list, buildings: list, agents_meta: list,
                       sample_agents: int | None = None,
                       step_stride: int = 1, rich_tracks: bool = False,
                       n_steps_override: int | None = None,
                       step_min_override: dict | None = None,
                       agent_ids_override: list | None = None,
                       step_minutes: int = STEP_MINUTES) -> dict:
    """viz/make_viewer.py build_data の位置再構成を移植・整理。
    positions[step][i] = [x, y, w]  (w: 0=路上 -1=範囲外 -2=睡眠 1000+bIdx*100+floor=屋内)
    moves[step][i] = [mode, pts] または None,  traffic[step] = {n, segs}

    B4-lite(スケール): sample_agents/step_stride で LOD 出力(既定=全量・現行と同一)。
    - sample_agents=N: 対象を先頭 N エージェントに絞る(位置配列の幅を N に)。
    - step_stride=K: 状態は毎 step 更新しつつ、K step ごとに1フレームだけ出力する。

    rich_tracks(--rich-tracks・既定 OFF=現行とバイト同一):
    - moves の mode に 3=タクシー を追加(ride イベントと同一 agent×step の car 区間を振替)。
    - positions の w に -3=電車で圏外 を追加(exit_area payload via=="train" のとき)。
    - meta に mode_legend / away_train を追加。

    *_override(--low-mem 経路・既定 None=従来どおり events から算出):
    load_track_events が「間引きで落とした行にしか無い情報」を先に集計して渡すための口。
    None のときは一切参照されないので、既定経路は従来とバイト同一。

    竹-4 持ち越し①(第86バッチ保守 M-4): 物理ゾーン(physics.zones)に所有されている間、
    その個体は**グラフ移動をしない**ので move_segment が 1 件も出ず、位置が入口で固まって
    出口で瞬間移動する。zone_gate(enter/exit。どちらも位置を持つ)で挟んで**直線補間**し、
    w=0(路上)として埋める。実績は meta["zone_interp"](= {method, segments, frames,
    unclosed})に残す。★これは実軌跡ではない: 物理の dt_sub 刻みの実軌跡は L1 に記録されて
    いない(将来課題)。zone_gate が 1 件も無いラン(= 物理 OFF の既存ラン)では
    この経路を一度も通らず、出力は従来とバイト同一。

    ★小粒E(2026-08-07): 本関数は「events を全部 RAM に持つ」API のまま据え置き、
    再構成の本体を `_reconstruct_core`(step 単位のバケツしか持たない)へ切り出した。
    ここでの by_step 構築と `_reconstruct_core` の pending 送りは同じ列を作るので、
    出力は 1 バイトも変わらない(tests/test_export3d.py が新旧突合で固定)。
    L1 から直に流す経路は `reconstruct_tracks_streaming`。"""
    n_steps = (max((e["step"] for e in events), default=-1) + 1
               if n_steps_override is None else int(n_steps_override))
    agent_ids_seed = (sorted({e["agent_id"] for e in events if e["agent_id"] >= 0})
                      if agent_ids_override is None else list(agent_ids_override))
    step_min: dict[int, int] = ({} if step_min_override is None
                                else {int(k): int(v) for k, v in step_min_override.items()})
    by_step: dict[int, list[dict]] = defaultdict(list)
    for e in events:
        by_step[e["step"]].append(e)
        if step_min_override is not None:
            continue
        sm = e.get("sim_min")
        if sm is not None:
            step_min.setdefault(e["step"], int(sm))
    groups = ((s, by_step[s]) for s in sorted(by_step))
    return _reconstruct_core(groups, n_steps, agent_ids_seed, step_min,
                             buildings, agents_meta, sample_agents=sample_agents,
                             step_stride=step_stride, rich_tracks=rich_tracks,
                             step_minutes=step_minutes)


def reconstruct_tracks_streaming(parquet_path: Path, buildings: list, agents_meta: list,
                                 sample_agents: int | None = None,
                                 step_stride: int = 1, rich_tracks: bool = False,
                                 columns: tuple = TRACK_COLUMNS,
                                 kinds: tuple = TRACK_KINDS,
                                 step_minutes: int = STEP_MINUTES) -> dict:
    """L1 parquet から**イベント列を全保持せずに** tracks を組む(--low-mem の主経路)。

    pass1 = `scan_track_meta`(3 列だけ走査)/ pass2 = `iter_track_events`(row group 逐次)。
    `reconstruct_tracks(load_track_events(...))` と **出力バイト同一**で、
    イベント側の常駐が O(全期間) → O(row group + 1 step) になる。

    L1 が step 昇順でないとき(通常あり得ない。logger は step 非減少で追記する)は、
    step 単位のバケツ化が成立しないので **黙って結果を変えず**従来の全保持経路へ退避する。
    """
    meta = scan_track_meta(parquet_path, columns=columns)
    if not meta.pop("step_sorted"):
        print("  [tracks] L1 が step 昇順でない → ストリーミング再構成を退避"
              "(従来の全保持経路で同一結果を作る)")
        events, ov = load_track_events(parquet_path, columns, kinds)
        return reconstruct_tracks(events, buildings, agents_meta,
                                  sample_agents=sample_agents, step_stride=step_stride,
                                  rich_tracks=rich_tracks, step_minutes=step_minutes, **ov)
    groups = _group_by_step(iter_track_events(parquet_path, columns, kinds))
    return _reconstruct_core(groups, int(meta["n_steps_override"]),
                             list(meta["agent_ids_override"]),
                             {int(k): int(v) for k, v in meta["step_min_override"].items()},
                             buildings, agents_meta, sample_agents=sample_agents,
                             step_stride=step_stride, rich_tracks=rich_tracks,
                             step_minutes=step_minutes)


def _reconstruct_core(step_groups, n_steps: int, agent_ids_seed: list,
                      step_min: dict, buildings: list, agents_meta: list,
                      sample_agents: int | None = None,
                      step_stride: int = 1, rich_tracks: bool = False,
                      step_minutes: int = STEP_MINUTES) -> dict:
    """位置再構成の本体。`step_groups` は `(step, その step のイベント list)` を
    **step 昇順**で出す iterable(list でも generator でもよい)。

    常駐するのは「いま処理している step のイベント」だけ = イベント側は O(1 step)。
    残る O(n_steps × n_agents) は **出力そのもの**(positions/moves)なので、これ以上は
    tracks.json の構造を変えないと減らない(tracks.bin 側の分割出力が別の答え)。
    """
    bld_idx = {b["id"]: i for i, b in enumerate(buildings)}
    bld_levels = [max(1, int(b.get("levels", 2) or 2)) for b in buildings]
    w_stats: dict = {}                 # B-1: floor クランプ / 未知建物の件数(黙って落とさない)

    agent_ids = list(agent_ids_seed)
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

    mode_code = {"walk": 0, "bicycle": 1, "car": 2}
    groups = iter(step_groups)
    pending = next(groups, None)
    positions, moves, traffic = [], [], []
    cur = [[0.0, 0.0, 0] for _ in agent_ids]
    # 竹-4 持ち越し①: ゾーン所有中(zone_gate enter → exit)の位置の穴。
    #   zone_open[i] = (enter step, (x, y)) / zone_fills = 埋める区間
    zone_open: dict[int, tuple[int, tuple[float, float]]] = {}
    zone_fills: list[tuple[int, int, tuple[float, float], int, tuple[float, float]]] = []
    for step in range(n_steps):
        mv = [None] * len(agent_ids)
        tr_step = {"n": 0, "segs": []}
        # この step のイベントだけを取り出す(step 昇順の前提。過去の step は捨てる)。
        while pending is not None and pending[0] < step:
            pending = next(groups, None)
        if pending is not None and pending[0] == step:
            evs = pending[1]
            pending = next(groups, None)
        else:
            evs = ()
        # --rich-tracks 用: taxi 乗車(ride イベント, payload mode=="taxi")の (agent_id, step) 集合。
        # 対応付け: ride と同一 agent×step の move_segment(mode=car)がタクシー乗車区間。
        # 実データ runs/demo_event_200a3d では ride 79件すべてが同一 step の move_segment 終点と
        # (x,y)一致し、うち mode=car は 33件(残り46件は sim が walk で記録=car ではないため非対象)。
        # 既定 OFF では空集合=振替なし=現行と完全同一。
        # ★参照は必ず「同一 step」なので、全期間ぶんを溜める必要はない(= step 内で完結)。
        taxi_steps: set = set()
        if rich_tracks:
            for e in evs:
                if e["kind"] == "ride":
                    pr = json.loads(e["payload"]) if e.get("payload") else {}
                    if pr.get("mode") == "taxi":
                        taxi_steps.add((e["agent_id"], e["step"]))
        for e in evs:
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
    sim_min = [step_min.get(s, start_min + s * int(step_minutes)) for s in emitted]

    agents_slim = [{
        "id": a["id"], "name": a.get("name", f"agent{a['id']}"),
        "visitor": bool(a.get("visitor", False)),
        "occupation": a.get("occupation", "?"),
        "age": a.get("age", 0), "gender": a.get("gender", "?"),
        "has_car": bool(a.get("has_car", False)),
        "has_bicycle": bool(a.get("has_bicycle", False)),
    } for a in agents_meta]

    meta = {"nSteps": len(positions), "step_minutes": int(step_minutes),
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


# ------------------------------------------------------------------ 逐次 JSON 書き出し
def dump_json_stream(path: Path, obj, *, ensure_ascii: bool = True,
                     separators: tuple = (",", ":"), chunk_chars: int = 1 << 20) -> None:
    """`json.dumps(obj, ...)` と **同一バイト列** をファイルへ逐次書き出す。

    なぜ: `Path.write_text(json.dumps(tracks, ...))` は「完成した JSON 文字列そのもの」を
    一度 RAM に作る。tracks.json は 1万体×1日で 65.8MB、10日ラン規模では数百 MB〜GB 級
    になるので、**tracks dict とは別に**同じだけのピークを積む。iterencode は同じ
    エンコーダで断片を出すだけなので、連結すれば dumps と 1 バイトも変わらない
    (tests/test_export3d.py が両者のバイト一致を固定する)。
    open() の引数は `Path.write_text(..., encoding="utf-8")` と同一(newline 変換も同条件)。

    どう書くか: 上位 `depth` 段だけ手で開いて(`{`/`[`/`,`/`:`)、**葉は `dumps` の
    C エンコーダに一発で投げる**。`json.JSONEncoder.iterencode` は C 経路を通らないので
    素朴に使うと遅い(実測 tracks.json 82.8MB で +5.5s)。tracks は
    `{"positions": [フレーム, …], …}` の形なので depth=2 で「1 フレームずつ C エンコード」
    になり、速度は dumps とほぼ同じままピークだけ落ちる。
    セパレータに空白が無いので `dumps(d) == "{" + ",".join(dumps(k)+":"+dumps(v)) + "}"`
    が厳密に成り立つ(= バイト同一の根拠)。非 str キーの dict は手で開かず丸ごと
    `dumps` へ落とす(キーの文字列化規則を再実装しない)。

    ★使いどころ: **--low-mem のときだけ**。既定経路は従来どおり dumps 一発。
    """
    enc = json.JSONEncoder(ensure_ascii=ensure_ascii, separators=separators)
    buf: list = []
    size = 0
    with open(path, "w", encoding="utf-8") as f:
        for piece in _iter_json_chunks(obj, enc, separators, 2):
            buf.append(piece)
            size += len(piece)
            if size >= chunk_chars:
                f.write("".join(buf))
                buf.clear()
                size = 0
        if buf:
            f.write("".join(buf))


def _iter_json_chunks(obj, enc, separators: tuple, depth: int):
    """`json.dumps(obj)` と同一の文字列を、上位 depth 段だけ分割して yield する。"""
    item_sep, key_sep = separators
    if depth <= 0 or not obj or not isinstance(obj, (dict, list)):
        yield enc.encode(obj)                     # 葉 = C エンコーダ一発
        return
    if isinstance(obj, dict):
        if not all(type(k) is str for k in obj):  # 非 str キーは dumps に任せる
            yield enc.encode(obj)
            return
        yield "{"
        first = True
        for k, v in obj.items():
            if not first:
                yield item_sep
            first = False
            yield enc.encode(k) + key_sep
            yield from _iter_json_chunks(v, enc, separators, depth - 1)
        yield "}"
        return
    yield "["
    first = True
    for v in obj:
        if not first:
            yield item_sep
        first = False
        yield from _iter_json_chunks(v, enc, separators, depth - 1)
    yield "]"


# ------------------------------------------------------------------ top-level
def export_run(run_dir: Path, map_path: Path | None = None,
               sample_agents: int | None = None, step_stride: int = 1,
               plateau_dir: Path | None = None, rich_tracks: bool = False,
               low_mem: bool = False, tracks_binary: bool = False,
               write_tracks_json: bool = True,
               chunk_bytes: int | None = None,
               plateau_tex: bool = False) -> dict:
    if not tracks_binary and not write_tracks_json:
        raise SystemExit("[export_3d] --no-tracks-json は --tracks-binary と併用する")
    if plateau_tex and plateau_dir is None:
        raise SystemExit("[export_3d] --plateau-tex は --plateau(または --plateau-dir)と併用する")
    if plateau_tex:
        # 標準経路(縮退策②): テクスチャ 65MB を積む分、軌跡は分離版から追い出す。
        # tracks.json は書いたままなので埋め込み版 viewer3d.html は自己完結を保つ。
        tracks_binary = True
    run_dir = Path(run_dir)
    l1_path = run_dir / "l1_events.parquet"
    if low_mem:            # 追加専用: 出力バイトは既定経路と同一(reconstruct_tracks_streaming)
        events = None      # ★イベント列は 1 度も丸ごと作らない(常駐 = row group + 1 step)
    else:
        import pyarrow.parquet as pq
        events = pq.read_table(l1_path).to_pylist()

    mp = _resolve_map_path(run_dir, map_path)
    city = json.loads(mp.read_text(encoding="utf-8"))
    buildings = city.get("buildings", [])

    am_path = run_dir / "agents.json"
    agents_meta = json.loads(am_path.read_text(encoding="utf-8")) if am_path.exists() else []

    # W2-3: 1 step の分数は **このランの run.dt_min**(Δt=10 なら 10 = 従来と同値)。
    step_minutes = _load_run_dt().min_per_step(run_dir)
    scene = build_scene(city, buildings, step_minutes)
    plateau = _load_plateau(plateau_dir) if plateau_dir is not None else None
    if plateau:
        for b in scene["buildings"]:
            if b["id"] in plateau["meshes"]:
                b["height"] = round(plateau["heights"][b["id"]], 2)  # LOD 実測で上書き
                b["plateau"] = 1                                     # 追加専用キー
                # 実効階高(B-2): levels は sim 側の意味を保つため据え置き、実測高との
                # 整合は floorH = height/levels で表す。ビューアの屋内フロア高はこれを使う
                # (levels*3.5 と実測高が食い違う建物で人が屋根を突き抜けるのを止める)。
                lv = max(1, int(b.get("levels", 1) or 1))
                b["floorH"] = round(b["height"] / lv, 3)
                b["depth"] = round(b["height"] - b["base"], 2)       # 押出し深さも実測へ
    # 地表グリッド(--plateau 時・terrain.npz/json がある場合のみ)。無ければ従来どおり gz なし。
    terrain = _load_terrain(plateau_dir) if plateau_dir is not None else None
    if terrain is not None:
        for b in scene["buildings"]:
            # A-6: 重心1点ではなく footprint 頂点の最小地表高。斜面上の建物で
            # 「基準面が実際の足元より高い」= 山側の壁が地面に埋まる/谷側が浮く のを防ぐ
            # (谷側の隙間は viewer 側の 3m スカートで埋める)。
            fp = b.get("footprint") or []
            gz = [_sample_terrain_gz(terrain, p[0], p[1]) for p in fp]
            b["gz"] = round(min(gz), 1) if gz else _sample_terrain_gz(
                terrain, b["cx"], b["cy"])                            # 追加専用キー
    if low_mem:
        tracks = reconstruct_tracks_streaming(
            l1_path, buildings, agents_meta, sample_agents=sample_agents,
            step_stride=step_stride, rich_tracks=rich_tracks,
            step_minutes=step_minutes)
    else:
        tracks = reconstruct_tracks(events, buildings, agents_meta,
                                    sample_agents=sample_agents,
                                    step_stride=step_stride, rich_tracks=rich_tracks,
                                    step_minutes=step_minutes)
        events = None                      # 以降は tracks しか要らない(早めに解放)
    glb = build_glb(scene["buildings"], plateau)

    out_dir = run_dir / "scene3d"
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_p = out_dir / "scene.json"
    tracks_p = out_dir / "tracks.json"
    glb_p = out_dir / "buildings.glb"
    scene_p.write_text(json.dumps(scene, ensure_ascii=False, separators=(",", ":")),
                       encoding="utf-8")
    if write_tracks_json:
        if low_mem:
            # --low-mem = メモリが律速のモード。dumps の戻り値(10日ランで GB 級の 1 本の
            # str)を作らずに書く。★バイト列は dumps と同一(tests が固定)。
            # 既定経路で使わない理由: 実測で **既定経路のピークは JSON 文字列で決まらない**
            # (wv_mock_30d 実測: 既定は逐次書きにしても 1985.5MB → 1985.7MB で不変。
            #  ピークは to_pylist() のイベント dict が握っている)。効くのは --low-mem の
            # ときだけ(1895.7MB → 1634.7MB)なので、作用のある側にだけ入れる。
            dump_json_stream(tracks_p, tracks, ensure_ascii=False, separators=(",", ":"))
        else:
            tracks_p.write_text(json.dumps(tracks, ensure_ascii=False,
                                           separators=(",", ":")), encoding="utf-8")
    glb_p.write_bytes(glb)
    res = {"scene": scene_p, "glb": glb_p,
           "n_buildings": len(scene["buildings"]), "n_steps": tracks["meta"]["nSteps"]}
    if write_tracks_json:
        res["tracks"] = tracks_p
    if tracks_binary:                                 # 追加専用: 既定 OFF=既存出力バイト同一
        tb = _load_tracks_bin()
        kw = {"chunk_bytes": chunk_bytes} if chunk_bytes else {}
        blob, header = tb.encode_tracks(tracks, **kw)
        bin_p = out_dir / "tracks.bin"
        meta_p = out_dir / "tracks_meta.json"
        bin_p.write_bytes(blob)
        # tracks.bin 内の JSON ヘッダと**同一文字列**(ビューアはこちらを埋め込む)
        meta_p.write_text(json.dumps(header, ensure_ascii=False, separators=(",", ":")),
                          encoding="utf-8")
        res["tracks_bin"] = bin_p
        res["tracks_meta"] = meta_p
    if plateau:
        web = build_plateau_web(scene["buildings"], plateau)
        extras = _load_extras(plateau_dir)                # extras.npz があれば地下街/橋を同梱
        if extras:                                        # 無ければ "extras" キー自体を出さない
            web["extras"] = build_extras_web(extras)
        ubld4 = _load_ubld4(plateau_dir)                   # LOD4.1 地下街(面種別+層)
        if ubld4 is not None:                              # 無ければ "ubld4" キー自体を出さない
            web["ubld4"] = build_ubld4_web(ubld4)
        web_p = out_dir / "plateau_web.json"
        web_p.write_text(json.dumps(web, separators=(",", ":")), encoding="utf-8")
        res["plateau_web"] = web_p
        res["n_plateau"] = len(web["matched_ids"])
        if extras:
            res["n_extras"] = {k: v["n_triangles"] for k, v in web["extras"].items()}
        if ubld4 is not None:
            res["n_ubld4"] = web["ubld4"]["n_triangles"]
    if plateau_tex:                                   # 追加専用: 既定 OFF=既存出力バイト同一
        tex = build_plateau_tex(plateau_dir / "tiles_lod2")
        tex_p = write_plateau_tex(run_dir / "plateau_tex.js", tex)
        res["plateau_tex"] = tex_p
        res["tex_stats"] = {k: tex[k] for k in
                            ("n_tiles", "n_vertices", "n_triangles",
                             "n_triangles_textured", "n_triangles_flat",
                             "n_triangles_dropped_shadowed", "n_atlas")}
    if terrain is not None:
        tw_p = out_dir / "terrain_web.json"
        tw_p.write_text(json.dumps(build_terrain_web(terrain), separators=(",", ":")),
                        encoding="utf-8")
        res["terrain_web"] = tw_p
    return res


def main(argv: list) -> int:
    if not argv:
        print(__doc__)
        return 1
    run_dir = Path(argv[0]).resolve()
    map_override = None
    if "--map" in argv:
        map_override = Path(argv[argv.index("--map") + 1])
    sample_agents = None
    if "--sample-agents" in argv:
        sample_agents = int(argv[argv.index("--sample-agents") + 1])
    step_stride = 1
    if "--step-stride" in argv:
        step_stride = int(argv[argv.index("--step-stride") + 1])
    plateau_dir = None
    if "--plateau" in argv:
        plateau_dir = REPO_ROOT / "data" / "plateau"
    if "--plateau-dir" in argv:
        plateau_dir = Path(argv[argv.index("--plateau-dir") + 1])
        if not plateau_dir.is_absolute():
            plateau_dir = REPO_ROOT / plateau_dir
    rich_tracks = "--rich-tracks" in argv
    low_mem = "--low-mem" in argv
    tracks_binary = "--tracks-binary" in argv
    write_tracks_json = "--no-tracks-json" not in argv
    chunk_bytes = None
    if "--chunk-mb" in argv:
        chunk_bytes = int(float(argv[argv.index("--chunk-mb") + 1]) * 1024 * 1024)
    plateau_tex = "--plateau-tex" in argv
    res = export_run(run_dir, map_override, sample_agents=sample_agents,
                     step_stride=step_stride, plateau_dir=plateau_dir,
                     rich_tracks=rich_tracks, low_mem=low_mem,
                     tracks_binary=tracks_binary,
                     write_tracks_json=write_tracks_json, chunk_bytes=chunk_bytes,
                     plateau_tex=plateau_tex)
    keys = (("scene",)
            + (("tracks",) if "tracks" in res else ())
            + ("glb",)
            + (("tracks_bin", "tracks_meta") if "tracks_bin" in res else ())
            + (("plateau_web",) if "plateau_web" in res else ())
            + (("terrain_web",) if "terrain_web" in res else ())
            + (("plateau_tex",) if "plateau_tex" in res else ()))
    for k in keys:
        sz = res[k].stat().st_size
        try:
            shown = res[k].relative_to(REPO_ROOT)
        except ValueError:
            shown = res[k]
        print(f"  {shown}  ({sz/1024:.1f} KB)")
    tail = f"  plateau={res['n_plateau']}" if "n_plateau" in res else ""
    if "n_extras" in res:
        tail += f"  extras={res['n_extras']}"
    if "n_ubld4" in res:
        tail += f"  ubld4={res['n_ubld4']}三角形"
    print(f"  buildings={res['n_buildings']}  steps={res['n_steps']}{tail}")
    if "tex_stats" in res:
        s = res["tex_stats"]
        print(f"  [tex] タイル {s['n_tiles']}(アトラス {s['n_atlas']})"
              f"  三角形 {s['n_triangles']}(テクスチャ {s['n_triangles_textured']}"
              f" / 無地 {s['n_triangles_flat']})"
              f"  影落とし除外 {s['n_triangles_dropped_shadowed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
