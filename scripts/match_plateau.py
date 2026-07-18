"""シミュの OSM 建物 ID(例 b123) ⇄ PLATEAU 建物(gml_id) の 1 対 1 対応表を作る。

下流の export_3d --plateau が「この建物は実形状メッシュに置換」を判定するために使う。

使い方:
  python scripts/match_plateau.py [--index data/plateau/plateau_index.json]
                                  [--osm data/shibuya_osm.json]
                                  [--out data/plateau/plateau_match.json]

入力:
  - OSM 側: data/shibuya_osm.json の buildings。id と footprint(local-m)。
    footprint の読み方・座標系は scripts/export_3d.py と同一(json をそのまま読み、
    city["buildings"][*]["footprint"] を local-m の [[x,y],...] として使う)。
  - PLATEAU 側: data/plateau/plateau_index.json(別工程で生成)。想定スキーマ:
    {"buildings":[{"gml_id":str,"footprint":[[x,y],...],"height":float,
                   "base":float,"n_tris":int,"lod":int}], "ground0":..., ...}
    footprint は local-m。ファイル未生成でも本モジュールのテストは合成データで通る
    (読み込みを関数分離し、コアの match() は dict を直接受け取る)。

アルゴリズム:
  1. 候補絞り込み: footprint 重心の全対全距離(numpy ブロードキャスト)→ 25m 以内を候補。
  2. footprint IoU: 0.5m グリッドのラスタ IoU。両 footprint の bbox 和集合を 0.5m 格子に
     離散化し、点 in 多角形(偶奇規則・純 numpy ベクトル化)で塗って intersection/union。
     凹形状もそのまま扱える(多角形クリッピング自作より頑健)。
  3. 採用: IoU>=0.4。1 つの gml_id に複数の osm_id が競合したら IoU 最大が勝ち、敗者は
     自分の次候補で再試行(IoU 降順の貪欲マッチング)。
  4. 出力: plateau_match.json(matches / unmatched_osm / unmatched_plateau_count / params)。
     stdout にマッチ率・IoU 分布(min/median/max)。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

# 既定パラメータ(出力 params にもそのまま記録する)
DEFAULT_KNN_RADIUS_M = 25.0
DEFAULT_IOU_MIN = 0.4
DEFAULT_GRID_M = 0.5

# グリッド生成の安全弁(1 ペアあたりのセル数上限)。通常の建物 bbox では到達しない。
_MAX_GRID_CELLS = 4_000_000


# ------------------------------------------------------------------ footprint 幾何
def _ring(fp) -> np.ndarray:
    """footprint を (V,2) float 配列にし、閉環の重複末尾頂点があれば除去する。
    偶奇規則の PIP は末尾→先頭の辺を暗黙に閉じるので、重複末尾は不要かつ零長辺を生む。"""
    a = np.asarray(fp, dtype=float)
    if a.ndim != 2 or a.shape[0] < 3 or a.shape[1] < 2:
        return a.reshape(-1, 2) if a.size else np.empty((0, 2), dtype=float)
    a = a[:, :2]
    if a.shape[0] >= 2 and np.array_equal(a[0], a[-1]):
        a = a[:-1]
    return a


def centroid(fp) -> np.ndarray:
    """footprint 重心。頂点平均(候補絞り込みの 25m 半径に対しては十分・退化にも頑健)。"""
    a = _ring(fp)
    if a.shape[0] == 0:
        return np.array([0.0, 0.0])
    return a.mean(axis=0)


def points_in_polygon(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """偶奇規則(ray casting)の点 in 多角形。純 numpy ベクトル化。

    pts:  (N,2) 判定点。 poly: (V,2) 多角形頂点(閉じていなくてよい)。
    戻り値: (N,) bool。辺上の扱いは PNPOLY 準拠(厳密な境界一致は不定だが面積計算では影響小)。
    """
    poly = np.asarray(poly, dtype=float)
    pts = np.asarray(pts, dtype=float)
    n = poly.shape[0]
    if n < 3 or pts.shape[0] == 0:
        return np.zeros(pts.shape[0], dtype=bool)
    x = pts[:, 0]
    y = pts[:, 1]
    inside = np.zeros(pts.shape[0], dtype=bool)
    xi = poly[:, 0]
    yi = poly[:, 1]
    xj = np.roll(xi, 1)   # 直前の頂点(j = i-1、末尾→先頭で閉じる)
    yj = np.roll(yi, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        for k in range(n):
            # 辺 (xj[k],yj[k]) -> (xi[k],yi[k]) が水平走査線 y をまたぐか
            cond = (yi[k] > y) != (yj[k] > y)
            # またぐ辺のみ交点 x を計算(cond=False の行は 0 除算になり得るが cond で除外)
            x_int = (xj[k] - xi[k]) * (y - yi[k]) / (yj[k] - yi[k]) + xi[k]
            inside ^= cond & (x < x_int)
    return inside


def raster_iou(fp_a, fp_b, grid_m: float = DEFAULT_GRID_M) -> float:
    """0.5m グリッドのラスタ IoU。両 footprint の bbox 和集合を格子に離散化して塗り分ける。"""
    a = _ring(fp_a)
    b = _ring(fp_b)
    if a.shape[0] < 3 or b.shape[0] < 3:
        return 0.0
    minx = min(a[:, 0].min(), b[:, 0].min())
    miny = min(a[:, 1].min(), b[:, 1].min())
    maxx = max(a[:, 0].max(), b[:, 0].max())
    maxy = max(a[:, 1].max(), b[:, 1].max())
    nx = int(np.ceil((maxx - minx) / grid_m))
    ny = int(np.ceil((maxy - miny) / grid_m))
    if nx <= 0 or ny <= 0:
        return 0.0
    if nx * ny > _MAX_GRID_CELLS:
        # 想定外に巨大な bbox: グリッドを粗くして安全弁(実運用の建物では起きない)
        scale = np.sqrt(nx * ny / _MAX_GRID_CELLS)
        grid_m = grid_m * scale
        nx = int(np.ceil((maxx - minx) / grid_m))
        ny = int(np.ceil((maxy - miny) / grid_m))
    # セル中心の座標
    xs = minx + (np.arange(nx) + 0.5) * grid_m
    ys = miny + (np.arange(ny) + 0.5) * grid_m
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    ina = points_in_polygon(pts, a)
    inb = points_in_polygon(pts, b)
    union = int(np.count_nonzero(ina | inb))
    if union == 0:
        return 0.0
    inter = int(np.count_nonzero(ina & inb))
    return inter / union


# ------------------------------------------------------------------ コアのマッチング
def match(osm_buildings: list, plateau_buildings: list,
          knn_radius_m: float = DEFAULT_KNN_RADIUS_M,
          iou_min: float = DEFAULT_IOU_MIN,
          grid_m: float = DEFAULT_GRID_M) -> dict:
    """OSM 建物と PLATEAU 建物の 1 対 1 対応を作る(実データ非依存・dict を直接受ける)。

    osm_buildings:     [{"id": "b123", "footprint": [[x,y],...]}, ...]
    plateau_buildings: [{"gml_id": "...", "footprint": [[x,y],...], ...}, ...]
    戻り値: {"matches", "unmatched_osm", "unmatched_plateau_count", "params"}。
    """
    osm_ids = [str(b["id"]) for b in osm_buildings]
    gml_ids = [str(b["gml_id"]) for b in plateau_buildings]
    osm_fps = [b["footprint"] for b in osm_buildings]
    pla_fps = [b["footprint"] for b in plateau_buildings]

    params = {"knn_radius_m": knn_radius_m, "iou_min": iou_min, "grid_m": grid_m}

    if not osm_ids or not gml_ids:
        return {
            "matches": {},
            "unmatched_osm": list(osm_ids),
            "unmatched_plateau_count": len(gml_ids),
            "params": params,
        }

    # 1) 候補絞り込み: 重心の全対全距離(numpy ブロードキャスト)→ 半径以内
    osm_c = np.array([centroid(fp) for fp in osm_fps])            # (M,2)
    pla_c = np.array([centroid(fp) for fp in pla_fps])            # (P,2)
    diff = osm_c[:, None, :] - pla_c[None, :, :]                  # (M,P,2)
    dist = np.sqrt((diff ** 2).sum(axis=2))                       # (M,P)
    cand = dist <= knn_radius_m

    # 2) 候補ペアの IoU を計算し、IoU>=iou_min のペアのみ残す
    #    pairs: (iou, dist, osm_index, pla_index)
    pairs = []
    oi_arr, pi_arr = np.nonzero(cand)
    for oi, pi in zip(oi_arr.tolist(), pi_arr.tolist()):
        iou = raster_iou(osm_fps[oi], pla_fps[pi], grid_m=grid_m)
        if iou >= iou_min:
            pairs.append((iou, float(dist[oi, pi]), oi, pi))

    # 3) 貪欲マッチング: IoU 降順。両者未使用なら採用(敗者は次候補ペアで自然に再試行される)
    #    同 IoU は距離が近い方を優先し、決定性のため index も tie-break に使う。
    pairs.sort(key=lambda t: (-t[0], t[1], t[2], t[3]))
    used_osm = set()
    used_pla = set()
    matches: dict = {}
    for iou, d, oi, pi in pairs:
        if oi in used_osm or pi in used_pla:
            continue
        matches[osm_ids[oi]] = {
            "gml_id": gml_ids[pi],
            "iou": round(float(iou), 4),
            "dist_m": round(float(d), 2),
        }
        used_osm.add(oi)
        used_pla.add(pi)

    unmatched_osm = [osm_ids[i] for i in range(len(osm_ids)) if i not in used_osm]
    unmatched_plateau_count = len(gml_ids) - len(used_pla)

    return {
        "matches": matches,
        "unmatched_osm": unmatched_osm,
        "unmatched_plateau_count": unmatched_plateau_count,
        "params": params,
    }


# ------------------------------------------------------------------ 入出力
def load_osm_buildings(path: Path) -> list:
    """shibuya_osm.json を export_3d.py と同じ読み方で読み、[{id, footprint}] を返す。"""
    city = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for b in city.get("buildings", []):
        fp = b.get("footprint")
        if not fp:
            continue
        out.append({"id": b["id"], "footprint": fp})
    return out


def load_plateau_index(path: Path) -> list:
    """plateau_index.json を読み、[{gml_id, footprint, ...}] を返す。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for b in data.get("buildings", []):
        fp = b.get("footprint")
        if not fp:
            continue
        out.append({"gml_id": b["gml_id"], "footprint": fp})
    return out


def _iou_distribution(matches: dict) -> tuple:
    """マッチした IoU の (min, median, max)。空なら (None, None, None)。"""
    if not matches:
        return (None, None, None)
    ious = np.array([m["iou"] for m in matches.values()], dtype=float)
    return (float(ious.min()), float(np.median(ious)), float(ious.max()))


# ------------------------------------------------------------------ CLI
def main(argv: list) -> int:
    # Windows コンソール(cp932)対策: 進捗 print の非 cp932 文字で死なない(synth_crowd と同修正)。
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(
        description="OSM 建物 ID ⇄ PLATEAU gml_id の 1 対 1 対応表を作る")
    ap.add_argument("--index", type=Path,
                    default=REPO_ROOT / "data" / "plateau" / "plateau_index.json",
                    help="PLATEAU インデックス(既定 data/plateau/plateau_index.json)")
    ap.add_argument("--osm", type=Path,
                    default=REPO_ROOT / "data" / "shibuya_osm.json",
                    help="シミュ側 OSM マップ(既定 data/shibuya_osm.json)")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "data" / "plateau" / "plateau_match.json",
                    help="対応表の出力先(既定 data/plateau/plateau_match.json)")
    ap.add_argument("--knn-radius-m", type=float, default=DEFAULT_KNN_RADIUS_M)
    ap.add_argument("--iou-min", type=float, default=DEFAULT_IOU_MIN)
    ap.add_argument("--grid-m", type=float, default=DEFAULT_GRID_M)
    args = ap.parse_args(argv)

    if not args.osm.exists():
        print(f"[match_plateau] OSM マップが見つからない: {args.osm}")
        return 1
    if not args.index.exists():
        print(f"[match_plateau] PLATEAU インデックスが未生成: {args.index}")
        print("  → 抽出(plateau_index.json 生成)完了後に再実行してください。"
              "偽データは作りません。")
        return 1

    osm_buildings = load_osm_buildings(args.osm)
    plateau_buildings = load_plateau_index(args.index)
    print(f"[match_plateau] OSM 建物 {len(osm_buildings)} 件 / "
          f"PLATEAU 建物 {len(plateau_buildings)} 件")

    result = match(osm_buildings, plateau_buildings,
                   knn_radius_m=args.knn_radius_m,
                   iou_min=args.iou_min, grid_m=args.grid_m)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")),
                        encoding="utf-8")

    n_match = len(result["matches"])
    n_osm = len(osm_buildings)
    rate = (n_match / n_osm * 100.0) if n_osm else 0.0
    lo, med, hi = _iou_distribution(result["matches"])
    try:
        shown = args.out.relative_to(REPO_ROOT)
    except ValueError:
        shown = args.out
    print(f"[match_plateau] マッチ {n_match}/{n_osm} ({rate:.1f}%)  "
          f"未マッチ PLATEAU {result['unmatched_plateau_count']} 件")
    if lo is not None:
        print(f"[match_plateau] IoU 分布  min={lo:.3f}  median={med:.3f}  max={hi:.3f}")
    print(f"[match_plateau] 出力: {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
