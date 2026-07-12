"""信号・車線・oneway・ゲートウェイのサイドカーデータを OSM から抽出する(交通 v2)。

コア地図 data/shibuya_osm.json はいじらない。ここで生成する
data/traffic_features_shibuya.json は、コア地図の車道エッジに
「車線数 / 一方通行 / 道路種別」を付与し、信号ノードと出入口(ゲートウェイ)を
足しただけの薄い注釈ファイル(サイドカー)。OD モードの背景交通が読む。

使い方:
    python scripts/build_traffic.py
        # → data/traffic_features_shibuya.json(コア地図と同じ既定範囲・最新 OSM)
    python scripts/build_traffic.py --map data/shibuya_osm_wide.json \
        --bbox 35.65226 139.68956 35.66674 139.71168 \
        --out data/traffic_features_wide.json
    python scripts/build_traffic.py --osm-date 2023-04-01
        # 過去スナップショット(build_map.py と同じ attic query)

出典: OpenStreetMap contributors(ODbL)。取得は build_map.py の Overpass 関数を再利用。

抽出:
  - highway=traffic_signals ノード → 位置 + 最寄り車道ノード id(コア地図のノード)
  - 車道 way の lanes / oneway / highway 種別 → コア地図の車道エッジへ座標一致で写像
  - ゲートウェイ = 車道が bbox 縁で 1 本だけ生えている端点(車の出入口)

写像は build_map.project(=決定論)で投影した座標を 1m 格子で突き合わせる。
コア地図と現行 OSM が同一スナップショットなら一致率は高い(統計を _report で出す)。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_build_map():
    """scripts/build_map.py を動的ロード(Overpass 取得・投影・定数を再利用)。"""
    spec = importlib.util.spec_from_file_location(
        "build_map", Path(__file__).with_name("build_map.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BM = _load_build_map()


# --------------------------------------------------------------------------- #
# タグの読み取り
# --------------------------------------------------------------------------- #
def _parse_lanes(raw) -> int | None:
    """lanes タグ(例 "2", "1", "2;1")を整数で。範囲は 1..8 に丸める。無ければ None。"""
    if raw is None:
        return None
    for tok in str(raw).replace(";", ",").split(","):
        tok = tok.strip()
        try:
            v = int(float(tok))
        except ValueError:
            continue
        if v >= 1:
            return min(v, 8)
    return None


def _oneway_dir(tags: dict) -> str | None:
    """一方通行の向き: "fwd"=way のノード順, "bwd"=逆向き, None=双方向。"""
    ow = str(tags.get("oneway", "")).lower()
    if ow in ("yes", "true", "1"):
        return "fwd"
    if ow in ("-1", "reverse"):
        return "bwd"
    if tags.get("junction") == "roundabout" and ow not in ("no", "false"):
        return "fwd"
    return None


def _cell(x: float, y: float) -> tuple[int, int]:
    """投影座標(m)を 1m 整数格子へ。OSM の微差・丸め差を吸収して突き合わせる。"""
    return (int(round(x)), int(round(y)))


# --------------------------------------------------------------------------- #
# 抽出本体(純関数=ネットワーク不要。テストが合成データで叩ける)
# --------------------------------------------------------------------------- #
def build_features(raw: dict, core: dict,
                   bbox: tuple[float, float, float, float],
                   osm_date: str | None = None) -> dict:
    """生 OSM(raw)とコア地図(core)から交通注釈サイドカーを作る。

    core = data/shibuya_osm.json 相当(nodes[{id,x,y}], edges[{u,v,klass,geometry,layer}],
    car_gateways[])。edges のキーはコア地図のノード id をそのまま使う。
    """
    project = BM.project
    drivable = BM.DRIVABLE

    # --- 生 OSM のノード緯度経度 ---
    nodes_ll: dict[int, tuple[float, float]] = {}
    node_tags: dict[int, dict] = {}
    ways: list[dict] = []
    for el in raw["elements"]:
        if el["type"] == "node":
            nodes_ll[el["id"]] = (el["lat"], el["lon"])
            if el.get("tags"):
                node_tags[el["id"]] = el["tags"]
        elif el["type"] == "way":
            ways.append(el)

    # --- 車道 way のセグメントを格子に索引(車線・通行方向)---
    allow: set[tuple[tuple[int, int], tuple[int, int]]] = set()  # 通行可能な有向セル対
    seg_lanes: dict[frozenset, int] = {}                         # 無向セル対 → 車線数
    seg_known: set[frozenset] = set()                            # 車道セグメントのセル対
    for way in ways:
        tags = way.get("tags", {})
        if tags.get("highway") not in drivable:
            continue
        ids = [n for n in way["nodes"] if n in nodes_ll]
        if len(ids) < 2:
            continue
        cells = [_cell(*project(*nodes_ll[n])) for n in ids]
        lanes = _parse_lanes(tags.get("lanes"))
        owdir = _oneway_dir(tags)
        for a, b in zip(cells, cells[1:]):
            if a == b:
                continue
            key = frozenset((a, b))
            seg_known.add(key)
            if lanes is not None:
                seg_lanes[key] = max(seg_lanes.get(key, 0), lanes)
            if owdir is None:
                allow.add((a, b))
                allow.add((b, a))
            elif owdir == "fwd":
                allow.add((a, b))
            else:                                   # "bwd"
                allow.add((b, a))

    # --- コア地図の車道地上エッジへ写像(u|v キーはコア地図ノード id)---
    node_xy: dict[str, tuple[float, float]] = {
        n["id"]: (float(n["x"]), float(n["y"])) for n in core["nodes"]}
    edges_out: dict[str, dict] = {}
    driv_deg: dict[str, int] = defaultdict(int)
    driv_nodes: set[str] = set()
    n_matched = 0
    n_lane = 0
    for e in core["edges"]:
        if e.get("klass") not in drivable or int(e.get("layer", 0)) != 0:
            continue
        u, v = e["u"], e["v"]
        driv_deg[u] += 1
        driv_deg[v] += 1
        driv_nodes.add(u)
        driv_nodes.add(v)
        geom = e.get("geometry") or [list(node_xy[u]), list(node_xy[v])]
        cells = [_cell(p[0], p[1]) for p in geom]
        matched = False
        lane_vals: list[int] = []
        fwd = bwd = True
        for a, b in zip(cells, cells[1:]):
            if a == b:
                continue
            key = frozenset((a, b))
            if key in seg_known:
                matched = True
                if key in seg_lanes:
                    lane_vals.append(seg_lanes[key])
                fwd = fwd and ((a, b) in allow)     # geometry は u→v 順
                bwd = bwd and ((b, a) in allow)
        lanes = max(lane_vals) if lane_vals else None
        if matched and fwd and not bwd:
            oneway = 1                              # u→v のみ
        elif matched and bwd and not fwd:
            oneway = -1                             # v→u のみ
        else:
            oneway = 0                              # 双方向(未一致もここ)
        if matched:
            n_matched += 1
        if lanes is not None:
            n_lane += 1
        edges_out[f"{u}|{v}"] = {"lanes": lanes, "oneway": oneway,
                                 "class": e["klass"]}

    # --- 信号ノード → 最寄り車道ノード(コア地図) ---
    driv_xy = {n: node_xy[n] for n in driv_nodes}
    signals_by_node: dict[str, dict] = {}
    for nid, tags in node_tags.items():
        if tags.get("highway") != "traffic_signals" or nid not in nodes_ll:
            continue
        sx, sy = project(*nodes_ll[nid])
        best, best_d = None, 1e18
        for node, (nx_, ny_) in driv_xy.items():
            d = math.hypot(nx_ - sx, ny_ - sy)
            if d < best_d:
                best, best_d = node, d
        if best is not None and best_d < 60.0 and best not in signals_by_node:
            signals_by_node[best] = {"node": best, "x": round(sx, 1),
                                     "y": round(sy, 1)}
    signals = [signals_by_node[n] for n in sorted(signals_by_node)]

    # --- ゲートウェイ = 縁に近い車道の 1 本刺しの端点(車の出入口)---
    sw = project(bbox[0], bbox[1])
    ne = project(bbox[2], bbox[3])

    def _near_border(x: float, y: float) -> bool:
        return (abs(x - sw[0]) < 60 or abs(x - ne[0]) < 60
                or abs(y - sw[1]) < 60 or abs(y - ne[1]) < 60)

    gateways = sorted(n for n in driv_nodes
                      if driv_deg[n] == 1 and _near_border(*driv_xy[n]))
    if len(gateways) < 2:                           # 予備: コア地図の車用出入口
        gateways = sorted(str(n) for n in core.get("car_gateways", [])
                          if n in node_xy)

    lane_rate = round(n_lane / len(edges_out), 3) if edges_out else 0.0
    meta = {
        "name": "traffic_features_shibuya",
        "source": "© OpenStreetMap contributors (ODbL) via Overpass",
        "core_map": core.get("meta", {}).get("name", ""),
        "bbox": list(bbox),
        "n_signals": len(signals),
        "n_edges": len(edges_out),
        "n_edges_matched": n_matched,
        "lane_data_rate": lane_rate,
        "n_gateways": len(gateways),
    }
    if osm_date:
        meta["osm_date"] = osm_date
    return {"meta": meta, "signals": signals, "edges": edges_out,
            "gateways": gateways}


def _report(feat: dict, out: Path) -> None:
    m = feat["meta"]
    print(f"written: {out}")
    print(f"  signals={m['n_signals']} edges={m['n_edges']} "
          f"(車道一致={m['n_edges_matched']}) "
          f"車線データ率={m['lane_data_rate']:.1%} gateways={m['n_gateways']}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="渋谷 交通注釈ビルダー(信号・車線・oneway・ゲートウェイ)")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("S", "W", "N", "E"),
                    default=None,
                    help="south west north east(既定=コア地図 meta の bbox)")
    ap.add_argument("--osm-date", default=None,
                    help='過去スナップショット日 "YYYY-MM-DD"(build_map と同じ attic query)')
    ap.add_argument("--map", default=str(REPO_ROOT / "data" / "shibuya_osm.json"),
                    help="コア地図 JSON(車道エッジ・ノードの参照元。既定=data/shibuya_osm.json)")
    ap.add_argument("--out",
                    default=str(REPO_ROOT / "data" / "traffic_features_shibuya.json"),
                    help="出力先(既定=data/traffic_features_shibuya.json)")
    args = ap.parse_args()

    map_path = Path(args.map)
    if not map_path.is_absolute():
        map_path = REPO_ROOT / map_path
    core = json.loads(map_path.read_text(encoding="utf-8"))
    bbox = tuple(args.bbox) if args.bbox else tuple(core["meta"]["bbox"])

    print(f"Overpass API から取得中... bbox={bbox} date={args.osm_date or '最新'}")
    raw = BM.fetch_overpass(bbox, args.osm_date)
    feat = build_features(raw, core, bbox, args.osm_date)

    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.write_text(json.dumps(feat, ensure_ascii=False), encoding="utf-8")
    _report(feat, out)


if __name__ == "__main__":
    main()
