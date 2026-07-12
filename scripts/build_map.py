"""OSM(Overpass API)から渋谷中心部の実道路網+全建物+POI+地下/デッキを取得し、地図を生成する。

使い方:
    python scripts/build_map.py                                   # → data/shibuya_osm.json(既定範囲・最新)
    python scripts/build_map.py --out data/shibuya_osm_wide.json \
        --bbox 35.65226 139.68956 35.66674 139.71168              # 拡張範囲(約2.0×1.6km)
    python scripts/build_map.py --osm-date 2023-04-01 --out data/shibuya_osm_2023.json
        # Overpass attic query で過去スナップショット(PLATEAU 基準日など)を取得

出典: OpenStreetMap contributors(ODbL)。ダウンロードは Overpass 公式 API のみ使用。

v6 の追加:
  - bbox / 取得日 / 出力先をコマンドラインで指定可能(既定=現行範囲・最新)
  - entrance=main/yes/exit ノード(建物外周上)を取り込み building.entrances に格納。
    main があれば building.entrance(最寄り道路ノード)を main 入口に最も近い道路ノードへ置換。
  - 地下判定を強化: highway=footway で layer<0 / level<0 / tunnel=(yes|building_passage) を
    地下(layer=-1)として取り込む(渋谷ちかみち等の地下歩行者網が routing にそのまま載る)。

v3 の追加:
  - 建物を全件取得(住宅・マンション含む)+用途分類 kind → 家の割当に使う
  - POI(飲食店・会社・店舗など)を実名で取得し、建物・最寄りノードに紐付け
  - 道路エッジに layer(-1=地下, 0=地上, 1=デッキ)を付与 → 渋谷ちかみち等
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# 既定範囲: 渋谷駅中心 約1.0km×0.7km(south, west, north, east)
DEFAULT_BBOX = (35.6560, 139.6950, 35.6625, 139.7060)
# スクランブル交差点 ≒ ローカル座標原点(bbox を変えても不変)
ORIGIN = (35.65950, 139.70062)

HIGHWAY_CLASSES = {
    "primary", "secondary", "tertiary", "unclassified", "residential",
    "living_street", "service", "pedestrian", "footway", "path", "steps",
    "cycleway", "corridor", "elevator",
}
# 車が走れる道 / 自転車が走れる道(徒歩は全部OK)
DRIVABLE = {"primary", "secondary", "tertiary", "unclassified", "residential",
            "living_street", "service"}
CYCLABLE = HIGHWAY_CLASSES - {"steps", "corridor", "elevator"}

# entrance タグの種別(建物外周ノード)。main を最優先で建物入口に採用する。
ENTRANCE_KINDS = {"main", "yes", "exit", "service", "staircase", "home", "emergency"}

# 実在ランドマーク(名前を最寄りノードに付与し、目的地候補=POI にする)
LANDMARKS = [
    ("スクランブル交差点", 35.65950, 139.70062, "square"),
    ("ハチ公前広場",       35.65905, 139.70046, "square"),
    ("渋谷駅ハチ公口",     35.65880, 139.70100, "station"),
    ("SHIBUYA109",         35.65946, 139.69838, "landmark"),
    ("渋谷センター街",     35.66030, 139.69820, "shopping"),
    ("道玄坂",             35.65800, 139.69600, "street"),
    ("文化村通り",         35.65990, 139.69590, "street"),
    ("スペイン坂",         35.66120, 139.69850, "street"),
    ("渋谷PARCO",          35.66210, 139.69880, "landmark"),
    ("公園通り",           35.66150, 139.69970, "street"),
    ("宮下公園",           35.66185, 139.70290, "park"),
    ("宮益坂下",           35.65920, 139.70330, "street"),
    ("渋谷ストリーム",     35.65710, 139.70260, "landmark"),
    ("渋谷マークシティ",   35.65850, 139.69800, "landmark"),
]

# 明示的に住宅系とわかる building タグ
RESIDENTIAL_TYPES = {"residential", "apartments", "house", "detached",
                     "dormitory", "terrace", "semidetached_house", "hut"}

# 待ち合わせ名所の名称マッチ(ハチ公・モヤイ・忠犬 等)。名前にこれを含む POI は
# landmark カテゴリに寄せる(OSM のタグが amenity/tourism 等でも待ち合わせ地点として扱う)。
LANDMARK_NAME_KWS = ("忠犬", "ハチ公", "モヤイ", "モアイ")

# ハチ公像の明示座標フォールバック(OSM で landmark として拾えなかった時に data へ置く)。
# 実在座標(渋谷駅ハチ公口前・忠犬ハチ公像)。cat=landmark で待ち合わせ候補に載る。
HACHIKO_FALLBACK = ("忠犬ハチ公像", 35.6590, 139.7005)


def build_query(bbox: tuple[float, float, float, float],
                osm_date: str | None = None) -> str:
    """Overpass QL を bbox から生成。osm_date 指定時は attic query(過去スナップショット)。"""
    date_setting = ""
    if osm_date:
        # "YYYY-MM-DD" → ISO8601 UTC。Overpass の [date:"…"] で当時の版を取得。
        date_setting = f'[date:"{osm_date}T00:00:00Z"]'
    header = f"[out:json][timeout:180]{date_setting};"
    return header + """
(
  way["highway"](%s,%s,%s,%s);
  way["building"](%s,%s,%s,%s);
  way["railway"~"^(rail|subway)$"](%s,%s,%s,%s);
  node["amenity"](%s,%s,%s,%s);
  node["shop"](%s,%s,%s,%s);
  node["office"](%s,%s,%s,%s);
  node["tourism"](%s,%s,%s,%s);
  node["leisure"](%s,%s,%s,%s);
  way["amenity"](%s,%s,%s,%s);
  way["shop"](%s,%s,%s,%s);
  way["office"](%s,%s,%s,%s);
  way["leisure"](%s,%s,%s,%s);
);
(._;>;);
out body;
""" % (bbox * 12)


def fetch_overpass(bbox: tuple[float, float, float, float],
                   osm_date: str | None = None, retries: int = 3) -> dict:
    """Overpass 公式 API から取得。混雑・タイムアウト時はバックオフ付きでリトライ。"""
    query = build_query(bbox, osm_date)
    body = ("data=" + urllib.request.quote(query)).encode("utf-8")
    endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    ]
    last_err: Exception | None = None
    for attempt in range(retries):
        endpoint = endpoints[attempt % len(endpoints)]
        try:
            req = urllib.request.Request(
                endpoint, data=body,
                headers={"Content-Type": "application/x-www-form-urlencoded",
                         "User-Agent": "shibuya-simulation research (contact: local)"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
            last_err = e
            wait = 15 * (attempt + 1)
            print(f"  取得失敗({type(e).__name__}: {e}) — {wait}s 待って再試行 "
                  f"[{attempt + 1}/{retries}]", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"Overpass 取得に失敗(全リトライ): {last_err}")


def project(lat: float, lon: float) -> tuple[float, float]:
    """緯度経度 → 原点基準ローカル平面(m)。1km 級なら十分な近似。"""
    x = (lon - ORIGIN[1]) * 111320.0 * math.cos(math.radians(ORIGIN[0]))
    y = (lat - ORIGIN[0]) * 110540.0
    return round(x, 1), round(y, 1)


def polyline_length(points: list[tuple[float, float]]) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1])
               for a, b in zip(points, points[1:]))


def _min_level(raw: str) -> int | None:
    """OSM の level タグ(例 "-1", "-2;-1", "0") から最下層を整数で返す。"""
    parts = []
    for p in str(raw).replace(";", ",").split(","):
        p = p.strip()
        if not p:
            continue
        try:
            parts.append(int(float(p)))
        except ValueError:
            continue
    return min(parts) if parts else None


def way_layer(tags: dict) -> int:
    """道路の垂直レイヤー: -1=地下(ちかみち等) / 0=地上 / 1=ペデストリアンデッキ。

    layer タグを最優先。無ければ level タグ(地下歩行者網は level=-1 等)を補助的に読む。
    tunnel=(yes|building_passage) は地下、bridge=yes はデッキ扱い。
    """
    layer: int | None = None
    raw_layer = tags.get("layer")
    if raw_layer is not None:
        try:
            layer = int(float(raw_layer))
        except ValueError:
            layer = None
    if layer is None:
        lvl = tags.get("level")
        if lvl is not None:
            layer = _min_level(lvl)
    if layer is None:
        layer = 0
    if tags.get("tunnel") in ("yes", "building_passage") and layer >= 0:
        layer = -1
    if tags.get("bridge") == "yes" and layer <= 0:
        layer = 1
    return max(-2, min(2, layer))


def point_in_poly(x: float, y: float, poly: list) -> bool:
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i][0], poly[i][1]
        xj, yj = poly[j][0], poly[j][1]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def poi_category(tags: dict) -> str | None:
    """OSM タグ → POI カテゴリ。v6 拡大(ユーザー要望 2026-07-06): 飲食だけでなく
    office(会社)/school(学校)/cinema(映画館)/hall(イベントホール・劇場)/
    landmark(待ち合わせ名所)まで対象を広げる。既存カテゴリ体系(food/nightlife/
    service/shop/office/hotel/attraction/leisure)に整合させて新値を追加する。
    ★旧地図(data/shibuya_osm.json)は再生成しない=旧 cat のまま。本関数の変更は
      新規に生成する地図(v6/wide)にのみ効く。"""
    name = tags.get("name:ja") or tags.get("name") or ""
    # 待ち合わせ名所(名称マッチ最優先: ハチ公/モヤイ/忠犬)。
    if any(kw in name for kw in LANDMARK_NAME_KWS):
        return "landmark"
    amenity = tags.get("amenity", "")
    if amenity in ("restaurant", "cafe", "fast_food", "food_court",
                   "ice_cream"):
        return "food"
    if amenity in ("bar", "pub", "nightclub", "karaoke_box", "izakaya"):
        return "nightlife"
    if amenity == "cinema":                              # 映画館(商業施設)
        return "cinema"
    if amenity in ("theatre", "events_venue", "conference_centre") \
            or tags.get("leisure") == "stadium":         # イベントホール・劇場・スタジアム
        return "hall"
    if amenity in ("bank", "pharmacy", "clinic", "dentist", "post_office",
                   "police", "library"):
        return "service"
    if amenity in ("school", "university", "college", "kindergarten") \
            or tags.get("building") in ("school", "university"):  # 学校・大学
        return "school"
    if "shop" in tags:
        return "shop"
    if "office" in tags:                                 # 会社・事務所
        return "office"
    if tags.get("tourism") in ("hotel", "hostel", "guest_house"):
        return "hotel"
    # 観光名所・彫像・記念碑は待ち合わせ名所(landmark)として扱う。
    if tags.get("tourism") in ("attraction", "artwork") \
            or tags.get("man_made") == "statue" \
            or tags.get("historic") in ("memorial", "monument"):
        return "landmark"
    if tags.get("tourism") in ("museum", "gallery"):
        return "attraction"
    if tags.get("leisure") in ("park", "garden", "playground", "pitch",
                               "fitness_centre", "sports_centre"):
        return "leisure"
    return None


def building_kind(tags: dict, area: float, name: str | None) -> str:
    btype = tags.get("building", "yes")
    use = tags.get("building:use", "")
    if btype in RESIDENTIAL_TYPES or use == "residential":
        return "residential"
    if btype in ("office", "commercial") or use == "office":
        return "office"
    if btype == "retail" or use == "retail":
        return "retail"
    if btype in ("train_station", "transportation", "station"):
        return "station"
    if btype in ("hotel",):
        return "hotel"
    if btype in ("school", "university", "college", "civic", "public",
                 "government", "hospital"):
        return "public"
    # 無名の小規模建物は住宅の可能性が高い(渋谷でも桜丘・神南の縁は住宅地)
    if name is None and area < 260:
        return "house?"
    return "generic"


def build(raw: dict, bbox: tuple[float, float, float, float],
          osm_date: str | None = None) -> dict:
    nodes_ll: dict[int, tuple[float, float]] = {}
    node_tags: dict[int, dict] = {}
    ways_road: list[dict] = []
    ways_building: list[dict] = []
    ways_rail: list[dict] = []
    ways_poi: list[dict] = []
    for el in raw["elements"]:
        if el["type"] == "node":
            nodes_ll[el["id"]] = (el["lat"], el["lon"])
            if el.get("tags"):
                node_tags[el["id"]] = el["tags"]
        elif el["type"] == "way":
            tags = el.get("tags", {})
            if tags.get("highway") in HIGHWAY_CLASSES:
                ways_road.append(el)
            if "building" in tags:
                ways_building.append(el)
            elif tags.get("railway") in ("rail", "subway"):
                ways_rail.append(el)
            if poi_category(tags):
                ways_poi.append(el)

    # --- 道路: 交差点でウェイを分割してエッジ化(layer 付き)---
    use_count: dict[int, int] = {}
    for way in ways_road:
        ids = [i for i in way["nodes"] if i in nodes_ll]
        for i in ids:
            use_count[i] = use_count.get(i, 0) + 1
        for i in (ids[0], ids[-1]):
            use_count[i] = use_count.get(i, 0) + 1  # 端点は交差点扱い

    edges: list[dict] = []
    for way in ways_road:
        tags = way["tags"]
        klass = tags["highway"]
        layer = way_layer(tags)
        ids = [i for i in way["nodes"] if i in nodes_ll]
        if len(ids) < 2:
            continue
        seg = [ids[0]]
        for node_id in ids[1:]:
            seg.append(node_id)
            if use_count.get(node_id, 0) >= 2 or node_id == ids[-1]:
                pts = [project(*nodes_ll[i]) for i in seg]
                length = polyline_length(pts)
                if length >= 1.0 and seg[0] != seg[-1]:
                    e = {"u": f"n{seg[0]}", "v": f"n{seg[-1]}",
                         "klass": klass, "geometry": pts,
                         "length": round(length, 1)}
                    if layer != 0:
                        e["layer"] = layer
                    edges.append(e)
                seg = [node_id]

    # --- 次数2ノードの縮約(地図を軽くする。ランドマーク付与前に実施)---
    from collections import defaultdict
    adj: dict[str, list[int]] = defaultdict(list)
    for idx, e in enumerate(edges):
        adj[e["u"]].append(idx)
        adj[e["v"]].append(idx)

    def other(e: dict, n: str) -> str:
        return e["v"] if e["u"] == n else e["u"]

    removed = set()
    changed = True
    while changed:
        changed = False
        for node, idxs in list(adj.items()):
            live = [i for i in idxs if i not in removed]
            if len(live) != 2:
                continue
            e1, e2 = edges[live[0]], edges[live[1]]
            a, b = other(e1, node), other(e2, node)
            if a == b or a == node or b == node:
                continue
            if e1["klass"] != e2["klass"]:
                continue
            if e1.get("layer", 0) != e2.get("layer", 0):
                continue
            g1 = e1["geometry"] if e1["v"] == node else list(reversed(e1["geometry"]))
            g2 = e2["geometry"] if e2["u"] == node else list(reversed(e2["geometry"]))
            merged = {"u": a, "v": b, "klass": e1["klass"],
                      "geometry": g1 + g2[1:],
                      "length": round(e1["length"] + e2["length"], 1)}
            if e1.get("layer", 0) != 0:
                merged["layer"] = e1["layer"]
            removed.update(live)
            edges.append(merged)
            new_idx = len(edges) - 1
            adj[a] = [i for i in adj[a] if i not in removed] + [new_idx]
            adj[b] = [i for i in adj[b] if i not in removed] + [new_idx]
            adj[node] = []
            changed = True

    edges = [e for i, e in enumerate(edges) if i not in removed]

    # --- 最大連結成分のみ残す(ちかみち・デッキは階段経由で地上と繋がる)---
    import networkx as nx
    g = nx.Graph()
    for i, e in enumerate(edges):
        g.add_edge(e["u"], e["v"], idx=i)
    if g.number_of_nodes() == 0:
        raise RuntimeError("道路が取得できていない")
    largest = max(nx.connected_components(g), key=len)
    edges = [e for e in edges if e["u"] in largest and e["v"] in largest]

    node_xy: dict[str, tuple[float, float]] = {}
    node_layer: dict[str, int] = {}
    for e in edges:
        node_xy[e["u"]] = e["geometry"][0]
        node_xy[e["v"]] = e["geometry"][-1]
        lyr = e.get("layer", 0)
        for n in (e["u"], e["v"]):
            # 複数レイヤーのエッジが接続するノードは接続点(地上扱い優先)
            node_layer[n] = 0 if node_layer.get(n, lyr) != lyr else lyr

    # --- ゲートウェイ(範囲の縁の行き止まり=外界への出口)---
    sw = project(bbox[0], bbox[1])
    ne = project(bbox[2], bbox[3])
    deg: dict[str, int] = defaultdict(int)
    for e in edges:
        deg[e["u"]] += 1
        deg[e["v"]] += 1
    gateways = []
    for node, d in deg.items():
        x, y = node_xy[node]
        near_border = (abs(x - sw[0]) < 40 or abs(x - ne[0]) < 40
                       or abs(y - sw[1]) < 40 or abs(y - ne[1]) < 40)
        if d == 1 and near_border:
            gateways.append(node)

    # 車用ゲートウェイ(背景交通の発生/消滅点): 縁に近い DRIVABLE の端点
    car_gateways = []
    for node, d in deg.items():
        x, y = node_xy[node]
        near_border = (abs(x - sw[0]) < 60 or abs(x - ne[0]) < 60
                       or abs(y - sw[1]) < 60 or abs(y - ne[1]) < 60)
        if not near_border:
            continue
        if any((e["u"] == node or e["v"] == node) and e["klass"] in DRIVABLE
               and e.get("layer", 0) == 0 for e in edges):
            car_gateways.append(node)

    # --- ランドマーク名を最寄りノードへ ---
    names: dict[str, tuple[str, str]] = {}
    for name, lat, lon, poi in LANDMARKS:
        lx, ly = project(lat, lon)
        best, best_d = None, 1e9
        for node, (x, y) in node_xy.items():
            d = math.hypot(x - lx, y - ly)
            if d < best_d:
                best, best_d = node, d
        if best is not None and best_d < 150:
            names[best] = (name, poi)

    def nearest_node(x: float, y: float, ground_only: bool = True) -> str:
        best, best_d = None, 1e9
        for node, (nx_, ny_) in node_xy.items():
            if ground_only and node_layer.get(node, 0) < 0:
                continue
            d = math.hypot(nx_ - x, ny_ - y)
            if d < best_d:
                best, best_d = node, d
        return best

    nodes_out = []
    for node, (x, y) in sorted(node_xy.items()):
        entry = {"id": node, "x": x, "y": y}
        if node_layer.get(node, 0) != 0:
            entry["layer"] = node_layer[node]
        if node in names:
            entry["name"], entry["poi"] = names[node]
        if node in gateways:
            entry["gateway"] = True
        nodes_out.append(entry)

    # --- 建物: 全件(住宅含む)+用途分類。家の割当・屋内活動に使う ---
    buildings = []
    n_main_override = 0
    n_with_entrances = 0
    for way in ways_building:
        tags = way.get("tags", {})
        ids = [i for i in way["nodes"] if i in nodes_ll]
        if len(ids) < 4:
            continue
        pts = [project(*nodes_ll[i]) for i in ids]
        area = abs(sum(pts[i][0] * pts[(i + 1) % len(pts)][1]
                       - pts[(i + 1) % len(pts)][0] * pts[i][1]
                       for i in range(len(pts)))) / 2
        if area < 25:
            continue
        name = tags.get("name:ja") or tags.get("name")
        levels = tags.get("building:levels")
        try:
            levels = max(1, int(float(levels))) if levels else (
                6 if area > 1500 else (4 if area > 400 else 2))
        except ValueError:
            levels = 3
        below = tags.get("building:levels:underground")
        try:
            below = max(0, int(float(below))) if below else 0
        except ValueError:
            below = 0
        kind = building_kind(tags, area, name)
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)

        # --- 入口(実データ): 建物外周ノードの entrance=* を収集 ---
        entrances: list[dict] = []
        for nid in way["nodes"]:
            etag = node_tags.get(nid, {}).get("entrance")
            if etag in ENTRANCE_KINDS and nid in nodes_ll:
                ex, ey = project(*nodes_ll[nid])
                entrances.append({"x": ex, "y": ey, "kind": etag})
        # main 入口があれば、そこに最も近い道路ノードを建物の入口に採用
        main = next((e for e in entrances if e["kind"] == "main"), None)
        if main is not None:
            entrance = nearest_node(main["x"], main["y"])
            n_main_override += 1
        else:
            entrance = nearest_node(cx, cy)

        b = {"id": f"b{way['id']}", "name": name or "",
             "levels": levels, "kind": kind, "footprint": pts,
             "entrance": entrance, "area": round(area),
             "cx": round(cx, 1), "cy": round(cy, 1)}
        if below:
            b["below"] = below
        if entrances:
            b["entrances"] = entrances
            n_with_entrances += 1
        buildings.append(b)
    buildings.sort(key=lambda b: -b["area"])

    # --- POI(実名の店・会社・施設)を建物と最寄りノードへ紐付け ---
    big_buildings = [b for b in buildings if b["area"] >= 100]
    pois = []
    seen_poi_names = set()

    def add_poi(tags: dict, x: float, y: float, src_id: str) -> None:
        cat = poi_category(tags)
        name = tags.get("name:ja") or tags.get("name")
        if not cat or not name:
            return
        if (name, cat) in seen_poi_names:
            return
        seen_poi_names.add((name, cat))
        host = None
        for b in big_buildings:
            if (abs(x - b["cx"]) < 120 and abs(y - b["cy"]) < 120
                    and point_in_poly(x, y, b["footprint"])):
                host = b
                break
        floor = 0
        lvl = tags.get("level")
        if lvl:
            try:
                floor = int(float(str(lvl).split(";")[0]))
            except ValueError:
                floor = 0
        p = {"id": f"p_{src_id}", "name": name, "cat": cat,
             "x": round(x, 1), "y": round(y, 1),
             "node": nearest_node(x, y)}
        if host:
            p["building"] = host["id"]
            p["floor"] = floor
        pois.append(p)

    for nid, tags in node_tags.items():
        if nid in nodes_ll:
            x, y = project(*nodes_ll[nid])
            add_poi(tags, x, y, f"n{nid}")
    for way in ways_poi:
        ids = [i for i in way["nodes"] if i in nodes_ll]
        if len(ids) < 3:
            continue
        pts = [project(*nodes_ll[i]) for i in ids]
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        add_poi(way.get("tags", {}), cx, cy, f"w{way['id']}")

    # --- 建物 kind に office/school を反映(POI 由来。用途不明の generic のみ格上げ)---
    #     住宅系(residential/house?)は家の割当に使うので触らない=既存の家割当は不変。
    pois_by_bld: dict[str, set[str]] = defaultdict(set)
    for p in pois:
        if p.get("building"):
            pois_by_bld[p["building"]].add(p["cat"])
    for b in buildings:
        if b["kind"] != "generic":
            continue
        cats = pois_by_bld.get(b["id"], set())
        if "school" in cats:
            b["kind"] = "public"
        elif "office" in cats:
            b["kind"] = "office"

    # --- ハチ公像フォールバック(OSM で landmark を拾えなかった場合に明示座標で置く)---
    has_hachiko = any(p["cat"] == "landmark"
                      and any(kw in p["name"] for kw in ("忠犬", "ハチ公"))
                      for p in pois)
    hachiko_source = "osm" if has_hachiko else "none"
    if not has_hachiko:
        hx, hy = project(HACHIKO_FALLBACK[1], HACHIKO_FALLBACK[2])
        hnode = nearest_node(hx, hy)
        if hnode is not None:
            pois.append({"id": "p_hachiko_fallback", "name": HACHIKO_FALLBACK[0],
                         "cat": "landmark", "x": round(hx, 1), "y": round(hy, 1),
                         "node": hnode})
            hachiko_source = "fallback"

    # --- 線路(可視化レイヤー用: 地上=rail / 地下鉄=subway)---
    railways = []
    for way in ways_rail:
        tags = way.get("tags", {})
        ids = [i for i in way["nodes"] if i in nodes_ll]
        if len(ids) < 2:
            continue
        pts = [project(*nodes_ll[i]) for i in ids]
        if polyline_length(pts) < 30:
            continue
        railways.append({"name": tags.get("name:ja") or tags.get("name") or "線路",
                         "kind": tags["railway"], "geometry": pts})

    meta = {
        "version": 6, "name": "shibuya_osm",
        "description": "OSM 実データ(Overpass)による渋谷駅中心部。"
                       "全建物+実入口+POI+地下/デッキ層。"
                       "座標=スクランブル交差点原点ローカル平面(m)。",
        "attribution": "© OpenStreetMap contributors (ODbL)",
        "origin_latlon": list(ORIGIN), "bbox": list(bbox),
        "crs": "local-m",
    }
    if osm_date:
        meta["osm_date"] = osm_date
    meta["_stats"] = {"main_entrance_override": n_main_override,
                      "buildings_with_entrances": n_with_entrances,
                      "hachiko_source": hachiko_source}
    return {
        "meta": meta,
        "nodes": nodes_out,
        "edges": edges,
        "buildings": buildings,
        "pois": pois,
        "railways": railways,
        "car_gateways": sorted(car_gateways),
    }


def _report(data: dict, out: Path) -> None:
    from collections import Counter
    n_res = sum(1 for b in data["buildings"]
                if b["kind"] in ("residential", "house?"))
    n_under = sum(1 for e in data["edges"] if e.get("layer", 0) < 0)
    n_deck = sum(1 for e in data["edges"] if e.get("layer", 0) > 0)
    n_under_nodes = sum(1 for n in data["nodes"] if n.get("layer", 0) < 0)
    n_ent = sum(1 for b in data["buildings"] if b.get("entrances"))
    stats = data["meta"].get("_stats", {})
    cats = Counter(p["cat"] for p in data["pois"])
    print(f"written: {out}")
    print(f"  nodes={len(data['nodes'])} edges={len(data['edges'])} "
          f"(地下 edge={n_under}/node={n_under_nodes} デッキ={n_deck}) "
          f"buildings={len(data['buildings'])} (住宅系={n_res}) "
          f"pois={len(data['pois'])} "
          f"gateways={sum(1 for n in data['nodes'] if n.get('gateway'))} "
          f"car_gateways={len(data['car_gateways'])}")
    print(f"  入口実データ: entrances付き建物={n_ent} "
          f"(main で入口置換={stats.get('main_entrance_override', 0)})")
    print(f"  POI 拡大(v6): office={cats.get('office', 0)} school={cats.get('school', 0)} "
          f"cinema={cats.get('cinema', 0)} hall={cats.get('hall', 0)} "
          f"landmark={cats.get('landmark', 0)}  ハチ公={stats.get('hachiko_source', '?')}")


def main() -> None:
    ap = argparse.ArgumentParser(description="渋谷 OSM 地図ビルダー(bbox/日付指定可)")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("S", "W", "N", "E"),
                    default=list(DEFAULT_BBOX),
                    help="south west north east(既定=現行範囲 約1.0×0.7km)")
    ap.add_argument("--osm-date", default=None,
                    help='過去スナップショット日 "YYYY-MM-DD"(Overpass attic query。既定=最新)')
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "shibuya_osm.json"),
                    help="出力先 JSON パス(既定=data/shibuya_osm.json)")
    args = ap.parse_args()

    bbox = tuple(args.bbox)
    print(f"Overpass API から取得中... bbox={bbox} date={args.osm_date or '最新'}",
          file=sys.stderr)
    raw = fetch_overpass(bbox, args.osm_date)
    print(f"  elements: {len(raw['elements'])}", file=sys.stderr)
    data = build(raw, bbox, args.osm_date)
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    _report(data, out)


if __name__ == "__main__":
    main()
