#!/usr/bin/env python3
"""実バスダイヤの静的表ビルド v-Ride-2(GTFS zip/dir → コンパクト静的表 JSON)。

正典: docs/research/sumo-live-transit.md §3.3(SUMO を回さない軽量版=静的表推奨)。
方式(決定論の保護): **オフライン取得 → data/odpt/ に静的キャッシュ → シミュはキャッシュのみ読む**
  (odpt-integration.md の大原則)。本スクリプトは事前に1回走らせて静的表を作る道具。実行中の
  ネット呼び出しは一切しない。pandas 不使用(純 Python csv + json)=既存 build_transit_odpt.py と同流儀。

入力: バス GTFS(zip または展開済みディレクトリ)。stops.txt / routes.txt / trips.txt /
      stop_times.txt /(任意)calendar.txt を読む。
絞り込み: 地図 bbox(--map の meta.bbox または --bbox S W N E)で渋谷付近の停留所に限定する。
投影: 停留所 経緯度 → 地図原点基準ローカル m(--map の meta.origin_latlon または --origin lat lon)。
      シミュの city.node_xy と同一座標系に載せる(build_map.project と同一式)。

出力(data/odpt/bus_table_shibuya.json):
  {"_meta": {...出典・origin_latlon・bbox・calendar・件数...},
   "stops":  [{"id","name","lat","lon","x","y"}, ...],
   "routes": [{"name","direction","stops":[stop_id...],"cum_sec":[0,...],"departures":[分,...]}, ...]}
  routes[k].cum_sec[i] = 系統の先頭停からの累積所要秒(便横断の中央値)。
  routes[k].departures = 平日・先頭停の発時刻(分・0..1440+。深夜便は 24:MM=1440+)を昇順。

実データが無い場合: tests/fixtures/bus_gtfs_synth(合成 GTFS・中立名)でビルド経路を検証できる
  (実バス停名は含めない=基盤の地名禁止ガードと無関係だが、合成表は中立名で作る)。

使い方:
  python scripts/build_bus_table.py --gtfs data/odpt/gtfs/Toei-Bus-GTFS.zip --map data/shibuya_osm.json
  python scripts/build_bus_table.py --gtfs tests/fixtures/bus_gtfs_synth \
      --origin 35.6595 139.70062 --bbox 35.656 139.695 35.6625 139.706 \
      --out /tmp/bus_table.json

出典: 公共交通オープンデータセンター(ODPT)https://www.odpt.org/(利用規約の出典表示に従う)。
"""
from __future__ import annotations

import argparse
import csv
import datetime
import io
import json
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_LON_M = 111320.0           # build_map.project / sumo_pipeline.project と同一の局所接平面定数
_LAT_M = 110540.0

SOURCE_META = {
    "source": "公共交通オープンデータセンター (Public Transportation Open Data Center, ODPT)",
    "source_url": "https://www.odpt.org/",
    "attribution": ("本データは公共交通オープンデータセンターのデータを利用して作成。"
                    "出典表示および ODPT 利用規約の遵守が必要。"),
    "disclaimer": "データの正確性・完全性は保証されない。ダイヤ乱れは扱わない(定刻近似)。",
    "usage_note": ("決定論保護のため、シミュ実行中はこの静的表のみ読む(実行時の取得は禁止)。"
                   "詳細: docs/research/sumo-live-transit.md §3 / odpt-integration.md"),
}


# ------------------------------------------------------------------ GTFS 読取
def _open_reader(src: Path):
    """GTFS zip / ディレクトリの両対応。rows(name)->DictReader を返すクロージャと close を返す。"""
    if src.is_dir():
        def rows(name):
            p = src / name
            if not p.is_file():
                return []
            return list(csv.DictReader(p.open(encoding="utf-8-sig")))
        return rows, (lambda: None)
    z = zipfile.ZipFile(src)
    names = set(z.namelist())

    def rows(name):
        if name not in names:
            return []
        return list(csv.DictReader(io.TextIOWrapper(z.open(name), encoding="utf-8-sig")))
    return rows, z.close


def _to_sec(text: str) -> int | None:
    """GTFS 時刻 "HH:MM:SS"(24 超可)→ 秒。空/不正は None。"""
    if not text:
        return None
    parts = text.strip().split(":")
    try:
        h, m = int(parts[0]), int(parts[1])
        s = int(parts[2]) if len(parts) > 2 else 0
    except (ValueError, IndexError):
        return None
    return h * 3600 + m * 60 + s


def _median_int(vals: list[int]) -> int:
    """整数中央値(決定論。偶数個は下側中央=昇順の中央要素)。"""
    v = sorted(vals)
    return v[len(v) // 2]


def _project(lat: float, lon: float, origin) -> tuple[float, float]:
    import math
    o_lat, o_lon = origin
    cos_lat = math.cos(math.radians(o_lat))
    return round((lon - o_lon) * _LON_M * cos_lat, 1), round((lat - o_lat) * _LAT_M, 1)


# ------------------------------------------------------------------ 表ビルド
def build_table(gtfs: Path, origin, bbox, station_label: str) -> dict:
    """GTFS(zip/dir)→ 静的表 dict。純関数・決定論(同入力なら generated_at 以外バイト一致)。"""
    rows, close = _open_reader(gtfs)
    try:
        s_lat, s_lon, n_lat, n_lon = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])

        # --- 停留所(bbox 内)---
        stops_in: dict[str, dict] = {}
        for s in rows("stops.txt"):
            try:
                lat, lon = float(s["stop_lat"]), float(s["stop_lon"])
            except (KeyError, ValueError):
                continue
            if not (s_lat <= lat <= n_lat and s_lon <= lon <= n_lon):
                continue
            sid = str(s["stop_id"])
            x, y = _project(lat, lon, origin)
            stops_in[sid] = {"id": sid, "name": str(s.get("stop_name", sid)),
                             "lat": round(lat, 6), "lon": round(lon, 6), "x": x, "y": y}

        # --- 平日 service(calendar.txt があれば 月〜金=1 かつ 土日=0。無ければ全 service)---
        cal = rows("calendar.txt")
        if cal:
            weekday_services = {c["service_id"] for c in cal
                                if c.get("monday") == "1" and c.get("friday") == "1"
                                and c.get("saturday") == "0" and c.get("sunday") == "0"}
        else:
            weekday_services = None      # None=全 trip を採用(合成 GTFS 等 calendar 無し)

        # --- routes / trips ---
        routes = {r["route_id"]: (r.get("route_short_name") or r.get("route_long_name")
                                  or r["route_id"]) for r in rows("routes.txt")}
        trips: dict[str, tuple] = {}     # trip_id -> (route_id, direction_id)
        for t in rows("trips.txt"):
            if weekday_services is not None and t.get("service_id") not in weekday_services:
                continue
            trips[str(t["trip_id"])] = (str(t["route_id"]), str(t.get("direction_id", "")))

        # --- stop_times を trip 別に集約(bbox 停のみ・時刻順)---
        by_trip: dict[str, list[tuple[int, str, int]]] = {}   # trip -> [(seq, stop_id, dep_sec)]
        for st in rows("stop_times.txt"):
            tid = str(st.get("trip_id", ""))
            if tid not in trips:
                continue
            sid = str(st.get("stop_id", ""))
            if sid not in stops_in:
                continue
            dep = _to_sec(st.get("departure_time") or st.get("arrival_time") or "")
            if dep is None:
                continue
            try:
                seq = int(st.get("stop_sequence", 0))
            except ValueError:
                seq = 0
            by_trip.setdefault(tid, []).append((seq, sid, dep))

        # --- (route, direction) 別に代表停留所列 + cum_sec + departures を再構成 ---
        # 停留所の代表 index = 便横断の stop_sequence 中央値。基準停 = その最小。
        groups: dict[tuple, list[list[tuple[int, str, int]]]] = {}
        for tid, seq_rows in by_trip.items():
            seq_rows.sort()                       # (seq, stop, dep) 昇順
            groups.setdefault(trips[tid], []).append(seq_rows)

        out_routes: list[dict] = []
        used_stops: set[str] = set()
        for (route_id, direction), trip_rows_list in sorted(groups.items()):
            # 停留所ごとの seq 位置(中央値)を集める
            seq_of: dict[str, list[int]] = {}
            for seq_rows in trip_rows_list:
                for seq, sid, _dep in seq_rows:
                    seq_of.setdefault(sid, []).append(seq)
            if len(seq_of) < 2:
                continue
            ordered = sorted(seq_of.keys(),
                             key=lambda sid: (_median_int(seq_of[sid]), sid))
            ref = ordered[0]
            # 各停留所の基準停からの所要秒(中央値)。ref を含む便でのみ差分を採る。
            delta_sec: dict[str, int] = {}
            for sid in ordered:
                diffs = []
                for seq_rows in trip_rows_list:
                    dmap = {s: d for _q, s, d in seq_rows}
                    if ref in dmap and sid in dmap and dmap[sid] >= dmap[ref]:
                        diffs.append(dmap[sid] - dmap[ref])
                if diffs:
                    delta_sec[sid] = _median_int(diffs)
            # cum_sec 昇順で系統の停留所列を確定(単調・順方向)
            seq_stops = sorted(delta_sec.keys(), key=lambda sid: (delta_sec[sid], sid))
            if len(seq_stops) < 2:
                continue
            cum = [float(delta_sec[sid]) for sid in seq_stops]
            # departures = 平日・基準停の発時刻(分)。ref を含む便の dep を分へ。
            deps_min = set()
            for seq_rows in trip_rows_list:
                for _q, sid, dep in seq_rows:
                    if sid == ref:
                        deps_min.add(dep // 60)
            if not deps_min:
                continue
            out_routes.append({
                "name": str(routes.get(route_id, route_id)),
                "direction": str(direction),
                "stops": seq_stops, "cum_sec": cum,
                "departures": sorted(deps_min)})
            used_stops.update(seq_stops)

        # 系統名・方向で決定論ソート
        out_routes.sort(key=lambda r: (r["name"], r["direction"]))
        stops_out = [stops_in[sid] for sid in sorted(used_stops)]

        meta = dict(SOURCE_META)
        meta.update({
            "generated_by": "scripts/build_bus_table.py",
            "generated_at": datetime.date.today().isoformat(),
            "station_label": station_label,
            "origin_latlon": [round(float(origin[0]), 6), round(float(origin[1]), 6)],
            "bbox": [s_lat, s_lon, n_lat, n_lon],
            "step_seconds": 600,
            "calendar": "weekday" if weekday_services is not None else "all",
            "n_stops": len(stops_out), "n_routes": len(out_routes)})
        return {"_meta": meta, "stops": stops_out, "routes": out_routes}
    finally:
        close()


def _load_map_geo(map_path: Path):
    meta = json.loads(map_path.read_text(encoding="utf-8")).get("meta", {})
    return meta.get("origin_latlon"), meta.get("bbox"), meta.get("name", map_path.stem)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="バス GTFS(zip/dir)→ 実ダイヤ静的表 JSON(渋谷付近・平日・決定論)")
    ap.add_argument("--gtfs", required=True, help="バス GTFS の zip または展開済みディレクトリ")
    ap.add_argument("--map", default=None,
                    help="地図 JSON(meta.origin_latlon / meta.bbox を使う。既定=data/shibuya_osm.json)")
    ap.add_argument("--origin", nargs=2, type=float, default=None,
                    metavar=("LAT", "LON"), help="投影原点(--map より優先)")
    ap.add_argument("--bbox", nargs=4, type=float, default=None,
                    metavar=("S", "W", "N", "E"), help="絞り込み bbox(--map より優先)")
    ap.add_argument("--station-label", default="渋谷付近", help="メタ表示用ラベル")
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "odpt" / "bus_table_shibuya.json"),
                    help="出力先 JSON(既定: data/odpt/bus_table_shibuya.json)")
    args = ap.parse_args(argv)
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")   # type: ignore[attr-defined]
        except Exception:  # pragma: no cover
            pass

    gtfs = Path(args.gtfs)
    if not gtfs.is_absolute():
        gtfs = REPO_ROOT / gtfs
    if not gtfs.exists():
        print(f"GTFS が見つからない: {gtfs}", file=sys.stderr)
        return 1

    origin, bbox = args.origin, args.bbox
    if origin is None or bbox is None:
        map_path = Path(args.map) if args.map else (REPO_ROOT / "data" / "shibuya_osm.json")
        if not map_path.is_absolute():
            map_path = REPO_ROOT / map_path
        if not map_path.is_file():
            print(f"--origin/--bbox 未指定かつ地図が無い: {map_path}", file=sys.stderr)
            return 1
        m_origin, m_bbox, _name = _load_map_geo(map_path)
        origin = origin or m_origin
        bbox = bbox or m_bbox
    if not origin or not bbox:
        print("origin / bbox を解決できない(--map か --origin/--bbox を指定)", file=sys.stderr)
        return 1

    table = build_table(gtfs, tuple(origin), tuple(bbox), args.station_label)
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {out_path}: 停留所 {table['_meta']['n_stops']} / 系統 {table['_meta']['n_routes']}"
          f"(bbox={bbox} origin={origin} calendar={table['_meta']['calendar']})")
    print("出典: 公共交通オープンデータセンター(ODPT)https://www.odpt.org/ — 利用規約に従い出典表示すること。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
