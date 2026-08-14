"""ODPT 実ダイヤ → シミュ transit ファイル変換(第10バッチ 2026-07-08)。

station_timetable_*.json(scripts/fetch_odpt.py の取得キャッシュ)から、渋谷駅の
**実発車時刻**を持つ路線の 始発・終電・運転間隔 を再構成し、既存の近似ダイヤ
data/transit_shibuya.json に重ねて data/transit_odpt.json を生成する。駅時刻表が
取れない路線は既存の近似(公表の始発終電・間隔)を維持し、source で実/近似を区別する。

キャッシュは data/odpt_challenge(チャレンジ参加者キーで取得した限定データ)を優先し、
無ければ data/odpt(オープン枠=メトロ3路線のみ駅時刻表あり)へフォールバックする。

方式はプロジェクトの決定論原則に従う「オフライン変換→静的ファイル→シミュはファイルのみ
読む」(docs/research/odpt-integration.md)。実行中の API 呼び出しはしない。

使い方:
    python scripts/fetch_odpt.py            # 先に取得(要 ODPT_API_KEY)
    python scripts/build_transit_odpt.py    # → data/transit_odpt.json
    (本番は conf/production.yaml の transit.file が生成物を指す)

出典: 公共交通オープンデータセンター(ODPT)https://www.odpt.org/
      本データは ODPT の提供データを加工して作成(利用規約に基づく出典表示)。
"""
from __future__ import annotations

import csv
import datetime
import io
import json
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ODPT_DIRS = [REPO_ROOT / "data" / "odpt_challenge",   # チャレンジ限定データ(あれば優先)
             REPO_ROOT / "data" / "odpt"]             # オープン枠
BASE = REPO_ROOT / "data" / "transit_shibuya.json"
OUT = REPO_ROOT / "data" / "transit_odpt.json"

# ODPT 路線キー → 基底ダイヤの路線名(部分一致)。駅時刻表が取れた路線だけ実ダイヤ化する
# (キャッシュに平日時刻表が無い路線は自動スキップ=オープン枠ではメトロ3路線だけ残る)。
_KEY_TO_NAME = {
    "metro_ginza": "銀座線",
    "metro_hanzomon": "半蔵門線",
    "metro_fukutoshin": "副都心線",
    "jr_saikyo": "埼京線",
    "tokyu_toyoko": "東横線",
    "tokyu_denentoshi": "田園都市線",
    "keio_inokashira": "井の頭線",
}

# 山手線は基底で 内回り/外回り が別路線 → odpt:railDirection で方向別に重ねる。
_DIRECTIONAL = {
    "jr_yamanote": [("InnerLoop", "内回り"), ("OuterLoop", "外回り")],
}

# 静的 GTFS(zip)のフォールバック(fetch_gtfs_odpt.py の取得キャッシュ)。
# StationTimetable が無い路線だけ GTFS から実ダイヤ化する(JR東の投入待ち受け皿)。
GTFS_DIRS = [(d / "gtfs", tier) for d, tier in
             zip(ODPT_DIRS, ("チャレンジ限定", "オープン枠"))]


def _to_min(text: str) -> int:
    parts = text.split(":")                       # "HH:MM"(駅時刻表) / "HH:MM:SS"(GTFS)
    return int(parts[0]) * 60 + int(parts[1])


def _fmt(minutes: int) -> str:
    """分 → "HH:MM"。1440 以上は 24:MM 表記(翌日=終電の慣例。Transit._to_min が解釈)。"""
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def weekday_departures(path: Path, direction_part: str | None = None) -> list[int] | None:
    """駅時刻表ファイルから平日の発車時刻(分。深夜 0-2 時台は +1440)を昇順で返す。

    direction_part 指定時は odpt:railDirection がそれを含むレコードに絞る(山手線の
    内回り/外回り用)。未指定は従来どおり本数の多い方面を代表にする。"""
    raw = json.loads(path.read_text(encoding="utf-8"))
    recs = raw.get("data", raw if isinstance(raw, list) else [])
    best: list[int] | None = None
    for r in recs:
        if "Weekday" not in str(r.get("odpt:calendar", "")):
            continue
        if direction_part and direction_part not in str(r.get("odpt:railDirection", "")):
            continue
        times = []
        for o in r.get("odpt:stationTimetableObject", []):
            t = o.get("odpt:departureTime")
            if not t:
                continue
            m = _to_min(t)
            if m < 3 * 60:                     # 0:00-2:59 発は前日ダイヤの深夜帯
                m += 1440
            times.append(m)
        times.sort()
        # 方面が複数あるなら本数の多い方面を代表にする(間隔が方面混在で半減しないように)
        if times and (best is None or len(times) > len(best)):
            best = times
    return best


def gtfs_shibuya_departures(zip_path: Path,
                            station_filters=("渋谷", "Shibuya")) -> dict[str, list[dict]]:
    """静的 GTFS zip → {路線名: [{direction, headsign, times(平日・対象駅発・分・昇順)}]}。

    平日 = calendar.txt で 月〜金=1 かつ 土日=0 の service。深夜 0-2 時台は +1440
    (GTFS は本来 24:MM 表記だが、念のため駅時刻表と同じ繰上げを適用)。
    station_filters: 対象駅の stop_name 部分一致リスト(既定=渋谷/Shibuya。街ごとに差替可)。"""
    z = zipfile.ZipFile(zip_path)

    def rows(name):
        return csv.DictReader(io.TextIOWrapper(z.open(name), encoding="utf-8-sig"))

    weekday_services = {c["service_id"] for c in rows("calendar.txt")
                        if c.get("monday") == "1" and c.get("friday") == "1"
                        and c.get("saturday") == "0" and c.get("sunday") == "0"}
    shibuya_ids = {s["stop_id"] for s in rows("stops.txt")
                   if any(f in s.get("stop_name", "") for f in station_filters)}
    routes = {r["route_id"]: (r.get("route_long_name") or r.get("route_short_name")
                              or r["route_id"]) for r in rows("routes.txt")}
    trips = {}                                     # trip_id -> (route_id, direction, headsign)
    for t in rows("trips.txt"):
        if t.get("service_id") in weekday_services:
            trips[t["trip_id"]] = (t["route_id"], t.get("direction_id", ""),
                                   t.get("trip_headsign", ""))
    deps: dict[tuple, list[int]] = {}              # (route, direction) -> times
    heads: dict[tuple, str] = {}
    for st in rows("stop_times.txt"):
        if st["stop_id"] not in shibuya_ids or not st.get("departure_time"):
            continue
        trip = trips.get(st["trip_id"])
        if trip is None:
            continue
        route_id, direction, headsign = trip
        m = _to_min(st["departure_time"])
        if m < 3 * 60:
            m += 1440
        key = (routes.get(route_id, route_id), direction)
        deps.setdefault(key, []).append(m)
        heads.setdefault(key, headsign)
    out: dict[str, list[dict]] = {}
    for (route, direction), times in deps.items():
        out.setdefault(route, []).append(
            {"direction": direction, "headsign": heads[(route, direction)],
             "times": sorted(times)})
    for groups in out.values():                     # 本数の多い方向を先頭に
        groups.sort(key=lambda g: -len(g["times"]))
    return out


def line_stats(times: list[int]) -> dict:
    """発車時刻列 → 始発/終電/日中(7-22時)の中央値間隔。"""
    day = [t for t in times if 7 * 60 <= t <= 22 * 60]
    gaps = sorted(b - a for a, b in zip(day, day[1:]) if 0 < b - a < 60) or \
        sorted(b - a for a, b in zip(times, times[1:]) if 0 < b - a < 60)
    headway = gaps[len(gaps) // 2] if gaps else 5
    return {"first": _fmt(times[0]), "last": _fmt(times[-1]),
            "headway_min": int(headway), "n_departures": len(times)}


def _cache_path(key: str) -> tuple[Path, str] | None:
    """優先順にキャッシュを探す。データ 0 件のファイル(オープン枠の JR 等)は次の候補へ。"""
    for d, tier in zip(ODPT_DIRS, ("チャレンジ限定", "オープン枠")):
        p = d / f"station_timetable_{key}.json"
        if not p.exists():
            continue
        try:
            if json.loads(p.read_text(encoding="utf-8")).get("data"):
                return p, tier
        except (json.JSONDecodeError, OSError):
            continue
    return None


def _load_keymap(path):
    """路線識別子表 JSON を読み {"key_to_name": {...}, "directional": {...}} を返す。

    形式: {"key_to_name": {"metro_ginza": "銀座線", ...},
           "directional": {"jr_yamanote": [["InnerLoop","内回り"], ...]}}。
    未指定なら組込み既定(現行9路線)。"""
    if not path:
        return dict(_KEY_TO_NAME), {k: list(v) for k, v in _DIRECTIONAL.items()}
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    k2n = dict(doc.get("key_to_name", {}))
    direc = {k: [tuple(pair) for pair in v]
             for k, v in (doc.get("directional", {}) or {}).items()}
    if not k2n and not direc:
        raise ValueError(f"keymap ファイルが空: {path}")
    return k2n, direc


def main(argv: list[str] | None = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(
        description="ODPT/GTFS キャッシュ → transit ファイル変換(駅名・路線識別子 指定可・既定=渋谷9路線)")
    ap.add_argument("--base", default=str(BASE),
                    help="基底ダイヤ JSON(既定=data/transit_shibuya.json)")
    ap.add_argument("--out", default=str(OUT),
                    help="出力先 JSON(既定=data/transit_odpt.json)")
    ap.add_argument("--station", nargs="+", default=["渋谷", "Shibuya"],
                    help="GTFS 停車駅の stop_name 部分一致(既定=渋谷 Shibuya)")
    ap.add_argument("--station-label", default="渋谷駅",
                    help="メタ・source 表示用の駅ラベル(既定=渋谷駅)")
    ap.add_argument("--keymap-file", default=None,
                    help="路線識別子表 JSON(key_to_name/directional)。既定=現行9路線")
    args = ap.parse_args(argv)

    base_path = Path(args.base) if Path(args.base).is_absolute() else REPO_ROOT / args.base
    out_path = Path(args.out) if Path(args.out).is_absolute() else REPO_ROOT / args.out
    station_filters = tuple(args.station)
    station_label = args.station_label
    key_to_name, directional = _load_keymap(args.keymap_file)

    base = json.loads(base_path.read_text(encoding="utf-8"))
    lines = [dict(line) for line in base["lines"]]
    n_real = 0
    # (キャッシュキー, 基底の路線名の部分一致, 方向フィルタ|None)
    jobs = [(k, n, None) for k, n in sorted(key_to_name.items())] + \
           [(k, n, d) for k, pairs in sorted(directional.items()) for d, n in pairs]
    for key, name_part, direction in jobs:
        label = f"{key}({direction})" if direction else key
        found = _cache_path(key)
        if found is None:
            print(f"  - {label}: 駅時刻表キャッシュなし(スキップ。fetch_odpt.py を先に)")
            continue
        path, tier = found
        times = weekday_departures(path, direction)
        if not times:
            print(f"  - {label}: 平日時刻表なし(スキップ)")
            continue
        stats = line_stats(times)
        hit = next((ln for ln in lines if name_part in ln["name"]), None)
        if hit is None:
            print(f"  - {label}: 基底に該当路線名なし({name_part})")
            continue
        old = (hit["first"], hit["last"], hit["headway_min"])
        hit.update({"first": stats["first"], "last": stats["last"],
                    "headway_min": stats["headway_min"],
                    # 第114 レーン 3b: 1 本ごとの実発車時刻を**畳まずに**残す。
                    #   これまでは (始発, 終電, 中央値間隔) の 3 値へ潰していたので、
                    #   シム側は「等間隔の再構成」しか見られず、朝ラッシュの短縮間隔も
                    #   列車ごとの塊(platoon)も原理的に再現できなかった。
                    "departures": list(times),
                    "source": f"ODPT実ダイヤ({station_label}・平日の駅時刻表・{tier})"})
        n_real += 1
        print(f"  + {hit['name']}: {old[0]}-{old[1]} 間隔{old[2]}分(近似) → "
              f"{stats['first']}-{stats['last']} 間隔{stats['headway_min']}分"
              f"(実ダイヤ {stats['n_departures']}本・{tier})")
    # ---- 静的 GTFS フォールバック(StationTimetable が無い路線のみ。JR東の受け皿)----
    #  実ダイヤ済みの路線は上書きせず、GTFS 側の値との相互検証だけ表示する。
    for gtfs_dir, tier in GTFS_DIRS:
        for zp in sorted(gtfs_dir.glob("*-Train-GTFS.zip")) if gtfs_dir.exists() else []:
            try:
                by_route = gtfs_shibuya_departures(zp, station_filters)
            except (zipfile.BadZipFile, KeyError, OSError) as e:
                print(f"  - GTFS {zp.name}: 読めない({type(e).__name__})")
                continue
            for route, groups in sorted(by_route.items()):
                hits = [ln for ln in lines if route in ln["name"]]
                if not hits:
                    continue
                if len(hits) == 1:
                    picks = [(hits[0], groups[0])]          # 本数の多い方向を代表に
                elif all(("内回り" in h["name"]) or ("外回り" in h["name"]) for h in hits):
                    # 方向別の基底(山手線)は headsign の 内回/外回 キーワードでだけ対応付ける
                    picks = []
                    for h in hits:
                        want = "内回" if "内回り" in h["name"] else "外回"
                        g = next((g for g in groups if want in g["headsign"]), None)
                        if g is None:
                            print(f"  - GTFS {route}: {h['name']} の方向を headsign から"
                                  f"特定できず(スキップ=近似のまま。要手動対応)")
                        else:
                            picks.append((h, g))
                else:
                    print(f"  - GTFS {route}: 基底の該当が複数で対応不能(スキップ)")
                    continue
                for hit, g in picks:
                    stats = line_stats(g["times"])
                    if "実ダイヤ" in hit.get("source", ""):   # 相互検証のみ(上書きしない)
                        print(f"  = GTFS照合 {hit['name']}: 駅時刻表 {hit['first']}-{hit['last']}"
                              f" 間隔{hit['headway_min']}分 / GTFS {stats['first']}-{stats['last']}"
                              f" 間隔{stats['headway_min']}分({stats['n_departures']}本)")
                        continue
                    old = (hit["first"], hit["last"], hit["headway_min"])
                    hit.update({"first": stats["first"], "last": stats["last"],
                                "headway_min": stats["headway_min"],
                                "departures": list(g["times"]),   # 3b: 実発車時刻(畳まない)
                                "source": f"ODPT実ダイヤ({station_label}・平日・静的GTFS・{tier})"})
                    n_real += 1
                    print(f"  + {hit['name']}: {old[0]}-{old[1]} 間隔{old[2]}分(近似) → "
                          f"{stats['first']}-{stats['last']} 間隔{stats['headway_min']}分"
                          f"(GTFS {stats['n_departures']}本・{tier})")

    for ln in lines:
        ln.setdefault("source", "近似(公表の始発終電・間隔)")
        # 実ダイヤの取れなかった路線には departures を**作らない**(空列を置くと
        # 「実時刻を持っている」と読めてしまう)。シム側は列の有無で経路を分ける。

    out = {
        "meta": {
            "note": (f"{station_label}の定刻ダイヤ。source に「実ダイヤ」とある路線=ODPT の"
                     "駅時刻表(平日)から再構成(オープン枠 or チャレンジ限定データ)、"
                     "他路線=公表値の近似(駅時刻表が非提供のため)。"
                     "ダイヤ乱れは disaster 層(H4)が担う。"
                     "★departures = 1 本ごとの実発車時刻(サービス日の分。0-2 時台は +1440)。"
                     "この列を持つ路線は transit.real_departures=true のとき等間隔の"
                     "再構成ではなく実時刻そのものが到着表になる(第114 レーン 3b)。"),
            "station": base.get("meta", {}).get("station", station_label),
            "generated_by": "scripts/build_transit_odpt.py",
            "generated_at": datetime.date.today().isoformat(),
            "attribution": ("公共交通オープンデータセンター(ODPT)https://www.odpt.org/ "
                            "のデータを加工して作成"),
        },
        "lines": lines,
        "bus_lines": base.get("bus_lines", []),
    }
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {out_path.name}: 実ダイヤ {n_real} 路線 / 近似 {len(lines) - n_real} 路線")


if __name__ == "__main__":
    main()
