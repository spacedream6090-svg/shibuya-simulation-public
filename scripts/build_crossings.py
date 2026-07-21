"""横断歩道(歩行者信号)ノードのサイドカー生成(PED-SIG / 第38バッチ P0)。

data/crossings_shibuya.json を生成する。中身は OSM の
  - highway=crossing        (歩行者横断)
  - crossing=traffic_signals(信号付き横断)
ノードの【位置(スクランブル交差点原点ローカル m)・信号有無・名前などのサブタグ】。
歩行者信号の SFM ゲート(viz/sfm.py)と scripts/synth_crowd.py --signals が読む
【描画専用】の薄い注釈ファイル(サイドカー)。

★コア地図(data/shibuya_osm*.json)は一切改変しない(ゴールデン保護)。コア地図は
  投影段階で highway=crossing / crossing タグを捨てているため、横断歩道は取れない
  (docs/research/pedestrian-signals.md §2.1)。よって本ファイルは OSM 再取得で作る。

使い方:
    python scripts/build_crossings.py
        # → data/crossings_shibuya.json(コア地図 meta と同じ既定 bbox・最新 OSM)
    python scripts/build_crossings.py --bbox 35.656 139.695 35.6625 139.706 \
        --out data/crossings_shibuya.json

周期(青/赤の秒数)は OSM にほぼ入らない(信号諸元は非公開が普通・§2.3)。よって
docs/research/pedestrian-signals.md の実測代表値(サイクル 140s / 歩行者青 37s +
青点滅 10s / 赤 93s。渋谷スクランブル 2020-08-25 実測ブログ・中確度)を信号付き横断の
既定として付す。交差点別に上書きしたい場合はサイドカーを手で編集する。

ネットワーク不通時: コア地図から crossing タグは復元できない(捨てられている)ため、
「生成できない」と正直に終了する(既定座標・既定秒数を捏造して書き出したりはしない)。

出典: © OpenStreetMap contributors (ODbL) via Overpass。再配布可(ODbL=出典表示)。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_build_map():
    """scripts/build_map.py を動的ロード(Overpass 取得・投影・原点・bbox を再利用)。"""
    spec = importlib.util.spec_from_file_location(
        "build_map", Path(__file__).with_name("build_map.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BM = _load_build_map()

# ── 信号付き横断の代表周期(docs/research/pedestrian-signals.md §2.3)──
#    渋谷スクランブル実測ブログ 2020-08-25(一次公的資料ではない=中確度)。
#    サイクル 140s = 歩行者(全方向)青 37s + 青点滅 10s + 赤 93s。
DEFAULT_CYCLE_S = 140.0
DEFAULT_GREEN_S = 37.0
DEFAULT_FLASH_S = 10.0
DEFAULT_RED_S = DEFAULT_CYCLE_S - DEFAULT_GREEN_S - DEFAULT_FLASH_S  # = 93.0

# 原点(スクランブル交差点=ローカル 0,0)からこの距離内の信号付き横断は
# 「渋谷スクランブル(全方向横断)」の一部=同相(offset 0)で一斉に青になる特殊ケース。
SCRAMBLE_RADIUS_M = 45.0

# 横断ノードから拾う代表サブタグ(§2.2 実例: 押ボタン式・点字ブロック・音響式)。
_SUBTAG_KEYS = ("button_operated", "tactile_paving", "traffic_signals:sound",
                "crossing:signals", "crossing_ref", "crossing:markings")


# --------------------------------------------------------------------------- #
# Overpass 取得(横断ノード専用クエリ。build_map のエンドポイント巡回を踏襲)
# --------------------------------------------------------------------------- #
def build_query(bbox: tuple[float, float, float, float]) -> str:
    """横断ノードだけを引く軽い Overpass QL(bbox=S,W,N,E)。"""
    s, w, n, e = bbox
    return (
        "[out:json][timeout:180];("
        f'node["highway"="crossing"]({s},{w},{n},{e});'
        f'node["crossing"]({s},{w},{n},{e});'
        f'node["crossing:signals"]({s},{w},{n},{e});'
        ");out body;"
    )


def fetch_overpass(bbox: tuple[float, float, float, float],
                   retries: int = 2) -> dict:
    """Overpass 公式 API から横断ノードを取得(1リクエスト・リトライ最小限)。

    build_map.fetch_overpass と同じミラー群を巡回するが、クエリは横断ノード専用の
    軽量版。失敗は呼び出し側でフォールバック(コア地図走査→正直終了)に回す。
    """
    query = build_query(bbox)
    body = ("data=" + urllib.request.quote(query)).encode("utf-8")
    endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass.private.coffee/api/interpreter",
        "https://overpass.osm.jp/api/interpreter",
    ]
    last_err: Exception | None = None
    for attempt in range(retries):
        endpoint = endpoints[attempt % len(endpoints)]
        try:
            req = urllib.request.Request(
                endpoint, data=body,
                headers={"Content-Type": "application/x-www-form-urlencoded",
                         "User-Agent": "shibuya-simulation research (contact: local)"})
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as ex:
            last_err = ex
            wait = 10 * (attempt + 1)
            print(f"  取得失敗({type(ex).__name__}: {ex}) — {wait}s 待って再試行 "
                  f"[{attempt + 1}/{retries}]", file=sys.stderr)
            if attempt + 1 < retries:
                time.sleep(wait)
    raise RuntimeError(f"Overpass 取得に失敗(全リトライ): {last_err}")


# --------------------------------------------------------------------------- #
# 抽出本体(純関数=ネットワーク不要。テストが合成データで叩ける)
# --------------------------------------------------------------------------- #
def _is_scramble(tags: dict) -> bool:
    """OSM が明示するスクランブル横断か(crossing_ref=pedestrian_scramble=権威タグ)。"""
    return str(tags.get("crossing_ref", "")).lower() == "pedestrian_scramble"


def _is_signalized(tags: dict) -> bool:
    """歩行者信号付き横断か。crossing=traffic_signals、または highway=traffic_signals と
    横断タグの併記(§2.2 の実例: 1ノードが両タグ持ち)、または pedestrian_scramble
    (=全方向信号横断=定義上信号付き)を信号付きとみなす。"""
    crossing = str(tags.get("crossing", ""))
    if "traffic_signals" in crossing:
        return True
    if str(tags.get("crossing:signals", "")).lower() in ("yes", "traffic_signals"):
        return True
    if tags.get("highway") == "traffic_signals" and (
            "crossing" in tags or tags.get("crossing:signals")):
        return True
    if _is_scramble(tags):
        return True
    return False


def build_crossings(raw: dict,
                    bbox: tuple[float, float, float, float],
                    origin: tuple[float, float] | None = None) -> dict:
    """生 OSM(raw)から横断歩道サイドカーを作る(決定論・純関数)。

    origin=None なら build_map.ORIGIN(スクランブル交差点)。座標はその原点基準の
    ローカル平面 m(コア地図と同じ投影)。
    """
    origin_ll = tuple(origin) if origin is not None else BM.ORIGIN
    project = BM._make_project(origin_ll)

    crossings: list[dict] = []
    n_signal = 0
    n_scramble = 0
    for el in raw.get("elements", []):
        if el.get("type") != "node":
            continue
        tags = el.get("tags") or {}
        is_crossing = (tags.get("highway") == "crossing"
                       or "crossing" in tags
                       or "crossing:signals" in tags)
        if not is_crossing:
            continue
        x, y = project(el["lat"], el["lon"])
        signal = _is_signalized(tags)
        # スクランブル: OSM の権威タグ(crossing_ref=pedestrian_scramble)を最優先。
        # タグが無くても原点(スクランブル交差点)近傍の信号横断は同交差点の一部とみなす。
        scramble = bool(signal and (_is_scramble(tags)
                                    or math.hypot(x, y) <= SCRAMBLE_RADIUS_M))
        rec: dict = {"id": int(el["id"]), "x": x, "y": y, "signal": signal}
        name = tags.get("name:ja") or tags.get("name")
        if name:
            rec["name"] = name
        if scramble:
            rec["scramble"] = True
        # 信号付きにだけ周期を付す(信号なし横断は待ちが発生しない)
        if signal:
            rec["cycle_s"] = DEFAULT_CYCLE_S
            rec["green_s"] = DEFAULT_GREEN_S
            rec["flash_s"] = DEFAULT_FLASH_S
            n_signal += 1
            if scramble:
                n_scramble += 1
        # 代表サブタグ(押ボタン式・点字ブロック・音響式など)を素通しで保存
        for key in _SUBTAG_KEYS:
            if key in tags:
                rec[key.replace(":", "_")] = tags[key]
        crossings.append(rec)

    crossings.sort(key=lambda c: c["id"])
    meta = {
        "name": "crossings_shibuya",
        "source": "© OpenStreetMap contributors (ODbL) via Overpass",
        "crs": "local-m",
        "origin_latlon": list(origin_ll),
        "bbox": list(bbox),
        "n_crossings": len(crossings),
        "n_signalized": n_signal,
        "n_scramble": n_scramble,
        "scramble_radius_m": SCRAMBLE_RADIUS_M,
        "signal_defaults": {
            "cycle_s": DEFAULT_CYCLE_S, "green_s": DEFAULT_GREEN_S,
            "flash_s": DEFAULT_FLASH_S, "red_s": DEFAULT_RED_S,
            "source": ("docs/research/pedestrian-signals.md §2.3 "
                       "(渋谷スクランブル実測ブログ 2020-08-25・中確度)"),
        },
    }
    return {"meta": meta, "crossings": crossings}


# --------------------------------------------------------------------------- #
# フォールバック(ネットワーク不通): コア地図に横断タグは無い=正直終了
# --------------------------------------------------------------------------- #
def crossings_from_core(map_path: Path) -> list[dict]:
    """コア地図から横断歩道を復元(できない=空)。捏造しないための誠実チェック。

    コア地図のノード属性は {id,x,y,name,poi,layer,gateway} のみで highway=crossing /
    crossing タグは持たない(投影段階で捨てられている・§2.1)。よって常に空を返す。
    将来コア地図が crossing を保持する形になれば、ここを実装で置き換えられる。
    """
    if not map_path.exists():
        return []
    city = json.loads(map_path.read_text(encoding="utf-8"))
    found = []
    for nd in city.get("nodes", []):
        if isinstance(nd, dict) and (nd.get("crossing") or nd.get("highway") == "crossing"):
            found.append(nd)
    return found


# --------------------------------------------------------------------------- #
def main() -> int:
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(errors="replace")   # cp932 コンソール対策(© 等)
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(
        description="横断歩道(歩行者信号)サイドカー生成(コア地図は不変)")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("S", "W", "N", "E"),
                    default=None, help="south west north east(既定=コア地図 meta の bbox)")
    ap.add_argument("--map",
                    default=str(REPO_ROOT / "data" / "shibuya_osm.json"),
                    help="bbox 既定と不通時フォールバックの参照コア地図")
    ap.add_argument("--out",
                    default=str(REPO_ROOT / "data" / "crossings_shibuya.json"),
                    help="出力先(既定=data/crossings_shibuya.json)")
    ap.add_argument("--retries", type=int, default=2,
                    help="Overpass リトライ回数(既定 2=最小限)")
    args = ap.parse_args()

    map_path = Path(args.map)
    if not map_path.is_absolute():
        map_path = REPO_ROOT / map_path
    core = json.loads(map_path.read_text(encoding="utf-8"))
    bbox = tuple(args.bbox) if args.bbox else tuple(core["meta"]["bbox"])
    origin = tuple(core.get("meta", {}).get("origin_latlon") or BM.ORIGIN)

    print(f"Overpass API から横断ノードを取得中... bbox={bbox}")
    try:
        raw = fetch_overpass(bbox, retries=args.retries)
    except RuntimeError as ex:
        print(f"  {ex}", file=sys.stderr)
        # フォールバック: コア地図に横断タグは無い=正直に生成不能で終了
        recovered = crossings_from_core(map_path)
        if not recovered:
            print("error: Overpass 不通、かつコア地図に横断(crossing)情報が無いため "
                  "サイドカーを生成できません。ネットワーク復旧後に再実行してください "
                  "(既定座標・既定秒数は捏造しません)。", file=sys.stderr)
            return 1
        raw = {"elements": recovered}

    feat = build_crossings(raw, bbox, origin=origin)
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.write_text(json.dumps(feat, ensure_ascii=False), encoding="utf-8")
    m = feat["meta"]
    print(f"written: {out}")
    print(f"  crossings={m['n_crossings']}  信号付き={m['n_signalized']}  "
          f"スクランブル={m['n_scramble']}")
    print(f"  周期既定(信号付き): cycle={DEFAULT_CYCLE_S:.0f}s 青={DEFAULT_GREEN_S:.0f}s "
          f"点滅={DEFAULT_FLASH_S:.0f}s 赤={DEFAULT_RED_S:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
