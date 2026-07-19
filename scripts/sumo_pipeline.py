#!/usr/bin/env python
"""SUMO v0 = 車限定オフライン合成パイプライン(第37バッチ Track S2 / SUMO-R v0)。

承認済み調査 docs/research/sumo-integration-research.md の「v0 = オフライン合成(推奨)」を
一気通しで実装する。シミュ本体(src/society)・conf・L1・ゴールデン・既存 tracks.json を
**一切触らない**(R1 完全無風)。synth_crowd.py と同型の「本体無風・成果物を横置き」パターン:
既存ランの OD(analyze_od.py が出す panel/od_matrix.parquet)を SUMO の車需要へ写し、信号停止
つきの車両軌跡を合成して、ビューアの交通データ(tracks.json の traffic segs 形式)へ変換する。

────────────────────────────────────────────────────────────────────────
stage 構成(各 stage 独立再実行可・synth_crowd / make_env と同じ流儀):

  net     生 OSM XML → netconvert(**--lefthand 必須**・車道のみ
          --keep-edges.by-vclass passenger・信号 --tls.guess-signals --tls.join・--seed 固定)
          → runs/<name>/sumo/<map>_car.net.xml
          生 OSM XML が無ければ Overpass 再取得コマンドを「手順として出力」する
          (既定は勝手に大容量 DL しない。--fetch-osm で明示取得=道路+信号のみの軽量クエリ)。

  demand  panel/od_matrix.parquet(analyze_od.py 出力)→ TAZ 定義 + tazRelation OD 行列 XML →
          od2trips → duarouter(全て --seed 固定)→ runs/<name>/sumo/routes.rou.xml
          ゾーン重心(地区セル=cx,cy から / 域外ゲートウェイ=ランの地図ノード座標から)を
          ローカル m → 経緯度 → net 座標(sumolib)へ写し、最寄り車道エッジへ TAZ を割当てる。

  run     sumo(GUI 不要 CLI)--seed 固定・--step-length 1.0・--fcd-output.geo(経緯度)
          --fcd-output fcd.xml → runs/<name>/sumo/fcd.xml

  convert fcd(経緯度)→ シミュのローカル m(原点=地図 meta.origin_latlon の局所接平面=
          scripts/build_map.project と同一式)へ変換し、10 分 step(=600s)窓へ集約 →
            runs/<name>/panel/sumo_traffic.parquet   … (step, veh_id, x, y)  車両位置
            runs/<name>/panel/sumo_traffic_segs.json … tracks.json の traffic segs 互換
                                                        (各 step {n, segs})=ビューア統合は後続
          **元の tracks.json は変更しない**。

  check   SUMO 本体の存在確認(sumo --version)。無ければ「pip install eclipse-sumo」を試行
          (pip 経由の公式バイナリ配布)。それも失敗したら docs/guides/sumo-setup.md の
          winget/MSI 手順を案内して正直に exit(捏造した軌跡は出さない)。

  all     check → net → demand → run → convert(既定)。

使い方:
  python scripts/sumo_pipeline.py runs/<name> [--stage all|net|demand|run|convert|check]
      [--seed 42] [--district-m 100] [--fetch-osm] [--scale 1.0]
      [--sumo-end SEC] [--fcd-period 10] [--taz-radius 120] [--taz-k 4]

誠実性(調査の誠実性方針を継承):
  - 信号推定(--tls.guess-signals)も OD も「近似」であり **渋谷の実測交通の再現ではない**。
    SUMO 版・seed・netconvert 引数・bbox は runs/<name>/panel/sumo_meta.json に刻む。
  - OD 行列(analyze_od)は mode 非分離。v0 では **全トリップを車需要の代理** として扱う
    (歩行/自転車/電車を車で微視展開する近似)。この近似を成果物 meta に明記する。
  - net に載らないゾーン(道路網 bbox 外)は落とし、落とした数を報告する(捏造しない)。
  - SUMO 不在時は軌跡を捏造せず、導入手順を案内して exit する。
────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

# 投影定数(scripts/build_map._make_project と同一式=座標整合の要)。
_LON_M = 111320.0
_LAT_M = 110540.0

STEP_SECONDS = 600.0        # 1 step = 10 分 = 600s(L1 / conf world.modes と同じ)
SECONDS_PER_HOUR = 3600     # hour_bin(0..23)→ SUMO 秒時刻の展開幅


# ─────────────────────────────────────────────── SUMO 探索(バイナリ / tools / typemap)
def sumo_home() -> str | None:
    """SUMO_HOME を返す。環境変数を最優先(公式 winget/MSI 版=XML I/O 正常)、次に公式既定パス、
    最後に eclipse-sumo(pip)の `sumo.SUMO_HOME`(pip 版 exe は Windows で XML 読込が
    即クラッシュする既知バグがあるため最後の手段)。"""
    h = os.environ.get("SUMO_HOME")
    if h and os.path.isdir(h):
        return h
    for cand in (r"C:\Program Files (x86)\Eclipse\Sumo", r"C:\Program Files\Eclipse\Sumo"):
        if os.path.isdir(cand):
            return cand
    try:
        import sumo  # type: ignore
        h = getattr(sumo, "SUMO_HOME", None)
        if h and os.path.isdir(h):
            return h
    except Exception:
        pass
    return None


def ensure_tools_on_path() -> str | None:
    """$SUMO_HOME/tools を sys.path に載せて sumolib を import 可能にする。tools パスを返す。"""
    h = sumo_home()
    if not h:
        return None
    tools = os.path.join(h, "tools")
    if os.path.isdir(tools) and tools not in sys.path:
        sys.path.insert(0, tools)
    return tools if os.path.isdir(tools) else None


def find_bin(name: str) -> str | None:
    """SUMO 実行ファイルの絶対パス。$SUMO_HOME/bin を **最優先**(pip の Scripts/*.EXE launcher
    は同梱 DLL を解決できず即クラッシュするため。本体は site-packages/sumo/bin にある)。
    無ければ PATH(which)を試す。それも無ければ None。"""
    h = sumo_home()
    if h:
        for cand in (os.path.join(h, "bin", name + ".exe"),
                     os.path.join(h, "bin", name)):
            if os.path.isfile(cand):
                return cand
    return shutil.which(name)


def _sumo_env() -> dict:
    """SUMO 実行用の環境: $SUMO_HOME/bin を PATH 先頭へ(87 個の同梱 DLL を解決)。"""
    env = dict(os.environ)
    h = sumo_home()
    if h:
        env["SUMO_HOME"] = h
        bindir = os.path.join(h, "bin")
        env["PATH"] = bindir + os.pathsep + env.get("PATH", "")
    return env


def typemap_path() -> str | None:
    h = sumo_home()
    if not h:
        return None
    p = os.path.join(h, "data", "typemap", "osmNetconvert.typ.xml")
    return p if os.path.isfile(p) else None


# ─────────────────────────────────────────────── 座標(pure・テスト対象)
def project(lat: float, lon: float, origin: tuple[float, float]) -> tuple[float, float]:
    """経緯度 → 原点基準ローカル平面 m(build_map._make_project と同一式)。"""
    o_lat, o_lon = origin
    cos_lat = math.cos(math.radians(o_lat))
    x = (lon - o_lon) * _LON_M * cos_lat
    y = (lat - o_lat) * _LAT_M
    return x, y


def inv_project(x: float, y: float, origin: tuple[float, float]) -> tuple[float, float]:
    """ローカル平面 m → 経緯度(project の逆写像)。"""
    o_lat, o_lon = origin
    cos_lat = math.cos(math.radians(o_lat))
    lat = o_lat + y / _LAT_M
    lon = o_lon + x / (_LON_M * cos_lat)
    return lat, lon


# ─────────────────────────────────────────────── ランの地図(原点・bbox・ゲートウェイ座標)
def _load_yaml(path: Path) -> dict:
    try:
        import yaml
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except ModuleNotFoundError:
        from omegaconf import OmegaConf
        return OmegaConf.to_container(OmegaConf.load(str(path)), resolve=True)


def resolve_map_path(run_dir: Path) -> Path:
    """config.yaml world.map(既定 data/shibuya_osm.json)を絶対パスで返す。"""
    cfg = run_dir / "config.yaml"
    mp = "data/shibuya_osm.json"
    if cfg.exists():
        try:
            c = _load_yaml(cfg)
            mp = (c.get("world", {}) or {}).get("map", mp)
        except Exception:
            pass
    p = Path(mp)
    return p if p.is_absolute() else (_ROOT / p)


def load_map(run_dir: Path) -> dict:
    """ランの地図から origin(lat,lon)・bbox(S,W,N,E)・node_xy(id→(x,y))を取り出す。"""
    map_path = resolve_map_path(run_dir)
    city = json.loads(map_path.read_text(encoding="utf-8"))
    meta = city.get("meta", {})
    origin = tuple(meta.get("origin_latlon", [35.65950, 139.70062]))
    bbox = tuple(meta.get("bbox", [35.6560, 139.6950, 35.6625, 139.7060]))
    node_xy = {str(n["id"]): (float(n["x"]), float(n["y"]))
               for n in city.get("nodes", []) if "x" in n and "y" in n}
    return {"map_path": map_path, "origin": origin, "bbox": bbox,
            "node_xy": node_xy, "map_name": meta.get("name", map_path.stem)}


# ─────────────────────────────────────────────── OD 行列 読み出し
def read_od_matrix(od_path: Path) -> list[dict]:
    """panel/od_matrix.parquet を (origin, dest, hour_bin, trips) の行 list へ。"""
    import pyarrow.parquet as pq
    t = pq.read_table(od_path, columns=["origin", "dest", "hour_bin", "trips"])
    return t.to_pylist()


# ─────────────────────────────────────────────── ゾーン重心 → 経緯度(pure・テスト対象)
def zone_centroid_lonlat(zone: str, district_m: float,
                         node_xy: dict[str, tuple[float, float]],
                         origin: tuple[float, float]) -> tuple[float, float] | None:
    """OD ゾーン id → (lon, lat)。
      地区セル "D:<cx>:<cy>" → セル中心((cx+0.5)*m, (cy+0.5)*m)[local-m]。
      域外   "G:<node>"      → 地図ノード座標[local-m](無ければ None)。
    どちらも build_map と同じ局所平面なので inv_project で経緯度へ戻す。"""
    if zone.startswith("D:"):
        _, cx, cy = zone.split(":")
        x = (int(cx) + 0.5) * district_m
        y = (int(cy) + 0.5) * district_m
    elif zone.startswith("G:"):
        xy = node_xy.get(zone[2:])
        if xy is None:
            return None
        x, y = xy
    else:
        return None
    lat, lon = inv_project(x, y, origin)
    return lon, lat


def taz_id(zone: str) -> str:
    """OD ゾーン id を TAZ id へ正規化(':' は SUMO 内部エッジ接頭辞と紛れるので '_' に)。"""
    return zone.replace(":", "_")


# ─────────────────────────────────────────────── TAZ / tazRelation XML(pure・テスト対象)
def build_taz_xml(zone_edges: dict[str, list[str]]) -> str:
    """{ゾーン id → [edge id,...]} を SUMO TAZ ファイル XML へ。ソート固定=決定論。"""
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<tazs>']
    for zone in sorted(zone_edges):
        edges = " ".join(zone_edges[zone])
        lines.append(f'  <taz id="{taz_id(zone)}" edges="{edges}"/>')
    lines.append('</tazs>')
    return "\n".join(lines) + "\n"


def build_tazrel_xml(rows: list[dict], kept_zones: set[str]) -> tuple[str, dict]:
    """OD 行 → tazRelation 形式 OD 行列 XML(od2trips -z 用)。
    hour_bin を interval(各 3600s)へ展開し count=trips を並べる。
    origin/dest とも kept_zones に残ったペアだけ採用(net に載らないゾーンは落とす)。
    返り値: (xml, 統計 dict)。決定論(hour→(o,d)ソート固定)。"""
    by_hour: dict[int, list[tuple[str, str, int]]] = defaultdict(list)
    n_kept = n_dropped = trips_kept = trips_dropped = 0
    for r in rows:
        o, d, n = r["origin"], r["dest"], int(r["trips"])
        if o in kept_zones and d in kept_zones:
            by_hour[int(r["hour_bin"])].append((o, d, n))
            n_kept += 1
            trips_kept += n
        else:
            n_dropped += 1
            trips_dropped += n
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<data>']
    for hb in sorted(by_hour):
        b, e = hb * SECONDS_PER_HOUR, (hb + 1) * SECONDS_PER_HOUR
        lines.append(f'  <interval id="h{hb}" begin="{b}" end="{e}">')
        for (o, d, n) in sorted(by_hour[hb]):
            lines.append(f'    <tazRelation from="{taz_id(o)}" '
                         f'to="{taz_id(d)}" count="{n}"/>')
        lines.append('  </interval>')
    lines.append('</data>')
    stats = {"n_rel_kept": n_kept, "n_rel_dropped": n_dropped,
             "trips_kept": trips_kept, "trips_dropped": trips_dropped,
             "hours": sorted(by_hour)}
    return "\n".join(lines) + "\n", stats


# ─────────────────────────────────────────────── fcd 解析 + 10 分窓集約(pure・テスト対象)
def parse_fcd_stream(source) -> list[tuple]:
    """fcd XML(--fcd-output.geo:vehicle.x=lon, y=lat)を stream 解析し
    (time, veh_id, lon, lat) の list を返す。source はパス or file-like。"""
    import xml.etree.ElementTree as ET
    out: list[tuple] = []
    t = 0.0
    for ev, el in ET.iterparse(source, events=("start", "end")):
        if ev == "start" and el.tag == "timestep":
            t = float(el.get("time", 0.0))
        elif ev == "end" and el.tag == "vehicle":
            out.append((t, el.get("id"), float(el.get("x")), float(el.get("y"))))
            el.clear()
        elif ev == "end" and el.tag == "timestep":
            el.clear()
    return out


def _downsample(pts: list, k: int) -> list:
    """時系列点列を最大 k 点へ等間隔間引き(先頭・末尾を必ず含む・決定論)。"""
    if k <= 0 or len(pts) <= k:
        return pts
    n = len(pts)
    idx = sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})
    return [pts[i] for i in idx]


def aggregate_fcd(records, origin: tuple[float, float],
                  step_seconds: float = STEP_SECONDS,
                  max_seg_pts: int = 24) -> tuple[dict, dict]:
    """(time, veh, lon, lat) 列 → 10 分窓集約。
      戻り(1) parquet 列 dict: step, veh_id, x, y(local-m・1 桁丸め)
      戻り(2) segs_by_step: {step(int) → [polyline,...]}(polyline=[[x,y],...]=1 車両の窓内軌跡)
    決定論: (window, veh_id) 昇順で構築。各車両の窓内点は時刻順→max_seg_pts へ間引き。
    step(=窓)は SUMO 秒 t を 600s で割った 0.. の窓番号(基準 1 日の 10 分粒度)。"""
    buckets: dict[tuple, list] = defaultdict(list)
    for (t, veh, lon, lat) in records:
        w = int(t // step_seconds)
        x, y = project(lat, lon, origin)
        buckets[(w, veh)].append((t, round(x, 1), round(y, 1)))

    par: dict[str, list] = {"step": [], "veh_id": [], "x": [], "y": []}
    segs_by_step: dict[int, list] = defaultdict(list)
    for (w, veh) in sorted(buckets, key=lambda k: (k[0], str(k[1]))):
        pts = sorted(buckets[(w, veh)], key=lambda r: r[0])
        pts = _downsample(pts, max_seg_pts)
        for (_t, x, y) in pts:
            par["step"].append(int(w))
            par["veh_id"].append(str(veh))
            par["x"].append(x)
            par["y"].append(y)
        poly = [[x, y] for (_t, x, y) in pts]
        if len(poly) == 1:            # 1 点だけの窓は停止点=同点2つで seg を成す
            poly = [poly[0], list(poly[0])]
        segs_by_step[int(w)].append(poly)
    return par, dict(segs_by_step)


def write_traffic_parquet(out_path: Path, par: dict) -> int:
    """(step, veh_id, x, y) を決定論 parquet で書く(synth_crowd と同じ堅い書き方)。"""
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.table({
        "step": pa.array(par["step"], type=pa.int32()),
        "veh_id": pa.array(par["veh_id"], type=pa.string()),
        "x": pa.array(par["x"], type=pa.float32()),
        "y": pa.array(par["y"], type=pa.float32()),
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path, compression="none", version="2.6",
                   write_statistics=False)
    return table.num_rows


def build_segs_json(segs_by_step: dict, meta: dict) -> str:
    """tracks.json の traffic segs 互換 JSON(各 step {n, segs})。ビューア統合は後続バッチ。"""
    steps = [{"step": w, "n": len(segs_by_step[w]), "segs": segs_by_step[w]}
             for w in sorted(segs_by_step)]
    data = {"meta": meta, "step_seconds": int(STEP_SECONDS), "steps": steps}
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


# ─────────────────────────────────────────────── check(pure ロジック・subprocess 注入で test 可)
GUIDE_MSG = (
    "SUMO 本体が見つからず pip install も失敗しました。捏造した軌跡は出しません。\n"
    "  docs/guides/sumo-setup.md の導入手順(winget / MSI / pip)を参照してください:\n"
    "    winget:  winget install --id Eclipse.SUMO\n"
    "    MSI:     https://sumo.dlr.de/docs/Downloads.php (sumo-win64-1.27.1.msi)\n"
    "    pip:     python -m pip install eclipse-sumo\n"
    "  導入後は SUMO_HOME を設定するか eclipse-sumo(pip)で再実行してください。")


def sumo_version(runner=subprocess.run) -> str | None:
    """`sumo --version` を実行しバージョン文字列を返す。不在/失敗は None(捏造しない)。"""
    exe = find_bin("sumo")
    if not exe:
        return None
    try:
        p = runner([exe, "--version"], capture_output=True, text=True,
                   timeout=60, env=_sumo_env())
    except Exception:
        return None
    if getattr(p, "returncode", 1) != 0:
        return None
    text = (getattr(p, "stdout", "") or "") + (getattr(p, "stderr", "") or "")
    for tok in text.replace("\n", " ").split():
        # "Eclipse SUMO sumo Version 1.27.1" 等から数字.数字[.数字] を拾う
        core = tok.lstrip("vV")
        parts = core.split(".")
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            return core
    return text.strip().splitlines()[0].strip() if text.strip() else "unknown"


def pip_install_sumo(runner=subprocess.run) -> bool:
    """pip 経由で eclipse-sumo(公式バイナリ配布)を導入試行。成功=True。"""
    try:
        p = runner([sys.executable, "-m", "pip", "install", "eclipse-sumo"],
                   capture_output=True, text=True, timeout=900)
    except Exception:
        return False
    return getattr(p, "returncode", 1) == 0


def ensure_sumo(version_fn=sumo_version, pip_fn=pip_install_sumo, log=print) -> dict:
    """SUMO を確保する: 存在確認 → 無ければ pip install → 再確認 → それも無理なら guide。
    返り値 {ok, how('present'|'pip'|'guide'), version}。捏造せず正直に返す。"""
    v = version_fn()
    if v:
        return {"ok": True, "how": "present", "version": v}
    log("SUMO 本体が見つかりません。pip install eclipse-sumo を試行します...")
    if pip_fn():
        ensure_tools_on_path()
        v = version_fn()
        if v:
            return {"ok": True, "how": "pip", "version": v}
    log(GUIDE_MSG)
    return {"ok": False, "how": "guide", "version": None}


# ─────────────────────────────────────────────── 生 OSM(手順出力 / 明示取得)
def _overpass_road_query(bbox: tuple[float, float, float, float]) -> str:
    """netconvert 用の軽量 OSM XML クエリ(車道 way + 信号ノード + 構成ノードのみ)。"""
    s, w, n, e = bbox
    return (f'[out:xml][timeout:180];\n'
            f'(\n'
            f'  way["highway"]({s},{w},{n},{e});\n'
            f'  node["highway"="traffic_signals"]({s},{w},{n},{e});\n'
            f');\n'
            f'(._;>;);\n'
            f'out body;\n')


def osm_fetch_instructions(bbox, raw_path: Path) -> str:
    """生 OSM XML が無い時に出す「手順」。勝手に大容量 DL しない=コマンドを案内する。"""
    s, w, n, e = bbox
    q = _overpass_road_query(bbox).replace("\n", " ")
    return (
        f"生 OSM XML が見つかりません: {raw_path}\n"
        f"  netconvert には生 OSM XML が要ります(コア地図 data/*.json は派生 JSON で不可)。\n"
        f"  以下のいずれかで 1 回だけ取得してください(bbox=このランの地図と同一):\n"
        f"  (A) SUMO 同梱 osmGet.py:\n"
        f"      python \"$SUMO_HOME/tools/osmGet.py\" "
        f"--bbox {w},{s},{e},{n} --prefix road -d \"{raw_path.parent}\"\n"
        f"  (B) Overpass(道路+信号のみ=軽量):\n"
        f'      curl -g "https://overpass-api.de/api/interpreter?data={urllib.parse.quote(q)}" '
        f'-o "{raw_path}"\n'
        f"  (C) 本スクリプトに任せて取得(道路+信号のみ):\n"
        f"      python scripts/sumo_pipeline.py <run> --stage net --fetch-osm\n")


def fetch_osm_xml(bbox, raw_path: Path, log=print) -> bool:
    """--fetch-osm 明示時のみ: Overpass から道路+信号だけの軽量 OSM XML を取得。"""
    q = _overpass_road_query(bbox)
    body = ("data=" + urllib.parse.quote(q)).encode("utf-8")
    endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass.private.coffee/api/interpreter",
    ]
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for i, ep in enumerate(endpoints):
        try:
            log(f"  Overpass 取得中(道路+信号のみ)... [{i + 1}/{len(endpoints)}] {ep}")
            req = urllib.request.Request(
                ep, data=body,
                headers={"Content-Type": "application/x-www-form-urlencoded",
                         "User-Agent": "shibuya-simulation SUMO v0 (contact: local)"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = resp.read()
            if b"<osm" not in data[:4000] and b"<?xml" not in data[:200]:
                raise ValueError("OSM XML ではない応答")
            raw_path.write_bytes(data)
            log(f"  取得完了: {raw_path} ({len(data) / 1e6:.1f} MB)")
            return True
        except Exception as ex:      # noqa: BLE001
            last = ex
            log(f"  失敗({type(ex).__name__}: {ex})")
    log(f"  Overpass 全滅: {last}")
    return False


# ─────────────────────────────────────────────── 外部コマンド実行
def run_cmd(cmd: list[str], log=print) -> int:
    """SUMO 系コマンドを実行。cp932 コンソールでも死なないよう errors=replace で受ける。"""
    log("  $ " + " ".join(str(c) for c in cmd))
    p = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=_sumo_env())
    if p.returncode != 0:
        tail = (p.stderr or p.stdout or "").strip().splitlines()[-15:]
        for ln in tail:
            log("    | " + ln)
    return p.returncode


def count_xml_tag(path: Path, tag: str) -> int:
    """XML 内の <tag ...> 出現数を stream 数える(vehicle 数・tlLogic 数の軽量カウント)。"""
    import xml.etree.ElementTree as ET
    n = 0
    for ev, el in ET.iterparse(str(path), events=("end",)):
        if el.tag == tag:
            n += 1
        el.clear()
    return n


# ─────────────────────────────────────────────── 各 stage
def sumo_dir(run_dir: Path) -> Path:
    return run_dir / "sumo"


def net_path(run_dir: Path, map_name: str) -> Path:
    return sumo_dir(run_dir) / f"{map_name}_car.net.xml"


def raw_osm_path(run_dir: Path, map_name: str) -> Path:
    return sumo_dir(run_dir) / f"{map_name}_road.osm.xml"


def stage_net(run_dir: Path, mp: dict, seed: int, fetch_osm: bool,
              log=print) -> dict:
    """生 OSM → netconvert(車道・左側通行・信号推定)→ *_car.net.xml。"""
    nc = find_bin("netconvert")
    if not nc:
        return {"ok": False, "reason": "netconvert 不在(--stage check を先に)"}
    sumo_dir(run_dir).mkdir(parents=True, exist_ok=True)
    raw = raw_osm_path(run_dir, mp["map_name"])
    out = net_path(run_dir, mp["map_name"])
    if not raw.exists():
        if fetch_osm:
            if not fetch_osm_xml(mp["bbox"], raw, log):
                return {"ok": False, "reason": "OSM 取得失敗"}
        else:
            log(osm_fetch_instructions(mp["bbox"], raw))
            return {"ok": False, "reason": "need_osm",
                    "raw": str(raw), "instructions": True}
    tm = typemap_path()
    cmd = [nc, "--osm-files", str(raw), "-o", str(out), "--seed", str(seed),
           "--lefthand",
           "--keep-edges.by-vclass", "passenger",
           "--tls.guess-signals", "--tls.join", "--tls.discard-simple",
           "--tls.default-type", "actuated",
           "--junctions.join", "--geometry.remove", "--ramps.guess",
           "--no-turnarounds", "true",
           "--no-warnings", "true"]
    if tm:
        cmd[1:1] = ["--type-files", tm]     # 挿入(osm-files の前で可)
    rc = run_cmd(cmd, log)
    if rc != 0 or not out.exists():
        return {"ok": False, "reason": f"netconvert 失敗(rc={rc})"}
    n_tls = count_xml_tag(out, "tlLogic")
    n_edges = count_xml_tag(out, "edge")
    log(f"  net: {out.name}  edges={n_edges}  信号(tlLogic)={n_tls}")
    return {"ok": True, "net": str(out), "n_signals": n_tls,
            "n_net_edges": n_edges, "netconvert_cmd": cmd}


def stage_demand(run_dir: Path, mp: dict, seed: int, district_m: float,
                 taz_radius: float, taz_k: int, scale: float, log=print) -> dict:
    """OD 行列 → TAZ + tazRelation → od2trips → duarouter → routes.rou.xml。"""
    net = net_path(run_dir, mp["map_name"])
    if not net.exists():
        return {"ok": False, "reason": "net.xml 不在(--stage net を先に)"}
    od_path = run_dir / "panel" / "od_matrix.parquet"
    if not od_path.exists():
        return {"ok": False,
                "reason": f"OD 行列不在: {od_path}(analyze_od.py を先に)"}
    od2 = find_bin("od2trips")
    dua = find_bin("duarouter")
    if not od2 or not dua:
        return {"ok": False, "reason": "od2trips/duarouter 不在"}

    ensure_tools_on_path()
    try:
        import sumolib  # type: ignore
    except Exception as ex:      # noqa: BLE001
        return {"ok": False, "reason": f"sumolib import 不可({ex})。SUMO_HOME 設定 or "
                                       f"pip install eclipse-sumo"}

    rows = read_od_matrix(od_path)
    zones = sorted({r["origin"] for r in rows} | {r["dest"] for r in rows})
    log(f"  OD: {len(rows)} 行 / {len(zones)} ゾーン を net へ写像中...")

    network = sumolib.net.readNet(str(net))
    zone_edges: dict[str, list[str]] = {}
    n_zone_ok = n_zone_drop = 0
    for zone in zones:
        ll = zone_centroid_lonlat(zone, district_m, mp["node_xy"], mp["origin"])
        if ll is None:
            n_zone_drop += 1
            continue
        lon, lat = ll
        x, y = network.convertLonLat2XY(lon, lat)
        cands = network.getNeighboringEdges(x, y, taz_radius)
        good = [(e, d) for (e, d) in cands
                if e.getFunction() != "internal" and e.allows("passenger")]
        good.sort(key=lambda ed: ed[1])
        edges = [e.getID() for e, _ in good[:taz_k]]
        if edges:
            zone_edges[zone] = edges
            n_zone_ok += 1
        else:
            n_zone_drop += 1

    kept = set(zone_edges)
    taz_xml = build_taz_xml(zone_edges)
    tazrel_xml, rel_stats = build_tazrel_xml(rows, kept)
    taz_file = sumo_dir(run_dir) / "taz.xml"
    rel_file = sumo_dir(run_dir) / "od_tazrel.xml"
    trips_file = sumo_dir(run_dir) / "trips.xml"
    routes_file = sumo_dir(run_dir) / "routes.rou.xml"
    taz_file.write_text(taz_xml, encoding="utf-8")
    rel_file.write_text(tazrel_xml, encoding="utf-8")
    log(f"  ゾーン写像: 採用={n_zone_ok} 落とし={n_zone_drop} / "
        f"OD 採用={rel_stats['n_rel_kept']}行({rel_stats['trips_kept']}台) "
        f"落とし={rel_stats['n_rel_dropped']}行({rel_stats['trips_dropped']}台)")
    if not rel_stats["hours"]:
        return {"ok": False, "reason": "net に載る OD ペアが 0(bbox/地図の不整合?)"}

    cmd_od = [od2, "-n", str(taz_file), "-z", str(rel_file),
              "-o", str(trips_file), "--seed", str(seed),
              "--tazrelation-attribute", "count", "--ignore-errors"]
    if scale != 1.0:
        cmd_od += ["--scale", str(scale)]
    if run_cmd(cmd_od, log) != 0 or not trips_file.exists():
        return {"ok": False, "reason": "od2trips 失敗"}

    # od2trips(1.27)は interval id(h0..h23)を車種 type= に書くが vType 定義は生成されず、
    # sumo が未知車種エラーで停止する → type 属性を除去して既定車種へ(決定論の純テキスト変換)
    txt = trips_file.read_text(encoding="utf-8")
    txt2 = re.sub(r'\s+type="h\d+"', "", txt)
    if txt2 != txt:
        trips_file.write_text(txt2, encoding="utf-8")
        log("  trips: interval 由来の type=hN 属性を除去(既定車種で走行)")

    cmd_dua = [dua, "-n", str(net), "-r", str(trips_file), "-o", str(routes_file),
               "--seed", str(seed), "--ignore-errors", "--no-warnings", "--repair"]
    if run_cmd(cmd_dua, log) != 0 or not routes_file.exists():
        return {"ok": False, "reason": "duarouter 失敗"}

    n_veh = count_xml_tag(routes_file, "vehicle")
    log(f"  demand: routes={routes_file.name}  車両={n_veh}")
    return {"ok": True, "routes": str(routes_file), "n_vehicles": n_veh,
            "n_zone_ok": n_zone_ok, "n_zone_drop": n_zone_drop,
            "rel_stats": rel_stats, "district_m": district_m,
            "taz_radius": taz_radius, "taz_k": taz_k, "scale": scale}


def stage_run(run_dir: Path, mp: dict, seed: int, sumo_end: float | None,
              fcd_period: float, log=print) -> dict:
    """sumo(CLI)で fcd(--fcd-output.geo)を吐く。--step-length 1.0・--seed 固定。"""
    net = net_path(run_dir, mp["map_name"])
    routes = sumo_dir(run_dir) / "routes.rou.xml"
    if not net.exists() or not routes.exists():
        return {"ok": False, "reason": "net/routes 不在(net・demand を先に)"}
    exe = find_bin("sumo")
    if not exe:
        return {"ok": False, "reason": "sumo 不在"}
    fcd = sumo_dir(run_dir) / "fcd.xml"
    end = sumo_end if sumo_end is not None else 24 * SECONDS_PER_HOUR
    cmd = [exe, "-n", str(net), "-r", str(routes),
           "--seed", str(seed), "--step-length", "1.0",
           "--begin", "0", "--end", str(int(end)),
           "--fcd-output", str(fcd), "--fcd-output.geo",
           "--device.fcd.period", str(fcd_period),
           "--fcd-output.skip-empty", "--no-step-log", "--no-warnings", "true",
           "--ignore-route-errors"]
    if run_cmd(cmd, log) != 0 or not fcd.exists():
        return {"ok": False, "reason": "sumo 失敗"}
    log(f"  run: fcd={fcd.name} ({fcd.stat().st_size / 1e6:.1f} MB)")
    return {"ok": True, "fcd": str(fcd), "sumo_end": int(end),
            "fcd_period": fcd_period}


def stage_convert(run_dir: Path, mp: dict, max_seg_pts: int,
                  meta_extra: dict | None = None, log=print) -> dict:
    """fcd(経緯度)→ local-m・10 分窓集約 → panel/sumo_traffic.parquet + _segs.json。"""
    fcd = sumo_dir(run_dir) / "fcd.xml"
    if not fcd.exists():
        return {"ok": False, "reason": "fcd.xml 不在(--stage run を先に)"}
    log("  fcd 解析中...")
    records = parse_fcd_stream(str(fcd))
    par, segs_by_step = aggregate_fcd(records, mp["origin"],
                                      STEP_SECONDS, max_seg_pts)
    panel = run_dir / "panel"
    par_path = panel / "sumo_traffic.parquet"
    segs_path = panel / "sumo_traffic_segs.json"
    n_rows = write_traffic_parquet(par_path, par)
    n_veh = len({v for v in par["veh_id"]})
    n_windows = len(segs_by_step)

    meta = {
        "run": run_dir.name,
        "source": "SUMO v0 offline (car-only) — approximate, not measured traffic",
        "map": mp["map_name"], "origin_latlon": list(mp["origin"]),
        "bbox": list(mp["bbox"]),
        "step_seconds": int(STEP_SECONDS),
        "n_vehicles": n_veh, "n_points": n_rows, "n_windows": n_windows,
        "note": ("segs は基準 1 日の 10 分窓(step=t//600)。OD は mode 非分離を車需要の代理と"
                 "して展開した近似。信号は netconvert の推定(実測現示ではない)。"),
    }
    if meta_extra:
        meta.update(meta_extra)
    segs_json = build_segs_json(segs_by_step, meta)
    segs_path.write_text(segs_json, encoding="utf-8")
    (panel / "sumo_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"  convert: 車両={n_veh} 軌跡点={n_rows} 窓={n_windows}")
    log(f"    -> {par_path}")
    log(f"    -> {segs_path}")
    return {"ok": True, "parquet": str(par_path), "segs": str(segs_path),
            "n_vehicles": n_veh, "n_points": n_rows, "n_windows": n_windows}


# ─────────────────────────────────────────────── オーケストレーション
def run_pipeline(run_dir: Path, stage: str, seed: int, district_m: float,
                 fetch_osm: bool, scale: float, sumo_end: float | None,
                 fcd_period: float, taz_radius: float, taz_k: int,
                 max_seg_pts: int, log=print) -> dict:
    mp = load_map(run_dir)
    log(f"[sumo_pipeline] run={run_dir.name} 地図={mp['map_name']} "
        f"原点={mp['origin']} stage={stage}")
    result: dict = {"stage": stage, "run_name": run_dir.name}

    do_check = stage in ("all", "check")
    if do_check:
        log("== check ==")
        chk = ensure_sumo(log=log)
        result["check"] = chk
        if not chk["ok"]:
            return result
        log(f"  SUMO {chk['version']}({chk['how']})")
    if stage == "check":
        return result

    if stage in ("all", "net"):
        log("== net ==")
        r = stage_net(run_dir, mp, seed, fetch_osm, log)
        result["net"] = r
        if not r["ok"]:
            log(f"  net 停止: {r['reason']}")
            return result
    if stage in ("all", "demand"):
        log("== demand ==")
        r = stage_demand(run_dir, mp, seed, district_m, taz_radius, taz_k,
                         scale, log)
        result["demand"] = r
        if not r["ok"]:
            log(f"  demand 停止: {r['reason']}")
            return result
    if stage in ("all", "run"):
        log("== run ==")
        r = stage_run(run_dir, mp, seed, sumo_end, fcd_period, log)
        result["run"] = r
        if not r["ok"]:
            log(f"  run 停止: {r['reason']}")
            return result
    if stage in ("all", "convert"):
        log("== convert ==")
        meta_extra = {"seed": seed}
        if "net" in result and result["net"].get("ok"):
            meta_extra["n_signals"] = result["net"]["n_signals"]
        if "demand" in result and result["demand"].get("ok"):
            meta_extra["n_vehicles_routed"] = result["demand"]["n_vehicles"]
        h = sumo_home()
        if h:
            meta_extra["sumo_home"] = h
        r = stage_convert(run_dir, mp, max_seg_pts, meta_extra, log)
        result["convert"] = r
        if not r["ok"]:
            log(f"  convert 停止: {r['reason']}")
    return result


def main(argv: list) -> int:
    for _s in (sys.stdout, sys.stderr):     # cp932 コンソール対策(synth_crowd と同修正)
        try:
            _s.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(
        description="SUMO v0 = 車限定オフライン合成(OD→車道網→軌跡→ビューア交通データ)")
    ap.add_argument("run_dir", type=Path, help="例: runs/demo_event_200a3d")
    ap.add_argument("--stage", default="all",
                    choices=["all", "check", "net", "demand", "run", "convert"])
    ap.add_argument("--seed", type=int, default=42, help="SUMO 各段の乱数 seed(固定=再現)")
    ap.add_argument("--district-m", type=float, default=100.0,
                    help="OD 地区セル辺長 m(od_matrix 生成時と同一に。既定100)")
    ap.add_argument("--fetch-osm", action="store_true",
                    help="net 段で生 OSM XML を明示取得(道路+信号のみの軽量クエリ)")
    ap.add_argument("--scale", type=float, default=1.0, help="od2trips の需要スケール")
    ap.add_argument("--sumo-end", type=float, default=None,
                    help="sumo の終了秒(既定=86400=24h。fcd を軽くしたい時に短縮)")
    ap.add_argument("--fcd-period", type=float, default=10.0,
                    help="fcd 記録周期 s(既定10。小さいほど密で大きい)")
    ap.add_argument("--taz-radius", type=float, default=120.0,
                    help="ゾーン重心→最寄り車道エッジ探索半径 m")
    ap.add_argument("--taz-k", type=int, default=4,
                    help="各ゾーンに割当てるエッジ数の上限")
    ap.add_argument("--max-seg-pts", type=int, default=24,
                    help="segs polyline の 1 車両 1 窓あたり最大点数")
    args = ap.parse_args(argv)

    run_dir = args.run_dir if args.run_dir.is_absolute() else (_ROOT / args.run_dir)
    if not run_dir.is_dir():
        print(f"error: ランが見つかりません: {run_dir}", file=sys.stderr)
        return 2

    res = run_pipeline(run_dir, args.stage, args.seed, args.district_m,
                       args.fetch_osm, args.scale, args.sumo_end, args.fcd_period,
                       args.taz_radius, args.taz_k, args.max_seg_pts)

    # 終了コード: check 不合格(SUMO 不在)は 3、stage 途中失敗は 1、成功は 0
    if args.stage in ("all", "check") and res.get("check", {}).get("ok") is False:
        return 3
    for key in ("net", "demand", "run", "convert"):
        if key in res and not res[key].get("ok", True):
            if res[key].get("reason") == "need_osm":
                return 4         # 生 OSM 手順を案内した(正常な中断)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
