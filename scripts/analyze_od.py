#!/usr/bin/env python
"""②OD行列(Origin-Destination・分析スイート W2・第31バッチ)。

    python scripts/analyze_od.py runs/<name> [--district-m 100] [--out ...]

docs/plans/analytics-suite.md(W2)・docs/research/analytics-methods.md(②)の
方針に沿う **L1 読み出し専用** スクリプト。runs/<name>/l1_events.parquet だけを読み、
sim 本体・schema・conf は一切変更しない(analyze_flows_grid / build_panel と同じ疎結合)。

やること:
  - トリップ = agent ごとの route_start → arrive の対応付け(明示目的地)。
      * route_start.exit=True の到着は域外ゲートウェイ行き(= arrive.node が gateway)。
      * enter_area(域外→渋谷)は直後の route_start の起点を域外ゲートウェイに差し替える。
    → 渋谷内の移動(interior→interior)・流出(interior→gateway)・流入(gateway→interior)が
      1 枚の行列に載る。
  - ゾーン = analyze_flows_grid と同じ格子を束ねた「地区セル」(--district-m 既定100m)。
      * 地区ゾーン id = "D:<cx>:<cy>"(cx,cy = floor(x/district_m))。重心 = セル中心。
      * 域外ゾーン id = "G:<gateway_node>"(exit_area/enter_area/exit到着の (x,y) を重心に採る)。
  - 目的帰属 = 到着後(arrive → 次の route_start まで)の滞在中行動で近似:
      spend.cat・到着建物の kind(建物カテゴリ)・day_plan.what → work/food/shop/leisure/home/other。
      判定不能は other(捏造禁止)。流出(域外行き)は homing なら home・そうでなければ other。
  - 時間ビン = 1日=144step・1step=10分 → hour_bin = (step % 144) // 6(0..23 の時刻帯)。
  - 出力:
      panel/od_matrix.parquet … (origin, dest, hour_bin, purpose, trips, trips_per_day)
      panel/od_report.md       … 上位フロー表・流出入収支・目的構成(誠実性の注記つき)
      od_flowmap.html          … 自己完結の単体 HTML(外部 CDN 非依存・地区重心間のフロー線・
                                 太さ=トリップ数・上位N フィルタ・時間帯切替)

誠実性(既存レポート群から継承):
  - トリップは route_start→arrive のペアで定義した「明示目的地つき移動」に限る。ペアにならない
    route_start(到着未記録)は数えない=捏造しない。目的帰属は近似であり実測ではない。
  - 該当イベント 0 件(トリップ 0)は「データ不足」と明記し、値は一切捏造しない。決定論(ソート固定)。
  - 数値は what-if 実験値でありログ由来(渋谷の実測ではない)。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

# Windows コンソール(cp932)対策。ファイル出力は常に UTF-8。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

if _HERE not in sys.path:     # 同ディレクトリの run_dt を import
    sys.path.insert(0, _HERE)

import run_dt                  # noqa: E402  (W2-3: ランの Δt の単一の源)

# W2-3: どちらも **ラン依存**(run.dt_min)。ここの値は正準 Δt=10 の既定で、実際の値は
# analyze() が run dir から読んで各段へ引数で配る(Δt=10 なら 144 / 6 = 従来と同値)。
# ★hour_bin は step 基準(step0=開始時刻)なので、sim_min 基準に変えると Δt=10 でも
#   ビンがずれる。流儀は変えずに Δt だけ吸収する。
STEPS_PER_DAY = run_dt.CANON_STEPS_PER_DAY    # 1日の step 数(Δt=10 で 144)
STEPS_PER_HOUR = run_dt.CANON_STEPS_PER_HOUR  # hour_bin 幅(Δt=10 で 6)

# 目的カテゴリ(固定順)。優先順位でもある(work が最も定義的・other は最弱)。
_PURPOSES = ["work", "food", "shop", "leisure", "home", "other"]
_PURPOSE_RANK = {p: i for i, p in enumerate(_PURPOSES)}

# --------------------------------------------------------------------------- #
# 目的帰属の写像(出典 = L1 実データのカテゴリ確認・2026-07-14)
#   spend.cat  観測値: fixed_cost / food / taxi / shop / nightlife / lodging / medical
#   建物 kind  観測値: station / generic / retail / office / residential / public / hotel / house?
#   day_plan.what 観測値: work / meal / park / leisure / home / visit / shop / walk
# いずれも 6 目的(work/food/shop/leisure/home/other)へ写す。未知・非情報カテゴリは other。
# --------------------------------------------------------------------------- #
_SPEND_PURPOSE = {
    "food": "food",
    "shop": "shop",
    "nightlife": "leisure",
    # 以下は目的を一意に決めない弱いカテゴリ → other(捏造回避)
    "lodging": "other", "taxi": "other", "fixed_cost": "other", "medical": "other",
}
_BUILDING_PURPOSE = {
    "office": "work",
    "retail": "shop",
    "residential": "home", "house?": "home",
    # station/public/generic/hotel は目的を一意に決めない → other
    "hotel": "other", "station": "other", "public": "other", "generic": "other",
}
_DAYPLAN_PURPOSE = {
    "work": "work",
    "meal": "food",
    "shop": "shop",
    "leisure": "leisure", "park": "leisure", "walk": "leisure", "visit": "leisure",
    "home": "home",
}


def hour_bin_of_step(step: int, spd: int = STEPS_PER_DAY,
                     sph: int = STEPS_PER_HOUR) -> int:
    """hour_bin = (step % steps_per_day) // steps_per_hour(0..23 の時刻帯)。

    W2-3: 既定は正準 Δt=10 の (144, 6) で従来と完全同値。Δt=1 では (1440, 60)。"""
    return (int(step) % int(spd)) // int(sph)


def district_zone(x, y, district_m: float) -> str:
    """(x,y)[local-m] を district_m メッシュの地区ゾーン id "D:<cx>:<cy>" へ。負座標可。"""
    cx = int(math.floor(float(x) / district_m))
    cy = int(math.floor(float(y) / district_m))
    return f"D:{cx}:{cy}"


def gateway_zone(node) -> str:
    """域外ゲートウェイのゾーン id。node が無ければ総称 "G:_"。"""
    return f"G:{node}" if node else "G:_"


def zone_centroid(zone: str, district_m: float,
                  gateway_xy: dict[str, tuple[float, float]]) -> tuple[float, float]:
    """ゾーン id から重心 (x,y)。地区セルはセル中心・ゲートウェイは観測 (x,y)。"""
    if zone.startswith("D:"):
        _, cx, cy = zone.split(":")
        return ((int(cx) + 0.5) * district_m, (int(cy) + 0.5) * district_m)
    return gateway_xy.get(zone, (0.0, 0.0))


def _pick_purpose(cands) -> str:
    """候補目的の集合から優先順(work>food>shop>leisure>home>other)で1つ選ぶ。
    other 以外が1つでもあればそれを、無ければ other。"""
    best = None
    for c in cands:
        if c is None or c == "other":
            continue
        if best is None or _PURPOSE_RANK[c] < _PURPOSE_RANK[best]:
            best = c
    return best or "other"


def attribute_purpose(window_events, building_kinds: dict[str, str]) -> str:
    """到着後の滞在窓(arrive → 次 route_start までのイベント列)から目的を近似する純関数。
    強い信号(spend.cat・到着建物 kind)を優先、無ければ day_plan.what を弱い信号として使う。
    どれも決め手を欠けば other(捏造禁止)。"""
    strong: list[str] = []
    weak: list[str] = []
    for e in window_events:
        k = e["kind"]
        p = e.get("payload", {}) or {}
        if k == "spend":
            strong.append(_SPEND_PURPOSE.get(p.get("cat"), "other"))
        elif k == "enter_building":
            kind = building_kinds.get(p.get("building"))
            if kind is not None:
                strong.append(_BUILDING_PURPOSE.get(kind, "other"))
        elif k == "day_plan":
            for it in (p.get("plan") or []):
                weak.append(_DAYPLAN_PURPOSE.get(it.get("what"), "other"))
    p_strong = _pick_purpose(strong)
    if p_strong != "other":
        return p_strong
    p_weak = _pick_purpose(weak)
    return p_weak if p_weak != "other" else "other"


# --------------------------------------------------------------------------- #
# L1 読み込み(analyze_flows_grid と同じ pq→batch 逐次・payload は JSON parse)
# OD に必要な kind だけを保持する(メモリ節約)。
# --------------------------------------------------------------------------- #
_OD_KINDS = {"route_start", "arrive", "enter_area", "exit_area",
             "enter_building", "spend", "day_plan"}


def _iter_events(run_dir: str, kinds: set[str] | None = None):
    """l1_events.parquet を列射影+batch 逐次で読み、payload を dict 展開して yield。
    kinds を指定するとその kind だけを返す(OD 用の絞り込み)。"""
    import pyarrow.parquet as pq
    path = os.path.join(run_dir, "l1_events.parquet")
    want = ["step", "sim_min", "agent_id", "kind", "x", "y", "payload"]
    avail = set(pq.read_schema(path).names)
    cols = [c for c in want if c in avail]
    pf = pq.ParquetFile(path)
    seq = 0
    for batch in pf.iter_batches(columns=cols):
        d = batch.to_pydict()
        n = len(d["step"])
        for i in range(n):
            kind = d["kind"][i]
            if kinds is not None and kind not in kinds:
                seq += 1
                continue
            raw = d["payload"][i] if "payload" in d else None
            try:
                payload = json.loads(raw) if raw else {}
            except (json.JSONDecodeError, TypeError):
                payload = {}
            yield {
                "seq": seq,                       # 元の出力順(同 step 内の安定ソート鍵)
                "step": int(d["step"][i]),
                "sim_min": int(d["sim_min"][i]),
                "agent_id": int(d["agent_id"][i]) if d["agent_id"][i] is not None else -1,
                "kind": kind,
                "x": d["x"][i],
                "y": d["y"][i],
                "payload": payload,
            }
            seq += 1


def load_building_kinds(run_dir: str) -> dict[str, str]:
    """config.yaml の world.map(既定 data/shibuya_osm.json)から building_id→kind。
    地図が無ければ空 dict(=建物カテゴリ信号は使わず spend/day_plan で帰属)。"""
    map_path = None
    cfg_path = os.path.join(run_dir, "config.yaml")
    if os.path.exists(cfg_path):
        try:
            import yaml
            cfg = yaml.safe_load(open(cfg_path, encoding="utf-8")) or {}
            mp = (cfg.get("world", {}) or {}).get("map", "data/shibuya_osm.json")
        except Exception:
            mp = "data/shibuya_osm.json"
        map_path = mp if os.path.isabs(mp) else os.path.join(_ROOT, mp)
    if not map_path or not os.path.exists(map_path):
        cand = os.path.join(_ROOT, "data", "shibuya_osm.json")
        map_path = cand if os.path.exists(cand) else None
    out: dict[str, str] = {}
    if not map_path:
        return out
    try:
        city = json.loads(Path(map_path).read_text(encoding="utf-8"))
    except Exception:
        return out
    for b in city.get("buildings", []):
        bid = b.get("id")
        kind = b.get("kind")
        if bid is not None and kind is not None:
            out[bid] = str(kind)
    return out


# --------------------------------------------------------------------------- #
# トリップ抽出(純関数=テスト対象)
# --------------------------------------------------------------------------- #
def extract_trips(events, district_m: float,
                  building_kinds: dict[str, str] | None = None,
                  spd: int = STEPS_PER_DAY, sph: int = STEPS_PER_HOUR) -> dict:
    """OD 関連イベント列(flat iterable)を agent ごとに束ね、トリップ列へ変換する純関数。

    返り値 dict:
      trips      : list of {origin, dest, hour_bin, purpose, day}
      gateway_xy : {gateway_zone -> (x,y)}(HTML 重心用に観測座標を記録)
      n_route_start / n_arrive / n_unpaired : 診断用カウント
    """
    building_kinds = building_kinds or {}
    # agent ごとにイベントを集める
    by_agent: dict[int, list] = defaultdict(list)
    gateway_xy: dict[str, tuple[float, float]] = {}
    n_route_start = 0
    n_arrive = 0
    for e in events:
        by_agent[e["agent_id"]].append(e)
        if e["kind"] == "route_start":
            n_route_start += 1
        elif e["kind"] == "arrive":
            n_arrive += 1

    def _remember_gateway(node, x, y):
        z = gateway_zone(node)
        if z not in gateway_xy and x is not None and y is not None:
            gateway_xy[z] = (float(x), float(y))
        return z

    trips: list[dict] = []
    n_unpaired = 0
    for aid in sorted(by_agent):
        evs = by_agent[aid]
        # 同 step 内は元の出力順(seq)で安定ソート=決定論
        evs.sort(key=lambda e: (e["step"], e["sim_min"], e["seq"]))
        open_trip = None            # route_start で開いた保留トリップ
        pending_ext = None          # enter_area 由来の域外起点 (gateway_zone, x, y)
        pairs: list[tuple] = []     # (origin_zone, dest_zone, start_step, arrive_idx, homing, external_dest)
        for idx, e in enumerate(evs):
            k = e["kind"]
            if k == "enter_area":
                gw = e["payload"].get("gateway")
                pending_ext = (_remember_gateway(gw, e["x"], e["y"]), e["x"], e["y"])
            elif k == "exit_area":
                # 域外ゲートウェイの重心を記録(トリップは route_start.exit+arrive で拾う)
                _remember_gateway(e["payload"].get("gateway"), e["x"], e["y"])
            elif k == "route_start":
                if pending_ext is not None:      # 直前に流入 → 起点を域外ゲートウェイに
                    origin_zone = pending_ext[0]
                    pending_ext = None
                elif e["x"] is not None and e["y"] is not None:
                    origin_zone = district_zone(e["x"], e["y"], district_m)
                else:
                    origin_zone = None
                open_trip = {
                    "origin_zone": origin_zone,
                    "exit": bool(e["payload"].get("exit")),
                    "homing": bool(e["payload"].get("homing")),
                    "start_step": e["step"],
                }
            elif k == "arrive":
                if open_trip is None or open_trip["origin_zone"] is None:
                    open_trip = None
                    continue
                if e["x"] is None or e["y"] is None:
                    open_trip = None
                    continue
                if open_trip["exit"]:            # 域外行き = arrive.node が gateway
                    dest_zone = _remember_gateway(
                        e["payload"].get("node"), e["x"], e["y"])
                    external_dest = True
                else:
                    dest_zone = district_zone(e["x"], e["y"], district_m)
                    external_dest = False
                pairs.append((open_trip["origin_zone"], dest_zone,
                              open_trip["start_step"], idx,
                              open_trip["homing"], external_dest))
                open_trip = None
        if open_trip is not None and open_trip["origin_zone"] is not None:
            n_unpaired += 1                       # route_start に対応 arrive 無し=数えない

        # 目的帰属: 各 arrive の後(次 route_start まで)の滞在窓から近似
        for (origin_zone, dest_zone, start_step, arrive_idx, homing,
             external_dest) in pairs:
            if external_dest:                     # 流出(域外行き)は homing で判定
                purpose = "home" if homing else "other"
            else:
                end = len(evs)
                for j in range(arrive_idx + 1, len(evs)):
                    if evs[j]["kind"] == "route_start":
                        end = j
                        break
                purpose = attribute_purpose(evs[arrive_idx + 1:end], building_kinds)
                # 到着後に決め手が無い域内トリップでも、route_start.homing=True なら
                # 帰宅行動(home)と帰属する(homing は「自宅へ向かう」明示フラグ=捏造でない)。
                if purpose == "other" and homing:
                    purpose = "home"
            trips.append({
                "origin": origin_zone, "dest": dest_zone,
                "hour_bin": hour_bin_of_step(start_step, spd, sph),
                "purpose": purpose,
                "day": int(start_step) // int(spd),
            })
    return {"trips": trips, "gateway_xy": gateway_xy,
            "n_route_start": n_route_start, "n_arrive": n_arrive,
            "n_unpaired": n_unpaired}


def od_rows(trips: list[dict], n_days: int) -> dict[str, list]:
    """トリップ列を (origin, dest, hour_bin, purpose) で集計し parquet 列へ。
    決定論ソート: origin, dest, hour_bin, purpose 昇順。trips_per_day = trips / n_days。"""
    nd = max(1, int(n_days))
    agg: dict[tuple, int] = defaultdict(int)
    for t in trips:
        agg[(t["origin"], t["dest"], t["hour_bin"], t["purpose"])] += 1
    cols: dict[str, list] = {c: [] for c in
                             ("origin", "dest", "hour_bin", "purpose",
                              "trips", "trips_per_day")}
    for (o, d, hb, pur) in sorted(agg):
        n = agg[(o, d, hb, pur)]
        cols["origin"].append(o)
        cols["dest"].append(d)
        cols["hour_bin"].append(int(hb))
        cols["purpose"].append(pur)
        cols["trips"].append(int(n))
        cols["trips_per_day"].append(round(n / nd, 6))
    return cols


def _od_schema():
    import pyarrow as pa
    f = pa.field
    return pa.schema([
        f("origin", pa.string()), f("dest", pa.string()),
        f("hour_bin", pa.int64()), f("purpose", pa.string()),
        f("trips", pa.int64()), f("trips_per_day", pa.float64()),
    ])


# --------------------------------------------------------------------------- #
# レポート
# --------------------------------------------------------------------------- #
def _zone_label(zone: str) -> str:
    """レポート表示用のゾーン名。地区セルは "地区(cx,cy)"・域外は "域外:node"。"""
    if zone.startswith("D:"):
        _, cx, cy = zone.split(":")
        return f"地区({cx},{cy})"
    return "域外:" + zone[2:]


def _is_gateway(zone: str) -> bool:
    return zone.startswith("G:")


def render_report(run_name: str, cols: dict[str, list], info: dict,
                  district_m: float, n_days: int) -> str:
    L: list[str] = []
    total_trips = sum(cols["trips"])
    nd = max(1, int(n_days))
    L.append(f"# OD行列(Origin-Destination)— {run_name}\n")
    L.append(f"- ゾーン: {district_m:g}m 地区セル + 域外ゲートウェイ / 時間ビン: "
             f"1時間(hour_bin=(step%144)//6・0..23)。座標系 local-m。")
    L.append(f"- 対象日数 n_days={n_days} / 総トリップ={total_trips} / "
             f"route_start={info['n_route_start']} 件・arrive={info['n_arrive']} 件・"
             f"未ペア route_start={info['n_unpaired']} 件(=対応 arrive 無し・不計上)。")
    L.append("")
    L.append("## 方法と誠実性の注記")
    L.append("- **トリップ = route_start → arrive のペア**(明示目的地つき移動)。exit=True の到着は"
             "域外ゲートウェイ行き(流出)、enter_area 直後の route_start は起点を域外ゲートウェイに"
             "差し替え(流入)。対応 arrive を欠く route_start は**数えない**(捏造回避)。")
    L.append("- **目的帰属は近似**: 到着後の滞在窓(arrive→次 route_start)の spend.cat・到着建物 kind・"
             "day_plan.what を work/food/shop/leisure/home/other へ写像(優先順 work>food>shop>"
             "leisure>home>other)。決め手を欠けば other。流出は homing なら home・他は other。")
    L.append("- 数値はログ由来の what-if 実験値(渋谷の実測ではない)。トリップ 0 は「データ不足」と明記。")
    L.append("")

    if total_trips == 0:
        L.append("## 結果")
        L.append("- **トリップ 0 件(データ不足)**: route_start→arrive のペアが構成できなかった。"
                 "可視化・上位フロー・目的構成は省略する(値は捏造しない)。")
        L.append("")
        return "\n".join(L)

    # ---- 上位フロー(origin, dest 集計・時間帯/目的をまたいだ総トリップ)----
    flow: dict[tuple, int] = defaultdict(int)
    for i in range(len(cols["origin"])):
        flow[(cols["origin"][i], cols["dest"][i])] += cols["trips"][i]
    L.append("## 上位フロー(origin→dest・全時間帯/目的の総和 上位20)")
    L.append("| 順位 | 起点 | 終点 | トリップ | /日 | 種別 |")
    L.append("|---|---|---|---|---|---|")
    top = sorted(flow.items(), key=lambda kv: (-kv[1], kv[0]))[:20]
    for r, ((o, d), n) in enumerate(top, 1):
        kind = ("流入" if _is_gateway(o) else ("流出" if _is_gateway(d) else "域内"))
        L.append(f"| {r} | {_zone_label(o)} | {_zone_label(d)} | {n} | "
                 f"{n / nd:.3g} | {kind} |")
    L.append("")

    # ---- 流出入収支(ゲートウェイ別)----
    inflow: dict[str, int] = defaultdict(int)     # 域外→渋谷(gateway が origin)
    outflow: dict[str, int] = defaultdict(int)    # 渋谷→域外(gateway が dest)
    for i in range(len(cols["origin"])):
        o, d, n = cols["origin"][i], cols["dest"][i], cols["trips"][i]
        if _is_gateway(o):
            inflow[o] += n
        if _is_gateway(d):
            outflow[d] += n
    gws = sorted(set(inflow) | set(outflow))
    tot_in = sum(inflow.values())
    tot_out = sum(outflow.values())
    L.append("## 流出入収支(域外ゲートウェイ別)")
    L.append(f"- 総流入(域外→渋谷)= {tot_in} / 総流出(渋谷→域外)= {tot_out} / "
             f"純流入 = {tot_in - tot_out}(+ は流入超過)。")
    if gws:
        L.append("")
        L.append("| ゲートウェイ | 流入 | 流出 | 純流入 |")
        L.append("|---|---|---|---|")
        for z in sorted(gws, key=lambda z: -(inflow[z] + outflow[z]))[:20]:
            L.append(f"| {_zone_label(z)} | {inflow[z]} | {outflow[z]} | "
                     f"{inflow[z] - outflow[z]} |")
    else:
        L.append("- ゲートウェイ経由のトリップ **0 件**(域外流出入なし=データ不足)。")
    L.append("")

    # ---- 目的構成 ----
    by_purpose: dict[str, int] = defaultdict(int)
    for i in range(len(cols["purpose"])):
        by_purpose[cols["purpose"][i]] += cols["trips"][i]
    L.append("## 目的構成(到着後の行動から近似)")
    L.append("| 目的 | トリップ | 構成比 |")
    L.append("|---|---|---|")
    for pur in _PURPOSES:
        n = by_purpose.get(pur, 0)
        pct = f"{100.0 * n / total_trips:.1f}%" if total_trips else "―"
        L.append(f"| {pur} | {n} | {pct} |")
    L.append("")

    # ---- 時間帯プロファイル ----
    by_hour: dict[int, int] = defaultdict(int)
    for i in range(len(cols["hour_bin"])):
        by_hour[cols["hour_bin"][i]] += cols["trips"][i]
    L.append("## 時間帯プロファイル(全トリップの総和)")
    L.append("| 時間帯 | トリップ |")
    L.append("|---|---|")
    for hb in sorted(by_hour):
        L.append(f"| {hb:02d}:00-{hb + 1:02d}:00 | {by_hour[hb]} |")
    L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# 可視化(自己完結 HTML・外部 CDN 非依存・地区重心間フロー線・時間帯切替・上位N)
# --------------------------------------------------------------------------- #
def build_html(run_name: str, cols: dict[str, list], gateway_xy: dict,
               district_m: float, n_days: int, top_per_hour: int = 300,
               n_bins: int = STEPS_PER_DAY // STEPS_PER_HOUR) -> str:
    """フロー map の単体 HTML を返す。サイズ抑制のため各時間帯 top_per_hour 本まで埋め込む
    (全量は parquet 側)。ゾーン重心・フロー(o,d,hour,trips)を JSON で内包する。"""
    n_bins = int(n_bins)                           # 24(= 1日の時間帯数。Δt 非依存)
    # (o,d,hour) で集計(目的は map では合算・目的構成はレポート側)
    agg: dict[tuple, int] = defaultdict(int)
    for i in range(len(cols["origin"])):
        agg[(cols["origin"][i], cols["dest"][i], cols["hour_bin"][i])] += \
            cols["trips"][i]
    # 時間帯ごとに上位 top_per_hour 本へ絞る(決定論: trips 降順→キー昇順)
    per_hour: dict[int, list] = defaultdict(list)
    for (o, d, hb), n in agg.items():
        per_hour[hb].append((o, d, n))
    kept: list[tuple] = []
    for hb in range(n_bins):
        rows = sorted(per_hour.get(hb, []), key=lambda r: (-r[2], r[0], r[1]))
        for (o, d, n) in rows[:top_per_hour]:
            kept.append((o, d, hb, n))
    # 使うゾーンの重心
    zones_used = sorted({o for (o, d, hb, n) in kept} | {d for (o, d, hb, n) in kept})
    zidx = {z: i for i, z in enumerate(zones_used)}
    zones = []
    for z in zones_used:
        x, y = zone_centroid(z, district_m, gateway_xy)
        zones.append({"id": z, "x": round(x, 2), "y": round(y, 2),
                      "g": 1 if _is_gateway(z) else 0})
    flows = [{"o": zidx[o], "d": zidx[d], "h": hb, "n": n}
             for (o, d, hb, n) in kept]
    data = {
        "run": run_name, "district_m": district_m, "n_bins": n_bins,
        "n_days": max(1, int(n_days)), "top_per_hour": top_per_hour,
        "zones": zones, "flows": flows,
    }
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return _HTML_TEMPLATE.replace("__RUN__", run_name).replace("__DATA__", payload)


# 自己完結 HTML(外部 CDN を一切読み込まない=viz/make_viewer / analyze_flows_grid と同方針)。
_HTML_TEMPLATE = r"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OD フローマップ __RUN__</title>
<style>
  :root{--bg:#0d1117;--panel:#161b22;--ink:#e6edf3;--dim:#9aa4b2;--line:#30363d;}
  *{box-sizing:border-box;margin:0;}
  body{background:var(--bg);color:var(--ink);
    font-family:system-ui,-apple-system,"Hiragino Sans","Yu Gothic UI","Segoe UI",sans-serif;}
  header{padding:12px 16px;border-bottom:1px solid var(--line);}
  h1{font-size:15px;font-weight:600;}
  .sub{color:var(--dim);font-size:12px;margin-top:3px;}
  #wrap{display:flex;flex-wrap:wrap;gap:14px;padding:14px 16px;}
  #cv{background:#0a0e14;border:1px solid var(--line);border-radius:10px;max-width:100%;}
  .side{min-width:240px;flex:1;}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin-bottom:12px;}
  .row{display:flex;align-items:center;gap:10px;margin:8px 0;}
  input[type=range]{width:100%;accent-color:#4dabf7;}
  button{background:#21262d;color:var(--ink);border:1px solid var(--line);border-radius:8px;
    padding:6px 12px;font-size:13px;cursor:pointer;}
  button:hover{background:#30363d;}
  .big{font-size:22px;font-weight:700;}
  .leg{display:flex;align-items:center;gap:8px;font-size:12px;margin:5px 0;}
  .sw{width:16px;height:3px;border-radius:2px;display:inline-block;}
  table{border-collapse:collapse;font-size:12px;width:100%;}
  td,th{border-bottom:1px solid var(--line);padding:3px 6px;text-align:right;}
  th{color:var(--dim);font-weight:500;}
  .note{color:var(--dim);font-size:11px;line-height:1.5;}
</style></head><body>
<header><h1>OD フローマップ — __RUN__</h1><div class="sub" id="sub"></div></header>
<div id="wrap">
  <canvas id="cv" width="720" height="560"></canvas>
  <div class="side">
    <div class="card">
      <div class="row"><button id="play">▶ 再生</button>
        <div class="big" id="hlab" style="margin-left:auto"></div></div>
      <div class="row"><input type="range" id="slider" min="0" value="0"></div>
      <div class="row"><label class="note"><input type="checkbox" id="allh"> 全時間帯を合算</label></div>
      <div class="row note">上位 N フロー
        <input type="range" id="topn" min="5" max="300" value="60" style="flex:1">
        <span id="topnlab"></span></div>
    </div>
    <div class="card">
      <div class="note" style="margin-bottom:6px">凡例</div>
      <div class="leg"><span class="sw" style="background:#4dabf7"></span>フロー線(太さ=トリップ数)</div>
      <div class="leg"><span class="sw" style="background:#f6c445;height:10px;width:10px;border-radius:50%"></span>地区セル重心</div>
      <div class="leg"><span class="sw" style="background:#e8590c;height:10px;width:10px;border-radius:50%"></span>域外ゲートウェイ</div>
    </div>
    <div class="card">
      <div class="note" style="margin-bottom:6px">この時間帯の上位フロー</div>
      <table id="tbl"><thead><tr><th>起点</th><th>終点</th><th>トリップ</th></tr></thead>
      <tbody></tbody></table>
    </div>
    <div class="card note">
      トリップ = route_start→arrive のペア(明示目的地)。線は起点→終点の重心を結び、太さは
      トリップ数に比例。橙点は域外ゲートウェイ(流出入)。表示は各時間帯 上位<span id="tph"></span>本まで
      内包(全量は panel/od_matrix.parquet)。外部 CDN 非依存の単体 HTML。
    </div>
  </div>
</div>
<script>
const D=__DATA__;
const cv=document.getElementById('cv'), g=cv.getContext('2d');
const sub=document.getElementById('sub');
sub.textContent=`${D.district_m}m 地区セル / ${D.zones.length} ゾーン / ${D.flows.length} フロー / n_days=${D.n_days}`;
document.getElementById('tph').textContent=D.top_per_hour;
// bbox(重心)→ 画面座標
let minx=1e9,maxx=-1e9,miny=1e9,maxy=-1e9;
for(const z of D.zones){ if(z.x<minx)minx=z.x; if(z.x>maxx)maxx=z.x; if(z.y<miny)miny=z.y; if(z.y>maxy)maxy=z.y; }
if(!D.zones.length){ minx=maxx=miny=maxy=0; }
const pad=30, W=cv.width, H=cv.height;
const sx=(maxx>minx)?(W-pad*2)/(maxx-minx):1, sy=(maxy>miny)?(H-pad*2)/(maxy-miny):1;
const sc=Math.min(sx,sy);
function px(x){ return pad+(x-minx)*sc; }
function py(y){ return H-pad-(y-miny)*sc; }   // Y 上向き=北
let hb=0, allH=false, topN=60, playing=false;
const slider=document.getElementById('slider'); slider.max=D.n_bins-1;
const hlab=document.getElementById('hlab');
function hstr(b){ return String(b).padStart(2,'0')+':00-'+String(b+1).padStart(2,'0')+':00'; }
function curFlows(){
  let arr;
  if(allH){
    const m=new Map();
    for(const f of D.flows){ const k=f.o+'_'+f.d; m.set(k,(m.get(k)||0)+f.n); }
    arr=[...m.entries()].map(([k,n])=>{ const [o,d]=k.split('_'); return {o:+o,d:+d,n}; });
  } else {
    arr=D.flows.filter(f=>f.h===hb).map(f=>({o:f.o,d:f.d,n:f.n}));
  }
  arr.sort((a,b)=>b.n-a.n);
  return arr.slice(0,topN);
}
function draw(){
  g.clearRect(0,0,W,H);
  hlab.textContent=allH?'全時間帯':hstr(hb);
  const arr=curFlows();
  let mx=1; for(const f of arr){ if(f.n>mx)mx=f.n; }
  // フロー線
  for(const f of arr){
    const zo=D.zones[f.o], zd=D.zones[f.d];
    const x0=px(zo.x),y0=py(zo.y),x1=px(zd.x),y1=py(zd.y);
    g.strokeStyle='rgba(77,171,247,'+(0.15+0.55*f.n/mx).toFixed(3)+')';
    g.lineWidth=0.6+3.4*f.n/mx;
    g.beginPath();
    const mxp=(x0+x1)/2+(y1-y0)*0.12, myp=(y0+y1)/2-(x1-x0)*0.12;  // 軽く湾曲
    g.moveTo(x0,y0); g.quadraticCurveTo(mxp,myp,x1,y1); g.stroke();
    // 終点側に矢羽根
    const ang=Math.atan2(y1-myp,x1-mxp);
    g.fillStyle='rgba(77,171,247,.7)';
    g.beginPath(); g.moveTo(x1,y1);
    g.lineTo(x1-7*Math.cos(ang-0.4),y1-7*Math.sin(ang-0.4));
    g.lineTo(x1-7*Math.cos(ang+0.4),y1-7*Math.sin(ang+0.4));
    g.closePath(); g.fill();
  }
  // ゾーン点
  const used=new Set(); for(const f of arr){ used.add(f.o); used.add(f.d); }
  for(const i of used){
    const z=D.zones[i];
    g.fillStyle=z.g? '#e8590c':'#f6c445';
    g.beginPath(); g.arc(px(z.x),py(z.y),z.g?4:3,0,7); g.fill();
  }
  g.strokeStyle='rgba(255,255,255,.08)'; g.strokeRect(pad,pad,W-pad*2,H-pad*2);
  // 表
  const tb=document.querySelector('#tbl tbody'); tb.innerHTML='';
  for(const f of arr.slice(0,10)){
    const tr=document.createElement('tr');
    const lo=D.zones[f.o].id, ld=D.zones[f.d].id;
    tr.innerHTML=`<td>${lo}</td><td>${ld}</td><td>${f.n}</td>`;
    tb.appendChild(tr);
  }
}
slider.oninput=()=>{ hb=+slider.value; draw(); };
document.getElementById('allh').onchange=e=>{ allH=e.target.checked; draw(); };
const topnlab=document.getElementById('topnlab');
document.getElementById('topn').oninput=e=>{ topN=+e.target.value; topnlab.textContent=topN; draw(); };
topnlab.textContent=topN;
document.getElementById('play').onclick=function(){
  playing=!playing; this.textContent=playing?'⏸ 停止':'▶ 再生';
  const tick=()=>{ if(!playing)return; hb=(hb+1)%D.n_bins; slider.value=hb; allH=false;
    document.getElementById('allh').checked=false; draw();
    setTimeout(()=>requestAnimationFrame(tick), 650); };
  if(playing) tick();
};
draw();
</script></body></html>"""


# --------------------------------------------------------------------------- #
# top-level
# --------------------------------------------------------------------------- #
def analyze(run_dir: str, district_m: float = 100.0,
            out_dir: str | None = None) -> dict:
    out_dir = out_dir or os.path.join(run_dir, "panel")
    os.makedirs(out_dir, exist_ok=True)
    run_name = os.path.basename(os.path.normpath(run_dir))

    building_kinds = load_building_kinds(run_dir)
    # W2-3: 「1日」「1時間」の step 数はこのランの run.dt_min で決まる(Δt=10 なら従来と同値)。
    spd = run_dt.steps_per_day(run_dir)
    sph = run_dt.steps_per_hour(run_dir)
    info = extract_trips(_iter_events(run_dir, _OD_KINDS), district_m, building_kinds,
                         spd, sph)
    trips = info["trips"]
    n_days = (max(t["day"] for t in trips) + 1) if trips else 0
    cols = od_rows(trips, n_days)

    # parquet
    import pyarrow as pa
    import pyarrow.parquet as pq
    od_path = os.path.join(out_dir, "od_matrix.parquet")
    pq.write_table(pa.Table.from_pydict(cols, schema=_od_schema()), od_path)

    # report
    report = render_report(run_name, cols, info, district_m, n_days)
    report_path = os.path.join(out_dir, "od_report.md")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")

    # html(runs/<name>/od_flowmap.html)
    html = build_html(run_name, cols, info["gateway_xy"], district_m, n_days)
    html_path = os.path.join(run_dir, "od_flowmap.html")
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(html)

    # 上位フロー(診断表示用)
    flow: dict[tuple, int] = defaultdict(int)
    for i in range(len(cols["origin"])):
        flow[(cols["origin"][i], cols["dest"][i])] += cols["trips"][i]
    top = sorted(flow.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
    return {
        "run": run_name, "district_m": district_m, "n_days": n_days,
        "n_trips": sum(cols["trips"]), "n_rows": len(cols["origin"]),
        "n_route_start": info["n_route_start"], "n_arrive": info["n_arrive"],
        "n_unpaired": info["n_unpaired"], "n_gateways": len(info["gateway_xy"]),
        "purpose_dist": dict(Counter(t["purpose"] for t in trips)),
        "top_flows": [{"o": o, "d": d, "trips": n} for (o, d), n in top],
        "od": od_path, "report": report_path, "html": html_path,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="②OD行列(L1 読み出し専用・地区セル+域外ゲートウェイ)")
    ap.add_argument("run_dir", help="例: runs/daily300_100d")
    ap.add_argument("--district-m", type=float, default=100.0,
                    help="地区セルの辺長[m](既定100)")
    ap.add_argument("--out", default=None,
                    help="parquet/report 出力先(既定 runs/<name>/panel)")
    args = ap.parse_args()
    if not os.path.isdir(args.run_dir):
        raise SystemExit(f"not a directory: {args.run_dir}")
    if args.district_m <= 0:
        raise SystemExit("--district-m は正の数にしてください")
    info = analyze(args.run_dir, district_m=args.district_m, out_dir=args.out)
    print(f"[analyze_od] {info['run']}  district={info['district_m']:g}m  "
          f"n_days={info['n_days']}")
    print(f"  トリップ={info['n_trips']}  行数={info['n_rows']}  "
          f"route_start={info['n_route_start']}  arrive={info['n_arrive']}  "
          f"未ペア={info['n_unpaired']}  ゲートウェイ={info['n_gateways']}")
    print(f"  目的構成={info['purpose_dist']}")
    for r in info["top_flows"]:
        print(f"  上位OD: {r['o']} -> {r['d']}  {r['trips']}")
    print(f"  -> {info['od']}\n  -> {info['report']}\n  -> {info['html']}")


if __name__ == "__main__":
    main()
