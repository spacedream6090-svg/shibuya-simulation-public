#!/usr/bin/env python
"""①人流ヒートマップ + ③混雑ランキング(分析スイート W1・第30バッチ)。

    python scripts/analyze_flows_grid.py runs/<name> [--cell-m 25] [--time-bin-h 1]

docs/plans/analytics-suite.md(W1)・docs/research/analytics-methods.md(①③)の
方針に沿う **L1 読み出し専用** スクリプト。runs/<name>/l1_events.parquet だけを読み、
sim 本体・schema・conf は一切変更しない(export_3d/build_panel と同じ疎結合)。

やること:
  - 25m メッシュ(--cell-m)× 1時間ビン(--time-bin-h)へ (x,y) をビニングし、
      pass_count   : move_segment の通過(線分が触れたセルごとに +1。from/to/中点/pts 端点)
      present_count: arrive/stay の滞在 + 在館(建物 centroid)の在圏観測
      unique_agents: そのセル×時間帯で観測された distinct エージェント数
    を集計 → runs/<name>/panel/heatmap_grid.parquet。
  - ③混雑: present_count ÷ 1日 ÷ セル面積(cell_m^2) = 人/m²(相対的な在圏観測密度)
    → **Fruin の歩行者 LOS A–F** で格付け(閾値は一次資料で確認済み=下記出典)。
    セル×時間帯を LOS 降順に並べた混雑ランキングと LOS 分布を heatmap_report.md に出す。
    crowd_surge イベントの発生セルと突合する。
  - 可視化: 自己完結の単体 HTML(外部 CDN 非依存・時間スライダ付き密度面)= runs/<name>/heatmap.html。

誠実性(既存レポート群から継承):
  - present_count は **在圏観測(イベント)数の proxy** であって瞬間頭数ではない。stay は
    本シミュでは 0 件、arrive はトリップ到着ごとの疎な発火なので、人/m² は「相対密度」であり
    実測群集密度そのものではない。LOS 格付けは Fruin 尺度を当てた**目安**である旨をレポートに明記。
  - 建物内は延床面積が無いので個別 LOS は出さない(在館は centroid セルへ畳む=計画の決定事項)。
  - 該当イベント 0 件は「データ不足」と明記し、値は一切捏造しない。決定論(ソート固定)。

Fruin LOS の一次確認(2026-07-14 WebFetch 成功):
  出典 = FHWA "Recommended Procedures for Chapter 13, Pedestrians, of the Highway
  Capacity Manual"(FHWA-RD-98/107 系・NCSU 著者)TABLE 3 "Walkway LOS thresholds by
  space (m2/ped) and flow rate (ped/m/min)" の **Fruin 列(m²/ped)**:
    A > 3.2 / B 2.3–3.2 / C 1.4–2.3 / D 0.9–1.4 / E 0.5–0.9 / F < 0.5  (m²/ped)。
  本スクリプトの指標は密度(人/m²)なので **面積の逆数**へ変換して境界に用いる(下記 _FRUIN_LOS）。
  gkstill(Fruin1.html)も臨界密度 2–3 人/m² を確認 → LOS F 開始 ~2.0 人/m² と整合。
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

if _HERE not in sys.path:                       # 同ディレクトリの run_dt を import
    sys.path.insert(0, _HERE)

import run_dt                                   # noqa: E402  (W2-3: ランの Δt の単一の源)

# W2-3: STEPS_PER_DAY は **ラン依存**(run.dt_min)。ここの値は正準 Δt=10 の既定で、
# 実際の値は analyze() が run dir から読んで aggregate() へ渡す(Δt=10 なら同値)。
# hour_bin は sim_min 基準なので Δt 非依存(MIN_PER_DAY は実時間の分/日)。
STEPS_PER_DAY = run_dt.CANON_STEPS_PER_DAY      # 144
MIN_PER_DAY = 1440

# --------------------------------------------------------------------------- #
# Fruin 歩行者 LOS(人/m²)= 面積閾値(m²/ped)の逆数。出典は module docstring。
# 各要素 (grade, lo, hi): lo <= 密度 < hi のとき grade。境界は面積閾値の 1/space。
#   space>3.2 → <0.3125 = A ;  2.3–3.2 → 0.3125–0.4348 = B ;
#   1.4–2.3 → 0.4348–0.7143 = C ; 0.9–1.4 → 0.7143–1.1111 = D ;
#   0.5–0.9 → 1.1111–2.0 = E ;  space<0.5 → >2.0 = F。
# --------------------------------------------------------------------------- #
_FRUIN_LOS = [
    ("A", 0.0, 1.0 / 3.2),
    ("B", 1.0 / 3.2, 1.0 / 2.3),
    ("C", 1.0 / 2.3, 1.0 / 1.4),
    ("D", 1.0 / 1.4, 1.0 / 0.9),
    ("E", 1.0 / 0.9, 1.0 / 0.5),
    ("F", 1.0 / 0.5, math.inf),
]
_LOS_ORDER = {g: i for i, (g, _, _) in enumerate(_FRUIN_LOS)}  # A=0 .. F=5
# LOS 色(HTML/凡例共用。A=緑…F=赤の連続)
_LOS_COLOR = {"A": "#2f9e44", "B": "#94c520", "C": "#f6c445",
              "D": "#f59f00", "E": "#f76707", "F": "#e03131"}


def los_grade(density_ppm2: float) -> str:
    """在圏観測密度(人/m²)を Fruin 歩行者 LOS A–F に格付け。負/None は 'A' 扱いの前に0で保護。"""
    d = float(density_ppm2 or 0.0)
    for g, lo, hi in _FRUIN_LOS:
        if lo <= d < hi:
            return g
    return "F" if d >= _FRUIN_LOS[-1][1] else "A"


def cell_of(x, y, cell_m: float) -> tuple[int, int]:
    """(x,y)[local-m] を cell_m メッシュのセル index (cell_x, cell_y) へ。負座標可。"""
    return (int(math.floor(float(x) / cell_m)), int(math.floor(float(y) / cell_m)))


def hour_bin_of(sim_min: int, time_bin_h: int) -> int:
    """sim_min を「時刻ビン」(1日内の時間帯 index)へ。既定 time_bin_h=1 で 0..23。
    暦日を跨いだ滞留は同じ時刻ビンへ集約(モバイル空間統計=メッシュ×時間帯の慣行)。"""
    minute_of_day = int(sim_min) % MIN_PER_DAY
    width = 60 * int(time_bin_h)
    return minute_of_day // width


# --------------------------------------------------------------------------- #
# L1 読み込み(export_3d と同じ pq→pylist・payload は JSON 文字列を parse)
# --------------------------------------------------------------------------- #
def _iter_events(run_dir: str):
    """l1_events.parquet を列射影+batch 逐次で読み、payload を dict 展開して yield。
    peak メモリを下げるため一括 to_pylist しない(measure.load_events と同流儀)。"""
    import pyarrow.parquet as pq
    path = os.path.join(run_dir, "l1_events.parquet")
    want = ["step", "sim_min", "agent_id", "kind", "x", "y", "payload"]
    avail = set(pq.read_schema(path).names)
    cols = [c for c in want if c in avail]
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(columns=cols):
        d = batch.to_pydict()
        n = len(d["step"])
        for i in range(n):
            raw = d["payload"][i] if "payload" in d else None
            try:
                payload = json.loads(raw) if raw else {}
            except (json.JSONDecodeError, TypeError):
                payload = {}
            yield {
                "step": int(d["step"][i]),
                "sim_min": int(d["sim_min"][i]),
                "agent_id": int(d["agent_id"][i]),
                "kind": d["kind"][i],
                "x": d["x"][i],
                "y": d["y"][i],
                "payload": payload,
            }


def load_building_centroids(run_dir: str) -> dict[str, tuple[float, float]]:
    """config.yaml の world.map(既定 data/shibuya_osm.json)から building_id→(cx,cy)。
    cx/cy が無ければ footprint 平均。地図が無ければ空 dict(=在館は event の (x,y) にフォールバック)。"""
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
    out: dict[str, tuple[float, float]] = {}
    if not map_path:
        return out
    try:
        city = json.loads(Path(map_path).read_text(encoding="utf-8"))
    except Exception:
        return out
    for b in city.get("buildings", []):
        bid = b.get("id")
        if bid is None:
            continue
        cx, cy = b.get("cx"), b.get("cy")
        if cx is None or cy is None:
            fp = b.get("footprint") or []
            if fp:
                cx = sum(float(p[0]) for p in fp) / len(fp)
                cy = sum(float(p[1]) for p in fp) / len(fp)
        if cx is not None and cy is not None:
            out[bid] = (float(cx), float(cy))
    return out


# --------------------------------------------------------------------------- #
# 集計(純関数=テスト対象)
# --------------------------------------------------------------------------- #
def _seg_points(e: dict) -> list[tuple[float, float]]:
    """move_segment の通過点候補: pts 全頂点(あれば)/ from_xy・to_xy / 端点(x,y) / 中点。
    過剰精度不要=これらのユニークセルへ +1 する簡易割当(analytics-methods ①)。"""
    p = e.get("payload", {})
    base: list[tuple[float, float]] = []               # 経路の元頂点(pts / from_xy・to_xy)
    raw = p.get("pts")
    if isinstance(raw, list) and raw:
        for q in raw:
            if isinstance(q, (list, tuple)) and len(q) >= 2:
                base.append((float(q[0]), float(q[1])))
    else:
        for key in ("from_xy", "to_xy"):
            q = p.get(key)
            if isinstance(q, (list, tuple)) and len(q) >= 2:
                base.append((float(q[0]), float(q[1])))
    pts = list(base)
    end = None
    if e.get("x") is not None and e.get("y") is not None:
        end = (float(e["x"]), float(e["y"]))           # 端点(到達点)
        pts.append(end)
    # from/to の中点(元頂点の両端から算定=長辺セルの取りこぼしを埋める)
    ext = base if len(base) >= 2 else ([base[0], end] if base and end else [])
    if len(ext) >= 2:
        (x0, y0), (x1, y1) = ext[0], ext[-1]
        pts.append(((x0 + x1) / 2.0, (y0 + y1) / 2.0))
    return pts


def aggregate(events, cell_m: float, time_bin_h: int,
              centroids: dict[str, tuple[float, float]] | None = None,
              spd: int = STEPS_PER_DAY) -> dict:
    """イベント列を (cell_x, cell_y, hour_bin) へ集計。純関数(events は iterable の dict)。
    返り値: pass/present(dict key->count)・uniq(key->set)・days(set)・surges(list)・
            n_present_events / n_pass_events(診断用)。

    W2-3: `spd`(1日の step 数)は day の切り方だけに効く。既定は正準 144(Δt=10)。
    hour_bin は sim_min 基準なので Δt に依存しない。"""
    centroids = centroids or {}
    pass_c: dict = defaultdict(int)
    present_c: dict = defaultdict(int)
    uniq: dict = defaultdict(set)
    days: set = set()
    surges: list = []
    n_pass_ev = 0
    n_present_ev = 0
    for e in events:
        kind = e["kind"]
        days.add(int(e["step"]) // int(spd))
        hb = hour_bin_of(e["sim_min"], time_bin_h)
        aid = e["agent_id"]
        if kind == "move_segment":
            n_pass_ev += 1
            cells = {cell_of(x, y, cell_m) for (x, y) in _seg_points(e)}
            for (cx, cy) in cells:
                pass_c[(cx, cy, hb)] += 1
                if aid is not None and aid >= 0:
                    uniq[(cx, cy, hb)].add(aid)
        elif kind in ("arrive", "stay"):
            if e.get("x") is None or e.get("y") is None:
                continue
            n_present_ev += 1
            key = (*cell_of(e["x"], e["y"], cell_m), hb)
            present_c[key] += 1
            if aid is not None and aid >= 0:
                uniq[key].add(aid)
        elif kind == "enter_building":
            bid = e["payload"].get("building")
            xy = centroids.get(bid)
            if xy is None:
                if e.get("x") is None or e.get("y") is None:
                    continue
                xy = (float(e["x"]), float(e["y"]))
            n_present_ev += 1
            key = (*cell_of(xy[0], xy[1], cell_m), hb)
            present_c[key] += 1
            if aid is not None and aid >= 0:
                uniq[key].add(aid)
        elif kind == "crowd_surge":
            if e.get("x") is None or e.get("y") is None:
                continue
            cx, cy = cell_of(e["x"], e["y"], cell_m)
            surges.append({
                "cell_x": cx, "cell_y": cy, "hour_bin": hb,
                "node": e["payload"].get("node"),
                "level": e["payload"].get("level"),
                "event": e["payload"].get("event"),
                "day": int(e["step"]) // int(spd),
            })
    return {"pass": pass_c, "present": present_c, "uniq": uniq, "days": days,
            "surges": surges, "n_pass_ev": n_pass_ev, "n_present_ev": n_present_ev}


def grid_rows(agg: dict) -> dict[str, list]:
    """集計 dict を parquet 列(決定論ソート: cell_x, cell_y, hour_bin 昇順)へ。"""
    keys = set(agg["pass"]) | set(agg["present"]) | set(agg["uniq"])
    ordered = sorted(keys)                              # (cx, cy, hb) タプル昇順=決定論
    cols: dict[str, list] = {c: [] for c in
                             ("cell_x", "cell_y", "hour_bin", "pass_count",
                              "present_count", "unique_agents")}
    for (cx, cy, hb) in ordered:
        cols["cell_x"].append(int(cx))
        cols["cell_y"].append(int(cy))
        cols["hour_bin"].append(int(hb))
        cols["pass_count"].append(int(agg["pass"].get((cx, cy, hb), 0)))
        cols["present_count"].append(int(agg["present"].get((cx, cy, hb), 0)))
        cols["unique_agents"].append(len(agg["uniq"].get((cx, cy, hb), ())))
    return cols


def _grid_schema():
    import pyarrow as pa
    f = pa.field
    return pa.schema([
        f("cell_x", pa.int64()), f("cell_y", pa.int64()),
        f("hour_bin", pa.int64()), f("pass_count", pa.int64()),
        f("present_count", pa.int64()), f("unique_agents", pa.int64()),
    ])


# --------------------------------------------------------------------------- #
# LOS ランキング / 分布
# --------------------------------------------------------------------------- #
def los_table(cols: dict[str, list], cell_m: float, n_days: int) -> list[dict]:
    """各 (cell, hour_bin) の在圏観測密度(人/m²/日)と Fruin LOS を計算し、
    LOS 降順→密度降順に並べた行を返す(present_count>0 のセルのみ)。"""
    area = float(cell_m) * float(cell_m)
    nd = max(1, int(n_days))
    rows = []
    for i in range(len(cols["cell_x"])):
        pc = cols["present_count"][i]
        if pc <= 0:
            continue
        density = pc / nd / area                        # 人/m²/日(相対在圏密度)
        rows.append({
            "cell_x": cols["cell_x"][i], "cell_y": cols["cell_y"][i],
            "hour_bin": cols["hour_bin"][i], "present_count": pc,
            "unique_agents": cols["unique_agents"][i],
            "density_ppm2": density, "los": los_grade(density),
        })
    rows.sort(key=lambda r: (-_LOS_ORDER[r["los"]], -r["density_ppm2"],
                             r["cell_x"], r["cell_y"], r["hour_bin"]))
    return rows


def _hhmm_bin(hb: int, time_bin_h: int) -> str:
    a = hb * time_bin_h
    b = a + time_bin_h
    return f"{a:02d}:00-{b:02d}:00"


# --------------------------------------------------------------------------- #
# レポート
# --------------------------------------------------------------------------- #
def render_report(run_name: str, cols: dict[str, list], agg: dict,
                  cell_m: float, time_bin_h: int, n_days: int) -> str:
    area = float(cell_m) * float(cell_m)
    L: list[str] = []
    L.append(f"# 人流ヒートマップ + 混雑 LOS ランキング — {run_name}\n")
    L.append(f"- グリッド: {cell_m:g}m メッシュ(セル面積 {area:g} m²)× "
             f"{time_bin_h}時間ビン({1440 // (60 * time_bin_h)} 時間帯)。座標系 local-m。")
    L.append(f"- 対象日数 n_days={n_days} / 集計セル数={len(cols['cell_x'])} "
             f"/ move_segment={agg['n_pass_ev']} 件・在圏観測={agg['n_present_ev']} 件。")
    L.append("")
    L.append("## 方法と誠実性の注記")
    L.append("- **pass_count** は move_segment が触れたセルへの通過数(from/to/中点/pts 端点の"
             "簡易割当・過剰精度なし)。**present_count** は arrive/stay の滞在 + 在館"
             "(建物 centroid)の在圏観測数。**unique_agents** はそのセル×時間帯の distinct 体数。")
    L.append("- **人/m² は「相対在圏密度」= present_count ÷ n_days ÷ セル面積** であり、"
             "瞬間頭数ではない(stay は本シミュ 0 件・arrive はトリップ到着ごとの疎な発火)。"
             "LOS は Fruin 尺度を当てた**目安**として読む。")
    L.append("- LOS 閾値の出典(一次確認済み): FHWA *Recommended Procedures for Chapter 13, "
             "Pedestrians, of the HCM* TABLE 3 の Fruin 列(m²/ped: A>3.2 / B 2.3–3.2 / "
             "C 1.4–2.3 / D 0.9–1.4 / E 0.5–0.9 / F<0.5)を密度(人/m²)へ逆数変換して使用。")
    L.append("- 建物内は延床面積が無いため個別 LOS を出さない(在館は centroid セルへ畳む=計画の決定事項)。")
    L.append("")

    # ---- 最混雑セル(present_count 総和)----
    present_by_cell: dict = defaultdict(int)
    pass_by_cell: dict = defaultdict(int)
    uniq_by_cell: dict = defaultdict(set)
    for i in range(len(cols["cell_x"])):
        c = (cols["cell_x"][i], cols["cell_y"][i])
        present_by_cell[c] += cols["present_count"][i]
        pass_by_cell[c] += cols["pass_count"][i]
    for (cx, cy, hb), s in agg["uniq"].items():
        uniq_by_cell[(cx, cy)] |= s

    def _cellm(cx, cy):
        return f"({cx*cell_m:g},{cy*cell_m:g})m〜"

    L.append("## 最混雑セル(在圏観測 present_count の総和 上位10)")
    L.append("| 順位 | セル(cell_x,cell_y) | 左下座標 | present総和 | pass総和 | distinct体 |")
    L.append("|---|---|---|---|---|---|")
    top_present = sorted(present_by_cell.items(),
                         key=lambda kv: (-kv[1], kv[0]))[:10]
    if top_present:
        for r, ((cx, cy), v) in enumerate(top_present, 1):
            L.append(f"| {r} | ({cx},{cy}) | {_cellm(cx,cy)} | {v} | "
                     f"{pass_by_cell.get((cx,cy),0)} | {len(uniq_by_cell.get((cx,cy),()))} |")
    else:
        L.append("| ― | ― | ― | 0 | ― | ― |(在圏観測なし=データ不足)")
    L.append("")

    L.append("## 最通過セル(pass_count の総和 上位10)")
    L.append("| 順位 | セル(cell_x,cell_y) | 左下座標 | pass総和 | present総和 |")
    L.append("|---|---|---|---|---|")
    top_pass = sorted(pass_by_cell.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
    if top_pass:
        for r, ((cx, cy), v) in enumerate(top_pass, 1):
            L.append(f"| {r} | ({cx},{cy}) | {_cellm(cx,cy)} | {v} | "
                     f"{present_by_cell.get((cx,cy),0)} |")
    else:
        L.append("| ― | ― | ― | 0 | ― |(通過なし=データ不足)")
    L.append("")

    # ---- 時間帯別の総在圏 ----
    present_by_hour: dict = defaultdict(int)
    for i in range(len(cols["cell_x"])):
        present_by_hour[cols["hour_bin"][i]] += cols["present_count"][i]
    L.append("## 混雑の時間帯プロファイル(全セル在圏観測の総和)")
    L.append("| 時間帯 | present総和 |")
    L.append("|---|---|")
    for hb in sorted(present_by_hour):
        L.append(f"| {_hhmm_bin(hb, time_bin_h)} | {present_by_hour[hb]} |")
    L.append("")

    # ---- LOS ランキング + 分布 ----
    rows = los_table(cols, cell_m, n_days)
    dist = Counter(r["los"] for r in rows)
    L.append("## 混雑 LOS ランキング(セル×時間帯・LOS 降順→密度降順 上位20)")
    L.append("| 順位 | LOS | セル | 時間帯 | 人/m²/日 | present | distinct体 |")
    L.append("|---|---|---|---|---|---|---|")
    if rows:
        for r, row in enumerate(rows[:20], 1):
            L.append(f"| {r} | {row['los']} | ({row['cell_x']},{row['cell_y']}) | "
                     f"{_hhmm_bin(row['hour_bin'], time_bin_h)} | "
                     f"{row['density_ppm2']:.4g} | {row['present_count']} | "
                     f"{row['unique_agents']} |")
    else:
        L.append("| ― | ― | ― | ― | ― | 0 | ― |(在圏観測なし=データ不足)")
    L.append("")
    L.append("## LOS 分布(present_count>0 のセル×時間帯)")
    L.append("| LOS | セル×時間帯数 |")
    L.append("|---|---|")
    total = sum(dist.values())
    for g, _lo, _hi in _FRUIN_LOS:
        n = dist.get(g, 0)
        pct = f"{100.0*n/total:.1f}%" if total else "―"
        L.append(f"| {g} | {n} ({pct}) |")
    L.append("")

    # ---- crowd_surge 突合 ----
    L.append("## crowd_surge との突合(検証)")
    surges = agg["surges"]
    if not surges:
        L.append("- crowd_surge イベント **0 件**(annual 群集イベントが無効なラン)=突合対象なし"
                 "(データ不足)。混雑ランキングは在圏観測のみに基づく。")
    else:
        rank_of = {(row["cell_x"], row["cell_y"], row["hour_bin"]): i + 1
                   for i, row in enumerate(rows)}
        L.append(f"- crowd_surge {len(surges)} 件を発生セル×時間帯に割り当て、LOS ランキング上の"
                 "順位と突合(発生セルが在圏密度でも上位なら整合)。")
        L.append("")
        L.append("| surge日 | 集会node | level | セル | 時間帯 | LOSランク | 同セルpresent |")
        L.append("|---|---|---|---|---|---|---|")
        present_lookup = {(cols["cell_x"][i], cols["cell_y"][i], cols["hour_bin"][i]):
                          cols["present_count"][i] for i in range(len(cols["cell_x"]))}
        for s in sorted(surges, key=lambda s: (s["day"], -(s["level"] or 0)))[:20]:
            key = (s["cell_x"], s["cell_y"], s["hour_bin"])
            rk = rank_of.get(key)
            L.append(f"| {s['day']} | {s['node']} | {s['level']} | "
                     f"({s['cell_x']},{s['cell_y']}) | "
                     f"{_hhmm_bin(s['hour_bin'], time_bin_h)} | "
                     f"{rk if rk else '圏外'} | {present_lookup.get(key, 0)} |")
    L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# 可視化(自己完結 HTML・外部 CDN 非依存・時間スライダ付き密度面)
# --------------------------------------------------------------------------- #
def build_html(run_name: str, cols: dict[str, list], cell_m: float,
               time_bin_h: int, n_days: int) -> str:
    area = float(cell_m) * float(cell_m)
    nd = max(1, int(n_days))
    n_bins = 1440 // (60 * int(time_bin_h))
    # セル→時間帯別 density / pass を圧縮した配列に整形
    per_cell: dict = {}
    for i in range(len(cols["cell_x"])):
        c = (cols["cell_x"][i], cols["cell_y"][i])
        d = per_cell.setdefault(c, {"d": [0.0] * n_bins, "p": [0] * n_bins,
                                    "u": [0] * n_bins})
        hb = cols["hour_bin"][i]
        d["d"][hb] = round(cols["present_count"][i] / nd / area, 5)
        d["p"][hb] = cols["pass_count"][i]
        d["u"][hb] = cols["unique_agents"][i]
    cells = [{"x": cx, "y": cy, "d": v["d"], "p": v["p"], "u": v["u"]}
             for (cx, cy), v in sorted(per_cell.items())]
    data = {
        "run": run_name, "cell_m": cell_m, "time_bin_h": time_bin_h,
        "n_bins": n_bins, "n_days": nd, "area": area,
        "los": [{"g": g, "lo": (0.0 if lo == 0 else round(lo, 5)),
                 "hi": (None if hi == math.inf else round(hi, 5)),
                 "c": _LOS_COLOR[g]} for g, lo, hi in _FRUIN_LOS],
        "cells": cells,
    }
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return _HTML_TEMPLATE.replace("__RUN__", run_name).replace("__DATA__", payload)


# 自己完結 HTML(three.js 等の外部 CDN を一切読み込まない=viz/make_viewer と同じ方針)。
_HTML_TEMPLATE = r"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>人流ヒートマップ __RUN__</title>
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
  .side{min-width:230px;flex:1;}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin-bottom:12px;}
  .row{display:flex;align-items:center;gap:10px;margin:8px 0;}
  input[type=range]{width:100%;accent-color:#f59f00;}
  button{background:#21262d;color:var(--ink);border:1px solid var(--line);border-radius:8px;
    padding:6px 12px;font-size:13px;cursor:pointer;}
  button:hover{background:#30363d;}
  .leg{display:flex;align-items:center;gap:8px;font-size:12px;margin:5px 0;}
  .sw{width:16px;height:16px;border-radius:4px;display:inline-block;}
  .big{font-size:22px;font-weight:700;}
  table{border-collapse:collapse;font-size:12px;width:100%;}
  td,th{border-bottom:1px solid var(--line);padding:3px 6px;text-align:right;}
  th{color:var(--dim);font-weight:500;}
  .note{color:var(--dim);font-size:11px;line-height:1.5;}
</style></head><body>
<header><h1>人流ヒートマップ / 混雑 LOS — __RUN__</h1>
<div class="sub" id="sub"></div></header>
<div id="wrap">
  <canvas id="cv" width="720" height="560"></canvas>
  <div class="side">
    <div class="card">
      <div class="row"><button id="play">▶ 再生</button>
        <div class="big" id="hlab" style="margin-left:auto"></div></div>
      <div class="row"><input type="range" id="slider" min="0" value="0"></div>
      <div class="row" style="justify-content:space-between">
        <label class="note"><input type="checkbox" id="mode"> 通過量(pass)で色付け</label>
        <span class="note" id="peak"></span></div>
    </div>
    <div class="card">
      <div class="note" style="margin-bottom:6px">Fruin 歩行者 LOS(人/m²/日・相対在圏密度)</div>
      <div id="legend"></div>
    </div>
    <div class="card">
      <div class="note" style="margin-bottom:6px">この時間帯の混雑セル 上位</div>
      <table id="tbl"><thead><tr><th>セル</th><th>人/m²/日</th><th>LOS</th><th>present</th></tr></thead>
      <tbody></tbody></table>
    </div>
    <div class="card note">
      present_count は arrive/stay + 在館(centroid)の在圏観測数。人/m²/日 は
      present ÷ n_days ÷ セル面積 の<b>相対密度</b>で瞬間頭数ではない。LOS 閾値は
      FHWA Ch.13(Fruin 列 m²/ped)を密度へ逆数変換した目安。外部 CDN 非依存の単体 HTML。
    </div>
  </div>
</div>
<script>
const D=__DATA__;
const cv=document.getElementById('cv'), g=cv.getContext('2d');
const losColor=v=>{ for(const l of D.los){ if(v>=l.lo && (l.hi===null||v<l.hi)) return l.c; } return D.los[D.los.length-1].c; };
const losGrade=v=>{ for(const l of D.los){ if(v>=l.lo && (l.hi===null||v<l.hi)) return l.g; } return 'F'; };
// セル bbox
let minx=1e9,maxx=-1e9,miny=1e9,maxy=-1e9;
for(const c of D.cells){ if(c.x<minx)minx=c.x; if(c.x>maxx)maxx=c.x; if(c.y<miny)miny=c.y; if(c.y>maxy)maxy=c.y; }
if(!D.cells.length){ minx=maxx=miny=maxy=0; }
const cols=maxx-minx+1, rows=maxy-miny+1;
const pad=24;
const s=Math.max(1, Math.min((cv.width-pad*2)/cols, (cv.height-pad*2)/rows));
const ox=(cv.width - cols*s)/2, oy=(cv.height - rows*s)/2;
// 最大値(スケール): density は LOS 上限(2.0)で頭打ち、pass は 95 パーセンタイル
let maxPass=1; for(const c of D.cells){ for(const p of c.p){ if(p>maxPass)maxPass=p; } }
let hb=0, playing=false, useDens=true;
const slider=document.getElementById('slider'); slider.max=D.n_bins-1;
const hlab=document.getElementById('hlab'), sub=document.getElementById('sub');
sub.textContent=`${D.cell_m}m メッシュ × ${D.time_bin_h}h ビン / ${D.cells.length} セル / n_days=${D.n_days}`;
function hstr(b){ const a=b*D.time_bin_h; return String(a).padStart(2,'0')+':00-'+String(a+D.time_bin_h).padStart(2,'0')+':00'; }
function draw(){
  g.clearRect(0,0,cv.width,cv.height);
  hlab.textContent=hstr(hb);
  for(const c of D.cells){
    const gx=ox+(c.x-minx)*s, gy=oy+(maxy-c.y)*s;   // Y 上向き=北
    let fill, a;
    if(useDens){ const v=c.d[hb]; if(v<=0) continue; fill=losColor(v); a=Math.min(1,0.25+v/2.0*0.75); }
    else { const v=c.p[hb]; if(v<=0) continue; fill=losColor(v/maxPass*2.0); a=Math.min(1,0.2+v/maxPass*0.8); }
    g.globalAlpha=a; g.fillStyle=fill; g.fillRect(gx,gy,Math.ceil(s),Math.ceil(s));
  }
  g.globalAlpha=1;
  // 枠
  g.strokeStyle='rgba(255,255,255,.10)'; g.strokeRect(ox,oy,cols*s,rows*s);
  // 上位表 & ピーク
  const arr=D.cells.map(c=>({x:c.x,y:c.y,d:c.d[hb],p:c.p[hb]})).filter(o=>o.d>0)
    .sort((a,b)=>b.d-a.d);
  const tb=document.querySelector('#tbl tbody'); tb.innerHTML='';
  for(const o of arr.slice(0,8)){
    const tr=document.createElement('tr');
    tr.innerHTML=`<td>(${o.x},${o.y})</td><td>${o.d.toFixed(3)}</td>`
      +`<td style="color:${losColor(o.d)}">${losGrade(o.d)}</td><td>${Math.round(o.d*D.n_days*D.area)}</td>`;
    tb.appendChild(tr);
  }
  document.getElementById('peak').textContent = arr.length? ('最大 '+arr[0].d.toFixed(3)+' 人/m²/日') : '在圏なし';
}
function legend(){
  const el=document.getElementById('legend');
  el.innerHTML=D.los.map(l=>{
    const rng = (l.hi===null)? ('≥'+l.lo.toFixed(2)) : (l.lo.toFixed(2)+'–'+l.hi.toFixed(2));
    return `<div class="leg"><span class="sw" style="background:${l.c}"></span>LOS ${l.g} <span class="note">(${rng} 人/m²)</span></div>`;
  }).join('');
}
slider.oninput=()=>{ hb=+slider.value; draw(); };
document.getElementById('mode').onchange=e=>{ useDens=!e.target.checked; draw(); };
document.getElementById('play').onclick=function(){
  playing=!playing; this.textContent=playing?'⏸ 停止':'▶ 再生';
  const tick=()=>{ if(!playing)return; hb=(hb+1)%D.n_bins; slider.value=hb; draw();
    setTimeout(()=>requestAnimationFrame(tick), 650); };
  if(playing) tick();
};
legend(); draw();
</script></body></html>"""


# --------------------------------------------------------------------------- #
# top-level
# --------------------------------------------------------------------------- #
def analyze(run_dir: str, cell_m: float = 25.0, time_bin_h: int = 1,
            out_dir: str | None = None) -> dict:
    out_dir = out_dir or os.path.join(run_dir, "panel")
    os.makedirs(out_dir, exist_ok=True)
    run_name = os.path.basename(os.path.normpath(run_dir))

    centroids = load_building_centroids(run_dir)
    agg = aggregate(_iter_events(run_dir), cell_m, time_bin_h, centroids,
                    run_dt.steps_per_day(run_dir))
    n_days = (max(agg["days"]) + 1) if agg["days"] else 0
    cols = grid_rows(agg)

    # parquet
    import pyarrow as pa
    import pyarrow.parquet as pq
    grid_path = os.path.join(out_dir, "heatmap_grid.parquet")
    pq.write_table(pa.Table.from_pydict(cols, schema=_grid_schema()), grid_path)

    # report
    report = render_report(run_name, cols, agg, cell_m, time_bin_h, n_days)
    report_path = os.path.join(out_dir, "heatmap_report.md")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")

    # html(runs/<name>/heatmap.html)
    html = build_html(run_name, cols, cell_m, time_bin_h, n_days)
    html_path = os.path.join(run_dir, "heatmap.html")
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(html)

    los_rows = los_table(cols, cell_m, n_days)
    los_dist = Counter(r["los"] for r in los_rows)
    return {
        "run": run_name, "cell_m": cell_m, "time_bin_h": time_bin_h,
        "n_days": n_days, "n_cells": len(cols["cell_x"]),
        "n_pass_ev": agg["n_pass_ev"], "n_present_ev": agg["n_present_ev"],
        "n_surges": len(agg["surges"]),
        "los_dist": dict(los_dist),
        "grid": grid_path, "report": report_path, "html": html_path,
        "top_present": (los_rows[0] if los_rows else None),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="①人流ヒートマップ + ③混雑 LOS ランキング(L1 読み出し専用)")
    ap.add_argument("run_dir", help="例: runs/daily300_100d")
    ap.add_argument("--cell-m", type=float, default=25.0, help="メッシュ辺長[m](既定25)")
    ap.add_argument("--time-bin-h", type=int, default=1, help="時間ビン[h](既定1)")
    ap.add_argument("--out", default=None, help="parquet/report 出力先(既定 runs/<name>/panel)")
    args = ap.parse_args()
    if not os.path.isdir(args.run_dir):
        raise SystemExit(f"not a directory: {args.run_dir}")
    if args.time_bin_h <= 0 or (1440 % (60 * args.time_bin_h)) != 0:
        raise SystemExit("--time-bin-h は 1440 分を割り切る正整数(1,2,3,4,6,8,12,24)にしてください")
    info = analyze(args.run_dir, cell_m=args.cell_m, time_bin_h=args.time_bin_h,
                   out_dir=args.out)
    print(f"[analyze_flows_grid] {info['run']}  cell={info['cell_m']:g}m "
          f"bin={info['time_bin_h']}h  n_days={info['n_days']}")
    print(f"  セル数={info['n_cells']}  move_segment={info['n_pass_ev']}  "
          f"在圏観測={info['n_present_ev']}  crowd_surge={info['n_surges']}")
    print(f"  LOS 分布={info['los_dist']}")
    tp = info["top_present"]
    if tp:
        print(f"  最混雑: ({tp['cell_x']},{tp['cell_y']}) LOS={tp['los']} "
              f"{tp['density_ppm2']:.4g} 人/m²/日")
    print(f"  -> {info['grid']}\n  -> {info['report']}\n  -> {info['html']}")


if __name__ == "__main__":
    main()
