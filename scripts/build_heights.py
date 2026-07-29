"""OSM 建物 ID → PLATEAU 実測高さ [m] のコンパクトな表を作る(A1 バッチ 2026-07-29)。

シム側(`src/society/world/map.py` の `CityMap.attach_heights`)が `world.heights.enabled: true`
のときだけ読む静的データ。既定 OFF なのでこの生成物が無くてもシムは従来どおり動く。

使い方:
  python scripts/build_heights.py [--match data/plateau/plateau_match.json]
                                  [--index data/plateau/plateau_index.json]
                                  [--out   data/building_heights_shibuya.json]
                                  [--round-m 0.1]

入力:
  - `plateau_match.json`(`scripts/match_plateau.py` の出力): OSM 建物 id → {gml_id, iou, dist_m}。
  - `plateau_index.json`(`scripts/plateau_extract.py` の出力): gml_id ごとに
    `height`(= zmax - ground0)と `base`(= zmin - ground0)を持つ(`plateau_extract.py:844-845`)。

★ 高さの定義: **実高さ h = height - base**。
  `plateau_index.json` の `height` は「ground0(原点の地表標高)を基準にした建物頂部の標高」であって
  建物そのものの高さではない。渋谷は起伏があるので坂の上の建物ほど base が大きくなり、height を
  そのまま使うと高さを過大評価する。実測(3,531 棟)でも
    corr(height - base, levels x 3.5) = 0.681 > corr(height, levels x 3.5) = 0.628
  で、height - base の方が階数由来の推定高さと整合する。
  (`scripts/export_3d.py` は表示用に ground0 基準の頂部標高が要るので height をそのまま使う。
   用途が違うだけで矛盾ではない。)

出力(JSON・キー昇順・丸め round_m):
  {"meta": {...生成元・件数・パラメータ...},
   "heights": {"<osm建物id>": {"h": <実高さm>, "src": "plateau"}, ...}}

決定論: 乱数を一切使わない。同じ入力からは同じバイト列が出る(キー昇順・固定丸め)。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SCHEMA = "building-heights-1.0"
DEFAULT_ROUND_M = 0.1


def _rel(p: Path) -> str:
    """リポジトリ相対の posix 表記(OS をまたいでも生成物が同一バイトになるように)。"""
    q = p.relative_to(REPO_ROOT) if p.is_relative_to(REPO_ROOT) else p
    return q.as_posix()


def build_table(matches: dict, index_buildings: list, round_m: float = DEFAULT_ROUND_M
                ) -> tuple[dict, dict]:
    """(heights, counts) を返す純関数(I/O なし・乱数なし)。

    matches:          {osm_id: {"gml_id": str, ...}}(plateau_match.json の "matches")
    index_buildings:  [{"gml_id": str, "height": float, "base": float, ...}, ...]
    round_m:          丸め幅 [m](0.1 なら小数第1位)。0 以下は丸めない。

    高さが 0 以下になる異常レコードは採用しない(捏造も切り上げもしない)。件数は counts に残す。
    """
    by_gml = {str(b["gml_id"]): b for b in index_buildings}
    ndigits = None
    if round_m and round_m > 0:
        # 0.1 -> 1 桁 / 0.01 -> 2 桁。10 のべき以外の丸め幅は使わない(決定論の単純さを優先)。
        ndigits = max(0, len(f"{round_m:.10f}".rstrip("0").split(".")[1]))
    heights: dict[str, dict] = {}
    n_no_index = n_nonpositive = 0
    for osm_id in sorted(matches):
        rec = matches[osm_id] or {}
        b = by_gml.get(str(rec.get("gml_id", "")))
        if b is None:
            n_no_index += 1
            continue
        h = float(b.get("height", 0.0)) - float(b.get("base", 0.0))
        if h <= 0.0:
            n_nonpositive += 1
            continue
        heights[str(osm_id)] = {"h": (round(h, ndigits) if ndigits is not None else h),
                                "src": "plateau"}
    counts = {"matches_in": len(matches), "index_buildings": len(by_gml),
              "written": len(heights), "skipped_no_index": n_no_index,
              "skipped_nonpositive": n_nonpositive}
    return heights, counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="OSM 建物 ID → PLATEAU 実高さ表の生成")
    ap.add_argument("--match", type=Path,
                    default=REPO_ROOT / "data" / "plateau" / "plateau_match.json")
    ap.add_argument("--index", type=Path,
                    default=REPO_ROOT / "data" / "plateau" / "plateau_index.json")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "data" / "building_heights_shibuya.json")
    ap.add_argument("--round-m", type=float, default=DEFAULT_ROUND_M,
                    help="丸め幅 [m](既定 0.1)")
    args = ap.parse_args(argv)

    for p in (args.match, args.index):
        if not p.exists():
            print(f"[build_heights] 入力が見つからない: {p}\n"
                  "  先に scripts/plateau_extract.py と scripts/match_plateau.py を実行する。",
                  file=sys.stderr)
            return 2

    match_doc = json.loads(args.match.read_text(encoding="utf-8"))
    index_doc = json.loads(args.index.read_text(encoding="utf-8"))
    matches = match_doc.get("matches", {}) or {}
    heights, counts = build_table(matches, index_doc.get("buildings", []) or [],
                                  round_m=args.round_m)

    hs = sorted(v["h"] for v in heights.values())
    stats = ({"min_m": hs[0], "median_m": hs[len(hs) // 2], "max_m": hs[-1],
              "mean_m": round(sum(hs) / len(hs), 3)} if hs else {})
    doc = {
        "meta": {
            "schema": SCHEMA,
            "generated_by": "scripts/build_heights.py",
            "sources": {"match": _rel(args.match), "index": _rel(args.index)},
            "height_def": "h = plateau_index.height - plateau_index.base "
                          "(= zmax - zmin。ground0 基準の頂部標高ではなく建物そのものの高さ)",
            "params": {"round_m": args.round_m},
            "match_params": match_doc.get("params", {}),
            "index_crs": index_doc.get("crs"),
            "index_ground0": index_doc.get("ground0"),
            "counts": counts,
            "height_stats": stats,
        },
        "heights": heights,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, ensure_ascii=False, sort_keys=True, indent=1),
                        encoding="utf-8")
    size_kb = args.out.stat().st_size / 1024.0
    print(f"[build_heights] 出力 {args.out} ({size_kb:.1f} KB)")
    print(f"[build_heights] counts={counts}")
    print(f"[build_heights] height_stats={stats}")
    return 0


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
