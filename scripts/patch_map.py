"""地図POIパッチ(第9バッチ 2026-07-07)。OSM 抽出で欠けた実在 POI を地図 JSON に補完する。

docs/research/poi-reality-check.md §4 の提案(WOMB・ホテル・ハチ公像・区役所本庁舎など)を
data/poi_patch_shibuya.json から読み、各 POI をローカル m 座標の**最寄り道路ノード**へ決定論で
紐付けて、新しい地図ファイルを生成する。**元の地図は上書きしない**(新ファイル生成のみ)。

使い方:
    python scripts/patch_map.py                          # 既定: v3 → data/shibuya_osm_v3p.json
    python scripts/patch_map.py --base data/shibuya_osm_wide_20250401.json \
        --patch data/poi_patch_shibuya.json --out data/shibuya_osm_wide_v7.json

規則:
- 地図 bbox(ノードの座標範囲+margin)外の POI はスキップして警告(端のノードへ引き寄せると
  位置が嘘になるため)。bbox 外項目は広域地図(wide)採用時に入れる。
- id は "p_patch_NN"(連番・決定論)。既存 POI と同名がある場合はスキップ(二重追加防止)。
- 項目が "subcat" を持つときだけ、地図 v8 の第2階層としてそのまま載せる(無ければキー無し)。
- meta.description に補完の来歴を追記(監査可能性)。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def nearest_node(nodes: list[dict], x: float, y: float) -> dict:
    """(x, y) の最寄りノード(ユークリッド2乗距離。同距離は id 順=決定論)。"""
    return min(nodes, key=lambda n: ((n["x"] - x) ** 2 + (n["y"] - y) ** 2,
                                     str(n["id"])))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/shibuya_osm.json")
    ap.add_argument("--patch", default="data/poi_patch_shibuya.json")
    ap.add_argument("--out", default="data/shibuya_osm_v3p.json")
    ap.add_argument("--margin", type=float, default=60.0,
                    help="bbox 判定の許容はみ出し(m)")
    args = ap.parse_args()

    base_path = REPO_ROOT / args.base if not Path(args.base).is_absolute() else Path(args.base)
    patch_path = REPO_ROOT / args.patch if not Path(args.patch).is_absolute() else Path(args.patch)
    out_path = REPO_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)

    city = load(base_path)
    patch = load(patch_path)
    nodes = city["nodes"]
    xs = [n["x"] for n in nodes]
    ys = [n["y"] for n in nodes]
    lo_x, hi_x = min(xs) - args.margin, max(xs) + args.margin
    lo_y, hi_y = min(ys) - args.margin, max(ys) + args.margin
    existing = {p["name"] for p in city["pois"]}

    added, skipped = [], []
    seq = 0
    for entry in patch["pois"]:
        name = str(entry["name"]).strip()
        x, y = float(entry["x"]), float(entry["y"])
        if name in existing:
            skipped.append((name, "同名POIが既存"))
            continue
        if not (lo_x <= x <= hi_x and lo_y <= y <= hi_y):
            skipped.append((name, f"bbox外 ({x:.0f},{y:.0f})"))
            continue
        node = nearest_node(nodes, x, y)
        dist = ((node["x"] - x) ** 2 + (node["y"] - y) ** 2) ** 0.5
        poi = {"id": f"p_patch_{seq:02d}", "name": name,
               "cat": str(entry["cat"])}
        # v8: 地図側の第2階層(subcat)をパッチからも通す。**キーがある項目だけ**なので、
        #     subcat を書いていない従来の patch JSON では出力が 1 バイトも変わらない。
        if entry.get("subcat"):
            poi["subcat"] = str(entry["subcat"])
        poi.update({"x": round(float(node["x"]), 1),
                    "y": round(float(node["y"]), 1),
                    "node": node["id"]})
        seq += 1
        city["pois"].append(poi)
        existing.add(name)
        added.append((name, entry["cat"], node["id"], round(dist, 1)))

    meta = city.setdefault("meta", {})
    meta["description"] = (str(meta.get("description", "")) +
                           f" / POIパッチ適用({patch_path.name}: +{len(added)}件。"
                           "poi-reality-check.md §4)").strip(" /")
    out_path.write_text(json.dumps(city, ensure_ascii=False), encoding="utf-8")

    print(f"base: {base_path.name}  patch: {patch_path.name}  → {out_path.name}")
    print(f"追加 {len(added)} 件:")
    for name, cat, node, dist in added:
        print(f"  + {name} [{cat}] node={node} (提案座標から {dist}m)")
    if skipped:
        print(f"スキップ {len(skipped)} 件:")
        for name, why in skipped:
            print(f"  - {name}: {why}")


if __name__ == "__main__":
    main()
