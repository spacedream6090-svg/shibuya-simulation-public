"""在館(occupancy)観測列のサイドカー生成(第58バッチ B8・事後スクリプト)。

runs/<run>/l1_events.parquet(と屋内サイドカー indoor_tracks_samples.parquet の有無)から
【建物・階・step ごとの在館人数】の時系列 parquet を構築する。ビューア(viz/make_viewer.py)は
occupancy.parquet があれば「🏙 在館」タブに埋め込む(無ければ本スクリプトの実行を案内)。

方針(既存 analyze_*.py / build_*.py の流儀):
  - argparse・決定論・乱数ゼロ・pyarrow(pandas/duckdb 不使用)・UTF-8。
  - コア地図・L1 は読むだけ(改変しない)。
  - 在館状態は build_data(viewer)と同じ carry-forward:
      enter_building / floor_move / space_move → (建物, 階)を保持
      exit_building / exit_area / enter_area   → 屋外(在館から外れる)
    初回の enter_building だけで以後の step も在館扱い(状態を持ち越す)。
  - 屋内トラッキング(space_move / indoor_tracks)が無いランは【建物粒度のみ】に正直に縮退する
    (階は enter_building の階だけ=intra-building の移動は追えない旨を meta に明記)。

使い方:
    python scripts/build_occupancy.py runs/<run>
    python scripts/build_occupancy.py runs/<run> --out runs/<run>/occupancy.parquet

出力:
    runs/<run>/occupancy.parquet       columns = building(str), floor(int), step(int), n(int)
    runs/<run>/occupancy_meta.json     {indoor, n_steps, n_buildings, n_rows, generated_by}
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- 本体(純関数=テスト可能)
def occupancy_rows(events: list, n_steps: int) -> tuple[list[dict], bool]:
    """L1 イベント列 → 在館系列 rows と「屋内トラッキング有無」。

    rows = [{building, floor, step, n}, ...](n>0 のセルのみ・(building,floor,step) 昇順)。
    第2要素 has_space_move = space_move イベントが1件でもあれば True(=階内トラッキング在り)。
    """
    by_step: dict[int, list[dict]] = defaultdict(list)
    has_space_move = False
    for e in events:
        if e["kind"] == "space_move":
            has_space_move = True
        by_step[int(e["step"])].append(e)

    # agent id → (building, floor) or None(屋外)。carry-forward。
    state: dict[int, tuple[str, int] | None] = {}
    # (building, floor, step) → 在館数
    cells: dict[tuple[str, int, int], int] = defaultdict(int)

    def _bf(p: dict, keep: tuple[str, int] | None) -> tuple[str, int] | None:
        bld = p.get("building")
        if not bld:
            return keep
        floor = p.get("floor", keep[1] if keep else 1)
        try:
            floor = int(floor)
        except (TypeError, ValueError):
            floor = 1
        return (str(bld), floor)

    for step in range(n_steps):
        for e in by_step.get(step, []):
            aid = e["agent_id"]
            if aid is None or aid < 0:
                continue
            kind = e["kind"]
            p = e.get("payload")
            p = json.loads(p) if isinstance(p, str) and p else (p or {})
            cur = state.get(aid)
            if kind in ("enter_building", "space_move"):
                state[aid] = _bf(p, cur)
            elif kind == "floor_move":
                state[aid] = _bf(p, cur)
            elif kind in ("exit_building", "exit_area", "enter_area"):
                state[aid] = None
        # この step のスナップショット
        for aid, bf in state.items():
            if bf is None:
                continue
            cells[(bf[0], bf[1], step)] += 1

    rows = [{"building": b, "floor": f, "step": s, "n": n}
            for (b, f, s), n in cells.items() if n > 0]
    rows.sort(key=lambda r: (r["building"], r["floor"], r["step"]))
    return rows, has_space_move


def build_occupancy(run_dir: Path) -> tuple[list[dict], dict]:
    """run ディレクトリから occupancy rows と meta を作る(ファイルは書かない)。"""
    l1 = run_dir / "l1_events.parquet"
    if not l1.exists():
        raise FileNotFoundError(f"{l1} が見つかりません(先にランを実行してください)")
    events = pq.read_table(l1).to_pylist()
    n_steps = (max((int(e["step"]) for e in events), default=-1) + 1)
    rows, has_sm = occupancy_rows(events, n_steps)
    indoor = has_sm or (run_dir / "indoor_tracks_samples.parquet").exists()
    n_bld = len({r["building"] for r in rows})
    meta = {"indoor": bool(indoor), "n_steps": int(n_steps),
            "n_buildings": int(n_bld), "n_rows": len(rows),
            "generated_by": "scripts/build_occupancy.py",
            "note": ("space_move / indoor_tracks 由来の階内トラッキングあり"
                     if indoor else
                     "屋内トラッキング無し=建物粒度のみ(階は enter_building の階のみ・"
                     "intra-building の移動は追えない=正直に縮退)")}
    return rows, meta


def write_occupancy(run_dir: Path, out_path: Path) -> dict:
    """occupancy.parquet + occupancy_meta.json を書き出し、meta を返す。"""
    rows, meta = build_occupancy(run_dir)
    schema = pa.schema([("building", pa.string()), ("floor", pa.int32()),
                        ("step", pa.int32()), ("n", pa.int32())])
    cols = {"building": [r["building"] for r in rows],
            "floor": [r["floor"] for r in rows],
            "step": [r["step"] for r in rows],
            "n": [r["n"] for r in rows]}
    pq.write_table(pa.table(cols, schema=schema), out_path)
    (run_dir / "occupancy_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return meta


# --------------------------------------------------------------------------- CLI
def main() -> int:
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(
        description="在館(occupancy)観測列サイドカー生成(L1 は不変・事後処理)")
    ap.add_argument("run_dir", help="runs/<run>(l1_events.parquet を含む)")
    ap.add_argument("--out", default=None,
                    help="出力先(既定=<run>/occupancy.parquet)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    out_path = Path(args.out) if args.out else run_dir / "occupancy.parquet"
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path

    try:
        meta = write_occupancy(run_dir, out_path)
    except FileNotFoundError as ex:
        print(f"error: {ex}", file=sys.stderr)
        return 1

    print(f"written: {out_path}")
    print(f"  建物={meta['n_buildings']}  行={meta['n_rows']}  "
          f"steps={meta['n_steps']}  屋内階={'yes' if meta['indoor'] else 'no(建物粒度)'}")
    print(f"  {meta['note']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
