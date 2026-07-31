"""中立 3D シーン → Unreal Engine 向け軌跡 (バッチE / UE 可視化)。

使い方:
  python scripts/export_ue.py runs/<name> [--offset X Y Z] [--heading DEG]
                                          [--no-yflip] [--scale 100] [--csv]
                                          [--binary [--no-json]]
生成物:  runs/<name>/scene3d/
  sim_ue.json    — sim(local-m, ENU, Z-up 右手系, m)を Unreal(cm, Z-up, 左手系)へ
                   *一度だけ* 変換した再生用データ。位置・移動ポリライン・車トラフィックは
                   すべて UE のワールド座標(uu=cm)で格納済み。
  sim_ue.csv     — --csv 指定時。(step, agent_id, ux, uy, uz, state) のフラット表。
  sim_ue.bin / sim_ue_meta.json — --binary 指定時。同じ内容の量子化型付きバイナリ
                   (int16 × 5uu = 0.05m 相当)。実測 684B/agent-step の JSON に対して
                   8B/agent-step = **1/85**。1万体×10日の JSON は ≈9.8GB で
                   `FJsonSerializer` に載らない(docs/research/dt-integration-deep.md §3B.12)。
                   UE 側は `FFileHelper::LoadFileToArray` → `TArray<int16>` で読む。
                   既定 OFF=このフラグ無しでは sim_ue.json は従来とバイト同一。
  --no-json      — --binary 併用時のみ。sim_ue.json を書かない(巨大ラン向け)。

設計方針(なぜ別ファイルか):
  export_3d.py の scene.json/tracks.json 契約は *変更しない*(読み取り専用)。
  local-m→UE の座標変換・`w`(屋内/範囲外/睡眠)デコード・階高→高さ換算という
  「厄介なロジック」を Python 側に一本化し、UE の C++/Blueprint 側は
  「配列を読んで線形補間して ISM を更新するだけ」に保つ(SimReplayActor_DESIGN.md)。
  → 変換を疑う必要があるのはこのファイルだけ。UE 側は素直な再生器で済む。

座標系(要点。詳細は viz/unreal/README_UE.md §座標合わせ):
  sim  : X=east, Y=north, Z=up  [m]  右手系  原点=スクランブル交差点(35.6595,139.70062)
  UE   : X=前, Y=右, Z=上        [cm=uu] 左手系  1uu=1cm
  基本写像(mirror を避けるため north を反転 = y_flip):
      ux = east ,  uy = -north ,  uz = up        (m)
  → heading[deg] だけ鉛直(UE-Z)まわりに回転 → ×scale(=100) → + offset[uu]
  offset の既定=(0,0,0)。PLATEAU を SDK でインポートする際に
  「基準座標系からのオフセット値」をスクランブル交差点の平面直角座標
  (EPSG:6677 第9系: 北距 X=-37768.576 m, 東距 Y=-12015.952 m)に設定すれば、
  PLATEAU 原点(UEの0,0,0)= スクランブル交差点 = sim 原点 となり offset 不要で一致する。
  heading と y_flip は「初回のみエディタで実測合わせ」する前提(README トラブルシュート)。
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

M_TO_UU = 100.0            # 1 m = 100 cm = 100 uu (UE 既定)
FLOOR_HEIGHT = 3.5         # tracks/scene meta が無い場合のフォールバック

# スクランブル交差点(35.65950,139.70062)の JGD2011 平面直角座標系 第9系(EPSG:6677)。
# 値は Kawase(2011) の Transverse Mercator 級数で算出(GSI 準拠)。
# 平面直角は X=北距(northing), Y=東距(easting) の順(日本の慣行)であることに注意。
ORIGIN_EPSG6677 = {"northing_m": -37768.576, "easting_m": -12015.952}


# ------------------------------------------------------------------ 入出力
def _load_mod(name: str):
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_export3d():
    return _load_mod("export_3d")


def _ensure_scene(run_dir: Path) -> tuple[dict, dict]:
    scene_dir = run_dir / "scene3d"
    scene_p = scene_dir / "scene.json"
    tracks_p = scene_dir / "tracks.json"
    bin_p = scene_dir / "tracks.bin"
    if not (scene_p.exists() and (tracks_p.exists() or bin_p.exists())):
        _load_export3d().export_run(run_dir)
    scene = json.loads(scene_p.read_text(encoding="utf-8"))
    # tracks.json が無いラン(export_3d --no-tracks-json)は tracks.bin を復号して読む
    tracks = _load_mod("tracks_bin").load_tracks(scene_dir)
    if tracks is None:
        raise SystemExit(f"[export_ue] {scene_dir} に tracks.json / tracks.bin が無い")
    return scene, tracks


# ------------------------------------------------------------------ 変換
class Transform:
    """local-m ENU(m) → UE ワールド(uu=cm)。純アフィン(スケール+鉛直回転+平行移動)。"""

    def __init__(self, scale=M_TO_UU, offset=(0.0, 0.0, 0.0),
                 heading_deg=0.0, y_flip=True):
        self.scale = float(scale)
        self.offset = tuple(float(v) for v in offset)
        self.heading = math.radians(float(heading_deg))
        self.y_flip = bool(y_flip)
        self._c = math.cos(self.heading)
        self._s = math.sin(self.heading)

    def ground(self, east_m, north_m):
        """地表 (east,north)[m] → (ux,uy)[uu]。z は別に扱う。"""
        nx = -north_m if self.y_flip else north_m
        ex = east_m
        rx = ex * self._c - nx * self._s
        ry = ex * self._s + nx * self._c
        return (rx * self.scale + self.offset[0],
                ry * self.scale + self.offset[1])

    def up(self, up_m):
        return up_m * self.scale + self.offset[2]

    def as_meta(self):
        return {
            "units": "cm", "up_axis": "Z", "handedness": "left(UE)",
            "scale_m_to_uu": self.scale,
            "offset_uu": list(self.offset),
            "heading_deg": math.degrees(self.heading),
            "y_flip": self.y_flip,
            "origin_latlon": [35.6595, 139.70062],
            "origin_epsg6677": ORIGIN_EPSG6677,
        }


# ------------------------------------------------------------------ 変換本体
def build_sim_ue(scene: dict, tracks: dict, tf: Transform) -> dict:
    fh_m = tracks.get("meta", {}).get("floor_height", FLOOR_HEIGHT)
    fh_uu = fh_m * tf.scale
    NS = tracks["meta"]["nSteps"]
    ids = tracks["ids"]
    P = tracks["positions"]
    M = tracks["moves"]
    TR = tracks["traffic"]

    def up_uu(w: float) -> float:
        if w >= 1000:
            floor = int((w - 1000) % 100)
            return max((floor - 0.5) * fh_uu, 0.8 * tf.scale)
        return 1.2 * tf.scale  # 路上に立つ人の腰高 ≈ 1.2 m

    positions = []   # [step][i] = [ux, uy, uz, state]
    moves = []       # [step][i] = [mode, [[ux,uy],...]] or null
    traffic = []     # [step] = [ [[ux,uy],[ux,uy]], ... ]
    for s in range(NS):
        row = []
        for i in range(len(ids)):
            x, y, w = P[s][i]
            ux, uy = tf.ground(x, y)
            row.append([round(ux, 1), round(uy, 1), round(up_uu(w), 1), int(w)])
        positions.append(row)

        mrow = []
        for i in range(len(ids)):
            mv = M[s][i] if s < len(M) else None
            if mv:
                mode, pts = mv[0], mv[1]
                upts = [[round(v, 1) for v in tf.ground(px, py)] for px, py in pts]
                mrow.append([mode, upts])
            else:
                mrow.append(None)
        moves.append(mrow)

        segs = TR[s].get("segs", []) if s < len(TR) else []
        usegs = [[[round(v, 1) for v in tf.ground(px, py)] for px, py in seg]
                 for seg in segs]
        traffic.append(usegs)

    meta = dict(tf.as_meta())
    meta.update({
        "nSteps": NS,
        "step_minutes": tracks["meta"].get("step_minutes", 10),
        "start_min": tracks["meta"].get("start_min", 0),
        "floor_height_uu": fh_uu,
        "sim_min": tracks.get("sim_min", []),
        "car_ground_uu": round(0.9 * tf.scale, 1),   # 車体を置く路面高
        "note": "positions/moves/traffic は既に UE ワールド(uu). state: 0=路上 -1=範囲外 -2=睡眠 >=1000=屋内",
    })
    return {
        "meta": meta,
        "agents": tracks["agents"],
        "ids": ids,
        "positions": positions,
        "moves": moves,
        "traffic": traffic,
    }


# ------------------------------------------------------------------ 書き出し
def write_csv(sim_ue: dict, path: Path) -> None:
    ids = sim_ue["ids"]
    with path.open("w", newline="", encoding="utf-8") as f:
        wtr = csv.writer(f)
        wtr.writerow(["step", "agent_id", "ux", "uy", "uz", "state"])
        for s, row in enumerate(sim_ue["positions"]):
            for i, (ux, uy, uz, st) in enumerate(row):
                wtr.writerow([s, ids[i], ux, uy, uz, st])


def export_run(run_dir: Path, tf: Transform, want_csv: bool = False,
               binary: bool = False, write_json: bool = True) -> dict:
    if not binary and not write_json:
        raise SystemExit("[export_ue] --no-json は --binary と併用する")
    run_dir = Path(run_dir)
    scene, tracks = _ensure_scene(run_dir)
    sim_ue = build_sim_ue(scene, tracks, tf)
    out_dir = run_dir / "scene3d"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {"n_agents": len(sim_ue["ids"]), "n_steps": sim_ue["meta"]["nSteps"]}
    if write_json:
        jp = out_dir / "sim_ue.json"
        jp.write_text(json.dumps(sim_ue, ensure_ascii=False, separators=(",", ":")),
                      encoding="utf-8")
        res["json"] = jp
    if binary:                       # 追加専用: 既定 OFF=sim_ue.json は従来とバイト同一
        tb = _load_mod("tracks_bin")
        blob, header = tb.encode_sim_ue(sim_ue)
        bp = out_dir / "sim_ue.bin"
        mp = out_dir / "sim_ue_meta.json"
        bp.write_bytes(blob)
        mp.write_text(json.dumps(header, ensure_ascii=False, separators=(",", ":")),
                      encoding="utf-8")
        res["bin"] = bp
        res["meta"] = mp
    if want_csv:
        cp = out_dir / "sim_ue.csv"
        write_csv(sim_ue, cp)
        res["csv"] = cp
    return res


# ------------------------------------------------------------------ CLI
def parse_args(argv):
    ap = argparse.ArgumentParser(description="scene3d → Unreal 再生用 sim_ue.json")
    ap.add_argument("run_dir", help="runs/<name>")
    ap.add_argument("--offset", nargs=3, type=float, default=[0.0, 0.0, 0.0],
                    metavar=("X", "Y", "Z"), help="UE ワールドオフセット[uu]")
    ap.add_argument("--heading", type=float, default=0.0,
                    help="鉛直まわり回転[deg](sim-north を PLATEAU/UE に合わせる)")
    ap.add_argument("--no-yflip", action="store_true",
                    help="north 反転を無効化(鏡像になる場合の切替)")
    ap.add_argument("--scale", type=float, default=M_TO_UU, help="m→uu(既定100)")
    ap.add_argument("--csv", action="store_true", help="sim_ue.csv も出力")
    ap.add_argument("--binary", action="store_true",
                    help="sim_ue.bin + sim_ue_meta.json(量子化型付きバイナリ)も出力")
    ap.add_argument("--no-json", action="store_true",
                    help="--binary 併用時のみ: sim_ue.json を書かない")
    return ap.parse_args(argv)


def main(argv: list) -> int:
    if not argv:
        print(__doc__)
        return 1
    args = parse_args(argv)
    tf = Transform(scale=args.scale, offset=tuple(args.offset),
                   heading_deg=args.heading, y_flip=not args.no_yflip)
    res = export_run(Path(args.run_dir).resolve(), tf, want_csv=args.csv,
                     binary=args.binary, write_json=not args.no_json)
    for k in ("json", "bin", "meta", "csv"):
        if k in res:
            p = res[k]
            try:
                shown = p.relative_to(REPO_ROOT)
            except ValueError:
                shown = p
            print(f"  {shown}  ({p.stat().st_size/1024:.1f} KB)")
    print(f"  agents={res['n_agents']}  steps={res['n_steps']}  "
          f"offset={args.offset}  heading={args.heading}  "
          f"yflip={not args.no_yflip}  scale={args.scale}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
