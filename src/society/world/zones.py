"""物理ゾーンの宣言・幾何・ゲート(竹-4 = P3 境界縫合の**純粋な部分**)。

正典: docs/plans/source/physics-instructions.md Part P3 /
      docs/plans/cognition-physics-plan.md §4「選定後: P3 境界縫合」/
      docs/research/physics-engine-selection.md ★P2 決定(2026-08-02)の条件 4〜6 /
      docs/plans/highfidelity-3d-physics-plan.md §2 竹-4。

本 module が持つもの(**世界を 1 バイトも変えない純関数と型だけ**)
------------------------------------------------------------------
  - `build_cfg()`  … conf の `physics` ブロックの正準化(既定 = ゾーン 0 件)
  - `Zone`         … ポリゴン + エンジン種別 + dt_sub + 壁 + 歩行可能面 + 信号 + ゲート設定
  - `point_in()`   … 点のゾーン内判定(ray casting・境界は内側扱い・決定論)
  - `gates_of()`   … ゾーンの**ゲート**(= ゾーン外にあってゾーン内ノードへ隣接するグラフノード)
  - `route_span()` … エージェントの残り経路のうち「ゾーンに入って出るまで」の区間
  - `project_on_path()` … 物理座標 → グラフ (node, next, edge_offset) への復帰射影

実行(積分・ゲート通過・観測)は `src/society/physics.py` に置く。本 module は
`sim` を一切参照しない(= 幾何とデータだけの層)。

なぜ「ゲート経由のみ」なのか(P2 決定 条件 6)
--------------------------------------------
領域分割型ハイブリッドの既知の失敗様式は**境界アーティファクト**(移管の瞬間に位置が飛ぶ・
速度が不連続になる・境界で詰まる)である。歩行者マルチスケール結合の代表例 TransiTUM
(Biedermann, Kielar, Handel & Borrmann, *Towards TransiTUM*, Transportation Research
Procedia 2 (2014) 495–500)は、
  - **排他所有**: 「If a pedestrian entity is put into another model, it is **deleted from
    its current scale and added in the new scale** at the nearest possible position.」
    = 同一時刻に 2 つのモデルへ属さない(delete + add の原子的移管)。
  - **移管はステップ境界だけ**(粗い側の 1 ステップの終わりにだけ所有権が動く)。
  - **置けなければ移管しない**(受け皿が空くまで延期する。詰まりを力ずくで解かない)。
  - **緩和帯(relaxation zone)**を設け、「変換で生じたアーティファクトはそこで減衰する。
    したがって **緩和帯で採った統計は慎重に扱う**(= 解析対象から外す)」。
を定めている。本実装はこの 4 点をそのまま採る:
  排他所有  … `agent._phys_zone` は単一値。ゾーン間の直接移籍は存在せず、
              いったんグラフ世界へ返してからでないと次のゾーンへ入れない。
  移管の時刻 … 流入・流出の確定は世界 tick(10 分 step)の境界のみ(P3(2) 時間整合)。
  延期      … guarded ゲート(入口が空いていなければ入れない。待たせる)。
  緩和帯    … `gate.band_m` のゲート帯。境界連続性指標は**ゲート帯と内部を分けて**出す。

★ TransiTUM の重複帯幅 `r_trans = v_max · Δt_coarse` はここでは採れない: 粗い側が
  メソスコピック場ではなく**グラフ経路**で、Δt_coarse = 600 s のため幅が 780 m になる
  (ゾーンより大きい)。本実装は「経路がゾーンへ入る step の頭で所有を移す」ので、
  そもそも粗い側がゾーン内部を進むことがない = 重複帯を持つ必要がない。
  代わりに `band_m` は**観測のための緩和帯**として使う(上の 4 点目)。

R1(既定 OFF)
--------------
`physics.zones_enabled: false`(既定)または `physics.zones: []`(既定)のとき
`build_cfg()` は `zones=()` を返し、`physics.py` の全関数が即 return する。
= 新経路ゼロ・新イベントゼロ・新 stream ゼロ・L1 バイト一致。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# 既定値(すべて conf から上書き可能。ここが**唯一の**既定の源)
# --------------------------------------------------------------------------- #
GATE_DEFAULTS = {
    # 入口の占有チェック余裕 [m](guarded)。ベンチ `_gate_admit` の min_gap と同義。
    "min_gap_m": 0.10,
    # ゲート帯(= 緩和帯)の幅 [m]。この帯の中の指標は内部と分けて出す。
    "band_m": 3.0,
    # 入場を待てる世界 step の上限。超えたら安全弁として強制的にグラフへ返す。
    "max_hold_steps": 3,
    # ゾーン滞在の上限 step。超えたら強制的に出口へ返す(詰まりの安全弁)。
    "max_zone_steps": 6,
    # グラフ復帰時に許す位置の跳び [m]。超えたら強制復帰扱いで記録する(監視用)。
    "handover_jump_max_m": 20.0,
}

PERCEPTION_DEFAULTS = {
    # 局所密度を測る半径 [m](Perception.body.local_density の定義)
    "density_radius_m": 2.0,
    # 接触判定の余裕 [m](体表間がこれ以下 = 押し合っている)
    "contact_gap_m": 0.05,
    # ★将来の発火チャンネル接続(ext.crowd 系)。**配線だけ**用意し既定 OFF。
    "channels": False,
}

# `physics.sfm`(竹-3 の宣言ブロック + P4-2/P4-3 の較正ブロック)。
# ★P4-2/P4-3(2026-08-05)で **far_field / v_of_s / wall だけが `physics.py` から読まれる**
#   ようになった。wall_A / wall_B / wall_range / wall_hash_cell / noise / noise_seed は
#   従来どおり「sfm_core の既定値の記録」で、本層は読まない(zone.sfm 側が正典)。
#   既定値は 3 機能とも **現行挙動と完全に恒等**:
#     far_field.enabled=false / v_of_s.enabled=false / wall.{a,b} = sfm_core の既定値
#   = `physics.py` は従来どおり `sfm_core.Crowd` を**同じ引数で**構築する(golden 無風)。
SFM_DEFAULTS = {
    "wall_A": 2000.0,
    "wall_B": 0.08,
    "wall_range": 2.0,
    "wall_hash_cell": 2.35,
    "noise": 0.0,
    "noise_seed": 20260802,
    # P4-2: 長距離 social 項(VISSIM 2 項構造の欠落側)。既定 OFF。
    "far_field": {"enabled": False, "a2": 0.119, "b2": 1.890,
                  "cutoff_factor": 2.5, "taper_m": 1.0},
    # P4-3: Tordeux 型の間隔ベース希望速度 V(s)=min{v0, max{0,(s−l)/T}}。既定 OFF。
    #   T / l は P4-3 の較正値(calib_p43_results.json の 0.4823492 / 0.2965060 を丸めた値)。
    "v_of_s": {"enabled": False, "T": 0.482, "l": 0.297},
    # P4-3: 壁斥力。**既定 = sfm_core.WALL_A_DEFAULT / WALL_B_DEFAULT と同値**。
    "wall": {"a": 2000.0, "b": 0.08},
}

ZONE_DEFAULTS = {
    "id": "",
    "polygon": (),                # [(x,y), …] 地図ローカル m。3 点以上。
    "engine": "sfm",              # "sfm" | "orca"(P2 決定: 既定 sfm・多方向交差流だけ orca)
    "dt_sub": 0.05,               # 物理サブステップ [s](P2 条件1: 0.02–0.05)
    "max_sub_steps": 12000,       # 1 世界 step で回すサブステップ上限(600s/0.05s = 12000)
                                  # ★§1.2 B5 / R7(第94バッチ OBS-U2): これは **正準 Δt=10 の
                                  #   step 長 600s** から導いた直書きであり、Δt に追随しない。
                                  #   physics.py:331 は n_sub = min(max_sub_steps,
                                  #   step_seconds/dt_sub) なので、Δt<10(1 分 = 1200 sub)では
                                  #   上限に届かず無害だが、**Δt>10 では積分が黙って打ち切られる**
                                  #   (Δt=20 で 24000 必要なのに 12000 で停止 = 半分の時間しか
                                  #   進まない)。Δt 掃引を上方向にもやるなら、ここを
                                  #   clock.step_seconds / dt_sub から導くよう直すこと。
                                  #   OBS-U2(Δt=1)では踏まないので第94では据え置く。
    "arrive_radius_m": 1.0,       # 出口ゲートへの到達判定半径 [m]
    "neighbor_cap": 12,           # 対人相互作用の近傍上限(SFM/ORCA 共通)
    "v_max_factor": 1.3,          # 最高速度 = 希望速度 × これ(Helbing2000)
    "walls": {"mode": "none", "segments": (), "path": "", "layers": ()},
    "walkable": {"mode": "none", "path": ""},
    "signal": {"mode": "none", "path": "", "crossing_id": 0,
               "cycle_s": 0.0, "green_s": 0.0, "flash_s": 0.0, "offset_s": 0.0},
    "sfm": {"noise": 0.0, "wall_range_m": 2.0},
    "orca": {"tau": 2.0, "tau_obst": 2.0, "neighbor_dist_m": 10.0,
             "wall_range_m": 2.0, "pref_noise": 0.05,
             "radius_margin_m": 0.05, "separation_iters": 64},
    "gate": dict(GATE_DEFAULTS),
}

ENGINES = ("sfm", "orca")
WALL_MODES = ("none", "inline", "layered_json")
WALKABLE_MODES = ("none", "polygons_json")
SIGNAL_MODES = ("none", "explicit", "table")


# --------------------------------------------------------------------------- #
# Zone
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Zone:
    """1 ゾーンの宣言(不変)。幾何はすべて地図ローカル m(X=east, Y=north)。"""

    id: str
    polygon: tuple[tuple[float, float], ...]
    engine: str
    dt_sub: float
    max_sub_steps: int
    arrive_radius_m: float
    neighbor_cap: int
    v_max_factor: float
    walls: tuple[tuple[tuple[float, float], tuple[float, float]], ...]
    walkable_area_m2: float
    signal: dict
    sfm: dict
    orca: dict
    gate: dict
    bbox: tuple[float, float, float, float] = field(default=(0.0, 0.0, 0.0, 0.0))

    # -- 幾何 ------------------------------------------------------------- #
    def contains(self, x: float, y: float) -> bool:
        return point_in(self.polygon, x, y, self.bbox)

    def area_m2(self) -> float:
        return polygon_area(self.polygon)

    def near_gate(self, x: float, y: float, gate_xy) -> bool:
        """ゲート帯(緩和帯)の中か。gate_xy は (x,y) の列。"""
        band = float(self.gate["band_m"])
        for gx, gy in gate_xy:
            if (x - gx) ** 2 + (y - gy) ** 2 <= band * band:
                return True
        return False


# --------------------------------------------------------------------------- #
# 幾何のプリミティブ(乱数ゼロ・決定論)
# --------------------------------------------------------------------------- #
def polygon_bbox(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return (min(xs), min(ys), max(xs), max(ys))


def polygon_area(poly) -> float:
    """符号なし面積 [m²](靴紐公式)。"""
    n = len(poly)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5


def point_in(poly, x: float, y: float, bbox=None) -> bool:
    """点がポリゴンの内側か(ray casting・水平右向き)。**境界上は内側**として扱う。

    決定論: 浮動小数の比較のみ・乱数なし・頂点順に依存しない(同一入力 → 同一 bool)。
    退化(3 点未満)は常に False。
    """
    n = len(poly)
    if n < 3:
        return False
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        if x < x0 or x > x1 or y < y0 or y > y1:
            return False
    # 境界上(辺への距離が極小)は内側
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        if _point_seg_dist2(x, y, ax, ay, bx, by) <= 1e-18:
            return True
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            xt = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < xt:
                inside = not inside
        j = i
    return inside


def _point_seg_dist2(px, py, ax, ay, bx, by) -> float:
    ex, ey = bx - ax, by - ay
    len2 = ex * ex + ey * ey
    if len2 <= 1e-18:
        return (px - ax) ** 2 + (py - ay) ** 2
    t = ((px - ax) * ex + (py - ay) * ey) / len2
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    cx, cy = ax + t * ex, ay + t * ey
    return (px - cx) ** 2 + (py - cy) ** 2


def polygons_overlap(a: Zone, b: Zone) -> bool:
    """ゾーン非重複の実務的な検査(P2 決定 条件 6「ゾーン非重複 + ゲート経由のみ」)。

    厳密な多角形交差判定はしない(過剰)。**どちらかの頂点または重心が相手の内側にある**
    ことを重複とみなす。辺だけが交差する星形のような病的形状は検出しない = 正直な限界。
    """
    for x, y in a.polygon:
        if b.contains(x, y):
            return True
    for x, y in b.polygon:
        if a.contains(x, y):
            return True
    ca = _centroid(a.polygon)
    cb = _centroid(b.polygon)
    return b.contains(*ca) or a.contains(*cb)


def _centroid(poly):
    n = len(poly)
    return (sum(p[0] for p in poly) / n, sum(p[1] for p in poly) / n)


# --------------------------------------------------------------------------- #
# 壁・歩行可能面の読み込み(データ由来。ここでも乱数ゼロ)
# --------------------------------------------------------------------------- #
def load_walls(spec: dict, repo_root: Path):
    """壁線分 [((x1,y1),(x2,y2)), …] を返す。mode="none" は空 tuple。

    mode:
      none          … 壁なし(開放平面)
      inline        … conf 直書きの線分列 `segments: [[x1,y1,x2,y2], …]`
      layered_json  … 層つき線分 JSON(`{"wall_segments":[{"xy":[x1,y1,x2,y2],"layer":k}, …]}`)。
                      `layers: [k, …]` で層を選ぶ(空 = 全層)。地下街 LOD4.1 の抽出物が該当。
    ★ 並びは入力順のまま(dedup も並べ替えもしない)= 同一ファイル → 同一配列。
    """
    mode = str(spec.get("mode", "none"))
    if mode == "none":
        return ()
    if mode == "inline":
        out = []
        for s in spec.get("segments", ()) or ():
            v = [float(t) for t in s]
            if len(v) != 4:
                raise ValueError(f"physics.zones[].walls.segments は [x1,y1,x2,y2]: {s}")
            out.append(((v[0], v[1]), (v[2], v[3])))
        return tuple(out)
    if mode == "layered_json":
        path = Path(str(spec.get("path", "")))
        if not path.is_absolute():
            path = repo_root / path
        data = json.loads(path.read_text(encoding="utf-8"))
        want = {int(k) for k in (spec.get("layers", ()) or ())}
        out = []
        for seg in data.get("wall_segments", ()):
            if want and int(seg.get("layer", 0)) not in want:
                continue
            v = [float(t) for t in seg["xy"]]
            out.append(((v[0], v[1]), (v[2], v[3])))
        return tuple(out)
    raise ValueError(f"未知の physics.zones[].walls.mode: {mode}(可: {WALL_MODES})")


def load_walkable_area(spec: dict, zone_poly, repo_root: Path) -> float:
    """歩行可能面の実面積 [m²](任意)。mode="none" ならゾーン面積そのもの。

    mode="polygons_json" は「クラス別ポリゴンの面積表を持つ JSON」を読み、
    **ゾーン内の歩行系面積の按分**ではなく `area_m2_by_class` の歩行系合計に対する
    ゾーン面積比の下限として使う…のではなく、**素直にゾーン面積を返す**。
    ★正直な限界: 道路 LOD3 のポリゴン実体は npz 側にあり、ゾーン切り抜きの厳密計算は
      竹-5(歩道幅係数の原点)の仕事なので**ここではやらない**。本関数は
      「歩行可能面をゾーン宣言から参照できる口」を開けるところまで(密度の分母は
      ゾーン面積のまま = 捏造しない)。読めた JSON の被覆率だけを記録に残す。
    """
    if str(spec.get("mode", "none")) == "none":
        return polygon_area(zone_poly)
    path = Path(str(spec.get("path", "")))
    if not path.is_absolute():
        path = repo_root / path
    data = json.loads(path.read_text(encoding="utf-8"))
    if "area_m2_by_class" not in data:
        raise ValueError(f"歩行可能面 JSON に area_m2_by_class が無い: {path}")
    return polygon_area(zone_poly)


def load_signal(spec: dict, repo_root: Path):
    """信号(SignalGate 用の周期パラメータ)を返す。mode="none" なら None。

    mode:
      explicit … conf 直書きの cycle_s / green_s / flash_s / offset_s
      table    … 交差点表 JSON(`{"crossings":[{"id":…, "cycle_s":…, "green_s":…,
                 "flash_s":…}, …]}`)から `crossing_id` の 1 件を引く
    """
    mode = str(spec.get("mode", "none"))
    if mode == "none":
        return None
    if mode == "explicit":
        return {"cycle_s": float(spec["cycle_s"]), "green_s": float(spec["green_s"]),
                "flash_s": float(spec.get("flash_s", 0.0)),
                "offset_s": float(spec.get("offset_s", 0.0))}
    if mode == "table":
        path = Path(str(spec.get("path", "")))
        if not path.is_absolute():
            path = repo_root / path
        data = json.loads(path.read_text(encoding="utf-8"))
        cid = int(spec.get("crossing_id", 0))
        for row in data.get("crossings", ()):
            if int(row.get("id", -1)) == cid:
                return {"cycle_s": float(row["cycle_s"]),
                        "green_s": float(row["green_s"]),
                        "flash_s": float(row.get("flash_s", 0.0)),
                        "offset_s": float(spec.get("offset_s", 0.0))}
        raise ValueError(f"交差点表に id={cid} が無い: {path}")
    raise ValueError(f"未知の physics.zones[].signal.mode: {mode}(可: {SIGNAL_MODES})")


# --------------------------------------------------------------------------- #
# cfg 正準化
# --------------------------------------------------------------------------- #
def _merge(defaults: dict, raw) -> dict:
    out = dict(defaults)
    for k, v in dict(raw or {}).items():
        if k not in out:
            raise KeyError(f"physics: 未知のキー {k}(既知: {sorted(out)})")
        out[k] = v
    return out


def build_cfg(raw, repo_root: Path | None = None) -> dict:
    """conf の `physics` ブロック → 正準 dict。既定は **ゾーン 0 件**(= 完全 no-op)。

    Returns: {"zones_enabled": bool, "zones": tuple[Zone, …], "perception": {...}}
    ★ `zones_enabled=false` または `zones` が空なら `zones=()`(= 一切の新経路を通らない)。
    """
    raw = dict(raw or {})
    repo_root = repo_root or Path(".")
    unknown = sorted(set(raw) - {"sfm", "zones_enabled", "zones", "perception"})
    if unknown:
        raise KeyError(f"physics: 未知のキー {unknown}")
    enabled = bool(raw.get("zones_enabled", False))
    perception = _merge(PERCEPTION_DEFAULTS, raw.get("perception"))
    perception["density_radius_m"] = float(perception["density_radius_m"])
    perception["contact_gap_m"] = float(perception["contact_gap_m"])
    perception["channels"] = bool(perception["channels"])
    sfm = _build_sfm(raw.get("sfm"))
    zones: list[Zone] = []
    if enabled:
        for spec in (raw.get("zones", ()) or ()):
            zones.append(_build_zone(dict(spec), repo_root))
    _check_disjoint(zones)
    return {"zones_enabled": enabled, "zones": tuple(zones), "perception": perception,
            "sfm": sfm}


def _build_sfm(raw) -> dict:
    """`physics.sfm` の正準化(P4-2/P4-3)。既定は**現行挙動と完全に恒等**。"""
    out = _merge(SFM_DEFAULTS, raw)
    ff = _merge(SFM_DEFAULTS["far_field"], out["far_field"])
    ff["enabled"] = bool(ff["enabled"])
    for k in ("a2", "b2", "cutoff_factor", "taper_m"):
        ff[k] = float(ff[k])
    if ff["enabled"] and not (ff["a2"] > 0.0 and ff["b2"] > 0.0
                              and ff["cutoff_factor"] > 0.0 and ff["taper_m"] >= 0.0):
        raise ValueError("physics.sfm.far_field: a2>0 / b2>0 / cutoff_factor>0 /"
                         " taper_m>=0 が必要")
    vs = _merge(SFM_DEFAULTS["v_of_s"], out["v_of_s"])
    vs["enabled"] = bool(vs["enabled"])
    vs["T"] = float(vs["T"])
    vs["l"] = float(vs["l"])
    if vs["enabled"] and not (vs["T"] > 0.0 and vs["l"] >= 0.0):
        raise ValueError("physics.sfm.v_of_s: T>0 / l>=0 が必要")
    wl = _merge(SFM_DEFAULTS["wall"], out["wall"])
    wl["a"] = float(wl["a"])
    wl["b"] = float(wl["b"])
    if not (wl["a"] >= 0.0 and wl["b"] > 0.0):
        raise ValueError("physics.sfm.wall: a>=0 / b>0 が必要")
    out["far_field"], out["v_of_s"], out["wall"] = ff, vs, wl
    return out


def _build_zone(spec: dict, repo_root: Path) -> Zone:
    merged = _merge(ZONE_DEFAULTS, spec)
    zid = str(merged["id"]).strip()
    if not zid:
        raise ValueError("physics.zones[].id は必須(空でない文字列)")
    poly = tuple((float(p[0]), float(p[1])) for p in (merged["polygon"] or ()))
    if len(poly) < 3:
        raise ValueError(f"zone {zid}: polygon は 3 点以上")
    engine = str(merged["engine"])
    if engine not in ENGINES:
        raise ValueError(f"zone {zid}: 未知の engine {engine}(可: {ENGINES})")
    dt_sub = float(merged["dt_sub"])
    if not (0.0 < dt_sub <= 0.1):
        # P2 条件1: dt≥0.1 は ρ≥1.5 で後退爆発(SFM)。上限は 0.1 で切る。
        raise ValueError(f"zone {zid}: dt_sub は (0, 0.1] の範囲(P2 決定 条件1)")
    walls_spec = _merge(ZONE_DEFAULTS["walls"], merged["walls"])
    walkable_spec = _merge(ZONE_DEFAULTS["walkable"], merged["walkable"])
    signal_spec = _merge(ZONE_DEFAULTS["signal"], merged["signal"])
    return Zone(
        id=zid,
        polygon=poly,
        engine=engine,
        dt_sub=dt_sub,
        max_sub_steps=int(merged["max_sub_steps"]),
        arrive_radius_m=float(merged["arrive_radius_m"]),
        neighbor_cap=int(merged["neighbor_cap"]),
        v_max_factor=float(merged["v_max_factor"]),
        walls=load_walls(walls_spec, repo_root),
        walkable_area_m2=load_walkable_area(walkable_spec, poly, repo_root),
        signal=load_signal(signal_spec, repo_root) or {},
        sfm=_merge(ZONE_DEFAULTS["sfm"], merged["sfm"]),
        orca=_merge(ZONE_DEFAULTS["orca"], merged["orca"]),
        gate=_merge(GATE_DEFAULTS, merged["gate"]),
        bbox=polygon_bbox(poly),
    )


def _check_disjoint(zones) -> None:
    """ゾーン非重複(P2 決定 条件 6)。重複したら**構築時に落とす**(ランを始めさせない)。"""
    for i in range(len(zones)):
        for j in range(i + 1, len(zones)):
            if polygons_overlap(zones[i], zones[j]):
                raise ValueError(
                    f"physics.zones が重複している: {zones[i].id} / {zones[j].id}。"
                    " ゾーン間の直接移籍はゲート経由のみ(境界アーティファクト回避)= 非重複が前提。")


# --------------------------------------------------------------------------- #
# ゲート(グラフ ⇄ ゾーン の唯一の出入口)
# --------------------------------------------------------------------------- #
def inside_nodes(zone: Zone, graph) -> tuple[str, ...]:
    """ゾーンの内側にあるグラフノード(id 昇順)。"""
    return tuple(sorted(n for n, d in graph.nodes(data=True)
                        if zone.contains(float(d["x"]), float(d["y"]))))


def gates_of(zone: Zone, graph) -> tuple[str, ...]:
    """ゾーンの**ゲートノード** = ゾーン外にあり、ゾーン内ノードへ隣接するノード(id 昇順)。

    ゲート id は文字列 `"{zone.id}:{node}"`(L1 の zone_gate イベントに載る)。
    ★グラフ上の出入口を列挙するだけ = 乱数ゼロ・都市地図だけの純関数。
    """
    ins = set(inside_nodes(zone, graph))
    out: set[str] = set()
    for n in ins:
        for m in graph.neighbors(n):
            if m not in ins:
                out.add(m)
    return tuple(sorted(out))


def gate_id(zone: Zone, node: str) -> str:
    return f"{zone.id}:{node}"


def route_span(zone: Zone, graph, node: str, route):
    """`node` から `route` を辿ったときの「ゾーンに入って出るまで」の区間を返す。

    Returns: (path, rest) or None
      path … [node, …, exit_node](exit_node は**ゾーン外**の最初のノード)
      rest … exit_node より先の残り経路(そのまま agent.route の後半になる)
    None を返す条件(= 物理が所有しない):
      - 経路がゾーン内ノードを 1 つも通らない
      - ゾーンに入ったあと**ゾーン外へ出るノードが残り経路に無い**
        (= 目的地がゾーン内。到着処理はグラフ側の責務なので所有しない = 正直な適用範囲)
    """
    seq = [node] + list(route or ())
    ins = [zone.contains(*_xy(graph, n)) for n in seq]
    try:
        first_in = ins.index(True)
    except ValueError:
        return None
    for j in range(first_in + 1, len(seq)):
        if not ins[j]:
            return seq[:j + 1], seq[j + 1:]
    return None


def _xy(graph, node):
    d = graph.nodes[node]
    return float(d["x"]), float(d["y"])


# --------------------------------------------------------------------------- #
# グラフ復帰の射影(物理座標 → (node, next_node, edge_offset))
# --------------------------------------------------------------------------- #
def project_on_path(city, path, x: float, y: float):
    """点 (x,y) を経路 `path`(ノード列)の折れ線へ射影する。

    Returns: (i, edge_offset, px, py, dist_m)
      i           … path[i] → path[i+1] のエッジ上に落ちた
      edge_offset … `city.xy_along(path[i], path[i+1], edge_offset)` にそのまま渡せる値
                    (world.mod の cost_scale が居るときはコスト長へ換算済み)
      (px,py)     … 射影点の座標
      dist_m      … 元の点から射影点までの距離 [m](= グラフ復帰時の位置の跳び)
    タイブレークは **経路の早い区間が勝つ**(同距離なら手前)= 決定論。
    """
    best = None
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        data = city.graph.edges[u, v]
        geom = list(data["geometry"])
        if data["u0"] != u:
            geom = list(reversed(geom))
        acc = 0.0
        for a, b in zip(geom, geom[1:]):
            seg = math.hypot(b[0] - a[0], b[1] - a[1])
            if seg <= 0.0:
                continue
            t = ((x - a[0]) * (b[0] - a[0]) + (y - a[1]) * (b[1] - a[1])) / (seg * seg)
            t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
            px, py = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
            d = math.hypot(x - px, y - py)
            if best is None or d < best[4] - 1e-12:
                off_geom = acc + seg * t
                cs = data.get("cost_scale")
                off = off_geom * cs if cs is not None else off_geom
                off = max(0.0, min(off, float(data["length"])))
                best = (i, off, px, py, d)
            acc += seg
    return best
