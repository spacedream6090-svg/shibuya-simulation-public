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

# `physics.cognitive`(第二段B。2026-08-17。docs/research/crowd-attention-physics.md 案B)。
# 対人斥力(SFM)と ORCA の**近傍選抜だけ**を「距離最近傍 k 体」から
# 「視野円錐 → 方位セクタごとに最前の 1 体 → 距離昇順 k 体」へ差し替える。
# 既定 OFF = `sfm_core.visual_neighbors` が 1 度も呼ばれない = 従来経路とバイト一致。
COGNITIVE_DEFAULTS = {
    "enabled": False,
    "neighbors": 12,      # k。既定 12 = 現行 neighbor_cap と同値の出発点
    "sectors": 16,        # 方位セクタ数(遮蔽の一次近似)。実質の近傍上限は min(k, sectors)
    "fov_deg": 360.0,     # 視野円錐の**全角** [deg]。既定 360 = 硬い円錐なし(角度重みは
                          #   SFM の異方性 λ が担う)。200 等に絞ると前方円錐のアブレーション
}

# `physics.density_far`(第二段C。同メモ 案C)。較正 far 項のペア和を密度場の連続体力へ。
# ★`physics.sfm.far_field.enabled` が ON のときだけ意味を持つ(置換する当の項が無ければ no-op)。
DENSITY_FAR_DEFAULTS = {
    "enabled": False,
    "cell_m": 1.0,        # 密度格子の一辺 [m]
    "blur": 2,            # 3×3 箱平滑の回数
    "update_every": 10,   # 格子を作り直すサブステップ周期(サンプルは毎サブステップ)
    # 同一ゾーン・同一 step でエンジンを作り直すとき(入場・退場のたびに起きる)、
    # 密度場と再構築カウンタを旧エンジンから引き継ぐか。**既定 false = 引き継がない
    # = 現行挙動と 1 バイト同一**。false のままだと入退場のたびに `update_every` の
    # 周期が 0 へ巻き戻る = 粗い周期で作り直すという設計が churn の分だけ効かなくなる。
    "carry_grid": False,
    # P4(2026-08-21): 密度格子の外延を「ゾーン polygon の外接矩形 ± この余白 [m]」との
    # 共通部分へ落とす。**既定 0.0 = クリップしない = 現行挙動と 1 バイト同一**。
    # なぜ要るか(第145 の実測): 外延は在場者の外接矩形で決まるのに、ゾーンの所有は
    # 「経路がゾーンを通り抜ける個体」に**現在地(数百 m 先)から**始まるので、
    # 38×29 m の広場に平均 117 万セル(~1 km 角)の場を作っていた。
    "clip_margin_m": 0.0,
}

# `physics.adaptive_dt`(第154 レバーB「密度適応 dt = 混雑 LOD」。2026-08-23)。
# ゾーン内の在場者数に応じて**サブステップ幅 dt を粗くする**(積分時間の総量は保存)。
# ★既定 OFF = 係数 1 = dt 固定 = 分岐を 1 度も通らない = 現行挙動と 1 バイト同一。
#
# 根拠(基本図): 密度 2 人/m² を超えると歩速は 0.5 m/s を下回る(Weidmann 1993 /
#   Older 1968 の基本図。本リポジトリの P4 較正も同じ作業点)。そこでの 1 サブステップ
#   変位は dt=0.4 s でも 0.5×0.4 = 20 cm 未満 = **粗い刻みで失う精度が最小の領域**である。
#   逆に閑散時は係数 1 のまま = コストゼロ(誰も損をしない)。
# ★正直な限界: `dt_sub` の上限 0.1 s は「dt≥0.1 は ρ≥1.5 で SFM が後退爆発する」という
#   P2 実測から来ている(zones._build_zone のバリデーション)。適応 dt はその上限を
#   **混雑時にだけ意図的に超える**ので、ON にするときは engine 別に破綻統計
#   (重なり min_gap / 壁貫通 / jump_max / accel_p99)を測り直すこと。
#   前進 Euler + v_max クリップなので 1 サブステップ変位は v_max·dt_eff で有界。
ADAPTIVE_DT_DEFAULTS = {
    "enabled": False,
    # [[在場者数 N, dt 係数], …]。「N を**超えたら**その係数」。N 昇順に正準化し、
    # 該当する中でいちばん大きい N の係数を採る(該当なし = 係数 1.0 = 現行 dt)。
    "thresholds": ((500, 2.0), (2000, 4.0)),
    # 係数を選び直すサブステップ周期(= 「塊」の長さ)。毎サブステップ判定は
    # それ自体がコストなので塊ごとに 1 回だけ引く。在場者数は決定論なので選択も決定論。
    "recheck_every": 20,
    # 適用するエンジン。**空 = 全エンジン**。`["orca"]` と書くと ORCA ゾーンだけが
    # 粗い dt を使い、SFM ゾーンは dt_sub のまま(= そのゾーンは 1 バイト同一)。
    # ★これが要る理由(第154 のベンチ実測 scripts/bench_physics_levers.py):
    #     ORCA(29×29 m・0.71 人/m²・係数 2)… 体表間 +0.066 → +0.049 m(重なりゼロを維持)
    #     SFM (93×9 m ・0.71 人/m²・係数 2)… 体表間 +0.003 → **−0.171 m(重なり)**、
    #                                         壁クリアランス +0.102 → **−0.105 m(貫通)**、
    #                                         平均速さ 0.40 → 0.62 m/s(斥力が解けきらず流れが速まる)
    #   = P2 決定の「dt≥0.1 は SFM が後退爆発」という実測が、そのまま再現する。
    #   ORCA は速度層で衝突を回避し位置層で押し戻す構成なので粗い刻みに強い。
    "engines": (),
}

ZONE_DEFAULTS = {
    "id": "",
    "polygon": (),                # [(x,y), …] 地図ローカル m。3 点以上。
    "layers": (),                 # 所属を許す垂直レイヤー(地図の node.layer)。
                                  # ★既定 () = **全レイヤー**(= 従来の幾何だけの判定と 1 バイト同値)。
                                  #   `[0]` と書くと地上ノードだけがゾーン内と見なされる。
                                  #   必要な理由(実測): ポリゴンは平面図形なので、地上の交差点を
                                  #   囲むと**その真下の地下通路ノードまで囲んでしまう**
                                  #   (実地図の交差点ポリゴン 1 件で layer -1/-2 のノードが
                                  #   3 件入った)。そのままだと地下を歩いている個体が地上の
                                  #   ゾーンに所有され、**地下で赤信号を待つ**。
                                  #   幾何(`contains`)は一切変えず、**ノードの所属判定だけ**を絞る。
    "engine": "sfm",              # "sfm" | "orca"(P2 決定: 既定 sfm・多方向交差流だけ orca)
    "dt_sub": 0.05,               # 物理サブステップ [s](P2 条件1: 0.02–0.05)
    "max_sub_steps": 12000,       # 1 世界 step で回すサブステップ上限(600s/0.05s = 12000)
                                  # ★§1.2 B5 / R7(第94バッチ OBS-U2)で「Δt に追随しない
                                  #   直書き」と指摘されていた値。**第99で導出化した**:
                                  #   ゾーン宣言が `max_sub_steps` を書いていなければ
                                  #   `derive_max_sub_steps(dt_sub, clock.step_seconds)`
                                  #   = round(step_seconds / dt_sub) が使われる。
                                  #   ここに残る 12000 は **step_seconds を渡せない呼び出し**
                                  #   (conf 直読みのツール等)のためのフォールバック兼
                                  #   「正準 Δt=10・dt_sub=0.05 での値」の宣言である。
                                  #   Δt=10 かつ dt_sub=0.05 では導出値 = 12000 = 現行と厳密同値。
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
    layers: tuple[int, ...]
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

    # ★第154 A5(docs/plans/step-time-audit.md): `node_in` のメモ化キャッシュ。
    #   **注釈を付けない**ので dataclass の field ではない(`fields()` / `asdict()` /
    #   `__init__` / `__eq__` / `__repr__` のどこにも現れない)。既定値はクラス側にだけ
    #   存在し、1 度も問い合わせていない Zone の instance `__dict__` は**空のまま**
    #   (第149 `memory` の ClassVar 既定と同型 = 属性が生えない = 状態が増えない)。
    #   中身は `(graph, frozenset(内側ノード))`。地図は静的なので純関数の答えを憶えてよい。
    _node_memo = None

    # -- 幾何 ------------------------------------------------------------- #
    def contains(self, x: float, y: float) -> bool:
        return point_in(self.polygon, x, y, self.bbox)

    def __getstate__(self):
        """pickle / deepcopy に**メモ化キャッシュを載せない**(第154 A5)。

        `_node_memo` は都市グラフへの強参照を含むので、載せると checkpoint が地図を
        まるごと抱き込む。地図は静的なので復元後に引き直せば同じ答えになる(純キャッシュ)。
        1 度も問い合わせていない Zone では `__dict__` に `_node_memo` が無いので、
        返る dict は従来の既定状態(= `self.__dict__` そのもの)と同じ内容・同じ順序
        = pickle バイト一致。
        """
        return {k: v for k, v in self.__dict__.items() if k != "_node_memo"}

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

    ★返す dict には `crossing_id` を必ず含める(actor model P2 / society/devices.py)。
      信号は「位相 = 絶対時刻の純関数」= 記憶を持たない固定スケジュール**装置**なのに、
      これまで **どの交差点の信号なのかを外から指す名前が無かった**。`SignalGate` は
      この値から安定 device_id を導く(`SignalGate(**zone.signal)` の呼び出し形は不変)。
      時刻計算には一切使わないので、既存ランの位相・現示は 1 バイトも動かない。
    """
    mode = str(spec.get("mode", "none"))
    if mode == "none":
        return None
    if mode == "explicit":
        return {"cycle_s": float(spec["cycle_s"]), "green_s": float(spec["green_s"]),
                "flash_s": float(spec.get("flash_s", 0.0)),
                "offset_s": float(spec.get("offset_s", 0.0)),
                "crossing_id": int(spec.get("crossing_id", 0))}
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
                        "offset_s": float(spec.get("offset_s", 0.0)),
                        "crossing_id": cid}
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


def build_cfg(raw, repo_root: Path | None = None,
              step_seconds: float | None = None) -> dict:
    """conf の `physics` ブロック → 正準 dict。既定は **ゾーン 0 件**(= 完全 no-op)。

    Returns: {"zones_enabled": bool, "zones": tuple[Zone, …], "perception": {...}}
    ★ `zones_enabled=false` または `zones` が空なら `zones=()`(= 一切の新経路を通らない)。

    `step_seconds`(= `clock.step_seconds`)は `max_sub_steps` **未指定**のゾーンで
    上限を Δt から導くために使う(第99・`derive_max_sub_steps`)。省略すると正準
    Δt=10 の値 12000 へ後退する = 既存の呼び出しはバイト単位で従来どおり。
    """
    raw = dict(raw or {})
    repo_root = repo_root or Path(".")
    unknown = sorted(set(raw) - {"sfm", "zones_enabled", "zones", "perception",
                                 "cognitive", "density_far", "adaptive_dt",
                                 "neighbor_cap", "separation_iters"})
    if unknown:
        raise KeyError(f"physics: 未知のキー {unknown}")
    enabled = bool(raw.get("zones_enabled", False))
    perception = _merge(PERCEPTION_DEFAULTS, raw.get("perception"))
    perception["density_radius_m"] = float(perception["density_radius_m"])
    perception["contact_gap_m"] = float(perception["contact_gap_m"])
    perception["channels"] = bool(perception["channels"])
    sfm = _build_sfm(raw.get("sfm"))
    cognitive = _build_cognitive(raw.get("cognitive"))
    density_far = _build_density_far(raw.get("density_far"))
    adaptive_dt = _build_adaptive_dt(raw.get("adaptive_dt"))
    neighbor_cap = _build_neighbor_cap(raw.get("neighbor_cap"), cognitive)
    separation_iters = _build_separation_iters(raw.get("separation_iters"))
    zones: list[Zone] = []
    if enabled:
        for spec in (raw.get("zones", ()) or ()):
            zones.append(_build_zone(dict(spec), repo_root, step_seconds,
                                     neighbor_cap, separation_iters))
    _check_disjoint(zones)
    return {"zones_enabled": enabled, "zones": tuple(zones), "perception": perception,
            "sfm": sfm, "cognitive": cognitive, "density_far": density_far,
            "adaptive_dt": adaptive_dt, "neighbor_cap": neighbor_cap,
            "separation_iters": separation_iters}


def _build_separation_iters(raw) -> int:
    """`physics.separation_iters`(第154 追補)。0 = 未指定 = ゾーン宣言のまま(既定)。

    >0 のときは**すべての ORCA ゾーン**の `orca.separation_iters`(既定 64)を置き換える。
    これは ORCA 専用の口である(SFM は接触斥力が硬く事後分離パスを持たない)。

    何を削るのか: ORCA は「速度層(半平面 LP)+ 位置層(決定論的 push-apart)」の二層で、
    位置層は重なりが消えるか上限反復に達するまで **(i<j) 辞書順の逐次 Gauss-Seidel** を回す。
    密集ではここが ORCA コストの支配項になる(第154 実測: 29×29 m・3,000 体・cap 7 で
    **ORCA 時間の ~84%**。反復上限 64 → 0 で 16.6 s → 2.6 s)。
    ★ただし削れば残留めり込みが増える(同実測 n=1,500 で 64 → 16 は体表間 −0.006 m =
      実質無害、n=3,000 では −0.188 m)。**上限に当たったかは `sep_iters_max` で監視できる**。
    ★0 は「未指定」の意味に予約してあるので、分離パスを**本当に切りたい**ときは
      ゾーン宣言側に `orca: {separation_iters: 0}` と書くこと(neighbor_cap と同じ規約)。
    """
    it = int(raw or 0)
    if it < 0:
        raise ValueError("physics.separation_iters: 0(=ゾーン宣言のまま)以上が必要")
    return it


def _build_adaptive_dt(raw) -> dict:
    """`physics.adaptive_dt` の正準化(第154 レバーB)。既定 OFF = dt 固定 = 恒等。

    thresholds は **N 昇順**へ正準化する(conf の記入順に依らない決定論)。
    係数は 1.0 未満を許さない: dt を細かくするのは「積分の総量は同じだがサブステップが
    増える」= 上限 `max_sub_steps` の意味が壊れるうえ、遅くなるだけで目的に反する。
    """
    out = _merge(ADAPTIVE_DT_DEFAULTS, raw)
    out["enabled"] = bool(out["enabled"])
    out["recheck_every"] = int(out["recheck_every"])
    eng = tuple(str(e) for e in (out["engines"] or ()))
    bad = [e for e in eng if e not in ENGINES]
    if bad:
        raise ValueError(f"physics.adaptive_dt.engines: 未知の engine {bad}(可: {ENGINES})")
    out["engines"] = eng
    pairs = []
    for item in (out["thresholds"] or ()):
        if len(tuple(item)) != 2:
            raise ValueError("physics.adaptive_dt.thresholds は [[N, 係数], …] の形")
        n, c = int(item[0]), float(item[1])
        if n < 0 or c < 1.0:
            raise ValueError("physics.adaptive_dt.thresholds: N>=0 / 係数>=1.0 が必要"
                             "(係数 1 未満 = dt を細かくする = 目的に反する)")
        pairs.append((n, c))
    pairs.sort(key=lambda p: (p[0], p[1]))
    out["thresholds"] = tuple(pairs)
    if out["enabled"] and out["recheck_every"] < 1:
        raise ValueError("physics.adaptive_dt: recheck_every>=1 が必要")
    return out


def _build_neighbor_cap(raw, cognitive: dict) -> int:
    """`physics.neighbor_cap`(第154 レバーD)。0 = 未指定 = ゾーン宣言のまま(既定)。

    >0 のときは **相互作用相手の上限を 1 点で絞る**:
      - すべてのゾーンの `neighbor_cap`(SFM の対人斥力 / ORCA の ORCA 線)を置き換える
      - `physics.cognitive` が ON なら、その k(`cognitive.neighbors`)も
        `min(neighbors, neighbor_cap)` へ絞る
    後者が要るのは、**認知的近傍 ON では `neighbor_cap` が 1 度も読まれない**から
    (sfm_core._repulsion_cognitive / orca_core.step はどちらも cog_neighbors を使う)。
    ここを揃えないと「cap を下げたのに何も起きない」= 二重 cap の罠になる。
    """
    cap = int(raw or 0)
    if cap < 0:
        raise ValueError("physics.neighbor_cap: 0(=ゾーン宣言のまま)以上が必要")
    if cap > 0 and cognitive.get("enabled"):
        cognitive["neighbors"] = min(int(cognitive["neighbors"]), cap)
    return cap


def _build_cognitive(raw) -> dict:
    """`physics.cognitive` の正準化(第二段B)。既定 = OFF = 従来経路と完全に恒等。"""
    out = _merge(COGNITIVE_DEFAULTS, raw)
    out["enabled"] = bool(out["enabled"])
    out["neighbors"] = int(out["neighbors"])
    out["sectors"] = int(out["sectors"])
    out["fov_deg"] = float(out["fov_deg"])
    if out["enabled"] and not (out["neighbors"] >= 1 and out["sectors"] >= 1
                               and 0.0 < out["fov_deg"] <= 360.0):
        raise ValueError("physics.cognitive: neighbors>=1 / sectors>=1 /"
                         " 0<fov_deg<=360 が必要")
    return out


def _build_density_far(raw) -> dict:
    """`physics.density_far` の正準化(第二段C)。既定 = OFF = far 項はペア和のまま。"""
    out = _merge(DENSITY_FAR_DEFAULTS, raw)
    out["enabled"] = bool(out["enabled"])
    out["cell_m"] = float(out["cell_m"])
    out["blur"] = int(out["blur"])
    out["update_every"] = int(out["update_every"])
    out["carry_grid"] = bool(out["carry_grid"])
    out["clip_margin_m"] = float(out["clip_margin_m"])
    if out["enabled"] and not (out["cell_m"] > 0.0 and out["blur"] >= 0
                               and out["update_every"] >= 1):
        raise ValueError("physics.density_far: cell_m>0 / blur>=0 /"
                         " update_every>=1 が必要")
    if out["clip_margin_m"] < 0.0:
        raise ValueError("physics.density_far: clip_margin_m>=0 が必要"
                         "(0=クリップしない)")
    return out


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


def derive_max_sub_steps(dt_sub: float, step_seconds: float | None) -> int:
    """`max_sub_steps` 未指定時の上限 = **1 世界 step ぶんちょうど**のサブステップ数。

    第99(物理見積 残①・OBS-U2 §1.2 B5 / R7 の是正)。physics.py の
    ``n_sub = min(max_sub_steps, max(1, round(step_seconds/dt_sub)))`` に対して、
    上限側を同じ式から導けば **上限が binding しなくなる** = Δt を変えても積分が
    黙って打ち切られない(Δt=20 分で 24000 必要なのに 12000 で止まる、が消える)。

    - ``step_seconds=None``(= Δt を知らない呼び出し)では正準値 12000 へ後退する。
    - **Δt=10 かつ dt_sub=0.05 では 600/0.05 = 12000 = 現行値と厳密に同じ**。
    - dt_sub を正準の 0.05 より細かくした宣言(例 0.02)では導出値が 12000 を超える。
      これは「1 step ぶんは必ず積む」という上限の契約を守った結果であり、旧実装が
      その条件で黙って打ち切っていた方が誤り(同じ B5 の症状の別の顔)。明示的に
      打ち切りたいときは宣言側に ``max_sub_steps`` を書けばそれが尊重される。
    """
    if step_seconds is None:
        return int(ZONE_DEFAULTS["max_sub_steps"])
    return max(1, int(round(float(step_seconds) / float(dt_sub))))


def _build_zone(spec: dict, repo_root: Path, step_seconds: float | None = None,
                neighbor_cap: int = 0, separation_iters: int = 0) -> Zone:
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
        layers=tuple(int(k) for k in (merged["layers"] or ())),
        engine=engine,
        dt_sub=dt_sub,
        # 明示宣言があればそれを尊重し、無ければ Δt から導く(第99)。
        # ★ `merged` ではなく `spec` を見るのは、_merge が既定 12000 を必ず埋めてしまい
        #   「書かれていない」と「12000 と書いた」が区別できなくなるため。
        max_sub_steps=(int(spec["max_sub_steps"]) if "max_sub_steps" in spec
                       else derive_max_sub_steps(dt_sub, step_seconds)),
        arrive_radius_m=float(merged["arrive_radius_m"]),
        # `physics.neighbor_cap`(>0)はゾーン宣言より優先する = 全ゾーン 1 点で絞る
        # (既定 0 では宣言どおり = 1 バイト同一)。
        neighbor_cap=(int(neighbor_cap) if int(neighbor_cap) > 0
                      else int(merged["neighbor_cap"])),
        v_max_factor=float(merged["v_max_factor"]),
        walls=load_walls(walls_spec, repo_root),
        walkable_area_m2=load_walkable_area(walkable_spec, poly, repo_root),
        signal=load_signal(signal_spec, repo_root) or {},
        sfm=_merge(ZONE_DEFAULTS["sfm"], merged["sfm"]),
        # `physics.separation_iters`(>0)はゾーン宣言より優先(既定 0 では宣言どおり)。
        orca=_orca_spec(_merge(ZONE_DEFAULTS["orca"], merged["orca"]),
                        separation_iters),
        gate=_merge(GATE_DEFAULTS, merged["gate"]),
        bbox=polygon_bbox(poly),
    )


def _orca_spec(orca: dict, separation_iters: int) -> dict:
    """ORCA 宣言に `physics.separation_iters`(>0)を上書きする。0 = 宣言のまま = 同一。"""
    if int(separation_iters) > 0:
        orca = dict(orca)
        orca["separation_iters"] = int(separation_iters)
    return orca


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
def _node_in_uncached(zone: Zone, graph, node: str) -> bool:
    """`node_in` の生の判定(第154 A5 のメモ化前の本体そのまま。逐語で保存する)。

    テストが「メモ化あり / なし」を全ノードで突合するための参照実装でもある。
    """
    d = graph.nodes[node]
    if not zone.contains(float(d["x"]), float(d["y"])):
        return False
    if zone.layers and int(d.get("layer", 0)) not in zone.layers:
        return False
    return True


def _inside_memo(zone: Zone, graph) -> frozenset:
    """(zone, graph) → 内側ノードの frozenset を Zone インスタンスに憶える(遅延・純関数)。

    ★第154 A5(docs/plans/step-time-audit.md §3): `node_in` は
      `physics._run_zone` から **在街かつ経路持ちの全個体 × 経路長 × ゾーン数**だけ
      呼ばれるのに、毎回 `graph.nodes[node]` の辞書引き + `Zone.contains` の ray casting を
      やり直していた(キャッシュ皆無)。地図は静的 = `node_in` は純関数なので、
      答えの集合(`inside_nodes` が既に返していたもの)を 1 度だけ作って憶える。

    - キーに graph を**同一性**で持つ(1 run 1 グラフ。テストが別グラフを渡したら作り直す)。
    - 書き込みは `object.__setattr__`(frozen dataclass)。`_node_memo` は field ではないので
      `fields()` / `asdict()` / `__eq__` / `__repr__` は 1 つも変わらない。
    - `__getstate__` が除くので pickle / checkpoint / deepcopy には載らない。
    - 乱数ゼロ・sim 非参照(本 module の「幾何とデータだけ」の約束を保つ)。
    """
    memo = zone._node_memo                      # 既定 None はクラス属性(instance は空)
    if memo is not None and memo[0] is graph:
        return memo[1]
    ins = frozenset(n for n in graph.nodes if _node_in_uncached(zone, graph, n))
    object.__setattr__(zone, "_node_memo", (graph, ins))
    return ins


def node_in(zone: Zone, graph, node: str) -> bool:
    """グラフノードがゾーンに**所属する**か(幾何 + 垂直レイヤー)。

    `zone.layers` が空(既定)なら幾何だけ = `contains` と完全に同値(従来どおり)。
    非空なら `graph.nodes[node]["layer"]` がその集合に含まれることも要求する。
    ★ `Zone.contains(x, y)` 側(= 物理座標の内外判定)は**一切変えない**。
      物理はもともと 2 次元平面で走るので、変えるべきは「どのノードを縫い付けるか」だけ。

    第154 A5: 判定はゾーン別 frozenset へメモ化する(地図は静的 = 純関数)。**返り値は
    `_node_in_uncached` と全ノードで一致**する(tests/test_zone_node_in_memo.py が
    実地図の全ノードで突合)。グラフに無いノードは従来どおり `KeyError`
    (`graph.nodes[node]` が投げていたもの)を投げる = 誤りを黙って False へ倒さない。
    """
    memo = zone._node_memo                      # 命中時は関数呼び 1 本ぶんも惜しむ(最内)
    ins = memo[1] if (memo is not None and memo[0] is graph) \
        else _inside_memo(zone, graph)
    if node in ins:
        return True
    if node in graph.nodes:
        return False
    raise KeyError(node)       # 未知ノードは従来(`graph.nodes[node]`)と同じ例外


def inside_nodes(zone: Zone, graph) -> tuple[str, ...]:
    """ゾーンの内側にあるグラフノード(id 昇順)。"""
    return tuple(sorted(_inside_memo(zone, graph)))


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
    ins = [node_in(zone, graph, n) for n in seq]
    try:
        first_in = ins.index(True)
    except ValueError:
        return None
    for j in range(first_in + 1, len(seq)):
        if not ins[j]:
            return seq[:j + 1], seq[j + 1:]
    return None


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
