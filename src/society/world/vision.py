"""擬似視覚(壁による遮蔽)。既定 OFF の本番トグル。

知覚(perception.hearers_of)の距離・同一階フィルタを通ったペアに対し、間仕切り壁
による視線(LOS)遮蔽を追加する層。建物内では建物 id + 階から決定論的に手続き生成した
「推定間取り」の壁セグメント集合と、2エージェント間の線分の交差判定で可視/不可視を決める。

間取りの手続き生成は viz/make_viewer.py の floorLayout(b, f) と同系列(同じ FNV ハッシュ
seed・mulberry32 乱数・長辺沿いの通路・重み付き列分割・中央コア)で、ビューアが描く区画と
「同種」の部屋割りを再現する。ビューアは壁を描かない(区画を塗るだけ)ため、部屋間の
開口部(ドア)は本層で追加する。完全一致の実態と限界は docs/research/vision-los.md に明記。

決定論: 乱数は使わず(壁は building id + floor の hash から生成)、キャッシュは (id, floor)。
no-fingerprint: この層は座標・幾何しか見ない(内部状態の因子名を参照しない)。
"""
from __future__ import annotations

import math

_U32 = 0xFFFFFFFF
_DOOR_HALF = 0.8   # 開口部(ドア)の半幅 m。部屋の通路側の辺の中央に開ける。


# ---- 決定論ハッシュ / PRNG(make_viewer.py の _hash/_rng/_cols と同一系列)----
def _hash(s) -> int:
    """FNV-1a 32bit(JS: h=2166136261; h^=c; h=Math.imul(h,16777619))。"""
    h = 2166136261
    for ch in str(s):
        h ^= ord(ch)
        h = (h * 16777619) & _U32
    return h & _U32


def _imul(x: int, y: int) -> int:
    """Math.imul: 32bit 乗算の下位32bit。符号は下位ビットに影響しないので uint32 で足りる。"""
    return ((x & _U32) * (y & _U32)) & _U32


def _rng(seed: int):
    """mulberry32(JS 実装と同一のビット列)。全て uint32 領域で扱う。"""
    a = seed & _U32

    def nxt() -> float:
        nonlocal a
        a = (a + 0x6D2B79F5) & _U32
        t = _imul(a ^ (a >> 15), 1 | a)
        t = ((t + _imul(t ^ (t >> 7), 61 | t)) & _U32) ^ t
        return ((t ^ (t >> 14)) & _U32) / 4294967296.0

    return nxt


def _cols(a0: float, a1: float, n: int, rng) -> list:
    """[a0,a1] を n 区画へ重み付き分割(重み=0.7+rng()*0.7)。JS _cols と同一。"""
    if n <= 0:
        return []
    weights = []
    total = 0.0
    for _ in range(n):
        v = 0.7 + rng() * 0.7
        weights.append(v)
        total += v
    out = []
    p = a0
    for v in weights:
        seg = (a1 - a0) * v / total
        out.append((p, p + seg))
        p += seg
    return out


# 各用途の区画候補プールの「長さ」だけを持つ(プール内は全て相異なる語なので、
# 重複判定は index 一致と等価 → 実際の語は幾何に不要)。make_viewer.py _POOL より。
_POOL_LEN = {
    "fashion": 6, "beauty": 5, "food": 6, "restaurant": 6, "lifestyle": 6,
    "office": 5, "shop": 4, "hall": 3, "theatre": 3, "hotel": 3, "station": 4,
    "park": 3, "nightlife": 3, "service": 3, "attraction": 3, "education": 3,
    "generic": 3,
}


def _kind_use(kind: str) -> str:
    """建物 kind → 区画プールの用途(POI/guide 無しのフォールバック。JS の三項と同順)。"""
    if kind == "office":
        return "office"
    if kind == "station":
        return "station"
    if kind == "retail":
        return "shop"
    return "generic"


def building_layout(building: dict, floor: int, city=None,
                    n_override: int | None = None) -> dict | None:
    """建物 building・階 floor の推定間取り(決定論)。

    返り値: {bbox:(x0,y0,x1,y1), corridor, core:(X0,Y0,X1,Y1), zones:[(x0,y0,x1,y1)...],
             horiz, nA, nB}。footprint が不正なら None。

    n_override: 区画数の外部指定(既定 None=完全従来動作=乱数消費順不変)。指定時は POI 経路
      (n_poi>0 分岐)と同型に、cols 前に乱数を1つも消費せず n=min(12, n_override) で分割する
      (呼び出し側が floor_layouts の shops/zone_mix 等で区画数を持つときの決定論 seam)。
    """
    fp = building.get("footprint") or []
    if len(fp) < 3:
        return None
    xs = [float(p[0]) for p in fp]
    ys = [float(p[1]) for p in fp]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return None
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0

    bid = building.get("id") or building.get("name") or "b"
    seed = (_hash(bid) + (int(floor) + 50) * 2654435761) & _U32
    rng = _rng(seed)

    # 区画数 n の決定(override > POI > 用途プールから決定論生成)。override/POI 経路は
    # どちらも cols 前に乱数を消費しない(=消費順を分岐間で一致させる)。
    if n_override is not None:
        n = min(12, int(n_override))             # 外部指定経路: cols 前に乱数を消費しない
    else:
        n_poi = 0
        if city is not None and building.get("id"):
            try:
                n_poi = len(city.pois_in_building(building["id"], floor))
            except Exception:
                n_poi = 0
        if n_poi > 0:
            n = min(10, n_poi)                   # POI 経路: cols 前に乱数を消費しない
        else:
            use = _kind_use(building.get("kind", "generic"))
            pool_n = _POOL_LEN.get(use, _POOL_LEN["generic"])
            n = 2 + int(rng() * min(4, pool_n - 1))
            used: set[int] = set()               # 重複回避ループ(乱数消費数を JS と一致させる)
            for _ in range(n):
                g = 0
                while True:
                    idx = int(rng() * pool_n)
                    cont = (idx in used) and (g < 8)
                    g += 1
                    if not cont:
                        break
                used.add(idx)
    if n <= 0:
        n = 1

    horiz = w >= h
    band = (h if horiz else w) * 0.09
    n_a = -(-n // 2)                              # ceil(n/2)
    n_b = n - n_a
    zones = []
    if horiz:
        yc0, yc1 = cy - band, cy + band
        for c in _cols(x0, x1, n_a, rng):        # 側A = 上(通路の上)
            zones.append((c[0], yc1, c[1], y1))
        for c in _cols(x0, x1, n_b, rng):        # 側B = 下
            zones.append((c[0], y0, c[1], yc0))
        corridor = (x0, yc0, x1, yc1)
    else:
        xc0, xc1 = cx - band, cx + band
        for c in _cols(y0, y1, n_a, rng):        # 側A = 右
            zones.append((xc1, c[0], x1, c[1]))
        for c in _cols(y0, y1, n_b, rng):        # 側B = 左
            zones.append((x0, c[0], xc0, c[1]))
        corridor = (xc0, y0, xc1, y1)
    cs = min(w, h) * 0.13
    core = (cx - cs / 2, cy - cs / 2, cx + cs / 2, cy + cs / 2)
    return {"bbox": (x0, y0, x1, y1), "corridor": corridor, "core": core,
            "zones": zones, "horiz": horiz, "nA": n_a, "nB": n_b}


def _wall_with_door_h(a: float, b: float, y: float) -> list:
    """水平線 y 上、x=a..b の壁(中央に開口部)。返り: 線分 [(p1,p2)...] 最大2本。"""
    if b <= a:
        return []
    xm = (a + b) / 2.0
    hd = min(_DOOR_HALF, (b - a) * 0.25)
    segs = []
    if xm - hd > a:
        segs.append(((a, y), (xm - hd, y)))
    if b > xm + hd:
        segs.append(((xm + hd, y), (b, y)))
    return segs


def _wall_with_door_v(a: float, b: float, x: float) -> list:
    """垂直線 x 上、y=a..b の壁(中央に開口部)。"""
    if b <= a:
        return []
    ym = (a + b) / 2.0
    hd = min(_DOOR_HALF, (b - a) * 0.25)
    segs = []
    if ym - hd > a:
        segs.append(((x, a), (x, ym - hd)))
    if b > ym + hd:
        segs.append(((x, ym + hd), (x, b)))
    return segs


def _walls_from_layout(lay: dict) -> list:
    """間取り → 壁セグメント集合。部屋は通路側にドア、部屋間の間仕切りは無開口、コアは無開口。"""
    x0, y0, x1, y1 = lay["bbox"]
    zones = lay["zones"]
    n_a, n_b = lay["nA"], lay["nB"]
    walls: list = []
    if lay["horiz"]:
        _, yc0, _, yc1 = lay["corridor"]         # (x0, yc0, x1, yc1)
        top = zones[:n_a]
        bot = zones[n_a:n_a + n_b]
        for (zx0, _zy0, zx1, _zy1) in top:       # 部屋⇄通路(上)= ドア付き壁
            walls += _wall_with_door_h(zx0, zx1, yc1)
        for (zx0, _zy0, zx1, _zy1) in bot:       # 部屋⇄通路(下)
            walls += _wall_with_door_h(zx0, zx1, yc0)
        for i in range(len(top) - 1):            # 隣接部屋の間仕切り(上、無開口)
            xb = top[i][2]
            walls.append(((xb, yc1), (xb, y1)))
        for i in range(len(bot) - 1):            # 隣接部屋の間仕切り(下)
            xb = bot[i][2]
            walls.append(((xb, y0), (xb, yc0)))
    else:
        xc0, _yc0, xc1, _yc1 = lay["corridor"]   # (xc0, y0, xc1, y1)
        right = zones[:n_a]
        left = zones[n_a:n_a + n_b]
        for (_zx0, zy0, _zx1, zy1) in right:     # 部屋⇄通路(右)
            walls += _wall_with_door_v(zy0, zy1, xc1)
        for (_zx0, zy0, _zx1, zy1) in left:      # 部屋⇄通路(左)
            walls += _wall_with_door_v(zy0, zy1, xc0)
        for i in range(len(right) - 1):
            yb = right[i][3]
            walls.append(((xc1, yb), (x1, yb)))
        for i in range(len(left) - 1):
            yb = left[i][3]
            walls.append(((x0, yb), (xc0, yb)))
    # コア(EV/階段)= 無開口の矩形4辺
    cx0, cy0, cx1, cy1 = lay["core"]
    walls += [((cx0, cy0), (cx1, cy0)), ((cx0, cy1), (cx1, cy1)),
              ((cx0, cy0), (cx0, cy1)), ((cx1, cy0), (cx1, cy1))]
    return walls


def building_walls(building: dict, floor: int, city=None) -> list:
    """建物・階の壁セグメント集合(決定論)。footprint が不正なら []。"""
    lay = building_layout(building, floor, city)
    if lay is None:
        return []
    return _walls_from_layout(lay)


def _door_span(a: float, b: float) -> tuple[float, float]:
    """区画の通路側辺 [a,b] に開ける開口部の (中心, 半幅)。

    壁生成 _wall_with_door_h/_v の中央開口と同一式(中心=(a+b)/2、半幅=min(_DOOR_HALF,
    幅*0.25))。この関数は doors_from_layout 専用の新規補助で、既存の壁生成関数の出力・
    乱数消費は一切変えない(vision.py の既存挙動バイト不変を保つための非破壊追加)。"""
    return (a + b) / 2.0, min(_DOOR_HALF, (b - a) * 0.25)


def doors_from_layout(lay: dict) -> list[dict]:
    """間取り → 各区画(zones と同順)の通路側ドア開口を決定論導出する。

    返り: [{"zone": i, "x": cx, "y": cy, "half": hd, "axis": "h"|"v"}...]。(cx,cy)=開口部の
    中心で、区画と通路を隔てる corridor 側の間仕切り線上に乗る。axis="h"=水平壁上(開口は
    x 方向に広がる)、"v"=垂直壁上(y 方向)。位置・半幅は _walls_from_layout が開ける開口と
    一致する(=壁の穴とドアが幾何整合)。lay が None/空・zones なしなら []。"""
    if not lay:
        return []
    zones = lay.get("zones") or []
    if not zones:
        return []
    n_a, n_b = lay["nA"], lay["nB"]
    doors: list[dict] = []
    if lay["horiz"]:
        _, yc0, _, yc1 = lay["corridor"]         # (x0, yc0, x1, yc1)
        top = zones[:n_a]                        # 通路の上=下辺(y=yc1)が通路側
        bot = zones[n_a:n_a + n_b]               # 通路の下=上辺(y=yc0)が通路側
        for i, (zx0, _zy0, zx1, _zy1) in enumerate(top):
            xm, hd = _door_span(zx0, zx1)
            doors.append({"zone": i, "x": xm, "y": yc1, "half": hd, "axis": "h"})
        for j, (zx0, _zy0, zx1, _zy1) in enumerate(bot):
            xm, hd = _door_span(zx0, zx1)
            doors.append({"zone": n_a + j, "x": xm, "y": yc0, "half": hd, "axis": "h"})
    else:
        xc0, _yc0, xc1, _yc1 = lay["corridor"]   # (xc0, y0, xc1, y1)
        right = zones[:n_a]                      # 通路の右=左辺(x=xc1)が通路側
        left = zones[n_a:n_a + n_b]              # 通路の左=右辺(x=xc0)が通路側
        for i, (_zx0, zy0, _zx1, zy1) in enumerate(right):
            ym, hd = _door_span(zy0, zy1)
            doors.append({"zone": i, "x": xc1, "y": ym, "half": hd, "axis": "v"})
        for j, (_zx0, zy0, _zx1, zy1) in enumerate(left):
            ym, hd = _door_span(zy0, zy1)
            doors.append({"zone": n_a + j, "x": xc0, "y": ym, "half": hd, "axis": "v"})
    return doors


# ---- 線分交差(LOS)----
def _orient(ax, ay, bx, by, cx, cy) -> float:
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _seg_cross(p1, p2, p3, p4) -> bool:
    """線分 p1p2 と p3p4 が真に交差するか(端点接触・共線は非交差=寛容側=ドアを塞がない)。"""
    d1 = _orient(p3[0], p3[1], p4[0], p4[1], p1[0], p1[1])
    d2 = _orient(p3[0], p3[1], p4[0], p4[1], p2[0], p2[1])
    d3 = _orient(p1[0], p1[1], p2[0], p2[1], p3[0], p3[1])
    d4 = _orient(p1[0], p1[1], p2[0], p2[1], p4[0], p4[1])
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def segment_blocked(p, q, walls) -> bool:
    """p-q の視線が walls のいずれかに遮られるか。"""
    for w in walls:
        if _seg_cross(p, q, w[0], w[1]):
            return True
    return False


class VisionOccluder:
    """視線遮蔽器。perception が距離・同一階フィルタ後のペアに blocks() を問う。

    indoor(同一建物・同一階=前段フィルタで保証): 間仕切り壁との LOS 判定。
    street(路上どうし): outdoor=True のときのみ、他建物フットプリントで遮蔽(既定 False=重い)。
    """

    def __init__(self, city, outdoor: bool = False):
        self.city = city
        self.outdoor = bool(outdoor)
        self._wall_cache: dict = {}
        self._bld_edges = None            # outdoor 用の全建物エッジ(遅延構築)

    def _walls(self, bid, floor) -> list:
        key = (bid, int(floor))
        w = self._wall_cache.get(key)
        if w is None:
            city = self.city
            if city is None or not getattr(city, "has_building", lambda _b: False)(bid):
                w = []
            else:
                w = building_walls(city.building(bid), floor, city)
            self._wall_cache[key] = w
        return w

    def blocks(self, a, b) -> bool:
        """a-b が壁で遮られる(=不可聴)なら True。"""
        if getattr(a, "loc", "street") == "outside" or getattr(b, "loc", "street") == "outside":
            return False
        if getattr(a, "building", None):
            walls = self._walls(a.building, a.floor)
            if not walls:
                return False
            return segment_blocked((a.x, a.y), (b.x, b.y), walls)
        if not self.outdoor:
            return False
        return self._blocks_outdoor(a, b)

    # ---- outdoor: 他建物フットプリントによる遮蔽(bbox 前フィルタ→辺交差)----
    def _building_edges(self) -> list:
        if self._bld_edges is None:
            edges = []
            for b in getattr(self.city, "buildings", []) or []:
                fp = b.get("footprint") or []
                if len(fp) < 3:
                    continue
                xs = [float(p[0]) for p in fp]
                ys = [float(p[1]) for p in fp]
                segs = [((float(fp[i][0]), float(fp[i][1])),
                         (float(fp[(i + 1) % len(fp)][0]), float(fp[(i + 1) % len(fp)][1])))
                        for i in range(len(fp))]
                edges.append((min(xs), min(ys), max(xs), max(ys), segs))
            self._bld_edges = edges
        return self._bld_edges

    def _blocks_outdoor(self, a, b) -> bool:
        p, q = (a.x, a.y), (b.x, b.y)
        lo_x, hi_x = (p[0], q[0]) if p[0] <= q[0] else (q[0], p[0])
        lo_y, hi_y = (p[1], q[1]) if p[1] <= q[1] else (q[1], p[1])
        for bx0, by0, bx1, by1, segs in self._building_edges():
            if bx1 < lo_x or bx0 > hi_x or by1 < lo_y or by0 > hi_y:
                continue
            for e in segs:
                if _seg_cross(p, q, e[0], e[1]):
                    return True
        return False


class IndoorLOSOccluder:
    """屋内知覚の区画粒度ゲート(B3b・indoor.los)。同一(建物,階)の同席ペアに対し、屋内実座標
    ind_x/ind_y の距離上限 + 間仕切り壁 LOS(segment_blocked)で「同席文脈から外す」判定を返す。

    休眠していた vision の LOS 資産(segment_blocked / 壁セグメント)の初配線。VisionOccluder と
    違い、壁は IndoorSpace(get の walls=キャッシュ済み)から取り、座標はマクロ x/y ではなく屋内
    ミクロ座標 ind_x/ind_y を使う(壁と ind 座標は同じフットプリント座標系)。街路/屋外ペア・ミクロ
    状態未割当(ind_zone is None=到着直後など)のペアは対象外(遮らない=従来同一)。座標・幾何しか
    見ない=no-fingerprint。前段(perception)が同一(建物,階)・距離を通したペアにだけ問われる。"""

    def __init__(self, indoor, max_dist_m: float = 0.0):
        self.indoor = indoor              # IndoorSpace(get(bid,floor)["walls"] を供給)
        self.max_dist_m = float(max_dist_m)   # 0=距離ゲート無効(壁 LOS のみ)

    def blocks(self, a, b) -> bool:
        """a-b が壁で遮られる/距離上限を超える(=同席文脈から外す)なら True。"""
        bid = getattr(a, "building", None)
        # 屋内どうし・同一(建物,階)のみ対象。街路/屋外は遮らない(前段フィルタの自衛確認)。
        if not bid or getattr(b, "building", None) != bid:
            return False
        if int(getattr(a, "floor", 0)) != int(getattr(b, "floor", 0)):
            return False
        # ミクロ状態未割当(_phase_indoor 前=到着直後)は判定材料が無い=遮らない(安全側)。
        if getattr(a, "ind_zone", None) is None or getattr(b, "ind_zone", None) is None:
            return False
        ax, ay = float(a.ind_x), float(a.ind_y)
        bx, by = float(b.ind_x), float(b.ind_y)
        if self.max_dist_m > 0.0 and math.hypot(ax - bx, ay - by) > self.max_dist_m:
            return True                   # 屋内で遠すぎる=区画粒度で不可視
        walls = self.indoor.get(bid, int(a.floor))["walls"]
        if not walls:
            return False
        return segment_blocked((ax, ay), (bx, by), walls)


def _cfg_get(cfg, dotted: str, default=None):
    """OmegaConf / dict どちらでも dotted パスで安全に読む。"""
    cur = cfg
    for part in dotted.split("."):
        if cur is None:
            return default
        if hasattr(cur, "get"):
            try:
                cur = cur.get(part)
                continue
            except Exception:
                return default
        cur = getattr(cur, part, None)
    return default if cur is None else cur


def occluder_from_cfg(city, cfg):
    """cfg.world.vision を読み、enabled のときだけ VisionOccluder を返す(既定 None=遮蔽なし)。"""
    enabled = bool(_cfg_get(cfg, "world.vision.enabled", False))
    if not enabled:
        return None
    outdoor = bool(_cfg_get(cfg, "world.vision.outdoor", False))
    return VisionOccluder(city, outdoor=outdoor)
