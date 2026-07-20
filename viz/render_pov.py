"""CPU 決定論 POV レンダラ(エージェント視覚 v1・休眠骨格)。

設計: docs/research/agent-vision.md §2b/§2e/§4 v1。GPU レンダは画素非決定(FMA/丸め/加算順が
ハード依存)で決定論と相性が悪い。ここは **CPU ソフトウェアで画素決定** の低解像度セマンティック
POV を numpy + zlib で自作し、**同一入力→同一バイト PNG** を保証する(タイムスタンプ非埋め込み・
zlib 固定レベル)。カメラ = エージェント位置 + heading。建物 footprint を簡易レイキャストで壁柱に
押し出し、地形(ElevationGrid があれば)で地平線を上下する。

決定論の要点:
- 乱数を一切使わない(色は建物 id の安定ハッシュから)。
- 浮動小数は numpy float64 の逐次演算のみ(同一プラットフォーム上で再現)。
- PNG は手書きエンコーダ(IHDR/IDAT/IEND のみ・時刻チャンクなし・zlib level 固定)。

本モジュールは観測ログ用の画像を作るだけで、シミュの決定論の骨格(発火・乱数・呼数)には
一切入力しない(agent-vision.md §2e の隔離)。make_viewer3d.py には触れない(別担当が編集中)。
"""
from __future__ import annotations

import math
import struct
import zlib

import numpy as np

# セマンティック配色(クラス=色)。VLM でも人が見ても意味が読める低情報密度の塗り。
_SKY = (176, 206, 235)
_GROUND = (120, 118, 108)
_ZLIB_LEVEL = 6            # 固定=決定論(環境非依存の圧縮結果)


def _stable_color(key: str) -> tuple[int, int, int]:
    """建物 id 等から安定な RGB を作る(FNV-1a 32bit・乱数なし=決定論)。"""
    h = 2166136261
    for ch in str(key):
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    r = 70 + (h & 0x7F)
    g = 70 + ((h >> 8) & 0x7F)
    b = 70 + ((h >> 16) & 0x7F)
    return (int(r), int(g), int(b))


def _bbox(footprint) -> tuple[float, float, float, float] | None:
    xs = [float(p[0]) for p in footprint]
    ys = [float(p[1]) for p in footprint]
    if len(xs) < 2:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def _ray_segment_t(px, py, dx, dy, ax, ay, bx, by) -> float | None:
    """半直線 (p + t·d, t>0) と線分 a-b の交差パラメタ t を返す(なければ None)。"""
    ex, ey = bx - ax, by - ay
    denom = dx * ey - dy * ex
    if abs(denom) < 1e-12:
        return None
    wx, wy = ax - px, ay - py
    t = (wx * ey - wy * ex) / denom          # 半直線側パラメタ(距離スケール、|d|=1 前提)
    s = (wx * dy - wy * dx) / denom           # 線分側パラメタ [0,1]
    if t > 1e-6 and -1e-9 <= s <= 1.0 + 1e-9:
        return t
    return None


def render_pov(*, cam_x: float, cam_y: float, heading: float,
               buildings, terrain=None, width: int = 96, height: int = 72,
               fov_deg: float = 90.0, max_dist_m: float = 200.0) -> bytes:
    """POV セマンティック画像を PNG バイト列で返す(同一入力→同一バイト)。

    引数:
      cam_x, cam_y : カメラ(=エージェント)位置(地図ローカル m)。
      heading      : 進行方向の角度 [rad](atan2(dy,dx))。
      buildings    : [{"id", "footprint": [[x,y],...]}] の列(名前は不要)。
      terrain      : ElevationGrid 互換(height_at(x,y))。None なら地平線は中央固定。
      width,height : 画素サイズ(低解像度)。fov_deg: 水平画角。max_dist_m: レイの最大距離。
    """
    W, H = int(width), int(height)
    img = np.empty((H, W, 3), dtype=np.uint8)
    # 地形で地平線行を上下(カメラ地点の地表高が高いほど遠くを広く見下ろす近似)。
    horizon = H // 2
    if terrain is not None:
        try:
            z = float(terrain.height_at(cam_x, cam_y))
            horizon = int(min(H - 1, max(1, H // 2 - int(round(z * 0.3)))))
        except Exception:
            horizon = H // 2
    img[:horizon] = _SKY
    img[horizon:] = _GROUND

    # レイキャスト対象を bbox で前フィルタ(max_dist 圏の建物だけ)。決定論(順序=入力順)。
    edges: list[tuple] = []
    r = float(max_dist_m)
    for b in buildings or []:
        fp = b.get("footprint") or []
        if len(fp) < 3:
            continue
        bb = _bbox(fp)
        if bb is None:
            continue
        x0, y0, x1, y1 = bb
        if x1 < cam_x - r or x0 > cam_x + r or y1 < cam_y - r or y0 > cam_y + r:
            continue
        col = _stable_color(b.get("id") or b.get("name") or "b")
        n = len(fp)
        for i in range(n):
            ax, ay = float(fp[i][0]), float(fp[i][1])
            bx, by = float(fp[(i + 1) % n][0]), float(fp[(i + 1) % n][1])
            edges.append((ax, ay, bx, by, col))

    if edges:
        half = math.radians(float(fov_deg)) / 2.0
        wall_scale = float(H) * 8.0           # 壁柱の見かけ高さの基準(1/距離でスケール)
        denom_col = max(1, W - 1)
        for c in range(W):
            ang = heading - half + (2.0 * half) * (c / denom_col)
            dx, dy = math.cos(ang), math.sin(ang)
            best_t = None
            best_col = None
            for (ax, ay, bx, by, col) in edges:
                t = _ray_segment_t(cam_x, cam_y, dx, dy, ax, ay, bx, by)
                if t is not None and t <= r and (best_t is None or t < best_t):
                    best_t = t
                    best_col = col
            if best_t is None:
                continue
            wall_h = int(min(H, max(1.0, wall_scale / max(best_t, 1.0))))
            # 距離で陰影(遠いほど暗く)。決定論の整数演算。
            shade = max(40, 255 - int(best_t / r * 160))
            r8 = best_col[0] * shade // 255
            g8 = best_col[1] * shade // 255
            b8 = best_col[2] * shade // 255
            top = max(0, horizon - wall_h // 2)
            bot = min(H, horizon + wall_h // 2)
            img[top:bot, c, 0] = r8
            img[top:bot, c, 1] = g8
            img[top:bot, c, 2] = b8

    return _encode_png(img)


def _encode_png(arr: np.ndarray) -> bytes:
    """(H,W,3) uint8 → PNG バイト列(手書き・時刻チャンクなし・zlib 固定=決定論)。"""
    H, W, _ = arr.shape
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    raw = bytearray()
    row_bytes = W * 3
    flat = arr.reshape(H, row_bytes)
    for y in range(H):
        raw.append(0)                          # フィルタタイプ 0(None)
        raw.extend(flat[y].tobytes())
    comp = zlib.compress(bytes(raw), _ZLIB_LEVEL)

    def _chunk(typ: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(typ + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + typ + data + struct.pack(">I", crc)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0)   # 8bit, color type 2 (RGB)
    return sig + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", comp) + _chunk(b"IEND", b"")


def heading_angle(dx: float, dy: float) -> float:
    """方向ベクトル → 角度 [rad]。ゼロは 0.0(東向き)へ。"""
    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        return 0.0
    return math.atan2(dy, dx)
