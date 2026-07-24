"""屋内人流コア(B2)の純関数テスト — 壁斥力 + ドアウェイポイント + 接触ペア抽出。

シミュ実行は不要。vision.building_layout / building_walls で実レイアウト(小さな矩形 footprint・
地名不使用)を合成し、society.world.indoor_flow の遷移駆動 SFM を決定論・軽量に検証する。
検収項目(タスク B2):
  - 壁非貫通(積分後の軌跡が壁を跨がない)
  - ドア通過(zone 間移動の軌跡がドア開口近傍を通る)
  - 決定論(同一入力 2 回 = repr 完全一致)
  - 接触ペア抽出(交差 2 移動者 = passing / 静止者近傍通過 = coplace)
  - 近傍 cap 挙動(最近傍 cap 体のみ寄与)
  - BFS フォールバック発火(waypoint 直線が壁と交差する場合の迂回)
"""
from __future__ import annotations

import numpy as np
import pytest

from society.world import indoor_flow as f
from society.world import vision


# ── fixture: 実レイアウト(vision の手続き生成。地名なしの矩形 footprint を合成) ──
# 区画数は building id + 階の決定論ハッシュで決まる。id="b" は水平間取りで上段3区画・下段2区画・
# どのドアもコアの x 範囲に重ならない(=廊下接近点がコア内に落ちない)クリーンな配置になる。
def _small_building(w=40.0, h=24.0):
    return {"id": "b", "kind": "office",
            "footprint": [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]]}


def _layout(w=40.0, h=24.0, floor=1):
    bld = _small_building(w, h)
    lay = vision.building_layout(bld, floor)
    walls = vision.building_walls(bld, floor)
    doors = f.doors_from_layout(lay)
    return lay, walls, doors


def _poly_min_dist(pt, poly):
    return min(f._point_seg(pt[0], pt[1], poly[i][0], poly[i][1],
                            poly[i + 1][0], poly[i + 1][1])[1]
               for i in range(len(poly) - 1))


def _mover_path(res, aid):
    return [(x, y) for (a, t, x, y) in res.samples if a == aid]


# ══════════════════════════════ 個体パラメータ(安定ハッシュ) ══════════════════════
def test_stable_params_in_range_and_deterministic():
    """v0∈[1.0,1.4)・radius∈[0.25,0.35) で agent_id の純関数(毎回同値=run.seed 非依存)。"""
    for aid in (1, 7, 42, 1000):
        v0, r = f.desired_speed(aid), f.body_radius(aid)
        assert f.V0_MIN <= v0 < f.V0_MAX
        assert f.RADIUS_MIN <= r < f.RADIUS_MAX
        assert f.desired_speed(aid) == v0        # 再呼び出しで同値
        assert f.body_radius(aid) == r
    # 異なる id は(ほぼ確実に)異なる値 = ハッシュが id を効かせている
    assert f.desired_speed(1) != f.desired_speed(2)


# ══════════════════════════════ 経路生成(route_waypoints) ══════════════════════
def test_route_waypoints_zone_to_zone():
    """zone→zone は 3〜5 waypoint で src ドア・dst ドアを経由する(直線・壁交差なし)。"""
    lay, walls, doors = _layout()
    src, dst = 0, lay["nA"]                       # 上段左 → 下段(いずれも左寄り=コア非横断)
    wps = f.route_waypoints(lay, doors, src, dst)
    assert 3 <= len(wps) <= 5
    # src/dst のドア点が waypoint 連鎖に含まれる(ドア経由)
    assert any(np.hypot(x - doors[src][0], y - doors[src][1]) < 1e-6 for (x, y) in wps)
    assert any(np.hypot(x - doors[dst][0], y - doors[dst][1]) < 1e-6 for (x, y) in wps)


def test_route_to_core():
    """dst='core' は 3 waypoint(src ドア→廊下接近→コア接近)で、コア矩形の外側を目標にする。"""
    lay, walls, doors = _layout()
    wps = f.route_waypoints(lay, doors, 0, "core")
    assert len(wps) == 3
    cx0, cy0, cx1, cy1 = lay["core"]
    gx, gy = wps[-1]
    # コア目標はコア矩形の外(内部に入らない近似目標)
    inside = (cx0 < gx < cx1) and (cy0 < gy < cy1)
    assert not inside


# ══════════════════════════════ 壁非貫通 + ドア通過 ══════════════════════════════
def test_wall_non_penetration():
    """積分後、移動者の連続サンプル間が壁を跨がない(壁貫通なし)・全点が bbox 内。"""
    lay, walls, doors = _layout()
    movers = [{"agent_id": 1, "src_zone": 0, "dst_zone": lay["nA"]}]
    res = f.integrate_transition(lay, walls, doors, movers, [])
    path = _mover_path(res, 1)
    assert len(path) >= 3
    crossings = sum(1 for i in range(len(path) - 1)
                    if f.segment_hits_wall(path[i], path[i + 1], walls))
    assert crossings == 0, f"壁貫通が {crossings} 回(壁斥力/経路が壁を跨いだ)"
    x0, y0, x1, y1 = lay["bbox"]
    for (x, y) in path:
        assert x0 - 0.6 <= x <= x1 + 0.6 and y0 - 0.6 <= y <= y1 + 0.6


def test_door_passage():
    """zone 間移動の軌跡が src ドア・dst ドアの開口近傍(< 0.6m)を通る。"""
    lay, walls, doors = _layout()
    src, dst = 0, lay["nA"]
    p = f.IndoorParams(sample_interval=2)          # ドア近傍判定のため少し細かくサンプル
    movers = [{"agent_id": 1, "src_zone": src, "dst_zone": dst}]
    res = f.integrate_transition(lay, walls, doors, movers, [], p)
    path = _mover_path(res, 1)
    assert _poly_min_dist(doors[src], path) < 0.6
    assert _poly_min_dist(doors[dst], path) < 0.6


# ══════════════════════════════ 決定論(repr 完全一致) ══════════════════════════════
def test_determinism_repr_identical():
    """同一入力 2 回で結果(軌跡サンプル + 接触ペア)が repr で完全一致(積分内乱数ゼロ)。"""
    lay, walls, doors = _layout()
    movers = [{"agent_id": 3, "src_zone": 0, "dst_zone": lay["nA"] + 1},
              {"agent_id": 1, "src_zone": 1, "dst_zone": lay["nA"]}]
    bys = [{"agent_id": 9, "pos": (lay["corridor"][0] + 8.0, 12.0)}]
    a = f.integrate_transition(lay, walls, doors, movers, bys)
    b = f.integrate_transition(lay, walls, doors, movers, bys)
    assert repr(a) == repr(b)
    assert a.samples == b.samples
    assert a.contacts == b.contacts


# ══════════════════════════════ 接触ペア抽出(passing / coplace) ══════════════════
def test_contact_passing_two_movers():
    """対向する 2 移動者が交差 = passing 接触が記録される(双方移動中)。"""
    lay, _walls, doors = _layout()
    movers = [{"agent_id": 1, "start": (-4.0, 0.1), "route": [(4.0, 0.1)]},
              {"agent_id": 2, "start": (4.0, -0.1), "route": [(-4.0, -0.1)]}]
    res = f.integrate_transition(lay, [], doors, movers, [])
    kinds = {(a, b): k for (a, b, k, _d) in res.contacts}
    assert (1, 2) in kinds
    assert kinds[(1, 2)] == "passing"


def test_contact_coplace_mover_near_bystander():
    """静止者の近傍を移動者が通過 = coplace 接触が記録される(一方が静止)。"""
    lay, _walls, doors = _layout()
    movers = [{"agent_id": 1, "start": (-4.0, 0.0), "route": [(4.0, 0.0)]}]
    bys = [{"agent_id": 5, "pos": (0.0, 0.3)}]
    res = f.integrate_transition(lay, [], doors, movers, bys)
    kinds = {(a, b): k for (a, b, k, _d) in res.contacts}
    assert (1, 5) in kinds
    assert kinds[(1, 5)] == "coplace"
    # duration は正の実数(連続サブステップ × dt)
    dur = [d for (a, b, k, d) in res.contacts if (a, b) == (1, 5)][0]
    assert dur > 0.0


# ══════════════════════════════ 近傍 cap 挙動 ══════════════════════════════════
def _ring(nn, d=1.0):
    return [[d * np.cos(2 * np.pi * i / nn), d * np.sin(2 * np.pi * i / nn)]
            for i in range(nn)]


def test_neighbor_cap_excludes_farthest():
    """cap=12 のとき、13 番目(最遠)の近傍は個体への斥力に寄与しない(cap の決定論選択)。"""
    pos12 = [[0.0, 0.0]] + _ring(12, 1.0)          # 個体0 + 距離1の12体
    pos13 = pos12 + [[1.9, 0.0]]                    # + 距離1.9の13体目(cutoff 内・最遠)
    c12 = f.WallCrowd(pos=pos12, vel=[[0.0, 0.0]] * 13, goal=[[10.0, 0.0]] * 13,
                      v0=[1.2] * 13, radius=[0.3] * 13, neighbor_cap=12)
    c13 = f.WallCrowd(pos=pos13, vel=[[0.0, 0.0]] * 14, goal=[[10.0, 0.0]] * 14,
                      v0=[1.2] * 14, radius=[0.3] * 14, neighbor_cap=12)
    f12, f13 = c12.forces()[0], c13.forces()[0]
    assert np.allclose(f12, f13), "13番目(最遠)が cap を越えて寄与している"
    # cap が実効している証拠: cap=6 では力が変わる(近傍集合が変わる)
    c6 = f.WallCrowd(pos=pos13, vel=[[0.0, 0.0]] * 14, goal=[[10.0, 0.0]] * 14,
                     v0=[1.2] * 14, radius=[0.3] * 14, neighbor_cap=6)
    assert not np.allclose(c6.forces()[0], f13)


# ══════════════════════════════ BFS フォールバック発火 ══════════════════════════════
def test_bfs_fallback_avoids_core():
    """左 zone→右 zone は廊下芯がコア(無開口)を横切る = BFS 迂回が発火し壁を跨がない。"""
    lay, walls, doors = _layout(60.0, 40.0)        # コア周りに迂回余地のある大きめレイアウト
    src, dst = 0, lay["nA"] - 1                     # 上段左 → 上段右(コアを挟む)
    straight = f.route_waypoints(lay, doors, src, dst, walls=None)     # 直線(壁チェックなし)
    detour = f.route_waypoints(lay, doors, src, dst, walls=walls)      # 壁チェック=BFS 迂回
    # 直線はコア壁を横切る / 迂回は横切らない
    straight_hits = sum(1 for i in range(len(straight) - 1)
                        if f.segment_hits_wall(straight[i], straight[i + 1], walls))
    detour_hits = sum(1 for i in range(len(detour) - 1)
                      if f.segment_hits_wall(detour[i], detour[i + 1], walls))
    assert straight_hits >= 1, "この配置では直線がコア壁を横切るはず(前提が崩れた)"
    assert detour_hits == 0, "BFS 迂回でも壁を跨いでいる"
    assert detour != straight and len(detour) > len(straight)          # 迂回点が増えている


def test_bfs_grid_path_avoids_wall_with_gap():
    """_grid_bfs_path: 隙間のある壁を回避する経路を返す(全区間が壁を跨がない・決定論)。"""
    lay, _w, _d = _layout(40.0, 24.0)
    # 廊下(y∈[9.84,14.16])を x=20 で塞ぐが上下に隙間を残す壁(BFS が迂回可能)
    blocker = [((20.0, 10.6), (20.0, 13.4))]
    start, goal = (5.0, 12.0), (35.0, 12.0)
    assert f.segment_hits_wall(start, goal, blocker)                   # 直線は塞がれている
    path = f._grid_bfs_path(lay, blocker, start, goal, cell=0.5)
    hits = sum(1 for i in range(len(path) - 1)
               if f.segment_hits_wall(path[i], path[i + 1], blocker))
    assert hits == 0 and len(path) >= 2
    # 決定論: 同一入力で同一経路
    assert f._grid_bfs_path(lay, blocker, start, goal, cell=0.5) == path
