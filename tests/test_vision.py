"""擬似視覚(壁による遮蔽)= バッチC のテスト。

- 決定論: 同じ建物 id/floor から同じ壁が生成される。
- LOS プリミティブ: 壁で遮蔽 / 開口部(ドア)越し・同室は可視。
- 同一建物・同一階で壁を挟む2エージェント → enabled 時は互いに不可聴、
  同室・開口部越し → 可聴。
- enabled=false(遮蔽器なし)→ hearers_of が現行の距離フィルタと完全一致。
- mock 24step ラン(enabled)がクラッシュせず完走。
"""
from __future__ import annotations

import math
from types import SimpleNamespace

from society.config import load_config
from society.engine.simulation import Simulation
from society.world import perception, vision
from society.world.perception import (build_index, clear_occluder, hearers_of,
                                       install_occluder)


# ---- 補助 ----
def _agent(aid, x, y, *, building=None, floor=0, loc=None, sleeping=False):
    return SimpleNamespace(id=aid, x=float(x), y=float(y),
                           building=building, floor=floor, sleeping=sleeping,
                           loc=loc or ("building" if building else "street"))


class _FakeCity:
    """building_walls 用の最小 city(POI なし → 決定論生成のフォールバック経路)。"""

    def __init__(self, buildings):
        self.buildings = buildings
        self._by_id = {b["id"]: b for b in buildings}

    def has_building(self, bid):
        return bid in self._by_id

    def building(self, bid):
        return self._by_id[bid]

    def pois_in_building(self, bid, floor):
        return []


def _zone_centroid(z):
    return ((z[0] + z[2]) / 2.0, (z[1] + z[3]) / 2.0)


def _big_building():
    return {"id": "bTEST", "kind": "retail",
            "footprint": [[0, 0], [120, 0], [120, 90], [0, 90]]}


def _floor_with_two_rooms(building):
    """side A(通路の片側)に2部屋以上ある階を決定論的に探す。"""
    for f in range(1, 60):
        lay = vision.building_layout(building, f)
        if lay and lay["nA"] >= 2:
            return f, lay
    raise AssertionError("nA>=2 の階が見つからない(生成ロジックの想定外)")


# ---- 1. 決定論 ----
def test_walls_are_deterministic_per_building():
    b = _big_building()
    w1 = vision.building_walls(b, 3)
    w2 = vision.building_walls(b, 3)
    assert w1 == w2 and len(w1) > 0
    # 別 id は別の壁(footprint も別)
    b2 = {"id": "bOTHER", "kind": "office",
          "footprint": [[0, 0], [70, 0], [70, 100], [0, 100]]}
    assert vision.building_walls(b2, 3) != w1
    # 階が違えば seed が違う → 一般に別間取り
    assert vision.building_walls(b, 4) != w1


def test_rng_hash_bit_exact_to_viewer():
    """viewer(JS)と同じビット列であることの回帰(mulberry32 / FNV-1a)。"""
    r = vision._rng(12345)
    got = [round(r(), 6) for _ in range(3)]
    assert got == [0.979728, 0.306752, 0.484205]
    assert vision._hash("b123") == 329446459


# ---- 2. LOS プリミティブ + 開口部 ----
def test_segment_blocked_primitive():
    wall = [((0, 0), (0, 10))]                 # x=0 の縦壁
    assert vision.segment_blocked((-1, 5), (1, 5), wall) is True
    assert vision.segment_blocked((1, 5), (2, 5), wall) is False   # 届かない


def test_door_is_an_opening():
    wall = vision._wall_with_door_h(0, 10, 5)  # y=5, x0..10, 中央に開口部
    assert vision.segment_blocked((5, 4), (5, 6), wall) is False   # ドア越し=可視
    assert vision.segment_blocked((1, 4), (1, 6), wall) is True    # 壁部=不可視


# ---- 3. 遮蔽器: 同室=可視 / 壁を挟む=不可視 ----
def test_occluder_same_room_visible_wall_blocks():
    b = _big_building()
    city = _FakeCity([b])
    occ = vision.VisionOccluder(city, outdoor=False)
    f, lay = _floor_with_two_rooms(b)
    top0, top1 = lay["zones"][0], lay["zones"][1]

    # 同室: 部屋の中心付近の2点 → 可視(遮蔽なし)
    cx0, cy0 = _zone_centroid(top0)
    a = _agent(1, cx0 - 1.0, cy0, building=b["id"], floor=f)
    bb = _agent(2, cx0 + 1.0, cy0, building=b["id"], floor=f)
    assert occ.blocks(a, bb) is False

    # 隣室: それぞれの中心 → 間仕切り(無開口)で不可視
    cx1, cy1 = _zone_centroid(top1)
    c = _agent(3, cx1, cy1, building=b["id"], floor=f)
    assert occ.blocks(a, c) is True


def test_occluder_ignores_outside_and_missing_building():
    city = _FakeCity([_big_building()])
    occ = vision.VisionOccluder(city)
    out = _agent(1, 0, 0, loc="outside")
    other = _agent(2, 1, 1, building="bTEST", floor=1)
    assert occ.blocks(out, other) is False
    # 地図に無い建物 → 遮蔽なし
    ghost_a = _agent(3, 0, 0, building="bGHOST", floor=1)
    ghost_b = _agent(4, 5, 5, building="bGHOST", floor=1)
    assert occ.blocks(ghost_a, ghost_b) is False


# ---- 4. enabled 時に hearers_of が壁で切れる ----
def test_hearers_of_respects_walls_when_enabled():
    b = _big_building()
    city = _FakeCity([b])
    occ = vision.VisionOccluder(city, outdoor=False)
    f, lay = _floor_with_two_rooms(b)
    top0, top1 = lay["zones"][0], lay["zones"][1]
    cx0, cy0 = _zone_centroid(top0)
    cx1, cy1 = _zone_centroid(top1)
    radius = 500.0                              # 距離では絶対に切れない=遮蔽だけを見る

    a = _agent(1, cx0, cy0, building=b["id"], floor=f)
    same = _agent(2, cx0 + 1.0, cy0, building=b["id"], floor=f)
    across = _agent(3, cx1, cy1, building=b["id"], floor=f)
    agents = [a, same, across]

    # 遮蔽なし: 同室・隣室とも可聴
    heard_off = {o.id for o in hearers_of(a, agents, radius)}
    assert heard_off == {2, 3}
    # 遮蔽あり: 同室(2)は可聴、隣室(3)は不可聴
    heard_on = {o.id for o in hearers_of(a, agents, radius, occluder=occ)}
    assert heard_on == {2}


# ---- 5. enabled=false(遮蔽器なし)= 現行挙動と完全一致 ----
def _reference_hearers(speaker, agents, radius):
    """occluder 導入前の hearers_of と同じ純距離フィルタ(参照実装)。"""
    def ctx(ag):
        if ag.loc == "outside":
            return ("outside", ag.id)
        if ag.building:
            return ("bld", ag.building, ag.floor)
        return ("street",)
    sc = ctx(speaker)
    out = [o for o in agents
           if o.id != speaker.id and not o.sleeping and ctx(o) == sc
           and math.hypot(o.x - speaker.x, o.y - speaker.y) <= radius]
    return sorted(out, key=lambda a: a.id)


def test_off_matches_current_behavior():
    clear_occluder()
    b = "bMIX"
    agents = [
        _agent(1, 0, 0, building=b, floor=2),
        _agent(2, 3, 0, building=b, floor=2),
        _agent(3, 200, 0, building=b, floor=2),       # 遠い=距離で切れる
        _agent(4, 1, 1, building=b, floor=3),          # 別階=文脈違い
        _agent(5, 2, 2, building=b, floor=2, sleeping=True),  # 睡眠=除外
        _agent(6, 10, 0, loc="street"),                # 路上=文脈違い
    ]
    radius = 40.0
    for sp in agents:
        ref = [a.id for a in _reference_hearers(sp, agents, radius)]
        legacy = [a.id for a in hearers_of(sp, agents, radius)]                 # occluder=None
        legacy_explicit_none = [a.id for a in hearers_of(sp, agents, radius, occluder=None)]
        idx = build_index(agents, radius)                                       # occluder=None
        indexed = [a.id for a in hearers_of(sp, idx, radius)]
        assert legacy == ref, f"legacy off が現行と不一致: {sp.id}"
        assert legacy_explicit_none == ref
        assert indexed == ref, f"index off が現行と不一致: {sp.id}"


def test_install_occluder_toggle_is_scoped():
    """install_occluder 据え付け → 効く。clear で外すと従来へ復帰。"""
    b = _big_building()
    city = _FakeCity([b])
    occ = vision.VisionOccluder(city)
    f, lay = _floor_with_two_rooms(b)
    cx0, cy0 = _zone_centroid(lay["zones"][0])
    cx1, cy1 = _zone_centroid(lay["zones"][1])
    a = _agent(1, cx0, cy0, building=b["id"], floor=f)
    across = _agent(3, cx1, cy1, building=b["id"], floor=f)
    agents = [a, across]
    try:
        install_occluder(occ)
        assert {o.id for o in hearers_of(a, agents, 500.0)} == set()   # 壁で不可聴
        idx = build_index(agents, 500.0)                                # 据え付けへ後退
        assert {o.id for o in hearers_of(a, idx, 500.0)} == set()
    finally:
        clear_occluder()
    assert {o.id for o in hearers_of(a, agents, 500.0)} == {3}          # 復帰=可聴


def test_occluder_from_cfg_default_off():
    cfg = load_config(["run.n_agents=2", "run.n_steps=1", "run.name=vcfg"])
    assert vision.occluder_from_cfg(None, cfg) is None                  # 既定=遮蔽なし
    cfg2 = load_config(["run.n_agents=2", "run.n_steps=1", "run.name=vcfg2",
                        "world.vision.enabled=true"])
    occ = vision.occluder_from_cfg(_FakeCity([_big_building()]), cfg2)
    assert isinstance(occ, vision.VisionOccluder) and occ.outdoor is False


# ---- 6. mock 24step ラン(enabled)が完走 ----
def test_mock_run_with_vision_enabled(tmp_path):
    # government.enabled=false は別バッチ所有 scheduler._gov の既知バグ回避(config に
    # government ブロックが無いと OmegaConf.to_container(plain dict) が落ちる)。本テスト
    # とは無関係で、既定の baseline mock ランも同様に落ちる=環境側の未修正事項。
    cfg = load_config(["run.seed=42", "run.n_agents=12", "run.n_steps=24",
                       "run.name=vision", "world.vision.enabled=true",
                       "government.enabled=false"])
    sim = Simulation(cfg, out_dir=tmp_path / "vision")
    occ = vision.occluder_from_cfg(sim.city, sim.cfg)
    assert occ is not None
    try:
        install_occluder(occ)
        summary = sim.run()
    finally:
        clear_occluder()
    assert summary["n_events"] > 0
    # 屋内会話が壁で全滅していない(=完走かつ知覚が壊れていない)
    assert summary["event_kinds"].get("move_segment", 0) > 0
