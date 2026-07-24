"""空間レイヤ核 B1: IndoorSpace + vision の n_override / doors_from_layout のテスト。

- 決定論: 同入力2回で同一結果(layout/walls/doors/zone_types)。
- n_override 経路が POI 経路と同型(cols 前に乱数を消費しない=同 n で完全一致)。
- vision.building_layout の n_override=None が従来動作とバイト不変(test_vision も無修正で緑=別担保)。
- SpaceType 割当規則(bulk を全区画・accent を最小側へ)。
- floor_layouts 不在フォールバック(spec 無しでも間取り正典で成立)。
- doors_from_layout の幾何整合(ドアが corridor 側の壁上・開口を通る視線は遮られない)。

fixture spec はテスト内に埋め込む(並行タスクの成果物 data/floor_layouts.json に依存しない)。
"""
from __future__ import annotations

from society.config import load_config
from society.world import indoor, vision


# ---- fixtures ---------------------------------------------------------------
def _big(kind="retail"):
    return {"id": "bTEST", "kind": kind,
            "footprint": [[0, 0], [120, 0], [120, 90], [0, 90]]}


class _FakeCity:
    """building()/has_building()/pois_in_building() だけを持つ最小 city(floor_guide なし)。"""

    def __init__(self, buildings, pois_by_bf=None):
        self._by_id = {b["id"]: b for b in buildings}
        self._pois = pois_by_bf or {}          # {(bid, floor): [poi...]}

    def has_building(self, bid):
        return bid in self._by_id

    def building(self, bid):
        return self._by_id[bid]

    def pois_in_building(self, bid, floor):
        return list(self._pois.get((bid, floor), []))


_SPACE_TYPES = {
    "office": ["desk", "meeting", "break"],
    "shop": ["sales", "register"],
    "restaurant": ["seating", "kitchen", "register"],
    "generic": ["sales", "service"],
}


def _area(z):
    return abs((z[2] - z[0]) * (z[3] - z[1]))


# ---- 1. 決定論 --------------------------------------------------------------
def test_indoor_deterministic():
    b = _big()
    city = _FakeCity([b])
    sp1 = indoor.IndoorSpace(city, space_types=_SPACE_TYPES)
    sp2 = indoor.IndoorSpace(city, space_types=_SPACE_TYPES)
    c1 = sp1.get("bTEST", 3)
    c2 = sp2.get("bTEST", 3)
    assert c1["layout"] == c2["layout"]
    assert c1["walls"] == c2["walls"] and len(c1["walls"]) > 0
    assert c1["doors"] == c2["doors"] and len(c1["doors"]) > 0
    assert c1["zone_types"] == c2["zone_types"]
    # キャッシュ: 同一インスタンスの再取得も同一
    assert sp1.get("bTEST", 3) is sp1.get("bTEST", 3)


def test_missing_building_is_empty():
    sp = indoor.IndoorSpace(_FakeCity([_big()]), space_types=_SPACE_TYPES)
    cell = sp.get("bGHOST", 1)
    assert cell["layout"] is None and cell["walls"] == [] and cell["zone_types"] == []


# ---- 2. n_override 経路 = POI 経路と同型 -----------------------------------
def test_n_override_matches_poi_path():
    """n_override=k と『k 個の POI がある city』が同一間取り(cols 前に乱数を消費しない同型)。"""
    b = _big()
    for k in range(1, 11):                      # POI 経路は min(10, k)
        city = _FakeCity([b], {("bTEST", 5): [{"cat": "shop", "floor": 5}] * k})
        via_poi = vision.building_layout(b, 5, city)
        via_override = vision.building_layout(b, 5, None, n_override=k)
        assert via_poi == via_override, f"n_override={k} が POI 経路と不一致"
        assert len(via_override["zones"]) == k


def test_n_override_clamps():
    b = _big()
    assert len(vision.building_layout(b, 2, None, n_override=20)["zones"]) == 12  # 上限
    assert len(vision.building_layout(b, 2, None, n_override=0)["zones"]) == 1    # 下限(n<=0→1)


def test_n_override_none_is_legacy():
    """n_override=None は従来 3引数呼びと完全一致(バイト不変の明示)。"""
    b = _big()
    for f in range(1, 8):
        assert vision.building_layout(b, f) == vision.building_layout(b, f, None, None)
        assert vision.building_walls(b, f) == \
            vision._walls_from_layout(vision.building_layout(b, f, None, n_override=None))


# ---- 3. SpaceType 割当規則 --------------------------------------------------
def test_assign_rule_unit():
    """_assign: bulk を全区画・accent を最小側へ(最小=末尾 accent)。"""
    # 面積が明確に異なる4区画(降順: z0>z1>z2>z3)
    zones = [(0, 0, 10, 10), (0, 0, 8, 8), (0, 0, 6, 6), (0, 0, 4, 4)]
    got = indoor.IndoorSpace._assign(zones, ["desk", "meeting", "break"])
    # desk=最大2つ, meeting=3番目, break=最小
    assert got == ["desk", "desk", "meeting", "break"]
    # accent が1つ(sales+register)
    assert indoor.IndoorSpace._assign(zones, ["sales", "register"]) == \
        ["sales", "sales", "sales", "register"]
    # 区画1つ = bulk のみ(accent を敷かない=bulk を最低1区画残す)
    assert indoor.IndoorSpace._assign([(0, 0, 5, 5)], ["desk", "meeting"]) == ["desk"]
    # 型リスト空 → 汎用 "zone"
    assert indoor.IndoorSpace._assign(zones, []) == ["zone", "zone", "zone", "zone"]


def test_zone_types_via_kind_office():
    """use が建物 kind から office に解決 → desk/meeting/break が規則どおり敷かれる。"""
    b = _big(kind="office")
    city = _FakeCity([b])
    specs = [{"match": ["bTEST"], "floors": [{"f": 3, "shops": 5}]}]
    sp = indoor.IndoorSpace(city, floor_specs=specs, space_types=_SPACE_TYPES)
    cell = sp.get("bTEST", 3)
    zones, types = cell["layout"]["zones"], cell["zone_types"]
    assert len(zones) == 5 and len(types) == 5
    assert types.count("desk") == 3 and types.count("meeting") == 1 and types.count("break") == 1
    # 面積昇順: 最小=break, 2番目に小=meeting
    order_small = sorted(range(5), key=lambda i: (_area(zones[i]), i))
    assert types[order_small[0]] == "break"
    assert types[order_small[1]] == "meeting"
    # 最大3区画はすべて desk
    for i in order_small[2:]:
        assert types[i] == "desk"


def test_zone_types_via_poi_cat():
    """floorguide/spec 無し・POI cat 多数決で use=restaurant → seating/kitchen/register。"""
    b = _big(kind="generic")
    pois = [{"cat": "restaurant", "floor": 2}] * 3 + [{"cat": "shop", "floor": 2}]
    city = _FakeCity([b], {("bTEST", 2): pois})
    sp = indoor.IndoorSpace(city, space_types=_SPACE_TYPES)
    cell = sp.get("bTEST", 2)
    types = cell["zone_types"]
    assert "seating" in types                   # bulk
    assert types.count("kitchen") == 1 and types.count("register") == 1


# ---- 4. spec 不在フォールバック / zone_mix ---------------------------------
def test_spec_fallback_uses_canon_layout():
    """floor_layouts 不在(specs=[])でも間取り正典で layout/zone_types が成立(kind→use)。"""
    b = _big(kind="retail")                     # kind retail → use shop
    sp = indoor.IndoorSpace(_FakeCity([b]), floor_specs=[], space_types=_SPACE_TYPES)
    cell = sp.get("bTEST", 4)
    assert cell["layout"] is not None and len(cell["zone_types"]) == len(cell["layout"]["zones"])
    assert "sales" in cell["zone_types"]        # shop の bulk
    assert cell["zone_types"].count("register") == 1


def test_load_floor_specs_missing_returns_empty(tmp_path):
    assert indoor.load_floor_specs(tmp_path / "nope.json") == []
    # 実在ファイル + match/spec_floor 規約
    p = tmp_path / "fl.json"
    p.write_text('{"buildings":[{"match":["ヒカリエ"],"floors":[{"f":2,"use":"fashion"}]}]}',
                 encoding="utf-8")
    specs = indoor.load_floor_specs(p)
    assert indoor.spec_floor(specs, "渋谷ヒカリエ", 2)["use"] == "fashion"   # 部分一致
    assert indoor.spec_floor(specs, "渋谷ヒカリエ", 9) is None                # 階なし


def test_zone_mix_drives_count_and_types():
    """zone_mix={キー:区画数} が区画数 override(合計)+型割当を駆動する。"""
    b = _big(kind="generic")
    specs = [{"match": ["bTEST"],
              "floors": [{"f": 6, "zone_mix": {"desk": 4, "meeting": 1}}]}]
    sp = indoor.IndoorSpace(_FakeCity([b]), floor_specs=specs, space_types=_SPACE_TYPES)
    cell = sp.get("bTEST", 6)
    assert len(cell["layout"]["zones"]) == 5    # 合計 4+1
    assert cell["zone_types"].count("desk") == 4 and cell["zone_types"].count("meeting") == 1


# ---- 5. doors_from_layout 幾何整合 ------------------------------------------
def test_doors_on_corridor_wall_and_passable():
    b = _big()
    for f in range(1, 12):
        lay = vision.building_layout(b, f)
        if lay is None:
            continue
        doors = vision.doors_from_layout(lay)
        walls = vision._walls_from_layout(lay)
        assert len(doors) == len(lay["zones"])
        x0, y0, x1, y1 = lay["bbox"]
        cx0, cyc0, cx1, cyc1 = lay["corridor"]
        for d in doors:
            if d["axis"] == "h":                # 水平壁=corridor の y 境界上
                assert abs(d["y"] - cyc0) < 1e-9 or abs(d["y"] - cyc1) < 1e-9
                assert x0 - 1e-9 <= d["x"] <= x1 + 1e-9
                seg_a = (d["x"], d["y"] - 0.3)
                seg_b = (d["x"], d["y"] + 0.3)
            else:                               # 垂直壁=corridor の x 境界上
                assert abs(d["x"] - cx0) < 1e-9 or abs(d["x"] - cx1) < 1e-9
                assert y0 - 1e-9 <= d["y"] <= y1 + 1e-9
                seg_a = (d["x"] - 0.3, d["y"])
                seg_b = (d["x"] + 0.3, d["y"])
            # 開口部の中心を横切る短い視線は壁に遮られない(ドア=通り抜け可)
            assert vision.segment_blocked(seg_a, seg_b, walls) is False


def test_doors_empty_on_bad_layout():
    assert vision.doors_from_layout(None) == []
    assert vision.doors_from_layout({}) == []


# ---- 6. indoor_from_cfg 既定 OFF -------------------------------------------
def test_indoor_from_cfg_default_off():
    cfg = load_config(["run.n_agents=2", "run.n_steps=1", "run.name=icfg"])
    assert indoor.indoor_from_cfg(None, cfg) is None            # 既定 OFF=空間レイヤ不使用
    cfg2 = load_config(["run.n_agents=2", "run.n_steps=1", "run.name=icfg2",
                        "indoor.enabled=true"])
    sp = indoor.indoor_from_cfg(_FakeCity([_big()]), cfg2)
    assert isinstance(sp, indoor.IndoorSpace)
    # conf の space_types が読めている(office→desk が入っている)
    assert sp.space_types.get("office", [])[:1] == ["desk"]
