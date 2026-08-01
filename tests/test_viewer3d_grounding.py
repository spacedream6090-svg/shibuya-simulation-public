"""3D ビューアの接地(grounding)契約の検収 — 地面の乱れ/エージェント浮遊の修正バッチ。

viewer3d.html は単一 HTML の JS なので、ここでは
  ①「JS の座標式を Python へ移植した参照実装」で数値契約を固定し
  ② 生成 HTML にその式・レイヤ既定が実際に入っていること
の 2 段で検証する(ブラウザ実行は範囲外)。

対象(いずれも本バッチの修正点):
- upOf(w): 屋内は建物基準(gz + (floor-1)*floorH)の**絶対 y**。floor は 0..99 にガード。
- footY(x,y,w): 足元の接地面(屋外=groundAt / 屋内=upOf)。
- 足元アンカー: カプセル中心 = 足元 + 半高×表示倍率 → 底面が接地面に一致する。
- 押出し深さ: base から depth=(levels+below)*FH で屋上が height に届く(地下階建物の沈み)。
- 地上線路は groundAt+z+0.6(絶対 0.6 ではない)。
- OSM ドレープは地形サーフェスと同一 geometry(別平面の 240 分割ではない)。
- 地下街(ubld)レイヤは既定 OFF・地表クリップあり。
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MV3 = _load("make_viewer3d", "viz/make_viewer3d.py")

FH = 3.5
AG_R, AG_BODY = 0.45, 0.9          # 素寸法(人間比 1.0 = 半径0.45m・全高1.8m)
AG_HALF = AG_BODY / 2 + AG_R       # 0.9m
BLD_SKIRT = 3.0


# --------------------------------------------------------------------------- #
# JS 参照移植(viz/make_viewer3d.py の upOf/footY/placeAgents と同一式)
# --------------------------------------------------------------------------- #
def floor_h_of(b):
    return b.get("floorH") or FH


def up_of(w, buildings):
    """屋内の足元(フロア面)の絶対 y[m]。路上は 0(呼び出し側が groundAt を使う)。"""
    if w >= 1000:
        bi = (w - 1000) // 100
        floor = (w - 1000) % 100
        floor = max(0, min(99, floor))
        if 0 <= bi < len(buildings):
            b = buildings[bi]
            return (b.get("gz") or 0.0) + max(floor - 1, 0) * floor_h_of(b)
        return max((floor - 1) * FH, 0.0)
    return 0.0


def foot_y(x, y, w, buildings, ground_at):
    return up_of(w, buildings) if w >= 1000 else ground_at(x, y)


def capsule_center_y(foot, scale):
    """placeAgents が置くインスタンス中心の y。"""
    return foot + AG_HALF * scale


def capsule_bottom_y(center, scale):
    """カプセル最下点(Lathe の最小 y=-(body/2+r))。"""
    return center - AG_HALF * scale


def extrude_span(b):
    """buildBuildings の押出し z 範囲 [下端, 上端](gz 込み・スカート込み)。"""
    depth = max(b.get("depth") or b.get("height") or FH, 1.0) + BLD_SKIRT
    z0 = (b.get("base") or 0.0) + (b.get("gz") or 0.0) - BLD_SKIRT
    return z0, z0 + depth


# --------------------------------------------------------------------------- #
# 1. upOf: 屋内は建物基準・floor ガード
# --------------------------------------------------------------------------- #
def test_upof_indoor_uses_building_floor_height_and_gz():
    """屋内の足元 = gz + (floor-1)*floorH。実測階高(floorH)がある建物はそれを使う。"""
    blds = [
        {"id": "b0", "levels": 10, "height": 35.0, "base": 0.0, "gz": 12.0},
        # PLATEAU 実測 60m/10 階 = 実効階高 6.0m(levels*3.5=35 とは食い違う)
        {"id": "b1", "levels": 10, "height": 60.0, "base": 0.0, "gz": -4.0,
         "floorH": 6.0},
    ]
    assert up_of(1000 + 0 * 100 + 1, blds) == pytest.approx(12.0)          # 1F=地表
    assert up_of(1000 + 0 * 100 + 4, blds) == pytest.approx(12.0 + 3 * 3.5)
    assert up_of(1000 + 1 * 100 + 1, blds) == pytest.approx(-4.0)
    assert up_of(1000 + 1 * 100 + 10, blds) == pytest.approx(-4.0 + 9 * 6.0)
    # 実測階高を使うので最上階でも屋根(gz+height)を超えない
    for f in range(1, 11):
        for bi, b in enumerate(blds):
            foot = up_of(1000 + bi * 100 + f, blds)
            assert foot <= b["gz"] + b["height"] + 1e-9


def test_upof_floor_guard_and_unknown_building():
    """floor は 0..99 にクランプ(桁あふれ由来の負値・巨大値でも破綻しない)。
    建物 index が範囲外の古い tracks でも例外にならず地表付近へ落ちる。"""
    blds = [{"id": "b0", "levels": 3, "height": 10.5, "base": 0.0, "gz": 5.0}]
    assert up_of(1000 + 0 * 100 + 0, blds) == pytest.approx(5.0)   # floor0 → 1F 扱い
    assert up_of(1000 + 0 * 100 + 99, blds) == pytest.approx(5.0 + 98 * FH)
    # 建物 index 越え(旧 tracks・未知建物 idx0 退避の名残)
    far = up_of(1000 + 7 * 100 + 2, blds)
    assert far == pytest.approx(FH)
    assert far >= 0.0


# --------------------------------------------------------------------------- #
# 2. 足元アンカー(浮遊/埋没ゼロ)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scale", [1.0, 2.0, 4.0])
def test_capsule_bottom_sits_exactly_on_ground(scale):
    """屋外: カプセル底面が地表と厳密一致(旧実装は中心配置で 3.9m 埋没していた)。"""
    def ground_at(x, y):
        return 0.02 * x - 0.01 * y + 3.0

    for (x, y) in [(0.0, 0.0), (120.0, -40.0), (-333.3, 77.7)]:
        foot = foot_y(x, y, 0, [], ground_at)
        center = capsule_center_y(foot, scale)
        assert capsule_bottom_y(center, scale) == pytest.approx(ground_at(x, y))
        # 旧実装(中心 groundAt+1.2・半高 5.1m)は 3.9m 埋没していた
        old_bottom = ground_at(x, y) + 1.2 - 5.1
        assert ground_at(x, y) - old_bottom == pytest.approx(3.9)


def test_capsule_bottom_sits_on_floor_slab_indoor():
    """屋内: カプセル底面 = そのフロアの床面(フロア板 y と 0.15m 以内で一致)。"""
    blds = [{"id": "b0", "levels": 8, "height": 28.0, "base": -7.0, "gz": 9.0}]
    for f in (1, 3, 8):
        w = 1000 + 0 * 100 + f
        foot = foot_y(0.0, 0.0, w, blds, lambda x, y: -99.0)   # 屋内は地表非依存
        center = capsule_center_y(foot, 2.0)
        assert capsule_bottom_y(center, 2.0) == pytest.approx(foot)
        plate_y = blds[0]["gz"] + max(f - 1, 0) * FH + 0.15    # _makePlate と同式
        assert abs(foot - plate_y) <= 0.15 + 1e-9


# --------------------------------------------------------------------------- #
# 3. 押出し深さ(地下階建物の沈み)
# --------------------------------------------------------------------------- #
def test_extrude_depth_puts_roof_at_height_even_with_basement():
    """base から depth=(levels+below)*FH 押し出せば屋上が height に一致する。
    旧実装(depth=height)は below*FH だけ屋上が沈んでいた。"""
    levels, below, gz = 10, 2, 6.0
    b = {"levels": levels, "below": below, "height": levels * FH,
         "base": -below * FH, "depth": (levels + below) * FH, "gz": gz}
    z0, z1 = extrude_span(b)
    assert z1 == pytest.approx(gz + levels * FH)          # 屋上 = gz + height
    assert z0 == pytest.approx(gz - below * FH - BLD_SKIRT)   # 地下 + スカート
    # depth が無い旧 scene.json(後方互換)は従来どおり height を使う
    old = dict(b)
    old.pop("depth")
    _z0, z1_old = extrude_span(old)
    assert z1_old == pytest.approx(gz + levels * FH - below * FH)
    assert z1 - z1_old == pytest.approx(below * FH)


# --------------------------------------------------------------------------- #
# 4. 生成 HTML に式・既定が入っている
# --------------------------------------------------------------------------- #
def _scene_json(with_terrain_keys=True):
    b = {"id": "b1", "kind": "retail", "name": "テストビル",
         "footprint": [[0, 0], [40, 0], [40, 30], [0, 30], [0, 0]],
         "levels": 6, "below": 1, "height": 21.0, "base": -3.5, "depth": 24.5,
         "cx": 20.0, "cy": 15.0}
    if with_terrain_keys:
        b["gz"] = 4.2
        b["floorH"] = 3.6
    return json.dumps({"meta": {"floor_height": 3.5, "origin_latlon": [35.6595, 139.70062]},
                       "buildings": [b],
                       "roads": [{"klass": "primary", "layer": 0,
                                  "g": [[0, 0], [120, 40]]}],
                       "rails": [{"name": "JR", "kind": "rail", "z": 0.0,
                                  "g": [[0, 0], [50, 0]]},
                                 {"name": "地下鉄", "kind": "subway", "z": -8.0,
                                  "g": [[0, 0], [50, 0]]}],
                       "pois": []}, ensure_ascii=False)


def _tracks_json():
    return json.dumps({"meta": {"nSteps": 2, "step_minutes": 10, "start_min": 420,
                                "floor_height": 3.5},
                       "agents": [{"id": 0, "name": "a", "visitor": False,
                                   "occupation": "?", "age": 20, "gender": "?"}],
                       "ids": [0], "positions": [[[0, 0, 0]], [[0, 0, 1001]]],
                       "moves": [[None], [None]],
                       "traffic": [{"n": 0, "segs": []}] * 2,
                       "sim_min": [420, 430]}, ensure_ascii=False)


def _terrain_json():
    import base64

    import numpy as np
    nx = ny = 8
    H = np.zeros((ny, nx), dtype="<i2")
    return json.dumps({"x0": -100.0, "y0": -100.0, "cell_m": 20.0, "nx": nx, "ny": ny,
                       "quant": 0.1,
                       "heights_b64": base64.b64encode(H.tobytes()).decode("ascii")})


def test_html_contains_foot_anchor_and_ground_formulas():
    html = MV3.build_html("t", _scene_json(), _tracks_json())
    # 足元アンカー + 表示倍率スライダー
    assert "footY(x, y, w) + AG_LIFT" in html
    assert 'id="agSize"' in html and 'id="agSizeV"' in html
    assert "function setAgentScale(" in html
    assert "const AG_R = 0.45, AG_BODY = 0.9;" in html
    # upOf は建物参照 + floor ガード
    assert "Math.max(0, Math.min(99, floor))" in html
    assert "floorHOf(b)" in html
    # 地上線路は地表基準(絶対 0.6 ではない)
    assert "groundAt(p[0], p[1]) + (r.z||0) + 0.6" in html
    # 押出しは depth + スカート
    assert "b.depth || b.height || FH" in html
    assert "const BLD_SKIRT = 3.0;" in html


def test_html_terrain_drape_shares_geometry_and_hides_grid():
    html = MV3.build_html("t", _scene_json(), _tracks_json(),
                          terrain_json=_terrain_json())
    # 旧: 別平面を 240 セグメントで作り直していた → その痕跡が無いこと
    assert "Math.min(240" not in html
    # 新: 地形 geometry をそのまま OSM マテリアルで描く
    assert "new THREE.Mesh(geo, dmat)" in html
    assert "polygonOffset = true" in html
    assert "uv.setXY(k," in html
    # 地形注入時は GridHelper 常時 OFF
    assert "gridHelper.visible = false;" in html
    assert "!osmOn && !TERRAIN" in html
    # 道路は地形セル幅で再分割
    assert "const SUB = TERRAIN ? TERRAIN.cell : 0;" in html


def test_html_underground_layer_defaults_off_and_clips():
    html = MV3.build_html("t", _scene_json(), _tracks_json(),
                          terrain_json=_terrain_json(), has_extras=True)
    assert '<input type="checkbox" id="lyUgai"> 地下街' in html   # checked ではない
    assert '<input type="checkbox" id="lyBridge" checked>' in html  # 歩道橋は従来どおり
    assert "meshOf(ex.ubld, 0x6f7fa8, 0.35, true)" in html          # 地表クリップ ON
    assert "meshOf(ex.brid, 0xb8bec7, 1.0, false)" in html          # 橋はクリップしない


def test_html_without_terrain_keeps_flat_behavior():
    """地形なしラン: 地形注入が無い = groundAt≡0 で従来動作(平面ドレープのまま)。"""
    html = MV3.build_html("t", _scene_json(with_terrain_keys=False), _tracks_json())
    assert "setupTerrain" not in html
    assert 'id="terrain-data"' not in html
    # OSM 平面は従来の PlaneGeometry のまま作られる
    assert "const geo = new THREE.PlaneGeometry(wxR-wxL, wyT-wyB);" in html
