"""B6: viz/make_viewer3d.py の屋内オーバレイ(接近フェード+フロア板+実座標)の検収。

- 旧ラン(space_move/samples 無し)= 生成 HTML がバイト同一(注入は完全ゲート)。
- 屋内データ有 = 注入マーカー(indoor-data / lyIndoor / floorLayout3d / フロア板 / フェード)。
- n_override パリティ: 埋め込み floorLayout3d の JS 移植(Python 参照)が world.vision.building_layout
  (= sim 側の間取り正典)と区画矩形レベルで一致(override 経路=shops/Σzone_mix、pool 経路の双方)。
- _ensure_indoor ゲート: samples parquet / space_move の有無で注入 or None。
- 実座標フレーム写像: indoor_tracks_samples.parquet の (agent_id,t_s,x,y) が frame 別・最大 t_s 採用で畳まれる。
- 埋め込みデータ整合: floor_layouts の建物 / positions.frames が JSON に載る。
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from society.world import vision

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MV3 = _load("make_viewer3d", "viz/make_viewer3d.py")


# --------------------------------------------------------------------------- #
# 共通フィクスチャ
# --------------------------------------------------------------------------- #
def _scene(buildings=None):
    if buildings is None:
        buildings = [{
            "id": "b1", "kind": "retail", "name": "渋谷スクランブルスクエア",
            "footprint": [[0, 0], [48, 0], [48, 32], [0, 32], [0, 0]],
            "levels": 10, "below": 2, "height": 35.0, "base": -7.0,
            "cx": 24.0, "cy": 16.0}]
    return json.dumps({"meta": {"floor_height": 3.5, "origin_latlon": None},
                       "buildings": buildings, "roads": [], "rails": [],
                       "pois": []}, ensure_ascii=False)


def _tracks(n=3, ids=(0,), positions=None):
    if positions is None:
        positions = [[[0, 0, 0]] for _ in range(n)]
    agents = [{"id": i, "name": f"a{i}", "visitor": False, "occupation": "?",
               "age": 20, "gender": "?"} for i in ids]
    return json.dumps({"meta": {"nSteps": n, "step_minutes": 10, "start_min": 420,
                                "floor_height": 3.5},
                       "agents": agents, "ids": list(ids),
                       "positions": positions,
                       "moves": [[None] * len(ids) for _ in range(n)],
                       "traffic": [{"n": 0, "segs": []}] * n,
                       "sim_min": [420 + 10 * s for s in range(n)]},
                      ensure_ascii=False)


_FL_SPECS = [{"match": ["渋谷スクランブルスクエア", "スクランブルスクエア"],
              "floors": [{"f": 1, "use": "food", "shops": 12},
                         {"f": 2, "use": "fashion", "shops": 11},
                         {"f": 4, "use": "restaurant",
                          "zone_mix": {"restaurant": 15, "fashion": 5,
                                       "lifestyle": 4, "office": 1}}]}]


def _indoor_json(with_pos=True):
    payload = {"floor_layouts": {"buildings": _FL_SPECS}}
    if with_pos:
        payload["positions"] = {"frames": {"1": [[0, 12.3, 8.4]]}}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


# --------------------------------------------------------------------------- #
# 1. バイト同一(注入ゲート)
# --------------------------------------------------------------------------- #
def test_html_byte_identical_without_indoor():
    scene, tracks = _scene(), _tracks()
    plain = MV3.build_html("t", scene, tracks)
    none_arg = MV3.build_html("t", scene, tracks, indoor_json=None)
    assert plain == none_arg


def test_html_byte_identical_all_none():
    """他レイヤ(notable 等)と同居しても indoor None は完全ゲート。"""
    scene, tracks = _scene(), _tracks()
    a = MV3.build_html("t", scene, tracks, notable_json=None, indoor_json=None)
    b = MV3.build_html("t", scene, tracks)
    assert a == b


# --------------------------------------------------------------------------- #
# 2. 注入マーカー
# --------------------------------------------------------------------------- #
def test_html_injects_indoor_markers():
    scene, tracks = _scene(), _tracks()
    html = MV3.build_html("t", scene, tracks, indoor_json=_indoor_json())
    for m in ('id="indoor-data"', 'id="lyIndoor"', "INDOOR_SKIP",
              "floorLayout3d", "window.__indoorXY", "ensurePlates",
              "updateProximity", "buildShells"):
        assert m in html, f"marker {m} not injected"
    # 一意アンカーは 1 回だけ
    assert html.count('id="indoor-data"') == 1
    assert html.count('id="lyIndoor"') == 1
    # scene/tracks は従来どおり健在
    assert 'id="scene-data"' in html and 'id="tracks-data"' in html
    # プレーン版より大きい(差分がある)
    assert len(html) > len(MV3.build_html("t", scene, tracks))


def test_placeagents_patched_for_indoor():
    scene, tracks = _scene(), _tracks()
    html = MV3.build_html("t", scene, tracks, indoor_json=_indoor_json())
    # 屋内分岐が placeAgents に入る(実座標/推定区画)
    assert "window.__indoorXY(i, w, t)" in html
    # フェードは X 線と min 合成(どちらか透明なら透明)
    assert "Math.min(base, fr)" in html


def test_injected_indoor_json_script_safe():
    scene, tracks = _scene(), _tracks()
    evil = json.dumps({"floor_layouts": {"buildings": [
        {"match": ["</script><b>x"], "floors": [{"f": 1, "use": "food"}]}]}},
        ensure_ascii=False)
    html = MV3.build_html("t", scene, tracks, indoor_json=evil)
    assert "</script><b>x" not in html
    assert "<\\/script><b>x" in html


# --------------------------------------------------------------------------- #
# 3. n_override パリティ(floorLayout3d ⇄ vision.building_layout)
# --------------------------------------------------------------------------- #
_U32 = 0xFFFFFFFF


def _fnv(s):
    h = 2166136261
    for ch in str(s):
        h ^= ord(ch)
        h = (h * 16777619) & _U32
    return h & _U32


def _imul(x, y):
    return ((x & _U32) * (y & _U32)) & _U32


def _rng(seed):
    a = seed & _U32

    def nxt():
        nonlocal a
        a = (a + 0x6D2B79F5) & _U32
        t = _imul(a ^ (a >> 15), 1 | a)
        t = ((t + _imul(t ^ (t >> 7), 61 | t)) & _U32) ^ t
        return ((t ^ (t >> 14)) & _U32) / 4294967296.0
    return nxt


def _cols(a0, a1, n, rng):
    if n <= 0:
        return []
    w = []
    s = 0.0
    for _ in range(n):
        v = 0.7 + rng() * 0.7
        w.append(v)
        s += v
    out = []
    p = a0
    for v in w:
        seg = (a1 - a0) * v / s
        out.append((p, p + seg))
        p += seg
    return out


_POOL_LEN = {"fashion": 6, "beauty": 5, "food": 6, "restaurant": 6,
             "lifestyle": 6, "office": 5, "shop": 4, "hall": 3, "theatre": 3,
             "hotel": 3, "station": 4, "park": 3, "nightlife": 3, "service": 3,
             "attraction": 3, "education": 3, "generic": 3}


def _kind_use(k):
    return ("office" if k == "office" else "station" if k == "station"
            else "shop" if k == "retail" else "generic")


def _n_override(spec):
    """埋め込み JS _nOverride と同一(= sim indoor._build と同一: shops > Σzone_mix > None)。"""
    if not spec:
        return None
    if spec.get("shops"):
        return int(spec["shops"])
    if spec.get("zone_mix"):
        return sum(int(v) for v in spec["zone_mix"].values())
    return None


def _js_floor_layout(b, f, nov):
    """viz/make_viewer3d.py に埋め込んだ floorLayout3d の Python 参照移植(区画矩形のみ)。"""
    fp = b["footprint"]
    xs = [p[0] for p in fp]
    ys = [p[1] for p in fp]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return None
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    rng = _rng((_fnv(b.get("id") or b.get("name") or "b")
                + (f + 50) * 2654435761) & _U32)
    if nov is not None:
        n = min(12, int(nov))
    else:
        use = _kind_use(b.get("kind", "generic"))
        pool_n = _POOL_LEN.get(use, _POOL_LEN["generic"])
        n = 2 + int(rng() * min(4, pool_n - 1))
        used = set()
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
    n_a = -(-n // 2)
    n_b = n - n_a
    zones = []
    if horiz:
        yc0, yc1 = cy - band, cy + band
        for c in _cols(x0, x1, n_a, rng):
            zones.append((c[0], yc1, c[1], y1))
        for c in _cols(x0, x1, n_b, rng):
            zones.append((c[0], y0, c[1], yc0))
    else:
        xc0, xc1 = cx - band, cx + band
        for c in _cols(y0, y1, n_a, rng):
            zones.append((xc1, c[0], x1, c[1]))
        for c in _cols(y0, y1, n_b, rng):
            zones.append((x0, c[0], xc0, c[1]))
    return zones


def _approx(a, b, eps=1e-7):
    if a is None or b is None or len(a) != len(b):
        return False
    for za, zb in zip(a, b):
        for u, v in zip(za, zb):
            if abs(u - v) > eps:
                return False
    return True


def test_n_override_parity_with_vision():
    """埋め込み floorLayout3d(JS 移植)== world.vision.building_layout(sim 間取り正典)。"""
    b = {"id": "way/123", "name": "渋谷スクランブルスクエア", "kind": "retail",
         "footprint": [[0, 0], [48, 0], [48, 32], [0, 32], [0, 0]]}
    checked = 0
    for f, nov in [(1, 12), (2, 11), (4, 25), (1, None), (6, None), (3, 1), (5, 29)]:
        exp = vision.building_layout(b, f, n_override=nov)
        got = _js_floor_layout(b, f, nov)
        ez = [tuple(z) for z in exp["zones"]]
        assert _approx(ez, got), f"zone mismatch f={f} nov={nov}"
        assert len(ez) == len(got)
        checked += 1
    assert checked == 7


def test_n_override_derivation_matches_spec_rule():
    """shops > Σzone_mix > None の派生規則が sim indoor._build と一致(spec 3 種)。"""
    assert _n_override({"use": "food", "shops": 12}) == 12
    assert _n_override({"use": "restaurant",
                        "zone_mix": {"restaurant": 15, "fashion": 5,
                                     "lifestyle": 4, "office": 1}}) == 25
    assert _n_override({"use": "office"}) is None


def test_n_override_parity_pool_path_office():
    """spec 無し階(pool 経路)も vision と一致(POI 無しの決定論生成)。"""
    b = {"id": "way/999", "name": "", "kind": "office",
         "footprint": [[0, 0], [30, 0], [30, 50], [0, 50], [0, 0]]}
    for f in (1, 3, 5):
        exp = [tuple(z) for z in vision.building_layout(b, f)["zones"]]
        got = _js_floor_layout(b, f, None)
        assert _approx(exp, got), f"pool-path mismatch f={f}"


def test_full_spec_chain_parity():
    """floor_layouts spec → n_override → 区画矩形が sim と一致(実 spec の全階)。"""
    b = {"id": "way/123", "name": "渋谷スクランブルスクエア", "kind": "retail",
         "footprint": [[0, 0], [48, 0], [48, 32], [0, 32], [0, 0]]}
    for spec in _FL_SPECS[0]["floors"]:
        f = spec["f"]
        nov = _n_override(spec)
        exp = [tuple(z) for z in vision.building_layout(b, f, n_override=nov)["zones"]]
        got = _js_floor_layout(b, f, nov)
        assert _approx(exp, got), f"spec-chain mismatch f={f} nov={nov}"


# --------------------------------------------------------------------------- #
# 4. _ensure_indoor ゲート + 実座標フレーム写像
# --------------------------------------------------------------------------- #
def _write_l1(run_dir: Path, rows: list):
    (run_dir / "scene3d").mkdir(parents=True, exist_ok=True)
    schema = pa.schema([
        ("step", pa.int32()), ("sim_min", pa.int32()), ("agent_id", pa.int32()),
        ("kind", pa.string()), ("x", pa.float32()), ("y", pa.float32()),
        ("payload", pa.string()), ("rng_stream", pa.string()),
        ("llm_call_id", pa.string())])
    cols = {k: [r[k] for r in rows] for k in schema.names}
    pq.write_table(pa.table(cols, schema=schema), run_dir / "l1_events.parquet")


def _ev(step, aid, kind, x=0.0, y=0.0, payload=None):
    return {"step": step, "sim_min": 420 + step * 10, "agent_id": aid, "kind": kind,
            "x": float(x), "y": float(y),
            "payload": json.dumps(payload or {}, ensure_ascii=False),
            "rng_stream": "", "llm_call_id": ""}


def _write_samples(run_dir: Path, rows: list):
    run_dir.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([
        ("agent_id", pa.int32()), ("t_s", pa.float32()), ("building", pa.string()),
        ("floor", pa.int32()), ("x", pa.float32()), ("y", pa.float32()),
        ("zone", pa.int32())])
    cols = {k: [r[k] for r in rows] for k in schema.names}
    pq.write_table(pa.table(cols, schema=schema), run_dir / "indoor_tracks_samples.parquet")


def test_ensure_indoor_none_when_no_signal(tmp_path):
    """space_move 無し・samples 無し = None(= build_html は従来とバイト同一の経路)。"""
    run = tmp_path / "plain"
    _write_l1(run, [_ev(s, 0, "move_segment", 1, 1, {"pts": [[0, 0], [1, 1]]})
                    for s in range(4)])
    assert MV3._ensure_indoor(run, _tracks(4)) is None
    assert not (run / "scene3d" / "indoor_overlay.json").exists()


def test_ensure_indoor_gate_on_space_move(tmp_path):
    run = tmp_path / "sm"
    _write_l1(run, [_ev(0, 0, "enter_building", 5, 5, {"building": "b1", "floor": 1}),
                    _ev(1, 0, "space_move", 5, 5,
                        {"building": "b1", "floor": 1, "from_zone": 0, "to_zone": 2})])
    txt = MV3._ensure_indoor(run, _tracks(2))
    assert txt is not None
    data = json.loads(txt)
    assert "floor_layouts" in data and data["floor_layouts"]["buildings"]
    # sidecar 透明性
    assert (run / "scene3d" / "indoor_overlay.json").exists()


def test_ensure_indoor_gate_on_samples(tmp_path):
    run = tmp_path / "sm2"
    (run / "scene3d").mkdir(parents=True)
    _write_samples(run, [
        {"agent_id": 0, "t_s": 605.0, "building": "b1", "floor": 1,
         "x": 10.0, "y": 6.0, "zone": 1}])
    txt = MV3._ensure_indoor(run, _tracks(3, ids=(0,)))
    assert txt is not None
    data = json.loads(txt)
    assert "positions" in data


def test_positions_frame_mapping(tmp_path):
    """t_s→frame(=step, stride=1)・frame 内は最大 t_s 採用・agent_id→idx。"""
    run = tmp_path / "pos"
    (run / "scene3d").mkdir(parents=True)
    _write_samples(run, [
        # agent 7 (idx1), step1: 2 サンプル → 最大 t_s(650)を採用
        {"agent_id": 7, "t_s": 610.0, "building": "b1", "floor": 1,
         "x": 1.0, "y": 1.0, "zone": 0},
        {"agent_id": 7, "t_s": 650.0, "building": "b1", "floor": 1,
         "x": 9.9, "y": 8.8, "zone": 2},
        # agent 3 (idx0), step2
        {"agent_id": 3, "t_s": 1205.0, "building": "b1", "floor": 2,
         "x": 4.4, "y": 5.5, "zone": 1},
        # ids に無い agent は捨てる
        {"agent_id": 99, "t_s": 620.0, "building": "b1", "floor": 1,
         "x": 0.0, "y": 0.0, "zone": 0}]),
    pos = MV3._indoor_positions_from_samples(
        run / "indoor_tracks_samples.parquet", _tracks(4, ids=(3, 7)))
    frames = pos["frames"]
    assert frames["1"] == [[1, 9.9, 8.8]]        # idx1=agent7, 最大 t_s
    assert frames["2"] == [[0, 4.4, 5.5]]        # idx0=agent3
    assert "99" not in json.dumps(frames)        # 未知 agent は落ちる


def test_positions_stride_mapping(tmp_path):
    """step_stride=3: raw step9 → frame3 に写像・nSteps-1 でクランプ。"""
    run = tmp_path / "stride"
    (run / "scene3d").mkdir(parents=True)
    tj = json.loads(_tracks(5, ids=(0,)))
    tj["meta"]["step_stride"] = 3
    _write_samples(run, [
        {"agent_id": 0, "t_s": 9 * 600 + 10.0, "building": "b1", "floor": 1,
         "x": 2.0, "y": 3.0, "zone": 0}])
    pos = MV3._indoor_positions_from_samples(
        run / "indoor_tracks_samples.parquet", json.dumps(tj))
    assert list(pos["frames"].keys()) == ["3"]   # 9 // 3
    assert pos["frames"]["3"] == [[0, 2.0, 3.0]]


# --------------------------------------------------------------------------- #
# 5. フル注入(_ensure_indoor → build_html)で埋め込みデータが載る
# --------------------------------------------------------------------------- #
def test_full_pipeline_embeds_layout_and_positions(tmp_path):
    run = tmp_path / "full"
    (run / "scene3d").mkdir(parents=True)
    _write_l1(run, [_ev(1, 0, "space_move", 5, 5,
                        {"building": "b1", "floor": 1, "from_zone": 0, "to_zone": 1})])
    _write_samples(run, [
        {"agent_id": 0, "t_s": 650.0, "building": "b1", "floor": 1,
         "x": 12.0, "y": 7.0, "zone": 1}])
    tracks = _tracks(3, ids=(0,),
                     positions=[[[0, 0, 0]], [[5, 5, 1001]], [[0, 0, 0]]])
    indoor_json = MV3._ensure_indoor(run, tracks)
    assert indoor_json is not None
    html = MV3.build_html("t", _scene(), tracks, indoor_json=indoor_json)
    # フロア板レイアウト spec が埋まる
    assert "渋谷スクランブルスクエア" in html
    # 実座標フレームが埋まる(agent idx0 の位置)
    assert '"frames"' in html
    assert 'id="indoor-data"' in html
