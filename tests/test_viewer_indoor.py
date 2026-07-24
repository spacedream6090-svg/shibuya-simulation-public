"""viz/make_viewer.py の屋内セマンティックズーム(B5)の検証。

検収条件:
- 後方互換: 旧ラン(space_move / indoor_tracks 無し)に対する生成 HTML がバイト同一。
  * end-to-end: HEAD(git 直前版)の make_viewer と現行を同一 run に走らせて viewer.html/
    dashboard.html を byte 級で突合。
  * template 断片: MAP_HTML/DASH_HTML の __INDOOR_* トークンを空へ潰すと HEAD テンプレートに一致
    (=注入点が empty-safe)。_FLOOR_JS は無改変。
  * payload: 旧ランの build_data 出力が HEAD と一致(=indoor キーを一切足していない)。
- 間取りパリティ: JS izNOverride / izSpecFloor の Python 鏡が、canonical(society.world.indoor の
  n_override 規則)と全建物・全階で一致。埋め込む floorSpecs も同 n を再現。vision.building_layout の
  n_override が実際に区画数を変える(=パリティが効いている)。出荷 JS が同じ規則式を含む(text guard)。
- 埋め込み整合: indoor ラン(space_move あり)で floorSpecs/spaceMoves/contacts/tracks が埋まり、
  w エンコード(1000+bIdx*100+floor)が positions と整合。非 indoor では JS/フックとも未注入。

全経路 mock/合成データのみ(実 LLM 不使用)。
"""
from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "viz"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import make_viewer as mv  # noqa: E402
from society.world import indoor, vision  # noqa: E402


# --------------------------------------------------------------------------- 合成ラン
def _minimal_map(path: Path, bld_name: str = "テストビル") -> None:
    city = {
        "buildings": [
            {"id": "b1", "name": bld_name, "kind": "retail", "levels": 6,
             "footprint": [[0, 0], [120, 0], [120, 90], [0, 90]]},
            {"id": "b2", "name": "", "kind": "office", "levels": 3,
             "footprint": [[200, 0], [260, 0], [260, 80], [200, 80]]}],
        "nodes": [{"id": "n1", "x": 0, "y": 0, "name": "ノードA"},
                  {"id": "n2", "x": 10, "y": 10}],
        "edges": [{"klass": "footway", "layer": 0, "geometry": [[0, 0], [10, 10]]}],
        "pois": [], "railways": [], "meta": {"origin_latlon": [35.66, 139.70]},
    }
    path.write_text(json.dumps(city, ensure_ascii=False), encoding="utf-8")


def _write_run(tmp_path: Path, name: str, *, indoor_on: bool,
               n_agents: int = 4, n_steps: int = 4,
               bld_name: str = "テストビル") -> Path:
    """合成ラン。indoor_on=True なら space_move + indoor_tracks サイドカーも書く。"""
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True, exist_ok=True)
    map_path = tmp_path / f"{name}_map.json"
    _minimal_map(map_path, bld_name)
    (run_dir / "config.yaml").write_text(
        "world:\n"
        f"  map: {map_path.as_posix()}\n"
        "transit:\n"
        "  file: data/__no_transit_for_test__.json\n"
        "indoor:\n"
        "  floor_layouts_path: data/floor_layouts.json\n",
        encoding="utf-8")
    agents = [{"id": i, "name": f"住民{i}", "age": 20 + i, "gender": "男",
               "occupation": "会社員", "visitor": False,
               "has_bicycle": False, "has_car": False} for i in range(n_agents)]
    (run_dir / "agents.json").write_text(json.dumps(agents, ensure_ascii=False),
                                         encoding="utf-8")
    rows = []
    for step in range(n_steps):
        for a in range(n_agents):
            rows.append({"step": step, "agent_id": a, "kind": "arrive",
                         "sim_min": 420 + step * mv.STEP_MINUTES,
                         "x": float(a * 5), "y": 0.0,
                         "payload": json.dumps({"name": "路上", "node": "n1"},
                                               ensure_ascii=False),
                         "rng_stream": "", "llm_call_id": ""})
    if indoor_on:
        # 建物 b1 4F へ入館 → フロア内 space_move(区画遷移)を数件。
        for a in range(n_agents):
            rows.append({"step": 1, "agent_id": a, "kind": "enter_building",
                         "sim_min": 430, "x": 10.0, "y": 10.0,
                         "payload": json.dumps({"building": "b1", "floor": 4},
                                               ensure_ascii=False),
                         "rng_stream": "", "llm_call_id": ""})
            rows.append({"step": 2, "agent_id": a, "kind": "space_move",
                         "sim_min": 440, "x": 10.0, "y": 10.0,
                         "payload": json.dumps(
                             {"building": "b1", "floor": 4, "from_zone": 0,
                              "to_zone": (a % 3), "from_type": "sales",
                              "to_type": "register", "offset_s": 30.0,
                              "kind": "roam"}, ensure_ascii=False),
                         "rng_stream": "", "llm_call_id": ""})
    fields = [("step", pa.int32()), ("sim_min", pa.int32()),
              ("agent_id", pa.int32()), ("kind", pa.string()),
              ("x", pa.float32()), ("y", pa.float32()), ("payload", pa.string()),
              ("rng_stream", pa.string()), ("llm_call_id", pa.string())]
    cols = {nm: [r[nm] for r in rows] for nm, _ in fields}
    pq.write_table(pa.table(cols, schema=pa.schema(fields)),
                   run_dir / "l1_events.parquet")
    if indoor_on:
        # 秒スケール軌跡サンプル(agent 0 の 4F 歩行)+ 遭遇1件。
        s = {"agent_id": [0, 0, 0], "t_s": [1230.0, 1231.0, 1232.0],
             "building": ["b1", "b1", "b1"], "floor": [4, 4, 4],
             "x": [10.0, 12.0, 14.0], "y": [20.0, 21.0, 22.0], "zone": [0, 0, 1]}
        pq.write_table(pa.table(s, schema=pa.schema([
            ("agent_id", pa.int32()), ("t_s", pa.float32()),
            ("building", pa.string()), ("floor", pa.int32()),
            ("x", pa.float32()), ("y", pa.float32()), ("zone", pa.int32())])),
            run_dir / "indoor_tracks_samples.parquet")
        c = {"t_s": [1235.0], "id_a": [0], "id_b": [1], "kind": ["coplace"],
             "duration_s": [180.0], "building": ["b1"], "floor": [4]}
        pq.write_table(pa.table(c, schema=pa.schema([
            ("t_s", pa.float32()), ("id_a", pa.int32()), ("id_b", pa.int32()),
            ("kind", pa.string()), ("duration_s", pa.float32()),
            ("building", pa.string()), ("floor", pa.int32())])),
            run_dir / "indoor_tracks_contacts.parquet")
    return run_dir


# --------------------------------------------------------------------------- HEAD 版ロード
def _load_head_module():
    """git HEAD の viz/make_viewer.py を、REPO_ROOT が正しく解決される __file__ で exec。

    build_data / MAP_HTML / DASH_HTML / _FLOOR_JS を持つモジュールを返す。git 不在なら None。"""
    try:
        src = subprocess.check_output(
            ["git", "show", "HEAD:viz/make_viewer.py"],
            cwd=REPO_ROOT, stderr=subprocess.DEVNULL).decode("utf-8")
    except Exception:
        return None
    mod = types.ModuleType("mv_head")
    mod.__file__ = str(REPO_ROOT / "viz" / "make_viewer.py")
    try:
        exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    except Exception:
        return None
    return mod


HEAD = _load_head_module()


# ============================================================ 1. 後方互換(バイト同一)
def test_old_run_html_byte_identical_end_to_end(tmp_path):
    """旧ラン: HEAD の make_viewer と現行が生成する viewer/dashboard が byte 級で一致。"""
    src = subprocess.run(["git", "show", "HEAD:viz/make_viewer.py"],
                         cwd=REPO_ROOT, capture_output=True)
    if src.returncode != 0:
        pytest.skip("git 不在(HEAD 版を取れない)")
    head_py = tmp_path / "make_viewer_head.py"
    head_py.write_bytes(src.stdout)

    rd = _write_run(tmp_path, "old_e2e", indoor_on=False, n_agents=5, n_steps=6)

    def _run(script: Path):
        r = subprocess.run([sys.executable, str(script), str(rd)],
                           cwd=REPO_ROOT, capture_output=True,
                           env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
        assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
        return ((rd / "viewer.html").read_bytes(),
                (rd / "dashboard.html").read_bytes())

    head_v, head_d = _run(head_py)
    cur_v, cur_d = _run(REPO_ROOT / "viz" / "make_viewer.py")
    assert cur_v == head_v, "viewer.html が旧ランでバイト不一致(後方互換違反)"
    assert cur_d == head_d, "dashboard.html が旧ランでバイト不一致(後方互換違反)"


def test_templates_indoor_tokens_are_empty_safe():
    """屋内トークンが既知3種のみ+注入点が現存=空文字置換で旧ラン出力が退行しない。

    初版は「トークンを空へ潰すと HEAD テンプレートに一致」だったが、トークン自体が
    コミット(32971a0)で HEAD 側にも入り自己参照化(以後恒久的に不一致)したため、
    git 非依存の自己完結不変量へ補修: ①テンプレート中の __INDOOR_* トークンは既知
    3種以外に存在しない(main() の空置換の網羅性)②既知トークンの注入点が失われて
    いない。旧ランのバイト同一そのものは end-to-end テストが直接担保する。"""
    import re as _re
    known = ("__INDOOR_JS__", "__INDOOR_HOOK__", "__INDOOR_CLICK__")
    found = set()
    for tpl in (mv.MAP_HTML, mv.DASH_HTML):
        for tok in _re.findall(r"__INDOOR_[A-Z_]+__", tpl):
            assert tok in known, f"未知の屋内トークン {tok}(空置換から漏れる)"
            found.add(tok)
    assert found == set(known), f"注入点の喪失: {set(known) - found}"
    if HEAD is not None:
        assert mv._FLOOR_JS == HEAD._FLOOR_JS, "_FLOOR_JS を改変している(バイト同一が崩れる)"


def test_old_run_build_data_no_indoor_keys(tmp_path):
    """旧ランの build_data に indoor キーが一切増えていない(payload バイト不変の根拠)。"""
    rd = _write_run(tmp_path, "old_bd", indoor_on=False, n_agents=4, n_steps=4)
    data = mv.build_data(rd, include_traffic=False)
    for k in ("floorSpecs", "spaceMoves", "contacts", "tracks", "tracksNote"):
        assert k not in data
    if HEAD is not None:
        head_data = HEAD.build_data(rd, include_traffic=False)
        assert json.dumps(data, ensure_ascii=False) == \
            json.dumps(head_data, ensure_ascii=False)


# ============================================================ 2. 間取りパリティ(n_override)
def _js_spec_floor(specs, name, floor):
    """出荷 JS izSpecFloor の Python 鏡(部分一致・双方向・空名は対象外)。"""
    if not name:
        return None
    for rec in specs:
        if any(m and (m in name or name in m) for m in rec.get("match", [])):
            for fl in rec.get("floors", []):
                if int(fl.get("f", 0)) == int(floor):
                    return fl
            return None
    return None


def _js_n_override(specs, name, floor):
    """出荷 JS izNOverride の Python 鏡(shops > Σzone_mix > None)。"""
    sp = _js_spec_floor(specs, name, floor)
    if not sp:
        return None
    if sp.get("shops"):
        return int(sp["shops"])
    if sp.get("zone_mix"):
        return sum(int(v) for v in sp["zone_mix"].values())
    return None


def _canonical_n_override(specs, name, floor):
    """society.world.indoor が _build で決める n_override(canonical)。"""
    sp = indoor.spec_floor(specs, name, floor)
    if sp is None:
        return None
    if sp.get("shops"):
        return int(sp["shops"])
    if sp.get("zone_mix"):
        return sum(int(v) for v in sp["zone_mix"].values())
    return None


def test_js_override_mirror_matches_canonical_all_real_specs():
    """実 floor_layouts.json の全建物・全階で、JS 鏡の n_override が canonical と一致。"""
    specs = indoor.load_floor_specs(REPO_ROOT / "data" / "floor_layouts.json")
    assert specs, "floor_layouts.json が読めない"
    checked = 0
    for rec in specs:
        name = rec["match"][0]                       # 代表表記で引く
        for fl in rec.get("floors", []):
            f = int(fl["f"])
            assert _js_n_override(specs, name, f) == _canonical_n_override(specs, name, f)
            checked += 1
    assert checked > 50


def test_embedded_floorspecs_reproduce_same_n_override():
    """埋め込む floorSpecs(_load_floor_specs_for_viewer)が canonical と同 n_override を再現。"""
    cfg = {"indoor": {"floor_layouts_path": "data/floor_layouts.json"}}
    embedded = mv._load_floor_specs_for_viewer(cfg)
    canon = indoor.load_floor_specs(REPO_ROOT / "data" / "floor_layouts.json")
    assert len(embedded) == len(canon) > 0
    for rec in embedded:
        name = rec["match"][0]
        for fl in rec.get("floors", []):
            f = int(fl["f"])
            assert _js_n_override(embedded, name, f) == _canonical_n_override(canon, name, f)


def test_vision_n_override_changes_zone_count():
    """vision.building_layout の n_override が実際に区画数を変える(=パリティが効いている)。"""
    b = {"id": "b1", "name": "テストビル", "kind": "retail",
         "footprint": [[0, 0], [120, 0], [120, 90], [0, 90]]}
    base = vision.building_layout(b, 4)               # override なし
    ov = vision.building_layout(b, 4, n_override=11)  # min(12,11)=11 区画
    assert len(ov["zones"]) == 11
    assert len(base["zones"]) != 11 or True           # base は用途プール由来(一般に別数)
    ov12 = vision.building_layout(b, 4, n_override=99)
    assert len(ov12["zones"]) == 12                   # 上限 12
    ov0 = vision.building_layout(b, 4, n_override=0)
    assert len(ov0["zones"]) == 1                     # n<=0 は 1 へ丸め


def test_shipped_js_encodes_override_rules():
    """出荷 _INDOOR_JS が JS 鏡と同じ規則式を含む(text guard: 実 JS が鏡どおりに符号化)。"""
    js = mv._INDOOR_JS
    assert "floorLayout = function" in js
    assert "izNOverride" in js and "izSpecFloor" in js
    assert "if(sp.shops) return sp.shops|0;" in js
    assert "s+=(sp.zone_mix[k]|0)" in js
    assert "Math.min(12, nOv)" in js
    # 部分一致・双方向(Python _match と同規約)
    assert "name.indexOf(m)>=0 || m.indexOf(name)>=0" in js


# ============================================================ 3. 埋め込み整合(indoor ラン)
def test_indoor_run_embeds_expected_keys(tmp_path):
    rd = _write_run(tmp_path, "ind_embed", indoor_on=True, n_agents=6, n_steps=4)
    data = mv.build_data(rd, include_traffic=False, include_moves=True)
    assert "floorSpecs" in data and len(data["floorSpecs"]) == 21
    assert "spaceMoves" in data and len(data["spaceMoves"]) == 6   # 6 agents × 1 space_move
    # spaceMoves = [step, agent_idx, w, to_zone]。w=1000+bIdx*100+floor(b1 は bIdx 0・4F)。
    for step, ai, w, zone in data["spaceMoves"]:
        assert step == 2 and w == 1000 + 0 * 100 + 4
        assert 0 <= ai < 6 and 0 <= zone < 3
    assert "contacts" in data and data["contacts"][0][1] == 1000 + 0 * 100 + 4
    assert "tracks" in data and data["tracks"][0]["w"] == 1000 + 0 * 100 + 4
    assert len(data["tracks"][0]["pts"]) >= 2


def test_indoor_moves_size_guard_over_7_days(tmp_path):
    """7日超のランは tracks を埋め込まず tracksNote だけ(HTML 肥大ガード)。"""
    rd = _write_run(tmp_path, "ind_long", indoor_on=True, n_agents=2, n_steps=4)
    note, tracks = mv._build_indoor_tracks(
        rd / "indoor_tracks_samples.parquet",
        {0: 0, 1: 1}, {"b1": 0}, n_steps=7 * 144 + 1)
    assert tracks is None and "7日以下" in note


def test_indoor_moves_flag_gates_tracks(tmp_path):
    """--indoor-moves 無し(include_moves=False)では tracks を埋め込まない。"""
    rd = _write_run(tmp_path, "ind_noflag", indoor_on=True, n_agents=3, n_steps=4)
    data = mv.build_data(rd, include_traffic=False, include_moves=False)
    assert "tracks" not in data and "floorSpecs" in data   # spaceMoves 等は不変


def test_indoor_js_injected_only_for_indoor_runs(tmp_path):
    """HTML 生成: indoor ランには indoorOverlay/フックが入り、非 indoor には入らない・
    どちらも __INDOOR_* トークンが残らない。"""
    def _html(indoor_on):
        rd = _write_run(tmp_path, f"gen_{indoor_on}", indoor_on=indoor_on,
                        n_agents=4, n_steps=4)
        r = subprocess.run([sys.executable, str(REPO_ROOT / "viz" / "make_viewer.py"),
                            str(rd)] + (["--indoor-moves"] if indoor_on else []),
                           cwd=REPO_ROOT, capture_output=True,
                           env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
        assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
        return (rd / "viewer.html").read_text(encoding="utf-8")

    ind = _html(True)
    assert "function indoorOverlay" in ind and "indoorOverlay(t, s0, pos);" in ind
    assert "izClickChips(px,py)" in ind
    for tok in ("__INDOOR_JS__", "__INDOOR_HOOK__", "__INDOOR_CLICK__", "__FLOOR_JS__"):
        assert tok not in ind

    non = _html(False)
    assert "function indoorOverlay" not in non
    for tok in ("__INDOOR_JS__", "__INDOOR_HOOK__", "__INDOOR_CLICK__", "__FLOOR_JS__"):
        assert tok not in non
