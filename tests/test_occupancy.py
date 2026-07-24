"""scripts/build_occupancy.py と viewer 🏙在館タブ(B8)の検証。

検収条件:
- build_occupancy の占有系列が L1 の carry-forward と整合(合成ランで検算)。
- 屋内(space_move)有=階内トラッキング / 無=建物粒度のみへ正直に縮退(meta.indoor)。
- occupancy.parquet が有る時だけ build_data が occupancy キーを埋め、🏙在館タブが出る。
  無ければキー無し=旧ランと同じ(バイト不変・タブ無し)。
- 決定論: 同一入力で rows が完全一致。

全経路 mock/合成データのみ(実 LLM 不使用)。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "viz"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import make_viewer as mv          # noqa: E402
import build_occupancy as bo      # noqa: E402


# --------------------------------------------------------------------------- 合成 L1
def _events(scenario: str = "indoor"):
    """(step, agent, kind, payload) の列。indoor=space_move あり / coarse=enter のみ。"""
    rows = []

    def ev(step, aid, kind, payload):
        rows.append({"step": step, "agent_id": aid, "kind": kind,
                     "sim_min": 420 + step * 10,
                     "x": 0.0, "y": 0.0,
                     "payload": json.dumps(payload, ensure_ascii=False),
                     "rng_stream": "", "llm_call_id": ""})

    n_steps = 10
    for step in range(n_steps):
        for a in range(3):
            ev(step, a, "arrive", {"name": "路上", "node": "n1"})
    ev(1, 0, "enter_building", {"building": "b1", "floor": 4})
    if scenario == "indoor":
        ev(2, 0, "space_move", {"building": "b1", "floor": 4, "to_zone": 1})
    ev(6, 0, "exit_building", {})
    ev(3, 1, "enter_building", {"building": "b1", "floor": 2})
    ev(2, 2, "enter_building", {"building": "b2", "floor": 1})
    ev(5, 2, "exit_area", {})
    return rows, n_steps


# --------------------------------------------------------------------------- 1. carry-forward 検算
def test_occupancy_rows_carry_forward():
    events, n_steps = _events("indoor")
    rows, has_sm = bo.occupancy_rows(events, n_steps)
    assert has_sm is True                       # space_move あり
    cell = {(r["building"], r["floor"], r["step"]): r["n"] for r in rows}
    # agent0 は step1〜5 まで b1/4F 在館(step6 で exit)
    for s in range(1, 6):
        assert cell[("b1", 4, s)] == 1
    assert ("b1", 4, 6) not in cell             # exit_building 適用後は非在館
    # agent1 は step3〜9 まで b1/2F
    for s in range(3, 10):
        assert cell[("b1", 2, s)] == 1
    # agent2 は step2〜4 まで b2/1F(step5 で exit_area)
    assert cell[("b2", 1, 2)] == 1 and cell[("b2", 1, 4)] == 1
    assert ("b2", 1, 5) not in cell
    # n>0 のセルのみ・(building,floor,step) 昇順
    assert all(r["n"] > 0 for r in rows)
    assert rows == sorted(rows, key=lambda r: (r["building"], r["floor"], r["step"]))


def test_occupancy_rows_deterministic():
    events, n_steps = _events("indoor")
    r1, _ = bo.occupancy_rows(events, n_steps)
    r2, _ = bo.occupancy_rows(events, n_steps)
    assert r1 == r2


def test_occupancy_coarse_degrades_but_keeps_buildings():
    """space_move 無し=建物粒度のみに縮退。だが建物在館 rows は残る(正直な縮退)。"""
    events, n_steps = _events("coarse")
    rows, has_sm = bo.occupancy_rows(events, n_steps)
    assert has_sm is False
    blds = {r["building"] for r in rows}
    assert blds == {"b1", "b2"}                 # 建物在館は取れる(階は enter の階)


# --------------------------------------------------------------------------- 2. 書き出し + meta
def _write_run(tmp_path: Path, name: str, scenario: str) -> Path:
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True, exist_ok=True)
    map_path = tmp_path / f"{name}_map.json"
    city = {"buildings": [
                {"id": "b1", "name": "ビルA", "kind": "office", "levels": 6,
                 "footprint": [[0, 0], [120, 0], [120, 90], [0, 90]]},
                {"id": "b2", "name": "ビルB", "kind": "retail", "levels": 3,
                 "footprint": [[200, 0], [260, 0], [260, 80], [200, 80]]}],
            "nodes": [{"id": "n1", "x": 0, "y": 0, "name": "N"}],
            "edges": [{"klass": "footway", "layer": 0, "geometry": [[0, 0], [1, 1]]}],
            "pois": [], "railways": [], "meta": {"origin_latlon": [35.66, 139.70]}}
    map_path.write_text(json.dumps(city, ensure_ascii=False), encoding="utf-8")
    (run_dir / "config.yaml").write_text(
        "world:\n"
        f"  map: {map_path.as_posix()}\n"
        "transit:\n"
        "  file: data/__no_transit_for_test__.json\n"
        "indoor:\n"
        "  floor_layouts_path: data/floor_layouts.json\n",
        encoding="utf-8")
    agents = [{"id": i, "name": f"住民{i}", "age": 30, "gender": "男",
               "occupation": "会社員", "visitor": False,
               "has_bicycle": False, "has_car": False} for i in range(3)]
    (run_dir / "agents.json").write_text(json.dumps(agents, ensure_ascii=False),
                                         encoding="utf-8")
    events, _ = _events(scenario)
    fields = [("step", pa.int32()), ("sim_min", pa.int32()),
              ("agent_id", pa.int32()), ("kind", pa.string()),
              ("x", pa.float32()), ("y", pa.float32()), ("payload", pa.string()),
              ("rng_stream", pa.string()), ("llm_call_id", pa.string())]
    cols = {nm: [r[nm] for r in events] for nm, _ in fields}
    pq.write_table(pa.table(cols, schema=pa.schema(fields)),
                   run_dir / "l1_events.parquet")
    return run_dir


def test_write_occupancy_produces_parquet_and_meta(tmp_path):
    rd = _write_run(tmp_path, "occ_write", "indoor")
    meta = bo.write_occupancy(rd, rd / "occupancy.parquet")
    assert (rd / "occupancy.parquet").exists()
    assert (rd / "occupancy_meta.json").exists()
    assert meta["indoor"] is True and meta["n_buildings"] == 2
    tbl = pq.read_table(rd / "occupancy.parquet")
    assert set(tbl.column_names) == {"building", "floor", "step", "n"}


def test_write_occupancy_coarse_meta_indoor_false(tmp_path):
    rd = _write_run(tmp_path, "occ_coarse", "coarse")
    meta = bo.write_occupancy(rd, rd / "occupancy.parquet")
    assert meta["indoor"] is False               # 屋内トラッキング無し=建物粒度


# --------------------------------------------------------------------------- 3. viewer 埋め込み
def test_build_data_embeds_occupancy_only_when_parquet(tmp_path):
    rd = _write_run(tmp_path, "occ_embed", "indoor")
    # parquet 生成前: occupancy キー無し(=旧ランと同じ)
    assert "occupancy" not in mv.build_data(rd, include_traffic=False)
    bo.write_occupancy(rd, rd / "occupancy.parquet")
    data = mv.build_data(rd, include_traffic=False)
    assert "occupancy" in data
    Oc = data["occupancy"]
    assert Oc["indoor"] is True and Oc["nBld"] == 2
    b1 = next(b for b in Oc["buildings"] if b["id"] == "b1")
    assert b1["series"][4] >= 1                   # step4 に b1 在館あり
    assert "4" in b1["floors"] and "2" in b1["floors"]   # 階別(indoor)


def test_load_occupancy_coarse_no_floor_breakdown(tmp_path):
    rd = _write_run(tmp_path, "occ_embed_coarse", "coarse")
    bo.write_occupancy(rd, rd / "occupancy.parquet")
    Oc = mv.build_data(rd, include_traffic=False)["occupancy"]
    assert Oc["indoor"] is False
    for b in Oc["buildings"]:
        assert b["floors"] == {}                  # 建物粒度のみ=階別ヒートマップ無し


def test_occupancy_tab_injected_only_with_parquet(tmp_path):
    rd = _write_run(tmp_path, "occ_html", "indoor")

    def _dash():
        r = subprocess.run([sys.executable, str(REPO_ROOT / "viz" / "make_viewer.py"), str(rd)],
                           cwd=REPO_ROOT, capture_output=True,
                           env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
        assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
        return (rd / "dashboard.html").read_text(encoding="utf-8")

    dash_before = _dash()
    assert 'data-tab="occ"' not in dash_before and "function occRender" not in dash_before
    bo.write_occupancy(rd, rd / "occupancy.parquet")
    dash_after = _dash()
    assert 'data-tab="occ"' in dash_after and "function occRender" in dash_after
    for tok in ("__LENS_TABS__", "__LENS_JS__"):
        assert tok not in dash_after
