"""建物の実高さ配線(world.heights・A1 第67バッチ)のテスト。

方針(既存の鉄則を継承):
- OFF(既定): 高さ表を開かない・建物 dict に height_m / height_src が生えない・summary に
  building_heights キーが出ない・144step の L1 が ON と完全一致(属性付与のみで消費者ゼロ)。
- 生成器 scripts/build_heights.py は純関数 build_table を持ち、合成データだけで検証できる
  (data/plateau/ は .gitignore なので実データに依存したテストは実ファイルがあるときだけ走らせる)。
- 高さの定義は h = plateau_index.height - plateau_index.base(= zmax - zmin)。ground0 基準の
  頂部標高ではないことを固定する(坂上の建物で過大評価しないため)。
- 未照合の建物は levels x fallback_m_per_level の推定値で、height_src="levels" と出自を必ず区別。
  階数が無い建物には属性を付けない(欠測を捏造しない)。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from society.config import load_config
from society.engine.simulation import Simulation
from society.world.map import CityMap

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import build_heights as bh                                   # noqa: E402

HEIGHTS_FILE = REPO_ROOT / "data" / "building_heights_shibuya.json"


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _sim(tmp_path, name, n=15, steps=1, **ov):
    dot = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
           "run.seed=42", "model.backend=mock", "observer.snapshot_every=144"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


# ------------------------------------------------------------------ 生成器(純関数)
def test_build_table_uses_height_minus_base():
    """h = height - base(建物そのものの高さ)。ground0 基準の頂部標高 height ではない。"""
    matches = {"b1": {"gml_id": "g1"}, "b2": {"gml_id": "g2"}}
    index = [{"gml_id": "g1", "height": 59.282, "base": 12.34},
             {"gml_id": "g2", "height": 10.0, "base": 0.0}]
    heights, counts = bh.build_table(matches, index, round_m=0.1)
    assert heights["b1"] == {"h": 46.9, "src": "plateau"}      # 59.282 - 12.34 = 46.942
    assert heights["b2"] == {"h": 10.0, "src": "plateau"}
    assert counts["written"] == 2 and counts["matches_in"] == 2


def test_build_table_skips_broken_records():
    """index に居ない gml_id・高さが 0 以下のレコードは採用せず、件数だけ残す(捏造しない)。"""
    matches = {"b1": {"gml_id": "missing"}, "b2": {"gml_id": "g2"}, "b3": {"gml_id": "g3"}}
    index = [{"gml_id": "g2", "height": 5.0, "base": 5.0},      # h=0 → 不採用
             {"gml_id": "g3", "height": 8.0, "base": 1.0}]
    heights, counts = bh.build_table(matches, index)
    assert list(heights) == ["b3"]
    assert counts["skipped_no_index"] == 1 and counts["skipped_nonpositive"] == 1


def test_build_table_is_deterministic_and_sorted():
    """同じ入力から同じ dict(キー昇順)=決定論・乱数ゼロ。"""
    matches = {"b9": {"gml_id": "g9"}, "b1": {"gml_id": "g1"}}
    index = [{"gml_id": "g1", "height": 3.0, "base": 0.0},
             {"gml_id": "g9", "height": 30.0, "base": 1.0}]
    a, _ = bh.build_table(matches, index)
    b, _ = bh.build_table(matches, index)
    assert a == b and list(a) == ["b1", "b9"]


# ------------------------------------------------------------------ 生成物(実ファイル)
def test_generated_file_schema_and_contents():
    """コミット済みの高さ表が想定スキーマ・全件 plateau 出自・正の高さ。"""
    doc = json.loads(HEIGHTS_FILE.read_text(encoding="utf-8"))
    assert doc["meta"]["schema"] == bh.SCHEMA
    assert doc["meta"]["generated_by"] == "scripts/build_heights.py"
    assert doc["meta"]["params"]["round_m"] == 0.1
    heights = doc["heights"]
    assert doc["meta"]["counts"]["written"] == len(heights) > 3000
    assert all(v["src"] == "plateau" and v["h"] > 0 for v in heights.values())
    assert all(round(v["h"], 1) == v["h"] for v in heights.values()), "0.1m 丸めでない値がある"
    assert HEIGHTS_FILE.stat().st_size < 1_000_000, "生成物が想定(数百KB)より大きい"


# ------------------------------------------------------------------ CityMap への付与
def test_attach_heights_breakdown_on_default_map():
    """既定地図で plateau 実測 / levels 推定 の内訳が立ち、合計が全建物数に一致する。"""
    city = CityMap(REPO_ROOT / "data" / "shibuya_osm.json")
    stat = city.attach_heights(HEIGHTS_FILE, fallback_m_per_level=3.5)
    assert stat["n_plateau"] + stat["n_levels"] + stat["n_missing"] == stat["n_buildings"]
    assert stat["n_plateau"] > 0 and stat["n_levels"] > 0
    table = json.loads(HEIGHTS_FILE.read_text(encoding="utf-8"))["heights"]
    n_p = n_l = 0
    for b in city.buildings:
        assert "height_m" in b and b["height_m"] > 0
        if b["height_src"] == "plateau":
            assert b["height_m"] == round(float(table[b["id"]]["h"]), 1)
            n_p += 1
        else:
            assert b["height_src"] == "levels"
            assert b["height_m"] == round(int(b["levels"]) * 3.5, 1)
            n_l += 1
    assert (n_p, n_l) == (stat["n_plateau"], stat["n_levels"])
    assert city.building_height_m(city.buildings[0]["id"]) == city.buildings[0]["height_m"]


def test_fallback_scales_levels_estimate(tmp_path):
    """fallback_m_per_level を変えると levels 由来の推定値だけが変わる(実測値は不変)。"""
    city_a = CityMap(REPO_ROOT / "data" / "shibuya_osm.json")
    city_a.attach_heights(HEIGHTS_FILE, fallback_m_per_level=3.5)
    city_b = CityMap(REPO_ROOT / "data" / "shibuya_osm.json")
    city_b.attach_heights(HEIGHTS_FILE, fallback_m_per_level=4.0)
    changed = 0
    for a, b in zip(city_a.buildings, city_b.buildings):
        if a["height_src"] == "plateau":
            assert a["height_m"] == b["height_m"]
        else:
            assert b["height_m"] == round(int(a["levels"]) * 4.0, 1)
            changed += 1
    assert changed > 0


def test_missing_levels_gets_no_attribute(tmp_path):
    """階数も実測も無い建物には height_m を付けない(欠測を 0 や 1 階で捏造しない)。"""
    map_doc = {
        "meta": {"crs": "local-m"},
        "nodes": [{"id": "n1", "x": 0.0, "y": 0.0}, {"id": "n2", "x": 100.0, "y": 0.0}],
        "edges": [{"u": "n1", "v": "n2", "klass": "footway", "length": 100.0,
                   "geometry": [[0.0, 0.0], [100.0, 0.0]]}],
        "buildings": [
            {"id": "b1", "kind": "yes", "levels": 4, "entrance": "n1",
             "footprint": [[0, 0], [10, 0], [10, 10], [0, 10]]},
            {"id": "b2", "kind": "yes", "levels": 0, "entrance": "n2",
             "footprint": [[20, 0], [30, 0], [30, 10], [20, 10]]},
        ],
        "pois": [],
    }
    mp = tmp_path / "tiny_map.json"
    mp.write_text(json.dumps(map_doc), encoding="utf-8")
    tbl = tmp_path / "tiny_heights.json"
    tbl.write_text(json.dumps({"meta": {"schema": bh.SCHEMA},
                               "heights": {"bX": {"h": 9.9, "src": "plateau"}}}),
                   encoding="utf-8")
    city = CityMap(mp)
    stat = city.attach_heights(tbl, fallback_m_per_level=3.5)
    assert stat == {"path": str(tbl), "n_buildings": 2, "n_plateau": 0, "n_levels": 1,
                    "n_missing": 1, "fallback_m_per_level": 3.5, "src_schema": bh.SCHEMA}
    assert city.building("b1")["height_m"] == 14.0
    assert "height_m" not in city.building("b2")
    assert city.building_height_m("b2") is None


# ------------------------------------------------------------------ 既定 OFF の不変条件
def test_off_has_no_height_attributes(tmp_path):
    """既定 OFF: 建物に属性が生えない・heights_stat が無い・summary に building_heights が無い。"""
    sim = _sim(tmp_path, "h_off", steps=3)
    assert sim.heights_stat is None
    assert not any("height_m" in b or "height_src" in b for b in sim.city.buildings)
    sim.run()
    summary = json.loads((tmp_path / "h_off" / "summary.json").read_text(encoding="utf-8"))
    assert "building_heights" not in summary


def test_on_records_summary_key(tmp_path):
    """ON: summary.json に building_heights(内訳)が追加キーとして残る。"""
    sim = _sim(tmp_path, "h_on", steps=3, **{"world.heights.enabled": "true"})
    sim.run()
    summary = json.loads((tmp_path / "h_on" / "summary.json").read_text(encoding="utf-8"))
    bhs = summary["building_heights"]
    assert bhs["path"] == "data/building_heights_shibuya.json"
    assert bhs["n_plateau"] > 0 and bhs["n_levels"] > 0
    assert bhs["n_plateau"] + bhs["n_levels"] + bhs["n_missing"] == bhs["n_buildings"]


def test_on_l1_identical_to_off_144step(tmp_path):
    """高さ ON は属性付与のみ=消費者ゼロ=15体144step の L1 がバイト一致(呼数も一致)。"""
    off = _sim(tmp_path, "hl_off", steps=144)
    off.run()
    on = _sim(tmp_path, "hl_on", steps=144, **{"world.heights.enabled": "true"})
    on.run()
    assert _l1(off) == _l1(on), "高さ配線が挙動に漏れている(属性付与のみのはず)"
    assert off.llm.calls == on.llm.calls


@pytest.mark.parametrize("enabled", ["false", "true"])
def test_map_untouched_except_building_attrs(tmp_path, enabled):
    """高さ配線はグラフ(ノード/エッジ)に一切触れない(移動・経路への波及ゼロ)。"""
    sim = _sim(tmp_path, f"hg_{enabled}", steps=1, **{"world.heights.enabled": enabled})
    assert all(set(d) == {"length", "klass", "layer", "geometry", "u0"}
               for _u, _v, d in sim.city.graph.edges(data=True))
