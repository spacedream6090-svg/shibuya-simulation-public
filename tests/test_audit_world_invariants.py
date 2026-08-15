"""scripts/audit_world_invariants.py(V2)のテスト。

方針:
  - **subprocess を使わない**(`build_report` / `render_markdown` を直接呼ぶ)。
  - **既知の違反を注入したフィクスチャ**で「検査が空回りしていない」ことを示す
    (tests/test_floor_clamp.py::test_off_run_actually_has_out_of_range_floor と同じ作法。
     全部 OK のランだけで検収すると、検査が何も見ていなくても緑になる)。
  - 実 LLM・ネットワークは使わない。合成 run dir + 合成 map を tmp_path に書く。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / "scripts", REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import audit_world_invariants as awi                        # noqa: E402

SPD = 144
MPS = 10


# --------------------------------------------------------------------------- #
# フィクスチャ
# --------------------------------------------------------------------------- #
def _write_map(path: Path) -> None:
    """b1 = 10m x 10m(面積 100 m2)・3 階建て / n1,n2 の 2 ノード。"""
    doc = {
        "meta": {"origin_latlon": [35.66, 139.70]},
        "nodes": [{"id": "n1", "x": 0.0, "y": 0.0},
                  {"id": "n2", "x": 50.0, "y": 50.0}],
        "edges": [],
        "buildings": [
            {"id": "b1", "name": "テストビル", "kind": "office", "levels": 3,
             "footprint": [[0, 0], [10, 0], [10, 10], [0, 10]]},
            {"id": "b2", "name": "小屋", "kind": "shop", "levels": 1,
             "footprint": [[0, 0], [2, 0], [2, 2], [0, 2]]},
        ],
        "pois": [], "railways": [],
    }
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")


def _write_run(run: Path, events: list[dict], agents: list[dict],
               map_path: Path) -> Path:
    run.mkdir(parents=True, exist_ok=True)
    n = len(events)
    tbl = pa.table({
        "step": pa.array([int(e["step"]) for e in events], pa.int32()),
        "sim_min": pa.array([int(e.get("sim_min", e["step"] * MPS + 420))
                             for e in events], pa.int32()),
        "agent_id": pa.array([e.get("agent_id") for e in events], pa.int32()),
        "kind": pa.array([e["kind"] for e in events], pa.string()),
        "x": pa.array([e.get("x", 0.0) for e in events], pa.float32()),
        "y": pa.array([e.get("y", 0.0) for e in events], pa.float32()),
        "payload": pa.array([json.dumps(e.get("payload", {}), ensure_ascii=False)
                             for e in events], pa.string()),
        "rng_stream": pa.array([""] * n, pa.string()),
        "llm_call_id": pa.array([None] * n, pa.string()),
    })
    pq.write_table(tbl, run / "l1_events.parquet", compression="zstd")
    (run / "agents.json").write_text(json.dumps(agents, ensure_ascii=False),
                                     encoding="utf-8")
    (run / "summary.json").write_text(
        json.dumps({"n_agents": len(agents), "n_steps": 288}), encoding="utf-8")
    (run / "config.yaml").write_text(
        f'run:\n  dt_min: {MPS}\n  start_tod: "07:00"\n'
        f'world:\n  map: {map_path.as_posix()}\n', encoding="utf-8")
    return run


def _agent(i: int, **kw) -> dict:
    a = {"id": i, "name": f"a{i}", "age": 35, "gender": "男", "occupation": "会社員",
         "visitor": False, "home_building": "b1", "home_floor": 2,
         "work_building": "b1", "part_time": False}
    a.update(kw)
    return a


def _clean(tmp_path: Path) -> Path:
    mp = tmp_path / "map.json"
    _write_map(mp)
    agents = [_agent(i) for i in range(5)]
    ev = [
        {"step": 0, "agent_id": 0, "kind": "arrive", "x": 0.0, "y": 0.0,
         "payload": {"node": "n1"}},
        {"step": 1, "agent_id": 0, "kind": "enter_building",
         "payload": {"building": "b1", "floor": 2}},
        {"step": 2, "agent_id": 0, "kind": "exit_building",
         "payload": {"building": "b1"}},
        {"step": 3, "agent_id": 1, "kind": "speak", "payload": {"hearers": [2, 3]}},
        {"step": 4, "agent_id": -1, "kind": "weather",
         "payload": {"cond": "晴れ", "holiday": False}},
    ]
    return _write_run(tmp_path / "clean", ev, agents, mp)


def _dirty(tmp_path: Path) -> Path:
    """**既知の違反を 1 種類ずつ注入**した run。検査が空回りしていない証拠になる。"""
    mp = tmp_path / "map.json"
    _write_map(mp)
    agents = [
        _agent(1),                                                    # 正常
        _agent(2, age=16),                          # 16 歳の会社員 = 年齢下限違反
        _agent(3, age=12, occupation="カフェ店員"),   # 12 歳の就業 = 児童労働
        _agent(4, home_building=""),                # 住居の無い居住者
        _agent(5, age=200, occupation="無職"),       # 年齢が定義域の外
        _agent(6, home_building="bZZZ"),            # 地図に無い建物
        _agent(7, age=44, occupation="大学生"),      # 30 歳以上の学生(帯)
    ]
    ev = [
        {"step": 0, "agent_id": 1, "kind": "arrive", "x": 0.0, "y": 0.0,
         "payload": {"node": "n1"}},
        # node と (x,y) の食い違い
        {"step": 1, "agent_id": 1, "kind": "arrive", "x": 99.0, "y": 99.0,
         "payload": {"node": "n1"}},
        # 地図に無い建物
        {"step": 2, "agent_id": 1, "kind": "enter_building",
         "payload": {"building": "bZZZ", "floor": 1}},
        # 階数超え(b1 は 3 階建て)
        {"step": 3, "agent_id": 1, "kind": "enter_building",
         "payload": {"building": "b1", "floor": 9}},
        # 入館せずに退館
        {"step": 4, "agent_id": 2, "kind": "exit_building",
         "payload": {"building": "b1"}},
        # 座標欠落 + 存在しない agent への参照
        {"step": 5, "agent_id": 3, "kind": "speak", "x": None, "y": None,
         "payload": {"hearers": [999]}},
        # 死亡 -> 死後の行動 -> 死後の金流 -> 二重の死
        {"step": 6, "agent_id": 1, "kind": "death",
         "payload": {"cause": "illness", "day": 0}},
        {"step": 7, "agent_id": 1, "kind": "speak", "payload": {}},
        {"step": 8, "agent_id": 1, "kind": "spend", "payload": {"amount": 100}},
        {"step": 9, "agent_id": 1, "kind": "death",
         "payload": {"cause": "illness", "day": 0}},
        # 転出 -> **当日は違反にしない** / 翌活動日は違反
        {"step": 10, "agent_id": 5, "kind": "life_event",
         "payload": {"kind": "emigrate", "day": 0}},
        {"step": 11, "agent_id": 5, "kind": "speak", "payload": {}},
        {"step": SPD + 5, "agent_id": 5, "kind": "speak", "payload": {}},
    ]
    return _write_run(tmp_path / "dirty", ev, agents, mp)


def _by_id(rep: dict) -> dict:
    return {c["id"]: c for c in rep["checks"]}


# --------------------------------------------------------------------------- #
# 完走 + スキーマ
# --------------------------------------------------------------------------- #
def test_report_schema_and_check_coverage(tmp_path: Path):
    rep = awi.build_report(str(_clean(tmp_path)))
    assert rep["schema_version"] == awi.SCHEMA_VERSION
    assert len(rep["checks"]) == len(awi.CHECKS)
    for c in rep["checks"]:
        assert set(c) == {"id", "category", "severity", "label", "description",
                          "n_violations", "n_checked", "rate", "band", "unit",
                          "min_n", "status", "examples", "extra"}
        assert c["severity"] in ("zero", "band")
        assert c["status"] in ("ok", "violation", "warn", "skipped", "small_n")
        assert len(c["examples"]) <= awi.MAX_EXAMPLES
    cats = {c["category"] for c in rep["checks"]}
    assert cats == {"position", "age_role", "lifecycle", "capacity", "integrity"}
    json.dumps(rep, ensure_ascii=False)


def test_zero_and_band_checks_are_reported_separately(tmp_path: Path):
    rep = awi.build_report(str(_dirty(tmp_path)))
    zero = [c for c in rep["checks"] if c["severity"] == "zero"]
    band = [c for c in rep["checks"] if c["severity"] == "band"]
    assert zero and band
    # 許容帯の検査は「0 であるべき」違反の一覧に混ざらない
    band_ids = {c["id"] for c in band}
    assert not (set(rep["totals"]["zero_violations"]) & band_ids)
    for c in band:
        assert c["band"] is not None, f"{c['id']} に帯が無い"
    for c in zero:
        assert c["band"] is None


def test_clean_run_has_no_zero_severity_violations(tmp_path: Path):
    rep = awi.build_report(str(_clean(tmp_path)))
    assert rep["totals"]["zero_violations"] == []
    assert rep["totals"]["violation"] == 0


# --------------------------------------------------------------------------- #
# ★注入した違反を実際に捕まえるか(検査の空回り防止)
# --------------------------------------------------------------------------- #
def test_injected_violations_are_all_detected(tmp_path: Path):
    rep = awi.build_report(str(_dirty(tmp_path)))
    by = _by_id(rep)
    expected = [
        "pos_missing_xy", "pos_node_xy_mismatch", "pos_unknown_building",
        "pos_exit_without_enter",
        "age_out_of_range", "age_child_worker", "age_below_occupation_min",
        "dead_agent_acts", "dead_agent_money", "death_duplicated",
        "emigrant_acts_later_day",
        "orphan_resident_no_home", "orphan_unknown_home_or_work",
        "dangling_agent_ref",
    ]
    for cid in expected:
        assert by[cid]["n_violations"] >= 1, f"{cid} が違反を検出していない"
        assert by[cid]["status"] == "violation", f"{cid} の判定が violation でない"
        assert by[cid]["examples"], f"{cid} に例示が無い"
    assert set(rep["totals"]["zero_violations"]) >= set(expected)


def test_examples_carry_enough_context_to_locate_the_row(tmp_path: Path):
    by = _by_id(awi.build_report(str(_dirty(tmp_path))))
    ex = by["dead_agent_money"]["examples"][0]
    assert ex["agent_id"] == 1 and ex["kind"] == "spend" and ex["death_step"] == 6
    ex = by["age_below_occupation_min"]["examples"][0]
    assert ex["occupation"] == "会社員" and ex["age"] == 16 and ex["min_age"] == 18
    ex = by["dangling_agent_ref"]["examples"][0]
    assert ex["missing_id"] == 999 and ex["field"] == "hearers"
    ex = by["pos_node_xy_mismatch"]["examples"][0]
    assert ex["node"] == "n1" and ex["node_xy"] == [0.0, 0.0]


def test_emigration_same_day_is_not_a_violation(tmp_path: Path):
    """★転出者は当日中 sim.agents に残る規約(tests/test_population.py)。

    当日ぶんまで違反にすると誤検出になるので、翌活動日以降だけを数える。
    """
    by = _by_id(awi.build_report(str(_dirty(tmp_path))))
    # step 11(同日)と step 149(翌日)の 2 件のうち、違反は 1 件だけ
    assert by["emigrant_acts_later_day"]["n_violations"] == 1
    assert by["emigrant_acts_later_day"]["examples"][0]["day"] == 1


def test_floor_out_of_range_is_a_band_not_a_zero_check(tmp_path: Path):
    """world.floor_clamp 既定 OFF では階超えが現に出る = 0 を要求しない。"""
    by = _by_id(awi.build_report(str(_dirty(tmp_path))))
    c = by["pos_floor_out_of_range"]
    assert c["severity"] == "band"
    assert c["n_violations"] >= 1
    assert c["examples"][0]["floor"] == 9 and c["examples"][0]["levels"] == 3
    assert c["examples"][0]["clamped_to"] == 3      # floors.clamp_floor の規則そのもの


def test_small_roster_reports_small_n_instead_of_a_false_warning(tmp_path: Path):
    """1% の帯は名簿 7 人では判定不能。OK とも WARN とも言わない。"""
    by = _by_id(awi.build_report(str(_dirty(tmp_path))))
    assert by["age_student_over_30"]["n_violations"] == 1     # 44 歳の大学生
    assert by["age_student_over_30"]["status"] == "small_n"
    assert by["age_clip_pileup"]["status"] == "small_n"


def test_density_check_flags_a_physically_impossible_crowd(tmp_path: Path):
    """b2 は 2m x 2m x 1 階 = 4 m2。そこへ 30 人入れると 7.5 人/m2 = 物理的に不可能。"""
    mp = tmp_path / "map.json"
    _write_map(mp)
    agents = [_agent(i) for i in range(30)]
    ev = [{"step": 1, "agent_id": i, "kind": "enter_building",
           "payload": {"building": "b2", "floor": 1}} for i in range(30)]
    rep = awi.build_report(str(_write_run(tmp_path / "crush", ev, agents, mp)))
    c = _by_id(rep)["cap_density_physical"]
    assert c["n_violations"] == 1 and c["status"] == "violation"
    assert c["examples"][0]["building"] == "b2"
    assert c["examples"][0]["density_per_m2"] > awi.PHYSICAL_DENSITY_LIMIT


def test_checks_are_skipped_when_the_map_cannot_be_read(tmp_path: Path):
    """地図が無いランでは、地図依存の検査は OK ではなく SKIP と言う。"""
    run = tmp_path / "nomap"
    run.mkdir()
    pq.write_table(pa.table({
        "step": pa.array([0], pa.int32()), "sim_min": pa.array([420], pa.int32()),
        "agent_id": pa.array([0], pa.int32()), "kind": pa.array(["arrive"], pa.string()),
        "x": pa.array([0.0], pa.float32()), "y": pa.array([0.0], pa.float32()),
        "payload": pa.array(['{"node": "n1"}'], pa.string()),
        "rng_stream": pa.array([""], pa.string()),
        "llm_call_id": pa.array([None], pa.string()),
    }), run / "l1_events.parquet")
    (run / "agents.json").write_text(json.dumps([_agent(0)]), encoding="utf-8")
    (run / "config.yaml").write_text('run:\n  dt_min: 10\n', encoding="utf-8")
    by = _by_id(awi.build_report(str(run)))
    for cid in ("pos_unknown_building", "pos_floor_out_of_range",
                "cap_density_physical", "orphan_unknown_home_or_work"):
        assert by[cid]["status"] == "skipped", f"{cid} が SKIP になっていない"


# --------------------------------------------------------------------------- #
# 出力
# --------------------------------------------------------------------------- #
def test_markdown_separates_the_two_tables_and_lists_violations(tmp_path: Path):
    md = awi.render_markdown(awi.build_report(str(_dirty(tmp_path))))
    assert "## 0 であるべき検査" in md and "## 許容帯の検査" in md
    assert "0 であるべき検査に違反がある" in md
    assert "dead_agent_money" in md
    assert "VIOLATION" in md


def test_analyze_writes_artifacts_and_is_deterministic(tmp_path: Path):
    run = _dirty(tmp_path)
    a = awi.analyze(str(run), str(tmp_path / "o1"))
    b = awi.analyze(str(run), str(tmp_path / "o2"))
    assert Path(a["paths"]["json"]).is_file() and Path(a["paths"]["md"]).is_file()
    assert Path(a["paths"]["json"]).read_bytes() == Path(b["paths"]["json"]).read_bytes()


def test_main_exit_code_signals_zero_severity_violations(tmp_path: Path):
    dirty = _dirty(tmp_path)
    clean = _clean(tmp_path)
    assert awi.main([str(dirty), "--out", str(tmp_path / "d"),
                     "--fail-on-violation"]) == 1
    assert awi.main([str(clean), "--out", str(tmp_path / "c"),
                     "--fail-on-violation"]) == 0
