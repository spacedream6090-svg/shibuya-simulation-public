"""scripts/reality_score.py(V1)のテスト。

方針(本 repo の作法):
  - **subprocess を使わない**。`build_report` / `render_markdown` / `load_registry` を
    関数として直接呼ぶ。
  - 実 LLM・ネットワークは一切使わない。合成 run dir を tmp_path に書くだけ。
  - conftest.py はリポに無いので、フィクスチャはこのファイルにローカル定義する
    (tests/test_judge.py / tests/test_occupancy.py と同じ流儀)。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / "scripts", REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import reality_score as rs                                  # noqa: E402

SPD = 144
MPS = 10


# --------------------------------------------------------------------------- #
# フィクスチャ(最小の run dir)
# --------------------------------------------------------------------------- #
def _write_l1(run: Path, events: list[dict]) -> None:
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


def _write_run(run: Path, events: list[dict], agents: list[dict], *,
               n_steps: int = 288, l2: dict | None = None,
               llm: list[dict] | None = None) -> Path:
    run.mkdir(parents=True, exist_ok=True)
    _write_l1(run, events)
    (run / "agents.json").write_text(json.dumps(agents, ensure_ascii=False),
                                     encoding="utf-8")
    (run / "summary.json").write_text(
        json.dumps({"n_agents": len(agents), "n_steps": n_steps}), encoding="utf-8")
    (run / "config.yaml").write_text(
        f'run:\n  dt_min: {MPS}\n  start_tod: "07:00"\n', encoding="utf-8")
    if l2:
        pq.write_table(pa.table(l2), run / "l2_metrics.parquet")
    if llm:
        pq.write_table(pa.table({
            "agent_id": pa.array([int(r["agent_id"]) for r in llm], pa.int64()),
            "purpose": pa.array([str(r["purpose"]) for r in llm], pa.string()),
            "step": pa.array([int(r.get("step", 0)) for r in llm], pa.int64()),
            "cached": pa.array([False] * len(llm), pa.bool_()),
        }), run / "l1b_llm.parquet")
    return run


def _agent(i: int, **kw) -> dict:
    a = {"id": i, "name": f"a{i}", "age": 30 + (i % 30), "gender": "女" if i % 2 else "男",
         "occupation": "会社員", "visitor": False, "home_building": "b1",
         "home_floor": 1, "work_building": "b1", "part_time": False}
    a.update(kw)
    return a


def _rich_run(tmp_path: Path) -> Path:
    """スコアが実際に付くだけの標本を持つ合成ラン(40 人 x 2 日)。"""
    agents = [_agent(i) for i in range(40)]
    ev: list[dict] = []
    for i in range(40):
        # 22:00 就寝 -> 翌 06:00 起床(sim_min は絶対分)
        ev.append({"step": 90, "sim_min": 1320, "agent_id": i, "kind": "sleep_start"})
        ev.append({"step": 138, "sim_min": 1800, "agent_id": i, "kind": "wake_up",
                   "payload": {"slept_steps": 48}})
        ev.append({"step": 150, "sim_min": 1920, "agent_id": i, "kind": "route_start",
                   "payload": {"dest": "n2"}})
        ev.append({"step": 200, "sim_min": 2420, "agent_id": i, "kind": "route_start",
                   "payload": {"dest": "n3"}})
        # 帰宅(自宅 b1)-> 60 分後に就寝
        ev.append({"step": 230, "sim_min": 2720, "agent_id": i,
                   "kind": "enter_building", "payload": {"building": "b1", "floor": 1}})
        ev.append({"step": 236, "sim_min": 2780, "agent_id": i, "kind": "sleep_start"})
        ev.append({"step": 260, "sim_min": 3020, "agent_id": i,
                   "kind": "exit_building", "payload": {"building": "b1"}})
    # 来街者の入退場(在場カーブに形を作る)
    for i in range(40, 60):
        ev.append({"step": 30, "sim_min": 720, "agent_id": i, "kind": "enter_area",
                   "payload": {"gateway": "g1"}})
        ev.append({"step": 100, "sim_min": 1420, "agent_id": i, "kind": "exit_area",
                   "payload": {"gateway": "g1"}})
    ev.append({"step": 0, "sim_min": 420, "agent_id": -1, "kind": "weather",
               "payload": {"cond": "晴れ", "holiday": False}})
    ev.sort(key=lambda e: e["step"])
    l2 = {"step": list(range(288)),
          "n_sleeping": [30 if (s % 144) < 60 else 2 for s in range(288)],
          "n_working": [20 if 60 <= (s % 144) < 110 else 0 for s in range(288)]}
    llm = [{"agent_id": i % 40, "purpose": p, "step": 10}
           for i in range(120) for p in ("plan", "reply", "face")]
    return _write_run(tmp_path / "rich", ev, agents, l2=l2, llm=llm)


def _bare_run(tmp_path: Path) -> Path:
    """ほぼ何も起きていないラン(N/A 処理の検証用)。"""
    agents = [_agent(i) for i in range(3)]
    ev = [{"step": 0, "sim_min": 420, "agent_id": 0, "kind": "arrive",
           "payload": {"node": "n1"}}]
    return _write_run(tmp_path / "bare", ev, agents, n_steps=144)


# --------------------------------------------------------------------------- #
# registry(ground truth 台帳)
# --------------------------------------------------------------------------- #
def test_registry_loads_and_every_anchor_declares_provenance():
    reg = rs.load_registry()
    assert reg["_by_id"], "アンカーが 1 件も無い"
    for aid, a in reg["_by_id"].items():
        for key in rs.REQUIRED_ANCHOR_KEYS:
            assert a.get(key), f"{aid} に {key} が無い"
        assert a["split"] in ("calibration", "holdout", "diagnostic")
        assert a["category"] in {c for c, _ in rs.CATEGORY_ORDER}
    # 判定帯は台帳側にある(コードに閾値を埋めない)
    assert set(reg["thresholds"]) >= {"jsd", "mape", "ks"}


def test_registry_records_that_ssb_2026_is_unpublished():
    """F17: 社会生活基本調査 2026 は提出前に未公表 = 2021 が正当な最新。"""
    reg = rs.load_registry()
    ssb = [v for v in reg["data_vintage"] if v["id"] == "ssb"]
    assert ssb, "社会生活基本調査の vintage が台帳に無い"
    assert ssb[0]["available_at_submission"] is False
    assert "2021" in str(ssb[0]["latest_published"])
    # メディアは年次更新済み(令和7年度)
    iicp = [v for v in reg["data_vintage"] if v["id"] == "iicp_media"][0]
    assert iicp["available_at_submission"] is True
    assert "令和7" in str(iicp["latest_published"])


def test_registry_has_spatial_support_for_every_anchor():
    reg = rs.load_registry()
    support = set(reg["spatial_support"])
    assert support >= {"bbox", "ward", "metro", "tokyo", "nation"}
    for aid, a in reg["_by_id"].items():
        assert a["spatial_support"] in support, f"{aid} の空間分母が未定義"


def test_registry_rejects_anchor_without_source(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "thresholds: {mape: {ok: 0.1, warn: 0.3}}\n"
        "anchors:\n"
        "  - id: x\n"
        "    category: population\n"
        "    label: y\n"
        "    year: 2020\n"
        "    spatial_support: ward\n"
        "    denominator: z\n"
        "    split: holdout\n", encoding="utf-8")
    with pytest.raises(rs.RegistryError):
        rs.load_registry(str(bad))


def test_registry_rejects_unknown_split(tmp_path: Path):
    bad = tmp_path / "bad2.yaml"
    bad.write_text(
        "anchors:\n"
        "  - {id: x, category: population, label: y, source: s, year: 2020,\n"
        "     spatial_support: ward, denominator: z, split: guess}\n", encoding="utf-8")
    with pytest.raises(rs.RegistryError):
        rs.load_registry(str(bad))


# --------------------------------------------------------------------------- #
# 指標(純関数)
# --------------------------------------------------------------------------- #
def test_jsd_is_zero_for_identical_and_one_for_disjoint():
    assert rs.jsd([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0, abs=1e-12)
    assert rs.jsd([10, 20, 30], [1, 2, 3]) == pytest.approx(0.0, abs=1e-12)  # 正規化
    assert rs.jsd([1, 0], [0, 1]) == pytest.approx(1.0, abs=1e-9)
    assert rs.jsd([1, 2], [1, 2, 3]) is None       # 長さ違いは値を作らない
    assert rs.jsd([0, 0], [1, 1]) is None          # 総和 0 は None


def test_jsd_is_symmetric():
    a, b = [3, 1, 6, 2], [1, 5, 2, 2]
    assert rs.jsd(a, b) == pytest.approx(rs.jsd(b, a))


def test_mape_handles_circular_hours():
    # 23:54 と 00:06 は 12 分差(円周)であって 23.8 時間差ではない
    d = rs.mape(23.9, 0.1, circular_hours=True)
    assert d == pytest.approx(0.2 / 24.0, abs=1e-9)
    assert rs.mape(100.0, 50.0) == pytest.approx(1.0)
    assert rs.mape(None, 1.0) is None
    assert rs.mape(1.0, 0.0) is None


def test_ks_vs_points_returns_none_when_sim_cannot_be_evaluated():
    pts = [{"hour": 1.0, "value": 0.5}, {"hour": 2.0, "value": 0.5}]
    assert rs.ks_vs_points(lambda h: 0.6, pts) == pytest.approx(0.1)
    assert rs.ks_vs_points(lambda h: None, pts) is None
    assert rs.ks_vs_points(lambda h: 0.5, []) is None


# --------------------------------------------------------------------------- #
# 完走 + スキーマ
# --------------------------------------------------------------------------- #
def test_build_report_runs_and_every_row_has_the_same_schema(tmp_path: Path):
    rep = rs.build_report(str(_rich_run(tmp_path)))
    assert rep["schema_version"] == rs.SCHEMA_VERSION
    names = [c["name"] for c in rep["categories"]]
    assert names == [k for k, _ in rs.CATEGORY_ORDER]
    n_rows = 0
    for cat in rep["categories"]:
        assert cat["rows"], f"{cat['name']} に行が無い"
        for row in cat["rows"]:
            assert set(row) == set(rs.ROW_KEYS), f"{row.get('id')} の列が違う"
            assert row["status"] in ("ok", "warn", "fail", rs._NA, "info")
            assert row["split"] in ("calibration", "holdout", "diagnostic")
            assert row["source"] and row["year"] and row["spatial_support"]
            n_rows += 1
    assert n_rows >= 25
    # JSON にそのまま落ちること(数値・辞書のみ)
    json.dumps(rep, ensure_ascii=False)


def test_report_never_collapses_into_a_single_score(tmp_path: Path):
    """★成分表示の掟: 総合点のキーを作らない。"""
    rep = rs.build_report(str(_rich_run(tmp_path)))
    flat = json.dumps(rep, ensure_ascii=False)
    for forbidden in ('"overall_score"', '"total_score"', '"reality_score":'):
        assert forbidden not in flat
    for cat in rep["categories"]:
        assert "score" not in cat


def test_calibration_and_holdout_are_both_present(tmp_path: Path):
    rep = rs.build_report(str(_rich_run(tmp_path)))
    splits = {r["split"] for c in rep["categories"] for r in c["rows"]}
    assert "calibration" in splits and "holdout" in splits and "diagnostic" in splits


def test_rich_run_scores_the_holdout_time_use_anchors(tmp_path: Path):
    rep = rs.build_report(str(_rich_run(tmp_path)))
    by_id = {r["id"]: r for c in rep["categories"] for r in c["rows"]}
    # 40 人 x 1 夜 = 40 標本 > min_samples なので値が付く
    assert by_id["tu_sleep_min_nhk2020"]["value"] is not None
    assert by_id["tu_sleep_min_nhk2020"]["sim"] == pytest.approx(480.0)
    # 帰宅(step230=45:20相当)-> 就寝(step236)= 60 分
    assert by_id["tu_home_to_sleep_gap_min"]["sim"] == pytest.approx(60.0)
    # 在場カーブは来街者の入退場で形が付く
    assert by_id["mob_presence_curve_weekday"]["value"] is not None
    assert by_id["mob_presence_peak_trough_ratio"]["sim"] > 1.0


def test_home_arrival_equals_bedtime_is_recorded_as_zero_not_missing(tmp_path: Path):
    """★第115 の実測「帰宅=即就寝」は gap=0 という**観測**であって欠測ではない。

    `>` で書くと標本 0 件 = N/A に化けて、既知の欠陥が表から消える。
    """
    agents = [_agent(i) for i in range(30)]
    ev = []
    for i in range(30):
        ev.append({"step": 230, "sim_min": 2720, "agent_id": i,
                   "kind": "enter_building", "payload": {"building": "b1", "floor": 1}})
        ev.append({"step": 230, "sim_min": 2720, "agent_id": i, "kind": "sleep_start"})
    run = _write_run(tmp_path / "instant", ev, agents)
    rep = rs.build_report(str(run))
    row = {r["id"]: r for c in rep["categories"] for r in c["rows"]}[
        "tu_home_to_sleep_gap_min"]
    assert row["status"] != rs._NA
    assert row["sim"] == pytest.approx(0.0)
    assert row["n"] == 30


# --------------------------------------------------------------------------- #
# N/A(データ不足を正直に出す)
# --------------------------------------------------------------------------- #
def test_bare_run_is_mostly_na_and_says_why(tmp_path: Path):
    rep = rs.build_report(str(_bare_run(tmp_path)))
    rows = [r for c in rep["categories"] for r in c["rows"]]
    na = [r for r in rows if r["status"] == rs._NA]
    assert len(na) >= len(rows) // 2, "小規模ランで N/A が少なすぎる(数字を作っている)"
    for r in na:
        assert r["value"] is None, f"{r['id']}: N/A なのに値がある"
        assert "データ不足" in r["note"], f"{r['id']}: N/A の理由が書かれていない"


def test_media_is_na_when_the_feature_is_off(tmp_path: Path):
    """media.enabled 既定 OFF = media_use が 0 件 -> 全行 N/A(0 分と書かない)。"""
    rep = rs.build_report(str(_rich_run(tmp_path)))
    media = [c for c in rep["categories"] if c["name"] == "media"][0]
    assert media["n_na"] == media["n_rows"]
    assert all("media_use" in r["note"] for r in media["rows"])


def test_media_is_scored_when_media_use_exists(tmp_path: Path):
    """media.enabled ON 相当のランでは平日行に値が付き、休日行は N/A のまま。"""
    agents = [_agent(i) for i in range(30)]
    ev = [{"step": 0, "sim_min": 420, "agent_id": -1, "kind": "weather",
           "payload": {"cond": "晴れ", "holiday": False}}]
    for i in range(30):
        # 1 人 18 step = 180 分 / 日
        ev.append({"step": 60, "sim_min": 1020, "agent_id": i, "kind": "media_use",
                   "payload": {"medium": "video", "steps": 18}})
    run = _write_run(tmp_path / "media", ev, agents, n_steps=144)
    rep = rs.build_report(str(run))
    by_id = {r["id"]: r for c in rep["categories"] for r in c["rows"]}
    wk = by_id["media_internet_min_weekday"]
    assert wk["status"] != rs._NA and wk["sim"] == pytest.approx(180.0, abs=1.0)
    assert by_id["media_internet_min_holiday"]["status"] == rs._NA
    assert "休日" in by_id["media_internet_min_holiday"]["note"]


def test_missing_l2_makes_time_use_curve_na(tmp_path: Path):
    agents = [_agent(i) for i in range(30)]
    ev = [{"step": 90, "sim_min": 1320, "agent_id": i, "kind": "sleep_start"}
          for i in range(30)]
    run = _write_run(tmp_path / "nol2", ev, agents)     # l2 を書かない
    rep = rs.build_report(str(run))
    row = {r["id"]: r for c in rep["categories"] for r in c["rows"]}[
        "tu_asleep_rate_curve_nhk2020_m20s"]
    assert row["status"] == rs._NA and "n_sleeping" in row["note"]


def test_cognition_is_na_without_l1b(tmp_path: Path):
    agents = [_agent(i) for i in range(30)]
    ev = [{"step": 1, "sim_min": 430, "agent_id": i, "kind": "arrive",
           "payload": {"node": "n1"}} for i in range(30)]
    run = _write_run(tmp_path / "nollm", ev, agents)
    rep = rs.build_report(str(run))
    by_id = {r["id"]: r for c in rep["categories"] for r in c["rows"]}
    assert by_id["cog_calls_per_person_day"]["status"] == rs._NA
    # 一方 zero-call 率は「全員 0 回」として測れる(こちらは N/A にしない)
    assert by_id["cog_zero_call_share"]["sim"] == pytest.approx(1.0)


def test_cognition_counts_zero_call_agents_in_the_denominator(tmp_path: Path):
    agents = [_agent(i) for i in range(30)]
    ev = [{"step": 1, "sim_min": 430, "agent_id": i, "kind": "arrive",
           "payload": {"node": "n1"}} for i in range(30)]
    llm = [{"agent_id": 0, "purpose": "plan", "step": 1}] * 10
    run = _write_run(tmp_path / "skew", ev, agents, llm=llm)
    rep = rs.build_report(str(run))
    by_id = {r["id"]: r for c in rep["categories"] for r in c["rows"]}
    assert by_id["cog_zero_call_share"]["sim"] == pytest.approx(29 / 30, abs=1e-4)
    dist = by_id["cog_calls_distribution"]["sim"]
    assert dist["p50"] == pytest.approx(0.0)      # 0 回を落としていない証拠
    assert dist["gini"] > 0.9
    cov = by_id["cog_lane_coverage"]["sim"]
    assert cov["life"] == pytest.approx(1 / 30, abs=1e-4) and cov["reply"] == 0.0


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
def test_markdown_carries_vintage_crosswalk_and_the_no_single_score_rule(tmp_path: Path):
    md = rs.render_markdown(rs.build_report(str(_rich_run(tmp_path))))
    assert "Data Vintage Ledger" in md
    assert "社会生活基本調査 2026" in md and "2021 年版が正当な最新" in md
    assert "Spatial Support Crosswalk" in md
    assert "総合 1 点は出さない" in md
    for col in ("出典", "年次", "空間分母", "分割"):
        assert col in md
    assert "holdout" in md and "calibration" in md


def test_analyze_writes_both_artifacts(tmp_path: Path):
    run = _rich_run(tmp_path)
    rep = rs.analyze(str(run), str(tmp_path / "out"))
    for key in ("json", "md"):
        assert Path(rep["paths"][key]).is_file()
    doc = json.loads(Path(rep["paths"]["json"]).read_text(encoding="utf-8"))
    assert doc["schema_version"] == rs.SCHEMA_VERSION


def test_analyze_is_deterministic(tmp_path: Path):
    run = _rich_run(tmp_path)
    a = rs.analyze(str(run), str(tmp_path / "o1"))
    b = rs.analyze(str(run), str(tmp_path / "o2"))
    assert Path(a["paths"]["json"]).read_bytes() == Path(b["paths"]["json"]).read_bytes()
    assert Path(a["paths"]["md"]).read_bytes() == Path(b["paths"]["md"]).read_bytes()
