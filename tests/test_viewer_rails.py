"""viz/make_viewer.py の build_rail_lines(線路→運行路線の構築)の単体テスト。

データ直読みの純関数テスト(mock 実行は不要)。渋谷駅の実態
(docs/research/rail-shibuya.md)に沿って、断片化した線路が
「連結済みの1本の運行路線」に組み直されていることを検証する。
  - 直通ペア(東横⇔副都心 / 田都⇔半蔵門)は 1 本の through 路線に連結
  - 銀座線・井の頭線は渋谷が物理的終端 → terminus(渋谷端で折り返し)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "viz"))

import make_viewer as mv  # noqa: E402

MAP_PATH = REPO_ROOT / "data" / "shibuya_osm_wide_v7.json"
TRANSIT_PATH = REPO_ROOT / "data" / "transit_odpt.json"


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _path_len(path):
    return sum(_dist(path[i - 1], path[i]) for i in range(1, len(path)))


def _load():
    if not MAP_PATH.exists() or not TRANSIT_PATH.exists():
        pytest.skip(f"必要なデータが無い: {MAP_PATH.name} / {TRANSIT_PATH.name}")
    city = json.loads(MAP_PATH.read_text(encoding="utf-8"))
    transit = json.loads(TRANSIT_PATH.read_text(encoding="utf-8"))
    return city, transit


@pytest.fixture(scope="module")
def rail_lines():
    city, transit = _load()
    return mv.build_rail_lines(city, transit)


def test_at_least_six_lines(rail_lines):
    assert len(rail_lines) >= 6, f"運行路線が少なすぎる: {len(rail_lines)}"


def test_schema(rail_lines):
    for rl in rail_lines:
        assert set(rl) >= {"name", "kind", "mode", "path",
                           "headway_min", "first_min", "last_min"}
        assert rl["mode"] in ("through", "terminus")
        assert isinstance(rl["path"], list) and len(rl["path"]) >= 2
        assert rl["first_min"] <= rl["last_min"]
        assert rl["headway_min"] >= 1


def test_through_pair_exists(rail_lines):
    hit = [rl for rl in rail_lines if rl["name"] == "東急東横線⇔副都心線"]
    assert hit, "直通連結された『東急東横線⇔副都心線』が存在しない"
    assert hit[0]["mode"] == "through"
    # もう一方の直通ペアも連結されていること
    assert any(rl["name"] == "田園都市線⇔半蔵門線" and rl["mode"] == "through"
               for rl in rail_lines)


def test_terminus_lines(rail_lines):
    by_mode = {}
    for rl in rail_lines:
        by_mode.setdefault(rl["mode"], []).append(rl["name"])
    ginza = [rl for rl in rail_lines if "銀座線" in rl["name"]]
    ido = [rl for rl in rail_lines if "井の頭" in rl["name"]]
    assert ginza and ginza[0]["mode"] == "terminus", "銀座線は terminus であるべき"
    assert ido and ido[0]["mode"] == "terminus", "井の頭線は terminus であるべき"


def test_paths_continuous_and_long(rail_lines):
    for rl in rail_lines:
        path = rl["path"]
        # 連続性: 隣接点の距離 < 100m(端点連結+等分割の担保)
        gaps = [_dist(path[i - 1], path[i]) for i in range(1, len(path))]
        assert max(gaps) < 100.0, f"{rl['name']}: 隣接点が離れすぎ max={max(gaps):.1f}"
        # 全長 > 500m
        assert _path_len(path) > 500.0, f"{rl['name']}: 短すぎ len={_path_len(path):.0f}"


def test_through_longer_than_single(rail_lines):
    """直通路線の全長 > 東横単体の全長(2本を連結した証拠)。"""
    city, transit = _load()
    # 東横だけの city を作って単体の連結長を得る
    toyoko_only = mv.build_rail_lines(
        {"railways": [r for r in city["railways"]
                      if "東横" in (r.get("name") or "")]}, transit)
    assert toyoko_only, "東横単体の路線が構築できない"
    single_len = max(_path_len(rl["path"]) for rl in toyoko_only)

    through = next(rl for rl in rail_lines if rl["name"] == "東急東横線⇔副都心線")
    assert _path_len(through["path"]) > single_len, (
        f"直通 {_path_len(through['path']):.0f}m が東横単体 {single_len:.0f}m を超えない")


def test_graceful_without_railways_or_transit():
    """railways / transit が欠けてもクラッシュしない(静かに劣化)。"""
    assert mv.build_rail_lines({}, {}) == []
    assert mv.build_rail_lines({"railways": []}, {"lines": []}) == []
    city, _ = _load()
    # transit 欠損でも既定の運行窓で路線は構築される
    lines = mv.build_rail_lines(city, {})
    assert len(lines) >= 6
    assert all(rl["first_min"] < rl["last_min"] for rl in lines)


def test_viewer_generation_smoke():
    """直近の runs/ があれば make_viewer の生成経路を関数呼び出しで通し、
    HTML が生成される(railLines が埋め込まれる)ことだけ確認。無ければ省略。"""
    runs = REPO_ROOT / "runs"
    if not runs.exists():
        pytest.skip("runs/ が無い")
    candidates = [runs / "day80"] if (runs / "day80" / "l1_events.parquet").exists() else []
    candidates += sorted(p.parent for p in runs.glob("*/l1_events.parquet"))
    run_dir = next((c for c in candidates if (c / "config.yaml").exists()), None)
    if run_dir is None:
        pytest.skip("l1_events.parquet を含む run が無い")

    data = mv.build_data(run_dir, include_traffic=False)
    assert "railLines" in data and isinstance(data["railLines"], list)
    payload = json.dumps(data, ensure_ascii=False)
    html = (mv.MAP_HTML
            .replace("__BASE_CSS__", mv._BASE_CSS)
            .replace("__ERR_JS__", mv._ERR_JS)
            .replace("__TIME_JS__", mv._TIME_JS)
            .replace("__THEME_JS__", mv._THEME_JS)
            .replace("__FLOOR_JS__", mv._FLOOR_JS)
            .replace("__RUN__", data["runName"])
            .replace("__DATA__", payload))
    assert "<html" in html and "RAILLINES" in html
    assert '"railLines"' in html
