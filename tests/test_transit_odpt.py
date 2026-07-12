"""ODPT 実ダイヤ変換(scripts/build_transit_odpt.py。第10バッチ)のテスト。

- 変換の単体: 平日抽出・深夜(0-2時台)の +1440 繰上げ・日中間隔の中央値。
- 生成物 data/transit_odpt.json が Transit ローダで読め、運行判定が妥当
  (昼=運行あり / 深夜3時台=全休)。
- 取得キャッシュ(data/odpt/)が無い環境では変換部分を skip(取得は要 ODPT_API_KEY)。
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from society.world.transit import Transit

REPO = Path(__file__).resolve().parents[1]
ODPT = REPO / "data" / "odpt"
OUT = REPO / "data" / "transit_odpt.json"
GTFS_METRO = ODPT / "gtfs" / "TokyoMetro-Train-GTFS.zip"


def _mod():
    spec = importlib.util.spec_from_file_location(
        "build_transit_odpt", REPO / "scripts" / "build_transit_odpt.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_line_stats_and_midnight_wrap():
    m = _mod()
    # 深夜 0:02 発は 24:02(=1442)へ繰上げ、日中の中央値間隔を採る
    times = sorted([m._to_min("05:01"), m._to_min("05:11")]
                   + [7 * 60 + i * 5 for i in range(0, 120)]   # 日中 5 分間隔
                   + [m._to_min("00:02") + 1440])
    st = m.line_stats(times)
    assert st["first"] == "05:01" and st["last"] == "24:02"
    assert st["headway_min"] == 5


@pytest.mark.skipif(not (ODPT / "station_timetable_metro_ginza.json").exists(),
                    reason="ODPT 取得キャッシュなし(要 ODPT_API_KEY で fetch)")
def test_weekday_departures_from_cache():
    m = _mod()
    times = m.weekday_departures(ODPT / "station_timetable_metro_ginza.json")
    assert times and len(times) > 200, "平日の発車時刻が抽出できていない"
    assert times == sorted(times)
    assert times[0] >= 4 * 60 and times[-1] > 23 * 60, "始発・終電の範囲が不自然"


@pytest.mark.skipif(not GTFS_METRO.exists(),
                    reason="静的 GTFS キャッシュなし(fetch_gtfs_odpt.py を先に)")
def test_gtfs_shibuya_departures():
    """静的 GTFS(メトロ・オープン枠)から渋谷発の平日ダイヤが再構成できる。

    駅時刻表との相互検証: 半蔵門線は両ソースが完全一致することを実測済み(2026-07-08)。
    この経路は JR東 GTFS(チャレンジ2026で提供予定・未投入)の受け皿でもある。"""
    m = _mod()
    by_route = m.gtfs_shibuya_departures(GTFS_METRO)
    for name in ("銀座線", "半蔵門線", "副都心線"):
        assert name in by_route, f"{name} が GTFS から取れていない"
    g = by_route["半蔵門線"][0]                      # 本数の多い方向が先頭
    assert len(g["times"]) > 200 and g["times"] == sorted(g["times"])
    st = m.line_stats(g["times"])
    assert st["first"] == "05:15" and st["last"] == "24:12" and st["headway_min"] == 3, \
        "駅時刻表との相互検証値(半蔵門線)と一致しない"


@pytest.mark.skipif(not OUT.exists(),
                    reason="transit_odpt.json 未生成(build_transit_odpt.py を先に)")
def test_generated_file_loads_and_serves():
    raw = json.loads(OUT.read_text(encoding="utf-8"))
    assert "ODPT" in raw["meta"]["attribution"], "出典表示がない(利用規約)"
    real = [ln for ln in raw["lines"] if "実ダイヤ" in ln.get("source", "")]
    assert len(real) >= 3, "実ダイヤ化された路線が3未満"
    tr = Transit(OUT)
    assert tr.has_service(12 * 60), "昼に運行していない"
    assert tr.has_service(5 * 60 + 30), "朝5時半に全休(始発が反映されていない)"
    assert not tr.has_service(3 * 60 + 30), "深夜3時半に運行している(終電の意味が壊れている)"
    # 実ダイヤ路線の運転間隔は現実的な範囲(メトロ=2〜5分、JR埼京線等の日中は〜15分)
    for ln in real:
        assert 2 <= ln["headway_min"] <= 15, f"間隔が不自然: {ln['name']}={ln['headway_min']}"
