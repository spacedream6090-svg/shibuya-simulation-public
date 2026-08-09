#!/usr/bin/env python3
"""P4境界較正: 取得済みの現実データを **1本の counts ファイル** に正規化する。

    出力: data/realworld/boundary_counts.json   (.gitignore 済み = コミットされない)

形は MATSim の `counts.xml` の意味論をそのまま平たい JSON にしたもの。1レコード =
「ある観測地点の、ある時間帯の、実測1件」:

    {counter_id, location, time_bin, observed_count, source, year, ...}

★ **この出力をシミュ本体は読まない。** `src/` からの参照はゼロで、較正(ラン前)と
   事後検証(ラン後)にのみ使う。conf/ にキーは1つも増えていない。

------------------------------------------------------------------ 設計の要(二重計上)
渋谷の駅の数字は、そのまま足すと必ず間違える。理由は3つあり、どれも別物:

  (1) 事業者で数え方が違う  JR東日本 = 乗車人員(入場のみ) / 東急・メトロ・京王 = 乗降人員。
  (2) 直通・改札内乗換の通過客  「乗降人員」は駅を通り抜けただけの客を含む。
  (3) 同一駅の複数路線の合算  路線別の「乗車」を足すと改札内乗換が二重に入る。

本スクリプトは足し合わせを**一切しない**。各レコードに `measure`(何を数えたか)と
`double_count`(どの罠が乗っているか)を必ず書き、判断を読む側に残す。唯一の例外は
`kind="derived"` の按分(水準×時刻構成比)で、これは `derived_from` に材料を列挙する。

境界流量として本当に欲しいのは「街と駅の間を出入りした人数」= **改札外流動**で、
これに一番近いのは第13回センサスの「初乗り・最終降車駅間移動人員」(渋谷発 22.9万人/日・
渋谷着 22.4万人/日・2021年度)。詳細は docs/research/boundary-calibration-data.md。

使い方:
    python scripts/build_boundary_counts.py --fetch       # 取得してから組み立て
    python scripts/build_boundary_counts.py               # 取得済みから組み立て
    python scripts/build_boundary_counts.py --report      # 中身を人が読む形で出す
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from rw_fetch import common, ledger, odpt_passenger_survey as ps  # noqa: E402
from rw_fetch import transport_census as tc  # noqa: E402

OUT_NAME = "boundary_counts.json"
SCHEMA_VERSION = 1

# counts レコードの必須欄(この6つは全レコードに必ずある)。
REQUIRED_FIELDS = ("counter_id", "location", "time_bin", "observed_count", "source", "year")

# 二重計上の種類。レコードごとに必ずどれかを名乗る。
DOUBLE_COUNT_KINDS = {
    "none": "そのまま使える(改札外の実移動・構成比など)",
    "through_passengers": "直通・改札内乗換の通過客を含む(乗降人員)",
    "intra_station_transfer": "路線別の乗車で、同一駅の他路線からの乗換を含む",
    "supply_not_demand": "輸送定員 = 供給側の数字で、実乗客数ではない",
    "partial_survey": "乗換施設調査の部分集計で、駅の全流動を覆っていない",
}

# シミュの出入口(src/society/cognition/plan_boundary.py の gateway)への対応。
# "station" = 駅ノード1点 / "edge" = city.gateways(徒歩で圏外へ出る縁)。
PORTAL_STATION = "station"
PORTAL_EDGE = "edge"

NOTES = [
    "★ 事業者で数え方が違う: JR東日本は乗車人員(入場のみ)、東急・東京メトロ・京王は"
    "乗降人員(入場+出場)。measure 欄が boarding か boarding_alighting かを必ず見ること。"
    "素朴に足すと JR だけ半分に数える。",
    "★ 乗降人員は直通運転・改札内乗換の通過客を含む。渋谷は乗換が支配的なので、"
    "街への流入量としては大きく過大。double_count=through_passengers で印を付けてある。",
    "★ 路線別の『乗車』を同一駅で足すと改札内乗換が二重に入る"
    "(double_count=intra_station_transfer)。実測: 第12回の路線別乗車合計は JR 渋谷で"
    "391,408 人/日 に対し、ODPT の JR 渋谷 乗車人員(2015年度)は 372,234 人/日 = 5.2% 過大。",
    "★ 街と駅の間の実流動に一番近いのは第13回の『初乗り・最終降車駅間移動人員』"
    "(渋谷発 228,988 / 渋谷着 223,865 人/日・2021年度)。実測: 同年度の ODPT の JR 渋谷"
    "乗車人員 248,505 に対し、渋谷を初乗り駅とする JR トリップは 87,049 = 35.0%。"
    "残りの 65% は他路線からの乗換で、街には出ていない。",
    "★ 第12回 = 2015年度(COVID 前)、第13回 = 2021年度(COVID 下)。regime 欄が違う"
    "レコードを同じ系列として並べてはいけない。",
    "★ 時間帯の形(time_bin が hour:* のもの)は首都圏全体・目的別の構成比であって"
    "渋谷駅の実測ではない。駅×時刻の無料の一次表は存在しない"
    "(docs/research/boundary-calibration-data.md の調査結果)。",
    "★ 按分(kind=derived)の目的の選び方で日変化の**向きが逆になる**。既定の『合計』の"
    "乗車構成比は住宅地から出る側の形(朝7-8時に山)なので、目的地側の渋谷の『改札入場"
    "(街→駅)』に当てると入場が朝ピークという逆向きの形になる(実測: 7時台 44,188 人/時)。"
    "渋谷の入場は帰りの流れなので --derive-purpose 帰宅、退場は通勤の到着なので 通勤 が近い。",
    "★ この出力はシミュ本体が読まない。較正とラン後の突き合わせにのみ使う。",
]


# ================================================================ 小道具(純関数)
def _rec(counter_id: str, location: str, time_bin: str, observed_count, source: str,
         year, *, measure: str, unit: str, kind: str = "observed",
         double_count: str = "none", portal: str = PORTAL_STATION, operator: str = "",
         regime: str = "", day_type: str = "weekday", source_detail: str = "",
         derived_from=None, note: str = "") -> dict:
    """counts レコードを1件組む。`observed_count` が None のときも**捨てずに**残す。"""
    if double_count not in DOUBLE_COUNT_KINDS:
        raise ValueError(f"未知の double_count: {double_count}")
    return {
        "counter_id": counter_id,
        "location": location,
        "time_bin": time_bin,
        "observed_count": observed_count,
        "source": source,
        "year": year,
        "measure": measure,
        "unit": unit,
        "kind": kind,                      # observed | derived
        "double_count": double_count,
        "portal": portal,                  # station | edge
        "operator": operator,
        "regime": regime,                  # pre_covid | post_covid | ""
        "day_type": day_type,
        "source_detail": source_detail,
        "derived_from": list(derived_from or []),
        "note": note,
    }


def hour_bin(label: str) -> str:
    """センサスの時間帯見出し → `hour:07` 形の time_bin。

    `～6時台` は 6 時以前をまとめた区間なので `hour:le06`、`0時台～` は深夜で `hour:ge00`。
    """
    s = str(label or "").strip()
    if s.startswith("～"):
        return "hour:le06"
    if s.endswith("～"):
        return "hour:ge00"
    digits = "".join(ch for ch in s if ch.isdigit())
    if not digits:
        raise ValueError(f"時間帯として読めない: {label!r}")
    return f"hour:{int(digits):02d}"


def _num(value):
    return value if isinstance(value, (int, float)) else None


def survey_slug(same_as: str) -> str:
    """`odpt.PassengerSurvey:Tokyu.Toyoko.Shibuya` → `tokyu.toyoko`。

    東急は東横線と田園都市線、メトロは銀座線と半蔵門/副都心が**別々の計上**なので、
    事業者名だけを counter_id にすると別物が同じ名前で潰れる。路線まで名前に入れる。
    """
    tail = str(same_as or "").split(":", 1)[-1]
    parts = [p for p in tail.split(".") if p]
    if parts and parts[-1] == "Shibuya":
        parts = parts[:-1]
    return ".".join(p.lower() for p in parts) or "unknown"


# ================================================================ 変換(純関数)
def from_passenger_survey(doc: dict | None) -> list[dict]:
    """ODPT PassengerSurvey → 事業者別の年間平均日レコード(全年度を残す)。"""
    if not doc or not isinstance(doc.get("data"), list):
        return []
    out: list[dict] = []
    for rec in doc["data"]:
        measure = rec.get("measure") or "boarding"
        dc = "through_passengers" if measure == "boarding_alighting" else "none"
        op = rec.get("operator_key") or ""
        slug = survey_slug(rec.get("same_as") or op)
        for year_s, value in sorted((rec.get("years") or {}).items()):
            out.append(_rec(
                counter_id=f"station.shibuya.{slug}.{measure}",
                location=f"station:shibuya:{op}",
                time_bin="daily",
                observed_count=_num(value),
                source="odpt_passenger_survey",
                year=int(year_s),
                measure=measure, unit="persons_per_day",
                double_count=dc, operator=op,
                day_type="annual_average_day",
                source_detail=str(rec.get("same_as") or ""),
                note=("乗車人員のみ(改札入場)。" if measure == "boarding"
                      else "乗降人員。直通・改札内乗換の通過客を含む。")))
    return out


def from_station_flow(doc: dict | None) -> list[dict]:
    """第12回 駅別発着 → 路線別の乗車/降車/通過(人/日)。合計券種・両方向だけを採る。"""
    if not doc or not isinstance(doc.get("data"), list):
        return []
    meta = doc.get("_meta", {})
    year = meta.get("survey_year") or 2015
    regime = meta.get("regime") or "pre_covid"
    measure_map = {"乗車": "line_boarding", "降車": "line_alighting", "通過": "line_through"}
    out: list[dict] = []
    for rec in doc["data"]:
        if not str(rec.get("ticket", "")).startswith("合計"):
            continue                      # 定期/普通の内訳は counts には出さない(合計だけ)
        measure = measure_map.get(str(rec.get("measure")))
        if not measure:
            continue
        line = str(rec.get("line") or "")
        direction = str(rec.get("direction") or "")
        dc = "intra_station_transfer" if measure == "line_boarding" else "through_passengers"
        out.append(_rec(
            counter_id=f"station.shibuya.line.{line}.{direction}.{measure}",
            location=f"station:shibuya:line:{line}",
            time_bin="daily",
            observed_count=_num(rec.get("value")),
            source="transport_census.station_flow",
            year=year, measure=measure, unit="persons_per_day",
            double_count=dc, regime=regime, day_type="weekday",
            source_detail=f"第12回 表3 {line} {direction}",
            note="路線単位。同一駅の複数路線を足すと改札内乗換が二重に入る。"))
    return out


def from_transfer(doc: dict | None) -> list[dict]:
    """第12回 ターミナル別乗換え → 乗換行列と小計。**比率にだけ使う**印を付ける。"""
    if not doc or not isinstance(doc.get("data"), list):
        return []
    meta = doc.get("_meta", {})
    year = meta.get("survey_year") or 2015
    regime = meta.get("regime") or "pre_covid"
    out: list[dict] = []
    for rec in doc["data"]:
        kind = str(rec.get("kind") or "")
        if kind == "total":
            label = str(rec.get("total_label") or "")
            slug = {"初乗り計": "gate_entry", "最終降車計": "gate_exit",
                    "乗換え計": "transfer", "合計": "all"}.get(label, label)
            cid = f"station.shibuya.transfer_survey.total.{slug}"
            detail = f"第12回 表4 小計 {label}"
            from_to = ""
        else:
            # ★ 方向を名前に入れないと `山手線->銀座線` が上り/下りで衝突する。
            frm = f"{rec.get('from_line') or '改札外'}{rec.get('from_direction') or ''}"
            to = f"{rec.get('to_line') or '改札外'}{rec.get('to_direction') or ''}"
            from_to = f"{frm}->{to}"
            cid = f"station.shibuya.transfer_survey.{kind}.{from_to}"
            detail = f"第12回 表4 {from_to}"
        base = dict(location="station:shibuya:transfer_survey",
                    source="transport_census.transfer", year=year,
                    measure=f"transfer_survey_{kind}", unit="persons_per_day",
                    double_count="partial_survey", regime=regime, day_type="weekday",
                    source_detail=detail,
                    note="乗換施設調査の部分集計。駅別発着の水準とは桁が合わないので"
                         "比率(乗換シェア)にだけ使う。")
        out.append(_rec(cid, time_bin="daily",
                        observed_count=_num(rec.get("daily")), **base))
        peak = _num(rec.get("peak_hourly"))
        if peak is not None:
            out.append(_rec(cid, time_bin=f"peak_hour:{rec.get('peak_window') or ''}",
                            observed_count=peak, **base))
    return out


def from_time_dist(doc: dict | None) -> list[dict]:
    """第12回 目的別乗車降車時刻分布 → 時間帯の構成比(圏域全体・駅の次元は無い)。"""
    if not doc or not isinstance(doc.get("data"), list):
        return []
    meta = doc.get("_meta", {})
    year = meta.get("survey_year") or 2015
    regime = meta.get("regime") or "pre_covid"
    flow_map = {"乗車": "boarding_share", "降車": "alighting_share"}
    out: list[dict] = []
    for rec in doc["data"]:
        measure = flow_map.get(str(rec.get("flow")))
        if not measure:
            continue
        purpose = str(rec.get("purpose") or "")
        out.append(_rec(
            counter_id=f"region.shutoken.{purpose}.{measure}",
            location="region:shutoken",
            time_bin=hour_bin(rec.get("bin")),
            observed_count=_num(rec.get("share")),
            source="transport_census.time_dist",
            year=year, measure=measure, unit="share_of_daily",
            double_count="none", portal=PORTAL_STATION, regime=regime, day_type="weekday",
            source_detail=f"第12回 参考表5 {purpose} {rec.get('flow')}",
            note="首都圏全体の構成比。渋谷駅の実測ではない(駅×時刻の無料表は存在しない)。"))
    return out


def from_line_capacity(doc: dict | None) -> list[dict]:
    """第12回 路線別着時間帯別 → 駅間の輸送**定員**(供給側)。時間帯の形の傍証。"""
    if not doc or not isinstance(doc.get("data"), list):
        return []
    meta = doc.get("_meta", {})
    year = meta.get("survey_year") or 2015
    regime = meta.get("regime") or "pre_covid"
    out: list[dict] = []
    for rec in doc["data"]:
        line = str(rec.get("line") or "")
        frm, to = str(rec.get("from_station") or ""), str(rec.get("to_station") or "")
        out.append(_rec(
            counter_id=f"link.{line}.{frm}->{to}.capacity",
            location=f"link:{frm}->{to}",
            time_bin=f"band:{rec.get('arrival_bin')}",
            observed_count=_num(rec.get("capacity")),
            source="transport_census.line_capacity",
            year=year, measure="transport_capacity", unit="persons",
            double_count="supply_not_demand", regime=regime, day_type="weekday",
            operator=str(rec.get("operator") or ""),
            source_detail=f"第12回 表9 {line} {rec.get('direction') or ''}",
            note="輸送定員 = 供給側。実乗客数ではない。"))
    return out


def from_station_od13(doc: dict | None, station: str = "渋谷") -> list[dict]:
    """第13回 初乗り・最終降車 → **改札外流動**の事業者別合計(境界流量に一番近い)。

    OD の全ペアではなく、渋谷を初乗り駅/最終降車駅とする合計だけを counts に出す
    (ペア明細は data/realworld/transport_census/station_od13_13.json に残っている)。
    """
    if not doc or not isinstance(doc.get("data"), list):
        return []
    meta = doc.get("_meta", {})
    year = meta.get("survey_year") or 2021
    regime = meta.get("regime") or "post_covid"
    dep: dict[str, int] = {}
    arr: dict[str, int] = {}
    for rec in doc["data"]:
        val = _num(rec.get("passengers"))
        if val is None:
            continue
        if rec.get("from_station") == station:
            key = str(rec.get("from_operator") or "")
            dep[key] = dep.get(key, 0) + int(val)
        if rec.get("to_station") == station:
            key = str(rec.get("to_operator") or "")
            arr[key] = arr.get(key, 0) + int(val)
    out: list[dict] = []
    for measure, table, human in (("gate_entry", dep, "初乗り(街→駅→乗車)"),
                                  ("gate_exit", arr, "最終降車(降車→駅→街)")):
        for op, value in sorted(table.items()):
            out.append(_rec(
                counter_id=f"station.shibuya.gate.{op}.{measure}",
                location=f"station:shibuya:gate:{op}",
                time_bin="daily", observed_count=value,
                source="transport_census.station_od13",
                year=year, measure=measure, unit="persons_per_day",
                double_count="none", operator=op, regime=regime, day_type="weekday",
                source_detail=f"第13回 初乗り・最終降車駅間移動人員 {human}",
                note="改札の外と出入りした実移動。乗換の通過客を含まない = 境界流量に最も近い。"))
        total = sum(table.values())
        if table:
            out.append(_rec(
                counter_id=f"station.shibuya.gate.all.{measure}",
                location="station:shibuya:gate", time_bin="daily",
                observed_count=total, source="transport_census.station_od13",
                year=year, measure=measure, unit="persons_per_day",
                double_count="none", regime=regime, day_type="weekday",
                source_detail=f"第13回 初乗り・最終降車駅間移動人員 {human} 全事業者計",
                note="街と駅の間の総交換量。シミュの駅ゲートウェイに当てるならこれ。"))
    return out


def derive_hourly(counts: list[dict], *, purpose: str = "合計") -> list[dict]:
    """水準(改札外の日合計)× 形(時刻構成比)= 時間帯別の推定流量。

    **観測ではないので `kind="derived"`** とし、材料を `derived_from` に列挙する。
    形は首都圏全体の構成比なので、渋谷固有の偏りは入っていない。

    ★★ 目的の選び方で向きが逆になる罠(実測で確認):
        `purpose="合計"` の乗車構成比は朝 7-8 時に山がある。これは**住宅地から出る側**の
        形なので、渋谷のような**目的地側**の駅の「改札入場(街→駅)」に当てると
        入場が朝ピーク = 現実と逆向きの日変化になる(実測: gate_entry が 7時台 44,188 人/時)。
        渋谷の入場は帰りの流れなので `purpose="帰宅"`、退場は通勤の到着なので
        `purpose="通勤"` を当てる方が形として正しい。既定は素の「合計」のままにして
        (勝手に賢いことをしない)、選択は `--derive-purpose` に委ねる。
    """
    # 水準 = 改札外流動の全事業者計(第13回)。形 = 首都圏の時刻構成比(第12回)。
    levels = {c["measure"]: c for c in counts
              if c["source"] == "transport_census.station_od13"
              and c["counter_id"] in ("station.shibuya.gate.all.gate_entry",
                                      "station.shibuya.gate.all.gate_exit")}
    prefix = f"region.shutoken.{purpose}."
    shares = [c for c in counts
              if c["source"] == "transport_census.time_dist"
              and c["counter_id"].startswith(prefix)]
    if not levels or not shares:
        return []
    pair = {"gate_entry": "boarding_share", "gate_exit": "alighting_share"}
    out: list[dict] = []
    for measure, level in sorted(levels.items()):
        want = pair.get(measure)
        base = level["observed_count"]
        if want is None or not isinstance(base, (int, float)):
            continue
        for sh in shares:
            if sh["measure"] != want or not isinstance(sh["observed_count"], (int, float)):
                continue
            out.append(_rec(
                # ★ 目的を名前に入れる(別の目的で組んだ系列が同じ名前で潰れないように)。
                counter_id=f"station.shibuya.gate.all.{measure}.hourly.{purpose}",
                location="station:shibuya:gate",
                time_bin=sh["time_bin"],
                observed_count=round(base * sh["observed_count"], 1),
                source="derived",
                year=level["year"], measure=measure, unit="persons_per_hour",
                kind="derived", double_count="none",
                regime=level["regime"], day_type="weekday",
                source_detail=f"{level['counter_id']} × {sh['counter_id']}",
                derived_from=[level["counter_id"], sh["counter_id"]],
                note=f"水準({level['year']}年度・改札外)×構成比({sh['year']}年度・首都圏全体"
                     f"・目的={purpose})。年度も空間もそろっていない粗い按分。"))
    return out


# ================================================================ 組み立て
def build(root: Path, *, station: str = "渋谷", with_derived: bool = True,
          derive_purpose: str = "合計") -> dict:
    """保存済みの取得物を読んで counts 文書を組む(**ネットワーク不使用**)。"""
    saved = tc.load_saved(root)
    survey = ps.load_saved(root)
    counts: list[dict] = []
    counts += from_passenger_survey(survey)
    counts += from_station_flow(saved.get("station_flow"))
    counts += from_transfer(saved.get("transfer"))
    counts += from_time_dist(saved.get("time_dist"))
    counts += from_line_capacity(saved.get("line_capacity"))
    counts += from_station_od13(saved.get("station_od13"), station)
    if with_derived:
        counts += derive_hourly(counts, purpose=derive_purpose)

    sources = sorted({c["source"] for c in counts})
    missing = [k for k in ("station_flow", "time_dist", "transfer",
                           "line_capacity", "station_od13") if k not in saved]
    meta = common.build_meta(
        "transport_census", module="../build_boundary_counts.py",
        urls=[t["url"] for t in tc.TABLES] + [u for u in ([] if not survey else
                                              survey.get("_meta", {}).get("source_urls", []))],
        n_records=len(counts),
        n_missing=sum(1 for c in counts if c["observed_count"] is None),
        caveats=tc.CAVEATS + (ps.CAVEATS if survey else []),
        units={"persons_per_day": "人/日", "persons_per_hour": "人/時",
               "share_of_daily": "日合計に対する構成比", "persons": "人(輸送定員)"},
        notes=NOTES,
        extra={"counts_schema_version": SCHEMA_VERSION,
               "required_fields": list(REQUIRED_FIELDS),
               "double_count_kinds": DOUBLE_COUNT_KINDS,
               "station": station,
               "sources_present": sources,
               "sources_missing": missing,
               "portal_mapping": {
                   "station": "src/society/cognition/plan_boundary.py の gateway='station' "
                              "= 駅ノード1点。station:shibuya:gate の counts を当てる。",
                   "edge": "city.gateways(徒歩で圏外へ出る縁)。鉄道の counts は当てない "
                           "= 街路の実測が別途必要(未取得)。"},
               "n_by_source": {s: sum(1 for c in counts if c["source"] == s) for s in sources},
               "n_derived": sum(1 for c in counts if c["kind"] == "derived")})
    if survey:
        meta["odpt_attribution"] = survey.get("_meta", {}).get("attribution", "")
        meta["redistribution"] = "restricted"   # ODPT チャレンジ枠が混ざるため厳しい方に倒す
    return {"_meta": meta, "counts": counts}


def validate_counts(doc) -> list[str]:
    """counts 文書のスキーマ検証(ネットワーク不使用)。問題のリストを返す(空 = 合格)。"""
    problems: list[str] = []
    if not isinstance(doc, dict):
        return ["トップレベルが dict でない"]
    if "_meta" not in doc:
        problems.append("_meta が無い")
    rows = doc.get("counts")
    if not isinstance(rows, list):
        return problems + ["counts が list でない"]
    if not rows:
        problems.append("counts が空")
    for i, rec in enumerate(rows):
        if not isinstance(rec, dict):
            problems.append(f"counts[{i}] が dict でない")
            continue
        for field in REQUIRED_FIELDS:
            if field not in rec:
                problems.append(f"counts[{i}].{field} が無い")
        if rec.get("kind") not in ("observed", "derived"):
            problems.append(f"counts[{i}].kind が observed/derived でない: {rec.get('kind')}")
        if rec.get("double_count") not in DOUBLE_COUNT_KINDS:
            problems.append(f"counts[{i}].double_count が未知: {rec.get('double_count')}")
        if rec.get("kind") == "derived" and not rec.get("derived_from"):
            problems.append(f"counts[{i}] は derived なのに derived_from が空")
        val = rec.get("observed_count")
        if val is not None and not isinstance(val, (int, float)):
            problems.append(f"counts[{i}].observed_count が数でも None でもない")
        if isinstance(val, (int, float)) and val < 0:
            problems.append(f"counts[{i}].observed_count が負: {val}")
    # ★ (counter_id, time_bin, year, source) は一意でなければならない。ここが衝突すると
    #   「別の観測が同じ名前で潰れている」= 静かなデータ損失なので必ず落とす。
    seen: dict[tuple, int] = {}
    for i, rec in enumerate(rows):
        if not isinstance(rec, dict):
            continue
        key = (rec.get("counter_id"), rec.get("time_bin"), rec.get("year"), rec.get("source"))
        if key in seen:
            problems.append(f"counts[{i}] が counts[{seen[key]}] と同一キー(重複): {key}")
        else:
            seen[key] = i
    meta = doc.get("_meta") or {}
    if isinstance(meta, dict) and meta.get("n_records") not in (None, len(rows)):
        problems.append(f"_meta.n_records={meta.get('n_records')} != len(counts)={len(rows)}")
    return problems


def out_path(root: Path) -> Path:
    return Path(root) / OUT_NAME


def format_report(doc: dict) -> str:
    """人が読む要約(--report)。"""
    counts = doc.get("counts") or []
    meta = doc.get("_meta") or {}
    lines = ["=" * 78, "境界較正 counts(data/realworld/boundary_counts.json)", "=" * 78,
             f"レコード {len(counts)} 件 / 欠測 {meta.get('n_missing')} 件 / "
             f"derived {meta.get('n_derived')} 件"]
    if meta.get("sources_missing"):
        lines.append(f"⚠ 未取得の表: {', '.join(meta['sources_missing'])} "
                     f"(--fetch / --with-od13 で取得できる)")
    lines += ["", f"{'source':<38}{'件数':>6}", "-" * 46]
    for src, n in sorted((meta.get("n_by_source") or {}).items()):
        lines.append(f"{src:<38}{n:>6}")
    # ODPT は事業者ごとに最新年度が違うので、**全事業者がそろう最新年度**だけを出す。
    odpt_years: dict[str, set] = {}
    for rec in counts:
        if rec["source"] == "odpt_passenger_survey":
            odpt_years.setdefault(rec["counter_id"], set()).add(rec["year"])
    shared = set.intersection(*odpt_years.values()) if odpt_years else set()
    ref_year = max(shared) if shared else None
    lines += ["", "-" * 78,
              f"駅の水準(日合計) — measure が違うものは足さないこと"
              f"{'' if ref_year is None else f' / ODPT は全事業者がそろう FY{ref_year}'}",
              "-" * 78]
    show = ("odpt_passenger_survey", "transport_census.station_od13")
    for rec in counts:
        if rec["time_bin"] != "daily" or rec["source"] not in show:
            continue
        if rec["source"] == "odpt_passenger_survey" and rec["year"] != ref_year:
            continue
        val = rec["observed_count"]
        lines.append(f"  {rec['counter_id']:<48} {'' if val is None else format(val, '>11,')}"
                     f"  [{rec['measure']} / {rec['double_count']} / FY{rec['year']}]")
    lines += ["", "-" * 78, "乗換の分解(第12回 表4 小計・比率にのみ使う)", "-" * 78]
    for rec in counts:
        if rec["source"] == "transport_census.transfer" and ".total." in rec["counter_id"] \
                and rec["time_bin"] == "daily":
            lines.append(f"  {rec['counter_id']:<48} "
                         f"{format(rec['observed_count'], '>11,')}")
    lines += ["", "-" * 78, "二重計上の注意", "-" * 78]
    for note in meta.get("notes") or []:
        lines.append(f"  {note}")
    return "\n".join(lines)


# ================================================================ CLI
def main(argv=None) -> int:
    common.reconfigure_stdio()
    ap = argparse.ArgumentParser(description="境界較正の counts ファイルを組み立てる")
    ap.add_argument("--root", default=str(common.DEFAULT_ROOT),
                    help="データ置き場(既定: data/realworld)")
    ap.add_argument("--fetch", action="store_true",
                    help="センサスと ODPT を取得してから組み立てる(既取得はスキップ)")
    ap.add_argument("--force", action="store_true", help="既取得でも取り直す")
    ap.add_argument("--with-od13", action="store_true",
                    help="第13回の初乗り・最終降車(18MB)も取る = 改札外流動が入る")
    ap.add_argument("--station", default="渋谷")
    ap.add_argument("--no-derived", action="store_true", help="按分(derived)を出さない")
    ap.add_argument("--derive-purpose", default="合計",
                    choices=["合計", "通勤", "通学", "業務", "私事", "帰宅"],
                    help="按分に使う時刻構成比の目的。渋谷のような目的地側の駅では"
                         "入場に『帰宅』、退場に『通勤』の形が近い(既定の『合計』は"
                         "住宅地側の形なので入場が朝ピークになる)")
    ap.add_argument("--report", action="store_true", help="組み立て後に要約を出す")
    ap.add_argument("--offline", action="store_true",
                    help="1リクエストも出さずに取得計画だけ出す")
    args = ap.parse_args(argv)
    root = Path(args.root)

    if args.offline:
        common.log("取得計画(1リクエストも出さない):")
        for title, url in tc.plan(with_od13=args.with_od13) + ps.plan():
            common.log(f"  {title}\n    {url}")
        common.log(f"\nHTTP 要求数 = {common.request_count()}")
        return 0

    if args.fetch:
        common.log("[1/2] 大都市交通センサス")
        _, rows = tc.fetch_all(root, with_od13=args.with_od13, force=args.force)
        ledger.append(root, rows)
        common.log("[2/2] ODPT 乗降者数")
        _, rows2 = ps.fetch_all(root, force=args.force)
        ledger.append(root, rows2)

    doc = build(root, station=args.station, with_derived=not args.no_derived,
                derive_purpose=args.derive_purpose)
    problems = validate_counts(doc)
    if problems:
        for p in problems[:20]:
            common.log(f"[schema] {p}", err=True)
        common.log(f"[schema] 問題 {len(problems)} 件 — 保存しない。", err=True)
        return 2
    path = common.write_json(out_path(root), doc)
    common.log(f"書き出し: {path} ({len(doc['counts'])} 件)")
    if args.report:
        common.log("")
        common.log(format_report(doc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
