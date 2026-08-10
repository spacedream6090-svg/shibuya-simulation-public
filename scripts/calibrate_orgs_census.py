#!/usr/bin/env python3
"""IV-1 経済センサス(令和3年)町丁目表 → 駅周辺エリアの産業構成を実測し、組織台帳と突き合わせる。

このスクリプトは **読むだけ**(data/realworld/census_r3/ 配下へ集計 JSON を書くほかは何も触らない)。
シミュ本体(src/)からは一切参照されない = conf キー追加ゼロ・engine 非依存(rw_fetch の掟を継承)。

------------------------------------------------------------------ 何を測るのか
現行の組織台帳 `data/organizations_shibuya_wide11k.json` は `build_orgs.py --dist` が
**東京商工会議所渋谷支部が引く事業所統計(平成18・区全域)** の業種別内訳を骨格に生成している
(build_orgs.INDUSTRY_SPEC の share。同ファイル §分布駆動モードの「正直な近似の記録」)。つまり

  (a) 調査年が平成18(2006)= 15 年前、
  (b) 空間が **区全域**(渋谷区 33,284 事業所 / 581,127 人)であって舞台の駅周辺ではない、
  (c) 従業者規模は全国の裾長構造からの近似で、産業別の従業者質量は生成の副産物、

の 3 点で舞台と食い違う。本スクリプトは (a)(b)(c) を **令和3年経済センサス-活動調査の
町丁目別表**で置き換えるための実測値を作る。

------------------------------------------------------------------ 面(エリア)の定義
舞台 = 渋谷駅を中心とする徒歩圏。町丁目は行政界なので「駅から半径 N m」とは一致しない。
ここでは **駅周辺の 13 町丁目**(`STATION_AREA`)の単純和をエリア値とする。境界の町丁目
(松濤・神山町・鶯谷町・恵比寿西 等)は入れていない = エリアは駅至近側に寄せた保守的な定義で、
「渋谷区全域」と「駅前だけ」の中間ではなく **駅前側**である。この線引きは恣意的なので、
JSON の `_meta.area_definition` に町丁目名をそのまま列挙して後から引き直せるようにしてある。

------------------------------------------------------------------ 出典と加工
出典: 経済センサス-活動調査(令和3年)町丁目別統計表
      第28表 区市町村、町丁目、産業大分類別民営事業所数及び従業者数
      第30表 区市町村、町丁目、産業中分類別民営事業所数及び男女別従業者数
政府統計の利用は政府標準利用規約(第2.0版)= CC BY 4.0 互換・商用可。**出典表記が必須**で、
加工した場合はその旨も書く(`_meta.attribution` / `_meta.modified_note`)。

★ 生 xlsx の**取得 URL は記録が残っていない**(本バッチ着手時点で既にローカルにあった)。
   捏造しないため `_meta.source_url` は null のままにし、代わりに検証可能な事実
   (ファイル名・sha256・シート名・シート先頭の表題文字列)を残す。再取得は e-Stat の
   「経済センサス-活動調査」から同じ表番号を辿ること。

使い方:
    python scripts/calibrate_orgs_census.py                    # 集計 + 台帳比較(markdown を stdout)
    python scripts/calibrate_orgs_census.py --no-compare       # 集計だけ
    python scripts/calibrate_orgs_census.py --sim data/organizations_shibuya_wide11k.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:      # rw_fetch を素の python 実行でも import できる
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

# xlsx の読み取りは既存の取得器と**同じ実装を共有**する(openpyxl 非依存・zip+XML のみ)。
# transport_census.py は読むだけで 1 バイトも編集しない。
from rw_fetch.transport_census import (  # noqa: E402
    _fill_forward, norm, read_xlsx_rows, sheet_names, strip_ruby, to_number,
)

# cp932 コンソール(Windows 既定)で Unicode を print しても死なない(devlog-protocol の掟)。
try:  # pragma: no cover - 環境依存
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:  # pragma: no cover
    pass

DATA = REPO_ROOT / "data"
DEFAULT_RAW_DIR = DATA / "realworld" / "census_r3" / "raw"
DEFAULT_OUT_DIR = DATA / "realworld" / "census_r3"
DEFAULT_SIM = DATA / "organizations_shibuya_wide11k.json"

TABLES: dict[str, dict] = {
    "dai": {"file": "ka21_dai.xlsx", "level": "産業大分類",
            "table_no": "第28表",
            "title": "区市町村、町丁目、産業大分類別民営事業所数及び従業者数"},
    "chu": {"file": "ka21_chu.xlsx", "level": "産業中分類",
            "table_no": "第30表",
            "title": "区市町村、町丁目、産業中分類別民営事業所数及び男女別従業者数"},
}

SURVEY = {"name": "経済センサス-活動調査", "year": 2021, "era": "令和3年",
          "reference_date": "2021-06-01", "municipality": "渋谷区"}

# ------------------------------------------------------------------ エリア(駅周辺13町丁目)
# 「丁目」を持つ町(渋谷・道玄坂・神南)は丁目行が末端、持たない町(宇田川町 等)は町行が末端。
# 末端行だけを足す(町行と丁目行を両方足すと二重計上になる)。
STATION_AREA: tuple[str, ...] = (
    "渋谷１丁目", "渋谷２丁目", "渋谷３丁目", "渋谷４丁目",
    "道玄坂１丁目", "道玄坂２丁目",
    "宇田川町",
    "神南１丁目", "神南２丁目",
    "桜丘町", "南平台町", "円山町", "神泉町",
)

CAVEATS = [
    "エリアは駅周辺13町丁目の単純和。町丁目は行政界なので『駅から半径 N m』とは一致しない"
    "(松濤・神山町・鶯谷町・恵比寿西など境界の町丁目は入れていない = 駅至近側に寄せた定義)。",
    "『事業所』は経営主体の拠点であって建物ではない。1 棟のビルに数十事業所が入る。",
    "『従業者数』は**その事業所で働く人**(昼間の就業者)で、夜間人口(居住者)とは別物。"
    "在宅勤務・出張・派遣先常駐の実態は反映されない(調査時点の名簿ベース)。",
    "秘匿(x)・皆無(-)のセルは null として数え、0 で埋めない(`n_missing`)。",
    "町丁目不詳の事業所があるため、町丁目の合計は区総数と一致しない(表の注記どおり)。",
    "中分類表の男女別従業者数は男女不詳を含むため、男+女は総数と一致しないことがある(表の注記どおり)。",
    "令和3年調査は COVID 下(2021-06-01 時点)。飲食・宿泊は平常年より低い可能性がある。",
]

# ------------------------------------------------------------------ 台帳 ↔ センサスの写像
# sim 側の語彙 = build_orgs.INDUSTRY_SPEC の industry_key。censusとの対応は **大分類の文字**で持つ。
# `middles` を持つバケットだけは大分類を中分類で割り、同じ大分類を複数バケットへ分ける。
#   LS(生活関連サービス)と AM(娯楽業)は census 大分類 N を 78/79 と 80 に割る = 台帳側で
#   「美容室(service)」と「ナイトライフ(nightlife)」を別の職場カテゴリとして生やすため。
# `census_only=True` のバケットは現行台帳に存在しない(= --census を渡したときだけ生える)。
SIM_BUCKETS: list[dict] = [
    dict(key="IT", name="情報通信業", majors=["G"]),
    dict(key="WR", name="卸売業・小売業", majors=["I"]),
    dict(key="FB", name="宿泊業・飲食サービス業", majors=["M"]),
    dict(key="PS", name="学術研究・専門・技術サービス業", majors=["L"]),
    dict(key="LS", name="生活関連サービス業", majors=["N"], middles=["78", "79"]),
    dict(key="AM", name="娯楽業", majors=["N"], middles=["80"], census_only=True),
    dict(key="RE", name="不動産業・物品賃貸業", majors=["K"]),
    dict(key="SV", name="サービス業(他に分類されないもの)", majors=["R"]),
    dict(key="MW", name="医療・福祉", majors=["P"]),
    dict(key="ED", name="教育・学習支援業", majors=["O"]),
    dict(key="FI", name="金融業・保険業", majors=["J"]),
    dict(key="CN", name="建設業", majors=["D"]),
    dict(key="MF", name="製造業", majors=["E"]),
    dict(key="TR", name="運輸業・郵便業", majors=["H"]),
    dict(key="CS", name="複合サービス事業", majors=["Q"]),
]
BUCKET_BY_KEY = {b["key"]: b for b in SIM_BUCKETS}

# 台帳に受け皿の無い大分類(= 写像しない。黙って他へ混ぜない)。
UNMAPPED_MAJORS: dict[str, str] = {
    "A": "農業，林業", "B": "漁業", "C": "鉱業，採石業，砂利採取業",
    "F": "電気・ガス・熱供給・水道業",
}


# ============================================================================ パーサ(純関数)
def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _split_major(label: str) -> tuple[str, str]:
    """`A農業，林業` → (`A`, `農業，林業`) / `総数` → (``, `総数`)。ルビは strip_ruby が落とす。"""
    s = strip_ruby(label)
    m = re.match(r"^([A-R])(.+)$", s)
    return (m.group(1), m.group(2)) if m else ("", s)


def _row_with_cells(rows: list[list[str]], *needles: str, limit: int = 12) -> int:
    """指定語を**セルの値そのもの**として持つ最初の行番号。

    rw_fetch の `_find_header` は行を連結して部分一致を見るので、表題行
    (`第28表 …民営事業所数及び従業者数`)を掴んでしまう。ここはセル単位の完全一致で探す。
    """
    for i, row in enumerate(rows[:limit]):
        cells = {strip_ruby(c) for c in row}
        if all(n in cells for n in needles):
            return i
    raise ValueError(f"見出し行が見つからない(探した語: {needles})")


def _pad(row: list[str], width: int) -> list[str]:
    """見出し行を表の最大幅まで空文字で伸ばす。

    xlsx の行は**末尾の空セルを持たない**ので、見出し行が本文行より短いことがある。
    伸ばさずに前方補完すると、右端の分類が所属大分類を失う(実際に踏んだ)。
    """
    row = [str(c or "") for c in (row or [])]
    return row + [""] * max(0, width - len(row))


def _row_with_codes(rows: list[list[str]], limit: int = 12) -> int:
    """2桁の中分類コード(`01`…`95`)が並ぶ行番号。"""
    for i, row in enumerate(rows[:limit]):
        if sum(1 for c in row if re.fullmatch(r"\d{2}", norm(c) or "")) >= 3:
            return i
    raise ValueError("中分類コード(2桁)の行が見つからない")


def parse_dai(rows: list[list[str]]) -> tuple[list[dict], int]:
    """第28表(大分類×町丁目) → 長形式レコード。

    列の構造(実測): 見出し3列(区市町村 / 町 / 丁目)+ [総数] + 大分類 A..R の
    それぞれ [事業所数, 従業者数] の対 = 41 列。
    戻り値 = (レコード, 欠測セル数)。レコードは 1 行 × 1 分類 = 1 件。
    """
    head = _row_with_cells(rows, "事業所数", "従業者数")
    width = max(len(r) for r in rows[max(0, head - 2):head + 1])
    measures = [strip_ruby(c) for c in _pad(rows[head], width)]
    grp1 = _fill_forward([strip_ruby(c) for c in _pad(rows[head - 1], width)]) if head >= 1 else []
    grp0 = _fill_forward([strip_ruby(c) for c in _pad(rows[head - 2], width)]) if head >= 2 else []
    cols: list[tuple[int, int, str, str]] = []      # (事業所列, 従業者列, code, name)
    for i, meas in enumerate(measures):
        if meas != "事業所数":
            continue
        label = (grp1[i] if i < len(grp1) else "") or (grp0[i] if i < len(grp0) else "")
        code, name = _split_major(label)
        if not code and name != "総数":
            continue
        cols.append((i, i + 1, code or "総数", name))
    if not cols:
        raise ValueError("事業所数/従業者数 の列対が1つも無い")

    out: list[dict] = []
    n_missing = 0
    ward = town = ""
    for row in rows[head + 1:]:
        def cell(j: int) -> str:
            return str(row[j] or "").strip() if j < len(row) else ""
        c0, c1, c2 = cell(0), cell(1), cell(2)
        if c0.startswith("注"):                       # 表末尾の注記
            continue
        if c0:
            ward, town = c0, ""
            level, block = "ward", c0
        elif c1:
            town = c1
            level, block = "town", c1
        elif c2:
            level, block = "block", c2
        else:
            continue
        for i_e, i_p, code, name in cols:
            est = to_number(cell(i_e))
            emp = to_number(cell(i_p))
            n_missing += (est is None) + (emp is None)
            out.append({"level": level, "ward": ward, "town": town or block, "block": block,
                        "class_level": "総数" if code == "総数" else "大分類",
                        "code": code, "name": name,
                        "establishments": est, "employees": emp})
    return out, n_missing


def parse_chu(rows: list[list[str]]) -> tuple[list[dict], int]:
    """第30表(中分類×町丁目) → 長形式レコード。

    行の構造(実測): 町丁目の見出し行のあとに [事業所数][従業者数][男][女] の 4 行が続く。
    列の構造(実測): 見出し5列 + [総数] + 大分類 A..R の小計列 + 中分類 01..95 の列。
    """
    head = _row_with_codes(rows) + 1                 # コード行の1つ下 = 分類名の行
    width = max(len(r) for r in rows[max(0, head - 2):head + 1])
    codes = [norm(c) for c in _pad(rows[head - 1], width)] if head >= 1 else []
    raw_letters = [norm(c) for c in _pad(rows[head - 2], width)] if head >= 2 else [""] * width
    letters = _fill_forward(raw_letters)
    names = [strip_ruby(c) for c in _pad(rows[head], width)]
    cols: list[tuple[int, str, str, str, str]] = []   # (列, class_level, code, name, major)
    for i in range(width):
        raw_letter = raw_letters[i] if i < len(raw_letters) else ""
        code = codes[i] if i < len(codes) else ""
        if raw_letter == "総数":
            cols.append((i, "総数", "総数", "総数", ""))
        elif re.fullmatch(r"\d{2}", code or ""):
            cols.append((i, "中分類", code, names[i], letters[i] if i < len(letters) else ""))
        elif re.fullmatch(r"[A-R]", raw_letter or ""):
            cols.append((i, "大分類", raw_letter, names[i], raw_letter))
    if not any(c[1] == "中分類" for c in cols):
        raise ValueError("中分類(2桁コード)の列が1つも無い")

    out: list[dict] = []
    n_missing = 0
    ward = town = block = ""
    level = "ward"
    for row in rows[head + 1:]:
        def cell(j: int) -> str:
            return str(row[j] or "").strip() if j < len(row) else ""
        c0, c1, c2 = cell(0), cell(1), cell(2)
        if c0.startswith("注"):
            continue
        if c0:
            ward, town, block, level = c0, "", c0, "ward"
            continue
        if c1:
            town, block, level = c1, c1, "town"
            continue
        if c2:
            block, level = c2, "block"
            continue
        meas = strip_ruby(cell(3))
        sex = strip_ruby(cell(4))
        if meas == "事業所数":
            field = "establishments"
        elif meas == "従業者数":
            field = "employees"
        elif sex == "男":
            field = "employees_male"
        elif sex == "女":
            field = "employees_female"
        else:
            continue
        for i, class_level, code, name, major in cols:
            val = to_number(cell(i))
            if val is None:
                n_missing += 1
            out.append({"level": level, "ward": ward, "town": town or block, "block": block,
                        "class_level": class_level, "code": code, "name": name,
                        "major": major, "field": field, "value": val})
    return out, n_missing


# ============================================================================ 集計(純関数)
def leaf_blocks(records: list[dict]) -> set[str]:
    """末端の町丁目名(丁目を持つ町の『町』行を除いた集合)。二重計上の防止に使う。"""
    towns_with_blocks = {r["town"] for r in records if r["level"] == "block"}
    out = {r["block"] for r in records if r["level"] == "block"}
    out |= {r["block"] for r in records if r["level"] == "town" and r["town"] not in towns_with_blocks}
    return out


def _sum_dai(dai: list[dict], blocks: set[str]) -> dict[str, dict]:
    """大分類 → {establishments, employees}(エリア内の末端行だけを足す)。"""
    acc: dict[str, dict] = {}
    for r in dai:
        if r["block"] not in blocks or r["level"] == "ward":
            continue
        d = acc.setdefault(r["code"], {"code": r["code"], "name": r["name"],
                                       "establishments": 0, "employees": 0})
        d["establishments"] += int(r["establishments"] or 0)
        d["employees"] += int(r["employees"] or 0)
    return acc


def _sum_chu(chu: list[dict], blocks: set[str]) -> dict[str, dict]:
    """中分類コード → {establishments, employees, employees_male, employees_female, major}。"""
    acc: dict[str, dict] = {}
    for r in chu:
        if r["block"] not in blocks or r["level"] == "ward" or r["class_level"] != "中分類":
            continue
        d = acc.setdefault(r["code"], {"code": r["code"], "name": r["name"], "major": r["major"],
                                       "establishments": 0, "employees": 0,
                                       "employees_male": 0, "employees_female": 0})
        d[r["field"]] += int(r["value"] or 0)
    return acc


def _sum_chu_major(chu: list[dict], blocks: set[str]) -> dict[str, dict]:
    """中分類表側の大分類小計(第28表との突き合わせ = 検算用)。"""
    acc: dict[str, dict] = {}
    for r in chu:
        if r["block"] not in blocks or r["level"] == "ward" or r["class_level"] != "大分類":
            continue
        d = acc.setdefault(r["code"], {"establishments": 0, "employees": 0})
        if r["field"] in d:
            d[r["field"]] += int(r["value"] or 0)
    return acc


def _block_totals(dai: list[dict], blocks: set[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in dai:
        if r["block"] not in blocks or r["level"] == "ward" or r["code"] != "総数":
            continue
        out[r["block"]] = {"establishments": int(r["establishments"] or 0),
                           "employees": int(r["employees"] or 0)}
    return out


def sim_buckets(by_major: dict[str, dict], by_middle: dict[str, dict]) -> tuple[dict, dict]:
    """センサス集計 → 台帳の産業バケット(build_orgs.industry_key)へ写像。

    `middles` を持つバケットは中分類の和、持たないバケットは大分類の値をそのまま使う。
    戻り値 = (buckets, unmapped)。`unmapped` は台帳に受け皿の無い大分類(黙って混ぜない)。
    """
    buckets: dict[str, dict] = {}
    for b in SIM_BUCKETS:
        est = emp = 0
        if b.get("middles"):
            for code in b["middles"]:
                m = by_middle.get(code)
                if m:
                    est += int(m["establishments"])
                    emp += int(m["employees"])
        else:
            for mj in b["majors"]:
                d = by_major.get(mj)
                if d:
                    est += int(d["establishments"])
                    emp += int(d["employees"])
        buckets[b["key"]] = {
            "industry": b["name"], "census_majors": list(b["majors"]),
            "census_middles": list(b.get("middles", [])),
            "census_only": bool(b.get("census_only", False)),
            "establishments": est, "employees": emp,
        }
    unmapped = {}
    for code, name in UNMAPPED_MAJORS.items():
        d = by_major.get(code)
        unmapped[code] = {"name": name,
                          "establishments": int(d["establishments"]) if d else 0,
                          "employees": int(d["employees"]) if d else 0}
    return buckets, unmapped


def aggregate(dai: list[dict], chu: list[dict], area: tuple[str, ...] = STATION_AREA) -> dict:
    """駅周辺エリアの産業構成を組み立てる(大分類・中分類・町丁目別・台帳バケット)。"""
    known = leaf_blocks(dai)
    missing = [b for b in area if b not in known]
    if missing:
        raise ValueError(f"エリア指定の町丁目が表に無い(全角数字を確認): {missing}")
    blocks = set(area)
    by_major = _sum_dai(dai, blocks)
    total = by_major.pop("総数", {"establishments": 0, "employees": 0})
    by_middle = _sum_chu(chu, blocks) if chu else {}
    cross = _sum_chu_major(chu, blocks) if chu else {}
    buckets, unmapped = sim_buckets(by_major, by_middle)
    # 検算: 大分類の和 = 総数 / 第28表 と 第30表 の大分類が一致するか。
    sum_major_est = sum(d["establishments"] for d in by_major.values())
    sum_major_emp = sum(d["employees"] for d in by_major.values())
    cross_diff = {k: {"establishments": cross[k]["establishments"] - by_major[k]["establishments"],
                      "employees": cross[k]["employees"] - by_major[k]["employees"]}
                  for k in sorted(set(cross) & set(by_major))
                  if cross[k] != {"establishments": by_major[k]["establishments"],
                                  "employees": by_major[k]["employees"]}}
    return {
        "area": {"blocks": list(area), "n_blocks": len(area),
                 "establishments": int(total["establishments"]),
                 "employees": int(total["employees"]),
                 "employees_per_establishment": round(
                     total["employees"] / total["establishments"], 2)
                 if total["establishments"] else 0.0},
        "by_major": {k: by_major[k] for k in sorted(by_major)},
        "by_middle": {k: by_middle[k] for k in sorted(by_middle)},
        "by_block": _block_totals(dai, blocks),
        "sim_buckets": buckets,
        "unmapped_majors": unmapped,
        "checks": {
            "sum_of_majors_establishments": sum_major_est,
            "sum_of_majors_employees": sum_major_emp,
            "total_establishments": int(total["establishments"]),
            "total_employees": int(total["employees"]),
            "majors_sum_matches_total": (sum_major_est == int(total["establishments"])
                                         and sum_major_emp == int(total["employees"])),
            "dai_vs_chu_major_diff": cross_diff,
            "dai_vs_chu_match": not cross_diff,
        },
    }


# ============================================================================ 台帳との比較
def sim_industry_stats(ledger: dict) -> dict:
    """組織台帳 → industry_key ごとの (件数, 従業者)。"""
    out: dict[str, dict] = {}
    for c in ledger.get("companies", []):
        key = str(c.get("industry_key", ""))
        d = out.setdefault(key, {"industry": c.get("industry", ""), "count": 0, "employees": 0})
        d["count"] += 1
        d["employees"] += int(c.get("size", {}).get("employees", 0) or 0)
    return out


def compare(agg: dict, sim: dict) -> dict:
    """台帳 vs センサス。**シェアで比べる**(台帳 1.1 万 : 実測 9,872 の規模差を潰す)。"""
    buckets = agg["sim_buckets"]
    sim_n = sum(d["count"] for d in sim.values()) or 1
    sim_e = sum(d["employees"] for d in sim.values()) or 1
    cen_n = sum(b["establishments"] for b in buckets.values()) or 1
    cen_e = sum(b["employees"] for b in buckets.values()) or 1
    rows = []
    for key in sorted(set(buckets) | set(sim), key=lambda k: -buckets.get(k, {}).get("employees", 0)):
        b = buckets.get(key, {})
        s = sim.get(key, {})
        c_n, c_e = int(b.get("establishments", 0)), int(b.get("employees", 0))
        s_n, s_e = int(s.get("count", 0)), int(s.get("employees", 0))
        rows.append({
            "key": key,
            "industry": b.get("industry") or s.get("industry", ""),
            "census_establishments": c_n, "census_establishment_share": round(c_n / cen_n, 4),
            "census_employees": c_e, "census_employee_share": round(c_e / cen_e, 4),
            "sim_count": s_n, "sim_count_share": round(s_n / sim_n, 4),
            "sim_employees": s_e, "sim_employee_share": round(s_e / sim_e, 4),
            "count_share_ratio": round((s_n / sim_n) / (c_n / cen_n), 3) if c_n else None,
            "employee_share_ratio": round((s_e / sim_e) / (c_e / cen_e), 3) if c_e else None,
            "in_sim_ledger": key in sim,
            "census_only": bool(b.get("census_only", False)),
        })
    return {
        "totals": {"sim_orgs": sim_n, "sim_employees": sim_e,
                   "census_establishments": cen_n, "census_employees": cen_e,
                   "sim_employees_per_org": round(sim_e / sim_n, 2),
                   "census_employees_per_establishment": round(cen_e / cen_n, 2)},
        "rows": rows,
        "unmapped_majors": agg["unmapped_majors"],
        "unmapped_sim_keys": sorted(set(sim) - set(buckets)),
        "note": ("台帳は約1.1万社・センサスのエリアは9,872事業所なので、比較は**シェア**で行う。"
                 "ratio > 1 = 台帳が過大、< 1 = 過小。"),
    }


def comparison_markdown(cmp: dict, agg: dict) -> str:
    t = cmp["totals"]
    lines = [
        "# 組織台帳 vs 経済センサス(令和3年・駅周辺13町丁目)", "",
        f"- センサス実測: **{t['census_establishments']:,} 事業所 / {t['census_employees']:,} 人**"
        f" (平均 {t['census_employees_per_establishment']} 人/事業所)",
        f"- 現行台帳     : **{t['sim_orgs']:,} 社 / {t['sim_employees']:,} 人**"
        f" (平均 {t['sim_employees_per_org']} 人/社)",
        f"- エリア: {' / '.join(agg['area']['blocks'])}", "",
        "## 産業別(シェア比較。ratio>1=台帳が過大 / <1=過小)", "",
        "| key | 産業 | センサス事業所 | 実測% | 台帳社数 | 台帳% | 件数比 | センサス従業者 | 実測% | 台帳従業者 | 台帳% | 従業者比 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in cmp["rows"]:
        cr = "-" if r["count_share_ratio"] is None else f"{r['count_share_ratio']:.2f}"
        er = "-" if r["employee_share_ratio"] is None else f"{r['employee_share_ratio']:.2f}"
        mark = "" if r["in_sim_ledger"] else " *(台帳に無い)*"
        lines.append(
            f"| {r['key']} | {r['industry']}{mark} | {r['census_establishments']:,} | "
            f"{r['census_establishment_share']:.1%} | {r['sim_count']:,} | {r['sim_count_share']:.1%} | {cr} | "
            f"{r['census_employees']:,} | {r['census_employee_share']:.1%} | "
            f"{r['sim_employees']:,} | {r['sim_employee_share']:.1%} | {er} |")
    lines += ["", "## 台帳に受け皿の無い大分類(写像せず・他へ混ぜない)", "",
              "| 大分類 | 名称 | 事業所 | 従業者 |", "|---|---|---:|---:|"]
    for code, d in cmp["unmapped_majors"].items():
        lines.append(f"| {code} | {d['name']} | {d['establishments']:,} | {d['employees']:,} |")
    if cmp["unmapped_sim_keys"]:
        lines += ["", f"★ センサスへ写像できない台帳キー: {', '.join(cmp['unmapped_sim_keys'])}"]
    lines += ["", "## 中分類の上位(従業者数)", "",
              "| 中分類 | 名称 | 事業所 | 従業者 |", "|---|---|---:|---:|"]
    top = sorted(agg["by_middle"].values(), key=lambda d: -d["employees"])[:15]
    for d in top:
        lines.append(f"| {d['code']} | {d['name']} | {d['establishments']:,} | {d['employees']:,} |")
    lines += ["", cmp["note"], ""]
    return "\n".join(lines)


# ============================================================================ 入出力
def load_tables(raw_dir: Path) -> tuple[list[dict], list[dict], dict]:
    """生 xlsx 2 表を読んでレコード化 + 出所メタ(sha256・シート名・表題)を作る。"""
    meta_tables = {}
    parsed: dict[str, list[dict]] = {}
    n_missing = 0
    for key, spec in TABLES.items():
        path = Path(raw_dir) / spec["file"]
        if not path.exists():
            raise FileNotFoundError(
                f"生 xlsx が無い: {path}\n"
                f"  経済センサス(令和3年)町丁目別表 {spec['table_no']} を data/realworld/census_r3/raw/ へ置くこと")
        body = path.read_bytes()
        rows = read_xlsx_rows(body)
        recs, miss = (parse_dai(rows) if key == "dai" else parse_chu(rows))
        parsed[key] = recs
        n_missing += miss
        meta_tables[key] = {
            "file": spec["file"], "sha256": _sha256(body),
            "sheet": (sheet_names(body) or [""])[0],
            "sheet_title": strip_ruby(rows[0][0]) if rows and rows[0] else "",
            "table_no": spec["table_no"], "title": spec["title"], "class_level": spec["level"],
            "n_rows_source": len(rows), "n_records": len(recs), "n_missing": miss,
        }
    return parsed["dai"], parsed["chu"], {"tables": meta_tables, "n_missing": n_missing}


def build_document(raw_dir: Path, area: tuple[str, ...] = STATION_AREA) -> dict:
    dai, chu, src = load_tables(Path(raw_dir))
    agg = aggregate(dai, chu, area)
    meta = {
        "generator": "scripts/calibrate_orgs_census.py",
        "survey": dict(SURVEY),
        "source": "経済センサス-活動調査(令和3年)町丁目別統計表",
        "source_url": None,
        "provenance_note": ("生 xlsx の取得 URL は記録が残っていないため null。検証可能な事実"
                            "(ファイル名/sha256/シート名/表題)を tables に残してある。再取得は"
                            "e-Stat の『経済センサス-活動調査』から同じ表番号を辿ること。"),
        "attribution": "出典: 総務省・経済産業省「令和3年経済センサス-活動調査」(町丁目別統計表)",
        "modified_note": ("加工: 駅周辺13町丁目の末端行のみを抽出して単純和を取り、産業大分類・"
                          "中分類・町丁目別に再集計した。原表の値そのものは改変していない。"),
        "license": "政府標準利用規約(第2.0版)= CC BY 4.0 互換。出典表記が必須。",
        "area_definition": {"name": "渋谷駅周辺(13町丁目)", "blocks": list(area),
                            "rule": "丁目を持つ町は丁目行、持たない町は町行(= 末端行)だけを足す"},
        "caveats": list(CAVEATS),
        "units": {"establishments": "事業所", "employees": "人"},
        "sim_mapping": {b["key"]: {"industry": b["name"], "majors": b["majors"],
                                   "middles": b.get("middles", []),
                                   "census_only": bool(b.get("census_only", False))}
                        for b in SIM_BUCKETS},
        "unmapped_majors_note": "台帳に受け皿の無い大分類は写像せず、`unmapped_majors` に残す(他へ混ぜない)。",
    }
    meta.update(src)
    return {"_meta": meta, **agg}


def _write_json(path: Path, doc: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="経済センサス(令和3)町丁目表 → 駅周辺の産業構成 + 台帳比較")
    ap.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR), help="生 xlsx の置き場")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="集計 JSON の出力先")
    ap.add_argument("--sim", default=str(DEFAULT_SIM), help="比較する組織台帳 JSON")
    ap.add_argument("--no-compare", action="store_true", help="集計だけ(台帳比較をしない)")
    args = ap.parse_args(argv)

    doc = build_document(Path(args.raw_dir))
    out_dir = Path(args.out_dir)
    p1 = _write_json(out_dir / "station_area_industry.json", doc)
    a = doc["area"]
    print(f"written: {p1}")
    print(f"  駅周辺{a['n_blocks']}町丁目: {a['establishments']:,} 事業所 / {a['employees']:,} 人 "
          f"(平均 {a['employees_per_establishment']} 人/事業所)")
    chk = doc["checks"]
    print(f"  検算: 大分類の和=総数 {chk['majors_sum_matches_total']} / "
          f"第28表と第30表の大分類一致 {chk['dai_vs_chu_match']} / 欠測セル {doc['_meta']['n_missing']}")
    if args.no_compare:
        return 0

    sim_path = Path(args.sim)
    if not sim_path.is_absolute():
        sim_path = REPO_ROOT / sim_path
    if not sim_path.exists():
        print(f"台帳が無いので比較を飛ばす: {sim_path}", file=sys.stderr)
        return 0
    ledger = json.loads(sim_path.read_text(encoding="utf-8"))
    cmp = compare(doc, sim_industry_stats(ledger))
    cmp["sim_file"] = str(sim_path.relative_to(REPO_ROOT)).replace("\\", "/")
    p2 = _write_json(out_dir / "sim_vs_census.json", {"_meta": doc["_meta"], **cmp})
    print(f"written: {p2}\n")
    print(comparison_markdown(cmp, doc))
    return 0


if __name__ == "__main__":                                   # pragma: no cover
    raise SystemExit(main())
