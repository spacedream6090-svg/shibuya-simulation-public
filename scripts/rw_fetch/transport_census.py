#!/usr/bin/env python3
"""P4境界較正 A: 大都市交通センサス(国土交通省)の駅別・時間帯別集計表。

**この取得器は日次スケジューラの対象ではない。** センサスは5年に1度の静的調査なので、
`scripts/rw_fetch_daily.py` には一切結線していない(= Windows タスクを触らない)。
使い方は単発の `python scripts/build_boundary_counts.py --fetch` だけ。

------------------------------------------------------------------ 実測(2026-08-10)
★ **第13回(2021年度・調査は post-COVID)には駅別×時間帯の表が存在しない。**
   e-Stat の第13回は 8 ファイルだけ(定期券調査2 + 一件明細調査6)で、いずれも
   ゾーン間 OD・駅間 OD・所要時間であって時間帯軸を持たない。
   → **駅×時刻の無料の一次表は存在しない**。第12回(2015年度・pre-COVID)の
     「水準(駅別)」×「形(圏域全体の時刻分布)」の分解で組み立てるしかない。
     この分解と限界は docs/research/boundary-calibration-data.md に明記した。

したがって本モジュールが取るのは次の5表(既定は第12回 首都圏の4表)。

  station_flow  第12回 資料編 表3「駅別発着・駅間通過人員表」     … 水準(人/日)
  time_dist     第12回 参考表 表5「目的別乗車降車時刻分布」       … 形(時刻・圏域全体)
  transfer      第12回 資料編 表4「ターミナル別乗換え人員表」     … 改札外/乗換の分解
  line_capacity 第12回 資料編 表9「路線別着時間帯別駅間輸送定員表」… 供給側の時間帯形
  station_od13  第13回 一件明細「初乗り・最終降車駅間移動人員」   … post-COVID 水準(既定OFF・18MB)

★ 保存物は **渋谷に触れる行だけ**に絞る(首都圏全駅だと station_flow だけで4万件を超える)。
   絞ったことは `_meta.station_filter` と `_meta.n_records_total` に必ず残す。生 xlsx は
   `data/realworld/census/raw/` にそのまま保存してあるので、後から絞り直せる。

★ ライセンス: 国交省サイトは公共データ利用規約(第1.0版)= PDL1.0、e-Stat 配布分は
   政府標準利用規約(第2.0版)で CC BY 4.0 互換・商用可。どちらも **出典表記が必須**で、
   加工した場合はその旨も書く(`_meta.attribution` / `_meta.modified_note`)。
"""
from __future__ import annotations

import io
import json
import re
import zipfile
import xml.etree.ElementTree as ET
from datetime import date as _date
from pathlib import Path

from . import common, ledger

SOURCE = "transport_census"
MLIT = "https://www.mlit.go.jp"
ESTAT = "https://www.e-stat.go.jp"

# 第12回の一覧ページ(表番号と xlsx の対応はここから実取得して確認した)。
INDEX_12 = f"{MLIT}/sogoseisaku/transport/sosei_transport_tk_000035.html"
INDEX_TOP = f"{MLIT}/sogoseisaku/transport/sosei_transport_tk_000007.html"
# 第13回は e-Stat のみ。手で辿る場合の入口(研究文書にも同じ URL を書いてある)。
ESTAT_PORTAL = f"{ESTAT}/stat-search/files?page=1&toukei=00600020&tstat=000001103355"

# 既定の対象駅。センサスの駅名は事業者をまたいで同一表記なのでこれで足りる。
DEFAULT_STATIONS = ("渋谷",)

TABLES: list[dict] = [
    {
        "key": "station_flow",
        "survey": 12, "survey_year": 2015, "regime": "pre_covid",
        "title": "第12回大都市交通センサス 鉄道調査 報告書資料編 表3 駅別発着・駅間通過人員表(首都圏)",
        "url": f"{MLIT}/common/001178992.xlsx",
        "parser": "station_flow", "role": "level", "unit": "persons_per_day",
        "default": True,
    },
    {
        "key": "time_dist",
        "survey": 12, "survey_year": 2015, "regime": "pre_covid",
        "title": "第12回大都市交通センサス 鉄道調査 参考表 表5 目的別乗車降車時刻分布(首都圏)",
        "url": f"{MLIT}/common/001179635.xlsx",
        "parser": "time_dist", "role": "shape", "unit": "share_of_daily",
        "default": True,
    },
    {
        "key": "transfer",
        "survey": 12, "survey_year": 2015, "regime": "pre_covid",
        "title": "第12回大都市交通センサス 鉄道調査 報告書資料編 表4 ターミナル別乗換え人員表(首都圏)",
        "url": f"{MLIT}/common/001178993.xlsx",
        "parser": "transfer", "role": "decomposition", "unit": "persons_per_day",
        "default": True,
    },
    {
        "key": "line_capacity",
        "survey": 12, "survey_year": 2015, "regime": "pre_covid",
        "title": "第12回大都市交通センサス 鉄道調査 報告書資料編 表9 路線別着時間帯別駅間輸送定員表(首都圏)",
        "url": f"{MLIT}/common/001178999.xlsx",
        "parser": "line_capacity", "role": "capacity", "unit": "seats_persons",
        "default": True,
    },
    {
        "key": "station_od13",
        "survey": 13, "survey_year": 2021, "regime": "post_covid",
        "title": "第13回大都市交通センサス 一件明細調査 初乗り・最終降車駅間移動人員(首都圏)",
        # e-Stat の直リンク(statInfId は 2026-08-10 に実取得して確認)。
        "url": f"{ESTAT}/stat-search/file-download?statInfId=000040443413&fileKind=0",
        "parser": "station_od", "role": "level", "unit": "persons_per_day",
        # 18MB・725,830 行。既定では取らない(--with-od13 で有効化)。
        "default": False, "large": True,
    },
]
TABLE_BY_KEY = {t["key"]: t for t in TABLES}

CAVEATS = [
    "★第13回(2021年度)には駅別×時間帯の集計表が無い。無料で手に入る駅×時刻の一次表は"
    "存在せず、第12回(2015年度)の『駅別水準』×『圏域全体の時刻分布』でしか組めない。",
    "第12回は2015年度 = pre-COVID。第13回は2021年度 = COVID 下の調査で、鉄道需要は"
    "平常時より大きく低い。両者を同じ系列として並べてはいけない。",
    "駅別発着表の『乗車』は**その路線に乗った人数**で、改札外からの入場と改札内乗換を"
    "区別しない。同一駅の複数路線を足すと駅内乗換が二重に入る(渋谷では乗換が支配的)。",
    "ターミナル別乗換え人員表の『初乗り計』『最終降車計』は乗換施設の実態調査に基づく"
    "部分集計で、駅別発着表の水準とは桁が合わない(渋谷: 初乗り計 6,863 人/日 に対し"
    "山手線だけで乗車 30 万人/日 超)。**この表は乗換行列と乗換比率にだけ使い、"
    "改札流入出の絶対水準には使わない。**",
    "路線別着時間帯別表の値は『輸送定員』= 供給側の座席・定員であって実乗客数ではない。",
    "時刻分布表は首都圏全体・目的別の構成比(合計1.0)で、駅別の次元を持たない。",
    "保存物は渋谷に触れる行だけに絞ってある(_meta.station_filter)。全件は"
    "data/realworld/census/raw/ の生 xlsx 側にある。",
]


# ================================================================ xlsx(標準ライブラリのみ)
_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def _col_index(ref: str) -> int:
    """セル参照 `BC12` → 列番号(0 起点)。"""
    m = re.match(r"([A-Z]+)", ref or "")
    if not m:
        return 0
    n = 0
    for ch in m.group(1):
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def read_xlsx_rows(data: bytes, sheet_index: int = 0) -> list[list[str]]:
    """xlsx → 行×セル(すべて文字列)。**openpyxl に依存しない**(zip + XML だけ)。

    値は書式を当てずに生のまま返す(数値は `"121926"` / `"6.44E-2"` のような文字列)。
    結合セルは左上だけに値が入るので、意味づけは各パーサ側で前方補完する。
    """
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ValueError(f"xlsx として開けない(zip でない): {exc}") from exc
    names = sorted(n for n in zf.namelist() if re.match(r"xl/worksheets/sheet\d+\.xml$", n))
    if not names:
        raise ValueError("xlsx にワークシートが無い")
    if not 0 <= sheet_index < len(names):
        raise ValueError(f"シート番号が範囲外: {sheet_index} (シート数 {len(names)})")
    shared: list[str] = []
    if "xl/sharedStrings.xml" in zf.namelist():
        root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
        for si in root.findall(f"{_NS}si"):
            shared.append("".join(t.text or "" for t in si.iter(f"{_NS}t")))
    root = ET.fromstring(zf.read(names[sheet_index]))
    rows: list[list[str]] = []
    for row in root.iter(f"{_NS}row"):
        cells: dict[int, str] = {}
        for c in row.findall(f"{_NS}c"):
            kind = c.get("t")
            if kind == "inlineStr":
                node = c.find(f"{_NS}is")
                val = "".join(t.text or "" for t in node.iter(f"{_NS}t")) if node is not None else ""
            else:
                v = c.find(f"{_NS}v")
                if v is None or v.text is None:
                    val = ""
                elif kind == "s":
                    idx = int(v.text)
                    val = shared[idx] if 0 <= idx < len(shared) else ""
                else:
                    val = v.text
            cells[_col_index(c.get("r") or "")] = val
        width = max(cells) + 1 if cells else 0
        rows.append([cells.get(i, "") for i in range(width)])
    return rows


def sheet_names(data: bytes) -> list[str]:
    zf = zipfile.ZipFile(io.BytesIO(data))
    wb = zf.read("xl/workbook.xml").decode("utf-8", "replace")
    return re.findall(r'<sheet[^>]*name="([^"]+)"', wb)


# ================================================================ セルの下ごしらえ
_KATAKANA_RUN = re.compile(r"[ァ-ヴー]+$")


def norm(cell) -> str:
    """全角空白・改行・前後空白を潰した比較用の文字列。"""
    return re.sub(r"[\s　]+", "", str(cell or ""))


def strip_ruby(label) -> str:
    """センサス表のセルは `駅名エキメイ` のように**ルビが本文に連結**されている。

    末尾のカタカナ列は、それが**残りの文字列と同じかそれ以上の長さ**のときだけルビとみなす。
    `東海道本線トウカイドウホンセン` → `東海道本線`(10 >= 5 で除去)
    `湘南新宿ライン` → そのまま(`ライン` は 3 < 4 なので路線名の一部として残す)
    """
    s = re.sub(r"[\s　]+", "", str(label or ""))
    m = _KATAKANA_RUN.search(s)
    if not m:
        return s
    run = m.group(0)
    rest = s[: m.start()]
    if rest and len(run) >= len(rest):
        return rest
    return s


def clean_time_label(label) -> str:
    """時間帯の見出し(`始発～6:59シハツ`)からルビを落とす。

    `strip_ruby` の「末尾カタカナが残りより長いときだけ」規則では `始発～6:59シハツ` の
    ルビが落ちない(3 < 8)。時間帯の見出しは数字・`:`・`～` しか含まないので、
    **この用途に限って**カタカナを全部落とす。
    """
    return re.sub(r"[ァ-ヴ]+", "", re.sub(r"[\s　]+", "", str(label or "")))


def to_number(cell):
    """`'121926'` → int / `'6.44E-2'` → float / 空・`-` → None。桁区切りは許す。"""
    s = re.sub(r"[\s　,]", "", str(cell or ""))
    if s in ("", "-", "－", "―", "…", "x", "X"):
        return None
    try:
        if re.fullmatch(r"[+-]?\d+", s):
            return int(s)
        return float(s)
    except ValueError:
        return None


def _fill_forward(values: list[str]) -> list[str]:
    """結合セル由来の空欄を直前の値で埋める(見出し行の意味づけ用)。"""
    out, last = [], ""
    for v in values:
        s = str(v or "").strip()
        if s:
            last = s
        out.append(last)
    return out


def _find_header(rows: list[list[str]], *needles: str, limit: int = 12) -> int:
    """先頭 `limit` 行から、指定語をすべて含む行の番号を探す。無ければ ValueError。"""
    for i, row in enumerate(rows[:limit]):
        joined = norm("".join(str(c) for c in row))
        if all(n in joined for n in needles):
            return i
    raise ValueError(f"見出し行が見つからない(探した語: {needles})")


# ================================================================ パーサ(純関数)
def parse_station_flow(rows: list[list[str]], stations=None) -> tuple[list[dict], int]:
    """表3 駅別発着・駅間通過人員表 → 長形式レコード。

    列の構造(実測): `駅名` + [定期券/普通券/合計] × [下り/上り] × [乗車/降車/通過] = 18 列。
    路線名は**データ行と同じ列0に単独で現れる小見出し**なので前方に持ち回る。
    戻り値 = (レコード, 欠測セル数)。`stations` を渡すとその駅だけに絞る。
    """
    head = _find_header(rows, "駅名", "乗車", "降車", "通過")
    measures = [strip_ruby(c) for c in rows[head]]
    tickets = _fill_forward([strip_ruby(c) for c in rows[head - 2]]) if head >= 2 else []
    dirs = _fill_forward([strip_ruby(c) for c in rows[head - 1]]) if head >= 1 else []
    cols: list[tuple[int, str, str, str]] = []
    for i in range(1, len(measures)):
        meas = measures[i]
        if meas in ("乗車", "降車", "通過"):
            cols.append((i, tickets[i] if i < len(tickets) else "",
                         dirs[i] if i < len(dirs) else "", meas))
    if not cols:
        raise ValueError("乗車/降車/通過 の列が1つも無い")
    want = set(stations) if stations else None
    out: list[dict] = []
    n_missing = 0
    line = ""
    for row in rows[head + 1:]:
        name = str(row[0] or "").strip() if row else ""
        if not name:
            continue
        rest = [str(c or "").strip() for c in row[1:]]
        if not any(rest):                      # 値がゼロ列 = 路線の小見出し
            line = strip_ruby(name)
            continue
        if want is not None and name not in want:
            continue
        for idx, ticket, direction, meas in cols:
            raw = row[idx] if idx < len(row) else ""
            val = to_number(raw)
            if val is None:
                n_missing += 1
            out.append({"line": line, "station": name, "ticket": ticket,
                        "direction": direction, "measure": meas, "value": val})
    return out, n_missing


def parse_time_dist(rows: list[list[str]]) -> tuple[list[dict], int]:
    """参考表5 目的別乗車降車時刻分布 → 目的×乗降×時間帯の構成比。

    時間帯は `～6時台` / `7時台`…`23時台` / `0時台～` の 19 区分 + `合計`(=1.0)。
    `合計` 列はレコードにせず、検算だけに使う(戻り値の `_check` ではなく呼び出し側で見る)。
    """
    head = _find_header(rows, "利用目的", "乗降")
    header = [strip_ruby(c) for c in rows[head]]
    bins: list[tuple[int, str]] = []
    total_col = None
    for i, label in enumerate(header):
        if i < 2 or not label:
            continue
        if label == "合計":
            total_col = i
            continue
        if "時台" in label:
            bins.append((i, label))
    if not bins:
        raise ValueError("時間帯の列(`…時台`)が見つからない")
    out: list[dict] = []
    n_missing = 0
    purpose = ""
    for row in rows[head + 1:]:
        first = strip_ruby(row[0]) if row else ""
        flow = strip_ruby(row[1]) if len(row) > 1 else ""
        if first and flow not in ("乗車", "降車"):
            continue                              # 注記行など
        if first:
            purpose = first
        if flow not in ("乗車", "降車"):
            continue
        total = to_number(row[total_col]) if (total_col is not None
                                              and total_col < len(row)) else None
        for idx, label in bins:
            val = to_number(row[idx]) if idx < len(row) else None
            if val is None:
                n_missing += 1
            out.append({"purpose": purpose, "flow": flow, "bin": label,
                        "share": val, "row_total": total})
    return out, n_missing


def parse_transfer(rows: list[list[str]], terminals=None) -> tuple[list[dict], int]:
    """表4 ターミナル別乗換え人員表 → 到着線→行先線の乗換行列 + ピーク時。

    行の種別:
      `到着線名 == "初乗り"`   … 改札外から入場してその路線に乗る(=gate_entry)
      `行先線名 == "最終降車"` … その路線で降りて改札外へ出る(=gate_exit)
      それ以外                 … 改札内の路線間乗換(=transfer)
      行先線名が `** 初乗り計 *:` 等 … ターミナル小計(=total_*)
    ★ 小計の水準は駅別発着表と桁が合わない(CAVEATS 参照)。比率にだけ使うこと。
    """
    head = _find_header(rows, "ターミナル", "到着線", "行先線")
    header = [norm(strip_ruby(c)) for c in rows[head]]

    def col_of(*needles, default=None):
        for i, h in enumerate(header):
            if all(n in h for n in needles):
                return i
        return default

    c_term = col_of("ターミナル")
    c_from = col_of("到着線名")
    c_fdir = col_of("到着線", "上り下り")
    c_to = col_of("行先線名")
    c_tdir = col_of("行先線", "上り")
    c_daily = col_of("終日")
    c_peak = col_of("ピーク時", "1時間")
    c_win = col_of("ピーク時間")
    if None in (c_term, c_from, c_to, c_daily):
        raise ValueError("ターミナル/到着線/行先線/終日 の列が揃っていない")
    want = set(terminals) if terminals else None
    out: list[dict] = []
    n_missing = 0
    term = from_line = from_dir = ""
    for row in rows[head + 1:]:
        def cell(i):
            return str(row[i] or "").strip() if (i is not None and i < len(row)) else ""
        t = strip_ruby(cell(c_term))
        if t:
            term = t
        f = strip_ruby(cell(c_from))
        if f:
            from_line, from_dir = f, strip_ruby(cell(c_fdir))
        elif cell(c_fdir):
            from_dir = strip_ruby(cell(c_fdir))
        to_line = strip_ruby(cell(c_to))
        # 小計行(`**　乗換え計　　*:`)は **行先線名ではなく行先線(上り下り)の列**に置かれている。
        marker = norm(to_line) or norm(cell(c_tdir))
        if not marker:
            continue
        if want is not None and term not in want:
            continue
        total_m = re.search(r"(初乗り計|最終降車計|乗換え計|合計)", marker)
        if total_m:
            kind, label = "total", total_m.group(1)
            from_l = from_d = to_l = to_d = ""
        else:
            label = ""
            from_l, from_d = from_line, from_dir
            to_l, to_d = to_line, strip_ruby(cell(c_tdir))
            if from_l == "初乗り":
                kind, from_l, from_d = "gate_entry", "", ""
            elif to_l == "最終降車":
                kind, to_l, to_d = "gate_exit", "", ""
            else:
                kind = "transfer"
        daily = to_number(cell(c_daily))
        peak = to_number(cell(c_peak))
        if daily is None:
            n_missing += 1
        out.append({"terminal": term, "kind": kind, "total_label": label,
                    "from_line": from_l, "from_direction": from_d,
                    "to_line": to_l, "to_direction": to_d,
                    "daily": daily, "peak_hourly": peak,
                    "peak_window": cell(c_win)})
    return out, n_missing


def parse_line_capacity(rows: list[list[str]], stations=None) -> tuple[list[dict], int]:
    """表9 路線別着時間帯別駅間輸送定員表 → 駅間×着時間帯の**輸送定員**(実乗客数ではない)。"""
    head = _find_header(rows, "事業者名", "路線名", "発駅名", "着駅名")
    header = [strip_ruby(c) for c in rows[head]]
    idx = {}
    for i, h in enumerate(header):
        key = norm(h)
        for name, needle in (("operator", "事業者名"), ("line", "路線名"), ("direction", "方向"),
                             ("from", "発駅名"), ("to", "着駅名")):
            if needle in key and name not in idx:
                idx[name] = i
    bins = [(i, clean_time_label(header[i]))
            for i in range(len(header))
            if i not in idx.values() and re.search(r"\d+:\d{2}|始発", str(header[i]))]
    if not bins:
        raise ValueError("着時間帯の列が見つからない")
    want = set(stations) if stations else None
    out: list[dict] = []
    n_missing = 0
    carry = {"operator": "", "line": "", "direction": ""}
    for row in rows[head + 1:]:
        def cell(name):
            i = idx.get(name)
            return str(row[i] or "").strip() if (i is not None and i < len(row)) else ""
        for key in carry:
            v = strip_ruby(cell(key))
            if v:
                carry[key] = v
        frm, to = cell("from"), cell("to")
        if not frm and not to:
            continue
        if want is not None and frm not in want and to not in want:
            continue
        for i, label in bins:
            val = to_number(row[i]) if i < len(row) else None
            if val is None:
                n_missing += 1
            out.append({"operator": carry["operator"], "line": carry["line"],
                        "direction": carry["direction"], "from_station": frm,
                        "to_station": to, "arrival_bin": label, "capacity": val})
    return out, n_missing


def parse_station_od(rows: list[list[str]], stations=None) -> tuple[list[dict], int]:
    """第13回 初乗り・最終降車駅間移動人員 → 駅間 OD(**改札外→改札外**の実移動)。

    セルは `事業者名␣駅名`(例 `東日本旅客鉄道 渋谷`)。駅名は**完全一致**で判定するので
    `小田急電鉄 高座渋谷` は `渋谷` に混ざらない。
    """
    head = _find_header(rows, "出発駅", "到着駅", "移動人員")
    want = set(stations) if stations else None

    def split(cell: str) -> tuple[str, str]:
        s = re.sub(r"[\s　]+", " ", str(cell or "")).strip()
        if not s:
            return "", ""
        op, _, st = s.rpartition(" ")
        return (op, st) if op else ("", s)

    out: list[dict] = []
    n_missing = 0
    for row in rows[head + 1:]:
        if len(row) < 3:
            continue
        f_op, f_st = split(row[0])
        t_op, t_st = split(row[1])
        if not f_st or not t_st:
            continue
        if want is not None and f_st not in want and t_st not in want:
            continue
        val = to_number(row[2])
        if val is None:
            n_missing += 1
        out.append({"from_operator": f_op, "from_station": f_st,
                    "to_operator": t_op, "to_station": t_st, "passengers": val})
    return out, n_missing


PARSERS = {
    "station_flow": parse_station_flow,
    "time_dist": parse_time_dist,
    "transfer": parse_transfer,
    "line_capacity": parse_line_capacity,
    "station_od": parse_station_od,
}


def run_parser(table: dict, rows: list[list[str]], stations=None) -> tuple[list[dict], int]:
    """表定義の `parser` に従ってパースする。時刻分布だけは駅の次元を持たない。"""
    fn = PARSERS[table["parser"]]
    if table["parser"] == "time_dist":
        return fn(rows)
    if table["parser"] == "transfer":
        return fn(rows, terminals=stations)
    return fn(rows, stations=stations)


# ================================================================ 保存先
def raw_path(root: Path, table: dict) -> Path:
    return Path(root) / SOURCE / "raw" / f"{table['key']}_{table['survey']}.xlsx"


def out_path(root: Path, table: dict) -> Path:
    return Path(root) / SOURCE / f"{table['key']}_{table['survey']}.json"


def already_fetched(root: Path, table: dict) -> Path | None:
    p = out_path(root, table)
    return p if p.exists() else None


def default_tables(*, with_od13: bool = False) -> list[dict]:
    return [t for t in TABLES if t.get("default") or (with_od13 and t["key"] == "station_od13")]


def plan(tables=None, *, with_od13: bool = False) -> list[tuple[str, str]]:
    """`--offline` 用の取得計画(1リクエストも出さない)。"""
    return [(t["title"], t["url"]) for t in (tables or default_tables(with_od13=with_od13))]


# ================================================================ 取得
def fetch_one(root: Path, table: dict, d: _date | None = None, *, stations=None,
              timeout: float = 300.0, retries: int = 2, save_raw: bool = True,
              sleep_fn=None) -> tuple[dict | None, list[dict]]:
    """1表を取得 → パース → JSON 保存。失敗時は生バイト列を残して台帳に理由を書く。"""
    d = d or common.now_jst().date()
    stations = tuple(stations or DEFAULT_STATIONS)
    res = common.http_get(table["url"], timeout=timeout, retries=retries, sleep_fn=sleep_fn)
    if not res.ok:
        return None, [ledger.make_entry(
            SOURCE, table["key"], ok=False, http_status=res.status, date_jst=d.isoformat(),
            error=res.error or "failed", extra={"survey": table["survey"]})]
    body = res.body or b""
    if save_raw:
        common.write_bytes(raw_path(root, table), body)
    try:
        rows = read_xlsx_rows(body)
        records, n_missing = run_parser(table, rows, stations)
    except (ValueError, KeyError, zipfile.BadZipFile) as exc:
        raw = common.save_raw_on_failure(root, SOURCE, f"{table['key']}.xlsx", body)
        return None, [ledger.make_entry(
            SOURCE, table["key"], ok=False, http_status=res.status, date_jst=d.isoformat(),
            error=f"parse: {type(exc).__name__}: {exc}", path=str(raw),
            extra={"survey": table["survey"]})]
    if not records:
        raw = common.save_raw_on_failure(root, SOURCE, f"{table['key']}.xlsx", body)
        return None, [ledger.make_entry(
            SOURCE, table["key"], ok=False, http_status=res.status, date_jst=d.isoformat(),
            error=f"parse: 対象行が0件(station_filter={stations})", path=str(raw),
            extra={"survey": table["survey"]})]
    meta = common.build_meta(
        SOURCE, module="transport_census.py", urls=[table["url"]], n_records=len(records),
        n_missing=n_missing, caveats=CAVEATS,
        units={"value": table["unit"]},
        notes=[f"表: {table['title']}",
               f"調査回: 第{table['survey']}回 / 調査年度: {table['survey_year']} "
               f"({'COVID 前' if table['regime'] == 'pre_covid' else 'COVID 下'})",
               "5年に1度の静的調査。日次スケジューラには結線していない(1回取れば足りる)。",
               f"一覧ページ: {INDEX_12 if table['survey'] == 12 else ESTAT_PORTAL}"],
        extra={"table": {k: v for k, v in table.items() if k != "url"},
               "survey": table["survey"], "survey_year": table["survey_year"],
               "regime": table["regime"], "role": table["role"],
               "station_filter": list(stations) if table["parser"] != "time_dist" else [],
               "n_rows_source": len(rows),
               "index_url": INDEX_12 if table["survey"] == 12 else ESTAT_PORTAL,
               "fetch_date_jst": d.isoformat()})
    doc = {"_meta": meta, "data": records}
    path = common.write_json(out_path(root, table), doc)
    return doc, [ledger.make_entry(
        SOURCE, table["key"], ok=True, http_status=200, n_records=len(records),
        n_missing=n_missing, path=str(path), date_jst=d.isoformat(),
        extra={"survey": table["survey"], "complete": True})]


def fetch_all(root: Path, d: _date | None = None, *, tables=None, stations=None,
              with_od13: bool = False, force: bool = False, timeout: float = 300.0,
              retries: int = 2, sleep: float = 1.5, save_raw: bool = True,
              sleep_fn=None) -> tuple[list[dict], list[dict]]:
    """既定4表(+ --with-od13)を取得。既取得は `force` でなければ**要求を出さずに**飛ばす。"""
    tables = list(tables or default_tables(with_od13=with_od13))
    docs, rows = [], []
    for i, table in enumerate(tables):
        if not force and already_fetched(root, table):
            common.log(f"  [census] skip {table['key']} (取得済み)")
            continue
        doc, entries = fetch_one(root, table, d, stations=stations, timeout=timeout,
                                 retries=retries, save_raw=save_raw, sleep_fn=sleep_fn)
        rows += entries
        if doc:
            docs.append(doc)
            common.log(f"  [census] {table['key']}: {len(doc['data'])} 件")
        else:
            common.log(f"  [census] {table['key']}: 失敗 ({entries[-1].get('error')})", err=True)
        if i < len(tables) - 1:
            common.polite_sleep(sleep, sleep_fn)
    return docs, rows


def load_saved(root: Path) -> dict[str, dict]:
    """保存済み JSON を `{table_key: doc}` で読む(ネットワーク不使用)。

    壊れている・まだ取っていない表は**黙って飛ばす**(呼び出し側が
    `_meta.sources_missing` で欠けを報告する)。
    """
    out: dict[str, dict] = {}
    for table in TABLES:
        path = out_path(Path(root), table)
        if not path.exists():
            continue
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if isinstance(doc, dict):
            out[table["key"]] = doc
    return out
