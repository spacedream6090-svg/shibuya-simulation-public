#!/usr/bin/env python
"""V1: reality_score v1 — カテゴリ別の現実整合スコア(**総合1点に潰さない**)。

    python scripts/reality_score.py runs/<name> [--out runs/<name>/reality]

位置づけ
--------
`docs/plans/beta-implementation-plan.md` §4 の **V1**、
`docs/plans/external-audit-triage.md` §3.2 V1 / §R4 の指標追補 / F17 の実体。
`scripts/calibrate_report.py`(第31/62/92バッチ)の**拡張**であって置き換えではない:

  - calibrate_report … 「現実バンド(lo..hi)に入るか」を 30 指標で広く浅く見る**健診**。
  - reality_score    … 「どの一次統計の、どの年次の、どの空間分母に対して、
                        どの距離尺度で、どれだけ離れているか」を**成分表示**する検証表。
                        ground truth はコード中に 1 つも書かず
                        `data/ground_truth/registry.yaml` にだけ置く。

再利用(二重実装をしない)
--------------------------
  - L1 の読みは `scripts/l1_stream.py`(row-group 逐次 + Arrow レベル kind 絞り込み)。
    **全件 RAM 展開はしない**。時間計算量は O(読んだイベント数 + step 数)。
  - KS 統計量・循環中央値・時刻換算は `scripts/calibrate_report.py` の既存実装を import。
  - Gini は `scripts/build_panel._gini`(凍結対象外)を import。**新しく書かない**。
  - Δt(1 step の分数)は `scripts/run_dt.py` が単一の源。
  - 本ファイルは metrics_spec_hash の SPEC_FILES 14 本に**含まれない**(凍結を破らない)。
    src/ は 1 バイトも触らず、読むだけ。

設計の 4 つの約束(監査への直接の回答)
--------------------------------------
 1. **総合点に潰さない**。カテゴリ内でも行ごとに出す。1 点に畳むと、どの成分が
    効いているかが消え、較正の圧力が「点を上げる」方向に働いて標本外性能を壊す。
 2. **calibration / holdout を必ず列で出す**。構築に使った統計との一致は
    予測力の証拠ではない。registry の `split` をそのまま運ぶ。
 3. **データ不足は N/A と言う**。標本が足りない・その機能が OFF・列が無い場合は
    値を作らず `status="N/A"` と理由を書く。小規模 mock ランでは大半が N/A に
    なるのが**正しい**(それが読めることを検収条件にしている)。
 4. **Data Vintage / Spatial Support をレポート本文に出す**。社会生活基本調査 2026 は
    2026 年 10 月調査で提出時点では存在せず、**2021 が正当な最新**である——という
    種類の事実を、読み手が台帳を開かずに読めるところに置く。

指標の使い分け
--------------
  JSD  … 離散分布どうし(年齢5歳階級・時刻別の在場カーブ)。0..1(底2)。
  MAPE … スカラー1点(標本1点なので実質 APE)。時刻量は円周距離を 24h で割る。
  KS   … 経験分布 vs 参照 CDF/レートカーブ(与えられた点の上での最大差)。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (os.path.join(_ROOT, "src"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Windows コンソール(cp932)でも印字できるように(ファイル出力は常に UTF-8)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import build_panel as bp                    # noqa: E402  (_gini の既存実装)
import calibrate_report as cr               # noqa: E402  (ks_stat / circ_median_h / hod)
import l1_stream as ls                      # noqa: E402  (有界メモリの L1 読み)
import run_dt                               # noqa: E402  (ランの Δt の単一の源)

REGISTRY_PATH = os.path.join(_ROOT, "data", "ground_truth", "registry.yaml")
RUNS_ROOT = os.path.join(_ROOT, "runs")
SCHEMA_VERSION = 1

MIN_PER_DAY = 1440

#: カテゴリの並びと表示名(registry の category 値 → 見出し)。
CATEGORY_ORDER: tuple[tuple[str, str], ...] = (
    ("population", "Population(人口)"),
    ("time_use", "Time Use(生活時間)"),
    ("mobility", "Mobility(移動・在場)"),
    ("media", "Media(メディア接触)"),
    ("cognition", "Cognition(認知の行き渡り)"),
)

#: L1 から読む kind(**これだけ**)。Arrow レベルで絞るので他の行は dict 化すらしない。
L1_KINDS: frozenset[str] = frozenset({
    "sleep_start", "wake_up",                  # 生活時間
    "enter_building", "exit_building",         # 在宅・帰宅→就寝
    "enter_area", "exit_area",                 # 在場区間(在場カーブ・人日の分母)
    "route_start",                             # トリップ
    "media_use",                               # メディア
    "weather",                                 # 平日/休日の別
})

#: 非就業を表す職業語(v1 186 語 + v2 追加語)。`scripts/persona_v2.nonworking_occupation`
#: が返しうる語 + v1 の同義語。ここを増やすときは persona_v2 側と突き合わせること。
NONWORKING_OCCUPATIONS: frozenset[str] = frozenset({
    "無職", "年金生活者", "主婦・主夫", "主婦", "主夫",
    "未就学児", "小学生", "中学生", "高校生", "大学生", "学生", "生徒", "院生",
})

_NA = "N/A"


# --------------------------------------------------------------------------- #
# 指標(JSD は新規・KS/Gini は既存実装を借りる)
# --------------------------------------------------------------------------- #
def _normalize(values) -> list[float] | None:
    """非負ベクトル → 総和 1 の分布。空・総和 0・負値混入は None(捏造しない)。"""
    xs = [float(v or 0.0) for v in values]
    if not xs or any(x < 0 for x in xs):
        return None
    total = sum(xs)
    if total <= 0:
        return None
    return [x / total for x in xs]


def jsd(p, q) -> float | None:
    """Jensen-Shannon divergence(底 2・0..1)。長さ違い・空は None。

    KL と違い有限で対称・平方根が距離になる。**正規化は内部で行う**(呼び出し側が
    人数と構成比を混ぜても同じ値が出るように)。
    """
    p = _normalize(p)
    q = _normalize(q)
    if p is None or q is None or len(p) != len(q):
        return None
    tot = 0.0
    for a, b in zip(p, q):
        m = 0.5 * (a + b)
        if m <= 0:
            continue
        if a > 0:
            tot += 0.5 * a * math.log2(a / m)
        if b > 0:
            tot += 0.5 * b * math.log2(b / m)
    return max(0.0, min(1.0, tot))


def mape(sim, ref, *, circular_hours: bool = False) -> float | None:
    """スカラー 1 点の相対誤差(標本 1 点なので実質 APE)。

    `circular_hours=True` は時刻量(23:50 と 00:10 が 20 分差)。円周距離を
    **24 時間で割った相対量**を返す(0..0.5)。分母に参照時刻を使うと 0 時付近で
    発散するため。
    """
    if sim is None or ref is None:
        return None
    if circular_hours:
        d = abs(float(sim) - float(ref)) % 24.0
        return min(d, 24.0 - d) / 24.0
    r = float(ref)
    if r == 0.0:
        return None
    return abs(float(sim) - r) / abs(r)


def ks_vs_points(sim_at, ref_points) -> float | None:
    """参照レートカーブ(点列)との最大差 = KS 統計量の点上版。

    `sim_at` は「時刻(実数時)→ sim 側の率」を返す callable。参照点で評価できない
    (sim 側が None)点があれば None(部分一致で数字を作らない)。
    """
    if not ref_points:
        return None
    d = 0.0
    for pt in ref_points:
        s = sim_at(float(pt["hour"]))
        if s is None:
            return None
        d = max(d, abs(float(s) - float(pt["value"])))
    return d


def quantile(xs, q: float) -> float | None:
    """線形補間の分位(numpy を使わない・空は None)。"""
    ys = sorted(float(x) for x in xs)
    if not ys:
        return None
    if len(ys) == 1:
        return ys[0]
    pos = (len(ys) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(ys) - 1)
    return ys[lo] + (ys[hi] - ys[lo]) * (pos - lo)


# --------------------------------------------------------------------------- #
# registry の読み込み
# --------------------------------------------------------------------------- #
class RegistryError(RuntimeError):
    """台帳が掟(source/year/spatial_support/denominator/split の 4+1)を満たさない。"""


REQUIRED_ANCHOR_KEYS = ("id", "category", "label", "source", "year",
                        "spatial_support", "denominator", "split")


def load_registry(path: str | None = None) -> dict:
    """`data/ground_truth/registry.yaml` を読む。**来歴の欠けたアンカーは弾く**。

    掟(registry.yaml 冒頭)を機械で守る唯一の場所。ここを緩めると、来歴のない
    数値が静かに増える。
    """
    p = Path(path or REGISTRY_PATH)
    if not p.is_file():
        raise RegistryError(f"ground truth registry が無い: {p}")
    import yaml                                            # noqa: PLC0415
    doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise RegistryError(f"registry の形式が不正: {p}")
    anchors = doc.get("anchors") or []
    seen: set = set()
    for a in anchors:
        if not isinstance(a, dict):
            raise RegistryError("anchors の要素が dict ではない")
        missing = [k for k in REQUIRED_ANCHOR_KEYS if not a.get(k)]
        if missing:
            raise RegistryError(f"anchor {a.get('id')!r} に必須欄が無い: {missing}")
        if a["split"] not in ("calibration", "holdout", "diagnostic"):
            raise RegistryError(f"anchor {a['id']!r} の split が不正: {a['split']}")
        if a["id"] in seen:
            raise RegistryError(f"anchor id が重複: {a['id']}")
        seen.add(a["id"])
    doc["_path"] = str(p)
    doc["_by_id"] = {a["id"]: a for a in anchors}
    return doc


def _anchor(reg: dict, anchor_id: str) -> dict:
    a = reg["_by_id"].get(anchor_id)
    if a is None:
        raise RegistryError(f"anchor が registry に無い: {anchor_id}")
    return a


# --------------------------------------------------------------------------- #
# 行(row)の組み立て — 全行が同じスキーマを持つ
# --------------------------------------------------------------------------- #
ROW_KEYS = ("id", "category", "label", "metric", "sim", "ref", "value", "status",
            "n", "unit", "source", "year", "spatial_support", "denominator",
            "split", "split_basis", "url", "note", "estimate", "derived")


def _status_of(metric: str | None, value, thresholds: dict) -> str:
    if metric is None or value is None:
        return "info"
    band = (thresholds or {}).get(metric)
    if not band:
        return "info"
    if value <= float(band["ok"]):
        return "ok"
    if value <= float(band["warn"]):
        return "warn"
    return "fail"


def make_row(anchor: dict, *, sim=None, ref=None, value=None, metric=None,
             n: int = 0, status: str | None = None, note: str = "",
             thresholds: dict | None = None) -> dict:
    """anchor(現実側)+ 実測(sim 側)→ レポート 1 行。**列は常に同じ**。"""
    m = metric if metric is not None else anchor.get("metric")
    if status is None:
        status = _NA if (value is None and m is not None) else \
            _status_of(m, value, thresholds or {})
    notes = [t for t in (anchor.get("note"), note) if t]
    return {
        "id": anchor["id"],
        "category": anchor["category"],
        "label": anchor["label"],
        "metric": m,
        "sim": sim,
        "ref": ref if ref is not None else anchor.get("value"),
        "value": value,
        "status": status,
        "n": int(n),
        "unit": anchor.get("unit", ""),
        "source": anchor["source"],
        "year": str(anchor["year"]),
        "spatial_support": anchor["spatial_support"],
        "denominator": str(anchor["denominator"]).strip(),
        "split": anchor["split"],
        "split_basis": str(anchor.get("split_basis", "")).strip(),
        "url": anchor.get("url", ""),
        "note": " / ".join(notes).strip(),
        "estimate": bool(anchor.get("estimate", False)),
        "derived": bool(anchor.get("derived", False)),
    }


def na_row(anchor: dict, reason: str, *, n: int = 0, sim=None) -> dict:
    """データ不足の行。**理由を必ず書く**(黙って空欄にしない)。"""
    return make_row(anchor, sim=sim, value=None, n=n, status=_NA,
                    note=f"データ不足: {reason}")


# --------------------------------------------------------------------------- #
# sim 側の観測(L1 を 1 パス・有界メモリ)
# --------------------------------------------------------------------------- #
def _read_start_tod_min(run_dir) -> int:
    """`run.start_tod`(既定 07:00)を分 of day で返す。読めなければ 420。"""
    p = Path(run_dir) / "config.yaml"
    if not p.is_file():
        return 420
    try:
        import yaml                                        # noqa: PLC0415
        doc = yaml.safe_load(p.read_text(encoding="utf-8"))
        v = (doc or {}).get("run", {}).get("start_tod")
    except Exception:                                      # noqa: BLE001
        return 420
    if v is None:
        return 420
    s = str(v).strip()
    if ":" in s:
        try:
            hh, mm = s.split(":", 1)
            return (int(hh) * 60 + int(mm)) % MIN_PER_DAY
        except ValueError:
            return 420
    try:
        return int(float(s)) % MIN_PER_DAY
    except ValueError:
        return 420


def load_agents(run_dir) -> list[dict]:
    """agents.json(**day0 の在場者スナップショット**)。無ければ空。

    ★プール回転ランでは途中入場者が載らない。全数の名簿が要る解析は
      `roster.parquet`(observer.roster_daily)を併用すること——本スクリプトは
      名簿依存の指標(人口・属性別 coverage)に限ってこれを使い、その旨を
      レポートの分母欄に出す。
    """
    p = Path(run_dir) / "agents.json"
    if not p.is_file():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [a for a in data if isinstance(a, dict)]


def _read_l2(run_dir) -> dict:
    p = Path(run_dir) / "l2_metrics.parquet"
    if not p.is_file():
        return {}
    cols: dict[str, list] = defaultdict(list)
    for d in ls.iter_table_columns(p):
        for k, v in d.items():
            cols[k].extend(v)
    return dict(cols)


def scan(run_dir, agents: list[dict], spd: int, mps: int, n_steps: int) -> dict:
    """L1 を **1 パス**走査して sim 側の生の観測を作る。O(読んだ行数 + step 数)。

    返す辞書(作れなかったものは空 = 後段が N/A を出せる形):
      bed_hours / wake_hours / slept_min  … 生活時間
      home_awake_min / gap_min            … 自宅在館分・帰宅→就寝
      presence                            … step ごとの在場数(長さ n_steps)
      person_days                         … 人日の分母(在場区間が跨いだ暦日の延べ)
      route_by_ad                         … (agent, day) -> route_start 件数
      media_min_by_day / media_agents     … メディア
      holiday_by_day                      … day -> bool(weather.holiday 由来)
    """
    home_of: dict[int, str] = {}
    for a in agents:
        try:
            home_of[int(a["id"])] = str(a.get("home_building") or "")
        except (KeyError, TypeError, ValueError):
            continue
    roster_ids = set(home_of)

    bed_hours: list[float] = []
    wake_hours: list[float] = []
    slept_min: list[float] = []
    home_awake_min: Counter = Counter()      # aid -> 在宅分(睡眠込み)
    gap_min: list[float] = []                # 帰宅 -> 就寝(分)
    home_in_at: dict[int, int] = {}          # aid -> 在宅開始の sim_min
    route_by_ad: Counter = Counter()         # (aid, day) -> route_start
    media_min_by_day: Counter = Counter()    # day -> 分
    media_agents: set = set()
    holiday_by_day: dict[int, bool] = {}
    open_at: dict[int, int] = {}             # aid -> 在場開始 step
    area_ids: set = set()                    # enter/exit_area に現れた aid
    intervals: list[tuple[int, int]] = []    # (start_step, end_step) 半開区間

    run_end = int(n_steps)

    for e in ls.iter_events(run_dir, ["step", "sim_min", "agent_id", "kind",
                                      "payload"], kinds=L1_KINDS):
        k = e["kind"]
        aid = e["agent_id"]
        step = int(e["step"])
        sm = int(e["sim_min"]) if e["sim_min"] is not None else step * mps
        # ★日の切り方は **活動日**(step // spd = 開始時刻起点)。build_panel /
        #   calibrate_report と同じ流儀で、既定 start_tod=07:00 では夜間睡眠が
        #   同一日に収まり `n_days = n_steps // spd` と分母が一致する。
        #   暦日(sim_min//1440)で切ると 2 日ランが 3 暦日に跨り人日が 1.5 倍に膨らむ。
        day = step // spd
        p = e["payload"] or {}

        if k == "weather":
            hol = p.get("holiday")
            if isinstance(hol, bool):
                holiday_by_day.setdefault(day, hol)
            continue

        if aid is None or aid < 0:
            continue

        if k == "sleep_start":
            bed_hours.append(cr.hod(sm))
            t0 = home_in_at.get(aid)
            if t0 is not None and sm >= t0:
                # ★ `>=` であることが重要: 「帰宅した瞬間に就寝」(第115 の実測)は
                #   gap=0 という**観測**であって欠測ではない。`>` にすると
                #   この既知の欠陥が「標本 0 件 = N/A」に化けて表から消える。
                gap_min.append(float(sm - t0))
        elif k == "wake_up":
            wake_hours.append(cr.hod(sm))
            ss = p.get("slept_steps")
            if ss:
                slept_min.append(float(ss) * float(mps))
        elif k == "enter_building":
            if home_of.get(aid) and str(p.get("building") or "") == home_of[aid]:
                home_in_at.setdefault(aid, sm)
        elif k == "exit_building":
            t0 = home_in_at.pop(aid, None)
            if t0 is not None and sm > t0:
                home_awake_min[aid] += float(sm - t0)
        elif k == "route_start":
            route_by_ad[(aid, day)] += 1
        elif k == "media_use":
            steps = p.get("steps")
            if steps:
                media_min_by_day[day] += float(steps) * float(mps)
                media_agents.add(aid)
        elif k == "enter_area":
            area_ids.add(aid)
            open_at[aid] = step
        elif k == "exit_area":
            area_ids.add(aid)
            s0 = open_at.pop(aid, 0)
            if step > s0:
                intervals.append((s0, step))

    # 走行終了時にまだ街に居る個体
    for aid, s0 in open_at.items():
        if run_end > s0:
            intervals.append((s0, run_end))
    # enter/exit_area に一度も現れなかった名簿個体 = 全期間在場
    n_always = len(roster_ids - area_ids)

    # 在宅区間が閉じないまま終わった分をラン末で締める
    end_min = run_end * mps
    for aid, t0 in home_in_at.items():
        if end_min > t0:
            home_awake_min[aid] += float(end_min - t0)

    # ---- 在場の差分配列(O(区間数 + step 数)) ----
    delta = [0] * (run_end + 2)
    for s0, s1 in intervals:
        delta[max(0, s0)] += 1
        delta[min(run_end, s1)] -= 1
    if n_always:
        delta[0] += n_always
        delta[run_end] -= n_always
    presence = [0] * run_end
    acc = 0
    for i in range(run_end):
        acc += delta[i]
        presence[i] = acc

    # ---- 人日(分母)---- 在場区間が跨いだ**活動日**の延べ
    start_tod = _read_start_tod_min(run_dir)

    def _day_of_step(s: int) -> int:
        return int(s) // int(spd)

    person_days = 0
    for s0, s1 in intervals:
        person_days += _day_of_step(max(0, s1 - 1)) - _day_of_step(s0) + 1
    if n_always and run_end > 0:
        person_days += n_always * (_day_of_step(run_end - 1) - _day_of_step(0) + 1)

    return {
        "bed_hours": bed_hours, "wake_hours": wake_hours, "slept_min": slept_min,
        "home_awake_min": home_awake_min, "gap_min": gap_min,
        "route_by_ad": route_by_ad,
        "media_min_by_day": media_min_by_day, "media_agents": media_agents,
        "holiday_by_day": holiday_by_day,
        "presence": presence, "person_days": int(person_days),
        "n_always": n_always, "start_tod": start_tod,
        "n_area_agents": len(area_ids),
    }


def tod_bins(presence: list[int], start_tod: int, mps: int) -> tuple[list[float], int]:
    """step 系列 → **時刻帯ごとの平均**(1440/Δt ビン)。空ビンは 0 のまま。"""
    nb = MIN_PER_DAY // int(mps)
    acc = [0.0] * nb
    cnt = [0] * nb
    for s, v in enumerate(presence):
        b = ((start_tod + s * mps) % MIN_PER_DAY) // mps
        acc[b] += float(v)
        cnt[b] += 1
    return [(acc[i] / cnt[i]) if cnt[i] else 0.0 for i in range(nb)], nb


def resample_reference_curve(path: str, key_col: str, val_col: str,
                             mps: int) -> list[float] | None:
    """参照カーブ CSV(時刻ごとの値)→ ランの Δt のビンへ畳む。

    CSV は `time`(HH:MM)列を持つ前提。持たない場合は `key_col` を step とみなし
    10 分刻みと解釈する(本 repo の jinryu 派生表の形)。
    """
    import csv                                             # noqa: PLC0415
    p = Path(path)
    if not p.is_file():
        return None
    nb = MIN_PER_DAY // int(mps)
    acc = [0.0] * nb
    cnt = [0] * nb
    with p.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if val_col not in row:
                return None
            t = row.get("time")
            if t and ":" in t:
                hh, mm = t.split(":", 1)
                minute = (int(hh) * 60 + int(mm)) % MIN_PER_DAY
            else:
                try:
                    minute = (int(row[key_col]) * 10) % MIN_PER_DAY
                except (KeyError, TypeError, ValueError):
                    return None
            try:
                v = float(row[val_col])
            except (TypeError, ValueError):
                continue
            b = minute // int(mps)
            if 0 <= b < nb:
                acc[b] += v
                cnt[b] += 1
    if not any(cnt):
        return None
    # 空ビン(Δt が参照より細かい場合)は直近の値で保つ = 階段状の保持
    out = [0.0] * nb
    last = None
    for i in range(nb):
        if cnt[i]:
            last = acc[i] / cnt[i]
        out[i] = last if last is not None else 0.0
    if out[0] == 0.0 and last is not None:                 # 先頭の空きを後ろから埋める
        for i in range(nb):
            if out[i]:
                break
            out[i] = last
    return out


# --------------------------------------------------------------------------- #
# カテゴリ別スコア
# --------------------------------------------------------------------------- #
_AGE_BIN_EDGES = [(0, 4), (5, 9), (10, 14), (15, 19), (20, 24), (25, 29),
                  (30, 34), (35, 39), (40, 44), (45, 49), (50, 54), (55, 59),
                  (60, 64), (65, 69), (70, 74), (75, 79), (80, 84), (85, 89),
                  (90, 94), (95, 99), (100, 200)]


def _age_bin(age) -> int | None:
    try:
        a = int(age)
    except (TypeError, ValueError):
        return None
    if a < 0:
        return None
    for i, (lo, hi) in enumerate(_AGE_BIN_EDGES):
        if lo <= a <= hi:
            return i
    return None


def _is_working_occupation(occ) -> bool:
    return bool(occ) and str(occ) not in NONWORKING_OCCUPATIONS


def score_population(agents: list[dict], reg: dict, thr: dict) -> list[dict]:
    """人口: 年齢5歳階級(JSD)・年少/高齢/女性割合・労働力率・非正規率・役員割合。"""
    rows: list[dict] = []
    residents = [a for a in agents if not a.get("visitor")]
    ages = [a.get("age") for a in residents]
    bins = [0] * len(_AGE_BIN_EDGES)
    n_aged = 0
    for age in ages:
        i = _age_bin(age)
        if i is not None:
            bins[i] += 1
            n_aged += 1
    min_dist = int((reg.get("min_samples") or {}).get("distribution", 30))
    min_scalar = int((reg.get("min_samples") or {}).get("scalar", 20))

    a = _anchor(reg, "pop_age_5y_resident")
    if n_aged < min_dist:
        rows.append(na_row(a, f"年齢の取れた住民 {n_aged} 人 < {min_dist}", n=n_aged))
    else:
        rows.append(make_row(a, sim=None, value=jsd(bins, a["values"]),
                             n=n_aged, thresholds=thr,
                             note=f"sim 側 5 歳階級の実数(上位3帯)="
                                  f"{sorted(zip(bins, [f'{lo}-{hi}' for lo, hi in _AGE_BIN_EDGES]), reverse=True)[:3]}"))

    def _share(pred) -> float | None:
        if n_aged < min_scalar:
            return None
        return sum(1 for age in ages if _age_bin(age) is not None and pred(int(age))) / n_aged

    for anchor_id, pred in (("pop_child_share", lambda x: x <= 14),
                            ("pop_elderly_share", lambda x: x >= 65)):
        a = _anchor(reg, anchor_id)
        v = _share(pred)
        if v is None:
            rows.append(na_row(a, f"年齢の取れた住民 {n_aged} 人 < {min_scalar}", n=n_aged))
        else:
            rows.append(make_row(a, sim=round(v, 5), value=mape(v, a["value"]),
                                 n=n_aged, thresholds=thr))

    a = _anchor(reg, "pop_female_share")
    genders = [str(x.get("gender") or "") for x in residents]
    known = [g for g in genders if g in ("男", "女")]
    if len(known) < min_scalar:
        rows.append(na_row(a, f"性別の取れた住民 {len(known)} 人 < {min_scalar}",
                           n=len(known)))
    else:
        v = sum(1 for g in known if g == "女") / len(known)
        rows.append(make_row(a, sim=round(v, 5), value=mape(v, a["value"]),
                             n=len(known), thresholds=thr))

    a = _anchor(reg, "pop_labour_force_rate_15_64")
    wa = [x for x in residents if (_age_bin(x.get("age")) is not None
                                   and 15 <= int(x["age"]) <= 64)]
    if len(wa) < min_scalar:
        rows.append(na_row(a, f"15-64 歳の住民 {len(wa)} 人 < {min_scalar}", n=len(wa)))
    else:
        v = sum(1 for x in wa if _is_working_occupation(x.get("occupation"))) / len(wa)
        rows.append(make_row(a, sim=round(v, 5), value=mape(v, a["value"]),
                             n=len(wa), thresholds=thr,
                             note="sim 側は「働く職業を持つ割合」= 失業者を含まないぶん低め"))

    a = _anchor(reg, "pop_nonregular_rate")
    emp = [x for x in residents if _is_working_occupation(x.get("occupation"))]
    if len(emp) < min_scalar:
        rows.append(na_row(a, f"就業者 {len(emp)} 人 < {min_scalar}", n=len(emp)))
    else:
        v = sum(1 for x in emp if x.get("part_time")) / len(emp)
        rows.append(make_row(a, sim=round(v, 5), value=mape(v, a["value"]),
                             n=len(emp), thresholds=thr))

    a = _anchor(reg, "pop_exec_share")
    ranked = [x for x in residents
              if x.get("employment_status") or x.get("rank")]
    if not ranked:
        rows.append(na_row(a, "agents.json に employment_status / rank 欄が無い"
                              "(v1 ペルソナプールでは正常)"))
    else:
        v = sum(1 for x in ranked
                if str(x.get("employment_status") or x.get("rank")) == "役員") / len(ranked)
        rows.append(make_row(a, sim=round(v, 5), value=mape(v, a["value"]),
                             n=len(ranked), thresholds=thr))
    return rows


def score_time_use(obs: dict, l2: dict, reg: dict, thr: dict, mps: int) -> list[dict]:
    """生活時間: 睡眠(2 調査)・就寝/起床時刻・睡眠行為者率カーブ・在宅覚醒・帰宅→就寝。"""
    rows: list[dict] = []
    min_scalar = int((reg.get("min_samples") or {}).get("scalar", 20))
    slept = obs["slept_min"]

    for anchor_id in ("tu_sleep_min_nhk2020", "tu_sleep_min_ssb2021"):
        a = _anchor(reg, anchor_id)
        if len(slept) < min_scalar:
            rows.append(na_row(a, f"睡眠標本 {len(slept)} 夜 < {min_scalar}",
                               n=len(slept)))
            continue
        v = st.mean(slept)
        rows.append(make_row(a, sim=round(v, 1), value=mape(v, a["value"]),
                             n=len(slept), thresholds=thr,
                             note="sim 側は wake_up.slept_steps の平均(1 夜 = 1 標本)"))

    a = _anchor(reg, "tu_bedtime_h_tokyo")
    bh = obs["bed_hours"]
    if len(bh) < min_scalar:
        rows.append(na_row(a, f"就寝イベント {len(bh)} 件 < {min_scalar}", n=len(bh)))
    else:
        v = cr.circ_median_h(bh)
        rows.append(make_row(a, sim=round(v, 3), value=mape(v, a["value"],
                                                           circular_hours=True),
                             n=len(bh), thresholds=thr,
                             note="円周中央値(日跨ぎ補正あり)。乖離は円周距離 / 24h"))

    a = _anchor(reg, "tu_wake_h_tokyo")
    wh = obs["wake_hours"]
    if len(wh) < min_scalar:
        rows.append(na_row(a, f"起床イベント {len(wh)} 件 < {min_scalar}", n=len(wh)))
    else:
        v = st.median(wh)
        rows.append(make_row(a, sim=round(v, 3), value=mape(v, a["value"],
                                                           circular_hours=True),
                             n=len(wh), thresholds=thr,
                             note="乖離は円周距離 / 24h"))

    # ---- 睡眠行為者率カーブ(KS)----
    a = _anchor(reg, "tu_asleep_rate_curve_nhk2020_m20s")
    sleeping = l2.get("n_sleeping")
    steps = l2.get("step")
    presence = obs["presence"]
    if not sleeping or not steps or not presence:
        rows.append(na_row(a, "L2 の n_sleeping 列 または 在場系列が無い"))
    else:
        num = defaultdict(float)
        den = defaultdict(float)
        for s, v in zip(steps, sleeping):
            s = int(s)
            if s >= len(presence):
                continue
            b = ((obs["start_tod"] + s * mps) % MIN_PER_DAY) // mps
            num[b] += float(v or 0)
            den[b] += float(presence[s])

        def _rate_at(hour: float):
            b = int((hour * 60.0) % MIN_PER_DAY) // mps
            if den.get(b, 0.0) <= 0:
                return None
            return num.get(b, 0.0) / den[b]

        d = ks_vs_points(_rate_at, a.get("points"))
        if d is None:
            rows.append(na_row(a, "参照時刻に対応する step が観測されていない",
                               n=len(steps)))
        else:
            rows.append(make_row(a, value=d, n=len(steps), thresholds=thr,
                                 note="sim 側 = 時刻帯ごとの (n_sleeping / 在場数)"))

    # ---- 在宅覚醒 ----
    a = _anchor(reg, "tu_home_awake_min_weekday")
    hm = obs["home_awake_min"]
    pd_ = max(1, obs["person_days"])
    if not hm:
        rows.append(na_row(a, "自宅の enter_building / exit_building が観測されていない"))
    else:
        at_home = sum(hm.values()) / pd_
        asleep = (sum(obs["slept_min"]) / pd_) if obs["slept_min"] else 0.0
        v = max(0.0, at_home - asleep)
        rows.append(make_row(a, sim=round(v, 1), value=mape(v, a["value"]),
                             n=len(hm), thresholds=thr,
                             note="sim 側 = (自宅在館分 - 睡眠分) / 人日。"
                                  "睡眠は全て自宅で起きたと仮定した近似"))

    # ---- 帰宅 -> 就寝 ----
    a = _anchor(reg, "tu_home_to_sleep_gap_min")
    gaps = obs["gap_min"]
    if len(gaps) < min_scalar:
        rows.append(na_row(a, f"帰宅→就寝の対 {len(gaps)} 件 < {min_scalar}",
                           n=len(gaps)))
    else:
        v = st.median(gaps)
        rows.append(make_row(a, sim=round(v, 1), value=mape(v, a["value"]),
                             n=len(gaps), thresholds=thr,
                             note="sim 側 = 自宅 enter_building から次の sleep_start までの中央値"))

    # ---- 仕事・食事(L2 由来 / 未対応)----
    a = _anchor(reg, "tu_work_min_ssb2021")
    nw = l2.get("n_working")
    if not nw or not presence:
        rows.append(na_row(a, "L2 の n_working 列 または 在場系列が無い"))
    else:
        tot = sum(float(v or 0) for v in nw) * mps
        v = tot / pd_
        rows.append(make_row(a, sim=round(v, 1), value=mape(v, a["value"]),
                             n=len(nw), thresholds=thr,
                             note="sim 側 = Σ(n_working) x Δt / 人日(総平均に対応)"))

    a = _anchor(reg, "tu_food_min_ssb2021")
    rows.append(na_row(a, "食事の滞在時間を表す L1/L2 の系列が v1 には無い"))
    return rows


def score_mobility(obs: dict, reg: dict, thr: dict, mps: int) -> list[dict]:
    """移動: トリップ原単位・外出率・時刻別在場カーブ(JSD)・変動幅・ピーク時刻。"""
    rows: list[dict] = []
    min_scalar = int((reg.get("min_samples") or {}).get("scalar", 20))
    min_curve = int((reg.get("min_samples") or {}).get("curve", 30))
    pd_ = obs["person_days"]
    route = obs["route_by_ad"]

    a = _anchor(reg, "mob_trips_per_person_day")
    if pd_ < min_scalar or not route:
        rows.append(na_row(a, f"人日 {pd_} < {min_scalar} または route_start が 0 件",
                           n=pd_))
    else:
        v = sum(route.values()) / pd_
        rows.append(make_row(a, sim=round(v, 3), value=mape(v, a["value"]),
                             n=pd_, thresholds=thr,
                             note="定義差あり: シムは寄り道・中断のたびに再探索 = 上界"))

    a = _anchor(reg, "mob_gaishutsu_rate")
    if pd_ < min_scalar:
        rows.append(na_row(a, f"人日 {pd_} < {min_scalar}", n=pd_))
    else:
        v = min(1.0, len(route) / pd_)
        rows.append(make_row(a, sim=round(v, 4), value=mape(v, a["value"]),
                             n=pd_, thresholds=thr,
                             note="sim 側 = その日 1 回でも route_start を出した (agent,day) の割合"))

    a = _anchor(reg, "mob_presence_curve_weekday")
    presence = obs["presence"]
    curve, nb = tod_bins(presence, obs["start_tod"], mps)
    ref_path = os.path.join(_ROOT, a["source_file"])
    ref = resample_reference_curve(ref_path, a.get("key_column", "step"),
                                   a["value_column"], mps)
    nonzero_bins = sum(1 for v in curve if v > 0)
    if ref is None:
        rows.append(na_row(a, f"参照カーブを読めない: {a['source_file']}"))
    elif nonzero_bins < min_curve:
        rows.append(na_row(a, f"在場が観測された時刻帯 {nonzero_bins} < {min_curve}",
                           n=nonzero_bins))
    elif len(set(round(v, 6) for v in curve)) <= 1:
        rows.append(na_row(a, "在場が終日一定(入退場イベントが無い)= 形の比較が成立しない",
                           n=nonzero_bins, sim=round(curve[0], 1)))
    else:
        amp = (max(curve) / min(v for v in curve if v > 0)) if any(curve) else 0.0
        rows.append(make_row(a, value=jsd(curve, ref), n=nonzero_bins,
                             thresholds=thr,
                             note=f"時刻帯 {nb} ビンの構成比どうし。絶対人数は比較しない。"
                                  f"★sim の日内変動幅 {amp:.2f} 倍(参照 4.39 倍)"
                                  f"= 平坦なカーブは JSD が小さく出るので次行と必ず併読する"))

    peak_i = max(range(len(curve)), key=lambda i: curve[i]) if curve else None
    lo = min((v for v in curve if v > 0), default=0.0)
    hi = max(curve) if curve else 0.0

    a = _anchor(reg, "mob_presence_peak_trough_ratio")
    if hi <= 0 or lo <= 0 or hi == lo:
        rows.append(na_row(a, "在場カーブに日内変動が無い(入退場イベントが無い)"))
    else:
        v = hi / lo
        rows.append(make_row(a, sim=round(v, 3), value=mape(v, a["value"]),
                             n=nonzero_bins, thresholds=thr))

    a = _anchor(reg, "mob_presence_peak_hour")
    if peak_i is None or hi <= 0 or hi == lo:
        rows.append(na_row(a, "在場カーブに日内変動が無い(ピーク時刻が定義できない)"))
    else:
        v = (peak_i * mps) / 60.0
        rows.append(make_row(a, sim=round(v, 2),
                             value=mape(v, a["value"], circular_hours=True),
                             n=nonzero_bins, thresholds=thr,
                             note="乖離は円周距離 / 24h"))
    return rows


def score_media(obs: dict, reg: dict, thr: dict) -> list[dict]:
    """メディア: 平日/休日の利用分/日。v1 の sim 側は media_use(既定 OFF)のみ。"""
    rows: list[dict] = []
    by_day = obs["media_min_by_day"]
    hol = obs["holiday_by_day"]
    pd_ = max(1, obs["person_days"])
    n_days = max(1, len(set(by_day) | set(hol)))
    # 人日は日ごとに割り振れないので「1 日あたりの平均人日」で按分する近似。
    pd_per_day = pd_ / float(n_days)

    def _minutes_per_person_day(days) -> float:
        return sum(by_day[d] for d in days) / max(1e-9, pd_per_day * len(days))

    weekday_ids = ("media_internet_min_weekday", "media_tv_realtime_min_weekday",
                   "media_home_media_min_weekday")
    holiday_ids = ("media_internet_min_holiday", "media_tv_realtime_min_holiday")
    off_reason = "L1 に media_use が 1 件も無い(media.enabled 既定 OFF では正常)"

    for anchor_id in weekday_ids:
        a = _anchor(reg, anchor_id)
        if not by_day:
            rows.append(na_row(a, off_reason))
            continue
        wk_days = [d for d in by_day if hol.get(d) is False]
        if wk_days:
            v, note = _minutes_per_person_day(wk_days), "平日のみ"
        else:
            v = sum(by_day.values()) / pd_
            note = "★平日/休日を分離できない(weather.holiday が無い)= 全日平均で代用"
        rows.append(make_row(a, sim=round(v, 1), value=mape(v, a["value"]),
                             n=len(obs["media_agents"]), thresholds=thr, note=note))
    for anchor_id in holiday_ids:
        a = _anchor(reg, anchor_id)
        hd_days = [d for d in by_day if hol.get(d) is True]
        if not by_day:
            rows.append(na_row(a, off_reason))
        elif not hd_days:
            rows.append(na_row(a, "休日を含まないラン(weather.holiday=True の日が無い)"))
        else:
            v = _minutes_per_person_day(hd_days)
            rows.append(make_row(a, sim=round(v, 1), value=mape(v, a["value"]),
                                 n=len(obs["media_agents"]), thresholds=thr))
    return rows


def score_cognition(run_dir, agents: list[dict], obs: dict, reg: dict,
                    thr: dict) -> list[dict]:
    """認知: 呼数/人日・zero-call 率・P50/P90/Gini・レーン別 / 属性別 coverage。

    ★現実側のアンカーは無い(diagnostic)。**平均だけでは 25 万人に届いたか分からない**
      への回答なので、必ず 0 回の個体を分母に含める。
    """
    rows: list[dict] = []
    path = Path(run_dir) / "l1b_llm.parquet"
    parts = ls.l1_paths(run_dir, "l1b_llm")
    calls_by_agent: Counter = Counter()
    lane_agents: dict[str, set] = {ln: set() for ln in ("life", "reply", "general")}
    total_calls = 0

    from society.cognition.lod import PURPOSE_LANE     # noqa: PLC0415 (読むだけ)

    if path.is_file() or parts:
        src = [path] if path.is_file() else parts
        for one in src:
            for d in ls.iter_table_columns(one, ["agent_id", "purpose"]):
                aids = d.get("agent_id") or []
                purposes = d.get("purpose") or [None] * len(aids)
                for aid, pur in zip(aids, purposes):
                    if aid is None or int(aid) < 0:
                        continue
                    aid = int(aid)
                    calls_by_agent[aid] += 1
                    total_calls += 1
                    lane_agents[PURPOSE_LANE.get(str(pur or ""), "general")].add(aid)

    present_ids = ls.distinct_agent_ids(run_dir)
    if not present_ids:
        present_ids = {int(a["id"]) for a in agents if "id" in a}
    n_present = len(present_ids)
    pd_ = max(1, obs["person_days"])
    n_days = pd_ / max(1, n_present)

    a = _anchor(reg, "cog_calls_per_person_day")
    if not total_calls:
        rows.append(na_row(a, "l1b_llm が無い / 0 行(LLM 呼が記録されていない)",
                           n=n_present))
    else:
        v = total_calls / pd_
        band = a.get("target_band") or []
        if band and len(band) == 2:
            lo, hi = float(band[0]), float(band[1])
            dev = 0.0 if lo <= v <= hi else (
                (lo - v) / lo if v < lo else (v - hi) / hi)
            rows.append(make_row(a, sim=round(v, 4), ref=f"{lo}-{hi}", value=dev,
                                 n=n_present, thresholds=thr,
                                 note=f"目標帯からの相対乖離(帯内 = 0)。総呼数 {total_calls}"))
        else:
            rows.append(make_row(a, sim=round(v, 4), value=None, n=n_present,
                                 status="info"))

    per_agent = [calls_by_agent.get(i, 0) for i in present_ids]

    a = _anchor(reg, "cog_zero_call_share")
    if not n_present:
        rows.append(na_row(a, "L1 に agent が 1 人も現れない"))
    else:
        z = sum(1 for c in per_agent if c == 0) / n_present
        rows.append(make_row(a, sim=round(z, 4), n=n_present, status="info",
                             note=f"在場 agent {n_present} 人中 {sum(1 for c in per_agent if c == 0)} 人が 0 回"))

    a = _anchor(reg, "cog_calls_distribution")
    if not n_present or not total_calls:
        rows.append(na_row(a, "呼が 0 件 / agent が 0 人", n=n_present))
    else:
        per_day = [c / max(1e-9, n_days) for c in per_agent]
        rows.append(make_row(
            a, sim={"p50": round(quantile(per_day, 0.5) or 0.0, 4),
                    "p90": round(quantile(per_day, 0.9) or 0.0, 4),
                    "gini": bp._gini(per_agent),
                    "mean": round(sum(per_day) / len(per_day), 4)},
            n=n_present, status="info",
            note="0 回の個体を分母に含めた分布(落とすと集中度が過小評価される)"))

    a = _anchor(reg, "cog_lane_coverage")
    if not n_present or not total_calls:
        rows.append(na_row(a, "呼が 0 件 / agent が 0 人", n=n_present))
    else:
        cov = {ln: round(len(ids & present_ids) / n_present, 4)
               for ln, ids in lane_agents.items()}
        rows.append(make_row(a, sim=cov, n=n_present, status="info",
                             note="lod.PURPOSE_LANE の写像に従う(未登録 purpose は general)"))

    a = _anchor(reg, "cog_attr_coverage")
    meta = {int(x["id"]): x for x in agents if "id" in x}
    if not n_present or not meta:
        rows.append(na_row(a, "agents.json が無い / agent が 0 人", n=n_present))
    else:
        by_band: dict[str, list[int]] = defaultdict(list)
        by_layer: dict[str, list[int]] = defaultdict(list)
        for i in present_ids:
            m = meta.get(i)
            if m is None:
                by_band["(名簿外=途中入場)"].append(calls_by_agent.get(i, 0))
                by_layer["(名簿外=途中入場)"].append(calls_by_agent.get(i, 0))
                continue
            bi = _age_bin(m.get("age"))
            lab = ("%d-%d" % _AGE_BIN_EDGES[bi]) if bi is not None else "(年齢不明)"
            by_band[lab].append(calls_by_agent.get(i, 0))
            by_layer["来街者" if m.get("visitor") else "居住者"].append(
                calls_by_agent.get(i, 0))

        def _cov(d):
            return {k: {"n": len(v),
                        "coverage": round(sum(1 for c in v if c > 0) / len(v), 4)}
                    for k, v in sorted(d.items()) if v}

        rows.append(make_row(a, sim={"age_band": _cov(by_band),
                                     "layer": _cov(by_layer)},
                             n=n_present, status="info",
                             note="coverage = その属性で 1 回以上呼ばれた割合"))
    return rows


# --------------------------------------------------------------------------- #
# 本体
# --------------------------------------------------------------------------- #
def build_report(run_dir, registry_path: str | None = None) -> dict:
    """run-dir → レポート辞書(JSON にそのまま落ちる形)。"""
    reg = load_registry(registry_path)
    thr = reg.get("thresholds") or {}
    spd = run_dt.steps_per_day(run_dir)
    mps = run_dt.min_per_step(run_dir)

    agents = load_agents(run_dir)
    summary = {}
    sp = Path(run_dir) / "summary.json"
    if sp.is_file():
        try:
            summary = json.loads(sp.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            summary = {}
    n_steps = int(summary.get("n_steps") or 0)
    if not n_steps:
        n_steps = ls.max_step(run_dir) + 1

    obs = scan(run_dir, agents, spd, mps, n_steps)
    l2 = _read_l2(run_dir)

    cat_rows: dict[str, list[dict]] = {
        "population": score_population(agents, reg, thr),
        "time_use": score_time_use(obs, l2, reg, thr, mps),
        "mobility": score_mobility(obs, reg, thr, mps),
        "media": score_media(obs, reg, thr),
        "cognition": score_cognition(run_dir, agents, obs, reg, thr),
    }

    categories = []
    for key, label in CATEGORY_ORDER:
        rows = cat_rows.get(key, [])
        tally = Counter(r["status"] for r in rows)
        categories.append({
            "name": key, "label": label, "rows": rows,
            "n_rows": len(rows),
            "n_ok": tally.get("ok", 0), "n_warn": tally.get("warn", 0),
            "n_fail": tally.get("fail", 0), "n_na": tally.get(_NA, 0),
            "n_info": tally.get("info", 0),
            "n_holdout": sum(1 for r in rows if r["split"] == "holdout"),
            "n_calibration": sum(1 for r in rows if r["split"] == "calibration"),
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "run_dir": os.path.abspath(str(run_dir)),
            "run": os.path.basename(os.path.normpath(str(run_dir))),
            "n_agents_roster": len(agents),
            "n_steps": n_steps,
            "dt_min": mps,
            "steps_per_day": spd,
            "n_days": round(n_steps / float(spd), 3) if spd else None,
            "person_days": obs["person_days"],
            "start_tod_min": obs["start_tod"],
            "n_agents_with_area_events": obs["n_area_agents"],
        },
        "registry": {"path": reg["_path"], "updated": reg.get("updated"),
                     "n_anchors": len(reg["_by_id"])},
        "data_vintage": reg.get("data_vintage") or [],
        "spatial_support": reg.get("spatial_support") or {},
        "thresholds": thr,
        "categories": categories,
    }


# --------------------------------------------------------------------------- #
# Markdown 出力
# --------------------------------------------------------------------------- #
_STATUS_MARK = {"ok": "OK", "warn": "WARN", "fail": "FAIL", _NA: _NA, "info": "--"}


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, float):
        if v != 0 and abs(v) < 0.001:
            return f"{v:.2e}"
        return f"{v:,.4g}"
    return str(v)


def render_markdown(rep: dict) -> str:
    m = rep["meta"]
    L: list[str] = [
        f"# reality_score v1: {m['run']}",
        "",
        f"- run: `{m['run_dir']}`",
        f"- 規模: 名簿 {m['n_agents_roster']} 人 / {m['n_steps']} step "
        f"({m['n_days']} 日・Δt={m['dt_min']} 分) / **人日 {m['person_days']}**",
        f"- ground truth: `{rep['registry']['path']}` "
        f"(更新 {rep['registry']['updated']} / アンカー {rep['registry']['n_anchors']} 件)",
        "",
        "> **総合 1 点は出さない。** カテゴリ内でも行ごとに成分表示する。1 点に畳むと"
        "どの成分が効いているかが消え、較正の圧力が「点を上げる」方向へ働いて標本外性能を壊す。",
        "> **calibration 行の一致は予測力の証拠ではない**(構築にその統計自身を使っている)。"
        "モデルの主張は **holdout 行**だけで支える。",
        "> **N/A は失点ではない**。「その機能が OFF」「標本が足りない」を数字で埋めないための表示。",
        "",
        "## 判定帯(事前登録・運用上の取り決めであって文献値ではない)",
        "",
        "| 指標 | OK | WARN | 用途 |",
        "|---|---|---|---|",
    ]
    use = {"jsd": "離散分布どうし(年齢階級・時刻カーブ)",
           "mape": "スカラー1点(時刻量は円周距離/24h)",
           "ks": "経験分布 vs 参照レートカーブ(点上の最大差)"}
    for k, band in (rep.get("thresholds") or {}).items():
        L.append(f"| {k.upper()} | <= {band.get('ok')} | <= {band.get('warn')} | "
                 f"{use.get(k, '')} |")

    for cat in rep["categories"]:
        L += ["", f"## {cat['label']}",
              "",
              f"- 行 {cat['n_rows']}(OK {cat['n_ok']} / WARN {cat['n_warn']} / "
              f"FAIL {cat['n_fail']} / {_NA} {cat['n_na']} / 参考 {cat['n_info']})"
              f" — holdout {cat['n_holdout']} / calibration {cat['n_calibration']}",
              "",
              "| 指標 | sim | 現実 | 尺度 | 値 | 判定 | n | 出典 | 年次 | 空間分母 | 分割 |",
              "|---|---|---|---|---|---|---:|---|---|---|---|"]
        for r in cat["rows"]:
            L.append(
                f"| {r['label']} | {_fmt(r['sim'])} | {_fmt(r['ref'])}"
                f"{(' ' + r['unit']) if r['unit'] and r['ref'] is not None else ''} | "
                f"{(r['metric'] or '-').upper()} | {_fmt(r['value'])} | "
                f"{_STATUS_MARK.get(r['status'], r['status'])} | {r['n']} | "
                f"{r['source']} | {r['year']} | {r['spatial_support']} | {r['split']} |")
        notes = [r for r in cat["rows"] if r["note"] or r["split_basis"]
                 or r["estimate"] or r["derived"]]
        if notes:
            L += ["", "<details><summary>注記(分母・定義差・来歴)</summary>", ""]
            for r in notes:
                tags = []
                if r["estimate"]:
                    tags.append("ESTIMATE")
                if r["derived"]:
                    tags.append("derived")
                head = f"- **{r['label']}**" + (f" [{'/'.join(tags)}]" if tags else "")
                L.append(head)
                L.append(f"  - 分母: {r['denominator']}")
                if r["note"]:
                    L.append(f"  - 注: {r['note']}")
                if r["split_basis"]:
                    L.append(f"  - {r['split']} の根拠: {r['split_basis']}")
                if r["url"]:
                    L.append(f"  - 出典 URL: {r['url']}")
            L += ["", "</details>"]

    L += ["", "## Data Vintage Ledger(いま入手できる最新版はどれか)", "",
          "| 調査 | 最新公表 | 次回 | 提出時点で入手可 | 備考 |",
          "|---|---|---|---|---|"]
    for v in rep.get("data_vintage") or []:
        L.append(f"| {v.get('name')} | {v.get('latest_published')} | "
                 f"{v.get('next_wave')} | "
                 f"{'YES' if v.get('available_at_submission') else '**NO**'} | "
                 f"{str(v.get('note', '')).strip()} |")
    L += ["",
          "> ★**社会生活基本調査 2026 は 2026 年 10 月調査**であり、本選提出時点(2026-08)"
          "では公表されていない。したがって **2021 年版が正当な最新**であって、"
          "2021 を使うことは古い値の使用ではない(F17)。",
          "> 一方、メディア(情報通信メディア調査)は令和7年度版が 2026-06 に公表済みなので"
          "**年次更新済み**であり、家計調査も 2025 年平均が入手可能である。"]

    L += ["", "## Spatial Support Crosswalk(その数字の分母はどの空間か)", "",
          "| キー | 空間 | 定義 |", "|---|---|---|"]
    for k, v in (rep.get("spatial_support") or {}).items():
        L.append(f"| {k} | {v.get('label')} | {str(v.get('definition', '')).strip()} |")
    L += ["",
          "> bbox / 渋谷区 / 東京都市圏 / 全国 は**分母が違う**。同じ「1 人あたり」でも"
          "空間が違えば直接比較にならないので、全行に空間分母の列を出している。",
          "> 名簿(agents.json)は day0 の在場者スナップショットである。プール回転ランで"
          "途中入場した個体は載らないので、人口系の行は `roster.parquet`"
          "(observer.roster_daily)を併せて見ること。"]
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# CLI(scripts/audit_uncertainty.py の house style に合わせる)
# --------------------------------------------------------------------------- #
def _pick_run(arg_run: str | None) -> str:
    if arg_run:
        d = arg_run if os.path.isabs(arg_run) else os.path.join(RUNS_ROOT, arg_run)
        if not os.path.isdir(d):
            raise SystemExit(f"[reality_score] run dir が無い: {d}")
        return d
    if not os.path.isdir(RUNS_ROOT):
        raise SystemExit(f"[reality_score] runs が無い: {RUNS_ROOT}")
    cands = []
    for name in os.listdir(RUNS_ROOT):
        pq_path = os.path.join(RUNS_ROOT, name, "l1_events.parquet")
        if os.path.isfile(pq_path):
            cands.append((os.path.getmtime(pq_path), os.path.join(RUNS_ROOT, name)))
    if not cands:
        raise SystemExit("[reality_score] l1_events.parquet を持つランが無い")
    cands.sort(reverse=True)
    return cands[0][1]


def analyze(run_dir: str, out_dir: str, registry_path: str | None = None) -> dict:
    rep = build_report(run_dir, registry_path)
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "reality_score.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=2, sort_keys=True)
    md_path = os.path.join(out_dir, "reality_score.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(rep) + "\n")
    rep["paths"] = {"json": json_path, "md": md_path}
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="カテゴリ別の現実整合スコア(読み出し専用・総合点に潰さない)")
    ap.add_argument("run_dir", nargs="?", default=None,
                    help="ラン名 or パス(既定: l1_events.parquet を持つ最新ラン)")
    ap.add_argument("--out", default=None, help="出力先(既定: <run>/reality)")
    ap.add_argument("--registry", default=None,
                    help=f"ground truth 台帳(既定: {REGISTRY_PATH})")
    a = ap.parse_args(argv)
    run_dir = _pick_run(a.run_dir)
    out_dir = a.out or os.path.join(run_dir, "reality")
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(run_dir, out_dir)
    rep = analyze(run_dir, out_dir, a.registry)
    print(render_markdown(rep))
    print(f"[reality_score] -> {rep['paths']['json']}")
    print(f"[reality_score] -> {rep['paths']['md']}")
    return 0


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())
