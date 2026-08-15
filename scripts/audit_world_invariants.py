#!/usr/bin/env python
"""V2: 世界の整合性の**事後・全数検査**(読み出し専用)。

    python scripts/audit_world_invariants.py runs/<name> [--out runs/<name>/invariants]

位置づけ
--------
`docs/plans/beta-implementation-plan.md` §4 の **V2**、
`docs/plans/external-audit-triage.md` §3.2 V2(「250k リハ後に実行する前提の
O(イベント数) 実装」)。**既存テストの検査式を流用**し、同じ規則を 2 度書かない。

流用元(検査式の出どころ。ここを変えるときは向こうも見ること)
--------------------------------------------------------------
  位置・階   … `tests/test_floor_clamp.py::_hi / _out_of_range`
               (規則の純関数は `src/society/world/floors.clamp_floor` / `W_FLOOR_MAX`)
  node と xy … `tests/test_lane_hei.py::test_hei4_...`(`(e.x, e.y) == city.node_xy(node)`)
  在館の閉じ … `scripts/build_occupancy.py::occupancy_rows` の carry-forward 規約
  年齢×職業 … `tests/test_persona_pool_v2.py::test_age_occupation_consistency`
               `test_no_clip_pileup`(単一年齢への人工堆積)
  死の永続性 … `src/society/health.py::_die / _exit_world`(`NEVER_RETURN`)+
               `tests/test_health_severity.py::test_death_is_a_permanent_exit_...`
  幽霊書込み … `tests/test_present_predicate.py`(第112 甲。金を動かす経路が急所)
  転出の規約 … `tests/test_population.py::test_emigrated_never_returns`
               ★**転出者は当日中は sim.agents に残る**(死と同型)。事後検査で
                 「転出したのにイベントがある」を当日ぶんまで違反にすると誤検出する。

「0 であるべき」と「許容帯」を分ける
------------------------------------
本スクリプトの出力は 2 群に分かれる。混ぜると読めない:
  severity=`zero` … 1 件でもあれば実装の誤り(位置の欠落・死者の金流・宙に浮いた参照)。
  severity=`band` … 0 でないのが正常な項目(`world.floor_clamp` 既定 OFF での階超過・
                    プール回転で名簿に載らない途中入場者・在館のままラン終了)。
                    **帯の外に出たときだけ**注意を出す。

計算量
------
L1 を 1 パス。O(イベント数)。payload の JSON parse は「その kind が要るとき」と
「生文字列に参照キーの綴りが現れたとき」だけに絞る(4.06e9 行で全件 parse しない)。
メモリは O(agent 数 + 建物数)。**全件 RAM 展開はしない**。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (os.path.join(_ROOT, "src"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import l1_stream as ls                      # noqa: E402
import run_dt                               # noqa: E402
from society.world.floors import W_FLOOR_MAX, clamp_floor   # noqa: E402

RUNS_ROOT = os.path.join(_ROOT, "runs")
SCHEMA_VERSION = 1
MAX_EXAMPLES = 3
XY_TOL = 1.0e-3                            # 座標一致の許容(float32 往復ぶん)

#: 位置・階・建物を payload に持つ kind(payload を必ず parse する)。
POSITION_KINDS: frozenset[str] = frozenset({
    "enter_building", "exit_building", "floor_move", "space_move",
    "go_to_bed", "arrive", "exit_area", "enter_area",
})
#: 生存・退場の kind。
LIFECYCLE_KINDS: frozenset[str] = frozenset({"death", "life_event"})
#: 金が動く kind(死者・退場者が触れていたら 保存則の破れになりうる急所)。
MONEY_KINDS: frozenset[str] = frozenset({
    "spend", "wage", "rent", "transfer", "inheritance", "asset_transfer",
    "venture_sale", "loan", "repay", "tax", "stipend", "payout",
})
#: payload の中で **他の agent の id** を指す綴り。生文字列にこれが出たときだけ parse。
REF_KEYS: tuple[str, ...] = ("hearers", "with", "to", "from", "pair", "partner",
                             "owner", "actor", "target", "peer", "other")
#: 上のうち「必ず agent id」であるキー(建物 id や文字列が入る "to"/"from" は除く)。
AGENT_REF_KEYS: tuple[str, ...] = ("hearers", "with", "pair", "partner", "peer")

#: 非就業を表す職業語(reality_score.py と同じ表。persona_v2.nonworking_occupation 準拠)。
NONWORKING_OCCUPATIONS: frozenset[str] = frozenset({
    "無職", "年金生活者", "主婦・主夫", "主婦", "主夫",
    "未就学児", "小学生", "中学生", "高校生", "大学生", "学生", "生徒", "院生",
})
#: 学生を表す職業語(30 歳以上の学生は実在するが多くはない)。
STUDENT_OCCUPATIONS: frozenset[str] = frozenset({
    "大学生", "学生", "生徒", "院生", "高校生", "中学生", "小学生", "未就学児",
})
#: 年齢の下限が制度で決まっている職業(流用元 = test_persona_pool_v2 の検査式)。
MIN_AGE_BY_OCCUPATION: tuple[tuple[str, int], ...] = (
    ("会社員", 18), ("公務員", 18), ("経営者", 23), ("議員", 25), ("医師", 24),
    ("弁護士", 24), ("教師", 22), ("大学教員", 24), ("保育士", 20),
)
#: 人が物理的に立てる密度の上限(人/m2)。群集事故の研究で 4-6 人/m2 が「crush」帯。
PHYSICAL_DENSITY_LIMIT = 4.0
#: 注意帯の下限(人/m2)。満員の店舗・ホームでも 0.5 を超えると相当な混雑。
DENSITY_WARN = 0.5


# --------------------------------------------------------------------------- #
# 検査の宣言(id / 分類 / severity / 帯)。**表の外に検査を足さない**。
# --------------------------------------------------------------------------- #
CHECKS: tuple[dict, ...] = (
    # ---- (1) 位置整合 ----
    dict(id="pos_missing_xy", cat="position", severity="zero",
         label="在場イベントに座標が無い",
         desc="agent_id>=0 の行で x または y が null。位置を持つ主体の行為が"
              "座標なしで記録されている = 空間解析が黙って壊れる。"),
    dict(id="pos_node_xy_mismatch", cat="position", severity="zero", needs_map=True,
         label="arrive の node と行の (x,y) が食い違う",
         desc="payload.node の地図座標と L1 行の (x,y) の差が 1e-3 を超える。"
              "流用元: tests/test_lane_hei.py(回転を跨いだ位置の不整合の検出式)。"),
    dict(id="pos_unknown_building", cat="position", severity="zero", needs_map=True,
         label="地図に無い建物 id を参照している",
         desc="payload.building が world.map の buildings に存在しない。"),
    dict(id="pos_exit_without_enter", cat="position", severity="zero",
         label="入館していないのに退館している",
         desc="直前に enter_building が無い exit_building。在館の carry-forward"
              "(scripts/build_occupancy.py の規約)が閉じない。"),
    dict(id="pos_floor_out_of_range", cat="position", severity="band",
         band=(0.0, 0.05), unit="件/在館イベント", needs_map=True,
         label="建物の階数を超える floor",
         desc="floor が [1, min(levels, 99)] の外。**world.floor_clamp 既定 OFF では"
              "実データ由来の逸脱が現に出る**(POI に階数超え 20 件・地下 31 件)ので"
              "0 を要求しない。ON のランでは 0 になるべき。"),
    dict(id="pos_indoor_left_open", cat="position", severity="band",
         band=(0.0, 1.0), unit="件/在館 agent",
         label="ラン終了時に在館のまま",
         desc="enter_building の後に exit_building / exit_area が来ないままランが終わる。"
              "打ち切りとして正常(1 人 1 件までは自然)。"),
    # ---- (2) 年齢 x 役職 ----
    dict(id="age_out_of_range", cat="age_role", severity="zero",
         label="年齢が定義域の外",
         desc="age < 0 または > 120。"),
    dict(id="age_child_worker", cat="age_role", severity="zero",
         label="15 歳未満が就業している",
         desc="age < 15 かつ職業が就業語。労基法 56 条(中学卒業まで就業禁止)。"),
    dict(id="age_below_occupation_min", cat="age_role", severity="zero",
         label="職業の年齢下限を下回る",
         desc="17 歳以下の会社員・公務員、22 歳以下の経営者 など。"
              "流用元: tests/test_persona_pool_v2.py::test_age_occupation_consistency。"),
    dict(id="age_student_over_30", cat="age_role", severity="band",
         band=(0.0, 0.01), unit="件/名簿", min_n=200,
         label="30 歳以上の学生",
         desc="社会人学生は実在するので 0 は求めない(帯 1% は流用元と同じ)。"
              "★1% の帯は名簿 200 人未満では判定不能(1 人が 0.5% を超える)。"),
    dict(id="age_clip_pileup", cat="age_role", severity="band",
         band=(0.0, 0.015), unit="超過分/名簿", min_n=500,
         label="単一年齢への人工的な堆積",
         desc="ある 1 歳の人数が近傍 5 歳の平均の 1.5 倍を超える分。v1 プールの"
              "「15 歳に 4.08%」(clip アーティファクト)を捕まえる検出器。"
              "★名簿 500 人未満では 1 歳あたりの期待人数が 1 人前後になり、"
              "近傍比が構造的に暴れるので判定しない。"),
    # ---- (3) 死亡・退場 ----
    dict(id="dead_agent_acts", cat="lifecycle", severity="zero",
         label="死亡後に行動している",
         desc="death イベントより後の step に、その agent_id を主体とする行が出る。"
              "死は境界 despawn と同型の永続退場(src/society/health.py)。"),
    dict(id="dead_agent_money", cat="lifecycle", severity="zero",
         label="死亡後に金が動いている",
         desc="上のうち金流 kind に限った内数。**保存則の破れは L1 上で帳尻が"
              "合ってしまい検出されない**(第112 甲の教訓)ので独立に数える。"),
    dict(id="emigrant_acts_later_day", cat="lifecycle", severity="zero",
         label="転出した翌日以降に行動している",
         desc="life_event{kind: emigrate} の**翌活動日以降**の行。"
              "★当日中は sim.agents に残る規約(tests/test_population.py)なので"
              "当日ぶんは違反にしない。"),
    dict(id="death_duplicated", cat="lifecycle", severity="zero",
         label="同じ agent が 2 回死んでいる",
         desc="death が 1 個体につき 2 件以上。"),
    # ---- (4) 容量 ----
    dict(id="cap_density_physical", cat="capacity", severity="zero", needs_map=True,
         label="物理的に不可能な在館密度",
         desc=f"同時在館者 / (footprint 面積 x levels) が {PHYSICAL_DENSITY_LIMIT} 人/m2 超。"
              "群集事故研究の crush 帯を超えるので、どんな混雑でも起こりえない。"),
    dict(id="cap_density_crowded", cat="capacity", severity="band",
         band=(0.0, 0.05), unit="建物/在館のあった建物", needs_map=True,
         label="極端に混雑した建物",
         desc=f"同時在館密度が {DENSITY_WARN} 人/m2 超の建物の割合。満員の店舗でも"
              "この密度は稀なので、多いと収容モデルの見直し候補。"),
    dict(id="cap_dwelling_load", cat="capacity", severity="band",
         band=(0.0, 12.0), unit="人/住戸(最大)",
         label="1 住戸あたりの居住者数(最大)",
         desc="(home_building, home_floor) ごとの居住者数の最大値。"
              "同一階に複数住戸がありうるので厳密な定員ではない = 帯で見る。"),
    # ---- (5) 孤児・宙に浮いた参照 ----
    dict(id="orphan_resident_no_home", cat="integrity", severity="zero",
         label="住居を持たない居住者",
         desc="visitor=false なのに home_building が空。"),
    dict(id="orphan_unknown_home_or_work", cat="integrity", severity="zero", needs_map=True,
         label="名簿の建物 id が地図に無い",
         desc="agents.json の home_building / work_building が world.map に存在しない。"),
    dict(id="dangling_agent_ref", cat="integrity", severity="zero",
         label="存在しない agent を参照している",
         desc="payload の hearers / with / pair / partner / peer が、名簿にも L1 にも"
              "現れない id を指す。関係台帳が宙に浮いている印。"),
    dict(id="l1_agent_not_in_roster", cat="integrity", severity="band",
         band=(0.0, 1.0), unit="件/L1 agent",
         label="名簿に無い agent が L1 に現れる",
         desc="agents.json は **day0 の在場者スナップショット**なので、プール回転ランで"
              "途中入場した個体は載らない = 0 でないのが正常。"
              "roster.parquet(observer.roster_daily)を ON にすると 0 に近づく。"),
)
_CHECK_BY_ID = {c["id"]: c for c in CHECKS}


# --------------------------------------------------------------------------- #
# 集計器
# --------------------------------------------------------------------------- #
class Tally:
    """1 検査ぶんの件数と例示(最大 3 件)。**例示は最初に出会った 3 件**(決定論)。"""

    def __init__(self, check_id: str) -> None:
        self.id = check_id
        self.n = 0
        self.examples: list[dict] = []
        self.denominator = 0
        self.extra: dict = {}

    def hit(self, example: dict | None = None, k: int = 1) -> None:
        self.n += int(k)
        if example is not None and len(self.examples) < MAX_EXAMPLES:
            self.examples.append(example)


class Tallies(dict):
    def __missing__(self, key):
        self[key] = Tally(key)
        return self[key]


# --------------------------------------------------------------------------- #
# 地図(read-only)
# --------------------------------------------------------------------------- #
def _polygon_area(pts) -> float:
    """靴紐公式。座標は局所メートル。3 点未満は 0。"""
    try:
        n = len(pts)
    except TypeError:
        return 0.0
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = pts[i][0], pts[i][1]
        x2, y2 = pts[(i + 1) % n][0], pts[(i + 1) % n][1]
        s += float(x1) * float(y2) - float(x2) * float(y1)
    return abs(s) * 0.5


def load_map(run_dir) -> dict:
    """ランの `config.yaml` の `world.map` を読む。**読むだけ**。

    返り値: {"buildings": {id: {"levels", "area"}}, "nodes": {id: (x, y)}, "path": str}
    地図が読めなければ空の索引(その分の検査は skipped になる)。
    """
    out = {"buildings": {}, "nodes": {}, "path": None}
    cfg = Path(run_dir) / "config.yaml"
    rel = None
    if cfg.is_file():
        try:
            import yaml                                    # noqa: PLC0415
            doc = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
            rel = (doc.get("world") or {}).get("map")
        except Exception:                                  # noqa: BLE001
            rel = None
    if not rel:
        return out
    p = Path(rel)
    if not p.is_absolute():
        p = Path(_ROOT) / rel
    if not p.is_file():
        return out
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return out
    for b in doc.get("buildings") or []:
        try:
            out["buildings"][str(b["id"])] = {
                "levels": int(b.get("levels") or 1),
                "area": _polygon_area(b.get("footprint") or []),
            }
        except (KeyError, TypeError, ValueError):
            continue
    for n in doc.get("nodes") or []:
        try:
            out["nodes"][str(n["id"])] = (float(n["x"]), float(n["y"]))
        except (KeyError, TypeError, ValueError):
            continue
    out["path"] = str(p)
    return out


def load_agents(run_dir) -> list[dict]:
    p = Path(run_dir) / "agents.json"
    if not p.is_file():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [a for a in data if isinstance(a, dict)]


# --------------------------------------------------------------------------- #
# (A) 名簿だけで判る検査(年齢 x 役職・孤児・住戸)
# --------------------------------------------------------------------------- #
def _is_working_occupation(occ) -> bool:
    return bool(occ) and str(occ) not in NONWORKING_OCCUPATIONS


def audit_roster(agents: list[dict], world: dict, T: Tallies) -> None:
    n = len(agents)
    for cid in ("age_out_of_range", "age_child_worker", "age_below_occupation_min",
                "age_student_over_30", "age_clip_pileup",
                "orphan_resident_no_home", "orphan_unknown_home_or_work",
                "cap_dwelling_load"):
        T[cid].denominator = n
    if not agents:
        return

    min_age = dict(MIN_AGE_BY_OCCUPATION)
    ages: Counter = Counter()
    dwelling: Counter = Counter()
    buildings = world["buildings"]

    for a in agents:
        aid = a.get("id")
        occ = str(a.get("occupation") or "")
        try:
            age = int(a.get("age"))
        except (TypeError, ValueError):
            T["age_out_of_range"].hit({"id": aid, "age": a.get("age"),
                                       "reason": "int にできない"})
            continue
        if age < 0 or age > 120:
            T["age_out_of_range"].hit({"id": aid, "age": age})
            continue
        ages[age] += 1
        if age < 15 and _is_working_occupation(occ):
            T["age_child_worker"].hit({"id": aid, "age": age, "occupation": occ})
        lo = min_age.get(occ)
        if lo is not None and age < lo:
            T["age_below_occupation_min"].hit(
                {"id": aid, "age": age, "occupation": occ, "min_age": lo})
        if age >= 30 and occ in STUDENT_OCCUPATIONS and occ != "未就学児":
            T["age_student_over_30"].hit({"id": aid, "age": age, "occupation": occ})

        home = str(a.get("home_building") or "")
        work = str(a.get("work_building") or "")
        if not a.get("visitor") and not home:
            T["orphan_resident_no_home"].hit({"id": aid, "occupation": occ})
        if buildings:
            for label, bid in (("home_building", home), ("work_building", work)):
                if bid and bid not in buildings:
                    T["orphan_unknown_home_or_work"].hit(
                        {"id": aid, "field": label, "building": bid})
        if home:
            dwelling[(home, a.get("home_floor"))] += 1

    # ---- 単一年齢への堆積(近傍 5 歳の平均の 1.5 倍を超えた分)----
    for age, c in sorted(ages.items()):
        nb = [ages.get(age + d, 0) for d in (-2, -1, 1, 2)]
        base = sum(nb) / 4.0
        if base <= 0:
            continue
        over = c - 1.5 * base
        if over > 0:
            T["age_clip_pileup"].hit({"age": age, "count": c,
                                      "neighbour_mean": round(base, 2)},
                                     k=int(round(over)))

    if dwelling:
        top = dwelling.most_common(MAX_EXAMPLES)
        t = T["cap_dwelling_load"]
        t.n = top[0][1]
        t.denominator = len(dwelling)
        t.examples = [{"building": b, "floor": f, "residents": c}
                      for (b, f), c in top]
        t.extra = {"n_dwellings": len(dwelling)}


# --------------------------------------------------------------------------- #
# (B) L1 の 1 パス(位置・生死・容量・参照)
# --------------------------------------------------------------------------- #
def _need_payload(kind: str, raw: str | None) -> bool:
    if kind in POSITION_KINDS or kind in LIFECYCLE_KINDS or kind in MONEY_KINDS:
        return True
    if not raw:
        return False
    return any(('"%s"' % k) in raw for k in REF_KEYS)


def audit_l1(run_dir, agents: list[dict], world: dict, spd: int, mps: int,
             T: Tallies) -> dict:
    """L1 を 1 パスして位置・生死・容量・参照を検査する。O(イベント数)。"""
    buildings = world["buildings"]
    nodes = world["nodes"]
    roster_ids = {int(a["id"]) for a in agents if "id" in a}

    inside: dict[int, str] = {}           # aid -> いま在館している building
    occupancy: Counter = Counter()        # building -> いまの在館者数
    peak: Counter = Counter()             # building -> 同時在館の最大
    dead_at: dict[int, int] = {}          # aid -> death の step
    emigrated_day: dict[int, int] = {}    # aid -> 転出した活動日
    death_count: Counter = Counter()
    l1_ids: set = set()
    referenced: dict[int, tuple] = {}     # 未知 id -> 最初に見た例
    n_rows = 0
    n_indoor_rows = 0
    kinds_after_death: Counter = Counter()

    cols = ["step", "sim_min", "agent_id", "kind", "x", "y", "payload"]
    for d in ls.iter_columns(run_dir, cols):
        steps, kinds = d["step"], d["kind"]
        aids, xs, ys, pays = d["agent_id"], d["x"], d["y"], d["payload"]
        for i in range(len(steps)):
            n_rows += 1
            kind = kinds[i]
            aid = aids[i]
            step = int(steps[i])
            if aid is None or int(aid) < 0:
                continue
            aid = int(aid)
            l1_ids.add(aid)

            # ---- 座標の欠落 ----
            if xs[i] is None or ys[i] is None:
                T["pos_missing_xy"].hit({"step": step, "agent_id": aid, "kind": kind})

            # ---- 死亡・転出の後の行 ----
            ds = dead_at.get(aid)
            if ds is not None and step > ds:
                kinds_after_death[kind] += 1
                T["dead_agent_acts"].hit({"step": step, "agent_id": aid,
                                          "kind": kind, "death_step": ds})
                if kind in MONEY_KINDS:
                    T["dead_agent_money"].hit({"step": step, "agent_id": aid,
                                               "kind": kind, "death_step": ds})
            ed = emigrated_day.get(aid)
            if ed is not None and (step // spd) > ed:
                T["emigrant_acts_later_day"].hit(
                    {"step": step, "agent_id": aid, "kind": kind,
                     "emigrate_day": ed, "day": step // spd})

            raw = pays[i] if pays is not None else None
            if not _need_payload(kind, raw):
                continue
            try:
                p = json.loads(raw) if raw else {}
            except (ValueError, TypeError):
                p = {}
            if not isinstance(p, dict):
                continue

            # ---- 生死 ----
            if kind == "death":
                death_count[aid] += 1
                if aid in dead_at:
                    T["death_duplicated"].hit({"agent_id": aid, "step": step,
                                               "first_death_step": dead_at[aid]})
                dead_at.setdefault(aid, step)
            elif kind == "life_event" and str(p.get("kind") or "") == "emigrate":
                emigrated_day.setdefault(aid, step // spd)

            # ---- 位置・建物・階 ----
            bid = p.get("building")
            if isinstance(bid, str) and bid:
                if buildings and bid not in buildings:
                    T["pos_unknown_building"].hit({"step": step, "agent_id": aid,
                                                   "kind": kind, "building": bid})
                elif buildings and p.get("floor") is not None:
                    n_indoor_rows += 1
                    lv = buildings[bid]["levels"]
                    f = p.get("floor")
                    try:
                        fi = int(f)
                    except (TypeError, ValueError):
                        fi = None
                    hi = max(1, min(int(lv or 1), W_FLOOR_MAX))
                    if fi is None or not (1 <= fi <= hi):
                        T["pos_floor_out_of_range"].hit(
                            {"step": step, "agent_id": aid, "kind": kind,
                             "building": bid, "floor": f, "levels": lv,
                             "clamped_to": clamp_floor(f, lv)})

            if kind == "enter_building" and isinstance(bid, str) and bid:
                prev = inside.get(aid)
                if prev is not None and occupancy[prev] > 0:
                    occupancy[prev] -= 1      # 退館なしの移動は在館だけ付け替える
                inside[aid] = bid
                occupancy[bid] += 1
                if occupancy[bid] > peak[bid]:
                    peak[bid] = occupancy[bid]
            elif kind in ("exit_building", "exit_area"):
                prev = inside.pop(aid, None)
                if prev is None:
                    if kind == "exit_building":
                        T["pos_exit_without_enter"].hit(
                            {"step": step, "agent_id": aid, "building": bid})
                elif occupancy[prev] > 0:
                    occupancy[prev] -= 1

            # ---- arrive の node と (x, y) ----
            if kind == "arrive" and nodes:
                nid = p.get("node")
                if isinstance(nid, str) and nid in nodes:
                    nx, ny = nodes[nid]
                    ex, ey = xs[i], ys[i]
                    if ex is not None and ey is not None and (
                            abs(float(ex) - nx) > XY_TOL
                            or abs(float(ey) - ny) > XY_TOL):
                        T["pos_node_xy_mismatch"].hit(
                            {"step": step, "agent_id": aid, "node": nid,
                             "event_xy": [float(ex), float(ey)],
                             "node_xy": [nx, ny]})

            # ---- 他 agent への参照 ----
            for key in AGENT_REF_KEYS:
                v = p.get(key)
                if v is None:
                    continue
                cand = v if isinstance(v, (list, tuple)) else [v]
                for one in cand:
                    try:
                        rid = int(one)
                    except (TypeError, ValueError):
                        continue
                    if rid < 0:
                        continue
                    referenced.setdefault(rid, (step, aid, kind, key))

    # ---- 参照の解決(L1 と名簿の和集合に無い id が宙に浮いた参照)----
    known = l1_ids | roster_ids
    for rid, (step, aid, kind, key) in sorted(referenced.items()):
        if rid not in known:
            T["dangling_agent_ref"].hit({"step": step, "agent_id": aid,
                                         "kind": kind, "field": key,
                                         "missing_id": rid})

    # ---- 在館のまま終了 ----
    t = T["pos_indoor_left_open"]
    t.n = len(inside)
    t.denominator = max(1, len(l1_ids))
    t.examples = [{"agent_id": a, "building": b}
                  for a, b in sorted(inside.items())[:MAX_EXAMPLES]]

    # ---- 名簿外の L1 agent ----
    t = T["l1_agent_not_in_roster"]
    extra_ids = sorted(l1_ids - roster_ids)
    t.n = len(extra_ids)
    t.denominator = max(1, len(l1_ids))
    t.examples = [{"agent_id": a} for a in extra_ids[:MAX_EXAMPLES]]

    # ---- 在館密度 ----
    dens: list[tuple[float, str, int, float]] = []
    for bid, pk in peak.items():
        meta = buildings.get(bid)
        if not meta or meta["area"] <= 0:
            continue
        cap_area = meta["area"] * max(1, meta["levels"])
        dens.append((pk / cap_area, bid, pk, cap_area))
    dens.sort(reverse=True)
    hard = [x for x in dens if x[0] > PHYSICAL_DENSITY_LIMIT]
    warn = [x for x in dens if x[0] > DENSITY_WARN]
    t = T["cap_density_physical"]
    t.n = len(hard)
    t.denominator = max(1, len(dens))
    t.examples = [{"building": b, "peak_occupants": pk,
                   "usable_area_m2": round(area, 1),
                   "density_per_m2": round(dv, 3)} for dv, b, pk, area in hard[:MAX_EXAMPLES]]
    t = T["cap_density_crowded"]
    t.n = len(warn)
    t.denominator = max(1, len(dens))
    t.examples = [{"building": b, "peak_occupants": pk,
                   "usable_area_m2": round(area, 1),
                   "density_per_m2": round(dv, 3)} for dv, b, pk, area in warn[:MAX_EXAMPLES]]

    T["pos_floor_out_of_range"].denominator = max(1, n_indoor_rows)
    for cid in ("pos_missing_xy", "pos_node_xy_mismatch", "pos_unknown_building",
                "pos_exit_without_enter", "dead_agent_acts", "dead_agent_money",
                "emigrant_acts_later_day", "death_duplicated",
                "dangling_agent_ref"):
        T[cid].denominator = max(1, n_rows)

    return {"n_rows": n_rows, "n_l1_agents": len(l1_ids),
            "n_deaths": len(dead_at), "n_emigrants": len(emigrated_day),
            "n_buildings_used": len(peak),
            "kinds_after_death": dict(kinds_after_death.most_common(10))}


# --------------------------------------------------------------------------- #
# 判定 -> レポート
# --------------------------------------------------------------------------- #
def _status(check: dict, t: Tally, has_map: bool) -> str:
    # 地図(world.map)が読めないランでは、地図依存の検査は **OK と言わない**。
    # 「検査していない」を「違反ゼロ」に見せかけないための分岐。
    if check.get("needs_map") and not has_map:
        return "skipped"
    if check["severity"] == "zero":
        return "ok" if t.n == 0 else "violation"
    # ★小標本では帯そのものが意味を持たない(1 件が帯幅を超える)。「OK」と言わず
    #   `small_n` と言う = 検査していないことを検査結果に見せかけない。
    if t.denominator < int(check.get("min_n", 0)):
        return "small_n"
    lo, hi = check.get("band", (0.0, 0.0))
    rate = _rate(check, t)
    if rate is None:
        return "ok"
    return "ok" if lo <= rate <= hi else "warn"


def _rate(check: dict, t: Tally):
    """帯検査の比率(単位が「最大値」の検査はその値そのもの)。"""
    if check["severity"] != "band":
        return None
    if check["id"] == "cap_dwelling_load":
        return float(t.n)
    den = t.denominator or 0
    if den <= 0:
        return None
    return t.n / float(den)


def build_report(run_dir) -> dict:
    spd = run_dt.steps_per_day(run_dir)
    mps = run_dt.min_per_step(run_dir)
    agents = load_agents(run_dir)
    world = load_map(run_dir)
    T = Tallies()
    audit_roster(agents, world, T)
    stats = audit_l1(run_dir, agents, world, spd, mps, T)

    has_map = bool(world["buildings"])
    checks = []
    for c in CHECKS:
        t = T[c["id"]]
        status = _status(c, t, has_map)
        checks.append({
            "id": c["id"], "category": c["cat"], "severity": c["severity"],
            "label": c["label"], "description": c["desc"],
            "n_violations": int(t.n), "n_checked": int(t.denominator),
            "rate": _rate(c, t), "band": list(c.get("band", ())) or None,
            "unit": c.get("unit", ""), "min_n": int(c.get("min_n", 0)),
            "status": status,
            "examples": t.examples, "extra": t.extra,
        })

    tally = Counter(c["status"] for c in checks)
    zero_bad = [c["id"] for c in checks
                if c["severity"] == "zero" and c["status"] == "violation"]
    return {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "run_dir": os.path.abspath(str(run_dir)),
            "run": os.path.basename(os.path.normpath(str(run_dir))),
            "n_agents_roster": len(agents),
            "map": world["path"], "dt_min": mps, "steps_per_day": spd,
            **stats,
        },
        "checks": checks,
        "totals": {
            "n_checks": len(checks), "ok": tally.get("ok", 0),
            "violation": tally.get("violation", 0), "warn": tally.get("warn", 0),
            "skipped": tally.get("skipped", 0),
            "small_n": tally.get("small_n", 0),
            "zero_violations": zero_bad,
        },
    }


_MARK = {"ok": "OK", "violation": "VIOLATION", "warn": "WARN",
         "skipped": "SKIP", "small_n": "小標本"}


def render_markdown(rep: dict) -> str:
    m = rep["meta"]
    tot = rep["totals"]
    L = [
        f"# 世界の整合性 全数検査: {m['run']}",
        "",
        f"- run: `{m['run_dir']}`",
        f"- L1 {m['n_rows']:,} 行 / L1 に現れた agent {m['n_l1_agents']:,} 人 "
        f"/ 名簿 {m['n_agents_roster']:,} 人",
        f"- 地図: `{m.get('map')}` / 死亡 {m['n_deaths']} 人 / 転出 {m['n_emigrants']} 人 "
        f"/ 在館のあった建物 {m['n_buildings_used']}",
        f"- 判定: OK {tot['ok']} / **VIOLATION {tot['violation']}** / "
        f"WARN {tot['warn']} / SKIP {tot['skipped']} / 小標本 {tot['small_n']}"
        f"(全 {tot['n_checks']} 検査)",
        "",
        "> **「0 であるべき」と「許容帯」は別の表に出す。** 前者は 1 件でも実装の誤り、"
        "後者は 0 でないのが正常な項目で、帯の外に出たときだけ注意を出す。",
    ]
    if tot["zero_violations"]:
        L += ["", "> ★**0 であるべき検査に違反がある**: "
                  + ", ".join(f"`{x}`" for x in tot["zero_violations"])]

    for sev, title, note in (
        ("zero", "0 であるべき検査", "1 件でもあれば実装の誤り。"),
        ("band", "許容帯の検査",
         "0 でないのが正常。帯の外に出たときだけ WARN。"),
    ):
        rows = [c for c in rep["checks"] if c["severity"] == sev]
        L += ["", f"## {title}", "", f"> {note}", "",
              "| 検査 | 分類 | 件数 | 母数 | " +
              ("率/値 | 帯 | " if sev == "band" else "") + "判定 |",
              "|---|---|---:|---:|" + ("---:|---|" if sev == "band" else "") + "---|"]
        for c in rows:
            mid = ""
            if sev == "band":
                r = c["rate"]
                band = c["band"] or [0, 0]
                mid = (f" {('-' if r is None else format(r, '.4g'))} | "
                       f"{band[0]}-{band[1]} {c['unit']} |")
            L.append(f"| {c['label']} | {c['category']} | {c['n_violations']:,} | "
                     f"{c['n_checked']:,} |{mid} {_MARK.get(c['status'], c['status'])} |")

    L += ["", "## 検査ごとの詳細(定義と例示)", ""]
    for c in rep["checks"]:
        L.append(f"### `{c['id']}` — {c['label']}  [{_MARK.get(c['status'])}]")
        L.append("")
        L.append(f"- 分類: {c['category']} / severity: **{c['severity']}** "
                 f"/ 件数 {c['n_violations']:,} / 母数 {c['n_checked']:,}")
        L.append(f"- 定義: {c['description']}")
        if c["examples"]:
            L.append(f"- 例示(最大 {MAX_EXAMPLES} 件):")
            for ex in c["examples"]:
                L.append(f"  - `{json.dumps(ex, ensure_ascii=False)}`")
        elif c["status"] == "ok":
            L.append("- 例示: なし(違反ゼロ)")
        if c["extra"]:
            L.append(f"- 補足: `{json.dumps(c['extra'], ensure_ascii=False)}`")
        L.append("")
    if m.get("kinds_after_death"):
        L += ["> 死亡後に出た kind の内訳: "
              f"`{json.dumps(m['kinds_after_death'], ensure_ascii=False)}`", ""]
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _pick_run(arg_run: str | None) -> str:
    if arg_run:
        d = arg_run if os.path.isabs(arg_run) else os.path.join(RUNS_ROOT, arg_run)
        if not os.path.isdir(d):
            raise SystemExit(f"[invariants] run dir が無い: {d}")
        return d
    if not os.path.isdir(RUNS_ROOT):
        raise SystemExit(f"[invariants] runs が無い: {RUNS_ROOT}")
    cands = []
    for name in os.listdir(RUNS_ROOT):
        pq_path = os.path.join(RUNS_ROOT, name, "l1_events.parquet")
        if os.path.isfile(pq_path):
            cands.append((os.path.getmtime(pq_path), os.path.join(RUNS_ROOT, name)))
    if not cands:
        raise SystemExit("[invariants] l1_events.parquet を持つランが無い")
    cands.sort(reverse=True)
    return cands[0][1]


def analyze(run_dir: str, out_dir: str) -> dict:
    rep = build_report(run_dir)
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "world_invariants.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=2, sort_keys=True)
    md_path = os.path.join(out_dir, "world_invariants.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(rep) + "\n")
    rep["paths"] = {"json": json_path, "md": md_path}
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="世界の整合性の事後・全数検査(読み出し専用)")
    ap.add_argument("run_dir", nargs="?", default=None,
                    help="ラン名 or パス(既定: l1_events.parquet を持つ最新ラン)")
    ap.add_argument("--out", default=None, help="出力先(既定: <run>/invariants)")
    ap.add_argument("--fail-on-violation", action="store_true",
                    help="0 であるべき検査に違反があれば終了コード 1")
    a = ap.parse_args(argv)
    run_dir = _pick_run(a.run_dir)
    out_dir = a.out or os.path.join(run_dir, "invariants")
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(run_dir, out_dir)
    rep = analyze(run_dir, out_dir)
    print(render_markdown(rep))
    print(f"[invariants] -> {rep['paths']['json']}")
    print(f"[invariants] -> {rep['paths']['md']}")
    if a.fail_on_violation and rep["totals"]["zero_violations"]:
        return 1
    return 0


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())
