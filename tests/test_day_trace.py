"""F2「1日トレース」抽出ツール(`scripts/day_trace.py`)の自己検定。

対象は**解析側だけ**(`src/` / `conf/` はゼロタッチ・凍結 SPEC_FILES にも含まれない)。

守るもの(検収基準の順)
  (1) 対応表を写していない/腐らせない: `PATIENT_KEYS` のキーは全て
      `observer/schema.py` に登録済みの kind で、`causality.PATIENT_ACTOR_OVERRIDES`
      の**上位集合**である(タイポと語彙の腐敗を機械で殺す)
  (2) 日の切り方が `world/clock.py: Clock.day`(= sim_min // 1440)と一致し、
      `start_tod` が 00:00 でも 07:00 でも step 窓が厳密に出る
  (3) 主イベント + 受動側の逆引きを両方拾い、**部分文字列の偽陽性は落ちる**
  (4) 同 step 内が「計画→欲求/熟考→移動→行為→会話→内省」に並ぶ(記録順は seq で復元可)
  (5) 計画 vs 実行の突合: 添字経路 / 組経路 / 多義の 3 つが正しく分かれる
  (6) 素材の欠損で**落ちない**・欠損が notes に**正直に出る**(縮退)
  (7) 決定論: 同じ入力で 2 回走らせて md / json / html がバイト一致
  (8) 存在しない agent / day / run dir は明示エラー
  (9) 完全に読み取り専用: ランの入力ファイルは 1 バイトも変わらない
 (10) L2 物語文は**注入した generate をちょうど 1 回**呼ぶだけ(既定は呼ばない)
"""
from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))          # scripts/ は package ではない

import day_trace as DT                              # noqa: E402

from society.observer import causality as CAUSALITY  # noqa: E402
from society.observer import schema as SCHEMA        # noqa: E402


# =========================================================================== #
# 合成ランの組み立て
# =========================================================================== #
L1_COLS = ("step", "sim_min", "agent_id", "kind", "x", "y", "payload",
           "rng_stream", "llm_call_id", "cause_type", "actor_id", "device_id")


def _cid(name: str) -> str:
    """実物と同じ **16 文字**の llm_call_id(= llm_cache が出す sha256[:16])。"""
    return name.ljust(16, "0")[:16]


def _jkey(name: str) -> str:
    """ジャーナルの `key`(sha256 全長 64 文字)。先頭 16 文字が llm_call_id。"""
    return _cid(name).ljust(64, "f")


def _ev(step, agent, kind, payload=None, *, call_id=None, start_min=0, dt=10):
    return {"step": step, "sim_min": start_min + step * dt, "agent_id": agent,
            "kind": kind, "x": 0.0, "y": 0.0,
            "payload": json.dumps(payload or {}, ensure_ascii=False, sort_keys=True),
            "rng_stream": "", "llm_call_id": call_id,
            "cause_type": None, "actor_id": None, "device_id": None}


def _write_l1(run_dir: Path, rows: list[dict]) -> None:
    cols = {c: [r.get(c) for r in rows] for c in L1_COLS}
    table = pa.table({
        "step": pa.array(cols["step"], pa.int32()),
        "sim_min": pa.array(cols["sim_min"], pa.int32()),
        "agent_id": pa.array(cols["agent_id"], pa.int32()),
        "kind": pa.array(cols["kind"], pa.string()),
        "x": pa.array(cols["x"], pa.float32()),
        "y": pa.array(cols["y"], pa.float32()),
        "payload": pa.array(cols["payload"], pa.string()),
        "rng_stream": pa.array(cols["rng_stream"], pa.string()),
        "llm_call_id": pa.array(cols["llm_call_id"], pa.string()),
        "cause_type": pa.array(cols["cause_type"], pa.string()),
        "actor_id": pa.array(cols["actor_id"], pa.int32()),
        "device_id": pa.array(cols["device_id"], pa.string()),
    })
    run_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, run_dir / "l1_events.parquet")


def _write_config(run_dir: Path, dt_min=10, start_tod="00:00") -> None:
    (run_dir / "config.yaml").write_text(
        f"run:\n  n_agents: 3\n  n_steps: 288\n  dt_min: {dt_min}\n"
        f"  start_tod: '{start_tod}'\n", encoding="utf-8")


PLAN_BLOCKS = [
    {"start": 480, "end": 540, "act": "meal", "place": "food",
     "aim": "sustenance", "priority": "must", "flex": "fixed"},
    {"start": 600, "end": 660, "act": "shop", "place": "shop",
     "aim": "errand", "priority": "should", "flex": "slideable"},
    {"start": 720, "end": 780, "act": "walk", "place": "street",
     "aim": "health", "priority": "could", "flex": "droppable"},
    {"start": 840, "end": 900, "act": "home", "place": "home",
     "aim": "rest", "priority": "must", "flex": "fixed"},
]


def make_run(tmp_path: Path, *, start_min=0, dt=10, with_block_index=True,
             sidecars=True, journal=True, name="synth") -> Path:
    """day1(step 144..287)に主人公 7 の 1 日ぶんを持つ最小ラン。"""
    rd = tmp_path / name
    E = lambda *a, **kw: _ev(*a, start_min=start_min, dt=dt, **kw)   # noqa: E731
    b = (lambda i: {"block": i}) if with_block_index else (lambda i: {})
    rows = [
        # ---- day0(窓の外に出ることの確認用)----
        E(10, 7, "arrive", {"node": "n0", "name": "自宅"}),
        # ---- day1 ----
        # 同 step に「内省 → 会話 → 移動 → 計画」を**逆順**で記録し、相順序を試す
        E(150, 7, "reflect", {"mode": "free", "context": "morning",
                              "summary": "朝の整理", "written_back": True},
          call_id=_cid("cid_reflect")),
        E(150, 7, "speak", {"text": "おはよう", "hearers": [8]}, call_id=_cid("cid_speak")),
        E(150, 7, "route_start", {"dest": "n1", "dest_name": "食堂", "dist_m": 100.0,
                                  "mode": "walk"}),
        E(150, 7, "plan_created", {"src": "llm", "model": "mock", "version": 1,
                                   "n": len(PLAN_BLOCKS), "blocks": PLAN_BLOCKS},
          call_id=_cid("cid_plan")),
        # 移動・到着・入店
        E(151, 7, "move_segment", {"dist_m": 80.0, "mode": "walk",
                                   "pts": [[0, 0], [1, 1]]}),
        E(152, 7, "arrive", {"node": "n1", "name": "食堂", "first_visit": True}),
        E(152, 7, "plan_block_start", dict(b(0), act="meal", place="food",
                                           start=480, priority="must",
                                           flex="fixed", node="n1", version=1)),
        E(153, 7, "spend", {"amount": 900.0, "cat": "food", "payee": "食堂"}),
        # 計画の逸脱 3 種
        E(160, 7, "plan_slide", dict(b(1), act="shop", place="shop", start=600,
                                     slid=30)),
        E(170, 7, "plan_block_drop", dict(b(2), act="walk", place="street",
                                          start=720, reason="grace")),
        E(175, 7, "plan_cont_fire", dict(b(1), act="shop", place="shop", start=600,
                                         cond="rain", then="swap_indoor",
                                         applied=True)),
        # 受動側(相手が主語の行に対象が現れる)
        E(180, 8, "speak", {"text": "こんにちは", "hearers": [7, 9]},
          call_id=_cid("cid_other")),
        E(180, 7, "hear", {"speaker": 8}),
        E(181, 8, "dm", {"to": 7, "text": "あとで会おう"}),
        E(182, 8, "relation_tier", {"other": 7, "tier": 2, "count": 3}),
        # ---- 偽陽性の罠: 部分文字列は一致するが id ではない/表に無いキー ----
        E(183, 8, "dm", {"to": 70, "text": "別人あて"}),          # "7" ⊂ "70"
        E(184, 8, "wage", {"amount": 7.0, "to": "cash", "source": "row:7"}),
        E(185, 8, "spend", {"amount": 700.0, "cat": "shop"}),
        # 夜
        E(200, 7, "enter_building", {"building": "b1", "name": "自宅", "home": True}),
        E(201, 7, "reflect", {"mode": "free", "context": "sleep",
                              "summary": "今日はよく歩いた",
                              "belief": "街は思ったより狭い", "written_back": True},
          call_id=_cid("cid_night")),
        E(202, 7, "sleep_start", {"building": "b1", "until_step": 250}),
    ]
    _write_l1(rd, rows)
    _write_config(rd, dt_min=dt, start_tod="00:00" if start_min == 0 else "07:00")
    (rd / "agents.json").write_text(json.dumps([
        {"id": 7, "name": "主人公", "age": 33, "occupation": "会社員",
         "home": "n0", "visitor": False},
        {"id": 8, "name": "相手", "age": 41, "occupation": "店主",
         "home": "n2", "visitor": False},
    ], ensure_ascii=False), encoding="utf-8")
    pq.write_table(pa.table({
        "llm_call_id": pa.array([_cid(c) for c in ("cid_plan", "cid_speak", "cid_reflect",
                                            "cid_night", "cid_other")], pa.string()),
        "agent_id": pa.array([7, 7, 7, 7, 8], pa.int64()),
        "purpose": pa.array(["plan", "social", "reflect", "reflect", "social"],
                            pa.string()),
        "step": pa.array([150, 150, 150, 201, 180], pa.int64()),
        "cached": pa.array([False] * 5, pa.bool_()),
    }), rd / "l1b_llm.parquet")
    if sidecars:
        pq.write_table(pa.table({
            "day": pa.array([1, 2, 2], pa.int32()),
            "agent_id": pa.array([7, 7, 7], pa.int32()),
            "src": pa.array(["buf", "buf", "ep"], pa.string()),
            "idx": pa.array([0, 0, 1], pa.int32()),
            "step": pa.array([152, 152, 180], pa.int32()),
            "kind": pa.array(["event", "event", "heard"], pa.string()),
            "importance": pa.array([4.0, 3.2, 2.0], pa.float32()),
            "text": pa.array(["食堂に入った", "食堂に入った", "相手の話"], pa.string()),
        }), rd / "memory.parquet")
        pq.write_table(pa.table({
            "day": pa.array([1, 2, 2], pa.int32()),
            "agent_id": pa.array([7, 7, 7], pa.int32()),
            "other_id": pa.array([8, 8, 9], pa.int32()),
            "closeness": pa.array([0.10, 0.42, 0.05], pa.float64()),
            "tier": pa.array([1, 2, 0], pa.int32()),
            "count": pa.array([1, 4, 1], pa.int32()),
            "dormant": pa.array([False] * 3, pa.bool_()),
        }), rd / "relations.parquet")
    if journal:
        recs = [
            {"seq": 0, "key": _jkey("cid_plan"), "prompt": "計画のプロンプト",
             "response": '{"blocks": []}', "backend": "mock", "cached": False},
            {"seq": 1, "key": _jkey("cid_night"), "prompt": "内省のプロンプト",
             "response": '{"summary": "今日はよく歩いた"}', "backend": "mock",
             "cached": False},
        ]
        with gzip.open(rd / "llm_journal.jsonl.gz", "wt", encoding="utf-8") as fh:
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rd


def _tree_hash(root: Path) -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.iterdir()) if p.is_file()}


# =========================================================================== #
# (1) 対応表を腐らせない
# =========================================================================== #
def _import_every_society_module() -> dict[str, str]:
    """``society.*`` を全て import して **register_event_kind を全部走らせる**。

    kind の登録は observer/schema.py だけではない(`society/lost_property.py` の
    lost_pickup など、feature module が自分で宣言する流儀がある)。
    `tests/test_causality.py::_import_every_society_module` と同じ手順。
    """
    import importlib
    import pkgutil

    import society
    failed: dict[str, str] = {}
    for mod in pkgutil.walk_packages(society.__path__, "society."):
        try:
            importlib.import_module(mod.name)
        except Exception as ex:                      # noqa: BLE001(理由ごと返す)
            failed[mod.name] = repr(ex)
    return failed


def test_patient_keys_are_registered_kinds():
    """`PATIENT_KEYS` のキーは全て登録済みの kind でなければならない(タイポ検知)。"""
    failed = _import_every_society_module()
    assert failed == {}, f"import できない module がある(網羅を保証できない): {failed}"
    unknown = sorted(k for k in DT.PATIENT_KEYS if k not in SCHEMA.EVENT_KINDS)
    assert unknown == [], f"未登録 kind が対応表に居る(タイポ?): {unknown}"


def test_patient_keys_superset_of_causality_overrides():
    """`causality.PATIENT_ACTOR_OVERRIDES` は本表の**部分集合**であること。"""
    for kind, key in CAUSALITY.PATIENT_ACTOR_OVERRIDES.items():
        assert kind in DT.PATIENT_KEYS, f"{kind} が対応表に無い"
        if isinstance(key, str):
            assert key in DT.PATIENT_KEYS[kind], f"{kind}.{key} が対応表に無い"


def test_patient_keys_exclude_non_agent_keys():
    """ノード/口座を指すキーを**入れていない**ことの明示的な固定(偽の相手を作らない)。"""
    for kind in ("ride", "wage", "move_home", "boredom_explore", "spend",
                 "delivery_trip", "relocate"):
        keys = DT.PATIENT_KEYS.get(kind, ())
        assert "from" not in keys and "to" not in keys, f"{kind} に from/to が居る"


def test_every_registered_kind_has_a_phase_or_default():
    """相の表は既定へ落ちてよい(全 kind を列挙する必要はない)= 落ちないことだけ固定。"""
    _import_every_society_module()
    for kind in SCHEMA.EVENT_KINDS:
        assert isinstance(DT._PHASE.get(kind, DT._DEFAULT_PHASE), int)


# =========================================================================== #
# (2) 日の切り方
# =========================================================================== #
@pytest.mark.parametrize("start_min,expect", [(0, (144, 287)), (420, (102, 245))])
def test_day_window_matches_clock_day(tmp_path, start_min, expect):
    """`day_window` は `sim_min // 1440`(Clock.day)と厳密に一致する。"""
    rd = make_run(tmp_path, start_min=start_min, name=f"w{start_min}")
    lo, hi = DT.day_window(str(rd), 1)
    assert lo == expect[0]
    # 上端はランの max_step で切られる(合成ランは step 202 で終わる)
    assert hi == min(expect[1], 202)
    for step in (lo, hi):
        assert (start_min + step * 10) // 1440 == 1
    assert (start_min + (lo - 1) * 10) // 1440 == 0


def test_observed_days(tmp_path):
    rd = make_run(tmp_path, name="days")
    assert DT.observed_days(str(rd)) == [0, 1]


def test_hhmm():
    assert DT.hhmm(0) == "00:00"
    assert DT.hhmm(1440 + 8 * 60 + 5) == "08:05"


# =========================================================================== #
# (3) 収集: 主 + 受動 + 偽陽性の排除
# =========================================================================== #
def test_collect_rows_picks_actor_and_patient(tmp_path):
    rd = make_run(tmp_path, name="collect")
    lo, hi = DT.day_window(str(rd), 1)
    rows = DT.collect_rows(str(rd), 7, lo, hi)
    kinds = [(r["kind"], r["side"]) for r in rows]
    # 主イベント
    assert ("plan_created", "actor") in kinds
    assert ("spend", "actor") in kinds
    # 受動側(相手が主語の行)
    assert ("speak", "patient") in kinds          # 8 が 7 を hearers に含む
    assert ("dm", "patient") in kinds             # 8 → 7
    assert ("relation_tier", "patient") in kinds  # 8 の other=7
    # day0 の行は窓の外
    assert all(r["step"] >= lo for r in rows)


def test_collect_rows_rejects_substring_false_positives(tmp_path):
    """`to=70` / `source="row:7"` / `amount=700` は **7 の行ではない**。"""
    rd = make_run(tmp_path, name="fp")
    lo, hi = DT.day_window(str(rd), 1)
    rows = DT.collect_rows(str(rd), 7, lo, hi)
    dms = [r for r in rows if r["kind"] == "dm"]
    assert [r["payload"]["to"] for r in dms] == [7]          # to=70 は落ちる
    assert not [r for r in rows if r["kind"] == "wage"]      # 表に無いキーは拾わない
    assert not [r for r in rows if r["kind"] == "spend" and r["side"] == "patient"]


def test_ids_in_handles_scalar_and_list():
    assert DT._ids_in({"hearers": [1, 2]}, ("hearers",)) == [1, 2]
    assert DT._ids_in({"to": "3"}, ("to",)) == [3]
    assert DT._ids_in({"to": None, "x": True}, ("to", "x")) == []


# =========================================================================== #
# (4) 同 step の相順序(記録順は seq で復元可)
# =========================================================================== #
def test_same_step_phase_order(tmp_path):
    rd = make_run(tmp_path, name="phase")
    lo, hi = DT.day_window(str(rd), 1)
    rows = [r for r in DT.collect_rows(str(rd), 7, lo, hi) if r["step"] == 150]
    assert [r["kind"] for r in rows] == [
        "plan_created", "route_start", "speak", "reflect"]
    # 記録順(逆順に書いた)は seq で復元できる = 情報を落としていない
    assert [r["kind"] for r in sorted(rows, key=lambda r: r["seq"])] == [
        "reflect", "speak", "route_start", "plan_created"]


# =========================================================================== #
# (5) 計画 vs 実行の突合
# =========================================================================== #
def test_plan_reconciliation_by_block_index(tmp_path):
    rd = make_run(tmp_path, with_block_index=True, name="planidx")
    tr = DT.build_trace(str(rd), 7, 1)
    plan = tr["plan"]
    assert plan["present"] and plan["attribution"] == "block_index"
    assert plan["ambiguous"] == 0 and plan["unmatched"] == 0
    st = {b["i"]: b["status"] for b in plan["blocks"]}
    assert st == {0: "executed", 1: "slid", 2: "dropped", 3: "untouched"}
    assert plan["blocks"][1]["slid"] == 30
    assert plan["blocks"][2]["drop_reason"] == "grace"
    assert plan["blocks"][1]["cont"] == {"cond": "rain", "then": "swap_indoor",
                                         "applied": True}
    assert plan["counts"] == {"executed": 1, "dropped": 1, "slid": 1, "untouched": 1}


def test_plan_reconciliation_falls_back_to_tuple_match(tmp_path):
    """`block` 添字が無い旧ランでも (act, place, start) で同じ結末が出る。"""
    rd = make_run(tmp_path, with_block_index=False, name="plantuple")
    plan = DT.build_trace(str(rd), 7, 1)["plan"]
    assert plan["attribution"] == "tuple_match"
    st = {b["i"]: b["status"] for b in plan["blocks"]}
    assert st == {0: "executed", 1: "slid", 2: "dropped", 3: "untouched"}


def test_plan_tuple_match_reports_ambiguity(tmp_path):
    """同値ブロックが 2 件あると**当てにいかず** ambiguous に数える。"""
    rd = tmp_path / "amb"
    dup = [PLAN_BLOCKS[0], dict(PLAN_BLOCKS[0])]
    _write_l1(rd, [
        _ev(150, 7, "plan_created", {"src": "llm", "n": 2, "blocks": dup}),
        _ev(152, 7, "plan_block_start", {"act": "meal", "place": "food",
                                         "start": 480}),
    ])
    _write_config(rd)
    plan = DT.build_plan(DT.collect_rows(str(rd), 7, 144, 287))
    assert plan["ambiguous"] == 1
    assert all(b["status"] == "untouched" for b in plan["blocks"])


def test_plan_absent_is_unmeasured_not_zero(tmp_path):
    rd = tmp_path / "noplan"
    _write_l1(rd, [_ev(150, 7, "arrive", {"node": "n1"})])
    _write_config(rd)
    plan = DT.build_plan(DT.collect_rows(str(rd), 7, 144, 287))
    assert plan["present"] is False and plan["blocks"] == []
    assert "plan_created" in plan["reason"]


# =========================================================================== #
# (6) シーンカード・統計・記憶・関係
# =========================================================================== #
def test_scenes_break_on_place_and_exclude_self(tmp_path):
    rd = make_run(tmp_path, name="scenes")
    tr = DT.build_trace(str(rd), 7, 1)
    scenes = tr["scenes"]
    assert len(scenes) >= 3
    assert any(s["place"] == "食堂" for s in scenes)
    assert any("自宅" in str(s["place"]) for s in scenes)
    # 相手集合に**本人が入らない**(hearers には 7 自身も入りうる)
    for s in scenes:
        assert 7 not in s["partners"]
    assert 8 in set().union(*[set(s["partners"]) for s in scenes])
    # 嵩張る move_segment はカードに 1 行ずつ載せず距離に畳む(L0 には残る)
    assert sum(s["bulk"].get("move_segment", 0) for s in scenes) == 1
    assert any(s["dist_m"] == 80.0 for s in scenes)
    assert any(r["kind"] == "move_segment" for r in tr["timeline"])


def test_stats_and_memory_and_relations(tmp_path):
    tr = DT.build_trace(str(make_run(tmp_path, name="stats")), 7, 1)
    s = tr["stats"]
    assert s["distance_m"] == 80.0 and s["n_routes"] == 1
    assert s["spend_total"] == 900.0 and s["spend_by_cat"] == {"food": 900.0}
    # 8 は直接の相手、9 は同じ発話の**共在の聞き手**(speak.hearers)
    assert s["partners"] == [8, 9] and s["n_partners"] == 2
    assert s["n_speak"] == 1 and s["n_hear"] == 1
    # 記憶: 同じ記憶が day1/day2 に重複しているが 1 件に畳む(減衰が見える)
    mem = tr["memory"]
    assert len(mem) == 2
    m0 = [m for m in mem if m["step"] == 152][0]
    assert (m0["first_day"], m0["last_day"]) == (1, 2)
    assert m0["importance_first"] == pytest.approx(4.0)
    assert m0["importance_last"] == pytest.approx(3.2)
    # 関係: day1 → day2 の差分(★当日の変化 = 翌日スナップ − 当日スナップ)
    rel = tr["relations"]
    assert rel["present"] is True
    d8 = [d for d in rel["deltas"] if d["other_id"] == 8][0]
    assert d8["delta"] == pytest.approx(0.32, abs=1e-9)
    assert d8["tier_before"] == 1 and d8["tier_after"] == 2
    d9 = [d for d in rel["deltas"] if d["other_id"] == 9][0]
    assert d9["new"] is True


def test_relations_last_day_is_unmeasured(tmp_path):
    """翌日のスナップが無い日は差分を**測らない**(0 と書かない)。"""
    rel = DT.load_relations(str(make_run(tmp_path, name="rel")), 7, 2)
    assert rel["present"] is False and "day3" in rel["reason"]


def test_reflections_quote_the_agent(tmp_path):
    tr = DT.build_trace(str(make_run(tmp_path, name="refl")), 7, 1)
    ctx = [r["context"] for r in tr["reflections"]]
    assert ctx == ["morning", "sleep"]
    night = tr["reflections"][1]
    assert night["belief"] == "街は思ったより狭い"
    assert night["purpose"] == "reflect"          # l1b_llm と結線できている
    assert "response_full" not in night           # --with-journal 無しでは全文を出さない


def test_with_journal_pulls_full_text(tmp_path):
    tr = DT.build_trace(str(make_run(tmp_path, name="jr")), 7, 1, with_journal=True)
    night = tr["reflections"][1]
    assert night["response_full"] == '{"summary": "今日はよく歩いた"}'
    assert tr["thoughts"][_cid("cid_plan")]["prompt"] == "計画のプロンプト"


# =========================================================================== #
# (7) 縮退: 素材が無くても落ちない・無いと正直に書く
# =========================================================================== #
def test_degrades_without_sidecars_and_journal(tmp_path):
    rd = make_run(tmp_path, sidecars=False, journal=False, name="bare")
    tr = DT.build_trace(str(rd), 7, 1, with_journal=True)
    assert tr["memory"] == []
    assert tr["relations"]["present"] is False
    notes = " ".join(tr["materials"]["notes"])
    assert "memory.parquet が無い" in notes
    assert "llm_journal" in notes
    assert "llm_role" in notes                     # provlink OFF も告知する
    md = DT.render_md(tr, journal_on=True)
    assert "記憶の節は出ない" in md
    assert DT.render_html(tr).startswith("<!doctype html>")


def test_l1_only_run_still_renders(tmp_path):
    """l1_events.parquet だけのランでも md/html が出る(l1b も名簿も無い)。"""
    rd = tmp_path / "l1only"
    _write_l1(rd, [_ev(150, 7, "arrive", {"node": "n1", "name": "広場"}),
                   _ev(151, 7, "speak", {"text": "やあ", "hearers": [8]})])
    tr = DT.build_trace(str(rd), 7, 1)
    assert tr["agent"] == {"id": 7}                # 名前は捏造しない
    md = DT.render_md(tr)
    assert "#7" in md                              # id のまま出る
    assert DT.render_html(tr)


def test_empty_day_is_not_an_error(tmp_path):
    rd = make_run(tmp_path, name="emptyday")
    tr = DT.build_trace(str(rd), 8, 0)             # 8 は day0 に 1 件も無い
    assert tr["timeline"] == [] and tr["scenes"] == []
    assert "1 件もイベントが無い" in DT.render_md(tr)


def test_provlink_is_tri_state(tmp_path):
    """行が 0 件の日に「llm_link OFF」と断定しない(判定不能は判定不能と書く)。"""
    rd = make_run(tmp_path, name="tri")
    empty = DT.build_trace(str(rd), 8, 0)
    assert empty["materials"]["provlink"] == "unknown"
    assert "判定不能" in DT.render_md(empty)
    assert not any("llm_role" in n for n in empty["materials"]["notes"])
    # 合成ランは llm_role を刻んでいない = OFF と判定され、根拠つきで告知される
    off = DT.build_trace(str(rd), 7, 1)
    assert off["materials"]["provlink"] is False
    assert any("llm_role" in n for n in off["materials"]["notes"])


# =========================================================================== #
# (8) CLI: 決定論・明示エラー・読み取り専用
# =========================================================================== #
def test_cli_is_deterministic(tmp_path):
    rd = make_run(tmp_path, name="det")
    outs = []
    for tag in ("a", "b"):
        o = tmp_path / tag
        assert DT.main([str(rd), "--agent", "7", "--day", "1", "--html",
                        "--out", str(o)]) == 0
        outs.append({p.name: p.read_bytes() for p in sorted(o.iterdir())})
    assert set(outs[0]) == {"agent7_day1.md", "agent7_day1.json", "agent7_day1.html"}
    assert outs[0] == outs[1], "同じ入力で出力がバイト一致しない"


def test_cli_all_days_writes_one_file_each(tmp_path):
    rd = make_run(tmp_path, name="alldays")
    o = tmp_path / "out_all"
    assert DT.main([str(rd), "--agent", "7", "--out", str(o)]) == 0
    assert sorted(p.name for p in o.iterdir()) == [
        "agent7_day0.json", "agent7_day0.md", "agent7_day1.json", "agent7_day1.md"]


def test_cli_errors_are_explicit(tmp_path):
    rd = make_run(tmp_path, name="err")
    o = tmp_path / "out_err"
    with pytest.raises(SystemExit, match="agent 999"):
        DT.main([str(rd), "--agent", "999", "--day", "1", "--out", str(o)])
    with pytest.raises(SystemExit, match="day 9"):
        DT.main([str(rd), "--agent", "7", "--day", "9", "--out", str(o)])
    with pytest.raises(SystemExit, match="ラン dir が無い"):
        DT.main([str(tmp_path / "nope"), "--agent", "7"])
    with pytest.raises(SystemExit, match="--agent"):
        DT.main([str(rd), "--out", str(o)])
    with pytest.raises(SystemExit, match="--day N"):
        DT.main([str(rd), "--top", "3", "--out", str(o)])


def test_cli_missing_l1_is_explicit(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit, match="l1_events.parquet"):
        DT.main([str(empty), "--agent", "7"])


def test_run_dir_is_never_modified(tmp_path):
    """完全に事後 = ランの入力ファイルは 1 バイトも変わらない(出力は --out のみ)。"""
    rd = make_run(tmp_path, name="ro")
    before = _tree_hash(rd)
    DT.main([str(rd), "--agent", "7", "--day", "1", "--html",
             "--with-journal", "--out", str(tmp_path / "ro_out")])
    DT.main([str(rd), "--day", "1", "--top", "5", "--out", str(tmp_path / "ro_out")])
    assert _tree_hash(rd) == before
    assert not (rd / "analysis").exists()


# =========================================================================== #
# (9) 人選補助 --top
# =========================================================================== #
def test_rank_agents_is_deterministic_and_ranks_by_surprisal(tmp_path):
    rd = tmp_path / "rank"
    rows = []
    for i in range(20):                      # 1 は「ありふれた行為」だけを大量に
        rows.append(_ev(150 + i, 1, "move_segment", {"dist_m": 1.0}))
    rows += [_ev(150, 2, "move_segment", {"dist_m": 1.0}),
             _ev(151, 2, "venture_open", {"name": "屋台"}),   # 希少
             _ev(152, 2, "event_host", {"title": "会"})]      # 希少
    _write_l1(rd, rows)
    _write_config(rd)
    res = DT.rank_agents(str(rd), 1, 10)
    assert [r["agent_id"] for r in res["top"]] == [2, 1]
    assert res["top"][0]["diversity"] == 3 and res["top"][1]["diversity"] == 1
    assert DT.rank_agents(str(rd), 1, 10) == res           # 決定論


def test_rank_agents_ties_break_by_agent_id(tmp_path):
    rd = tmp_path / "tie"
    _write_l1(rd, [_ev(150, 9, "arrive", {"node": "a"}),
                   _ev(151, 3, "arrive", {"node": "b"}),
                   _ev(152, 5, "arrive", {"node": "c"})])
    _write_config(rd)
    res = DT.rank_agents(str(rd), 1, 10)
    assert [r["agent_id"] for r in res["top"]] == [3, 5, 9]


# =========================================================================== #
# (10) L2 物語文: 既定 OFF・注入した generate をちょうど 1 回
# =========================================================================== #
def test_narrative_is_off_by_default(tmp_path):
    tr = DT.build_trace(str(make_run(tmp_path, name="noll")), 7, 1)
    assert tr["narrative"] is None
    assert "物語文" not in DT.render_md(tr)


def test_narrate_calls_injected_generator_exactly_once(tmp_path):
    tr = DT.build_trace(str(make_run(tmp_path, name="nar")), 7, 1)
    calls: list[str] = []

    def fake(prompt: str) -> str:
        calls.append(prompt)
        return "きょうは食堂で昼を食べた。散歩は諦めた。"

    tr["narrative"] = DT.narrate(tr, fake, backend="mock-injected", model="m")
    assert len(calls) == 1
    # プロンプトは決定論に組まれ、計画の結末と本人の内省を含む
    assert "朝の計画" in calls[0] and "meal@food" in calls[0]
    assert "街は思ったより狭い" in calls[0]
    assert "上の事実だけで日記" in calls[0]
    md = DT.render_md(tr)
    assert "きょうは食堂で昼を食べた" in md
    assert "生成物" in md                       # 一次事実でないと明記する
    assert "生成物" in DT.render_html(tr)
    # 2 回組んでも同じプロンプト(決定論)
    assert DT.narrate_prompt(tr) == calls[0]


def test_render_line_marks_passive_actor(tmp_path):
    """受動行は「誰の行為か」を頭に付ける(誤読を防ぐ)。"""
    tr = DT.build_trace(str(make_run(tmp_path, name="passive")), 7, 1)
    names = {int(k): v for k, v in tr["names"].items()}
    row = [r for r in tr["timeline"]
           if r["kind"] == "relation_tier" and r["side"] == "patient"][0]
    line = DT.line_of(row, names)
    assert line.startswith("[相手(8) の行]")
