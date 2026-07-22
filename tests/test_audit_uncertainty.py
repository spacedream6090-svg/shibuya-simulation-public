"""第51バッチ: scripts/audit_uncertainty.py の検証(不確実性の監査)。

house style は tests/test_analyze_founders.py に倣う(scripts/ を path 追加して import・
純関数は dict 列を直接渡し・end-to-end は tmp に最小ランを書いて analyze)。

確かめること:
  1. 運対応表 LUCK_KINDS の全 kind が schema(society.observer.schema)に登録済み。
  2. 帰属ルール: agent スコープ本人・crime は被害者・phase フィルタ(onset のみ)。
  3. world スコープ(agent_id=-1 のショック)は「その日 街に居た住民全員」へ曝露。
  4. 出力スキーマ + 決定論(同一入力 → 同一 JSON バイト)。
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))          # scripts/ は package ではない
sys.path.insert(0, str(_ROOT / "src"))              # society.observer.schema

import audit_uncertainty as au                       # noqa: E402

SPD = au.STEPS_PER_DAY


def _ev(step, agent, kind, payload=None):
    return {"step": step, "sim_min": step * 10, "agent": agent,
            "kind": kind, "payload": payload or {}}


# --------------------------------------------------------------------------- #
# 1. 対応表の kind は全て schema 登録済み(将来 kind 追加時の回帰防止)
# --------------------------------------------------------------------------- #
def test_luck_kinds_all_registered():
    from society.observer.schema import EVENT_KINDS
    unregistered = [k for k in au.LUCK_KINDS if k not in EVENT_KINDS]
    assert unregistered == [], f"schema 未登録の luck kind: {unregistered}"


def test_luck_kinds_have_required_fields():
    for kind, spec in au.LUCK_KINDS.items():
        assert spec["scope"] in ("agent", "world"), kind
        assert spec["valence"] in ("neg", "pos", "mixed"), kind
        assert isinstance(spec["category"], str) and spec["category"], kind


# --------------------------------------------------------------------------- #
# 2. 帰属ルール: agent スコープ・crime 被害者・phase フィルタ
# --------------------------------------------------------------------------- #
def test_agent_scope_attribution():
    events = [
        _ev(0, 1, "detour", {"to": "n1"}),
        _ev(3, 1, "interrupt", {"act": "shop"}),
        _ev(5, 2, "boredom_explore", {"to_kind": "cafe"}),
    ]
    attr = au.attribute_chances(events, {})
    pa = attr["per_agent"]
    assert pa[1]["detour"] == 1 and pa[1]["interrupt"] == 1
    assert pa[2]["curiosity"] == 1
    assert attr["by_kind"]["detour"] == 1


def test_crime_attributed_to_victim_not_offender():
    # 加害者 agent_id=5、被害者 payload.victim=3 → 帰属は 3(不運を被った側)。
    events = [_ev(10, 5, "crime", {"kind": "theft", "victim": 3, "offender": 5})]
    attr = au.attribute_chances(events, {})
    assert attr["per_agent"][3]["crime"] == 1
    assert 5 not in attr["per_agent"]                # 加害者には帰属しない


def test_phase_filter_onset_only():
    # illness は state==onset のみ、disaster/infra は phase==onset のみ計上。
    events = [
        _ev(0, 1, "illness", {"state": "onset"}),
        _ev(20, 1, "illness", {"state": "recover"}),   # 回復は数えない
    ]
    attr = au.attribute_chances(events, {})
    assert attr["per_agent"][1]["health"] == 1
    assert attr["by_kind"]["illness"] == 1


# --------------------------------------------------------------------------- #
# 3. world スコープ: その日 街に居た住民全員へ曝露
# --------------------------------------------------------------------------- #
def test_world_scope_exposure_to_active_residents():
    agents = {1: {"visitor": False}, 2: {"visitor": False},
              3: {"visitor": False}, 9: {"visitor": True}}
    events = [
        _ev(0, 1, "arrive", {"node": "n1"}),           # 1,2 は day0 に在街
        _ev(0, 2, "arrive", {"node": "n2"}),
        _ev(SPD, 3, "arrive", {"node": "n3"}),         # 3 は day1 に在街(day0 は不在)
        _ev(5, -1, "disaster", {"kind": "台風", "phase": "onset"}),   # day0 の世界ショック
        _ev(6, -1, "disaster", {"kind": "台風", "phase": "clear"}),   # clear は数えない
    ]
    attr = au.attribute_chances(events, agents)
    # day0 在街の住民 1,2 は shock 曝露を受ける。day1 の 3 は受けない。
    assert attr["per_agent"][1]["shock"] == 1
    assert attr["per_agent"][2]["shock"] == 1
    assert attr["per_agent"].get(3, Counter())["shock"] == 0
    assert attr["n_world_exposures"] == 2              # 住民2人 × onset1件
    assert attr["by_kind"]["disaster"] == 1            # onset のみ(clear は除外)


def test_transit_delay_world_exposure():
    agents = {1: {"visitor": False}, 2: {"visitor": False}}
    events = [
        _ev(0, 1, "arrive", {"node": "n1"}),
        _ev(0, 2, "arrive", {"node": "n2"}),
        _ev(4, -1, "transit_delay", {"kind": "遅延"}),   # phase なし=常に計上
    ]
    attr = au.attribute_chances(events, agents)
    assert attr["per_agent"][1]["delay"] == 1
    assert attr["per_agent"][2]["delay"] == 1
    assert attr["n_world_exposures"] == 2


# --------------------------------------------------------------------------- #
# 4. 監査サマリのスキーマ + 分布統計
# --------------------------------------------------------------------------- #
def test_audit_summary_schema():
    agents = {1: {"visitor": False}, 2: {"visitor": False}}
    events = [
        _ev(0, 1, "arrive", {}), _ev(0, 2, "arrive", {}),
        _ev(1, 1, "detour", {}), _ev(2, 1, "detour", {}),
        _ev(3, 2, "interrupt", {}),
    ]
    res = au.audit(events, agents)
    for key in ("n_events_total", "n_agent_events", "chance_events_attributed",
                "chance_fraction_of_agent_events", "per_agent_day", "by_kind",
                "by_category", "luck_kinds_registered"):
        assert key in res, key
    d = res["per_agent_day"]
    for key in ("n_cells", "mean", "median", "p90", "max", "histogram"):
        assert key in d, key
    # agent1=2件(detour×2)、agent2=1件(interrupt)。2 セル。
    assert res["chance_events_attributed"] == 3
    assert d["max"] == 2
    assert d["n_cells"] == 2


def test_quantile_helper():
    assert au._quantile([], 0.5) == 0.0
    assert au._quantile([5], 0.9) == 5.0
    assert au._quantile([0, 10], 0.5) == 5.0          # 線形補間


# --------------------------------------------------------------------------- #
# 5. end-to-end: 合成ランを tmp に書いて analyze(決定論=同一 JSON バイト)
# --------------------------------------------------------------------------- #
def _write_run(run_dir: Path, events, agents_list):
    import pyarrow as pa
    import pyarrow.parquet as pq

    run_dir.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([
        pa.field("step", pa.int32()), pa.field("sim_min", pa.int32()),
        pa.field("agent_id", pa.int32()), pa.field("kind", pa.string()),
        pa.field("payload", pa.string()),
    ])
    cols = {"step": [], "sim_min": [], "agent_id": [], "kind": [], "payload": []}
    for e in events:
        cols["step"].append(e["step"])
        cols["sim_min"].append(e["step"] * 10)
        cols["agent_id"].append(e["agent"])
        cols["kind"].append(e["kind"])
        cols["payload"].append(json.dumps(e["payload"], ensure_ascii=False))
    pq.write_table(pa.Table.from_pydict(cols, schema=schema),
                   str(run_dir / "l1_events.parquet"))
    (run_dir / "agents.json").write_text(
        json.dumps(agents_list, ensure_ascii=False), encoding="utf-8")


def _scenario():
    agents_list = [{"id": i, "occupation": "A", "age": 30 + i, "visitor": False}
                   for i in range(4)]
    events = []
    for aid in range(4):                               # 全員 day0 在街
        events.append(_ev(aid, aid, "arrive", {"node": f"n{aid}"}))
    events += [_ev(1, 0, "detour", {}), _ev(2, 0, "interrupt", {}),
               _ev(3, 1, "boredom_explore", {}),
               _ev(4, 2, "crime", {"victim": 3, "offender": 2}),   # 被害者=3
               _ev(5, -1, "transit_delay", {"kind": "遅延"})]      # world 曝露
    return events, agents_list


def test_end_to_end_and_determinism(tmp_path):
    events, agents_list = _scenario()
    r1 = tmp_path / "p1" / "syn"
    r2 = tmp_path / "p2" / "syn"
    _write_run(r1, events, agents_list)
    _write_run(r2, events, agents_list)
    res1 = au.analyze(str(r1), str(r1 / "audit"))
    au.analyze(str(r2), str(r2 / "audit"))

    # 帰属: detour1+interrupt1(0)、boredom1(1)、crime→victim3、delay は全4人へ曝露。
    assert res1["by_category"]["detour"] == 1
    assert res1["by_category"]["crime"] == 1           # 被害者3へ
    assert res1["world_exposures"] == 4                # 住民4人 × 遅延1件
    assert Path(res1["paths"]["json"]).exists()

    j1 = (r1 / "audit" / "uncertainty.json").read_text(encoding="utf-8")
    j2 = (r2 / "audit" / "uncertainty.json").read_text(encoding="utf-8")
    assert j1 == j2                                    # 決定論(同一入力→同一 JSON)


def test_degrade_no_chance_events(tmp_path):
    """揺らぎイベントが1件も無い旧ランでも例外なく 0 件で縮退。"""
    agents_list = [{"id": i, "occupation": "A", "age": 30, "visitor": False}
                   for i in range(3)]
    events = [_ev(0, 0, "arrive", {}), _ev(1, 1, "speak", {"hearers": [2]})]
    run_dir = tmp_path / "old"
    _write_run(run_dir, events, agents_list)
    res = au.analyze(str(run_dir), str(run_dir / "audit"))
    assert res["chance_events_attributed"] == 0
    assert res["per_agent_day"]["max"] == 0
    assert Path(res["paths"]["txt"]).exists()
