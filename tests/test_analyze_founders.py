"""第37バッチ Track A: scripts/analyze_founders.py の検証。

設計哲学 = 行動からファウンダーを検出(定義で強制しない)。ここでは既知の正解を持つ
小さな合成 L1 イベントで、検出の一致・対照群マッチングの決定論・イベント欠損時の縮退動作を確かめる。

house style: scripts/ を path 追加して import(tests/test_detect_emergence.py に倣う)。
純関数は dict 列を直接渡し、end-to-end は tmp に最小ラン(parquet + agents/traits)を書いて analyze。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))          # scripts/ は package ではない

import analyze_founders as af                        # noqa: E402

SPD = af.STEPS_PER_DAY


def _ev(step, agent, kind, payload=None):
    return {"step": step, "sim_min": step * 10, "agent": agent,
            "kind": kind, "payload": payload or {}}


# --------------------------------------------------------------------------- #
# 1. 起業型の検出(venture_open / permit / fulltime / deviance)
# --------------------------------------------------------------------------- #
def test_detect_venture_founders():
    events = [
        _ev(300, 1, "venture_open", {"name": "屋台A"}),            # open
        _ev(360, 1, "venture_fulltime", {"name": "屋台A"}),        # 同一人の本業化
        _ev(200, 4, "venture_open", {"name": "闇市", "permitted": False}),  # deviance
        _ev(50, 5, "venture_permit", {"name": "申請店", "outcome": "denied"}),
    ]
    out = af.detect_venture_founders(events)
    assert set(out) == {1, 4, 5}
    assert out[1]["onset_step"] == 300               # open が fulltime より前
    assert out[1]["subtypes"] == {"open", "fulltime"}
    assert out[4]["subtypes"] == {"deviance"}
    assert out[5]["subtypes"] == {"permit"}


# --------------------------------------------------------------------------- #
# 2. 政治型の検出(candidacy + 高頻度 proposal)+ 欠損時の縮退
# --------------------------------------------------------------------------- #
def test_detect_political_founders_candidacy_and_proposer():
    events = [
        _ev(150, 2, "candidacy", {"day": 1, "deposit": 300000}),
        _ev(100, 5, "proposal", {"proposal_id": "p1"}),
        _ev(250, 5, "proposal", {"proposal_id": "p2"}),            # 2件目で閾値到達
        _ev(400, 5, "proposal", {"proposal_id": "p3"}),
        _ev(120, 6, "proposal", {"proposal_id": "q1"}),           # 1件のみ=非該当
    ]
    out = af.detect_political_founders(events, proposal_min=2)
    assert set(out) == {2, 5}
    assert out[2]["subtypes"] == {"candidacy"}
    assert out[2]["onset_step"] == 150
    assert out[5]["subtypes"] == {"proposer"}
    assert out[5]["onset_step"] == 250                # proposal_min 件目の step
    assert 6 not in out


def test_detect_political_degrade_when_no_candidacy():
    """candidacy が1件も無い旧ランでも例外なく動く(proposal だけで縮退)。"""
    events = [
        _ev(100, 5, "proposal", {"proposal_id": "p1"}),
        _ev(200, 5, "proposal", {"proposal_id": "p2"}),
        _ev(10, 1, "speak", {"text": "x"}),           # 無関係イベントが混じっても平気
    ]
    out = af.detect_political_founders(events, proposal_min=2)
    assert set(out) == {5}
    # candidacy が無いイベント列でも KeyError 等を出さない
    assert af.detect_political_founders([], proposal_min=2) == {}


# --------------------------------------------------------------------------- #
# 3. ハブ型: 次数・媒介中心性(Brandes)・発信超過
# --------------------------------------------------------------------------- #
def test_relation_graph_and_hub_degree():
    # agent 3 が 4,5,6 と関係(スター)= 次数3。他は次数1。
    events = [
        _ev(10, 3, "relation_tier", {"other": 4, "tier": 1}),
        _ev(20, 3, "relation_tier", {"other": 5, "tier": 1}),
        _ev(30, 3, "relation_tier", {"other": 6, "tier": 1}),
    ]
    adj, reach = af.build_relation_graph(events)
    assert adj[3] == {4, 5, 6}
    assert adj[4] == {3}                              # 双方向に張られる
    assert reach[3]["degree_step"][3] == 30           # 3人目に達した step
    comm = af.communication_stats(events)
    hub = af.detect_hub_founders(adj, reach, comm, hub_degree=3, top_k=5)
    assert 3 in hub
    assert "degree" in hub[3]["signals"]
    assert hub[3]["degree"] == 3
    assert hub[3]["onset_step"] == 30                 # 次数閾値の到達 step
    # 次数1の葉はハブでない
    assert 4 not in hub


def test_betweenness_path_bridge():
    """パス 0-1-2 の中央ノード1が唯一の橋 → 媒介中心性>0、両端は0。"""
    adj = {0: {1}, 1: {0, 2}, 2: {1}}
    cb = af.betweenness_bfs(adj)
    assert cb[1] == 1.0                               # 無向 Brandes: (0-2)の唯一経路が1を通る
    assert cb[0] == 0.0 and cb[2] == 0.0


def test_hub_broadcast_surplus():
    """speak/dm 発信 > hear 受信 の発信超過者を top_k で拾う。"""
    events = [
        _ev(5, 9, "speak", {"text": "a", "hearers": [1]}),
        _ev(6, 9, "speak", {"text": "b", "hearers": [2]}),
        _ev(7, 9, "dm", {"to": 1, "text": "c"}),
        _ev(8, 1, "hear", {"speaker": 9}),            # 9 が他者に聴取された
    ]
    comm = af.communication_stats(events)
    assert comm[9]["speak_out"] == 2 and comm[9]["dm_out"] == 1
    assert comm[9]["heard_by"] == 1
    hub = af.detect_hub_founders({}, {}, comm, hub_degree=99, top_k=5)
    assert 9 in hub and "broadcast" in hub[9]["signals"]
    assert hub[9]["broadcast_surplus"] == 3           # (2+1) 発信 - 0 受信


# --------------------------------------------------------------------------- #
# 4. 組織候補: 反復共在クラスタ(既存台帳=勤務先/自宅の除外)
# --------------------------------------------------------------------------- #
def test_copresence_clusters():
    agents = {
        0: {"work_building": "bwork", "home_building": "bh0"},
        1: {"work_building": "bwork", "home_building": "bh1"},
        7: {"work_building": "bw7", "home_building": "bh7"},
        8: {"work_building": "bw8", "home_building": "bh8"},
    }
    events = []
    for d in range(3):                                # 0,1,2 の3日連続
        s = d * SPD
        events += [_ev(s, 0, "enter_building", {"building": "bwork"}),
                   _ev(s, 1, "enter_building", {"building": "bwork"}),
                   _ev(s, 7, "enter_building", {"building": "bcafe"}),
                   _ev(s, 8, "enter_building", {"building": "bcafe"})]
    res = af.detect_org_candidates(events, agents, {}, min_days=3, min_core=2, max_core=12)
    # bcafe = 誰の勤務先/自宅でもない → emergent 候補
    assert res["n_emergent"] == 1
    cand = res["candidates"][0]
    assert cand["building"] == "bcafe"
    assert cand["members"] == [7, 8]
    assert cand["copresent_days"] == 3
    # bwork = メンバー過半の勤務先 → known(候補から除外)
    assert res["n_known"] == 1
    assert res["known_workplace_home"][0]["building"] == "bwork"


def test_copresence_requires_consecutive_days():
    """連続していない共在(gap あり)は min_days 未満で候補にならない。"""
    agents = {7: {"work_building": "bw7", "home_building": "bh7"},
              8: {"work_building": "bw8", "home_building": "bh8"}}
    events = []
    for d in (0, 1, 3):                               # 2日目が欠け=連続3日にならない
        s = d * SPD
        events += [_ev(s, 7, "enter_building", {"building": "bcafe"}),
                   _ev(s, 8, "enter_building", {"building": "bcafe"})]
    res = af.detect_org_candidates(events, agents, {}, min_days=3, min_core=2, max_core=12)
    assert res["n_emergent"] == 0


# --------------------------------------------------------------------------- #
# 5. 対照群マッチング(同職業×同年齢帯・決定論)
# --------------------------------------------------------------------------- #
def test_control_matching_deterministic():
    agents = {
        1: {"occupation": "A", "age": 31},            # founder
        0: {"occupation": "A", "age": 30},            # 同 occ/band=30 の非ファウンダー
        6: {"occupation": "A", "age": 33},
        2: {"occupation": "B", "age": 40},
    }
    founders = {1: {"types": {"venture"}, "ref_step": 300, "detail": {}}}
    m1 = af.match_controls(founders, agents, per_founder=1)
    m2 = af.match_controls(founders, agents, per_founder=1)
    assert m1 == m2                                   # 決定論
    # 選ばれた対照は同職業・同年齢帯で、ファウンダーではない
    (cid, fid), = m1.items()
    assert fid == 1 and cid in (0, 6)
    assert agents[cid]["occupation"] == "A"
    assert af._age_band(agents[cid]["age"]) == af._age_band(agents[1]["age"])
    assert cid != 1


# --------------------------------------------------------------------------- #
# end-to-end: 合成ランを tmp に書いて analyze(既知正解: 起業1・立候補1・ハブ1・共在1組)
# --------------------------------------------------------------------------- #
def _write_run(run_dir: Path, events, agents_list, traits):
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
    (run_dir / "traits.json").write_text(
        json.dumps(traits, ensure_ascii=False), encoding="utf-8")


def _scenario():
    """既知の正解を持つ合成シナリオ。返り値 (events, agents_list, traits)。"""
    agents_list = [
        {"id": 0, "occupation": "A", "age": 30, "work_building": "bwork", "home_building": "bh0"},
        {"id": 1, "occupation": "A", "age": 31, "work_building": "bwork", "home_building": "bh1"},
        {"id": 2, "occupation": "B", "age": 40, "work_building": "bw2", "home_building": "bh2"},
        {"id": 3, "occupation": "C", "age": 25, "work_building": "bw3", "home_building": "bh3"},
        {"id": 4, "occupation": "C", "age": 26, "work_building": "bw4", "home_building": "bh4"},
        {"id": 5, "occupation": "B", "age": 41, "work_building": "bw5", "home_building": "bh5"},
        {"id": 6, "occupation": "A", "age": 33, "work_building": "bwork", "home_building": "bh6"},
        {"id": 7, "occupation": "D", "age": 50, "work_building": "bw7", "home_building": "bh7"},
        {"id": 8, "occupation": "D", "age": 51, "work_building": "bw8", "home_building": "bh8"},
        {"id": 9, "occupation": "E", "age": 60, "work_building": "bw9", "home_building": "bh9"},
    ]
    traits = {str(i): {"fire_weight": 0.1 * i, "drive_threshold": 0.5,
                       "internal_locus": 0.3, "nfc": 0.6, "risk_tolerance": 0.4}
              for i in range(10)}
    events = []
    # 起業型: agent1 が day2 に開業
    events.append(_ev(2 * SPD + 12, 1, "venture_open", {"name": "屋台"}))
    # 政治型: agent2 が day1 に立候補
    events.append(_ev(1 * SPD + 6, 2, "candidacy", {"day": 1, "deposit": 300000}))
    # ハブ型: agent3 が day0 に 4,5,6 と関係(次数3)
    events += [_ev(10, 3, "relation_tier", {"other": 4, "tier": 1}),
               _ev(20, 3, "relation_tier", {"other": 5, "tier": 1}),
               _ev(30, 3, "relation_tier", {"other": 6, "tier": 1})]
    # 形成前履歴の中身(資金・意見・grievance を少し)
    events += [_ev(0, 1, "wage", {"amount": 1000.0, "balance": 5000.0}),
               _ev(SPD + 5, 1, "spend", {"amount": 200.0, "balance": 4800.0, "cat": "food"}),
               _ev(5, 1, "opinion_shift", {"old": 0.0, "new": 0.2, "source": 9}),
               _ev(3, 1, "state_update", {"name": "grievance", "new": 0.3}),
               _ev(4, 2, "wage", {"amount": 800.0, "balance": 3000.0})]
    # 共在: 既存台帳一致(bwork: 0,1,6)と emergent(bcafe: 7,8)を3日連続
    for d in range(3):
        s = d * SPD
        events += [_ev(s, 0, "enter_building", {"building": "bwork"}),
                   _ev(s, 1, "enter_building", {"building": "bwork"}),
                   _ev(s, 6, "enter_building", {"building": "bwork"}),
                   _ev(s, 7, "enter_building", {"building": "bcafe"}),
                   _ev(s, 8, "enter_building", {"building": "bcafe"})]
    return events, agents_list, traits


def test_end_to_end_known_answers(tmp_path):
    run_dir = tmp_path / "syn"
    events, agents_list, traits = _scenario()
    _write_run(run_dir, events, agents_list, traits)
    out_dir = run_dir / "panel"
    res = af.analyze(str(run_dir), str(out_dir), pre_days=2, hub_degree=3)

    assert res["n_venture"] == 1
    assert res["n_political"] == 1
    assert res["n_hub"] == 1
    assert set(res["founders"]) == {1, 2, 3}
    # 対照: 起業(A/30)→0、政治(B/40)→5、ハブ(C/20)→4
    assert res["n_controls"] == 3
    # 組織候補: bcafe(7,8)= emergent 1件、bwork = known
    assert res["org"]["n_emergent"] == 1
    assert res["org"]["candidates"][0]["members"] == [7, 8]
    assert res["org"]["n_known"] >= 1

    # パネルが書かれ、founder=0/1 の両方がある
    assert Path(res["paths"]["panel"]).exists()
    import pyarrow.parquet as pq
    tbl = pq.read_table(res["paths"]["panel"]).to_pydict()
    founders_flags = set(tbl["founder"])
    assert founders_flags == {0, 1}
    # ファウンダー1(day2 開業)は前2日+当日=3行、その day0 に wage 1000 が入る
    f1_rows = [i for i, a in enumerate(tbl["agent_id"])
               if a == 1 and tbl["founder"][i] == 1]
    assert len(f1_rows) == 3
    day0 = [i for i in f1_rows if tbl["day"][i] == 0][0]
    assert tbl["money_income"][day0] == 1000.0
    assert tbl["k_fire_weight"][day0] == 0.1          # traits[1].fire_weight
    # ハブ agent3 の day0 に degree_cum=3
    h_rows = [i for i, a in enumerate(tbl["agent_id"])
              if a == 3 and tbl["founder"][i] == 1]
    assert len(h_rows) == 1
    assert tbl["degree_cum"][h_rows[0]] == 3

    # org_candidates.json が書かれている
    org = json.loads(Path(res["paths"]["org"]).read_text(encoding="utf-8"))
    assert org["run"] == "syn"
    assert org["n_emergent"] == 1


def test_end_to_end_determinism(tmp_path):
    events, agents_list, traits = _scenario()
    # 同一 basename(=同一 run 名)を別の親に置く → panel の run 列も同値でバイト比較できる
    r1 = tmp_path / "p1" / "syn"
    r2 = tmp_path / "p2" / "syn"
    _write_run(r1, events, agents_list, traits)
    _write_run(r2, events, agents_list, traits)
    af.analyze(str(r1), str(r1 / "panel"), pre_days=2, hub_degree=3)
    af.analyze(str(r2), str(r2 / "panel"), pre_days=2, hub_degree=3)
    # 同一入力 → panel parquet はバイト一致 / org json も一致
    b1 = (r1 / "panel" / "founders.parquet").read_bytes()
    b2 = (r2 / "panel" / "founders.parquet").read_bytes()
    assert b1 == b2
    j1 = (r1 / "panel" / "org_candidates.json").read_text(encoding="utf-8")
    j2 = (r2 / "panel" / "org_candidates.json").read_text(encoding="utf-8")
    assert j1 == j2


def test_end_to_end_degrade_missing_events(tmp_path):
    """candidacy / venture / proposal が1件も無い旧ラン: 例外なく縮退(起業0・政治0)。"""
    run_dir = tmp_path / "old"
    agents_list = [{"id": i, "occupation": "A", "age": 30 + i,
                    "work_building": f"bw{i}", "home_building": f"bh{i}"} for i in range(6)]
    traits = {}
    events = [_ev(10, 0, "relation_tier", {"other": 1, "tier": 1}),
              _ev(20, 0, "relation_tier", {"other": 2, "tier": 1}),
              _ev(30, 0, "relation_tier", {"other": 3, "tier": 1}),
              _ev(0, 0, "speak", {"text": "x"})]
    _write_run(run_dir, events, agents_list, traits)
    res = af.analyze(str(run_dir), str(run_dir / "panel"), pre_days=2, hub_degree=3)
    assert res["n_venture"] == 0
    assert res["n_political"] == 0
    assert res["n_hub"] == 1                          # ハブは検出できる
    assert Path(res["paths"]["panel"]).exists()
    assert Path(res["paths"]["org"]).exists()
