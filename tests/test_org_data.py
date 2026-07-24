"""会社観測データ層 B4(work.service 傘下。すべて既定 OFF)のテスト。

R1 の鉄則を継承:
- OFF(既定): serve payload バイト不変・org_ledger.parquet 不在・org_output に basis/day なし。
- ON(indoor_fields): serve に org_id/floor。解決順=スタッフ経由(主)→ unstaffed は node→org 一意時のみ
  → 多義ノードは null=unknown を正直開示(3 通りを合成で直接検証)。
- ON(office.by_org): org_output を org_id 単位に分解。indoor ON でミクロ在席分(attendance_min)・
  indoor OFF で在席頭数(headcount)。basis フィールドで式を自己記述。
- ON(ledger): runs/<run>/org_ledger.parquet を日次1行/社で書く。スキーマ厳守(B7 が読む契約)・
  L1 集計(既存 in-memory org_ledger)との検算一致・resume==straight・同 seed 2 ラン一致。
- LLM 呼数は ON/OFF 完全一致(追加 LLM 呼ゼロ)。
検証は mock / 固定 LLM のみ(実LLM 禁止・≤144 step=1日)。乱数不使用・追加 LLM 呼ゼロ。
"""
from __future__ import annotations

import json

import pyarrow.parquet as pq

from society.config import load_config
from society.engine import checkpoint, scheduler
from society.engine.simulation import Simulation
from society.observer.org_ledger import OrgLedger
from society.observer.schema import Event

_LEDGER_COLS = ["day", "org_id", "production", "revenue_est",
                "wage_paid", "serve_count", "attendance_min"]

_SVC = {"work.service.enabled": "true"}
_FIELDS = {"work.service.enabled": "true", "work.service.indoor_fields": "true"}


def _sim(tmp_path, name, n=25, steps=24, **ov):
    dot = ["run.seed=42", f"run.n_agents={n}", f"run.n_steps={steps}",
           f"run.name={name}", "model.backend=mock"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _orgsim(tmp_path, name, n=30, steps=144, **ov):
    """組織 ON + work.service + ledger の実ラン用(personas_80 で production/wage が出る)。"""
    dot = ["run.seed=42", f"run.n_agents={n}", f"run.n_steps={steps}",
           f"run.name={name}", "model.backend=mock",
           "agents.personas_file=data/personas_80.json",
           "organizations.enabled=true", "work.service.enabled=true",
           "work.service.ledger.enabled=true"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _serves(sim):
    return [e for e in sim.logger.events if e.kind == "serve"]


def _spend(node, x, y, cust_id, step=5, sim_min=700):
    return Event(step=step, sim_min=sim_min, agent_id=cust_id, kind="spend",
                 x=x, y=y, payload={"cat": "food"})


# ============================================================ OFF: バイト不変
def test_off_serve_payload_and_no_ledger(tmp_path):
    """work.service ON でも B4 全 OFF なら serve に org_id/floor が付かず org_ledger も作られない。"""
    sim = _sim(tmp_path, "off_b4", **_SVC)
    node = sim.city.pois_by_cat("food")[0]["node"]
    x, y = sim.city.node_xy(node)
    for a in sim.agents:
        a.work_start_min = -1
    cust, staff = sim.agents[0], sim.agents[1]
    cust.node = node
    cust.x, cust.y = x, y
    staff.node = staff.work_node = node
    staff.x, staff.y = x, y
    staff.work_start_min, staff.work_end_min = 0, 1440
    since = len(sim.logger.events)
    sim.logger.log(_spend(node, x, y, cust.id))
    scheduler._phase_work_service(sim, 5, 700, since)
    s = _serves(sim)
    assert len(s) == 1
    assert "org_id" not in s[0].payload and "floor" not in s[0].payload, \
        "B4 OFF なのに serve に org_id/floor が付いた(バイト不変違反)"
    sim.run()
    assert not (sim.out_dir / "org_ledger.parquet").exists(), \
        "ledger OFF なのに org_ledger.parquet が作られた"
    assert sim.org_ledger_sc is None


# ============================================================ (1) serve org_id/floor
def test_serve_org_id_via_staff(tmp_path):
    """スタッフ経由=帰属スタッフの org_id/floor が主経路で付く。"""
    sim = _sim(tmp_path, "svc_staff", **_FIELDS)
    node = sim.city.pois_by_cat("food")[0]["node"]
    x, y = sim.city.node_xy(node)
    for a in sim.agents:
        a.work_start_min = -1
    cust, staff = sim.agents[0], sim.agents[1]
    cust.node = node
    cust.x, cust.y = x, y
    staff.node = staff.work_node = node
    staff.x, staff.y = x, y
    staff.work_start_min, staff.work_end_min = 0, 1440
    staff.org_id = "ORG_A"
    staff.floor = 3
    since = len(sim.logger.events)
    sim.logger.log(_spend(node, x, y, cust.id))
    scheduler._phase_work_service(sim, 5, 700, since)
    s = _serves(sim)
    assert len(s) == 1 and s[0].agent_id == staff.id
    assert s[0].payload["org_id"] == "ORG_A"
    assert s[0].payload["floor"] == 3


def test_serve_org_id_unstaffed_unique(tmp_path):
    """unstaffed 消費: node→org が一意(1社)のときだけ org_id を付ける。"""
    sim = _sim(tmp_path, "svc_uniq", **_FIELDS)
    node = sim.city.pois_by_cat("food")[0]["node"]
    x, y = sim.city.node_xy(node)
    for a in sim.agents:
        a.work_start_min = -1
        a.work_node = ""
        a.org_id = None
    worker = sim.agents[2]                     # 勤務窓は閉じたまま(=staff にならず unstaffed)
    worker.work_node = node
    worker.org_id = "ORG_U"
    cust = sim.agents[0]
    cust.node = node
    cust.x, cust.y = x, y
    since = len(sim.logger.events)
    sim.logger.log(_spend(node, x, y, cust.id, step=3))
    scheduler._phase_work_service(sim, 3, 700, since)
    s = _serves(sim)
    assert len(s) == 1 and s[0].agent_id == -1
    assert s[0].payload["org_id"] == "ORG_U"
    assert s[0].payload["floor"] == int(getattr(cust, "floor", 0) or 0)


def test_serve_org_id_unstaffed_ambiguous_null(tmp_path):
    """多義ノード(同一 node に複数社)は org_id=null=unknown を正直開示(推測しない)。"""
    sim = _sim(tmp_path, "svc_amb", **_FIELDS)
    node = sim.city.pois_by_cat("food")[0]["node"]
    x, y = sim.city.node_xy(node)
    for a in sim.agents:
        a.work_start_min = -1
        a.work_node = ""
        a.org_id = None
    w1, w2 = sim.agents[2], sim.agents[3]
    w1.work_node = node
    w1.org_id = "ORG_X"
    w2.work_node = node
    w2.org_id = "ORG_Y"
    cust = sim.agents[0]
    cust.node = node
    cust.x, cust.y = x, y
    since = len(sim.logger.events)
    sim.logger.log(_spend(node, x, y, cust.id, step=3))
    scheduler._phase_work_service(sim, 3, 700, since)
    s = _serves(sim)
    assert len(s) == 1
    assert "org_id" in s[0].payload and s[0].payload["org_id"] is None, \
        "多義ノードで org_id が null(unknown)開示になっていない"


def test_serve_floor_gate_requires_same_floor(tmp_path):
    """indoor.enabled かつ indoor_fields ON: 客とスタッフの floor 不一致は応対しない(unstaffed 化)。"""
    sim = _sim(tmp_path, "svc_floor", **_FIELDS)
    sim.indoor = object()                      # _indoor_on → True(floor 絞り込みを有効化)
    node = sim.city.pois_by_cat("food")[0]["node"]
    x, y = sim.city.node_xy(node)
    for a in sim.agents:
        a.work_start_min = -1
    cust, staff = sim.agents[0], sim.agents[1]
    cust.node = node
    cust.x, cust.y = x, y
    cust.building, cust.floor = "B1", 2
    staff.node = staff.work_node = node
    staff.x, staff.y = x, y
    staff.work_start_min, staff.work_end_min = 0, 1440
    staff.building, staff.floor = "B1", 5      # 別階=応対不可
    staff.org_id = "ORG_A"
    since = len(sim.logger.events)
    sim.logger.log(_spend(node, x, y, cust.id))
    scheduler._phase_work_service(sim, 5, 700, since)
    s = _serves(sim)
    assert len(s) == 1 and s[0].agent_id == -1, "別階なのにスタッフに帰属した(floor 絞り込み不発)"
    # 同階なら応対する
    staff.floor = 2
    since = len(sim.logger.events)
    sim.logger.log(_spend(node, x, y, cust.id, step=6, sim_min=710))
    scheduler._phase_work_service(sim, 6, 710, since)
    s2 = [e for e in sim.logger.events if e.kind == "serve" and e.step == 6]
    assert len(s2) == 1 and s2[0].agent_id == staff.id and s2[0].payload["floor"] == 2


# ============================================================ (2) org_output by_org
def test_org_output_by_org_headcount(tmp_path):
    """by_org ON + indoor OFF: org_output を org_id 単位・basis=headcount・output=Σrole重み。"""
    sim = _sim(tmp_path, "byorg_hc",
               **{"work.service.enabled": "true", "work.service.office.by_org": "true"})
    scheduler._org_day_entry(sim, "ORG_A")["workers"].update({0, 1, 2})
    scheduler._org_day_entry(sim, "ORG_B")["workers"].update({3, 4})
    rows = scheduler._emit_org_day(sim, 10, 100, 0)
    outs = [e for e in sim.logger.events if e.kind == "org_output"]
    assert len(outs) == 2 and rows == []       # ledger OFF → 行なし
    by = {e.payload["org"]: e.payload for e in outs}
    assert by["ORG_A"]["n"] == 3 and by["ORG_A"]["basis"] == "headcount"
    assert by["ORG_A"]["output"] == 3.0 and by["ORG_A"]["day"] == 0
    assert by["ORG_B"]["n"] == 2 and by["ORG_B"]["output"] == 2.0


def test_org_output_by_org_attendance(tmp_path):
    """by_org ON + indoor ON: output=ミクロ在席分 attendance_min・basis=attendance_min。"""
    sim = _sim(tmp_path, "byorg_att",
               **{"work.service.enabled": "true", "work.service.office.by_org": "true"})
    sim.indoor = object()                      # _indoor_on → True
    e = scheduler._org_day_entry(sim, "ORG_A")
    e["workers"].update({0, 1})
    e["attendance_min"] = 120
    scheduler._emit_org_day(sim, 10, 100, 0)
    outs = [ev for ev in sim.logger.events if ev.kind == "org_output"]
    assert len(outs) == 1
    assert outs[0].payload["basis"] == "attendance_min"
    assert outs[0].payload["output"] == 120.0 and outs[0].payload["n"] == 2


def test_attendance_accumulate_from_ind_zone(tmp_path):
    """ミクロ在席分は ind_space_type が職務区画(desk/meeting)の step だけ +10分(break は数えない)。"""
    sim = _sim(tmp_path, "att_acc",
               **{"work.service.enabled": "true", "work.service.ledger.enabled": "true"})
    sim.indoor = object()
    node = sim.city.pois_by_cat("office")[0]["node"]
    for a in sim.agents:
        a.work_start_min = -1
        a.work_node = ""
        a.org_id = None
    w = sim.agents[0]
    w.node = w.work_node = node
    w.work_start_min, w.work_end_min = 0, 1440
    w.org_id = "ORG_O"
    w.ind_space_type = "desk"                  # 職務区画 → 在席
    scheduler._phase_org_accumulate(sim, 5, 700)
    ent = sim._org_day["ORG_O"]
    assert ent["attendance_min"] == 10 and 0 in ent["workers"]
    w.ind_space_type = "break"                 # 休憩区画 → 在席に数えない
    scheduler._phase_org_accumulate(sim, 6, 710)
    assert sim._org_day["ORG_O"]["attendance_min"] == 10, "break を在席に数えている"
    w.ind_space_type = "meeting"               # 会議区画 → 在席
    scheduler._phase_org_accumulate(sim, 7, 720)
    assert sim._org_day["ORG_O"]["attendance_min"] == 20


# ============================================================ (3) org_ledger サイドカー
def test_ledger_schema(tmp_path):
    """OrgLedger のスキーマ厳守(会社 UI=B7 が読む契約)。列名・型・値。"""
    sc = OrgLedger(tmp_path)
    sc.add_rows([(0, "ORG_A", 3, 15.0, 10.0, 5, 120),
                 (1, "ORG_B", 0, 0.0, 0.0, 2, 0)])
    p = sc.finalize()
    t = pq.read_table(p)
    assert t.column_names == _LEDGER_COLS
    assert str(t.schema.field("day").type) == "int32"
    assert str(t.schema.field("org_id").type) == "string"
    assert str(t.schema.field("production").type) == "int32"
    assert str(t.schema.field("revenue_est").type) == "double"
    assert str(t.schema.field("wage_paid").type) == "double"
    assert str(t.schema.field("serve_count").type) == "int32"
    assert str(t.schema.field("attendance_min").type) == "int32"
    d = t.to_pylist()
    assert d[0] == {"day": 0, "org_id": "ORG_A", "production": 3,
                    "revenue_est": 15.0, "wage_paid": 10.0,
                    "serve_count": 5, "attendance_min": 120}


def test_ledger_skips_all_zero_orgs(tmp_path):
    """全列 0 の社は 1 行も書かない(活動があった社のみ)。"""
    sim = _sim(tmp_path, "ledger_zero",
               **{"work.service.enabled": "true", "work.service.ledger.enabled": "true"})
    scheduler._org_day_entry(sim, "ORG_ZERO")  # 触れただけ(全列 0)
    ent = scheduler._org_day_entry(sim, "ORG_ACTIVE")
    ent["production"] = 2
    ent["wage_paid"] = 8.0
    ent["serve_count"] = 1
    rows = scheduler._emit_org_day(sim, 10, 100, 3)
    assert rows == [(3, "ORG_ACTIVE", 2, 0.0, 8.0, 1, 0)], \
        f"全 0 の社が書かれた/活動社の行が誤り: {rows}"


def test_ledger_integration_and_l1_crosscheck(tmp_path):
    """実ラン: org_ledger.parquet が書かれ、production/wage/revenue が既存 in-memory 会計と一致。"""
    sim = _orgsim(tmp_path, "ledger_integ")
    sim.run()
    p = sim.out_dir / "org_ledger.parquet"
    assert p.exists(), "ledger ON の実ランで org_ledger.parquet が作られていない"
    rows = pq.read_table(p).to_pylist()
    assert rows, "org_ledger.parquet が空(production も serve も出ていない)"
    assert pq.read_table(p).column_names == _LEDGER_COLS
    # 全日合算 per org == 既存 in-memory org_ledger(同じ _log_org_output が両方を積む=検算)
    prod = {}
    rev = {}
    wage = {}
    for r in rows:
        prod[r["org_id"]] = prod.get(r["org_id"], 0) + r["production"]
        rev[r["org_id"]] = rev.get(r["org_id"], 0.0) + r["revenue_est"]
        wage[r["org_id"]] = wage.get(r["org_id"], 0.0) + r["wage_paid"]
    active = {k: v for k, v in sim.org_ledger.items() if v["production_count"] > 0}
    assert active, "in-memory org_ledger に production が無い(実ランの前提が崩れている)"
    for oid, led in active.items():
        assert prod.get(oid, 0) == led["production_count"], \
            f"production 検算不一致 {oid}: {prod.get(oid)} vs {led['production_count']}"
        assert abs(wage.get(oid, 0.0) - led["wage_paid"]) < 1e-3, "wage_paid 検算不一致"
        assert abs(rev.get(oid, 0.0) - led["revenue_est"]) < 1e-3, "revenue_est 検算不一致"
    # L1 の production イベント総数と ledger の production 総数が一致
    n_prod = sum(1 for e in sim.logger.events if e.kind == "production")
    assert sum(prod.values()) == n_prod, "L1 production 件数と ledger production 総数が不一致"


def test_ledger_same_seed_two_runs_identical(tmp_path):
    """同 seed 2 ランで org_ledger.parquet の行が完全一致(決定論)。"""
    a = _orgsim(tmp_path, "led_det_a")
    a.run()
    b = _orgsim(tmp_path, "led_det_b")
    b.run()
    ra = pq.read_table(a.out_dir / "org_ledger.parquet").to_pylist()
    rb = pq.read_table(b.out_dir / "org_ledger.parquet").to_pylist()
    assert ra and ra == rb, "org_ledger の決定論が崩れている"


def _run_resume_ledger(tmp_path, name, split, total):
    """phase1: split step 手動実行→ckpt+segment→中断。phase2: 新 Simulation で resume→total→finalize。"""
    d = tmp_path / name
    every = {"observer.checkpoint_every": str(split)}
    sim1 = _orgsim(tmp_path, name, steps=split, **every)
    for step in range(split):
        scheduler.run_step(sim1, step)
    checkpoint.save(sim1, split, d / "checkpoint" / f"ckpt-{split:06d}.pkl.gz")
    sim1.logger.flush_segment()
    if sim1.org_ledger_sc is not None:         # B4: サイドカーも logger と同じ点でセグメント化
        sim1.org_ledger_sc.flush_segment()
    sim2 = _orgsim(tmp_path, name, steps=total, **every)
    sim2.run(resume_from=d)
    return d


def test_ledger_resume_matches_straight(tmp_path):
    """一気 144step と 72+resume で org_ledger.parquet が全行一致(resume==straight)。"""
    straight = _orgsim(tmp_path, "led_straight", steps=144)
    straight.run()
    resumed = _run_resume_ledger(tmp_path, "led_resumed", 72, 144)
    a = pq.read_table(straight.out_dir / "org_ledger.parquet").to_pylist()
    b = pq.read_table(resumed / "org_ledger.parquet").to_pylist()
    assert a, "straight ラン org_ledger が空(検証にならない)"
    assert a == b, "resume==straight の org_ledger が不一致"


# ============================================================ (4) agents.json org_id/org_role
def test_agents_json_org_fields(tmp_path):
    """B4 ON(ledger/indoor_fields)+org 配属ありのランは agents.json に org_id/org_role が載る。
    B4 OFF は載らない(org 配属は run 中の遅延初期化のため __init__ 出力には出ない=既存とバイト一致)。"""
    import json as _json
    on = _orgsim(tmp_path, "aj_on", steps=1)
    on.run()
    recs_on = _json.loads((on.out_dir / "agents.json").read_text(encoding="utf-8"))
    withorg = [a for a in recs_on if "org_id" in a]
    assert withorg, "B4 ON なのに agents.json に org_id が1つも無い"
    assert all("org_role" in a for a in withorg)

    off = _sim(tmp_path, "aj_off", n=30, steps=1,
               **{"agents.personas_file": "data/personas_80.json",
                  "organizations.enabled": "true"})   # 組織 ON だが B4 OFF
    off.run()
    recs_off = _json.loads((off.out_dir / "agents.json").read_text(encoding="utf-8"))
    assert not any("org_id" in a for a in recs_off), \
        "B4 OFF なのに agents.json に org_id が載った(既存挙動を変えた)"


# ============================================================ (5) LLM 呼数不変
class _FixedLLM:
    def __init__(self):
        self.response = json.dumps({"action": "speak", "text": "やあ"}, ensure_ascii=False)
        self.calls = 0
        self.hits = 0

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        self.calls += 1
        return self.response, str(self.calls), False


def _run_fixed(tmp_path, name, **ov):
    sim = _sim(tmp_path, name, **{"work.service.enabled": "true",
                                  "prompts.interstitial.enabled": "true", **ov})
    sim.llm = _FixedLLM()
    sim.run()
    return sim


def test_llm_call_count_invariant(tmp_path):
    """B4 全 ON/OFF で LLM 呼数が完全一致(追加 LLM 呼ゼロ=R1)。"""
    on = _run_fixed(tmp_path, "cc_on",
                    **{"work.service.indoor_fields": "true",
                       "work.service.office.by_org": "true",
                       "work.service.ledger.enabled": "true"})
    off = _run_fixed(tmp_path, "cc_off")
    assert on.llm.calls == off.llm.calls and on.llm.calls > 0, \
        f"B4 で LLM 呼数が変化(R1 違反): on={on.llm.calls} off={off.llm.calls}"
