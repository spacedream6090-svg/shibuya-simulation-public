"""世帯・家族・恋愛(現実ギャップ 後続波 H2 2026-07-07)= 世帯グループ化/家族文脈/パートナー のテスト。

方針(既存の鉄則を継承):
- OFF(既定): partner_formed/life_event が 0 件・誰も世帯に属さない・新 stream "household"/"date" も
  引かない・イベント列は純粋既定と L1 完全一致(144 step。ゴールデン golden_baseline_l1.json を守る)。
- 世帯 ON: 居住者が決定論で世帯にまとまり、同一世帯で home(建物・ノード・階)を共有し、同居者・恋人が
  同席したとき「同居する家族/同居人/恋人の○○」がプロンプト文脈に載る(単体)。
- パートナー ON: G2 relations の親密度 closeness が相互に閾値超の2者が partner_formed になり
  life_event(kind="partner")が出る(単体・決定論)。
- 決定論: ON 同士2回で L1 完全一致。
- R1: 住居共有・デートは物理位置を変え対面 co-location が変化しうる(career G5 / crowd G4 / 健康 H1 と同型)。
  よって呼数不変は compute_matched 下の k 不変性(k=free と k=off の generate 呼数完全一致)で担保する。
検証は mock のみ(実LLM 禁止)。
"""
from __future__ import annotations

import json

from society import household
from society.config import load_config
from society.engine.simulation import Simulation

_HH = {"household.enabled": "true"}
# 恋愛は G2 closeness を読む(要 relations.enabled)。パートナー成立を効かせた ON 設定。
_ON = {**_HH, "relations.enabled": "true",
       "household.partner_closeness": "2.0", "household.date_bias": "0.8"}


def _sim(tmp_path, name, n=30, steps=1, **ov):
    dot = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
           "observer.snapshot_every=1"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _kind(sim, kind):
    return [e for e in sim.logger.events if e.kind == kind]


# --------------------------------------------------------------------- OFF 既定
def test_off_matches_pure_default(tmp_path):
    """明示 OFF と純粋既定が L1 完全一致(144 step)。H2 の新イベントは 1 件も出ず誰も世帯に属さない。"""
    pure = _sim(tmp_path, "pure", steps=144)
    pure.run()
    off = _sim(tmp_path, "expl_off", steps=144, **{"household.enabled": "false"})
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(H2 seam が no-op でない)"
    for k in ("partner_formed", "life_event"):
        assert not _kind(pure, k), f"OFF で {k} が出ている"
    assert all(a.household_id is None and a.partner_id is None for a in pure.agents), \
        "OFF なのに世帯/パートナーが付いている"


# --------------------------------------------------------------------- 世帯グループ化
def test_households_share_home_and_group(tmp_path):
    """世帯 ON: 居住者が2人以上の世帯にまとまり、同一世帯で home を共有し housemates が整合する。"""
    sim = _sim(tmp_path, "hh", n=30, **_HH)
    grouped = [a for a in sim.agents if a.household_id is not None]
    assert grouped, "世帯 ON なのに誰も世帯に属していない"
    by_hh: dict = {}
    for a in grouped:
        by_hh.setdefault(a.household_id, []).append(a)
    assert any(len(m) >= 2 for m in by_hh.values()), "2人以上の世帯が1つも無い"
    for _hh, members in by_hh.items():
        assert len(members) >= 2, "単身に household_id が付いている(世帯を組んでいない)"
        rep_home = (members[0].home_building, members[0].home_node, members[0].home_floor)
        ids = sorted(m.id for m in members)
        for m in members:
            assert (m.home_building, m.home_node, m.home_floor) == rep_home, \
                "同一世帯で home が共有されていない(夜間 co-location が生まれない)"
            assert sorted(m.housemates + [m.id]) == ids, "housemates が世帯メンバと一致しない"
            assert m.household_kind in ("family", "roommate"), "世帯種別が family/roommate でない"
        assert not any(m.visitor for m in members), "来街者が世帯に入っている(居住者のみのはず)"


def test_family_context_line(tmp_path):
    """同居者・恋人が同席したとき context_line が「同居する家族/同居人/恋人の○○」を返す。"""
    sim = _sim(tmp_path, "ctx", n=30, **_HH)
    a = next(x for x in sim.agents if x.household_id is not None)
    mate_id = a.housemates[0]
    mate_name = sim.agent_by_id[mate_id].name
    label = "同居する家族" if a.household_kind == "family" else "同居人"
    assert household.context_line(a, [mate_id], sim.agent_by_id) == f"{label}の{mate_name}"
    # 同席者に居ない=1行も足さない(None)
    absent = next(o for o in sim.agent_by_id if o != a.id and o not in a.housemates)
    assert household.context_line(a, [absent], sim.agent_by_id) is None, \
        "非同居者に家族文脈が付いている"
    # 恋人は「恋人の○○」が優先
    a.partner_id = mate_id
    assert household.context_line(a, [mate_id], sim.agent_by_id) == f"恋人の{mate_name}"


# --------------------------------------------------------------------- パートナー形成
def test_partner_forms_from_mutual_closeness(tmp_path):
    """恋愛 ON: 相互 closeness が閾値超の2者を決定論でパートナーにし partner_formed/life_event を出す。"""
    sim = _sim(tmp_path, "partner", n=30,
               **{**_HH, "household.partner_closeness": "5.0"})
    a, b = sim.agents[0], sim.agents[1]
    a.visitor = b.visitor = False
    a.partner_id = b.partner_id = None
    a.mem.relations[b.id] = {"name": b.name, "count": 9, "last_step": 0,
                             "last": "", "closeness": 6.0}
    b.mem.relations[a.id] = {"name": a.name, "count": 9, "last_step": 0,
                             "last": "", "closeness": 6.0}
    household.form_partners(sim, step=0, sim_min=0)
    assert a.partner_id == b.id and b.partner_id == a.id, \
        "相互高 closeness でパートナーが成立していない"
    pf = _kind(sim, "partner_formed")
    le = [e for e in _kind(sim, "life_event") if e.payload["kind"] == "partner"]
    assert pf and pf[0].payload["other"] == b.id, "partner_formed が正しく出ていない"
    assert le, "life_event(kind=partner)が出ていない"


def test_partner_needs_mutual_high_closeness(tmp_path):
    """片側だけ高 closeness ではパートナーは成立しない(相互性の要求)。"""
    sim = _sim(tmp_path, "one_side", n=30,
               **{**_HH, "household.partner_closeness": "5.0"})
    a, b = sim.agents[0], sim.agents[1]
    a.partner_id = b.partner_id = None
    a.mem.relations[b.id] = {"name": b.name, "count": 9, "last_step": 0,
                             "last": "", "closeness": 9.0}
    b.mem.relations[a.id] = {"name": a.name, "count": 1, "last_step": 0,
                             "last": "", "closeness": 1.0}   # 片側は閾値未満
    household.form_partners(sim, step=0, sim_min=0)
    assert a.partner_id is None and b.partner_id is None, "片側だけでパートナーが成立した"
    assert not _kind(sim, "partner_formed"), "非成立なのに partner_formed が出ている"


def test_partner_date_shared_destination(tmp_path):
    """デート: パートナー2者が同一の共有目的地を選ぶ(合流=co-location)。date_bias=1 で必ず選択。"""
    sim = _sim(tmp_path, "date", n=30,
               **{**_HH, "household.date_bias": "1.0"})
    a, b = sim.agents[0], sim.agents[1]
    a.partner_id, b.partner_id = b.id, a.id
    a.visitor = b.visitor = False
    da = household.date_dest(a, sim, step=5, sim_min=500)
    db = household.date_dest(b, sim, step=5, sim_min=500)
    assert da is not None and db is not None, "date_bias=1 なのにデート先が選ばれない"
    assert da == db, "パートナー2者のデート先が一致しない(合流できない)"


# --------------------------------------------------------------------- 決定論
def test_all_on_deterministic(tmp_path):
    """世帯+恋愛 ON 同士 2 回で L1 完全一致(決定論・mock 144 step)。"""
    a = _sim(tmp_path, "det_a", n=30, steps=144, **_ON)
    a.run()
    b = _sim(tmp_path, "det_b", n=30, steps=144, **_ON)
    b.run()
    assert _l1(a) == _l1(b), "ON の決定論が崩れている"


# --------------------------------------------------------------------- R1 k 不変性
class _FixedLLM:
    """挙動を固定する backend(応答をプロンプトに依存させない)。呼数だけ数える。"""

    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.hits = 0

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        self.calls += 1
        return self.response, str(self.calls), False


def _run_k(tmp_path, name, *, writeback):
    """世帯+恋愛 ON を compute_matched(k 掃引で使う対照)下で回し generate 呼数を数える。"""
    sim = _sim(tmp_path, name, n=30, steps=144,
               **{**_ON, "controls.mode": "compute_matched", "k.writeback": writeback})
    sim.llm = _FixedLLM(json.dumps({"action": "speak", "text": "x"},
                                   ensure_ascii=False))
    sim.run()
    return sim


def test_household_call_count_k_invariant(tmp_path):
    """住居共有・デートは対面 co-location を変え ON!=OFF になりうる(実在する紐帯の必然)。だが R1 の本旨=
    「呼数が k(writeback)に依存しない」ことは compute_matched 下で厳密に保たれる。H2 の機構は名簿+物理位置+
    config+relations closeness(観測イベント由来=k 非依存の観測量)+新 stream のみ参照し、発火判断に k・内面
    状態(構成概念)を食わせないため、k=free と k=off で generate 呼数が完全一致する(career G5 / 健康 H1 と同型)。"""
    free = _run_k(tmp_path, "hk_free", writeback="free")
    off = _run_k(tmp_path, "hk_off", writeback="off")
    assert free.llm.calls == off.llm.calls and free.llm.calls > 0, \
        f"household の呼数が k に依存(R1 違反): free={free.llm.calls} off={off.llm.calls}"
    assert _kind(free, "partner_formed"), \
        "パートナー成立日なのに partner_formed が出ていない(機構が不発)"
