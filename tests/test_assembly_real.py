"""議会選挙の現実化(渋谷区議会準拠・batch37 2026-07-19)= 告示→立候補→SNTV→予算/条例/署名。

設計根拠: docs/research/council-vs-reality.md。新設定 institution_routes.assembly.realism(既定 OFF)。
方針(既存の鉄則を継承):
- OFF(realism 既定): elect_assembly の挙動は現行(意見最近傍選挙=council_elected)と完全同一。
  新イベント(candidacy/election_result/council_budget/ordinance_vote)は 1 件も出ない。
- ON: 立候補の資格ゲート(年齢/供託金/来街者)/ SNTV 開票の決定論数値(手計算一致)/
  法定得票の供託金没収境界 / 候補0のフォールバック / 予算否決の係数適用 / 署名閾値・条例議決。
- 決定論・非LLM・新規乱数 stream なし。検証は mock のみ(実LLM 禁止)。
"""
from __future__ import annotations

import json
import math

from society.config import load_config
from society.engine.simulation import Simulation
from society.tools import (_council_budget, _is_eligible_candidate, _is_voter,
                           _run_sntv, routes_of)

_REAL = "institution_routes.assembly.realism"


def _sim(tmp_path, name, n=12, steps=1, *, gov=False, **ov):
    dot = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
           "observer.snapshot_every=1"]
    if gov:
        dot.append("government.enabled=true")
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _kind(sim, kind):
    return [e for e in sim.logger.events if e.kind == kind]


_NEW_KINDS = ("candidacy", "election_result", "council_budget", "ordinance_vote")


# --------------------------------------------------------------------- OFF 不変
def test_realism_off_matches_plain_assembly(tmp_path):
    """realism を明示 OFF にした議会と、realism キー無しの素の議会が L1 完全一致(現行挙動不変)。"""
    plain = _sim(tmp_path, "asm_plain", n=15, steps=40,
                 **{"institution_routes.assembly.enabled": "true",
                    "institution_routes.assembly.size": "5"})
    plain.run()
    off = _sim(tmp_path, "asm_off", n=15, steps=40,
               **{"institution_routes.assembly.enabled": "true",
                  "institution_routes.assembly.size": "5",
                  f"{_REAL}.enabled": "false"})
    off.run()
    assert _l1(plain) == _l1(off), "realism OFF が素の議会と不一致(seam が no-op でない)"
    assert _kind(plain, "council_elected"), "OFF で従来の council_elected が出ていない"
    for k in _NEW_KINDS:
        assert not _kind(plain, k), f"realism OFF で {k} が出ている"


# --------------------------------------------------------------------- 立候補の資格ゲート
def test_candidacy_eligibility_gate(tmp_path):
    """被選挙権(25歳)・供託金(所持金+口座)・非来街者の資格ゲート。合格者のみ納付→candidacy。"""
    sim = _sim(tmp_path, "elig", n=8,
               **{"institution_routes.assembly.enabled": "true",
                  f"{_REAL}.enabled": "true",
                  f"{_REAL}.deposit": "1000"})
    sim.run()
    realism = routes_of(sim)["assembly"]["realism"]
    a = sim.agents[0]
    a.visitor, a.age, a.money, a.account = False, 30, 5000.0, 0.0
    assert _is_eligible_candidate(a, realism)
    a.age = 24
    assert not _is_eligible_candidate(a, realism), "24歳が資格を通っている(被選挙権25歳)"
    a.age, a.money, a.account = 30, 500.0, 200.0            # 700 < 1000
    assert not _is_eligible_candidate(a, realism), "供託金不足が資格を通っている"
    a.account = 600.0                                       # 500+600 = 1100 >= 1000(口座も算入)
    assert _is_eligible_candidate(a, realism)
    a.visitor = True
    assert not _is_eligible_candidate(a, realism), "来街者が資格を通っている"

    # 告示期間を開いて受理: 合格者は供託金納付(所持金減)+candidacy 記録
    sim.council_campaign = {"term_end": 999, "open": True, "candidates": {}}
    a.visitor, a.age, a.money, a.account = False, 30, 5000.0, 0.0
    sim.tools.declare_candidacy(sim, a, step=0, sim_min=0)
    assert a.id in sim.council_campaign["candidates"], "合格者が候補に登録されていない"
    assert a.money == 4000.0, "供託金1000が納付されていない"
    cand = _kind(sim, "candidacy")
    assert cand and cand[-1].agent_id == a.id and cand[-1].payload["deposit"] == 1000.0

    # 資格外(24歳)は静かに no-op(供託金も減らない・候補にも入らない)
    b = sim.agents[1]
    b.visitor, b.age, b.money, b.account = False, 24, 5000.0, 0.0
    sim.tools.declare_candidacy(sim, b, step=1, sim_min=1)
    assert b.id not in sim.council_campaign["candidates"]
    assert b.money == 5000.0, "資格外なのに供託金が引かれた"


# --------------------------------------------------------------------- SNTV 開票(手計算一致)
def _make_voters(sim, opinions):
    """先頭 len(opinions) 人だけを住民(30歳)有権者にし、残りは来街者で除外。closeness/評判は 0。"""
    for a in sim.agents:
        a.visitor = True
    vs = sim.agents[:len(opinions)]
    for a, op in zip(vs, opinions):
        a.visitor, a.age, a.opinion = False, 30, op
        a.mem.relations.clear()
        a._reputation = 0.0
    return vs


def test_sntv_counting_deterministic(tmp_path):
    """3候補・5有権者の SNTV を手計算と突合(意見距離・評判・親密度の3項すべてを検証)。"""
    sim = _sim(tmp_path, "sntv", n=10,
               **{"institution_routes.assembly.enabled": "true",
                  f"{_REAL}.enabled": "true",
                  "institution_routes.assembly.size": "2",
                  f"{_REAL}.deposit": "1000"})
    sim.run()
    asm = routes_of(sim)["assembly"]
    # C0 op-0.6 / C1 op0.0 / C2 op0.6, 有権者 V3 op0.5 / V4 op-0.55(候補も投票する=5有権者)
    v = _make_voters(sim, [-0.6, 0.0, 0.6, 0.5, -0.55])
    c = {i: v[i].id for i in range(3)}

    def run_once():
        sim.council = None
        sim.council_campaign = {"term_end": 0, "open": True,
                                "candidates": {c[0]: 1000.0, c[1]: 1000.0, c[2]: 1000.0}}
        _run_sntv(sim, step=0, sim_min=0, asm=asm, day=0)
        return _kind(sim, "election_result")[-1].payload

    # (a) 意見のみ: 各自の最近傍。C0<-{id0,id4}=2, C1<-{id1}=1, C2<-{id2,id3}=2
    p = run_once()
    assert p["votes"] == {c[0]: 2, c[1]: 1, c[2]: 2}, "意見距離の開票が手計算と不一致"
    assert p["elected"] == [c[0], c[2]], "同点(2票)は id 昇順で当選していない"
    assert p["forfeited"] == [], "全候補が法定得票以上のはず(没収なし)"

    # (b) 評判項: C1 に評判1.5(w_rep0.3=+0.45)。V3(op0.5)の票が C2(-0.1)→C1(-0.05)へ動く
    v[1]._reputation = 1.5
    p = run_once()
    assert p["votes"] == {c[0]: 2, c[1]: 2, c[2]: 1}, "評判項が票を動かしていない"

    # (c) 親密度項: 評判を戻し、V3→C1 の closeness を 1.0(w_cl0.5=+0.5)にすると同じく票が動く
    v[1]._reputation = 0.0
    v[3].mem.relations[c[1]] = {"name": "c1", "count": 1, "last_step": 0,
                                "last": "", "closeness": 1.0}
    p = run_once()
    assert p["votes"] == {c[0]: 2, c[1]: 2, c[2]: 1}, "親密度項が票を動かしていない"


# --------------------------------------------------------------------- 供託金没収の境界
def test_deposit_forfeit_boundary_and_ward_income(tmp_path):
    """法定得票(有効投票÷定数÷10)未満のみ没収。境界(=法定得票ちょうど)は没収しない。没収は区歳入へ。"""
    sim = _sim(tmp_path, "forfeit", n=14, gov=True,
               **{"institution_routes.assembly.enabled": "true",
                  f"{_REAL}.enabled": "true",
                  "institution_routes.assembly.size": "1",
                  f"{_REAL}.legal_divisor": "10",
                  f"{_REAL}.deposit": "1000"})
    sim.run()
    from society.engine import scheduler
    gov = scheduler._gov(sim)
    asm = routes_of(sim)["assembly"]

    for a in sim.agents:
        a.visitor = True
    voters = sim.agents[:10]
    for i, a in enumerate(voters):                         # 10 有権者を配置(得票を C1=7/C2=2/C3=1/C4=0 に)
        a.visitor, a.age = False, 30
        a.mem.relations.clear()
        a.opinion = -0.8 if i < 7 else (-0.35 if i < 9 else 0.35)
    cands = sim.agents[10:14]                              # 候補は来街者にして投票から除外(得票を制御)
    for a, op in zip(cands, [-0.9, -0.3, 0.3, 0.9]):
        a.visitor, a.age, a.opinion, a.money = True, 30, op, 0.0
        a._reputation = 0.0
    cid = [a.id for a in cands]
    ward0 = gov.balance["ward"]

    sim.council = None
    sim.council_campaign = {"term_end": 0, "open": True,
                            "candidates": {c: 1000.0 for c in cid}}
    _run_sntv(sim, step=0, sim_min=0, asm=asm, day=0)
    p = _kind(sim, "election_result")[-1].payload
    assert p["votes"] == {cid[0]: 7, cid[1]: 2, cid[2]: 1, cid[3]: 0}, "得票分布が手計算と不一致"
    assert p["elected"] == [cid[0]], "最多得票が当選していない(size=1)"
    # 有効投票10 / 定数1 / 10 = 法定得票 1.0。C4(0票)のみ没収、C3(1票=境界)は没収しない
    assert p["forfeited"] == [cid[3]], "法定得票の没収境界が誤り(0票のみ没収のはず)"
    assert cands[2].money == 1000.0, "境界(1票)候補に供託金が返還されていない"
    assert cands[3].money == 0.0, "没収候補に供託金が返っている"
    assert abs((gov.balance["ward"] - ward0) - 1000.0) < 1e-6, "没収供託金が区歳入になっていない"
    dep = _kind(sim, "deposit")
    assert any(e.agent_id == cid[3] and e.payload.get("phase") == "forfeit" for e in dep)
    assert any(e.agent_id == cid[2] and e.payload.get("phase") == "refund" for e in dep)


# --------------------------------------------------------------------- 候補0のフォールバック
def test_fallback_when_no_candidates(tmp_path):
    """立候補が1人も無い改選は現行の全住民方式へフォールバック(election_result に fallback:true)。"""
    sim = _sim(tmp_path, "fallback", n=12,
               **{"institution_routes.assembly.enabled": "true",
                  f"{_REAL}.enabled": "true",
                  "institution_routes.assembly.size": "3"})
    sim.run()                                              # day0 の初回改選=候補ゼロ→フォールバック
    er = _kind(sim, "election_result")
    assert er and er[0].payload.get("fallback") is True, "候補0でフォールバックしていない"
    assert sim.council and sim.council["members"], "フォールバックで議会が座っていない"
    assert len(sim.council["members"]) <= 3
    assert not _kind(sim, "candidacy"), "誰も立候補していないのに candidacy が出ている"


# --------------------------------------------------------------------- 予算否決 → 係数適用
def test_budget_rejection_applies_cut_ratio(tmp_path):
    """議会が区予算を否決(過半 no)すると government の区歳出執行が cut_ratio(0.8)に絞られる。"""
    sim = _sim(tmp_path, "budget", n=10, gov=True,
               **{"institution_routes.assembly.enabled": "true",
                  f"{_REAL}.enabled": "true",
                  "institution_routes.assembly.size": "3",
                  f"{_REAL}.cut_ratio": "0.8"})
    sim.run()
    from society.engine import scheduler
    gov = scheduler._gov(sim)
    asm = routes_of(sim)["assembly"]
    members = sim.council["members"]
    assert members, "改選で議会が座っていない(前提破綻)"

    # 全議員を高不満 → 予算否決 → exec_ratio=cut_ratio
    for aid in members:
        sim.agent_by_id[aid].states["grievance"] = 0.9
    sim._council_budget_day = -1
    _council_budget(sim, step=5, sim_min=1440 * 2, asm=asm, cur=sim.council)
    cb = _kind(sim, "council_budget")[-1].payload
    assert cb["approved"] is False, "高不満の議会が予算を承認している"
    assert gov.exec_ratio == 0.8, "否決で exec_ratio が cut_ratio になっていない"
    bal0 = gov.balance["ward"]
    gov.expense("ward", 1000.0)
    assert abs((bal0 - gov.balance["ward"]) - 800.0) < 1e-6, "区歳出が0.8倍で執行されていない"

    # 低不満に戻すと承認 → exec_ratio=1.0(区歳出は満額)
    for aid in members:
        sim.agent_by_id[aid].states["grievance"] = 0.0
    sim._council_budget_day = -1
    _council_budget(sim, step=6, sim_min=1440 * 2, asm=asm, cur=sim.council)
    assert _kind(sim, "council_budget")[-1].payload["approved"] is True
    assert gov.exec_ratio == 1.0, "承認で exec_ratio が 1.0 に戻っていない"
    bal1 = gov.balance["ward"]
    gov.expense("ward", 1000.0)
    assert abs((bal1 - gov.balance["ward"]) - 1000.0) < 1e-6, "承認時に区歳出が満額でない"


# --------------------------------------------------------------------- 署名モデル + 条例議決
def test_signature_threshold_and_ordinance_vote(tmp_path):
    """(c) 住民提案の閾値=ceil(有権者/signature_divisor)。(b) 議会採決は ordinance_vote を追加記録。"""
    sim = _sim(tmp_path, "sig", n=10,
               **{"institution_routes.assembly.enabled": "true",
                  f"{_REAL}.enabled": "true",
                  f"{_REAL}.signature_divisor": "3",
                  "institution_routes.vote.enabled": "true",
                  "institution_routes.vote.grievance_threshold": "0.3"})
    sim.run()
    realism = routes_of(sim)["assembly"]["realism"]
    voters = sum(1 for a in sim.agents if _is_voter(a, realism))
    assert sim.tools._signature_threshold(sim, realism) == math.ceil(voters / 3), \
        "署名閾値が ceil(有権者/50→/3) になっていない"

    # 議会採決 → ordinance_vote(vote_result は不変=追加専用)
    m0, m1 = sim.agents[0], sim.agents[1]
    sim.council = {"members": [m0.id, m1.id], "elected_day": 0, "term": 1}
    for m in (m0, m1):
        m.states["grievance"], m.opinion = 0.9, 0.0        # 高不満 → 両者 yes → 可決
    pr = {"id": 99, "author": m0.id, "text": "夜間の騒音を規制する",
          "supporters": set(), "passed": False, "rule": None}
    sim.tools._resolve_vote(sim, pr, step=5, sim_min=60, vcfg=routes_of(sim)["vote"])
    ov = _kind(sim, "ordinance_vote")
    assert ov and ov[-1].payload["proposal_id"] == 99, "ordinance_vote が出ていない"
    assert ov[-1].payload["passed"] is True and ov[-1].payload.get("by") == "council"
    vr = _kind(sim, "vote_result")
    assert vr and vr[-1].payload["proposal_id"] == 99, "既存 vote_result が消えている(追加専用でない)"


# --------------------------------------------------------------------- 決定論(ON 2回一致)
def test_realism_on_deterministic(tmp_path):
    """realism ON(+government)同士 2 回で L1 完全一致(新規乱数 stream を引かない=決定論)。"""
    def run(name):
        sim = _sim(tmp_path, name, n=20, steps=144, gov=True,
                   **{"institution_routes.assembly.enabled": "true",
                      f"{_REAL}.enabled": "true",
                      "institution_routes.assembly.size": "5",
                      f"{_REAL}.deposit": "1000",
                      "institution_routes.vote.enabled": "true"})
        sim.run()
        return _l1(sim)
    assert run("det_a") == run("det_b"), "realism ON で非決定論(イベント列が一致しない)"


# ===================================================================== 名簿制議会(from_roster)
# ユーザー決定 2026-07-19: 本番シミュ内で選挙は行わない。あらかじめ persona で議員と決まっている
# 住民が議員として生活する。realism 実装は資産として温存し、名簿制(from_roster)を別途足す。
def _seat_roster(sim, occ_by_index):
    """全 agent を住民(非来街者)にし、指定 index を議員 occupation・他を非議員にする合成名簿。"""
    for a in sim.agents:
        a.visitor = False
        a.occupation = "会社員"
    for idx, occ in occ_by_index.items():
        sim.agents[idx].occupation = occ


def test_from_roster_cfg_default_false():
    """from_roster は正準化で既定 False(=現行の選挙挙動を触らない=ゴールデン維持)。"""
    from society.tools import build_routes_cfg
    assert build_routes_cfg(None)["assembly"]["from_roster"] is False


def test_from_roster_seats_councilors_without_election(tmp_path):
    """from_roster ON + 議員2人の合成名簿 → 選挙なしで2人が id 昇順で議会入り。
    council_elected の payload に from_roster:true。改選しない(再呼び出しで議会不変)。"""
    from society.tools import elect_assembly
    sim = _sim(tmp_path, "roster_on", n=8,
               **{"institution_routes.assembly.enabled": "true",
                  "institution_routes.assembly.from_roster": "true",
                  "institution_routes.assembly.size": "9"})
    sim.run()                                              # day0=名簿に議員0人→フォールバック選挙
    _seat_roster(sim, {0: "議員", 1: "区議会議員"})        # 同義語(区議会議員)も議員として拾う
    want = sorted([sim.agents[0].id, sim.agents[1].id])
    sim.council = None
    n_before = len(_kind(sim, "council_elected"))
    elect_assembly(sim, step=0, sim_min=0)
    ce = _kind(sim, "council_elected")
    assert len(ce) == n_before + 1, "名簿制で council_elected が出ていない"
    assert ce[-1].payload["from_roster"] is True, "payload に from_roster:true が無い"
    assert ce[-1].payload["members"] == want, "議員2人が id 昇順で議会入りしていない"
    assert sim.council["members"] == want
    assert sim.council.get("from_roster") is True
    assert not _kind(sim, "candidacy"), "名簿制なのに立候補(選挙)が発生している"
    # 改選しない: term_days を無視し、後日呼んでも council_elected は増えない(任期継続)
    elect_assembly(sim, step=200 * 144, sim_min=200 * 1440)
    assert len(_kind(sim, "council_elected")) == n_before + 1, "名簿制で改選が起きた(任期継続でない)"
    assert sim.council["members"] == want


def test_from_roster_truncates_to_size(tmp_path):
    """議員が size を超えたら id 昇順で size 人に切り捨て(補充選挙はしない)。"""
    from society.tools import elect_assembly
    sim = _sim(tmp_path, "roster_trunc", n=8,
               **{"institution_routes.assembly.enabled": "true",
                  "institution_routes.assembly.from_roster": "true",
                  "institution_routes.assembly.size": "2"})
    sim.run()
    _seat_roster(sim, {0: "議員", 1: "議員", 2: "議員"})   # 議員3人・定数2
    ids = sorted([sim.agents[0].id, sim.agents[1].id, sim.agents[2].id])
    sim.council = None
    elect_assembly(sim, step=0, sim_min=0)
    assert sim.council["members"] == ids[:2], "size 超過分が id 昇順で切り捨てられていない"
    assert sim.council.get("from_roster") is True


def test_from_roster_falls_back_when_no_councilors(tmp_path, capsys):
    """名簿に議員0人 → 警告 print のうえ従来の選挙へフォールバック(council が座る・candidacy は出ない)。"""
    sim = _sim(tmp_path, "roster_none", n=8,
               **{"institution_routes.assembly.enabled": "true",
                  "institution_routes.assembly.from_roster": "true",
                  "institution_routes.assembly.size": "3"})
    sim.run()                                             # day0=議員0人→警告+フォールバック選挙
    out = capsys.readouterr().out
    assert "from_roster" in out and "フォールバック" in out, "0人時の警告 print が出ていない"
    assert sim.council and sim.council["members"], "フォールバック選挙で議会が座っていない"
    assert sim.council.get("from_roster") is not True, "0人なのに名簿制で座っている(縮退が不正直)"
    assert _kind(sim, "council_elected"), "フォールバックで従来の council_elected が出ていない"
    assert not _kind(sim, "candidacy"), "選挙 realism OFF なので candidacy は出ないはず"


def test_from_roster_off_matches_plain_assembly(tmp_path):
    """from_roster 既定/明示 OFF は素の議会と L1 完全一致(seam が no-op=既存挙動不変)。"""
    plain = _sim(tmp_path, "roster_off_plain", n=12, steps=40,
                 **{"institution_routes.assembly.enabled": "true",
                    "institution_routes.assembly.size": "5"})
    plain.run()
    off = _sim(tmp_path, "roster_off_explicit", n=12, steps=40,
               **{"institution_routes.assembly.enabled": "true",
                  "institution_routes.assembly.size": "5",
                  "institution_routes.assembly.from_roster": "false"})
    off.run()
    assert _l1(plain) == _l1(off), "from_roster 明示 OFF が素の議会と不一致(seam が no-op でない)"
    for k in ("candidacy", "election_result"):
        assert not _kind(plain, k), f"from_roster OFF 素の議会で {k} が出ている"
