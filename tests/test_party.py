"""来街者 party の実体化(関係性パッケージ完結 第45バッチ S-R5)= 連れのグループ化テスト。

方針(既存の鉄則を継承):
- OFF(既定): joint_activity(type=party)が 0 件・誰も party_today を持たない・新 stream "party" も
  引かない・イベント列は純粋既定と L1 完全一致(144 step。ゴールデン golden_baseline_l1.json を守る)。
- ON: party_size>=2 の来街者が連れとグループ化=同一初期ノード・相互 relations・共有回遊 POI。
  観測は joint_activity{type:"party"}(既存 kind 再利用)。residents は対象外(joint/household 側)。
- presence 整合: presence 純関数の draw 順に触れない(present 実体化の後にグループ化するだけ)。
検証は mock のみ(実LLM 禁止)。
"""
from __future__ import annotations

import json

from society import party
from society.config import load_config
from society.engine.simulation import Simulation

_ON = {"party.enabled": "true", "party.roam_bias": "1.0"}


def _sim(tmp_path, name, n=20, steps=1, **ov):
    dot = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
           "observer.snapshot_every=144"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _kind(sim, kind):
    return [e for e in sim.logger.events if e.kind == kind]


def _make_party_visitors(sim, ids, party_size):
    """指定 id の agent を来街者にし party_size を付す(異なる初期ノードに置く)。"""
    for k, i in enumerate(ids):
        a = sim.agent_by_id[i]
        a.visitor = True
        a.party_size = party_size
        a.node = sim.dests[k % len(sim.dests)]     # わざと別ノードに置いて寄せを検証


# --------------------------------------------------------------------- OFF 既定
def test_off_matches_pure_default(tmp_path):
    """明示 OFF と純粋既定が L1 完全一致(144 step)。party の joint_activity は出ず party_today も付かない。"""
    pure = _sim(tmp_path, "pure", steps=144)
    pure.run()
    off = _sim(tmp_path, "expl_off", steps=144, **{"party.enabled": "false"})
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(party seam が no-op でない)"
    party_ev = [e for e in _kind(pure, "joint_activity")
                if e.payload.get("type") == "party"]
    assert not party_ev, "OFF で party の joint_activity が出ている"
    assert all(getattr(a, "party_today", None) is None for a in pure.agents), \
        "OFF なのに party_today が付いている"


# --------------------------------------------------------------------- 実体化(同時入街+同一初期ノード)
def test_party_members_colocate_same_node(tmp_path):
    """party ON: party_size>=2 の来街者が連れと同一初期ノードに寄り、相互 relations が注入され、
    joint_activity{type:party} が1件記録される(連れは同時入街=同一日境界で実体化済み)。"""
    sim = _sim(tmp_path, "colo", n=20, **_ON)
    _make_party_visitors(sim, [0, 1, 2], party_size=3)
    leader = sim.agent_by_id[0]
    leader_node = leader.node
    party.form_parties(sim, step=0, sim_min=0)
    # 同一初期ノードへ寄る(代表=最小 id のノードを共有)
    for i in (0, 1, 2):
        assert sim.agent_by_id[i].node == leader_node, \
            f"連れ {i} が代表と同一初期ノードに居ない(同席していない)"
        assert sim.agent_by_id[i].party_today is not None, "連れに party_today(共有回遊予定)が無い"
    # 相互 relations(連れ同士)
    assert 1 in leader.mem.relations and 2 in leader.mem.relations, "連れへの relations が注入されていない"
    assert 0 in sim.agent_by_id[1].mem.relations, "連れ→代表の相互 relations が無い"
    # 観測 joint_activity{type:party}
    pe = [e for e in _kind(sim, "joint_activity") if e.payload.get("type") == "party"]
    assert pe, "party の joint_activity が記録されていない"
    assert set(pe[0].payload["with"]) == {0, 1, 2}, "party メンバが記録と一致しない"
    assert pe[0].payload["place"] == leader_node, "party の place が代表の初期ノードでない"


def test_party_size_one_forms_no_group(tmp_path):
    """party_size=1(単独来街)は連れを組まない=joint_activity(party)も party_today も出ない。"""
    sim = _sim(tmp_path, "solo", n=20, **_ON)
    _make_party_visitors(sim, [0, 1], party_size=1)
    party.form_parties(sim, step=0, sim_min=0)
    assert all(sim.agent_by_id[i].party_today is None for i in (0, 1)), \
        "party_size=1 なのに連れ予定が付いている"
    assert not [e for e in _kind(sim, "joint_activity") if e.payload.get("type") == "party"], \
        "party_size=1 なのに party イベントが出ている"


# --------------------------------------------------------------------- 回遊の共有
def test_party_shared_roaming_dest(tmp_path):
    """回遊帯に連れの行き先=代表と共有の POI(party_dest。roam_bias=1 で必ず選択)。"""
    sim = _sim(tmp_path, "roam", n=20, **_ON)
    _make_party_visitors(sim, [0, 1], party_size=2)
    party.form_parties(sim, step=0, sim_min=0)
    poi = sim.agent_by_id[0].party_today["poi"]
    # 帯内(既定 10:00-22:00 の中=720 分)で共有 POI へ寄る(現在地から離す)
    mate = sim.agent_by_id[1]
    mate.node = sim.dests[-1] if sim.dests[-1] != poi else sim.dests[0]
    dest = party.party_dest(mate, sim, step=5, sim_min=720)
    assert dest == poi, "連れが共有 POI へ回遊していない"
    # 帯外は None(乱数も引かない)
    assert party.party_dest(mate, sim, step=5, sim_min=300) is None, "回遊帯外で寄せている"


# --------------------------------------------------------------- 旅を畳む(第109 レーン FIX)
def test_pulling_members_folds_their_trip(tmp_path):
    """★本選ブロッカーの回帰固定: 寄せた個体に**存在しない辺**の route が残らない。

    第109バッチ実測: pool の日次ローテーション(_phase_pool_rotation)から呼ばれると、
    連れには「前日から移動中」の route が残っていることがある。node だけ書き換えると
    次の _phase_move が (新 node, 旧 route[0]) を引いて KeyError で落ちた(step 144=day 1)。
    """
    sim = _sim(tmp_path, "fold", n=20, **_ON)
    _make_party_visitors(sim, [0, 1, 2], party_size=3)
    for i in (0, 1, 2):                                # 全員を「移動中」にする(別ノード行き)
        a = sim.agent_by_id[i]
        a.route = [d for d in sim.dests if d != a.node][:2]
        a.dest = a.route[-1]
        a.edge_offset = 3.5
        a.exit_intent = True
        a.homing = True
        a._ride_pending = {"mode": "train", "fare": 200}
    party.form_parties(sim, step=0, sim_min=0)
    for i in (0, 1, 2):
        a = sim.agent_by_id[i]
        assert a.route == [], f"連れ {i} の route が畳まれていない(存在しない辺が残る)"
        assert a.dest is None and a.edge_offset == 0.0, f"連れ {i} の旅が中途半端に残っている"
        assert not a.exit_intent and not a.homing, f"連れ {i} に移動しない退出意図が残っている"
        assert a._ride_pending is None, f"連れ {i} に到着しない乗車予約が残っている"


def test_folded_members_survive_a_move_phase(tmp_path):
    """寄せた直後に _phase_move を回しても例外が出ない(KeyError の直接再現→解消)。"""
    from society.engine import scheduler
    sim = _sim(tmp_path, "moveok", n=20, **_ON)
    _make_party_visitors(sim, [0, 1, 2], party_size=3)
    for i in (0, 1, 2):
        a = sim.agent_by_id[i]
        a.route = [d for d in sim.dests if d != a.node][:2]
        a.loc = "street"
        a.sleeping = False
    party.form_parties(sim, step=0, sim_min=0)
    scheduler._phase_move(sim, 1, 10)                  # ここで落ちていた


# --------------------------------------------------------------------- 決定論
def test_form_parties_deterministic(tmp_path):
    """同一設定で form_parties 2 回のグループ化(メンバ・共有 POI・寄せノード)が完全一致。"""
    def _run(name):
        s = _sim(tmp_path, name, n=20, **_ON)
        _make_party_visitors(s, [0, 1, 2, 3, 4], party_size=3)
        party.form_parties(s, step=0, sim_min=0)
        return {a.id: (a.node, a.party_today["poi"] if a.party_today else None)
                for a in s.agents if a.visitor and a.party_today}
    assert _run("det_a") == _run("det_b"), "party のグループ化が決定論でない"
