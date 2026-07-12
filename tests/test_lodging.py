"""宿泊・ホテル滞在(現実ギャップ 後続波 Wave L 2026-07-07)= 来街者の夜のホテル泊 のテスト。

方針(既存の鉄則を継承。test_annual_crowd.py の書式を踏襲):
- OFF(既定): lodging_checkin/lodging_checkout が 0 件・イベント列は純粋既定と L1 完全一致
  (144 step。ゴールデン golden_baseline_l1.json を守る)。"lodging" stream も引かない。
- ON 時: visitor がチェックインする(prob=1.0 で強制)、支払いが発生(spend cat="lodging")、
  checkout が出る、max_nights 超で退出。
- 決定論: ON 同士2回で L1 完全一致。
- k 不変性: controls.mode=compute_matched + FixedLLM で k=free と k=off の generate 呼数が完全一致
  (test_annual_crowd.py の test_crowd_call_count_k_invariant と同型)。宿泊は物理位置=co-location を
  変え FixedLLM で ON!=OFF になりうるが「呼数が k に依存しない」ことは compute_matched 下で厳密に保たれる。
検証は mock のみ(実LLM 禁止)。
"""
from __future__ import annotations

import json

from society.config import load_config
from society.engine import scheduler
from society.engine.simulation import Simulation

_ON = {"lodging.enabled": "true", "lodging.prob": "1.0"}


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


def _make_visitors(sim, bedtime_min=16 * 60, money=200000.0):
    """全エージェントを裕福な来街者に仕立てる(procedural 生成は visitor=0 のため)。決定論・同値。

    夜(bedtime)に範囲外退出しようとする→ホテルへ寄る、を確実に踏ませるための足場。"""
    for a in sim.agents:
        a.visitor = True
        a.commute = False
        a.money = float(money)
        a.bedtime_min = int(bedtime_min)


# --------------------------------------------------------------------- OFF 既定
def test_off_matches_pure_default(tmp_path):
    """明示 OFF と純粋既定が L1 完全一致(144 step)。Wave L の新イベントは 1 件も出ない(seam が no-op)。"""
    pure = _sim(tmp_path, "pure", steps=144)
    pure.run()
    off = _sim(tmp_path, "expl_off", steps=144, **{"lodging.enabled": "false"})
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(Wave L seam が no-op でない)"
    for k in ("lodging_checkin", "lodging_checkout"):
        assert not _kind(pure, k), f"OFF で {k} が出ている"


# --------------------------------------------------------------------- チェックイン + 支払い
def test_checkin_and_spend(tmp_path):
    """ON(prob=1.0): 夜の帰宅退出をしようとした visitor がホテルへ寄りチェックイン+支払いする。"""
    sim = _sim(tmp_path, "checkin", steps=1, **_ON)
    sim.run()                                          # step0 を1回回して据え付け(cfg/city)
    a = sim.agents[0]
    a.visitor, a.money = True, 200000.0
    a.loc, a.building, a.sleeping, a.route = "street", None, False, []
    a.lodging, a.lodging_nights = False, 0
    gate = sim.city.station_node or sim.city.gateways[0]
    a.node = gate
    a.x, a.y = sim.city.node_xy(gate)
    a.homing, a.exit_intent = True, True             # 夜の帰宅退出のタイミング

    step = 130
    scheduler._try_exit(sim, a, step, sim.clock.sim_min(step))
    assert a.loc != "outside", "宿泊するはずが範囲外へ退出してしまった"
    if not a.lodging:                                 # ホテルへ移動中 → 到着まで進める
        assert a.lodging_intent and a.route, "ホテルへ経路が張られていない"
        for s in range(step + 1, step + 80):
            scheduler._phase_move(sim, s, sim.clock.sim_min(s))
            if a.lodging:
                break
    assert a.lodging, "ホテルに到着してもチェックインしていない"
    ci = _kind(sim, "lodging_checkin")
    assert ci and ci[-1].agent_id == a.id and ci[-1].payload["nights"] == 1
    sp = [e for e in _kind(sim, "spend") if e.payload.get("cat") == "lodging"]
    assert sp and sp[-1].payload["amount"] == 12000.0, "宿泊費 spend(cat=lodging)が出ていない"
    assert a.sleeping and a.lodging_nights == 1, "チェックイン後に就寝・連泊数が正しくない"


# --------------------------------------------------------------------- チェックアウト
def test_checkout_resumes_activity(tmp_path):
    """チェックイン後、checkout_hour に達したら lodging_checkout が出て就寝・宿泊が解ける。"""
    sim = _sim(tmp_path, "checkout", steps=1, **{**_ON, "lodging.checkout_hour": "9"})
    sim.run()
    a = sim.agents[0]
    a.visitor, a.money = True, 200000.0
    a.loc, a.building, a.sleeping, a.route = "street", None, False, []
    a.lodging, a.lodging_nights = False, 0
    a.node = sim.city.station_node or sim.city.gateways[0]
    a.x, a.y = sim.city.node_xy(a.node)
    a.homing, a.exit_intent = True, True

    step = 20                                          # 深夜想定(sim_min=200 分)
    scheduler._try_exit(sim, a, step, sim.clock.sim_min(step))
    if not a.lodging:
        for s in range(step + 1, step + 80):
            scheduler._phase_move(sim, s, sim.clock.sim_min(s))
            if a.lodging:
                break
    assert a.lodging, "チェックインしていない"
    until = a.sleep_until
    assert until > step, "checkout(sleep_until)が未来に設定されていない"
    # checkout 直前は退館しない
    scheduler._phase_lodging(sim, until - 1, sim.clock.sim_min(until - 1))
    assert a.lodging and a.sleeping, "checkout 前に退館している"
    # checkout 時刻でチェックアウト
    scheduler._phase_lodging(sim, until, sim.clock.sim_min(until))
    co = _kind(sim, "lodging_checkout")
    assert co and co[-1].agent_id == a.id and co[-1].payload["nights_stayed"] == 1
    assert not a.lodging and not a.sleeping and a.building is None, "チェックアウトで活動再開していない"


# --------------------------------------------------------------------- 連泊上限
def test_max_nights_gate(tmp_path):
    """want_lodge は連泊数が max_nights に達したら False(=退出/帰宅)。未満なら prob=1.0 で True。"""
    from society import lodging
    sim = _sim(tmp_path, "maxn", steps=1, **{**_ON, "lodging.max_nights": "2"})
    sim.run()
    a = sim.agents[0]
    a.visitor, a.homing, a.money = True, True, 200000.0
    a.lodging_nights = 0
    assert lodging.want_lodge(sim, a, step=10), "連泊上限未満なのに宿泊できない"
    a.lodging_nights = 2                               # 上限に到達
    assert not lodging.want_lodge(sim, a, step=10), "連泊上限に達しても宿泊してしまう"
    a.homing = False                                   # 夜の帰宅退出でないなら宿泊しない
    a.lodging_nights = 0
    assert not lodging.want_lodge(sim, a, step=10), "homing でないのに宿泊してしまう"


def test_exit_when_max_nights_reached(tmp_path):
    """連泊上限に達した visitor は宿泊せず通常退出(exit_area)し、連泊数がリセットされる。"""
    sim = _sim(tmp_path, "maxexit", steps=1, **{**_ON, "lodging.max_nights": "1"})
    sim.run()
    a = sim.agents[0]
    a.visitor, a.money = True, 200000.0
    a.loc, a.building, a.sleeping, a.route = "street", None, False, []
    a.lodging_nights = 1                               # 既に上限(1泊)まで泊まった
    a.node = sim.city.station_node or sim.city.gateways[0]
    a.x, a.y = sim.city.node_xy(a.node)
    a.homing, a.exit_intent = True, True

    step = 130
    scheduler._try_exit(sim, a, step, sim.clock.sim_min(step))
    assert a.loc == "outside", "連泊上限なのに退出していない"
    assert not a.lodging, "連泊上限なのに宿泊してしまった"
    assert a.lodging_nights == 0, "退出で連泊数がリセットされていない"
    assert _kind(sim, "exit_area"), "exit_area が出ていない"


# --------------------------------------------------------------------- 決定論
def test_all_on_deterministic(tmp_path):
    """宿泊 ON(prob=1.0)同士 2 回で L1 完全一致(決定論・mock 144 step)。チェックインが実際に起きる。"""
    a = _sim(tmp_path, "det_a", n=30, steps=144, **_ON)
    _make_visitors(a)
    a.run()
    b = _sim(tmp_path, "det_b", n=30, steps=144, **_ON)
    _make_visitors(b)
    b.run()
    assert _l1(a) == _l1(b), "ON の決定論が崩れている"
    assert _kind(a, "lodging_checkin"), "宿泊 ON なのにチェックインが1件も起きていない(足場が不十分)"


# --------------------------------------------------------------------- k 不変性(R1)
class _FixedLLM:
    """挙動を固定する backend(応答をプロンプトに依存させない)。呼数だけ数える。"""

    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.hits = 0

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        self.calls += 1
        return self.response, str(self.calls), False


def _run_lodging_k(tmp_path, name, *, writeback):
    """宿泊 ON を compute_matched(k 掃引で実際に使う対照)下で回し呼数を数える。"""
    sim = _sim(tmp_path, name, n=30, steps=144,
               **{**_ON, "controls.mode": "compute_matched", "k.writeback": writeback})
    _make_visitors(sim)
    sim.llm = _FixedLLM(json.dumps({"action": "speak", "text": "x"},
                                   ensure_ascii=False))
    sim.run()
    return sim


def test_lodging_call_count_k_invariant(tmp_path):
    """宿泊は対面 co-location を変え ON!=OFF になりうるが、R1 の本旨=「呼数が k(writeback)に依存
    しない」ことは compute_matched 下で厳密に保たれる。宿泊の移動は暦+物理位置+config+所持金+連泊数
    のみ参照し k・内面状態を一切読まないため、k=free と k=off で呼数が完全一致する。"""
    free = _run_lodging_k(tmp_path, "lk_free", writeback="free")
    off = _run_lodging_k(tmp_path, "lk_off", writeback="off")
    assert free.llm.calls == off.llm.calls and free.llm.calls > 0, \
        f"宿泊の呼数が k に依存(R1 違反): free={free.llm.calls} off={off.llm.calls}"
    assert _kind(free, "lodging_checkin"), "宿泊 ON なのにチェックインが起きていない(機構が不発)"
