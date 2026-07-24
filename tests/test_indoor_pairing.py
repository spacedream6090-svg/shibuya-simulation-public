"""B3b: encounter→会話ペアリング接続 + 屋内 LOS 知覚 のテスト。

方針(co-location 変化系の正契約 / R1 を継承):
- OFF(既定): indoor.encounter.pairing=false・indoor.los.enabled=false=新経路を一切通さず、
  indoor engine ON の L1 とバイト一致(pairing/los の seam が no-op)。ゴールデンは test_scenario。
- pairing ON: 直近(前 step)の屋内遭遇相手が同席者に居れば返答相手に優先(遭遇 duration 降順/id 昇順)。
  変えるのは「誰と話すか」だけ=hearer 集合・会話発生・LLM 呼数は不変。物理の遭遇しか読まない=
  compute_matched 下の k 不変性(k=free==k=off の generate 呼数一致)で呼数を担保(群集 G4 と同型)。
- los ON: 発火プロンプトの同席リストを ind_x/ind_y の距離上限+壁 LOS(vision.IndoorLOSOccluder)で絞る。
  発火判断・返答権はゲートしない=会話発生/LLM 呼数不変(内容のみ変化)。座標・幾何しか読まない。
- 決定論: ON 同士2回で L1 完全一致。resume(分割)== straight(pairing ON)。
検証は mock のみ(実LLM 禁止)。
"""
from __future__ import annotations

import json
import types
from pathlib import Path

import pyarrow.parquet as pq

from society.config import load_config
from society.engine import checkpoint, scheduler
from society.engine.simulation import Simulation
from society.world import vision
from society.world.perception import hearers_of

# 屋内エンジン核(markov/sfm/meeting)。pairing/los/tracks は各テストで個別に付ける。
_ON = ["indoor.enabled=true", "indoor.markov.enabled=true",
       "indoor.sfm.enabled=true", "indoor.meeting.enabled=true"]


def _cfg(name: str, n_steps: int, n_agents: int = 20, on: bool = True,
         pairing: bool = False, los: bool = False, tracks: bool = False, **ov):
    dot = ["run.seed=42", f"run.n_agents={n_agents}", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock"]
    if on:
        dot += list(_ON)
    if tracks:
        dot += ["indoor.tracks.enabled=true"]
    if pairing:
        dot += ["indoor.encounter.pairing=true"]
    if los:
        dot += ["indoor.los.enabled=true"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def _l1(sim, skip_space=False):
    return [(e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True))
            for e in sim.logger.events
            if not (skip_space and e.kind == "space_move")]


def _rows(run_dir: Path, stem: str):
    p = Path(run_dir) / f"{stem}.parquet"
    return pq.read_table(p).to_pylist() if p.exists() else None


def _find_cell(sim, min_zones=3, need_type=None):
    """区画数 min_zones 以上(必要なら need_type を含む)の (building, floor, zone_types) を探す。"""
    for b in sim.city.buildings:
        bid = b["id"]
        levels = int(b.get("levels", 1) or 1)
        for f in range(1, levels + 1):
            zt = sim.indoor.zone_types(bid, f)
            if len(zt) >= min_zones and (need_type is None or need_type in zt):
                return bid, f, list(zt)
    return None


def _place_workers(sim, bid, floor, zt, ids):
    """指定 agent を (bid, floor) の非会合区画へ勤務者として据える(初期割当済み=markov モード)。"""
    layout = sim.indoor.get(bid, floor)["layout"]
    n = len(zt)
    for k, a in enumerate(ids):
        zi = k % n
        a.loc = "street"
        a.sleeping = False
        a.building = bid
        a.floor = floor
        a.activity = "working"
        a.ind_zone = zi
        a.ind_space_type = zt[zi]
        zx0, zy0, zx1, zy1 = layout["zones"][zi]
        a.ind_x, a.ind_y = (zx0 + zx1) / 2.0, (zy0 + zy1) / 2.0
        a._ind_ctx = (bid, floor)


# --------------------------------------------------------------------------- #
# 1. OFF(既定)= pairing/los seam が no-op(indoor engine ON の L1 とバイト一致)
# --------------------------------------------------------------------------- #
def test_off_pairing_los_inert(tmp_path):
    """pairing/los を明示 OFF にしても、指定なし(既定 false)と L1 完全一致=seam が no-op。"""
    base = Simulation(_cfg("base", 24, n_agents=40), out_dir=tmp_path / "base")
    base.run()
    off = Simulation(_cfg("off", 24, n_agents=40,
                          **{"indoor.encounter.pairing": "false",
                             "indoor.los.enabled": "false"}),
                     out_dir=tmp_path / "off")
    off.run()
    assert _l1(base) == _l1(off), "pairing/los 明示 OFF が既定と L1 不一致(seam が no-op でない)"
    assert any(e.kind == "space_move" for e in base.logger.events), "indoor engine が不発"


# --------------------------------------------------------------------------- #
# 2. pairing ON: 遭遇相手が返答相手に優先される(直接検証)
# --------------------------------------------------------------------------- #
def test_pairing_prefers_encounter_partner_direct(tmp_path):
    """_select_partner: 前 step の遭遇相手が同席者に居れば、より近い非遭遇者より優先して選ぶ。"""
    sim = Simulation(_cfg("sp", 1, n_agents=6, pairing=True), out_dir=tmp_path / "sp")
    A, B, C = sim.agents[0], sim.agents[1], sim.agents[2]
    A.x, A.y = 0.0, 0.0
    C.x, C.y = 1.0, 0.0      # 最寄り(pairing OFF ならこちら)
    B.x, B.y = 50.0, 0.0     # 遠い(だが遭遇相手 → pairing ON はこちら)
    A._indoor_recent = {B.id: 5.0}
    assert scheduler._pairing_on(sim) is True
    assert scheduler._select_partner(sim, A, [B, C]).id == B.id, \
        "pairing ON で遭遇相手(B)が優先されていない"
    # duration 降順の決定論: C の遭遇が長ければ C を選ぶ
    A._indoor_recent = {B.id: 3.0, C.id: 9.0}
    assert scheduler._select_partner(sim, A, [B, C]).id == C.id, "duration 降順で選ばれていない"
    # 同 duration は id 昇順
    A._indoor_recent = {B.id: 5.0, C.id: 5.0}
    assert scheduler._select_partner(sim, A, [B, C]).id == min(B.id, C.id), \
        "同 duration の同点処理が id 昇順でない"
    # 遭遇相手が同席者に居なければ従来(最寄り C)へ後退
    A._indoor_recent = {999: 5.0}
    assert scheduler._select_partner(sim, A, [B, C]).id == C.id, "非同席の遭遇相手が後退しない"
    # pairing OFF は常に最寄り(C)= 現行と一字一句同一
    sim.indoorcfg["encounter"]["pairing"] = False
    A._indoor_recent = {B.id: 5.0}
    assert scheduler._select_partner(sim, A, [B, C]).id == C.id, "pairing OFF が最寄りを選ばない"


# --------------------------------------------------------------------------- #
# 3. pairing: _phase_indoor が遭遇を _indoor_recent へ焼き込む(前 step 用の材料)
# --------------------------------------------------------------------------- #
def test_pairing_recent_populated_from_phase_indoor(tmp_path):
    """dwell=1 の全員遷移で遭遇が起き、各遭遇 (ia,ib) が双方の _indoor_recent に対称に載る。"""
    sim = Simulation(_cfg("rec", 4, n_agents=10, pairing=True,
                          **{"indoor.markov.dwell_steps": 1,
                             "indoor.meeting.enabled": "false"}),
                     out_dir=tmp_path / "rec")
    cell = _find_cell(sim, min_zones=4)
    assert cell is not None, "区画4以上のセルが見つからない"
    bid, floor, zt = cell
    _place_workers(sim, bid, floor, zt, sim.agents)
    step = 3
    scheduler._phase_indoor(sim, step, sim.clock.sim_min(step))
    assert sim._indoor_encounters_step, "遭遇が1件も出ていない(全員遷移のはず)"
    # 各遭遇が双方の _indoor_recent に載り、duration>0 で対称
    for (_t, ia, ib, _k, dur, _b, _f) in sim._indoor_encounters_step:
        ra = sim.agent_by_id[ia]._indoor_recent
        rb = sim.agent_by_id[ib]._indoor_recent
        assert ib in ra and ra[ib] > 0.0, f"{ia} の recent に相手 {ib} が無い"
        assert ia in rb and rb[ia] > 0.0, f"{ib} の recent に相手 {ia} が無い"
        assert ra[ib] == rb[ia], "遭遇 duration が非対称"
    # 遭遇していない相手は recent に載らない(1-step 限定・当 step の遭遇のみ)
    someone = sim.agents[0]
    partners = set(someone._indoor_recent)
    encountered = {ib for (_t, ia, ib, *_r) in sim._indoor_encounters_step if ia == someone.id}
    encountered |= {ia for (_t, ia, ib, *_r) in sim._indoor_encounters_step if ib == someone.id}
    assert partners == encountered, "recent が当 step の遭遇集合と一致しない"


# --------------------------------------------------------------------------- #
# 4. pairing ON: 同 seed 2ラン = L1 完全一致(決定論)
# --------------------------------------------------------------------------- #
def test_pairing_two_runs_identical(tmp_path):
    a = Simulation(_cfg("pa", 24, n_agents=40, pairing=True, tracks=True),
                   out_dir=tmp_path / "pa")
    a.run()
    b = Simulation(_cfg("pb", 24, n_agents=40, pairing=True, tracks=True),
                   out_dir=tmp_path / "pb")
    b.run()
    assert _l1(a) == _l1(b), "pairing ON の L1 が 2 ラン間で不一致(非決定論)"
    for stem in ("indoor_tracks_samples", "indoor_tracks_contacts", "l2_metrics"):
        assert _rows(tmp_path / "pa", stem) == _rows(tmp_path / "pb", stem), \
            f"{stem} が 2 ラン間で不一致"


# --------------------------------------------------------------------------- #
# 5. pairing ON: LLM 呼数の k 不変性(compute_matched: k=free==k=off)
# --------------------------------------------------------------------------- #
def _run_pairing_k(tmp_path, name, writeback):
    """pairing ON を compute_matched 下で回し generate 呼数を数える(遭遇は物理のみ=k 非依存)。"""
    sim = Simulation(_cfg(name, 48, n_agents=30, pairing=True,
                          **{"controls.mode": "compute_matched",
                             "k.writeback": writeback}),
                     out_dir=tmp_path / name)
    sim.run()
    return sim


def test_pairing_call_count_k_invariant(tmp_path):
    """pairing の相手選択は物理の遭遇(duration/ids)しか読まない=k=free と k=off で呼数完全一致。

    相手が変われば ON!=OFF になりうる(co-location 変化の必然=群集 G4 と同型)が、R1 の本旨=
    「呼数が k に依存しない」ことは compute_matched 下で厳密に保たれる。"""
    free = _run_pairing_k(tmp_path, "pk_free", "free")
    off = _run_pairing_k(tmp_path, "pk_off", "off")
    assert free.llm.calls == off.llm.calls and free.llm.calls > 0, \
        f"pairing の呼数が k に依存(R1 違反): free={free.llm.calls} off={off.llm.calls}"
    assert free.llm.hits == off.llm.hits, "cache hits も k 不変であるべき"


# --------------------------------------------------------------------------- #
# 6. pairing ON: resume(12+12)== straight(24)の L1 + tracks 一致
# --------------------------------------------------------------------------- #
def _run_resume(tmp_path, name, split, total, **ov):
    d = tmp_path / name
    every = {"observer.checkpoint_every": split}
    sim1 = Simulation(_cfg(name, split, pairing=True, tracks=True, **every, **ov),
                      out_dir=d)
    for step in range(split):
        scheduler.run_step(sim1, step)
    checkpoint.save(sim1, split, d / "checkpoint" / f"ckpt-{split:06d}.pkl.gz")
    sim1.logger.flush_segment()
    if sim1.indoor_tracks is not None:
        sim1.indoor_tracks.flush_segment()
    sim2 = Simulation(_cfg(name, total, pairing=True, tracks=True, **every, **ov),
                      out_dir=d)
    sim2.run(resume_from=d)
    return d


def test_pairing_resume_matches_straight(tmp_path):
    """pairing ON: _indoor_recent は per-agent 属性ゆえ checkpoint(agents pickle)で持ち越され、
    分割再開(12+12)が一気通し(24)と L1 + tracks 完全一致(checkpoint.py は不変更)。"""
    straight = tmp_path / "straight"
    Simulation(_cfg("straight", 24, n_agents=40, pairing=True, tracks=True),
               out_dir=straight).run()
    resumed = _run_resume(tmp_path, "resumed", 12, 24, n_agents=40)
    assert _rows(straight, "l1_events") == _rows(resumed, "l1_events"), "pairing resume の L1 不一致"
    for stem in ("indoor_tracks_samples", "indoor_tracks_contacts",
                 "l2_metrics", "l3_snapshots"):
        assert _rows(straight, stem) == _rows(resumed, stem), f"{stem} 不一致"


# --------------------------------------------------------------------------- #
# 7. los: 壁 LOS + 距離ゲート の直接検証(IndoorLOSOccluder 単体)
# --------------------------------------------------------------------------- #
class _FakeIndoor:
    """IndoorSpace.get(bid,floor)['walls'] だけを返す最小スタブ(壁を明示注入する単体検証用)。"""

    def __init__(self, walls):
        self._walls = walls

    def get(self, bid, floor):
        return {"walls": self._walls}


def _ag(**kw):
    return types.SimpleNamespace(**kw)


def test_los_occluder_blocks_wall_and_distance_direct():
    """壁越しは遮る/同室は遮らない/距離超過は遮る/街路・未割当は遮らない。"""
    wall = [((0.0, -10.0), (0.0, 10.0))]     # x=0 の垂直壁
    occ = vision.IndoorLOSOccluder(_FakeIndoor(wall), max_dist_m=0.0)
    a = _ag(id=1, building="b", floor=1, ind_zone=0, ind_x=-5.0, ind_y=0.0)
    b = _ag(id=2, building="b", floor=1, ind_zone=1, ind_x=5.0, ind_y=0.0)
    c = _ag(id=3, building="b", floor=1, ind_zone=0, ind_x=-3.0, ind_y=0.0)
    assert occ.blocks(a, b) is True, "壁越しの相手を遮っていない"
    assert occ.blocks(a, c) is False, "同室(壁なし)を誤って遮った"
    # 距離ゲート(壁なし)
    occ_d = vision.IndoorLOSOccluder(_FakeIndoor([]), max_dist_m=5.0)
    far = _ag(id=4, building="b", floor=1, ind_zone=0, ind_x=100.0, ind_y=0.0)
    assert occ_d.blocks(a, far) is True, "距離上限超過を遮っていない"
    assert occ_d.blocks(a, c) is False, "近距離(壁なし)を誤って遮った"
    # 街路・屋外・別階・ミクロ未割当は対象外(遮らない=従来同一)
    street = _ag(id=5, building=None, floor=0, ind_zone=None, ind_x=0.0, ind_y=0.0)
    other_floor = _ag(id=6, building="b", floor=2, ind_zone=0, ind_x=5.0, ind_y=0.0)
    unset = _ag(id=7, building="b", floor=1, ind_zone=None, ind_x=0.0, ind_y=0.0)
    assert occ.blocks(street, b) is False, "街路ペアを遮った"
    assert occ.blocks(a, other_floor) is False, "別階ペアを遮った"
    assert occ.blocks(a, unset) is False, "ミクロ未割当ペアを遮った"


# --------------------------------------------------------------------------- #
# 8. los ON: 同席文脈(近傍リスト)から屋内で遠い/壁越しの相手が消える(統合)
# --------------------------------------------------------------------------- #
def test_los_removes_from_company(tmp_path):
    """hearers_of に _los_occluder を渡すと、同一(建物,階)でも屋内距離超過の相手が同席文脈から外れる。

    マクロ x/y は同一(base 半径は通過)にして、屋内 ind_x/ind_y の距離ゲートだけで外れることを示す。"""
    sim = Simulation(_cfg("losc", 1, n_agents=6, los=True,
                          **{"indoor.los.max_dist_m": 5.0}),
                     out_dir=tmp_path / "losc")
    cell = _find_cell(sim, min_zones=2)
    assert cell is not None
    bid, floor, zt = cell
    A, B = sim.agents[0], sim.agents[1]
    for a in (A, B):
        a.loc = "street"
        a.sleeping = False
        a.building = bid
        a.floor = floor
        a.x, a.y = 100.0, 100.0        # マクロ同一 → base 距離 0(半径内)
        a.ind_zone = 0
        a.ind_space_type = zt[0]
    A.ind_x, A.ind_y = 0.0, 0.0
    B.ind_x, B.ind_y = 50.0, 0.0        # 屋内 50m > max_dist 5m
    radius = float(sim.cfg.world.perception_radius_m)
    occ = scheduler._los_occluder(sim)
    assert occ is not None and scheduler._los_on(sim) is True
    # occluder なし(los OFF 相当)= B は同席文脈に残る
    assert any(h.id == B.id for h in hearers_of(A, [A, B], radius, occluder=None)), \
        "ゲートなしで B が同席文脈に居ない(前提が壊れている)"
    # occluder あり = 屋内で遠すぎる B は同席文脈から消える(hearer 集合の中身が変わる)
    assert all(h.id != B.id for h in hearers_of(A, [A, B], radius, occluder=occ)), \
        "los ON で屋内距離超過の B が同席文脈から外れていない"


# --------------------------------------------------------------------------- #
# 9. los ON: 同 seed 2ラン = L1 完全一致(決定論)
# --------------------------------------------------------------------------- #
def test_los_two_runs_identical(tmp_path):
    a = Simulation(_cfg("la", 24, n_agents=40, los=True), out_dir=tmp_path / "la")
    a.run()
    b = Simulation(_cfg("lb", 24, n_agents=40, los=True), out_dir=tmp_path / "lb")
    b.run()
    assert _l1(a) == _l1(b), "los ON の L1 が 2 ラン間で不一致(非決定論)"


# --------------------------------------------------------------------------- #
# 10. los ON: LLM 呼数の k 不変性(compute_matched: k=free==k=off)
# --------------------------------------------------------------------------- #
def _run_los_k(tmp_path, name, writeback):
    sim = Simulation(_cfg(name, 48, n_agents=30, los=True,
                          **{"controls.mode": "compute_matched",
                             "k.writeback": writeback}),
                     out_dir=tmp_path / name)
    sim.run()
    return sim


def test_los_call_count_k_invariant(tmp_path):
    """los の同席ゲートは座標・幾何しか読まない=k=free と k=off で generate 呼数完全一致。"""
    free = _run_los_k(tmp_path, "lk_free", "free")
    off = _run_los_k(tmp_path, "lk_off", "off")
    assert free.llm.calls == off.llm.calls and free.llm.calls > 0, \
        f"los の呼数が k に依存(R1 違反): free={free.llm.calls} off={off.llm.calls}"
    assert free.llm.hits == off.llm.hits, "cache hits も k 不変であるべき"
