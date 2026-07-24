"""屋内エンジン配線 B3: scheduler._phase_indoor + indoor_tracks サイドカー + L2 の検収。

R1 ドクトリン:
- OFF(既定)= L1 に space_move が 1 件も出ず・sim.indoor=None・新 stream も引かない(ゴールデンは
  test_scenario が担保。本 file は OFF 追加不変を独立に固定)。
- ON/OFF で LLM 呼数一致(indoor は新 named stream のみ・プロンプト不変=cache キー不変)。
- ON 同 seed 2ラン = L1 + tracks が行レベル一致(mock 24step スモーク)。
- ON resume(分割)== straight の L1 + tracks 一致。
- tracks.enabled=false でも L1(space_move)は不変(記録と動力学の分離)。
- space_move の from/to が実 IndoorSpace の zone_types と整合・遭遇が実軌跡交差で記録。
- 会議: 同一職場グループが同 step に meeting 区画へ集まる(決定論)。
"""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from society.config import load_config
from society.engine import checkpoint, scheduler
from society.engine.simulation import Simulation

# 屋内エンジン全 ON(markov/sfm/meeting/tracks)。tests は個別に上書きする。
_ON = ["indoor.enabled=true", "indoor.markov.enabled=true",
       "indoor.sfm.enabled=true", "indoor.meeting.enabled=true",
       "indoor.tracks.enabled=true"]


def _cfg(name: str, n_steps: int, n_agents: int = 20, on=True, **ov):
    dot = ["run.seed=42", f"run.n_agents={n_agents}", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock"]
    if on:
        dot += list(_ON)
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


# --------------------------------------------------------------------------- #
# 1. OFF(既定)= space_move が出ない・indoor None(ゴールデンは test_scenario)
# --------------------------------------------------------------------------- #
def test_off_no_space_move(tmp_path):
    sim = Simulation(_cfg("off", 20, on=False), out_dir=tmp_path / "off")
    sim.run()
    assert sim.indoor is None
    assert sim.indoor_tracks is None
    assert not any(e.kind == "space_move" for e in sim.logger.events)
    assert not (tmp_path / "off" / "indoor_tracks_samples.parquet").exists()


# --------------------------------------------------------------------------- #
# 2. ON 同 seed 2ラン = L1 + tracks 行一致(決定論)
# --------------------------------------------------------------------------- #
def test_on_two_runs_identical(tmp_path):
    a = Simulation(_cfg("a", 24, n_agents=40), out_dir=tmp_path / "a")
    a.run()
    b = Simulation(_cfg("b", 24, n_agents=40), out_dir=tmp_path / "b")
    b.run()
    assert _l1(a) == _l1(b), "L1(space_move 含む)が 2 ラン間で不一致"
    assert any(e.kind == "space_move" for e in a.logger.events), "space_move が出ていない"
    for stem in ("indoor_tracks_samples", "indoor_tracks_contacts", "l2_metrics"):
        assert _rows(tmp_path / "a", stem) == _rows(tmp_path / "b", stem), \
            f"{stem} が 2 ラン間で不一致"


# --------------------------------------------------------------------------- #
# 3. ON resume(12+12)== straight(24)の L1 + tracks 一致
# --------------------------------------------------------------------------- #
def _run_resume(tmp_path, name, split, total, **ov):
    d = tmp_path / name
    every = {"observer.checkpoint_every": split}
    sim1 = Simulation(_cfg(name, split, **every, **ov), out_dir=d)
    for step in range(split):
        scheduler.run_step(sim1, step)
    checkpoint.save(sim1, split, d / "checkpoint" / f"ckpt-{split:06d}.pkl.gz")
    sim1.logger.flush_segment()
    if sim1.indoor_tracks is not None:
        sim1.indoor_tracks.flush_segment()
    sim2 = Simulation(_cfg(name, total, **every, **ov), out_dir=d)
    sim2.run(resume_from=d)
    return d


def test_resume_matches_straight(tmp_path):
    straight = tmp_path / "straight"
    Simulation(_cfg("straight", 24, n_agents=40), out_dir=straight).run()
    resumed = _run_resume(tmp_path, "resumed", 12, 24, n_agents=40)
    assert _rows(straight, "l1_events") == _rows(resumed, "l1_events"), "L1 不一致"
    for stem in ("indoor_tracks_samples", "indoor_tracks_contacts",
                 "l2_metrics", "l3_snapshots"):
        assert _rows(straight, stem) == _rows(resumed, stem), f"{stem} 不一致"


# --------------------------------------------------------------------------- #
# 4. ON/OFF で LLM 呼数一致(compute_matched: 新 stream のみ・プロンプト不変)
# --------------------------------------------------------------------------- #
def test_llm_calls_match_on_off(tmp_path):
    off = Simulation(_cfg("loff", 24, n_agents=30, on=False), out_dir=tmp_path / "loff")
    off.run()
    on = Simulation(_cfg("lon", 24, n_agents=30), out_dir=tmp_path / "lon")
    on.run()
    assert off.llm.calls == on.llm.calls, \
        f"LLM 呼数が ON/OFF で乖離: off={off.llm.calls} on={on.llm.calls}"
    assert off.llm.hits == on.llm.hits
    # indoor は space_move を「足すだけ」= それ以外の L1 は OFF と完全一致
    assert _l1(off) == _l1(on, skip_space=True), "space_move 以外の L1 が ON で変化した"


# --------------------------------------------------------------------------- #
# 5. tracks.enabled=false でも L1(動力学)不変(記録と動力学の分離)
# --------------------------------------------------------------------------- #
def test_tracks_off_l1_unchanged(tmp_path):
    on = Simulation(_cfg("tron", 24, n_agents=30), out_dir=tmp_path / "tron")
    on.run()
    off = Simulation(_cfg("troff", 24, n_agents=30,
                          **{"indoor.tracks.enabled": "false"}),
                     out_dir=tmp_path / "troff")
    off.run()
    assert on.indoor_tracks is not None and off.indoor_tracks is None
    assert _l1(on) == _l1(off), "tracks の ON/OFF で L1(動力学)が変化した"
    assert not (tmp_path / "troff" / "indoor_tracks_samples.parquet").exists()


# --------------------------------------------------------------------------- #
# 6. space_move の整合(from!=to・to_type が実 zone_types と一致)+ 遭遇の記録
# --------------------------------------------------------------------------- #
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


def test_space_move_consistency_and_encounters(tmp_path):
    # dwell_steps=1 = 全在館者が毎 step 遷移 → 1セルに多数 mover = 軌跡交差 → 遭遇。
    sim = Simulation(_cfg("smc", 4, n_agents=10,
                          **{"indoor.markov.dwell_steps": 1,
                             "indoor.meeting.enabled": "false"}),
                     out_dir=tmp_path / "smc")
    cell = _find_cell(sim, min_zones=4)
    assert cell is not None, "区画4以上のセルが見つからない(地図が想定外)"
    bid, floor, zt = cell
    _place_workers(sim, bid, floor, zt, sim.agents)   # 全 10 体を1セルへ
    step = 3
    scheduler._phase_indoor(sim, step, sim.clock.sim_min(step))
    sm = [e for e in sim.logger.events if e.kind == "space_move"]
    assert sm, "space_move が1件も出ていない(全員遷移のはず)"
    zt_now = sim.indoor.zone_types(bid, floor)
    for e in sm:
        p = e.payload
        assert p["from_zone"] != p["to_zone"], "space_move の from==to(実遷移でない)"
        assert p["to_type"] == zt_now[p["to_zone"]], "to_type が実 zone_types と不一致"
        assert p["from_type"] == zt_now[p["from_zone"]], "from_type が実 zone_types と不一致"
        assert 0.0 <= p["offset_s"] < 600.0
    # 遭遇: 一時状態と per-step カウンタが整合し、多数 mover の交差で >0
    assert sim._indoor_n_encounter == len(sim._indoor_encounters_step)
    assert sim._indoor_n_encounter > 0, "多数 mover の交差で遭遇が記録されていない"
    for (t_s, ia, ib, knd, dur, b, f) in sim._indoor_encounters_step:
        assert ia < ib and knd in ("passing", "coplace") and dur > 0.0
        assert b == bid and f == floor


# --------------------------------------------------------------------------- #
# 7. 会議: 同一職場グループが同 step に meeting 区画へ集まる(決定論)
# --------------------------------------------------------------------------- #
def test_meeting_gathers_colleagues(tmp_path):
    step = 5
    # まず素の sim を作って会議 step の tod を得る(window をその 1 バケットに合わせて確実に開催)。
    probe = Simulation(_cfg("mprobe", 8, n_agents=6, on=False), out_dir=tmp_path / "mp")
    tod = probe.clock.sim_min(step) % 1440
    ov = {"indoor.markov.enabled": "false", "indoor.sfm.enabled": "true",
          "indoor.meeting.prob": 1.0, "indoor.meeting.min_party": 2,
          "indoor.meeting.window_min": f"[{tod},{tod + 1}]"}
    sim = Simulation(_cfg("meet", 8, n_agents=6, **ov), out_dir=tmp_path / "meet")
    cell = _find_cell(sim, min_zones=3, need_type="meeting")
    assert cell is not None, "meeting 区画を持つセルが見つからない"
    bid, floor, zt = cell
    workers = list(sim.agents)
    _place_workers(sim, bid, floor, zt, workers)
    for a in workers:                                  # 同一職場グループ(work_building)へ束ねる
        a.work_building = bid
        # 会合区画に初期配置しない(=集まる余地を残す): _place_workers が zi=k%n で撒く。
    mz = [i for i, t in enumerate(zt) if t == "meeting"][0]
    for a in workers:                                  # 全員を会合区画以外から始める
        if a.ind_zone == mz:
            a.ind_zone = (mz + 1) % len(zt)
            a.ind_space_type = zt[a.ind_zone]
    scheduler._phase_indoor(sim, step, sim.clock.sim_min(step))
    gathered = sum(1 for a in workers if a.ind_zone == mz)
    assert gathered >= 2, f"会合区画に集まった勤務者が {gathered} 人(min_party 未満)"
    assert sim._indoor_n_meeting >= 1, "meeting カウンタが増えていない"
    # 決定論: 同一設定で再構築 → 同一の集合
    sim2 = Simulation(_cfg("meet2", 8, n_agents=6, **ov), out_dir=tmp_path / "meet2")
    _place_workers(sim2, bid, floor, zt, sim2.agents)
    for a in sim2.agents:
        a.work_building = bid
        if a.ind_zone == mz:
            a.ind_zone = (mz + 1) % len(zt)
            a.ind_space_type = zt[a.ind_zone]
    scheduler._phase_indoor(sim2, step, sim2.clock.sim_min(step))
    g1 = sorted(a.id for a in workers if a.ind_zone == mz)
    g2 = sorted(a.id for a in sim2.agents if a.ind_zone == mz)
    assert g1 == g2, "会議参加者が決定論でない(2 構築で不一致)"


# --------------------------------------------------------------------------- #
# 追加: 階間移動のワープ差替(現階→新階の到着レッグを実軌跡で記録・kind="floor")
# --------------------------------------------------------------------------- #
def test_floor_move_records_arrival(tmp_path):
    sim = Simulation(_cfg("floor", 4, n_agents=4,
                          **{"indoor.markov.enabled": "false",
                             "indoor.meeting.enabled": "false"}),
                     out_dir=tmp_path / "floor")
    # 2階以上ある建物で、両階に区画のあるものを探す
    target = None
    for b in sim.city.buildings:
        bid = b["id"]
        levels = int(b.get("levels", 1) or 1)
        if levels < 2:
            continue
        f1, f2 = 1, 2
        if sim.indoor.zone_types(bid, f1) and sim.indoor.zone_types(bid, f2):
            target = (bid, f1, f2)
            break
    assert target is not None, "2階以上で両階に区画のある建物が無い"
    bid, f_old, f_new = target
    a = sim.agents[0]
    a.loc = "street"
    a.sleeping = False
    a.building = bid
    a.floor = f_new
    a.activity = "working"
    a.ind_zone = 0                       # 旧階に居た区画(from_zone)
    a.ind_space_type = "x"
    a._ind_ctx = (bid, f_old)            # 直前は別階=floor_move を誘発
    step = 3
    scheduler._phase_indoor(sim, step, sim.clock.sim_min(step))
    fm = [e for e in sim.logger.events
          if e.kind == "space_move" and e.agent_id == a.id
          and e.payload.get("kind") == "floor"]
    assert fm, "階間移動(kind=floor)の space_move が記録されていない"
    p = fm[0].payload
    assert p["building"] == bid and p["floor"] == f_new
    assert p["from_zone"] == 0
    zt_new = sim.indoor.zone_types(bid, f_new)
    assert 0 <= p["to_zone"] < len(zt_new)
    assert p["to_type"] == zt_new[p["to_zone"]]
    assert a.ind_zone == p["to_zone"]    # 正典状態が新階の目的区画へ更新
