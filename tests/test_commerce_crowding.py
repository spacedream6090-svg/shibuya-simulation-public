"""CRWD 混雑不満(購入/受給の**成立時**)のテスト。第147。

正典: docs/plans/inventory-two-tier-plan.md §1.6。
リサーチ: docs/research/crowding-dissatisfaction-empirics.md(§6 業態表・§7 予期・§8 常連補正の採用棄却)。

何を機械固定するか:
- OFF(既定): L1 が純粋既定とバイト一致(seam が no-op)・grievance も記憶も 1 バイトも動かない。
- 式(純関数): ramp の境界 / w_e=0 は純絶対負荷 / E 超過の効き / U 字の符号 / 閑散罰 /
  表に無い業態は 0 / 飽和判定。
- 在館数表は **毎 step 1 回だけ**作られる(購入ごとの全走査を復活させない=250k の再発防止)。
- grievance の書き込みは ON のときだけ・cause は factors 側の 1 か所に閉じる。
- 記憶は飽和帯だけ(nightlife の**負帯では残さない**)。
- 決定論(同 seed 2 ラン一致)・resume == straight・LLM 呼数が k に依存しない(R1)。
検証は mock のみ(実LLM 禁止・≤144 step)。乱数は 1 本も引かない。
"""
from __future__ import annotations

import json

from society import commerce
from society.config import load_config
from society.engine import scheduler
from society.engine.simulation import Simulation

_CRWD = {"commerce.crowding.enabled": "true"}


def _sim(tmp_path, name, n=30, steps=1, **ov):
    dot = ["run.seed=42", f"run.n_agents={n}", f"run.n_steps={steps}",
           f"run.name={name}", "model.backend=mock"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _crowd_updates(sim):
    return [e for e in sim.logger.events
            if e.kind == "state_update" and e.payload.get("cause") == "store_crowding"]


def _cfg(**over):
    """commerce の正準 cfg を **Simulation を作らずに**組む(式の単体テスト用)。"""
    raw = {"crowding": {"enabled": True}}
    raw["crowding"].update(over.pop("crowding", {}))
    raw.update(over)
    return commerce.build_cfg(raw)


# --------------------------------------------------------------------- OFF 既定
def test_off_matches_pure_default(tmp_path):
    """明示 OFF と純粋既定が L1 完全一致(CRWD seam が no-op=ゴールデンの世界を動かさない)。"""
    pure = _sim(tmp_path, "cw_pure", steps=48)
    pure.run()
    off = _sim(tmp_path, "cw_off", steps=48, **{"commerce.crowding.enabled": "false"})
    off.run()
    assert _l1(pure) == _l1(off), "CRWD OFF が純粋既定と不一致(seam が no-op でない)"
    assert not _crowd_updates(pure), "OFF で混雑由来の state_update が出ている"


def test_off_does_not_build_the_counts_table(tmp_path, monkeypatch):
    """OFF では在館数表を **1 回も**作らない(= 追加コストがゼロであることの機械証明)。"""
    calls = {"n": 0}
    real = commerce.node_counts

    def _counted(sim):
        calls["n"] += 1
        return real(sim)

    monkeypatch.setattr(commerce, "node_counts", _counted)
    sim = _sim(tmp_path, "cw_nocount", steps=6)
    for step in range(6):
        scheduler.run_step(sim, step)
    assert calls["n"] == 0, f"OFF なのに在館数表を {calls['n']} 回作っている"
    assert sim._crowd_counts is None


def test_counts_table_is_built_exactly_once_per_step(tmp_path, monkeypatch):
    """★ON でも在館数表は **1 step につき 1 回だけ**(購入ごとの全走査を復活させない)。"""
    calls = {"n": 0}
    real = commerce.node_counts

    def _counted(sim):
        calls["n"] += 1
        return real(sim)

    monkeypatch.setattr(commerce, "node_counts", _counted)
    n_steps = 12
    sim = _sim(tmp_path, "cw_once", steps=n_steps, **_CRWD)
    for step in range(n_steps):
        scheduler.run_step(sim, step)
    assert calls["n"] == n_steps, \
        f"在館数表の構築が {calls['n']} 回(期待 {n_steps} = 1 step 1 回)"
    assert isinstance(sim._crowd_counts, dict)


def test_counts_table_matches_occupancy(tmp_path):
    """表の値は occupancy(全走査)と常に一致する(同値変換であることの機械固定)。"""
    sim = _sim(tmp_path, "cw_eq", steps=1, **_CRWD)
    counts = commerce.step_counts(sim)
    for node in sorted({a.node for a in sim.agents}):
        assert counts.get(node, 0) == commerce.occupancy(sim, node), \
            f"{node} の在館数が全走査と食い違う"


# --------------------------------------------------------------------- 式(純関数)
def test_ramp_boundaries():
    """ramp(x;L0,L1)=clamp((x−L0)/(L1−L0),0,1): 下端 0 / 上端 1 / 中点 0.5 / 退化幅は段差。"""
    assert commerce.ramp(0.5, 0.6, 1.3) == 0.0, "L0 未満で 0 でない"
    assert commerce.ramp(0.6, 0.6, 1.3) == 0.0, "L0 ちょうどで 0 でない"
    assert commerce.ramp(1.3, 0.6, 1.3) == 1.0, "L1 ちょうどで飽和していない"
    assert commerce.ramp(99.0, 0.6, 1.3) == 1.0, "L1 超で頭打ちになっていない(線形無限伸長)"
    assert abs(commerce.ramp(0.95, 0.6, 1.3) - 0.5) < 1e-12, "中点が 0.5 でない"
    # 退化した幅(L1<=L0)は 0 除算を作らず段差になる
    assert commerce.ramp(0.9, 1.0, 1.0) == 0.0 and commerce.ramp(1.0, 1.0, 1.0) == 1.0


def test_band_lookup_covers_the_day():
    """時間帯バンドは朝/昼/夕/夜を覆い、夜だけが日を跨ぐ(時刻の純関数)。"""
    cfg = _cfg()
    assert commerce.crowding_band(cfg, 7 * 60) == "morning"
    assert commerce.crowding_band(cfg, 12 * 60) == "midday"
    assert commerce.crowding_band(cfg, 17 * 60) == "evening"
    assert commerce.crowding_band(cfg, 22 * 60) == "night"
    assert commerce.crowding_band(cfg, 3 * 60) == "night", "深夜が夜帯に入っていない"


def test_w_e_zero_is_pure_absolute_load():
    """w_e=0 なら L̃ は純粋な絶対負荷 L=在館数/cap(予期の減衰を完全に切れる)。"""
    cfg = _cfg(crowding={"w_e": 0.0})
    # food の cap=20。時間帯を変えても値が動かない=E を一切見ていないことの証明。
    for sim_min in (7 * 60, 12 * 60, 17 * 60, 23 * 60):
        assert abs(commerce.crowding_load(cfg, "food", 10, sim_min) - 0.5) < 1e-12


def test_expected_load_reduces_the_burden():
    """E 超過だけを見る(w_e=1)と、平常負荷の高い時間帯ほど L̃ が小さくなる(予期の減衰)。"""
    cfg = _cfg(crowding={"w_e": 1.0})
    # food: 朝 E=0.2 / 昼 E=0.8。同じ在館数 20 人(L=1.0)でも昼は超過が小さい。
    morning = commerce.crowding_load(cfg, "food", 20, 7 * 60)
    midday = commerce.crowding_load(cfg, "food", 20, 12 * 60)
    assert abs(morning - 0.8) < 1e-12, "E を引いた超過分になっていない"
    assert abs(midday - 0.2) < 1e-12, "ピーク帯で罰が軽くなっていない"
    assert midday < morning, "予期(時間帯平常値)の減衰が効いていない"
    # E を超えない側は 0 で止まる(負にならない=max(0, L−E))
    assert commerce.crowding_load(cfg, "food", 2, 12 * 60) == 0.0


def test_expected_is_mixed_half_and_half_at_default_w_e():
    """既定 w_e=0.6 では L̃=(1−w_e)L + w_e·max(0,L−E)(絶対負荷と超過の混合)。"""
    cfg = _cfg()
    load, e, w = 1.0, 0.8, 0.6                       # food 昼: L=20/20=1.0, E=0.8
    want = (1.0 - w) * load + w * max(0.0, load - e)
    assert abs(commerce.crowding_load(cfg, "food", 20, 12 * 60) - want) < 1e-12


def test_magnitude_is_monotone_then_saturates():
    """物販: L0 未満は 0 → 単調増加 → L1 で m に飽和(閾値つき飽和=線形無限伸長でない)。"""
    cfg = _cfg(crowding={"w_e": 0.0})                # 絶対負荷だけを見る
    m = cfg["crowding"]["table"]["shop"]["m"]
    vals = [commerce.crowding_magnitude(cfg, "shop", occ, 12 * 60)
            for occ in (0, 15, 24, 30, 39, 60, 300)]  # cap=30 → L=0/0.5/0.8/1.0/1.3/2.0/10
    assert vals[0] == 0.0 and vals[1] == 0.0, "L0 未満で不満が出ている"
    assert 0.0 < vals[2] < vals[3] < vals[4], "単調増加になっていない"
    assert abs(vals[4] - m) < 1e-12, "L1 で m に飽和していない"
    assert vals[5] == vals[6] == vals[4], "L1 超で伸び続けている(飽和していない)"
    assert m == 0.008, "shop の上限が業態表(§6)と食い違う"


def test_food_quiet_penalty_below_threshold():
    """飲食: L<0.3 は**閑散罰**(+0.004。Tse 2002 の social proof=空きすぎも負)。"""
    cfg = _cfg(crowding={"w_e": 0.0})
    assert commerce.crowding_magnitude(cfg, "food", 1, 12 * 60) == 0.004, "閑散罰が出ていない"
    # 0.3 以上 0.7(L0)未満は「普通」= 0
    assert commerce.crowding_magnitude(cfg, "food", 8, 12 * 60) == 0.0
    assert commerce.crowding_magnitude(cfg, "food", 28, 12 * 60) == 0.010, "food の飽和が 0.010 でない"


def test_nightlife_u_shape_signs():
    """夜遊びは U 字: 閑散は正・中密度帯は**負**(社会的高揚)・高負荷で正に戻る。"""
    cfg = _cfg(crowding={"w_e": 0.0})                # cap=40
    quiet = commerce.crowding_magnitude(cfg, "nightlife", 4, 22 * 60)     # L=0.1
    mid = commerce.crowding_magnitude(cfg, "nightlife", 24, 22 * 60)      # L=0.6
    edge = commerce.crowding_magnitude(cfg, "nightlife", 44, 22 * 60)     # L=1.1(帯の上端)
    over = commerce.crowding_magnitude(cfg, "nightlife", 60, 22 * 60)     # L=1.5
    packed = commerce.crowding_magnitude(cfg, "nightlife", 200, 22 * 60)  # L=5.0
    assert quiet == 0.004, "夜遊びの閑散罰が出ていない"
    assert mid == -0.004, "中密度帯が負(社会的高揚)になっていない"
    assert edge == -0.004, "U 字の帯の上端が負帯に入っていない"
    assert over > 0.0, "高負荷側で正に戻っていない(圧迫)"
    assert packed == 0.010, "夜遊びの飽和が 0.010 でない"


def test_unknown_category_has_no_effect():
    """業態表に無いカテゴリは 0.0(= その業態には混雑不満を入れない、という宣言)。"""
    cfg = _cfg()
    assert commerce.crowding_magnitude(cfg, "office", 999, 12 * 60) == 0.0
    assert commerce.crowding_saturated(cfg, "office", 999, 12 * 60) is False


def test_saturation_flag_tracks_l1():
    """飽和判定は L̃ >= L1(記憶を残すかの唯一の入力)。"""
    cfg = _cfg(crowding={"w_e": 0.0})
    assert not commerce.crowding_saturated(cfg, "food", 27, 12 * 60)   # L=1.35 < 1.4
    assert commerce.crowding_saturated(cfg, "food", 28, 12 * 60)       # L=1.40


def test_cap_zero_is_safe():
    """cap<=0(病的 conf)でも 0 除算せず 0.0(不満なし)へ落ちる。"""
    cfg = _cfg(crowding={"cap": {"food": 0}})
    assert commerce.crowding_load(cfg, "food", 100, 12 * 60) == 0.0


# --------------------------------------------------------------------- seam(成立時)
def _put(sim, agent, node, occ):
    """agent を node に置き、その step の在館数表を occ で作る(表の作り方は本体と同じ形)。"""
    agent.node = node
    sim._crowd_counts = {node: int(occ)}


def test_apply_crowding_writes_grievance_only_when_on(tmp_path):
    """ON のときだけ grievance が動く(OFF は state も記録も 1 バイトも動かさない)。"""
    off = _sim(tmp_path, "cw_seam_off", steps=1)
    a = off.agents[0]
    _put(off, a, a.node, 40)
    g0 = a.states["grievance"]
    assert commerce.apply_crowding(off, a, "food", 0, 12 * 60) == 0.0
    assert a.states["grievance"] == g0, "OFF なのに不満が動いた"
    assert not _crowd_updates(off)

    on = _sim(tmp_path, "cw_seam_on", steps=1, **_CRWD)
    b = on.agents[0]
    _put(on, b, b.node, 40)                            # food cap=20 → L=2.0 = 飽和
    g1 = b.states["grievance"]
    delta = commerce.apply_crowding(on, b, "food", 0, 12 * 60)
    assert delta > 0.0 and on.agents[0].states["grievance"] > g1, "混雑で不満が上がっていない"
    su = _crowd_updates(on)
    assert su and su[-1].payload["new"] > su[-1].payload["old"], \
        "混雑→grievance が factors 経由(cause=store_crowding)で入っていない"


def test_remember_only_at_saturation(tmp_path):
    """記憶は**飽和帯のときだけ** 1 行(平常の混雑では残さない)。"""
    sim = _sim(tmp_path, "cw_mem", steps=1, **_CRWD)
    a, b = sim.agents[0], sim.agents[1]
    _put(sim, a, a.node, 40)                           # L=2.0 >= L1(1.4)= 飽和
    commerce.apply_crowding(sim, a, "food", 0, 12 * 60)
    assert any(commerce._CROWD_TEXT in str(m.text) for m in a.mem.buffer), \
        "飽和した混雑が記憶に残っていない"
    _put(sim, b, b.node, 16)                           # L=0.8 → L̃<1.4 = 飽和していない
    commerce.apply_crowding(sim, b, "food", 0, 12 * 60)
    assert not any(commerce._CROWD_TEXT in str(m.text) for m in b.mem.buffer), \
        "飽和していない混雑まで記憶に残している"


def test_nightlife_negative_band_lowers_grievance_and_leaves_no_memory(tmp_path):
    """夜遊びの中密度帯は grievance が**下がり**、「混んでいた」記憶も残さない。"""
    sim = _sim(tmp_path, "cw_night", steps=1,
               **{**_CRWD, "commerce.crowding.w_e": "0.0"})
    a = sim.agents[0]
    a.states["grievance"] = 0.5                        # 下がる余地を作る(clip[0,1] の中)
    _put(sim, a, a.node, 24)                           # nightlife cap=40 → L=0.6 = 負帯
    delta = commerce.apply_crowding(sim, a, "nightlife", 0, 22 * 60)
    assert delta < 0.0 and a.states["grievance"] < 0.5, "中密度帯で高揚(grievance−)していない"
    assert not any(commerce._CROWD_TEXT in str(m.text) for m in a.mem.buffer), \
        "快い混雑を「混んでいた」と記憶している(負帯で remember しない契約の違反)"


def test_seam_is_noop_without_the_counts_table(tmp_path):
    """表が無い step(表を作る前に呼ばれた等)では何もしない(黙って全走査に落ちない)。"""
    sim = _sim(tmp_path, "cw_notable", steps=1, **_CRWD)
    a = sim.agents[0]
    sim._crowd_counts = None
    g0 = a.states["grievance"]
    assert commerce.apply_crowding(sim, a, "food", 0, 12 * 60) == 0.0
    assert a.states["grievance"] == g0


# --------------------------------------------------------------------- 統合・決定論
def test_run_moves_grievance_and_stays_deterministic(tmp_path):
    """フル run で混雑由来の state_update が出て、同 seed 2 ランが L1 完全一致(決定論)。"""
    a = _sim(tmp_path, "cw_det_a", n=30, steps=144, **_CRWD)
    a.run()
    assert _crowd_updates(a), "ON なのに混雑由来の state_update が 1 件も出ていない"
    b = _sim(tmp_path, "cw_det_b", n=30, steps=144, **_CRWD)
    b.run()
    assert _l1(a) == _l1(b), "CRWD ON の決定論が崩れている"


def test_implementation_has_no_rng_identifiers():
    """★実装に乱数の識別子が存在しない(traces / devices / night と同じ AST 固定)= 新 stream 0 本。"""
    import ast
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "src" / "society"
           / "commerce.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for bad in ("rng", "random", "hub", "stream", "integers", "shuffle", "uniform"):
        assert bad not in names and bad not in attrs, \
            f"commerce.py に乱数の識別子 {bad} がある(CRWD/PRICE-B は決定論の契約)"
    # R1: 混雑不満は**発火系(drive)に接続しない**(呼数を 1 本も動かさない)。
    #     ★docstring の「drive には接続しない」という**説明**は残るので、素の文字列検索では
    #       なく識別子(Name/Attribute)で見る。
    assert "drive" not in names and "drive" not in attrs, \
        "commerce.py が drive(発火系)を呼んでいる(R1 違反)"


# --------------------------------------------------------------------- resume == straight
_SPLIT, _TOTAL = 77, 144


def _cfg_of(name, n_steps, every=0):
    ov = dict(_CRWD)
    if every:
        ov["observer.checkpoint_every"] = str(every)
    dot = ["run.seed=42", "run.n_agents=30", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def _rows(run_dir):
    import pyarrow.parquet as pq
    return pq.read_table(run_dir / "l1_events.parquet").to_pylist()


def test_resume_equals_straight(tmp_path):
    """★分割走行 == 一気通し。在館数表は step ローカルな派生量なので搬送不要で一致する。"""
    from society.engine import checkpoint

    st_dir = tmp_path / "cw_st"
    straight = Simulation(_cfg_of("cw_st", _TOTAL), out_dir=st_dir)
    straight.run()
    assert _crowd_updates(straight), "空回り(混雑由来の state_update が 1 件も無い)"

    d = tmp_path / "cw_rs"
    s1 = Simulation(_cfg_of("cw_rs", _SPLIT, every=_SPLIT), out_dir=d)
    for step in range(_SPLIT):
        scheduler.run_step(s1, step)
    checkpoint.save(s1, _SPLIT, d / "checkpoint" / f"ckpt-{_SPLIT:06d}.pkl.gz")
    s1.logger.flush_segment()

    s2 = Simulation(_cfg_of("cw_rs", _TOTAL, every=_SPLIT), out_dir=d)
    s2.run(resume_from=d)
    assert _rows(d) == _rows(st_dir), "CRWD ON の resume が straight と不一致"


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
    sim = _sim(tmp_path, name, n=30, steps=144,
               **{**_CRWD, "controls.mode": "compute_matched", "k.writeback": writeback})
    sim.llm = _FixedLLM(json.dumps({"action": "speak", "text": "x"}, ensure_ascii=False))
    sim.run()
    return sim


def test_call_count_is_k_invariant(tmp_path):
    """CRWD は在館数(観測量)・時刻・config しか見ない=k=free と k=off で呼数が完全一致(R1)。"""
    free = _run_k(tmp_path, "cw_k_free", writeback="free")
    off = _run_k(tmp_path, "cw_k_off", writeback="off")
    assert free.llm.calls == off.llm.calls and free.llm.calls > 0, \
        f"CRWD の呼数が k に依存(R1 違反): free={free.llm.calls} off={off.llm.calls}"


# --------------------------------------------------------------------- 宣言(registry)
def test_registry_declares_the_toggle():
    from society import registry as R
    feat = R.BY_ID.get("commerce.crowding.enabled")
    assert feat is not None, "commerce.crowding.enabled が registry に未宣言"
    assert feat.repro_tier == "strict" and feat.affects_k is False
