"""承諾/拒否判断の内生化 第62バッチ フェーズ1(relations.endogenous_accept)のテスト。

方針(既存の鉄則を継承):
- OFF(既定): joint_invite 0 件・sim._endo_state 不在・L2 に joint_* 列なし・純粋既定と L1 完全一致
  (ゴールデンは test_scenario が固定。ここでは明示 OFF==純粋既定を144stepで固定)。
- always-draw: endo ON でも "joint" stream の decision 単位の draw は不変(veto 時も draw して破棄)。
  材料が全て fallback の条件では ON と OFF が joint_invite を除き L1 完全一致+draw 総数一致。
  veto で結果が変わる場合も**同一誘い手の stream 消費数は不変**(差は名簿=assigned 経由でのみ波及
  =stream は (agent,day) キーなので draw 列のズレは構造的に起きない)。
- 判定材料の各経路(帳簿衝突/誘い主との予定/計画 with/単独志向/明示キュー/fallback)を合成シナリオで直接検証。
- 合成式: prior_weight 両端(1.0=完全較正/0.0=完全内生)・clamp 後に gossip 減算(第61 不変則)。
- L2 4列の手計算一致・LLM 呼数 ON/OFF 一致(compute_matched)・同 seed 2 ラン一致・
  resume==straight(日境界跨ぎ)。検証は mock / _FixedLLM のみ(実LLM 禁止)。
"""
from __future__ import annotations

import json

import pyarrow.parquet as pq

from society import joint as _joint
from society import relations_endo as _endo
from society.config import load_config
from society.engine import checkpoint, scheduler
from society.engine.simulation import Simulation

# joint 機構が確実に発火する状態(test_joint の _JOINT_ON と同型)。
_JOINT_ON = {"joint.enabled": "true", "joint.daily_rate": "1.0",
             "joint.accept_base": "1.0", "household.enabled": "true",
             "relations.enabled": "true"}
_ENDO_ON = {**_JOINT_ON, "relations.endogenous_accept.enabled": "true"}


def _sim(tmp_path, name, n=20, steps=1, **ov):
    dot = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
           "run.seed=42", "model.backend=mock", "observer.snapshot_every=144"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _l1(sim, exclude=()):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events if e.kind not in exclude]


def _kind(sim, kind):
    return [e for e in sim.logger.events if e.kind == kind]


class _CountingHub:
    """"joint" stream の draw 消費を数えるプロキシ(他 stream は素通し)。"""

    def __init__(self, inner):
        self._inner = inner
        self.joint_draws = 0
        self.per_key: dict = {}

    def stream(self, *key):
        g = self._inner.stream(*key)
        if not (key and key[0] == "joint"):
            return g
        outer, kt = self, tuple(key)

        class _W:
            def _tick(self):
                outer.joint_draws += 1
                outer.per_key[kt] = outer.per_key.get(kt, 0) + 1

            def random(self, *a, **k):
                self._tick()
                return g.random(*a, **k)

            def integers(self, *a, **k):
                self._tick()
                return g.integers(*a, **k)

            def choice(self, *a, **k):
                self._tick()
                return g.choice(*a, **k)

        return _W()

    def key_name(self, *key):
        return self._inner.key_name(*key)

    @property
    def master_seed(self):
        return self._inner.master_seed


# --------------------------------------------------------------------- OFF 既定
def test_off_matches_pure_default(tmp_path):
    """明示 OFF と純粋既定が L1 完全一致(144 step)。joint_invite 0 件・_endo_state 不在・
    L2 に joint_* 列も無い(ゴールデン維持=既定バイト一致)。"""
    pure = _sim(tmp_path, "pure", steps=144)
    pure.run()
    off = _sim(tmp_path, "expl_off", steps=144,
               **{"relations.endogenous_accept.enabled": "false"})
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(endo seam が no-op でない)"
    assert not _kind(pure, "joint_invite") and not _kind(off, "joint_invite")
    assert getattr(off, "_endo_state", None) is None, "OFF なのに _endo_state が生えている"
    cols = pq.read_table(tmp_path / "expl_off" / "l2_metrics.parquet").column_names
    assert not [c for c in cols if c.startswith("joint_")], "OFF なのに L2 に joint_* 列"


def test_joint_on_endo_off_no_new_events(tmp_path):
    """joint ON + endo OFF は従来の joint 経路のまま: joint_invite 0 件・_endo_state 不在。"""
    sim = _sim(tmp_path, "joff", steps=144, **_JOINT_ON)
    sim.run()
    assert not _kind(sim, "joint_invite"), "endo OFF なのに joint_invite が出ている"
    assert getattr(sim, "_endo_state", None) is None


# --------------------------------------------------------------------- always-draw
def test_draw_count_invariant_when_all_fallback(tmp_path):
    """材料源(schedule/framework/dialog)を全て OFF にすると全誘いが fallback=較正確率のまま:
    ON と OFF で "joint" stream の draw 総数が一致し、joint_invite を除く L1 が完全一致する
    (always-draw 配線の直接検証: draw の追加/削減があれば成立グループが割れて即差が出る)。"""
    off = _sim(tmp_path, "d_off", steps=144, **_JOINT_ON)
    off.hub = _CountingHub(off.hub)
    off.run()
    on = _sim(tmp_path, "d_on", steps=144, **_ENDO_ON)
    on.hub = _CountingHub(on.hub)
    on.run()
    assert off.hub.joint_draws == on.hub.joint_draws > 0, \
        f"joint stream の draw 数が ON/OFF で不一致: {off.hub.joint_draws} vs {on.hub.joint_draws}"
    assert _l1(off) == _l1(on, exclude=("joint_invite",)), \
        "fallback のみの ON が OFF と L1 不一致(合成式の fallback が較正と等価でない)"
    inv = _kind(on, "joint_invite")
    assert inv and all(e.payload["basis"] == "fallback" for e in inv)


def _two_resident_sim(tmp_path, name, **ov):
    """居住者2人(a=誘い手, b=候補)だけの合成シナリオ(他は来街者化=編成対象外)。
    a→b は tier3(親友)で b が同伴候補の先頭になる。plan_day は未実行のまま返す。"""
    sim = _sim(tmp_path, name, n=20, steps=1, **ov)
    a, b = sim.agents[0], sim.agents[1]
    for x in sim.agents:
        x.visitor = x.id not in (a.id, b.id)
    a.mem.relations[b.id] = {"name": b.name, "count": 3, "last_step": 0,
                             "last": "", "closeness": 12.0, "tier": 3}
    b.mem.relations[a.id] = {"name": a.name, "count": 3, "last_step": 0,
                             "last": "", "closeness": 12.0, "tier": 3}
    return sim, a, b


def test_draw_count_invariant_per_inviter_under_veto(tmp_path):
    """veto(確定拒否)で結果が変わっても、**同一誘い手の "joint" stream 消費数は不変**
    (draw して破棄=conditionally-use)。差は名簿(assigned)経由でのみ波及する。"""
    ov = {**_JOINT_ON, "schedule.enabled": "true"}
    counts = {}
    key_a = None
    for name, endo in (("v_off", "false"), ("v_on", "true")):
        sim, a, b = _two_resident_sim(
            tmp_path, name, **{**ov, "relations.endogenous_accept.enabled": endo})
        key_a = ("joint", a.id, 0)
        # b の当日予定が全時間帯と衝突(band 語は全て拾われる=どの活動 band とも重なる)
        b.schedule = [{"day": 0, "when": w, "what": "", "place": "",
                       "with": [999], "src_step": 0}
                      for w in ("昼", "午後", "夕方", "夜")]
        sim.hub = _CountingHub(sim.hub)
        _joint.plan_day(sim, 0, 420)
        counts[name] = dict(sim.hub.per_key)
        if endo == "true":
            inv = _kind(sim, "joint_invite")
            assert inv and inv[0].payload["basis"] == "conflict" \
                and inv[0].payload["accepted"] is False \
                and inv[0].payload["p_final"] == 0.0, "veto が確定拒否になっていない"
            # ※ b は a の失敗後に自分の編成で a を誘い得る(名簿=assigned 経由の正当な波及)。
            #   stream は (agent,day) キーなので draw 列のズレは起きない=誘い手 a の消費数で固定する。
    assert counts["v_off"].get(key_a) == counts["v_on"].get(key_a) is not None, \
        f"誘い手 stream の draw 数が veto で変わった: {counts}"


# --------------------------------------------------------------------- 判定材料の各経路
def test_material_conflict_and_appointment(tmp_path):
    """帳簿: 他予定と時間帯衝突=reject(conflict)/ 誘い主との当日予定=accept(appointment)。
    両方あるときは誘い主との予定が勝つ(整合的)。when 空の予定は衝突を主張しない。"""
    sim, a, b = _two_resident_sim(tmp_path, "mat_sched",
                                  **{**_ENDO_ON, "schedule.enabled": "true"})
    sim._joint_day = 0
    cfg = _endo.cfg_of(sim)
    band = [1020, 1320]                                   # night band
    b.schedule = [{"day": 0, "when": "夜", "what": "", "place": "",
                   "with": [999], "src_step": 0}]
    assert _endo.decide_accept(sim, a, b, "meal_cafe", band, cfg) == ("reject", "conflict")
    # 時間帯が重ならない予定は衝突でない → fallback
    b.schedule = [{"day": 0, "when": "朝", "what": "", "place": "",
                   "with": [999], "src_step": 0}]
    assert _endo.decide_accept(sim, a, b, "meal_cafe", band, cfg) == (None, "fallback")
    # when 空=時間不明の予定は衝突を主張しない(正直な判定不能)
    b.schedule = [{"day": 0, "when": "", "what": "", "place": "",
                   "with": [999], "src_step": 0}]
    assert _endo.decide_accept(sim, a, b, "meal_cafe", band, cfg) == (None, "fallback")
    # 誘い主との当日予定=受諾材料(衝突予定と同居しても受諾が勝つ)
    b.schedule = [{"day": 0, "when": "夜", "what": "", "place": "",
                   "with": [999], "src_step": 0},
                  {"day": 0, "when": "", "what": "", "place": "",
                   "with": [a.id], "src_step": 0}]
    assert _endo.decide_accept(sim, a, b, "meal_cafe", band, cfg) == ("accept", "appointment")
    # 過去日の予定は無関係
    b.schedule = [{"day": 5, "when": "夜", "what": "", "place": "",
                   "with": [999], "src_step": 0}]
    assert _endo.decide_accept(sim, a, b, "meal_cafe", band, cfg) == (None, "fallback")


def test_material_plan_with_and_solo(tmp_path):
    """前日 day_schedule: with に誘い主=accept(plan_with)/ 単独志向が過半=reject(solo_plan)。"""
    sim, a, b = _two_resident_sim(tmp_path, "mat_plan", **_ENDO_ON)
    sim._joint_day = 0
    cfg = _endo.cfg_of(sim)
    band = [1020, 1320]

    def item(what, with_=(), anchor=False):
        return {"intent": "", "cat": "discretionary", "what": what, "place": "",
                "when": "夜", "start_min": 1200, "dur_min": 60, "flex": "flexible",
                "with": list(with_), "alt": "", "anchor": anchor}

    b.day_schedule = [item("leisure", with_=[a.name])]     # with に誘い主名
    assert _endo.decide_accept(sim, a, b, "cinema", band, cfg) == ("accept", "plan_with")
    b.day_schedule = [item("walk"), item("home"), item("meal")]  # 単独志向 2/3=過半
    assert _endo.decide_accept(sim, a, b, "cinema", band, cfg) == ("reject", "solo_plan")
    b.day_schedule = [item("walk"), item("meal"), item("leisure")]  # 1/3=過半でない
    assert _endo.decide_accept(sim, a, b, "cinema", band, cfg) == (None, "fallback")
    # 勤務アンカーは志向の母数に入れない(walk 1/1 で過半)
    b.day_schedule = [item("work", anchor=True), item("walk")]
    assert _endo.decide_accept(sim, a, b, "cinema", band, cfg) == ("reject", "solo_plan")


def test_material_dialog_cue(tmp_path):
    """前日発話の明示キュー(prompts.dialog_history ON 時のみ): 被誘者自身の「(誘い主名)と…
    行きたい/行こう」型のみ accept。婉曲拒否(「行きたいのは山々ですが」)は判定しない=fallback。"""
    sim, a, b = _two_resident_sim(tmp_path, "mat_cue",
                                  **{**_ENDO_ON, "prompts.dialog_history": "true"})
    sim._joint_day = 0
    cfg = _endo.cfg_of(sim)
    band = [660, 1020]
    b._dialog_hist = {a.id: [(b.name, f"{a.name}と映画に行きたいな")]}
    assert _endo.decide_accept(sim, a, b, "cinema", band, cfg) == ("accept", "dialog_cue")
    b._dialog_hist = {a.id: [(b.name, f"{a.name}とは行きたいのは山々ですが…")]}
    # 婉曲拒否は拒否判定しない(明示キュー辞書に拒否方向が無い設計)。
    # ※上の文は「(名)と…行きたい」の受諾型にも一致しない(「とは」で助詞が違う)= fallback。
    assert _endo.decide_accept(sim, a, b, "cinema", band, cfg) == (None, "fallback")
    # 相手(誘い主)の発話は材料にしない(当人の意向のみ)
    b._dialog_hist = {a.id: [(a.name, f"{b.name}と映画に行きたいな")]}
    assert _endo.decide_accept(sim, a, b, "cinema", band, cfg) == (None, "fallback")
    # dialog_history OFF なら同じ履歴でも読まない
    sim2, a2, b2 = _two_resident_sim(tmp_path, "mat_cue_off", **_ENDO_ON)
    sim2._joint_day = 0
    b2._dialog_hist = {a2.id: [(b2.name, f"{a2.name}と映画に行きたいな")]}
    assert _endo.decide_accept(sim2, a2, b2, "cinema", band,
                               _endo.cfg_of(sim2)) == (None, "fallback")


def test_material_fallback_when_no_evidence(tmp_path):
    """材料が何も無ければ (None, "fallback")=較正確率へ後退。"""
    sim, a, b = _two_resident_sim(tmp_path, "mat_fb", **_ENDO_ON)
    sim._joint_day = 0
    assert _endo.decide_accept(sim, a, b, "meal_cafe", [1020, 1320],
                               _endo.cfg_of(sim)) == (None, "fallback")


# --------------------------------------------------------------------- 合成式
def test_compose_prior_weight_extremes():
    """prior_weight=1.0 は完全較正(p==p_calib)・0.0 は完全内生(p==p_endo)。単調・有界。"""
    full_calib = _endo.build_cfg({"enabled": True, "prior_weight": 1.0})
    full_endo = _endo.build_cfg({"enabled": True, "prior_weight": 0.0})
    for verdict, basis in (("accept", "plan_with"), ("reject", "solo_plan"),
                           (None, "fallback")):
        p, forced = _endo.compose(0.5, verdict, basis, full_calib)
        assert not forced and abs(p - 0.5) < 1e-12, "w=1.0 で p!=p_calib"
    p, _f = _endo.compose(0.5, "accept", "plan_with", full_endo)
    assert abs(p - 0.85) < 1e-12                       # p_endo = 0.5+0.35
    p, _f = _endo.compose(0.5, "reject", "solo_plan", full_endo)
    assert abs(p - 0.15) < 1e-12                       # p_endo = 0.5−0.35
    # 有界: p_calib が 1 超(accept_base+tier_bonus)でも clamp される
    half = _endo.build_cfg({"enabled": True, "prior_weight": 0.5})
    p, _f = _endo.compose(1.3, "accept", "plan_with", half)
    assert p == 1.0, "clamp されていない"
    # veto は prior_weight と無関係に確定拒否
    assert _endo.compose(1.3, "reject", "conflict", half) == (0.0, True)


def test_gossip_penalty_applied_after_clamp(tmp_path, monkeypatch):
    """gossip 減算は clamp 後=常に最後(第61 不変則)。mix が 1.0 に clamp される状況で
    p_final = 1.0 − gp になる(clamp 前に引くと 1.0 のままになり検出できる)。"""
    ov = {**_ENDO_ON, "schedule.enabled": "true"}
    sim, a, b = _two_resident_sim(tmp_path, "gp_last", **ov)
    b.schedule = [{"day": 0, "when": "", "what": "", "place": "",
                   "with": [a.id], "src_step": 0}]        # accept 材料(appointment)
    monkeypatch.setattr(_joint._gossip, "joint_penalty", lambda *_a: 0.15)
    _joint.plan_day(sim, 0, 420)
    inv = _kind(sim, "joint_invite")
    assert inv, "誘いが出ていない(シナリオ前提が崩れた)"
    e = inv[0].payload
    # p_calib = accept_base 1.0 + tier_bonus 0.3 = 1.3 / p_endo = 1.0 / mix = clamp(1.15) = 1.0
    assert e["verdict"] == "accept" and e["basis"] == "appointment"
    assert e["p_calib"] == 1.3
    assert e["p_final"] == 0.85, \
        f"gossip 減算が clamp 前に混入している(p_final={e['p_final']}, 期待 0.85)"


# --------------------------------------------------------------------- L2 4列
def test_l2_columns_match_hand_computation(tmp_path):
    """L2 の joint_accept_rate / joint_endo_share / joint_accept_calib_gap が当日の
    joint_invite からの手計算と一致し、joint_fulfill_rate が [0,1] に収まる。"""
    sim = _sim(tmp_path, "l2check", steps=144, **_ENDO_ON)
    sim.run()
    rows = pq.read_table(tmp_path / "l2check" / "l2_metrics.parquet").to_pylist()
    last = rows[-1]
    last_day = max(e.sim_min // 1440 for e in _kind(sim, "joint_invite"))
    inv = [e for e in _kind(sim, "joint_invite") if e.sim_min // 1440 == last_day]
    assert inv, "最終日の誘いが無い(daily_rate=1.0 の前提が崩れた)"
    n = len(inv)
    acc = sum(1 for e in inv if e.payload["accepted"])
    endo = sum(1 for e in inv if e.payload["verdict"] != "none")
    p_sum = sum(e.payload["p_calib"] for e in inv)
    assert abs(last["joint_accept_rate"] - round(acc / n, 6)) < 1e-9
    assert abs(last["joint_endo_share"] - round(endo / n, 6)) < 1e-9
    assert abs(last["joint_accept_calib_gap"] - round(acc / n - p_sum / n, 6)) < 1e-6
    assert 0.0 <= last["joint_fulfill_rate"] <= 1.0
    # fallback 率 = 1 − endo_share がタリーからも一致
    st = sim._endo_state
    assert st["invites"] == n and st["accepts"] == acc and st["endo"] == endo


def test_scalars_hand_computation(tmp_path):
    """scalars() の手計算一致(タリー直接注入): 4/2/3 件・履行 1/2。"""
    sim = _sim(tmp_path, "sc", steps=1, **_ENDO_ON)
    sim._endo_state = {"day": 0, "invites": 4, "accepts": 2, "endo": 3,
                       "p_calib_sum": 2.6, "accepted": {1, 2}, "fulfilled": {2}}
    s = _endo.scalars(sim)
    assert s == {"joint_accept_rate": 0.5, "joint_endo_share": 0.75,
                 "joint_accept_calib_gap": round(0.5 - 0.65, 6),
                 "joint_fulfill_rate": 0.5}


def test_fulfill_marked_on_colocation(tmp_path):
    """mark_fulfilled: 承諾者がランデブー POI で2人以上同席した時だけ fulfilled に入る。"""
    sim, a, b = _two_resident_sim(tmp_path, "ful", **_ENDO_ON)
    st = _endo.day_state(sim, 0)
    st["accepted"].add(b.id)
    grp = {"members": [a.id, b.id], "poi": "X", "band": [0, 1440], "logged": False}
    a.node, b.node = "X", "Y"                       # 1人だけ=同席でない
    _endo.mark_fulfilled(sim, grp, st)
    assert not st["fulfilled"]
    b.node = "X"                                    # 2人同席
    _endo.mark_fulfilled(sim, grp, st)
    assert st["fulfilled"] == {b.id}                # 承諾者のみ(誘い手は数えない)


# --------------------------------------------------------------------- 決定論・R1・resume
def test_on_deterministic(tmp_path):
    """endo ON(全材料源つき)同 seed 2 ラン で L1 完全一致。"""
    ov = {**_ENDO_ON, "schedule.enabled": "true",
          "planning.framework.enabled": "true", "prompts.dialog_history": "true"}
    a = _sim(tmp_path, "det_a", steps=144, **ov)
    a.run()
    b = _sim(tmp_path, "det_b", steps=144, **ov)
    b.run()
    assert _l1(a) == _l1(b), "endo ON の決定論が崩れている"


class _FixedLLM:
    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.hits = 0

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        self.calls += 1
        return self.response, str(self.calls), False


def test_llm_call_count_k_invariant(tmp_path):
    """endo ON のまま compute_matched 下で k=free と k=off の generate 呼数が完全一致(R1)。"""
    def _run(name, writeback):
        sim = _sim(tmp_path, name, steps=100,
                   **{**_ENDO_ON, "controls.mode": "compute_matched",
                      "k.writeback": writeback})
        sim.llm = _FixedLLM(json.dumps({"action": "speak", "text": "x"},
                                       ensure_ascii=False))
        sim.run()
        return sim
    free = _run("k_free", "free")
    off = _run("k_off", "off")
    assert free.llm.calls == off.llm.calls > 0, \
        f"endo の呼数が k に依存(R1 違反): free={free.llm.calls} off={off.llm.calls}"


def test_resume_matches_straight_across_day_boundary(tmp_path):
    """endo(+joint+relations+schedule)ON の resume==straight(日境界跨ぎ=split105>境界102)。
    joint_day/joint_groups/joint_total/endo_state の checkpoint 中央管理を固定する。"""
    ov = {**_ENDO_ON, "schedule.enabled": "true"}

    def _cfg(name, n_steps, **extra):
        dot = ["run.seed=42", "run.n_agents=20", f"run.n_steps={n_steps}",
               f"run.name={name}", "model.backend=mock"]
        dot += [f"{k}={v}" for k, v in {**ov, **extra}.items()]
        return load_config(dot)

    straight_dir = tmp_path / "e_straight"
    Simulation(_cfg("e_straight", 120), out_dir=straight_dir).run()
    d = tmp_path / "e_resumed"
    split, total = 105, 120
    sim1 = Simulation(_cfg("e_resumed", split,
                           **{"observer.checkpoint_every": split}), out_dir=d)
    for step in range(split):
        scheduler.run_step(sim1, step)
    checkpoint.save(sim1, split, d / "checkpoint" / f"ckpt-{split:06d}.pkl.gz")
    sim1.logger.flush_segment()
    sim2 = Simulation(_cfg("e_resumed", total,
                           **{"observer.checkpoint_every": split}), out_dir=d)
    sim2.run(resume_from=d)
    for stem in ("l1_events", "l2_metrics", "l3_snapshots"):
        sa = pq.read_table(straight_dir / f"{stem}.parquet").to_pylist()
        sb = pq.read_table(d / f"{stem}.parquet").to_pylist()
        assert sa == sb, f"{stem} 不一致(endo resume)"
    # 直接検証: _joint_day / _endo_state が round-trip で復元される(空回り防止)
    assert getattr(sim1, "_joint_day", -1) >= 1, "日境界が未処理(テスト前提が崩れた)"
    sim3 = Simulation(_cfg("e_inspect", split,
                           **{"observer.checkpoint_every": split}),
                      out_dir=tmp_path / "e_inspect")
    checkpoint.load(sim3, d / "checkpoint" / f"ckpt-{split:06d}.pkl.gz")
    assert sim3._joint_day == sim1._joint_day
    assert sim3._endo_state == sim1._endo_state
    assert sim3._joint_total == sim1._joint_total
