"""誘う相手の内生選抜 第64バッチ フェーズ3(relations.endogenous_invite)のテスト。

方針(第62 test_endogenous_accept の鉄則を継承):
- OFF(既定): 純粋既定と L1 完全一致・"joint" stream 消費数・**候補順**とも従来と完全一致・
  sim._invite_state 不在・L2 に invite_* 列なし(ゴールデンは test_scenario が固定)。
- ON: 候補集合の並べ替え・拡張のみ(枠組み不変)。①resolve_with 最優先のまま ②明示キュー相手を
  次点挿入 ③closeness 降順 ④弱い紐帯枠(tier=1・(agent,day) 安定ハッシュ=乱数ゼロ)は末尾。
  weak_tie_slots=0 で従来順+resolve_with のみ。
- source フィールドの整合(経路ラベル=実際の選抜元)・accept 単独 ON の payload に source 無し。
- L2 2列の手計算一致・LLM 呼数 ON/OFF 一致(compute_matched)・同 seed 2 ラン一致・
  resume==straight(_invite_state の checkpoint 中央管理)。検証は mock のみ(実LLM 禁止)。
実験投入はフェーズ2 phase3_go ゲート(実装と実験実施の分離=計画書§4)。ここでは実装のみを固定。
"""
from __future__ import annotations

import json

import pyarrow.parquet as pq

from society import joint as _joint
from society import relations_endo as _endo
from society.config import load_config
from society.engine import checkpoint, scheduler
from society.engine.simulation import Simulation

_JOINT_ON = {"joint.enabled": "true", "joint.daily_rate": "1.0",
             "joint.accept_base": "1.0", "household.enabled": "true",
             "relations.enabled": "true"}
_INV_ON = {**_JOINT_ON, "relations.endogenous_invite.enabled": "true"}
_BOTH_ON = {**_INV_ON, "relations.endogenous_accept.enabled": "true"}


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
    """"joint" stream の draw 消費を数えるプロキシ(test_endogenous_accept と同型)。"""

    def __init__(self, inner):
        self._inner = inner
        self.joint_draws = 0

    def stream(self, *key):
        g = self._inner.stream(*key)
        if not (key and key[0] == "joint"):
            return g
        outer = self

        class _W:
            def random(self, *a, **k):
                outer.joint_draws += 1
                return g.random(*a, **k)

            def integers(self, *a, **k):
                outer.joint_draws += 1
                return g.integers(*a, **k)

            def choice(self, *a, **k):
                outer.joint_draws += 1
                return g.choice(*a, **k)

        return _W()

    def key_name(self, *key):
        return self._inner.key_name(*key)

    @property
    def master_seed(self):
        return self._inner.master_seed


def _rel(x, y, clo):
    """x→y の relations エントリ(closeness=clo)を双方向で張る。"""
    x.mem.relations[y.id] = {"name": y.name, "count": 3, "last_step": 0,
                             "last": "", "closeness": float(clo),
                             "tier": 3 if clo >= 12 else (2 if clo >= 5 else 1)}
    y.mem.relations[x.id] = {"name": x.name, "count": 3, "last_step": 0,
                             "last": "", "closeness": float(clo),
                             "tier": 3 if clo >= 12 else (2 if clo >= 5 else 1)}


def _synth_sim(tmp_path, name, **ov):
    """居住者 a(誘い手)+b(親友12)+c(友人6)+d/e(知人3=tier1)の合成シナリオ。
    他は来街者化=編成対象外。plan_day は未実行のまま返す。"""
    sim = _sim(tmp_path, name, n=20, steps=1, **ov)
    a, b, c, d, e = sim.agents[:5]
    for x in sim.agents:
        x.visitor = x.id not in (a.id, b.id, c.id, d.id, e.id)
    _rel(a, b, 12.0)     # tier3 親友
    _rel(a, c, 6.0)      # tier2 友人
    _rel(a, d, 3.0)      # tier1 知人(弱い紐帯)
    _rel(a, e, 3.0)      # tier1 知人(弱い紐帯)
    return sim, a, b, c, d, e


# --------------------------------------------------------------------- OFF 既定
def test_off_matches_pure_default(tmp_path):
    """明示 OFF と純粋既定が L1 完全一致(144 step)。_invite_state 不在・L2 に invite_* 列なし。"""
    pure = _sim(tmp_path, "pure", steps=144)
    pure.run()
    off = _sim(tmp_path, "expl_off", steps=144,
               **{"relations.endogenous_invite.enabled": "false"})
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(invite seam が no-op でない)"
    assert getattr(off, "_invite_state", None) is None, "OFF なのに _invite_state が生えている"
    cols = pq.read_table(tmp_path / "expl_off" / "l2_metrics.parquet").column_names
    assert not [c for c in cols if c.startswith("invite_")], "OFF なのに L2 に invite_* 列"


def test_off_joint_stream_and_l1_unchanged(tmp_path):
    """joint ON + invite 明示 OFF が invite 未指定と L1 完全一致し "joint" stream の draw 総数も
    一致する(invite OFF は候補順・消費とも従来と完全同一=seam の直接検証)。"""
    base = _sim(tmp_path, "s_base", steps=144, **_JOINT_ON)
    base.hub = _CountingHub(base.hub)
    base.run()
    off = _sim(tmp_path, "s_off", steps=144,
               **{**_JOINT_ON, "relations.endogenous_invite.enabled": "false"})
    off.hub = _CountingHub(off.hub)
    off.run()
    assert base.hub.joint_draws == off.hub.joint_draws > 0, \
        f"joint stream の draw 数が invite OFF で変わった: {base.hub.joint_draws} vs {off.hub.joint_draws}"
    assert _l1(base) == _l1(off), "invite 明示 OFF が未指定と L1 不一致"


def test_off_candidate_order_unchanged(tmp_path):
    """_companions: 従来呼び出し(icfg なし)と disabled icfg 付き呼び出しが完全同一の候補順。"""
    sim, a, b, c, d, e = _synth_sim(tmp_path, "ord_off", **_JOINT_ON)
    legacy = _joint._companions(sim, a, sim.jointcfg, set())
    assert legacy == [b.id, c.id], "従来の friend 経路(closeness 降順)の前提が崩れた"
    srcs: dict = {}
    disabled = _joint._companions(sim, a, sim.jointcfg, set(), "friend",
                                  _endo.build_invite_cfg({"enabled": False}), srcs)
    assert disabled == legacy, "disabled icfg 付きで候補順が変わった(OFF 純度違反)"


# --------------------------------------------------------------------- ON: 並べ替え・拡張
def test_on_dialog_cue_promoted(tmp_path):
    """明示キュー相手(誘い手自身の「(相手名)と…行きたい」)が closeness 順より上位に挿入される。
    hedge(婉曲)共起の発話はキューに数えない=従来順のまま。"""
    ov = {**_INV_ON, "prompts.dialog_history": "true"}
    sim, a, b, c, d, e = _synth_sim(tmp_path, "cue_on", **ov)
    icfg = _endo.invite_cfg_of(sim)
    # a 自身の発話に c への明示キュー → c が b(closeness 12)より上位へ
    a._dialog_hist = {c.id: [(a.name, f"{c.name}と映画に行きたいな")]}
    srcs: dict = {}
    cands = _joint._companions(sim, a, sim.jointcfg, set(), "friend", icfg, srcs)
    assert cands[0] == c.id and srcs[c.id] == "dialog_cue", \
        f"明示キュー相手が上位に来ない: {cands} {srcs}"
    assert cands[1] == b.id and srcs[b.id] == "closeness"
    # hedge 共起(「…行きたいのは山々ですが」)はキューに数えない → 従来順
    a._dialog_hist = {c.id: [(a.name, f"{c.name}と行きたいのは山々ですが難しい")]}
    srcs2: dict = {}
    cands2 = _joint._companions(sim, a, sim.jointcfg, set(), "friend", icfg, srcs2)
    assert cands2[:2] == [b.id, c.id], "hedge 共起の発話が明示キューに誤検出された"
    # 相手(他人)の発話は材料にしない(誘い手自身の意向のみ)
    a._dialog_hist = {c.id: [(c.name, f"{c.name}と映画に行きたいな")]}
    cands3 = _joint._companions(sim, a, sim.jointcfg, set(), "friend", icfg, {})
    assert cands3[:2] == [b.id, c.id], "他人の発話がキューに使われた"


def test_on_weak_tie_slot(tmp_path):
    """弱い紐帯枠: tier=1(知人)から決定論選択され候補**末尾**に入る。resolve_with/friends は
    従来位置のまま。weak_tie_slots=0 なら従来順+resolve_with のみ。"""
    sim, a, b, c, d, e = _synth_sim(tmp_path, "wt_on", **_INV_ON)
    icfg = _endo.invite_cfg_of(sim)
    srcs: dict = {}
    cands = _joint._companions(sim, a, sim.jointcfg, set(), "friend", icfg, srcs)
    assert cands[:2] == [b.id, c.id], "友人部分の従来順(closeness 降順)が保たれていない"
    assert len(cands) == 3 and cands[2] in (d.id, e.id), \
        f"弱い紐帯枠(1枠)が末尾に入っていない: {cands}"
    assert srcs[cands[2]] == "weak_tie"
    # slots=0 → 弱い紐帯なし=従来と同一
    sim0, a0, b0, c0, d0, e0 = _synth_sim(
        tmp_path, "wt_zero",
        **{**_INV_ON, "relations.endogenous_invite.weak_tie_slots": "0"})
    cands0 = _joint._companions(sim0, a0, sim0.jointcfg, set(), "friend",
                                _endo.invite_cfg_of(sim0), {})
    assert cands0 == [b0.id, c0.id], f"slots=0 で従来順にならない: {cands0}"


def test_weak_tie_pick_deterministic_and_rotating(tmp_path):
    """弱い紐帯枠の選択は (agent,day) 安定ハッシュ=同 day 同一・day 替わりで別の知人へ回る
    (乱数ゼロ=stream 無風)。exclude された知人は選ばれない。"""
    sim, a, b, c, d, e = _synth_sim(tmp_path, "wt_det", **_INV_ON)
    p0 = _endo.weak_tie_candidates(sim, a, 0, 1, set())
    p0b = _endo.weak_tie_candidates(sim, a, 0, 1, set())
    p1 = _endo.weak_tie_candidates(sim, a, 1, 1, set())
    assert p0 == p0b and len(p0) == 1, "同 (agent,day) で選択が揺れた"
    assert p1 != p0, "day 替わりで探索枠が回らない(pool=2 で同一選択)"
    assert set(p0 + p1) == {d.id, e.id}, "tier=1 以外が弱い紐帯枠に入った"
    assert _endo.weak_tie_candidates(sim, a, 0, 1, {d.id, e.id}) == [], \
        "exclude された知人が選ばれた"
    # tier2 以上(友人・親友)は弱い紐帯枠の pool に入らない
    assert b.id not in p0 + p1 and c.id not in p0 + p1


# --------------------------------------------------------------------- source 整合
def test_source_field_matches_selection_path(tmp_path):
    """joint_invite.source が実際の選抜元と一致する(plan_with / weak_tie / closeness)。
    タリー(_invite_state)も同じ内訳になる。"""
    ov = {**_BOTH_ON, "planning.framework.enabled": "true"}
    sim, a, b, c, d, e = _synth_sim(tmp_path, "src_ok", **ov)
    # c/e を来街者化=候補は b(親友)+d(知人)のみに絞って経路を確定させる
    c.visitor = True
    e.visitor = True
    a.day_schedule = [{"intent": "", "cat": "discretionary", "what": "leisure",
                       "place": "", "when": "夜", "start_min": 1200, "dur_min": 60,
                       "flex": "flexible", "with": [b.name], "alt": "",
                       "anchor": False}]
    _joint.plan_day(sim, 0, 420)
    inv = _kind(sim, "joint_invite")
    by_invitee = {e2.payload["invitee"]: e2.payload for e2 in inv
                  if e2.agent_id == a.id}
    assert by_invitee, "誘いが出ていない(シナリオ前提が崩れた)"
    assert by_invitee[b.id]["source"] == "plan_with", \
        f"resolve_with 相手の source が plan_with でない: {by_invitee[b.id]}"
    assert d.id in by_invitee and by_invitee[d.id]["source"] == "weak_tie", \
        f"弱い紐帯枠の source が weak_tie でない: {by_invitee}"
    st = sim._invite_state
    n_a = len(by_invitee)
    assert st["invites"] >= n_a and st["weak_tie"] >= 1 and st["endo"] >= 1, \
        f"タリーが source 内訳と不整合: {st}"


def test_no_source_when_invite_off(tmp_path):
    """accept 単独 ON(invite OFF)の joint_invite payload に source キーが無い=第62 の
    イベント形とバイト一致のまま。"""
    sim, a, b, c, d, e = _synth_sim(
        tmp_path, "src_off",
        **{**_JOINT_ON, "relations.endogenous_accept.enabled": "true"})
    _joint.plan_day(sim, 0, 420)
    inv = _kind(sim, "joint_invite")
    assert inv, "誘いが出ていない(シナリオ前提が崩れた)"
    assert all("source" not in e2.payload for e2 in inv), \
        "invite OFF なのに payload に source が付いた"


# --------------------------------------------------------------------- L2 2列
def test_l2_columns_match_hand_computation(tmp_path):
    """L2 の invite_weak_tie_rate / invite_endo_share が最終日の joint_invite(source 付き)
    からの手計算と一致し、タリーとも一致する。"""
    sim = _sim(tmp_path, "l2inv", steps=144, **_BOTH_ON)
    sim.run()
    rows = pq.read_table(tmp_path / "l2inv" / "l2_metrics.parquet").to_pylist()
    last = rows[-1]
    assert "invite_weak_tie_rate" in last and "invite_endo_share" in last, \
        "invite ON なのに L2 に invite_* 列が無い"
    inv_all = _kind(sim, "joint_invite")
    assert inv_all, "誘いが無い(daily_rate=1.0 の前提が崩れた)"
    last_day = max(e.sim_min // 1440 for e in inv_all)
    inv = [e for e in inv_all if e.sim_min // 1440 == last_day]
    n = len(inv)
    wt = sum(1 for e in inv if e.payload["source"] == "weak_tie")
    endo = sum(1 for e in inv
               if e.payload["source"] in ("plan_with", "dialog_cue"))
    assert abs(last["invite_weak_tie_rate"] - round(wt / n, 6)) < 1e-9
    assert abs(last["invite_endo_share"] - round(endo / n, 6)) < 1e-9
    st = sim._invite_state
    assert st["invites"] == n and st["weak_tie"] == wt and st["endo"] == endo


def test_invite_only_without_accept(tmp_path):
    """invite ON + accept OFF: joint_invite イベントは出ない(第62 の記録経路は accept 側)が、
    L2 の invite_* 2列とタリーは独立に機能する(正直な限界=source 内訳の L1 観測は accept ON 時のみ)。"""
    sim = _sim(tmp_path, "inv_only", steps=144, **_INV_ON)
    sim.run()
    assert not _kind(sim, "joint_invite"), "accept OFF なのに joint_invite が出た"
    cols = pq.read_table(tmp_path / "inv_only" / "l2_metrics.parquet").column_names
    assert "invite_weak_tie_rate" in cols and "invite_endo_share" in cols
    assert not [c for c in cols if c.startswith("joint_")], \
        "accept OFF なのに joint_*(accept 4列)が出た"
    assert getattr(sim, "_invite_state", None) is not None
    assert sim._invite_state["invites"] > 0, "誘いタリーが積まれていない"


def test_invite_scalars_hand_computation(tmp_path):
    """invite_scalars() の手計算一致(タリー直接注入): 5 件中 weak_tie 1・endo 2。"""
    sim = _sim(tmp_path, "sc_inv", steps=1, **_INV_ON)
    sim._invite_state = {"day": 0, "invites": 5, "weak_tie": 1, "endo": 2}
    s = _endo.invite_scalars(sim)
    assert s == {"invite_weak_tie_rate": 0.2, "invite_endo_share": 0.4}
    # 誘い 0 件の日は両列 0.0
    sim._invite_state = {"day": 0, "invites": 0, "weak_tie": 0, "endo": 0}
    assert _endo.invite_scalars(sim) == {"invite_weak_tie_rate": 0.0,
                                         "invite_endo_share": 0.0}


# --------------------------------------------------------------------- 決定論・R1・resume
def test_on_deterministic(tmp_path):
    """invite+accept(全材料源つき)同 seed 2 ラン で L1 完全一致。"""
    ov = {**_BOTH_ON, "schedule.enabled": "true",
          "planning.framework.enabled": "true", "prompts.dialog_history": "true"}
    a = _sim(tmp_path, "det_a", steps=144, **ov)
    a.run()
    b = _sim(tmp_path, "det_b", steps=144, **ov)
    b.run()
    assert _l1(a) == _l1(b), "invite ON の決定論が崩れている"


class _FixedLLM:
    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.hits = 0

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        self.calls += 1
        return self.response, str(self.calls), False


def test_llm_call_count_k_invariant(tmp_path):
    """invite+accept ON のまま compute_matched 下で k=free と k=off の generate 呼数が完全一致(R1)。"""
    def _run(name, writeback):
        sim = _sim(tmp_path, name, steps=100,
                   **{**_BOTH_ON, "controls.mode": "compute_matched",
                      "k.writeback": writeback})
        sim.llm = _FixedLLM(json.dumps({"action": "speak", "text": "x"},
                                       ensure_ascii=False))
        sim.run()
        return sim
    free = _run("ik_free", "free")
    off = _run("ik_off", "off")
    assert free.llm.calls == off.llm.calls > 0, \
        f"invite の呼数が k に依存(R1 違反): free={free.llm.calls} off={off.llm.calls}"


def test_resume_matches_straight_across_day_boundary(tmp_path):
    """invite(+accept+joint+relations+schedule)ON の resume==straight(日境界跨ぎ=split105>境界102)。
    _invite_state の checkpoint 中央管理を固定する。"""
    ov = {**_BOTH_ON, "schedule.enabled": "true"}

    def _cfg(name, n_steps, **extra):
        dot = ["run.seed=42", "run.n_agents=20", f"run.n_steps={n_steps}",
               f"run.name={name}", "model.backend=mock"]
        dot += [f"{k}={v}" for k, v in {**ov, **extra}.items()]
        return load_config(dot)

    straight_dir = tmp_path / "i_straight"
    Simulation(_cfg("i_straight", 120), out_dir=straight_dir).run()
    d = tmp_path / "i_resumed"
    split, total = 105, 120
    sim1 = Simulation(_cfg("i_resumed", split,
                           **{"observer.checkpoint_every": split}), out_dir=d)
    for step in range(split):
        scheduler.run_step(sim1, step)
    checkpoint.save(sim1, split, d / "checkpoint" / f"ckpt-{split:06d}.pkl.gz")
    sim1.logger.flush_segment()
    sim2 = Simulation(_cfg("i_resumed", total,
                           **{"observer.checkpoint_every": split}), out_dir=d)
    sim2.run(resume_from=d)
    for stem in ("l1_events", "l2_metrics", "l3_snapshots"):
        sa = pq.read_table(straight_dir / f"{stem}.parquet").to_pylist()
        sb = pq.read_table(d / f"{stem}.parquet").to_pylist()
        assert sa == sb, f"{stem} 不一致(invite resume)"
    # 直接検証: _invite_state が round-trip で復元される(空回り防止)
    assert getattr(sim1, "_joint_day", -1) >= 1, "日境界が未処理(テスト前提が崩れた)"
    sim3 = Simulation(_cfg("i_inspect", split,
                           **{"observer.checkpoint_every": split}),
                      out_dir=tmp_path / "i_inspect")
    checkpoint.load(sim3, d / "checkpoint" / f"ckpt-{split:06d}.pkl.gz")
    assert sim3._invite_state == sim1._invite_state
    assert sim3._invite_state is not None
