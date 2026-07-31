"""ダンバー認知枠(関係の維持コスト)+ 忘却/再会 第75バッチ IDEA⑤(relations.dunbar)のテスト。

方針(第62/64/65 の関係内生化テストの鉄則を継承):
- OFF(既定): 純粋既定と L1 完全一致・"joint" stream 消費数一致・台帳に dormant キーが生えない・
  sim._dunbar_state 不在・L2 に列なし(ゴールデンは test_scenario が固定)。
- relations OFF なら dunbar ON でも完全 no-op(closeness が動く経路が前提)。
- ON: 上限超過で**最も弱い紐帯**から休眠(closeness 昇順・同値は id 昇順)・直近接触/同居人は
  保護・休眠は**退避であって削除でない**(count/last/記憶が残り closeness を退避)・再接触で
  割引つき再会・休眠は日次減衰の対象外。
- 第64 endogenous_invite との整合: 休眠は closeness 降順経路から自動的に外れ、弱い紐帯探索枠
  でのみ再会し得る(dunbar OFF では pool も選択も従来と完全同一)。
- L2 3列の状態一致・同 seed 2 ラン一致・resume==straight(_dunbar_state の checkpoint 中央管理)・
  LLM 呼数の k 不変。検証は mock のみ(実LLM 禁止)。
"""
from __future__ import annotations

import json

import pyarrow.parquet as pq

from society import dunbar
from society import joint as _joint
from society import registry as R
from society import relations as _relations
from society import relations_endo as _endo
from society.config import load_config
from society.engine import checkpoint, scheduler
from society.engine.simulation import Simulation

_REL_ON = {"relations.enabled": "true", "friend_graph.enabled": "true",
           "joint.enabled": "true", "household.enabled": "true"}
_DUN_ON = {**_REL_ON, "relations.dunbar.enabled": "true"}
# 拘束が実際に効く縮約(caps: close1 / friend1 / acquaint3 / total8)。既定 scale=0.34 は
# total=51 で N=20 の mock では原理的に効かない(相手の総数が上限に届かない)ため。
_TIGHT = {**_DUN_ON, "relations.dunbar.scale": "0.05"}


def _sim(tmp_path, name, n=20, steps=1, **ov):
    dot = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
           "run.seed=42", "model.backend=mock", "observer.snapshot_every=144"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _kind(sim, kind):
    return [e for e in sim.logger.events if e.kind == kind]


class _CountingHub:
    """"joint" stream の draw 消費を数えるプロキシ(test_endogenous_invite と同型)。"""

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


def _rel(x, y, clo, last_step=0, count=3):
    """x→y の関係台帳(closeness=clo)を張る(片方向。テストの直接注入)。"""
    x.mem.relations[y.id] = {"name": y.name, "count": count, "last_step": int(last_step),
                             "last": "", "closeness": float(clo),
                             "tier": 3 if clo >= 12 else (2 if clo >= 5 else
                                                          (1 if clo >= 2 else 0))}
    return x.mem.relations[y.id]


def _synth(tmp_path, name, n=8, **ov):
    """居住者 a(主体)と b..h(相手)だけの合成シナリオ(plan_day 未実行)。"""
    sim = _sim(tmp_path, name, n=n, steps=1, **ov)
    for x in sim.agents:
        x.visitor = False
        x.mem.relations = {}
        x.housemates = []          # 世帯の同居人は保護対象=注入した関係だけを見るため空にする
    return sim, sim.agents[0], sim.agents[1:]


# --------------------------------------------------------------------- OFF 既定
def test_off_matches_pure_default(tmp_path):
    """明示 OFF と純粋既定が L1 完全一致(144 step)。_dunbar_state 不在・L2 に列なし。"""
    pure = _sim(tmp_path, "pure", steps=144)
    pure.run()
    off = _sim(tmp_path, "expl_off", steps=144,
               **{"relations.dunbar.enabled": "false"})
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(dunbar seam が no-op でない)"
    assert getattr(off, "_dunbar_state", None) is None, "OFF なのに _dunbar_state が生えている"
    cols = pq.read_table(tmp_path / "expl_off" / "l2_metrics.parquet").column_names
    for c in ("active_relations_mean", "dormant_total", "rekindle_total"):
        assert c not in cols, f"OFF なのに L2 に {c} 列"


def test_off_relations_on_l1_and_joint_draws_unchanged(tmp_path):
    """relations/friend_graph/joint ON + dunbar 明示 OFF が未指定と L1 完全一致し、
    "joint" stream の draw 総数も一致する(OFF は候補順・消費とも従来と完全同一)。
    台帳に dormant キーが 1 つも生えないことも直接固定する。"""
    base = _sim(tmp_path, "d_base", steps=144, **_REL_ON)
    base.hub = _CountingHub(base.hub)
    base.run()
    off = _sim(tmp_path, "d_off", steps=144,
               **{**_REL_ON, "relations.dunbar.enabled": "false"})
    off.hub = _CountingHub(off.hub)
    off.run()
    assert base.hub.joint_draws == off.hub.joint_draws > 0, \
        f"joint draw 数が dunbar OFF で変わった: {base.hub.joint_draws} vs {off.hub.joint_draws}"
    assert _l1(base) == _l1(off), "dunbar 明示 OFF が未指定と L1 不一致"
    keys = {k for a in off.agents for rel in a.mem.relations.values() for k in rel}
    assert not (keys & {"dormant", "dormant_closeness", "dormant_step"}), \
        f"OFF なのに休眠キーが台帳に生えた: {sorted(keys)}"


def test_requires_relations_enabled(tmp_path):
    """relations OFF のまま dunbar ON にしても完全 no-op(closeness が動く経路が前提)。"""
    base = _sim(tmp_path, "nr_base", steps=144)
    base.run()
    on = _sim(tmp_path, "nr_on", steps=144, **{"relations.dunbar.enabled": "true"})
    on.run()
    assert not dunbar.enabled(on), "relations OFF なのに dunbar が実効になっている"
    assert _l1(base) == _l1(on), "relations OFF + dunbar ON が既定と L1 不一致"
    assert getattr(on, "_dunbar_state", None) is None


# --------------------------------------------------------------------- 上限の導出
def test_caps_derivation_pure_function():
    """caps_of: 既定 (5,15,50,150)×0.34 → 2/5/17/51。入れ子の単調性と下限 1 を保証。"""
    cfg = dunbar.build_cfg({})
    assert dunbar.caps_of(cfg) == {"close": 2, "friend": 5, "acquaint": 17, "total": 51}
    tight = dunbar.build_cfg({"scale": 0.05})
    caps = dunbar.caps_of(tight)
    assert caps == {"close": 1, "friend": 1, "acquaint": 3, "total": 8}, caps
    # 入れ子の単調性: 内側 ≤ 外側(層値が逆転していても崩れない)
    weird = dunbar.build_cfg({"close": 100, "friend": 3, "acquaint": 4,
                              "total": 5, "scale": 1.0})
    c = dunbar.caps_of(weird)
    assert c["close"] <= c["friend"] <= c["acquaint"] <= c["total"]
    # scale=0 でも下限 1(上限ゼロで全滅させない)
    assert dunbar.caps_of(dunbar.build_cfg({"scale": 0.0}))["total"] == 1


# --------------------------------------------------------------------- 休眠(忘却)
def test_weakest_tie_goes_dormant_first(tmp_path):
    """上限超過で closeness の**最も弱い**紐帯から休眠する(強い紐帯は残る)。"""
    sim, a, others = _synth(tmp_path, "weak", n=8, **_TIGHT)
    for i, o in enumerate(others[:6]):           # closeness 1..6(弱い順=id 順ではない)
        _rel(a, o, float(6 - i))                 # b:6 c:5 d:4 e:3 f:2 g:1
    sim.dunbarcfg = dunbar.build_cfg({"enabled": True, "close": 3, "friend": 3,
                                      "acquaint": 3, "total": 3, "scale": 1.0})
    n = dunbar.enforce(sim, a, 300, 1440 * 3)    # day3=直近接触(day0)から離れている
    assert n == 3, f"超過 3 件が休眠になっていない: {n}"
    assert dunbar.active_ids(a) == sorted(o.id for o in others[:3]), \
        "強い上位3件(closeness 6/5/4)が残っていない"
    assert dunbar.dormant_ids(a) == sorted(o.id for o in others[3:6]), \
        "弱い下位3件(closeness 3/2/1)が休眠になっていない"
    ev = _kind(sim, "relation_dormant")
    assert len(ev) == 3 and {e.payload["other"] for e in ev} == set(dunbar.dormant_ids(a))
    assert ev[0].payload["gap_days"] == 3, f"gap_days が最終接触からの日数でない: {ev[0].payload}"


def test_tie_break_is_agent_id(tmp_path):
    """closeness 同値のときは相手 id 昇順(=小さい id から)で落ちる(乱数ゼロの全順序)。"""
    sim, a, others = _synth(tmp_path, "tie", n=8, **_TIGHT)
    for o in others[:4]:
        _rel(a, o, 2.0)                          # 全員同値
    sim.dunbarcfg = dunbar.build_cfg({"enabled": True, "total": 2, "close": 2,
                                      "friend": 2, "acquaint": 2, "scale": 1.0})
    dunbar.enforce(sim, a, 300, 1440 * 3)
    ids = sorted(o.id for o in others[:4])
    assert dunbar.dormant_ids(a) == ids[:2], "同値のタイ破りが id 昇順でない"


def test_recent_contact_and_housemate_are_protected(tmp_path):
    """直近 keep_days 以内に接触した相手と同居人は休眠にしない(保護が優先=上限超過のまま)。"""
    sim, a, others = _synth(tmp_path, "prot", n=8, **_TIGHT)
    b, c, d, e = others[:4]
    _rel(a, b, 1.0, last_step=300)               # 当日接触=保護(keep_days=0)
    _rel(a, c, 1.5)                              # 旧接触=淘汰対象(最弱)
    _rel(a, d, 2.0)                              # 旧接触=淘汰対象
    _rel(a, e, 0.5)                              # 同居人=保護
    a.housemates = [e.id]
    sim.dunbarcfg = dunbar.build_cfg({"enabled": True, "total": 1, "close": 1,
                                      "friend": 1, "acquaint": 1, "scale": 1.0})
    dunbar.enforce(sim, a, 300, 1440 * 2)        # day2: b は同日(step300)接触=保護
    assert dunbar.dormant_ids(a) == sorted([c.id, d.id]), \
        f"保護対象が落ちた/淘汰対象が残った: dormant={dunbar.dormant_ids(a)}"
    assert set(dunbar.active_ids(a)) == {b.id, e.id}
    assert len(dunbar.active_ids(a)) > 1, "保護で上限を超えたままになる限界が再現していない"


def test_dormant_is_stash_not_delete(tmp_path):
    """休眠は**退避**であって削除でない: 台帳エントリ・count・last・記憶が残り closeness を退避。"""
    sim, a, others = _synth(tmp_path, "stash", n=8, **_TIGHT)
    b, c = others[0], others[1]
    _rel(a, b, 9.0, count=7)
    _rel(a, c, 1.0, count=5)
    a.mem.relations[c.id]["last"] = "こんにちは"
    a.mem.observe(0, f"{c.name}と話した")
    sim.dunbarcfg = dunbar.build_cfg({"enabled": True, "total": 1, "close": 1,
                                      "friend": 1, "acquaint": 1, "scale": 1.0})
    dunbar.enforce(sim, a, 300, 1440 * 3)
    rel = a.mem.relations[c.id]
    assert rel["dormant"] is True and rel["closeness"] == 0.0 and rel["tier"] == 0
    assert rel["dormant_closeness"] == 1.0 and rel["dormant_step"] == 300
    assert rel["count"] == 5 and rel["last"] == "こんにちは", "台帳の履歴が消えた(削除になっている)"
    assert any(c.name in ep.text for ep in a.mem.buffer), "記憶(episodes/buffer)が消えた"
    assert _relations.tier_of(rel["closeness"], sim.relationscfg) == 0


def test_dormant_relation_is_not_decayed(tmp_path):
    """休眠関係は relations.decay_day の長期不在減衰を受けない(退避値のまま凍結)。"""
    sim, a, others = _synth(tmp_path, "nodecay", n=8, **_TIGHT)
    b, c = others[0], others[1]
    _rel(a, b, 9.0)
    _rel(a, c, 1.0)
    sim.dunbarcfg = dunbar.build_cfg({"enabled": True, "total": 1, "close": 1,
                                      "friend": 1, "acquaint": 1, "scale": 1.0})
    dunbar.enforce(sim, a, 300, 1440 * 3)
    _relations.decay_day(sim, sim.relationscfg, 300, 1440 * 3)
    assert a.mem.relations[c.id]["closeness"] == 0.0, "休眠関係が減衰で負に振れた"
    assert a.mem.relations[c.id]["dormant_closeness"] == 1.0, "退避値が減衰で書き換わった"
    assert a.mem.relations[b.id]["closeness"] < 9.0, "活性関係の従来の減衰が止まっている"


# --------------------------------------------------------------------- 再会
def test_rekindle_restores_discounted(tmp_path):
    """再接触で休眠前 closeness × rekindle_discount に復元し relation_rekindle を出す。"""
    sim, a, others = _synth(tmp_path, "rek", n=8, **_TIGHT)
    b, c = others[0], others[1]
    _rel(a, b, 9.0)
    _rel(a, c, 8.0)
    sim.dunbarcfg = dunbar.build_cfg({"enabled": True, "total": 1, "close": 1,
                                      "friend": 1, "acquaint": 1, "scale": 1.0,
                                      "rekindle_discount": 0.5})
    dunbar.enforce(sim, a, 300, 1440 * 3)
    assert dunbar.dormant_ids(a) == [c.id]
    dunbar.on_contact(sim, a, c.id, 320, 1440 * 4)
    rel = a.mem.relations[c.id]
    assert "dormant" not in rel and "dormant_closeness" not in rel, "休眠キーが残っている"
    assert rel["closeness"] == 4.0, f"割引復元(8.0×0.5)になっていない: {rel['closeness']}"
    assert rel["tier"] == _relations.tier_of(4.0, sim.relationscfg)
    ev = _kind(sim, "relation_rekindle")
    assert len(ev) == 1
    # gap_days は clock.day(休眠step)との差(既定 07:00 開始なので step300=day2・sim_min 4×1440=day4)
    assert ev[0].payload == {"other": c.id, "before": 8.0, "closeness": 4.0,
                             "tier": rel["tier"], "gap_days": 2}


def test_contact_rekindles_before_applying_delta(tmp_path):
    """engine seam の順序: 再会(復元)→ note_contact(交流分の加算)。復元値が増分を潰さない。"""
    sim, a, others = _synth(tmp_path, "order", n=8, **_TIGHT)
    b, c = others[0], others[1]
    _rel(a, b, 9.0)
    _rel(a, c, 8.0)
    sim.dunbarcfg = dunbar.build_cfg({"enabled": True, "total": 1, "close": 1,
                                      "friend": 1, "acquaint": 1, "scale": 1.0})
    dunbar.enforce(sim, a, 300, 1440 * 3)
    scheduler._contact(sim, a, c.id, c.name, "やあ", 1.0, 320, 1440 * 4)
    # 8.0×0.5(復元)+ pos_weight 1.0(この交流)= 5.0。復元だけなら 4.0・加算だけなら 1.0 になる
    # ので、この 1 値が「復元 → 加算」の順序を一意に固定する。
    assert a.mem.relations[c.id]["closeness"] == 5.0, a.mem.relations[c.id]
    assert a.mem.relations[c.id]["count"] == 4, "交流の count が積まれていない"


def test_joint_copresence_maintains_and_rekindles(tmp_path):
    """同席(joint)は維持行為: 活性なら last_step を進め、休眠なら再会させる(会話は捏造しない)。"""
    sim, a, others = _synth(tmp_path, "touch", n=8, **_TIGHT)
    b, c = others[0], others[1]
    _rel(a, b, 9.0)
    _rel(a, c, 8.0)
    _rel(b, a, 9.0)
    sim.dunbarcfg = dunbar.build_cfg({"enabled": True, "total": 1, "close": 1,
                                      "friend": 1, "acquaint": 1, "scale": 1.0})
    dunbar.enforce(sim, a, 300, 1440 * 3)
    assert dunbar.dormant_ids(a) == [c.id]
    dunbar.touch_group(sim, [a.id, b.id, c.id], 320, 1440 * 4)
    assert a.mem.relations[c.id]["closeness"] == 4.0, "同席で再会していない"
    assert a.mem.relations[b.id]["last_step"] == 320, "同席で活性関係が維持されていない"
    assert a.mem.relations[b.id]["count"] == 3, "同席が会話(count)を捏造している"
    # 台帳に無い相手には何もしない(同席だけでは知り合わない)
    assert others[2].id not in a.mem.relations


def test_day_phase_brings_everyone_within_cap(tmp_path):
    """日次バッチ: 全 agent が上限内へ収まり、L2 スカラーが焼き直される。"""
    sim, a, others = _synth(tmp_path, "dayp", n=8, **_TIGHT)
    for x in sim.agents:
        for y in sim.agents:
            if x.id != y.id:
                _rel(x, y, 2.0 + (y.id % 3))
    sim.dunbarcfg = dunbar.build_cfg({"enabled": True, "total": 2, "close": 2,
                                      "friend": 2, "acquaint": 2, "scale": 1.0})
    dunbar.day_phase(sim, 300, 1440 * 3)
    assert all(len(dunbar.active_ids(x)) <= 2 for x in sim.agents), "上限内に収まっていない"
    st = sim._dunbar_state
    assert st["active_mean"] == 2.0 and st["dormant_total"] == 5 * len(sim.agents)
    assert st["dormant_events"] == st["dormant_total"]
    assert dunbar.scalars(sim) == {"active_relations_mean": 2.0,
                                   "dormant_total": float(st["dormant_total"]),
                                   "rekindle_total": 0.0}


# --------------------------------------------------- 第64 誘い選抜との整合規則
def test_dormant_leaves_closeness_path_and_enters_weak_tie_pool(tmp_path):
    """休眠関係は closeness 降順の同伴候補から**自動的に**外れ(tier 0)、弱い紐帯探索枠に入る。"""
    ov = {**_TIGHT, "relations.endogenous_invite.enabled": "true"}
    sim, a, others = _synth(tmp_path, "inv_on", n=8, **ov)
    b, c, d = others[0], others[1], others[2]
    _rel(a, b, 12.0)                             # tier3 親友(残る)
    _rel(a, c, 6.0)                              # tier2 友人(休眠させる)
    _rel(a, d, 3.0)                              # tier1 知人(従来からの弱い紐帯)
    sim.dunbarcfg = dunbar.build_cfg({"enabled": True, "total": 2, "close": 2,
                                      "friend": 2, "acquaint": 2, "scale": 1.0})
    dunbar._make_dormant(sim, a, c.id, 300, 1440 * 3)
    icfg = _endo.invite_cfg_of(sim)
    srcs: dict = {}
    cands = _joint._companions(sim, a, sim.jointcfg, set(), "friend", icfg, srcs)
    assert c.id not in [oid for oid in cands if srcs.get(oid) == "closeness"], \
        "休眠関係が closeness 経路の同伴候補に残っている"
    assert cands[0] == b.id and srcs[b.id] == "closeness"
    pool = _endo.weak_tie_candidates(sim, a, 0, 5, set())
    assert c.id in pool, "休眠関係が弱い紐帯探索枠の pool に入っていない(再会経路が無い)"
    assert d.id in pool, "従来の tier1 知人が pool から外れた"
    assert b.id not in pool


def test_weak_tie_pool_unchanged_when_dunbar_off(tmp_path):
    """dunbar OFF では dormant フラグがあっても pool は従来どおり tier1 のみ(第64 の挙動不変)。"""
    ov = {**_REL_ON, "relations.endogenous_invite.enabled": "true"}
    sim, a, others = _synth(tmp_path, "inv_off", n=8, **ov)
    b, c, d = others[0], others[1], others[2]
    _rel(a, b, 12.0)
    _rel(a, c, 6.0)
    _rel(a, d, 3.0)
    a.mem.relations[c.id].update({"dormant": True, "dormant_closeness": 6.0,
                                  "dormant_step": 0, "closeness": 0.0, "tier": 0})
    assert not dunbar.enabled(sim)
    pool = _endo.weak_tie_candidates(sim, a, 0, 5, set())
    assert pool == [d.id], f"dunbar OFF で pool が変わった: {pool}"


def test_explicit_plan_reaches_dormant_partner(tmp_path):
    """明示的な意向(前日計画の with)は休眠相手にも通る(名指しは認知枠の淘汰より上位)。"""
    ov = {**_TIGHT, "relations.endogenous_invite.enabled": "true",
          "planning.framework.enabled": "true"}
    sim, a, others = _synth(tmp_path, "plan_dorm", n=8, **ov)
    b = others[0]
    _rel(a, b, 6.0)
    dunbar._make_dormant(sim, a, b.id, 300, 1440 * 3)
    a.day_schedule = [{"intent": "", "cat": "discretionary", "what": "leisure",
                       "place": "", "when": "夜", "start_min": 1200, "dur_min": 60,
                       "flex": "flexible", "with": [b.name], "alt": "", "anchor": False}]
    srcs: dict = {}
    cands = _joint._companions(sim, a, sim.jointcfg, set(), "friend",
                               _endo.invite_cfg_of(sim), srcs)
    assert cands and cands[0] == b.id and srcs[b.id] == "plan_with", \
        f"休眠相手が明示計画の同伴候補に上がらない: {cands} {srcs}"


# --------------------------------------------------------------------- L2 / ラン
def test_on_run_fires_events_and_l2_matches_state(tmp_path):
    """ON スモーク(288 step): relation_dormant/rekindle が発火し、L2 3列が state と一致し、
    日境界後の活性関係数が上限内(日内の一時超過は最大 1 日分)に収まる。"""
    sim = _sim(tmp_path, "on_run", n=20, steps=288, **_TIGHT)
    sim.run()
    dorm = _kind(sim, "relation_dormant")
    assert dorm, "ON なのに relation_dormant が 1 件も出ていない"
    st = sim._dunbar_state
    cap = dunbar.caps_of(dunbar.cfg_of(sim))["total"]
    assert st["dormant_events"] == len(dorm)
    assert st["rekindle_events"] == len(_kind(sim, "relation_rekindle"))
    rows = pq.read_table(tmp_path / "on_run" / "l2_metrics.parquet").to_pylist()
    last = rows[-1]
    assert last["active_relations_mean"] == st["active_mean"]
    assert last["dormant_total"] == float(st["dormant_total"])
    assert last["rekindle_total"] == float(st["rekindle_events"])
    assert st["active_mean"] <= cap, \
        f"日境界で焼いた活性関係数の平均が上限を超えている: {st['active_mean']} > {cap}"
    # 休眠は退避=台帳から消えない(記憶を消さない設計の直接確認)
    assert all("dormant_closeness" in rel
               for a in sim.agents for rel in a.mem.relations.values()
               if rel.get("dormant"))


def test_on_deterministic(tmp_path):
    """dunbar ON の同 seed 2 ランで L1 完全一致(乱数を1本も引かない=決定論)。"""
    a = _sim(tmp_path, "det_a", n=20, steps=288, **_TIGHT)
    a.run()
    b = _sim(tmp_path, "det_b", n=20, steps=288, **_TIGHT)
    b.run()
    assert _l1(a) == _l1(b), "dunbar ON の決定論が崩れている"


class _FixedLLM:
    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.hits = 0

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        self.calls += 1
        return self.response, str(self.calls), False


def test_llm_call_count_k_invariant(tmp_path):
    """dunbar ON のまま compute_matched 下で k=free と k=off の generate 呼数が完全一致(R1)。"""
    def _run(name, writeback):
        sim = _sim(tmp_path, name, n=20, steps=110,
                   **{**_TIGHT, "controls.mode": "compute_matched",
                      "k.writeback": writeback})
        sim.llm = _FixedLLM(json.dumps({"action": "speak", "text": "x"},
                                       ensure_ascii=False))
        sim.run()
        return sim
    free = _run("dk_free", "free")
    off = _run("dk_off", "off")
    assert free.llm.calls == off.llm.calls > 0, \
        f"dunbar の呼数が k に依存(R1 違反): free={free.llm.calls} off={off.llm.calls}"


def test_resume_matches_straight_across_day_boundary(tmp_path):
    """dunbar ON の resume==straight(日境界跨ぎ=split105>境界102)。_dunbar_state の
    checkpoint 中央管理と、休眠状態(agents pickle 同梱)の往復を固定する。"""
    # friend_graph は起動時 1 回の friend_graph_built を出す=resume 側の再構築で 1 件増えて
    # L1 比較が壊れる(この機能とは無関係の既知の性質)ため、この検収では外し、代わりに上限を
    # 1 まで絞って会話由来の関係だけで休眠を起こす。
    ov = {"relations.enabled": "true", "joint.enabled": "true",
          "household.enabled": "true", "relations.dunbar.enabled": "true",
          "relations.dunbar.scale": "0.005"}

    def _cfg(name, n_steps, **extra):
        dot = ["run.seed=42", "run.n_agents=20", f"run.n_steps={n_steps}",
               f"run.name={name}", "model.backend=mock"]
        dot += [f"{k}={v}" for k, v in {**ov, **extra}.items()]
        return load_config(dot)

    straight_dir = tmp_path / "d_straight"
    straight = Simulation(_cfg("d_straight", 120), out_dir=straight_dir)
    straight.run()
    assert _kind(straight, "relation_dormant"), "テスト前提が崩れた(休眠が起きていない)"
    d = tmp_path / "d_resumed"
    split, total = 105, 120
    sim1 = Simulation(_cfg("d_resumed", split,
                           **{"observer.checkpoint_every": split}), out_dir=d)
    for step in range(split):
        scheduler.run_step(sim1, step)
    checkpoint.save(sim1, split, d / "checkpoint" / f"ckpt-{split:06d}.pkl.gz")
    sim1.logger.flush_segment()
    sim2 = Simulation(_cfg("d_resumed", total,
                           **{"observer.checkpoint_every": split}), out_dir=d)
    sim2.run(resume_from=d)
    for stem in ("l1_events", "l2_metrics", "l3_snapshots"):
        sa = pq.read_table(straight_dir / f"{stem}.parquet").to_pylist()
        sb = pq.read_table(d / f"{stem}.parquet").to_pylist()
        assert sa == sb, f"{stem} 不一致(dunbar resume)"
    # 直接検証: _dunbar_state と休眠台帳が round-trip で復元される(空回り防止)
    sim3 = Simulation(_cfg("d_inspect", split,
                           **{"observer.checkpoint_every": split}),
                      out_dir=tmp_path / "d_inspect")
    checkpoint.load(sim3, d / "checkpoint" / f"ckpt-{split:06d}.pkl.gz")
    assert sim3._dunbar_state == sim1._dunbar_state is not None
    assert sim3._dunbar_state["dormant_events"] > 0
    assert [dunbar.dormant_ids(a) for a in sim3.agents] == \
           [dunbar.dormant_ids(a) for a in sim1.agents]


# --------------------------------------------------------------------- レジストリ
def test_registry_declares_dunbar_toggles():
    """第72の制約: 新 conf トグルはレジストリ宣言必須(未宣言検出は test_registry_modes)。"""
    for fid in ("relations.dunbar.enabled", "relations.dunbar.protect_housemates"):
        f = R.BY_ID.get(fid)
        assert f is not None, f"{fid} がレジストリに宣言されていない"
        assert f.repro_tier == "strict", f"{fid}: 決定論・LLM 自由文非依存なので strict のはず"
        assert f.affects_k is False, f"{fid}: generate() の呼び出し点を足さない"
    assert R.BY_ID["relations.dunbar.enabled"].fingerprint_risk == "possible", \
        "休眠で tier が落ちるとプロンプトの間柄行が変わる=possible"
