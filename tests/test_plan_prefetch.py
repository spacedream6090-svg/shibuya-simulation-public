"""PPF 夜間計画プリフェッチ(``planning.day_plan.prefetch``)の検収。

正典: `src/society/cognition/plan_prefetch.py` の module docstring
      (ユーザー要望 2026-08-25「計画は夜のうちに作って処理を分散」)

守るもの(検収基準の順)
  (1) **既定 OFF = 1 バイトも動かない**: ON 側のコード経路を 1 行も通らず、`sim` に属性が
      生えず、小規模ランの L1 が純粋既定とバイト一致する。
  (2) 候補の述語(就寝中 / 街の中 / 予約なし / 当日計画なし / 内省の予約なし)と
      全順序((sleep_until, id) 昇順 = 起床の早い順)が決定論であること。
  (3) **二重生成なし**: 前倒しで計画を持った個体が起床しても、その暦日の計画は 1 本のまま。
  (4) 窓(既定 00:10-05:50)の外では 1 件も撃たない。
  (5) 予算が尽きたら静かに翌 step へ譲る(新しい欠落モードを作らない)。
  (6) `resume == straight`(ON・分割ランの L1 一致)= 新しい搬送状態がゼロであることの機械証明。
  (7) レジストリ宣言と conf の整合 / `plan_created` の `prefetch` 欄。
"""
from __future__ import annotations

import json

import pyarrow.parquet as pq

from society import registry as R
from society.cognition import plan_prefetch as PPF
from society.config import load_config
from society.engine import checkpoint, scheduler
from society.engine.simulation import Simulation

DP = {"planning.day_plan.enabled": "true"}
PF = {"planning.day_plan.prefetch.enabled": "true"}


# --------------------------------------------------------------------------- #
# 共通ヘルパ(test_dph.py / test_day_plan.py と同型)
# --------------------------------------------------------------------------- #
def _cfg(name, n_steps=144, n_agents=12, **ov):
    dot = ["run.seed=42", f"run.n_agents={n_agents}", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock", "observer.snapshot_every=144"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def _sim(tmp_path, name, n_steps=144, n_agents=12, **ov):
    return Simulation(_cfg(name, n_steps, n_agents, **ov), out_dir=tmp_path / name)


def _run(tmp_path, name, n_steps=144, n_agents=12, **ov):
    sim = _sim(tmp_path, name, n_steps, n_agents, **ov)
    sim.run()
    return sim


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _kind(sim, kind):
    return [e for e in sim.logger.events if e.kind == kind]


def _staged(tmp_path, name, n_agents=6, **ov):
    """全員「自宅で就寝中・当日の計画なし・予約なし・内省済み」に揃えた 1 step の sim。

    プリフェッチの述語だけを裸で見るための台。各テストがここから 1 つずつ条件を壊す。
    """
    sim = _sim(tmp_path, name, n_steps=1, n_agents=n_agents, **DP, **PF, **ov)
    for i, a in enumerate(sim.agents):
        a.plan_day, a.plan_step = -1, -1
        a.reflect_step = -1
        a.sleeping, a.loc = True, "street"
        a.node = a.home_node
        a.building = a.home_building
        a.sleep_until = 200 + i          # 既定は id 昇順と同じ並び(テスト側で入れ替える)
    return sim


#: 台が使う「窓の中の夜」の (step, sim_min)。sim_min%1440 = 60 = 01:00。
NIGHT_STEP, NIGHT_MIN = 108, 1440 + 60


def _prefetched(sim):
    return [e for e in _kind(sim, "plan_created") if e.payload.get("prefetch")]


# =========================================================================== #
# (1) 既定 OFF = 1 バイトも動かない
# =========================================================================== #
def test_shipped_default_is_off():
    cfg = load_config()
    assert bool(cfg.planning.day_plan.prefetch.enabled) is False
    assert PPF.build_cfg(None)["enabled"] is False
    assert PPF.DEFAULTS["start_min"] == 10 and PPF.DEFAULTS["end_min"] == 350


def test_default_off_never_enters_the_prefetch_path(tmp_path, monkeypatch):
    """★ON 側のコードを 1 行も通らない(選抜も印も予算取りも走らない)。"""
    def _boom(*_a, **_kw):                 # 通ったら即座に落ちる
        raise AssertionError("既定 OFF なのにプリフェッチ経路へ入った")
    monkeypatch.setattr(PPF, "pick", _boom)
    monkeypatch.setattr(PPF, "mark_begin", _boom)
    sim = _run(tmp_path, "pf_never", **DP)   # day_plan は ON・prefetch は既定 OFF
    assert _kind(sim, "plan_created"), "前提が崩れた(計画が 1 本も立っていない)"
    assert getattr(sim, "_ppf_mark", None) is None, "OFF なのに印が生えた"


def test_default_off_is_byte_identical(tmp_path):
    """明示 false と純粋既定が L1 完全一致(seam が no-op であることの機械証明)。"""
    pure = _run(tmp_path, "pf_pure")
    off = _run(tmp_path, "pf_off",
               **{"planning.day_plan.prefetch.enabled": "false"})
    assert _l1(pure) == _l1(off)
    assert len(pure.logger.llm_calls) == len(off.logger.llm_calls)


def test_default_off_adds_no_payload_key(tmp_path):
    """OFF では `plan_created` に prefetch 欄が 1 つも生えない。"""
    sim = _run(tmp_path, "pf_nokey", **DP)
    created = _kind(sim, "plan_created")
    assert created
    assert all("prefetch" not in e.payload for e in created)


# =========================================================================== #
# (2) 候補の述語と全順序
# =========================================================================== #
class _A:
    """述語だけを見るための最小の個体(engine には触らない)。"""

    def __init__(self, aid=1, plan_day=-1, plan_step=-1, sleeping=True,
                 loc="street", reflect_step=-1, sleep_until=0):
        self.id = aid
        self.plan_day, self.plan_step = plan_day, plan_step
        self.sleeping, self.loc = sleeping, loc
        self.reflect_step, self.sleep_until = reflect_step, sleep_until


def test_candidate_predicate_covers_every_gate():
    step, today = 100, 1
    assert PPF.is_candidate(_A(), step, today) is True
    assert PPF.is_candidate(_A(plan_day=1), step, today) is False        # 当日の計画あり
    assert PPF.is_candidate(_A(plan_step=101), step, today) is False     # 通常の予約あり
    assert PPF.is_candidate(_A(sleeping=False), step, today) is False    # 覚醒している
    assert PPF.is_candidate(_A(loc="outside"), step, today) is False     # 街の外
    assert PPF.is_candidate(_A(reflect_step=100), step, today) is False  # 今 step に内省
    assert PPF.is_candidate(_A(reflect_step=101), step, today) is False  # 繰り越し中
    # 発火せずに残った過去の予約(k.writeback=off)は「もう発火しない」= 対象に入れてよい
    assert PPF.is_candidate(_A(reflect_step=99), step, today) is True
    # 前日の計画しか持っていない個体は当然対象
    assert PPF.is_candidate(_A(plan_day=0), step, today) is True


def test_pick_is_ordered_by_wake_time_then_id():
    """全順序 = (sleep_until, id) 昇順。同着は id で割る。"""
    class _S:
        agents = [_A(9, sleep_until=30), _A(3, sleep_until=10),
                  _A(7, sleep_until=30), _A(1, sleep_until=50)]
    got = PPF.pick(_S(), step=100, sim_min=1440 + 60, limit=4)
    assert [a.id for a in got] == [3, 7, 9, 1]
    assert [a.id for a in PPF.pick(_S(), 100, 1440 + 60, 2)] == [3, 7], \
        "上限を切っても『起床の早い順』の先頭から取れていない"
    assert PPF.pick(_S(), 100, 1440 + 60, 0) == []


def test_pick_equals_full_sort():
    """`heapq.nsmallest` の最適化が `sorted(...)[:n]` と厳密に同値であること。"""
    class _S:
        agents = [_A(i, sleep_until=(37 * i) % 11) for i in range(1, 40)]
    want = sorted(_S.agents, key=lambda a: (a.sleep_until, a.id))[:7]
    assert PPF.pick(_S(), 100, 1440 + 60, 7) == want


def test_prefetch_takes_the_earliest_risers_first(tmp_path):
    """★実機: 予算 2 枠しか無い夜は「最も早く起きる 2 人」だけが前倒しされる。"""
    sim = _staged(tmp_path, "pf_order", n_agents=6,
                  **{"lod.max_llm_per_step": "2"})
    for a in sim.agents:                    # id 昇順と逆の起床順にして id 順との差を作る
        a.sleep_until = 500 - int(a.id)
    want = [a.id for a in sorted(sim.agents, key=lambda a: (a.sleep_until, a.id))[:2]]
    scheduler._phase_plan_prefetch(sim, NIGHT_STEP, NIGHT_MIN)
    got = [e.agent_id for e in _prefetched(sim)]
    assert got == want, f"起床の早い順に前倒しされていない: {got} != {want}"
    assert sorted(a.id for a in sim.agents if a.plan_day == 1) == sorted(want), \
        "簿記(plan_day)が撃った個体と一致していない"


# =========================================================================== #
# (3) 二重生成なし
# =========================================================================== #
def test_prefetched_agent_is_not_scheduled_again_on_wake(tmp_path):
    """前倒し済みの個体が起床しても `_schedule_plan` は予約を立てない。"""
    sim = _staged(tmp_path, "pf_once", n_agents=4)
    scheduler._phase_plan_prefetch(sim, NIGHT_STEP, NIGHT_MIN)
    assert len(_prefetched(sim)) == 4
    n0 = len(sim.logger.events)
    for a in sim.agents:                    # 起床 → 既存の予約経路を通す
        scheduler._schedule_plan(sim, a, NIGHT_STEP + 20, NIGHT_MIN + 200)
        assert a.plan_step == -1, "前倒し済みなのに起床時の予約が立った"
    scheduler._phase_planning(sim, NIGHT_STEP + 21, NIGHT_MIN + 210)
    assert [e for e in sim.logger.events[n0:] if e.kind == "plan_created"] == [], \
        "同じ暦日に 2 本目の計画が立った"


def test_one_plan_per_agent_day_in_a_full_run(tmp_path):
    """★ラン全体でも「1 個体 1 暦日 ≤ 1 本」が保たれる(ガードを触っていない証拠)。"""
    from collections import Counter

    sim = _run(tmp_path, "pf_guard", n_steps=460, n_agents=20, **DP, **PF)
    per = Counter((e.agent_id, e.sim_min // 1440) for e in _kind(sim, "plan_created"))
    assert per, "計画が 1 本も立っていない(前提が崩れた)"
    assert max(per.values()) == 1, f"同じ暦日に 2 本立った: {per.most_common(3)}"
    assert _prefetched(sim), "ON なのに前倒しが 1 件も起きていない"


# =========================================================================== #
# (4) 窓の外では不発
# =========================================================================== #
def test_window_predicate():
    cfg = PPF.build_cfg({"enabled": True})
    assert PPF.in_window(cfg, 1440 + 9) is False       # 00:09(日境界処理の直後は避ける)
    assert PPF.in_window(cfg, 1440 + 10) is True       # 00:10
    assert PPF.in_window(cfg, 1440 + 349) is True      # 05:49
    assert PPF.in_window(cfg, 1440 + 350) is False     # 05:50(終端は含まない)
    assert PPF.in_window(cfg, 1440 + 720) is False     # 昼は絶対に撃たない


def test_no_prefetch_outside_the_window(tmp_path):
    """窓外の step では候補が居ても 1 件も撃たない。"""
    sim = _staged(tmp_path, "pf_window", n_agents=4)
    scheduler._phase_plan_prefetch(sim, 150, 1440 + 720)      # 12:00
    assert _prefetched(sim) == []
    assert all(a.plan_day == -1 for a in sim.agents), "窓外なのに簿記が動いた"
    scheduler._phase_plan_prefetch(sim, 108, 1440 + 60)       # 01:00 = 窓の中
    assert len(_prefetched(sim)) == 4


def test_window_is_configurable_and_never_inverted():
    assert PPF.build_cfg({"start_min": 60, "end_min": 120})["start_min"] == 60
    # 逆転した窓は「窓なし」へ潰す(1 件も撃たない = 静かな既定へ倒す)
    cfg = PPF.build_cfg({"enabled": True, "start_min": 300, "end_min": 100})
    assert cfg["end_min"] == cfg["start_min"] == 300
    assert PPF.in_window(cfg, 1440 + 300) is False


# =========================================================================== #
# (5) 内省 / outside / 覚醒 の除外(実機)
# =========================================================================== #
def test_pending_reflection_is_excluded_in_the_engine(tmp_path):
    sim = _staged(tmp_path, "pf_reflect", n_agents=4)
    sim.agents[0].reflect_step = NIGHT_STEP        # この step に内省が発火する
    sim.agents[1].reflect_step = NIGHT_STEP + 3    # 繰り越し中
    scheduler._phase_plan_prefetch(sim, NIGHT_STEP, NIGHT_MIN)
    assert sorted(e.agent_id for e in _prefetched(sim)) == \
        sorted(a.id for a in sim.agents[2:])
    assert sim.agents[0].plan_day == sim.agents[1].plan_day == -1


def test_outside_and_awake_agents_are_excluded_in_the_engine(tmp_path):
    sim = _staged(tmp_path, "pf_awake", n_agents=4)
    sim.agents[0].loc = "outside"                  # 街の外
    sim.agents[1].sleeping = False                 # 起きている
    sim.agents[2].plan_step = NIGHT_STEP + 1       # 通常の予約を持っている
    scheduler._phase_plan_prefetch(sim, NIGHT_STEP, NIGHT_MIN)
    assert [e.agent_id for e in _prefetched(sim)] == [sim.agents[3].id]


def test_existing_sleeping_skip_is_untouched(tmp_path):
    """★既存の「予約 step に眠っていたら計画を失う」規則は 1 バイトも変えていない。"""
    sim = _staged(tmp_path, "pf_skip", n_agents=3,
                  **{"observer.starvation.enabled": "true"})
    agent = sim.agents[0]
    agent.plan_day, agent.plan_step = 1, NIGHT_STEP   # 当日予約済み・かつ就寝中
    scheduler._phase_planning(sim, NIGHT_STEP, NIGHT_MIN)
    skipped = [e for e in _kind(sim, "plan_skipped")
               if e.agent_id == agent.id and e.payload["reason"] == "sleeping"]
    assert len(skipped) == 1, "既存のスキップ規則が反転している"


# =========================================================================== #
# (6) 予算
# =========================================================================== #
def test_exhausted_budget_prefetches_nothing_and_logs_nothing(tmp_path):
    """予算が尽きた step は静かに何もしない(新しい欠落モードを作らない)。"""
    sim = _staged(tmp_path, "pf_broke", n_agents=4,
                  **{"observer.starvation.enabled": "true"})
    while sim.budget.take("media"):                # 総量を使い切らせる
        pass
    n0 = len(sim.logger.events)
    scheduler._phase_plan_prefetch(sim, NIGHT_STEP, NIGHT_MIN)
    assert sim.logger.events[n0:] == [], "予算枯渇なのに何か記録した"
    assert all(a.plan_day == -1 for a in sim.agents), "予算枯渇なのに簿記が動いた"
    # 翌 step(予算が戻る)では普通に撃てる = 「取り零しの新モード」ではない
    sim.budget.reset()
    scheduler._phase_plan_prefetch(sim, NIGHT_STEP + 1, NIGHT_MIN + 10)
    assert len(_prefetched(sim)) == 4


def test_prefetch_never_exceeds_the_per_step_cap(tmp_path):
    """二層予算 ON(= 予算外呼がゼロの構成)で 1 step の総呼数が cap を超えない。

    ★tiers OFF の既存経路(`_phase_planning` の素のループ / 夜の内省)は**元から予算外**
      なので、そこと足し合わせた総数を cap で縛ることはできない(DPH-B が解いた問題)。
      プリフェッチ自身は tiers の ON/OFF に依らず**必ず `take` を通してから撃つ**。
    """
    from collections import Counter

    cap = 4
    sim = _run(tmp_path, "pf_cap", n_steps=200, n_agents=30, **DP, **PF,
               **{"lod.max_llm_per_step": str(cap),
                  "lod.budget.tiers.enabled": "true"})
    per_step = Counter(c["step"] for c in sim.logger.llm_calls)
    assert per_step and max(per_step.values()) <= cap
    assert _prefetched(sim), "cap 下でも前倒しが起きること(前提)"


def test_prefetch_uses_the_life_lane(tmp_path):
    """二層予算 ON では life レーン(purpose="plan")から取る = 返答枠を食わない。"""
    sim = _staged(tmp_path, "pf_lane", n_agents=4,
                  **{"lod.budget.tiers.enabled": "true",
                     "lod.max_llm_per_step": "20"})   # life 枠 = ceil(20×0.30) = 6 ≥ 4
    assert sim.budget.tiers is not None
    scheduler._phase_plan_prefetch(sim, NIGHT_STEP, NIGHT_MIN)
    n = len(_prefetched(sim))
    assert n == 4
    assert sim.budget.lane_used["life"] == n, "life レーン以外から取っている"
    assert sim.budget.lane_used["reply"] == 0, "返答保証の予約枠を食った"
    assert sim.budget.lane_used["general"] == 0, "自発発火の枠を食った"


# =========================================================================== #
# (7) payload の印
# =========================================================================== #
def test_payload_marks_prefetched_plans_only(tmp_path):
    sim = _run(tmp_path, "pf_mark", n_steps=460, n_agents=20, **DP, **PF)
    created = _kind(sim, "plan_created")
    pre = _prefetched(sim)
    assert pre, "前倒しが 1 件も起きていない"
    assert all(e.payload["prefetch"] is True for e in pre)
    assert all(e.payload.get("prefetch") is None for e in created if e not in pre)
    # 印は撃っている間だけ立っている(フェーズを出たら必ず降りている)
    assert PPF.marked(sim) is False


def test_mark_is_lowered_even_if_generation_raises(tmp_path, monkeypatch):
    """生成が例外で落ちても印は降りる(以後の計画に偽の prefetch が付かない)。"""
    sim = _staged(tmp_path, "pf_raise", n_agents=2)

    def _boom(*_a, **_kw):
        raise RuntimeError("boom")
    monkeypatch.setattr(scheduler.planning, "make_plan", _boom)
    try:
        scheduler._phase_plan_prefetch(sim, NIGHT_STEP, NIGHT_MIN)
    except RuntimeError:
        pass
    assert PPF.marked(sim) is False


# =========================================================================== #
# (8) resume == straight(ON・分割ランの L1 一致)
# =========================================================================== #
def _rows(run_dir, stem="l1_events"):
    return pq.read_table(run_dir / f"{stem}.parquet").to_pylist()


def test_resume_matches_straight_with_prefetch_on(tmp_path):
    """★分割点を夜(プリフェッチ帯)より前に置いて L1 完全一致を見る。

    プリフェッチは**毎 step 再計算するステートレスな述語**なので、新しい搬送状態は
    1 つも要らない。それを機械で証明する(足りなければここが必ず落ちる)。
    """
    ov = dict(DP, **PF, **{"observer.checkpoint_every": "90"})
    straight = tmp_path / "pf_straight"
    Simulation(_cfg("pf_straight", 140, 14, **ov), out_dir=straight).run()

    resumed = tmp_path / "pf_resumed"
    s1 = Simulation(_cfg("pf_resumed", 90, 14, **ov), out_dir=resumed)
    for step in range(90):                        # 90 step = 22:00 = 夜の窓の手前
        scheduler.run_step(s1, step)
    checkpoint.save(s1, 90, resumed / "checkpoint" / "ckpt-000090.pkl.gz")
    s1.logger.flush_segment()
    s2 = Simulation(_cfg("pf_resumed", 140, 14, **ov), out_dir=resumed)
    s2.run(resume_from=resumed)

    assert _rows(straight) == _rows(resumed), "resume != straight(L1)"
    got = [r for r in _rows(straight)
           if r["kind"] == "plan_created" and '"prefetch": true' in r["payload"]]
    assert got, "分割点より後に前倒しが起きていない(検証にならない)"


def test_prefetch_adds_no_checkpoint_state(tmp_path):
    """新しい sim 側の保存状態を 1 つも作らない(印は step 内で必ず降りる)。"""
    sim = _run(tmp_path, "pf_state", n_steps=144, n_agents=12, **DP, **PF)
    assert getattr(sim, "_ppf_mark") is False, "印が立ったまま step を出た"
    blob = checkpoint.save(sim, 144, tmp_path / "ck" / "c.pkl.gz")
    assert blob.exists()
    import gzip
    import pickle
    with gzip.open(blob, "rb") as fh:
        data = pickle.load(fh)
    assert not [k for k in data["runtime"] if "prefetch" in k or "ppf" in k], \
        "checkpoint に PPF 専用の欄が増えている"


# =========================================================================== #
# (9) 宣言と conf の整合
# =========================================================================== #
def test_toggle_is_declared():
    by_id = {f.id: f for f in R.FEATURES}
    fid = "planning.day_plan.prefetch.enabled"
    assert fid in by_id, f"{fid} がレジストリに無い"
    f = by_id[fid]
    assert f.repro_tier == "strict"          # 述語は整数比較のみ・乱数ゼロ
    assert f.affects_k is True               # generate() の**発生点(step)**を動かす
    assert f.fingerprint_risk == "none"      # プロンプトを 1 バイトも変えない
    assert f.off_value is False


def test_conf_block_exists_with_the_documented_defaults():
    cfg = load_config()
    pf = cfg.planning.day_plan.prefetch
    assert bool(pf.enabled) is False
    assert int(pf.start_min) == 10 and int(pf.end_min) == 350
    assert R._select(load_config(), "planning.day_plan.prefetch.enabled") is False


def test_event_schema_documents_the_new_payload_field():
    from society.observer.schema import EVENT_KINDS
    assert "prefetch" in EVENT_KINDS["plan_created"], \
        "plan_created の schema 記述に prefetch 欄が書かれていない"


def test_enabled_requires_planning(tmp_path):
    """`planning.enabled=false` では ON にしても実効にならない(前提の明示)。"""
    sim = _sim(tmp_path, "pf_noplanning", n_steps=1, n_agents=3, **PF,
               **{"planning.enabled": "false"})
    assert PPF.enabled(sim) is False
    on = _sim(tmp_path, "pf_planning", n_steps=1, n_agents=3, **PF)
    assert PPF.enabled(on) is True
