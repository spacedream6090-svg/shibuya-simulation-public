"""第81バッチ: 閾値発火 + 認知イベントキュー + 同期バリア + ダブルバッファのテスト。

正典: docs/plans/source/cognition-design-record.md §2.3(S 判定)・§3.3(同期バリア)・
      §6.2(T1/T2)/ docs/plans/source/physics-instructions.md Part P0 /
      docs/plans/cognition-physics-plan.md §4 第81行・§6-3。

守るもの(検収基準の順)
  (1) 既定 OFF = キュー不在・新 kind ゼロ件・バリア恒等(1バイトも変えない)
  (2) 後方互換の要(P0(2)): 「全員の基本周期 10 分・他の発火源なし」で **現行と L1 バイト一致**
  (3) ON: 同 seed 2 ラン一致 / resume==straight(キューの checkpoint 中央管理)
  (4) T1 完了順序不変性: 到着順を撹拌しても世界がバイト一致
  (5) T2 発火オラクル: 既知偏差を注入した個体**だけ**が発火し、無関係な個体は発火しない
  (6) 宣言: registry(journal / affects_k=true)・timeconv・manifest 来歴・較正テーブルの θ
"""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from society import registry as R
from society import timeconv as T
from society.cognition import calib as CALIB
from society.cognition import channels as CH
from society.cognition import fire as F
from society.config import load_config
from society.engine import checkpoint, scheduler
from society.engine.simulation import Simulation

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIPPED_CALIB = REPO_ROOT / CALIB.CALIB_DEFAULT_REL

# 後方互換の特殊ケース(P0(2)「全員の基本周期が10分・他の発火源なし」)
COMPAT = {
    "cognition.fire.enabled": "true",
    "cognition.fire.sources": "[periodic]",
    "cognition.fire.period_override_min": 10,
    "cognition.fire.period_cv_scale": 0,
    "cognition.fire.sleep_period_mult": 1.0,
}
FULL = {"cognition.fire.enabled": "true"}
NEW_KINDS = {"cog_fire", "cog_event"}


# --------------------------------------------------------------------------- #
# 共通ヘルパ
# --------------------------------------------------------------------------- #
def _cfg(name: str, n_steps: int = 24, n_agents: int = 24, **ov):
    dot = ["run.seed=42", f"run.n_agents={n_agents}", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def _run(tmp_path, name: str, n_steps: int = 24, n_agents: int = 24, **ov):
    out = tmp_path / name
    sim = Simulation(_cfg(name, n_steps, n_agents, **ov), out_dir=out)
    sim.run()
    return sim, out


def _l1(sim, skip=frozenset()):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events if e.kind not in skip]


def _rows(out_dir: Path, stem: str):
    path = out_dir / f"{stem}.parquet"
    return pq.read_table(path).to_pylist() if path.exists() else None


def _cog(sim, kind="cog_fire"):
    return [e for e in sim.logger.events if e.kind == kind]


def _schedule_series(sim):
    """スケジュール列そのもの(P0-4 の新不変量の左辺)。"""
    return [(e.step, e.agent_id, e.payload["reason"], e.payload["due"])
            for e in _cog(sim)]


class _CountingHub:
    """stream 派生を数えるプロキシ(test_channels と同型)。"""

    def __init__(self, inner):
        self._inner = inner
        self.counts: dict[str, int] = {}

    def stream(self, *key):
        name = str(key[0]) if key else ""
        self.counts[name] = self.counts.get(name, 0) + 1
        return self._inner.stream(*key)

    def __getattr__(self, item):
        return getattr(self._inner, item)


# --------------------------------------------------------------------------- #
# (A) 認知イベントキュー: (発火時刻, agent_id) の全順序
# --------------------------------------------------------------------------- #
def test_queue_pops_in_time_then_agent_id_order():
    """★同時刻の処理順が実行環境に依らない(DES の tie-breaking = 決定論の前提)。"""
    q = F.CogQueue()
    # 意図的に「時刻降順・id ばらばら」で投入する
    for at, aid in [(30, 7), (10, 5), (10, 1), (20, 9), (10, 3), (20, 2)]:
        q.schedule(aid, at, F.PERIODIC)
    got = [(at, aid) for at, aid, _r in q.pop_due(30)]
    assert got == [(10, 1), (10, 3), (10, 5), (20, 2), (20, 9), (30, 7)]


def test_queue_pop_is_insertion_order_independent():
    """同じ集合を別の投入順で作っても pop 列が一致する(挿入順が結果に漏れない)。"""
    events = [(10, 1), (10, 3), (10, 5), (20, 2), (20, 9), (30, 7)]
    a, b = F.CogQueue(), F.CogQueue()
    for at, aid in events:
        a.schedule(aid, at, F.PERIODIC)
    for at, aid in reversed(events):
        b.schedule(aid, at, F.PERIODIC)
    assert a.pop_due(99) == b.pop_due(99)


def test_queue_keeps_one_live_event_per_agent():
    """再スケジュールは差し替え(残骸は lazy deletion で捨てる)= 二重発火しない。"""
    q = F.CogQueue()
    q.schedule(4, 10, F.PERIODIC)
    q.schedule(4, 25, F.PERIODIC)
    assert q.order() == [(25, 4)]
    assert q.pop_due(20) == []            # 差し替え済みの残骸で発火してはいけない
    assert q.pop_due(30) == [(25, 4, F.PERIODIC)]


def test_queue_advance_only_moves_earlier():
    """割込みは**繰り上げのみ**(後ろへずらして思考を先送りする経路を作らない)。"""
    q = F.CogQueue()
    q.schedule(4, 20, F.PERIODIC)
    q.advance(4, 30, F.SALIENCE)          # 遅い → 無視
    assert q.order() == [(20, 4)]
    q.advance(4, 12, F.SALIENCE)          # 早い → 繰り上げ
    assert q.pop_due(15) == [(12, 4, F.SALIENCE)]


def test_queue_state_roundtrip_preserves_order():
    """checkpoint 往復でキューの順序が保たれる(ヒープは pending から再構築)。"""
    q = F.CogQueue()
    for at, aid in [(30, 7), (10, 5), (10, 1), (20, 9)]:
        q.schedule(aid, at, F.PERIODIC)
    back = F.CogQueue.restore(json.loads(json.dumps(q.state())))
    assert back.order() == q.order()
    assert back.pop_due(99) == q.pop_due(99)


# --------------------------------------------------------------------------- #
# (B) 既定 OFF(検収基準 1)
# --------------------------------------------------------------------------- #
def test_shipped_default_is_off():
    cfg = load_config()
    assert bool(cfg.cognition.fire.enabled) is False


def test_off_creates_no_queue_and_no_events(tmp_path):
    sim, out = _run(tmp_path, "f_off", 12, 12)
    assert sim.cogq is None
    assert F.enabled(sim) is False
    assert not [e for e in sim.logger.events if e.kind in NEW_KINDS]
    man = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert "cognition" not in man, "OFF なのに manifest に来歴キーが生えた"


def test_off_barrier_is_the_identity(tmp_path):
    """OFF のバリアは**同一オブジェクト**を返す(並べ替えも複製もしない=バイト一致)。"""
    sim = Simulation(_cfg("f_id", 1, 4), out_dir=tmp_path / "f_id")
    payload = [("a", 1), ("b", 2)]
    assert F.barrier(sim, payload) is payload


def test_off_does_not_touch_agents(tmp_path):
    """OFF では観測の凍結も期待値もエージェントに生えない。"""
    sim, _out = _run(tmp_path, "f_clean", 6, 8)
    for agent in sim.agents:
        assert not hasattr(agent, "_fire_obs")
        assert not hasattr(agent, "_fire_pred")


# --------------------------------------------------------------------------- #
# (C) ★後方互換の要(P0(2)・検収基準 2)
# --------------------------------------------------------------------------- #
def test_ten_minute_period_reproduces_current_firing_byte_for_byte(tmp_path):
    """「全員の基本周期 10 分・他の発火源なし」= 現行の特殊ケース。

    physics-instructions.md P0(2):「現行の10分固定は『全員の基本周期が10分・他の発火源
    なし』という特殊ケースとして表現できること。**これが後方互換の要です**」。
    この設定では毎 step 全員が期限を迎えるので候補集合が現行と一致し、周期発火は
    従来どおり閾値+個人重み抽選+予算のゲートを通る → 発火判定がバイト一致する。
    """
    off = Simulation(_cfg("bc_off", 36, 30), out_dir=tmp_path / "bc_off")
    off.hub = _CountingHub(off.hub)
    off.run()
    on = Simulation(_cfg("bc_on", 36, 30, **COMPAT), out_dir=tmp_path / "bc_on")
    on.hub = _CountingHub(on.hub)
    on.run()

    assert _l1(off) == _l1(on, skip=NEW_KINDS), \
        "10分固定の特殊ケースで L1(新 kind を除く)が現行と一致しない"
    assert [dict(m) for m in off.logger.metrics] == \
           [dict(m) for m in on.logger.metrics], "L2 が変わった"
    assert [dict(s) for s in off.logger.snapshots] == \
           [dict(s) for s in on.logger.snapshots], "L3 が変わった"
    assert off.llm.calls == on.llm.calls > 0, \
        f"LLM 呼数が変わった: {off.llm.calls} vs {on.llm.calls}"
    assert off.hub.counts == on.hub.counts, \
        f"乱数 stream の消費が変わった: {off.hub.counts} vs {on.hub.counts}"


def test_ten_minute_period_makes_every_agent_due_every_step(tmp_path):
    """特殊ケースでは全員が毎 step 期限を迎える(= 現行の『毎 step 全員申請』)。"""
    sim, _out = _run(tmp_path, "bc_due", 12, 15, **COMPAT)
    fires = _cog(sim)
    assert len(fires) == 12 * 15, f"期限イベント数が全 step×全 agent と違う: {len(fires)}"
    assert {e.payload["reason"] for e in fires} == {F.PERIODIC}


def test_zero_cv_draws_no_randomness(tmp_path):
    """period_cv_scale=0 は専用 stream を **1 本も引かない**(決定論の特殊ケース)。"""
    sim = Simulation(_cfg("bc_rng", 12, 12, **COMPAT), out_dir=tmp_path / "bc_rng")
    sim.hub = _CountingHub(sim.hub)
    sim.run()
    assert "cog_fire" not in sim.hub.counts, \
        f"cv=0 なのに乱数を引いている: {sim.hub.counts.get('cog_fire')}"


def test_jittered_period_uses_the_dedicated_stream(tmp_path):
    """ばらつきを使うときは専用 stream からだけ引く(R1: 用途別 named stream)。"""
    sim = Simulation(_cfg("jit", 12, 12, **FULL), out_dir=tmp_path / "jit")
    sim.hub = _CountingHub(sim.hub)
    sim.run()
    assert sim.hub.counts.get("cog_fire", 0) > 0


# --------------------------------------------------------------------------- #
# (D) ON の決定論・resume(検収基準 3)
# --------------------------------------------------------------------------- #
def test_same_seed_two_runs_are_identical(tmp_path):
    """★新不変量(P0-4): 認知イベントのスケジュール列が同一なら世界状態が同一。"""
    a, _oa = _run(tmp_path, "det_a", 36, 24, **FULL)
    b, _ob = _run(tmp_path, "det_b", 36, 24, **FULL)
    assert _schedule_series(a) == _schedule_series(b), "スケジュール列が不一致"
    assert _l1(a) == _l1(b), "スケジュール列は同一なのに世界が不一致"
    assert a.llm.calls == b.llm.calls
    assert a.cogq.order() == b.cogq.order(), "終端のキュー状態が不一致"


def test_resume_matches_straight(tmp_path):
    """★キュー状態の checkpoint 中央管理(第62/70/75/W2 と同型のギャップ潰し)。"""
    straight = tmp_path / "s"
    Simulation(_cfg("s", 24, 24, **FULL), out_dir=straight).run()

    d = tmp_path / "r"
    every = {"observer.checkpoint_every": 12}
    sim1 = Simulation(_cfg("r", 12, 24, **FULL, **every), out_dir=d)
    for step in range(12):
        scheduler.run_step(sim1, step)
    checkpoint.save(sim1, 12, d / "checkpoint" / "ckpt-000012.pkl.gz")
    sim1.logger.flush_segment()
    sim2 = Simulation(_cfg("r", 24, 24, **FULL, **every), out_dir=d)
    sim2.run(resume_from=d)

    assert _rows(straight, "l1_events") == _rows(d, "l1_events"), "resume≠straight (L1)"
    assert _rows(straight, "l2_metrics") == _rows(d, "l2_metrics"), "resume≠straight (L2)"


def test_checkpoint_carries_the_queue(tmp_path):
    """キューを保存しないと resume 直後に『全員が今すぐ期限』へ戻る(退行検知)。"""
    d = tmp_path / "ck"
    sim = Simulation(_cfg("ck", 12, 16, **FULL), out_dir=d)
    for step in range(12):
        scheduler.run_step(sim, step)
    saved = sim.cogq.order()
    assert saved, "キューが空(この時点で予約が無いのは異常)"
    path = checkpoint.save(sim, 12, d / "checkpoint" / "ckpt-000012.pkl.gz")
    sim2 = Simulation(_cfg("ck", 24, 16, **FULL), out_dir=tmp_path / "ck2")
    checkpoint.load(sim2, path)
    assert sim2.cogq.order() == saved, "checkpoint でキューの予約が失われた"


# --------------------------------------------------------------------------- #
# (E) ★T1 完了順序不変性(設計 §6.2「これが通らないとバリアを入れた意味がない」)
# --------------------------------------------------------------------------- #
def _reverse_probe(pairs):
    """到着順(推論の完了順)を意図的に反転させるプローブ。"""
    return list(reversed(pairs))


def _rotate_probe(pairs):
    seq = list(pairs)
    return seq[3:] + seq[:3]


@pytest.mark.parametrize("probe", [_reverse_probe, _rotate_probe])
def test_completion_order_does_not_leak_into_the_world(tmp_path, probe):
    """★T1: 同じ step の**到着順**を変えても世界がバイト一致すること。

    設計 §3.3:「到着順に反映すると完了順序(= GPU の気まぐれ)が結果に漏れる」。
    バリアは結果を agent_id 昇順へ畳んでから適用するので、到着順は世界に漏れない。
    """
    base = Simulation(_cfg("t1_base", 30, 24, **FULL), out_dir=tmp_path / "t1_base")
    base.run()
    shuffled = Simulation(_cfg("t1_shuf", 30, 24, **FULL), out_dir=tmp_path / "t1_shuf")
    shuffled._fire_arrival_probe = probe
    shuffled.run()

    assert _l1(base) == _l1(shuffled), "到着順が L1 に漏れている"
    assert [dict(m) for m in base.logger.metrics] == \
           [dict(m) for m in shuffled.logger.metrics], "到着順が L2 に漏れている"
    assert [dict(s) for s in base.logger.snapshots] == \
           [dict(s) for s in shuffled.logger.snapshots], "到着順が L3 に漏れている"
    assert base.llm.calls == shuffled.llm.calls


def test_concurrent_issue_does_not_change_the_world(tmp_path):
    """実際に並行発行させても世界が一致(engine.batch_llm workers=4 vs 逐次)。

    T1 の姉妹検査。到着順プローブが人工的な撹拌であるのに対し、こちらは
    ThreadPoolExecutor が本当に走る経路で「完了順序が漏れない」ことを確かめる。
    """
    serial, _a = _run(tmp_path, "t1_ser", 30, 24, **FULL)
    batched, _b = _run(tmp_path, "t1_bat", 30, 24, **FULL,
                       **{"engine.batch_llm.enabled": "true",
                          "engine.batch_llm.workers": 4})
    assert _l1(serial) == _l1(batched), "並行発行で世界が変わった"
    assert serial.llm.calls == batched.llm.calls > 0


def test_barrier_canonicalises_to_agent_id_ascending(tmp_path):
    sim = Simulation(_cfg("t1_ord", 1, 6, **FULL), out_dir=tmp_path / "t1_ord")

    class _A:
        def __init__(self, i):
            self.id = i

    pairs = [(_A(5), "e"), (_A(1), "a"), (_A(3), "c")]
    out = F.barrier(sim, pairs)
    assert [a.id for a, _ in out] == [1, 3, 5]


def test_probe_actually_permutes(tmp_path):
    """プローブが恒等だと T1 が空回りする(テスト自身の健全性)。"""
    pairs = list(range(6))
    assert _reverse_probe(pairs) != pairs and _rotate_probe(pairs) != pairs


# --------------------------------------------------------------------------- #
# (F) ★T2 発火オラクル(既知偏差の注入)
# --------------------------------------------------------------------------- #
def _fire_sim(tmp_path, name, n_agents=12, **ov):
    """キューだけを直接叩くための最小 sim(run はしない)。"""
    sim = Simulation(_cfg(name, 1, n_agents, **{**FULL, **ov}), out_dir=tmp_path / name)
    return sim


def _flat_obs(value=0.0):
    """全チャンネル一定値の観測タプル(未実装チャンネルは null のまま)。"""
    return tuple(None if not c.implemented else value for c in CH.CHANNELS)


def _inject(sim, deviant_id: int, magnitude: float):
    """全員に「期待どおり」の観測を与え、1 体だけへ既知偏差を注入する。"""
    base = _flat_obs(0.0)
    dev = list(base)
    for i, _cid, _sigma in F.usable_channels(sim):
        dev[i] = magnitude
    for agent in sim.agents:
        agent._fire_pred = base
        agent._fire_obs = tuple(dev) if agent.id == deviant_id else base
        agent._fire_prev_drive = float(agent.drive)


def test_injected_deviation_fires_only_the_intended_agent(tmp_path):
    """★T2: 意図した個体**だけ**が驚き発火し、無関係な個体は発火しない。"""
    sim = _fire_sim(tmp_path, "t2", n_agents=16,
                    **{"cognition.fire.sources": "[salience]"})
    assert F.usable_channels(sim), "σ_c 凍結ファイルに usable が無い(前提が崩れている)"
    target = sim.agents[5].id
    _inject(sim, target, magnitude=50.0)          # θ を確実に超える大偏差
    # キューを「まだ誰も期限でない」状態にする(周期発火と混ざらないように)
    for agent in sim.agents:
        sim.cogq.schedule(int(agent.id), 10_000, F.PERIODIC)

    due = F.due_events(sim, step=1, sim_min=10, active=list(sim.agents))
    assert set(due) == {target}, f"意図しない個体が発火した: {sorted(due)}"
    assert due[target]["reason"] == F.SALIENCE
    assert due[target]["interrupt"] is True
    assert due[target]["s"] > due[target]["theta"] > 0.0
    assert due[target]["contrib"], "S の寄与内訳が記録されていない"


def test_no_deviation_means_no_salience_firing(tmp_path):
    """期待どおりの世界では驚き発火は 1 件も起きない(偽陽性がない)。"""
    sim = _fire_sim(tmp_path, "t2b", n_agents=16,
                    **{"cognition.fire.sources": "[salience]"})
    _inject(sim, deviant_id=-1, magnitude=0.0)    # 誰にも偏差を与えない
    for agent in sim.agents:
        sim.cogq.schedule(int(agent.id), 10_000, F.PERIODIC)
    assert F.due_events(sim, step=1, sim_min=10, active=list(sim.agents)) == {}


def test_salience_scales_with_the_number_of_sigmas(tmp_path):
    """S は「期待から何σ外れたか」の合計(σ で割ることが必須=設計 §2.3)。"""
    sim = _fire_sim(tmp_path, "t2c", n_agents=4)
    usable = F.usable_channels(sim)
    base = _flat_obs(0.0)
    one = list(base)
    for i, _cid, sigma in usable:                  # 各チャンネルをちょうど 1σ ずらす
        one[i] = sigma
    s, contrib = F.salience_of(tuple(one), base, usable, 3)
    assert s == pytest.approx(float(len(usable))), "1σ ずらしたのに S がチャンネル数と違う"
    assert len(contrib) == 3 and all(abs(v - 1.0) < 1e-6 for _cid, v in contrib)


def test_missing_channels_are_not_counted(tmp_path):
    """欠測(None)は寄与に数えない(0 で埋めて偽の一致を作らない)。"""
    sim = _fire_sim(tmp_path, "t2d", n_agents=4)
    usable = F.usable_channels(sim)
    base = _flat_obs(0.0)
    obs = list(base)
    for i, _cid, _sigma in usable:
        obs[i] = None
    s, contrib = F.salience_of(tuple(obs), base, usable, 3)
    assert s == 0.0 and contrib == []


def test_unset_expectation_gives_zero_salience(tmp_path):
    """ô が未確定(まだ 1 度も認知イベントを経ていない)なら S=0。"""
    sim = _fire_sim(tmp_path, "t2e", n_agents=4)
    assert F.salience_of(_flat_obs(9.0), None, F.usable_channels(sim), 3) == (0.0, [])


def test_only_usable_channels_enter_S(tmp_path):
    """σ=0(定数)・未実装・標本なしは S から**除外**される(床を敷かない=第80の方針)。"""
    sim = _fire_sim(tmp_path, "t2f", n_agents=4)
    ids = {cid for _i, cid, _s in F.usable_channels(sim)}
    assert ids, "usable が空"
    assert "pred.unmet" not in ids, "未実装チャンネルが S に入っている"
    assert all(s > 0.0 for _i, _c, s in F.usable_channels(sim))
    frozen = CALIB.sigma_of(sim.cognition_sigma)
    assert ids == set(frozen), "usable の集合が σ_c 凍結ファイルと食い違う"


# --------------------------------------------------------------------------- #
# (G) 発火源の分類と第一級化(§6-3)
# --------------------------------------------------------------------------- #
def test_sources_are_canonicalised_and_unknown_is_rejected():
    assert F.build_cfg({"sources": ["social", "periodic"]})["sources"] == \
        (F.PERIODIC, F.SOCIAL), "宣言順が世界に漏れないよう正典順へ畳むこと"
    with pytest.raises(ValueError):
        F.build_cfg({"sources": ["telepathy"]})


def test_every_fire_event_carries_a_reason(tmp_path):
    """発火イベントには reason が必ず付く(P0-4: 発火時刻・理由・トリガ元の記録)。"""
    sim, _out = _run(tmp_path, "reasons", 36, 24, **FULL)
    fires = _cog(sim)
    assert fires
    for e in fires:
        assert e.payload["reason"] in F.SOURCES
        assert "due" in e.payload and "s" in e.payload and "theta" in e.payload
        assert "contrib" in e.payload and "granted" in e.payload


def test_social_sources_are_recorded_as_first_class_events(tmp_path):
    """計画書 §6-3: 会話返答・夜内省・朝計画も第一級の発火源としてキューに載る。"""
    sim, _out = _run(tmp_path, "social", 60, 30, **FULL)
    vias = {e.payload["via"] for e in _cog(sim, "cog_event")}
    assert vias, "social イベントが 1 件も記録されていない"
    assert vias <= {"reply", "reflect", "plan"}
    assert all(e.payload["reason"] == F.SOCIAL for e in _cog(sim, "cog_event"))


def test_social_source_off_records_nothing(tmp_path):
    sim, _out = _run(tmp_path, "nosocial", 36, 24,
                     **{**FULL, "cognition.fire.sources": "[periodic]"})
    assert not _cog(sim, "cog_event")


def test_sleeping_agents_get_longer_periods(tmp_path):
    """P0(2)「睡眠中は伸びる」。同じ文脈でも睡眠中は周期が伸びること。"""
    sim = _fire_sim(tmp_path, "sleep", n_agents=4,
                    **{"cognition.fire.period_cv_scale": 0})
    agent = sim.agents[0]
    agent.sleeping = False
    awake = F._period_min(sim, agent, F.RESTING, 0)
    agent.sleeping = True
    asleep = F._period_min(sim, agent, F.RESTING, 0)
    assert asleep > awake
    assert asleep == pytest.approx(round(awake * sim.firecfg["sleep_period_mult"]), abs=1)


def test_context_classification_is_deterministic_and_covers_the_table(tmp_path):
    sim = _fire_sim(tmp_path, "ctx", n_agents=4)
    agent = sim.agents[0]
    agent.sleeping, agent.route = True, []
    assert F.context_of(agent, None) == F.RESTING
    agent.sleeping, agent.route = False, ["n1"]
    assert F.context_of(agent, None) == F.WALKING
    agent.route = []
    obs = list(_flat_obs(0.0))
    obs[F._ENC_IDX] = 2.0
    assert F.context_of(agent, tuple(obs)) == F.TALKING
    obs[F._ENC_IDX] = 0.0
    agent.node = agent.work_node = "w1"
    assert F.context_of(agent, tuple(obs)) == F.WORKING
    agent.work_node = ""
    agent.home_node = "w1"
    assert F.context_of(agent, tuple(obs)) == F.RESTING
    # 較正テーブルはこの 4 文脈をすべて持っていなければならない
    table = CALIB.load_calib(None)["table"]
    for ctx in (F.WALKING, F.TALKING, F.WORKING, F.RESTING):
        assert table["base_period"][ctx]["mean_min"] > 0
        assert table["salience"][ctx]["theta"] > 0


# --------------------------------------------------------------------------- #
# (H) no-fingerprint: 発火機構はプロンプトに漏れない
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("extra", [
    {},                                                    # 第81 の素の fire ON
    {"cognition.watch.enabled": "true"},                   # 第82: watch 節が増える
    {"cognition.watch.enabled": "true",                    # 第82: g/θ 更新も ON
     "cognition.g_update.enabled": "true"},
    {"cognition.g_update.enabled": "true",                 # 第82: 実験条件 N
     "experiment.g_init.mode": "noise"},
])
def test_fire_reason_never_reaches_the_prompt(tmp_path, extra):
    """発火理由は L1 の cog_fire にだけ出す(プロンプト語彙は従来のまま)。

    第82バッチで watch 節と model-revision の 1 行がプロンプトに増えたので、
    **増えた後も**発火源の語彙・実験条件の語彙・因子名が出ないことを固定する
    (増分だけを足すのではなく、条件を変えて同じ検査を掛け直すのが目的)。
    """
    name = "fp" + str(abs(hash(tuple(sorted(extra.items())))) % 10_000)
    sim = Simulation(_cfg(name, 30, 24, **FULL, **extra), out_dir=tmp_path / name)
    seen: list[str] = []
    inner = sim.llm.generate

    def _spy(prompt, **kw):
        seen.append(prompt)
        return inner(prompt, **kw)

    sim.llm.generate = _spy
    sim.run()
    assert seen, "LLM が 1 度も呼ばれていない"
    blob = "\n".join(seen)
    for token in ("periodic", "salience", "cog_fire", "認知イベント", "発火時刻",
                  # 第82: 機構語・実験条件語・因子名(チャンネル id 経由の漏洩を含む)
                  "g_update", "g_init", "experiment", "感度", "慣れ", "感作",
                  "予測誤差", "監視仕様", "efficacy", "ownership", "body.state"):
        assert token not in blob, f"発火機構の語がプロンプトに漏れている: {token}"


# --------------------------------------------------------------------------- #
# (I) 宣言(検収基準 6)
# --------------------------------------------------------------------------- #
def test_feature_is_declared_in_the_registry():
    feature = {f.id: f for f in R.FEATURES}.get("cognition.fire.enabled")
    assert feature is not None, "機能レジストリに宣言が無い"
    assert feature.repro_tier == "journal", "LLM 呼の発生点を差し替えるので journal"
    assert feature.affects_k is True, "★affects_k=true を正直に宣言すること"
    assert feature.fingerprint_risk == "none", "プロンプトを 1 バイトも変えない"


def test_verify_mode_disables_fire():
    """journal 等級なので run.mode=verify(対照実験)では自動 OFF になる。"""
    cfg = load_config(["run.mode=verify", "cognition.fire.enabled=true"])
    cfg, report = R.apply_mode(cfg)
    assert bool(cfg.cognition.fire.enabled) is False
    assert "cognition.fire.enabled" in {f["id"] for f in report["auto_disabled"]}


def test_new_conf_keys_are_classified_in_timeconv():
    """★周期系は『分』で表されるので Δt 非依存(step ではない)。"""
    for key in ("cognition.fire.period_override_min", "cognition.fire.period_scale",
                "cognition.fire.period_cv_scale", "cognition.fire.sleep_period_mult",
                "cognition.fire.theta_scale", "cognition.fire.max_contrib"):
        assert T.covers(key), f"Δt 分類テーブルに宣言が無い: {key}"
        assert T.classify(key)[0] == T.INVARIANT, f"{key} は Δt 非依存でなければならない"


def test_manifest_declares_the_firing_mechanism(tmp_path):
    _sim, out = _run(tmp_path, "man", 6, 8, **FULL)
    man = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    fire = man["cognition"]["fire"]
    assert fire["sources"] == list(F.SOURCES)
    assert fire["n_usable_channels"] > 0
    # ★第82 の 2 トグル(watch / g_update)が OFF のままなら、第81 の暫定実装で走ったと
    #   来歴に正直に書いてある(ON にしたときの宣言は tests/test_watch.py が固定する)
    assert "persistence" in fire["expectation_model"]
    assert fire["g_policy"].startswith("fixed_1.0")
    assert fire["trigger_dsl"].startswith("disabled")
    assert fire["theta_source"] == "calib_table(provisional)"
    assert "watch" not in man["cognition"], "watch OFF なのに来歴キーが生えた"
    assert "g_update" not in man["cognition"], "g_update OFF なのに来歴キーが生えた"
    assert man["cognition"]["calib"]["status"] == "provisional"


def test_calib_table_carries_provisional_theta():
    table = CALIB.load_calib(None)["table"]
    assert set(table["salience"]) == set(CALIB.CONTEXTS)
    assert CALIB.load_calib(None)["table"]["status"] == "provisional"


@pytest.mark.parametrize("mutate", [
    lambda d: d.pop("salience"),
    lambda d: d["salience"].pop("resting"),
    lambda d: d["salience"]["walking"].update({"theta": 0.0}),
    lambda d: d["salience"]["walking"].update({"theta": -1.0}),
    lambda d: d.update({"schema": 1}),
])
def test_broken_theta_block_raises(mutate):
    """θ の欠けは**黙って既定値へ落ちない**(発火率を直接決める量なので)。"""
    from omegaconf import OmegaConf
    doc = OmegaConf.to_container(OmegaConf.load(SHIPPED_CALIB), resolve=True)
    mutate(doc)
    with pytest.raises(ValueError):
        CALIB.validate_calib(doc, "test")


# --------------------------------------------------------------------------- #
# (J) 認知時間と世界 tick の分離(P0(1))
# --------------------------------------------------------------------------- #
def test_thinking_period_is_expressed_in_minutes_not_steps(tmp_path):
    """周期は**分**で持つ(P0(1): 認知イベントは世界 tick から独立)。

    period_override_min=20(= 2 step 相当)にすると期限イベント数がおおよそ半減する。
    """
    fast, _a = _run(tmp_path, "p10", 24, 20,
                    **{**COMPAT, "cognition.fire.period_override_min": 10})
    slow, _b = _run(tmp_path, "p20", 24, 20,
                    **{**COMPAT, "cognition.fire.period_override_min": 20})
    n_fast, n_slow = len(_cog(fast)), len(_cog(slow))
    assert n_fast == 24 * 20
    assert n_slow == pytest.approx(n_fast / 2, rel=0.15), \
        f"周期を 2 倍にしたのに期限イベント数が半減しない: {n_fast} -> {n_slow}"


def test_variable_thinking_changes_the_llm_call_count(tmp_path):
    """affects_k=true の実体: 可変思考 ON では呼数が現行と変わりうる。"""
    off, _a = _run(tmp_path, "k_off", 36, 24)
    on, _b = _run(tmp_path, "k_on", 36, 24, **FULL)
    assert off.llm.calls > 0 and on.llm.calls > 0
    assert off.llm.calls != on.llm.calls, \
        "可変思考 ON で呼数が全く変わらないなら発火機構が効いていない"
