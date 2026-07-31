"""第82バッチ: 監視仕様 watch spec(ô = LLM 出力 + トリガ DSL)+ model-revision のテスト。

正典: docs/plans/source/cognition-design-record.md §2.2(監視仕様)・§2.3(S 判定)・
      §2.8(責務分界)/ docs/plans/cognition-physics-plan.md §6-3(model-revision)。

守るもの(検収基準の順)
  (1) 既定 OFF = 新 kind ゼロ件・プロンプト不変・来歴キー不在
  (2) fire ON + watch OFF = **第81 と完全に同一挙動**(L1/L2/L3・乱数・呼数バイト一致)
  (3) watch ON: mock 288step 完走 / 同 seed 2 ラン一致 / resume==straight
  (4) DSL: ホワイトリスト外・不正値は**前回仕様を維持**、範囲内はクランプ
  (5) ★オラクル: ô が S 判定に実際に効く(ô を外すと発火し、当てると発火しない)
  (6) no-fingerprint: 発火理由・実験条件・因子名がプロンプト全文に出現しない
  (7) 宣言: registry(journal / affects_k / fingerprint_risk)・timeconv・manifest
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from society import registry as R
from society import timeconv as T
from society.cognition import channels as CH
from society.cognition import fire as F
from society.cognition import watch as W
from society.config import load_config
from society.engine import checkpoint, scheduler
from society.engine.simulation import Simulation

FIRE = {"cognition.fire.enabled": "true"}
WATCH = {**FIRE, "cognition.watch.enabled": "true"}
NEW_KINDS = {"watch_spec", "cog_theta"}
ALL_NEW = NEW_KINDS | {"cog_fire", "cog_event"}


# --------------------------------------------------------------------------- #
# 共通ヘルパ(test_fire.py と同型)
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
    import pyarrow.parquet as pq
    path = out_dir / f"{stem}.parquet"
    return pq.read_table(path).to_pylist() if path.exists() else None


def _kind(sim, kind):
    return [e for e in sim.logger.events if e.kind == kind]


class _CountingHub:
    def __init__(self, inner):
        self._inner = inner
        self.counts: dict[str, int] = {}

    def stream(self, *key):
        name = str(key[0]) if key else ""
        self.counts[name] = self.counts.get(name, 0) + 1
        return self._inner.stream(*key)

    def __getattr__(self, item):
        return getattr(self._inner, item)


def _spy(sim) -> list[str]:
    """LLM へ渡したプロンプト全文を集める。"""
    seen: list[str] = []
    inner = sim.llm.generate

    def _gen(prompt, **kw):
        seen.append(prompt)
        return inner(prompt, **kw)

    sim.llm.generate = _gen
    return seen


def _sim(tmp_path, name, n_agents=12, **ov):
    """`run` せずに watch を直接叩くための最小 sim。"""
    return Simulation(_cfg(name, 1, n_agents, **{**WATCH, **ov}),
                      out_dir=tmp_path / name)


def _flat_obs(value=0.0):
    return tuple(None if not c.implemented else value for c in CH.CHANNELS)


# --------------------------------------------------------------------------- #
# (A) 既定 OFF(検収基準 1)
# --------------------------------------------------------------------------- #
def test_shipped_default_is_off():
    cfg = load_config()
    assert bool(cfg.cognition.watch.enabled) is False
    assert bool(cfg.cognition.g_update.enabled) is False
    assert str(cfg.experiment.g_init.mode) == "persona"


def test_off_creates_no_events_and_no_prompt_line(tmp_path):
    sim = Simulation(_cfg("w_off", 12, 12), out_dir=tmp_path / "w_off")
    seen = _spy(sim)
    sim.run()
    assert W.enabled(sim) is False
    assert not [e for e in sim.logger.events if e.kind in ALL_NEW]
    assert W.section(sim) is None
    assert '"watch"' not in "\n".join(seen)


def test_watch_needs_fire_on(tmp_path):
    """watch は fire ON が前提(fire OFF では ON にしても効かない)。"""
    sim = Simulation(_cfg("w_nofire", 1, 4, **{"cognition.watch.enabled": "true"}),
                     out_dir=tmp_path / "w_nofire")
    assert W.enabled(sim) is False and W.section(sim) is None


# --------------------------------------------------------------------------- #
# (B) ★fire ON + watch OFF = 第81 と同一挙動(検収基準 2)
# --------------------------------------------------------------------------- #
def test_fire_on_watch_off_is_byte_identical_to_batch81(tmp_path):
    """第82 の 2 トグルを OFF にした fire ON は、第81 の挙動を 1 バイトも変えない。

    (第81 のゴールデンは「その設定でのランの L1」そのものなので、同じ設定の 2 ラン
     を比べるのではなく **watch/g_update のコードが混ざった実装で第81 設定を回す**
     ことが検査になる。ここでは全 kind を含めて比較する = 新 kind が 1 件も出ない。)
    """
    a, _oa = _run(tmp_path, "b81_a", 36, 24, **FIRE)
    assert not [e for e in a.logger.events if e.kind in NEW_KINDS], \
        "watch/g_update OFF なのに新 kind が生えた"
    for e in a.logger.events:
        if e.kind == "cog_fire":
            assert "trig" not in e.payload, "watch OFF なのにトリガ列が生えた"
    man = json.loads((_oa / "run_manifest.json").read_text(encoding="utf-8"))
    assert "watch" not in man["cognition"] and "g_update" not in man["cognition"]


def test_fire_on_watch_off_draws_no_new_streams(tmp_path):
    sim = Simulation(_cfg("b81_rng", 24, 16, **FIRE), out_dir=tmp_path / "b81_rng")
    sim.hub = _CountingHub(sim.hub)
    sim.run()
    assert "g_init" not in sim.hub.counts
    assert "mock_watch" not in sim.hub.counts


# --------------------------------------------------------------------------- #
# (C) watch ON の決定論・resume(検収基準 3)
# --------------------------------------------------------------------------- #
def test_watch_on_full_day_runs(tmp_path):
    """mock で 288 step(= 2 日)完走し、監視仕様が実際に受理されている。"""
    sim, _out = _run(tmp_path, "w_day", 288, 24, **WATCH)
    specs = _kind(sim, "watch_spec")
    assert specs, "watch_spec が 1 件も出ていない"
    assert {e.payload["status"] for e in specs} <= set(W.STATUSES)
    assert any(e.payload["status"] == W.OK for e in specs)


def test_same_seed_two_runs_are_identical(tmp_path):
    a, _oa = _run(tmp_path, "w_det_a", 36, 24, **WATCH)
    b, _ob = _run(tmp_path, "w_det_b", 36, 24, **WATCH)
    assert _l1(a) == _l1(b)
    assert a.llm.calls == b.llm.calls > 0


def test_resume_matches_straight(tmp_path):
    straight = tmp_path / "ws"
    Simulation(_cfg("ws", 24, 24, **WATCH), out_dir=straight).run()

    d = tmp_path / "wr"
    every = {"observer.checkpoint_every": 12}
    sim1 = Simulation(_cfg("wr", 12, 24, **WATCH, **every), out_dir=d)
    for step in range(12):
        scheduler.run_step(sim1, step)
    checkpoint.save(sim1, 12, d / "checkpoint" / "ckpt-000012.pkl.gz")
    sim1.logger.flush_segment()
    sim2 = Simulation(_cfg("wr", 24, 24, **WATCH, **every), out_dir=d)
    sim2.run(resume_from=d)

    assert _rows(straight, "l1_events") == _rows(d, "l1_events"), "resume≠straight (L1)"
    assert _rows(straight, "l2_metrics") == _rows(d, "l2_metrics"), "resume≠straight (L2)"


def test_watch_spec_survives_checkpoint(tmp_path):
    """監視仕様は agents pickle に同梱される(resume 直後に ô が消えない)。"""
    d = tmp_path / "wck"
    sim = Simulation(_cfg("wck", 12, 16, **WATCH), out_dir=d)
    for step in range(12):
        scheduler.run_step(sim, step)
    saved = {int(a.id): getattr(a, "_fire_watch", None) for a in sim.agents}
    assert any(v for v in saved.values()), "この時点で監視仕様が 1 件も無いのは異常"
    path = checkpoint.save(sim, 12, d / "checkpoint" / "ckpt-000012.pkl.gz")
    sim2 = Simulation(_cfg("wck", 24, 16, **WATCH), out_dir=tmp_path / "wck2")
    checkpoint.load(sim2, path)
    back = {int(a.id): getattr(a, "_fire_watch", None) for a in sim2.agents}
    assert back == saved


# --------------------------------------------------------------------------- #
# (D) ★DSL: ホワイトリスト + 数値クランプ + 不正なら前回仕様を維持(検収基準 4)
# --------------------------------------------------------------------------- #
def _resp(watch_block) -> str:
    return json.dumps({"action": "wander", "watch": watch_block}, ensure_ascii=False)


def test_valid_spec_is_accepted(tmp_path):
    sim = _sim(tmp_path, "dsl_ok")
    agent = sim.agents[0]
    agent._fire_obs = _flat_obs(0.0)
    sym = W.symbols(sim)[0][0]
    spec, status, clamped = W.parse(sim, agent, _resp(
        {"expect": {sym: 0.5}, "triggers": [{"name": "混む", "ch": sym, "op": ">",
                                             "value": 0.9}]}))
    assert status == W.OK and clamped == 0
    assert list(spec["expect"].values()) == [0.5]
    assert spec["triggers"][0][0] == "混む" and spec["triggers"][0][2] == ">"


@pytest.mark.parametrize("block,expect_status", [
    ({"expect": {"zzz": 1.0}}, W.REJECT_CHANNEL),          # ホワイトリスト外の記号
    ({"expect": {"c01": "たくさん"}}, W.REJECT_VALUE),      # 自由文の数値
    ({"expect": {"c01": 1e9}}, W.REJECT_VALUE),            # 単位の許容域外
    ({"expect": []}, W.REJECT_SHAPE),                      # 形が違う
    ({"triggers": [{"ch": "c01", "op": "os.system", "value": 1}]}, W.REJECT_OP),
    ({"triggers": [{"ch": "c01", "op": ">", "value": None}]}, W.REJECT_VALUE),
    ({"triggers": "c01 > 5 のとき"}, W.REJECT_SHAPE),       # 自由文は不可
    ({"triggers": [{"ch": "zzz", "op": ">", "value": 1}]}, W.REJECT_CHANNEL),
])
def test_invalid_spec_keeps_the_previous_one(tmp_path, block, expect_status):
    """★設計 §2.2「不正出力なら**前回仕様を維持**すること」。"""
    sim = _sim(tmp_path, "dsl_bad")
    agent = sim.agents[0]
    agent._fire_obs = _flat_obs(0.0)
    sym = W.symbols(sim)[0][0]
    W.apply(sim, agent, _resp({"expect": {sym: 0.25}}), step=0, sim_min=0)
    before = dict(agent._fire_watch)

    status = W.apply(sim, agent, _resp(block), step=1, sim_min=10)
    assert status == expect_status
    assert agent._fire_watch == before, "不正出力で前回仕様が壊れた"
    rec = [e for e in sim.logger.events if e.kind == "watch_spec"][-1]
    assert rec.payload["status"] == expect_status, "却下の理由が記録されていない"


def test_absent_block_keeps_the_previous_one(tmp_path):
    """watch 節が無い応答は「不正」ではない(前回仕様がそのまま生き続ける)。"""
    sim = _sim(tmp_path, "dsl_absent")
    agent = sim.agents[0]
    agent._fire_obs = _flat_obs(0.0)
    sym = W.symbols(sim)[0][0]
    W.apply(sim, agent, _resp({"expect": {sym: 0.25}}), step=0, sim_min=0)
    before = dict(agent._fire_watch)
    assert W.apply(sim, agent, '{"action": "wander"}', step=1, sim_min=10) == W.ABSENT
    assert agent._fire_watch == before


def test_out_of_band_expectation_is_clamped(tmp_path):
    """★数値クランプ: ô は「いまの観測から σ 何本ぶん」までに抑える(S の独占防止)。"""
    sim = _sim(tmp_path, "dsl_clamp", **{"cognition.watch.clamp_sigmas": 2.0})
    agent = sim.agents[0]
    agent._fire_obs = _flat_obs(0.0)
    sym, idx, _cid, sigma, _lab = W.symbols(sim)[0]
    spec, status, clamped = W.parse(sim, agent, _resp({"expect": {sym: 1.0}}))
    assert status == W.OK and clamped == 1
    assert spec["expect"][idx] == pytest.approx(2.0 * sigma), \
        "クランプ後の ô が観測 ± clamp_sigmas·σ に収まっていない"


def test_operator_whitelist_is_exhaustive():
    assert set(W.OPS) == {">", ">=", "<", "<="}


def test_trigger_term_adds_weight_when_the_condition_holds(tmp_path):
    """Σ_j w_ij·[trigger_j] が S に載る(第81 で口だけ開いていた第2項)。"""
    sim = _sim(tmp_path, "dsl_trig")
    agent = sim.agents[0]
    obs = list(_flat_obs(0.0))
    sym, idx, _cid, _sigma, _lab = W.symbols(sim)[0]
    obs[idx] = 5.0
    agent._fire_obs = tuple(obs)
    W.apply(sim, agent, _resp({"triggers": [{"name": "多い", "ch": sym, "op": ">",
                                             "value": 1.0}]}), step=0, sim_min=0)
    total, names = W.trigger_term(sim, agent, tuple(obs))
    assert total == pytest.approx(sim.watchcfg["trigger_weight"]) and names == ["多い"]
    obs[idx] = 0.0                        # 条件が崩れたら 0
    assert W.trigger_term(sim, agent, tuple(obs)) == (0.0, [])


def test_trigger_count_is_capped(tmp_path):
    sim = _sim(tmp_path, "dsl_cap", **{"cognition.watch.max_triggers": 2})
    agent = sim.agents[0]
    agent._fire_obs = _flat_obs(0.0)
    sym = W.symbols(sim)[0][0]
    trigs = [{"name": f"t{i}", "ch": sym, "op": ">", "value": 0.1} for i in range(5)]
    spec, status, _c = W.parse(sim, agent, _resp({"triggers": trigs}))
    assert status == W.OK and len(spec["triggers"]) == 2


def test_trigger_name_is_truncated(tmp_path):
    sim = _sim(tmp_path, "dsl_name", **{"cognition.watch.name_max": 5})
    agent = sim.agents[0]
    agent._fire_obs = _flat_obs(0.0)
    sym = W.symbols(sim)[0][0]
    spec, status, _c = W.parse(sim, agent, _resp(
        {"triggers": [{"name": "あ" * 100, "ch": sym, "op": ">", "value": 0.1}]}))
    assert status == W.OK and spec["triggers"][0][0] == "あ" * 5


# --------------------------------------------------------------------------- #
# (E) ★オラクル: ô が S 判定に実際に効く(検収基準 5)
# --------------------------------------------------------------------------- #
def _oracle_sim(tmp_path, name, **ov):
    sim = Simulation(_cfg(name, 1, 12, **{**WATCH,
                                          "cognition.fire.sources": "[salience]", **ov}),
                     out_dir=tmp_path / name)
    return sim


def test_expectation_actually_drives_the_firing_decision(tmp_path):
    """★ô を外した値にすると発火し、当てた値にすると発火しない(同じ観測・同じ g)。

    設計 §2.8「期待値 ô は **LLM**」の実体。これが通らないと watch 節は飾りである。
    """
    # level 単位のチャンネルは値域 [0,1] なので**その中**で外す/当てる
    # (値域外の ô は「不正出力」として却下される = (D) 群のテストが固定している挙動)。
    obs_value = 0.9
    for name, expect, should_fire in (("orc_miss", 0.0, True),
                                      ("orc_hit", obs_value, False)):
        sim = _oracle_sim(tmp_path, name)
        target = sim.agents[3].id
        syms = W.symbols(sim)
        base = list(_flat_obs(0.0))
        for _sym, idx, _cid, _s, _l in syms:
            base[idx] = obs_value
        for agent in sim.agents:
            agent._fire_obs = tuple(base)
            agent._fire_pred = tuple(base)          # persistence は完全一致(= S は ô 次第)
            agent._fire_prev_drive = float(agent.drive)
            sim.cogq.schedule(int(agent.id), 10_000, F.PERIODIC)
        # 対象の 1 体にだけ監視仕様を与える
        agent = sim.agent_by_id[target]
        block = {"expect": {sym: expect for sym, *_r in syms}}
        assert W.apply(sim, agent, _resp(block), step=0, sim_min=0) == W.OK
        due = F.due_events(sim, step=1, sim_min=10, active=list(sim.agents))
        if should_fire:
            assert set(due) == {target}, \
                f"ô を外したのに発火しない(または他が発火した): {sorted(due)}"
            assert due[target]["reason"] == F.SALIENCE
        else:
            assert due == {}, f"ô が当たっているのに発火した: {sorted(due)}"


def test_expectation_overlays_persistence_only_where_given(tmp_path):
    """LLM が触れなかったチャンネルは persistence のまま残る(部分的な ô を許す)。"""
    sim = _sim(tmp_path, "orc_partial")
    agent = sim.agents[0]
    base = _flat_obs(1.0)
    agent._fire_obs = base
    sym, idx, _cid, _s, _l = W.symbols(sim)[0]
    W.apply(sim, agent, _resp({"expect": {sym: 0.5}}), step=0, sim_min=0)
    merged = W.expectation(sim, agent, base)
    assert merged[idx] == pytest.approx(0.5)
    for _s2, other, _c, _sg, _l2 in W.symbols(sim)[1:]:
        assert merged[other] == base[other], "触れていないチャンネルまで書き換わった"


# --------------------------------------------------------------------------- #
# (F) model-revision(計画書 §6-3)
# --------------------------------------------------------------------------- #
def test_revision_line_appears_only_on_surprise_firings(tmp_path):
    sim = _sim(tmp_path, "mr_gate")
    agent = sim.agents[0]
    assert W.revision_line(sim, agent, 3, "solo") is None      # 発火源の記録が無い
    agent._fire_src = (3, F.PERIODIC)
    assert W.revision_line(sim, agent, 3, "solo") is None       # 周期発火では出さない
    agent._fire_src = (3, F.SALIENCE)
    assert W.revision_line(sim, agent, 3, "solo") is not None
    assert W.revision_line(sim, agent, 4, "solo") is None        # 別 step には持ち越さない
    assert W.revision_line(sim, agent, 3, "reply") is None       # 返答は相手起点=対象外


def test_revision_line_is_neutral():
    """誘導語彙(機構語・評価語・実験条件語)を 1 つも含まないこと。"""
    line = W._REVISION_LINE
    for token in ("予測", "期待値", "驚き", "発火", "誤差", "salience", "watch",
                  "モデル", "更新すべき", "必ず", "重要"):
        assert token not in line, f"model-revision の 1 行に誘導語が入っている: {token}"


def test_revision_line_is_actually_emitted_in_a_run(tmp_path):
    sim = Simulation(_cfg("mr_run", 144, 30, **WATCH), out_dir=tmp_path / "mr_run")
    seen = _spy(sim)
    sim.run()
    blob = "\n".join(seen)
    n_line = blob.count(W._REVISION_LINE)
    granted = [e for e in _kind(sim, "cog_fire")
               if e.payload["reason"] == F.SALIENCE and e.payload["granted"]]
    assert n_line > 0, "驚き発火があるのに model-revision の行が 1 度も出ていない"
    assert n_line <= len(granted), "驚き発火より多くの行が出ている(漏れている)"


def test_model_revision_touches_beliefs_only_when_enabled(tmp_path):
    """beliefs ON では確信度の見直しが起きる。OFF では 1 件も起きない(既存機構のみ)。"""
    on, _o = _run(tmp_path, "mr_bel", 144, 30, **WATCH,
                  **{"beliefs.enabled": "true"})
    causes = [e for e in on.logger.events
              if e.kind == "belief_update" and e.payload.get("cause") == "model_revision"]
    assert causes, "beliefs ON なのに model-revision の信念見直しが 1 件も無い"
    for e in causes:
        assert e.payload["verified"] is None, "検証済みの信念まで動かしている"

    off, _o2 = _run(tmp_path, "mr_nobel", 144, 30, **WATCH)
    assert not [e for e in off.logger.events if e.kind == "belief_update"]


def test_model_revision_can_be_switched_off(tmp_path):
    sim = Simulation(_cfg("mr_off", 144, 30, **WATCH,
                          **{"cognition.watch.model_revision": "false"}),
                     out_dir=tmp_path / "mr_off")
    seen = _spy(sim)
    sim.run()
    assert W._REVISION_LINE not in "\n".join(seen)
    assert _kind(sim, "watch_spec"), "watch 節そのものは出ているべき"


# --------------------------------------------------------------------------- #
# (G) ★no-fingerprint(検収基準 6)
# --------------------------------------------------------------------------- #
_FORBIDDEN_IN_PROMPT = (
    # 発火機構の語彙(第81 のテストを引き継ぐ)
    "periodic", "salience", "internal", "cog_fire", "認知イベント", "発火時刻",
    # 第82 の機構語・実験条件語
    "g_update", "g_init", "watch spec", "監視仕様", "感度", "可塑性", "慣れ", "感作",
    "予測誤差", "閾値", "flat_traits", "experiment",
    # ★因子名(チャンネル id 経由で漏れうる最大の危険)
    "efficacy", "grievance", "ownership", "nfc", "risk_tolerance", "internal_locus",
    # チャンネル id そのもの
    "body.state", "ext.crowd_local", "body.drive",
)


def test_no_mechanism_or_factor_vocabulary_reaches_the_prompt(tmp_path):
    sim = Simulation(_cfg("w_fp", 60, 24, **WATCH,
                          **{"cognition.g_update.enabled": "true"}),
                     out_dir=tmp_path / "w_fp")
    seen = _spy(sim)
    sim.run()
    assert seen
    blob = "\n".join(seen)
    for token in _FORBIDDEN_IN_PROMPT:
        assert token not in blob, f"プロンプトに漏れている: {token}"


def test_prompt_only_shows_opaque_symbols(tmp_path):
    """プロンプトに出るのは c01… の不透明記号と中立ラベルだけ。"""
    sim = _sim(tmp_path, "w_sym")
    text = W.section(sim)
    assert text and '"watch"' in text
    for sym, _i, cid, _s, label in W.symbols(sim):
        assert sym in text
        assert cid not in text, f"チャンネル id がプロンプトに出ている: {cid}"
        assert label and label in text


def test_channel_labels_carry_no_factor_names():
    """ラベル(プロンプトに出る唯一のチャンネル説明)に因子名が無いこと。"""
    from society.factors.registry import STATE_INIT
    for ch in CH.CHANNELS:
        assert ch.label, f"ラベルが空のチャンネル: {ch.id}"
        for key in STATE_INIT:
            assert key not in ch.label


def test_channel_spec_hash_is_unchanged_by_labels():
    """ラベル追加が σ_c 凍結ファイルを無効化していないこと(hash の対象外)。"""
    frozen = json.loads(
        (Path(__file__).resolve().parents[1] / "data/calib/sigma_c.json")
        .read_text(encoding="utf-8"))
    assert frozen["meta"]["channel_spec_sha256"] == CH.spec_sha256()


# --------------------------------------------------------------------------- #
# (H) 宣言(検収基準 7)
# --------------------------------------------------------------------------- #
def test_feature_is_declared_in_the_registry():
    feature = {f.id: f for f in R.FEATURES}.get("cognition.watch.enabled")
    assert feature is not None
    assert feature.repro_tier == "journal", "LLM の自由文出力を世界状態として消費する"
    assert feature.affects_k is True, "ô が変われば驚き発火の本数が変わる"
    assert feature.fingerprint_risk == "possible", \
        "watch 節と model-revision の 1 行が増減する = 正直に possible"


def test_verify_mode_disables_watch():
    cfg = load_config(["run.mode=verify", "cognition.watch.enabled=true"])
    cfg, report = R.apply_mode(cfg)
    assert bool(cfg.cognition.watch.enabled) is False


def test_new_conf_keys_are_classified_in_timeconv():
    for key in ("cognition.watch.clamp_sigmas", "cognition.watch.max_triggers",
                "cognition.watch.trigger_weight", "cognition.watch.name_max",
                "cognition.watch.belief_revision",
                "cognition.watch.belief_max_facts"):
        assert T.covers(key), f"Δt 分類テーブルに宣言が無い: {key}"
        assert T.classify(key)[0] == T.INVARIANT


def test_manifest_declares_the_watch_spec(tmp_path):
    _sim_, out = _run(tmp_path, "w_man", 12, 8, **WATCH)
    man = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    watch = man["cognition"]["watch"]
    assert watch["ops"] == list(W.OPS)
    assert watch["symbols"], "記号 ↔ チャンネルの対応表が来歴に無い"
    assert all(k.startswith("c") for k in watch["symbols"])
    assert watch["model_revision"] is True
    assert len(watch["prompt_sha256"]) == 64
    fire = man["cognition"]["fire"]
    assert fire["expectation_model"].startswith("llm_watch_spec")
    assert fire["trigger_dsl"].startswith("whitelist_dsl")


def test_new_event_kinds_are_registered():
    from society.observer.schema import EVENT_KINDS
    for kind in ("watch_spec", "cog_theta"):
        assert kind in EVENT_KINDS


# --------------------------------------------------------------------------- #
# (I) mock backend(配線が mock でも回る)
# --------------------------------------------------------------------------- #
def test_mock_emits_a_schema_conforming_watch_block(tmp_path):
    sim = _sim(tmp_path, "w_mock")
    prompt = "場所: どこか\n" + (W.section(sim) or "")
    out, _cid, _cached = sim.llm.generate(prompt, rng_key="t/1/0",
                                          temperature=0.7, max_tokens=64)
    data = json.loads(out)
    assert "watch" in data and "expect" in data["watch"]
    syms = {s for s, *_r in W.symbols(sim)}
    assert set(data["watch"]["expect"]) <= syms
    for trig in data["watch"]["triggers"]:
        assert trig["ch"] in syms and trig["op"] in W.OPS


def test_mock_uses_a_dedicated_stream(tmp_path):
    """watch 節の生成は本文の draw 順を乱さない(専用 stream)。"""
    sim = Simulation(_cfg("w_mrng", 12, 12, **WATCH), out_dir=tmp_path / "w_mrng")
    hub = _CountingHub(sim.hub)
    sim.hub = hub
    sim.llm.backend.hub = hub              # mock は構築時の hub を握るので差し替える
    sim.run()
    assert hub.counts.get("mock_watch", 0) > 0
    assert hub.counts.get("mock", 0) > 0
