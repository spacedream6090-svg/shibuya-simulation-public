"""D7 対照(controls)+ D9 過正当化(rewards)の配線テスト。

- compute_matched: off でも内省 LLM を実行して全破棄 → off と sham の reflect 呼数が一致
  (計算量・トークンを条件間で揃える = R1 交絡の排除)。
- null_series: 発火ごとに固定本数の内容非結合ダミー呼び出し(llm_null)。
- controls=none: 旧挙動とイベント列が完全一致(ゴールデン)。
- rewards.enabled: 採用に外的報酬(reward)。既定 off では発生しない(過正当化 ablation)。
"""
from __future__ import annotations

from society.config import load_config
from society.engine.simulation import Simulation


def _run(tmp_path, name, n_agents=12, n_steps=144, **overrides):
    dot = [f"run.n_agents={n_agents}", f"run.n_steps={n_steps}",
           f"run.name={name}", "observer.snapshot_every=60"]
    dot += [f"{k}={v}" for k, v in overrides.items()]
    cfg = load_config(dot)
    sim = Simulation(cfg, out_dir=tmp_path / name)
    sim.run()
    return sim


def _events(sim, kind):
    return [e for e in sim.logger.events if e.kind == kind]


def test_compute_matched_off_matches_sham_reflect_count(tmp_path):
    """compute_matched では off でも内省 LLM を実行 → off と sham の reflect 呼数が一致。"""
    off = _run(tmp_path, "cm_off", **{"controls.mode": "compute_matched",
                                      "k.writeback": "off"})
    sham = _run(tmp_path, "cm_sham", **{"controls.mode": "compute_matched",
                                        "k.writeback": "sham"})
    n_off = len(_events(off, "reflect"))
    n_sham = len(_events(sham, "reflect"))
    assert n_off > 0 and n_sham > 0, "内省が発生していない(steps を増やす)"
    assert n_off == n_sham, f"compute_matched で off={n_off} != sham={n_sham}"
    # off は結果を全破棄 = belief を書き戻さない
    assert all(not e.payload["written_back"] for e in _events(off, "reflect"))
    # controls キーが reflect payload に載る(none 以外)
    assert all(e.payload.get("controls") == "compute_matched"
               for e in _events(off, "reflect"))


def test_compute_matched_activates_off_reflection(tmp_path):
    """none+off は内省ゼロ、compute_matched+off は内省が走る(計算量マッチの要)。"""
    none_off = _run(tmp_path, "none_off", **{"controls.mode": "none",
                                             "k.writeback": "off"})
    cm_off = _run(tmp_path, "cm_off2", **{"controls.mode": "compute_matched",
                                          "k.writeback": "off"})
    assert len(_events(none_off, "reflect")) == 0
    assert len(_events(cm_off, "reflect")) > 0


def test_null_series_emits_fixed_dummy_calls(tmp_path):
    """null_series: llm_null イベントが granted 数 × null_calls 本 出る。"""
    for nc in (1, 2):
        sim = _run(tmp_path, f"null{nc}", **{"controls.mode": "null_series",
                                             "controls.null_calls": nc})
        granted = sum(1 for e in _events(sim, "drive_request")
                      if e.payload["granted"])
        n_null = len(_events(sim, "llm_null"))
        assert granted > 0
        assert n_null == granted * nc, f"null_calls={nc}: {n_null} != {granted*nc}"
    # none では llm_null は一切出ない(現状不変)
    base = _run(tmp_path, "null_none", **{"controls.mode": "none"})
    assert len(_events(base, "llm_null")) == 0


def test_controls_none_is_golden(tmp_path):
    """controls=none は controls 指定なしと完全にイベント列一致(旧挙動保存)。"""
    a = _run(tmp_path, "g_none", **{"controls.mode": "none"})
    b = _run(tmp_path, "g_unspec")            # controls 未指定(= 既定 none)
    ea = [(e.step, e.agent_id, e.kind, e.payload) for e in a.logger.events]
    eb = [(e.step, e.agent_id, e.kind, e.payload) for e in b.logger.events]
    assert ea == eb
    # none では reflect payload に controls キーを足さない(旧スキーマ厳守)
    assert all("controls" not in e.payload for e in _events(a, "reflect"))


def test_rewards_default_off_no_reward_events(tmp_path):
    sim = _run(tmp_path, "rw_off", n_agents=20, n_steps=200)
    assert len(_events(sim, "reward")) == 0


def test_rewards_enabled_pays_creator(tmp_path):
    """rewards.enabled: 造語が他者に採用されると創出者に報酬(reward イベント)。"""
    sim = _run(tmp_path, "rw_on", n_agents=20, n_steps=200,
               **{"rewards.enabled": "true", "rewards.amount_per_adoption": 300})
    rewards = _events(sim, "reward")
    assert rewards, "報酬イベントが出ていない(採用が起きていない)"
    for e in rewards:
        assert e.payload["amount"] == 300.0
        assert e.payload["balance"] >= 300.0
