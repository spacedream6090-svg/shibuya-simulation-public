"""第78バッチ: アブレーション 4 種(llm_off / propagation_off / cognitive_tier /
shuffle_partners)のテスト。

受入基準(docs/plans/dual-mode-observe-verify-plan.md §2 第78行 /
        docs/plans/source/dual-mode-instructions.md Phase 1):
  - 全スイッチ既定 OFF で純粋既定と **L1 バイト一致**(ゴールデン golden_baseline_l1.json を守る)
  - llm_off        … LLM 呼 0・llm_* イベント 0・それでも世界は完走する
  - propagation_off… 他エージェントへ内容が渡らない(transmission/adopt/heard 記憶が 0)。
                     **LLM 呼の発生箇所は 1 つも消えない**(発話生成も返答権付与も従来どおり)
  - cognitive_tier … rule=llm_off と同一 / full=既定と同一 / fleet へ強制割当 /
                     fleet 非使用ランでの縮退を正直に宣言する
  - shuffle_partners… 相手選択が一様乱択になる。OFF は 1 バイトも変わらない(always-draw)
"""
from __future__ import annotations

import json

import pytest

from society import ablate as A
from society.config import load_config
from society.engine import scheduler
from society.engine.simulation import Simulation
from society.llm.fleet import FleetLLM


# --------------------------------------------------------------------------- #
def _cfg(name: str, n_steps: int = 24, n_agents: int = 12, **ov):
    dot = ["run.seed=42", f"run.n_agents={n_agents}", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def _run(tmp_path, name: str, n_steps: int = 24, n_agents: int = 12, **ov):
    sim = Simulation(_cfg(name, n_steps, n_agents, **ov), out_dir=tmp_path / name)
    summary = sim.run()
    return sim, summary


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _mem_heard(sim) -> int:
    return sum(1 for a in sim.agents for e in a.mem.buffer if e.kind == "heard")


# --------------------------------------------------------------------------- #
# (0) 設定の正準化
# --------------------------------------------------------------------------- #
def test_defaults_are_all_off():
    cfg = A.build_cfg(None)
    assert cfg == {"llm_off": False, "propagation_off": False,
                   "cognitive_tier": "full", "shuffle_partners": False,
                   # 第89バッチ: プラセボ L1 3 種(同一軸=相互排他。既定は全 OFF)
                   "context_shuffle": False, "persona_swap": False,
                   "context_sever": False}
    assert A.build_cfg({}) == cfg


def test_shipped_config_defaults_are_all_off():
    """出荷 conf の ablate ブロックが**全 OFF** であること(R1: 新機能は既定 OFF)。"""
    cfg = load_config()
    assert A.build_cfg(cfg.get("ablate")) == A.build_cfg(None)


def test_unknown_tier_raises():
    with pytest.raises(ValueError):
        A.build_cfg({"cognitive_tier": "genius"})


def test_predicates_without_sim_are_all_false():
    class _Bare:
        pass
    bare = _Bare()
    assert not A.llm_off(bare) and not A.propagation_off(bare)
    assert not A.shuffle_partners(bare) and not A.any_on(bare)
    assert A.describe(bare) is None


# --------------------------------------------------------------------------- #
# (1) R1: 全 OFF は純粋既定とバイト一致
# --------------------------------------------------------------------------- #
def test_all_off_matches_pure_default(tmp_path):
    """ablate ブロックを明示的に全 OFF で書いても純粋既定と L1 バイト一致。"""
    a, _ = _run(tmp_path, "ab_base")
    b, _ = _run(tmp_path, "ab_off", **{"ablate.llm_off": "false",
                                       "ablate.propagation_off": "false",
                                       "ablate.cognitive_tier": "full",
                                       "ablate.shuffle_partners": "false"})
    assert _l1(a) == _l1(b)


def test_all_off_does_not_add_manifest_key(tmp_path):
    """全 OFF のランは run_manifest.json に ablate キーを足さない(既存ランと同形)。"""
    _sim, _ = _run(tmp_path, "ab_man_off", n_steps=6)
    man = json.loads((tmp_path / "ab_man_off" / "run_manifest.json")
                     .read_text(encoding="utf-8"))
    assert "ablate" not in man


# --------------------------------------------------------------------------- #
# (2) ablate.llm_off
# --------------------------------------------------------------------------- #
def test_llm_off_makes_zero_llm_calls(tmp_path):
    sim, summary = _run(tmp_path, "ab_llmoff", **{"ablate.llm_off": "true"})
    assert summary["llm_calls"] == 0
    kinds = summary["event_kinds"]
    for k in ("llm_deliberate", "reflect", "day_plan", "speak", "sns_post", "dm"):
        assert kinds.get(k, 0) == 0, f"llm_off なのに {k} が出ている"
    assert sim.logger.llm_calls == [], "L1b に LLM 呼の行が残っている"


def test_llm_off_world_still_runs(tmp_path):
    """LLM が無くてもルール層(routine.decide)だけで世界は完走し、人は動く。"""
    _sim, summary = _run(tmp_path, "ab_llmoff2", **{"ablate.llm_off": "true"})
    kinds = summary["event_kinds"]
    assert summary["n_events"] > 0
    assert kinds.get("arrive", 0) + kinds.get("route_start", 0) > 0, \
        "移動が 1 件も起きていない(ルール層への後退が効いていない)"


def test_llm_off_is_deterministic(tmp_path):
    a, _ = _run(tmp_path, "ab_llmoff_d1", **{"ablate.llm_off": "true"})
    b, _ = _run(tmp_path, "ab_llmoff_d2", **{"ablate.llm_off": "true"})
    assert _l1(a) == _l1(b)


def test_llm_off_llm_speak_returns_none_without_calling(tmp_path):
    """構造の確認: _llm_speak が **backend を呼ばずに** None を返す。"""
    sim = Simulation(_cfg("ab_seam", n_steps=1), out_dir=tmp_path / "ab_seam")
    sim.ablatecfg = A.build_cfg({"llm_off": True})
    agent = sim.agents[0]
    before = sim.llm.calls
    assert scheduler._llm_speak(sim, agent, "solo", 0, 0) is None
    assert sim.llm.calls == before


# --------------------------------------------------------------------------- #
# (3) ablate.propagation_off
# --------------------------------------------------------------------------- #
def test_propagation_off_blocks_cross_agent_content(tmp_path):
    base, s_base = _run(tmp_path, "ab_prop_base", n_steps=144, n_agents=20)
    off, s_off = _run(tmp_path, "ab_prop_off", n_steps=144, n_agents=20,
                      **{"ablate.propagation_off": "true"})
    # 伝播の実体(語彙の授受)が完全に止まる
    assert s_base["event_kinds"].get("transmission", 0) > 0, "対照の前提が崩れている"
    assert s_off["event_kinds"].get("transmission", 0) == 0
    assert s_off["event_kinds"].get("label_adopt", 0) == 0
    # 他者の発話が記憶に積まれない
    assert _mem_heard(base) >= 0
    assert _mem_heard(off) == 0
    # ★交流の**量**は保つ: 発話も hear も起き続ける(会話が消える ablation ではない)
    assert s_off["event_kinds"].get("speak", 0) > 0
    assert s_off["event_kinds"].get("hear", 0) > 0


def test_propagation_off_keeps_speaker_self_continuity(tmp_path):
    """話者側の自己連続性(自分の発話の記憶・said)は保たれる。"""
    off, _ = _run(tmp_path, "ab_prop_self", n_steps=144, n_agents=20,
                  **{"ablate.propagation_off": "true"})
    said = sum(1 for a in off.agents for e in a.mem.buffer if e.kind == "said")
    assert said > 0, "自分の発話まで消えている(自己連続性が壊れた)"


def test_propagation_off_still_grants_reply_and_calls_llm(tmp_path):
    """**LLM 呼の発生箇所が 1 つも消えない**ことの構造的確認。

    (a) 発話の適用で返答権(_reply_to)は従来どおり相手に渡る = 返答の LLM 呼は撃たれる
    (b) _llm_speak は backend を 1 回呼ぶ(llm_off と違い呼を落とさない)
    """
    for prop_off in (False, True):
        name = f"ab_prop_seam_{int(prop_off)}"
        sim = Simulation(_cfg(name, n_steps=1), out_dir=tmp_path / name)
        sim.ablatecfg = A.build_cfg({"propagation_off": prop_off})
        a, b = sim.agents[0], sim.agents[1]
        b.x, b.y, b.node, b.loc = a.x, a.y, a.node, a.loc
        b.sleeping = False
        b.conv_turns_left = 3
        b.conv_cooldown_until = 0
        scheduler._apply(sim, a, {"type": "speak", "text": "こんにちは、いい天気ですね",
                                  "use_items": []}, 0, 0)
        assert b._reply_to is not None, "返答権が渡らない=返答の LLM 呼が消える"
        before = sim.llm.calls
        scheduler._llm_speak(sim, a, "solo", 0, 0)
        assert sim.llm.calls == before + 1, "発話生成そのものが実行されていない"


def test_propagation_off_reply_prompt_has_no_utterance(tmp_path):
    """返答プロンプトに相手の発話文が 1 文字も入らない(状況行ごと出さない)。"""
    from society.cognition import deliberate
    agent = None
    sim = Simulation(_cfg("ab_prop_prompt", n_steps=1),
                     out_dir=tmp_path / "ab_prop_prompt")
    agent = sim.agents[0]
    with_text = deliberate.build_prompt(agent, place_name="街", surprise="reply",
                                        nearby_names=[], nearby_ids=[],
                                        sim_min=0, step=0, nearby_pois=[],
                                        reply_to=("太郎", "秘密のあいことば"))
    without = deliberate.build_prompt(agent, place_name="街", surprise="reply",
                                      nearby_names=[], nearby_ids=[],
                                      sim_min=0, step=0, nearby_pois=[],
                                      reply_to=None)
    assert "秘密のあいことば" in with_text
    assert "秘密のあいことば" not in without
    assert "話しかけられた" not in without, "内容抜きの合成状況文を注入してはいけない"


def test_propagation_off_call_count_stays_close(tmp_path):
    """呼数の実測(正直な記録)。

    構造上どの呼び出し点も消していないが、**知覚が変われば欲求ゲージ経由で発火数は
    間接的に動く**(registry が affects_k=False の機能について明記している性質と同型)。
    したがって完全一致は原理的に保証できない。ここでは「桁で違わない」ことだけを固定し、
    厳密な呼数一致が要るときは controls.mode=compute_matched を併用する。
    """
    _b, s_base = _run(tmp_path, "ab_prop_k0", n_steps=144, n_agents=20)
    _o, s_off = _run(tmp_path, "ab_prop_k1", n_steps=144, n_agents=20,
                     **{"ablate.propagation_off": "true"})
    base, off = s_base["llm_calls"], s_off["llm_calls"]
    assert base > 0
    assert abs(off - base) / base < 0.10, f"呼数が大きくずれた: {base} -> {off}"


def test_propagation_off_records_manifest(tmp_path):
    _sim, _ = _run(tmp_path, "ab_prop_man", n_steps=6,
                   **{"ablate.propagation_off": "true"})
    man = json.loads((tmp_path / "ab_prop_man" / "run_manifest.json")
                     .read_text(encoding="utf-8"))
    assert man["ablate"]["propagation_off"] is True
    assert man["ablate"]["cognitive_tier"] == "full"


# --------------------------------------------------------------------------- #
# (4) ablate.cognitive_tier
# --------------------------------------------------------------------------- #
def test_tier_full_is_byte_identical(tmp_path):
    a, _ = _run(tmp_path, "ab_tier_base")
    b, _ = _run(tmp_path, "ab_tier_full", **{"ablate.cognitive_tier": "full"})
    assert _l1(a) == _l1(b)


def test_tier_rule_equals_llm_off(tmp_path):
    a, _ = _run(tmp_path, "ab_tier_rule", **{"ablate.cognitive_tier": "rule"})
    b, _ = _run(tmp_path, "ab_tier_llmoff", **{"ablate.llm_off": "true"})
    assert _l1(a) == _l1(b)


def test_tier_small_degenerates_without_fleet(tmp_path):
    """FleetLLM 非使用ラン(mock)では small/mid は**効かない**ことを正直に宣言する。"""
    sim, _ = _run(tmp_path, "ab_tier_small", n_steps=6,
                  **{"ablate.cognitive_tier": "small"})
    eff = A.tier_effectiveness(sim)
    assert eff["applied"] is False and "縮退" in eff["reason"]
    man = json.loads((tmp_path / "ab_tier_small" / "run_manifest.json")
                     .read_text(encoding="utf-8"))
    assert man["ablate"]["tier_effective"]["applied"] is False


def test_tier_small_on_mock_is_byte_identical_to_full(tmp_path):
    """縮退の中身: fleet が無ければ small は full と 1 バイトも変わらない。"""
    a, _ = _run(tmp_path, "ab_tier_f2")
    b, _ = _run(tmp_path, "ab_tier_s2", **{"ablate.cognitive_tier": "small"})
    assert _l1(a) == _l1(b)


def test_fleet_force_tier_routes_all_purposes():
    """FleetLLM の force_tier が purpose を無視して指定プールへ寄せる(単体)。"""
    fleet = FleetLLM(["http://a:8000"], "m",
                     tiers={"default": ["http://a:8000"],
                            "small": ["http://s:8000"],
                            "mid": ["http://m:8000"]})
    assert fleet._pool("deliberate") == ["http://a:8000"]
    fleet.force_tier = "small"
    for purpose in ("deliberate", "reflect", "plan", ""):
        assert fleet._pool(purpose) == ["http://s:8000"]
    fleet.force_tier = "mid"
    assert fleet._pool("deliberate") == ["http://m:8000"]
    fleet.force_tier = "nonexistent"               # 未定義ティアは従来の割当へ後退
    assert fleet._pool("deliberate") == ["http://a:8000"]


def test_fleet_force_tier_is_applied_by_ablate():
    """ablate.apply_fleet_tier が fleet を見つけて焼き付ける(CachedLLM 越しでも)。"""
    from society.llm.cache import CachedLLM

    class _Sim:
        pass
    fleet = FleetLLM(["http://a:8000"], "m",
                     tiers={"default": ["http://a:8000"], "mid": ["http://m:8000"]})
    sim = _Sim()
    sim.llm = CachedLLM(fleet, enabled=False)
    sim.ablatecfg = A.build_cfg({"cognitive_tier": "mid"})
    assert A.tier_effectiveness(sim)["applied"] is True
    A.apply_fleet_tier(sim)
    assert fleet.force_tier == "mid"
    assert fleet._pool("deliberate") == ["http://m:8000"]


def test_tier_does_not_leak_real_time(tmp_path):
    """実時間の漏れ防止(Part D(2)): ティアを変えても世界時刻は sim clock だけで決まる。"""
    a, _ = _run(tmp_path, "ab_tier_t1", n_steps=12)
    b, _ = _run(tmp_path, "ab_tier_t2", n_steps=12,
                **{"ablate.cognitive_tier": "mid"})
    assert [e.sim_min for e in a.logger.events] == [e.sim_min for e in b.logger.events]


# --------------------------------------------------------------------------- #
# (5) ablate.shuffle_partners
# --------------------------------------------------------------------------- #
def test_shuffle_off_is_byte_identical(tmp_path):
    """always-draw(専用 stream)なので OFF でも既存 stream の draw 順は不変。"""
    a, _ = _run(tmp_path, "ab_shuf_base")
    b, _ = _run(tmp_path, "ab_shuf_off", **{"ablate.shuffle_partners": "false"})
    assert _l1(a) == _l1(b)


def test_pick_partner_always_draws():
    """OFF でも必ず 1 本引く(ON/OFF で draw 数が変わらない)。"""
    class _Hub:
        def __init__(self):
            self.keys = []

        def stream(self, *key):
            from society.rng import RngHub
            self.keys.append(key)
            return RngHub(7).stream(*key)

    class _Sim:
        pass

    class _A:
        def __init__(self, i):
            self.id = i
            self.now_step = 3
    hearers = [_A(1), _A(2), _A(3)]
    for on in (False, True):
        sim = _Sim()
        sim.hub = _Hub()
        sim.ablatecfg = A.build_cfg({"shuffle_partners": on})
        got = A.pick_partner(sim, _A(0), hearers)
        assert len(sim.hub.keys) == 1, "always-draw が守られていない"
        assert sim.hub.keys[0][0] == A.STREAM
        assert (got is not None) is on
    # 候補ゼロなら引かない(無意味な消費をしない)
    sim = _Sim()
    sim.hub = _Hub()
    sim.ablatecfg = A.build_cfg({"shuffle_partners": True})
    assert A.pick_partner(sim, _A(0), []) is None
    assert sim.hub.keys == []


def test_shuffle_partners_replaces_structural_ranking(tmp_path):
    """closeness 最大の相手が必ず選ばれる状況で、ON にすると選択が乱択へ変わる。"""
    sim = Simulation(_cfg("ab_shuf_unit", n_steps=1),
                     out_dir=tmp_path / "ab_shuf_unit")
    speaker = sim.agents[0]
    hearers = sim.agents[1:6]
    for i, h in enumerate(hearers):                # 距離を id 昇順に単調にする
        h.x, h.y = speaker.x + 1.0 + i, speaker.y
    # OFF(既定 nearest)は必ず最寄り = hearers[0]
    sim.ablatecfg = A.build_cfg(None)
    assert scheduler._select_partner(sim, speaker, hearers) is hearers[0]
    # ON は step を変えると別の相手が出る(= 構造的順位づけではない)
    sim.ablatecfg = A.build_cfg({"shuffle_partners": True})
    picked = set()
    for step in range(40):
        speaker.now_step = step
        picked.add(scheduler._select_partner(sim, speaker, hearers).id)
    assert len(picked) > 1, "ON なのに常に同じ相手(乱択になっていない)"
    assert picked <= {h.id for h in hearers}


def test_shuffle_partners_run_completes_and_changes_partners(tmp_path):
    _sim, s_on = _run(tmp_path, "ab_shuf_on", n_steps=144, n_agents=20,
                      **{"ablate.shuffle_partners": "true"})
    _sim2, s_off = _run(tmp_path, "ab_shuf_off2", n_steps=144, n_agents=20)
    assert s_on["n_events"] > 0
    # 会話が消えないこと(相手が変わるだけ)
    assert s_on["event_kinds"].get("speak", 0) > 0
    assert s_on["event_kinds"] != s_off["event_kinds"], \
        "相手選択を乱択にしても何も変わらない(効いていない)"


# --------------------------------------------------------------------------- #
# (6) レジストリ宣言(未宣言検出の相棒)
# --------------------------------------------------------------------------- #
def test_registry_declares_all_ablation_switches():
    from society import registry as R
    want = {"ablate.llm_off", "ablate.propagation_off", "ablate.cognitive_tier",
            "ablate.shuffle_partners", "observer.state_hash.enabled"}
    assert want <= set(R.BY_ID), f"未宣言: {sorted(want - set(R.BY_ID))}"
    # llm_off / cognitive_tier は呼数を変える(rule 段でゼロになる)= 正直な宣言
    assert R.BY_ID["ablate.llm_off"].affects_k is True
    assert R.BY_ID["ablate.cognitive_tier"].affects_k is True
    assert R.BY_ID["ablate.cognitive_tier"].off_value == "full"
    # propagation_off は「合成文を入れない」方針の帰結として known
    assert R.BY_ID["ablate.propagation_off"].fingerprint_risk == "known"
    # アブレーションは verify モード(strict のみ許可)で落ちてはならない
    assert all(R.BY_ID[i].repro_tier == "strict" for i in want)


def test_ablations_survive_verify_mode(tmp_path):
    """run.mode=verify でもアブレーションは自動 OFF されない(対照実験の道具だから)。"""
    sim, _ = _run(tmp_path, "ab_verify", n_steps=6,
                  **{"run.mode": "verify", "ablate.propagation_off": "true"})
    assert A.propagation_off(sim) is True
    dropped = {d["id"] for d in sim.run_mode_report["auto_disabled"]}
    assert not any(i.startswith("ablate.") for i in dropped)
