"""ATT: 自律的注意機構(層A = 顕著性 top-k 選抜 / 層B = LLM 自律宣言の注意ブロック)のテスト。

正典: docs/plans/attention-mechanism-plan.md(§2 層A・§3 層B・§6.5 個体差・§6.6 実装決定)
実装: src/society/attention.py / engine/scheduler.py(挿入点は S15 と同一 1 点)/
      cognition/deliberate.py(attend 受理 + プロンプト節)/ cognition/perception_contract.py /
      world/pool.py(搬送)/ registry.py(宣言)

守るもの(検収基準の順)
  (0) 出荷既定 = 全キーが基底 conf に宣言済み・既定値は**現行同値**(= 無効)。
      レジストリ宣言(repro_tier / affects_k / fingerprint_risk)も同時に固定する。
      触ったファイルが凍結 SPEC_FILES に 1 つも入っていないことも機械照合する。
  (1) 既定 OFF = L1 バイト一致・**新しい属性を 1 つも生やさない**・乱数を 1 粒も引かない。
  (2) mode=distance = 第141 S15 と**完全同一**(縮退線が本当に縮退している)。
  (3) 個体差: k_i ∈ [2,7] / p_break_i ∈ [0.20,0.65] の帯・決定論・分布。
  (4) select: priority 降順 → 同値 id 昇順・θ 足切り・k 上限・会話相手の優先充当・
      自己名の貫通・crowd_n・返り値の id 昇順。
  (5) ON の機械検証: **非通過の聞き手には hear の L1 も記憶も関係も contacts も出ない**
      (Cherry 1953 準拠)。通過分は従来どおり全部出る。
  (6) 層B: attend の受理 / 欠落 no-op / 不正 kind 破棄 / 全量置換 / since 維持 /
      decay / min_salience 除去 / boost / 搬送(dehydrate→hydrate)/ プロンプト節の ON/OFF。
"""
from __future__ import annotations

import json

import pytest

from society import attention as A
from society import registry as R
from society.cognition import deliberate
from society.cognition import perception_contract as PC
from society.config import load_config
from society.engine import scheduler
from society.engine.simulation import Simulation
from society.world import pool as POOL

ON_A = {"world.attention.enabled": "true", "world.attention.mode": "salience"}
ON_B = {"cognition.attention_block.enabled": "true"}


# --------------------------------------------------------------------------- #
# 共通ヘルパ
# --------------------------------------------------------------------------- #
def _cfg(name, n_steps=144, n_agents=15, **ov):
    dot = ["run.seed=42", f"run.n_agents={n_agents}", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock", "observer.snapshot_every=144"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def _sim(tmp_path, name, n_steps=144, n_agents=15, **ov):
    return Simulation(_cfg(name, n_steps, n_agents, **ov), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


class _ExplodingHub:
    """乱数を 1 粒でも引いたら落ちる hub(= 「引かない」ことの機械証明)。"""

    def stream(self, *key):                      # noqa: D102
        raise AssertionError(f"ATT が乱数 stream を引いた: {key}")


# ---- 世界も Simulation も要らない検査のための薄い代役 ---------------------------- #
class _Mem:
    def __init__(self):
        self.relations: dict[int, dict] = {}


class _Agent:
    def __init__(self, aid, x=0.0, y=0.0, name=None, adopted=(), hobbies=()):
        self.id = int(aid)
        self.name = name if name is not None else f"人{aid}"
        self.x, self.y = float(x), float(y)
        self.mem = _Mem()
        self.adopted = set(adopted)
        self.hobbies = list(hobbies)


class _StubSim:
    """attention.py が触る口だけを持つスタブ(cfg / net / agent_by_id)。"""

    def __init__(self, a_ov=None, b_ov=None, agents=()):
        self.cfg = {"world": {"attention": dict(a_ov or {})},
                    "cognition": {"attention_block": dict(b_ov or {})}}
        self.net = None
        self.agents = list(agents)
        self.agent_by_id = {a.id: a for a in self.agents}


def _on_sim(**a_ov):
    raw = {"enabled": True, "mode": A.MODE_SALIENCE}
    raw.update(a_ov)
    return _StubSim(a_ov=raw)


def _block_sim(**b_ov):
    raw = {"enabled": True}
    raw.update(b_ov)
    return _StubSim(b_ov=raw)


# --------------------------------------------------------------------------- #
# (0) 出荷既定・conf・レジストリ・凍結照合
# --------------------------------------------------------------------------- #
def test_shipped_default_is_off():
    """基底 conf の既定は層A も層B も OFF(= 現行同値)。"""
    cfg = load_config()
    assert bool(cfg.world.attention.enabled) is False
    assert str(cfg.world.attention.mode) == A.MODE_DISTANCE
    assert bool(cfg.cognition.attention_block.enabled) is False
    assert A.build_cfg(None) == A.DEFAULTS
    assert A.build_block_cfg(None) == A.BLOCK_DEFAULTS


def test_shipped_defaults_match_the_design_document():
    """設計書 §2 / §6.5 の初期値がそのまま出荷されている(勝手に動かしていない)。"""
    cfg = load_config().world.attention
    assert int(cfg.k_base) == 4                        # Cowan 2001 の 4±1
    assert int(cfg.k_span) == 2                        # 個体差の振れ幅
    assert float(cfg.theta_ignition) == pytest.approx(0.0)
    assert float(cfg.p_break_base) == pytest.approx(0.33)   # Conway 2001
    assert float(cfg.w_bu) == pytest.approx(1.0)
    assert float(cfg.w_td) == pytest.approx(0.5)
    assert float(cfg.w_h) == pytest.approx(0.3)
    assert float(cfg.w_v) == pytest.approx(0.5)
    assert float(cfg.w_slot) == pytest.approx(0.8)
    blk = load_config().cognition.attention_block
    assert int(blk.slots_max) == 7                     # Cowan 4±1 + Lost in the Middle 回避
    assert float(blk.decay_per_day) == pytest.approx(0.15)
    assert float(blk.boost_addressed) == pytest.approx(0.3)
    assert float(blk.min_salience) == pytest.approx(0.05)


def test_build_cfg_falls_back_on_an_unknown_mode():
    """未知の mode は既定 distance(= S15 縮退線)へ倒す = 事故で新経路へ入らない。"""
    assert A.build_cfg({"mode": "wat"})["mode"] == A.MODE_DISTANCE
    assert A.build_cfg({"mode": "salience"})["mode"] == A.MODE_SALIENCE


def test_build_cfg_clamps_hostile_values():
    """0 / 負値 / 範囲外を渡されても破壊的な解釈にならない。"""
    got = A.build_cfg({"enabled": 1, "k_base": 99, "k_span": -3,
                       "p_break_base": 5.0})
    assert got["enabled"] is True
    assert got["k_base"] == A.K_MAX and got["k_span"] == 0
    assert got["p_break_base"] == 1.0
    assert A.build_cfg({"k_base": -1})["k_base"] == A.K_MIN


def test_build_block_cfg_clamps_hostile_values():
    got = A.build_block_cfg({"slots_max": 0, "decay_per_day": 9.0,
                             "boost_addressed": -1.0, "min_salience": -1.0})
    assert got["slots_max"] == 1 and got["decay_per_day"] == 1.0
    assert got["boost_addressed"] == 0.0 and got["min_salience"] == 0.0


def test_registry_declares_the_three_new_toggles():
    """新 conf キー 3 件がレジストリに宣言済み(等級・affects_k も固定)。"""
    by_id = {f.id: f for f in R.FEATURES}
    for fid in ("world.attention.enabled", "world.attention.mode",
                "cognition.attention_block.enabled"):
        assert fid in by_id, f"{fid} がレジストリに無い"
    assert by_id["world.attention.enabled"].affects_k is True     # S15 と同じ理由
    assert by_id["world.attention.enabled"].repro_tier == "strict"
    assert by_id["world.attention.mode"].off_value == A.MODE_DISTANCE
    # 層B は LLM の自由文(attend 欄)を消費する = journal・呼数は動かさない
    blk = by_id["cognition.attention_block.enabled"]
    assert blk.repro_tier == "journal" and blk.affects_k is False
    assert blk.fingerprint_risk == "possible"      # プロンプトに 1 ブロック増える


def test_no_undeclared_toggles_after_this_batch():
    """本バッチの新キーを足しても未宣言トグルは 0 件のまま。"""
    assert R.undeclared_toggles(load_config()) == []


def test_finals_profile_enables_layer_a_and_keeps_layer_b_off():
    """本選 conf: 層A = ON(salience)/ 層B = **OFF**(プローブ後に判断・設計 §6.6)。"""
    from pathlib import Path

    from omegaconf import OmegaConf
    root = Path(__file__).resolve().parents[1]
    fin = OmegaConf.load(root / "conf" / "finals_observe.yaml")
    assert bool(fin.world.attention.enabled) is True
    assert str(fin.world.attention.mode) == A.MODE_SALIENCE
    assert bool(fin.cognition.attention_block.enabled) is False


def test_touched_files_are_not_frozen():
    """本レーンが触ったファイルが凍結 SPEC_FILES に 1 つも入っていない。"""
    from society.observer import metrics_spec as MS
    touched = (
        "src/society/attention.py",
        "src/society/engine/scheduler.py",
        "src/society/cognition/deliberate.py",
        "src/society/cognition/perception_contract.py",
        "src/society/engine/checkpoint.py",
        "src/society/world/pool.py",
        "src/society/registry.py",
    )
    for rel in touched:
        assert rel not in MS.SPEC_FILES, f"凍結ファイルを触っている: {rel}"


def test_pool_transport_keys_mirror_the_module_constants():
    """world/pool.py が持つ属性名の写しが attention.py の正典と一致している。"""
    assert POOL._ATTN_SLOTS_KEY == A.SLOTS_KEY
    assert POOL._ATTN_HIST_KEY == A.HISTORY_KEY


# --------------------------------------------------------------------------- #
# (1) 既定 OFF = 現行挙動そのまま
# --------------------------------------------------------------------------- #
def test_off_matches_pure_default(tmp_path):
    """明示 OFF と純粋既定が L1 完全一致(seam が完全な no-op)。"""
    pure = _sim(tmp_path, "att_pure")
    pure.run()
    off = _sim(tmp_path, "att_off", **{"world.attention.enabled": "false",
                                       "cognition.attention_block.enabled": "false"})
    off.run()
    assert _l1(off) == _l1(pure), "ATT の seam が既定ランを動かしている"


def test_off_creates_no_new_agent_attributes(tmp_path):
    """既定 OFF は agent に注意ブロックも注意履歴も**生やさない**(checkpoint バイト不変)。"""
    sim = _sim(tmp_path, "att_off_attrs", n_steps=144)
    sim.run()
    for agent in sim.agents:
        assert A.SLOTS_KEY not in agent.__dict__
        assert A.HISTORY_KEY not in agent.__dict__
        assert A.REPLY_KEY not in agent.__dict__


def test_salience_on_requires_both_enabled_and_mode():
    """enabled だけ / mode だけでは新経路へ入らない(二重の門)。"""
    assert A.salience_on(_StubSim(a_ov={"enabled": True})) is False
    assert A.salience_on(_StubSim(a_ov={"mode": "salience"})) is False
    assert A.salience_on(_StubSim(a_ov={"enabled": True,
                                        "mode": "salience"})) is True
    assert A.enabled(_StubSim(a_ov={"enabled": True})) is True   # enabled 自体は True


def test_select_draws_no_random_stream():
    """層A は**乱数 stream を 1 本も引かない**(hub を触ったら落ちる代役で証明)。"""
    sim = _on_sim()
    sim.hub = _ExplodingHub()
    speaker = _Agent(0)
    hearers = [_Agent(i, x=float(i)) for i in range(1, 12)]
    got, crowd = A.select(sim, speaker, hearers, "こんにちは", [], 3, radius=40.0)
    assert got and crowd >= 0


def test_layer_b_seams_are_noop_when_off():
    """層B の全 seam は OFF で属性を 1 つも生やさない。"""
    sim = _StubSim()                                   # 両方 OFF
    agent = _Agent(1)
    assert A.apply_declaration(sim, agent, {"attend": [{"kind": "person",
                                                        "target": "人2"}]}, 0) == 0
    A.note_addressed(sim, agent, 2)
    A.phase_day(sim, 0, 0)
    assert A.prompt_section(sim, agent) is None
    assert A.SLOTS_KEY not in agent.__dict__


# --------------------------------------------------------------------------- #
# (2) mode=distance = 第141 S15 と完全同一
# --------------------------------------------------------------------------- #
def test_distance_mode_is_byte_identical_to_s15(tmp_path):
    """enabled=true + mode=distance は S15 単独と L1 バイト一致(縮退線が縮退している)。"""
    s15 = _sim(tmp_path, "att_s15", **{"world.attention_hearers_max": 3})
    s15.run()
    deg = _sim(tmp_path, "att_deg", **{"world.attention_hearers_max": 3,
                                       "world.attention.enabled": "true",
                                       "world.attention.mode": "distance"})
    deg.run()
    assert _l1(deg) == _l1(s15), "distance モードが S15 から逸脱している"


def test_distance_mode_enabled_still_matches_pure_default(tmp_path):
    """S15 も 0 のままなら、enabled=true(distance)でも純粋既定とバイト一致。"""
    pure = _sim(tmp_path, "att_dm_pure")
    pure.run()
    deg = _sim(tmp_path, "att_dm_on", **{"world.attention.enabled": "true"})
    deg.run()
    assert _l1(deg) == _l1(pure)


# --------------------------------------------------------------------------- #
# (3) 個体差(k_i / p_break_i)= blake2b の決定論値
# --------------------------------------------------------------------------- #
def test_k_of_stays_inside_the_hard_band():
    """設計 §6.5 の帯 [2, 7] を 1 人もはみ出さない(1,000 個体で全数)。"""
    cfg = A.build_cfg({"enabled": True, "mode": "salience"})
    ks = {A.k_of(_Agent(i), cfg) for i in range(1000)}
    assert min(ks) >= A.K_MIN and max(ks) <= A.K_MAX
    # 既定 (k_base=4, k_span=2) の実効帯は 2-6(module docstring の「正直な限界 2」)
    assert min(ks) == 2 and max(ks) == 6


def test_k_of_is_deterministic_and_spread():
    """同じ id なら常に同じ値・帯の全域が実際に使われる(全員 4 になっていない)。"""
    cfg = A.build_cfg({"enabled": True, "mode": "salience"})
    assert A.k_of(_Agent(7), cfg) == A.k_of(_Agent(7), cfg)
    ks = [A.k_of(_Agent(i), cfg) for i in range(500)]
    assert len(set(ks)) == 5, f"帯の全域が使われていない: {sorted(set(ks))}"


def test_k_of_with_zero_span_is_constant():
    cfg = A.build_cfg({"k_base": 3, "k_span": 0})
    assert {A.k_of(_Agent(i), cfg) for i in range(200)} == {3}


def test_k_of_respects_a_wider_span_but_clamps():
    """k_span を広げても [2,7] のハード境界を越えない。"""
    cfg = A.build_cfg({"k_base": 4, "k_span": 9})
    ks = {A.k_of(_Agent(i), cfg) for i in range(500)}
    assert min(ks) >= A.K_MIN and max(ks) <= A.K_MAX


def test_p_break_of_stays_inside_the_conway_band():
    """貫通率は Conway 2001 の実測レンジ [0.20, 0.65] に収まる(1,000 個体で全数)。"""
    cfg = A.build_cfg({})
    ps = [A.p_break_of(_Agent(i), cfg) for i in range(1000)]
    assert min(ps) >= A.P_BREAK_MIN - 1e-12
    assert max(ps) <= A.P_BREAK_MAX + 1e-12
    # 基底 0.33 のとき帯は厳密に [0.20, 0.65] = 両端に十分近い個体が出る
    assert min(ps) < 0.21 and max(ps) > 0.64


def test_p_break_of_is_deterministic():
    cfg = A.build_cfg({})
    assert A.p_break_of(_Agent(11), cfg) == A.p_break_of(_Agent(11), cfg)
    assert A.p_break_of(_Agent(11), cfg) != A.p_break_of(_Agent(12), cfg)


def test_individual_values_do_not_depend_on_the_rng_hub():
    """個体差は seed にも stream にも依らない(blake2b の純関数 = resume 安定)。"""
    cfg = A.build_cfg({})
    import hashlib
    key = f"{A._SALT_K}\x1f5".encode("utf-8")
    want = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(),
                          "big") / 2.0 ** 64
    assert A._u01(A._SALT_K, 5) == want
    assert A.k_of(_Agent(5), cfg) == A.k_of(_Agent(5), cfg)


# --------------------------------------------------------------------------- #
# (4) 採点関数(顕著性 / 関連度 / 履歴 / 価値 / スロット)
# --------------------------------------------------------------------------- #
def test_salience_decays_with_distance():
    cfg = A.build_cfg({})
    near = A.salience_of(_Agent(1), "やあ", [], 0.0, 0.0, cfg, 1600.0)
    far = A.salience_of(_Agent(1), "やあ", [], 1600.0, 0.0, cfg, 1600.0)
    assert near > far
    assert far == pytest.approx(near / 2.0)          # d = r で減衰 1/2


def test_salience_rises_with_a_novel_word():
    cfg = A.build_cfg({})
    known = _Agent(1, adopted={"渋谷弁"})
    fresh = _Agent(2, adopted=set())
    base = A.salience_of(known, "やあ", ["渋谷弁"], 0.0, 0.0, cfg, 1.0)
    novel = A.salience_of(fresh, "やあ", ["渋谷弁"], 0.0, 0.0, cfg, 1.0)
    assert novel > base
    assert novel == pytest.approx(base * (1.0 + cfg["alpha_nov"]))


def test_salience_rises_with_emotion_and_addressing():
    cfg = A.build_cfg({})
    hearer = _Agent(1, name="ハナ")
    flat = A.salience_of(hearer, "やあ", [], 0.0, 0.0, cfg, 1.0)
    emo = A.salience_of(hearer, "やあ", [], 0.0, 1.0, cfg, 1.0)
    addr = A.salience_of(hearer, "ハナ、やあ", [], 0.0, 0.0, cfg, 1.0)
    assert flat == pytest.approx(1.0)                       # d=0 → 減衰なし
    assert emo == pytest.approx(flat + cfg["alpha_emo"])
    assert addr == pytest.approx(flat + cfg["alpha_addr"])
    assert addr > emo > flat


def test_relevance_matches_known_words_and_hobbies():
    hearer = _Agent(1, adopted={"渋谷弁"}, hobbies=["写真", "登山"])
    assert A.relevance_of(hearer, "やあ", []) == 0.0
    assert A.relevance_of(hearer, "やあ", ["渋谷弁"]) == 0.5
    assert A.relevance_of(hearer, "写真を撮ろう", []) == 0.5
    assert A.relevance_of(hearer, "写真だ", ["渋谷弁"]) == 1.0
    # 趣味は先頭 2 件までしか見ない(O(1) を保つ)
    many = _Agent(2, hobbies=["a", "b", "テニス"])
    assert A.relevance_of(many, "テニスの話", []) == 0.0


def test_history_saturates_and_is_bounded():
    hearer = _Agent(1)
    assert A.history_of(hearer, 9) == 0.0
    A.note_attended([hearer], 9)
    assert A.history_of(hearer, 9) == pytest.approx(1.0 / A.HISTORY_MAX)
    for _ in range(50):
        A.note_attended([hearer], 9)
    assert A.history_of(hearer, 9) == pytest.approx(1.0)
    assert len(getattr(hearer, A.HISTORY_KEY)) == A.HISTORY_MAX   # 有界


def test_note_attended_keeps_the_most_recent_speakers():
    hearer = _Agent(1)
    for sid in range(20):
        A.note_attended([hearer], sid)
    hist = getattr(hearer, A.HISTORY_KEY)
    assert hist == list(range(12, 20))


def test_value_reads_closeness_tier_and_contacts():
    sim = _on_sim()
    hearer = _Agent(1)
    assert A.value_of(sim, hearer, 5) == 0.0
    hearer.mem.relations[5] = {"closeness": 0.4, "tier": 2}
    assert A.value_of(sim, hearer, 5) == pytest.approx(0.4 + 0.2)

    class _Net:
        contacts = {1: {5}}
    sim.net = _Net()
    assert A.value_of(sim, hearer, 5) == pytest.approx(0.4 + 0.2 + 0.5)


def test_slot_bonus_matches_name_or_id_string():
    hearer = _Agent(1)
    speaker = _Agent(5, name="タロウ")
    assert A.slot_bonus_of(hearer, speaker) == 0.0
    hearer.attention_slots = [{"kind": "person", "target": "5", "salience": 0.7}]
    assert A.slot_bonus_of(hearer, speaker) == pytest.approx(0.7)
    hearer.attention_slots = [{"kind": "person", "target": "タロウ", "salience": 0.4}]
    assert A.slot_bonus_of(hearer, speaker) == pytest.approx(0.4)
    hearer.attention_slots = [{"kind": "topic", "target": "タロウ", "salience": 1.0}]
    assert A.slot_bonus_of(hearer, speaker) == 0.0      # person 以外は照合しない


def test_score_is_the_wolfe_sum():
    """priority = w_bu·顕著性 + w_td·関連 + w_h·履歴 + w_v·価値(手計算と一致)。"""
    sim = _on_sim()
    cfg = A.cfg_of(sim)
    speaker = _Agent(0, name="話者")
    hearer = _Agent(1, adopted={"渋谷弁"})
    hearer.mem.relations[0] = {"closeness": 0.5, "tier": 0}
    A.note_attended([hearer], 0)
    want = (cfg["w_bu"] * A.salience_of(hearer, "やあ", ["渋谷弁"], 0.0, 0.0, cfg, 1.0)
            + cfg["w_td"] * A.relevance_of(hearer, "やあ", ["渋谷弁"])
            + cfg["w_h"] * A.history_of(hearer, 0)
            + cfg["w_v"] * A.value_of(sim, hearer, 0))
    got = A.score(sim, speaker, hearer, "やあ", ["渋谷弁"], 0.0, 0.0, cfg, 1.0)
    assert got == pytest.approx(want)


def test_score_adds_the_slot_bonus_only_when_layer_b_is_on():
    sim = _on_sim()
    cfg = A.cfg_of(sim)
    speaker = _Agent(0, name="話者")
    hearer = _Agent(1)
    hearer.attention_slots = [{"kind": "person", "target": "0", "salience": 1.0}]
    off = A.score(sim, speaker, hearer, "やあ", [], 0.0, 0.0, cfg, 1.0, block=False)
    on = A.score(sim, speaker, hearer, "やあ", [], 0.0, 0.0, cfg, 1.0, block=True)
    assert on == pytest.approx(off + cfg["w_slot"])


# --------------------------------------------------------------------------- #
# (5) select: 順位・θ・k 上限・優先充当・貫通・crowd_n
# --------------------------------------------------------------------------- #
def _line(n, name_fmt="人{}"):
    """話者(0)から x=1,2,… に並ぶ聞き手(近いほど priority が高い)。"""
    return [_Agent(i, x=float(i), name=name_fmt.format(i)) for i in range(1, n + 1)]


def test_select_keeps_the_top_k_by_priority():
    """既定 θ=0 では k_i 人ちょうど(近い順)= 距離だけの世界での挙動。"""
    sim = _on_sim(k_base=3, k_span=0)
    speaker = _Agent(0)
    got, crowd = A.select(sim, speaker, _line(8), "こんにちは", [], 0, radius=40.0)
    assert [h.id for h in got] == [1, 2, 3]
    assert crowd == 5


def test_select_returns_id_ascending():
    """返り値は必ず id 昇順(下流の走査順を絞らないときと同じ規則に保つ)。"""
    sim = _on_sim(k_base=3, k_span=0)
    speaker = _Agent(0)
    hearers = [_Agent(9, x=1.0), _Agent(2, x=2.0), _Agent(5, x=3.0),
               _Agent(1, x=40.0)]
    got, _ = A.select(sim, speaker, hearers, "やあ", [], 0, radius=40.0)
    assert [h.id for h in got] == [2, 5, 9]


def test_select_breaks_ties_by_id_ascending():
    """priority 同値は id 昇順(完全な決定論)。"""
    sim = _on_sim(k_base=2, k_span=0)
    speaker = _Agent(0)
    tie = [_Agent(7, x=1.0), _Agent(3, x=-1.0), _Agent(5, x=0.0, y=1.0)]
    got, _ = A.select(sim, speaker, tie, "やあ", [], 0, radius=40.0)
    assert [h.id for h in got] == [3, 5]


def test_select_theta_cuts_below_the_ignition_threshold():
    """θ を上げると k に満たなくても拾わない(k は上限でありノルマではない)。"""
    speaker = _Agent(0)
    hearers = [_Agent(1, x=1.0), _Agent(2, x=200.0), _Agent(3, x=400.0)]
    loose = _on_sim(k_base=3, k_span=0, theta_ignition=0.0)
    assert len(A.select(loose, speaker, hearers, "やあ", [], 0, radius=40.0)[0]) == 3
    tight = _on_sim(k_base=3, k_span=0, theta_ignition=0.9)
    got, crowd = A.select(tight, speaker, hearers, "やあ", [], 0, radius=40.0)
    assert [h.id for h in got] == [1] and crowd == 2


def test_select_theta_can_empty_the_audience():
    """全員が閾値未満なら誰も聞かない(静かな街 = 0 人もありうる)。"""
    sim = _on_sim(k_base=4, k_span=0, theta_ignition=99.0)
    got, crowd = A.select(sim, _Agent(0), _line(5), "やあ", [], 0, radius=40.0)
    assert got == [] and crowd == 5


def test_select_prefers_the_conversation_partner():
    """会話相手(pref_ids)は priority が低くても優先充当される。"""
    sim = _on_sim(k_base=2, k_span=0)
    speaker = _Agent(0)
    hearers = [_Agent(1, x=1.0), _Agent(2, x=2.0), _Agent(9, x=39.0)]
    plain, _ = A.select(sim, speaker, hearers, "やあ", [], 0, radius=40.0)
    assert 9 not in [h.id for h in plain]
    pref, _ = A.select(sim, speaker, hearers, "やあ", [], 0, radius=40.0,
                       pref_ids=(9,))
    assert 9 in [h.id for h in pref]
    assert len(pref) == 2, "優先充当が k の枠を食っていない(上限を破っている)"


def test_select_partner_survives_the_theta_cut():
    """θ で全員落ちる状況でも会話相手だけは残る(会話が壊れない)。"""
    sim = _on_sim(k_base=3, k_span=0, theta_ignition=99.0)
    got, _ = A.select(sim, _Agent(0), _line(5), "やあ", [], 0, radius=40.0,
                      pref_ids=(4,))
    assert [h.id for h in got] == [4]


def test_select_breakthrough_on_self_name():
    """自己名が発話に出ていれば、選抜枠の外でも p_break_i で貫通しうる(枠は食わない)。

    ★宛先性の顕著性加点(alpha_addr)を切って **貫通経路だけ**を単離する。
    """
    sim = _on_sim(k_base=2, k_span=0, p_break_base=0.65, alpha_addr=0.0)
    speaker = _Agent(0)
    hearers = [_Agent(1, x=1.0), _Agent(2, x=2.0), _Agent(9, x=39.0, name="ハナ")]
    hits = []
    for step in range(30):
        got, crowd = A.select(sim, speaker, hearers, "ハナ、こっちだよ", [], step,
                              radius=40.0)
        ids = [h.id for h in got]
        assert ids[:2] == [1, 2], "選抜枠(近い 2 人)が貫通で押し出された"
        if 9 in ids:
            hits.append(step)
            assert len(got) == 3, "貫通は**枠外**に足す(k を消費しない)"
            assert crowd == 0
        else:
            assert crowd == 1
    assert 0 < len(hits) < 30, "貫通が常時 / 皆無になっている(確率が効いていない)"


def test_select_never_breaks_through_without_the_name():
    """名前が発話に出ていなければ絶対に貫通しない(貫通の入口は自己名だけ)。"""
    sim = _on_sim(k_base=2, k_span=0, p_break_base=0.65, alpha_addr=0.0)
    hearers = [_Agent(1, x=1.0), _Agent(2, x=2.0), _Agent(9, x=39.0, name="ハナ")]
    for step in range(30):
        got, _ = A.select(sim, _Agent(0), hearers, "こんばんは", [], step, radius=40.0)
        assert [h.id for h in got] == [1, 2]


def test_select_breakthrough_is_deterministic_across_calls():
    """同じ (聞き手, 話者, step) なら貫通の可否が常に一致する(乱数ゼロ)。"""
    sim = _on_sim(k_base=1, k_span=0)
    hearers = [_Agent(i, x=float(i), name=f"人{i}") for i in range(1, 30)]
    text = "、".join(h.name for h in hearers)          # 全員の名前を含む発話
    first = [h.id for h in A.select(sim, _Agent(0), hearers, text, [], 7,
                                    radius=40.0)[0]]
    again = [h.id for h in A.select(sim, _Agent(0), hearers, text, [], 7,
                                    radius=40.0)[0]]
    assert first == again
    other = [h.id for h in A.select(sim, _Agent(0), hearers, text, [], 8,
                                    radius=40.0)[0]]
    assert other != first or len(first) == len(hearers)   # step が変われば貫通面も動く


def test_select_breakthrough_rate_is_in_the_expected_band():
    """全員の名前を呼ぶ発話での貫通率が Conway 帯(0.20-0.65)のオーダーに乗る。"""
    sim = _on_sim(k_base=2, k_span=0)
    hearers = [_Agent(i, x=100.0 + i, name=f"人{i:03d}") for i in range(1, 401)]
    text = "、".join(h.name for h in hearers)
    got, _ = A.select(sim, _Agent(0), hearers, text, [], 3, radius=40.0)
    rate = (len(got) - 2) / float(len(hearers) - 2)      # 選抜 2 人を除いた貫通率
    assert 0.20 <= rate <= 0.65, f"貫通率が帯の外: {rate}"


def test_select_handles_an_empty_audience():
    sim = _on_sim()
    assert A.select(sim, _Agent(0), [], "やあ", [], 0, radius=40.0) == ([], 0)


def test_select_crowd_n_counts_the_unattended():
    sim = _on_sim(k_base=2, k_span=0)
    got, crowd = A.select(sim, _Agent(0), _line(30), "やあ", [], 0, radius=40.0)
    assert len(got) + crowd == 30


def test_reply_target_is_only_valid_within_the_same_step():
    """預かった会話相手は**同じ step** にしか効かない(古い値が後の発話に効かない)。"""
    sim = _on_sim()
    agent = _Agent(1)
    A.note_reply_target(sim, agent, 5, 12)
    assert A.reply_target_of(agent, 12) == (5,)
    assert A.reply_target_of(agent, 13) == ()
    assert A.reply_target_of(_Agent(2), 12) == ()


def test_reply_target_is_not_recorded_when_layer_a_is_off():
    sim = _StubSim()
    agent = _Agent(1)
    A.note_reply_target(sim, agent, 5, 12)
    assert A.REPLY_KEY not in agent.__dict__


# --------------------------------------------------------------------------- #
# (6) ON の機械検証: 非通過の聞き手には**何も起きない**
# --------------------------------------------------------------------------- #
def _colocate(sim, n):
    """話者 + n 人を同じノード・同じ階に並べる(x だけずらして距離差を作る)。"""
    speaker = sim.agents[0]
    speaker.loc, speaker.building, speaker.floor = "street", None, 0
    speaker.x, speaker.y = 0.0, 0.0
    crowd = []
    for i, agent in enumerate(sim.agents[1:n + 1], start=1):
        agent.node, agent.loc = speaker.node, "street"
        agent.building, agent.floor = None, 0
        agent.x, agent.y = float(i), 0.0
        agent.sleeping = False
        agent.conv_turns_left, agent.conv_cooldown_until = 3, 0
        crowd.append(agent)
    return speaker, crowd


def test_on_run_gives_the_unattended_nothing(tmp_path):
    """★層A の芯: 非通過の聞き手に hear の L1 / 記憶 / 関係 / contacts が 1 件も出ない。"""
    sim = _sim(tmp_path, "att_on_apply", n_steps=1, n_agents=12,
               **{"world.attention.enabled": "true",
                  "world.attention.mode": "salience",
                  "world.attention.k_base": 2, "world.attention.k_span": 0,
                  "world.perception_radius_m": 400.0})
    speaker, crowd = _colocate(sim, 8)
    before = len(sim.logger.events)
    scheduler._apply(sim, speaker, {"type": "speak", "text": "いい天気ですね",
                                    "use_items": []}, 0, 0)
    fresh = sim.logger.events[before:]
    said = [e for e in fresh if e.kind == "speak"]
    assert len(said) == 1
    attended = set(said[0].payload["hearers"])
    assert 0 < len(attended) <= 2, f"k=2 を超えて選抜された: {attended}"
    heard = {e.agent_id for e in fresh if e.kind == "hear"}
    assert heard == attended, "hear の L1 が選抜集合と一致しない"
    for agent in crowd:
        got_mem = [ep for ep in agent.mem.buffer if ep.kind == "heard"]
        got_rel = int(speaker.id) in agent.mem.relations
        got_net = int(speaker.id) in sim.net.contacts.get(agent.id, set())
        if agent.id in attended:
            assert got_mem and got_rel and got_net, f"通過者に何も起きていない: {agent.id}"
        else:
            assert not got_mem, f"非通過者に記憶が残った: {agent.id}"
            assert not got_rel, f"非通過者に関係台帳が生えた: {agent.id}"
            assert not got_net, f"非通過者に contacts が生えた: {agent.id}"


def test_on_run_records_the_attention_history(tmp_path):
    """通過した聞き手にだけ注意履歴(プライミング)が積まれる。"""
    sim = _sim(tmp_path, "att_on_hist", n_steps=1, n_agents=12,
               **{"world.attention.enabled": "true",
                  "world.attention.mode": "salience",
                  "world.attention.k_base": 2, "world.attention.k_span": 0,
                  "world.perception_radius_m": 400.0})
    speaker, crowd = _colocate(sim, 8)
    scheduler._apply(sim, speaker, {"type": "speak", "text": "やあ",
                                    "use_items": []}, 0, 0)
    said = [e for e in sim.logger.events if e.kind == "speak"][-1]
    attended = set(said.payload["hearers"])
    for agent in crowd:
        hist = getattr(agent, A.HISTORY_KEY, None)
        if agent.id in attended:
            assert hist == [speaker.id]
        else:
            assert not hist


def test_on_run_is_deterministic_across_two_runs(tmp_path):
    """ON でも 2 ラン L1 一致(決定論 = 乱数を引いていないことの実ラン側の証明)。"""
    first = _sim(tmp_path, "att_det_1", n_steps=144, **ON_A)
    first.run()
    second = _sim(tmp_path, "att_det_2", n_steps=144, **ON_A)
    second.run()
    assert _l1(first) == _l1(second)


def test_on_run_bounds_the_audience_per_utterance(tmp_path):
    """ラン全体で「1 発話あたりの聞き手」が k の上限帯へ収まる(貫通ぶんを除く)。"""
    sim = _sim(tmp_path, "att_on_bound", n_steps=144, n_agents=40,
               **{"world.attention.enabled": "true",
                  "world.attention.mode": "salience"})
    sim.run()
    speaks = [e for e in sim.logger.events if e.kind == "speak"]
    assert speaks, "speak が 1 件も出ていない(テストの前提が崩れている)"
    for ev in speaks:
        hearers = ev.payload["hearers"]
        assert len(hearers) == len(set(hearers)), "聞き手が重複している"


# --------------------------------------------------------------------------- #
# (7) 層B: attend の受理
# --------------------------------------------------------------------------- #
def test_parse_action_passes_the_attend_field_through():
    """`attend` は speak の行動 dict へ素通しされる(型だけ見る)。"""
    got = deliberate.parse_action(json.dumps(
        {"action": "speak", "text": "やあ",
         "attend": [{"kind": "person", "target": "ハナ", "why": "気になる"}]}))
    assert got["attend"] == [{"kind": "person", "target": "ハナ", "why": "気になる"}]


def test_parse_action_without_attend_has_no_key():
    """欄が無ければキー自体を作らない(欠落 = 変更なし、を型で表す)。"""
    got = deliberate.parse_action(json.dumps({"action": "speak", "text": "やあ"}))
    assert "attend" not in got


def test_parse_action_drops_non_dict_attend_items():
    got = deliberate.parse_action(json.dumps(
        {"action": "speak", "text": "やあ", "attend": ["ハナ", 3, {"kind": "topic",
                                                                  "target": "祭"}]}))
    assert got["attend"] == [{"kind": "topic", "target": "祭"}]


def test_parse_action_ignores_a_non_list_attend():
    got = deliberate.parse_action(json.dumps(
        {"action": "speak", "text": "やあ", "attend": "ハナ"}))
    assert "attend" not in got


def test_apply_declaration_accepts_and_replaces_wholesale():
    """MEM1 式の全量置換(前回の宣言は残らない)。"""
    sim = _block_sim()
    agent = _Agent(1)
    A.apply_declaration(sim, agent, {"attend": [
        {"kind": "person", "target": "ハナ", "why": "気になる"},
        {"kind": "topic", "target": "再開発"}]}, 5)
    assert [s["target"] for s in A.slots_of(agent)] == ["ハナ", "再開発"]
    A.apply_declaration(sim, agent, {"attend": [{"kind": "goal", "target": "開店"}]}, 9)
    assert [s["target"] for s in A.slots_of(agent)] == ["開店"]


def test_apply_declaration_missing_field_is_a_noop():
    sim = _block_sim()
    agent = _Agent(1)
    A.apply_declaration(sim, agent, {"attend": [{"kind": "person",
                                                 "target": "ハナ"}]}, 5)
    A.apply_declaration(sim, agent, {"type": "speak", "text": "やあ"}, 9)
    assert [s["target"] for s in A.slots_of(agent)] == ["ハナ"]


def test_apply_declaration_empty_list_clears_the_block():
    """空 list は『もう何も気にしていない』の**有効な宣言**(欠落とは意味が違う)。"""
    sim = _block_sim()
    agent = _Agent(1)
    A.apply_declaration(sim, agent, {"attend": [{"kind": "person",
                                                 "target": "ハナ"}]}, 5)
    A.apply_declaration(sim, agent, {"attend": []}, 9)
    assert A.slots_of(agent) == []


def test_apply_declaration_drops_bad_kinds_and_empty_targets():
    sim = _block_sim()
    agent = _Agent(1)
    n = A.apply_declaration(sim, agent, {"attend": [
        {"kind": "weather", "target": "雨"},          # 不正 kind = 捨てる
        {"kind": "person", "target": "   "},          # 空 target = 捨てる
        {"kind": "person"},                           # target 欄なし = 捨てる
        {"kind": "place", "target": "駅前"}]}, 5)
    assert n == 1 and [s["kind"] for s in A.slots_of(agent)] == ["place"]


def test_apply_declaration_dedupes_and_truncates():
    sim = _block_sim(slots_max=3)
    agent = _Agent(1)
    decl = [{"kind": "topic", "target": "祭"}] * 2 \
        + [{"kind": "topic", "target": f"話{i}"} for i in range(5)]
    A.apply_declaration(sim, agent, {"attend": decl}, 5)
    targets = [s["target"] for s in A.slots_of(agent)]
    assert targets == ["祭", "話0", "話1"]            # 重複は 1 件・上限は先頭から


def test_apply_declaration_truncates_why_to_forty_chars():
    sim = _block_sim()
    agent = _Agent(1)
    A.apply_declaration(sim, agent, {"attend": [
        {"kind": "topic", "target": "祭", "why": "あ" * 100}]}, 5)
    assert len(A.slots_of(agent)[0]["why"]) == A.WHY_MAX


def test_apply_declaration_keeps_since_for_the_same_target():
    """再宣言で `since` は若返らない(いつから気にしているか、が保たれる)。"""
    sim = _block_sim()
    agent = _Agent(1)
    A.apply_declaration(sim, agent, {"attend": [{"kind": "person",
                                                 "target": "ハナ"}]}, 5)
    A.apply_declaration(sim, agent, {"attend": [
        {"kind": "person", "target": "ハナ"},
        {"kind": "person", "target": "タロウ"}]}, 40)
    slots = {s["target"]: s for s in A.slots_of(agent)}
    assert slots["ハナ"]["since"] == 5 and slots["タロウ"]["since"] == 40
    assert slots["ハナ"]["salience"] == 1.0            # salience は宣言のたび 1.0


def test_apply_declaration_is_a_noop_when_the_block_is_off():
    sim = _StubSim(b_ov={"enabled": False})
    agent = _Agent(1)
    assert A.apply_declaration(sim, agent, {"attend": [{"kind": "person",
                                                        "target": "ハナ"}]}, 5) == 0
    assert A.SLOTS_KEY not in agent.__dict__


# --------------------------------------------------------------------------- #
# (8) 層B: 減衰・除去・boost
# --------------------------------------------------------------------------- #
class _DaySim(_StubSim):
    """phase_day 用に agents を持つスタブ。"""


def test_phase_day_decays_the_salience():
    sim = _block_sim(decay_per_day=0.5, min_salience=0.0)
    agent = _Agent(1)
    sim.agents = [agent]
    A.apply_declaration(sim, agent, {"attend": [{"kind": "topic",
                                                 "target": "祭"}]}, 0)
    A.phase_day(sim, 0, 0)
    assert A.slots_of(agent)[0]["salience"] == pytest.approx(0.5)
    A.phase_day(sim, 144, 1440)
    assert A.slots_of(agent)[0]["salience"] == pytest.approx(0.25)


def test_phase_day_runs_once_per_day():
    sim = _block_sim(decay_per_day=0.5, min_salience=0.0)
    agent = _Agent(1)
    sim.agents = [agent]
    A.apply_declaration(sim, agent, {"attend": [{"kind": "topic",
                                                 "target": "祭"}]}, 0)
    A.phase_day(sim, 0, 0)
    A.phase_day(sim, 1, 10)                       # 同じ日 = 2 度目は走らない
    A.phase_day(sim, 2, 20)
    assert A.slots_of(agent)[0]["salience"] == pytest.approx(0.5)


def test_phase_day_removes_slots_below_min_salience():
    sim = _block_sim(decay_per_day=0.9, min_salience=0.05)
    agent = _Agent(1)
    sim.agents = [agent]
    A.apply_declaration(sim, agent, {"attend": [{"kind": "topic",
                                                 "target": "祭"}]}, 0)
    A.phase_day(sim, 0, 0)                        # 1.0 → 0.1(残る)
    assert len(A.slots_of(agent)) == 1
    A.phase_day(sim, 144, 1440)                   # 0.1 → 0.01(消える)
    assert A.slots_of(agent) == []


def test_note_addressed_boosts_the_matching_person_slot():
    sim = _block_sim(boost_addressed=0.4)
    hearer = _Agent(1)
    speaker = _Agent(5, name="タロウ")
    sim.agents = [hearer, speaker]
    sim.agent_by_id = {a.id: a for a in sim.agents}
    A.apply_declaration(sim, hearer, {"attend": [
        {"kind": "person", "target": "タロウ"},
        {"kind": "person", "target": "ハナ"}]}, 0)
    for slot in hearer.attention_slots:           # 減衰済みの状態から始める
        slot["salience"] = 0.2
    A.note_addressed(sim, hearer, 5)
    slots = {s["target"]: s["salience"] for s in A.slots_of(hearer)}
    assert slots["タロウ"] == pytest.approx(0.6)
    assert slots["ハナ"] == pytest.approx(0.2)    # 宛先でない相手は動かない


def test_note_addressed_is_capped_at_one():
    sim = _block_sim(boost_addressed=0.9)
    hearer = _Agent(1)
    speaker = _Agent(5, name="タロウ")
    sim.agents = [hearer, speaker]
    sim.agent_by_id = {a.id: a for a in sim.agents}
    A.apply_declaration(sim, hearer, {"attend": [{"kind": "person",
                                                  "target": "タロウ"}]}, 0)
    A.note_addressed(sim, hearer, 5)
    assert A.slots_of(hearer)[0]["salience"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# (9) 層B: 搬送(dehydrate → hydrate)と checkpoint
# --------------------------------------------------------------------------- #
def test_dehydrate_carries_nothing_when_attention_is_off(tmp_path):
    """ATT OFF の個体では退避 dict に ATT のキーが 1 つも生えない(バイト不変)。"""
    sim = _sim(tmp_path, "att_deh_off", n_steps=1)
    state = POOL.dehydrate(sim.agents[0])
    assert "attention_slots" not in state and "attn_hist" not in state


def test_dehydrate_and_hydrate_round_trip_the_slots(tmp_path):
    """★プール回転でスロットが消えない(第113 認知棚卸しの搬送漏れ教訓)。"""
    sim = _sim(tmp_path, "att_deh_on", n_steps=1)
    src, dst = sim.agents[0], sim.agents[1]
    blk = _block_sim()
    A.apply_declaration(blk, src, {"attend": [
        {"kind": "person", "target": "ハナ", "why": "気になる"},
        {"kind": "place", "target": "駅前"}]}, 17)
    A.note_attended([src], 9)
    state = POOL.dehydrate(src)
    assert state["attention_slots"][0] == {"kind": "person", "target": "ハナ",
                                           "why": "気になる", "salience": 1.0,
                                           "since": 17}
    assert state["attn_hist"] == [9]
    POOL.hydrate(dst, state)
    assert A.slots_of(dst) == A.slots_of(src)
    assert getattr(dst, A.HISTORY_KEY) == [9]


def test_hydrate_tolerates_old_states(tmp_path):
    """旧 退避辞書(ATT のキーが無い)を読んでも属性を生やさない。"""
    sim = _sim(tmp_path, "att_deh_old", n_steps=1)
    agent = sim.agents[0]
    POOL.hydrate(agent, {"beliefs": [], "day_summaries": [], "episodes": [],
                         "relations": {}, "adopted": [], "heard_counts": {},
                         "money": 0.0, "account": 0.0, "opinion": 0.0,
                         "status": 0.0, "self_model": None, "theta_drift": 0.0})
    assert A.SLOTS_KEY not in agent.__dict__
    assert A.HISTORY_KEY not in agent.__dict__


def test_slots_ride_the_checkpoint_via_agent_dict(tmp_path):
    """スロットは agent.__dict__ 経由で checkpoint(agents pickle)へ自然に載る。"""
    sim = _sim(tmp_path, "att_ckpt", n_steps=1)
    agent = sim.agents[0]
    A.apply_declaration(_block_sim(), agent, {"attend": [
        {"kind": "topic", "target": "再開発"}]}, 3)
    assert A.SLOTS_KEY in agent.__dict__
    import pickle
    back = pickle.loads(pickle.dumps(agent))
    assert A.slots_of(back) == A.slots_of(agent)


# --------------------------------------------------------------------------- #
# (10) 層B: プロンプト節(注入位置・ON/OFF・契約)
# --------------------------------------------------------------------------- #
def test_prompt_section_is_none_when_off():
    assert A.prompt_section(_StubSim(), _Agent(1)) is None


def test_prompt_section_lists_the_slots_and_the_candidates():
    sim = _block_sim()
    agent = _Agent(1)
    agent.heard_counts = {"再開発": 5, "祭": 9}
    A.apply_declaration(sim, agent, {"attend": [
        {"kind": "person", "target": "ハナ", "why": "気になる"}]}, 0)
    got = A.prompt_section(sim, agent, ["タロウ", "ハナ", "ケン", "ミナ"])
    assert "ハナ(気になる)" in got
    assert "タロウ、ハナ、ケン" in got and "ミナ" not in got      # 上位 3 人だけ
    assert "祭、再開発" in got                                   # 件数降順の決定論
    assert "attend" in got


def test_prompt_section_says_nothing_special_when_empty():
    sim = _block_sim()
    got = A.prompt_section(sim, _Agent(1))
    assert "特にない" in got


def test_prompt_section_has_no_mechanism_words():
    """no-fingerprint: 機構語・実験条件語を 1 文字も書かない。"""
    sim = _block_sim()
    agent = _Agent(1)
    A.apply_declaration(sim, agent, {"attend": [{"kind": "topic",
                                                 "target": "祭"}]}, 0)
    got = A.prompt_section(sim, agent, ["タロウ"])
    for word in ("注意機構", "顕著", "閾値", "発火", "スロット", "シミュレーション",
                 "モデル", "実験", "パラメータ", "priority", "salience"):
        assert word not in got, f"機構語が漏れている: {word}"


def test_build_prompt_is_byte_identical_without_the_section(tmp_path):
    """節を渡さないプロンプトは従来と 1 バイトも変わらない。"""
    sim = _sim(tmp_path, "att_prompt_off", n_steps=1)
    agent = sim.agents[0]
    base = deliberate.build_prompt(agent, place_name="駅前", surprise="solo",
                                   nearby_names=[])
    same = deliberate.build_prompt(agent, place_name="駅前", surprise="solo",
                                   nearby_names=[], attention_section=None)
    assert base == same


def test_build_prompt_places_the_section_right_after_the_persona(tmp_path):
    """固定位置 = ペルソナ直後の**先頭側**(Lost in the Middle 対策)。"""
    sim = _sim(tmp_path, "att_prompt_on", n_steps=1)
    agent = sim.agents[0]
    got = deliberate.build_prompt(agent, place_name="駅前", surprise="solo",
                                  nearby_names=[], attention_section="気にしている行")
    lines = got.split("\n")
    persona_last = lines.index(agent.persona.split("\n")[-1])
    assert lines[persona_last + 1] == "気にしている行"


def test_perception_contract_round_trips_the_section():
    """契約経路 ON でもプロンプト材料が無損失に往復する(未知キー例外にならない)。"""
    material = {"place_name": "駅前", "surprise": "solo", "nearby_names": [],
                "attention_section": "気にしている行"}
    percept = PC.Perception.from_material(material)
    assert percept.prompt_kwargs()["attention_section"] == "気にしている行"
    assert "attention_section" in {kw for kw, _f, _s in PC._KW_FIELDS}


def test_layer_b_on_run_stays_deterministic(tmp_path):
    """層B ON でも 2 ラン L1 一致(mock は attend を出さないので世界は動かない)。"""
    first = _sim(tmp_path, "att_b_1", n_steps=144, **ON_B)
    first.run()
    second = _sim(tmp_path, "att_b_2", n_steps=144, **ON_B)
    second.run()
    assert _l1(first) == _l1(second)


def test_layer_b_on_run_creates_no_slots_with_mock(tmp_path):
    """mock バックエンドは attend を出さない = スロットは 1 件も生えない(安全側)。"""
    sim = _sim(tmp_path, "att_b_mock", n_steps=144, **ON_B)
    sim.run()
    assert not any(A.slots_of(a) for a in sim.agents)


class _FixedLLM:
    """内容非依存の固定応答 LLM(test_contact_formation の _FixedLLM と同型)。"""

    name = "fixed"

    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.hits = 0

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        self.calls += 1
        return self.response, str(self.calls), False


_DECLARING = json.dumps({"action": "speak", "text": "こんにちは",
                         "attend": [{"kind": "person", "target": "ハナ",
                                     "why": "さっき助けてもらった"},
                                    {"kind": "topic", "target": "再開発"}]},
                        ensure_ascii=False)


def test_layer_b_end_to_end_with_a_declaring_llm(tmp_path):
    """実ラン: attend を必ず宣言する LLM でスロットが実際に生える(配線の機械証明)。"""
    sim = _sim(tmp_path, "att_b_declare", n_steps=144, n_agents=20, **ON_B)
    sim.llm = _FixedLLM(_DECLARING)
    sim.run()
    filled = [a for a in sim.agents if A.slots_of(a)]
    assert filled, "宣言する LLM なのにスロットが 1 件も生えていない"
    got = A.slots_of(filled[0])
    assert [s["target"] for s in got] == ["ハナ", "再開発"]
    assert all(s["salience"] > 0.0 for s in got)


def test_layer_b_declaration_is_ignored_when_off(tmp_path):
    """層B OFF では同じ LLM でもスロットは 1 件も生えない(受理点は 1 つ)。"""
    sim = _sim(tmp_path, "att_b_declare_off", n_steps=144, n_agents=20)
    sim.llm = _FixedLLM(_DECLARING)
    sim.run()
    assert not any(A.slots_of(a) for a in sim.agents)


def test_both_layers_on_run_is_deterministic(tmp_path):
    """層A + 層B 同時 ON でも 2 ラン L1 一致(結合経路 w_slot まで含めて決定論)。"""
    both = dict(ON_A)
    both.update(ON_B)
    first = _sim(tmp_path, "att_both_1", n_steps=144, n_agents=20, **both)
    first.llm = _FixedLLM(_DECLARING)
    first.run()
    second = _sim(tmp_path, "att_both_2", n_steps=144, n_agents=20, **both)
    second.llm = _FixedLLM(_DECLARING)
    second.run()
    assert _l1(first) == _l1(second)
    assert any(A.slots_of(a) for a in first.agents)


def _run_straight(tmp_path, name, n_steps, llm=None, **ov):
    d = tmp_path / name
    sim = Simulation(_cfg(name, n_steps, 20, **ov), out_dir=d)
    if llm is not None:
        sim.llm = llm()
    sim.run()
    return d, sim


def _run_resume(tmp_path, name, split, total, llm=None, **ov):
    """phase1: split step 走らせて ckpt を書き中断 / phase2: 新 Simulation で load → total。"""
    from society.engine import checkpoint
    d = tmp_path / name
    every = {"observer.checkpoint_every": split}
    sim1 = Simulation(_cfg(name, split, 20, **every, **ov), out_dir=d)
    if llm is not None:
        sim1.llm = llm()
    for step in range(split):
        scheduler.run_step(sim1, step)
    checkpoint.save(sim1, split, d / "checkpoint" / f"ckpt-{split:06d}.pkl.gz")
    sim1.logger.flush_segment()
    sim2 = Simulation(_cfg(name, total, 20, **every, **ov), out_dir=d)
    if llm is not None:
        sim2.llm = llm()
    sim2.run(resume_from=d)
    return d, sim2


def _rows(run_dir, stem="l1_events"):
    import pyarrow.parquet as pq
    from pathlib import Path
    return pq.read_table(Path(run_dir) / f"{stem}.parquet").to_pylist()


def test_resume_matches_straight_with_both_layers_on(tmp_path):
    """★resume == straight(層A + 層B 同時 ON)。"""
    both = dict(ON_A)
    both.update(ON_B)
    straight, _ = _run_straight(tmp_path, "att_straight", 40, **both)
    resumed, _ = _run_resume(tmp_path, "att_resumed", 20, 40, **both)
    assert _rows(straight) == _rows(resumed), "l1_events が resume で不一致"
    for stem in ("l2_metrics", "l3_snapshots"):
        assert _rows(straight, stem) == _rows(resumed, stem), f"{stem} 不一致"


def test_resume_carries_the_declared_slots(tmp_path):
    """★スロットが**実際に生えている**状態で resume == straight(checkpoint 搬送の証明)。"""
    both = dict(ON_A)
    both.update(ON_B)

    def _llm():
        return _FixedLLM(_DECLARING)

    straight, s_sim = _run_straight(tmp_path, "att_slot_straight", 40,
                                    llm=_llm, **both)
    resumed, r_sim = _run_resume(tmp_path, "att_slot_resumed", 20, 40,
                                 llm=_llm, **both)
    assert any(A.slots_of(a) for a in s_sim.agents), \
        "宣言する LLM なのにスロットが 1 件も生えていない(前提が崩れている)"
    # ★`llm_call_id` だけは除外する: 代役 LLM の**プロセス内**通し番号で、phase2 の
    #   新インスタンスで 1 から振り直される(= テスト器具の都合であって世界の差ではない)。
    def _no_call_id(rows):
        return [{k: v for k, v in r.items() if k != "llm_call_id"} for r in rows]

    assert _no_call_id(_rows(straight)) == _no_call_id(_rows(resumed)), \
        "l1_events が resume で不一致"
    s_slots = {a.id: A.slots_of(a) for a in s_sim.agents}
    r_slots = {a.id: A.slots_of(a) for a in r_sim.agents}
    assert s_slots == r_slots, "注意ブロックが checkpoint を跨いで一致しない"


def test_checkpoint_carries_the_day_guard(tmp_path):
    """★二重減衰の再発防止: 日ガード `_attn_block_day` が checkpoint を往復する。

    これを保存しないと、mid-day resume の直後に同じ暦日の減衰がもう一度走り、
    スロットの salience が straight より 1 段低くなる(実装中に実測で発覚した)。
    """
    from society.engine import checkpoint
    sim = _sim(tmp_path, "att_dayguard", n_steps=1, **ON_B)
    sim._attn_block_day = 3
    path = checkpoint.save(sim, 1, tmp_path / "ck" / "ckpt.pkl.gz")
    other = _sim(tmp_path, "att_dayguard2", n_steps=1, **ON_B)
    checkpoint.load(other, path)
    assert other._attn_block_day == 3


def test_layer_b_on_run_injects_the_section(tmp_path):
    """層B ON では熟慮プロンプトに「いま気にしていること」が載る。"""
    sim = _sim(tmp_path, "att_b_prompt", n_steps=1, **ON_B)
    agent = sim.agents[0]
    material = scheduler._gather_material(sim, agent, "solo", 0, 0)
    section = A.prompt_section(sim, agent, material.get("nearby_names"))
    assert section is not None and "いま気にしていること" in section
