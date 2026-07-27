"""関係の質の内生化 第65バッチ フェーズ4(relations.endogenous_quality)のテスト。

方針(第62/64 の鉄則を継承):
- OFF(既定): 純粋既定と L1 完全一致・sim._quality_state 不在・L2 に quality_* 列なし・
  note_contact の magnitude 既定 1.0 が従来の closeness と厳密一致(ゴールデンは test_scenario)。
- ON: 会話由来 magnitude が [mag_min, mag_max] の範囲で closeness の増減**量**にのみ載る
  (長文↑・往復↑・明示キュー↑・hedge 共起は中立 1.0・clamp)。
- **片方向 hook**: magnitude は決定に一切流れない。tier 閾値を凍結(遷移が起きない設定)すれば
  ON/OFF で L1 が完全一致し LLM 呼数も一致する=会話の発生・相手選択・イベント列は不変で、
  変わるのは台帳の closeness の大きさだけ、を直接固定する。
  (通常設定では closeness の蓄積速度が変わり tier 遷移の**時期**が動く→プロンプト/発話が
   変わる=これは treatment そのもの。だから不変性テストは閾値を凍結して測る=正直な設計。)
- L2 1列(quality_magnitude_mean)の検算・同 seed 2 ラン一致・resume==straight(日境界跨ぎ)。
検証は mock のみ(実LLM 禁止)。
"""
from __future__ import annotations

import json

import pyarrow.parquet as pq

from society import relations
from society import relations_endo as _endo
from society.config import load_config
from society.engine import checkpoint, scheduler
from society.engine.simulation import Simulation

_REL_ON = {"relations.enabled": "true"}
_QUAL_ON = {**_REL_ON, "relations.endogenous_quality.enabled": "true"}
# tier 閾値の凍結(到達不能な巨大値)= closeness がどれだけ動いても tier は 0 のまま
# → relation_tier/relation_break が出ず、プロンプトの間柄行も従来の「N回話した仲」のまま。
_TIER_FROZEN = {"relations.tier_acquaintance": "1e9",
                "relations.tier_friend": "1e9",
                "relations.tier_close": "1e9"}


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


def _closeness_map(sim):
    """全 agent の関係台帳の closeness を (self, other)→値 で吸い出す(比較用)。"""
    out = {}
    for a in sim.agents:
        for oid, rel in a.mem.relations.items():
            if "closeness" in rel:
                out[(a.id, oid)] = round(float(rel["closeness"]), 6)
    return out


def _fresh_pair(sim):
    a, b = sim.agents[0], sim.agents[1]
    a.mem.relations.pop(b.id, None)
    b.mem.relations.pop(a.id, None)
    return a, b


# --------------------------------------------------------------------- OFF 既定
def test_off_matches_pure_default(tmp_path):
    """明示 OFF と純粋既定が L1 完全一致(144 step)。_quality_state 不在・L2 に quality_* 列なし。"""
    pure = _sim(tmp_path, "pure", steps=144)
    pure.run()
    off = _sim(tmp_path, "expl_off", steps=144,
               **{"relations.endogenous_quality.enabled": "false"})
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(quality seam が no-op でない)"
    assert getattr(off, "_quality_state", None) is None, \
        "OFF なのに _quality_state が生えている"
    cols = pq.read_table(tmp_path / "expl_off" / "l2_metrics.parquet").column_names
    assert not [c for c in cols if c.startswith("quality_")], \
        "OFF なのに L2 に quality_* 列"


def test_off_with_relations_on_unchanged(tmp_path):
    """relations ON のまま quality を明示 OFF にしても、未指定ランと L1・closeness 台帳が
    完全一致(seam が relations 経路を一切汚さない)。"""
    base = _sim(tmp_path, "q_base", steps=144, **_REL_ON)
    base.run()
    off = _sim(tmp_path, "q_off", steps=144,
               **{**_REL_ON, "relations.endogenous_quality.enabled": "false"})
    off.run()
    assert _l1(base) == _l1(off), "quality 明示 OFF が未指定と L1 不一致"
    assert _closeness_map(base) == _closeness_map(off), \
        "quality 明示 OFF で closeness が変わった"


def test_note_contact_default_magnitude_is_identity(tmp_path):
    """note_contact の magnitude 既定 1.0 が従来と厳密一致し、明示値は増減**量**にのみ効く
    (符号=valence の解釈は不変)。"""
    sim = _sim(tmp_path, "ident", n=10)
    cfg = relations.build_cfg({"enabled": True})
    a, b = _fresh_pair(sim)
    for i in range(3):                                    # magnitude 省略=従来経路
        relations.note_contact(a, b.id, b.name, "", 1.0, cfg,
                               step=i, sim_min=i * 10, logger=sim.logger)
    legacy = a.mem.relations[b.id]["closeness"]
    a.mem.relations.pop(b.id, None)
    for i in range(3):                                    # magnitude=1.0 を明示
        relations.note_contact(a, b.id, b.name, "", 1.0, cfg,
                               step=i, sim_min=i * 10, logger=sim.logger,
                               magnitude=1.0)
    assert a.mem.relations[b.id]["closeness"] == legacy == 3.0, \
        "magnitude 既定 1.0 が従来の closeness と厳密一致しない"
    a.mem.relations.pop(b.id, None)
    relations.note_contact(a, b.id, b.name, "", 1.0, cfg, step=9, sim_min=90,
                           logger=sim.logger, magnitude=2.0)
    assert a.mem.relations[b.id]["closeness"] == 2.0, "ポジ交流に magnitude が効いていない"
    relations.note_contact(a, b.id, b.name, "", -1.0, cfg, step=10, sim_min=100,
                           logger=sim.logger, magnitude=0.5)
    assert a.mem.relations[b.id]["closeness"] == 1.0, \
        "ネガ交流(−neg_weight×magnitude=−1.0)に magnitude が効いていない"


# --------------------------------------------------------------------- 抽出(純関数)
def test_magnitude_of_thickness_and_cue():
    """厚い会話(長文・往復多)で magnitude↑・明示キュー共起で↑・材料ゼロは 1.0。"""
    cfg = _endo.build_cfg({"enabled": True})
    q = _endo.build_quality_cfg({"enabled": True})
    thin = _endo.magnitude_of("そう", 0, cfg, q)
    assert thin > 1.0, "短文でも長さ由来の加点は正(0 文字でない)"
    long_text = "あ" * 60
    assert _endo.magnitude_of(long_text, 0, cfg, q) == 1.5, \
        "発話長 60 文字で len_gain=0.5 満点にならない"
    assert _endo.magnitude_of("あ" * 200, 0, cfg, q) == 1.5, "長さの加点に上限が無い"
    # 往復数(相手別バッファの発話数): 1 発話=加点なし → 4 発話で turn_max=0.45
    assert _endo.magnitude_of(long_text, 1, cfg, q) == 1.5
    assert _endo.magnitude_of(long_text, 2, cfg, q) == 1.65
    assert _endo.magnitude_of(long_text, 4, cfg, q) == 1.95
    assert _endo.magnitude_of(long_text, 9, cfg, q) == 1.95, "往復の加点に上限が無い"
    # 明示キュー(positive_cues の共起)= 感情/意向の表出 → cue_gain
    base = _endo.magnitude_of("今日は天気がいいね", 0, cfg, q)
    cued = _endo.magnitude_of("今日は天気がいいね遊ぼう", 0, cfg, q)
    assert cued - base > 0.4 - 1e-9, f"明示キューの加点が効いていない: {base} {cued}"
    # 材料ゼロ(空文字・往復なし)は 1.0=従来と同値
    assert _endo.magnitude_of("", 0, cfg, q) == 1.0


def test_magnitude_of_hedge_is_neutral():
    """hedge_markers(婉曲/逆接)の共起は他の材料を見ずに中立 1.0(誤読回避=保守側)。"""
    cfg = _endo.build_cfg({"enabled": True})
    q = _endo.build_quality_cfg({"enabled": True})
    hedged = "あ" * 60 + "一緒に行きたいのは山々ですが"
    assert _endo.magnitude_of(hedged, 4, cfg, q) == 1.0, \
        "hedge 共起なのに厚み/キューの加点が載っている"
    assert _endo.magnitude_of("また今度", 4, cfg, q) == 1.0
    # hedge 語彙は accept ブロックと共用(単一の源)= conf で消せば中立化も消える
    cfg2 = _endo.build_cfg({"enabled": True, "hedge_markers": []})
    assert _endo.magnitude_of(hedged, 4, cfg2, q) > 1.0


def test_magnitude_of_clamped_to_conf_range():
    """[mag_min, mag_max] の clamp(既定の最大加点合計 2.35 は 2.0 で頭打ち)。"""
    cfg = _endo.build_cfg({"enabled": True})
    q = _endo.build_quality_cfg({"enabled": True})
    top = _endo.magnitude_of("あ" * 60 + "遊ぼう", 4, cfg, q)
    assert top == 2.0, f"上限 clamp が効いていない: {top}"
    q_low = _endo.build_quality_cfg({"enabled": True, "mag_max": 1.2})
    assert _endo.magnitude_of("あ" * 60 + "遊ぼう", 4, cfg, q_low) == 1.2
    # 下限: 加点を負に振っても mag_min を下回らない
    q_neg = _endo.build_quality_cfg({"enabled": True, "len_gain": -5.0,
                                     "mag_min": 0.5})
    assert _endo.magnitude_of("あ" * 60, 0, cfg, q_neg) == 0.5


def test_contact_magnitude_reads_dialog_buffer(tmp_path):
    """contact_magnitude: 相手別バッファ(_dialog_hist)の発話数を往復材料に使い、当日タリーへ
    加算する。OFF なら常に 1.0 でタリーも作らない。"""
    off = _sim(tmp_path, "cm_off", n=10, **_REL_ON)
    a, b = off.agents[0], off.agents[1]
    assert _endo.contact_magnitude(off, a, b.id, "あ" * 60, 0) == 1.0
    assert getattr(off, "_quality_state", None) is None

    sim = _sim(tmp_path, "cm_on", n=10, **_QUAL_ON)
    a, b = sim.agents[0], sim.agents[1]
    m0 = _endo.contact_magnitude(sim, a, b.id, "あ" * 60, 0)
    a._dialog_hist = {b.id: [(a.name, "x"), (b.name, "y"), (a.name, "z")]}
    m1 = _endo.contact_magnitude(sim, a, b.id, "あ" * 60, 0)
    assert m1 > m0 == 1.5, f"往復の蓄積が magnitude に効いていない: {m0} {m1}"
    st = sim._quality_state
    assert st["day"] == 0 and st["n"] == 2 and abs(st["sum"] - (m0 + m1)) < 1e-12
    # 中立(hedge)件数も数える=品質の正直な指標
    assert _endo.contact_magnitude(sim, a, b.id, "また今度", 0) == 1.0
    assert sim._quality_state["neutral"] == 1
    # 日が変われば作り直す(日境界フェーズが走らない経路でも当日集計になる)
    _endo.contact_magnitude(sim, a, b.id, "あ", 1440)
    assert sim._quality_state["day"] == 1 and sim._quality_state["n"] == 1


def test_quality_scalars_hand_computation(tmp_path):
    """quality_scalars() の手計算一致(タリー直接注入)。会話 0 件の日は 0.0。
    relations OFF / quality OFF は None=列なし。"""
    sim = _sim(tmp_path, "sc_q", **_QUAL_ON)
    sim._quality_state = {"day": 0, "n": 4, "sum": 6.0, "neutral": 1}
    assert _endo.quality_scalars(sim) == {"quality_magnitude_mean": 1.5}
    sim._quality_state = {"day": 0, "n": 0, "sum": 0.0, "neutral": 0}
    assert _endo.quality_scalars(sim) == {"quality_magnitude_mean": 0.0}
    rel_off = _sim(tmp_path, "sc_reloff",
                   **{"relations.endogenous_quality.enabled": "true"})
    rel_off._quality_state = {"day": 0, "n": 4, "sum": 6.0, "neutral": 0}
    assert _endo.quality_scalars(rel_off) is None, "relations OFF で列が出ている"
    assert _endo.quality_scalars(_sim(tmp_path, "sc_off", **_REL_ON)) is None


# --------------------------------------------------------------------- 片方向 hook
def test_one_way_hook_events_and_calls_unchanged(tmp_path):
    """片方向性の直接検証: tier 閾値を凍結(遷移が起きない=プロンプト・イベントに closeness が
    現れない)した上で quality ON/OFF を回すと L1 が**完全一致**し LLM 呼数も一致する。
    変わるのは台帳 closeness の大きさだけ=magnitude は決定に一切流れていない。"""
    ov = {**_REL_ON, **_TIER_FROZEN, "prompts.dialog_history": "true"}
    off = _sim(tmp_path, "ow_off", steps=144, **ov)
    off.run()
    on = _sim(tmp_path, "ow_on", steps=144,
              **{**ov, "relations.endogenous_quality.enabled": "true"})
    on.run()
    assert _l1(off) == _l1(on), "quality ON で L1 が変わった(片方向 hook 違反)"
    assert off.llm.calls == on.llm.calls > 0, \
        f"LLM 呼数が ON/OFF で違う: {off.llm.calls} vs {on.llm.calls}"
    for kind in ("speak", "hear", "dm"):
        assert len(_kind(off, kind)) == len(_kind(on, kind))
    # tier 遷移は凍結の通り 0 件(前提の確認)
    assert not _kind(on, "relation_tier") and not _kind(on, "relation_break")
    # ただし closeness の**大きさ**は動いている(hook が実際に発火した証拠)
    c_off, c_on = _closeness_map(off), _closeness_map(on)
    assert set(c_off) == set(c_on), "関係の**集合**が変わった(量以外に効いている)"
    assert c_off != c_on, "quality ON でも closeness が全く変わっていない(hook 未発火)"
    for key, v in c_on.items():
        assert v != 0.0 or c_off[key] == 0.0


def test_magnitude_in_conf_range_during_run(tmp_path):
    """実ラン中に算出された magnitude が全件 [mag_min, mag_max] に収まる(clamp の実地確認)。"""
    seen = []
    orig = _endo.contact_magnitude

    def _spy(sim, speaker, other_id, text, sim_min):
        m = orig(sim, speaker, other_id, text, sim_min)
        seen.append(m)
        return m

    scheduler.relations_endo_mod.contact_magnitude = _spy
    try:
        sim = _sim(tmp_path, "range", steps=144, **_QUAL_ON)
        sim.run()
    finally:
        scheduler.relations_endo_mod.contact_magnitude = orig
    assert seen, "会話が 1 件も起きていない(前提が崩れた)"
    q = _endo.quality_cfg_of(sim)
    assert min(seen) >= q["mag_min"] and max(seen) <= q["mag_max"]


class _FixedLLM:
    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.hits = 0

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        self.calls += 1
        return self.response, str(self.calls), False


def test_llm_call_count_k_invariant(tmp_path):
    """quality ON のまま compute_matched 下で k=free と k=off の generate 呼数が完全一致(R1)。"""
    def _run(name, writeback):
        sim = _sim(tmp_path, name, steps=100,
                   **{**_QUAL_ON, "controls.mode": "compute_matched",
                      "k.writeback": writeback})
        sim.llm = _FixedLLM(json.dumps({"action": "speak", "text": "x"},
                                       ensure_ascii=False))
        sim.run()
        return sim
    free = _run("qk_free", "free")
    off = _run("qk_off", "off")
    assert free.llm.calls == off.llm.calls > 0, \
        f"quality の呼数が k に依存(R1 違反): free={free.llm.calls} off={off.llm.calls}"


# --------------------------------------------------------------------- L2・決定論・resume
def test_l2_column_matches_tally(tmp_path):
    """L2 quality_magnitude_mean が当日タリー(_quality_state)からの手計算と一致する。

    200 step(=日境界 step102 を越えて day1 の日中まで)回す: 144 step だと最終行が day1 の
    未明(全員就寝中=会話ゼロ)に当たり、当日タリーが空で検算にならない。"""
    sim = _sim(tmp_path, "l2q", steps=200, **_QUAL_ON)
    sim.run()
    rows = pq.read_table(tmp_path / "l2q" / "l2_metrics.parquet").to_pylist()
    last = rows[-1]
    assert "quality_magnitude_mean" in last, "ON なのに L2 に quality 列が無い"
    st = sim._quality_state
    assert st["n"] > 0, "会話由来のタリーが積まれていない"
    assert abs(last["quality_magnitude_mean"]
               - round(st["sum"] / st["n"], 6)) < 1e-9
    q = _endo.quality_cfg_of(sim)
    assert q["mag_min"] <= last["quality_magnitude_mean"] <= q["mag_max"]


def test_on_deterministic(tmp_path):
    """quality ON(材料源つき)同 seed 2 ランで L1 完全一致(決定論=乱数を引かない)。"""
    ov = {**_QUAL_ON, "prompts.dialog_history": "true"}
    a = _sim(tmp_path, "qdet_a", steps=144, **ov)
    a.run()
    b = _sim(tmp_path, "qdet_b", steps=144, **ov)
    b.run()
    assert _l1(a) == _l1(b), "quality ON の決定論が崩れている"
    assert _closeness_map(a) == _closeness_map(b), "closeness が 2 ランで一致しない"


def test_resume_matches_straight_across_day_boundary(tmp_path):
    """quality(+relations+dialog_history)ON の resume==straight(日境界 step102 を**再開後に**
    跨ぐ mid-day split=60)。当日タリーが積まれた状態で checkpoint を切る=_quality_state の
    中央管理(未保存なら当日平均が resume で食い違う)を直接固定する。"""
    ov = {**_QUAL_ON, "prompts.dialog_history": "true"}

    def _cfg(name, n_steps, **extra):
        dot = ["run.seed=42", "run.n_agents=20", f"run.n_steps={n_steps}",
               f"run.name={name}", "model.backend=mock"]
        dot += [f"{k}={v}" for k, v in {**ov, **extra}.items()]
        return load_config(dot)

    straight_dir = tmp_path / "q_straight"
    Simulation(_cfg("q_straight", 120), out_dir=straight_dir).run()
    d = tmp_path / "q_resumed"
    split, total = 60, 120
    sim1 = Simulation(_cfg("q_resumed", split,
                           **{"observer.checkpoint_every": split}), out_dir=d)
    for step in range(split):
        scheduler.run_step(sim1, step)
    checkpoint.save(sim1, split, d / "checkpoint" / f"ckpt-{split:06d}.pkl.gz")
    sim1.logger.flush_segment()
    sim2 = Simulation(_cfg("q_resumed", total,
                           **{"observer.checkpoint_every": split}), out_dir=d)
    sim2.run(resume_from=d)
    for stem in ("l1_events", "l2_metrics", "l3_snapshots"):
        sa = pq.read_table(straight_dir / f"{stem}.parquet").to_pylist()
        sb = pq.read_table(d / f"{stem}.parquet").to_pylist()
        assert sa == sb, f"{stem} 不一致(quality resume)"
    # 直接検証: _quality_state が round-trip で復元される(空回り防止)
    assert getattr(sim1, "_quality_state", None) is not None
    assert sim1._quality_state["n"] > 0, "タリーが空(テスト前提が崩れた)"
    assert getattr(sim2, "_rel_day", -1) >= 1, "再開後に日境界を跨いでいない(前提が崩れた)"
    sim3 = Simulation(_cfg("q_inspect", split,
                           **{"observer.checkpoint_every": split}),
                      out_dir=tmp_path / "q_inspect")
    checkpoint.load(sim3, d / "checkpoint" / f"ckpt-{split:06d}.pkl.gz")
    assert sim3._quality_state == sim1._quality_state
