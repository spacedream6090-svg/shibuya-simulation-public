"""場所の意味づけ最小版 D1 第69バッチ(labeling.place_binding)のテスト。

方針(第62-67バッチの鉄則を継承):
- OFF(既定): 純粋既定と L1 完全一致・**15体144step が golden_baseline_l1.json とバイト一致**・
  draw 数(全 stream)同一・`labels.place_bind is None`・place_label_bind 0 件・
  L2 に place_bound_labels 列なし・熟慮プロンプトが 1 バイトも変わらない。
- ON: 熟慮 coin_label の受理で発生ノードへ束縛(語ごとに最初の 1 回だけ・上書きなし)→
  同ノードの他者の熟慮プロンプトに中立 1 行が入る(プロンプト実体で検証)。
  min_adopters ゲート / 複数語の決定論選択(束縛 step 降順 → 語の辞書順)/ prompt_line=false で
  束縛だけ行い知覚には出さない / tools 経路(node 非指定)は束縛しない。
- 乱数ゼロ: ON でも新 stream を切らず既存 draw も増やさない(OFF/ON の draw 数一致)。
- checkpoint: 束縛台帳は LabelSystem 同梱で resume==straight(round-trip で直接固定)。
- R1: ON のまま compute_matched 下で k=free と k=off の LLM 呼数が一致。
検証は mock のみ(実 LLM 禁止)。
"""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from society.config import load_config
from society.engine import checkpoint, scheduler
from society.engine.simulation import Simulation
from society.observer.schema import EVENT_KINDS

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN = REPO_ROOT / "tests" / "data" / "golden_baseline_l1.json"

# test_scenario.py:45 と同じ「意図的な既定挙動追加」の中立化(ゴールデン比較用)
_GOLDEN_NEUTRAL = {"economy.wages.自営": 0, "planning.enabled": "false",
                   "transit_ride.taxi.enabled": "false",
                   "transit_ride.bus.enabled": "false", "rules.enabled": "false"}

_ON = {"labeling.place_binding.enabled": "true"}
_LINE_MARK = "呼ばれることがある"


def _sim(tmp_path, name, n=15, steps=1, **ov):
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
    """全 stream の draw を数えるプロキシ(test_worldmod と同型)。"""

    def __init__(self, inner):
        self._inner = inner
        self.per_stream: dict = {}

    def stream(self, *key):
        g = self._inner.stream(*key)
        return _CountingGen(g, self, str(key[0]) if key else "")

    def key_name(self, *key):
        return self._inner.key_name(*key)

    @property
    def master_seed(self):
        return self._inner.master_seed


class _CountingGen:
    def __init__(self, g, hub, name):
        self._g, self._hub, self._name = g, hub, name

    def __getattr__(self, attr):
        target = getattr(self._g, attr)
        if not callable(target):
            return target
        hub, name = self._hub, self._name

        def wrapped(*a, **k):
            hub.per_stream[name] = hub.per_stream.get(name, 0) + 1
            return target(*a, **k)
        return wrapped


class _RecordingLLM:
    """発行されたプロンプト実体を記録する薄いプロキシ(注入の直接検証用)。"""

    def __init__(self, inner):
        self._inner = inner
        self.prompts: list[str] = []

    def generate(self, prompt, **kw):
        self.prompts.append(prompt)
        return self._inner.generate(prompt, **kw)

    def __getattr__(self, attr):
        return getattr(self._inner, attr)


class _FixedLLM:
    """内容非依存の固定応答(呼数の k 不変検証用。test_endogenous_invite と同型)。"""

    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.hits = 0

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        self.calls += 1
        return self.response, str(self.calls), False


def _colocate(src, dst):
    """dst を src と同じ知覚文脈・同じノードへ移す(合成シナリオ用)。"""
    dst.node = src.node
    dst.x, dst.y = src.x, src.y
    dst.building, dst.floor, dst.loc = src.building, src.floor, src.loc
    dst.sleeping = False


def _speak_prompt(sim, agent, trigger="solo"):
    """_llm_speak を 1 回だけ直接呼び、組まれたプロンプト実体を返す。"""
    rec = _RecordingLLM(sim.llm)
    sim.llm = rec
    scheduler._llm_speak(sim, agent, trigger, 0, 0)
    assert rec.prompts, "プロンプトが 1 本も組まれていない(テスト前提が崩れた)"
    return rec.prompts[-1]


# ============================================================ 既定 OFF の不変条件
def test_off_matches_pure_default(tmp_path):
    """明示 OFF と純粋既定が L1 完全一致(15体144step)。状態・イベント・L2 列も生えない。"""
    pure = _sim(tmp_path, "pb_pure", steps=144)
    pure.run()
    off = _sim(tmp_path, "pb_off", steps=144,
               **{"labeling.place_binding.enabled": "false"})
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(place_binding seam が no-op でない)"
    assert off.labels.place_bind is None, "OFF なのに束縛台帳が生えている"
    assert not _kind(off, "place_label_bind"), "OFF なのに place_label_bind が出ている"
    cols = pq.read_table(tmp_path / "pb_off" / "l2_metrics.parquet").column_names
    assert "place_bound_labels" not in cols, "OFF なのに L2 に place_bound_labels 列がある"


def test_off_matches_golden(tmp_path):
    """明示 OFF が変更前ゴールデンと一字一句一致(ゴールデンは再生成しない=掟2)。"""
    golden = json.load(open(GOLDEN, encoding="utf-8"))
    sim = _sim(tmp_path, "pb_golden", steps=144,
               **{**_GOLDEN_NEUTRAL, "labeling.place_binding.enabled": "false"})
    sim.run()
    assert _l1(sim) == golden, "place_binding の seam が no-op でない(ゴールデン不一致)"


def test_off_draw_counts_identical(tmp_path):
    """OFF は純粋既定と draw 数(stream 別)が完全一致=CRN 共分散を壊さない。"""
    def _draws(name, **ov):
        sim = _sim(tmp_path, name, steps=24, **ov)
        sim.hub = _CountingHub(sim.hub)
        sim.run()
        return sim.hub.per_stream

    pure = _draws("pb_dpure")
    off = _draws("pb_doff", **{"labeling.place_binding.enabled": "false"})
    assert pure == off and sum(pure.values()) > 0


def test_on_draw_counts_identical(tmp_path):
    """ON でも draw 数(stream 別)が OFF と一致=新 stream ゼロ・既存 draw の追加消費ゼロ。

    プロンプト内容が変わると mock の応答は変わるが(mock は prompt を stream キーに含む)、
    **引く回数の構造**は変わらないことを固定する。差が出れば乱数を足したことの直接検知になる。
    """
    def _draws(name, **ov):
        sim = _sim(tmp_path, name, steps=24, **ov)
        sim.hub = _CountingHub(sim.hub)
        sim.run()
        return sim.hub.per_stream

    off = _draws("pb_don2")
    on = _draws("pb_don", **_ON)
    assert set(off) == set(on), f"stream 集合が ON で変わった: {set(on) ^ set(off)}"
    assert off == on, f"draw 数が ON で変わった: {off} vs {on}"


def test_off_prompt_line_absent(tmp_path):
    """OFF: 束縛を試みてもプロンプトに 1 行も足さない(place_bind が無いので束縛自体しない)。"""
    sim = _sim(tmp_path, "pb_offline")
    a, b = sim.agents[0], sim.agents[1]
    sim.labels.coin(a, "たまり場", step=0, sim_min=0, logger=sim.logger, node=a.node)
    _colocate(a, b)
    assert not _kind(sim, "place_label_bind")
    assert _LINE_MARK not in _speak_prompt(sim, b)


def test_event_kind_registered():
    assert "place_label_bind" in EVENT_KINDS


# ==================================================================== ON: 束縛
def test_on_binds_first_coin_only(tmp_path):
    """ON: coin_label の受理ごとに束縛 1 件。語ごとに 1 回だけ・vocab_coin と 1:1 対応。"""
    sim = _sim(tmp_path, "pb_on", steps=144, **_ON)
    sim.run()
    binds = _kind(sim, "place_label_bind")
    assert binds, "ON なのに束縛が 1 件も出ていない(mock の造語前提が崩れた)"
    words = [e.payload["word"] for e in binds]
    assert len(words) == len(set(words)), "同じ語が 2 回束縛されている(上書き禁止違反)"
    bound = sim.labels.place_bind["bound"]
    assert set(words) == set(bound), "イベントと束縛台帳が食い違う"
    # 各束縛は同 step・同 agent の vocab_coin と対応する(= coin 受理の瞬間に張られている)
    coins = {(e.step, e.agent_id, e.payload["text"]) for e in _kind(sim, "vocab_coin")}
    for e in binds:
        assert (e.step, e.agent_id, e.payload["word"]) in coins, \
            f"vocab_coin と対応しない束縛: {e.payload}"
        assert bound[e.payload["word"]]["node"] == e.payload["node"]
        assert bound[e.payload["word"]]["coiner"] == e.agent_id


def test_on_reinvention_does_not_rebind(tmp_path):
    """既存語の再発明(別の人・別の場所)は束縛を上書きしない=最初の場所が残る。"""
    sim = _sim(tmp_path, "pb_rebind", **_ON)
    a, b = sim.agents[0], sim.agents[1]
    other = next(n for n in sim.city.graph.nodes if n != a.node)
    sim.labels.coin(a, "たまり場", step=0, sim_min=0, logger=sim.logger, node=a.node)
    sim.labels.coin(b, "たまり場", step=5, sim_min=50, logger=sim.logger, node=other)
    assert len(_kind(sim, "place_label_bind")) == 1
    rec = sim.labels.place_bind["bound"]["たまり場"]
    assert rec["node"] == a.node and rec["step"] == 0 and rec["coiner"] == a.id


def test_on_tools_path_does_not_bind(tmp_path):
    """node を渡さない coin(tools のグループ名/提案文・メディア発)は束縛しない。"""
    sim = _sim(tmp_path, "pb_tools", **_ON)
    a = sim.agents[0]
    sim.labels.coin(a, "みんなの会", step=0, sim_min=0, logger=sim.logger)
    sim.labels.coin_media("おしらせ", step=0, sim_min=0, logger=sim.logger)
    assert not _kind(sim, "place_label_bind")
    assert sim.labels.place_bind["bound"] == {}


# ================================================================== ON: 知覚
def test_on_prompt_line_for_colocated_other(tmp_path):
    """同ノードの**別のエージェント**の熟慮プロンプトに中立 1 行が入る(プロンプト実体で検証)。"""
    sim = _sim(tmp_path, "pb_line", **_ON)
    a, b = sim.agents[0], sim.agents[1]
    sim.labels.coin(a, "たまり場", step=0, sim_min=0, logger=sim.logger, node=a.node)
    _colocate(a, b)
    prompt = _speak_prompt(sim, b)
    line = next(ln for ln in prompt.splitlines() if _LINE_MARK in ln)
    assert line == "この場所の呼ばれ方: 「たまり場」と呼ばれることがある。"
    # 行動指示・評価語を含まない(状態記述のみ=促進しない)
    assert not any(w in line for w in ("しましょう", "おすすめ", "ぜひ", "べき", "良い", "人気"))


def test_on_prompt_line_absent_at_other_node(tmp_path):
    """束縛ノード以外に居る人のプロンプトには入らない(場所キーで効いている)。"""
    sim = _sim(tmp_path, "pb_other", **_ON)
    a, b = sim.agents[0], sim.agents[1]
    sim.labels.coin(a, "たまり場", step=0, sim_min=0, logger=sim.logger, node=a.node)
    b.node = next(n for n in sim.city.graph.nodes if n != a.node)
    b.building, b.floor, b.sleeping = None, 0, False
    assert _LINE_MARK not in _speak_prompt(sim, b)


def test_min_adopters_gate(tmp_path):
    """min_adopters=2: 造語者 1 人だけでは出ず、2 人目が採用した時点で出る。"""
    sim = _sim(tmp_path, "pb_gate", **{**_ON, "labeling.place_binding.min_adopters": "2"})
    a, b = sim.agents[0], sim.agents[1]
    sim.labels.coin(a, "たまり場", step=0, sim_min=0, logger=sim.logger, node=a.node)
    _colocate(a, b)
    assert sim.labels.place_line(a.node) is None, "採用者 1 人で min_adopters=2 を通過した"
    assert _LINE_MARK not in _speak_prompt(sim, b)
    # 2 人目が採用(complex contagion: adopt_threshold=2 回聞く)
    for _ in range(int(sim.cfg.labeling.adopt_threshold)):
        sim.labels.on_hear(b, ["たまり場"], a.id, step=1, sim_min=10,
                           channel="face", logger=sim.logger)
    assert "たまり場" in b.adopted, "テスト前提: 2 回聴取で採用されるはず"
    assert sim.labels.place_line(a.node) == "この場所の呼ばれ方: 「たまり場」と呼ばれることがある。"


def test_multi_word_deterministic_choice(tmp_path):
    """同一ノードに複数語 → 束縛 step 降順 → 同 step は語の辞書順で 1 語だけ選ぶ。"""
    sim = _sim(tmp_path, "pb_multi", **_ON)
    a, b, c = sim.agents[0], sim.agents[1], sim.agents[2]
    _colocate(a, b)
    _colocate(a, c)
    sim.labels.coin(a, "あさの顔", step=0, sim_min=0, logger=sim.logger, node=a.node)
    assert "あさの顔" in sim.labels.place_line(a.node)
    sim.labels.coin(b, "ゆうの顔", step=3, sim_min=30, logger=sim.logger, node=a.node)
    assert "ゆうの顔" in sim.labels.place_line(a.node), "新しい束縛(step 降順)が優先されない"
    # 同 step の同着は語の辞書順(決定論)。"あ" < "い" < "ゆ"(コードポイント順)
    sim.labels.coin(c, "いまの顔", step=3, sim_min=30, logger=sim.logger, node=a.node)
    assert "いまの顔" in sim.labels.place_line(a.node), "同 step の同着が辞書順で解決されていない"


def test_prompt_line_false_binds_without_perception(tmp_path):
    """prompt_line=false: 束縛と観測(イベント・L2)は行うが知覚には出さない。"""
    sim = _sim(tmp_path, "pb_noline", **{**_ON, "labeling.place_binding.prompt_line": "false"})
    a, b = sim.agents[0], sim.agents[1]
    sim.labels.coin(a, "たまり場", step=0, sim_min=0, logger=sim.logger, node=a.node)
    _colocate(a, b)
    assert len(_kind(sim, "place_label_bind")) == 1
    assert sim.labels.place_bound_count() == 1
    assert sim.labels.place_line(a.node) is None
    assert _LINE_MARK not in _speak_prompt(sim, b)


# ==================================================================== 観測 L2
def test_l2_place_bound_labels(tmp_path):
    """ON: L2 に累計束縛語数が出て、末尾値が束縛台帳のサイズと一致し単調非減少。"""
    sim = _sim(tmp_path, "pb_l2", steps=144, **_ON)
    sim.run()
    rows = pq.read_table(tmp_path / "pb_l2" / "l2_metrics.parquet").to_pylist()
    series = [r["place_bound_labels"] for r in rows]
    assert series[-1] == len(sim.labels.place_bind["bound"]) > 0
    assert all(b >= a for a, b in zip(series, series[1:])), "累計が減っている"


# ============================================================ resume / R1 契約
def test_resume_matches_straight(tmp_path):
    """ON の resume==straight(束縛台帳は LabelSystem 同梱で checkpoint に載る)。

    round-trip で台帳そのものの復元も直接固定する(mock で束縛が偶然 0 件でも空回りしないよう
    「split 時点で束縛済み」を前提として明示 assert する)。
    """
    def _cfg(name, n_steps, **ov):
        dot = ["run.seed=42", "run.n_agents=15", f"run.n_steps={n_steps}",
               f"run.name={name}", "model.backend=mock"]
        dot += [f"{k}={v}" for k, v in {**_ON, **ov}.items()]
        return load_config(dot)

    def _rows(run_dir, stem):
        return pq.read_table(Path(run_dir) / f"{stem}.parquet").to_pylist()

    split, total = 60, 105
    straight = tmp_path / "pb_straight"
    Simulation(_cfg("pb_straight", total), out_dir=straight).run()

    resumed = tmp_path / "pb_resumed"
    every = {"observer.checkpoint_every": split}
    sim1 = Simulation(_cfg("pb_resumed", split, **every), out_dir=resumed)
    for step in range(split):
        scheduler.run_step(sim1, step)
    bound_at_split = dict(sim1.labels.place_bind["bound"])
    assert bound_at_split, "split 時点で束縛が 0 件(テスト前提が崩れた=split を後ろへ)"
    ckpt = checkpoint.save(sim1, split, resumed / "checkpoint" / f"ckpt-{split:06d}.pkl.gz")
    sim1.logger.flush_segment()
    sim2 = Simulation(_cfg("pb_resumed", total, **every), out_dir=resumed)
    sim2.run(resume_from=resumed)

    for stem in ("l1_events", "l2_metrics", "l3_snapshots"):
        assert _rows(straight, stem) == _rows(resumed, stem), f"{stem} 不一致(place_binding resume)"

    # 直接検証: checkpoint round-trip で束縛台帳が復元される(空回り防止)
    sim3 = Simulation(_cfg("pb_ck", split, **every), out_dir=tmp_path / "pb_ck")
    checkpoint.load(sim3, ckpt)
    assert sim3.labels.place_bind["bound"] == bound_at_split, "束縛台帳が resume で復元されていない"
    assert sim3.labels.place_bind["adopters"] == sim1.labels.place_bind["adopters"]


def test_llm_call_count_k_invariant(tmp_path):
    """ON のまま compute_matched 下で k=free と k=off の generate 呼数が完全一致(R1)。"""
    def _run(name, writeback):
        sim = _sim(tmp_path, name, steps=100,
                   **{**_ON, "controls.mode": "compute_matched",
                      "k.writeback": writeback})
        sim.llm = _FixedLLM(json.dumps({"action": "speak", "text": "x"},
                                       ensure_ascii=False))
        sim.run()
        return sim

    free = _run("pb_kfree", "free")
    off = _run("pb_koff", "off")
    assert free.llm.calls == off.llm.calls > 0, \
        f"place_binding の呼数が k に依存(R1 違反): free={free.llm.calls} off={off.llm.calls}"


def test_on_is_deterministic(tmp_path):
    """同 seed の 2 ラン(ON)が L1 完全一致=乱数を足していない・順序が安定。"""
    a = _sim(tmp_path, "pb_det_a", steps=72, **_ON)
    a.run()
    b = _sim(tmp_path, "pb_det_b", steps=72, **_ON)
    b.run()
    assert _l1(a) == _l1(b), "place_binding ON の決定論が崩れている"
