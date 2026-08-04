"""IF-C(情報オブジェクトの一般化 = 噂 `information.rumors`)のテスト。

正典
  - docs/research/if-lane-research.md **§2**(Daley-Kendall / Maki-Thompson の stifler 化・
    理論値 = 最終 ignorant ≈ 20% / spreader ピーク 1−ln2 ≈ 0.307・
    OASIS の scale / depth / max_breadth・Generative Agents は事後インタビュー測定で
    ID オブジェクト非所持 = 本シムの Item + transmissions が**先行**)
  - docs/research/llm-world-interface-audit.md **§3**(情報カテゴリの現状 = Item.kind は
    rumor / institution を宣言済みだが生成は vocab のみ・伝聞はテキスト直接注入)
  - docs/plans/if-sv-p4-plan.md 「IF-C」行

守るもの(検収基準の順)
  ① OFF(既定)= ゴールデン L1 バイト一致・Item 0 件・キー不発生・属性も state も生えない
  ② ON: 構造化イベント(host_event 等)→ rumor_born → **当事者と同席者**が初期 knower
  ③ speak/dm 相乗り: transmission 1 件 + 聞き手の記憶 1 行。**LLM 呼数完全一致**
     (呼数とプロンプトの差は**記憶経由だけ**であることを明示テスト)
  ④ MT stifler: 既知の相手への冗長接触で語りが止まる / stifle="none" では止まらない
  ⑤ 伝播木が transmissions から再構成でき depth / max_breadth が期待値
  ⑥ ON ランの同 seed 2 ラン一致
  ⑦ resume == straight
  ⑧ 噂文面に数字・実験条件語が無い機械検査
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq

from society import registry as R
from society import rumors as RM
from society.config import load_config
from society.engine import checkpoint, scheduler
from society.engine.simulation import Simulation
from society.observer.schema import EVENT_KINDS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyze_rumors as AR                                        # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "data" / "golden_baseline_l1.json"

# test_scenario.py:45 / test_rejection_notify.py:44 と同じ「意図的な既定挙動追加」の中立化
_GOLDEN_NEUTRAL = {"economy.wages.自営": 0, "planning.enabled": "false",
                   "transit_ride.taxi.enabled": "false",
                   "transit_ride.bus.enabled": "false", "rules.enabled": "false"}

OFF = {"information.rumors.enabled": "false"}
ON = {"information.rumors.enabled": "true"}
ON_NONE = {**ON, "information.rumors.stifle": "none"}


# --------------------------------------------------------------------------- #
# 共通ヘルパ(test_rejection_notify.py と同型)
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


def _l1x(sim):
    return [[e.step, e.agent_id, e.kind, e.llm_call_id,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _kind(sim, kind):
    return [e for e in sim.logger.events if e.kind == kind]


def _mem(agent, kind=None) -> list[str]:
    return [ep.text for ep in agent.mem.buffer if kind is None or ep.kind == kind]


class _FixedLLM:
    """**プロンプト非依存**の巡回応答スタブ(test_rejection_notify / test_day_plan と同型)。

    ★mock は `hub.stream("mock", rng_key, prompt)` = **プロンプト全文**で乱数を引くので、
      記憶に 1 行増えるだけで応答が変わり、世界が分岐して呼数の比較にならない。
      ここでは応答を**呼び出し番号の関数**にすることで「記憶に 1 行増えたこと」が
      行動列を変えない世界を作り、呼数の比較を純粋にする(検収基準 ③)。
      同時に「応答の順序が ON/OFF でずれないこと」も暗黙に固定される。
    """

    name = "fixed"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.hits = 0
        self.prompts: list[str] = []

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        out = self.responses[self.calls % len(self.responses)]
        self.calls += 1
        self.prompts.append(prompt)
        return out, str(self.calls), False


def _acts(*objs) -> list[str]:
    return [json.dumps(o, ensure_ascii=False) for o in objs]


# 源イベント(host_event)と会話(speak)の両方が立つ巡回応答。
_HOST_AND_TALK = _acts(
    {"action": "host_event", "title": "集まり", "hours_later": 1},
    {"action": "speak", "text": "こんにちは"},
    {"action": "speak", "text": "そうですね"},
    {"action": "speak", "text": "また今度"},
)


def _colocate(dst, src) -> None:
    """dst を src と同席させる(hearers_of の context + 半径を満たす)。"""
    dst.x, dst.y = src.x, src.y
    dst.node = src.node
    dst.loc = src.loc
    dst.building = src.building
    dst.floor = src.floor
    dst.sleeping = False


def _host(sim, agent, step=0, sim_min=0, title="集まり"):
    """実コード(tools._host_event)で source イベントを 1 件起こす。"""
    sim.tools._host_event(sim, agent, {"title": title, "hours_later": 1},
                          step, sim_min)


# --------------------------------------------------------------------------- #
# (A) 出荷既定・宣言(検収基準 ①の前段)
# --------------------------------------------------------------------------- #
def test_shipped_default_is_off():
    cfg = load_config()
    assert bool(cfg.information.rumors.enabled) is False
    assert str(cfg.information.rumors.stifle) == RM.MT
    assert list(cfg.information.rumors.sources) == list(RM.DEFAULT_SOURCES)


def test_registry_and_schema_declared():
    feat = {f.id: f for f in R.FEATURES}["information.rumors.enabled"]
    assert feat.repro_tier == "strict"
    assert feat.affects_k is False          # generate() の呼び出しサイトを 1 つも足さない
    assert feat.fingerprint_risk == "possible"
    assert feat.off_value is False
    assert "rumor_born" in EVENT_KINDS and "rumor_stifle" in EVENT_KINDS


def test_source_table_is_finite_and_defaults_are_conservative():
    """源イベント表が有限で、既定は**公共の出来事だけ**(私生活の 2 種は既定に入れない)。"""
    assert set(RM.DEFAULT_SOURCES) <= set(RM.SRC_SPECS)
    assert set(RM.DEFAULT_SOURCES) == {"event_host", "venture_open", "enforcement"}
    private = set(RM.SRC_SPECS) - set(RM.DEFAULT_SOURCES)
    assert private == {"partner_formed", "relation_break"}
    # 表の形が壊れていない(form は 2 種・テンプレのプレースホルダは 1 つだけ)
    for kind, (form, tmpl, extra) in RM.SRC_SPECS.items():
        assert form in ("place", "actor"), kind
        holder = "{place}" if form == "place" else "{actor}"
        other = "{actor}" if form == "place" else "{place}"
        assert tmpl.count(holder) == 1 and other not in tmpl, kind
        assert isinstance(extra, tuple)
    # 源イベント種は L1 のスキーマに実在する(綴り間違いで永久に 0 件になるのを防ぐ)
    for kind in RM.SRC_SPECS:
        assert kind in EVENT_KINDS, f"源イベント種 {kind} が L1 スキーマに無い"


def test_unknown_config_values_degrade_to_defaults():
    cfg = RM.build_cfg({"enabled": True, "stifle": "とんでも",
                        "sources": ["event_host", "存在しない種"],
                        "stifle_threshold": 0})
    assert cfg["stifle"] == RM.MT                  # 未知の停止規則は既定へ倒す
    assert cfg["sources"] == ["event_host"]        # 定型文を持たない種は黙って捨てる
    assert cfg["stifle_threshold"] == 1            # 0 は無意味なので下限 1


def test_frozen_metric_spec_files_are_untouched():
    """凍結 14 ファイル(metrics_spec.SPEC_FILES)に本バッチの痕跡が 1 つも無い。

    truth_ledger.py は 1 バイトも触らず、伝播記録も既存の provenance.transmit に
    相乗りするので metrics_spec_hash は無風。
    """
    from society.observer import metrics_spec as MS
    root = Path(__file__).resolve().parents[1]
    assert len(MS.SPEC_FILES) == 14
    for rel in MS.SPEC_FILES:
        text = (root / rel).read_text(encoding="utf-8")
        assert "rumors" not in text and "rumor_born" not in text, \
            f"凍結ファイル {rel} に IF-C の痕跡がある(metrics_spec_hash が動く)"


def test_provenance_and_labels_are_untouched():
    """``observer/provenance.py`` / ``labeling/labels.py`` を**変更していない**ことを固定する。

    IF-C は既存 API(``ItemStore.new_item`` / ``transmit``)だけで足りる = 伝播記録系の
    二重化を作らない(if-lane-research §2-3 の示唆 2)。
    """
    root = Path(__file__).resolve().parents[1] / "src" / "society"
    for rel in ("observer/provenance.py", "labeling/labels.py"):
        text = (root / rel).read_text(encoding="utf-8")
        assert "rumors" not in text, f"{rel} に IF-C の痕跡がある(二重化の兆候)"


def test_module_creates_no_llm_call_site_and_no_rng_stream():
    """噂は generate() の呼び出しサイトも乱数 stream も 1 つも作らない(静的検査)。

    散文(docstring / コメント)は対象外 — **実際に評価される識別子**だけを見る。
    """
    tree = ast.parse(Path(RM.__file__).read_text(encoding="utf-8"))
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert "generate" not in attrs and "generate" not in names
    assert "llm" not in attrs, "噂が LLM を触っている"
    assert "stream" not in attrs, "噂が乱数 stream を引いている(決定論違反)"
    assert "hub" not in attrs and "random" not in attrs


def test_on_talk_does_not_take_the_utterance_text():
    """設計判断 (2) の機械固定: 伝播口は**発話テキストを引数に取らない**。"""
    tree = ast.parse(Path(RM.__file__).read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "on_talk")
    args = [a.arg for a in fn.args.args]
    assert args == ["sim", "speaker", "listeners", "channel", "step", "sim_min"]
    assert not any("text" in a for a in args), "噂の伝播口が発話本文を受け取っている"


# --------------------------------------------------------------------------- #
# (B) OFF = 現行と 1 バイトも変わらない(検収基準 ①)
# --------------------------------------------------------------------------- #
def test_off_matches_golden(tmp_path):
    golden = json.load(open(GOLDEN, encoding="utf-8"))
    sim = _sim(tmp_path, "rm_golden", **_GOLDEN_NEUTRAL)
    sim.run()
    assert _l1(sim) == golden, "IF-C の seam がゴールデンを動かしている"


def test_off_matches_pure_default(tmp_path):
    pure = _sim(tmp_path, "rm_pure")
    pure.run()
    off = _sim(tmp_path, "rm_off", **OFF)
    off.run()
    assert _l1x(pure) == _l1x(off)


def test_off_emits_nothing_and_grows_no_state(tmp_path):
    """OFF では源イベントを起こしても Item も L1 も state も属性も 1 も動かない。"""
    sim = _sim(tmp_path, "rm_off_noop", n_steps=1)
    a = sim.agents[0]
    n_items, n_events = len(sim.items.items), len(sim.logger.events)
    _host(sim, a)
    RM.phase(sim, 0, 0)
    RM.on_talk(sim, a, sim.agents[1:3], "face", 0, 0)
    assert len(sim.items.items) == n_items, "OFF なのに Item が増えた(item_id の採番が動く)"
    new = sim.logger.events[n_events:]
    assert not [e for e in new if e.kind in ("rumor_born", "rumor_stifle",
                                             "transmission")]
    assert getattr(sim, "_rumor_state", None) is None
    assert getattr(a, RM.RUMOR_KEY, None) is None
    assert RM.provenance(sim) is None


def test_off_summary_has_no_key(tmp_path):
    sim = _sim(tmp_path, "rm_sum_off", n_steps=24, n_agents=10)
    sim.run()
    summary = json.loads((tmp_path / "rm_sum_off" / "summary.json")
                         .read_text(encoding="utf-8"))
    assert "rumors" not in summary


def test_off_prompts_are_byte_identical(tmp_path):
    def _prompts(name, **ov):
        sim = _sim(tmp_path, name, n_steps=72, n_agents=12, **ov)
        seen: list = []
        inner = sim.llm.generate

        def _gen(prompt, **kw):
            seen.append(prompt)
            return inner(prompt, **kw)
        sim.llm.generate = _gen
        sim.run()
        return seen

    base = _prompts("rm_p_base")
    off = _prompts("rm_p_off", **OFF)
    assert base and base == off


# --------------------------------------------------------------------------- #
# (C) 噂の誕生: 当事者 + 同席者が初期 knower(検収基準 ②)
# --------------------------------------------------------------------------- #
def test_host_event_births_a_rumor_with_witnesses(tmp_path):
    sim = _sim(tmp_path, "rm_birth", n_steps=1, **ON)
    host, witness, far = sim.agents[0], sim.agents[1], sim.agents[2]
    _colocate(witness, host)
    far.loc, far.building = "outside", ""          # 同席していない(=知らないままのはず)
    n0 = len(sim.logger.events)
    _host(sim, host)
    RM.phase(sim, 0, 0)
    born = [e for e in sim.logger.events[n0:] if e.kind == "rumor_born"]
    assert len(born) == 1, f"rumor_born が {len(born)} 件"
    p = born[0].payload
    assert p["src_kind"] == "event_host"
    assert p["item_id"].startswith("rumor-")
    assert host.id in p["knowers"] and witness.id in p["knowers"]
    assert far.id not in p["knowers"], "同席していない者が初期 knower になっている"
    assert p["knowers"] == sorted(p["knowers"]), "knowers が agent_id 昇順でない"
    # 当事者・目撃者は spreader(ignorant は属性そのものを持たない)
    for a in (host, witness):
        assert RM.knows(a, p["item_id"])
        assert getattr(a, RM.RUMOR_KEY)[p["item_id"]]["role"] == RM.SPREADER
    assert getattr(far, RM.RUMOR_KEY, None) is None
    # 噂 Item は語彙辞書に載らない(= 採用しうる「語」にしない)
    item = sim.items.items[p["item_id"]]
    assert item.kind == RM.KIND
    assert item.text not in sim.labels.text_to_item
    # 初期 knower の記憶には 1 行も入らない(当人はその場に居た=伝聞ではない)
    assert not _mem(host, RM.KIND) and not _mem(witness, RM.KIND)


def test_source_kinds_outside_the_conf_list_never_birth(tmp_path):
    """conf の sources に無い種からは 1 件も生まれない(既定は公共 3 種だけ)。"""
    sim = _sim(tmp_path, "rm_src_gate", n_steps=1, **ON,
               **{"information.rumors.sources": "[venture_open]"})
    a = sim.agents[0]
    n0 = len(sim.logger.events)
    _host(sim, a)                                  # event_host は sources に無い
    RM.phase(sim, 0, 0)
    assert not [e for e in sim.logger.events[n0:] if e.kind == "rumor_born"]


def test_max_per_step_caps_the_birth_rate(tmp_path):
    sim = _sim(tmp_path, "rm_cap", n_steps=1, **ON,
               **{"information.rumors.max_per_step": 2})
    a = sim.agents[0]
    n0 = len(sim.logger.events)
    for i in range(5):
        _host(sim, a, title=f"集まり{i}")
    RM.phase(sim, 0, 0)
    born = [e for e in sim.logger.events[n0:] if e.kind == "rumor_born"]
    assert len(born) == 2, f"max_per_step が効いていない({len(born)} 件)"


# --------------------------------------------------------------------------- #
# (D) 伝播: 会話への相乗り(検収基準 ③)
# --------------------------------------------------------------------------- #
def _seed_rumor(sim, host, others=()):
    """host に噂を 1 件持たせる(others も同席させて初期 knower にする)。"""
    for o in others:
        _colocate(o, host)
    _host(sim, host)
    RM.phase(sim, 0, 0)
    return [e for e in sim.logger.events if e.kind == "rumor_born"][-1].payload["item_id"]


def test_speak_carries_the_rumor_and_writes_one_memory_line(tmp_path):
    sim = _sim(tmp_path, "rm_talk", n_steps=1, **ON)
    speaker, listener = sim.agents[0], sim.agents[1]
    listener.loc, listener.building = "outside", ""     # 誕生時は同席させない
    iid = _seed_rumor(sim, speaker)
    listener.loc, listener.building = speaker.loc, speaker.building
    _colocate(listener, speaker)
    n0, m0 = len(sim.logger.events), len(listener.mem.buffer)
    assert RM.on_talk(sim, speaker, [listener], "face", 1, 10) == 1
    tr = [e for e in sim.logger.events[n0:] if e.kind == "transmission"]
    assert len(tr) == 1, "transmission が 1 件でない"
    assert tr[0].payload == {"item_id": iid, "from": speaker.id, "channel": "face"}
    assert tr[0].agent_id == listener.id
    assert len(listener.mem.buffer) == m0 + 1, "記憶が 1 行だけ増えていない"
    assert _mem(listener, RM.KIND) == [sim.items.items[iid].text]
    assert RM.knows(listener, iid)
    # 既存 API に乗っている = Item.transmissions からもカスケード木が再構成できる
    assert sim.items.items[iid].transmissions == [(1, speaker.id, listener.id, "face")]


def test_propagation_off_blocks_the_rumor_channel(tmp_path):
    """ablate.propagation_off では噂も他者の文脈へ入らない(内容チャネルの一貫性)。"""
    sim = _sim(tmp_path, "rm_propoff", n_steps=1, **ON,
               **{"ablate.propagation_off": "true"})
    speaker, listener = sim.agents[0], sim.agents[1]
    listener.loc, listener.building = "outside", ""
    iid = _seed_rumor(sim, speaker)
    listener.loc, listener.building = speaker.loc, speaker.building
    _colocate(listener, speaker)
    n0 = len(sim.logger.events)
    assert RM.on_talk(sim, speaker, [listener], "face", 1, 10) == 0
    assert not [e for e in sim.logger.events[n0:] if e.kind == "transmission"]
    assert not RM.knows(listener, iid)


def test_max_per_talk_limits_how_many_rumors_ride_one_utterance(tmp_path):
    sim = _sim(tmp_path, "rm_talkcap", n_steps=1, **ON_NONE)
    speaker, listener = sim.agents[0], sim.agents[1]
    listener.loc, listener.building = "outside", ""
    _seed_rumor(sim, speaker)
    _seed_rumor(sim, speaker)
    assert len(getattr(speaker, RM.RUMOR_KEY)) == 2
    listener.loc, listener.building = speaker.loc, speaker.building
    _colocate(listener, speaker)
    assert RM.on_talk(sim, speaker, [listener], "face", 1, 10) == 1  # 既定 max_per_talk=1


def test_llm_call_count_is_identical_between_on_and_off(tmp_path):
    """★検収基準 ③: 噂 ON/OFF で **generate() の呼数が完全一致**する。

    プロンプト**内容**は変わりうる(伝聞の 1 行が記憶に載るため)。ここが本テストの
    肝で、「プロンプト盲ではないが呼数は動かない」= 追加 LLM 呼ゼロ・発火点不変を
    固定する。差が**記憶経由だけ**であることも同時に確かめる
    (差分行が全て伝聞の定型文であること)。
    """
    def _run(name, **ov):
        sim = _sim(tmp_path, name, n_steps=216, n_agents=18, **ov)
        sim.llm = _FixedLLM(_HOST_AND_TALK)
        sim.run()
        return sim

    off = _run("rm_k_off", **OFF)
    on = _run("rm_k_on", **ON)
    assert on.llm.calls == off.llm.calls > 0, \
        f"噂 ON で呼数が動いた: on={on.llm.calls} off={off.llm.calls}"
    assert _kind(on, "rumor_born"), "テストが空振り(ON ランで噂が 1 件も生まれていない)"
    assert _kind(on, "transmission"), "テストが空振り(伝播が 1 件も起きていない)"
    # 差は**記憶経由だけ** = 差分行はすべて記憶由来の欄に閉じ、新しい欄は 1 つも増えない
    # (deliberate.build_prompt で agent.mem から作られる 3 欄。噂の 1 行はここへ混ざる)。
    mem_prefixes = ("直近の出来事:", "記憶に残っていること:", "ふと思い出したこと:")
    texts = {i.text for i in on.items.items.values() if i.kind == RM.KIND}
    diff_lines: set[str] = set()
    for pa, pb in zip(off.llm.prompts, on.llm.prompts):
        if pa == pb:
            continue
        la, lb = pa.splitlines(), pb.splitlines()
        diff_lines |= (set(lb) - set(la)) | (set(la) - set(lb))
    assert diff_lines, "ON なのにプロンプトが 1 行も変わっていない(空振り)"
    for line in diff_lines:
        assert line.startswith(mem_prefixes), \
            f"記憶欄の外にプロンプトの差分が出た(新しい帯域が増えた): {line!r}"
    assert any(t in line for line in diff_lines for t in texts), \
        "差分に伝聞の定型文が 1 つも現れていない(噂が記憶へ載っていない)"
    # プロンプトに現れる噂の定型文は**記憶由来の欄にしか**現れない
    for prompt in on.llm.prompts:
        for line in prompt.splitlines():
            if any(t in line for t in texts):
                assert line.startswith(mem_prefixes), \
                    f"噂が記憶欄以外の帯域からプロンプトへ入った: {line!r}"


def test_llm_call_count_is_k_invariant(tmp_path):
    """compute_matched 下で k=free と k=off の呼数が完全一致(R1)。"""
    def _run(name, writeback):
        sim = _sim(tmp_path, name, n_steps=144, n_agents=15, **ON,
                   **{"controls.mode": "compute_matched", "k.writeback": writeback})
        sim.llm = _FixedLLM(_HOST_AND_TALK)
        sim.run()
        return sim
    free = _run("rm_kfree", "free")
    off = _run("rm_koff", "off")
    assert free.llm.calls == off.llm.calls > 0


# --------------------------------------------------------------------------- #
# (E) stifler 化(検収基準 ④)
# --------------------------------------------------------------------------- #
def test_mt_stifles_after_a_redundant_contact(tmp_path):
    """Maki-Thompson: 既に知っている相手へ語ったら以後その噂を語らない。"""
    sim = _sim(tmp_path, "rm_mt", n_steps=1, **ON)
    a, b, c = sim.agents[0], sim.agents[1], sim.agents[2]
    for x in (b, c):
        x.loc, x.building = "outside", ""
    iid = _seed_rumor(sim, a)
    for x in (b, c):
        x.loc, x.building = a.loc, a.building
        _colocate(x, a)
    assert RM.on_talk(sim, a, [b], "face", 1, 10) == 1          # 未知の相手 → 伝わる
    n0 = len(sim.logger.events)
    assert RM.on_talk(sim, a, [b], "face", 2, 20) == 0          # 既知の相手 → 冗長接触
    st = [e for e in sim.logger.events[n0:] if e.kind == "rumor_stifle"]
    assert len(st) == 1 and st[0].payload == {"item_id": iid}
    assert st[0].agent_id == a.id
    assert getattr(a, RM.RUMOR_KEY)[iid]["role"] == RM.STIFLER
    # 黙った後は、まだ知らない相手にも語らない(= 噂が止まる)
    assert RM.on_talk(sim, a, [c], "face", 3, 30) == 0
    assert not RM.knows(c, iid)
    assert sim._rumor_state["stiflers"] == 1


def test_none_never_stifles(tmp_path):
    """stifle="none" では語り続ける(= 現行の伝播機構と同じ「止まらない世界」)。"""
    sim = _sim(tmp_path, "rm_none", n_steps=1, **ON_NONE)
    a, b, c = sim.agents[0], sim.agents[1], sim.agents[2]
    for x in (b, c):
        x.loc, x.building = "outside", ""
    iid = _seed_rumor(sim, a)
    for x in (b, c):
        x.loc, x.building = a.loc, a.building
        _colocate(x, a)
    assert RM.on_talk(sim, a, [b], "face", 1, 10) == 1
    n0 = len(sim.logger.events)
    assert RM.on_talk(sim, a, [b], "face", 2, 20) == 0          # 伝わりはしない(既知)
    assert not [e for e in sim.logger.events[n0:] if e.kind == "rumor_stifle"]
    assert getattr(a, RM.RUMOR_KEY)[iid]["role"] == RM.SPREADER
    assert RM.on_talk(sim, a, [c], "face", 3, 30) == 1, "none なのに語りが止まった"


def test_stifle_threshold_delays_the_stop(tmp_path):
    sim = _sim(tmp_path, "rm_thr", n_steps=1, **ON,
               **{"information.rumors.stifle_threshold": 3})
    a, b = sim.agents[0], sim.agents[1]
    b.loc, b.building = "outside", ""
    iid = _seed_rumor(sim, a)
    b.loc, b.building = a.loc, a.building
    _colocate(b, a)
    RM.on_talk(sim, a, [b], "face", 1, 10)
    for step in (2, 3):
        RM.on_talk(sim, a, [b], "face", step, step * 10)
        assert getattr(a, RM.RUMOR_KEY)[iid]["role"] == RM.SPREADER
    RM.on_talk(sim, a, [b], "face", 4, 40)
    assert getattr(a, RM.RUMOR_KEY)[iid]["role"] == RM.STIFLER


def test_a_full_on_run_actually_stifles(tmp_path):
    """ラン全体でも stifler 化が実際に立つ(空振り検収を防ぐ)。"""
    sim = _sim(tmp_path, "rm_run", n_steps=432, n_agents=25, **ON)
    sim.run()
    assert _kind(sim, "rumor_born"), "ON ランで噂が 1 件も生まれていない"
    assert _kind(sim, "rumor_stifle"), "ON ランで stifler 化が 1 件も起きていない"
    summary = json.loads((tmp_path / "rm_run" / "summary.json")
                         .read_text(encoding="utf-8"))
    prov = summary["rumors"]
    assert prov["born"] == len(_kind(sim, "rumor_born"))
    assert prov["stiflers"] == len(_kind(sim, "rumor_stifle"))
    assert prov["stifle"] == RM.MT and prov["sources"] == list(RM.DEFAULT_SOURCES)
    assert sum(prov["reach_hist"].values()) == prov["born"]


# --------------------------------------------------------------------------- #
# (F) 忘却 TTL
# --------------------------------------------------------------------------- #
def test_forget_ttl_drops_old_rumors_at_the_day_boundary(tmp_path):
    sim = _sim(tmp_path, "rm_ttl", n_steps=1, **ON,
               **{"information.rumors.forget_steps": 10})
    a = sim.agents[0]
    iid = _seed_rumor(sim, a)
    RM.phase(sim, 1, 1440)                         # 日境界だが TTL 未経過
    assert RM.knows(a, iid)
    RM.phase(sim, 100, 2880)                       # 次の日境界 = TTL 経過
    assert not RM.knows(a, iid), "TTL を超えた噂が忘れられていない"
    assert sim._rumor_state["forgotten"] == 1


def test_forget_off_by_default_never_sweeps(tmp_path):
    sim = _sim(tmp_path, "rm_ttl_off", n_steps=1, **ON)
    a = sim.agents[0]
    iid = _seed_rumor(sim, a)
    for step, sm in ((100, 1440), (200, 2880), (300, 4320)):
        RM.phase(sim, step, sm)
    assert RM.knows(a, iid)
    assert sim._rumor_state["forgotten"] == 0


# --------------------------------------------------------------------------- #
# (G) 伝播木の再構成(検収基準 ⑤)= scripts/analyze_rumors.py(読み取り専用)
# --------------------------------------------------------------------------- #
def _ev(step, agent_id, kind, payload):
    return {"step": step, "agent_id": agent_id, "kind": kind, "payload": payload}


def test_cascade_is_reconstructed_with_expected_depth_and_breadth():
    """根 0 →(1,2)→(3) の木: scale=4 / depth=2 / max_breadth=2。"""
    events = [
        _ev(0, 0, "rumor_born", {"item_id": "rumor-00001", "src_kind": "event_host",
                                 "node": "n1", "knowers": [0]}),
        _ev(1, 1, "transmission", {"item_id": "rumor-00001", "from": 0, "channel": "face"}),
        _ev(1, 2, "transmission", {"item_id": "rumor-00001", "from": 0, "channel": "face"}),
        _ev(2, 3, "transmission", {"item_id": "rumor-00001", "from": 1, "channel": "face"}),
        # 既知の相手への再到達 = 木の辺にならない(redundant に数える)
        _ev(3, 2, "transmission", {"item_id": "rumor-00001", "from": 1, "channel": "face"}),
        # 語彙の transmission は混ざらない(item_id 接頭辞で分離)
        _ev(3, 9, "transmission", {"item_id": "vocab-00007", "from": 8, "channel": "face"}),
    ]
    cas = AR.build_cascades(events)
    assert set(cas) == {"rumor-00001"}
    m = AR.cascade_metrics(cas["rumor-00001"])
    assert m["scale"] == 4 and m["depth"] == 2 and m["max_breadth"] == 2
    assert m["breadth_by_level"] == {"0": 1, "1": 2, "2": 1}
    assert m["n_edges"] == 3 and m["n_redundant"] == 1
    res = AR.summarize(events, None)
    assert res["counts"]["rumors_born"] == 1
    assert res["counts"]["rumor_transmissions"] == 4
    assert res["counts"]["vocab_transmissions"] == 1
    assert res["oasis"]["depth"]["max"] == 2


def test_analyze_rumors_runs_on_a_real_run(tmp_path):
    sim = _sim(tmp_path, "rm_an", n_steps=432, n_agents=25, **ON)
    sim.run()
    events = AR.load_events(str(tmp_path / "rm_an"))
    res = AR.summarize(events, None)
    assert res["counts"]["rumors_born"] == len(_kind(sim, "rumor_born"))
    assert res["counts"]["rumor_transmissions"] == \
        sum(1 for i in sim.items.items.values() if i.kind == RM.KIND
            for _t in i.transmissions)
    # 到達人数(解析側)と reach(シム側 summary)が一致する = 二重帳簿が食い違わない
    sim_reach = {k: v["reach"] for k, v in sim._rumor_state["items"].items()}
    for c in res["cascades"]:
        assert c["scale"] == sim_reach[c["item_id"]], c["item_id"]
    md = AR.render(res)
    assert "OASIS" in md and "Daley-Kendall" in md


# --------------------------------------------------------------------------- #
# (H) no-fingerprint: 噂文面の機械検査(検収基準 ⑧)
# --------------------------------------------------------------------------- #
_BANNED = ("silent", "memory", "engaged", "ablate", "placebo", "seed", "rumor",
           "item", "spreader", "stifler",
           "実験", "条件", "閾値", "発火", "驚き", "エピソード", "モデル",
           "シミュレーション", "エージェント", "アブレーション", "対照", "観測")


def test_templates_have_no_digits_and_no_condition_words():
    """テンプレート**そのもの**に数字・実験条件語・機構語が 1 文字も無い。"""
    for kind, (_form, tmpl, _extra) in RM.SRC_SPECS.items():
        bare = tmpl.replace("{place}", "").replace("{actor}", "")
        assert not re.search(r"[0-9０-９]", bare), f"{kind}: テンプレに数字が入った"
        for w in _BANNED:
            assert w not in bare, f"{kind}: 禁止語 {w} が入った"


def test_sentence_is_a_pure_function_of_kind_and_proper_noun():
    """文面は (源イベント種, 固有名) の純関数 = 固有名を抜くと数字が 1 つも残らない。

    ★場所名そのものには "SHIBUYA 109" のような数字を含む実在名がある。したがって
      検査するのは「**世界が所有する固有名を除いた残り**に数字が無い」ことである
      (数字を持ち込んでいるのは実世界の地名であって本 module ではない)。
    """
    for kind, (form, _tmpl, _extra) in RM.SRC_SPECS.items():
        probe = "SHIBUYA 109" if form == "place" else "田中1郎"
        s = RM.sentence(kind, place=probe, actor=probe)
        assert probe in s
        rest = s.replace(probe, "")
        assert not re.search(r"[0-9０-９]", rest), f"{kind}: 固有名以外に数字が入った"
        for w in _BANNED:
            assert w not in rest, f"{kind}: 禁止語 {w} が入った"
    assert RM.sentence("存在しない種", place="どこか") == ""   # 語彙外は捏造しない
    assert RM.sentence("event_host", place="") == ""            # 解決不能なら作らない


def test_sentence_does_not_depend_on_experiment_conditions(tmp_path):
    """同じ源からは**全実験条件で同一バイト列**(no-fingerprint の機械固定)。"""
    variants = (
        {}, {"k.writeback": "off"}, {"k.writeback": "sham"},
        {"ablate.cognitive_tier": "small"}, {"controls.mode": "compute_matched"},
        {"cognition.rejection_notify": "memory"},
        {"information.rumors.stifle": "none"},
    )
    seen: set[str] = set()
    for i, extra in enumerate(variants):
        sim = _sim(tmp_path, f"rm_pf{i}", n_steps=1, **ON, **extra)
        a = sim.agents[0]
        iid = _seed_rumor(sim, a)
        seen.add(sim.items.items[iid].text)
    assert len(seen) == 1, f"条件によって噂の文面が変わった {seen}"


def test_item_id_never_reaches_a_prompt(tmp_path):
    """item_id は観測側だけの概念 — プロンプトにも記憶にも 1 度も現れない。"""
    sim = _sim(tmp_path, "rm_leak", n_steps=216, n_agents=18, **ON)
    seen: list[str] = []
    inner = sim.llm.generate

    def _gen(prompt, **kw):
        seen.append(prompt)
        return inner(prompt, **kw)
    sim.llm.generate = _gen
    sim.run()
    ids = [i.item_id for i in sim.items.items.values() if i.kind == RM.KIND]
    assert ids, "テストが空振り(噂が 1 件も生まれていない)"
    blob = "\n".join(seen)
    for iid in ids:
        assert iid not in blob, f"item_id {iid} がプロンプトへ漏れた"
    for a in sim.agents:
        for text in _mem(a):
            for iid in ids:
                assert iid not in text, f"item_id {iid} が記憶へ漏れた"


def test_l1_never_carries_the_rumor_text():
    """L1 の payload に噂の**本文**を出さない(本文は Item と記憶にだけ載る)。"""
    for kind in ("rumor_born", "rumor_stifle"):
        assert "text" not in EVENT_KINDS[kind]


# --------------------------------------------------------------------------- #
# (I) 決定論・resume(検収基準 ⑥⑦)
# --------------------------------------------------------------------------- #
def test_on_is_deterministic_across_two_runs(tmp_path):
    a = _sim(tmp_path, "rm_det_a", n_steps=288, n_agents=20, **ON)
    a.run()
    b = _sim(tmp_path, "rm_det_b", n_steps=288, n_agents=20, **ON)
    b.run()
    assert _l1x(a) == _l1x(b), "噂 ON の決定論が崩れている"
    assert _kind(a, "rumor_born"), "テストが空振り"
    assert a._rumor_state["items"] == b._rumor_state["items"]


def test_resume_matches_straight(tmp_path):
    """噂 ON で resume==straight(parquet バイト一致 + 累積タリー一致)。"""
    ov = {**ON, "run.start_tod": "00:00", "run.natural_start": "true"}
    split, total = 144, 288
    straight_dir = tmp_path / "rm_straight"
    straight = Simulation(_cfg("rm_straight", total, 20, **ov), out_dir=straight_dir)
    straight.run()
    assert _kind(straight, "rumor_born"), "テスト前提が崩れた(噂ゼロ)"

    d = tmp_path / "rm_resumed"
    sim1 = Simulation(_cfg("rm_resumed", split, 20, **ov,
                           **{"observer.checkpoint_every": split}), out_dir=d)
    for step in range(split):
        scheduler.run_step(sim1, step)
    checkpoint.save(sim1, split, d / "checkpoint" / f"ckpt-{split:06d}.pkl.gz")
    sim1.logger.flush_segment()
    sim2 = Simulation(_cfg("rm_resumed", total, 20, **ov,
                           **{"observer.checkpoint_every": split}), out_dir=d)
    sim2.run(resume_from=d)
    for stem in ("l1_events", "l2_metrics", "l3_snapshots"):
        sa = pq.read_table(straight_dir / f"{stem}.parquet").to_pylist()
        sb = pq.read_table(d / f"{stem}.parquet").to_pylist()
        assert sa == sb, f"{stem} 不一致(rumors resume)"
    ja = json.loads((straight_dir / "summary.json").read_text(encoding="utf-8"))
    jb = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    assert ja["rumors"] == jb["rumors"], \
        "累積タリー/reach が resume で straight と食い違う(中央管理の漏れ)"
