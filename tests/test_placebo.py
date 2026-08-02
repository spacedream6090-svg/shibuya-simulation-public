"""第89バッチ: プラセボ・アブレーション L1 3 種のテスト
(ablate.context_shuffle / persona_swap / context_sever)。

正典: docs/plans/source/design-discussion-20260802.md §1 —
  「L1 として『**呼び出し回数・フォーマット・乱数消費は同一のまま中身だけ壊した**
   プラセボ LLM』(文脈シャッフル・ペルソナ入れ替え・文脈遮断)を挟む」

受入基準(第89 の検収項目)
  1. 既定 OFF … 純粋既定と **L1 バイト一致**・golden 一致・**draw 数(stream 別)一致**・
                 個体に placebo 属性が生えない・manifest / summary にキーが増えない。
  2. 各 ON(mock)
     - **呼び出しサイトが 1 つも増減しない**(構造の不変)+ 実測呼数の乖離は許容内。
       ★正直な限界: プロンプトが変われば応答が変わり発火ドライブが動くので、実測 k は
         数 % 動く(第78 propagation_off +1.61% / flat_traits と同じ間接ドライブ経路)。
         「呼び出し点の増減ゼロ」は seam で直接固定し、実測は許容幅で固定する。
     - **プロンプト構造検査**(journal 全走査): 行数・節の接頭辞・区切り・項目数が保たれ、
       中身だけが違う。
     - 決定論(同 seed 2 ラン一致)・resume==straight。
     - persona_swap の**対合性**(A↔B の相互交換=全単射)。
  3. 条件間で世界が実際に変わる(変わらなければプラセボとして無意味)。
  4. 相互排他(3 種同時 ON / propagation_off 併用は ValueError)。

検証は mock のみ(実 LLM 禁止)。
"""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from society import ablate as A
from society.cognition import deliberate
from society.config import load_config
from society.engine import checkpoint, scheduler
from society.engine.simulation import Simulation
from society.llm.journal import iter_records

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN = REPO_ROOT / "tests" / "data" / "golden_baseline_l1.json"

# test_scenario.py:45 と同じ「意図的な既定挙動追加」の中立化(ゴールデン比較用)
_GOLDEN_NEUTRAL = {"economy.wages.自営": 0, "planning.enabled": "false",
                   "transit_ride.taxi.enabled": "false",
                   "transit_ride.bus.enabled": "false", "rules.enabled": "false"}

MODES = ("context_shuffle", "persona_swap", "context_sever")
CONTEXT_MODES = ("context_shuffle", "context_sever")


# --------------------------------------------------------------------------- #
def _cfg(name: str, n_steps: int = 24, n_agents: int = 12, **ov):
    dot = ["run.seed=42", f"run.n_agents={n_agents}", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def _run(tmp_path, name: str, n_steps: int = 24, n_agents: int = 12, **ov):
    sim = Simulation(_cfg(name, n_steps, n_agents, **ov), out_dir=tmp_path / name)
    return sim, sim.run()


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


class _CountingHub:
    """全 stream の draw を数えるプロキシ(test_place_binding と同型)。"""

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


_BLIND_PROMPT = "場所: どこか\n状況: ふと立ち止まって考え事。"


class _PromptBlindLLM:
    """プロンプト**内容を捨てて** rng_key だけで応答を決める LLM プロキシ。

    これを噛ませると「プラセボがプロンプトを書き換えた」効果だけが**完全に消える**。
    したがって ON/OFF が **L1 バイト一致・呼数完全一致**になることが、
      「プラセボの影響経路は *LLM の応答* ただ 1 本であり、
        呼び出しサイト・乱数消費・観測経路には 1 バイトも副作用が無い」
    ことの直接証明になる(実測の呼数ドリフトは全て応答経由の間接効果だと確定する)。
    """

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def generate(self, prompt, **kw):
        return self._inner.generate(_BLIND_PROMPT, **kw)

    def generate_many(self, reqs, **kw):
        return self._inner.generate_many(
            [{**r, "prompt": _BLIND_PROMPT} for r in reqs], **kw)


class _FakeHub:
    """build_pairing / _draw の単体検証用(RngHub と同じ口を持つ最小実装)。"""

    def __init__(self, seed: int = 7):
        from society.rng import RngHub
        self._inner = RngHub(seed)

    def stream(self, *key):
        return self._inner.stream(*key)


def _journal_prompts(out_dir: Path) -> list[str]:
    rows = []
    for p in sorted(Path(out_dir).glob("llm_journal*.jsonl.gz")):
        rows += [r.get("prompt", "") for r in iter_records(p)]
    return rows


def _journal_rows(out_dir: Path) -> list[tuple[int, str]]:
    """(agent_id, prompt) の列。rng_key は "<用途>/<agent_id>/<step>" 形式。"""
    out: list[tuple[int, str]] = []
    for p in sorted(Path(out_dir).glob("llm_journal*.jsonl.gz")):
        for rec in iter_records(p):
            parts = str(rec.get("rng_key", "")).split("/")
            if len(parts) >= 2 and parts[1].isdigit():
                out.append((int(parts[1]), rec.get("prompt", "")))
    return out


def _sections_of(prompt: str) -> dict[str, list[str]]:
    """プロンプトから「節 → 行」を抜き出す(接頭辞は ablate.SECTIONS が唯一の源)。"""
    out: dict[str, list[str]] = {}
    for line in prompt.split("\n"):
        for sec in A.SECTIONS:
            if line.startswith(sec.prefix):
                out.setdefault(sec.kind, []).append(line)
                break
        else:
            if A._reply_parts(line) is not None:
                out.setdefault(A.REPLY_KIND, []).append(line)
    return out


_JOURNAL = {"observer.llm_journal.enabled": "true"}


# =========================================================================== #
# (0) 設定の正準化と相互排他
# =========================================================================== #
def test_defaults_place_all_three_placebos_off():
    cfg = A.build_cfg(None)
    for key in MODES:
        assert cfg[key] is False, f"{key} の既定が OFF でない"
    assert A.placebo_mode(object()) is None


def test_shipped_config_has_all_three_off():
    """出荷 conf の ablate ブロックにキーが在り、全 OFF であること(R1)。"""
    raw = load_config().get("ablate")
    for key in MODES:
        assert key in raw, f"conf/config.yaml に ablate.{key} が無い"
    assert A.build_cfg(raw) == A.build_cfg(None)


@pytest.mark.parametrize("pair", [("context_shuffle", "persona_swap"),
                                  ("context_shuffle", "context_sever"),
                                  ("persona_swap", "context_sever")])
def test_two_placebos_at_once_is_rejected(pair):
    """同一軸の 2 種同時 ON は解釈不能なので**構築時に落とす**。"""
    with pytest.raises(ValueError, match="同時に 1 つだけ"):
        A.build_cfg({pair[0]: True, pair[1]: True})


@pytest.mark.parametrize("mode", MODES)
def test_placebo_with_propagation_off_is_rejected(mode):
    """propagation_off は節を消し、プラセボは節を潰す/入れ替える。重ねると同定不能。"""
    with pytest.raises(ValueError, match="併用できない"):
        A.build_cfg({mode: True, "propagation_off": True})


@pytest.mark.parametrize("mode", MODES)
def test_placebo_with_llm_off_warns_but_passes(mode, caplog):
    """llm_off はプロンプトを 1 つも組まない=プラセボが無効。落とさず WARNING で告知。"""
    with caplog.at_level("WARNING"):
        cfg = A.build_cfg({mode: True, "llm_off": True})
    assert cfg[mode] and cfg["llm_off"]
    assert any("1 バイトも効かない" in r.getMessage() for r in caplog.records)


# =========================================================================== #
# (1) 既定 OFF の不変条件(R1)
# =========================================================================== #
def test_off_matches_pure_default(tmp_path):
    """3 種を明示 OFF で書いても純粋既定と L1 バイト一致・placebo 属性も生えない。"""
    a, _ = _run(tmp_path, "pl_base")
    b, _ = _run(tmp_path, "pl_off", **{f"ablate.{m}": "false" for m in MODES})
    assert _l1(a) == _l1(b)
    assert b._placebo is None
    assert not any(hasattr(ag, "placebo") for ag in b.agents), \
        "OFF なのに個体に placebo 属性が生えている"


def test_off_matches_golden(tmp_path):
    """明示 OFF が変更前ゴールデンと一字一句一致(ゴールデンは再生成しない=掟2)。"""
    golden = json.load(open(GOLDEN, encoding="utf-8"))
    ov = {**_GOLDEN_NEUTRAL, **{f"ablate.{m}": "false" for m in MODES}}
    sim = Simulation(_cfg("pl_golden", 144, 15, **ov), out_dir=tmp_path / "pl_golden")
    sim.run()
    assert _l1(sim) == golden, "プラセボの seam が no-op でない(ゴールデン不一致)"


def test_off_draw_counts_identical(tmp_path):
    """OFF は純粋既定と draw 数(stream 別)が完全一致=新 stream ゼロ。"""
    def _draws(name, **ov):
        sim = Simulation(_cfg(name, 24, 12, **ov), out_dir=tmp_path / name)
        sim.hub = _CountingHub(sim.hub)
        sim.run()
        return sim.hub.per_stream

    pure = _draws("pl_dpure")
    off = _draws("pl_doff", **{f"ablate.{m}": "false" for m in MODES})
    assert pure == off and sum(pure.values()) > 0
    assert A.STREAM_SWAP not in pure and A.STREAM_SHUFFLE not in pure


def test_off_adds_no_manifest_or_summary_key(tmp_path):
    sim, summary = _run(tmp_path, "pl_man_off", n_steps=6)
    man = json.loads((tmp_path / "pl_man_off" / "run_manifest.json")
                     .read_text(encoding="utf-8"))
    assert "ablate" not in man
    assert "placebo" not in summary
    assert A.provenance(sim) is None


# =========================================================================== #
# (2) ON: 呼び出しサイトの不変(k 非依存の構造的根拠)
# =========================================================================== #
@pytest.mark.parametrize("mode", MODES)
def test_llm_speak_still_calls_backend_exactly_once(mode, tmp_path):
    """seam の直接検証: プラセボ ON でも _llm_speak は backend を**ちょうど 1 回**呼ぶ。

    llm_off(呼を落とす条件)との対比で「呼び出し点が 1 つも減らない」ことを固定する。
    """
    name = f"pl_seam_{mode}"
    sim = Simulation(_cfg(name, 1, 12, **{f"ablate.{mode}": "true"}),
                     out_dir=tmp_path / name)
    agent = sim.agents[0]
    before = sim.llm.calls
    scheduler._llm_speak(sim, agent, "solo", 0, 0)
    assert sim.llm.calls == before + 1


def _blind_run(tmp_path, name, **ov):
    sim = Simulation(_cfg(name, 144, 20, **ov), out_dir=tmp_path / name)
    sim.llm = _PromptBlindLLM(sim.llm)
    return sim, sim.run()


@pytest.mark.parametrize("mode", MODES)
def test_call_count_is_exactly_equal_under_prompt_blind_llm(mode, tmp_path):
    """★**呼数完全一致**の本命。プロンプト内容を捨てる LLM の下で ON==OFF(L1 バイト一致)。

    「呼び出し回数・フォーマット・乱数消費は同一のまま中身だけ壊す」(正典 §1)を
    **文字どおり**固定する唯一のテスト。実ランで呼数が動くのは応答が変わるからであって、
    プラセボが呼び出しサイト・乱数・観測経路に触ったからではない、を切り分ける。
    """
    off, s_off = _blind_run(tmp_path, f"pl_blind_off_{mode}")
    on, s_on = _blind_run(tmp_path, f"pl_blind_on_{mode}", **{f"ablate.{mode}": "true"})
    assert s_off["llm_calls"] > 0
    assert s_on["llm_calls"] == s_off["llm_calls"], \
        f"{mode}: プロンプト盲下でも呼数が違う=呼び出しサイトが増減している"
    assert _l1(on) == _l1(off), \
        f"{mode}: プロンプト盲下でも L1 が違う=応答以外の副作用がある"


@pytest.mark.parametrize("mode", MODES)
def test_llm_call_count_drift_is_bounded(mode, tmp_path):
    """実ランの呼数ドリフトが有界(< 25%)であること。

    ★正直な宣言: **完全一致ではない**。呼び出しサイトの増減はゼロ(上 2 本が固定)だが、
      プロンプトが変われば応答が変わり、応答が変われば発火ドライブ(語彙の未知度・
      驚き・会話量)が動く。この間接経路は原理的に消せない — 消せたらプラセボとして無意味。
      第78 propagation_off(+1.61%)・experiment.flat_traits と同じ構造で、こちらは
      「知っている言葉」節まで壊すぶん振れ幅が大きい(実測 40体288step で -14.0〜-1.8%)。
      **k を揃えたい比較では k.compute_matched 対照を併用すること。**
    """
    _off, s_off = _run(tmp_path, f"pl_k_off_{mode}", n_steps=144, n_agents=20)
    _on, s_on = _run(tmp_path, f"pl_k_on_{mode}", n_steps=144, n_agents=20,
                     **{f"ablate.{mode}": "true"})
    base = s_off["llm_calls"]
    assert base > 0
    drift = abs(s_on["llm_calls"] - base) / base
    assert drift < 0.25, (f"{mode}: 呼数が {base} → {s_on['llm_calls']} "
                          f"({drift:.2%})。間接ドリフトの想定幅を超えた")


# =========================================================================== #
# (3) ON: プロンプト構造検査(journal 全走査)
# =========================================================================== #
@pytest.mark.parametrize("mode", MODES)
def test_prompt_line_count_is_preserved(mode, tmp_path):
    """行数の分布が OFF と一致 = 節を消したり足したりしていない(書式の不変)。"""
    off_dir = tmp_path / "pl_fmt_off"
    Simulation(_cfg("pl_fmt_off", 72, 15, **_JOURNAL), out_dir=off_dir).run()
    on_dir = tmp_path / f"pl_fmt_{mode}"
    Simulation(_cfg(f"pl_fmt_{mode}", 72, 15,
                    **{**_JOURNAL, f"ablate.{mode}": "true"}), out_dir=on_dir).run()
    off = _journal_prompts(off_dir)
    on = _journal_prompts(on_dir)
    assert off and on, "journal にプロンプトが 1 本も残っていない"
    # 行数の**集合**ではなく分布の形(最小・最大)で見る(応答が変われば発火列は変わるため)
    assert min(len(p.split("\n")) for p in on) >= min(len(p.split("\n")) for p in off) - 1
    assert max(len(p.split("\n")) for p in on) <= max(len(p.split("\n")) for p in off) + 1


def test_context_sever_keeps_prefix_separator_and_item_count(tmp_path):
    """context_sever: 接頭辞・区切り・**項目数**が保たれ、中身だけが "…" になる。"""
    d = tmp_path / "pl_sever_fmt"
    Simulation(_cfg("pl_sever_fmt", 144, 20,
                    **{**_JOURNAL, "ablate.context_sever": "true"}), out_dir=d).run()
    prompts = _journal_prompts(d)
    assert prompts
    seen = 0
    for prompt in prompts:
        for kind, lines in _sections_of(prompt).items():
            for line in lines:
                seen += 1
                if kind == A.REPLY_KIND:
                    who, said, _tail = A._reply_parts(line)
                    assert who == said == A.SEVER_TOKEN, f"返答の状況行が潰れていない: {line}"
                    continue
                sec = next(s for s in A.SECTIONS if s.kind == kind)
                body = sec.match(line)
                items = body.split(sec.sep)
                assert set(items) == {A.SEVER_TOKEN}, f"中身が残っている: {line}"
                assert items, "項目が 0 個になった(区切りが壊れた)"
    assert seen > 0, "文脈節が 1 つも出ていない(検査になっていない)"


def test_context_sever_leaves_own_state_untouched(tmp_path):
    """「自分の状態は保持」: 時刻・場所・気分・自分の発話は 1 バイトも潰さない。"""
    d = tmp_path / "pl_sever_own"
    Simulation(_cfg("pl_sever_own", 144, 20,
                    **{**_JOURNAL, "ablate.context_sever": "true"}), out_dir=d).run()
    keep = ("時刻: ", "場所: ", "気分: ", "あなたがさっき言ったこと: ", "昨日までの日記: ")
    hits = 0
    for prompt in _journal_prompts(d):
        for line in prompt.split("\n"):
            for pref in keep:
                if line.startswith(pref):
                    hits += 1
                    assert line[len(pref):].strip("…") != "", \
                        f"自分の状態まで潰れている: {line}"
    assert hits > 0


def test_context_shuffle_replaces_content_but_keeps_prefix(tmp_path):
    """context_shuffle: 接頭辞は同じで中身が**別の実在の節**へ入れ替わる。

    ドナーは「同一ラン内の直近の同種節」なので、ON のプロンプトに現れた節の中身は
    OFF ラン相当の語彙(= 実在の節)であり、"…" の羅列にはならない(枯渇時を除く)。
    """
    d = tmp_path / "pl_shuf_fmt"
    sim = Simulation(_cfg("pl_shuf_fmt", 144, 20,
                          **{**_JOURNAL, "ablate.context_shuffle": "true"}), out_dir=d)
    sim.run()
    pl = sim._placebo
    assert pl.n_sections_shuffled > 0, "1 つも入れ替わっていない(プラセボとして無効)"
    real = 0
    for prompt in _journal_prompts(d):
        for kind, lines in _sections_of(prompt).items():
            if kind == A.REPLY_KIND:
                continue
            sec = next(s for s in A.SECTIONS if s.kind == kind)
            for line in lines:
                body = sec.match(line)
                assert body, f"中身が空になった: {line}"
                if set(body.split(sec.sep)) != {A.SEVER_TOKEN}:
                    real += 1
    assert real > 0, "全部が枯渇後退(…)になっている=シャッフルが効いていない"


def test_persona_swap_keeps_context_and_replaces_persona(tmp_path):
    """persona_swap: ペルソナ行だけが**他人のもの**になり、文脈節は 1 つも壊れない。"""
    d = tmp_path / "pl_swap_fmt"
    sim = Simulation(_cfg("pl_swap_fmt", 144, 20,
                          **{**_JOURNAL, "ablate.persona_swap": "true"}), out_dir=d)
    sim.run()
    personas = {a.persona for a in sim.agents}
    by_id = {a.id: a.persona for a in sim.agents}
    pl = sim._placebo
    swapped = 0
    for aid, prompt in _journal_rows(d):
        # ヘッダ自体が複数行なので「行番号」ではなく名簿との突き合わせで同定する。
        hits = [ln for ln in prompt.split("\n") if ln in personas]
        assert len(hits) == 1, f"ペルソナ行が 1 本でない({len(hits)} 本)"
        # ★載っているのは**相手のペルソナ**(対合表どおり)= 入れ替えが実際に効いている
        assert hits[0] == by_id[pl.partner_of(aid)], \
            f"agent {aid} のプロンプトに載っているのが対合相手のペルソナでない"
        assert hits[0] != by_id[aid], f"agent {aid} のペルソナが入れ替わっていない"
        swapped += 1
        # 文脈節は "…" に潰れていない(= persona_swap は文脈を触らない)
        for kind, lines in _sections_of(prompt).items():
            if kind == A.REPLY_KIND:
                continue
            sec = next(s for s in A.SECTIONS if s.kind == kind)
            for ln in lines:
                assert set(sec.match(ln).split(sec.sep)) != {A.SEVER_TOKEN}, \
                    f"persona_swap なのに文脈が潰れている: {ln}"
    assert swapped > 0
    assert sim._placebo.n_persona_swapped > 0


def test_section_prefixes_actually_occur_in_build_prompt():
    """SECTIONS の接頭辞が build_prompt の実出力と一字一句一致する(定義のズレ検知)。

    ここがズレると 3 種とも「触るべき節を触らない」静かな無効化になるので機械で固定する。
    """
    class _Mem:
        day_summaries: list = []
        relations: dict = {}

        def recent(self, n):
            return ["出来事A", "出来事B"]

        def retrieve(self, *a, **k):
            return ["記憶X"]

        def query_ex(self, *a, **k):
            class _R:
                hits = ["想起Y"]
                failed = False
                cue = "手掛かり"
            return _R()

        def relation_line(self, ids):
            return None

    class _Agent:
        id = 0
        persona = "私はテスト用の人間です。"
        activity = ""
        states = {"grievance": 0.1, "efficacy": 0.1, "energy": 0.5}
        beliefs: list = []
        adopted = {"語1", "語2"}
        said: list = []
        money = 0
        mem = _Mem()

    prompt = deliberate.build_prompt(
        _Agent(), place_name="どこか", surprise="reply",
        nearby_names=["甲", "乙"], nearby_ids=[1, 2],
        reply_to=("甲", "こんにちは"), feed_texts=["投稿1", "投稿2"],
        dialog_history=[("甲", "やあ"), ("私", "どうも")],
        relation_line="甲とは友人", household_line="恋人の乙",
        pull_query="何か")
    found = _sections_of(prompt)
    for kind in ("known_words", "recent", "retrieved", "pull", "relation",
                 "household", "nearby", "feed", "dialog", A.REPLY_KIND):
        assert kind in found, f"SECTIONS の '{kind}' が build_prompt 出力と噛み合っていない"


# =========================================================================== #
# (4) 決定論・対合性・輪の単体検証
# =========================================================================== #
@pytest.mark.parametrize("mode", MODES)
def test_same_seed_two_runs_match(mode, tmp_path):
    a, _ = _run(tmp_path, f"pl_det1_{mode}", n_steps=72, n_agents=15,
                **{f"ablate.{mode}": "true"})
    b, _ = _run(tmp_path, f"pl_det2_{mode}", n_steps=72, n_agents=15,
                **{f"ablate.{mode}": "true"})
    assert _l1(a) == _l1(b), f"{mode} が非決定(専用 stream 以外を引いている疑い)"


def test_persona_pairing_is_an_involution():
    """対合性: partner(partner(a)) == a。偶数人口では自己写像ゼロ=全単射。"""
    pl = A.Placebo("persona_swap", _FakeHub())
    ids = list(range(20))
    pl.build_pairing(ids)
    assert set(pl.pairing) == set(ids)
    for a in ids:
        b = pl.pairing[a]
        assert pl.pairing[b] == a, f"対合が壊れている: {a}->{b}->{pl.pairing[b]}"
        assert b != a, "偶数人口なのに自分自身へ写っている"
    assert sorted(pl.pairing.values()) == ids, "全単射でない(ペルソナ分布が保存されない)"


def test_persona_pairing_odd_population_has_one_fixed_point():
    pl = A.Placebo("persona_swap", _FakeHub())
    pl.build_pairing(range(7))
    fixed = [a for a, b in pl.pairing.items() if a == b]
    assert len(fixed) == 1, f"奇数人口の自己写像が 1 人でない: {fixed}"
    for a, b in pl.pairing.items():
        assert pl.pairing[b] == a


def test_persona_swap_is_mutual_in_a_run(tmp_path):
    """ラン内でも A↔B が相互交換になっている(対合表と実際の名簿の突き合わせ)。"""
    sim, summary = _run(tmp_path, "pl_swap_inv", n_steps=24, n_agents=16,
                        **{"ablate.persona_swap": "true"})
    pl = sim._placebo
    for a in sim.agents:
        b = pl.partner_of(a.id)
        assert pl.partner_of(b) == a.id
    assert summary["placebo"]["involution"] is True
    assert summary["placebo"]["pairs"] == 8


def test_shuffle_ring_never_returns_own_content():
    """輪は「自分のものでない中身」しか返さない(= 自分の文脈が戻ってこない)。"""
    pl = A.Placebo("context_shuffle", _FakeHub())
    pl._push("recent", "自分の出来事")
    assert pl._draw("recent", "自分の出来事", 0, 0) is None      # 候補が自分だけ=枯渇
    pl._push("recent", "他人の出来事")
    for step in range(10):
        got = pl._draw("recent", "自分の出来事", 0, step)
        assert got == "他人の出来事"


def test_shuffle_ring_is_bounded():
    pl = A.Placebo("context_shuffle", _FakeHub(), cap=4)
    for i in range(30):
        pl._push("recent", f"内容{i}")
    assert len(pl.rings["recent"]) == 4
    assert pl.rings["recent"] == [f"内容{i}" for i in range(26, 30)], "FIFO でない"


def test_sever_preserves_item_count():
    sec = next(s for s in A.SECTIONS if s.kind == "recent")
    assert sec.sever("a / b / c") == "… / … / …"
    assert sec.sever("a") == "…"
    known = next(s for s in A.SECTIONS if s.kind == "known_words")
    assert known.sever("語1、語2") == "…、…"


# =========================================================================== #
# (5) resume == straight
# =========================================================================== #
@pytest.mark.parametrize("mode", CONTEXT_MODES + ("persona_swap",))
def test_resume_matches_straight(mode, tmp_path):
    """mid-day resume が straight と L1/L2/L3 一致(輪の checkpoint 中央管理)。"""
    on = {f"ablate.{mode}": "true"}
    ov = {**on, "observer.checkpoint_every": 24}
    split, total, n = 24, 72, 15
    straight_dir = tmp_path / f"pl_st_{mode}"
    Simulation(_cfg(f"pl_st_{mode}", total, n, **on), out_dir=straight_dir).run()

    d = tmp_path / f"pl_rs_{mode}"
    sim1 = Simulation(_cfg(f"pl_rs_{mode}", split, n, **ov), out_dir=d)
    for step in range(split):
        scheduler.run_step(sim1, step)
    checkpoint.save(sim1, split, d / "checkpoint" / f"ckpt-{split:06d}.pkl.gz")
    sim1.logger.flush_segment()
    sim2 = Simulation(_cfg(f"pl_rs_{mode}", total, n, **ov), out_dir=d)
    sim2.run(resume_from=d)
    for stem in ("l1_events", "l2_metrics", "l3_snapshots"):
        sa = pq.read_table(straight_dir / f"{stem}.parquet").to_pylist()
        sb = pq.read_table(d / f"{stem}.parquet").to_pylist()
        assert sa == sb, f"{stem} 不一致(placebo {mode} の resume)"


def test_checkpoint_round_trip_restores_ring_and_rewires_agents(tmp_path):
    """直接検証: 輪が round-trip で戻り、復元された個体が構築時の Placebo を指す。"""
    ov = {"ablate.context_shuffle": "true", "observer.checkpoint_every": 24}
    d = tmp_path / "pl_ck"
    sim1 = Simulation(_cfg("pl_ck", 24, 15, **ov), out_dir=d)
    for step in range(24):
        scheduler.run_step(sim1, step)
    assert any(sim1._placebo.rings.values()), "輪が空(検証にならない)"
    path = d / "checkpoint" / "ckpt-000024.pkl.gz"
    checkpoint.save(sim1, 24, path)

    sim2 = Simulation(_cfg("pl_ck2", 24, 15, **ov), out_dir=tmp_path / "pl_ck2")
    checkpoint.load(sim2, path)
    assert sim2._placebo.rings == sim1._placebo.rings
    assert all(a.placebo is sim2._placebo for a in sim2.agents), \
        "復元個体が古い Placebo を指したまま(輪が 2 つに割れる)"


def test_off_run_strips_placebo_attribute_on_resume(tmp_path):
    """OFF の checkpoint を OFF で読み直しても placebo 属性が生えない(後方互換)。"""
    d = tmp_path / "pl_ck_off"
    sim1 = Simulation(_cfg("pl_ck_off", 12, 12), out_dir=d)
    for step in range(12):
        scheduler.run_step(sim1, step)
    path = d / "checkpoint" / "ckpt-000012.pkl.gz"
    checkpoint.save(sim1, 12, path)
    sim2 = Simulation(_cfg("pl_ck_off2", 12, 12), out_dir=tmp_path / "pl_ck_off2")
    checkpoint.load(sim2, path)
    assert sim2._placebo is None
    assert not any(hasattr(a, "placebo") for a in sim2.agents)


# =========================================================================== #
# (6) 世界が実際に変わる(= プラセボとして有効)+ 記録
# =========================================================================== #
@pytest.mark.parametrize("mode", MODES)
def test_world_actually_changes(mode, tmp_path):
    """★変わらなければプラセボとして無意味。L1 イベント列が OFF と食い違うことを固定する。"""
    off, _ = _run(tmp_path, f"pl_w_off_{mode}", n_steps=144, n_agents=20)
    on, _ = _run(tmp_path, f"pl_w_on_{mode}", n_steps=144, n_agents=20,
                 **{f"ablate.{mode}": "true"})
    assert _l1(on) != _l1(off), (f"{mode}: 世界が 1 バイトも変わっていない。"
                                 "プラセボが実際には効いていない疑いがある")


@pytest.mark.parametrize("mode", MODES)
def test_manifest_and_summary_record_the_damage(mode, tmp_path):
    """manifest / summary に「何をどれだけ壊したか」が残る(risk=known の説明責任)。"""
    _sim, summary = _run(tmp_path, f"pl_rec_{mode}", n_steps=72, n_agents=15,
                         **{f"ablate.{mode}": "true"})
    man = json.loads((tmp_path / f"pl_rec_{mode}" / "run_manifest.json")
                     .read_text(encoding="utf-8"))
    assert man["ablate"][mode] is True
    assert man["ablate"]["placebo"]["mode"] == mode
    assert man["ablate"]["placebo"]["fingerprint_risk"] == "known"
    prov = summary["placebo"]
    assert prov["mode"] == mode and prov["level"] == "L1" and prov["prompts"] > 0
    damage = (prov.get("persona_swapped", 0) + prov.get("sections_shuffled", 0)
              + prov.get("sections_severed", 0))
    assert damage > 0, "1 つも壊していない=プラセボとして無効なランになっている"


@pytest.mark.parametrize("mode", MODES)
def test_registry_declares_all_three(mode):
    from society.registry import BY_ID
    feat = BY_ID.get(f"ablate.{mode}")
    assert feat is not None, f"ablate.{mode} が registry に未宣言"
    assert feat.repro_tier == "strict"
    assert feat.affects_k is False, "プラセボは呼び出し点を変えない(affects_k=False)"
    assert feat.fingerprint_risk == "known", "risk を正直に known で宣言していない"
