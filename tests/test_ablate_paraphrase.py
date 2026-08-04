"""第92バッチ SV-U1 B4: プロンプト言い換え ablate(ablate.prompt_paraphrase = サーベイ S-16)。

正典: docs/research/sv-items-research.md §3 / docs/research/llm-social-sim-survey.md §3 S-16 /
      docs/plans/stationarity-preregistration.md 前文 ④(b)

受入基準
  ① 既定 OFF … 純粋既定と L1 バイト一致・**golden 一致**・個体に paraphrase 属性が生えない・
     manifest / summary にキーが増えない・draw 数も不変(乱数を 1 本も引かない)。
  ② ON … 同 seed 2 ランが一致(決定論)。
  ③ **プロンプト盲 LLM の下で呼数完全一致 + L1 バイト一致**
     = 影響経路は「LLM の応答」ただ 1 本で、呼び出しサイト・乱数・観測経路に副作用が無い。
  ④ プラセボ 3 種 / propagation_off との併用は ValueError(llm_off は WARNING 素通し)。
  ⑤ 凍結表の機械検査: **SECTIONS 接頭辞と JSON キーを壊していない**
     - 置換対のどちら側にも JSON キー・行動語彙(動詞名)が現れない。
     - 置換元は全て build_prompt の実出力に現れる(死んだ対が無い)。
     - 言い換えられた SECTIONS 接頭辞も「ラベル + ': '」の形を保つ(節の同定可能性)。
     - 実プロンプトに対して JSON 書式ブロックが**バイト一致**で残る(パース互換)。
  ⑥ 未知セット名は build_cfg で ValueError。

検証は mock のみ(実 LLM 禁止=ユーザー記憶 validation-runs-short)。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from society import ablate as A
from society.cognition import deliberate
from society.config import load_config
from society.engine.simulation import Simulation

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN = REPO_ROOT / "tests" / "data" / "golden_baseline_l1.json"

# test_placebo.py:45 と同じ「意図的な既定挙動追加」の中立化(ゴールデン比較用)
_GOLDEN_NEUTRAL = {"economy.wages.自営": 0, "planning.enabled": "false",
                   "transit_ride.taxi.enabled": "false",
                   "transit_ride.bus.enabled": "false", "rules.enabled": "false"}

SETS = ("v1", "v2", "v3", "v4")
PLACEBOS = ("context_shuffle", "persona_swap", "context_sever")


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
    """全 stream の draw を数えるプロキシ(test_placebo と同型)。"""

    def __init__(self, inner):
        self._inner = inner
        self.per_stream: dict = {}

    def stream(self, *key):
        return _CountingGen(self._inner.stream(*key), self,
                            str(key[0]) if key else "")

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

    これを噛ませると「言い換えがプロンプトを書き換えた」効果だけが**完全に消える**。
    ON/OFF が L1 バイト一致・呼数完全一致になることが、
      「言い換えの影響経路は *LLM の応答* ただ 1 本であり、
        呼び出しサイト・乱数消費・観測経路には 1 バイトも副作用が無い」
    ことの直接証明になる(test_placebo.py の同型テストの流用)。
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


# --------------------------------------------------------------------------- #
# build_prompt の実出力(全 surprise 分岐)。⑤の機械検査の突き合わせ先。
# --------------------------------------------------------------------------- #
class _Mem:
    day_summaries = ["昨日はよく歩いた"]
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
    activity = "working"
    states = {"grievance": 0.1, "efficacy": 0.1, "energy": 0.5}
    beliefs = ["世界は変わる"]
    adopted = {"語1", "語2"}
    said = ["さっきの一言"]
    money = 100000
    self_model = {"self": "よく歩く人", "ties": "甲"}
    implicit_self = "少し疲れている"
    mem = _Mem()


def _render(surprise: str, agent=None) -> str:
    return deliberate.build_prompt(
        agent or _Agent(), place_name="どこか", surprise=surprise,
        nearby_names=["甲", "乙"], nearby_ids=[1, 2], nearby_pois=["店A", "店B"],
        reply_to=("甲", "こんにちは"), dm_target="乙",
        feed_texts=["投稿1", "投稿2"],
        dialog_history=[("甲", "やあ"), ("私", "どうも")],
        relation_line="甲とは友人", household_line="恋人の乙",
        pull_query="何か", familiar_places=["場A"], institutions=["取り決め1"],
        memberships=["会1"], equip_all=True, city_name="渋谷")


_SURPRISES = ("social", "reply", "post", "dm", "solo", "congestion")


def _all_prompts() -> dict[str, str]:
    return {s: _render(s) for s in _SURPRISES}


# =========================================================================== #
# (0) 設定の正準化と相互排他(④⑥)
# =========================================================================== #
def test_default_is_off():
    assert A.build_cfg(None)["prompt_paraphrase"] == ""
    assert A.build_cfg({})["prompt_paraphrase"] == ""
    assert A.prompt_paraphrase(object()) == ""


def test_shipped_config_has_key_and_is_off():
    raw = load_config().get("ablate")
    assert "prompt_paraphrase" in raw, "conf/config.yaml に ablate.prompt_paraphrase が無い"
    assert A.build_cfg(raw) == A.build_cfg(None)


@pytest.mark.parametrize("name", SETS)
def test_known_sets_are_accepted(name):
    assert A.build_cfg({"prompt_paraphrase": name})["prompt_paraphrase"] == name


@pytest.mark.parametrize("bad", ["v0", "v5", "lexical", "true", "  v9 "])
def test_unknown_set_name_raises(bad):
    """⑥ 未知セット名は**構築時に**落とす(黙って OFF へ後退しない)。"""
    with pytest.raises(ValueError, match="prompt_paraphrase"):
        A.build_cfg({"prompt_paraphrase": bad})


@pytest.mark.parametrize("placebo", PLACEBOS)
def test_paraphrase_with_placebo_is_rejected(placebo):
    """④ プラセボは接頭辞リテラル一致で節を同定する。言い換えると同定できない。"""
    with pytest.raises(ValueError, match="併用できない"):
        A.build_cfg({"prompt_paraphrase": "v1", placebo: True})


def test_paraphrase_with_propagation_off_is_rejected():
    with pytest.raises(ValueError, match="併用できない"):
        A.build_cfg({"prompt_paraphrase": "v1", "propagation_off": True})


def test_paraphrase_with_llm_off_warns_but_passes(caplog):
    """llm_off はプロンプトを 1 つも組まない=言い換えが無効。落とさず WARNING で告知。"""
    with caplog.at_level("WARNING"):
        cfg = A.build_cfg({"prompt_paraphrase": "v2", "llm_off": True})
    assert cfg["prompt_paraphrase"] == "v2" and cfg["llm_off"]
    assert any("1 バイトも効かない" in r.getMessage() for r in caplog.records)


# =========================================================================== #
# (1) ⑤ 凍結表の機械検査(SECTIONS 接頭辞 / JSON キーを壊していない)
# =========================================================================== #
def test_frozen_table_schema_and_sets():
    doc, digest = A.paraphrase_doc()
    assert doc["schema"] == A.PARAPHRASE_SCHEMA
    assert set(doc["sets"]) == set(SETS), "凍結表のセット名が v1..v4 でない"
    assert len(digest) == 64
    for name in SETS:
        spec = doc["sets"][name]
        assert spec["pairs"], f"{name}: 置換対が空"
        srcs = [s for s, _ in spec["pairs"]]
        assert len(set(srcs)) == len(srcs), f"{name}: 置換元が重複(適用順で結果が変わる)"
        for s, d in spec["pairs"]:
            assert s and d and s != d, f"{name}: 退化した対 {s!r}->{d!r}"


def test_frozen_table_never_touches_json_keys_or_action_verbs():
    """⑤-a パース互換: JSON キーと行動語彙(動詞名)は対のどちら側にも現れない。"""
    doc, _ = A.paraphrase_doc()
    protected = (doc["protected"]["json_keys"] + doc["protected"]["action_verbs"])
    assert protected
    for name in SETS:
        for s, d in doc["sets"][name]["pairs"]:
            for tok in protected:
                assert tok not in s and tok not in d, \
                    f"{name}: 保護語 {tok!r} が置換対に含まれる: {s!r} -> {d!r}"


def test_every_replacement_source_occurs_in_real_prompts():
    """⑤-b 死んだ対が無い: 全ての置換元が build_prompt の実出力に現れる。"""
    joined = "\n".join(_all_prompts().values())
    doc, _ = A.paraphrase_doc()
    for name in SETS:
        for s, _d in doc["sets"][name]["pairs"]:
            assert s in joined, f"{name}: 置換元が build_prompt 出力に現れない: {s!r}"


def test_section_prefixes_keep_label_colon_shape():
    """⑤-c 節の同定可能性: SECTIONS 接頭辞を言い換えても「ラベル + ': '」の形を保つ。

    ★ここが崩れると「接頭辞 + 中身」という節の構造そのものが壊れ、事後解析(節ごとの
      抽出)が静かに死ぬ。**接頭辞を言い換えること自体は許す**(だからプラセボと相互排他)。
    """
    doc, _ = A.paraphrase_doc()
    prefixes = {sec.prefix for sec in A.SECTIONS}
    covered = set()
    for name in SETS:
        for s, d in doc["sets"][name]["pairs"]:
            if s in prefixes:
                covered.add(s)
                assert d.endswith(": "), f"{name}: 節の接頭辞が ': ' で終わっていない: {d!r}"
                assert len(d) > 2, f"{name}: 節のラベルが空になっている: {d!r}"
    assert covered == prefixes, \
        f"言い換え表が触っていない SECTIONS 接頭辞がある: {sorted(prefixes - covered)}"


@pytest.mark.parametrize("name", SETS)
def test_json_format_block_is_byte_identical_after_paraphrase(name):
    """⑤-d 実プロンプトに掛けても JSON 書式ブロックが**バイト一致**で残る。"""
    doc, digest = A.paraphrase_doc()
    para = A.Paraphrase(name, doc["sets"][name], digest)
    for surprise, prompt in _all_prompts().items():
        lines = prompt.split("\n")
        out = para.apply(lines)
        assert len(out) == len(lines), f"{name}/{surprise}: 行数が変わった"
        for before, after in zip(lines, out):
            if '"action"' in before:
                assert after == before, \
                    f"{name}/{surprise}: JSON 書式行が書き換わった: {before!r} -> {after!r}"
        assert "".join(out) != "".join(lines), \
            f"{name}/{surprise}: 1 文字も言い換わっていない(検査になっていない)"


def test_replacement_never_chains():
    """置換は**単一パス同時置換**= 連鎖(A→B したあと B→C)が起きない。

    str.replace を順に掛ける実装だと、凍結表のどこにも書いていない文字列("C")が生まれる。
    正規表現の交替で 1 回走査すれば置換元は必ず**原文**の側だけを見る。
    """
    para = A.Paraphrase("t", {"level": "test", "pairs": [["A", "B"], ["B", "C"]]}, "x" * 64)
    assert para.apply(["A"]) == ["B"], "連鎖している(単一パスでない)"
    assert para.apply(["B"]) == ["C"]
    assert para.apply(["AB"]) == ["BC"]


def test_longest_source_wins():
    """重なる置換元は**最長一致優先**(でないと部分置換で意味が壊れる)。"""
    para = A.Paraphrase("t", {"level": "test",
                              "pairs": [["場所: ", "地点: "],
                                        ["周りにある店・場所: ", "周辺の店・場所: "]]},
                        "x" * 64)
    assert para.apply(["周りにある店・場所: 店A"]) == ["周辺の店・場所: 店A"]
    assert para.apply(["場所: どこか"]) == ["地点: どこか"]


def test_real_table_resolves_overlapping_prefixes_longest_first():
    """実表・実プロンプトでの確認: 「周りにある店・場所: 」が「場所: 」に食われない。"""
    doc, digest = A.paraphrase_doc()
    para = A.Paraphrase("v1", doc["sets"]["v1"], digest)
    out = para.apply(_render("solo").split("\n"))
    assert any(ln.startswith("周辺の店・場所: ") for ln in out), \
        "重なる接頭辞が最長一致で解決されていない"
    assert not any(ln.startswith("周りにある店・") for ln in out)


def test_apply_is_noop_for_lines_without_sources():
    doc, digest = A.paraphrase_doc()
    para = A.Paraphrase("v1", doc["sets"]["v1"], digest)
    lines = ["まったく無関係な行", "another unrelated line"]
    assert para.apply(lines) == lines
    assert para.n_lines_changed == 0


# =========================================================================== #
# (2) ① 既定 OFF の不変条件(R1)
# =========================================================================== #
def test_off_matches_pure_default(tmp_path):
    a, _ = _run(tmp_path, "pp_base")
    b, _ = _run(tmp_path, "pp_off", **{"ablate.prompt_paraphrase": '""'})
    assert _l1(a) == _l1(b)
    assert b._paraphrase is None
    assert not any(hasattr(ag, "paraphrase") for ag in b.agents), \
        "OFF なのに個体に paraphrase 属性が生えている"


def test_off_matches_golden(tmp_path):
    """明示 OFF が変更前ゴールデンと一字一句一致(ゴールデンは再生成しない=掟2)。"""
    golden = json.load(open(GOLDEN, encoding="utf-8"))
    ov = {**_GOLDEN_NEUTRAL, "ablate.prompt_paraphrase": '""'}
    sim = Simulation(_cfg("pp_golden", 144, 15, **ov), out_dir=tmp_path / "pp_golden")
    sim.run()
    assert _l1(sim) == golden, "言い換えの seam が no-op でない(ゴールデン不一致)"


def test_off_draw_counts_identical(tmp_path):
    """OFF は純粋既定と draw 数(stream 別)が完全一致=新 stream ゼロ。

    ON でも同じであるべき(表引きは決定論で乱数を 1 本も引かない)ことを併せて固定する。
    """
    def _draws(name, **ov):
        sim = Simulation(_cfg(name, 24, 12, **ov), out_dir=tmp_path / name)
        sim.hub = _CountingHub(sim.hub)
        sim.run()
        return sim.hub.per_stream

    pure = _draws("pp_dpure")
    off = _draws("pp_doff", **{"ablate.prompt_paraphrase": '""'})
    assert pure == off and sum(pure.values()) > 0
    assert not any("paraphrase" in k for k in pure), "言い換え用の stream が生えている"


def test_off_adds_no_manifest_or_summary_key(tmp_path):
    sim, summary = _run(tmp_path, "pp_man_off", n_steps=6)
    man = json.loads((tmp_path / "pp_man_off" / "run_manifest.json")
                     .read_text(encoding="utf-8"))
    assert "ablate" not in man
    assert "prompt_paraphrase" not in summary
    assert A.describe(sim) is None


# =========================================================================== #
# (3) ② 決定論 / ③ 呼数完全一致
# =========================================================================== #
@pytest.mark.parametrize("name", SETS)
def test_same_seed_two_runs_match(name, tmp_path):
    """② 同 seed 2 ランが L1 バイト一致(表引きは決定論)。"""
    a, _ = _run(tmp_path, f"pp_det1_{name}", n_steps=72, n_agents=15,
                **{"ablate.prompt_paraphrase": name})
    b, _ = _run(tmp_path, f"pp_det2_{name}", n_steps=72, n_agents=15,
                **{"ablate.prompt_paraphrase": name})
    assert _l1(a) == _l1(b), f"{name} が非決定"


def _blind_run(tmp_path, name, **ov):
    sim = Simulation(_cfg(name, 144, 20, **ov), out_dir=tmp_path / name)
    sim.llm = _PromptBlindLLM(sim.llm)
    return sim, sim.run()


@pytest.mark.parametrize("name", SETS)
def test_call_count_is_exactly_equal_under_prompt_blind_llm(name, tmp_path):
    """③ ★本命。プロンプト内容を捨てる LLM の下で ON==OFF(呼数完全一致 + L1 バイト一致)。

    「呼び出しサイトを 1 つも足さない/減らさない」(affects_k=False の構造的根拠)を
    **文字どおり**固定する唯一のテスト。実ランで呼数が動くのは応答が変わるからであって、
    言い換えが呼び出しサイト・乱数・観測経路に触ったからではない、を切り分ける。
    """
    off, s_off = _blind_run(tmp_path, f"pp_blind_off_{name}")
    on, s_on = _blind_run(tmp_path, f"pp_blind_on_{name}",
                          **{"ablate.prompt_paraphrase": name})
    assert s_off["llm_calls"] > 0
    assert s_on["llm_calls"] == s_off["llm_calls"], \
        f"{name}: プロンプト盲下でも呼数が違う=呼び出しサイトが増減している"
    assert _l1(on) == _l1(off), \
        f"{name}: プロンプト盲下でも L1 が違う=応答以外の副作用がある"


# =========================================================================== #
# (4) ON: 実際に効いていること + 記録
# =========================================================================== #
@pytest.mark.parametrize("name", SETS)
def test_prompts_are_actually_paraphrased(name, tmp_path):
    sim, _ = _run(tmp_path, f"pp_eff_{name}", n_steps=72, n_agents=15,
                  **{"ablate.prompt_paraphrase": name})
    pa = sim._paraphrase
    assert pa is not None and pa.mode == name
    assert pa.n_prompts > 0, "プロンプトが 1 つも組まれていない(検査になっていない)"
    assert pa.n_subs > 0, f"{name}: 1 文字も言い換えていない=検査として無効"
    assert all(getattr(ag, "paraphrase", None) is pa for ag in sim.agents)


@pytest.mark.parametrize("name", SETS)
def test_manifest_records_mode_and_sha256(name, tmp_path):
    """manifest は**ラン開始時**に書かれるので、載るのは mode と凍結表 sha256(静的来歴)。"""
    _sim, summary = _run(tmp_path, f"pp_man_{name}", n_steps=24, n_agents=12,
                         **{"ablate.prompt_paraphrase": name})
    man = json.loads((tmp_path / f"pp_man_{name}" / "run_manifest.json")
                     .read_text(encoding="utf-8"))
    ab = man["ablate"]
    assert ab["prompt_paraphrase"] == name
    prov = ab["paraphrase"]
    assert prov["mode"] == name
    assert prov["sets_sha256"] == A.paraphrase_digest() and len(prov["sets_sha256"]) == 64
    assert prov["file"] == A.PARAPHRASE_REL
    assert prov["fingerprint_risk"] == "known"
    assert prov["pairs"] > 0
    # 「実際にどれだけ言い換えたか」は走り切ってからでないと判らないので summary 側に残す
    # (プラセボの summary["placebo"] と同じ流儀。0 件のランは検査として無効と事後に判る)。
    run_prov = summary["prompt_paraphrase"]
    assert run_prov["mode"] == name
    assert run_prov["prompts"] > 0 and run_prov["substitutions"] > 0
    assert run_prov["sets_sha256"] == prov["sets_sha256"]


def test_world_actually_changes(tmp_path):
    """★変わらなければ検査として無意味。L1 が OFF と食い違うことを固定する。"""
    off, _ = _run(tmp_path, "pp_w_off", n_steps=144, n_agents=20)
    on, _ = _run(tmp_path, "pp_w_on", n_steps=144, n_agents=20,
                 **{"ablate.prompt_paraphrase": "v3"})
    assert _l1(on) != _l1(off), \
        "言い換えても世界が 1 バイトも変わっていない(効いていない疑い)"


def test_registry_declares_prompt_paraphrase():
    from society.registry import BY_ID
    feat = BY_ID.get("ablate.prompt_paraphrase")
    assert feat is not None, "ablate.prompt_paraphrase が registry に未宣言"
    assert feat.repro_tier == "strict"
    assert feat.affects_k is False, "呼び出し点は変えない(プラセボと同じ affects_k=False)"
    assert feat.fingerprint_risk == "known", "risk を正直に known で宣言していない"
    assert feat.off_value == "", "自動 OFF の書き込み値が空文字でない"
