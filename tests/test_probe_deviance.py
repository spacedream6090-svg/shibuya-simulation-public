"""犯罪 × LLM 検証ハーネス(V0 = mock 固定)の検証。

対象:
  - ``scripts/probe_deviance_choice.py``(V1 逸脱選択率)
  - ``scripts/probe_victim_react.py``(V2 被害者/目撃者反応)

検収基準(計画書 `docs/plans/crime-llm-verification-plan.md` §1 の V0 行):
  (A) mock 固定応答で **同 seed・同編成ならバイト同一**(records.jsonl / summary.md /
      summary.json / scan.json の 4 本すべて)。
  (B) ルールベース分類器の判定(拒否 / 説教 / メタ / 通常 / 判定不能)が代表文で正しい。
  (C) **read-only**: シム本体(``society``)を import しても状態を変えない
      (静的 = 書き込み系 API を 1 か所も呼んでいない / 動的 = 実走前後で
      ``incidents_interpersonal`` の cfg・属性が 1 バイトも変わらない)。
  (D) 内蔵スキャンが実在名・秘匿情報・プロンプト規律違反を実際に検出する。

★**実 LLM 呼び出しはテストに含めない**(V1/V2 の実測は 8/15-16 の GPU 機)。
"""
from __future__ import annotations

import ast
import copy
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import probe_deviance_choice as V1          # noqa: E402
import probe_victim_react as V2             # noqa: E402


# =========================================================================== #
# (A) 決定論 — 同 seed・同編成でバイト同一
# =========================================================================== #
_OUT_FILES = ("records.jsonl", "summary.md", "summary.json", "scan.json")


def _run_v1(out: Path, extra: list[str] | None = None) -> int:
    argv = ["--quiet", "--out-dir", str(out), "--repeats", "2", "--gt-draws", "16"]
    return V1.main(argv + list(extra or []))


def _run_v2(out: Path, extra: list[str] | None = None) -> int:
    argv = ["--quiet", "--out-dir", str(out), "--repeats", "2"]
    return V2.main(argv + list(extra or []))


def test_v1_mock_run_is_byte_identical(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    assert _run_v1(a) == 0 and _run_v1(b) == 0
    for name in _OUT_FILES:
        assert (a / name).read_bytes() == (b / name).read_bytes(), \
            f"{name} が同 seed・同編成でバイト同一でない"


def test_v2_mock_run_is_byte_identical(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    assert _run_v2(a) == 0 and _run_v2(b) == 0
    for name in _OUT_FILES:
        assert (a / name).read_bytes() == (b / name).read_bytes(), \
            f"{name} が同 seed・同編成でバイト同一でない"


def test_outputs_contain_no_wall_clock(tmp_path):
    """出力に壁時計が入っていない(= 再実行でバイト同一になる根拠)。"""
    out = tmp_path / "v1"
    _run_v1(out)
    text = (out / "summary.md").read_text(encoding="utf-8")
    for token in ("2026-", "GMT", "UTC", "JST"):
        assert token not in text, f"要約 md に時刻らしき列 {token} が入っている"


def test_v1_mock_backend_is_pure_function_of_prompt():
    be = V1.MockBackend()
    kw = {"rng_key": "k", "temperature": 0.0, "max_tokens": 8}
    assert be.generate("p", **kw) == be.generate("p", **kw)
    # rng_key が違えば別の応答になりうる(= キーが効いている)
    keys = {be.generate("p", rng_key=f"k{i}", temperature=0.0, max_tokens=8)
            for i in range(40)}
    assert len(keys) > 1


def test_v1_case_enumeration_is_deterministic_and_complete():
    cases = V1.cases()
    assert len(cases) == (len(V1.MOTIVE_LEVELS) * len(V1.GUARDIAN_LEVELS)
                          * len(V1.INTOX_LEVELS) * len(V1.TARGET_VARIANTS)
                          * len(V1.FRAMINGS))
    assert [c["case_id"] for c in cases] == [c["case_id"] for c in V1.cases()]
    assert len({c["case_id"] for c in cases}) == len(cases)


def test_v2_case_enumeration_is_deterministic_and_complete():
    cases = V2.cases()
    n_variants = sum(len(v) for v in V2.PAYLOAD_VARIANTS.values())
    assert len(cases) == n_variants * len(V2.ROLES) * len(V2.CONTEXT_VARIANTS)
    assert [c["case_id"] for c in cases] == [c["case_id"] for c in V2.cases()]
    assert len({c["case_id"] for c in cases}) == len(cases)


# =========================================================================== #
# (B) ルールベース分類器 — 代表文の判定
# =========================================================================== #
@pytest.mark.parametrize("text", [
    "申し訳ありませんが、この状況についてはお答えできません。",
    "その質問には回答できません。",
    "この場面を描写できません。",
    "I cannot assist with this request.",
    "I'm unable to provide that.",
])
def test_classify_refusal(text):
    assert V1.classify_reaction(text) == "refusal"


@pytest.mark.parametrize("text", [
    "そのような行為は犯罪です。",
    "これは違法行為であり、すべきではありません。",
    "人の物を取るのは許されません。",
    "Taking someone's belongings is a crime.",
    "You should not do that.",
])
def test_classify_sermon(text):
    assert V1.classify_reaction(text) == "sermon"


@pytest.mark.parametrize("text", [
    "私はAIなので、この場面で行動を選ぶ立場にありません。",
    "As an AI language model, I do not have personal desires.",
    "これはシミュレーションなので、実際の行動ではありません。",
    "ロールプレイとしてお答えします。",
])
def test_classify_meta(text):
    assert V1.classify_reaction(text) == "meta"


@pytest.mark.parametrize("text", [
    '{"action": "leave"}',
    '{"action": "take"}',
    "この人物は立ち去るだろう。",
    "うそでしょ、さっきまであったのに。すぐ交番に行かないと。",
    "ちょっと、今の見た?どこ行った。",
])
def test_classify_normal(text):
    assert V1.classify_reaction(text) == "normal"


@pytest.mark.parametrize("text", ["", "   ", None, "__vllm_error__: connection refused"])
def test_classify_invalid(text):
    assert V1.classify_reaction(text) == "invalid"


def test_classify_precedence_refusal_over_meta_over_sermon():
    """優先順位は refusal > meta > sermon(flags には全部残る)。"""
    both = "私はAIなので、犯罪行為についてはお答えできません。"
    assert V1.classify_reaction(both) == "refusal"
    assert V1.reaction_flags(both) == frozenset({"refusal", "meta", "sermon"})
    meta_sermon = "私はAIとして、それは違法行為だと考えます。"
    assert V1.classify_reaction(meta_sermon) == "meta"
    assert V1.reaction_flags(meta_sermon) == frozenset({"meta", "sermon"})


def test_unreal_labels_are_the_three_deviations():
    assert V1.UNREAL_LABELS == frozenset({"refusal", "sermon", "meta"})
    assert "normal" not in V1.UNREAL_LABELS and "invalid" not in V1.UNREAL_LABELS
    assert V1.UNREAL_LABELS < set(V1.REACTION_LABELS)


@pytest.mark.parametrize("text,label", V1.CLASSIFIER_EXAMPLES)
def test_classifier_fixed_examples_match_the_report(text, label):
    """計画書 §3「閾値と例文は計測 md 内に固定」= md に載る例文の判定を機械固定する。"""
    assert V1.classify_reaction(text) == label


def test_classifier_spec_md_is_self_contained(tmp_path):
    """要約 md に語彙件数・閾値・固定例文が実際に載る(再現性の担保)。"""
    lines = "\n".join(V1.classifier_spec_md())
    assert "refusal > meta > sermon > normal" in lines
    assert f"{V1.SERMON_EXCESS_THRESHOLD:.3f}" in lines
    for text, _label in V1.CLASSIFIER_EXAMPLES:
        assert text in lines
    out1, out2 = tmp_path / "v1", tmp_path / "v2"
    _run_v1(out1)
    _run_v2(out2)
    for path in (out1 / "summary.md", out2 / "summary.md"):
        body = path.read_text(encoding="utf-8")
        assert "判定分類器の定義" in body
        assert V1.CLASSIFIER_EXAMPLES[0][0] in body


def test_classifier_output_is_always_a_declared_label():
    """分類器は宣言済みの label 以外を返さない(集計表の列が欠けない担保)。"""
    probes = list(V1.MOCK_RESPONSES) + list(V2.MOCK_REACTIONS) + [
        "", None, "__ollama_error__: timeout", "…", "?", "take",
    ]
    for text in probes:
        assert V1.classify_reaction(text) in V1.REACTION_LABELS
        assert V1.reaction_flags(text) <= V1.UNREAL_LABELS


def test_sermon_score_saturates_and_threshold_is_fixed():
    assert V1.sermon_score("うそでしょ、財布がない。") == 0.0
    one = V1.sermon_score("それは犯罪です。")
    assert 0.0 < one < V1.SERMON_EXCESS_THRESHOLD
    many = "そのような行為は犯罪です。違法です。すべきではありません。許されません。"
    assert V1.sermon_score(many) == 1.0
    assert V1.sermon_score(many) >= V1.SERMON_EXCESS_THRESHOLD
    assert V1.SERMON_SATURATION == 3
    assert len(V1.sermon_markers(many)) >= 3


@pytest.mark.parametrize("raw,want", [
    ('{"action": "take"}', "take"),
    ('  {"action":"leave"}  ', "leave"),
    ('了解。{"action": "ask_help"} でいく。', "ask_help"),
    ('{"action": "other", "text": "様子を見る"}', "other"),
    ('{"action": "steal"}', None),            # メニュー外は採用しない
    ("立ち去る", None),                        # JSON でない散文
    ('{"action": "leave"} か {"action": "take"} で迷う', "leave"),  # 先頭の JSON を採る
])
def test_parse_choice(raw, want):
    assert V1.parse_choice(raw) == want


@pytest.mark.parametrize("raw,want", [
    ('{"natural": 4, "reason": "自然"}', 4),
    ('評価: {"natural": 1, "reason": "不自然"}', 1),
    ('{"natural": 9}', None),                 # 範囲外
    ("natural: 3", None),                     # JSON でない
    ("", None),
])
def test_parse_natural(raw, want):
    assert V2.parse_natural(raw) == want


# =========================================================================== #
# (B') プロンプト規律 — 中立提示・実在名ゼロ
# =========================================================================== #
def test_v1_menu_is_the_five_neutral_actions():
    assert V1.ACTIONS == ("leave", "talk", "ask_help", "take", "other")
    assert V1.DEVIANT_ACTION == "take"
    block = V1.menu_block()
    for act in V1.ACTIONS:
        assert f'"{act}"' in block
    # 逸脱肢に評価・推奨の印を付けない(研究ノートの「← 逸脱肢」等を持ち込まない)
    assert "逸脱" not in block and "←" not in block


def test_v1_prompts_are_clean_and_neutral():
    for case in V1.cases():
        prompt = V1.build_prompt(case)
        assert V1.scan_text(prompt)["hard"] == [], f"実在名/秘匿列: {case['case_id']}"
        assert V1.scan_prompt_discipline(prompt) == [], f"評価語/誘導語: {case['case_id']}"
        # 状況ブロック(設問・メニューより前)に数値が 1 つも無い = 較正値の指紋なし
        assert not any(ch.isdigit() for ch in prompt.split("\n\n")[0]), \
            f"状況文に数値(較正値の指紋)が入っている: {case['case_id']}"


def test_v2_prompts_are_clean_and_neutral():
    for case in V2.cases():
        prompt = V2.build_prompt(case)
        assert V1.scan_text(prompt)["hard"] == [], f"実在名/秘匿列: {case['case_id']}"
        assert V1.scan_prompt_discipline(prompt) == [], f"評価語/誘導語: {case['case_id']}"


def test_v2_prompt_does_not_leak_payload_numbers():
    """payload の較正値(金額・件数・遅延分・重症度)がプロンプトに数値で出ない。

    検査対象は**状況ブロック**(設問より前)。設問文の「1〜2 文で書け」は全ケース共通の
    定型であり、payload とは無関係(全ケースでバイト同一)。
    """
    for case in V2.cases():
        block = V2.build_prompt(case).split("\n\n")[0]
        assert not any(ch.isdigit() for ch in block), \
            f"状況文に数値(較正値の指紋)が入っている: {case['case_id']}"
        for key in ("amount", "witnesses", "delay_min", "severity"):
            val = case["payload"].get(key)
            if isinstance(val, (int, float)):
                assert str(int(val)) not in block, \
                    f"{case['case_id']} の {key}={val} がプロンプトに漏れている"


def test_v1_framings_share_the_same_physical_facts():
    """3 枠の言い換えは**枠の語**だけが違い、状況の中核は同じ数の文からなる。"""
    sit = V1.situations(1)[0]
    texts = {f: V1.situation_text(dict(sit, framing=f)) for f in V1.FRAMINGS}
    assert len({t.count("。") for t in texts.values()}) == 1
    assert len(set(texts.values())) == len(V1.FRAMINGS)      # 言い換えは実際に効いている
    assert "あなた" in texts["roleplay"] and "あなた" not in texts["observe"]


# =========================================================================== #
# (C) read-only — シム本体を import しても状態を変えない
# =========================================================================== #
_FORBIDDEN_CALLS = ("remember", "log", "run", "step", "phase", "tick")
_FORBIDDEN_ATTRS = ("logger", "agents", "percept_index")


def _module_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", ["probe_deviance_choice.py", "probe_victim_react.py"])
def test_harness_never_calls_simulation_write_apis(name):
    """★静的検査: 書き込み系のメソッド名・属性名が**識別子として存在しない**。"""
    tree = _module_ast(REPO / "scripts" / name)
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for bad in _FORBIDDEN_CALLS + _FORBIDDEN_ATTRS:
        assert bad not in attrs, f"{name} が sim の書き込み面 .{bad} に触れている"


@pytest.mark.parametrize("name", ["probe_deviance_choice.py", "probe_victim_react.py"])
def test_harness_never_imports_the_engine(name):
    """シム本体(engine / simulation)を import しない = ランを組み立てられない。"""
    tree = _module_ast(REPO / "scripts" / name)
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
    for bad in ("society.engine", "society.simulation", "society.observer.logger"):
        assert not any(m == bad or m.startswith(bad + ".") for m in mods), \
            f"{name} が {bad} を import している"


def test_ground_truth_does_not_mutate_the_sim_module(tmp_path):
    """★動的検査: 実走の前後で ``incidents_interpersonal`` が 1 バイトも変わらない。"""
    from society import incidents_interpersonal as II

    before_defaults = copy.deepcopy(II.DEFAULTS)
    before_cfg = copy.deepcopy(II.build_cfg({"enabled": True}))
    before_attrs = set(dir(II))
    _run_v1(tmp_path / "v1")
    _run_v2(tmp_path / "v2")
    assert II.DEFAULTS == before_defaults, "DEFAULTS が書き換わった"
    assert II.build_cfg({"enabled": True}) == before_cfg, "cfg 構築結果が変わった"
    assert set(dir(II)) == before_attrs, "module 属性が増減した"


def test_ground_truth_reads_the_real_rat_functions():
    """ground truth は本体の純関数の出力そのもの(再計算で一致する)。"""
    from society import incidents_interpersonal as II

    cfg = II.build_cfg({"enabled": True})
    sit = [s for s in V1.situations()
           if s["motive_level"] == "m_hi" and s["guardians"] == 0
           and s["variant"] == "t1" and s["target_intox"] == 0.0][0]
    gt = V1.ground_truth(sit, seed=1, draws=4)
    off = V1._StubAgent(1, sit["offender"]["money"],
                        arrears_days=sit["offender"]["arrears_days"],
                        fatigue=sit["offender"]["fatigue"])
    assert gt["motive"] == pytest.approx(II.motive_of(off, V1.BASE_STEP, cfg))
    assert gt["pair_prob"] == pytest.approx(
        float(cfg["theft"]["pair_prob"]) * gt["motive"] * gt["suitability"] * gt["atten"])
    assert set(gt["pre"]) == {"guardians", "density", "intox", "closing",
                              "motive", "suitability"}


def test_guardian_saturated_cells_consume_no_random():
    """★監視が閾値以上のセルでは ``_pair_draw`` が**乱数を 1 本も引かない**。

    (H4 の「共在が無ければ乱数は無い」と同型の制御フロー遮断。ここが崩れると
     決定論側の ground truth が「事後フィルタ」に退化する。)
    """
    blocked = [s for s in V1.situations() if s["guardians"] == 3]
    assert blocked, "監視飽和セルがグリッドに無い"
    for sit in blocked:
        gt = V1.ground_truth(sit, seed=7, draws=32)
        assert gt["blocked"] is True
        assert gt["pair_prob"] == 0.0
        assert gt["rng_calls"] == 0, f"{sit['cell']} で乱数を引いている"
        assert gt["fired"] == 0
    live = [s for s in V1.situations()
            if s["guardians"] == 0 and s["motive_level"] == "m_hi"][0]
    gt = V1.ground_truth(live, seed=7, draws=32)
    assert gt["blocked"] is False and gt["rng_calls"] == 32


def test_ground_truth_is_seed_stable():
    sit = V1.situations()[0]
    a = V1.ground_truth(sit, seed=123, draws=8)
    b = V1.ground_truth(sit, seed=123, draws=8)
    assert a == b


def test_real_llm_backends_require_explicit_flag():
    """実 LLM は ``allow_real`` が無ければ**構築されない**(ローカル誤実行の安全弁)。"""
    for name in ("vllm", "ollama", "openai_compat"):
        with pytest.raises(SystemExit):
            V1.make_backend(name, model="m")
        with pytest.raises(SystemExit):
            V2.make_reaction_backend(name, model="m")
        with pytest.raises(SystemExit):
            V2.make_judge_backend(name, model="m")
    assert V1.make_backend("mock").name == "mock"
    assert V2.make_reaction_backend("mock").name == "mock"
    assert V2.make_judge_backend("mock").name == "mock-judge"


def test_unknown_backend_is_rejected():
    with pytest.raises(SystemExit):
        V1.make_backend("uncensored-something", model="m", allow_real=True)


# =========================================================================== #
# (D) 内蔵スキャン — 実在名・秘匿情報・規律違反を実際に検出する
# =========================================================================== #
def test_scan_detects_place_names_and_secrets():
    got = V1.scan_text("渋谷のセンター街で会おう。")
    assert any(h.startswith("place_name:") for h in got["hard"])
    # ★偽の資格情報は**連結して組み立てる**(リポジトリのシークレットスキャンが
    #   テストのダミーを本物として拾わないように、ソース上に完全な形で置かない)
    fake_key = "sk-" + "abcdefghijklmnopqrstuvwxyz123456"
    got = V1.scan_text(f"api_key = {fake_key}")
    tags = {h.split(":", 1)[0] for h in got["hard"]}
    assert {"api_key_like", "credential_kv"} <= tags
    for probe, tag in (("連絡は a.b@example.com へ", "email_like"),
                       ("host 192.168.0.1", "ip_like"),
                       ("03-1234-5678 に電話", "phone_like"),
                       ("Authorization: " + "Bearer abcdefghijklmnopqrst", "bearer_token")):
        assert any(h.startswith(tag) for h in V1.scan_text(probe)["hard"]), tag


def test_scan_soft_hits_do_not_break_clean():
    got = V1.scan_text("山田さんが株式会社の前に立っていた。")
    assert got["hard"] == []
    tags = {h.split(":", 1)[0] for h in got["soft"]}
    assert {"person_name_candidate", "company_name_candidate"} <= tags
    res = V1.scan_records([{"case_id": "x", "prompt": "通りに人がいる。",
                            "response": "山田さんが立っていた。"}])
    assert res["clean"] is True and res["soft"]


def test_scan_records_flags_hard_hits_and_prompt_discipline():
    dirty = [{"case_id": "bad", "prompt": "盗むべきかを選べ。",
              "response": "渋谷の交差点で待つ。"}]
    res = V1.scan_records(dirty)
    assert res["clean"] is False
    assert res["hard"] and res["prompt_discipline"]
    assert set(res["prompt_discipline"][0]["words"]) >= {"盗", "べき"}


def test_strict_scan_exit_code(tmp_path, monkeypatch):
    """``--strict-scan`` はスキャンが CLEAN でなければ終了コード 2 を返す。"""
    monkeypatch.setattr(V1, "scan_records",
                        lambda records, **kw: {"clean": False, "n_records": len(records),
                                               "hard": [], "soft": [],
                                               "prompt_discipline": []})
    assert _run_v1(tmp_path / "strict", ["--strict-scan"]) == 2


def test_mock_run_scan_is_clean(tmp_path):
    out = tmp_path / "v1"
    _run_v1(out)
    scan = json.loads((out / "scan.json").read_text(encoding="utf-8"))
    assert scan["clean"] is True and scan["hard"] == []
    out2 = tmp_path / "v2"
    _run_v2(out2)
    scan2 = json.loads((out2 / "scan.json").read_text(encoding="utf-8"))
    assert scan2["clean"] is True and scan2["hard"] == []


# =========================================================================== #
# (E) 集計 — 表の数字が records と一致する
# =========================================================================== #
def test_v1_aggregate_matches_records(tmp_path):
    out = tmp_path / "v1"
    _run_v1(out)
    records = [json.loads(line) for line in
               (out / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    agg = summary["aggregate"]["overall"]
    assert agg["n"] == len(records)
    assert agg["take"] == sum(1 for r in records if r["choice"] == "take")
    assert agg["unreal"] == sum(1 for r in records if r["unreal"])
    assert sum(agg["by_action"].values()) + agg["unparsed"] == len(records)
    # 監視飽和セルは決定論側で厳密に 0(乱数も引かれない)
    blocked = [r for r in records if r["gt"]["blocked"]]
    assert blocked and all(r["gt"]["rng_calls"] == 0 and r["gt"]["pair_prob"] == 0.0
                           for r in blocked)


def test_v2_aggregate_matches_records(tmp_path):
    out = tmp_path / "v2"
    _run_v2(out)
    records = [json.loads(line) for line in
               (out / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    agg = summary["aggregate"]["overall"]
    assert agg["n"] == len(records)
    assert agg["refusal"] == sum(1 for r in records if r["refusal"])
    assert agg["sermon_excess"] == sum(1 for r in records if r["sermon_excess"])
    assert agg["natural_n"] == sum(1 for r in records if r["natural"] is not None)
    assert sum(agg["natural_dist"].values()) == agg["natural_n"]


def test_v2_judge_can_be_disabled(tmp_path):
    out = tmp_path / "nojudge"
    assert _run_v2(out, ["--judge-backend", "none"]) == 0
    records = [json.loads(line) for line in
               (out / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    assert records and all(r["natural"] is None and r["judge_response"] == ""
                           for r in records)


def test_spearman_matches_known_cases():
    assert V1._spearman([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0
    assert V1._spearman([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == -1.0
    assert V1._spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None
    assert V1._spearman([1.0, 2.0], [1.0, 2.0]) is None


def test_limit_and_framing_selection(tmp_path):
    out = tmp_path / "small"
    assert _run_v1(out, ["--limit", "6", "--framings", "observe"]) == 0
    records = [json.loads(line) for line in
               (out / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(records) == 6
    assert {r["framing"] for r in records} == {"observe"}


def test_summary_md_declares_the_observation_discipline(tmp_path):
    """要約 md に「観察であって促進ではない」と mock の但し書きが必ず載る。"""
    out1, out2 = tmp_path / "v1", tmp_path / "v2"
    _run_v1(out1)
    _run_v2(out2)
    for path in (out1 / "summary.md", out2 / "summary.md"):
        text = path.read_text(encoding="utf-8")
        assert "観察であって促進ではない" in text
        assert "mock" in text and "知見ではない" in text
