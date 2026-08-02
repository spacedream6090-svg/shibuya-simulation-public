"""第90バッチ: モデル人間らしさテストバッテリー(scripts/model_battery/)の検証。

検査するのは4種類:
  (a) **刺激の決定論** — 同じ seed/scale なら prompt 列も seed 列もバイト一致。
      C/E は前ターン応答に依存するので「同じ応答を返す偽 ask」の下で一致すること。
  (b) **指標の既知値一致** — JSD・エントロピー・分位点・スケジュール展開・会話統計を
      手計算/numpy と突き合わせる(オラクルのある部分は絶対値で固定する)。
  (c) **プラセボが沈む合成ケース** — 「人間らしい」合成モデルとプラセボを同じハーネスで
      走らせ、全層でプラセボが最下位になること(= 指標が何かを測っている証拠)。
      あわせて D 層の判定線が**引数必須**(未指定なら合否を出さない)ことも固定する。
  (d) **参照統計の来歴** — 出典・ライセンス・取得日が欠けた参照は読み込まない。
      再配布不可/不明の統計に数値本体が付いていたら拒否する。

house style: scripts/ を path 追加して import(tests/test_analyze_specialization.py に倣う)。
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))          # scripts/ は package ではない

from model_battery import clients as C               # noqa: E402
from model_battery import harness as H               # noqa: E402
from model_battery import metrics as M               # noqa: E402
from model_battery import reference as R             # noqa: E402
from model_battery import report as RP               # noqa: E402
from model_battery import stimuli as S               # noqa: E402


# ================================================================ (a) 決定論
def test_stimulus_determinism_same_seed_same_prompts():
    """同一 seed/scale で刺激列(prompt・seed・順序)が完全一致する。"""
    cfg = S.BatteryConfig(seed=1234, temperature=0.7, scale=0.4)
    resp = lambda call: f"応答{call.turn}"          # noqa: E731 (決定論な偽 ask)
    a = S.collect_calls(cfg, S.LAYER_IDS, resp)
    b = S.collect_calls(cfg, S.LAYER_IDS, resp)
    assert len(a) == len(b) > 0
    assert [x.prompt for x in a] == [x.prompt for x in b]
    assert [x.seed for x in a] == [x.seed for x in b]
    assert [(x.layer, x.test, x.case_id, x.turn) for x in a] == \
           [(x.layer, x.test, x.case_id, x.turn) for x in b]
    assert H.chain(x.prompt_sha256 for x in a) == \
           H.chain(x.prompt_sha256 for x in b)

    # seed を変えれば seed 列は変わるが、A/B/D のプロンプト本文は変わらない
    # (プロンプトはモデルにもシードにも依存しない=全モデル同一刺激の担保)
    other = S.collect_calls(S.BatteryConfig(seed=9999, scale=0.4), ("A", "B", "D"),
                            resp)
    base = S.collect_calls(S.BatteryConfig(seed=1234, scale=0.4), ("A", "B", "D"),
                           resp)
    assert [x.prompt for x in other] == [x.prompt for x in base]
    assert [x.seed for x in other] != [x.seed for x in base]


def test_prompts_are_model_agnostic_and_scale_monotone():
    """プロンプトにモデル名が混ざらない/scale が件数を単調に決める。"""
    calls = S.collect_calls(S.BatteryConfig(scale=1.0), S.LAYER_IDS)
    joined = "\n".join(c.prompt for c in calls)
    for banned in ("qwen", "ollama", "gpt", "claude", "llama", "placebo"):
        assert banned not in joined.lower(), f"刺激にモデル名 {banned} が混入"
    # D 層は全サンプルで同一プロンプト(シードだけ変える)= 分散測定の前提
    d = [c for c in calls if c.layer == "D"]
    assert len({c.prompt for c in d}) == 1
    assert len({c.seed for c in d}) == len(d)

    n_full = len(calls)
    n_small = len(S.collect_calls(S.BatteryConfig(scale=0.2), S.LAYER_IDS))
    assert 0 < n_small < n_full
    assert S.BatteryConfig(scale=1.0).n_d_samples == 30      # 正典どおり
    assert S.BatteryConfig(scale=1.0).n_e_turns == 50
    assert S.BatteryConfig(scale=0.2).n_d_samples == 6       # 1/5 縮小版


def test_layer_spec_parsing():
    assert S.iter_layers("all") == S.LAYER_IDS
    assert S.iter_layers("C,A") == ("A", "C")       # 正典順に正規化
    with pytest.raises(ValueError):
        S.iter_layers("A,Z")


# ================================================================ (b) 既知値
def test_jensen_shannon_known_values():
    assert M.jensen_shannon([1, 0], [1, 0]) == pytest.approx(0.0, abs=1e-12)
    assert M.jensen_shannon([1, 0], [0, 1]) == pytest.approx(1.0, abs=1e-12)
    # [1,0] と [0.5,0.5]: m=[0.75,0.25] → 0.5*1*log2(1/0.75) + 0.5*(0.5*log2(2/3)... )
    p, q = [1.0, 0.0], [0.5, 0.5]
    expect = 0.5 * (1.0 * math.log2(1 / 0.75)) + \
        0.5 * (0.5 * math.log2(0.5 / 0.75) + 0.5 * math.log2(0.5 / 0.25))
    assert M.jensen_shannon(p, q) == pytest.approx(expect, abs=1e-12)
    # 生の度数で渡しても正規化されるので同値
    assert M.jensen_shannon([10, 0], [3, 3]) == pytest.approx(expect, abs=1e-12)
    with pytest.raises(ValueError):
        M.jensen_shannon([1, 0], [0, 0])


def test_entropy_gini_known_values():
    assert M.normalized_entropy([1, 1, 1, 1]) == pytest.approx(1.0)
    assert M.normalized_entropy([1, 0, 0, 0]) == pytest.approx(0.0)
    assert M.normalized_entropy([3, 1]) == pytest.approx(0.8112781244591328)
    assert M.gini_simpson([1, 1]) == pytest.approx(0.5)
    assert M.gini_simpson([5, 0]) == pytest.approx(0.0)
    assert M.gini_simpson([1, 1, 1, 1]) == pytest.approx(0.75)


def test_quantile_matches_numpy():
    xs = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
    for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        assert M.quantile(xs, q) == pytest.approx(float(np.quantile(xs, q)))
    with pytest.raises(ValueError):
        M.quantile([], 0.5)


def test_schedule_to_slots_budget_and_conflicts():
    blocks = [
        {"start": "00:00", "end": "06:00", "activity": "睡眠"},
        {"start": "06:00", "end": "07:00", "activity": "身の回りの用事"},
        {"start": "07:00", "end": "08:00", "activity": "食事"},
        {"start": "23:00", "end": "01:00", "activity": "睡眠"},   # 日跨ぎ
    ]
    slots, conflicts = M.schedule_to_slots(blocks, slot_minutes=15)
    assert len(slots) == 96
    assert conflicts == 0                       # 0:00-1:00 は既に睡眠=同一値
    assert slots[0] == "睡眠" and slots[23] == "睡眠"      # 00:00-06:00
    assert slots[24] == "身の回りの用事"                    # 06:00
    assert slots[28] == "食事"                              # 07:00
    assert slots[32] is None                                # 08:00 は未計画
    assert slots[92] == "睡眠"                              # 23:00
    assert M.slot_coverage(slots) == pytest.approx((24 + 4 + 4 + 4) / 96)
    budget = M.time_budget(slots, slot_minutes=15)
    assert budget[M.CATEGORY_INDEX["睡眠"]] == pytest.approx(6 * 60 + 60)
    assert budget[M.CATEGORY_INDEX["食事"]] == pytest.approx(60)
    assert len(budget) == len(M.ACTIVITY_CATEGORIES) == 20

    # 衝突は「別の行動で上書きしようとした回数」
    _, conf2 = M.schedule_to_slots(
        [{"start": "09:00", "end": "10:00", "activity": "仕事"},
         {"start": "09:00", "end": "10:00", "activity": "食事"}], slot_minutes=15)
    assert conf2 == 4


def test_wake_sleep_commute_extraction():
    blocks = [
        {"start": "00:00", "end": "06:30", "activity": "睡眠"},
        {"start": "06:30", "end": "07:30", "activity": "身の回りの用事"},
        {"start": "08:00", "end": "09:00", "activity": "通勤"},      # 別名
        {"start": "09:00", "end": "18:00", "activity": "仕事"},
        {"start": "23:30", "end": "24:00", "activity": "睡眠"},
    ]
    slots, _ = M.schedule_to_slots(blocks, slot_minutes=15)
    assert M.wake_time(slots, slot_minutes=15) == 6 * 60 + 30
    assert M.sleep_time(slots, slot_minutes=15) == 23 * 60 + 30
    assert M.first_activity_time(slots, "通勤・通学", slot_minutes=15) == 8 * 60
    gap = M.quantile_gap([390.0, 390.0, 390.0], [384.0] * 3)
    assert gap["q50"] == pytest.approx(6.0)
    assert gap["mean_abs"] == pytest.approx(6.0)


def test_normalize_activity_aliases():
    assert M.normalize_activity("睡眠") == "睡眠"
    assert M.normalize_activity("通勤") == "通勤・通学"
    assert M.normalize_activity("移動(通勤・通学以外)") == "移動"
    assert M.normalize_activity("テレビ") == "テレビ・ラジオ・新聞・雑誌"
    assert M.normalize_activity("昼食") == "食事"
    assert M.normalize_activity("量子コンピュータの設計") == "その他"
    for cat in M.ACTIVITY_CATEGORIES:               # 正典語は必ず自分に戻る
        assert M.normalize_activity(cat) == cat


def test_extract_json_lenient():
    assert M.extract_json('{"a":1}') == {"a": 1}
    assert M.extract_json('説明します。\n```json\n{"a": 2}\n```\n以上') == {"a": 2}
    assert M.extract_json('前置き {"a": {"b": 3}} 後書き') == {"a": {"b": 3}}
    assert M.extract_json('{"a": "文字列の中の } は無視"}') == \
        {"a": "文字列の中の } は無視"}
    assert M.extract_json('{"a": 4') == {"a": 4}          # 途中で切れた JSON
    assert M.extract_json("何も無い") is None
    assert M.extract_json("") is None


def test_conversation_metrics_known_values():
    utts = ["うん。", "そうだね", "でも、それはちょっと違うと思うんだよね",
            "渋谷の人混みは本当にどうにかしてほしい"]
    assert M.is_backchannel("うん。") and M.is_backchannel("そうだね")
    assert not M.is_backchannel("うん、それは違うと思うけどね")
    assert M.backchannel_rate(utts) == pytest.approx(0.5)
    assert M.disagreement_rate(utts) == pytest.approx(0.25)
    assert M.utterance_lengths(["あいうえ お"]) == [5]
    # 同一文の繰り返し: 2-gram は完全重複 → distinct=0.5、repetition=0.5
    assert M.distinct_n(["あいう", "あいう"], 2) == pytest.approx(0.5)
    assert M.repetition_rate(["あいう", "あいう"], 2) == pytest.approx(0.5)
    assert M.mean_pairwise_distance(["あいう", "あいう"], 2) == pytest.approx(0.0)
    assert M.mean_pairwise_distance(["あいう", "かきく"], 2) == pytest.approx(1.0)
    assert M.topic_persistence(["あいう", "あいう"], 2) == pytest.approx(1.0)
    assert M.topic_persistence(["あいう", "かきく"], 2) == pytest.approx(0.0)
    assert M.cv([2.0, 2.0, 2.0]) == pytest.approx(0.0)
    assert M.slope([0.1, 0.2, 0.3]) == pytest.approx(0.1)
    assert M.windows(list(range(9)), 3) == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]


def test_contradiction_detection_and_variance_ratio():
    assert M.contradiction_count(["私は納豆が好き。"]) == 0
    assert M.contradiction_count(["私は納豆が好き。", "私は納豆が嫌い。"]) == 1
    assert M.contradiction_count(["私は納豆が好き。", "私は寿司が嫌い。"]) == 0
    assert M.variance_ratio(0.6, 0.8) == pytest.approx(0.75)
    with pytest.raises(ValueError):
        M.variance_ratio(0.6, 0.0)


def _rec(resp, **meta):
    return {"response": resp, "error": False, "case_id": meta.pop("case", "c"),
            "meta": meta}


def test_validity_gate_and_activity_label_rate():
    """実測で踏んだ2つの穴を塞いだことの固定。

    (1) activity キーを落として start/end だけ返すモデル → 黙って "その他" に
        畳まず、ラベル率を出して解釈不能と言う。
    (2) 英語の前置きを本文に書くモデル → 会話統計が見かけ上よくなるのを
        日本語率ゲートで殺す。
    """
    assert M.latin_ratio(["あいう"]) == pytest.approx(0.0)
    assert M.latin_ratio(["abcあ"]) == pytest.approx(0.75)
    assert M.latin_ratio(["ab", "あい"]) == pytest.approx(0.5)
    assert M.latin_ratio([]) == pytest.approx(0.0)

    labeled = json.dumps({"blocks": [
        {"start": "00:00", "end": "07:00", "activity": "睡眠"},
        {"start": "07:00", "end": "24:00", "activity": "仕事"}]}, ensure_ascii=False)
    unlabeled = json.dumps({"blocks": [{"start": "00:00", "end": "07:00"},
                                       {"start": "07:00", "end": "24:00"}]})
    ok = RP.score_a([_rec(labeled, persona="p1", daytype="平日", case="p1/平日"),
                     _rec(labeled, persona="p2", daytype="休日", case="p2/休日")],
                    None)
    assert ok["activity_labeled"] == pytest.approx(1.0)
    assert ok["schema_ok"] == pytest.approx(1.0) and ok["score"] is not None
    ng = RP.score_a([_rec(unlabeled, persona="p1", daytype="平日", case="p1/平日")],
                    None)
    assert ng["activity_labeled"] == pytest.approx(0.0)
    assert ng["schema_ok"] == pytest.approx(0.0)
    assert ng["score"] is None and "行動ラベル率" in ng["note"]

    ja = ["今朝は霧が出ていたね", "駅前の工事はいつ終わるんだろう",
          "傘を店に置いてきてしまった", "値上げの話ばかりで滅入る"]
    en = ["Okay, I need to think about what Tamura would say here about the weather"
          for _ in ja]

    def _say(t):
        return _rec(json.dumps({"say": t}, ensure_ascii=False))

    c_ja = RP.score_c([_say(t) for t in ja], None)
    c_en = RP.score_c([_say(t) for t in en], None)
    assert c_ja["schema_ok"] == pytest.approx(1.0)
    # 封筒が壊れた応答は発話として数えない(生 JSON が語彙統計に混ざらない)
    broken = RP.score_c([_rec("Okay, let me think about this first.")] * 4, None)
    assert broken["schema_ok"] == pytest.approx(0.0)
    assert broken["utterances"] == 0 and broken["score"] is None
    assert c_ja["japanese_ratio"] == pytest.approx(1.0)
    assert c_en["japanese_ratio"] < 0.1
    assert c_ja["score"] == pytest.approx(c_ja["score_before_gate"])
    assert c_en["score"] < c_en["score_before_gate"]
    assert c_en["score"] < c_ja["score"]


# ================================================================ (d) 来歴
def _ref_doc(**over) -> dict:
    doc = {
        "schema": R.SCHEMA,
        "id": "dummy",
        "title": "ダミー統計",
        "status": "取得済",
        "attribution": "出典: ダミー",
        "source": {"publisher": "誰か", "title": "何か", "url": "https://example.invalid",
                   "accessed": "2026-08-03"},
        "license": {"name": "CC BY 4.0", "url": "https://example.invalid/l",
                    "redistribution": "permitted_with_attribution"},
        "series": {"x": {"values": [1, 2, 3]}},
    }
    doc.update(over)
    return doc


def test_reference_provenance_is_enforced():
    R.validate(_ref_doc())                                   # 正常系
    with pytest.raises(R.ReferenceError):                    # 出典 URL 欠落
        R.validate(_ref_doc(source={"publisher": "誰か", "title": "何か",
                                    "accessed": "2026-08-03"}))
    with pytest.raises(R.ReferenceError):                    # ライセンス欠落
        R.validate(_ref_doc(license={"name": "?", "url": ""}))
    with pytest.raises(R.ReferenceError):                    # 未知の再配布区分
        R.validate(_ref_doc(license={"name": "x", "url": "u",
                                     "redistribution": "maybe"}))
    with pytest.raises(R.ReferenceError):                    # 再配布不可なのに値がある
        R.validate(_ref_doc(license={"name": "x", "url": "u",
                                     "redistribution": "prohibited"}))
    with pytest.raises(R.ReferenceError):                    # 未取得なのに値がある
        R.validate(_ref_doc(status="未取得"))
    with pytest.raises(R.ReferenceError):                    # 取得済なのに値が無い
        R.validate(_ref_doc(series={"x": {"values": None}}))
    with pytest.raises(R.ReferenceError):                    # schema 不一致
        R.validate(_ref_doc(schema="other/1"))


def test_shipped_reference_files_validate():
    """リポジトリに置いた参照統計が全部 掟を満たす(出典・ライセンス・取得日)。"""
    d = _ROOT / "data" / "battery" / "reference"
    files = sorted(d.glob("*.json"))
    assert files, f"参照統計が1つも無い: {d}"
    docs = R.load_dir(d)
    assert len(docs) == len(files)
    for doc in docs.values():
        assert doc["source"]["url"].startswith("http")
        assert doc["license"]["redistribution"] in R.REDISTRIBUTION_VALUES
        # 再配布不可/不明のものは値を持たない(生データを置いていない)
        if doc["license"]["redistribution"] not in ("permitted_with_attribution",
                                                    "permitted", "derived_stats_only"):
            for s in (doc.get("series") or {}).values():
                assert s.get("values") is None
    assert RP.REF_TIME_USE in docs, "A 層の参照統計(社会生活基本調査)が無い"
    assert RP.REF_DIALOGUE in docs, "C 層の参照統計(会話コーパス)が無い"
    # A 層の本命(時刻別行動者率)は 24 行 × カテゴリ数で揃っていること
    s = docs[RP.REF_TIME_USE]["series"][RP.HOURLY_KEY]
    assert len(s["values"]) == 24
    assert all(len(row) == len(s["categories"]) for row in s["values"])
    assert all(0.0 <= sum(row) <= 100.0 for row in s["values"])
    assert all(c in M.CATEGORY_INDEX for c in s["categories"])
    # 週平均の時間配分は 20 分類ぶんの長さで、未収載は null(推測で埋めていない)
    wk = docs[RP.REF_TIME_USE]["series"][RP.BUDGET_KEY]["values"]
    assert len(wk) == len(M.ACTIVITY_CATEGORIES)
    assert sum(v for v in wk if v is not None) <= 1440
    assert any(v is None for v in wk)
    # C 層で必要な発話統計は「未確認」= 絶対評価に使わせない
    for k in ("mean_utterance_chars", "backchannel_rate", "human_choice_entropy"):
        assert R.series_values(docs[RP.REF_DIALOGUE], k) is None
    # CEJC は契約ゲートなので値を1つも持たない
    cejc = docs["cejc_descriptive_out_of_scope"]
    assert cejc["license"]["redistribution"] == "prohibited"
    assert all(v.get("values") is None for v in cejc["series"].values())


def test_reference_driven_scores_are_zero_at_perfect_match():
    """時刻別 JSD / 時間配分 JSD の既知値: 参照に一致するスケジュールなら 0。"""
    ref = {"series": {
        RP.HOURLY_KEY: {"categories": ["睡眠", "仕事"],
                        "values": [[50.0, 50.0]] * 24},
        RP.BUDGET_KEY: {"values": [720, None, None, None, 720] + [None] * 15},
    }}
    all_sleep, _ = M.schedule_to_slots(
        [{"start": "00:00", "end": "24:00", "activity": "睡眠"}], slot_minutes=15)
    all_work, _ = M.schedule_to_slots(
        [{"start": "00:00", "end": "24:00", "activity": "仕事"}], slot_minutes=15)
    parsed = {
        "a/平日": {"slots": all_sleep, "daytype": "平日",
                   "budget": M.time_budget(all_sleep, slot_minutes=15)},
        "b/平日": {"slots": all_work, "daytype": "平日",
                   "budget": M.time_budget(all_work, slot_minutes=15)},
    }
    assert RP.hourly_jsd(parsed, ref, "平日") == pytest.approx(0.0, abs=1e-12)
    assert RP.budget_jsd(parsed, ref) == pytest.approx(0.0, abs=1e-12)
    # 参照が全く違う形(全部睡眠)なら JSD は 0 より大きい
    ref2 = {"series": {RP.HOURLY_KEY: {"categories": ["睡眠", "仕事"],
                                       "values": [[100.0, 0.0]] * 24}}}
    # 既知値: モデル [0.5,0.5,0] vs 参照 [1,0,0] の JSD
    assert RP.hourly_jsd(parsed, ref2, "平日") == pytest.approx(
        M.jensen_shannon([0.5, 0.5, 0.0], [1.0, 0.0, 0.0]))
    assert RP.hourly_jsd(parsed, ref2, "平日") == pytest.approx(0.3112781244591328)
    # 参照が無い/曜日が無ければ None(黙って 0 を返さない)
    assert RP.hourly_jsd(parsed, {"series": {}}, "平日") is None
    assert RP.hourly_jsd(parsed, ref, "休日") is None
    assert RP.budget_jsd(parsed, {"series": {}}) is None


# ================================================================ (c) プラセボ
class _SyntheticHumanClient(C.BatteryClient):
    """「人間らしい」合成モデル。ペルソナ・摂動・シードに反応する決定論の作り物。

    実 LLM の代役ではなく、**指標が向きを持つことの確認**に使う。
    """

    backend = "synthetic"

    def __init__(self, name: str = "synthetic/good"):
        self.model = "good"
        self.name = name

    @staticmethod
    def _persona_of(prompt: str) -> str:
        for p in S.PERSONAS:
            if f"名前: {p.name}" in prompt:
                return p.pid
        return "unknown"

    def generate(self, prompt, *, temperature, max_tokens, seed, json_mode):
        pid = self._persona_of(prompt)
        off = (abs(hash(pid)) % 5) * 15
        holiday = "今日は休日です" in prompt
        if '"blocks"' in prompt:
            wake = 360 + off + (90 if holiday else 0)
            def hhmm(m):
                return f"{(m // 60) % 24:02d}:{m % 60:02d}"
            work = "休養・くつろぎ" if holiday else "仕事"
            return json.dumps({"blocks": [
                {"start": "00:00", "end": hhmm(wake), "activity": "睡眠"},
                {"start": hhmm(wake), "end": hhmm(wake + 45), "activity": "身の回りの用事"},
                {"start": hhmm(wake + 45), "end": hhmm(wake + 75), "activity": "食事"},
                {"start": hhmm(wake + 75), "end": hhmm(wake + 135), "activity": "通勤・通学"},
                {"start": hhmm(wake + 135), "end": hhmm(wake + 375), "activity": work},
                {"start": hhmm(wake + 375), "end": hhmm(wake + 435), "activity": "食事"},
                {"start": hhmm(wake + 435), "end": hhmm(wake + 735), "activity": work},
                {"start": hhmm(wake + 735), "end": hhmm(wake + 795), "activity": "移動"},
                {"start": hhmm(wake + 795), "end": hhmm(wake + 855), "activity": "食事"},
                {"start": hhmm(wake + 855), "end": "24:00",
                 "activity": "テレビ・ラジオ・新聞・雑誌"},
            ]}, ensure_ascii=False)
        if '"change"' in prompt:
            if "雨が降り出した" in prompt:
                return json.dumps({"change": True, "new_activity": "休養・くつろぎ",
                                   "delta_minutes": 0,
                                   "reason": f"{pid}なので濡れたくない"},
                                  ensure_ascii=False)
            if "遅延" in prompt:
                return json.dumps({"change": True, "new_activity": "通勤・通学",
                                   "delta_minutes": -30 - off,
                                   "reason": f"{pid}は遅刻したくない"},
                                  ensure_ascii=False)
            if "臨時休業" in prompt:
                return json.dumps({"change": True, "new_activity": "買い物",
                                   "delta_minutes": 15,
                                   "reason": f"{pid}は別の店を探す"},
                                  ensure_ascii=False)
            accept = off % 2 == 0
            return json.dumps({"change": accept,
                               "new_activity": "交際・付き合い" if accept else "家事",
                               "delta_minutes": 30 if accept else 0,
                               "reason": f"{pid}の都合による"}, ensure_ascii=False)
        if '"choice"' in prompt:
            ch = S.D_CHOICES[seed % len(S.D_CHOICES)]
            return json.dumps({"choice": ch,
                               "why": f"{ch}のは{seed % 97}番目の気分だから"},
                              ensure_ascii=False)
        if '"plan"' in prompt:
            day = seed % 7
            return json.dumps({"day": day, "plan": [
                {"when": f"{8 + day % 3:02d}:00", "what": f"用事{seed % 53}"},
                {"when": f"{13 + day % 4:02d}:30", "what": f"寄り道{seed % 31}"},
                {"when": "19:00", "what": f"帰宅して{seed % 17}をする"}],
                "mood": f"気分は{seed % 11}"}, ensure_ascii=False)
        # 発話: 断片プールから 1〜4 個を選んでつなぐ = 長さも語彙も散らばる
        n = 1 + seed % 4
        picks = [_FRAGMENTS[(seed + i * 7) % len(_FRAGMENTS)] for i in range(n)]
        text = "。".join(picks) + "。"
        if '"say"' in prompt:
            return json.dumps({"say": text}, ensure_ascii=False)
        return text


# 「人間らしい」自由文を作るための断片プール(長さも語彙もばらける素材)。
_FRAGMENTS = (
    "今朝の坂の上は霧が出ていて向こうの信号がぜんぜん見えなかった",
    "駅前の工事はいったいいつ終わるのか誰も知らないらしい",
    "昨日入った蕎麦屋のつゆが思っていたよりずいぶん辛かった",
    "窓から見えた看板の色が去年と違っている気がする",
    "母から届いた葉書にみかんの木の話が書いてあった",
    "夜中に目が覚めてしまってしばらく天井を眺めていた",
    "図書館の三階は暖房が効きすぎていて眠くなる",
    "自転車の鍵をどこかに落としたかもしれない",
    "近所にいた三毛猫を最近見かけない",
    "そういえば傘を店に置いてきた",
    "値上げの話ばかり耳に入ってきて気が滅入る",
    "同僚の結婚式の招待状が机の上に置いてあった",
    "帰り道に古い映画館の跡地を見つけて立ち止まった",
    "冷蔵庫の奥から賞味期限の切れた瓶が出てきた",
    "階段の踊り場から見える空だけが妙に広い",
    "去年の手帳を開いたら知らない電話番号が挟まっていた",
    "洗濯物が乾かないまま三日たっている",
    "隣の部屋から夜更けにギターの音がする",
)


def _run_battery(tmp: Path, model_clients, scale=0.2):
    cfg = S.BatteryConfig(seed=777, temperature=0.8, scale=scale)
    out = tmp / "raw"
    for client in model_clients:
        runner = H.Runner(client, cfg, out, verbose=False)
        runner.run(S.LAYER_IDS)
        runner.write(runner.determinism_probe(2))
    return out


def test_placebo_sinks_in_every_layer(tmp_path):
    """合成「人間らしい」モデル vs プラセボ → 全層でプラセボが最下位。"""
    raw = _run_battery(tmp_path, [_SyntheticHumanClient(), C.PlaceboClient()])
    profiles = {slug: RP.profile_model(e, {})
                for slug, e in sorted(RP.load_raw(raw).items())}
    RP.add_variance_ratio(profiles, None)
    sanity = RP.placebo_sanity(profiles)
    assert sanity["verdict"] == "PASS", json.dumps(sanity, ensure_ascii=False)
    for layer in S.LAYER_IDS:
        st = sanity["by_layer"][layer]
        assert st["status"] == "PASS", f"{layer}: {st}"
        assert st["margin"] > 0
    # D 層(生命線)はプラセボがゼロ分散に沈むこと
    pl = next(p for s, p in profiles.items() if "placebo" in s)
    assert pl["layers"]["D"]["score"] == pytest.approx(0.0)
    assert pl["layers"]["D"]["distinct_choices"] == 1


def test_d_ratio_floor_is_required_and_not_hardcoded(tmp_path):
    """判定線は引数必須。未指定なら合否を出さない(事後の閾値いじりを封じる)。"""
    raw = _run_battery(tmp_path, [_SyntheticHumanClient("synthetic/a"),
                                  _SyntheticHumanClient("synthetic/b"),
                                  C.PlaceboClient()])
    profiles = {slug: RP.profile_model(e, {})
                for slug, e in sorted(RP.load_raw(raw).items())}
    RP.add_variance_ratio(profiles, None)
    assert RP.shortlist_verdict(profiles, None)["verdict"] == "NOT_JUDGED"
    lo = RP.shortlist_verdict(profiles, 0.5)
    assert lo["verdict"] == "GO" and len(lo["passed"]) >= 2
    hi = RP.shortlist_verdict(profiles, 1.5)          # 誰も超えられない下限
    assert hi["verdict"] == "NO_GO" and hi["passed"] == []
    # ソースに閾値のハードコードが無いこと(既定値 None)
    src = (_ROOT / "scripts" / "model_battery" / "report.py").read_text(
        encoding="utf-8")
    assert '"--d-ratio-floor", type=float, default=None' in src


# ================================================================ ハーネス
def test_harness_end_to_end_is_reproducible(tmp_path):
    """プラセボで端から端まで走り、2回走らせて決定論部分の鎖ハッシュが一致する。"""
    args = ["--model", "placebo:template", "--layers", "all", "--scale", "0.2",
            "--seed", "424242", "--quiet", "--determinism-probe", "2", "--out"]
    assert H.main(args + [str(tmp_path / "r1")]) == 0
    assert H.main(args + [str(tmp_path / "r2")]) == 0
    m1 = json.loads((tmp_path / "r1" / "placebo_template" / "manifest.json")
                    .read_text(encoding="utf-8"))
    m2 = json.loads((tmp_path / "r2" / "placebo_template" / "manifest.json")
                    .read_text(encoding="utf-8"))
    assert m1["digests"] == m2["digests"]
    assert m1["files"].keys() == m2["files"].keys()
    assert m1["calls"] == m2["calls"] == 39 and m1["errors"] == 0
    assert m1["determinism_probe"]["rate"] == 1.0        # プラセボは必ず再現する
    assert m1["config"]["seed"] == 424242
    assert m1["config"]["n_d_samples"] == 6

    # raw 行の来歴が揃っている(モデルID/温度/シード/プロンプト sha256)
    line = (tmp_path / "r1" / "placebo_template" / "D_variance.jsonl") \
        .read_text(encoding="utf-8").splitlines()[0]
    rec = json.loads(line)
    assert rec["schema"] == H.RAW_SCHEMA
    assert rec["model"]["name"] == "placebo/template"
    assert rec["params"]["temperature"] == 0.8
    assert isinstance(rec["params"]["seed"], int)
    assert len(rec["prompt_sha256"]) == 64 and len(rec["response_sha256"]) == 64
    assert rec["prompt_sha256"] == H.sha256_text(rec["prompt"])
    # 決定論ダイジェストは elapsed_s を含まない(時間で揺れない)
    d1 = H.det_digest(rec)
    assert H.det_digest({**rec, "elapsed_s": 999.0}) == d1


def test_model_spec_parsing_and_error_convention():
    c = C.parse_model_spec("ollama:qwen3:4b")
    assert (c.backend, c.model, c.name) == ("ollama", "qwen3:4b", "ollama/qwen3:4b")
    assert c.slug == "ollama_qwen3_4b"
    c2 = C.parse_model_spec("ollama:qwen3:8b|host=http://h:1/|timeout_s=5")
    assert c2.host == "http://h:1" and c2.timeout_s == 5.0
    # seed は options に必ず載る(src の OllamaBackend との差分=バッテリーの前提)
    pl = c2._payload("p", 0.5, 32, 4242, True)
    assert pl["options"]["seed"] == 4242 and pl["format"] == "json"
    assert pl["think"] is False
    assert "seed" not in json.dumps(pl["model"])
    c3 = C.parse_model_spec("openai:gpt-x|base_url=http://x/v1|api_key_env=FOO_KEY")
    assert c3.name == "api/gpt-x" and c3.api_key_env == "FOO_KEY"
    assert C.parse_model_spec("placebo:template").backend == "placebo"
    for bad in ("qwen3:4b", "ollama:", "unknown:x", "ollama:x|nokv"):
        with pytest.raises(ValueError):
            C.parse_model_spec(bad)
    assert C.is_error("__ollama_error__: boom")
    assert C.is_error("__api_error__: HTTP 400 Bad Request")
    assert not C.is_error('{"action":"speak"}')


def test_report_renders_markdown_and_json(tmp_path):
    raw = _run_battery(tmp_path, [_SyntheticHumanClient(), C.PlaceboClient()])
    out = tmp_path / "out"
    rc = RP.main(["--raw", str(raw), "--reference", str(tmp_path / "none"),
                  "--out", str(out), "--d-ratio-floor", "0.5"])
    assert rc == 0
    rep = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert rep["schema"] == RP.REPORT_SCHEMA
    assert rep["placebo_sanity"]["verdict"] == "PASS"
    assert set(rep["profiles"]) == {"synthetic_good", "placebo_template"}
    md = (out / "report.md").read_text(encoding="utf-8")
    for layer_label in RP.LAYER_LABEL.values():
        assert layer_label in md
    assert "プラセボ健全性" in md and "判定線" in md
    # 参照統計が無いランでは、絶対評価をしていないことが本文に残る
    assert "参照統計なし" in json.dumps(rep, ensure_ascii=False)
