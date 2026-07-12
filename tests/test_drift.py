"""バッチB: semantic drift(M1c)+ 4層Y統合(D1)の純関数テスト。

- edit_distance / char_ngrams / ngram_cosine の数値検証
- communities が会話グラフを2群に割る
- drift_metrics が語形/文脈の分岐を検出する
- agent_features(y_weights=None) が現行出力と辞書一致(後方互換)
- y_weights 指定で Y_composite が正しい線形結合
"""
from __future__ import annotations

from society.observer import measure
from society.observer.measure import _TOOL_COLS


def _ev(step, agent_id, kind, **payload):
    """load_events 出力と同じ形の 1 イベント dict。"""
    return {"step": step, "sim_min": step, "agent_id": agent_id, "kind": kind,
            "x": 0.0, "y": 0.0, "rng_stream": "", "llm_call_id": None,
            "payload": payload}


# --------------------------------------------------------------------------- #
# 1. 文字列/プロファイルのプリミティブ
# --------------------------------------------------------------------------- #
def test_edit_distance_values():
    assert measure.edit_distance("kitten", "sitting") == 3
    assert measure.edit_distance("", "abc") == 3
    assert measure.edit_distance("abc", "") == 3
    assert measure.edit_distance("abc", "abc") == 0
    assert measure.edit_distance("grievance", "grievanse") == 1
    assert measure.edit_distance("flaw", "lawn") == 2


def test_char_ngrams_profile():
    prof = measure.char_ngrams("abcabc", 3)
    assert dict(prof) == {"abc": 2, "bca": 1, "cab": 1}
    # 長さ < n の非空文字列は全体を1 gram に
    assert dict(measure.char_ngrams("ab", 3)) == {"ab": 1}
    assert dict(measure.char_ngrams("", 3)) == {}


def test_ngram_cosine_values():
    same = measure.char_ngrams("hello world", 3)
    assert abs(measure.ngram_cosine(same, same) - 1.0) < 1e-9
    # 完全に素なプロファイル → 0
    a = measure.char_ngrams("aaaa", 3)
    b = measure.char_ngrams("bbbb", 3)
    assert measure.ngram_cosine(a, b) == 0.0
    # 部分一致: "abcd"={abc,bcd}, "abce"={abc,bce} → 共通1, cos = 1/2
    p1 = measure.char_ngrams("abcd", 3)
    p2 = measure.char_ngrams("abce", 3)
    assert abs(measure.ngram_cosine(p1, p2) - 0.5) < 1e-9
    # 空との比較 → 0
    assert measure.ngram_cosine(same, measure.char_ngrams("", 3)) == 0.0


# --------------------------------------------------------------------------- #
# 2. コミュニティ分割(決定論的 label propagation)
# --------------------------------------------------------------------------- #
def _two_group_events():
    """{0,1,2} と {3,4,5} が別々に会話する非連結グラフ + それぞれ別変種を使う。"""
    return [
        _ev(0, 0, "vocab_coin", item_id="A", text="grievance"),
        _ev(0, 0, "speak", text="grievance is real", items=["grievance"],
            hearers=[1, 2]),
        _ev(1, 1, "speak", text="grievance grievance yeah", items=["grievance"],
            hearers=[0, 2]),
        _ev(2, 2, "speak", text="grievance ok fine", items=["grievance"],
            hearers=[0, 1]),
        _ev(3, 3, "vocab_coin", item_id="B", text="grievanse"),
        _ev(3, 3, "speak", text="grievanse totally different now here",
            items=["grievanse"], hearers=[4, 5]),
        _ev(4, 4, "speak", text="grievanse zzz qqq mmm", items=["grievanse"],
            hearers=[3, 5]),
        _ev(5, 5, "speak", text="grievanse xxx uuu", items=["grievanse"],
            hearers=[3, 4]),
    ]


def test_communities_split_into_two():
    comm = measure.communities(_two_group_events(), 6)
    assert comm[0] == comm[1] == comm[2]
    assert comm[3] == comm[4] == comm[5]
    assert comm[0] != comm[3]
    # 正準 id = 所属集団の最小メンバ
    assert comm[0] == 0 and comm[3] == 3


def test_communities_empty_safe():
    assert measure.communities([], 0) == {}
    # 会話が無ければ各自が単独コミュニティ
    comm = measure.communities([], 3)
    assert comm == {0: 0, 1: 1, 2: 2}


# --------------------------------------------------------------------------- #
# 3. drift_metrics(語形分岐 + 文脈分岐)
# --------------------------------------------------------------------------- #
def test_drift_metrics_detects_divergence():
    out = measure.drift_metrics(_two_group_events(), 6)
    # 2変種は編集距離1で同一クラスタ → 代表語 "grievance"
    assert "grievance" in out
    d = out["grievance"]
    # 語形分岐: コミュニティ間で別変種 → 平均編集距離 = 1
    assert d["form_divergence"] == 1.0
    # 文脈分岐: 別コミュニティのテキストは異なる → cosine 距離 > 0
    assert d["context_divergence"] > 0.0
    assert d["n_communities"] == 2


def test_drift_metrics_empty_and_single_community():
    # 造語ゼロ → 空 dict(壊れない)
    assert measure.drift_metrics([_ev(0, 0, "speak", text="hi", items=[],
                                      hearers=[1])], 2) == {}
    # 単一コミュニティ(全員が会話)で1語のみ → 分岐なし
    events = [
        _ev(0, 0, "vocab_coin", item_id="A", text="wave"),
        _ev(0, 0, "speak", text="wave now", items=["wave"], hearers=[1, 2]),
        _ev(1, 1, "speak", text="wave again", items=["wave"], hearers=[0, 2]),
    ]
    out = measure.drift_metrics(events, 3)
    assert "wave" in out
    assert out["wave"]["form_divergence"] == 0.0
    # 単一コミュニティ → 文脈分岐は 0(比較相手なし)
    assert out["wave"]["context_divergence"] == 0.0
    assert out["wave"]["n_communities"] == 1


# --------------------------------------------------------------------------- #
# 4. 後方互換: agent_features(y_weights=None) は現行と一致
# --------------------------------------------------------------------------- #
def _feature_events():
    return [
        _ev(0, 0, "vocab_coin", item_id="X", text="foo"),
        _ev(1, 0, "speak", hearers=[1], items=["X"], text="..."),
        _ev(2, 1, "transmission", item_id="X", **{"from": 0}, channel="face"),
        _ev(3, 2, "transmission", item_id="X", **{"from": 1}, channel="face"),
        _ev(4, 2, "label_adopt", item_id="X", text="foo"),
        _ev(5, 1, "label_adopt", item_id="X", text="foo"),
        _ev(6, 0, "reflect", mode="free", written_back=True, belief="abcd"),
        _ev(7, 0, "venture_open", name="stall"),
        _ev(8, 0, "venture_sale", amount=100),
    ]


def test_agent_features_backward_compatible():
    events = _feature_events()
    meta = [{"id": 0, "name": "A"}, {"id": 1, "name": "B"}, {"id": 2, "name": "C"}]
    default = measure.agent_features(events, meta)
    explicit_none = measure.agent_features(events, meta, None, None)
    # 既定 == 明示 None(新引数が既定挙動を1バイトも変えない)
    assert default == explicit_none
    # 新列は一切足されない
    for row in default:
        assert "Y_composite" not in row
    # 既知の列集合ちょうど(traits 無し)。ツール列 + 意見操作量(#16)+ SNS 反応(#14)。
    expected = {
        "id", "name", "occupation", "Y_external", "Y_internal",
        "c_transmission", "c_label_adopt", "c_new_relations", "c_sns_reach",
        "reflect_writeback", "belief_chars", "compute", "n_fires_received",
        "c_opinion_moved", "sns_likes_received", "sns_reshares_received",
    } | set(_TOOL_COLS)
    assert set(default[0]) == expected


def test_agent_features_y_composite_linear():
    events = _feature_events()
    meta = [{"id": 0, "name": "A"}, {"id": 1, "name": "B"}, {"id": 2, "name": "C"}]
    weights = {"c_transmission": 2.0, "venture_revenue": 0.5,
               "nonexistent_col": 99.0}
    feats = {f["id"]: f for f in measure.agent_features(events, meta, None, weights)}
    a0 = feats[0]
    # a0: Y_external=4, c_transmission=2, venture_revenue=100
    #   Y_composite = 4 + 2.0*2 + 0.5*100 = 58 (存在しない列キーは無視)
    assert a0["venture_revenue"] == 100
    assert a0["Y_composite"] == 58.0
    # 重み対象が全て 0/欠損の agent2 → Y_composite == Y_external
    a2 = feats[2]
    assert a2["Y_composite"] == float(a2["Y_external"])


def test_r2_traits_extra_targets():
    # 完全線形の Y_composite に対して R^2 ≈ 1、既定出力は不変
    t1 = [0.1, 0.9, 0.4, 0.7, 0.2, 0.6, 0.95, 0.33]
    feats = []
    for i, a in enumerate(t1):
        feats.append({"id": i, "t1": a, "Y_external": a, "Y_internal": a,
                      "Y_composite": 2.0 * a + 1.0})
    base = measure.r2_traits(feats, ["t1"])
    assert set(base) == {"external", "internal", "trait_names"}   # 既定は不変
    ext = measure.r2_traits(feats, ["t1"], extra_targets=("Y_composite",))
    assert "Y_composite" in ext
    assert abs(ext["Y_composite"]["r2"] - 1.0) < 1e-9
    # external/internal は base と一致
    assert ext["external"] == base["external"]
