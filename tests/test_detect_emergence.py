"""P1: 創発の後付けテキスト検出 scripts/detect_emergence.py の純関数テスト。

検出器はすべて「dict のリストを受ける純関数」なので、parquet I/O 無しに
小さな dict 列を渡して検証する(house style: tests/test_worldview_analysis.py に倣い
scripts/ を path 追加してからモジュールを import)。

検証:
  1. fiction 命中(未知カタカナ→候補化)+ ホワイトリスト除外(実在名は候補にしない)
  2. coined 分離(正規造語は fiction から除外)
  3. norms 各パターン(べき/してはいけない/普通/マナー/みんな〜する/英語 should)
  4. attention の窓・人数条件(24step 内 3人 → 命中、窓外や人数不足は非命中)
  5. attention の provenance(coined vs organic)+ extract_text_records の reflect 条件
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))          # scripts/ は package ではない

import detect_emergence as de                        # noqa: E402


def _rec(step, agent, text, kind="speak"):
    return {"step": step, "agent": agent, "kind": kind, "text": text}


# --------------------------------------------------------------------------- #
# 1. fiction 命中 + ホワイトリスト除外
# --------------------------------------------------------------------------- #
def test_fiction_hit_and_whitelist_exclusion():
    names = ["渋谷ヒカリエ", "観世能楽堂", "山手線"]
    idx = de.NameIndex(names)
    recs = [
        # 未知のカタカナ固有名詞(2回)= 命中してほしい
        _rec(0, 1, "今日はメガロマートで買い物した"),
        _rec(5, 2, "メガロマートは安いね"),
        # 実在名(渋谷ヒカリエ)は候補化しない
        _rec(3, 3, "渋谷ヒカリエで待ち合わせ"),
    ]
    out = de.detect_fiction(recs, idx)
    cands = {f["candidate"]: f for f in out}
    assert "メガロマート" in cands, "未知カタカナが fiction 候補に出るべき"
    assert cands["メガロマート"]["count"] == 2
    assert cands["メガロマート"]["n_agents"] == 2
    # 実在名は候補に出ない(部分一致で既知判定)
    assert "渋谷ヒカリエ" not in cands
    # 出現多い順ソート(先頭が count>=2)
    assert out[0]["candidate"] == "メガロマート"


def test_fiction_ascii_and_partial_match():
    names = ["スターバックス"]
    idx = de.NameIndex(names)
    recs = [
        _rec(0, 1, "新しい店 ZonkleMart に行った"),          # 未知 ASCII
        _rec(1, 2, "スターバックスコーヒーで休憩"),           # 実在名 ⊂ 候補 → 既知
    ]
    out = de.detect_fiction(recs, idx)
    cands = {f["candidate"] for f in out}
    assert "ZonkleMart" in cands
    # 「スターバックスコーヒー」は実在名「スターバックス」を含むので除外される
    assert not any("スターバックス" in c for c in cands)


# --------------------------------------------------------------------------- #
# 2. coined 分離
# --------------------------------------------------------------------------- #
def test_coined_excluded_from_fiction():
    idx = de.NameIndex([])                            # ホワイトリスト空
    recs = [
        _rec(0, 1, "ワクワクポイントを貯めよう"),
        _rec(2, 2, "ワクワクポイント最高"),
        _rec(3, 3, "ナゾナゾ屋台に行きたい"),
    ]
    # 「ワクワクポイント」は正規造語 → fiction から除外、ナゾナゾ屋台は残る
    out = de.detect_fiction(recs, idx, coined={"ワクワクポイント"})
    cands = {f["candidate"] for f in out}
    assert "ワクワクポイント" not in cands
    assert "ナゾナゾ" in cands


def test_collect_coined_from_events():
    events = [
        {"step": 4, "agent": 7, "kind": "vocab_coin",
         "payload": {"item_id": "v1", "text": "シブヤメーター"}},
        {"step": 9, "agent": 3, "kind": "label_coin",
         "payload": {"item_id": "l2", "text": "朝活クラブ"}},
        {"step": 1, "agent": 2, "kind": "speak", "payload": {"text": "ふつうの発話"}},
    ]
    coined = de.collect_coined(events)
    texts = {c["text"] for c in coined}
    assert texts == {"シブヤメーター", "朝活クラブ"}


# --------------------------------------------------------------------------- #
# 3. norms 各パターン
# --------------------------------------------------------------------------- #
def test_norms_patterns():
    recs = [
        _rec(0, 1, "ゴミは分別すべきだ"),
        _rec(1, 2, "ここでタバコを吸ってはいけない"),
        _rec(2, 3, "挨拶するのが普通でしょう"),
        _rec(3, 4, "電車ではマナーを守ろう"),
        _rec(4, 5, "みんなでルールを守っている"),
        _rec(5, 6, "You should recycle your trash"),
        _rec(6, 7, "今日はいい天気だね"),           # 規範ではない → 非検出
    ]
    out = de.detect_norms(recs)
    got = {n["agent"]: n["label"] for n in out}
    assert got[1] == "義務 (〜べき)"
    assert got[2] == "禁止 (してはいけない)"
    assert got[3].startswith("社会通念")
    assert got[4] == "マナー"
    assert 5 in got                                   # みんな〜 or ルール遵守 いずれか命中
    assert got[6] == "duty (en)"
    assert 7 not in got                               # 非規範は拾わない
    # 1発話1件(重複しない)
    assert len(out) == len({n["agent"] for n in out})


# --------------------------------------------------------------------------- #
# 4. attention の窓・人数条件
# --------------------------------------------------------------------------- #
def test_attention_window_and_agent_threshold():
    # 3人が 24step 以内に同語(2字漢字複合語)→ 命中
    recs_hit = [
        _rec(0, 1, "議会に行こう"),
        _rec(10, 2, "議会が楽しみ"),
        _rec(20, 3, "議会の準備をする"),
    ]
    out = de.detect_attention(recs_hit, window_steps=24, min_agents=3)
    terms = {a["term"]: a for a in out}
    assert "議会" in terms
    assert terms["議会"]["n_agents"] == 3
    assert terms["議会"]["provenance"] == "organic"

    # 同語だが窓の外(0,10,40)→ 24step 窓に3人そろわない → 非命中
    recs_wide = [
        _rec(0, 1, "議会に行こう"),
        _rec(10, 2, "議会が楽しみ"),
        _rec(40, 3, "議会の準備をする"),
    ]
    assert not de.detect_attention(recs_wide, window_steps=24, min_agents=3)

    # 2人だけ(=人数不足)→ 非命中
    recs_few = [_rec(0, 1, "議会だ"), _rec(5, 2, "議会だね")]
    assert not de.detect_attention(recs_few, window_steps=24, min_agents=3)


def test_attention_provenance_and_exclusion():
    recs = [
        _rec(0, 1, "ワクワクポイント貯める"),
        _rec(3, 2, "ワクワクポイントいいね"),
        _rec(6, 3, "ワクワクポイント最高"),
        # 地名(除外トークン)は 3人でも attention に出ない
        _rec(0, 1, "渋谷に行く"),
        _rec(2, 2, "渋谷なう"),
        _rec(4, 3, "渋谷たのしい"),
    ]
    coined_toks = de.whitelist_tokens(["ワクワクポイント"])
    excl = de.whitelist_tokens(["渋谷"])
    out = de.detect_attention(recs, exclude_tokens=excl, coined_tokens=coined_toks,
                              window_steps=24, min_agents=3)
    terms = {a["term"]: a for a in out}
    assert terms.get("ワクワクポイント", {}).get("provenance") == "coined"
    assert "渋谷" not in terms                         # ホワイトリスト断片は除外


def test_attention_tie_order_is_deterministic():
    """W4-D 検収で発見した決定論バグの回帰(2026-08-08 修正)。

    (n_agents, occurrences, first_step) が同値の語は、修正前は occ dict の挿入順
    (= set(_tokenize) 反復順 = PYTHONHASHSEED 依存)で並び、同一ランを2回通すだけで
    順序が変わっていた(実測 297 件中 51 件)。term を最終タイブレークに置いたので、
    入力レコードの並びを逆にしても出力は同一・同値語は term 昇順で並ぶ。"""
    def _both(term_a, term_b):
        recs = []
        for term in (term_a, term_b):
            recs += [_rec(0, 1, f"{term}が気になる"),
                     _rec(5, 2, f"{term}いいね"),
                     _rec(10, 3, f"{term}最高")]
        return recs

    fwd = de.detect_attention(_both("パルソニア", "ガルベリン"),
                              window_steps=24, min_agents=3)
    rev = de.detect_attention(_both("ガルベリン", "パルソニア"),
                              window_steps=24, min_agents=3)
    assert fwd == rev, "入力順(=dict挿入順)が出力順に漏れている"
    tied = [a["term"] for a in fwd
            if (a["n_agents"], a["occurrences"], a["first_step"]) ==
               (fwd[0]["n_agents"], fwd[0]["occurrences"], fwd[0]["first_step"])]
    assert tied == sorted(tied), "同値語が term 昇順になっていない"
    assert {"パルソニア", "ガルベリン"} <= {a["term"] for a in fwd}


# --------------------------------------------------------------------------- #
# 5. extract_text_records の reflect 条件(written_back のみ)
# --------------------------------------------------------------------------- #
def test_extract_text_records_reflect_and_agent_filter():
    events = [
        {"step": 0, "agent": 1, "kind": "speak", "payload": {"text": "やあ"}},
        {"step": 1, "agent": 2, "kind": "reflect",
         "payload": {"belief": "定着した信念", "written_back": True}},
        {"step": 2, "agent": 3, "kind": "reflect",
         "payload": {"belief": "書き戻さない信念", "written_back": False}},
        {"step": 3, "agent": -1, "kind": "speak", "payload": {"text": "世界イベント"}},
        {"step": 4, "agent": 4, "kind": "dm", "payload": {"text": "DM本文", "to": 5}},
    ]
    recs = de.extract_text_records(events)
    texts = {(r["agent"], r["text"]) for r in recs}
    assert (1, "やあ") in texts
    assert (2, "定着した信念") in texts                # written_back=True は採用
    assert (4, "DM本文") in texts
    assert all(r["text"] != "書き戻さない信念" for r in recs)   # False は除外
    assert all(r["agent"] >= 0 for r in recs)         # 世界イベント(-1)は除外
