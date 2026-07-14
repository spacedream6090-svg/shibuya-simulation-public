"""A3: 作話の接地率メトリクス scripts/detect_emergence.py の純関数テスト。

reality-levers P2。発話中の固有名詞を 実在名 / シミュ内実在(創発名)/ 作話 の3値に
分類し日次集計する compute_grounding とその周辺を、parquet I/O 抜き(合成発話)で検証する。
既存の test_detect_emergence.py と同じ house style(scripts/ を path 追加して import)。

検証:
  1. 3分類 + 日次集計(実在名/創発名/作話・rate 系の値)
  2. 創発名を作話に混ぜない(本メトリクスの肝)+ 一般語は分母から除外
  3. 0件日の縮退(n_proper=0 → rate=None=データ不足)+ reflect は対象外
  4. collect_sim_names(造語・店名・コミュニティ・制度名の収集)
  5. whitelist_names の組織台帳マージ
  6. parquet 書き出し(schema と None 保持)
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))          # scripts/ は package ではない

import detect_emergence as de                        # noqa: E402


def _rec(step, agent, text, kind="speak"):
    return {"step": step, "agent": agent, "kind": kind, "text": text}


def _real_sim():
    real = de.NameIndex(["スターバックス", "渋谷ヒカリエ", "山手線"])
    sim = de.NameIndex(["メロメロ", "ナゾ屋台"])     # シミュ内で正当に創発した名
    return real, sim


# --------------------------------------------------------------------------- #
# 1. 3分類 + 日次集計
# --------------------------------------------------------------------------- #
def test_three_way_classification_and_daily():
    real, sim = _real_sim()
    recs = [
        # --- day 0 ---
        _rec(0, 1, "スターバックスで休憩した", "speak"),      # 実在名 grounded
        _rec(5, 2, "メガロマートは安いね", "speak"),          # 作話 fiction
        _rec(10, 3, "メロメロを貯めよう", "sns_post"),        # 創発名 sim_coined
        _rec(20, 4, "コーヒーが好き", "speak"),               # 一般語 → 分母から除外
        _rec(50, 5, "ホニャララ団の集会", "reflect"),          # reflect は対象外(除外)
        # --- day 1 (step>=144) ---
        _rec(144, 1, "またメガロマートに行った", "speak"),    # 作話
        _rec(150, 2, "ズンドコ団子が名物だ", "dm"),           # 作話
    ]
    g = de.compute_grounding(recs, real, sim)
    days = {row["day"]: row for row in g["daily"]}

    # reflect は数えない → day0 の発話数は4
    d0 = days[0]
    assert d0["n_utter"] == 4
    # 一般語コーヒーは分母外 → 固有名詞は3(スターバックス/メガロマート/メロメロ)
    assert d0["n_proper"] == 3
    assert d0["n_grounded"] == 1
    assert d0["n_sim_coined"] == 1
    assert d0["n_fiction"] == 1
    assert abs(d0["rate"] - 1 / 3) < 1e-3           # rate は小数4桁に丸め
    assert abs(d0["rate_incl_sim"] - 2 / 3) < 1e-3
    assert abs(d0["fabrication_rate"] - 1 / 3) < 1e-3

    d1 = days[1]
    assert d1["n_proper"] == 2 and d1["n_grounded"] == 0 and d1["n_fiction"] == 2
    assert d1["rate"] == 0.0
    assert d1["fabrication_rate"] == 1.0

    # 全期間: 固有名詞5(実在1/創発1/作話3)、rate=1/5
    tot = g["totals"]
    assert tot["n_proper"] == 5 and tot["n_grounded"] == 1
    assert tot["n_sim_coined"] == 1 and tot["n_fiction"] == 3
    assert abs(tot["rate"] - 0.2) < 1e-3
    assert g["n_utterances"] == 6                      # reflect を除いた発話数


# --------------------------------------------------------------------------- #
# 2. 創発名を作話に混ぜない(肝)+ 一般語除外
# --------------------------------------------------------------------------- #
def test_coined_not_counted_as_fiction():
    real, sim = _real_sim()
    recs = [
        _rec(0, 1, "メロメロを配る", "speak"),         # 創発名
        _rec(1, 2, "メガロマートで買った", "speak"),   # 作話
    ]
    g = de.compute_grounding(recs, real, sim)
    fic = {f["candidate"] for f in g["fiction"]}
    assert "メロメロ" not in fic                       # 創発名は作話リストに出ない
    assert "メガロマート" in fic
    # 分類器の直接確認
    assert de._classify_proper("メロメロ", "katakana", real, sim) == "sim_coined"
    assert de._classify_proper("メガロマート", "katakana", real, sim) == "fiction"
    assert de._classify_proper("スターバックス", "katakana", real, sim) == "grounded"


def test_sim_index_none_falls_back_to_fiction():
    """sim_index=None(創発名の台帳が空)のとき、未知名は素直に作話へ。"""
    real, _ = _real_sim()
    assert de._classify_proper("メロメロ", "katakana", real, None) == "fiction"


# --------------------------------------------------------------------------- #
# 3. 0件日の縮退 + fiction 集約
# --------------------------------------------------------------------------- #
def test_zero_proper_day_is_data_insufficient():
    real, sim = _real_sim()
    recs = [
        _rec(288, 1, "今日はいい天気だね", "speak"),   # 固有名詞なし
        _rec(290, 2, "うれしい気持ちになった", "speak"),  # 固有名詞なし
    ]
    g = de.compute_grounding(recs, real, sim)
    d2 = g["daily"][0]
    assert d2["day"] == 2
    assert d2["n_utter"] == 2 and d2["n_proper"] == 0
    assert d2["rate"] is None                          # データ不足
    assert d2["rate_incl_sim"] is None
    assert d2["fabrication_rate"] is None
    assert g["totals"]["rate"] is None                 # 全期間も分母0 → None


def test_fiction_examples_aggregated_and_sorted():
    real, sim = _real_sim()
    recs = [
        _rec(0, 1, "メガロマートに行く", "speak"),
        _rec(5, 2, "メガロマート最高", "speak"),
        _rec(9, 1, "ズンドコ団子", "speak"),
    ]
    g = de.compute_grounding(recs, real, sim)
    top = g["fiction"][0]
    assert top["candidate"] == "メガロマート"          # 出現多い順
    assert top["count"] == 2 and top["n_agents"] == 2
    assert top["examples"]                             # 文脈例つき


# --------------------------------------------------------------------------- #
# 4. collect_sim_names(創発名の収集)
# --------------------------------------------------------------------------- #
def test_collect_sim_names():
    events = [
        {"step": 1, "agent": 2, "kind": "vocab_coin",
         "payload": {"item_id": "v1", "text": "シブヤメーター"}},
        {"step": 2, "agent": 3, "kind": "venture_open",
         "payload": {"name": "月見屋台", "node": "n1"}},
        {"step": 3, "agent": 4, "kind": "group_found",
         "payload": {"name": "朝活クラブ", "group_id": "g1"}},
        {"step": 4, "agent": 5, "kind": "institution",
         "payload": {"name": "みどりの約束", "norm_text": "..."}},
        {"step": 5, "agent": 6, "kind": "speak",
         "payload": {"text": "ふつうの発話"}},           # 名前を持たない → 無視
    ]
    got = {s["text"] for s in de.collect_sim_names(events)}
    assert got == {"シブヤメーター", "月見屋台", "朝活クラブ", "みどりの約束"}


# --------------------------------------------------------------------------- #
# 5. whitelist_names の組織台帳マージ
# --------------------------------------------------------------------------- #
def test_whitelist_merges_orgs_and_residence():
    map_data = {"pois": [{"name": "スクランブル交差点"}]}
    agents = [{"name": "山田太郎", "work_name": "渋谷区役所",
               "residence_line": "山手線"}]
    orgs = {"companies": [{"name": "(株)ハチ公ラボ"}],
            "schools": [{"name": "渋谷第一小学校"}]}
    names = de.whitelist_names(map_data, agents, orgs)
    for expected in ["スクランブル交差点", "山田太郎", "渋谷区役所", "山手線",
                     "(株)ハチ公ラボ", "渋谷第一小学校"]:
        assert expected in names
    # orgs 省略でも後方互換(落ちない)
    assert "山田太郎" in de.whitelist_names(map_data, agents)


# --------------------------------------------------------------------------- #
# 6. parquet 書き出し(schema と None 保持)
# --------------------------------------------------------------------------- #
def test_write_grounding_parquet(tmp_path):
    import pyarrow.parquet as pq

    real, sim = _real_sim()
    recs = [
        _rec(0, 1, "スターバックスで休憩", "speak"),
        _rec(5, 2, "メガロマートは安い", "speak"),
        _rec(288, 3, "いい天気だね", "speak"),          # day2 = 0件 → rate None
    ]
    g = de.compute_grounding(recs, real, sim)
    out = tmp_path / "panel"
    path = de.write_grounding_parquet("testrun", g, str(out))
    tbl = pq.read_table(path)
    assert tbl.column_names == de._GROUNDING_COLS
    d = tbl.to_pydict()
    # day0 は rate 有り、day2(0件)は None
    by_day = {d["day"][i]: i for i in range(len(d["day"]))}
    assert d["rate"][by_day[0]] is not None
    assert d["rate"][by_day[2]] is None
    assert set(d["run"]) == {"testrun"}
