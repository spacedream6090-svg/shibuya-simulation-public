"""エコー/自己反復の計測(第70バッチ IDEA①)のテスト。

方針(R1 の鉄則を継承):
- **観測側のみ**: プロンプト 1 バイト不変・L1 イベント追加ゼロ・LLM 呼数不変・乱数ゼロ。
  `observer.echo.enabled` を切り替えても L1 は完全に一致する(変わるのは L2 の 5 列だけ)。
- 値は**手計算と一致**させる(Jaccard・反復率・新規伝播の除外規則)。
- ランタイム(L2)と事後解析(measure/stream)で**同じ判定**になることを固定する。
- resume: `_echo_state` が checkpoint round-trip で復元される(L2 常設列のため必須)。

検証は mock のみ(実 LLM 禁止)。
"""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from society.config import load_config
from society.engine import checkpoint
from society.engine.simulation import Simulation
from society.observer import echo as echo_mod
from society.observer import measure as m
from society.observer import stream as st

REPO_ROOT = Path(__file__).resolve().parents[1]


def _ev(step, agent_id, kind, **payload):
    return {"step": step, "sim_min": step, "agent_id": agent_id, "kind": kind,
            "x": 0.0, "y": 0.0, "rng_stream": "", "llm_call_id": None,
            "payload": payload}


def _sim(tmp_path, name, n=10, steps=8, **ov):
    dot = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
           "run.seed=42", "model.backend=mock", "observer.snapshot_every=144",
           f"run.out_dir={(tmp_path / 'runs').as_posix()}"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _l2_cols(sim):
    return pq.read_table(sim.out_dir / "l2_metrics.parquet").column_names


# --------------------------------------------------------------------------- #
# 1) 純ヘルパ(手計算一致)
# --------------------------------------------------------------------------- #
def test_ngram_and_jaccard_hand_computed():
    assert echo_mod.ngram_set("abcde", 4) == {"abcd", "bcde"}
    assert echo_mod.ngram_set("ab", 4) == {"ab"}      # n 未満は全体を 1 要素
    assert echo_mod.ngram_set("", 4) == set()
    # 完全一致 = 1.0 / 共通ゼロ = 0.0
    assert echo_mod.jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert echo_mod.jaccard({"a"}, {"b"}) == 0.0
    # |A∩B|=1, |A∪B|=3 → 1/3
    assert abs(echo_mod.jaccard({"a", "b"}, {"b", "c"}) - 1 / 3) < 1e-12
    # 片方が空 = 比較材料なし → 0.0(類似とみなさない)
    assert echo_mod.jaccard(set(), {"a"}) == 0.0


# --------------------------------------------------------------------------- #
# 2) echo_novelty(事後解析・純関数)の手計算一致
# --------------------------------------------------------------------------- #
def test_echo_max_is_one_for_ten_identical_utterances():
    """同一文を 10 回積むと echo_max_run = 1.0・エコー率も 1 回目を除き 1.0。"""
    events = [_ev(i, 1, "speak", text="今日は人が多いね") for i in range(10)]
    out = m.echo_novelty(events)
    assert out["echo_max_run"] == 1.0
    assert out["n_utterances"] == 10
    # 初回は「窓内の再出」でも「直前との類似」でもない → 9 件がエコー
    assert out["n_echo_utterances"] == 9
    assert out["echo_utterance_rate"] == round(9 / 10, 6)
    assert out["self_similarity_mean"] == 1.0


def test_echo_max_low_for_all_distinct_utterances():
    """全部違う文なら echo_max_run = 1/n(低値)・エコー 0 件・自己類似も低い。"""
    events = [_ev(i, 1, "speak", text=f"まったく別の話題{i}です{i}{i}")
              for i in range(10)]
    out = m.echo_novelty(events)
    assert out["echo_max_run"] == round(1 / 10, 6)
    assert out["n_echo_utterances"] == 0
    assert out["echo_utterance_rate"] == 0.0
    assert out["self_similarity_mean"] < 0.6      # 閾値未満(エコー判定されない)


def test_paraphrase_is_counted_as_echo():
    """『微妙な言い換えで同テーマ連投』(完全一致ではない)がエコーとして立つ。"""
    a = "この時間の街はいつも混んでいて歩きにくい"
    b = "この時間の街はいつも混んでいて歩きづらい"      # 語尾だけ変えた言い換え
    out = m.echo_novelty([_ev(0, 1, "speak", text=a),
                          _ev(1, 1, "speak", text=b)])
    assert out["n_echo_utterances"] == 1
    assert out["self_similarity_mean"] >= 0.6


def test_min_utterances_gate_avoids_degenerate_one_shot():
    """発話 1 件だけの個体は echo_max の対象外(1 件で 1.0 になる退化を防ぐ)。"""
    out = m.echo_novelty([_ev(0, 1, "speak", text="ひとこと")])
    assert out["echo_max_run"] == 0.0


def test_transmission_novel_excludes_only_same_speaker_reuse():
    """同一話者の再送出だけを落とし、他者からの再伝播は落とさない。"""
    events = [
        _ev(0, 2, "transmission", item_id="X", **{"from": 1}, channel="face"),
        _ev(1, 3, "transmission", item_id="X", **{"from": 1}, channel="face"),  # 1 の再送出=エコー
        _ev(2, 4, "transmission", item_id="X", **{"from": 2}, channel="face"),  # 別人=新規
        _ev(3, 5, "transmission", item_id="Y", **{"from": 1}, channel="face"),  # 別語=新規
    ]
    out = m.echo_novelty(events)
    assert out["n_transmission"] == 4
    assert out["n_transmission_novel"] == 3
    assert out["transmission_novel_rate"] == round(3 / 4, 6)


def test_transmission_novel_window_expires():
    """窓(既定 144 step)を越えた再送出はエコーではなく新規に戻る。"""
    events = [
        _ev(0, 2, "transmission", item_id="X", **{"from": 1}, channel="face"),
        _ev(143, 3, "transmission", item_id="X", **{"from": 1}, channel="face"),
        _ev(300, 4, "transmission", item_id="X", **{"from": 1}, channel="face"),
    ]
    out = m.echo_novelty(events)
    assert out["n_transmission_novel"] == 2      # step0 と step300 が新規


def test_media_transmission_is_always_novel():
    """from<0(メディア発)は『話者の自己反復』ではないので常に新規扱い。"""
    events = [_ev(i, 2, "transmission", item_id="X", **{"from": -1},
                  channel="news") for i in range(3)]
    assert m.echo_novelty(events)["n_transmission_novel"] == 3


def test_adopt_novel_requires_two_distinct_senders():
    """1 人が 2 回繰り返して成立した採用は adopt の分子から外れる。"""
    solo = [
        _ev(0, 2, "transmission", item_id="X", **{"from": 1}, channel="face"),
        _ev(1, 2, "transmission", item_id="X", **{"from": 1}, channel="face"),
        _ev(2, 2, "label_adopt", item_id="X", text="w"),
    ]
    out = m.echo_novelty(solo)
    assert out["n_label_adopt"] == 1 and out["n_label_adopt_novel"] == 0
    duo = solo[:1] + [
        _ev(1, 2, "transmission", item_id="X", **{"from": 3}, channel="face"),
        _ev(2, 2, "label_adopt", item_id="X", text="w"),
    ]
    out2 = m.echo_novelty(duo)
    assert out2["n_label_adopt_novel"] == 1
    assert out2["adopt_novel_rate"] == 1.0


# --------------------------------------------------------------------------- #
# 3) item_cascades の新列(既存キーは不変・並記のみ)
# --------------------------------------------------------------------------- #
def test_item_cascades_size_novel_is_appended_without_touching_size():
    events = [
        _ev(0, 1, "vocab_coin", item_id="X", text="foo"),
        _ev(1, 2, "transmission", item_id="X", **{"from": 1}, channel="face"),
        _ev(2, 3, "transmission", item_id="X", **{"from": 2}, channel="face"),
        _ev(3, 4, "transmission", item_id="X", **{"from": 1}, channel="face"),
    ]
    x = {c["item_id"]: c for c in m.item_cascades(events)}["X"]
    # 既存キーは従来値のまま(ID-U3: 既存列は 1 バイトも変えない)
    assert x["size"] == 3 and x["n_adopters"] == 3 and x["depth"] == 2
    # 新列: from=1 の 2 回目だけがエコー → 2 件が新規
    assert x["size_novel"] == 2
    assert x["echo_share"] == round(1.0 - 2 / 3, 6)


# --------------------------------------------------------------------------- #
# 4) L2 常設列(既定 ON)と OFF スイッチ
# --------------------------------------------------------------------------- #
def test_echo_columns_present_by_default(tmp_path):
    sim = _sim(tmp_path, "e_on")
    sim.run()
    cols = _l2_cols(sim)
    for c in echo_mod.COLUMNS:
        assert c in cols, f"常設のはずの echo 列が無い: {c}"


def test_echo_disabled_removes_columns_and_keeps_l1_identical(tmp_path):
    """observer.echo.enabled=false は 5 列とも消える。L1 と LLM 呼数は完全一致。"""
    on = _sim(tmp_path, "e2_on")
    on.run()
    off = _sim(tmp_path, "e2_off", **{"observer.echo.enabled": "false"})
    off.run()
    assert _l1(on) == _l1(off), "echo の ON/OFF で L1 が変わっている(観測側のみのはず)"
    assert on.llm.calls == off.llm.calls
    assert [c for c in _l2_cols(off) if c in echo_mod.COLUMNS] == []
    assert sorted(c for c in _l2_cols(on) if c in echo_mod.COLUMNS) == \
        sorted(echo_mod.COLUMNS)


def test_echo_l2_values_are_deterministic(tmp_path):
    """同一設定 2 ラン で L2 の echo 列がバイト一致(乱数ゼロ)。"""
    a = _sim(tmp_path, "e3_a")
    a.run()
    b = _sim(tmp_path, "e3_b")
    b.run()
    ta = pq.read_table(a.out_dir / "l2_metrics.parquet").to_pydict()
    tb = pq.read_table(b.out_dir / "l2_metrics.parquet").to_pydict()
    for c in echo_mod.COLUMNS:
        assert ta[c] == tb[c], f"{c} が決定論でない"


def test_runtime_echo_matches_posthoc_definition(tmp_path):
    """ランタイム(L2 最終行)と事後解析の判定が同じ規則で動く。

    窓の取り方が違う量(echo_max)は比較せず、**窓の中で決まる量**でもなく
    「ラン全体 ⊂ 1 窓」に収まる短いランを使って伝播の新規性を厳密に突き合わせる。
    """
    sim = _sim(tmp_path, "e4", steps=24)     # 24 step < window 144 = 1 窓に収まる
    sim.run()
    events = [{"step": e.step, "sim_min": e.sim_min, "agent_id": e.agent_id,
               "kind": e.kind, "payload": e.payload} for e in sim.logger.events]
    post = m.echo_novelty(events)
    l2 = pq.read_table(sim.out_dir / "l2_metrics.parquet").to_pydict()
    assert l2["transmission_novel"][-1] == post["n_transmission_novel"]
    assert l2["transmission_novel_rate"][-1] == post["transmission_novel_rate"]
    assert l2["echo_utterance_rate"][-1] == post["echo_utterance_rate"]
    assert l2["self_similarity_mean"][-1] == post["self_similarity_mean"]


def test_stream_and_measure_agree(tmp_path):
    """ストリーミング版と純関数版が同一出力(analyze.py はストリーミング版を使う)。"""
    sim = _sim(tmp_path, "e5", steps=24)
    sim.run()
    run_dir = str(sim.out_dir)
    events = m.load_events(run_dir)
    assert st.echo_novelty(run_dir) == m.echo_novelty(events)
    assert st.item_cascades(run_dir) == m.item_cascades(events)


# --------------------------------------------------------------------------- #
# 5) resume(常設列なので checkpoint 中央管理が必須)
# --------------------------------------------------------------------------- #
def test_echo_state_survives_checkpoint_round_trip(tmp_path):
    """`_echo_state` が checkpoint round-trip で復元され、`_echo_processed` は 0 に戻る。"""
    sim = _sim(tmp_path, "e6", steps=12)
    sim.run()
    assert getattr(sim, "_echo_state", None) is not None
    before = dict(sim._echo_state)
    path = checkpoint.save(sim, 12, tmp_path / "e6" / "checkpoint" / "ckpt.pkl.gz")

    fresh = _sim(tmp_path, "e6", steps=12)
    checkpoint.load(fresh, path)
    st_new = fresh._echo_state
    assert fresh._echo_processed == 0
    assert st_new["cur_step"] == before["cur_step"]
    assert list(st_new["utt"]) == list(before["utt"])
    assert st_new["trans_total"] == before["trans_total"]
    assert st_new["trans_novel"] == before["trans_novel"]
    assert st_new["echo_n"] == before["echo_n"]
