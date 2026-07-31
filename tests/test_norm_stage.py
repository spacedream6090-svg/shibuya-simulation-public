"""規範化ステージ検出器 + コホートタグ + 下方因果(第74バッチ IDEA③ / Part E)のテスト。

方針(R1 の鉄則を継承):
- **観測側のみ**: プロンプト 1 バイト不変・L1 イベント追加ゼロ・LLM 呼数不変・乱数ゼロ。
  `labeling.norm_stage.enabled` を切り替えても L1 は完全に一致する(変わるのは L2 の列だけ)。
- 4 段階(S1 初出 / S2 他者引用 / S3 定冠詞化 / S4 制度化)を**合成発話**で固定する。
  mock の発話テンプレートには定冠詞相当・合意参照の表現が無いので、実発火の検証は
  合成イベント列で行い、mock ランでは S1/S2 が立つことだけを確かめる(正直な限界)。
- **coiner と institutionalizer の分離**が正しく出ることを固定する。
- 検出語彙は conf(`labeling.norm_stage.markers`)がデータの単一の源: 表を空にすると
  S3/S4 が 0 件になる = コードに語彙が埋まっていないことの証明。
- resume: `_norm_state` が checkpoint round-trip で復元される。

検証は mock のみ(実 LLM 禁止)。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq

from society.config import load_config
from society.engine import checkpoint
from society.engine.simulation import Simulation
from society.observer import measure as m
from society.observer import norms as N
from society.observer import stream as st

REPO_ROOT = Path(__file__).resolve().parents[1]

DEF = ["例の", "いつもの"]
AGR = ["さっき決めた", "決まりだから"]
MARKERS = {"definite": DEF, "agreement": AGR}


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


# 4 段階が順に立つ合成イベント列(語 "ゆる坂"・coiner=1)。
def _four_stage_events():
    return [
        _ev(1, 1, "label_coin", item_id="i1", text="ゆる坂"),
        _ev(1, 1, "speak", text="ここをゆる坂って呼ぼう。"),        # 本人の使用=S2 にならない
        _ev(3, 2, "speak", text="ゆる坂、いい名前だね。"),          # S2(他者引用)
        _ev(5, 3, "speak", text="例のゆる坂で待ってる。"),          # S3(定冠詞化)
        _ev(9, 4, "speak", text="さっき決めたとおり、ゆる坂に集合ね。"),  # S4(制度化)
    ]


# --------------------------------------------------------------------------- #
# 1) 純ヘルパ(手計算一致)
# --------------------------------------------------------------------------- #
def test_definite_before_and_agreement_in():
    assert N.definite_before("例のゆる坂で待つ", "ゆる坂", DEF, 0)
    assert N.definite_before("例の「ゆる坂」で待つ", "ゆる坂", DEF, 2)   # 括弧 1 文字を吸収
    assert not N.definite_before("例の「ゆる坂」で待つ", "ゆる坂", DEF, 0)
    assert not N.definite_before("ゆる坂で待つ", "ゆる坂", DEF, 2)
    assert not N.definite_before("例のゆる坂で待つ", "ゆる坂", [], 2)    # マーカー空
    # 語の 2 個目の出現にだけマーカーが付く場合も拾う
    assert N.definite_before("ゆる坂、そう例のゆる坂", "ゆる坂", DEF, 0)
    assert N.agreement_in("さっき決めたとおり", AGR)
    assert not N.agreement_in("さっき決めたとおり", [])
    assert not N.agreement_in("なんとなく行こう", AGR)


# --------------------------------------------------------------------------- #
# 2) 4 段階の検出(合成発話で固定)
# --------------------------------------------------------------------------- #
def test_four_stages_in_order():
    res = m.norm_stages(_four_stage_events(), MARKERS)
    (w,) = res["words"]
    assert w["word"] == "ゆる坂" and w["coiner"] == 1
    assert (w["s1_step"], w["s2_step"], w["s3_step"], w["s4_step"]) == (1, 3, 5, 9)
    assert (w["s2_agent"], w["s3_agent"], w["s4_agent"]) == (2, 3, 4)
    assert w["stage"] == 4
    s = res["summary"]
    assert (s["n_stage2"], s["n_stage3"], s["n_stage4"], s["max_stage"]) == (1, 1, 1, 4)
    assert s["steps_to_agreement_median"] == 8      # 9 - 1
    assert s["steps_to_quote_median"] == 2          # 3 - 1


def test_coiner_and_institutionalizer_are_separate_roles():
    res = m.norm_stages(_four_stage_events(), MARKERS)
    s = res["summary"]
    assert s["coiners"] == [1] and s["institutionalizers"] == [4]
    assert s["n_institutionalized"] == 1
    assert s["n_institutionalizer_is_other"] == 1
    assert s["institutionalizer_distinct_share"] == 1.0
    # 同一人物が制度化した場合は「別人率」が 0 になる
    ev = _four_stage_events()
    ev[-1] = _ev(9, 1, "speak", text="さっき決めたとおり、ゆる坂に集合ね。")
    s2 = m.norm_stages(ev, MARKERS)["summary"]
    assert s2["institutionalizers"] == [1]
    assert s2["n_institutionalizer_is_other"] == 0
    assert s2["institutionalizer_distinct_share"] == 0.0


def test_no_stage_skipping_and_no_reverse_order():
    # S2 が無いまま S3/S4 の形が出ても記録しない(飛び段の禁止)
    ev = [_ev(1, 1, "label_coin", item_id="i1", text="ゆる坂"),
          _ev(2, 1, "speak", text="さっき決めた例のゆる坂に行く。")]   # 本人だけ
    (w,) = m.norm_stages(ev, MARKERS)["words"]
    assert w["stage"] == 1 and w["s2_step"] is None and w["s3_step"] is None
    # coin より前の発話は語がまだ存在しないので何も立たない(逆順の禁止)
    ev2 = [_ev(1, 2, "speak", text="例のゆる坂で待ってる。"),
           _ev(5, 1, "label_coin", item_id="i1", text="ゆる坂")]
    (w2,) = m.norm_stages(ev2, MARKERS)["words"]
    assert w2["stage"] == 1 and w2["s1_step"] == 5


def test_same_utterance_can_cascade_two_stages():
    # 他者が「例の〈語〉」と言えば 1 発話で S2 と S3 が同時に立つ
    ev = [_ev(1, 1, "label_coin", item_id="i1", text="ゆる坂"),
          _ev(4, 2, "speak", text="例のゆる坂に行こう。")]
    (w,) = m.norm_stages(ev, MARKERS)["words"]
    assert (w["s2_step"], w["s3_step"], w["stage"]) == (4, 4, 3)
    assert (w["s2_agent"], w["s3_agent"]) == (2, 2)


def test_empty_marker_table_kills_stage3_and_4():
    res = m.norm_stages(_four_stage_events(), {"definite": [], "agreement": []})
    (w,) = res["words"]
    assert w["stage"] == 2 and w["s3_step"] is None and w["s4_step"] is None
    assert res["summary"]["n_stage3"] == 0 and res["summary"]["n_stage4"] == 0


def test_stage2_from_transmission_event():
    # 発話テキストが無くても transmission の from != coiner で S2 が立つ
    ev = [_ev(1, 1, "label_coin", item_id="i1", text="ゆる坂"),
          _ev(2, 9, "transmission", item_id="i1", **{"from": 1}),   # coiner 発=立たない
          _ev(4, 9, "transmission", item_id="i1", **{"from": 7})]   # 他者発=立つ
    (w,) = m.norm_stages(ev, MARKERS)["words"]
    assert (w["s2_step"], w["s2_agent"], w["s2_source"]) == (4, 7, "transmission")


def test_min_word_len_filters_short_words():
    ev = [_ev(1, 1, "label_coin", item_id="i1", text="坂"),
          _ev(2, 2, "speak", text="坂に行く")]
    assert m.norm_stages(ev, MARKERS, min_word_len=2)["words"] == []
    assert len(m.norm_stages(ev, MARKERS, min_word_len=1)["words"]) == 1


def test_first_coin_wins_and_media_flag():
    ev = [_ev(1, -1, "label_coin", item_id="i1", text="ゆる坂", media=True),
          _ev(3, 5, "label_coin", item_id="i2", text="ゆる坂")]
    (w,) = m.norm_stages(ev, MARKERS)["words"]
    assert w["s1_step"] == 1 and w["media"] is True and w["coiner"] == -1


# --------------------------------------------------------------------------- #
# 3) 規範候補(Part E(2)): 成立 step は「ステージ到達」と「利用者数到達」の遅い方
# --------------------------------------------------------------------------- #
def test_norm_candidates_threshold_and_established_step():
    ev = _four_stage_events()
    res = m.norm_stages(ev, MARKERS, usage=True)
    # 使用者は 1(coin+発話), 2, 3, 4 の 4 人
    users = {a for _, a in res["usage"]["ゆる坂"]}
    assert users == {1, 2, 3, 4}
    # 利用者 2 人到達 = step 3 / ステージ 2 到達 = step 3 → 成立 step 3
    (c,) = N.norm_candidates(res, 2, 2, res["usage"])
    assert c["established_step"] == 3 and c["n_users"] == 4
    # 利用者 4 人を要求すると step 9 まで待つ(ステージ 2 は既に立っている)
    (c4,) = N.norm_candidates(res, 2, 4, res["usage"])
    assert c4["established_step"] == 9
    # 到達しない閾値なら候補ゼロ
    assert N.norm_candidates(res, 2, 5, res["usage"]) == []
    # ステージ 4 を要求 → 成立 step は S4 の step
    (c9,) = N.norm_candidates(res, 4, 2, res["usage"])
    assert c9["established_step"] == 9 and c9["institutionalizer"] == 4


# --------------------------------------------------------------------------- #
# 4) OFF / ON の不変条件(R1)
# --------------------------------------------------------------------------- #
def test_off_l1_identical_and_no_l2_columns(tmp_path):
    off = _sim(tmp_path, "n_off", steps=24)
    off.run()
    on = _sim(tmp_path, "n_on", steps=24, **{"labeling.norm_stage.enabled": "true"})
    on.run()
    assert _l1(off) == _l1(on), "観測トグルで L1 が動いてはならない"
    assert off.llm.calls == on.llm.calls, "LLM 呼数は不変"
    assert "norm_stage_max" not in _l2_cols(off)
    assert "norm_stage_max" in _l2_cols(on)
    assert getattr(off, "_norm_state", None) is None   # OFF は状態を生やさない


def test_off_leaves_no_state_and_analysis_still_works(tmp_path):
    """enabled=false のランでも **解析側**は保存済み L1 から同じ判定を再現する。"""
    sim = _sim(tmp_path, "n_post", n=12, steps=48)
    sim.run()
    ev = m.load_events(str(sim.out_dir))
    pure = m.norm_stages(ev, MARKERS)
    strm = st.norm_stages(str(sim.out_dir), MARKERS)
    assert pure["words"] == strm["words"] and pure["summary"] == strm["summary"]
    assert pure["summary"]["n_words"] > 0, "mock は造語する"


def test_runtime_l2_matches_post_hoc(tmp_path):
    sim = _sim(tmp_path, "n_match", n=12, steps=48,
               **{"labeling.norm_stage.enabled": "true"})
    sim.run()
    col = pq.read_table(sim.out_dir / "l2_metrics.parquet").to_pydict()["norm_stage_max"]
    cfg = N.cfg_of(sim)
    post = st.norm_stages(str(sim.out_dir), cfg["markers"], cfg["min_word_len"],
                          cfg["max_gap"])
    assert col[-1] == post["summary"]["max_stage"]


def test_mock_reaches_stage1_and_2_only(tmp_path):
    """正直な限界の固定: mock の語彙には定冠詞相当・合意参照が無いので S3/S4 は 0 件。"""
    sim = _sim(tmp_path, "n_mock", n=16, steps=144)
    sim.run()
    res = st.norm_stages(str(sim.out_dir), MARKERS)
    s = res["summary"]
    assert s["n_stage1"] > 0 and s["n_stage2"] > 0
    assert s["n_stage3"] == 0 and s["n_stage4"] == 0
    assert s["max_stage"] == 2


# --------------------------------------------------------------------------- #
# 5) resume(checkpoint round-trip)
# --------------------------------------------------------------------------- #
def test_norm_state_survives_checkpoint(tmp_path):
    sim = _sim(tmp_path, "n_ckpt", n=12, steps=36,
               **{"labeling.norm_stage.enabled": "true"})
    sim.run()
    assert getattr(sim, "_norm_state", None) is not None
    path = tmp_path / "ck.pkl.gz"
    checkpoint.save(sim, 36, path)
    sim2 = _sim(tmp_path, "n_ckpt2", n=12, steps=36,
                **{"labeling.norm_stage.enabled": "true"})
    checkpoint.load(sim2, path)
    assert sim2._norm_state["order"] == sim._norm_state["order"]
    assert sim2._norm_state["max_stage"] == sim._norm_state["max_stage"]
    assert sim2._norm_processed == 0 and sim2._norm_cache is None


def test_norm_resume_matches_straight(tmp_path):
    """一気 120step と 60+resume で L2 の norm 列が全行一致(累積列なので中央管理が必須)。"""
    from society.engine import checkpoint as _ckpt
    from society.engine import scheduler as _sched

    def _mk(name, steps):
        dot = ["run.seed=42", "run.n_agents=12", f"run.n_steps={steps}",
               f"run.name={name}", "model.backend=mock",
               "labeling.norm_stage.enabled=true", "observer.checkpoint_every=60"]
        return Simulation(load_config(dot), out_dir=tmp_path / name)

    straight = _mk("nr_straight", 120)
    straight.run()
    assert straight._norm_state["max_stage"] >= 2, "検証が空回り(語が段階に到達していない)"

    d = tmp_path / "nr_resumed"
    sim1 = _mk("nr_resumed", 60)
    for step in range(60):
        _sched.run_step(sim1, step)
    _ckpt.save(sim1, 60, d / "checkpoint" / "ckpt-000060.pkl.gz")
    sim1.logger.flush_segment()
    sim2 = _mk("nr_resumed", 120)
    sim2.run(resume_from=d)

    cols = ["norm_stage_max"]
    a = pq.read_table(straight.out_dir / "l2_metrics.parquet").to_pylist()
    b = pq.read_table(d / "l2_metrics.parquet").to_pylist()
    ka = [{c: r.get(c) for c in cols} for r in a]
    kb = [{c: r.get(c) for c in cols} for r in b]
    assert ka and ka == kb, "norm_stage の resume==straight が崩れている"


# --------------------------------------------------------------------------- #
# 6) コホートタグ(Part E1)
# --------------------------------------------------------------------------- #
def test_first_presence_from_l1_only(tmp_path):
    sim = _sim(tmp_path, "n_coh", n=8, steps=24)
    sim.run()
    ev = m.load_events(str(sim.out_dir))
    pure = m.first_presence(ev)
    strm = st.first_presence(str(sim.out_dir))
    assert pure == strm
    # pool OFF では全員が step 0 = コホートは 1 群しかできない(正直な限界)
    assert set(pure.values()) == {0}


def test_first_presence_detects_late_entrants():
    ev = [_ev(0, 1, "stay", node="a"), _ev(0, 2, "stay", node="a"),
          _ev(5, 1, "stay", node="a"), _ev(7, 3, "stay", node="b"),
          _ev(9, -1, "presence_change", day=1)]
    assert m.first_presence(ev) == {1: 0, 2: 0, 3: 7}


# --------------------------------------------------------------------------- #
# 7) scripts/analyze_norms.py(閾値必須・完走)
# --------------------------------------------------------------------------- #
def _run_cli(*args):
    # Windows の既定コンソール(cp932)では日本語レポートの照合ができないので、
    # 子プロセスの標準出力を UTF-8 に固定してから読む(スクリプト側の挙動は不変)。
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    return subprocess.run([sys.executable,
                           str(REPO_ROOT / "scripts" / "analyze_norms.py"), *args],
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace", env=env)


def test_analyze_norms_requires_thresholds(tmp_path):
    sim = _sim(tmp_path, "n_cli", n=10, steps=48)
    sim.run()
    r = _run_cli(str(sim.out_dir))
    assert r.returncode != 0, "閾値未指定はエラーで止まらなければならない(事前登録 U-10)"
    assert "--norm-stage" in (r.stderr or "")
    r2 = _run_cli(str(sim.out_dir), "--norm-stage", "2")
    assert r2.returncode != 0 and "--norm-threshold" in (r2.stderr or "")


def test_analyze_norms_runs_end_to_end(tmp_path):
    sim = _sim(tmp_path, "n_cli2", n=12, steps=96)
    sim.run()
    out = tmp_path / "norms.json"
    r = _run_cli(str(sim.out_dir), "--norm-stage", "2", "--norm-threshold", "2",
                 "--mc", "200", "--out", str(out))
    assert r.returncode == 0, r.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["params"]["norm_stage"] == 2
    assert data["runs"][0]["stages"]["n_words"] > 0
    # pool OFF = コホートが 1 群 → 群間比較は成立しない(正直に usable=False)
    assert data["downward"]["usable"] is False
    assert "群間比較ができない" in r.stdout


def test_two_sample_perm_hand_checked():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import analyze_norms as an
    # 完全分離(a=[0,0,0], b=[1,1,1]): 全列挙 C(6,3)=20 通りのうち |diff| が最大の
    # 並びは 2 通り(元の並びとその反転)= p = 2/20 = 0.1
    t = an.two_sample_perm([0, 0, 0], [1, 1, 1])
    assert t["method"] == "exhaustive" and t["p"] == 0.1 and t["diff"] == 1.0
    # 定数列は検定不能(p=1.0)
    assert an.two_sample_perm([1, 1], [1, 1])["method"] == "degenerate_constant"
    # 片側が空なら p=None(捏造しない)
    assert an.two_sample_perm([], [1, 2])["p"] is None
