"""第78バッチ: 状態ハッシュチェーン(observer.state_hash)のテスト。

受入基準(docs/plans/source/dual-mode-instructions.md Phase 0(4)・Phase 2 T1/T6):
  T1 … 同一 seed の 2 ラン(mock)でチェーンが**完全一致**
  T6 … generate_many の並列度(engine.batch_llm.workers)を 1 と 4 に変えても一致
  改竄検知 … state_hash.jsonl の 1 レコードを書き換えるとチェーン検証が落ちる
  R1     … 既定 OFF ではファイルを作らず L1 もバイト一致(書くだけ・読む経路なし)
"""
from __future__ import annotations

import json
import math
import time

from society.config import load_config
from society.engine.simulation import Simulation
from society.observer import state_hash as SH


def _cfg(name: str, n_steps: int = 24, n_agents: int = 12, **ov):
    dot = ["run.seed=42", f"run.n_agents={n_agents}", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def _run(tmp_path, name: str, n_steps: int = 24, n_agents: int = 12, **ov):
    sim = Simulation(_cfg(name, n_steps, n_agents, **ov), out_dir=tmp_path / name)
    sim.run()
    return sim, tmp_path / name


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _hashes(out_dir):
    return [r["hash"] for r in SH.read_chain(out_dir / SH.FILENAME)]


# --------------------------------------------------------------------------- #
# (0) 正準化の単体
# --------------------------------------------------------------------------- #
def test_fixed_decimal_normalization():
    assert SH.fixed(1.0, 6) == "1.000000"
    assert SH.fixed(-0.0, 6) == "0.000000"          # -0.0 と 0.0 を同一視
    assert SH.fixed(0.0, 6) == "0.000000"
    assert SH.fixed(1 / 3, 3) == "0.333"
    assert SH.fixed(float("nan"), 6) == "nan"
    assert SH.fixed(float("inf"), 6) == "inf"
    assert SH.fixed(float("-inf"), 6) == "-inf"
    assert SH.fixed("abc", 6) == "null"             # 欠測を 0 と偽らない
    # 桁を変えれば別の文字列(= ハッシュが変わる)
    assert SH.fixed(math.pi, 2) != SH.fixed(math.pi, 6)


def test_chain_is_order_sensitive():
    a = SH.chain_next("", "aa")
    b = SH.chain_next(a, "bb")
    assert SH.chain_next(SH.chain_next("", "bb"), "aa") != b


def test_cfg_defaults_off():
    assert SH.cfg_from_block(None) == {"enabled": False, "interval": 1,
                                       "float_digits": 6}
    assert SH.cfg_of_config(load_config())["enabled"] is False


# --------------------------------------------------------------------------- #
# (1) R1: 既定 OFF
# --------------------------------------------------------------------------- #
def test_off_writes_nothing_and_l1_identical(tmp_path):
    a, dir_a = _run(tmp_path, "sh_off")
    assert not (dir_a / SH.FILENAME).exists()
    b, _ = _run(tmp_path, "sh_off2", **{"observer.state_hash.enabled": "false"})
    assert _l1(a) == _l1(b)


def test_on_does_not_change_l1(tmp_path):
    """ON にしても L1 は 1 バイトも変わらない(書くだけ・読む経路なし)。"""
    a, _ = _run(tmp_path, "sh_l1_off")
    b, dir_b = _run(tmp_path, "sh_l1_on", **{"observer.state_hash.enabled": "true"})
    assert _l1(a) == _l1(b)
    assert (dir_b / SH.FILENAME).exists()


# --------------------------------------------------------------------------- #
# (2) T1: 同一 seed の 2 ランでチェーン完全一致
# --------------------------------------------------------------------------- #
def test_t1_same_seed_two_runs_identical_chain(tmp_path):
    _a, dir_a = _run(tmp_path, "sh_t1a", **{"observer.state_hash.enabled": "true"})
    _b, dir_b = _run(tmp_path, "sh_t1b", **{"observer.state_hash.enabled": "true"})
    ha, hb = _hashes(dir_a), _hashes(dir_b)
    assert len(ha) == 24 and ha == hb
    assert SH.compare(dir_a / SH.FILENAME, dir_b / SH.FILENAME)["identical"]


def test_different_seed_diverges(tmp_path):
    """片側判定の確認: 世界が違えばチェーンは必ず割れる(分岐 step も出る)。"""
    _a, dir_a = _run(tmp_path, "sh_s42", **{"observer.state_hash.enabled": "true"})
    _b, dir_b = _run(tmp_path, "sh_s43", **{"observer.state_hash.enabled": "true",
                                            "run.seed": 43})
    rep = SH.compare(dir_a / SH.FILENAME, dir_b / SH.FILENAME)
    assert rep["identical"] is False
    assert rep["diverged_at"] is not None


# --------------------------------------------------------------------------- #
# (3) T6: generate_many の並列度を変えても一致
# --------------------------------------------------------------------------- #
def test_t6_workers_1_vs_4_identical_chain(tmp_path):
    ov = {"observer.state_hash.enabled": "true",
          "engine.batch_llm.enabled": "true"}
    _a, dir_a = _run(tmp_path, "sh_w1", n_steps=48,
                     **{**ov, "engine.batch_llm.workers": 1})
    _b, dir_b = _run(tmp_path, "sh_w4", n_steps=48,
                     **{**ov, "engine.batch_llm.workers": 4})
    ha, hb = _hashes(dir_a), _hashes(dir_b)
    assert ha and ha == hb, "並列度でチェーンが割れた(逐次格納の決定論が壊れている)"


# --------------------------------------------------------------------------- #
# (4) 改竄検知
# --------------------------------------------------------------------------- #
def test_verify_chain_accepts_intact_file(tmp_path):
    _a, dir_a = _run(tmp_path, "sh_ok", **{"observer.state_hash.enabled": "true"})
    rep = SH.verify_chain(SH.read_chain(dir_a / SH.FILENAME))
    assert rep["ok"] is True and rep["n"] == 24


def test_tampering_state_field_is_detected(tmp_path):
    _a, dir_a = _run(tmp_path, "sh_tamper1",
                     **{"observer.state_hash.enabled": "true"})
    recs = SH.read_chain(dir_a / SH.FILENAME)
    recs[10]["state"] = "0" * 64                   # 1 レコードの state だけ書き換える
    rep = SH.verify_chain(recs)
    assert rep["ok"] is False and rep["first_bad"] == 10


def test_tampering_hash_field_is_detected(tmp_path):
    _a, dir_a = _run(tmp_path, "sh_tamper2",
                     **{"observer.state_hash.enabled": "true"})
    recs = SH.read_chain(dir_a / SH.FILENAME)
    recs[5]["hash"] = "f" * 64                     # チェーン値だけ書き換える
    rep = SH.verify_chain(recs)
    assert rep["ok"] is False and rep["first_bad"] == 5


def test_dropping_a_record_is_detected(tmp_path):
    """1 行削除もチェーンが繋がらなくなるので検出できる。"""
    _a, dir_a = _run(tmp_path, "sh_drop", **{"observer.state_hash.enabled": "true"})
    recs = SH.read_chain(dir_a / SH.FILENAME)
    del recs[7]
    assert SH.verify_chain(recs)["ok"] is False


def test_broken_json_line_raises(tmp_path):
    _a, dir_a = _run(tmp_path, "sh_broken",
                     **{"observer.state_hash.enabled": "true"})
    path = dir_a / SH.FILENAME
    path.write_text(path.read_text(encoding="utf-8") + "{not json\n",
                    encoding="utf-8")
    try:
        SH.read_chain(path)
    except ValueError:
        return
    raise AssertionError("壊れた行を黙って無視している(欠測を隠している)")


# --------------------------------------------------------------------------- #
# (5) interval と性能
# --------------------------------------------------------------------------- #
def test_interval_thins_records(tmp_path):
    _a, dir_a = _run(tmp_path, "sh_iv", n_steps=24,
                     **{"observer.state_hash.enabled": "true",
                        "observer.state_hash.interval": 6})
    recs = SH.read_chain(dir_a / SH.FILENAME)
    assert [r["step"] for r in recs] == [5, 11, 17, 23]
    assert SH.verify_chain(recs)["ok"] is True


def test_overhead_is_bounded(tmp_path):
    """性能: 24step / 12体 mock で ON のオーバーヘッドを実測(壊滅的でないことの固定)。"""
    t0 = time.perf_counter()
    _run(tmp_path, "sh_perf_off", n_steps=24)
    off = time.perf_counter() - t0
    t1 = time.perf_counter()
    _run(tmp_path, "sh_perf_on", n_steps=24,
         **{"observer.state_hash.enabled": "true"})
    on = time.perf_counter() - t1
    assert on < off * 3.0 + 5.0, f"state_hash の overhead が大きすぎる: {off=} {on=}"


def test_canonical_state_is_stable_and_sorted(tmp_path):
    """同じ world state からは何度でも同じ正準 JSON が出る(反復順に依存しない)。"""
    sim = Simulation(_cfg("sh_canon", n_steps=1), out_dir=tmp_path / "sh_canon")
    a = SH.canonical_state(sim, 0, 6)
    b = SH.canonical_state(sim, 0, 6)
    assert a == b
    blob = json.loads(a)
    ids = [row[0] for row in blob["agents"]]
    assert ids == sorted(ids), "エージェントが id 昇順で直列化されていない"
