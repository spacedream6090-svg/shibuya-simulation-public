"""P2-S6b: LLM一括発行(engine.batch_llm / CachedLLM.generate_many)のテスト。

一括発行は逐次実行と「イベント列・カウンタ・キャッシュ内容」が完全同一であることが
契約(決定論・R1)。並行性は実LLMサーバ向けで、mock では同一経路の逐次になる。
"""
from __future__ import annotations

import json
import time

from society.config import load_config
from society.engine.simulation import Simulation
from society.llm.cache import CachedLLM


class _StubBackend:
    """rng_key をそのまま返す決定論スタブ。呼び出し記録と遅延を持つ。"""
    name = "stub"

    def __init__(self, delay: float = 0.0):
        self.delay = delay
        self.calls: list[str] = []

    def generate(self, prompt, *, rng_key, temperature, max_tokens,
                 think=False):
        if self.delay:
            time.sleep(self.delay)
        self.calls.append(rng_key)
        return f"resp:{rng_key}"


def _reqs(n):
    return [{"prompt": f"p{i}", "rng_key": f"k{i}", "temperature": 0.7,
             "max_tokens": 64} for i in range(n)]


def test_generate_many_matches_sequential(tmp_path):
    """generate_many は逐次 generate の列と応答・counters・キャッシュ・保存が一致。"""
    seq = CachedLLM(_StubBackend(), enabled=True, path=tmp_path / "seq.jsonl")
    got_seq = [seq.generate(r["prompt"], rng_key=r["rng_key"],
                            temperature=r["temperature"],
                            max_tokens=r["max_tokens"]) for r in _reqs(6)]
    bat = CachedLLM(_StubBackend(), enabled=True, path=tmp_path / "bat.jsonl")
    got_bat = bat.generate_many(_reqs(6), workers=1)
    assert got_seq == got_bat
    assert (seq.calls, seq.hits) == (bat.calls, bat.hits)
    assert (tmp_path / "seq.jsonl").read_text(encoding="utf-8") == \
           (tmp_path / "bat.jsonl").read_text(encoding="utf-8")


def test_generate_many_dedupes_like_sequential(tmp_path):
    """同一プロンプト2件: 逐次では2件目がキャッシュ命中。一括でも同じに見える。"""
    dup = _reqs(1) * 2
    seq = CachedLLM(_StubBackend(), enabled=True)
    got_seq = [seq.generate(r["prompt"], rng_key=r["rng_key"],
                            temperature=r["temperature"],
                            max_tokens=r["max_tokens"]) for r in dup]
    bat = CachedLLM(_StubBackend(), enabled=True)
    got_bat = bat.generate_many(dup, workers=4)
    assert got_seq == got_bat
    assert got_bat[1][2] is True                      # 2件目= cached
    assert len(bat.backend.calls) == 1                # 生成は初出の1回だけ
    assert (seq.calls, seq.hits) == (bat.calls, bat.hits)


def test_generate_many_parallel_and_order(tmp_path):
    """workers>1 で並行実行され(壁時間<逐次合計)、結果は要求順を保つ。"""
    be = _StubBackend(delay=0.05)
    llm = CachedLLM(be, enabled=True)
    t0 = time.time()
    got = llm.generate_many(_reqs(8), workers=8)
    elapsed = time.time() - t0
    assert [g[0] for g in got] == [f"resp:k{i}" for i in range(8)]
    assert elapsed < 0.05 * 8                         # 逐次(0.4s)より確実に速い


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _run(tmp_path, name, steps=24, **ov):
    dot = ["run.seed=42", "run.n_agents=30", f"run.n_steps={steps}",
           f"run.name={name}"] + [f"{k}={v}" for k, v in ov.items()]
    sim = Simulation(load_config(dot), out_dir=tmp_path / name)
    sim.run()
    return sim


def _kinds(sim):
    return {e.kind for e in sim.logger.events}


def test_batched_run_is_byte_identical(tmp_path):
    """mock 1日(144step): batch_llm ON(workers=1/4)は OFF と L1 バイト一致。

    朝の計画は起床時刻(朝)に発火するため 24step では通らない——1日回して
    計画・内省の両パスが実際に発火したことを kind で確認してから比較する。"""
    off = _run(tmp_path, "off", steps=144)
    on1 = _run(tmp_path, "on1", steps=144,
               **{"engine.batch_llm.enabled": "true",
                  "engine.batch_llm.workers": "1"})
    on4 = _run(tmp_path, "on4", steps=144,
               **{"engine.batch_llm.enabled": "true",
                  "engine.batch_llm.workers": "4"})
    assert "day_plan" in _kinds(off)                  # 計画パスを実際に通った
    assert "reflect" in _kinds(off)                   # 内省パスを実際に通った
    assert _l1(off) == _l1(on1)
    assert _l1(off) == _l1(on4)
    assert off.llm.calls == on1.llm.calls == on4.llm.calls


def test_batched_reflect_identical_with_agentic_pull(tmp_path):
    """agentic pull ON(1体2呼=recall→本呼の2ラウンド)でも ON=OFF バイト一致。
    BufferSink の遅延放出が逐次のイベント並び(個体毎に recall→reflect)を再現する検証。"""
    pull = {"memory.agentic_pull": "true"}
    off = _run(tmp_path, "poff", steps=144, **pull)
    on = _run(tmp_path, "pon", steps=144,
              **{**pull, "engine.batch_llm.enabled": "true",
                 "engine.batch_llm.workers": "4"})
    assert "memory_recall" in _kinds(off)             # recall ラウンドを実際に通った
    assert _l1(off) == _l1(on)
    assert off.llm.calls == on.llm.calls
