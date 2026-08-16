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


# --------------------------------------------------------------------------- #
# 第118 レーン B5: 日中熟慮(deliberate)の step 単位 batch 化
#
# 契約は上と同じ「逐次と完全同一」だが、熟慮は朝計画・夜内省と違い **個体間で独立で
# ない**(1 個体の後段が世界へ書いた結果を次の個体の材料収集が読む)。そのため
# scheduler 側は
#   ① LLM を撃たない個体はその場で最後まで走らせる
#   ② 撃つ個体だけを応答待ちで中断 → 一括発行 → id 順に再開
#   ③ 遅らせた後段が他個体を読み書きしうる個体は batch に入れない(安全弁)
#   ④ 観測出力(L1 / L1b)の並びを個体単位で逐次と同じ順へ戻す
# という境界を持つ。以下はその 4 つを固定する。
# --------------------------------------------------------------------------- #
import hashlib

import pyarrow.parquet as pq
import pytest

from society.cognition import deliberate as _deliberate_mod
from society.engine import checkpoint as _checkpoint
from society.engine import scheduler as _scheduler


def _l1b(sim):
    return [dict(r) for r in sim.logger.llm_calls]


def _digest(sim):
    """最終状態の要約(位置・所持金・欲求・語彙・計画)。"""
    return [[a.id, a.node, a.building, a.floor, a.loc, bool(a.sleeping),
             round(float(a.x), 6), round(float(a.y), 6),
             round(float(a.drive), 9), round(float(a.money), 6),
             sorted(a.adopted), len(a.day_plan or [])] for a in sim.agents]


def _cache_mem(sim):
    return sorted(getattr(sim.llm, "_mem", {}).items())


_BATCH_ON = {"engine.batch_llm.enabled": "true"}


def test_deliberate_actually_uses_batch_path(tmp_path):
    """熟慮が **実際に一括経路を通った**ことの機械確認(逐次に落ちていない)。"""
    on = _run(tmp_path, "dbg", steps=144, **{**_BATCH_ON,
                                             "engine.batch_llm.workers": "8"})
    st = on._batch_decide_stats
    assert st["batched"] > 0, "熟慮が 1 件も batch に乗っていない"
    assert st["requests"] == st["batched"]     # 1 個体 1 要求(2 本目は安全弁で除外済み)
    assert st["max_batch"] >= 2, "同 step に 2 件以上まとまった実績が無い"
    assert st["rounds"] > 0 and st["no_llm"] > 0
    assert st["serial_llm"] > 0, "安全弁(逐次退避)の枝が 1 度も踏まれていない"
    assert st["deferred_fallback"] == 0        # 厳密一致の証人
    # OFF 側には統計そのものが生えない(= 一括経路を 1 度も通らない)
    off = _run(tmp_path, "dbg_off", steps=144)
    assert not hasattr(off, "_batch_decide_stats")


def test_batched_deliberate_is_identical_to_sequential(tmp_path):
    """OFF / ON(workers=1)/ ON(workers=8): L1・L1b・キャッシュ・カウンタ・最終状態が一致。"""
    off = _run(tmp_path, "d_off", steps=144)
    on1 = _run(tmp_path, "d_on1", steps=144,
               **{**_BATCH_ON, "engine.batch_llm.workers": "1"})
    on8 = _run(tmp_path, "d_on8", steps=144,
               **{**_BATCH_ON, "engine.batch_llm.workers": "8"})
    assert "llm_deliberate" in _kinds(off)      # 熟慮パスを実際に通った
    for on in (on1, on8):
        assert on._batch_decide_stats["batched"] > 0
        assert _l1(off) == _l1(on)              # L1 イベント列(並びまで)
        assert _l1b(off) == _l1b(on)            # L1b(呼の並び・cached フラグ)
        assert (off.llm.calls, off.llm.hits) == (on.llm.calls, on.llm.hits)
        assert _cache_mem(off) == _cache_mem(on)
        assert _digest(off) == _digest(on)


def test_batched_deliberate_identical_under_parse_failures(tmp_path, monkeypatch):
    """壊れた JSON(= 後退経路)が混ざっても ON=OFF。

    後退経路(routine.decide / _phone)は遅延させると他個体との順序が変わりうる箇所
    なので、**わざと 4 回に 1 回パースを壊して**両経路を突き合わせる。安全弁が効いて
    いれば L1 は一致し、遅延側で後退が起きた回数(deferred_fallback)は記録される。
    """
    real = _deliberate_mod.parse_action

    def flaky(response):
        h = hashlib.blake2b(str(response).encode("utf-8"), digest_size=4).digest()
        return None if h[0] % 4 == 0 else real(response)

    monkeypatch.setattr(_deliberate_mod, "parse_action", flaky)
    off = _run(tmp_path, "f_off", steps=144)
    on = _run(tmp_path, "f_on", steps=144, **{**_BATCH_ON,
                                              "engine.batch_llm.workers": "4"})
    assert "fallback" in _kinds(off), "後退経路(fallback)が 1 件も起きていない"
    assert on._batch_decide_stats["deferred_fallback"] > 0, \
        "遅延した個体で後退経路が 1 度も起きていない(= この試験が無風)"
    assert _l1(off) == _l1(on)
    assert _l1b(off) == _l1b(on)
    assert _digest(off) == _digest(on)


def test_batched_deliberate_identical_with_finals_like_subsystems(tmp_path,
                                                                  monkeypatch):
    """本選相当の下位系 ON + 壊れた JSON でも ON=OFF。

    遅延した個体の後退経路(`routine.decide`)が **他個体へ書きうる**のは宅配の注文・
    共同行動/party の合流先・サービス来店といった下位系が ON のときだけなので、それらを
    まとめて立てた状態で突き合わせる(= 設計注記 ⑤ の残り 1 点を実測で潰す試験)。
    """
    real = _deliberate_mod.parse_action

    def flaky(response):
        h = hashlib.blake2b(str(response).encode("utf-8"), digest_size=4).digest()
        return None if h[0] % 4 == 0 else real(response)

    monkeypatch.setattr(_deliberate_mod, "parse_action", flaky)
    subs = {"delivery.enabled": "true", "joint.enabled": "true",
            "party.enabled": "true", "services.enabled": "true",
            "household.enabled": "true", "diversity.enabled": "true",
            "relations.enabled": "true", "inner_life.enabled": "true"}
    off = _run(tmp_path, "g_off", steps=144, **subs)
    on = _run(tmp_path, "g_on", steps=144, **{**subs, **_BATCH_ON,
                                              "engine.batch_llm.workers": "8"})
    assert on._batch_decide_stats["deferred_fallback"] > 0
    assert _l1(off) == _l1(on)
    assert _l1b(off) == _l1b(on)
    assert _digest(off) == _digest(on)


def test_policy_cache_disables_deliberate_batch(tmp_path):
    """方針キャッシュ ON では熟慮の一括発行を丸ごと外す(全体 LRU は遅延できない)。"""
    pc = {"cognition.policy_cache.enabled": "true"}
    off = _run(tmp_path, "pc_off", steps=144, **pc)
    on = _run(tmp_path, "pc_on", steps=144, **{**pc, **_BATCH_ON})
    assert not hasattr(on, "_batch_decide_stats"), "安全弁が外れている"
    assert _l1(off) == _l1(on)                  # 計画・内省の一括は従来どおり効く


def test_reorder_decide_log_rejects_broken_split():
    """反証: 区間分割が元の並びを覆っていなければ黙って並べ替えず即座に落ちる。"""
    class _Log:
        def __init__(self):
            self.events = ["a", "b", "c"]
            self.llm_calls = []

    class _Sim:
        pass

    sim = _Sim()
    sim.logger = _Log()
    try:
        _scheduler._reorder_decide_log(sim, 0, 0, [[(0, 1, 0, 0), (1, 2, 0, 0)],
                                                   [(2, 3, 0, 0)]])
    except RuntimeError:                        # 覆っていれば通る(下の壊れた分割と対比)
        raise AssertionError("正しい分割で落ちてはいけない")
    sim.logger.events = ["a", "b", "c"]
    with pytest.raises(RuntimeError):
        _scheduler._reorder_decide_log(sim, 0, 0, [[(0, 1, 0, 0), (1, 2, 0, 0)]])


def test_batched_deliberate_resume_equals_straight(tmp_path):
    """batch ON で checkpoint を跨いでも resume == straight(L1 parquet 一致)。"""
    from society.config import load_config as _load

    def _cfg(name, n_steps, **ov):
        dot = ["run.seed=42", "run.n_agents=30", f"run.n_steps={n_steps}",
               f"run.name={name}", "model.backend=mock",
               "engine.batch_llm.enabled=true", "engine.batch_llm.workers=4",
               "observer.checkpoint_every=48"]
        return _load(dot + [f"{k}={v}" for k, v in ov.items()])

    straight = tmp_path / "b_straight"
    Simulation(_cfg("b_straight", 96), out_dir=straight).run()

    d = tmp_path / "b_resumed"
    sim1 = Simulation(_cfg("b_resumed", 48), out_dir=d)
    for step in range(48):
        _scheduler.run_step(sim1, step)
    _checkpoint.save(sim1, 48, d / "checkpoint" / "ckpt-000048.pkl.gz")
    sim1.logger.flush_segment()
    sim2 = Simulation(_cfg("b_resumed", 96), out_dir=d)
    sim2.run(resume_from=d)

    def _rows(run_dir):
        return pq.read_table(run_dir / "l1_events.parquet").to_pylist()

    assert _rows(d) == _rows(straight)


def test_batch_decide_stats_exported_to_summary(tmp_path):
    """第130: batch ON の summary.json に `batch_decide`(deferred_fallback=厳密一致の
    証人)が出る。OFF では属性が生えない=キー自体を出さない(既存 summary と同形)。"""
    on = _run(tmp_path, "stats_on", steps=144,
              **{"engine.batch_llm.enabled": "true",
                 "engine.batch_llm.workers": "4"})
    s_on = json.loads((tmp_path / "stats_on" / "summary.json").read_text(encoding="utf-8"))
    bd = s_on["batch_decide"]
    assert set(bd) == {"batched", "serial_llm", "no_llm", "rounds",
                       "requests", "steps", "max_batch", "deferred_fallback"}
    assert bd["deferred_fallback"] == 0          # mock は常にパース成立=同値の証人
    assert bd["batched"] + bd["serial_llm"] > 0  # LLM を撃つ個体が実在した
    assert bd["batched"] == bd["requests"]       # 一括発行された呼数と整合
    # 属性そのものと一致(丸めや取りこぼしがない)
    assert bd == {k: int(v) for k, v in on._batch_decide_stats.items()}

    off = _run(tmp_path, "stats_off", steps=144)
    s_off = json.loads((tmp_path / "stats_off" / "summary.json").read_text(encoding="utf-8"))
    assert "batch_decide" not in s_off
    assert not hasattr(off, "_batch_decide_stats")
