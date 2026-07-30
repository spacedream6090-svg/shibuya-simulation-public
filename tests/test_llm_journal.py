"""第71バッチ: LLM 入出力ジャーナル / REPLAY fail-fast / run_manifest.json のテスト。

受入基準(docs/plans/dual-mode-observe-verify-plan.md 第71行 / source/dual-mode-requirements.md §5):
  T2 … FREE ラン → 同じキャッシュで cache_mode=replay 再走 → L1 バイト一致 + 呼数一致
  T5 … キャッシュから1レコード削除 → replay が即例外(rng_key=誰のどの step かが判る)
  + ジャーナル完全性(行数 == llm.calls・prompt 全行非空・cached 整合)
  + resume で二重記録しない(checkpoint 後に走ってクラッシュした分を巻き戻す)
  + 既定 ON でも L1 は 1 バイトも変わらない(新規別ファイルだから)
"""
from __future__ import annotations

import gzip
import json
import shutil

import pyarrow.parquet as pq
import pytest

from society.config import load_config
from society.engine import checkpoint, scheduler
from society.engine.simulation import Simulation
from society.llm.cache import CachedLLM
from society.llm.journal import LlmJournal, iter_records
from society.observer import manifest as manifest_mod


# --------------------------------------------------------------------------- #
# 共通ヘルパ
# --------------------------------------------------------------------------- #
def _cfg(name: str, n_steps: int = 48, n_agents: int = 12, **ov):
    dot = ["run.seed=42", f"run.n_agents={n_agents}", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def _run(tmp_path, name: str, n_steps: int = 48, n_agents: int = 12, **ov):
    out = tmp_path / name
    sim = Simulation(_cfg(name, n_steps, n_agents, **ov), out_dir=out)
    summary = sim.run()
    return out, summary


def _l1(run_dir):
    return pq.read_table(run_dir / "l1_events.parquet").to_pylist()


def _journal(run_dir, stem: str = "llm_journal.jsonl.gz"):
    return list(iter_records(run_dir / stem))


class _StubBackend:
    """rng_key をそのまま返す決定論スタブ(test_batch_llm と同流儀)。"""
    name = "stub"

    def __init__(self):
        self.calls: list[str] = []

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        self.calls.append(rng_key)
        return f"resp:{rng_key}"


def _reqs(n):
    return [{"prompt": f"p{i}", "rng_key": f"k{i}", "temperature": 0.7,
             "max_tokens": 64} for i in range(n)]


# --------------------------------------------------------------------------- #
# (1) ジャーナルの完全性
# --------------------------------------------------------------------------- #
def test_journal_covers_every_llm_call(tmp_path):
    """行数 == llm.calls・全行 prompt 非空・seq が 0..N-1 で連番・cached 数が summary と整合。"""
    out, summary = _run(tmp_path, "cov")
    recs = _journal(out)
    assert summary["llm_calls"] > 0, "LLM を1回も呼んでいない(テスト前提が崩れた)"
    assert len(recs) == summary["llm_calls"], \
        f"ジャーナル行数 {len(recs)} != llm.calls {summary['llm_calls']}"
    assert [r["seq"] for r in recs] == list(range(len(recs))), "seq が連番でない"
    assert all(r["prompt"] for r in recs), "prompt が空の行がある(全文記録の欠落)"
    assert all(r["response"] is not None for r in recs)
    assert sum(1 for r in recs if r["cached"]) == summary["llm_cache_hits"], \
        "cached フラグの合計がキャッシュ命中数と一致しない"
    for r in recs:                                  # 呼び出しの同定に要る鍵が全部載っている
        assert len(r["key"]) == 64 and r["rng_key"] and r["backend"] == "mock"
        assert isinstance(r["temperature"], float) and isinstance(r["max_tokens"], int)
        assert isinstance(r["think"], bool)


def test_journal_prompt_is_the_real_prompt(tmp_path):
    """記録された prompt が実際に渡された文字列そのもの(key を再計算して一致を確かめる)。"""
    out, _summary = _run(tmp_path, "key")
    recs = _journal(out)
    llm = Simulation(_cfg("keyprobe", 1), out_dir=tmp_path / "keyprobe").llm
    for r in recs[:20]:
        assert llm._key(r["prompt"], r["temperature"], r["max_tokens"],
                        r["think"]) == r["key"], \
            "記録した prompt/params から cache key を再現できない(全文が欠けている)"


def test_journal_default_on_leaves_l1_and_calls_unchanged(tmp_path):
    """既定 ON でも L1 は 1 バイトも変わらない(= 新規別ファイルであることの固定・R1)。"""
    on, s_on = _run(tmp_path, "jon")
    off, s_off = _run(tmp_path, "joff", **{"model.journal": "false"})
    assert _l1(on) == _l1(off), "journal の有無で L1 が変わっている"
    assert s_on["llm_calls"] == s_off["llm_calls"]
    assert s_on["llm_cache_hits"] == s_off["llm_cache_hits"]
    assert not (off / "llm_journal.jsonl.gz").exists(), "OFF なのにジャーナルが出ている"
    assert (on / "llm_journal.jsonl.gz").exists()


def test_journal_is_valid_multimember_gzip(tmp_path):
    """flush 単位で完結した gzip メンバを追記する = 途中で読んでも常に有効な .gz。"""
    j = LlmJournal(tmp_path / "j.jsonl.gz", flush_records=2)
    for i in range(7):
        j.record(key=f"k{i}" * 16, rng_key=f"p/{i}/0", prompt=f"prompt {i}",
                 response=f"resp {i}", temperature=0.7, max_tokens=64,
                 think=False, cached=False, backend="stub")
    j.close()
    with gzip.open(tmp_path / "j.jsonl.gz", "rt", encoding="utf-8") as f:
        rows = [json.loads(x) for x in f if x.strip()]
    assert [r["seq"] for r in rows] == list(range(7))
    # 3 メンバ(2+2+2)+ close の 1 メンバ(1 件)= gzip マジックが 4 個
    raw = (tmp_path / "j.jsonl.gz").read_bytes()
    assert raw.count(b"\x1f\x8b\x08") == 4, "メンバ境界が期待どおりでない"


def test_journal_tolerates_truncated_tail(tmp_path):
    """クラッシュで最終メンバが切れても、そこまでのレコードは読める(全損させない)。"""
    path = tmp_path / "t.jsonl.gz"
    j = LlmJournal(path, flush_records=2)

    def _rec(i):
        j.record(key="k" * 64, rng_key=f"p/{i}/0", prompt=f"P{i}", response="R",
                 temperature=0.7, max_tokens=64, think=False, cached=False,
                 backend="stub")
    _rec(0)
    _rec(1)                                          # ここで 1 メンバ目が確定
    first_member = path.stat().st_size
    _rec(2)
    _rec(3)
    j.close()
    assert len(list(iter_records(path))) == 4
    raw = path.read_bytes()
    path.write_bytes(raw[:first_member + 10])        # 2 メンバ目をヘッダだけにして中断
    assert len(list(iter_records(path))) == 2, "切れた尻尾で全損している/切れていない"


def test_journal_index_lets_a_fresh_object_continue(tmp_path):
    """index サイドカーがあるので、開き直しても seq が巻き戻らない(.gz を数え直さない)。"""
    p = tmp_path / "i.jsonl.gz"
    j = LlmJournal(p, flush_records=2)
    for i in range(5):
        j.record(key="k" * 64, rng_key=f"p/{i}/0", prompt="P", response="R",
                 temperature=0.7, max_tokens=64, think=False, cached=False,
                 backend="stub")
    j.close()
    j2 = LlmJournal(p, flush_records=2)
    assert j2.seq == 5, "index から続き番号を復元できていない"
    j2.record(key="k" * 64, rng_key="p/9/0", prompt="P", response="R",
              temperature=0.7, max_tokens=64, think=False, cached=False,
              backend="stub")
    j2.close()
    assert [r["seq"] for r in iter_records(p)] == list(range(6))


# --------------------------------------------------------------------------- #
# (1b) generate_many 経路も漏らさない
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("workers", [1, 4])
def test_generate_many_journal_matches_sequential(tmp_path, workers):
    """一括発行のジャーナルは逐次 generate のジャーナルと完全一致(要求順・cached 込み)。"""
    reqs = _reqs(5) + _reqs(2)                       # 重複キーを含める(2件目は cached=True)
    seq_j = LlmJournal(tmp_path / "seq.jsonl.gz", flush_records=3)
    seq = CachedLLM(_StubBackend(), enabled=True, path=tmp_path / "seq.jsonl",
                    journal=seq_j)
    for r in reqs:
        seq.generate(r["prompt"], rng_key=r["rng_key"],
                     temperature=r["temperature"], max_tokens=r["max_tokens"])
    seq_j.close()
    bat_j = LlmJournal(tmp_path / "bat.jsonl.gz", flush_records=3)
    bat = CachedLLM(_StubBackend(), enabled=True, path=tmp_path / "bat.jsonl",
                    journal=bat_j)
    bat.generate_many(reqs, workers=workers)
    bat_j.close()
    assert list(iter_records(tmp_path / "seq.jsonl.gz")) == \
           list(iter_records(tmp_path / "bat.jsonl.gz"))
    assert len(list(iter_records(tmp_path / "bat.jsonl.gz"))) == len(reqs)


def test_batch_llm_run_journal_identical_to_sequential(tmp_path):
    """シム丸ごとでも engine.batch_llm ON/OFF でジャーナルが完全一致(配線の検証)。"""
    off, s_off = _run(tmp_path, "boff", n_steps=144, n_agents=20)
    on, s_on = _run(tmp_path, "bon", n_steps=144, n_agents=20,
                    **{"engine.batch_llm.enabled": "true",
                       "engine.batch_llm.workers": "4"})
    assert s_off["llm_calls"] == s_on["llm_calls"] > 0
    assert _l1(off) == _l1(on)
    assert _journal(off) == _journal(on), "一括発行でジャーナルの並び/内容が変わっている"


def test_router_children_are_journalled(tmp_path):
    """router 配線でも子キャッシュごとにジャーナルが付く(漏らさない)。"""
    out = tmp_path / "routed"
    sim = Simulation(_cfg("routed", 24, 10, **{
        "model.backend": "router",
        "model.router.default.backend": "mock",
        "model.router.purpose.reflect.backend": "mock"}), out_dir=out)
    summary = sim.run()
    # 同一 spec の子は共有 = キャッシュ/ジャーナルとも子の name 別に 1 本
    assert [j.path.name for j in sim._journals] == ["llm_journal.mock.jsonl.gz"]
    recs = _journal(out, "llm_journal.mock.jsonl.gz")
    assert len(recs) == summary["llm_calls"] > 0
    assert all(r["prompt"] for r in recs)


def test_router_distinct_children_get_distinct_journals(tmp_path):
    """異なる子(モデル名が違う)には別ファイルのジャーナルが付く(構築のみ検証)。"""
    sim = Simulation(_cfg("routed2", 1, 4, **{
        "model.backend": "router",
        "model.router.default.backend": "mock",
        "model.router.purpose.reflect.backend": "ollama",
        "model.router.purpose.reflect.name": "qwen3:4b"}), out_dir=tmp_path / "routed2")
    names = sorted(j.path.name for j in sim._journals)
    assert names == ["llm_journal.mock.jsonl.gz",
                     "llm_journal.ollama_qwen3_4b.jsonl.gz"]
    # キャッシュファイル名(子の name 由来)と 1:1 で対応している
    assert names == sorted(f"llm_journal.{c.path.name[len('llm_cache.'):-len('.jsonl')]}"
                           f".jsonl.gz"
                           for c in {id(x): x for x in sim.llm.children.values()}.values())


# --------------------------------------------------------------------------- #
# (2) REPLAY fail-fast
# --------------------------------------------------------------------------- #
def test_t2_replay_reproduces_free_run_bytewise(tmp_path):
    """T2: FREE ラン → 同じ llm_cache で cache_mode=replay 再走 → L1 バイト一致 + 呼数一致。"""
    free, s_free = _run(tmp_path, "free")
    rep = tmp_path / "replay"
    rep.mkdir(parents=True)
    shutil.copy(free / "llm_cache.jsonl", rep / "llm_cache.jsonl")   # キャッシュは run 間共有可
    sim = Simulation(_cfg("free", 48, 12, **{"model.cache_mode": "replay"}),
                     out_dir=rep)
    s_rep = sim.run()
    assert _l1(free) == _l1(rep), "REPLAY の L1 が FREE と一致しない"
    assert s_free["llm_calls"] == s_rep["llm_calls"], "呼数が一致しない"
    assert s_rep["llm_cache_hits"] == s_rep["llm_calls"], \
        "REPLAY なのに新規推論が混ざっている(全件キャッシュ命中のはず)"
    # ジャーナルは REPLAY 側でも全文が残る(cached=True で)
    jf, jr = _journal(free), _journal(rep)
    assert [r["prompt"] for r in jf] == [r["prompt"] for r in jr]
    assert [r["response"] for r in jf] == [r["response"] for r in jr]
    assert all(r["cached"] for r in jr) and not any(r["cached"] for r in jf)


def test_t5_replay_fails_fast_on_missing_record(tmp_path):
    """T5: キャッシュから1レコード削除 → 即例外。誰のどの step かがメッセージに出る。"""
    free, _s = _run(tmp_path, "t5free")
    broken = tmp_path / "t5broken"
    broken.mkdir(parents=True)
    lines = (free / "llm_cache.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) > 4
    del lines[len(lines) // 2]
    (broken / "llm_cache.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    sim = Simulation(_cfg("t5free", 48, 12, **{"model.cache_mode": "replay"}),
                     out_dir=broken)
    with pytest.raises(RuntimeError) as ei:
        sim.run()
    msg = str(ei.value)
    assert "replay" in msg
    assert "rng_key=" in msg and "key=" in msg
    rng_key = msg.split("rng_key=")[1].split()[0]
    assert len(rng_key.split("/")) >= 3, f"rng_key に agent/step が無い: {rng_key}"


def test_replay_generate_many_fails_before_any_inference(tmp_path):
    """一括発行でもフェーズ1で落ちる = 1件も新規推論しない(決定論的な fail 位置)。"""
    be = _StubBackend()
    llm = CachedLLM(be, enabled=True, path=tmp_path / "c.jsonl", mode="replay")
    with pytest.raises(RuntimeError) as ei:
        llm.generate_many(_reqs(4), workers=4)
    assert be.calls == [], "REPLAY なのに backend が呼ばれた(フォールバックの混入)"
    assert "rng_key=k0" in str(ei.value), "最初の未命中要求で落ちていない(非決定的)"
    assert "4 件が未命中" in str(ei.value)


def test_replay_hits_do_not_call_backend(tmp_path):
    """全件命中なら REPLAY は backend を一度も呼ばない(再生専用であることの固定)。"""
    path = tmp_path / "c.jsonl"
    warm = CachedLLM(_StubBackend(), enabled=True, path=path)
    for r in _reqs(3):
        warm.generate(r["prompt"], rng_key=r["rng_key"],
                      temperature=r["temperature"], max_tokens=r["max_tokens"])
    be = _StubBackend()
    replay = CachedLLM(be, enabled=True, path=path, mode="replay")
    got = replay.generate_many(_reqs(3), workers=2)
    assert be.calls == []
    assert [g[2] for g in got] == [True, True, True]


def test_replay_without_cache_is_a_startup_error(tmp_path):
    """model.cache=false と replay は矛盾 → 起動時に落とす(黙って free に落とさない)。"""
    with pytest.raises(ValueError, match="矛盾"):
        Simulation(_cfg("bad", 1, 4, **{"model.cache": "false",
                                        "model.cache_mode": "replay"}),
                   out_dir=tmp_path / "bad")


def test_unknown_cache_mode_is_a_startup_error(tmp_path):
    with pytest.raises(ValueError, match="cache_mode"):
        Simulation(_cfg("bad2", 1, 4, **{"model.cache_mode": "strict"}),
                   out_dir=tmp_path / "bad2")


def test_free_mode_is_the_default_and_unchanged(tmp_path):
    """既定は free = 従来動作そのまま(明示 free と完全一致)。"""
    a, sa = _run(tmp_path, "dflt")
    b, sb = _run(tmp_path, "expl", **{"model.cache_mode": "free"})
    assert _l1(a) == _l1(b) and sa["llm_calls"] == sb["llm_calls"]


# --------------------------------------------------------------------------- #
# (1c) resume で二重記録しない
# --------------------------------------------------------------------------- #
def test_resume_journal_has_no_duplicate_records(tmp_path):
    """checkpoint 後にさらに走ってクラッシュ → resume でジャーナルが巻き戻り、二重記録しない。

    checkpoint(step20)を書いた後 step28 まで走ってから落ちた状況を作る。素朴な追記実装だと
    step20..27 の呼び出しが「クラッシュ前の分」+「再走した分」で 2 回記録され、seq も重複する。
    """
    straight, _s = _run(tmp_path, "jstraight", n_steps=40, n_agents=20)
    d = tmp_path / "jresumed"
    ov = {"observer.checkpoint_every": 20}
    sim1 = Simulation(_cfg("jresumed", 40, 20, **ov), out_dir=d)
    for step in range(20):
        scheduler.run_step(sim1, step)
    checkpoint.save(sim1, 20, d / "checkpoint" / "ckpt-000020.pkl.gz")
    sim1.logger.flush_segment()
    for step in range(20, 28):                    # checkpoint 後の「失われる分」
        scheduler.run_step(sim1, step)
    for j in sim1._journals:
        j.flush()                                 # クラッシュ直前まで書けていた状態
    over = len(_journal(d))
    assert over > 0
    sim2 = Simulation(_cfg("jresumed", 40, 20, **ov), out_dir=d)
    sim2.run(resume_from=d)
    js, jr = _journal(straight), _journal(d)
    assert [r["seq"] for r in jr] == list(range(len(jr))), "seq が重複/巻き戻っている"
    assert len(jr) == len(js), f"ジャーナル件数が straight と不一致: {len(jr)} vs {len(js)}"
    assert [r["prompt"] for r in jr] == [r["prompt"] for r in js]
    assert _l1(d) == _l1(straight), "resume の L1 が straight と不一致"


def test_checkpoint_carries_journal_mark(tmp_path):
    """checkpoint に確定点が載り、load で巻き戻る(空回り防止の直接検証)。"""
    d = tmp_path / "mark"
    sim = Simulation(_cfg("mark", 24, 10, **{"observer.checkpoint_every": 12}),
                     out_dir=d)
    for step in range(12):
        scheduler.run_step(sim, step)
    p = checkpoint.save(sim, 12, d / "checkpoint" / "ckpt-000012.pkl.gz")
    import gzip as _gz
    import pickle
    with _gz.open(p, "rb") as f:
        blob = pickle.loads(f.read())
    marks = blob["runtime"]["llm_journal"]
    assert "llm_journal.jsonl.gz" in marks
    assert marks["llm_journal.jsonl.gz"]["records"] == sim._journals[0].seq > 0
    assert marks["llm_journal.jsonl.gz"]["bytes"] == \
        (d / "llm_journal.jsonl.gz").stat().st_size


# --------------------------------------------------------------------------- #
# (3) run_manifest.json
# --------------------------------------------------------------------------- #
def test_manifest_written_with_required_fields(tmp_path):
    out, _s = _run(tmp_path, "man", n_steps=12, n_agents=6)
    man = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert man["schema"] == manifest_mod.SCHEMA
    assert set(man["git"]) == {"sha", "branch", "dirty"}
    assert len(man["config_sha256"]) == 64 and len(man["config_determinism_sha256"]) == 64
    assert len(man["event_schema_sha256"]) == 64
    assert man["run"]["seed"] == 42 and man["run"]["n_agents"] == 6
    assert man["run"]["n_steps"] == 12
    assert man["model"]["backend"] == "mock"
    assert man["model"]["cache_mode"] == "free" and man["model"]["journal"] is True
    assert man["started_at"] and man["started_at_epoch"] > 0
    assert "run" in man and "start_date" in man["run"]
    assert man["files"]["llm_journal"] == ["llm_journal.jsonl.gz"]
    # 全スイッチ状態(真偽値リーフの機械採取)= 既知のトグルが漏れなく載る
    tg = man["toggles"]
    for key in ("model.cache", "model.journal", "observer.echo.enabled",
                "observer.run_manifest", "relations.enabled"):
        assert key in tg, f"toggles に {key} が無い"
    assert tg["model.cache"] is True


def test_manifest_config_sha_tracks_config(tmp_path):
    """設定を1つ変えれば config_sha256 が変わる(事後の設定すり替えが露見する)。"""
    a, _ = _run(tmp_path, "cs1", n_steps=6, n_agents=4)
    b, _ = _run(tmp_path, "cs2", n_steps=6, n_agents=4,
                **{"relations.enabled": "true"})
    ma = json.loads((a / "run_manifest.json").read_text(encoding="utf-8"))
    mb = json.loads((b / "run_manifest.json").read_text(encoding="utf-8"))
    assert ma["config_sha256"] != mb["config_sha256"]
    # 再計算で一致すること(記録が config の実体を指していることの固定)
    assert manifest_mod.config_sha256(
        _cfg("cs1", 6, 4)) == ma["config_sha256"]


def test_manifest_keeps_history_when_out_dir_is_reused(tmp_path):
    """同じ out_dir で作り直しても前回分を history に畳んで残す(来歴を失わない)。"""
    out = tmp_path / "hist"
    Simulation(_cfg("hist", 6, 4), out_dir=out).run()
    first = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert "history" not in first
    Simulation(_cfg("hist", 12, 4), out_dir=out).run()
    second = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert len(second["history"]) == 1
    assert second["history"][0]["n_steps"] == 6
    assert second["run"]["n_steps"] == 12


def test_manifest_can_be_disabled(tmp_path):
    out, _s = _run(tmp_path, "noman", n_steps=6, n_agents=4,
                   **{"observer.run_manifest": "false"})
    assert not (out / "run_manifest.json").exists()


def test_manifest_does_not_change_l1(tmp_path):
    on, s_on = _run(tmp_path, "mon", n_steps=48)
    off, s_off = _run(tmp_path, "moff", n_steps=48,
                      **{"observer.run_manifest": "false"})
    assert _l1(on) == _l1(off) and s_on["llm_calls"] == s_off["llm_calls"]


# --------------------------------------------------------------------------- #
# (4) 観測専用であることの静的固定
# --------------------------------------------------------------------------- #
def test_journal_has_no_read_path_in_the_simulator():
    """シム本体がジャーナルを**読む**経路を作らない(記録は観測専用・R1)。

    REPLAY が読むのは llm_cache.jsonl であってジャーナルではない。ここが破られると
    「観測がシムを変える」ので、静的検査で固定する。
    """
    from pathlib import Path
    import society
    root = Path(society.__file__).resolve().parent
    offenders = []
    for p in root.rglob("*.py"):
        if p.name == "journal.py" and p.parent.name == "llm":
            continue
        text = p.read_text(encoding="utf-8")
        if "iter_records" in text:
            offenders.append(str(p))
    assert not offenders, f"シム本体にジャーナル読み出しがある: {offenders}"
