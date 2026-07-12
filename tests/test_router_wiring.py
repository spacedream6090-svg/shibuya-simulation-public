"""合成ルータの配線テスト(第23バッチ M2・simulation.py の backend=router 分岐)。

RouterLLM 単体は tests/test_router.py。ここは load_config → Simulation → run の結合を検証する:
- mock 子だけの router は素の mock と L1 バイト一致(配線が呼数・順序・内容を変えない=R1)
- 呼数集計(llm_calls/llm_cache_hits)が router 経由でも summary に出る
- default 欠落は ValueError
"""
from __future__ import annotations

import json

import pytest

from society.config import load_config
from society.engine.simulation import Simulation


def _cfg(name, n=10, steps=24, extra=()):
    dot = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
           "observer.snapshot_every=144"]
    dot += list(extra)
    return load_config(dot)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def test_router_mock_matches_plain_mock(tmp_path):
    """mock 子だけの router は素の mock と完全同一(L1・呼数とも)。"""
    plain = Simulation(_cfg("plain"), out_dir=tmp_path / "plain")
    plain.run()
    routed = Simulation(_cfg("routed", extra=(
        "model.backend=router",
        "model.router.default.backend=mock",
        "model.router.purpose.reflect.backend=mock",   # 同一 spec=同一子を共有(重複構築しない)
    )), out_dir=tmp_path / "routed")
    routed.run()
    assert _l1(plain) == _l1(routed), "router(mock子)が素の mock と不一致(配線が挙動を変えた)"
    assert plain.llm.calls == routed.llm.calls
    assert plain.llm.hits == routed.llm.hits


def test_router_requires_default(tmp_path):
    with pytest.raises(ValueError):
        Simulation(_cfg("nodef", extra=(
            "model.backend=router",
            "model.router.purpose.reflect.backend=mock",
        )), out_dir=tmp_path / "nodef")


def test_router_cache_file_per_child_name(tmp_path):
    """キャッシュファイルが子の name 別(llm_cache.<name>.jsonl)に作られる。"""
    sim = Simulation(_cfg("cachefile", steps=6, extra=(
        "model.backend=router",
        "model.router.default.backend=mock",
    )), out_dir=tmp_path / "cachefile")
    sim.run()
    assert (tmp_path / "cachefile" / "llm_cache.mock.jsonl").exists()
    assert not (tmp_path / "cachefile" / "llm_cache.jsonl").exists()
