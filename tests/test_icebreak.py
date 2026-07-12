"""アイスブレイク(初期関係形成)の生成と読み込みのテスト。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

from society.config import load_config
from society.engine.simulation import Simulation

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_icebreak  # noqa: E402


def _personas(n: int = 40) -> list[dict]:
    path = REPO_ROOT / "data" / f"personas_{n}.json"
    return json.loads(path.read_text(encoding="utf-8"))["personas"]


def _build(seed: int = 42, n_similar: int = 2, n_random: int = 1):
    return build_icebreak.build(_personas(), n_similar=n_similar,
                                n_random=n_random, turns=3, seed=seed,
                                mock=True, model="qwen3:8b",
                                host="http://localhost:11434")


def test_pairing_is_deterministic(tmp_path):
    r1 = _build(seed=42)
    r2 = _build(seed=42)
    assert r1["pairs"] == r2["pairs"], "同 seed で出力が一致しない(非決定論)"
    r3 = _build(seed=7)
    assert r1["pairs"] != r3["pairs"], "seed を変えても出力が同じ(乱数が効いていない)"


def test_similar_and_random_counts():
    n = len(_personas())
    res = _build(seed=42, n_similar=2, n_random=1)
    kinds = [p["kind"] for p in res["pairs"]]
    assert "similar" in kinds and "random" in kinds
    # 各エージェントは張り直しで必ず n_random 件の新規ランダムペアを足す。
    assert kinds.count("random") == n * 1
    # 類似ペアは無向 dedupe 済みで、上限(n×n_similar)以下かつ正。
    assert 0 < kinds.count("similar") <= n * 2
    # 各ペアは 3 往復 = 6 発話。
    assert all(len(p["turns"]) == 6 for p in res["pairs"])


def test_simulation_loads_relations_and_contacts(tmp_path):
    res = _build(seed=42)
    ib_path = tmp_path / "icebreak.json"
    ib_path.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")

    cfg = load_config(["run.n_agents=40", "run.n_steps=1", "run.name=ib",
                       "observer.snapshot_every=1",
                       "agents.personas_file=data/personas_40.json"])
    OmegaConf.update(cfg, "agents.icebreak_file", str(ib_path))
    sim = Simulation(cfg, out_dir=tmp_path / "ib")

    pair = res["pairs"][0]
    a = sim.agent_by_id[pair["a"]]
    b = sim.agent_by_id[pair["b"]]
    # 関係台帳が両者に入る
    assert pair["b"] in a.mem.relations, "a の関係台帳に相手が入っていない"
    assert pair["a"] in b.mem.relations, "b の関係台帳に相手が入っていない"
    # SNS の知り合い関係(DM 可)に入る
    assert pair["b"] in sim.net.contacts.get(pair["a"], set())
    # 「初めて話した」エピソードが記憶に入る
    assert any("初めて話した" in ep.text for ep in a.mem.buffer)
