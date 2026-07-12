"""D16: チェックポイント / 途中再開の一致テスト(合格条件そのもの)。

合格条件:
- 「40step 一気」==「20step で ckpt → 新プロセス相当で load → 40step まで」の l1_events が
  完全一致(全カラム・全行。少なくとも kind/step/agent_id/payload)。l2/l3 も一致。
- checkpoint 無効(既定)時に挙動・出力が従来と完全一致(part を一切作らない)。
- straight run 中に checkpoint を挟んでも(flush_segment→finalize 結合)出力が不変。
- scenario shock_closure を跨いだ resume(封鎖中に save→load して復元後も封鎖が効く)。
"""
from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from society.config import load_config
from society.engine import checkpoint, scheduler
from society.engine.simulation import Simulation


def _cfg(name: str, n_steps: int, **ov):
    dot = ["run.seed=42", "run.n_agents=20", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def _rows(run_dir: Path, stem: str = "l1_events") -> list[dict]:
    return pq.read_table(Path(run_dir) / f"{stem}.parquet").to_pylist()


def _run_straight(tmp_path, name: str, n_steps: int, **ov) -> Path:
    d = tmp_path / name
    Simulation(_cfg(name, n_steps, **ov), out_dir=d).run()
    return d


def _run_resume(tmp_path, name: str, split: int, total: int, **ov) -> Path:
    """phase1: split step 走らせて ckpt を書き finalize せず中断(クラッシュ相当)。
       phase2: 新 Simulation で load → total まで走らせ finalize(part 結合)。"""
    d = tmp_path / name
    every = {"observer.checkpoint_every": split}
    # --- phase 1(中断) ---
    sim1 = Simulation(_cfg(name, split, **every, **ov), out_dir=d)
    for step in range(split):
        scheduler.run_step(sim1, step)
    checkpoint.save(sim1, split, d / "checkpoint" / f"ckpt-{split:06d}.pkl.gz")
    sim1.logger.flush_segment()
    # --- phase 2(途中再開) ---
    sim2 = Simulation(_cfg(name, total, **every, **ov), out_dir=d)
    sim2.run(resume_from=d)
    return d


# --------------------------------------------------------------------------- #
def test_resume_matches_straight_all_layers(tmp_path):
    """一気 40step と 20+resume の l1/l2/l3 が全行・全カラム一致。"""
    straight = _run_straight(tmp_path, "straight", 40)
    resumed = _run_resume(tmp_path, "resumed", 20, 40)
    a = _rows(straight, "l1_events")
    b = _rows(resumed, "l1_events")
    assert len(a) == len(b), f"l1 行数不一致: {len(a)} vs {len(b)}"
    assert a == b, "l1_events が byte 級で不一致"
    for stem in ("l2_metrics", "l3_snapshots"):
        assert _rows(straight, stem) == _rows(resumed, stem), f"{stem} 不一致"


def test_checkpoint_disabled_is_byte_identical(tmp_path):
    """既定(checkpoint_every=0)は part を作らず、明示 0 とも完全一致(従来挙動の保存)。"""
    a = _run_straight(tmp_path, "def", 30)                       # 既定 0
    b = _run_straight(tmp_path, "zero", 30, **{"observer.checkpoint_every": 0})
    assert _rows(a) == _rows(b)
    assert not list(a.glob("l1_events.part-*.parquet")), "既定で part が作られている"


def test_segmented_straight_matches_plain(tmp_path):
    """straight run 中に checkpoint を挟んでも(flush_segment→結合)出力が不変。"""
    plain = _run_straight(tmp_path, "plain", 40)                 # every=0
    seg = _run_straight(tmp_path, "seg", 40,
                        **{"observer.checkpoint_every": 10})     # 4 セグメントを結合
    assert _rows(plain, "l1_events") == _rows(seg, "l1_events")
    for stem in ("l2_metrics", "l3_snapshots"):
        assert _rows(plain, stem) == _rows(seg, stem)
    # part は finalize で結合・削除され、canonical だけが残る
    assert not list(seg.glob("l1_events.part-*.parquet"))
    assert (seg / "l1_events.parquet").exists()


def test_resume_across_shock_closure(tmp_path):
    """封鎖中(step 5..35)に step 20 で save→load。復元後も封鎖が効き、l1 が一致する。"""
    ov = {"world.scenario": "shock_closure",
          "world.scenario_params":
              "{at_step: 5, duration_steps: 30, center: [0,0], radius_m: 150}"}
    straight = _run_straight(tmp_path, "sc_straight", 40, **ov)
    resumed = _run_resume(tmp_path, "sc_resume", 20, 40, **ov)
    assert _rows(straight, "l1_events") == _rows(resumed, "l1_events"), \
        "shock_closure を跨いだ resume の l1 が不一致"

    # step 20 の checkpoint を素の sim へ load → 封鎖が復元されていることを検査
    sim = Simulation(_cfg("sc_inspect", 40, **ov), out_dir=tmp_path / "sc_inspect")
    step = checkpoint.load(sim, tmp_path / "sc_resume" / "checkpoint" / "ckpt-000020.pkl.gz")
    assert step == 20
    assert sim.scenario.active and sim.scenario.closed, "封鎖状態が復元されていない"
    n_flags = sum(1 for _u, _v, d in sim.city.graph.edges(data=True) if d.get("closed"))
    assert n_flags == len(sim.scenario.closed) > 0, "closed フラグが city へ再適用されていない"
