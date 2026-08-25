"""`--resume-accept-config` = 観察ランの運用変更を受理して再開する(2026-08-25)。

正典: docs/plans/llm-budget-respec.md §4 / §6。実装 src/society/engine/checkpoint.py
      (`load(..., accept_config_mismatch=)`)・src/society/engine/simulation.py(`run`)・
      scripts/run.py(CLI フラグ)。

何を解く問題か
--------------
`checkpoint.load` は保存時と現在の config_hash を照合し、不一致なら必ず ValueError を出す
(決定論ガード)。ところが本番の観察ランでは「走行中に呼数 cap / レーン配分 / workers だけを
変えて直近 checkpoint から再開する」という運用が要る。`_VOLATILE_KEYS` は 4 キーだけなので、
この操作は**一律に拒否**されていた。

受入基準
  (1) **既定は 1 バイトも変わらない**: フラグ無しの不一致は従来どおり ValueError で、
      文言も従来のまま(Δt 二重変換の手掛かりを含む)。
  (2) フラグ true = WARNING を 1 行残して続行(stored / current の両ハッシュが出る)。
  (3) config が**一致**しているときはフラグの有無で 1 ビットも差が出ない(警告も出ない)。
  (4) CLI: `--resume-accept-config` が解析され、`--resume` 経路へ配線されている。
  (5) 世界の復元そのものは従来と同一(受理しても load の中身は 1 行も変わらない)。
"""
from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

import pytest

from society.config import load_config
from society.engine import checkpoint, scheduler
from society.engine.simulation import Simulation

_REPO = Path(__file__).resolve().parents[1]

#: 従来の文言(変えてはならない)。Δt 二重変換の手掛かりは第94 OBS-U2 の資産。
_LEGACY_MESSAGE_MARKS = (
    "checkpoint の config が現在の config と不整合(決定論が壊れる)。",
    "seed/n_agents/因子など resume 対象外のキーが変わっている可能性。",
    "load_config(path=…, apply_dt=False)",
)


# --------------------------------------------------------------------------- #
# 共通ヘルパ
# --------------------------------------------------------------------------- #
def _cfg(name: str, n_steps: int, **ov):
    dot = ["run.seed=42", "run.n_agents=12", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def _saved_checkpoint(tmp_path, name: str = "src", steps: int = 8) -> Path:
    """`steps` 回まわして checkpoint を 1 世代書き、そのパスを返す。"""
    d = tmp_path / name
    sim = Simulation(_cfg(name, steps, **{"observer.checkpoint_every": steps}),
                     out_dir=d)
    for step in range(steps):
        scheduler.run_step(sim, step)
    path = checkpoint.save(sim, steps, d / "checkpoint" / f"ckpt-{steps:06d}.pkl.gz")
    sim.logger.flush_segment()
    return path


def _fresh(tmp_path, name: str, n_steps: int = 16, **ov) -> Simulation:
    """checkpoint を読ませる側の sim(`ov` で config を変える = hash を動かす)。"""
    return Simulation(_cfg(name, n_steps, **ov), out_dir=tmp_path / name)


#: 「運用レバー」の代表(本レーンの動機そのもの)。世界の構造は変えない。
_OPS_CHANGE = {"lod.max_llm_per_step": 5}


# =========================================================================== #
# (1) 既定 = 従来どおり ValueError(文言も不変)
# =========================================================================== #
def test_config_mismatch_still_raises_by_default(tmp_path):
    path = _saved_checkpoint(tmp_path)
    dst = _fresh(tmp_path, "dst", **_OPS_CHANGE)
    with pytest.raises(ValueError) as ei:
        checkpoint.load(dst, path)                 # 既定 = accept_config_mismatch=False
    for mark in _LEGACY_MESSAGE_MARKS:
        assert mark in str(ei.value), f"従来の文言が変わっている: {mark!r}"


def test_explicit_false_is_the_same_as_the_default(tmp_path):
    path = _saved_checkpoint(tmp_path)
    dst = _fresh(tmp_path, "dst_false", **_OPS_CHANGE)
    with pytest.raises(ValueError):
        checkpoint.load(dst, path, accept_config_mismatch=False)


def test_run_resume_raises_by_default(tmp_path):
    """engine 経路(`sim.run(resume_from=…)`)の既定も従来どおり拒否する。"""
    path = _saved_checkpoint(tmp_path, "eng", 8)
    sim = Simulation(_cfg("eng", 16, **{"observer.checkpoint_every": 8},
                          **_OPS_CHANGE), out_dir=path.parents[1])
    with pytest.raises(ValueError):
        sim.run(resume_from=path.parents[1])


# =========================================================================== #
# (2) フラグ true = WARNING を残して続行
# =========================================================================== #
def test_accept_flag_warns_and_continues(tmp_path, caplog):
    path = _saved_checkpoint(tmp_path, "warn_src")
    src_hash = checkpoint.config_hash(
        _cfg("warn_src", 8, **{"observer.checkpoint_every": 8}))
    dst = _fresh(tmp_path, "warn_dst", **_OPS_CHANGE)
    cur_hash = checkpoint.config_hash(dst.cfg)
    assert src_hash != cur_hash, "テスト前提(hash が動く config 変更)が崩れた"

    with caplog.at_level(logging.WARNING, logger="society.engine"):
        step = checkpoint.load(dst, path, accept_config_mismatch=True)
    assert step == 8, "受理しても load の返り値(次に実行すべき step)は同じ"

    recs = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert recs, "受理を**黙って**行っている(WARNING が 1 行も出ていない)"
    text = "\n".join(r.getMessage() for r in recs)
    assert src_hash in text and cur_hash in text, \
        f"stored / current の両ハッシュが出ていない: {text}"
    assert "受理" in text, "『操作者が受理した』ことが文面に無い"


def test_accept_flag_restores_the_world_exactly_like_a_matching_load(tmp_path):
    """受理しても load の中身は 1 行も変わらない(復元結果が一致)。"""
    path = _saved_checkpoint(tmp_path, "same_src")
    match = _fresh(tmp_path, "same_match")                  # config 一致
    mismatch = _fresh(tmp_path, "same_mis", **_OPS_CHANGE)  # config 不一致 = 受理して読む
    assert checkpoint.load(match, path) == 8
    assert checkpoint.load(mismatch, path, accept_config_mismatch=True) == 8
    assert [a.id for a in match.agents] == [a.id for a in mismatch.agents]
    assert [a.money for a in match.agents] == [a.money for a in mismatch.agents]
    assert [(a.node, a.x, a.y) for a in match.agents] \
        == [(a.node, a.x, a.y) for a in mismatch.agents]


def test_run_resume_accepts_the_ops_change_end_to_end(tmp_path):
    """engine 経路: 受理フラグを渡すと運用変更込みの resume が最後まで走る。"""
    d = tmp_path / "e2e"
    first = Simulation(_cfg("e2e", 8, **{"observer.checkpoint_every": 8}), out_dir=d)
    first.run()
    second = Simulation(_cfg("e2e", 16, **{"observer.checkpoint_every": 8},
                             **_OPS_CHANGE), out_dir=d)
    summary = second.run(resume_from=d, accept_config_mismatch=True)
    assert int(summary["n_steps"]) == 16
    assert second.budget.max_per_step == 5, "新しい cap が効いていない"


# =========================================================================== #
# (3) 一致時はフラグの有無で無差
# =========================================================================== #
@pytest.mark.parametrize("accept", [False, True])
def test_matching_config_is_unaffected_by_the_flag(tmp_path, caplog, accept):
    path = _saved_checkpoint(tmp_path, f"ok_{int(accept)}")
    dst = _fresh(tmp_path, f"ok_dst_{int(accept)}")
    with caplog.at_level(logging.WARNING, logger="society.engine"):
        assert checkpoint.load(dst, path, accept_config_mismatch=accept) == 8
    assert not [r for r in caplog.records
                if "config" in r.getMessage() and r.levelno >= logging.WARNING], \
        "config が一致しているのに警告が出ている"


def test_relaxation_is_at_the_door_not_in_the_hash():
    """緩和は**受理フラグ**で行う = `_VOLATILE_KEYS`(照合の除外)は 1 件も減っていない。

    除外リストを広げる方式だと「そのキーは決定論に効かない」という**偽の宣言**が
    残ってしまう(実際には効く)。フラグ方式なら照合そのものは走り続け、
    受理した事実だけが WARNING として残る。
    """
    legacy = [("run", "n_steps"), ("run", "name"), ("run", "out_dir"),
              ("observer", "checkpoint_every")]
    for entry in legacy:
        assert entry in checkpoint._VOLATILE_KEYS, f"resume 制御キー {entry} が消えた"
    assert ("lod", "max_llm_per_step") not in checkpoint._VOLATILE_KEYS, \
        "運用レバーを hash の除外へ入れている(= 決定論に効かないという偽の宣言)"


# =========================================================================== #
# (4) CLI(scripts/run.py)
# =========================================================================== #
def _load_run_cli():
    """scripts/run.py を module として読み込む(tests/test_timeconv.py と同じイディオム)。"""
    spec = importlib.util.spec_from_file_location("run_entry_accept",
                                                  _REPO / "scripts" / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cli(run_mod, monkeypatch, argv: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["run.py", *argv])
    run_mod.main()


def test_cli_flag_is_documented_and_parsed():
    """フラグ名が CLI の使い方と解析の両方に在る(綴りのズレを機械で止める)。"""
    src = (_REPO / "scripts" / "run.py").read_text(encoding="utf-8")
    assert '"--resume-accept-config"' in src, "CLI 引数として解析されていない"
    assert "accept_config_mismatch=accept_config" in src, "resume 経路へ配線されていない"
    assert "--resume-accept-config" in src.split('"""')[1], \
        "モジュール docstring の使い方に載っていない"


def test_cli_resume_rejects_the_ops_change_without_the_flag(tmp_path, monkeypatch):
    """★既定の CLI は従来どおり落ちる(この経路の現行挙動ピン)。"""
    run_mod = _load_run_cli()
    base = ["model.backend=mock", "run.seed=42", "run.n_agents=8",
            f"run.out_dir={tmp_path.as_posix()}", "observer.checkpoint_every=8"]
    _cli(run_mod, monkeypatch, [*base, "run.name=cli_no", "run.n_steps=8"])
    with pytest.raises(ValueError):
        _cli(run_mod, monkeypatch,
             ["--resume", str(tmp_path / "cli_no"), "run.n_steps=16",
              "lod.max_llm_per_step=5"])


def test_cli_resume_accepts_the_ops_change_with_the_flag(tmp_path, monkeypatch):
    """`--resume-accept-config` を足すと同じ操作が通り、新しい cap で続行する。"""
    import pyarrow.parquet as pq

    run_mod = _load_run_cli()
    base = ["model.backend=mock", "run.seed=42", "run.n_agents=8",
            f"run.out_dir={tmp_path.as_posix()}", "observer.checkpoint_every=8"]
    _cli(run_mod, monkeypatch, [*base, "run.name=cli_ok", "run.n_steps=8"])
    _cli(run_mod, monkeypatch,
         ["--resume", str(tmp_path / "cli_ok"), "--resume-accept-config",
          "run.n_steps=16", "lod.max_llm_per_step=5"])
    rows = pq.read_table(str(tmp_path / "cli_ok" / "l1_events.parquet")).to_pylist()
    assert max(int(r["step"]) for r in rows) >= 8, "後半チャンクが 1 step も走っていない"
    saved = load_config(path=tmp_path / "cli_ok" / "config.yaml", apply_dt=False)
    assert int(saved.lod.max_llm_per_step) == 5, "運用変更が run dir へ記録されていない"


def test_cli_flag_order_does_not_matter(tmp_path, monkeypatch):
    """`--resume-accept-config` は値を取らない = --resume の前後どちらでも解析される。"""
    run_mod = _load_run_cli()
    base = ["model.backend=mock", "run.seed=42", "run.n_agents=8",
            f"run.out_dir={tmp_path.as_posix()}", "observer.checkpoint_every=8"]
    _cli(run_mod, monkeypatch, [*base, "run.name=cli_ord", "run.n_steps=8"])
    _cli(run_mod, monkeypatch,
         ["--resume-accept-config", "--resume", str(tmp_path / "cli_ord"),
          "run.n_steps=16", "lod.budget.tiers.enabled=true"])
