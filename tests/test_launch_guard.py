"""β6 / β10(第117バッチ 2026-08-16。レーン B1 安全系)のテスト。

正典: docs/plans/beta-implementation-plan.md §1 / docs/plans/external-audit-triage.md F5・F15。

受入基準:
  β6 - 本番規模(n_agents>=10,000)× model.backend=mock は **起動時に RuntimeError**
     - `run.allow_mock_production=true` の逃し弁で通る
     - 閾値未満 / mock 以外のバックエンドは従来どおり無風
     - 起動バナーに backend / モデル名 / servers 数 / pool dir / present_cap /
       lod.max_llm_per_step が出る(stdout・1 回)
     - `run.allow_mock_production` は**基底 conf に宣言済み**(finals 限定の未宣言トグルを作らない)
  β6 - scripts/freeze_config.py: 凍結 YAML を読み直すと元の解決結果と**完全一致**する
  β10 - run_manifest に「起動側申告」欄(launch)が出る。送っていない値は捏造せず null。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from society import registry as R
from society.config import load_config
from society.engine.simulation import Simulation
from society.observer import manifest as manifest_mod

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, rel: str):
    """scripts/ 配下のスクリプトをモジュールとして読む(scripts/ は package ではない)。"""
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RUN = _load_script("run_entry_mod", "scripts/run.py")
FC = _load_script("freeze_config_mod", "scripts/freeze_config.py")


# --------------------------------------------------------------------------- #
# β6 (1) mock fail-fast
# --------------------------------------------------------------------------- #
def test_production_scale_mock_raises():
    """25 万体 mock ラン(= LLM が 1 本も走らないラン)は起動で落ちる。"""
    cfg = load_config(["run.n_agents=250000", "model.backend=mock"])
    with pytest.raises(RuntimeError) as exc:
        RUN.check_mock_production(cfg)
    msg = str(exc.value)
    assert "mock" in msg and "250,000" in msg
    assert "allow_mock_production" in msg, "逃し弁の名前が案内に無い"


def test_threshold_is_ten_thousand():
    """閾値ちょうどで落ち、1 つ下では落ちない(境界の固定)。"""
    assert RUN.MOCK_PRODUCTION_MIN_AGENTS == 10_000
    with pytest.raises(RuntimeError):
        RUN.check_mock_production(load_config(["run.n_agents=10000",
                                               "model.backend=mock"]))
    RUN.check_mock_production(load_config(["run.n_agents=9999",
                                           "model.backend=mock"]))


def test_escape_hatch_lets_large_mock_run_through():
    """run.allow_mock_production=true = 「mock で回すのが目的」の明示 → 通す。"""
    cfg = load_config(["run.n_agents=250000", "model.backend=mock",
                       "run.allow_mock_production=true"])
    RUN.check_mock_production(cfg)          # 例外が出ないこと自体が検査


def test_real_backend_at_scale_is_untouched():
    """実 LLM バックエンドは規模に関わらず無風(ガードは mock だけを見る)。"""
    cfg = load_config(["run.n_agents=250000", "model.backend=vllm"])
    RUN.check_mock_production(cfg)


def test_default_small_run_is_untouched():
    """既定(10 体 mock)は 1 バイトも変わらない=ゴールデン経路が無風。"""
    RUN.check_mock_production(load_config([]))


def test_allow_mock_production_declared_in_base_conf():
    """finals 限定の未宣言トグルを作らない(第114 レーン 1d の教訓)。"""
    cfg = load_config([])
    assert cfg.run.get("allow_mock_production") is False, "基底 conf に既定値が無い"
    known = {f.id for f in R.FEATURES} | set(R.ALLOWLIST)
    assert "run.allow_mock_production" in known, "レジストリ/ALLOWLIST に宣言が無い"
    # 網羅そのものは tests/test_registry_modes.py が全 conf を対象に見る(定義を二重に持たない)
    assert "run.allow_mock_production" not in R.undeclared_toggles(cfg)


# --------------------------------------------------------------------------- #
# β6 (2) 起動バナー
# --------------------------------------------------------------------------- #
def test_banner_reports_launch_conditions(capsys):
    cfg = load_config(["run.n_agents=7", "model.backend=mock",
                       "model.name=mock-v0", "pool.present_cap=1234",
                       "lod.max_llm_per_step=42"])
    RUN.print_banner(cfg)
    out = capsys.readouterr().out
    for token in ("backend=mock", "model=mock-v0", "servers=0", "n_agents=7",
                  "pool=off", "pool_dir=data/persona_pool",
                  "present_cap=1234", "max_llm_per_step=42"):
        assert token in out, f"バナーに {token} が無い:\n{out}"
    assert out.count("[launch]") == 2, "バナーが 1 回だけ出ていない"


def test_banner_counts_servers_and_pool_state():
    cfg = load_config(["model.backend=vllm", "pool.enabled=true",
                       'model.servers=["http://a:8000","http://b:8000"]'])
    text = "\n".join(RUN.banner_lines(cfg))
    assert "servers=2" in text and "pool=on" in text


def test_banner_is_read_only():
    """バナーは config を 1 バイトも書き換えない(観測が条件を変えない)。"""
    cfg = load_config(["run.n_agents=5"])
    before = OmegaConf.to_yaml(cfg, resolve=True)
    RUN.banner_lines(cfg)
    assert OmegaConf.to_yaml(cfg, resolve=True) == before


# --------------------------------------------------------------------------- #
# A7(第118 レーンC): 使用済み run dir への fresh 起動を拒否
#
#   同名 (`run.name`) の再実行は、前回の `*.part-*.parquet` を finalize が黙って結合し
#   (observer/finalize.py の glob は無条件)、前回の `llm_cache.jsonl` を再生する
#   (llm/cache.py が起動時に丸ごと読む)= **2 つのランが混ざった成果物**。しかも L1 は
#   完全に正常な形をしているので事後に気づけない。起動口で落とす。
# --------------------------------------------------------------------------- #
def _dirty_cfg(tmp_path, name="dirty", **ov):
    dot = ["model.backend=mock", "run.n_agents=4", "run.n_steps=2",
           f"run.name={name}", f"run.out_dir={tmp_path.as_posix()}"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def test_out_dir_resolution_matches_the_simulation(tmp_path):
    """★起動口の run dir 計算が Simulation.__init__ と同一(ずれたらガードが空振りする)。"""
    cfg = _dirty_cfg(tmp_path, "resolv")
    assert RUN.resolve_out_dir(cfg) == Simulation(cfg).out_dir


def test_clean_or_missing_out_dir_passes(tmp_path):
    """存在しない dir / 無関係なファイルしか無い dir は素通り(既定の起動は無風)。"""
    RUN.check_dirty_outdir(_dirty_cfg(tmp_path, "never_used"))
    used = tmp_path / "harmless"
    used.mkdir()
    for stem in ("config.yaml", "run.out.log", "agents.json", "summary.json"):
        (used / stem).write_text("x", encoding="utf-8")
    RUN.check_dirty_outdir(_dirty_cfg(tmp_path, "harmless"))


@pytest.mark.parametrize("rel", [
    "l1_events.part-0000.parquet",          # finalize の glob が無条件に結合する
    "l2_metrics.part-0003.parquet",
    "l1_events.parquet",                    # 完走済みの成果物を黙って上書きしない
    "checkpoint/ckpt-000144.pkl.gz",        # --resume の意図だった可能性
    "checkpoint/dormant-000144.pkl.gz",
    "llm_cache.jsonl",                      # 起動時に丸ごと読み戻される
    "llm_cache.rank0.jsonl",
])
def test_fresh_start_rejects_a_used_out_dir(tmp_path, rel):
    d = tmp_path / "used"
    (d / rel).parent.mkdir(parents=True, exist_ok=True)
    (d / rel).write_bytes(b"x")
    with pytest.raises(RuntimeError) as exc:
        RUN.check_dirty_outdir(_dirty_cfg(tmp_path, "used"))
    msg = str(exc.value)
    assert Path(rel).name in msg, f"当たったファイル名が案内に無い:\n{msg}"
    assert "--resume" in msg and "run.name" in msg, "取るべき手が案内されていない"
    assert "run.allow_dirty_outdir=true" in msg, "逃し弁の名前が案内に無い"


def test_escape_hatch_allows_writing_into_a_used_dir(tmp_path):
    """`run.allow_dirty_outdir=true` = 「承知の上で重ねて書く」の明示 → 通す。"""
    d = tmp_path / "hatch"
    d.mkdir()
    (d / "l1_events.part-0000.parquet").write_bytes(b"x")
    with pytest.raises(RuntimeError):
        RUN.check_dirty_outdir(_dirty_cfg(tmp_path, "hatch"))
    RUN.check_dirty_outdir(_dirty_cfg(tmp_path, "hatch",
                                      **{"run.allow_dirty_outdir": "true"}))


def test_dirty_check_is_read_only(tmp_path):
    """検査はファイルを 1 つも作らない/消さない(観測が世界を変えない)。"""
    d = tmp_path / "ro"
    d.mkdir()
    (d / "l1_events.part-0000.parquet").write_bytes(b"x")
    before = sorted(p.name for p in d.iterdir())
    with pytest.raises(RuntimeError):
        RUN.check_dirty_outdir(_dirty_cfg(tmp_path, "ro"))
    assert sorted(p.name for p in d.iterdir()) == before
    assert RUN.dirty_outdir_hits(tmp_path / "does_not_exist") == []


def test_dirty_hits_are_bounded(tmp_path):
    """案内は有界(part が 1000 本あってもエラーメッセージが暴れない)。"""
    d = tmp_path / "many"
    d.mkdir()
    for i in range(30):
        (d / f"l1_events.part-{i:04d}.parquet").write_bytes(b"x")
    assert len(RUN.dirty_outdir_hits(d)) == 6


# --------------------------------------------------------------------------- #
# β6 (3) scripts/freeze_config.py
# --------------------------------------------------------------------------- #
def test_frozen_config_reresolves_identically(tmp_path):
    """凍結 YAML を読み直した結果 == 元の解決結果(完全一致)。"""
    overrides = ["run.n_agents=12", "model.temperature=0.3", "run.seed=7"]
    out = tmp_path / "frozen.yaml"
    rep = FC.freeze(overrides=overrides, out=out)
    direct = OmegaConf.to_container(load_config(overrides), resolve=True)
    # 凍結ファイルは **apply_dt 済み**なので読み直しは apply_dt=False(docstring の規約)
    reread = OmegaConf.to_container(load_config(path=out, apply_dt=False),
                                    resolve=True)
    assert reread == direct
    assert rep["path"] == out and rep["sha256"] == hashlib.sha256(
        out.read_bytes()).hexdigest()


def test_frozen_config_captures_profile_and_dotlist(tmp_path):
    """profile 単独では mock のまま(監査 F5 の危険そのもの)/ dotlist まで畳み込める。"""
    prof = REPO_ROOT / "conf" / "finals_observe.yaml"
    bare = tmp_path / "bare.yaml"
    FC.freeze(profile=prof, out=bare)
    assert OmegaConf.load(bare).model.backend == "mock", \
        "profile 単独実行が mock になる事実(F5)が凍結ファイルに現れていない"
    fixed = tmp_path / "fixed.yaml"
    FC.freeze(profile=prof, overrides=["model.backend=vllm"], out=fixed)
    assert OmegaConf.load(fixed).model.backend == "vllm"
    assert FC.sha256_text(bare.read_text(encoding="utf-8")) \
        != FC.sha256_text(fixed.read_text(encoding="utf-8"))


def test_freeze_writes_sha256_sidecar(tmp_path):
    out = tmp_path / "f.yaml"
    rep = FC.freeze(overrides=["run.n_agents=3"], out=out)
    line = rep["sha256_path"].read_text(encoding="utf-8")
    assert line == f"{rep['sha256']}  {out.name}\n", "sha256sum -c 形式ではない"


def test_freeze_is_deterministic(tmp_path):
    """同じ入力なら sha256 も同じ(条件の同一性を数字で言える)。"""
    a = FC.freeze(overrides=["run.n_agents=3"], out=tmp_path / "a.yaml")
    b = FC.freeze(overrides=["run.n_agents=3"], out=tmp_path / "b.yaml")
    assert a["sha256"] == b["sha256"]


def test_default_out_path_shape():
    p = FC.default_out_path("20260822")
    assert p.name == "finals_20260822_frozen.yaml" and p.parent.name == "conf"


# --------------------------------------------------------------------------- #
# β10 起動側申告(run_manifest の launch 欄)
# --------------------------------------------------------------------------- #
def _sim(tmp_path, **ov):
    dot = ["run.seed=42", "run.n_agents=4", "run.n_steps=2",
           "model.backend=mock"] + [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / "man")


def test_manifest_launch_declaration(tmp_path):
    man = manifest_mod.build(_sim(tmp_path))
    launch = man["launch"]
    assert launch["declared_by"] == "launcher" and launch["verified"] is False
    assert launch["backend"] == "mock" and launch["n_servers"] == 0
    assert launch["vllm_version"] is None, "vLLM 以外で版を捏造している"
    smp = launch["sampling"]
    assert smp["temperature"] == pytest.approx(0.7)
    assert smp["max_tokens"] > 0 and smp["reflect_max_tokens"] > 0
    # ★client が送っていないパラメータは null(0 で埋めない)
    assert smp["top_p"] is None and smp["top_k"] is None
    assert smp["seed"] is None and smp["seed_enabled"] is False


def test_manifest_launch_records_request_seed_policy(tmp_path):
    man = manifest_mod.build(_sim(tmp_path, **{"llm.request_seed.enabled": "true"}))
    smp = man["launch"]["sampling"]
    assert smp["seed_enabled"] is True and smp["seed_run_seed"] == 42
    assert "blake2b" in smp["seed"], "seed の導出式が残っていない"


def test_manifest_launch_is_serializable(tmp_path):
    """manifest は JSON で書かれるので launch 欄も素直に直列化できる。"""
    man = manifest_mod.build(_sim(tmp_path))
    json.loads(json.dumps(man["launch"], ensure_ascii=False))
