"""第72バッチ: 機能レジストリ(repro_tier)+ ランモード observe/journal/verify のテスト。

受入基準(docs/plans/dual-mode-observe-verify-plan.md 第72行 /
        docs/plans/source/dual-mode-instruments.md Part A):
  - 既定(run.mode 未指定/none)は現行動作そのまま = L1 バイト一致・config を触らない
  - verify モードで journal / none 等級が全て OFF になり、それでも短ランが完走する
  - 自動 OFF が**黙って**起きない(WARNING ログ + run_manifest.json の features.auto_disabled)
  - observe / journal / verify のどのモードでも manifest に run_mode と等級一覧が出る
  - **未宣言トグルの検出**: conf の bool リーフがレジストリにも allowlist にも無ければ fail
    (= 新機能を足すと repro_tier の宣言を忘れられない)
  - ラン間比較ガード: 等級混在は既定で拒否・明示フラグで警告つき続行・同一なら無言
"""
from __future__ import annotations

import json
import logging

import pyarrow.parquet as pq
import pytest
from omegaconf import OmegaConf

from society import registry as R
from society.config import load_config
from society.engine.simulation import Simulation


# --------------------------------------------------------------------------- #
# 共通ヘルパ
# --------------------------------------------------------------------------- #
def _cfg(name: str, n_steps: int = 24, n_agents: int = 10, **ov):
    dot = ["run.seed=42", f"run.n_agents={n_agents}", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def _run(tmp_path, name: str, n_steps: int = 24, n_agents: int = 10, **ov):
    out = tmp_path / name
    sim = Simulation(_cfg(name, n_steps, n_agents, **ov), out_dir=out)
    sim.run()
    return sim, out


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _manifest(out_dir):
    return json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# (1) レジストリの健全性
# --------------------------------------------------------------------------- #
def test_registry_entries_are_well_formed():
    ids = [f.id for f in R.FEATURES]
    assert len(ids) == len(set(ids)), "id が重複している"
    for f in R.FEATURES:
        assert f.repro_tier in R.TIERS, f"{f.id}: 未知の repro_tier {f.repro_tier}"
        assert f.fingerprint_risk in R.RISKS, f"{f.id}: 未知の fingerprint_risk"
        assert isinstance(f.affects_k, bool)
        assert f.description.strip(), f"{f.id}: description が空"
    # allowlist は「機能でないキー」専用 = レジストリと重ならない
    assert not (set(ids) & set(R.ALLOWLIST)), "allowlist と FEATURES が重複している"
    assert all(v.strip() for v in R.ALLOWLIST.values()), "allowlist に理由が無い項目がある"


def test_registered_ids_all_exist_in_config():
    """宣言した id は conf に実在するキーでなければならない(死んだ宣言の検出)。"""
    cfg = OmegaConf.to_container(load_config(), resolve=True)
    dead = [f.id for f in R.FEATURES if R._select(cfg, f.id) is None]
    assert dead == [], f"conf に存在しない id が宣言されている: {dead}"


def test_auto_disable_targets_are_switchable():
    """journal / none 等級は自動 OFF の対象なので、off 値へ落とせる型でなければならない。"""
    cfg = OmegaConf.to_container(load_config(), resolve=True)
    for f in R.FEATURES:
        if f.repro_tier == "strict":
            continue
        val = R._select(cfg, f.id)
        assert isinstance(val, bool) or f.off_value is not False, (
            f"{f.id}: bool でないのに off_value が既定 False のまま")


# --------------------------------------------------------------------------- #
# (2) 未宣言トグルの検出(CI ゲート)
# --------------------------------------------------------------------------- #
def test_no_undeclared_toggles_in_shipped_config():
    """conf/config.yaml の bool リーフは全て宣言済み(または allowlist)である。"""
    assert R.undeclared_toggles(load_config()) == []


def test_no_undeclared_toggles_in_any_shipped_profile():
    """★第114 レーン 1d: **同梱の全プロファイル**の bool リーフが宣言済みである。

    第109 の発見: 上の `test_no_undeclared_toggles_in_shipped_config` は
    conf/config.yaml しか見ないので、**基底 conf に姿が無く finals_observe.yaml だけが
    触るトグル**は網に掛からなかった(実測 8 件: economy の bank/consumption/payment/vc・
    institution_routes.assembly.from_roster/.realism.enabled・memory.actr.enabled・
    prompts.dialog_history)。「本選で ON にするものほど宣言が漏れる」という最悪の
    向きの穴だったので、走査を conf/*.yaml 全体へ広げて再発を止める。
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    known = {f.id for f in R.FEATURES} | set(R.ALLOWLIST)
    missing: dict[str, list[str]] = {}
    for path in sorted((root / "conf").glob("*.yaml")):
        leaves = R.flatten_bools(OmegaConf.load(path))
        dead = [k for k in sorted(leaves) if k not in known]
        if dead:
            missing[path.name] = dead
    assert missing == {}, f"未宣言トグルを含むプロファイル: {missing}"


def test_finals_profile_declares_the_eight_recovered_toggles():
    """第114 レーン 1d で拾い上げた 8 件が、基底 conf にもレジストリにも在る。"""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    recovered = ("economy.bank.enabled", "economy.consumption.enabled",
                 "economy.payment.enabled", "economy.vc.enabled",
                 "institution_routes.assembly.from_roster",
                 "institution_routes.assembly.realism.enabled",
                 "memory.actr.enabled", "prompts.dialog_history")
    ids = {f.id for f in R.FEATURES}
    base = OmegaConf.to_container(load_config(), resolve=True)
    fin = R.flatten_bools(OmegaConf.load(root / "conf" / "finals_observe.yaml"))
    for fid in recovered:
        assert fid in ids, f"{fid} がレジストリに無い"
        assert R._select(base, fid) is not None, f"{fid} が基底 conf に無い"
        assert fid in fin, f"{fid} が本選 conf から消えた(検査の前提が崩れている)"


def test_fake_unregistered_toggle_is_detected():
    """わざと未登録の偽キーを足すと検出される(= 宣言忘れが CI で落ちる)。"""
    cfg = OmegaConf.to_container(load_config(), resolve=True)
    cfg["weather"]["brand_new_feature"] = True          # 既存 namespace 配下
    cfg["totally_new_block"] = {"enabled": False}       # 新 namespace
    found = R.undeclared_toggles(cfg)
    assert "weather.brand_new_feature" in found
    assert "totally_new_block.enabled" in found


def test_allowlisted_key_is_not_reported():
    cfg = OmegaConf.to_container(load_config(), resolve=True)
    cfg["run"]["seed_auto"] = True                      # ALLOWLIST 済み
    assert R.undeclared_toggles(cfg) == []


# --------------------------------------------------------------------------- #
# (3) 既定(run.mode=none)は現行動作そのまま
# --------------------------------------------------------------------------- #
def test_default_mode_is_none_and_touches_nothing():
    cfg = load_config()
    assert str(cfg.run.mode) == "none"
    gated, rep = R.apply_mode(cfg)
    assert gated is cfg, "既定モードで config オブジェクトが差し替わっている"
    assert rep["run_mode"] == "none"
    assert rep["auto_disabled"] == []


def test_mode_none_matches_default_l1_byte_for_byte(tmp_path):
    """run.mode を明示しても既定ランと L1 が一字一句一致する(1バイトも変えない)。"""
    a, _ = _run(tmp_path, "m_default")
    b, _ = _run(tmp_path, "m_none", **{"run.mode": "none"})
    assert _l1(a) == _l1(b)


def test_apply_mode_does_not_mutate_caller_config():
    cfg = load_config(["run.mode=verify", "freedom.open_actions=true"])
    gated, rep = R.apply_mode(cfg)
    assert cfg.freedom.open_actions is True, "呼び出し側の config が書き換えられた"
    assert gated.freedom.open_actions is False
    assert gated is not cfg


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        R.mode_of(load_config(["run.mode=turbo"]))


# --------------------------------------------------------------------------- #
# (4) モードごとの自動取捨
# --------------------------------------------------------------------------- #
def test_observe_allows_every_tier():
    """observe は none 等級まで許可する(制限なし・宣言のみ)。"""
    cfg = load_config(["run.mode=observe", "transit_ride.live.enabled=true",
                       "freedom.open_actions=true"])
    gated, rep = R.apply_mode(cfg)
    assert rep["auto_disabled"] == []
    assert gated.transit_ride.live.enabled is True
    assert gated.freedom.open_actions is True
    tiers = {e["repro_tier"] for e in rep["enabled"]}
    assert {"strict", "journal", "none"} <= tiers


def test_journal_mode_drops_only_none_tier():
    cfg = load_config(["run.mode=journal", "transit_ride.live.enabled=true",
                       "freedom.open_actions=true", "weather.enabled=true"])
    gated, rep = R.apply_mode(cfg)
    dropped = {d["id"] for d in rep["auto_disabled"]}
    assert "transit_ride.live.enabled" in dropped          # none 等級 → OFF
    assert "freedom.open_actions" not in dropped           # journal は許可
    assert gated.freedom.open_actions is True
    assert gated.weather.enabled is True
    assert rep["counts"]["none"] == 0


def test_verify_mode_drops_journal_and_none():
    cfg = load_config(["run.mode=verify", "transit_ride.live.enabled=true",
                       "freedom.open_actions=true", "weather.enabled=true",
                       "schedule.enabled=true"])
    gated, rep = R.apply_mode(cfg)
    dropped = {d["id"] for d in rep["auto_disabled"]}
    assert {"transit_ride.live.enabled", "freedom.open_actions",
            "schedule.enabled"} <= dropped
    assert gated.weather.enabled is True                   # strict は残る
    assert rep["counts"]["journal"] == 0 and rep["counts"]["none"] == 0
    assert rep["counts"]["strict"] > 0


def test_verify_drops_default_on_journal_features():
    """出荷既定で ON の journal 機能(朝の計画・ツール・制度DSL)も verify では落ちる。"""
    _gated, rep = R.apply_mode(load_config(["run.mode=verify"]))
    dropped = {d["id"] for d in rep["auto_disabled"]}
    assert {"planning.enabled", "tools.enabled", "rules.enabled"} <= dropped
    # 出荷既定 ON = 実験者が明示的に立てたのではない → explicit は立たない
    assert all(d["explicit"] is False for d in rep["auto_disabled"])


def test_warning_names_explicitly_enabled_features(caplog):
    """明示 ON にした機能が落ちたら WARNING で機能名が列挙される(黙って落とさない)。"""
    with caplog.at_level(logging.WARNING, logger="society.registry"):
        R.apply_mode(load_config(["run.mode=verify", "freedom.open_actions=true"]))
    text = "\n".join(r.message for r in caplog.records
                     if r.levelno >= logging.WARNING)
    assert "freedom.open_actions" in text
    assert "★" in text                        # 明示 ON の印
    assert "run.mode=verify" in text


# --------------------------------------------------------------------------- #
# (5) 実ラン: verify で完走し、manifest とログに自動 OFF が出る
# --------------------------------------------------------------------------- #
def test_verify_short_run_completes_and_records_auto_disabled(tmp_path):
    sim, out = _run(tmp_path, "m_verify", n_steps=24,
                    **{"run.mode": "verify", "freedom.open_actions": "true"})
    assert len(sim.logger.events) > 0, "verify モードのランが空"
    assert (out / "l1_events.parquet").exists()
    assert len(pq.read_table(out / "l1_events.parquet")) == len(sim.logger.events)

    man = _manifest(out)
    assert man["run_mode"] == "verify" and man["run"]["mode"] == "verify"
    feats = man["features"]
    assert feats["counts"]["journal"] == 0 and feats["counts"]["none"] == 0
    dropped = {d["id"] for d in feats["auto_disabled"]}
    assert "freedom.open_actions" in dropped
    assert any(d["explicit"] for d in feats["auto_disabled"])
    # 保存 config も落とし切った後の状態(事後に「何が有効だったか」が config からも判る)
    saved = OmegaConf.load(out / "config.yaml")
    assert saved.freedom.open_actions is False
    assert saved.planning.enabled is False
    assert sim.cfg.freedom.open_actions is False


def test_verify_completes_with_every_journal_and_none_feature_requested(tmp_path):
    """Part A(3): journal / none 等級を**全部 ON で要求**しても、verify では全て落ち、
    それでも世界は最後まで回り妥当な状態に収まる(= これらの層が抜けても因果が壊れない)。"""
    base = OmegaConf.to_container(load_config(), resolve=True)
    forced = {f.id: "true" for f in R.FEATURES
              if f.repro_tier != "strict" and isinstance(R._select(base, f.id), bool)}
    assert len(forced) >= 25, "journal/none のトグルを拾えていない"
    sim, out = _run(tmp_path, "m_verify_all", n_steps=24,
                    **{"run.mode": "verify", **forced})

    # 全部 OFF に落ちている
    for fid in forced:
        assert R._select(sim.cfg, fid) is False, f"{fid} が落ちていない"
    # 世界は妥当な範囲: 全員が生きていて、移動・熟慮・会計が動いている
    assert len(sim.agents) == 10
    assert all(isinstance(a.money, (int, float)) and a.money == a.money
               for a in sim.agents)
    assert all(a.node is not None for a in sim.agents)
    kinds = {e.kind for e in sim.logger.events}
    assert {"arrive", "move_segment", "llm_deliberate"} <= kinds
    assert all(0 <= e.step < 24 for e in sim.logger.events)
    man = _manifest(out)
    assert len(man["features"]["auto_disabled"]) >= len(forced)


@pytest.mark.parametrize("mode", ["none", "observe", "journal", "verify"])
def test_manifest_reports_mode_and_tiers(tmp_path, mode):
    _sim, out = _run(tmp_path, f"m_man_{mode}", n_steps=12, n_agents=8,
                     **{"run.mode": mode})
    man = _manifest(out)
    assert man["run_mode"] == mode
    feats = man["features"]
    assert feats["max_tier"] == R.MODE_MAX_TIER[mode]
    assert feats["registry_size"] == len(R.FEATURES)
    assert feats["undeclared"] == []
    assert set(feats["counts"]) == set(R.TIERS)
    assert sum(feats["counts"].values()) == len(feats["enabled"])
    for e in feats["enabled"]:                       # 等級つきの一覧であること
        assert set(e) == {"id", "repro_tier", "affects_k", "fingerprint_risk"}
    if mode in ("none", "observe"):
        assert feats["counts"]["journal"] > 0        # 既定 ON の journal 機能が残る
        assert feats["auto_disabled"] == []
    else:
        assert feats["counts"]["none"] == 0


# --------------------------------------------------------------------------- #
# (6) ラン間比較ガード
# --------------------------------------------------------------------------- #
def _fake_run(tmp_path, name: str, mode: str | None, tiers=("strict",)):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    if mode is not None:
        man = {"run_mode": mode, "run": {"mode": mode},
               "features": {"run_mode": mode,
                            "enabled": [{"id": f"x{i}", "repro_tier": t,
                                         "affects_k": False,
                                         "fingerprint_risk": "none"}
                                        for i, t in enumerate(tiers)]}}
        (d / "run_manifest.json").write_text(json.dumps(man), encoding="utf-8")
    return str(d)


def test_guard_is_silent_when_modes_match(tmp_path):
    a = _fake_run(tmp_path, "g_a", "verify")
    b = _fake_run(tmp_path, "g_b", "verify")
    said: list[str] = []
    rep = R.guard_or_die([a, b], echo=said.append)
    assert said == [] and rep["ok"] and not rep["mismatch"]


def test_guard_refuses_mode_mismatch(tmp_path):
    a = _fake_run(tmp_path, "g_obs", "observe", tiers=("strict", "journal"))
    b = _fake_run(tmp_path, "g_ver", "verify", tiers=("strict",))
    said: list[str] = []
    with pytest.raises(SystemExit):
        R.guard_or_die([a, b], echo=said.append)
    assert any("observe" in m and "verify" in m for m in said)


def test_guard_refuses_tier_set_mismatch_even_in_same_mode(tmp_path):
    a = _fake_run(tmp_path, "g_t1", "observe", tiers=("strict",))
    b = _fake_run(tmp_path, "g_t2", "observe", tiers=("strict", "none"))
    with pytest.raises(SystemExit):
        R.guard_or_die([a, b], echo=lambda _m: None)


def test_guard_continues_with_explicit_flag(tmp_path):
    a = _fake_run(tmp_path, "g_a2", "observe")
    b = _fake_run(tmp_path, "g_b2", "verify")
    said: list[str] = []
    rep = R.guard_or_die([a, b], allow_mismatch=True, echo=said.append)
    assert rep["ok"] and rep["mismatch"] and not rep["refuse"]
    assert any("allow-tier-mismatch" in m for m in said)     # 警告は必ず出す


def test_guard_treats_missing_manifest_as_unknown(tmp_path):
    a = _fake_run(tmp_path, "g_old1", None)      # 第72以前のラン(manifest 無し)
    b = _fake_run(tmp_path, "g_old2", None)
    said: list[str] = []
    rep = R.guard_or_die([a, b], echo=said.append)
    assert rep["ok"] and not rep["mismatch"]
    assert rep["unknown"] == ["g_old1", "g_old2"]
    assert any("不明" in m for m in said)         # 警告のみ(過去ラン互換を壊さない)


def test_default_echo_survives_cp932_stdout(monkeypatch):
    """第98小粒: Windows 非TTY stdout(cp932)では警告文の ⚠/✖ が UnicodeEncodeError
    になり解析スクリプトごと落ちる実バグ(小粒C検収で発見)の回帰。
    既定 echo(_safe_echo)は本文を変えず、表示不能文字だけ置換して必ず生き残る。"""
    import io
    import sys
    buf = io.TextIOWrapper(io.BytesIO(), encoding="cp932")
    monkeypatch.setattr(sys, "stdout", buf)
    R._safe_echo("⚠ ✖ cp932 に無い記号を含む警告")   # 落ちないことが本体
    buf.flush()
    raw = buf.buffer.getvalue().decode("cp932")
    assert "警告" in raw and "?" in raw               # 本文は残り記号だけ置換される


def test_real_run_signature_is_readable(tmp_path):
    _sim, out = _run(tmp_path, "m_sig", n_steps=12, n_agents=8,
                     **{"run.mode": "observe"})
    sig = R.run_signature(out)
    assert sig["known"] and sig["run_mode"] == "observe" and "strict" in sig["tiers"]


def test_compare_runs_exposes_allow_flag():
    """解析側の口(scripts/compare_runs.py)がガードとフラグを持っている。"""
    import inspect
    src = inspect.getsource(_compare_runs().main)
    assert "allow_tier_mismatch" in inspect.signature(
        _compare_runs().run_compare).parameters
    assert "--allow-tier-mismatch" in src


def _compare_runs():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import compare_runs as cr
    return cr


def _stub_run_for_compare(tmp_path, name: str, mode: str, seed: int):
    """compare_runs が読める最小のラン(l2_metrics.parquet + config.yaml + manifest)。"""
    import pyarrow as pa
    import pyarrow.parquet as pq
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"step": [0, 1, 2],
                             "n_working": [10.0, 11.0, 12.0]}),
                   d / "l2_metrics.parquet")
    pq.write_table(pa.table({
        "step": [0, 1], "sim_min": [0, 10], "agent_id": [0, 0],
        "kind": ["arrive", "arrive"], "x": [0.0, 1.0], "y": [0.0, 1.0],
        "payload": ["{}", "{}"]}), d / "l1_events.parquet")
    OmegaConf.save(OmegaConf.create({"run": {"seed": seed, "mode": mode}}),
                   d / "config.yaml")
    (d / "run_manifest.json").write_text(json.dumps(
        {"run_mode": mode, "run": {"mode": mode, "seed": seed},
         "features": {"run_mode": mode,
                      "enabled": [{"id": "x", "repro_tier": "strict",
                                   "affects_k": False,
                                   "fingerprint_risk": "none"}]}}),
        encoding="utf-8")
    return str(d)


def test_compare_runs_refuses_and_then_allows_tier_mismatch(tmp_path, capsys):
    """実際の比較スクリプト経由: 混在は拒否 → 明示フラグで警告つき続行。"""
    cr = _compare_runs()
    a = _stub_run_for_compare(tmp_path, "cmp_obs_s1", "observe", 1)
    b = _stub_run_for_compare(tmp_path, "cmp_ver_s1", "verify", 1)
    out = str(tmp_path / "_cmp")
    with pytest.raises(SystemExit):
        cr.run_compare([a], [b], out)
    info = cr.run_compare([a], [b], out, allow_tier_mismatch=True)
    assert info["tier_guard"]["mismatch"] and info["tier_guard"]["ok"]
    assert "allow-tier-mismatch" in info["md"]          # レポートにも残る


def test_compare_runs_is_silent_when_modes_match(tmp_path):
    cr = _compare_runs()
    a = _stub_run_for_compare(tmp_path, "cmp_ver_a", "verify", 1)
    b = _stub_run_for_compare(tmp_path, "cmp_ver_b", "verify", 2)
    info = cr.run_compare([a], [b], str(tmp_path / "_cmp2"))
    assert info["tier_guard"]["messages"] == []
    assert not info["md"].startswith(">")
