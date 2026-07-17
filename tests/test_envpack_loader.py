"""EnvPack ローダ(基盤モデル抽出 D1-W4)のテスト。

検証:
- env.yaml → 既存 config キー群への写像(world.map / transit.file / agents.personas_file /
  organizations.* / envpack.* / institutions.* / economy.min_wage_hourly …)。
- 優先順位: 基底 < env < profile < dotlist(env=場所・profile=用途)。
- **検収の核心**: `--env env/shibuya` の mock ラン == 現行値を手で束ねた従来ラン(L1 完全一致・同seed)。
  government ON でも一致(institutions を ref から解決した値=現行コード既定と一致する証明)。
- 必須キー欠落・データファイル不在は明確なエラー。transit 無し env は縮退(徒歩の街)して回る。
検証は mock のみ(実LLM 禁止)。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from society.config import build_env_overlay, load_config
from society.engine.simulation import Simulation

REPO = Path(__file__).resolve().parents[1]
ENV_SHIBUYA = REPO / "env" / "shibuya"

# env/shibuya が束ねる「現行値」(=従来ランを手で指定するときの差分)。
_DATA_OVERRIDES = [
    "world.map=data/shibuya_osm_wide_v7.json",
    "transit.file=data/transit_odpt.json",
    "agents.personas_file=data/personas_100_civic.json",
    "organizations.book=data/organizations_shibuya_wide.json",
    "organizations.assignments=data/org_assignments_100_civic_wide.json",
    "economy.min_wage_hourly=1226",
]


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _run(cfg, tmp_path, name):
    sim = Simulation(cfg, out_dir=tmp_path / name)
    sim.run()
    return sim


# --------------------------------------------------------------- 写像
def test_env_maps_manifest_to_config_keys():
    """env/shibuya/env.yaml が既存 config キー群へ正しく写像される。"""
    cfg = load_config(env=str(ENV_SHIBUYA))
    assert str(cfg.world.map).endswith("shibuya_osm_wide_v7.json")   # env が基底を上書き
    assert str(cfg.transit.file).endswith("transit_odpt.json")
    assert str(cfg.agents.personas_file).endswith("personas_100_civic.json")
    assert str(cfg.organizations.book).endswith("organizations_shibuya_wide.json")
    assert str(cfg.organizations.assignments).endswith("org_assignments_100_civic_wide.json")
    # envpack(語彙・駅名・番組名)= 渋谷(基底と同値)
    assert cfg.envpack.lexicon.place_name == "渋谷"
    assert list(cfg.envpack.transit.station_filters) == ["渋谷", "Shibuya"]
    # ② 制度: pref=tokyo を ref から解決
    assert cfg.economy.min_wage_hourly == 1226                       # 東京都最低賃金
    assert cfg.institutions.resident.ward_share == 0.6              # 特別区 区:都=6:4
    assert cfg.institutions.consumption.national_share == 0.78
    # ③ 地点固有: 議会・供託金
    assert cfg.institution_routes.assembly.size == 9
    assert cfg.institution_routes.assembly.term_days == 1460
    assert cfg.institution_routes.vote.deposit == 30000


def test_env_resolved_institutions_match_ref():
    """env の institutions 展開値が ref/institutions_jp.yaml と一致(② 解決の健全性)。"""
    ov = build_env_overlay(str(ENV_SHIBUYA))
    ref = OmegaConf.to_container(
        OmegaConf.load(REPO / "ref" / "institutions_jp.yaml"), resolve=True)
    assert ov["economy"]["min_wage_hourly"] == ref["prefectures"]["tokyo"]["min_wage_hourly"]
    assert ov["institutions"]["consumption"]["rate"] == ref["national"]["consumption"]["rate"]
    brackets = [b["rate"] for b in ov["institutions"]["income_brackets"]]
    assert brackets == [b["rate"] for b in ref["national"]["income_tax_effective"]]


# --------------------------------------------------------------- 優先順位
def test_priority_env_over_base_profile_over_env_dotlist_over_all(tmp_path):
    """基底 < env < profile < dotlist の順で勝つ。"""
    prof = tmp_path / "prof.yaml"
    prof.write_text("world:\n  map: PROFILE_MAP\n", encoding="utf-8")
    # env > 基底
    assert str(load_config(env=str(ENV_SHIBUYA)).world.map).endswith("wide_v7.json")
    # profile > env
    assert load_config(env=str(ENV_SHIBUYA), profile=prof).world.map == "PROFILE_MAP"
    # dotlist > profile
    got = load_config(env=str(ENV_SHIBUYA), profile=prof,
                      overrides=["world.map=DOT_MAP"]).world.map
    assert got == "DOT_MAP"


# --------------------------------------------------------------- L1 同一性(検収の核心)
def test_env_run_equals_manual_current_values(tmp_path):
    """`--env env/shibuya` == 現行値を手で束ねた従来ラン(L1 完全一致・同seed・24step)。"""
    common = ["run.seed=42", "run.n_agents=15", "run.n_steps=24"]
    env_cfg = load_config(overrides=common + ["run.name=env"], env=str(ENV_SHIBUYA))
    man_cfg = load_config(overrides=common + ["run.name=man"] + _DATA_OVERRIDES)
    assert _l1(_run(env_cfg, tmp_path, "env")) == _l1(_run(man_cfg, tmp_path, "man")), \
        "--env のランが現行値の従来ランと L1 不一致(env.yaml が現行値を束ねていない)"


def test_env_run_equals_manual_with_government_on(tmp_path):
    """government ON でも一致 = institutions を ref から解決した値が現行コード既定と同じ証明。"""
    common = ["run.seed=42", "run.n_agents=15", "run.n_steps=24",
              "government.enabled=true"]
    env_cfg = load_config(overrides=common + ["run.name=genv"], env=str(ENV_SHIBUYA))
    man_cfg = load_config(overrides=common + ["run.name=gman"] + _DATA_OVERRIDES)
    assert _l1(_run(env_cfg, tmp_path, "genv")) == _l1(_run(man_cfg, tmp_path, "gman")), \
        "government ON で --env と従来ランが不一致(institutions の ref 解決が現行値とズレている)"


# --------------------------------------------------------------- エラー / 縮退
def test_missing_manifest_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(env=str(tmp_path / "no_such_env"))


def test_missing_required_key_errors(tmp_path):
    (tmp_path / "env.yaml").write_text("data:\n  map: data/shibuya_osm.json\n",
                                       encoding="utf-8")   # env.name 欠落
    with pytest.raises(ValueError):
        load_config(env=str(tmp_path))


def test_missing_map_errors(tmp_path):
    (tmp_path / "env.yaml").write_text("env:\n  name: x\ndata: {}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(env=str(tmp_path))


def test_missing_data_file_errors(tmp_path):
    (tmp_path / "env.yaml").write_text(
        "env:\n  name: x\ndata:\n  map: data/does_not_exist.json\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        load_config(env=str(tmp_path))


def test_transitless_env_degenerates_and_runs(tmp_path):
    """地図だけの最小 env(transit 無し=徒歩の街)がロード・実行できる(縮退)。"""
    (tmp_path / "env.yaml").write_text(
        "env:\n  name: mini\ndata:\n  map: data/shibuya_osm.json\n", encoding="utf-8")
    cfg = load_config(overrides=["run.seed=42", "run.n_agents=8", "run.n_steps=12",
                                 "run.name=mini"], env=str(tmp_path))
    assert str(cfg.world.map).endswith("shibuya_osm.json")
    sim = _run(cfg, tmp_path, "mini")
    assert sim.logger.events, "最小 env のランがイベントを1件も出していない"
