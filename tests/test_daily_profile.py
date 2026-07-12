"""平凡な日常プロファイル conf/daily.yaml(第14バッチ 2026-07-09)のテスト。

daily.yaml が load_config(profile=...) で読め、日常較正の要点が反映されること
(ラン実行は不要=cfg 値の検査のみ。ペルソナ300ファイルの存在には依存しない)。
"""
from __future__ import annotations

from pathlib import Path

from society.config import load_config

REPO = Path(__file__).resolve().parents[1]
DAILY = REPO / "conf" / "daily.yaml"


def _cfg():
    return load_config(profile=DAILY)


def test_daily_profile_loads():
    cfg = _cfg()
    assert cfg is not None


def test_disaster_shocks_removed():
    """日常を壊す外生ショック(台風・停電・運休・日常遅延)は全除去=disaster OFF。"""
    assert _cfg().disaster.enabled is False


def test_assembly_term_is_four_years():
    """区議会は4年任期=ラン内改選なし(30日改選の非現実を廃す)。"""
    assert int(_cfg().institution_routes.assembly.term_days) == 1460


def test_eviction_and_bankruptcy_realistic():
    """立退き90日(実法3ヶ月)・破産240日(現実の稀さ)。"""
    acc = _cfg().economy.accounts
    assert int(acc.eviction_days) == 90
    assert int(acc.bankruptcy_days) == 240


def test_n_agents_is_300():
    assert int(_cfg().run.n_agents) == 300


def test_token_calibration():
    """トークン較正: plan 448 / reflect 2048 / 発話 320 据え置き。"""
    model = _cfg().model
    assert int(model.plan_max_tokens) == 448
    assert int(model.reflect_max_tokens) == 2048
    assert int(model.max_tokens) == 320


def test_uses_300_rosters():
    """名簿・組織台帳・配属が 300 版を指す(別担当が並行生成中の3ファイル)。"""
    cfg = _cfg()
    assert cfg.agents.personas_file == "data/personas_300_civic.json"
    assert cfg.organizations.book == "data/organizations_shibuya_wide300.json"
    assert cfg.organizations.assignments == "data/org_assignments_300_civic_wide.json"


def test_production_profile_unchanged_by_comparison():
    """土台の production.yaml は不変(較正値が生きている)= daily とは別物であることの担保。"""
    prod = load_config(profile=REPO / "conf" / "production.yaml")
    assert prod.disaster.enabled is True
    assert int(prod.institution_routes.assembly.term_days) == 30
    assert int(prod.economy.accounts.eviction_days) == 30
    assert int(prod.run.n_agents) == 100
