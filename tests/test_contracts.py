"""D14-3/4: seam 契約 + no-fingerprint 静的チェック。"""
from __future__ import annotations

import re
from pathlib import Path

from society.config import load_config
from society.observer.schema import EVENT_KINDS

SRC = Path(__file__).resolve().parents[1] / "src" / "society"

# engine/cognition/actions/labeling は因子名を名指ししてはならない(design §11)
FORBIDDEN = re.compile(
    r"nfc|risk_tolerance|internal_locus|efficacy|ownership|world_change")
CHECKED_DIRS = ["engine", "cognition", "actions", "labeling", "world"]


def test_no_factor_names_outside_factors():
    for dirname in CHECKED_DIRS:
        for path in (SRC / dirname).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            hits = [m.group(0) for m in FORBIDDEN.finditer(text)]
            assert not hits, f"{path.name} が因子を名指ししている: {hits}(指紋の禁止)"


def test_event_kinds_registered():
    assert len(EVENT_KINDS) >= 14
    for user_requested in ["vocab_coin", "vocab_use", "transmission"]:
        assert user_requested in EVENT_KINDS


def test_config_loads_with_overrides():
    cfg = load_config(overrides=["run.seed=99", "k.writeback=sham"])
    assert cfg.run.seed == 99
    assert cfg.k.writeback == "sham"


def test_writeback_modes_valid():
    for mode in ["free", "degraded", "sham", "off"]:
        cfg = load_config(overrides=[f"k.writeback={mode}"])
        assert str(cfg.k.writeback) == mode
