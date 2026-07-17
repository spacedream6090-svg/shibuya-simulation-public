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

# 基盤ディレクトリ(認知・エンジン・世界・因子)に「場所の知識」を新たに入れない
# (基盤モデル抽出 D1-W2 契約)。地名は envpack(cfg=conf/config.yaml の envpack ブロック)
# 経由で受ける=基盤コードに直書きしない。コメント・docstring も含めて全文を検査する
# (因子名ガードと同じ流儀)。envpack の **読み出し**(cfg["lexicon"]["place_name"] 等)は
# 地名リテラルを含まないので抵触しない。
PLACE_FORBIDDEN = re.compile(r"渋谷|スクランブル|ハチ公|道玄坂|センター街")
PLACE_CHECKED_DIRS = ["cognition", "engine", "world", "factors"]


def test_no_factor_names_outside_factors():
    for dirname in CHECKED_DIRS:
        for path in (SRC / dirname).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            hits = [m.group(0) for m in FORBIDDEN.finditer(text)]
            assert not hits, f"{path.name} が因子を名指ししている: {hits}(指紋の禁止)"


def test_no_place_names_in_foundation():
    """基盤(cognition/engine/world/factors)に地名リテラルが無い(D1-W2 地名禁止ガード)。"""
    for dirname in PLACE_CHECKED_DIRS:
        for path in (SRC / dirname).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            hits = [m.group(0) for m in PLACE_FORBIDDEN.finditer(text)]
            assert not hits, \
                f"{path.name} に地名リテラル {hits}(基盤は envpack 経由で受ける)"


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
