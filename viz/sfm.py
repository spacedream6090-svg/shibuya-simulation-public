"""後方互換シム — SFM コアはシム正典 src/society/world/sfm_core.py へ移設済み。

歴史的経緯: この最小 Social Force Model コア(Crowd / SignalGate)は当初この viz/ 側に
置かれ、scripts/synth_crowd.py・scripts/crossing_demo.py・tests/ が `import sfm` /
`from sfm import Crowd, SignalGate, RADIUS_MIN, RADIUS_MAX` で参照してきた。B2 で正準を
src/society/world/sfm_core.py へ逐語移設し(挙動変更ゼロ)、この module はそこからの
再export だけを行う薄いシムにした。既存の import 経路(`import sfm` / `from sfm import …`)
は一字も変えずに動く(移設の逐語性の証明=既存 SFM 系テストが無修正で緑)。

新規コードは `from society.world.sfm_core import Crowd, SignalGate` を直接使うこと。
"""
from __future__ import annotations

import sys
from pathlib import Path

# viz/ は src/ を PYTHONPATH に含めない前提でも単体で `import sfm` が動くよう、
# リポジトリ直下の src/ を sys.path に足してから正準モジュールを引く(既存 viz 群の
# REPO_ROOT = parents[1] と同流儀。pytest 実行時は pythonpath=["src"] 済みで二重登録は無害)。
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# 逐語移設した正準コアからの全 public API + 参照される定数の再export。
from society.world.sfm_core import (  # noqa: E402,F401
    Crowd,
    SignalGate,
    TAU_DEFAULT,
    A_DEFAULT,
    B_DEFAULT,
    MASS_DEFAULT,
    LAMBDA_DEFAULT,
    V_MAX_FACTOR,
    RADIUS_MIN,
    RADIUS_MAX,
    CUTOFF_M,
    _EXP_ARG_MAX,
)

__all__ = [
    "Crowd", "SignalGate", "TAU_DEFAULT", "A_DEFAULT", "B_DEFAULT",
    "MASS_DEFAULT", "LAMBDA_DEFAULT", "V_MAX_FACTOR", "RADIUS_MIN",
    "RADIUS_MAX", "CUTOFF_M",
]
