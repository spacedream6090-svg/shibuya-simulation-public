"""サブプロセス系テストの並列直列化(第86バッチ保守 M-2)が**配線されたまま**であることの回帰。

背景: `test_watchdog` の実 run.py スモークと `test_taxi_live` の SUMO ブリッジは
単体では緑だが、`-n auto` のフルゲートで並列フレークを 2 例観測した(いずれも
「子プロセスを起こして実時間で待つ」形。CPU の奪い合いでストール判定や配車が揺れる)。

対策は pytest-xdist 3.8 の **loadgroup 分配**:
  - `pyproject.toml` の `addopts` に `--dist loadgroup`(既定の `--dist load` では
    マーカーが**黙って無視される**ので、常用コマンドを変えずに効かせるにはここが要る)
  - 対象テストに `@pytest.mark.xdist_group("subproc_run")`

このファイルは「その 2 つが外れていないか」だけを見る(実際の並列挙動は測らない)。

棚卸し(grep: `subprocess.run|Popen|check_output`)の判断:
  - **入れた**: 子プロセスが長命 or 外部資源を掴む/実時間に依存するもの
      tests/test_watchdog.py(監督ループ全体・poll/ストール判定・実 run.py)
      tests/test_live_viewer.py の統合 3 本(run.py を Popen で走らせながら追いかける)
      tests/test_taxi_live.py::test_sumo_bridge_determinism_same_seed(SUMO + traci)
  - **入れない**: 解析/ビューアスクリプトを `subprocess.run` で一撃するだけのもの
      (test_analyze_*, test_lens, test_structure, test_org_ui, test_channels の
       measure_sigma など)。tmp 配下で完結し、実時間の閾値も外部ポートも持たないので
      フレーク源ではなく、全部を直列化するとフルゲートの壁時計を無駄に伸ばすだけ。
"""
from __future__ import annotations

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # Python 3.10(GPUサーバー=Ubuntu 22.04.5 実測)は stdlib に無い
    import tomli as tomllib  # type: ignore[no-redef]
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# (ファイル, 関数名 or None=モジュール全体)
GROUPED = [
    ("test_watchdog.py", None),
    ("test_live_viewer.py", "test_integration_chase_matches_real_data"),
    ("test_live_viewer.py", "test_integration_observation_does_not_change_the_run"),
    ("test_live_viewer.py", "test_cli_once_on_finished_run"),
    ("test_taxi_live.py", "test_sumo_bridge_determinism_same_seed"),
]
GROUP_NAME = "subproc_run"


def test_addopts_enables_loadgroup_distribution():
    """`--dist loadgroup` が addopts にある(無いと xdist_group マーカーは無効)。"""
    cfg = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    addopts = cfg["tool"]["pytest"]["ini_options"].get("addopts", [])
    if isinstance(addopts, str):
        addopts = addopts.split()
    joined = " ".join(addopts)
    assert "--dist" in joined and "loadgroup" in joined, \
        f"addopts に --dist loadgroup が無い: {addopts}"


def test_subprocess_heavy_tests_are_in_one_group():
    """対象テストに xdist_group マーカーが付いたままである。"""
    for fname, func in GROUPED:
        src = (REPO_ROOT / "tests" / fname).read_text(encoding="utf-8")
        assert f'xdist_group("{GROUP_NAME}")' in src, \
            f"{fname} に xdist_group マーカーが無い"
        if func is None:                     # モジュール全体
            assert f'pytestmark = pytest.mark.xdist_group("{GROUP_NAME}")' in src, \
                f"{fname} のモジュール全体マーカーが外れている"
            continue
        # 関数直前の装飾行の中にマーカーがあること(装飾の並び順は問わない)
        head, _, tail = src.partition(f"def {func}(")
        assert tail, f"{fname} に {func} が見当たらない"
        decor = head.rsplit("\n\n", 1)[-1]
        assert f'xdist_group("{GROUP_NAME}")' in decor, \
            f"{fname}::{func} にマーカーが付いていない"


def test_xdist_is_available_with_loadgroup_support():
    """pytest-xdist が入っていて loadgroup を解せる版である(3.x)。"""
    import xdist
    major = int(str(xdist.__version__).split(".")[0])
    assert major >= 2, f"xdist が古く loadgroup 非対応: {xdist.__version__}"
