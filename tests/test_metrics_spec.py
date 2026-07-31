"""第78バッチ Part G: metrics_spec_hash(指標定義コードの凍結ハッシュ)のテスト。

受入基準(docs/plans/source/dual-mode-instructions.md Phase 0(6)・T7):
  T7 … 指標コードを**1 文字**変更すると metrics_spec_hash が変わる
  加えて本実装固有の要件:
    - ファイルリスト自体もハッシュ対象(リストから外す改竄も露見する)
    - 正規化は BOM 除去 + 改行 LF 統一だけ(CRLF/LF 差でハッシュが揺れない)
    - run_manifest.json に metrics_spec_hash と内訳が載る
"""
from __future__ import annotations

import json

from society.config import REPO_ROOT, load_config
from society.engine.simulation import Simulation
from society.observer import metrics_spec as MS


def _run(tmp_path, name: str, n_steps: int = 3):
    cfg = load_config(["run.seed=42", "run.n_agents=6", f"run.n_steps={n_steps}",
                       f"run.name={name}", "model.backend=mock"])
    sim = Simulation(cfg, out_dir=tmp_path / name)
    sim.run()
    return tmp_path / name


# --------------------------------------------------------------------------- #
def test_all_spec_files_exist():
    """凍結対象に**死んだパス**が無い(リストの保守漏れを検出)。"""
    missing = [rel for rel in MS.SPEC_FILES if not (REPO_ROOT / rel).exists()]
    assert missing == [], f"SPEC_FILES に存在しないパスがある: {missing}"
    assert MS.compute()["missing"] == []


def test_hash_is_stable_and_hex():
    a = MS.compute()
    b = MS.compute()
    assert a == b
    assert len(a["metrics_spec_hash"]) == 64
    int(a["metrics_spec_hash"], 16)                 # 16 進として解釈できる
    assert a["n_files"] == len(MS.SPEC_FILES)


def test_t7_one_char_change_changes_hash(tmp_path):
    """T7: 指標コードを 1 文字変えると hash が変わる(事後の指標いじりが露見する)。"""
    root = tmp_path / "repo"
    for rel in MS.SPEC_FILES:                       # 最小の複製リポジトリを作る
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((REPO_ROOT / rel).read_bytes())
    before = MS.spec_hash(root)
    target = root / "src/society/observer/aggregate.py"
    target.write_bytes(target.read_bytes() + b"#")  # ★1 文字だけ足す
    after = MS.spec_hash(root)
    assert before != after
    # どのファイルが変わったかも内訳から判る
    fa, fb = MS.compute(root)["files"], None
    target.write_bytes(target.read_bytes()[:-1])    # 戻すと元の hash に一致する
    fb = MS.compute(root)["files"]
    assert MS.spec_hash(root) == before
    assert fa != fb


def test_file_list_change_changes_hash(tmp_path):
    """リストからファイルを外す改竄も hash に出る(リスト自体がハッシュ対象)。"""
    root = tmp_path / "repo2"
    for rel in MS.SPEC_FILES:
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((REPO_ROOT / rel).read_bytes())
    before = MS.spec_hash(root)
    original = MS.SPEC_FILES
    try:
        MS.SPEC_FILES = tuple(x for x in original
                              if x != "scripts/diagnose_stationarity.py")
        assert MS.spec_hash(root) != before
    finally:
        MS.SPEC_FILES = original
    assert MS.spec_hash(root) == before


def test_normalization_ignores_line_endings_and_bom():
    body = b"a = 1\nb = 2\n"
    assert MS.normalize(body) == body
    assert MS.normalize(b"a = 1\r\nb = 2\r\n") == body
    assert MS.normalize(b"a = 1\rb = 2\r") == body
    assert MS.normalize(b"\xef\xbb\xbfa = 1\nb = 2\n") == body
    # コメントは**除去しない**(但し書きも定義の一部という設計判断)
    assert MS.normalize(b"# c\n") != MS.normalize(b"")


def test_missing_file_is_recorded_not_faked(tmp_path):
    """読めないファイルは "missing" として記録する(欠測を偽の値で埋めない)。"""
    rep = MS.compute(tmp_path / "does_not_exist")
    assert set(rep["files"].values()) == {"missing"}
    assert rep["missing"] == sorted(MS.SPEC_FILES)
    assert len(rep["metrics_spec_hash"]) == 64      # それでもハッシュは出る


def test_manifest_contains_metrics_spec_hash(tmp_path):
    out = _run(tmp_path, "ms_man")
    man = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert man["metrics_spec_hash"] == MS.spec_hash()
    assert man["metrics_spec"]["n_files"] == len(MS.SPEC_FILES)
    assert man["metrics_spec"]["missing"] == []
    assert set(man["metrics_spec"]["files"]) == set(MS.SPEC_FILES)
