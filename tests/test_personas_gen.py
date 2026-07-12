"""ペルソナ自動生成パイプライン(scripts/gen_personas.py)の検収テスト。

検査:
  (a) 同一 seed で完全同一出力(決定論)。
  (b) スキーマが data/personas_100_civic.json と同一キー集合。
  (c) 抽出結果に公務員(区職員/警察官/消防士)が最低3名含まれる。
  (d) 名前の重複が無い。
  (e) pool > sample のとき抽出は pool の部分集合。
scripts/ はパッケージではないので sys.path をテスト内で調整して import する。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import gen_personas as gp  # noqa: E402

_CIVIL = {"区職員", "警察官", "消防士"}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _key_union(personas: list[dict]) -> set[str]:
    keys: set[str] = set()
    for p in personas:
        keys |= set(p.keys())
    return keys


def _gen(tmp_path: Path, *, pool: int, sample: int, seed: int,
         with_pool: bool = False) -> tuple[dict, dict | None]:
    out = tmp_path / f"sample_{seed}_{pool}_{sample}.json"
    argv = ["--pool", str(pool), "--sample", str(sample),
            "--seed", str(seed), "--out", str(out)]
    pool_out = None
    if with_pool:
        pool_out = tmp_path / f"pool_{seed}_{pool}.json"
        argv += ["--pool-out", str(pool_out)]
    gp.main(argv)
    return _load(out), (_load(pool_out) if pool_out else None)


def test_deterministic_same_seed(tmp_path, capsys):
    """(a) 同一 seed → バイト列まで完全一致。異なる seed → 異なる。"""
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    c = tmp_path / "c.json"
    gp.main(["--pool", "300", "--sample", "80", "--seed", "42", "--out", str(a)])
    gp.main(["--pool", "300", "--sample", "80", "--seed", "42", "--out", str(b)])
    gp.main(["--pool", "300", "--sample", "80", "--seed", "7", "--out", str(c)])
    assert a.read_bytes() == b.read_bytes(), "同一 seed で出力が一致しない"
    assert a.read_bytes() != c.read_bytes(), "seed 差で出力が変わらない"


def test_schema_matches_reference(tmp_path, capsys):
    """(b) 既存 data/personas_100_civic.json とキー集合(union)が一致する。"""
    ref = _load(REPO_ROOT / "data" / "personas_100_civic.json")["personas"]
    gen, _ = _gen(tmp_path, pool=500, sample=100, seed=42)
    gen_personas = gen["personas"]
    assert _key_union(gen_personas) == _key_union(ref)
    # 各ペルソナのキーは参照の union の部分集合(未知キーを増やさない)
    ref_union = _key_union(ref)
    for p in gen_personas:
        assert set(p.keys()) <= ref_union
    # 型・値域も参照に整合(代表フィールド)
    for p in gen_personas:
        assert isinstance(p["age"], int) and 18 <= p["age"] <= 75
        assert p["gender"] in {"男", "女"}
        assert set(p["traits"]) == {"internal_locus", "nfc", "risk_tolerance"}
        for v in p["traits"].values():
            assert 0.0 <= v <= 1.0
        assert 0.30 <= p["drive_threshold"] <= 0.85
        assert 0.15 <= p["fire_weight"] <= 0.90
        assert isinstance(p["visitor"], bool) and isinstance(p["commute"], bool)


def test_civil_servants_present(tmp_path, capsys):
    """(c) 抽出 100 名に公務員が最低3名。"""
    gen, _ = _gen(tmp_path, pool=500, sample=100, seed=42)
    n_civil = sum(1 for p in gen["personas"] if p["occupation"] in _CIVIL)
    assert n_civil >= 3, f"公務員が {n_civil} 名しかいない(>=3 が必要)"


def test_no_duplicate_names(tmp_path, capsys):
    """(d) 抽出・プールとも名前の重複が無い。"""
    gen, pool = _gen(tmp_path, pool=500, sample=100, seed=42, with_pool=True)
    sample_names = [p["name"] for p in gen["personas"]]
    assert len(sample_names) == len(set(sample_names)), "抽出名簿に同名"
    pool_names = [p["name"] for p in pool["personas"]]
    assert len(pool_names) == len(set(pool_names)), "プールに同名"


def test_sample_is_subset_of_pool(tmp_path, capsys):
    """(e) pool > sample のとき抽出は pool の部分集合。"""
    gen, pool = _gen(tmp_path, pool=500, sample=100, seed=42, with_pool=True)
    assert len(pool["personas"]) > len(gen["personas"])
    pool_names = {p["name"] for p in pool["personas"]}
    sample_names = {p["name"] for p in gen["personas"]}
    assert sample_names <= pool_names, "抽出がプールの部分集合でない"
