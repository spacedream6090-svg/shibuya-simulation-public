"""ペルソナ100万プールの決定論生成テスト(W2 P5)。

scripts/build_persona_pool.py を小 fraction で回し、以下を検証する(実シミュ不要・軽量):
  - 同 seed 同出力(シャード先頭行の blake2b ハッシュ一致)
  - 層別件数が目標範囲(L1〜L5 の比率・合計 = 目標総数)
  - 年齢/職業の周辺分布(L1)が出典統計(shibuya_population.json)と許容誤差内
  - L2/L3学生の org_id が組織台帳に実在
  - 議員(occupation ∈ tools.COUNCILOR_OCCS)が議会定数以上
  - 実在人名の混入なし(名前は手続き生成=姓∈_FAMILY・名∈_GIVEN の合成のみ)
  - presence キーが層ごとに §5 のローテーション区分に一致
  - スキーマが personas_300_civic.json の必須フィールドを満たす
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "src"))

import build_persona_pool as bpp   # noqa: E402
from society.tools import COUNCILOR_OCCS  # noqa: E402

# 必須フィールド(data/personas_300_civic.json = persona.build_agent が読む基本スキーマ)
_BASE_FIELDS = {"name", "age", "gender", "occupation", "visitor", "commute",
                "persona", "traits", "drive_threshold", "fire_weight",
                "bedtime_min", "sleep_steps", "has_bicycle", "has_car"}

_FRACTION = 0.02   # 総数 ~20,000(高速・分布統計に十分)


def _build(out_dir: Path, seed: int = 42, fraction: float = _FRACTION):
    orgs = json.loads((_ROOT / "data" / "organizations_shibuya_wide11k.json")
                      .read_text(encoding="utf-8"))
    pop = json.loads((_ROOT / "data" / "shibuya_population.json")
                     .read_text(encoding="utf-8"))
    meta, councilors = bpp.build_pool(out_dir, seed, fraction, orgs, pop,
                                      total_target=1_000_000)
    return meta, councilors, orgs


def _read_layer(out_dir: Path, layer: str):
    recs = []
    for f in sorted(glob.glob(str(out_dir / layer / "*.jsonl"))):
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            if line:
                recs.append(json.loads(line))
    return recs


@pytest.fixture(scope="module")
def pool(tmp_path_factory):
    out = tmp_path_factory.mktemp("persona_pool")
    meta, councilors, orgs = _build(out)
    return {"dir": out, "meta": meta, "councilors": councilors, "orgs": orgs}


# --------------------------------------------------------------- 決定論
def test_same_seed_same_output(tmp_path):
    """同 seed・同 fraction なら全シャードの先頭行ハッシュが一致(再現性)。"""
    m1, _, _ = _build(tmp_path / "a")
    m2, _, _ = _build(tmp_path / "b")
    h1 = [(s["file"], s["first_row_blake2b"]) for s in m1["shards"]]
    h2 = [(s["file"], s["first_row_blake2b"]) for s in m2["shards"]]
    assert h1 == h2
    assert m1["layer_counts"] == m2["layer_counts"]


def test_different_seed_differs(tmp_path):
    """seed を変えれば出力が変わる(乱数が効いている)。"""
    m1, _, _ = _build(tmp_path / "s1", seed=1)
    m2, _, _ = _build(tmp_path / "s2", seed=2)
    h1 = [s["first_row_blake2b"] for s in m1["shards"]]
    h2 = [s["first_row_blake2b"] for s in m2["shards"]]
    assert h1 != h2


# --------------------------------------------------------------- 層別件数
def test_layer_counts_and_total(pool):
    lc = pool["meta"]["layer_counts"]
    total = pool["meta"]["total_generated"]
    # 合計 = 目標総数(fraction 倍)
    assert total == round(1_000_000 * _FRACTION)
    # 5 層すべて存在
    for ly in ("L1", "L2", "L3", "L4", "L5"):
        assert lc.get(ly, 0) > 0
    # L4(非定期来街)が最大の回転層
    assert lc["L4"] == max(lc.values())
    # L2(域内従業者)は台帳 employees 由来 → 全体の 20〜30% レンジ
    assert 0.18 < lc["L2"] / total < 0.32
    # L1(住民)は ~3% 規模(30k / 1M)
    assert 0.02 < lc["L1"] / total < 0.05


# --------------------------------------------------------------- L1 周辺分布の整合
def test_L1_age_gender_marginals(pool):
    pop = json.loads((_ROOT / "data" / "shibuya_population.json")
                     .read_text(encoding="utf-8"))
    l1 = _read_layer(pool["dir"], "L1")
    n = len(l1)
    assert n > 400
    # 性別
    frac_f = sum(1 for r in l1 if r["gender"] == "女") / n
    assert abs(frac_f - pop["gender"]["女"]) < 0.06
    # 年齢帯
    bands = pop["age_bands"]

    def band_of(a):
        for b in bands:
            if b["band"][0] <= a <= b["band"][1]:
                return f"{b['band'][0]}-{b['band'][1]}"
        return "oob"
    from collections import Counter
    got = Counter(band_of(r["age"]) for r in l1)
    for b in bands:
        key = f"{b['band'][0]}-{b['band'][1]}"
        assert abs(got[key] / n - b["share"]) < 0.06, key


def test_L1_occupation_marginals(pool):
    pop = json.loads((_ROOT / "data" / "shibuya_population.json")
                     .read_text(encoding="utf-8"))
    l1 = _read_layer(pool["dir"], "L1")
    n = len(l1)
    from collections import Counter
    got = Counter(r["occupation"] for r in l1)
    src = {o["name"]: o["share"] for o in pop["occupations"]}
    # 上位職(会社員=0.30)の乖離が小さいこと
    assert abs(got["会社員"] / n - src["会社員"]) < 0.06
    # 生成された職業はすべて出典の職業語彙に含まれる
    assert set(got).issubset(set(src))


# --------------------------------------------------------------- 需要駆動(org_id 実在)
def test_L2_org_ids_exist(pool):
    orgs = pool["orgs"]
    valid = set(c["id"] for c in orgs["companies"]) | set(s["id"] for s in orgs["schools"])
    l2 = _read_layer(pool["dir"], "L2")
    assert l2
    for r in l2:
        assert r["org_id"] in valid
        assert r["role"]
        assert r["presence"] == "workday_shift"
        assert "shift_pattern" in r


def test_L3_student_org_ids_exist(pool):
    orgs = pool["orgs"]
    school_ids = set(s["id"] for s in orgs["schools"])
    l3 = _read_layer(pool["dir"], "L3")
    students = [r for r in l3 if r.get("subtype") == "student"]
    regulars = [r for r in l3 if r.get("subtype") == "regular"]
    assert students and regulars
    for r in students:
        assert r["org_id"] in school_ids
        assert r["visit_cadence"] == "school_day"


# --------------------------------------------------------------- 議員(定数以上)
def test_councilors_meet_quorum(pool):
    councilors = pool["councilors"]
    # config の assembly.size 既定(9)以上。現実の渋谷区議会=34 を採用
    assert len(councilors) >= 9
    assert len(councilors) == bpp.N_COUNCILORS == 34
    for c in councilors:
        assert c["occupation"] in COUNCILOR_OCCS
        assert c["visitor"] is False        # from_roster は非来街者のみ着席
        assert "seat_id" in c
        # 基本スキーマを満たす(from_roster→build_agent が読める)
        assert _BASE_FIELDS.issubset(set(c))


def test_councilors_json_written(tmp_path):
    """personas_councilors.json が標準スキーマで書ける(from_roster 互換)。"""
    _, councilors, _ = _build(tmp_path / "c")
    # ★コミット対象の実ファイルは上書きしない(temp に書いて検証)
    path = bpp._write_councilors_json(councilors, seed=42, path=tmp_path / "councilors.json")
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["meta"]["count"] == len(councilors) >= 9
    assert all(p["occupation"] in COUNCILOR_OCCS for p in doc["personas"])
    assert all(not p["visitor"] for p in doc["personas"])


def test_councilors_fraction_independent(tmp_path):
    """議員名簿は --fraction に依存しない(固定 id・専用乱数ストリーム)。"""
    _, c_small, _ = _build(tmp_path / "small", fraction=0.02)
    _, c_full, _ = _build(tmp_path / "big", fraction=0.2)
    assert [c["id"] for c in c_small] == [c["id"] for c in c_full]
    assert [c["name"] for c in c_small] == [c["name"] for c in c_full]


# --------------------------------------------------------------- 名前の手続き生成(実在人名なし)
def test_names_are_procedural(pool):
    """全レコードの名前が 姓∈_FAMILY + 名∈_GIVEN の合成であること(実在人物名を作らない)。"""
    fam = bpp._FAMILY_SET
    giv = bpp._GIVEN_SET

    def name_ok(name: str) -> bool:
        for f in fam:
            if name.startswith(f) and name[len(f):] in giv:
                return True
        return False
    bad = []
    for ly in ("L1", "L2", "L3", "L4", "L5"):
        for r in _read_layer(pool["dir"], ly):
            if not name_ok(r["name"]):
                bad.append(r["name"])
    assert not bad, f"手続き生成外の名前: {bad[:10]}"


# --------------------------------------------------------------- presence キーと基本スキーマ
def test_presence_keys_per_layer(pool):
    expect = {"L1": {"resident"}, "L2": {"workday_shift"}, "L3": {"cadence"},
              "L4": {"stochastic"}, "L5": {"duty", "resident"}}
    for ly, allowed in expect.items():
        keys = set(r["presence"] for r in _read_layer(pool["dir"], ly))
        assert keys.issubset(allowed), (ly, keys)


def test_base_schema_and_types(pool):
    """全レコードが基本スキーマの必須フィールドを持ち、commuter は追加属性を持つ。"""
    for ly in ("L1", "L2", "L3", "L4", "L5"):
        for r in _read_layer(pool["dir"], ly):
            assert _BASE_FIELDS.issubset(set(r)), (ly, _BASE_FIELDS - set(r))
            assert set(r["traits"]) == {"internal_locus", "nfc", "risk_tolerance"}
            assert 0.30 <= r["drive_threshold"] <= 0.85
            assert 0.15 <= r["fire_weight"] <= 0.90
            if r["commute"]:
                assert "residence_line" in r


def test_llm_targets_within_range(pool):
    """LLM 上塗り対象(llm_targets.json)が全体の 2〜5% レンジ(L5全員+L3常連+L1一部)。"""
    doc = json.loads((pool["dir"] / "llm_targets.json").read_text(encoding="utf-8"))
    total = pool["meta"]["total_generated"]
    frac = doc["count"] / total
    assert 0.015 <= frac <= 0.06, frac
    # 一意 id・重複なし
    assert len(doc["ids"]) == len(set(doc["ids"])) == doc["count"]


# ================================================ レーン D1(第109): stale shard の掃除
# 第108 の発見: ビルダは part を**上書きするだけ**で古い part を消さない。層が縮む再生成
# (L2 が減る等)を掛けると前回の末尾 part が残り、ディレクトリを直に舐める外部ツールが
# 幽霊レコードを数える(PoolStore は meta.shards しか読まないのでプールの中身は汚れない)。
# 既定は**破壊的でない**(警告 + meta へ非参照の明示)。--clean のときだけ削除する。
def _stale_setup(out_dir: Path) -> Path:
    """既存プールへ「今回は書き直されない」古い part を仕込む。"""
    ghost = out_dir / "L2" / "part-9999.jsonl"
    ghost.write_text('{"id": "GHOST", "layer": "L2"}\n', encoding="utf-8")
    return ghost


def test_stale_part_is_detected_and_left_in_place_by_default(tmp_path):
    """既定(--clean なし)は**消さずに**検出し、meta へ非参照として列挙する。"""
    out = tmp_path / "stale_warn"
    _build(out)
    ghost = _stale_setup(out)
    meta, _c, _o = _build(out)                       # 同じ出力先へ再生成

    assert meta["stale_shards"] == ["L2/part-9999.jsonl"]
    assert meta["stale_shards_removed"] == []
    assert ghost.exists(), "既定で古い part を消してしまっている(破壊的)"
    # 幽霊は meta.shards から参照されない = PoolStore の索引に載らない
    assert "L2/part-9999.jsonl" not in {s["file"] for s in meta["shards"]}


def test_stale_part_is_removed_with_clean(tmp_path):
    """--clean 相当(build_pool(clean=True))で古い part を削除し、meta へ記録する。"""
    out = tmp_path / "stale_clean"
    _build(out)
    ghost = _stale_setup(out)
    orgs = json.loads((_ROOT / "data" / "organizations_shibuya_wide11k.json")
                      .read_text(encoding="utf-8"))
    pop = json.loads((_ROOT / "data" / "shibuya_population.json").read_text(encoding="utf-8"))
    meta, _councilors = bpp.build_pool(out, 42, _FRACTION, orgs, pop,
                                       total_target=1_000_000, clean=True)

    assert meta["stale_shards"] == []
    assert meta["stale_shards_removed"] == ["L2/part-9999.jsonl"]
    assert not ghost.exists(), "--clean で古い part が消えていない"


def test_clean_never_touches_the_parts_written_this_run(tmp_path):
    """--clean は**今回書いた part を 1 つも消さない**(scan は生成の前に採る)。"""
    out = tmp_path / "stale_safe"
    meta1, _c, _o = _build(out)
    orgs = json.loads((_ROOT / "data" / "organizations_shibuya_wide11k.json")
                      .read_text(encoding="utf-8"))
    pop = json.loads((_ROOT / "data" / "shibuya_population.json").read_text(encoding="utf-8"))
    meta2, _councilors = bpp.build_pool(out, 42, _FRACTION, orgs, pop,
                                        total_target=1_000_000, clean=True)
    assert meta2["stale_shards_removed"] == [] == meta2["stale_shards"]
    assert {s["file"] for s in meta1["shards"]} == {s["file"] for s in meta2["shards"]}
    for s in meta2["shards"]:
        assert (out / s["file"]).exists(), f"今回書いた part が消えている: {s['file']}"


def test_stale_part_leaks_into_llm_targets_until_cleaned(tmp_path):
    """★影響の切り分け: 名簿(meta.shards)は汚れないが llm_targets.json は汚れる。

    `_collect_llm_targets` は層ディレクトリを **glob する**ので、幽霊 part の id が
    深いペルソナ化の対象名簿へ入り込む。--clean はその手前で消すので混ざらない
    (= stale の始末を `_collect_llm_targets` の**前**に置いていることの機械固定)。
    """
    orgs = json.loads((_ROOT / "data" / "organizations_shibuya_wide11k.json")
                      .read_text(encoding="utf-8"))
    pop = json.loads((_ROOT / "data" / "shibuya_population.json").read_text(encoding="utf-8"))

    def ghost_in_targets(out: Path) -> bool:
        doc = json.loads((out / "llm_targets.json").read_text(encoding="utf-8"))
        return "GHOST_L5" in set(doc["ids"])

    def seed_ghost(out: Path) -> None:
        # L5 は「全員が対象」なので、幽霊が混ざるかどうかが一番はっきり出る層
        (out / "L5" / "part-9999.jsonl").write_text(
            '{"id": "GHOST_L5", "layer": "L5"}\n', encoding="utf-8")

    dirty = tmp_path / "leak_warn"
    _build(dirty)
    seed_ghost(dirty)
    bpp.build_pool(dirty, 42, _FRACTION, orgs, pop, total_target=1_000_000)
    assert ghost_in_targets(dirty), "幽霊が llm_targets に混ざらない(前提が崩れた)"

    clean = tmp_path / "leak_clean"
    _build(clean)
    seed_ghost(clean)
    bpp.build_pool(clean, 42, _FRACTION, orgs, pop, total_target=1_000_000, clean=True)
    assert not ghost_in_targets(clean), "--clean でも llm_targets に幽霊が残っている"


def test_cli_declares_the_clean_flag():
    """--clean が CLI に生えている(既定 False = 破壊的でない)。"""
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.suppress(SystemExit):
        bpp.main(["--help"])
    assert "--clean" in buf.getvalue()
