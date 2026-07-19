"""組織台帳の分布駆動生成(P6: 52 → ~1.1万)のテスト。

検収基準(docs/plans/w2-execution-plan.md §4.1 P6):
- 生成の決定論(同 seed 同出力・別 seed で相違)
- 産業大分類×従業者規模帯の実現分布が *目標分布* と許容誤差内
- 全組織に workplace_poi(建物割付)が付く(POI割付率 100%)
- 従業者総和が目標 25.7万 ±10%
- 実在企業名の混入なし(OSM 由来の実在名と disjoint・簡易ブロックリスト)

生成は決定論・オフライン(ネットワーク不使用)。scripts/build_orgs.py を importlib で読み込む
(pythonpath は src のみ=既存 test_make_env.py と同じ流儀)。
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MAP = REPO / "data" / "shibuya_osm_wide_v7.json"
COMMITTED = REPO / "data" / "organizations_shibuya_wide11k.json"

TARGET_EMPLOYEES = 257_000
COUNT = 11_000
SEED = 42


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BO = _load("build_orgs", "scripts/build_orgs.py")


@pytest.fixture(scope="module")
def ledger() -> dict:
    """seed=42・count=11000 の分布駆動台帳(モジュール内で一度だけ生成)。"""
    return BO.build_ledger_dist(MAP, COUNT, SEED)


# --------------------------------------------------------------------- 決定論
def test_deterministic_same_seed():
    """同 seed・同 count で会社列がバイト一致(seq id・名前・規模・POI 全て)。"""
    a = BO.build_ledger_dist(MAP, 2000, SEED)["companies"]
    b = BO.build_ledger_dist(MAP, 2000, SEED)["companies"]
    dump = lambda xs: json.dumps(xs, ensure_ascii=False, sort_keys=True)
    assert dump(a) == dump(b), "同 seed で出力が非決定的"


def test_different_seed_differs():
    """別 seed では名称・規模の割当が変わる(生成が seed に効いている)。"""
    a = BO.build_ledger_dist(MAP, 2000, 42)["companies"]
    c = BO.build_ledger_dist(MAP, 2000, 7)["companies"]
    dump = lambda xs: json.dumps(xs, ensure_ascii=False, sort_keys=True)
    assert dump(a) != dump(c), "別 seed でも同一出力(seed が効いていない)"


# ----------------------------------------------------------- 産業×規模帯の分布整合
def test_company_count(ledger):
    assert len(ledger["companies"]) == COUNT
    assert ledger["meta"]["counts"]["schools"] >= 1  # 学校ブロックが同梱される


def test_industry_distribution_matches_target(ledger):
    """実現の産業別件数シェアが目標(正規化した share)と ±0.01 以内。"""
    n = len(ledger["companies"])
    realized: dict[str, float] = {}
    for c in ledger["companies"]:
        realized[c["industry_key"]] = realized.get(c["industry_key"], 0) + 1
    shares = BO._normalize([it["share"] for it in BO.INDUSTRY_SPEC])
    target = {it["key"]: s for it, s in zip(BO.INDUSTRY_SPEC, shares)}
    for key, tgt in target.items():
        got = realized.get(key, 0) / n
        assert abs(got - tgt) <= 0.01, f"{key}: 実現 {got:.4f} が目標 {tgt:.4f} から乖離"
    # 全産業大分類が代表される(share>0 のものは 1 件以上)
    assert set(realized) == set(target), "生成されていない産業大分類がある"


def test_size_band_distribution_matches_target(ledger):
    """実現の規模帯シェアが、目標(産業share×規模クラス分布の合成)と ±0.02 以内。"""
    band_labels = [b[0] for b in BO.SIZE_BANDS]
    n = len(ledger["companies"])
    realized = {lab: 0 for lab in band_labels}
    for c in ledger["companies"]:
        realized[c["size"]["band"]] += 1
    # 目標の合成: Σ_産業 share_i × 正規化規模クラス分布
    shares = BO._normalize([it["share"] for it in BO.INDUSTRY_SPEC])
    expected = {lab: 0.0 for lab in band_labels}
    for it, s in zip(BO.INDUSTRY_SPEC, shares):
        cls = BO._normalize(BO.SIZE_CLASS_DIST[it["cls"]])
        for lab, p in zip(band_labels, cls):
            expected[lab] += s * p
    for lab in band_labels:
        got = realized[lab] / n
        assert abs(got - expected[lab]) <= 0.02, \
            f"規模帯 {lab}: 実現 {got:.4f} が目標 {expected[lab]:.4f} から乖離"


def test_employees_total_within_tolerance(ledger):
    """従業者総和が目標 25.7万 ±10% 以内(硬いアンカー=shibuya-population.md §5)。"""
    total = sum(int(c["size"]["employees"]) for c in ledger["companies"])
    lo, hi = TARGET_EMPLOYEES * 0.9, TARGET_EMPLOYEES * 1.1
    assert lo <= total <= hi, f"従業者総和 {total} が {int(lo)}〜{int(hi)} の外"
    # 規模帯の下限・上限に従業者が収まる(サンプリングの健全性)
    bounds = {b[0]: (b[1], b[2]) for b in BO.SIZE_BANDS}
    for c in ledger["companies"]:
        lo_b, hi_b = bounds[c["size"]["band"]]
        assert lo_b <= int(c["size"]["employees"]) <= hi_b, \
            f"{c['id']}: 従業者 {c['size']['employees']} が規模帯 {c['size']['band']} 外"


# --------------------------------------------------------------- POI(建物)割付
def test_all_orgs_have_workplace_poi(ledger):
    """全会社に workplace_poi が付き、node は地図の実ノード・建物は実在建物。"""
    nodes = {n["id"] for n in json.loads(MAP.read_text(encoding="utf-8"))["nodes"]}
    bset = {b["id"] for b in json.loads(MAP.read_text(encoding="utf-8"))["buildings"]}
    for c in ledger["companies"]:
        wp = c.get("workplace_poi")
        assert wp, f"{c['id']}: workplace_poi 欠落"
        assert wp["node"] in nodes, f"{c['id']}: workplace_poi.node が地図に無い"
        assert wp["building"] in bset, f"{c['id']}: workplace_poi.building が地図に無い"
        assert wp["floor"] >= 1
    assert ledger["meta"]["poi_coverage"] == 1.0


def test_buildings_are_multi_tenant(ledger):
    """建物×階に複数組織を束ねる(オフィス=多数・路面=少数)の実現。"""
    from collections import Counter
    per_b = Counter(c["workplace_poi"]["building"] for c in ledger["companies"])
    assert max(per_b.values()) > 1, "どの建物も単一テナント(多テナント束ねが効いていない)"
    # 少数テナントの建物も存在する(路面店的な束ね)
    assert min(per_b.values()) >= 1
    assert len(per_b) < len(ledger["companies"]), "1組織1建物になっている(束ねていない)"


# ------------------------------------------------------------- 実在企業名の非混入
def test_no_real_company_names(ledger):
    """生成名が OSM 由来の実在名(建物・POI)と disjoint、かつ既知実在社名を含まない(R17)。"""
    _, real_names = BO.load_buildings(MAP)
    gen = [c["name"] for c in ledger["companies"]]
    genset = set(gen)
    leak = genset & real_names
    assert not leak, f"OSM 実在名が台帳に混入: {sorted(leak)[:5]}"
    # 簡易ブロックリスト(有名実在企業。部分一致でも弾く)
    blocked = ["サイバーエージェント", "GMO", "ミクシィ", "mixi", "DeNA", "LINE",
               "楽天", "Google", "Amazon", "アムウェイ", "ヒカリエ", "PARCO", "パルコ",
               "東急", "セブン-イレブン", "ローソン", "スターバックス", "マクドナルド"]
    for name in gen:
        for tok in blocked:
            assert tok not in name, f"実在企業トークン '{tok}' が生成名 '{name}' に混入"
    # id は一意
    ids = [c["id"] for c in ledger["companies"]]
    assert len(set(ids)) == len(ids), "組織 id が重複"


# ------------------------------------------------------- 既存台帳を上書きしない/新ファイル
def test_committed_artifact_reproducible(ledger):
    """出力先の新ファイルが存在し、seed=42 の再生成と会社列が一致(再現性)。"""
    assert COMMITTED.exists(), "data/organizations_shibuya_wide11k.json が無い"
    disk = json.loads(COMMITTED.read_text(encoding="utf-8"))
    assert disk["meta"]["mode"] == "distribution-driven"
    dump = lambda xs: json.dumps(xs, ensure_ascii=False, sort_keys=True)
    assert dump(disk["companies"]) == dump(ledger["companies"]), \
        "コミット済み台帳が seed=42 再生成と不一致(再生成手順の不整合)"


def test_schema_consumable_by_engine(ledger):
    """society.organizations の消費関数(org_line/daily_output/load_book)が動く schema。"""
    import sys
    sys.path.insert(0, str(REPO / "src"))
    from society import organizations as O
    book = {c["id"]: c for c in ledger["companies"]}
    for c in list(ledger["companies"])[:50]:
        assert O.daily_output(c, 0) is not None, f"{c['id']}: daily_output が None"
        line = O.org_line(c, c["roles"][0])
        assert line.startswith("職場: ") and c["name"] in line
    assert len(O.employer_ids(book)) == len(ledger["companies"])
