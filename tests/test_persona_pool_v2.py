"""ペルソナプール v2 のテスト(PPv2-A/B/C/D/E/F + AGE-B)。

正典: docs/plans/persona-pool-v2-plan.md / docs/plans/age-diversity-plan.md。

検証するもの:
  - **v1 が 1 バイトも動かない**(--v2 なしの経路は v2 のコードを 1 行も通らない)
  - 年齢が全年齢(0-100+)になり、5 歳階級が住基の実数と一致する
  - `np.clip(normal)` のクリップ堆積(v1 は全人口の 4.08% が「15 歳」1 点)が消える
  - 年齢 × 職業の整合(「15 歳の会社員」「30 歳以上の学生」が実質ゼロ)
  - 業種 × 役職の分離(第10-3表の行% と 賃構の役職構成比に寄る)
  - 教職員の内訳(教員:職員 = 実数の 4.4〜5.3 : 1。v1 は i%2 の 1:1)
  - 保育士が**法令の配置基準**から逆算される
  - 世帯の年齢整合(親子の年齢差・子どもが世帯に入る)
  - 姓名が出生コホートに条件づく
  - **職業語彙が v1 の上位集合**であり、新語が全部 src の表に登録済み(静かな不発の防止)
  - `visit_purpose` の 7 語が 1 語も変わらない(presence 較正表と結合)
"""
from __future__ import annotations

import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "src"))

import build_persona_pool as bpp        # noqa: E402
import persona_v2 as PV2                # noqa: E402
from society import economy as E        # noqa: E402
from society.agents import persona as PERS  # noqa: E402

_FRACTION = 0.02                        # 総数 ~20,000(分布統計に十分・数秒)
_ORGS = _ROOT / "data" / "organizations_shibuya_census.json"
_CHILDCARE = _ROOT / "data" / "organizations_shibuya_childcare.json"


def _load_orgs():
    orgs = json.loads(_ORGS.read_text(encoding="utf-8"))
    if _CHILDCARE.exists():
        side = json.loads(_CHILDCARE.read_text(encoding="utf-8"))
        orgs = dict(orgs)
        orgs["schools"] = list(orgs["schools"]) + list(side["schools"])
    return orgs


def _build_v2(out_dir: Path, seed: int = 42, fraction: float = _FRACTION):
    orgs = _load_orgs()
    pop = json.loads((_ROOT / "data" / "shibuya_population.json").read_text(encoding="utf-8"))
    meta, councilors = bpp.build_pool(out_dir, seed, fraction, orgs, pop,
                                      total_target=1_000_000, v2=True)
    return meta, councilors, orgs


def _read(out_dir: Path, layer: str | None = None):
    recs = []
    pat = str(out_dir / (layer or "*") / "*.jsonl")
    for f in sorted(glob.glob(pat)):
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            if line:
                recs.append(json.loads(line))
    return recs


@pytest.fixture(scope="module")
def v2(tmp_path_factory):
    out = tmp_path_factory.mktemp("persona_pool_v2")
    meta, councilors, orgs = _build_v2(out)
    return {"dir": out, "meta": meta, "orgs": orgs,
            "all": _read(out), "L1": _read(out, "L1"), "L2": _read(out, "L2")}


# =========================================================== v1 が動かないこと
def test_v1_path_is_untouched_by_v2(tmp_path):
    """``v2=False``(既定)は v2 のコードを 1 行も通らない = 従来の名簿と完全一致。

    ★これが本レーンの最重要の不変条件。v2 は**別ディレクトリの別世界**であって、
      既存プール・golden・凍結テストの前提を 1 つも動かさない。
    """
    orgs = json.loads((_ROOT / "data" / "organizations_shibuya_wide11k.json")
                      .read_text(encoding="utf-8"))
    pop = json.loads((_ROOT / "data" / "shibuya_population.json").read_text(encoding="utf-8"))
    a, _ = bpp.build_pool(tmp_path / "a", 42, _FRACTION, orgs, pop, 1_000_000)
    b, _ = bpp.build_pool(tmp_path / "b", 42, _FRACTION, orgs, pop, 1_000_000, v2=False)
    assert [(s["file"], s["first_row_blake2b"]) for s in a["shards"]] == \
           [(s["file"], s["first_row_blake2b"]) for s in b["shards"]]
    assert a["schema_version"] == bpp.SCHEMA_VERSION
    assert a["v2"] is False and a["v2_params"] is None


def test_v2_is_deterministic(tmp_path):
    m1, _, _ = _build_v2(tmp_path / "a")
    m2, _, _ = _build_v2(tmp_path / "b")
    assert [(s["file"], s["first_row_blake2b"]) for s in m1["shards"]] == \
           [(s["file"], s["first_row_blake2b"]) for s in m2["shards"]]
    assert m1["layer_counts"] == m2["layer_counts"]
    assert m1["schema_version"] == bpp.V2_SCHEMA_VERSION
    assert m1["v2"] is True


# =========================================================== 全年齢(PPv2-A)
def test_all_ages_exist(v2):
    """0 歳から 85+ まで実在する(v1 は L1 の最小 18 歳・85+ がゼロだった)。"""
    ages = [r["age"] for r in v2["L1"]]
    assert min(ages) == 0
    assert max(ages) >= 85
    n = len(ages)
    kid = sum(1 for a in ages if a < 15) / n
    old = sum(1 for a in ages if a >= 65) / n
    o85 = sum(1 for a in ages if a >= 85) / n
    assert 0.07 <= kid <= 0.13, f"0-14 = {kid:.4f}(実 0.0996)"
    assert 0.15 <= old <= 0.23, f"65+ = {old:.4f}(実 0.1877)"
    assert 0.02 <= o85 <= 0.06, f"85+ = {o85:.4f}(実 0.0383)"


def test_l1_age_bands_match_juki(v2):
    """L1 の 5 歳階級が住基 2026-08-01(総数 231,047)と ±2.0pt 以内。"""
    ages = [r["age"] for r in v2["L1"]]
    n = len(ages)
    tot = sum(m + f for _lo, _hi, m, f in PV2.AGE_BANDS)
    worst = 0.0
    for lo, hi, m, f in PV2.AGE_BANDS:
        sim = sum(1 for a in ages if lo <= a <= hi) / n
        real = (m + f) / tot
        worst = max(worst, abs(sim - real))
    assert worst < 0.020, f"最大乖離 {worst:.4f}"


def test_no_clip_pileup(v2):
    """単一の年齢値への人工的な堆積が無い。

    v1 は L4 の ``clip(normal(36,14), 15, 82)`` が左裾 7.66% を 15 歳 1 点へ積み、
    近傍平均比の過剰が **全人口の 4.08%(40,830 人)** あった。v2 はカテゴリカル抽出
    + 帯内一様なので、残るのは 5 歳階級の境界段差だけ(1% 未満)。
    """
    ac = Counter(r["age"] for r in v2["all"])
    n = len(v2["all"])
    for a, cnt in ac.most_common(10):
        near = sum(ac.get(a + k, 0) for k in (-3, -2, -1, 1, 2, 3)) / 6.0
        assert (cnt - near) / n < 0.015, f"age={a} の過剰 {(cnt-near)/n:.4f}"


def test_age_occupation_consistency(v2):
    """年齢と職業の統計的独立(v1 の欠陥)が解消している。"""
    recs = v2["all"]
    n = len(recs)

    def share(pred):
        return sum(1 for r in recs if pred(r)) / n
    assert share(lambda r: r["occupation"] == "会社員" and r["age"] <= 17) == 0.0
    assert share(lambda r: r["occupation"] == "経営者" and r["age"] <= 22) == 0.0
    assert share(lambda r: r["occupation"] == "公務員" and r["age"] <= 17) == 0.0
    # 社会人学生は実在する(非労働力の「通学」が 30 代にも 68 人)ので 0 は求めない
    assert share(lambda r: r["occupation"] == "学生" and r["age"] >= 30) < 0.01
    # 15 歳未満に就業状態が付かない(労働力率 0)
    assert not [r for r in recs if r["age"] < 15 and r["employment"] != "非就業"]


def test_visit_purpose_is_age_conditioned(v2):
    """7 目的の平均年齢が分かれる(v1 は 7 目的すべて 35.9〜36.1 歳で同一だった)。"""
    by = defaultdict(list)
    for r in v2["all"]:
        if r.get("visit_purpose"):
            by[r["visit_purpose"]].append(r["age"])
    assert set(by) == {p for p, _w in bpp._VISIT_PURPOSES}
    means = {k: sum(v) / len(v) for k, v in by.items()}
    assert means["通院・用事"] > means["エンタメ・イベント"] + 8.0, means
    assert means["ビジネス来訪"] > means["エンタメ・イベント"] + 5.0, means


def test_visit_purpose_vocabulary_is_frozen():
    """★7 語は変更禁止(presence.py の曜日プロファイル表と雨弾性表が完全一致で引く)。"""
    assert [p for p, _w in bpp._VISIT_PURPOSES] == [
        "観光・見物", "買い物", "飲食", "エンタメ・イベント",
        "ビジネス来訪", "友人と会う", "通院・用事"]
    assert set(PV2.PURPOSE_AGE_ELASTICITY) == {p for p, _w in bpp._VISIT_PURPOSES}


# =========================================================== 業種 × 役職(PPv2-B)
def test_orthogonal_axes_present(v2):
    """5 軸が直交で載り、`occupation` は派生欄として残る(後方互換)。"""
    for r in v2["all"]:
        for k in ("industry_key", "industry_major", "occupation_major", "rank",
                  "employment", "birth_year", "workplace_scope", "commute_mode"):
            assert k in r, (k, r["id"])
        assert r["birth_year"] == bpp.V2_BASE_YEAR - r["age"]
        assert r["occupation"]                     # 派生欄は必ず埋まる


def test_role_weights_reproduce_census_row():
    """(業種, ロール) の重みが第10-3表の行% を再現する。"""
    roles = ("エンジニア", "デザイナー", "プロダクトマネージャー", "営業", "コーポレート")
    w = PV2.role_weights("IT", "Webメディア", roles)
    assert abs(sum(w) - 1.0) < 1e-9
    tech = w[0] + w[1] + w[2]                      # 専門技術
    # ★実行%(専門技術 57.8)そのものではなく、**到達できる大分類だけで再正規化した値**に
    #   なる。IT の 5 ロールは 専門技術/事務/販売 の 3 大分類しか覆わないので、
    #   管理 5.3・生産工程 1.4 等の質量(計 7.4)が落ちて 57.8/92.6 = 62.4% になる。
    #   台帳に無いロールを発明しない規律の帰結であり、意図した挙動。
    assert 0.58 <= tech <= 0.66, f"IT の専門技術 = {tech:.3f}(行% 0.578 / 再正規化 0.624)"
    assert 0.06 <= w[3] <= 0.11, f"IT の販売 = {w[3]:.3f}(行% 0.076 / 再正規化 0.082)"
    # v1 の均等ラウンドロビンは各 0.20 だった = 業種内の職種構成を平坦化していた
    assert not all(abs(x - 0.2) < 0.01 for x in w)


def test_rank_pyramid_matches_wage_survey(v2):
    """L2 の役職構成比が賃構(部長 3.74 / 課長 7.22 / 係長 6.49 / 非役職 74.33)に寄る。"""
    rc = Counter(r["rank"] for r in v2["L2"])
    n = len(v2["L2"])
    assert rc["一般"] / n > 0.65
    assert 0.015 <= rc["部長級"] / n <= 0.06, rc["部長級"] / n
    assert 0.03 <= rc["課長級"] / n <= 0.10, rc["課長級"] / n
    # ★係長級/課長級 = 0.90 で 1 を下回る(厳密なピラミッド木ではない)= 実データの形
    assert rc["係長級"] <= rc["課長級"] * 1.15


def test_rank_slots_respects_small_firms():
    """★賃構の母集団は事業所規模 10 人以上。9 人以下の社に部長級を置かない。"""
    assert PV2.rank_slots(4, "IT", 4) == ["一般"] * 4
    assert set(PV2.rank_slots(9, "WR", 9)) == {"一般"}
    big = PV2.rank_slots(500, "IT", 1000)
    assert big.count("部長級") > 0 and big.count("課長級") > 0
    # 職長級は鉱業/建設/製造のみ(公式定義)
    assert "職長級" not in PV2.rank_slots(500, "IT", 1000)
    assert "職長級" in PV2.rank_slots(500, "MF", 1000)


def test_rank_carry_keeps_aggregate_proportions():
    """社をまたぐ端数の持ち越しで、小規模社ばかりでも集計値が実比率へ収束する。"""
    carry: dict[int, float] = {}
    out: list[str] = []
    for _ in range(2000):
        out.extend(PV2.rank_slots(50, "IT", 10, carry=carry))
    c = Counter(out)
    n = len(out)
    assert abs(c["部長級"] / n - 0.046) < 0.008, c["部長級"] / n
    assert abs(c["課長級"] / n - 0.0629) < 0.01, c["課長級"] / n


def test_industry_share_matches_census(v2):
    """L1 就業者の産業構成が渋谷区の常住就業者(第5-3表)に寄る。"""
    emp = [r for r in v2["L1"] if r["employment"] != "非就業"]
    c = Counter(r["industry_key"] for r in emp)
    n = len(emp)
    target = dict(PV2.RESIDENT_INDUSTRY_SHARE)
    tot = sum(target.values())
    for key in ("IT", "WR", "PS"):
        assert abs(c[key] / n - target[key] / tot) < 0.05, (key, c[key] / n)


def test_employment_status_matches_census(v2):
    """従業上の地位。★渋谷区は役員 14.2%・自営系 25.8% の「社長の街」。"""
    emp = [r for r in v2["L1"] if r["employment"] != "非就業"]
    c = Counter(r["employment"] for r in emp)
    n = len(emp)
    assert 0.45 <= c["正規"] / n <= 0.68
    assert 0.05 <= c["役員"] / n <= 0.20
    assert 0.05 <= c["自営業主"] / n <= 0.18


# =========================================================== 教職員・保育(PPv2-D/D2)
def test_school_staff_ratio_is_per_school_type():
    """v1 の ``capacity/12`` 一律 + ``i%2`` の 1:1 割当を、校種別の 2 本の式が置き換える。"""
    t_es, s_es = PV2.school_staff_counts("区立小学校", 620)
    assert t_es == round(620 / 16.16) and s_es == round(620 / 75.6)
    assert 4.0 <= t_es / s_es <= 5.5, "教員:職員 = 実数の 4.4〜5.3 : 1"
    t_hs, s_hs = PV2.school_staff_counts("高校", 900)
    assert 4.5 <= t_hs / s_hs <= 6.5
    # 大学は v1 の 12:1 が約 2 倍の過大(附属病院込みの全国値 15.42 ではなく実効 22.5)
    t_un, _ = PV2.school_staff_counts("大学", 6000)
    assert abs(t_un - round(6000 / 22.5)) <= 1


def test_school_staff_split_in_pool(v2):
    """名簿上の 教員:職員 が 1:1 でない(v1 の ``st_roles[i % 2]`` の是正)。

    比は fraction=1.0 のスロット列で見る(小 fraction では 1 校あたり職員 1 人の床が
    効いて比が潰れる)。生成物の側では両方の役割が実在することだけを見る。
    """
    school_ids = {s["id"] for s in v2["orgs"]["schools"]}
    c_pool = Counter(r["role"] for r in v2["L2"] if r.get("org_id") in school_ids)
    assert c_pool["教員"] > 0 and c_pool["職員"] > 0

    slots = bpp._build_L2_slots_v2(v2["orgs"], 1.0)
    c = Counter(role for (oid, role, *_rest) in slots if oid in school_ids)
    assert c["教員"] / max(1, c["職員"]) > 3.5, c      # 実数は 4.4〜5.3 : 1(v1 は 1:1)
    assert c["保育士"] > 0 and c["調理員"] > 0


def test_nursery_staff_from_statute():
    """保育士は法令の配置基準(0歳3:1 / 1-2歳6:1 / 3歳15:1 / 4-5歳25:1・下限2人)から逆算。"""
    staff, cook = PV2.nursery_staff(91)
    assert staff >= PV2.NURSERY_MIN_STAFF and cook >= 1
    # 手計算: 91*(0.14/3 + 0.30/6 + 0.16/15 + 0.40/25) = 91*0.1213 ≈ 11.0
    assert 9 <= staff <= 14, staff
    assert PV2.nursery_staff(1)[0] == PV2.NURSERY_MIN_STAFF   # 下限 2 人
    # 定員 1,545(bbox 概数)なら保育士は概ね 180〜200 人
    assert 150 <= PV2.nursery_staff(1545)[0] <= 230


def test_childcare_orgs_produce_staff_not_students(v2):
    """保育所・幼稚園は **職員だけ**を生やす。0-5 歳は L1(住民)側に実在する。

    ★保育所を L3(区外から通学する定期来街者)にすると「区外から通園する 0 歳児」になる。
    """
    if not _CHILDCARE.exists():
        pytest.skip("childcare サイドカー台帳が未生成")
    cc = {s["id"] for s in json.loads(_CHILDCARE.read_text(encoding="utf-8"))["schools"]}
    l3 = _read(v2["dir"], "L3")
    assert not [r for r in l3 if r.get("org_id") in cc]
    occ = Counter(r["occupation"] for r in v2["L2"] if r.get("org_id") in cc)
    assert occ["保育士"] > 0 and occ["調理員"] > 0
    # 0-5 歳は住民として実在し、school_stage が行き先を指す
    stages = Counter(r["school_stage"] for r in v2["L1"] if r["age"] <= 5)
    assert stages["保育所"] > 0 and stages["幼稚園"] > 0


# =========================================================== 来街水準(PPv2-E)
def test_l2_attendance_rate_applied(v2):
    """L2 の名簿人数に出勤率 0.62 が掛かり、空いた枠が L4(残余)へ移る。"""
    ledger = sum(c["size"]["employees"] for c in v2["orgs"]["companies"])
    company_ids = {c["id"] for c in v2["orgs"]["companies"]}
    n_company = sum(1 for r in v2["L2"] if r.get("org_id") in company_ids)
    # 式そのものは厳密(丸めは 1 回だけ)
    expect = sum(int(round(c["size"]["employees"] * _FRACTION * PV2.L2_ATTENDANCE_RATE))
                 for c in v2["orgs"]["companies"])
    assert n_company == expect
    # ★比を見るのは fraction=1.0 のとき。小 fraction は 1 社 1 回の丸めで
    #   従業者 40 人未満の社(台帳の大多数)が 0 に落ちるため比が下振れする。
    full = sum(int(round(c["size"]["employees"] * PV2.L2_ATTENDANCE_RATE))
               for c in v2["orgs"]["companies"])
    assert abs(full / ledger - PV2.L2_ATTENDANCE_RATE) < 0.005, full / ledger
    assert n_company < ledger * _FRACTION           # 小 fraction でも必ず減る側
    # 総数は保存されている(移すのは内訳)
    assert v2["meta"]["total_generated"] == round(1_000_000 * _FRACTION)


def test_visit_rate_has_a_tail(v2):
    """「週 1〜2 回来る常連」が実在する(v1 は log-uniform [0.003,0.06] で 1 人も居なかった)。"""
    rates = [r["visit_rate"] for r in v2["all"] if r.get("visit_rate")]
    assert rates
    assert max(rates) > 0.3, max(rates)
    mean = sum(rates) / len(rates)
    assert 0.05 <= mean <= 0.12, mean                  # v1 は 0.0190
    assert sum(1 for x in rates if x >= 0.15) / len(rates) > 0.10


def test_workplace_scope_matches_commute_statistics(v2):
    """区外通勤 55% / 在宅従業 16.8% / 区内 28%(就業者を分母にした率)。"""
    emp = [r for r in v2["L1"] if r["employment"] != "非就業"]
    c = Counter(r["workplace_scope"] for r in emp)
    n = len(emp)
    assert abs(c["out_area"] / n - 0.550) < 0.06, c["out_area"] / n
    assert abs(c["at_home"] / n - 0.168) < 0.05, c["at_home"] / n
    # 非就業者は職場を持たない
    assert all(r["workplace_scope"] == "none"
               for r in v2["L1"] if r["employment"] == "非就業")


# =========================================================== 世帯(PPv2-C)
def test_households_are_age_consistent(v2):
    """親子の年齢差・子どもの同居。v1 は「どの世帯にも子どもが入りえない」構造だった。"""
    hh = defaultdict(list)
    for r in v2["L1"]:
        assert r["household_id"] and r["household_role"] and r["household_type"]
        hh[r["household_id"]].append(r)
    kids = [r for r in v2["L1"] if r["age"] < 18]
    assert kids, "18 歳未満の住民が居ない = v1 の欠陥が再生産されている"
    assert all(len(hh[r["household_id"]]) >= 2 for r in kids), "子が単独世帯に居る"
    bad = 0
    for members in hh.values():
        ck = [m["age"] for m in members if m["household_role"] == "子"]
        pa = [m["age"] for m in members if m["household_role"] in ("世帯主", "夫", "妻")]
        if ck and pa and min(pa) - max(ck) < 16:
            bad += 1
    assert bad / len(hh) < 0.01, f"親子年齢差 < 16 の世帯 {bad}/{len(hh)}"


def test_household_members_are_contiguous_in_the_roster(v2):
    """L1 は**世帯単位で連続して**並ぶ。

    src/society/household.py の ``_pool_partition`` は居住者名簿の連続 n 人を 1 世帯として
    束ねる(名簿の並びが唯一の手がかり)。世帯順に吐いておけば household.py を 1 行も
    触らずに「隣り合う人は実際に家族」が成り立つ。並びは 世帯主 → 配偶者 → 子。
    """
    seen: dict[str, int] = {}
    for i, r in enumerate(v2["L1"]):
        hid = r["household_id"]
        if hid in seen:
            assert i == seen[hid] + 1, f"世帯 {hid} が名簿上で分断されている"
        seen[hid] = i
    for a, b in zip(v2["L1"], v2["L1"][1:]):
        if a["household_id"] == b["household_id"] and b["household_role"] == "子":
            assert a["age"] >= b["age"], (a["age"], b["age"])   # 親が先・年長順


def test_household_size_distribution(v2):
    """単独 64.53%(国勢調査 第26-1表)に寄る。"""
    hh = Counter(r["household_id"] for r in v2["L1"])
    sizes = Counter(min(5, v) for v in hh.values())
    n = len(hh)
    assert abs(sizes[1] / n - 0.6453) < 0.07, sizes[1] / n
    assert abs(len(v2["L1"]) / n - 1.61) < 0.25, len(v2["L1"]) / n


# =========================================================== 姓名(PPv2-F)
def test_names_are_cohort_conditioned(v2):
    """出生年コホートの名しか付かない(v1 は「57 歳の花音」が出ていた)。"""
    fam = set(PV2.family_names(bpp._FAMILY))
    for r in v2["all"]:
        ck = PV2.cohort_key(bpp.V2_BASE_YEAR - r["age"])
        pool = set(PV2.given_pool(ck, r["gender"]))
        hit = any(r["name"][:k] in fam and r["name"][k:] in pool
                  for k in (1, 2, 3))
        assert hit, f"{r['name']} (age={r['age']}, {r['gender']}) がコホート外"


def test_name_diversity_improved(v2):
    """同姓同名が v1(5,760 通り・平均 174 人)より 1 桁改善する。"""
    names = Counter(r["name"] for r in v2["all"])
    assert len(names) > 5760, len(names)
    assert len(PV2.family_names(bpp._FAMILY)) >= 190


# =========================================================== AGE-B 年齢別日課
def test_daily_rhythm_is_age_conditioned(v2):
    """睡眠長と就寝時刻が年齢で分かれる(v1 は全年齢で同一だった)。"""
    by = defaultdict(list)
    for r in v2["all"]:
        by[min(80, (r["age"] // 10) * 10)].append(r)

    def sleep(b):
        return sum(x["sleep_steps"] for x in by[b]) / len(by[b])

    def bed(b):
        return sum((x["bedtime_min"] if x["bedtime_min"] > 700
                    else x["bedtime_min"] + 1440) for x in by[b]) / len(by[b])
    # NHK 表15: きれいな U 字(10 代が長く 40-50 代が最短・70+ で最長に戻る)
    assert sleep(10) > sleep(40) + 2.0, (sleep(10), sleep(40))
    assert sleep(70) > sleep(50) + 2.0, (sleep(70), sleep(50))
    # NHK 表16: 加齢でおよそ 2 時間の位相前進(22 時就寝率は 70+ が 20 代の 4.4 倍)
    assert bed(20) - bed(70) > 90.0, (bed(20), bed(70))
    assert bed(0) < bed(20)


def test_sleep_and_bedtime_are_single_draw():
    """乱数消費は v1 と同数(1 draw)= 層内の以降の draw をずらさない。"""
    ages = np.array([10, 30, 70] * 20)
    genders = ["女", "男"] * 30
    r1 = np.random.default_rng(7)
    r2 = np.random.default_rng(7)
    PV2.sleep_steps_by_age(r1, ages, genders)
    r1_after = r1.random()
    r2.integers(0, PV2.SLEEP_SPREAD_STEPS, len(ages))
    assert r1_after == r2.random()


# =========================================================== 職業語彙(最大リスク)
def test_v2_occupations_are_superset_of_v1(tmp_path, v2):
    """★v2 の職業語彙は v1 の**上位集合**であり、新語は明示登録された 5 語だけ。

    計画書 §7 R1(最大リスク): `occupation` 文字列は事実上の外部 API で、src/ の 25 テーブルと
    conf の 14 箇所が名指しで引く。語彙を壊すと**例外を出さずに**賃金 0・職場なし・
    担い手 0 人になる(第109/111 で実際に起きた)。
    """
    orgs = json.loads((_ROOT / "data" / "organizations_shibuya_wide11k.json")
                      .read_text(encoding="utf-8"))
    pop = json.loads((_ROOT / "data" / "shibuya_population.json").read_text(encoding="utf-8"))
    m1, _ = bpp.build_pool(tmp_path / "v1", 42, 0.05, orgs, pop, 1_000_000)
    v1_vocab = set(m1["occupations"])
    # v1 台帳(census)側の語彙も足す(v2 はこちらの台帳で作る)
    census = _load_orgs()
    m1b, _ = bpp.build_pool(tmp_path / "v1b", 42, 0.05, census, pop, 1_000_000)
    v1_vocab |= set(m1b["occupations"])
    v2_vocab = set(v2["meta"]["occupations"])
    new = v2_vocab - v1_vocab
    assert new <= set(PV2.V2_NEW_OCCUPATIONS), f"未登録の新職業名: {sorted(new)}"


def test_v2_new_occupations_are_registered_in_src_tables():
    """新語が全部 src の表に登録済み = 沈黙の不発(賃金 0・職場なし)が起きない。"""
    for occ in PV2.V2_NEW_OCCUPATIONS:
        grp = E.occupation_group(occ, "")
        if occ in ("未就学児", "年金生活者"):
            # 賃金労働をしない層 = **明示的に無給**と宣言してあること
            assert occ in E.WAGE_CAT and E.WAGE_CAT[occ] is None, occ
            assert E.wage_eligible(occ, "") is False, occ
            assert occ in E.MONEY_INIT, occ
            assert PERS._WORK_CAT.get(occ) is None and occ in PERS._WORK_CAT, occ
        else:
            # 被用者 = wage_profile(産業 × 規模帯 × 職種群)が払う。既定 general へ落ちない
            assert grp != "general", f"{occ} が OCC_GROUP_RULES の既定へ落ちる"
            assert PERS._WORK_CAT.get(occ), f"{occ} に職場 POI カテゴリが無い"
            assert E.wage_eligible(occ, "") is True, occ


def test_every_v2_occupation_resolves_somewhere(v2):
    """v2 の全職業が **賃金/勤務のいずれかの経路で解決する**ことの全数検査。

    解決経路は 4 本(どれか 1 本に当たれば「静かな 0 円」にはならない):
      ① WAGE_CAT(定額の本業日給)② CIVIL_SERVANTS(行政ペイロール)
      ③ ROLE_PAY(L5 役割職の専用経路)④ wage_profile(org 配属者。産業×規模帯×職種群)
    ④ は `wage_eligible=True` かつ職場を持つことが条件なので、そこも一緒に見る。
    """
    cfg = E.build_economy(None)
    unresolved = []
    for occ, n in v2["meta"]["occupations"].items():
        if occ in E.WAGE_CAT or occ in E.CIVIL_SERVANTS:
            continue
        if occ in E.ROLE_PAY:
            continue
        if E.wage_eligible(occ, ""):
            continue
        unresolved.append((occ, n))
    assert not unresolved, f"どの賃金経路にも乗らない職業: {unresolved}"
    assert cfg["wages"]                                   # 経済設定そのものは健在


def test_l2_occupations_have_a_wage_group(v2):
    """域内従業者(L2)の職業が OCC_GROUP_RULES の既定へ大量に落ちない。

    既定 "general" は倍率 1.00 なので 0 円にはならないが、**職種による賃金の差が消える**。
    v2 が新しく作る職業名は 1 語も既定へ落ちないことを固定する。
    """
    c = Counter(r["occupation"] for r in v2["L2"])
    fell = {o: n for o, n in c.items()
            if E.occupation_group(o, "") == "general" and o in PV2.V2_NEW_OCCUPATIONS}
    assert not fell, fell


def test_meta_records_sources_and_honesty(v2):
    """meta に一次統計の出典と**暫定である旨**が残る(実数と偽装しない規律)。"""
    p = v2["meta"]["v2_params"]
    assert p["l2_attendance_rate"] == PV2.L2_ATTENDANCE_RATE
    assert len(p["sources"]) >= 8
    assert "暫定" in p["honesty"] and "ESTIMATE" not in p["honesty"].split("★")[0]
    assert "スクランブル交差点" in p["honesty"]           # 使っていない旨の明記
