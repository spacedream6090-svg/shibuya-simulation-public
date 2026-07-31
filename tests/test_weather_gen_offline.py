"""天候の生成型実データ化 前半のテスト(第79バッチ 2026-08-01)。

対象は **scripts/ と data/snapshot/ だけ**(src/ には一切触れない=シム挙動は不変)。
検証するもの:
  A. scripts/fetch_weather_history.py — HTML パース・スキーマ検証・SHA-256(ネットワーク不使用)
  B. scripts/fit_weather_gen.py       — 較正の決定論・数値ユーティリティの正しさ
  C. 生成器のオフライン統計検証        — 較正パラメータから合成した系列が実測の
                                        気温分布・猛暑連長分布に KS/モーメントで近いこと
  D. 現行合成モデル(src/society/weather.py)の鏡が本物と一致すること + 構造的欠陥の固定
     (=36℃以上が出せない・日間自己相関がゼロ)

ネットワークは一切使わない。実データファイル(data/snapshot/*.json)が無い環境では
C 群は skip する(A/B/D は常に走る)。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import fetch_weather_history as FH        # noqa: E402
import fit_weather_gen as FG              # noqa: E402

HISTORY = REPO_ROOT / "data" / "snapshot" / "weather_tokyo_aug.json"
PARAMS = REPO_ROOT / "data" / "snapshot" / "weather_gen_params.json"
needs_data = pytest.mark.skipif(
    not (HISTORY.exists() and PARAMS.exists()),
    reason="data/snapshot/weather_*.json が未生成(scripts/fetch_weather_history.py 先行要)")


# ============================================================ A. fetch のパースとスキーマ
def _row(day: str, cells: list[str]) -> str:
    """daily_s1.php の1行を合成する(実 HTML の形だけを写した中立データ)。"""
    assert len(cells) == FH.N_CELLS - 1
    tds = "".join(f'<td class="data_0_0">{c}</td>' for c in cells)
    return ('<tr class="mtx" style="text-align:right;">'
            f'<td style="white-space:nowrap"><div class="a_print">'
            f'<a href="hourly_s1.php">{day}</a></div></td>' + tds + "</tr>")


def _synth_html(rows: list[str]) -> str:
    return ("<html><body><table id='tablefix1' class='data2_s'>"
            + "".join(rows) + "</table></body></html>")


# 1日目: 通常値 / 2日目: 「--」(現象なし) / 3日目: 注記つき(準正常値・資料不足)
_CELLS_NORMAL = ["999.0", "1001.8", "6.0", "4.5", "4.0", "26.4", "30.1", "23.8",
                 "83", "65", "3.3", "5.8", "北北東", "12.5", "北", "0.6", "--", "--",
                 "曇時々雨", "雨時々曇"]
_CELLS_DASH = ["1005.0", "1008.0", "--", "--", "--", "28.5", "33.5", "24.5",
               "74", "51", "2.4", "4.6", "東南東", "9.5", "東", "9.3", "--", "--",
               "晴", "晴"]
_CELLS_ANNOT = ["1005.0", "1008.0", "0.0)", "0.0", "0.0", "29.0", "34.0]", "25.0",
                "70", "50", "2.0", "4.0", "南", "8.0", "南", "8.0", "--", "--",
                "薄曇", "曇"]


def test_parse_daily_html_basic():
    html = _synth_html([_row("1", _CELLS_NORMAL), _row("2", _CELLS_DASH),
                        _row("3", _CELLS_ANNOT)])
    recs = FH.parse_daily_html(html, 2025, 8)
    assert len(recs) == 3
    r0 = recs[0]
    assert r0["date"] == "2025-08-01" and r0["year"] == 2025 and r0["day"] == 1
    assert r0["temp_hi"] == 30.1 and r0["temp_lo"] == 23.8 and r0["temp_mean"] == 26.4
    assert r0["precip_mm"] == 6.0 and r0["humid_mean"] == 83.0 and r0["humid_min"] == 65.0
    assert r0["sunshine_h"] == 0.6 and r0["wind_mean"] == 3.3
    assert r0["cond_day"] == "曇時々雨" and r0["cond_night"] == "雨時々曇"
    # 出力に残すのは KEEP 列だけ(気圧・風向は落とす)
    assert "pres_local" not in r0 and "wind_max_dir" not in r0


def test_parse_dash_means_zero_for_precip_and_snow():
    """気象庁の「--」= 現象なし。降水/降雪/日照は 0.0、気温は 0 と偽らない。"""
    html = _synth_html([_row("2", _CELLS_DASH)])
    r = FH.parse_daily_html(html, 2024, 8)[0]
    assert r["precip_mm"] == 0.0
    assert r["snow_cm"] == 0.0
    assert r["temp_hi"] == 33.5           # 気温は「--」ではないので通常値


def test_parse_annotations_recorded_as_flags():
    """準正常値『)』・資料不足『]』は値を採りつつ flags に残す(黙って落とさない)。"""
    html = _synth_html([_row("3", _CELLS_ANNOT)])
    r = FH.parse_daily_html(html, 2024, 8)[0]
    assert r["precip_mm"] == 0.0 and r["temp_hi"] == 34.0
    assert r["flags"]["precip_mm"] == "quasi_normal"
    assert r["flags"]["temp_hi"] == "insufficient"


def test_parse_rejects_layout_change():
    """列数が変わったら黙って埋めずに落ちる(気象庁のレイアウト変更の検知)。"""
    bad = ('<tr class="mtx" style="text-align:right;">'
           + "".join(f"<td>{i}</td>" for i in range(5)) + "</tr>")
    with pytest.raises(ValueError, match="列数"):
        FH.parse_daily_html(_synth_html([bad]), 2025, 8)
    with pytest.raises(ValueError, match="行が見つからない"):
        FH.parse_daily_html("<html></html>", 2025, 8)


def _doc(days=None):
    html = _synth_html([_row("1", _CELLS_NORMAL), _row("2", _CELLS_DASH)])
    days = days if days is not None else FH.parse_daily_html(html, 2025, 8)
    return FH.build_document(days, FH.STATIONS["tokyo"], [2025], [8],
                             urls=["https://example.invalid/x"],
                             fetched_at_utc="2026-08-01T00:00:00Z")


def test_document_validates_and_carries_license():
    doc = _doc()
    assert FH.validate_document(doc) == []
    src = doc["meta"]["source"]
    assert "気象庁" in src["attribution"]                 # PDL1.0 の出典明示は必須条件
    assert src["license_url"].startswith("https://")
    assert doc["meta"]["caveats"], "観測点移転などの限界注記が空"
    assert any("2014-12-02" in c for c in doc["meta"]["caveats"])


def test_payload_sha256_ignores_fetch_time_but_not_data():
    """同じ観測値なら取得日時が違っても同じハッシュ / 値を1つ変えれば必ず変わる。"""
    a = _doc()
    b = _doc()
    b["meta"]["fetched_at_utc"] = "2030-01-01T00:00:00Z"
    assert FH.payload_sha256(a) == FH.payload_sha256(b)
    c = _doc()
    c["days"][0]["temp_hi"] = 30.2
    assert FH.payload_sha256(a) != FH.payload_sha256(c)


def test_validation_catches_tampering():
    doc = _doc()
    doc["days"][0]["temp_hi"] = 30.2            # ハッシュを直さずに書き換える
    problems = FH.validate_document(doc)
    assert any("payload_sha256 不一致" in p for p in problems)

    doc2 = _doc()
    doc2["days"][0]["temp_lo"] = 99.0           # 最低 > 最高 の物理矛盾
    doc2["meta"]["payload_sha256"] = FH.payload_sha256(doc2)
    assert any("temp_lo > temp_hi" in p for p in FH.validate_document(doc2))

    doc3 = _doc()
    doc3["meta"]["source"]["attribution"] = ""  # 出典表示を落とす
    doc3["meta"]["payload_sha256"] = FH.payload_sha256(doc3)
    assert any("attribution" in p for p in FH.validate_document(doc3))

    doc4 = _doc()
    doc4["days"].append(dict(doc4["days"][0]))  # 日付重複
    doc4["meta"]["n_days"] = len(doc4["days"])
    doc4["meta"]["payload_sha256"] = FH.payload_sha256(doc4)
    assert any("日付重複" in p for p in FH.validate_document(doc4))


def test_emit_schema_fallback_has_manual_instructions():
    """取得できないときの手動投入テンプレ(捏造しないための逃げ道)。"""
    doc = FH.empty_schema_document(FH.STATIONS["tokyo"], [1996, 2025], [8])
    assert doc["days"] == []
    assert doc["meta"]["manual_entry_instructions"]
    assert "obsdl" in " ".join(doc["meta"]["manual_entry_instructions"])
    assert "temp_hi" in doc["meta"]["days_schema"]
    assert FH.payload_sha256(doc) == doc["meta"]["payload_sha256"]


def test_daily_url_is_the_documented_endpoint():
    u = FH.daily_url(44, 47662, 2025, 8)
    assert u.startswith("https://www.data.jma.go.jp/stats/etrn/view/daily_s1.php")
    assert "prec_no=44" in u and "block_no=47662" in u and "year=2025" in u and "month=8" in u


# ============================================================ B. fit の数値ユーティリティ
def test_run_lengths():
    assert FG.run_lengths(np.array([0, 1, 1, 0, 1, 1, 1], bool).astype(bool)) == [2, 3]
    assert FG.run_lengths(np.array([1, 1], bool)) == [2]
    assert FG.run_lengths(np.array([0, 0], bool)) == []


def test_ks_two_sample_identical_and_disjoint():
    a = np.arange(100, dtype=float)
    d_same, p_same = FG.ks_two_sample(a, a)
    assert d_same == 0.0 and p_same == 1.0
    d_far, p_far = FG.ks_two_sample(a, a + 1000.0)
    assert d_far == 1.0 and p_far < 1e-6


def test_matalas_ab_solves_the_defining_equations():
    """A = M1 M0⁻¹ と B Bᵀ = M0 − A M1ᵀ を満たすこと(Matalas 1967)。"""
    m0 = np.array([[1.0, 0.73], [0.73, 1.0]])
    m1 = np.array([[0.53, 0.55], [0.54, 0.63]])
    a, b, note = FG.matalas_ab(m0, m1)
    assert np.allclose(a @ m0, m1, atol=1e-10)
    assert np.allclose(b @ b.T, m0 - a @ m1.T, atol=1e-10)
    assert note == "cholesky"


def test_year_effect_variance_separates_between_and_within():
    """群間に既知のシフトを入れると σ_year がそれを拾う(σ_within は元の分散のまま)。"""
    rng = np.random.default_rng(7)
    within = 2.0
    shifts = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    groups = [rng.normal(s, within, 4000) for s in shifts]
    var_y, var_w = FG.year_effect_variance(groups)
    assert abs(np.sqrt(var_y) - shifts.std(ddof=0)) < 0.3
    assert abs(np.sqrt(var_w) - within) < 0.1


def test_gamma_mom_recovers_shape_and_scale():
    rng = np.random.default_rng(11)
    x = rng.gamma(2.0, 5.0, 200000)
    k, theta = FG.gamma_mom(x)
    assert abs(k - 2.0) < 0.1 and abs(theta - 5.0) < 0.2


def test_wbgt_ono2014_matches_hand_computation():
    """環境省掲載式 WBGT = 0.735Ta + 0.0374RH + 0.00292 Ta RH + 7.619SR − 4.557SR² − 0.0572WS − 4.064"""
    ta, rh, sr, ws = 36.8, 46.0, 0.5, 2.7
    expect = (0.735 * ta + 0.0374 * rh + 0.00292 * ta * rh
              + 7.619 * sr - 4.557 * sr ** 2 - 0.0572 * ws - 4.064)
    assert abs(FG.wbgt_ono2014(ta, rh, sr, ws) - expect) < 1e-12
    # 単調性の健全性: 気温・湿度・日射が上がれば WBGT は上がる、風が上がれば下がる
    base = FG.wbgt_ono2014(30.0, 60.0, 0.4, 2.0)
    assert FG.wbgt_ono2014(32.0, 60.0, 0.4, 2.0) > base
    assert FG.wbgt_ono2014(30.0, 70.0, 0.4, 2.0) > base
    assert FG.wbgt_ono2014(30.0, 60.0, 0.5, 2.0) > base
    assert FG.wbgt_ono2014(30.0, 60.0, 0.4, 4.0) < base


def test_sr_proxy_is_bounded():
    assert FG.sr_proxy(0.0) == 0.0
    assert FG.sr_proxy(100.0) == pytest.approx(FG.SR_CLEAR_KW)
    assert 0.0 < FG.sr_proxy(6.0) < FG.SR_CLEAR_KW


def test_label_state_uses_jma_definitions():
    """雨 = 日降水量 1.0mm 以上(気象庁の降水日)。晴/曇 は天気概況の先頭語。"""
    assert FG.label_state({"precip_mm": 1.0, "cond_day": "晴"}, 1.0) == FG.RAIN
    assert FG.label_state({"precip_mm": 0.5, "cond_day": "晴一時曇"}, 1.0) == 0
    assert FG.label_state({"precip_mm": 0.0, "cond_day": "薄曇"}, 1.0) == 1
    assert FG.label_state({"precip_mm": 0.0, "cond_day": "快晴"}, 1.0) == 0
    assert FG.label_state({"precip_mm": 0.0, "snow_cm": 2.0, "cond_day": "雪"}, 1.0) == 3


# ============================================================ B'. 較正の決定論
@needs_data
def test_fit_is_deterministic_bytewise(tmp_path):
    """同じ入力・同じ引数 → バイト同一の JSON(壁時計時刻を書かないので成立する)。"""
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    for p in (a, b):
        rc = FG.main(["--in", str(HISTORY), "--out", str(p), "--validate-years", "40"])
        assert rc == 0
    assert a.read_bytes() == b.read_bytes()


@needs_data
def test_frozen_params_hash_matches_and_points_at_frozen_history():
    params = json.loads(PARAMS.read_text(encoding="utf-8"))
    hist = json.loads(HISTORY.read_text(encoding="utf-8"))
    assert FG.payload_sha256(params) == params["meta"]["payload_sha256"]
    assert FH.payload_sha256(hist) == hist["meta"]["payload_sha256"]
    # 較正パラメータは「どの凍結データから作ったか」を持つ(来歴の連鎖)
    assert params["meta"]["source_payload_sha256"] == hist["meta"]["payload_sha256"]
    assert params["meta"]["source_attribution"] == hist["meta"]["source"]["attribution"]
    assert params["meta"]["caveats"] and params["wbgt"]["citation"]


@needs_data
def test_simulate_is_deterministic():
    pm = json.loads(PARAMS.read_text(encoding="utf-8"))["months"]["8"]
    a = FG.simulate(pm, 30, 31, seed=123)
    b = FG.simulate(pm, 30, 31, seed=123)
    c = FG.simulate(pm, 30, 31, seed=124)
    for key in ("temp_hi", "temp_lo", "state", "precip_mm", "humid_mean"):
        assert np.array_equal(a[key], b[key]), key
    assert not np.array_equal(a["temp_hi"], c["temp_hi"])


# ============================================================ C. 生成器のオフライン統計検証
SEG = 31
N_YEARS = 400
SEED = 20260801


@pytest.fixture(scope="module")
def data():
    params = json.loads(PARAMS.read_text(encoding="utf-8"))
    pm = params["months"]["8"]
    hist = json.loads(HISTORY.read_text(encoding="utf-8"))
    y0, y1 = params["meta"]["fit_window"]["years"]
    days = [d for d in hist["days"] if d["month"] == 8 and y0 <= d["year"] <= y1]
    days.sort(key=lambda d: d["date"])
    obs_hi = np.array([d["temp_hi"] for d in days], float)
    obs_lo = np.array([d["temp_lo"] for d in days], float)
    obs_wet = np.array([(d["precip_mm"] or 0.0) >= 1.0 for d in days])
    sim = FG.simulate(pm, N_YEARS, SEG, SEED)
    cur = FG.simulate_synthetic_current(N_YEARS, SEG, SEED)
    return {"params": params, "pm": pm, "obs_hi": obs_hi, "obs_lo": obs_lo,
            "obs_wet": obs_wet, "sim": sim, "cur": cur}


@needs_data
class TestGeneratorReproducesObservations:
    """較正パラメータから合成した系列が、実測の気温分布・猛暑連長分布に整合すること。"""

    SEG = SEG
    N_YEARS = N_YEARS
    SEED = SEED

    def test_moments_of_daily_tmax(self, data):
        o, g = data["obs_hi"], data["sim"]["temp_hi"]
        assert abs(g.mean() - o.mean()) < 0.5, (g.mean(), o.mean())
        assert abs(g.std(ddof=1) - o.std(ddof=1)) < 0.5
        lo_o, lo_g = data["obs_lo"], data["sim"]["temp_lo"]
        assert abs(lo_g.mean() - lo_o.mean()) < 0.8

    def test_ks_distance_of_tmax_beats_current_synthetic(self, data):
        d_gen, _ = FG.ks_two_sample(data["obs_hi"], data["sim"]["temp_hi"])
        d_cur, _ = FG.ks_two_sample(data["obs_hi"], data["cur"]["temp_hi"])
        assert d_gen < 0.15, d_gen
        assert d_gen < d_cur * 0.6, (d_gen, d_cur)

    def test_hot_day_rate(self, data):
        o, g = data["obs_hi"], data["sim"]["temp_hi"]
        p_o, p_g = float((o >= 35).mean()), float((g >= 35).mean())
        assert abs(p_g - p_o) < 0.06, (p_g, p_o)
        # 現行合成が出せない 36℃以上を生成器は出す(実測にも 7% ある)
        assert float((g >= 36).mean()) > 0.02
        assert float((data["cur"]["temp_hi"] >= 36).mean()) == 0.0

    def test_hot_spell_length_distribution(self, data):
        def lens(hi):
            return [x for s in hi.reshape(-1, self.SEG) for x in FG.run_lengths(s >= 35.0)]
        lo = np.array(lens(data["obs_hi"]), float)
        lg = np.array(lens(data["sim"]["temp_hi"]), float)
        lc = np.array(lens(data["cur"]["temp_hi"]), float)
        d_gen, _ = FG.ks_two_sample(lo, lg)
        d_cur, _ = FG.ks_two_sample(lo, lc)
        assert d_gen < 0.20, d_gen
        assert d_gen < d_cur * 0.6, (d_gen, d_cur)
        assert abs(lg.mean() - lo.mean()) < 0.6, (lg.mean(), lo.mean())

    def test_long_heat_waves_are_reachable(self, data):
        """本件の目的: 10日連続の猛暑(2025年8月に実際に起きた)が生成できること。"""
        segs = data["sim"]["temp_hi"].reshape(-1, self.SEG)
        year_max = np.array([max(FG.run_lengths(s >= 35.0) or [0]) for s in segs], float)
        assert year_max.max() >= 10, year_max.max()
        assert (year_max >= 5).mean() > 0.10
        cur_segs = data["cur"]["temp_hi"].reshape(-1, self.SEG)
        cur_max = np.array([max(FG.run_lengths(s >= 35.0) or [0]) for s in cur_segs], float)
        assert cur_max.max() < 10          # 現行合成では構造的に届かない
        assert (cur_max >= 5).mean() < 0.02

    def test_daily_persistence_is_materially_positive(self, data):
        """日間自己相関: 現行合成はゼロ。生成器は実測(0.52)には届かないが 0.3 以上ある。"""
        def lag1(hi):
            segs = hi.reshape(-1, self.SEG)
            return float(np.mean([np.corrcoef(s[:-1], s[1:])[0, 1]
                                  for s in segs if s.std() > 0]))
        obs_segs = data["obs_hi"].reshape(-1, self.SEG)
        r_obs = float(np.mean([np.corrcoef(s[:-1], s[1:])[0, 1] for s in obs_segs]))
        r_gen = lag1(data["sim"]["temp_hi"])
        r_cur = lag1(data["cur"]["temp_hi"])
        assert r_obs > 0.4
        assert r_gen > 0.30, r_gen
        assert abs(r_cur) < 0.05, r_cur
        # 既知の未達(params に記録済み)であることを固定する
        gaps = data["params"]["validation"]["known_gaps"]
        assert any("lag-1" in g["metric"] for g in gaps)

    def test_interannual_variability_is_preserved(self, data):
        """年効果(低周波成分)。現行合成は年ごとの差がほぼ無い。"""
        def y_sd(hi):
            return float(hi.reshape(-1, self.SEG).mean(axis=1).std(ddof=1))
        sd_o = y_sd(data["obs_hi"])
        sd_g = y_sd(data["sim"]["temp_hi"])
        sd_c = y_sd(data["cur"]["temp_hi"])
        assert abs(sd_g - sd_o) < 0.6, (sd_g, sd_o)
        assert sd_c < 0.6 < sd_o

    def test_rain_frequency_and_markov_persistence(self, data):
        pm = data["pm"]
        p11 = pm["markov2_wet_dry"]["p_wet_given_wet"]
        p01 = pm["markov2_wet_dry"]["p_wet_given_dry"]
        assert p11 > p01, "湿日の後は湿日になりやすい(2状態マルコフの持続)"
        pi = p01 / (1.0 + p01 - p11)
        assert abs(pi - float(data["obs_wet"].mean())) < 0.05
        p_rain_gen = float((data["sim"]["state"] == FG.RAIN).mean())
        assert abs(p_rain_gen - float(data["obs_wet"].mean())) < 0.05

    def test_generated_values_stay_physical(self, data):
        g = data["sim"]
        assert g["temp_lo"].max() <= g["temp_hi"].max()
        assert bool((g["temp_lo"] <= g["temp_hi"]).all())
        clip = data["pm"]["temp"]["clip"]
        assert g["temp_hi"].max() <= clip["hi_max"] + 1e-9
        assert g["temp_hi"].min() >= clip["hi_min"] - 1e-9
        assert bool((g["humid_mean"] >= 10).all()) and bool((g["humid_mean"] <= 100).all())
        assert bool((g["precip_mm"][g["state"] != FG.RAIN] == 0.0).all())
        assert g["share_clipped"] < 0.02       # 箍が常時効いているなら較正が壊れている

    def test_year_effect_is_what_creates_long_spells(self, data):
        """年効果を外すと連続猛暑が減る(= 低周波成分が持続の主因であることの実証)。"""
        no_y = FG.simulate(data["pm"], self.N_YEARS, self.SEG, self.SEED, year_effect=False)
        def p_ge5(hi):
            segs = hi.reshape(-1, self.SEG)
            ym = np.array([max(FG.run_lengths(s >= 35.0) or [0]) for s in segs])
            return float((ym >= 5).mean())
        assert p_ge5(data["sim"]["temp_hi"]) > p_ge5(no_y["temp_hi"]) * 1.3


# ============================================================ D. 現行合成モデルの鏡と構造的欠陥
def test_mirror_matches_src_weather_py():
    """scripts 側の鏡が src/society/weather.py の _sample と同じ値を出すこと。

    weather.py が変わったら **このテストが落ちる**(= 鏡が古くなった合図)。
    """
    weather = pytest.importorskip("society.weather")
    if not hasattr(weather, "_sample") or not hasattr(weather, "_MONTH_CLIMATE"):
        pytest.skip("society.weather の内部 API が変わった(鏡の見直しが要る)")
    assert weather._MONTH_CLIMATE[8] == (32, 25, 0.30, 0.0), "8月の気候テーブルが変わった"
    seed, n = 4242, 200
    mirror = FG.simulate_synthetic_current(1, n, seed)
    for i in range(n):
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, i])))
        got = weather._sample(rng, 8, weather._MONTH_CLIMATE, weather._NEUTRAL_CLIMATE)
        assert got["cond"] == FG.STATES[mirror["state"][i]], i
        assert got["temp_hi"] == mirror["temp_hi"][i], i
        assert got["temp_lo"] == mirror["temp_lo"][i], i


def test_current_synthetic_structural_limits_are_fixed():
    """現行合成の構造的欠陥を数値で固定する(統合後に何が直ったかを言えるようにする)。"""
    cur = FG.simulate_synthetic_current(2000, 31, 99)
    hi = cur["temp_hi"]
    assert hi.max() == 35.0 and hi.min() == 29.0        # 32 ± 3 の一様=上限 35・36以上は不可能
    assert float((hi >= 36).mean()) == 0.0
    assert abs(float((hi == 35).mean()) - 1.0 / 7.0) < 0.01     # 35℃ ちょうど 14.3%
    segs = hi.reshape(-1, 31)
    lag1 = float(np.mean([np.corrcoef(s[:-1], s[1:])[0, 1] for s in segs if s.std() > 0]))
    assert abs(lag1) < 0.05                              # 日間の持続はゼロ(独立同分布)
    assert float(np.mean([len(set(s.tolist())) for s in segs])) > 5.0
