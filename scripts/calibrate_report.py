#!/usr/bin/env python
"""現実整合キャリブレーション・レポート(2026-07-09 ユーザー要望)。

    python scripts/calibrate_report.py runs/<name> [--out runs/<name>/calibration]

シミュレーション出力(L1/L2/agents.json/summary.json)を読むだけで、「現実の一次統計」
とのズレを表にする(sim⇄観測は疎結合。シム本体・schema は一切変更しない)。

現実基準値はすべて近似バンド(lo..hi)で、出典名を注記する(精密値の捏造はしない)。
mock LLM ラン では LLM 依存の行動頻度(発話・投稿・グループ設立など)は行動分布として
意味が薄いので、表では「LLM依存」と明記して機械系(睡眠・通勤・経済・事件率)と分ける。

判定: ✅=バンド内 / 高=上限の何倍か / 低=下限の何分の1か。

第92バッチ(SV-U1 B3・2026-08-05)で **個体間分散の節**を追加した(S-03)。
「LLM は average personality を示し集団の行動的異質性を狭める」(Wu, Peng, Ito & Xiao 2025,
arXiv:2506.19806)への直接の応答で、**平均が ✅ でも分散が現実の何割かを併記**する。
実装の約束:
  - Gini は**新規に書かない**。同ディレクトリの build_panel._gini(凍結対象外)を import。
  - CV / 分位も scripts/model_battery/metrics.py の既存実装(cv / quantile)を import。
  - 現実側は data/battery/reference/ の provenance JSON からのみ読む(捏造しない)。
  - 本ファイルは metrics_spec_hash の SPEC_FILES 14 本に**含まれない**ので凍結は破れない。
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import statistics as st
import sys
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _HERE)                     # 同ディレクトリの build_panel を import

# Windows コンソール(cp932)でも ✅ 等を印字できるように(ファイル出力は常に UTF-8)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import build_panel as bp                       # noqa: E402  (活動復元の再利用=二重実装回避)
from model_battery import metrics as bm        # noqa: E402  (cv / quantile の既存実装)
from model_battery import reference as bref    # noqa: E402  (現実バンドの来歴強制ローダ)
from society.observer.measure import stream_events  # noqa: E402

MIN_PER_DAY = 1440
MIN_PER_STEP = 10
STEPS_PER_DAY = 144

# 現実側の分散バンド(来歴つき)。読み込み失敗は「バンド無し」へ後退する(レポートは出る)。
REF_DIR = os.path.join(_ROOT, "data", "battery", "reference")
REF_CV_LB = "estat_activity_rate_cv_lower_bound"      # 行動時間の CV 下界(片側)
REF_CONTACT = "sociopatterns_contact_heterogeneity"   # 接触・発話(現状すべて未取得)

# --------------------------------------------------------------------------- #
# 現実基準(近似バンド)。value の単位は metric ごとに合わせる。
#   (key, ラベル, lo, hi, 単位, 出典・注記)
# --------------------------------------------------------------------------- #
REALITY = [
    ("sleep_h",       "睡眠時間(住民)",          6.9,   8.1,  "h/日",
     "NHK国民生活時間調査2020 平日7.2h / 社会生活基本調査R3 7.9h"),
    ("bedtime_h",     "就寝時刻の中央値",          22.5,  24.8, "時",
     "NHK2020: 平日23時台に就寝が最多"),
    ("wake_h",        "起床時刻の中央値",          5.5,   7.5,  "時",
     "NHK2020: 平日6時台前半"),
    ("work_h",        "労働時間(被雇用・出勤日)",  7.0,   9.8,  "h/出勤日",
     "毎月勤労統計 一般労働者 所定7.8h+残業"),
    ("taxi_pp",       "タクシー乗車",              0.01,  0.12, "回/人日",
     "東京PT調査: タクシー分担率~2%×2.6トリップ/日(地図は徒歩圏2km。鉄道は域外行き=対象外)"),
    ("monthly_net",   "手取り月収(受給者平均)",    130000, 350000, "円/月",
     "毎月勤労統計+非正規込みの手取り近似 15-30万"),
    ("crime_pp",      "窃盗被害",                1e-5,  3e-4, "件/人日",
     "渋谷区刑法犯~5千/年÷昼間人口54万≈2.8e-5(観測性のため10倍まで許容)"),
    ("nuisance_pp",   "迷惑行為への遭遇",          0.03,  0.5,  "件/人日",
     "渋谷中心部の客引き・路上トラブル遭遇の体感近似(公式統計なし)"),
    ("illness_pp",    "急性疾患の発症",            0.004, 0.017, "回/人日",
     "感冒2-6回/年+その他急性疾患"),
    ("layoff_pp",     "失業(解雇・雇止め)",       2e-5,  2e-4, "回/被雇用者日",
     "雇用動向調査 非自発的離職~1-2%/年(5倍圧縮まで許容)"),
    ("switch_pp",     "転職",                    1e-4,  5e-4, "回/被雇用者日",
     "転職率~5-10%/年(2倍圧縮まで許容)"),
    ("unemp_rate",    "失業率(ラン末)",          0.015, 0.045, "",
     "労働力調査 2.5-2.8%"),
    ("rent_ratio",    "家賃/手取り(月次)",        0.20,  0.40, "",
     "目安25-30%(設計値0.3)"),
    ("engel",         "食費/消費支出",            0.18,  0.35, "",
     "家計調査 単身エンゲル係数~26%"),
    ("rainy_share",   "降雨日の割合(7-8月)",      0.25,  0.55, "",
     "東京7月の降水日(1mm以上)~11-15日/月"),
    ("disaster_100d", "災害ショック",              0.3,   3.0,  "件/100日",
     "台風接近~3回/年+有感地震・大雪"),
    ("delay_share",   "交通遅延・運休の日割合",     0.03,  0.4,  "",
     "主要線区の遅延証明発行は月10日超も(幅広)"),
    ("evict_100d",    "立退き",                  0.0,   2.0,  "件/100日100人",
     "明渡訴訟~2万/年÷5500万世帯≈0(制度観測のため2件まで許容)"),
    ("bankrupt_100d", "自己破産",                0.0,   1.5,  "件/100日100人",
     "自己破産~7万/年÷成人1億≈0(同上)"),
    ("savings_rate",  "貯蓄率(Δ資産/手取り・月次)", -0.10, 0.45, "",
     "家計調査 勤労単身の黒字率~30-40%(全世帯はもっと低い)"),
    ("grievance_max", "不満の飽和ガード(最大値)",  0.0,   0.90, "",
     "設計ガード: 慢性飽和(≈1.0張り付き)は非現実"),
    ("status_gini",   "地位のジニ係数(ラン末)",   0.25,  0.70, "",
     "所得ジニ0.38(再分配後)〜金融資産ジニ0.6台の間"),
    # ---- 共同行動・承諾(第62バッチ。joint / endogenous_accept ON のランのみ値が出る。0件は ―)----
    ("joint_pp",      "共同外出への参加(住民)",    0.03,  0.45, "回/人日",
     "直接統計なし: レジャー白書2024 外食39.2%/映画33.7%/カラオケ20.2%(年間参加率)+友人と会うのは"
     "月1最頻(§2.3)→月1〜週3相当のプロキシ帯(daily_rate=0.15≈週1の設計較正を挟む)"),
    ("joint_accept",  "誘いの承諾率 [直接統計なし]", 0.30,  0.75, "",
     "承諾率の公的統計は不在(捏造しない)。プロキシ: 職場飲み会は誘われても「断る」が過半"
     "(ツナグ働き方研究所2020・日経報道=受諾<5割)/同世代の任意の集まりは「好き」50.8%"
     "(SHIBUYA109 lab.=受諾>5割)→ accept_base=0.5 を中心に ±0.2〜0.25 の帯"),
    # ---- 以下は LLM 依存(mock ランでは参考値。実LLMランで再判定する)----
    ("sns_post_pp",   "SNS投稿 [LLM依存]",        0.05,  1.5,  "件/人日",
     "X利用率~46%×投稿者は少数(1:9論)"),
    ("speak_pp",      "対面の発話ターン [LLM依存]",  3.0,   40.0, "回/人日",
     "日常会話の交換回数の粗い近似"),
    ("group_100d",    "グループ設立 [LLM依存]",     0.0,   10.0, "件/100日100人",
     "サークル・団体の新設は稀(観測性許容込み)"),
    ("event_pp",      "イベント主催 [LLM依存]",     0.0,   0.05, "件/人日",
     "主催は月1もあれば多い"),
]
_R = {k: (lo, hi, unit, src) for k, _, lo, hi, unit, src in REALITY}
_LABEL = {k: lab for k, lab, *_ in REALITY}


# --------------------------------------------------------------------------- #
# 小道具
# --------------------------------------------------------------------------- #
def hod(sim_min: int) -> float:
    """時刻(0-24h)。sim_min は絶対分(step0=420=07:00)。"""
    return (sim_min % MIN_PER_DAY) / 60.0


def day_of(sim_min: int) -> int:
    return sim_min // MIN_PER_DAY


def circ_median_h(hours: list[float], split: float = 12.0) -> float | None:
    """就寝時刻など日跨ぎの中央値: split より前は +24h して並べる(24.3=翌0:20 のまま返す)。"""
    if not hours:
        return None
    shifted = [h + 24.0 if h < split else h for h in hours]
    return st.median(shifted)


def fmt(v) -> str:
    if v is None:
        return "―"
    if isinstance(v, float):
        if v != 0 and abs(v) < 0.01:
            return f"{v:.2e}"
        return f"{v:,.3g}" if abs(v) >= 1000 else f"{v:.3g}"
    return str(v)


def verdict(key: str, v) -> str:
    if v is None or key not in _R:
        return "―"
    lo, hi, _, _ = _R[key]
    if lo <= v <= hi:
        return "✅"
    if v > hi:
        return f"高 ×{v / hi:.1f}" if hi > 0 else f"高 (+{v:.2g})"
    return f"低 ×{v / lo:.2f}" if lo > 0 else f"低 ({v:.2g})"


def read_l2(run_dir: str) -> dict[str, list]:
    import pyarrow.parquet as pq
    path = os.path.join(run_dir, "l2_metrics.parquet")
    if not os.path.exists(path):
        return {}
    return pq.read_table(path).to_pydict()


# --------------------------------------------------------------------------- #
# ④行動統計(第31バッチ W3): 分布一致指標(1次元・純Python自作。scipy 非依存)
#   KS = 経験CDFの最大差 / EMD(Wasserstein-1)= |CDF差| の積分。標本サイズが違っても可。
# --------------------------------------------------------------------------- #
def ks_stat(a: list, b: list):
    """2標本 Kolmogorov-Smirnov 統計量 D = max_x |F_a(x) − F_b(x)|。空群は None。"""
    a = sorted(float(x) for x in a if x is not None)
    b = sorted(float(x) for x in b if x is not None)
    if not a or not b:
        return None
    na, nb = len(a), len(b)
    d = 0.0
    for x in sorted(set(a) | set(b)):
        fa = bisect.bisect_right(a, x) / na
        fb = bisect.bisect_right(b, x) / nb
        d = max(d, abs(fa - fb))
    return d


def emd_1d(a: list, b: list):
    """1次元 Earth Mover's Distance(Wasserstein-1)= ∫|F_a(x) − F_b(x)|dx。
    値の単位は入力と同じ(分など)。空群は None。決定論(乱数なし)。"""
    a = sorted(float(x) for x in a if x is not None)
    b = sorted(float(x) for x in b if x is not None)
    if not a or not b:
        return None
    na, nb = len(a), len(b)
    pts = sorted(set(a) | set(b))
    tot = 0.0
    for i in range(len(pts) - 1):
        x, w = pts[i], pts[i + 1] - pts[i]
        fa = bisect.bisect_right(a, x) / na
        fb = bisect.bisect_right(b, x) / nb
        tot += abs(fa - fb) * w
    return tot


# 生活時間配分の参考バンド(既存 REALITY の睡眠・労働バンドを分/日へ換算・出典継承)。
# 現実の「カテゴリ別 分/日 の完全分布」はリポジトリ内に無いため、参照は睡眠・労働のみ。
_ACT_LABEL_JP = {"sleep": "睡眠", "home": "在宅", "work": "仕事", "food": "飲食",
                 "shop": "買物", "leisure": "余暇(視聴)", "move": "移動",
                 "other": "その他(判定不能)"}
_ACT_REF_BAND = {   # category -> (lo_min, hi_min, 出典・注記)
    "sleep": (6.9 * 60, 8.1 * 60,
              "NHK国民生活時間調査2020/社会生活基本調査R3(全活動日平均・就床時間)"),
    "work": (7.0 * 60, 9.8 * 60,
             "毎月勤労統計 所定+残業(※現実は出勤日あたり・本表は全活動日平均で分母が異なる)"),
}


def activity_summary(recon: dict) -> dict:
    """reconstruct_activity の出力から、カテゴリ別 分/日(全体/平日/休日)と
    平日vs休日の分布一致(KS/EMD)を計算する。"""
    adc = recon["adc"]
    hol = recon["holiday_by_day"]
    cats = recon["cats"]
    by_ad: dict[tuple, dict] = defaultdict(lambda: defaultdict(float))
    for (aid, d, c), n in adc.items():
        by_ad[(aid, d)][c] += n * MIN_PER_STEP
    all_ad = sorted(by_ad)
    wk = [ad for ad in all_ad if hol.get(ad[1]) is False]
    hd = [ad for ad in all_ad if hol.get(ad[1]) is True]

    def means(subset):
        k = len(subset)
        if k == 0:
            return {c: None for c in cats}, 0
        return ({c: sum(by_ad[ad].get(c, 0.0) for ad in subset) / k
                 for c in cats}, k)

    overall, n_all = means(all_ad)
    weekday, n_wk = means(wk)
    holiday, n_hd = means(hd)
    dist: dict[str, dict] = {}
    if wk and hd:
        for c in ("sleep", "work"):
            av = [by_ad[ad].get(c, 0.0) for ad in wk]
            bv = [by_ad[ad].get(c, 0.0) for ad in hd]
            dist[c] = {"ks": ks_stat(av, bv), "emd": emd_1d(av, bv)}
    return {"cats": cats, "overall": overall, "weekday": weekday,
            "holiday": holiday, "n_all": n_all, "n_wk": n_wk, "n_hd": n_hd,
            "has_split": bool(wk and hd), "dist": dist}


# --------------------------------------------------------------------------- #
# ⑤個体間分散(第92バッチ SV-U1 B3 = S-03)
#
# 何を測るか: 「平均は現実バンドに入るが、個体間のばらつきが現実より痩せている」を検出する。
#   Wu, Peng, Ito & Xiao (2025) arXiv:2506.19806 の提言②
#   =「平均の一致と並べて**分散を明示的に報告せよ**」への直接の応答。
#
# 3 指標を併記する理由(単独では捉えられない変化があるため):
#   CV       … σ/μ。スケール不変で散らばりをスカラー1個に落とす。
#   Gini     … 集中度。CV と同一関数の L¹/L² ノルムの関係にあり強く相関するが同一ではない。
#   上位10%  … Gini は「上位0.1%だけが伸びた」場合を見落とす。裾(= k* の主張の直撃点)を掴む。
#
# **観測単位は agent×day**(person-day)。現実側の行動者率が person-day 単位なので、
# agent 単位に日平均してから CV を採ると分散が縮んで比較にならない。
# --------------------------------------------------------------------------- #
def _top_share(values, pct: float = 0.10):
    """上位 pct 割合が占める合計の割合(winner-take-all)。総和<=0/空は None。

    ★これは Gini ではない(Gini は build_panel._gini を import して使う=6個目を書かない)。
    観測数 n に対し上位 ceil(n*pct) 件(最低1件)を採る。
    """
    xs = sorted((float(v) for v in values if v is not None), reverse=True)
    if not xs:
        return None
    total = sum(xs)
    if total <= 0:
        return None
    k = max(1, int(math.ceil(len(xs) * pct)))
    return round(sum(xs[:k]) / total, 4)


def _dispersion(values) -> dict:
    """1 本の分布 → 個体間分散の要約。n<2 は全て None(データ不足を捏造しない)。"""
    xs = [float(v) for v in values if v is not None]
    out = {"n": len(xs), "mean": None, "cv": None, "gini": None,
           "top10": None, "p50": None, "p90": None, "zero_share": None}
    if len(xs) < 2:
        return out
    out["mean"] = sum(xs) / len(xs)
    out["cv"] = round(bm.cv(xs), 4)                   # 既存実装(母集団 sd / mean)
    out["gini"] = bp._gini(xs)                        # 既存実装(scripts 側・凍結対象外)
    out["top10"] = _top_share(xs, 0.10)
    out["p50"] = round(bm.quantile(xs, 0.5), 3)
    out["p90"] = round(bm.quantile(xs, 0.9), 3)
    out["zero_share"] = round(sum(1 for x in xs if x <= 0) / len(xs), 4)
    return out


# シムの活動カテゴリ(build_panel.reconstruct_activity)→ 社会生活基本調査 20 分類の対応。
# **対応が曖昧なものは載せない**(load-bearing な捏造になるため):
#   home  … 「在宅」は場所であって行動分類ではない(睡眠・家事・くつろぎ等に跨る)。
#   other … 「判定不能」であって調査の「その他」ではない。
# move だけは特別扱い: シムは通勤とそれ以外の移動を分離しないので、和集合の行動者率
#   p_union <= 0.397 + 0.246 = 0.643 という**上からの**抑えを使う。CV 下界は p について
#   単調減少なので、p の上界からは有効な下界が出る(p の下界を使うと下界にならない)。
_VAR_ACT_TO_ESTAT = {
    "sleep": "睡眠",
    "work": "仕事",
    "food": "食事",
    "shop": "買い物",
    "leisure": "テレビ・ラジオ・新聞・雑誌",
    "move": "__move_union__",
}


def load_reference_bands() -> dict:
    """現実側の分散バンドを data/battery/reference/ から読む(来歴強制)。

    掟(model_battery/reference.py): 出典・ライセンス・取得日が揃わない参照は読み込まない。
    **未取得は values=null** で入っているので、ここでは「無い」がそのまま伝播する。
    """
    out = {"cv_lower_bound": {}, "docs": [], "errors": []}
    for ref_id in (REF_CV_LB, REF_CONTACT):
        path = os.path.join(REF_DIR, f"{ref_id}.json")
        try:
            doc = bref.load(path)
        except bref.ReferenceError as exc:       # 来歴違反・欠損は「バンド無し」へ後退
            out["errors"].append(f"{ref_id}: {exc}")
            continue
        out["docs"].append(doc)
        if ref_id != REF_CV_LB:
            continue
        series = doc.get("series") or {}
        cats = (series.get("activity_participation_rate_week") or {}).get("categories") or []
        lbs = (series.get("cv_lower_bound_by_activity") or {}).get("values") or []
        for cat, lb in zip(cats, lbs):
            out["cv_lower_bound"][cat] = lb
        mv = (series.get("cv_lower_bound_move_union") or {}).get("values") or {}
        if mv.get("cv_lower_bound") is not None:
            out["cv_lower_bound"]["__move_union__"] = mv["cv_lower_bound"]
    return out


def variance_summary(recon: dict | None, speak_by_ad: dict, contact_by_ad: dict,
                     agent_days: set) -> dict:
    """agent×day 単位の 3 系統(行動時間・発話数・接触人数)の個体間分散。

    行動時間は build_panel.reconstruct_activity の adc(=(aid, day, cat) → step 数)を
    そのまま畳んで分/日にする(二重実装しない)。発話・接触は L1 の 1 パスで拾った
    (aid, day) 台帳から作り、**その日 L1 に現れた agent は 0 も 1 件として数える**
    (0 を落とすと「話す人だけ」の分布になり分散が過小評価される)。
    """
    rows: list[dict] = []

    # ---- (1) 行動時間(活動別・分/日)----
    if recon:
        adc = recon["adc"]
        by_ad: dict[tuple, dict] = defaultdict(lambda: defaultdict(float))
        for (aid, d, c), n in adc.items():
            by_ad[(aid, d)][c] += n * MIN_PER_STEP
        # ★ L1 に実際にイベントが記録された agent×day に限る。活動復元は睡眠区間を
        #   最終 step の先まで延ばすことがあり(sleep_start の until_step)、その端数日を
        #   1 日として数えると分母が狂う。ここで揃えると発話・接触の行と n も一致する。
        keys = [k for k in sorted(by_ad) if k in agent_days]
        for cat in recon["cats"]:
            vals = [by_ad[k].get(cat, 0.0) for k in keys]
            rows.append({"group": "行動時間", "key": cat,
                         "label": f"{_ACT_LABEL_JP.get(cat, cat)}(分/日)",
                         "estat": _VAR_ACT_TO_ESTAT.get(cat),
                         **_dispersion(vals)})

    # ---- (2) 発話数/日・(3) 接触人数/日 ----
    ads = sorted(agent_days)
    if ads:
        rows.append({"group": "発話", "key": "speak_per_day", "label": "発話数(回/人日)",
                     "estat": None,
                     **_dispersion([speak_by_ad.get(k, 0) for k in ads])})
        rows.append({"group": "接触", "key": "contacts_per_day",
                     "label": "接触人数(相異なる対面相手/人日)", "estat": None,
                     **_dispersion([len(contact_by_ad.get(k, ())) for k in ads])})

    bands = load_reference_bands()
    for r in rows:
        lb = bands["cv_lower_bound"].get(r.get("estat")) if r.get("estat") else None
        r["cv_real_lb"] = lb
        r["cv_ratio"] = (round(r["cv"] / lb, 3)
                         if (lb and r.get("cv") is not None and lb > 0) else None)
    return {"rows": rows, "n_agent_days": len(ads), "bands": bands}


# --------------------------------------------------------------------------- #
# 本体
# --------------------------------------------------------------------------- #
def analyze(run_dir: str) -> dict:
    agents = json.load(open(os.path.join(run_dir, "agents.json"), encoding="utf-8"))
    summary = json.load(open(os.path.join(run_dir, "summary.json"), encoding="utf-8"))
    kinds: dict[str, int] = summary.get("event_kinds", {})
    n_steps = int(summary.get("n_steps", 0))
    n_days = max(1.0, n_steps / 144.0)

    residents = [a for a in agents if not a.get("visitor")]
    # 組織雇用(解雇・転職の母数): 在勤先 or パートを持つ住民
    org_employed = [a for a in residents
                    if a.get("work_building") or a.get("part_time")]
    res_ids = {a["id"] for a in residents}
    n_all, n_res, n_emp = len(agents), len(residents), len(org_employed)

    # ---- L1 走査(1パス・P4 row-group 逐次で有界メモリ)----
    sleep_h: list[float] = []
    bed_hours: list[float] = []
    wake_hours: list[float] = []
    ride_mode = Counter()
    ride_hist = [0] * 24
    spend_by_cat = Counter()
    wage_by: dict[int, float] = defaultdict(float)
    rent_by: dict[int, float] = defaultdict(float)
    illness_onset = 0
    lost = 0
    rehired = 0
    switched = 0
    rainy_days: set[int] = set()
    weather_days: set[int] = set()
    delay_days: set[int] = set()
    disaster_onsets = 0
    reflect_n = 0
    reflect_deep = 0
    crime_thefts = 0
    joint_participants = 0
    joint_invites = 0
    joint_accepts = 0
    # ---- S-03 個体間分散(第92バッチ): agent×day の台帳 ----
    #   日の切り方は build_panel と同じ step//144(07:00 起点で夜間睡眠が同一日に収まる)。
    #   ★メモリ: agent×day のキー数に比例する(接触は集合を持つ)。25 万体×10 日級では
    #     bp.reconstruct_activity の adc((aid, day, cat))が既に同規模なので支配項ではない。
    agent_days: set[tuple] = set()
    speak_by_ad: Counter = Counter()
    contact_by_ad: dict[tuple, set] = defaultdict(set)

    for e in stream_events(run_dir):
        k = e["kind"]
        p = e["payload"]
        aid = e["agent_id"]
        sm = e["sim_min"]
        d_ad = e["step"] // STEPS_PER_DAY
        if aid is not None and aid >= 0:
            agent_days.add((aid, d_ad))
        if k == "speak":
            # 対面のみ(S-12: offline/online を混ぜない)。DM/SNS は接触人数に数えない。
            speak_by_ad[(aid, d_ad)] += 1
            for h in (p.get("hearers") or []):
                h = int(h)
                if h < 0 or h == aid:
                    continue
                contact_by_ad[(aid, d_ad)].add(h)
                contact_by_ad[(h, d_ad)].add(aid)
        if k == "wake_up":
            if aid in res_ids:
                ss = p.get("slept_steps")
                if ss:
                    sleep_h.append(float(ss) * 10.0 / 60.0)
                wake_hours.append(hod(sm))
        elif k == "sleep_start":
            if aid in res_ids:
                bed_hours.append(hod(sm))
        elif k == "ride":
            ride_mode[str(p.get("mode", "?"))] += 1
            ride_hist[int(hod(sm)) % 24] += 1
        elif k == "spend":
            spend_by_cat[str(p.get("cat", "?"))] += float(p.get("amount") or 0)
        elif k == "wage":
            wage_by[aid] += float(p.get("amount") or 0)
        elif k == "rent":
            rent_by[aid] += float(p.get("amount") or 0)
        elif k == "illness":
            if p.get("state") == "onset":
                illness_onset += 1
        elif k == "unemployment":
            if p.get("state") == "lost":
                lost += 1
        elif k == "job_change":
            cause = str(p.get("cause", ""))
            if cause == "switch":
                switched += 1
            elif cause in ("rehire", "hired"):
                rehired += 1
        elif k == "weather":
            d = day_of(sm)
            weather_days.add(d)
            if "雨" in str(p.get("cond", "")):
                rainy_days.add(d)
        elif k == "transit_delay":
            delay_days.add(day_of(sm))
        elif k == "disaster":
            if p.get("phase") in ("onset", None):
                disaster_onsets += 1
        elif k == "reflect":
            reflect_n += 1
            if p.get("deep"):
                reflect_deep += 1
        elif k == "crime":
            if p.get("kind") == "theft":
                crime_thefts += 1
        elif k == "joint_activity":
            joint_participants += len(p.get("with") or [])
        elif k == "joint_invite":
            joint_invites += 1
            if p.get("accepted"):
                joint_accepts += 1

    # ---- L2 曲線 ----
    l2 = read_l2(run_dir)
    steps = l2.get("step", [])
    m: dict = {}

    def _daily_series(col: str) -> list[float]:
        """列を日ごとに合計(step列基準・1日=144step)。"""
        vals = l2.get(col)
        if not vals:
            return []
        acc: dict[int, float] = defaultdict(float)
        for s, v in zip(steps, vals):
            acc[s // 144] += float(v or 0)
        return [acc[d] for d in sorted(acc)]

    # 労働時間: 日ごとの Σ(n_working)×10分 ÷ その日の最大同時就業者数(実働者の近似)。
    # 名簿上の被雇用者数だと来街者の店員などを取りこぼすため、実測ピークを分母にする。
    work_daily = _daily_series("n_working")
    peak_daily: dict[int, float] = defaultdict(float)
    for s, v in zip(steps, l2.get("n_working", [])):
        d = s // 144
        peak_daily[d] = max(peak_daily[d], float(v or 0))
    peaks = [peak_daily[d] for d in sorted(peak_daily)]
    work_h_days = [w * 10.0 / 60.0 / p for w, p in zip(work_daily, peaks) if p >= 5]
    workdays = [w for w in work_h_days if w >= 3.0]
    m["work_h"] = st.mean(workdays) if workdays else None
    m["workday_share"] = len(work_h_days) / len(work_daily) if work_daily else None
    m["peak_workforce"] = max(peaks) if peaks else None

    # 貯蓄率: Δ(現金+口座の平均)を受給者平均の手取り(同期間)で割る(家計調査の黒字率に対応)
    mm = l2.get("mean_money", [])
    ma = l2.get("mean_account", [])
    m["savings_rate"] = None
    if mm and ma:
        tot0, tot1 = mm[0] + ma[0], mm[-1] + ma[-1]
        m["money_start"], m["money_end"] = tot0, tot1
        m["_wealth_delta"] = tot1 - tot0
    mg = l2.get("mean_grievance", [])
    m["grievance_max"] = max(mg) if mg else None
    m["grievance_end"] = mg[-1] if mg else None
    sg = [v for v in l2.get("status_gini", []) if v is not None]
    m["status_gini"] = sg[-1] if sg else None
    st10 = [v for v in l2.get("status_top10_share", []) if v is not None]
    m["status_top10"] = st10[-1] if st10 else None
    nb = l2.get("n_broke", [])
    m["broke_end"] = nb[-1] if nb else None

    # ---- 時間使途 ----
    m["sleep_h"] = st.median(sleep_h) if sleep_h else None
    m["sleep_h_mean"] = st.mean(sleep_h) if sleep_h else None
    m["bedtime_h"] = circ_median_h(bed_hours)
    m["wake_h"] = st.median(wake_hours) if wake_hours else None

    # ---- 交通機関(地図は徒歩圏: 域内はタクシー/バスのみ。鉄道=来街・ビューア用)----
    m["taxi_pp"] = ride_mode.get("taxi", 0) / (n_all * n_days) if n_all else None
    m["ride_hist"] = ride_hist
    m["ride_mode"] = dict(ride_mode)

    # ---- 経済 ----
    spend_total = sum(spend_by_cat.values())
    food = sum(v for c, v in spend_by_cat.items()
               if c in ("food", "meal", "食費", "cafe", "restaurant", "grocery"))
    consum = spend_total - spend_by_cat.get("rent", 0.0)
    m["engel"] = (food / consum) if consum > 0 and food > 0 else None
    m["spend_by_cat"] = {c: round(v) for c, v in spend_by_cat.most_common()}
    # 家賃/手取り: 家賃を払った本人の(家賃合計/賃金合計)の平均(受給者全体で希釈しない)
    ratios = [rent_by[a] / wage_by[a] for a in rent_by if wage_by.get(a, 0) > 0]
    m["rent_ratio"] = st.mean(ratios) if ratios else None
    m["n_renters"] = len(rent_by)
    # 手取り月収: 賃金イベントの受給者平均を 30 日換算
    n_earners = len(wage_by)
    m["n_earners"] = n_earners
    m["monthly_net"] = (sum(wage_by.values()) / n_earners * (30.0 / n_days)
                        if n_earners else None)
    m["spend_pp_day"] = spend_total / (n_all * n_days) if n_all else None
    # 貯蓄率の分子は全 agent 平均のΔ資産 → 分母も全 agent 平均の手取り(期間全体)
    if m.get("_wealth_delta") is not None and n_all:
        income_pp = sum(wage_by.values()) / n_all
        m["savings_rate"] = (m["_wealth_delta"] / income_pp) if income_pp > 0 else None

    # ---- 事件・制度(人日あたり)----
    m["crime_pp"] = crime_thefts / (n_all * n_days)
    m["nuisance_pp"] = kinds.get("nuisance", 0) / (n_all * n_days)
    m["illness_pp"] = illness_onset / (n_all * n_days)
    m["layoff_pp"] = lost / (n_emp * n_days) if n_emp else None
    m["switch_pp"] = switched / (n_emp * n_days) if n_emp else None
    m["unemp_now"] = max(0, lost - rehired)      # 復職=rehire のみ戻す
    m["unemp_rate"] = m["unemp_now"] / n_emp if n_emp else None
    m["rainy_share"] = len(rainy_days) / len(weather_days) if weather_days else None
    m["disaster_100d"] = disaster_onsets * (100.0 / n_days)
    m["delay_share"] = len(delay_days) / n_days
    m["evict_100d"] = kinds.get("eviction", 0) * (100.0 / n_days) * (100.0 / n_all)
    m["bankrupt_100d"] = kinds.get("bankruptcy", 0) * (100.0 / n_days) * (100.0 / n_all)

    # ---- LLM 依存(mock では参考値)----
    m["sns_post_pp"] = kinds.get("sns_post", 0) / (n_all * n_days)
    m["speak_pp"] = kinds.get("speak", 0) / (n_all * n_days)
    m["group_100d"] = kinds.get("group_found", 0) * (100.0 / n_days) * (100.0 / n_all)
    m["event_pp"] = kinds.get("event_host", 0) / (n_all * n_days)
    m["reflect_pp"] = reflect_n / (n_all * n_days)
    m["reflect_deep_share"] = reflect_deep / reflect_n if reflect_n else None
    m["relation_break_pp"] = kinds.get("relation_break", 0) / (n_all * n_days)

    # ---- 共同行動・承諾(第62バッチ)。0件は None=「―」(joint/endo OFF と発生ゼロを混同しない)----
    m["joint_pp"] = (joint_participants / (n_res * n_days)
                     if joint_participants and n_res else None)
    m["joint_accept"] = (joint_accepts / joint_invites) if joint_invites else None
    m["joint_invites"] = joint_invites or None

    # ---- ④生活時間配分(第31バッチ W3): build_panel の活動復元を再利用 ----
    recon = None
    try:
        recon = bp.reconstruct_activity(stream_events(run_dir), agents)
        m["_activity"] = activity_summary(recon)
    except Exception:
        m["_activity"] = None

    # ---- ⑤個体間分散(第92バッチ SV-U1 B3 = S-03)----
    m["_variance"] = variance_summary(recon, speak_by_ad, contact_by_ad, agent_days)

    # 既存 L2 の集中度列は**読むだけ**(観測系 aggregate.py は凍結対象なので触らない)。
    # 無い列(そのランで lens が OFF)は None のまま=「―」で出す。
    m["_l2_concentration"] = {}
    for col in ("status_gini", "status_top10_share", "trust_gini",
                "deviation_top_share", "asset_gini", "asset_top10_share"):
        vals = [v for v in l2.get(col, []) if v is not None]
        m["_l2_concentration"][col] = vals[-1] if vals else None

    m["_meta"] = {
        "run_dir": run_dir, "n_agents": n_all, "n_residents": n_res,
        "n_employed": n_emp, "n_days": n_days, "n_steps": n_steps,
        "n_sleep_samples": len(sleep_h),
    }
    return m


# --------------------------------------------------------------------------- #
# レポート出力
# --------------------------------------------------------------------------- #
def render(m: dict) -> str:
    meta = m["_meta"]
    lines = [
        f"# 現実整合レポート: {os.path.basename(meta['run_dir'])}",
        "",
        f"- agents: {meta['n_agents']}(住民 {meta['n_residents']} / 被雇用 "
        f"{meta['n_employed']})、{meta['n_days']:.1f} 日({meta['n_steps']} step)",
        f"- 睡眠サンプル: {meta['n_sleep_samples']} 夜",
        "",
        "| 指標 | sim | 現実バンド | 判定 | 出典(近似) |",
        "|---|---|---|---|---|",
    ]
    for key, label, lo, hi, unit, src in REALITY:
        v = m.get(key)
        band = f"{fmt(lo)}〜{fmt(hi)} {unit}".strip()
        lines.append(f"| {label} | {fmt(v)} | {band} | {verdict(key, v)} | {src} |")
    lines += [
        "",
        "## 補助(バンドなし・参考)",
        f"- 出勤日率: {fmt(m.get('workday_share'))}(暦の平日率≈5/7=0.71 と比較)"
        f" / 最大同時就業 {fmt(m.get('peak_workforce'))} 人",
        f"- 賃金受給者: {fmt(m.get('n_earners'))} 人 / 家賃支払者: {fmt(m.get('n_renters'))} 人",
        f"- 消費/人日: {fmt(m.get('spend_pp_day'))} 円(家計調査 単身~6千円/日)",
        f"- 支出内訳: {json.dumps(m.get('spend_by_cat', {}), ensure_ascii=False)}",
        f"- 乗車モード内訳: {json.dumps(m.get('ride_mode', {}), ensure_ascii=False)}",
        f"- 乗車の時間帯分布(0-23時): {m.get('ride_hist')}",
        f"- 不満 mean_grievance: 最大 {fmt(m.get('grievance_max'))} / "
        f"末値 {fmt(m.get('grievance_end'))}(飽和ガード<0.9)",
        f"- 資産: 初値 {fmt(m.get('money_start'))} → 末値 {fmt(m.get('money_end'))} 円/人",
        f"- 無一文(ラン末): {fmt(m.get('broke_end'))} 人",
        f"- 内省: {fmt(m.get('reflect_pp'))} 回/人日(深い内省の比率 "
        f"{fmt(m.get('reflect_deep_share'))})",
        f"- 関係の降格(接触断): {fmt(m.get('relation_break_pp'))} 回/人日",
        f"- 地位 top10 シェア: {fmt(m.get('status_top10'))}",
        "",
        "> 注: [LLM依存] の行動頻度は mock ランでは行動分布として意味が薄い(実LLMで再判定)。",
        "> 現実バンドは公表統計の近似(出典名を明記)。バンド外=即誤りではなく調律候補。",
    ]
    lines += _render_activity(m.get("_activity"))
    lines += _render_variance(m.get("_variance"), m.get("_l2_concentration"))
    return "\n".join(lines)


def _render_variance(var: dict | None, l2conc: dict | None) -> list[str]:
    """個体間分散の節(S-03)。**平均だけで「現実的」と言わないための表**。"""
    L = ["", "## 個体間分散(S-03: 平均は合っても分散が痩せていないか)"]
    if not var or not var.get("rows"):
        L.append("(agent×day の観測が取れない=データ不足)")
        return L
    L.append(
        "観測単位は **agent×day**(person-day)。現実側の行動者率が person-day 単位なので、"
        "agent 単位に日平均してから CV を採ると分散が縮んで比較にならない。"
        f"対象 agent×day = {var['n_agent_days']}。")
    L.append("")
    L.append("| 指標 | n | sim 平均 | sim CV | 現実 CV(下界) | **分散比** | sim Gini | "
             "sim top10% | 中央値 | p90 | ゼロの割合 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in var["rows"]:
        L.append(
            f"| {r['label']} | {r['n']} | {fmt(r['mean'])} | {fmt(r['cv'])} | "
            f"{fmt(r['cv_real_lb'])} | {fmt(r['cv_ratio'])} | {fmt(r['gini'])} | "
            f"{fmt(r['top10'])} | {fmt(r['p50'])} | {fmt(r['p90'])} | {fmt(r['zero_share'])} |")
    L += [
        "",
        "### この表の読み方(★片側バンドの非対称性)",
        "- **現実 CV は下界である**。公表されるのは行動者率 p・行動者平均時間 m・総平均時間 p·m "
        "の3つだけで、個体内のばらつきは不明。下界は `Var(x) ≥ p(1−p)m²` ⇒ "
        "`CV ≥ √((1−p)/p)`(= 行動者内のばらつきを 0 と仮定した最も保守的な値)。",
        "- したがって **分散比 < 1 なら「シムの分散は現実より確実に痩せている」と言えるが、"
        "分散比 ≥ 1 でも「十分」とは言えない**。この非対称性を報告から落とさないこと。",
        "- 分散比が事前登録の閾値を下回った指標については、**主張を集団レベルの定性的パターンに"
        "限定する**(Wu, Peng, Ito & Xiao 2025, arXiv:2506.19806 の基準③ → "
        "docs/plans/stationarity-preregistration.md 前文 ③(d))。",
        "- 現実 CV が「―」の行は**現実側の一次統計が未取得**という意味であり、合格ではない。",
        "- 対応の曖昧な活動カテゴリ(在宅・判定不能)には現実側を当てていない"
        "(「在宅」は場所であって行動分類ではなく、「判定不能」は調査の「その他」ではない)。",
        "- 「移動」は通勤・通学とそれ以外の移動を分離しないため、和集合の行動者率の"
        "**上界** p ≤ 0.643 から下界を採っている(p の下界を使うと下界にならない)。",
        "- 発話数・接触人数の現実バンドは **未取得**(個体間分布を与える一次統計を特定できていない)。",
    ]
    for doc in var["bands"]["docs"]:
        src = doc.get("source", {})
        L.append(f"- 参照: {doc.get('title')} — {src.get('publisher')} "
                 f"(取得 {src.get('accessed')} / status={doc.get('status')})")
    for err in var["bands"]["errors"]:
        L.append(f"- ⚠ 参照統計の読み込み失敗(バンド無しへ後退): {err}")
    att = bref.attribution_lines(var["bands"]["docs"])
    if att:
        L.append("")
        L.append("出典表記:")
        L += att

    L.append("")
    L.append("### 既存 L2 の集中度列(**読むだけ**・ラン末値)")
    L.append("| 列 | 値 |")
    L.append("|---|---:|")
    for col, v in (l2conc or {}).items():
        L.append(f"| {col} | {fmt(v)} |")
    L.append("")
    L.append("> これらは observer 側(凍結対象 `metrics_spec_hash`)で計算済みの列であり、"
             "本スクリプトは 1 バイトも再計算していない。列が「―」= そのランで該当 lens が OFF。")
    return L


def _render_activity(act: dict | None) -> list[str]:
    """生活時間配分表(カテゴリ×分/日・平日/休日別)+ 分布一致(KS/EMD)の節。"""
    L = ["", "## 生活時間配分(活動カテゴリ別 分/日)"]
    if not act or not act.get("n_all"):
        L.append("(活動を復元できるイベントが不足=データ不足)")
        return L
    L.append("L1 から各エージェントの在圏状態を step 単位(1step=10分)で復元し、活動日"
             "あたりの分に集計(判定できない step は計上しない=捏造なし)。")
    hdr = "| カテゴリ | 全体 分/日 |"
    sep = "|---|---|"
    if act["has_split"]:
        hdr += " 平日 分/日 | 休日 分/日 |"
        sep += "---|---|"
    hdr += " 参考バンド(分/日) |"
    sep += "---|"
    L.append(hdr)
    L.append(sep)
    for c in act["cats"]:
        ov = act["overall"].get(c)
        row = f"| {_ACT_LABEL_JP.get(c, c)} | {fmt(ov)} |"
        if act["has_split"]:
            row += f" {fmt(act['weekday'].get(c))} | {fmt(act['holiday'].get(c))} |"
        band = _ACT_REF_BAND.get(c)
        row += (f" {band[0]:.0f}〜{band[1]:.0f} |" if band else " ― |")
        L.append(row)
    L.append("")
    L.append(f"- 集計対象: 活動日 {act['n_all']}(平日 {act['n_wk']} / 休日 {act['n_hd']})"
             "= agent×日 の延べ。")
    for c in ("sleep", "work"):
        b = _ACT_REF_BAND.get(c)
        if b:
            L.append(f"- 参考({_ACT_LABEL_JP[c]}): {b[2]}")

    L.append("")
    L.append("## 分布一致指標(KS統計量・EMD・1次元自作)")
    L.append("> **参照分布の扱い**: 現実の生活時間調査の「カテゴリ別 分/日 の完全分布」は"
             "リポジトリ内に無いため、外部参照との KS/EMD は算出しない(上表のみ)。"
             "KS/EMD は純関数として提供し、ラン間比較・同一ラン内の平日/休日比較に適用する。")
    if act["has_split"] and act["dist"]:
        L.append("")
        L.append("同一ラン内の**平日 vs 休日**の分布差(agent×日 の分/日 分布):")
        L.append("| カテゴリ | KS統計量 | EMD(分) |")
        L.append("|---|---|---|")
        for c in ("sleep", "work"):
            d = act["dist"].get(c, {})
            L.append(f"| {_ACT_LABEL_JP.get(c, c)} | {fmt(d.get('ks'))} | "
                     f"{fmt(d.get('emd'))} |")
    else:
        L.append("(平日/休日の区分が取れない=weather 未観測のため、平日休日比較は非該当)")
    return L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_dir = args.out or os.path.join(args.run_dir, "calibration")
    os.makedirs(out_dir, exist_ok=True)
    m = analyze(args.run_dir)
    md = render(m)
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(md + "\n")
    slim = {k: v for k, v in m.items() if not isinstance(v, (list, dict))}
    slim["_meta"] = m["_meta"]
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=1, default=str)
    print(md)


if __name__ == "__main__":
    main()
