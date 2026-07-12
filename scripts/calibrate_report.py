#!/usr/bin/env python
"""現実整合キャリブレーション・レポート(2026-07-09 ユーザー要望)。

    python scripts/calibrate_report.py runs/<name> [--out runs/<name>/calibration]

シミュレーション出力(L1/L2/agents.json/summary.json)を読むだけで、「現実の一次統計」
とのズレを表にする(sim⇄観測は疎結合。シム本体・schema は一切変更しない)。

現実基準値はすべて近似バンド(lo..hi)で、出典名を注記する(精密値の捏造はしない)。
mock LLM ラン では LLM 依存の行動頻度(発話・投稿・グループ設立など)は行動分布として
意味が薄いので、表では「LLM依存」と明記して機械系(睡眠・通勤・経済・事件率)と分ける。

判定: ✅=バンド内 / 高=上限の何倍か / 低=下限の何分の1か。
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))

# Windows コンソール(cp932)でも ✅ 等を印字できるように(ファイル出力は常に UTF-8)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from society.observer.measure import load_events  # noqa: E402

MIN_PER_DAY = 1440

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

    evs = load_events(run_dir)

    # ---- L1 走査(1パス)----
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

    for e in evs:
        k = e["kind"]
        p = e["payload"]
        aid = e["agent_id"]
        sm = e["sim_min"]
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
    return "\n".join(lines)


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
