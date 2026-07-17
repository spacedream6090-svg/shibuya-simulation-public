"""ペルソナ生成(D6 の簡易版)。

P0 = 名前/性別/年齢/職業/家(実在の住宅建物)/職場(実在 POI)/就寝時刻の簡易
サンプリング + trait(tail 確保)。
本実装(IPF×e-Stat 渋谷 + LLM 文章化 + Verbalized Sampling)は P1 で差し替え(seam)。
注意: persona 文に構成概念(効力感・不満 等)を書かない(frame 分離)。
"""
from __future__ import annotations

import numpy as np

from ..economy import assign_part_time, initial_money, split_account, wage_amount
from ..factors.registry import (STATE_INIT, drift_params, drive_params,
                                reflect_trigger_params,
                                opinion_params, sample_traits)
from .agent import Agent

_GIVEN_F = ["美咲", "陽菜", "結衣", "葵", "凛", "さくら", "芽衣", "千夏", "真央", "花音"]
_GIVEN_M = ["翔太", "蓮", "大輝", "颯太", "拓海", "悠人", "健太", "亮", "光", "駿"]
_FAMILY = ["佐藤", "鈴木", "高橋", "田中", "伊藤", "渡辺", "山本", "中村", "小林", "加藤"]
_OCCUPATIONS = ["大学生", "会社員", "カフェ店員", "デザイナー", "エンジニア", "美容師",
                "フリーランス", "アパレル店員", "バンドマン", "配達員", "写真家", "無職"]

# 職業 → 職場 POI のカテゴリ(None = 決まった職場なし)
_WORK_CAT: dict[str, str | None] = {
    "会社員": "office", "エンジニア": "office", "デザイナー": "office",
    "カフェ店員": "food", "アパレル店員": "shop", "美容師": "shop",
    "大学生": "education",
    "フリーランス": None, "バンドマン": None, "配達員": None,
    "写真家": None, "無職": None,
}


def _pick_workplace(occupation: str, rng: np.random.Generator, city):
    """実在 POI から職場を選ぶ。(name, node, building, floor, start_min, end_min)

    ★v6 の POI 拡大で学校の cat が "education" → "school" になった(build_map)。旧地図
      (data/shibuya_osm.json)は "education" のまま=再生成しない。両対応のため、education
      が空のときだけ school へフォールバックする(旧地図では pois_by_cat('school')=[] で
      無効=既存の割当結果は不変。バイト一致を保つ)。"""
    cat = _WORK_CAT.get(occupation)
    if cat is None:
        return None
    pois = city.pois_by_cat(cat)
    if not pois and cat == "education":
        pois = city.pois_by_cat("school")           # 新地図: 学校は "school" cat
    pois = pois or city.pois_by_cat("office") or city.pois_by_cat("shop")
    if not pois:
        return None
    p = pois[int(rng.integers(len(pois)))]
    start = (8 * 60 + int(rng.integers(0, 16)) * 10)      # 8:00〜10:30
    hours = 5 if occupation == "大学生" else 8
    floor = int(p.get("floor", 0))
    bld = p.get("building", "")
    if bld and floor == 0:
        levels = int(city.building(bld)["levels"])
        floor = int(rng.integers(1, levels + 1))
    return (p["name"], p["node"], bld, max(1, floor),
            start, start + hours * 60)


def build_agent(agent_id: int, rng: np.random.Generator, nodes: list[str],
                city, tail_frac: float = 0.1,
                entry: dict | None = None,
                economy: dict | None = None,
                threshold_dist: str = "normal",
                drift: dict | None = None,
                reflection: dict | None = None,
                place_name: str = "この街") -> Agent:
    """entry(scripts/build_personas.py の名簿)があればそれを使い、
    無ければ手続き生成(v2 以前の互換動作: 全員居住者)。

    economy が渡されれば手持ち初期値・本業日給・バイトを職業別に決める(経済 v0)。
    threshold_dist は手続き生成経路のみに作用(名簿 entry 使用時は名簿値優先=現状のまま)。"""
    if entry:
        gender = entry["gender"]
        name = entry["name"]
        age = int(entry["age"])
        occupation = entry["occupation"]
        visitor = bool(entry.get("visitor", False))
        commute = bool(entry.get("commute", False))       # 流入通勤者(visitor の下位型)
        traits = {k: float(v) for k, v in entry["traits"].items()}
        drive_threshold = float(entry["drive_threshold"])
        fire_weight = float(entry["fire_weight"])
        bedtime_min = int(entry["bedtime_min"])
        sleep_steps = int(entry["sleep_steps"])
        has_bicycle = bool(entry["has_bicycle"])
        has_car = bool(entry["has_car"])
        persona_txt = str(entry["persona"])
    else:
        gender = "女" if rng.random() < 0.5 else "男"
        given = _GIVEN_F if gender == "女" else _GIVEN_M
        name = (f"{_FAMILY[int(rng.integers(len(_FAMILY)))]}"
                f"{given[int(rng.integers(len(given)))]}")
        age = int(rng.integers(18, 70))
        occupation = _OCCUPATIONS[int(rng.integers(len(_OCCUPATIONS)))]
        visitor = False
        traits = sample_traits(rng, tail_frac=tail_frac)
        drive_threshold, fire_weight = drive_params(traits, rng, threshold_dist)
        bedtime_min = (22 * 60 + int(rng.integers(0, 24)) * 10) % 1440
        sleep_steps = int(rng.integers(39, 49))  # 6.5h〜8h(10分刻み)
        has_bicycle = bool(rng.random() < 0.15)
        has_car = bool(rng.random() < 0.08)
        persona_txt = ""
        commute = False
    start = nodes[int(rng.integers(len(nodes)))]

    # ---- 家: 居住者=実在の住宅建物 / 来街者=街の外(駅・縁が帰路の起点)----
    home_building, home_floor = "", 1
    res = city.residential_buildings
    if visitor:
        home = city.station_node or nodes[int(rng.integers(len(nodes)))]
    elif res:
        b = res[int(rng.integers(len(res)))]
        home_building = b["id"]
        home_floor = int(rng.integers(1, int(b["levels"]) + 1))
        home = b["entrance"]
    else:                                   # 地図に住宅が無い場合の後退動作
        home = nodes[int(rng.integers(len(nodes)))]

    # ---- 職場 = 実在 POI(会社・店)。来街者=通勤流入もここで決まる ----
    work = _pick_workplace(occupation, rng, city)
    work_name, work_node, work_building, work_floor = "", "", "", 0
    work_start, work_end = -1, 0
    if work:
        work_name, work_node, work_building, work_floor, work_start, work_end = work

    # ---- 流入通勤者: 到着時刻(二峰分布)+ 夕方の退勤=帰宅トリガ(bedtime を流用)----
    #  すべて決定論(rng を引かない)= 既存名簿(commute 無し)の draw 順は不変。
    #  arrival_min = 職場 start から逆算(名簿の arrival_lead_min ぶん前に到着)。
    #  bedtime_min(=帰宅トリガ)は勤務終了より後になるよう max で持ち上げる(退勤後に homing)。
    arrival_min = -1
    commute_gateway = "station"
    residence_line = ""
    if commute:
        commute_gateway = str(entry.get("commute_gateway", "station"))
        residence_line = str(entry.get("residence_line", ""))
        lead = int(entry.get("arrival_lead_min", 40))
        if work_start >= 0:
            arrival_min = max(5 * 60, work_start - lead)         # 職場 start の lead 分前
            bedtime_min = max(int(bedtime_min), work_end + 10)   # 退勤後に帰宅
        else:
            arrival_min = 8 * 60 + 30                            # 職場不明時の既定(8:30)
        bedtime_min = min(int(bedtime_min), 23 * 60 + 50)        # 当日内に収める

    if not persona_txt:
        # 街名は基盤に持たせず envpack(place_name)から受ける。既定=渋谷(config.yaml)=バイト一致。
        if commute:
            where = residence_line or f"{place_name}の外"
            living = f"{where}に住んでいて、毎日{place_name}に通勤・通学している"
        elif visitor:
            living = f"{place_name}の街の外に住んでいて、よく{place_name}に来る"
        else:
            living = f"{place_name}の街で暮らしている"
        persona_txt = (f"あなたは{name}、{age}歳の{occupation}({gender}性)。"
                       f"{living}。自分の言葉で自然に、短く話す。")
    if work_name and work_name not in persona_txt:
        persona_txt += f"職場は{work_name}。"

    # ---- 経済 v0: 手持ち・本業日給・バイト(職業別、rng 経由)----
    money, wage, part_time = 0.0, 0.0, None
    account = 0.0
    if economy and economy.get("enabled"):
        money = initial_money(occupation, visitor, rng, economy)
        wage = wage_amount(occupation, work_start >= 0, economy)
        part_time = assign_part_time(occupation, visitor, city, rng, economy)
        # 口座 E5(既定 OFF): 居住者は初期資産を口座8割/現金2割に分割(rng 不使用=決定論)。
        money, account = split_account(money, visitor, economy)

    # ---- 意見力学(FJ): 感受性(traits→写像)+ 初期アンカー N(0, 0.3) clip[-1,1]。----
    # ★ rng 消費は build_agent の**末尾**に置く(既存ストリームの draw 位置を動かさない
    #   = 既定の再現性を維持)。中立気味の分布(平均0)で初期化する。
    opinion_susceptibility = opinion_params(traits, rng)
    opinion_anchor = float(np.clip(rng.normal(0.0, 0.3), -1.0, 1.0))

    # ---- 内省ドリフト E2(既定 OFF): 有効時のみ 1 draw を **末尾**で引く。無効時は
    #   一切引かない=既存ストリームの draw 位置を動かさない(既定挙動バイト一致)。----
    drift_p = None
    if drift and drift.get("enabled"):
        drift_p = drift_params(traits, rng, drift)

    # ---- 出来事誘発の深い内省+無意識層(第12バッチ、既定 OFF): drift と同じ規律。
    #   deep 有効時のみ 1 draw を末尾で引く。implicit_self 有効時は行動カウンタを持たせる。----
    reflect_p = None
    behav_today = behav_ema = None
    if reflection and reflection.get("deep", {}).get("enabled"):
        reflect_p = reflect_trigger_params(traits, rng, reflection["deep"])
    if reflection and reflection.get("implicit_self", {}).get("enabled"):
        behav_today, behav_ema = {}, {}

    return Agent(id=agent_id, name=name, age=age, occupation=occupation,
                 visitor=visitor, commute=commute, arrival_min=arrival_min,
                 commute_gateway=commute_gateway, residence_line=residence_line,
                 drive_threshold=drive_threshold, fire_weight=fire_weight,
                 gender=gender, persona=persona_txt, traits=traits,
                 states=dict(STATE_INIT), node=start,
                 home_node=home, home_building=home_building, home_floor=home_floor,
                 work_name=work_name, work_node=work_node,
                 work_building=work_building, work_floor=work_floor,
                 work_start_min=work_start, work_end_min=work_end,
                 bedtime_min=bedtime_min, sleep_steps=sleep_steps,
                 has_bicycle=has_bicycle, has_car=has_car,
                 money=money, wage=wage, part_time=part_time, account=account,
                 drift_p=drift_p,
                 reflect_p=reflect_p, behav_today=behav_today, behav_ema=behav_ema,
                 opinion=opinion_anchor, opinion_anchor=opinion_anchor,
                 opinion_susceptibility=opinion_susceptibility)
