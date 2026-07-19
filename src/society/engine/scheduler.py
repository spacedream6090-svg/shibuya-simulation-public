"""1 step の実行(フェーズ制・agent_id 昇順適用 = 決定論 D13)。

フェーズ: 起床/帰還 → 移動(道なり・経路ポリラインをログ) → 意思決定 →
行動適用 → 経験→state(seam) → 就寝直後の内省(個別分散) → 観測。
- 範囲外(loc=outside)と睡眠中は計算しない/知覚されない。
- ★発話は必ず LLM(routine の socialize → deliberate)。定型文なし(2026-07-04)。
engine は因子名を一切参照しない(no-fingerprint、tests/test_contracts.py で担保)。
"""
from __future__ import annotations

import math

from .. import annual as annual_mod
from .. import commerce as commerce_mod
from .. import disaster as disaster_mod
from .. import diversity as diversity_mod
from .. import freedom_p2 as freedom_p2_mod
from .. import health as health_mod
from .. import household as household_mod
from .. import inner_life as inner_life_mod
from .. import lodging as lodging_mod
from .. import opinion as opinion_mod
from .. import relations as relations_mod
from .. import status as status_mod
from .. import street as street_mod
from .. import worldview as worldview_mod
from ..net import infoenv as infoenv_mod
from ..cognition import deliberate, drive, planning, routine
from ..cognition.reflection import maybe_reflect
from ..economy import CIVIL_SERVANTS, civil_servant_pay, gig_profile, price_of
from ..government import Government, build_government_cfg
from .. import organizations, schedule, weather
from ..world import calendar
from ..lang.sentiment import valence
from ..factors import affect
from ..factors import update as factor_update
from ..observer.aggregate import collect
from ..observer.schema import Event
from ..rules import apply_bonus
from ..world.geom import rdp
from ..world.perception import build_index, hearers_of, salience_gate


def _edge_key(u: str, v: str) -> tuple[str, str]:
    return (u, v) if u <= v else (v, u)


# ---------------------------------------------------------------- 感情・興味・注意ハブ(affect)
def _affect_on(sim) -> bool:
    """感情・興味・注意ハブ(arousal + salience)が有効か。既定 OFF=新経路を一切通さない。"""
    return bool(getattr(sim, "affectcfg", {}).get("enabled"))


def _arouse(sim, agent, cause: str, step: int, sim_min: int, **signals) -> None:
    """観測イベントで agent の覚醒度 arousal を動かす(affect OFF なら完全 no-op=バイト一致)。

    signals は factors/affect の観測量(valence_abs / novelty / addressed / congestion)。
    on_arousal は gain=0 でも no-op(arousal 不変・ログなし)。engine は不透明な数値だけを渡す。"""
    if not _affect_on(sim):
        return
    factor_update.on_arousal(agent, sim.affectcfg, cause=cause, step=step,
                             sim_min=sim_min, logger=sim.logger, **signals)


def _imp_bonus(sim, agent, novelty: float = 0.0) -> float:
    """観測時の memory.importance 加点(affect OFF なら 0=固定 3.0 のまま)。"""
    if not _affect_on(sim):
        return 0.0
    return affect.importance_bonus(agent, sim.affectcfg, novelty=novelty)


# ---------------------------------------------------------------- 社会関係の質(Wave G2)
def _relations_on(sim) -> bool:
    """社会関係の質(tier/decay/reputation/派閥)が有効か。既定 OFF=新経路を一切通さない
    (record_contact に closeness を渡さない=台帳・プロンプト・イベント列がバイト一致)。"""
    cfg = getattr(sim, "relationscfg", None)
    return bool(cfg and cfg["enabled"])


def _contact(sim, actor, other_id: int, other_name: str, text: str,
             valence: float, step: int, sim_min: int) -> None:
    """交流1件を actor→other の関係台帳へ記録する。relations OFF なら従来の record_contact
    のみ(closeness を付けない=バイト一致)。ON なら符号つき closeness 更新 + tier 変化ログ。"""
    if _relations_on(sim):
        relations_mod.note_contact(actor, other_id, other_name, text, valence,
                                   sim.relationscfg, step, sim_min, sim.logger)
    else:
        actor.mem.record_contact(other_id, other_name, step, text)


def _steps_until_tod(cur_sim_min: int, target_min: int) -> int:
    """現在の sim_min から、次に time-of-day=target_min になるまでの step 数(正)。

    流入通勤者の帰宅(夕)→翌朝の到着(arrival_min)までの外部滞在を決める。二峰分布の
    jitter は arrival_min 自体に内包済みなので、ここは決定論(乱数なし)。"""
    delta = (int(target_min) - cur_sim_min % 1440) % 1440
    if delta == 0:
        delta = 1440                                   # 同時刻 = 次の周回(翌日)
    return delta // 10


def _percept(sim):
    """位置が安定なフェーズ(_phase_drive/_decide)用の知覚ソース。

    run_step が step ごとに1回だけ張る空間索引 sim.percept_index を返す。索引が
    無い(内部関数を直接呼ぶテスト等)ときは全 agent リストへ後退=live 走査で
    従来と完全同一。位置が動く _apply は索引を使わず sim.agents を直接渡す。
    """
    idx = getattr(sim, "percept_index", None)
    return idx if idx is not None else sim.agents


def _edge_pts(city, u: str, v: str) -> list[tuple[float, float]]:
    data = city.graph.edges[u, v]
    geometry = data["geometry"]
    return list(geometry) if data["u0"] == u else list(reversed(geometry))


def _place_of(sim, agent) -> str:
    if agent.building:
        b = sim.city.building(agent.building)
        if agent.building == agent.home_building:
            return "自宅(マンションの部屋)"
        name = b["name"] or "雑居ビル"
        floor_pois = sim.city.pois_in_building(agent.building, agent.floor)
        if floor_pois:
            return f"{name} {agent.floor}階({floor_pois[0]['name']})"
        return f"{name} {agent.floor}階"
    if agent.node == agent.home_node:
        return "自宅の前"
    return sim.city.place_label(agent.node)


# ---------------------------------------------------------------- 経済 v0
def _economy_on(sim) -> bool:
    return bool(getattr(sim, "economy", {}).get("enabled"))


def _accounts_on(sim) -> bool:
    """口座(銀行)概念 E5 が有効か(経済有効 かつ economy.accounts.enabled)。"""
    econ = getattr(sim, "economy", {}) or {}
    return bool(econ.get("enabled")) and bool(econ.get("accounts", {}).get("enabled"))


def _money_median(sim) -> float:
    """参照集団(現在この街に居る=loc!='outside')の所持金の中央値(Wave G1・決定論・乱数なし)。

    相対的剥奪の参照点。sorted で決定論的に算出(屋外不在者=現在の街に居ない者は除外)。
    参照は簡潔に全体(街に居る全員)の中央値(近傍限定は将来版)。参照が空なら 0.0。"""
    vals = sorted(a.money for a in sim.agents if a.loc != "outside")
    if not vals:
        return 0.0
    n = len(vals)
    mid = n // 2
    if n % 2 == 1:
        return float(vals[mid])
    return float((vals[mid - 1] + vals[mid]) / 2.0)


# ---------------------------------------------------------------- 行政(税・給与・給付)
def _gov(sim):
    """行政主体(ward/metro/nation)を遅延構築して返す(simulation.py は別バッチ所有=編集不可)。

    初回アクセス時に config の government ブロックから構築して sim にキャッシュ。以降は同一実体を
    再利用(残高がラン内で持続する)。構築自体は乱数を引かず・ログもしない=OFF 時の副作用ゼロ。
    """
    gov = getattr(sim, "government", None)
    if gov is None:
        from omegaconf import OmegaConf
        raw = sim.cfg.get("government", None)
        raw = (OmegaConf.to_container(raw, resolve=True)
               if OmegaConf.is_config(raw) else dict(raw or {}))
        # 制度値は simulation.py が正準化した institutions ブロック(D1-W3)から供給する。
        inst = getattr(sim, "institutionscfg", None)
        gov = Government(build_government_cfg(raw, institutions=inst))
        sim.government = gov
    return gov


def _government_on(sim) -> bool:
    """行政(税・給与・給付)が有効か。既定 OFF(config に government が無ければ False)。"""
    return bool(_gov(sim).cfg["enabled"])


def _orgs_on(sim) -> bool:
    """組織(職場・学校の実態)が有効か。既定 OFF(config に organizations が無ければ False)。"""
    cfg = getattr(sim, "orgscfg", None)
    if cfg is None:
        cfg = organizations.build_orgs_cfg(sim.cfg.get("organizations", None))
        sim.orgscfg = cfg
    return bool(cfg["enabled"])


def _ensure_orgs(sim) -> None:
    """組織台帳の遅延初期化(step 先頭で1回)。OFF なら完全 no-op=既存挙動と不変。

    simulation.py を編集せずに済ませる据え付け(government の _gov と同型)。台帳・配属は
    決定論の事前計算データ(乱数なし)。配属の無い agent には何も付かない(仕様 §6)。"""
    if not _orgs_on(sim) or getattr(sim, "orgs", None) is not None:
        return
    sim.orgs = organizations.load_book(sim.orgscfg)
    sim.org_ledger = {}
    personas_file = sim.cfg.get("agents", {}).get("personas_file")
    organizations.attach(sim.agents, sim.orgs, personas_file, city=sim.city,
                         commute_to_poi=bool(sim.orgscfg.get("commute_to_poi", False)),
                         assignments_path=sim.orgscfg.get("assignments"))


def _log_org_output(sim, agent, step: int, sim_min: int) -> None:
    """勤務完遂→産出(production)/ 登校完遂→学習(study)を1行記録(非LLM・乱数なし)。

    産出は雇用主 org に帰属(物理の職場 POI を台帳へ束ねる変更は仕様 §3 のオプトイン seam
    として保留=正直な近似)。組織会計(org_ledger)は「集計だけ」(仕様 §1.3、市場は B 段)。"""
    org = getattr(sim, "orgs", {}).get(getattr(agent, "org_id", None) or "")
    if not org:
        return
    day = sim_min // 1440
    if getattr(agent, "org_role", "") == "学生":
        subject = organizations.daily_subject(org, day)
        if subject is None:
            return
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                             kind="study", x=agent.x, y=agent.y,
                             payload={"org": str(org["id"]), "subject": subject,
                                      "role": "学生"}))
        return
    out = organizations.daily_output(org, day)
    if out is None:
        return
    output, kind = out
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                         kind="production", x=agent.x, y=agent.y,
                         payload={"org": str(org["id"]), "output": output,
                                  "kind": kind}))
    led = sim.org_ledger.setdefault(str(org["id"]), {"production_count": 0,
                                                     "revenue_est": 0.0,
                                                     "wage_paid": 0.0})
    led["production_count"] += 1
    if _economy_on(sim):
        wage = float(sim.economy["wages"].get(str(org.get("wage_tier", "")), 0.0))
        led["revenue_est"] += wage * float(sim.orgscfg["revenue_margin"])
        led["wage_paid"] += wage


def _log_tax(sim, agent, tax: str, amount: float, to: str, base: float,
             step: int, sim_min: int) -> None:
    """税の徴収を記録(kind=tax)。所得税=income→nation / 住民税=resident→ward,metro /
    消費税=consumption→nation(国分),metro(地方分)。base=課税の元(名目賃金 or 名目価格)。"""
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                         kind="tax", x=agent.x, y=agent.y,
                         payload={"tax": tax, "amount": round(float(amount), 1),
                                  "to": to, "base": round(float(base), 1)}))


def _withhold_wage(sim, agent, gross: float, step: int, sim_min: int) -> tuple[float, float]:
    """賃金の源泉徴収: 所得税(→nation)+住民税(→ward6:metro4)を控除し (手取り, 税合計) を返す。

    税は該当予算に歳入計上し、tax イベントで記録。手取り = 名目 − 税(engine 側で口座/現金に入金)。
    """
    gov = sim.government
    it = gov.income_tax(gross)
    rw, rm = gov.resident_tax(gross)
    if it > 0:
        gov.collect("nation", it)
        _log_tax(sim, agent, "income", it, "nation", gross, step, sim_min)
    if rw > 0:
        gov.collect("ward", rw)
        _log_tax(sim, agent, "resident", rw, "ward", gross, step, sim_min)
    if rm > 0:
        gov.collect("metro", rm)
        _log_tax(sim, agent, "resident", rm, "metro", gross, step, sim_min)
    tax_total = it + rw + rm
    return gross - tax_total, tax_total


def _record_consumption_tax(sim, agent, price: float, cat: str,
                            step: int, sim_min: int) -> None:
    """消費税の内訳計上(価格は名目不変=内税)。国分→nation / 地方分→metro に歳入計上+tax 記録。"""
    national, local, _rate = sim.government.consumption_tax(price, cat)
    if national > 0:
        sim.government.collect("nation", national)
        _log_tax(sim, agent, "consumption", national, "nation", price, step, sim_min)
    if local > 0:
        sim.government.collect("metro", local)
        _log_tax(sim, agent, "consumption", local, "metro", price, step, sim_min)


def _pay_wage(sim, agent, amount: float, step: int, sim_min: int,
              source: str | None = None, fund_level: str | None = None) -> None:
    """賃金の支給(本業の勤務完遂・バイトのシフト完遂・自営の日銭・月給まとめ・公務員給与)。

    source を渡すと payload に載せる(自営の日銭は "gig"、月給は "salary"、公務員は "civil")。
    既定 None のときは載せない=口座 OFF・行政 OFF の本業/バイト wage payload は現行と完全同一。
    口座 ON(E5): 入金先を口座にする("to":"account")。家賃計算の元(period_income)も積む。
    行政 ON: 名目 gross から所得税+住民税を源泉徴収し**手取り(net)を入金**(payload.amount=手取り、
      追加キー gross/tax を載せる=手取り+税=名目)。fund_level 指定(公務員給与)なら gross を該当
      予算から歳出(expense)計上する(区職員=ward / 警察官・消防士=metro)。行政 OFF 時は
      gross=net で追加キーも出さない=既存 wage payload とバイト一致。"""
    if amount <= 0:
        return
    gross = float(amount)
    tax_total = 0.0
    if _government_on(sim):
        if fund_level is not None:                     # 公務員給与は予算が出所(歳出)
            sim.government.expense(fund_level, gross)
        amount, tax_total = _withhold_wage(sim, agent, gross, step, sim_min)  # 手取り
    if _accounts_on(sim):
        agent.account += amount
        agent.period_income += amount          # 月収相当(家賃=share×これ)
        payload = {"amount": round(float(amount), 1),
                   "balance": round(agent.money, 1),        # 現金(不変)
                   "to": "account", "account": round(agent.account, 1)}
    else:
        agent.money += amount
        payload = {"amount": round(float(amount), 1),
                   "balance": round(agent.money, 1)}
    if source is not None:
        payload["source"] = source
    if tax_total > 0:                          # 行政 ON のみ: 名目と税を併記(会計の検証点)
        payload["gross"] = round(gross, 1)
        payload["tax"] = round(tax_total, 1)
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                         kind="wage", x=agent.x, y=agent.y, payload=payload))


def _atm_withdraw(sim, agent, need: float, step: int, sim_min: int) -> None:
    """現金不足時の自動引き出し(kind="withdraw")。移動は伴わない簡易版=ATM はどこにでも
    ある近似(conbini/shop/駅 が街中に密にある想定を正直に置く)。口座から atm_withdraw を
    基本単位に、不足分を満たすだけ現金へ移す(口座残高が上限)。"""
    acc = sim.economy["accounts"]
    if agent.account <= 0.0:
        return
    draw = min(agent.account, max(float(acc["atm_withdraw"]), need))
    agent.account -= draw
    agent.money += draw
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                         kind="withdraw", x=agent.x, y=agent.y,
                         payload={"amount": round(draw, 1),
                                  "cash": round(agent.money, 1),
                                  "account": round(agent.account, 1)}))


def _spend(sim, agent, amount: float, cat: str, step: int, sim_min: int,
           chosen: bool = False) -> None:
    """消費(食事・買い物・nightlife・taxi・bus)。残高は 0 未満にしない。

    口座 ON(E5): 金額 ≥ card_threshold は口座から(カード)。未満は現金で、現金が
    足りなければ最寄り ATM で自動引き出し(withdraw)してから支払う。
    chosen(P2 #7 buy・既定 False=既存呼び出しは payload 不変=バイト一致): LLM が発火時に
    能動選択した消費に payload へ chosen:true を添える(非発火の抽選消費と区別する観測用)。"""
    if amount <= 0:
        return
    if not _accounts_on(sim):
        agent.money = max(0.0, agent.money - amount)
        payload = {"amount": round(float(amount), 1),
                   "balance": round(agent.money, 1), "cat": cat}
        if chosen:
            payload["chosen"] = True
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                             kind="spend", x=agent.x, y=agent.y, payload=payload))
        if _government_on(sim):                 # 消費税を内訳計上(価格は名目不変)
            _record_consumption_tax(sim, agent, amount, cat, step, sim_min)
        return
    acc = sim.economy["accounts"]
    if amount >= float(acc["card_threshold"]) and agent.account >= amount:
        agent.account -= amount                # カード払い(口座から直接)
        src = "card"
    else:
        if agent.money < amount:               # 現金不足 → ATM 引き出し
            _atm_withdraw(sim, agent, amount - agent.money, step, sim_min)
        agent.money = max(0.0, agent.money - amount)
        src = "cash"
    payload = {"amount": round(float(amount), 1),
               "balance": round(agent.money, 1), "cat": cat,
               "src": src, "account": round(agent.account, 1)}
    if chosen:
        payload["chosen"] = True
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                         kind="spend", x=agent.x, y=agent.y, payload=payload))
    if _government_on(sim):                     # 消費税を内訳計上(価格は名目不変)
        _record_consumption_tax(sim, agent, amount, cat, step, sim_min)


def _settle_work(sim, agent, step: int, sim_min: int) -> None:
    """退館時: 勤務・バイトの完遂 = 達成経験(factors 側で state+)+ 賃金(経済 v0)。"""
    m = sim_min % 1440
    main_done = (agent.building == agent.work_building and agent.work_start_min >= 0
                 and m >= agent.work_end_min and agent.activity == "working")
    pt = agent.part_time
    pt_done = bool(pt and agent.building == pt["building"]
                   and m >= pt["end_min"] and agent.activity == "working")
    if not (main_done or pt_done):
        return
    delta = factor_update.on_work_done(agent, mags=sim.mags, step=step,
                                       sim_min=sim_min, logger=sim.logger)
    if delta:
        drive.add(agent, "state_change", sim.drivecfg, scale=abs(delta))
    if _economy_on(sim):
        # 口座 ON(E5): 月給者(本業日給を持つ会社員・店員)は日割り支給をやめ、勤務日数を
        #   積んで給料日にまとめ支給する。バイト・日銭は従来どおり都度(口座へ)入金。
        if main_done and _accounts_on(sim) and agent.wage > 0:
            agent.work_days += 1
        else:
            _pay_wage(sim, agent, agent.wage if main_done else float(pt["pay"]),
                      step, sim_min)
    if main_done and _orgs_on(sim):     # 組織: 勤務完遂→産出 / 登校完遂→学習(既定 OFF)
        _log_org_output(sim, agent, step, sim_min)


def _charge_meal(sim, agent, step: int, sim_min: int) -> None:
    """飲食 POI 到着時に代金を支払う(食事/nightlife)。

    商業ダイナミクス(H3)ON 時は需要(在館数)に応じて価格係数を掛け(price_change)、需要集中で
    品切れ/行列なら購入を抑制する(stock_out+不満)。commerce OFF なら基準価格をそのまま払う(不変)。"""
    if not _economy_on(sim):
        return
    cat = next((p["cat"] for p in sim.city.pois_at_node(agent.node)
                if p["cat"] in ("food", "nightlife")), None)
    if cat is None:
        return
    amount = price_of(cat, sim.economy, getattr(sim, "rulebook", None))
    if _commerce_on(sim):                          # 動的価格/在庫を消費額に反映(既存 draw の外・決定論)
        amount = commerce_mod.on_purchase(sim, agent, cat, amount, step, sim_min)
        if amount is None:                         # 品切れ/行列 → 購入抑制(spend を出さない)
            return
    _spend(sim, agent, amount, cat, step, sim_min)


def _charge_ride(sim, agent, ride: dict, step: int, sim_min: int) -> None:
    """交通機関(タクシー/簡易バス)の到着時課金。乗車を ride で記録し、運賃を spend。

    cat は乗り物の種別(taxi/bus)。経済が無効なら _spend が no-op(残高は 0 のまま)。"""
    mode = str(ride.get("mode", "taxi"))
    fare = float(ride.get("fare", 0.0))
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                         kind="ride", x=agent.x, y=agent.y,
                         payload={"mode": mode, "fare": round(fare, 1),
                                  "from": ride.get("from"), "to": ride.get("to")}))
    _spend(sim, agent, fare, mode, step, sim_min)


# ---------------------------------------------------------------- マイクロ移動
def _phase_jitter(sim, step: int, sim_min: int) -> None:
    """路上での滞在中は完全静止しない: 同一ノード周辺で ±5m の揺らぎ(seed 付き・毎step)。

    移動中(route あり)・睡眠・屋内・範囲外は動かさない。勤務・バイト中(路面の職場を
    含む)も動かさない(ユーザー方針: work と睡眠は変えない)。滞在でも発話でも、路上に
    留まっている限り毎 step わずかに位置を調整する(「数十分同じ座標で静止」を解消)。
    """
    for agent in sim.agents:
        if agent.loc != "street" or agent.sleeping or agent.building or agent.route:
            continue
        at_work = (routine.in_work_window(agent, sim_min,
                                          getattr(sim, "calendarcfg", None))
                   and agent.node == agent.work_node)
        at_pt = (agent.part_time and routine.in_part_time_window(agent, sim_min)
                 and agent.node == agent.part_time["node"])
        if at_work or at_pt:
            continue
        bx, by = sim.city.node_xy(agent.node)
        rng = sim.hub.stream("jitter", agent.id, step)
        agent.x = bx + float(rng.uniform(-5.0, 5.0))
        agent.y = by + float(rng.uniform(-5.0, 5.0))


# ---------------------------------------------------------------- 起床・帰還
def _schedule_plan(sim, agent, step: int, sim_min: int) -> None:
    """起床(来街者は帰還)の直後の step に朝の一日計画を1回予約する(1日1回)。

    planning 無効時は何もしない(agent の計画フィールドに触れない=旧挙動の再現性)。"""
    if not sim.planningcfg["enabled"]:
        return
    day = sim_min // 1440
    if agent.plan_day == day:                     # その日はもう予約/生成済み
        return
    agent.plan_day = day                          # 同日の再予約を防ぐ
    agent.plan_step = step + 1


def _phase_planning(sim, step: int, sim_min: int) -> None:
    """予約された step に達したエージェントの朝の一日計画を LLM で生成する。

    起床時刻は個体差があるため呼び出しは自然に時間分散する。k 条件に依らず全員・毎朝
    1回(R1)。外・睡眠中はスキップ(その日は予約済み扱いのまま=二重生成しない)。"""
    if not sim.planningcfg["enabled"]:
        return
    for agent in sim.agents:
        if agent.plan_step != step:
            continue
        agent.plan_step = -1
        if agent.loc == "outside" or agent.sleeping:
            continue
        planning.make_plan(sim, agent, step, sim_min, _place_of(sim, agent))


def _phase_wake_and_returns(sim, step: int, sim_min: int) -> None:
    for agent in sim.agents:
        if agent.sleeping and step >= agent.sleep_until:
            agent.sleeping = False
            agent.stay_until = step + 1
            sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                                 kind="wake_up", x=agent.x, y=agent.y,
                                 payload={"slept_steps": agent.sleep_steps}))
            _schedule_plan(sim, agent, step, sim_min)   # 起床 → 朝の計画を予約
        if agent.loc == "outside" and step >= agent.return_at:
            via_station = (agent.return_gateway == sim.city.station_node)
            if via_station and not sim.transit.has_service(sim_min):
                continue                          # 終電後は帰れない(始発待ち)
            agent.loc = "street"
            agent.node = agent.return_gateway
            agent.x, agent.y = sim.city.node_xy(agent.node)
            agent.stay_until = step + 1
            sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                                 kind="enter_area", x=agent.x, y=agent.y,
                                 payload={"gateway": agent.return_gateway,
                                          "via": "train" if via_station else "walk"}))
            if agent.visitor:                     # 来街者の帰還 → 朝の計画を予約
                _schedule_plan(sim, agent, step, sim_min)
                # 来街者の財布補充(改善 P2 第9バッチ・既定 OFF): 帰宅のたび手持ちを基準額まで
                # 戻す(家で財布を補充する近似)。来街者は反復収入が無く長期ランで恒久破綻し
                # 消費・宿泊から脱落する(sim-improvement-analysis.md P2)ことへの対処。
                # 決定論(乱数なし・上限までの差額補充)・k 非依存(R1)。OFF=補充なし=従来と完全同一。
                if _visitor_refresh_on(sim):
                    base = float(sim.economy.get("allowance_visitor", 20000))
                    if agent.money < base:
                        delta = base - agent.money
                        agent.money = base
                        sim.logger.log(Event(step=step, sim_min=sim_min,
                                             agent_id=agent.id, kind="wage",
                                             x=agent.x, y=agent.y,
                                             payload={"amount": round(delta, 1),
                                                      "balance": round(agent.money, 1),
                                                      "to": "cash",
                                                      "source": "home_refill"}))


# ---------------------------------------------------------------- 移動
def _phase_move(sim, step: int, sim_min: int) -> None:
    occupancy: dict[tuple[str, str], int] = {}
    for agent in sim.agents:
        if agent.loc == "street" and not agent.sleeping and agent.route:
            key = _edge_key(agent.node, agent.route[0])
            occupancy[key] = occupancy.get(key, 0) + 1

    capacity = int(sim.cfg.world.edge_capacity)
    speeds = sim.cfg.world.modes.speeds
    for agent in sim.agents:
        agent._arrived_new = False
        if agent.loc != "street" or agent.sleeping or not agent.route:
            agent._congestion = 1.0
            continue
        key = _edge_key(agent.node, agent.route[0])
        count = occupancy.get(key, 0)
        factor = 1.0 if count <= capacity else max(0.3, capacity / count)
        agent._congestion = factor

        pts: list[tuple[float, float]] = [(agent.x, agent.y)]
        budget_m = float(speeds[agent.trip_mode]) * factor
        moved = 0.0
        while budget_m > 0 and agent.route:
            edge_len = sim.city.edge_length(agent.node, agent.route[0])
            remain = edge_len - agent.edge_offset
            if budget_m >= remain:
                geom = _edge_pts(sim.city, agent.node, agent.route[0])
                pts.extend(geom[1:])
                moved += remain
                budget_m -= remain
                agent.node = agent.route.pop(0)
                agent.edge_offset = 0.0
            else:
                agent.edge_offset += budget_m
                moved += budget_m
                budget_m = 0.0
        if agent.route:
            agent.x, agent.y = sim.city.xy_along(agent.node, agent.route[0],
                                                 agent.edge_offset)
            pts.append((agent.x, agent.y))
        else:
            agent.x, agent.y = sim.city.node_xy(agent.node)
        # 経路ポリライン(道なりの見た目再現用)。形状を保って間引く(RDP)。
        pts = rdp(pts, epsilon=4.0, max_points=20)
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                             kind="move_segment", x=agent.x, y=agent.y,
                             payload={"dist_m": round(moved, 1),
                                      "mode": agent.trip_mode,
                                      "congestion": round(factor, 3),
                                      "pts": [[round(p[0], 1), round(p[1], 1)]
                                              for p in pts]}))
        delta = factor_update.on_congestion(agent, factor, mags=sim.mags,
                                            step=step, sim_min=sim_min,
                                            logger=sim.logger)
        if factor < 1.0:
            drive.add(agent, "congestion", sim.drivecfg, scale=(1.0 - factor))
            _arouse(sim, agent, "congestion", step, sim_min,
                    congestion=(1.0 - factor))          # 混雑ストレス→覚醒↑(affect OFF=no-op)
        if delta:
            drive.add(agent, "state_change", sim.drivecfg, scale=abs(delta))
        if not agent.route:                        # 到着
            agent.trip_mode = "walk"
            agent.activity = getattr(agent, "_pending_activity", "") or ""
            agent._pending_activity = ""
            if agent.activity == "eating":         # 飲食 POI 到着 → 代金を払う(経済 v0)
                _charge_meal(sim, agent, step, sim_min)
            ride = getattr(agent, "_ride_pending", None)
            if ride:                               # 交通機関で来た → 到着時に運賃を払う
                _charge_ride(sim, agent, ride, step, sim_min)
                agent._ride_pending = None
            first = agent.visits[agent.node] == 0
            agent.visits[agent.node] += 1
            agent._arrived_new = first
            agent.stay_until = step + agent._pending_stay
            agent.dest = None
            if first:
                drive.add(agent, "novel_place", sim.drivecfg)
                _arouse(sim, agent, "novel_place", step, sim_min,
                        novelty=1.0)                    # 初訪問=新奇→覚醒↑(affect OFF=no-op)
            # 公園・緑地に着いた → 回復環境(grievance−)+ 制度DSL: bonus(park)
            if any(p["cat"] == "leisure" for p in sim.city.pois_at_node(agent.node)):
                factor_update.on_park(agent, mags=sim.mags, step=step,
                                      sim_min=sim_min, logger=sim.logger)
                apply_bonus(getattr(sim, "rulebook", None), sim, agent, "park",
                            step, sim_min)
            sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                                 kind="arrive", x=agent.x, y=agent.y,
                                 payload={"node": agent.node,
                                          "name": sim.city.node_name(agent.node),
                                          "first_visit": first}))
            tools = getattr(sim, "tools", None)        # イベント参加確定/ビラ閲覧/屋台購入
            if tools is not None:
                tools.on_arrive(sim, agent, step, sim_min)
            if getattr(agent, "lodging_intent", False) \
                    and agent.node == agent.lodging_node:   # ホテル到着=チェックイン(Wave L。既定OFF=フラグ立たず no-op)
                _lodging_checkin(sim, agent, step, sim_min)
            if agent.exit_intent:
                _try_exit(sim, agent, step, sim_min)   # homing はこの中で参照する
            if not agent.route:                        # 再ルート(終電後の徒歩帰宅)以外
                agent.homing = False


def _try_exit(sim, agent, step: int, sim_min: int) -> None:
    if _lodging_on(sim) and _maybe_lodge(sim, agent, step, sim_min):
        return                                     # 夜の帰宅の代わりにホテルへ(退出しない。Wave L)
    via_station = (agent.node == sim.city.station_node)
    if via_station and not sim.transit.has_service(sim_min):
        if agent.homing and sim.city.gateways:     # 終電後の帰宅: 徒歩で縁へ
            gate = sim.city.gateways[agent.id % len(sim.city.gateways)]
            path, used_mode = sim.router.route(agent.node, gate, "walk")
            if len(path) >= 2:
                agent.route = path[1:]
                agent.edge_offset = 0.0
                agent.trip_mode = used_mode
                return                             # exit_intent 維持 → 縁で再試行
        agent.exit_intent = False                  # 終電後: 出かけるのをやめる
        return
    agent.exit_intent = False
    agent.loc = "outside"
    agent.return_gateway = agent.node
    rng = sim.hub.stream("outside", agent.id, step)
    homing_exit = agent.homing
    if homing_exit:                                # 来街者の帰宅: 睡眠+朝の移動で戻る
        agent.homing = False
        if _lodging_on(sim):
            agent.lodging_nights = 0               # 帰宅(範囲外退出)= 連泊カウントをリセット(Wave L)
        if getattr(agent, "commute", False) and agent.arrival_min >= 0:
            # 流入通勤者: 翌朝の到着時刻(arrival_min)に再流入(往復時刻を安定させる)
            agent.return_at = step + _steps_until_tod(sim_min, agent.arrival_min)
        else:
            agent.return_at = step + agent.sleep_steps + int(rng.integers(0, 7))
        agent.reflect_step = step + 1              # 帰路の電車で今日を内省(k 処置)
    else:
        span = sim.cfg.world.outside_steps
        agent.return_at = step + int(rng.integers(int(span[0]), int(span[1])))
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                         kind="exit_area", x=agent.x, y=agent.y,
                         payload={"gateway": agent.node, "homing": homing_exit,
                                  "via": "train" if via_station else "walk"}))


# ---------------------------------------------------------------- 背景交通
def _phase_traffic(sim, step: int, sim_min: int) -> None:
    """エージェント以外の通過車両(実規模の車の流れ)。可視化用に折れ線を記録。

    world.traffic.mode=od のとき、車を個体化した OD 走行に切り替える(1度だけ遅延設定)。
    既定 ambient は現行と完全同一(payload の n/total/segs も不変。log_extra() は空を返す)。
    """
    sim.traffic.ensure_mode(sim.cfg)
    segs = sim.traffic.step(step, sim_min)
    if not segs and not sim.traffic.enabled:
        return
    payload = {"n": len(segs), "total": sim.traffic.total_spawned,
               "segs": [s["pts"] for s in segs[:sim.traffic.max_log]]}
    payload.update(sim.traffic.log_extra())    # ambient={} / od={mode, cars}
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=-1,
                         kind="traffic_flow", x=0.0, y=0.0, payload=payload))


# ---------------------------------------------------------------- 共通: 聴取
def _hear_words(sim, listener, words: list[str], from_id: int, channel: str,
                step: int, sim_min: int) -> None:
    """語の聴取(対面/SNS/DM/検索/ニュース共通)。未知語は検索キューにも積む。"""
    if not words:
        return
    unknown = [w for w in words if w not in listener.adopted]
    # 多言語の伝播障壁(後続波 H5): 話者と聞き手の言語が異なるなら採用閾値を上げる(異言語間は
    # 語が広まりにくい)。diversity OFF は barrier=0=on_hear が従来と完全同一(バイト一致)。
    barrier = diversity_mod.cross_barrier(sim, from_id, listener) \
        if _diversity_on(sim) else 0
    # SNS/DM が架橋した物理距離(第22バッチ P2)。既定 OFF=None → payload・乱数・呼数とも不変。
    dist_m = None
    if channel in ("sns", "dm") and getattr(sim, "snsgeo_on", False):
        sender = sim.agent_by_id.get(from_id)
        if sender is not None:
            dist_m = round(math.hypot(sender.x - listener.x,
                                      sender.y - listener.y), 1)
    got_unknown = sim.labels.on_hear(listener, words, from_id, step=step,
                                     sim_min=sim_min, channel=channel,
                                     logger=sim.logger, extra_threshold=barrier,
                                     dist_m=dist_m)
    listener._heard_unknown = listener._heard_unknown or got_unknown
    if channel in ("face", "dm"):
        drive.add(listener, "addressed", sim.drivecfg)
    if got_unknown:
        drive.add(listener, "unknown_word", sim.drivecfg)
    for w in unknown:
        if channel != "search" and w not in listener._search_queue \
                and len(listener._search_queue) < 3:
            listener._search_queue.append(w)
    # 自分の造語が誰かに採用された → 創出者の当事者意識・効力感(影響の実感)
    for w in unknown:
        if w in listener.adopted:                  # このやり取りで採用に至った
            item = sim.labels.text_to_item.get(w)
            creator = sim.agent_by_id.get(item.creator) if item else None
            if creator is not None and creator.id != listener.id:
                delta = factor_update.on_own_adopted(
                    creator, mags=sim.mags, step=step, sim_min=sim_min,
                    logger=sim.logger)
                if delta:
                    drive.add(creator, "state_change", sim.drivecfg,
                              scale=abs(delta))
                # 評判(Wave G2): 自分の造語が採用された=影響が広まった → 評判+(既定 OFF=no-op)。
                if _relations_on(sim):
                    relations_mod.gain_reputation(
                        creator, sim.relationscfg["rep_adopted"], "own_adopted",
                        step, sim_min, sim.logger)
                # D9 過正当化 ablation: 採用に外的報酬を付与(既定off=完全に現状維持)
                if getattr(sim, "rewards_on", False):
                    creator.money += sim.reward_amount
                    sim.logger.log(Event(
                        step=step, sim_min=sim_min, agent_id=creator.id,
                        kind="reward", x=creator.x, y=creator.y,
                        payload={"amount": round(float(sim.reward_amount), 1),
                                 "balance": round(creator.money, 1)}))


# ---------------------------------------------------------------- 意見更新(FJ)
def _shift_opinion(sim, listener, expressed_value: float, source_id: int,
                   w: float, step: int, sim_min: int) -> None:
    """聞いた/読んだテキストの感情価で聞き手の意見を Friedkin-Johnsen 更新(#16)。

    決定論(乱数なし)。**プロンプトには一切注入しない**(no-fingerprint・R1 無風)。
    |Δ|<1e-4 は記録しない(ログ肥大防止)。source が listener の意見を動かした量は
    observer(measure)で「他者の認知を操作した量」に集計される。
    """
    ocfg = getattr(sim, "opinioncfg", None)
    if ocfg is not None and not ocfg.get("enabled", True):
        return
    old = listener.opinion
    new = opinion_mod.expose(listener, expressed_value, w)
    if abs(new - old) < 1e-4:
        return
    listener.opinion = new
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=listener.id,
                         kind="opinion_shift", x=listener.x, y=listener.y,
                         payload={"source": source_id, "old": round(old, 4),
                                  "new": round(new, 4)}))


# ---------------------------------------------------------------- Lynch 認知地図
def _familiar_places(sim, agent) -> list[str] | None:
    """よく知っている場所(visits 上位3の地名)。Lynch プラグイン ON 時のみ呼ばれる。

    engine は visits(訪問回数)と地名しか触らない=因子名は見ない。名前の無い路上は除く。
    """
    names: list[str] = []
    for node, _cnt in agent.visits.most_common(6):
        name = sim.city.node_name(node)
        if name and name != "路上" and name not in names:
            names.append(name)
        if len(names) >= 3:
            break
    return names or None


# ---------------------------------------------------------------- 予定(スケジュール帳)
def _schedule_on(sim) -> bool:
    """会話からの予定抽出・注入が有効か。既定 OFF=新経路を一切通さない(バイト一致)。"""
    scfg = getattr(sim, "schedulecfg", None)
    return bool(scfg and scfg["enabled"])


def _schedule_line(sim, agent, sim_min: int) -> str | None:
    """近い将来の予定1行(発火プロンプト用)。既定 OFF / 注入無効 / 予定なし=None(不変)。"""
    scfg = getattr(sim, "schedulecfg", None)
    if not (scfg and scfg["enabled"] and scfg["inject_prompt"]):
        return None
    return schedule.next_line(agent, today=sim_min // 1440,
                              horizon_days=scfg["horizon_days"])


def _record_appointments(sim, speaker, text: str, party_ids: list[int],
                         step: int, sim_min: int) -> None:
    """発話/DM テキストから未来の予定を決定論抽出し、話者と相手の帳簿へ記入する。

    追加 LLM 呼び出しゼロ(既に生成済みのテキストを正規表現で解析するだけ)。記入時に
    `appointment` を1件ログ(話者視点。with=相手 id)。抽出結果が空なら何もしない。
    """
    if not _schedule_on(sim):
        return
    scfg = sim.schedulecfg
    day = sim_min // 1440
    appts = schedule.extract(text, base_day=day, base_min=sim_min,
                             cal=getattr(sim, "calendarcfg", None),
                             horizon_days=scfg["horizon_days"],
                             place_hints=sim.envpackcfg["lexicon"]["place_hints"])
    if not appts:
        return
    hearers = [sim.agent_by_id[i] for i in party_ids if i in sim.agent_by_id]
    for appt in appts:
        others = sorted(set(party_ids))
        schedule.gc(speaker, day)
        schedule.record(speaker, {**appt, "with": others}, step=step,
                        max_items=scfg["max_items"])
        for h in hearers:                              # 相手側の with は自分以外(=話者+他相手)
            co = sorted(({speaker.id, *party_ids}) - {h.id})
            schedule.gc(h, day)
            schedule.record(h, {**appt, "with": co}, step=step,
                            max_items=scfg["max_items"])
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=speaker.id,
                             kind="appointment", x=speaker.x, y=speaker.y,
                             payload={"day": appt["day"], "when": appt["when"],
                                      "what": appt["what"], "place": appt["place"],
                                      "with": others}))


# ---------------------------------------------------------------- 会話強化(会話品質の構造改善)
# 2機構とも新 conf キー(prompts.dialog_history / prompts.reply_partner)・既定 OFF=ゴールデン完全維持。
# 設定は cfg.get 既定値で読む(schema 非依存)。乱数は一切引かない(全決定論・R1・新 stream なし)。
_DIALOG_MAX_TURNS = 4        # 相手あたりに保持する発話数(=直近2往復)
_DIALOG_MAX_PARTNERS = 8     # 全体で保持する相手数(LRU 上限)


def _dialog_on(sim) -> bool:
    """対話履歴の注入が有効か(prompts.dialog_history=true)。既定 OFF=状態も増やさない。"""
    try:
        return bool((sim.cfg.get("prompts", {}) or {}).get("dialog_history", False))
    except Exception:
        return False


def _reply_partner_mode(sim) -> str:
    """返答宛先の選び方(prompts.reply_partner)。既定 "nearest"=現行と完全同一の選択。"""
    try:
        return str((sim.cfg.get("prompts", {}) or {}).get("reply_partner", "nearest"))
    except Exception:
        return "nearest"


def _select_partner(sim, agent, hearers):
    """発話の返答権/宛先を決定論選択する(乱数なし・R1)。hearer 集合は不変=聞く人は
    変わらない。返す人(宛先)だけが変わる。

    "nearest"(既定): 最寄り1人(距離同点は id 小)=現行の min() と一字一句同一。
    "closeness": score = closeness(話者→候補)·10.0 − dist_m·0.1 の argmax(同点は id 昇順)。"""
    if not hearers:
        return None
    if _reply_partner_mode(sim) == "closeness":
        rels = agent.mem.relations

        def _score(h):
            c = float((rels.get(h.id) or {}).get("closeness", 0.0))
            dist = math.hypot(h.x - agent.x, h.y - agent.y)
            return (c * 10.0 - dist * 0.1, -h.id)   # 同点は id 最小(-id が最大)を選ぶ

        return max(hearers, key=_score)
    # 既定 nearest: 現行の min() と一字一句同一(ゴールデン維持)。
    return min(hearers, key=lambda h: (
        math.hypot(h.x - agent.x, h.y - agent.y), h.id))


def _dialog_push(agent, partner_id: int, speaker_name: str, text: str) -> None:
    """相手別リングバッファへ1発話を追記(ON 時のみ呼ばれる)。相手あたり最大4発話・
    全体最大8相手 LRU(最古の相手を退避)。バッファは動的属性 _dialog_hist に持つ
    ので、OFF では一度も生成されない=状態変化なし=構造的にバイト一致。"""
    hist = getattr(agent, "_dialog_hist", None)
    if hist is None:
        hist = {}
        agent._dialog_hist = hist
    turns = hist.pop(partner_id, None)          # pop+再挿入で最新利用へ更新(LRU)
    if turns is None:
        turns = []
    turns.append((speaker_name, text))
    if len(turns) > _DIALOG_MAX_TURNS:
        turns = turns[-_DIALOG_MAX_TURNS:]
    hist[partner_id] = turns
    while len(hist) > _DIALOG_MAX_PARTNERS:
        del hist[next(iter(hist))]              # 挿入順先頭=LRU 最古の相手を退避


def _dialog_get(agent, partner_id) -> list | None:
    """相手 partner_id との直近対話(最大4発話=2往復)を返す。無ければ None。"""
    if partner_id is None:
        return None
    hist = getattr(agent, "_dialog_hist", None)
    if not hist:
        return None
    turns = hist.get(partner_id)
    return list(turns) if turns else None


# ---------------------------------------------------------------- LLM 発話
def _llm_speak(sim, agent, trigger: str, step: int, sim_min: int, *,
               dm_target: str | None = None,
               feed_texts: list[str] | None = None,
               reply_to: tuple[str, str] | None = None,
               partner_id: int | None = None) -> dict | None:
    """LLM に発話/行動を生成させる。解釈不能なら None(沈黙)。
    予算は欲求フェーズ(_phase_drive)で消費済み(発火権を得た者だけが来る)。"""
    radius = float(sim.cfg.world.perception_radius_m)
    company = hearers_of(agent, _percept(sim), radius)
    if agent.building:
        pois = sim.city.pois_in_building(agent.building, agent.floor)
    else:
        pois = sim.city.pois_at_node(agent.node)
    # 生活の自己決定 P2(D3・既定 全 OFF は None=1行も足さない=バイト一致)。中立提示・客観条件つき。
    p2_offers = _p2_offers(sim, agent, trigger, pois, company)
    tool_offers = memberships = None
    tools = getattr(sim, "tools", None)
    if tools is not None and trigger in ("solo", "social", "post"):
        tool_offers = tools.offer_text(sim, agent)      # 中立提示(R1: k と無関係)
        memberships = tools.membership_names(agent)
    # 標準装備(equip_all): 全発火プロンプトに「所持ツール」節を中立告知(既定 OFF=不変)。
    # 呼数は変えずプロンプト内容のみ。条件表示は客観(所持金)のみで k 非依存(R1)。
    tcfg = tools.cfg if tools is not None else {}
    equip_all = bool(tcfg.get("equip_all", False))
    equip_venture_cost = float(tcfg.get("venture_cost", 30000.0))
    # 行動心理プラグイン(既定 OFF): OFF 時は None のままでプロンプトに1行も足さない=不変。
    familiar_places = institutions = None
    psychcfg = sim.psychcfg
    if psychcfg["lynch"]["enabled"]:                    # Lynch: 馴染みの場所 上位3を1行注入
        familiar_places = _familiar_places(sim, agent)
    if psychcfg["searle"]["enabled"] and tools is not None:  # Searle: この街の取り決め(全員平等)
        institutions = tools.institution_lines()
    place = _place_of(sim, agent)
    # agentic pull(発火時): fire_reason+場所文字列で決定論想起を1行注入(LLM は増やさない)
    pull_query = f"{trigger} {place}" if getattr(sim, "agentic_pull", False) else None
    schedule_line = _schedule_line(sim, agent, sim_min)   # 近い予定1行(既定 None=不変)
    # 社会関係の質(Wave G2): tier/派閥つきの間柄行 + 評判行(既定 OFF は None=従来の
    # relation_line に後退=バイト一致)。決定論・LLM 非増。
    relation_line = reputation_line = None
    if _relations_on(sim):
        relation_line, reputation_line = relations_mod.social_lines(
            agent, [a.id for a in company], sim.relationscfg, tools=tools)
    # 世帯・恋愛(後続波 H2): 同席する同居者・恋人の間柄行(既定 OFF は None=1行も足さない
    # =バイト一致)。決定論・LLM 非増(名簿と household 状態のみ参照)。
    household_line = None
    if _household_on(sim):
        household_line = household_mod.context_line(
            agent, [a.id for a in company], sim.agent_by_id)
    # 観光・多言語(後続波 H5): 観光客/非日本語話者の文脈を1行注入(既定 OFF は None=1行も足さない
    # =バイト一致)。決定論・LLM 非増(agent の名簿属性のみ参照=k 非依存)。
    diversity_line = (diversity_mod.context_line(agent, getattr(sim, "place_name", ""))
                      if _diversity_on(sim) else None)
    # 内面本格版(後続波 H6): 離散感情・長期目標・趣味の文脈を1行ずつ注入(既定 OFF は None=1行も
    # 足さない=バイト一致)。決定論・LLM 非増(注入は内容のみ=呼数不変=R1 の ON==OFF)。感情は
    # affect ON 前提(arousal が動かないと無効)。目標/趣味は precompute 済みの決定論値を読むだけ。
    emotion_line = goal_line = hobby_line = None
    if inner_life_mod.enabled(sim):
        icfg = sim.innerlifecfg
        emotion_line = inner_life_mod.emotion_line(agent, icfg,
                                                   affect_on=_affect_on(sim))
        goal_line = inner_life_mod.goal_line(agent, icfg)
        hobby_line = inner_life_mod.hobby_line(agent, icfg)
    # 再帰性(第9バッチ・既定 OFF): いま実効の取り決め+昨日の街の動き(客観カウント)を
    # 1行ずつ注入(全員平等・k 非依存・内容のみ=呼数不変)。OFF は None=1行も足さない=バイト一致。
    norm_line = digest_line = None
    recur = getattr(sim, "recursion", None)
    if recur is not None:
        norm_line = recur.norm_line(sim)
        digest_line = recur.digest_line()
    # 発話の定型化ガード(改善 P3 第9バッチ・既定 OFF): 情景報告の決まり文句を避ける注意書きを
    # social/reply/solo の状況文に足す(内容のみ=呼数不変=R1)。OFF=一字も足さない=バイト一致。
    variety_hint = bool(getattr(sim, "promptscfg", {}).get("variety_hint", False))
    # 街路の環境情報(第18バッチ・既定 OFF は None=1行も足さない=バイト一致)。広告=想起窓内の
    # 直近接触1行(中立提示)、群衆視覚=同席者の決定論要約(記述的規範)。呼数不変=R1。
    ads_line = street_mod.ads_line(agent, getattr(sim, "adscfg", None), step)
    crowd_line = street_mod.crowd_line(company, getattr(sim, "crowdcfg", None))
    # 主観的世界モデル(第20バッチ・既定 OFF は None=不変)。期待/可制御性/規範予期の
    # 自然文(閾値を超えたときだけ=トークン節約)。決定論・呼数不変=R1。
    wvcfg = getattr(sim, "worldviewcfg", None)
    wv_expect_line = worldview_mod.expect_line(sim, agent, sim_min)
    wv_self_line = worldview_mod.ctrl_line(agent, wvcfg)
    wv_norm_line = worldview_mod.norm_line(sim, wvcfg)
    # 対話履歴の注入(会話強化・prompts.dialog_history=true のみ)。相手=reply は話者
    # (partner_id)、social は同席者からの決定論選択。OFF は None=1行も足さない=バイト一致。
    # 呼数不変=R1(注入はプロンプト内容のみ=キャッシュキーは変わってよい)。
    dialog_history = None
    if _dialog_on(sim) and trigger in ("social", "reply"):
        pid = partner_id
        if pid is None and trigger == "social":
            p = _select_partner(sim, agent, company)
            pid = p.id if p is not None else None
        dialog_history = _dialog_get(agent, pid)
    prompt = deliberate.build_prompt(agent, place_name=place,
                                     surprise=trigger,
                                     nearby_names=[a.name for a in company],
                                     nearby_ids=[a.id for a in company],
                                     sim_min=sim_min, step=step,
                                     nearby_pois=[p["name"] for p in pois],
                                     dm_target=dm_target, feed_texts=feed_texts,
                                     reply_to=reply_to, tool_offers=tool_offers,
                                     memberships=memberships, pull_query=pull_query,
                                     familiar_places=familiar_places,
                                     institutions=institutions,
                                     equip_all=equip_all,
                                     venture_cost=equip_venture_cost,
                                     city_name=getattr(sim, "place_name", ""),
                                     date_line=getattr(sim, "today_date_line", None),
                                     weather_line=getattr(sim, "today_weather_line", None),
                                     schedule_line=schedule_line,
                                     event_line=getattr(sim, "today_event_line", None),
                                     disaster_line=getattr(sim, "today_disaster_line", None),
                                     ads_line=ads_line,
                                     crowd_line=crowd_line,
                                     wv_expect_line=wv_expect_line,
                                     wv_self_line=wv_self_line,
                                     wv_norm_line=wv_norm_line,
                                     relation_line=relation_line,
                                     reputation_line=reputation_line,
                                     household_line=household_line,
                                     diversity_line=diversity_line,
                                     emotion_line=emotion_line,
                                     goal_line=goal_line,
                                     hobby_line=hobby_line,
                                     norm_line=norm_line,
                                     digest_line=digest_line,
                                     variety_hint=variety_hint,
                                     labeling_mode=sim.labels.mode,
                                     open_actions=bool(getattr(sim, "freedomcfg", None)
                                                       and sim.freedomcfg["open_actions"]),
                                     p2_offers=p2_offers,
                                     dialog_history=dialog_history)
    rng_key = f"deliberate/{agent.id}/{step}"
    response, call_id, cached = sim.llm.generate(
        prompt, rng_key=rng_key,
        temperature=float(sim.cfg.model.temperature),
        max_tokens=int(sim.cfg.model.max_tokens))
    sim.logger.log_llm_call({"llm_call_id": call_id, "agent_id": agent.id,
                             "purpose": trigger, "step": step, "cached": cached})
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                         kind="llm_deliberate", x=agent.x, y=agent.y,
                         llm_call_id=call_id, payload={"trigger": trigger}))
    action = deliberate.parse_action(response)
    if action is None:                             # D16: 壊れたら沈黙して続行
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                             kind="fallback", x=agent.x, y=agent.y,
                             payload={"reason": "parse_error"}))
    return action


# ---------------------------------------------------------------- 対照: null 系列
# 内容非結合の固定ダミー文(D7 null 系列)。呼び出し量だけを増やし結合はゼロにする。
_NULL_PROMPT = ("次の一般常識の問いに一言で答えてください。1年は何か月ですか。")


def _null_calls(sim, agent, step: int, sim_min: int) -> None:
    """発火権付与の直後に、内容非結合のダミー LLM を固定本数だけ呼ぶ(出力は即破棄)。

    呼数は k と無関係の固定数(R1)。rng_key は agent/step/i で分離、新 stream 名 'null'。
    """
    for i in range(int(sim.null_calls)):
        _resp, call_id, cached = sim.llm.generate(
            _NULL_PROMPT, rng_key=f"null/{agent.id}/{step}/{i}",
            temperature=float(sim.cfg.model.temperature),
            max_tokens=int(sim.cfg.model.max_tokens))
        sim.logger.log_llm_call({"llm_call_id": call_id, "agent_id": agent.id,
                                 "purpose": "null", "step": step, "cached": cached})
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                             kind="llm_null", x=agent.x, y=agent.y,
                             llm_call_id=call_id, payload={"i": i}))


# ---------------------------------------------------------------- 欲求フェーズ
def _phase_drive(sim, step: int, sim_min: int) -> None:
    """欲求ゲージの減衰→申請→個人重みの抽選→予算。発火権を配る(Phase A)。

    申請順は欲求残量順(id順バイアスなし)。抽選落ち=数十%減衰して再蓄積、
    抽選成功だが予算切れ=ゲージ維持で翌stepへ持ち越し(ユーザー仕様 2026-07-04)。
    """
    cfg = sim.drivecfg
    radius = float(sim.cfg.world.perception_radius_m)
    stats = {"requests": 0, "fires": 0, "face_fires": 0, "replies": 0}
    # 勾留中(detained_until。既定 0=全員 step>=0 で常に通過=不変)は欲求発火の対象外
    active = [a for a in sim.agents if a.loc != "outside" and not a.sleeping
              and step >= getattr(a, "detained_until", 0)]
    affect_on = _affect_on(sim)
    health_on = _health_on(sim)
    # 離散感情ラベル(内面 H6): affect ON 前提で core affect(mood_valence × arousal)から離散ラベルを
    # 決定論写像し、変化時のみ emotion_label を記録(sparse)。affect OFF / inner_life OFF / emotion 無効
    # なら完全 no-op(ラベル更新なし・イベント 0 件=バイト一致)。drive(発火系)には接続しない(R1)。
    emotion_on = affect_on and inner_life_mod.enabled(sim) \
        and inner_life_mod.cfg_of(sim)["emotion"]["enabled"]
    for agent in active:
        drive.step_tick(agent, cfg, step)
        if affect_on:                       # 覚醒度を毎step baseline へ漏れ減衰(drive と並列)
            affect.decay(agent, sim.affectcfg)
            if emotion_on:                  # core affect → 離散感情ラベル(決定論・drive 非接続)
                inner_life_mod.update_emotion(agent, sim.innerlifecfg, step,
                                              sim_min, sim.logger)

    def _eff_thr(a):
        """実効閾値(E2 ドリフト + 覚醒の逆U字変調 + 疲労)。affect/health OFF or gain=0 で恒等=バイト一致。"""
        d = affect.threshold_delta(a, sim.affectcfg) if affect_on else 0.0
        if health_on:                       # 高疲労→閾値↑(休息へ寄る)。gain=0 で 0=恒等
            d += health_mod.fatigue_threshold_delta(a, sim.healthcfg)
        return drive.effective_threshold(a, d)

    # 発火の関数形(B段 seam)。fixed=閾値ゲート+個人重み抽選(現行)。
    # logistic=閾値を soft 化し p=σ(slope·(drive−threshold)) に一本化。
    logistic = (cfg.get("firing", "fixed") == "logistic")
    null_series = (getattr(sim, "controls_mode", "none") == "null_series")
    if logistic:
        requesters = [a for a in active if step >= a.refractory_until]
    else:
        requesters = [a for a in active
                      if a.drive >= _eff_thr(a)
                      and step >= a.refractory_until]
    requesters.sort(key=lambda a: (-a.drive, a.id))
    for agent in requesters:
        stats["requests"] += 1
        req_drive = agent.drive
        reason = drive.top_reason(agent)
        rng = sim.hub.stream("drive", agent.id, step)
        # 対面会話: 同席者がいて会話クールダウン外なら抽選なしで確定発火(logistic でも不変)。
        face = bool(hearers_of(agent, _percept(sim), radius)) \
            and step >= agent.conv_cooldown_until
        granted = False
        if face:
            lottery = None
            if sim.budget.take():                  # 予算切れ=ゲージ維持で持ち越し
                granted = True
                agent._fire_reason = "social_face"
                drive.on_fire(agent, step, cfg)
                stats["fires"] += 1
                stats["face_fires"] += 1
        else:                                      # 媒体/独り言: 抽選
            if logistic:
                p = 1.0 / (1.0 + math.exp(
                    -cfg["slope"] * (agent.drive - _eff_thr(agent))))
            else:
                p = agent.fire_weight
            lottery = bool(rng.random() < p)
            if lottery and sim.budget.take():
                granted = True
                agent._fire_reason = reason
                drive.on_fire(agent, step, cfg)
                stats["fires"] += 1
            elif not lottery:
                drive.on_reject(agent, cfg)
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                             kind="drive_request", x=agent.x, y=agent.y,
                             payload={"drive": round(req_drive, 3),
                                      "threshold": round(_eff_thr(agent), 3),
                                      "mode": "face" if face else "media",
                                      "lottery": lottery, "granted": granted,
                                      "reason": reason}))
        if granted and null_series:                # 対照: 発火に紐付けたダミー呼び出し
            _null_calls(sim, agent, step, sim_min)
    sim.drive_stats = stats


def _feed_texts(sim, agent, step: int, sim_min: int) -> list[str]:
    """タイムラインを「@著者名: 本文」形式で見せる(場所・時刻の宣言化を防ぐ)。

    推薦(Wave G6)ON なら意見整合で選別された TL を見せる(infoenv.timeline)。OFF は従来の
    時系列 TL(sim.net.timeline_for をそのまま呼ぶ=バイト一致)。"""
    out = []
    for p in infoenv_mod.timeline(sim, agent, step, sim_min):
        meta = sim.agent_by_id.get(p["author"])
        who = "公式" if p["author"] == -1 or meta is None else meta.name
        out.append(f"@{who}: {p['text']}")
    return out


def _fire_llm(sim, agent, reason: str, step: int, sim_min: int,
              rng) -> dict | None:
    """発火権を得た思考の文脈選択 → LLM。きっかけ(reason)で表現形を選ぶ。

    同席時の対面会話は _phase_drive で確定発火(reason="social_face")済み。
    """
    if reason == "social_face":
        return _llm_speak(sim, agent, "social", step, sim_min)
    if reason in ("novel_place", "congestion", "unknown_word"):
        return _llm_speak(sim, agent, reason, step, sim_min)
    if reason == "dm_received" and agent._last_dm_from is not None:
        target = sim.agent_by_id.get(agent._last_dm_from)
        if target is not None:
            action = _llm_speak(sim, agent, "dm", step, sim_min,
                                dm_target=target.name)
            if action is not None and action.get("type") == "dm":
                action["to"] = target.id
            return action
    if reason == "news":
        return _llm_speak(sim, agent, "post", step, sim_min,
                          feed_texts=_feed_texts(sim, agent, step, sim_min))
    if agent.has_phone and rng.random() < 0.45:
        contacts = sorted(sim.net.contacts.get(agent.id, ()))
        if contacts and rng.random() < 0.35:
            target_id = contacts[int(rng.integers(len(contacts)))]
            meta = sim.agent_by_id.get(target_id)
            if meta is not None:
                action = _llm_speak(sim, agent, "dm", step, sim_min,
                                    dm_target=meta.name)
                if action is not None and action.get("type") == "dm":
                    action["to"] = target_id
                return action
        return _llm_speak(sim, agent, "post", step, sim_min,
                          feed_texts=_feed_texts(sim, agent, step, sim_min))
    return _llm_speak(sim, agent, "solo", step, sim_min)


# ---------------------------------------------------------------- 意思決定
def _decide(sim, agent, step: int, sim_min: int) -> dict:
    # 勾留(制度深化2・既定 0=フラグ立たず不変): 拘束中は行動しない(返答・発火・移動なし)。
    if step < getattr(agent, "detained_until", 0):
        return {"type": "stay"}
    rng = sim.hub.stream("decide", agent.id, step)
    radius = float(sim.cfg.world.perception_radius_m)
    company = hearers_of(agent, _percept(sim), radius)
    if company:
        drive.add(agent, "company", sim.drivecfg)
    agent._heard_unknown = False

    # 返答保証: 話しかけられたら抽選なしで必ずLLMで返答(予算があれば)。
    if agent._reply_to is not None:
        speaker_id, said = agent._reply_to
        agent._reply_to = None
        if sim.budget.take():
            speaker = sim.agent_by_id.get(speaker_id)
            reply_to = (speaker.name, said) if speaker is not None else None
            action = _llm_speak(sim, agent, "reply", step, sim_min,
                                reply_to=reply_to, partner_id=speaker_id)
            agent.conv_turns_left -= 1
            if agent.conv_turns_left <= 0:         # セッション上限 → クールダウン
                agent.conv_cooldown_until = step + sim.drivecfg["conv_cooldown_steps"]
            sim.drive_stats["replies"] = sim.drive_stats.get("replies", 0) + 1
            if action is not None:
                return action

    if agent._fire_reason:                         # 欲求フェーズで発火権を得た
        reason = agent._fire_reason
        agent._fire_reason = ""
        agent._last_fire_reason = reason           # 造語の発生きっかけの記録用
        action = _fire_llm(sim, agent, reason, step, sim_min, rng)
        if action is not None:
            return action

    action = routine.decide(agent, step, sim, _place_of(sim, agent), rng,
                            has_company=bool(company))
    if action["type"] == "stay" and agent.has_phone and not agent.sleeping:
        phone_action = _phone(sim, agent, step, sim_min)
        if phone_action is not None:
            return phone_action
    return action


# ---------------------------------------------------------------- 検索エンジン
def _search_index(sim, query: str) -> list[str]:
    """シミュ内検索エンジン(世界内データベース)。実APIは使わない(D13 再現性
    +架空世界の閉性)。索引 = 語彙の来歴 / ニュース / 実在 POI(OSM)。"""
    results: list[str] = []
    item = sim.labels.text_to_item.get(query)
    if item is not None:
        src = "メディア発表" if item.creator == -1 else \
            f"{sim.agent_by_id[item.creator].name}が言い始めた"
        results.append(f"「{query}」: 最近使われ始めた言葉({src}、"
                       f"{len(item.transmissions)}回拡散)")
    for article in reversed(sim.net.news[-8:]):
        if query in article["title"] or query in article["text"]:
            results.append(f"ニュース: {article['title']} — {article['text'][:40]}")
    n_poi = 0
    for poi in sim.city.poi_list:
        if query in poi["name"] and n_poi < 3:
            results.append(f"地図: {poi['name']}({poi['cat']})")
            n_poi += 1
    for post in reversed(sim.net.posts[-30:]):
        if len(results) >= 5:
            break
        if query in post["text"]:
            who = "公式" if post["author"] == -1 else \
                sim.agent_by_id[post["author"]].name
            results.append(f"SNS: {who}の投稿「{post['text'][:36]}」")
    return results[:5]


# ---------------------------------------------------------------- SNS 反応
def _sns_react(sim, agent, post: dict, kind: str, log_kind: str,
               step: int, sim_min: int) -> None:
    """いいね/リシェアを net に反映し、著者に drive.add("sns")。states は触らない。

    著者が sleeping/outside でも drive.add は安全(ゲージのみ更新)。著者=メディア(-1)
    や不在 id はスキップ。リシェアは "RT @元著者名: 本文" を net が新規投稿として追記し、
    フォロワーへの再配信は既存 timeline 配信+_hear_words("sns")に自動で乗る。
    """
    author_meta = sim.agent_by_id.get(post["author"])
    author_name = "公式" if post["author"] == -1 or author_meta is None \
        else author_meta.name
    author = sim.net.react(agent.id, post["id"], kind, step,
                           author_name=author_name)
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                         kind=log_kind, x=agent.x, y=agent.y,
                         payload={"post_id": post["id"], "author": author}))
    if author_meta is not None:                    # メディア/不在はスキップ(states 不変)
        drive.add(author_meta, "sns", sim.drivecfg)


# ---------------------------------------------------------------- スマホ
def _phone(sim, agent, step: int, sim_min: int) -> dict | None:
    """暇なときのスマホ行動: 検索 / SNS閲覧 / ニュース / 投稿(LLM) / DM(LLM)。"""
    ncfg = sim.netcfg
    if not ncfg["enabled"]:
        return None
    # インフラ障害(通信断・停電=H4、既定 OFF は infra_out=False で不変): スマホ行動(検索/閲覧/
    # ニュース/投稿/DM)を抑制する(通信断=生活麻痺)。障害が無ければ従来どおり(バイト一致)。
    if disaster_mod.infra_out(sim):
        return None
    rng = sim.hub.stream("phone", agent.id, step)
    affect_on = _affect_on(sim)

    # 知らない言葉を聞いていたら、まず検索して調べる
    if agent._search_queue:
        word = agent._search_queue.pop(0)
        results = _search_index(sim, word)
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                             kind="search", x=agent.x, y=agent.y,
                             payload={"query": word, "results": results}))
        _hear_words(sim, agent, [word], -1, "search", step, sim_min)
        top = results[0] if results else "検索してもよくわからなかった"
        agent.remember(f"「{word}」を検索: {top}")
        return {"type": "stay"}

    # 閲覧系のみ(非LLM)。発信(post/dm)は欲求発火(_fire_llm)経由に一本化。
    r = rng.random()
    p_browse = float(ncfg["browse_prob"])
    p_news = p_browse + float(ncfg["news_prob"])

    if r < p_browse:                               # SNS タイムライン閲覧
        feed = infoenv_mod.timeline(sim, agent, step, sim_min)  # 推薦 ON=意見整合で選別
        if feed:
            tools = getattr(sim, "tools", None)
            # いいね/リシェア判定は新stream(既存 phone stream の draw 順に影響しない)
            react_rng = sim.hub.stream("sns_react", agent.id, step)
            like_prob = float(ncfg.get("like_prob", 0.15))
            reshare_prob = float(ncfg.get("reshare_prob", 0.03))
            for post in feed:
                pv = valence(post["text"])
                _hear_words(sim, agent, post["items"], post["author"],
                            "sns", step, sim_min)
                factor_update.on_heard_valence(
                    agent, pv, mags=sim.mags, step=step,
                    sim_min=sim_min, logger=sim.logger, scale=0.5)
                _arouse(sim, agent, "sns", step, sim_min,
                        valence_abs=abs(pv))       # TL 投稿の情動強度→覚醒↑(OFF=no-op)
                # SNS 閲覧での意見更新(FJ、w=w_sns)。source=投稿の著者。
                _shift_opinion(sim, agent, pv, post["author"],
                               sim.opinioncfg["w_sns"], step, sim_min)
                eid = post.get("event_id")             # イベント告知の閲覧 → 参加告知
                if eid is not None and tools is not None:
                    tools.invite(agent, eid)
                # いいね・リシェア(#14): 状態更新は net、著者には drive.add("sns")のみ
                like = react_rng.random() < like_prob
                reshare = react_rng.random() < reshare_prob
                if like:
                    _sns_react(sim, agent, post, "like", "sns_like", step, sim_min)
                if reshare:
                    _sns_react(sim, agent, post, "reshare", "sns_reshare",
                               step, sim_min)
            if affect_on and sim.affectcfg["salience_k"] > 0:
                # 注意の容量制約(Cowan): 同時知覚の TL 投稿を salience 上位 K だけ記憶へ符号化する。
                # 反応(意見/いいね/著者への drive)は全 item に効いたまま=発火・呼数は不変(R1)。
                # 入力解像度LOD(第30バッチ): ゲートが有効なときだけ K を個体別に上書き
                # (salience_k=0 の水準は現行のグローバル値のまま=OFF/mid はバイト一致)。
                _irk = int((getattr(agent, "input_res", None) or {})
                           .get("salience_k", 0))
                _sk = _irk if _irk > 0 else sim.affectcfg["salience_k"]
                scores = [affect.salience_score(agent, sim.affectcfg,
                                                valence_abs=valence(p["text"]))
                          for p in feed]
                for p in salience_gate(feed, scores, _sk):
                    agent.remember(f"SNSで見た: 「{p['text'][:24]}」", kind="sns",
                                   importance_bonus=_imp_bonus(sim, agent))
            else:
                agent.remember(f"SNSで見た: 「{feed[-1]['text'][:24]}」", kind="sns")
            drive.add(agent, "sns", sim.drivecfg)
            sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                                 kind="sns_read", x=agent.x, y=agent.y,
                                 payload={"n_posts": len(feed),
                                          "authors": [p["author"] for p in feed]}))
        return {"type": "stay"}
    if r < p_news:                                 # ニュースアプリ
        articles = sim.net.latest_news()
        if articles:
            for a in articles:
                _hear_words(sim, agent, a["items"], -1, "news", step, sim_min)
                av = valence(a["title"] + a["text"])
                factor_update.on_heard_valence(
                    agent, av, mags=sim.mags,
                    step=step, sim_min=sim_min, logger=sim.logger, scale=0.7)
                _arouse(sim, agent, "news", step, sim_min,
                        valence_abs=abs(av))       # 驚き(大 |valence| news)→覚醒↑(OFF=no-op)
            if affect_on and sim.affectcfg["salience_k"] > 0:
                # 入力解像度LOD: feed 側と同じ個体別 K の上書き(ゲート有効時のみ)
                _irk = int((getattr(agent, "input_res", None) or {})
                           .get("salience_k", 0))
                _sk = _irk if _irk > 0 else sim.affectcfg["salience_k"]
                scores = [affect.salience_score(agent, sim.affectcfg,
                                                valence_abs=valence(a["title"] + a["text"]))
                          for a in articles]
                for a in salience_gate(articles, scores, _sk):
                    agent.remember(f"ニュースで見た: 「{a['title']}」", kind="news",
                                   importance_bonus=_imp_bonus(sim, agent))
            else:
                agent.remember(f"ニュースで見た: 「{articles[-1]['title']}」", kind="news")
            drive.add(agent, "news", sim.drivecfg)
            sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                                 kind="news_read", x=agent.x, y=agent.y,
                                 payload={"titles": [a["title"] for a in articles]}))
        return {"type": "stay"}
    return None


# ---------------------------------------------------------------- 行動適用
def _apply(sim, agent, action: dict, step: int, sim_min: int) -> None:
    kind = action["type"]

    if kind in ("host_event", "post_flyer", "found_group", "propose",
                "open_venture"):
        tools = getattr(sim, "tools", None)
        if tools is not None:
            tools.apply(sim, agent, action, step, sim_min)
        return

    if kind == "free_action":
        # 開放行動(第17バッチ)。freedom OFF や解釈不能では静かに無視(=wander 相当)。
        fcfg = getattr(sim, "freedomcfg", None)
        if fcfg is not None and fcfg["open_actions"]:
            _apply_free_action(sim, agent, action, step, sim_min)
        return

    if kind in ("move_home", "buy", "study", "propose_partnership", "break_up"):
        # 生活の自己決定 P2(D3)。該当項目が OFF や解釈不能では静かに無視(=wander 相当)。
        _apply_p2(sim, agent, action, step, sim_min)
        return

    if kind == "go_to_bed":
        # 自宅前 → 実在の住宅建物に入って就寝(v3: 路上睡眠の解消)。
        # 立退き中(制度深化3・既定 False=不変)は自宅建物に入れない=路上就寝。
        if (agent.home_building and not agent.evicted
                and sim.city.has_building(agent.home_building)):
            bld = sim.city.building(agent.home_building)
            agent.building = bld["id"]
            agent.floor = agent.home_floor
            agent.x, agent.y = bld["centroid"]
            sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                                 kind="enter_building", x=agent.x, y=agent.y,
                                 payload={"building": bld["id"],
                                          "name": bld["name"] or "自宅",
                                          "floor": agent.floor,
                                          "levels": int(bld["levels"]),
                                          "home": True}))
        kind = "sleep"

    if kind == "sleep":
        agent.sleeping = True
        agent.activity = ""
        agent.sleep_until = step + agent.sleep_steps
        agent.reflect_step = step + 1              # 就寝直後に記憶整理(内省)
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                             kind="sleep_start", x=agent.x, y=agent.y,
                             payload={"until_step": agent.sleep_until,
                                      "building": agent.building}))
        return

    if kind == "move_to":
        path, used_mode = sim.router.route(agent.node, action["dest"],
                                           action.get("mode", "walk"))
        if len(path) < 2:
            return
        agent.route = path[1:]
        agent.edge_offset = 0.0
        agent.dest = action["dest"]
        agent.trip_mode = used_mode
        agent.exit_intent = bool(action.get("exit"))
        agent.homing = bool(action.get("homing"))
        agent._pending_stay = int(action.get("stay_steps", 2))
        agent._pending_activity = str(action.get("activity", ""))
        agent._ride_pending = action.get("ride")   # 交通機関の乗車(到着時に課金)。無ければ None
        agent.activity = "commuting" if action.get("activity") == "commuting" else ""
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                             kind="route_start", x=agent.x, y=agent.y,
                             payload={"dest": action["dest"],
                                      "dest_name": sim.city.node_name(action["dest"]),
                                      "mode": used_mode,
                                      "exit": agent.exit_intent,
                                      "homing": agent.homing,
                                      "n_nodes": len(path),
                                      "dist_m": round(sim.router.route_length(path), 1)}))
        return

    if kind == "enter_building":
        bld = sim.city.building(action["building"])
        rng = sim.hub.stream("floor", agent.id, step)
        agent.building = bld["id"]
        if action.get("floor"):
            agent.floor = int(action["floor"])
        else:
            agent.floor = int(rng.integers(1, int(bld["levels"]) + 1))
        agent.stay_until = step + int(action.get("stay_steps", 3))
        activity = action.get("activity")
        if not activity:                           # 建物の中身(POI)から推定
            cats = {p["cat"] for p in sim.city.pois_in_building(bld["id"])}
            activity = ("eating" if "food" in cats or "nightlife" in cats
                        else "shopping" if "shop" in cats else "")
        agent.activity = activity
        # 建物内の消費(飲食・買い物)。払える範囲でのみ購入(経済 v0)
        if _economy_on(sim) and activity in ("eating", "shopping"):
            in_cats = {p["cat"] for p in sim.city.pois_in_building(bld["id"])}
            cat = ("shop" if activity == "shopping"
                   else "food" if "food" in in_cats else "nightlife")
            price = price_of(cat, sim.economy, getattr(sim, "rulebook", None))
            if _commerce_on(sim):                  # 動的価格/在庫を消費額に反映(既存 draw の外・決定論)
                price = commerce_mod.on_purchase(sim, agent, cat, price, step, sim_min)
            if price is not None and agent.money >= price:   # 品切れ(None)= 購入抑制
                _spend(sim, agent, price, cat, step, sim_min)
        cx, cy = bld["centroid"]
        agent.x = cx + float(rng.uniform(-8, 8))
        agent.y = cy + float(rng.uniform(-8, 8))
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                             kind="enter_building", x=agent.x, y=agent.y,
                             payload={"building": bld["id"], "name": bld["name"],
                                      "floor": agent.floor,
                                      "levels": int(bld["levels"]),
                                      "activity": activity}))
        return

    if kind == "exit_building" and agent.building:
        bld = sim.city.building(agent.building)
        # 勤務・バイトの完遂 = 達成経験(Bandura mastery)+ 賃金(経済 v0)
        _settle_work(sim, agent, step, sim_min)
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                             kind="exit_building", x=agent.x, y=agent.y,
                             payload={"building": bld["id"]}))
        agent.building = None
        agent.floor = 0
        agent.activity = ""
        agent.node = bld["entrance"]
        agent.x, agent.y = sim.city.node_xy(agent.node)
        agent.stay_until = step + 1
        return

    if kind == "floor_move" and agent.building:
        agent.floor = int(action["floor"])
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                             kind="floor_move", x=agent.x, y=agent.y,
                             payload={"building": agent.building,
                                      "floor": agent.floor}))
        return

    if kind == "post":                             # SNS 投稿
        words = [w for w in action.get("use_items", []) if w in agent.adopted]
        if not words:                              # 本文中の既知語も items 扱い
            words = [w for w in sorted(agent.adopted) if w in action["text"]]
        sim.net.post(agent.id, action["text"], words, step)
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                             kind="sns_post", x=agent.x, y=agent.y,
                             payload={"text": action["text"], "items": words}))
        for word in words:
            sim.labels.use(agent, word, step=step, sim_min=sim_min, logger=sim.logger)
        agent.remember(f"SNSに投稿した:「{action['text']}」")
        agent.said.append(f"(SNS)「{action['text']}」")
        agent.said = agent.said[-4:]
        return

    if kind == "dm":                               # 1対1メッセージ
        to = action.get("to")
        recipient = sim.agent_by_id.get(to) if to is not None else None
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                             kind="dm", x=agent.x, y=agent.y,
                             payload={"to": to, "text": action["text"]}))
        # 会話(DM)からの予定抽出→双方の帳簿へ記入(既定 OFF=no-op)。to にも入れる。
        _record_appointments(sim, agent, action["text"],
                             [to] if to is not None else [], step, sim_min)
        if recipient is not None and recipient.loc != "outside" \
                and not recipient.sleeping:
            recipient.remember(f"{agent.name}からメッセージ:「{action['text']}」",
                               kind="dm")
            recipient._last_dm_from = agent.id
            v_dm = valence(action["text"])
            # 関係台帳(Wave G2 の交流符号=DM の感情価)。OFF は従来の record_contact と完全同一。
            _contact(sim, recipient, agent.id, agent.name, action["text"],
                     v_dm, step, sim_min)
            _contact(sim, agent, recipient.id, recipient.name, "",
                     v_dm, step, sim_min)
            drive.add(recipient, "dm_received", sim.drivecfg)
            _arouse(sim, recipient, "dm", step, sim_min,
                    valence_abs=abs(v_dm), addressed=1.0)   # DM=被話しかけ→覚醒↑(OFF=no-op)
            d_v = factor_update.on_heard_valence(recipient, v_dm,
                                                 mags=sim.mags, step=step,
                                                 sim_min=sim_min, logger=sim.logger)
            if d_v:
                drive.add(recipient, "state_change", sim.drivecfg, scale=abs(d_v))
            words = [w for w in sorted(agent.adopted) if w in action["text"]]
            _hear_words(sim, recipient, words, agent.id, "dm", step, sim_min)
            # DM での意見更新(FJ、w=w_dm)。source=送信者。
            _shift_opinion(sim, recipient, v_dm, agent.id,
                           sim.opinioncfg["w_dm"], step, sim_min)
        agent.said.append(f"(DM)「{action['text']}」")
        agent.said = agent.said[-4:]
        return

    if kind == "coin_label":
        # 造語の「発生過程・きっかけ」を自然観察する(促進はしない)。文脈を coin に渡す。
        radius = float(sim.cfg.world.perception_radius_m)
        company = hearers_of(agent, sim.agents, radius)
        context = {
            "fire_reason": getattr(agent, "_last_fire_reason", "") or "",
            "drive": round(agent.drive, 3),
            "recent_mem": agent.mem.recent(4),
            "company_ids": [c.id for c in company],
            "saw_feed": any(ep.kind in ("sns", "news")
                            for ep in agent.mem.buffer[-4:]),
            "place": _place_of(sim, agent),
            "adopted_n": len(agent.adopted),        # その時までに採用済みの語数
        }
        item = sim.labels.coin(agent, action["word"], step=step, sim_min=sim_min,
                               logger=sim.logger, context=context)
        if item is None:                           # constrained で棄却された = 沈黙
            return
        action = {"type": "speak", "text": action["text"], "use_items": [item.text]}
        kind = "speak"                             # 造語はそのまま口に出す

    if kind == "speak":
        radius = float(sim.cfg.world.perception_radius_m)
        hearers = hearers_of(agent, sim.agents, radius)
        # 返答保証: 返答権を渡す相手を決定論選択(既定 nearest=最寄り1人=現行と同一。
        # prompts.reply_partner=closeness で関係加重の宛先へ)。hearer 集合は不変=聞く人は
        # 変わらない・返す人だけが変わる。乱数なし=R1。
        if hearers:
            partner = _select_partner(sim, agent, hearers)
            if (not partner.sleeping and partner.conv_turns_left > 0
                    and step >= partner.conv_cooldown_until):
                partner._reply_to = (agent.id, action["text"])
            # 対話履歴(prompts.dialog_history=true のみ)。話者と相手の相手別バッファへ同一発話を
            # 積む(次回 social/reply のプロンプトに直近2往復を注入)。OFF は _dialog_on=false →
            # 一切触らない=状態変化なし=バイト一致。
            if _dialog_on(sim):
                _dialog_push(agent, partner.id, agent.name, action["text"])
                _dialog_push(partner, agent.id, agent.name, action["text"])
        words = [w for w in action.get("use_items", []) if w in agent.adopted]
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                             kind="speak", x=agent.x, y=agent.y,
                             payload={"text": action["text"], "items": words,
                                      "hearers": [a.id for a in hearers]}))
        # 会話(対面)からの予定抽出→話者と聞き手全員の帳簿へ記入(既定 OFF=no-op)。
        _record_appointments(sim, agent, action["text"],
                             [h.id for h in hearers], step, sim_min)
        for word in words:
            sim.labels.use(agent, word, step=step, sim_min=sim_min, logger=sim.logger)
        agent.remember(f"「{action['text']}」と話した", kind="said")
        agent.said.append(f"「{action['text']}」")
        if len(agent.said) > 4:
            agent.said = agent.said[-4:]
        # 集団プラグイン(既定 OFF): in-group の発話は heard_valence を boost 倍。
        # OFF 時は boost=1.0 のままで on_heard_valence と完全同一(バイト不変)。
        _collective = sim.psychcfg["collective"]
        _tools = getattr(sim, "tools", None)
        # 被傾聴/無視 = 社会的説得の成否(Bandura)
        d_spoke = factor_update.on_spoke(agent, len(hearers), mags=sim.mags,
                                         step=step, sim_min=sim_min,
                                         logger=sim.logger)
        if d_spoke:
            drive.add(agent, "state_change", sim.drivecfg, scale=abs(d_spoke))
        # 評判(Wave G2): 聞き手のいる発話=社会的可視性 → 話者の評判+(既定 OFF=no-op)。
        if hearers and _relations_on(sim):
            relations_mod.gain_reputation(agent, sim.relationscfg["rep_mention"],
                                          "mention", step, sim_min, sim.logger)
        v_text = valence(action["text"])
        for hearer in hearers:
            sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=hearer.id,
                                 kind="hear", x=hearer.x, y=hearer.y,
                                 payload={"speaker": agent.id, "items": words}))
            # 感情・興味・注意ハブ: 対面で聞いた=被話しかけ(addressed)+ |valence| → 覚醒↑。
            # その覚醒/新奇(未知語=items)を顕著記憶の importance 加点に反映(affect OFF=no-op)。
            _arouse(sim, hearer, "heard", step, sim_min,
                    valence_abs=abs(v_text), addressed=1.0,
                    novelty=(1.0 if words else 0.0))
            hearer.remember(f"{agent.name}が「{action['text']}」と言っていた",
                            kind="heard",
                            importance_bonus=_imp_bonus(sim, hearer,
                                                        novelty=(1.0 if words else 0.0)))
            sim.net.add_contact(agent.id, hearer.id)   # 対面で知り合い→DM可・フォロー
            # 関係台帳(Wave G2 の交流符号=発話の感情価)。OFF は従来の record_contact と完全同一。
            _contact(sim, hearer, agent.id, agent.name, action["text"],
                     v_text, step, sim_min)
            _contact(sim, agent, hearer.id, hearer.name, "",
                     v_text, step, sim_min)
            _hear_words(sim, hearer, words, agent.id, "face", step, sim_min)
            # 聞いた言葉の感情価(SIMCA: affective)+ 語の使用の目撃(代理経験)。
            # in-group 判定=グループ所属の照合だけ engine が行い、倍率(不透明 float)を渡す。
            hv_boost = 1.0
            if _collective["enabled"] and _tools is not None \
                    and _tools.share_group(agent.id, hearer.id):
                hv_boost = _collective["ingroup_boost"]
            d_v = factor_update.on_ingroup_heard(hearer, v_text, hv_boost,
                                                 mags=sim.mags, step=step,
                                                 sim_min=sim_min, logger=sim.logger)
            if words:
                d_v += factor_update.on_vicarious(hearer, mags=sim.mags, step=step,
                                                  sim_min=sim_min, logger=sim.logger)
            if d_v:
                drive.add(hearer, "state_change", sim.drivecfg, scale=abs(d_v))
            # 対面での意見更新(FJ、w=w_face)。source=話者。
            _shift_opinion(sim, hearer, v_text, agent.id,
                           sim.opinioncfg["w_face"], step, sim_min)


# ---------------------------------------------------------------- 世界イベント
def _phase_world_events(sim, step: int, sim_min: int) -> None:
    """シナリオイベント(公式発表など)をニュース+SNS で配信。伝播の根=メディア。"""
    for ev in sim.world_events:
        if ev["step"] != step:
            continue
        items = []
        if ev.get("word"):
            sim.labels.coin_media(ev["word"], step=step, sim_min=sim_min,
                                  logger=sim.logger)
            items = [ev["word"]]
        sim.net.publish_news(ev["title"], ev.get("text", ""), items, step)
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=-1,
                             kind="world_event", x=0.0, y=0.0,
                             payload={"title": ev["title"],
                                      "text": ev.get("text", ""),
                                      "word": ev.get("word")}))


# ---------------------------------------------------------------- 口座 E5(月次境界)
def _log_rent(sim, agent, amount: float, paid: float, step: int, sim_min: int,
              phase: str) -> None:
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                         kind="rent", x=agent.x, y=agent.y,
                         payload={"amount": round(float(amount), 1),
                                  "paid": round(float(paid), 1),
                                  "carry": round(agent.rent_due, 1),
                                  "account": round(agent.account, 1),
                                  "phase": phase}))


def _phase_accounts_day(sim, step: int, sim_min: int) -> None:
    """口座 E5 の暦日境界(1日=144step、run開始日=1日として day%30)。

    給料日(payday_dom): 月給者(本業日給を持つ会社員・店員)へ economy.wages×勤務日数を
    まとめ支給(口座へ)。給料日の翌日: 家賃 = 月収相当(period_income)×rent_share を口座
    から引き落とし。残高不足は rent_due に繰越し、翌日以降に回収+money_pressure が効く。
    すべて決定論(乱数なし)。来街者は街の外に家=口座/家賃なし。"""
    block_day = step // 144                        # run 開始ブロック=0(=暦の1日目)
    if block_day == getattr(sim, "_acct_day", -1):
        return
    sim._acct_day = block_day
    acc = sim.economy["accounts"]
    dom = block_day % 30 + 1                        # 暦の何日目(1..30)。run開始=1日
    payday = int(acc["payday_dom"])
    rent_dom = payday % 30 + 1                      # 給料日の翌日
    share = float(acc["rent_share"])
    for agent in sim.agents:
        if agent.visitor:
            continue
        if agent.rent_due > 0.0 and agent.account > 0.0:   # 繰越家賃をまず回収
            pay = min(agent.account, agent.rent_due)
            agent.account -= pay
            agent.rent_due -= pay
            _log_rent(sim, agent, pay, pay, step, sim_min, "carry")
        if dom == payday and agent.wage > 0 and agent.work_days > 0:
            salary = agent.wage * agent.work_days      # 月給まとめ = 日給 × 勤務日数
            agent.last_salary = salary
            agent.work_days = 0
            _pay_wage(sim, agent, salary, step, sim_min, source="salary")
        if dom == rent_dom and not agent.evicted:      # 立退き中=住居なし→家賃は発生しない
            rent = agent.period_income * share         # 月収相当 × rent_share
            agent.period_income = 0.0
            if rent > 0.0:
                pay = min(agent.account, rent)
                agent.account -= pay
                agent.rent_due += (rent - pay)         # 残高不足=翌日繰越
                _log_rent(sim, agent, rent, pay, step, sim_min, "rent")
        _eviction_bankruptcy_day(sim, agent, acc, step, sim_min)


def _eviction_bankruptcy_day(sim, agent, acc: dict, step: int, sim_min: int) -> None:
    """立退き・破産サイクル(制度深化3 2026-07-08)。暦日ごとに1回・決定論・乱数なし。

    現実の対応(rights-institutions-gap の見送り分): 滞納の継続→信頼関係破壊の法理で
    契約解除=立退き(住居喪失)、さらに支払不能が続く→自己破産(免責=滞納債務の消滅+
    自由財産を残して資産圧縮+制限期間=出店不可)。完済で再入居=サイクルが閉じる。
    既定 eviction_days=0 / bankruptcy_days=0 では滞納カウントごと何もしない=E5 従来と
    完全同一(rent_due 自体も accounts OFF では常に 0)。"""
    evict_days = int(acc["eviction_days"])
    bank_days = int(acc["bankruptcy_days"])
    if evict_days <= 0 and bank_days <= 0:
        return
    if agent.rent_due <= 0.0:
        agent.arrears_days = 0
        if agent.evicted:                              # 滞納完済 → 再入居(サイクルの閉じ)
            agent.evicted = False
            sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                                 kind="eviction", x=agent.x, y=agent.y,
                                 payload={"phase": "rehoused"}))
            agent.remember("滞納を払い終えて部屋に戻れた", importance_bonus=0.3)
        return
    agent.arrears_days += 1
    # 破産(立退きより深い段): 免責=債務消滅、自由財産(keep)以外を圧縮、制限期間を開始
    if bank_days > 0 and agent.arrears_days >= bank_days:
        debt = agent.rent_due
        agent.rent_due = 0.0
        agent.arrears_days = 0
        keep = float(acc["bankruptcy_keep"])
        total = agent.money + agent.account
        seized = max(0.0, total - keep)
        if seized > 0.0:                               # 現金を優先して残し、超過分を処分
            new_total = total - seized
            agent.money = min(agent.money, new_total)
            agent.account = new_total - agent.money
        agent.bankrupt_until = step + int(acc["bankruptcy_restrict_days"]) * 144
        tools = getattr(sim, "tools", None)
        had_venture = bool(tools is not None
                           and tools.force_close_venture(sim, agent, step, sim_min,
                                                         reason="bankruptcy"))
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                             kind="bankruptcy", x=agent.x, y=agent.y,
                             payload={"debt": round(debt, 1),
                                      "seized": round(seized, 1),
                                      "keep": round(keep, 1),
                                      "until_step": agent.bankrupt_until,
                                      "venture_closed": had_venture}))
        if had_venture:                                # 店の倒産は社会に見える(報道)
            sim.net.publish_news("店じまい", "経営難で店を畳んだ人がいる", [], step)
        factor_update.on_bankrupt(agent, float(acc["bankruptcy_grievance"]),
                                  step=step, sim_min=sim_min, logger=sim.logger)
        agent.remember("破産手続きをした。借金は消えたが、当分は店も出せない",
                       importance_bonus=0.5)
        return
    # 立退き: 滞納の継続で住居を失う(自宅建物に入れない=路上就寝。家賃の新規発生は停止)
    if evict_days > 0 and not agent.evicted and agent.arrears_days >= evict_days:
        agent.evicted = True
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                             kind="eviction", x=agent.x, y=agent.y,
                             payload={"phase": "evicted",
                                      "arrears": round(agent.rent_due, 1),
                                      "days": agent.arrears_days}))
        factor_update.on_evicted(agent, float(acc["eviction_grievance"]),
                                 step=step, sim_min=sim_min, logger=sim.logger)
        agent.remember("家賃の滞納で部屋を立ち退かされた", importance_bonus=0.5)


# ---------------------------------------------------------------- 日次境界(経済)
def _phase_daily(sim, step: int, sim_min: int) -> None:
    """日付が変わったら、手持ちが逼迫している人に 1日1回の心理的圧(grievance+)。

    R9: factors 側は金額・イベントのみ受け取る。逼迫判定(残高<閾値)と頻度はここが担う。
    口座 ON(E5): 給料日・家賃の月次境界も処理し、逼迫判定は現金+口座の合算で行う。

    Wave G1(2026-07-07・すべて既定 OFF=バイト一致):
    - 相対的剥奪: 参照集団(街に居る他者)の所持金中央値を下回る量に応じた不満。engine は
      正規化差 × 係数の**不透明な magnitude**だけを factors に渡す(因子名は書かない)。coef==0
      なら中央値算出ごとスキップ=完全 no-op。判定は当日の稼ぎ・固定費控除の前(=中央値と同一基準)。
    - 固定費: 家賃以外の日次固定支出(既存 spend の作法。cat="fixed_cost")。fixed_cost_daily>0
      のときだけ控除=OFF は spend 0 件。控除は逼迫判定より前=固定費が同日の生活圧に効く。
    """
    if not _economy_on(sim):
        return
    accounts_on = _accounts_on(sim)
    if accounts_on:
        _phase_accounts_day(sim, step, sim_min)    # 暦日境界: 月給まとめ支給・家賃引落し
    day = sim_min // 1440
    if day == getattr(sim, "_econ_day", -1):
        return
    sim._econ_day = day
    thr = float(sim.economy["money_pressure_threshold"])
    fixed_cost = float(sim.economy.get("fixed_cost_daily", 0.0))     # 固定費(既定0=OFF)
    rd_coef = factor_update.relative_deprivation_coef(sim.mags)      # 相対的剥奪係数(既定0=OFF)
    # 参照集団の所持金中央値(coef>0 のときだけ算出=OFF なら完全 no-op・決定論)。
    ref_median = _money_median(sim) if rd_coef > 0.0 else None
    for agent in sim.agents:
        # 相対的剥奪: 参照中央値を下回る量(正規化差)に応じた不満。当日の稼ぎ・固定費控除の前の
        # 所持金で判定=中央値と同一基準(rd_coef==0 なら ref_median=None でこのブロックごとスキップ)。
        # ここは grievance state の**個体差復活**のみを目的とする(飽和を破り R²(k) 測定に分散を戻す)。
        # money_pressure と違い drive.add は呼ばない=発火動態に触れない(R1: FixedLLM で ON==OFF 呼数不変・
        # 実験の交絡回避)。grievance→発火の結合を有効化するかは別途 Fable5 判断(有効化すると呼数不変が崩れる)。
        if ref_median is not None and agent.money < ref_median:
            mag = rd_coef * min(1.0, (ref_median - agent.money) / max(ref_median, 1.0))
            factor_update.on_relative_deprivation(
                agent, mag, step=step, sim_min=sim_min, logger=sim.logger)
        # 自営(固定職場なし)の日銭: 日額の元手 × 出来高 U(0.2, 1.4)。来街者には出ない。
        # 新 stream "gig"(既存 draw 順に影響しない)。日次で1回、決定論。
        if not agent.visitor:
            prof = gig_profile(agent.occupation, sim.economy)
            if prof and prof["daily_base"] > 0:
                grng = sim.hub.stream("gig", agent.id, step)
                _pay_wage(sim, agent, prof["daily_base"] * float(grng.uniform(0.2, 1.4)),
                          step, sim_min, source="gig")
        # 固定費(光熱費・サブスク等): 家賃以外の恒常的生活圧。来街者は街の外に住居=対象外
        # (既存 rent と同じ扱い)。既定 0.0=控除ゼロ=spend イベントなし=バイト一致。
        if fixed_cost > 0.0 and not agent.visitor:
            _spend(sim, agent, fixed_cost, "fixed_cost", step, sim_min)
        bal = agent.money + (agent.account if accounts_on else 0.0)   # ON=合算で逼迫判定
        if bal >= thr:
            continue
        delta = factor_update.on_money_pressure(agent, mags=sim.mags, step=step,
                                                sim_min=sim_min, logger=sim.logger)
        if delta and agent.loc != "outside":
            drive.add(agent, "state_change", sim.drivecfg, scale=abs(delta))


# ---------------------------------------------------------------- 行政(日次境界)
def _gov_payroll(sim, gov, step: int, sim_min: int) -> None:
    """公務員へ日給を支給(源泉徴収つき)。区職員=ward予算 / 警察官・消防士=metro予算(歳出)。

    勤務地写像(persona 側=別バッチ所有で編集不可)に公務員が無く通常の勤務完遂経路に乗らないため、
    行政のペイロールとして日次で予算から直接支給する(WAGE_CAT 外=gig/本業と二重にならない)。
    sim.agents は id 昇順・乱数なし(決定論)。通勤(来街)公務員もこの街で働く扱いで支給。"""
    for agent in sim.agents:
        pay = civil_servant_pay(agent.occupation, sim.economy)
        if pay is None:
            continue
        gross, fund = pay
        _pay_wage(sim, agent, gross, step, sim_min, source="civil", fund_level=fund)


def _gov_benefits(sim, gov, step: int, sim_min: int) -> None:
    """セーフティネット(区の生活困窮者支援): 所持金総額が閾値未満の居住者へ ward から給付。

    決定論(乱数なし・id 昇順)。給付は ward の歳出(civic_service で記録)。所持金の改善が翌日以降の
    逼迫(money_pressure)を間接的に緩める(因子へは書かない=金銭の増減のみ)。来街者は対象外。"""
    thr = float(gov.cfg["benefit_threshold"])
    amt = float(gov.cfg["benefit_amount"])
    if amt <= 0:
        return
    accounts_on = _accounts_on(sim)
    for agent in sim.agents:
        if agent.visitor:
            continue
        total = agent.money + (agent.account if accounts_on else 0.0)
        if total >= thr:
            continue
        agent.money += amt
        gov.expense("ward", amt)
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                             kind="civic_service", x=agent.x, y=agent.y,
                             payload={"service": "welfare_benefit", "level": "ward",
                                      "amount": round(amt, 1),
                                      "detail": "生活困窮者支援"}))


def _phase_government(sim, step: int, sim_min: int) -> None:
    """日次境界: 前日会計の締め(public_budget)+ 公務員ペイロール + 困窮者給付。

    行政 OFF なら完全 no-op(tax/civic_service/public_budget 0 件・既存イベント列に無影響)。
    public_budget の agent_id は世界レベル(-1、world_event の既存流儀を踏襲)。給付を _phase_daily
    より前に行い、給付後の残高で逼迫(money_pressure)が判定される=セーフティネットの間接経路。"""
    if not _government_on(sim):
        return
    gov = sim.government
    day = sim_min // 1440
    records = gov.daily(day)                   # 空=同日(何もしない)
    if not records:
        return
    for rec in records:                        # 3主体の当日会計を締めて出力(Σ歳入−Σ歳出=残高変化)
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=-1,
                             kind="public_budget", x=0.0, y=0.0,
                             payload={"level": rec["level"],
                                      "revenue": round(rec["revenue"], 1),
                                      "expense": round(rec["expense"], 1),
                                      "balance": round(rec["balance"], 1)}))
    _gov_payroll(sim, gov, step, sim_min)      # 当日分の歳出は次の日境界で締め・出力
    _gov_benefits(sim, gov, step, sim_min)


# ---------------------------------------------------------------- 制度DSL(日次境界)
def _resolve_place_node(sim, place: str) -> str | None:
    """place(POI 名)を最初に部分一致した POI のノードへ解決する(決定論・乱数なし)。"""
    place = str(place or "").strip()
    if not place:
        return None
    for p in sim.city.poi_list:
        if place in p["name"]:
            return p["node"]
    return None


def _phase_rules(sim, step: int, sim_min: int) -> None:
    """制度DSL の日次境界処理: 期限切れルールの失効 + weekly_event の発火。

    発火 = world ニュース配信 + その場所を今日の余暇候補としてブースト(routine が読む)。
    rules 無効時は完全に no-op(イベント列不変)。
    """
    rb = getattr(sim, "rulebook", None)
    if rb is None or not rb.cfg["enabled"]:
        return
    day = sim_min // 1440
    res = rb.advance_day(day, step)
    if res is None:                                # 同日 = 何もしない
        return
    expired, weekly = res
    for r in expired:                              # 有効期限切れ(duration_days)
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=r["proposer"],
                             kind="rule_expired", x=0.0, y=0.0,
                             payload={"rule_id": r["id"], "type": r["type"],
                                      "reason": "duration"}))
    for r in weekly:                               # 期日到来 → ニュース + 場所ブースト
        node = _resolve_place_node(sim, r["place"])
        sim.net.publish_news("定期イベント", f"{r['title']}(@{r['place']})", [], step)
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=r["proposer"],
                             kind="rule_weekly_fire", x=0.0, y=0.0,
                             payload={"rule_id": r["id"], "title": r["title"],
                                      "place": r["place"], "node": node}))
        if node is not None:
            rb.today_boost.append(node)


def _phase_recursion(sim, step: int, sim_min: int) -> None:
    """再帰性(第9バッチ)の日次境界: 昨日の客観カウント確定(norm_digest)+ 執行多発ルールの
    ニュース化。既定 OFF=完全 no-op(イベント・ニュース・乱数とも不変=ゴールデンを守る)。"""
    recur = getattr(sim, "recursion", None)
    if recur is not None:
        recur.on_day(sim, sim_min // 1440, step, sim_min)


def _phase_assembly(sim, step: int, sim_min: int) -> None:
    """代表制議会の改選(制度深化3 2026-07-08)。既定 OFF=完全 no-op(乱数なし)。

    任期・改選日のゲートは tools.elect_assembly 側(議会が無い or 任期満了の日だけ改選)。"""
    from ..tools import elect_assembly
    elect_assembly(sim, step, sim_min)


def _phase_status(sim, step: int, sim_min: int) -> None:
    """日次境界: 社会的地位スコアの再計算(第11バッチ・ヒエラルキー 2026-07-08)。

    客観カウント(評判/フォロワー/資産/制度実績/商い/主催)の母集団内百分位を重み混合して
    各 agent.status(0..1)を決定論算出する(乱数なし・LLM なし・k 非参照)。暦日境界のゲートは
    status.phase_day が自前で持つ。hierarchy 無効なら完全 no-op(status 不変・イベント 0 件=バイト一致)。"""
    status_mod.phase_day(sim, step, sim_min)


def _phase_reflect_day(sim, step: int, sim_min: int) -> None:
    """(第12バッチ)日次境界: 無意識層「最近の自分」の更新+日内衝撃ゲージのリセット。

    reflection.deep / implicit_self とも OFF(既定)なら完全 no-op(バイト一致)。
    更新はリセットの前(昨日のカウントを材料に組む=Bem 自己知覚の日次サイクル)。
    乱数なし・LLM なし。"""
    rcfg = getattr(sim, "reflectcfg", None)
    if rcfg is None:
        return
    imp_on = rcfg["implicit_self"]["enabled"]
    deep_on = rcfg["deep"]["enabled"]
    if not (imp_on or deep_on):
        return
    day = sim_min // 1440
    if day == 0 or day == getattr(sim, "_reflect_day", -1):
        return
    sim._reflect_day = day
    ema = float(rcfg["implicit_self"]["ema"])
    from ..cognition.reflection import update_implicit_self
    for a in sim.agents:
        if imp_on:
            update_implicit_self(a, ema)
        a.impact_today = 0.0
        a.impact_neg_today = 0.0
        a.impact_pos_today = 0.0
        if a.behav_today is not None:
            a.behav_today = {}


def _free_dest(sim, where: str) -> str | None:
    """開放行動の行き先名を決定論で解決(完全一致→部分一致。ソート順=決定論)。"""
    idx = getattr(sim, "_free_name_index", None)
    if idx is None:
        idx = {}
        for p in sorted(sim.city.poi_list, key=lambda q: str(q.get("name") or "")):
            nm = str(p.get("name") or "").strip()
            if nm and nm not in idx:
                idx[nm] = p["node"]
        sim._free_name_index = idx
    if where in idx:
        return idx[where]
    for nm in sorted(idx):
        if where in nm or nm in where:
            return idx[nm]
    return None


# ---------------------------------------------------------------- 生活の自己決定 P2(D3 棚卸し)
def _p2cfg(sim) -> dict | None:
    """freedom.p2 サブ設定(既定 全 OFF)。freedom 無効なら None。"""
    fc = getattr(sim, "freedomcfg", None)
    return fc.get("p2") if fc else None


def _freedom_tally(sim, key: str) -> None:
    """自由度の観測カウンタ(その step の choice_points / exercised)を1増やす(L2 追加列の元)。"""
    fs = getattr(sim, "freedom_stats", None)
    if fs is None:
        fs = sim.freedom_stats = {"choice_points": 0, "exercised": 0}
    fs[key] = fs.get(key, 0) + 1


def _p2_offers(sim, agent, trigger: str, pois, company) -> str | None:
    """発火プロンプトに載せる P2 メニュー(中立提示・客観条件つき・決定論・乱数なし)。

    覚醒・自由時間の非会話デリバレーション発火(solo/social/post/novel_place/congestion/
    unknown_word)で、条件を満たす選択肢だけを1行ずつ置く(POI 到着=novel_place は buy/study の
    自然な機会)。会話系(reply/dm)は相手への応答が主眼なので載せない。提示があれば choice_points
    を1計上する。P2 全 OFF / 非該当なら None(=build_prompt が1行も足さない=バイト一致)。"""
    p2 = _p2cfg(sim)
    if not p2 or trigger not in ("solo", "social", "post",
                                 "novel_place", "congestion", "unknown_word"):
        return None
    if not any(p2.get(k) for k in ("move_home", "buy", "study", "partnership", "deviance")):
        return None
    free_time = (agent.activity not in ("working", "commuting")
                 and not agent.sleeping and agent.loc != "outside")
    cats = {p["cat"] for p in pois}
    tools = getattr(sim, "tools", None)
    venture_cost = float(getattr(tools, "cfg", {}).get("venture_cost", 30000.0)) \
        if tools is not None else 30000.0
    text = freedom_p2_mod.menu(
        agent, cfg=p2, venture_cost=venture_cost,
        near_commercial=bool(cats & freedom_p2_mod.COMMERCIAL_CATS),
        near_school=bool(cats & freedom_p2_mod.STUDY_CATS),
        free_time=free_time, has_company=bool(company),
        accounts_on=_accounts_on(sim))
    if text:
        _freedom_tally(sim, "choice_points")
    return text


def _apply_p2(sim, agent, action: dict, step: int, sim_min: int) -> None:
    """P2 行動(#6-#10)の裁定ディスパッチ。該当項目が OFF なら静かに無視(=wander 相当)。

    いずれの裁定も money/closeness/物理位置(k 非依存の観測量)と config・新 stream のみを見る
    (k を発火判断に食わせない)。行使できたら exercised を1計上(自由度行使率の観測)。"""
    p2 = _p2cfg(sim)
    if p2 is None:
        return
    kind = action["type"]
    if kind == "move_home":
        _apply_move_home(sim, agent, action, step, sim_min, p2)
    elif kind == "buy":
        _apply_buy(sim, agent, action, step, sim_min, p2)
    elif kind == "study":
        _apply_study(sim, agent, action, step, sim_min, p2)
    elif kind == "propose_partnership":
        _apply_partnership(sim, agent, action, step, sim_min, p2)
    elif kind == "break_up":
        _apply_break_up(sim, agent, step, sim_min, p2)


def _apply_move_home(sim, agent, action, step, sim_min, p2) -> None:
    """#6 住居移転: 敷金(=現金障壁)を払えるとき、空き住戸(他エージェントの home でない
    residential)へ新 stream "move_home" で決定論転居する。家賃額は現行の収入比のまま
    (建物別家賃の内生化はしない=正直な近似)。イベント move_home(from/to/deposit)。"""
    if not p2["move_home"]:
        return
    deposit = float(p2["deposit"])
    accounts_on = _accounts_on(sim)
    total = agent.money + (agent.account if accounts_on else 0.0)
    if total < deposit:                                # 敷金が払えない=引っ越せない(客観条件)
        return
    rng = sim.hub.stream("move_home", agent.id, step)  # 新 stream(既存 draw 順に不干渉)
    bld = freedom_p2_mod.pick_home(sim, agent, action.get("area"), rng)
    if bld is None:                                    # 空き住戸なし
        return
    old = agent.home_building
    dep = deposit                                      # 敷金の控除(口座→現金の順)
    if accounts_on:
        take = min(agent.account, dep)
        agent.account -= take
        dep -= take
    agent.money = max(0.0, agent.money - dep)
    levels = int(bld.get("levels", 1) or 1)
    agent.home_building = bld["id"]
    agent.home_node = bld["entrance"]
    agent.home_floor = 1 + int(rng.integers(max(1, levels)))
    if agent.home_floor > levels:
        agent.home_floor = max(1, levels)
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                         kind="move_home", x=agent.x, y=agent.y,
                         payload={"from": old, "to": bld["id"],
                                  "deposit": round(deposit, 1)}))
    agent.remember("新しい住まいに引っ越した")
    _freedom_tally(sim, "exercised")


def _apply_buy(sim, agent, action, step, sim_min, p2) -> None:
    """#7 消費の意思: 既存の消費経路(_spend)で即時支出し、spend payload に chosen:true を足す
    (新 kind 不要)。非発火の既存 buy 抽選はフォールバックとしてそのまま残す(完全置換しない)。"""
    if not p2["buy"]:
        return
    cat = str(action.get("cat") or "").strip().lower()
    price_cat = freedom_p2_mod.BUY_PRICE_CAT.get(cat, "shop")
    if _economy_on(sim):
        price = price_of(price_cat, sim.economy, getattr(sim, "rulebook", None))
        total = agent.money + (agent.account if _accounts_on(sim) else 0.0)
        if price > 0 and total >= price:
            _spend(sim, agent, price, price_cat, step, sim_min, chosen=True)
    agent.remember(f"欲しい物を買った({cat or price_cat})")
    _freedom_tally(sim, "exercised")


def _apply_study(sim, agent, action, step, sim_min, p2) -> None:
    """#8 学び直し: 既存 study イベントで記録+その場に滞在。効果は記録のみ(既存の記憶接触に
    留める)。★賃金/生産性への経路は Skill(技能蓄積)討議後(ユーザー決定)。"""
    if not p2["study"]:
        return
    topic = str(action.get("topic") or "").strip()[:60] or "自習"
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                         kind="study", x=agent.x, y=agent.y,
                         payload={"subject": topic, "role": "self", "chosen": True}))
    agent.stay_until = max(agent.stay_until, step + 2)  # その場に留まる(聴講)
    agent.remember(f"学んだ: {topic}")
    _freedom_tally(sim, "exercised")


def _apply_partnership(sim, agent, action, step, sim_min, p2) -> None:
    """#9 交際の申込: 相手が近傍に実在し、既存 relations の closeness が household の形成閾値
    以上なら partner_formed(既存 kind・既存の世帯結合処理を再利用)、未満なら partnership_declined
    (新 kind)。相手側の発火時応答は将来拡張(今回は既存閾値の決定論判定=正直な簡略化)。"""
    if not p2["partnership"]:
        return
    to_name = str(action.get("to") or "").strip()
    if not to_name or getattr(agent, "partner_id", None) is not None:
        return
    radius = float(sim.cfg.world.perception_radius_m)
    company = hearers_of(agent, sim.agents, radius)
    target = next((c for c in company if c.name == to_name), None)
    if target is None:                                 # 名前の部分一致で救済
        target = next((c for c in company if to_name in c.name), None)
    if target is None or getattr(target, "partner_id", None) is not None:
        return                                         # 相手が近くにいない/既に交際中
    thr = freedom_p2_mod.partner_threshold(sim, p2)
    rel = agent.mem.relations.get(target.id)
    closeness = float(rel.get("closeness", 0.0)) if rel else 0.0
    if closeness >= thr:
        household_mod.bond(sim, agent, target, step, sim_min)   # 世帯結合処理を再利用
    else:
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                             kind="partnership_declined", x=agent.x, y=agent.y,
                             payload={"to": int(target.id),
                                      "closeness": round(closeness, 3)}))
        agent.remember(f"{target.name}に交際を申し込んだが、まだその段階ではなかった")
    _freedom_tally(sim, "exercised")


def _apply_break_up(sim, agent, step, sim_min, p2) -> None:
    """#9 別れ: 既存 relation_break 流儀+世帯分離(household.unbond で双方の partner_id を外す)。"""
    if not p2["partnership"]:
        return
    if household_mod.unbond(sim, agent, step, sim_min):
        _freedom_tally(sim, "exercised")


def _apply_free_action(sim, agent, action: dict, step: int, sim_min: int) -> None:
    """開放行動 "do" の裁定(第17バッチ)。物理・所持金・拘束(勤務/就寝)の客観ゲートだけ
    かけて、意味づけは価値タグ(辞書+中立自己申告)で観測する。乱数なし=決定論。

    効果: 消費カテゴリなら中央値価格を支出(所持金の範囲)/価値の充足(sat)/記憶/
    L1 "free_action"。行き先が解決できれば徒歩で移動(自由時間のみ=現実拘束の維持)。"""
    from .. import values as values_mod
    fcfg = sim.freedomcfg
    what = str(action["what"])
    minutes = min(int(action.get("minutes", 30)), fcfg["max_minutes"])
    category, lex_tags = values_mod.classify(what)
    report = values_mod.parse_report(action.get("value_report"))
    tags = values_mod.blend_tags(lex_tags, report)
    match = values_mod.value_match(agent, tags)
    cost = values_mod.category_cost(category)
    paid = 0.0
    if cost > 0 and agent.money >= cost:
        _spend(sim, agent, cost, f"free_{category}", step, sim_min)
        paid = cost
    dest = None
    where = action.get("where")
    if where and not agent.sleeping and agent.activity != "working":
        dest = _free_dest(sim, str(where))
    if dest and dest != agent.node:
        path, used_mode = sim.router.route(agent.node, dest, "walk")
        if len(path) >= 2:
            agent.route = path[1:]
            agent.edge_offset = 0.0
            agent.dest = dest
            agent.trip_mode = used_mode
            agent._pending_stay = max(1, min(24, round(minutes / 10)))
            agent._pending_activity = ""
        else:
            dest = None
    sat_after = values_mod.satisfy(agent, tags, fcfg)
    agent.remember(f"自由な時間: {what}")
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                         kind="free_action", x=agent.x, y=agent.y,
                         payload={"what": what[:120], "category": category,
                                  "tags": tags, "match": match,
                                  "report": report or None, "minutes": minutes,
                                  "cost": paid, "dest": dest,
                                  "sat": sat_after or None}))


def _phase_freedom_day(sim, step: int, sim_min: int) -> None:
    """(第17バッチ)日次境界: 価値充足 sat の中立回帰(需要の再蓄積)。既定 OFF=完全 no-op。

    乱数なし・LLM なし。sat 属性が無い(freedom OFF)エージェントは values 側で no-op。"""
    fcfg = getattr(sim, "freedomcfg", None)
    if fcfg is None or not fcfg["open_actions"]:
        return
    day = sim_min // 1440
    if day == 0 or day == getattr(sim, "_freedom_day", -1):
        return
    sim._freedom_day = day
    from .. import values as values_mod
    for a in sim.agents:
        values_mod.decay_daily(a, fcfg)


# ---------------------------------------------------------------- 制度改変の3ルート(Wave G3)
# 執行(条例執行)を担う公務員=東京都の公安職(警察官)。区職員/消防士は執行しない。
POLICE_OCCS = ("警察官",)


def _routes(sim) -> dict:
    """institution_routes(労働争議/投票/執行)設定を返す(既定 OFF。tools.routes_of に委譲)。"""
    from ..tools import routes_of
    return routes_of(sim)


def _enforce_ventures(sim, officer_at, step: int, sim_min: int) -> None:
    """#10 逸脱: 無許可出店(venture の permitted:false)を近傍(同ノード)の警察官が摘発する。

    摘発 = 罰金(所持金−。government ON なら区の歳入)+ grievance+(factors 経由=no-fingerprint)
    + 強制閉店(既存 force_close_venture)。罰金額は新パラメータ freedom.p2.deviance_fine、grievance は
    既存 enforcement の係数を再利用(=既存の執行・罰金機構への接続)。決定論・非LLM・乱数なし。
    deviance OFF / tools 無し / 無許可出店なし なら完全 no-op(enforcement イベント 0 件=バイト一致)。"""
    p2 = _p2cfg(sim)
    if p2 is None or not p2["deviance"]:
        return
    tools = getattr(sim, "tools", None)
    if tools is None:
        return
    fine = float(p2["deviance_fine"])
    griev = float(_routes(sim)["enforcement"]["grievance"])
    gov_on = _government_on(sim)
    for node in sorted(officer_at):                    # node 昇順=決定論
        officer = officer_at[node]
        for v in list(tools.ventures_by_node.get(node, [])):   # 閉店で index が縮むので複製を走査
            if v.get("permitted", True):               # 許可済みは対象外(既定 True=不変)
                continue
            owner = sim.agent_by_id.get(v["owner"])
            if owner is None or owner.id == officer.id:
                continue
            penalty = min(owner.money, fine)           # 罰金(所持金の範囲で)
            owner.money = max(0.0, owner.money - fine)
            if gov_on and penalty > 0:
                sim.government.collect("ward", penalty)
            factor_update.on_enforcement(owner, griev, step=step, sim_min=sim_min,
                                         logger=sim.logger)
            x, y = sim.city.node_xy(node)
            sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=officer.id,
                                 kind="enforcement", x=float(x), y=float(y),
                                 payload={"rule_id": None, "officer": officer.id,
                                          "target": owner.id,
                                          "penalty": round(penalty, 1),
                                          "venture": v["name"]}))
            tools.force_close_venture(sim, owner, step, sim_min, reason="unpermitted")
            owner.remember(f"無許可の出店を摘発され、店を畳んだ(罰金{round(penalty, 1)}円)")


def _phase_enforcement(sim, step: int, sim_min: int) -> None:
    """執行ルート(Wave G3): prohibit ルール下で、近傍(同ノード)に居る警察官が違反者を執行。

    執行 = 罰金(所持金−。government ON なら区の歳入へ)+ grievance+(factors 経由=no-fingerprint)。
    違反 = 禁止カテゴリの POI がある場所に居る非公務員(その場で禁止行為に及んでいる代理判定)。
    決定論(乱数なし・id 昇順)。enforcement OFF / rules 無効 / prohibit ルール無し / 警察官不在なら
    完全 no-op(enforcement イベント 0 件・所持金/grievance 不変=ゴールデンを守る)。"""
    if not _routes(sim)["enforcement"]["enabled"]:
        return
    officer_at: dict[str, object] = {}                 # node -> 最小 id の警察官(id 昇順で先勝ち)
    for a in sim.agents:
        if a.occupation in POLICE_OCCS and a.loc != "outside" and not a.sleeping \
                and a.node not in officer_at:
            officer_at[a.node] = a
    if not officer_at:
        return
    # #10 逸脱: 無許可出店(permitted:false)を近傍の警察官が摘発(deviance ON のみ。既存機構に接続)。
    # deviance OFF なら完全 no-op=禁止POI執行の従来経路はこの下で不変(=ゴールデンを守る)。
    _enforce_ventures(sim, officer_at, step, sim_min)
    rb = getattr(sim, "rulebook", None)
    if rb is None or not rb.cfg["enabled"] or not rb.has_prohibit():
        return
    hour = (sim_min % 1440) // 60
    prohibited = rb.prohibited_cats(hour)
    if not prohibited:
        return
    ecfg = _routes(sim)["enforcement"]
    fine = float(ecfg["fine"])
    griev = float(ecfg["grievance"])
    gov_on = _government_on(sim)
    for a in sim.agents:
        if a.loc == "outside" or a.sleeping:
            continue
        if a.occupation in CIVIL_SERVANTS:             # 公務員(警察官・区職員・消防士)は対象外
            continue
        officer = officer_at.get(a.node)
        if officer is None or officer.id == a.id:
            continue
        cats = {p["cat"] for p in sim.city.pois_at_node(a.node)}
        hit = sorted(cats & prohibited)
        if not hit:
            continue
        rule_id = rb.first_prohibit(hit[0])
        penalty = min(a.money, fine)                   # 罰金(所持金の範囲で)
        a.money = max(0.0, a.money - fine)
        if gov_on and penalty > 0:                     # 罰金は区の歳入(government ON 時)
            sim.government.collect("ward", penalty)
        factor_update.on_enforcement(a, griev, step=step, sim_min=sim_min,
                                     logger=sim.logger)   # 執行を受けた不満(factors 経由)
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=officer.id,
                             kind="enforcement", x=a.x, y=a.y,
                             payload={"rule_id": rule_id, "officer": officer.id,
                                      "target": a.id, "penalty": round(penalty, 1)}))
        # 再帰性(第9バッチ・既定 OFF): どのルールで執行されたかを計数し、被執行者の記憶にも
        # ルール名を残す(=知覚→不服→repeal 提案の材料)。OFF は従来の記憶文と完全同一。
        recur = getattr(sim, "recursion", None)
        rule_rec = None
        if recur is not None and recur.cfg["enabled"] and rule_id is not None:
            for r in rb.active:
                if r["id"] == rule_id:
                    rule_rec = r
                    break
            recur.note("enforced", rule_id=rule_id,
                       rule_name=rule_rec["name"] if rule_rec else "")
        if rule_rec is not None:
            a.remember(f"路上で取り締まりを受けた(罰金{round(penalty, 1)}円、"
                       f"「{rule_rec['name']}」による)")
        else:
            a.remember(f"路上で取り締まりを受けた(罰金{round(penalty, 1)}円)")
        # 勾留の最小形(制度深化2 第10バッチ・既定 0=拘束なし=従来と完全同一): 違反者を
        # その場で detention_steps 步の行動停止(会話・発火・移動なし。_decide/_phase_drive が
        # detained_until を見る)。時間による自由の剥奪=執行の重みの質を変える。決定論・非LLM。
        det = int(ecfg.get("detention_steps", 0))
        if det > 0:
            a.detained_until = step + det
            a.route = []
            a.dest = None
            a.exit_intent = False
            a.homing = False
            a.activity = ""
            sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=officer.id,
                                 kind="detention", x=a.x, y=a.y,
                                 payload={"target": a.id, "officer": officer.id,
                                          "rule_id": rule_id, "steps": det}))
            a.remember(f"取り締まりで{det * 10}分間その場に留め置かれた")


# ---------------------------------------------------------------- ツール(世界改変)
def _phase_tools(sim, step: int, sim_min: int) -> None:
    """ツールのライフサイクル(イベント開催/終了・ビラ失効・閉店・加入・署名)。

    R4: 効果は客観カウント(参加人数・署名数・売上)。LLM 審判は無い。
    """
    tools = getattr(sim, "tools", None)
    if tools is not None:
        tools.phase(sim, step, sim_min)


# ---------------------------------------------------------------- 日付・天気(日次境界)
def _phase_calendar_weather(sim, step: int, sim_min: int) -> None:
    """日付・天気(第7バッチ 2026-07-07)。日境界で当日の date_line/weather を確定し sim に保持。

    calendar/weather の両方 OFF なら完全 no-op(乱数・イベント・プロンプトとも既定と不変)。
    天気は新 stream 'weather' から1日1回だけ引く(既存 stream の draw 順は不変=決定論)。
    全エージェントが同じ当日の日付・天気を見る(グローバル文脈・k 非依存)。天気→不快感は
    factors 係数レイヤー経由(この関数は不透明な magnitude を渡すだけ=no-fingerprint)。
    """
    cal = sim.calendarcfg
    wea = sim.weathercfg
    if not cal["enabled"] and not wea["enabled"]:
        return
    day = sim_min // 1440
    if day == getattr(sim, "_cal_day", -1):
        return
    sim._cal_day = day
    if cal["enabled"]:
        sim.today_date_line = calendar.date_line(cal, sim_min)
    if wea["enabled"]:
        w = weather.weather_for(sim, day)
        sim.today_weather = w
        sim.today_weather_line = weather.weather_line(w)
        payload = {"cond": w["cond"], "temp_hi": w["temp_hi"]}
        if cal["enabled"]:                          # 暦がある日は日付・曜日・休日も併記
            payload["date"] = calendar.date_of(cal, sim_min).isoformat()
            payload["weekday"] = calendar.weekday_jp(cal, sim_min)
            payload["holiday"] = calendar.is_holiday(cal, sim_min)
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=-1,
                             kind="weather", x=0.0, y=0.0, payload=payload))
        mag = weather.discomfort_delta(w, wea)      # 不透明 magnitude(0.0 なら加算なし)
        if mag:
            for agent in sim.agents:
                if agent.loc == "outside":          # 街の外(不在)には効かせない
                    continue
                factor_update.on_weather(agent, mag, step=step, sim_min=sim_min,
                                         logger=sim.logger)


# ---------------------------------------------------------------- 文化カレンダー・群集(Wave G4)
def _annual_on(sim) -> bool:
    """年中行事・群集(Wave G4)が有効か。既定 OFF=新経路を一切通さない(バイト一致)。"""
    cfg = getattr(sim, "annualcfg", None)
    return bool(cfg and cfg["enabled"])


def _phase_annual(sim, step: int, sim_min: int) -> None:
    """日境界: 当日の年中行事を確定し annual_event を1件記録する(annual/calendar 有効時のみ)。

    行事日には today_event_line(プロンプト文脈)と today_crowd_event(群集フラグ)を確定する。
    annual OFF or calendar OFF(日付が定まらない)なら完全 no-op(プロンプト・イベント列・乱数とも
    不変)。決定論(乱数なし)。年中行事の発生は暦から純関数導出(schema: annual_event)。"""
    if not _annual_on(sim):
        return
    cal = sim.calendarcfg
    if not cal["enabled"]:                         # 日付判定に暦が要る(暦 OFF=年中行事も無効=不変)
        return
    day = sim_min // 1440
    if day == getattr(sim, "_annual_day", -1):
        return
    sim._annual_day = day
    ev = annual_mod.event_today(sim.annualcfg, cal, sim_min)
    sim.today_event_line = annual_mod.event_line(sim.annualcfg, cal, sim_min)
    sim.today_crowd_event = ev["name"] if (ev and ev["crowd"]) else None
    if ev is not None:                             # 年中行事の発生を1日1回記録(agent_id=-1)
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=-1,
                             kind="annual_event", x=0.0, y=0.0,
                             payload={"name": ev["name"],
                                      "date": calendar.date_of(cal, sim_min).isoformat()}))


def _phase_crowd(sim, step: int, sim_min: int) -> None:
    """群集(大規模行事型)の集中を観測し crowd_surge を記録する(annual 有効時のみ)。

    非群集日・annual OFF は完全 no-op。発火系(grievance/drive)には一切接続しない(混雑=雰囲気
    密度のみ=観測/可視化)。決定論・非LLM(R1: generate を1回も追加しない=呼数不変)。"""
    if not _annual_on(sim):
        return
    annual_mod.check_surge(sim, step, sim_min)


# ---------------------------------------------------------------- 予定(日次境界の GC)
def _phase_schedule_gc(sim, step: int, sim_min: int) -> None:
    """日境界で過去の予定を失効させる(当日経過で自動 GC)。既定 OFF=完全 no-op。"""
    if not _schedule_on(sim):
        return
    day = sim_min // 1440
    if day == getattr(sim, "_sched_day", -1):
        return
    sim._sched_day = day
    for agent in sim.agents:
        schedule.gc(agent, day)


# ---------------------------------------------------------------- 社会関係(日次境界)
def _phase_relations_day(sim, step: int, sim_min: int) -> None:
    """日境界: 長期不在の相手との親密度減衰(断絶=relation_break)+ 評判の風化(Wave G2)。

    relations 無効なら完全 no-op(closeness/reputation とも触れず・イベント 0 件=バイト一致)。
    決定論(乱数なし)。既存 _phase_* の並びは壊さない(OFF は即 return)。"""
    if not _relations_on(sim):
        return
    day = sim_min // 1440
    if day == getattr(sim, "_rel_day", -1):
        return
    sim._rel_day = day
    relations_mod.decay_day(sim, sim.relationscfg, step, sim_min)


# ---------------------------------------------------------------- 世帯・恋愛(後続波 H2)
def _household_on(sim) -> bool:
    """世帯・家族・恋愛(後続波 H2)が有効か。既定 OFF=新経路を一切通さない(バイト一致)。"""
    cfg = getattr(sim, "householdcfg", None)
    return bool(cfg and cfg["enabled"])


def _phase_household(sim, step: int, sim_min: int) -> None:
    """日次境界: パートナー形成(G2 closeness から相互閾値超の2者を決定論で恋人に。後続波 H2)。

    household 無効なら完全 no-op(partner_formed/life_event 0 件=バイト一致)。決定論(乱数なし)。
    R1: form_partners は generate() を1本も足さず、k・内面状態(構成概念)を読まず、relations の
    closeness(観測イベント由来=k 非依存の観測量)・名簿のみ参照する=呼数は k 非依存。世帯構築は
    起動時(simulation)で済み・デートは routine が担う。既存 _phase_* の並びは壊さない(OFF は即 return)。"""
    if not _household_on(sim):
        return
    day = sim_min // 1440
    if day == getattr(sim, "_partner_day", -1):
        return
    sim._partner_day = day
    household_mod.form_partners(sim, step, sim_min)


# ---------------------------------------------------------------- キャリア転換(Wave G5)
def _career_on(sim) -> bool:
    """キャリア転換(失業/求職/転職)が有効か。既定 OFF=新経路を一切通さない(バイト一致)。"""
    cfg = getattr(sim, "careercfg", None)
    return bool(cfg and cfg["enabled"])


def _log_career(sim, agent, kind: str, payload: dict, step: int, sim_min: int) -> None:
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id, kind=kind,
                         x=agent.x, y=agent.y, payload=payload))


def _phase_career(sim, step: int, sim_min: int) -> None:
    """日次境界: 失業/求職/転職(Wave G5・既定 OFF=no-op)。生活基盤の非連続=行動と動機の転換点。

    決定論: 確率判定は**新 stream "career"**(agent, step)から引く=既存 draw 順に挿入しない
    (ゴールデンと決定論の保護)。R1: career 機構は generate() を1回も足さず、k・内面状態
    (不満・効力・意欲・信念といった構成概念)を一切読まず、暦・config・新 stream のみ参照する(呼数は k 非依存)。
    失業は「勤務に行かない」=物理位置が変わり対面 co-location が変化しうる(FixedLLM で ON!=OFF に
    なりうる=群集 G4 と同型)→ 呼数不変は compute_matched 下の k 不変性で担保する。失業→grievance は
    factors hook(on_job_loss)に不透明 magnitude を渡す(engine は因子内部を読まない=no-fingerprint)。

    再配属(rehire/switch)には organizations の会社台帳(sim.orgs)が要る=organizations.enabled が前提。
    台帳が無い/被雇用者が居ないときは失業のみ発生しうる(rehire/switch は候補ゼロで no-op)。来街者は
    街の外に生活基盤=対象外。すべて日境界で1日1回・id 昇順(決定論)。"""
    if not _career_on(sim):
        return
    day = sim_min // 1440
    if day == getattr(sim, "_career_day", -1):
        return
    sim._career_day = day
    cfg = sim.careercfg
    griev = float(cfg["unemployment_grievance"])
    layoff_p = float(cfg["layoff_prob"])
    switch_p = float(cfg["switch_prob"])
    rehire_p = float(cfg["rehire_prob"])
    book = getattr(sim, "orgs", None)
    employers = organizations.employer_ids(book)
    commute = bool(getattr(sim, "orgscfg", {}) and sim.orgscfg.get("commute_to_poi"))

    def _pick_org(rng, exclude):
        cands = [e for e in employers if e != exclude]
        if not cands:
            return None
        return cands[int(rng.integers(len(cands)))]

    for agent in sim.agents:                           # id 昇順=決定論
        if agent.visitor:
            continue
        rng = sim.hub.stream("career", agent.id, step)
        if organizations.is_employee(agent):
            if layoff_p > 0.0 and rng.random() < layoff_p:      # 失業
                wage_daily = float(getattr(agent, "wage", 0.0))
                from_org = organizations.lay_off(agent)
                # 解雇規制の最小形(制度深化2 第10バッチ・既定 0/0=従来と完全同一):
                # 退職金=日給×severance_days(現金へ。wage source="severance")。
                # 不当解雇=unfair_ratio の割合(>0 のときだけ同 stream から追加 draw=既定は
                # draw 数も不変)で、生活不安(unemployment_grievance)を unfair_grievance_mult 倍。
                sev = wage_daily * float(cfg["severance_days"])
                if sev > 0.0:
                    agent.money += sev
                    sim.logger.log(Event(step=step, sim_min=sim_min,
                                         agent_id=agent.id, kind="wage",
                                         x=agent.x, y=agent.y,
                                         payload={"amount": round(sev, 1),
                                                  "balance": round(agent.money, 1),
                                                  "to": "cash",
                                                  "source": "severance"}))
                unfair = bool(float(cfg["unfair_ratio"]) > 0.0
                              and rng.random() < float(cfg["unfair_ratio"]))
                g = griev * (float(cfg["unfair_grievance_mult"]) if unfair else 1.0)
                factor_update.on_job_loss(agent, g, step=step, sim_min=sim_min,
                                          logger=sim.logger)     # 生活不安→不満(factors 経由)
                lost_payload = {"state": "lost", "org": from_org}
                if float(cfg["unfair_ratio"]) > 0.0:   # 既定 0 では payload も従来と同一
                    lost_payload["unfair"] = unfair
                _log_career(sim, agent, "unemployment", lost_payload, step, sim_min)
                _log_career(sim, agent, "job_change",
                            {"from_org": from_org, "to_org": None, "cause": "layoff"},
                            step, sim_min)
                agent.remember("勤め先を失った")
            elif switch_p > 0.0 and rng.random() < switch_p:     # 転職(別 org へ)
                new_org = _pick_org(rng, getattr(agent, "org_id", None))
                if new_org is not None and book is not None:
                    from_org = organizations.switch_org(
                        agent, book[new_org], city=sim.city, commute_to_poi=commute)
                    _log_career(sim, agent, "job_change",
                                {"from_org": from_org, "to_org": new_org,
                                 "cause": "switch"}, step, sim_min)
                    agent.remember("別の職場に移った")
        elif organizations.is_laid_off(agent):
            if rehire_p > 0.0 and rng.random() < rehire_p:       # 求職→再就職
                new_org = _pick_org(rng, None)
                if new_org is not None and book is not None:
                    organizations.rehire(agent, book[new_org], city=sim.city,
                                         commute_to_poi=commute)
                    _log_career(sim, agent, "unemployment",
                                {"state": "hired", "org": new_org}, step, sim_min)
                    _log_career(sim, agent, "job_change",
                                {"from_org": None, "to_org": new_org,
                                 "cause": "rehire"}, step, sim_min)
                    agent.remember("新しい仕事が見つかった")


# ---------------------------------------------------------------- 健康・疲労・病気(後続波 H1)
def _health_on(sim) -> bool:
    """健康(疲労・病気・メンタル)が有効か。既定 OFF=新経路を一切通さない(バイト一致)。"""
    cfg = getattr(sim, "healthcfg", None)
    return bool(cfg and cfg["enabled"])


def _log_health(sim, agent, kind: str, payload: dict, step: int, sim_min: int) -> None:
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id, kind=kind,
                         x=agent.x, y=agent.y, payload=payload))


def _phase_health_tick(sim, step: int, sim_min: int) -> None:
    """毎step: 疲労ゲージの更新(活動で+、睡眠で回復。内部 transient・RNG不要・決定論)。

    health OFF なら完全 no-op(fatigue 不変・health_update 0 件=バイト一致)。ON でも fatigue_gain=0
    なら fatigue は動かず=実質 no-op。位置・活動が確定した _phase_move の後に呼ぶ(その step の
    疲労を発火閾値 _eff_thr が読む)。"""
    if not _health_on(sim):
        return
    cfg = sim.healthcfg
    for agent in sim.agents:
        health_mod.tick_fatigue(agent, cfg, step, sim_min, sim.logger)


def _phase_health(sim, step: int, sim_min: int) -> None:
    """日次境界: 病気の発症/回復・受診(新 stream "health")+ メンタル(慢性高 grievance→引きこもり)。

    決定論: 発症/回復/受診の確率は**新 stream "health"**(agent, step)から引く=既存 draw 順に挿入
    しない(ゴールデンと決定論の保護)。R1: health 機構は generate() を1回も足さず、発火判断に k・内面
    状態(構成概念)を食わせず、暦・config・新 stream・物理位置・grievance(k 非依存の state)のみ参照する。
    病気=欠勤(work_window 無効)+在宅(routine が自宅へ寄せる)で物理位置が変わり対面 co-location が
    変化しうる(career G5 / crowd G4 と同型)→ 呼数不変は compute_matched 下の k 不変性で担保する。
    受診は既存 spend(cat="medical")。メンタルの grievance 参照は health.py(CHECKED_DIRS 外)に閉じる。

    health OFF なら完全 no-op(illness/medical_visit/health_update 0 件・health stream も引かない=
    乱数消費不変=ゴールデンを守る)。来街者は街の外に生活基盤=対象外(economy/career と同型)。"""
    if not _health_on(sim):
        return
    day = sim_min // 1440
    if day == getattr(sim, "_health_day", -1):
        return
    sim._health_day = day
    cfg = sim.healthcfg
    medical_cost = float(cfg["medical_cost"])
    for agent in sim.agents:                           # id 昇順=決定論
        if agent.visitor:
            continue
        rng = sim.hub.stream("health", agent.id, step)  # 新 stream(既存 draw 順に影響しない)
        ill = health_mod.roll_illness(agent, cfg, rng, step)
        if ill is not None:
            _log_health(sim, agent, "illness", ill, step, sim_min)
            if ill["state"] == "onset":
                agent.remember("体調を崩した")
                if health_mod.roll_medical(cfg, rng):   # 受診 → 医療消費(spend)+ medical_visit
                    _spend(sim, agent, medical_cost, "medical", step, sim_min)
                    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                                         kind="medical_visit", x=agent.x, y=agent.y,
                                         payload={"node": agent.node,
                                                  "cost": round(medical_cost, 1)}))
            else:
                agent.remember("体調が回復した")
        # メンタル: 慢性高 grievance→引きこもり(health.py が grievance を読み withdrawn を更新)
        health_mod.update_mental(agent, cfg, step, sim_min, sim.logger)


# ---------------------------------------------------------------- 商業ダイナミクス(後続波 H3)
def _commerce_on(sim) -> bool:
    """商業(営業時間・動的価格・在庫)が有効か。既定 OFF=新経路を一切通さない(バイト一致)。"""
    return commerce_mod.enabled(sim)


def _phase_commerce(sim, step: int, sim_min: int) -> None:
    """毎step: 店舗の開閉遷移(営業時間)を検知し shop_state を記録する(既定OFF=no-op。後続波 H3)。

    営業時間は時刻の純関数(RNG不要・決定論)。遷移はカテゴリ単位=1日数件の sparse な世界イベント
    (agent_id=-1)。動的価格・在庫は購入時(_charge_meal / 建物内消費)に commerce.on_purchase で適用する
    (ここでは開閉遷移のログのみ)。commerce OFF なら完全 no-op(shop_state 0 件=バイト一致)。"""
    commerce_mod.tick_shop_state(sim, step, sim_min)


# ---------------------------------------------------------------- 都市・環境ショック(後続波 H4)
def _phase_disaster(sim, step: int, sim_min: int) -> None:
    """日次境界: 災害(台風/地震/大雪)・交通の遅延/運休・インフラ障害(停電/通信断/断水)(既定 OFF=no-op。H4)。

    生活を麻痺させる外生ショック=集団的 grievance の源。決定論: 発生/遅延/障害の確率判定は**新 stream
    "disaster"**(agent, step / world -1, step)から引く=既存 draw 順に挿入しない(ゴールデンと決定論の保護)。
    発生日指定(days)は暦不要の day-index からの純関数(乱数なし)。R1: disaster 機構は generate() を1回も足さず、
    発火判断に k・内面状態(構成概念)を食わせず、day-index・config・新 stream・物理位置のみ参照する。災害=在宅
    (routine が自宅へ寄せる)+運休(transit.has_service 停止=駅経由の退出/帰還を止める)で物理位置・移動が
    変わり対面 co-location が変化しうる(career G5 / crowd G4 / 健康 H1 / 世帯 H2 / 商業 H3 と同型)→ 呼数不変は
    compute_matched 下の k 不変性で担保する。grievance は factors hook(on_disaster)へ不透明 magnitude を渡す
    (engine は因子内部を読まない=no-fingerprint)。既存の摂動シナリオ(sim.scenario)とは独立に動く(挙動不変)。

    disaster OFF なら完全 no-op(disaster/transit_delay/infra_outage 0 件・transit.suspended も触らない・
    "disaster" stream も引かない=乱数消費不変=ゴールデンを守る)。位置が確定する _phase_wake_and_returns の
    前に呼ぶ(その step の帰還/退出の has_service が運休を読む)。"""
    disaster_mod.tick_day(sim, step, sim_min)


# ---------------------------------------------------------------- 観光・多言語・犯罪・治安(後続波 H5)
def _diversity_on(sim) -> bool:
    """観光・多言語・犯罪・治安(後続波 H5)が有効か。既定 OFF=新経路を一切通さない(バイト一致)。"""
    return diversity_mod.enabled(sim)


def _phase_diversity(sim, step: int, sim_min: int) -> None:
    """毎step: 街内の犯罪・迷惑行為(窃盗→被害者 money−+grievance / 迷惑→周囲 grievance。既定OFF=no-op。H5)。

    位置が確定した後(_phase_move / _phase_enforcement の後)に呼ぶ=co-location(同ノード)と近傍警察官の
    抑止(G3 執行との接続)を正しく判定する。決定論: 発生判定は**専用 stream "crime"**(agent, step)から
    引く=既存 draw 順に挿入しない(ゴールデンと決定論の保護)。R1: 犯罪機構は generate() を1回も足さず、
    発火判断に k・内面状態(構成概念)を食わせず、config・新 stream・物理位置(co-location)のみ参照する。
    窃盗=被害者 money− は物理位置(残高→行き先選択)を変えうる(career G5 / 商業 H3 と同型)→ 呼数不変は
    compute_matched 下の k 不変性で担保する。grievance は factors hook(on_crime)へ不透明 magnitude を渡す
    (engine は因子内部を読まない=no-fingerprint)。犯罪履歴は危険地帯(治安回避)として routine が読む。

    diversity OFF なら完全 no-op(crime/nuisance 0 件・money/grievance 不変・"crime" stream も引かない=
    乱数消費不変=ゴールデンを守る)。観光回遊(tourist_visit)・危険地帯回避は routine 側、伝播障壁は
    labeling 側、文脈注入は deliberate 側で処理する(いずれも OFF はバイト一致)。"""
    diversity_mod.tick_crime(sim, step, sim_min)


# ---------------------------------------------------------------- 宿泊・ホテル滞在(後続波 Wave L)
def _lodging_on(sim) -> bool:
    """宿泊・ホテル滞在(Wave L)が有効か。既定 OFF=新経路を一切通さない(バイト一致)。"""
    return lodging_mod.enabled(sim)


def _visitor_refresh_on(sim) -> bool:
    """来街者の財布補充(改善 P2 第9バッチ)が有効か。既定 OFF=補充なし(バイト一致)。"""
    eco = getattr(sim, "economy", None)
    return bool(eco and eco.get("enabled") and eco.get("visitor_refresh"))


def _maybe_lodge(sim, agent, step: int, sim_min: int) -> bool:
    """夜の帰宅退出(homing)の代わりにホテルへ向かわせる。lodging するなら True(caller は退出しない)。

    lodging.want_lodge(有効・visitor・homing・所持金・連泊上限・hotel POI 有・専用 stream "lodging" の
    抽選)が当選したときのみ、最寄り hotel POI へ経路を張る(既に最寄りホテル前なら即チェックイン)。
    経路が張れない(候補なし/到達不能)なら False=通常の退出へ後退する。R1: generate() を足さない。"""
    if not lodging_mod.want_lodge(sim, agent, step):
        return False
    hotel = lodging_mod.nearest_hotel(sim, agent.x, agent.y)
    if hotel is None:
        return False
    agent.exit_intent = False
    agent.homing = False
    agent.lodging_intent = True
    agent.lodging_node = hotel["node"]
    agent.lodging_poi = hotel.get("name") or "ホテル"
    agent.lodging_building = hotel.get("building")
    if agent.node == hotel["node"]:                # 既に最寄りホテル前 → 即チェックイン
        _lodging_checkin(sim, agent, step, sim_min)
        return True
    path, used_mode = sim.router.route(agent.node, hotel["node"], "walk")
    if len(path) < 2:                              # 到達不能 → 宿泊を諦めて通常退出へ
        agent.lodging_intent = False
        return False
    agent.route = path[1:]
    agent.edge_offset = 0.0
    agent.trip_mode = used_mode
    agent._pending_stay = 0
    agent._pending_activity = ""
    agent._ride_pending = None
    return True


def _lodging_checkin(sim, agent, step: int, sim_min: int) -> None:
    """ホテル到着 → チェックイン(lodging_checkin + spend cat="lodging")→ 建物内で就寝(既存 sleep 機構)。

    連泊数を1増やし(毎晩支払い)、ホテル建物があれば屋内へ入り、checkout_hour まで眠る(sleep_until を
    次の checkout_hour に合わせる)。就寝は睡眠フェーズ(_phase_lodging のチェックアウト)まで維持。"""
    cfg = sim.lodgingcfg
    price = float(cfg["price_per_night"])
    agent.lodging_intent = False
    agent.lodging = True
    agent.lodging_nights = int(getattr(agent, "lodging_nights", 0)) + 1
    bld_id = getattr(agent, "lodging_building", None)
    if bld_id and sim.city.has_building(bld_id):   # 実在のホテル建物があれば屋内滞在
        bld = sim.city.building(bld_id)
        agent.building = bld["id"]
        agent.floor = 1
        agent.x, agent.y = bld["centroid"]
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                         kind="lodging_checkin", x=agent.x, y=agent.y,
                         payload={"poi": agent.lodging_poi, "node": agent.lodging_node,
                                  "price": round(price, 1),
                                  "nights": agent.lodging_nights}))
    _spend(sim, agent, price, "lodging", step, sim_min)
    agent.sleeping = True                          # 就寝(既存 sleep 機構)。checkout_hour まで眠る
    agent.activity = ""
    agent.sleep_until = step + _steps_until_tod(sim_min, int(cfg["checkout_hour"]) * 60)
    agent.reflect_step = step + 1                  # 就寝直後の内省(自宅就寝・帰路退出と同格の k 処置。
    #                                                lodging の抽選は k を読まない=呼数の k 不変は保たれる)
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                         kind="sleep_start", x=agent.x, y=agent.y,
                         payload={"until_step": agent.sleep_until,
                                  "building": agent.building}))


def _phase_lodging(sim, step: int, sim_min: int) -> None:
    """毎step: 宿泊中のエージェントの checkout_hour チェックアウト(既定 OFF=no-op。Wave L)。

    チェックイン時に sleep_until を次の checkout_hour へ合わせてあるので、そこに達したら退館させる
    (lodging_checkout を記録し、ホテル建物を出て通常活動へ戻す)。sleeping を落とすので後段の
    _phase_wake_and_returns は二重に起こさない(sleeping=False で素通り)。連泊数(lodging_nights)は
    退出=帰宅時に _try_exit でリセットするので、翌晩また max_nights まで宿泊できる。決定論(RNG不要)。

    lodging OFF なら完全 no-op(lodging_checkout 0 件=バイト一致)。位置が確定する
    _phase_wake_and_returns の前に呼ぶ(退館直後の当日の行動を routine が担えるように)。"""
    if not _lodging_on(sim):
        return
    for agent in sim.agents:
        if not getattr(agent, "lodging", False):
            continue
        if step < int(getattr(agent, "sleep_until", 0)):
            continue
        agent.sleeping = False                     # チェックアウト(= 起床)
        agent.lodging = False
        agent.lodging_intent = False
        if agent.building:                         # ホテル建物を出る(路面ノードへ)
            bld = sim.city.building(agent.building)
            agent.building = None
            agent.floor = 0
            agent.node = bld["entrance"]
            agent.x, agent.y = sim.city.node_xy(agent.node)
        agent.activity = ""
        agent.stay_until = step + 1
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                             kind="lodging_checkout", x=agent.x, y=agent.y,
                             payload={"poi": getattr(agent, "lodging_poi", ""),
                                      "nights_stayed": int(getattr(agent, "lodging_nights", 0))}))


# ---------------------------------------------------------------- 内面本格版(後続波 H6)
def _phase_inner_life(sim, step: int, sim_min: int) -> None:
    """起動後1回: 長期目標・趣味を決定論で付与(long_goal を記録)。感情ラベルは _phase_drive が毎step 担う。

    inner_life OFF なら完全 no-op(付与なし・long_goal 0 件=バイト一致)。付与は決定論(乱数なし)で
    id 昇順・1回だけ(sim._inner_life_init フラグ)。R1: generate() を1本も足さず、k・内面状態(構成概念)を
    読まず、名簿(価値プロファイル/職業/traits)のみ参照する=呼数は k 非依存。目標は行動を長期的に
    方向づける keystone 駆動源(現状は1日計画のみ)。趣味は下位集団形成の核 + プロンプト文脈。"""
    if not inner_life_mod.enabled(sim):
        return
    if getattr(sim, "_inner_life_init", False):
        return
    sim._inner_life_init = True
    inner_life_mod.precompute(sim, step, sim_min)


# ---------------------------------------------------------------- 1 step
def run_step(sim, step: int) -> None:
    sim.budget.reset()
    sim.freedom_stats = {"choice_points": 0, "exercised": 0}  # 自由度観測(P2)の step 境界。OFF は L2 で列不在
    sim_min = sim.clock.sim_min(step)
    _ensure_orgs(sim)                              # 組織台帳の遅延初期化(既定OFF=no-op)
    for agent in sim.agents:
        agent.now_step = step                      # remember() の時刻付け
    _phase_inner_life(sim, step, sim_min)          # 起動後1回: 長期目標・趣味の付与(既定OFF=no-op。H6)
    _phase_calendar_weather(sim, step, sim_min)    # 日次境界: 当日の日付・天気(既定OFF=no-op)
    _phase_annual(sim, step, sim_min)              # 日次境界: 年中行事の確定・記録(既定OFF=no-op)
    _phase_schedule_gc(sim, step, sim_min)         # 日次境界: 過去予定の GC(既定OFF=no-op)
    _phase_relations_day(sim, step, sim_min)       # 日次境界: 関係の断絶/評判の風化(既定OFF=no-op)
    _phase_household(sim, step, sim_min)           # 日次境界: パートナー形成(既定OFF=no-op。後続波 H2)
    _phase_commerce(sim, step, sim_min)            # 毎step: 店舗の開閉遷移 shop_state(既定OFF=no-op。後続波 H3)

    sim.scenario.on_step(sim, step)                # 摂動シナリオ(baseline=即 return)
    _phase_government(sim, step, sim_min)          # 日次境界: 行政会計・公務員給与・給付(既定OFF)
    _phase_daily(sim, step, sim_min)               # 日次境界: 経済的逼迫の心理圧
    _phase_career(sim, step, sim_min)              # 日次境界: 失業/求職/転職(既定OFF=no-op。Wave G5)
    _phase_health(sim, step, sim_min)              # 日次境界: 病気の発症/回復・受診・メンタル(既定OFF=no-op。H1)
    _phase_disaster(sim, step, sim_min)            # 日次境界: 災害・交通遅延/運休・インフラ障害(既定OFF=no-op。H4)
    _phase_rules(sim, step, sim_min)               # 日次境界: 制度DSL(期限失効・定期イベント)
    _phase_recursion(sim, step, sim_min)           # 日次境界: 再帰性(昨日の街の動き。既定OFF=no-op)
    _phase_assembly(sim, step, sim_min)            # 日次境界: 代表制議会の改選(既定OFF=no-op。制度深化3)
    _phase_reflect_day(sim, step, sim_min)         # 日次境界: 無意識層+衝撃ゲージのリセット(既定OFF=no-op。第12バッチ)
    _phase_freedom_day(sim, step, sim_min)         # 日次境界: 価値充足の減衰(既定OFF=no-op。第17バッチ)
    _phase_status(sim, step, sim_min)              # 日次境界: 社会的地位スコアの再計算(既定OFF=no-op。第11バッチ)
    _phase_world_events(sim, step, sim_min)
    _phase_lodging(sim, step, sim_min)             # 宿泊のチェックアウト(checkout_hour。既定OFF=no-op。Wave L)
    _phase_wake_and_returns(sim, step, sim_min)
    _phase_planning(sim, step, sim_min)            # 起床/帰還の直後 step に朝の一日計画
    _phase_tools(sim, step, sim_min)               # イベント開始で会場へ発つ→次の移動で反映
    _phase_move(sim, step, sim_min)
    _phase_traffic(sim, step, sim_min)
    _phase_enforcement(sim, step, sim_min)         # 執行ルート: 警察官が近傍の違反者を執行(既定OFF)
    _phase_diversity(sim, step, sim_min)           # 毎step: 犯罪・迷惑行為(近傍警察官で抑止。既定OFF=no-op。後続波 H5)
    _phase_health_tick(sim, step, sim_min)         # 疲労ゲージの毎step更新(既定OFF=no-op。後続波 H1)
    street_mod.phase(sim, step, sim_min)           # 街頭広告の視認判定(既定OFF=no-op。第18バッチ)
    # 位置が確定したこの時点で空間索引を1回だけ張る。以降の _phase_drive/_decide の
    # 知覚判定を近傍9セルだけの走査にする(全対全 O(n²) の回避)。_apply は位置が
    # 動くので索引を使わず live 走査(perception.py の設計注記を参照)。
    sim.percept_index = build_index(
        sim.agents, float(sim.cfg.world.perception_radius_m))
    worldview_mod.phase(sim, step, sim_min)        # 主観的世界モデル: 期待の検証と更新(既定OFF=no-op。第20バッチ)
    _phase_drive(sim, step, sim_min)               # 欲求→申請→抽選→発火権

    active = [a for a in sim.agents
              if a.loc != "outside" and not a.sleeping]   # 外・睡眠中=計算しない
    actions = [(agent, _decide(sim, agent, step, sim_min)) for agent in active]
    for agent, action in actions:
        _apply(sim, agent, action, step, sim_min)
    _phase_jitter(sim, step, sim_min)              # 路上滞在の完全静止を解消(微移動)
    _phase_crowd(sim, step, sim_min)               # 群集(大規模行事型)の集中を観測(既定OFF=no-op)
    infoenv_mod.phase(sim, step, sim_min)          # 情報環境: バイラル加重・誤情報/炎上(既定OFF=no-op。Wave G6)

    for agent in sim.agents:      # 内省(就寝直後 or 来街者の帰路。個別時刻に分散)
        if agent.reflect_step == step:
            maybe_reflect(agent, step=step, sim_min=sim_min,
                          writeback=str(sim.cfg.k.writeback),
                          alpha=float(sim.cfg.k.degraded_alpha), llm=sim.llm,
                          place_name="自宅",
                          city_name=getattr(sim, "place_name", ""),
                          rng=sim.hub.stream("writeback", agent.id, step),
                          logger=sim.logger,
                          max_tokens=int(sim.cfg.model.reflect_max_tokens),
                          think=bool(sim.cfg.model.reflect_think),
                          controls=getattr(sim, "controls_mode", "none"),
                          agentic_pull=getattr(sim, "agentic_pull", False),
                          date_line=getattr(sim, "today_date_line", None),
                          weather_line=getattr(sim, "today_weather_line", None),
                          reflect_cfg=getattr(sim, "reflectcfg", None),
                          reflect_variety=bool(getattr(sim, "promptscfg", {})
                                               .get("reflect_variety", False)))

    sim.logger.log_metrics(step, collect(sim))
    if step % int(sim.cfg.observer.snapshot_every) == 0:
        # 既定(drift/accounts OFF)は下の dict がそのまま=L3 バイト一致。ON 時のみ追記。
        drift_on = sim.drivecfg["drift"]["enabled"]
        accounts_on = _accounts_on(sim)
        # 覚醒度は affect 有効 かつ gain>0 のときだけ snapshot に載せる(既定は載せない=L3 バイト一致)。
        arousal_on = _affect_on(sim) and sim.affectcfg["arousal_gain"] > 0.0
        hierarchy_on = status_mod.enabled(sim)     # 地位機構 ON のときだけ status を載せる(既定=載せない)
        snap = []
        for a in sim.agents:
            d = {"id": a.id, "x": round(a.x, 1), "y": round(a.y, 1),
                 "node": a.node, "loc": a.loc, "sleeping": a.sleeping,
                 "building": a.building, "floor": a.floor,
                 "activity": a.activity,
                 "money": round(a.money, 1),
                 "opinion": round(a.opinion, 4),
                 "states": {k: round(v, 4) for k, v in a.states.items()},
                 "adopted": sorted(a.adopted)}
            if drift_on:
                d["theta_drift"] = round(a.theta_drift, 4)
            if accounts_on:
                d["account"] = round(a.account, 1)
            if arousal_on:
                d["arousal"] = round(a.arousal, 4)
            if hierarchy_on:
                d["status"] = round(a.status, 4)
            snap.append(d)
        sim.logger.log_snapshot(step, {"agents": snap})
