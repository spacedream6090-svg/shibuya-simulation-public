"""1 step の実行(フェーズ制・agent_id 昇順適用 = 決定論 D13)。

フェーズ: 起床/帰還 → 移動(道なり・経路ポリラインをログ) → 意思決定 →
行動適用 → 経験→state(seam) → 就寝直後の内省(個別分散) → 観測。
- 範囲外(loc=outside)と睡眠中は計算しない/知覚されない。
- ★発話は必ず LLM(routine の socialize → deliberate)。定型文なし(2026-07-04)。
engine は因子名を一切参照しない(no-fingerprint、tests/test_contracts.py で担保)。
"""
from __future__ import annotations

import math

from .. import ablate as ablate_mod
from .. import aging as aging_mod
from .. import annual as annual_mod
from .. import assets as assets_mod
from .. import attention as attention_mod
from .. import chance as chance_mod
from .. import commerce as commerce_mod
from .. import conversation as conversation_mod
from .. import disaster as disaster_mod
from .. import diversity as diversity_mod
from .. import dunbar as dunbar_mod
from .. import economy_sfc as sfc_mod
from .. import envfeedback as envfb_mod
from .. import freedom_p2 as freedom_p2_mod
from .. import gossip as gossip_mod
from .. import goods as goods_mod
from .. import health as health_mod
from .. import home_awake as home_awake_mod
from .. import household as household_mod
from .. import incidents_interpersonal as incidents_mod
from .. import inner_life as inner_life_mod
from .. import joint as joint_mod
from .. import lodging as lodging_mod
from .. import lost_property as lost_mod
from .. import medical as medical_mod
from .. import mind as mind_mod
from .. import mobility as mobility_mod
from .. import night as night_mod
from .. import opinion as opinion_mod
from .. import party as party_mod
from .. import physics as physics_mod
from .. import population as population_mod
from .. import provlink as provlink_mod
from ..net import contact_formation as contact_mod
from ..observer import causality as causality_mod
from ..observer import decision_mode as decmode_mod
from ..observer import gt_extras as gt_extras_mod
from ..observer import starvation as starvation_mod
from .. import reject as reject_mod
from .. import relations as relations_mod
from .. import relations_endo as relations_endo_mod
from .. import rumors as rumors_mod
from .. import pov as pov_mod
from .. import b2b as b2b_mod
from .. import city_ops as city_ops_mod
from .. import delivery as delivery_mod
from .. import devices as devices_mod
from .. import facility_devices as facilities_mod
from .. import incidents_env as incidents_env_mod
from .. import services as services_mod
from .. import status as status_mod
from .. import street as street_mod
from .. import street_life as street_life_mod
from .. import traces as traces_mod
from .. import transit_interior as transit_interior_mod
from .. import transit_live as transit_live
from .. import transit_staff as transit_staff_mod
from .. import truth_ledger as truth_ledger_mod
from .. import work as work_mod
from .. import worldview as worldview_mod
from ..net import infoenv as infoenv_mod
from ..agents.ref import AgentRef
from ..cognition import deliberate, drive, planning, routine
from ..cognition import age_cog as age_cog_mod
from ..cognition import day_plan as day_plan_mod
from ..cognition import engaged as engaged_mod
from ..cognition import fire as fire_mod
from ..cognition import perception_contract as contract_mod
from ..cognition import plan_boundary as boundary_mod
from ..cognition import plasticity as plasticity_mod
from ..cognition import prompt_p1 as prompt_p1_mod
from ..cognition import watch as watch_mod
from ..cognition import reflect_timing as reflect_timing_mod
from ..cognition import reflection as reflection_mod
from ..cognition.reflection import maybe_reflect
from ..economy import CIVIL_SERVANTS, civil_servant_pay, gig_profile, price_of
from .. import economy as economy_mod
from ..government import Government, build_government_cfg
from .. import organizations, schedule, weather
from ..world import calendar
from ..world import floors as floors_mod
from ..world import indoor as indoor_mod
from ..world import indoor_flow as indoor_flow_mod
from ..world import presence as presence_mod
from ..world import pool as pool_mod
from ..world.clock import STEP_MINUTES
from ..world import scene_desc as scene_desc_mod
from ..world import vision as vision_mod
from ..lang.sentiment import valence
from ..factors import affect
from ..factors import update as factor_update
from ..observer.aggregate import collect
from ..observer.schema import Event
from ..rules import apply_bonus
from ..world.geom import rdp
from ..world import perception as perception_mod
from ..world.perception import (build_index, cell_m_of, count_hearers,
                                hearers_of, salience_gate)


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
             valence: float, step: int, sim_min: int,
             magnitude: float = 1.0) -> None:
    """交流1件を actor→other の関係台帳へ記録する。relations OFF なら従来の record_contact
    のみ(closeness を付けない=バイト一致)。ON なら符号つき closeness 更新 + tier 変化ログ。

    magnitude(第65バッチ)は _quality_mag が返す不透明係数(既定 1.0=従来と同値)。増減の
    **量**にのみ載る=engine はここ以外の判定(発火・相手選択・段階)へ一切流さない。

    第75バッチ(ダンバー認知枠。relations.dunbar 既定 OFF=hook は即 return=バイト一致):
    記録**前**に on_contact を呼ぶ(相手が休眠中なら割引つき再会=relation_rekindle)。
    note_contact より前でなければならない(後だと当該交流分の closeness 増分が復元値で潰れる)。
    上限の適用は日境界の 1 回に一本化してある(dunbar.enforce の docstring=毎接触だと休眠/再会が
    振動する)。engine は「呼ぶだけ」で層の値・休眠規則を一切知らない(society/dunbar.py に閉じる)。"""
    if _relations_on(sim):
        dunbar_mod.on_contact(sim, actor, other_id, step, sim_min)
        relations_mod.note_contact(actor, other_id, other_name, text, valence,
                                   sim.relationscfg, step, sim_min, sim.logger,
                                   magnitude)
    else:
        actor.mem.record_contact(other_id, other_name, step, text)


def _quality_mag(sim, speaker, other_id: int, text: str, sim_min: int) -> float:
    """交流1件に載せる不透明係数を society 層から受け取る(既定 OFF/relations OFF は 1.0
    =従来と同値=バイト一致)。既生成テキストからの決定論抽出のみ=LLM 呼・乱数ともゼロ。

    engine は数値を運ぶだけで中身(何を材料にどう測るか)を知らない。片方向 hook:
    戻り値は _contact 以外へ渡さない=発話の発生・相手選択・イベント列は ON/OFF で不変。"""
    if not _relations_on(sim):
        return 1.0
    return relations_endo_mod.contact_magnitude(sim, speaker, other_id, text,
                                                sim_min)


def _steps_until_tod(cur_sim_min: int, target_min: int,
                     step_minutes: int = STEP_MINUTES) -> int:
    """現在の sim_min から、次に time-of-day=target_min になるまでの step 数(正)。

    流入通勤者の帰宅(夕)→翌朝の到着(arrival_min)までの外部滞在を決める。二峰分布の
    jitter は arrival_min 自体に内包済みなので、ここは決定論(乱数なし)。

    A1(第94バッチ OBS-U2): 分 → step 換算は Δt に依存する量なので `clock.min_to_steps`
    と同じ式(`分 // step_minutes`)を使う。既定 Δt=10 では `delta // 10` と厳密に同値
    (整数除算・同一被除数)= 従来と 1 ビットも変わらない。直書きのままだと Δt=1 で
    外部滞在・宿のチェックアウト待ちが実時間 1/10 に縮む。"""
    delta = (int(target_min) - cur_sim_min % 1440) % 1440
    if delta == 0:
        delta = 1440                                   # 同時刻 = 次の周回(翌日)
    return delta // int(step_minutes)


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


# -------------------------------------------------------------- 経済深化 E(第37バッチ配線)
# economy.py 冒頭 E セクションの純ロジック(consumption/payment/bank/vc・全て既定 OFF)を配線する。
# 全機能 OFF 時は新経路を一切通さず・新 stream("payment")も引かない=バイト一致(ゴールデン)。
# 判定は observables のみ(money/account/period_income/sales_total/occupancy/relations)= k 非依存(R1)。
def _consumption_on(sim) -> bool:
    """E-W3 消費行動が有効か(economy.consumption.enabled)。既定 OFF=budget 圧縮なし=消費額不変。"""
    econ = getattr(sim, "economy", {}) or {}
    return bool(econ.get("consumption", {}).get("enabled"))


def _payment_on(sim) -> bool:
    """E-W3 決済手段が有効か(economy.payment.enabled)。既定 OFF=spend payload に method を足さない。"""
    econ = getattr(sim, "economy", {}) or {}
    return bool(econ.get("payment", {}).get("enabled"))


def _bank_on(sim) -> bool:
    """E-W1 銀行が有効か(口座有効 かつ economy.bank.enabled)。融資/利息は口座前提。既定 OFF=完全 no-op。"""
    econ = getattr(sim, "economy", {}) or {}
    return bool(_accounts_on(sim) and econ.get("bank", {}).get("enabled"))


def _bank(sim):
    """銀行会計主体 Bank を遅延構築して返す(_gov と同型)。構築は乱数を引かず・ログもしない。"""
    bank = getattr(sim, "bank", None)
    if bank is None:
        bank = economy_mod.Bank(sim.economy["bank"])
        sim.bank = bank
    return bank


def _has_group(sim, agent) -> bool:
    """連帯保証グループに属すか(§3 社会的担保=与信スコア加点の条件)。tools 無し/未所属は False。"""
    tools = getattr(sim, "tools", None)
    if tools is None:
        return False
    return bool(tools.member_of.get(agent.id))


_BUDGET_CAT = {"taxi": "transport", "bus": "transport"}   # 消費カテゴリ→家計調査費目(§5)への写像


def _budget_amount(sim, agent, cat: str, base_amount: float) -> float:
    """E-W3 消費行動(既定 OFF): consumption.enabled なら個体の予算制約で base_amount を置換する。

    OFF は base_amount をそのまま返す(乱数を引かない=決定論=バイト一致)。budget_shares に無い
    カテゴリ(venture/fixed_cost 等)は budget_amount が base_amount を返す=不変(会計保存)。
    交通(taxi/bus)は budget_shares の transport 費目へ写像する。income=period_income(月収相当)。"""
    if not _consumption_on(sim):
        return base_amount
    ccfg = sim.economy["consumption"]
    bcat = _BUDGET_CAT.get(cat, cat)
    traits = getattr(agent, "traits", None) or {}
    income = float(getattr(agent, "period_income", 0.0))
    profile = economy_mod.consumption_profile(traits, income, ccfg)
    return economy_mod.budget_amount(profile, bcat, base_amount, income,
                                     float(sim.economy.get("fixed_cost_daily", 0.0)), ccfg)


def _maybe_loan(sim, agent, need: float, step: int, sim_min: int,
                *, income: float | None = None) -> float:
    """E-W1 融資(既定 OFF): 現金不足点で銀行融資を試みる。承認され need≤loan_limit なら need を
    口座へ入金し loan_grant をログして入金額を返す。OFF/非承認/上限超/既存融資あり/来街者は 0.0(不変)。

    与信は observables のみ(income=period_income / assets=money+account / arrears_days /
    グループ所属)= k 非依存(R1)。返済履歴は neutral 0.5。乱数を引かない=既存 draw 順を乱さない。
    income を明示で渡せる(家賃引落は period_income を 0 にした後に呼ぶため、控除前の値を渡す)。"""
    if not _bank_on(sim) or need <= 0.0 or agent.visitor:
        return 0.0
    bank = _bank(sim)
    if agent.id in bank.loans:                          # 返済中の融資が残るなら重複融資しない
        return 0.0
    bcfg = sim.economy["bank"]
    inc = float(agent.period_income if income is None else income)
    assets = float(agent.money) + float(agent.account)
    arrears = float(getattr(agent, "arrears_days", 0))
    score = bank.score(inc, assets, 0.5, arrears, _has_group(sim, agent))
    if not economy_mod.loan_approved(score, bcfg):
        return 0.0
    if float(need) > economy_mod.loan_limit(inc, bcfg):
        return 0.0
    day = sim_min // 1440
    loan = bank.grant(agent.id, float(need), score, day)
    agent.account += float(need)
    # IF-F W3: 与信を通したのは**銀行**(agent_id は借りた側 = 患者)。device_id=bank:main。
    devices_mod.log_device(
        sim, Event(step=step, sim_min=sim_min, agent_id=agent.id,
                   kind="loan_grant", x=agent.x, y=agent.y,
                   payload={"amount": round(float(need), 1),
                            "rate": round(loan["rate"], 4),
                            "term_days": loan["term_days"],
                            "score": round(score, 4),
                            "account": round(agent.account, 1),
                            "total_due": round(loan["total_due"], 1)}),
        devices_mod.DEV_BANK_MAIN)
    return float(need)


def _phase_bank_day(sim, step: int, sim_min: int) -> None:
    """E-W1 融資の日次返済フェーズ(bank ON 時のみ)。loan_due の融資を repay_installment で回収し
    口座から控除(loan_repay)。完済で loans から除去。延滞が default_arrears_days に達したら
    bank.write_off + 未回収残を家賃滞納(rent_due)へ積んで既存 accounts の破産サイクルへ接続する。
    OFF/融資なしは完全 no-op(loan_repay 0 件=バイト一致)。決定論(乱数なし)。

    IF-F W3: 回収・貸倒を決めたのは銀行なので 3 つの emit 点すべてに device_id=bank:main
    を刻む(agent_id は返済した / できなかった個体 = 患者)。窓(cause_scope)ではなく
    per-emit にしてあるのは、このフェーズが将来ほかの種を出したときに巻き込まないため。"""
    if not _bank_on(sim):
        return
    bank = _bank(sim)
    if not bank.loans:
        return
    day = sim_min // 1440
    if day == getattr(sim, "_bank_day", -1):
        return
    sim._bank_day = day
    bcfg = sim.economy["bank"]
    for aid in sorted(bank.loans):                      # id 昇順=決定論(sorted は snapshot=途中 pop 安全)
        loan = bank.loans[aid]
        # ★取り立ては**在場者にだけ**行う(``agent_by_id`` は退場者も返す = 幽霊)。退場中の
        #   個体は脱水済みで、``account -= paid`` を書いても hydrate で捨てられる = 銀行だけが
        #   受け取って街の総額が湧く。街に居ない間は**猶予**(next_due_day を進めず延滞も刻まない)
        #   = 翌日以降、再来街したその日に自動で再開する。
        agent = sim.present_agent(aid)
        if agent is None or not economy_mod.loan_due(loan, day):
            continue
        paid, status = economy_mod.repay_installment(loan, agent.account, day)
        if paid > 0.0:
            agent.account -= paid
            bank.receive(paid)
            devices_mod.log_device(
                sim, Event(step=step, sim_min=sim_min, agent_id=aid,
                           kind="loan_repay", x=agent.x, y=agent.y,
                           payload={"amount": round(paid, 1),
                                    "remaining": round(loan["remaining"], 1),
                                    "account": round(agent.account, 1),
                                    "status": status}),
                devices_mod.DEV_BANK_MAIN)
            if status == "complete":
                bank.loans.pop(aid, None)
            continue
        devices_mod.log_device(                          # 返済不能=延滞を記録
            sim, Event(step=step, sim_min=sim_min, agent_id=aid,
                       kind="loan_repay", x=agent.x, y=agent.y,
                       payload={"amount": 0.0,
                                "remaining": round(loan["remaining"], 1),
                                "account": round(agent.account, 1),
                                "arrears": int(loan["arrears_days"]),
                                "status": "arrears"}),
            devices_mod.DEV_BANK_MAIN)
        if economy_mod.loan_defaulted(loan, bcfg):      # 延滞閾値到達→貸倒→破産サイクルへ接続
            loss = bank.write_off(aid)
            agent.rent_due += loss                      # 未回収残を滞納へ=既存の破産処理が引き取る
            devices_mod.log_device(
                sim, Event(step=step, sim_min=sim_min, agent_id=aid,
                           kind="loan_repay", x=agent.x, y=agent.y,
                           payload={"amount": 0.0, "remaining": 0.0,
                                    "account": round(agent.account, 1),
                                    "status": "defaulted",
                                    "write_off": round(loss, 1)}),
                devices_mod.DEV_BANK_MAIN)


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
    # resume 時は配属(org_id/work_node)を checkpoint(agents pickle)から復元済み=再 attach しない。
    # 再 attach すると career/求職の転職(switch_org)を元配属へ潰し resume!=straight になる(第60バッチ b
    # で顕在化。book だけ再構築すれば book[new_org] 参照・求職マッチは動く)。fresh ランは従来どおり attach。
    if getattr(sim.logger, "_resumed", False):
        return
    personas_file = sim.cfg.get("agents", {}).get("personas_file")
    organizations.attach(sim.agents, sim.orgs, personas_file, city=sim.city,
                         commute_to_poi=bool(sim.orgscfg.get("commute_to_poi", False)),
                         assignments_path=sim.orgscfg.get("assignments"))


def _sfc_arm(sim, step: int, sim_min: int) -> None:
    """IF-E2(既定 OFF=即 return): 非エージェント残高を**先に**実体化してから org 預金を配る。

    ``Government`` / ``Bank`` / ``VCFund`` はどれも遅延構築(``_gov`` / ``_bank`` /
    ``tools._vc_fund``)で、初回アクセスの瞬間に conf の初期資本(区/都/国の予算・銀行資本・
    ファンド原資)を**世界へ突然出現させる**。案B の不変量「Σ(全主体残高)+RoW 累積=一定」は
    その出現を階段状のジャンプとして拾ってしまうので、ON のランでは step 先頭で 3 つとも
    実体化して**期首の基準に含める**。構築自体は乱数を引かず L1 も出さない(=挙動不変)。"""
    if not sfc_mod.enabled(sim):
        return
    _gov(sim)
    if _bank_on(sim):
        _bank(sim)
    tools = getattr(sim, "tools", None)
    vccfg = (getattr(sim, "economy", {}) or {}).get("vc") or {}
    if tools is not None and vccfg.get("enabled"):
        tools._vc_fund(sim, vccfg)
    sfc_mod.arm(sim, step, sim_min)


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
    # ★式(日給×revenue_margin)も発火条件も従来と 1 バイトも変えない。IF-E2 のために
    #   計算を log の**前**へ移しただけで、OFF では payload も led も従来と完全同一。
    d_rev = d_wage = 0.0
    if _economy_on(sim):
        wage = float(sim.economy["wages"].get(str(org.get("wage_tier", "")), 0.0))
        d_rev = wage * float(sim.orgscfg["revenue_margin"])
        d_wage = wage
    # IF-E2 案B(既定 OFF=0.0=payload 不変): 域内に客が居ない org(office/education)の輸出代金。
    exported = sfc_mod.on_production(sim, org, d_rev)
    prod_payload = {"org": str(org["id"]), "output": output, "kind": kind}
    if exported:                       # RoW → org の実入金(行列に載る唯一の観測点)
        prod_payload["revenue"] = round(float(exported), 1)
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                         kind="production", x=agent.x, y=agent.y,
                         payload=prod_payload))
    led = sim.org_ledger.setdefault(str(org["id"]), {"production_count": 0,
                                                     "revenue_est": 0.0,
                                                     "wage_paid": 0.0})
    led["production_count"] += 1
    if _economy_on(sim):
        led["revenue_est"] += d_rev
        led["wage_paid"] += d_wage
    if _org_ledger_on(sim):    # 会社観測データ層 B4: 日次アキュムレータへ同じ会計を積む(サイドカー用)
        ent = _org_day_entry(sim, org["id"])
        ent["production"] += 1
        ent["revenue_est"] += d_rev
        ent["wage_paid"] += d_wage
    b2b_mod.on_production(sim, org, step, sim_min)    # B2B ⑤: 卸/製造 org は生産で在庫が増える(既定OFF=no-op)


def _log_tax(sim, agent, tax: str, amount: float, to: str, base: float,
             step: int, sim_min: int) -> None:
    """税の徴収を記録(kind=tax)。所得税=income→nation / 住民税=resident→ward,metro /
    消費税=consumption→nation(国分),metro(地方分)。base=課税の元(名目賃金 or 名目価格)。

    IF-F W3: agent_id は**納税者**(= 金を取られた側 = 患者)であって行為者ではない。
    徴収したのは徴税という制度なので device_id=gov:tax を刻む(源泉徴収も消費税も
    この 1 関数を通るので、刻む場所は 1 箇所で足りる)。★行政フェーズ(gov:main の窓)の
    中で出る tax もここで明示した id が勝つ = 「会計の締め」ではなく「徴税」だと言える。"""
    devices_mod.log_device(
        sim, Event(step=step, sim_min=sim_min, agent_id=agent.id,
                   kind="tax", x=agent.x, y=agent.y,
                   payload={"tax": tax, "amount": round(float(amount), 1),
                            "to": to, "base": round(float(base), 1)}),
        devices_mod.DEV_GOV_TAX)


def _withhold_wage(sim, agent, gross: float, step: int, sim_min: int,
                   annual_income: float | None = None) -> tuple[float, float]:
    """賃金の源泉徴収: 所得税(→nation)+住民税(→ward6:metro4)を控除し (手取り, 税合計) を返す。

    税は該当予算に歳入計上し、tax イベントで記録。手取り = 名目 − 税(engine 側で口座/現金に入金)。
    annual_income(既定 None=既存の全呼び出し)は所得税の**年換算の分母**を支給周期に合わせて
    渡す口(賃金多様性 WAGE)。None のときは従来どおり「日給 × annual_workdays」= バイト一致。
    住民税は所得割の定率なので年換算に依存しない(掛け率は gross のまま)。
    """
    gov = sim.government
    it = gov.income_tax(gross, annual=annual_income)
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
    """消費税の内訳計上(価格は名目不変=内税)。国分→nation / 地方分→metro に歳入計上+tax 記録。

    ★B6(第117 レーンE): ``price`` は **実支払**(``_spend`` の床クリップ後に実際に減った額)
      であって名目ではない。名目に課税すると「残高不足で 100 円しか払えなかった客から
      120 円ぶんの税を取る」ことになり、``economy_sfc.on_spend`` が受け手へ配る
      ``実支払 − 消費税`` が負に落ちる(受け手の預金を減らす)。課税標準を実支払に揃えると
      ``sim.government.consumption_tax`` の同一式・同一引数から実現税額が出るので、
      **行政の歳入計上と受け手の配分が厳密に同額**になる。クリップの無い支払
      (実支払 == 名目 = ほぼ全件)では従来と 1 円も変わらない。"""
    national, local, _rate = sim.government.consumption_tax(price, cat)
    if national > 0:
        sim.government.collect("nation", national)
        _log_tax(sim, agent, "consumption", national, "nation", price, step, sim_min)
    if local > 0:
        sim.government.collect("metro", local)
        _log_tax(sim, agent, "consumption", local, "metro", price, step, sim_min)


def _pay_wage(sim, agent, amount: float, step: int, sim_min: int,
              source: str | None = None, fund_level: str | None = None,
              payer_org: str | None = None,
              annual_income: float | None = None) -> None:
    """賃金の支給(本業の勤務完遂・バイトのシフト完遂・自営の日銭・月給まとめ・公務員給与)。

    source を渡すと payload に載せる(自営の日銭は "gig"、月給は "salary"、公務員は "civil")。
    既定 None のときは載せない=口座 OFF・行政 OFF の本業/バイト wage payload は現行と完全同一。
    口座 ON(E5): 入金先を口座にする("to":"account")。家賃計算の元(period_income)も積む。
    行政 ON: 名目 gross から所得税+住民税を源泉徴収し**手取り(net)を入金**(payload.amount=手取り、
      追加キー gross/tax を載せる=手取り+税=名目)。fund_level 指定(公務員給与)なら gross を該当
      予算から歳出(expense)計上する(区職員=ward / 警察官・消防士=metro)。行政 OFF 時は
      gross=net で追加キーも出さない=既存 wage payload とバイト一致。
    IF-E2 案B(economy.org_accounting。既定 OFF=完全 no-op=payload バイト一致): payer_org を
      渡すと**その org の預金 gross を引き落とす**(不足は自動当座借越)。渡されない/台帳に無い
      ときは rest-of-world(域外の雇用主・域外クライアント)が払う。どちらの場合も payload に
      支払側を示す payer キーが 1 つ増える(= 個人→会社→個人 の追跡が org_id で繋がる)。
    annual_income(賃金多様性 WAGE。既定 None=既存の全呼び出しはバイト一致): 所得税の年換算を
      支給周期に合わせる口。月給まとめ・賞与は「その人の年収」を渡す(渡さないと日給前提の
      ×245 が掛かって最高税率になる)。日給・バイト・日銭は None のままが正しい。"""
    if amount <= 0:
        return
    # T4 自助努力(第52バッチ): 自力累積(skill)の賃金乗数を1箇所だけ適用。全 wage 源(本業/バイト/
    # 日銭 gig/月給/公務員)がこの唯一の支給点を通るので、乗数はここで一様にかかる=労働生産性の反映。
    # 既定 OFF / wage_coef=0.0 は乗数 1.0 → amount を一切触らない(会計不変=ゴールデン L1 バイト一致)。
    mult = services_mod.self_dev_wage_mult(sim, agent)
    if mult != 1.0:
        amount = float(amount) * mult
    gross = float(amount)
    tax_total = 0.0
    payer = None
    if _government_on(sim):
        if fund_level is not None:                     # 公務員給与は予算が出所(歳出)
            sim.government.expense(fund_level, gross)
        amount, tax_total = _withhold_wage(sim, agent, gross, step, sim_min,
                                           annual_income=annual_income)  # 手取り
    if fund_level is not None:                         # 公務員給与=行政が払う(既に歳出計上済み)
        payer = "government" if sfc_mod.enabled(sim) else None
    else:                                              # IF-E2: org / RoW が払う(既定 OFF=None)
        payer = sfc_mod.on_wage(sim, agent, gross, source, payer_org, step, sim_min)
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
    if payer is not None:                      # IF-E2(既定 OFF=キーなし): 支払側 org / RoW チャネル
        payload["payer"] = payer
    # IF-F W3: agent_id は**受け取った本人**(患者)。払った主体を device_id で名乗らせる。
    #   fund_level あり … 公務員給与 = 予算からの歳出 → gov:payroll
    #   payer_org あり  … 配属 org の預金からの支払い → org:<org_id>
    #   どちらも無い    … 本業/バイト/日銭/退職金/財布補充。**雇い主が emit 点に無い**ので
    #                     刻まない(欠測を偽の id で埋めない = 名簿の正直さ)。
    _wage_device = (devices_mod.DEV_GOV_PAYROLL if fund_level is not None
                    else (devices_mod.org_device_id(payer_org) if payer_org else None))
    event = Event(step=step, sim_min=sim_min, agent_id=agent.id,
                  kind="wage", x=agent.x, y=agent.y, payload=payload)
    if _wage_device is None:
        sim.logger.log(event)
    else:
        devices_mod.log_device(sim, event, _wage_device)


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
           chosen: bool = False, item: str | None = None,
           payee_node: str | None = None) -> None:
    """消費(食事・買い物・nightlife・taxi・bus)。残高は 0 未満にしない。

    口座 ON(E5): 金額 ≥ card_threshold は口座から(カード)。未満は現金で、現金が
    足りなければ最寄り ATM で自動引き出し(withdraw)してから支払う。
    chosen(P2 #7 buy・既定 False=既存呼び出しは payload 不変=バイト一致): LLM が発火時に
    能動選択した消費に payload へ chosen:true を添える(非発火の抽選消費と区別する観測用)。
    item(物流②・既定 None=既存呼び出しは payload 不変=バイト一致): 買った物(商品実体)を
    payload へ添える(会計不変=金額は変えない)。
    IF-E2 案B(economy.org_accounting。既定 OFF=完全 no-op=payload バイト一致): 支払の**その場で**
    受け手(org / venture / RoW)を台帳の静的索引で解決し、**実支払−消費税**を入金する
    (venture 既存経路と同じ流儀=支払者を減らし受取者を増やすのが同一操作。Caiani の deposit
    transfer)。payload に受け手を示す payee キーが 1 つ増える。
    payee_node(H2 医療・既定 None=既存呼び出しは 1 バイトも変わらない): 受け手が**払った人の
    居場所では決まらない**消費のために、支払先の場所を明示で渡す口。医療がその唯一の例で、
    受診・入院は自宅や路上で発火するため従来は受け手が解決できず黙って RoW へ漏れていた。"""
    if amount <= 0:
        return
    sfc_on = sfc_mod.enabled(sim)
    # ★B6(第117 レーンE): 残高スナップは **sfc の ON/OFF に依らず**取る。消費税の課税標準を
    #   実支払へ揃えるのに `before` が要るからで、算術だけ = 世界も L1 も 1 バイトも動かない。
    before = agent.money + float(getattr(agent, "account", 0.0) or 0.0)
    if not _accounts_on(sim):
        agent.money = max(0.0, agent.money - amount)
        payload = {"amount": round(float(amount), 1),
                   "balance": round(agent.money, 1), "cat": cat}
        if chosen:
            payload["chosen"] = True
        if item is not None:
            payload["item"] = item
        # 実支払 = 床クリップ後の実際の減少額(名目 amount とは残高不足のときだけ食い違う)
        actual = before - (agent.money + float(getattr(agent, "account", 0.0) or 0.0))
        if sfc_on:                              # 受け手へ入金(実支払を基準に配る)
            payee = sfc_mod.on_spend(sim, agent, amount, actual, cat, step, sim_min,
                                     payee_node=payee_node)
            if payee is not None:
                payload["payee"] = payee
            if abs(actual - float(amount)) > 1e-9:   # 床クリップで名目より少なく払った(正直開示)
                payload["paid"] = round(actual, 1)
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                             kind="spend", x=agent.x, y=agent.y, payload=payload))
        if _government_on(sim):                 # 消費税を内訳計上(★課税標準=実支払。B6)
            _record_consumption_tax(sim, agent, actual, cat, step, sim_min)
        return
    acc = sim.economy["accounts"]
    method = None
    use_account = amount >= float(acc["card_threshold"]) and agent.account >= amount
    if _payment_on(sim):                       # E-W3 決済(既定 OFF=method なし・card_threshold 分岐不変)
        method = economy_mod.choose_payment(
            float(amount), economy_mod.payment_pref(getattr(agent, "traits", None) or {}),
            sim.hub.stream("payment", agent.id, step), sim.economy["payment"])
        use_account = (method == "cashless") and agent.account >= amount   # cashless=口座
    if use_account:
        agent.account -= amount                # 口座から(カード/キャッシュレス)
        src = "card"
    else:
        if agent.money < amount:               # 現金不足 → ATM 引き出し
            _atm_withdraw(sim, agent, amount - agent.money, step, sim_min)
        agent.money = max(0.0, agent.money - amount)
        src = "cash"
    payload = {"amount": round(float(amount), 1),
               "balance": round(agent.money, 1), "cat": cat,
               "src": src, "account": round(agent.account, 1)}
    if method is not None:                      # 決済 ON のみ payload に method(貨幣は新生しない=会計不変)
        payload["method"] = method
    if chosen:
        payload["chosen"] = True
    if item is not None:                        # 物流②: 買った物(会計不変=金額は変えない)
        payload["item"] = item
    # 実支払 = 現金 + 口座の実減少額(ATM 引き出しは内部移動なので相殺されて効かない)
    actual = before - (agent.money + float(getattr(agent, "account", 0.0) or 0.0))
    if sfc_on:                                  # IF-E2: 受け手へ入金(実支払を基準に配る)
        payee = sfc_mod.on_spend(sim, agent, amount, actual, cat, step, sim_min,
                                 payee_node=payee_node)
        if payee is not None:
            payload["payee"] = payee
        if abs(actual - float(amount)) > 1e-9:  # 床クリップで名目より少なく払った(正直開示)
            payload["paid"] = round(actual, 1)
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                         kind="spend", x=agent.x, y=agent.y, payload=payload))
    if _government_on(sim):                     # 消費税を内訳計上(★課税標準=実支払。B6)
        _record_consumption_tax(sim, agent, actual, cat, step, sim_min)


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
        # 賃金多様性 WAGE(既定 OFF=`_wage_covered` が常に False=以下は現行と 1 バイトも
        #   変わらない): プランを持つ被用者の**本業の賃金は日次清算フェーズが唯一の支給点**
        #   なので、ここでは 1 円も払わず勤務日も積まない(二重支給の禁止)。バイト
        #   (part_time)の支給は WAGE の対象外なので従来どおり下の経路を通る。
        # 口座 ON(E5): 月給者(本業日給を持つ会社員・店員)は日割り支給をやめ、勤務日数を
        #   積んで給料日にまとめ支給する。バイト・日銭は従来どおり都度(口座へ)入金。
        if main_done and _wage_covered(sim, agent):
            pass
        elif main_done and _accounts_on(sim) and agent.wage > 0:
            agent.work_days += 1
        else:
            # IF-E2(既定 OFF=引数は無視される): 本業の賃金は配属 org の預金から出る。
            # バイト(part_time)は org でない職場なので払い手は RoW(域外/未帰属の雇用主)。
            _pay_wage(sim, agent, agent.wage if main_done else float(pt["pay"]),
                      step, sim_min,
                      payer_org=(getattr(agent, "org_id", None) if main_done else None))
    if main_done and _orgs_on(sim):     # 組織: 勤務完遂→産出 / 登校完遂→学習(既定 OFF)
        _log_org_output(sim, agent, step, sim_min)


def _charge_meal(sim, agent, step: int, sim_min: int) -> None:
    """飲食 POI 到着時に代金を支払う(食事/nightlife)。

    商業ダイナミクス(H3)ON 時は需要(在館数)に応じて価格係数を掛け(price_change)、需要集中で
    品切れ/行列なら購入を抑制する(stock_out+不満)。commerce OFF なら基準価格をそのまま払う(不変)。

    価格の合成順(PRICE-B。既定=空/OFF は恒等=1 ビットも変わらない):
      基準価格 → 在館連動係数(既存) → 事前公表の時間帯料金表 × 閉店前見切り → 消費行動の予算。
    購入が**成立した**あとに、その場の混み具合を不満へ写す(CRWD。既定 OFF=no-op)。"""
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
    amount = commerce_mod.apply_price(sim, amount, cat, agent.node, sim_min)
    item = None
    if _goods_on(sim):                             # 物流①②: 実在庫を1単位消費・買った物を付与(決定論)
        ok, item = goods_mod.on_purchase(sim, agent, cat, step, sim_min)
        if not ok:                                 # 実在庫の品切れ → 購入不成立(spend を出さない)
            return
    amount = _budget_amount(sim, agent, cat, amount)   # E-W3 消費行動(既定 OFF=不変)
    _spend(sim, agent, amount, cat, step, sim_min, item=item)
    commerce_mod.apply_crowding(sim, agent, cat, step, sim_min)


def _services_on(sim) -> bool:
    """サービスの実体(来店+効果。第46バッチ ③)が有効か。既定 OFF=新経路を一切通さない(バイト一致)。"""
    return services_mod.enabled(sim)


def _charge_service(sim, agent, step: int, sim_min: int) -> None:
    """サービス POI 到着時の受給(第46バッチ ③): 課金(_spend)+ 効果(factors on_service)+ 記憶 + service_use。

    経済無効なら受給しない(_charge_meal と同型)。実体は services.charge_service に閉じる(engine は
    サービス名・効果・因子を書かない=no-fingerprint)。課金は既存 _spend 経路(会計不変)。RNG は引かない。

    受給が**成立した**ときだけ、その場の混み具合を不満へ写す(CRWD。既定 OFF=no-op=バイト一致)。"""
    if not _economy_on(sim):
        agent._service_pending = None
        return
    if services_mod.charge_service(sim, agent, step, sim_min, _spend):
        commerce_mod.apply_crowding(sim, agent, services_mod.spend_cat(sim),
                                    step, sim_min)


def _charge_ride(sim, agent, ride: dict, step: int, sim_min: int) -> None:
    """交通機関(タクシー/簡易バス)の到着時課金。乗車を ride で記録し、運賃を spend。

    cat は乗り物の種別(taxi/bus)。経済が無効なら _spend が no-op(残高は 0 のまま)。"""
    mode = str(ride.get("mode", "taxi"))
    fare = float(ride.get("fare", 0.0))
    fare = _budget_amount(sim, agent, mode, fare)   # E-W3 消費行動(既定 OFF=不変。ride.fare と spend を整合)
    payload = {"mode": mode, "fare": round(fare, 1),
               "from": ride.get("from"), "to": ride.get("to")}
    # SUMO ライブ連成(v-Ride-1): 配車待ち/乗車時間/自由流超過を必ず記録(観測可能性=間接影響を
    # L2 で事後に測れる状態にする。traffic-indirect-effects.md §5 条件(5))。live OFF は付かない=バイト一致。
    # v-Ride-2 バス静的表: wait_s/ride_s/delay_s(実ダイヤ近似)。v-Ride-3 相乗り: shared(同乗者数)。
    # いずれも OFF は ride に無い=payload に付かない=バイト一致。
    for _k in ("wait_s", "ride_s", "delay_s", "shared"):
        if _k in ride:
            payload[_k] = ride[_k]
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                         kind="ride", x=agent.x, y=agent.y, payload=payload))
    _spend(sim, agent, fare, mode, step, sim_min)


def _taxi_live_dispatch(sim, agent, dest, step: int, sim_min: int) -> None:
    """SUMO ライブ連成タクシー配車 v-Ride-1 の本体側フック(_apply move_to から呼ぶ)。

    既定 OFF(sim.taxi_live is None)は即 return=何も起きない=SUMO を起動しない=ゴールデン L1
    バイト一致。ON かつ乗車が taxi のときだけ:予約を SUMO へ注入 → 配車待ち wait_s/乗車時間 ride_s を
    取得 → 到着 step の追加待ち hold_steps へ量子化する。捕まらない(未配車)なら乗車を取消して徒歩
    フォールバック(taxi_unmatched を記録)。乗車判断そのものは routine._ride_extra(非LLM)のまま=
    LLM 呼数を1本も足さない・k/belief を配車へ入れない(no-fingerprint / R1)。"""
    tl = getattr(sim, "taxi_live", None)
    if tl is None:
        return
    ride = getattr(agent, "_ride_pending", None)
    if not ride or ride.get("mode") != "taxi":
        return
    fx, fy = sim.city.node_xy(agent.node)
    tx, ty = sim.city.node_xy(dest)
    res = tl.request(agent_id=agent.id, from_node=agent.node, to_node=dest,
                     from_xy=(fx, fy), to_xy=(tx, ty), step=step, sim_min=sim_min)
    if not res.get("matched"):                     # 捕まらない → 乗車取消・徒歩へフォールバック
        agent._ride_pending = None
        agent.trip_mode = "walk"
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                             kind="taxi_unmatched", x=agent.x, y=agent.y,
                             payload={"from": agent.node, "to": dest}))
        return
    ride["wait_s"] = res["wait_s"]                  # payload 観測(_charge_ride が記録)
    ride["ride_s"] = res["ride_s"]
    ride["delay_s"] = res["delay_s"]
    if "shared" in res:                             # v-Ride-3 相乗り(同乗者数)。単発時は付かない=v-Ride-1 同一
        ride["shared"] = res["shared"]
    hold = int(res["hold_steps"])
    if hold > 0:                                    # 配車待ち+超過を到着 step の追加待ちへ(車が来るまで動かない)
        agent._taxi_hold_until = step + hold


def _bus_ride_hold(sim, agent, dest, step: int) -> None:
    """実バスダイヤ静的表 v-Ride-2 の到着 step 反映(_apply move_to から呼ぶ)。

    既定 OFF(bus_table 無効=ride に wait_s が無い)は即 return=何も起きない=バイト一致。ON かつ
    バスの実ダイヤ近似のときだけ:次便待ち wait_s + 自由流超過(区間所要 − 直線自由流)を到着 step の
    追加待ち hold へ量子化し、観測用 delay_s を ride に足す(到着 step ≈ ceil((呼時刻+待ち+乗車)/step))。
    乱数・k/belief を一切見ない純幾何+純ダイヤ計算(no-fingerprint)。taxi live と同じ hold 機構を共有。
    ride["from"/"to"] は停留所 id(=地図ノードとは限らない)なので、自由流は agent.node→dest で測る。"""
    ride = getattr(agent, "_ride_pending", None)
    if not ride or ride.get("mode") != "bus" or "wait_s" not in ride:
        return
    speeds = sim.cfg.world.modes.speeds
    car_m_per_s = float(speeds["car"]) / sim.clock.step_seconds
    fx, fy = sim.city.node_xy(agent.node)
    tx, ty = sim.city.node_xy(dest)
    free_s = math.hypot(fx - tx, fy - ty) / max(1e-6, car_m_per_s)
    delay_s = max(0.0, float(ride["ride_s"]) - free_s)
    ride["delay_s"] = round(delay_s, 1)
    hold = transit_live.quantize_hold(float(ride["wait_s"]), delay_s,
                                      sim.clock.step_seconds)
    if hold > 0:
        agent._taxi_hold_until = step + hold


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
        at_pt = (agent.part_time and routine.in_part_time_window(
                     agent, sim_min, getattr(sim, "calendarcfg", None))
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
    # ablate.llm_off(第78): 朝の一日計画は LLM 呼なので撃たない(= planning.enabled=false と
    # 同じ状態。既存の「計画なし」経路は routine.decide が素で扱える=新しい分岐を作らない)。
    if not sim.planningcfg["enabled"] or ablate_mod.llm_off(sim):
        return
    bl = (sim.cfg.get("engine", {}) or {}).get("batch_llm", {}) or {}
    if bool(bl.get("enabled", False)):
        _phase_planning_batched(sim, step, sim_min,
                                workers=int(bl.get("workers", 8)))
        return
    if sim.budget.tiers:                           # DPH-B(既定 OFF=この分岐に入らない)
        _phase_planning_tiered(sim, step, sim_min)
        return
    for agent in sim.agents:
        if agent.plan_step != step:
            continue
        agent.plan_step = -1
        if agent.loc == "outside" or agent.sleeping:
            starvation_mod.note_plan_skipped(       # DPH-O ③(OFF は即 return=不変)
                sim, agent, step, sim_min,
                "outside" if agent.loc == "outside" else "sleeping")
            continue
        # 行間補間(P2 S2): 前回発火以降の客観ダイジェスト。OFF は None=注入せず不変。
        planning.make_plan(sim, agent, step, sim_min, _place_of(sim, agent),
                           interstitial_digest=_isl_take(sim, agent))


def _defer_first(agent, attr: str, step: int) -> int:
    """FIFO キューの「最初に予約された step」(初回は今の step)。"""
    first = int(getattr(agent, attr, -1))
    return step if first < 0 else first


def _phase_planning_tiered(sim, step: int, sim_min: int) -> None:
    """DPH-B: 朝の計画を**予算の中**で撃つ(既定 OFF=この関数は呼ばれない)。

    現行は「予約された step で撃つか、`sleeping`/`outside` なら永久に失う」の 2 択で、
    予算という概念すら無かった(= 予算外呼)。ここでは
      (1) `life` レーンの予約枠(足りなければ general の余り)を取れた個体だけが撃ち、
      (2) 取れなかった個体は **翌 step へ FIFO 繰り越し**(失わない)、
      (3) 繰り越しが `max_defer_steps` を超えたら **骨格計画へ落とす**(LLM ゼロ)
    の 3 段にする。順序は `(最初に予約された sim step, agent_id)` の全順序 = 決定論で、
    キューの実体は **agent の属性 1 つ**(`plan_due_step`)= sim 級の状態を新設しない
    (agents pickle に自然同梱される = checkpoint 追加作業ゼロ)。
    """
    cap = int(sim.budget.tiers["max_defer_steps"])
    due = [a for a in sim.agents if a.plan_step == step]
    due.sort(key=lambda a: (_defer_first(a, "plan_due_step", step), a.id))
    for agent in due:
        agent.plan_step = -1
        first = _defer_first(agent, "plan_due_step", step)
        waited = step - first
        if agent.loc == "outside" or agent.sleeping:
            agent.plan_due_step = -1
            starvation_mod.note_plan_skipped(
                sim, agent, step, sim_min,
                "outside" if agent.loc == "outside" else "sleeping", waited)
            continue
        if sim.budget.take("plan"):
            agent.plan_due_step = -1
            planning.make_plan(sim, agent, step, sim_min, _place_of(sim, agent),
                               interstitial_digest=_isl_take(sim, agent))
        elif waited >= cap:                        # 待たせすぎ → 骨格(LLM を 1 本も呼ばない)
            agent.plan_due_step = -1
            starvation_mod.note_plan_skipped(sim, agent, step, sim_min,
                                             "defer_cap", waited)
            day_plan_mod.install_skeleton(sim, agent, step, sim_min)
        else:                                      # 翌 step へ繰り越す(失わない)
            agent.plan_due_step = first
            agent.plan_step = step + 1


def _phase_planning_batched(sim, step: int, sim_min: int, workers: int) -> None:
    """朝計画の一括発行(P2 S6b・engine.batch_llm)。

    計画は個体間で独立(自分の状態・環境のみ参照・同 step の他者の計画に依存しない)
    ため、build を id 順に済ませてから未命中分だけを並行発行し、応答を id 順に適用する。
    イベント列・カウンタ・キャッシュ内容は逐次経路と完全同一(mock で test_batch_llm が
    バイト一致を固定)。実 LLM サーバでは並行発行が継続バッチングを充填し
    スループットが per-call レイテンシから解放される。
    """
    pending: list[tuple[object, dict]] = []
    tiers = sim.budget.tiers                       # DPH-B(既定 OFF=None=以下は従来どおり)
    cap = int(tiers["max_defer_steps"]) if tiers else 0
    order = sorted((a for a in sim.agents if a.plan_step == step),
                   key=lambda a: (_defer_first(a, "plan_due_step", step), a.id)) \
        if tiers else sim.agents
    for agent in order:
        if agent.plan_step != step:
            continue
        agent.plan_step = -1
        first = _defer_first(agent, "plan_due_step", step)
        if agent.loc == "outside" or agent.sleeping:
            if tiers:
                agent.plan_due_step = -1
            starvation_mod.note_plan_skipped(       # DPH-O ③(OFF は即 return=不変)
                sim, agent, step, sim_min,
                "outside" if agent.loc == "outside" else "sleeping",
                step - first if tiers else 0)
            continue
        if tiers:                                  # 予算の中で撃つ / 繰り越す / 骨格へ落とす
            if sim.budget.take("plan"):
                agent.plan_due_step = -1
            elif step - first >= cap:
                agent.plan_due_step = -1
                starvation_mod.note_plan_skipped(sim, agent, step, sim_min,
                                                 "defer_cap", step - first)
                day_plan_mod.install_skeleton(sim, agent, step, sim_min)
                continue
            else:
                agent.plan_due_step = first
                agent.plan_step = step + 1
                continue
        req = planning.build_plan_request(
            sim, agent, step, sim_min, _place_of(sim, agent),
            interstitial_digest=_isl_take(sim, agent))
        if req is not None:
            pending.append((agent, req))
    if not pending:
        return
    results = sim.llm.generate_many([r for _a, r in pending], workers=workers)
    for (agent, req), (response, call_id, cached) in zip(pending, results):
        planning.apply_plan_response(sim, agent, step, sim_min, req,
                                     response, call_id, cached)


def _phase_reflect_batched(sim, step: int, sim_min: int, workers: int) -> None:
    """夜内省の一括発行(P2 S6b・engine.batch_llm)。

    agentic pull 有効時は1体2呼(recall→本呼)が依存連鎖するため2ラウンドで発行する:
    ①発火ゲート(id順)→ recall 要求を一括発行 → 解決(観測出力は BufferSink に遅延)
    ②本呼要求を組んで一括発行 → id 順に「recall の遅延イベント→内省適用」の順で放出。
    = イベント列・カウンタは逐次実行と完全同一(test_batch_llm がバイト一致を固定)。
    個体間の独立性: 内省は自分の記憶・状態のみ参照(他者の同 step 内省に依存しない)。
    """
    wb = str(sim.cfg.k.writeback)
    alpha = float(sim.cfg.k.degraded_alpha)
    controls = getattr(sim, "controls_mode", "none")
    agentic_pull = getattr(sim, "agentic_pull", False)
    city = getattr(sim, "place_name", "")
    date_line = getattr(sim, "today_date_line", None)
    weather_line = getattr(sim, "today_weather_line", None)
    rcfg = getattr(sim, "reflectcfg", None)
    variety = bool(getattr(sim, "promptscfg", {}).get("reflect_variety", False))
    isl_on = _interstitial_on(sim)
    mt = int(sim.cfg.model.reflect_max_tokens)
    think = bool(sim.cfg.model.reflect_think)
    rfx_tag = reflect_timing_mod.context_tag_on(sim)     # RFX-O(既定 OFF=payload 不変)
    rfx_sleepy = bool(reflect_timing_mod.cfg_of(sim)["sleep_task_rewrite"])
    p1_on = prompt_p1_mod.enabled(sim)                   # V-P1(既定 OFF=プロンプト不変)
    p1_recall_t = prompt_p1_mod.recall_temperature(sim)  # 未設定(null)=従来の 0.7

    due: list[dict] = []
    for agent in _reflect_due(sim, step, sim_min):  # DPH-B(OFF は sim.agents=不変)
        if agent.reflect_step != step:
            continue
        # 逐次経路の引数評価と同順(rng 派生→ダイジェスト消費)を保つ
        rng = sim.hub.stream("writeback", agent.id, step)
        digest = _isl_take(sim, agent)
        st = reflection_mod.begin_reflect(agent, step=step, writeback=wb,
                                          controls=controls)
        if st is None:
            continue
        due.append({"agent": agent, "rng": rng, "digest": digest,
                    "discard": st["discard"], "recalled": [],
                    "recall_fail": None, "sink": None})
    if not due:
        return

    if agentic_pull:                     # ラウンド1: recall(常に+1呼=R1)
        rreqs = [reflection_mod.build_recall_request(
                     d["agent"], step=step, place_name=_reflect_place(sim, d["agent"]),
                     date_line=date_line, weather_line=weather_line,
                     city_name=city, p1=p1_on,
                     temperature=p1_recall_t) for d in due]
        rres = sim.llm.generate_many(rreqs, workers=workers)
        for d, (response, call_id, cached) in zip(due, rres):
            sink = reflection_mod.BufferSink()
            d["recalled"], d["recall_fail"] = reflection_mod.resolve_recall(
                d["agent"], step=step, sim_min=sim_min, response=response,
                call_id=call_id, cached=cached, sink=sink)
            d["sink"] = sink

    reqs = [reflection_mod.build_reflect_request(   # ラウンド2: 内省本体
                d["agent"], step=step, sim_min=sim_min,
                place_name=_reflect_place(sim, d["agent"]),
                date_line=date_line, weather_line=weather_line,
                reflect_cfg=rcfg, reflect_variety=variety,
                interstitial_digest=d["digest"], interstitial=isl_on,
                city_name=city, max_tokens=mt, think=think,
                recalled=d["recalled"], recall_fail=d["recall_fail"],
                # RFX-A / RFX-O(既定 OFF=None/False=従来定数とバイト一致)
                moment=reflect_timing_mod.when_of(d["agent"]),
                sleepy=rfx_sleepy, tag=rfx_tag,
                p1=p1_on)                        # V-P1(既定 OFF=従来とバイト一致)
            for d in due]
    results = sim.llm.generate_many(reqs, workers=workers)
    for d, req, (response, call_id, cached) in zip(due, reqs, results):
        if d["sink"] is not None:        # recall 系イベントを逐次と同じ並びで放出
            d["sink"].flush(sim.logger)
        reflection_mod.apply_reflect_response(
            d["agent"], step=step, sim_min=sim_min, writeback=wb, alpha=alpha,
            rng=d["rng"], logger=sim.logger, controls=controls, req=req,
            response=response, call_id=call_id, cached=cached,
            discard=d["discard"],
            gt_extras=gt_extras_mod.enabled(sim))   # G7-②(既定 OFF=キーなし)


def _reflect_place(sim, agent) -> str:
    """内省を書かせるときの場所名。**既定 mode では常に "自宅"**(従来とバイト一致)。

    RFX-A で夕方の路上・在宅メディア中に発火した個体は自宅に居るとは限らないので、
    そのときだけ実在の場所名(`_place_of`)を渡す。「眠りにつく前に」というタスク文と
    同じく、場所名が嘘になるのを避けるための 1 点(reflection-leisure-plan §2.5)。
    """
    if reflect_timing_mod.when_of(agent) is None:
        return "自宅"
    return _place_of(sim, agent)


def _reflect_due(sim, step: int, sim_min: int):
    """内省の対象者(DPH-B。既定 OFF では `sim.agents` をそのまま返す=バイト一致)。

    ON では朝の計画(`_phase_planning_tiered`)と**同じ 3 段**を内省へも当てる:
    予算が取れた個体だけを返し、取れなければ翌 step へ FIFO 繰り越し、
    `max_defer_steps` を超えたら諦めて 1 件記録する(内省には骨格に相当する退路が無い
    = 「撃たなかった」を捏造せず数える。DPH-O ③')。
    """
    tiers = sim.budget.tiers
    if not tiers:
        return sim.agents
    cap = int(tiers["max_defer_steps"])
    due = [a for a in sim.agents if a.reflect_step == step]
    due.sort(key=lambda a: (_defer_first(a, "reflect_due_step", step), a.id))
    out = []
    for agent in due:
        first = _defer_first(agent, "reflect_due_step", step)
        if sim.budget.take("reflect"):
            agent.reflect_due_step = -1
            out.append(agent)
        elif step - first >= cap:
            agent.reflect_due_step = -1
            agent.reflect_step = -1                # 消費済み(同じ step を二度見ない)
            starvation_mod.note_reflect_dropped(sim, agent, step, sim_min,
                                                step - first)
        else:
            agent.reflect_due_step = first
            agent.reflect_step = step + 1          # 翌 step へ繰り越す
    return out


def _phase_wake_and_returns(sim, step: int, sim_min: int) -> None:
    for agent in sim.agents:
        if agent.sleeping and step >= agent.sleep_until:
            agent.sleeping = False
            agent.stay_until = step + 1
            sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                                 kind="wake_up", x=agent.x, y=agent.y,
                                 payload={"slept_steps": agent.sleep_steps}))
            _schedule_plan(sim, agent, step, sim_min)   # 起床 → 朝の計画を予約
            # RFX-A(既定 mode="sleep" では no-op): 起床 → 「今日ぶんの内省」を 1 枚 armed に。
            # 予約の起点を務めの終わりへ前倒しするための唯一の据え付け点。
            reflect_timing_mod.on_wake(sim, agent, sim_min)
        if agent.loc == "outside" and step >= agent.return_at:
            via_station = (agent.return_gateway == sim.city.station_node)
            if via_station and not sim.transit.has_service(sim_min):
                continue                          # 終電後は帰れない(始発待ち)
            # 環境フィードバック 規則1(第84。既定 OFF=常に False=バイト一致): 電車が遅れている
            # ぶんだけ帰着が遅れる(1 回の帰還につき max_hold_steps を超えて待たせない=上限)。
            if via_station and envfb_mod.hold_return(sim, agent, step, sim_min):
                continue
            agent.loc = "street"
            agent.node = agent.return_gateway
            agent.x, agent.y = sim.city.node_xy(agent.node)
            agent.stay_until = step + 1
            # 計画駆動の圏外滞在(actor model P4。既定 OFF=常に None=payload バイト一致):
            # 圏外では出来事を 1 件も生成していないので、帰還のこの 1 点でだけ
            # 「計画どおり・特筆事項なし」の圧縮記憶を 1 行入れ、ブロックを消化済みにする
            # (定型文=場所と予定時刻の純関数・LLM ゼロ・乱数ゼロ)。
            payload = {"gateway": agent.return_gateway,
                       "via": "train" if via_station else "walk"}
            # 駅到着のパルス量子化(world.inflow_pulse。既定 OFF は属性自体が生えない
            # =この 2 行を通らない=payload バイト一致)。どの列車で来たかを L1 に残す
            # (新しい kind は 1 つも足さない = 既存 enter_area への追記のみ)。
            if hasattr(agent, "pulse_train_min"):
                payload.update({"train_min": agent.pulse_train_min,
                                "line": agent.pulse_line})
            bnd = boundary_mod.on_return(sim, agent, step, sim_min)
            if bnd is not None:
                payload.update(bnd)
            sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                                 kind="enter_area", x=agent.x, y=agent.y,
                                 payload=payload))
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
                        payload = {"amount": round(delta, 1),
                                   "balance": round(agent.money, 1),
                                   "to": "cash", "source": "home_refill"}
                        # IF-E2(既定 OFF=キーなし): 来街者の域内消費は地域会計では
                        # **サービスの輸出**(IRTS 2008 §4.21 / SNA §9.80)= RoW → 家計。
                        payer = sfc_mod.on_wage(sim, agent, delta, "home_refill",
                                                None, step, sim_min)
                        if payer is not None:
                            payload["payer"] = payer
                        sim.logger.log(Event(step=step, sim_min=sim_min,
                                             agent_id=agent.id, kind="wage",
                                             x=agent.x, y=agent.y, payload=payload))


# ---------------------------------------------------------------- 移動
def _phase_move(sim, step: int, sim_min: int) -> None:
    occupancy: dict[tuple[str, str], int] = {}
    for agent in sim.agents:
        # SUMO ライブ連成の配車待ち(_taxi_hold_until>step)は移動しない=占有にも数えない(既定 OFF:
        # フラグ未設定→getattr=-1→常に含める=バイト一致)。
        # P3 境界縫合(竹-4): 物理ゾーンが所有している個体もグラフ移動しない=占有にも数えない
        # (既定 OFF: _phys_zone 属性が生えない→getattr=None→owned()=False=バイト一致)。
        if agent.loc == "street" and not agent.sleeping and agent.route \
                and getattr(agent, "_taxi_hold_until", -1) <= step \
                and not physics_mod.owned(agent):
            key = _edge_key(agent.node, agent.route[0])
            occupancy[key] = occupancy.get(key, 0) + 1

    capacity = int(sim.cfg.world.edge_capacity)
    speeds = sim.cfg.world.modes.speeds
    elev = getattr(sim, "elevation", None)     # 3D Phase 0(無効時 None=payload 不変)
    for agent in sim.agents:
        agent._arrived_new = False
        # 配車待ち: 車が来る step(_taxi_hold_until)まで原地で待つ(車道は塞がない=移動なし)。
        # ON 時のみのフラグ=既定 OFF は素通り=バイト一致。到来 step 以降はフラグを解除して通常移動。
        if getattr(agent, "_taxi_hold_until", -1) > step:
            agent._congestion = 1.0
            continue
        # P3 境界縫合(竹-4): 物理ゾーンの所有下ではグラフ移動をしない(排他所有=同一時刻に
        # 2 つのモデルへ属さない)。位置は physics.phase() が既に据えてある。既定 OFF は素通り。
        if physics_mod.owned(agent):
            agent._congestion = 1.0
            continue
        if agent.loc != "street" or agent.sleeping or not agent.route:
            agent._congestion = 1.0
            continue
        key = _edge_key(agent.node, agent.route[0])
        count = occupancy.get(key, 0)
        factor = 1.0 if count <= capacity else max(0.3, capacity / count)
        agent._congestion = factor

        pts: list[tuple[float, float]] = [(agent.x, agent.y)]
        # P3 境界縫合(竹-4): この step の途中でゾーンを抜けた個体は、物理で使った秒数ぶん
        # グラフ側の予算を減らす(= 二重移動の防止)。既定 OFF は係数 1.0=バイト一致。
        budget_m = (float(speeds[agent.trip_mode]) * factor
                    * physics_mod.budget_scale(sim, agent, step))
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
        seg_payload = {"dist_m": round(moved, 1),
                       "mode": agent.trip_mode,
                       "congestion": round(factor, 3),
                       "pts": [[round(p[0], 1), round(p[1], 1)]
                               for p in pts]}
        if elev is not None:
            seg_payload["z"] = elev.height_at(agent.x, agent.y)
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                             kind="move_segment", x=agent.x, y=agent.y,
                             payload=seg_payload))
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
            elif agent.activity == "service" and _services_on(sim):  # サービス POI 到着 → 受給(第46バッチ ③)
                _charge_service(sim, agent, step, sim_min)
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
            arr_payload = {"node": agent.node,
                           "name": sim.city.node_name(agent.node),
                           "first_visit": first}
            if elev is not None:
                arr_payload["z"] = elev.height_at(agent.x, agent.y)
            sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                                 kind="arrive", x=agent.x, y=agent.y,
                                 payload=arr_payload))
            tools = getattr(sim, "tools", None)        # イベント参加確定/ビラ閲覧/屋台購入
            if tools is not None:
                tools.on_arrive(sim, agent, step, sim_min)
            if getattr(agent, "lodging_intent", False) \
                    and agent.node == agent.lodging_node:   # ホテル到着=チェックイン(Wave L。既定OFF=フラグ立たず no-op)
                _lodging_checkin(sim, agent, step, sim_min)
            night_mod.on_arrive(sim, agent, step, sim_min)  # 夜の避難先に到着=始発まで滞在(既定OFF=no-op)
            if agent.exit_intent:
                _try_exit(sim, agent, step, sim_min)   # homing はこの中で参照する
            if not agent.route:                        # 再ルート(終電後の徒歩帰宅)以外
                agent.homing = False


def _try_exit(sim, agent, step: int, sim_min: int) -> None:
    if _lodging_on(sim) and _maybe_lodge(sim, agent, step, sim_min):
        return                                     # 夜の帰宅の代わりにホテルへ(退出しない。Wave L)
    via_station = (agent.node == sim.city.station_node)
    if via_station and not sim.transit.has_service(sim_min):
        if night_mod.take_refuge(sim, agent, step, sim_min):
            return                                 # 終電後: 縁へ歩く代わりに夜の避難先へ(第101 III-1)
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
    # 装置層 P2 改札(actor model。既定 OFF=常に False=バイト一致): 改札は「目標を持たない
    # アクター」= 通路数 × 処理間隔(既定 60 人/分/通路)の待ち行列で、通れなければ駅に留まる。
    # ★envfeedback 規則2(入場規制)の**細粒度な置き換え**であり二重には効かない:
    #   装置 ON のとき envfeedback.update は規則2 を評価しない(devices.faregate_active)。
    if via_station and devices_mod.hold_exit(sim, agent, step, sim_min):
        return
    # 環境フィードバック 規則1/2(第84。既定 OFF=常に False=バイト一致): 遅延ぶん・入場規制中は
    # 改札を通れない=駅に留まる(exit_intent は維持。再試行は _phase_move 直後の pending_exits)。
    # 運行の有無(上)を先に見る=運休は遅延より上位の制約。待ちの合計は max_hold_steps で
    # 頭打ち=必ず通す安全弁があるので駅に溜まり続けない(T5)。
    if via_station and envfb_mod.hold_exit(sim, agent, step, sim_min):
        return
    agent.exit_intent = False
    agent.loc = "outside"
    agent.return_gateway = agent.node
    homing_exit = agent.homing
    # 計画駆動の圏外滞在(actor model P4。既定 OFF=常に None=以下は従来と完全同一):
    # 朝の計画ブロックが境界の時刻表 = 帰還 step は**ブロックの終了時刻**から決まる
    # ("outside" stream を 1 本も引かない)。統計駆動の流入通勤者(commute/arrival_min)
    # とは別の台帳で、payload の boundary 欄だけで機械的に分離できる。
    bnd = boundary_mod.on_exit(sim, agent, step, sim_min)
    if bnd is None:
        rng = sim.hub.stream("outside", agent.id, step)
        if homing_exit:                            # 来街者の帰宅: 睡眠+朝の移動で戻る
            agent.homing = False
            if _lodging_on(sim):
                agent.lodging_nights = 0           # 帰宅(範囲外退出)= 連泊カウントをリセット(Wave L)
            if getattr(agent, "commute", False) and agent.arrival_min >= 0:
                # 流入通勤者: 翌朝の到着時刻(arrival_min)に再流入(往復時刻を安定させる)
                agent.return_at = step + _steps_until_tod(sim_min, agent.arrival_min,
                                                          sim.clock.step_minutes)
            else:
                agent.return_at = (step + agent.sleep_steps
                                   + sim.clock.dur_steps(int(rng.integers(0, 7))))
            reflect_timing_mod.arm(sim, agent, step)   # 帰路の電車で今日を内省(k 処置)。
            #                                            RFX-A: 早期発火済みならこの 1 回を見送る
        else:
            span = sim.cfg.world.outside_steps
            agent.return_at = step + int(rng.integers(int(span[0]), int(span[1])))
    payload = {"gateway": agent.node, "homing": homing_exit,
               "via": "train" if via_station else "walk"}
    if bnd is not None:
        payload.update(bnd)
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                         kind="exit_area", x=agent.x, y=agent.y,
                         payload=payload))


# ---------------------------------------------------------------- 背景交通
def _phase_traffic(sim, step: int, sim_min: int) -> None:
    """エージェント以外の通過車両(実規模の車の流れ)。可視化用に折れ線を記録。

    world.traffic.mode=od のとき、車を個体化した OD 走行に切り替える(1度だけ遅延設定)。
    既定 ambient は現行と完全同一(payload の n/total/segs も不変。log_extra() は空を返す)。

    IF-F W3: 背景交通は**エージェントではない車の発生器**なので、causality ON のときだけ
    device_id=traffic:<mode> を刻む(ambient / od は別の装置 = 同じ id で呼ばない)。
    これが無いと traffic_flow は「1 step に 1 件・行為者不明」で -1 質量の最大種に居座る。
    """
    sim.traffic.ensure_mode(sim.cfg)
    segs = sim.traffic.step(step, sim_min)
    if not segs and not sim.traffic.enabled:
        return
    payload = {"n": len(segs), "total": sim.traffic.total_spawned,
               "segs": [s["pts"] for s in segs[:sim.traffic.max_log]]}
    payload.update(sim.traffic.log_extra())    # ambient={} / od={mode, cars}
    devices_mod.log_device(
        sim, Event(step=step, sim_min=sim_min, agent_id=-1,
                   kind="traffic_flow", x=0.0, y=0.0, payload=payload),
        devices_mod.traffic_device_id(sim.traffic.mode))


# ---------------------------------------------------------------- 共通: 聴取
def _hear_words(sim, listener, words: list[str], from_id: int, channel: str,
                step: int, sim_min: int) -> None:
    """語の聴取(対面/SNS/DM/検索/ニュース共通)。未知語は検索キューにも積む。"""
    if not words:
        return
    # ---- ablate.propagation_off(第78バッチ・既定 OFF=この分岐に入らない)----
    # **語彙(内容)は他エージェントから渡さない**が、「話しかけられた」という交流の**量**は
    # 従来どおり残す(= 会話量は不変で語彙授受だけが止まる = 専門化スコアの帰無モデル)。
    #
    # 判定は **チャネル名ではなく from_id >= 0**(= 送り手がエージェント)で行う。
    # チャネルは face / dm / sns / event / flyer と増えてきた歴史があり、名前の列挙は
    # 新チャネルが増えるたびに穴になる。「人から来たか、世界(媒体)から来たか」は
    # from_id の符号が構造的に表しているので、そちらを唯一の判定にする。
    #   from_id >= 0 … 他エージェント発(face/dm/sns 投稿/イベント/貼り紙)→ **遮断**
    #   from_id < 0  … 媒体発("news")・自分で調べた("search")→ 世界チャネルなので通す
    if from_id >= 0 and from_id != listener.id and ablate_mod.propagation_off(sim):
        if channel in ("face", "dm"):
            drive.add(listener, "addressed", sim.drivecfg)
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
            # ★レーン乙 ブロック7: 造語の作者は「任意に古い」= 退場済みである確率が高い。
            #   ``agent_by_id`` は退場者も保持するので ``is None`` が真にならず、
            #   factors/drive/評判の更新が捨てられる実体へ書かれ、D9 ablation の
            #   ``creator.money +=``(下)は **SFC 側の行だけ実在して現金が消える**。
            creator = sim.present_agent(item.creator) if item else None
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
                    rw_payload = {"amount": round(float(sim.reward_amount), 1),
                                  "balance": round(creator.money, 1)}
                    # IF-E2(既定 OFF=キーなし): ablation の外生報酬は域外(実験装置)から来る。
                    if sfc_mod.enabled(sim):
                        rw_payload["payer"] = sfc_mod.row_in(
                            sim, "shock", float(sim.reward_amount))
                    sim.logger.log(Event(
                        step=step, sim_min=sim_min, agent_id=creator.id,
                        kind="reward", x=creator.x, y=creator.y, payload=rw_payload))


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
    # ★レーン乙 ブロック7: 在場者だけが予定を書き留められる(不在者への record は捨てられる)。
    hearers = [h for h in (sim.present_agent(i) for i in party_ids) if h is not None]
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


def _attention_cap(sim) -> int:
    """S15: 発話 1 件あたりの聴衆上限(``world.attention_hearers_max``)。既定 0 = 無制限。

    毎発話 conf を辿らないよう sim に 1 度だけキャッシュする(値は L1 にも乱数にも現れない)。
    """
    cap = getattr(sim, "_attn_hearers_max", None)
    if cap is None:
        try:
            cap = max(0, int((sim.cfg.get("world", {}) or {})
                             .get("attention_hearers_max", 0) or 0))
        except Exception:                          # noqa: BLE001(旧 config 互換)
            cap = 0
        sim._attn_hearers_max = cap
    return cap


def _attention_limited(sim, speaker, hearers):
    """S15(**層3 = 世界の力学に触る**): 聴衆を話者に近い順 K 人へ絞る。既定 0 では素通し。

    正典: docs/plans/ram-rootcause-and-fix-plan.md §3 層3 S15。

    - 順序規則: **距離二乗の昇順 → id 昇順**(平方根も乱数も使わない完全な決定論)。
    - 適用範囲: この speak ハンドラが作る ``hearers`` **だけ**。``hearers_of`` の他の
      用途(同席判定・発火条件・company)には 1 バイトも触らない。ここで絞った集合が
      そのまま hear / remember / _arouse / _contact / SNC の遭遇 / 噂 / 意見更新 /
      L1 の ``hearers`` 欄に流れる = ON 時の意味論が 1 つに揃う。
    - **既定 0 では ``hearers`` を返すだけ**(リストの同一性まで保つ)= バイト一致。
    """
    cap = _attention_cap(sim)
    if cap <= 0 or len(hearers) <= cap:
        return hearers
    sx, sy = float(speaker.x), float(speaker.y)

    def _key(h):
        dx, dy = float(h.x) - sx, float(h.y) - sy
        return (dx * dx + dy * dy, int(h.id))
    # ★元の並び(hearers_of の id 昇順)は保ったまま「誰を残すか」だけを近さで決める:
    #   選抜後に id 昇順へ戻すので、下流の走査順は絞らないときと同じ規則になる。
    keep = sorted(sorted(hearers, key=_key)[:cap], key=lambda h: int(h.id))
    return keep


# ---------------------------------------------------------------- 声の段階(作用点0)
# 正典: docs/plans/hearer-cap-plan.md §2「作用点0」。物理表(段階→距離)は
# world/perception.py が持ち、ここは「発話の文脈 → 段階」の**決定論写像**だけを持つ。
# ★LLM の自由文は 1 バイトも読まない(発話テキストの解析は禁止 = 較正リスクと
#   no-fingerprint の両方の理由)。材料は既存の世界状態(名簿の役割・持ち場・イベント台帳・
#   身体の重症度)だけ。曖昧なものは**保守側 normal** へ倒す。
#
#   叫び(shout)… 話者の身体重症度が severe 以上(= 悲鳴・助けを求める声)
#   張り上げ(raised)… ①開催中イベントの**主催者**が会場ノードに立っている(集会・演説)
#                      ②声を張る路上の生業(演説者・演奏者・募金・キッチンカーの呼び込み)が
#                        **自分の持ち場に立っている**
#   通常(normal)  … 上記以外すべて(既定・大多数)
_PROJECTING_STREET_OCCS: frozenset = frozenset({
    street_life_mod.SPEECH,        # 街頭演説
    street_life_mod.MUSICIAN,      # 路上パフォーマンス
    street_life_mod.FUNDRAISER,    # 街頭募金の呼びかけ
    street_life_mod.KITCHEN,       # 屋台の呼び込み
})


def _live_event_hosts(sim, step: int) -> frozenset:
    """開催中イベントの (主催者id, 会場ノード) 集合を **step ごとに 1 度だけ**組む。

    声の段階が OFF のときは一度も呼ばれない(= 台帳走査ゼロ)。イベント台帳は
    終了分も残るので、発話ごとに走査せずキャッシュする。"""
    cached = getattr(sim, "_speech_hosts", None)
    if cached is not None and cached[0] == step:
        return cached[1]
    hosts: set = set()
    tools = getattr(sim, "tools", None)
    events = getattr(tools, "events", None) if tools is not None else None
    if events:
        for ev in events.values():
            if ev.get("started") and not ev.get("ended"):
                hosts.add((int(ev.get("host", -1)), str(ev.get("node", ""))))
    out = frozenset(hosts)
    sim._speech_hosts = (step, out)
    return out


def _speech_level(sim, agent, step: int) -> str:
    """この発話の声の段階(normal / raised / shout)を決定論写像する(乱数ゼロ・LLM 非参照)。"""
    if int(getattr(agent, "severity", 0)) >= health_mod.S_SEVERE:
        return "shout"
    if (int(agent.id), str(getattr(agent, "node", "") or "")) in \
            _live_event_hosts(sim, step):
        return "raised"
    if str(getattr(agent, "occupation", "")) in _PROJECTING_STREET_OCCS \
            and str(getattr(agent, "street_post", "") or "") == \
            str(getattr(agent, "node", "") or ""):
        return "raised"
    return "normal"


def _speak_bounds(sim, agent, step: int) -> tuple[int, float | None]:
    """speak ハンドラの列挙に渡す (cap, radius_eff)。既定は (0, None) = 現行と完全同一。

    cap は S15(`world.attention_hearers_max`)を**列挙段へ配管**したもの。結果集合は
    `_attention_limited`(フル列挙後の選抜)と同一規則なので二重適用しても冪等
    (= _attention_limited は保険としてそのまま残す)。"""
    scfg = perception_mod.speech_cfg_of(sim)
    radius_eff = (perception_mod.speech_radius(scfg, _speech_level(sim, agent, step),
                                               agent)
                  if scfg["enabled"] else None)
    return _attention_cap(sim), radius_eff


def _select_partner(sim, agent, hearers):
    """発話の返答権/宛先を決定論選択する(乱数なし・R1)。hearer 集合は不変=聞く人は
    変わらない。返す人(宛先)だけが変わる。

    "nearest"(既定): 最寄り1人(距離同点は id 小)=現行の min() と一字一句同一。
    "closeness": score = closeness(話者→候補)·10.0 − dist_m·0.1 の argmax(同点は id 昇順)。

    第78バッチ ablate.shuffle_partners: ON のとき、下の構造的な順位づけ(遭遇優先・
    closeness・nearest)を **同席者からの一様乱択**へ置き換える(専用 stream・always-draw)。
    悪評による候補外し(gossip)は「順位づけ」ではなく**資格の絞り込み**なので残す
    (= 誰と会話が起きるかの母集団は他条件と同じに保ち、順位づけだけを壊す)。"""
    if not hearers:
        return None
    # 負の評判(第61バッチ c): 悪評を知る相手を返答相手選択から後退させる(B3b の遭遇優先と同じ相手
    # 選択層。全員が悪評対象なら素通り=会話は必ず起き相手だけ変わる)。OFF は分岐に入らない=ゴールデン維持。
    if gossip_mod.enabled(sim):
        hearers = gossip_mod.demote_partners(sim, agent, hearers)
    # ablate.shuffle_partners(第78): **always-draw**。OFF でも 1 本引いて None を受け取り、
    # 従来の決定論選択へ落ちる(新 stream なので既存 stream の draw 順には干渉しない)。
    shuffled = ablate_mod.pick_partner(sim, agent, hearers)
    if shuffled is not None:
        return shuffled
    # B3b 遭遇→ペアリング(indoor.encounter.pairing): 直近(前 step)の屋内遭遇相手が同席者に
    # 居れば、そこから (遭遇 duration 降順, id 昇順) の決定論で1人を優先返答相手にする(乱数なし・R1)。
    # 該当者ゼロなら下の nearest/closeness へ後退。hearer 集合・会話発生・LLM 呼数は不変(相手だけ変わる)。
    # _indoor_recent は _phase_indoor 末で焼き込まれる per-agent マップ(相手id→遭遇 duration_s)。
    if _pairing_on(sim):
        recent = getattr(agent, "_indoor_recent", None)
        if recent:
            cands = [h for h in hearers if h.id in recent]
            if cands:
                return max(cands, key=lambda h: (recent[h.id], -h.id))
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


# ---------------------------------------------------------------- ナラティブ補間(P2 S2)
# 各 LLM 発火(計画/発話/内省)のプロンプトに、前回発火以降のその個体の客観的な出来事を機械
# ダイジェスト(意味づけをしない列挙=立ち寄った場所・会った人・起きたこと)として注入する。
# 蓄積は engine 側の実行時リングバッファ(agent._isl_buf・有界=最大30件)。observer への記録とは
# 別建てで、ディスクは読まない(run_step 末に in-memory の logger.events の増分だけを走査して振り分け)。
# 追加 LLM 呼はゼロ=呼数 k 非依存(R1)。既定 OFF=バッファも作らず・digest=None=1行も足さない=バイト一致。
# 決定論(乱数不使用)。テンプレに構成概念名・地名リテラルは置かない(場所名は arrive 等の payload=実行時値)。
_ISL_BUF_MAX = 30                    # リングバッファの上限(有界)

# 出来事 kind → ダイジェストの「起きたこと」1句(客観・意味づけしない)。arrive/hear は別扱い。
_ISL_ACT = {
    "spend":       "買い物や支払いをした",
    "appointment": "予定ができた",
    "sns_post":    "SNSに投稿した",
    "sns_read":    "SNSを見た",
    "news_read":   "ニュースを見た",
    "dm":          "メッセージを送った",
    "ride":        "乗り物で移動した",
    "speak":       "人と話した",
}


def _interstitial_on(sim) -> bool:
    """ナラティブ補間が有効か(prompts.interstitial.enabled=true)。既定 OFF=状態も増やさない。"""
    try:
        isl = (sim.cfg.get("prompts", {}) or {}).get("interstitial", {}) or {}
        return bool(isl.get("enabled", False))
    except Exception:
        return False


def _channels_on(sim, step: int) -> bool:
    """観測チャンネル(第80)をこの step で採るか。既定 OFF = サイドカー不在で即 False。

    サイドカーの有無だけを見る属性チェック(毎 step 呼ばれるので config を掘らない)。
    """
    if getattr(sim, "channels_sc", None) is None:
        return False
    every = int(getattr(sim, "channelscfg", {}).get("every_steps", 1) or 1)
    return int(step) % every == 0


def _isl_record(sim, kind: str, payload: dict):
    """1イベントを実行時バッファ用の軽量レコード (cat, val) に写す(該当なしは None)。決定論。"""
    if kind == "arrive":
        name = str((payload or {}).get("name") or "")
        return ("place", name) if name else None
    if kind == "hear":
        sid = (payload or {}).get("speaker")
        other = sim.agent_by_id.get(int(sid)) if sid is not None else None
        return ("person", other.name) if other is not None else None
    act = _ISL_ACT.get(kind)
    return ("act", act) if act else None


def _isl_accumulate(sim, since_idx: int) -> None:
    """run_step 末: この step で新規に記録されたイベント(logger.events[since_idx:])を、
    各個体の実行時リングバッファへ振り分ける(有界=先頭を捨てる)。決定論・乱数不使用。"""
    a_by = sim.agent_by_id
    for e in sim.logger.events[since_idx:]:
        aid = e.agent_id
        if aid < 0:                      # 世界イベント(agent_id=-1)は個体バッファに入れない
            continue
        rec = _isl_record(sim, e.kind, e.payload)
        if rec is None:
            continue
        agent = a_by.get(aid)
        if agent is None:
            continue
        buf = getattr(agent, "_isl_buf", None)
        if buf is None:
            buf = []
            agent._isl_buf = buf
        buf.append(rec)
        if len(buf) > _ISL_BUF_MAX:      # リングバッファ: 上限超は最古を捨てる
            del buf[0]


def _isl_digest(agent) -> str | None:
    """個体のバッファ+当日計画から客観ダイジェストを組む(意味づけしない)。空なら None。

    「計画との差分・立ち寄った場所・会った人・起きたこと」を客観列挙するだけ(なぜ印象に
    残ったか等の意味づけは夜内省の LLM の仕事=二段分離)。順序保持の重複畳みで決定論。"""
    buf = getattr(agent, "_isl_buf", None) or []
    places: list[str] = []
    persons: list[str] = []
    acts: list[str] = []
    for cat, val in buf:
        bucket = places if cat == "place" else persons if cat == "person" else acts
        if val and val not in bucket:    # 重複は畳む(先出し順を保つ=決定論)
            bucket.append(val)
    facts: list[str] = []
    plan = getattr(agent, "day_plan", None) or []
    done = sum(1 for it in plan if isinstance(it, dict) and it.get("done"))
    if plan and done:                    # 計画との差分(客観カウント)
        facts.append(f"今日の予定は{len(plan)}件中{done}件を済ませた")
    if places:
        facts.append("・".join(places[:3]) + "に立ち寄った")
    if persons:
        facts.append("・".join(persons[:3]) + "と会った")
    facts.extend(acts[:3])
    # L2 業務の実体(work.service): 勤務中に積んだ接客の要約を1事実として供給(OFF/非該当は
    # アキュムレータ不在=None=1行も足さない=バイト一致)。テキストは work モジュールに閉じる。
    wf = work_mod.digest_line(agent)
    if wf:
        facts.append(wf)
    # 物流②(商品実体): この間に買った物を1事実として供給(OFF/非該当は None=1行も足さない=バイト一致)。
    # テキストは goods モジュール/config に閉じる(engine には商品名を書かない)。
    gf = goods_mod.digest_line(agent)
    if gf:
        facts.append(gf)
    if not facts:
        return None
    return "この間のこと: " + "。".join(facts)


def _isl_take(sim, agent) -> str | None:
    """発火時: ダイジェストを組んで返し、バッファを空にする(次の「前回発火以降」を仕切り直す)。
    OFF は None(build_prompt が1行も足さない=バイト一致)。

    行間レイヤ配線(v0-2): scene_desc も ON なら「客観の見え」1文を足す(意味づけしない客観記述の
    規律を守る=解釈は夜内省の LLM の仕事)。scene_desc OFF は 1 文字も足さない=バイト一致。"""
    if not _interstitial_on(sim):
        return None
    dig = _isl_digest(agent)
    buf = getattr(agent, "_isl_buf", None)
    if buf:
        buf.clear()
    work_mod.clear_digest(agent)                   # L2 業務の実体: 当日業務の仕切り直し(OFF/非該当は no-op)
    goods_mod.clear_digest(agent)                  # 物流②: 当日購入の仕切り直し(OFF/非該当は no-op)
    scfg = getattr(sim, "scenecfg", None)
    if scene_desc_mod.enabled(scfg):
        sline = scene_desc_mod.digest_line(
            agent, city=sim.city, cfg=scfg, elevation=getattr(sim, "elevation", None))
        if sline:
            dig = (dig + "。" + sline) if dig else sline
    return dig


# ---------------------------------------------------------------- L2 業務の実体(work.service。既定 OFF)
def _work_service_on(sim) -> bool:
    """L2 業務の実体(接客 serve / オフィス産出 org_output)が有効か。既定 OFF=新経路を通さない。"""
    cfg = getattr(sim, "workcfg", None)
    return bool(cfg and cfg["enabled"])


# ---------------------------------------------------------------- 会社観測データ層 B4(既定 OFF)
# work.service 傘下の会社ごと観測。すべて work.service.enabled が前提(データ源の統合スイッチ)。
#  - indoor_fields: serve に org_id/floor を付ける(スタッフ経由が主・unstaffed は node→org 一意時のみ)。
#  - office.by_org: org_output を org_id 単位に分解(同居複数社)+ indoor ON でミクロ在席分へ。
#  - ledger.enabled: runs/<run>/org_ledger.parquet(日次1行/社)を書く。動力学は per-day アキュムレータ
#    (sim._org_day)へ積み、observer(OrgLedger)が日次境界で読むだけ=記録と動力学の分離を維持。
def _org_ledger_on(sim) -> bool:
    cfg = getattr(sim, "workcfg", None)
    return bool(cfg and cfg["enabled"] and cfg["ledger"]["enabled"])


def _org_byorg_on(sim) -> bool:
    cfg = getattr(sim, "workcfg", None)
    return bool(cfg and cfg["enabled"] and cfg["office"]["by_org"])


def _org_daily_on(sim) -> bool:
    """日次アキュムレータ/締めが要るか(org_output by_org か ledger のいずれか)。"""
    return _org_ledger_on(sim) or _org_byorg_on(sim)


def _org_indoor_fields_on(sim) -> bool:
    cfg = getattr(sim, "workcfg", None)
    return bool(cfg and cfg["enabled"] and cfg["indoor_fields"])


def _org_day_entry(sim, org_id) -> dict:
    """per-day アキュムレータの 1 社エントリ(無ければ 0 初期化)。workers は在席頭数(headcount 用)の集合。"""
    acc = getattr(sim, "_org_day", None)
    if acc is None:
        acc = {}
        sim._org_day = acc
    return acc.setdefault(str(org_id), {
        "production": 0, "revenue_est": 0.0, "wage_paid": 0.0,
        "serve_count": 0, "attendance_min": 0, "workers": set()})


def _org_node_org_ids(sim) -> dict:
    """work_node → その node を勤務地とする org_id の集合(present 個体由来・決定論)。

    unstaffed serve の node→org 解決に使う(集合が一意=1社のときだけ org_id を付与し、多義ノードは
    null=unknown を正直開示=推測しない)。pool ローテーションで在場が変わっても毎回再構成で整合。"""
    m: dict[str, set] = {}
    for a in sim.agents:
        oid = getattr(a, "org_id", None)
        wn = getattr(a, "work_node", "")
        if oid and wn:
            m.setdefault(wn, set()).add(str(oid))
    return m


def _org_worker_present_office(sim, a, sim_min: int, cal, ocfg) -> bool:
    """agent a がこの step、自社(office 系)work_node に出勤中か(在席頭数/ミクロ在席分の母集団)。"""
    oid = getattr(a, "org_id", None)
    wn = getattr(a, "work_node", "")
    if not oid or not wn or a.node != wn:
        return False
    if getattr(a, "work_start_min", -1) < 0:
        return False
    if not routine.in_work_window(a, sim_min, cal):
        return False
    return work_mod.is_office_node(sim.city, wn, ocfg)


def _phase_org_accumulate(sim, step: int, sim_min: int) -> None:
    """この step の office 系出勤者を per-day アキュムレータへ積む(在席頭数 workers・ミクロ在席分
    attendance_min)。attendance は indoor.enabled かつ ind_space_type が職務区画(attendance_zones)の
    step だけ +STEP_MINUTES 分。_phase_indoor(区画確定)の後に呼ぶ。既定 OFF=即 return=バイト一致。"""
    if not _org_daily_on(sim):
        return
    ocfg = sim.workcfg["office"]
    cal = getattr(sim, "calendarcfg", None)
    ind_on = _indoor_on(sim)
    zones = set(ocfg["attendance_zones"])
    for a in sim.agents:
        if not _org_worker_present_office(sim, a, sim_min, cal, ocfg):
            continue
        ent = _org_day_entry(sim, a.org_id)
        ent["workers"].add(int(a.id))
        if ind_on and str(getattr(a, "ind_space_type", "")) in zones:
            ent["attendance_min"] += sim.clock.step_minutes


def _emit_org_day(sim, step: int, sim_min: int, day: int) -> list:
    """per-day アキュムレータ(sim._org_day)を締める。(a) by_org ON なら org_output を org_id 単位で
    L1 へ 1 社 1 件(indoor ON はミクロ在席分 attendance_min・OFF は在席頭数×role重み。basis で自己記述)。
    (b) ledger ON なら 1 社 1 行のサイドカー行(全列 0 の社は書かない)を返す。org_id 昇順=決定論。"""
    acc = getattr(sim, "_org_day", None) or {}
    rows: list = []
    byorg = _org_byorg_on(sim)
    ledger = _org_ledger_on(sim)
    ind_on = _indoor_on(sim)
    cfg = sim.workcfg
    base_w = float(cfg["office"]["base_weight"])
    for oid in sorted(acc):
        ent = acc[oid]
        workers = ent["workers"]
        att = int(ent["attendance_min"])
        if byorg and workers:                      # (a) org_output(在席のあった office 社のみ)
            if ind_on:
                out, basis = float(att), "attendance_min"
            else:
                out = round(sum(
                    (work_mod.role_weight(sim.agent_by_id.get(w), cfg)
                     if sim.agent_by_id.get(w) is not None else base_w)
                    for w in sorted(workers)), 3)
                basis = "headcount"
            # IF-F W3: 日次産出を出したのは**その会社**なので device_id=org:<org_id>。
            # by_org 経路は org_id が主キーなので同一性が一意に決まる(下の職場キー経路と違う)。
            devices_mod.log_device(
                sim, Event(step=step, sim_min=sim_min, agent_id=-1,
                           kind="org_output", x=0.0, y=0.0,
                           payload={"org": str(oid), "output": out,
                                    "n": len(workers), "kind": "office",
                                    "basis": basis, "day": int(day)}),
                devices_mod.org_device_id(oid))
        if ledger:                                 # (b) ledger 行(いずれかの列が非0の社のみ)
            prod = int(ent["production"])
            rev = float(ent["revenue_est"])
            wage = float(ent["wage_paid"])
            serve = int(ent["serve_count"])
            if prod or rev or wage or serve or att:
                rows.append((int(day), str(oid), prod, round(rev, 6),
                             round(wage, 6), serve, att))
    return rows


def _phase_org_ledger_roll(sim, step: int, sim_min: int) -> None:
    """日次境界: 前日の per-day アキュムレータを締めて org_output(by_org)/ledger 行を出し、リセットする。

    早期(当日の出勤・産出・接客の集計より前)に置く=当日分は新しい日へ正しく積まれる(オフバイワン回避)。
    最終日は finalize_org_day が締める。既定 OFF=即 return=バイト一致。"""
    if not _org_daily_on(sim):
        return
    day = sim_min // 1440
    prev = getattr(sim, "_org_ledger_day", -1)
    if prev >= 0 and day != prev:
        rows = _emit_org_day(sim, step, sim_min, prev)
        sc = getattr(sim, "org_ledger_sc", None)
        if sc is not None and rows:
            sc.add_rows(rows)
        sim._org_day = {}
    sim._org_ledger_day = day


def finalize_org_day(sim) -> None:
    """run 終了時: 最終日の per-day アキュムレータを締める(simulation.finalize が logger.flush の前に呼ぶ)。

    最終日の org_output(by_org)は末 step の (step, sim_min) で記録=straight/split で同一。既定 OFF=no-op。"""
    if not _org_daily_on(sim):
        return
    day = getattr(sim, "_org_ledger_day", -1)
    if day < 0:
        return
    step = max(0, int(sim.cfg.run.n_steps) - 1)
    sim_min = sim.clock.sim_min(step)
    rows = _emit_org_day(sim, step, sim_min, day)
    sc = getattr(sim, "org_ledger_sc", None)
    if sc is not None and rows:
        sc.add_rows(rows)
    sim._org_day = {}


def _work_office_output(sim, step: int, sim_min: int, cfg: dict) -> None:
    """日次境界: オフィス系職場に在場・在職の出勤者を職場単位で束ね、出勤者数×role重みを
    org_output として1件記録(会社が『何かを作っている』の最小観測形)。決定論・乱数なし・非LLM。

    出勤者=work_node がオフィス系(poi_cats)かつ在職(work_start_min>=0)の present 個体。職場単位=
    work_building(無ければ work_node)。pool ローテーションで present が変われば出勤者数=産出が変わる。

    ★在場・生存の規約は ``commerce.occupancy`` / ``delivery`` の ``loc != "outside"`` 方式に
      統一する(レーン甲 2026-08-13)。以前はここだけ loc も dead も見ておらず、**街の外に居る人**と
      **死者**が会社の産出に永久に計上され続けていた(死者は sim.agents から消えないため)。"""
    ocfg = cfg["office"]
    if not ocfg["enabled"]:
        return
    day = sim_min // 1440
    if day == getattr(sim, "_work_day", -1):
        return
    sim._work_day = day
    units: dict[str, list] = {}
    for a in sim.agents:
        if a.loc == "outside" or getattr(a, "dead", False):
            continue                               # 街に居ない / 亡くなった人は出勤していない
        wn = getattr(a, "work_node", "")
        if not wn or getattr(a, "work_start_min", -1) < 0:
            continue
        if not work_mod.is_office_node(sim.city, wn, ocfg):
            continue
        key = getattr(a, "work_building", "") or wn
        units.setdefault(key, []).append(a)
    for key in sorted(units):
        workers = sorted(units[key], key=lambda a: a.id)
        output = round(sum(work_mod.role_weight(a, cfg) for a in workers), 3)
        # IF-F W3: この経路は org 台帳(by_org)が OFF のときの集計なので、持っている
        # 同一性は**職場キー**(work_building か work_node)しかない。org:<職場キー> を
        # 刻んで「会社そのものではなく職場単位でしか名乗れない」という限界を id で開示する
        # (捏造しない: org_id を推測して埋めることはしない)。
        devices_mod.log_device(
            sim, Event(step=step, sim_min=sim_min, agent_id=-1,
                       kind="org_output", x=0.0, y=0.0,
                       payload={"org": str(key), "output": output,
                                "n": len(workers), "kind": "office"}),
            devices_mod.org_device_id(key))


def _phase_work_service(sim, step: int, sim_min: int, since_idx: int) -> None:
    """L2 業務の実体(work.service。既定 OFF=即 return=バイト一致)。

    run_step 末で呼ぶ。(1) 日次境界にオフィス産出を集計。(2) この step の客の消費(spend の
    接客カテゴリ)を走査し、同一 work_node に在場・勤務中のスタッフに serve を帰属する
    (決定論・乱数ゼロ・LLM 呼ゼロ)。客側の既存イベントは不変(新イベントを足すだけ)。"""
    cfg = getattr(sim, "workcfg", None)
    if not (cfg and cfg["enabled"]):
        return
    if not _org_byorg_on(sim):                     # by_org ON 時は org_id 分解を日次境界(_phase_org_ledger_roll)へ委譲
        _work_office_output(sim, step, sim_min, cfg)   # 日次境界: オフィス系の産出(node/building キー)
    if since_idx < 0:                              # この step の増分イベントが無ければ接客帰属なし
        return
    # 勤務中スタッフを work_node で索引(在場=node==work_node かつ 勤務時間帯)。id 昇順で決定論。
    cal = getattr(sim, "calendarcfg", None)
    staff_by_node: dict[str, list] = {}
    for a in sim.agents:
        # ★_work_office_output と同じ規約(commerce.occupancy 方式)。死者・街の外に居る人を
        #   接客スタッフに数えない(node は死んだ場所のまま残るので職場と一致しうる)。
        if a.loc == "outside" or getattr(a, "dead", False):
            continue
        wn = getattr(a, "work_node", "")
        if not wn or a.node != wn:
            continue
        if not routine.in_work_window(a, sim_min, cal):
            continue
        staff_by_node.setdefault(wn, []).append(a)
    for lst in staff_by_node.values():
        lst.sort(key=lambda a: a.id)
    max_serve = cfg["max_serve_per_event"]
    # ダイジェスト供給は interstitial の消費者があるときだけ(OFF なら業務アキュムレータを作らない)
    to_digest = cfg["digest"] and _interstitial_on(sim)
    # 会社観測データ層 B4(既定 OFF=以下の org_id/floor/serve_count は付かない=serve payload バイト不変)。
    fields_on = _org_indoor_fields_on(sim)         # serve に org_id/floor を付けるか
    ledger_on = _org_ledger_on(sim)                # serve_count を per-day アキュムレータへ積むか
    floor_gate = fields_on and _indoor_on(sim)     # (建物,階)絞り込み(客とスタッフの floor 一致要求)
    node_orgs = _org_node_org_ids(sim) if (fields_on or ledger_on) else {}
    for e in sim.logger.events[since_idx:]:         # スライスはコピー=帰属 serve の追記中も安全
        if e.kind != "spend":
            continue
        cat = (e.payload or {}).get("cat")
        label = work_mod.serve_label(cfg, cat)
        if label is None:                          # 接客対象外(taxi/bus 等)
            continue
        customer = sim.agent_by_id.get(e.agent_id)
        if customer is None:
            continue
        node = customer.node
        staff = [s for s in staff_by_node.get(node, []) if s.id != e.agent_id]
        if floor_gate:                             # indoor ON+本トグル ON: 同一(建物,階)のスタッフだけ応対
            cb, cf = getattr(customer, "building", ""), int(getattr(customer, "floor", 0) or 0)
            staff = [s for s in staff
                     if getattr(s, "building", "") == cb
                     and int(getattr(s, "floor", 0) or 0) == cf]
        if staff:
            for s in staff[:max_serve]:
                payload = {"cat": str(cat), "label": label,
                           "customer": int(e.agent_id), "node": node}
                s_oid = getattr(s, "org_id", None) or None
                if fields_on:                      # スタッフ経由=帰属スタッフの org_id が主経路
                    payload["org_id"] = s_oid
                    payload["floor"] = int(getattr(s, "floor", 0) or 0)
                sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=s.id,
                                     kind="serve", x=s.x, y=s.y, payload=payload))
                if to_digest:
                    work_mod.note_serve(s, label)
                if ledger_on and s_oid:            # serve_count は付与した org_id に一致(検算整合)
                    _org_day_entry(sim, s_oid)["serve_count"] += 1
        elif cfg["record_unstaffed"]:              # 不在=記録のみ(挙動変更なし)
            payload = {"cat": str(cat), "node": node, "unstaffed": True}
            u_oid = None
            if fields_on or ledger_on:             # unstaffed は node→org が一意のときだけ解決(多義=null)
                ids = sorted(node_orgs.get(node, ()))
                u_oid = ids[0] if len(ids) == 1 else None
            if fields_on:
                payload["org_id"] = u_oid          # 一意なら org_id・多義/不在は null=unknown を正直開示
                payload["floor"] = int(getattr(customer, "floor", 0) or 0)
            # IF-F W3: 無人の応対は**店頭の販売時点(pos)が応じた**ことにする(agent_id=-1 の
            # 行に所在を与える唯一の手段)。★スタッフが応対した上の分岐には装置 id を付けない
            # = あちらは agent_id を持つ**個体の行為**であり、装置に化けさせてはならない。
            # 個体名はノード(その店)。org_id は多義のとき null に落ちるので同一性に使えない。
            devices_mod.log_device(
                sim, Event(step=step, sim_min=sim_min, agent_id=-1,
                           kind="serve", x=customer.x, y=customer.y,
                           payload=payload),
                devices_mod.pos_device_id(node))
            if ledger_on and u_oid:
                _org_day_entry(sim, u_oid)["serve_count"] += 1


# ---------------------------------------------------------------- LLM 発話
def _gather_material(sim, agent, trigger: str, step: int, sim_min: int, *,
                     dm_target: str | None = None,
                     feed_texts: list[str] | None = None,
                     reply_to: tuple[str, str] | None = None,
                     partner_id: int | None = None) -> dict:
    """1 回の思考へ渡す世界情報を集める(**世界側の唯一の収集点**・第85バッチ P1)。

    元は `_llm_speak` の本体に直書きされていた材料収集をそのまま切り出したもので、
    **順序も内容も 1 つも変えていない**(= 既定 OFF はゴールデンとバイト一致)。
    切り出す理由は P1 の要求「Perception の生成は世界側の責務」を満たすため:
    契約経路 ON ではこの dict を `Perception.from_material` が包み、OFF では従来どおり
    そのまま `build_prompt` のキーワードになる。

    ★行間ダイジェスト(`_isl_take`)と watch 節はここに**含めない**。前者はバッファを
      空にする副作用を持ち、方針キャッシュ命中時には実行してはならないため
      (呼び出し元が再利用判定の**後**に足す = 従来の評価順を厳密に保つ)。
    """
    radius = float(sim.cfg.world.perception_radius_m)
    # B3b 屋内 LOS ゲート(indoor.los): 同席文脈(発火プロンプトの近傍リスト)を壁 LOS+距離で絞る。
    # occluder=None(既定/los OFF)は従来と完全にバイト一致。ここは「プロンプトに載る同席者」だけを
    # ゲートし、発火判断(_phase_drive の face)・返答権(speak ハンドラの hearers)は素通し=LLM 呼数不変。
    company = hearers_of(agent, _percept(sim), radius, occluder=_los_occluder(sim))
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
    # 場所の意味づけ D1(labeling.place_binding・既定 OFF は None=1行も足さない=バイト一致)。
    # 現在ノードに束縛された語(採用者数が閾値以上)を中立 1 行に。決定論・乱数ゼロ・呼数不変=R1。
    place_label_line = sim.labels.place_line(getattr(agent, "node", None))
    # 場所の痕跡 IF-D(第96バッチ・world.traces・既定 OFF は None=1行も足さない=バイト一致)。
    # 現在ノードに残る痕跡のうち**最強 1 件**を中立 1 行に(強度・件数・階層名は出さない)。
    # 集約時に同席していた当事者には出さない。決定論・乱数ゼロ・LLM 呼ゼロ=R1。
    trace_line = traces_mod.line_for(sim, agent)
    # 来店時の棚知覚(INV-A・commerce.inventory.two_tier.percept・既定 OFF は None=1行も足さない
    # =バイト一致)。いま居る店の棚が **非潤沢のときだけ** 3値(残りわずか/空)を 1 行に。
    # 決定論・乱数ゼロ・LLM 呼数不変=R1(在庫台帳を O(1) で読むだけ=trace_line と同型 seam)。
    shelf_line = goods_mod.shelf_note(sim, agent)
    # 構造化シーン記述 v0(scene_desc 有効時のみ。方向つき視界/注視対象/垂直関係の 2〜4 行)。
    # 決定論・追加 LLM 呼ゼロ・乱数ゼロの純関数。company(同席者=LOS 済み)を注視の人ソースに
    # 使う。OFF は None=1行も足さない=バイト一致(crowd_line と同型 seam)。
    scene_lines = None
    _scenecfg = getattr(sim, "scenecfg", None)
    if scene_desc_mod.enabled(_scenecfg):
        scene_lines = scene_desc_mod.scene_lines(
            agent, city=sim.city, company=company, cfg=_scenecfg,
            elevation=getattr(sim, "elevation", None)) or None
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
    return {"place_name": place,
            "surprise": trigger,
            "nearby_names": [a.name for a in company],
            "nearby_ids": [a.id for a in company],
            "sim_min": sim_min, "step": step,
            "nearby_pois": [p["name"] for p in pois],
            "dm_target": dm_target, "feed_texts": feed_texts,
            "reply_to": reply_to, "tool_offers": tool_offers,
            "memberships": memberships, "pull_query": pull_query,
            "familiar_places": familiar_places,
            "institutions": institutions,
            "equip_all": equip_all,
            "venture_cost": equip_venture_cost,
            "city_name": getattr(sim, "place_name", ""),
            "date_line": getattr(sim, "today_date_line", None),
            "weather_line": getattr(sim, "today_weather_line", None),
            "schedule_line": schedule_line,
            "event_line": getattr(sim, "today_event_line", None),
            "disaster_line": getattr(sim, "today_disaster_line", None),
            "ads_line": ads_line,
            "place_label_line": place_label_line,
            "trace_line": trace_line,
            "shelf_line": shelf_line,
            "crowd_line": crowd_line,
            "wv_expect_line": wv_expect_line,
            "wv_self_line": wv_self_line,
            "wv_norm_line": wv_norm_line,
            "relation_line": relation_line,
            "reputation_line": reputation_line,
            "household_line": household_line,
            "diversity_line": diversity_line,
            "emotion_line": emotion_line,
            "goal_line": goal_line,
            "hobby_line": hobby_line,
            "norm_line": norm_line,
            "digest_line": digest_line,
            "scene_lines": scene_lines,
            "variety_hint": variety_hint,
            "labeling_mode": sim.labels.mode,
            "open_actions": bool(getattr(sim, "freedomcfg", None)
                                 and sim.freedomcfg["open_actions"]),
            "explicit_nothing": bool(getattr(sim, "freedomcfg", None)
                                     and sim.freedomcfg.get("explicit_nothing")),
            # 検証行動(第73バッチ Part B)。渡すのは bool 1 個だけ
            # =真偽台帳の値・ID・文字列は build_prompt へ到達しない。
            "verify_actions": truth_ledger_mod.verify_actions_on(sim),
            "p2_offers": p2_offers,
            "dialog_history": dialog_history,
            # V-P1(prompts.p1・既定 OFF は None=build_prompt が 1 分岐も踏まない
            # =バイト一致)。渡すのは「この呼びは熟慮である」という札 1 個だけで、
            # 世界の情報でも実験条件でもない(= 契約の提示規約の族)。
            "p1_purpose": prompt_p1_mod.purpose_for(sim, "deliberate")}


# --------------------------------------------------------------------------- #
# Perception / Intent 契約(第85バッチ・physics-instructions.md Part P1)
#
# ★ Perception の**生成は世界側の責務**(P1(1))。生成関数は本 1 本だけで、
#   cognition/perception_contract.py は型と純関数しか持たない(sim を 1 度も見ない)。
# ★ 既定 `cognition.contract.enabled: false` では下の 2 関数は 1 度も呼ばれない。
# --------------------------------------------------------------------------- #
def _contract_on(sim) -> bool:
    return bool((getattr(sim, "contractcfg", None) or {}).get("enabled"))


def _percept_salience(sim, agent) -> dict:
    """項目別の顕著性(**第80 channels の値を再利用**。二重計算しない)。

    源は第81 fire が前 step 末に採った観測 `agent._fire_obs` と期待値 `agent._fire_pred`
    で、σ 正規化誤差 |o−ô|/σ を `fire.terms_of`(S の材料の唯一の計算点)から得る。
    観測が無いラン(channels/fire OFF、または初回発火前)は **空 dict = 欠測**にする
    (0.0 で埋めない = 本リポジトリの既定則)。乱数ゼロ・LLM ゼロ・副作用なし。
    """
    obs = getattr(agent, "_fire_obs", None)
    pred = getattr(agent, "_fire_pred", None)
    if obs is None or pred is None:
        return {}
    usable = fire_mod.usable_channels(sim)
    if not usable:
        return {}
    errs, _parts = fire_mod.terms_of(obs, pred, usable)
    by_id = {i: cid for i, cid, _s in usable}
    return contract_mod.salience_from_channels(
        {by_id[i]: v for i, v in errs.items() if i in by_id})


def build_perception(sim, agent, material: dict):
    """世界情報 → `Perception`(P1(1) の唯一の生成関数)。

    `material` は `_gather_material` が集めた既存のプロンプト材料そのもの。
    ここで足すのは **プロンプトへ出ない 3 項目**だけ:
      body     … 位置・同席・移動の成否(P3 の物理由来項目の着地点)
      internal … ニーズ・時刻感覚(同上)
      salience … 第80 channels 由来の顕著性(P0 の顕著性発火が参照する値と同一)
    ★この 3 つは `prompt_kwargs()` に出ない = 物理由来の項目がプロンプトへ裏口から
      入らないことが構造で保証される(P3 の no-fingerprint 受け入れ基準の前倒し)。
    """
    body = {"x": agent.x, "y": agent.y, "node": agent.node,
            "building": agent.building, "floor": agent.floor,
            "loc": agent.loc,
            "companions": len(material["nearby_ids"]),
            # 進行の阻害・接触・局所密度は物理領域を持たない現行のグラフ世界には
            # **存在しない**。捏造せず None で欠測を明示する。
            "blocked": None, "contact": None, "local_density": None}
    # ---- 竹-4(P3(3) 物理→知覚): 直近のゾーン滞在で**実測した**値だけを入れる。
    #  ゾーンに一度も入っていない個体・物理 OFF のランでは None のまま(欠測を捏造しない)。
    #  ★この 3 欄は `prompt_kwargs()` に出ない = プロンプト文字列は 1 バイトも変わらない
    #    (第85 契約の構造による no-fingerprint 保証。tests/test_physics_zones.py が固定)。
    phys_body = physics_mod.body_of(agent)
    if phys_body:
        body.update(phys_body)
    internal = {"drive": float(getattr(agent, "drive", 0.0) or 0.0),
                "fatigue": float(getattr(agent, "fatigue", 0.0) or 0.0),
                "arousal": float(getattr(agent, "arousal", 0.0) or 0.0),
                "sim_min": int(material["sim_min"]),
                "minute_of_day": int(material["sim_min"]) % 1440}
    return contract_mod.Perception.from_material(
        material, body=body, internal=internal,
        salience=_percept_salience(sim, agent))


def _gen_call(sim, req: dict):
    """生成器が yield した要求を 1 本だけ逐次発行する(従来の `sim.llm.generate` と同一)。"""
    return sim.llm.generate(req["prompt"], rng_key=req["rng_key"],
                            temperature=req["temperature"],
                            max_tokens=req["max_tokens"])


def _run_gen(sim, gen, first: dict | None = None):
    """熟慮生成器を**その場で**逐次に回す(= engine.batch_llm OFF の既定経路)。

    生成器が yield する要求をその都度 `sim.llm.generate` へ渡して送り返すだけなので、
    生成器へ切り出す前の一本道の実装と **呼び出し順・乱数消費・イベント列が 1 命令も
    違わない**。`first` は「既に 1 回 advance 済み」の要求(batch 経路で安全弁に落ちた
    個体をここへ引き継ぐときに使う)。
    """
    try:
        req = gen.send(None) if first is None else first
        while True:
            req = gen.send(_gen_call(sim, req))
    except StopIteration as stop:
        return stop.value


def _llm_speak(sim, agent, trigger: str, step: int, sim_min: int, *,
               dm_target: str | None = None,
               feed_texts: list[str] | None = None,
               reply_to: tuple[str, str] | None = None,
               partner_id: int | None = None) -> dict | None:
    """LLM に発話/行動を生成させる(逐次経路)。解釈不能なら None(沈黙)。

    実体は生成器 `_llm_speak_g` 1 本で、ここはそれを回すだけの薄い駆動。分割の目的は
    engine.batch_llm(一括発行)で「要求を組む」と「応答を適用する」を切り離せるように
    することだけで、`_phase_planning_batched` の build/apply 分割と同じ作法である
    (逐次経路の処理順は分割前と完全同一)。
    """
    return _run_gen(sim, _llm_speak_g(sim, agent, trigger, step, sim_min,
                                      dm_target=dm_target,
                                      feed_texts=feed_texts,
                                      reply_to=reply_to,
                                      partner_id=partner_id))


def _llm_speak_g(sim, agent, trigger: str, step: int, sim_min: int, *,
                 dm_target: str | None = None,
                 feed_texts: list[str] | None = None,
                 reply_to: tuple[str, str] | None = None,
                 partner_id: int | None = None):
    """(生成器)LLM 要求を 1 回 yield し `(response, call_id, cached)` を受けて適用する。
    予算は欲求フェーズ(_phase_drive)で消費済み(発火権を得た者だけが来る)。"""
    # ---- ablate.llm_off(第78バッチ・既定 OFF=この 2 行は素通り)----
    # LLM を 1 本も呼ばず即 None を返す。呼び出し元(_decide)は既存の「解釈不能=沈黙」経路と
    # 同じく routine.decide(既存のニーズ充足ロジック + POI 選好)へ後退する。
    # ★新しいヒューリスティックは 1 つも足していない(比較のベースラインなので素朴さに価値がある)。
    # ★プロンプトを 1 つも組まないので、この分岐に入った時点で「実験条件がプロンプトに現れる」
    #   経路は原理的に存在しない(fingerprint_risk=none の根拠)。
    if ablate_mod.llm_off(sim):
        return None
    material = _gather_material(sim, agent, trigger, step, sim_min,
                                dm_target=dm_target, feed_texts=feed_texts,
                                reply_to=reply_to, partner_id=partner_id)
    # 方針キャッシュ(P2 S7): 類似状況の熟慮を再利用し LLM をスキップ(既定 OFF は即 None)。
    # ダイジェスト消費(_isl_take)より前に置く=再利用時は「実際の LLM 発火」ではないので
    # 行間バッファを仕切り直さない。
    reused = deliberate.maybe_reuse_action(sim, agent, step, sim_min, trigger)
    if reused is not None:
        return reused
    # 行間補間(P2 S2): 前回発火以降の客観ダイジェスト。OFF は None=1行も足さない=バイト一致。
    # 発火のたびに1度だけ組み、バッファを仕切り直す(=次の「前回発火以降」の起点)。
    # 第82: 監視仕様 watch(既定 OFF は None=1行も足さない=バイト一致)。§6-3 の
    # model-revision は驚き発火のときだけ中立 1 行が増える。
    # ★ここまでの 3 行の評価順は build_prompt の引数評価順(従来)と厳密に同じ。
    material["interstitial_digest"] = _isl_take(sim, agent)
    material["watch_section"] = watch_mod.section(sim)
    material["revision_line"] = watch_mod.revision_line(sim, agent, step, trigger)
    # 第87: 会話ターンだけに載る終結の宣言路(既定 OFF は None=1行も足さない=バイト一致)。
    material["engaged_section"] = engaged_mod.prompt_section(sim, agent, trigger)
    # 第94 IF-B: 再計画エピソード中だけに載る「予定が果たせなかった理由」1 行
    # (既定 rejection_notify=silent は None=1行も足さない=バイト一致)。
    material["reject_line"] = reject_mod.why_line(sim, agent)
    # 第117 SNC v2: 返答のときだけ載る判定の説明 2 行(relate / follow をJSONに足してよい)。
    # 既定 net.contact_formation.enabled=false は None=1行も足さない=バイト一致
    # (engaged_section / reject_line と完全同型の seam)。
    material["snc_section"] = contact_mod.prompt_section(sim, trigger)
    # ATT 層B(第143・cognition.attention_block): 「いま気にしていること」節 +
    # 目の前の人・耳に入っている言葉 + attend を書いてよいという 1 行。
    # 既定 OFF は None=1 行も足さない=バイト一致(snc_section と完全同型の seam)。
    material["attention_section"] = attention_mod.prompt_section(
        sim, agent, material.get("nearby_names"))
    # ATT 層A: この呼びが返答なら「誰へ返すか」を預かる(= 同 step の発話で会話相手を
    # 優先充当する)。層A の顕著性選抜が OFF なら属性を 1 つも生やさない。
    attention_mod.note_reply_target(sim, agent, partner_id, step)
    # ---- 第85バッチ: 契約経路(既定 OFF)------------------------------------- #
    #  ON では world → Perception → prompt に一本化する。**プロンプト文字列は 1 バイトも
    #  変わらない**(prompt_kwargs() が material と完全に等しい dict を返す)= P1(3)
    #  「この時点では挙動は変わらない」。Perception も既存の canary 関門へ通す(P1 受入 3)。
    if _contract_on(sim):
        percept = build_perception(sim, agent, material)
        truth_ledger_mod.check_percept(percept)
        kwargs = percept.prompt_kwargs()
    else:
        kwargs = material
    prompt = deliberate.build_prompt(agent, **kwargs)
    rng_key = f"deliberate/{agent.id}/{step}"
    # ★ここが build/apply の境目(engine.batch_llm の唯一の切断点)。逐次経路では
    #   `_run_gen` が即座に `sim.llm.generate` を回して送り返すので従来と同一。
    response, call_id, cached = yield {
        "prompt": prompt, "rng_key": rng_key,
        "temperature": float(sim.cfg.model.temperature),
        "max_tokens": int(sim.cfg.model.max_tokens)}
    sim.logger.log_llm_call({"llm_call_id": call_id, "agent_id": agent.id,
                             "purpose": trigger, "step": step, "cached": cached})
    # V3 決定モード(observer.decision_mode。既定 OFF は即 return=1 バイトも動かない)。
    # ★ここは `log_llm_call` と**同じ位置**なので、数える呼は l1b_llm の行と 1:1 になる。
    decmode_mod.note_llm_call(sim, sim_min, trigger)
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                         kind="llm_deliberate", x=agent.x, y=agent.y,
                         llm_call_id=call_id, payload={"trigger": trigger}))
    # 第87: LLM を 1 回通した = エピソードのターンが 1 進んだ(**唯一の計上点**)。
    # 表現形(発話/返答/投稿/DM/独り言)によらず「1 回の思考」なので同じ 1 ターン。
    # 既定 OFF・エピソード不在なら完全 no-op。
    engaged_mod.note_turn(sim, agent, step, sim_min)
    action = deliberate.parse_action(response)
    # ---- 第85バッチ: Intent 契約(既定 OFF)--------------------------------- #
    #  思考の出力を型化して世界へ返す唯一の経路。`Intent` は移動目標・急ぎ度・回避傾向・
    #  対話意図・滞在意図までしか持たず、**経路の各点・速度の時系列は持たない**(P1(2))。
    #  現行のグラフ世界では実行側が従来の行動 dict を消費するので、ここで**無損失に**
    #  戻す(from_action → to_action が恒等 = 情報等価)。P3 で物理側が Intent を直接
    #  受け取るようになったとき、この行が写像の分岐点になる。
    if action is not None and _contract_on(sim):
        action = contract_mod.Intent.from_action(action).to_action()
    # 第82: 監視仕様の受理(ホワイトリスト検証→数値クランプ→不正なら**前回仕様を維持**)。
    # 行動のパース成否とは独立(行動が壊れていても watch だけ読めることがある)。
    # 既定 OFF は即 return で 1 バイトも触らない。
    watch_mod.apply(sim, agent, response, step, sim_min)
    # ATT 層B(第143): 構造化出力の `attend` を受理して注意ブロックを全量置換する
    # (MEM1 式)。**欄が無ければ 1 バイトも触らない**(SNC の relate/follow と同じ流儀・
    # mock は出さない)。既定 OFF は即 return。追加 LLM 呼ゼロ・乱数ゼロ。
    attention_mod.apply_declaration(sim, agent, action, step)
    _model_revision_beliefs(sim, agent, step, sim_min, trigger)
    if action is None:                             # D16: 壊れたら沈黙して続行
        # V3: `fallback{reason:"parse_error"}` には trigger が載らない(用途別の内訳が
        # 既存 L1 から復元できない)ので、ここでだけ用途つきで数える。既定 OFF は no-op。
        decmode_mod.note_llm_unparsed(sim, sim_min, trigger)
        _log_reject(sim, agent, response, trigger, step, sim_min)
    else:
        deliberate.store_action(sim, agent, step, sim_min, trigger, action)
        # IF-1(observer.llm_link。既定 OFF=この 1 行は即 return=キーを積まない)。
        # role は l1b_llm に記録した purpose と**同じ値**(= trigger)を使う
        # (新語彙を作らない。設計 provlink.py / if-lane-research.md §4-3-1)。
        # ★store_action の**後**に積む: 方針キャッシュに一時キーを持ち込まない。
        provlink_mod.stamp(sim, action, call_id, trigger)
    return action


def _model_revision_beliefs(sim, agent, step: int, sim_min: int,
                            trigger: str) -> None:
    """model-revision(§6-3): 驚き発火では**信念の確信度も**決定論規則で見直す。

    計画書 §6-3「予測誤差が大きいときに起きるのは『単に考える』ではなく**世界モデルの
    書き換え**」。ô(監視仕様)の書き換えは LLM が行うが、**信念台帳の側は第73の既存
    機構を呼ぶだけ**(新しい規則を発明しない・乱数ゼロ・LLM 呼ゼロ)。
    真偽台帳は cognition/ から触れない契約なので、接続はここ(engine 側)に置く。
    """
    if watch_mod.revision_line(sim, agent, step, trigger) is None:
        return
    if not truth_ledger_mod.enabled(sim):
        return
    cfg = sim.watchcfg
    truth_ledger_mod.revise_on_surprise(
        sim, agent, step, sim_min, factor=float(cfg["belief_revision"]),
        max_facts=int(cfg["belief_max_facts"]))


def _log_reject(sim, agent, response: str, trigger: str,
                step: int, sim_min: int) -> None:
    """パース不成立の記録。既定は従来どおり `fallback{reason:"parse_error"}` 1 件。

    未定義行動レジスタ(第70バッチ IDEA②・freedom.undefined_register=true)のときだけ、
    「JSON は読めて "action" もあるが既知動詞のどれでもない」出力を `fallback` **ではなく**
    `undefined_action` として記録する(= 行動空間の外へ出ようとした痕跡を捨てない)。
    振り分けなので分子は排他 = `llm_fallback_rate + undefined_action_rate` が旧
    `llm_fallback_rate` に一致する(保存則)。既定 OFF は 1 バイトも変わらない。
    乱数ゼロ・LLM 呼ゼロ・プロンプト不変。
    """
    fcfg = getattr(sim, "freedomcfg", None)
    if fcfg is not None and fcfg.get("undefined_register"):
        reason, info = deliberate.classify_reject(response)
        if reason == "unknown_action":
            payload = dict(info)
            payload["trigger"] = trigger
            sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                                 kind="undefined_action", x=agent.x, y=agent.y,
                                 payload=payload))
            return
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                         kind="fallback", x=agent.x, y=agent.y,
                         payload={"reason": "parse_error"}))


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
    # 勾留中(detained_until。既定 0=全員 step>=0 で常に通過=不変)は欲求発火の対象外。
    # 在宅覚醒中(HOME_AWAKE β9・既定 OFF=属性が存在せず home_awake_mod.muted は常に False)
    # も同様に対象外にする = **起きている時間が伸びたぶんだけ呼数が増える**という副作用を
    # 構造で断つ(R1: この機能は LLM を 1 呼も足さない)。睡眠中と同じ扱い = 差分は
    # 「その時間に家に居るか路上に居るか」だけに絞られる。
    active = [a for a in sim.agents if a.loc != "outside" and not a.sleeping
              and step >= getattr(a, "detained_until", 0)
              and not home_awake_mod.muted(a, sim)]
    affect_on = _affect_on(sim)
    health_on = _health_on(sim)
    # 離散感情ラベル(内面 H6): affect ON 前提で core affect(mood_valence × arousal)から離散ラベルを
    # 決定論写像し、変化時のみ emotion_label を記録(sparse)。affect OFF / inner_life OFF / emotion 無効
    # なら完全 no-op(ラベル更新なし・イベント 0 件=バイト一致)。drive(発火系)には接続しない(R1)。
    emotion_on = affect_on and inner_life_mod.enabled(sim) \
        and inner_life_mod.cfg_of(sim)["emotion"]["enabled"]
    _gt_extras_on = gt_extras_mod.enabled(sim)   # G7(既定 OFF=payload にキーを足さない)
    for agent in active:
        drive.step_tick(agent, cfg, step)
        if affect_on:                       # 覚醒度を毎step baseline へ漏れ減衰(drive と並列)
            affect.decay(agent, sim.affectcfg)
            if emotion_on:                  # core affect → 離散感情ラベル(決定論・drive 非接続)
                inner_life_mod.update_emotion(agent, sim.innerlifecfg, step,
                                              sim_min, sim.logger,
                                              gt_extras=_gt_extras_on)

    age_cog_on = age_cog_mod.enabled(sim)   # AGE-C(既定 OFF=以下の 1 行を通らない)

    def _eff_thr(a):
        """実効閾値(E2 ドリフト + 覚醒の逆U字変調 + 疲労 + 年齢)。OFF or gain=0 で恒等=バイト一致。"""
        d = affect.threshold_delta(a, sim.affectcfg) if affect_on else 0.0
        if health_on:                       # 高疲労→閾値↑(休息へ寄る)。gain=0 で 0=恒等
            d += health_mod.fatigue_threshold_delta(a, sim.healthcfg)
            # 軽症でも出勤している個体の性能デバフ(H1 severity。既定 OFF=0.0=恒等)
            d += health_mod.severity_threshold_delta(a, sim.healthcfg)
        if age_cog_on:                      # AGE-C: 年齢由来の不透明な delta(既定 OFF=0.0=恒等)
            d += age_cog_mod.threshold_delta(sim, a)
        return drive.effective_threshold(a, d)

    # 発火の関数形(B段 seam)。fixed=閾値ゲート+個人重み抽選(現行)。
    # logistic=閾値を soft 化し p=σ(slope·(drive−threshold)) に一本化。
    logistic = (cfg.get("firing", "fixed") == "logistic")
    null_series = (getattr(sim, "controls_mode", "none") == "null_series")
    # ---- 第81バッチ: 認知イベントキュー(既定 OFF)。**発火機構の唯一の置換点** ----
    #  ON では「誰がこの tick に思考の申請をするか」をキューが決める(= 世界 tick と
    #  思考の頻度の分離)。周期発火は従来どおり閾値+個人重み抽選+予算のゲートを通し、
    #  驚き/内部の割込みだけが抽選を飛ばす(対面会話の確定発火と同格)。
    #  「全員の基本周期 10 分・他の発火源なし」設定では due が毎 step 全員になるので
    #  requesters は現行と厳密に同じ集合になる(P0(2) の後方互換の要)。
    fire_on = fire_mod.enabled(sim)
    due: dict = {}
    forced: set = set()
    granted_ids: set = set()
    if fire_on:
        # ---- 第87バッチ: engaged モード(既定 OFF=以下 3 行はすべて no-op / 恒等)----
        #  pre_tick … 生きているエピソードの継続を「今」へ繰り上げる(点 → 区間の実体)。
        #             due_events の**前**に置くので、繰り上げた個体は S 計算・ô 更新・
        #             plasticity・再スケジュールの通常経路をそのまま通る(fire の内部規約を
        #             1 つも迂回しない)。
        #  update  … 脱出 4 条件 → 突入 5 条件の順に評価し、**割込み権**を返す。
        #             ここで返った id は下の `forced` に合流して抽選を飛ばす確定発火になる
        #             (= 対面会話・驚き割込みと同格。新しい呼び出しサイトではない)。
        engaged_mod.pre_tick(sim, step, sim_min)
        due = fire_mod.due_events(sim, step, sim_min, active)
        forced = {aid for aid, ev in due.items() if ev["interrupt"]}
        # 対面会話が起きうる個体(= 下の requesters ループが face と判定する集合)。
        # engaged が「話しかける側」の突入を判定するために要る。ON のときだけ組む
        # (空間索引があるので O(N) の近傍参照。判定式は下の face と厳密に同じ)。
        face_ids = frozenset(
            a.id for a in active
            if step >= a.conv_cooldown_until
            and hearers_of(a, _percept(sim), radius)) \
            if engaged_mod.enabled(sim) else frozenset()
        forced |= engaged_mod.update(sim, step, sim_min, due, active, face_ids)
        requesters = [a for a in active
                      if a.id in due and step >= a.refractory_until
                      and (a.id in forced or logistic or a.drive >= _eff_thr(a))]
    elif logistic:
        requesters = [a for a in active if step >= a.refractory_until]
    else:
        requesters = [a for a in active
                      if a.drive >= _eff_thr(a)
                      and step >= a.refractory_until]
    requesters.sort(key=lambda a: (-a.drive, a.id))
    for agent in requesters:
        # evening_talk(β9・既定 OFF=home_awake_now は常に False=この行は無風): 在宅覚醒中の
        # 個体をこの phase に入れているのは **同居人との対面会話のためだけ**なので、同席者が
        # 居ない / 会話クールダウン中なら申請そのものを持たせない(= SNS 投稿・DM・独り言と
        # いった外部への経路は閉じたまま。muted 側で同居人の在宅覚醒は確認済み)。
        if home_awake_mod.home_awake_now(agent) and not (
                step >= agent.conv_cooldown_until
                and hearers_of(agent, _percept(sim), radius)):
            continue
        stats["requests"] += 1
        req_drive = agent.drive
        reason = drive.top_reason(agent)
        rng = sim.hub.stream("drive", agent.id, step)
        # 対面会話: 同席者がいて会話クールダウン外なら抽選なしで確定発火(logistic でも不変)。
        face = bool(hearers_of(agent, _percept(sim), radius)) \
            and step >= agent.conv_cooldown_until
        # 第81: 驚き/内部の割込みは抽選なしの確定発火(face と同格)。OFF は forced 空=不変。
        interrupt = agent.id in forced
        granted = False
        if face or interrupt:
            lottery = None
            # purpose 引数は **観測ラベルだけ**(DPH-O ④)。二層予算 OFF ではレーンも
            # 判定も従来と同一(lod.PURPOSE_LANE が face/interrupt を general へ写す)。
            if sim.budget.take("face" if face else "interrupt"):  # 予算切れ=ゲージ維持で持ち越し
                granted = True
                # 割込みでも LLM へ渡す「きっかけ」は既存語彙のまま(no-fingerprint:
                # 発火機構の ON/OFF がプロンプトから読めない)。理由は L1 の cog_fire にだけ残る。
                agent._fire_reason = "social_face" if face else reason
                drive.on_fire(agent, step, cfg)
                stats["fires"] += 1
                if face:
                    stats["face_fires"] += 1
        else:                                      # 媒体/独り言: 抽選
            if logistic:
                p = 1.0 / (1.0 + math.exp(
                    -cfg["slope"] * (agent.drive - _eff_thr(agent))))
            else:
                p = agent.fire_weight
            lottery = bool(rng.random() < p)
            if lottery and sim.budget.take("media"):
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
        if granted:
            granted_ids.add(agent.id)
            if fire_on:
                # 第82(model-revision §6-3): この step の**認知イベントの発火源**を控える。
                # プロンプトへ出るのは中立な 1 行だけで、発火源の語彙(periodic/salience/…)は
                # 1 文字も出さない(第81 の no-fingerprint テストをそのまま維持する)。
                # 同席(face)で表現形が会話になっても、発火源が驚きなら世界モデルの
                # 書き換えは起きる(= 表現形と認知モードは別軸)。
                agent._fire_src = (int(step), due[agent.id]["reason"])
        if granted and null_series:                # 対照: 発火に紐付けたダミー呼び出し
            _null_calls(sim, agent, step, sim_min)
    if fire_on:                                    # スケジュール列そのものを L1 に残す(P0-4)
        fire_mod.log_events(sim, step, sim_min, due, granted_ids)
    sim.drive_stats = stats


def _phase_c2(sim, step: int, sim_min: int) -> None:
    """会話3層 C2/C3(P2 S3)。既定 OFF=完全 no-op(新 stream c2_meet を引かず・conversation 0 件)。

    _phase_drive とは独立の新フェーズ。空間索引(sim.percept_index)で同席・会話可能な対を
    決定論列挙し、専用 stream で構造化会話(LLMなし)を成立させる。オーケストレータ本体は
    src/society/conversation.py(関係/意見/語彙/drive を呼ぶだけ=追加 LLM 呼ゼロ)。C2→C1 昇格は
    drive ゲージを押し上げ、次以降の step の _phase_drive が既存経路でフル発話を発火する。"""
    conversation_mod.run_phase(sim, step, sim_min)


def _feed_texts(sim, agent, step: int, sim_min: int) -> list[str]:
    """タイムラインを「@著者名: 本文」形式で見せる(場所・時刻の宣言化を防ぐ)。

    推薦(Wave G6)ON なら意見整合で選別された TL を見せる(infoenv.timeline)。OFF は従来の
    時系列 TL(sim.net.timeline_for をそのまま呼ぶ=バイト一致)。

    ablate.propagation_off(第78): TL は**他者が書いた本文をプロンプトへ直に注入する経路**
    なので、ON では 1 件も渡さない([] = build_prompt が TL 行を出さない)。既読マーク等の
    net 側の進行は timeline() を呼ぶこと自体で従来どおり進む(= 呼数・イベントは不変)。"""
    out = []
    for p in infoenv_mod.timeline(sim, agent, step, sim_min):
        meta = sim.agent_by_id.get(p["author"])
        who = "公式" if p["author"] == -1 or meta is None else meta.name
        out.append(f"@{who}: {p['text']}")
    return [] if ablate_mod.propagation_off(sim) else out


def _fire_llm_g(sim, agent, reason: str, step: int, sim_min: int, rng):
    """(生成器)発火権を得た思考の文脈選択 → LLM。きっかけ(reason)で表現形を選ぶ。

    同席時の対面会話は _phase_drive で確定発火(reason="social_face")済み。
    どの枝も `_llm_speak_g` をちょうど 1 回だけ通す(= 1 発火 1 呼)。
    """
    if reason == "social_face":
        return (yield from _llm_speak_g(sim, agent, "social", step, sim_min))
    if reason in ("novel_place", "congestion", "unknown_word"):
        return (yield from _llm_speak_g(sim, agent, reason, step, sim_min))
    if reason == "dm_received" and agent._last_dm_from is not None:
        target = sim.agent_by_id.get(agent._last_dm_from)
        if target is not None:
            action = yield from _llm_speak_g(sim, agent, "dm", step, sim_min,
                                             dm_target=target.name)
            if action is not None and action.get("type") == "dm":
                action["to"] = target.id
            return action
    if reason == "news":
        return (yield from _llm_speak_g(
            sim, agent, "post", step, sim_min,
            feed_texts=_feed_texts(sim, agent, step, sim_min)))
    if agent.has_phone and rng.random() < 0.45:
        contacts = sorted(sim.net.contacts.get(agent.id, ()))
        if contacts and rng.random() < 0.35:
            target_id = contacts[int(rng.integers(len(contacts)))]
            meta = sim.agent_by_id.get(target_id)
            if meta is not None:
                action = yield from _llm_speak_g(sim, agent, "dm", step,
                                                 sim_min, dm_target=meta.name)
                if action is not None and action.get("type") == "dm":
                    action["to"] = target_id
                return action
        return (yield from _llm_speak_g(
            sim, agent, "post", step, sim_min,
            feed_texts=_feed_texts(sim, agent, step, sim_min)))
    return (yield from _llm_speak_g(sim, agent, "solo", step, sim_min))


# ---------------------------------------------------------------- 意思決定
def _decide(sim, agent, step: int, sim_min: int) -> dict:
    """意思決定(逐次経路)。実体は生成器 `_decide_g` で、ここは回すだけの薄い駆動。"""
    return _run_gen(sim, _decide_g(sim, agent, step, sim_min))


def _decide_g(sim, agent, step: int, sim_min: int):
    # V3 決定モード(observer.decision_mode。既定 OFF は note_* が即 return)。
    #   `_why` = 「LLM がこの決定を決めなかった理由」。ルール層へ落ちた**その 1 度だけ**
    #   `note_rule` が消費する = 決定 1 回につき記録も 1 件(不変式 points == llm+reuse+rule)。
    #   ★逐次経路と一括発行経路は同じ `_decide_g` を回すので、batch ON/OFF で記録は同一。
    _why = "no_fire"
    # 勾留(制度深化2・既定 0=フラグ立たず不変): 拘束中は行動しない(返答・発火・移動なし)。
    if step < getattr(agent, "detained_until", 0):
        decmode_mod.note_rule(sim, sim_min, "detained")
        return {"type": "stay"}
    rng = sim.hub.stream("decide", agent.id, step)
    radius = float(sim.cfg.world.perception_radius_m)
    # 第152 小修正: ここで要るのは**同席者がいるか否か**の 1 ビットだけ(下の drive 加算と
    #   routine.decide の has_company)。リストは 1 度も読まないのに `hearers_of` は
    #   40m 圏を全列挙して id 昇順に整列していた(250k step0 の py-spy で step 時間の
    #   62.6%)。`count_hearers` は `len(hearers_of(...))` と**全入力で同値**(第150 で
    #   機械照合済み: tests/test_count_hearers.py)なので `bool(company)` == (count > 0)。
    #   半径・知覚ソース・呼び順・乱数はここでは 1 つも変えていない。
    has_company = count_hearers(agent, _percept(sim), radius) > 0
    if has_company:
        drive.add(agent, "company", sim.drivecfg)
    agent._heard_unknown = False

    # 返答保証: 話しかけられたら抽選なしで必ずLLMで返答(予算があれば)。
    #  ★在宅覚醒(HOME_AWAKE β9・既定 OFF=reply_open は常に True=従来経路): 睡眠中と同じ
    #    扱いでここを素通りする。`_reply_to` は**消さずに預かったまま**にするので、OFF で
    #    就寝中に保留された返事が起床後に撃たれるのと同じ挙動になる(返事を落とさない =
    #    DPH-B の「返事保証 100%」を壊さない)。evening_talk ON のときは「在宅覚醒中の
    #    同居人からの返答権」だけがここで開く(相手が誰かを見て決める = 外部経路は閉じたまま)。
    if agent._reply_to is not None and home_awake_mod.reply_open(agent, sim):
        # ---- 第87バッチ: 関係の薄い相手からの定型接触は 1 ターンのテンプレで流す(§8 突入 2)。
        #  **LLM を 1 本も呼ばず・予算も取らず**返す = ENGAGED に突入しない = この後
        #  _apply(speak) の handoff_ok が返答権を渡さないので、やりとりは 1 ターンで終わる。
        #  既定 OFF は reply_mode が常に "engage" = 従来経路(バイト一致)。
        if engaged_mod.reply_mode(sim, agent) == "template":
            agent._reply_to = None
            agent.conv_turns_left -= 1
            if agent.conv_turns_left <= 0:
                agent.conv_cooldown_until = step + sim.drivecfg["conv_cooldown_steps"]
            decmode_mod.note_rule(sim, sim_min, "template")
            return engaged_mod.template_reply(sim, agent, step, sim_min)
        speaker_id, said = agent._reply_to
        agent._reply_to = None
        # DPH-O ①: 予算切れで落ちた返事は**痕跡ゼロ**だった(_reply_to は上で既に消えている)。
        #   observer.starvation OFF では note_* が即 return = L1 も state も 1 バイト不変。
        #   予算の取り方(引数 1 つ)も返り値も従来と同じ = 呼数・乱数・分岐は不変。
        _reply_ok = sim.budget.take("reply")
        if not _reply_ok:
            starvation_mod.note_reply_dropped(sim, agent, step, sim_min, speaker_id)
            _why = "reply_starved"                 # V3(記録用の局所変数。分岐に使わない)
        if _reply_ok:
            speaker = sim.agent_by_id.get(speaker_id)
            reply_to = (speaker.name, said) if speaker is not None else None
            # ablate.propagation_off(第78): **返答の LLM 呼はこれまでどおり撃つ**(予算も
            # 消費済み=呼数の構造が不変)が、相手の発話内容はプロンプトへ 1 文字も入れない。
            # reply_to=None にすると build_prompt は状況行そのものを出さない(=「話しかけ
            # られたのに内容が空」という不自然な行は生まれない)。
            if ablate_mod.propagation_off(sim):
                reply_to = None
            action = yield from _llm_speak_g(sim, agent, "reply", step, sim_min,
                                             reply_to=reply_to,
                                             partner_id=speaker_id)
            # SNC v2 の C1/F2(第117・既定 OFF は即 return=1 バイトも動かない)。
            # **返答イベントの因果の中**(同じ step/sim_min・返答を出した当人の直後)で、
            # 返答 JSON の relate / follow を読んで関係を結ぶ。欄が無ければ false=何もしない
            # (mock は 2 欄を出さないので mock ランでは 1 件も成立しない)。
            # LLM 呼は 1 本も増えない(この 1 行は既に返ってきた応答を読むだけ)。
            contact_mod.on_reply(sim, agent, speaker_id, action, step, sim_min)
            agent.conv_turns_left -= 1
            if agent.conv_turns_left <= 0:         # セッション上限 → クールダウン
                agent.conv_cooldown_until = step + sim.drivecfg["conv_cooldown_steps"]
            sim.drive_stats["replies"] = sim.drive_stats.get("replies", 0) + 1
            if action is not None:
                return action                      # V3: LLM / 再利用が決めた(内側で計上済み)
            _why = "reply_unparsed"                # 撃ったが行動にならなかった(or ablate.llm_off)

    if agent._fire_reason:                         # 欲求フェーズで発火権を得た
        reason = agent._fire_reason
        agent._fire_reason = ""
        agent._last_fire_reason = reason           # 造語の発生きっかけの記録用
        action = yield from _fire_llm_g(sim, agent, reason, step, sim_min, rng)
        if action is not None:
            return action                          # V3: LLM / 再利用が決めた(内側で計上済み)
        _why = "fire_unparsed"

    action = routine.decide(agent, step, sim, _place_of(sim, agent), rng,
                            has_company=has_company)
    # V3: ここから先の行動は**ルール層**が決めたもの(スマホ行動も含む)。`routine.decide`
    #   の中で朝の計画のブロックが立ったなら、その来歴が出所として一緒に記録される。
    #   既定 OFF は即 return = 世界も L1 も 1 バイト不変。
    decmode_mod.note_rule(sim, sim_min, _why)
    if action["type"] == "stay" and agent.has_phone and not agent.sleeping:
        phone_action = _phone(sim, agent, step, sim_min)
        if phone_action is not None:
            return phone_action
    return action


# --------------------------------------------------------------------------- #
# 日中熟慮の一括発行(engine.batch_llm。既定 OFF=以下は 1 度も呼ばれない)
#
# 朝計画・夜内省(_phase_planning_batched / _phase_reflect_batched)と違い、日中熟慮は
# **個体間で独立ではない**。`_decide` は 1 個体ぶんが
#     prefix(材料収集・プロンプト構成)→ LLM → suffix(応答の適用/場合により後退経路)
# という並びで、suffix が世界へ書いた結果を **次の個体の prefix が読む**経路が実在する。
# したがって「全員の prefix を先に回す」形の素朴な一括化は成立しない。ここでの境界は:
#
#   ① LLM を 1 本も撃たない個体(= 大多数)は **その場で最後まで走らせる**。
#      → 後退経路(routine.decide / _phone → _hear_words / SNS 反応 / イベント参加)の
#        他個体への書き込みは、逐次と同じ位置で起きる。
#   ② LLM を撃つ個体だけを応答待ちで中断し、まとめて発行して id 順に再開する。
#      遅らせるのは「応答を適用する部分」だけで、そこは自個体の状態と L1 に閉じている。
#   ③ ただし **パース不成立の後退経路**(壊れた JSON → routine.decide/_phone、
#      返事が壊れた個体の 2 本目の熟慮)は他個体を読み書きする。これが起こりうる個体は
#      `_deliberate_defer_ok` が **batch に入れず**、その場で逐次に走らせる(= OFF と同一経路)。
#   ④ 観測出力の並びは `_reorder_decide_log` が個体単位で逐次と同じ順へ戻す。
#      (L1 events と L1b llm_calls の両方。step 内の他フェーズは走っていないので安全)
#   ⑤ **残る 1 点を正直に書く**: ③の安全弁を通った個体でもパースが不成立になれば
#      後退の `routine.decide` は走る。`routine.decide` 自体が他個体へ書く経路は
#      すべて既定 OFF の下位系(宅配の注文・共同行動/デート/party/火種の合流先・
#      サービス来店・相乗り)に限られるが、それらが ON のランでは「壊れた JSON を
#      返した個体の後退が、逐次なら先に起きていた」という 1 step 内の順序差になりうる。
#      そこで `deferred_fallback` を数える: **この値が 0 のあいだ batch 経路は逐次
#      経路と機械的に同値**(= 遅延した部分に後退経路が 1 度も現れていない)。
#      ラン後にこの 1 個の整数を見れば「厳密一致だったか」が判定できる。
# --------------------------------------------------------------------------- #
def _batch_llm_cfg(sim) -> dict:
    return (sim.cfg.get("engine", {}) or {}).get("batch_llm", {}) or {}


def _policy_cache_on(sim) -> bool:
    """方針キャッシュが有効か(conf を読むだけ・副作用ゼロ)。"""
    raw = (sim.cfg.get("cognition", {}) or {}).get("policy_cache", {}) or {}
    return bool(raw.get("enabled", False))


def _deliberate_batch_on(sim) -> bool:
    """日中熟慮を一括発行してよいか(step ごとに 1 度だけ判定する phase 級の安全弁)。

    方針キャッシュ ON では `store_action` が **全体 LRU** を持つ共有キャッシュを触る
    (= 他個体の枠を追い出す)ので、適用を遅らせると後続個体の `reuse_action` の
    命中が変わりうる。exact でない一括化はしない、が本レーンの規律なので丸ごと外す。
    """
    return bool(_batch_llm_cfg(sim).get("enabled", False)) \
        and not _policy_cache_on(sim)


def _deliberate_defer_ok(sim, agent, step: int) -> bool:
    """この個体の**応答適用以降**を後回しにしてよいか(個体級の安全弁)。

    後回しにしてよいのは「遅らせる部分が他個体の状態を読みも書きもしない」ときだけ。
    応答を適用する部分(記録 → parse → watch → 信念 → provlink)は自個体と L1 に
    閉じているが、**パース不成立のときに続く後退経路**は違う:

      ① 返事(reply)が壊れた個体はこの後 **もう 1 本**(発火)の熟慮を撃つ。その
         材料収集は他個体の状態(評判・語・イベント・TL)を**読む**。
      ② 後退の `routine.decide` → `_phone` は検索/TL 閲覧/ニュースで `_hear_words`・
         SNS 反応・イベント参加を起こし、造語者や投稿者といった **他個体へ書く**。

    ①は `agent._fire_reason`(発火権が残っているか)で判る。②は `_phone` が実際に
    何かをする枝へ入るかで判り、その枝分岐は `phone` stream の 1 draw で決まる。
    `RngHub.stream` は毎回**新しい Generator** を作るので、ここで `random()` を引いても
    後で `_phone` が引く値は 1 ビットも変わらない(= 覗き見・乱数消費ゼロ)。
    """
    if agent._fire_reason:                         # ①: 2 本目の熟慮が控えている
        return False
    if not agent.has_phone or agent.sleeping:      # ②: _phone へ到達しない
        return True
    ncfg = sim.netcfg
    if not ncfg["enabled"] or disaster_mod.infra_out(sim):
        return True                                # _phone は即 None(何もしない)
    if agent._search_queue:                        # 検索 → _hear_words(造語者へ書く)
        return False
    r = float(sim.hub.stream("phone", agent.id, step).random())
    return r >= float(ncfg["browse_prob"]) + float(ncfg["news_prob"])


class _JournalRelay:
    """このフェーズのジャーナル書き込みを退避する中継(第71 LlmJournal と同じ口)。

    ジャーナルは `seq` を書き込み順に振るので、**一括発行で入れ替わった順のまま**
    本物へ流すと ON/OFF でファイルが変わってしまう(tests/test_llm_journal.py の契約)。
    ここで受けて `_replay_decide_journal` が個体順に流し直す。record 以外の呼び出し
    (flush / path / seq …)は本物へそのまま委譲する。
    """

    def __init__(self, real, sink: list) -> None:
        self.__dict__["_real"] = real
        self.__dict__["_sink"] = sink

    def record(self, **kw) -> None:
        self._sink.append((self._real, kw))

    def __getattr__(self, name):
        return getattr(self.__dict__["_real"], name)


def _journal_owners(llm) -> list:
    """`journal` を持つ層(= CachedLLM)を重複なく列挙する(router / mind 配線を含む)。"""
    out, seen, stack = [], set(), [llm]
    while stack:
        x = stack.pop()
        if x is None or id(x) in seen:
            continue
        seen.add(id(x))
        if getattr(x, "journal", None) is not None:
            out.append(x)
        ch = getattr(x, "children", None)
        if isinstance(ch, dict):
            stack.extend(ch.values())
        d = getattr(x, "default", None)
        if d is not None:
            stack.append(d)
    return out


def _replay_decide_journal(sink: list, active: list) -> None:
    """退避したジャーナル行を**個体順**(= 逐次実行の書き込み順)で本物へ流す。

    熟慮の rng_key は `deliberate/<agent_id>/<step>` なので個体は行から一意に判る。
    同じ個体が 2 本撃った場合(返事 → 発火)は捕捉順がそのまま保たれる(安定ソート)。
    """
    if not sink:
        return
    pos = {int(a.id): i for i, a in enumerate(active)}

    def _key(row):
        parts = str(row[1].get("rng_key", "")).split("/")
        try:
            return pos.get(int(parts[1]), -1)
        except (IndexError, ValueError):
            return -1

    for real, kw in sorted(sink, key=_key):
        real.record(**kw)


def _reorder_decide_log(sim, ev0: int, lc0: int, frags: list) -> None:
    """遅延した個体の観測出力を**逐次実行と同じ並び**(個体単位の連結)へ戻す。

    `frags[i]` は個体 i が書いた区間 `(ev_a, ev_b, lc_a, lc_b)` の列(生成順)。
    このフェーズの間に L1/L1b へ書くのは `_decide` だけ(一括発行はジャーナルへしか
    書かない)なので、区間の連結は元の並びの完全な分割になる。分割が漏れていたら
    黙って並べ替えず**即座に落とす**(静かな L1 破壊を作らないため)。
    """
    if all(len(f) <= 1 for f in frags):            # 分割された個体が無い=既に逐次と同一
        return
    ev, lc = sim.logger.events, sim.logger.llm_calls
    ev_tail, lc_tail = ev[ev0:], lc[lc0:]
    new_ev, new_lc = [], []
    for one in frags:
        for a, b, c, d in one:
            new_ev.extend(ev_tail[a - ev0:b - ev0])
            new_lc.extend(lc_tail[c - lc0:d - lc0])
    if len(new_ev) != len(ev_tail) or len(new_lc) != len(lc_tail):
        raise RuntimeError(
            "batch_llm(熟慮)の観測出力の区間分割が壊れている"
            f"(L1 {len(new_ev)}/{len(ev_tail)} L1b {len(new_lc)}/{len(lc_tail)})。")
    ev[ev0:] = new_ev
    lc[lc0:] = new_lc


def _phase_decide_batched(sim, active: list, step: int, sim_min: int,
                          workers: int) -> list:
    """日中熟慮の一括発行。返り値は逐次経路と同じ `[(agent, action), ...]`。"""
    n = len(active)
    actions: list = [None] * n
    ev, lc = sim.logger.events, sim.logger.llm_calls
    ev0, lc0 = len(ev), len(lc)
    frags: list[list[tuple[int, int, int, int]]] = [[] for _ in range(n)]
    pending: list[tuple[int, object, dict]] = []
    stats = getattr(sim, "_batch_decide_stats", None)
    if stats is None:                              # ラン通算(観測用。L1 にも state にも出ない)
        stats = {"batched": 0, "serial_llm": 0, "no_llm": 0, "rounds": 0,
                 "requests": 0, "steps": 0, "max_batch": 0,
                 # ★厳密一致の証人(下の設計注記):遅延した個体で**パース不成立**が
                 #   起きた回数。0 のあいだ、batch 経路は逐次経路と機械的に同値である。
                 "deferred_fallback": 0}
        sim._batch_decide_stats = stats
    stats["steps"] += 1
    # LLM ジャーナル(第71)の書き込みも並び替えの対象なので、このフェーズのあいだだけ
    # 中継へ差し替えて退避する(finally で必ず戻す)。
    jsink: list = []
    jsaved = [(c, c.journal) for c in _journal_owners(sim.llm)]
    for _c, _j in jsaved:
        _c.journal = _JournalRelay(_j, jsink)
    try:
        _decide_rounds(sim, active, step, sim_min, workers, actions, frags,
                       pending, stats, ev, lc)
    finally:
        for _c, _j in jsaved:
            _c.journal = _j
        _replay_decide_journal(jsink, active)
    _reorder_decide_log(sim, ev0, lc0, frags)
    return actions


def _decide_rounds(sim, active: list, step: int, sim_min: int, workers: int,
                   actions: list, frags: list, pending: list, stats: dict,
                   ev: list, lc: list) -> None:
    """`_phase_decide_batched` の本体(ラウンド 0 → 一括発行 → id 順の再開)。"""
    # ---- ラウンド 0(id 順・逐次): 最初の要求まで走らせる ----
    for i, agent in enumerate(active):
        ea, la = len(ev), len(lc)
        gen = _decide_g(sim, agent, step, sim_min)
        try:
            req = gen.send(None)
        except StopIteration as stop:              # LLM を 1 本も撃たない個体=完了
            actions[i] = (agent, stop.value)
            stats["no_llm"] += 1
        else:
            if _deliberate_defer_ok(sim, agent, step):
                pending.append((i, gen, req))
                stats["batched"] += 1
            else:                                  # 安全弁: その場で逐次に走らせる
                actions[i] = (agent, _run_gen(sim, gen, first=req))
                stats["serial_llm"] += 1
        frags[i].append((ea, len(ev), la, len(lc)))
    # ---- ラウンド 1..: 未解決分を一括発行し id 順に再開する ----
    while pending:
        stats["rounds"] += 1
        stats["requests"] += len(pending)
        stats["max_batch"] = max(stats["max_batch"], len(pending))
        results = sim.llm.generate_many([r for _i, _g, r in pending],
                                        workers=workers)
        nxt: list[tuple[int, object, dict]] = []
        for (i, gen, _req), res in zip(pending, results):
            ea, la = len(ev), len(lc)
            try:
                req = gen.send(res)
            except StopIteration as stop:
                actions[i] = (active[i], stop.value)
            else:
                nxt.append((i, gen, req))
            for _e in ev[ea:]:                     # 厳密一致の証人(設計注記 ⑤)
                if _e.kind in ("fallback", "undefined_action"):
                    stats["deferred_fallback"] += 1
                    break
            frags[i].append((ea, len(ev), la, len(lc)))
        pending = nxt


# ---------------------------------------------------------------- 検索エンジン
def _search_index(sim, query: str) -> list[str]:
    """シミュ内検索エンジン(世界内データベース)。実APIは使わない(D13 再現性
    +架空世界の閉性)。索引 = 語彙の来歴 / ニュース / 実在 POI(OSM)。

    ★A9(第117 レーンE): 拡散回数は ``Item.transmissions_count``(int)から読む。
      以前は ``len(item.transmissions)`` = **観測用の明細台帳**を直接読んでいたので、
      台帳の実装(上限を掛ける・明細を外へ出す)を変えるとプロンプトが動いてしまった。
      counter は ``provenance.transmit`` が list と同じ 1 箇所で更新するので
      **値は常に len と同値** = 生成される文字列は 1 バイトも変わらない。"""
    results: list[str] = []
    item = sim.labels.text_to_item.get(query)
    if item is not None:
        src = "メディア発表" if item.creator == -1 else \
            f"{sim.agent_by_id[item.creator].name}が言い始めた"
        results.append(f"「{query}」: 最近使われ始めた言葉({src}、"
                       f"{int(getattr(item, 'transmissions_count', 0))}回拡散)")
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
    # ★レーン乙 ブロック7: 「メディア/不在はスキップ」を実際に成立させる(在場述語)。
    #   ``agent_by_id`` では退場した著者にも drive が積まれ、次の hydrate で捨てられる。
    author_meta = sim.present_agent(post["author"])
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
            # propagation_off(第78): SNS も「他者の書いた内容が自分の文脈に入る」経路。
            # 語彙の授受と本文の記憶化だけを切り、閲覧の発生・いいね/リシェアの抽選・
            # drive/覚醒/意見のスカラーは従来どおり(= 閲覧量と呼数の構造を保つ)。
            _prop_off = ablate_mod.propagation_off(sim)
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
                    # SNC v2 の F3(タイムライン発見。第117・既定 OFF は即 return =
                    # **乱数を 1 粒も引かない**=バイト一致)。いいねした投稿の著者を
                    # 確率 follow_like_p でフォローする。乱数は用途別の新 named stream
                    # "snc_follow" を (agent, step, post) で引くので、既存の draw 列
                    # (phone / sns_react)は 1 粒も動かない。
                    contact_mod.on_like(sim, agent, post, step, sim_min)
                if reshare:
                    _sns_react(sim, agent, post, "reshare", "sns_reshare",
                               step, sim_min)
            if _prop_off:                           # 本文は記憶へ入れない(合成文も入れない)
                pass
            elif affect_on and sim.affectcfg["salience_k"] > 0:
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
    """行為の適用(**唯一のディスパッチャ**)+ 来歴スコープの開閉(IF-1)。

    observer.llm_link=true のときだけ、行為 dict に積まれた一時キー ``_prov``
    (= その行為を決めた LLM 呼の ``(llm_call_id, role)``)を取り出し、
    この ``_apply`` のあいだ **行為者自身が出す L1 イベント**へ刻む
    (PROV の ``wasInformedBy`` 辺 1 本)。tools / P2 / verify / free_action の
    サブディスパッチャは 1 行も改変せずに巻き込める(logger 側で刻むため)。

    既定 OFF では ``_prov`` が**そもそも積まれない**ので ``prov is None`` の
    分岐しか通らない = ゴールデン L1 バイト一致(構造による保証)。

    IF-F(第100バッチ・observer.causality。既定 OFF)も同じ場所で**因果スコープ**を
    開閉する。「いまこの行為を適用中」の間に**その行為者自身**が出したイベントは、
    エンジンが「誰が起こしたか」を直接知っている(静的表の推定より強い証拠)。
    どちらも OFF なら従来どおり ``_apply_action`` を素通しする 1 本道しか通らない。
    """
    prov = provlink_mod.take(action)
    cause_on = causality_mod.enabled(sim)          # conf を 1 度読んで sim にキャッシュ
    if prov is None and not cause_on:
        _apply_action(sim, agent, action, step, sim_min)
        return
    if prov is not None:
        sim.logger.set_prov(prov[0], prov[1], int(agent.id))
    if cause_on:
        sim.logger.set_cause(causality_mod.AGENT, int(agent.id))
    try:
        _apply_action(sim, agent, action, step, sim_min)
    finally:
        if prov is not None:
            sim.logger.clear_prov()
        if cause_on:
            sim.logger.clear_cause()


def _apply_action(sim, agent, action: dict, step: int, sim_min: int) -> None:
    kind = action["type"]

    if kind in ("host_event", "post_flyer", "found_group", "propose",
                "open_venture", "job_search"):
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

    if kind == "verify":
        # 検証行動(第73バッチ Part B)。beliefs.verify_actions OFF では静かに無視
        # (=wander 相当)= 既定 OFF のイベント 0 件・状態変化ゼロ。裁定は truth_ledger に
        # 閉じる(engine は真偽台帳の中身を読まない=呼ぶだけ)。LLM 呼ゼロ・乱数ゼロ。
        # ---- IF-B(第94): 空振り(no_target / no_witness / no_channel)の通知 ------- #
        #  従来は L1 belief_verify に残るだけで**当人には何も届かない**(監査 §2-C)。
        #  ★裁定 module(truth_ledger.py)は **observer/metrics_spec.py の凍結 14 ファイル**
        #    なので 1 バイトも触らない。代わりにこの呼び出しが出した L1 の **outcome 1 語だけ**を
        #    読んで通知する(fact の ID も値も canary も読まない=漏洩契約は無風)。
        #  ★既定 silent ではスライスも走らない = 従来と完全同一。
        if not reject_mod.enabled(sim):
            truth_ledger_mod.apply_verify(sim, agent, action, step, sim_min)
            return
        _n0 = len(sim.logger.events)
        truth_ledger_mod.apply_verify(sim, agent, action, step, sim_min)
        reject_mod.note_verify(sim, agent, sim.logger.events[_n0:], step, sim_min)
        return

    if kind in ("plan", "recall", "reflect"):
        # ---- 明示分岐(監査 §5-2 の穴。第93バッチ IF-A)------------------------ #
        #  この 3 種は **熟慮(deliberate)経路では文脈外**の行為である:
        #    plan    … 朝の計画呼(cognition/planning.py)だけが消費する。日中の熟慮で
        #              返ってきても実行できる予定表の入れ物が無い。
        #    recall  … 内省の agentic pull 第 1 段(memory.agentic_pull)が消費する。
        #    reflect … 夜の内省(cognition/reflection.py)が消費する。
        #  従来はここまで if 連鎖を素通りして **無音の no-op** になっていた。
        #  KNOWN_ACTIONS に含まれるので undefined_action にも落ちず、fallback にも
        #  数えられない = 「LLM は行為を主張したのに世界のどこにも記録が無い」観測の穴。
        #  ★世界への作用は従来どおり **完全 no-op**(1 バイトも動かさない)。
        #    観測だけを足す = observer.llm_link ON のときにこの 1 件を出す。
        #  ★fallback 集計との整合: kind は既存の "fallback" を再利用するので
        #    observer.llm_health の llm_fallback_rate の分子に入る。payload の
        #    reason("misrouted_action" / 従来の "parse_error")で事後に切り分けられる
        #    (_log_reject の undefined_action 振り分けと同じ「分子は排他」の流儀)。
        if provlink_mod.enabled(sim):
            sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                                 kind="fallback", x=agent.x, y=agent.y,
                                 payload={"reason": "misrouted_action",
                                          "action": kind}))
        return

    if kind == "stay":
        # 沈黙の第一級化(第70バッチ IDEA②)。stay は従来どおり何もしない(=この分岐が無くても
        # 下の if 連鎖はすべて外れて no-op に落ちる)。「関わらないことを選んだ」という**意思表示**
        # だけを、freedom.explicit_nothing=true のときに 1 件記録する。既定 OFF ではイベント 0 件
        # =従来の stay と完全同一(ゴールデン L1 バイト一致)。乱数ゼロ・LLM 呼ゼロ。
        fcfg = getattr(sim, "freedomcfg", None)
        if (action.get("reason") == "chosen_nothing"
                and fcfg is not None and fcfg.get("explicit_nothing")):
            sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                                 kind="stay", x=agent.x, y=agent.y,
                                 payload={"reason": "chosen_nothing",
                                          "node": agent.node}))
        return

    if kind == "go_to_bed":
        # 自宅前 → 実在の住宅建物に入って就寝(v3: 路上睡眠の解消)。
        # 立退き中(制度深化3・既定 False=不変)は自宅建物に入れない=路上就寝。
        if (agent.home_building and not agent.evicted
                and sim.city.has_building(agent.home_building)):
            bld = sim.city.building(agent.home_building)
            agent.building = bld["id"]
            agent.floor = floors_mod.clamp(sim, bld, agent.home_floor)
            agent.x, agent.y = bld["centroid"]
            sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                                 kind="enter_building", x=agent.x, y=agent.y,
                                 payload={"building": bld["id"],
                                          "name": bld["name"] or "自宅",
                                          "floor": agent.floor,
                                          "levels": int(bld["levels"]),
                                          "home": True}))
        # 在宅覚醒 HOME_AWAKE(β9・既定 OFF): routine が awake=True を付けたときだけ
        # 「入館したが寝ない」で止める。既定 conf では awake キーが存在しない
        # (routine._home_awake_begin が常に False)= ここは 1 ビットも変わらない。
        if action.get("awake"):
            return
        kind = "sleep"

    if kind == "sleep":
        agent.sleeping = True
        agent.activity = ""
        agent.sleep_until = step + agent.sleep_steps
        reflect_timing_mod.arm(sim, agent, step)   # 就寝直後に記憶整理(内省)。
        #                                            RFX-A: 早期発火済みならこの 1 回を見送る
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                             kind="sleep_start", x=agent.x, y=agent.y,
                             payload={"until_step": agent.sleep_until,
                                      "building": agent.building}))
        return

    if kind == "move_to":
        path, used_mode = sim.router.route(agent.node, action["dest"],
                                           action.get("mode", "walk"))
        if len(path) < 2:
            # IF-B(第94): 経路が張れない = 監査 §2-C の無音拒否(`len(path)<2: return`)。
            # 既定 silent では 1 バイトも動かない(ゴールデン維持)。
            reject_mod.notify(sim, agent, "move_to", "unreachable", step, sim_min)
            return
        agent.route = path[1:]
        agent.edge_offset = 0.0
        agent.dest = action["dest"]
        agent.trip_mode = used_mode
        agent.exit_intent = bool(action.get("exit"))
        agent.homing = bool(action.get("homing"))
        agent._pending_stay = int(action.get("stay_steps", sim.clock.dur_steps(2)))
        agent._pending_activity = str(action.get("activity", ""))
        agent._ride_pending = action.get("ride")   # 交通機関の乗車(到着時に課金)。無ければ None
        agent.activity = "commuting" if action.get("activity") == "commuting" else ""
        # SUMO ライブ連成タクシー配車 v-Ride-1(既定 OFF=None=no-op=used_mode 不変=バイト一致)。
        # ON 時のみ予約注入→配車待ち/超過を到着 step へ反映。未配車なら trip_mode を walk へ差し替える。
        _taxi_live_dispatch(sim, agent, action["dest"], step, sim_min)
        # 実バスダイヤ静的表 v-Ride-2(既定 OFF=ride に wait_s 無し=no-op=バイト一致)。
        # ON 時のみバスの次便待ち+区間所要を到着 step の追加待ちへ反映(taxi と同じ hold 機構)。
        _bus_ride_hold(sim, agent, action["dest"], step)
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                             kind="route_start", x=agent.x, y=agent.y,
                             payload={"dest": action["dest"],
                                      "dest_name": sim.city.node_name(action["dest"]),
                                      "mode": agent.trip_mode,
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
            # 3D-U0(world.floor_clamp・既定 OFF=そのまま): 職場/バイト先の階は POI 由来で
            # 建物の実階数を超えうる(実データ 20 件)。ON のときだけ表示側と同一規則で丸める。
            agent.floor = floors_mod.clamp(sim, bld, int(action["floor"]))
        else:
            agent.floor = floors_mod.clamp(
                sim, bld, int(rng.integers(1, int(bld["levels"]) + 1)))
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
            # PRICE-B(既定=空/OFF は恒等=1 ビットも変わらない): 事前公表の時間帯料金表 ×
            # 閉店前見切り。_charge_meal と**同じ合成順**で乗せる。
            price = commerce_mod.apply_price(sim, price, cat, agent.node, sim_min)
            item = None
            if price is not None and _goods_on(sim):   # 物流①②: 実在庫を消費・買った物を付与(決定論)
                ok, item = goods_mod.on_purchase(sim, agent, cat, step, sim_min)
                if not ok:                         # 実在庫の品切れ → 購入不成立
                    price = None
            if price is not None and agent.money >= price:   # 品切れ(None)= 購入抑制
                _spend(sim, agent, price, cat, step, sim_min, item=item)
                # CRWD(既定 OFF=no-op): 買えた=成立時にその場の混み具合を不満へ写す。
                commerce_mod.apply_crowding(sim, agent, cat, step, sim_min)
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
        agent.floor = floors_mod.clamp(sim, agent.building, int(action["floor"]))
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
        # ★レーン乙 ブロック7: 在場述語。相手が街を出ていると ``loc``/``sleeping`` は
        #   脱水時のスナップショットのままなので下の物理述語を素通りし、受信側の
        #   記憶・関係・drive・覚醒が丸ごと捨てられる(送信側の _contact だけ実在=
        #   関係台帳が片側だけ育つ)。
        recipient = sim.present_agent(to) if to is not None else None
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                             kind="dm", x=agent.x, y=agent.y,
                             payload={"to": to, "text": action["text"]}))
        # 会話(DM)からの予定抽出→双方の帳簿へ記入(既定 OFF=no-op)。to にも入れる。
        # propagation_off: 受信者の帳簿へは書かない(送信者自身の抽出は自己読み取りなので残す)。
        _prop_off = ablate_mod.propagation_off(sim)
        _record_appointments(sim, agent, action["text"],
                             ablate_mod.heard_ids(
                                 sim, [to] if to is not None else []),
                             step, sim_min)
        if recipient is not None and recipient.loc != "outside" \
                and not recipient.sleeping:
            # propagation_off: 本文は受信者の記憶へ入れない(合成文も入れない)。
            # `_last_dm_from`(誰から来たか)と drive/覚醒は残す=「着信はあったが内容が
            # 文脈に入らない」= DM の発生量と返信の発火構造は保たれる。
            if not _prop_off:
                recipient.remember(f"{agent.name}からメッセージ:「{action['text']}」",
                                   kind="dm")
            recipient._last_dm_from = agent.id
            v_dm = valence(action["text"])
            # 関係台帳(Wave G2 の交流符号=DM の感情価)。OFF は従来の record_contact と完全同一。
            # 第65バッチ: DM も文面から同じ係数を1回算出して両方向へ(既定 OFF=1.0)。
            _dtext = ablate_mod.heard_text(sim, action["text"])
            mag_dm = _quality_mag(sim, agent, recipient.id, _dtext, sim_min)
            _contact(sim, recipient, agent.id, agent.name, _dtext,
                     v_dm, step, sim_min, mag_dm)
            _contact(sim, agent, recipient.id, recipient.name, "",
                     v_dm, step, sim_min, mag_dm)
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
            # 噂の相乗り IF-C(第95・既定 OFF=即 return=バイト一致)。**本文は触らない**:
            # 送った DM の文面はそのままで、話者が知っている噂だけが「別チャネルの伝聞」として
            # 受信者へ渡る(transmission 1 件 + 受信者の記憶 1 行)。LLM 呼ゼロ・乱数ゼロ。
            rumors_mod.on_talk(sim, agent, [recipient], "dm", step, sim_min)
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
        # 場所の意味づけ D1: 発生ノードを渡す(束縛は labels 側・既定 OFF は無視されて no-op)。
        # node を渡すのはこの熟慮 coin_label 経路だけ = tools の group 名/提案文は束縛しない。
        item = sim.labels.coin(agent, action["word"], step=step, sim_min=sim_min,
                               logger=sim.logger, context=context,
                               node=getattr(agent, "node", None))
        if item is None:                           # constrained で棄却された = 沈黙
            return
        action = {"type": "speak", "text": action["text"], "use_items": [item.text]}
        kind = "speak"                             # 造語はそのまま口に出す

    if kind == "speak":
        radius = float(sim.cfg.world.perception_radius_m)
        # ---- ATT 層A(第143・`world.attention`)の**唯一の挿入点** = S15 と同一点 ----
        #  既定(enabled:false / mode:"distance")では `salience_on` が False を返し、
        #  下の 1 行だけが走る = 第141 S15 と**バイト同一**(ゴールデン維持)。
        #  ON(mode:"salience")では、聞こえた全員から priority 上位 k_i 人(+ 自己名の
        #  貫通)だけを選び、その集合が hear の L1 / 記憶 / 関係 / 覚醒 / SNC 遭遇 /
        #  意見更新 / 噂 へそのまま流れる(= 非通過者には**何も起きない**・Cherry 1953)。
        # ---- 有界化(hearer-cap-plan §2 作用点0/B)。**両方 既定なら現行の呼び出しそのまま** ----
        #   作用点0 = 声の段階(発話文脈 → 実効半径)。作用点B = S15 cap を列挙段へ配管。
        #   下の `_attention_limited` は保険として残す(同一規則なので二重適用は冪等)。
        _b_cap, _b_rad = _speak_bounds(sim, agent, step)
        if _b_cap or _b_rad is not None:
            _att_all = hearers_of(agent, sim.agents, radius,
                                  cap=_b_cap, radius_eff=_b_rad)
        else:
            _att_all = hearers_of(agent, sim.agents, radius)
        if attention_mod.salience_on(sim):
            _att_words = [w for w in action.get("use_items", []) if w in agent.adopted]
            hearers, _att_crowd = attention_mod.select(
                sim, agent, _att_all, action["text"], _att_words, step,
                radius=radius, valence_abs=abs(valence(action["text"])),
                pref_ids=attention_mod.reply_target_of(agent, step))
            attention_mod.note_attended(hearers, agent.id)
        else:
            hearers = _attention_limited(sim, agent, _att_all)
        # ablate.propagation_off(第78・既定 OFF=False=以下は全て従来経路)。
        # 「発話は通常どおり生成する / 内容は他者の文脈へ一切入らない」を実現するための
        # 唯一の判定。**聞き手集合・イベント・返答権の付与・接触台帳は触らない**
        # (= 会話量と LLM 呼数の構造を保つ)。切るのは内容だけ。
        _prop_off = ablate_mod.propagation_off(sim)
        # 返答保証: 返答権を渡す相手を決定論選択(既定 nearest=最寄り1人=現行と同一。
        # prompts.reply_partner=closeness で関係加重の宛先へ)。hearer 集合は不変=聞く人は
        # 変わらない・返す人だけが変わる。乱数なし=R1。
        _att_addressed = -1              # ATT 層B: 「宛先」= 返答権を渡す相手(boost の対象)
        if hearers:
            partner = _select_partner(sim, agent, hearers)
            _att_addressed = int(partner.id)
            if (not partner.sleeping and partner.conv_turns_left > 0
                    # 在宅覚醒(HOME_AWAKE β9・既定 OFF=pair_open は常に True=従来経路)。
                    # 在宅覚醒中の相手には睡眠中と同じ扱いで返答権を渡さない = この機能が
                    # **返事の LLM 呼を 1 本も足さない**ことを構造で保証する(R1)。
                    # evening_talk ON のときだけ「同一世帯 かつ 両者 HOME_AWAKE」の
                    # ペアに限って開く(同居人以外・来客・外部への経路は閉じたまま)。
                    and home_awake_mod.pair_open(agent, partner, sim)
                    and step >= partner.conv_cooldown_until
                    # 第87: **会話は両者 ENGAGED が成立条件**(§8 補助規則 2)。話者が
                    # AUTOPILOT(通りすがりの独り言・定型で流した側)なら返答権を渡さない
                    # = 相手は返さない = 挨拶で終わる。既定 OFF は常に True=従来経路。
                    and engaged_mod.handoff_ok(sim, agent, partner)):
                # propagation_off: 返答権は渡す(呼数不変)が、相手の状態に本文を残さない。
                partner._reply_to = (agent.id,
                                     ablate_mod.heard_text(sim, action["text"]))
            # 第87: closing move(別れの挨拶)。**双方**が出したときだけ解消になる
            # (Schegloff & Sacks 1973 の terminal exchange = 片方の pre-closing は申し出)。
            # 既定 OFF・end 欄なしなら完全 no-op。
            if action.get("end"):
                engaged_mod.note_closing(sim, agent, partner, step, sim_min)
            # 対話履歴(prompts.dialog_history=true のみ)。話者と相手の相手別バッファへ同一発話を
            # 積む(次回 social/reply のプロンプトに直近2往復を注入)。OFF は _dialog_on=false →
            # 一切触らない=状態変化なし=バイト一致。
            if _dialog_on(sim):
                _dialog_push(agent, partner.id, agent.name, action["text"])
                if not _prop_off:                  # 相手側へは積まない(自己連続性だけ残す)
                    _dialog_push(partner, agent.id, agent.name, action["text"])
        words = [w for w in action.get("use_items", []) if w in agent.adopted]
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                             kind="speak", x=agent.x, y=agent.y,
                             payload={"text": action["text"], "items": words,
                                      "hearers": [a.id for a in hearers]}))
        # 会話(対面)からの予定抽出→話者と聞き手全員の帳簿へ記入(既定 OFF=no-op)。
        # propagation_off: 聞き手の帳簿へは書かない(話者自身の抽出は自己読み取りなので残す)。
        _record_appointments(sim, agent, action["text"],
                             ablate_mod.heard_ids(sim, [h.id for h in hearers]),
                             step, sim_min)
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
        # SNS・知り合い形成 v2(SNC。第117・net.contact_formation)。**既定 OFF は False**
        # = 下の分岐は 1 度も通らず従来経路がそのまま走る(バイト一致)。step ごとに
        # 何度も conf を辿らないよう、聞き手ループの外で 1 度だけ引く。
        _snc_on = contact_mod.enabled(sim)
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
            # propagation_off: **聞き手の記憶に発話文を積まない**(合成文の代入もしない=
            # 他条件に現れない痕跡を作らないため)。記憶に何も残らない=「静かな街」になる。
            if not _prop_off:
                hearer.remember(f"{agent.name}が「{action['text']}」と言っていた",
                                kind="heard",
                                importance_bonus=_imp_bonus(
                                    sim, hearer, novelty=(1.0 if words else 0.0)))
            # ★SNC v2(第117)の唯一の置換点。**既定 OFF ではこの 1 行がそのまま走る**
            #   (= 聞こえた全員と知り合い + 自動フォロー。10k×2 日で follows/contacts が
            #   実質完全グラフ 19,738×約19,400 → RSS 100GiB → 25 万で OOM 死する当のバグ)。
            #   ON では**傍聴者に contacts も follows も生やさない**。代わりに遭遇回数だけを
            #   有界に数え、同一相手と k 回目で挨拶級の知り合いへ昇格する(C3)。
            #   ★上の `hearer.remember`(聞いた内容の記憶)は ON/OFF で**不変**なので、
            #     新語・情報の対面伝播は 1 ビットも死なない(設計書 §2「消すもの・変えないもの」)。
            if _snc_on:
                contact_mod.note_encounter(sim, hearer, agent, step, sim_min)
            else:
                sim.net.add_contact(agent.id, hearer.id)  # 対面で知り合い→DM可・フォロー
            # ATT 層B(第143・`cognition.attention_block`・既定 OFF は即 return=1 バイトも
            # 動かない): **自分宛て**の発話(= 返答権を渡された相手)を受けた step だけ、
            # その話者の注意スロットの salience を上げる(イベント駆動 boost・乱数ゼロ)。
            if hearer.id == _att_addressed:
                attention_mod.note_addressed(sim, hearer, agent.id)
            # 関係台帳(Wave G2 の交流符号=発話の感情価)。OFF は従来の record_contact と完全同一。
            # 第65バッチ: 交流の**量**に載る係数を(話者,聞き手)1組につき1回だけ算出し両方向へ
            # 同じ値で渡す(既定 OFF=1.0=従来と同値)。算出は society 層=engine は運ぶだけ。
            # propagation_off: 本文を渡さない(quality は中立係数へ・rel["last"] も空になる)。
            _htext = ablate_mod.heard_text(sim, action["text"])
            mag = _quality_mag(sim, agent, hearer.id, _htext, sim_min)
            _contact(sim, hearer, agent.id, agent.name, _htext,
                     v_text, step, sim_min, mag)
            _contact(sim, agent, hearer.id, hearer.name, "",
                     v_text, step, sim_min, mag)
            # propagation_off の遮断は _hear_words 内で一元化(語彙は渡さず「話しかけられた」
            # という交流の量だけ残す)。ここは従来どおり実 words を渡す。
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
        # 噂の相乗り IF-C(第95・既定 OFF=即 return=バイト一致)。**発話テキストは 1 バイトも
        # 変えない**(引数にも取らない): 話者が spreader である噂だけが「別チャネルの伝聞」として
        # 聞き手へ渡り、聞き手の記憶に定型文 1 行が載る(transmission = 既存 API)。
        # 既知の相手に語ってしまったら Maki-Thompson 流に stifler 化する(= 噂が止まる唯一の理由)。
        # LLM 呼ゼロ・乱数ゼロ・世界状態(位置・所持金・関係・drive)は 1 つも動かさない。
        rumors_mod.on_talk(sim, agent, hearers, "face", step, sim_min)


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
    payload = {"amount": round(float(amount), 1),
               "paid": round(float(paid), 1),
               "carry": round(agent.rent_due, 1),
               "account": round(agent.account, 1),
               "phase": phase}
    # O4 権利行(第114 優先2。既定 OFF=None=1 行も通らない): 登記簿の own 行に家主が
    # 居るなら**その口座へ着金させる**(域内 org の預金 / 個人家主の口座)。家賃が街の外へ
    # 出ていくのは「家主が域外に居る」ときだけになる = 賃料が街の中で循環する。
    payee = assets_mod.settle_rent(sim, agent, float(paid), step, sim_min)
    if payee is not None:
        payload["payee"] = payee
    # IF-E2(既定 OFF=キーなし): 家主が街に居ない → 域外の不動産所有者(RoW)が受け取る。
    elif sfc_mod.enabled(sim) and float(paid) > 0.0:
        payload["payee"] = sfc_mod.row_out(sim, "rent_landlord", float(paid))
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                         kind="rent", x=agent.x, y=agent.y, payload=payload))


# --------------------------------------------------------- 死者の除外規約(★B7・第117 レーンE)
# `health._die` は **sim.agents から個体を抜かない**(他レーンが持つ反復の前提を壊さないため)。
# 死は `dead=True` + `loc="outside"` + 実質無限の `return_at` で表現され、`sick` は
# `_clear_severity` で False に戻る。したがって「病気だから飛ばす」系の門は死者を素通しし、
# 経済 5 フェーズ(月給まとめ・家賃引落・WAGE 日次清算・日銭/利息/固定費・公務員給与・困窮者給付)
# は死者の口座を動かし続けていた(相続で空にした財布へ給料が振り込まれる)。
# 除外規約は `_work_office_output` / `_phase_work_service`(レーン甲)と同じ `dead` 判定を使う。
#
# ★**除外条件は `dead` だけ**である。`loc == "outside"` を足してはならない:
#   プール回転で街の外に居る個体へ、戻った最初の清算日に給料日を遡って払う
#   **不在時キャッチアップ支給**(第112 WAGE の `_wage_fires` 区間判定)が設計仕様であり、
#   「振込は在不在に関わらず着金する」という現実の意味論そのものだからである。
#   在場を条件にした瞬間、通勤者・回転層の給与が構造的に消える。
def _phase_accounts_day(sim, step: int, sim_min: int) -> None:
    """口座 E5 の暦日境界(1日=144step、run開始日=1日として day%30)。

    給料日(payday_dom): 月給者(本業日給を持つ会社員・店員)へ economy.wages×勤務日数を
    まとめ支給(口座へ)。給料日の翌日: 家賃 = 月収相当(period_income)×rent_share を口座
    から引き落とし。残高不足は rent_due に繰越し、翌日以降に回収+money_pressure が効く。
    すべて決定論(乱数なし)。来街者は街の外に家=口座/家賃なし。

    A2(第94バッチ OBS-U2): 144 の直書きを `clock.steps_per_day` へ。★`clock.day()` には
    しない — day() は sim_min//1440(深夜0時境界)なので start_tod="07:00" では境界 step が
    102 になり、Δt=10 でも会計日が動いて golden が壊れる。`step // steps_per_day` なら
    Δt=10 で `step // 144` と厳密同値(= 開始ブロック基準という現行の定義そのまま)。"""
    block_day = step // sim.clock.steps_per_day     # run 開始ブロック=0(=暦の1日目)
    if block_day == getattr(sim, "_acct_day", -1):
        return
    sim._acct_day = block_day
    acc = sim.economy["accounts"]
    dom = block_day % 30 + 1                        # 暦の何日目(1..30)。run開始=1日
    payday = int(acc["payday_dom"])
    rent_dom = payday % 30 + 1                      # 給料日の翌日
    share = float(acc["rent_share"])
    for agent in sim.agents:
        if getattr(agent, "dead", False):
            continue                            # ★B7: 死者(_die は sim.agents から抜かない)
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
            # IF-E2(既定 OFF=無視): 月給まとめの流出も配属 org の預金から(§3-a の初期残高が効く点)
            _pay_wage(sim, agent, salary, step, sim_min, source="salary",
                      payer_org=getattr(agent, "org_id", None))
        if dom == rent_dom and not agent.evicted:      # 立退き中=住居なし→家賃は発生しない
            rent = agent.period_income * share         # 月収相当 × rent_share
            inc0 = agent.period_income                 # 控除前の月収(与信の income に使う)
            agent.period_income = 0.0
            if rent > 0.0:
                if agent.account < rent:               # 現金不足点: 不足分を融資で補填(bank ON 時)
                    _maybe_loan(sim, agent, rent - agent.account, step, sim_min, income=inc0)
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
        # A6(第94バッチ OBS-U2): 日 → step の 144 直書きを Clock へ(Δt=10 で厳密同値)
        agent.bankrupt_until = (step + int(acc["bankruptcy_restrict_days"])
                                * sim.clock.steps_per_day)
        tools = getattr(sim, "tools", None)
        had_venture = bool(tools is not None
                           and tools.force_close_venture(sim, agent, step, sim_min,
                                                         reason="bankruptcy"))
        bk_payload = {"debt": round(debt, 1), "seized": round(seized, 1),
                      "keep": round(keep, 1), "until_step": agent.bankrupt_until,
                      "venture_closed": had_venture}
        # IF-E2(既定 OFF=キーなし): 圧縮された資産の債権者が街に居ない → RoW が受け取る。
        if sfc_mod.enabled(sim) and seized > 0.0:
            bk_payload["payee"] = sfc_mod.row_out(sim, "seizure", seized)
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                             kind="bankruptcy", x=agent.x, y=agent.y,
                             payload=bk_payload))
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


# -------------------------------------------------- 賃金多様性 WAGE(第112・既定 OFF)
# 設計の要点(なぜ既存の 2 経路に足さず、新しい日次清算フェーズを 1 本立てるのか):
#   ① 既存の支給点 `_settle_work` は **建物からの退館イベント**に吊られている。路面の職場
#      (work_building 空)・日跨ぎシフトの個体は退館を出さないので、そこに月給を吊ると
#      構造的に無給のままになる。だから勤務日は「在場 × 勤務窓 × 暦の平日ゲート」という
#      routine.in_work_window と同基準の決定論カウントで数える。
#   ② 既存の月給まとめ `_phase_accounts_day` は `if agent.visitor: continue` の内側にある。
#      これは**家賃・立退き・破産のための門**であって賃金のための門ではない(域外に住む
#      通勤者にこの街の家賃は発生しないが、給料は出る)。家賃側の門は 1 バイトも触らず、
#      賃金だけがこの新経路を通る = L2(域内従業者 224,240 人)に給料が届く。
#   ③ 給料日は `block_day % 30 + 1` では**実暦の 25 日に永遠に到達しない**(10 日ランの
#      実測)。実暦判定は新トグル(wage_profile.calendar)の配下にだけ置き、OFF のときは
#      口座 E5 と同じ 30 日周期の式をそのまま使う(既存経路を 1 行も動かさない)。
#: 不在中に過ぎた給料日を遡って探す上限(日)。プールの回転で数日抜ける層を拾えれば十分で、
#  ここを無限にすると長期ランの日次コストが線形に伸びる。
_WAGE_CATCHUP_MAX = 45


def _wage_cfg(sim) -> dict | None:
    """economy.wage_profile(ON のときだけ dict)。OFF・経済 OFF は None=以降を 1 行も通らない。"""
    if not _economy_on(sim):
        return None
    cfg = (getattr(sim, "economy", {}) or {}).get("wage_profile")
    return cfg if cfg and cfg.get("enabled") else None


def wage_assign(sim, agent, record: dict | None = None) -> dict | None:
    """1 体へ賃金プランを与える(ON 時のみ・冪等・決定論・乱数ゼロ)。OFF は常に None。

    simulation.build_pool_agent(日次ローテーションの再実体化)と `_ensure_wage_profile`
    (名簿経路の起動時 1 回)の両方から呼ばれる唯一の入口。"""
    cfg = _wage_cfg(sim)
    if cfg is None:
        return None
    return economy_mod.assign_wage_plan(agent, getattr(sim, "orgs", None), cfg,
                                        record=record)


def _wage_covered(sim, agent) -> bool:
    """本業の賃金を WAGE の日次清算が持っている個体か(`_settle_work` の二重支給ガード)。

    既定 OFF では `_wage_cfg` が None を返して即 False = 既存の分岐と完全同一。
    第114 レーン 1a: L5 役割職(§ROLE)も日次清算が持つので True を返す。"""
    if _role_plan(sim, agent) is not None:
        return True
    return wage_assign(sim, agent) is not None


def _wage_dom(sim, cfg: dict, day: int) -> tuple[int, int, int]:
    """暦日インデックス → (月の何日, その月の日数, 月)。

    calendar=true かつ world.calendar 有効なら**実暦**。それ以外は口座 E5 と同じ
    「run 開始日=1 日」の 30 日周期(短いランでも給料日が必ず来る近似)。"""
    cal = getattr(sim, "calendarcfg", None) or {}
    if cfg["calendar"] and cal.get("enabled"):
        sm = int(day) * 1440
        d = calendar.date_of(cal, sm)
        return int(d.day), int(calendar.days_in_month(cal, sm)), int(d.month)
    return int(day) % 30 + 1, 30, int(day) // 30 + 1


def _wage_fires(sim, cfg: dict, last_day: int, today: int, dom_target: int,
                months: tuple | None = None) -> bool:
    """(last_day, today] に指定の支給日が 1 度でも来たか。

    「来たか」を区間で見るのが**不在時キャッチアップ**の実体である: 給料日に街を出て
    いた個体(プール回転で退場中)は、戻った最初の清算日にその給料日を拾う。振込は在不在に
    関わらず着金するという現実の意味論を、L1 イベントは本人が実在する日に出す形で満たす。
    dom_target=0 は「月末」。"""
    start = max(int(last_day) + 1, int(today) - _WAGE_CATCHUP_MAX)
    for d in range(start, int(today) + 1):
        dom, ndays, month = _wage_dom(sim, cfg, d)
        if months is not None and month not in months:
            continue
        if dom == int(dom_target) or (int(dom_target) == 0 and dom == ndays):
            return True
    return False


def _wage_worked_today(sim, agent, sim_min: int) -> bool:
    """今日この個体が働いたか。`routine.in_work_window` と**同じ基準**(時刻の条件だけ外す)。

    病欠・勤務窓なし(未就業/失職)・暦の休日は働かない。在場していること自体は
    呼び出し側(sim.agents の走査)が保証している = 在場が勤務日の資格そのもの。

    ★第144 `respect_work_days`(既定 false = 旧コードと 1 バイト同一): 暦ゲートの分岐は
      `routine.in_work_window` と**同一の式**でなければならない —— 食い違うと「働いたのに
      賃金が出ない / 休んだのに賃金が出る」という会計の破れになるので、テストが同値を固定する。
    """
    if getattr(agent, "sick", False):
        return False
    if int(getattr(agent, "work_start_min", -1)) < 0:
        return False
    cal = getattr(sim, "calendarcfg", None) or {}
    if cal.get("enabled") and cal.get("weekday_work"):
        spec = str(getattr(agent, "work_dow", "") or "") if cal.get("respect_work_days") else ""
        if spec:
            if not calendar.days_match(spec, calendar.weekday_of(cal, sim_min)):
                return False
        elif not calendar.is_workday(cal, sim_min):
            return False
    return True


def _wage_settle(sim, agent, plan: dict, cfg: dict, day: int,
                 step: int, sim_min: int) -> None:
    """1 体ぶんの日次清算(勤務日カウント → 給料日 → 賞与)。決定論・乱数ゼロ。"""
    last = int(getattr(agent, "wp_settled_day", -1))
    if day <= last:                                    # 同じ日を二度清算しない
        return
    worked = _wage_worked_today(sim, agent, sim_min)
    org = plan["org"]
    if plan["period"] == "daily":                      # 日給者: 働いた日にその日ぶんを受け取る
        if worked:
            _pay_wage(sim, agent, plan["daily"], step, sim_min,
                      source="daily", payer_org=org)   # 年換算は既存式(日給×245)が正しい
        agent.wp_settled_day = day
        return
    if worked:
        agent.wp_days = int(getattr(agent, "wp_days", 0)) + 1
    pay_due = _wage_fires(sim, cfg, last, day, plan["payday"])
    bonus = plan["bonus"]
    bonus_due = bool(getattr(agent, "wp_bonus_pending", False)) or (
        bonus is not None
        and _wage_fires(sim, cfg, last, day, bonus["dom"],
                        months=tuple(bonus["months"])))
    if pay_due:
        amount = economy_mod.salary_amount(plan, getattr(agent, "wp_days", 0), cfg)
        agent.wp_days = 0
        if amount > 0.0:
            agent.last_salary = amount
            _pay_wage(sim, agent, amount, step, sim_min, source="salary",
                      payer_org=org, annual_income=plan["annual"])
        # ★同一 agent・同一 step に賃金を 2 本重ねない: analyze_accounting の源泉税の帰属
        #   突合は |gross−tax.base| の**最初の一致で break** するので、重ねると誤帰属する。
        #   賞与は翌清算日へ持ち越す(支給日は元々給料日と重ならないよう選んである)。
        agent.wp_bonus_pending = bool(bonus_due)
    elif bonus_due:
        amount = economy_mod.bonus_amount(plan)
        agent.wp_bonus_pending = False
        if amount > 0.0:
            _pay_wage(sim, agent, amount, step, sim_min, source="bonus",
                      payer_org=org, annual_income=plan["annual"])
    agent.wp_settled_day = day


def _ensure_wage_profile(sim) -> None:
    """賃金プランの初回割当(step 先頭で 1 回)。OFF は完全 no-op。

    ★`_sfc_arm` の**前**に置くこと: IF-E2 の org 初期預金は「配属者の日給合計 ×
      month_days × σ」なので、agent.wage が WAGE の値に載る前に配ると初期預金だけが
      旧 3 値のまま取り残される。pool 経路は build_pool_agent が既に配っているので、
      ここは名簿経路(agents.personas_file 指定)のための後追い 1 回。"""
    if _wage_cfg(sim) is None or getattr(sim, "_wage_armed", False):
        return
    sim._wage_armed = True
    for agent in sim.agents:
        wage_assign(sim, agent)


# ------------------------------------------- L5 役割職の賃金 §ROLE(第114 レーン 1a)
# 対象は 3 職(タクシー運転手 350 / 配信者 16 / 議員 34)。どれも「勤務窓を与える経路が
# 1 本も無い」ため第112 WAGE の対象条件②を満たせず日給 0 円のまま残っていた層である。
# ★行動は 1 バイトも変えない: 勤務窓も職場 POI も与えず、**支給だけ**を在場に吊る。
# ★二重支給ガードは 3 重に張ってある:
#   ① `economy.assign_wage_plan` が役割職には通常プランを作らない(§ROLE のコメント)
#   ② `_wage_covered` が True を返すので `_settle_work` の都度払いが走らない
#      (そもそも `wage_amount` は WAGE_CAT に無い 3 職へ 0 を返すので実額も 0)
#   ③ `gig_profile` は WAGE_CAT=="自営" の層にしか出ないので `_phase_daily` の日銭と
#      重ならない(3 職はどれも WAGE_CAT に無い)
def _role_plan(sim, agent) -> dict | None:
    """1 体の L5 役割職プラン(ON のときだけ dict)。OFF・非該当は None=以降を通らない。

    ★「この街で働いているか」は通常プラン(`wage_target` ③)と同じ基準を使う:
      居住者 / org 配属 / 通勤者 のいずれか。L4(非定期来街)に同名の職業タグが
      紛れ込んでも支給しない。名簿の L5 3 職はタクシー/配信者が commute=true、
      議員が居住者なので全員この門を通る。"""
    cfg = _wage_cfg(sim)
    if cfg is None:
        return None
    plan = economy_mod.role_pay_of(getattr(agent, "occupation", ""), cfg)
    if plan is None:
        return None
    if getattr(agent, "visitor", False) and not getattr(agent, "commute", False) \
            and not getattr(agent, "org_id", None):
        return None
    return plan


def _role_settle(sim, agent, plan: dict, cfg: dict, day: int,
                 step: int, sim_min: int) -> None:
    """1 体ぶんの日次清算(歩合はその日ぶん・議員報酬は給料日に月額)。決定論・乱数ゼロ。"""
    last = int(getattr(agent, "rp_settled_day", -1))
    if day <= last:                                    # 同じ日を二度清算しない
        return
    key = str(getattr(agent, "pool_pid", "") or getattr(agent, "id", 0))
    if plan["kind"] == "gig":
        # 歩合: 病臥の日は稼ぎが立たない(車も出せず配信もできない)。在場は
        # 呼び出し側(sim.agents の走査)が保証している = 在場が稼働の資格そのもの。
        if not getattr(agent, "sick", False):
            amount = economy_mod.role_gig_amount(plan, key, day)
            if amount > 0.0:
                _pay_wage(sim, agent, amount, step, sim_min, source="gig")
        agent.rp_settled_day = day
        return
    # 議員報酬: 勤務日数に依らない月額(自治法 203 条)。支給日は**表に書いた 1 日**で、
    # 民間の payday_weights(職場ごとに散る慣行分布)は借りない —— 借りると安定ハッシュが
    # 月末を引いた瞬間に 10 日ランで 1 円も出ず、第112 が塞いだのと同型の穴が復活する。
    if _wage_fires(sim, cfg, last, day, int(plan["payday"])):
        amount = float(plan["monthly"])
        if amount > 0.0:
            agent.last_salary = amount
            # 出所は区の一般会計(fund_level="ward" = 行政 ON なら実際に歳出計上)。
            _pay_wage(sim, agent, amount, step, sim_min, source="stipend",
                      fund_level="ward", annual_income=plan["annual"])
    agent.rp_settled_day = day


def _phase_wage_profile(sim, step: int, sim_min: int) -> None:
    """日次境界: 在場している被用者の賃金を清算する(既定 OFF=即 return=バイト一致)。"""
    cfg = _wage_cfg(sim)
    if cfg is None:
        return
    day = sim_min // 1440
    if day == getattr(sim, "_wage_day", -1):
        return
    sim._wage_day = day
    for agent in sim.agents:                           # id 昇順(sim.agents の順)= 決定論
        if getattr(agent, "dead", False):
            continue                                   # ★B7: 死者に給料日は来ない
        role = _role_plan(sim, agent)                  # §ROLE(第114 1a)を先に見る
        if role is not None:
            _role_settle(sim, agent, role, cfg, day, step, sim_min)
            continue
        plan = wage_assign(sim, agent)
        if plan is None:
            continue
        _wage_settle(sim, agent, plan, cfg, day, step, sim_min)


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
    services_mod.self_dev_daily(sim, step, sim_min)   # T4 自助努力: skill/fitness の日次自然減衰(既定 decay=0=no-op)
    thr = float(sim.economy["money_pressure_threshold"])
    fixed_cost = float(sim.economy.get("fixed_cost_daily", 0.0))     # 固定費(既定0=OFF)
    rd_coef = factor_update.relative_deprivation_coef(sim.mags)      # 相対的剥奪係数(既定0=OFF)
    # 参照集団の所持金中央値(coef>0 のときだけ算出=OFF なら完全 no-op・決定論)。
    ref_median = _money_median(sim) if rd_coef > 0.0 else None
    bank_on = _bank_on(sim)                                          # E-W1 預金利息(既定 OFF=付与なし)
    bcfg = sim.economy["bank"]
    for agent in sim.agents:
        if getattr(agent, "dead", False):
            continue                # ★B7: 死者に日銭も利息も固定費も生活圧も発生しない
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
        # E-W1 預金利息(bank ON かつ居住者のみ): 口座残高に日次利息を付与(interest_paid)。OFF=no-op。
        if bank_on and not agent.visitor:
            itr = economy_mod.daily_interest(agent.account, bcfg)
            if itr > 0.0:
                agent.account += itr
                # IF-E2(既定 OFF=従来どおり Bank.capital を減らさない=貨幣創出のまま):
                # 利息の支払側を対称化する(§3-e の 1 行)。Bank は「最後の貸手=capital が
                # 負でも貸せる」と既に宣言しているので理論的に整合する。
                if sfc_mod.enabled(sim):
                    _bank(sim).capital -= itr
                # IF-F W3: 利息を付けたのは銀行(agent_id は受け取った預金者 = 患者)。
                devices_mod.log_device(
                    sim, Event(step=step, sim_min=sim_min, agent_id=agent.id,
                               kind="interest_paid", x=agent.x, y=agent.y,
                               payload={"amount": round(itr, 2),
                                        "balance": round(agent.account, 1)}),
                    devices_mod.DEV_BANK_MAIN)
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
        if getattr(agent, "dead", False):
            continue                               # ★B7: 死者は公務員名簿から外れる
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
        if getattr(agent, "dead", False):
            continue                               # ★B7: 死者は給付の対象ではない
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
    より前に行い、給付後の残高で逼迫(money_pressure)が判定される=セーフティネットの間接経路。

    IF-F W2: 会計の締め・ペイロール・給付は**行政という装置**の仕事なので、causality ON の
    ときだけ装置スコープ gov:main を開く(既定 OFF は NO_SCOPE=従来と同じ 1 本道)。"""
    if not _government_on(sim):
        return
    gov = sim.government
    day = sim_min // 1440
    records = gov.daily(day)                   # 空=同日(何もしない)
    if not records:
        return
    with devices_mod.cause_scope(sim, devices_mod.DEV_GOV_MAIN):
        for rec in records:                    # 3主体の当日会計を締めて出力(Σ歳入−Σ歳出=残高変化)
            sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=-1,
                                 kind="public_budget", x=0.0, y=0.0,
                                 payload={"level": rec["level"],
                                          "revenue": round(rec["revenue"], 1),
                                          "expense": round(rec["expense"], 1),
                                          "balance": round(rec["balance"], 1)}))
        _gov_payroll(sim, gov, step, sim_min)  # 当日分の歳出は次の日境界で締め・出力
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
    if total < deposit:                                # 敷金不足=現金不足点: 融資で補填(bank ON 時)
        total += _maybe_loan(sim, agent, deposit - total, step, sim_min)
        if total < deposit:                            # 融資後も払えない=引っ越せない(客観条件)
            # IF-B(第94): 敷金不足。既定 silent は完全 no-op(従来と完全同一)。
            reject_mod.notify(sim, agent, "move_home", "no_money", step, sim_min)
            return
    rng = sim.hub.stream("move_home", agent.id, step)  # 新 stream(既存 draw 順に不干渉)
    bld = freedom_p2_mod.pick_home(sim, agent, action.get("area"), rng)
    if bld is None:                                    # 空き住戸なし
        # IF-B(第94): 空き住戸なし。★draw は上で済ませてあるので通知は draw 順に不干渉。
        reject_mod.notify(sim, agent, "move_home", "no_room", step, sim_min)
        return
    old = agent.home_building
    dep = deposit                                      # 敷金の控除(口座→現金の順)
    before_total = agent.money + float(getattr(agent, "account", 0.0) or 0.0)
    if accounts_on:
        take = min(agent.account, dep)
        agent.account -= take
        dep -= take
    agent.money = max(0.0, agent.money - dep)
    mh_payee = None
    if sfc_mod.enabled(sim):                           # IF-E2: 敷金の受け手=不在家主(RoW)
        paid = before_total - (agent.money + float(getattr(agent, "account", 0.0) or 0.0))
        if paid > 0.0:
            mh_payee = sfc_mod.row_out(sim, "deposit_landlord", paid)
    levels = int(bld.get("levels", 1) or 1)
    agent.home_building = bld["id"]
    agent.home_node = bld["entrance"]
    agent.home_floor = 1 + int(rng.integers(max(1, levels)))
    if agent.home_floor > levels:
        agent.home_floor = max(1, levels)
    mh_payload = {"from": old, "to": bld["id"], "deposit": round(deposit, 1)}
    if mh_payee is not None:                           # IF-E2(既定 OFF=キーなし)
        mh_payload["payee"] = mh_payee
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                         kind="move_home", x=agent.x, y=agent.y,
                         payload=mh_payload))
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
    if target is None:
        # IF-B(第94): 相手が近傍にいない(監査 §2-C の「相手不在」)。silent は完全 no-op。
        # ★「既に交際中」は本バッチの対象外(監査が挙げたのは相手不在の 1 件だけ。
        #   理由コードを増やす前に対象一覧を監査と 1 対 1 に保つ)= 従来どおり無音。
        reject_mod.notify(sim, agent, "propose_partnership", "absent",
                          step, sim_min)
        return
    if getattr(target, "partner_id", None) is not None:
        return                                         # 既に交際中(対象外=従来どおり無音)
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
    # 建物内では route を張らない(exit_building が node を入口へ張り替えるため、
    # 屋内から張った route は非隣接エッジになり _phase_move が KeyError で落ちる。
    # truth_ledger._route_to / tools._free_to_move と同一の guard)
    if where and not agent.sleeping and agent.activity != "working" and not agent.building:
        dest = _free_dest(sim, str(where))
    if dest and dest != agent.node:
        path, used_mode = sim.router.route(agent.node, dest, "walk")
        if len(path) >= 2:
            agent.route = path[1:]
            agent.edge_offset = 0.0
            agent.dest = dest
            agent.trip_mode = used_mode
            agent._pending_stay = max(1, min(sim.clock.dur_steps(24),
                                            round(minutes / sim.clock.step_minutes)))
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
            # ★店主が街に居ない屋台は摘発しない(本人不在で罰金は取れない)。``agent_by_id`` は
            #   退場者も返すので、以前は幽霊の財布から罰金を引き(hydrate で消える)区の歳入だけが
            #   増えていた = **現金が湧く**。在場述語で解決する。
            owner = sim.present_agent(v["owner"])
            if owner is None or owner.id == officer.id:
                continue
            penalty = min(owner.money, fine)           # 罰金(所持金の範囲で)
            owner.money = max(0.0, owner.money - fine)
            if gov_on and penalty > 0:
                sim.government.collect("ward", penalty)
            factor_update.on_enforcement(owner, griev, step=step, sim_min=sim_min,
                                         logger=sim.logger)
            x, y = sim.city.node_xy(node)
            ef_payload = {"rule_id": None, "officer": officer.id,
                          "target": owner.id, "penalty": round(penalty, 1),
                          "venture": v["name"]}
            # IF-E2(既定 OFF=キーなし): 行政 OFF の世界では徴収主体が街に居ない → RoW。
            if sfc_mod.enabled(sim) and penalty > 0:
                ef_payload["payee"] = ("government" if gov_on else
                                       sfc_mod.row_out(sim, "fine_no_authority", penalty))
            sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=officer.id,
                                 kind="enforcement", x=float(x), y=float(y),
                                 payload=ef_payload))
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
        ef2_payload = {"rule_id": rule_id, "officer": officer.id,
                       "target": a.id, "penalty": round(penalty, 1)}
        if sfc_mod.enabled(sim) and penalty > 0:           # IF-E2(既定 OFF=キーなし)
            ef2_payload["payee"] = ("government" if gov_on else
                                    sfc_mod.row_out(sim, "fine_no_authority", penalty))
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=officer.id,
                             kind="enforcement", x=a.x, y=a.y, payload=ef2_payload))
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
            # §1.2 B6(第94バッチ OBS-U2): step → 分の 10 直書きはプロンプト入力なので、
            # Δt=1 だと本人が実際の 10 倍の拘束時間を信じる(認知汚染)。Δt=10 では
            # step_minutes==10 = 文字列が 1 バイトも変わらない。
            a.remember(f"取り締まりで{det * sim.clock.step_minutes}分間その場に留め置かれた")


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
        sim.today_weather_line = weather.weather_line(w, wea)
        # payload は synthetic では従来どおり {cond, temp_hi} の2キーのまま(L1 バイト一致)。
        # generated / table のときだけ観測用の列(最低気温・降水・湿度・暑さ指数)を足す。
        payload = weather.event_payload(w, wea)
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
    # 第65バッチ: 会話由来 magnitude の当日タリーを日境界で初期化(OFF は None=状態も作らない
    # =バイト一致)。観測専用=closeness/イベント/乱数には触れない。
    relations_endo_mod.quality_day_state(sim, day)
    # 第75バッチ(ダンバー認知枠): 減衰**後**の closeness で活性関係の上限を課し(超過分は最弱
    # から休眠=relation_dormant)、L2 スカラーを焼き直す。全ペア走査はここ(1日1回)だけ=毎 step
    # の走査は無い。既定 OFF は即 return(状態も作らない=バイト一致)。
    dunbar_mod.day_phase(sim, step, sim_min)


# ---------------------------------------------------------------- 負の評判の内生伝播(第61バッチ c)
def _phase_gossip(sim, step: int, sim_min: int) -> None:
    """毎 step: 悪評の種スキャン(毎 step)+ 種/伝播(complex contagion)/忘却(日境界)。

    gossip 無効なら完全 no-op(gossip_seed/spread/fade 0 件・stream "gossip" も引かない=バイト一致)。
    ロジックは gossip.py(src/society 直下=CHECKED_DIRS 外=no-fingerprint)に閉じる。infoenv(誤情報)
    確定後・collect(L2)前に置く=当日の負イベントを同 step でスキャンし L2 スカラーへ即反映する。
    R1: generate() を 1 本も足さず、k・内面状態(構成概念)を読まず、既存イベント列(負イベント種は conf
    マップ)・会話接触の relations 台帳(k 非依存の観測量)・専用 stream "gossip"・config のみ参照する
    (呼数は k 非依存)。制裁(相手選択後退/joint 誘い低下)は対面 co-location を変えうる(career/joint と
    同型で許容)→ 呼数不変は compute_matched 下の k 不変性で担保する。既存 _phase_* の並びは壊さない。"""
    gossip_mod.phase(sim, step, sim_min)


# ---------------------------------------------------------------- 世帯・恋愛(後続波 H2)
def _phase_beliefs(sim, step: int, sim_min: int) -> None:
    """真偽台帳ミニマル(第73バッチ Part B)。既定 OFF=完全 no-op。

    beliefs 無効なら 1 バイトも変わらない(fact/belief/イベント 0 件・新フィールド無し・
    乱数ゼロ)。ロジックは truth_ledger.py(src/society 直下=CHECKED_DIRS 外=no-fingerprint
    契約に触れない層)に閉じる。engine は**台帳の中身を読まない**(呼ぶだけ)。
    位置と発話が確定した _apply の後・collect(L2)の前に置く=当日の世界イベントと発話を
    同 step のうちに取り込み、L2 スカラーへ即反映する(gossip と同じ配置の流儀)。"""
    truth_ledger_mod.phase(sim, step, sim_min)


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


# ---------------------------------------------------------------- 共同行動エンジン(第44バッチ S-R3)
def _phase_joint(sim, step: int, sim_min: int) -> None:
    """日次境界: 当日の共同行動を編成する(誘い→承諾→ランデブー POI。既定 OFF=no-op。第44バッチ S-R3)。

    joint 無効なら完全 no-op(joint_activity 0 件・"joint" stream も引かない=バイト一致)。
    R1: plan_day は generate() を1本も足さず、k・内面状態(構成概念)を読まず、名簿・config・
    relations の closeness(観測イベント由来=k 非依存の観測量)・housemates・物理位置のみ参照する
    (呼数は k 非依存)。合流(同一 POI 収束)は物理位置=対面 co-location を変えうる(career/health/
    household と同型で許容)→ 呼数不変は compute_matched 下の k 不変性で担保する。編成は日境界のみ・
    実際の同席観測(joint_activity)は毎 step の joint_mod.observe が担う。"""
    joint_mod.plan_day(sim, step, sim_min)


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
                    sv_payload = {"amount": round(sev, 1),
                                  "balance": round(agent.money, 1),
                                  "to": "cash", "source": "severance"}
                    # IF-E2(既定 OFF=キーなし): 退職金の出所は**解雇した元 org** の預金。
                    sv_payer = sfc_mod.on_wage(sim, agent, sev, "severance",
                                               from_org, step, sim_min)
                    if sv_payer is not None:
                        sv_payload["payer"] = sv_payer
                    sim.logger.log(Event(step=step, sim_min=sim_min,
                                         agent_id=agent.id, kind="wage",
                                         x=agent.x, y=agent.y, payload=sv_payload))
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


# ---------------------------------------------------------------- 内部可動性(第60バッチ b。既定 OFF)
def _housing_on(sim) -> bool:
    """転居 or 同棲(内部可動性)が有効か。既定 OFF=新経路を一切通さない(バイト一致)。"""
    hcfg = getattr(sim, "housingcfg", None)
    reloc = bool(hcfg and hcfg["enabled"])
    hh = getattr(sim, "householdcfg", None)
    cohabit = bool(hh and hh.get("enabled") and (hh.get("cohabit") or {}).get("enabled"))
    return reloc or cohabit


def _phase_housing(sim, step: int, sim_min: int) -> None:
    """日次境界: 同棲(bond→N日→move_in)→ 転居(職場/家賃逼迫→relocate)を内生的に処理する。

    転居も同棲も「エージェント側の状態・選択に由来する内生変化」=火種介入とは別物(devlog Entry 50)。
    決定論: 確率・行き先は新 stream "housing"(agent, step)から引く=既存 draw 順に挿入しない。
    R1: mobility 機構は generate() を1回も足さず、k・内面状態(構成概念)を発火判断に食わせず、暦・
    config・物理位置(home/work)・relations の closeness(k 非依存の観測量)・新 stream のみ参照する
    (呼数は k 非依存)。同棲を先に処理して(その日の世帯併合を確定)から転居を評価する(id 昇順・決定論)。
    既定 OFF は即 return(relocate/move_in を1件も出さず housing stream も引かない=乱数消費不変)。
    _housing_day は checkpoint.py 中央管理(mid-day checkpoint でも resume==straight を保つ=B4 前例)。"""
    if not _housing_on(sim):
        return
    day = sim_min // 1440
    if day == getattr(sim, "_housing_day", -1):
        return
    sim._housing_day = day
    mobility_mod.cohabit_day(sim, step, sim_min)       # ② bond→同棲(世帯併合。household.cohabit)
    mobility_mod.relocate_day(sim, step, sim_min)      # ① 転居(職場/家賃逼迫。housing.relocation)


def _phase_population(sim, step: int, sim_min: int) -> None:
    """日次境界: 存在の内生化 POP(恒久転出 → 転入=定着昇格 → 出生)。既定 OFF=即 return。

    「名簿そのものが誰を含むか」の変化を、日次の抽選割当ではなく**個体状態の関数**として
    起こす層(``src/society/population.py``)。転出=Wolpert 1965 の閾値型(居住ストレスの
    蓄積が個体固定の閾値を跨ぐ)/ 転入=案A「L4 来街者の定着昇格」(来街履歴 × 縁 ×
    空き住戸 × 求人という**域内状態への応答**)/ 出生=パートナー継続日数と年齢帯からの決定論。

    ★**_phase_housing の後**に置く: 同棲・転居でその日の世帯と住居が確定してから、
      「この街に住み続けるか」「住み始めるか」を評価する(空き住戸の判定が 1 日ずれない)。
    ★日境界の進行は ``sim._pop_state["day"]`` が中央管理する(mid-day checkpoint でも
      resume==straight。checkpoint.py が runtime["pop_state"] で丸ごと運ぶ)。
    R1: generate() を 1 本も足さず、乱数 stream を 1 本も引かず(全て安定ハッシュ)、
    新しい L1 kind を 1 つも作らない(既存 life_event / job_change / relation_dormant)。
    """
    population_mod.phase(sim, step, sim_min)


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


def _medical_spend(sim, step: int, sim_min: int):
    """医療の支払い口(既存 ``_spend`` の薄い包み)。**受診先ノードを任意で受ける**。

    H1(受診)も H2(入院)もこの 1 本を使う。第 4 引数 ``payee_node`` は既定 None =
    従来と 1 バイトも変わらない(受け手は今までどおり支払者の居場所から解決される)。"""
    def _pay(agent, amount, cat, payee_node=None):
        _spend(sim, agent, amount, cat, step, sim_min, payee_node=payee_node)
    return _pay


def _phase_health_severity(sim, step: int, sim_min: int) -> None:
    """毎step: 予約された発症の発火・回復・転帰(死)。**乱数ゼロの純粋な状態機械**。

    ★``city_ops.phase`` の**直前**に置く: S3/S4 への遷移が刻む ``sev_collapse_step`` を、
      同じ step のうちに救急の行為連鎖(倒れる→通報→出動)が読むため。
    health OFF / severity OFF なら完全 no-op(新経路を 1 行も通さない=バイト一致)。"""
    if not _health_on(sim) or not health_mod.severity_on(sim):
        return
    health_mod.severity_tick(sim, step, sim_min, _medical_spend(sim, step, sim_min))


def _phase_health(sim, step: int, sim_min: int) -> None:
    """日次境界: 病気の発症/回復・受診(新 stream "health")+ メンタル(慢性高 grievance→引きこもり)。

    決定論: 発症/回復/受診の確率は**新 stream "health"**(agent, step)から引く=既存 draw 順に挿入
    しない(ゴールデンと決定論の保護)。R1: health 機構は generate() を1回も足さず、発火判断に k・内面
    状態(構成概念)を食わせず、暦・config・新 stream・物理位置・grievance(k 非依存の state)のみ参照する。
    病気=欠勤(work_window 無効)+在宅(routine が自宅へ寄せる)で物理位置が変わり対面 co-location が
    変化しうる(career G5 / crowd G4 と同型)→ 呼数不変は compute_matched 下の k 不変性で担保する。
    受診は既存 spend(cat="medical")。メンタルの grievance 参照は health.py(CHECKED_DIRS 外)に閉じる。

    health OFF なら完全 no-op(illness/medical_visit/health_update 0 件・health stream も引かない=
    乱数消費不変=ゴールデンを守る)。来街者は街の外に生活基盤=対象外(economy/career と同型)。

    ★世代交代(レーン H1・health.severity.enabled。既定 false=下の現行経路をそのまま通る):
    ON のときは単一の真偽値 sick を **S0〜S4 の重症度状態機械**へ置き換え、5 つの発症チャネル
    (急病/熱中症/急性アルコール/外傷/心停止)を独立ハザード × frailty で引く(新 stream
    "health_onset")。★来街者も対象にする: 熱中症・急性アルコール・転倒・路上の心停止は
    **街頭人口に起きる出来事**で、居住者に限ると来街者主体の街の実態を大きく取りこぼすため
    (この差は severity ON のときだけ生じる = 既定挙動は 1 バイトも変わらない)。"""
    if not _health_on(sim):
        return
    day = sim_min // 1440
    if day == getattr(sim, "_health_day", -1):
        return
    sim._health_day = day
    cfg = sim.healthcfg
    medical_cost = float(cfg["medical_cost"])
    if health_mod.severity_on(sim):                    # H1 世代交代(既定 OFF=この枝を通らない)
        health_mod.severity_day(sim, step, sim_min,
                                _medical_spend(sim, step, sim_min))
        for agent in sim.agents:
            health_mod.update_mental(agent, cfg, step, sim_min, sim.logger)
        return
    for agent in sim.agents:                           # id 昇順=決定論
        if agent.visitor:
            continue
        rng = sim.hub.stream("health", agent.id, step)  # 新 stream(既存 draw 順に影響しない)
        ill = health_mod.roll_illness(agent, cfg, rng, step,
                                      sim.clock.steps_per_day)
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
    (ここでは開閉遷移のログのみ)。commerce OFF なら完全 no-op(shop_state 0 件=バイト一致)。

    IF-F W2: 開閉遷移は**営業時間という装置**の仕事なので、causality ON のときだけ
    装置スコープ commerce:hours を開く(OFF は NO_SCOPE=割り当てゼロ・logger 不触)。"""
    with devices_mod.cause_scope(sim, devices_mod.DEV_COMMERCE_HOURS):
        commerce_mod.tick_shop_state(sim, step, sim_min)


def _goods_on(sim) -> bool:
    """物流の実体化(店舗在庫・日次補充・商品実体)が有効か。既定 OFF=新経路を一切通さない(バイト一致)。"""
    cfg = getattr(sim, "goodscfg", None)
    return bool(cfg and cfg["enabled"])


def _phase_goods(sim, step: int, sim_min: int) -> None:
    """毎step: 補充トリップの到着処理 + 日次 (s,S) レビュー(既定OFF=no-op。物流①)。

    在庫は購入点(_charge_meal / 建物内消費)で decrement される(ここでは補充=物流のみ)。到着は毎step、
    レビューは日次(restock_hour 以降の最初の step)。封鎖(災害の運休/shock_closure)で補充失敗→欠品波及。
    決定論・乱数ゼロ=新 stream を引かない。既存の scenario/disaster が確定した後に呼ぶ(その日の封鎖を読む)。
    goods OFF なら完全 no-op(delivery_trip/restock/stock_low 0 件=乱数消費不変=ゴールデンを守る)。

    IF-F W2: 補充は**物流という装置**の仕事なので、causality ON のときだけ装置スコープ
    logistics:goods を開く(stock_out は購入 seam 側で出るのでこの窓には入らない)。"""
    with devices_mod.cause_scope(sim, devices_mod.DEV_LOGISTICS_GOODS):
        goods_mod.tick(sim, step, sim_min)


def _phase_goods_staff(sim, step: int, sim_min: int) -> None:
    """毎step: 棚出し(店員の行動)+ 自店の発注(店主の行動)(既定OFF=no-op。INV-B)。

    **位置確定後**に呼ぶ(street_life / city_ops と同じ位置): 「自分の職場に居るか」を
    この step の確定した co-location で判定するため。世界の状態(棚が減った)は当人の
    知覚トリガであり、変化(棚が埋まる・発注される)は当人の行動の結果として起きる。
    担い手が 1 人も割り当てられていない POI だけ、宣言つきでエンジンが代替再現する。
    LLM 呼ゼロ・乱数ゼロ・k 非依存。2層 OFF なら完全 no-op(shelf_restock/stock_order 0 件)。"""
    goods_mod.staff_phase(sim, step, sim_min)


def _phase_delivery(sim, step: int, sim_min: int) -> None:
    """毎step: 宅配④の物理配車(dispatch)+ 到着処理(受給+課金+gig収入)(既定OFF=no-op。スライス④)。

    注文生成は routine の食事分岐(delivery.maybe_order)に相乗り=ここでは配車と到着のみ。配車は _phase_move の
    前に済ませ(配達員をその step のうちに動かす)、到着(eta)で注文者へ課金(_spend)+ 配達員へ gig 収入
    (_pay_wage source="gig")=会計は既存経路に閉じる(engine 側で注入)。決定論・乱数ゼロ=新 stream を引かない。
    delivery OFF なら完全 no-op(order/deliver 0 件・注文台帳も配達員フラグも生えない=ゴールデンを守る)。"""
    delivery_mod.tick(sim, step, sim_min, _spend, _pay_wage)


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


# ---------------------------------------------------------------- 生活の偶発イベント層(第54バッチ)
def _phase_chance(sim, step: int, sim_min: int) -> None:
    """日次境界: 生活の偶発事(臨時収入/財布紛失=money± / 偶然の出会い=closeness+)。既定 OFF=no-op。

    純観察=不確実性許容モードの柱。日境界に専用 stream "chance"((agent, day) 個体別キー=編成順非依存)で
    1人1日 daily_rate の確率で偶発事に遭い、重み付きカタログから1件を適用する。効果は money/relations/記憶
    のみ=ドライブ/発火に流さない(既存の間接経路で自然に波及)。R1: chance 機構は generate() を1本も足さず、
    発火判断に k・内面状態(構成概念)を食わせず、物理量(現在地・所持金・関係台帳の既知相手)・config・新
    stream のみ参照する。効果は物理位置・対面 co-location を変えうる(career G5 / 健康 H1 と同型)→ 呼数不変は
    compute_matched 下の k 不変性で担保する。偶発ロジックは chance.py(CHECKED_DIRS 外)に閉じる(no-fingerprint)。

    chance OFF なら完全 no-op(chance_event 0 件・"chance" stream も引かない=乱数消費不変=ゴールデンを守る)。"""
    chance_mod.tick_day(sim, step, sim_min)


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
    agent.sleep_until = step + _steps_until_tod(sim_min, int(cfg["checkout_hour"]) * 60,
                                                sim.clock.step_minutes)
    reflect_timing_mod.arm(sim, agent, step)       # 就寝直後の内省(自宅就寝・帰路退出と同格の k 処置。
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


# ---------------------------------------------------------------- 医療の受け皿(H2)
def _phase_medical(sim, step: int, sim_min: int) -> None:
    """毎step: 救急搬送の到着(=入院)と退院(既定 OFF=no-op。H2 医療の受け皿)。

    ★``_phase_lodging`` と**同じ位置**(_phase_wake_and_returns の直前)に置く: 退院で
      sleeping を落とすので後段の起床フェーズが二重に起こさない(宿泊のチェックアウトと同型)。
    入院費の支払いは既存 ``_spend`` 経路(cat="medical")で、受け手には**病院ノード**を渡す。
    medical OFF なら完全 no-op(新 4 種の L1 0 件・state なし・乱数消費不変=ゴールデンを守る)。"""
    medical_mod.phase(sim, step, sim_min, _medical_spend(sim, step, sim_min))


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


# ---------------------------------------------------------------- 日次ローテーション/presence(W2 P3)
def _phase_pool_rotation(sim, step: int, sim_min: int) -> None:
    """日境界で presence を引き直し、退場者をドーマント化・入場者を実体化する(pool ON 時のみ)。

    - presence は stream("presence", pid, day) の純関数(k/trait 非依存・resume 不変)。
      cap の充足規則は pool.tier_quota.enabled で切替(既定 OFF=層優先 break=現行と完全一致。
      ON=層別クォータ=DP-U3 案A。どちらも乱数の引き方は同じ=追加乱数ゼロ)。
    - 在場の内生化(PRES-A1/A2。conf pool.presence。既定=現行と完全一致):
      habit ON で stochastic 層が個体固定の習慣カレンダー(+目的×曜日×天候)へ、
      mode=emergent で cap を一切見ない(資格者=在場者)。天候は **generated 限定**で
      `weather.peek_bad_day`(副作用ゼロ)から覗く — 本フェーズは run_step の先頭で走り
      `_phase_calendar_weather`(当日の天気確定)は**後**なので、`sim.today_weather` を
      読むと前日の天気を掴むオフバイワンになる。
    - 退場者(present→absent): dehydrate してスリム状態を DormantStore へ退避し sim.agents から除去。
    - 入場者(absent→present): P5 record から build_pool_agent で実体化。退避済み状態があれば hydrate
      (再来街=同一実体の記憶・信念・所持金・関係が続く)。agent.id はペルソナ id で安定。
    - S6a の N 比例 cap を当日の在場数で更新。日境界で presence_change を1件記録(agent_id=-1)。

    pool 無効(sim._pool is None)なら即 return=既定挙動バイト一致(新経路を1本も通さない)。"""
    pool = getattr(sim, "_pool", None)
    if pool is None:
        return
    day = sim_min // 1440
    if day == getattr(sim, "_pool_day", 0):        # 同一日の再入場を防ぐ(日境界でのみ実行)
        return
    sim._pool_day = day
    weekday = sim._pool_weekday(day)
    pres = getattr(sim, "_pool_presence", None) or presence_mod.build_presence_cfg(None)
    new_ids = set(presence_mod.present_for_day(
        pool.presence_records(), day, sim._pool_present_cap, sim.hub, weekday,
        getattr(sim, "_pool_tier_quota", False),
        habit=pres["habit"], emergent=pres["emergent"],
        rain=sim._pool_rain(day)))
    # 存在の内生化 POP(既定 OFF=台帳が無い=同一オブジェクトがそのまま返る=バイト一致)。
    # ★presence 純関数の**外側**で当てる: あちらの入力は「暦 + 名簿のペルソナ属性」だけ、
    #   という契約(k 非依存・trait 非依存・resume 不変)を保つため。転出(永久失格)と
    #   定着昇格(resident 資格の獲得)は名簿の属性ではなく**ラン内に起きた出来事**であり、
    #   台帳は checkpoint が中央管理する(= resume でも同じ集合になる)。
    new_ids = population_mod.apply_presence(sim, new_ids)
    cur = {getattr(a, "pool_pid", None): a for a in sim.agents}
    cur.pop(None, None)
    old_ids = set(cur)
    exits = old_ids - new_ids
    enters = new_ids - old_ids
    departed_now = []                               # ★A1: この日の退場者(下で軽量参照へ)
    for pid in sorted(exits):                       # 退場者 → ドーマント化(コストゼロ・記憶保持)
        departed_now.append(cur[pid])
        sim._dormant.save(pid, pool_mod.dehydrate(cur[pid]))
    kept = [a for a in sim.agents if getattr(a, "pool_pid", None) in new_ids]
    entered = []
    for pid in sorted(enters):                      # 入場者 → 実体化(+ 退避済みなら hydrate)
        agent = sim.build_pool_agent(pid, pool.get(pid))
        saved = sim._dormant.pop(pid)
        if saved is not None:
            pool_mod.hydrate(agent, saved)
        else:
            # ★レーン R1 A2(**意図的な挙動変更**): リッチな退避状態が LRU 上限
            #   (pool.dormant_cap)で捨てられていても、金銭・債権の恒久台帳(vital)は
            #   容量に依らず残る。ここでオーバレイして所持金・口座・家賃債権・未清算の
            #   勤務実績・人口会計の**連続性**を回復する。
            #   旧挙動 = 何もしない = build_agent の**初期所持金で新規鋳造** =
            #   退場時の残高が真に消え、同時に無から現金が湧いていた(どの保存チャネルにも
            #   痕跡が残らない破れ)。vital が無い個体(= 一度も退場していない)は None。
            vital = sim._dormant.pop_vital(pid)
            if vital:
                pool_mod.overlay_vital(agent, vital)
        kept.append(agent)
        entered.append(agent)
    sim.agents = kept
    sim.invalidate_present_index()   # 在場索引を作り直す(enters==exits でも取りこぼさない)
    # agent_by_id は「これまで実体化した全個体」の id→参照(退場者を消さない)。造語の作者名・
    # DM 送信者名・関係台帳など**過去の参照**が退場後も解決できるようにする(present 個体は
    # hydrate 済みの最新実体で上書き)。tick(計算)対象は sim.agents(present)だけ=コストは在場分。
    # ★レーン R1 A1(RAM の本丸): 退場者の参照は**軽量参照へ差し替える**。過去の参照の解決に
    #   要るのは名前などの数十バイトなのに、これまでは統合エピソード 120 + 未統合 30 +
    #   関係台帳 + 信念 + persona 文を抱えたフル Agent を**累計ぶん**掴み続けていた
    #   (本選 = 累計 45.9 万体)。運ぶ欄/落とす欄は agents/ref.py の表が唯一の源。
    # ★差し替えは**入場者の実体化が全て終わった後**に置く: 実体化中は sim.agents がまだ
    #   前日の名簿なので present 述語が退場者を「在場」と答える窓があり(_link_colocated の
    #   顔なじみ張りがそこを通る)、その窓の中では従来どおりフル Agent を見せる = 差し替えが
    #   その日の入場処理を 1 バイトも動かさない。
    for gone in departed_now:
        sim.agent_by_id[gone.id] = AgentRef(gone)
    for a in kept:
        sim.agent_by_id[a.id] = a
    # 片側パートナーの清算(レーン丙 5。household.pool_bind 配下=既定 OFF は 1 行も通らない)。
    # ★**hydrate と在場索引の作り直しの後**でなければならない: 退避辞書の partner_id が戻るのは
    #   hydrate で、相手の在場は present_agent(= 索引)でしか引けないから。入場者だけを見る。
    for agent in entered:
        household_mod.reconcile_partner(sim, agent)
    sim._pool_update_budget()                       # S6a: N=当日の在場数(1行の接続)
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=-1,
                         kind="presence_change", x=0.0, y=0.0,
                         payload={"day": int(day), "n_enter": len(enters),
                                  "n_exit": len(exits), "n_present": len(sim.agents)}))
    # 来街者 party の実体化(第45バッチ S-R5。既定 OFF=no-op)。presence 純関数の後=当日 present な
    # 来街者だけをグループ化(presence の draw 順・resume 不変=test_pool_rotation を守る)。
    party_mod.form_parties(sim, step, sim_min)


# ---------------------------------------------------------------- 1 step
def _phase_workplace_bound_report(sim, step: int, sim_min: int) -> None:
    """職場束ね直し(work.bind_workplace)の day0 coverage 統計を起動時 1 件で記録する。

    step0 でのみ発火(resume は start>0 で素通り=一気 run の segment に既出=二重記録なし)。
    既定 OFF(_workbind_stat is None)は 1 件も出さない=イベント列バイト一致。世界イベント agent_id=-1。"""
    if step != 0:
        return
    stat = getattr(sim, "_workbind_stat", None)
    if not stat or getattr(sim, "_workbind_reported", False):
        return
    sim._workbind_reported = True
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=-1,
                         kind="workplace_bound", x=0.0, y=0.0, payload=dict(stat)))


def _phase_mind_report(sim, step: int, sim_min: int) -> None:
    """心のモデル固定(第88)を L1 に残す(agent_id → model_id の**個別対応**)。

    原文書 §5「モデルと人格の交絡が生じるため、agent_id と model_id の対応は必ずログに
    残す」。割当そのものは誕生時(Simulation.__init__ / build_pool_agent)に済んでいて、
    ここは**記録だけ**(乱数ゼロ・LLM ゼロ・状態を 1 バイトも書き換えない)。

    出し終えた id は `_mind_logged` に控え、pool ローテーションで途中入場した個体も
    その step に 1 件だけ出る。`_mind_logged` は checkpoint が中央管理するので、resume で
    既出ぶんを二重に記録しない(= resume==straight)。既定 OFF は 1 件も出さない。
    """
    if not mind_mod.enabled(sim):
        return
    logged = sim._mind_logged
    if len(logged) >= len(sim.agents):        # 既出だけ = 走査を丸ごと省く(常時のコストを消す)
        return
    for agent in sim.agents:
        aid = int(agent.id)
        if aid in logged:
            continue
        m = getattr(agent, "mind", None)
        if not m:
            continue
        logged.add(aid)
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=aid,
                             kind="mind_assign", x=agent.x, y=agent.y,
                             payload={"model": m["model"], "tier": m["tier"]}))


# ---------------------------------------------------------------- 屋内エンジン配線(B3)
# 屋内ミクロ状態=単一の真実。マクロ(建物内在館数)はこの集約。設計原則: ①認知/LLM step=10分は不変
# (LLM 呼数・会話ペアリングに一切影響させない=それは次バッチ B3b)②物理は遷移駆動(空間遷移が起きた
# step だけ SFM 積分)③観測(記録)と動力学を構造分離(動力学は observer バッファを読まない=下の向きは
# 動力学→ sim._indoor_*_step 一時状態→観測)。既定 OFF(sim.indoor is None)は _phase_indoor が即 return
# =新経路を一切通らず・新 stream も引かず=ゴールデン L1 バイト一致。乱数は新 named stream
# "indoor"((agent,step)キー)/"indoor_meet"((group,day)キー)のみ=既存 stream を消費しない(R1)。
def _indoor_on(sim) -> bool:
    """屋内エンジンが有効か(sim.indoor= IndoorSpace が据わっているか)。既定 OFF=None=完全 no-op。"""
    return getattr(sim, "indoor", None) is not None


# ---------------------------------------------------------------- B3b 動力学接続(既定 OFF)
def _pairing_on(sim) -> bool:
    """遭遇→対面会話ペアリングが有効か(indoor.encounter.pairing)。indoor OFF なら常に False。

    ON でも変えるのは「誰を返答相手に選ぶか」だけ(hearer 集合・会話発生・LLM 呼数は不変)。物理の
    遭遇(duration/ids)しか読まない=k・内面構成概念を読まない=compute_matched 下の k 不変性で担保。"""
    if getattr(sim, "indoor", None) is None:
        return False
    return bool(sim.indoorcfg.get("encounter", {}).get("pairing", False))


def _los_on(sim) -> bool:
    """屋内知覚の壁 LOS ゲートが有効か(indoor.los.enabled)。indoor OFF なら常に False。

    ON でも変えるのは発火プロンプトの同席リスト(近傍リスト)の中身だけ(発火判断・返答権・hearer 集合は
    ゲートしない=会話発生/LLM 呼数は不変)。座標・幾何しか読まない=no-fingerprint。"""
    if getattr(sim, "indoor", None) is None:
        return False
    return bool(sim.indoorcfg.get("los", {}).get("enabled", False))


def _los_occluder(sim):
    """indoor.los ON のとき屋内 LOS 遮蔽器を返す(遅延構築・sim にキャッシュ)。既定 None=遮蔽なし。

    None のとき hearers_of は従来と完全にバイト一致(occluder=None=既定)。occluder 実体は sim.indoor
    (壁供給)を参照する=checkpoint 非対象(resume で再構築=状態を持たない読取専用ゲート)。"""
    if not _los_on(sim):
        return None
    occ = getattr(sim, "_los_occluder_obj", None)
    if occ is None:
        los = sim.indoorcfg.get("los", {})
        occ = vision_mod.IndoorLOSOccluder(
            sim.indoor, max_dist_m=float(los.get("max_dist_m", 0.0)))
        sim._los_occluder_obj = occ
    return occ


def _indoor_pref(cfg_map: dict, activity: str) -> list:
    """活動 → 優先型リスト(conf 由来。型語はコードに書かない=no-fingerprint)。無指定は []。"""
    v = (cfg_map or {}).get(str(activity or ""))
    return list(v) if v else []


def _meeting_zone(zone_types: list, meeting_types: list, exclude=None):
    """会合を開く区画 index(meeting_types の先頭一致・exclude を除く)。無ければ None(=不開催)。

    pick_zone と違い bulk へフォールバックしない=会合区画(型)が実在する階でのみ会議が成立する。"""
    for t in (meeting_types or []):
        for i, zt in enumerate(zone_types):
            if str(zt) == str(t) and i != exclude:
                return i
    return None


def _indoor_cell_offset(building: str, floor: int, step: int,
                        step_seconds: float) -> float:
    """遷移の step 内オフセット秒 [0, step_seconds)(2層: t = step*step_seconds + offset + サブ時刻)。

    セル(建物,階)+step の安定ハッシュ=決定論・run.seed 非依存・resume 不変。1セル1積分に共通の
    オフセットを与え(遭遇の相対時刻整合を保つ)、正直な近似: 個別遷移ごとではなくセル単位の offset。

    A4(第94バッチ OBS-U2): step_seconds は **必須引数**(既定値 600.0 を廃止)。以前は
    呼び出し 2 箇所のうち片方だけが `clock.step_seconds` を渡し、もう片方は既定 600.0 を
    使っていたため、Δt=1(step 長 60 秒)では step 内オフセットが step 長を超えていた。
    必須化して再発を構造的に防ぐ。Δt=10 では両呼び出しとも 600.0 = 従来と完全同値。"""
    return indoor_flow_mod._stable_uniform(f"{building}:{int(floor)}:{int(step)}",
                                           "indoor_offset") * float(step_seconds)


def _indoor_zone_point(layout: dict, zi: int, agent_id, building: str, floor: int):
    """区画矩形内の決定論点(安定ハッシュで内側 20% を余白にジッタ)。resume 不変・run.seed 非依存。"""
    zx0, zy0, zx1, zy1 = layout["zones"][zi]
    key = f"{building}:{int(floor)}:{int(zi)}"
    u = indoor_flow_mod._stable_uniform(key, f"px:{agent_id}")
    v = indoor_flow_mod._stable_uniform(key, f"py:{agent_id}")
    mx, my = (zx1 - zx0) * 0.2, (zy1 - zy0) * 0.2
    return (zx0 + mx + u * max(0.0, zx1 - zx0 - 2 * mx),
            zy0 + my + v * max(0.0, zy1 - zy0 - 2 * my))


def _work_group_key(agent):
    """職場グループのキー(org_id 優先→職場建物→職場ノード)。無所属は None(会議対象外)。

    グループ概念は汎用に扱う(業種名・組織名の中身はキーに使うだけでコードに書かない=no-fingerprint)。"""
    org = getattr(agent, "org_id", None)
    if org:
        return f"org:{org}"
    wb = getattr(agent, "work_building", "") or ""
    if wb:
        return f"wb:{wb}"
    wn = getattr(agent, "work_node", "") or ""
    if wn:
        return f"wn:{wn}"
    return None


def _indoor_meeting_plan(sim, gk: str, day: int, prob: float, w0: int, w1: int):
    """(group, day) の会議計画 (occurs, meet_min) を決定論導出(stream "indoor_meet"・memoize)。

    キャッシュは pickle しない=resume でも同一 (gk,day) が同一 stream から同値を再導出する(独立キー
    =描画順非依存)。draw は occurs/meet_min の2本を常に引く(条件分岐で消費数を変えない=決定論)。"""
    cache = sim._indoor_meet_plan
    key = (gk, int(day))
    if key in cache:
        return cache[key]
    rng = sim.hub.stream("indoor_meet", str(gk), int(day))
    occurs = bool(float(rng.random()) < float(prob))
    span = max(1, int(w1) - int(w0))
    meet_min = int(w0) + int(float(rng.random()) * span)
    cache[key] = (occurs, meet_min)
    return cache[key]


def _indoor_meetings(sim, step: int, sim_min: int, cells: dict, icfg: dict) -> set:
    """本 step に会議へ集まるべき agent.id 集合を返す(決定論)。会議成立ごとに meeting カウンタ +1。

    同一 (building,floor) セル内で職場グループ別に min_party 人以上の勤務者が居り、(group,day) の会議
    計画が本 step の分バケットに該当し、そのセルに会合区画(型)が実在するグループを集める。"""
    mcfg = icfg["meeting"]
    min_party = int(mcfg["min_party"])
    prob = float(mcfg["prob"])
    w0, w1 = int(mcfg["window_min"][0]), int(mcfg["window_min"][1])
    mtypes = mcfg["meeting_types"]
    day = sim_min // 1440
    tod = sim_min % 1440
    targets: set = set()
    for cell_key in sorted(cells.keys()):
        building, floor = cell_key
        workers = [a for a in cells[cell_key] if a.activity == "working"]
        if len(workers) < min_party:
            continue
        groups: dict = {}
        for a in workers:
            gk = _work_group_key(a)
            if gk is not None:
                groups.setdefault(gk, []).append(a)
        for gk in sorted(groups.keys()):
            members = groups[gk]
            if len(members) < min_party:
                continue
            occurs, meet_min = _indoor_meeting_plan(sim, gk, day, prob, w0, w1)
            if not (occurs and meet_min <= tod < meet_min + sim.clock.step_minutes):
                continue
            cell = sim.indoor.get(building, int(floor))
            if _meeting_zone(cell["zone_types"], mtypes) is None:
                continue                              # 会合区画の無い階では不開催
            for a in members:
                targets.add(a.id)
            sim._indoor_n_meeting += 1                # 1職場・1日1回の開催(per-step カウンタ)
    return targets


def _phase_indoor(sim, step: int, sim_min: int) -> None:
    """屋内フェーズ: 在館者の区画割当・フロア内 markov 遷移・階間到着の実軌跡差替・会議・遭遇。

    位置(building/floor)が確定した後(_apply 後)に呼ぶ。動力学は sim._indoor_encounters_step /
    _indoor_samples_step の step 内一時状態へ積み、観測(indoor_tracks サイドカー)はそれを読むだけ。
    L1 space_move はここ(動力学)で記録(logger への書き込みは全層共通=観測バッファの読み取りではない)。"""
    if not _indoor_on(sim):
        return
    space = sim.indoor
    icfg = sim.indoorcfg
    params = sim.indoorparams
    markov_on = bool(icfg["markov"]["enabled"])
    sfm_on = bool(icfg["sfm"]["enabled"])
    meeting_on = bool(icfg["meeting"]["enabled"])
    dwell_steps = max(1, int(icfg["markov"]["dwell_steps"]))
    assign_map = icfg["markov"]["assign_types"]
    roam_map = icfg["markov"]["roam_types"]
    mtypes = icfg["meeting"]["meeting_types"]
    byst_cap = int(icfg["encounter"]["bystander_cap"])
    tracks = getattr(sim, "indoor_tracks", None)

    # step 内一時状態(動力学→ここ→観測)。毎 step リセット。カウンタも per-step(resume 安全=
    # 累積しない=分割再開でも各 step の値が state のみから再現される)。
    sim._indoor_encounters_step = []
    sim._indoor_samples_step = []
    sim._indoor_n_space_move = 0
    sim._indoor_n_encounter = 0
    sim._indoor_n_meeting = 0

    # ── 在館者を (building, floor) セルへ。屋外/睡眠/建物外はミクロ状態をクリア ──
    cells: dict = {}
    for agent in sim.agents:
        if agent.loc == "outside" or agent.sleeping or not agent.building:
            if agent.ind_zone is not None:
                agent.ind_zone = None
                agent.ind_space_type = ""
                agent.ind_x = 0.0
                agent.ind_y = 0.0
            agent._ind_ctx = None
            continue
        cells.setdefault((agent.building, int(agent.floor)), []).append(agent)

    meet_targets = _indoor_meetings(sim, step, sim_min, cells, icfg) if meeting_on else set()

    for cell_key in sorted(cells.keys()):
        building, floor = cell_key
        members = sorted(cells[cell_key], key=lambda a: a.id)
        cell = space.get(building, floor)
        layout, zone_types = cell["layout"], cell["zone_types"]
        if not layout or not zone_types:             # レイアウト無し=ミクロ状態を持てない
            for agent in members:
                agent.ind_zone = None
                agent.ind_space_type = ""
                agent._ind_ctx = cell_key
            continue
        doors = cell["doors"]
        walls = cell["walls"] if sfm_on else []
        n_zones = len(zone_types)

        movers: list = []           # SFM 積分対象
        bystanders: list = []       # 静止在館者(斥力源+接触対象)
        for agent in members:
            prev_ctx = getattr(agent, "_ind_ctx", None)
            same_cell = (prev_ctx == cell_key)
            cur_zone = (agent.ind_zone if (same_cell and agent.ind_zone is not None
                                           and 0 <= agent.ind_zone < n_zones) else None)
            old_zone = agent.ind_zone if agent.ind_zone is not None else None
            floor_move = (prev_ctx is not None and prev_ctx[0] == building
                          and prev_ctx[1] != floor and old_zone is not None)
            act = agent.activity or ""

            src = None
            dst = None
            kind = ""
            from_zone = None
            if agent.id in meet_targets:             # 会議: 会合区画へ集める(markov より優先・逃がさない)
                if cur_zone is None:
                    dst = _meeting_zone(zone_types, mtypes)        # 到着直後=会合区画へ配置(軌跡なし)
                else:
                    mz = _meeting_zone(zone_types, mtypes, exclude=cur_zone)
                    if mz is not None:
                        dst, src, from_zone, kind = mz, cur_zone, cur_zone, "meeting"
                    # else: 既に会合区画に居る → 滞在
            elif cur_zone is None:                   # 初期割当 / 階到着
                rng = sim.hub.stream("indoor", agent.id, step)
                dst = indoor_mod.pick_zone(zone_types, _indoor_pref(assign_map, act), rng)
                if floor_move and dst is not None:   # 階到着=コア→目的区画の実軌跡へ差替
                    src, kind, from_zone = "core", "floor", old_zone
            elif markov_on:                          # フロア内 markov(幾何 dwell)
                rng = sim.hub.stream("indoor", agent.id, step)
                if float(rng.random()) < 1.0 / dwell_steps:
                    roam = _indoor_pref(roam_map, act) or _indoor_pref(assign_map, act)
                    d2 = indoor_mod.pick_zone(zone_types, roam, rng, exclude=cur_zone)
                    if d2 is not None and d2 != cur_zone:
                        dst, src, from_zone, kind = d2, cur_zone, cur_zone, "roam"

            if dst is None:                          # 滞在(初回未割当も含む=位置は据え置き)
                agent._ind_ctx = cell_key
                bystanders.append(agent)
                continue

            old_x, old_y = agent.ind_x, agent.ind_y
            nx, ny = _indoor_zone_point(layout, dst, agent.id, building, floor)
            to_type = str(zone_types[dst])
            integrate = bool(from_zone is not None and sfm_on and src is not None)
            if from_zone is not None:                # 実遷移=space_move(L1)。placement は記録しない
                offset_s = _indoor_cell_offset(building, floor, step,
                                               sim.clock.step_seconds)
                from_type = str(zone_types[from_zone]) if 0 <= from_zone < n_zones else ""
                sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                                     kind="space_move", x=agent.x, y=agent.y,
                                     payload={"building": building, "floor": int(floor),
                                              "from_zone": int(from_zone), "to_zone": int(dst),
                                              "from_type": from_type, "to_type": to_type,
                                              "offset_s": round(offset_s, 1), "kind": kind}))
                sim._indoor_n_space_move += 1
                if integrate:
                    mv = {"agent_id": agent.id, "src_zone": src, "dst_zone": dst}
                    if src != "core":
                        mv["start"] = (old_x, old_y)
                    movers.append(mv)
            # 正典ミクロ状態を目的区画へ更新(位置=区画内決定論点。軌跡は tracks 専用)
            agent.ind_zone = int(dst)
            agent.ind_space_type = to_type
            agent.ind_x, agent.ind_y = float(nx), float(ny)
            agent._ind_ctx = cell_key
            if not integrate:
                bystanders.append(agent)

        # ── 遷移が起きたときだけ SFM 積分(1セル1回・movers 同士 + 静止者 frozen) ──
        if sfm_on and movers:
            byst = sorted(bystanders, key=lambda a: a.id)
            byst = byst[:byst_cap] if byst_cap > 0 else []
            by_recs = [{"agent_id": a.id, "pos": (a.ind_x, a.ind_y)} for a in byst]
            res = indoor_flow_mod.integrate_transition(layout, walls, doors, movers,
                                                       by_recs, params)
            offset_s = _indoor_cell_offset(building, floor, step,
                                           sim.clock.step_seconds)
            base_t = step * sim.clock.step_seconds + offset_s
            dstmap = {m["agent_id"]: m["dst_zone"] for m in movers}
            if tracks is not None:
                for (aid, t_sub, x, y) in res.samples:
                    sim._indoor_samples_step.append(
                        (int(aid), float(base_t + t_sub), building, int(floor),
                         float(x), float(y), int(dstmap.get(aid, -1))))
            for (a, b, knd, dur) in res.contacts:
                sim._indoor_encounters_step.append(
                    (float(base_t), int(a), int(b), str(knd), float(dur),
                     building, int(floor)))
            sim._indoor_n_encounter += len(res.contacts)

    # ── 観測(記録)= 一時状態を読むだけ(動力学は本ブロックを読まない=方向厳守) ──
    if tracks is not None:
        tracks.add_samples(sim._indoor_samples_step)
        tracks.add_contacts(sim._indoor_encounters_step)

    # ── B3b 遭遇→ペアリング動力学(indoor.encounter.pairing ON のみ)──────────────────
    # この step の遭遇を各個体の直近遭遇マップ _indoor_recent(相手id→duration_s)へ焼き込む。
    # 次 step の _apply(会話ペアリング)がこれを読む=phase 順(_phase_indoor は _apply の後)ゆえ
    # 「前 step の遭遇」を使う 1-step ラグ(正直な近似)。毎 step 全個体を上書き=持ち越しは 1 step 限定。
    # per-agent 属性ゆえ checkpoint(agents を pickle)へ自然に含まれ resume==straight がバイト一致
    # (checkpoint.py は不変更)。OFF は一切書かない=属性を生やさない=バイト一致。
    if _pairing_on(sim):
        recent: dict = {}
        for (_t_s, ia, ib, _knd, dur, _b, _f) in sim._indoor_encounters_step:
            ma = recent.setdefault(ia, {})
            ma[ib] = max(ma.get(ib, 0.0), float(dur))
            mb = recent.setdefault(ib, {})
            mb[ia] = max(mb.get(ia, 0.0), float(dur))
        for agent in sim.agents:
            agent._indoor_recent = recent.get(agent.id) or {}


def run_step(sim, step: int) -> None:
    sim.budget.reset()
    sim.freedom_stats = {"choice_points": 0, "exercised": 0}  # 自由度観測(P2)の step 境界。OFF は L2 で列不在
    sim_min = sim.clock.sim_min(step)
    # G4/G5 日次サイドカー(記憶ストリーム / 関係台帳の差分。既定 OFF=属性 None=分岐 1 回だけ)。
    # ★**step の先頭・在場ローテーションより前**に置く 3 つの理由(観測しかしない):
    #   (a) C3 すれ違いカウンタは conversation._roll_day が日境界に空へ戻すので、後ろで撮ると
    #       前日ぶんが取れない(常に 0 になる)。
    #   (b) 退場する個体の記憶と関係が、dormant へ退避される前にここで 1 度だけ残る。
    #   (c) 「day D の行」= day D-1 の終わりの状態、という 1 つの意味に揃う。
    # ★変数名に `sc` を含めない: tests/test_indoor_invariance.py の静的検査は
    #   `sc = getattr(sim, "…_sc")` で束ねた**局所変数名の部分一致**でサイドカー参照を
    #   拾うので、`_dsc.on_step` は `sc.on_step` として誤検出される(A13 roster が
    #   `_roster` と名付けているのと同じ理由)。
    for _daily_log in (getattr(sim, "memory_sc", None),
                       getattr(sim, "relations_sc", None)):
        if _daily_log is not None:
            _daily_log.on_step(sim, step, sim_min)
    _phase_pool_rotation(sim, step, sim_min)       # 日次境界: 在場ローテーション(既定OFF=no-op。W2 P3)
    _phase_workplace_bound_report(sim, step, sim_min)  # 起動時1回: 職場束ね直しの coverage 統計(既定OFF=no-op)
    _phase_mind_report(sim, step, sim_min)         # 誕生時1回/個体: 心のモデル固定の記録(既定OFF=no-op。第88)
    # 行間補間(P2 S2): この step 開始時点の logger.events 長を控える(末で増分を各個体バッファへ
    # 振り分ける)。OFF は -1 で以降の蓄積を完全にスキップ=状態も出力もバイト一致。
    _isl_idx = len(sim.logger.events) if _interstitial_on(sim) else -1
    # 観測チャンネル(第80。既定 OFF=sim.channels_sc None=-1 で以降を完全スキップ)。
    # この step で新規に記録される L1(受信発話・掲示視認)の起点を控える(_isl と同じ流儀)。
    _ch_idx = len(sim.logger.events) if _channels_on(sim, step) else -1
    # 閾値発火(第81。既定 OFF=-1 で以降を完全スキップ)。step 末に o_c(t) を採って各個体へ
    # **凍結**し、次 tick の S 判定はその 1 枚のスナップショットだけを読む(ダブルバッファ)。
    _fire_idx = len(sim.logger.events) if fire_mod.enabled(sim) else -1
    # 環境フィードバック(第84。既定 OFF=-1 で以降を完全スキップ)。改札の流入レートを
    # 「この step に駅ノードへ入った件数」で測るため、L1 の起点を控える(_isl/_ch と同じ流儀)。
    _env_idx = len(sim.logger.events) if envfb_mod.enabled(sim) else -1
    # L2 業務の実体(work.service。既定OFF=-1でこの step の接客帰属を完全スキップ=バイト一致)。
    _work_idx = len(sim.logger.events) if _work_service_on(sim) else -1
    _ensure_orgs(sim)                              # 組織台帳の遅延初期化(既定OFF=no-op)
    _ensure_wage_profile(sim)                      # WAGE: 賃金プランの初回割当(既定OFF=no-op・乱数ゼロ)。
                                                   # ★_sfc_arm の前=org 初期預金が新しい日給を読む
    _sfc_arm(sim, step, sim_min)                   # IF-E2: org 預金の期首配賦(既定OFF=no-op・乱数ゼロ)
    # 所有権レイヤー O1(登記簿の初期配賦。既定OFF=即 return=バイト一致)。**_ensure_orgs の後**に
    # 置く: 賃貸住戸の家主を「域内の不動産 org」から選ぶ(ユーザー決定 §5-1)ので、組織台帳が
    # 載ってからでないと家主が 1 社も見つからない。_sfc_arm と同じ「起動時 1 回・L1 ゼロ件・
    # 世界状態を 1 バイトも変えない」規約で、乱数は新 stream "asset_alloc" 1 本だけ。
    assets_mod.arm(sim, step, sim_min)
    sfc_mod.day_roll(sim, step, sim_min)           # IF-E2: 日次境界に前日の域外収支を締める(既定OFF=no-op)
    _phase_org_ledger_roll(sim, step, sim_min)     # 会社観測データ層 B4: 日次境界に前日の org_output/ledger を締める(既定OFF=no-op)。
                                                   # 当日の産出/接客/在席より前に置く=当日分は新しい日へ積む(オフバイワン回避)
    for agent in sim.agents:
        agent.now_step = step                      # remember() の時刻付け
    _phase_inner_life(sim, step, sim_min)          # 起動後1回: 長期目標・趣味の付与(既定OFF=no-op。H6)
    _phase_calendar_weather(sim, step, sim_min)    # 日次境界: 当日の日付・天気(既定OFF=no-op)
    _phase_annual(sim, step, sim_min)              # 日次境界: 年中行事の確定・記録(既定OFF=no-op)
    aging_mod.phase_day(sim, step, sim_min)        # 日次境界 AGE-F: 誕生日 → age+1(既定OFF=no-op)。
                                                   # ★_phase_calendar_weather の後=当日の実日付が確定してから読む
    _phase_schedule_gc(sim, step, sim_min)         # 日次境界: 過去予定の GC(既定OFF=no-op)
    _phase_relations_day(sim, step, sim_min)       # 日次境界: 関係の断絶/評判の風化(既定OFF=no-op)
    attention_mod.phase_day(sim, step, sim_min)    # 日次境界 ATT 層B: 注意スロットの減衰+消滅(既定OFF=no-op)
    _phase_household(sim, step, sim_min)           # 日次境界: パートナー形成(既定OFF=no-op。後続波 H2)
    _phase_joint(sim, step, sim_min)               # 日次境界: 共同行動の編成(既定OFF=no-op。第44バッチ S-R3)
    _phase_commerce(sim, step, sim_min)            # 毎step: 店舗の開閉遷移 shop_state(既定OFF=no-op。後続波 H3)

    sim.scenario.on_step(sim, step)                # 摂動シナリオ(baseline=即 return)
    _phase_government(sim, step, sim_min)          # 日次境界: 行政会計・公務員給与・給付(既定OFF)
    _phase_wage_profile(sim, step, sim_min)        # 日次境界: 賃金多様性 WAGE の清算(既定OFF=no-op)。
                                                   # ★_phase_daily の前=給料日の入金がその日の家賃
                                                   #   (月収相当×share)と逼迫判定に間に合う
    _phase_daily(sim, step, sim_min)               # 日次境界: 経済的逼迫の心理圧
    _phase_bank_day(sim, step, sim_min)            # 日次境界: 融資の定期返済・延滞→破産接続(既定OFF=no-op。E-W1)
    _phase_career(sim, step, sim_min)              # 日次境界: 失業/求職/転職(既定OFF=no-op。Wave G5)
    _phase_housing(sim, step, sim_min)             # 日次境界: 同棲(move_in)→転居(relocate)(既定OFF=no-op。第60バッチ b)
    _phase_population(sim, step, sim_min)          # 日次境界: 転出/転入(定着昇格)/出生(既定OFF=no-op。POP)
    _phase_health(sim, step, sim_min)              # 日次境界: 病気の発症/回復・受診・メンタル(既定OFF=no-op。H1)
    _phase_disaster(sim, step, sim_min)            # 日次境界: 災害・交通遅延/運休・インフラ障害(既定OFF=no-op。H4)
    _phase_chance(sim, step, sim_min)              # 日次境界: 生活の偶発(臨時収入/財布紛失/偶然の出会い。既定OFF=no-op。第54バッチ)
    _phase_goods(sim, step, sim_min)               # 物流: 補充トリップ到着+日次(s,S)レビュー(既定OFF=no-op。①)。
                                                   # disaster/scenario 確定後=その日の封鎖(運休/shock_closure)を読む
    _phase_delivery(sim, step, sim_min)            # 宅配④: 配達員の物理配車+到着で受給/課金/gig収入(既定OFF=no-op)。
                                                   # _phase_move の前=配車した配達員をこの step のうちに動かす
    _phase_rules(sim, step, sim_min)               # 日次境界: 制度DSL(期限失効・定期イベント)
    _phase_recursion(sim, step, sim_min)           # 日次境界: 再帰性(昨日の街の動き。既定OFF=no-op)
    _phase_assembly(sim, step, sim_min)            # 日次境界: 代表制議会の改選(既定OFF=no-op。制度深化3)
    _phase_reflect_day(sim, step, sim_min)         # 日次境界: 無意識層+衝撃ゲージのリセット(既定OFF=no-op。第12バッチ)
    _phase_freedom_day(sim, step, sim_min)         # 日次境界: 価値充足の減衰(既定OFF=no-op。第17バッチ)
    _phase_status(sim, step, sim_min)              # 日次境界: 社会的地位スコアの再計算(既定OFF=no-op。第11バッチ)
    _phase_world_events(sim, step, sim_min)
    _phase_lodging(sim, step, sim_min)             # 宿泊のチェックアウト(checkout_hour。既定OFF=no-op。Wave L)
    _phase_medical(sim, step, sim_min)             # 搬送の到着=入院 / 退院(既定OFF=no-op。H2 医療)
    _phase_wake_and_returns(sim, step, sim_min)
    # 第81(記録専用・OFF は no-op): 朝計画の対象者を _phase_planning が消費する前に控える。
    # 「内省・会話も第一級の発火源」(計画書 §6-3)を認知イベント列に載せるためだけの1行で、
    # 発火権の配布そのものは _phase_drive の単一作用点に閉じている。
    fire_mod.note_plan_due(sim, step)
    _phase_planning(sim, step, sim_min)            # 起床/帰還の直後 step に朝の一日計画
    _phase_tools(sim, step, sim_min)               # イベント開始で会場へ発つ→次の移動で反映
    # 装置層(actor model P2。既定 OFF=即 return=バイト一致): 装置の δ_int(整備窓の更新)+
    # 入場方向(この step の帰還=enter_area)の δ_ext + 要約の記録。**physics と同じ位置**に
    # 置く = この step の退出(_try_exit / 下の再試行)が同じ整備窓の残り定員を消費する。
    devices_mod.phase(sim, step, sim_min)
    # CRWD 混雑不満(既定 OFF=None=1 走査も作らない=バイト一致): この step の**開始時点**の
    # 「ノード→在館・覚醒人数」表を **1 回だけ**(O(N))作る。購入/受給の成立点はこの表を
    # O(1) で引く=購入 1 件ごとの全走査(本選規模で数十億比較)を復活させない。
    # ★位置を動かすのは下の physics / _phase_move なので、ここが「この step で全員が同じ
    #   時点を見る」唯一のスナップショットになる(日次のローテーション・転出入は確定済み)。
    # ★step ローカルな派生量なので checkpoint には載せない(位置から作り直せる)。
    sim._crowd_counts = commerce_mod.step_counts(sim)
    # P3 境界縫合(竹-4。既定 OFF=即 return=バイト一致): **_phase_move の直前**に置く。
    # この時点の (x,y) が「この step の開始時の位置」= 2 層タイムラインの下層(dt_sub)が
    # 刻むのはまさにこの step の 600 秒だから。物理が所有した個体は _phase_move が飛ばす。
    physics_mod.phase(sim, step, sim_min)
    _phase_move(sim, step, sim_min)
    # 環境フィードバック(第84。既定 OFF=空 tuple=ループ 0 回=バイト一致): 前 step までに
    # 遅延/入場規制で待たされ、駅に留まっている個体の**再試行**。_try_exit は「到着した step」
    # にしか呼ばれないので、待たせた個体にはここでしか通る口が無い。
    for _held in envfb_mod.pending_exits(sim, step):
        _try_exit(sim, _held, step, sim_min)
    # 装置層 P2(既定 OFF=空 tuple=ループ 0 回=バイト一致): 改札の待ち行列で駅に留まって
    # いる個体の再試行(上と同じ理由・同じ形。両者が同じ個体を返さないことは
    # devices.pending_exits 側で保証している=同 step に _try_exit が 2 回走らない)。
    for _held in devices_mod.pending_exits(sim, step):
        _try_exit(sim, _held, step, sim_min)
    # 計画駆動の圏外滞在(actor model P4。既定 OFF=空 tuple=ループ 0 回=バイト一致):
    # 上 2 つと**同じ形・同じ位置**の再試行口。_try_exit は「ノードに到着した step」に
    # しか呼ばれないので、(a) 計画開始時に既に縁に居た(移動が発生しない)
    # (b) 終電・遅延・改札待ちで保留された の 2 つはここでしか通る口が無い。
    # 既に退出済みの個体は exit_intent が落ちているので二重には拾わない。
    for _held in boundary_mod.pending_exits(sim, step):
        _try_exit(sim, _held, step, sim_min)
    # 位置確定後: 共同行動/夕食共食の同席観測(既定OFF=no-op。第44バッチ)。編成→収束は済み、
    # ここで実際に POI/home で2人以上同席したグループを joint_activity で1件記録する。
    joint_mod.observe(sim, step, sim_min)          # 共同行動(S-R3)
    household_mod.observe_dinner(sim, step, sim_min)  # 夕食の共食(S-R1)
    _phase_traffic(sim, step, sim_min)
    _phase_enforcement(sim, step, sim_min)         # 執行ルート: 警察官が近傍の違反者を執行(既定OFF)
    _phase_diversity(sim, step, sim_min)           # 毎step: 犯罪・迷惑行為(近傍警察官で抑止。既定OFF=no-op。後続波 H5)
    # 路上の生業 + 条例パトロール(Wave 4 III-3。既定OFF=即 return=バイト一致)。
    # **_phase_enforcement / _phase_diversity と同じ位置**(位置確定後)に置く: 持ち場に
    # 立っているか・近傍に警察官が居るかを、この step の確定した co-location で判定するため。
    # 乱数ゼロ・LLM 追加呼ゼロ・プロンプトの欄ゼロ増(路上の出来事は既存の記憶欄に 1 行入るだけ)。
    street_life_mod.phase(sim, step, sim_min)
    # 都市運営 = 見えない労働力(Wave 4 III-4。既定OFF=即 return=バイト一致)。
    # **street_life と同じ位置**(位置確定後)に置く: 収集班が担当ノードに居るか・夜間清掃員が
    # 担当ビルに入っているか・倒れた個体の近くに誰が居合わせたかを、この step の確定した
    # co-location で判定するため。救急は「倒れる → 近くの誰かが通報する → 当直が応える」の
    # 行為連鎖で、時刻表では 1 件も撃たない。乱数ゼロ・LLM 追加呼ゼロ・プロンプトの欄ゼロ増。
    _phase_health_severity(sim, step, sim_min)     # 身体: 発症の発火・回復・転帰(既定OFF=no-op。H1)
    city_ops_mod.phase(sim, step, sim_min)
    # 2層在庫の担い手(INV-B。既定OFF=即 return=バイト一致)。**city_ops と同じ位置**
    # (位置確定後)に置く: 店員が自分の店に居るかを、この step の確定した co-location で
    # 判定するため。棚出しはフラグの立った店だけ=イベント駆動(新しい全走査を足さない)。
    _phase_goods_staff(sim, step, sim_min)
    _phase_health_tick(sim, step, sim_min)         # 疲労ゲージの毎step更新(既定OFF=no-op。後続波 H1)
    street_mod.phase(sim, step, sim_min)           # 街頭広告の視認判定(既定OFF=no-op。第18バッチ)
    # 位置が確定したこの時点で空間索引を1回だけ張る。以降の _phase_drive/_decide の
    # 知覚判定を近傍9セルだけの走査にする(全対全 O(n²) の回避)。_apply は位置が
    # 動くので索引を使わず live 走査(perception.py の設計注記を参照)。
    # cell_m(`world.perception_cell_m`。既定 0 = セル寸法 = 半径 = 現行と完全同一)。
    # > 0 では実効半径の小さい有界クエリ(声の段階 5m の C2)だけが細格子へ回る。
    sim.percept_index = build_index(
        sim.agents, float(sim.cfg.world.perception_radius_m),
        cell_m=cell_m_of(sim))
    # 対人事件の収束化 H4(既定 OFF=即 return=バイト一致): 事件を「1人1step のレート抽選」
    # ではなく **共在ペアの上の条件付き確率**(Birks/Groff の RAT)にする層。**唯一の共在索引**
    # (直上で張った sim.percept_index)の上でしか発火しないので、この位置でなければならない
    # (計画書 §3「共在判定の一本化」)。共在が無ければ乱数を 1 本も引かない = 「誰もいなければ
    # 起きない」が制御フローの構造で保証される。LLM 追加呼ゼロ・プロンプトの欄ゼロ増。
    incidents_mod.phase(sim, step, sim_min)
    # 遺失物ループ H3(既定 OFF=即 return=バイト一致): 落とす→気づく→拾う→届ける/無視/着服→
    # 返還+報労金/時効。**incidents(H4)と同じ位置**(直上で張った sim.percept_index の上)に
    # 置く: 落下地点の共在をこの step の確定した co-location で判定するため(計画書 §3
    # 「共在判定の一本化」= 新しい全対全スキャンを 1 つも足さない)。落とすかどうかだけが
    # 確率事象(新 stream "lost_drop")で、拾う/届ける/着服は乱数を 1 粒も引かない決定論。
    # LLM 追加呼ゼロ・プロンプトの欄ゼロ増(当事者の記憶に定型 1 行が入るだけ)。
    lost_mod.phase(sim, step, sim_min)
    worldview_mod.phase(sim, step, sim_min)        # 主観的世界モデル: 期待の検証と更新(既定OFF=no-op。第20バッチ)
    _phase_drive(sim, step, sim_min)               # 欲求→申請→抽選→発火権
    _phase_c2(sim, step, sim_min)                  # 会話3層 C2/C3(既定OFF=no-op。P2 S3)。
                                                   # drive 押し上げは次 step の _phase_drive が拾う
    pov_mod.run_phase(sim, step, sim_min)          # エージェント視覚 v1: 顕著性POV(既定OFF=no-op)。
                                                   # 専用stream "pov_salience"・追加LLM呼ゼロ・画素決定

    active = [a for a in sim.agents
              if a.loc != "outside" and not a.sleeping]   # 外・睡眠中=計算しない
    # V3 決定モード(observer.decision_mode。既定 OFF は即 return)。**決定点の分母**は
    #   ここでしか判らない: `_decide` は必ず 1 個体 1 行動を返すのに、ルール層が決めた分は
    #   L1 にも l1b にも 1 バイトも残らない = 「LLM 被覆率」の分母が原理的に無かった。
    #   逐次経路も一括発行経路も同じ `active` を回すので、この 1 行で両経路が同じ値になる。
    decmode_mod.note_points(sim, sim_min, len(active))
    # 日中熟慮の一括発行(engine.batch_llm。既定 OFF=下の内包表記=従来と 1 命令も同じ)。
    if _deliberate_batch_on(sim):
        actions = _phase_decide_batched(
            sim, active, step, sim_min,
            workers=int(_batch_llm_cfg(sim).get("workers", 8)))
    else:
        actions = [(agent, _decide(sim, agent, step, sim_min))
                   for agent in active]
    # 同期バリア + ダブルバッファ(第81・設計 §3.3)。既定 OFF は引数をそのまま返す恒等。
    # ON では推論結果の **world への適用順を agent_id 昇順に正準化**する(到着順=推論の
    # 完了順が世界に漏れない。T1 完了順序不変性テストがこれを固定する)。
    actions = fire_mod.barrier(sim, actions)
    # 噂の誕生 IF-C 残③(第98 W2-5): **源イベントと同じ step のうちに**噂を成立させる。
    #   第95 の初版は誕生走査が step 末(rumors_mod.phase)の 1 回だけで、伝播口
    #   rumors_mod.on_talk は _apply の中(speak/dm)で呼ばれる = 誕生は必ず会話より後
    #   → その step に起きた出来事は**次 step の会話**にしか乗らなかった(1 step の遅れ)。
    #   適用の各回の**直前**に走査を挟むと、
    #     ・enforcement(_apply より前のフェーズが出す)は最初の 1 回で、
    #     ・host_event / venture_open(先に適用された個体の行為)はその直後の回で
    #   誕生し、後から適用される個体の会話に同 step で乗る。取りこぼし(最後の個体の行為・
    #   _apply より後のフェーズ)は従来どおり step 末の rumors_mod.phase が拾う。
    # ★_apply の**外**に置く理由: 誕生は「世界の出来事」であって会話の一部ではない。
    #   _apply の内側は来歴スコープ(provlink の set_prov/clear_prov)なので、そこで
    #   L1 を出すと rumor_born に**別人の発話を決めた llm_call_id** が刻まれてしまう
    #   (observer.llm_link ON のとき)。外に置けば構造的に起きない。
    # ★二重誕生は起きない: 走査は L1 の watermark 1 本(rumors._new_events)に閉じている。
    # ★OFF は bool 1 個を見るだけで 1 度も呼ばない = ゴールデン L1 バイト一致。
    _rumors_on = rumors_mod.enabled(sim)
    for agent, action in actions:
        if _rumors_on:
            rumors_mod.birth_scan(sim, step, sim_min)
        _apply(sim, agent, action, step, sim_min)
    # 屋内エンジン配線(B3): building/floor が確定した _apply 後に、在館者の区画割当・フロア内
    # markov 遷移・階間到着の実軌跡差替・会議・遭遇を回す(既定 OFF=sim.indoor None=即 return=バイト一致)。
    _phase_indoor(sim, step, sim_min)
    _phase_jitter(sim, step, sim_min)              # 路上滞在の完全静止を解消(微移動)
    _phase_crowd(sim, step, sim_min)               # 群集(大規模行事型)の集中を観測(既定OFF=no-op)
    infoenv_mod.phase(sim, step, sim_min)          # 情報環境: バイラル加重・誤情報/炎上(既定OFF=no-op。Wave G6)
    _phase_work_service(sim, step, sim_min, _work_idx)  # L2業務: 接客serve/オフィスorg_output(既定OFF=no-op)
    _phase_org_accumulate(sim, step, sim_min)      # 会社観測データ層 B4: 当日の office 在席頭数/ミクロ在席分を積む(既定OFF=no-op)
    _phase_gossip(sim, step, sim_min)              # 負の評判の内生伝播: 種スキャン+種/伝播/忘却(既定OFF=no-op。第61バッチ c)。
                                                   # infoenv(誤情報)確定後・collect(L2)前=当日の負イベントを同 step でスキャン
    _phase_beliefs(sim, step, sim_min)             # 真偽台帳: fact 抽出→直接目撃→伝聞→現場確認(既定OFF=no-op。第73バッチ B)。
                                                   # _apply 後=この step の発話・世界イベントを同 step で取り込む / collect(L2)前
    rumors_mod.phase(sim, step, sim_min)           # 情報オブジェクト IF-C: 取りこぼしの誕生+忘却掃引(既定OFF=no-op。第95バッチ)。
                                                   # 誕生の本体は _apply ループ内の birth_scan へ移した(第98 W2-5=1 step 遅れの解消)。
                                                   # ここが拾うのは「最後に適用された個体の行為」と「_apply より後のフェーズ」だけ
                                                   # (この step にはもう語り手が居ない=遅れを生まない)/ collect(L2)前
    traces_mod.phase(sim, step, sim_min)           # 痕跡 IF-D: 場所への集約(aggregation)+日境界の蒸発(evaporation)
                                                   # (既定OFF=no-op。第96バッチ)。**演算はこの 2 つだけ**=拡散なし
                                                   # (Parunak の propagation factor 0)。rumors の後=同じ L1 を別の
                                                   # watermark で走査する独立層(噂は人へ・痕跡は場所へ)
    # 事件レイヤー H5(環境側 3 族: 火災 / 交通 / 群集。既定OFF=即 return=バイト一致)。
    # **step 末**に置く理由: (a) 火災の第一発見者・事故の被害者・通報者は _apply 後の確定した
    # 位置で決めたい (b) 群集は physics.phase がこの step に測った密度(sim._phys_state)を
    # **読むだけ**で、状態を 1 ミリも動かさない (c) 交通の曝露は _phase_traffic が進めた
    # 背景交通の瞬時のエッジ占有を読む。乱数は新 stream 2 本・LLM 追加呼ゼロ・欄ゼロ増。
    incidents_env_mod.phase(sim, step, sim_min)
    # 設備 = 摩耗する装置(昇降設備の DEVS。既定OFF=即 return=バイト一致)。
    # incidents_env の**後**に置く: 利用(δ_ext)は L1 の floor_move / enter_building を
    # 自前の watermark で走査するので、この step の階移動を同 step で摩耗に反映できる。
    facilities_mod.phase(sim, step, sim_min)
    # 所有権レイヤー(既存イベントの台帳写像。既定OFF=即 return=バイト一致)。
    # **step 末・_apply の後**に置く理由: (a) 相続の引き金である death は日境界とこの step の
    # severity 両方から出る (b) 持ち家者の転居 move_home は _apply の中で起きるので、ここより
    # 前だと 1 step 遅れる (c) 動かすのは現金と台帳の行だけで位置・関係・記憶を 1 つも触らない
    # ので、この step の L3 スナップショット(下)に相続後の残高が正しく載る。
    # traces / facilities と同じ **L1 の watermark 走査**(死亡・転居があった step だけ働く)。
    assets_mod.phase(sim, step, sim_min)
    # A13 日次入場者名簿(レーン丙 2。既定 OFF=sim.roster_sc is None=分岐 1 回だけ)。
    # **step 末**に置く理由: 当日の入場者は _phase_pool_rotation(step 先頭)で実体化されるが、
    # 賃金プラン・org 配属・世帯・観光といった日境界の割当がその後の各フェーズで載るので、
    # 先頭で撮ると「まだ何も持っていない個体」が名簿に残る。観測しかしない(世界を触らない)。
    _roster = getattr(sim, "roster_sc", None)
    if _roster is not None:
        _roster.on_step(sim, step, sim_min)

    # レーンG 案B 意図収束サイドカー(既定 OFF=sim.gathering_sc is None=分岐 1 回だけ)。
    # **step 末・roster の直後**に置く理由: 朝の計画(_dayplan.blocks の解決済みノード)も
    # 予定帳への記入も、この step の _apply の中で確定するので、末尾なら 1 step の遅れが
    # 出ない。撮るのは 1 日 1 回(capture_min を最初に跨いだ step)だけで、それ以外の
    # step は int 比較 2 回で戻る。**観測しかしない**(世界も乱数も L1 も 1 バイト不変)。
    _gathering = getattr(sim, "gathering_sc", None)
    if _gathering is not None:
        _gathering.on_step(sim, step, sim_min)

    # RFX-A(既定 mode="sleep" は即 return=バイト一致): 内省的瞬間を迎えた個体の予約を
    # **この step で満期にする**。新しい発火点は 1 つも足していない —— 立てた予約は
    # 「後で立つはずだった予約の前倒し」で、`reflect_suppress_arm` がその夜の予約を
    # 1 回見送らせる(= 1 予約 1 発火)。内省ループの**直前**に置く。
    reflect_timing_mod.arm_moments(sim, step, sim_min)

    _bl = (sim.cfg.get("engine", {}) or {}).get("batch_llm", {}) or {}
    _rfx_tag = reflect_timing_mod.context_tag_on(sim)   # RFX-O(既定 OFF=payload 不変)
    _rfx_sleepy = bool(reflect_timing_mod.cfg_of(sim)["sleep_task_rewrite"])
    _p1_on = prompt_p1_mod.enabled(sim)                 # V-P1(既定 OFF=プロンプト不変)
    _p1_recall_t = prompt_p1_mod.recall_temperature(sim)  # 未設定(null)=従来の 0.7
    if ablate_mod.llm_off(sim):
        pass                                       # ablate.llm_off(第78): 夜の内省も撃たない
    elif bool(_bl.get("enabled", False)):
        _phase_reflect_batched(sim, step, sim_min,
                               workers=int(_bl.get("workers", 8)))
    else:
        for agent in _reflect_due(sim, step, sim_min):  # 内省(就寝直後 or 来街者の帰路)
            if agent.reflect_step == step:
                maybe_reflect(agent, step=step, sim_min=sim_min,
                              writeback=str(sim.cfg.k.writeback),
                              alpha=float(sim.cfg.k.degraded_alpha), llm=sim.llm,
                              place_name=_reflect_place(sim, agent),
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
                                                   .get("reflect_variety", False)),
                              interstitial=_interstitial_on(sim),
                              interstitial_digest=_isl_take(sim, agent),
                              # G7-②(既定 OFF=payload にキーを 1 つも足さない)
                              gt_extras=gt_extras_mod.enabled(sim),
                              # RFX-A / RFX-O(既定 OFF=None/False=従来定数とバイト一致)
                              moment=reflect_timing_mod.when_of(agent),
                              sleepy=_rfx_sleepy, tag=_rfx_tag,
                              # V-P1(既定 OFF=False/None=プロンプトも温度も従来と同一)
                              p1=_p1_on,
                              recall_temperature=_p1_recall_t)

    # 行間補間(P2 S2): この step で新規に記録されたイベントを各個体のバッファへ振り分ける
    # (OFF は _isl_idx=-1 で完全 no-op=バッファも作らない=バイト一致)。発火時の _isl_take が
    # バッファを空にしているので、以降はこの step 分から「前回発火以降」が積み直る。
    if _isl_idx >= 0:
        _isl_accumulate(sim, _isl_idx)

    # 環境フィードバック(第84・設計 §4.5「step 末に集約状態から一括」。既定 OFF=即 return)。
    # ★ここで**世界の位置は 1 つも動かさない**: 集約量から環境状態(遅延・入場規制・待ち行列)を
    #   進めるだけで、それを読むのは次 step 以降の作用点(改札・帰還・行き先候補)。step 途中で
    #   計算すると step 内の処理順が結果に混入して同期バリア(第81)の意味が消える。
    # ★観測チャンネル(下)より前に置く: ext.transit_delay がこの step 末の遅延を見る。
    envfb_mod.update(sim, step, sim_min, _env_idx)
    # ラッシュ時の車内(Wave 4 II-1。既定 OFF=即 return=バイト一致): (路線, 到着分)の群
    # =「同じ 1 本の列車の乗客」に**車両と区画**を与え、降車で train_ride / train_copresence を
    # 出す。★置き場所は環境フィードバックの更新と駅員・車掌フェーズの**あいだ**:
    #   (a) 降車(_phase_wake_and_returns)は step の前半に済んでいるので、この時点で
    #       「まだ圏外に居る = 車内」が確定している。
    #   (b) 下の車掌(transit_staff)が**同じ step のうちに**車内負荷を読めるように、
    #       車掌より先に車内を確定させる(読む側が 1 step 古い数字を見ない)。
    # ★会話経路は 1 本も作らない(車内は静かな同席)= generate() 呼数は ON/OFF で完全一致。
    transit_interior_mod.phase(sim, step, sim_min)
    # 駅員・車掌アクター(actor model P3a。既定 OFF=即 return=バイト一致): 持ち場への束ね
    # (初回 1 回・冪等)+ **当直の車掌のドア閉判断**(ホーム負荷 → 停車時間 → delay_min)。
    # ★envfb_mod.update の**直後**に置く: 規則1 は ON のとき本 module へ譲る(二重適用の防止)
    #   ので、譲られた側が同じ step のうちに delay_min を確定させないと、下の観測チャンネル
    #   ext.transit_delay が 1 step 古い遅延を読む。集約(_env_idx)は規則1 と同じ起点を渡す。
    transit_staff_mod.phase(sim, step, sim_min, _env_idx)

    # 観測チャンネル o_c(t)(第80。既定 OFF は上の _ch_idx=-1 でここに入らない)。
    # **読むだけ**: 世界状態から決定論計算してサイドカーへ積むだけで、L1/L2/L3・乱数・
    # LLM 呼数のいずれも触らない。L3 スナップショットの直前=同じ step 終了時の世界を見る。
    if _ch_idx >= 0:
        from ..cognition import channels as _channels_mod
        _rows = _channels_mod.observe(sim, step, sim_min, _ch_idx)
        if getattr(sim, "channels_sat", False):   # G6: 価値 4 軸の充足 sat(既定 OFF=列なし)
            from ..observer import channels as _obs_channels_mod
            _rows = _obs_channels_mod.append_sat(sim, _rows)
        sim.channels_sc.add_rows(_rows)

    # 閾値発火(第81。既定 OFF は _fire_idx=-1 で完全 no-op)。この step 末の o_c(t) を
    # 各個体へ凍結し、次 tick の S 判定の唯一の入力にする(全員が同じ state(t) を読む)。
    fire_mod.observe_end(sim, step, sim_min, _fire_idx)

    # 感度 g / 閾値倍率 θ の全軌跡(第82・設計 §2.7/§8。既定 OFF は sidecar 不在)。
    # **読むだけ**の観測層で、動力学はこのバッファを読まない(channels と同じ構造)。
    if getattr(sim, "cognition_g_sc", None) is not None and plasticity_mod.due(sim, step):
        sim.cognition_g_sc.add_rows(plasticity_mod.rows(sim, step, sim_min))

    # DPH-O ⑤(observer.starvation。既定 OFF は即 return=1 バイトも動かない): この step の
    # LLM 予算の使用量 used / cap を 1 件積む。**読むだけ**で、この値を見て動く行はシム側に
    # 1 行も無い。位置は「全ての budget.take が済んだあと・次 step 頭の reset の前」= ここ。
    starvation_mod.note_step_budget(sim, sim.budget)
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
