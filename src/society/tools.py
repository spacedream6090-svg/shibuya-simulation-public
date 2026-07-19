"""エージェントが「世界を変える」ために使える道具(affordance)。5 種。

4 軸(ユーザー枠組み 2026-07-05)を最小で覆う:
  - モノを作る(起業・技術)      → open_venture(出店・経済連動)
  - 制度を作る(法律・規格)      → propose(提案・署名)
  - 人を作る(教育)             → host_event(勉強会=複雑感染の閾値を1回で満たす)
  - 虚構を書き換える(新概念普及) → host_event / post_flyer / found_group(共通インフラ)

原則(絶対制約):
- R1: ツールの提示・可否は k.writeback と無関係(所持金・在庫など客観条件のみ)。
- R4: ツールの効果は客観カウント(参加人数・売上・署名数)。LLM 審判は無い。
- R9: factors の更新則は traits を見ない(既存 on_own_adopted 等をそのまま使う)。
- 中立性: ツールは選択肢として提示するだけ。使用を勧めない(造語と同じ観察対象)。

engine/cognition から因子名を名指ししないため、状態は本モジュールに閉じ、更新は
既存の factors/labeling/net/observer 経由で行う(このファイルは CHECKED_DIRS 外)。
"""
from __future__ import annotations

from collections import defaultdict

from . import status as status_mod
from .cognition import drive
from .factors import update as factor_update
from .lang.sentiment import valence
from .observer.schema import Event
from .rules import apply_bonus
from .world.clock import STEP_MINUTES


def build_tools_cfg(raw: dict | None) -> dict:
    raw = dict(raw or {})
    return {
        "enabled": bool(raw.get("enabled", True)),
        # 標準装備(H): true で全発火プロンプトに「所持ツール」節を中立告知(既定 false=不変)
        "equip_all": bool(raw.get("equip_all", False)),
        # 営業許可(制度深化2 第10バッチ。保健所・風営法等の参入障壁の最小形。既定 0/0=不変):
        #   permit_steps     許可待ち(申請〜開業までの step 数。待ちの間は販売なし)
        #   permit_deny_prob 却下確率(新 stream "permit"。却下=開業費は返る・出店なし)
        "permit_steps": int(raw.get("permit_steps", 0)),
        "permit_deny_prob": float(raw.get("permit_deny_prob", 0.0)),
        "venture_cost": float(raw.get("venture_cost", 30000)),
        "venture_price": float(raw.get("venture_price", 800)),
        "venture_close_days": int(raw.get("venture_close_days", 3)),
        "proposal_threshold": float(raw.get("proposal_threshold", 0.25)),
        "event_duration_steps": int(raw.get("event_duration_steps", 6)),
        "flyer_ttl_steps": int(raw.get("flyer_ttl_steps", 144)),
        "flyer_max_per_node": int(raw.get("flyer_max_per_node", 3)),
        "attend_base": float(raw.get("attend_base", 0.25)),
        "attend_relation_bonus": float(raw.get("attend_relation_bonus", 0.35)),
        "group_join_prob": float(raw.get("group_join_prob", 0.4)),
        "buy_prob": float(raw.get("buy_prob", 0.3)),
    }


# ---------------------------------------------------------------- 制度改変の3ルート(Wave G3。既定 OFF)
def build_routes_cfg(raw) -> dict:
    """institution_routes(労働争議/投票/執行)の設定を正準化。すべて既定 OFF=現行挙動と完全同一。

    dotlist / OmegaConf どちらでも受ける(config.py の作法を局所化)。OFF 時は labor/vote/enforcement
    の enabled が False になり、tools/rules/scheduler 側で完全 no-op(既存 propose/rules/government の
    OFF・現行挙動を一切変えない=ゴールデンを守る)。"""
    from omegaconf import OmegaConf
    if OmegaConf.is_config(raw):
        raw = OmegaConf.to_container(raw, resolve=True)
    raw = dict(raw or {})
    labor = dict(raw.get("labor", {}) or {})
    vote = dict(raw.get("vote", {}) or {})
    enf = dict(raw.get("enforcement", {}) or {})
    delib = dict(raw.get("deliberation", {}) or {})
    asm = dict(raw.get("assembly", {}) or {})
    asm_real = dict(asm.get("realism", {}) or {})
    return {
        "labor": {"enabled": bool(labor.get("enabled", False))},
        "vote": {
            "enabled": bool(vote.get("enabled", False)),
            # 決定論投票の yes 判定: grievance がこの閾値以上、or 提案語への opinion 整合で yes。
            "grievance_threshold": float(vote.get("grievance_threshold", 0.5)),
            "majority": float(vote.get("majority", 0.5)),   # 可決=yes 比率がこの値超(既定=過半数)
            # 供託金(制度深化 第9バッチ。公選法の供託金モデル=政治参入の所持金障壁)。既定 0=不変。
            # propose の正式受理に deposit 円の拠出が要る(払えなければ演説どまり)。可決 or
            # yes 率 >= refund_share で返還、未満は没収(government ON なら区の歳入)。
            "deposit": float(vote.get("deposit", 0.0)),
            "refund_share": float(vote.get("refund_share", 0.1)),
        },
        "enforcement": {
            "enabled": bool(enf.get("enabled", False)),
            "fine": float(enf.get("fine", 1000.0)),         # 執行時の罰金(円。歳入=区)
            "grievance": float(enf.get("grievance", 0.05)),  # 執行を受けた違反者の不満(factors 経由)
            # 勾留の最小形(制度深化2 第10バッチ。逮捕→勾留→処分の「時間による自由の剥奪」を
            # 1段で表す)。執行時に違反者をその場で detention_steps 步の行動停止(会話・発火・
            # 移動なし)にする。既定 0=拘束なし=従来と完全同一(即時罰金のみ)。
            "detention_steps": int(enf.get("detention_steps", 0)),
        },
        # 審議・パブコメ段階(制度深化 第9バッチ。現実の「請願→パブコメ→議会審議→議決(否決可)」
        # の多段パイプラインの最小形。摩擦ゼロの署名即成立を、否決されうる熟議過程にする)。既定 OFF=
        # 署名到達で従来どおり即成立/即投票(現行挙動と完全同一)。
        "deliberation": {
            "enabled": bool(delib.get("enabled", False)),
            "days": int(delib.get("days", 2)),              # 審議期間(日)。この間に反対が集まる
            "reject_ratio": float(delib.get("reject_ratio", 0.5)),  # 否決=反対数 > 賛同数×これ
        },
        # 代表制議会(制度深化3 2026-07-08。rights-institutions-gap の見送り分)。既定 OFF=
        # 従来どおり露出者の直接投票(現行挙動と完全同一)。ON: term_days ごとに住民(非来街者)が
        # 「意見が最も近い他者」へ投票する決定論選挙(Downs の空間投票の最近傍形・非LLM・乱数なし)で
        # size 人の議会を選び、提案の採決は露出者でなく議会が行う=代表による意思決定の歪みが入る。
        "assembly": {
            "enabled": bool(asm.get("enabled", False)),
            "size": int(asm.get("size", 9)),                # 議席数
            "term_days": int(asm.get("term_days", 30)),     # 任期(日)。満了で改選
            # 議会選挙の現実化(渋谷区議会準拠。制度深化3 batch37。既定 OFF=現行の意見最近傍選挙と
            # 完全同一=ゴールデン維持)。ON: 告示→自発的立候補(供託金)→SNTV開票→予算承認/条例議決/
            # 住民署名。すべて決定論(新規乱数 stream を引かない)。設計根拠 docs/research/council-vs-reality.md。
            "realism": {
                "enabled": bool(asm_real.get("enabled", False)),
                "announce_days": int(asm_real.get("announce_days", 5)),   # 告示期間(任期満了の何日前から)
                "deposit": float(asm_real.get("deposit", 300000.0)),      # 供託金(公選法30万円)
                "candidate_age_min": int(asm_real.get("candidate_age_min", 25)),  # 被選挙権(満25歳)
                "voter_age_min": int(asm_real.get("voter_age_min", 18)),  # 選挙権(満18歳)
                "w_op": float(asm_real.get("w_op", 1.0)),                 # SNTV効用: 意見距離の重み
                "w_cl": float(asm_real.get("w_cl", 0.5)),                 #   親密度(closeness)の重み
                "w_rep": float(asm_real.get("w_rep", 0.3)),               #   評判(reputation)の重み
                "legal_divisor": float(asm_real.get("legal_divisor", 10.0)),  # 法定得票=有効投票÷定数÷これ
                "cut_ratio": float(asm_real.get("cut_ratio", 0.8)),       # 予算否決時の歳出執行係数
                "grievance_threshold": float(asm_real.get("grievance_threshold", 0.5)),  # 予算承認の議員判定
                "signature_divisor": int(asm_real.get("signature_divisor", 50)),  # 住民提案=署名(有権者÷これ)
                "signature_min": int(asm_real.get("signature_min", 0)),   # 明示閾値(>0 なら優先)
            },
        },
    }


_ROUTES_OFF = build_routes_cfg(None)


def routes_of(sim) -> dict:
    """institution_routes 設定を遅延構築して sim にキャッシュ(government の _gov と同型)。

    構築は乱数を引かず・ログもしない=OFF 時の副作用ゼロ。sim.cfg が無い/未設定でも安全に
    既定 OFF(_ROUTES_OFF)へ後退する。"""
    cfg = getattr(sim, "routescfg", None)
    if cfg is not None:
        return cfg
    raw = getattr(sim, "cfg", None)
    raw = raw.get("institution_routes", None) if raw is not None else None
    cfg = build_routes_cfg(raw)
    try:
        sim.routescfg = cfg
    except Exception:
        pass
    return cfg


def elect_assembly(sim, step: int, sim_min: int) -> None:
    """代表制議会の改選(制度深化3 2026-07-08。既定 OFF=完全 no-op)。

    日次で呼ばれ、議会が無い or 任期満了の日にだけ改選する。選挙は決定論・非LLM・乱数なし:
    住民(非来街者)それぞれが「意見(opinion)が最も近い他の住民」に投票する(Downs の
    空間投票の最近傍形。opinion は FJ 力学で更新される非LLM状態)。得票上位 size 人が議員。
    同数は id 昇順。意見分布の密な山から代表が出る=民意の粗い写像になり、採決は
    _resolve_vote が議会に委ねる(直接投票との違い=代表制の歪みが観測できる)。"""
    routes = routes_of(sim)
    asm = routes["assembly"]
    if not asm["enabled"]:
        return
    if asm["realism"]["enabled"]:                      # 議会選挙の現実化(batch37。既定 OFF=下は不変)
        _assembly_realism(sim, step, sim_min, asm)
        return
    day = sim_min // 1440
    cur = getattr(sim, "council", None)
    if cur is not None and day < cur["elected_day"] + max(1, int(asm["term_days"])):
        return
    residents = [a for a in sim.agents if not a.visitor]
    if len(residents) < 2:
        return
    order = sorted(residents, key=lambda a: (a.opinion, a.id))
    votes: dict[int, int] = {}
    for i, voter in enumerate(order):                  # 最近傍候補は opinion 順で隣接に限る
        cands = [order[j] for j in (i - 1, i + 1) if 0 <= j < len(order)]
        best = min(cands, key=lambda c: (abs(c.opinion - voter.opinion), c.id))
        votes[best.id] = votes.get(best.id, 0) + 1
    size = max(1, int(asm["size"]))
    ranked = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))
    members = [aid for aid, _ in ranked[:size]]
    term = (cur["term"] + 1) if cur else 1
    sim.council = {"members": members, "elected_day": day, "term": term}
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=-1,
                         kind="council_elected", x=0.0, y=0.0,
                         payload={"term": term, "day": day, "members": members,
                                  "n_candidates": len(votes)}))
    sim.net.publish_news("議会の改選", f"住民の代表 {len(members)} 人が選ばれた", [], step)
    for aid in members:
        a = sim.agent_by_id.get(aid)
        if a is not None:
            a.remember("議会の議員に選ばれた", importance_bonus=0.4)


# ================================================================ 議会選挙の現実化(batch37・realism ON)
# 渋谷区議会準拠: 告示→自発的立候補(供託金)→SNTV開票→予算承認/条例議決/住民署名。すべて決定論・
# 非LLM・新規乱数 stream なし。realism OFF では下の関数はどれも呼ばれない(elect_assembly の分岐で return)。
def _is_voter(agent, realism) -> bool:
    """有権者か(選挙権): 非来街者かつ満 voter_age_min(既定18)歳以上。"""
    return (not agent.visitor
            and int(getattr(agent, "age", 0)) >= int(realism["voter_age_min"]))


def _is_eligible_candidate(agent, realism) -> bool:
    """被選挙権+供託資力の資格ゲート: 満 candidate_age_min(既定25)歳以上・非来街者・
    所持金+口座 ≥ deposit(既定30万円)。k 非依存の客観条件のみ(R1)。"""
    if agent.visitor:
        return False
    if int(getattr(agent, "age", 0)) < int(realism["candidate_age_min"]):
        return False
    funds = float(getattr(agent, "money", 0.0)) + float(getattr(agent, "account", 0.0))
    return funds >= float(realism["deposit"])


def _nn_ranked(sim) -> list:
    """現行の意見最近傍方式の得票順(フォールバック/縮退補充の補助)。elect_assembly の OFF 本体と
    同一ロジックだが本体には触れない(ゴールデンを守る)。決定論・乱数なし・id 昇順で同点解決。"""
    residents = [a for a in sim.agents if not a.visitor]
    if len(residents) < 2:
        return [a.id for a in residents]
    order = sorted(residents, key=lambda a: (a.opinion, a.id))
    votes: dict[int, int] = {}
    for i, voter in enumerate(order):
        cands = [order[j] for j in (i - 1, i + 1) if 0 <= j < len(order)]
        best = min(cands, key=lambda c: (abs(c.opinion - voter.opinion), c.id))
        votes[best.id] = votes.get(best.id, 0) + 1
    ranked = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))
    return [aid for aid, _ in ranked]


def _seat_council(sim, members, step) -> None:
    """当選者を議員として着任(ニュース+本人の記憶)。OFF 本体の着任と同型。"""
    sim.net.publish_news("区議会の改選", f"新しい区議 {len(members)} 人が決まった", [], step)
    for aid in members:
        a = sim.agent_by_id.get(aid)
        if a is not None:
            a.remember("区議会議員に選ばれた", importance_bonus=0.4)


def _settle_candidacy_deposit(sim, agent, deposit, step, sim_min, *, refund) -> None:
    """立候補供託金の返還/没収(SNTV開票で1回)。没収は government ON なら区歳入へ(_settle_deposit と同型)。"""
    if deposit <= 0.0:
        return
    if refund:
        agent.money += deposit
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                             kind="deposit", x=agent.x, y=agent.y,
                             payload={"candidacy": True, "amount": round(deposit, 1),
                                      "phase": "refund", "balance": round(agent.money, 1)}))
        return
    gov = getattr(sim, "government", None)
    if gov is not None and getattr(gov, "enabled", True):
        try:
            gov.collect("ward", deposit)               # 没収=区の歳入(government ON 時)
        except Exception:
            pass
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                         kind="deposit", x=agent.x, y=agent.y,
                         payload={"candidacy": True, "amount": round(deposit, 1),
                                  "phase": "forfeit"}))


def _council_budget(sim, step, sim_min, asm, cur) -> None:
    """(a) 予算承認: 現議会が government の区(ward)期次予算を日次で承認/減額(gov ON 時のみ)。

    各議員 yes/no は _vote_yes と同型の決定論(不満=grievance が閾値以上の議員は減額に回る)。
    否決(過半 no)なら government.exec_ratio を cut_ratio(既定0.8)へ=区の歳出執行が絞られる。
    council_budget イベントに承認可否・倍率・賛否を記録。1日1回(同日再実行しない)。"""
    gov = getattr(sim, "government", None)
    if gov is None or not bool(gov.cfg.get("enabled", False)):
        return                                          # government OFF なら予算承認は起きない
    day = sim_min // 1440
    if getattr(sim, "_council_budget_day", -1) == day:
        return
    sim._council_budget_day = day
    realism = asm["realism"]
    members = list(cur.get("members", []))
    if not members:
        return
    thr = float(realism["grievance_threshold"])
    yes = no = 0
    for aid in members:                                 # 各議員 yes/no(決定論・_vote_yes 同型)
        a = sim.agent_by_id.get(aid)
        if a is None:
            continue
        approve = float(a.states.get("grievance", 0.0)) < thr   # 高不満の議員は執行減額へ
        yes += 1 if approve else 0
        no += 0 if approve else 1
    total = yes + no
    approved = bool(total > 0 and (yes / total) > 0.5)  # 過半 yes で承認(否決=減額)
    gov.exec_ratio = 1.0 if approved else float(realism["cut_ratio"])
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=-1,
                         kind="council_budget", x=0.0, y=0.0,
                         payload={"term": int(cur.get("term", 0)), "day": day,
                                  "approved": approved,
                                  "amount": round(float(gov.balance.get("ward", 0.0)), 1),
                                  "ratio": round(gov.exec_ratio, 3),
                                  "yes": yes, "no": no}))


def _run_sntv(sim, step, sim_min, asm, day) -> None:
    """SNTV(単記非移譲式)開票=任期満了日の改選。有権者(18歳以上住民)が全候補から効用最大の
    1人へ1票: score = -|opinion差|·w_op + closeness·w_cl + reputation·w_rep(同点は id 昇順)。
    上位 size 人当選。法定得票(有効投票÷定数÷legal_divisor)未満は供託金没収。候補が定数未満なら
    全候補当選+不足分は現行方式で補充(縮退)。候補0なら現行の全住民方式へフォールバック(fallback:true)。"""
    realism = asm["realism"]
    size = max(1, int(asm["size"]))
    camp = getattr(sim, "council_campaign", None)
    deposits = dict(camp["candidates"]) if camp and camp.get("candidates") else {}
    sim.council_campaign = None                         # 開票で立候補受付を締める
    cur = getattr(sim, "council", None)
    term = (int(cur["term"]) + 1) if cur else 1
    cand_ids = sorted(deposits)                         # 候補(id 昇順=決定論の同点解決基準)
    cand_agents = [(cid, sim.agent_by_id.get(cid)) for cid in cand_ids]
    cand_agents = [(cid, a) for cid, a in cand_agents if a is not None]
    if not cand_agents:                                 # 候補0 → 現行の全住民方式へフォールバック
        elected = _nn_ranked(sim)[:size]
        sim.council = {"members": elected, "elected_day": day, "term": term}
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=-1,
                             kind="election_result", x=0.0, y=0.0,
                             payload={"day": day, "term": term, "votes": {},
                                      "elected": elected, "forfeited": [],
                                      "fallback": True}))
        _seat_council(sim, elected, step)
        return
    w_op = float(realism["w_op"])
    w_cl = float(realism["w_cl"])
    w_rep = float(realism["w_rep"])
    votes = {cid: 0 for cid, _ in cand_agents}
    voters = [a for a in sim.agents if _is_voter(a, realism)]
    for voter in sorted(voters, key=lambda a: a.id):    # id 昇順=決定論
        best_id = None
        best_key = None
        for cid, cand in cand_agents:                   # cand_agents は id 昇順
            closeness = float((voter.mem.relations.get(cid) or {}).get("closeness", 0.0))
            rep = float(getattr(cand, "_reputation", 0.0))
            score = (-abs(float(cand.opinion) - float(voter.opinion)) * w_op
                     + closeness * w_cl + rep * w_rep)
            key = (score, -cid)                         # 効用最大、同点は id 昇順(=-cid が大きい方)
            if best_key is None or key > best_key:
                best_key = key
                best_id = cid
        if best_id is not None:
            votes[best_id] += 1
    total = sum(votes.values())                         # 有効投票=投票した有権者数
    legal = (total / size / float(realism["legal_divisor"])) if size > 0 else 0.0
    ranked = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))
    elected = [cid for cid, _ in ranked[:size]]
    filled = []
    if len(cand_agents) < size:                         # 候補<定数: 全候補当選+現行方式で不足分補充(縮退)
        elected = [cid for cid, _ in ranked]
        for aid in _nn_ranked(sim):
            if len(elected) >= size:
                break
            if aid not in elected:
                elected.append(aid)
                filled.append(aid)
    forfeited = []
    for cid, cand in cand_agents:                       # 法定得票未満は供託金没収、それ以外は返還
        below = votes[cid] < legal
        if below:
            forfeited.append(cid)
        _settle_candidacy_deposit(sim, cand, float(deposits.get(cid, 0.0)),
                                  step, sim_min, refund=not below)
    sim.council = {"members": elected, "elected_day": day, "term": term}
    payload = {"day": day, "term": term, "votes": votes, "elected": elected,
               "forfeited": forfeited}
    if filled:                                          # 縮退補充を明記(現行方式で埋めた分)
        payload["filled"] = filled
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=-1,
                         kind="election_result", x=0.0, y=0.0, payload=payload))
    _seat_council(sim, elected, step)


def _assembly_realism(sim, step, sim_min, asm) -> None:
    """議会選挙の現実化の日次ドライバ(elect_assembly から realism ON 時のみ呼ばれる)。

    毎 step 呼ばれるが、各処理は日/期でゲートして冪等: (a) 現議会が区予算を承認/減額、
    告示期間に入れば立候補受付を1度だけ開く、任期満了日(or 初回=議会不在)に SNTV 開票。決定論。"""
    realism = asm["realism"]
    day = sim_min // 1440
    term_days = max(1, int(asm["term_days"]))
    announce_days = max(1, int(realism["announce_days"]))
    cur = getattr(sim, "council", None)
    if cur is not None and cur.get("members"):          # (a) 予算承認(gov ON 時のみ・1日1回)
        _council_budget(sim, step, sim_min, asm, cur)
    if cur is not None:                                 # 告示: 任期満了の announce_days 前から受付開始
        term_end = int(cur["elected_day"]) + term_days
        if term_end - announce_days <= day < term_end:
            camp = getattr(sim, "council_campaign", None)
            if camp is None or camp.get("term_end") != term_end:
                sim.council_campaign = {"term_end": term_end, "open": True,
                                        "candidates": {}}
                sim.net.publish_news(
                    "区議会議員選挙の告示",
                    f"立候補の受付が始まった(供託金 {int(realism['deposit'])} 円)", [], step)
    if cur is None or day >= int(cur["elected_day"]) + term_days:   # 開票(満了 or 初回)
        _run_sntv(sim, step, sim_min, asm, day)


def _is_employee(agent) -> bool:
    """被雇用者か(職域ルートの起点資格): 組織に配属され、かつ学生でない。

    org_id は organizations(職場・学校台帳)ON 時のみ付く。OFF なら False=労働争議は起きない。"""
    return (bool(getattr(agent, "org_id", None))
            and getattr(agent, "org_role", "") != "学生")


class Tools:
    """5 ツールの状態レジストリ + ライフサイクル。メソッドは sim を明示的に受ける。"""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.events: dict[int, dict] = {}
        self.groups: dict[int, dict] = {}
        self.proposals: dict[int, dict] = {}
        self.ventures: dict[int, dict] = {}            # owner_id -> 開店中の屋台(1人1店)
        self.flyers_by_node: dict[str, list] = defaultdict(list)
        self.ventures_by_node: dict[str, list] = defaultdict(list)
        self.member_of: dict[int, set] = defaultdict(set)   # agent_id -> {group_id}
        self.institutions: list[dict] = []                  # Searle 制度化(既定 OFF 時は空)
        self._event_seq = 0
        self._group_seq = 0
        self._proposal_seq = 0
        self.n_flyers_total = 0
        self.n_ventures_total = 0

    # ---------------------------------------------------------------- 行動心理プラグイン補助
    def share_group(self, a_id: int, b_id: int) -> bool:
        """2人が同じグループに属すか(集団プラグインの in-group 判定。engine から呼ぶ)。"""
        ga = self.member_of.get(a_id)
        gb = self.member_of.get(b_id)
        return bool(ga and gb and (ga & gb))

    def institution_lines(self) -> list[str] | None:
        """発火プロンプトに注入する「この街の取り決め」(最新2件の規範文)。無ければ None。"""
        if not self.institutions:
            return None
        return [inst["norm_text"] for inst in self.institutions[-2:]]

    # ---------------------------------------------------------------- 提示
    def offer_text(self, sim, agent) -> str | None:
        """発火時プロンプトに中立提示する「他にできること」1 行。実行可能な物だけ。"""
        if not self.cfg["enabled"]:
            return None
        rb = getattr(sim, "rulebook", None)
        # 制度DSL 有効時のみ、propose に機械可読の制度スロット例を1行で併記(中立トーン維持)。
        # 成立すれば rule がシミュ世界の実効ルールとして自動制定される。使用は勧めない。
        if rb is not None and rb.cfg["enabled"]:
            # 再帰性(第9バッチ・既定 OFF): repeal(既存の取り決めの廃止)を型の選択肢に併記。
            # OFF 時は従来と一字一句同一の文字列(バイト一致=ゴールデンを守る)。
            types = "fee/bonus/curfew/weekly_event"
            rec = getattr(sim, "recursion", None)
            if rec is not None and rec.cfg["enabled"] and rec.cfg["allow_repeal"]:
                types += ('/repeal。repeal は既存の取り決めの廃止: '
                          '{"type":"repeal","target":"取り決め名の一部"}')
            propose = ('{"action":"propose","text":"…","rule":{"type":"fee",'
                       '"target_cat":"food/cafe/shop/nightlife/taxi/bus","delta":-200}}'
                       f'(rule は任意。type は {types} から選べる)')
        else:
            propose = '{"action":"propose","text":"…"}'
        offers = [
            '{"action":"host_event","title":"…","hours_later":2}',
            '{"action":"post_flyer","text":"…"}',
            '{"action":"found_group","name":"…","purpose":"…"}',
            propose,
        ]
        if agent.money >= self.cfg["venture_cost"] and agent.id not in self.ventures:
            offers.append('{"action":"open_venture","name":"…","offer":"…"}')
        # 議会選挙の現実化(realism ON): 告示期間中の資格者にだけ「立候補」を1件追加(既定 OFF=一字も
        # 足さない=ゴールデン維持)。誰が立候補するかは LLM 判断=ファウンダー観察の一部(勧めない)。
        cand = self._candidacy_offer(sim, agent)
        if cand:
            offers.append(cand)
        return "他にできること(使っても使わなくてもよい): " + " / ".join(offers)

    # -------------------------------------------------- 議会選挙の現実化: 立候補(realism ON)
    def _candidacy_offer(self, sim, agent) -> str | None:
        """告示期間中の資格者にだけ返す「立候補」提示文(それ以外は None=一字も足さない)。

        既存 tools 提示と同型の JSON(propose 形式+rule 印)。realism OFF・告示期間外・立候補済み・
        資格なし(年齢/来街者/供託金)のいずれかなら None(=OFF ではメニュー文字列が完全不変)。"""
        asm = routes_of(sim)["assembly"]
        if not (asm["enabled"] and asm["realism"]["enabled"]):
            return None
        camp = getattr(sim, "council_campaign", None)
        if not camp or not camp.get("open"):
            return None
        if agent.id in camp.get("candidates", {}):
            return None
        if not _is_eligible_candidate(agent, asm["realism"]):
            return None
        return ('{"action":"propose","text":"区議会議員に立候補します",'
                '"rule":{"type":"candidacy"}}')

    def _is_candidacy_action(self, sim, action) -> bool:
        """propose アクションが「立候補」印つきか(realism ON かつ rule.type=="candidacy")。"""
        asm = routes_of(sim)["assembly"]
        if not (asm["enabled"] and asm["realism"]["enabled"]):
            return False
        rule = action.get("rule")
        return isinstance(rule, dict) and str(rule.get("type", "")) == "candidacy"

    def declare_candidacy(self, sim, agent, step, sim_min) -> None:
        """自発的立候補の受理(realism ON・告示期間中・資格者)。供託金を納付(money→不足分は口座)し
        candidacy を記録。強制しない=誰が立つかはファウンダー観察の一部。資格外/期間外は静かに no-op。"""
        asm = routes_of(sim)["assembly"]
        if not (asm["enabled"] and asm["realism"]["enabled"]):
            return
        camp = getattr(sim, "council_campaign", None)
        if not camp or not camp.get("open"):
            return                                      # 告示期間外
        if agent.id in camp.get("candidates", {}):
            return                                      # 既に立候補済み
        realism = asm["realism"]
        if not _is_eligible_candidate(agent, realism):
            return                                      # 資格ゲート(年齢/来街者/供託金)不合格
        deposit = float(realism["deposit"])
        pay = min(float(agent.money), deposit)          # 供託金納付: 所持金→不足分は口座
        agent.money -= pay
        rem = deposit - pay
        if rem > 0.0:
            agent.account = float(getattr(agent, "account", 0.0)) - rem
        camp["candidates"][agent.id] = deposit
        self._log(sim, step, sim_min, agent, "candidacy",
                  {"day": sim_min // 1440, "deposit": round(deposit, 1),
                   "age": int(getattr(agent, "age", 0)),
                   "balance": round(agent.money, 1)})
        sim.net.post(agent.id, "区議会議員選挙に立候補します。", [], step)
        agent.remember("区議会議員選挙に立候補した")

    def _signature_threshold(self, sim, realism) -> int:
        """住民提案の署名閾値(realism ON): 明示 signature_min>0 があれば優先、なければ ceil(有権者/50)。"""
        if int(realism.get("signature_min", 0)) > 0:    # 現行キー(明示閾値)があれば優先
            return int(realism["signature_min"])
        voters = sum(1 for a in sim.agents if _is_voter(a, realism))
        div = max(1, int(realism["signature_divisor"]))
        return (voters + div - 1) // div                # ceil(有権者 / signature_divisor)

    def membership_names(self, agent) -> list[str] | None:
        gids = self.member_of.get(agent.id)
        if not gids:
            return None
        names = [self.groups[g]["name"] for g in sorted(gids) if g in self.groups]
        return names or None

    def invite(self, agent, event_id: int) -> None:
        known = getattr(agent, "_known_events", None)
        if known is None:
            agent._known_events = known = set()
        known.add(int(event_id))

    # ---------------------------------------------------------------- 行動適用
    def apply(self, sim, agent, action: dict, step: int, sim_min: int) -> None:
        kind = action["type"]
        if kind == "host_event":
            self._host_event(sim, agent, action, step, sim_min)
        elif kind == "post_flyer":
            self._post_flyer(sim, agent, action, step, sim_min)
        elif kind == "found_group":
            self._found_group(sim, agent, action, step, sim_min)
        elif kind == "propose":
            # 議会選挙の現実化(realism ON): 「立候補」はメニュー上 propose 形式で提示し、rule 印
            # {"type":"candidacy"} で判別する(既存 tools 行使経路=deliberate の parse_action→_apply→
            # tools.apply にそのまま乗る)。印つきかつ realism ON のときだけ立候補として処理。
            if self._is_candidacy_action(sim, action):
                self.declare_candidacy(sim, agent, step, sim_min)
            else:
                self._propose(sim, agent, action, step, sim_min)
        elif kind == "open_venture":
            self._open_venture(sim, agent, action, step, sim_min)

    # -- 1. host_event -----------------------------------------------------
    def _host_event(self, sim, agent, action, step, sim_min) -> None:
        title = str(action.get("title", "")).strip()
        if not title:
            return
        hours = max(1, min(6, int(action.get("hours_later", 1))))
        start_step = step + hours * 6
        start_min = sim.clock.sim_min(start_step)
        if (start_min % 1440) < 6 * 60:                # 深夜(0-6時)開始 → 翌朝9時に繰延
            day = start_min // 1440
            start_step = (day * 1440 + 9 * 60 - sim.clock.start_min) // STEP_MINUTES
        eid = self._event_seq
        self._event_seq += 1
        from .engine import scheduler
        place = scheduler._place_of(sim, agent)
        self.events[eid] = {
            "id": eid, "host": agent.id, "node": agent.node,
            "building": agent.building, "title": title, "place": place,
            "start_step": int(start_step),
            "end_step": int(start_step) + self.cfg["event_duration_steps"],
            "attendees": set(), "started": False, "ended": False,
        }
        self._log(sim, step, sim_min, agent, "event_host",
                  {"event_id": eid, "title": title, "node": agent.node,
                   "place": place, "start_step": int(start_step)})
        # 自動 SNS 告知(既存の閲覧・伝播に乗る。event_id で参加告知に紐付く)
        items = [w for w in sorted(agent.adopted) if w in title]
        sim.net.post(agent.id, f"【イベント】{title} @{place}", items, step,
                     event_id=eid)
        for w in items:
            sim.labels.use(agent, w, step=step, sim_min=sim_min, logger=sim.logger)
        # 関係台帳の上位5人へ自動 DM 告知(既存 dm 配線・drive に乗る)
        top = [oid for oid, _ in sorted(
            agent.mem.relations.items(),
            key=lambda kv: (-kv[1]["count"], kv[0]))[:5]]
        self._announce_dm(sim, agent, top, f"{title}やります。よかったら来て。",
                          eid, step, sim_min)
        agent.remember(f"イベントを企画した:「{title}」")

    def _announce_dm(self, sim, host, recipients, text, event_id, step, sim_min):
        for rid in recipients:
            rec = sim.agent_by_id.get(rid)
            if rec is None or rec.loc == "outside" or rec.sleeping:
                continue
            rec.remember(f"{host.name}からの誘い:「{text}」", kind="dm")
            rec._last_dm_from = host.id
            rec.mem.record_contact(host.id, host.name, step, text)
            host.mem.record_contact(rid, rec.name, step)
            drive.add(rec, "dm_received", sim.drivecfg)
            self.invite(rec, event_id)
            sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=host.id,
                                 kind="dm", x=host.x, y=host.y,
                                 payload={"to": rid, "text": text}))

    # -- 2. post_flyer -----------------------------------------------------
    def _post_flyer(self, sim, agent, action, step, sim_min) -> None:
        text = str(action.get("text", "")).strip()
        if not text:
            return
        node = agent.node
        items = [w for w in sorted(agent.adopted) if w in text]
        flyer = {"author": agent.id, "node": node, "text": text, "items": items,
                 "expire_step": step + self.cfg["flyer_ttl_steps"], "viewers": set()}
        lst = self.flyers_by_node[node]
        lst.append(flyer)
        while len(lst) > self.cfg["flyer_max_per_node"]:   # 古い順に上書き(最大3枚)
            old = lst.pop(0)
            self._log_node(sim, step, sim_min, old["author"], node, "flyer_expire",
                           {"author": old["author"], "node": node,
                            "reason": "overflow"})
        self.n_flyers_total += 1
        for w in items:
            sim.labels.use(agent, w, step=step, sim_min=sim_min, logger=sim.logger)
        self._log(sim, step, sim_min, agent, "flyer_post",
                  {"author": agent.id, "node": node, "text": text, "items": items})
        agent.remember(f"貼り紙を出した:「{text[:20]}」")

    # -- 3. found_group ----------------------------------------------------
    def _found_group(self, sim, agent, action, step, sim_min) -> None:
        name = str(action.get("name", "")).strip()
        if not name:
            return
        purpose = str(action.get("purpose", "")).strip()
        sim.labels.coin(agent, name, step=step, sim_min=sim_min, logger=sim.logger)
        gid = self._group_seq
        self._group_seq += 1
        self.groups[gid] = {"id": gid, "founder": agent.id, "name": name,
                            "purpose": purpose, "members": {agent.id}}
        self.member_of[agent.id].add(gid)
        self._log(sim, step, sim_min, agent, "group_found",
                  {"group_id": gid, "name": name, "purpose": purpose,
                   "founder": agent.id})
        sim.net.post(agent.id, f"「{name}」というグループを作りました。{purpose}",
                     [name], step)
        sim.labels.use(agent, name, step=step, sim_min=sim_min, logger=sim.logger)
        agent.remember(f"グループ「{name}」を結成した")

    # -- 4. propose --------------------------------------------------------
    def _propose(self, sim, agent, action, step, sim_min) -> None:
        text = str(action.get("text", "")).strip()
        if not text:
            return
        sim.labels.coin(agent, text, step=step, sim_min=sim_min, logger=sim.logger)
        # 供託金(制度深化 第9バッチ・既定 0=不変): 投票ルート ON かつ deposit>0 のとき、正式受理には
        # 所持金からの拠出が要る。払えなければ「演説どまり」=語の使用と SNS 投稿はするが提案は受理
        # されない(署名も集まらない)。政治参入の客観的な所持金障壁(R1: k 非依存)。
        vcfg = routes_of(sim)["vote"]
        deposit = float(vcfg["deposit"]) if vcfg["enabled"] else 0.0
        if deposit > 0.0 and agent.money < deposit:
            self._log(sim, step, sim_min, agent, "deposit",
                      {"amount": round(deposit, 1), "phase": "insufficient",
                       "balance": round(agent.money, 1)})
            sim.net.post(agent.id, f"【提案】{text}", [text], step)
            sim.labels.use(agent, text, step=step, sim_min=sim_min, logger=sim.logger)
            agent.remember(f"提案を訴えたが供託金が足りず受理されなかった:「{text[:20]}」")
            return
        pid = self._proposal_seq
        self._proposal_seq += 1
        self.proposals[pid] = {"id": pid, "author": agent.id, "text": text,
                               "supporters": {agent.id}, "passed": False,
                               "rule": action.get("rule")}   # 制度DSL: 機械可読スロット
        if deposit > 0.0:                              # 供託金の拠出(受理料。開票/否決まで預かり)
            agent.money = max(0.0, agent.money - deposit)
            self.proposals[pid]["deposit"] = deposit
            self._log(sim, step, sim_min, agent, "deposit",
                      {"proposal_id": pid, "amount": round(deposit, 1),
                       "phase": "paid", "balance": round(agent.money, 1)})
        self._log(sim, step, sim_min, agent, "proposal",
                  {"proposal_id": pid, "text": text})
        rec = getattr(sim, "recursion", None)         # 再帰性: 客観カウント(OFF は no-op)
        if rec is not None:
            rec.note("proposals")
        sim.net.post(agent.id, f"【提案】{text}", [text], step)
        sim.labels.use(agent, text, step=step, sim_min=sim_min, logger=sim.logger)
        agent.remember(f"提案をした:「{text[:20]}」")
        # 労働争議(職域ルート Wave G3・既定 OFF): 被雇用者の提案は「労働争議」variant として記録し、
        # 賛同/露出を同じ org の同僚に偏らせる(下記 _proposal_support)。起点は既存 propose の再利用
        # =新規 LLM 呼び出しなし。OFF or 非被雇用者なら何も足さない(バイト一致)。
        if routes_of(sim)["labor"]["enabled"] and _is_employee(agent):
            pr = self.proposals[pid]
            pr["labor"] = True
            pr["org"] = getattr(agent, "org_id", None)
            self._log(sim, step, sim_min, agent, "labor_action",
                      {"org": pr["org"], "demand": text[:24], "initiator": agent.id})

    # -- 5. open_venture ---------------------------------------------------
    def _open_venture(self, sim, agent, action, step, sim_min) -> None:
        cost = self.cfg["venture_cost"]
        if agent.money < cost or agent.id in self.ventures:
            return                                     # 所持金不足/1人1店(防御)
        name = str(action.get("name", "")).strip()
        if not name:
            return
        offer = str(action.get("offer", "")).strip()
        # 破産の制限期間(制度深化3・既定 0=無効): 免責直後は出店できない(破産の資格制限のモデル)。
        # permit の乱数 draw より前に返す(stream はステートレス=他 agent の draw に不干渉)。
        if step < int(getattr(agent, "bankrupt_until", 0)):
            self._log(sim, step, sim_min, agent, "venture_permit",
                      {"name": name, "outcome": "denied", "reason": "bankruptcy"})
            agent.remember("破産直後で出店の許可が下りなかった")
            return
        # 軽微な逸脱(P2 #10 deviance・既定 OFF): 無許可出店を「選べる」。deviance 有効かつ
        # permit:false のとき、許可待ち・却下抽選をスキップして即開店するが permitted:false
        # (=近傍の警察官が違反として摘発する=既存 enforcement 機構に接続。暴力・実在人物は扱わない R17)。
        fcfg = getattr(sim, "freedomcfg", None)
        deviance_on = bool(fcfg and fcfg.get("p2", {}).get("deviance"))
        unpermitted = bool(deviance_on and action.get("permit") is False)
        # 営業許可(制度深化2・既定 0/0=従来と完全同一): 却下なら開業費を払わず出店なし。
        # 許可待ち(permit_steps)の間は開店済みでも販売できない(open_at まで pending)。
        permit_steps = int(self.cfg["permit_steps"])
        if unpermitted:
            permit_steps = 0                            # 無許可=許可待ちなしで即時営業(だが違反)
        elif float(self.cfg["permit_deny_prob"]) > 0.0:
            rng = sim.hub.stream("permit", agent.id, step)   # 新 stream(既存 draw 順に不干渉)
            if rng.random() < float(self.cfg["permit_deny_prob"]):
                self._log(sim, step, sim_min, agent, "venture_permit",
                          {"name": name, "outcome": "denied"})
                agent.remember(f"「{name}」の出店許可が下りなかった")
                return
        agent.money = max(0.0, agent.money - cost)
        venture = {"owner": agent.id, "node": agent.node, "name": name,
                   "offer": offer, "price": self.cfg["venture_price"],
                   "opened_step": step, "last_sale_step": step + permit_steps,
                   "open_at": step + permit_steps,     # 許可待ち明けから販売可(既定 0=即時)
                   "permitted": not unpermitted,        # 無許可出店(#10)は摘発対象(既定 True=不変)
                   "sales_total": 0.0, "fulltime": False}   # 起業転換(Wave G5)の累計売上・転換済みフラグ
        self.ventures[agent.id] = venture
        self.ventures_by_node[agent.node].append(venture)
        self.n_ventures_total += 1
        open_payload = {"name": name, "offer": offer, "node": agent.node,
                        "cost": round(cost, 1), "balance": round(agent.money, 1)}
        if unpermitted:                                 # 既定(permitted)は payload 不変=バイト一致
            open_payload["permitted"] = False
        self._log(sim, step, sim_min, agent, "venture_open", open_payload)
        if permit_steps > 0:                           # 許可制: 開業待ちを記録(既定 0=出さない)
            self._log(sim, step, sim_min, agent, "venture_permit",
                      {"name": name, "outcome": "granted",
                       "open_step": step + permit_steps})
        sim.net.post(agent.id, f"「{name}」始めました。{offer}", [], step)
        agent.remember(f"「{name}」を開店した")

    # ---------------------------------------------------------------- 毎 step
    def phase(self, sim, step: int, sim_min: int) -> None:
        if not self.cfg["enabled"]:
            return
        self._events_start(sim, step, sim_min)
        self._events_retry(sim, step, sim_min)     # 屋内→退館後の保留参加者を再試行(#12)
        self._events_end(sim, step, sim_min)
        self._flyers_expire(sim, step, sim_min)
        self._ventures_close(sim, step, sim_min)
        self._ventures_fulltime(sim, step, sim_min)   # 起業転換(Wave G5・既定 OFF=no-op)
        self._group_joins(sim, step, sim_min)
        self._proposal_support(sim, step, sim_min)

    def _events_start(self, sim, step, sim_min) -> None:
        for eid in sorted(self.events):
            ev = self.events[eid]
            if ev["started"] or step < ev["start_step"]:
                continue
            ev["started"] = True
            rng = sim.hub.stream("event", ev["id"], step)
            base = self.cfg["attend_base"]
            bonus = self.cfg["attend_relation_bonus"]
            # 社会的ヒエラルキー(動員): 主催者の地位ほど集客が増える(OFF=0=従来と完全同一)。
            attract = status_mod.attract_bonus(sim, ev["host"])
            from .engine import scheduler
            for agent in sim.agents:                   # id 昇順=決定論
                if ev["id"] not in getattr(agent, "_known_events", ()):
                    continue
                if self._free_to_move(agent):
                    # 街路にいて空いている: 従来どおり ("event",...) で抽選(draw 順不変)
                    p = base + (bonus if ev["host"] in agent.mem.relations else 0.0)
                    if attract:
                        p = min(1.0, p + attract)      # 主催者 status で上乗せ(閾値のみ変更)
                    if rng.random() >= p:
                        continue
                    if agent.node == ev["node"]:       # すでに会場ノードにいる → 即参加
                        self._record_attend(sim, ev, agent, step, sim_min)
                    else:                              # 会場へ向かう(移動は _phase_move)
                        self._route_to(sim, agent, ev, step, sim_min)
                elif self._indoor_free(agent):
                    # 屋内(勤務・睡眠でない): 建物を出てから会場へ。抽選は退館後の再試行で。
                    # ★ここでは ("event",...) を一切引かない=街路経路の draw 順を完全維持。
                    agent._attending_event = ev["id"]
                    agent.stay_until = step             # routine の exit_building を誘発

    def _free_to_move(self, agent) -> bool:
        return (agent.loc == "street" and not agent.sleeping and not agent.building
                and agent.activity not in ("working", "commuting")
                and not agent.route and not agent.exit_intent and not agent.homing)

    def _indoor_free(self, agent) -> bool:
        """屋内で自由参加できる状態(#12): 建物内・非睡眠・勤務/通勤でない・帰宅意図なし。"""
        return (bool(agent.building) and not agent.sleeping
                and agent.activity not in ("working", "commuting")
                and not agent.exit_intent and not agent.homing)

    def _events_retry(self, sim, step, sim_min) -> None:
        """開始済み・未終了イベントの保留参加者(退館待ち→街路に出た人)を再試行(#12)。

        再試行の抽選は**新stream ("event_retry", event_id, step)**(既存 ("event",...)
        の draw 順に影響しない)。街路に出て空いたら 1 度だけ抽選し、通れば会場へ発つ。
        """
        base = self.cfg["attend_base"]
        bonus = self.cfg["attend_relation_bonus"]
        for eid in sorted(self.events):
            ev = self.events[eid]
            if not ev["started"] or ev["ended"]:
                continue
            rng = sim.hub.stream("event_retry", ev["id"], step)
            attract = status_mod.attract_bonus(sim, ev["host"])   # 主催者 status(OFF=0)
            for agent in sim.agents:                   # id 昇順=決定論
                if getattr(agent, "_attending_event", None) != ev["id"]:
                    continue
                if agent.id in ev["attendees"]:
                    agent._attending_event = None
                    continue
                if not self._free_to_move(agent):      # まだ退館中/移動中 → 次stepで再試行
                    continue
                if agent.node == ev["node"]:           # 会場ノードに出られた → 即参加
                    self._record_attend(sim, ev, agent, step, sim_min)
                    continue
                p = base + (bonus if ev["host"] in agent.mem.relations else 0.0)
                if attract:
                    p = min(1.0, p + attract)          # 主催者 status で上乗せ(閾値のみ変更)
                if rng.random() < p:
                    self._route_to(sim, agent, ev, step, sim_min)
                else:
                    agent._attending_event = None      # 抽選落ち: 参加をやめる(再試行は1回)

    def _route_to(self, sim, agent, ev, step, sim_min) -> None:
        path, mode = sim.router.route(agent.node, ev["node"], "walk")
        if len(path) < 2:
            return
        agent.route = path[1:]
        agent.edge_offset = 0.0
        agent.dest = ev["node"]
        agent.trip_mode = mode
        agent.activity = ""
        agent._pending_stay = self.cfg["event_duration_steps"]
        agent._pending_activity = ""
        agent._attending_event = ev["id"]
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                             kind="route_start", x=agent.x, y=agent.y,
                             payload={"dest": ev["node"],
                                      "dest_name": sim.city.node_name(ev["node"]),
                                      "mode": mode, "exit": False, "homing": False,
                                      "n_nodes": len(path),
                                      "dist_m": round(sim.router.route_length(path), 1)}))

    def _record_attend(self, sim, ev, agent, step, sim_min) -> None:
        if agent.id in ev["attendees"] or ev["ended"]:
            agent._attending_event = None
            return
        ev["attendees"].add(agent.id)
        agent.stay_until = max(agent.stay_until,
                               step + self.cfg["event_duration_steps"])
        agent._attending_event = None
        self._log(sim, step, sim_min, agent, "event_attend",
                  {"event_id": ev["id"], "host": ev["host"], "title": ev["title"]})
        agent.remember(f"「{ev['title']}」に参加した", kind="event")
        apply_bonus(getattr(sim, "rulebook", None), sim, agent, "event_attend",
                    step, sim_min)                     # 制度DSL: bonus(event_attend)

    def _events_end(self, sim, step, sim_min) -> None:
        for eid in sorted(self.events):
            ev = self.events[eid]
            if ev["ended"] or not ev["started"] or step < ev["end_step"]:
                continue
            ev["ended"] = True
            self._educate(sim, ev, step, sim_min)
            # 集団プラグイン(既定 OFF): グループイベントに十分な参加 → 集団的成功
            self._collective_event_success(sim, ev, step, sim_min)

    def _educate(self, sim, ev, step, sim_min) -> None:
        """勉強会=教育軸: 主催者の adopted 語がタイトルにあれば、出席者へ2回分聴取。

        complex contagion の閾値2を1回の教育で満たす(=「人を作る」の帯域)。
        """
        host = sim.agent_by_id.get(ev["host"])
        if host is None:
            return
        words = [w for w in sorted(host.adopted) if w in ev["title"]]
        if not words:
            return
        from .engine import scheduler
        for aid in sorted(ev["attendees"]):
            if aid == host.id:
                continue
            learner = sim.agent_by_id.get(aid)
            if learner is None:
                continue
            for w in words:
                scheduler._hear_words(sim, learner, [w], host.id, "event",
                                      step, sim_min)
                scheduler._hear_words(sim, learner, [w], host.id, "event",
                                      step, sim_min)

    def _flyers_expire(self, sim, step, sim_min) -> None:
        for node in sorted(self.flyers_by_node):
            keep = []
            for f in self.flyers_by_node[node]:
                if step >= f["expire_step"]:
                    self._log_node(sim, step, sim_min, f["author"], node,
                                   "flyer_expire",
                                   {"author": f["author"], "node": node,
                                    "reason": "ttl"})
                else:
                    keep.append(f)
            self.flyers_by_node[node] = keep

    def _ventures_close(self, sim, step, sim_min) -> None:
        close_steps = self.cfg["venture_close_days"] * 144
        for owner in sorted(self.ventures):
            v = self.ventures[owner]
            if step - v["last_sale_step"] < close_steps:
                continue
            self._log_node(sim, step, sim_min, owner, v["node"], "venture_close",
                           {"name": v["name"], "node": v["node"]})
            self.ventures_by_node[v["node"]] = [
                x for x in self.ventures_by_node.get(v["node"], []) if x is not v]
            del self.ventures[owner]

    def force_close_venture(self, sim, agent, step, sim_min, *, reason: str) -> bool:
        """屋台の強制閉店(制度深化3: 破産の清算)。閉店したら True、持っていなければ False。

        _ventures_close と同じ後始末(venture_close イベント+両索引から除去)に reason を添える。"""
        v = self.ventures.get(agent.id)
        if v is None:
            return False
        self._log_node(sim, step, sim_min, agent.id, v["node"], "venture_close",
                       {"name": v["name"], "node": v["node"], "reason": reason})
        self.ventures_by_node[v["node"]] = [
            x for x in self.ventures_by_node.get(v["node"], []) if x is not v]
        del self.ventures[agent.id]
        return True

    def _ventures_fulltime(self, sim, step, sim_min) -> None:
        """起業転換(Wave G5・既定 OFF): 好調な open_venture の所有者が本業を辞め出店を本業にする。

        career.enabled かつ venture_fulltime_sales>0 のときのみ。累計売上 ≥ 閾値 かつ 開店から
        venture_fulltime_days 以上 経った屋台の所有者を、被雇用者から自営(venture 一本)へ完全転換する
        (organizations.go_fulltime_venture が本業 org・賃金をクリアし勤務地を屋台へ移す)。1店1回限り。
        career OFF / 閾値未達 / 転換済み / 来街者 なら完全 no-op(venture_fulltime 0 件=バイト一致)。"""
        cfg = getattr(sim, "careercfg", None)
        if not cfg or not cfg["enabled"]:
            return
        min_sales = float(cfg["venture_fulltime_sales"])
        if min_sales <= 0.0:
            return
        min_days = int(cfg["venture_fulltime_days"])
        from .organizations import go_fulltime_venture
        for owner in sorted(self.ventures):
            v = self.ventures[owner]
            if v.get("fulltime"):
                continue
            if v.get("sales_total", 0.0) < min_sales:
                continue
            if (step - v["opened_step"]) // 144 < min_days:
                continue
            agent = sim.agent_by_id.get(owner)
            if agent is None or agent.visitor:
                continue
            v["fulltime"] = True
            go_fulltime_venture(agent, v)
            self._log_node(sim, step, sim_min, owner, v["node"], "venture_fulltime",
                           {"name": v["name"], "node": v["node"]})
            agent.remember(f"「{v['name']}」を本業にした")

    def _group_joins(self, sim, step, sim_min) -> None:
        prob = self.cfg["group_join_prob"]
        for gid in sorted(self.groups):
            g = self.groups[gid]
            members = g["members"]
            # 社会的ヒエラルキー(動員): 創設者の地位ほど加入が増える(OFF=0=従来と完全同一)。
            bonus = status_mod.attract_bonus(sim, g["founder"])
            p = min(1.0, prob + bonus) if bonus else prob
            for agent in sim.agents:                   # id 昇順
                if agent.id in members:
                    continue
                if g["name"] not in agent.adopted:     # グループ名を聞いた=採用済み
                    continue
                if not any(m in agent.mem.relations for m in members):  # 関係者がいる
                    continue
                rng = sim.hub.stream("group", gid, agent.id, step)
                if rng.random() >= p:
                    continue
                members.add(agent.id)
                self.member_of[agent.id].add(gid)
                # #13: 新メンバーは founder の投稿をタイムライン優先枠に載せる
                sim.net.set_priority(agent.id, g["founder"])
                for m in sorted(members):              # 相互フォロー(メンバー全員と)
                    if m != agent.id:
                        sim.net.add_contact(agent.id, m)
                        sim.net.add_contact(m, agent.id)
                self._log(sim, step, sim_min, agent, "group_join",
                          {"group_id": gid, "name": g["name"],
                           "founder": g["founder"]})
                agent.remember(f"「{g['name']}」に加入した")

    def _proposal_support(self, sim, step, sim_min) -> None:
        n = len(sim.agents)
        routes = routes_of(sim)                         # 3ルート設定(既定 OFF=現行挙動)
        asm = routes["assembly"]
        # (c) 住民提案=署名モデル(議会現実化 realism ON): 成立閾値を「署名=有権者の1/50」で解釈する。
        # OFF は従来どおり proposal_threshold×人口(バイト一致=ゴールデン維持)。
        if asm["enabled"] and asm["realism"]["enabled"]:
            thr = self._signature_threshold(sim, asm["realism"])
        else:
            thr = self.cfg["proposal_threshold"] * n
        labor_on = routes["labor"]["enabled"]
        vote_on = routes["vote"]["enabled"]
        for pid in sorted(self.proposals):
            pr = self.proposals[pid]
            if pr.get("decided"):                       # 投票モードで可決/否決が確定済み
                continue
            is_labor = bool(labor_on and pr.get("labor"))
            org = pr.get("org")
            # 審議中(制度深化・既定 OFF)は露出=自動署名でなく**立場表明**: 賛成(決定論 yes)なら
            # 署名、反対なら review.opposed へ(パブコメ=賛否の意見募集)。審議前は従来どおり自動署名。
            review = pr.get("review") if routes["deliberation"]["enabled"] else None
            for agent in sim.agents:                   # id 昇順
                if agent.id in pr["supporters"]:
                    continue
                if review is not None and agent.id in review["opposed"]:
                    continue
                if pr["text"] in agent.adopted:        # 露出2回で採用=署名とみなす(民主/一般)
                    if review is not None \
                            and not self._vote_yes(agent, pr, routes["vote"]):
                        review["opposed"].add(agent.id)   # 反対に回る(署名しない)
                        agent.remember(f"提案に反対した:「{pr['text'][:20]}」")
                        continue
                elif is_labor and getattr(agent, "org_id", None) == org \
                        and _is_employee(agent):        # 職域: 同じ org の同僚は職場露出で賛同(偏り)
                    pass
                else:
                    continue
                pr["supporters"].add(agent.id)
                self._log(sim, step, sim_min, agent, "proposal_support",
                          {"proposal_id": pid, "author": pr["author"]})
                agent.remember(f"提案に賛同した:「{pr['text'][:20]}」")
            if pr["passed"] or len(pr["supporters"]) < thr:
                continue
            # 審議・パブコメ段階(制度深化 第9バッチ・既定 OFF=即成立/即投票のまま)。署名到達で
            # 即決せず、審議期間のあいだ反対(認知者のうち決定論の no)を集め、多ければ否決する。
            # 現実の「請願→パブコメ→議会審議→議決(否決可)」の最小形=制度改変の摩擦・正当性コスト。
            if routes["deliberation"]["enabled"]:
                if not self._deliberate_proposal(sim, pr, step, sim_min, routes):
                    continue                            # 審議中 or 否決 → 今stepは決しない
            if vote_on:                                 # 民主ルート: 露出者で決定論投票→過半数可決
                self._resolve_vote(sim, pr, step, sim_min, routes["vote"])
                pr["decided"] = True
            else:                                       # 署名モード(既定): 閾値到達で象徴的成立
                self._pass_proposal(sim, pr, step, sim_min)

    # ------------------------------------------------------------ 審議・パブコメ(制度深化 第9バッチ)
    def _deliberate_proposal(self, sim, pr, step, sim_min, routes) -> bool:
        """署名到達後の審議段階。True=審議満了・可決経路へ進んでよい / False=審議中 or 否決。

        開始: review レコードを作り proposal_review(start)。審議中の反対の蓄積は _proposal_support
        本体が担う(露出=自動署名でなく立場表明: 決定論 yes なら署名・no なら opposed)。
        満了: 反対 > 賛同×reject_ratio なら否決(proposal_review(reject)+供託金没収)、
        でなければ proposal_review(approve) を出して可決経路(投票 or 署名成立)へ。
        決定論・非LLM・乱数なし(R1: generate() を足さない)。既定 OFF=この関数は呼ばれない。"""
        delib = routes["deliberation"]
        day = sim_min // 1440
        rv = pr.get("review")
        author = sim.agent_by_id.get(pr["author"])
        ax, ay = (author.x, author.y) if author is not None else (0.0, 0.0)
        if rv is None:
            pr["review"] = {"start_day": day, "opposed": set()}
            sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=pr["author"],
                                 kind="proposal_review", x=ax, y=ay,
                                 payload={"proposal_id": pr["id"], "phase": "start",
                                          "supporters": len(pr["supporters"])}))
            sim.net.publish_news("提案が審議入り",
                                 f"「{pr['text'][:24]}」の賛否の意見を募集", [], step)
            return False
        opposed = rv["opposed"]
        if day < rv["start_day"] + delib["days"]:
            return False                                # 審議継続
        if len(opposed) > delib["reject_ratio"] * len(pr["supporters"]):
            pr["decided"] = True                        # 否決(現実の議会否決に相当)
            sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=pr["author"],
                                 kind="proposal_review", x=ax, y=ay,
                                 payload={"proposal_id": pr["id"], "phase": "reject",
                                          "supporters": len(pr["supporters"]),
                                          "opposed": len(opposed)}))
            sim.net.publish_news("提案は否決",
                                 f"「{pr['text'][:24]}」は反対多数で見送り", [], step)
            self._settle_deposit(sim, pr, step, sim_min, refund=False)
            return False
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=pr["author"],
                             kind="proposal_review", x=ax, y=ay,
                             payload={"proposal_id": pr["id"], "phase": "approve",
                                      "supporters": len(pr["supporters"]),
                                      "opposed": len(opposed)}))
        return True

    def _settle_deposit(self, sim, pr, step, sim_min, *, refund: bool) -> None:
        """供託金の返還/没収(1回だけ)。没収は government ON なら区の歳入へ。既定 0=何も起きない。"""
        dep = float(pr.get("deposit") or 0.0)
        if dep <= 0.0 or pr.get("deposit_settled"):
            return
        pr["deposit_settled"] = True
        author = sim.agent_by_id.get(pr["author"])
        ax, ay = (author.x, author.y) if author is not None else (0.0, 0.0)
        if refund and author is not None:
            author.money += dep
            sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=pr["author"],
                                 kind="deposit", x=ax, y=ay,
                                 payload={"proposal_id": pr["id"],
                                          "amount": round(dep, 1), "phase": "refund",
                                          "balance": round(author.money, 1)}))
            return
        gov = getattr(sim, "government", None)
        if gov is not None and getattr(gov, "enabled", True):
            try:
                gov.collect("ward", dep)               # 没収=区の歳入(government ON 時)
            except Exception:
                pass
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=pr["author"],
                             kind="deposit", x=ax, y=ay,
                             payload={"proposal_id": pr["id"],
                                      "amount": round(dep, 1), "phase": "forfeit"}))

    def _pass_proposal(self, sim, pr, step, sim_min) -> None:
        """提案の成立処理(署名モード・投票可決の共通経路)。既存の proposal_passed 系を一切変えない。"""
        pr["passed"] = True
        pid = pr["id"]
        author = sim.agent_by_id.get(pr["author"])
        ax, ay = (author.x, author.y) if author else (0.0, 0.0)
        sim.logger.log(Event(step=step, sim_min=sim_min,
                             agent_id=pr["author"], kind="proposal_passed",
                             x=ax, y=ay,
                             payload={"proposal_id": pid, "text": pr["text"],
                                      "supporters": len(pr["supporters"])}))
        sim.net.publish_news("住民提案が受理", pr["text"], [pr["text"]], step)
        rec = getattr(sim, "recursion", None)         # 再帰性: 客観カウント(OFF は no-op)
        if rec is not None:
            rec.note("passed")
        if author is not None:                         # 制度を作った当事者性(既存則)
            factor_update.on_own_adopted(author, mags=sim.mags, step=step,
                                         sim_min=sim_min, logger=sim.logger)
        # 行動心理プラグイン(既定 OFF): Searle 制度化 / 集団効力感(所属グループ)
        self._on_proposal_passed(sim, pr, author, step, sim_min)
        # 制度DSL(既定 ON): 機械可読 rule を実効ルールとして自動制定(searle と共存)
        self._enact_rule(sim, pr, author, step, sim_min)

    # ---------------------------------------------------------------- 民主ルート: 投票(Wave G3)
    def _vote_yes(self, agent, pr, vcfg) -> bool:
        """露出した agent の yes/no を決定論で決める(LLM を呼ばない=R1)。

        grievance が閾値以上 or 提案語の感情価に opinion が整合していれば yes。traits は見ない。"""
        if float(agent.states.get("grievance", 0.0)) >= vcfg["grievance_threshold"]:
            return True
        val = valence(pr["text"])
        return bool(val != 0.0 and agent.opinion * val > 0.0)

    def _resolve_vote(self, sim, pr, step, sim_min, vcfg) -> None:
        """露出者(supporters)で決定論投票を集計し、過半数 yes なら可決(vote_cast/vote_result)。

        代表制議会(制度深化3・既定 OFF): 議会がある間は採決を露出者でなく議員が行う
        (有権者→代表→議決の間接化。議会の意見構成しだいで直接投票と結果が変わり得る=
        代表制の歪み。payload に council を付けて観測層で区別できるようにする)。"""
        council = getattr(sim, "council", None)
        by_council = bool(routes_of(sim)["assembly"]["enabled"] and council
                          and council.get("members"))
        electorate = list(council["members"]) if by_council \
            else sorted(pr["supporters"])              # id 昇順=決定論(議会は得票順のまま)
        yes = no = 0
        for aid in electorate:
            agent = sim.agent_by_id.get(aid)
            if agent is None:
                continue
            v = self._vote_yes(agent, pr, vcfg)
            yes += 1 if v else 0
            no += 0 if v else 1
            payload = {"proposal_id": pr["id"], "vote": "yes" if v else "no"}
            if by_council:
                payload["council"] = True
            sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=aid,
                                 kind="vote_cast", x=agent.x, y=agent.y,
                                 payload=payload))
        total = yes + no
        passed = bool(total > 0 and (yes / total) > vcfg["majority"])
        author = sim.agent_by_id.get(pr["author"])
        ax, ay = (author.x, author.y) if author else (0.0, 0.0)
        res = {"proposal_id": pr["id"], "yes": yes, "no": no, "passed": passed}
        if by_council:
            res["by"] = "council"
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=pr["author"],
                             kind="vote_result", x=ax, y=ay, payload=res))
        # (b) 条例議決の明示化(議会現実化 realism ON): 議会採決の終端を ordinance_vote としても記録する
        # (既存 vote_result は不変=追加専用)。OFF は一切足さない=バイト一致。
        asm = routes_of(sim)["assembly"]
        if asm["enabled"] and asm["realism"]["enabled"]:
            op = {"proposal_id": pr["id"], "passed": passed, "yes": yes, "no": no}
            if by_council:
                op["by"] = "council"
            sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=pr["author"],
                                 kind="ordinance_vote", x=ax, y=ay, payload=op))
        # 供託金の清算(制度深化 第9バッチ・既定 0=不変): 可決 or 得票率 >= refund_share で返還、
        # 未満は没収(公選法の没収ラインのモデル。政治参入の実費リスク)。
        share = (yes / total) if total else 0.0
        self._settle_deposit(sim, pr, step, sim_min,
                             refund=bool(passed or share >= vcfg["refund_share"]))
        if passed:
            self._pass_proposal(sim, pr, step, sim_min)

    # ---------------------------------------------------------------- 制度DSL(既定 ON)
    def _enact_rule(self, sim, pr, author, step, sim_min) -> None:
        """成立提案に機械可読 rule があれば実効ルールとして制定する(壊れない=無効は文言制度)。

        成立した瞬間、エンジンが実際に効く世界ルール(価格・支給・行き先抑制・定期イベント)
        として自動制定する。検証に失敗した rule は従来どおり「文言だけの制度」として成立。
        """
        rb = getattr(sim, "rulebook", None)
        rule_raw = pr.get("rule")
        if rb is None or rule_raw is None:
            return
        # 再帰性(第9バッチ・既定 OFF): repeal 提案の成立=既存ルールの廃止。OFF 時は repeal 型が
        # validate で弾かれ従来どおり文言制度に降格する(=現行挙動と完全同一)。
        recur = getattr(sim, "recursion", None)
        if (recur is not None and recur.cfg["enabled"] and recur.cfg["allow_repeal"]
                and isinstance(rule_raw, dict)
                and str(rule_raw.get("type", "")) == "repeal"):
            self._repeal_rule(sim, pr, author, step, sim_min, rb, rule_raw)
            return
        day = sim_min // 1440
        rec = rb.enact(rule_raw, name=pr["text"][:24], proposer=pr["author"],
                       step=step, day=day)
        if rec is None:                        # 検証失敗 → 文言制度のまま(降格)
            return
        if recur is not None:                  # 再帰性: 客観カウント(OFF は no-op)
            recur.note("enacted")
        ax, ay = (author.x, author.y) if author is not None else (0.0, 0.0)
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=pr["author"],
                             kind="institution_rule", x=ax, y=ay,
                             payload={"rule_id": rec["id"], "type": rec["type"],
                                      "name": rec["name"], "proposer": rec["proposer"],
                                      "rule": rb.public_rule(rec)}))
        sim.net.publish_news("新制度", f"新しい取り決めが定まった:{pr['text'][:24]}",
                             [], step)
        for ev in rb.evict_over_limit():       # 同時上限超過 → 古い順に失効
            sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=ev["proposer"],
                                 kind="rule_expired", x=0.0, y=0.0,
                                 payload={"rule_id": ev["id"], "type": ev["type"],
                                          "reason": "max_active"}))

    # ---------------------------------------------------------------- 再帰性: 廃止(第9バッチ・既定 OFF)
    def _repeal_rule(self, sim, pr, author, step, sim_min, rb, rule_raw) -> None:
        """成立した repeal 提案で既存ルールを廃止する(監視→提案→廃止の再帰が一周)。

        対象は rule_raw の target_rule_id(id 完全一致)or target(名前の部分一致)で解決。
        見つからなければ従来どおり文言制度として成立するだけ(壊れない)。決定論・非LLM。"""
        target = rb.find_active(rule_id=rule_raw.get("target_rule_id"),
                                name=rule_raw.get("target"))
        if target is None:
            return
        rb.repeal(target)
        recur = getattr(sim, "recursion", None)
        if recur is not None:
            recur.note("repealed")
        ax, ay = (author.x, author.y) if author is not None else (0.0, 0.0)
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=pr["author"],
                             kind="rule_repealed", x=ax, y=ay,
                             payload={"rule_id": target["id"], "type": target["type"],
                                      "name": target["name"],
                                      "proposer": target["proposer"],
                                      "repealed_by": pr["author"]}))
        sim.net.publish_news("取り決めの廃止",
                             f"「{target['name']}」が住民提案により廃止された", [], step)

    # ---------------------------------------------------------------- 行動心理プラグイン(既定 OFF)
    def _on_proposal_passed(self, sim, pr, author, step, sim_min) -> None:
        """提案成立時の心理プラグイン処理。OFF 時は何もしない=イベント列に一切足さない。"""
        pc = getattr(sim, "psychcfg", None)
        if not pc:
            return
        # Searle: 成立提案を制度(status function)として登録し、以後全員の発火プロンプトに注入。
        if pc["searle"]["enabled"]:
            text = pr["text"]
            inst = {"name": text[:20], "norm_text": text,
                    "created_by": pr["author"], "step": step}
            self.institutions.append(inst)
            ax, ay = (author.x, author.y) if author is not None else (0.0, 0.0)
            sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=pr["author"],
                                 kind="institution", x=ax, y=ay,
                                 payload={"name": inst["name"], "norm_text": text,
                                          "created_by": pr["author"]}))
            # 発起人ロールの効果=既存 on_own_adopted を再利用済み(直前で1回)。二重には呼ばない。
        # 集団効力感: 提案者の所属グループのメンバーへ集団的成功(SIMCA/Bandura)。
        if pc["collective"]["enabled"] and author is not None:
            self._group_success(sim, author.id, pc["collective"], step, sim_min)

    def _group_success(self, sim, agent_id, ccfg, step, sim_min) -> None:
        """agent が属する各グループの全メンバーへ集団的成功を配る(id 昇順=決定論)。"""
        mag = ccfg["group_success_magnitude"]
        for gid in sorted(self.member_of.get(agent_id, ())):
            g = self.groups.get(gid)
            if not g:
                continue
            for mid in sorted(g["members"]):
                m = sim.agent_by_id.get(mid)
                if m is not None:
                    factor_update.on_group_success(m, mag, step=step,
                                                   sim_min=sim_min, logger=sim.logger)

    def _collective_event_success(self, sim, ev, step, sim_min) -> None:
        """イベント主催者のグループから参加者が閾値以上いれば、そのメンバーへ集団的成功。"""
        pc = getattr(sim, "psychcfg", None)
        if not pc or not pc["collective"]["enabled"]:
            return
        ccfg = pc["collective"]
        thr = ccfg["attend_threshold"]
        mag = ccfg["group_success_magnitude"]
        attendees = ev["attendees"]
        for gid in sorted(self.member_of.get(ev["host"], set())):
            g = self.groups.get(gid)
            if not g:
                continue
            member_attendees = sorted(g["members"] & attendees)
            if len(member_attendees) >= thr:
                for mid in member_attendees:
                    m = sim.agent_by_id.get(mid)
                    if m is not None:
                        factor_update.on_group_success(m, mag, step=step,
                                                       sim_min=sim_min,
                                                       logger=sim.logger)

    # ---------------------------------------------------------------- 到着
    def on_arrive(self, sim, agent, step: int, sim_min: int) -> None:
        """ノード到着時: イベント参加の確定 + ビラ閲覧 + 屋台での購入。"""
        if not self.cfg["enabled"]:
            return
        eid = getattr(agent, "_attending_event", None)
        if eid is not None:
            ev = self.events.get(eid)
            if ev is not None and not ev["ended"] and agent.node == ev["node"]:
                self._record_attend(sim, ev, agent, step, sim_min)
            else:
                agent._attending_event = None
        self._view_flyers(sim, agent, step, sim_min)
        self._buy_at_ventures(sim, agent, step, sim_min)

    def _view_flyers(self, sim, agent, step, sim_min) -> None:
        from .engine import scheduler
        for f in self.flyers_by_node.get(agent.node, []):
            if agent.id == f["author"] or agent.id in f["viewers"]:
                continue
            f["viewers"].add(agent.id)
            self._log(sim, step, sim_min, agent, "flyer_view",
                      {"author": f["author"], "node": agent.node})
            scheduler._hear_words(sim, agent, f["items"], f["author"], "flyer",
                                  step, sim_min)
            d_v = factor_update.on_heard_valence(
                agent, valence(f["text"]), mags=sim.mags, step=step,
                sim_min=sim_min, logger=sim.logger, scale=0.8)
            if d_v:
                drive.add(agent, "state_change", sim.drivecfg, scale=abs(d_v))
            agent.remember(f"貼り紙を見た:「{f['text'][:20]}」")
            apply_bonus(getattr(sim, "rulebook", None), sim, agent, "flyer_view",
                        step, sim_min)                 # 制度DSL: bonus(flyer_view)

    def _buy_at_ventures(self, sim, agent, step, sim_min) -> None:
        from .engine import scheduler
        for v in self.ventures_by_node.get(agent.node, []):
            if step < int(v.get("open_at", 0)):        # 営業許可待ち(既定 0=常に営業中)
                continue
            if agent.id == v["owner"] or agent.money < v["price"]:
                continue
            rng = sim.hub.stream("venture", v["owner"], agent.id, step)
            # 社会的ヒエラルキー(引力): 店主の地位ほど購買が増える(OFF=倍率1=従来と完全同一)。
            mult = status_mod.buy_multiplier(sim, v["owner"])
            p = min(1.0, self.cfg["buy_prob"] * mult) if mult != 1.0 else self.cfg["buy_prob"]
            if rng.random() >= p:
                continue
            scheduler._spend(sim, agent, v["price"], "venture", step, sim_min)
            owner = sim.agent_by_id.get(v["owner"])
            if owner is not None:
                # 売上は口座へ入金(口座 E5 ON 時)。OFF 時は現金=従来と完全一致。
                if scheduler._accounts_on(sim):
                    owner.account += v["price"]
                    owner.period_income += v["price"]
                else:
                    owner.money += v["price"]
                v["last_sale_step"] = step
                v["sales_total"] = v.get("sales_total", 0.0) + v["price"]  # 起業転換(G5)の累計
                self._log_node(sim, step, sim_min, owner.id, v["node"],
                               "venture_sale",
                               {"amount": round(v["price"], 1),
                                "balance": round(owner.money, 1), "buyer": agent.id})
                owner.remember(f"「{v['name']}」が売れた")
            agent.remember(f"「{v['name']}」で買い物した")
            return                                     # 1 到着につき1 購入

    # ---------------------------------------------------------------- ログ補助
    def _log(self, sim, step, sim_min, agent, kind, payload) -> None:
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                             kind=kind, x=agent.x, y=agent.y, payload=payload))

    def _log_node(self, sim, step, sim_min, agent_id, node, kind, payload) -> None:
        x, y = sim.city.node_xy(node)
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent_id,
                             kind=kind, x=float(x), y=float(y), payload=payload))
