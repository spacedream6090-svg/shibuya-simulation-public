"""在場述語(``sim.is_present`` / ``sim.present_agent``)と、幽霊オブジェクト書き込みの一掃。

何を守るか(レーン甲 2026-08-13)
--------------------------------
``sim.agent_by_id`` は「これまで実体化した**全**個体」の索引で、プール回転で街を出た個体を
**意図的に消さない**(造語の作者名・DM 送信者名・関係台帳など**過去の参照**が退場後も解決
できるようにするため。``engine/scheduler.py`` の該当コメントが正典)。したがって

    sim.agent_by_id.get(aid) is None      # ← **決して真にならない**

であり、「街から居なくなった」の判定にこれを使っていた分岐は全て死んでいた。死んだ分岐の
裏側では**脱水済み(detached)のオブジェクトへ書き込んで**おり、書いた値は次の hydrate
(退場時スナップショット)で捨てられる。金を動かす経路ではこれが**保存則の破れ**になる:

  - 遺失物(C1): 返還・報労金・時効取得を幽霊へ渡すと ``hold`` からだけ引かれて現金が消える。
  - 相続(C2): 不在の相続人へ着金を書くと、故人の残高だけが 0 になって現金が消える。
  - 屋台の売上 / 無許可出店の罰金: 買い手・区が動いた分だけ現金が湧く / 消える。

本ファイルは (0) 述語そのもの、(1) C1/C2 の保存、(2) D3 の二重入居防止、(3) E2/E3 の
死者・域外者除外 を機械固定する。**退場の再現は本物と同じ形**で行う: ``sim.agents`` から
外し(= プール回転の ``sim.agents = kept``)、``agent_by_id`` には残す(= 名簿は消さない)。

R1: 既定プロファイル(pool OFF・health OFF)では在場 = 名簿なので、ここで直す分岐はどれも
到達不能 = ゴールデン L1 はバイト一致(``tests/test_scenario.py`` が固定)。
"""
from __future__ import annotations

import pytest

from society import assets as A
from society import economy_sfc as SFC
from society import freedom_p2 as F
from society import lost_property as L
from society.config import load_config
from society.engine import scheduler
from society.engine.simulation import Simulation
from society.world.perception import build_index


# --------------------------------------------------------------------------- #
# 共通ヘルパ
# --------------------------------------------------------------------------- #
def _sim(tmp_path, name, n_steps=1, n_agents=12, **ov):
    dot = ["run.seed=42", f"run.n_agents={n_agents}", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _depart(sim, agent) -> None:
    """プール回転の退場を**本物と同じ形**で再現する。

    ``_phase_pool_rotation`` は ``sim.agents`` を present だけの列に差し替え、``agent_by_id``
    からは**消さない**(過去の参照の解決先として残す)。退場者の実体はここから先「幽霊」=
    書き込んでも次の hydrate で捨てられる。
    """
    sim.agents = [a for a in sim.agents if a.id != agent.id]
    sim.invalidate_present_index()
    assert sim.agent_by_id.get(agent.id) is agent, "名簿から消してしまっている(前提が違う)"


def _kind(sim, kind):
    return [e for e in sim.logger.events if e.kind == kind]


def _cash(sim) -> float:
    """街の総現金(在場 + 退場者の幽霊も含む = 「どこかに在るはず」の全額)。"""
    return sum(float(getattr(a, "money", 0.0) or 0.0)
               + float(getattr(a, "account", 0.0) or 0.0)
               for a in sim.agent_by_id.values())


# =========================================================================== #
# (0) 述語そのもの
# =========================================================================== #
def test_agent_by_id_never_reports_absence(tmp_path):
    """★前提の固定: ``agent_by_id.get`` は退場者も返す(= None 判定は死んだ分岐)。"""
    sim = _sim(tmp_path, "pp_roster")
    gone = sim.agents[0]
    _depart(sim, gone)
    assert sim.agent_by_id.get(gone.id) is gone, \
        "agent_by_id が退場者を落とした(過去の参照が解決できなくなる)"


def test_present_agent_distinguishes_departed_from_roster(tmp_path):
    """在場述語は「名簿に居る」と「街に居る」を分ける。"""
    sim = _sim(tmp_path, "pp_pred")
    stay, gone = sim.agents[0], sim.agents[1]
    _depart(sim, gone)
    assert sim.is_present(stay.id) and sim.present_agent(stay.id) is stay
    assert not sim.is_present(gone.id), "退場者が在場と報告された"
    assert sim.present_agent(gone.id) is None, "退場者の幽霊が返った(書き込み先になる)"
    assert sim.present_agent(-1) is None and sim.present_agent(10 ** 9) is None


def test_present_index_is_rebuilt_when_the_roster_swaps_at_equal_size(tmp_path):
    """★人数が変わらない入替(enters == exits)でも索引が古びない = 明示の無効化が効く。"""
    sim = _sim(tmp_path, "pp_swap")
    a, b = sim.agents[0], sim.agents[1]
    assert sim.is_present(a.id) and sim.is_present(b.id)
    sim.agents = [x for x in sim.agents if x.id != a.id]      # 1 人出て
    sim.agents.append(sim.agent_by_id[a.id])                  # …と思わせて同数のまま
    sim.agents = [x for x in sim.agents if x.id != b.id]      # b を落として a を戻す(同数)
    sim.invalidate_present_index()
    assert sim.is_present(a.id) and not sim.is_present(b.id)


def test_present_index_recovers_without_an_explicit_invalidation(tmp_path):
    """要素数が変われば明示の無効化が無くても作り直す(checkpoint.load 本体を触らないための帯)。"""
    sim = _sim(tmp_path, "pp_lazy")
    gone = sim.agents[0]
    assert sim.is_present(gone.id)                            # 索引をここで作らせる
    sim.agents = [a for a in sim.agents if a.id != gone.id]   # invalidate を呼ばない
    assert not sim.is_present(gone.id), "要素数の変化を拾えていない"


def test_medical_lookup_uses_the_present_predicate(tmp_path):
    """``medical._agent_by_id``(線形走査の正典)が同じ答えを返す = 挙動同一・O(1) 化。"""
    from society import medical as M
    sim = _sim(tmp_path, "pp_med")
    stay, gone = sim.agents[0], sim.agents[1]
    _depart(sim, gone)
    assert M._agent_by_id(sim, stay.id) is stay
    assert M._agent_by_id(sim, gone.id) is None
    assert M._agent_by_id(sim, -1) is None


# =========================================================================== #
# (1) C1 遺失物: 街を出た当事者に金を渡さない(保存則)
# =========================================================================== #
def _turned_in(sim, cfg, st, owner, finder, cash=30_000.0):
    """財布を落として交番へ届いた状態(TURNED_IN)を作る。戻り: (rec, 中の現金)。"""
    owner.money = cash
    L._drop(sim, cfg, st, owner, L.WALLET, 0, 420, 1, False, False, False)
    rec = list(st["items"].values())[0]
    rec["status"], rec["finder"], rec["turn_step"] = L.TURNED_IN, int(finder.id), 1
    rec["noticed"], rec["reported"] = True, True
    return rec, float(rec["cash"])


def test_return_is_held_back_while_the_owner_is_away(tmp_path):
    """持ち主が街に居ない間は返還しない(物は交番に留まる・金は 1 円も動かない)。"""
    sim = _sim(tmp_path, "pp_lp_owner", **{"lost_property.enabled": "true",
                                           "lost_property.statute_steps": "9999"})
    cfg, st = L.cfg_of(sim), L._state(sim)
    owner, finder = sim.agents[0], sim.agents[1]
    finder.money = 0.0
    rec, cash = _turned_in(sim, cfg, st, owner, finder)
    total0 = _cash(sim) + float(st["hold"]) + float(st["lapsed"])
    _depart(sim, owner)
    sim.percept_index = build_index(sim.agents, 1.0)
    L._phase_resolve(sim, cfg, st, 20, 620, 8)
    assert rec["status"] == L.TURNED_IN, "持ち主不在なのに決着した"
    assert not _kind(sim, "lost_return"), "幽霊へ返還してしまっている"
    assert float(finder.money) == 0.0, "不在の持ち主から報労金が湧いた"
    assert _cash(sim) + float(st["hold"]) + float(st["lapsed"]) == pytest.approx(
        total0, abs=1e-6), "貨幣保存が破れた"
    assert float(st["hold"]) == pytest.approx(cash, abs=1e-6)


def test_statute_lapses_instead_of_paying_a_departed_finder(tmp_path):
    """時効の瞬間に拾得者が街に居なければ**失効**へ倒す(遺失物法 36 条)= 現金は消えない。"""
    sim = _sim(tmp_path, "pp_lp_finder", **{"lost_property.enabled": "true",
                                            "lost_property.statute_steps": "5"})
    cfg, st = L.cfg_of(sim), L._state(sim)
    owner, finder = sim.agents[0], sim.agents[1]
    finder.money = 0.0
    rec, cash = _turned_in(sim, cfg, st, owner, finder)
    total0 = _cash(sim) + float(st["hold"]) + float(st["lapsed"])
    _depart(sim, finder)
    sim.percept_index = build_index(sim.agents, 1.0)
    L._phase_resolve(sim, cfg, st, 20, 620, 8)
    ev = _kind(sim, "lost_expire")[-1]
    assert ev.payload["to"] == "void", "退場した拾得者に時効取得させている(幽霊への着金)"
    assert float(finder.money) == 0.0, "幽霊の財布に書いた(hydrate で捨てられる額)"
    assert float(st["lapsed"]) == pytest.approx(cash, abs=1e-6)
    assert _cash(sim) + float(st["hold"]) + float(st["lapsed"]) == pytest.approx(
        total0, abs=1e-6), "貨幣保存が破れた"


def test_notice_is_not_written_to_a_departed_owner(tmp_path):
    """気づき(遺失届)も在場者だけ = 幽霊の記憶に 1 行も書かない。"""
    sim = _sim(tmp_path, "pp_lp_notice", **{"lost_property.enabled": "true"})
    cfg, st = L.cfg_of(sim), L._state(sim)
    owner = sim.agents[0]
    owner.money = 20_000.0
    L._drop(sim, cfg, st, owner, L.WALLET, 0, 420, 1, False, False, False)
    rec = list(st["items"].values())[0]
    rec["step"] = -999                             # notice_delay を必ず超えさせる
    _depart(sim, owner)
    n_mem = len(owner.mem.episodes)
    L._phase_notice(sim, cfg, st, 20, 620, 8)
    assert not _kind(sim, "lost_notice"), "退場者が遺失届を出した"
    assert len(owner.mem.episodes) == n_mem, "幽霊の記憶へ書き込んだ"


# =========================================================================== #
# (2) C2 相続: 不在の相続人へ着金しない(保存則)
# =========================================================================== #
#: ``tests/test_assets.py`` の ON_HH と同じ(登記簿 + 組織 + 世帯)。
_HH = {"world.assets.enabled": "true", "organizations.enabled": "true",
       "household.enabled": "true"}


def _household_victim(sim):
    for a in sim.agents:
        if getattr(a, "housemates", None):
            return a
    return None


def _armed(tmp_path, name, n_agents=24, **ov):
    """``arm`` だけを走らせた登記簿(``tests/test_assets.py::_armed`` と同型)。"""
    sim = _sim(tmp_path, name, n_agents=n_agents, **dict(_HH, **ov))
    scheduler._ensure_orgs(sim)
    A.arm(sim, 0, 0)
    return sim


def _die(sim, agent, step=1, sim_min=10):
    """故人を作る(**health を通さない**= 相続の写像だけを隔離して見る)。"""
    from society.observer.schema import Event
    agent.dead = True
    sim.logger.log(Event(step=int(step), sim_min=int(sim_min), agent_id=int(agent.id),
                         kind="death", x=float(agent.x), y=float(agent.y),
                         payload={"cause": "cardiac"}))


def test_absent_heir_is_counted_but_never_credited(tmp_path):
    """★街に居ない世帯員には 1 円も書かない。人数は ``absent`` に載り、金は国庫へ出る。"""
    sim = _armed(tmp_path, "pp_heir")
    victim = _household_victim(sim)
    assert victim is not None, "テスト前提が崩れた(世帯が 1 つも組まれていない)"
    mates = [sim.agent_by_id[i] for i in victim.housemates if i != victim.id]
    assert mates, "テスト前提が崩れた(同居人が居ない)"
    for m in mates:                                # 全員を街から出す = 受け皿が居ない
        _depart(sim, m)
    victim.money = 12_345.0
    mate_before = {m.id: float(m.money) for m in mates}
    total0 = _cash(sim)
    _die(sim, victim)
    A.phase(sim, 1, 10)
    ev = _kind(sim, "inheritance")[-1]
    assert ev.payload["absent"] == len(mates), ev.payload
    assert ev.payload["heirs"] == 0 and ev.payload["to"] == "row"
    for m in mates:
        assert float(m.money) == mate_before[m.id], "幽霊の財布に着金した(hydrate で消える額)"
    assert A.cash_escheated(sim) == pytest.approx(12_345.0, abs=1e-6)
    assert _cash(sim) + A.cash_escheated(sim) == pytest.approx(total0, abs=1e-6), \
        "相続で現金が消えた/湧いた"


def test_present_heirs_still_inherit_when_one_mate_is_away(tmp_path):
    """在場の相続人が 1 人でも居れば、その人たちだけで等分する(総額は不変)。"""
    sim = _armed(tmp_path, "pp_heir_mix", n_agents=40)
    victim = None
    for a in sim.agents:
        mates = [i for i in (getattr(a, "housemates", None) or ()) if i != a.id]
        if len(mates) >= 2:
            victim = a
            break
    if victim is None:
        pytest.skip("2 人以上の同居人を持つ世帯がこの seed では組まれない")
    mates = [sim.agent_by_id[i] for i in victim.housemates if i != victim.id]
    gone, stays = mates[0], mates[1:]
    _depart(sim, gone)
    gone_before = float(gone.money)
    stay_before = sum(float(m.money) for m in stays)
    victim.money = 9_000.0
    total0 = _cash(sim)
    _die(sim, victim)
    A.phase(sim, 1, 10)
    ev = _kind(sim, "inheritance")[-1]
    assert ev.payload["absent"] == 1 and ev.payload["heirs"] == len(stays)
    assert float(gone.money) == gone_before, "退場中の同居人に着金した"
    assert sum(float(m.money) for m in stays) == pytest.approx(
        stay_before + 9_000.0, abs=1e-6)
    assert _cash(sim) == pytest.approx(total0, abs=1e-6), "相続で現金が消えた/湧いた"


def test_a_departed_decedent_is_not_probated(tmp_path):
    """故人自身が街に居ない回は相続を回さない(遺産は退避側に凍る = 湧かない)。"""
    sim = _armed(tmp_path, "pp_dead_gone")
    victim = _household_victim(sim)
    assert victim is not None
    victim.money = 5_000.0
    total0 = _cash(sim)
    _die(sim, victim)
    _depart(sim, victim)                           # 死亡イベントの処理前に街を出た
    A.phase(sim, 1, 10)
    assert not _kind(sim, "inheritance"), "幽霊の遺産を配ってしまった(現金が湧く)"
    assert float(victim.money) == 5_000.0
    assert _cash(sim) == pytest.approx(total0, abs=1e-6)


# =========================================================================== #
# (3) D3 二重入居の防止
# =========================================================================== #
def test_pick_home_treats_a_departed_residents_home_as_occupied(tmp_path):
    """★脱水中の住人の家は「空き」ではない(在場だけを見ると二重入居が単調に増える)。

    候補を「退場者の家 1 棟だけ」に絞る = 旧実装なら必ずそれを掴み、新実装なら空きゼロ。
    分岐を運に任せない形にしてある。
    """
    sim = _sim(tmp_path, "pp_home", n_agents=12)
    mover, gone = sim.agents[0], sim.agents[1]
    home_id = getattr(gone, "home_building", "")
    assert home_id, "テスト前提が崩れた(home_building が無い)"
    only = [b for b in sim.city.residential_buildings if b["id"] == home_id]
    assert only, "テスト前提が崩れた(home_building が住宅系建物に無い)"
    sim.city.residential_buildings = only          # 候補は退場者の家 1 棟だけ
    _depart(sim, gone)
    # ★旧規約(在場だけを走査)ではこの家は「空き」に見える = 修正前は必ず掴んでいた。
    present_homes = {getattr(a, "home_building", "") for a in sim.agents}
    assert home_id not in present_homes, "検収の空回り(在場者の誰かが同じ家に住んでいる)"
    rng = sim.hub.stream("move_home", int(mover.id), 0)
    assert F.pick_home(sim, mover, None, rng) is None, \
        "退場中の住人の家を空きとして掴んだ(二重入居が単調に増える)"


def test_pick_home_still_finds_a_genuinely_empty_dwelling(tmp_path):
    """誰の home でもない住戸は変わらず選べる(修正が空き探索を殺していない)。"""
    sim = _sim(tmp_path, "pp_home_ok", n_agents=12)
    mover = sim.agents[0]
    homes = {getattr(a, "home_building", "") for a in sim.agent_by_id.values()}
    free = [b for b in sim.city.residential_buildings if b["id"] not in homes]
    if not free:
        pytest.skip("空き住戸が 1 つも無い地図構成")
    sim.city.residential_buildings = free[:3]
    rng = sim.hub.stream("move_home", int(mover.id), 0)
    picked = F.pick_home(sim, mover, None, rng)
    assert picked is not None and picked["id"] not in homes


# =========================================================================== #
# (3b) C7 VC 持分: 閉店で持分が落ちる(pop 系が無かった欄)
# =========================================================================== #
def test_vc_equity_is_released_when_the_venture_closes(tmp_path):
    """★屋台が閉じたら VC の持分も落ちる(残ると再審査から永久に外れ、原資が枯れる)。"""
    from society import commerce as C
    sim = _sim(tmp_path, "pp_vc", n_agents=8, **{"commerce.enabled": "true"})
    vccfg = (getattr(sim, "economy", {}) or {}).get("vc") or {}
    assert vccfg, "economy.vc の設定が無い(テスト前提が崩れた)"
    fund = C.VCFund(vccfg)
    sim.vc_fund = fund
    owner = sim.agents[0]
    fund.invest(owner.id)
    assert fund.equity.get(owner.id, 0.0) > 0.0
    tools = sim.tools
    tools.ventures[owner.id] = {"owner": owner.id, "name": "屋台", "node": owner.node,
                                "price": 100.0, "last_sale_step": 0, "opened_step": 0,
                                "permitted": True}
    tools.ventures_by_node[owner.node].append(tools.ventures[owner.id])
    assert tools.force_close_venture(sim, owner, 1, 10, reason="test")
    assert owner.id not in fund.equity, "閉店したのに VC 持分が残っている"
    assert fund.equity_written_off > 0.0, "失効した持分が損金計上されていない"


# =========================================================================== #
# (4) E2/E3 死者・域外者の除外(loc/dead ガードの規約統一)
# =========================================================================== #
_WORK_ON = {"work.service.enabled": "true"}


def _office_payload(sim, node):
    rows = [e for e in sim.logger.events
            if e.kind == "org_output" and e.payload["org"] == node]
    return rows[-1].payload if rows else None


def _staff_office(tmp_path, name, n=10):
    sim = _sim(tmp_path, name, n_agents=n, **_WORK_ON)
    node = sim.city.pois_by_cat("office")[0]["node"]
    for a in sim.agents:                           # 既定の配属を排除(当該ノードだけ数える)
        a.work_node = ""
        a.work_start_min = -1
    for a in sim.agents[:4]:
        a.work_node = node
        a.work_building = ""
        a.work_start_min, a.work_end_min = 540, 1080
    return sim, node


def _unguarded_n(sim, node) -> int:
    """loc/dead を**見なかった**ときの出勤者数(= 修正前の実装が数えていた人数)。"""
    return sum(1 for a in sim.agents
               if getattr(a, "work_node", "") == node
               and getattr(a, "work_start_min", -1) >= 0)


def test_org_output_excludes_the_dead(tmp_path):
    """★死者が会社の産出に永久計上されない(死者は sim.agents から消えない)。"""
    sim, node = _staff_office(tmp_path, "pp_e2_dead")
    sim.agents[0].dead = True
    assert _unguarded_n(sim, node) == 4, "検収の空回り(合成配置が壊れている)"
    sim._work_day = -1
    scheduler._work_office_output(sim, 0, 0, sim.workcfg)
    p = _office_payload(sim, node)
    assert p is not None and p["n"] == 3, f"死者が出勤者に数えられている: {p}"
    assert p["output"] == pytest.approx(3.0, abs=1e-6)


def test_org_output_excludes_people_outside_the_area(tmp_path):
    """域外(loc=outside)の在職者も出勤していない(commerce.occupancy と同じ規約)。"""
    sim, node = _staff_office(tmp_path, "pp_e2_out")
    sim.agents[0].loc = "outside"
    sim.agents[1].loc = "outside"
    assert _unguarded_n(sim, node) == 4, "検収の空回り(合成配置が壊れている)"
    sim._work_day = -1
    scheduler._work_office_output(sim, 0, 0, sim.workcfg)
    p = _office_payload(sim, node)
    assert p is not None and p["n"] == 2, f"域外者が出勤者に数えられている: {p}"


def test_serve_is_not_attributed_to_a_dead_staffer(tmp_path):
    """★接客も同じ規約: 死者は接客していない(node は死んだ場所のまま職場と一致しうる)。"""
    from society.observer.schema import Event
    sim = _sim(tmp_path, "pp_e3", n_agents=10, **dict(_WORK_ON,
                                                      **{"work.service.record_unstaffed": "true"}))
    node = sim.city.pois_by_cat("food")[0]["node"]
    x, y = sim.city.node_xy(node)
    for a in sim.agents:
        a.work_start_min = -1
    customer, staff = sim.agents[0], sim.agents[1]
    customer.node, customer.x, customer.y = node, x, y
    staff.node = staff.work_node = node
    staff.x, staff.y = x, y
    staff.work_start_min, staff.work_end_min = 0, 1440
    # ★旧規約では node == work_node と勤務窓だけを見ていた = 死者がそのまま接客していた。
    from society.cognition import routine
    assert staff.node == staff.work_node and routine.in_work_window(
        staff, 700, getattr(sim, "calendarcfg", None)), "検収の空回り(合成配置が壊れている)"
    staff.dead = True                              # 亡くなったスタッフ
    since = len(sim.logger.events)
    sim.logger.log(Event(step=5, sim_min=700, agent_id=customer.id, kind="spend",
                         x=x, y=y, payload={"amount": 900.0, "balance": 9100.0,
                                            "cat": "food"}))
    scheduler._phase_work_service(sim, 5, 700, since)
    serves = [e for e in sim.logger.events[since:] if e.kind == "serve"]
    assert serves, "検収の空回り(serve が 1 件も出ていない)"
    assert all(e.agent_id != staff.id for e in serves), \
        "死んだスタッフに接客が帰属した"
    assert all(e.agent_id == -1 for e in serves), "スタッフ不在の記録になっていない"


# =========================================================================== #
# (5) 保存則の総まとめ(IF-E の閉じた不変量)
# =========================================================================== #
def _closed_total(sim, st) -> float:
    """街の全現金(名簿ぜんぶ)+ 遺失物の在庫 + 失効累計 + 国庫 = 閉じた不変量。

    ★``economy_sfc.total_money`` は**在場者**を数えるので、退場そのもので値が動く
      (= 保存則の検査には使えない)。ここは「どこかに在るはずの全額」を数える。
    """
    return (_cash(sim) + float(st["hold"]) + float(st["lapsed"])
            + float(A.cash_escheated(sim) or 0.0))


def test_city_total_money_is_unchanged_by_every_ghost_path(tmp_path):
    """★遺失物 + 相続を同じランで回しても、閉じた不変量は 1 円も動かない。"""
    sim = _armed(tmp_path, "pp_conserve", n_agents=24,
                 **{"lost_property.enabled": "true",
                    "lost_property.statute_steps": "5",
                    "economy.org_accounting.enabled": "true"})
    cfg, st = L.cfg_of(sim), L._state(sim)
    owner, finder = sim.agents[0], sim.agents[1]
    _turned_in(sim, cfg, st, owner, finder, cash=25_000.0)   # 財布を落として交番へ
    before = _closed_total(sim, st)                          # ★ここが基準(hold に在庫あり)
    assert float(st["hold"]) > 0.0, "検収の空回り(中の現金がゼロ)"
    ghosts: dict = {}

    def _leave(a):
        _depart(sim, a)
        ghosts[int(a.id)] = (float(a.money),
                             float(getattr(a, "account", 0.0) or 0.0))

    _leave(finder)
    sim.percept_index = build_index(sim.agents, 1.0)
    L._phase_resolve(sim, cfg, st, 20, 620, 8)
    victim = _household_victim(sim)
    if victim is not None and victim.id not in (owner.id, finder.id):
        for i in list(victim.housemates):
            m = sim.present_agent(i)
            if m is not None and m.id != victim.id:
                _leave(m)
        _die(sim, victim)
        A.phase(sim, 21, 630)
        assert _kind(sim, "inheritance"), "検収の空回り(相続が 1 件も起きていない)"
    after = _closed_total(sim, st)
    assert after == pytest.approx(before, abs=1e-6), \
        f"閉じた不変量が破れた: {before} -> {after}"
    # ★閉じた不変量だけでは足りない: 幽霊の財布に書いた額はこの合計には**現れる**が、
    #   次の hydrate で捨てられて初めて消える。だから「退場者の残高が 1 円も動いていない」
    #   ことを別途固定する(これが本レーンの本質)。
    for aid in ghosts:
        assert float(sim.agent_by_id[aid].money) == pytest.approx(
            ghosts[aid][0], abs=1e-6), f"退場者 {aid} の現金が動いた(hydrate で消える額)"
        assert float(getattr(sim.agent_by_id[aid], "account", 0.0) or 0.0) == pytest.approx(
            ghosts[aid][1], abs=1e-6), f"退場者 {aid} の口座が動いた"
    assert SFC.total_money(sim) >= 0.0                # SFC 側は在場ベース = 参考値
