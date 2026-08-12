"""役割バインドの**日次追随**(第110 レーン PRES-C。``*.rebind_daily``・既定 OFF)のテスト。

何を守るのか
------------
``street_life.bind`` / ``city_ops.bind`` は**起動時 1 回**しか走らなかった。100万ペルソナ
プールの在場ローテーション(``_phase_pool_rotation``)が毎日回るランでは、途中入場した個体は
誰にも束ねられず、退場した担い手の枠は空いたまま残る = **担い手が日を追って痩せる**
(第109 の実測発見。10日ランで顕著)。本レーンはその追随を入れたので、以下を機械固定する:

  ① 既定 OFF = 従来どおり起動時 1 回きり(conf の出荷値・DEFAULTS・provenance の申告)
  ② ON でも**入替の無い世界では 1 バイトも変わらない**(L1 完全一致 = 冪等の証明)
  ③ 日境界の補充: 退場者が空けた**枠**を、当日の在場者が**同じ決定論規則**で埋める
     (持ち場ノード・当直窓・地区が、退場者が持っていたものと一致する)
  ④ **既に束ねた在場者は 1 バイトも触らない**(冪等)。とりわけ**出動中の救急隊**を
     日境界に待機拠点へ引き戻さない(これは最適化ではなく正しさ)
  ⑤ OFF のままだと途中入場者は永久に束ねられない(= ①の裏 = 追随が実効を持つ証拠)
  ⑥ provenance に**日ごとの担い手数**(``bind_by_day``)が残る
  ⑦ 実プール回転(小プール・mock)で担い手が痩せない
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

import build_persona_pool as bpp                        # noqa: E402
from society import city_ops as CO                      # noqa: E402
from society import registry as R                       # noqa: E402
from society import street_life as SL                   # noqa: E402
from society import transit_staff as TS                 # noqa: E402
from society.config import load_config                  # noqa: E402
from society.engine.simulation import Simulation        # noqa: E402

DAY = 1440

SL_ON = {"street_life.enabled": "true"}
SL_ON_RE = {"street_life.enabled": "true", "street_life.rebind_daily": "true"}
CO_ON = {"city_ops.enabled": "true"}
CO_ON_RE = {"city_ops.enabled": "true", "city_ops.rebind_daily": "true"}


# --------------------------------------------------------------------------- #
# 共通ヘルパ(test_street_life.py / test_city_ops.py と同型)
# --------------------------------------------------------------------------- #
def _cfg(name, n_steps=24, n_agents=20, **ov):
    dot = ["run.seed=42", f"run.n_agents={n_agents}", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock", "observer.snapshot_every=144"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def _sim(tmp_path, name, n_steps=24, n_agents=20, **ov):
    return Simulation(_cfg(name, n_steps, n_agents, **ov), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _ags(sim):
    return sorted(sim.agents, key=lambda a: int(a.id))


def _holders(sim, attr) -> int:
    """いま在場している個体のうち、その枠を持っている人数(担い手の実数)。"""
    return sum(1 for a in sim.agents if getattr(a, attr, None) is not None)


def _leave(sim, agent) -> None:
    """在場ローテーションの**退場**を手で作る(pool が sim.agents から外すのと同じ形)。"""
    sim.agents = [a for a in sim.agents if a is not agent]


# =========================================================================== #
# ① 既定 OFF(出荷値・宣言)
# =========================================================================== #
def test_shipped_default_is_off():
    """出荷 conf の既定は false(= 従来の起動時 1 回きり)。"""
    cfg = load_config([])
    assert cfg.street_life.rebind_daily is False
    assert cfg.city_ops.rebind_daily is False
    assert cfg.transit_staff.bind.rebind_daily is False
    assert SL.DEFAULTS["rebind_daily"] is False
    assert CO.DEFAULTS["rebind_daily"] is False
    assert TS.DEFAULTS["bind"]["rebind_daily"] is False


def test_registry_declares_both_toggles():
    """★未宣言トグル検出(第72)に引っかからないよう、全部をレジストリに宣言する。"""
    for fid in ("street_life.rebind_daily", "city_ops.rebind_daily",
                "transit_staff.bind.rebind_daily"):
        f = R.BY_ID.get(fid)
        assert f is not None, f"{fid} がレジストリに無い"
        assert f.repro_tier == "strict"          # 乱数ゼロ・LLM 呼数増分ゼロ
        assert f.affects_k is False
        assert f.fingerprint_risk == "none"      # プロンプトを 1 バイトも変えない


def test_cfg_canonicalization_accepts_the_key():
    """dict / 未指定のどちらでも既定へ落ちる(build_cfg の型強制)。"""
    assert SL.build_cfg({})["rebind_daily"] is False
    assert SL.build_cfg({"rebind_daily": True})["rebind_daily"] is True
    assert CO.build_cfg({})["rebind_daily"] is False
    assert CO.build_cfg({"rebind_daily": True})["rebind_daily"] is True
    assert TS.build_cfg({})["bind"]["rebind_daily"] is False
    assert TS.build_cfg({"bind": {"rebind_daily": True}})["bind"]["rebind_daily"] is True


def test_toggle_alone_is_a_noop_while_the_layer_is_off(tmp_path):
    """親が OFF のとき、追随トグルだけを立てても完全 no-op(純粋既定と L1 一致)。"""
    pure = _sim(tmp_path, "pure", n_steps=24)
    pure.run()
    on = _sim(tmp_path, "toggle_only", n_steps=24,
              **{"street_life.rebind_daily": "true", "city_ops.rebind_daily": "true",
                 "transit_staff.bind.rebind_daily": "true"})
    on.run()
    assert _l1(pure) == _l1(on), "親 OFF なのに追随トグルが世界を動かした"


# =========================================================================== #
# ② ON でも入替が無ければ 1 バイトも変わらない(冪等の証明)
# =========================================================================== #
TS_ON = {"transit_staff.enabled": "true"}
TS_ON_RE = {"transit_staff.enabled": "true",
            "transit_staff.bind.rebind_daily": "true"}


@pytest.mark.parametrize("base,rebind",
                         [(SL_ON, SL_ON_RE), (CO_ON, CO_ON_RE), (TS_ON, TS_ON_RE)])
def test_rebind_changes_nothing_without_rotation(tmp_path, base, rebind):
    """在場が入れ替わらない世界(pool OFF)では、追随 ON/OFF の L1 が完全一致する。

    = 日境界の呼び直しが「既に束ねた在場者を 1 バイトも触らない」ことの機械固定。
    """
    key = {id(SL_ON): "sl", id(CO_ON): "co", id(TS_ON): "ts"}[id(base)]
    off = _sim(tmp_path, f"norot_off_{key}", n_steps=300, **base)
    off.run()
    on = _sim(tmp_path, f"norot_on_{key}", n_steps=300, **rebind)
    on.run()
    assert _l1(off) == _l1(on), "入替の無い世界で追随がイベント列を変えた"


def test_bind_repeated_call_returns_the_same_counts(tmp_path):
    """``bind`` を何度呼んでも返り値(= いまの担い手数)は同じ = 冪等。"""
    sim = _sim(tmp_path, "idem_counts", n_steps=1, **SL_ON_RE)
    ags = _ags(sim)
    for i, a in enumerate(ags[:8]):
        a.occupation = SL.STREET_OCCS[i % len(SL.STREET_OCCS)]
    first = SL.bind(sim)
    rep: dict = {}
    second = SL.bind(sim, rep)
    assert first == second
    assert rep["n_new"] == 0, "2 回目に新しく束ねた人が居る(冪等でない)"

    sim2 = _sim(tmp_path, "idem_counts_co", n_steps=1, **CO_ON_RE)
    ags2 = _ags(sim2)
    for i, a in enumerate(ags2[:12]):
        a.occupation = (CO.CITY_OPS_OCCS + ("警察官",))[i % 5]
    f2 = CO.bind(sim2)
    rep2: dict = {}
    s2 = CO.bind(sim2, rep2)
    assert f2 == s2
    assert rep2["n_new"] == 0


# =========================================================================== #
# ③ 日境界の補充(退場者の枠を当日の在場者が同じ規則で埋める)
# =========================================================================== #
def test_street_life_refills_the_vacated_slot(tmp_path):
    """路上の役割: 退場した担い手の**枠と持ち場**を、当日の在場者がそのまま引き継ぐ。"""
    sim = _sim(tmp_path, "sl_refill", n_steps=1, n_agents=24, **SL_ON_RE)
    ags = _ags(sim)
    crew = ags[:4]
    for a in crew:
        a.occupation = SL.MUSICIAN
    SL.phase(sim, 0, 19 * 60)                              # day0 = 起動時の束ね
    slots = {int(a.id): (a.street_slot, a.street_post) for a in crew}
    assert sorted(s for s, _n in slots.values()) == [0, 1, 2, 3]

    gone = crew[1]                                         # 枠 1 が空く
    vacated = slots[int(gone.id)]
    _leave(sim, gone)
    newcomer = ags[12]                                     # 当日の在場者が担い手になる
    newcomer.occupation = SL.MUSICIAN
    assert not hasattr(newcomer, "street_slot")

    SL.phase(sim, 144, DAY + 19 * 60)                      # day1 = 日次追随
    assert (newcomer.street_slot, newcomer.street_post) == vacated, \
        "空いた枠(と持ち場)が引き継がれていない"
    for a in (crew[0], crew[2], crew[3]):                  # 残った担い手は 1 バイトも動かない
        assert (a.street_slot, a.street_post) == slots[int(a.id)]
    st = sim._sl_state
    assert st["bind_by_day"]["1"]["n_role"] == st["bind_by_day"]["0"]["n_role"], \
        "担い手が痩せた(補充されていない)"
    assert st["refilled"] == 1 and st["rebinds"] == 1


def test_city_ops_refills_the_vacated_police_slot(tmp_path):
    """交番配置: 退場した警察官の**枠(= 当直窓と交番)**を在場者が引き継ぐ。"""
    sim = _sim(tmp_path, "co_refill", n_steps=1, n_agents=24, **CO_ON_RE)
    ags = _ags(sim)
    crew = ags[:8]
    for a in crew:
        a.occupation = "警察官"
    CO.phase(sim, 0, 6 * 60)
    before = {int(a.id): (a.city_ops_slot_police, str(getattr(a, "city_ops_post", "")),
                          a.work_start_min, a.work_end_min) for a in crew}
    gone = crew[2]                                         # 持ち場つきの枠(2 % 4 != 3)
    vacated = before[int(gone.id)]
    assert vacated[1], "前提が崩れている(枠 2 は交番配置のはず)"
    _leave(sim, gone)
    newcomer = ags[15]
    newcomer.occupation = "警察官"

    CO.phase(sim, 144, DAY + 6 * 60)
    got = (newcomer.city_ops_slot_police, str(newcomer.city_ops_post),
           newcomer.work_start_min, newcomer.work_end_min)
    assert got == vacated, "空いた交番の枠(持ち場・当直窓)が引き継がれていない"
    for a in crew:
        if a is gone:
            continue
        assert (a.city_ops_slot_police, str(getattr(a, "city_ops_post", "")),
                a.work_start_min, a.work_end_min) == before[int(a.id)]
    st = sim._co_state
    assert st["bind_by_day"]["1"]["police"] == st["bind_by_day"]["0"]["police"]


def test_city_ops_refills_every_group(tmp_path):
    """5 群(交番/収集/納品/夜間清掃/救急)のどれも痩せない(担い手数が日を跨いで一定)。"""
    sim = _sim(tmp_path, "co_groups", n_steps=1, n_agents=40, **CO_ON_RE)
    ags = _ags(sim)
    roles = (CO.WASTE, CO.DRIVER, CO.NIGHT_CLEANER, CO.EMS_CREW, "警察官")
    for i, a in enumerate(ags[:20]):
        a.occupation = roles[i % len(roles)]
    CO.phase(sim, 0, 6 * 60)
    day0 = dict(sim._co_state["bind_by_day"]["0"])
    # 各群から 1 人ずつ退場させ、同数の在場者を同じ役割へ立たせる(= プール回転の縮図)
    for i, occ in enumerate(roles):
        gone = next(a for a in ags[:20] if str(a.occupation) == occ)
        _leave(sim, gone)
        ags[20 + i].occupation = occ
    CO.phase(sim, 144, DAY + 6 * 60)
    day1 = dict(sim._co_state["bind_by_day"]["1"])
    assert day1 == day0, f"担い手数が日を跨いで変化した: {day0} -> {day1}"


def test_touting_follows_the_nightlife_roster(tmp_path):
    """客引きは職業ではなく**夜間店舗の従業者の決定論部分集合**なので、日次に引き直す。"""
    sim = _sim(tmp_path, "tout_follow", n_steps=1, n_agents=24, **SL_ON_RE)
    night = SL.posts(sim, SL.P_NIGHT)
    if not night:
        pytest.skip("この地図には夜間店舗ノードが無い(客引きが原理的に成立しない)")
    ags = _ags(sim)
    every = sim.streetlifecfg["tout_every"]
    picked = [a for a in ags if int(a.id) % every == 0][:3]
    assert len(picked) >= 2
    picked[0].work_node = str(night[0])
    SL.phase(sim, 0, 20 * 60)
    assert getattr(picked[0], "street_tout", False)
    n0 = sim._sl_state["bind_by_day"]["0"]["n_tout"]

    _leave(sim, picked[0])                                 # 退場
    picked[1].work_node = str(night[0])                    # 途中入場相当(夜の店に職場を得た)
    SL.phase(sim, 144, DAY + 20 * 60)
    assert getattr(picked[1], "street_tout", False), "新しい従業者が客引きに立っていない"
    assert sim._sl_state["bind_by_day"]["1"]["n_tout"] == n0


def test_transit_staff_follows_the_roster_daily(tmp_path):
    """駅員・乗務員(L5 duty 層)も日境界で追随する。既定 OFF は従来どおり束ねない。"""
    ts_on = {"transit_staff.enabled": "true"}
    ts_re = {"transit_staff.enabled": "true",
             "transit_staff.bind.rebind_daily": "true"}
    sim = _sim(tmp_path, "ts_rebind", n_steps=1, n_agents=24, **ts_re)
    ags = _ags(sim)
    ags[0].occupation = "車掌"
    TS.phase(sim, 0, 6 * 60)
    station = str(sim.city.station_node)
    assert str(ags[0].work_node) == station
    ags[6].occupation = "車掌"                             # 途中入場相当
    ags[6].work_node = "N_elsewhere"
    TS.phase(sim, 144, DAY + 6 * 60)
    assert str(ags[6].work_node) == station, "途中入場した乗務員が駅へ立っていない"
    assert sim._transit_staff_rebinds == 1

    off = _sim(tmp_path, "ts_no_rebind", n_steps=1, n_agents=24, **ts_on)
    oags = _ags(off)
    oags[0].occupation = "車掌"
    TS.phase(off, 0, 6 * 60)
    oags[6].occupation = "車掌"
    oags[6].work_node = "N_elsewhere"
    TS.phase(off, 144, DAY + 6 * 60)
    assert str(oags[6].work_node) == "N_elsewhere", "追随 OFF なのに束ねられた"


def test_transit_staff_rebind_keeps_placed_crew(tmp_path):
    """既に駅へ立っている個体は日境界の追随でも 1 バイトも動かない(冪等)。"""
    sim = _sim(tmp_path, "ts_keep", n_steps=1, n_agents=24,
               **{"transit_staff.enabled": "true",
                  "transit_staff.bind.rebind_daily": "true"})
    ags = _ags(sim)
    for a in ags[:4]:
        a.occupation = "駅員"
    TS.phase(sim, 0, 6 * 60)
    snap = [(a.id, a.work_node, a.work_start_min, a.work_end_min) for a in ags[:4]]
    TS.phase(sim, 144, DAY + 6 * 60)
    assert snap == [(a.id, a.work_node, a.work_start_min, a.work_end_min)
                    for a in ags[:4]]


# =========================================================================== #
# ④ 既に束ねた在場者を触らない(出動中の救急隊を引き戻さない)
# =========================================================================== #
def test_dispatched_crew_is_not_pulled_back_at_the_day_boundary(tmp_path):
    """★出動中(``work_node`` が現場)の救急隊を、日境界の追随が待機拠点へ戻さない。"""
    sim = _sim(tmp_path, "co_ems_hold", n_steps=1, n_agents=24, **CO_ON_RE)
    ags = _ags(sim)
    for a in ags[:4]:
        a.occupation = CO.EMS_CREW
    CO.phase(sim, 0, 6 * 60)
    crew = ags[0]
    base = str(crew.city_ops_ems_base)
    scene = next(n for n in sorted(sim.city.graph.nodes) if str(n) != base)
    crew.city_ops_ems_home = base                          # request_ems と同じ印を手で置く
    crew.city_ops_ems_until = 10_000
    crew.work_node = str(scene)

    CO.phase(sim, 144, DAY + 6 * 60)
    assert str(crew.work_node) == str(scene), "出動中の隊を日境界の追随が引き戻した"


def test_sheltered_rough_sleeper_is_not_put_back_on_the_street(tmp_path):
    """★尊厳規約: 支援で住居へ移行した個体を、日次追随が路上へ戻さない(枠を保つ)。"""
    sim = _sim(tmp_path, "sl_shelter_hold", n_steps=1, n_agents=24, **SL_ON_RE)
    ags = _ags(sim)
    for a in ags[:3]:
        a.occupation = SL.ROUGH
    SL.phase(sim, 0, 19 * 60)
    moved = ags[0]
    moved.home_building = "B_test"                         # shelter_move と同じ形
    moved.home_node = "N_test"
    moved.street_sleep_node = ""

    SL.phase(sim, 144, DAY + 19 * 60)
    assert moved.home_building == "B_test" and moved.street_sleep_node == "", \
        "住居へ移行した個体が路上の夜の居場所へ戻された"


def test_rough_sleeper_exclusion_set_is_refreshed_on_rotation(tmp_path):
    """★尊厳規約 2: 途中入場した路上生活者も犯罪/迷惑機構の除外集合に入る(キャッシュ更新)。"""
    sim = _sim(tmp_path, "sl_dignity", n_steps=1, n_agents=24, **SL_ON_RE)
    ags = _ags(sim)
    ags[0].occupation = SL.ROUGH
    SL.phase(sim, 0, 19 * 60)
    assert SL.rough_sleeper_ids(sim) == frozenset({int(ags[0].id)})
    ags[5].occupation = SL.ROUGH                           # 途中入場相当
    SL.phase(sim, 144, DAY + 19 * 60)
    assert int(ags[5].id) in SL.rough_sleeper_ids(sim), \
        "途中入場した路上生活者が除外集合に入っていない(キャッシュが古い)"


# =========================================================================== #
# ⑤ OFF のままなら途中入場者は永久に束ねられない(追随が実効を持つ証拠)
# =========================================================================== #
def test_without_rebind_the_newcomer_stays_unbound(tmp_path):
    """既定(追随 OFF)は従来どおり: 日を跨いでも途中入場者は担い手にならない。"""
    sim = _sim(tmp_path, "sl_no_rebind", n_steps=1, n_agents=24, **SL_ON)
    ags = _ags(sim)
    ags[0].occupation = SL.MUSICIAN
    SL.phase(sim, 0, 19 * 60)
    ags[5].occupation = SL.MUSICIAN
    SL.phase(sim, 144, DAY + 19 * 60)
    assert not hasattr(ags[5], "street_slot"), "追随 OFF なのに束ねられた"
    assert sim._sl_state["rebinds"] == 0
    assert list(sim._sl_state["bind_by_day"]) == ["0"], "OFF は day0 の 1 行だけ"

    co = _sim(tmp_path, "co_no_rebind", n_steps=1, n_agents=24, **CO_ON)
    cags = _ags(co)
    cags[0].occupation = "警察官"
    CO.phase(co, 0, 6 * 60)
    cags[5].occupation = "警察官"
    CO.phase(co, 144, DAY + 6 * 60)
    assert not hasattr(cags[5], "city_ops_slot_police")
    assert co._co_state["rebinds"] == 0


# =========================================================================== #
# ⑥ provenance(日次の担い手数)
# =========================================================================== #
def test_provenance_reports_the_daily_workforce(tmp_path):
    """summary の ``bind_by_day`` に日ごとの担い手数が並ぶ(痩せていないかを読む口)。"""
    sim = _sim(tmp_path, "prov", n_steps=1, n_agents=24, **SL_ON_RE)
    ags = _ags(sim)
    for a in ags[:4]:
        a.occupation = SL.MUSICIAN
    SL.phase(sim, 0, 19 * 60)
    SL.phase(sim, 144, DAY + 19 * 60)
    SL.phase(sim, 288, 2 * DAY + 19 * 60)
    prov = SL.provenance(sim)
    assert prov["rebind_daily"] is True
    assert list(prov["bind_by_day"]) == ["0", "1", "2"]
    assert all(v["n_role"] == 4 for v in prov["bind_by_day"].values())
    assert prov["rebinds"] == 2

    co = _sim(tmp_path, "prov_co", n_steps=1, n_agents=24, **CO_ON_RE)
    cags = _ags(co)
    for a in cags[:8]:
        a.occupation = "警察官"
    CO.phase(co, 0, 6 * 60)
    CO.phase(co, 144, DAY + 6 * 60)
    cprov = CO.provenance(co)
    assert cprov["rebind_daily"] is True
    assert list(cprov["bind_by_day"]) == ["0", "1"]
    assert cprov["bind_by_day"]["1"]["police"] == cprov["bind_by_day"]["0"]["police"]


def test_off_provenance_declares_the_toggle(tmp_path):
    """追随 OFF のランでも provenance が申告する(黙って通り過ぎない)。"""
    sim = _sim(tmp_path, "prov_off", n_steps=1, **SL_ON)
    assert SL.provenance(sim)["rebind_daily"] is False
    co = _sim(tmp_path, "prov_off_co", n_steps=1, **CO_ON)
    assert CO.provenance(co)["rebind_daily"] is False


# =========================================================================== #
# ⑦ 実プール回転(小プール・mock)= 担い手が痩せない
# =========================================================================== #
@pytest.fixture(scope="module")
def small_pool(tmp_path_factory):
    """P5 生成器で小プールを tmp に生成(実プール 736MB は触らない)。

    ``test_pool_rotation.py`` の同名 fixture と同じ作法。fraction を少し厚めに取るのは
    L5(路上の生業・都市運営)の役割が 1 人ずつしか出ないと「痩せた」が観測できないため。
    """
    out = tmp_path_factory.mktemp("rebind_pool")
    orgs = json.loads((_ROOT / "data" / "organizations_shibuya_wide11k.json")
                      .read_text(encoding="utf-8"))
    pop = json.loads((_ROOT / "data" / "shibuya_population.json")
                     .read_text(encoding="utf-8"))
    bpp.build_pool(out, seed=42, fraction=0.004, orgs=orgs, pop=pop,
                   total_target=1_000_000)
    return out


_REBIND_ON = {"street_life.rebind_daily": "true", "city_ops.rebind_daily": "true",
              "transit_staff.bind.rebind_daily": "true"}


def _pool_sim(tmp_path, name, pool_dir, n_steps, cap, **ov):
    dot = ["run.seed=42", "run.n_agents=10", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock", "observer.snapshot_every=1440",
           "pool.enabled=true", f"pool.dir={pool_dir}", f"pool.present_cap={cap}",
           "street_life.enabled=true", "city_ops.enabled=true",
           "transit_staff.enabled=true"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


_SLOT_ATTRS = ("street_slot", "city_ops_slot_police", "city_ops_slot_waste",
               "city_ops_slot_driver", "city_ops_slot_clean", "city_ops_slot_ems")


def test_pool_rotation_keeps_the_workforce(tmp_path, small_pool):
    """★本丸: 在場ローテーションが回るランで、追随 ON の担い手が day0 から痩せない。

    比較対象は同一 seed の追随 OFF ラン(= 現行の起動時 1 回きり)。OFF では退場者の枠が
    空いたまま残るので、担い手の実数が day0 より減る(または増えない)。
    """
    steps = 2 * 144 + 1                                    # day0 → day1 の境界を跨ぐ
    off = _pool_sim(tmp_path, "pool_off", small_pool, steps, 250)
    off.run()
    on = _pool_sim(tmp_path, "pool_on", small_pool, steps, 250, **_REBIND_ON)
    on.run()
    assert [e for e in on.logger.events if e.kind == "presence_change"], \
        "在場ローテーションが 1 度も起きていない(前提が崩れている)"
    held_off = {k: _holders(off, k) for k in _SLOT_ATTRS}
    held_on = {k: _holders(on, k) for k in _SLOT_ATTRS}
    for k in _SLOT_ATTRS:
        assert held_on[k] >= held_off[k], \
            f"{k}: 追随 ON の担い手が OFF を下回った({held_on[k]} < {held_off[k]})"
    assert sum(held_on.values()) > 0, "そもそも担い手が 1 人も居ない(前提が崩れている)"
    # 追随 ON では「在場していて役割を持つ個体」が全員束ねられている(取り残しゼロ)
    unbound = [a for a in on.agents
               if str(getattr(a, "occupation", "")) in SL.ROLE_SPECS
               and getattr(a, "street_slot", None) is None]
    assert not unbound, f"路上の役割なのに束ねられていない在場者が {len(unbound)} 人"
    prov = SL.provenance(on)
    assert prov["rebinds"] >= 1 and len(prov["bind_by_day"]) >= 2


def test_rebind_resume_matches_straight(tmp_path, small_pool):
    """★resume 跨ぎ一致: 追随 ON でも「一気 vs 中断 → resume」の L1 が完全一致する。

    ``tests/test_pool_rotation.py::test_on_resume_byte_matches_straight`` と同じ形。
    追随の状態(``_sl_state`` / ``_co_state`` の day・bind_by_day)は checkpoint の対象外だが、
    **枠は agent の属性**として pickle に載るので、resume 後の ``bind`` は「全員が枠を
    持っている」世界を見て 1 バイトも書かない = straight と同じ軌跡になる。
    (日境界は既定 start_tod=07:00 のため step 102 / 246 = 中断点の前後に 1 つずつ入る)
    """
    import pyarrow.parquet as pq
    from society.engine import checkpoint, scheduler

    def rows(d):
        return pq.read_table(Path(d) / "l1_events.parquet").to_pylist()

    straight = tmp_path / "rb_straight"
    _pool_sim(tmp_path, "rb_straight", small_pool, 300, 250, **_REBIND_ON).run()

    rs = tmp_path / "rb_resume"
    s1 = _pool_sim(tmp_path, "rb_resume", small_pool, 150, 250,
                   **{**_REBIND_ON, "observer.checkpoint_every": "150"})
    for step in range(150):
        scheduler.run_step(s1, step)
    checkpoint.save(s1, 150, rs / "checkpoint" / "ckpt-000150.pkl.gz")
    s1._save_pool_sidecar(150)
    s1.logger.flush_segment()
    s2 = _pool_sim(tmp_path, "rb_resume", small_pool, 300, 250,
                   **{**_REBIND_ON, "observer.checkpoint_every": "150"})
    s2.run(resume_from=rs)
    assert rows(straight) == rows(rs), "追随 ON の resume が straight と byte 不一致"
