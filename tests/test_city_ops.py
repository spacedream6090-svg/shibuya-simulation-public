"""都市運営 = 見えない労働力(Wave 4 III-4 ``city_ops``)のテスト。

守るもの(検収基準の順)
  ① OFF(既定)= 純粋既定と L1 バイト一致・新 6 種 0 件・agent に属性が生えない・
     sim に state が生えない・summary キーなし・``delivery_trip`` の payload が従来のまま
  ② 交番配置: 持ち場が**地図データから導いた警察施設ノード集合と厳密一致**し
     (wide 地図では 8 件)、警察官がそこへ束ねられ、巡回ユニットは持ち場を持たない
  ③ ごみ収集: 地区別の曜日表どおりの日にだけ巡回が立つ / 経路は決定論 / 1 班 1 日 1 件
  ④ 納品: 当直の運転手が居れば ``delivery_trip`` が運転手の agent_id を運び、
     居なければ ``unstaffed`` マーカー。★**数量・(s,S)・eta は ON/OFF で完全一致**
  ⑤ 夜間清掃: 夜の窓の中でしか出ない / 1 棟 1 晩 1 件
  ⑥ 救急 = **通報から始まる行為の連鎖**: 倒れる → 近くの誰かが通報 → 当直が応える。
     通報者の選定(最近傍・id 昇順)/ 近くに誰も居なければ自分で通報 /
     患者の状態が治療期間のあとに元へ戻る(健康状態は 1 バイトも書き換えない)/
     健康機構 OFF では 1 件も起きない / 当直不在は unstaffed マーカー
  ⑦ L1 の上限が効く / 新 6 種が全部 causality に分類されている / registry 宣言
  ⑧ ON 同 seed 2 回で完全一致 / **乱数 stream を 1 本も引かない**(AST 検査)
  ⑨ LLM 呼数 ON/OFF 完全一致(FixedLLM)
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from society import city_ops as CO
from society import goods as GOODS
from society import registry as R
from society.config import load_config
from society.engine.simulation import Simulation
from society.observer import causality as C
from society.observer.schema import EVENT_KINDS

MODULE = Path(CO.__file__)

# 本 module が出す L1 種(すべて材料側 registration)
NEW_KINDS = ("city_ops_bound", "waste_collection", "night_cleaning",
             "collapse", "ems_call", "ems_dispatch")

OFF = {"city_ops.enabled": "false"}
ON = {"city_ops.enabled": "true"}

# 本番想定の広域地図(v7 = 現行 / v8 = 別レーンが敷設中)。どちらも交番・警察署は 8 件。
WIDE_MAPS = ("data/shibuya_osm_wide_v7.json", "data/shibuya_osm_wide_v8.json")


# --------------------------------------------------------------------------- #
# 共通ヘルパ(test_street_life.py / test_traces.py と同型)
# --------------------------------------------------------------------------- #
def _cfg(name, n_steps=48, n_agents=15, **ov):
    dot = ["run.seed=42", f"run.n_agents={n_agents}", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock", "observer.snapshot_every=144"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def _sim(tmp_path, name, n_steps=48, n_agents=15, **ov):
    return Simulation(_cfg(name, n_steps, n_agents, **ov), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _kind(sim, kind):
    return [e for e in sim.logger.events if e.kind == kind]


def _place(sim, a, node):
    """個体をノードの路上に立たせる(位置確定後の世界を手で作る)。"""
    a.node = str(node)
    a.x, a.y = sim.city.node_xy(node)
    a.loc = "street"
    a.building = ""
    a.sleeping = False
    a.route = []
    return a


def _expected_police_nodes(cfg) -> set:
    """★地図データ**そのもの**から期待値を作る(実装と同じ表を読まない)。"""
    raw = json.loads(Path(str(cfg.world.map)).read_text(encoding="utf-8"))
    return {str(p["node"]) for p in raw.get("pois", [])
            if str(p.get("cat") or "") == "service"
            and any(w in str(p.get("name") or "") for w in ("交番", "警察署"))}


class _FixedLLM:
    """**プロンプト非依存**の巡回応答スタブ(test_street_life.py と同型)。"""

    name = "fixed"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.hits = 0
        self.prompts: list[str] = []

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        out = self.responses[self.calls % len(self.responses)]
        self.calls += 1
        self.prompts.append(prompt)
        return out, str(self.calls), False


# =========================================================================== #
# ① OFF(既定)
# =========================================================================== #
def test_off_matches_pure_default(tmp_path):
    """明示 OFF と純粋既定が L1 完全一致。新 6 種は 1 件も出ない(seam が no-op)。"""
    pure = _sim(tmp_path, "pure")
    pure.run()
    off = _sim(tmp_path, "expl_off", **OFF)
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(city_ops seam が no-op でない)"
    for k in NEW_KINDS:
        assert not _kind(pure, k), f"OFF で {k} が出ている"


def test_off_grows_no_state_and_no_attributes(tmp_path):
    """OFF は sim に state を生やさず・agent に属性を生やさない(第96 traces と同流儀)。"""
    sim = _sim(tmp_path, "off_state", n_steps=24, **OFF)
    sim.run()
    for attr in ("_co_state", "_co_districts", "_co_police_posts", "_co_offices",
                 "_co_ems_base"):
        assert getattr(sim, attr, None) is None, f"OFF で sim.{attr} が生えた"
    for a in sim.agents:
        for attr in ("city_ops_post", "city_ops_patrol", "city_ops_district",
                     "city_ops_driver", "city_ops_down", "city_ops_ems_until",
                     "city_ops_clean_building"):
            assert not hasattr(a, attr), f"OFF で agent に {attr} が生えた"


def test_off_provenance_is_none(tmp_path):
    sim = _sim(tmp_path, "off_prov", n_steps=1, **OFF)
    assert CO.provenance(sim) is None


def test_off_bind_and_phase_are_noop(tmp_path):
    """OFF では bind も phase も何も返さず何も書かない(直接呼んでも安全)。"""
    sim = _sim(tmp_path, "off_call", n_steps=1, **OFF)
    a = sorted(sim.agents, key=lambda z: int(z.id))[0]
    a.occupation = CO.WASTE
    out = CO.bind(sim)
    CO.phase(sim, 0, 6 * 60)
    assert out["waste"] == 0 and not hasattr(a, "city_ops_district")
    assert CO.assign_delivery_driver(sim, a.node, 0, 5 * 60) is None
    assert not [e for e in sim.logger.events if e.kind in NEW_KINDS]


def test_off_is_unchanged_even_with_city_ops_roles_in_the_roster(tmp_path):
    """名簿に都市運営の役割が居ても OFF は完全 no-op(純粋既定と L1 一致)。"""
    base = _sim(tmp_path, "role_base", n_steps=24)
    for i, a in enumerate(sorted(base.agents, key=lambda z: int(z.id))[:8]):
        a.occupation = CO.CITY_OPS_OCCS[i % len(CO.CITY_OPS_OCCS)]
    base.run()
    off = _sim(tmp_path, "role_off", n_steps=24, **OFF)
    for i, a in enumerate(sorted(off.agents, key=lambda z: int(z.id))[:8]):
        a.occupation = CO.CITY_OPS_OCCS[i % len(CO.CITY_OPS_OCCS)]
    off.run()
    assert _l1(base) == _l1(off), "都市運営の役割入り名簿で seam が no-op でない"


def test_off_keeps_delivery_trip_payload_byte_identical(tmp_path):
    """★物流① の ``delivery_trip`` は OFF で従来の 5 キー・agent_id=-1 のまま。"""
    inv = {"commerce.inventory.enabled": "true"}
    sim = _sim(tmp_path, "dt_off", n_steps=1, **{**OFF, **inv})
    node = sorted(sim.city.graph.nodes)[0]
    sim._goods_stock[(node, "food")] = 0
    GOODS.review_and_order(sim, 0, 5 * 60)
    trips = _kind(sim, "delivery_trip")
    assert trips, "対照が空回り(delivery_trip が出ていない)"
    got = trips[0]
    assert got.agent_id == -1 and got.x == 0.0 and got.y == 0.0
    assert set(got.payload) == {"from", "to", "cat", "qty", "eta"}


# =========================================================================== #
# ② 交番配置
# =========================================================================== #
def test_police_posts_match_the_map_exactly(tmp_path):
    """持ち場は**地図データから導いた警察施設ノード集合と厳密一致**する。"""
    cfg = _cfg("posts", n_steps=1, **ON)
    sim = Simulation(cfg, out_dir=tmp_path / "posts")
    assert set(CO.police_posts(sim)) == _expected_police_nodes(cfg)
    assert CO.police_posts(sim) == sorted(CO.police_posts(sim))   # 決定論の安定順


@pytest.mark.parametrize("wide", WIDE_MAPS)
def test_wide_map_has_eight_police_posts(tmp_path, wide):
    """★wide 地図(本番想定)では交番・警察署が **8 件**(研究事実の固定)。

    v8 でも同じであることまで固定する: v8 は subcat を通すが警察施設は依然
    ``cat=service`` なので、本 module の判定(カテゴリ ∧ 一般名詞)がそのまま効く。
    """
    if not Path(wide).exists():
        pytest.skip(f"{wide} が無い環境")
    name = "posts8_" + Path(wide).stem[-2:]
    cfg = _cfg(name, n_steps=1, **{**ON, "world.map": wide})
    sim = Simulation(cfg, out_dir=tmp_path / name)
    posts = CO.police_posts(sim)
    assert set(posts) == _expected_police_nodes(cfg)
    assert len(posts) == 8, f"{wide} の警察施設が 8 件でない: {len(posts)}"


@pytest.mark.parametrize("wide", WIDE_MAPS)
def test_no_vending_poi_exists_so_the_feature_is_skipped(tmp_path, wide):
    """★自販機補充を作らなかった理由の**機械固定**: 地図に自販機 POI が 1 件も無い。

    v8 は subcat を通している(convenience 等)ので「subcat が無いから」ではなく
    「**自販機そのものが地図に無い**」ことが理由である、と実データで示す。
    トリップワイヤー: 地図生成器が自販機を通した日にこのテストが落ち、
    そのとき初めて役割を足す判断ができる。
    """
    if not Path(wide).exists():
        pytest.skip(f"{wide} が無い環境")
    raw = json.loads(Path(wide).read_text(encoding="utf-8"))
    pois = raw.get("pois", [])
    vending = [p for p in pois
               if "vending" in str(p.get("subcat") or "")
               or "自販" in str(p.get("name") or "")
               or "自動販売" in str(p.get("name") or "")]
    assert vending == [], f"自販機 POI が現れた(補充役割の追加を検討する): {vending[:3]}"
    if Path(wide).name.endswith("v8.json"):
        subs = {str(p.get("subcat")) for p in pois if p.get("subcat")}
        assert subs, "v8 に subcat が 1 件も無い(理由の前提が崩れた)"


def test_bus_stop_named_after_a_police_box_is_not_a_post(tmp_path):
    """★『〜交番前』のバス停(cat=office)は持ち場にしない(カテゴリで絞る効き目)。"""
    cfg = _cfg("posts_neg", n_steps=1, **ON)
    raw = json.loads(Path(str(cfg.world.map)).read_text(encoding="utf-8"))
    decoys = {str(p["node"]) for p in raw.get("pois", [])
              if str(p.get("cat") or "") != "service"
              and "交番" in str(p.get("name") or "")}
    if not decoys:
        pytest.skip("この地図には紛らわしい非 service POI が無い")
    sim = Simulation(cfg, out_dir=tmp_path / "posts_neg")
    assert not (decoys & set(CO.police_posts(sim)))


def test_officers_are_bound_to_the_posts_and_a_patrol_subset_stays_free(tmp_path):
    """警察官は交番へ束ねられ、``patrol_every`` 人に 1 人は持ち場を持たない巡回ユニット。"""
    sim = _sim(tmp_path, "police", n_steps=1, n_agents=20, **ON)
    ags = sorted(sim.agents, key=lambda a: int(a.id))
    for a in ags[:8]:
        a.occupation = "警察官"
    for a in ags[8:]:
        a.occupation = "会社員"
    CO.bind(sim)
    posts = set(CO.police_posts(sim))
    officers = [a for a in ags[:8]]
    posted = [a for a in officers if getattr(a, "city_ops_post", "")]
    patrol = [a for a in officers if getattr(a, "city_ops_patrol", False)]
    assert posted and patrol, "配置も巡回もどちらか一方しか出ていない"
    assert len(posted) + len(patrol) == len(officers)
    assert len(patrol) == len(officers) // 4                   # patrol_every=4
    for a in posted:
        assert a.work_node in posts, "警察官が交番以外に束ねられた"
        assert a.work_building == "", "交番を建物内勤務にしている"
        assert 0 <= a.work_start_min < a.work_end_min <= 1440


def test_police_shifts_cover_the_whole_day(tmp_path):
    """3 直(既定)で 24 時間を覆う = 交番が無人になる時間帯を作らない。"""
    pcfg = CO.DEFAULTS["police"]
    windows = [CO.duty_window(pcfg["first_open"], pcfg["shift_hours"],
                              pcfg["n_shifts"], i)
               for i in range(pcfg["n_shifts"])]
    covered = set()
    for open_min, close_min in windows:
        covered |= set(range(open_min, close_min))
    assert covered == set(range(0, 1440)), f"24 時間を覆えていない: {windows}"


def test_bind_is_idempotent(tmp_path):
    """★resume 安全: bind を 2 回呼んでも結果が 1 バイトも変わらない。"""
    sim = _sim(tmp_path, "idem", n_steps=1, n_agents=20, **ON)
    ags = sorted(sim.agents, key=lambda a: int(a.id))
    for i, a in enumerate(ags[:12]):
        a.occupation = (CO.CITY_OPS_OCCS + ("警察官",))[i % 5]
    first = CO.bind(sim)
    snap = [(a.id, a.work_node, a.work_start_min, a.work_end_min) for a in ags]
    second = CO.bind(sim)
    assert first == second
    assert snap == [(a.id, a.work_node, a.work_start_min, a.work_end_min)
                    for a in ags]


# =========================================================================== #
# ③ ごみ収集
# =========================================================================== #
def test_weekday_table_is_two_days_a_week_per_district():
    """可燃 2 回/週の地区別曜日表(月木 / 火金 / 水土)= 純関数の固定。"""
    wcfg = CO.DEFAULTS["waste"]
    for idx, pair in enumerate(((0, 3), (1, 4), (2, 5))):
        days = [d for d in range(7) if CO._collect_day(wcfg, idx, d)]
        assert days == list(pair), f"地区 {idx} の曜日表が {days}"
        assert len(days) == 2, "週 2 回になっていない"
    # 地区は 3 組を循環する(地区数が組数より多くても曜日表が途切れない)
    assert CO._collect_day(wcfg, 3, 0) and not CO._collect_day(wcfg, 3, 1)


def test_districts_are_derived_from_the_map_deterministically(tmp_path):
    """地区は地図の格子分割で決まり、2 回引いても同一(乱数ゼロ)。"""
    sim = _sim(tmp_path, "districts", n_steps=1, **ON)
    first = CO.districts(sim)
    assert first and len(first) <= 6                        # 3x2 格子(空区画は落ちる)
    names = [d["name"] for d in first]
    assert names == sorted(names) and len(set(names)) == len(names)
    sim._co_districts = None
    assert [d["ward_stops"] for d in CO.districts(sim)] == \
        [d["ward_stops"] for d in first]


def _waste_stage(tmp_path, name, day, minute, **ov):
    """収集作業員 1 人を担当地区の先頭ノードに立たせた舞台。"""
    sim = _sim(tmp_path, name, n_steps=1, n_agents=12, **{**ON, **ov})
    ags = sorted(sim.agents, key=lambda a: int(a.id))
    crew = ags[0]
    crew.occupation = CO.WASTE
    for other in ags[1:]:
        other.loc = "outside"
    CO.bind(sim)
    _place(sim, crew, crew.work_node)
    return sim, crew, day * 1440 + minute


def test_collection_runs_only_on_the_scheduled_weekday(tmp_path):
    """★担当地区の収集日にだけ巡回が立つ(曜日表どおり)。

    暦 OFF のランでは初日=月曜(経過日の剰余)。地区 d0 の担当は月木なので、
    day0(月)= 出る / day1(火)= 出ない / day3(木)= 出る。
    """
    got = {}
    for day in (0, 1, 3):
        sim, crew, sim_min = _waste_stage(tmp_path, f"ws{day}", day, 6 * 60)
        assert getattr(crew, "city_ops_district", "") == "d0"
        CO.phase(sim, day * 144, sim_min)
        got[day] = len(_kind(sim, "waste_collection"))
    assert got == {0: 1, 1: 0, 3: 1}, f"曜日表どおりに巡回していない: {got}"


def test_collection_runs_only_inside_the_morning_window(tmp_path):
    """収集の帯(05:00〜07:30)の外では 1 件も出ない(繁華街の期限)。"""
    for minute, want in ((4 * 60, 0), (6 * 60, 1), (9 * 60, 0)):
        sim, _crew, sim_min = _waste_stage(tmp_path, f"wm{minute}", 0, minute)
        CO.phase(sim, 0, sim_min)
        assert len(_kind(sim, "waste_collection")) == want, f"{minute} 分で {want} 件でない"


def test_collection_is_one_event_per_crew_per_day(tmp_path):
    """1 班 1 日 1 件(何 step 回っても L1 は 1 件 = throttle)。"""
    sim, crew, sim_min = _waste_stage(tmp_path, "wonce", 0, 5 * 60)
    seen = []
    for i in range(12):
        CO.phase(sim, i, sim_min + i * 10)
        seen.append(str(crew.work_node))
    assert len(_kind(sim, "waste_collection")) == 1
    got = _kind(sim, "waste_collection")[0]
    assert got.payload["kind"] == "ward" and got.payload["stops"] == 6
    assert got.payload["deadline_min"] == 450
    assert len(set(seen)) > 1, "巡回の持ち場が 1 度も進んでいない(経路になっていない)"
    route = list(CO.districts(sim)[0]["ward_stops"])
    order = [route.index(n) for n in seen]
    assert order == sorted(order), f"巡回が行きつ戻りつしている: {order}"


def test_commercial_crews_are_not_bound_without_night_economy(tmp_path):
    """★深夜の事業系は夜間開放 ON が前提(OFF では束ねず、理由を notes に残す)。"""
    sim = _sim(tmp_path, "wnight_off", n_steps=1, n_agents=12, **ON)
    for a in sorted(sim.agents, key=lambda z: int(z.id))[:4]:
        a.occupation = CO.WASTE
    stat = CO.bind(sim)
    assert stat["waste_night"] == 0
    assert "commercial_waste_needs_night_economy" in stat["notes"]
    on = _sim(tmp_path, "wnight_on", n_steps=1, n_agents=12,
              **{**ON, "world.night_economy.enabled": "true"})
    for a in sorted(on.agents, key=lambda z: int(z.id))[:4]:
        a.occupation = CO.WASTE
    stat2 = CO.bind(on)
    assert stat2["waste_night"] > 0
    assert "commercial_waste_needs_night_economy" not in stat2["notes"]


# =========================================================================== #
# ④ 納品の運転手化
# =========================================================================== #
def _driver_stage(tmp_path, name, **ov):
    sim = _sim(tmp_path, name, n_steps=1, n_agents=12,
               **{**ON, "commerce.inventory.enabled": "true", **ov})
    ags = sorted(sim.agents, key=lambda a: int(a.id))
    driver = ags[0]
    driver.occupation = CO.DRIVER
    for other in ags[1:]:
        other.loc = "outside"
    CO.bind(sim)
    _place(sim, driver, driver.work_node)
    node = sorted(sim.city.graph.nodes)[0]
    sim._goods_stock[(node, "food")] = 0
    return sim, driver, node


def test_delivery_trip_carries_the_driver_when_on_duty(tmp_path):
    """当直の運転手が居れば ``delivery_trip`` が運転手の agent_id を運ぶ。"""
    sim, driver, _node = _driver_stage(tmp_path, "drv_on")
    GOODS.review_and_order(sim, 0, 5 * 60)                   # 05:00 = 納品帯の中
    trips = _kind(sim, "delivery_trip")
    assert trips and trips[0].agent_id == int(driver.id)
    assert trips[0].payload["driver"] == int(driver.id)
    assert "unstaffed" not in trips[0].payload
    assert CO.provenance(sim)["trips_staffed"] == len(trips)


def test_delivery_trip_is_marked_unstaffed_outside_the_driver_window(tmp_path):
    """納品帯の外(= 当直不在)は ``unstaffed`` マーカー(黙って無人で運ばない)。"""
    sim, _driver, _node = _driver_stage(tmp_path, "drv_off")
    GOODS.review_and_order(sim, 0, 20 * 60)                  # 20:00 = 納品帯の外
    trips = _kind(sim, "delivery_trip")
    assert trips and trips[0].agent_id == -1
    assert trips[0].payload["unstaffed"] is True
    assert "driver" not in trips[0].payload
    assert CO.provenance(sim)["trips_unstaffed"] == len(trips)


def test_sleeping_driver_is_not_on_duty(tmp_path):
    """眠っている運転手は当直にならない(正直な稼働率)。"""
    sim, driver, node = _driver_stage(tmp_path, "drv_sleep")
    driver.sleeping = True
    assert CO.assign_delivery_driver(sim, node, 0, 5 * 60) is None


def test_restock_quantities_are_identical_on_and_off(tmp_path):
    """★**行動中立**: 数量・(s,S) 判定・eta・発注先は ON/OFF で 1 も変わらない。"""
    payloads = {}
    for name, gate in (("q_off", OFF), ("q_on", ON)):
        sim = _sim(tmp_path, name, n_steps=1, n_agents=12,
                   **{**gate, "commerce.inventory.enabled": "true"})
        ags = sorted(sim.agents, key=lambda a: int(a.id))
        ags[0].occupation = CO.DRIVER
        for other in ags[1:]:
            other.loc = "outside"
        if gate is ON:
            CO.bind(sim)
            _place(sim, ags[0], ags[0].work_node)
        for i, node in enumerate(sorted(sim.city.graph.nodes)[:5]):
            sim._goods_stock[(node, "food")] = i
        GOODS.review_and_order(sim, 0, 5 * 60)
        payloads[name] = [
            {k: v for k, v in e.payload.items() if k not in ("driver", "unstaffed")}
            for e in _kind(sim, "delivery_trip")]
        payloads[name + "_low"] = [dict(e.payload) for e in _kind(sim, "stock_low")]
    assert payloads["q_off"] and payloads["q_off"] == payloads["q_on"], \
        "運転手の割当が補充の数量/発注先/eta を動かした"
    assert payloads["q_off_low"] == payloads["q_on_low"], "stock_low が動いた"


# =========================================================================== #
# ⑤ ビル夜間清掃
# =========================================================================== #
def _clean_stage(tmp_path, name, **ov):
    sim = _sim(tmp_path, name, n_steps=1, n_agents=12,
               **{**ON, "world.night_economy.enabled": "true", **ov})
    ags = sorted(sim.agents, key=lambda a: int(a.id))
    worker = ags[0]
    worker.occupation = CO.NIGHT_CLEANER
    for other in ags[1:]:
        other.loc = "outside"
    CO.bind(sim)
    worker.loc = "street"
    worker.sleeping = False
    worker.node = str(worker.work_node)
    worker.x, worker.y = sim.city.node_xy(worker.node)
    worker.building = str(worker.city_ops_clean_building)   # 担当ビルの中に居る
    worker.floor = 1
    return sim, worker


def test_night_cleaning_only_inside_the_night_window(tmp_path):
    """夜の窓(22:00〜05:00)の中でしか出ない。"""
    for minute, want in ((14 * 60, 0), (23 * 60, 1), (3 * 60, 1), (10 * 60, 0)):
        sim, _w = _clean_stage(tmp_path, f"nc{minute}")
        CO.phase(sim, 0, minute)
        assert len(_kind(sim, "night_cleaning")) == want, f"{minute} 分で {want} 件でない"


def test_night_cleaning_is_one_event_per_building_per_night(tmp_path):
    """1 棟 1 晩 1 件(同じ晩に何 step 居ても増えない)。"""
    sim, worker = _clean_stage(tmp_path, "nc_once")
    for i in range(8):
        CO.phase(sim, i, 22 * 60 + i * 10)
    assert len(_kind(sim, "night_cleaning")) == 1
    got = _kind(sim, "night_cleaning")[0]
    assert got.agent_id == int(worker.id)
    assert got.payload["building"] == str(worker.city_ops_clean_building)
    # 翌日の夜は別の晩 = もう 1 件出る
    CO.phase(sim, 200, 1440 + 22 * 60)
    assert len(_kind(sim, "night_cleaning")) == 2


def test_night_cleaning_needs_the_worker_inside_the_building(tmp_path):
    """担当ビルの中に居ない夜は 1 件も出ない(持ち場に着けなければ何も起きない)。"""
    sim, worker = _clean_stage(tmp_path, "nc_out")
    worker.building = ""
    CO.phase(sim, 0, 23 * 60)
    assert not _kind(sim, "night_cleaning")


def test_night_cleaning_window_is_clamped_without_night_economy(tmp_path):
    """夜間開放 OFF では日跨ぎ窓を表現できないので 24:00 へ畳み、理由を notes に残す。"""
    sim = _sim(tmp_path, "nc_clamp", n_steps=1, n_agents=12, **ON)
    worker = sorted(sim.agents, key=lambda a: int(a.id))[0]
    worker.occupation = CO.NIGHT_CLEANER
    stat = CO.bind(sim)
    assert stat["cleaner"] == 1
    assert "night_cleaning_window_clamped" in stat["notes"]
    assert worker.work_end_min == 1440 and not getattr(worker, "work_wraps", False)


# =========================================================================== #
# ⑥ 救急 = 通報から始まる行為の連鎖
# =========================================================================== #
HEALTH_ON = {"health.enabled": "true", "health.onset_prob": "0.0"}


def _ems_stage(tmp_path, name, n_agents=12, crowd=2, **ov):
    """病人 1 人 + 近くの通行人 + 当直の救急隊 1 人を立たせた舞台。"""
    # ★倒れる日の抽選(1 万人あたり n 人)は舞台では常に当たりにする = 検査したいのは
    #   **連鎖の形**であって抽選の刻みではない(刻みそのものは別テストで固定する)。
    sim = _sim(tmp_path, name, n_steps=1, n_agents=n_agents,
               **{**ON, **HEALTH_ON, "city_ops.ems.collapse_per_10k": "10000", **ov})
    ags = sorted(sim.agents, key=lambda a: int(a.id))
    crew = ags[0]
    crew.occupation = CO.EMS_CREW
    # ★``bind`` を直に呼ぶと ``phase`` 側が「まだ束ねていない」と見て**もう一度束ねる**
    #   (冪等なので結果は同じだが、下で手作りした当直帯まで既定へ戻ってしまう)。
    #   束ね済みの印ごと立てる内部入口を使い、舞台を確定させてから窓を上書きする。
    CO._ensure_bound(sim, -1, 0)
    patient = ags[1]
    patient.occupation = "会社員"
    patient.sick = True
    nodes = sorted(sim.city.graph.nodes)
    node = nodes[0]
    _place(sim, patient, node)
    # ★救急隊は**別のノード**に待機させる: 同じノードに立たせると隊員自身が
    #   「いちばん近くに居合わせた人」= 通報者になってしまい、通報者選定の検査が
    #   隊員の配置に引きずられる(連鎖の形と選定規則を分けて固定するための舞台作り)。
    _place(sim, crew, nodes[-1])
    crew.work_start_min, crew.work_end_min = 0, 1440       # 当直に立たせる
    near = ags[2:2 + crowd]
    for other in near:
        other.occupation = "会社員"
        _place(sim, other, node)
    for other in ags[2 + crowd:]:
        other.loc = "outside"
    return sim, patient, crew, near


def _collapse_min(patient, sim) -> int:
    """その個体が倒れる step-of-day(実装と同じ純関数を使わず素で再現する)。"""
    day = 0
    spd = int(sim.clock.steps_per_day)
    return CO._stable_hash(f"collapse_slot/{int(patient.id)}/{day}") % spd


def test_collapse_needs_the_existing_health_state(tmp_path):
    """★健康機構が成立させた状態が無ければ 1 件も起きない(正直な依存関係)。"""
    sim, patient, _crew, _near = _ems_stage(tmp_path, "ems_nohealth")
    patient.sick = False                                   # 病気でない = 引き金が無い
    for step in range(int(sim.clock.steps_per_day)):
        CO.phase(sim, step, 420 + step * 10)
    assert not _kind(sim, "collapse") and not _kind(sim, "ems_call")


def test_the_chain_is_collapse_then_call_then_dispatch(tmp_path):
    """★倒れる → **近くの誰かが通報** → 当直が応える、の 3 段が同じ step に立つ。"""
    sim, patient, crew, near = _ems_stage(tmp_path, "ems_chain")
    slot = _collapse_min(patient, sim)
    CO.phase(sim, slot, 420 + slot * 10)
    downs, calls, disp = (_kind(sim, "collapse"), _kind(sim, "ems_call"),
                          _kind(sim, "ems_dispatch"))
    assert len(downs) == 1 and len(calls) == 1 and len(disp) == 1
    assert downs[0].agent_id == int(patient.id)
    assert downs[0].payload["source"] == "illness"
    # 通報者 = 患者ではない近傍の個体(= この行為が出動の原因)
    assert calls[0].agent_id in {int(a.id) for a in near}
    assert calls[0].payload["patient"] == int(patient.id)
    assert calls[0].payload["self_call"] is False
    # 出動は通報に紐づく(payload が因果の親を持つ)
    assert disp[0].agent_id == int(crew.id)
    assert disp[0].payload["caller"] == calls[0].agent_id
    assert disp[0].payload["patient"] == int(patient.id)
    assert disp[0].payload["unstaffed"] is False
    assert disp[0].payload["response_min"] is not None
    assert crew.work_node == str(patient.node), "救急隊の持ち場が現場へ移っていない"


def test_the_caller_is_the_nearest_agent_with_id_tiebreak(tmp_path):
    """通報者は**最近傍**(同距離は id 昇順)。決定論で一意に決まる。"""
    sim, patient, _crew, near = _ems_stage(tmp_path, "ems_caller", n_agents=16,
                                           crowd=3)
    assert len(near) == 3
    far, mid, close = near
    # 距離を作る(同一ノードの座標を手でずらす = 位置確定後の世界)
    far.x, far.y = patient.x + 30.0, patient.y
    mid.x, mid.y = patient.x + 20.0, patient.y
    close.x, close.y = patient.x + 5.0, patient.y
    slot = _collapse_min(patient, sim)
    CO.phase(sim, slot, 420 + slot * 10)
    calls = _kind(sim, "ems_call")
    assert len(calls) == 1 and calls[0].agent_id == int(close.id)
    assert calls[0].payload["dist_m"] == pytest.approx(5.0, abs=0.05)


def test_the_patient_calls_when_nobody_is_near(tmp_path):
    """近くに誰も居なければ**本人が自分で呼ぶ**(出動が主体なしにならない)。"""
    sim, patient, _crew, near = _ems_stage(tmp_path, "ems_self")
    for other in near:
        other.loc = "outside"                              # 隊員は舞台の時点で遠くに居る
    slot = _collapse_min(patient, sim)
    CO.phase(sim, slot, 420 + slot * 10)
    calls = _kind(sim, "ems_call")
    assert len(calls) == 1
    assert calls[0].agent_id == int(patient.id)
    assert calls[0].payload["self_call"] is True


def test_the_patient_round_trips_without_touching_health_state(tmp_path):
    """★患者は治療期間だけその場に留まり、**健康状態は 1 バイトも書き換えない**。"""
    sim, patient, _crew, _near = _ems_stage(tmp_path, "ems_round")
    slot = _collapse_min(patient, sim)
    node_before = str(patient.node)
    CO.phase(sim, slot, 420 + slot * 10)
    assert patient.city_ops_down is True and patient.sleeping is True
    assert patient.route == [] and str(patient.node) == node_before
    assert patient.sick is True, "本 module が健康状態を書き換えた"
    treat = CO.DEFAULTS["ems"]["treatment_steps"]
    assert patient.sleep_until == slot + treat
    CO.phase(sim, slot + treat, 420 + (slot + treat) * 10)
    assert patient.city_ops_down is False, "倒れている印が落ちない"
    assert patient.sick is True, "回復のついでに病気まで治した"


def test_the_crew_returns_to_base_after_the_scene(tmp_path):
    """救急隊は現着期間のあと待機拠点へ戻る(持ち場の round trip)。"""
    sim, patient, crew, _near = _ems_stage(tmp_path, "ems_back")
    base = str(crew.work_node)
    slot = _collapse_min(patient, sim)
    CO.phase(sim, slot, 420 + slot * 10)
    assert crew.work_node == str(patient.node), "持ち場が現場へ移っていない"
    on_scene = CO.DEFAULTS["ems"]["on_scene_steps"]
    CO.phase(sim, slot + on_scene, 420 + (slot + on_scene) * 10)
    assert crew.work_node == base, "待機拠点へ戻っていない"
    assert crew.city_ops_ems_until == -1


def test_dispatch_is_marked_unstaffed_without_an_on_duty_crew(tmp_path):
    """当直が居ない回は agent_id=-1 + unstaffed(装置 id は作らない)。"""
    sim, patient, crew, _near = _ems_stage(tmp_path, "ems_unstaffed")
    crew.work_start_min, crew.work_end_min = 0, 1           # 当直帯の外にする
    slot = _collapse_min(patient, sim)
    CO.phase(sim, slot, 420 + slot * 10)
    disp = _kind(sim, "ems_dispatch")
    assert len(disp) == 1 and disp[0].agent_id == -1
    assert disp[0].payload["unstaffed"] is True and disp[0].payload["crew"] == -1
    assert disp[0].payload["response_min"] is None


def test_fatigue_can_be_the_trigger_when_illness_is_not_required(tmp_path):
    """``require_sick=false`` では既存の疲労ゲージも引き金になる(健康 OFF でも demo 可)。"""
    sim, patient, _crew, _near = _ems_stage(tmp_path, "ems_fatigue",
                                            **{"city_ops.ems.require_sick": "false"})
    patient.sick = False
    patient.fatigue = 1.0
    slot = _collapse_min(patient, sim)
    CO.phase(sim, slot, 420 + slot * 10)
    downs = _kind(sim, "collapse")
    assert len(downs) == 1 and downs[0].payload["source"] == "fatigue"


def test_calls_are_capped_per_day(tmp_path):
    """1 日に出す連鎖の上限が効く(L1 の安全弁)。"""
    sim = _sim(tmp_path, "ems_cap", n_steps=1, n_agents=40,
               **{**ON, **HEALTH_ON, "city_ops.ems.max_calls_per_day": "1",
                  "city_ops.ems.collapse_per_10k": "10000"})
    ags = sorted(sim.agents, key=lambda a: int(a.id))
    node = sorted(sim.city.graph.nodes)[0]
    CO.bind(sim)
    for a in ags:
        a.occupation = "会社員"
        a.sick = True
        _place(sim, a, node)
    # ★1 日ぶんだけ回す(開始が 07:00 なので step 102 で日が変わり cap が明けてしまう)
    for step in range(100):
        CO.phase(sim, step, 420 + step * 10)
    assert len(_kind(sim, "ems_call")) == 1


# =========================================================================== #
# ⑦ 台帳・分類・安全弁
# =========================================================================== #
def test_all_new_kinds_are_registered_and_classified():
    """新 6 種が EVENT_KINDS(材料側 registration)と causality の両方に載っている。"""
    for k in NEW_KINDS:
        assert k in EVENT_KINDS, f"{k} が schema に未登録"
        assert k in C.CAUSE_OF_KIND, f"{k} が因果台帳に未分類"
        assert C.CAUSE_OF_KIND[k] in C.CAUSE_TYPES


def test_the_call_chain_is_classified_as_an_agent_act():
    """★通報と出動は**行為**、倒れるのは**身体**(時刻表・装置ではない)。"""
    assert C.CAUSE_OF_KIND["ems_call"] == C.AGENT
    assert C.CAUSE_OF_KIND["ems_dispatch"] == C.AGENT
    assert C.CAUSE_OF_KIND["collapse"] == C.PHYSICS
    assert C.CAUSE_OF_KIND["waste_collection"] == C.AGENT
    assert C.CAUSE_OF_KIND["night_cleaning"] == C.AGENT
    assert C.CAUSE_OF_KIND["city_ops_bound"] == C.BOUNDARY
    # 出動に装置 id は与えない(救急を運行する装置は世界に存在しない)
    assert C.CAUSE_OF_KIND["ems_dispatch"] not in C.DEVICE_STAMPABLE


def test_delivery_trip_stays_a_device_act():
    """★補充トリップの分類は動かさない: 発注を決めるのは (s,S) 方策(装置)で、
    運転手はそれを**運んだ人**である(``dwell_decision`` と同じ線引き)。"""
    assert C.CAUSE_OF_KIND["delivery_trip"] == C.DEVICE


def test_registry_declares_the_toggles():
    ids = {f.id for f in R.FEATURES}
    for key in ("city_ops.enabled", "city_ops.police.enabled",
                "city_ops.waste.enabled", "city_ops.waste.commercial_enabled",
                "city_ops.waste.require_night", "city_ops.driver.enabled",
                "city_ops.night_cleaning.enabled", "city_ops.ems.enabled",
                "city_ops.ems.require_sick"):
        assert key in ids, f"{key} が registry に未宣言"
    f = next(f for f in R.FEATURES if f.id == "city_ops.enabled")
    assert f.repro_tier == "strict" and f.affects_k is False


def test_shipped_default_is_off():
    cfg = load_config()
    assert bool(cfg.city_ops.enabled) is False


def test_l1_budget_caps_events_per_step(tmp_path):
    """1 step の L1 件数は max_events_per_step で頭打ちになり、超過は捨てて数える。"""
    sim = _sim(tmp_path, "budget", n_steps=1, n_agents=40,
               **{**ON, **HEALTH_ON, "city_ops.max_events_per_step": "2",
                  "city_ops.ems.collapse_per_10k": "10000"})
    ags = sorted(sim.agents, key=lambda a: int(a.id))
    node = sorted(sim.city.graph.nodes)[0]
    CO.bind(sim)
    for a in ags:
        a.occupation = "会社員"
        a.sick = True
        _place(sim, a, node)
    for step in range(int(sim.clock.steps_per_day)):
        CO.phase(sim, step, 420 + step * 10)
        if sim._co_state["dropped"]:
            break
    # ★city_ops_bound は**ラン全体で 1 件**の起動時記録なので予算の対象外
    #   (workplace_bound / transit_staff_bound と同型)。予算が縛るのは毎 step の出来事。
    per_step: dict[int, int] = {}
    for e in sim.logger.events:
        if e.kind in NEW_KINDS and e.kind != "city_ops_bound":
            per_step[e.step] = per_step.get(e.step, 0) + 1
    assert per_step and max(per_step.values()) <= 2
    assert len(_kind(sim, "city_ops_bound")) == 1
    assert sim._co_state["dropped"] >= 1


def test_measured_l1_volume_stays_small(tmp_path):
    """★実測: 本 module の L1 は総量の 0.2% 未満(全種 throttle 済み)。"""
    sim = _sim(tmp_path, "vol", n_steps=144, n_agents=40,
               **{**ON, "world.night_economy.enabled": "true",
                  "health.enabled": "true", "health.onset_prob": "0.35",
                  "commerce.inventory.enabled": "true"})
    roles = list(CO.CITY_OPS_OCCS) + ["警察官"]
    for i, a in enumerate(sorted(sim.agents, key=lambda z: int(z.id))[:20]):
        a.occupation = roles[i % len(roles)]
    sim.run()
    own = sum(1 for e in sim.logger.events if e.kind in NEW_KINDS)
    total = len(sim.logger.events)
    assert total > 1000, "対照が小さすぎて割合の意味が無い"
    assert own / total < 0.002, f"L1 の占有率が高い: {own}/{total}"
    prov = CO.provenance(sim)
    assert prov["dropped"] == 0 and prov["schema"] == CO.SCHEMA
    assert "ward" in prov["rounds_by_kind"]


# =========================================================================== #
# ⑧ 決定論・乱数ゼロ
# =========================================================================== #
def test_on_is_deterministic_across_two_runs(tmp_path):
    """ON 同 seed 2 ランで L1 完全一致。"""
    outs = []
    for i in (1, 2):
        sim = _sim(tmp_path, f"det{i}", n_steps=48, n_agents=20,
                   **{**ON, "health.enabled": "true", "health.onset_prob": "0.4",
                      "world.night_economy.enabled": "true"})
        roles = list(CO.CITY_OPS_OCCS) + ["警察官"]
        for j, a in enumerate(sorted(sim.agents, key=lambda z: int(z.id))[:10]):
            a.occupation = roles[j % len(roles)]
        sim.run()
        outs.append(_l1(sim))
    assert outs[0] == outs[1]


def test_memory_lines_are_place_and_number_free():
    """★no-fingerprint: 記憶へ入る定型文に**地名・数字・実験条件・機構語**が 1 つも無い。

    (``traces.sentence`` / ``street_life.NOTICE_TEXT`` と同じ規約。全実験条件で同一
    バイト列になることが「当人から見て条件差が観測できない」の担保である。)
    """
    texts = [CO.COLLAPSE_TEXT, CO.CALL_TEXT, CO.SELF_CALL_TEXT, CO.DISPATCH_TEXT,
             CO.WASTE_TEXT, CO.CLEAN_TEXT]
    banned = ("渋谷", "道玄坂", "宇田川", "ハチ公", "センター街", "city_ops",
              "collapse", "config", "step", "%", "円")
    for t in texts:
        assert t and not any(ch.isdigit() for ch in t), f"数字が入っている: {t}"
        for w in banned:
            assert w not in t, f"禁止語 {w} が入っている: {t}"


def test_district_names_are_ordinals_not_place_names(tmp_path):
    """★地区名は序数(``d0``…)であって地名ではない(区の実区割りを持っていないため)。"""
    import re
    sim = _sim(tmp_path, "dnames", n_steps=1, **ON)
    for d in CO.districts(sim):
        assert re.fullmatch(r"d\d+", d["name"]), f"地区名が序数でない: {d['name']}"


def test_module_draws_no_random_stream():
    """★AST 静的検査: 本 module に乱数の呼び出しが**識別子として存在しない**。"""
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    for banned in ("stream", "random", "integers", "uniform", "shuffle", "choice",
                   "default_rng", "hub"):
        assert banned not in names, f"乱数の識別子 {banned} が city_ops.py にある"


def test_module_makes_no_llm_call():
    """★AST 静的検査: generate() の呼び出しサイトが 1 つも無い(k 非依存)。"""
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert "generate" not in names
    assert "llm" not in {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}


def test_module_never_writes_health_state():
    """★AST 静的検査: ``sick`` / ``sick_until`` / ``fatigue`` へ**代入しない**
    (健康機構は別レーンの所有。本 module は読むだけ)。"""
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    written = set()
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        for t in targets:
            if isinstance(t, ast.Attribute):
                written.add(t.attr)
    for banned in ("sick", "sick_until", "fatigue", "withdrawn", "chronic_days"):
        assert banned not in written, f"city_ops.py が健康状態 {banned} を書いている"


# =========================================================================== #
# ⑨ LLM 呼数(R1)
# =========================================================================== #
def test_the_phase_hook_itself_adds_no_llm_call(tmp_path):
    """★R1: 名簿に都市運営の役割が 1 人も居なければ ON/OFF で generate() の呼数が完全一致。

    ★正直な限界(street_life / diversity と同型): 名簿に役割が**居る**ランでは、
      持ち場への束ねが物理位置を変えるので co-location 経由で発火数が動きうる。
      それは「generate() の呼び出しサイトを足したか」という R1 の問いとは別で、
      本 module は呼び出しサイトを 1 つも持たない(上の AST 検査が機械固定)。
    """
    calls = []
    for name, ov in (("k_off", OFF), ("k_on", ON)):
        sim = _sim(tmp_path, name, n_steps=48, n_agents=20, **ov)
        sim.llm = _FixedLLM([json.dumps({"action": "stay"}, ensure_ascii=False)])
        sim.run()
        calls.append(sim.llm.calls)
    assert calls[0] == calls[1], f"フック自体が LLM 呼数を動かした: {calls}"


def test_collapse_does_not_add_a_morning_plan_call(tmp_path):
    """★倒れて起きても朝の計画は増えない(plan_day の 1 日 1 回ガードに守られる)。"""
    calls = []
    for name, ov in (("plan_off", {**OFF, **HEALTH_ON}),
                     ("plan_on", {**ON, **HEALTH_ON})):
        sim = _sim(tmp_path, name, n_steps=48, n_agents=16,
                   **{**ov, "planning.enabled": "true"})
        for a in sorted(sim.agents, key=lambda z: int(z.id))[:8]:
            a.sick = True
        sim.llm = _FixedLLM([json.dumps({"action": "stay"}, ensure_ascii=False)])
        sim.run()
        calls.append(sim.llm.calls)
    assert calls[0] == calls[1], f"倒れた個体の起床が LLM 呼を増やした: {calls}"


# =========================================================================== #
# B4(第118 レーンC): ``_co_state`` を checkpoint で運ぶ
#
#   救急の 1 日上限(``calls_by_day``)と夜間清掃の「1 棟 1 晩 1 件」(``cleaned_night``)、
#   役割バインドの日次追随(``day`` / ``bind`` / ``bind_by_day``)は **世界側の状態**で、
#   agents pickle には載らない。未保存だったため resume 直後にどちらのガードも 0 へ戻り、
#   同じ日に上限ぶんもう一度出動し、同じ棟の ``night_cleaning`` がもう 1 件出ていた
#   (第80 W2 と同型。ただし状態の本体が sim 側にあるので trace_state 族の中央管理)。
# =========================================================================== #
def _ckpt(sim, tmp_path, name, step=100):
    from society.engine import checkpoint
    return checkpoint.save(sim, step,
                           tmp_path / name / "checkpoint" / f"ckpt-{step:06d}.pkl.gz")


def _ems_cap_stage(tmp_path, name):
    """全員が病人・上限 1 件の舞台(既存 ``test_calls_are_capped_per_day`` と同じ作り)。"""
    sim = _sim(tmp_path, name, n_steps=1, n_agents=40,
               **{**ON, **HEALTH_ON, "city_ops.ems.max_calls_per_day": "1",
                  "city_ops.ems.collapse_per_10k": "10000"})
    node = sorted(sim.city.graph.nodes)[0]
    CO.bind(sim)
    for a in sorted(sim.agents, key=lambda x: int(x.id)):
        a.occupation = "会社員"
        a.sick = True
        _place(sim, a, node)
    return sim


#: 07:00 開始なので step 102 で暦日が変わる(= 同じ日に残っている step の上限)。
_DAY_END_STEP = 102


def _collapse_slots(sim) -> list[int]:
    """全員の「その日の何 step 目に倒れるか」(実装と同じ純関数を素で再現・昇順)。"""
    spd = max(1, int(sim.clock.steps_per_day))
    return sorted(CO._stable_hash(f"collapse_slot/{int(a.id)}/0") % spd
                  for a in sim.agents)


def test_ems_daily_cap_survives_a_checkpoint(tmp_path):
    """★① 昼のうちに上限へ達した日は、resume しても追加の出動が出ない。"""
    from society.engine import checkpoint

    src = _ems_cap_stage(tmp_path, "co_cap_src")
    slots = _collapse_slots(src)
    split = slots[0] + 1                        # 上限に達した**直後**で checkpoint
    rest = [s for s in slots[1:] if split <= s < _DAY_END_STEP]
    assert rest, "同じ日に 2 人目が倒れない舞台(n_agents/split の再調整が要る)"
    for step in range(split):
        CO.phase(src, step, 420 + step * 10)
    assert len(_kind(src, "ems_call")) == 1, "上限に達していない(舞台の前提が崩れた)"
    day_cap = dict(src._co_state["calls_by_day"])
    p = _ckpt(src, tmp_path, "co_cap_src", step=split)

    dst = _ems_cap_stage(tmp_path, "co_cap_dst")
    assert checkpoint.load(dst, p) == split
    assert dict(dst._co_state["calls_by_day"]) == day_cap, "日次カウンタが戻っていない"
    for step in range(split, _DAY_END_STEP):    # ★同じ暦日の残り(2 人目の slot を含む)
        CO.phase(dst, step, 420 + step * 10)
    assert not _kind(dst, "ems_call"), "resume 後に同じ日の出動が二重に出ている"

    # 負の対照: カウンタを捨てれば必ずもう 1 件出る(= この検査は空回りしていない)
    naive = _ems_cap_stage(tmp_path, "co_cap_naive")
    checkpoint.load(naive, p)
    naive._co_state["calls_by_day"] = {}
    for step in range(split, _DAY_END_STEP):
        CO.phase(naive, step, 420 + step * 10)
    assert _kind(naive, "ems_call"), "対照が発火しない(舞台/step 範囲の再調整が要る)"


def test_night_cleaning_guard_survives_a_checkpoint(tmp_path):
    """★② 夜帯(22:00-05:00)の途中で resume しても同じ棟の再清掃が出ない。"""
    from society.engine import checkpoint

    src, worker = _clean_stage(tmp_path, "co_nc_src")
    CO.phase(src, 0, 22 * 60)
    assert len(_kind(src, "night_cleaning")) == 1, "1 件目が出ていない(舞台の前提が崩れた)"
    night_id = int(22 * 60) // 1440
    assert str(worker.city_ops_clean_building) in src._co_state["cleaned_night"][night_id]
    p = _ckpt(src, tmp_path, "co_nc_src", step=1)

    dst, _w = _clean_stage(tmp_path, "co_nc_dst")
    assert checkpoint.load(dst, p) == 1
    for i in range(1, 8):                       # 同じ晩の残り(22:10〜23:10)
        CO.phase(dst, i, 22 * 60 + i * 10)
    assert not _kind(dst, "night_cleaning"), "resume 後に同じ晩の清掃が二重に出ている"
    CO.phase(dst, 200, 1440 + 22 * 60)          # 翌晩は別の晩 = ちゃんと出る
    assert len(_kind(dst, "night_cleaning")) == 1, "翌晩まで抑止されている(過剰抑止)"

    # 負の対照: 1 晩ガードを捨てれば必ずもう 1 件出る
    naive, _w2 = _clean_stage(tmp_path, "co_nc_naive")
    checkpoint.load(naive, p)
    naive._co_state["cleaned_night"] = {}
    CO.phase(naive, 1, 22 * 60 + 10)
    assert len(_kind(naive, "night_cleaning")) == 1


def test_co_state_roundtrips_including_the_bind_ledger(tmp_path):
    """束ねの日次追随(day / bind / bind_by_day / rebinds)も往復で戻る。"""
    from society.engine import checkpoint

    src = _sim(tmp_path, "co_rt_src", n_steps=1, n_agents=12, **ON)
    CO.phase(src, 0, 420)
    st = src._co_state
    assert st["bound"] and st["bind_by_day"], "束ねが走っていない(前提が崩れた)"
    st["calls_by_day"][3] = 7                   # 実ランで出にくい欄も表で押さえる
    st["cleaned_night"][2] = {"bldg-9": 1}
    st["rounds_by_kind"]["ward"] = 5
    p = _ckpt(src, tmp_path, "co_rt_src", step=9)

    dst = _sim(tmp_path, "co_rt_dst", n_steps=1, n_agents=12, **ON)
    assert checkpoint.load(dst, p) == 9
    assert dst._co_state == st, "_co_state が 1 件でも欠けている"
    # ★集合を 1 つも持たない(= pickle の集合反復順非保存の影響を受けない)ことの機械固定
    def _no_set(obj, path="_co_state"):
        assert not isinstance(obj, (set, frozenset)), f"{path} に集合が混ざった"
        if isinstance(obj, dict):
            for k, v in obj.items():
                _no_set(v, f"{path}[{k!r}]")
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                _no_set(v, f"{path}[{i}]")
    _no_set(dst._co_state)


def test_old_checkpoint_without_co_state_keeps_legacy_behaviour(tmp_path):
    """``co_state`` を持たない旧 checkpoint でも落ちず、従来どおり遅延生成へ落ちる。"""
    import gzip
    import pickle

    from society.engine import checkpoint

    src = _sim(tmp_path, "co_old_src", n_steps=1, n_agents=12, **ON)
    CO.phase(src, 0, 420)
    p = _ckpt(src, tmp_path, "co_old_src", step=4)
    with gzip.open(p, "rb") as f:
        blob = pickle.loads(f.read())
    blob["runtime"].pop("co_state", None)
    with gzip.open(p, "wb") as f:
        f.write(pickle.dumps(blob, protocol=pickle.HIGHEST_PROTOCOL))

    dst = _sim(tmp_path, "co_old_dst", n_steps=1, n_agents=12, **ON)
    assert checkpoint.load(dst, p) == 4
    assert getattr(dst, "_co_state", None) is None      # 旧 ckpt = 従来どおり
