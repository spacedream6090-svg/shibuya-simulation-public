"""夜間開放(Wave 4 III-1・world.night_economy。既定 OFF)= 夜を表現可能にする層のテスト。

方針(既存の鉄則を継承):
- OFF(既定): night_refuge 0 件・agent に属性が生えない・営業時間表が 1 エントリも変わらず・
  勤務窓の丸めが従来コードのまま・イベント列は純粋既定と L1 完全一致(ゴールデンを守る)。
- 日跨ぎシフト ON: 22:00→06:00 が表現でき(work._window)、勤務窓判定が円環になる
  (routine.in_work_window: 23:00 True / 03:00 True / 12:00 False)。
- 営業時間 ON: subcat(コンビニ等)の 24 時間営業が引き当てられる。**現行地図 v7 は subcat を
  持たない**ので、実データでは no-op であることも明示的に固定する(正直な限界の機械化)。
- 避難先 ON: 終電後の homing が縁のゲートウェイでなく徒歩圏の営業中 POI へ移り、始発で帰る。
  選定は乱数ゼロの純関数 = 2 回走らせて L1 完全一致。
- 生成器(scripts/build_orgs.py / build_persona_pool.py): 夜勤語彙・L2 継承・就寝時刻ずらし。
  データファイルは .gitignore の生成物なので、**合成入力に対する生成関数の単体テスト**で固定する。
検証は mock のみ(実LLM 禁止)。
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from society import commerce, night, work
from society.cognition import routine
from society.config import load_config
from society.engine import scheduler
from society.engine.simulation import Simulation

REPO_ROOT = Path(__file__).resolve().parents[1]
_ON = {"world.night_economy.enabled": "true"}


def _load_script(name: str):
    """scripts/<name>.py を module として読む(scripts はパッケージではないので直読み)。"""
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_script_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _sim(tmp_path, name, n=30, steps=1, **ov):
    dot = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
           "observer.snapshot_every=1"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _kind(sim, kind):
    return [e for e in sim.logger.events if e.kind == kind]


class _FakeAgent:
    """勤務窓判定だけを見る最小のスタブ(Agent の全フィールドは要らない)。"""

    def __init__(self, start, end, wraps=False):
        self.work_start_min = start
        self.work_end_min = end
        self.sick = False
        if wraps:
            self.work_wraps = True


# ===================================================================== OFF 既定
def test_off_matches_pure_default(tmp_path):
    """明示 OFF と純粋既定が L1 完全一致(144 step)。夜間層の新イベントは 1 件も出ない。"""
    pure = _sim(tmp_path, "pure", steps=144)
    pure.run()
    off = _sim(tmp_path, "expl_off", steps=144,
               **{"world.night_economy.enabled": "false"})
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(III-1 seam が no-op でない)"
    assert not _kind(pure, "night_refuge"), "OFF で night_refuge が出ている"


def test_off_leaves_no_attributes_and_no_table_change(tmp_path):
    """OFF: agent に夜間層の属性が 1 つも生えず、営業時間表も同一オブジェクトのまま。"""
    sim = _sim(tmp_path, "off_attrs", steps=1)
    sim.run()
    for a in sim.agents:
        assert not hasattr(a, "work_wraps"), "OFF なのに work_wraps が生えている"
        assert not hasattr(a, "_night_refuge_node"), "OFF なのに避難先属性が生えている"
    assert night.cat_hours(sim.commercecfg, sim.nightcfg) is sim.commercecfg["hours"], \
        "OFF でカテゴリ営業時間表が作り直されている(同一オブジェクトでない)"
    assert sim.workcfg["bind_workplace"]["midnight_shift"] is False


# ===================================================================== ① 日跨ぎシフト
def test_window_clamp_is_literal_when_off():
    """OFF: 22:00→06:00 は従来どおり open+8h(=1800 分)へ丸められ、日跨ぎにならない。"""
    bcfg = work.build_cfg(None)["bind_workplace"]
    assert work._window(bcfg, {"open": "22:00", "close": "06:00"}, {}) == (1320, 1800, False)
    # 正常な日勤は ON/OFF で同一
    assert work._window(bcfg, {"open": "09:00", "close": "18:00"}, {}) == (540, 1080, False)


def test_window_crosses_midnight_when_on():
    """ON: close<open は「翌朝まで」= (1320, 360, True)。close==open は退化なので従来の丸め。"""
    wcfg = work.build_cfg(None)
    night.wire_work(night.build_cfg({"enabled": True}), wcfg)
    bcfg = wcfg["bind_workplace"]
    assert bcfg["midnight_shift"] is True
    assert work._window(bcfg, {"open": "22:00", "close": "06:00"}, {}) == (1320, 360, True)
    assert work._window(bcfg, {"open": "20:00", "close": "04:00"}, {}) == (1200, 240, True)
    assert work._window(bcfg, {"open": "09:00", "close": "18:00"}, {}) == (540, 1080, False)
    assert work._window(bcfg, {"open": "09:00", "close": "09:00"}, {}) == (540, 1020, False)
    # プール record の shift_pattern 経路(台帳 entry が無い層)でも同じ読み方になる
    assert work._window(bcfg, None,
                        {"shift_pattern": {"open": "23:00", "close": "07:00"}}) \
        == (1380, 420, True)
    # midnight_shift だけ切ると従来の丸めへ戻る(子トグルの独立性)
    wcfg2 = work.build_cfg(None)
    night.wire_work(night.build_cfg({"enabled": True, "midnight_shift": False}), wcfg2)
    assert work._window(wcfg2["bind_workplace"],
                        {"open": "22:00", "close": "06:00"}, {}) == (1320, 1800, False)


def test_in_work_window_wraps_over_midnight():
    """夜勤(22:00-06:00)の勤務窓判定: 23:00 True / 03:00 True / 12:00 False / 21:00 False。"""
    a = _FakeAgent(22 * 60, 6 * 60, wraps=True)
    assert routine.in_work_window(a, 23 * 60), "23:00 に夜勤者が勤務時間帯でない"
    assert routine.in_work_window(a, 3 * 60), "03:00(翌朝)に夜勤者が勤務時間帯でない"
    assert routine.in_work_window(a, 5 * 60 + 50), "05:50 に夜勤者が勤務時間帯でない"
    assert not routine.in_work_window(a, 12 * 60), "正午に夜勤者が勤務時間帯扱い"
    assert not routine.in_work_window(a, 21 * 60), "21:00(出勤前)に勤務時間帯扱い"
    assert not routine.in_work_window(a, 6 * 60), "06:00(退勤時刻ちょうど)に勤務時間帯扱い"


def test_in_work_window_day_shift_unchanged():
    """work_wraps が無い個体は従来式そのまま(属性不在=既定 False=バイト一致の根拠)。"""
    a = _FakeAgent(9 * 60, 18 * 60)
    assert not hasattr(a, "work_wraps")
    assert routine.in_work_window(a, 12 * 60)
    assert not routine.in_work_window(a, 3 * 60)
    assert not routine.in_work_window(a, 22 * 60)


def test_bind_workplace_sets_work_wraps_only_when_on(tmp_path):
    """work.bind_workplace が夜勤台帳を読んだときだけ work_wraps を立てる(OFF は立てない)。"""
    sim = _sim(tmp_path, "bindnight", steps=1)
    agent = sim.agents[0]
    agent.work_start_min = -1
    agent.work_node = ""
    book = {"o1": {"node": None, "building": None, "floor": None, "poi_id": None,
                   "cat": "service", "open": "22:00", "close": "06:00"}}
    rec = {"org_id": "o1", "occupation": "会社員"}
    wcfg = work.build_cfg(None)
    night.wire_work(night.build_cfg({"enabled": True}), wcfg)
    work.bind_workplace(agent, rec, sim.city, book, wcfg["bind_workplace"])
    assert (agent.work_start_min, agent.work_end_min) == (1320, 360)
    assert getattr(agent, "work_wraps", False) is True
    # OFF 経路は属性を立てず、丸められた窓になる
    a2 = sim.agents[1]
    a2.work_start_min = -1
    a2.work_node = ""
    work.bind_workplace(a2, rec, sim.city, book, work.build_cfg(None)["bind_workplace"])
    assert (a2.work_start_min, a2.work_end_min) == (1320, 1800)
    assert not hasattr(a2, "work_wraps")


def test_night_commuter_arrives_before_the_shift(tmp_path):
    """夜勤者の通勤の向きが直る: 到着は出勤 lead 分前・帰宅トリガは勤務窓の外(退勤後)。

    ★これが無いと ``persona.build_agent`` の後退値(職場不明 → 到着 08:30)のまま朝に街へ入り、
      就寝(帰宅)トリガに触れて即帰る = 夜勤者が夜に 1 人も居ないという実質 no-op になる。"""
    sim = _sim(tmp_path, "nightcommute", steps=1)
    wcfg = work.build_cfg(None)
    night.wire_work(night.build_cfg({"enabled": True}), wcfg)
    book = {"o1": {"node": None, "building": None, "floor": None, "poi_id": None,
                   "cat": "service", "open": "22:00", "close": "06:00"}}
    a = sim.agents[0]
    a.work_start_min = -1
    a.work_node = ""
    a.commute = True
    a.arrival_min = 8 * 60 + 30                 # 職場不明時の後退値(朝の流入)
    a.bedtime_min = 23 * 60                     # 勤務窓のまっただ中
    work.bind_workplace(a, {"org_id": "o1", "arrival_lead_min": 40}, sim.city,
                        book, wcfg["bind_workplace"])
    assert a.arrival_min == 22 * 60 - 40, "出勤 lead 分前に到着していない"
    assert a.bedtime_min == 6 * 60 + 30, "帰宅トリガが勤務窓の中に残っている"
    assert not routine.in_work_window(a, a.bedtime_min), "帰宅トリガの時刻がまだ勤務中"
    # 生成器が既に退勤後へずらしてある個体の個体差は潰さない
    b = sim.agents[2]
    b.work_start_min = -1
    b.work_node = ""
    b.commute = True
    b.bedtime_min = 7 * 60 + 20
    work.bind_workplace(b, {"org_id": "o1", "arrival_lead_min": 40}, sim.city,
                        book, wcfg["bind_workplace"])
    assert b.bedtime_min == 7 * 60 + 20, "生成器がずらした就寝時刻を上書きした"


def test_night_workers_are_actually_at_work_at_23_and_03(tmp_path):
    """統合: 夜勤窓を持つ個体は 23:00 も 03:00 も**職場で働いている**(夜が空でなくなる)。

    単体(in_work_window)だけでは「routine が実際に出勤させるか」「途中で寝ないか」が
    判らないので、20:00 開始の mock ランで実際の在場を見る。"""
    book = {"o1": {"node": None, "building": None, "floor": None, "poi_id": None,
                   "cat": "service", "open": "22:00", "close": "06:00"}}
    for steps, label in ((18, "23:00"), (42, "03:00")):
        sim = _sim(tmp_path, f"nw{steps}", n=20, steps=steps,
                   **{**_ON, "run.start_tod": "20:00"})
        picked = []
        for a in sim.agents:
            if a.visitor:
                continue
            a.work_start_min = -1
            a.work_node = ""
            a.sleeping = False
            work.bind_workplace(a, {"org_id": "o1"}, sim.city, book,
                                sim.workcfg["bind_workplace"])
            if getattr(a, "work_wraps", False):
                picked.append(a.id)
        assert picked, "夜勤窓を持つ個体が 1 人も作れていない"
        sim.run()
        by_id = {a.id: a for a in sim.agents}
        working = [i for i in picked if by_id[i].activity == "working"]
        sleeping = [i for i in picked if by_id[i].sleeping]
        assert not sleeping, f"{label} に夜勤者が寝ている: {sleeping}"
        assert len(working) >= len(picked) - 1, \
            f"{label} に働いている夜勤者が {len(working)}/{len(picked)} しか居ない"


# ===================================================================== ② 営業時間
def test_hours_override_convenience_is_24h():
    """ON + subcat=convenience の POI は 03:00 でも開いている(24 時間営業 = 開==閉)。"""
    ccfg = commerce.build_cfg(None)
    ncfg = night.build_cfg({"enabled": True})
    conv = {"cat": "shop", "subcat": "convenience", "name": "夜の店"}
    assert commerce.is_open_poi(ccfg, conv, 3 * 60, ncfg), "ON でコンビニが 03:00 に閉じている"
    assert commerce.is_open_poi(ccfg, conv, 15 * 60, ncfg)
    # 同じ POI を OFF(ncfg=None)で見ると汎用 shop の 10-21 に戻る
    assert not commerce.is_open_poi(ccfg, conv, 3 * 60), "OFF で subcat が効いている"
    assert not commerce.is_open_poi(ccfg, conv, 22 * 60)


def test_hours_off_is_byte_identical_to_category_table():
    """OFF: どの POI も従来のカテゴリ表と 1 ビットも違わない判定(全カテゴリ×全 step)。"""
    ccfg = commerce.build_cfg(None)
    off = night.build_cfg(None)
    cats = ["food", "shop", "nightlife", "office", "service", "leisure", "hotel"]
    for cat in cats:
        poi = {"cat": cat, "name": "x"}
        for m in range(0, 1440, 10):
            base = commerce.is_open(ccfg, cat, m)
            assert commerce.is_open_poi(ccfg, poi, m) is base
            assert commerce.is_open_poi(ccfg, poi, m, off) is base, \
                f"OFF の夜間 cfg が判定を変えている({cat} @ {m})"


def test_subcat_is_never_guessed_by_default():
    """既定では POI 名から subcat を推測しない(ETHICS: ブランド名を書かない・誤判定を作らない)。

    v7 実データで一般名詞マッチをやると Apple Store 等をコンビニと誤判定するので、
    ``subcat_keywords`` が空の間は**必ず None** になることを機械で固定する。"""
    ncfg = night.build_cfg({"enabled": True})
    assert ncfg["subcat_keywords"] == {}
    for name in ["Apple Store", "ABCマート", "東急ストア", "コンビニ", "ヤマザキデイリーストア"]:
        assert night.poi_subcat(ncfg, {"cat": "shop", "name": name}) is None
    # 運用者が明示したときだけ効く(POI 自身の subcat は常に優先)
    kw = night.build_cfg({"enabled": True, "subcat_keywords": {"convenience": ["夜間商店"]}})
    assert night.poi_subcat(kw, {"cat": "shop", "name": "夜間商店 神南"}) == "convenience"
    assert night.poi_subcat(kw, {"cat": "shop", "subcat": "sauna", "name": "x"}) == "sauna"


def test_v7_map_has_no_subcat_so_overrides_are_documented_noop():
    """★正直な限界の機械化: 現行地図(v7 / 既定地図)の POI は subcat を 1 件も持たない。

    したがって既定の convenience / net_cafe / sauna / karaoke / club のエントリは
    **今は 1 件も当たらない**(地図 v8 待ち)。この事実が変わったら(= v8 が入ったら)
    このテストが落ちて、conf のコメントを更新する契機になる。"""
    ncfg = night.build_cfg({"enabled": True})
    for path in ("data/shibuya_osm.json", "data/shibuya_osm_wide_v7.json"):
        doc = json.loads((REPO_ROOT / path).read_text(encoding="utf-8"))
        subs = {night.poi_subcat(ncfg, p) for p in doc["pois"]}
        assert subs == {None}, f"{path} に subcat を持つ POI がある(v8 到来? conf の注記を更新すること)"


def test_cat_level_override_applies_when_on(tmp_path):
    """cat に当たるキー(nightlife)は ON で実効する(subcat と違い v7 でも効く)。"""
    ccfg = commerce.build_cfg(None)
    ncfg = night.build_cfg({"enabled": True, "hours": {"nightlife": [20, 2]}})
    merged = night.cat_hours(ccfg, ncfg)
    assert merged["nightlife"] == (20 * 60, 2 * 60)
    assert merged["food"] == ccfg["hours"]["food"], "上書きしていないカテゴリが動いた"
    poi = {"cat": "nightlife", "name": "x"}
    assert not commerce.is_open_poi(ccfg, poi, 19 * 60, ncfg)
    assert commerce.is_open_poi(ccfg, poi, 21 * 60, ncfg)
    assert commerce.is_open_poi(ccfg, poi, 1 * 60, ncfg)


# ===================================================================== ③ 終電後の避難先
def _strand(sim, agent):
    """終電後に駅で帰路を失った個体の状態を作る(homing で駅に居る)。"""
    agent.sleeping = False
    agent.loc = "street"
    agent.building = None
    agent.node = sim.city.station_node
    agent.x, agent.y = sim.city.node_xy(agent.node)
    agent.route = []
    agent.homing = True
    agent.exit_intent = True
    return agent


def test_refuge_only_when_on(tmp_path):
    """OFF: take_refuge は False で、_try_exit は従来どおり縁のゲートウェイへ歩かせる。"""
    off = _sim(tmp_path, "ref_off", steps=1)
    a = _strand(off, off.agents[0])
    assert night.take_refuge(off, a, 0, 25 * 60) is False
    scheduler._try_exit(off, a, 0, 25 * 60)
    assert not getattr(a, "_night_refuge_node", ""), "OFF で避難先が設定された"
    assert not _kind(off, "night_refuge")


def test_refuge_takes_stranded_agent_to_an_open_poi(tmp_path):
    """ON: 終電後の homing は営業中の避難先へ向かい、始発まで滞在して駅へ戻る経路を持つ。"""
    sim = _sim(tmp_path, "ref_on", steps=1, **_ON)
    assert not sim.transit.has_service(25 * 60), "テスト前提(01:00 は終電後)が崩れている"
    a = _strand(sim, sim.agents[0])
    assert night.take_refuge(sim, a, 0, 25 * 60) is True
    node = a._night_refuge_node
    assert node and node != sim.city.station_node, "避難先が駅ノード(= 避難していない)"
    assert a.route and a.exit_intent is False and a.homing is False, "移動中の状態が不正"
    # 到着 → チェックイン
    a.node = node
    a.route = []
    a.x, a.y = sim.city.node_xy(node)
    night.on_arrive(sim, a, 1, 25 * 60)
    ev = _kind(sim, "night_refuge")
    assert len(ev) == 1, "night_refuge が 1 件でない"
    assert ev[0].payload["node"] == node and ev[0].payload["first_train"] is True
    assert a.sleeping is True and a.exit_intent is True and a.homing is True
    assert a.route, "始発後に駅へ戻る経路が張られていない"
    assert not getattr(a, "_night_refuge_node", ""), "チェックイン後も避難先フラグが残っている"
    # until_min = 始発(04:34 の次の Δt 刻み)。sleep_until はその step
    wait, found = night.next_service_min(sim, 25 * 60, sim.nightcfg["refuge"]["max_stay_min"])
    assert found and ev[0].payload["until_min"] == 25 * 60 + wait
    assert a.sleep_until == 1 + max(1, sim.clock.min_to_steps(wait))


def test_refuge_agent_leaves_at_first_train(tmp_path):
    """保存則: 避難した個体は**始発の時刻に**起きる(街に固まらない)。

    起床時刻の妥当性は既存の運行述語で検算する: その 1 step 前には運行が無く、
    起床時刻には運行がある。"""
    sim = _sim(tmp_path, "ref_first", steps=1, **_ON)
    a = _strand(sim, sim.agents[0])
    assert night.take_refuge(sim, a, 0, 25 * 60)
    a.node = a._night_refuge_node
    a.route = []
    night.on_arrive(sim, a, 1, 25 * 60)
    until = _kind(sim, "night_refuge")[0].payload["until_min"]
    assert sim.transit.has_service(until), "起床時刻に運行が無い(始発でない)"
    assert not sim.transit.has_service(until - sim.clock.step_minutes), \
        "1 step 前にも運行がある(始発より遅く起きている)"


def test_refuge_stay_is_capped_when_service_never_returns(tmp_path):
    """運休(災害)で 1 日中サービスが無くても max_stay_min で必ず打ち切る(発散対策の安全弁)。"""
    sim = _sim(tmp_path, "ref_cap", steps=1, **_ON)
    sim.transit.suspended = True
    wait, found = night.next_service_min(sim, 25 * 60, sim.nightcfg["refuge"]["max_stay_min"])
    assert found is False and wait == sim.nightcfg["refuge"]["max_stay_min"]


def test_refuge_choice_is_deterministic_and_draws_no_randomness(tmp_path):
    """避難先の選定は (距離, node, poi id) の純関数 = 同条件 2 回で完全一致・乱数を引かない。"""
    a_sim = _sim(tmp_path, "ref_d1", steps=1, **_ON)
    b_sim = _sim(tmp_path, "ref_d2", steps=1, **_ON)
    picks = []
    for sim in (a_sim, b_sim):
        agent = _strand(sim, sim.agents[0])
        poi = night.nearest_refuge(sim, agent, 25 * 60)
        assert poi is not None, "既定地図に営業中の避難先が無い(テスト前提が崩れている)"
        picks.append((poi["id"], poi["node"]))
    assert picks[0] == picks[1], "避難先の選定が非決定"
    # 実装に乱数の識別子が存在しないことを AST で固定(traces / devices と同じ流儀)
    import ast
    src = (REPO_ROOT / "src" / "society" / "night.py").read_text(encoding="utf-8")
    names = {n.id for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Name)}
    attrs = {n.attr for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Attribute)}
    for bad in ("rng", "random", "hub", "stream", "integers", "shuffle"):
        assert bad not in names and bad not in attrs, f"night.py に乱数の識別子 {bad} がある"


def _night_run(tmp_path, name, on=True, steps=40):
    """深夜 01:00 開始(終電後)。8 人を駅へ向かわせて立ち往生させる 40 step の mock ラン。"""
    ov = dict(_ON) if on else {"world.night_economy.enabled": "false"}
    sim = _sim(tmp_path, name, n=24, steps=steps, **{**ov, "run.start_tod": "01:00"})
    assert not sim.transit.has_service(60), "テスト前提(01:00 は終電後)が崩れている"
    for a in sim.agents[:8]:
        a.sleeping = False
        a.loc = "street"
        a.building = None
        path, mode = sim.router.route(a.node, sim.city.station_node, "walk")
        if len(path) < 2:
            continue
        a.route = path[1:]
        a.edge_offset = 0.0
        a.trip_mode = mode
        a.homing = True
        a.exit_intent = True
    sim.run()
    return sim


def test_refuge_run_is_deterministic(tmp_path):
    """ON 同士 2 回のランで L1 完全一致(避難が実際に起きていることも同時に固定)。"""
    a = _night_run(tmp_path, "nrun1")
    b = _night_run(tmp_path, "nrun2")
    assert _kind(a, "night_refuge"), "テストが空振り(避難が 1 件も起きていない)"
    assert _l1(a) == _l1(b), "ON のランが非決定"


def test_refuge_run_conserves_agents_and_needs_the_gate(tmp_path):
    """保存則: 避難した個体は全員、始発で起きて街から出る(夜に固まらない)。OFF は 0 件。"""
    on = _night_run(tmp_path, "ncons_on")
    refugees = {e.agent_id for e in _kind(on, "night_refuge")}
    assert refugees, "避難が 1 件も起きていない"
    by_id = {a.id: a for a in on.agents}
    for aid in refugees:
        a = by_id[aid]
        assert not a.sleeping, f"agent {aid} が始発後も眠ったまま"
        assert a.loc == "outside", f"agent {aid} が街から出ていない(帰路に乗れていない)"
    woke = {e.agent_id for e in _kind(on, "wake_up")}
    assert refugees <= woke, "避難した個体に wake_up が出ていない"
    off = _night_run(tmp_path, "ncons_off", on=False)
    assert not _kind(off, "night_refuge"), "OFF で避難が起きている"


# ===================================================================== ④ 生成器(夜勤者)
def test_build_orgs_night_vocabulary():
    """夜勤語彙: 該当業種 × 十分な規模の社にだけ night_shift が付く(決定論・乱数ゼロ)。"""
    bo = _load_script("build_orgs")
    assert set(bo.NIGHT_SHIFT_BY_KEY) <= {it["key"] for it in bo.INDUSTRY_SPEC}
    for key, spec in bo.NIGHT_SHIFT_BY_KEY.items():
        assert spec["close"] < spec["open"], f"{key} の夜勤が日跨ぎでない(文字列比較でも成立する形)"
        assert spec["share"] > 0.0 and spec["min_employees"] >= 1 and spec["roles"]
    assert bo.night_shift_for("SV", 50)["open"] == "22:00"
    assert bo.night_shift_for("SV", 1) is None, "1 人の事業所に夜勤枠を作っている"
    assert bo.night_shift_for("IT", 500) is None, "オフィス系に夜勤枠を作っている"


def test_build_orgs_night_flag_is_opt_in_and_changes_nothing_else():
    """--night-shifts なしの台帳は従来と完全同一。ON でも増えるのは night_shift キー 1 つだけ。"""
    bo = _load_script("build_orgs")
    base = bo.build_companies_dist(set(), 300, 7, night_shifts=False)
    on = bo.build_companies_dist(set(), 300, 7, night_shifts=True)
    assert len(base) == len(on)
    assert all("night_shift" not in c for c in base), "既定 OFF で night_shift が生えている"
    assert any("night_shift" in c for c in on), "ON でも夜勤枠が 1 社も無い"
    for b, o in zip(base, on):
        assert b == {k: v for k, v in o.items() if k != "night_shift"}, \
            "夜勤枠の有無で他のフィールドが動いた(乱数消費順が変わっている)"


def test_pool_night_slot_count_matches_the_ledger_generator():
    """写経(build_persona_pool.night_slot_count)が台帳側の式と一致することを機械で固定。"""
    bo = _load_script("build_orgs")
    bp = _load_script("build_persona_pool")
    for share in (0.0, 0.1, 0.18, 0.3, 1.0):
        co = {"night_shift": {"share": share}}
        for k in (0, 1, 3, 5, 10, 37, 100):
            assert bp.night_slot_count(co, k) == bo.night_slot_count(co, k)
    assert bp.night_slot_count({}, 100) == 0, "night_shift の無い社に夜勤枠ができている"


def test_pool_L2_inherits_night_shift_and_shifts_bedtime(tmp_path):
    """L2 スロット: 夜勤枠は末尾から確保され、専用 role と日跨ぎ shift_pattern を継ぐ。"""
    bp = _load_script("build_persona_pool")
    night_co = {"id": "co_sv_1", "size": {"employees": 10},
                "roles": ["オペレーション", "事務"],
                "shift_pattern": {"open": "09:00", "close": "19:00", "days": "mon-sat",
                                  "rotates": True},
                "night_shift": {"open": "22:00", "close": "06:00", "days": "all",
                                "shift_hours": 8, "rotates": True, "share": 0.30,
                                "roles": ["夜間清掃", "設備巡回"]}}
    day_co = {k: v for k, v in night_co.items() if k != "night_shift"}
    orgs = {"companies": [night_co], "schools": []}
    slots = bp._build_L2_slots(orgs, 1.0)
    assert len(slots) == 10
    n_night = sum(1 for s in slots if bp.is_night_shift(s[3]))
    assert n_night == 3, "夜勤枠が share の丸め(10×0.30)と一致しない"
    assert all(bp.is_night_shift(s[3]) for s in slots[-3:]), "夜勤枠が末尾から確保されていない"
    assert {s[1] for s in slots[-3:]} <= {"夜間清掃", "設備巡回"}
    # 日勤側の並びは night_shift の有無で 1 バイトも動かない
    base = bp._build_L2_slots({"companies": [day_co], "schools": []}, 1.0)
    assert base[:7] == slots[:7], "夜勤枠を足したら日勤スロットが動いた"
    assert all(not bp.is_night_shift(s[3]) for s in base)


def test_pool_L2_night_bedtime_is_after_the_shift_ends(tmp_path):
    """夜勤者の bedtime_min(= 帰宅トリガ)が退勤(06:00)の後に来る。決定論・rng 非消費。"""
    bp = _load_script("build_persona_pool")
    sp = {"open": "22:00", "close": "06:00"}
    vals = {bp.night_bedtime_min(sp, i) for i in range(12)}
    assert vals == {6 * 60 + 30 + d * 10 for d in range(6)}, "就寝時刻の散らし方が決定論でない"
    assert all(6 * 60 < v < 8 * 60 for v in vals), "夜勤者の就寝時刻が退勤より前"
    assert bp.night_bedtime_min(sp, 3) == bp.night_bedtime_min(sp, 9), "i の純関数でない"
    assert bp.is_night_shift({"open": "09:00", "close": "18:00"}) is False


def test_pool_gen_L2_writes_night_records(tmp_path):
    """gen_L2: 夜勤スロットは日跨ぎ shift_pattern + 朝の bedtime_min + 夜勤のペルソナ文を書く。"""
    bp = _load_script("build_persona_pool")
    night_co = {"id": "co_sv_1", "size": {"employees": 10}, "roles": ["オペレーション"],
                "shift_pattern": {"open": "09:00", "close": "19:00", "days": "mon-sat",
                                  "rotates": True},
                "night_shift": {"open": "22:00", "close": "06:00", "days": "all",
                                "shift_hours": 8, "rotates": True, "share": 0.30,
                                "roles": ["夜間清掃"]}}
    slots = bp._build_L2_slots({"companies": [night_co], "schools": []}, 1.0)
    w = bp.ShardWriter(tmp_path / "pool", "L2")
    bp.gen_L2(w, slots, 42)
    w.close()
    recs = [json.loads(ln) for ln in
            (tmp_path / "pool" / "L2" / "part-0000.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(recs) == 10
    night_recs = [r for r in recs if r["shift_pattern"]["open"] == "22:00"]
    day_recs = [r for r in recs if r["shift_pattern"]["open"] == "09:00"]
    assert len(night_recs) == 3 and len(day_recs) == 7
    for r in night_recs:
        assert r["shift_pattern"]["close"] == "06:00"
        assert 6 * 60 < r["bedtime_min"] < 8 * 60, "夜勤者の bedtime が朝になっていない"
        assert "夜勤で通勤している" in r["persona"]
        assert r["occupation"] == "夜間清掃"
    for r in day_recs:
        assert 16 * 60 <= r["bedtime_min"] <= 23 * 60 + 50
        assert "夜勤" not in r["persona"]
