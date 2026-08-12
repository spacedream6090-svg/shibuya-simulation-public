"""事件レイヤー H5 = 環境側 3 族(火災 / 交通 / 群集。``incidents_env``)のテスト。

正典
  - docs/plans/body-incident-layer-plan.md §3(火災 = 薄いレート + 完全アクター連鎖 /
    交通 = 曝露の積・被害者は実在の横断中エージェント / 群集 = **生成しない = 状態が事件**)
  - 同 §7(**事件インフレ = 災害映画化**の防止 = 重度分布を頻度と**同時に**較正する /
    分母の罠 / 年齢構造 / **管轄混同**)
  - 同 §6-1(死は H1 の管轄・尊厳規約)

守るもの(検収基準の順)
  ① OFF(既定)= 純粋既定と L1 バイト一致・新 7 種 0 件・state も属性も生えない・
     乱数 stream を 1 本も引かない
  ② 較正: **重度分布が頻度と同時に固定されている**(ぼや 175 : 全焼 0 / 死亡 0)・
     器具帰属 73%・管轄 → 区の換算が conf に出ている・規模比は地図 bbox の純関数
  ③ 火災 = 完全アクター連鎖: 出火 → 第一発見者 → 119 → 出場 → 鎮火。
     **誰も見つけなければ通報も出場も起きない**
  ④ 負傷 → **既存 kind(injury / ems_call / ems_dispatch)で救急連鎖に乗る**
     (city_ops.py を 1 バイトも編集していないこと・依存ヘルパが実在すること)
  ⑤ 交通 = 曝露の積だけが引き金・**被害者は実在の横断中エージェント**・車が無ければ 0 件
  ⑥ 群集 = 閾値跨ぎそのものが事件・1 エピソード 1 件・**観測が状態を動かさない**・
     物理ゾーンが無いランでは 0 件(測っていない量を捏造しない)
  ⑦ L1 は 1 行 + **前兆状態**を payload 同梱(内生性の機械検証)
  ⑧ ON 同 seed 2 ラン一致 / resume == straight
  ⑨ ★静的検査: generate() を呼ばない・乱数は新 stream 2 本だけ・凍結 14 ファイル無変更
"""
from __future__ import annotations

import ast
import copy
import json
import re
from pathlib import Path

import pyarrow.parquet as pq

from society import city_ops as CO
from society import health as HEALTH
from society import incidents_env as IE
from society import registry as R
from society.config import load_config
from society.engine import checkpoint, scheduler
from society.engine.simulation import Simulation
from society.observer import causality as C
from society.observer.schema import EVENT_KINDS

MODULE = Path(IE.__file__)
REPO = Path(__file__).resolve().parents[1]

#: 本 module が出す L1 種(すべて材料側 registration)
NEW_KINDS = ("fire_start", "fire_report", "fire_dispatch", "fire_out",
             "traffic_accident", "crowd_density_incident", "injury")

OFF = {"incidents_env.enabled": "false"}
ON = {"incidents_env.enabled": "true"}

#: 火災を必ず起こす設定(規模比 1.0 × 管轄アンカーを 1 日 144 件へ = 1 件/step)
FIRE_ON = {**ON, "incidents_env.area_share": "1.0",
           "incidents_env.fire.jurisdiction_per_day": "144",
           "incidents_env.traffic.enabled": "false",
           "incidents_env.crowd.enabled": "false"}


# --------------------------------------------------------------------------- #
# 共通ヘルパ(test_city_ops.py / test_traces.py と同型)
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


def _place(sim, agent, node, *, building="", route=()):
    """個体をノードへ立たせる(位置確定後の世界を手で作る)。"""
    agent.node = str(node)
    agent.x, agent.y = sim.city.node_xy(node)
    agent.loc = "street"
    agent.building = str(building)
    agent.sleeping = False
    agent.route = list(route)
    return agent


def _exile(sim, keep=()):
    """keep 以外の全員を範囲外へ(= 発見者・通報者を意図した者だけにする)。"""
    ids = {int(a.id) for a in keep}
    for a in sim.agents:
        if int(a.id) not in ids:
            a.loc, a.building, a.route = "outside", "", []


# =========================================================================== #
# (A) 出荷既定・宣言・較正アンカー(検収基準 ①の前段・②)
# =========================================================================== #
def test_shipped_default_is_off():
    cfg = load_config()
    assert bool(cfg.incidents_env.enabled) is False
    # 子トグルは親配下で既定 ON(親が OFF なので世界は 1 バイトも動かない)
    assert bool(cfg.incidents_env.fire.enabled) is True
    assert bool(cfg.incidents_env.traffic.enabled) is True
    assert bool(cfg.incidents_env.crowd.enabled) is True
    assert float(cfg.incidents_env.area_share) == 0.0     # 0 = 地図から自動計算


def test_registry_declares_every_toggle():
    feats = {f.id: f for f in R.FEATURES}
    parent = feats["incidents_env.enabled"]
    assert parent.repro_tier == "strict"
    assert parent.affects_k is False        # generate() の呼び出しサイトを 1 つも足さない
    assert parent.fingerprint_risk == "possible"
    assert parent.off_value is False
    for child in ("incidents_env.fire.enabled", "incidents_env.traffic.enabled",
                  "incidents_env.crowd.enabled"):
        assert feats[child].repro_tier == "strict"
        assert feats[child].off_value is True          # conf 既定が true の子トグル
    assert feats["incidents_env.traffic.signalized_only"].off_value is False


def test_responder_vocabulary_moved_from_src_to_conf_without_changing_it():
    """★第109バッチ D2: 担い手の語を src ハードコードから conf キーへ。**前後同値**。

    (a) 既定(conf 無指定)= module 定数と 1 語も違わない
    (b) conf/config.yaml の宣言値 = module 定数と 1 語も違わない(基底を動かしていない)
    (c) src には**ハードコードした frozenset が 1 つも残っていない**
    """
    empty = IE.build_cfg(None)
    assert empty["fire"]["occupations"] == IE.FIRE_OCCS
    assert empty["crowd"]["guard_occupations"] == IE.GUARD_OCCS
    shipped = IE.build_cfg(load_config().incidents_env)
    assert shipped["fire"]["occupations"] == IE.FIRE_OCCS
    assert shipped["crowd"]["guard_occupations"] == IE.GUARD_OCCS
    text = MODULE.read_text(encoding="utf-8")
    assert "frozenset(FIRE_OCCS)" not in text and "frozenset(GUARD_OCCS)" not in text, \
        "src のハードコードが残っている(conf で差し替えられない)"


def test_responder_vocabulary_is_canonicalised_like_the_other_word_lists():
    """語リストの正準化は既存の作法どおり(空 → 既定へ戻す・str 1 個 → 1 要素・宣言順保存)。"""
    got = IE.build_cfg({"fire": {"occupations": []},
                        "crowd": {"guard_occupations": ""}})
    assert got["fire"]["occupations"] == IE.FIRE_OCCS          # 空は既定へ戻す
    assert got["crowd"]["guard_occupations"] == IE.GUARD_OCCS
    got = IE.build_cfg({"fire": {"occupations": "消防隊員"},
                        "crowd": {"guard_occupations": ["常駐警備", "警備員", "警察官"]}})
    assert got["fire"]["occupations"] == ("消防隊員",)
    assert got["crowd"]["guard_occupations"] == ("常駐警備", "警備員", "警察官")  # 宣言順


def test_fire_crew_is_selected_by_the_conf_vocabulary(tmp_path):
    """★conf の語が実際に出場者の選抜へ届く(既定は名簿に居ない = 正直な unstaffed)。"""
    common = {"incidents_env.fire.max_per_day": "1",
              "incidents_env.fire.discover_radius_m": "1e9"}

    def _one(name, occupation, **ov):
        sim = _fire_sim(tmp_path, name, **{**common, **ov})
        finder, crew = sim.agents[0], sim.agents[1]
        _exile(sim, keep=(finder, crew))
        node = sim.city.station_node or sorted(sim.city.graph.nodes)[0]
        _place(sim, finder, node)
        _place(sim, crew, node)
        crew.occupation = occupation
        crew.work_start_min, crew.work_end_min = 0, 1440
        crew.sick = False
        IE.phase(sim, 0, 0)
        return sim, _kind(sim, "fire_dispatch")

    # (a) 名簿に「消防士」が居ない世界 = 既定のままでは出場者ゼロ(黙って埋めない)
    sim_a, disp_a = _one("ie_occ_default", "設備巡回")
    assert len(disp_a) == 1 and disp_a[0].payload["unstaffed"] is True
    assert sim_a._incenv_state["fire_unstaffed"] == 1
    # (b) conf で名簿に実在する語を指すと、**同じ世界**で出場者が立つ
    sim_b, disp_b = _one("ie_occ_conf", "設備巡回",
                         **{"incidents_env.fire.occupations": "[設備巡回]"})
    assert len(disp_b) == 1 and disp_b[0].payload["unstaffed"] is False
    assert disp_b[0].agent_id == sim_b.agents[1].id


def test_guard_count_is_read_through_the_conf_vocabulary(tmp_path):
    """雑踏警備の頭数も conf の語から数える(**読むだけ** = 世界は 1 バイトも動かない)。"""
    sim = _sim(tmp_path, "ie_guard_occ", n_steps=1, **ON)
    for agent in sim.agents[:3]:
        agent.occupation = "常駐警備"
    assert IE._roster_n(sim, IE.GUARD_OCCS) == 0        # 既定の語は名簿に居ない
    assert IE._roster_n(sim, ("常駐警備",)) == 3
    assert IE.build_cfg({"crowd": {"guard_occupations": ["常駐警備"]}})[
        "crowd"]["guard_occupations"] == ("常駐警備",)


def test_provenance_declares_the_responder_vocabulary_and_its_roster(tmp_path):
    """★第108 縦煙の教訓: 「較正が外れた」のか「名簿に居ない」のかを成果物が申告する。"""
    sim = _sim(tmp_path, "ie_roster", n_steps=1, n_agents=15, **ON)
    prov = IE.provenance(sim)
    assert prov["fire_occupations"] == list(IE.FIRE_OCCS)
    assert prov["guard_occupations"] == list(IE.GUARD_OCCS)
    assert prov["fire_roster"] == 0, "既定名簿に消防士が居る(テストの前提が変わった)"
    sim.agents[0].occupation = "消防士"
    assert IE.provenance(sim)["fire_roster"] == 1


def test_every_new_kind_is_registered_and_classified():
    for kind in NEW_KINDS:
        assert kind in EVENT_KINDS, kind
        assert C.cause_of(kind) in C.CAUSE_TYPES, kind
    # 分類の意味(この 4 つは設計の主張そのものなので固定する)
    assert C.cause_of("fire_report") == C.AGENT       # 通報は**行為**
    assert C.cause_of("fire_dispatch") == C.AGENT     # 出場も**行為**
    assert C.cause_of("crowd_density_incident") == C.PHYSICS  # 状態が事件
    assert C.cause_of("injury") == C.PHYSICS          # 負傷は身体の出来事
    # 交通事故は背景交通(装置)の出来事 = 世界に居ない運転者を捏造しない
    assert C.cause_of("traffic_accident") == C.DEVICE
    assert C.cause_of("traffic_accident") in C.DEVICE_STAMPABLE


def test_fire_severity_is_calibrated_together_with_frequency():
    """★災害映画化の防止: 重度分布のアンカー(ぼや 175 : 全焼 0)が既定で立っている。"""
    cfg = IE.build_cfg(load_config().incidents_env)
    weights = cfg["fire"]["severity_weights"]
    assert weights["total"] == 0.0, "全焼の重みが 0 でない(災害映画化の防止が外れている)"
    total = sum(weights.values())
    assert abs(weights["boya"] / total - 175.0 / 241.0) < 1e-9
    # 実際の抽選(純関数)を [0,1) 全域で走らせて分布を確かめる = 乱数に依存しない検査
    n = 24100
    got = {}
    for i in range(n):
        pick = IE._pick(IE.FIRE_SEVERITIES, weights, i / n)
        got[pick] = got.get(pick, 0) + 1
    assert "total" not in got, "全焼が 1 件でも出た(重み 0 が効いていない)"
    assert abs(got["boya"] / n - 175.0 / 241.0) < 0.01
    # 頻度アンカーも同時に conf へ出ている(管轄 → 区の換算が独立キーである)
    assert cfg["fire"]["jurisdiction_per_day"] == 0.66
    assert cfg["fire"]["jurisdiction_to_ward"] == 1.0    # 少なく見積もる向きの既定


def test_crash_severity_has_no_death_by_default():
    """★死は H1(身体レイヤー)の管轄。本 module は 1 件も作らない(尊厳規約)。"""
    cfg = IE.build_cfg(load_config().incidents_env)
    weights = cfg["traffic"]["severity_weights"]
    assert weights["fatal"] == 0.0
    n = 9900
    got = {IE._pick(IE.CRASH_SEVERITIES, weights, i / n) for i in range(n)}
    assert "fatal" not in got
    assert cfg["traffic"]["jurisdiction_per_day"] == 0.9
    assert cfg["traffic"]["pedestrian_share"] == 0.22    # 人対車両だけが巻き込まれる


def test_appliance_attribution_is_73_percent():
    """★出火原因の 73% が器具使用に帰属可能(実測アンカー)。"""
    cfg = IE.build_cfg(load_config().incidents_env)
    weights = cfg["fire"]["appliance_weights"]
    attributed = sum(weights[k] for k in ("kitchen", "heating", "electrical"))
    assert abs(attributed - 0.73) < 1e-9
    assert abs(weights["none"] - 0.27) < 1e-9
    # 共変量は **73% の内訳だけ**を動かす(非器具の族は動かさない)
    warm = IE._appliance_weights(cfg["fire"], "eating", 1.0)
    cold = IE._appliance_weights(cfg["fire"], "eating", 1.5)
    assert warm["none"] == cold["none"] == weights["none"]
    assert cold["heating"] > warm["heating"]
    assert warm["kitchen"] > IE._appliance_weights(cfg["fire"], "office",
                                                   1.0)["kitchen"]


def test_area_share_is_a_pure_function_of_the_map(tmp_path):
    """規模比は**地図 bbox の純関数**(発明した定数を置かない)。conf 上書きが勝つ。"""
    sim = _sim(tmp_path, "ie_share", n_steps=1, **ON)
    share = IE.area_share(sim)
    assert 0.0 < share < 1.0, share
    # 手計算(渋谷区 15.11 km²・緯度 1 度 111.32 km)と一致する
    lat0, lon0, lat1, lon1 = (float(v) for v in sim.city.meta["bbox"])
    import math
    dlat = abs(lat1 - lat0) * IE.KM_PER_DEG
    dlon = abs(lon1 - lon0) * IE.KM_PER_DEG * math.cos(math.radians((lat0 + lat1) / 2))
    assert abs(share - dlat * dlon / IE.WARD_AREA_KM2) < 1e-9
    over = _sim(tmp_path, "ie_share2", n_steps=1, **ON,
                **{"incidents_env.area_share": "0.5"})
    assert IE.area_share(over) == 0.5


def test_unknown_config_values_degrade_to_defaults():
    cfg = IE.build_cfg({"enabled": True,
                        "fire": {"severity_weights": {"boya": 1.0, "存在しない": 9.9},
                                 "injury_severities": ["half", "でたらめ"],
                                 "burn_steps": {"boya": 0, "無い重度": 5}},
                        "traffic": {"pedestrian_share": -1.0},
                        "crowd": {"levels": [4.0, "x", -2.0], "hysteresis": 5.0}})
    assert set(cfg["fire"]["severity_weights"]) == set(IE.FIRE_SEVERITIES)
    assert cfg["fire"]["severity_weights"]["boya"] == 1.0
    assert cfg["fire"]["injury_severities"] == ("half",)      # 語彙外は捨てる
    assert cfg["fire"]["burn_steps"]["boya"] == 1             # 0 は退化なので 1 へ
    assert "無い重度" not in cfg["fire"]["burn_steps"]
    assert cfg["traffic"]["pedestrian_share"] == 0.0          # 負値は 0 へ
    assert cfg["crowd"]["levels"] == [4.0]                    # 非数・負値は捨てる
    assert cfg["crowd"]["hysteresis"] == 1.0                  # [0,1] へクリップ


# =========================================================================== #
# (B) OFF = 完全 no-op(検収基準 ①)
# =========================================================================== #
def test_off_is_a_pure_noop(tmp_path):
    """既定 OFF は純粋既定と L1 バイト一致・state も属性も stream も生えない。"""
    plain = _sim(tmp_path, "ie_plain", n_steps=24)
    plain.run()
    off = _sim(tmp_path, "ie_off", n_steps=24, **OFF)
    off.run()
    assert _l1(plain) == _l1(off)
    for kind in NEW_KINDS:
        assert _kind(off, kind) == [], kind
    assert getattr(off, "_incenv_state", None) is None
    assert getattr(off, "_incenv_fire_targets", None) is None
    assert IE.provenance(off) is None
    for agent in off.agents:
        assert not hasattr(agent, "incenv_fire_until")
        assert not hasattr(agent, "incenv_fire_home")


def test_off_never_touches_the_new_streams(tmp_path, monkeypatch):
    """OFF では新 stream("incident_fire" / "incident_traffic")を 1 本も引かない。"""
    sim = _sim(tmp_path, "ie_stream", n_steps=24, **OFF)
    seen: list[str] = []
    original = sim.hub.stream

    def spy(*key):
        seen.append(str(key[0]))
        return original(*key)

    monkeypatch.setattr(sim.hub, "stream", spy)
    sim.run()
    assert not [k for k in seen if k.startswith("incident_")], sorted(set(seen))


# =========================================================================== #
# (C) 火災 = 完全アクター連鎖(検収基準 ③)
# =========================================================================== #
def _fire_sim(tmp_path, name, **ov):
    sim = _sim(tmp_path, name, n_steps=1, **{**FIRE_ON, **ov})
    return sim


def test_fire_chain_report_and_dispatch(tmp_path):
    """出火 → 第一発見者 → 119 → 出場。**発見者が居れば必ず 3 段が揃う**。"""
    sim = _fire_sim(tmp_path, "ie_fire", **{
        "incidents_env.fire.max_per_day": "1",
        "incidents_env.fire.discover_radius_m": "1e9"})   # 誰かが必ず見つける
    finder = sim.agents[0]
    crew = sim.agents[1]
    _exile(sim, keep=(finder, crew))
    node = sim.city.station_node or sorted(sim.city.graph.nodes)[0]
    _place(sim, finder, node)
    _place(sim, crew, node)
    crew.occupation = "消防士"
    crew.work_start_min, crew.work_end_min = 0, 1440
    crew.sick = False
    IE.phase(sim, 0, 0)
    starts = _kind(sim, "fire_start")
    assert len(starts) == 1
    reports = _kind(sim, "fire_report")
    assert len(reports) == 1 and reports[0].agent_id in (finder.id, crew.id)
    dispatch = _kind(sim, "fire_dispatch")
    assert len(dispatch) == 1
    assert dispatch[0].payload["unstaffed"] is False
    assert dispatch[0].agent_id == crew.id
    # ★実測アンカー(現着 9 分)は payload に併記される(モデル値と取り違えない)
    assert dispatch[0].payload["reference_min"] == 9.0
    assert dispatch[0].payload["response_min"] is not None
    # 隊は持ち場を現場へ移す(移動は既存 routine が行う)
    assert crew.work_node == starts[0].payload["node"]
    assert int(getattr(crew, "incenv_fire_until", -1)) > 0


def test_fire_nobody_sees_is_never_reported(tmp_path):
    """★誰も見つけなければ通報も出場も起きない(行為が無ければ応答も無い)。"""
    sim = _fire_sim(tmp_path, "ie_fire_blind",
                    **{"incidents_env.fire.max_per_day": "1"})
    _exile(sim)                                   # 全員が範囲外
    IE.phase(sim, 0, 0)
    assert len(_kind(sim, "fire_start")) == 1
    assert _kind(sim, "fire_report") == []
    assert _kind(sim, "fire_dispatch") == []
    assert sim._incenv_state["fire_undiscovered"] == 1


def test_fire_start_payload_carries_precursor_state(tmp_path):
    """★L1 は 1 行 + **前兆状態**(用途・器具種・階数)= 内生性の機械検証。"""
    sim = _fire_sim(tmp_path, "ie_fire_pay",
                    **{"incidents_env.fire.max_per_day": "1"})
    IE.phase(sim, 0, 0)
    payload = _kind(sim, "fire_start")[0].payload
    assert payload["use"] in IE.USES
    assert payload["appliance"] in IE.APPLIANCES
    assert payload["attributed"] == (payload["appliance"] != "none")
    assert payload["severity"] in IE.FIRE_SEVERITIES
    assert int(payload["levels"]) >= 1
    assert sim.city.has_building(payload["building"])


def test_fire_out_follows_the_burn_clock(tmp_path):
    """鎮火は重度ごとの燃焼長のあとに 1 件だけ出る(1 件の火は 1 棟に閉じる)。"""
    sim = _fire_sim(tmp_path, "ie_fire_out", **{
        "incidents_env.fire.max_per_day": "1",
        "incidents_env.fire.severity_weights.boya": "1.0",
        "incidents_env.fire.severity_weights.partial": "0.0",
        "incidents_env.fire.severity_weights.half": "0.0"})
    IE.phase(sim, 0, 0)
    assert _kind(sim, "fire_out") == []
    IE.phase(sim, 1, 10)
    outs = _kind(sim, "fire_out")
    assert len(outs) == 1
    assert outs[0].payload["severity"] == "boya"
    assert outs[0].payload["burn_steps"] == 1
    assert outs[0].payload["injured"] == 0          # ぼやでは負傷者が出ない
    assert sim._incenv_state["live_fires"] == {}


# =========================================================================== #
# (D) 負傷 → 既存の救急連鎖(検収基準 ④)
# =========================================================================== #
def test_city_ops_is_untouched_and_its_seam_exists():
    """★依存する**公開シーム**が実在する / あちらに H5 の痕跡が 1 つも無い。

    シームが消えると本 module の救急連鎖は**黙って 0 件**になるので、ここで固定する
    (rumors / traces が「源イベント種が L1 スキーマに在る」を固定したのと同じ理由)。
    ★H2 レーン(2026-08-10)で ``city_ops`` 側に公開シーム ``request_ems`` が生えたので、
      本 module は private ヘルパ ``_on_duty_crew`` を直接呼ぶのをやめた(挙動は同値)。
      それでも「あちらは H5 を 1 バイトも知らない」= 依存は片方向、という規約は変わらない。
    """
    assert callable(getattr(CO, "request_ems", None)), \
        "city_ops.request_ems が消えた(H5 の救急連鎖が黙って 0 件になる)"
    assert callable(getattr(HEALTH, "on_injury", None)), \
        "health.on_injury が消えた(H5 の負傷が身体へ届かなくなる)"
    for name in ("city_ops.py", "health.py", "chance.py"):
        src = (REPO / "src" / "society" / name).read_text(encoding="utf-8")
        for word in ("incidents_env", "facility_devices"):
            assert word not in src, \
                f"{name} に H5 の痕跡がある(H5 からは**読むだけ**のはず)"


def test_fire_injury_enters_the_existing_ems_chain(tmp_path):
    """負傷 → ``injury`` → ``ems_call``(行為)→ ``ems_dispatch``(既存 kind の再利用)。"""
    ov = {"incidents_env.fire.max_per_day": "1",
          "incidents_env.fire.severity_weights.boya": "0.0",
          "incidents_env.fire.severity_weights.partial": "0.0",
          "incidents_env.fire.severity_weights.half": "1.0",
          "incidents_env.fire.burn_steps.half": "1"}
    # ---- pass 1: どの建物が燃えるかを見る(選定は rng(step) と地図だけの関数)----
    probe = _fire_sim(tmp_path, "ie_probe", **ov)
    IE.phase(probe, 0, 0)
    start = _kind(probe, "fire_start")[0].payload
    building = start["building"]
    assert start["severity"] == "half"
    # ---- pass 2: その建物に在館者を置いて同じ火を起こす ----
    sim = _fire_sim(tmp_path, "ie_injury", **ov)
    victim, bystander = sim.agents[0], sim.agents[1]
    _exile(sim, keep=(victim, bystander))
    node = sim.city.building(building)["entrance"]
    _place(sim, victim, node, building=building)
    _place(sim, bystander, node)
    IE.phase(sim, 0, 0)
    assert _kind(sim, "fire_start")[0].payload["building"] == building
    IE.phase(sim, 1, 10)
    injuries = _kind(sim, "injury")
    assert len(injuries) == 1
    assert injuries[0].payload["victim"] == victim.id
    assert injuries[0].payload["source"] == "fire"
    calls = _kind(sim, "ems_call")
    assert len(calls) == 1 and calls[0].agent_id == bystander.id
    assert calls[0].payload["self_call"] is False
    assert calls[0].payload["patient"] == victim.id
    # city_ops.ems が OFF のランでは救急の実体が世界に無い = 正直な unstaffed マーカー
    dispatch = _kind(sim, "ems_dispatch")
    assert len(dispatch) == 1 and dispatch[0].payload["unstaffed"] is True
    assert dispatch[0].agent_id == -1
    # ★身体の状態は 1 バイトも書き換えない(重症度と搬送は H1/H2 の管轄)
    assert victim.sleeping is False
    assert getattr(victim, "sick", False) is False


def test_ems_dispatch_uses_city_ops_crew_when_available(tmp_path):
    """city_ops.ems が ON なら**あちらの選定規則と復帰経路**にそのまま乗る。"""
    ov = {"incidents_env.fire.max_per_day": "1",
          "incidents_env.fire.severity_weights.boya": "0.0",
          "incidents_env.fire.severity_weights.partial": "0.0",
          "incidents_env.fire.severity_weights.half": "1.0",
          "incidents_env.fire.burn_steps.half": "1",
          "city_ops.enabled": "true"}
    probe = _fire_sim(tmp_path, "ie_probe2", **ov)
    IE.phase(probe, 0, 0)
    building = _kind(probe, "fire_start")[0].payload["building"]
    sim = _fire_sim(tmp_path, "ie_crew", **ov)
    victim, medic = sim.agents[0], sim.agents[1]
    _exile(sim, keep=(victim, medic))
    node = sim.city.building(building)["entrance"]
    _place(sim, victim, node, building=building)
    _place(sim, medic, node)
    medic.occupation = "救急隊員"
    medic.work_start_min, medic.work_end_min = 0, 1440
    medic.sick = False
    IE.phase(sim, 0, 0)
    IE.phase(sim, 1, 10)
    dispatch = _kind(sim, "ems_dispatch")
    assert len(dispatch) == 1
    assert dispatch[0].payload["unstaffed"] is False
    assert dispatch[0].agent_id == medic.id
    # ★印は **city_ops のもの** = 持ち場戻しは city_ops._ems_restore が行う(新経路を作らない)
    assert int(getattr(medic, "city_ops_ems_until", -1)) > 0
    assert hasattr(medic, "city_ops_ems_home")


# =========================================================================== #
# (E) 交通 = 曝露の積(検収基準 ⑤)
# =========================================================================== #
TRAFFIC_ON = {**ON, "incidents_env.fire.enabled": "false",
              "incidents_env.crowd.enabled": "false",
              "incidents_env.traffic.hazard_per_exposure": "1.0"}


def _road_node(sim):
    """車道に接する node(横断部の候補)を 1 つ、決定論で選ぶ。"""
    roads = IE.road_nodes(sim)
    assert roads, "テストが空振り(車道に接する node が地図に無い)"
    return sorted(roads)[0]


def test_traffic_without_vehicles_is_zero(tmp_path):
    """車が 1 台も走っていなければ曝露はゼロ = 事故も 0 件(内生の証明)。"""
    sim = _sim(tmp_path, "ie_tr_zero", n_steps=1, **TRAFFIC_ON)
    walker = sim.agents[0]
    _exile(sim, keep=(walker,))
    node = _road_node(sim)
    _place(sim, walker, node, route=[node])
    sim.traffic.last_n = 0
    IE.phase(sim, 0, 0)
    assert _kind(sim, "traffic_accident") == []


def test_traffic_victim_is_a_real_crossing_agent(tmp_path):
    """★被害者は**実在の横断中エージェント**(抽象的被害者を生成しない)。"""
    sim = _sim(tmp_path, "ie_tr", n_steps=1, **TRAFFIC_ON)
    node = _road_node(sim)
    walkers = [sim.agents[0], sim.agents[1]]
    _exile(sim, keep=walkers)
    for a in walkers:
        _place(sim, a, node, route=[node])
    sim.traffic.enabled = True
    sim.traffic.last_n = 5
    IE.phase(sim, 0, 0)
    crashes = _kind(sim, "traffic_accident")
    assert crashes, "曝露があるのに 1 件も起きていない(テストが空振り)"
    payload = crashes[0].payload
    assert payload["victim"] in {a.id for a in walkers}
    assert crashes[0].agent_id == payload["victim"]
    assert payload["ped_n"] == 2 and payload["veh_n"] == 5
    assert payload["exposure"] == 10.0          # ★前兆状態 = 曝露の積が payload に載る
    assert payload["severity"] in IE.CRASH_SEVERITIES
    assert payload["severity"] != "fatal"
    # 負傷 → 既存の救急連鎖(通報者は同じ場所に居合わせたもう 1 人)
    assert _kind(sim, "injury"), "負傷が救急連鎖へ渡っていない"
    assert _kind(sim, "ems_call")


def test_traffic_standing_agents_are_not_exposed(tmp_path):
    """**横断中**(経路が残っている)個体だけが曝露に数えられる。"""
    sim = _sim(tmp_path, "ie_tr_idle", n_steps=1, **TRAFFIC_ON)
    node = _road_node(sim)
    idle = sim.agents[0]
    _exile(sim, keep=(idle,))
    _place(sim, idle, node, route=[])            # 立ち止まっている = 横断していない
    sim.traffic.enabled = True
    sim.traffic.last_n = 5
    IE.phase(sim, 0, 0)
    assert _kind(sim, "traffic_accident") == []


def test_traffic_off_road_agents_are_not_exposed(tmp_path):
    """★車道に接していない node(歩行者専用路)に居る個体は曝露に数えない。"""
    sim = _sim(tmp_path, "ie_tr_foot", n_steps=1, **TRAFFIC_ON)
    roads = IE.road_nodes(sim)
    foot = next((n for n in sorted(sim.city.graph.nodes) if n not in roads), None)
    assert foot is not None, "テストが空振り(全 node が車道に接している)"
    walker = sim.agents[0]
    _exile(sim, keep=(walker,))
    _place(sim, walker, foot, route=[foot])
    sim.traffic.enabled = True
    sim.traffic.last_n = 5
    IE.phase(sim, 0, 0)
    assert _kind(sim, "traffic_accident") == []


def test_road_nodes_come_from_the_map_drivable_table(tmp_path):
    """★「どこが車道か」の判定は world/map.DRIVABLE の**唯一の表**をそのまま読む。"""
    from society.world.map import DRIVABLE
    sim = _sim(tmp_path, "ie_roads", n_steps=1, **TRAFFIC_ON)
    want = set()
    for u, v, data in sim.city.graph.edges(data=True):
        if str(data.get("klass") or "") in DRIVABLE:
            want.add(str(u))
            want.add(str(v))
    assert IE.road_nodes(sim) == frozenset(want)


# =========================================================================== #
# (F) 群集 = 生成しない = 状態が事件(検収基準 ⑥)
# =========================================================================== #
CROWD_ON = {**ON, "incidents_env.fire.enabled": "false",
            "incidents_env.traffic.enabled": "false"}


def _phys(sim, density, occupancy=12):
    sim._phys_state = {"by_zone": {"scramble": {"density": float(density),
                                                "occupancy": int(occupancy)}}}


def test_crowd_requires_physics_zones(tmp_path):
    """★密度を測っているのは物理層だけ = 無いランでは 0 件(捏造しない)。"""
    sim = _sim(tmp_path, "ie_cr_none", n_steps=1, **CROWD_ON)
    assert getattr(sim, "_phys_state", None) is None
    IE.phase(sim, 0, 0)
    assert _kind(sim, "crowd_density_incident") == []


def test_crowd_threshold_crossing_is_the_incident(tmp_path):
    """閾値を跨ぐこと**そのもの**が事件。1 エピソード 1 件(ヒステリシス)。"""
    sim = _sim(tmp_path, "ie_cr", n_steps=1, **CROWD_ON)
    _phys(sim, 4.5)
    IE.phase(sim, 0, 0)
    got = _kind(sim, "crowd_density_incident")
    assert len(got) == 1
    assert got[0].payload["level"] == 1 and got[0].payload["near_miss"] is True
    assert got[0].payload["threshold"] == 4.0
    assert got[0].payload["density"] == 4.5      # ★前兆状態 = 密度そのもの
    assert got[0].payload["zone"] == "scramble"
    assert got[0].agent_id == -1                 # 誰かが集めたのではない
    # 同じ水準に留まるあいだは何度も出さない
    _phys(sim, 4.8)
    IE.phase(sim, 1, 10)
    assert len(_kind(sim, "crowd_density_incident")) == 1
    # 次の水準を跨げばもう 1 件(near_miss ではない)
    _phys(sim, 6.5)
    IE.phase(sim, 2, 20)
    got = _kind(sim, "crowd_density_incident")
    assert len(got) == 2 and got[1].payload["level"] == 2
    assert got[1].payload["near_miss"] is False
    # 下振れで再武装 → もう一度跨げばまた 1 件
    _phys(sim, 1.0)
    IE.phase(sim, 3, 30)
    _phys(sim, 4.5)
    IE.phase(sim, 4, 40)
    assert len(_kind(sim, "crowd_density_incident")) == 3


def test_crowd_observation_does_not_move_the_world(tmp_path):
    """★観測は状態を動かさない(物理層のゾーン統計を 1 バイトも書き換えない)。"""
    sim = _sim(tmp_path, "ie_cr_ro", n_steps=1, **CROWD_ON)
    _phys(sim, 14.0, occupancy=99)
    before = copy.deepcopy(sim._phys_state)
    IE.phase(sim, 0, 0)
    assert sim._phys_state == before
    got = _kind(sim, "crowd_density_incident")
    assert {e.payload["level"] for e in got} == {1, 2, 3}   # 3 水準すべてを跨いだ
    assert all(e.payload["guards"] == 0 for e in got)       # 雑踏警備は読むだけ


def test_crowd_levels_are_the_documented_thresholds():
    cfg = IE.build_cfg(load_config().incidents_env)
    assert cfg["crowd"]["levels"] == [4.0, 6.0, 13.0]
    assert IE.CROWD_LEVELS == (4.0, 6.0, 13.0)


# =========================================================================== #
# (G) 決定論・resume(検収基準 ⑧)
# =========================================================================== #
def test_on_is_deterministic_across_two_runs(tmp_path):
    a = _sim(tmp_path, "ie_det_a", n_steps=72, n_agents=20, **FIRE_ON)
    a.run()
    b = _sim(tmp_path, "ie_det_b", n_steps=72, n_agents=20, **FIRE_ON)
    b.run()
    assert _l1(a) == _l1(b), "ON の決定論が崩れている"
    assert _kind(a, "fire_start"), "テストが空振り(火災 0 件)"
    assert a._incenv_state["fires"] == b._incenv_state["fires"]
    assert a._incenv_state["live_fires"] == b._incenv_state["live_fires"]


def test_resume_matches_straight(tmp_path):
    """ON で resume==straight(parquet バイト一致 + 進行中の火災の一致)。"""
    ov = {**FIRE_ON, "run.start_tod": "00:00"}
    split, total = 36, 72
    straight_dir = tmp_path / "ie_straight"
    straight = Simulation(_cfg("ie_straight", total, 20, **ov), out_dir=straight_dir)
    straight.run()
    assert _kind(straight, "fire_start"), "テスト前提が崩れた(火災ゼロ)"

    d = tmp_path / "ie_resumed"
    sim1 = Simulation(_cfg("ie_resumed", split, 20, **ov,
                           **{"observer.checkpoint_every": split}), out_dir=d)
    for step in range(split):
        scheduler.run_step(sim1, step)
    checkpoint.save(sim1, split, d / "checkpoint" / f"ckpt-{split:06d}.pkl.gz")
    sim1.logger.flush_segment()
    sim2 = Simulation(_cfg("ie_resumed", total, 20, **ov,
                           **{"observer.checkpoint_every": split}), out_dir=d)
    sim2.run(resume_from=d)
    for stem in ("l1_events", "l2_metrics"):
        sa = pq.read_table(straight_dir / f"{stem}.parquet").to_pylist()
        sb = pq.read_table(d / f"{stem}.parquet").to_pylist()
        assert sa == sb, f"{stem} 不一致(incidents_env resume)"
    assert straight._incenv_state["fires"] == sim2._incenv_state["fires"]
    assert straight._incenv_state["live_fires"] == sim2._incenv_state["live_fires"]
    assert straight._incenv_state["crowd_armed"] == sim2._incenv_state["crowd_armed"]


def test_provenance_reports_measurement_against_the_anchor(tmp_path):
    """★較正が合っているかをランの成果物が自己申告する(黙って外れない)。"""
    sim = _sim(tmp_path, "ie_prov", n_steps=48, n_agents=20, **FIRE_ON)
    sim.run()
    prov = IE.provenance(sim)
    assert prov["area_share"] == 1.0
    assert prov["fire_reference_per_day"] == 144.0
    assert prov["fires"] == len(_kind(sim, "fire_start"))
    assert set(prov["fires_by_severity"]) <= set(IE.FIRE_SEVERITIES)
    assert "total" not in prov["fires_by_severity"], "全焼が出た(較正破壊)"
    assert 0.0 <= prov["fire_attributed_rate"] <= 1.0
    assert prov["ward_area_km2"] == 15.11


# =========================================================================== #
# (H) 静的検査(検収基準 ⑨)
# =========================================================================== #
def test_module_calls_no_llm_and_only_two_new_streams():
    """generate() を 1 度も呼ばず、乱数は**新 stream 2 本だけ**。"""
    text = MODULE.read_text(encoding="utf-8")
    tree = ast.parse(text)
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    idents = attrs | names
    assert "generate" not in idents, "LLM の呼び出しサイトを作っている"
    assert "llm" not in attrs
    streams = set(re.findall(r'hub\.stream\(\s*"([a-z_]+)"', text))
    assert streams == {"incident_fire", "incident_traffic"}, streams
    # 群集は乱数を 1 本も引かない(閾値跨ぎは完全決定論)
    crowd = text.split("def _crowd_tick")[1].split("\ndef ")[0]
    assert "stream" not in crowd and "rng" not in crowd


def test_module_never_writes_health_state():
    """身体の状態(sick / severity / sleeping)は 1 バイトも書かない(H1 の管轄)。"""
    text = MODULE.read_text(encoding="utf-8")
    tree = ast.parse(text)
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
    for banned in ("sick", "sick_until", "sleeping", "sleep_until", "fatigue",
                   "money", "drive", "opinion", "severity"):
        assert banned not in written, f"H5 が {banned} を書き換えている"


def test_frozen_metric_spec_files_are_untouched():
    """凍結 14 ファイルに本レーンの痕跡が 1 つも無い。"""
    from society.observer import metrics_spec as MS
    assert len(MS.SPEC_FILES) == 14
    for rel in MS.SPEC_FILES:
        text = (REPO / rel).read_text(encoding="utf-8")
        for word in ("incidents_env", "facility_devices", "fire_start",
                     "crowd_density_incident"):
            assert word not in text, f"凍結ファイル {rel} に H5 の痕跡がある"
