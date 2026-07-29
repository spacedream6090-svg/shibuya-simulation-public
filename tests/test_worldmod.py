"""環境改変条件 world.mod(A1 第67バッチ)のテスト。

方針(既存の鉄則を継承):
- OFF(既定): プロファイルを開かない・sim.worldmod is None・地図(closed/cost_scale/base_length)も
  commerce 設定も不変・summary に world_mod キーなし・新イベント 0 件・**15体144step の L1 が
  tests/data/golden_baseline_l1.json とバイト一致**・draw 数(全 stream)同一。
- 乱数ゼロ: ON でも worldmod は 1 本も draw しない(適用はワールド構築時の決定論 1 回)。
- ON スモーク: edges_closed で routing が迂回 / edge_speed_scale で走行コスト長と移動が変わる /
  open_hours(cat)が commerce の行き先フィルタに効く。
- 予約フィールド(gate_capacity・open_hours.pois)は受理・記録するが世界に効かない
  ことを summary の reserved_not_consumed で固定する(正直な記録)。
- R1: mod ON のまま compute_matched 下で k=free と k=off の LLM 呼数が一致。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from society import commerce as _commerce
from society.config import load_config
from society.engine.simulation import Simulation
from society.observer.schema import EVENT_KINDS
from society.world import worldmod as _wm
from society.world.map import CityMap

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN = REPO_ROOT / "tests" / "data" / "golden_baseline_l1.json"
DEFAULT_MAP = REPO_ROOT / "data" / "shibuya_osm.json"
EXAMPLE = REPO_ROOT / "conf" / "worldmod" / "example_counterfactual.yaml"

# test_scenario.py:45 と同じ「意図的な既定挙動追加」の中立化(ゴールデン比較用)
_GOLDEN_NEUTRAL = {"economy.wages.自営": 0, "planning.enabled": "false",
                   "transit_ride.taxi.enabled": "false",
                   "transit_ride.bus.enabled": "false", "rules.enabled": "false"}


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _sim(tmp_path, name, n=15, steps=1, **ov):
    dot = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
           "run.seed=42", "model.backend=mock", "observer.snapshot_every=144"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _profile(tmp_path, name: str, doc: dict) -> str:
    """プロファイル YAML を tmp_path に書き、dotlist に渡せる posix パスを返す。"""
    p = tmp_path / f"{name}.yaml"
    OmegaConf.save(OmegaConf.create(doc), p)
    return p.as_posix()


def _mod_on(path: str) -> dict:
    return {"world.mod.enabled": "true", "world.mod.profile": path}


class _CountingHub:
    """全 stream の draw を数えるプロキシ(OFF 同一性の「draw 数同一」検証用)。"""

    def __init__(self, inner):
        self._inner = inner
        self.per_stream: dict = {}

    def stream(self, *key):
        g = self._inner.stream(*key)
        return _CountingGen(g, self, str(key[0]) if key else "")

    def key_name(self, *key):
        return self._inner.key_name(*key)

    @property
    def master_seed(self):
        return self._inner.master_seed


class _CountingGen:
    def __init__(self, g, hub, name):
        self._g, self._hub, self._name = g, hub, name

    def __getattr__(self, attr):
        target = getattr(self._g, attr)
        if not callable(target):
            return target
        hub, name = self._hub, self._name

        def wrapped(*a, **k):
            hub.per_stream[name] = hub.per_stream.get(name, 0) + 1
            return target(*a, **k)
        return wrapped


# ============================================================ 既定 OFF の不変条件
def test_off_matches_golden(tmp_path):
    """明示 OFF(mod/heights とも)が変更前ゴールデンと一字一句一致=両 seam が完全 no-op。"""
    golden = json.load(open(GOLDEN, encoding="utf-8"))
    sim = _sim(tmp_path, "gmod", steps=144,
               **{**_GOLDEN_NEUTRAL, "world.mod.enabled": "false",
                  "world.heights.enabled": "false"})
    sim.run()
    assert _l1(sim) == golden, "world.mod / world.heights の seam が no-op でない"


def test_off_leaves_no_state(tmp_path):
    """OFF: worldmod なし・グラフに closed/cost_scale/base_length なし・commerce 既定のまま。"""
    sim = _sim(tmp_path, "moff", steps=3, **{"commerce.enabled": "true"})
    assert sim.worldmod is None
    for _u, _v, d in sim.city.graph.edges(data=True):
        assert not any(k in d for k in ("closed", "cost_scale", "base_length"))
    assert sim.commercecfg["hours"] == _commerce.build_cfg({"enabled": True})["hours"]
    sim.run()
    summary = json.loads((tmp_path / "moff" / "summary.json").read_text(encoding="utf-8"))
    assert "world_mod" not in summary


def test_off_draw_counts_identical(tmp_path):
    """OFF は純粋既定と draw 数(stream 別)が完全一致=CRN 共分散を壊さない。"""
    def _draws(name, **ov):
        sim = _sim(tmp_path, name, steps=24, **ov)
        sim.hub = _CountingHub(sim.hub)
        sim.run()
        return sim.hub.per_stream

    pure = _draws("d_pure")
    off = _draws("d_off", **{"world.mod.enabled": "false", "world.heights.enabled": "false"})
    assert pure == off and sum(pure.values()) > 0


def test_on_adds_no_event_kinds(tmp_path):
    """ON でも新しいイベント種は 1 つも増えない(worldmod は L1 に何も書かない)。"""
    path = _profile(tmp_path, "quiet", {"name": "quiet",
                                        "edge_speed_scale": [
                                            {"scale": 0.8,
                                             "bbox": [[-100.0, -100.0, 100.0, 100.0]]}]})
    sim = _sim(tmp_path, "kinds", steps=24, **_mod_on(path))
    sim.run()
    kinds = {e.kind for e in sim.logger.events}
    assert kinds <= set(EVENT_KINDS), f"未登録のイベント種: {sorted(kinds - set(EVENT_KINDS))}"
    assert not any("mod" in k or "height" in k for k in kinds)


def test_on_draws_zero_from_worldmod(tmp_path):
    """worldmod 自身は乱数を 1 本も引かない(適用は構築時の決定論・専用 stream も作らない)。"""
    path = _profile(tmp_path, "nodraw", {"name": "nodraw",
                                         "edges_closed": {"bbox": [[-30.0, -30.0, 30.0, 30.0]]}})
    cfg = load_config(["run.seed=42", "run.n_agents=4", "run.n_steps=1",
                       "run.name=nodraw", "model.backend=mock", *[f"{k}={v}" for k, v in
                                                                  _mod_on(path).items()]])
    sim = Simulation(cfg, out_dir=tmp_path / "nodraw_run")
    hub = _CountingHub(sim.hub)
    before = dict(hub.per_stream)
    mod = _wm.load(cfg, REPO_ROOT)
    mod.apply_world(sim.city)
    mod.apply_commerce(sim.commercecfg)
    assert hub.per_stream == before == {}


# ============================================================ セレクタ(純関数)
def test_select_edges_explicit_and_bbox():
    """明示リストと矩形の 2 通り+和集合。矩形は端点いずれかが内側なら選択(昇順=決定論)。"""
    city = CityMap(DEFAULT_MAP)
    u, v = next(iter(city.graph.edges()))
    explicit = _wm.select_edges(city, {"edges": [[u, v]]})
    assert explicit == [_wm.edge_key(u, v)]
    assert _wm.select_edges(city, {"edges": [{"u": v, "v": u}]}) == explicit, "無向の正準化"

    small = _wm.select_edges(city, {"bbox": [[-30.0, -30.0, 30.0, 30.0]]})
    big = _wm.select_edges(city, {"bbox": [[-120.0, -120.0, 120.0, 120.0]]})
    assert 0 < len(small) < len(big)
    assert set(small) <= set(big)
    assert big == sorted(set(big)), "決定論(昇順・重複なし)でない"
    for a, b in small:
        ax, ay = city.node_xy(a)
        bx, by = city.node_xy(b)
        assert (abs(ax) <= 30 and abs(ay) <= 30) or (abs(bx) <= 30 and abs(by) <= 30)
    union = _wm.select_edges(city, {"edges": [[u, v]], "bbox": [[-30.0, -30.0, 30.0, 30.0]]})
    assert set(union) == set(small) | set(explicit)


def test_select_edges_rejects_unknown_edge_and_key():
    city = CityMap(DEFAULT_MAP)
    with pytest.raises(ValueError, match="存在しないエッジ"):
        _wm.select_edges(city, {"edges": [["zzz", "yyy"]]})
    with pytest.raises(ValueError, match="未知のセレクタキー"):
        _wm.select_edges(city, {"radius_m": 10})


# ============================================================ ON: エッジ無効化
def test_edges_closed_reroutes(tmp_path):
    """edges_closed のエッジは新規経路に 1 本も現れず、無改変時の経路とは変わる。"""
    base = _sim(tmp_path, "ec_base", n=4, steps=1)
    city = base.city
    box = [-60.0, -60.0, 60.0, 60.0]
    target = set(_wm.select_edges(city, {"bbox": [box]}))
    gws = base.city.gateways
    src = dst = pre = None
    for i in range(len(gws)):
        for j in range(len(gws)):
            if i != j:
                p = base.router.route(gws[i], gws[j], "walk")[0]
                if len(p) >= 2 and any(_wm.edge_key(u, v) in target
                                       for u, v in zip(p, p[1:])):
                    src, dst, pre = gws[i], gws[j], p
                    break
        if src:
            break
    assert src is not None, "封鎖対象をまたぐ経路が見つからない"

    path = _profile(tmp_path, "closed", {"name": "closed",
                                         "edges_closed": {"bbox": [box]}})
    on = _sim(tmp_path, "ec_on", n=4, steps=1, **_mod_on(path))
    assert on.worldmod.applied["edges_closed"]["n_edges"] == len(target)
    post = on.router.route(src, dst, "walk")[0]
    assert not any(_wm.edge_key(u, v) in target for u, v in zip(post, post[1:])), \
        "無効化したエッジを新規経路が通っている"
    assert post != pre, "無効化で経路が変わっていない(迂回していない)"


def test_edges_closed_explicit_list(tmp_path):
    """明示リスト指定でもちょうどそのエッジだけが closed になる。"""
    city = CityMap(DEFAULT_MAP)
    pairs = [list(e) for e in sorted(city.graph.edges())[:3]]
    path = _profile(tmp_path, "expl", {"name": "expl", "edges_closed": {"edges": pairs}})
    sim = _sim(tmp_path, "ec_expl", n=4, steps=1, **_mod_on(path))
    closed = {_wm.edge_key(u, v) for u, v, d in sim.city.graph.edges(data=True)
              if d.get("closed")}
    assert closed == {_wm.edge_key(*p) for p in pairs}


# ============================================================ ON: 速度係数
def test_edge_speed_scale_changes_cost_and_position(tmp_path):
    """scale=0.5 で走行コスト長が 2 倍になり、xy_along がコスト長→幾何長を正しく戻す。"""
    plain = CityMap(DEFAULT_MAP)
    box = [-60.0, -60.0, 60.0, 60.0]
    picked = _wm.select_edges(plain, {"bbox": [box]})
    u, v = picked[0]
    base_len = plain.edge_length(u, v)
    base_mid = plain.xy_along(u, v, base_len / 2.0)
    base_end = plain.xy_along(u, v, base_len)

    path = _profile(tmp_path, "slow", {"name": "slow",
                                       "edge_speed_scale": [{"scale": 0.5, "bbox": [box]}]})
    sim = _sim(tmp_path, "sp_on", n=4, steps=1, **_mod_on(path))
    d = sim.city.graph.edges[u, v]
    assert d["base_length"] == base_len
    assert sim.city.edge_length(u, v) == pytest.approx(base_len * 2.0, abs=0.1)
    assert d["cost_scale"] == pytest.approx(2.0, abs=0.01)
    # コスト長の全長を進むと幾何の終端、半分で幾何の中点(位置は改変前と同じ場所)
    assert sim.city.xy_along(u, v, sim.city.edge_length(u, v)) == pytest.approx(base_end)
    assert sim.city.xy_along(u, v, base_len) == pytest.approx(base_mid)
    # 経路の総走行コストも増える(= routing の走行時間に効く)
    rule = sim.worldmod.applied["edge_speed_scale"][0]
    assert rule == {"scale": 0.5, "n_selected": len(picked), "n_applied": len(picked)}


def test_edge_speed_scale_changes_travel(tmp_path):
    """速度係数は移動の実挙動を変える(同一 seed で L1 が変わる)= 単なる記録ではない。"""
    box = [-900.0, -700.0, 800.0, 700.0]            # 地図全体
    a = _sim(tmp_path, "tv_base", steps=12)
    a.run()
    path = _profile(tmp_path, "slowall",
                    {"name": "slowall",
                     "edge_speed_scale": [{"scale": 0.25, "bbox": [box]}]})
    b = _sim(tmp_path, "tv_slow", steps=12, **_mod_on(path))
    b.run()
    assert _l1(a) != _l1(b), "速度係数が移動に効いていない"
    dist_a = sum(e.payload["dist_m"] for e in a.logger.events if e.kind == "move_segment")
    dist_b = sum(e.payload["dist_m"] for e in b.logger.events if e.kind == "move_segment")
    assert dist_a > 0 and dist_b > 0
    # 到着(arrive)が遅れる=同じ step 数で着ける人が減る
    arr_a = len([e for e in a.logger.events if e.kind == "arrive"])
    arr_b = len([e for e in b.logger.events if e.kind == "arrive"])
    assert arr_b < arr_a, f"減速したのに到着数が減っていない: {arr_a} -> {arr_b}"


def test_edge_speed_scale_last_rule_wins(tmp_path):
    """同一エッジが複数ルールに該当したら記述順で後勝ち(決定論)。"""
    box = [-60.0, -60.0, 60.0, 60.0]
    plain = CityMap(DEFAULT_MAP)
    u, v = _wm.select_edges(plain, {"bbox": [box]})[0]
    base_len = plain.edge_length(u, v)
    path = _profile(tmp_path, "twice",
                    {"name": "twice",
                     "edge_speed_scale": [{"scale": 0.5, "bbox": [box]},
                                          {"scale": 2.0, "bbox": [box]}]})
    sim = _sim(tmp_path, "sp_last", n=4, steps=1, **_mod_on(path))
    assert sim.city.edge_length(u, v) == pytest.approx(base_len / 2.0, abs=0.1)


def test_edge_speed_scale_rejects_nonpositive(tmp_path):
    path = _profile(tmp_path, "bad", {"name": "bad",
                                      "edge_speed_scale": [{"scale": 0.0,
                                                            "bbox": [[-10, -10, 10, 10]]}]})
    with pytest.raises(ValueError, match="scale は正の数"):
        _sim(tmp_path, "sp_bad", n=4, steps=1, **_mod_on(path))


# ============================================================ ON: 営業時間
def test_open_hours_cat_override_affects_open_pois(tmp_path):
    """open_hours.cats が commerce の行き先フィルタ(filter_open)に効く。"""
    cat = "food"
    night = 3 * 60                                     # 03:00 = 既定([11,23])では閉店
    base = _sim(tmp_path, "oh_base", n=4, steps=1, **{"commerce.enabled": "true"})
    pois = base.city.pois_by_cat(cat)
    assert pois, "検証用のカテゴリ POI が地図に無い"
    assert _commerce.filter_open(base, pois, night) == []

    path = _profile(tmp_path, "always", {"name": "always",
                                         "open_hours": {"cats": {cat: [0, 0]}}})
    on = _sim(tmp_path, "oh_on", n=4, steps=1,
              **{**_mod_on(path), "commerce.enabled": "true"})
    on_pois = on.city.pois_by_cat(cat)
    assert _commerce.filter_open(on, on_pois, night) == on_pois
    assert on.commercecfg["hours"][cat] == (0, 0)
    assert on.worldmod.applied["open_hours_cats"] == {cat: [0, 0]}
    assert on.worldmod.applied["commerce_enabled"] is True


def test_open_hours_narrowing_closes_shops(tmp_path):
    """逆向き(営業時間を狭める)も効く。既定で開いている時刻を閉店にできる。"""
    cat = "food"
    noon = 12 * 60
    path = _profile(tmp_path, "narrow", {"name": "narrow",
                                         "open_hours": {"cats": {cat: [18, 23]}}})
    sim = _sim(tmp_path, "oh_narrow", n=4, steps=1,
               **{**_mod_on(path), "commerce.enabled": "true"})
    pois = sim.city.pois_by_cat(cat)
    assert _commerce.filter_open(sim, pois, noon) == []
    assert _commerce.filter_open(sim, pois, 20 * 60) == pois


# ============================================================ 予約フィールド(未消費)
def test_reserved_fields_recorded_but_not_consumed(tmp_path):
    """gate_capacity / open_hours.pois は受理・記録されるが世界には効かない(正直な記録)。"""
    path = _profile(tmp_path, "reserved",
                    {"name": "reserved",
                     "gate_capacity": {"main": 0.5},
                     "open_hours": {"pois": {"p1": [9, 21]}}})
    sim = _sim(tmp_path, "rsv", n=4, steps=3,
               **{**_mod_on(path), "commerce.enabled": "true"})
    before = _commerce.build_cfg({"enabled": True})["hours"]
    assert sim.commercecfg["hours"] == before, "POI 単位の指定が cat 表に漏れている"
    sim.run()
    summary = json.loads((tmp_path / "rsv" / "summary.json").read_text(encoding="utf-8"))
    rsv = summary["world_mod"]["reserved_not_consumed"]
    assert rsv["gate_capacity"] == {"n_entries": 1, "consumed": False}
    assert rsv["open_hours_pois"] == {"n_entries": 1, "consumed": False}


def test_gate_capacity_rejects_nonpositive(tmp_path):
    path = _profile(tmp_path, "gbad", {"name": "gbad", "gate_capacity": {"main": 0.0}})
    with pytest.raises(ValueError, match="gate_capacity"):
        _sim(tmp_path, "g_bad", n=4, steps=1, **_mod_on(path))


# ============================================================ プロファイル解決・検証
def test_profile_resolved_by_bare_name(tmp_path):
    """裸の名前は conf/worldmod/<name>.yaml に解決される。"""
    assert _wm.resolve_profile_path("example_counterfactual", REPO_ROOT) == EXAMPLE
    assert _wm.resolve_profile_path("conf/worldmod/example_counterfactual.yaml",
                                    REPO_ROOT) == EXAMPLE


def test_example_profile_applies(tmp_path):
    """同梱の見本プロファイルが既定地図に対して適用でき、条件名が summary に残る。"""
    sim = _sim(tmp_path, "ex", n=6, steps=3,
               **{"world.mod.enabled": "true",
                  "world.mod.profile": "example_counterfactual",
                  "commerce.enabled": "true"})
    sim.run()
    summary = json.loads((tmp_path / "ex" / "summary.json").read_text(encoding="utf-8"))
    wm = summary["world_mod"]
    assert wm["profile"] == "example_counterfactual"
    assert wm["path"] == "conf/worldmod/example_counterfactual.yaml"
    assert wm["applied"]["edges_closed"]["n_edges"] > 0
    assert len(wm["applied"]["edge_speed_scale"]) == 2
    assert wm["applied"]["open_hours_cats"]


def test_enabled_without_profile_raises(tmp_path):
    with pytest.raises(ValueError, match="profile が未指定"):
        _sim(tmp_path, "noprof", n=4, steps=1, **{"world.mod.enabled": "true"})


def test_missing_profile_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _sim(tmp_path, "nofile", n=4, steps=1,
             **{"world.mod.enabled": "true", "world.mod.profile": "does_not_exist"})


def test_unknown_profile_key_raises(tmp_path):
    path = _profile(tmp_path, "weird", {"name": "weird", "sidewalk_width": 3.0})
    with pytest.raises(ValueError, match="未知のキー"):
        _sim(tmp_path, "weird_run", n=4, steps=1, **_mod_on(path))


# ============================================================ R1(呼数の k 不変)
class _FixedLLM:
    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.hits = 0

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        self.calls += 1
        return self.response, str(self.calls), False


def test_llm_call_count_k_invariant(tmp_path):
    """mod ON のまま compute_matched 下で k=free と k=off の generate 呼数が完全一致(R1)。"""
    path = _profile(tmp_path, "k_mod",
                    {"name": "k_mod",
                     "edges_closed": {"bbox": [[-40.0, -40.0, 40.0, 40.0]]},
                     "edge_speed_scale": [{"scale": 0.7,
                                           "bbox": [[-200.0, -100.0, 200.0, 100.0]]}]})

    def _run(name, writeback):
        sim = _sim(tmp_path, name, steps=100,
                   **{**_mod_on(path), "controls.mode": "compute_matched",
                      "k.writeback": writeback})
        sim.llm = _FixedLLM(json.dumps({"action": "speak", "text": "x"},
                                       ensure_ascii=False))
        sim.run()
        return sim

    free = _run("k_free", "free")
    off = _run("k_off", "off")
    assert free.llm.calls == off.llm.calls > 0, \
        f"world.mod の呼数が k に依存(R1 違反): free={free.llm.calls} off={off.llm.calls}"


def test_determinism_same_seed(tmp_path):
    """mod ON でも同一 seed の 2 ランが L1 バイト一致(適用は決定論・乱数ゼロ)。"""
    path = _profile(tmp_path, "det",
                    {"name": "det",
                     "edge_speed_scale": [{"scale": 0.6,
                                           "bbox": [[-300.0, -300.0, 300.0, 300.0]]}]})
    a = _sim(tmp_path, "det_a", steps=24, **_mod_on(path))
    a.run()
    b = _sim(tmp_path, "det_b", steps=24, **_mod_on(path))
    b.run()
    assert _l1(a) == _l1(b)
