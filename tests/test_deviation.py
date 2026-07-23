"""第55バッチ タスクA: ペルソナ逸脱率レンズのテスト。

検証項目(タスク検収 2026-07-24):
  (R1)   deviation OFF は ON と L1 が完全一致(レンズは読むだけ=乱数/LLM 呼を増やさない)。
         OFF は L2 に deviation_* 列を一切足さない(L2 不変)/ サイドカーも書かない。
  (det)  deviation ON は決定論(同一入力で 2 回 → L2 の逸脱列がバイト一致)。
  (map)  behavior_map の全トップレベルキーが schema 登録済み kind(タイポ検出)。
  (pure) classify/measurable の純関数(期待一致=逸脱0 / 明確な不一致=逸脱>0 / 未対応 kind=None)。
  (母数) 裁量限定の母数(義務除外)が全時間より小さい(義務ルーチンの水増しを外している)。
  (agg)  observe の当日タリー整合 + 暦日境界リセット + scalars の全体スカラー。
  (k)    compute_matched 下で k=free と k=off の generate 呼数が完全一致(観測のみ=自明だが1本)。
  (viz)  mock 合成ランで build_data が逸脱を事後計算し、dashboard HTML 生成が通る。
         deviation_map.json が無ければ従来出力とバイト同一(後方互換)。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "viz"))

from society.config import load_config                        # noqa: E402
from society.engine.simulation import Simulation              # noqa: E402
from society.observer import deviation as dev                 # noqa: E402
from society.observer.schema import EVENT_KINDS, Event        # noqa: E402

import make_viewer as mv                                      # noqa: E402

VIEWER = REPO_ROOT / "viz" / "make_viewer.py"

DEV_COLS = ("deviation_mean", "deviation_var", "deviation_top_share",
            "deviation_fulltime_mean")


# --------------------------------------------------------------------------- #
# sim ヘルパ
# --------------------------------------------------------------------------- #
def _run(tmp_path, name, **ov):
    dot = ["run.seed=42", "run.n_agents=14", "run.n_steps=22", f"run.name={name}"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    cfg = load_config(dot)
    sim = Simulation(cfg, out_dir=tmp_path / name)
    sim.run()
    return sim


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


# --------------------------------------------------------------------------- #
# (R1) deviation OFF == ON の L1・OFF は L2 に列なし・サイドカーなし
# --------------------------------------------------------------------------- #
def test_deviation_is_read_only_l1_invariant(tmp_path):
    off = _run(tmp_path, "doff", **{"lens.deviation.enabled": "false"})
    on = _run(tmp_path, "don", **{"lens.deviation.enabled": "true"})
    assert _l1(off) == _l1(on), "deviation ON が L1 を変えた(読むだけでない=R1 違反)"


def test_deviation_off_no_l2_columns(tmp_path):
    off = _run(tmp_path, "d2off", **{"lens.deviation.enabled": "false"})
    for row in off.logger.metrics:
        for col in DEV_COLS:
            assert col not in row, f"deviation OFF なのに L2 に {col} 列がある(L2 不変違反)"


def test_deviation_on_emits_l2_columns(tmp_path):
    on = _run(tmp_path, "d2on", **{"lens.deviation.enabled": "true"})
    assert on.logger.metrics, "L2 が空"
    for row in on.logger.metrics:
        for col in DEV_COLS:
            assert col in row, f"deviation ON なのに L2 に {col} 列が無い"
            assert isinstance(row[col], float)


def test_deviation_on_deterministic_l2(tmp_path):
    a = _run(tmp_path, "dda", **{"lens.deviation.enabled": "true"})
    b = _run(tmp_path, "ddb", **{"lens.deviation.enabled": "true"})
    va = [[r["step"], *[r[c] for c in DEV_COLS]] for r in a.logger.metrics]
    vb = [[r["step"], *[r[c] for c in DEV_COLS]] for r in b.logger.metrics]
    assert va == vb, "deviation ON の L2 逸脱列が決定論でない(2 回で不一致)"


def test_deviation_off_sidecar_absent(tmp_path):
    off = _run(tmp_path, "dsc_off", **{"lens.deviation.enabled": "false"})
    assert not (off.out_dir / "deviation_map.json").exists(), \
        "deviation OFF でサイドカーが書かれた"


def test_deviation_on_sidecar_written(tmp_path):
    on = _run(tmp_path, "dsc_on", **{"lens.deviation.enabled": "true"})
    p = on.out_dir / "deviation_map.json"
    assert p.exists(), "deviation ON でサイドカーが書かれていない"
    m = json.loads(p.read_text(encoding="utf-8"))
    assert m["work_category"] == dev.WORK_CATEGORY
    assert "会社員" in m["hobby_map"] and "会社員" in m["occupation_map"]
    assert "spend" in m["behavior_map"]


# --------------------------------------------------------------------------- #
# (map) behavior_map の全トップレベルキーが schema 登録済み kind(タイポ検出)
# --------------------------------------------------------------------------- #
def test_behavior_map_keys_are_registered():
    for k in dev.DEFAULT_BEHAVIOR_MAP:
        assert k in EVENT_KINDS, f"behavior_map の未登録 kind '{k}'(タイポ?)"


def test_maps_are_canonical():
    """hobby_map の値(期待カテゴリ)が behavior_map の値域に閉じる(勤務カテゴリ以外)+全職業網羅。"""
    beh_cats = set()

    def _leaf(spec):
        if isinstance(spec, str):
            beh_cats.add(spec)
        elif isinstance(spec, dict):
            for kk, vv in spec.items():
                if kk not in ("__payload__", "__payload_argmax__") and isinstance(vv, str):
                    beh_cats.add(vv)
    for spec in dev.DEFAULT_BEHAVIOR_MAP.values():
        _leaf(spec)
    beh_cats.discard(dev.WORK_CATEGORY)
    for occ, cats in dev.DEFAULT_HOBBY_MAP.items():
        for c in cats:
            assert c in beh_cats, f"hobby_map[{occ}] の {c} が behavior_map の値域に無い"
    # occupation_map の勤務カテゴリも behavior_map の値域(education 等)に載る or POI カテゴリ
    for occ, c in dev.DEFAULT_OCCUPATION_MAP.items():
        assert isinstance(c, str) and c


# --------------------------------------------------------------------------- #
# (pure) classify / measurable の純関数
# --------------------------------------------------------------------------- #
def test_classify_pure():
    cfg = dev.build_cfg({"enabled": True})
    # 会社員: 勤務=office / 趣味=food
    assert dev.classify("spend", {"cat": "food"}, "会社員", cfg) == ("food", False, True)
    assert dev.classify("spend", {"cat": "nightlife"}, "会社員", cfg) == ("nightlife", False, False)
    assert dev.classify("spend", {"cat": "cafe"}, "会社員", cfg) == ("food", False, True)   # cafe→food
    assert dev.classify("wage", {}, "会社員", cfg) == ("@work", True, True)                # 義務=常に従順
    assert dev.classify("service_use", {"service": "gym"}, "会社員", cfg) == ("leisure", False, False)
    # 交通/未対応 kind は None(カテゴリ非対応=数えない)
    assert dev.classify("spend", {"cat": "taxi"}, "会社員", cfg) is None
    assert dev.classify("move_segment", {}, "会社員", cfg) is None
    # バンドマン: 裁量趣味=nightlife。勤務(義務)行動は職業によらず常に従順(条件1=機械導出)。
    assert dev.classify("spend", {"cat": "nightlife"}, "バンドマン", cfg) == ("nightlife", False, True)
    assert dev.classify("wage", {}, "バンドマン", cfg) == ("@work", True, True)


def test_measurable_gate():
    hmap = dev.DEFAULT_HOBBY_MAP
    assert dev.measurable("会社員", hmap) and dev.measurable("バンドマン", hmap)
    assert not dev.measurable("宇宙飛行士", hmap)          # 未収載職業=測定対象外
    assert not dev.measurable("", hmap)


# --------------------------------------------------------------------------- #
# observe / scalars(合成 FakeSim)
# --------------------------------------------------------------------------- #
class _FakeLogger:
    def __init__(self, events):
        self.events = events
        self._n_flushed = 0


class _FakeAgent:
    def __init__(self, aid, occ):
        self.id = aid
        self.occupation = occ


class _FakeSim:
    def __init__(self, events, agents, enabled=True):
        self.logger = _FakeLogger(events)
        self.agents = agents
        self.cfg = {"lens": {"deviation": {"enabled": enabled}}}


def _ev(step, sim_min, kind, agent=1, payload=None):
    return Event(step=step, sim_min=sim_min, agent_id=agent, kind=kind,
                 x=0.0, y=0.0, payload=payload or {})


def test_observe_discretionary_subset_of_fulltime():
    """裁量母数(義務除外)< 全時間母数(義務込み)= 勤務行動を裁量から外している。"""
    ev = [
        _ev(0, 100, "wage"),                         # 義務(勤務)= 全時間のみ
        _ev(0, 100, "spend", payload={"cat": "food"}),       # 裁量・一致
        _ev(0, 100, "spend", payload={"cat": "nightlife"}),  # 裁量・逸脱
    ]
    sim = _FakeSim(ev, [_FakeAgent(1, "会社員")])
    st = dev.observe(sim)
    rec = st["per"][1]
    assert rec["disc_total"] == 2 and rec["disc_dev"] == 1      # 裁量: 2件中1逸脱
    assert rec["full_total"] == 3 and rec["full_dev"] == 1      # 全時間: 3件中1逸脱(wage=従順)
    assert rec["disc_total"] < rec["full_total"], "裁量母数が全時間より小さくない"


def test_observe_conform_zero_deviate_positive():
    """期待一致行動のみ=逸脱0 / 明確な不一致のみ=逸脱率1.0 → 全体平均が中間。"""
    ev = [
        _ev(0, 100, "spend", agent=1, payload={"cat": "food"}),       # 会社員・一致
        _ev(0, 100, "spend", agent=1, payload={"cat": "food"}),       # 会社員・一致
        _ev(0, 100, "spend", agent=2, payload={"cat": "nightlife"}),  # 会社員・逸脱
        _ev(0, 100, "spend", agent=2, payload={"cat": "shop"}),       # 会社員・逸脱
    ]
    sim = _FakeSim(ev, [_FakeAgent(1, "会社員"), _FakeAgent(2, "会社員")])
    s = dev.scalars(sim)
    assert s["deviation_mean"] == 0.5                # (0 + 1)/2
    assert s["deviation_top_share"] == 0.5           # agent2 のみ >= 0.5
    assert s["deviation_var"] == 0.25                # ((0-.5)^2+(1-.5)^2)/2
    # per-agent 確認
    st = sim._dev_state
    assert st["per"][1]["disc_dev"] == 0             # 一致のみ=逸脱0
    assert st["per"][2]["disc_dev"] == 2             # 不一致のみ


def test_observe_daily_reset():
    ev = [_ev(0, 100, "spend", payload={"cat": "nightlife"})]     # day0・逸脱
    sim = _FakeSim(ev, [_FakeAgent(1, "会社員")])
    st = dev.observe(sim)
    assert st["per"][1]["disc_dev"] == 1
    sim.logger.events.append(_ev(144, 1500, "spend", payload={"cat": "food"}))  # 翌日・一致
    st2 = dev.observe(sim)
    assert st2["day"] == 1 and st2["per"][1]["disc_dev"] == 0     # 当日タリーがリセット
    assert st2["per"][1]["disc_total"] == 1


def test_observe_off_returns_none():
    sim = _FakeSim([_ev(0, 100, "spend", payload={"cat": "food"})],
                   [_FakeAgent(1, "会社員")], enabled=False)
    assert dev.observe(sim) is None
    assert dev.scalars(sim) is None


def test_observe_unmeasurable_and_world_event_skipped():
    ev = [
        _ev(0, 100, "spend", agent=1, payload={"cat": "food"}),      # 未収載職業=数えない
        _ev(0, 100, "spend", agent=-1, payload={"cat": "food"}),     # 世界イベント=数えない
    ]
    sim = _FakeSim(ev, [_FakeAgent(1, "宇宙飛行士")])
    st = dev.observe(sim)
    assert st["per"] == {}, "測定対象外/世界イベントが母数に混入した"


def test_observe_idempotent_within_step():
    sim = _FakeSim([_ev(0, 100, "spend", payload={"cat": "nightlife"})],
                   [_FakeAgent(1, "会社員")])
    a = dev.observe(sim)
    b = dev.observe(sim)          # 同 step 内の再呼出は二度数えない
    assert a["per"][1]["disc_total"] == 1 and b["per"][1]["disc_total"] == 1


# --------------------------------------------------------------------------- #
# (k) compute_matched 下で k=free==k=off の generate 呼数一致(観測のみ=自明だが1本)
# --------------------------------------------------------------------------- #
class _FixedLLM:
    """挙動を固定する backend(応答をプロンプト非依存)。呼数だけ数える(test_chance と同型)。"""

    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.hits = 0

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        self.calls += 1
        return self.response, str(self.calls), False


def _run_k(tmp_path, name, *, writeback):
    dot = ["run.seed=7", "run.n_agents=20", "run.n_steps=144", f"run.name={name}",
           "lens.deviation.enabled=true", "controls.mode=compute_matched",
           f"k.writeback={writeback}"]
    sim = Simulation(load_config(dot), out_dir=tmp_path / name)
    sim.llm = _FixedLLM(json.dumps({"action": "speak", "text": "x"}, ensure_ascii=False))
    sim.run()
    return sim


def test_deviation_call_count_k_invariant(tmp_path):
    """逸脱レンズは L1 を読むだけ=発火判断に一切食わせない=compute_matched 下で k 呼数不変。"""
    free = _run_k(tmp_path, "dk_free", writeback="free")
    off = _run_k(tmp_path, "dk_off", writeback="off")
    assert free.llm.calls == off.llm.calls and free.llm.calls > 0, \
        f"deviation の呼数が k に依存(R1 違反): free={free.llm.calls} off={off.llm.calls}"


# --------------------------------------------------------------------------- #
# (viz) 合成ランで build_data が逸脱を事後計算 + HTML 生成
# --------------------------------------------------------------------------- #
def _minimal_map(path: Path) -> None:
    city = {
        "buildings": [{"id": "b1", "name": "テストビル", "kind": "generic",
                       "levels": 3, "footprint": [[0, 0], [10, 0], [10, 10], [0, 10]]}],
        "nodes": [{"id": "n1", "x": 0, "y": 0, "name": "ノードA"},
                  {"id": "n2", "x": 10, "y": 10}],
        "edges": [{"klass": "footway", "layer": 0, "geometry": [[0, 0], [10, 10]]}],
        "pois": [], "railways": [], "meta": {"origin_latlon": [35.66, 139.70]},
    }
    path.write_text(json.dumps(city, ensure_ascii=False), encoding="utf-8")


def _write_run(tmp_path: Path, name: str, events: list[dict], *,
               dev_map: dict | None = None, agents: list[dict] | None = None) -> Path:
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True, exist_ok=True)
    map_path = tmp_path / f"{name}_map.json"
    _minimal_map(map_path)
    (run_dir / "config.yaml").write_text(
        "world:\n"
        f"  map: {map_path.as_posix()}\n"
        "transit:\n"
        "  file: data/__no_transit_for_test__.json\n",
        encoding="utf-8")
    if agents is None:
        agents = [{"id": i, "name": f"住民{i}", "age": 20 + i, "gender": "男",
                   "occupation": "会社員", "has_bicycle": False, "has_car": False}
                  for i in range(4)]
    (run_dir / "agents.json").write_text(json.dumps(agents, ensure_ascii=False),
                                         encoding="utf-8")
    cols = {
        "step": [int(e["step"]) for e in events],
        "sim_min": [int(e.get("sim_min", 420 + int(e["step"]) * 10)) for e in events],
        "agent_id": [int(e["agent_id"]) for e in events],
        "kind": [str(e["kind"]) for e in events],
        "x": [float(e.get("x", 0.0)) for e in events],
        "y": [float(e.get("y", 0.0)) for e in events],
        "payload": [json.dumps(e.get("payload", {}), ensure_ascii=False) for e in events],
    }
    pq.write_table(pa.table(cols), run_dir / "l1_events.parquet")
    if dev_map is not None:
        (run_dir / "deviation_map.json").write_text(json.dumps(dev_map, ensure_ascii=False),
                                                    encoding="utf-8")
    return run_dir


def _dev_map():
    return dev.resolved_maps(dev.build_cfg({"enabled": True}))


def _evd(step, agent, kind, sim_min=None, **payload):
    d = {"step": step, "agent_id": agent, "kind": kind, "payload": payload}
    if sim_min is not None:
        d["sim_min"] = sim_min
    return d


def test_build_data_deviation_present(tmp_path):
    ev = [
        _evd(0, 0, "wage", sim_min=100),                          # 義務(全時間のみ)
        _evd(0, 0, "spend", sim_min=100, cat="nightlife"),        # agent0 裁量・逸脱
        _evd(1, 0, "spend", sim_min=110, cat="food"),            # agent0 裁量・一致
        _evd(1, 1, "spend", sim_min=110, cat="food"),            # agent1 裁量・一致
    ]
    rd = _write_run(tmp_path, "dz", ev, dev_map=_dev_map())
    data = mv.build_data(rd, include_traffic=False)
    assert "deviation" in data
    V = data["deviation"]
    assert V["n_measured"] == 2
    # 最逸脱者 = agent0(裁量逸脱率 0.5)
    assert V["top"][0]["id"] == 0
    assert abs(V["top"][0]["disc_ratio"] - 0.5) < 1e-9
    assert V["top"][0]["occupation"] == "会社員"
    assert V["top"][0]["expect_hobby"] == ["food"]
    assert V["top"][0]["actual"].get("nightlife") == 1
    # 分布ヒスト: 2 人(0.0 のビン0 と 0.5 のビン5)
    assert sum(V["hist"]) == 2 and V["hist"][0] == 1 and V["hist"][5] == 1


def test_build_data_no_deviation_map_backward_compat(tmp_path):
    ev = [_evd(0, 0, "spend", sim_min=100, cat="food")]
    rd = _write_run(tmp_path, "nodz", ev)          # deviation_map.json 無し
    data = mv.build_data(rd, include_traffic=False)
    assert "deviation" not in data


def test_dashboard_html_deviation_smoke(tmp_path):
    ev = [
        _evd(0, 0, "wage", sim_min=100),
        _evd(0, 0, "spend", sim_min=100, cat="nightlife"),
        _evd(1, 1, "spend", sim_min=110, cat="food"),
    ]
    rd = _write_run(tmp_path, "dsmoke", ev, dev_map=_dev_map())
    r = subprocess.run([sys.executable, str(VIEWER), str(rd)], capture_output=True)
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    dash = (rd / "dashboard.html").read_text(encoding="utf-8")
    viewer = (rd / "viewer.html").read_text(encoding="utf-8")
    assert "__LENS_JS__" not in dash and "__LENS_TABS__" not in dash
    assert 'data-tab="deviation"' in dash
    for sym in ("renderDeviation", "devRender"):
        assert sym in dash, f"{sym} が dashboard に無い"
    assert '"deviation":' in dash
    # viewer(地図)側には逸脱タブは無い
    assert 'data-tab="deviation"' not in viewer


def test_dashboard_html_no_deviation_smoke(tmp_path):
    """deviation_map 無しのランでも HTML 生成が通り、逸脱タブは非表示(後方互換)。"""
    ev = [{"step": s, "agent_id": 0, "kind": "arrive", "x": 1.0, "y": 2.0,
           "payload": {"name": "路上"}} for s in range(3)]
    rd = _write_run(tmp_path, "dsmoke_none", ev)
    r = subprocess.run([sys.executable, str(VIEWER), str(rd)], capture_output=True)
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    dash = (rd / "dashboard.html").read_text(encoding="utf-8")
    assert 'data-tab="deviation"' not in dash and "function devRender" not in dash
