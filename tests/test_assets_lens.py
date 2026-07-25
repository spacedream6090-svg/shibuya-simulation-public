"""第59バッチ スライス(a): 資産分布 observable のテスト。

検証項目(タスク検収 2026-07-25):
  (R1)   assets OFF は ON と L1 が完全一致(レンズは読むだけ=乱数/LLM 呼を増やさない)。
         OFF は L2 に asset_* 列を一切足さない(L2 不変)/ サイドカーも書かない。
  (det)  assets ON は決定論(同一入力で 2 回 → L2 の資産列がバイト一致)。
  (pure) _gini/_median/_top_share/_kendall_tau の純関数(既知ケース=手計算一致)。
  (agg)  scalars の全体スカラー = 合成 wealth の手計算 Gini/中央値/平均/上位集中と一致。
         asset_rank_tau=前日比 τ(初日=初期値 1.0 / 翌日=前日 wealth との τ)。暦日境界リセット + idempotent。
  (k)    compute_matched 下で k=free と k=off の generate 呼数が完全一致(観測のみ=自明だが1本)。
  (viz)  合成 L3 スナップ(money+account)から build_data が資産分布を事後計算し、dashboard HTML 生成が通る。
         assets_map.json が無ければ従来出力とバイト同一(後方互換=資産タブ非表示)。
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
from society.observer import assets as A                      # noqa: E402
from society.observer.schema import Event                     # noqa: E402

import make_viewer as mv                                      # noqa: E402

VIEWER = REPO_ROOT / "viz" / "make_viewer.py"

AS_COLS = ("asset_gini", "asset_top10_share", "asset_median",
           "asset_mean", "asset_rank_tau")


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
# (R1) assets OFF == ON の L1・OFF は L2 に列なし・サイドカーなし
# --------------------------------------------------------------------------- #
def test_assets_is_read_only_l1_invariant(tmp_path):
    off = _run(tmp_path, "aoff", **{"lens.assets.enabled": "false"})
    on = _run(tmp_path, "aon", **{"lens.assets.enabled": "true"})
    assert _l1(off) == _l1(on), "assets ON が L1 を変えた(読むだけでない=R1 違反)"


def test_assets_off_no_l2_columns(tmp_path):
    off = _run(tmp_path, "a2off", **{"lens.assets.enabled": "false"})
    for row in off.logger.metrics:
        for col in AS_COLS:
            assert col not in row, f"assets OFF なのに L2 に {col} 列がある(L2 不変違反)"


def test_assets_on_emits_l2_columns(tmp_path):
    on = _run(tmp_path, "a2on", **{"lens.assets.enabled": "true"})
    assert on.logger.metrics, "L2 が空"
    for row in on.logger.metrics:
        for col in AS_COLS:
            assert col in row, f"assets ON なのに L2 に {col} 列が無い"
            assert isinstance(row[col], float)
    # 初日の rows は前日比なし=初期値 1.0(型安定=part concat 安全のための規約)
    assert on.logger.metrics[0]["asset_rank_tau"] == 1.0


def test_assets_on_deterministic_l2(tmp_path):
    a = _run(tmp_path, "ada", **{"lens.assets.enabled": "true"})
    b = _run(tmp_path, "adb", **{"lens.assets.enabled": "true"})
    va = [[r["step"], *[r[c] for c in AS_COLS]] for r in a.logger.metrics]
    vb = [[r["step"], *[r[c] for c in AS_COLS]] for r in b.logger.metrics]
    assert va == vb, "assets ON の L2 資産列が決定論でない(2 回で不一致)"


def test_assets_off_sidecar_absent(tmp_path):
    off = _run(tmp_path, "asc_off", **{"lens.assets.enabled": "false"})
    assert not (off.out_dir / "assets_map.json").exists(), \
        "assets OFF でサイドカーが書かれた"


def test_assets_on_sidecar_written(tmp_path):
    on = _run(tmp_path, "asc_on", **{"lens.assets.enabled": "true", "lens.assets.top_pct": "0.2"})
    p = on.out_dir / "assets_map.json"
    assert p.exists(), "assets ON でサイドカーが書かれていない"
    m = json.loads(p.read_text(encoding="utf-8"))
    assert abs(m["top_pct"] - 0.2) < 1e-9 and m["tau_init"] == 1.0


# --------------------------------------------------------------------------- #
# (pure) 純関数の既知ケース
# --------------------------------------------------------------------------- #
def test_gini_median_topshare_pure():
    assert A._gini([10, 10, 10]) == 0.0                    # 完全平等
    assert abs(A._gini([0, 0, 0, 100]) - 0.75) < 1e-9      # 1人集中
    assert A._median([1, 2, 3, 4]) == 2.5                  # 偶数=中央2値平均
    assert A._median([1, 2, 3]) == 2                       # 奇数=中央
    # 上位10%: n=10, k=max(1,int(1.0))=1 → 100/(100+9)=0.9174...
    assert abs(A._top_share([100] + [1] * 9, 0.1) - 100 / 109) < 1e-9
    assert A._gini([1, 2]) >= 0.0 and A._top_share([], 0.1) == 0.0


def test_kendall_tau_known_cases():
    assert A._kendall_tau([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0       # 完全一致
    assert A._kendall_tau([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0      # 完全逆順
    assert A._kendall_tau([1], [1]) is None                        # n<2
    assert A._kendall_tau([5, 5, 5], [1, 2, 3]) == 0.0             # 片側定数=分母0→0.0(measure 仕様)


# --------------------------------------------------------------------------- #
# scalars / observe(合成 FakeSim)
# --------------------------------------------------------------------------- #
class _FakeLogger:
    def __init__(self, events):
        self.events = events
        self._n_flushed = 0


class _FakeAgent:
    def __init__(self, aid, money, account=0.0):
        self.id = aid
        self.money = money
        self.account = account


class _FakeSim:
    def __init__(self, events, agents, enabled=True, top_pct=0.1):
        self.logger = _FakeLogger(events)
        self.agents = agents
        self.cfg = {"lens": {"assets": {"enabled": enabled, "top_pct": top_pct}}}


def _ev(step, sim_min, kind="spend", agent=1, payload=None):
    return Event(step=step, sim_min=sim_min, agent_id=agent, kind=kind,
                 x=0.0, y=0.0, payload=payload or {})


def test_scalars_match_hand_computation():
    """wealth=[0,0,0,100] → Gini 0.75 / 中央値 0 / 平均 25 / 上位10%集中=1.0(k=1)。初日 τ=1.0。"""
    agents = [_FakeAgent(1, 0), _FakeAgent(2, 0), _FakeAgent(3, 0), _FakeAgent(4, 100)]
    sim = _FakeSim([_ev(0, 100)], agents)
    s = A.scalars(sim)
    assert abs(s["asset_gini"] - 0.75) < 1e-6
    assert s["asset_median"] == 0.0
    assert s["asset_mean"] == 25.0
    assert s["asset_top10_share"] == 1.0                  # n=4,k=1 → 100/100
    assert s["asset_rank_tau"] == 1.0                     # 初日=初期値


def test_wealth_uses_money_plus_account():
    a = _FakeAgent(1, 30.0, account=70.0)
    assert A.wealth(a) == 100.0


def test_rank_tau_previous_day():
    """翌日=前日 wealth ベクトルとの前日比 τ。順位が完全反転 → τ=-1。"""
    agents = [_FakeAgent(1, 10), _FakeAgent(2, 20), _FakeAgent(3, 30)]
    sim = _FakeSim([_ev(0, 100)], agents)
    s0 = A.scalars(sim)
    assert s0["asset_rank_tau"] == 1.0                    # 初日=初期値
    # 翌日: 残高を反転({1:30,2:20,3:10})し day1 のイベントを追加
    agents[0].money, agents[2].money = 30, 10
    sim.logger.events.append(_ev(144, 1500))
    s1 = A.scalars(sim)
    assert s1["asset_rank_tau"] == -1.0                   # 順位完全反転


def test_observe_daily_reset_and_idempotent():
    agents = [_FakeAgent(1, 10), _FakeAgent(2, 20), _FakeAgent(3, 30)]
    sim = _FakeSim([_ev(0, 100)], agents)
    a = A.observe(sim)
    b = A.observe(sim)                                    # 同 step 内の再呼出は二度処理しない
    assert a["day"] == 0 and b["day"] == 0 and a["prev"] == b["prev"]
    sim.logger.events.append(_ev(144, 1500))
    c = A.observe(sim)
    assert c["day"] == 1                                  # 暦日が進んだ


def test_observe_off_returns_none():
    sim = _FakeSim([_ev(0, 100)], [_FakeAgent(1, 10)], enabled=False)
    assert A.observe(sim) is None
    assert A.scalars(sim) is None


# --------------------------------------------------------------------------- #
# (k) compute_matched 下で k=free==k=off の generate 呼数一致(観測のみ=自明だが1本)
# --------------------------------------------------------------------------- #
class _FixedLLM:
    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.hits = 0

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        self.calls += 1
        return self.response, str(self.calls), False


def _run_k(tmp_path, name, *, writeback):
    dot = ["run.seed=7", "run.n_agents=20", "run.n_steps=144", f"run.name={name}",
           "lens.assets.enabled=true", "controls.mode=compute_matched",
           f"k.writeback={writeback}"]
    sim = Simulation(load_config(dot), out_dir=tmp_path / name)
    sim.llm = _FixedLLM(json.dumps({"action": "speak", "text": "x"}, ensure_ascii=False))
    sim.run()
    return sim


def test_assets_call_count_k_invariant(tmp_path):
    free = _run_k(tmp_path, "ak_free", writeback="free")
    off = _run_k(tmp_path, "ak_off", writeback="off")
    assert free.llm.calls == off.llm.calls and free.llm.calls > 0, \
        f"assets の呼数が k に依存(R1 違反): free={free.llm.calls} off={off.llm.calls}"


# --------------------------------------------------------------------------- #
# (viz) 合成 L3 スナップ → build_data が資産分布を事後計算 + HTML 生成
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


def _write_run(tmp_path: Path, name: str, events: list, *,
               l3_days: dict | None = None, assets_map: dict | None = None,
               n_agents: int = 6) -> Path:
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
    agents = [{"id": i, "name": f"住民{i}", "age": 20 + i, "gender": "男",
               "occupation": "会社員", "has_bicycle": False, "has_car": False}
              for i in range(n_agents)]
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
    # 合成 L3 スナップ(step -> 各 agent の money/account)。l3_days={step: {aid: (money, account)}}
    if l3_days is not None:
        steps, states = [], []
        for step in sorted(l3_days):
            steps.append(int(step))
            snap = [{"id": aid, "money": m, "account": ac}
                    for aid, (m, ac) in sorted(l3_days[step].items())]
            states.append(json.dumps({"agents": snap}, ensure_ascii=False))
        pq.write_table(pa.table({"step": steps, "state": states}),
                       run_dir / "l3_snapshots.parquet")
    if assets_map is not None:
        (run_dir / "assets_map.json").write_text(json.dumps(assets_map, ensure_ascii=False),
                                                 encoding="utf-8")
    return run_dir


def _evd(step, agent, kind="spend", sim_min=None, **payload):
    d = {"step": step, "agent_id": agent, "kind": kind, "payload": payload}
    if sim_min is not None:
        d["sim_min"] = sim_min
    return d


def test_build_data_assets_present(tmp_path):
    # 2 日分の L3。start_min=420(step0 の sim_min=420)。day(step)= (420+step*10)//1440。
    #   step0 -> day0, step144 -> day1。
    ev = [_evd(0, 0, sim_min=420, cat="food"), _evd(144, 1, sim_min=1860, cat="food")]
    l3 = {
        0:   {i: (float(i * 10), 0.0) for i in range(6)},        # day0: 0,10,20,30,40,50
        144: {i: (float((5 - i) * 10), 0.0) for i in range(6)},  # day1: 完全反転
    }
    amap = A.resolved_maps(A.build_cfg({"enabled": True}))
    rd = _write_run(tmp_path, "az", ev, l3_days=l3, assets_map=amap)
    data = mv.build_data(rd, include_traffic=False)
    assert "assets" in data
    V = data["assets"]
    assert V["n_agents"] == 6 and V["last_day"] == 1
    # day0 は全員異なる残高 → Gini>0、τ 初日 None
    assert V["gini"][0] > 0 and V["tau"][0] is None
    # day1 は完全反転 → 前日比 τ = -1
    assert abs(V["tau"][1] - (-1.0)) < 1e-9
    # 最終日ドリルダウン: 上位1位=agent0(残高50)・下位=agent5(残高0)
    assert V["top"][0]["id"] == 0 and V["top"][0]["rank"] == 1
    assert V["bottom"][0]["id"] == 5
    assert sum(V["hist"]) == 6


def test_build_data_no_assets_map_backward_compat(tmp_path):
    ev = [_evd(0, 0, sim_min=420, cat="food")]
    l3 = {0: {i: (float(i), 0.0) for i in range(6)}}
    rd = _write_run(tmp_path, "noaz", ev, l3_days=l3)     # assets_map.json 無し
    data = mv.build_data(rd, include_traffic=False)
    assert "assets" not in data


def test_build_data_assets_map_but_no_l3(tmp_path):
    """assets_map は有るが L3 スナップが無い → 資産タブは None(全体スカラーは分析タブに残る)。"""
    ev = [_evd(0, 0, sim_min=420, cat="food")]
    amap = A.resolved_maps(A.build_cfg({"enabled": True}))
    rd = _write_run(tmp_path, "azno_l3", ev, assets_map=amap)   # l3_days 無し
    data = mv.build_data(rd, include_traffic=False)
    assert "assets" not in data


def test_dashboard_html_assets_smoke(tmp_path):
    ev = [_evd(0, 0, sim_min=420, cat="food"), _evd(144, 1, sim_min=1860, cat="food")]
    l3 = {0: {i: (float(i * 10), 0.0) for i in range(6)},
          144: {i: (float((5 - i) * 10), 5.0) for i in range(6)}}
    amap = A.resolved_maps(A.build_cfg({"enabled": True}))
    rd = _write_run(tmp_path, "asmoke", ev, l3_days=l3, assets_map=amap)
    r = subprocess.run([sys.executable, str(VIEWER), str(rd)], capture_output=True)
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    dash = (rd / "dashboard.html").read_text(encoding="utf-8")
    viewer = (rd / "viewer.html").read_text(encoding="utf-8")
    assert "__LENS_JS__" not in dash and "__LENS_TABS__" not in dash
    assert 'data-tab="assets"' in dash
    for sym in ("renderAssets", "assetRender"):
        assert sym in dash, f"{sym} が dashboard に無い"
    assert '"assets":' in dash
    assert 'data-tab="assets"' not in viewer               # 地図側には資産タブは無い


def test_dashboard_html_no_assets_smoke(tmp_path):
    """assets_map 無しのランでも HTML 生成が通り、資産タブは非表示(後方互換)。"""
    ev = [{"step": s, "agent_id": 0, "kind": "arrive", "sim_min": 420 + s * 10,
           "payload": {"name": "路上"}} for s in range(3)]
    rd = _write_run(tmp_path, "asmoke_none", ev)
    r = subprocess.run([sys.executable, str(VIEWER), str(rd)], capture_output=True)
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    dash = (rd / "dashboard.html").read_text(encoding="utf-8")
    assert 'data-tab="assets"' not in dash and "function assetRender" not in dash


# --------------------------------------------------------------------------- #
# (resume) assets ON の resume==straight(τ の前日状態が checkpoint 経由で継続する・検収補修の固定)
# --------------------------------------------------------------------------- #
def test_assets_resume_matches_straight(tmp_path):
    """一気 260step と 150+resume で L2 の資産列が全行一致。

    start_tod 07:00 → 暦日境界は step102(前日 wealth スナップ=分割前)と step246(τ 実計算=resume 後)。
    前日状態が checkpoint を跨いで届かないと、resume 側は再開時点の wealth を「前日」と誤スナップし
    τ が食い違う配置。straight 側で τ 実計算が起きたこと(state.tau is not None)も固定し、
    検証が空回りしていないことを保証する。"""
    from society.engine import checkpoint as _ckpt, scheduler as _sched

    def _mk(name, steps):
        dot = ["run.seed=42", "run.n_agents=10", f"run.n_steps={steps}", f"run.name={name}",
               "lens.assets.enabled=true", "observer.checkpoint_every=150"]
        return Simulation(load_config(dot), out_dir=tmp_path / name)

    straight = _mk("ars_straight", 260)
    straight.run()
    st = getattr(straight, "_assets_state", None)
    assert st is not None and st.get("tau") is not None, \
        "straight 側で τ 実計算が起きていない(境界配置が想定とずれた=テスト再調整が必要)"

    d = tmp_path / "ars_resumed"
    sim1 = _mk("ars_resumed", 150)
    for step in range(150):
        _sched.run_step(sim1, step)
    _ckpt.save(sim1, 150, d / "checkpoint" / "ckpt-000150.pkl.gz")
    sim1.logger.flush_segment()
    sim2 = _mk("ars_resumed", 260)
    sim2.run(resume_from=d)

    a = pq.read_table(straight.out_dir / "l2_metrics.parquet").to_pylist()
    b = pq.read_table(d / "l2_metrics.parquet").to_pylist()
    ka = [{c: r.get(c) for c in AS_COLS} for r in a]
    kb = [{c: r.get(c) for c in AS_COLS} for r in b]
    assert ka and ka == kb, "assets ON の resume==straight が崩れている(L2 資産列不一致)"
