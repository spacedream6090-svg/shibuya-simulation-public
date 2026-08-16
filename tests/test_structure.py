"""第56バッチ タスクB: 社会構造の内生変動 observable のテスト。

検証項目(タスク検収 2026-07-24):
  (R1)   structure OFF は ON と L1 が完全一致(レンズは読むだけ=乱数/LLM 呼を増やさない)。
         OFF は L2 に structure 列(edges_*/edge_churn_rate)を一切足さない(L2 不変)。
  (det)  structure ON は決定論(同一入力で 2 回 → L2 のchurn列がバイト一致)。
  (churn)observe の当日タリー=formed(relation_tier{tier==1})/broken(break{cause!=absence})/
         decayed(break{cause==absence})の計数が合成イベントで正しい + 暦日境界リセット + idempotent。
  (母数) edge_churn_rate の母数=活性関係数(tier≥1 の live 台帳)= churn/max(active,1)。
  (τ)    Kendall τ 自前実装の単体=既知ケース(同一=1 / 逆順=-1 / 定数=None)。
  (τ/N)  第133 レーンR2: τ の実体を凍結 measure._kendall_tau(O(N²))から
         assets._kendall_tau(O(N log N))へ差し替えた同値性=境界/乱数/tie で完全一致・
         rank_stability の出力が差し替え前後でまるごと一致・n=250,000 完走スモーク。
  (固着) _stagnant_intervals の区間出力=閾値以下で連続 N 日の区間化。
  (再構成)reconstruct_churn が L1 relation イベントから日次 churn を in-sim と同一定義で再構成。
  (k)    compute_matched 下で k=free と k=off の generate 呼数が完全一致(観測のみ=自明だが1本)。
  (viz)  mock 合成ラン→analyze_structure→structure.json→dashboard HTML 生成が通る。
         structure.json が無ければ従来出力とバイト同一(後方互換=社会構造タブ非表示)。
"""
from __future__ import annotations

import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "viz"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from society.config import load_config                        # noqa: E402
from society.engine.simulation import Simulation              # noqa: E402
from society.observer import measure as m                     # noqa: E402
from society.observer import structure as st                  # noqa: E402
from society.observer.schema import Event                     # noqa: E402

import make_viewer as mv                                      # noqa: E402
import analyze_structure as ast                               # noqa: E402

VIEWER = REPO_ROOT / "viz" / "make_viewer.py"
ANALYZER = REPO_ROOT / "scripts" / "analyze_structure.py"

ST_COLS = ("edges_formed", "edges_broken", "edges_decayed", "edge_churn_rate")


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
# (R1) structure OFF == ON の L1(relations ON でも構造レンズは読むだけ)・OFF は L2 に列なし
# --------------------------------------------------------------------------- #
def test_structure_is_read_only_l1_invariant(tmp_path):
    base = {"relations.enabled": "true"}
    off = _run(tmp_path, "soff", **base, **{"lens.structure.enabled": "false"})
    on = _run(tmp_path, "son", **base, **{"lens.structure.enabled": "true"})
    assert _l1(off) == _l1(on), "structure ON が L1 を変えた(読むだけでない=R1 違反)"


def test_structure_off_no_l2_columns(tmp_path):
    off = _run(tmp_path, "s2off", **{"lens.structure.enabled": "false"})
    for row in off.logger.metrics:
        for col in ST_COLS:
            assert col not in row, f"structure OFF なのに L2 に {col} 列がある(L2 不変違反)"


def test_structure_on_emits_l2_columns(tmp_path):
    on = _run(tmp_path, "s2on", **{"lens.structure.enabled": "true"})
    assert on.logger.metrics, "L2 が空"
    for row in on.logger.metrics:
        for col in ST_COLS:
            assert col in row, f"structure ON なのに L2 に {col} 列が無い"
    # 件数列は int / 率は float
    r = on.logger.metrics[0]
    assert isinstance(r["edges_formed"], int) and isinstance(r["edge_churn_rate"], float)


def test_structure_on_deterministic_l2(tmp_path):
    a = _run(tmp_path, "sda", **{"lens.structure.enabled": "true", "relations.enabled": "true"})
    b = _run(tmp_path, "sdb", **{"lens.structure.enabled": "true", "relations.enabled": "true"})
    va = [[r["step"], *[r[c] for c in ST_COLS]] for r in a.logger.metrics]
    vb = [[r["step"], *[r[c] for c in ST_COLS]] for r in b.logger.metrics]
    assert va == vb, "structure ON の L2 churn列が決定論でない(2 回で不一致)"


# --------------------------------------------------------------------------- #
# observe の churn 計数(合成 FakeSim)
# --------------------------------------------------------------------------- #
class _FakeLogger:
    def __init__(self, events):
        self.events = events
        self._n_flushed = 0


class _FakeMem:
    def __init__(self, relations):
        self.relations = relations


class _FakeAgent:
    def __init__(self, aid, relations):
        self.id = aid
        self.mem = _FakeMem(relations)


class _FakeSim:
    def __init__(self, events, agents, enabled=True):
        self.logger = _FakeLogger(events)
        self.agents = agents
        self.cfg = {"lens": {"structure": {"enabled": enabled}}}


def _tier(step, sim_min, other, tier, agent=1):
    return Event(step=step, sim_min=sim_min, agent_id=agent, kind="relation_tier",
                 x=0.0, y=0.0, payload={"other": other, "tier": tier, "count": 3})


def _break(step, sim_min, other, to_tier, cause, agent=1):
    return Event(step=step, sim_min=sim_min, agent_id=agent, kind="relation_break",
                 x=0.0, y=0.0, payload={"other": other, "from_tier": to_tier + 1,
                                        "to_tier": to_tier, "cause": cause})


def test_observe_churn_counts():
    """formed=relation_tier{tier==1} / broken=break{cause=contact} / decayed=break{cause=absence}。
    深化(tier==2)は組み替えでないので数えない。母数=活性関係数(tier≥1 の台帳)。"""
    ev = [
        _tier(0, 100, other=2, tier=1),                 # 形成(0→1)
        _tier(0, 100, other=3, tier=2),                 # 深化(1→2)=数えない
        _break(0, 100, other=4, to_tier=0, cause="contact"),   # 断絶
        _break(0, 100, other=5, to_tier=1, cause="absence"),   # 風化(2→1 の降格)
    ]
    # 活性関係(tier≥1)= agent1 に 2 件、agent2 に 1 件 → 母数 3
    agents = [_FakeAgent(1, {2: {"tier": 1}, 3: {"tier": 2}}),
              _FakeAgent(2, {9: {"tier": 3}})]
    sim = _FakeSim(ev, agents)
    st_state = st.observe(sim)
    assert st_state["formed"] == 1 and st_state["broken"] == 1 and st_state["decayed"] == 1
    assert st_state["active"] == 3
    s = st.scalars(sim)
    assert s["edges_formed"] == 1 and s["edges_broken"] == 1 and s["edges_decayed"] == 1
    assert s["edge_churn_rate"] == round(3 / 3, 6)     # churn=3 / active=3


def test_observe_daily_reset():
    ev = [_tier(0, 100, other=2, tier=1)]               # day0: formed 1
    agents = [_FakeAgent(1, {2: {"tier": 1}})]
    sim = _FakeSim(ev, agents)
    st.observe(sim)
    assert sim._struct_state["formed"] == 1
    sim.logger.events.append(_break(144, 1500, other=2, to_tier=0, cause="contact"))  # 翌日
    s2 = st.observe(sim)
    assert s2["day"] == 1 and s2["formed"] == 0 and s2["broken"] == 1  # 当日タリーがリセット


def test_observe_idempotent_within_step():
    sim = _FakeSim([_tier(0, 100, other=2, tier=1)], [_FakeAgent(1, {2: {"tier": 1}})])
    a = st.observe(sim)
    b = st.observe(sim)                                 # 同 step 内の再呼出は二度数えない
    assert a["formed"] == 1 and b["formed"] == 1


def test_observe_off_returns_none():
    sim = _FakeSim([_tier(0, 100, other=2, tier=1)], [_FakeAgent(1, {2: {"tier": 1}})],
                   enabled=False)
    assert st.observe(sim) is None
    assert st.scalars(sim) is None


def test_active_ties_zero_without_tier():
    """relations OFF(tier キー無し)なら活性関係=0 → churn_rate は churn/1(graceful)。"""
    sim = _FakeSim([_break(0, 100, other=2, to_tier=0, cause="contact")],
                   [_FakeAgent(1, {2: {"count": 5}})])   # tier キー無し
    s = st.scalars(sim)
    assert s["edges_broken"] == 1 and s["edge_churn_rate"] == round(1 / 1, 6)


# --------------------------------------------------------------------------- #
# (τ) Kendall τ 自前実装(measure._kendall_tau 再利用)の単体
# --------------------------------------------------------------------------- #
def test_kendall_tau_known_cases():
    assert ast.kendall_tau([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0      # 完全一致
    assert ast.kendall_tau([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0     # 完全逆順
    # 1 ペアだけ入れ替わり(4要素6ペア中1不一致)→ τ=(5-1)/6
    assert abs(ast.kendall_tau([1, 2, 3, 4], [1, 2, 4, 3]) - (4 / 6)) < 1e-6
    assert ast.kendall_tau([1], [1]) is None                       # n<2
    assert ast.kendall_tau([5, 5, 5], [1, 2, 3]) == 0.0            # 片側定数=τ-b 分母0→0.0(measure 仕様)


# --------------------------------------------------------------------------- #
# (τ O(N log N) 化・第133 レーンR2)
#
#   `analyze_structure.kendall_tau` の実体を 凍結 `measure._kendall_tau`(明示二重ループ
#   = O(N²))から `assets._kendall_tau`(マージソート反転数 = O(N log N))へ差し替えた。
#   ★凍結ファイル measure.py は 1 バイトも触っていない = 下の参照は「置き換え前の
#     kendall_tau そのもの」であり、これと新実装が完全一致することを機械固定する。
#   実測(2026-08-16 本機): n=250,000 で 0.60-0.74 秒。旧実装は同 n で外挿 1.5 時間/回
#   (n=500/1000/2000 の実測から n² 外挿)= 前日比+前週比×30日で数日級だった。
# --------------------------------------------------------------------------- #
def _tau_frozen(a, b):
    """差し替え前の `analyze_structure.kendall_tau`(凍結 measure 実装を呼ぶ形)。"""
    if len(a) < 2 or len(b) < 2:
        return None
    t = m._kendall_tau(a, b)
    if t is None or (isinstance(t, float) and math.isnan(t)):
        return None
    return round(float(t), 6)


def test_kendall_tau_matches_frozen_measure_implementation():
    """境界・tie なし・tie 多発・実データ相当(0円の塊)で凍結実装と完全一致。"""
    # --- 境界 ---
    for a, b in ([[1], [1]], [[], []],
                 [[1.0, 2.0], [1.0, 2.0]], [[1.0, 2.0], [2.0, 1.0]],
                 [[5.0] * 8, [5.0] * 8], [[5.0] * 8, [float(i) for i in range(8)]],
                 [list(range(30)), list(range(29, -1, -1))],
                 [[1, 2, 3, 4], [1, 2, 4, 3]]):
        assert ast.kendall_tau(a, b) == _tau_frozen(a, b), (a, b)
    # --- tie なし乱数 ---
    rnd = random.Random(20260816)
    for case in range(20):
        n = rnd.randint(2, 120)
        a = [float(v) for v in rnd.sample(range(10 ** 6), n)]
        b = [float(v) for v in rnd.sample(range(10 ** 6), n)]
        assert ast.kendall_tau(a, b) == _tau_frozen(a, b), f"no-tie case{case}"
    # --- tie 多発(値域を狭める)---
    for case in range(20):
        n = rnd.randint(2, 150)
        span = rnd.choice([1, 2, 3, 5])
        a = [float(rnd.randint(0, span)) for _ in range(n)]
        b = [float(rnd.randint(0, span)) for _ in range(n)]
        assert ast.kendall_tau(a, b) == _tau_frozen(a, b), f"tie case{case}"
    # --- 実データ相当(status / reputation の連続値 + 同値の塊)---
    for case in range(10):
        n = rnd.randint(50, 200)
        a = [0.0 if rnd.random() < 0.3 else rnd.uniform(0, 100.0) for _ in range(n)]
        b = [v * rnd.uniform(0.8, 1.2) if rnd.random() < 0.9 else 0.0 for v in a]
        assert ast.kendall_tau(a, b) == _tau_frozen(a, b), f"real-like case{case}"


def _rep_events(days, vals_by_day):
    """`reputation_update` の合成 L1(rank_stability の評判経路を踏ませる)。"""
    by_day = {}
    for d in days:
        by_day[d] = [{"kind": "reputation_update", "agent_id": aid,
                      "step": d * ast.STEPS_PER_DAY, "payload": {"new": v}}
                     for aid, v in sorted(vals_by_day[d].items())]
    return by_day


def test_rank_stability_output_is_identical_after_the_swap(tmp_path, monkeypatch):
    """差し替え前後で `rank_stability` の出力が**まるごと**一致する(τ 系列も race も)。

    ★L3 が無い run dir なので評判経路(`_reputation_by_day`)を通る = 前日比 τ と
      前週比 τ の両方が実際に計算される日数(10 日)を用意する。
    """
    rnd = random.Random(31)
    days = list(range(10))
    vals = {}
    base = {aid: rnd.uniform(0, 50.0) for aid in range(40)}
    for d in days:                       # 日ごとに少しだけ順位が動く + 同値の塊も混ぜる
        base = {aid: (0.0 if rnd.random() < 0.2 else v + rnd.uniform(-3, 3))
                for aid, v in base.items()}
        vals[d] = dict(base)
    by_day = _rep_events(days, vals)
    run_dir = str(tmp_path)              # l3_snapshots.parquet も agents.json も無い

    got = ast.rank_stability(run_dir, by_day, days, top_k=10)
    monkeypatch.setattr(ast, "_tau_impl", m._kendall_tau)   # 凍結実装へ戻す
    ref = ast.rank_stability(run_dir, by_day, days, top_k=10)
    assert got == ref, "τ 実装の差し替えで rank_stability の出力が変わった"
    assert got["source"] == "reputation", got["source"]
    assert any(t is not None for t in got["tau_prev_day"]), "前日比 τ が 1 つも出ていない"
    assert got["tau_prev_week"][7] is not None, "前週比 τ が出ていない(検査の前提が崩れた)"


@pytest.mark.slow
def test_kendall_tau_scales_to_the_finals_population():
    """★本選規模 n=250,000 の完走スモーク(旧 O(N²) 実装ならここで数時間かかる)。

    値の正しさは上の同値性テストが担保するので、ここは完走時間と粗い健全性だけを見る。
    """
    n = 250_000
    rnd = random.Random(5)
    a = [rnd.uniform(0, 5_000_000.0) for _ in range(n)]
    b = list(a)
    b[10], b[20] = b[20], b[10]                    # 1 ペアだけ反転
    t0 = time.perf_counter()
    tau = ast.kendall_tau(a, b)
    elapsed = time.perf_counter() - t0
    assert 0.999 < tau < 1.0, tau
    assert elapsed < 60.0, f"n=250,000 に {elapsed:.1f}s(O(N log N) になっていない)"


# --------------------------------------------------------------------------- #
# (固着) 連続 N 日区間の検出
# --------------------------------------------------------------------------- #
def test_stagnant_intervals():
    days = list(range(6))                                          # [0,1,2,3,4,5]
    flags = [True, True, True, False, True, True]
    segs = ast._stagnant_intervals(flags, days, min_days=3)
    assert segs == [{"start_day": 0, "end_day": 2, "len": 3}]      # 末尾の2連続は N=3 未満
    segs2 = ast._stagnant_intervals(flags, days, min_days=2)
    assert [s["len"] for s in segs2] == [3, 2]
    # 平均値の付与
    vals = [0.05, 0.02, 0.04, None, 0.9, 0.9]
    segs3 = ast._stagnant_intervals(flags, days, min_days=3, values=vals)
    assert abs(segs3[0]["mean_value"] - (0.05 + 0.02 + 0.04) / 3) < 1e-6


# --------------------------------------------------------------------------- #
# (再構成) reconstruct_churn が L1 relation イベントから日次 churn を in-sim と同一定義で再構成
# --------------------------------------------------------------------------- #
def _ev(step, sim_min, kind, payload, agent=1):
    return {"step": step, "sim_min": sim_min, "agent_id": agent, "kind": kind,
            "x": 0.0, "y": 0.0, "payload": payload}


def test_reconstruct_churn_matches_definitions():
    by_day = {
        0: [_ev(0, 100, "relation_tier", {"other": 2, "tier": 1}),           # 形成
            _ev(0, 100, "relation_tier", {"other": 3, "tier": 1}),           # 形成
            _ev(0, 100, "relation_break", {"other": 4, "to_tier": 0, "cause": "contact"})],  # 断絶
        1: [_ev(144, 1500, "relation_break", {"other": 2, "to_tier": 0, "cause": "absence"})],  # 風化
    }
    days = [0, 1]
    ch = ast.reconstruct_churn(by_day, days)
    assert ch["edges_formed"] == [2, 0]
    assert ch["edges_broken"] == [1, 0]
    assert ch["edges_decayed"] == [0, 1]
    assert ch["has_relation_events"] is True
    # 母数(当日開始の活性関係数): day0=0(空から) / day1=2(day0 で other2,3 が tier1・other4 は 0 で除外)
    assert ch["active_ties"] == [0, 2]
    assert ch["churn_rate"][0] == round(3 / 1, 6) and ch["churn_rate"][1] == round(1 / 2, 6)


def test_reconstruct_churn_graceful_without_relations():
    by_day = {0: [_ev(0, 100, "speak", {"hearers": [2]})]}
    ch = ast.reconstruct_churn(by_day, [0])
    assert ch["has_relation_events"] is False
    assert ch["edges_formed"] == [0] and ch["churn_rate"] == [0.0]


def test_centrality_and_community_from_conversation():
    """中心性 turnover とコミュニティ変化が会話グラフ(speak/dm)の日次窓から出る(既存資産の再利用)。"""
    by_day = {
        0: [_ev(0, 100, "speak", {"hearers": [1, 2]}, agent=0),
            _ev(0, 105, "speak", {"hearers": [0]}, agent=1)],
        1: [_ev(144, 1500, "speak", {"hearers": [4, 5]}, agent=3)],
    }
    days = [0, 1]
    cent = ast.centrality_churn(by_day, days, top_k=3)
    assert cent["turnover"][0] is None                             # 初日は前日なし
    assert 0.0 <= cent["turnover"][1] <= 1.0
    comm = ast.community_change(by_day, days, n_agents=6)
    assert len(comm["n_communities"]) == 2
    assert comm["change_rate"][0] is None


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
           "lens.structure.enabled=true", "relations.enabled=true",
           "controls.mode=compute_matched", f"k.writeback={writeback}"]
    sim = Simulation(load_config(dot), out_dir=tmp_path / name)
    sim.llm = _FixedLLM(json.dumps({"action": "speak", "text": "x"}, ensure_ascii=False))
    sim.run()
    return sim


def test_structure_call_count_k_invariant(tmp_path):
    free = _run_k(tmp_path, "sk_free", writeback="free")
    off = _run_k(tmp_path, "sk_off", writeback="off")
    assert free.llm.calls == off.llm.calls and free.llm.calls > 0, \
        f"structure の呼数が k に依存(R1 違反): free={free.llm.calls} off={off.llm.calls}"


# --------------------------------------------------------------------------- #
# (viz) mock ラン→analyze_structure→structure.json→dashboard HTML 生成
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


def _write_run(tmp_path: Path, name: str, events: list[dict]) -> Path:
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
              for i in range(6)]
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
    return run_dir


def _mock_events():
    ev = []
    # 3 日分: 会話(中心性/コミュニティ)+ 関係イベント(churn)+ 評判(順位 τ)
    for d in range(3):
        base = d * 1440 + 100
        step = d * 144
        ev += [
            {"step": step, "agent_id": 0, "kind": "speak", "sim_min": base,
             "payload": {"text": "やあ", "hearers": [1, 2]}},
            {"step": step, "agent_id": 1, "kind": "speak", "sim_min": base + 5,
             "payload": {"text": "どうも", "hearers": [0, 2]}},
            {"step": step, "agent_id": 3, "kind": "speak", "sim_min": base + 5,
             "payload": {"text": "こんにちは", "hearers": [4]}},
            {"step": step, "agent_id": 0, "kind": "relation_tier", "sim_min": base,
             "payload": {"other": 1, "tier": 1}},
            {"step": step, "agent_id": 0, "kind": "reputation_update", "sim_min": base,
             "payload": {"old": d, "new": d + 1.0, "cause": "mention"}},
            {"step": step, "agent_id": 1, "kind": "reputation_update", "sim_min": base,
             "payload": {"old": 0.0, "new": 0.5, "cause": "mention"}},
        ]
    return ev


def test_analyze_structure_and_dashboard(tmp_path):
    rd = _write_run(tmp_path, "sz", _mock_events())
    res = ast.analyze(str(rd))
    # 事後層の出力スキーマの要点
    assert set(res["churn"]) >= {"edges_formed", "edges_broken", "edges_decayed",
                                 "active_ties", "churn_rate", "has_relation_events"}
    assert res["rank"]["source"] == "reputation"          # L3 status 無し→評判へ後退
    assert res["churn"]["has_relation_events"] is True
    assert "combined" in res["stagnation"] and "by_signal" in res["stagnation"]
    (rd / "structure.json").write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    data = mv.build_data(rd, include_traffic=False)
    assert "structure" in data
    r = subprocess.run([sys.executable, str(VIEWER), str(rd)], capture_output=True)
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    dash = (rd / "dashboard.html").read_text(encoding="utf-8")
    viewer = (rd / "viewer.html").read_text(encoding="utf-8")
    assert 'data-tab="structure"' in dash
    for sym in ("renderStructure", "structRender"):
        assert sym in dash, f"{sym} が dashboard に無い"
    assert '"structure":' in dash
    assert 'data-tab="structure"' not in viewer          # 地図側にはタブ無し


def test_analyze_structure_cli(tmp_path):
    rd = _write_run(tmp_path, "scli", _mock_events())
    r = subprocess.run([sys.executable, str(ANALYZER), str(rd)], capture_output=True)
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    assert (rd / "structure.json").exists() and (rd / "structure_report.md").exists()
    js = json.loads((rd / "structure.json").read_text(encoding="utf-8"))
    assert js["n_days"] == 3 and len(js["days"]) == 3


def test_dashboard_no_structure_backward_compat(tmp_path):
    """structure.json 無しのランでは build_data に structure を足さず・社会構造タブ非表示(後方互換)。"""
    ev = [{"step": s, "agent_id": 0, "kind": "arrive", "sim_min": 420 + s * 10,
           "payload": {"name": "路上"}} for s in range(3)]
    rd = _write_run(tmp_path, "sz_none", ev)
    data = mv.build_data(rd, include_traffic=False)
    assert "structure" not in data
    r = subprocess.run([sys.executable, str(VIEWER), str(rd)], capture_output=True)
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    dash = (rd / "dashboard.html").read_text(encoding="utf-8")
    assert 'data-tab="structure"' not in dash and "function structRender" not in dash
