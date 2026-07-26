"""第63バッチ フェーズ2: endogenous_accept treatment 実験プロトコルのテスト。

検証項目:
  (manifest) conf/experiments/endogenous_accept*.yaml の展開: 6セル × seed 列=CRN 同一 seed
             ペア・override 内容(k.writeback / endogenous_accept.enabled / 共通 ON 群)・
             ペア隣接の展開順(時間切れ時に H3 両端ペアから確保)。
  (sign-flip) 既知ペア差 → p 値の手計算一致(小 n の全 2^n 列挙経路)・n>12 のモンテカルロ経路
             (決定論=2回同値・列挙との近似一致)・退化(全ゼロ)・Phipson&Smyth の (b+1)/(m+1)。
  (block)    副検定(日次ブロック符号反転)の決定論と自明ケース。
  (analyze)  合成ラン群(config.yaml+L1+L2 を手書き)→ ペア組み立て・既知のペア差・交互作用の
             差の差・KPI(L2 由来)・出力ファイル(json/parquet/md)・決定論(2回実行で JSON
             バイト同一)・make_endo_report の HTML 生成。
  (judge)    機械判定(計画書§3: 符号一致≥3seed + 乖離±15pp ゲート)の直接検証。
実 LLM 禁止(全て合成データ or mock 相当)。pandas/duckdb 不使用。
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import analyze_endo_treatment as ana        # noqa: E402
import make_endo_report as rep              # noqa: E402


def _run_experiment_mod():
    spec = importlib.util.spec_from_file_location(
        "run_experiment", REPO / "scripts" / "run_experiment.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# (manifest) 6セル × CRN seed ペア
# --------------------------------------------------------------------------- #
CELLS = ("koff_eoff", "koff_eon", "kfree_eoff", "kfree_eon",
         "kdg_eoff", "kdg_eon")


def test_manifest_main_expansion():
    m = _run_experiment_mod()
    manifest = m.load_manifest(REPO / "conf" / "experiments" /
                               "endogenous_accept.yaml")
    assert manifest["name"] == "endo14"
    assert manifest["seeds"] == [11, 13, 17, 19, 23]
    names = [c["name"] for c in manifest["conditions"]]
    # ペア隣接 + 両端の k(off→free)を先=時間切れでも H3 両端ペアから確保
    assert names == list(CELLS)
    jobs = m.expand_matrix(manifest)
    assert len(jobs) == 6 * 5 == 30
    # CRN: 全セル同一 seed 列(同一 seed の OFF/ON がペアになる)
    by_cond: dict[str, list] = {}
    for j in jobs:
        by_cond.setdefault(j["condition"], []).append(j["seed"])
    assert all(v == [11, 13, 17, 19, 23] for v in by_cond.values())
    # override 内容: k とendo フラグがセル通りに dotlist 化される
    j0 = next(j for j in jobs if j["run"] == "endo14_koff_eoff_s11")
    assert "k.writeback=off" in j0["overrides"]
    assert "relations.endogenous_accept.enabled=false" in j0["overrides"]
    j1 = next(j for j in jobs if j["run"] == "endo14_kfree_eon_s23")
    assert "k.writeback=free" in j1["overrides"]
    assert "relations.endogenous_accept.enabled=true" in j1["overrides"]
    jd = next(j for j in jobs if j["run"] == "endo14_kdg_eon_s11")
    assert "k.writeback=degraded" in jd["overrides"]
    assert "k.degraded_alpha=0.5" in jd["overrides"]
    # 共通 ON 群(フェーズ1材料の前提+タスクB レンズ)が全ランに載る
    for need in ("run.n_agents=100", "run.n_steps=2016", "joint.enabled=true",
                 "relations.enabled=true", "schedule.enabled=true",
                 "friend_graph.enabled=true", "planning.framework.enabled=true",
                 "prompts.dialog_history=true", "lens.structure.enabled=true"):
        assert need in j0["overrides"], f"common に {need} が無い"
    assert manifest["profile"] == "conf/daily.yaml"


def test_manifest_pilot_and_wiring_expansion():
    m = _run_experiment_mod()
    pilot = m.load_manifest(REPO / "conf" / "experiments" /
                            "endogenous_accept_pilot.yaml")
    assert [c["name"] for c in pilot["conditions"]] == list(CELLS)
    assert pilot["seeds"] == [11, 13, 17]
    jobs = m.expand_matrix(pilot)
    assert len(jobs) == 18
    assert "run.n_steps=1008" in jobs[0]["overrides"]
    assert "model.backend=mock" in jobs[0]["overrides"]      # 予備=mock 固定
    wiring = m.load_manifest(REPO / "conf" / "experiments" /
                             "endogenous_accept_wiring.yaml")
    wjobs = m.expand_matrix(wiring)
    assert len(wjobs) == 12                                  # 6セル × seed2
    assert "run.n_agents=60" in wjobs[0]["overrides"]
    assert "model.backend=mock" in wjobs[0]["overrides"]


# --------------------------------------------------------------------------- #
# (sign-flip) 手計算一致・全列挙経路・MC 経路
# --------------------------------------------------------------------------- #
def test_sign_flip_exhaustive_hand_computation():
    """[2,3,4]: 2^3=8 通りの符号列挙で |mean|≥3 は +++ と --- の 2 通り → p=2/8=0.25。"""
    t = ana.sign_flip_test([2.0, 3.0, 4.0])
    assert t["method"] == "exhaustive" and t["n_perm"] == 8
    assert abs(t["p"] - 0.25) < 1e-12
    assert t["mean_diff"] == 3.0 and t["n_pos"] == 3 and t["n_neg"] == 0
    assert abs(t["min_p"] - 0.25) < 1e-12          # 両側理論下限 2/2^n
    # 対称性: 全符号反転で p 同一
    t2 = ana.sign_flip_test([-2.0, -3.0, -4.0])
    assert t2["p"] == t["p"] and t2["mean_diff"] == -3.0
    # [1,1,-3]: |mean|=1/3 は全 8 通りで超えられる → p=1.0(手で列挙して確認済み)
    t3 = ana.sign_flip_test([1.0, 1.0, -3.0])
    assert abs(t3["p"] - 1.0) < 1e-12
    # n=1 は両側で常に p=1
    assert ana.sign_flip_test([5.0])["p"] == 1.0
    # 退化(全ゼロ)
    t0 = ana.sign_flip_test([0.0, 0.0])
    assert t0["method"] == "degenerate_zero" and t0["p"] == 1.0
    # 空・None のみ
    assert ana.sign_flip_test([None, None])["p"] is None


def test_sign_flip_monte_carlo_deterministic_and_close_to_exact():
    ds13 = [float(i) for i in range(1, 14)]                  # n=13 > 12 → MC
    a = ana.sign_flip_test(ds13)
    b = ana.sign_flip_test(ds13)
    assert a["method"] == "monte_carlo" and a == b           # 決定論(rng seed 固定)
    assert 0.0 < a["p"] <= 1.0
    assert a["min_p"] == round(1 / (ana.MC_ITER + 1), 6)     # (b+1)/(m+1) の下限
    # 同一データで列挙と MC が近い(n=6 を exhaust_max=5 で強制 MC): 厳密 p=2/64
    ds6 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    exact = ana.sign_flip_test(ds6)
    mc = ana.sign_flip_test(ds6, exhaust_max=5)
    assert exact["method"] == "exhaustive" and abs(exact["p"] - 2 / 64) < 1e-12
    assert mc["method"] == "monte_carlo" and abs(mc["p"] - exact["p"]) < 0.02


def test_block_signflip_basic():
    zero = ana.block_signflip_test([[0.0] * 10, [0.0] * 10])
    assert zero["method"] == "degenerate_zero" and zero["p"] == 1.0
    strong = [[1.0] * 14 for _ in range(3)]                  # 全日 +1 の差 × 3seed
    a = ana.block_signflip_test(strong)
    b = ana.block_signflip_test(strong)
    assert a == b, "block 副検定が決定論でない"
    assert a["p"] < 0.01 and a["block_len"] == 2             # round(14^(1/3))=2
    assert ana.block_signflip_test([])["p"] is None


# --------------------------------------------------------------------------- #
# (analyze) 合成ラン群 → 既知ペア差・交互作用・KPI・決定論
# --------------------------------------------------------------------------- #
def _ev(step, sim_min, kind, payload, agent=1):
    return {"step": step, "sim_min": sim_min, "agent_id": agent, "kind": kind,
            "x": 0.0, "y": 0.0, "payload": payload}


def _base_events():
    """3 日分の会話(中心性・コミュニティ用)+ day0 の関係形成 1 件(全ラン共通)。"""
    ev = []
    for d in range(3):
        base = d * 1440 + 500
        step = d * 144
        ev += [
            _ev(step, base, "speak", {"text": "a", "hearers": [2, 3]}, agent=1),
            _ev(step, base + 10, "speak", {"text": "b", "hearers": [1]}, agent=2),
        ]
    ev.append(_ev(0, 500, "relation_tier", {"other": 2, "tier": 1}, agent=1))
    return ev


def _extra_forms(day_others: dict[int, list[int]]):
    """{day: [other,...]} の追加 relation_tier(tier=1=形成)イベント。"""
    ev = []
    for d, others in sorted(day_others.items()):
        for o in others:
            ev.append(_ev(d * 144, d * 1440 + 600, "relation_tier",
                          {"other": o, "tier": 1}, agent=1))
    return ev


def _write_run(tmp_path: Path, name: str, *, k: str, endo: bool, seed: int,
               events: list[dict], l2_joint: dict | None = None) -> Path:
    rd = tmp_path / name
    rd.mkdir(parents=True, exist_ok=True)
    alpha_line = "  degraded_alpha: 0.5\n" if k == "degraded" else ""
    (rd / "config.yaml").write_text(
        "run:\n"
        f"  seed: {seed}\n"
        "  n_agents: 6\n"
        "  start_tod: \"07:00\"\n"
        "k:\n"
        f"  writeback: \"{k}\"\n" + alpha_line +
        "relations:\n"
        "  endogenous_accept:\n"
        f"    enabled: {'true' if endo else 'false'}\n",
        encoding="utf-8")
    cols = {
        "step": [int(e["step"]) for e in events],
        "sim_min": [int(e["sim_min"]) for e in events],
        "agent_id": [int(e["agent_id"]) for e in events],
        "kind": [str(e["kind"]) for e in events],
        "x": [0.0] * len(events),
        "y": [0.0] * len(events),
        "payload": [json.dumps(e["payload"], ensure_ascii=False) for e in events],
    }
    pq.write_table(pa.table(cols), rd / "l1_events.parquet")
    if l2_joint is not None:
        steps = [0, 144, 288]            # 07:00 起点 → day 0/1/2 の代表行
        tbl = {"step": pa.array(steps, pa.int32())}
        for c, v in l2_joint.items():
            tbl[c] = pa.array([float(v)] * len(steps), pa.float64())
        pq.write_table(pa.table(tbl), rd / "l2_metrics.parquet")
    return rd


def _build_synthetic(tmp_path: Path) -> list[str]:
    """2 k 水準(off/free)× endo(off/on)× seed(1,2)= 8 ラン。
    edge churn(skip_days=1 → day1,2 の平均)を手計算可能に設計:
      OFF:      day1,2 とも形成 0 → churn_rate 平均 0
      ON(koff): day1 形成1(母数1=rate 1.0)/ day2 形成1(母数2=rate 0.5)→ 平均 0.75
      ON(kfree):day1 形成2(rate 2.0)/ day2 形成2(母数3=rate 2/3)→ 平均 1.333333
    → ペア差: koff=+0.75 / kfree=+1.333333、交互作用(free−off)=+0.583333。"""
    kpi = {"joint_accept_rate": 0.5, "joint_endo_share": 0.4,
           "joint_accept_calib_gap": 0.1, "joint_fulfill_rate": 0.6}
    dirs = []
    for seed in (1, 2):
        dirs.append(_write_run(tmp_path, f"syn_koff_eoff_s{seed}", k="off",
                               endo=False, seed=seed, events=_base_events()))
        dirs.append(_write_run(
            tmp_path, f"syn_koff_eon_s{seed}", k="off", endo=True, seed=seed,
            events=_base_events() + _extra_forms({1: [3], 2: [4]}),
            l2_joint=kpi))
        dirs.append(_write_run(tmp_path, f"syn_kfree_eoff_s{seed}", k="free",
                               endo=False, seed=seed, events=_base_events()))
        dirs.append(_write_run(
            tmp_path, f"syn_kfree_eon_s{seed}", k="free", endo=True, seed=seed,
            events=_base_events() + _extra_forms({1: [3, 4], 2: [5, 6]}),
            l2_joint=kpi))
    return [str(d) for d in dirs]


def test_analyze_synthetic_pairs_interaction_kpi(tmp_path):
    dirs = _build_synthetic(tmp_path)
    out = tmp_path / "out"
    res = ana.analyze(dirs, out=str(out))
    assert res["k_order"] == ["off", "free"]
    # ペア差の手計算一致(day0 は skip=friend_graph 初期化バースト除外の既定)
    po = res["pairs"]["off"]["edge_churn_rate"]
    assert abs(po["1"]["diff"] - 0.75) < 1e-6
    assert abs(po["2"]["diff"] - 0.75) < 1e-6
    pf = res["pairs"]["free"]["edge_churn_rate"]
    assert abs(pf["1"]["diff"] - 1.333333) < 1e-5
    # sign-flip: n=2 同符号 → 全列挙 2^2=4 中 |mean|≥obs は ++/-- → p=0.5
    t = res["tests"]["pair"]["off"]["edge_churn_rate"]
    assert t["method"] == "exhaustive" and abs(t["p"] - 0.5) < 1e-12
    assert t["n_pos"] == 2 and t["n_neg"] == 0
    # 交互作用(free−off)の差の差 = 1.333333 − 0.75 = 0.583333(seed 両方)
    it = res["tests"]["interaction"]["free-off"]["edge_churn_rate"]
    assert len(it["_diffs"]) == 2
    assert all(abs(d - 0.583333) < 1e-5 for d in it["_diffs"])
    # KPI は ON セルの L2 由来(OFF は L2 なし=None にならず ON 平均が入る)
    assert abs(res["kpi"]["off"]["joint_accept_calib_gap"] - 0.1) < 1e-9
    assert abs(res["kpi"]["free"]["joint_fulfill_rate"] - 0.6) < 1e-9
    # 乖離ゲート(0.1 ≤ 0.15)は内・ただし seed2 本では符号一致 3 に届かない=正直な不成立
    jd = res["judgment"]
    assert jd["accept_gap_gate"]["pass"] is True
    assert jd["H1"]["pass"] is False
    assert jd["H1"]["by_k"]["off"]["enough_seeds"] is False
    assert "検出力" in jd["power_note"] or "α=0.05" in jd["power_note"]
    # 出力ファイル一式
    for f in ("endo_treatment.json", "endo_pairs.parquet",
              "endo_tests.parquet", "endo_treatment_report.md"):
        assert (out / f).exists(), f"{f} が出力されていない"
    tbl = pq.read_table(out / "endo_pairs.parquet")
    assert set(tbl.column_names) == {"k", "metric", "seed", "off", "on", "diff"}
    assert tbl.num_rows == 2 * len(ana.STRUCT_METRICS) * 2   # k2 × metric5 × seed2


def test_analyze_deterministic_and_report_html(tmp_path):
    dirs = _build_synthetic(tmp_path)
    out1, out2 = tmp_path / "o1", tmp_path / "o2"
    ana.analyze(dirs, out=str(out1))
    ana.analyze(dirs, out=str(out2))
    j1 = (out1 / "endo_treatment.json").read_bytes()
    j2 = (out2 / "endo_treatment.json").read_bytes()
    assert j1 == j2, "analyze が決定論でない(2 回実行で JSON 不一致)"
    # HTML レポート生成(自己完結・入力同一ならバイト同一)
    res = json.loads(j1.decode("utf-8"))
    h1 = rep.build_html(res)
    h2 = rep.build_html(res)
    assert h1 == h2
    assert "<svg" in h1 and "endogenous_accept" in h1
    assert "cdn" not in h1.lower() and "http://" not in h1 and "https://" not in h1, \
        "自己完結 HTML に外部参照が混入"
    (out1 / "endo_report.html").write_text(h1, encoding="utf-8")
    assert (out1 / "endo_report.html").stat().st_size > 2000


# --------------------------------------------------------------------------- #
# (judge) 機械判定の直接検証(計画書§3)
# --------------------------------------------------------------------------- #
def _pairs_for_judge(diffs_by_k: dict[str, list[float]],
                     stag_on: float = 2.0) -> dict:
    pairs: dict = {}
    for k, ds in diffs_by_k.items():
        pairs[k] = {
            "edge_churn_rate": {str(i + 1): {"off": 0.0, "on": d, "diff": d}
                                for i, d in enumerate(ds)},
            "stagnant_days": {str(i + 1): {"off": 0.0, "on": stag_on,
                                           "diff": stag_on}
                              for i in range(len(ds))},
        }
    return pairs


def test_judge_h1_h3_gate():
    ks = ["off", "free"]
    # H1: free で 3 seed 符号一致(+)→ 成立。gate: |gap|=0.05 ≤ 0.15 → 内
    pairs = _pairs_for_judge({"off": [0.0, 0.0, 0.0],
                              "free": [0.1, 0.2, 0.3]})
    inter = {"free-off": {"edge_churn_rate":
                          {"_diffs": [0.1, 0.2, 0.3], "n_pos": 3, "n_neg": 0}}}
    kpi = {"off": {"joint_accept_calib_gap": 0.05},
           "free": {"joint_accept_calib_gap": -0.05}}
    jd = ana.judge(pairs, {"interaction": inter}, kpi, ks)
    assert jd["H1"]["pass"] is True and jd["H1"]["by_k"]["free"]["consistent"]
    assert jd["H3"]["pass"] is True and jd["H3"]["primary_contrast"] == "free-off"
    assert jd["accept_gap_gate"]["pass"] is True
    assert jd["phase3_go"] is True
    # H2: H1 成立の k で ON の固着日数 > 0 → 併発
    assert jd["H2"]["pass"] is True
    # gate 超過(+0.2 > 0.15)で phase3 NO-GO
    kpi_bad = {"off": {"joint_accept_calib_gap": 0.2},
               "free": {"joint_accept_calib_gap": 0.05}}
    jd2 = ana.judge(pairs, {"interaction": inter}, kpi_bad, ks)
    assert jd2["accept_gap_gate"]["pass"] is False and jd2["phase3_go"] is False
    # 符号割れ(2+ / 1−)は不成立(≥3 かつ多数でない)
    pairs3 = _pairs_for_judge({"off": [0.0, 0.0, 0.0],
                               "free": [0.1, 0.2, -0.3]})
    jd3 = ana.judge(pairs3, {"interaction": {}}, kpi, ks)
    assert jd3["H1"]["pass"] is False and jd3["H3"]["pass"] is False
