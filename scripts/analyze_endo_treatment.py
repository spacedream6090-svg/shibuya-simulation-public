#!/usr/bin/env python
"""第63バッチ フェーズ2: endogenous_accept treatment 比較の横断解析(事後層・読むだけ)。

    python scripts/analyze_endo_treatment.py "runs/endo14_*"
    python scripts/analyze_endo_treatment.py "runs/endo7_*" --out runs/_endo7

設計正典: docs/plans/endogenous-relations-plan.md §3(6セル=endogenous_accept {OFF,ON} ×
k {off, degraded(α0.5), free}・CRN同一seedペア・sign-flip permutation)。
実験ランは scripts/run_experiment.py + conf/experiments/endogenous_accept*.yaml が生成する。

■ 何を測るか
  - ペア差分析(CRN): 同一 seed の OFF/ON ペアで、タスクB構造指標
      edge_churn_rate(edge組み替え率)/ community_change_rate(コミュニティJaccard変化率)/
      centrality_turnover(中心性入れ替わり)/ rank_tau_prev_day(順位固着 Kendall τ)/
      stagnant_days(固着延べ日数=detect_stagnation)
    の per-run スカラー差(ON−OFF)を計算する。計算は analyze_structure.py の関数を import
    再利用(単一の源=in-sim L2 と同一定義)。
  - フェーズ1 KPI(L2 4列): joint_accept_rate / joint_endo_share / joint_accept_calib_gap /
    joint_fulfill_rate。**ON セルのみ存在**(OFF は joint_invite を記録しない=較正抽選のまま)
    ため差でなく ON 水準として報告し、承諾率乖離 ±15pp ゲート(計画書§3)の判定に使う。

■ 検定(すべて決定論)
  主検定: **seedペア差の sign-flip permutation test**(one-sample, 両側)。
    - CRN により同一 seed の OFF/ON ランは共通乱数を共有する=ペア差が交換可能単位。
    - n ≤ 12 は全 2^n 符号反転を厳密列挙(paired sign-flip の標準実装。例: evalci
      arXiv:2607.04429 §sign-flip も n≤12 で全列挙)。n > 12 はモンテカルロ1万回
      (numpy default_rng(0) 固定=決定論)で p=(b+1)/(m+1)(Phipson & Smyth 2010
      "Permutation P-values Should Never Be Zero", doi:10.2202/1544-6115.1585)。
    - ネットワーク構造比較の注意: ノード/エッジ単位の素朴な置換はネットワーク内の非独立性で
      **過大有意**になる(Farine & Carter 2022 の double permutation の趣旨,
      doi:10.1111/2041-210X.13741)。本解析の主検定は **ラン(seedペア)を交換単位**に取る
      ことでネットワーク内依存を検定単位から外す(ノード置換をしない)。
  副検定(感度分析): **ラン内日次系列のブロック符号反転**。日次ペア差系列 d_t を連続ブロック
    (既定長 max(2, round(T^(1/3)))=ブロックリサンプリングの慣用長。Künsch 1989 の moving
    block bootstrap の趣旨=ブロック内の時間依存を保存)ごとに符号反転し、日次差の全平均を
    統計量とする。時間自己相関を保ったまま「日次でも差が出るか」を見る補助。
  交互作用(H3): k×endo の差の差(interaction)= 各 seed の diff(k_hi) − diff(k_lo) に
    同じ sign-flip。本命は free−off(計画書 H3: k↑で ON のみ構造変動が増える)。

■ 検出力の正直な注記(出力にも印字)
  n=3〜5 seed ペアの sign-flip は両側 p の理論下限が 2/2^n(n=3: 0.25 / n=5: 0.0625)で
  **α=0.05 に届き得ない**。よって有意性でなく「符号一致(≥3 seed)+効果量の報告」を主とする
  (計画書§3 の合否基準そのもの)。

■ 出力(--out、既定 runs/_endo_treatment)
  endo_treatment.json        … 全結果(セル・per-runスカラー・日次系列・ペア差・検定・機械判定)
  endo_pairs.parquet         … ペア差の長形式テーブル(k × metric × seed)
  endo_tests.parquet         … 検定結果テーブル(pair / interaction / block)
  endo_treatment_report.md   … 日本語要約 + 仮説 H1/H2/H3 の機械判定
依存: pyarrow / numpy / omegaconf のみ(pandas・duckdb 禁止=リポジトリ不変則)。
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from society.observer import measure as m  # noqa: E402

import analyze_structure as _ast  # noqa: E402  (タスクB指標の単一の源を再利用)
import run_dt                    # noqa: E402  (W2-3: ランの Δt の単一の源)

# --------------------------------------------------------------------------- #
# 定数(決定論のためスクリプト定数。CLI で上書き可)
# --------------------------------------------------------------------------- #
# W2-3: **ラン依存**(run.dt_min)。既定は正準 Δt=10 = 従来と 1 ビットも変わらない。
STEP_MIN = run_dt.CANON_DT_MIN   # 1 step の sim 分
SKIP_DAYS = 1                    # per-run スカラーから除くウォームアップ日数(day0 は
                                 # friend_graph 注入等の初期化バーストが churn を汚すため)
MC_ITER = 10000                  # sign-flip モンテカルロ回数(n>EXHAUST_MAX のとき)
EXHAUST_MAX = 12                 # この n 以下は全 2^n 列挙(厳密)
GATE_PP = 0.15                   # 承諾率乖離ゲート ±15pp(計画書§3)
MIN_AGREE = 3                    # 符号一致とみなす最少 seed 数(計画書§3「3seed以上で符号一致」)

STRUCT_SERIES = ("edge_churn_rate", "community_change_rate",
                 "centrality_turnover", "rank_tau_prev_day")
STRUCT_METRICS = STRUCT_SERIES + ("stagnant_days",)
# フェーズ3(第64バッチ)の invite 2列は L2 由来なので KPI_COLS へ接続(第63の引き継ぎ事項=
# 1定数追記で _l2_kpi_daily→kpi 集計→レポートへ載る)。endogenous_invite OFF のランには列自体が
# 無く自動で欠測(None)扱い=既存 accept 実験の解析はバイト不変。
# フェーズ4(第65バッチ)の quality 1列も同じ理由で接続(列が無いラン=OFF は自動で欠測扱い)。
# 第70バッチ(IDEA①)のエコー 3 列も同じ配線で接続する。これらは **常設列** なので全ランに
# 存在するが、observer.echo.enabled=false のランでは列が無く自動で欠測扱い(既存解析はバイト不変)。
# 目的は「条件間で伝播 KPI が動いた」ときに、それが LLM の反復癖の差ではないことを同じ表で示すこと。
KPI_COLS = ("joint_accept_rate", "joint_endo_share",
            "joint_accept_calib_gap", "joint_fulfill_rate",
            "invite_weak_tie_rate", "invite_endo_share",
            "quality_magnitude_mean",
            "echo_max", "self_similarity_mean", "transmission_novel_rate")
PRIMARY_METRIC = "edge_churn_rate"        # H1/H3 の主指標(計画書§3)


# --------------------------------------------------------------------------- #
# セル読み取り(analyze_sweep._read_cond と同じ正規化。k ラベル + endo フラグ)
# --------------------------------------------------------------------------- #
def _k_label(writeback, alpha) -> str:
    if writeback == "degraded" and alpha is not None:
        return "dg%g" % float(alpha)
    return str(writeback)


def _k_eff(writeback, alpha):
    return {"off": 0.0, "free": 1.0, "sham": 0.0}.get(
        writeback, float(alpha) if alpha is not None else None)


def read_cell(run_dir: str) -> dict:
    """run の config.yaml スナップショットから (k ラベル, endo, seed) を読む。"""
    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    wb = OmegaConf.select(cfg, "k.writeback")
    if isinstance(wb, bool):          # 無引用の off を YAML が False と読む対策(config.py と同じ)
        wb = "off" if not wb else "free"
    wb = str(wb)
    alpha = OmegaConf.select(cfg, "k.degraded_alpha")
    alpha = float(alpha) if alpha is not None else None
    endo = bool(OmegaConf.select(cfg, "relations.endogenous_accept.enabled",
                                 default=False))
    tod = str(OmegaConf.select(cfg, "run.start_tod", default="07:00") or "07:00")
    try:
        hh, mm = tod.split(":")
        start_min = int(hh) * 60 + int(mm)
    except Exception:
        start_min = 420
    return {"k": _k_label(wb, alpha), "k_eff": _k_eff(wb, alpha),
            "endo": "on" if endo else "off",
            "seed": int(OmegaConf.select(cfg, "run.seed", default=0)),
            "n_agents": int(OmegaConf.select(cfg, "run.n_agents", default=0)),
            "start_min": start_min}


# --------------------------------------------------------------------------- #
# per-run 指標(analyze_structure の関数を再利用=in-sim L2 と同一定義)
# --------------------------------------------------------------------------- #
def _mean_skip(series: list, skip_days: int):
    vals = [v for v in series[skip_days:] if v is not None]
    return round(float(sum(vals)) / len(vals), 6) if vals else None


def _l2_kpi_daily(run_dir: str, start_min: int, n_days: int,
                  step_min: int = STEP_MIN) -> dict[str, list]:
    """L2 の joint_* 4列を「暦日の最終行」で日次化(当日タリーの終値=in-sim 定義と一致)。
    列が無い(endo OFF / joint OFF)ランは空 dict。"""
    l2 = m.load_l2(run_dir)
    if not l2 or not l2.get("step"):
        return {}
    cols = [c for c in KPI_COLS if c in l2]
    if not cols:
        return {}
    last_idx: dict[int, int] = {}
    for i, s in enumerate(l2["step"]):
        d = (start_min + int(s) * int(step_min)) // 1440
        prev = last_idx.get(d)
        if prev is None or int(s) >= int(l2["step"][prev]):
            last_idx[d] = i
    out = {}
    for c in cols:
        out[c] = [(round(float(l2[c][last_idx[d]]), 6) if d in last_idx else None)
                  for d in range(n_days)]
    return out


def run_metrics(run_dir: str, skip_days: int = SKIP_DAYS,
                top_k: int = _ast.TOP_K) -> dict:
    """1 ラン分: タスクB日次系列 + per-run スカラー + フェーズ1 KPI(存在時)。"""
    events = m.load_events(run_dir)
    agents = m.load_agents(run_dir)
    n_agents = len(agents) or (max((e["agent_id"] for e in events
                                    if e["agent_id"] >= 0), default=-1) + 1)
    # W2-3: 「1 日」はこのランの run.dt_min で決まる(Δt=10 なら 144 = 従来と同値)。
    spd = run_dt.steps_per_day(run_dir)
    by_day, days = _ast.bucket_by_day(events, spd)
    churn = _ast.reconstruct_churn(by_day, days)
    rank = _ast.rank_stability(run_dir, by_day, days, top_k, spd)
    cent = _ast.centrality_churn(by_day, days, top_k)
    comm = _ast.community_change(by_day, days, n_agents)
    stag = _ast.detect_stagnation(days, churn, rank, cent, _ast.MIN_DAYS,
                                  _ast.CENT_CHURN_LOW, _ast.EDGE_CHURN_LOW,
                                  _ast.TAU_HIGH)
    series = {
        "edge_churn_rate": churn["churn_rate"],
        "community_change_rate": comm["change_rate"],
        "centrality_turnover": cent["turnover"],
        "rank_tau_prev_day": rank["tau_prev_day"],
    }
    scalars = {k: _mean_skip(v, skip_days) for k, v in series.items()}
    scalars["stagnant_days"] = int(stag["total_stagnant_days"])
    return {"n_days": len(days), "series": series, "scalars": scalars,
            "stagnation": {"total_stagnant_days": stag["total_stagnant_days"],
                           "n_intervals": len(stag["combined"])}}


# --------------------------------------------------------------------------- #
# sign-flip permutation test(one-sample・両側・決定論)
# --------------------------------------------------------------------------- #
def sign_flip_test(diffs, n_mc: int = MC_ITER, rng_seed: int = 0,
                   exhaust_max: int = EXHAUST_MAX) -> dict:
    """seedペア差の sign-flip permutation(統計量=平均・両側 |mean|)。

    n ≤ exhaust_max: 全 2^n 符号列挙(観測を含む厳密分布。p ≥ 1/2^n)。
    n > exhaust_max: モンテカルロ n_mc 回・p=(b+1)/(n_mc+1)(Phipson & Smyth 2010
    =p が 0 にならない不偏側の推定)。乱数は default_rng(rng_seed) 固定=決定論。
    返り値の min_p は「全 seed 符号一致時に達しうる理論下限」(両側=2/2^n)=
    小 n の検出力限界を出力へ正直に載せるための値。"""
    ds = [float(d) for d in diffs if d is not None]
    n = len(ds)
    n_pos = sum(1 for d in ds if d > 0)
    n_neg = sum(1 for d in ds if d < 0)
    base = {"n": n, "n_pos": n_pos, "n_neg": n_neg,
            "mean_diff": round(float(np.mean(ds)), 6) if n else None}
    if n == 0:
        return {**base, "p": None, "method": "none", "min_p": None}
    obs = abs(float(np.mean(ds)))
    if all(abs(d) < 1e-15 for d in ds):
        return {**base, "p": 1.0, "method": "degenerate_zero", "min_p": 1.0}
    if n <= exhaust_max:
        total = 1 << n
        count = 0
        for mask in range(total):
            s = 0.0
            for i, d in enumerate(ds):
                s += -d if (mask >> i) & 1 else d
            if abs(s / n) >= obs - 1e-12:
                count += 1
        return {**base, "p": round(count / total, 6), "method": "exhaustive",
                "n_perm": total, "min_p": round(2.0 / total, 6)}
    rng = np.random.default_rng(rng_seed)
    arr = np.asarray(ds)
    signs = rng.integers(0, 2, size=(n_mc, n)) * 2 - 1
    perm = np.abs((signs * arr).mean(axis=1))
    b = int(np.sum(perm >= obs - 1e-12))
    return {**base, "p": round((b + 1) / (n_mc + 1), 6), "method": "monte_carlo",
            "n_perm": n_mc, "min_p": round(1.0 / (n_mc + 1), 6)}


def block_signflip_test(series_list: list[list], block_len: int = 0,
                        n_mc: int = MC_ITER, rng_seed: int = 0) -> dict:
    """副検定(感度分析): 日次ペア差系列のブロック符号反転(時間自己相関を保存)。

    series_list = seed ごとの日次差 [d_0..d_{T-1}](None は欠測=平均から除外)。
    連続する block_len 日を1単位として符号反転する(ブロック内の時間依存を壊さない=
    Künsch 1989 moving block bootstrap の趣旨)。block_len=0 は max(2, round(T^(1/3)))。
    統計量=全 seed × 全日次差の平均(両側)。決定論(default_rng(rng_seed))。"""
    clean = [[(float(v) if v is not None else None) for v in s]
             for s in series_list if s]
    flat = [v for s in clean for v in s if v is not None]
    if not flat:
        return {"p": None, "method": "none", "n_obs": 0, "block_len": None}
    T = max(len(s) for s in clean)
    L = block_len if block_len >= 1 else max(2, int(round(T ** (1.0 / 3.0))))
    obs = abs(float(np.mean(flat)))
    if all(abs(v) < 1e-15 for v in flat):
        return {"p": 1.0, "method": "degenerate_zero", "n_obs": len(flat),
                "block_len": L, "mean_diff": 0.0}
    rng = np.random.default_rng(rng_seed)
    n_blocks = [(len(s) + L - 1) // L for s in clean]
    b_count = 0
    for _ in range(n_mc):
        tot, cnt = 0.0, 0
        for s, nb in zip(clean, n_blocks):
            signs = rng.integers(0, 2, size=nb) * 2 - 1
            for i, v in enumerate(s):
                if v is None:
                    continue
                tot += v * signs[i // L]
                cnt += 1
        if cnt and abs(tot / cnt) >= obs - 1e-12:
            b_count += 1
    return {"p": round((b_count + 1) / (n_mc + 1), 6), "method": "block_mc",
            "n_obs": len(flat), "block_len": L, "n_perm": n_mc,
            "mean_diff": round(float(np.mean(flat)), 6)}


# --------------------------------------------------------------------------- #
# ペア組み立て・交互作用・機械判定
# --------------------------------------------------------------------------- #
def build_pairs(runs: dict) -> dict:
    """{k: {metric: {seed: {off, on, diff}}}}(diff=ON−OFF。両側が揃う seed のみ)。"""
    cell: dict[tuple, dict] = {}
    for name, r in runs.items():
        cell[(r["k"], r["endo"], r["seed"])] = r
    pairs: dict = {}
    ks = sorted({r["k"] for r in runs.values()})
    for k in ks:
        seeds = sorted({r["seed"] for r in runs.values() if r["k"] == k})
        for metric in STRUCT_METRICS:
            for seed in seeds:
                off = cell.get((k, "off", seed))
                on = cell.get((k, "on", seed))
                if off is None or on is None:
                    continue
                vo = off["scalars"].get(metric)
                vn = on["scalars"].get(metric)
                d = (round(vn - vo, 6) if vo is not None and vn is not None
                     else None)
                pairs.setdefault(k, {}).setdefault(metric, {})[str(seed)] = {
                    "off": vo, "on": vn, "diff": d}
    return pairs


def build_pair_series(runs: dict, metric: str, k: str) -> list[list]:
    """同一 seed ペアの日次差系列(ON−OFF。日 t の両側が非 None のときだけ差)。"""
    cell = {(r["k"], r["endo"], r["seed"]): r for r in runs.values()}
    out = []
    for seed in sorted({r["seed"] for r in runs.values() if r["k"] == k}):
        off = cell.get((k, "off", seed))
        on = cell.get((k, "on", seed))
        if off is None or on is None:
            continue
        so, sn = off["series"][metric], on["series"][metric]
        T = min(len(so), len(sn))
        out.append([(round(sn[t] - so[t], 6)
                     if so[t] is not None and sn[t] is not None else None)
                    for t in range(T)])
    return out


def interaction_diffs(pairs: dict, metric: str, k_hi: str, k_lo: str) -> list:
    """差の差(H3): 各 seed の diff(k_hi) − diff(k_lo)(両方が揃う seed のみ)。"""
    hi = pairs.get(k_hi, {}).get(metric, {})
    lo = pairs.get(k_lo, {}).get(metric, {})
    out = []
    for seed in sorted(set(hi) & set(lo), key=int):
        dh, dl = hi[seed]["diff"], lo[seed]["diff"]
        if dh is not None and dl is not None:
            out.append(round(dh - dl, 6))
    return out


def _consistency(diffs: list, expect_positive: bool = True,
                 min_agree: int = MIN_AGREE) -> dict:
    ds = [d for d in diffs if d is not None]
    n_pos = sum(1 for d in ds if d > 0)
    n_neg = sum(1 for d in ds if d < 0)
    agree = n_pos if expect_positive else n_neg
    other = n_neg if expect_positive else n_pos
    return {"n": len(ds), "n_pos": n_pos, "n_neg": n_neg,
            "consistent": bool(agree >= min_agree and agree > other),
            "enough_seeds": bool(len(ds) >= min_agree)}


def judge(pairs: dict, tests: dict, kpi: dict, k_order: list[str],
          min_agree: int = MIN_AGREE, gate_pp: float = GATE_PP) -> dict:
    """計画書§3 の合否基準を機械判定する(有意性でなく符号一致+効果量が主)。"""
    # H1: ON で edge churn↑(各 k 水準のペア差が ≥min_agree seed で正に符号一致)
    h1_by_k = {}
    for k in k_order:
        diffs = [v["diff"] for v in pairs.get(k, {})
                 .get(PRIMARY_METRIC, {}).values()]
        h1_by_k[k] = _consistency(diffs, expect_positive=True,
                                  min_agree=min_agree)
    h1_pass = any(c["consistent"] for c in h1_by_k.values())
    # H3: 交互作用(free−off 本命)の差の差が正に符号一致
    h3 = {}
    inter = tests.get("interaction", {})
    for contrast, per_metric in inter.items():
        t = per_metric.get(PRIMARY_METRIC)
        if t is None:
            continue
        ds = t.get("_diffs", [])
        h3[contrast] = _consistency(ds, expect_positive=True,
                                    min_agree=min_agree)
    primary_contrast = next((c for c in ("free-off",) if c in h3),
                            (sorted(h3)[0] if h3 else None))
    h3_pass = bool(primary_contrast and h3[primary_contrast]["consistent"])
    # 承諾率乖離ゲート(ON セルの joint_accept_calib_gap が全 k で ±gate_pp 以内)
    gate = {}
    for k in k_order:
        g = kpi.get(k, {}).get("joint_accept_calib_gap")
        gate[k] = {"gap": g,
                   "within": (g is not None and abs(g) <= gate_pp)}
    gate_ok = bool(gate) and all(v["within"] for v in gate.values())
    # H2: 固着帯の併発(H1 が立つ k で ON の固着日数 > 0=churn↑と固着の両立)
    h2_by_k = {}
    for k in k_order:
        st = pairs.get(k, {}).get("stagnant_days", {})
        on_stag = [v["on"] for v in st.values() if v["on"] is not None]
        h2_by_k[k] = {
            "on_stagnant_mean": (round(float(np.mean(on_stag)), 6)
                                 if on_stag else None),
            "diffs": {s: v["diff"] for s, v in sorted(st.items(), key=lambda x: int(x[0]))},
            "co_occurrence": bool(h1_by_k.get(k, {}).get("consistent")
                                  and on_stag and float(np.mean(on_stag)) > 0),
        }
    h2_pass = any(v["co_occurrence"] for v in h2_by_k.values())
    n_seeds = max((c["n"] for c in h1_by_k.values()), default=0)
    return {
        "primary_metric": PRIMARY_METRIC,
        "H1": {"by_k": h1_by_k, "pass": h1_pass,
               "hypothesis": "ON で edge churn↑(主体選択が構造を動かす)"},
        "H2": {"by_k": h2_by_k, "pass": h2_pass,
               "hypothesis": "churn↑ と同時にコミュニティ固着帯も発生(H1と両立)"},
        "H3": {"by_contrast": h3, "primary_contrast": primary_contrast,
               "pass": h3_pass,
               "hypothesis": "k×ON 交互作用(k↑で ON のみ構造変動が増える=本命)"},
        "accept_gap_gate": {"by_k": gate, "pass": gate_ok,
                            "threshold_pp": gate_pp * 100},
        "phase3_go": bool((h1_pass or h3_pass) and gate_ok),
        "power_note": (
            f"n={n_seeds} seedペアの sign-flip は両側pの理論下限が 2/2^n"
            f"(n=3:0.25 / n=5:0.0625)で α=0.05 に届かない。"
            "判定は有意性でなく符号一致(≥3 seed)+効果量の報告を主とする"
            "(計画書§3 の合否基準)。"),
    }


# --------------------------------------------------------------------------- #
# 出力(parquet + markdown)
# --------------------------------------------------------------------------- #
def _write_parquets(out_dir: str, pairs: dict, tests: dict) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows_k, rows_m, rows_s = [], [], []
    rows_off, rows_on, rows_d = [], [], []
    for k in sorted(pairs):
        for metric in sorted(pairs[k]):
            for seed in sorted(pairs[k][metric], key=int):
                v = pairs[k][metric][seed]
                rows_k.append(k)
                rows_m.append(metric)
                rows_s.append(int(seed))
                rows_off.append(v["off"])
                rows_on.append(v["on"])
                rows_d.append(v["diff"])
    pq.write_table(pa.table({
        "k": pa.array(rows_k, pa.string()),
        "metric": pa.array(rows_m, pa.string()),
        "seed": pa.array(rows_s, pa.int32()),
        "off": pa.array([float(x) if x is not None else None for x in rows_off],
                        pa.float64()),
        "on": pa.array([float(x) if x is not None else None for x in rows_on],
                       pa.float64()),
        "diff": pa.array([float(x) if x is not None else None for x in rows_d],
                         pa.float64()),
    }), os.path.join(out_dir, "endo_pairs.parquet"), compression="zstd")

    trows = {"kind": [], "scope": [], "metric": [], "n": [], "mean_diff": [],
             "p": [], "method": [], "n_pos": [], "n_neg": [], "block_len": []}
    for kind in ("pair", "interaction", "block"):
        for scope in sorted(tests.get(kind, {})):
            for metric in sorted(tests[kind][scope]):
                t = tests[kind][scope][metric]
                trows["kind"].append(kind)
                trows["scope"].append(scope)
                trows["metric"].append(metric)
                trows["n"].append(int(t.get("n", t.get("n_obs", 0)) or 0))
                trows["mean_diff"].append(
                    float(t["mean_diff"]) if t.get("mean_diff") is not None
                    else None)
                trows["p"].append(float(t["p"]) if t.get("p") is not None
                                  else None)
                trows["method"].append(str(t.get("method", "")))
                trows["n_pos"].append(int(t.get("n_pos", 0) or 0))
                trows["n_neg"].append(int(t.get("n_neg", 0) or 0))
                trows["block_len"].append(int(t["block_len"])
                                          if t.get("block_len") else None)
    pq.write_table(pa.table({
        "kind": pa.array(trows["kind"], pa.string()),
        "scope": pa.array(trows["scope"], pa.string()),
        "metric": pa.array(trows["metric"], pa.string()),
        "n": pa.array(trows["n"], pa.int32()),
        "mean_diff": pa.array(trows["mean_diff"], pa.float64()),
        "p": pa.array(trows["p"], pa.float64()),
        "method": pa.array(trows["method"], pa.string()),
        "n_pos": pa.array(trows["n_pos"], pa.int32()),
        "n_neg": pa.array(trows["n_neg"], pa.int32()),
        "block_len": pa.array(trows["block_len"], pa.int32()),
    }), os.path.join(out_dir, "endo_tests.parquet"), compression="zstd")


def _f(v, nd=4):
    if v is None:
        return "—"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def write_report(path: str, result: dict) -> None:
    pairs, tests, kpi = result["pairs"], result["tests"], result["kpi"]
    jd = result["judgment"]
    ks = result["k_order"]
    L = [f"# endogenous_accept treatment 比較レポート({result['experiment']})\n"]
    L.append(f"- ラン数: {len(result['runs'])} / k 水準: {ks} / "
             f"seedペア(CRN): 最大 {max((t.get('n', 0) for pm in tests['pair'].values() for t in pm.values()), default=0)}")
    L.append(f"- 主指標: **{jd['primary_metric']}**(H1/H3)。"
             "検定=seedペア差 sign-flip(主)+ 日次ブロック符号反転(副=感度分析)。")
    L.append(f"- ⚠ {jd['power_note']}\n")

    L.append("## ペア差(ON−OFF・CRN 同一seed)\n")
    for metric in STRUCT_METRICS:
        L.append(f"### {metric}\n")
        L.append("| k | seed差(ON−OFF) | 平均差 | 符号(+/−) | sign-flip p(両側) | "
                 "p理論下限 | 副検定 block p |")
        L.append("|---|---|---|---|---|---|---|")
        for k in ks:
            t = tests["pair"].get(k, {}).get(metric, {})
            b = tests["block"].get(k, {}).get(metric, {})
            per = pairs.get(k, {}).get(metric, {})
            cell = ", ".join(f"s{s}:{_f(per[s]['diff'])}"
                             for s in sorted(per, key=int))
            L.append(f"| {k} | {cell} | {_f(t.get('mean_diff'))} | "
                     f"{t.get('n_pos', 0)}/{t.get('n_neg', 0)} | "
                     f"{_f(t.get('p'))} ({t.get('method', '—')}) | "
                     f"{_f(t.get('min_p'))} | {_f(b.get('p'))} |")
        L.append("")

    L.append("## 交互作用(H3: 差の差 = diff(k_hi) − diff(k_lo))\n")
    L.append("| 対比 | metric | 差の差(seed別) | 平均 | 符号(+/−) | sign-flip p |")
    L.append("|---|---|---|---|---|---|")
    for contrast in sorted(tests.get("interaction", {})):
        for metric in STRUCT_METRICS:
            t = tests["interaction"][contrast].get(metric)
            if not t:
                continue
            ds = ", ".join(_f(d) for d in t.get("_diffs", []))
            L.append(f"| {contrast} | {metric} | {ds} | {_f(t.get('mean_diff'))} | "
                     f"{t.get('n_pos', 0)}/{t.get('n_neg', 0)} | {_f(t.get('p'))} |")
    L.append("")

    L.append("## フェーズ1 KPI(ON セルのみ=OFF は joint_invite を記録しない)\n")
    L.append("フェーズ3 invite 2列(第64バッチ)は endogenous_invite ON のランのみ値が入る"
             "(OFF は列なし=—)。")
    L.append("| k | accept_rate | endo_share | calib_gap(±0.15 ゲート) | fulfill_rate | "
             "invite_weak_tie | invite_endo |")
    L.append("|---|---|---|---|---|---|---|")
    for k in ks:
        kv = kpi.get(k, {})
        g = jd["accept_gap_gate"]["by_k"].get(k, {})
        mark = "✓" if g.get("within") else "✗"
        L.append(f"| {k} | {_f(kv.get('joint_accept_rate'))} | "
                 f"{_f(kv.get('joint_endo_share'))} | "
                 f"{_f(kv.get('joint_accept_calib_gap'))} {mark} | "
                 f"{_f(kv.get('joint_fulfill_rate'))} | "
                 f"{_f(kv.get('invite_weak_tie_rate'))} | "
                 f"{_f(kv.get('invite_endo_share'))} |")
    L.append("")

    L.append("## 仮説の機械判定(計画書§3: 符号一致 ≥3seed + 乖離±15pp)\n")
    for h in ("H1", "H2", "H3"):
        d = jd[h]
        L.append(f"- **{h}**: {d['hypothesis']} → "
                 f"**{'成立' if d['pass'] else '不成立'}**")
    gg = jd["accept_gap_gate"]
    L.append(f"- **承諾率乖離ゲート**(±{gg['threshold_pp']:.0f}pp): "
             f"**{'内' if gg['pass'] else '外/データなし'}**")
    L.append(f"- **フェーズ3進出判定**: {'GO' if jd['phase3_go'] else 'NO-GO'}"
             "(H1 or H3 の符号一致 かつ 乖離ゲート内)\n")
    L.append("## 検定実装の出典")
    L.append("- sign-flip(n≤12 全列挙 / n>12 MC): evalci arXiv:2607.04429 §sign-flip・"
             "Phipson & Smyth 2010(doi:10.2202/1544-6115.1585)= MC p=(b+1)/(m+1)")
    L.append("- ネットワーク置換の過大有意への注意: Farine & Carter 2022 "
             "(doi:10.1111/2041-210X.13741)→ 本解析はラン(seedペア)単位の置換")
    L.append("- ブロック符号反転(副検定)のブロック化: Künsch 1989 moving block bootstrap の趣旨")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


# --------------------------------------------------------------------------- #
# 本体
# --------------------------------------------------------------------------- #
def analyze(pattern, out=None, skip_days: int = SKIP_DAYS, mc: int = MC_ITER,
            block_len: int = 0, min_agree: int = MIN_AGREE,
            allow_tier_mismatch: bool = False) -> dict:
    """実験ラン群を横断解析し、結果 dict を返す(out 指定時はファイルも書く)。"""
    if isinstance(pattern, (list, tuple)):
        candidates = sorted(str(p) for p in pattern)
    else:
        candidates = sorted(glob.glob(str(pattern)))
    dirs = [d for d in candidates
            if os.path.isfile(os.path.join(d, "config.yaml"))
            and os.path.isfile(os.path.join(d, "l1_events.parquet"))]
    if not dirs:
        raise SystemExit(f"no runs matched: {pattern}")
    # 第72バッチ: ラン間比較ガード。CRN ペア差の前提は「同じ装置で回した2本」なので、
    # run_mode / 再現性等級が混ざったセル群は既定で拒否する。
    from society.registry import guard_or_die
    guard_or_die(dirs, allow_mismatch=allow_tier_mismatch)

    runs: dict = {}
    n_days_max = 0
    for d in dirs:
        cell = read_cell(d)
        rm = run_metrics(d, skip_days=skip_days)
        n_days_max = max(n_days_max, rm["n_days"])
        kpi_daily = _l2_kpi_daily(d, cell["start_min"], rm["n_days"],
                                  run_dt.min_per_step(d))
        kpi_scalars = {c: _mean_skip(v, skip_days)
                       for c, v in kpi_daily.items()}
        name = os.path.basename(os.path.normpath(d))
        runs[name] = {**cell, "dir": d, **rm,
                      "kpi_series": kpi_daily, "kpi": kpi_scalars}

    k_order = [k for _e, k in sorted(
        {(r["k_eff"] if r["k_eff"] is not None else 99.0, r["k"])
         for r in runs.values()})]
    pairs = build_pairs(runs)

    tests: dict = {"pair": {}, "interaction": {}, "block": {}}
    for k in k_order:
        for metric in STRUCT_METRICS:
            per = pairs.get(k, {}).get(metric, {})
            diffs = [per[s]["diff"] for s in sorted(per, key=int)]
            tests["pair"].setdefault(k, {})[metric] = sign_flip_test(
                diffs, n_mc=mc)
        for metric in STRUCT_SERIES:
            tests["block"].setdefault(k, {})[metric] = block_signflip_test(
                build_pair_series(runs, metric, k), block_len=block_len,
                n_mc=mc)
    for i in range(len(k_order)):
        for j in range(i + 1, len(k_order)):
            k_lo, k_hi = k_order[i], k_order[j]
            contrast = f"{k_hi}-{k_lo}"
            for metric in STRUCT_METRICS:
                ds = interaction_diffs(pairs, metric, k_hi, k_lo)
                t = sign_flip_test(ds, n_mc=mc)
                t["_diffs"] = ds
                tests["interaction"].setdefault(contrast, {})[metric] = t

    # ON セル KPI(k 水準ごとに ON ラン平均)
    kpi: dict = {}
    for k in k_order:
        vals: dict[str, list] = defaultdict(list)
        for r in runs.values():
            if r["k"] == k and r["endo"] == "on":
                for c in KPI_COLS:
                    v = r["kpi"].get(c)
                    if v is not None:
                        vals[c].append(v)
        kpi[k] = {c: (round(float(np.mean(vs)), 6) if vs else None)
                  for c, vs in ((c, vals.get(c, [])) for c in KPI_COLS)}

    judgment = judge(pairs, tests, kpi, k_order, min_agree=min_agree)
    result = {
        "experiment": (pattern if isinstance(pattern, str)
                       else ";".join(str(p) for p in pattern)),
        "params": {"skip_days": skip_days, "mc": mc, "block_len": block_len,
                   "min_agree": min_agree, "gate_pp": GATE_PP,
                   "exhaust_max": EXHAUST_MAX, "primary_metric": PRIMARY_METRIC,
                   "struct_source": "analyze_structure.py(単一の源)"},
        "k_order": k_order,
        "runs": {name: {k2: v for k2, v in r.items() if k2 != "dir"}
                 for name, r in sorted(runs.items())},
        "pairs": pairs,
        "tests": tests,
        "kpi": kpi,
        "judgment": judgment,
    }

    if out:
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "endo_treatment.json"), "w",
                  encoding="utf-8") as fh:
            fh.write(json.dumps(result, sort_keys=True, ensure_ascii=False))
        _write_parquets(out, pairs, tests)
        write_report(os.path.join(out, "endo_treatment_report.md"), result)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(
        description="第63バッチ フェーズ2: endogenous_accept treatment 比較(CRN ペア差 + sign-flip)")
    ap.add_argument("pattern", help='glob。例 "runs/endo14_*"')
    ap.add_argument("--out", default=None,
                    help="出力ディレクトリ(既定 runs/_endo_treatment)")
    ap.add_argument("--skip-days", type=int, default=SKIP_DAYS,
                    help="per-run スカラーから除く冒頭日数(既定 1=day0 の初期化バースト除外)")
    ap.add_argument("--mc", type=int, default=MC_ITER,
                    help="モンテカルロ回数(n>12 のときのみ使用。既定 10000)")
    ap.add_argument("--block-len", type=int, default=0,
                    help="副検定のブロック長(0=auto max(2, round(T^(1/3))))")
    ap.add_argument("--min-agree", type=int, default=MIN_AGREE,
                    help="符号一致とみなす最少 seed 数(既定 3=計画書§3)")
    ap.add_argument("--allow-tier-mismatch", action="store_true",
                    help="run_mode / 再現性等級が異なるラン同士の比較を明示的に許可する"
                         "(既定は拒否)")
    args = ap.parse_args()
    out = args.out or os.path.join("runs", "_endo_treatment")
    result = analyze(args.pattern, out=out, skip_days=args.skip_days,
                     mc=args.mc, block_len=args.block_len,
                     min_agree=args.min_agree,
                     allow_tier_mismatch=args.allow_tier_mismatch)
    jd = result["judgment"]
    print(f"[endo] {len(result['runs'])} runs / k={result['k_order']} -> {out}")
    print(f"  H1={'成立' if jd['H1']['pass'] else '不成立'} "
          f"H2={'成立' if jd['H2']['pass'] else '不成立'} "
          f"H3={'成立' if jd['H3']['pass'] else '不成立'} "
          f"gate={'内' if jd['accept_gap_gate']['pass'] else '外'} "
          f"phase3={'GO' if jd['phase3_go'] else 'NO-GO'}")
    print(f"  ⚠ {jd['power_note']}")


if __name__ == "__main__":
    main()
