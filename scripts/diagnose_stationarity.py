#!/usr/bin/env python
"""D0 非定常性診断(後処理・L1 を読むだけ・シム本体に非依存)。

    python scripts/diagnose_stationarity.py runs/<name>
    python scripts/diagnose_stationarity.py runs/daily300_100d --pairs 2:5,2:20,2:99

設計正典: docs/plans/twin-physics-vision-affordance-plan.md §2 レーン1 D0 行 + 付録 D 節。
事前登録: docs/plans/stationarity-preregistration.md(閾値の定義・確定手続き)。

■ 問い(原指示 D)
  「25万体×10日が『饒舌な日常』の反復に収束するリスク」に対し、**Day t と Day t' が
  統計的に区別できるか**を本選前に測り、閾値を事前登録する。実査(計画書 §0-6)で反復の
  実体は EPR(訪問回数比例の優先的回帰・`routine.py:367`)+ 決定論日次予定と判明している。

■ 測るもの(すべて `<run>/l1_events.parquet` から。sim 本体 import ゼロ)
  1. 新規語彙出現率  … label_coin/vocab_coin(造語)・label_adopt(複雑接触での採用)・
                       vocab_use(使用)。日ごとの新規ユニーク語数/採用数/**新規性シェア**
                       (その日使われた語のうち初出の割合)/ 使用語彙エントロピー。
  2. 関係グラフ構造  … 接触グラフ(無向・speak(speaker→hearers)+dm(sender→to)=
                       `scripts/analyze_weak_ties.py:62 build_weighted_graph` と同一チャネル)の
                       日ごとの新規紐帯数 + 累積グラフの平均次数/クラスタ係数/直径。
                       加えて relation_tier(新規ペア初出)/ relation_break(断絶)の日次件数。
  3. 行動反復度      … arrive から「既訪問 POI への回帰率(=1−first_visit 率。EPR 回帰の直接量)」
                       「訪問先分布の日次エントロピー」「エージェントごとの訪問ノード集合の
                       前日との Jaccard 類似」。
  4. 日次自己相似性(**主判定**)
                     … エージェント×日の特徴ベクトル(イベント種別ヒストグラム + 訪問分布)を作り、
                       Day t と Day t' の**対応のある**差 d_i = v_i(t) − v_i(t') に対して
                       **符号反転置換検定**(sign-flip permutation)を行う。同一エージェントが
                       両日に現れるので日ラベルの交換 = 差の符号反転が交換可能単位になる。
                       統計量 = 効果量そのもの(TVD)。

■ 効果量(事前登録の主軸)
  TVD(total variation distance) = 0.5 · Σ_j |mean_i(v_ij(t)) − mean_i(v_ij(t'))|
  特徴ベクトルは各ブロック内で和 1 に正規化した確率ベクトルなので、TVD ∈ [0,1] は
  「平均的なエージェントの1日のプロフィールのうち何割が入れ替わったか」と読める。
  補助として次元ごとの対応のある Cohen's dz(mean/sd)の平均・最大も出す。

■ 検定(決定論)
  paired sign-flip permutation(両側・統計量 = TVD)。
    - n ≤ 12 は 2^n 全列挙(厳密)。n > 12 は MC 既定 10,000 回、p=(b+1)/(m+1)
      (Phipson & Smyth 2010 "Permutation P-values Should Never Be Zero",
       doi:10.2202/1544-6115.1585)。乱数は `numpy.random.default_rng(0)` 固定。
    - ネットワーク内のノード置換はしない(Farine & Carter 2022, doi:10.1111/2041-210X.13741 の
      過大有意の指摘を避けるため、交換単位は**エージェント**であってグラフの辺ではない)。
    - 実装は `scripts/analyze_endo_treatment.py:203 sign_flip_test` の作法をそのまま踏襲
      (importせずに本ファイル内で多変量版として再実装)。
  ★n が大きい(数百体)と、ごく小さな差でも p は容易に 0.05 を下回る。したがって
    合否は **p と効果量(TVD)の連言**で判定する(事前登録文書の判定式)。
  ★ノイズ床: 隣接日ペア(d, d+1)の TVD 中央値を「差が無いに等しい」基準として併記し、
    `tvd_ratio = TVD(t,t') / median(隣接日 TVD)` を報告する。

■ 読み込み戦略(大規模ラン対応・md にも明記)
  - `pyarrow.parquet.ParquetFile.iter_batches` による**ストリーム走査**。全件 RAM 展開はしない
    (`society.observer.measure.load_events` は 10^7 行で破綻するので使わない)。
  - **2 パス**: パス1は列 [sim_min, agent_id, kind] のみ(payload を触らない=最安)で
    kind 語彙・エージェント id 集合・日数を確定。パス2で [sim_min, agent_id, kind, payload] を
    読み、**payload の JSON パースは必要な kind の行だけ**(pyarrow の `is_in` マスク →
    `filter` → その行だけ `to_pylist`)。daily300_100d では 11.9M 行中 1.27M 行のみパース。
  - 特徴量は密 numpy 配列 (agent, day, dim) に `np.bincount` で直接畳み込む(Python ループ無し)。
    セル数が上限を超える場合は kind 上位K・エージェント等間隔サブサンプルへ自動縮退し、
    その旨を出力に記録する(黙って落とさない)。

出力:
  <run>/diagnose_stationarity.json  … 機械可読(sort_keys=True でバイト同一)
  <run>/diagnose_stationarity.md    … 人間向けの表

依存: pyarrow / numpy のみ(pandas・duckdb 禁止=リポジトリ不変則)。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict

try:  # Windows cp932 対策
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# --------------------------------------------------------------------------- #
# 定数(決定論のためスクリプト定数。CLI で上書き可)
# --------------------------------------------------------------------------- #
SIM_MIN_PER_DAY = 1440          # 1 sim日 = 1440 sim分(= 144 step × 10 分)
BATCH_ROWS = 262_144            # iter_batches のバッチ行数
MC_ITER = 10_000                # 置換検定のモンテカルロ反復
EXHAUST_MAX = 12                # n ≤ これ なら 2^n 全列挙(analyze_endo_treatment と同値)
RNG_SEED = 0                    # numpy.random.default_rng(0) 固定
MAX_FEATURE_KINDS = 96          # 特徴ベクトルに使う kind の上限(超過分は "_other" へ集約)
MAX_FEATURE_AGENTS = 20_000     # 特徴ベクトルに使うエージェント数の上限(等間隔サブサンプル)
TOP_VISIT_NODES = 64            # 訪問分布の次元(上位Nノード + "_other")
TOPO_MAX_NODES = 1_500          # これ以下なら numpy 厳密トポロジ。超えたら近似(BFS 始点サンプル)
TOPO_BFS_SOURCES = 64           # 近似時の BFS 始点数
MAX_EDGES = 5_000_000           # 接触グラフの辺数上限(超過は打ち切り+フラグ)
PERM_CHUNK_BYTES = 64 << 20     # 置換行列のチャンク上限(メモリ保護)

# payload の JSON パースが必要な kind(これ以外は payload に触らない)
COIN_KINDS = ("label_coin", "vocab_coin")
ADOPT_KINDS = ("label_adopt",)
USE_KINDS = ("vocab_use",)
REL_KINDS = ("relation_tier", "relation_break")
ARRIVE_KIND = "arrive"
CONTACT_KINDS = ("speak", "dm")
PAYLOAD_KINDS = tuple(sorted(set(COIN_KINDS + ADOPT_KINDS + USE_KINDS + REL_KINDS
                                 + CONTACT_KINDS + (ARRIVE_KIND,))))

# ---- 事前登録(ドラフト)の閾値。正典は docs/plans/stationarity-preregistration.md ----
# ★ステータス: ドラフト。ユーザー承認(U-10)+ 実 LLM 診断ランを経て確定する。
DRAFT_THRESHOLDS = {
    "p_max": 0.05,                 # 主判定 Day2 vs Day5 の置換検定 p
    "tvd_min": 0.05,               # 効果量(combined ブロックの TVD)の下限
    "tvd_ratio_min": 1.50,         # TVD(2,5) / median(隣接日 TVD)
    "vocab_novelty_share_min": 0.05,   # 後半日の「その日使われた語のうち初出」割合
    "new_edge_rate_min": 0.02,     # 後半日の新規紐帯率(新規辺 / 累積辺)
    "revisit_rate_max": 0.90,      # 後半日の既訪問 POI 回帰率(これ以上は反復とみなす)
    "late_window_frac": 0.5,       # 「後半日」= 全日数の後ろ半分
}


# --------------------------------------------------------------------------- #
# 小道具
# --------------------------------------------------------------------------- #
def _entropy_bits(counts) -> float:
    """Shannon エントロピー(bit)。counts は非負の反復可能。"""
    tot = 0.0
    vals = []
    for c in counts:
        c = float(c)
        if c > 0:
            vals.append(c)
            tot += c
    if tot <= 0.0:
        return 0.0
    h = 0.0
    for c in vals:
        p = c / tot
        h -= p * math.log2(p)
    return h


def _r(x, nd=6):
    """None 安全な round(JSON をバイト安定にする)。"""
    if x is None:
        return None
    if isinstance(x, (bool, int)):
        return x
    v = float(x)
    if not math.isfinite(v):
        return None
    return round(v, nd)


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = len(a | b)
    return (len(a & b) / u) if u else 1.0


# --------------------------------------------------------------------------- #
# グラフ位相: クラスタ係数・直径
#   純 Python 版は scripts/analyze_weak_ties.py:85 clustering_and_diameter を**コピー**
#   (既存ファイルは読むだけ・変更禁止のため import せず複製。出典: 第64バッチ フェーズ3)。
#   numpy 版は同じ定義の高速路(N ≤ TOPO_MAX_NODES)。両者の一致は単体テストで固定する。
# --------------------------------------------------------------------------- #
def clustering_and_diameter_py(graph: dict) -> dict:
    """[COPY] scripts/analyze_weak_ties.py:85 clustering_and_diameter(第64バッチ)。

    無向・非重みの平均局所クラスタ係数(Watts-Strogatz 1998)と最大連結成分の直径。
    - clustering_coeff: 次数≥2 のノードの局所係数(隣接間の実辺/可能辺)の平均。母数 0 なら 0.0。
    - diameter_lcc: 最大連結成分(タイは最小ノード id)内の全点対 BFS 最遠距離。
    全て決定論(ノードは id 昇順・純 Python・乱数なし)。"""
    adj: dict = defaultdict(set)
    for (u, v) in graph:
        adj[u].add(v)
        adj[v].add(u)
    nodes = sorted(adj)
    if not nodes:
        return {"clustering_coeff": 0.0, "diameter_lcc": 0, "lcc_size": 0,
                "clustering_n_nodes": 0, "diameter_exact": True}
    coeffs = []
    for n in nodes:
        nb = sorted(adj[n])
        k = len(nb)
        if k < 2:
            continue
        links = sum(1 for i in range(k) for j in range(i + 1, k)
                    if nb[j] in adj[nb[i]])
        coeffs.append(2.0 * links / (k * (k - 1)))
    cc = round(sum(coeffs) / len(coeffs), 6) if coeffs else 0.0
    seen: set = set()
    best: list = []
    for start in nodes:
        if start in seen:
            continue
        comp = [start]
        seen.add(start)
        i = 0
        while i < len(comp):
            for w in sorted(adj[comp[i]]):
                if w not in seen:
                    seen.add(w)
                    comp.append(w)
            i += 1
        if len(comp) > len(best) or (len(comp) == len(best)
                                     and comp and best and min(comp) < min(best)):
            best = comp
    lcc = set(best)
    # ---- ここだけ原本から変更: 大規模時は BFS 始点をサンプルして下界を返す(exact フラグ付き) ----
    srcs = sorted(lcc)
    exact = True
    if len(srcs) > TOPO_BFS_SOURCES:
        step = max(1, len(srcs) // TOPO_BFS_SOURCES)
        srcs = srcs[::step][:TOPO_BFS_SOURCES]
        exact = False
    diam = 0
    for src in srcs:
        dist = {src: 0}
        q = [src]
        i = 0
        while i < len(q):
            u = q[i]
            i += 1
            for w in sorted(adj[u]):
                if w in lcc and w not in dist:
                    dist[w] = dist[u] + 1
                    q.append(w)
        if dist:
            diam = max(diam, max(dist.values()))
    return {"clustering_coeff": cc, "diameter_lcc": int(diam),
            "lcc_size": len(best), "clustering_n_nodes": len(coeffs),
            "diameter_exact": exact}


def clustering_and_diameter_np(adj_bool: np.ndarray) -> dict:
    """numpy 厳密版(定義は clustering_and_diameter_py と同一)。adj_bool = 対称・対角 False。"""
    n = int(adj_bool.shape[0])
    if n == 0 or not adj_bool.any():
        return {"clustering_coeff": 0.0, "diameter_lcc": 0, "lcc_size": 0,
                "clustering_n_nodes": 0, "diameter_exact": True}
    A = adj_bool.astype(np.float32)
    deg = A.sum(axis=1)
    A2 = A @ A
    tri = (A2 * A).sum(axis=1)               # = 2 × (隣接間の実辺数)
    mask = deg >= 2
    with np.errstate(invalid="ignore", divide="ignore"):
        c = tri / (deg * (deg - 1.0))
    cc = round(float(c[mask].mean()), 6) if int(mask.sum()) else 0.0
    # ---- 全点対 BFS(レベル同期・行列積)----
    visited = np.eye(n, dtype=bool)
    frontier = visited.copy()
    dist = np.full((n, n), -1, dtype=np.int32)
    np.fill_diagonal(dist, 0)
    d = 0
    while frontier.any():
        d += 1
        nxt = (frontier.astype(np.float32) @ A) > 0
        nxt &= ~visited
        if not nxt.any():
            break
        dist[nxt] = d
        visited |= nxt
        frontier = nxt
    # 連結成分 = visited の行(到達集合)。最大成分(タイは最小 id)
    comp_size = visited.sum(axis=1)
    best_size = int(comp_size.max())
    best_row = int(np.argmax(comp_size))      # argmax はタイで最小 index を返す = 決定論
    lcc = visited[best_row]
    sub = dist[np.ix_(lcc, lcc)]
    diam = int(sub.max()) if sub.size else 0
    return {"clustering_coeff": cc, "diameter_lcc": diam,
            "lcc_size": best_size, "clustering_n_nodes": int(mask.sum()),
            "diameter_exact": True}


# --------------------------------------------------------------------------- #
# 置換検定(多変量・対応あり・符号反転)
#   作法は scripts/analyze_endo_treatment.py:203 sign_flip_test を踏襲(n≤12 全列挙 /
#   超は MC・p=(b+1)/(m+1)・default_rng(seed) 固定)。統計量を TVD に置き換えた多変量版。
# --------------------------------------------------------------------------- #
def paired_signflip_tvd(D: np.ndarray, n_mc: int = MC_ITER, rng_seed: int = RNG_SEED,
                        exhaust_max: int = EXHAUST_MAX) -> dict:
    """対応のある差行列 D (n, dims) に対する符号反転置換検定。

    統計量 = TVD = 0.5 · Σ_j |mean_i D_ij|(= 効果量そのもの)。両側。
    帰無仮説「各エージェントについて日ラベルは情報を持たない」の下で d_i → −d_i は交換可能。"""
    D = np.asarray(D, dtype=np.float64)
    if D.ndim != 2 or D.shape[0] == 0:
        return {"n": 0, "tvd": None, "p": None, "method": "none", "min_p": None}
    n = int(D.shape[0])
    obs = 0.5 * float(np.abs(D.mean(axis=0)).sum())
    base = {"n": n, "tvd": _r(obs)}
    if float(np.abs(D).max()) < 1e-15:
        return {**base, "p": 1.0, "method": "degenerate_zero", "min_p": 1.0, "n_perm": 0}
    if n <= exhaust_max:
        total = 1 << n
        count = 0
        for mask in range(total):
            s = np.array([-1.0 if (mask >> i) & 1 else 1.0 for i in range(n)])
            stat = 0.5 * float(np.abs((s[:, None] * D).mean(axis=0)).sum())
            if stat >= obs - 1e-12:
                count += 1
        return {**base, "p": round(count / total, 6), "method": "exhaustive",
                "n_perm": total, "min_p": round(2.0 / total, 6)}
    rng = np.random.default_rng(rng_seed)
    chunk = max(1, min(n_mc, int(PERM_CHUNK_BYTES // max(1, n * 8))))
    b = 0
    done = 0
    while done < n_mc:
        m = min(chunk, n_mc - done)
        signs = (rng.integers(0, 2, size=(m, n)) * 2 - 1).astype(np.float64)
        stats = 0.5 * np.abs((signs @ D) / n).sum(axis=1)
        b += int(np.sum(stats >= obs - 1e-12))
        done += m
    return {**base, "p": round((b + 1) / (n_mc + 1), 6), "method": "monte_carlo",
            "n_perm": n_mc, "min_p": round(1.0 / (n_mc + 1), 6)}


def paired_dz(D: np.ndarray) -> dict:
    """次元ごとの対応のある Cohen's dz = mean/sd(ddof=1)。sd=0 の次元は除外。"""
    D = np.asarray(D, dtype=np.float64)
    if D.shape[0] < 2:
        return {"dz_mean_abs": None, "dz_max_abs": None, "n_dims_used": 0}
    mu = D.mean(axis=0)
    sd = D.std(axis=0, ddof=1)
    ok = sd > 1e-12
    if not bool(ok.any()):
        return {"dz_mean_abs": None, "dz_max_abs": None, "n_dims_used": 0}
    dz = np.abs(mu[ok] / sd[ok])
    return {"dz_mean_abs": _r(float(dz.mean())), "dz_max_abs": _r(float(dz.max())),
            "n_dims_used": int(ok.sum())}


# --------------------------------------------------------------------------- #
# パス1: 列 [sim_min, agent_id, kind] のみ(payload に触らない=最安)
# --------------------------------------------------------------------------- #
def _scan_pass1(path: str, batch_rows: int) -> dict:
    pf = pq.ParquetFile(path)
    kind_counts: Counter = Counter()
    agent_ids: set = set()
    max_sim_min = -1
    min_sim_min = None
    n_rows = 0
    monotonic = True
    prev_last = -1
    for b in pf.iter_batches(batch_size=batch_rows, columns=["sim_min", "agent_id", "kind"]):
        n_rows += b.num_rows
        if b.num_rows == 0:
            continue
        sm = b.column("sim_min").to_numpy(zero_copy_only=False)
        if sm.size:
            if int(sm[0]) < prev_last or bool(np.any(np.diff(sm) < 0)):
                monotonic = False
            prev_last = int(sm[-1])
            mx = int(sm.max())
            mn = int(sm.min())
            max_sim_min = max(max_sim_min, mx)
            min_sim_min = mn if min_sim_min is None else min(min_sim_min, mn)
        aid = b.column("agent_id").to_numpy(zero_copy_only=False)
        u = np.unique(aid[aid >= 0])
        agent_ids.update(int(v) for v in u)
        vc = pc.value_counts(b.column("kind"))
        for row in vc.to_pylist():
            kind_counts[row["values"]] += int(row["counts"])
    return {"kind_counts": kind_counts, "agent_ids": sorted(agent_ids),
            "max_sim_min": max_sim_min, "min_sim_min": min_sim_min or 0,
            "n_rows": n_rows, "sim_min_monotonic": monotonic}


# --------------------------------------------------------------------------- #
# パス2: 特徴量+ドメイン系列の本走査
# --------------------------------------------------------------------------- #
class Accum:
    """1 パスで全指標の素材を貯める容器(密 numpy + 小さな dict)。"""

    def __init__(self, n_days: int, all_ids: list, kinds: list, kind_lut: np.ndarray,
                 sel_ids: list, max_edges: int, full_kinds: list):
        self.n_days = n_days
        self.kinds = kinds                      # 特徴 kind ラベル(末尾 "_other" を含みうる)
        self.n_kinds = len(kinds)
        self.kind_lut = kind_lut                # 全kind index -> 特徴kind index
        self.full_kind_arr = pa.array(list(full_kinds))
        # 全エージェント(活動判定用)
        self.all_ids = all_ids
        self.n_all = len(all_ids)
        max_id = (max(all_ids) if all_ids else -1)
        self.all_lut = np.full(max_id + 2, -1, dtype=np.int64)
        for i, aid in enumerate(all_ids):
            self.all_lut[aid] = i
        self.active = np.zeros((self.n_all, n_days), dtype=bool)
        # 特徴対象(サブサンプル)
        self.sel_ids = sel_ids
        self.n_sel = len(sel_ids)
        self.sel_lut = np.full(max_id + 2, -1, dtype=np.int64)
        for i, aid in enumerate(sel_ids):
            self.sel_lut[aid] = i
        self.kind_cells = np.zeros(self.n_sel * n_days * self.n_kinds, dtype=np.int64)
        self.day_kind = np.zeros(n_days * self.n_kinds, dtype=np.int64)
        self.day_events = np.zeros(n_days, dtype=np.int64)
        # 部分日(ラン開始/終了で切れた日)の検出用: 日ごとの sim_min の最小/最大
        self.day_min_sm = np.full(n_days, np.iinfo(np.int64).max, dtype=np.int64)
        self.day_max_sm = np.full(n_days, -1, dtype=np.int64)
        # ---- 語彙 ----
        self.item_first_coin: dict = {}
        self.item_first_adopt: dict = {}
        self.item_first_use: dict = {}
        self.day_coin_events = np.zeros(n_days, dtype=np.int64)
        self.day_adopt_events = np.zeros(n_days, dtype=np.int64)
        self.day_use_events = np.zeros(n_days, dtype=np.int64)
        self.day_item_use: list = [Counter() for _ in range(n_days)]
        # ---- 接触グラフ ----
        self.edge_first_day: dict = {}
        self.edges_truncated = False
        self.max_edges = max_edges
        # ---- relations 系 ----
        self.rel_pair_first_day: dict = {}
        self.day_rel_tier = np.zeros(n_days, dtype=np.int64)
        self.day_rel_break = np.zeros(n_days, dtype=np.int64)
        # ---- 訪問 ----
        self.visit: dict = defaultdict(Counter)   # (sel_idx, day) -> Counter[node]
        self.node_total: Counter = Counter()
        self.day_arrive = np.zeros(n_days, dtype=np.int64)
        self.day_first_visit = np.zeros(n_days, dtype=np.int64)
        self.visit_cells_truncated = False


def _scan_pass2(path: str, acc: Accum, batch_rows: int) -> None:
    pf = pq.ParquetFile(path)
    payload_set = pa.array(list(PAYLOAD_KINDS))
    coin = set(COIN_KINDS)
    adopt = set(ADOPT_KINDS)
    use = set(USE_KINDS)
    nd, nk = acc.n_days, acc.n_kinds
    for b in pf.iter_batches(batch_size=batch_rows,
                             columns=["sim_min", "agent_id", "kind", "payload"]):
        if b.num_rows == 0:
            continue
        sm = b.column("sim_min").to_numpy(zero_copy_only=False).astype(np.int64)
        day = np.clip(sm // SIM_MIN_PER_DAY, 0, nd - 1)
        aid = b.column("agent_id").to_numpy(zero_copy_only=False).astype(np.int64)
        ki = pc.index_in(b.column("kind"), value_set=acc.full_kind_arr) \
            .fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
        fk = acc.kind_lut[ki]
        # ---- 日次イベント数・日×kind ----
        acc.day_events += np.bincount(day, minlength=nd)
        acc.day_kind += np.bincount(day * nk + fk, minlength=nd * nk)
        for d in np.unique(day):                       # 日ごとの sim_min 範囲(部分日検出)
            s = sm[day == d]
            di = int(d)
            acc.day_min_sm[di] = min(int(acc.day_min_sm[di]), int(s.min()))
            acc.day_max_sm[di] = max(int(acc.day_max_sm[di]), int(s.max()))
        # ---- 活動フラグ(全エージェント)----
        ok = (aid >= 0) & (aid < acc.all_lut.size)
        if bool(ok.any()):
            ai_all = acc.all_lut[aid[ok]]
            ok2 = ai_all >= 0
            if bool(ok2.any()):
                acc.active[ai_all[ok2], day[ok][ok2]] = True
            # ---- 特徴セル(サブサンプル)----
            ai_sel = acc.sel_lut[aid[ok]]
            s2 = ai_sel >= 0
            if bool(s2.any()):
                flat = (ai_sel[s2] * nd + day[ok][s2]) * nk + fk[ok][s2]
                acc.kind_cells += np.bincount(flat, minlength=acc.n_sel * nd * nk)
        # ---- payload が要る行だけ抜いて JSON パース ----
        mask = pc.is_in(b.column("kind"), value_set=payload_set)
        fb = b.filter(mask)
        if fb.num_rows == 0:
            continue
        f_day = np.clip(fb.column("sim_min").to_numpy(zero_copy_only=False).astype(np.int64)
                        // SIM_MIN_PER_DAY, 0, nd - 1)
        f_aid = fb.column("agent_id").to_pylist()
        f_kind = fb.column("kind").to_pylist()
        f_pay = fb.column("payload").to_pylist()
        for i in range(fb.num_rows):
            k = f_kind[i]
            d = int(f_day[i])
            a = f_aid[i]
            try:
                p = json.loads(f_pay[i]) if f_pay[i] else {}
            except Exception:
                continue
            if not isinstance(p, dict):
                continue
            if k in coin:
                acc.day_coin_events[d] += 1
                iid = p.get("item_id")
                if iid is not None and iid not in acc.item_first_coin:
                    acc.item_first_coin[iid] = d
            elif k in adopt:
                acc.day_adopt_events[d] += 1
                iid = p.get("item_id")
                if iid is not None and iid not in acc.item_first_adopt:
                    acc.item_first_adopt[iid] = d
            elif k in use:
                acc.day_use_events[d] += 1
                iid = p.get("item_id")
                if iid is not None:
                    acc.day_item_use[d][iid] += 1
                    if iid not in acc.item_first_use:
                        acc.item_first_use[iid] = d
            elif k == "speak":
                if not isinstance(a, int) or a < 0:
                    continue
                for h in (p.get("hearers") or []):
                    if isinstance(h, int) and h >= 0 and h != a:
                        e = (min(a, h), max(a, h))
                        if e not in acc.edge_first_day:
                            if len(acc.edge_first_day) >= acc.max_edges:
                                acc.edges_truncated = True
                                continue
                            acc.edge_first_day[e] = d
            elif k == "dm":
                if not isinstance(a, int) or a < 0:
                    continue
                to = p.get("to")
                if isinstance(to, int) and to >= 0 and to != a:
                    e = (min(a, to), max(a, to))
                    if e not in acc.edge_first_day:
                        if len(acc.edge_first_day) >= acc.max_edges:
                            acc.edges_truncated = True
                        else:
                            acc.edge_first_day[e] = d
            elif k == "relation_tier":
                acc.day_rel_tier[d] += 1
                o = p.get("other")
                if isinstance(a, int) and a >= 0 and isinstance(o, int) and o >= 0 and o != a:
                    e = (min(a, o), max(a, o))
                    if e not in acc.rel_pair_first_day:
                        acc.rel_pair_first_day[e] = d
            elif k == "relation_break":
                acc.day_rel_break[d] += 1
            elif k == ARRIVE_KIND:
                node = p.get("node")
                if node is None:
                    continue
                acc.day_arrive[d] += 1
                if p.get("first_visit"):
                    acc.day_first_visit[d] += 1
                if not isinstance(a, int) or a < 0 or a >= acc.sel_lut.size:
                    continue
                si = int(acc.sel_lut[a])
                if si < 0:
                    continue
                acc.visit[(si, d)][node] += 1
                acc.node_total[node] += 1


# --------------------------------------------------------------------------- #
# 指標の組み立て
# --------------------------------------------------------------------------- #
def _daily_vocab(acc: Accum) -> list:
    nd = acc.n_days
    coin_new = np.zeros(nd, dtype=np.int64)
    adopt_new = np.zeros(nd, dtype=np.int64)
    use_new = np.zeros(nd, dtype=np.int64)
    for d in acc.item_first_coin.values():
        coin_new[d] += 1
    for d in acc.item_first_adopt.values():
        adopt_new[d] += 1
    for d in acc.item_first_use.values():
        use_new[d] += 1
    out = []
    prev_items: set = set()
    for d in range(nd):
        items = set(acc.day_item_use[d])
        n_items = len(items)
        out.append({
            "coin_events": int(acc.day_coin_events[d]),
            "new_items_coined": int(coin_new[d]),
            "adopt_events": int(acc.day_adopt_events[d]),
            "new_items_adopted": int(adopt_new[d]),
            "use_events": int(acc.day_use_events[d]),
            "items_used": n_items,
            "new_items_used": int(use_new[d]),
            "novelty_share": _r(use_new[d] / n_items) if n_items else None,
            "item_jaccard_prev": _r(_jaccard(items, prev_items)) if (d > 0 and (items or prev_items)) else None,
            "use_entropy_bits": _r(_entropy_bits(acc.day_item_use[d].values())),
        })
        prev_items = items
    return out


def _daily_graph(acc: Accum, topo_max_nodes: int) -> list:
    nd = acc.n_days
    by_day: list = [[] for _ in range(nd)]
    for e, d in acc.edge_first_day.items():
        by_day[d].append(e)
    rel_new = np.zeros(nd, dtype=np.int64)
    for d in acc.rel_pair_first_day.values():
        rel_new[d] += 1
    # ノード番号を密 index に(numpy 経路用)
    nodes = sorted({u for e in acc.edge_first_day for u in e})
    idx = {u: i for i, u in enumerate(nodes)}
    n_nodes = len(nodes)
    use_np = 0 < n_nodes <= topo_max_nodes
    A = np.zeros((n_nodes, n_nodes), dtype=bool) if use_np else None
    cum: dict = {}
    out = []
    for d in range(nd):
        for e in by_day[d]:
            cum[e] = 1
            if use_np:
                i, j = idx[e[0]], idx[e[1]]
                A[i, j] = True
                A[j, i] = True
        n_edges = len(cum)
        seen_nodes = {u for e in cum for u in e}
        n_seen = len(seen_nodes)
        if use_np:
            if n_seen:
                act = np.array([idx[u] for u in sorted(seen_nodes)], dtype=np.int64)
                topo = clustering_and_diameter_np(A[np.ix_(act, act)])
            else:
                topo = clustering_and_diameter_np(np.zeros((0, 0), dtype=bool))
        else:
            topo = clustering_and_diameter_py(cum)
        out.append({
            "new_edges": len(by_day[d]),
            "cum_edges": n_edges,
            "cum_nodes": n_seen,
            "new_edge_rate": _r(len(by_day[d]) / n_edges) if n_edges else None,
            "mean_degree": _r(2.0 * n_edges / n_seen) if n_seen else None,
            "clustering_coeff": topo["clustering_coeff"],
            "clustering_n_nodes": topo["clustering_n_nodes"],
            "diameter_lcc": topo["diameter_lcc"],
            "diameter_exact": topo["diameter_exact"],
            "lcc_size": topo["lcc_size"],
            "relation_tier_events": int(acc.day_rel_tier[d]),
            "new_relation_pairs": int(rel_new[d]),
            "relation_break_events": int(acc.day_rel_break[d]),
        })
    return out


def _daily_behavior(acc: Accum, ref_day: int) -> tuple:
    nd = acc.n_days
    # (agent, day) -> node set
    sets: dict = defaultdict(set)
    day_node_counts: list = [Counter() for _ in range(nd)]
    for (si, d), cnt in acc.visit.items():
        sets[(si, d)] = set(cnt)
        day_node_counts[d].update(cnt)
    v_total = len({n for c in day_node_counts for n in c})
    log_v = math.log2(v_total) if v_total > 1 else 0.0
    out = []
    for d in range(nd):
        arr = int(acc.day_arrive[d])
        fv = int(acc.day_first_visit[d])
        h = _entropy_bits(day_node_counts[d].values())
        # 前日 Jaccard(両日に訪問のあるエージェントのみ)
        js = []
        js_ref = []
        for si in range(acc.n_sel):
            cur = sets.get((si, d))
            if not cur:
                continue
            if d > 0:
                prv = sets.get((si, d - 1))
                if prv:
                    js.append(_jaccard(cur, prv))
            if ref_day is not None and d != ref_day:
                rf = sets.get((si, ref_day))
                if rf:
                    js_ref.append(_jaccard(cur, rf))
        out.append({
            "arrivals": arr,
            "first_visits": fv,
            "revisit_rate": _r(1.0 - fv / arr) if arr else None,
            "distinct_nodes": len(day_node_counts[d]),
            "visit_entropy_bits": _r(h),
            "visit_entropy_norm": _r(h / log_v) if log_v > 0 else None,
            "jaccard_prev_day": _r(float(np.mean(js))) if js else None,
            "jaccard_n_agents": len(js),
            "jaccard_vs_ref_day": _r(float(np.mean(js_ref))) if js_ref else None,
        })
    return out, v_total


def _build_features(acc: Accum, top_visit_nodes: int) -> dict:
    """(agent, day, dim) の確率ベクトル群。ブロック kind / visit / combined。"""
    nd, nk, ns = acc.n_days, acc.n_kinds, acc.n_sel
    K = acc.kind_cells.reshape(ns, nd, nk).astype(np.float32)
    ksum = K.sum(axis=2, keepdims=True)
    Kp = np.divide(K, ksum, out=np.zeros_like(K), where=ksum > 0)
    top_nodes = [n for n, _c in sorted(acc.node_total.items(), key=lambda kv: (-kv[1], str(kv[0])))
                 [:top_visit_nodes]]
    node_idx = {n: i for i, n in enumerate(top_nodes)}
    nv = len(top_nodes) + 1                      # +1 = "_other"
    V = np.zeros((ns, nd, nv), dtype=np.float32)
    for (si, d), cnt in acc.visit.items():
        for node, c in cnt.items():
            V[si, d, node_idx.get(node, nv - 1)] += c
    vsum = V.sum(axis=2, keepdims=True)
    Vp = np.divide(V, vsum, out=np.zeros_like(V), where=vsum > 0)
    C = np.concatenate([0.5 * Kp, 0.5 * Vp], axis=2)
    present = (ksum[:, :, 0] > 0)
    return {"kind": Kp, "visit": Vp, "combined": C, "present": present,
            "dims": {"kind": nk, "visit": nv, "combined": nk + nv},
            "top_nodes": top_nodes}


def _pair_test(F: dict, block: str, d1: int, d2: int, n_mc: int, seed: int) -> dict:
    X = F[block]
    pres = F["present"]
    sel = pres[:, d1] & pres[:, d2]
    n = int(sel.sum())
    if n == 0:
        return {"block": block, "day_a": d1, "day_b": d2, "n": 0, "p": None,
                "tvd": None, "method": "none"}
    D = (X[sel, d1, :] - X[sel, d2, :]).astype(np.float64)
    res = paired_signflip_tvd(D, n_mc=n_mc, rng_seed=seed)
    res.update(paired_dz(D))
    res["block"] = block
    res["day_a"] = int(d1)
    res["day_b"] = int(d2)
    return res


def _tvd_matrix(F: dict, block: str, full_days: np.ndarray) -> dict:
    """日ごとの平均プロフィール間 TVD 行列(置換なし=安価な全体像)。

    ノイズ床(adjacent)と lag 構造は **完全な日どうし** のペアのみで集計する
    (ラン開始/終了で切れた部分日は分布が別物なので混ぜない)。"""
    X = F[block]
    pres = F["present"]
    nd = X.shape[1]
    prof = np.zeros((nd, X.shape[2]), dtype=np.float64)
    for d in range(nd):
        s = pres[:, d]
        if int(s.sum()):
            prof[d] = X[s, d, :].mean(axis=0)
    M = np.zeros((nd, nd), dtype=np.float64)
    for i in range(nd):
        M[i] = 0.5 * np.abs(prof - prof[i]).sum(axis=1)
    adj = [float(M[d, d + 1]) for d in range(nd - 1)
           if full_days[d] and full_days[d + 1]]
    by_lag: dict = {}
    fidx = [d for d in range(nd) if full_days[d]]
    acc_lag: dict = defaultdict(list)
    for ii, a in enumerate(fidx):
        for b in fidx[ii + 1:]:
            acc_lag[b - a].append(float(M[a, b]))
    for lag in sorted(acc_lag):
        vals = acc_lag[lag]
        by_lag[str(lag)] = {"n": len(vals), "mean": _r(float(np.mean(vals))),
                            "median": _r(float(np.median(vals)))}
    # 「TVD が lag とともに増えるか」= ドリフトの有無(定常なら lag に平坦)
    slope = None
    if len(by_lag) >= 3:
        lags = np.array([float(k) for k in by_lag], dtype=np.float64)
        means = np.array([by_lag[k]["mean"] for k in by_lag], dtype=np.float64)
        A = np.vstack([lags, np.ones_like(lags)]).T
        slope = float(np.linalg.lstsq(A, means, rcond=None)[0][0])
    return {"matrix": [[round(float(v), 6) for v in row] for row in M],
            "adjacent": [round(v, 6) for v in adj],
            "adjacent_median": _r(float(np.median(adj))) if adj else None,
            "adjacent_max": _r(max(adj)) if adj else None,
            "by_lag": by_lag, "lag_slope": _r(slope)}


def _saturation_and_period(daily: list, full_days: np.ndarray, tv: dict,
                           thresholds: dict) -> dict:
    """立ち上がり(burn-in)の終端推定と周期性の検出(どちらも派生量・検定なし)。

    ★10 sim日ランでは日次自己相似性が「単に蓄積の過渡期」で有意になりうる。過渡が
      いつ終わるかを別に出しておかないと、Day2 vs Day5 のゲートは自明に通ってしまう。"""
    nd = len(daily)

    def _first_day(sub, key, op, thr):
        for d in range(nd):
            if not full_days[d]:
                continue
            v = daily[d][sub].get(key)
            if v is None:
                continue
            if (op == "<" and v < thr) or (op == ">" and v > thr):
                return d
        return None

    sat = {
        "vocab_novelty_below": {
            "threshold": thresholds["vocab_novelty_share_min"],
            "first_day": _first_day("vocab", "novelty_share", "<",
                                    thresholds["vocab_novelty_share_min"])},
        "new_edge_rate_below": {
            "threshold": thresholds["new_edge_rate_min"],
            "first_day": _first_day("graph", "new_edge_rate", "<",
                                    thresholds["new_edge_rate_min"])},
        "revisit_rate_above": {
            "threshold": thresholds["revisit_rate_max"],
            "first_day": _first_day("behavior", "revisit_rate", ">",
                                    thresholds["revisit_rate_max"])},
    }
    ends = [v["first_day"] for v in sat.values() if v["first_day"] is not None]
    sat["burn_in_end_estimate"] = max(ends) if ends else None
    sat["note"] = ("burn_in_end_estimate = 3 指標が飽和側へ入りきる最初の日(全て到達した場合のみ)。"
                   "本選 10 日ランはこの推定値より短い可能性が高く、その場合の非定常性は"
                   "「蓄積の過渡」であって「日常の非反復性」ではない。")

    by_lag = tv.get("by_lag", {})
    per = {"lag1_mean": None, "lag_min": None, "lag_min_mean": None, "lag_max_scanned": None}
    if by_lag:
        keys = sorted((int(k) for k in by_lag))
        per["lag1_mean"] = by_lag.get("1", {}).get("mean")
        scan = [k for k in keys if 2 <= k <= min(14, max(keys))]
        per["lag_max_scanned"] = max(scan) if scan else None
        if scan:
            best = min(scan, key=lambda k: (by_lag[str(k)]["mean"], k))
            per["lag_min"] = best
            per["lag_min_mean"] = by_lag[str(best)]["mean"]
    per["note"] = ("lag_min = lag 2..14 のうち平均 TVD が最小の lag。7 が出れば週次リズム"
                   "(world.calendar.weekday_work など)の指紋。周期は非定常性ではなく"
                   "**定常な周期構造**なので、日ペア検定の解釈時に必ず併読すること。")
    return {"saturation": sat, "periodicity": per}


# --------------------------------------------------------------------------- #
# 判定(事前登録ドラフトの閾値を機械適用)
# --------------------------------------------------------------------------- #
def _verdict(daily: list, pairs: list, adj_median, thresholds: dict, primary_pair,
             full_days: np.ndarray, late_pair=None) -> dict:
    nd = len(daily)
    fidx = [d for d in range(nd) if full_days[d]] or list(range(nd))
    lo_pos = int(len(fidx) * float(thresholds["late_window_frac"]))
    late_days = fidx[lo_pos:] or fidx
    lo = late_days[0]
    late = [daily[d] for d in late_days]
    def _mean(key, sub):
        vals = [row[sub][key] for row in late if row[sub].get(key) is not None]
        return float(np.mean(vals)) if vals else None
    nov = _mean("novelty_share", "vocab")
    ner = _mean("new_edge_rate", "graph")
    rvr = _mean("revisit_rate", "behavior")
    prim = None
    for p in pairs:
        if p.get("block") == "combined" and primary_pair and \
           (p.get("day_a"), p.get("day_b")) == tuple(primary_pair):
            prim = p
            break
    checks = {}
    if prim and prim.get("p") is not None and prim.get("tvd") is not None:
        ratio = (prim["tvd"] / adj_median) if (adj_median and adj_median > 0) else None
        checks["selfsim_p"] = {"value": prim["p"], "threshold": thresholds["p_max"],
                               "op": "<", "pass": bool(prim["p"] < thresholds["p_max"])}
        checks["selfsim_tvd"] = {"value": prim["tvd"], "threshold": thresholds["tvd_min"],
                                 "op": ">=", "pass": bool(prim["tvd"] >= thresholds["tvd_min"])}
        checks["selfsim_tvd_ratio"] = {"value": _r(ratio),
                                       "threshold": thresholds["tvd_ratio_min"], "op": ">=",
                                       "pass": bool(ratio is not None
                                                    and ratio >= thresholds["tvd_ratio_min"])}
    checks["vocab_novelty_share_late"] = {
        "value": _r(nov), "threshold": thresholds["vocab_novelty_share_min"], "op": ">=",
        "pass": bool(nov is not None and nov >= thresholds["vocab_novelty_share_min"])}
    checks["new_edge_rate_late"] = {
        "value": _r(ner), "threshold": thresholds["new_edge_rate_min"], "op": ">=",
        "pass": bool(ner is not None and ner >= thresholds["new_edge_rate_min"])}
    checks["revisit_rate_late"] = {
        "value": _r(rvr), "threshold": thresholds["revisit_rate_max"], "op": "<=",
        "pass": bool(rvr is not None and rvr <= thresholds["revisit_rate_max"])}
    # ---- 主判定と**同じ lag** の後期ペア(過渡か常態かの切り分け。事前登録 §4 の sustained)----
    info = {"late_same_lag_pair": list(late_pair) if late_pair else None}
    sustained = None
    if late_pair:
        for p in pairs:
            if p.get("block") == "combined" and \
               (p.get("day_a"), p.get("day_b")) == tuple(late_pair):
                ratio = (p["tvd"] / adj_median) if (p.get("tvd") is not None
                                                    and adj_median) else None
                sustained = bool(
                    p.get("p") is not None and p["p"] < thresholds["p_max"]
                    and p.get("tvd") is not None and p["tvd"] >= thresholds["tvd_min"]
                    and ratio is not None and ratio >= thresholds["tvd_ratio_min"])
                info.update({"tvd": p.get("tvd"), "p": p.get("p"),
                             "tvd_ratio": _r(ratio), "would_pass_primary": sustained})
                break
        info["note"] = ("主判定ペアと同じ lag を後期(過渡が終わった後)で取り直したもの。"
                        "主判定が PASS でこちらが FAIL なら、非定常性の正体は"
                        "**立ち上がりの過渡**であって日常の非反復性ではない。")

    # 主判定 = 自己相似性(p かつ 効果量 かつ 比)。構造3指標は補助(参考)。
    # ラベルは事前登録 docs/plans/stationarity-preregistration.md §4 の判定式と逐語一致。
    keys = ("selfsim_p", "selfsim_tvd", "selfsim_tvd_ratio")
    if all(k in checks for k in keys):
        primary_pass = all(checks[k]["pass"] for k in keys)
        if not primary_pass:
            label = "STATIONARY_LIKE"
        elif sustained is True:
            label = "SUSTAINED_NONSTATIONARY"
        elif sustained is False:
            label = "TRANSIENT_ONLY"
        else:
            label = "NONSTATIONARY_UNVERIFIED"     # 後期ペアが取れず過渡と分離できない
    else:
        primary_pass = None
        label = "INCONCLUSIVE"
    aux = [checks[k]["pass"] for k in ("vocab_novelty_share_late", "new_edge_rate_late",
                                       "revisit_rate_late")]
    return {"status": "DRAFT (unapproved thresholds)", "label": label,
            "late_same_lag": info, "sustained_pass": sustained,
            "primary_pass": primary_pass, "n_aux_pass": int(sum(1 for v in aux if v)),
            "n_aux": len(aux), "checks": checks,
            "primary_pair": list(primary_pair) if primary_pair else None,
            "late_window_days": [lo, late_days[-1]],
            "late_window_is_full_days_only": bool(full_days.any()),
            "note": ("閾値は docs/plans/stationarity-preregistration.md のドラフト値。"
                     "ユーザー承認(U-10)+実 LLM 診断ランを経るまで確定ではない。")}


# --------------------------------------------------------------------------- #
# 本体
# --------------------------------------------------------------------------- #
def analyze(run_dir: str, *, mc: int = MC_ITER, seed: int = RNG_SEED,
            pairs: list | None = None, ref_day: int | None = None,
            max_feature_kinds: int = MAX_FEATURE_KINDS,
            max_feature_agents: int = MAX_FEATURE_AGENTS,
            top_visit_nodes: int = TOP_VISIT_NODES,
            topo_max_nodes: int = TOPO_MAX_NODES,
            batch_rows: int = BATCH_ROWS,
            max_edges: int = MAX_EDGES,
            thresholds: dict | None = None) -> dict:
    path = os.path.join(run_dir, "l1_events.parquet")
    if not os.path.isfile(path):
        raise SystemExit(f"l1_events.parquet が無い: {path}")
    thresholds = dict(thresholds or DRAFT_THRESHOLDS)
    run_name = os.path.basename(os.path.normpath(run_dir))
    notes: list = []

    # ---- パス1 ----
    p1 = _scan_pass1(path, batch_rows)
    n_days = int(p1["max_sim_min"] // SIM_MIN_PER_DAY) + 1
    all_ids = p1["agent_ids"]
    if not all_ids:
        raise SystemExit("agent_id >= 0 のイベントが無い(診断不能)")
    if not p1["sim_min_monotonic"]:
        notes.append("sim_min が単調でない行がある(day = sim_min//1440 の割り当ては維持)。")

    # 特徴 kind: 出現数上位 K + "_other"
    full_kinds = sorted(p1["kind_counts"])
    ranked = [k for k, _c in sorted(p1["kind_counts"].items(), key=lambda kv: (-kv[1], kv[0]))]
    if len(ranked) > max_feature_kinds:
        keep = sorted(ranked[:max_feature_kinds])
        feat_kinds = keep + ["_other"]
        notes.append(f"kind が {len(ranked)} 種 > 上限 {max_feature_kinds}: "
                     f"上位 {max_feature_kinds} 種を残し残りを '_other' に集約した。")
    else:
        feat_kinds = list(full_kinds)
    fk_index = {k: i for i, k in enumerate(feat_kinds)}
    other_i = fk_index.get("_other", len(feat_kinds) - 1)
    kind_lut = np.array([fk_index.get(k, other_i) for k in full_kinds], dtype=np.int64)

    # 特徴エージェント: 上限超過なら等間隔サブサンプル(乱数不使用=決定論)
    if len(all_ids) > max_feature_agents:
        step = max(1, len(all_ids) // max_feature_agents)
        sel_ids = all_ids[::step][:max_feature_agents]
        notes.append(f"エージェント {len(all_ids)} 体 > 上限 {max_feature_agents}: "
                     f"id 昇順の等間隔サブサンプル {len(sel_ids)} 体で特徴量・訪問系を計算した"
                     "(日次の集計量は全エージェント)。")
    else:
        sel_ids = list(all_ids)

    acc = Accum(n_days, all_ids, feat_kinds, kind_lut, sel_ids, max_edges, full_kinds)

    # ---- パス2 ----
    _scan_pass2(path, acc, batch_rows)
    if acc.edges_truncated:
        notes.append(f"接触グラフの辺数が上限 {max_edges} に達して打ち切った(グラフ指標は下界)。")

    # ---- 部分日(ラン開始/終了で切れた日)の検出 ----
    span = np.where(acc.day_max_sm >= 0, acc.day_max_sm - acc.day_min_sm, -1)
    span_ref = int(span.max()) if span.size and span.max() > 0 else 0
    full_days = (span >= 0.9 * span_ref) & (acc.day_events > 0) if span_ref > 0 \
        else (acc.day_events > 0)
    if not bool(full_days.any()):
        full_days = acc.day_events > 0
    n_partial = int((~full_days & (acc.day_events > 0)).sum())
    if n_partial:
        notes.append(f"部分日(sim_min の張り出しが最大日の 90% 未満)が {n_partial} 日ある: "
                     f"{[int(d) for d in np.where(~full_days & (acc.day_events > 0))[0]]}。"
                     "ノイズ床・lag 構造・後半窓・既定ペアからは除外した(日次系列には残す)。")

    # ---- 日次指標 ----
    ref = ref_day if ref_day is not None else min(2, n_days - 1)
    vocab = _daily_vocab(acc)
    graph = _daily_graph(acc, topo_max_nodes)
    behavior, v_total = _daily_behavior(acc, ref)
    n_active = acc.active.sum(axis=0)
    daily = []
    for d in range(n_days):
        daily.append({"day": d, "n_events": int(acc.day_events[d]),
                      "n_active_agents": int(n_active[d]),
                      "full_day": bool(full_days[d]),
                      "sim_min_span": int(span[d]) if span[d] >= 0 else None,
                      "vocab": vocab[d], "graph": graph[d], "behavior": behavior[d]})

    # ---- 自己相似性 ----
    F = _build_features(acc, top_visit_nodes)
    mat = {b: _tvd_matrix(F, b, full_days) for b in ("kind", "visit", "combined")}
    adj_median = mat["combined"]["adjacent_median"]

    fidx = [d for d in range(n_days) if full_days[d]]
    fset = set(fidx)
    user_pairs = pairs is not None
    if pairs is None:
        cand = [d for d in (2, 5, 20) if d in fset]
        if fidx:
            cand.append(fidx[-1])
        cand = sorted(set(cand))
        pairs = []
        base = cand[0] if cand else 0
        for d in cand[1:]:
            pairs.append((base, d))
        if len(fidx) >= 2:                  # ノイズ床(末尾の完全日どうしの隣接ペア)
            pairs.append((fidx[-2], fidx[-1]))
        if not pairs and n_days >= 2:
            pairs = [(0, n_days - 1)]
    primary_pair = None
    for a, b in pairs:
        if (a, b) == (2, 5):
            primary_pair = (a, b)
            break
    if primary_pair is None and pairs:
        primary_pair = tuple(sorted(pairs)[0])
    # ---- 主判定と同じ lag を「後期」で取り直すペア(過渡 vs 常態の切り分け・参考)----
    late_pair = None
    if primary_pair and len(fidx) >= 2:
        lag = int(primary_pair[1] - primary_pair[0])
        start = fidx[min(len(fidx) - 1, int(0.6 * len(fidx)))]
        cands = [d for d in fidx if d >= start and (d + lag) in fset]
        if not cands:
            cands = [d for d in fidx if (d + lag) in fset]
        if cands:
            cand_late = (cands[-1], cands[-1] + lag)
            if cand_late != tuple(primary_pair):
                late_pair = cand_late
                if not user_pairs and late_pair not in pairs:
                    pairs.append(late_pair)
    pairs = sorted(set(tuple(p) for p in pairs))

    tests = []
    for (a, b) in pairs:
        for block in ("combined", "kind", "visit"):
            tests.append(_pair_test(F, block, a, b, mc, seed))

    verdict = _verdict(daily, tests, adj_median, thresholds, primary_pair, full_days,
                       late_pair=late_pair)
    derived = _saturation_and_period(daily, full_days, mat["combined"], thresholds)

    return {
        "run": run_name,
        "generated_by": "scripts/diagnose_stationarity.py",
        "spec": ("docs/plans/twin-physics-vision-affordance-plan.md §2 レーン1 D0 / "
                 "docs/plans/stationarity-preregistration.md"),
        "params": {"mc_iter": mc, "rng_seed": seed, "exhaust_max": EXHAUST_MAX,
                   "sim_min_per_day": SIM_MIN_PER_DAY, "batch_rows": batch_rows,
                   "max_feature_kinds": max_feature_kinds,
                   "max_feature_agents": max_feature_agents,
                   "top_visit_nodes": top_visit_nodes, "topo_max_nodes": topo_max_nodes,
                   "max_edges": max_edges, "ref_day": ref,
                   "pairs": [list(p) for p in pairs],
                   "primary_pair": list(primary_pair) if primary_pair else None,
                   "late_same_lag_pair": list(late_pair) if late_pair else None,
                   "thresholds": thresholds},
        "read_strategy": {
            "source": "l1_events.parquet only (sim 本体 import ゼロ)",
            "pass1_columns": ["sim_min", "agent_id", "kind"],
            "pass2_columns": ["sim_min", "agent_id", "kind", "payload"],
            "payload_parsed_kinds": list(PAYLOAD_KINDS),
            "method": ("pyarrow ParquetFile.iter_batches によるストリーム走査。"
                       "payload は is_in マスクで必要 kind の行だけ to_pylist して json.loads。"
                       "特徴量は np.bincount で密配列へ直接畳み込み(Python ループ無し)。"),
        },
        "meta": {"n_events": int(p1["n_rows"]), "n_days": n_days,
                 "n_full_days": int(full_days.sum()),
                 "full_days": [int(d) for d in fidx],
                 "n_agents_total": len(all_ids), "n_agents_featured": len(sel_ids),
                 "n_kind_types": len(full_kinds), "feature_dims": F["dims"],
                 "n_distinct_visited_nodes": v_total,
                 "sim_min_monotonic": bool(p1["sim_min_monotonic"]),
                 "contact_edges_total": len(acc.edge_first_day),
                 "vocab_items_coined": len(acc.item_first_coin),
                 "vocab_items_adopted": len(acc.item_first_adopt),
                 "vocab_items_used": len(acc.item_first_use)},
        "daily": daily,
        "self_similarity": {"tests": tests, "tvd": mat},
        "derived": derived,
        "verdict": verdict,
        "notes": notes + [
            "本ツールは L1 を読むだけの事後観測であり、シミュレーションの決定論・乱数・LLM 呼数に一切入力しない。",
            "TVD は『平均的なエージェントの1日の活動プロフィールのうち何割が入れ替わったか』。",
            "n が大きいと極小の差でも p<0.05 になる。判定は p と効果量の連言で行うこと。",
            "mock ランの語彙・発話は決定論なので、語彙系の水準は実 LLM と挙動が異なる(配線確認+構造系の基準値として読む)。",
        ],
    }


# --------------------------------------------------------------------------- #
# 日本語レポート
# --------------------------------------------------------------------------- #
def _fmt(v, nd=4):
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return str(v)
    return f"{v:.{nd}f}"


def write_report(res: dict, path: str) -> None:
    L: list = []
    meta = res["meta"]
    par = res["params"]
    L.append(f"# 非定常性診断: {res['run']}\n")
    L.append("> **ステータス: ドラフト判定**。閾値は `docs/plans/stationarity-preregistration.md` の")
    L.append("> ドラフト値であり、ユーザー承認(U-10)と実 LLM 診断ラン(GPU 開放初日)を経て確定する。\n")
    L.append("Day t と Day t' が統計的に区別できるかを L1 だけから測る(原指示 D / 計画書 §2 レーン1 D0)。\n")

    L.append("## 0. 読み込み戦略(大規模ラン対応)")
    rs = res["read_strategy"]
    L.append(f"- 入力: `{rs['source']}`")
    L.append(f"- パス1 列: `{rs['pass1_columns']}` / パス2 列: `{rs['pass2_columns']}`")
    L.append(f"- {rs['method']}")
    L.append(f"- payload を JSON パースする kind: {', '.join(rs['payload_parsed_kinds'])}")
    L.append("")

    L.append("## 1. ラン概要")
    L.append("| 項目 | 値 |")
    L.append("|---|---:|")
    L.append(f"| イベント総数 | {meta['n_events']:,} |")
    L.append(f"| sim 日数(day = sim_min // 1440) | {meta['n_days']}"
             f"(うち完全日 {meta['n_full_days']}) |")
    L.append(f"| エージェント(全 / 特徴計算対象) | {meta['n_agents_total']} / {meta['n_agents_featured']} |")
    L.append(f"| イベント種別数 | {meta['n_kind_types']} |")
    L.append(f"| 特徴次元(kind / visit / combined) | {meta['feature_dims']['kind']} / "
             f"{meta['feature_dims']['visit']} / {meta['feature_dims']['combined']} |")
    L.append(f"| 接触グラフ 総辺数 | {meta['contact_edges_total']:,} |")
    L.append(f"| 訪問ノード種類数 | {meta['n_distinct_visited_nodes']:,} |")
    L.append(f"| 造語 item(coin / adopt / use) | {meta['vocab_items_coined']} / "
             f"{meta['vocab_items_adopted']} / {meta['vocab_items_used']} |")
    L.append("")

    L.append("## 2. 日次系列")
    daily = res["daily"]
    nd = len(daily)
    stride = 1 if nd <= 24 else max(1, nd // 20)
    shown = [d for d in range(nd) if d % stride == 0]
    if nd - 1 not in shown:
        shown.append(nd - 1)
    L.append(f"(全 {nd} 日。{'全日' if stride == 1 else f'{stride} 日ごと + 最終日'}を表示。全系列は JSON。)\n")
    L.append("| day | events | active | 新語 | 新規採用 | 使用語彙 | 新規性 | 語彙H(bit) | "
             "新規辺 | 累積辺 | 平均次数 | クラスタ係数 | 直径 | 回帰率 | 訪問H(bit) | 訪問H正規 | 前日Jaccard |")
    L.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for d in shown:
        r = daily[d]
        v, g, b = r["vocab"], r["graph"], r["behavior"]
        tag = "" if r.get("full_day", True) else "*"
        L.append(
            f"| {r['day']}{tag} | {r['n_events']:,} | {r['n_active_agents']} | "
            f"{v['new_items_coined']} | {v['new_items_adopted']} | {v['items_used']} | "
            f"{_fmt(v['novelty_share'], 3)} | {_fmt(v['use_entropy_bits'], 2)} | "
            f"{g['new_edges']} | {g['cum_edges']} | {_fmt(g['mean_degree'], 2)} | "
            f"{_fmt(g['clustering_coeff'], 3)} | {g['diameter_lcc']} | "
            f"{_fmt(b['revisit_rate'], 3)} | {_fmt(b['visit_entropy_bits'], 2)} | "
            f"{_fmt(b['visit_entropy_norm'], 3)} | {_fmt(b['jaccard_prev_day'], 3)} |")
    L.append("")
    L.append("`*` 付きの day は**部分日**(ラン開始/終了で切れた日)= ノイズ床・lag 構造・後半窓・"
             "既定ペアからは除外している。")
    L.append("")
    L.append("列の定義: 新語=その日に初めて造語された item 数(label_coin/vocab_coin の item_id 初出)/ "
             "新規採用=label_adopt の item_id 初出 / 新規性=その日使われた語のうち初出の割合 / "
             "新規辺=接触グラフ(speak.hearers + dm.to)で初出の無向ペア / 回帰率=1−first_visit率(EPR 優先的回帰)/ "
             "訪問H正規=訪問エントロピー ÷ log2(ラン全体の訪問ノード種類数) / "
             "前日Jaccard=各エージェントの訪問ノード集合の前日との Jaccard 平均。")
    L.append("")

    L.append("## 3. 日次自己相似性(主判定): Day t vs Day t'")
    tv = res["self_similarity"]["tvd"]["combined"]
    L.append(f"- ノイズ床: 隣接日 TVD 中央値 = **{_fmt(tv['adjacent_median'], 4)}** "
             f"(最大 {_fmt(tv['adjacent_max'], 4)})")
    L.append(f"- 検定: paired sign-flip permutation(統計量=TVD・両側・`default_rng({par['rng_seed']})` 固定・"
             f"n>{EXHAUST_MAX} は MC {par['mc_iter']:,} 回で p=(b+1)/(m+1))")
    L.append("")
    L.append("| ブロック | Day A | Day B | n(両日在席) | TVD | TVD/隣接中央値 | p | 方法 | dz平均 | dz最大 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---|---:|---:|")
    am = tv["adjacent_median"]
    for t in res["self_similarity"]["tests"]:
        ratio = (t["tvd"] / am) if (t.get("tvd") is not None and am) else None
        L.append(f"| {t['block']} | {t['day_a']} | {t['day_b']} | {t['n']} | "
                 f"{_fmt(t.get('tvd'), 4)} | {_fmt(ratio, 2)} | {_fmt(t.get('p'), 4)} | "
                 f"{t.get('method', '—')} | {_fmt(t.get('dz_mean_abs'), 3)} | "
                 f"{_fmt(t.get('dz_max_abs'), 3)} |")
    L.append("")
    L.append("> ★n が数百あると **ごく小さな差でも p<0.05 になる**。したがって合否は p と効果量(TVD)の")
    L.append("> **連言**で判定する(事前登録文書 §4 の判定式)。TVD/隣接中央値は「隣接日どうしの揺らぎの")
    L.append("> 何倍離れているか」= スケールフリーな比。")
    L.append("")
    L.append("### 3b. lag 構造(完全日どうしの全ペア・置換なし)")
    L.append("定常なら TVD は lag に対して**平坦**、ドリフトがあれば lag とともに増える。")
    L.append("")
    L.append("| lag(日) | ペア数 | TVD 平均 | TVD 中央値 |")
    L.append("|---:|---:|---:|---:|")
    by_lag = tv.get("by_lag", {})
    lag_keys = sorted(by_lag, key=lambda s: int(s))
    for k in (lag_keys if len(lag_keys) <= 20
              else lag_keys[:10] + lag_keys[len(lag_keys) // 2:len(lag_keys) // 2 + 5]
              + lag_keys[-5:]):
        r = by_lag[k]
        L.append(f"| {k} | {r['n']} | {_fmt(r['mean'], 4)} | {_fmt(r['median'], 4)} |")
    L.append("")
    L.append(f"- lag 回帰の傾き(TVD 平均 ~ lag の OLS): **{_fmt(tv.get('lag_slope'), 6)}** "
             "(0 に近い = 定常的 / 正で大きい = ドリフト)")
    L.append("")

    der = res.get("derived", {})
    sat = der.get("saturation", {})
    per = der.get("periodicity", {})
    L.append("### 3c. 立ち上がり(burn-in)と周期性")
    L.append("| 指標 | 閾値 | 到達した最初の day |")
    L.append("|---|---:|---:|")
    for k, lbl in (("vocab_novelty_below", "語彙新規性が閾値未満"),
                   ("new_edge_rate_below", "新規紐帯率が閾値未満"),
                   ("revisit_rate_above", "回帰率が閾値超")):
        c = sat.get(k, {})
        L.append(f"| {lbl} | {_fmt(c.get('threshold'), 4)} | "
                 f"{c.get('first_day') if c.get('first_day') is not None else '未到達'} |")
    L.append("")
    L.append(f"- **burn_in_end_estimate = {sat.get('burn_in_end_estimate')}**(3 指標すべてが"
             "飽和側へ入りきる最初の日。未到達なら None)")
    L.append(f"- {sat.get('note', '')}")
    L.append(f"- 周期性: lag 2..{per.get('lag_max_scanned')} で平均 TVD が最小になる lag = "
             f"**{per.get('lag_min')}**(平均 TVD {_fmt(per.get('lag_min_mean'), 4)} / "
             f"lag1 は {_fmt(per.get('lag1_mean'), 4)})")
    L.append(f"- {per.get('note', '')}")
    L.append("")

    lsl = res["verdict"].get("late_same_lag") or {}
    if lsl.get("late_same_lag_pair"):
        L.append("### 3d. 過渡 vs 常態(主判定と同じ lag の後期ペア・参考)")
        L.append(f"- 後期ペア: Day {lsl['late_same_lag_pair']} / TVD = {_fmt(lsl.get('tvd'), 4)} / "
                 f"p = {_fmt(lsl.get('p'), 4)} / TVD比 = {_fmt(lsl.get('tvd_ratio'), 2)} / "
                 f"主判定条件を満たすか = {lsl.get('would_pass_primary')}")
        L.append(f"- {lsl.get('note', '')}")
        L.append("")

    L.append("## 4. 判定(ドラフト閾値の機械適用)")
    vd = res["verdict"]
    L.append(f"- **判定ラベル: {vd['label']}** / ステータス: {vd['status']}")
    L.append(f"- 主判定ペア: Day {vd['primary_pair']} / 後半窓: day {vd['late_window_days'][0]}"
             f"–{vd['late_window_days'][1]}")
    L.append("")
    L.append("| チェック | 実測 | 演算 | 閾値 | 合否 |")
    L.append("|---|---:|:--:|---:|:--:|")
    for k, c in vd["checks"].items():
        L.append(f"| {k} | {_fmt(c['value'], 4)} | {c['op']} | {_fmt(c['threshold'], 4)} | "
                 f"{'PASS' if c['pass'] else 'FAIL'} |")
    L.append("")
    L.append(f"- primary(主判定ペアの 3 条件の連言): {vd['primary_pass']}")
    L.append(f"- sustained(後期・同 lag ペアの 3 条件の連言): {vd.get('sustained_pass')}")
    L.append(f"- 補助(構造 3 指標): {vd['n_aux_pass']}/{vd['n_aux']} PASS")
    L.append(f"- {vd['note']}")
    L.append("")
    L.append("ラベルの意味(事前登録文書 §4 の判定式):")
    L.append("- **SUSTAINED_NONSTATIONARY** = primary ∧ sustained。持続的に非定常 → D1 は OFF のまま。")
    L.append("- **TRANSIENT_ONLY** = primary ∧ ¬sustained。非定常性は**立ち上がりの過渡のみ** → "
             "D1 は OFF のまま(ON は別条件ラン)。")
    L.append("- **STATIONARY_LIKE** = ¬primary。Day t と Day t' が区別できない(『饒舌な日常』の反復)"
             " → D1 の ON をユーザーへ提案。")
    L.append("- **NONSTATIONARY_UNVERIFIED** = primary だが後期ペアが取れず過渡と分離できない。")
    L.append("- **INCONCLUSIVE** = 判定に必要な量が欠測。合格に読み替えない。")
    L.append("")
    L.append("いずれの場合も結果は**設計空間の上界として報告する**(事前登録文書 §5)。")
    L.append("ON/OFF の最終判断は必ずユーザーが行う(本ツールは機械判定を出すだけ)。")
    L.append("")

    L.append("## 5. 正直註記")
    for n in res["notes"]:
        L.append(f"- {n}")
    L.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


# --------------------------------------------------------------------------- #
def _parse_pairs(s: str) -> list:
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        a, b = tok.split(":")
        out.append((int(a), int(b)))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="D0 非定常性診断(L1 を読むだけ・シム本体に非依存)")
    ap.add_argument("run_dir", help="例: runs/daily300_100d")
    ap.add_argument("--pairs", default=None,
                    help="比較する sim 日ペア(0始まり)。例: 2:5,2:20,2:99。既定は 2:5,2:20,2:last,末尾隣接")
    ap.add_argument("--ref-day", type=int, default=None, help="Jaccard の参照日(既定 2)")
    ap.add_argument("--mc", type=int, default=MC_ITER, help=f"置換 MC 反復(既定 {MC_ITER})")
    ap.add_argument("--seed", type=int, default=RNG_SEED, help="numpy default_rng の seed(既定 0)")
    ap.add_argument("--max-feature-kinds", type=int, default=MAX_FEATURE_KINDS)
    ap.add_argument("--max-feature-agents", type=int, default=MAX_FEATURE_AGENTS)
    ap.add_argument("--top-visit-nodes", type=int, default=TOP_VISIT_NODES)
    ap.add_argument("--topo-max-nodes", type=int, default=TOPO_MAX_NODES)
    ap.add_argument("--batch-rows", type=int, default=BATCH_ROWS)
    ap.add_argument("--max-edges", type=int, default=MAX_EDGES)
    ap.add_argument("--out-dir", default=None, help="出力先(既定: run_dir 直下)")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.run_dir):
        raise SystemExit(f"not a directory: {args.run_dir}")
    pairs = _parse_pairs(args.pairs) if args.pairs else None

    res = analyze(args.run_dir, mc=args.mc, seed=args.seed, pairs=pairs,
                  ref_day=args.ref_day, max_feature_kinds=args.max_feature_kinds,
                  max_feature_agents=args.max_feature_agents,
                  top_visit_nodes=args.top_visit_nodes,
                  topo_max_nodes=args.topo_max_nodes, batch_rows=args.batch_rows,
                  max_edges=args.max_edges)

    out_dir = args.out_dir or args.run_dir
    os.makedirs(out_dir, exist_ok=True)
    js_path = os.path.join(out_dir, "diagnose_stationarity.json")
    md_path = os.path.join(out_dir, "diagnose_stationarity.md")
    with open(js_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(res, sort_keys=True, ensure_ascii=False))
    write_report(res, md_path)

    vd = res["verdict"]
    prim = None
    for t in res["self_similarity"]["tests"]:
        if t["block"] == "combined" and vd["primary_pair"] and \
           [t["day_a"], t["day_b"]] == vd["primary_pair"]:
            prim = t
            break
    print(f"[stationarity] {args.run_dir}: days={res['meta']['n_days']} "
          f"agents={res['meta']['n_agents_total']} events={res['meta']['n_events']:,}")
    if prim:
        print(f"  primary Day{prim['day_a']} vs Day{prim['day_b']}: "
              f"TVD={prim['tvd']} p={prim['p']} n={prim['n']}")
    print(f"  verdict={vd['label']} ({vd['status']})")
    print(f"  -> {js_path}")
    print(f"  -> {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
