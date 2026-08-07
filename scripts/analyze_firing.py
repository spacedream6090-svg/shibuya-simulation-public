#!/usr/bin/env python3
"""思考頻度の観測装置化(第83バッチ)= 認知イベント列そのものを解析する。

正典: docs/plans/source/physics-instructions.md **Part P5**(エージェント別・文脈別の
      思考間隔分布 / 顕著性発火の原因種別内訳 / 可変思考 ON-OFF 対照)・
      docs/plans/source/cognition-design-record.md **§8**(観測量)。

    python scripts/analyze_firing.py runs/day1_fire
    python scripts/analyze_firing.py runs/day1_fire --min-contrib 1.0 --hit-sigma 1.0

設計 §8 が名指しした観測量のうち、本スクリプトが担う 5 つ
---------------------------------------------------------
① **思考間隔分布**(エージェント別・文脈別)。P5 の第 1 項。
② **顕著性発火の原因種別内訳**。設計 §2.4 の 3 源(外界 / 身体 / 予測不成立)+
   名前付きトリガ項の寄与を、L1 `cog_fire.contrib` の寄与内訳から按分する。
③ **発火数の時系列とバースト検出**。§8:「バースト = 集合的注意のイベント。
   L2(物語的虚構)の発生指標になりうる」。同時発火数も出す。
④ **発火連鎖グラフ**。§8:「A の行動が B のチャンネルに誤差を生み B が発火した、という
   **因果経路**が記録から直接辿れる」。★**候補経路として出力し確度を付す**(下記)。
⑤ **予測的中率(チャンネル別)**。§8「世界モデルの精度」。

実装前リサーチ(バースト検出とバースト性の標準)
------------------------------------------------
- **Kleinberg (2002)** *Bursty and hierarchical structure in streams*, Data Min Knowl
  Disc 7(4):373-397。イベント列を**無限状態オートマトン**でモデル化し、状態遷移として
  バーストを取り出す。上位状態への遷移にはコストが掛かり(下位へは無料)、Viterbi で
  最適な状態列を解く。本実装は同論文の **batched arrivals(離散ビンの二項モデル)版の
  2 状態**(平常 p₀ / バースト p₁ = s·p₀)を使う。階層版(無限状態)は本 step 数では
  過剰なので採らない。ビンは step、母数 n_t はその step の在場エージェント数。
- **Goh & Barabási (2008)** *Burstiness and memory in complex systems*, EPL 81:48002 の
  **バースト性パラメータ** `B = (r−1)/(r+1)`(r = 間隔の SD/平均)。
  B = −1 完全周期 / 0 ポアソン / →1 極端にバースト的。
- **Kim & Jo (2016)** *Measuring burstiness for finite event sequences*, Phys Rev E
  94:032311。B は**有限標本で系統的に負に偏る**(短い列ほど −1 側へ寄る)ので、
  有限長補正 `A_n` を併記する。本実装は両方出す(補正なしの B だけを載せない)。
- 人間活動の間隔分布が**裾の重いバースト構造**を持つことは Barabási (2005) Nature
  435:207 以来の標準的知見。したがって「平均間隔」だけを報告するのは不適切で、
  分位点・CV・B を必ず併記する(本スクリプトはそうしている)。

④ 発火連鎖について — **過剰主張しない**
----------------------------------------
本スクリプトが出すのは **候補経路(candidate)** であって因果の証明ではない。

  * 決定論規則で復元する: 「B が step N に salience 発火した」「その S の主寄与が
    チャンネル c だった」「c を動かしうる L1 イベントが step N−lag に存在した」
    の 3 点が揃った組だけを候補として出す。
  * **lag は 1 step が正しい**(推測ではない)。`fire.observe_end()` は step 末に o_c を
    凍結し、次 step の S 判定はその 1 枚だけを読む(ダブルバッファ)。よって step N の
    偏差を作った出来事は **step N−1 の世界**にある。
  * 確度は 3 段: `high` = L1 イベント自身が当事者 2 人を名指ししている
    (`hear{speaker}` / `dm{to}` / `flyer_view{author}`)。`medium` = 近接からの推定で
    候補が 1 人。`low` = 近接候補が複数(誰の寄与かは分離できない)。
  * **偽陰性も偽陽性もある**: ô が既にその出来事を織り込んでいれば誤差は出ないし
    (偽陰性)、たまたま同時刻・同場所に居ただけの他人も候補に挙がる(偽陽性)。
    件数を「A が B を発火させた回数」と読んではならない。

★ 本スクリプトは **読み取り専用**。シムを 1 バイトも変えない・乱数ゼロ・LLM ゼロ。

正直な限界
----------
- **`contrib` は上位 `cognition.fire.max_contrib` 件しか L1 に残らない**。既定 3 なので、
  ⑤ のチャンネル別統計は「上位寄与に入ったとき」に条件づけられた量になる。
  `max_contrib >= usable チャンネル数` のランでのみ完全であり、本スクリプトはそれを
  ランの config から機械判定して `complete: true/false` を出力に載せる。
- ⑤ は `contrib = g·|o−ô|/σ` を g で割って `|o−ô|/σ` に戻す。g は `cognition_g.parquet`
  の **1 step 前の行**(発火判定が読む g は前 step 末の値)。サイドカーが無い場合、
  g_update が OFF なら g=1 が厳密に正しく、ON なら**復元不能**として宣言する。
- P5 の「物理領域内外での思考頻度の差」は **物理層(Part P3)が未実装**のため出せない。
  出力に `physics_zones: not_implemented` と明記する(黙って省略しない)。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts") not in sys.path:      # l1_stream(共有の逐次読み)用
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from society.cognition import channels as CH          # noqa: E402
from society.cognition import fire as FIRE            # noqa: E402

SCHEMA = 1

# 思考間隔ヒストグラムのビン境界[分](最後は +inf)。
BIN_EDGES: tuple[float, ...] = (0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0,
                                90.0, 120.0, 180.0)

# ---- ④ 発火連鎖の対応表: 「チャンネル c を動かしうる L1 イベント」の決定論規則 ----
#  形式: channel_id -> [(L1 kind, 突き合わせ方, 相手を指すキー)]
#    "self_named"  … そのイベントの agent_id が **発火した本人 B** で、payload が相手 A を名指す
#    "other_named" … そのイベントの agent_id が **相手 A** で、payload が B を名指す
#    "nearby"      … 近接からの推定(位置の距離 <= radius)。候補が複数なら確度 low
CHAIN_RULES: dict[str, tuple[tuple[str, str, str], ...]] = {
    "ext.heard": (("hear", "self_named", "speaker"),
                  ("dm", "other_named", "to")),
    "ext.signage": (("flyer_view", "self_named", "author"),),
    "ext.crowd_local": (("arrive", "nearby", ""),
                        ("enter_building", "nearby", ""),
                        ("exit_building", "nearby", "")),
    "ext.encounter": (("arrive", "nearby", ""),
                      ("enter_building", "nearby", ""),
                      ("exit_building", "nearby", ""),
                      ("wake_up", "nearby", "")),
}
NEARBY_KINDS = frozenset(k for rules in CHAIN_RULES.values()
                         for k, how, _ in rules if how == "nearby")
NAMED_KINDS = frozenset(k for rules in CHAIN_RULES.values()
                        for k, how, _ in rules if how != "nearby")


# --------------------------------------------------------------------------- #
# 統計(stdlib だけ。解析環境を選ばない = analyze_g.py と同じ流儀)
# --------------------------------------------------------------------------- #
def _mean(xs) -> float:
    xs = list(xs)
    return math.fsum(xs) / len(xs) if xs else 0.0


def _sd(xs) -> float:
    xs = list(xs)
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(max(0.0, math.fsum((x - m) ** 2 for x in xs) / (len(xs) - 1)))


def _q(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    pos = (len(s) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def burstiness(intervals: list[float]) -> dict:
    """Goh & Barabási (2008) の B と Kim & Jo (2016) の有限長補正 A_n。

    B は短い列で系統的に −1 側へ偏る。**補正なしの B だけを報告しない**。
    """
    n = len(intervals)
    m = _mean(intervals)
    if n < 2 or m <= 0.0:
        return {"n": n, "cv": None, "B": None, "A_n": None}
    r = _sd(intervals) / m
    b = (r - 1.0) / (r + 1.0)
    num = math.sqrt(n + 1.0) * r - math.sqrt(n - 1.0)
    den = (math.sqrt(n + 1.0) - 2.0) * r + math.sqrt(n - 1.0)
    return {"n": n, "cv": r, "B": b, "A_n": (num / den) if den != 0.0 else None}


def summarize(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "mean": _mean(xs), "sd": _sd(xs),
            "min": min(xs), "p10": _q(xs, 0.10), "median": _q(xs, 0.50),
            "p90": _q(xs, 0.90), "max": max(xs), **burstiness(xs)}


def histogram(xs: list[float], edges=BIN_EDGES) -> list[dict]:
    bins = [{"lo": edges[i], "hi": (edges[i + 1] if i + 1 < len(edges) else None),
             "n": 0} for i in range(len(edges))]
    for x in xs:
        idx = 0
        for i, lo in enumerate(edges):
            if x >= lo:
                idx = i
        bins[idx]["n"] += 1
    return bins


# --------------------------------------------------------------------------- #
# ③ Kleinberg (2002) 2 状態バースト検出(batched arrivals 版)
# --------------------------------------------------------------------------- #
def kleinberg_bursts(counts: list[int], at_risk: list[int], *,
                     s: float = 2.0, gamma: float = 1.0) -> dict:
    """離散ビンの 2 状態バースト検出。返り値にバースト区間と使ったパラメータを載せる。

    平常状態の発生確率 p₀ = Σd/Σn、バースト状態 p₁ = min(1⁻, s·p₀)。
    ビンごとのコスト σ(q) = −[d·ln p_q + (n−d)·ln(1−p_q)](二項係数は状態に依らないので略)。
    上位状態への遷移コスト τ = γ·ln(T)(下位へは 0)= Kleinberg の定式そのもの。
    """
    T = len(counts)
    out = {"model": "kleinberg2002_2state_batched", "s": float(s),
           "gamma": float(gamma), "n_bins": T, "bursts": []}
    if T < 2:
        out["status"] = "too_few_bins"
        return out
    total_d, total_n = sum(counts), sum(at_risk)
    if total_d <= 0 or total_n <= 0:
        out["status"] = "no_events"
        return out
    p0 = total_d / total_n
    p1 = min(1.0 - 1e-12, p0 * float(s))
    if not (0.0 < p0 < 1.0) or p1 <= p0:
        out["status"] = "degenerate_rates"
        return out
    out.update({"p0": p0, "p1": p1})
    tau = float(gamma) * math.log(float(T))

    def cost(q: int, d: int, n: int) -> float:
        p = p0 if q == 0 else p1
        d = min(int(d), int(n))
        return -(d * math.log(p) + (n - d) * math.log(1.0 - p))

    prev = [cost(0, counts[0], at_risk[0]), cost(1, counts[0], at_risk[0]) + tau]
    back: list[tuple[int, int]] = []
    for t in range(1, T):
        cur = [0.0, 0.0]
        chosen = [0, 0]
        for q in (0, 1):
            best_q, best_c = 0, prev[0] + (tau if q == 1 else 0.0)
            alt = prev[1] + (0.0 if q == 0 else 0.0)
            if alt < best_c:
                best_q, best_c = 1, alt
            chosen[q] = best_q
            cur[q] = best_c + cost(q, counts[t], at_risk[t])
        back.append((chosen[0], chosen[1]))
        prev = cur
    states = [0] * T
    states[T - 1] = 0 if prev[0] <= prev[1] else 1
    for t in range(T - 1, 0, -1):
        states[t - 1] = back[t - 1][states[t]]

    bursts = []
    t = 0
    while t < T:
        if states[t] != 1:
            t += 1
            continue
        j = t
        while j + 1 < T and states[j + 1] == 1:
            j += 1
        weight = math.fsum(cost(0, counts[k], at_risk[k]) - cost(1, counts[k], at_risk[k])
                           for k in range(t, j + 1))
        bursts.append({"start_bin": t, "end_bin": j, "n_bins": j - t + 1,
                       "n_events": sum(counts[t:j + 1]), "weight": weight})
        t = j + 1
    out["status"] = "ok"
    out["bursts"] = bursts
    out["n_burst_bins"] = sum(b["n_bins"] for b in bursts)
    return out


# --------------------------------------------------------------------------- #
# 読み込み
# --------------------------------------------------------------------------- #
COG_KINDS = ("cog_fire", "cog_event")
CAUSE_KINDS = tuple(sorted(NEARBY_KINDS | NAMED_KINDS))
WANTED = frozenset(COG_KINDS) | frozenset(CAUSE_KINDS) | {"watch_spec"}


def load_events(run_dir: Path) -> dict:
    """L1 から `WANTED` の 3 種族(cog / 原因 / watch_spec)だけを読み出す。

    W4-D: 旧実装は `pq.read_table(...).to_pylist()` で **7 列 × 全行**を Python の
    list にしてから `if kind not in WANTED: continue` で捨てていた。捨てる行の
    Python オブジェクト生成費が全体の支配項だったので、`l1_stream.iter_columns`
    の **Arrow レベル kind 絞り込み**に置き換える。

    O(1) 化するもの / 残るもの(正直な注記):
      - 捨てる行のオブジェクト生成 … **消える**(pyarrow の filter が先に効く)。
      - `cog` / `causes` / `watch` の蓄積 … **残る**。この解析は「発火 1 件ごとに
        直前 step の原因候補を引く」ものなので、対象行そのものは全期間ぶん要る。
      - 生 payload の JSON 復号は `WANTED` の行にしか起きない。

    返り値・行順・`cog` のソートキーは 1 バイトも変えていない(payload の
    `json.loads` も**旧実装の逐語**=壊れた JSON は従来どおり例外になる)。"""
    import l1_stream as ls
    if not ls.l1_paths(run_dir):
        raise SystemExit(f"L1 が無い: {Path(run_dir) / 'l1_events.parquet'}")
    cog: list[dict] = []
    causes: dict[tuple[int, str], list[dict]] = {}
    watch: list[dict] = []
    for cols in ls.iter_columns(run_dir, ["step", "sim_min", "agent_id", "kind",
                                          "x", "y", "payload"], kinds=WANTED):
        for i, kind in enumerate(cols["kind"]):
            blob = cols["payload"][i]
            rec = json.loads(blob) if blob else {}
            row = {"step": int(cols["step"][i]), "sim_min": int(cols["sim_min"][i]),
                   "agent_id": int(cols["agent_id"][i]), "kind": kind,
                   "x": float(cols["x"][i]), "y": float(cols["y"][i]), "p": rec}
            if kind in COG_KINDS:
                cog.append(row)
            elif kind == "watch_spec":
                watch.append(row)
            else:
                causes.setdefault((row["step"], kind), []).append(row)
    cog.sort(key=lambda r: (r["sim_min"], r["agent_id"], r["kind"]))
    return {"cog": cog, "causes": causes, "watch": watch}


def load_run_meta(run_dir: Path) -> dict:
    """ランの設定・来歴(max_contrib / usable チャンネル / dt / g_update の有無)。"""
    from omegaconf import OmegaConf
    from society.cognition import calib as _calib
    # ★ 絶対パスをそのまま残すと実行環境のユーザー名が成果物に混入する(公開ミラーへ出る)。
    #   リポジトリ内ならルート相対に畳む(cognition/calib.py: rel_path と同一の規則)。
    out = {"dir": _calib.rel_path(run_dir), "max_contrib": None,
           "usable_channels": None, "dt_min": 10, "g_update": None, "watch": None,
           "theta_scale": None}
    cfg_path = Path(run_dir) / "config.yaml"
    if cfg_path.exists():
        cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True) or {}
        cog = (cfg.get("cognition") or {})
        fire = (cog.get("fire") or {})
        out["max_contrib"] = int(fire.get("max_contrib", 3))
        out["theta_scale"] = float(fire.get("theta_scale", 1.0))
        out["fire_enabled"] = bool(fire.get("enabled", False))
        out["g_update"] = bool((cog.get("g_update") or {}).get("enabled", False))
        out["watch"] = bool((cog.get("watch") or {}).get("enabled", False))
        out["dt_min"] = int((cfg.get("run") or {}).get("dt_min", 10))
        out["n_agents_cfg"] = int((cfg.get("run") or {}).get("n_agents", 0))
    man = Path(run_dir) / "run_manifest.json"
    if man.exists():
        try:
            doc = json.loads(man.read_text(encoding="utf-8"))
            fire = ((doc.get("cognition") or {}).get("fire") or {})
            out["usable_channels"] = list(fire.get("usable_channels") or []) or None
        except (OSError, ValueError):
            pass
    return out


def load_g(run_dir: Path) -> dict:
    """`{(step, agent_id): {channel_id: g}}`。サイドカーが無ければ空。"""
    path = Path(run_dir) / "cognition_g.parquet"
    if not path.exists():
        return {}
    table = pq.read_table(path)
    names = list(table.column_names)
    g_cols = [c for c in names if c.startswith("g_") and not c.startswith("g0_")]
    col_to_id = {ch.column: ch.id for ch in CH.CHANNELS}
    out: dict[tuple[int, int], dict[str, float]] = {}
    for row in table.to_pylist():
        key = (int(row["step"]), int(row["agent_id"]))
        out[key] = {col_to_id.get(c[2:], c[2:]): float(row[c]) for c in g_cols}
    return out


# --------------------------------------------------------------------------- #
# ① 思考間隔分布(エージェント別・文脈別)
# --------------------------------------------------------------------------- #
def analyze_intervals(cog: list[dict]) -> dict:
    by_agent: dict[int, list[dict]] = {}
    for row in cog:
        by_agent.setdefault(row["agent_id"], []).append(row)
    all_gaps: list[float] = []
    by_ctx: dict[str, list[float]] = {}
    per_agent: dict[int, dict] = {}
    for aid in sorted(by_agent):
        seq = by_agent[aid]
        gaps = []
        for a, b in zip(seq, seq[1:]):
            gap = float(b["sim_min"] - a["sim_min"])
            if gap <= 0.0:
                continue                       # 同一 step の重複(social + 期限)は数えない
            gaps.append(gap)
            # 文脈は「その待ち時間を過ごした側」= 手前のイベントの ctx
            ctx = str(a["p"].get("ctx", "")) or "unknown"
            by_ctx.setdefault(ctx, []).append(gap)
        all_gaps += gaps
        per_agent[aid] = {"n_events": len(seq), "n_gaps": len(gaps),
                          "mean": _mean(gaps), "median": _q(gaps, 0.5) if gaps else 0.0}
    means = [v["mean"] for v in per_agent.values() if v["n_gaps"] > 0]
    return {
        "overall": summarize(all_gaps),
        "histogram_min": histogram(all_gaps),
        "by_context": {ctx: summarize(v) for ctx, v in sorted(by_ctx.items())},
        "by_agent": {
            "n_agents": len(per_agent),
            "mean_of_agent_means": _mean(means),
            "sd_of_agent_means": _sd(means),
            "min_agent_mean": (min(means) if means else None),
            "max_agent_mean": (max(means) if means else None),
            "per_agent": {str(a): v for a, v in sorted(per_agent.items())},
        },
        "note": "間隔は cog_fire と cog_event の**両方**(= 実際に考えた機会)を並べた列の差分[分]",
    }


# --------------------------------------------------------------------------- #
# ② 顕著性発火の原因種別内訳
# --------------------------------------------------------------------------- #
def analyze_salience_sources(cog: list[dict], meta: dict,
                             trigger_weight: float) -> dict:
    src_of = {ch.id: ch.source for ch in CH.CHANNELS}
    mass_src: dict[str, float] = {}
    mass_ch: dict[str, float] = {}
    top_src: dict[str, int] = {}
    top_ch: dict[str, int] = {}
    n_sal = n_trig = 0
    trig_mass = 0.0
    total_mass = 0.0
    for row in cog:
        if row["kind"] != "cog_fire" or row["p"].get("reason") != FIRE.SALIENCE:
            continue
        n_sal += 1
        contrib = row["p"].get("contrib") or []
        best = None
        for cid, value in contrib:
            v = float(value)
            src = src_of.get(str(cid), "unknown")
            mass_src[src] = mass_src.get(src, 0.0) + v
            mass_ch[str(cid)] = mass_ch.get(str(cid), 0.0) + v
            total_mass += v
            if best is None or v > best[1]:
                best = (str(cid), v, src)
        trigs = row["p"].get("trig") or []
        if trigs:
            n_trig += 1
            tw = float(trigger_weight) * len(trigs)
            trig_mass += tw
            total_mass += tw
            mass_src["trigger"] = mass_src.get("trigger", 0.0) + tw
            if best is None or tw > best[1]:
                best = ("<named_trigger>", tw, "trigger")
        if best is not None:
            top_src[best[2]] = top_src.get(best[2], 0) + 1
            top_ch[best[0]] = top_ch.get(best[0], 0) + 1
    complete = (meta.get("max_contrib") is not None
                and meta.get("usable_channels")
                and int(meta["max_contrib"]) >= len(meta["usable_channels"]))
    return {
        "n_salience": n_sal,
        "n_with_named_trigger": n_trig,
        "mass_by_source": {k: v for k, v in sorted(mass_src.items())},
        "share_by_source": ({k: v / total_mass for k, v in sorted(mass_src.items())}
                            if total_mass > 0 else {}),
        "mass_by_channel": {k: v for k, v in sorted(mass_ch.items())},
        "top_contributor_counts_by_source": {k: v for k, v in sorted(top_src.items())},
        "top_contributor_counts_by_channel": {k: v for k, v in sorted(top_ch.items())},
        "trigger_mass": trig_mass,
        "contrib_complete": bool(complete),
        "note": ("source は cognition/channels.py の 3 分類(external=外界 / body=身体 / "
                 "prediction=予測不成立)+ 名前付きトリガ項。contrib_complete=false なら "
                 "上位 max_contrib 件で切られており内訳は下方に偏る"),
    }


# --------------------------------------------------------------------------- #
# ③ 発火数の時系列 + バースト検出 + 同時発火
# --------------------------------------------------------------------------- #
def analyze_timeline(cog: list[dict], *, s: float, gamma: float) -> dict:
    steps = sorted({row["step"] for row in cog})
    if not steps:
        return {"status": "no_events"}
    lo, hi = steps[0], steps[-1]
    span = list(range(lo, hi + 1))
    idx = {st: i for i, st in enumerate(span)}
    n_sal = [0] * len(span)
    n_all = [0] * len(span)
    present: list[set] = [set() for _ in span]
    sal_agents: dict[int, list[int]] = {}
    for row in cog:
        i = idx[row["step"]]
        present[i].add(row["agent_id"])
        n_all[i] += 1
        if row["kind"] == "cog_fire" and row["p"].get("reason") == FIRE.SALIENCE:
            n_sal[i] += 1
            sal_agents.setdefault(row["step"], []).append(row["agent_id"])
    at_risk = [max(1, len(s_)) for s_ in present]
    bursts = kleinberg_bursts(n_sal, at_risk, s=s, gamma=gamma)
    for b in bursts.get("bursts", []):
        b["start_step"] = span[b["start_bin"]]
        b["end_step"] = span[b["end_bin"]]
        b["agents"] = sorted({a for st in range(b["start_step"], b["end_step"] + 1)
                              for a in sal_agents.get(st, [])})
        b["n_agents"] = len(b["agents"])
    cofire = [v for v in n_sal if v > 0]
    top = sorted(((n_sal[i], span[i]) for i in range(len(span)) if n_sal[i] > 0),
                 key=lambda t: (-t[0], t[1]))[:10]
    return {
        "status": "ok",
        "step_first": lo, "step_last": hi,
        "n_salience": sum(n_sal), "n_events": sum(n_all),
        "per_step": [{"step": span[i], "n_all": n_all[i], "n_salience": n_sal[i],
                      "n_present": len(present[i])} for i in range(len(span))],
        "bursts": bursts,
        "cofire": {
            "summary": summarize([float(v) for v in cofire]),
            "max_simultaneous": (max(n_sal) if n_sal else 0),
            "top_steps": [{"step": st, "n_salience": n} for n, st in top],
            "note": "同時発火 = 同一 step に salience 発火した人数(= 集合的注意の指標)",
        },
        "at_risk_note": ("Kleinberg の母数 n_t は『その step に認知イベントを持った"
                         "相異なるエージェント数』= 在場人数の代理。pool ローテーションで"
                         "在場が変わるランでは代理でしかない"),
    }


# --------------------------------------------------------------------------- #
# ④ 発火連鎖グラフ(候補経路 + 確度)
# --------------------------------------------------------------------------- #
def analyze_chains(cog: list[dict], causes: dict, *, lag: int, radius_m: float,
                   min_contrib: float, max_examples: int,
                   all_reasons: bool = False) -> dict:
    src_of = {ch.id: ch.source for ch in CH.CHANNELS}
    edges: list[dict] = []
    n_fire = 0
    n_selforigin = 0
    n_nocause = 0
    for row in cog:
        if row["kind"] != "cog_fire":
            continue
        reason = row["p"].get("reason")
        if not all_reasons and reason != FIRE.SALIENCE:
            continue
        n_fire += 1
        b = row["agent_id"]
        s_total = float(row["p"].get("s", 0.0)) or 1.0
        cause_step = int(row["step"]) - int(lag)
        found = False
        for cid, value in (row["p"].get("contrib") or []):
            cid = str(cid)
            v = float(value)
            if v < float(min_contrib):
                continue
            rules = CHAIN_RULES.get(cid)
            if not rules:
                continue                     # body.* / pred.* は他者由来の経路を持たない
            for kind, how, key in rules:
                pool = causes.get((cause_step, kind)) or []
                if not pool:
                    continue
                if how == "self_named":
                    for ev in pool:
                        if ev["agent_id"] != b:
                            continue
                        a = ev["p"].get(key)
                        if a is None or int(a) == b:
                            continue
                        edges.append(_edge(row, int(a), b, cid, v, s_total, kind,
                                           "high", 1, cause_step, src_of))
                        found = True
                elif how == "other_named":
                    for ev in pool:
                        target = ev["p"].get(key)
                        if target is None or int(target) != b or ev["agent_id"] == b:
                            continue
                        edges.append(_edge(row, int(ev["agent_id"]), b, cid, v,
                                           s_total, kind, "high", 1, cause_step,
                                           src_of))
                        found = True
                else:                        # nearby: 近接からの推定
                    near = [ev for ev in pool
                            if ev["agent_id"] != b
                            and math.dist((ev["x"], ev["y"]),
                                          (row["x"], row["y"])) <= float(radius_m)]
                    conf = "medium" if len(near) == 1 else "low"
                    for ev in near:
                        edges.append(_edge(row, int(ev["agent_id"]), b, cid, v,
                                           s_total, kind, conf, len(near),
                                           cause_step, src_of))
                        found = True
        if not found:
            top = (row["p"].get("contrib") or [[None, 0.0]])[0][0]
            if top is not None and src_of.get(str(top)) in ("body", "prediction"):
                n_selforigin += 1
            else:
                n_nocause += 1

    by_conf: dict[str, int] = {}
    by_channel: dict[str, int] = {}
    pair: dict[tuple[int, int], int] = {}
    for e in edges:
        by_conf[e["confidence"]] = by_conf.get(e["confidence"], 0) + 1
        by_channel[e["channel"]] = by_channel.get(e["channel"], 0) + 1
        pair[(e["from"], e["to"])] = pair.get((e["from"], e["to"]), 0) + 1
    order = {"high": 0, "medium": 1, "low": 2}
    examples = sorted(edges, key=lambda e: (order.get(e["confidence"], 9),
                                            -e["contrib"], e["step"], e["from"],
                                            e["to"]))[:int(max_examples)]
    return {
        "n_fire_examined": n_fire,
        "n_candidate_edges": len(edges),
        "n_fire_self_origin": n_selforigin,
        "n_fire_no_candidate": n_nocause,
        "lag_steps": int(lag), "radius_m": float(radius_m),
        "min_contrib_sigma": float(min_contrib),
        "by_confidence": {k: by_conf.get(k, 0) for k in ("high", "medium", "low")},
        "by_channel": {k: v for k, v in sorted(by_channel.items())},
        "n_distinct_pairs": len(pair),
        "top_pairs": [{"from": a, "to": b, "n": n} for (a, b), n in
                      sorted(pair.items(), key=lambda t: (-t[1], t[0]))[:10]],
        "examples": examples,
        "edges": edges,
        "rules": {cid: [list(r) for r in rules] for cid, rules in
                  sorted(CHAIN_RULES.items())},
        "caveat": ("**候補経路であって因果の証明ではない**。ô が既に織り込んでいれば"
                   "誤差が出ない(偽陰性)し、同時刻・同場所に居ただけの他人も挙がる"
                   "(偽陽性)。件数を『A が B を発火させた回数』と読んではならない。"),
    }


def _edge(fire_row: dict, a: int, b: int, cid: str, value: float, s_total: float,
          via: str, confidence: str, n_cand: int, cause_step: int,
          src_of: dict) -> dict:
    return {
        "step": int(fire_row["step"]), "sim_min": int(fire_row["sim_min"]),
        "cause_step": int(cause_step), "from": int(a), "to": int(b),
        "channel": cid, "source": src_of.get(cid, "unknown"),
        "contrib": float(value), "share_of_S": float(value) / float(s_total),
        "s": float(fire_row["p"].get("s", 0.0)),
        "theta": float(fire_row["p"].get("theta", 0.0)),
        "ctx": str(fire_row["p"].get("ctx", "")),
        "via": via, "confidence": confidence, "n_candidates": int(n_cand),
    }


# --------------------------------------------------------------------------- #
# ⑤ 予測的中率(チャンネル別)
# --------------------------------------------------------------------------- #
def analyze_prediction(cog: list[dict], watch: list[dict], meta: dict,
                       g_table: dict, *, hit_sigma: float) -> dict:
    usable = list(meta.get("usable_channels") or [])
    complete = bool(meta.get("max_contrib") is not None and usable
                    and int(meta["max_contrib"]) >= len(usable))
    if meta.get("g_update") and not g_table:
        g_source = "unavailable(g_update ON だがサイドカーが無い= |o−ô|/σ は復元不能)"
    elif g_table:
        g_source = "cognition_g.parquet の 1 step 前の行(発火判定が読む g)"
    else:
        g_source = "fixed_1.0(g_update OFF なので contrib がそのまま |o−ô|/σ)"
    recoverable = not (meta.get("g_update") and not g_table)

    errs: dict[str, list[float]] = {cid: [] for cid in usable}
    n_events = 0
    for row in cog:
        if row["kind"] != "cog_fire":
            continue
        n_events += 1
        gmap = g_table.get((int(row["step"]) - 1, int(row["agent_id"])))
        if gmap is None:
            gmap = g_table.get((int(row["step"]), int(row["agent_id"])))
        seen = set()
        for cid, value in (row["p"].get("contrib") or []):
            cid = str(cid)
            g = float((gmap or {}).get(cid, 1.0)) if recoverable else 1.0
            if g <= 0.0:
                continue
            errs.setdefault(cid, []).append(float(value) / g)
            seen.add(cid)
        if complete:                # 上位で切られていない = 挙がらなかった channel は誤差 0
            for cid in usable:
                if cid not in seen:
                    errs.setdefault(cid, []).append(0.0)

    by_channel = {}
    for cid in sorted(errs):
        xs = errs[cid]
        if not xs:
            continue
        by_channel[cid] = {
            "n": len(xs), "mean_err_sigma": _mean(xs), "median_err_sigma": _q(xs, 0.5),
            "p90_err_sigma": _q(xs, 0.9), "max_err_sigma": max(xs),
            "hit_rate": sum(1 for x in xs if x <= float(hit_sigma)) / len(xs),
        }
    status: dict[str, int] = {}
    clamped = 0
    n_expect: list[float] = []
    n_trigger: list[float] = []
    for row in watch:
        st = str(row["p"].get("status", ""))
        status[st] = status.get(st, 0) + 1
        clamped += int(row["p"].get("clamped", 0) or 0)
        if "n_expect" in row["p"]:
            n_expect.append(float(row["p"]["n_expect"]))
        if "n_trigger" in row["p"]:
            n_trigger.append(float(row["p"]["n_trigger"]))
    n_watch = len(watch)
    return {
        "hit_sigma": float(hit_sigma),
        "n_cog_fire": n_events,
        "contrib_complete": complete,
        "g_source": g_source,
        "err_recoverable": recoverable,
        "by_channel": by_channel,
        "watch_spec": {
            "n": n_watch, "by_status": {k: v for k, v in sorted(status.items())},
            "compliance_rate": (status.get("ok", 0) / n_watch) if n_watch else None,
            "n_clamped_values": clamped,
            "mean_n_expect": _mean(n_expect) if n_expect else None,
            "mean_n_trigger": _mean(n_trigger) if n_trigger else None,
        },
        "note": ("的中率 = |o−ô|/σ <= hit_sigma の割合。contrib_complete=false のときは"
                 "『上位寄与に入った』条件つきの分布なので**過小な的中率**になる"),
    }


# --------------------------------------------------------------------------- #
# 出力
# --------------------------------------------------------------------------- #
def _f(value, fmt: str = ".3g", none: str = "n/a") -> str:
    """None 安全な数値フォーマット(表の中で f-string を入れ子にしないため)。"""
    if value is None:
        return none
    return format(float(value), fmt)


def _pct(value, fmt: str = ".1f", none: str = "n/a") -> str:
    if value is None:
        return none
    return format(float(value) * 100.0, fmt) + "%"


def render(res: dict) -> str:
    meta = res["run"]
    iv, sal = res["intervals"], res["salience_sources"]
    tl, ch, pr = res["timeline"], res["chains"], res["prediction"]
    out = ["# 思考頻度の観測(P5 / 設計 §8)", "",
           f"ラン: `{meta['dir']}`  /  Δt={meta['dt_min']} 分 / "
           f"theta_scale={meta['theta_scale']} / "
           f"watch={'ON' if meta.get('watch') else 'OFF'} / "
           f"g_update={'ON' if meta.get('g_update') else 'OFF'}", ""]

    o = iv["overall"]
    agg = iv["by_agent"]
    out += ["## ① 思考間隔分布(エージェント別・文脈別)", ""]
    if o.get("n"):
        out += [f"- 全体: n={o['n']} / 平均 {_f(o['mean'])} 分 / 中央 {_f(o['median'])} / "
                f"p10 {_f(o['p10'])} / p90 {_f(o['p90'])} / 最大 {_f(o['max'])}",
                f"- ばらつき: CV={_f(o['cv'])} / バースト性 B={_f(o['B'])} "
                f"(Goh & Barabási 2008) / 有限長補正 A_n={_f(o['A_n'])} "
                f"(Kim & Jo 2016)",
                f"- 個体間: 平均間隔の平均 {_f(agg['mean_of_agent_means'])} 分 / "
                f"SD {_f(agg['sd_of_agent_means'])} / "
                f"最短 {_f(agg['min_agent_mean'])} / 最長 {_f(agg['max_agent_mean'])}",
                ""]
        out += ["| 文脈 | n | 平均[分] | 中央 | p90 | CV | B |",
                "|---|---|---|---|---|---|---|"]
        for ctx, cell in iv["by_context"].items():
            if not cell.get("n"):
                continue
            out.append(f"| {ctx} | {cell['n']} | {_f(cell['mean'])} | "
                       f"{_f(cell['median'])} | {_f(cell['p90'])} | "
                       f"{_f(cell.get('cv'))} | {_f(cell.get('B'))} |")
        bars = ", ".join(
            f"[{_f(b['lo'], 'g')},{'∞' if b['hi'] is None else _f(b['hi'], 'g')})="
            f"{b['n']}" for b in iv["histogram_min"])
        out += ["", "間隔ヒストグラム[分]: " + bars, ""]
    else:
        out += ["(認知イベントが 1 件も無い)", ""]

    out += ["## ② 顕著性発火の原因種別内訳(設計 §2.4 の 3 源)", "",
            f"- salience 発火 {sal['n_salience']} 件"
            f"(うち名前付きトリガ成立 {sal['n_with_named_trigger']} 件)",
            f"- 寄与内訳が完全か: "
            f"{'完全(max_contrib >= usable)' if sal['contrib_complete'] else '**上位で切られている**'}",
            "", "| 源 | S 寄与の質量 | 割合 | 最大寄与になった発火数 |",
            "|---|---|---|---|"]
    for src, mass in sal["mass_by_source"].items():
        share = sal["share_by_source"].get(src, 0.0)
        top = sal["top_contributor_counts_by_source"].get(src, 0)
        out.append(f"| {src} | {_f(mass, '.4g')} | {_pct(share)} | {top} |")
    out += ["", "チャンネル別の質量: " +
            ", ".join(f"{k}={_f(v)}" for k, v in sal["mass_by_channel"].items()), ""]

    out += ["## ③ 発火数の時系列とバースト(§8「バースト = 集合的注意」)", ""]
    if tl.get("status") == "ok":
        b = tl["bursts"]
        out += [f"- step {tl['step_first']}..{tl['step_last']} / "
                f"salience {tl['n_salience']} 件",
                f"- 同時発火の最大 {tl['cofire']['max_simultaneous']} 人 / "
                f"発火のあった step の平均同時発火数 "
                f"{_f(tl['cofire']['summary'].get('mean'))}",
                f"- Kleinberg({b.get('model')}): status={b.get('status')} / "
                f"p0={_f(b.get('p0'), '.4g')} p1={_f(b.get('p1'), '.4g')} / "
                f"バースト {len(b.get('bursts', []))} 区間", ""]
        if b.get("bursts"):
            out += ["| # | step 範囲 | step 数 | 発火数 | 人数 | 重み |",
                    "|---|---|---|---|---|---|"]
            for i, br in enumerate(sorted(b["bursts"],
                                          key=lambda r: -r["weight"])[:10], 1):
                out.append(f"| {i} | {br['start_step']}..{br['end_step']} | "
                           f"{br['n_bins']} | {br['n_events']} | {br['n_agents']} | "
                           f"{_f(br['weight'], '.4g')} |")
            out.append("")
    else:
        out += [f"(時系列を出せない: {tl.get('status')})", ""]

    out += ["## ④ 発火連鎖グラフ(候補経路)", "",
            f"- 検査した salience 発火 {ch['n_fire_examined']} 件 → 候補辺 "
            f"{ch['n_candidate_edges']} 本 / 相異なる (A→B) 対 {ch['n_distinct_pairs']}",
            f"- 確度: high {ch['by_confidence']['high']} / "
            f"medium {ch['by_confidence']['medium']} / low {ch['by_confidence']['low']}",
            f"- 主寄与が身体・予測由来で外部原因を持たない発火 {ch['n_fire_self_origin']} 件 / "
            f"外界由来だが候補が見つからない発火 {ch['n_fire_no_candidate']} 件",
            f"- 規則: lag={ch['lag_steps']} step(観測凍結の構造から一意)/ "
            f"近接半径 {ch['radius_m']:g} m / 寄与下限 {ch['min_contrib_sigma']:g}σ", ""]
    if ch["examples"]:
        out += ["| step | A → B | チャンネル | 経由 | 寄与[σ] | S 中の割合 | 確度 |",
                "|---|---|---|---|---|---|---|"]
        for e in ch["examples"]:
            out.append(f"| {e['step']} | {e['from']} → {e['to']} | {e['channel']} | "
                       f"{e['via']} | {_f(e['contrib'])} | "
                       f"{_pct(e['share_of_S'], '.0f')} | {e['confidence']} |")
        out.append("")
    out += [f"> {ch['caveat']}", ""]

    out += ["## ⑤ 予測的中率(チャンネル別)", "",
            f"- 母数 cog_fire {pr['n_cog_fire']} 件 / 的中の定義 |o−ô|/σ ≤ "
            f"{pr['hit_sigma']:g}",
            f"- g の出所: {pr['g_source']}",
            f"- 寄与内訳が完全か: {'完全' if pr['contrib_complete'] else '**上位で切られている**'}",
            ""]
    if pr["by_channel"]:
        out += ["| channel | n | 平均誤差[σ] | 中央 | p90 | 的中率 |",
                "|---|---|---|---|---|---|"]
        for cid, cell in pr["by_channel"].items():
            out.append(f"| {cid} | {cell['n']} | {_f(cell['mean_err_sigma'])} | "
                       f"{_f(cell['median_err_sigma'])} | "
                       f"{_f(cell['p90_err_sigma'])} | {_pct(cell['hit_rate'])} |")
        out.append("")
    ws = pr["watch_spec"]
    out += [f"監視仕様の受理: n={ws['n']} / 内訳 {ws['by_status']} / "
            f"遵守率 {_pct(ws['compliance_rate'])} / "
            f"クランプ {ws['n_clamped_values']} 値", ""]

    out += ["## 限界(必ず併記する)", ""] + [f"- {x}" for x in res["limits"]] + [""]
    return "\n".join(out)


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="解析するランの出力ディレクトリ")
    ap.add_argument("--out", default=None, help="出力先(既定: ランのディレクトリ)")
    ap.add_argument("--lag-steps", type=int, default=1,
                    help="発火連鎖の因果ラグ[step](既定 1 = 観測凍結の構造から一意)")
    ap.add_argument("--radius-m", type=float, default=None,
                    help="近接推定の半径[m](既定: ランの world.perception_radius_m)")
    ap.add_argument("--min-contrib", type=float, default=0.5,
                    help="連鎖の主張に必要な最小寄与[σ](既定 0.5)")
    ap.add_argument("--hit-sigma", type=float, default=1.0,
                    help="予測的中とみなす誤差の上限[σ](既定 1.0)")
    ap.add_argument("--burst-s", type=float, default=2.0,
                    help="Kleinberg のバースト状態の倍率 s(既定 2.0)")
    ap.add_argument("--burst-gamma", type=float, default=1.0,
                    help="Kleinberg の状態遷移コスト γ(既定 1.0)")
    ap.add_argument("--max-examples", type=int, default=20,
                    help="発火連鎖の実例として出す本数")
    ap.add_argument("--all-reasons", action="store_true",
                    help="連鎖解析を salience 以外の発火にも広げる(既定は salience のみ)")
    ap.add_argument("--no-edges", action="store_true",
                    help="JSON に全候補辺を載せない(巨大ラン向け)")
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    meta = load_run_meta(run_dir)
    data = load_events(run_dir)
    if not data["cog"]:
        raise SystemExit(
            f"認知イベント(cog_fire / cog_event)が 1 件も無い: {run_dir}\n"
            "  cognition.fire.enabled=true のランが必要。")
    g_table = load_g(run_dir)

    radius = args.radius_m
    trigger_weight = 1.0
    cfg_path = run_dir / "config.yaml"
    if cfg_path.exists():
        from omegaconf import OmegaConf
        cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True) or {}
        if radius is None:
            radius = float((cfg.get("world") or {}).get("perception_radius_m", 40.0))
        trigger_weight = float(((cfg.get("cognition") or {}).get("watch") or {})
                               .get("trigger_weight", 1.0))
    if radius is None:
        radius = 40.0

    res = {
        "schema": SCHEMA,
        "run": meta,
        "intervals": analyze_intervals(data["cog"]),
        "salience_sources": analyze_salience_sources(data["cog"], meta, trigger_weight),
        "timeline": analyze_timeline(data["cog"], s=float(args.burst_s),
                                     gamma=float(args.burst_gamma)),
        "chains": analyze_chains(data["cog"], data["causes"], lag=int(args.lag_steps),
                                 radius_m=float(radius),
                                 min_contrib=float(args.min_contrib),
                                 max_examples=int(args.max_examples),
                                 all_reasons=bool(args.all_reasons)),
        "prediction": analyze_prediction(data["cog"], data["watch"], meta, g_table,
                                         hit_sigma=float(args.hit_sigma)),
        "physics_zones": "not_implemented(P3 未実装のため『物理領域内外の思考頻度の差』は出せない)",
        "limits": [
            "④ は**候補経路**であって因果の証明ではない(偽陰性・偽陽性の両方がある)。",
            "⑤ は contrib(上位 max_contrib 件)由来。完全なのは max_contrib >= usable の"
            "ランだけで、そうでなければ的中率は過小に出る。",
            "バースト性 B は有限標本で負に偏る。A_n(Kim & Jo 2016)を併記してある。",
            "★思考間隔の**分解能は Δt が下限**である。較正テーブルの基本周期が Δt より"
            "短い文脈(既定では talking=2 分 < Δt=10 分)は step 格子に丸められ、間隔が"
            "Δt に張り付いて CV≈0・B≈−1 になる。これは世界の性質ではなく時間刻みの人工物"
            "であり、Δt を下げた条件でしか分離できない。",
            "P5 の「物理領域内外の思考頻度の差」は物理層が未実装のため出せない。",
            "mock バックエンドのランでは watch 仕様も発話量も実 LLM と別物である。",
        ],
    }
    if args.no_edges:
        res["chains"].pop("edges", None)

    out_dir = Path(args.out or run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "firing_report.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n")
    md = render(res)
    (out_dir / "firing_report.md").write_text(md, encoding="utf-8", newline="\n")
    print(md)
    print(f"[written] {out_dir / 'firing_report.json'}")
    print(f"[written] {out_dir / 'firing_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
