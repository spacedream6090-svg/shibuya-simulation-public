"""資産分布 observable(第59バッチ スライス(a) 2026-07-25。既定 OFF)。

口座 E5(economy.accounts)で money+account が永続ストックになった今、「残高そのもの」の
分布と、格差が固定化するか(資産順位の固着)を正面から測る観測レンズ。合成地位 status の
Gini(status_gini)は既にあるが、それは百分位混合の合成スコアであって残高ではない。ここは
wealth = money + account の生の分布(Gini / 上位10%集中 / 中央値 / 平均)と、資産順位の前日比
Kendall τ を観測する。**観測のみ**=分配・課税・再分配などの介入は一切しない(日常観察方針
=Entry 47)。wealth の合成規則は status.py:151(money + account)と同一の鏡。

  設計正典: devlog Entry 50(3スライス精査)/ closed-world-daily-observation の日常観察方針。
  lens.py(第50バッチ)・deviation.py(第55バッチ)・structure.py(第56バッチ)の作法
  (build_cfg 正準化・cfg_of 遅延キャッシュ・scalars で L2 全体スカラー・DEFAULT を単一の源・
  既定 OFF でバイト一致)をそのまま踏襲する。

--- 設計判断(正直な報告)------------------------------------------------------------------
(1) L2 は **全体スカラーのみ**(deviation/structure と同じ「住民別は L2 に出さない」規律=条件2):
      asset_gini        = wealth の Gini 係数(0=平等..1=集中。aggregate._gini と同式の自前実装)
      asset_top10_share = 上位10%が占める総資産の割合(winner-take-all。status_top10_share と同型)
      asset_median      = wealth の中央値
      asset_mean        = wealth の平均
    これらは毎 step の live 状態(agent.money + agent.account)から決定論算出する(status_gini と
    同じ per-step スナップ。乱数/LLM ゼロ・読むだけ)。住民別の分布ヒスト・上位/下位ドリルダウンは
    ビューアが L3 スナップ(l3_snapshots.parquet の money[+account])から事後計算する(sim⇄viz 疎結合。
    deviation/structure/trust と同型。build_data は再計算だけ=挙動不変)。
(2) 格差の固定化 = **資産順位の前日比 Kendall τ**(asset_rank_tau)を1列追加。
    ★「前日状態をシムに持たない」設計は **不可能**(前日比 τ には前日のランキングが要る)。structure.py は
      τ を事後層(analyze_structure.py)へ回してこれを避けたが、本タスクは asset_rank_tau を **in-sim L2
      列**として要求する。そこで status.py が _status_rank_mobility で既に採っている
      「前日 rank(=前日 wealth ベクトル)を sim に 1 本だけ持つ」方式に合わせる(status.py:174-183 の鏡)。
      当日境界で前日 wealth と現 wealth を共通 id 上で突き合わせ τ-b を出す(measure._kendall_tau と同式・
      自前純Python=engine 経路へ numpy を持ち込まない)。**checkpoint に載せる**(検収補修): 当初は
      status.py の鏡で非搭載としたが、それでは assets ON の resume 初日に τ が初期値へ落ち
      resume==straight が崩れる。B4 の org アキュムレータと同じく checkpoint.py の中央管理へ
      day/prev/tau を保存する(processed はプロセス内 logger カウンタ由来なので保存せず load で 0 に戻す)。
      status._status_prev_rank 側の同種ギャップは既存仕様のまま(本補修のスコープ外・註記のみ)。
    ★τ 未定義(初日 / 共通 id<2)の L2 値 = **1.0**(_TAU_INIT)。理由: L2 parquet は 1 列の型を全 part で
      一致させる必要がある(checkpoint 分割ラン=longrun30 で part を pa.concat_tables する。初日 part が
      all-None だと null 型 vs double 型で結合が壊れる)。float 一定にするため、status.py が day0 の
      mobility を 0.0(=「まだ動いていない」)で初期化したのと同じ流儀で、相関側の「まだ動いていない」=
      1.0(=順位不変)で初期化する。**初日の 1.0 は前日比の観測値ではなく初期値**である旨をダッシュボードの
      注記に明示する(τ 系列チャートは事後 L3 から初日=ギャップで描くので誤読しない)。

--- R1 ドクトリン ---------------------------------------------------------------------------
既定 OFF(lens.assets.enabled=false): observe()/scalars() が即 None・L2 に列を足さず・サイドカーも
  書かない=L1/L2/乱数・ゴールデンがバイト一致で不変。ON でも「読むだけ」= 行動・乱数・LLM 呼を一切
  増やさない。ロジックは observer 層(本 module)+ conf(lens.assets ブロック)に閉じ、engine/cognition/
  world/factors/actions/labeling には資産語を一切書かない(no-fingerprint。engine は write_sidecar を
  1 行呼ぶだけ)。決定論・RNG stream 不要(残高は既定の客観ストックの集計)。pandas/duckdb 不使用。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

# 上位集中度の分位(既定 上位10%)。conf lens.assets.top_pct で上書き。
_DEFAULT_TOP_PCT = 0.1
# 資産順位 τ の初期値(初日 / 共通 id<2 の未定義時。設計判断(2)参照=status.py の day0 初期化の鏡)。
_TAU_INIT = 1.0


# --------------------------------------------------------------------------- #
# cfg 正準化(structure.build_cfg / deviation.build_cfg と同型: dict/OmegaConf 両対応)
# --------------------------------------------------------------------------- #
def _to_plain(raw):
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(raw):
            return OmegaConf.to_container(raw, resolve=True)
    except Exception:
        pass
    return raw


def build_cfg(raw) -> dict:
    """conf の lens.assets ブロックを正準化(既定 OFF=現行挙動と完全同一)。

    enabled= 独立マスター(value4/motives/trust/deviation/structure とは別の observable。lens.enabled には
    依存しない=資産レンズだけ ON にできる)。top_pct= 上位集中度の分位(既定 0.1=上位10%)。"""
    raw = _to_plain(raw) or {}
    raw = dict(raw)
    pct = float(raw.get("top_pct", _DEFAULT_TOP_PCT))
    if not (0.0 < pct < 1.0):
        pct = _DEFAULT_TOP_PCT
    return {"enabled": bool(raw.get("enabled", False)), "top_pct": pct}


def cfg_of(sim) -> dict:
    """assets 設定を返す(初回のみ sim.cfg.lens.assets から遅延構築してキャッシュ)。

    simulation.py は読み取り専用のため cfg は本 module が遅延構築する(lens/deviation/structure.cfg_of と
    同型)。キャッシュ属性 sim.assetscfg は L1/L2/L3/乱数に一切現れない=既定 OFF のバイト一致を壊さない。"""
    c = getattr(sim, "assetscfg", None)
    if c is None:
        raw = None
        try:
            raw = (sim.cfg.get("lens", None) or {}).get("assets", None)
        except Exception:
            raw = None
        c = build_cfg(raw)
        sim.assetscfg = c
    return c


def enabled(sim) -> bool:
    return bool(cfg_of(sim)["enabled"])


# --------------------------------------------------------------------------- #
# 純関数(決定論・sim 非依存)
# --------------------------------------------------------------------------- #
def wealth(agent) -> float:
    """資産ストック = money + account(status.py:151 の合成規則の鏡)。account 無し(口座 OFF)は 0.0。"""
    return float(getattr(agent, "money", 0.0)) + float(getattr(agent, "account", 0.0))


def _gini(values) -> float:
    """ジニ係数(0=完全平等..1=完全集中)。aggregate._gini と同式(numpy 非依存)。負値・合計0は 0.0。"""
    xs = sorted(float(v) for v in values)
    n = len(xs)
    if n == 0:
        return 0.0
    total = sum(xs)
    if total <= 0.0:
        return 0.0
    cum = sum(i * x for i, x in enumerate(xs, start=1))
    return (2.0 * cum) / (n * total) - (n + 1.0) / n


def _median(values) -> float:
    """中央値(偶数個は中央2値の平均)。空なら 0.0。"""
    xs = sorted(float(v) for v in values)
    n = len(xs)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return xs[mid]
    return (xs[mid - 1] + xs[mid]) / 2.0


def _top_share(values, pct: float) -> float:
    """上位 pct 割合が占める総資産の割合(winner-take-all。status_top10_share と同型)。総和≤0 は 0.0。"""
    vals = sorted((float(v) for v in values), reverse=True)
    total = sum(vals)
    if not vals or total <= 0.0:
        return 0.0
    k = max(1, int(len(vals) * pct))
    return sum(vals[:k]) / total


def _tie_pairs(sorted_vals) -> int:
    """**ソート済み**列の「同値ペア」総数 Σ c(c-1)/2(c=同値ランの長さ)。O(n)。

    同値は `==` で判定する(明示二重ループの `da == 0` と同じ述語。-0.0 と 0.0、int と float の
    混在も `==` の意味のまま=旧実装と同じ分類になる)。"""
    total = 0
    run = 1
    for i in range(1, len(sorted_vals)):
        if sorted_vals[i] == sorted_vals[i - 1]:
            run += 1
        else:
            total += run * (run - 1) // 2
            run = 1
    return total + run * (run - 1) // 2


def _inversions(seq) -> int:
    """列の反転数(i<j かつ seq[i] > seq[j] のペア数)をマージソートで数える。O(n log n)。

    ボトムアップ(非再帰)なので n=25 万でも再帰上限に触れない。入力列は破壊しない。"""
    buf = list(seq)
    n = len(buf)
    if n < 2:
        return 0
    tmp = [None] * n
    inv = 0
    width = 1
    while width < n:
        for lo in range(0, n, 2 * width):
            mid = lo + width
            if mid >= n:
                break                      # 右半分が無い=そのまま(整列済み)
            hi = min(lo + 2 * width, n)
            i, j, k = lo, mid, lo
            while i < mid and j < hi:
                if buf[j] < buf[i]:        # 右が小さい = 左の残り(mid-i 個)すべてと反転
                    inv += mid - i
                    tmp[k] = buf[j]
                    j += 1
                else:
                    tmp[k] = buf[i]
                    i += 1
                k += 1
            while i < mid:
                tmp[k] = buf[i]
                i += 1
                k += 1
            while j < hi:
                tmp[k] = buf[j]
                j += 1
                k += 1
            buf[lo:hi] = tmp[lo:hi]
        width *= 2
    return inv


def _kendall_tau(a, b):
    """Kendall τ-b を **O(n log n)** で自作(tie 補正込み。measure._kendall_tau と同値・純Python)。

    長さ<2 は None。片側定数(分母0)は 0.0(measure 仕様に一致)。

    ★なぜ書き換えたか(レーンP A3): 旧実装は明示二重ループ = **O(n²)** で、本選 25 万体では
      1 日境界あたり 312 億回の比較(= ハング級)。τ-b の値は「同値ペア数」と「反転数」だけで
      決まるので、tie を別集計し、残りの concord/discord を**マージソートの反転数**から出す。
      pandas/numpy を持ち込まない方針は据え置き(engine 経路は純Python)。

    ★同値性(旧 O(n²) 実装との一致):
        n0        = 全ペア数 n(n-1)/2
        tie_a     = a が同値のペア数(b がどうであれ数える)      … 旧: `if da == 0: tie_a += 1`
        tie_b     = b が同値のペア数                              … 旧: `if db == 0: tie_b += 1`
        tie_ab    = a も b も同値のペア数(包除の交わり)
        concord+discord = n0 - (tie_a + tie_b - tie_ab)           … 旧: tie でも s>0/s<0 でもない
        discord   = a 昇順(同値 a 内は b 昇順)に並べた b 列の反転数
      並べ替えで a[i]<a[j] のペアだけが残り、同値 a のランの中は b 昇順なのでランの内側からは
      反転が出ない = 反転数はちょうど「a が増えて b が減るペア」= discord に一致する。
      返り値の式(int 差 / float の denom)は旧実装と同一なので浮動小数の丸めまで同じ値になる。

    ★契約の外(旧実装と一致しない入力): NaN/±inf、および |da|×|db| が
      アンダーフローして 0 になるほど極端な差(両側とも ~1e-154 未満)。旧実装は積 `da*db` の
      符号で分類するのでそこだけ分類が変わる。wealth = money + account は有限の実数額なので
      本経路では起こらない(tests/test_assets_tau_scale.py が有限入力での完全一致を固定する)。
    """
    n = len(a)
    if n < 2:
        return None
    pairs = sorted(zip(a, b))                       # a 昇順・同値 a 内は b 昇順
    n0_int = n * (n - 1) // 2
    tie_a = _tie_pairs([p[0] for p in pairs])
    tie_b = _tie_pairs(sorted(b))
    tie_ab = _tie_pairs(pairs)                      # (a,b) ともに同値なペア(包除の交わり)
    discord = _inversions([p[1] for p in pairs])
    concord = n0_int - tie_a - tie_b + tie_ab - discord
    n0 = n * (n - 1) / 2.0
    denom = math.sqrt((n0 - tie_a) * (n0 - tie_b))
    if denom == 0:
        return 0.0
    return (concord - discord) / denom


def rank_tau(cur: dict, prev: dict):
    """現 wealth と前日 wealth の共通 id 上での前日比 Kendall τ(未定義=None)。決定論(id 昇順)。"""
    if not prev:
        return None
    common = sorted(set(cur) & set(prev))
    if len(common) < 2:
        return None
    return _kendall_tau([cur[i] for i in common], [prev[i] for i in common])


def _fresh_state() -> dict:
    return {"day": -1, "prev": None, "tau": None, "processed": 0}


# --------------------------------------------------------------------------- #
# observe(前日比 τ の当日境界更新): 前日 wealth ベクトルを 1 本だけ保持(status.py の鏡)
# --------------------------------------------------------------------------- #
def observe(sim) -> dict | None:
    """当日境界を検出して asset_rank_tau を更新し state を返す(OFF は None)。

    lens/deviation/structure.observe と同型: collect(sim) が step 末に 1 回だけ呼ぶ経路に載り、前回
    処理済み総数 sim._assets_state["processed"] 以降の新規イベントだけを走査して当日(暦日)を進める
    (idempotent)。暦日が進んだら現 wealth をスナップし、前日スナップとの前日比 τ を出す(前日を 1 本だけ
    保持。day/prev/tau は checkpoint.py が保存・復元し processed だけ load で 0 に戻る)。gini/median/mean/top10 は scalars が
    毎 step 現状から出すので本関数は τ の state 管理のみ。決定論・乱数/LLM 呼ゼロ・読むだけ。"""
    cfg = cfg_of(sim)
    if not cfg["enabled"]:
        return None
    logger = getattr(sim, "logger", None)
    if logger is None:
        return None
    st = getattr(sim, "_assets_state", None)
    if st is None:
        st = _fresh_state()
        sim._assets_state = st
    events = logger.events
    total = int(getattr(logger, "_n_flushed", 0)) + len(events)   # ランを通じ単調増加
    processed = int(st.get("processed", 0))
    new = total - processed
    if new < 0:
        new = 0
    if new > len(events):          # flush 前に処理済みのはず。安全側=buffer 内だけ見る
        new = len(events)
    maxday = st["day"]
    for e in events[len(events) - new:]:
        d = int(e.sim_min) // 1440
        if d > maxday:
            maxday = d
    if maxday > st["day"]:          # 暦日が進んだ → 現 wealth をスナップし前日比 τ を出す(境界 1 回)
        cur = {a.id: wealth(a) for a in getattr(sim, "agents", [])}
        st["tau"] = rank_tau(cur, st["prev"])   # 前日が無ければ None(初日=未定義)
        st["prev"] = cur
        st["day"] = maxday
    st["processed"] = total
    return st


def scalars(sim) -> dict | None:
    """observe の state + 現 wealth → L2 全体スカラー 5 列(OFF は None)。住民別は L2 に出さない(条件2)。"""
    cfg = cfg_of(sim)
    if not cfg["enabled"]:
        return None
    st = observe(sim)
    vals = [wealth(a) for a in getattr(sim, "agents", [])]
    tau = (st or {}).get("tau")
    return {
        "asset_gini": round(_gini(vals), 6),                     # 残高の集中度(0=平等..1=集中)
        "asset_top10_share": round(_top_share(vals, cfg["top_pct"]), 6),  # 上位割合の占有率
        "asset_median": round(_median(vals), 4),                 # 残高の中央値
        "asset_mean": round((sum(vals) / len(vals)) if vals else 0.0, 4),  # 残高の平均
        # 前日比 順位 τ(高=順位不変=格差固定)。初日/共通 id<2 は _TAU_INIT(=初期値。設計判断(2))。
        "asset_rank_tau": round(tau if tau is not None else _TAU_INIT, 6),
    }


# --------------------------------------------------------------------------- #
# サイドカー(sim⇄viz 疎結合): 資産タブの表示ゲート + 分位を書き出す(ビューアが読む。deviation と同型)
# --------------------------------------------------------------------------- #
def resolved_maps(cfg: dict) -> dict:
    return {"top_pct": cfg["top_pct"], "tau_init": _TAU_INIT}


def write_sidecar(sim, out_dir) -> None:
    """assets ON のときだけ runs/<name>/assets_map.json を書く(OFF は何もしない=後方互換でバイト同一)。

    このファイルの存在が「資産レンズ ON だったラン」のゲート=ビューアは有る時だけ 💰資産タブを出す
    (旧ランは列なし=タブ非表示=バイト同一)。engine 側(simulation.py)は本関数を 1 行呼ぶだけ=
    資産語は本 module に閉じる(no-fingerprint)。"""
    cfg = cfg_of(sim)
    if not cfg["enabled"]:
        return
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "assets_map.json").write_text(
        json.dumps(resolved_maps(cfg), ensure_ascii=False), encoding="utf-8")
