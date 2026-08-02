"""退行シグナル監視(第91バッチ。**観測側のみ**・シム本体は 1 バイトも変えない)。

正典: docs/plans/source/design-discussion-20260802.md **§3**

    「監視すべき退行シグナルは、行動分散の単調減少、訪問地点エントロピーの低下、
      発話の語彙エントロピー低下と n-gram 重複率上昇、発火率の 0 または飽和への張り付きである。」

25万人ランで最も怖いのは落ちることではなく、**終了後に全員が同じ行動へ収束していたと
判明すること**である(§3)。本 module はその 4 群を L2 の rolling 窓列として出し、
`scripts/detect_regression.py` がトレンド検定と分散崩壊診断を行う。

設計の掟(R1)
-------------
- **世界は歪めない**: L1 を読むだけ。プロンプト 1 バイト不変・LLM 呼数不変・乱数ゼロ・
  L1 イベント追加ゼロ。observer/echo.py と完全に同型の観測層である。
- **新しい世界読み取りを増やさない**: `sim.logger.events` だけを見る。`sim.agents` も
  `agent.*` も 1 つも触らない(= 観測がシムを変える経路が構造上存在しない)。
  その代償として「母集団」は **窓内に 1 件でも L1 イベントを出した個体**で定義する
  (下の `reg_act_agents` / 発火率 4 列の分母。正直な限界を参照)。
- **既定 OFF**(`observer.regression.enabled: false`)= **列が 1 つも生えない**
  (第75 dunbar / 第86 day_plan / 第87 engaged と同構造)。ゴールデンは無風。

L2 列(既定 OFF。ON で 9 列 + `cognition.fire` ON なら更に 5 列 = 最大 14 列)
------------------------------------------------------------------------------
すべて **rolling 窓**(既定 144 step = 1 シミュ日)の集計で、窓の定義は 1 つだけ。

① 行動分散(単調減少の検出用)
  `reg_act_between_var`   … 個体ごとの act 分布 p_i(`ACT_KINDS` 上の確率ベクトル)を作り、
                            **次元ごとの個体間分散を次元平均**したもの。
                            分散は **不偏(n−1)** を使う ← 分散崩壊診断の要(§3)。
                            iid な個体からの標本なら期待値は N に依存しない。したがって
                            N を増やして下がるなら、それは統計的平滑化ではなく
                            **モデル由来の均質化**である(scripts/detect_regression.py が判定)。
  `reg_act_entropy_mean`  … 個体内の行動エントロピー H(p_i)[bit] の平均(1 人の行動の広さ)
  `reg_act_agents`        … 窓内に act を 1 件でも出した個体数(= 上の分散の n。診断の横軸)

② 訪問地点エントロピー
  `reg_visit_entropy`     … 窓内 `arrive` の訪問先ノード分布の Shannon エントロピー[bit]
  `reg_visit_nodes`       … 窓内に現れた相異なる訪問先ノード数

③ 発話の語彙(第87 の申し送り対応 = **定型応答を除外**する)
  `reg_vocab_entropy`     … 窓内発話の文字 n-gram(既定 2)の Shannon エントロピー[bit]
  `reg_ngram_repeat_rate` … n-gram の重複率 = 1 − (相異なり数 ÷ 総数)。**上昇が退行**
  `reg_vocab_tokens`      … 窓内の n-gram 総数(上 2 列の分母。独立検算の材料)
  `reg_vocab_excluded`    … 窓内で**語彙系から除外した発話数**(= 定型応答の件数)

  ★除外の根拠(第87バッチ 5.4 の申し送り): engaged モードの定型応答
  `cognition.engaged.TEMPLATE`(単一の固定文)は**機構由来の定数**であって
  LLM が生んだ語彙ではない。除外しないと engaged ON のランだけ
  語彙エントロピーが下がり n-gram 重複率が上がる = 機構が退行シグナルを捏造する。
  除外した件数を列に出すのは「**黙って落とさない**」ため(捨てた量が事後に判る)。

④ 発火率の張り付き(**`cognition.fire` ON のときだけ 5 列生える**)
  個体別の発火率 r_i = 窓内 `cog_fire` 件数 ÷ 窓の実効 step 数。
  `reg_fire_rate_p10` / `reg_fire_rate_p50` / `reg_fire_rate_p90` … 分位(線形補間)
  `reg_fire_zero_frac`  … r_i = 0 の個体割合(**下端への張り付き** = 誰も考えない世界)
  `reg_fire_sat_frac`   … r_i ≥ `fire_sat_per_step`(既定 1.0)の個体割合
                          (**上端への張り付き** = 予算崩壊)

状態と resume
-------------
状態は `sim._regression_state`(**step ごとのバケツ deque + 窓合計**のみ)に閉じ、
`engine/checkpoint.py` の runtime へ中央管理で同梱する(第70 echo / 第75 dunbar と同流儀)。
`_regression_processed` は**プロセス内 logger カウンタ由来**なので保存せず load 側で 0 に戻す。
集合(set)は 1 つも使わない(dict / Counter / deque のみ)= pickle の集合反復順非保存の
影響を受けない = 決定論監査が自明。

メモリ
------
バケツは高々 `window_steps` 個。窓合計 `act_totals` / `fire_totals` / `active_totals` は
**窓内に現れた個体数**に比例する(N=2,500 の縦煙で数 MB 台)。
**横煙(N=25万)では既定 OFF のまま回すこと**(conf/smoke_wide.yaml がそう書いてある)。

正直な限界
----------
1. **母集団は「窓内に L1 イベントを出した個体」**。睡眠中で 1 件も出さない個体は分母から
   落ちる(`reg_fire_zero_frac` は「起きていて発火しない率」に近い)。世界を読まない代償で、
   この定義は L1 だけから完全に再計算できる(= 独立検算が可能)ことと引き換えである。
2. **act は L1 イベント種の固定ホワイトリスト**(`ACT_KINDS`)。「その個体が何をしたか」の
   代理であって行動そのものではない。次元集合を固定するのはランをまたいで分散を比較する
   ためで、ここに種を足すと過去ランと比較できなくなる(足すときは `SCHEMA` を上げる)。
3. **語彙は文字 n-gram**(形態素解析なし)。analyze_specialization.py と同じ流儀で、
   外部辞書に依存せず決定論であることを優先した結果、「語」ではなく「文字の並び」を数える。
4. **rolling 窓は強く自己相関する**(隣接行は窓の 99% を共有する)。列を直接トレンド検定に
   かけると有意性が壊滅的に過大評価される。`scripts/detect_regression.py` は既定で
   **窓幅ぶん間引いてから**検定する(非重複標本)。
5. `reg_visit_*` は `arrive` イベントに依存する。移動が起きないランでは 0 のままになる。
"""
from __future__ import annotations

import math
from collections import Counter, deque

from ..cognition import engaged as _engaged

SCHEMA = 1

# --------------------------------------------------------------------------- #
# 固定の語彙(**ランをまたいで比較するため凍結**。増やすときは SCHEMA を上げる)
# --------------------------------------------------------------------------- #
# 「その個体が何をしたか」の代理にする L1 イベント種。移動の内訳(route_start /
# move_segment)は 1 回の行動が step 数ぶん増殖して分布を潰すので**入れない**。
# 世界イベント(agent_id < 0)・観測専用種(cog_fire / cog_event / worldview 等)も入れない。
ACT_KINDS = (
    "arrive", "stay", "speak", "hear", "sns_post", "sns_read", "dm",
    "news_read", "search", "spend", "media_use", "study", "production",
    "free_action", "reflect", "ride", "enter_building", "exit_building",
    "sleep_start", "wake_up", "event_attend", "flyer_post", "flyer_view",
    "group_join", "proposal_support", "venture_open",
)
_ACT_INDEX = {k: i for i, k in enumerate(ACT_KINDS)}

# 発話とみなすイベント種(observer/echo.py・norms.py・analyze_specialization.py と同一定義)
UTTERANCE_KINDS = ("speak", "sns_post", "dm")

# 発火のイベント種(cognition/fire.py:log_events が出す唯一の種)
FIRE_KIND = "cog_fire"

# トークン化から落とす文字(空白・句読点・括弧・記号)。
# **語彙リテラルは 1 語も置かない**(scripts/analyze_specialization.py と同一集合)。
_DROP = set(" \t\n　、。「」『』()()[]【】,.!?！?…・:;:;\"'‘’“”-—~〜/\\")

_DEFAULTS = {
    "enabled": False,           # 既定 OFF = 列が 1 つも生えない
    "window_steps": 144,        # rolling 窓(144 = 1 シミュ日)
    "ngram": 2,                 # 語彙の文字 n-gram の n(analyze_specialization と同値)
    "exclude_template": True,   # 第87 定型応答を語彙系から除外する(申し送り対応)
    "fire_sat_per_step": 1.0,   # 発火率がこれ以上なら「飽和に張り付き」とみなす
    "min_agents": 2,            # 個体間分散を出す最小個体数(1 人では分散が定義できない)
}
_INT_KEYS = ("window_steps", "ngram", "min_agents")
_BOOL_KEYS = ("enabled", "exclude_template")

# L2 列名(単一の源。aggregate.py と detect_regression.py が両方ここを読む)
BASE_COLUMNS = (
    "reg_act_between_var", "reg_act_entropy_mean", "reg_act_agents",
    "reg_visit_entropy", "reg_visit_nodes",
    "reg_vocab_entropy", "reg_ngram_repeat_rate", "reg_vocab_tokens",
    "reg_vocab_excluded",
)
FIRE_COLUMNS = (
    "reg_fire_rate_p10", "reg_fire_rate_p50", "reg_fire_rate_p90",
    "reg_fire_zero_frac", "reg_fire_sat_frac",
)
COLUMNS = BASE_COLUMNS + FIRE_COLUMNS

# 「下がったら退行」= -1 / 「上がったら退行」= +1(detect_regression.py の判定方向)。
# ここが方向の**単一の源**で、スクリプト側は 1 行も方向を持たない。
REGRESSION_DIRECTION = {
    "reg_act_between_var": -1,      # 行動分散の単調減少(§3 の筆頭)
    "reg_act_entropy_mean": -1,     # 1 人の行動の広さが縮む
    "reg_visit_entropy": -1,        # 訪問地点エントロピーの低下(§3)
    "reg_vocab_entropy": -1,        # 語彙エントロピー低下(§3)
    "reg_ngram_repeat_rate": +1,    # n-gram 重複率上昇(§3)
    "reg_fire_zero_frac": +1,       # 発火率 0 への張り付き(§3)
    "reg_fire_sat_frac": +1,        # 発火率 飽和への張り付き(§3)
}


# --------------------------------------------------------------------------- #
# 純ヘルパ(stdlib のみ・完全決定論)
# --------------------------------------------------------------------------- #
def tokens(text: str, n: int = 2) -> tuple:
    """文字 n-gram のタプル(記号・空白を落としてから切る)。空文字は空タプル。"""
    if not text:
        return ()
    s = "".join(ch for ch in text if ch not in _DROP)
    if not s:
        return ()
    if len(s) < n:
        return (s,)
    return tuple(s[i:i + n] for i in range(len(s) - n + 1))


def entropy_bits(counts) -> float:
    """カウント辞書/イテラブルの Shannon エントロピー[bit]。空なら 0.0。"""
    vals = [float(v) for v in (counts.values() if hasattr(counts, "values") else counts)
            if float(v) > 0.0]
    total = sum(vals)
    if total <= 0.0 or len(vals) <= 1:
        return 0.0
    h = 0.0
    for v in vals:
        p = v / total
        h -= p * math.log2(p)
    return h


def quantile(sorted_vals, q: float) -> float:
    """線形補間の分位(numpy 既定と同じ定義。**入力は昇順ソート済み**であること)。"""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return float(sorted_vals[0])
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return float(sorted_vals[lo]) * (1.0 - frac) + float(sorted_vals[hi]) * frac


def unbiased_var(values) -> float:
    """不偏分散(n−1)。n < 2 は 0.0。

    ★ここが分散崩壊診断の要(§3)。iid 標本なら期待値は n に依存しないので、
      N を増やして下がるなら統計的平滑化ではなく**モデル由来の均質化**である。
    """
    xs = [float(v) for v in values]
    n = len(xs)
    if n < 2:
        return 0.0
    mean = sum(xs) / n
    return sum((x - mean) ** 2 for x in xs) / (n - 1)


def act_dispersion(act_by_agent: dict) -> tuple:
    """個体ごとの act 分布から (次元平均の個体間不偏分散, 個体内エントロピー平均, n)。

    `act_by_agent` = {agent_id: {act_index: count}}。純関数(テストの独立検算に使う)。
    """
    rows = []
    ent_sum = 0.0
    for aid in sorted(act_by_agent):
        counts = act_by_agent[aid]
        total = sum(counts.values())
        if total <= 0:
            continue
        vec = [0.0] * len(ACT_KINDS)
        for idx, c in counts.items():
            vec[int(idx)] = c / total
        rows.append(vec)
        ent_sum += entropy_bits([c for c in counts.values() if c > 0])
    n = len(rows)
    if n == 0:
        return 0.0, 0.0, 0
    dim_var = 0.0
    for j in range(len(ACT_KINDS)):
        dim_var += unbiased_var([r[j] for r in rows])
    return dim_var / len(ACT_KINDS), ent_sum / n, n


# --------------------------------------------------------------------------- #
# cfg 正準化(observer/echo.cfg_of と同型: 取得不能でも落ちない)
# --------------------------------------------------------------------------- #
def cfg_of(sim) -> dict:
    cfg = dict(_DEFAULTS)
    block = None
    try:
        obs = sim.cfg.observer
        block = obs.get("regression", None) if hasattr(obs, "get") else None
    except Exception:                                       # noqa: BLE001
        block = None
    if block is not None:
        for key in _DEFAULTS:
            try:
                val = block.get(key, None) if hasattr(block, "get") else None
            except Exception:                               # noqa: BLE001
                val = None
            if val is None:
                continue
            if key in _BOOL_KEYS:
                cfg[key] = bool(val)
            elif key in _INT_KEYS:
                cfg[key] = int(val)
            else:
                cfg[key] = float(val)
    cfg["window_steps"] = max(1, int(cfg["window_steps"]))
    cfg["ngram"] = max(1, int(cfg["ngram"]))
    cfg["min_agents"] = max(1, int(cfg["min_agents"]))
    return cfg


def enabled(sim) -> bool:
    return bool(cfg_of(sim)["enabled"])


def fire_on(sim) -> bool:
    """発火率 5 列を出すか(= `cognition.fire` が ON か)。取得不能なら False。"""
    try:
        cfg = getattr(sim, "firecfg", None)
        return bool(cfg and cfg["enabled"])
    except Exception:                                       # noqa: BLE001
        return False


def excluded_texts(cfg: dict) -> tuple:
    """語彙系から除外する固定文の集合(第87 の申し送り)。

    engaged の `TEMPLATE` は**単一の固定文**で、機構が撃った回数だけ発話分布に現れる。
    これを語彙エントロピー・n-gram 重複率の分子から外す(件数は列に残す)。
    """
    if not cfg.get("exclude_template", True):
        return ()
    return (_engaged.TEMPLATE,)


# --------------------------------------------------------------------------- #
# 窓つきタリー(step ごとのバケツ deque + 窓合計。有界メモリ・pickle 可能・集合ゼロ)
# --------------------------------------------------------------------------- #
def _fresh_state() -> dict:
    return {
        "schema": SCHEMA,
        "cur_step": -1,
        "first_step": -1,
        "buckets": deque(),        # [{"step", "acts", "visits", "toks", ...}]
        # 窓合計
        "act_totals": {},          # aid -> {act_index: n}
        "visit_totals": Counter(),
        "tok_totals": Counter(),
        "tok_n": 0,
        "utt_n": 0,
        "excl_n": 0,
        "fire_totals": {},         # aid -> n
        "active_totals": {},       # aid -> L1 イベント件数(母集団の定義)
        # ラン累計(summary/provenance 用。窓とは別物)
        "excl_cum": 0,
        "utt_cum": 0,
    }


def _new_bucket(step: int) -> dict:
    return {"step": int(step), "acts": {}, "visits": Counter(),
            "toks": Counter(), "tok_n": 0, "utt_n": 0, "excl_n": 0,
            "fires": {}, "active": {}}


def _sub_nested(totals: dict, part: dict) -> None:
    """{aid: {k: n}} の入れ子カウンタから part を引く(0 になったら削除)。"""
    for aid, inner in part.items():
        row = totals.get(aid)
        if row is None:
            continue
        for k, n in inner.items():
            left = row.get(k, 0) - n
            if left <= 0:
                row.pop(k, None)
            else:
                row[k] = left
        if not row:
            totals.pop(aid, None)


def _sub_flat(totals: dict, part: dict) -> None:
    """{aid: n} から part を引く(0 になったら削除)。"""
    for aid, n in part.items():
        left = totals.get(aid, 0) - n
        if left <= 0:
            totals.pop(aid, None)
        else:
            totals[aid] = left


def _sub_counter(totals: Counter, part: Counter) -> None:
    for k, n in part.items():
        left = totals.get(k, 0) - n
        if left <= 0:
            del totals[k]
        else:
            totals[k] = left


def _prune(st: dict, floor_step: int) -> None:
    """floor_step より前のバケツを窓から落とす(窓合計も同時に戻す)。"""
    buckets = st["buckets"]
    while buckets and buckets[0]["step"] < floor_step:
        b = buckets.popleft()
        _sub_nested(st["act_totals"], b["acts"])
        _sub_counter(st["visit_totals"], b["visits"])
        _sub_counter(st["tok_totals"], b["toks"])
        st["tok_n"] -= b["tok_n"]
        st["utt_n"] -= b["utt_n"]
        st["excl_n"] -= b["excl_n"]
        _sub_flat(st["fire_totals"], b["fires"])
        _sub_flat(st["active_totals"], b["active"])


def _bucket_for(st: dict, step: int, window: int) -> dict:
    """step のバケツを得る(必要なら新設し、窓外を落とす)。step は非減少の前提。"""
    buckets = st["buckets"]
    if buckets and buckets[-1]["step"] == step:
        return buckets[-1]
    b = _new_bucket(step)
    buckets.append(b)
    st["cur_step"] = int(step)
    if st["first_step"] < 0:
        st["first_step"] = int(step)
    _prune(st, step - window + 1)
    return b


def _add_nested(totals: dict, aid: int, key, n: int = 1) -> None:
    row = totals.get(aid)
    if row is None:
        row = {}
        totals[aid] = row
    row[key] = row.get(key, 0) + n


def _add_flat(totals: dict, aid: int, n: int = 1) -> None:
    totals[aid] = totals.get(aid, 0) + n


def _ingest(st: dict, events, cfg: dict) -> None:
    """新規イベントを窓へ取り込む(決定論・乱数ゼロ・読むだけ)。"""
    window = cfg["window_steps"]
    n_gram = cfg["ngram"]
    drop_texts = excluded_texts(cfg)
    for e in events:
        step = int(e.step)
        aid = int(e.agent_id)
        kind = e.kind
        b = _bucket_for(st, step, window)
        if aid >= 0:                                  # 母集団 = 世界イベント以外
            _add_flat(b["active"], aid)
            _add_flat(st["active_totals"], aid)
        idx = _ACT_INDEX.get(kind)
        if idx is not None and aid >= 0:
            _add_nested(b["acts"], aid, idx)
            _add_nested(st["act_totals"], aid, idx)
        if kind == "arrive":
            node = e.payload.get("node")
            if node is not None:
                key = str(node)
                b["visits"][key] += 1
                st["visit_totals"][key] += 1
        elif kind in UTTERANCE_KINDS:
            text = e.payload.get("text")
            if isinstance(text, str) and text.strip():
                txt = text.strip()
                b["utt_n"] += 1
                st["utt_n"] += 1
                st["utt_cum"] += 1
                if txt in drop_texts:                 # 第87 定型応答 = 機構由来の定数
                    b["excl_n"] += 1
                    st["excl_n"] += 1
                    st["excl_cum"] += 1
                else:
                    for tok in tokens(txt, n_gram):
                        b["toks"][tok] += 1
                        st["tok_totals"][tok] += 1
                    n_tok = len(tokens(txt, n_gram))
                    b["tok_n"] += n_tok
                    st["tok_n"] += n_tok
        elif kind == FIRE_KIND and aid >= 0:
            _add_flat(b["fires"], aid)
            _add_flat(st["fire_totals"], aid)


# --------------------------------------------------------------------------- #
# observe / scalars(L2 用。collect(sim) が step 末に 1 回呼ぶ経路に載る)
# --------------------------------------------------------------------------- #
def observe(sim) -> dict | None:
    """前回処理済み総数以降の新規イベントだけを窓へ取り込み state を返す(OFF は None)。

    observer/echo.observe と完全に同型: idempotent(同 step 内に複数回呼んでも二度数えない)。
    flush_segment で buffer が空になっても logger の累計 `_n_flushed` を含む「総数」で
    数えるので欠落・二重計上が無い。決定論・乱数/LLM 呼ゼロ・読むだけ。
    """
    cfg = cfg_of(sim)
    if not cfg["enabled"]:
        return None
    logger = getattr(sim, "logger", None)
    if logger is None:
        return None
    events = logger.events
    total = int(getattr(logger, "_n_flushed", 0)) + len(events)   # 単調増加
    processed = int(getattr(sim, "_regression_processed", 0))
    new = total - processed
    if new < 0:
        new = 0
    if new > len(events):          # flush 前に処理済みのはず。安全側=buffer 内だけ数える
        new = len(events)
    st = getattr(sim, "_regression_state", None)
    if st is None:
        st = _fresh_state()
        sim._regression_state = st
    if new:
        _ingest(st, events[len(events) - new:], cfg)
        sim._regression_cache = None
    sim._regression_processed = total
    return st


def window_span(st: dict, window: int) -> int:
    """窓の実効 step 数(ラン冒頭は窓が満ちていない)。最低 1。"""
    if st["cur_step"] < 0:
        return 1
    span = st["cur_step"] - max(st["first_step"], st["cur_step"] - window + 1) + 1
    return max(1, int(span))


def finalize(st: dict, cfg: dict, with_fire: bool) -> dict:
    """窓合計から L2 の列値を作る純関数(テストの独立検算が同じ入口を使えるように公開)。"""
    var, ent, n_act = act_dispersion(st["act_totals"])
    if n_act < cfg["min_agents"]:
        var = 0.0
    out = {
        "reg_act_between_var": round(var, 9),
        "reg_act_entropy_mean": round(ent, 6),
        "reg_act_agents": int(n_act),
        "reg_visit_entropy": round(entropy_bits(st["visit_totals"]), 6),
        "reg_visit_nodes": int(len(st["visit_totals"])),
        "reg_vocab_entropy": round(entropy_bits(st["tok_totals"]), 6),
        "reg_ngram_repeat_rate": (round(1.0 - len(st["tok_totals"]) / st["tok_n"], 6)
                                  if st["tok_n"] else 0.0),
        "reg_vocab_tokens": int(st["tok_n"]),
        "reg_vocab_excluded": int(st["excl_n"]),
    }
    if not with_fire:
        return out
    span = window_span(st, cfg["window_steps"])
    pop = sorted(st["active_totals"])
    rates = sorted(st["fire_totals"].get(aid, 0) / span for aid in pop)
    n_pop = len(rates)
    sat = float(cfg["fire_sat_per_step"])
    out.update({
        "reg_fire_rate_p10": round(quantile(rates, 0.10), 6),
        "reg_fire_rate_p50": round(quantile(rates, 0.50), 6),
        "reg_fire_rate_p90": round(quantile(rates, 0.90), 6),
        "reg_fire_zero_frac": (round(sum(1 for r in rates if r <= 0.0) / n_pop, 6)
                               if n_pop else 0.0),
        "reg_fire_sat_frac": (round(sum(1 for r in rates if r >= sat) / n_pop, 6)
                              if n_pop else 0.0),
    })
    return out


def scalars(sim) -> dict | None:
    """L2 列(OFF は None=列なし=L2 バイト不変)。fire OFF なら発火率 5 列は出さない。"""
    cfg = cfg_of(sim)
    if not cfg["enabled"]:
        return None
    st = observe(sim)
    if st is None:
        return None
    cache = getattr(sim, "_regression_cache", None)
    if cache is not None:
        return cache
    out = finalize(st, cfg, fire_on(sim))
    sim._regression_cache = out
    return out


def provenance(sim) -> dict | None:
    """run_manifest / summary の `regression` キー(OFF は None=キー自体を出さない)。"""
    cfg = cfg_of(sim)
    if not cfg["enabled"]:
        return None
    st = getattr(sim, "_regression_state", None)
    with_fire = fire_on(sim)
    return {
        "schema": SCHEMA,
        "window_steps": int(cfg["window_steps"]),
        "ngram": int(cfg["ngram"]),
        "exclude_template": bool(cfg["exclude_template"]),
        "fire_sat_per_step": float(cfg["fire_sat_per_step"]),
        "fire_columns": bool(with_fire),
        "columns": list(COLUMNS if with_fire else BASE_COLUMNS),
        "act_kinds": list(ACT_KINDS),
        # 「黙って落とさない」= ランを通して語彙系から外した発話の総数と分母
        "utterances_total": int(st["utt_cum"]) if st else 0,
        "excluded_total": int(st["excl_cum"]) if st else 0,
        "note": ("rolling 窓の観測列。隣接行は窓をほぼ共有するので、"
                 "トレンド検定は窓幅ぶん間引いてから行うこと"
                 "(scripts/detect_regression.py の既定)。"),
    }
