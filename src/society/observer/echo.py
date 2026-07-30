"""エコー/自己反復の計測(第70バッチ IDEA①。**観測側のみ**・シム本体は 1 バイトも変えない)。

狙い(docs/research/hackathon1-analysis/ideas-shortlist.md §2 ③)
------------------------------------------------------------
「同じ語が繰り返し出た」が **伝播** なのか **LLM の反復癖** なのかを切り分ける。対策が無いと
造語(coin_label)・誤情報・規範化のすべての伝播数値が「LLM が自分の発話を内省できず微妙な
言い換えで同テーマを連投する」性質の関数になり得る(第1回ハッカソン ebiyama 事象F)。

設計の掟(R1)
-------------
- **世界は歪めない**: 発話の抑制・強制 silent 化は一切しない(ebiyama は類似度フィルタで強制
  silent 化して「深掘りまで消える」副作用を自認した=同じ轍を踏まない)。ここは L1 を読むだけ。
- **プロンプト 1 バイト不変・LLM 呼数不変・乱数ゼロ・L1 イベント追加ゼロ**。
  観測は L2 列(本 module)と解析側(observer/measure.echo_novelty・stream.echo_novelty)で完結する。
- `labeling` の採用判定(adopt_threshold)には**一切影響させない**。除外するのは L2/解析の分子だけ。

L2 列(5 列・**常設**)
----------------------
`speech_diversity` / `speech_pairwise_var`(R12 崩壊検知)と同じ「常設の観測列」として出す。
既定 OFF にしない理由: これらは他の全ての伝播数値の**信頼性の前提**であり、後から
「あのランには echo 列が無い」となると比較ができない。読むだけなので L1・乱数・LLM 呼数・
ゴールデン(tests/data/golden_baseline_l1.json)は 1 バイトも動かない。
`observer.echo.enabled: false` にすれば 5 列とも消える(過去ランとの L2 完全一致が要る場合の逃げ道)。

  echo_max                … 直近 window_steps の中で、ある個体の発話に占める**同一文の最大反復率**
                            (発話 2 件以上の個体のみ対象。全員 1 件以下なら 0.0)
  self_similarity_mean    … 同一話者の**連続発話間** Jaccard(文字 n-gram)の平均
  echo_utterance_rate     … 窓内の発話のうち「エコー」と判定された割合
                            (= 窓内で完全同一文の再出 **または** 直前の自分の発話との類似 ≥ 閾値)
  transmission_novel      … 窓内の transmission のうち「同一話者による同一語の再送出」を除いた件数
  transmission_novel_rate … transmission_novel ÷ 窓内 transmission 総数(0 件なら 0.0)

窓(既定 144 step = 1 日)は **rolling**。事後解析の `measure.echo_novelty` と定義を揃えてある
(同じイベント列に対して同じ判定になる)。

状態と resume
-------------
状態は `sim._echo_state`(deque とカウンタのみ)に閉じ、`engine/checkpoint.py` の runtime へ
中央管理で同梱する(第62バッチ joint 状態と同じ流儀)。`_echo_processed` は **プロセス内 logger
カウンタ由来**なので保存せず、load 側で 0 に戻す(assets/gossip と同流儀)。
これにより mid-day resume でも L2 が straight と一致する。

正直な限界
----------
- `echo_max` は**完全一致**の反復しか捉えない。言い換え反復は `self_similarity_mean` /
  `echo_utterance_rate`(閾値つき Jaccard)が担当するが、閾値は恣意的で感度分析の対象である。
- `transmission_novel` の除外対象は「**同一話者**の再送出」だけ。他者からの再伝播は落とさない
  (=複数人が同じ語を使う本来の伝播は保存される)。
- 判定は文字 n-gram の集合類似で、意味的な言い換え(語彙が全部違うが同じ主張)は捉えられない。
"""
from __future__ import annotations

from collections import Counter, deque

# 発話とみなすイベント種(payload["text"] を持つ 3 種)。L1 の他の種は一切見ない。
UTTERANCE_KINDS = ("speak", "sns_post", "dm")

_DEFAULTS = {
    "enabled": True,        # 常設(false にすると 5 列とも消える=過去ランとの L2 一致用の逃げ道)
    "window_steps": 144,    # rolling 窓(144 = 1 日)
    "ngram": 4,             # 文字 n-gram の n(自己類似度の粒度)
    "sim_threshold": 0.6,   # 直前の自分の発話との Jaccard がこれ以上ならエコー扱い
    "min_utterances": 2,    # echo_max の対象にする最小発話数(1 件で 1.0 になる退化を防ぐ)
}
_INT_KEYS = ("window_steps", "ngram", "min_utterances")


# --------------------------------------------------------------------------- #
# 純ヘルパ(stdlib のみ・完全決定論)
# --------------------------------------------------------------------------- #
def ngram_set(text: str, n: int = 4) -> set:
    """文字 n-gram の集合。n 未満の短文は全体を 1 要素にする(空文字は空集合)。"""
    if not text:
        return set()
    if len(text) < n:
        return {text}
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def jaccard(a: set, b: set) -> float:
    """Jaccard 類似度(0..1)。両方空なら 0.0(比較材料なし=類似とみなさない)。"""
    if not a or not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


# --------------------------------------------------------------------------- #
# cfg 正準化(lens.cfg_of / aggregate._llm_health_on と同型: 取得不能でも落ちない)
# --------------------------------------------------------------------------- #
def cfg_of(sim) -> dict:
    cfg = dict(_DEFAULTS)
    block = None
    try:
        obs = sim.cfg.observer
        block = obs.get("echo", None) if hasattr(obs, "get") else None
    except Exception:
        block = None
    if block is not None:
        for key in _DEFAULTS:
            try:
                val = block.get(key, None) if hasattr(block, "get") else None
            except Exception:
                val = None
            if val is None:
                continue
            if key == "enabled":
                cfg[key] = bool(val)
            elif key in _INT_KEYS:
                cfg[key] = int(val)
            else:
                cfg[key] = float(val)
    cfg["window_steps"] = max(1, int(cfg["window_steps"]))
    cfg["ngram"] = max(1, int(cfg["ngram"]))
    cfg["min_utterances"] = max(1, int(cfg["min_utterances"]))
    return cfg


def enabled(sim) -> bool:
    return bool(cfg_of(sim)["enabled"])


# --------------------------------------------------------------------------- #
# 窓つきタリー(すべて deque + カウンタ = 有界メモリ・pickle 可能)
# --------------------------------------------------------------------------- #
def _fresh_state() -> dict:
    return {
        "cur_step": -1,
        # 発話: (step, agent_id, text)。by_agent は agent -> Counter[text](窓内)。
        "utt": deque(),
        "by_agent": {},
        # 直前の自分の発話: agent -> (step, text)。窓外になったら比較に使わない。
        "last_utt": {},
        # 連続発話間の類似: (step, similarity) と合計
        "pair": deque(),
        "pair_sum": 0.0,
        # エコー判定フラグ: (step, 0/1) と合計
        "echo": deque(),
        "echo_n": 0,
        # 伝播: (step, key, novel) / key=(from, item_id) の窓内出現数 / 総数・新規数
        "trans": deque(),
        "emit_count": {},
        "trans_total": 0,
        "trans_novel": 0,
    }


def _prune(st: dict, floor_step: int) -> None:
    """floor_step より前のエントリを窓から落とす(カウンタも同時に戻す)。"""
    utt, by_agent = st["utt"], st["by_agent"]
    while utt and utt[0][0] < floor_step:
        _s, aid, text = utt.popleft()
        c = by_agent.get(aid)
        if c is not None:
            c[text] -= 1
            if c[text] <= 0:
                del c[text]
            if not c:
                del by_agent[aid]
    pair = st["pair"]
    while pair and pair[0][0] < floor_step:
        st["pair_sum"] -= pair.popleft()[1]
    ech = st["echo"]
    while ech and ech[0][0] < floor_step:
        st["echo_n"] -= ech.popleft()[1]
    trans, emit = st["trans"], st["emit_count"]
    while trans and trans[0][0] < floor_step:
        _s, key, novel = trans.popleft()
        n = emit.get(key, 0) - 1
        if n <= 0:
            emit.pop(key, None)
        else:
            emit[key] = n
        st["trans_total"] -= 1
        st["trans_novel"] -= novel


def _add_utterance(st: dict, step: int, agent_id: int, text: str,
                   cfg: dict) -> None:
    n = cfg["ngram"]
    window = cfg["window_steps"]
    grams = ngram_set(text, n)
    prev = st["last_utt"].get(agent_id)
    sim_val = None
    if prev is not None and prev[0] >= step - window + 1:
        sim_val = jaccard(grams, ngram_set(prev[1], n))
        st["pair"].append((step, sim_val))
        st["pair_sum"] += sim_val
    counter = st["by_agent"].setdefault(agent_id, Counter())
    is_echo = bool(counter.get(text, 0) > 0
                   or (sim_val is not None and sim_val >= cfg["sim_threshold"]))
    counter[text] += 1
    st["utt"].append((step, agent_id, text))
    st["echo"].append((step, 1 if is_echo else 0))
    st["echo_n"] += 1 if is_echo else 0
    st["last_utt"][agent_id] = (step, text)


def _add_transmission(st: dict, step: int, frm, item_id) -> None:
    key = (frm, item_id)
    # from < 0(メディア・世界イベント発)は「話者の自己反復」ではないので常に新規扱い。
    self_repeat = (isinstance(frm, int) and frm >= 0
                   and st["emit_count"].get(key, 0) > 0)
    novel = 0 if self_repeat else 1
    st["emit_count"][key] = st["emit_count"].get(key, 0) + 1
    st["trans"].append((step, key, novel))
    st["trans_total"] += 1
    st["trans_novel"] += novel


def _ingest(st: dict, events, cfg: dict) -> None:
    window = cfg["window_steps"]
    for e in events:
        s = int(e.step)
        if s > st["cur_step"]:
            st["cur_step"] = s
            _prune(st, s - window + 1)
        kind = e.kind
        if kind in UTTERANCE_KINDS:
            text = e.payload.get("text")
            if isinstance(text, str) and text.strip():
                aid = int(e.agent_id)
                if aid >= 0:
                    _add_utterance(st, s, aid, text.strip(), cfg)
        elif kind == "transmission":
            iid = e.payload.get("item_id")
            if iid is not None:
                _add_transmission(st, s, e.payload.get("from"), iid)


# --------------------------------------------------------------------------- #
# observe / scalars(L2 用。collect(sim) が step 末に 1 回呼ぶ経路に載る)
# --------------------------------------------------------------------------- #
def observe(sim) -> dict | None:
    """前回処理済み総数以降の新規イベントだけを窓へ取り込み state を返す(OFF は None)。

    lens/structure.observe と同型: idempotent(同 step 内に複数 aggregator が呼んでも
    二度数えない)。flush_segment で buffer が空になっても logger の累計 `_n_flushed` を
    含む「総数」で数えるので欠落・二重計上が無い。決定論・乱数/LLM 呼ゼロ・読むだけ。
    """
    cfg = cfg_of(sim)
    if not cfg["enabled"]:
        return None
    logger = getattr(sim, "logger", None)
    if logger is None:
        return None
    events = logger.events
    total = int(getattr(logger, "_n_flushed", 0)) + len(events)   # 単調増加
    processed = int(getattr(sim, "_echo_processed", 0))
    new = total - processed
    if new < 0:
        new = 0
    if new > len(events):          # flush 前に処理済みのはず。安全側=buffer 内だけ数える
        new = len(events)
    st = getattr(sim, "_echo_state", None)
    if st is None:
        st = _fresh_state()
        sim._echo_state = st
    if new:
        _ingest(st, events[len(events) - new:], cfg)
        sim._echo_cache = None
    sim._echo_processed = total
    return st


def _echo_max(st: dict, min_utterances: int) -> float:
    best = 0.0
    for counter in st["by_agent"].values():
        total = sum(counter.values())
        if total < min_utterances:
            continue
        top = max(counter.values())
        rate = top / total
        if rate > best:
            best = rate
    return best


def scalars(sim) -> dict | None:
    """L2 全体スカラー 5 列(OFF は None=列なし=L2 バイト不変)。"""
    cfg = cfg_of(sim)
    if not cfg["enabled"]:
        return None
    st = observe(sim)
    if st is None:
        return None
    cache = getattr(sim, "_echo_cache", None)
    if cache is not None:
        return cache
    n_utt = len(st["utt"])
    n_pair = len(st["pair"])
    out = {
        "echo_max": round(_echo_max(st, cfg["min_utterances"]), 6),
        "self_similarity_mean": round(st["pair_sum"] / n_pair, 6) if n_pair else 0.0,
        "echo_utterance_rate": round(st["echo_n"] / n_utt, 6) if n_utt else 0.0,
        "transmission_novel": int(st["trans_novel"]),
        "transmission_novel_rate": (round(st["trans_novel"] / st["trans_total"], 6)
                                    if st["trans_total"] else 0.0),
    }
    sim._echo_cache = out
    return out


# L2 列名(analyze 側の配線が単一の源から読むため公開する)
COLUMNS = ("echo_max", "self_similarity_mean", "echo_utterance_rate",
           "transmission_novel", "transmission_novel_rate")
