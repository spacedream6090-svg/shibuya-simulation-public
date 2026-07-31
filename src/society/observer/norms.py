"""規範化ステージ検出器(第74バッチ IDEA③。**観測側のみ**・シム本体は 1 バイトも変えない)。

狙い(docs/research/hackathon1-analysis/ideas-shortlist.md §2 ②)
----------------------------------------------------------------
`coin_label`(自然造語)の **下流** に「その語が規則になったか」を段階で測る。
lunar_simulation の実観測(無冠詞命名 → 複製 → 定冠詞つき参照 → 既存ルール化)を
語ごとの 4 段階として決定論検出し、**命名者(coiner)と制度化者(institutionalizer)を
別役割**として出す。k* が二層化する(発案の k* と制度化の k* が別値)なら論文級。

4 段階(すべて既存 L1 イベントの読み取りだけで判定する)
--------------------------------------------------------
  S1 初出(coin)        … `label_coin` の step。主 = **coiner**
  S2 他者引用(quote)   … coiner **以外**による最初の使用。
                          源は 2 つ:(a) `transmission` の `from != coiner`
                          (b) coiner 以外の発話テキストに語が現れた初出
                          → 早い方を採る(どちらも決定論)
  S3 定冠詞化(definite)… 発話テキストで「〈定冠詞相当マーカー〉+ 語」の形が現れた初出。
                          主 = **definitizer**
  S4 制度化(institution)… 発話テキストで「〈合意参照マーカー〉」と語が **同一発話内に共起**
                          した初出。主 = **institutionalizer**

**マーカー語彙はコードに 1 語も書かない**(envpack 流儀)。`conf/config.yaml` の
`labeling.norm_stage.markers.{definite,agreement}` がデータの単一の源で、本 module の
既定は **空リスト**(= マーカー未設定なら S3/S4 は永久に 0 件)。

順序の掟(飛び段・逆順を記録しない)
------------------------------------
段階 n は「段階 n-1 が既に成立している」ときにしか記録しない。同一発話の中では
S2 → S3 → S4 の順に連鎖判定する(coiner 以外が「例の〈語〉」と言えば、その 1 発話で
S2 と S3 が同時に立つ)。coiner 自身の発話は S2 を立てないが、S2 成立後なら S3/S4 は
立てられる(= 制度化者が命名者と同一人物である場合を測るため)。

R1(既定 OFF の保証)
--------------------
- プロンプト **1 バイト不変**・LLM 呼数不変・乱数ゼロ・L1 イベント追加ゼロ。読むだけ。
- `labeling.norm_stage.enabled: false`(既定)では L2 に **列が出ない**
  (`observer.llm_health` と同じ「OFF 時列なし」方式)。
- 解析側(`measure.norm_stages` / `stream.norm_stages` / `scripts/analyze_norms.py`)は
  conf の enabled に依存せず、**保存済み L1 に対していつでも**同じ判定を再現する。

状態と resume(ON 時のみ)
--------------------------
状態は `sim._norm_state`(dict のみ・pickle 可)に閉じ、`engine/checkpoint.py` の runtime へ
中央管理で同梱する(第62 joint / 第70 echo と同じ流儀)。`_norm_processed` は
**プロセス内 logger カウンタ由来**なので保存せず load 側で 0 に戻す。

正直な限界
----------
- S3/S4 は **表層文字列の一致**でしか判定しない。婉曲な参照・語順の入れ替え・
  言い換えによる制度化は捉えられない(過小検出=下限の推定)。
- mock バックエンドの発話テンプレートには定冠詞相当・合意参照の表現が無いため、
  mock ランでは **S1/S2 までしか立たない**(S3/S4 = 0 件が正常)。実 LLM ラン専用の観測。
- S2 の (b) 経路は部分文字列一致なので、短い造語が他の語の一部に埋まると過検出しうる。
  `min_word_len`(既定 2)で 1 文字語を落としているが、完全な防御ではない。
"""
from __future__ import annotations

# 発話とみなすイベント種(payload["text"] を持つ 3 種)。observer/echo.py と同じ定義。
UTTERANCE_KINDS = ("speak", "sns_post", "dm")

STAGE_NAMES = {0: "none", 1: "coin", 2: "quote", 3: "definite", 4: "institution"}

# ★マーカー語彙は **1 語もコードに書かない**(conf がデータの単一の源)。
_DEFAULTS = {
    "enabled": False,      # L2 2 列(OFF = 列なし = L2 バイト不変)
    "min_word_len": 2,     # 検出対象にする語の最小文字数(1 文字語の部分一致誤爆よけ)
    "max_gap": 2,          # 定冠詞マーカーと語の間に許す文字数(「例の『語』」の 『 を吸収)
}
_INT_KEYS = ("min_word_len", "max_gap")
_MARKER_KEYS = ("definite", "agreement")

# L2 列名(analyze 側の配線が単一の源から読むため公開する)
COLUMNS = ("norm_stage_max", "norm_steps_to_agreement")


# --------------------------------------------------------------------------- #
# cfg 正準化(echo.cfg_of と同型: 取得不能でも落ちない)
# --------------------------------------------------------------------------- #
def _as_str_list(val) -> list[str]:
    if val is None:
        return []
    try:
        return [str(v) for v in val if str(v)]
    except TypeError:
        return []


def markers_from_block(block) -> dict[str, list[str]]:
    """conf の `labeling.norm_stage` ブロック → {definite: [...], agreement: [...]}。"""
    out = {k: [] for k in _MARKER_KEYS}
    if block is None:
        return out
    try:
        mk = block.get("markers", None) if hasattr(block, "get") else None
    except Exception:                              # noqa: BLE001
        mk = None
    if mk is None:
        return out
    for key in _MARKER_KEYS:
        try:
            out[key] = _as_str_list(mk.get(key, None) if hasattr(mk, "get") else None)
        except Exception:                          # noqa: BLE001
            out[key] = []
    return out


def cfg_from_block(block) -> dict:
    """conf ブロック(または None)→ 正準 cfg。conf 非搭載でも既定へ落ちる。"""
    cfg = dict(_DEFAULTS)
    if block is not None:
        for key in _DEFAULTS:
            try:
                val = block.get(key, None) if hasattr(block, "get") else None
            except Exception:                      # noqa: BLE001
                val = None
            if val is None:
                continue
            cfg[key] = bool(val) if key == "enabled" else int(val)
    cfg["min_word_len"] = max(1, int(cfg["min_word_len"]))
    cfg["max_gap"] = max(0, int(cfg["max_gap"]))
    cfg["markers"] = markers_from_block(block)
    return cfg


def _block_of(cfg_root):
    """resolved config(DictConfig / dict)から labeling.norm_stage を取り出す。"""
    node = cfg_root
    for part in ("labeling", "norm_stage"):
        if node is None:
            return None
        try:
            node = node.get(part, None) if hasattr(node, "get") else None
        except Exception:                          # noqa: BLE001
            return None
    return node


def cfg_of_config(cfg_root) -> dict:
    """resolved config 全体 → 正準 cfg(解析スクリプトが run dir の config.yaml に使う)。"""
    return cfg_from_block(_block_of(cfg_root))


def cfg_of(sim) -> dict:
    return cfg_from_block(_block_of(getattr(sim, "cfg", None)))


def enabled(sim) -> bool:
    return bool(cfg_of(sim)["enabled"])


# --------------------------------------------------------------------------- #
# 純ヘルパ(stdlib のみ・完全決定論)
# --------------------------------------------------------------------------- #
def definite_before(text: str, word: str, markers, max_gap: int = 0) -> bool:
    """`text` の中に「マーカー(+最大 max_gap 文字)+ word」の形があるか。

    日本語の定冠詞相当(「例の◯◯」「いつもの◯◯」)は語の**直前**に置かれるので、
    語の出現位置ごとに直前の文字列がマーカーで終わるかを見る。max_gap は
    「例の『◯◯』」のような括弧 1〜2 文字を吸収するための許容幅。
    """
    if not markers or not word or not text:
        return False
    start = 0
    while True:
        i = text.find(word, start)
        if i < 0:
            return False
        head = text[:i]
        for gap in range(0, int(max_gap) + 1):
            h = head if gap == 0 else head[:len(head) - gap]
            if not h:
                continue
            for mk in markers:
                if mk and h.endswith(mk):
                    return True
        start = i + 1


def agreement_in(text: str, markers) -> bool:
    """合意参照マーカーが発話内に現れるか(語との**共起**判定なので位置は問わない)。"""
    if not markers or not text:
        return False
    return any(mk and mk in text for mk in markers)


def _median(values: list) -> float | None:
    vs = sorted(float(v) for v in values)
    n = len(vs)
    if not n:
        return None
    mid = n // 2
    return vs[mid] if n % 2 else (vs[mid - 1] + vs[mid]) / 2.0


# --------------------------------------------------------------------------- #
# 1 パス集計(runtime L2 / measure / stream が実装を共有する)
# --------------------------------------------------------------------------- #
def accumulator(markers=None, min_word_len: int = 2, max_gap: int = 2) -> dict:
    """規範化ステージの 1 パス集計状態(pickle 可能な素の dict/list のみ)。"""
    md = dict(markers or {})
    return {
        "cfg": {"definite": list(_as_str_list(md.get("definite"))),
                "agreement": list(_as_str_list(md.get("agreement"))),
                "min_word_len": max(1, int(min_word_len)),
                "max_gap": max(0, int(max_gap))},
        "words": {},        # word -> record
        "order": [],        # coin 順(決定論の出力順)
        "by_item": {},      # item_id -> word(transmission から語を引く)
        "max_stage": 0,
    }


def _new_record(word: str, item_id, agent_id, step: int, media: bool) -> dict:
    return {"word": word, "item_id": item_id,
            "coiner": (int(agent_id) if isinstance(agent_id, int) else None),
            "media": bool(media), "stage": 1,
            "s1_step": int(step),
            "s2_step": None, "s2_agent": None, "s2_source": None,
            "s3_step": None, "s3_agent": None,
            "s4_step": None, "s4_agent": None}


def _set_stage2(acc: dict, rec: dict, step: int, agent_id, source: str) -> None:
    rec["s2_step"], rec["s2_agent"], rec["s2_source"] = int(step), agent_id, source
    rec["stage"] = 2
    if acc["max_stage"] < 2:
        acc["max_stage"] = 2


def feed(acc: dict, kind: str, step, agent_id, payload: dict) -> None:
    """1 イベントを集計へ流し込む(step 昇順で呼ぶこと。決定論・副作用は acc のみ)。"""
    cfg = acc["cfg"]
    s = int(step)
    if kind == "label_coin":
        word = payload.get("text")
        if not isinstance(word, str):
            return
        word = word.strip()
        if len(word) < cfg["min_word_len"] or word in acc["words"]:
            return                                   # 最初の coin だけを S1 にする
        rec = _new_record(word, payload.get("item_id"), agent_id, s,
                          bool(payload.get("media")))
        acc["words"][word] = rec
        acc["order"].append(word)
        iid = payload.get("item_id")
        if iid is not None:
            acc["by_item"].setdefault(iid, word)
        if acc["max_stage"] < 1:
            acc["max_stage"] = 1
        return
    if kind == "transmission":
        word = acc["by_item"].get(payload.get("item_id"))
        if word is None:
            return
        rec = acc["words"][word]
        if rec["stage"] != 1:
            return
        frm = payload.get("from")
        if isinstance(frm, int) and frm >= 0 and frm != rec["coiner"]:
            _set_stage2(acc, rec, s, frm, "transmission")
        return
    if kind not in UTTERANCE_KINDS:
        return
    text = payload.get("text")
    if not isinstance(text, str) or not text:
        return
    aid = agent_id if isinstance(agent_id, int) else None
    definite, agreement = cfg["definite"], cfg["agreement"]
    has_agreement = None                             # 1 発話につき 1 回だけ判定する
    for word in acc["order"]:
        rec = acc["words"][word]
        if rec["stage"] >= 4 or word not in text:
            continue
        # S2: coiner 以外による使用(発話テキスト経路)
        if rec["stage"] == 1 and aid is not None and aid >= 0 \
                and aid != rec["coiner"]:
            _set_stage2(acc, rec, s, aid, "utterance")
        # S3: 定冠詞相当マーカー + 語(S2 成立後のみ=飛び段を作らない)
        if rec["stage"] == 2 and definite_before(text, word, definite,
                                                 cfg["max_gap"]):
            rec["s3_step"], rec["s3_agent"], rec["stage"] = s, aid, 3
            if acc["max_stage"] < 3:
                acc["max_stage"] = 3
        # S4: 合意参照マーカーとの共起(S3 成立後のみ)
        if rec["stage"] == 3:
            if has_agreement is None:
                has_agreement = agreement_in(text, agreement)
            if has_agreement:
                rec["s4_step"], rec["s4_agent"], rec["stage"] = s, aid, 4
                if acc["max_stage"] < 4:
                    acc["max_stage"] = 4


def finish(acc: dict) -> dict:
    """集計状態 → {words: [...], summary: {...}}(出力順は coin 順=決定論)。"""
    words = [dict(acc["words"][w]) for w in acc["order"]]
    reached = {n: sum(1 for r in words if r["stage"] >= n) for n in (1, 2, 3, 4)}
    agree_lag = [r["s4_step"] - r["s1_step"] for r in words if r["s4_step"] is not None]
    quote_lag = [r["s2_step"] - r["s1_step"] for r in words if r["s2_step"] is not None]
    s4 = [r for r in words if r["stage"] >= 4]
    distinct = [r for r in s4 if r["s4_agent"] is not None
                and r["s4_agent"] != r["coiner"]]
    return {
        "words": words,
        "summary": {
            "n_words": len(words),
            "n_stage1": reached[1], "n_stage2": reached[2],
            "n_stage3": reached[3], "n_stage4": reached[4],
            "max_stage": int(acc["max_stage"]),
            "steps_to_quote_median": _median(quote_lag),
            "steps_to_agreement_median": _median(agree_lag),
            "n_institutionalized": len(s4),
            "n_institutionalizer_is_other": len(distinct),
            "institutionalizer_distinct_share": (round(len(distinct) / len(s4), 6)
                                                 if s4 else None),
            "coiners": sorted({r["coiner"] for r in words
                               if r["coiner"] is not None and r["coiner"] >= 0}),
            "institutionalizers": sorted({r["s4_agent"] for r in s4
                                          if r["s4_agent"] is not None
                                          and r["s4_agent"] >= 0}),
            "markers": {"definite": list(acc["cfg"]["definite"]),
                        "agreement": list(acc["cfg"]["agreement"])},
            "min_word_len": acc["cfg"]["min_word_len"],
            "max_gap": acc["cfg"]["max_gap"],
        },
    }


def norm_candidates(result: dict, min_stage: int, min_agents: int,
                    usage: dict | None = None) -> list[dict]:
    """規範候補(成立語)とその **成立 step** を返す(Part E(2))。

    成立の判定線は **呼び出し側が明示的に渡す**(事前登録 U-10: 既定値を埋め込まない)。
      min_stage  … 到達を要求する最小ステージ(1..4)
      min_agents … その語を使った **相異なるエージェント数** の下限
      usage      … {word: [(step, agent), ...]}(measure/stream 側が集める使用記録)。
                   None のときはステージ到達だけで判定する(= min_agents を課さない)。
    成立 step = 「ステージ到達 step」と「相異なる利用者が min_agents 人に達した step」の
    **遅い方**(= 両条件が初めて同時に満たされた step)。
    """
    ms = int(min_stage)
    out: list[dict] = []
    for rec in result["words"]:
        if rec["stage"] < ms:
            continue
        stage_step = rec.get(f"s{ms}_step") if ms > 1 else rec["s1_step"]
        if stage_step is None:
            continue
        agent_step = None
        n_users = None
        if usage is not None:
            seen: set = set()
            for st, ag in usage.get(rec["word"], []):
                if ag is None or ag < 0:
                    continue
                seen.add(ag)
                if agent_step is None and len(seen) >= int(min_agents):
                    agent_step = int(st)
            n_users = len(seen)
            if agent_step is None:
                continue
        established = max(int(stage_step), int(agent_step or stage_step))
        out.append({"word": rec["word"], "coiner": rec["coiner"],
                    "stage": rec["stage"], "stage_step": int(stage_step),
                    "n_users": n_users, "agents_step": agent_step,
                    "established_step": established,
                    "institutionalizer": rec["s4_agent"]})
    out.sort(key=lambda r: (r["established_step"], r["word"]))
    return out


# --------------------------------------------------------------------------- #
# runtime(L2 用。既定 OFF = 列なし。lens/echo と同型の idempotent 増分走査)
# --------------------------------------------------------------------------- #
def observe(sim) -> dict | None:
    """前回処理済み総数以降の新規イベントだけを取り込み state を返す(OFF は None)。"""
    cfg = cfg_of(sim)
    if not cfg["enabled"]:
        return None
    logger = getattr(sim, "logger", None)
    if logger is None:
        return None
    events = logger.events
    total = int(getattr(logger, "_n_flushed", 0)) + len(events)   # 単調増加
    processed = int(getattr(sim, "_norm_processed", 0))
    new = total - processed
    if new < 0:
        new = 0
    if new > len(events):
        new = len(events)
    st = getattr(sim, "_norm_state", None)
    if st is None:
        st = accumulator(cfg["markers"], cfg["min_word_len"], cfg["max_gap"])
        sim._norm_state = st
    if new:
        for e in events[len(events) - new:]:
            feed(st, e.kind, e.step, e.agent_id, e.payload)
        sim._norm_cache = None
    sim._norm_processed = total
    return st


def scalars(sim) -> dict | None:
    """L2 2 列(OFF は None=列なし=L2 バイト不変)。"""
    st = observe(sim)
    if st is None:
        return None
    cache = getattr(sim, "_norm_cache", None)
    if cache is not None:
        return cache
    lags = [r["s4_step"] - r["s1_step"] for r in st["words"].values()
            if r["s4_step"] is not None]
    out = {"norm_stage_max": int(st["max_stage"]),
           "norm_steps_to_agreement": _median(lags)}
    sim._norm_cache = out
    return out
