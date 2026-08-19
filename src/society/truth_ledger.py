"""真偽台帳ミニマル 第73バッチ(Part B。2026-07-31。既定 OFF)。

正典: docs/plans/source/dual-mode-instruments.md Part B /
      docs/plans/dual-mode-observe-verify-plan.md §1.3 訂正3・§2 第73行。

何を解く問題か
--------------
このシミュレーションで最も強い「人工オラクル」は、噂と信念の分岐である。**シミュレータ自身が
「実際に何が起きたか」を知っている**ため、正解が世界の内部に存在する。エージェントの信念が
正解からどれだけ離れたか、誰が確かめに行き、誰が受動的に受け入れたかを、観察ではなく
**距離**として測れる。これは観察ラン中に記録されていなければ事後に作れない。

構成(4 つ)
-----------
1. **真偽台帳(Ground Truth Ledger)**: 世界で実際に起きた事実 fact を、既存の L1 イベント
   から**決定論**で抽出して台帳に積む(新しい乱数 stream は 1 本も足さない)。各 fact は
   ID・発生 step・発生場所(node/POI/座標)・真の値・直接目撃可能条件(半径と時間窓)を持つ。
   fact 種は conf の `beliefs.fact_kinds` マップ = **データ駆動**(コードに固有名詞を書かない)。
2. **信念状態**: agent ごと・fact ごとに 値/確信度/情報源/取得 step/検証済みフラグ/親ノード。
   親ノードがあるので**伝播木**が事後に構成できる。取得経路は
   ① 直接目撃(fact の目撃条件 × 位置ログの決定論判定)
   ② 伝聞(既存の speak/dm イベントに乗る。話題の一致判定 → 決定論規則で伝達)。
3. **検証行動 3 種**(`beliefs.verify_actions` ON のみ): 現場へ行く go / 当事者に聞く ask /
   ネットで裏取り net。`freedom.open_actions` と同型の seam で行動空間に 1 行だけ足す。
   **誘導する文言は書かない**(誰が確かめに行くかが観測対象そのもの)。LLM 呼は 1 本も足さない
   (既存 deliberate 応答のパースで受理するだけ)。
4. **漏洩検査**: 台帳がプロンプトへ混入していないことを 3 点セットで担保する(下記)。

台帳のエージェント不可視(**ここは実装ミスが起きると実験全体が無価値になる**)
--------------------------------------------------------------------------------
- (a) **静的検査**: 本 module は `src/society` 直下に置く。プロンプト構築層である
  `src/society/cognition/` からは **import されない**ことを tests が grep で固定する。
  `cognition/deliberate.py` が受け取るのは `verify_actions: bool` の 1 個だけで、
  fact も belief も渡らない(値も ID も文字列も渡らない)。
- (b) **実行時アサーション(canary)**: 各 fact に一意の canary 文字列を持たせ、全プロンプトが
  通る唯一の関門(`llm/cache.py` の CachedLLM)で `check_prompt()` を通す。canary は共通接頭辞
  `CANARY_PREFIX` を持つので、検査は**接頭辞 1 回の substring 走査**で済む(fact 数に依存しない)。
  台帳が 1 件も無い間は `_ARMED=False` で即 return = 既定 OFF は 1 命令も増えない。
- (c) **canary テスト**: 一意の canary を持つ fact を注入し、第71の llm_journal に残った
  **全プロンプト**に一切現れないことを検証する(tests/test_beliefs.py)。

R1(既定 OFF=バイト一致)
-------------------------
`beliefs.enabled=false` のとき phase() は即 return し、fact/belief/イベントを 0 件、
agent/sim に新フィールドを一切生やさない。`beliefs.verify_actions=false` では
deliberate のヘッダに 1 バイトも足さない。**乱数は 1 個も引かない**(全経路が決定論。
新 stream ゼロ)。**LLM 呼は 1 本も足さない**(k 不変)。ON でもプロンプトに増えるのは
verify_actions の 1 行だけで、その内容は fact に依存しない固定文字列(= 指紋を作らない)。

resume(第62 joint / 第70 echo と同型の注意点)
-----------------------------------------------
sim 側の台帳(`_tl_facts` / `_tl_keys` / `_tl_stats`)は `engine/checkpoint.py` の runtime へ
**中央管理**で同梱する。agent 側の信念(`_fact_beliefs` / `_tl_pending`)は agents pickle に
自然同梱。watermark はプロセス内 logger カウンタ由来なので保存せず load で 0 に戻す
(assets/gossip と同流儀)。

正直な限界
----------
- 値(true value)は [0,1] の 1 次元スカラーで、既定は「報告どおりの状態が成り立つ=1.0」。
  truth_status 5 分類・情報源別減衰などのフル版は本選後(ID-U2 決定)。ここはミニマル。
- 変形(distortion)は**伝聞ホップ数による無情報方向 0.5 へのドリフト**と**確信度減衰**だけ。
  意味的な言い換え・意図的な誇張はモデル化しない(凝った変形モデルは入れない方針)。
- 話題の一致判定は**部分文字列**(場所名・題名)。意味的な言及(場所名を出さずに同じ話をする)
  は捉えられない。逆に地名が頻出する街では過剰一致しうるので、上限
  (`max_facts_per_utterance`)と鮮度(`fact_ttl_steps`)で抑えてある。
- net 検証はインターネットを「信頼できる公開記録」としてモデル化する(誤情報の混入はフル版の担当)。

WIT-1(2026-08-20。**ユーザー承認による凍結解除**)
--------------------------------------------------
本 module は `observer/metrics_spec.py` の `SPEC_FILES` 14 本の 1 つで、通常は 1 バイトも
触らない。WIT-1 は「情報目撃のチャネル×注意ゲート再設計」第 1 段としてユーザーが明示承認した
例外である(正典: docs/plans/witness-channel-attention-plan.md)。入ったのは 2 つだけ:

1. **走査の作り直し(挙動は完全同値)**: `_witness_pass` の判定は現行と 1 件もずれず、
   L1 の並びも 1 バイト変わらない。速いのは走査の仕方だけ(§`_witness_pass` の docstring)。
2. **チャネル×注意ゲート**(`beliefs.channels`。**既定 OFF = 現行挙動とバイト一致**):
   ON のとき fact 種ごとに目撃半径を上書きし、目撃候補へ通過確率 p_notice の
   決定論ハッシュ判定をかける。乱数 stream は 1 本も足さない(blake2b のみ)。
   種の表は conf 側の `beliefs.channels.kinds` = **データ駆動**(コードに固有名詞を書かない)。
   表に載らない種(放送に相当するもの)は上書きもゲートも受けない = 現行のまま。
"""
from __future__ import annotations

import bisect
import hashlib
import math

import numpy as np

from .observer.schema import Event

# --------------------------------------------------------------------------- #
# canary(実行時アサーション)。**接頭辞 1 本**なので検査コストは fact 数に依存しない。
# --------------------------------------------------------------------------- #
CANARY_PREFIX = "⟦tl-canary:"      # ⟦tl-canary:<fact_id>⟧
_ARMED = False                          # canary を 1 個でも登録したか(既定 False=検査ゼロ)
_CANARIES: dict[str, str] = {}          # canary 文字列 → fact_id(報告用)


class LedgerLeak(RuntimeError):
    """真偽台帳がプロンプトへ混入した(= 実験が無価値になる致命的欠陥)。"""


def canary_for(fact_id: str) -> str:
    return f"{CANARY_PREFIX}{fact_id}⟧"


def register_canary(fact_id: str) -> str:
    """fact の canary を登録して返す(以降 check_prompt が武装する)。"""
    global _ARMED
    token = canary_for(fact_id)
    _CANARIES[token] = str(fact_id)
    _ARMED = True
    return token


def armed() -> bool:
    return _ARMED


def reset_canaries() -> None:
    """テスト用: 登録を消して武装解除する(本体からは呼ばない)。"""
    global _ARMED
    _CANARIES.clear()
    _ARMED = False


def check_prompt(prompt: str, *, where: str = "prompt") -> None:
    """プロンプト文字列に台帳由来の canary が混入していないか検査する(ON 時のみ実質動作)。

    全 LLM 呼び出しが通る唯一の関門(CachedLLM.generate / generate_many)から呼ぶ。
    未武装(fact ゼロ・機能 OFF)は 1 命令で return = 既定ランのコストはゼロ。
    """
    if not _ARMED or not prompt:
        return
    if CANARY_PREFIX not in prompt:
        return
    hit = next((f for tok, f in _CANARIES.items() if tok in prompt), "?")
    raise LedgerLeak(
        f"真偽台帳がプロンプトへ漏洩した({where}): fact_id={hit}。"
        " 台帳はエージェントから絶対に読めてはならない"
        "(docs/plans/source/dual-mode-instruments.md Part B)。")


def check_percept(percept, *, where: str = "perception") -> None:
    """`Perception`(第85バッチ P1)にも同じ canary 検査をかける。

    P1 の受け入れ基準「Perception に真偽台帳由来の値が混入しないこと(既存の漏洩検査に
    統合)」。Perception は **1 回の思考が受け取る世界情報の唯一の型**なので、ここが
    塞がっていれば「台帳 → 世界情報 → プロンプト」の経路は構造的に存在しない。

    ★呼び出しは engine 側から行う(cognition/ は台帳 module を import も参照もしない
      = tests/test_beliefs.py の静的検査を維持するため)。Perception 側は
      `text_blob()`(プロンプトに出うる文字列材料の連結)を返すだけ。
    未武装(fact ゼロ・機能 OFF)は 1 命令で return = 既定ランのコストはゼロ。
    """
    if not _ARMED or percept is None:
        return
    check_prompt(percept.text_blob(), where=where)


# --------------------------------------------------------------------------- #
# 設定(gossip.build_cfg と同型: dict / OmegaConf 両対応・dotlist 文字列も型強制)
# --------------------------------------------------------------------------- #
DEFAULTS = {
    "enabled": False,
    "verify_actions": False,        # 「確かめる」を行動空間に足す(プロンプト 1 行・中立提示)
    # ---- 台帳(fact の供給源)。L1 の event kind → fact 仕様(データ駆動)----
    #   topic_key  … 話題トークンに使う payload キー(場所名に加えて 1 個)
    #   value_key  … 真の値に使う payload の数値キー(無ければ定数 value)
    #   value      … 定数の真値(既定 1.0 = 「報告どおりの状態が成り立つ」)
    #   value_scale… value_key を [0,1] に正規化する除数
    #   radius_m   … 目撃半径(<=0 は市域全体=その場に居る全員が目撃しうる)
    #   window     … 目撃可能な時間窓 [step, step+window)
    "fact_kinds": {
        "event_host":   {"topic_key": "title"},      # 組織/集合(node 確定)
        "venture_open": {"topic_key": "name"},       # 経済/出店(node 確定)
        "flyer_post":   {},                          # 情報物の掲示(node 確定)
        "group_found":  {"topic_key": "name"},       # 組織の結成(座標確定)
        "crime":        {"topic_key": "kind"},       # 治安(座標確定)
        "stock_out":    {"topic_key": "poi"},        # 経済(欠品。poi は POI 名)
        "price_change": {"topic_key": "cat", "value_key": "ratio",
                         "value_scale": 2.0},        # 経済(価格急変=真値が連続量)
        "disaster":     {"topic_key": "kind", "radius_m": 0},  # 外生ショック(市域全体)
    },
    # ---- 目撃(直接) ----
    "witness_radius_m": 60.0,       # 発生地点からこの距離以内なら目撃しうる
    "witness_window": 6,            # 発生から何 step 目撃可能か(6 step = 1 時間)
    # ---- 伝聞(hearsay) ----
    "fact_ttl_steps": 288,          # これより古い fact は伝聞に乗らない(鮮度=2 日)
    "max_facts_per_utterance": 2,   # 1 発話で伝わる fact 数の上限(過剰一致の安全弁)
    "max_beliefs_per_agent": 32,    # 1 人が保持する信念数の上限(古い順に捨てる)
    "min_topic_len": 2,             # 話題トークンの最小長(1 文字は雑音)
    "hop_drift": 0.15,              # 1 ホップごとに真値から無情報 0.5 へ寄る率(唯一の変形)
    "conf_decay": 0.8,              # 1 ホップごとの確信度の減衰係数
    "min_conf": 0.05,               # これを下回る確信度の信念は伝えない(噂が消える)
    # ---- 検証行動 ----
    "verify_deadline_steps": 36,    # 現場確認(go)の有効期限(6 時間)
    "net_conf": 0.7,                # ネット裏取りで得られる確信度(公開記録=真値だが伝聞より弱い)
    "ask_conf": 0.9,                # 当事者/目撃者に聞いて得られる確信度
}

# ---- WIT-1: チャネル×注意ゲート(既定 OFF=現行挙動とバイト一致)----
#   enabled   … OFF なら半径の上書きもゲートも 1 度も評価しない
#   ambient_k … 環境注意の 1 step あたり上限(**WIT-2 で使う**。本段では未使用)
#   kinds     … fact 種 → {radius_m, p_notice | p_notice_day/p_notice_night}
#               radius_m を書けば目撃半径を上書き、p_notice を書けば通過確率の判定が付く。
#               **書かない種は上書きもゲートも受けない**(= 現行のまま。放送系はここ)。
_CHANNEL_DEFAULTS = {
    "enabled": False,
    "ambient_k": 2,
    "kinds": {
        "price_change": {"radius_m": 10.0, "p_notice": 0.30},
        "stock_out":    {"radius_m": 10.0, "p_notice": 0.30},
        "flyer_post":   {"radius_m": 15.0, "p_notice": 0.50},
        "venture_open": {"radius_m": 20.0, "p_notice": 0.55},
        "event_host":   {"radius_m": 30.0, "p_notice": 0.55},
        "group_found":  {"radius_m": 25.0, "p_notice": 0.55},
        "crime":        {"radius_m": 40.0, "p_notice_day": 0.55,
                         "p_notice_night": 0.35},
    },
}
DEFAULTS["channels"] = _CHANNEL_DEFAULTS

# 通過確率のキー(昼夜を分ける種は day/night の 2 本を持つ)
_P_KEYS = ("p_notice", "p_notice_day", "p_notice_night")
# 昼帯 [DAY_FROM, DAY_TO) の分 of day(街頭事象の通過率が昼夜で変わる)
_DAY_FROM_MIN = 6 * 60
_DAY_TO_MIN = 18 * 60

_INT_KEYS = ("witness_window", "fact_ttl_steps", "max_facts_per_utterance",
             "max_beliefs_per_agent", "min_topic_len", "verify_deadline_steps")
_FLOAT_KEYS = ("witness_radius_m", "hop_drift", "conf_decay", "min_conf",
               "net_conf", "ask_conf")
_BOOL_KEYS = ("enabled", "verify_actions")

# 発生場所として解釈する payload キー(node id → 地図上の実ノード)
_NODE_KEYS = ("node", "at", "to_node", "dest")
# 場所名(POI 名など。node id でない文字列)として話題トークンに使うキー
_PLACE_NAME_KEYS = ("poi", "place")


def _to_plain(raw):
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(raw):
            return OmegaConf.to_container(raw, resolve=True)
    except Exception:                      # noqa: BLE001
        pass
    return raw


def _channels_cfg(raw) -> dict:
    """`beliefs.channels` を正準化する(fact_kinds と同じデータ駆動の流儀)。

    `kinds` を書いたら**丸ごと置き換え**(fact_kinds と同じ規則)。既知キー以外は捨てる
    = conf の綴り間違いが黙って効くことがない。"""
    raw = dict(_to_plain(raw) or {})
    out = {"enabled": bool(raw.get("enabled", _CHANNEL_DEFAULTS["enabled"])),
           "ambient_k": int(raw.get("ambient_k", _CHANNEL_DEFAULTS["ambient_k"]))}
    spec = _to_plain(raw.get("kinds")) if "kinds" in raw else None
    if spec is None:
        out["kinds"] = {k: dict(v) for k, v in _CHANNEL_DEFAULTS["kinds"].items()}
        return out
    kinds: dict[str, dict] = {}
    for k, v in (spec or {}).items():
        src = dict(_to_plain(v) or {})
        one: dict = {}
        if src.get("radius_m") is not None:
            one["radius_m"] = float(src["radius_m"])
        for pk in _P_KEYS:
            if src.get(pk) is not None:
                one[pk] = float(src[pk])
        kinds[str(k)] = one
    out["kinds"] = kinds
    return out


def build_cfg(raw) -> dict:
    """conf の beliefs ブロックを正準化する(既定 OFF=現行挙動と完全同一)。"""
    raw = dict(_to_plain(raw) or {})
    cfg = dict(DEFAULTS)
    cfg["fact_kinds"] = {k: dict(v) for k, v in DEFAULTS["fact_kinds"].items()}
    cfg["channels"] = _channels_cfg(None)
    for key, val in raw.items():
        if key in _BOOL_KEYS:
            cfg[key] = bool(val)
        elif key == "fact_kinds":
            spec = _to_plain(val) or {}
            cfg["fact_kinds"] = {str(k): dict(_to_plain(v) or {})
                                 for k, v in spec.items()}
        elif key == "channels":
            cfg["channels"] = _channels_cfg(val)
        elif key in _INT_KEYS:
            cfg[key] = int(val)
        elif key in _FLOAT_KEYS:
            cfg[key] = float(val)
    return cfg


def cfg_of(sim) -> dict:
    """beliefs 設定を返す(初回のみ sim.cfg.beliefs から遅延構築してキャッシュ)。

    simulation.py は読み取り専用のため cfg は本 module が遅延構築する(gossip.cfg_of と同型)。
    キャッシュ属性 sim.beliefscfg は L1/L2/L3/乱数に一切現れない。"""
    c = getattr(sim, "beliefscfg", None)
    if c is None:
        try:
            raw = sim.cfg.get("beliefs", None)
        except Exception:                  # noqa: BLE001
            raw = None
        c = build_cfg(raw)
        sim.beliefscfg = c
    return c


def enabled(sim) -> bool:
    return bool(cfg_of(sim)["enabled"])


def verify_actions_on(sim) -> bool:
    cfg = cfg_of(sim)
    return bool(cfg["enabled"] and cfg["verify_actions"])


# --------------------------------------------------------------------------- #
# 状態アクセサ(ON 経路でのみ生やす=OFF はバイト一致)
# --------------------------------------------------------------------------- #
def facts_of(sim) -> dict:
    """fact_id → fact レコード(**エージェント不可視の真偽台帳**)。"""
    d = getattr(sim, "_tl_facts", None)
    if d is None:
        d = {}
        sim._tl_facts = d
    return d


def beliefs_of(agent) -> dict:
    """agent の信念 fact_id → {value, conf, src, from, step, hop, verified}。

    ★ `agent.beliefs`(内省の書き戻し文字列リスト)とは**別物**。名前衝突を避けて
    `_fact_beliefs` に置く。"""
    d = getattr(agent, "_fact_beliefs", None)
    if d is None:
        d = {}
        agent._fact_beliefs = d
    return d


def _stats(sim) -> dict:
    st = getattr(sim, "_tl_stats", None)
    if st is None:
        st = {"facts": 0, "witness": 0, "transmit": 0, "verify_ok": 0,
              "verify_try": 0}
        sim._tl_stats = st
    return st


# --------------------------------------------------------------------------- #
# fact の抽出(既存 L1 イベント → 台帳。決定論・乱数ゼロ)
# --------------------------------------------------------------------------- #
def _poi_by_name(sim) -> dict:
    idx = getattr(sim, "_tl_poi_index", None)
    if idx is None:
        idx = {}
        for p in sorted(sim.city.poi_list, key=lambda q: str(q.get("name") or "")):
            nm = str(p.get("name") or "").strip()
            if nm and nm not in idx:
                idx[nm] = p
        sim._tl_poi_index = idx
    return idx


def _resolve_place(sim, payload: dict) -> tuple[str | None, str | None]:
    """payload から (node, 場所名) を決定論で解決する。解決できなければ (None, None)。"""
    graph = sim.city.graph
    for key in _NODE_KEYS:
        v = payload.get(key)
        if isinstance(v, str) and v and graph.has_node(v):
            return v, sim.city.place_label(v)
    for key in _PLACE_NAME_KEYS:
        v = payload.get(key)
        if not isinstance(v, str) or not v.strip():
            continue
        v = v.strip()
        if graph.has_node(v):
            return v, sim.city.place_label(v)
        poi = _poi_by_name(sim).get(v)
        if poi is not None:
            return poi["node"], v
        return None, v                     # 名前だけ判る(座標は event の x,y を使う)
    return None, None


def _true_value(spec: dict, payload: dict) -> float:
    """fact の真の値 [0,1]。既定は 1.0 =「報告どおりの状態が成り立つ」。"""
    key = spec.get("value_key")
    if key is not None:
        raw = payload.get(key)
        try:
            val = float(raw)
        except (TypeError, ValueError):
            val = None
        if val is not None:
            scale = float(spec.get("value_scale", 1.0)) or 1.0
            return max(0.0, min(1.0, val / scale))
    try:
        return max(0.0, min(1.0, float(spec.get("value", 1.0))))
    except (TypeError, ValueError):
        return 1.0


def _topics(place_name: str | None, spec: dict, payload: dict,
            min_len: int) -> list[str]:
    """話題トークン(部分文字列一致の材料)。場所名 + topic_key の値。決定論・重複なし。"""
    out: list[str] = []
    for cand in (place_name, payload.get(spec.get("topic_key")) if spec.get("topic_key") else None):
        if isinstance(cand, str):
            tok = cand.strip()
            if len(tok) >= min_len and tok not in out:
                out.append(tok[:40])
    return out


def _new_events(sim) -> list:
    """前回処理済み総数以降の**新規 L1 イベント**(gossip._scan_seeds と同型の watermark)。"""
    logger = getattr(sim, "logger", None)
    if logger is None:
        return []
    events = logger.events
    total = int(getattr(logger, "_n_flushed", 0)) + len(events)
    processed = int(getattr(sim, "_tl_watermark", 0))
    new = total - processed
    new = 0 if new < 0 else (len(events) if new > len(events) else new)
    sim._tl_watermark = total
    return events[len(events) - new:] if new else []


def _record_fact(sim, cfg: dict, e, spec: dict, step: int, sim_min: int) -> dict | None:
    """1 イベントを fact として台帳へ積む(重複は積まない)。"""
    facts = facts_of(sim)
    keys = getattr(sim, "_tl_keys", None)
    if keys is None:
        keys = {}
        sim._tl_keys = keys
    key = f"{e.kind}|{e.step}|{e.agent_id}|{round(float(e.x), 2)}|{round(float(e.y), 2)}"
    if key in keys:
        return None
    node, place_name = _resolve_place(sim, e.payload or {})
    if node is not None:
        x, y = sim.city.node_xy(node)
    else:
        x, y = float(e.x), float(e.y)
    radius = float(spec.get("radius_m", cfg["witness_radius_m"]))
    fid = f"f{len(facts):05d}"
    fact = {
        "id": fid,
        "kind": e.kind,
        "step": int(e.step),
        "sim_min": int(e.sim_min),
        "node": node,
        "place": place_name,
        "x": round(float(x), 2),
        "y": round(float(y), 2),
        "value": round(_true_value(spec, e.payload or {}), 6),
        "actor": int(e.agent_id),
        "topics": _topics(place_name, spec, e.payload or {}, int(cfg["min_topic_len"])),
        # 直接目撃可能条件: 半径(<=0 は市域全体)と時間窓
        "radius_m": round(radius, 2),
        "window": int(spec.get("window", cfg["witness_window"])),
        # 実行時アサーション用の一意マーカー(プロンプトに現れたら即例外)
        "canary": register_canary(fid),
    }
    facts[fid] = fact
    keys[key] = fid
    _stats(sim)["facts"] += 1
    return fact


# --------------------------------------------------------------------------- #
# 信念の書き込み(唯一の入口。L1 belief_update を出す)
# --------------------------------------------------------------------------- #
def _log(sim, agent, kind: str, payload: dict, step: int, sim_min: int) -> None:
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=int(agent.id),
                         kind=kind, x=agent.x, y=agent.y, payload=payload))


def _templates(fact: dict, *, value: float, conf: float, src: str, parent: int,
               hop: int, cause: str, verified: str | None,
               step: int) -> tuple[dict, dict]:
    """信念 rec と belief_update payload の雛形。

    同じ (fact, step, 由来) なら中身は全受信者で共通(個体差は Event の agent_id/x/y
    だけ)なので、1 度組んで配るときに複製する = 目撃 pass の書込みが軽くなる。"""
    rec = {"value": round(float(value), 6), "conf": round(float(conf), 6),
           "src": src, "from": int(parent), "step": int(step), "hop": int(hop),
           "verified": verified, "kind": fact["kind"]}
    payload = {"fact": fact["id"], "fact_kind": fact["kind"],
               "value": rec["value"], "conf": rec["conf"], "src": src,
               "from": int(parent), "hop": int(hop), "cause": cause,
               "verified": verified}
    return rec, payload


def _put_belief(sim, cfg: dict, agent, fid: str, rec_t: dict, payload_t: dict,
                step: int, sim_min: int) -> dict:
    """雛形から 1 個体ぶんの信念を書く + L1 belief_update。決定論・乱数ゼロ。

    ★rec も payload も**必ず複製**する: payload は logger.log が setdefault で
      書き足す先なので共有すると 2 人目以降に前の個体の値が残る。"""
    bels = beliefs_of(agent)
    rec = rec_t.copy()
    bels[fid] = rec
    cap = int(cfg["max_beliefs_per_agent"])
    if cap > 0 and len(bels) > cap:            # 古い順(取得 step)に捨てる=有界・決定論
        drop = sorted(bels, key=lambda k: (bels[k]["step"], k))[:len(bels) - cap]
        for k in drop:
            del bels[k]
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=int(agent.id),
                         kind="belief_update", x=agent.x, y=agent.y,
                         payload=payload_t.copy()))
    return rec


def _set_belief(sim, cfg: dict, agent, fact: dict, *, value: float, conf: float,
                src: str, parent: int, hop: int, cause: str,
                verified: str | None, step: int, sim_min: int) -> dict:
    """信念を作る/上書きする + L1 belief_update。決定論・乱数ゼロ。"""
    rec_t, payload_t = _templates(fact, value=value, conf=conf, src=src,
                                  parent=parent, hop=hop, cause=cause,
                                  verified=verified, step=step)
    return _put_belief(sim, cfg, agent, fact["id"], rec_t, payload_t,
                       step, sim_min)


# --------------------------------------------------------------------------- #
# ① 直接目撃(fact の目撃条件 × 位置ログ。決定論)
# --------------------------------------------------------------------------- #
def _present(agent) -> bool:
    return (not getattr(agent, "sleeping", False)
            and getattr(agent, "loc", "") != "outside")


# --- 走査の作り直し(WIT-1)。**帰結集合も並びも現行と 1 件もずれない** ---------- #
# 二乗距離 s = dx*dx + dy*dy と半径二乗 t の比較は、numpy と Python で同じ IEEE 演算
# (引き算・掛け算・足し算)だけを使うのでビット一致する。hypot との判定不一致は境界の
# 相対 1ulp 帯にしか出ないので、確定しない帯だけ**現行式そのまま**で再判定する。
#   ★距離そのものは **math.hypot でしか計算しない**: numpy 側の同名関数は結果が違う入力が
#     実測 17.6% ある(numba#1570)。numexpr / FMA 融合も同じ理由で禁止
#     (docs/research/witness-vectorization-bench.md。tests が機械固定する)。
_BAND = 2.0 ** -48
# 均一グリッドのセル幅 [m]。**速度にしか効かない**(どの幅でも通過集合は同一)。
# 代表半径の 2 倍あたりが平坦な最適域(60-240m で実測差なし)。
_CELL_M = 120.0


def _build_grid(xs, ys, idx, cell: float) -> dict:
    """present な個体だけの均一グリッド。セル内は `idx` の並び(=走査順)を保つ。"""
    gx = np.floor(xs[idx] / cell).astype(np.int64)
    gy = np.floor(ys[idx] / cell).astype(np.int64)
    gx0, gx1 = int(gx.min()), int(gx.max())
    gy0, gy1 = int(gy.min()), int(gy.max())
    ny = gy1 - gy0 + 1
    # 鍵は「占有矩形の中でだけ」一意。照会側も同じ矩形へクランプするので衝突は起きない。
    key = (gx - gx0) * ny + (gy - gy0)
    order = np.argsort(key, kind="stable")     # stable = セル内の並びを崩さない
    ks = key[order]
    head = np.flatnonzero(np.concatenate(([True], ks[1:] != ks[:-1])))
    return {"idx": idx, "order": order, "uniq": ks[head],
            "bound": np.concatenate((head, [len(ks)])).astype(np.int64),
            "cell": float(cell), "gx0": gx0, "gx1": gx1, "gy0": gy0, "gy1": gy1,
            "ny": ny}


def _grid_near(grid: dict, fx: float, fy: float, radius: float):
    """半径円に触れるセルに居る個体の**グローバル添字**(昇順)。無ければ None。

    セル範囲は上下 1 セルぶん広げてある(座標差の丸めで境界のセルを取り落とさない)。
    余計に拾っても後段の厳密判定で落ちるだけ = 帰結は変わらない。"""
    cell = grid["cell"]
    c0x = max(grid["gx0"], int(math.floor((fx - radius) / cell)) - 1)
    c1x = min(grid["gx1"], int(math.floor((fx + radius) / cell)) + 1)
    c0y = max(grid["gy0"], int(math.floor((fy - radius) / cell)) - 1)
    c1y = min(grid["gy1"], int(math.floor((fy + radius) / cell)) + 1)
    if c0x > c1x or c0y > c1y:
        return None
    ny, gx0, gy0 = grid["ny"], grid["gx0"], grid["gy0"]
    keys = [(cx - gx0) * ny + (cy - gy0)
            for cx in range(c0x, c1x + 1) for cy in range(c0y, c1y + 1)]
    uniq, bound, order = grid["uniq"], grid["bound"], grid["order"]
    pos = np.searchsorted(uniq, np.array(keys, dtype=np.int64))
    n_uniq = len(uniq)
    parts = [order[bound[j]:bound[j + 1]]
             for j, k in zip(pos.tolist(), keys)
             if j < n_uniq and int(uniq[j]) == k]
    if not parts:
        return None
    loc = np.concatenate(parts)
    loc.sort(kind="stable")
    return grid["idx"][loc]


def _inside(xs, ys, cand, fx: float, fy: float, radius: float):
    """候補のうち現行式 `math.hypot(dx, dy) <= radius` を満たすものだけ(昇順)。"""
    dx = xs[cand] - fx
    dy = ys[cand] - fy
    s = dx * dx + dy * dy
    t = radius * radius
    ok = s <= t * (1.0 - _BAND)                       # ここは確実に内側
    # 「確実に外側」でもない残り(= 境界帯・非有限)だけ現行式で厳密に決める
    edge = np.flatnonzero(~ok & ~(s >= t * (1.0 + _BAND)))
    for p in edge.tolist():
        j = int(cand[p])
        ok[p] = not (math.hypot(xs[j] - fx, ys[j] - fy) > radius)
    return cand[ok]


def _u01(salt: str, *parts) -> float:
    """(用途 salt, 安定キー) → [0,1)。attention._u01 / economy.stable_unit と同流儀。

    ★他 module から import せず**ここに同型を置く**: 本 module は凍結ハッシュ
      (observer/metrics_spec.py)の対象なので、目撃判定に効く式が凍結の外にあると
      「hash は無風のまま判定だけ変わる」経路ができてしまう。"""
    key = f"{salt}\x1f" + "\x1f".join(str(p) for p in parts)
    h = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") / 2.0 ** 64


_WIT_SALT = "wit"


def _p_notice(spec: dict, sim_min: int) -> float | None:
    """チャネル仕様から通過確率を取る(昼夜を分ける種は分 of day で選ぶ)。"""
    if "p_notice_day" in spec or "p_notice_night" in spec:
        tod = int(sim_min) % 1440
        day = _DAY_FROM_MIN <= tod < _DAY_TO_MIN
        p = spec.get("p_notice_day" if day else "p_notice_night")
        if p is not None:
            return float(p)
    p = spec.get("p_notice")
    return None if p is None else float(p)


def _witness_pass(sim, cfg: dict, step: int, sim_min: int) -> None:
    """目撃窓が開いている fact について、条件を満たす個体へ直接目撃の信念を与える。

    判定は現行と同じ論理積(fid 既知でない ∧ (actor ∨ (present ∧ 半径内)))で、
    並びも現行と同じ(fact 生成順 × `sim.agents` の並び)。変わったのは走査の仕方だけ:
      ・`_present` と座標は fact に依存しないので step 先頭で 1 回だけ読む
        (この pass の中で sleeping/loc/x/y を書き換える経路が無いのが根拠)。
      ・radius<=0(市域全体)は距離計算を 1 回もしない。
      ・radius>0 は均一グリッドで近傍セルだけ見る(結果はセル幅に依存しない)。
      ・「既知なら飛ばす」は候補にだけ効かせる(全員に対して先に見る必要がない
        = 4 条件は互いに独立で、順番を変えても通過集合は同じ)。

    `beliefs.channels.enabled` のときだけ、種ごとの半径上書きと通過確率の判定が乗る。
    抽選のキーに step を**入れない**ので、同じ (fact, 個体) は窓の間に 1 回だけ引かれる
    (較正値が「通過あたりの確率」だから)。当事者(actor)は判定を受けない。
    ★上書きが効くのは**この受動的な目撃だけ**。自分で確かめに行く経路(apply_verify /
      _pending_pass)は fact 自身の半径を使う = 「見に行った人は見える」は変えない。"""
    facts = facts_of(sim)
    if not facts:
        return
    open_ids = [fid for fid, f in facts.items()
                if f["step"] <= step < f["step"] + max(1, int(f["window"]))]
    if not open_ids:
        return
    stats = _stats(sim)
    agents = sim.agents
    present = [i for i, a in enumerate(agents) if _present(a)]
    # WIT-1 以前に作られた cfg(古い checkpoint の cfg キャッシュ)でも落ちないよう既定へ倒す
    chan = cfg.get("channels") or _CHANNEL_DEFAULTS
    chan_on = bool(chan["enabled"])
    chan_kinds = chan["kinds"]
    xs = ys = grid = idx_by_id = None
    for fid in open_ids:                       # 挿入順(= fact 生成順)= 決定論
        fact = facts[fid]
        radius = float(fact["radius_m"])
        p_notice = None
        if chan_on:
            spec = chan_kinds.get(fact["kind"])
            if spec is not None:
                if "radius_m" in spec:
                    radius = float(spec["radius_m"])
                p_notice = _p_notice(spec, sim_min)
        fx, fy = float(fact["x"]), float(fact["y"])
        if radius > 0.0 and present:
            if grid is None:
                n = len(agents)
                xs = np.fromiter((float(a.x) for a in agents),
                                 dtype=np.float64, count=n)
                ys = np.fromiter((float(a.y) for a in agents),
                                 dtype=np.float64, count=n)
                grid = _build_grid(xs, ys, np.array(present, dtype=np.int64),
                                   _CELL_M)
            near = _grid_near(grid, fx, fy, radius)
            cand = [] if near is None else _inside(xs, ys, near, fx, fy,
                                                   radius).tolist()
        elif radius > 0.0:
            cand = []
        else:
            cand = list(present)               # 市域全体 = 距離計算をしない
        actor = int(fact["actor"])
        if actor >= 0:                         # 当事者は present も半径も通り抜ける
            if idx_by_id is None:
                idx_by_id = {int(a.id): i for i, a in enumerate(agents)}
            ai = idx_by_id.get(actor)
            if ai is not None:                 # 昇順を保ったまま差し込む
                at = bisect.bisect_left(cand, ai)
                if at == len(cand) or cand[at] != ai:
                    cand.insert(at, ai)
        if not cand:
            continue
        rec_t, payload_t = _templates(fact, value=fact["value"], conf=1.0,
                                      src="direct", parent=-1, hop=0,
                                      cause="witness", verified="witness",
                                      step=step)
        for i in cand:                         # 昇順 = sim.agents の並び = 決定論
            agent = agents[i]
            # 読むだけの参照は beliefs_of() を使わない(全個体に空 dict を生やさない)
            if fid in (getattr(agent, "_fact_beliefs", None) or ()):
                continue
            if p_notice is not None and int(agent.id) != actor \
                    and _u01(_WIT_SALT, fid, int(agent.id)) >= p_notice:
                continue                       # 注意が向かなかった
            _put_belief(sim, cfg, agent, fid, rec_t, payload_t, step, sim_min)
            stats["witness"] += 1


# --------------------------------------------------------------------------- #
# ② 伝聞(既存の speak / dm イベントに乗る。話題の一致判定 → 決定論規則で伝達)
# --------------------------------------------------------------------------- #
# 伝聞が乗るイベント種 → (受け手キー, チャネル名)。**発話の生成には一切干渉しない**
# (既に L1 に載った発話を読むだけ = LLM 呼数・プロンプトともに不変)。
_UTTERANCE = {"speak": ("hearers", "face"), "dm": ("to", "dm")}


def _matches(text: str, topics: list[str]) -> str | None:
    """話題の一致判定。fact の話題トークンが発話に**部分文字列**として現れるか。

    判定規則は決定論で、入力(発話テキスト)は L1 と llm_journal に残る = journal 等級の根拠。
    最初に一致したトークンを返す(根拠として payload に残す)。"""
    for tok in topics:
        if tok and tok in text:
            return tok
    return None


def _receivers(sim, payload: dict, key: str) -> list:
    raw = payload.get(key)
    ids = raw if isinstance(raw, list) else ([raw] if raw is not None else [])
    out = []
    for rid in ids:
        try:
            other = sim.agent_by_id.get(int(rid))
        except (TypeError, ValueError):
            continue
        if other is not None and _present(other):
            out.append(other)
    return sorted(out, key=lambda a: a.id)     # id 昇順 = 決定論


def _transmit_pass(sim, cfg: dict, utterances: list, step: int,
                   sim_min: int) -> None:
    """この step の発話を走査し、話者が持つ信念のうち話題が一致するものを聞き手へ伝える。"""
    facts = facts_of(sim)
    if not facts or not utterances:
        return
    stats = _stats(sim)
    ttl = int(cfg["fact_ttl_steps"])
    cap = int(cfg["max_facts_per_utterance"])
    drift = float(cfg["hop_drift"])
    decay = float(cfg["conf_decay"])
    floor = float(cfg["min_conf"])
    for e, recv_key, channel in utterances:
        speaker = sim.agent_by_id.get(int(e.agent_id))
        if speaker is None:
            continue
        held = getattr(speaker, "_fact_beliefs", None)      # 読むだけ(生やさない)
        if not held:
            continue
        text = str((e.payload or {}).get("text") or "")
        if not text:
            continue
        receivers = _receivers(sim, e.payload or {}, recv_key)
        if not receivers:
            continue
        # 話題が一致し、かつ確信度が伝達下限を超える信念を、確信度降順→fact_id 昇順で上位のみ
        cands = []
        for fid, bel in held.items():
            fact = facts.get(fid)
            if fact is None or step - fact["step"] > ttl:
                continue
            if float(bel["conf"]) * decay < floor:
                continue
            tok = _matches(text, fact["topics"])
            if tok is None:
                continue
            cands.append((-float(bel["conf"]), fid, tok))
        if not cands:
            continue
        for _negc, fid, tok in sorted(cands)[:max(1, cap)]:
            fact = facts[fid]
            bel = held[fid]
            hop = int(bel["hop"]) + 1
            # 唯一の変形モデル: 1 ホップごとに真値から無情報 0.5 へ寄る(Bartlett 型の劣化)
            value = float(bel["value"]) + (0.5 - float(bel["value"])) * drift
            conf = float(bel["conf"]) * decay
            got = []
            for other in receivers:
                if other.id == speaker.id:
                    continue
                prev = (getattr(other, "_fact_beliefs", None) or {}).get(fid)
                if prev is not None and float(prev["conf"]) >= conf:
                    continue               # 既により確かな信念を持つ=上書きしない
                _set_belief(sim, cfg, other, fact, value=value, conf=conf,
                            src="agent", parent=int(speaker.id), hop=hop,
                            cause="transmit", verified=None,
                            step=step, sim_min=sim_min)
                got.append(int(other.id))
                stats["transmit"] += 1
            if got:
                _log(sim, speaker, "belief_transmit",
                     {"fact": fid, "fact_kind": fact["kind"], "to": got,
                      "channel": channel, "hop": hop, "topic": tok,
                      "conf": round(conf, 6)}, step, sim_min)


# --------------------------------------------------------------------------- #
# ③ 検証行動 3 種(go=現場 / ask=当事者に聞く / net=ネットで裏取り)
# --------------------------------------------------------------------------- #
_HOW_ALIASES = {
    "go": "go", "site": "go", "visit": "go", "現場": "go", "行く": "go",
    "ask": "ask", "person": "ask", "人に聞く": "ask", "聞く": "ask",
    "net": "net", "internet": "net", "search": "net", "ネット": "net",
    "調べる": "net",
}


def normalize_how(raw) -> str:
    """自由記述の how を 3 種へ正準化する(既定 go)。決定論・乱数ゼロ。"""
    text = str(raw or "").strip().lower()
    if not text:
        return "go"
    if text in _HOW_ALIASES:
        return _HOW_ALIASES[text]
    for key, val in _HOW_ALIASES.items():      # 部分一致(プロンプトの括弧つき表記に耐える)
        if key in text:
            return val
    return "go"


def _pick_target(sim, cfg: dict, agent, about: str, step: int):
    """検証対象の fact を決定論で選ぶ。返り値 (fact, match)。

    ① about の文字列が fact の話題トークンと一致するもの(match="topic")
    ② 無ければ**最も新しい未検証の信念**(match="recent")。
    ★ ②は「確かめようとしたが対象が言語的に特定できなかった」ケースを捨てないための後退で、
      検証率を水増ししうる。payload の match で区別できるようにしてある(解析側で除外可能)。"""
    facts = facts_of(sim)
    held = getattr(agent, "_fact_beliefs", None)            # 読むだけ(生やさない)
    if not held:
        return None, None
    text = str(about or "")
    ttl = int(cfg["fact_ttl_steps"])
    live = [(fid, bel) for fid, bel in held.items()
            if fid in facts and step - facts[fid]["step"] <= ttl]
    if text:
        for fid, _bel in sorted(live, key=lambda kv: kv[0]):
            if _matches(text, facts[fid]["topics"]) is not None:
                return facts[fid], "topic"
    unver = [(bel["step"], fid) for fid, bel in live if not bel.get("verified")]
    if unver:
        return facts[max(unver)[1]], "recent"
    return None, None


def _nearby(sim, agent) -> list:
    radius = 30.0
    try:
        radius = float(sim.cfg.world.perception_radius_m)
    except Exception:                          # noqa: BLE001
        pass
    out = [a for a in sim.agents
           if a.id != agent.id and _present(a)
           and math.hypot(a.x - agent.x, a.y - agent.y) <= radius]
    return sorted(out, key=lambda a: a.id)


def _net_on(sim) -> bool:
    try:
        return bool((sim.cfg.get("net", {}) or {}).get("enabled", True))
    except Exception:                          # noqa: BLE001
        return False


def _route_to(sim, agent, node: str) -> bool:
    """現場へ徒歩で向かわせる(_apply_free_action / lodging の経路設定と同型)。

    現実拘束の維持: 就寝中・勤務中・圏外・**建物の中**では動かさない。建物内を除くのは
    engine の exit_building が `agent.node` を入口へ張り替える(= 保持中の route が
    宙に浮く)ためで、ここで route を張ると次の移動フェーズで不整合になる。
    動かせないときは False を返し、呼び手が pending として記録する(= 到着待ちのまま、
    たまたま現場に近づいたら _pending_pass が確定させる)。"""
    if node is None or node == agent.node:
        return False
    if getattr(agent, "sleeping", False) or getattr(agent, "activity", "") == "working":
        return False
    if getattr(agent, "loc", "") != "street" or getattr(agent, "building", None):
        return False
    try:
        path, used_mode = sim.router.route(agent.node, node, "walk")
    except Exception:                          # noqa: BLE001 (経路不能は静かに失敗)
        return False
    if len(path) < 2:
        return False
    agent.route = path[1:]
    agent.edge_offset = 0.0
    agent.dest = node
    agent.trip_mode = used_mode
    # 進行中の「退出/帰宅」意図は上書きする(=確かめに行くことを選んだ)。到着時の滞在は
    # 既存の move_to 経路と同じ既定に揃える(engine の到着処理がこれらを読む)。
    agent.exit_intent = False
    agent.homing = False
    agent._pending_stay = sim.clock.dur_steps(2)
    agent._pending_activity = ""
    agent._ride_pending = None
    return True


def apply_verify(sim, agent, action: dict, step: int, sim_min: int) -> None:
    """検証行動の裁定(scheduler._apply からの唯一の入口)。OFF なら呼ばれない。

    LLM 呼び出しは 1 本も足さない(既存 deliberate 応答のパース結果を受けるだけ)。
    乱数ゼロ・完全決定論。結果は L1 "belief_verify"(根拠つき)+ 成功時 "belief_update"。"""
    cfg = cfg_of(sim)
    if not (cfg["enabled"] and cfg["verify_actions"]):
        return
    stats = _stats(sim)
    stats["verify_try"] += 1
    how = normalize_how(action.get("how"))
    about = str(action.get("about") or "")[:120]
    fact, match = _pick_target(sim, cfg, agent, about, step)
    if fact is None:
        _log(sim, agent, "belief_verify",
             {"fact": None, "how": how, "about": about,
              "outcome": "no_target", "match": None}, step, sim_min)
        return
    fid = fact["id"]
    payload = {"fact": fid, "fact_kind": fact["kind"], "how": how,
               "about": about, "match": match}

    if how == "net":
        if not _net_on(sim):
            payload["outcome"] = "no_channel"
            _log(sim, agent, "belief_verify", payload, step, sim_min)
            return
        _set_belief(sim, cfg, agent, fact, value=fact["value"],
                    conf=float(cfg["net_conf"]), src="net", parent=-1, hop=0,
                    cause="verify_net", verified="net", step=step, sim_min=sim_min)
        payload["outcome"] = "ok"
        payload["evidence"] = "net"
        stats["verify_ok"] += 1
        _log(sim, agent, "belief_verify", payload, step, sim_min)
        return

    if how == "ask":
        for other in _nearby(sim, agent):
            bel = (getattr(other, "_fact_beliefs", None) or {}).get(fid)
            if bel is None or not bel.get("verified"):
                continue
            _set_belief(sim, cfg, agent, fact, value=float(bel["value"]),
                        conf=min(float(cfg["ask_conf"]), float(bel["conf"])),
                        src="agent", parent=int(other.id), hop=int(bel["hop"]) + 1,
                        cause="verify_ask", verified="ask", step=step, sim_min=sim_min)
            payload["outcome"] = "ok"
            payload["evidence"] = f"agent:{int(other.id)}"
            stats["verify_ok"] += 1
            _log(sim, agent, "belief_verify", payload, step, sim_min)
            return
        payload["outcome"] = "no_witness"
        _log(sim, agent, "belief_verify", payload, step, sim_min)
        return

    # how == "go": 現場に居れば即確認、居なければ現場へ向かう(到着で確定=_pending_pass)
    if math.hypot(agent.x - fact["x"], agent.y - fact["y"]) <= max(
            1.0, float(fact["radius_m"]) if float(fact["radius_m"]) > 0 else 1e9):
        _confirm_site(sim, cfg, agent, fact, step, sim_min)
        payload["outcome"] = "ok"
        payload["evidence"] = "site"
        stats["verify_ok"] += 1
        _log(sim, agent, "belief_verify", payload, step, sim_min)
        return
    moved = _route_to(sim, agent, fact["node"])
    agent._tl_pending = {"fact": fid, "until": int(step) + int(cfg["verify_deadline_steps"])}
    payload["outcome"] = "pending" if moved else "pending_no_route"
    _log(sim, agent, "belief_verify", payload, step, sim_min)


def _confirm_site(sim, cfg: dict, agent, fact: dict, step: int, sim_min: int) -> None:
    _set_belief(sim, cfg, agent, fact, value=fact["value"], conf=1.0,
                src="direct", parent=-1, hop=0, cause="verify_site",
                verified="site", step=step, sim_min=sim_min)


def _pending_pass(sim, cfg: dict, step: int, sim_min: int) -> None:
    """現場確認(go)の到着判定。期限切れは静かに諦める(記録は belief_verify の pending 1 件)。"""
    facts = facts_of(sim)
    for agent in sim.agents:                   # id 昇順 = 決定論
        pend = getattr(agent, "_tl_pending", None)
        if not pend:
            continue
        if step > int(pend["until"]):
            agent._tl_pending = None
            continue
        fact = facts.get(pend["fact"])
        if fact is None:
            agent._tl_pending = None
            continue
        radius = float(fact["radius_m"])
        reach = max(1.0, radius) if radius > 0 else 1e9
        if math.hypot(agent.x - fact["x"], agent.y - fact["y"]) > reach:
            continue
        agent._tl_pending = None
        _confirm_site(sim, cfg, agent, fact, step, sim_min)
        _stats(sim)["verify_ok"] += 1
        _log(sim, agent, "belief_verify",
             {"fact": fact["id"], "fact_kind": fact["kind"], "how": "go",
              "about": "", "match": "pending", "outcome": "ok",
              "evidence": "site"}, step, sim_min)


# --------------------------------------------------------------------------- #
# model-revision(第82バッチ・計画書 §6-3。engine から驚き発火のときだけ呼ばれる)
# --------------------------------------------------------------------------- #
def revise_on_surprise(sim, agent, step: int, sim_min: int, *,
                       factor: float, max_facts: int) -> int:
    """予測外れの発火で、**未検証の伝聞信念**の確信度を決定論的に引き下げる。

    計画書 §6-3:「予測誤差が大きいときに起きるのは『単に考える』ではなく**世界モデルの
    書き換え**」。ô(監視仕様)の書き換えは LLM が担い、こちらは**台帳側の既存機構
    (`_set_belief`)を呼ぶだけ**の決定論規則である。

    規則(意図的に最小):
      - 対象は `verified is None` かつ `src != "direct"` の信念だけ。
        **自分で目撃した/現場で確かめた信念は動かさない**(証拠の等級を尊重する)。
      - 確信度に `factor`(<1)を掛ける。値(value)は変えない — 「どちらが本当か」を
        コードが決めてしまわないため。動かすのは**確信の度合い**だけ。
      - 直近取得(step 降順)の `max_facts` 件まで。イベント量を有界にする。

    乱数ゼロ・LLM 呼ゼロ。beliefs OFF や対象なしでは 1 件も記録しない(= 0 を返す)。
    """
    if not enabled(sim) or max_facts <= 0 or not (0.0 <= factor < 1.0):
        return 0
    cfg = cfg_of(sim)
    facts = facts_of(sim)
    bels = beliefs_of(agent)
    targets = [fid for fid, rec in bels.items()
               if rec.get("verified") is None and rec.get("src") != "direct"
               and fid in facts]
    targets.sort(key=lambda fid: (-int(bels[fid]["step"]), fid))   # 決定論の全順序
    n = 0
    for fid in targets[:max_facts]:
        rec = bels[fid]
        _set_belief(sim, cfg, agent, facts[fid], value=float(rec["value"]),
                    conf=float(rec["conf"]) * float(factor), src=str(rec["src"]),
                    parent=int(rec["from"]), hop=int(rec["hop"]),
                    cause="model_revision", verified=None,
                    step=step, sim_min=sim_min)
        n += 1
    return n


# --------------------------------------------------------------------------- #
# phase(scheduler から毎 step 1 回)
# --------------------------------------------------------------------------- #
def phase(sim, step: int, sim_min: int) -> None:
    """真偽台帳フェーズ: 台帳追記 → 直接目撃 → 伝聞 → 現場確認の到着判定。

    既定 OFF=即 return(fact 0 件・イベント 0 件・新フィールド無し=ゴールデン L1 バイト一致)。
    乱数を 1 個も引かない(新 stream ゼロ)。LLM 呼を 1 本も足さない(k 不変)。"""
    cfg = cfg_of(sim)
    if not cfg["enabled"]:
        return
    spec_map = cfg["fact_kinds"]
    utterances = []
    for e in _new_events(sim):                 # L1 を1回だけ走査(fact 抽出と発話収集を兼ねる)
        spec = spec_map.get(e.kind)
        if spec is not None:
            _record_fact(sim, cfg, e, spec, step, sim_min)
        utt = _UTTERANCE.get(e.kind)
        if utt is not None and int(e.agent_id) >= 0:
            utterances.append((e, utt[0], utt[1]))
    _witness_pass(sim, cfg, step, sim_min)
    # ablate.propagation_off(第78バッチ): 伝聞は「他者の発話本文から相手の信念を書き換える」
    # 経路そのものなので、この対照条件では 1 件も通さない(直接目撃 _witness_pass と
    # 現場確認 _pending_pass は物理経路なので残す)。既定 OFF=従来どおり全件通る。
    from . import ablate as _ablate_mod
    if _ablate_mod.propagation_off(sim):
        utterances = []
    _transmit_pass(sim, cfg, utterances, step, sim_min)
    _pending_pass(sim, cfg, step, sim_min)


# --------------------------------------------------------------------------- #
# L2 全体スカラー 5 列(ON 時のみ・OFF は None=列なし=L2 不変。gossip/echo と同型)
# --------------------------------------------------------------------------- #
COLUMNS = ("belief_facts_total", "belief_distance_mean", "belief_verified_rate",
           "belief_hop_mean", "belief_holders_mean")


def scalars(sim) -> dict | None:
    """L2 全体スカラー(OFF は None)。個体別・fact 別は L1 から事後再構成する。

      belief_facts_total   … 台帳に積まれた fact 総数
      belief_distance_mean … 信念距離 = mean |信念の値 − 真値|(全 (agent,fact) 対)
      belief_verified_rate … 検証済み(目撃/現場/伝聞先/ネット)の信念の割合
      belief_hop_mean      … 伝播木の平均深さ(直接目撃=0)
      belief_holders_mean  … 1 fact あたり平均保持者数"""
    cfg = cfg_of(sim)
    if not cfg["enabled"]:
        return None
    facts = facts_of(sim)
    n_dist = 0
    dist_sum = 0.0
    hop_sum = 0
    n_ver = 0
    for agent in getattr(sim, "agents", []) or []:
        for fid, bel in (getattr(agent, "_fact_beliefs", None) or {}).items():
            fact = facts.get(fid)
            if fact is None:
                continue
            dist_sum += abs(float(bel["value"]) - float(fact["value"]))
            hop_sum += int(bel["hop"])
            n_ver += 1 if bel.get("verified") else 0
            n_dist += 1
    return {
        "belief_facts_total": int(len(facts)),
        "belief_distance_mean": round(dist_sum / n_dist, 6) if n_dist else 0.0,
        "belief_verified_rate": round(n_ver / n_dist, 6) if n_dist else 0.0,
        "belief_hop_mean": round(hop_sum / n_dist, 6) if n_dist else 0.0,
        "belief_holders_mean": round(n_dist / len(facts), 6) if facts else 0.0,
    }


# --------------------------------------------------------------------------- #
# 台帳のサイドカー出力(**L1 には真値を出さない**。解析はこのファイルと L1 を突き合わせる)
# --------------------------------------------------------------------------- #
LEDGER_NAME = "beliefs_ledger.json"


def dump(sim) -> None:
    """真偽台帳を runs/<name>/beliefs_ledger.json に書き出す(ON 時のみ・finalize から 1 回)。

    L1(エージェント由来のイベント列)とは**別ファイル**に分けてあるのは、真値が
    L1 の消費経路(行間ダイジェスト等)へ紛れ込む余地を構造的に断つため。"""
    import json
    cfg = cfg_of(sim)
    if not cfg["enabled"]:
        return
    facts = facts_of(sim)
    out = {"n_facts": len(facts),
           "canary_prefix": CANARY_PREFIX,
           "facts": [facts[fid] for fid in sorted(facts)]}
    path = sim.out_dir / LEDGER_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                    encoding="utf-8")
