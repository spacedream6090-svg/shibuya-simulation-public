"""観測レンズ機構(第50バッチ 2026-07-23。既定 OFF)。

外部アドバイスを observable へ翻訳する3本のレンズを、1つのデータ駆動機構に載せる。
どのレンズも「読むだけ」= 行動・乱数・LLM 呼を一切増やさない(R1)。ロジックは observer 層
(本 module + aggregate.py)+ conf(lens ブロック)に閉じ、engine/cognition/world/factors/
actions/labeling には軸語を一切書かない(no-fingerprint)。

  T1 価値4軸(value4): イベント種 → 価値の4軸(values.TAGS=utility/emotion/social/epistemic を
      正準とする=第17バッチの再利用。新しい分類体系は作らない)。当日のイベント数構成を観測。
  T2 3M欲望(motives): イベント種 → earn(儲けたい)/ love(モテたい)/ recognition(認められたい)。
  T6 信用内訳(trust): status.py の合成地位スコアの材料別内訳・分布(status.material_breakdown と
      aggregate の trust_* 列)。ここでは kind_map を持たない(status を可視化するだけ)。

分類は「イベント種 → 単一の支配的な軸」への写像(kind_map)。多くの出来事は本来多価
(例: joint_activity は social+emotion、event_host は social+recognition)であり、単一軸は損失の
ある射影である。payload の下位タイプで割れる種(service_use の service、life_event の kind、
free_action の tags)は kind+payload の2段マッチを許す。純粋に機構的・移動的な種(route/move/
arrive/stay/enter_*/sleep/wake/traffic)や行政の世界イベントは「価値を表明しない」ものとして
意図的に未対応(=どの軸にも数えない)。これらは網羅ではなく探索的観測のヒューリスティックである。

正準の kind_map は本 module(下記 DEFAULT_*)を単一の源とする(status.weights と同じ流儀=既定は
コード、conf で上書き)。ラン後のビューアはこの写像を lens_map.json サイドカー経由で読む(sim⇄viz
疎結合。communities.json / agents.json と同型)。

既定 OFF(lens.enabled=false): observe() が即 None・L2 に列を足さず・サイドカーも書かない=
L1/L2/乱数・ゴールデンがバイト一致で不変。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..values import TAGS as VALUE_AXES   # ("utility","emotion","social","epistemic")

MOTIVES = ("earn", "love", "recognition")

# --------------------------------------------------------------------------- #
# T1 価値4軸 kind_map(イベント種 → 価値軸)。値は str(軸)or 2段マッチ dict。
#   2段 dict の形:
#     {"__payload__": "service", "grooming": "emotion", "_else": "utility"}   payload[key]で分岐
#     {"__payload_argmax__": "tags", "_else": "emotion"}                       payload[key]=軸→重み の argmax
# --------------------------------------------------------------------------- #
DEFAULT_VALUE4_MAP: dict[str, Any] = {
    # utility(実用: 生活の道具的・経済的行為)
    "spend": "utility", "wage": "utility", "rent": "utility", "tax": "utility",
    "withdraw": "utility", "ride": "utility", "order": "utility", "deliver": "utility",
    "restock": "utility", "b2b_trade": "utility", "production": "utility",
    "org_output": "utility", "medical_visit": "utility", "civic_service": "utility",
    "loan_grant": "utility", "loan_repay": "utility",
    # emotion(感情: 快・気分・娯楽)
    "media_use": "emotion", "lodging_checkin": "emotion", "emotion_label": "emotion",
    "affect_update": "emotion",
    # social(社会: つながり・交流)
    "speak": "social", "hear": "social", "dm": "social", "sns_post": "social",
    "sns_like": "social", "sns_reshare": "social", "conversation": "social",
    "joint_activity": "social", "event_attend": "social", "event_host": "social",
    "group_join": "social", "group_found": "social", "relation_tier": "social",
    "partner_formed": "social", "life_event": "social", "proposal_support": "social",
    "flyer_view": "social", "appointment": "social",
    # epistemic(認識: 好奇心・学び・情報)
    "study": "epistemic", "search": "epistemic", "news_read": "epistemic",
    "sns_read": "epistemic", "boredom_explore": "epistemic", "vocab_coin": "epistemic",
    "label_coin": "epistemic", "memory_recall": "epistemic", "worldview": "epistemic",
    # 2段マッチ: free_action は本人申告+辞書の 4軸重み tags を持つ→ argmax を採用(第17バッチと整合)
    "free_action": {"__payload_argmax__": "tags", "_else": "emotion"},
    # 2段マッチ: service_use は理美容=感情価/塾・習い事=学び/他=実用
    "service_use": {"__payload__": "service", "grooming": "emotion",
                    "lesson": "epistemic", "gym": "emotion", "_else": "utility"},
}

# --------------------------------------------------------------------------- #
# T2 3M欲望 kind_map(イベント種 → 欲望)。love 軸は本質的に疎(partner_formed/date系が稀)なので
#   外見・魅力の代理として理美容 service_use{service:grooming} を含める=判断であり厳密対応でない旨を明記。
#   recognition と social は重なる(主催は両価)が、「見られたい・認められたい」動機を拾うため
#   主催・発信・提案・地位系のみ recognition に置く(交流一般は value4 の social が拾う)。
# --------------------------------------------------------------------------- #
DEFAULT_MOTIVE_MAP: dict[str, Any] = {
    # earn(儲けたい: 収入・商い・投資)
    "wage": "earn", "venture_sale": "earn", "venture_open": "earn",
    "venture_fulltime": "earn", "loan_grant": "earn", "vc_investment": "earn",
    "interest_paid": "earn", "b2b_trade": "earn",
    "deliver": "earn",   # 配達員の gig 収入(agent_id=配達員 の受給側)
    # love(モテたい/親密: 恋愛・外見。疎)
    "partner_formed": "love", "partnership_declined": "love",
    "life_event": {"__payload__": "kind", "partner": "love", "_else": None},
    "service_use": {"__payload__": "service", "grooming": "love", "_else": None},
    # recognition(認められたい: 主催・発信・提案・地位)
    "event_host": "recognition", "group_found": "recognition",
    "proposal": "recognition", "proposal_passed": "recognition",
    "flyer_post": "recognition", "candidacy": "recognition",
    "election_result": "recognition", "institution": "recognition",
    "council_elected": "recognition",
}


# --------------------------------------------------------------------------- #
# cfg 正準化(status.build_status_cfg / values.build_cfg と同型: dict/OmegaConf 両対応)
# --------------------------------------------------------------------------- #
def _to_plain(raw):
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(raw):
            return OmegaConf.to_container(raw, resolve=True)
    except Exception:
        pass
    return raw


def _merge_map(default: dict, override) -> dict:
    """既定 kind_map に conf の上書き(あれば)を重ねる。上書き値が空/None のキーは削除。"""
    out = dict(default)
    override = _to_plain(override) or {}
    if isinstance(override, dict):
        for k, v in override.items():
            if v in (None, "", {}):
                out.pop(str(k), None)
            else:
                out[str(k)] = _to_plain(v)
    return out


def build_lens_cfg(raw) -> dict:
    """conf の lens ブロックを正準化(既定 OFF=現行挙動と完全同一)。

    enabled=マスター。value4/motives/trust は個別 enabled(既定 True=マスター ON 時に全部出す)。
    kind_map は本 module の DEFAULT に conf 上書きを重ねる(単一の源=コード側)。"""
    raw = _to_plain(raw) or {}
    raw = dict(raw)
    v = dict(_to_plain(raw.get("value4")) or {})
    m = dict(_to_plain(raw.get("motives")) or {})
    t = dict(_to_plain(raw.get("trust")) or {})
    return {
        "enabled": bool(raw.get("enabled", False)),
        "value4": {
            "enabled": bool(v.get("enabled", True)),
            "kind_map": _merge_map(DEFAULT_VALUE4_MAP, v.get("kind_map")),
        },
        "motives": {
            "enabled": bool(m.get("enabled", True)),
            "kind_map": _merge_map(DEFAULT_MOTIVE_MAP, m.get("kind_map")),
        },
        "trust": {"enabled": bool(t.get("enabled", True))},
    }


def cfg_of(sim) -> dict:
    """lens 設定を返す(初回のみ sim.cfg から遅延構築してキャッシュ)。

    simulation.py は読み取り専用のため cfg は本 module が遅延構築する(status.cfg_of と同型)。
    キャッシュ属性 sim.lenscfg の設定は L1/L2/L3/乱数に一切現れない=既定 OFF のバイト一致を壊さない。"""
    c = getattr(sim, "lenscfg", None)
    if c is None:
        raw = None
        try:
            raw = sim.cfg.get("lens", None)
        except Exception:
            raw = None
        c = build_lens_cfg(raw)
        sim.lenscfg = c
    return c


def enabled(sim) -> bool:
    return bool(cfg_of(sim)["enabled"])


# --------------------------------------------------------------------------- #
# 2段マッチの解決(純関数・決定論)
# --------------------------------------------------------------------------- #
def resolve_axis(kind: str, payload, kind_map: dict):
    """イベント種+payload → 軸名 or None。str=直接、dict=2段マッチ。"""
    spec = kind_map.get(kind)
    if spec is None:
        return None
    if isinstance(spec, str):
        return spec or None
    if isinstance(spec, dict):
        if "__payload_argmax__" in spec:
            key = spec["__payload_argmax__"]
            weights = payload.get(key) if isinstance(payload, dict) else None
            if isinstance(weights, dict) and weights:
                # 決定論の tie-break: 軸名昇順に並べてから最大重みの先頭を採る
                items = sorted(weights.items(), key=lambda kv: str(kv[0]))
                best = max(items, key=lambda kv: float(kv[1]))
                return str(best[0]) or None
            return spec.get("_else")
        pkey = spec.get("__payload__")
        val = payload.get(pkey) if (isinstance(payload, dict) and pkey) else None
        if val is not None and str(val) in spec:
            return spec[str(val)]
        return spec.get("_else")
    return None


def _fresh_counts(cfg: dict) -> dict:
    return {"value": {t: 0 for t in VALUE_AXES},
            "motive": {m: 0 for m in MOTIVES}}


def observe(sim) -> dict | None:
    """L2 用: 当日(暦日)のイベント数構成を軸別に集計して返す(OFF は None=列なし=L2 不変)。

    collect(sim) が step 末に 1 回だけ呼ぶ経路に載る。前回処理済み総数 sim._lens_processed 以降の
    新規イベントだけを走査して当日タリーへ加算し(暦日境界でリセット)、idempotent(同 step 内の
    複数 aggregator が呼んでも二度数えない)。flush_segment で buffer が空になっても、logger の累計
    _n_flushed を含む「総数」で数える(flush は collect より後=このタリーは flush 前に確定済み・
    flush 済みイベントは再走査せず新規分だけを取る)ので、大規模ランでも欠落・二重計上が無い。
    決定論・乱数/LLM 呼ゼロ・読むだけ。"""
    cfg = cfg_of(sim)
    if not cfg["enabled"]:
        return None
    logger = getattr(sim, "logger", None)
    if logger is None:
        return None
    events = logger.events
    total = int(getattr(logger, "_n_flushed", 0)) + len(events)   # ランを通じ単調増加
    processed = int(getattr(sim, "_lens_processed", 0))
    new = total - processed
    if new < 0:
        new = 0
    if new > len(events):          # 想定外(flush 前に処理済みのはず)。安全側=buffer 内だけ数える
        new = len(events)
    st = getattr(sim, "_lens_state", None)
    if st is None:
        st = {"day": -1, **_fresh_counts(cfg)}
        sim._lens_state = st
    vmap = cfg["value4"]["kind_map"]
    mmap = cfg["motives"]["kind_map"]
    for e in events[len(events) - new:]:
        d = int(e.sim_min) // 1440
        if d != st["day"]:         # 暦日が進んだ → 当日タリーをリセット
            fresh = _fresh_counts(cfg)
            st["value"] = fresh["value"]
            st["motive"] = fresh["motive"]
            st["day"] = d
        va = resolve_axis(e.kind, e.payload, vmap)
        if va in st["value"]:
            st["value"][va] += 1
        ma = resolve_axis(e.kind, e.payload, mmap)
        if ma in st["motive"]:
            st["motive"][ma] += 1
    sim._lens_processed = total
    return st


# --------------------------------------------------------------------------- #
# サイドカー(sim⇄viz 疎結合): 解決済み kind_map をラン dir へ書き出す。ビューアが読む。
# --------------------------------------------------------------------------- #
def resolved_maps(cfg: dict) -> dict:
    return {
        "value_axes": list(VALUE_AXES),
        "motives": list(MOTIVES),
        "value4": cfg["value4"]["kind_map"],
        "motive_map": cfg["motives"]["kind_map"],
        "value_enabled": bool(cfg["value4"]["enabled"]),
        "motive_enabled": bool(cfg["motives"]["enabled"]),
        "trust_enabled": bool(cfg["trust"]["enabled"]),
    }


def write_sidecar(sim, out_dir) -> None:
    """lens ON のときだけ runs/<name>/lens_map.json を書く(OFF は何もしない=後方互換)。

    engine 側(simulation.py)は本関数を 1 行呼ぶだけ=軸語は本 module に閉じる(no-fingerprint)。"""
    cfg = cfg_of(sim)
    if not cfg["enabled"]:
        return
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "lens_map.json").write_text(
        json.dumps(resolved_maps(cfg), ensure_ascii=False), encoding="utf-8")
