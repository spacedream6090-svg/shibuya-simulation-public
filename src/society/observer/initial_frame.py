"""初期フレーム共変量(第74バッチ IDEA④。**完全な事後処理**=世界に一切効かない)。

狙い(docs/research/hackathon1-analysis/ideas-shortlist.md §2 ⑤ 後半)
--------------------------------------------------------------------
第1回ハッカソン workplace チームの実測「完全分化でも完全収束でもなく、環境の空隙
(ニッチ)の形に従う」を検証するには、**そのランが最初の数日をどう解釈したか**を
ラン単位のラベルとして持っておく必要がある。ラン内の時系列(L2)では表現できない
「このランはこういう入り方をした」という **共変量** を summary.json に残し、
`analyze_norms.py` / `analyze_sweep.py` / `analyze_endo_treatment.py` が層別に使う。

設計(R1 の観点で最も安全な形を選んだ)
--------------------------------------
- **step ループには 1 行も入らない**。`finalize()` が `logger.flush()` で確定させた
  `l1_events.parquet` を読み直して集計する(= L1/L2/プロンプト/乱数/LLM 呼数のいずれも
  1 バイトも動かない。ランの途中状態にも触れない)。
- `observer.initial_frame.days: 0`(既定)では **1 バイトも読まない**し、summary.json に
  キーも足さない(既存ランと同形)。
- 集計は完全決定論(出現順・辞書順で切る)。乱数・LLM ゼロ。
- resume したランでも canonical な l1_events.parquet は先頭から結合済みなので、
  「最初の N 日」は必ずラン全体の先頭になる。

正直な限界
----------
- 「行動」の代理はイベント種の構成比であって、エージェントの主観的な意味づけではない。
- 規範マーカー率は `labeling.norm_stage.markers` の表層一致で、mock の発話テンプレートには
  該当表現が無いため 0.0 になる(実 LLM ラン向けの列)。
- L2 平均は列が存在するランでだけ出る(機能 OFF で列が無いランはキーごと出ない)。
"""
from __future__ import annotations

import os

from ..world.clock import STEP_MINUTES

# 窓の中で平均を採る L2 列(存在するものだけ)。「初期の温度感」を表す少数の常設列に絞る。
L2_COLUMNS = ("mean_grievance", "mean_drive", "n_fires", "n_drive_requests",
              "distinct_vocab_in_use", "total_adoptions", "speech_diversity")
_UTTERANCE_KINDS = ("speak", "sns_post", "dm")


def steps_per_day() -> int:
    return max(1, 1440 // int(STEP_MINUTES))


def cfg_of_config(cfg_root) -> dict:
    """resolved config → {days, top_kinds}(未搭載の旧 config でも既定 0=OFF へ落ちる)。"""
    node = cfg_root
    for part in ("observer", "initial_frame"):
        if node is None:
            break
        try:
            node = node.get(part, None) if hasattr(node, "get") else None
        except Exception:                              # noqa: BLE001
            node = None
    days, top = 0, 20
    if node is not None:
        try:
            days = int(node.get("days", 0) or 0)
        except Exception:                              # noqa: BLE001
            days = 0
        try:
            top = int(node.get("top_kinds", 20) or 20)
        except Exception:                              # noqa: BLE001
            top = 20
    return {"days": max(0, days), "top_kinds": max(1, top)}


def _l2_means(run_dir: str, window: int) -> dict:
    from . import measure as m
    l2 = m.load_l2(run_dir)
    if not l2:
        return {}
    out: dict[str, float] = {}
    for col in L2_COLUMNS:
        vals = l2.get(col)
        if not vals:
            continue
        take = [v for v in vals[:window] if v is not None]
        if take:
            out[col] = round(sum(float(v) for v in take) / len(take), 6)
    return out


def summarize(run_dir: str, days: int, top_kinds: int = 20,
              markers: dict | None = None) -> dict | None:
    """最初の `days` 日の解釈状態サマリ(ラン単位共変量)。days<=0 / L1 不在なら None。"""
    if int(days) <= 0:
        return None
    path = os.path.join(str(run_dir), "l1_events.parquet")
    if not os.path.exists(path):
        return None
    from . import measure as m
    from . import norms as _norms

    spd = steps_per_day()
    window = int(days) * spd
    md = dict(markers or {})
    marker_list = [s for s in (list(md.get("definite") or [])
                               + list(md.get("agreement") or [])) if s]

    kinds: dict[str, int] = {}
    agents: set = set()
    n_events = n_utt = n_marked = n_coin = n_coin_media = 0
    for e in m.stream_events(str(run_dir), ["step", "agent_id", "kind", "payload"]):
        if int(e["step"]) >= window:
            break                                    # ファイル順 = step 昇順(打ち切り可)
        kind = e["kind"]
        n_events += 1
        kinds[kind] = kinds.get(kind, 0) + 1
        aid = e["agent_id"]
        if isinstance(aid, int) and aid >= 0:
            agents.add(aid)
        if kind == "label_coin":
            n_coin += 1
            if e["payload"].get("media"):
                n_coin_media += 1
        elif kind in _UTTERANCE_KINDS:
            text = e["payload"].get("text")
            if isinstance(text, str) and text:
                n_utt += 1
                if _norms.agreement_in(text, marker_list):
                    n_marked += 1
    if not n_events:
        return None
    # 件数降順 → 種名昇順の決定論で上位を切る(share は窓内全イベントを分母にする)。
    top = sorted(kinds.items(), key=lambda kv: (-kv[1], kv[0]))[:int(top_kinds)]
    return {
        "days": int(days),
        "window_steps": window,
        "n_events": n_events,
        "n_agents_active": len(agents),
        "n_kinds": len(kinds),
        "kind_share": {k: round(v / n_events, 6) for k, v in top},
        "kind_count": {k: v for k, v in top},
        "n_coin": n_coin,
        "n_coin_media": n_coin_media,
        "n_utterances": n_utt,
        "norm_marker_rate": round(n_marked / n_utt, 6) if n_utt else 0.0,
        "l2_mean": _l2_means(str(run_dir), window),
    }
