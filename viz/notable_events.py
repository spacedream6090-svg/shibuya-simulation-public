"""顕著イベント抽出(3D ビューア「顕著イベント」パネル用)。

ランのイベントログ(l1_events.parquet)から「物語として顕著」なイベントだけを
決定論的に抜き出し、時刻順のリストにする。make_viewer3d.py が読み込み、
notable-data として HTML に注入する(データ無し時は注入されず従来とバイト同一)。

入力源は export_3d.py と同一の l1_events.parquet(sim 本体には非依存)。
乱数は一切使わない(抽選・サンプリングは step 等分の決定論選択)。

対応表(kind → 日本語ラベル・重要度・text 用 payload キー)は NOTABLE_KINDS 一箇所で保守する。
件数が多い kind(viral_cascade / label_adopt 等)は PER_KIND_CAP で上位 N に間引く
(magnitude を持つ kind は値の降順で上位 N、それ以外は step 等分サンプル)。
間引いた件数は caps に必ず記録する(silent cap 禁止)。
"""
from __future__ import annotations

import json

# ------------------------------------------------------------------ 対応表
# kind: (日本語ラベル, 重要度 1..5, text に載せる payload キー列)
#   重要度 5 = 都市/一生の一大事、4 = 制度改変・人生の節目、3 = 顕著な社会活動、2 = 普及・その他。
# ここ 1 箇所を編集するだけで抽出対象・表示が変わる(保守しやすさ最優先)。
NOTABLE_KINDS: dict[str, tuple[str, int, list[str]]] = {
    # --- 都市/世界レベルのショック ---
    "disaster":         ("災害", 5, ["kind", "phase"]),
    "world_event":      ("世界イベント", 5, ["title", "word"]),
    "scenario_shock":   ("摂動シナリオ", 5, ["kind", "phase"]),
    "annual_event":     ("年中行事", 4, ["name", "date"]),
    "infra_outage":     ("インフラ障害", 4, ["kind", "phase"]),
    "transit_delay":    ("交通の遅延・運休", 4, ["line", "kind"]),
    "crowd_surge":      ("大規模群集", 4, ["level", "event"]),
    # --- 選挙・議会・制度(制度改変) ---
    "council_elected":  ("議会改選", 5, ["term", "members"]),
    "election_result":  ("選挙開票", 5, ["elected", "votes"]),
    "candidacy":        ("立候補", 4, ["day", "deposit"]),
    "ordinance_vote":   ("条例議決", 4, ["passed", "yes", "no"]),
    "proposal":         ("提案・署名運動", 3, ["text"]),
    "proposal_passed":  ("提案の成立", 4, ["text", "supporters"]),
    "institution":      ("制度化", 4, ["name", "norm_text"]),
    "institution_rule": ("ルールの制定", 4, ["name", "type"]),
    "rule_repealed":    ("ルールの廃止", 4, ["name", "type"]),
    # --- 世界を変えるツール行使 ---
    "group_found":      ("コミュニティ結成", 4, ["name", "purpose"]),
    "event_host":       ("イベント開催", 3, ["title", "place"]),
    "flyer_post":       ("掲示・ビラ", 3, ["text"]),
    "venture_open":     ("出店・開業", 4, ["name", "offer"]),
    "venture_fulltime": ("起業(本業化)", 4, ["name"]),
    "labor_action":     ("労働争議", 4, ["org", "demand"]),
    "vc_investment":    ("ベンチャー出資", 4, ["venture", "amount"]),
    # --- 制度イベント(破産・執行など) ---
    "bankruptcy":       ("自己破産", 5, ["debt"]),
    "eviction":         ("立退き/再入居", 4, ["phase", "arrears"]),
    "detention":        ("勾留", 4, ["target", "steps"]),
    "enforcement":      ("制度の執行", 4, ["target", "penalty"]),
    "crime":            ("犯罪・被害", 4, ["kind", "victim", "offender"]),
    # --- 人生の節目 ---
    "partner_formed":   ("交際成立", 4, ["other"]),
    "life_event":       ("ライフイベント", 4, ["kind", "other"]),
    "job_change":       ("転職・異動", 4, ["from_org", "to_org"]),
    "unemployment":     ("失業・求職", 4, ["state", "org"]),
    "long_goal":        ("人生目標の設定", 4, ["goal"]),
    "illness":          ("病気", 3, ["state", "kind"]),
    "medical_visit":    ("医療機関の受診", 3, ["cost"]),
    # --- 語の誕生・普及(自然コイン観察) ---
    "label_coin":       ("新語の創出", 4, ["text"]),
    "label_adopt":      ("新語の普及", 2, ["item_id"]),
    # --- 情報カスケード ---
    "viral_cascade":    ("バイラル拡散", 3, ["reach"]),
    "misinfo":          ("誤情報・炎上", 3, ["kind"]),
    "nuisance":         ("迷惑行為", 2, ["kind"]),
}

# 件数が多い kind の 1 kind あたり上限(超過分は間引き=caps に記録)。
DEFAULT_PER_KIND_CAP = 50

# 間引き時に「上位 N」を決める magnitude(payload の数値キー)。
# ここに無い kind は step 等分サンプル(時系列の広がりを保つ決定論選択)。
MAGNITUDE_KEYS: dict[str, str] = {
    "viral_cascade":   "reach",
    "proposal_passed": "supporters",
    "vc_investment":   "amount",
    "bankruptcy":      "debt",
    "crime":           "amount",
}

# text に list 値を「N名」と出す kind(それ以外の list は「N件」)。
_MEMBER_LIST_KINDS = {"council_elected", "election_result"}


# ------------------------------------------------------------------ 内部ヘルパ
def _as_payload(raw) -> dict:
    """payload は parquet では JSON 文字列・test では dict のどちらもあり得る。"""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw:
        try:
            v = json.loads(raw)
            return v if isinstance(v, dict) else {}
        except Exception:
            return {}
    return {}


def _fmt_val(kind: str, v) -> str:
    if isinstance(v, list):
        n = len(v)
        return f"{n}名" if kind in _MEMBER_LIST_KINDS else f"{n}件"
    if isinstance(v, bool):
        return "可決" if v else "否決"
    if isinstance(v, float):
        return str(int(v)) if v.is_integer() else f"{v:.2f}"
    return str(v)


def _summarize(kind: str, keys: list[str], payload: dict) -> str:
    """payload から text 用の短い要約を作る(決定論・HTML エスケープはビューア側 JS)。"""
    parts = []
    for k in keys:
        if k not in payload:
            continue
        v = payload[k]
        if v is None or v == "" or v == []:
            continue
        parts.append(_fmt_val(kind, v))
    text = "・".join(parts)
    return text[:80]


def _magnitude(kind: str, payload: dict) -> float:
    mk = MAGNITUDE_KEYS.get(kind)
    if mk is None:
        return 0.0
    v = payload.get(mk)
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _even_indices(n: int, cap: int) -> list[int]:
    """0..n-1 から cap 個を等間隔で選ぶ index(両端を含む・決定論)。"""
    if n <= cap:
        return list(range(n))
    if cap <= 1:
        return [0]
    return [round(i * (n - 1) / (cap - 1)) for i in range(cap)]


# ------------------------------------------------------------------ 抽出本体
def extract_notable(events, n_steps: int = 0, stride: int = 1,
                    per_kind_cap: int = DEFAULT_PER_KIND_CAP) -> dict:
    """イベント dict 列 → 顕著イベントの時刻順リスト(決定論)。

    events: {step, sim_min, agent_id, kind, x, y, payload(JSON文字列 or dict)} の列。
    n_steps: tracks のフレーム数(0=未知)。stride: tracks の step 間引き幅(既定 1)。
    戻り値: {"kinds", "events", "caps", "n_total", "n_kept"}。events が空なら注入されない。
    """
    stride = max(1, int(stride or 1))

    # 1) 対象 kind だけ拾って正規化(1 kind ずつ束ねる)。
    by_kind: dict[str, list[dict]] = {}
    for e in events:
        kind = e.get("kind")
        spec = NOTABLE_KINDS.get(kind)
        if spec is None:
            continue
        label, importance, keys = spec
        payload = _as_payload(e.get("payload"))
        try:
            x = round(float(e.get("x", 0.0) or 0.0), 1)
            y = round(float(e.get("y", 0.0) or 0.0), 1)
        except (TypeError, ValueError):
            x = y = 0.0
        step = int(e.get("step", 0))
        sim_min = e.get("sim_min")
        sim_min = int(sim_min) if sim_min is not None else step * 10
        agent_id = int(e.get("agent_id", -1))
        rec = {
            "step": step,
            "sim_min": sim_min,
            "kind": kind,
            "label": label,
            "importance": importance,
            "agent_id": agent_id,
            "x": x,
            "y": y,
            # 世界イベント(agent_id=-1)や原点は位置が無い=カメラ移動しない
            "has_pos": not (x == 0.0 and y == 0.0),
            "text": _summarize(kind, keys, payload),
            "_mag": _magnitude(kind, payload),
        }
        by_kind.setdefault(kind, []).append(rec)

    # 2) kind ごとに間引き(magnitude 降順 上位 N / それ以外は step 等分)。
    caps: dict[str, dict] = {}
    kept_all: list[dict] = []
    for kind, recs in by_kind.items():
        recs.sort(key=lambda r: (r["step"], r["sim_min"]))
        total = len(recs)
        if total > per_kind_cap:
            if kind in MAGNITUDE_KEYS:
                top = sorted(recs, key=lambda r: (-r["_mag"], r["step"]))[:per_kind_cap]
                sel = sorted(top, key=lambda r: (r["step"], r["sim_min"]))
            else:
                idxs = _even_indices(total, per_kind_cap)
                sel = [recs[i] for i in idxs]
            kept = len(sel)
        else:
            sel = recs
            kept = total
        caps[kind] = {"total": total, "kept": kept, "dropped": total - kept}
        kept_all.extend(sel)

    # 3) 時刻順に並べ(同時刻は重要度降順→kind)、フレーム番号を付与。
    kept_all.sort(key=lambda r: (r["step"], -r["importance"], r["kind"]))
    out_events = []
    for r in kept_all:
        frame = r["step"] // stride
        if n_steps > 0:
            frame = min(frame, n_steps - 1)
        out_events.append({
            "step": r["step"],
            "frame": int(frame),
            "sim_min": r["sim_min"],
            "kind": r["kind"],
            "label": r["label"],
            "importance": r["importance"],
            "agent_id": r["agent_id"],
            "x": r["x"],
            "y": r["y"],
            "has_pos": r["has_pos"],
            "text": r["text"],
        })

    kinds_present = {k: {"label": NOTABLE_KINDS[k][0],
                         "importance": NOTABLE_KINDS[k][1]}
                     for k in sorted(by_kind)}
    return {
        "kinds": kinds_present,
        "events": out_events,
        "caps": caps,
        "n_total": sum(c["total"] for c in caps.values()),
        "n_kept": len(out_events),
    }


# ------------------------------------------------------------------ parquet 入力
def load_events(parquet_path) -> list[dict]:
    """l1_events.parquet から顕著 kind の行だけを読む(pushdown filter で軽量化)。

    export_3d.py と同じ pyarrow 経路。filter が使えない環境では列読み+Python 絞り込みに退避。
    """
    import pyarrow.parquet as pq

    cols = ["step", "sim_min", "agent_id", "kind", "x", "y", "payload"]
    kinds = list(NOTABLE_KINDS)
    try:
        table = pq.read_table(parquet_path, columns=cols,
                              filters=[("kind", "in", kinds)])
        return table.to_pylist()
    except Exception:
        # フォールバック: kind 列だけ読んで対象行を特定 → Python で絞り込み。
        table = pq.read_table(parquet_path, columns=cols)
        rows = table.to_pylist()
        kset = set(kinds)
        return [r for r in rows if r.get("kind") in kset]


def extract_from_run(parquet_path, n_steps: int = 0, stride: int = 1,
                     per_kind_cap: int = DEFAULT_PER_KIND_CAP) -> dict:
    """ラン(parquet)→ 顕著イベント dict。make_viewer3d から呼ぶ入口。"""
    events = load_events(parquet_path)
    return extract_notable(events, n_steps=n_steps, stride=stride,
                           per_kind_cap=per_kind_cap)
