"""イベント表示の**単一レジストリ**+ 顕著イベント抽出(3D ビューア「顕著イベント」パネル用)。

ランのイベントログ(l1_events.parquet)から「物語として顕著」なイベントだけを
決定論的に抜き出し、時刻順のリストにする。make_viewer3d.py が読み込み、
notable-data として HTML に注入する(データ無し時は注入されず従来とバイト同一)。

入力源は export_3d.py と同一の l1_events.parquet(sim 本体には非依存)。
乱数は一切使わない(抽選・サンプリングは step 等分の決定論選択)。

■ 単一レジストリ(F1・第137)
    docs/research/emergent-events-and-narrative-ui.md §2-2 が指摘した二重管理
    (`NOTABLE_KINDS`(3D 顕著パネル)と `scripts/live_viewer.HIGHLIGHT_KINDS`
    (ライブ・ティッカー)が集合も性質も不一致)を、**KIND_REGISTRY 一表**へ統合した。
      - `NOTABLE_KINDS`    = registry の notable=True を畳んだ従来と同内容の派生 dict
      - `MAGNITUDE_KEYS`   = registry の mag を畳んだ従来と同内容の派生 dict
      - `HIGHLIGHT_KINDS`  = registry の live=True を畳んだ集合(live_viewer が import)
      - `FEED_KINDS`       = registry 全体(= 2 系統の和集合。イベントフィードの母集合)
    したがって **kind を足す/重要度を変える作業はこのファイル 1 箇所**で完結する。
    派生 3 者が従来値と 1 文字も違わないことは tests/test_event_feed.py が固定する
    (= 3D ビューアもライブ画面も、統合によって挙動が変わらない)。

対応表の各欄:
    label      日本語ラベル(UI 表示名)
    importance 重要度 1..5(5=都市/一生の一大事、4=制度改変・人生の節目、3=顕著な社会活動、
               2=普及・反復、1=配管寄り)。フィードのランキング第1項。
    keys       text 用 payload キー列(先頭から順に拾って「・」で連結)
    mag        規模を表す payload の数値キー(間引き/ magnitude_z に使う。無ければ None)
    story      ストーリーライン鍵の作り方: payload キーの候補列 or "pair"(当事者2名)or None
    icon       フィード行の絵文字
    notable    3D「顕著イベント」パネルの対象か
    live       ライブ・ティッカーの対象か

件数が多い kind(viral_cascade / label_adopt 等)は PER_KIND_CAP で上位 N に間引く
(magnitude を持つ kind は値の降順で上位 N、それ以外は step 等分サンプル)。
間引いた件数は caps に必ず記録する(silent cap 禁止)。
"""
from __future__ import annotations

import json


# ------------------------------------------------------------------ 単一レジストリ
def _spec(label: str, importance: int, keys=(), *, mag: str | None = None,
          story=None, icon: str = "•", notable: bool = True,
          live: bool = False) -> dict:
    """レジストリ 1 行(キーワード引数で欄を明示=位置ずれ事故を起こさない)。"""
    return {"label": label, "importance": int(importance), "keys": list(keys),
            "mag": mag, "story": story, "icon": icon,
            "notable": bool(notable), "live": bool(live)}


# kind → 表示仕様。**ここ 1 箇所**を編集すれば 3D 顕著パネル/ライブ・ティッカー/
# イベントフィードの全部が追随する。
KIND_REGISTRY: dict[str, dict] = {
    # --- 都市/世界レベルのショック ---
    "disaster":         _spec("災害", 5, ["kind", "phase"], icon="🌊"),
    "world_event":      _spec("世界イベント", 5, ["title", "word"], icon="📰",
                              story=("title",), live=True),
    "scenario_shock":   _spec("摂動シナリオ", 5, ["kind", "phase"], icon="⚡", live=True),
    "annual_event":     _spec("年中行事", 4, ["name", "date"], icon="🎌", story=("name",)),
    "infra_outage":     _spec("インフラ障害", 4, ["kind", "phase"], icon="🔌"),
    "transit_delay":    _spec("交通の遅延・運休", 4, ["line", "kind"], icon="🚃",
                              story=("line",)),
    "crowd_surge":      _spec("大規模群集", 4, ["level", "event"], icon="👥",
                              mag=None, story=("event", "node")),
    # --- 選挙・議会・制度(制度改変) ---
    "council_elected":  _spec("議会改選", 5, ["term", "members"], icon="🏛"),
    "election_result":  _spec("選挙開票", 5, ["elected", "votes"], icon="🗳", live=True),
    "candidacy":        _spec("立候補", 4, ["day", "deposit"], icon="✋", live=True),
    "ordinance_vote":   _spec("条例議決", 4, ["passed", "yes", "no"], icon="📜", live=True),
    "proposal":         _spec("提案・署名運動", 3, ["text"], icon="📝",
                              story=("proposal_id", "text"), live=True),
    "proposal_passed":  _spec("提案の成立", 4, ["text", "supporters"], icon="✅",
                              mag="supporters", story=("proposal_id", "text"), live=True),
    "institution":      _spec("制度化", 4, ["name", "norm_text"], icon="⚖",
                              story=("name",), live=True),
    "institution_rule": _spec("ルールの制定", 4, ["name", "type"], icon="⚖",
                              story=("name",), live=True),
    "rule_repealed":    _spec("ルールの廃止", 4, ["name", "type"], icon="🚫",
                              story=("name",)),
    # --- 世界を変えるツール行使 ---
    "group_found":      _spec("コミュニティ結成", 4, ["name", "purpose"], icon="🤝",
                              story=("group_id", "name"), live=True),
    "event_host":       _spec("イベント開催", 3, ["title", "place"], icon="🎪",
                              story=("event_id", "title"), live=True),
    "flyer_post":       _spec("掲示・ビラ", 3, ["text"], icon="📄", live=True),
    "venture_open":     _spec("出店・開業", 4, ["name", "offer"], icon="🏪",
                              story=("venture_id", "org", "name"), live=True),
    "venture_fulltime": _spec("起業(本業化)", 4, ["name"], icon="🚀",
                              story=("venture_id", "org", "name")),
    "labor_action":     _spec("労働争議", 4, ["org", "demand"], icon="✊",
                              story=("org",), live=True),
    "vc_investment":    _spec("ベンチャー出資", 4, ["venture", "amount"], icon="💵",
                              mag="amount", story=("venture",)),
    # --- 制度イベント(破産・執行など) ---
    "bankruptcy":       _spec("自己破産", 5, ["debt"], icon="💥", mag="debt"),
    "eviction":         _spec("立退き/再入居", 4, ["phase", "arrears"], icon="🏚"),
    "detention":        _spec("勾留", 4, ["target", "steps"], icon="🚓", story="pair"),
    "enforcement":      _spec("制度の執行", 4, ["target", "penalty"], icon="🔨",
                              story="pair"),
    "crime":            _spec("犯罪・被害", 4, ["kind", "victim", "offender"], icon="🔪",
                              mag="amount", story="pair"),
    # --- 人生の節目 ---
    "partner_formed":   _spec("交際成立", 4, ["other"], icon="💞", story="pair"),
    "life_event":       _spec("ライフイベント", 4, ["kind", "other"], icon="🎊",
                              story="pair"),
    "job_change":       _spec("転職・異動", 4, ["from_org", "to_org"], icon="💼"),
    "unemployment":     _spec("失業・求職", 4, ["state", "org"], icon="📉"),
    "long_goal":        _spec("人生目標の設定", 4, ["goal"], icon="🎯", live=True),
    "illness":          _spec("病気", 3, ["state", "kind"], icon="🤒"),
    "medical_visit":    _spec("医療機関の受診", 3, ["cost"], icon="🏥"),
    # --- 語の誕生・普及(自然コイン観察) ---
    "label_coin":       _spec("新語の創出", 4, ["text"], icon="✨",
                              story=("item_id", "text"), live=True),
    "label_adopt":      _spec("新語の普及", 2, ["item_id"], icon="🗣",
                              story=("item_id",), live=True),
    # --- 情報カスケード ---
    "viral_cascade":    _spec("バイラル拡散", 3, ["reach"], icon="📈", mag="reach",
                              story=("post_id", "item_id")),
    "misinfo":          _spec("誤情報・炎上", 3, ["kind"], icon="🔥",
                              story=("item_id",)),
    "nuisance":         _spec("迷惑行為", 2, ["kind"], icon="⚠"),
    # ================= 以下は notable=False(=3D 顕著パネルの対象外)=================
    # ライブ・ティッカー(旧 HIGHLIGHT_KINDS)だけが対象にしていた kind。フィードは
    # 両者の和集合を母集合にするので、ここに importance/label を与えて層化に載せる。
    "vocab_coin":       _spec("新語の創出", 4, ["text"], icon="✨",
                              story=("item_id", "text"), notable=False, live=True),
    "place_label_bind": _spec("場所に名がついた", 4, ["word", "node"], icon="📍",
                              story=("item_id", "word"), notable=False, live=True),
    "joint_activity":   _spec("共同行動", 3, ["type", "place"], icon="👫",
                              story="pair", notable=False, live=True),
    "joint_invite":     _spec("共同行動の誘い", 2, ["verdict", "source"], icon="✉",
                              story="pair", notable=False, live=True),
    "undefined_action": _spec("未定義の行動", 3, ["action", "what"], icon="❓",
                              notable=False, live=True),
    "free_action":      _spec("自由行動", 2, ["what", "category"], icon="🎨",
                              notable=False, live=True),
    "belief_update":    _spec("信念の更新", 2, ["claim", "verdict"], icon="💭",
                              story=("item_id", "claim"), notable=False, live=True),
    "belief_transmit":  _spec("信念の伝播", 2, ["claim", "hop"], icon="💬",
                              story=("item_id", "claim"), notable=False, live=True),
    "belief_verify":    _spec("信念の検証", 3, ["claim", "verdict"], icon="🔎",
                              story=("item_id", "claim"), notable=False, live=True),
    "group_join":       _spec("コミュニティ参加", 2, ["name", "group_id"], icon="➕",
                              story=("group_id", "name"), notable=False, live=True),
    "venture_close":    _spec("廃業", 4, ["name", "reason"], icon="🏚",
                              story=("venture_id", "org", "name"),
                              notable=False, live=True),
    "relation_tier":    _spec("関係の深化", 2, ["tier", "other"], icon="🔗",
                              story="pair", notable=False, live=True),
    "relation_break":   _spec("関係の断絶", 3, ["other", "cause"], icon="💔",
                              story="pair", notable=False, live=True),
    "relation_dormant": _spec("疎遠になった", 2, ["other"], icon="🌫",
                              story="pair", notable=False, live=True),
    "relation_rekindle": _spec("再びつながった", 3, ["other"], icon="🔥",
                              story="pair", notable=False, live=True),
    "move_home":        _spec("転居", 4, ["to", "reason"], icon="📦",
                              notable=False, live=True),
    "chance_event":     _spec("偶然の出来事", 3, ["kind", "what"], icon="🎲",
                              notable=False, live=True),
    "fallback":         _spec("LLM フォールバック", 1, ["reason", "purpose"], icon="🩹",
                              notable=False, live=True),
}

# ------------------------------------------------------------------ 派生ビュー(単一レジストリの畳み込み)
# 3D「顕著イベント」パネルの対応表(従来と同内容・同型)。
NOTABLE_KINDS: dict[str, tuple[str, int, list[str]]] = {
    k: (v["label"], v["importance"], v["keys"])
    for k, v in KIND_REGISTRY.items() if v["notable"]
}

# 間引き時に「上位 N」を決める magnitude(payload の数値キー)。
# ここに無い kind は step 等分サンプル(時系列の広がりを保つ決定論選択)。
MAGNITUDE_KEYS: dict[str, str] = {
    k: v["mag"] for k, v in KIND_REGISTRY.items() if v["notable"] and v["mag"]
}

# ライブ・ティッカー(scripts/live_viewer.py)が出す「見せ場」イベント。
HIGHLIGHT_KINDS: set[str] = {k for k, v in KIND_REGISTRY.items() if v["live"]}

# イベントフィード(viz/feed_rank.py)の母集合 = レジストリ全体。
FEED_KINDS: set[str] = set(KIND_REGISTRY)

# 件数が多い kind の 1 kind あたり上限(超過分は間引き=caps に記録)。
DEFAULT_PER_KIND_CAP = 50

# text 用キーが指定されていない/payload に無い時の汎用フォールバック順
# (scripts/live_viewer._TEXT_KEYS と同じ発想。レジストリ側で 1 本化する)。
GENERIC_TEXT_KEYS = ("text", "word", "title", "name", "what", "goal", "action",
                     "norm_text", "label", "type", "kind", "verdict", "tier",
                     "claim", "topic", "place", "reason", "demand", "output")

# storyline 鍵で「当事者ペア」を作る時に見る payload キー(先頭優先)。
_PAIR_KEYS = ("other", "with", "invitee", "target", "victim", "offender",
              "to", "partner", "peer")

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


# ------------------------------------------------------------------ レジストリ公開 API
def kind_spec(kind: str) -> dict | None:
    """kind の表示仕様(レジストリ 1 行)。未登録なら None。"""
    return KIND_REGISTRY.get(kind)


def summarize(kind: str, payload: dict) -> str:
    """レジストリの keys(足りなければ GENERIC_TEXT_KEYS)で payload を 1 行に畳む。

    `_summarize` は NOTABLE_KINDS の keys しか見ない(= 3D 顕著パネルの従来挙動を凍結)。
    こちらはフィード用で、レジストリの keys が空/payload に無い kind でも汎用順で拾う。
    """
    spec = KIND_REGISTRY.get(kind)
    keys = list(spec["keys"]) if spec else []
    text = _summarize(kind, keys, payload)
    if text:
        return text
    for k in GENERIC_TEXT_KEYS:
        if k in payload and payload[k] not in (None, "", [], {}):
            return _fmt_val(kind, payload[k])[:80]
    return ""


def feed_magnitude(kind: str, payload: dict) -> float | None:
    """フィードの magnitude(レジストリ mag。notable かどうかに依らず読む)。無指定は None。"""
    spec = KIND_REGISTRY.get(kind)
    mk = spec["mag"] if spec else None
    if not mk:
        return None
    v = payload.get(mk)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None                      # NaN は無効値として捨てる


def storyline_key(kind: str, agent_id: int, payload: dict) -> str | None:
    """同一の物語(語・提案・イベント・当事者ペア・組織)を束ねる鍵。無ければ None。

    memo §4-1 の dedup(ストーリー折りたたみ)用。決定論・乱数ゼロ。
    """
    spec = KIND_REGISTRY.get(kind)
    if spec is None:
        return None
    st = spec["story"]
    if not st:
        return None
    if st == "pair":
        other = None
        for k in _PAIR_KEYS:
            v = payload.get(k)
            if isinstance(v, bool):
                continue
            if isinstance(v, int) or (isinstance(v, float) and float(v).is_integer()):
                other = int(v)
                break
        if other is None:
            return None
        a, b = sorted((int(agent_id), other))
        return f"pair:{a}-{b}"
    for k in st:
        v = payload.get(k)
        if v not in (None, "", [], {}) and not isinstance(v, (list, dict)):
            return f"{k}:{v}"
    return None


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
