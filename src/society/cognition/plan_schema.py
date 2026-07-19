"""日課計画フレームワーク(P2 S1)= 型スキーマ + 計画コンパイラ。

既定 OFF(planning.framework.enabled=false)。ON のときだけ planning.make_plan が
ここを通す。実行側(routine.py)は触らず、コンパイル結果は現行の day_plan 表現
({when, what, place, done})へ**降格写像**して既存の消化機構に載せる(精緻な時刻窓
実行の接続は S4/S6 の仕事)。

活動スキーマ(F1): 各項目は自由文 intent を**先頭**に持ち(構造化出力の推論劣化回避=
Let Me Speak Freely 論争の落とし所)、3分類 cat(mandatory|maintenance|discretionary)・
活動タイプ what(語彙は設定値=EnvPack 思想)・place・時間窓(when/start/dur_min)・
柔軟性 flex(fixed|flexible)・同伴者 with・失敗時代替 alt を持つ。

計画コンパイラ(F2): 勤務・就学のアンカーを先に置き、残り時間窓に裁量活動を充填する
(activity-based travel demand models 共通の骨格)。

R1: 新経路・OFF は現行とバイト一致・呼数 k 非依存。この基盤コードは性格特性(因子)も
地名リテラルも書かない(test_contracts の禁止語ガード)。語彙・分類は設定値で受ける。
"""
from __future__ import annotations

# ---- 設定が語彙を供給しないときのフォールバック(現行8語彙 + 必要な追加)----
_DEFAULT_VOCAB = ["work", "study", "meal", "shop", "leisure", "park",
                  "walk", "home", "visit", "personal", "escort"]
# what → cat の既定導出(設定 cat_map が無いときのフォールバック)。
_DEFAULT_CAT_MAP = {
    "work": "mandatory", "study": "mandatory",
    "meal": "maintenance", "shop": "maintenance", "personal": "maintenance",
    "escort": "maintenance", "home": "maintenance",
    "leisure": "discretionary", "park": "discretionary",
    "walk": "discretionary", "visit": "discretionary",
}
# cat → 既定 flex(mandatory=fixed, それ以外=flexible)。逸脱確率(S4)の入力になる。
_CAT_FLEX = {"mandatory": "fixed", "maintenance": "flexible",
             "discretionary": "flexible"}
# cat → 既定所要(分)。設定 dur_min が無いときのフォールバック。
_CAT_DUR = {"mandatory": 240, "maintenance": 45, "discretionary": 60}
_CATS = ("mandatory", "maintenance", "discretionary")

# 時間帯バンドの開始分(routine._time_band と同じ境界)。降格写像の when と整合させる。
_BAND_START = {"朝": 5 * 60, "昼": 11 * 60, "午後": 14 * 60,
               "夕方": 17 * 60, "夜": 19 * 60}


def _band_of(minute: int) -> str:
    """分 of day を時間帯バンドへ写す(routine._time_band と同一境界)。"""
    h = (int(minute) % 1440) // 60
    if 5 <= h < 11:
        return "朝"
    if 11 <= h < 14:
        return "昼"
    if 14 <= h < 17:
        return "午後"
    if 17 <= h < 19:
        return "夕方"
    return "夜"


def _parse_hhmm(text: str) -> int | None:
    """"HH:MM" を分 of day へ。壊れていれば None(=バンド開始にフォールバック)。"""
    if not text or ":" not in text:
        return None
    try:
        hh, mm = text.split(":", 1)
        h, m = int(hh), int(mm)
    except (TypeError, ValueError):
        return None
    return h * 60 + m if 0 <= h < 24 and 0 <= m < 60 else None


def _as_list(value) -> list:
    """with(同伴者)を文字列リストへ正規化する。"""
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def framework_cfg(sim) -> dict | None:
    """planning.framework 設定を返す。無効/未設定なら None(=現行の計画経路)。"""
    pcfg = sim.cfg.get("planning", {}) or {}
    fw = pcfg.get("framework", {}) or {}
    if not fw.get("enabled", False):
        return None
    vocab = list(fw.get("vocab") or []) or _DEFAULT_VOCAB
    cat_map = {str(k): str(v) for k, v in (fw.get("cat_map") or {}).items()}
    dur_min = {str(k): int(v) for k, v in (fw.get("dur_min") or {}).items()}
    return {"vocab": [str(w) for w in vocab],
            "cat_map": cat_map or dict(_DEFAULT_CAT_MAP),
            "dur_min": dur_min or dict(_CAT_DUR)}


def _cat_of(what: str, fw: dict) -> str:
    """what から cat を導出(設定 cat_map → 既定マップ → discretionary)。"""
    return fw["cat_map"].get(what) or _DEFAULT_CAT_MAP.get(what) or "discretionary"


def schema_prompt(fw: dict) -> str:
    """計画プロンプトへ足すスキーマ指示(ON のみ)。冒頭に "計画スキーマ:" マーカーを置く
    (mock はこのマーカーでスキーマ準拠応答へ分岐する)。intent を必ず先頭に置かせる。"""
    vocab = "・".join(fw["vocab"])
    return (
        "\n計画スキーマ: 各予定は次の項目を持つ JSON オブジェクトにしてください"
        "(intent を必ず先頭に置く)。"
        "\n  intent: この予定で何をなぜしたいか(自由文・必須)"
        "\n  cat: mandatory(仕事・学業などの義務)|maintenance(買い物・用足しなどの維持)"
        "|discretionary(遊び・交際などの裁量)"
        f"\n  what: 活動の種類({vocab} のいずれか)"
        "\n  place: 場所の名前(任意)"
        "\n  when: 朝|昼|午後|夕方|夜"
        "\n  start: 開始時刻 HH:MM(任意) / dur_min: 所要分(任意)"
        "\n  flex: fixed(動かせない)|flexible(融通できる)(任意)"
        '\n出力例 {"action": "plan", "items": [{"intent": "…", "cat": "discretionary",'
        ' "what": "shop", "place": "", "when": "午後", "flex": "flexible"}, ...]}')


def normalize_items(raw: list, fw: dict, max_items: int) -> list:
    """LLM 出力(または前日流用 dict)を型スキーマへ正規化し、既定を補完する。

    後方互換: 任意フィールドが無ければ cat←what / flex←cat / dur_min←cat定数 /
    start←when バンド開始 で埋める。intent は自由文(空でも受ける)。冪等
    (正規化済み dict を再度通しても同じ)。
    """
    out: list = []
    for it in raw[:max_items]:
        if not isinstance(it, dict):
            continue
        what = str(it.get("what") or "").strip()
        cat = str(it.get("cat") or "").strip()
        if cat not in _CATS:
            cat = _cat_of(what, fw)
        start_min = _parse_hhmm(str(it.get("start") or "").strip())
        when = str(it.get("when") or "").strip()
        if start_min is None:
            start_min = _BAND_START.get(when, _BAND_START["朝"])
        if when not in _BAND_START:
            when = _band_of(start_min)
        flex = str(it.get("flex") or "").strip()
        if flex not in ("fixed", "flexible"):
            flex = _CAT_FLEX.get(cat, "flexible")
        try:
            dur = int(it.get("dur_min"))
        except (TypeError, ValueError):
            dur = fw["dur_min"].get(cat) or _CAT_DUR.get(cat, 60)
        out.append({"intent": str(it.get("intent") or "").strip(),
                    "cat": cat, "what": what,
                    "place": str(it.get("place") or "").strip(),
                    "when": when, "start_min": int(start_min),
                    "dur_min": int(dur), "flex": flex,
                    "with": _as_list(it.get("with")),
                    "alt": str(it.get("alt") or "").strip(),
                    "anchor": False})
    return out


def _anchor(what: str, start_min: int, dur_min: int, place: str,
            intent: str) -> dict:
    return {"intent": intent, "cat": "mandatory", "what": what,
            "place": place, "when": _band_of(start_min),
            "start_min": int(start_min), "dur_min": max(0, int(dur_min)),
            "flex": "fixed", "with": [], "alt": "", "anchor": True}


def anchors_from_shift(agent, sim_min: int) -> list:
    """勤務・就学のシフト窓(ペルソナ/組織台帳由来の work 窓 + バイト窓)を mandatory
    アンカーとして返す。当日のバイト曜日のみ計上する。"""
    out: list = []
    ws = int(getattr(agent, "work_start_min", -1) or -1)
    if ws >= 0:
        we = int(getattr(agent, "work_end_min", 0) or 0)
        out.append(_anchor("work", ws, we - ws,
                            str(getattr(agent, "work_name", "") or ""), "勤務"))
    pt = getattr(agent, "part_time", None)
    if pt:
        day = (sim_min // 1440) % 7
        if day in (pt.get("days") or []):
            ps, pe = int(pt["start_min"]), int(pt["end_min"])
            out.append(_anchor("work", ps, pe - ps, "", "バイトのシフト"))
    return out


def compile_plan(agent, sim, items: list, sim_min: int, fw: dict):
    """正規化済みスキーマ → (schedule, day_plan, summary)。

    アンカー先置き(勤務・就学の mandatory を先に配置)→ 残り時間窓に maintenance/
    discretionary を充填する。LLM 由来の mandatory は勤務窓側が担うので混ぜない。窓が
    埋まっていれば落とす(充填できない=正常系。activity-based model の合意)。

    schedule: 時刻順の実行スケジュール(観測用)。各 = 正規化 dict + anchor フラグ。
    day_plan: routine が消化する降格リスト [{when, what, place, done}](アンカーは含めない
              = 勤務窓は routine が独立に処理する)。
    summary : {n, cats, anchors} 観測用の要約。
    """
    anchors = anchors_from_shift(agent, sim_min)
    occupied = {a["when"] for a in anchors}          # アンカーが押さえたバンド
    fill: list = []
    for it in items:
        if it["cat"] == "mandatory":                 # 勤務窓が担う=LLM 由来は落とす
            continue
        if it["when"] in occupied:                   # 窓が埋まっている=落とす(#121 正常系)
            continue
        occupied.add(it["when"])                     # 1バンド1件(粗い重複解消)
        fill.append(it)
    schedule = sorted(anchors + fill, key=lambda e: e["start_min"])
    day_plan = [{"when": e["when"], "what": e["what"],
                 "place": e["place"], "done": False}
                for e in schedule if not e["anchor"]]
    cats = {c: 0 for c in _CATS}
    for e in schedule:
        cats[e["cat"]] = cats.get(e["cat"], 0) + 1
    summary = {"n": len(schedule), "cats": cats,
               "anchors": sum(1 for e in schedule if e["anchor"])}
    return schedule, day_plan, summary


def default_skeleton(agent, fw: dict) -> list:
    """初日フォールバックの職業別デフォルト骨格(LLM なし・決定論)。

    勤務窓を持つ人は退勤後の食事+帰宅、持たない人は日中の外出+食事+くつろぎ、の粗い
    骨格。place は空(routine がカテゴリから解決)。翌朝はまた LLM 計画に戻る(恒久
    ルール化はしない)。
    """
    has_work = int(getattr(agent, "work_start_min", -1) or -1) >= 0 \
        or bool(getattr(agent, "part_time", None))
    if has_work:
        raw = [{"intent": "退勤後に食事をとる", "what": "meal", "when": "夕方"},
               {"intent": "家に帰ってくつろぐ", "what": "home", "when": "夜"}]
    else:
        raw = [{"intent": "日中に買い物へ行く", "what": "shop", "when": "昼"},
               {"intent": "夕方に食事をとる", "what": "meal", "when": "夕方"},
               {"intent": "夜はゆっくり過ごす", "what": "leisure", "when": "夜"}]
    return normalize_items(raw, fw, max_items=len(raw))
