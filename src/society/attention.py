"""ATT: 自律的注意機構(層A = 顕著性 top-k 選抜 / 層B = LLM 自律宣言の注意ブロック)。

正典
----
- docs/plans/attention-mechanism-plan.md(設計全文。§2 層A・§3 層B・§6.5 個体差・§6.6 実装決定)
- docs/research/attention-cognitive-foundations.md §7(統合設計案)

何を解く問題か
--------------
現行の発話ハンドラは「聞こえた全員」を等しく処理する(hear の L1・記憶・関係台帳・
覚醒・SNC 遭遇)。Cherry 1953 / Wood & Cowan 1995 の古典が示すとおり **非注意の発話は
記憶に残らない**ので、これは人間からの乖離であり、同時に RAM と step 時間が聴衆数に
比例して膨らむ原因でもある。層A は Wolfe GS6(2021)の優先度統一式で「誰を知覚するか」を
k_i 件へ絞り、層B は「いま何を気にしているか」を LLM 自身に宣言させて永続化する。

配置(src/society 直下 = engine/cognition/actions/labeling/world の CHECKED_DIRS 外)
------------------------------------------------------------------------------------
gossip.py / truth_ledger.py と同じ流儀。因子名(no-fingerprint)を綴らないことは
本 module 自身が守る(下の `prompt_section` は機構語・実験条件語・因子名を 1 文字も
出さない)。engine は「呼ぶだけ」「返った list を配るだけ」。

R1 ドクトリン(この module が守る規律)
--------------------------------------
- 既定 `world.attention.enabled: false`(かつ `mode: "distance"`)では層A の分岐へ
  1 度も入らない = 第141 S15(`scheduler._attention_limited`)がそのまま走る = L1 バイト一致。
- 既定 `cognition.attention_block.enabled: false` では層B の全 seam が即 return し、
  **agent に `attention_slots` 属性を 1 つも生やさない**(= checkpoint / dehydrate の
  バイト列が現行と完全に一致する)。
- **乱数 stream を 1 本も引かない**。個体差(k_i / p_break_i)と貫通判定はすべて
  blake2b の安定ハッシュ(第141 の S15 / GRP と同じ流儀 = プロセス跨ぎ・resume 安定)。
- **新しい L1 kind を 1 つも作らない**(observer/schema.py・causality.py は不触)。
- 追加 LLM 呼ゼロ。層B の宣言は既存 deliberate 呼の構造化出力へ相乗りする(SNC と同型)。

正直な限界(設計書との差分。報告に明記したもの)
------------------------------------------------
1. **群衆チャンクは実装していない**。設計 §2 は「非注意ぶんを『周りが騒がしかった』の
   集約 1 件へ」と書くが、聞き手はそもそも通過者しか居ない(非通過者に集約記憶を入れると
   「知覚しなかったのに記憶が残る」= Cherry と矛盾する)。本実装は **非通過 = 完全に無処理**
   に倒し、`select` は `crowd_n`(非選抜数)を返すだけにした。群衆の雰囲気は将来の
   appraisal EMA(個体の連続量)で扱う設計とする。
2. **k_i の実効帯**: 既定 (k_base=4, k_span=2) では 2-6 で、設計 §6.5 の「2-7」の上端 7 は
   出ない(7 を出すには k_base=5 または k_span=3)。ハード境界だけ [2, 7] で留めてある。
3. **person スロットの target は「宣言された文字列そのもの」**。設計は id 文字列と書くが、
   LLM が見ているのは名前だけなので id を書きようがない。照合(層A ボーナス・宛先 boost)は
   **名前と id 文字列の両方**を受ける。
"""
from __future__ import annotations

import hashlib

SCHEMA = 1

# --------------------------------------------------------------------------- #
# 定数(conf に出さない構造定数)
# --------------------------------------------------------------------------- #
MODE_DISTANCE = "distance"      # 第141 S15 と完全同一(縮退線)
MODE_SALIENCE = "salience"      # 層A(顕著性選抜)
MODES = (MODE_DISTANCE, MODE_SALIENCE)

#: k_i のハード境界(Cowan 2001 の 4±1 と Dunbar 1995 の会話上限 4 人から、
#: 個体差を許す帯として 2-7。conf の k_base ± k_span はこの内側へクランプされる)。
K_MIN, K_MAX = 2, 7

#: p_break_i のハード境界(Conway 2001 の実測レンジ 0.20-0.65)。
P_BREAK_MIN, P_BREAK_MAX = 0.20, 0.65
#: 基底からの振れ幅(既定 p_break_base=0.33 でちょうど [0.20, 0.65] になる非対称幅)。
P_BREAK_DOWN, P_BREAK_UP = 0.13, 0.32

#: 個体側に持つ「直近で注意した話者」の有界リスト長(プライミング/履歴項の分母)。
HISTORY_MAX = 8
HISTORY_KEY = "_attn_hist"

#: 返答の相手(優先充当)の預かり欄。`(step, speaker_id)` を持ち、**同じ step のときだけ**
#: 有効(古い値が後の発話へ効かない = 掃除を要らなくする設計)。
REPLY_KEY = "_attn_reply"

#: 層B のスロット列(agent 側)。
SLOTS_KEY = "attention_slots"
SLOT_KINDS = ("person", "place", "topic", "goal")
WHY_MAX = 40                    # why の切り詰め長(≤40 字)

_SALT_K = "att_k"
_SALT_P = "att_pbreak"
_SALT_BREAK = "att_break"


# --------------------------------------------------------------------------- #
# conf 正準化(既定 = 現行同値)
# --------------------------------------------------------------------------- #
DEFAULTS: dict = {
    "enabled": False,
    "mode": MODE_DISTANCE,
    "k_base": 4,
    "k_span": 2,
    "theta_ignition": 0.0,      # 0 = 閾値なし(k だけが効く)
    "p_break_base": 0.33,
    # Wolfe GS6 の 4 重み
    "w_bu": 1.0,
    "w_td": 0.5,
    "w_h": 0.3,
    "w_v": 0.5,
    # 顕著性の内訳(距離減衰 × (1 + α_nov·新奇語 + α_emo·|情動価| + α_addr·宛先性))
    "alpha_nov": 0.5,
    "alpha_emo": 0.3,
    "alpha_addr": 1.0,
    # 層B との結合(宣言済みの相手はトップダウン注意の持続ボーナス)
    "w_slot": 0.8,
}

BLOCK_DEFAULTS: dict = {
    "enabled": False,
    "slots_max": 7,
    "decay_per_day": 0.15,
    "boost_addressed": 0.3,
    "min_salience": 0.05,
}


def _sub(cfg, *path):
    """OmegaConf / dict のどちらでも辿れる安全な下降(無ければ None)。"""
    node = cfg
    for key in path:
        if node is None:
            return None
        try:
            node = node.get(key, None)
        except (AttributeError, TypeError):
            return None
    return node


def _get(raw, key, default):
    try:
        value = raw.get(key, default)
    except (AttributeError, TypeError):
        return default
    return default if value is None else value


def build_cfg(raw) -> dict:
    """`world.attention` を型強制つきで正準化する(既定 = 現行同値 = 無効)。

    未知の `mode` は既定 `"distance"`(= 第141 S15 挙動)へ倒す。
    """
    raw = raw if raw is not None else {}
    mode = str(_get(raw, "mode", DEFAULTS["mode"]))
    if mode not in MODES:
        mode = DEFAULTS["mode"]
    k_base = int(_get(raw, "k_base", DEFAULTS["k_base"]))
    k_span = max(0, int(_get(raw, "k_span", DEFAULTS["k_span"])))
    return {
        "enabled": bool(_get(raw, "enabled", DEFAULTS["enabled"])),
        "mode": mode,
        "k_base": min(K_MAX, max(K_MIN, k_base)),
        "k_span": k_span,
        "theta_ignition": float(_get(raw, "theta_ignition",
                                     DEFAULTS["theta_ignition"])),
        "p_break_base": min(1.0, max(0.0, float(_get(raw, "p_break_base",
                                                    DEFAULTS["p_break_base"])))),
        "w_bu": float(_get(raw, "w_bu", DEFAULTS["w_bu"])),
        "w_td": float(_get(raw, "w_td", DEFAULTS["w_td"])),
        "w_h": float(_get(raw, "w_h", DEFAULTS["w_h"])),
        "w_v": float(_get(raw, "w_v", DEFAULTS["w_v"])),
        "alpha_nov": float(_get(raw, "alpha_nov", DEFAULTS["alpha_nov"])),
        "alpha_emo": float(_get(raw, "alpha_emo", DEFAULTS["alpha_emo"])),
        "alpha_addr": float(_get(raw, "alpha_addr", DEFAULTS["alpha_addr"])),
        "w_slot": float(_get(raw, "w_slot", DEFAULTS["w_slot"])),
    }


def build_block_cfg(raw) -> dict:
    """`cognition.attention_block` を型強制つきで正準化する(既定 OFF)。"""
    raw = raw if raw is not None else {}
    return {
        "enabled": bool(_get(raw, "enabled", BLOCK_DEFAULTS["enabled"])),
        "slots_max": max(1, int(_get(raw, "slots_max",
                                     BLOCK_DEFAULTS["slots_max"]))),
        "decay_per_day": min(1.0, max(0.0, float(
            _get(raw, "decay_per_day", BLOCK_DEFAULTS["decay_per_day"])))),
        "boost_addressed": max(0.0, float(
            _get(raw, "boost_addressed", BLOCK_DEFAULTS["boost_addressed"]))),
        "min_salience": max(0.0, float(
            _get(raw, "min_salience", BLOCK_DEFAULTS["min_salience"]))),
    }


def cfg_of(sim) -> dict:
    """層A の正準化済み設定(sim へ 1 度だけキャッシュ = 毎発話 conf を辿らない)。"""
    got = getattr(sim, "_attn_cfg", None)
    if isinstance(got, dict):
        return got
    got = build_cfg(_sub(getattr(sim, "cfg", None), "world", "attention"))
    try:
        sim._attn_cfg = got
    except Exception:                       # noqa: BLE001(スタブ sim = 属性を持てない)
        pass
    return got


def block_cfg_of(sim) -> dict:
    """層B の正準化済み設定(同上)。"""
    got = getattr(sim, "_attn_block_cfg", None)
    if isinstance(got, dict):
        return got
    got = build_block_cfg(_sub(getattr(sim, "cfg", None),
                               "cognition", "attention_block"))
    try:
        sim._attn_block_cfg = got
    except Exception:                       # noqa: BLE001
        pass
    return got


def enabled(sim) -> bool:
    """層A が有効か(mode に依らない。既定 OFF)。"""
    return bool(cfg_of(sim)["enabled"])


def salience_on(sim) -> bool:
    """層A の**顕著性選抜**が効くか(enabled かつ mode=salience)。

    ★`mode: "distance"` は enabled でも **第141 S15 と完全同一**(= `_attention_limited`
      がそのまま走る)。ここが False を返す限り、engine は 1 分岐も新経路へ入らない。
    """
    cfg = cfg_of(sim)
    return bool(cfg["enabled"]) and cfg["mode"] == MODE_SALIENCE


def block_on(sim) -> bool:
    """層B(注意ブロック)が有効か。既定 OFF。"""
    return bool(block_cfg_of(sim)["enabled"])


# --------------------------------------------------------------------------- #
# 個体差(blake2b の安定ハッシュ。**乱数 stream を 1 本も引かない**)
# --------------------------------------------------------------------------- #
def _u01(salt: str, *parts) -> float:
    """(用途 salt, 安定キー) → [0,1)。economy.stable_unit / diversity._u01 と同流儀。"""
    key = f"{salt}\x1f" + "\x1f".join(str(p) for p in parts)
    h = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") / 2.0 ** 64


def k_of(agent, cfg: dict) -> int:
    """個体の深処理容量 k_i(設計 §6.5)。

    k_base ± k_span の一様整数を blake2b から引き、ハード境界 [K_MIN, K_MAX] へクランプ。
    ★ペルソナ側に集中力/作業記憶容量の特性は**存在しない**(factors.TRAITS =
      nfc / risk_tolerance / internal_locus の 3 つだけ・persona v2 にも無い)ので、
      加味する材料が無い = ハッシュのみで決める(設計指示どおり調査して報告済み)。
    """
    span = int(cfg["k_span"])
    base = int(cfg["k_base"])
    if span <= 0:
        return min(K_MAX, max(K_MIN, base))
    n = 2 * span + 1                       # -span .. +span
    off = int(_u01(_SALT_K, int(agent.id)) * n)
    off = min(n - 1, max(0, off)) - span
    return min(K_MAX, max(K_MIN, base + off))


def p_break_of(agent, cfg: dict) -> float:
    """個体の貫通率 p_break_i(カクテルパーティ効果の自己名貫通。Conway 2001 の 0.20-0.65)。

    基底 `p_break_base` から下へ P_BREAK_DOWN・上へ P_BREAK_UP の非対称帯を張り、
    ハード境界 [P_BREAK_MIN, P_BREAK_MAX] へクランプする(既定 0.33 で帯は厳密に
    [0.20, 0.65] に一致する)。
    """
    base = float(cfg["p_break_base"])
    lo = max(P_BREAK_MIN, base - P_BREAK_DOWN)
    hi = min(P_BREAK_MAX, base + P_BREAK_UP)
    if hi <= lo:
        return min(P_BREAK_MAX, max(P_BREAK_MIN, base))
    return lo + _u01(_SALT_P, int(agent.id)) * (hi - lo)


# --------------------------------------------------------------------------- #
# 層A: 優先度マップ(発話 1 件 × 聴取候補 1 人あたり O(1))
# --------------------------------------------------------------------------- #
def _addressed(hearer, text: str) -> float:
    """宛先性 = 自分の名前が発話に現れるか(自己名・自己言及)。"""
    name = getattr(hearer, "name", "") or ""
    return 1.0 if (name and text and name in text) else 0.0


def salience_of(hearer, text: str, words, dist_sq: float, valence_abs: float,
                cfg: dict, ref_sq: float) -> float:
    """ボトムアップ顕著性 = 距離減衰 × (1 + α_nov·新奇語 + α_emo·|情動価| + α_addr·宛先性)。

    距離減衰は `1 / (1 + d² / r²)`(平方根も乱数も使わない。r = 知覚半径 = 減衰が 1/2 になる距離)。
    新奇語 = 話者が使った語のうち**この聞き手がまだ知らない**ものが 1 つでもあるか。
    """
    decay = 1.0 / (1.0 + (float(dist_sq) / ref_sq if ref_sq > 0.0 else 0.0))
    nov = 0.0
    if words:
        adopted = getattr(hearer, "adopted", None) or ()
        nov = 1.0 if any(w not in adopted for w in words) else 0.0
    boost = (1.0 + float(cfg["alpha_nov"]) * nov
             + float(cfg["alpha_emo"]) * float(valence_abs)
             + float(cfg["alpha_addr"]) * _addressed(hearer, text))
    return decay * boost


def relevance_of(hearer, text: str, words) -> float:
    """トップダウン関連度 = **既存タグの照合だけ**(新しい語彙も重い計算も持ち込まない)。

    ① 話者の使った語が自分の既知語彙にある(= 話が通じる話題)… +0.5
    ② 自分の趣味語(先頭 2 件まで)が発話に現れる…………………… +0.5
    どちらも有界(語は use_items = 数個・趣味は先頭 2 件)なので O(1)。
    """
    score = 0.0
    if words:
        adopted = getattr(hearer, "adopted", None) or ()
        if adopted and any(w in adopted for w in words):
            score += 0.5
    hobbies = getattr(hearer, "hobbies", None) or ()
    if text:
        for hob in list(hobbies)[:2]:
            if hob and hob in text:
                score += 0.5
                break
    return score


def history_of(hearer, speaker_id: int) -> float:
    """履歴(プライミング)= 直近 HISTORY_MAX 件のうちこの話者を注意した割合(飽和つき)。"""
    hist = getattr(hearer, HISTORY_KEY, None)
    if not hist:
        return 0.0
    return min(1.0, hist.count(int(speaker_id)) / float(HISTORY_MAX))


def value_of(sim, hearer, speaker_id: int) -> float:
    """価値 = 関係台帳の closeness(+ tier)+ contacts 在籍。台帳の**読み取りだけ**。"""
    score = 0.0
    mem = getattr(hearer, "mem", None)
    rels = getattr(mem, "relations", None) if mem is not None else None
    if rels:
        rec = rels.get(int(speaker_id))
        if rec:
            score += float(rec.get("closeness", 0.0) or 0.0)
            score += 0.1 * min(3, int(rec.get("tier", 0) or 0))
    net = getattr(sim, "net", None)
    contacts = getattr(net, "contacts", None) if net is not None else None
    if contacts:
        mine = contacts.get(int(getattr(hearer, "id", -1)))
        if mine and int(speaker_id) in mine:
            score += 0.5
    return score


def slot_bonus_of(hearer, speaker) -> float:
    """層B 結合: 話者が自分の注意ブロックに居れば、その salience をボーナスとして返す。

    ★照合は **名前と id 文字列の両方**を受ける(LLM は名前しか見ていないため。
      module docstring の「正直な限界 3」)。
    """
    slots = getattr(hearer, SLOTS_KEY, None)
    if not slots:
        return 0.0
    sid = str(int(getattr(speaker, "id", -1)))
    sname = getattr(speaker, "name", "") or ""
    for slot in slots:
        if slot.get("kind") != "person":
            continue
        target = slot.get("target")
        if target == sid or (sname and target == sname):
            return float(slot.get("salience", 1.0) or 0.0)
    return 0.0


def score(sim, speaker, hearer, text: str, words, dist_sq: float,
          valence_abs: float, cfg: dict, ref_sq: float,
          block: bool = False) -> float:
    """Wolfe GS6(2021)の優先度統一式(設計 §2)。

        priority = w_bu·salience + w_td·relevance + w_h·history + w_v·value
                   (+ w_slot·層B スロット salience)
    """
    sid = int(getattr(speaker, "id", -1))
    prio = (float(cfg["w_bu"]) * salience_of(hearer, text, words, dist_sq,
                                             valence_abs, cfg, ref_sq)
            + float(cfg["w_td"]) * relevance_of(hearer, text, words)
            + float(cfg["w_h"]) * history_of(hearer, sid)
            + float(cfg["w_v"]) * value_of(sim, hearer, sid))
    if block:
        prio += float(cfg["w_slot"]) * slot_bonus_of(hearer, speaker)
    return prio


def select(sim, speaker, hearers, text: str, words, step: int,
           radius: float = 0.0, valence_abs: float = 0.0,
           pref_ids=(), cfg: dict | None = None,
           block: bool | None = None) -> tuple[list, int]:
    """層A の選抜。返り値 `(attended, crowd_n)`。

    規則(全部決定論・乱数ゼロ):
      1. 会話相手(`pref_ids`)は**優先充当**(θ を問わず先に枠を取る)。
      2. 残りを priority 降順(同値は id 昇順)に見て、`priority ≥ θ_ignition` かつ
         枠が残っている間だけ通す(**k は上限でありノルマではない**)。
      3. 貫通: 自分の名前が発話に出ている非選抜者は、`blake2b(hearer, speaker, step)` の
         [0,1) 値が `p_break_i` 未満なら**枠外でも**追加される(カクテルパーティ効果)。
      4. 返り値は **id 昇順**(下流の走査順を絞らないときと同じ規則に保つ = S15 と同流儀)。

    `crowd_n` = 非選抜の人数。**非選抜者には何もしない**(記憶も L1 も関係も遭遇も
    1 件も作らない)= Cherry 1953 準拠。群衆チャンクは実装しない(module docstring)。
    """
    if not hearers:
        return ([], 0)
    cfg = cfg_of(sim) if cfg is None else cfg
    if block is None:
        block = block_on(sim)
    ref = float(radius)
    ref_sq = ref * ref if ref > 0.0 else 1.0
    sx, sy = float(speaker.x), float(speaker.y)
    pref = {int(i) for i in pref_ids}
    ranked = []
    for hearer in hearers:
        dx, dy = float(hearer.x) - sx, float(hearer.y) - sy
        prio = score(sim, speaker, hearer, text, words, dx * dx + dy * dy,
                     valence_abs, cfg, ref_sq, block=block)
        ranked.append((prio, int(hearer.id), hearer))
    cap = k_of(speaker, cfg)
    theta = float(cfg["theta_ignition"])
    chosen: dict[int, object] = {}
    # ① 会話相手の優先充当(枠は消費する = k は上限のまま)
    for _p, hid, hearer in ranked:
        if hid in pref and len(chosen) < cap:
            chosen[hid] = hearer
    # ② priority 降順 → 同値は id 昇順
    for prio, hid, hearer in sorted(ranked, key=lambda t: (-t[0], t[1])):
        if len(chosen) >= cap:
            break
        if hid in chosen:
            continue
        if prio < theta:
            break                          # 以降はもっと低い = 全部落ちる
        chosen[hid] = hearer
    # ③ 貫通(自己名・自己言及)。**選抜枠の外**へ足す。
    if text:
        for _p, hid, hearer in ranked:
            if hid in chosen or not _addressed(hearer, text):
                continue
            if _u01(_SALT_BREAK, hid, int(speaker.id), int(step)) \
                    < p_break_of(hearer, cfg):
                chosen[hid] = hearer
    attended = [chosen[hid] for hid in sorted(chosen)]
    return (attended, len(hearers) - len(attended))


def note_attended(attended, speaker_id: int) -> None:
    """選抜された聞き手の履歴(プライミング)へ話者を 1 件積む(有界 = HISTORY_MAX)。

    `select` を純関数のままにするため、状態を書くのはこの関数だけにしてある。
    """
    sid = int(speaker_id)
    for hearer in attended:
        hist = getattr(hearer, HISTORY_KEY, None)
        if hist is None:
            hist = []
            setattr(hearer, HISTORY_KEY, hist)
        hist.append(sid)
        if len(hist) > HISTORY_MAX:
            del hist[:-HISTORY_MAX]


# --------------------------------------------------------------------------- #
# 会話相手(返答文脈)の預かり: 「いま返事をしている相手」を優先充当するための 1 欄
# --------------------------------------------------------------------------- #
def note_reply_target(sim, agent, speaker_id, step: int) -> None:
    """返答を撃つ直前に「誰へ返すか」を預かる(層A の顕著性選抜が有効なときだけ)。

    値は `(step, speaker_id)` で、**同じ step の発話にしか効かない**(古い値が後の
    発話へ効かない = 明示的な掃除が要らない)。OFF では属性を 1 つも生やさない。
    """
    if speaker_id is None or not salience_on(sim):
        return
    try:
        setattr(agent, REPLY_KEY, (int(step), int(speaker_id)))
    except Exception:                       # noqa: BLE001
        pass


def reply_target_of(agent, step: int) -> tuple:
    """この step の返答相手 id(無ければ空 tuple)。"""
    got = getattr(agent, REPLY_KEY, None)
    if not got:
        return ()
    try:
        when, who = got
    except (TypeError, ValueError):
        return ()
    return (int(who),) if int(when) == int(step) else ()


# --------------------------------------------------------------------------- #
# 層B: 注意ブロック(LLM が構造化出力で編集する有限スロット)
# --------------------------------------------------------------------------- #
def slots_of(agent) -> list:
    """現在の注意スロット(無ければ空 list。**属性は生やさない**)。"""
    got = getattr(agent, SLOTS_KEY, None)
    return got if isinstance(got, list) else []


def apply_declaration(sim, agent, action, step: int) -> int:
    """発火の構造化出力 `attend` を受理する(MEM1 式の**全量再宣言**)。返り値 = 採用件数。

    - **欄が無い = 変更なし**(mock バックエンドは出さない = mock ランでは 1 件も動かない)。
      SNC の `relate`/`follow` と同じ流儀。
    - list であれば(空 list でも)**全量置換**。空 list = 「もう何も気にしていない」の宣言。
    - `kind` が SLOT_KINDS 以外 / `target` が空 の項目は捨てる。
    - 同一 target の重複は先頭だけ採る。`slots_max` 超過分は**先頭から数えて上限まで**
      (LLM が並べた順 = 本人にとっての重要度順、と読む)。
    - `salience` は宣言のたび 1.0 で初期化し、`since` は**既存の同一 target なら維持**する
      (「いつから気にしているか」は宣言の再掲で若返らない)。
    """
    if not block_on(sim) or not isinstance(action, dict):
        return 0
    decl = action.get("attend")
    if not isinstance(decl, list):
        return 0                            # 欠落 = 変更なし(1 バイトも触らない)
    cfg = block_cfg_of(sim)
    prev = {str(s.get("target")): s for s in slots_of(agent)}
    out: list[dict] = []
    seen: set[str] = set()
    for item in decl:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind", "") or "").strip()
        if kind not in SLOT_KINDS:
            continue
        target = str(item.get("target", "") or "").strip()
        if not target or target in seen:
            continue
        seen.add(target)
        why = str(item.get("why", "") or "").strip()[:WHY_MAX]
        old = prev.get(target)
        out.append({"kind": kind, "target": target, "why": why,
                    "salience": 1.0,
                    "since": int(old["since"]) if old else int(step)})
    agent.attention_slots = out[:int(cfg["slots_max"])]
    return len(agent.attention_slots)


def note_addressed(sim, hearer, speaker_id: int) -> None:
    """自分宛ての発話を受けた step の boost(該当 person スロットの salience を上げる)。"""
    if not block_on(sim):
        return
    slots = slots_of(hearer)
    if not slots:
        return
    cfg = block_cfg_of(sim)
    boost = float(cfg["boost_addressed"])
    if boost <= 0.0:
        return
    sid = str(int(speaker_id))
    name = ""
    by_id = getattr(sim, "agent_by_id", None)
    if by_id is not None:
        who = by_id.get(int(speaker_id))
        name = getattr(who, "name", "") or "" if who is not None else ""
    for slot in slots:
        if slot.get("kind") != "person":
            continue
        target = slot.get("target")
        if target == sid or (name and target == name):
            slot["salience"] = min(1.0, float(slot.get("salience", 0.0) or 0.0)
                                   + boost)


def phase_day(sim, step: int, sim_min: int) -> None:
    """日境界: 減衰(Ebbinghaus 的)+ `min_salience` 未満のスロットを落とす。決定論・乱数ゼロ。

    既定 OFF は即 return(state も L1 も 1 バイトも動かない)。
    """
    if not block_on(sim):
        return
    day = sim_min // 1440
    if day == getattr(sim, "_attn_block_day", -1):
        return
    sim._attn_block_day = day
    cfg = block_cfg_of(sim)
    keep = 1.0 - float(cfg["decay_per_day"])
    floor = float(cfg["min_salience"])
    for agent in getattr(sim, "agents", ()):
        slots = getattr(agent, SLOTS_KEY, None)
        if not slots:
            continue
        alive = []
        for slot in slots:
            sal = float(slot.get("salience", 0.0) or 0.0) * keep
            slot["salience"] = sal
            if sal >= floor:
                alive.append(slot)
        agent.attention_slots = alive


# --------------------------------------------------------------------------- #
# プロンプト節(層B ON のときだけ・固定位置 = ペルソナ直後)
#
# ★no-fingerprint: 機構語(注意機構・顕著性・閾値・発火・スロット・モデル・実験)も
#   実験条件語も因子名も 1 文字も書かない。書くのは「いま気にしていること」と
#   「JSON に attend を足してよい」だけ。
# ★Lost in the Middle 対策で**先頭側**に置く(設計 §3)。
# --------------------------------------------------------------------------- #
_DECL_LINE = ('JSON には attend を足してよい: '
              '[{"kind":"person|place|topic|goal","target":"…","why":"…"}]。'
              'いま気にかけているものを全部書く(書かなければ今のまま)。')


def prompt_section(sim, agent, nearby_names=None) -> str | None:
    """「いま気にしていること」節。**OFF は None**(1 行も足さない = バイト一致)。"""
    if not block_on(sim):
        return None
    lines = []
    slots = slots_of(agent)
    if slots:
        items = "、".join(
            f"{s['target']}({s['why']})" if s.get("why") else str(s["target"])
            for s in slots)
        lines.append(f"いま気にしていること: {items}")
    else:
        lines.append("いま気にしていること: 特にない")
    names = list(nearby_names or [])[:3]
    if names:
        lines.append(f"目の前にいる人: {'、'.join(names)}")
    topics = _recent_topics(agent)
    if topics:
        lines.append(f"耳に入っている言葉: {'、'.join(topics)}")
    lines.append(_DECL_LINE)
    return "\n".join(lines)


def _recent_topics(agent, n: int = 2) -> list:
    """よく耳にした語の上位 n 件(件数降順 → 語昇順の決定論。無ければ空)。"""
    counts = getattr(agent, "heard_counts", None)
    if not counts:
        return []
    ranked = sorted(counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
    return [str(k) for k, _v in ranked[:max(0, int(n))]]


# --------------------------------------------------------------------------- #
# プール回転の搬送(world/pool.py が写しを持つ。一致は tests/test_attention.py が固定)
# --------------------------------------------------------------------------- #
def slots_slim(agent) -> list | None:
    """dehydrate 用: スロット列を JSON 安全なプリミティブへ(空/未設定は None = キー無し)。"""
    slots = slots_of(agent)
    if not slots:
        return None
    return [{"kind": str(s.get("kind", "")), "target": str(s.get("target", "")),
             "why": str(s.get("why", "")),
             "salience": float(s.get("salience", 0.0) or 0.0),
             "since": int(s.get("since", 0) or 0)} for s in slots]


def slots_apply(agent, slim) -> None:
    """hydrate 用: 退避スロット列を個体へ戻す(空/None は属性を生やさない)。"""
    if not slim:
        return
    agent.attention_slots = [
        {"kind": str(s.get("kind", "")), "target": str(s.get("target", "")),
         "why": str(s.get("why", "")),
         "salience": float(s.get("salience", 0.0) or 0.0),
         "since": int(s.get("since", 0) or 0)} for s in slim]


def history_slim(agent) -> list | None:
    """dehydrate 用: 履歴(直近注意した話者 id)。空/未設定は None = キー無し。"""
    hist = getattr(agent, HISTORY_KEY, None)
    if not hist:
        return None
    return [int(x) for x in hist][-HISTORY_MAX:]


def history_apply(agent, slim) -> None:
    """hydrate 用: 履歴を戻す(空/None は属性を生やさない)。"""
    if not slim:
        return
    setattr(agent, HISTORY_KEY, [int(x) for x in slim][-HISTORY_MAX:])
