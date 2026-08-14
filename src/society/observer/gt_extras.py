"""G7 小物束(``observer.gt_extras``・**既定 OFF**)= 「射影で落ちている正解ラベル」5 件。

正典: docs/plans/metaverse-projection-plan.md §1「残らないもの」/ §4 G7。

塞ぐ穴(どれも「世界は持っているのに成果物に出ない」)
------------------------------------------------------
  ① ``plan_created`` の payload は blocks の 7 欄しか射影せず、**計画の気分 mood /
     前日からの持ち越し carry / ブロックごとの同行者 with** を落としている
     (= 「なぜその日をそう設計したか」が事後に読めない)
  ② ``reflect`` の payload は自己モデルの**更新有無 bool** しか出さず、
     自己像の本文(self / ties。どちらも 80 字上限で既に短い)を落としている
  ③ ``emotion_label`` の payload はラベルだけで、プロンプトへ実際に注入された
     **感情句**(同じラベルでも文面が違う)を落としている
  ④ ``traits.json`` には needs の**5 軸プロファイル**も **needs_mods**(reason→倍率)も
     載らない = 「何を欲しやすい個体か」という静的ラベルが名簿に無い
  ⑤ 日次 ``worldview`` イベントは期待表の**件数**しか出さず、期待の**中身**を落としている

規律(R1)
---------
- **1 つのトグル** ``observer.gt_extras.enabled``(既定 OFF)で 5 件まとめて切る。
  OFF では payload に一時キーすら積まれない = **L1 バイト一致**(golden 無風)。
- 足すのは**既存イベントの payload キー**と ``traits.json`` の欄だけ。新しいイベント種も
  新しいファイルも作らない(= 事後解析の入力形が変わらない)。
- 読むだけ: 世界状態を 1 バイトも書き換えず、乱数を 1 粒も引かず、LLM 呼数を 1 も変えず、
  **プロンプトを 1 バイトも変えない**(当人から観測できる差分が存在しない)。
- 本 module は因子名も価値次元名も**綴らない**(no-fingerprint 契約)。needs の 5 軸は
  ``agent._needs_profile`` の**キーをそのまま**書き出す(factors 層が唯一の源)。

規模: finals 構成(25 万 × 10 日)で **+500 MB 前後**(大半は ⑤ の期待表 上位 k 件)。
"""
from __future__ import annotations

#: 日次 worldview に載せる期待表の件数(⑤)。全部載せると場所×時間帯の全格子になる。
DEFAULT_EXPECT_TOP_K = 8


def build_cfg(raw) -> dict:
    """conf の ``observer.gt_extras`` を正準化(既定 OFF)。

    ★``isinstance(raw, dict)`` で判定しないこと: conf は OmegaConf の ``DictConfig`` で、
      dict の部分型ではない(素の dict しか受けない書き方だと **ON にしても OFF に落ちる**)。
    """
    cfg = {"enabled": False, "expect_top_k": DEFAULT_EXPECT_TOP_K}
    if raw is not None and hasattr(raw, "get"):
        cfg["enabled"] = bool(raw.get("enabled", False))
        cfg["expect_top_k"] = max(0, int(raw.get("expect_top_k",
                                                 DEFAULT_EXPECT_TOP_K) or 0))
    return cfg


def cfg_of_config(config) -> dict:
    """``sim.cfg`` から本層の設定を引く(未宣言の旧 config でも既定 OFF に落ちる)。"""
    try:
        obs = config.get("observer", None) or {}
        raw = obs.get("gt_extras", None)
    except Exception:                              # noqa: BLE001(旧 config 互換)
        raw = None
    return build_cfg(raw)


def enabled(sim) -> bool:
    """本層が有効か(``sim`` が cfg を持たないスタブでも False に落ちる)。

    ``Simulation`` が構築時に据える ``sim.gt_extras`` を使い、無ければ conf から作り直す。
    """
    cfg = getattr(sim, "gt_extras", None)
    if cfg is None:
        cfg = cfg_of_config(getattr(sim, "cfg", None) or {})
    return bool(cfg["enabled"])


def top_k(sim) -> int:
    cfg = getattr(sim, "gt_extras", None)
    if cfg is None:
        cfg = cfg_of_config(getattr(sim, "cfg", None) or {})
    return int(cfg["expect_top_k"])


# --------------------------------------------------------------------------- #
# ① 朝の計画(plan_created)
# --------------------------------------------------------------------------- #
def plan_extras(plan: dict, blocks: list) -> dict:
    """``plan_created`` payload への追記キー(mood / carry / withs)。

    ``withs`` は **blocks と同じ添字**の同行者リスト列(``[[id, ...], ...]``)。ブロック 1 件が
    dict に混ざると添字がずれるので、独立した 1 キーとして並べる。同行者を 1 人も持たない
    計画では ``withs`` を足さない(既定的な計画で payload が太らないようにする)。
    """
    out: dict = {}
    mood = str(plan.get("mood", "") or "")
    carry = str(plan.get("carry", "") or "")
    if mood:
        out["mood"] = mood
    if carry:
        out["carry"] = carry
    withs = [[str(w) for w in (b.get("with") or [])] for b in blocks]
    if any(withs):
        out["withs"] = withs
    return out


# --------------------------------------------------------------------------- #
# ② 内省(reflect)
# --------------------------------------------------------------------------- #
def reflect_extras(self_model) -> dict:
    """``reflect`` payload への追記キー(self / ties の本文)。

    本文はどちらも生成時点で 80 字に切られている(``cognition/reflection.py``)ので、
    ここで再度切らない(切ると「切られた」ことが 2 箇所の仕様になる)。
    """
    if not isinstance(self_model, dict):
        return {}
    out: dict = {}
    for key in ("self", "ties"):
        text = str(self_model.get(key, "") or "")
        if text:
            out[key] = text
    return out


# --------------------------------------------------------------------------- #
# ③ 感情ラベル(emotion_label)
# --------------------------------------------------------------------------- #
def emotion_extras(phrase: str) -> dict:
    """``emotion_label`` payload への追記キー(プロンプトへ入る感情句そのもの)。"""
    text = str(phrase or "")
    return {"phrase": text} if text else {}


# --------------------------------------------------------------------------- #
# ④ needs プロファイル(traits.json)
# --------------------------------------------------------------------------- #
def needs_extras(agent) -> dict:
    """``traits.json`` の 1 個体レコードへの追記キー(5 軸プロファイル + reason 倍率)。

    ★次元名も reason 名も**コードに綴らない**: 元の dict のキーをそのまま写す
      (factors 層 / needs 層が唯一の源 = no-fingerprint 契約)。needs OFF の個体では
      どちらの属性も無いので **1 キーも足さない**(= 既存 traits.json とバイト一致)。
    """
    out: dict = {}
    profile = getattr(agent, "_needs_profile", None)
    if isinstance(profile, dict) and profile:
        out["needs"] = {str(k): round(float(v), 4) for k, v in sorted(profile.items())}
    mods = getattr(agent, "needs_mods", None)
    if isinstance(mods, dict) and mods:
        out["needs_mods"] = {str(k): round(float(v), 4) for k, v in sorted(mods.items())}
    return out


# --------------------------------------------------------------------------- #
# ⑤ 世界観の期待表(worldview 日次イベント)
# --------------------------------------------------------------------------- #
def expect_extras(agent, k: int) -> dict:
    """日次 ``worldview`` payload への追記キー(期待表の上位 k 件)。

    期待表 ``wv_expect`` は ``{(場所キー, 時間帯): 期待人数}``。「上位」は**期待人数の
    降順**(= その人が一番混むと思っている場所)で、同値は (場所キー, 時間帯) の昇順で
    解く(決定論)。値は ``[[場所キー, 時間帯, 期待人数], ...]`` の 3 つ組列で出す
    (タプルキーは JSON に載らないため)。
    """
    exp = getattr(agent, "wv_expect", None)
    if not exp or k <= 0:
        return {}
    items = sorted(exp.items(), key=lambda kv: (-float(kv[1]), str(kv[0][0]),
                                                int(kv[0][1])))[:k]
    return {"expect": [[str(key[0]), int(key[1]), round(float(v), 3)]
                       for key, v in items]}
