"""Perception / Intent 契約(第85バッチ 2026-08-01)= エージェントコアと世界の結合面。

正典: docs/plans/source/physics-instructions.md **Part P1** /
      docs/plans/cognition-physics-plan.md §4 第85行。

何を解く問題か
--------------
現状、1 回の思考に渡る世界情報は `engine/scheduler.py::_llm_speak` が 40 個以上の
ばらばらの局所変数として集め、`deliberate.build_prompt` へ **49 個のキーワード引数**
として流し込んでいる。結合面が「引数リスト」でしかないため、

  - 物理エンジン(P3)が「知覚に何を足せばよいか」の着地点が無い、
  - 「世界の情報がどこからプロンプトへ入ったか」を機械的に検査できない、
  - 思考の出力(行動 dict)も型が無く、経路や速度を書き込む余地が構造的に残る、

という 3 つの問題がある。本 module はその結合面を **2 つのデータ型**に限定する:

  Perception … 1 回の思考が受け取る世界情報の**唯一の型**(世界 → 認知)
  Intent     … 1 回の思考が出す意図の**唯一の型**(認知 → 世界)

P1 の要求(この時点では挙動は変わらない)
------------------------------------------
「Perception の中身が従来システムが渡していた情報リストと等価になるだけ」。
したがって本 module の中心的な契約は次の 2 つの**無損失性**である:

  1. `Perception.prompt_kwargs()` が `build_prompt` へ渡していた dict と**完全に等しい**
     (= 契約経路 ON でもプロンプト文字列が 1 バイトも変わらない)
  2. `Intent.from_action(a).to_action() == a`(= 行動 dict の情報が 1 つも落ちない)

この 2 つが成り立つ限り、契約 ON/OFF で世界状態は一致する。一致を機械で固定したうえで、
以後の Part(P3 物理縫合)は**契約経路の上にだけ**積む。

責務の分界(重要)
------------------
- **Perception の生成は世界側の責務**。本 module は「型と純関数」しか持たない。
  生成関数は `engine/scheduler.py::build_perception` **1 本**に集約する。
- 本 module は `sim`(世界オブジェクト)を **1 度も参照しない**。
  tests/test_perception_contract.py が識別子 `sim` の不在を静的に固定する。
- 本 module は真偽台帳(第73)を import も参照もしない。漏洩検査は
  `text_blob()` が返す「プロンプトに出うる文字列材料」を engine 側が既存の canary 関門へ
  通すことで統合する(台帳の識別子は cognition/ に 1 つも現れてはならない)。

salience(顕著性)
------------------
各項目の salience は **第80バッチ channels の値を再利用**する(二重計算しない)。
具体的には `cognition/fire.py::terms_of` が出す σ 正規化誤差 |o−ô|/σ を、
`SALIENCE_MAP` でチャンネル id → Perception の項目名へ写す。
チャンネル機構が OFF のラン(= 観測が無いラン)では **salience は空**にする。
0.0 で埋めない(欠測は欠測と判る値にする = 本リポジトリの既定則)。

R1 ドクトリンとの関係
---------------------
- 既定 `cognition.contract.enabled: false` では本 module のどの関数も **1 度も呼ばれない**
  (scheduler が従来経路の分岐に入る)= L1/L2/L3・乱数・LLM 呼数のいずれも不変。
- ON でも乱数を 1 本も引かず、LLM を 1 本も呼ばず、プロンプト文字列を 1 バイトも変えない
  (= 情報の通り道だけが変わる)。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, fields
from typing import Any, Mapping, Sequence

SCHEMA = 1


# --------------------------------------------------------------------------- #
# cfg 正準化(既定 OFF)
# --------------------------------------------------------------------------- #
def build_cfg(raw: dict | None) -> dict:
    """conf の `cognition.contract` ブロックを型強制つきで正準化する(既定 OFF)。"""
    raw = dict(raw or {})
    return {"enabled": bool(raw.get("enabled", False))}


# --------------------------------------------------------------------------- #
# Perception(世界 → 認知)
#
# ★ `build_prompt` のキーワード名との対応は下の `_KW_FIELDS` が**唯一の源**。
#    ここに書いていないキーワードは契約経路を通れない(テストが署名との突合を固定する)。
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Perception:
    """1 回の思考が受け取る世界情報の唯一の型。

    グループ分けは physics-instructions.md P1(1) の 4 分類に従う:
      視覚 / 聴覚 / 身体 / 内部状態 + 既存のプロンプト材料(環境・社会文脈)。

    ★エージェント自身が持っている情報(ペルソナ・記憶・信念・気分)は
      Perception には入れない。それは「世界から受け取るもの」ではなく
      「自分の中にあるもの」で、`build_prompt` が agent から直接読む(従来どおり)。
    """

    # ---- 視覚: 見えている場所・POI・人・掲示 --------------------------------- #
    place_name: str = ""
    nearby_pois: tuple[str, ...] = ()
    nearby_names: tuple[str, ...] = ()
    nearby_ids: tuple[int, ...] = ()
    scene_lines: tuple[str, ...] | None = None
    crowd_line: str | None = None
    signage_line: str | None = None          # 街頭掲示の概況(従来の ads_line)
    place_label_line: str | None = None
    trace_line: str | None = None            # 場所に残った痕跡の 1 行(第96 IF-D)
    familiar_places: tuple[str, ...] | None = None

    # ---- 聴覚: 聞こえた発話とその文脈 --------------------------------------- #
    heard_reply: tuple[str, str] | None = None      # (話者名, 発話) = 従来の reply_to
    dialog_history: tuple[Any, ...] | None = None
    feed_texts: tuple[str, ...] | None = None
    dm_target: str | None = None

    # ---- 身体: 位置・移動の成否・混雑による遅延・接触 ------------------------ #
    #   ★ P3(物理縫合)の着地点。現行のグラフ世界では位置と同席数までしか無い。
    #     **プロンプトには 1 バイトも出さない**(= 観測・下流の物理写像専用)。
    body: Mapping[str, Any] = field(default_factory=dict)

    # ---- 内部状態: ニーズ・時刻感覚 ----------------------------------------- #
    #   ★ 同上。build_prompt は agent から直接読むので、ここは物理/発火側の入口。
    internal: Mapping[str, Any] = field(default_factory=dict)

    # ---- 時刻・きっかけ ----------------------------------------------------- #
    step: int = 0
    sim_min: int = 0
    trigger: str | None = None               # 従来の surprise

    # ---- 環境・社会文脈(既存のプロンプト材料) ------------------------------ #
    date_line: str | None = None
    weather_line: str | None = None
    schedule_line: str | None = None
    event_line: str | None = None
    disaster_line: str | None = None
    institutions: tuple[str, ...] | None = None
    norm_line: str | None = None
    digest_line: str | None = None
    interstitial_digest: str | None = None
    wv_expect_line: str | None = None
    wv_self_line: str | None = None
    wv_norm_line: str | None = None
    relation_line: str | None = None
    reputation_line: str | None = None
    household_line: str | None = None
    diversity_line: str | None = None
    emotion_line: str | None = None
    goal_line: str | None = None
    hobby_line: str | None = None
    memberships: tuple[str, ...] | None = None
    tool_offers: str | None = None
    p2_offers: str | None = None
    watch_section: str | None = None
    revision_line: str | None = None
    engaged_section: str | None = None
    reject_line: str | None = None
    pull_query: str | None = None

    # ---- 提示規約(世界がプロンプトの**形**を決める定数。run 内で不変) ------- #
    labeling_mode: str = "constrained"
    open_actions: bool = False
    explicit_nothing: bool = False
    verify_actions: bool = False
    equip_all: bool = False
    venture_cost: float = 30000.0
    city_name: str = ""
    variety_hint: bool = False
    # V-P1(既定 None = 現行のプロンプトとバイト一致)。「このプロンプトはどの用途か」の札で、
    # 世界(conf)が run 内で決める提示規約の一種なのでここに置く。値そのものは
    # `prompt_p1.purpose_for(sim, "deliberate")` = ON なら "deliberate" / OFF なら None。
    p1_purpose: str | None = None

    # ---- salience(第80 channels の σ 正規化誤差を再利用。二重計算しない) ---- #
    salience: Mapping[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # 変換
    # ------------------------------------------------------------------ #
    @classmethod
    def from_material(cls, material: Mapping[str, Any], *,
                      body: Mapping[str, Any] | None = None,
                      internal: Mapping[str, Any] | None = None,
                      salience: Mapping[str, float] | None = None) -> "Perception":
        """世界側が集めた材料(build_prompt のキーワード dict)を契約型へ包む。

        ★ここが「世界 → Perception」の唯一の入口。呼ぶのは engine/scheduler だけ。
        未知のキーは**黙って捨てない**(契約から漏れる情報を作らないため即エラー)。
        """
        unknown = sorted(set(material) - set(_FIELD_OF_KW))
        if unknown:
            raise KeyError(
                f"Perception 契約に無いプロンプト材料: {unknown}。"
                " build_prompt にキーワードを足したら _KW_FIELDS にも足すこと"
                "(契約の外から世界情報がプロンプトへ入る穴を作らない)。")
        kwargs: dict[str, Any] = {}
        for kw, fname, seq in _KW_FIELDS:
            if kw not in material:
                continue
            value = material[kw]
            kwargs[fname] = tuple(value) if (seq and value is not None) else value
        return cls(body=dict(body or {}), internal=dict(internal or {}),
                   salience=dict(salience or {}), **kwargs)

    def prompt_kwargs(self) -> dict[str, Any]:
        """`build_prompt` へ渡すキーワード dict を**再構成**して返す。

        ★契約の中心。`from_material` に渡した dict と**完全に等しい**ものが返る
          (= 契約経路 ON でもプロンプト文字列は 1 バイトも変わらない)。
        ★body / internal / salience は**ここに出さない**。物理由来の項目が
          プロンプトへ裏口から入らないことを構造で保証する(P3 の no-fingerprint)。
        """
        out: dict[str, Any] = {}
        for kw, fname, seq in _KW_FIELDS:
            value = getattr(self, fname)
            out[kw] = list(value) if (seq and value is not None) else value
        return out

    # ------------------------------------------------------------------ #
    # salience(第80 channels 由来。無ければ None = 欠測)
    # ------------------------------------------------------------------ #
    def salience_of(self, field_name: str) -> float | None:
        """項目 1 つの顕著性。観測が無いラン・対応チャンネルが無い項目は None。"""
        value = self.salience.get(field_name)
        return None if value is None else float(value)

    def top_salience(self, n: int = 3) -> list[tuple[str, float]]:
        """顕著な項目の上位 n 件(降順・同点は項目名昇順=決定論)。"""
        ranked = sorted(self.salience.items(), key=lambda kv: (-float(kv[1]), kv[0]))
        return [(k, float(v)) for k, v in ranked[:max(0, int(n))]]

    # ------------------------------------------------------------------ #
    # 漏洩検査用のテキスト材料(engine 側の canary 関門へ渡す)
    # ------------------------------------------------------------------ #
    def text_blob(self) -> str:
        """プロンプトに出うる**文字列材料**をすべて連結して返す(検査専用・副作用なし)。

        真偽台帳の値がここに混ざっていれば、engine 側の既存 canary 関門が止める。
        本 module は台帳の識別子を 1 つも綴らない(検査の呼び出しは engine 側)。
        """
        parts: list[str] = []
        for kw, fname, _seq in _KW_FIELDS:
            _collect_text(getattr(self, fname), parts)
        return "\n".join(parts)


def _collect_text(value: Any, out: list[str]) -> None:
    """文字列(および文字列を含む列)を平坦に集める。数値・真偽値は対象外。"""
    if isinstance(value, str):
        if value:
            out.append(value)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _collect_text(item, out)


# --------------------------------------------------------------------------- #
# build_prompt のキーワード ↔ Perception の項目(**唯一の対応表**)
#   (build_prompt のキーワード名, Perception の項目名, 列型か)
#   列型 = 元が list / None の項目。round-trip で list に戻す(型まで従来と同一にする)。
# --------------------------------------------------------------------------- #
_KW_FIELDS: tuple[tuple[str, str, bool], ...] = (
    # 視覚
    ("place_name", "place_name", False),
    ("nearby_pois", "nearby_pois", True),
    ("nearby_names", "nearby_names", True),
    ("nearby_ids", "nearby_ids", True),
    ("scene_lines", "scene_lines", True),
    ("crowd_line", "crowd_line", False),
    ("ads_line", "signage_line", False),
    ("place_label_line", "place_label_line", False),
    ("trace_line", "trace_line", False),
    ("familiar_places", "familiar_places", True),
    # 聴覚
    ("reply_to", "heard_reply", False),
    ("dialog_history", "dialog_history", True),
    ("feed_texts", "feed_texts", True),
    ("dm_target", "dm_target", False),
    # 時刻・きっかけ
    ("step", "step", False),
    ("sim_min", "sim_min", False),
    ("surprise", "trigger", False),
    # 環境・社会文脈
    ("date_line", "date_line", False),
    ("weather_line", "weather_line", False),
    ("schedule_line", "schedule_line", False),
    ("event_line", "event_line", False),
    ("disaster_line", "disaster_line", False),
    ("institutions", "institutions", True),
    ("norm_line", "norm_line", False),
    ("digest_line", "digest_line", False),
    ("interstitial_digest", "interstitial_digest", False),
    ("wv_expect_line", "wv_expect_line", False),
    ("wv_self_line", "wv_self_line", False),
    ("wv_norm_line", "wv_norm_line", False),
    ("relation_line", "relation_line", False),
    ("reputation_line", "reputation_line", False),
    ("household_line", "household_line", False),
    ("diversity_line", "diversity_line", False),
    ("emotion_line", "emotion_line", False),
    ("goal_line", "goal_line", False),
    ("hobby_line", "hobby_line", False),
    ("memberships", "memberships", True),
    ("tool_offers", "tool_offers", False),
    ("p2_offers", "p2_offers", False),
    ("watch_section", "watch_section", False),
    ("revision_line", "revision_line", False),
    ("engaged_section", "engaged_section", False),
    ("reject_line", "reject_line", False),
    ("pull_query", "pull_query", False),
    # 提示規約
    ("labeling_mode", "labeling_mode", False),
    ("open_actions", "open_actions", False),
    ("explicit_nothing", "explicit_nothing", False),
    ("verify_actions", "verify_actions", False),
    ("equip_all", "equip_all", False),
    ("venture_cost", "venture_cost", False),
    ("city_name", "city_name", False),
    ("variety_hint", "variety_hint", False),
    ("p1_purpose", "p1_purpose", False),
)

_FIELD_OF_KW: dict[str, str] = {kw: fname for kw, fname, _s in _KW_FIELDS}
PROMPT_KEYWORDS: tuple[str, ...] = tuple(kw for kw, _f, _s in _KW_FIELDS)

# プロンプトへ**決して出さない**項目(物理・観測専用)。テストが固定する。
NON_PROMPT_FIELDS: tuple[str, ...] = ("body", "internal", "salience")


# --------------------------------------------------------------------------- #
# salience の写像(チャンネル id → Perception の項目名)
#
# ★第80 channels の σ 正規化誤差をそのまま使う(二重計算しない)。
# ★`body.state.*`(factors 層のキーを含む)は**写さない**: 個別の項目に対応しないうえ、
#   因子名を認知層の表に綴らないため(no-fingerprint 契約)。身体の総体は "internal" へ。
# --------------------------------------------------------------------------- #
SALIENCE_MAP: dict[str, tuple[str, ...]] = {
    "ext.crowd_local": ("crowd_line", "nearby_names"),
    "ext.encounter": ("nearby_names", "nearby_ids"),
    "ext.heard": ("dialog_history", "heard_reply"),
    "ext.signage": ("signage_line",),
    "ext.transit_delay": ("disaster_line",),
    "ext.weather_temp": ("weather_line",),
    "body.drive": ("internal",),
    "body.boredom": ("internal",),
    "body.fatigue": ("internal",),
    "body.arousal": ("internal",),
}


def salience_from_channels(errs_by_channel: Mapping[str, float]) -> dict[str, float]:
    """チャンネル別の σ 正規化誤差 → Perception 項目別の salience(決定論)。

    1 項目に複数チャンネルが写るときは **最大**を採る(注意を引くのは最も外れた源)。
    写像先の無いチャンネルは捨てる(捏造しない)。入力が空なら空 dict(= 欠測)。
    """
    out: dict[str, float] = {}
    for cid in sorted(errs_by_channel):
        value = errs_by_channel[cid]
        if value is None:
            continue
        for fname in SALIENCE_MAP.get(cid, ()):
            got = float(value)
            if fname not in out or got > out[fname]:
                out[fname] = got
    return out


# --------------------------------------------------------------------------- #
# Intent(認知 → 世界)
# --------------------------------------------------------------------------- #
# 意図から**物理的な実行の詳細**を読み取ろうとするキー名(P1(2) の禁止事項)。
# テストが Intent の項目名にこれらが現れないことを固定する。
FORBIDDEN_INTENT_TERMS: tuple[str, ...] = (
    "path", "route", "waypoint", "trajectory", "velocity", "speed", "heading",
    "accel", "dt", "tick", "step_xy",
)

_MOVE_KEYS = ("where", "area", "place", "destination")
_TOPIC_KEYS = ("about", "topic", "subject", "purpose")
_PARTNER_KEYS = ("to", "target", "partner")
_STAY_KINDS = ("stay",)


@dataclass(frozen=True)
class Intent:
    """1 回の思考が出す意図の唯一の型(= `parse_action` の出力の型化)。

    **物理的な実行の詳細(経路の各点・速度の時系列)は含めない**(P1(2))。
    実行は世界側・物理側の責務で、Intent が渡すのは「どこへ / どれくらい急いで /
    人混みをどう扱って / 誰と何を / 留まるか」までである。

    `payload` は行動の内容そのもの(発話文・造語・提案文など)。ここを持つことで
    `to_action()` が元の行動 dict を**完全に**復元できる(情報等価)。
    """

    kind: str
    move_poi: str | None = None
    move_xy: tuple[float, float] | None = None
    urgency: float = 0.0
    avoidance: float = 0.0
    talk_to: str | None = None
    topic: str | None = None
    stay: bool = False
    payload: Mapping[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_action(cls, action: Mapping[str, Any] | None) -> "Intent | None":
        """`deliberate.parse_action` の出力(行動 dict)を Intent へ写す。

        None(解釈不能=沈黙)はそのまま None を返す(呼び出し側の分岐を変えない)。
        """
        if action is None:
            return None
        if not isinstance(action, Mapping) or "type" not in action:
            raise TypeError(f"行動 dict ではない: {type(action).__name__}")
        kind = str(action["type"])
        payload = {k: v for k, v in action.items() if k != "type"}
        return cls(
            kind=kind,
            move_poi=_first_str(payload, _MOVE_KEYS),
            move_xy=_xy_of(payload),
            urgency=_num(payload.get("urgency")),
            avoidance=_num(payload.get("avoidance")),
            talk_to=_first_str(payload, _PARTNER_KEYS),
            topic=_first_str(payload, _TOPIC_KEYS),
            stay=kind in _STAY_KINDS,
            payload=payload,
        )

    def to_action(self) -> dict[str, Any]:
        """行動 dict へ戻す(**完全復元**: from_action の入力と等しい)。"""
        return {"type": self.kind, **dict(self.payload)}


def _first_str(payload: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _xy_of(payload: Mapping[str, Any]) -> tuple[float, float] | None:
    """座標指定の移動目標(現行の行動スキーマには存在しない = 常に None)。

    P3 で物理領域内の目標点を渡す口として先に開けておく枠。**捏造しない**ので、
    x/y が数値で揃っているときだけ値を持つ。
    """
    x, y = payload.get("x"), payload.get("y")
    if isinstance(x, bool) or isinstance(y, bool):
        return None
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return (float(x), float(y))
    return None


def _num(value: Any) -> float:
    """急ぎ度・回避傾向。現行の行動スキーマには欄が無いので既定 0.0(捏造しない)。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


# --------------------------------------------------------------------------- #
# 自己点検(import 時に 1 度。契約が壊れた状態で世界が走らないようにする)
# --------------------------------------------------------------------------- #
def _self_check() -> None:
    names = {f.name for f in fields(Perception)}
    mapped = {fname for _kw, fname, _s in _KW_FIELDS}
    missing = mapped - names
    if missing:
        raise RuntimeError(f"Perception に無い項目が対応表にある: {sorted(missing)}")
    if set(NON_PROMPT_FIELDS) & mapped:
        raise RuntimeError("物理・観測専用の項目がプロンプト対応表に混ざっている")
    intent_names = " ".join(f.name for f in fields(Intent))
    bad = [t for t in FORBIDDEN_INTENT_TERMS
           if re.search(rf"(^|_){re.escape(t)}(_|$)", intent_names)]
    if bad:
        raise RuntimeError(f"Intent が物理的な実行の詳細を持っている: {bad}")


_self_check()
