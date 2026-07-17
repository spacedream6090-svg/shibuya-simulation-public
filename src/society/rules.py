"""制度DSL(ホワイトリスト型ルール)。ユーザー構想 2026-07-06。

エージェント自身はコードを書けないが、**制度を作る行動(propose)をして成立したとき、
その仕組みをシミュ世界の実効ルールとして自動的に制定する**。自由なコード生成は非決定・
危険なので、機械可読の「制度スロット」= 4 型のホワイトリスト DSL に閉じる:

  1. fee        : 消費カテゴリの価格を ±円 で調整(economy.price_of が合算適用)
  2. bonus      : 特定行動(park/event_attend/flyer_view)に円を支給(区=発行)
  3. curfew     : 時間帯 × カテゴリの行き先選択を不透明係数で抑制(禁止でなく抑制)
  4. weekly_event: 期日ごとに world ニュース + その場所を余暇候補にブースト

原則(既存の鉄則を継承):
- 決定論: state は本モジュールに閉じ、iteration は list(挿入順)のみ(no set 反復)。
- R1(全員平等・k 非依存): ルールは提示・可否・効果とも k.writeback を見ない。
- R4(客観測定): 効果は客観カウント(価格・支給額・行き先分布)。LLM 審判は無い。
- 壊れない: 検証に失敗した rule は「文言だけの制度」に降格して成立(従来挙動)。
- OFF 不変: rules.enabled=false のとき全アクセサが恒等(価格・分布・支給が完全に不変)。

engine/cognition は本モジュールの不透明な数値(調整後価格・抑制係数・支給額)だけを
受け取り、因子名を一切見ない(no-fingerprint 契約に触れない)。
"""
from __future__ import annotations

import math

from .observer.schema import Event
from .world.clock import STEPS_PER_DAY

# ホワイトリスト(型ごとの許容値)
FEE_CATS = ("food", "cafe", "shop", "nightlife", "taxi", "bus")
BONUS_BEHAVIORS = ("park", "event_attend", "flyer_view")
CURFEW_CATS = ("nightlife", "leisure", "shop")
# prohibit(執行ルート Wave G3): 特定場所での行為を禁止(curfew=抑制 と違い執行対象)。
PROHIBIT_CATS = ("nightlife", "shop", "food", "leisure")


def build_rules_cfg(raw: dict | None) -> dict:
    raw = dict(raw or {})
    return {
        "enabled": bool(raw.get("enabled", True)),
        # 権利創設・宣言型(第9バッチ 制度深化。地域のパートナーシップ条例のモデル化)。既定 OFF=
        # declare 型は従来どおり検証失敗→文言制度へ降格(現行挙動と完全同一)。
        "allow_declare": bool(raw.get("allow_declare", False)),
        "max_active": int(raw.get("max_active", 5)),
        "duration_days": int(raw.get("duration_days", 0)),   # 0 = 無期限
        "fee_max_ratio": float(raw.get("fee_max_ratio", 0.5)),  # fee delta 上限=基準価格比
        "bonus_max": float(raw.get("bonus_max", 500)),          # bonus 額の上限(円)
        "boost_prob": float(raw.get("boost_prob", 0.5)),        # weekly place を選ぶ確率/日
    }


def _in_window(from_h: int, to_h: int, hour: int) -> bool:
    """[from_h, to_h) の時間帯に hour が入るか(from>to は日跨ぎ)。from==to は空。"""
    if from_h == to_h:
        return False
    if from_h < to_h:
        return from_h <= hour < to_h
    return hour >= from_h or hour < to_h


class RuleBook:
    """成立提案から制定された実効ルールのレジストリ + ライフサイクル。

    self に閉じた plain データのみ(sim 参照を持たない)= checkpoint に丸ごと同梱できる。
    """

    def __init__(self, cfg: dict, ref_prices: dict | None = None):
        self.cfg = cfg
        # fee delta の妥当性判定に使う基準価格(economy.prices + taxi/bus 初乗り)
        self.ref_prices: dict[str, float] = {str(k): float(v)
                                             for k, v in dict(ref_prices or {}).items()}
        self.active: list[dict] = []      # 実効ルール(挿入順=決定論)
        self._seq = 0
        self.n_enacted = 0                # 累計制定数(検証を通った活性化)
        self.n_repealed = 0               # 累計廃止数(再帰性 第9バッチ。repeal 提案の成立)
        self.last_day = -1                # 日次境界処理の進行管理
        self.today_boost: list[str] = []  # 今日ブーストする余暇候補ノード

    # ------------------------------------------------------------------ 検証
    def validate(self, rule) -> dict | None:
        """機械可読 rule を正準化(有効なら spec dict、無効なら None=文言制度へ降格)。"""
        if not isinstance(rule, dict):
            return None
        rtype = rule.get("type")
        if rtype == "fee":
            cat = rule.get("target_cat")
            if cat not in FEE_CATS:
                return None
            delta = _as_float(rule.get("delta"))
            if delta is None:
                return None
            ref = self.ref_prices.get(cat, 0.0)
            if ref <= 0 or abs(delta) > self.cfg["fee_max_ratio"] * ref:
                return None                       # 過大 delta / 無参照 → 降格
            return {"type": "fee", "target_cat": cat, "delta": delta}
        if rtype == "bonus":
            beh = rule.get("behavior")
            if beh not in BONUS_BEHAVIORS:
                return None
            amount = _as_float(rule.get("amount"))
            if amount is None or amount <= 0 or amount > self.cfg["bonus_max"]:
                return None
            return {"type": "bonus", "behavior": beh, "amount": amount}
        if rtype == "curfew":
            cat = rule.get("target_cat")
            if cat not in CURFEW_CATS:
                return None
            from_h = _as_int(rule.get("from_h"))
            to_h = _as_int(rule.get("to_h"))
            weight = _as_float(rule.get("weight"))
            if from_h is None or to_h is None or weight is None:
                return None
            if not (0 <= from_h <= 23 and 0 <= to_h <= 23 and 0.0 <= weight <= 1.0):
                return None
            return {"type": "curfew", "target_cat": cat, "from_h": from_h,
                    "to_h": to_h, "weight": weight}
        if rtype == "weekly_event":
            title = str(rule.get("title", "")).strip()
            place = str(rule.get("place", "")).strip()
            every = _as_int(rule.get("every_days"))
            if not title or not place or every is None or not (1 <= every <= 7):
                return None
            return {"type": "weekly_event", "title": title, "place": place,
                    "every_days": every}
        if rtype == "declare":
            # 権利創設・宣言型(制度深化 第9バッチ): 特定の承認・権利・理念を公式な取り決めとして
            # 樹立する。エンジンの価格・移動には作用しない**象徴的制度**(現実の例: 地域の
            # パートナーシップ条例)。再帰性の norm_line・institution_rule・ニュースで知覚され、
            # Y の symbolic/network 層で「ローカル制度の樹立・波及」として測る。既定 OFF。
            if not self.cfg["allow_declare"]:
                return None
            title = str(rule.get("title", "")).strip()
            if not title:
                return None
            return {"type": "declare", "title": title[:40]}
        if rtype == "prohibit":
            # 執行ルート(Wave G3): 特定カテゴリの場所での行為を禁止。curfew(行き先抑制の不透明
            # 係数)と違い、公務員(警察官)が近傍の違反者に罰を科す執行対象。既定は終日(0-24)。
            cat = rule.get("target_cat")
            if cat not in PROHIBIT_CATS:
                return None
            from_h = _as_int(rule.get("from_h", 0))
            to_h = _as_int(rule.get("to_h", 24))
            if from_h is None or to_h is None:
                return None
            if not (0 <= from_h <= 24 and 0 <= to_h <= 24):
                return None
            return {"type": "prohibit", "target_cat": cat, "from_h": from_h,
                    "to_h": to_h}
        return None

    # ------------------------------------------------------------------ 制定
    def enact(self, rule, *, name: str, proposer: int, step: int,
              day: int) -> dict | None:
        """rule を検証して活性化する。有効なら record を返し、無効なら None。"""
        if not self.cfg["enabled"]:
            return None
        spec = self.validate(rule)
        if spec is None:
            return None
        rid = self._seq
        self._seq += 1
        dur = self.cfg["duration_days"]
        expire_step = (step + dur * STEPS_PER_DAY) if dur > 0 else None
        record = {**spec, "id": rid, "name": name, "proposer": int(proposer),
                  "enacted_step": int(step), "enacted_day": int(day),
                  "expire_step": expire_step, "last_fired_day": None, "spent": 0.0}
        self.active.append(record)
        self.n_enacted += 1
        return record

    def evict_over_limit(self) -> list[dict]:
        """同時アクティブ上限を超えた分を古い順に失効させ、失効 record 列を返す。"""
        evicted: list[dict] = []
        while len(self.active) > self.cfg["max_active"]:
            evicted.append(self.active.pop(0))
        return evicted

    def public_rule(self, r: dict) -> dict:
        """ログ用に DSL スロットだけを抜き出す(bookkeeping を除く)。"""
        t = r["type"]
        if t == "fee":
            return {"type": t, "target_cat": r["target_cat"], "delta": r["delta"]}
        if t == "bonus":
            return {"type": t, "behavior": r["behavior"], "amount": r["amount"]}
        if t == "curfew":
            return {"type": t, "target_cat": r["target_cat"], "from_h": r["from_h"],
                    "to_h": r["to_h"], "weight": r["weight"]}
        if t == "prohibit":
            return {"type": t, "target_cat": r["target_cat"],
                    "from_h": r["from_h"], "to_h": r["to_h"]}
        if t == "declare":
            return {"type": t, "title": r["title"]}
        return {"type": t, "title": r["title"], "place": r["place"],
                "every_days": r["every_days"]}

    # ------------------------------------------------------------------ 日次境界
    def advance_day(self, day: int, step: int):
        """新しい日なら (expired, weekly_due) を返す。同日なら None(何もしない)。"""
        if day == self.last_day:
            return None
        self.last_day = day
        self.today_boost = []
        expired: list[dict] = []
        kept: list[dict] = []
        for r in self.active:
            if r["expire_step"] is not None and step >= r["expire_step"]:
                expired.append(r)
            else:
                kept.append(r)
        self.active = kept
        weekly: list[dict] = []
        for r in self.active:
            if r["type"] != "weekly_event" or r["last_fired_day"] == day:
                continue
            since = day - r["enacted_day"]
            if since > 0 and since % r["every_days"] == 0:
                r["last_fired_day"] = day
                weekly.append(r)
        return expired, weekly

    # ------------------------------------------------------------------ 実効アクセサ
    def fee_price(self, cat: str, base: float) -> float:
        """アクティブな fee ルールを合算して cat の価格を調整(±fee_max_ratio でクランプ)。

        該当 fee ルールが無ければ base をそのまま返す(=OFF・無ルール時は完全に不変)。
        """
        if not self.cfg["enabled"]:
            return base
        delta = 0.0
        for r in self.active:
            if r["type"] == "fee" and r["target_cat"] == cat:
                delta += r["delta"]
        if delta == 0.0:
            return base
        ratio = self.cfg["fee_max_ratio"]
        lo = max(0.0, base * (1.0 - ratio))
        hi = base * (1.0 + ratio)
        return min(hi, max(lo, base + delta))

    def has_curfew(self) -> bool:
        return any(r["type"] == "curfew" for r in self.active)

    def curfew_weight(self, hour: int, dest_cats: set) -> float:
        """hour・dest カテゴリに掛かる curfew 係数の積(該当なし=1.0=抑制なし)。"""
        w = 1.0
        for r in self.active:
            if r["type"] == "curfew" and r["target_cat"] in dest_cats \
                    and _in_window(r["from_h"], r["to_h"], hour):
                w *= r["weight"]
        return w

    def bonus_rules(self, behavior: str) -> list[dict]:
        return [r for r in self.active
                if r["type"] == "bonus" and r["behavior"] == behavior]

    # ------------------------------------------------------------------ 執行(Wave G3)
    def has_prohibit(self) -> bool:
        return any(r["type"] == "prohibit" for r in self.active)

    def prohibited_cats(self, hour: int) -> set:
        """今この時刻に禁止されている場所カテゴリの集合(該当なし=空=執行なし)。決定論。"""
        out = set()
        for r in self.active:
            if r["type"] == "prohibit" and _in_window(r["from_h"], r["to_h"], hour):
                out.add(r["target_cat"])
        return out

    def first_prohibit(self, cat: str):
        """カテゴリ cat を禁止する最初の(=最古の)prohibit ルール id(無ければ None)。"""
        for r in self.active:
            if r["type"] == "prohibit" and r["target_cat"] == cat:
                return r["id"]
        return None

    # ------------------------------------------------------------------ 再帰性: 廃止(第9バッチ)
    def find_active(self, *, rule_id=None, name=None) -> dict | None:
        """廃止対象の解決: rule_id(完全一致)を優先し、無ければ name の部分一致(最古優先)。

        ルール名は提案文の先頭24字なので、提案者が全文を引用しても拾えるよう部分一致は双方向。
        決定論(active は挿入順の list)。見つからなければ None(呼び出し側が文言制度へ降格)。"""
        rid = _as_int(rule_id) if rule_id is not None else None
        if rid is not None:
            for r in self.active:
                if r["id"] == rid:
                    return r
            return None
        text = str(name or "").strip()
        if not text:
            return None
        for r in self.active:
            if text in r["name"] or r["name"] in text:
                return r
        return None

    def repeal(self, record) -> None:
        """record を active から除く(廃止)。ログ・ニュース配信は呼び出し側(tools)が担う。"""
        self.active = [r for r in self.active if r is not record]
        self.n_repealed += 1


def _as_float(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _as_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def apply_bonus(rulebook: RuleBook | None, sim, agent, behavior: str,
                step: int, sim_min: int) -> float:
    """behavior に該当する bonus ルールをすべて agent へ支給(区=発行、累計を記録)。

    rules 無効/該当ルール無しなら副作用ゼロ(ログも金額も動かさない=OFF 不変)。
    """
    if rulebook is None or not rulebook.cfg["enabled"]:
        return 0.0
    paid = 0.0
    for r in rulebook.active:
        if r["type"] != "bonus" or r["behavior"] != behavior:
            continue
        amt = float(r["amount"])
        agent.money += amt
        r["spent"] += amt
        paid += amt
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                             kind="rule_bonus", x=agent.x, y=agent.y,
                             payload={"rule_id": r["id"], "behavior": behavior,
                                      "amount": round(amt, 1),
                                      "balance": round(agent.money, 1)}))
    return paid
