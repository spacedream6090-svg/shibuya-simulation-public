"""再帰性(reflexivity)= 規範・現状の監視 → 知覚 → フィードバック → 改変(第9バッチ 2026-07-07)。

これまでの実装では、制度DSL のルールは価格・行き先抑制・支給としてエンジン側で「静かに」
効き、エージェント(LLM)は自分が受けた影響(取り締まり・価格)を通してしか制度を知れなかった。
本モジュールは既定 OFF の4機構で「エージェントが世界の規範や現状を監視・知覚し、その
フィードバックから世界を変えようと行動できる」閉ループを完成させる:

  1) 規範の知覚(rules_in_prompt): いま実効の取り決めを全発火プロンプトに1行注入
     (全員平等・k 非依存・内容のみ=呼数不変)。
  2) 現状の知覚(digest_in_prompt): 昨日の街の動き(提案/成立/廃止/取り締まりの客観
     カウント)を1行注入。カウントは R4(客観測定)の範囲=因子・内面状態を一切読まない。
  3) 改変行動の完結(allow_repeal): propose の rule に {"type":"repeal","target":"名前の一部"}
     を許し、成立時に既存ルールを廃止する(rule_repealed)。監視→不満→廃止の再帰が一周する。
  4) 世界の自己観測(impact_news): 取り締まりが多発したルールを日次でニュース化
     (publish_news)。社会が自分の制度の帰結を報道として観測する(メディア=社会の鏡)。

原則: 非LLM・決定論・乱数不使用(新 stream も引かない)。R1: generate() を1本も足さない
(注入は内容のみ=呼数不変)。impact_news は聞かれる語が変わり発火が動きうるが、機構は
k・内面状態を読まない=compute_matched 下の k 不変性で担保(G4 群集と同型)。
既定 OFF=注入なし・rule_repealed/norm_digest 0 件・カウンタ据置=現行挙動と完全同一
(ゴールデン golden_baseline_l1.json を守る)。状態は plain データのみ=checkpoint 同梱可。
"""
from __future__ import annotations

from .observer.schema import Event

# ルールを日本語1句に描画するための対訳(プロンプト注入用。構成概念語は使わない)
_CAT_JP = {"food": "飲食", "cafe": "カフェ", "shop": "買い物", "nightlife": "夜の店",
           "taxi": "タクシー", "bus": "バス", "leisure": "余暇の場所"}
_BEH_JP = {"park": "公園で過ごす", "event_attend": "イベントに参加する",
           "flyer_view": "貼り紙を見る"}


def build_cfg(raw) -> dict:
    """recursion 設定の正準化。既定 OFF=現行挙動と完全同一(build_routes_cfg と同型)。"""
    from omegaconf import OmegaConf
    if OmegaConf.is_config(raw):
        raw = OmegaConf.to_container(raw, resolve=True)
    raw = dict(raw or {})
    return {
        "enabled": bool(raw.get("enabled", False)),
        "rules_in_prompt": bool(raw.get("rules_in_prompt", True)),
        "digest_in_prompt": bool(raw.get("digest_in_prompt", True)),
        "allow_repeal": bool(raw.get("allow_repeal", True)),
        "impact_news": bool(raw.get("impact_news", True)),
        "impact_news_threshold": int(raw.get("impact_news_threshold", 3)),
        "max_rules_in_prompt": int(raw.get("max_rules_in_prompt", 3)),
    }


def rule_jp(r: dict) -> str:
    """実効ルール1件を中立な日本語1句に描画(プロンプト・ニュース共用)。決定論。"""
    t = r["type"]
    name = r.get("name", "")
    if t == "fee":
        cat = _CAT_JP.get(r["target_cat"], r["target_cat"])
        return f"「{name}」({cat}の価格{r['delta']:+.0f}円)"
    if t == "bonus":
        beh = _BEH_JP.get(r["behavior"], r["behavior"])
        return f"「{name}」({beh}と{r['amount']:.0f}円支給)"
    if t == "curfew":
        cat = _CAT_JP.get(r["target_cat"], r["target_cat"])
        return f"「{name}」({r['from_h']}時〜{r['to_h']}時は{cat}を控える取り決め)"
    if t == "prohibit":
        cat = _CAT_JP.get(r["target_cat"], r["target_cat"])
        return (f"「{name}」({r['from_h']}時〜{r['to_h']}時は{cat}での滞在が禁止・"
                "取り締まりあり)")
    if t == "weekly_event":
        return f"「{name}」({r['every_days']}日ごとに{r['place']}で{r['title']})"
    if t == "declare":
        return f"「{name}」(宣言: {r['title']})"
    return f"「{name}」"


class Recursion:
    """再帰性の状態(日次の客観カウンタ+昨日のスナップショット)。

    plain データのみ(sim 参照を持たない)= checkpoint に丸ごと同梱できる。
    OFF(cfg.enabled=false)では note/on_day/各 line とも完全 no-op。
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.last_day = -1
        self.today = {"proposals": 0, "passed": 0, "enacted": 0,
                      "repealed": 0, "enforced": 0}
        self.yesterday: dict | None = None
        self.enforce_by_rule: dict[int, int] = {}   # rule_id -> 今日の執行件数
        self._enforce_names: dict[int, str] = {}    # rule_id -> ルール名(ニュース文用)

    # ---------------------------------------------------------------- 監視(客観カウント)
    def note(self, kind: str, rule_id=None, rule_name: str = "") -> None:
        """出来事1件の計数。R4: 因子・内面状態は一切読まない(件数だけ)。OFF は no-op。"""
        if not self.cfg["enabled"]:
            return
        if kind in self.today:
            self.today[kind] += 1
        if kind == "enforced" and rule_id is not None:
            rid = int(rule_id)
            self.enforce_by_rule[rid] = self.enforce_by_rule.get(rid, 0) + 1
            if rule_name:
                self._enforce_names[rid] = rule_name

    # ---------------------------------------------------------------- 日次境界
    def on_day(self, sim, day: int, step: int, sim_min: int) -> None:
        """昨日の客観カウントを確定(norm_digest)+ 執行多発ルールのニュース化。

        決定論(乱数なし・rule_id 昇順)。初日は前日が無いので確定のみ。OFF は no-op。"""
        if not self.cfg["enabled"] or day == self.last_day:
            return
        first = self.last_day < 0
        self.last_day = day
        if first:
            return
        snap = dict(self.today)
        by_rule = dict(self.enforce_by_rule)
        names = dict(self._enforce_names)
        self.today = {k: 0 for k in self.today}
        self.enforce_by_rule = {}
        self._enforce_names = {}
        rb = getattr(sim, "rulebook", None)
        active = len(rb.active) if rb is not None and rb.cfg["enabled"] else 0
        self.yesterday = {**snap, "active_rules": active}
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=-1,
                             kind="norm_digest", x=0.0, y=0.0,
                             payload={"day": int(day), **self.yesterday}))
        if self.cfg["impact_news"]:
            thr = self.cfg["impact_news_threshold"]
            for rid in sorted(by_rule):
                n = by_rule[rid]
                if n >= thr:
                    name = names.get(rid, f"取り決め{rid}")
                    sim.net.publish_news(
                        "取り締まり多発",
                        f"「{name}」に基づく取り締まりが昨日{n}件行われた", [], step)

    # ---------------------------------------------------------------- 知覚(プロンプト1行)
    def norm_line(self, sim) -> str | None:
        """いま実効の取り決め(最新 max_rules_in_prompt 件)。無ければ None=1行も足さない。"""
        if not (self.cfg["enabled"] and self.cfg["rules_in_prompt"]):
            return None
        rb = getattr(sim, "rulebook", None)
        if rb is None or not rb.cfg["enabled"] or not rb.active:
            return None
        k = self.cfg["max_rules_in_prompt"]
        parts = [rule_jp(r) for r in rb.active[-k:]]
        return "いまの街の取り決め(実際に効いている): " + " / ".join(parts)

    def digest_line(self) -> str | None:
        """昨日の街の動き(客観カウント)。全て0か初日なら None=1行も足さない。"""
        if not (self.cfg["enabled"] and self.cfg["digest_in_prompt"]):
            return None
        y = self.yesterday
        if not y:
            return None
        bits = []
        if y["proposals"]:
            bits.append(f"新しい提案{y['proposals']}件")
        if y["passed"]:
            bits.append(f"取り決めの成立{y['passed']}件")
        if y["repealed"]:
            bits.append(f"取り決めの廃止{y['repealed']}件")
        if y["enforced"]:
            bits.append(f"取り締まり{y['enforced']}件")
        if not bits:
            return None
        return "昨日の街の動き: " + "・".join(bits)
