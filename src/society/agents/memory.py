"""記憶 v2(生態系フェーズ Phase B)。Generative Agents 型の push 想起。

3層構造:
  1. エピソード緩衝(buffer): 直近の出来事(未統合)。
  2. 統合記憶(episodes/day_summaries): 就寝時の内省 LLM 呼び出しに**同居**して
     日次要約+顕著エピソード(重要度1-10)を作る。呼び出し回数は増やさない。
     ★統合は k の全条件で実行(計算量同一)。k のゲートは beliefs のみ(D7)。
  3. 意味記憶: beliefs(Agent 側、k の作用点)+ 関係台帳(relations)。

想起(retrieve)は非LLM の push 型(Generative Agents の検索式に準拠):
  score = 0.5·recency + 2.0·importance + 3.0·relevance(各 0-1 正規化の加重和)
係数は GA **公式実装の実効比 recency:relevance:importance = 0.5:3:2**(RQ2 検収
2026-07-04。論文は等重みだがコード既定が上記)、recency 減衰は GA の 0.99/時 を
1step=10分 に換算した 0.9983/step。relevance は埋め込みの代わりに文脈語
(場所名・同席者名)の包含で安価に代理する。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# 自由文クエリを想起の文脈語へ分解する区切り(空白・句読点・括弧など)
_QUERY_SPLIT = re.compile(r"[\s、。,.!?！？「」『』（）()\[\]・:：;；　/]+")


@dataclass
class Episode:
    step: int
    text: str
    kind: str = "event"          # event / heard / said / dm / sns / news / search
    importance: float = 3.0      # 1-10(統合時に LLM 採点で上書きされ得る)


@dataclass
class MemoryStore:
    buffer: list[Episode] = field(default_factory=list)        # 未統合の直近
    episodes: list[Episode] = field(default_factory=list)      # 統合済み顕著記憶
    day_summaries: list[str] = field(default_factory=list)     # 日次要約(日記)
    relations: dict[int, dict] = field(default_factory=dict)   # 相手id→台帳
    # 台帳エントリ = {name, count, last_step, last[, closeness, tier]}。closeness/tier は
    # 社会関係の質(Wave G2)で record_contact に closeness_delta が渡された時だけ付く
    # (relations OFF=渡されない=フィールドを一切足さない=台帳・プロンプトともバイト一致)。
    buffer_cap: int = 30
    store_cap: int = 120
    recency_decay: float = 0.9983    # /step(= GA の 0.99/時 を10分stepに換算)
    relations_max: int = 0           # >0 で関係台帳の上限(B6)。0=無制限=従来と完全同一

    # ---- 記録 ----
    def observe(self, step: int, text: str, kind: str = "event",
                importance_bonus: float = 0.0) -> None:
        # 観測時 importance = 3.0 + (arousal/novelty 加点)。clip 1–10。既定 bonus=0 で固定 3.0
        # (=現行と完全一致=バイト一致)。加点の中身は factors/affect.importance_bonus が決める
        # (§1.6 LaBar&Cabeza / §2.3 Gruber+ 2014: 高覚醒・高好奇心の出来事を顕著記憶へ)。
        imp = 3.0 + float(importance_bonus)
        imp = 1.0 if imp < 1.0 else 10.0 if imp > 10.0 else imp
        self.buffer.append(Episode(step=step, text=text, kind=kind, importance=imp))
        if len(self.buffer) > self.buffer_cap:
            self.buffer = self.buffer[-self.buffer_cap:]

    def record_contact(self, other_id: int, name: str, step: int,
                       text: str = "", closeness_delta: float | None = None) -> dict:
        rel = self.relations.setdefault(
            other_id, {"name": name, "count": 0, "last_step": step, "last": ""})
        rel["count"] += 1
        rel["last_step"] = step
        if text:
            rel["last"] = text[:40]
        # 社会関係の質(Wave G2)。closeness_delta が渡された時だけ親密度 weight を更新する
        # (ポジ交流で+/ネガ交流で−。tier 導出は relations.py 側)。既定 None(relations OFF)
        # なら closeness フィールドを一切足さない=台帳・プロンプトともバイト一致で不変。
        if closeness_delta is not None:
            rel["closeness"] = float(rel.get("closeness", 0.0)) + float(closeness_delta)
        # B6: 上限を超えたら LRU(last_step 最古、同点は相手id小)を退避。いま触れた
        # other_id は除外(最新接触なので落とさない)。決定論。既定 0 なら何もしない。
        if self.relations_max > 0:
            while len(self.relations) > self.relations_max:
                victim = min((k for k in self.relations if k != other_id),
                             key=lambda k: (self.relations[k]["last_step"], k))
                del self.relations[victim]
        return rel

    # ---- 統合(就寝時、内省 LLM 呼び出しに同居)----
    def consolidate(self, step: int, summary: str | None,
                    salient: list[tuple[str, float]] | None) -> None:
        if summary:
            self.day_summaries.append(summary)
            self.day_summaries = self.day_summaries[-7:]
        if salient:
            for text, imp in salient[:5]:
                self.episodes.append(Episode(
                    step=step, text=text, kind="salient",
                    importance=max(1.0, min(10.0, float(imp)))))
        else:                                   # LLM が採点をくれなかった時の後退動作
            for ep in self.buffer[-3:]:
                self.episodes.append(Episode(step=step, text=ep.text,
                                             kind="salient", importance=4.0))
        self.buffer = self.buffer[-5:]          # 直近だけ残して整理(睡眠の忘却)
        if len(self.episodes) > self.store_cap:
            self.episodes.sort(key=lambda e: (-e.importance, -e.step))
            self.episodes = self.episodes[:self.store_cap]
            self.episodes.sort(key=lambda e: e.step)

    # ---- 想起 ----
    def recent(self, n: int = 4) -> list[str]:
        return [e.text for e in self.buffer[-n:]]

    def retrieve(self, step: int, context: list[str], n: int = 3) -> list[str]:
        """統合記憶+緩衝から、いま関係の深い記憶を n 件(非LLM・決定論)。"""
        ctx = [c for c in context if c]
        pool = self.episodes + self.buffer[:-4]     # 直近4件は recent() が出すので除外
        scored: list[tuple[float, int, str]] = []
        for ep in pool:
            recency = self.recency_decay ** max(0, step - ep.step)
            relevance = (min(1.0, sum(1 for c in ctx if c in ep.text) / 2.0)
                         if ctx else 0.0)
            score = (0.5 * recency + 2.0 * (ep.importance / 10.0)
                     + 3.0 * relevance)             # GA 公式実装の実効比
            scored.append((score, ep.step, ep.text))
        scored.sort(key=lambda t: (-t[0], -t[1], t[2]))
        out, seen = [], set()
        for _, _, text in scored:
            if text not in seen:
                out.append(text)
                seen.add(text)
            if len(out) >= n:
                break
        return out

    def query(self, step: int, query_text: str, n: int = 3) -> list[str]:
        """agentic pull(ユーザー採用決定): 自由文クエリで能動的に記憶を掘る。

        内省の第1段が「何を思い出したいか」を出し、その文をここで想起に変換する。
        retrieve と同じ採点式(文字包含relevance+importance+recency)を再利用する
        決定論・非LLM の検索。クエリ文は区切りで文脈語に分解して包含判定に回す。
        """
        terms = [t for t in _QUERY_SPLIT.split(query_text or "") if len(t) >= 2]
        if not terms:                    # 手掛かりが無いクエリは何も掘らない(想起の暴発防止)
            return []
        return self.retrieve(step, terms, n=n)

    def relation_line(self, nearby_ids: list[int]) -> str | None:
        """同席者との関係(会話回数)をプロンプト用の一文に。"""
        parts = []
        for oid in nearby_ids:
            rel = self.relations.get(oid)
            if rel and rel["count"] >= 2:
                parts.append(f"{rel['name']}とは{rel['count']}回話した仲")
        return "、".join(parts[:2]) if parts else None
