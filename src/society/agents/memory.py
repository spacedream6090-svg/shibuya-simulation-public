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

── ACT-R 活性化ベースの人間らしい記憶(M-W。既定 OFF=`actr is None`)────────────
設計正典: docs/research/memory-cognitive-research.md 実装スケッチ M-W。
`actr` 設定 dict を持たせると retrieve/query が ACT-R 活性化ベースに切り替わる:
  - **基礎活性化** B = ln(Σ_j (t_now − t_ref_j)^−d)(冪乗則忘却。d=0.5、step を時間単位)。
    参照履歴 refs は Episode の**動的属性**(OFF 時は付けない=pickle/checkpoint バイト一致。
    先例=relations の closeness 動的キー / 状態フィールドの動的付与 S3)。
  - **連想活性化**(§3 fan effect): 現行 relevance を S − ln(fan) 型に置換。頻出手掛かりは
    fan が大きく後押しが薄まる=ありふれた記憶は特定しにくい(干渉)。
  - **感情/重要度オフセット**(§5): β = 0.3·(importance−3)/7。importance_bonus 経路は不変。
  - **ノイズ付き閾値**: A = 基礎+連想+β+ε(ε=ロジスティックノイズ・専用 stream "recall")。
    A < τ の候補は想起失敗。**全候補が失敗**したら query_ex.failed=True(memory_fail の材料)。
  - **再強化**(testing effect): 想起成功時に refs へ現在 step を追記(想起自体が記憶を強くする)。
  - **忘却**: consolidate の日次要約時、基礎活性化が下限未満の episodes を確率的に間引く
    (完全削除でなく要約=day_summaries 側に痕跡が残る現行構造を維持)。
R1(バイト一致): `actr is None` のとき①専用 stream を一度も引かない②refs を一度も書かない
③retrieve/query/consolidate は現行式に完全一致。ON でも新 stream(actr["seed"] 由来)が唯一の
追加乱数消費者で、既存の sim.hub 系列には一切触れない=決定論リプレイ不変。LLM 呼数は不変
(活性化計算は全て決定論+専用 stream)。
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

from ..rng import _stable_hash
from ..timeconv import CANON_DT_MIN

# 自由文クエリを想起の文脈語へ分解する区切り(空白・句読点・括弧など)
_QUERY_SPLIT = re.compile(r"[\s、。,.!?！？「」『』（）()\[\]・:：;；　/]+")

# ACT-R 既定パラメータ(docs/research/memory-cognitive-research.md §実装スケッチ (f))。
#   d   基礎活性化の減衰 = 0.5(ACT-R 標準 §1,§2)
#   tau 想起閾値 = -2.0(ACT-R 既定。**正準 step(=10分)を時間単位として**単一参照が
#       e^4≈55 正準 step ≈9h で τ 割れ。A7(第94)で MemoryStore.dt_min により経過 step を
#       正準換算するので、この 9h という実時間の意味は Δt を変えても保存される)
#   s   想起ノイズ尺度 = 0.5(ACT-R 既定。ガウス σ とは s=√3·σ/π)
#   S   最大連想強度(fan)= 2.0(ACT-R 応用の代表値 §3)
#   W   注意総重み = 1.0(W_j=W/n_cues §1)
#   forget_floor 忘却の下限活性化 = -2.0(=τ。想起閾値を割った記憶が物理忘却の候補)
#   forget_prob  下限未満 episode を日次で間引く確率 = 0.5(痕跡は day_summaries に残る)
#   seed 専用 stream のマスター種(有効化側が run の master_seed を渡す。既定 0)
ACTR_DEFAULTS = {
    "d": 0.5, "tau": -2.0, "s": 0.5, "S": 2.0, "W": 1.0,
    "forget_floor": -2.0, "forget_prob": 0.5, "seed": 0,
}


@dataclass
class Episode:
    step: int
    text: str
    kind: str = "event"          # event / heard / said / dm / sns / news / search
    importance: float = 3.0      # 1-10(統合時に LLM 採点で上書きされ得る)
    # ★参照履歴 refs は**動的属性**(ここにフィールドとして持たせない)。ACT-R 有効時のみ
    #   _strengthen() が `ep.refs=[生成step, …]` を付ける。OFF 時は属性自体が存在しない=
    #   __dict__ に現れない=pickle/checkpoint バイト一致(研究ドク R1 の要請)。読取は
    #   常に getattr(ep, "refs", None) or [ep.step](未強化なら生成 step 一点を暗黙 refs に)。


@dataclass
class _Recall:
    """query_ex の戻り。hits=想起テキスト、failed=手掛かりはあるが全候補が閾値未達。"""
    hits: list[str]
    failed: bool = False
    cue: str = ""
    best_activation: float | None = None
    tau: float | None = None


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
    actr: dict | None = None         # ACT-R 活性化設定(None=OFF=現行 0.5:2:3=バイト一致)
    # A7(第94バッチ OBS-U2): ACT-R 基礎活性化の時間単位。ACT-R の冪乗則忘却は **実時間**の
    # 関数だが、実装は step 差をそのまま t に使っている。Δt を細かくすると同じ実時間が
    # 大きな step 差になり、忘却が実時間で ~10/Δt 倍速になる(同じ Episode を評価する
    # recency_decay は simulation.py:1423 で scale_keep 済み=記憶モジュール内に二重基準)。
    # 経過 step を「正準 Δt(10 分)換算の step」へ読み替えて実時間基準に揃える。
    # 既定 10 = CANON = 読み替えの分岐自体を通らない(= 従来と 1 ビットも変わらない)。
    dt_min: int = 10

    # ---- ACT-R 設定の組み立て(有効化側/テストが使う)----
    @staticmethod
    def actr_config(**overrides) -> dict:
        """ACT-R 既定に overrides を重ねた設定 dict を返す。`mem.actr = actr_config(seed=..)`。"""
        cfg = dict(ACTR_DEFAULTS)
        cfg.update(overrides)
        return cfg

    # ---- 記録 ----
    def observe(self, step: int, text: str, kind: str = "event",
                importance_bonus: float = 0.0) -> None:
        # 観測時 importance = 3.0 + (arousal/novelty 加点)。clip 1–10。既定 bonus=0 で固定 3.0
        # (=現行と完全一致=バイト一致)。加点の中身は factors/affect.importance_bonus が決める。
        # ★感情増強(研究ドク §5 LaBar&Cabeza 2006 / Gruber+ 2014): 高覚醒・高好奇心の出来事を
        #   importance で底上げ→顕著記憶へ。ACT-R では β=0.3·(importance−3)/7 として活性化の
        #   オフセットに載る(_det_activation)ので「残るが薄れる」情動記憶の持続性が自然に出る。
        #   §5 の妥当な重み(値は変更しない・対応表のみ): 高覚醒→bonus≈+2〜+3 / 中覚醒→+1 / 中性→0。
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
                    salient: list[tuple[str, float]] | None,
                    *, agent_id: int = 0) -> None:
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
        # ACT-R 忘却(睡眠時): 基礎活性化が下限未満の episodes を確率的に間引く。要約(day_summaries)
        # 側に痕跡は残る現行構造を維持。OFF は一度も引かない=バイト一致。cap 整理より前に行う。
        if self.actr is not None:
            self._forget(step, agent_id)
        if len(self.episodes) > self.store_cap:
            self.episodes.sort(key=lambda e: (-e.importance, -e.step))
            self.episodes = self.episodes[:self.store_cap]
            self.episodes.sort(key=lambda e: e.step)

    # ---- 想起 ----
    def recent(self, n: int = 4) -> list[str]:
        return [e.text for e in self.buffer[-n:]]

    def retrieve(self, step: int, context: list[str], n: int = 3,
                 *, agent_id: int = 0) -> list[str]:
        """統合記憶+緩衝から、いま関係の深い記憶を n 件(非LLM・決定論)。"""
        ctx = [c for c in context if c]
        pool = self.episodes + self.buffer[:-4]     # 直近4件は recent() が出すので除外
        if self.actr is None:
            # ===== 現行 GA 0.5:2:3(1バイトも変えない)=====
            scored: list[tuple[float, int, str]] = []
            for ep in pool:
                recency = self.recency_decay ** max(0, step - ep.step)
                relevance = (min(1.0, sum(1 for c in ctx if c in ep.text) / 2.0)
                             if ctx else 0.0)
                score = (0.5 * recency + 2.0 * (ep.importance / 10.0)
                         + 3.0 * relevance)         # GA 公式実装の実効比
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
        # ===== ACT-R 活性化ベース(閾値+ノイズ+fan+再強化)=====
        hits, _failed, _best = self._retrieve_actr(step, ctx, pool, n, agent_id)
        return hits

    def query(self, step: int, query_text: str, n: int = 3,
              *, agent_id: int = 0) -> list[str]:
        """agentic pull(ユーザー採用決定): 自由文クエリで能動的に記憶を掘る。

        内省の第1段が「何を思い出したいか」を出し、その文をここで想起に変換する。
        retrieve と同じ採点式を再利用する決定論・非LLM の検索。手掛かり無しなら [](暴発防止)。
        """
        return self.query_ex(step, query_text, n=n, agent_id=agent_id).hits

    def query_ex(self, step: int, query_text: str, n: int = 3,
                 *, agent_id: int = 0) -> _Recall:
        """query の拡張版。想起失敗(手掛かりはあるが全候補が閾値未達)を failed で通知する。

        ACT-R OFF 時は failed 常に False=hits は query と同一=呼び出し側は従来どおり(バイト一致)。
        ON 時のみ failed=True を返し得る(memory_fail イベント+「思い出せない」1行の材料)。
        """
        terms = [t for t in _QUERY_SPLIT.split(query_text or "") if len(t) >= 2]
        cue = (query_text or "").strip()
        if not terms:                    # 手掛かりが無いクエリは何も掘らない(想起の暴発防止)
            return _Recall(hits=[], failed=False, cue=cue)
        if self.actr is None:
            return _Recall(hits=self.retrieve(step, terms, n=n), failed=False, cue=cue)
        pool = self.episodes + self.buffer[:-4]
        hits, failed, best = self._retrieve_actr(step, terms, pool, n, agent_id)
        return _Recall(hits=hits, failed=failed, cue=cue,
                       best_activation=best, tau=self.actr["tau"])

    # ---- ACT-R 活性化の内部機構(決定論。actr is None のときは一切呼ばれない)----
    def _retrieve_actr(self, step: int, ctx: list[str], pool: list[Episode],
                       n: int, agent_id: int) -> tuple[list[str], bool, float | None]:
        """活性化 A=基礎+連想(fan)+β(重要度)+ノイズ を計算し、A≥τ の上位 n を返す。

        戻り = (surfaced テキスト, failed, best_activation)。failed=手掛かりも候補もあるのに
        全候補が τ 未満(=思い出そうとして失敗)。surfaced した episode は refs へ現在 step を追記
        (testing effect=想起が記憶を強める)。
        """
        p = self.actr
        d, tau, s, S, W0 = p["d"], p["tau"], p["s"], p["S"], p["W"]
        # fan_j = 手掛かり語 j を含む pool 内 episode 数(連想活性化の希釈源=干渉)。
        fan = {c: (sum(1 for ep in pool if c in ep.text) or 1) for c in ctx}
        W = W0 / max(1, len(ctx))                       # W_j = W/n_cues
        scored: list[tuple[float, Episode]] = []
        for ep in pool:
            detA = self._det_activation(ep, step, ctx, fan, W, d, S)
            A = detA + self._recall_noise(agent_id, step, ep, s)   # +ロジスティックノイズ
            scored.append((A, ep))
        scored.sort(key=lambda t: (-t[0], -t[1].step, t[1].text))
        best = scored[0][0] if scored else None
        out: list[str] = []
        seen: set[str] = set()
        chosen: list[Episode] = []
        for A, ep in scored:
            if A < tau:                    # 閾値未達=この記憶は想起されない(忘却/失敗)
                continue
            if ep.text in seen:
                continue
            seen.add(ep.text)
            out.append(ep.text)
            chosen.append(ep)
            if len(out) >= n:
                break
        for ep in chosen:                  # 再強化(testing effect)
            self._strengthen(ep, step)
        failed = bool(pool and ctx and not out)
        return out, failed, best

    def _det_activation(self, ep: Episode, step: int, ctx: list[str],
                        fan: dict[str, int], W: float, d: float, S: float) -> float:
        """決定論部の総活性化 A_det = B(基礎) + β(重要度/感情) + Σ 連想(fan 希釈)。"""
        B = self._base_activation(ep, step, d)
        beta = 0.3 * (ep.importance - 3.0) / 7.0        # §5 感情/重要度オフセット
        spread = 0.0
        for c in ctx:                                   # §3 連想: S_{j,i} = S − ln(fan_j)
            if c in ep.text:
                spread += W * (S - math.log(fan[c]))
        return B + beta + spread

    def _base_activation(self, ep: Episode, step: int, d: float) -> float:
        """基礎活性化 B = ln(Σ_j (t_now − t_ref_j)^−d)。refs 未強化なら生成 step 一点を使う。

        A7: 経過 step は **正準 Δt(10 分)換算**へ読み替える(冪乗則忘却は実時間の関数)。
        既定 Δt=10 では下の分岐に入らず従来式そのもの=バイト一致。"""
        refs = getattr(ep, "refs", None) or [ep.step]
        total = 0.0
        if self.dt_min == CANON_DT_MIN:                 # ★恒等パス(既定): 従来式のまま
            for t in refs:
                total += max(1, step - t) ** (-d)
            return math.log(total)
        r = self.dt_min / float(CANON_DT_MIN)           # 1 step が正準の何倍の実時間か
        for t in refs:
            total += max(1.0, (step - t) * r) ** (-d)
        return math.log(total)

    def _recall_noise(self, agent_id: int, step: int, ep: Episode, s: float) -> float:
        """ロジスティックノイズ ε ~ Logistic(0, s)。専用 stream "recall"(actr seed 由来)。

        episode 同一性でキーするので pool の順序に依らず決定論。他の sim.hub 系列には触れない。
        """
        u = float(self._stream("recall", int(agent_id), int(step),
                               int(ep.step), ep.text).random())
        u = min(max(u, 1e-12), 1.0 - 1e-12)             # logit の発散回避
        return s * math.log(u / (1.0 - u))

    def _strengthen(self, ep: Episode, step: int) -> None:
        """想起成功時に refs へ現在 step を追記(初回は生成 step を種にして動的属性を作る)。"""
        refs = getattr(ep, "refs", None)
        if refs is None:
            refs = [ep.step]                            # 生成=最初の参照。ここで初めて属性を付与
            ep.refs = refs
        if step not in refs:
            refs.append(step)

    def _forget(self, step: int, agent_id: int) -> None:
        """基礎活性化が下限未満の episodes を確率的に間引く(決定論)。痕跡は day_summaries に残る。"""
        p = self.actr
        floor, prob, d = p["forget_floor"], p["forget_prob"], p["d"]
        kept: list[Episode] = []
        for ep in self.episodes:
            if self._base_activation(ep, step, d) < floor:
                u = float(self._stream("forget", int(agent_id), int(step),
                                       int(ep.step), ep.text).random())
                if u < prob:
                    continue                            # 間引く(=物理忘却。要約に痕跡)
            kept.append(ep)
        self.episodes = kept

    def _stream(self, *key: int | str) -> np.random.Generator:
        """actr["seed"] を種にした専用 RNG(RngHub.stream と同型)。sim.hub には一切触れない。"""
        ints: list[int] = [int(self.actr.get("seed", 0))]
        for part in key:
            ints.append(part if isinstance(part, int) else _stable_hash(part))
        return np.random.Generator(np.random.PCG64(np.random.SeedSequence(ints)))

    def relation_line(self, nearby_ids: list[int]) -> str | None:
        """同席者との関係(会話回数)をプロンプト用の一文に。"""
        parts = []
        for oid in nearby_ids:
            rel = self.relations.get(oid)
            if rel and rel["count"] >= 2:
                parts.append(f"{rel['name']}とは{rel['count']}回話した仲")
        return "、".join(parts[:2]) if parts else None
