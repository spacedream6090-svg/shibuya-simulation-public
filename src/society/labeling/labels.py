"""ラベル/語彙システム(OPEN#4: constrained 既定)。

- 造語(coin)= label_coin + vocab_coin を記録、provenance に item 登録。
- 聴取 = transmission 記録 + 閾値(complex contagion: 既定2回)到達で採用(label_adopt)。
- 使用 = vocab_use を記録。
エージェントは「言葉を作れる・広められる」だけ(affordance)。何が広まるかは創発。

場所の意味づけ最小版 D1(labeling.place_binding・既定 OFF)
--------------------------------------------------------
造語を「発生した場所(ノード)」に束縛し、そのノードに居る人が「この場所の呼ばれ方」を
知覚できるようにする最小機構。命名 → 場所 → 他者の知覚 → 採用 という痕跡の蓄積経路を開く。
- 束縛は `coin(..., node=...)` に **node が明示的に渡されたときだけ**(= 熟慮の coin_label 受理
  経路のみ。グループ名・提案文の coin は node を渡さない=束縛しない)。
- 束縛は語ごとに 1 回だけ(最初の coin。上書きしない)。
- 全決定論・乱数ゼロ・LLM 呼ゼロ。状態は `self.place_bind`(ON 時のみ dict)に閉じ、
  LabelSystem ごと checkpoint に同梱される(engine/checkpoint.py の "labels")=resume 対応。
- OFF は `place_bind is None` = 辞書を作らない・イベント 0 件・プロンプト 1 バイト不変。
"""
from __future__ import annotations

from ..observer.logger import ObserverLogger
from ..observer.provenance import Item, ItemStore
from ..observer.schema import Event

# 場所の呼ばれ方の 1 行(中立提示: 状態記述のみ。行動指示・評価語は書かない=no-fingerprint)。
_PLACE_LINE = "この場所の呼ばれ方: 「{word}」と呼ばれることがある。"


class LabelSystem:
    def __init__(self, items: ItemStore, adopt_threshold: int = 2,
                 mode: str = "constrained",
                 place_binding: dict | None = None):
        self.items = items
        self.adopt_threshold = adopt_threshold
        # 粒度スイッチ(OPEN#4 / D9)。既定 constrained は現行の受理挙動を壊さない。
        self.mode = mode if mode in ("constrained", "open") else "constrained"
        self.text_to_item: dict[str, Item] = {}
        # 場所の意味づけ D1(既定 OFF は None=辞書を一切作らない=状態が生えない)。
        self.place_bind: dict | None = None
        if place_binding and place_binding.get("enabled"):
            self.place_bind = {
                "min_adopters": max(1, int(place_binding.get("min_adopters", 1))),
                "prompt_line": bool(place_binding.get("prompt_line", True)),
                "bound": {},        # word -> {"node", "step", "coiner"}(束縛の台帳)
                "by_node": {},      # node -> [word, ...](知覚の逆引き。coin 順)
                "adopters": {},     # word -> 採用者数(coin/label_adopt で加算)
            }

    # ---- 場所の意味づけ D1 の内部ヘルパ(ON 時のみ実体が動く) ---------------- #
    def _pb(self) -> dict | None:
        """束縛状態(OFF・旧 checkpoint 由来のインスタンスでは None)。"""
        return getattr(self, "place_bind", None)

    def _note_adopt(self, word: str) -> None:
        """語の採用者を 1 人数える(呼び出し側が『新規に adopted へ入った』ことを保証する)。"""
        pb = self._pb()
        if pb is None:
            return
        pb["adopters"][word] = pb["adopters"].get(word, 0) + 1

    def _bind_place(self, agent, word: str, node, *, step: int, sim_min: int,
                    logger: ObserverLogger) -> None:
        """造語を発生ノードへ束縛し place_label_bind を 1 件記録(最初の coin のみ)。"""
        pb = self._pb()
        if pb is None or node is None or word in pb["bound"]:
            return
        pb["bound"][word] = {"node": node, "step": int(step),
                             "coiner": int(agent.id)}
        pb["by_node"].setdefault(node, []).append(word)
        logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                         kind="place_label_bind", x=agent.x, y=agent.y,
                         payload={"word": word, "node": node}))

    def place_line(self, node) -> str | None:
        """そのノードに束縛された語の 1 行(既定 OFF / 該当なしは None=1行も足さない)。

        採用者数 >= min_adopters の束縛語のうち 1 語だけを決定論で選ぶ
        (束縛 step 降順 → 語の辞書順)。乱数ゼロ・LLM 呼ゼロ・k 非依存(R1)。
        """
        pb = self._pb()
        if pb is None or not pb["prompt_line"] or node is None:
            return None
        words = pb["by_node"].get(node)
        if not words:
            return None
        floor = pb["min_adopters"]
        cands = [(-pb["bound"][w]["step"], w) for w in words
                 if pb["adopters"].get(w, 0) >= floor]
        if not cands:
            return None
        return _PLACE_LINE.format(word=min(cands)[1])

    def place_bound_count(self) -> int | None:
        """束縛された語の累計数(L2 place_bound_labels)。OFF は None=列を出さない。"""
        pb = self._pb()
        return None if pb is None else len(pb["bound"])

    def _normalize(self, word) -> str | None:
        """造語をモードに応じて正規化・検証。棄却なら None(= 沈黙)。

        - constrained: 前後空白を除去。改行・句読点・内部空白を含む『文』や 12 文字超は
          棄却する。短く句読点のない呼び名(現行 mock が作る語)はそのまま通す
          =既存の受理挙動を変えない。
        - open: フレーズ可。40 文字への切詰めのみ(棄却しない)。
        """
        if not isinstance(word, str):
            return None
        w = word.strip()
        if not w:
            return None
        if self.mode == "open":
            return w[:40]
        if len(w) > 12:
            return None
        if any(ch in w for ch in "\n\r。、．，！？!?,.") or " " in w or "　" in w:
            return None
        return w

    def coin(self, agent, word: str, *, step: int, sim_min: int,
             logger: ObserverLogger, context: dict | None = None,
             node=None) -> Item | None:
        """造語の受理。node を渡した呼び出しだけが「場所の意味づけ」の束縛対象(D1)。

        node は熟慮の coin_label 受理経路(scheduler)からのみ渡る。既定 OFF
        (place_bind is None)では node を渡されても何も起きない=バイト一致。
        """
        word = self._normalize(word)
        if word is None:                       # constrained で棄却 = 沈黙(item にしない)
            return None
        if word in self.text_to_item:          # 既存語の再発明は「使用」扱い
            item = self.text_to_item[word]
        else:
            item = self.items.new_item("vocab", word, agent.id, step)
            self.text_to_item[word] = item
            base = {"item_id": item.item_id, "text": word}
            for kind in ("label_coin", "vocab_coin"):
                payload = dict(base)
                # 造語の「発生過程・きっかけ」を vocab_coin に載せる(自然観察・促進はしない)
                if kind == "vocab_coin" and context:
                    payload.update(context)
                logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                                 kind=kind, x=agent.x, y=agent.y, payload=payload))
            # 場所の意味づけ D1: 最初の coin だけを発生ノードへ束縛(OFF は no-op)
            self._bind_place(agent, word, node, step=step, sim_min=sim_min,
                             logger=logger)
        if word not in agent.adopted:          # D1: 採用者数の計上(OFF は no-op)
            self._note_adopt(word)
        agent.adopted.add(word)
        return item

    def coin_media(self, word: str, *, step: int, sim_min: int,
                   logger: ObserverLogger) -> Item:
        """メディア(公式発表)発の語・商品名など。creator=-1。シナリオイベント用。"""
        if word in self.text_to_item:
            return self.text_to_item[word]
        item = self.items.new_item("vocab", word, -1, step)
        self.text_to_item[word] = item
        for kind in ("label_coin", "vocab_coin"):
            logger.log(Event(step=step, sim_min=sim_min, agent_id=-1,
                             kind=kind, x=0.0, y=0.0,
                             payload={"item_id": item.item_id, "text": word,
                                      "media": True}))
        return item

    def use(self, agent, word: str, *, step: int, sim_min: int,
            logger: ObserverLogger) -> None:
        item = self.text_to_item.get(word)
        if item is None:
            return
        logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                         kind="vocab_use", x=agent.x, y=agent.y,
                         payload={"item_id": item.item_id}))

    def on_hear(self, listener, words: list[str], speaker_id: int, *,
                step: int, sim_min: int, channel: str,
                logger: ObserverLogger, extra_threshold: int = 0,
                dist_m: float | None = None) -> bool:
        """聴取処理。未知語を聞いたら True(= LOD の驚きトリガー材料)。

        extra_threshold(既定 0)= 採用に必要な追加聴取回数。多言語の伝播障壁(後続波 H5、異言語間は
        語が広まりにくい)を diversity 層が不透明な整数として渡す。0 のとき従来と完全同一(バイト一致)。
        dist_m(既定 None)= 送り手との物理距離(SNS/DM 架橋距離・第22バッチ P2)。None なら payload 不変。"""
        heard_unknown = False
        threshold = self.adopt_threshold + int(extra_threshold)
        for word in words:
            item = self.text_to_item.get(word)
            if item is None:
                continue
            if word not in listener.adopted:
                heard_unknown = True
            self.items.transmit(logger, item, step=step, sim_min=sim_min,
                                from_agent=speaker_id, to_agent=listener.id,
                                channel=channel, x=listener.x, y=listener.y,
                                dist_m=dist_m)
            listener.heard_counts[item.item_id] += 1
            if (word not in listener.adopted
                    and listener.heard_counts[item.item_id] >= threshold):
                listener.adopted.add(word)
                self._note_adopt(word)         # D1: 採用者数の計上(OFF は no-op)
                logger.log(Event(step=step, sim_min=sim_min, agent_id=listener.id,
                                 kind="label_adopt", x=listener.x, y=listener.y,
                                 payload={"item_id": item.item_id, "text": word}))
        return heard_unknown
