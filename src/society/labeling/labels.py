"""ラベル/語彙システム(OPEN#4: constrained 既定)。

- 造語(coin)= label_coin + vocab_coin を記録、provenance に item 登録。
- 聴取 = transmission 記録 + 閾値(complex contagion: 既定2回)到達で採用(label_adopt)。
- 使用 = vocab_use を記録。
エージェントは「言葉を作れる・広められる」だけ(affordance)。何が広まるかは創発。
"""
from __future__ import annotations

from ..observer.logger import ObserverLogger
from ..observer.provenance import Item, ItemStore
from ..observer.schema import Event


class LabelSystem:
    def __init__(self, items: ItemStore, adopt_threshold: int = 2,
                 mode: str = "constrained"):
        self.items = items
        self.adopt_threshold = adopt_threshold
        # 粒度スイッチ(OPEN#4 / D9)。既定 constrained は現行の受理挙動を壊さない。
        self.mode = mode if mode in ("constrained", "open") else "constrained"
        self.text_to_item: dict[str, Item] = {}

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
             logger: ObserverLogger, context: dict | None = None) -> Item | None:
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
                logger.log(Event(step=step, sim_min=sim_min, agent_id=listener.id,
                                 kind="label_adopt", x=listener.x, y=listener.y,
                                 payload={"item_id": item.item_id, "text": word}))
        return heard_unknown
