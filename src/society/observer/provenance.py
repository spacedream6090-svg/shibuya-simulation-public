"""伝播系譜(provenance)— ユーザー指定ログ(2026-07-03)。

広まるモノ(ラベル・語彙・噂・制度案)ごとに item_id を発行し、
「誰から誰へ・どのチャネルで」をすべて transmission イベントとして記録する。
→ 事後にカスケード木(家系図)を完全再構成できる = M1b/complex contagion の実測。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .logger import ObserverLogger
from .schema import Event


@dataclass
class Item:
    item_id: str
    kind: str            # label | vocab | rumor | institution ...
    text: str
    creator: int
    born_step: int
    transmissions: list[tuple[int, int, int, str]] = field(default_factory=list)
    # (step, from_agent, to_agent, channel)


class ItemStore:
    def __init__(self) -> None:
        self.items: dict[str, Item] = {}
        self._seq = 0

    def new_item(self, kind: str, text: str, creator: int, step: int) -> Item:
        self._seq += 1
        item = Item(item_id=f"{kind}-{self._seq:05d}", kind=kind, text=text,
                    creator=creator, born_step=step)
        self.items[item.item_id] = item
        return item

    def transmit(self, logger: ObserverLogger, item: Item, *, step: int, sim_min: int,
                 from_agent: int, to_agent: int, channel: str, x: float, y: float,
                 dist_m: float | None = None) -> None:
        item.transmissions.append((step, from_agent, to_agent, channel))
        payload = {"item_id": item.item_id, "from": from_agent, "channel": channel}
        if dist_m is not None:   # SNS/DM が架橋した物理距離(sns_geo ON のときのみ。None=従来と同一 payload)
            payload["dist_m"] = dist_m
        logger.log(Event(step=step, sim_min=sim_min, agent_id=to_agent,
                         kind="transmission", x=x, y=y, payload=payload))
