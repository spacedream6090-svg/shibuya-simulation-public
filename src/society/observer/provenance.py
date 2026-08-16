"""伝播系譜(provenance)— ユーザー指定ログ(2026-07-03)。

広まるモノ(ラベル・語彙・噂・制度案)ごとに item_id を発行し、
「誰から誰へ・どのチャネルで」をすべて transmission イベントとして記録する。
→ 事後にカスケード木(家系図)を完全再構成できる = M1b/complex contagion の実測。

A9(第117 レーンE)— 観測台帳がエンジン入力になっていた問題
------------------------------------------------------------
``scheduler._search_index``(世界内検索エンジン)が ``len(item.transmissions)`` を
**プロンプト文字列**(「N 回拡散」)に使っていた。これは「観測のための台帳を、エンジンが
入力として読む」= 観測と世界の分離(R1)の破れである。台帳の実装(list を持つか、
明細をどこへ置くか、上限を掛けるか)を変えた瞬間にプロンプトが動く = 性能改善が
実験条件の変更になってしまう。

そこで **``transmissions_count``(int)を併設**し、プロンプトはこちらだけを読む。
``transmit`` が list と counter を同じ 1 箇所で更新するので、値は常に
``transmissions_count == len(transmissions)``(同値性テストで固定)= プロンプト文字列は
1 バイトも変わらない。以後「明細 list をどうするか」はプロンプトから切り離された
純粋な観測側の判断になる。

★**list の無界成長そのものは本バッチでは直していない**(報告事項)。理由は、明細 list の
  読者が凍結 14 本の外にまだ 2 つ残っているためである:
    - ``engine/simulation.py`` の summary ``n_transmissions``(= Σ len)
    - カスケード木を ``transmissions`` の中身そのものから再構成しているテスト
  正典は既に L1 の ``transmission`` イベント列(event sourcing)なので、明細は原理的に
  事後再構成できる。撤去は上記 2 読者の付け替えとセットで行う別バッチとする。
  25 万体 10 日の見積りで ~1 億タプル = 12GB 級 + checkpoint 時間の単調増。
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
    #: A9: 伝播回数。**常に ``len(transmissions)`` と同値**(``transmit`` が唯一の更新点)。
    #: エンジン側(プロンプト)が読むのはこちらだけ = 明細 list の実装から独立させる。
    transmissions_count: int = 0

    def __setstate__(self, state: dict) -> None:
        """旧 checkpoint 互換: ``transmissions_count`` が無い pickle から復元できる。

        dataclass の既定 pickle は ``__dict__`` をそのまま流し込むので、A9 より前に
        保存された Item にはこのキーが無い。**明細 list の長さから埋め直す**ことで
        同値性(counter == len)は復元後も保たれる(欠測を 0 で偽らない)。
        """
        self.__dict__.update(state)
        if "transmissions_count" not in state:
            self.__dict__["transmissions_count"] = len(state.get("transmissions") or [])


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
        # A9: 明細 list と counter は**この 1 行だけ**が更新点 = 常に同値であることが構成上
        #     保証される(counter を増やして list を伸ばさない経路も、その逆も存在しない)。
        item.transmissions.append((step, from_agent, to_agent, channel))
        item.transmissions_count = int(getattr(item, "transmissions_count", 0)) + 1
        payload = {"item_id": item.item_id, "from": from_agent, "channel": channel}
        if dist_m is not None:   # SNS/DM が架橋した物理距離(sns_geo ON のときのみ。None=従来と同一 payload)
            payload["dist_m"] = dist_m
        logger.log(Event(step=step, sim_min=sim_min, agent_id=to_agent,
                         kind="transmission", x=x, y=y, payload=payload))
