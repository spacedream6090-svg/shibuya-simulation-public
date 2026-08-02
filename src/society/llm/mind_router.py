"""心モデルの解決層 MindRouter(第88バッチ)= agent_id → モデルの dispatch。

配線上の位置づけ(RouterLLM と同じ「dispatcher」であって**バックエンドではない**):
  - 最上位で `sim.llm` に据わる。子は **各自 CachedLLM に包まれて**渡される前提
    (= キャッシュキーには子の name(モデル名ベース・D13)が入るので、モデルを混ぜても
    応答はモデルごとに別キーで独立にキャッシュされる)。
  - ★**MindRouter 自身を CachedLLM で包んではいけない**: name が "mind" でモデル非依存の
    ため、包むと全モデルが同一キーに潰れてモデル別キャッシュが失われる(router.py と同じ罠)。
  - `default` は「機械的判断・対照系列・agent_id を持たない呼び出し」を受ける既定の子で、
    第88 以前に組み上がっていた `sim.llm`(CachedLLM または RouterLLM)**そのもの**を渡す。
    = 新しいバックエンドを 1 つも作らず、既存の配線をそのまま下敷きにする。

dispatch 規則(たった 2 つ):
  1. rng_key = "purpose/agent_id/step" の purpose が **心の呼び出し**(mind.MIND_PURPOSES)
     でなければ default。
  2. agent_id が取れなければ default。取れれば resolve(agent_id) が返す model_id の子へ。
     未知の model_id(pool から外れた過去ランの id 等)も default へ後退する。

R1(呼数の k 非依存):
  - dispatch は rng_key の**構文**と誕生時固定の割当だけを見る。k も traits も読まない。
  - 呼数・引数・戻り値を一切改変しない(戻り値は子のものをそのまま返す)。
  - `generate_many` は要求を子ごとに分けて発行し、**要求順に**組み直して返す
    (逐次 generate と同じ並び・同じ内容。子ごとの重複キー畳み込みは子の CachedLLM が行う)。
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:                            # 型注釈用(実行時 import は不要)
    from .base import LLMBackend


class MindRouter:
    name = "mind"                            # ★モデル非依存 = キャッシュキーには関与させない

    def __init__(self, default: "LLMBackend", children: dict[str, "LLMBackend"],
                 resolve: Callable[[str], str | None],
                 purposes: tuple[str, ...] | frozenset[str]):
        if default is None:
            raise ValueError("MindRouter には default(共有の既定バックエンド)が必須。")
        if not children:
            raise ValueError("MindRouter には children(model_id → バックエンド)が必須。")
        self.default = default
        self.children = dict(children)
        self._resolve = resolve
        self.purposes = frozenset(str(p) for p in purposes)

    # ------------------------------------------------------------------ 解決
    @staticmethod
    def _parse_key(rng_key: str) -> tuple[str, str | None]:
        """fleet.py / router.py と同流儀("purpose/agent_id/step" の先頭 2 つ)。"""
        parts = str(rng_key).split("/")
        purpose = parts[0] if parts else ""
        agent = parts[1] if len(parts) >= 2 and parts[1] != "" else None
        return purpose, agent

    def model_for(self, rng_key: str) -> str | None:
        """この呼び出しを受け持つ model_id(default 行きなら None)。観測・テスト用。"""
        purpose, agent = self._parse_key(rng_key)
        if agent is None or purpose not in self.purposes:
            return None
        mid = self._resolve(agent)
        return mid if mid in self.children else None

    def _child(self, rng_key: str):
        mid = self.model_for(rng_key)
        return self.children[mid] if mid is not None else self.default

    def _unique_children(self) -> list:
        """id() で重複排除した子の一覧(default 込み。同一子が複数 model に載っても 1 つ)。"""
        seen: dict[int, object] = {}
        seen.setdefault(id(self.default), self.default)
        for child in self.children.values():
            seen.setdefault(id(child), child)
        return list(seen.values())

    # ------------------------------------------------------------------ 集計
    @property
    def calls(self) -> int:
        return sum(int(getattr(c, "calls", 0)) for c in self._unique_children())

    @property
    def hits(self) -> int:
        return sum(int(getattr(c, "hits", 0)) for c in self._unique_children())

    # ------------------------------------------------------------------ 発行
    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        # 子の戻り値をそのまま返す(型に関知しない = CachedLLM のタプルも素の str も透過)
        return self._child(rng_key).generate(
            prompt, rng_key=rng_key, temperature=temperature,
            max_tokens=max_tokens, think=think)

    def generate_many(self, requests: list[dict], *, workers: int = 1) -> list:
        """一括発行(P2 S6b)。子ごとに分けて発行し、**要求順**に組み直す。

        逐次 generate を回した場合と結果・カウンタ・キャッシュ内容が一致する
        (子の CachedLLM が持つ「初出だけ生成し 2 個目以降は cached=True」の規約は
        子の中で閉じており、別の子へ散った要求どうしは元々独立)。
        """
        groups: dict[int, tuple[object, list[int]]] = {}
        order: list[int] = []
        for i, r in enumerate(requests):
            child = self._child(r["rng_key"])
            key = id(child)
            if key not in groups:
                groups[key] = (child, [])
                order.append(key)
            groups[key][1].append(i)
        out: list = [None] * len(requests)
        for key in order:                       # 初出順 = 決定論(dict 順に依存しない)
            child, idxs = groups[key]
            res = child.generate_many([requests[i] for i in idxs], workers=workers)
            for i, one in zip(idxs, res):
                out[i] = one
        return out
