"""合成ルータ RouterLLM(M2): purpose(呼種)別に子バックエンドへ委譲する薄い層。

配線上の位置づけ(重要):
  - RouterLLM は **最上位で CachedLLM の「代わり」に置かれる**。つまり simulation は
    従来 CachedLLM を呼んでいた場所で RouterLLM を呼ぶ。
  - 子は配線側で **各自 CachedLLM に包まれて** 渡される前提
    ({"default": CachedLLM(ollama), "reflect": CachedLLM(api), ...})。
  - キャッシュキーには各子バックエンドの name(モデル名ベース=D13)が入るので、モデルを混ぜても
    応答はモデルごとに別キーで独立にキャッシュされる。
  - ★**RouterLLM 自身を CachedLLM で包んではいけない**: router の name は "router" で
    モデル非依存のため、包むと全モデルが同一キーに潰れてモデル別キャッシュが失われる。
    キャッシュは必ず「子の側」で行うこと。

戻り値について:
  - generate() は選んだ子の generate() の戻り値を **そのまま** 返す(型に関知しない)。
    子が CachedLLM なら (response, call_id, cached) のタプル、素のバックエンドなら str。
    RouterLLM は dispatch だけを担い、呼数・引数・戻り値を一切改変しない(R1)。

R1(呼数の k 非依存)ガード:
  - dispatch は rng_key 先頭セグメント(purpose)の決め打ちのみ。全エージェント一様なので
    呼数もモデル割当も k と交絡しない(fleet.py の tier seam と同思想)。
  - **agent_tier ルーティングは未実装**(M3・ユーザー討議後)。将来の誤用を防ぐため、children の
    キーに "agent_tier" が来たら ValueError にする。

集計プロパティ:
  - calls / hits はユニークな子(id() で重複排除。同一子オブジェクトが複数 purpose に載っても
    1回だけ数える)の getattr(child, "calls"/"hits", 0) の総和。simulation の summary
    (llm_calls / llm_cache_hits)がここを読む。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:                            # 型注釈用(実行時 import は不要)
    from .base import LLMBackend


class RouterLLM:
    name = "router"                          # ★モデル非依存 = キャッシュキーには関与させない

    def __init__(self, children: dict[str, "LLMBackend"]):
        if "default" not in children:
            raise ValueError('RouterLLM の children には "default" が必須。')
        if "agent_tier" in children:
            raise ValueError(
                'RouterLLM は agent_tier ルーティング未実装(M3)。'
                '"agent_tier" キーは使えない(purpose 別 dispatch のみ)。')
        self.children = dict(children)

    @staticmethod
    def _purpose(rng_key: str) -> str:
        # fleet.py の _parse_key と同流儀("purpose/agent_id/step" の先頭)
        parts = str(rng_key).split("/")
        return parts[0] if parts else ""

    def _child(self, rng_key: str):
        # 未知 purpose(または空)は default へフォールバック
        return self.children.get(self._purpose(rng_key), self.children["default"])

    def _unique_children(self) -> list:
        """id() で重複排除した子の一覧(同一子が複数 purpose に載っても1つ)。"""
        seen: dict[int, object] = {}
        for child in self.children.values():
            seen.setdefault(id(child), child)
        return list(seen.values())

    @property
    def calls(self) -> int:
        return sum(int(getattr(c, "calls", 0)) for c in self._unique_children())

    @property
    def hits(self) -> int:
        return sum(int(getattr(c, "hits", 0)) for c in self._unique_children())

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        # 子の戻り値をそのまま返す(型に関知しない=CachedLLM のタプルも素の str も透過)
        return self._child(rng_key).generate(
            prompt, rng_key=rng_key, temperature=temperature,
            max_tokens=max_tokens, think=think)
