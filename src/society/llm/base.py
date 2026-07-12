"""LLM バックエンド抽象(D10)。mock ⇄ vllm は config 1行で切替(D17)。"""
from __future__ import annotations

from abc import ABC, abstractmethod


class LLMBackend(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, prompt: str, *, rng_key: str, temperature: float,
                 max_tokens: int, think: bool = False) -> str:
        """1回の生成。rng_key = 決定論用のストリームキー(mock が使用)。

        think=True で思考モード(qwen3 系: thinking と response が分離して返る)。
        """
        raise NotImplementedError
