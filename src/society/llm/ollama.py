"""Ollama バックエンド(手元 PC の本物 LLM)。標準ライブラリのみで HTTP 呼び出し。

使い方: python scripts/run.py model.backend=ollama model.name=qwen3:8b ...
JSON 出力を Ollama の format=json で強制(guided decoding の簡易版、D16 の崩れ対策)。
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from .base import LLMBackend


class OllamaBackend(LLMBackend):
    def __init__(self, model: str, host: str = "http://localhost:11434",
                 timeout_s: float = 120.0):
        self.name = f"ollama/{model}"
        self.model = model
        self.host = host.rstrip("/")
        self.timeout_s = timeout_s

    def generate(self, prompt: str, *, rng_key: str, temperature: float,
                 max_tokens: int, think: bool = False) -> str:
        # qwen3 系: think=True で思考モード(thinking と response が分離。response を使う)。
        # think=False は既定(有効だと予算が思考に消え空応答になりやすい)。
        body = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "think": bool(think),
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/generate", data=body,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            # D16: 失敗は上位で fallback(routine)に落ちるよう「壊れた応答」を返す
            return f"__ollama_error__: {exc}"
