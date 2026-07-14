"""Ollama バックエンド(手元 PC の本物 LLM)。標準ライブラリのみで HTTP 呼び出し。

使い方: python scripts/run.py model.backend=ollama model.name=qwen3:8b ...
JSON 出力を Ollama の format=json で強制(guided decoding の簡易版、D16 の崩れ対策)。
format は model.format ノブで制御(第33バッチ 計画A。docs/research/constrained-decoding.md):
  - "json"(既定)= 従来挙動そのまま。"none" = format を送らない(プロンプト規約のみ)。
  - ★think=True の呼には format を送らない(GBNF 文法が <think> を禁じ思考を殺す/応答が
    破損する実測 → 同リサーチ §1)。format ノブと think 境界ガードは不可分。
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from .base import LLMBackend

FORMAT_MODES = ("none", "json")   # "schema" は将来拡張(単一行動の狙い撃ち用)


class OllamaBackend(LLMBackend):
    def __init__(self, model: str, host: str = "http://localhost:11434",
                 timeout_s: float = 120.0, format_mode: str = "json"):
        if format_mode not in FORMAT_MODES:
            raise ValueError(f"model.format '{format_mode}' は未対応({FORMAT_MODES})。")
        self.name = f"ollama/{model}"
        self.model = model
        self.host = host.rstrip("/")
        self.timeout_s = timeout_s
        self.format_mode = format_mode
        # キャッシュキー拡張(CachedLLM._key が参照): 既定 "json" は None=従来キーと同一
        # (過去ランの llm_cache.jsonl 再生互換を保つ)。"none" のみ別キー。
        self.cache_extra = None if format_mode == "json" else {"f": format_mode}

    def generate(self, prompt: str, *, rng_key: str, temperature: float,
                 max_tokens: int, think: bool = False) -> str:
        # qwen3 系: think=True で思考モード(thinking と response が分離。response を使う)。
        # think=False は既定(有効だと予算が思考に消え空応答になりやすい)。
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "think": bool(think),
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        # think 境界ガード: 思考モードの呼に format を併せない(冒頭 docstring 参照)。
        if not think and self.format_mode == "json":
            payload["format"] = "json"
        body = json.dumps(payload).encode("utf-8")
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
