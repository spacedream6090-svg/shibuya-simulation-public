"""応答キャッシュ(D13)= 実 LLM 運用時の再現性の実体。

key = sha256(model + params + prompt)。初回は実呼び出し+保存、リプレイ時は再生。
mock でも同じ経路を通す(配線を常時検証するため)。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .base import LLMBackend


class CachedLLM:
    def __init__(self, backend: LLMBackend, enabled: bool = True,
                 path: Path | None = None):
        self.backend = backend
        self.enabled = enabled
        self.path = path
        self._mem: dict[str, str] = {}
        self.calls = 0
        self.hits = 0
        if path and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                self._mem[row["key"]] = row["response"]

    def _key(self, prompt: str, temperature: float, max_tokens: int,
             think: bool) -> str:
        # think をキーに含める(含めないと思考モード切替時に旧キャッシュを誤再生する)。
        parts = {"model": self.backend.name, "t": temperature,
                 "m": max_tokens, "think": bool(think), "p": prompt}
        # format 等のバックエンド属性差もキーに含める(第33バッチ 計画A)。
        # 既定(format=json)では cache_extra=None=従来キーと同一 → 過去ランの再生互換を保つ。
        extra = getattr(self.backend, "cache_extra", None)
        if extra:
            parts.update(extra)
        blob = json.dumps(parts, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def generate(self, prompt: str, *, rng_key: str, temperature: float,
                 max_tokens: int, think: bool = False) -> tuple[str, str, bool]:
        """(response, call_id, cached) を返す。"""
        self.calls += 1
        key = self._key(prompt, temperature, max_tokens, think)
        if self.enabled and key in self._mem:
            self.hits += 1
            return self._mem[key], key[:16], True
        response = self.backend.generate(prompt, rng_key=rng_key,
                                         temperature=temperature,
                                         max_tokens=max_tokens, think=think)
        if self.enabled:
            self._mem[key] = response
            if self.path:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"key": key, "response": response},
                                       ensure_ascii=False) + "\n")
        return response, key[:16], False
