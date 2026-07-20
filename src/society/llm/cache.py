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

    def generate_many(self, requests: list[dict], *,
                      workers: int = 1) -> list[tuple[str, str, bool]]:
        """独立な要求の一括発行(P2 S6b)。要求順の [(response, call_id, cached), ...] を返す。

        requests の各要素: {"prompt", "rng_key", "temperature", "max_tokens",
        "think"(省略=False)}。逐次に generate を回した場合と結果・カウンタ・キャッシュ
        内容が一致するよう、共有状態(calls/hits/_mem/ファイル追記)は**逐次フェーズでのみ**
        更新する(ロック不要・決定論)。workers>1 のときだけ、キャッシュ未命中分の
        backend.generate を並行発行する(vLLM 等の継続バッチングを充填する目的。
        重複キーは初出の1回だけ生成し、2個目以降は逐次実行時と同じく cached=True)。
        """
        n = len(requests)
        results: list[tuple[str, str, bool] | None] = [None] * n
        self.calls += n
        # --- フェーズ1(逐次): キャッシュ解決と未命中の列挙(初出順) ---
        keys = [self._key(r["prompt"], r["temperature"], r["max_tokens"],
                          bool(r.get("think", False))) for r in requests]
        miss_order: list[int] = []          # 初出ミスの requests 添字(この順で生成・格納)
        first_of: dict[str, int] = {}       # key -> 初出ミスの requests 添字
        for i, key in enumerate(keys):
            if self.enabled and key in self._mem:
                self.hits += 1
                results[i] = (self._mem[key], key[:16], True)
            elif self.enabled and key in first_of:
                self.hits += 1              # 逐次なら初出が埋めた直後に命中する分
                results[i] = None           # フェーズ3で初出の応答を充填
            else:
                if self.enabled:
                    first_of[key] = i
                miss_order.append(i)
        # --- フェーズ2: 未命中の生成(workers>1 のみ並行。共有状態は触らない) ---
        def _gen(i: int) -> str:
            r = requests[i]
            return self.backend.generate(r["prompt"], rng_key=r["rng_key"],
                                         temperature=r["temperature"],
                                         max_tokens=r["max_tokens"],
                                         think=bool(r.get("think", False)))
        if workers > 1 and len(miss_order) > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=int(workers)) as ex:
                responses = list(ex.map(_gen, miss_order))
        else:
            responses = [_gen(i) for i in miss_order]
        # --- フェーズ3(逐次): 格納(初出順=決定論)と結果充填 ---
        for i, response in zip(miss_order, responses):
            key = keys[i]
            if self.enabled:
                self._mem[key] = response
                if self.path:
                    with self.path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({"key": key, "response": response},
                                           ensure_ascii=False) + "\n")
            results[i] = (response, key[:16], False)
        for i, key in enumerate(keys):
            if results[i] is None:          # 重複キーの2個目以降
                results[i] = (self._mem[key], key[:16], True)
        return results
