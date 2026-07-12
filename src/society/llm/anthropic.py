"""Anthropic Messages API ネイティブ バックエンド(M1)。標準ライブラリのみで HTTP を叩く。

OpenAI 互換とは (1)エンドポイント(/v1/messages)、(2)認証ヘッダ(x-api-key + anthropic-version)、
(3)サンプリング制御の扱い が異なるため別実装にする(docs/research/multi-model-lod.md §3.1)。

流儀(既存 vllm.py / ollama.py に合わせる):
  - name はモデル名ベース(URL 非依存)= CachedLLM のキー安定性(D13)。
  - エラー時は "__anthropic_error__: ..." を返す(D16: 上位で fallback へ落とす。例外は投げない)。
  - API キーは環境変数 ANTHROPIC_API_KEY からのみ読む。★キーの値をコード・ログ・エラー文字列に
    一切出さない(ODPT キーと同じ規律)。

現行世代(Opus 4.8 / Sonnet 5)固有の注意:
  - **temperature / top_p / seed は送らない**(撤廃済み=送ると HTTP 400。§3.1)。generate() は
    temperature 引数を受けるが**無視する**(サンプリング制御は API 側に存在しない)。再現性は
    llm_cache(初回保存→再生)が唯一の実体。
  - max_tokens は必須パラメータ。ここでは呼び出し側の max_tokens をそのまま送る。
  - think 引数も受けるが、この経路ではネイティブ思考制御(thinking パラメータ)を配線しておらず
    **無視する**(最終テキストを返す)。厚い思考制御が要るなら thinking を明示配線すること。
  - キー未設定なら HTTP を呼ばずに即エラー文字列を返す(無駄な接続・課金を避ける)。
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from .base import LLMBackend

_ANTHROPIC_VERSION = "2023-06-01"


class AnthropicBackend(LLMBackend):
    def __init__(self, model: str, api_key_env: str = "ANTHROPIC_API_KEY",
                 base_url: str = "https://api.anthropic.com", timeout_s: float = 120.0):
        self.name = f"anthropic/{model}"     # ★URL 非依存(D13)
        self.model = model
        self.api_key_env = api_key_env       # ★キー本体ではなく「環境変数名」だけ保持する
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)

    # ---- HTTP ----
    def _post(self, body: dict, key: str) -> dict:
        data = json.dumps(body).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "x-api-key": key,                # ★ヘッダにのみ載せる(ログ・エラーには出さない)
            "anthropic-version": _ANTHROPIC_VERSION,
        }
        req = urllib.request.Request(
            f"{self.base_url}/v1/messages", data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    @staticmethod
    def _extract(data: dict) -> str:
        # content は block の配列。最初の text ブロックを返す(thinking を混ぜても本文を拾える)。
        blocks = data.get("content") or []
        for b in blocks:
            if isinstance(b, dict) and b.get("type") == "text":
                return b.get("text", "")
        if blocks and isinstance(blocks[0], dict):
            return blocks[0].get("text", "")
        return ""

    def generate(self, prompt: str, *, rng_key: str, temperature: float,
                 max_tokens: int, think: bool = False) -> str:
        # temperature / seed は送らない(現行世代は撤廃済み)。think もこの経路では無視する。
        key = os.environ.get(self.api_key_env)
        if not key:
            # ★キーの値ではなく「どの環境変数に入れるか」だけ案内する
            return f"__anthropic_error__: no api key (set {self.api_key_env})"
        body = {
            "model": self.model,
            "max_tokens": max_tokens,        # ★必須パラメータ
            "messages": [{"role": "user", "content": prompt}],
        }
        try:
            return self._extract(self._post(body, key))
        except urllib.error.HTTPError as exc:
            # ★ reason のみ(キーやヘッダは urllib 例外に含まれない)
            return f"__anthropic_error__: HTTP {exc.code} {exc.reason}"
        except (urllib.error.URLError, TimeoutError, OSError,
                json.JSONDecodeError, KeyError, IndexError) as exc:
            return f"__anthropic_error__: {exc}"
