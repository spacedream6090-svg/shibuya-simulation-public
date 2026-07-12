#!/usr/bin/env python
"""API バックエンドの手動疎通確認ツール(pytest には入れない・実 API を1回だけ叩く)。

1プロンプトだけ送り、応答先頭200字と所要秒を表示する。openai_compat / anthropic に対応。

使い方:
    # Ollama の OpenAI 互換エンドポイントなら **キー不要・無料** で疎通確認できる:
    python scripts/check_llm_backends.py --backend openai_compat \\
        --base-url http://localhost:11434/v1 --model qwen3:4b

    # OpenAI 本家(要 OPENAI_API_KEY):
    python scripts/check_llm_backends.py --backend openai_compat --model gpt-5.4-mini

    # Gemini の OpenAI 互換エンドポイント(要 GEMINI_API_KEY を --api-key-env で指定):
    python scripts/check_llm_backends.py --backend openai_compat \\
        --base-url https://generativelanguage.googleapis.com/v1beta/openai \\
        --model gemini-2.5-flash-lite --api-key-env GEMINI_API_KEY

    # Anthropic(要 ANTHROPIC_API_KEY):
    python scripts/check_llm_backends.py --backend anthropic --model claude-haiku-4-5

キー未設定・接続不可なら正直にエラーを表示する(★キーの値は絶対に表示しない)。
実際に課金の発生しうる呼び出しなので、キーを入れて叩く前に対象モデルの単価を確認すること。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Windows コンソール(cp932)で日本語や記号を print しても落ちないように
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from society.llm.anthropic import AnthropicBackend        # noqa: E402
from society.llm.openai_compat import OpenAICompatBackend  # noqa: E402

_DEFAULT_PROMPT = (
    "あなたは渋谷の街を歩く人です。今の気分を短い JSON で返してください。"
    'キーは action(必ず "speak")と text の2つ。例: {"action":"speak","text":"..."}'
)


def _build(args) -> object:
    if args.backend == "anthropic":
        key_env = args.api_key_env or "ANTHROPIC_API_KEY"
        base = args.base_url or "https://api.anthropic.com"
        return AnthropicBackend(args.model, api_key_env=key_env,
                                base_url=base, timeout_s=args.timeout)
    # openai_compat
    key_env = args.api_key_env or "OPENAI_API_KEY"
    base = args.base_url or "https://api.openai.com/v1"
    return OpenAICompatBackend(args.model, base_url=base,
                               api_key_env=key_env, timeout_s=args.timeout)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="API バックエンドの手動疎通確認(1プロンプト送信)")
    ap.add_argument("--backend", required=True,
                    choices=["openai_compat", "anthropic"])
    ap.add_argument("--model", required=True, help="モデル名(例: qwen3:4b / claude-haiku-4-5)")
    ap.add_argument("--base-url", default=None,
                    help="openai_compat の /v1 まで(既定=OpenAI 本家)。anthropic は既定で本家")
    ap.add_argument("--api-key-env", default=None,
                    help="API キーを入れる環境変数名(既定: openai_compat=OPENAI_API_KEY / "
                         "anthropic=ANTHROPIC_API_KEY)。ローカル互換サーバならキー不要")
    ap.add_argument("--prompt", default=_DEFAULT_PROMPT)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()

    backend = _build(args)
    print(f"[check] backend={args.backend} model={args.model} "
          f"name={backend.name}", file=sys.stderr, flush=True)

    t0 = time.perf_counter()
    resp = backend.generate(args.prompt, rng_key="deliberate/0/0",
                            temperature=0.0, max_tokens=args.max_tokens)
    elapsed = time.perf_counter() - t0

    # エラーは "__xxx_error__: ..." 文字列で返る(キーの値は含まれない設計)
    is_error = resp.startswith("__") and "_error__" in resp[:32]
    tag = "ERROR" if is_error else "OK"
    print(f"[check] {tag} in {elapsed:.2f}s")
    print(f"[check] response[:200]: {resp[:200]}")
    return 1 if is_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
