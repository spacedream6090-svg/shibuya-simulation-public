"""OpenAI 互換 /v1/chat/completions バックエンド(M1)。標準ライブラリのみで HTTP を叩く。

1実装で以下を横断カバーする(いずれも OpenAI 互換の chat/completions を喋る):
  - OpenAI 本家(base_url="https://api.openai.com/v1")
  - Gemini の OpenAI 互換エンドポイント
  - Ollama の OpenAI 互換口(base_url="http://localhost:11434/v1")← キー不要で無料疎通できる
  - llama.cpp server / LM Studio など

流儀(既存 vllm.py / ollama.py に合わせる):
  - name はモデル名ベース(URL 非依存)= CachedLLM のキー安定性(D13)。サーバ構成や base_url が
    変わっても同一プロンプトの応答キャッシュが有効。
  - JSON 強制: response_format={"type":"json_object"} を付ける。非対応(HTTP 400)なら
    それを外して1度だけ再送(vllm.py と同流儀)。
  - エラー時は "__api_error__: ..." を返す(D16: 上位で fallback へ落とす。例外は投げない)。
  - API キーは環境変数からのみ読む(os.environ.get)。無ければ Authorization ヘッダを付けずに送る
    (ローカル互換サーバ=キー不要のため)。★キーの値をコード・ログ・エラー文字列に一切出さない
    (ODPT キーと同じセキュリティ規律)。

think 制御の限界(重要):
  - chat/completions 経由では思考モードを一般に制御できないモデルが多い。ここでは qwen3 系向けの
    ソフトスイッチのみ実装する: think=False のときだけユーザーメッセージ末尾に "\\n/no_think" を付す
    (qwen3 固有の規約。非 qwen モデルでは無害な文字列で実質 no-op)。
  - think=True の完全強制はこの経路では保証できない(モデル/サーバ依存)。厚い思考が要る呼種は
    ネイティブ制御を持つバックエンド(vLLM chat_template_kwargs 等)を使うこと。
  - ★実測(2026-07-12 検収): ollama 互換口の qwen3:4b は /no_think ソフトスイッチが効かず
    (2507 系 qwen3 はソフトスイッチ廃止)、思考が max_tokens を食い尽くして content が空になる。
    ローカル qwen3 は本バックエンドでなく **ネイティブ ollama.py(think 引数で制御可)** を使うこと。
    空 content は握り潰さず "__api_error__" にして上位 fallback(D16)へ落とす(下記 generate)。
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

from .base import LLMBackend

# qwen3 が think=True 時に本文へ混ぜる思考ブロックを剥がす(ollama/vllm と同挙動に合わせる)
_THINK_BLOCK = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


class OpenAICompatBackend(LLMBackend):
    def __init__(self, model: str, base_url: str = "https://api.openai.com/v1",
                 api_key_env: str = "OPENAI_API_KEY", timeout_s: float = 120.0):
        self.name = f"api/{model}"           # ★URL 非依存(D13: キャッシュキー安定)
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env       # ★キー本体ではなく「環境変数名」だけ保持する
        self.timeout_s = float(timeout_s)

    # ---- HTTP ----
    def _post(self, path: str, body: dict) -> dict:
        data = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        # キーは呼び出しごとに環境変数から読む(self には載せない)。無ければヘッダを付けない。
        key = os.environ.get(self.api_key_env)
        if key:
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(
            f"{self.base_url}{path}", data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _body(self, prompt: str, temperature: float, max_tokens: int,
              think: bool, json_fmt: bool) -> dict:
        # qwen3 ソフトスイッチ: think=False のときだけ /no_think(非 qwen では無害な文字列)
        content = prompt if think else f"{prompt}\n/no_think"
        body: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_fmt:
            body["response_format"] = {"type": "json_object"}
        return body

    @staticmethod
    def _extract(data: dict) -> str:
        choice = (data.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content", "")
        return _THINK_BLOCK.sub("", text or "")

    def generate(self, prompt: str, *, rng_key: str, temperature: float,
                 max_tokens: int, think: bool = False) -> str:
        for json_fmt in (True, False):       # response_format 非対応(400)なら外して1度だけ再送
            try:
                body = self._body(prompt, temperature, max_tokens, think, json_fmt)
                text = self._extract(self._post("/chat/completions", body))
                if not text.strip():
                    # 思考モデルが max_tokens を思考(reasoning)で使い切ると content が空になる
                    # (reflect_think 飢餓と同族)。空を成功として返さない=上位 fallback へ。
                    return ("__api_error__: empty content(思考が max_tokens を消費した可能性。"
                            "ローカル qwen3 は backend: ollama を使う)")
                return text
            except urllib.error.HTTPError as exc:
                if exc.code == 400 and json_fmt:
                    continue                 # response_format 非対応 → 外して再送
                # ★ reason のみ(キーやヘッダは urllib 例外に含まれない)
                return f"__api_error__: HTTP {exc.code} {exc.reason}"
            except (urllib.error.URLError, TimeoutError, OSError,
                    json.JSONDecodeError, KeyError, IndexError) as exc:
                return f"__api_error__: {exc}"
        return "__api_error__: request failed"
