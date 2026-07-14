"""vLLM(OpenAI 互換 HTTP)バックエンド。本選環境の実 LLM。標準ライブラリのみで呼ぶ。

- /v1/completions を優先し、無ければ(HTTP 404)/v1/chat/completions へ恒久フォールバック。
- JSON 強制: response_format={"type":"json_object"}(vLLM 対応)。非対応(HTTP 400)なら
  それを外して再送し、プロンプト規約のみで JSON を得る(ollama の format=json 相当の代替)。
- qwen3 系の思考制御:
    * chat 経路 = chat_template_kwargs={"enable_thinking": think}(vLLM 拡張。対応 template でのみ有効)。
    * completions 経路 = think=False 時にソフトスイッチ "/no_think" をプロンプト末尾へ付す
      (qwen3 固有の規約。非 qwen モデルでは無害な文字列で実質 no-op)。
      think=True の完全強制は chat 経路のみ(raw completions は chat template を適用しないため)。
  ★実機(GPU)未検証。上記 think 制御はドキュメント準拠の実装であり、本選で疎通確認すること。
- name はモデル名ベース(URL 非依存)= CachedLLM のキー安定性(D13)。サーバ構成が
  変わっても同一プロンプトの応答キャッシュが有効。
- エラー時は "__vllm_error__: ..." を返す(D16: 上位で fallback へ落とす流儀。例外は投げない)。
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

from .base import LLMBackend

# qwen3 が think=True 時に本文へ混ぜる思考ブロックを剥がす(ollama は response のみ返す挙動に合わせる)
_THINK_BLOCK = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


class VllmBackend(LLMBackend):
    def __init__(self, model: str, base_url: str = "http://localhost:8000",
                 timeout_s: float = 120.0, format_mode: str = "json"):
        if format_mode not in ("none", "json"):
            raise ValueError(f"model.format '{format_mode}' は未対応(none | json)。")
        self.name = f"vllm/{model}"          # ★URL 非依存(D13: キャッシュキー安定)
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self._mode = "completions"           # 404 を見たら "chat" へ1度だけ切替
        self.format_mode = format_mode
        # キャッシュキー拡張(CachedLLM._key)。既定 "json"=None=従来キー互換。
        self.cache_extra = None if format_mode == "json" else {"f": format_mode}

    # ---- HTTP ----
    def _post(self, path: str, body: dict) -> dict:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}{path}", data=data,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _completions_body(self, prompt: str, temperature: float,
                          max_tokens: int, think: bool, json_fmt: bool) -> dict:
        # qwen3 ソフトスイッチ: think=False のときだけ /no_think を明示(非 qwen では無害な文字列)
        p = prompt if think else f"{prompt}\n/no_think"
        body: dict = {"model": self.model, "prompt": p, "stream": False,
                      "temperature": temperature, "max_tokens": max_tokens}
        if json_fmt:
            body["response_format"] = {"type": "json_object"}
        return body

    def _chat_body(self, prompt: str, temperature: float,
                   max_tokens: int, think: bool, json_fmt: bool) -> dict:
        body: dict = {"model": self.model,
                      "messages": [{"role": "user", "content": prompt}],
                      "stream": False, "temperature": temperature,
                      "max_tokens": max_tokens,
                      # vLLM 拡張: qwen3 の思考モード切替(対応 chat template のみ有効)
                      "chat_template_kwargs": {"enable_thinking": bool(think)}}
        if json_fmt:
            body["response_format"] = {"type": "json_object"}
        return body

    @staticmethod
    def _extract(data: dict, chat: bool) -> str:
        choice = (data.get("choices") or [{}])[0]
        if chat:
            text = (choice.get("message") or {}).get("content", "")
        else:
            text = choice.get("text", "")
        return _THINK_BLOCK.sub("", text or "")

    def generate(self, prompt: str, *, rng_key: str, temperature: float,
                 max_tokens: int, think: bool = False) -> str:
        # think 境界ガード(ollama.py と対称。guided decoding が思考を拘束しうるため
        # think=True の呼には response_format を送らない。実機未検証=本選で疎通確認)。
        if think or self.format_mode == "none":
            attempts: tuple[bool, ...] = (False,)
        else:
            attempts = (True, False)
        for json_fmt in attempts:            # response_format 非対応(400)なら外して1度だけ再送
            try:
                if self._mode == "completions":
                    body = self._completions_body(prompt, temperature,
                                                  max_tokens, think, json_fmt)
                    return self._extract(self._post("/v1/completions", body),
                                         chat=False)
                body = self._chat_body(prompt, temperature, max_tokens,
                                       think, json_fmt)
                return self._extract(self._post("/v1/chat/completions", body),
                                     chat=True)
            except urllib.error.HTTPError as exc:
                if exc.code == 404 and self._mode == "completions":
                    self._mode = "chat"      # completions 無し → chat へ恒久切替して再試行
                    return self.generate(prompt, rng_key=rng_key,
                                         temperature=temperature,
                                         max_tokens=max_tokens, think=think)
                if exc.code == 400 and json_fmt:
                    continue                 # response_format 非対応 → 外して再送
                return f"__vllm_error__: HTTP {exc.code} {exc.reason}"
            except (urllib.error.URLError, TimeoutError, OSError,
                    json.JSONDecodeError, KeyError, IndexError) as exc:
                return f"__vllm_error__: {exc}"
        return "__vllm_error__: request failed"
