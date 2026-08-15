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

request-level stable seed(β11。第117バッチ 2026-08-16。conf `llm.request_seed.enabled`)
----------------------------------------------------------------------------------------
監査の確認事実(external-audit-triage.md F14): 本 backend は `seed` を 1 度も送っておらず、
vLLM は SamplingParams.seed 未指定時にサーバ側のグローバル乱数から引く。つまり **同じ
プロンプト・同じ温度でも応答が毎回変わりうる**(再現しているのは llm_cache が命中する範囲だけ)。
ON にすると 1 リクエストごとに

    seed = blake2b(run_seed, agent_id, step, purpose, ordinal) & 0x7fffffff

を送出ボディへ足す。材料は **既存の rng_key**(`"purpose/agent_id/step[/ordinal]"`)と
run.seed だけで、新しい乱数 stream は 1 本も引かない(決定論・resume 不変・呼数不変)。
★**既定 OFF では送出ボディがバイト単位で従来と同一**(`seed` キー自体を作らない)。
★主張できるのは「乱数条件を明示して保存した」ことまで。連続バッチングのバッチ構成で
  数値がわずかに動きうるのは vLLM の既知の性質であり、bit 決定論は約束しない。
★プロンプトは 1 バイトも変わらない = CachedLLM のキーも不変 = 過去ランの再生互換を保つ。
"""
from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.request

from .base import LLMBackend
from .deadline import DEFAULT_DEADLINE_S, request_json

# qwen3 が think=True 時に本文へ混ぜる思考ブロックを剥がす(ollama は response のみ返す挙動に合わせる)
_THINK_BLOCK = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)

#: seed の材料を連結するときの区切り(値の中に現れない制御文字 = 境界が曖昧にならない)。
_SEED_SEP = b"\x1f"
#: vLLM / OpenAI 互換 API が受ける seed の範囲へ収める(正の 31bit)。
_SEED_MASK = 0x7FFFFFFF


def split_rng_key(rng_key: str) -> tuple[str, str, str, str]:
    """rng_key を (purpose, agent_id, step, ordinal) へ分解する。

    形式は `"purpose/agent_id/step"`(既定)または `"purpose/agent_id/step/ordinal"`
    (第一級の再試行 `plan/…/retry`・D7 null 系列 `null/…/i` 等)。欠けている欄は
    空文字にする(**捏造しない**)。fleet.py の `_parse_key` と同じ読み方。
    """
    parts = str(rng_key).split("/")
    purpose = parts[0] if parts else ""
    agent = parts[1] if len(parts) > 1 else ""
    step = parts[2] if len(parts) > 2 else ""
    ordinal = "/".join(parts[3:]) if len(parts) > 3 else ""
    return purpose, agent, step, ordinal


def stable_request_seed(run_seed: int, rng_key: str) -> int:
    """(run_seed, agent_id, step, purpose, ordinal) の安定ハッシュ → 31bit の非負整数。

    プロセス非依存(blake2b。Python の `hash()` は使わない)・順序非依存の材料のみ・
    副作用なし。同じ材料なら何度呼んでも同じ値=resume でも再走でも同一 seed になる。
    """
    h = hashlib.blake2b(digest_size=8)
    purpose, agent, step, ordinal = split_rng_key(rng_key)
    for field in (str(int(run_seed)), agent, step, purpose, ordinal):
        h.update(field.encode("utf-8"))
        h.update(_SEED_SEP)
    return int.from_bytes(h.digest(), "big") & _SEED_MASK


class VllmBackend(LLMBackend):
    def __init__(self, model: str, base_url: str = "http://localhost:8000",
                 timeout_s: float = 120.0, format_mode: str = "json",
                 deadline_s: float = DEFAULT_DEADLINE_S,
                 request_seed: int | None = None):
        if format_mode not in ("none", "json"):
            raise ValueError(f"model.format '{format_mode}' は未対応(none | json)。")
        self.name = f"vllm/{model}"          # ★URL 非依存(D13: キャッシュキー安定)
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)
        # 呼び出し開始からの絶対時限(M-1。llm/deadline.py)。timeout_s では止まらない
        # 「細々と流れ続ける病的生成」をここで切る。0 以下で無効=従来経路。
        self.deadline_s = float(deadline_s)
        self._mode = "completions"           # 404 を見たら "chat" へ1度だけ切替
        self.format_mode = format_mode
        # キャッシュキー拡張(CachedLLM._key)。既定 "json"=None=従来キー互換。
        self.cache_extra = None if format_mode == "json" else {"f": format_mode}
        # β11: None(既定)= request seed を送らない = 送出ボディが従来とバイト一致。
        #      int  = その値を run_seed として 1 リクエストごとの安定 seed を導出する。
        # ★キャッシュキーには**入れない**(seed はプロンプトでもモデルパラメータでもなく、
        #   同じ入力に対する応答の再現条件。過去ランの llm_cache.jsonl を再生できる状態を保つ)。
        self.request_seed = None if request_seed is None else int(request_seed)

    # ---- request seed(β11) ----
    def request_seed_for(self, rng_key: str) -> int | None:
        """この呼び出しへ送る seed。OFF(既定)では **None**(= 欄自体を作らない)。

        CachedLLM が journal へ記録するときにも同じ口を使う(記録と送出で値がずれない)。
        """
        if self.request_seed is None:
            return None
        return stable_request_seed(self.request_seed, rng_key)

    # ---- HTTP ----
    def _post(self, path: str, body: dict) -> dict:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}{path}", data=data,
            headers={"Content-Type": "application/json"})
        return request_json(req, timeout_s=self.timeout_s,
                            deadline_s=self.deadline_s)

    def _completions_body(self, prompt: str, temperature: float,
                          max_tokens: int, think: bool, json_fmt: bool,
                          seed: int | None = None) -> dict:
        # qwen3 ソフトスイッチ: think=False のときだけ /no_think を明示(非 qwen では無害な文字列)
        p = prompt if think else f"{prompt}\n/no_think"
        body: dict = {"model": self.model, "prompt": p, "stream": False,
                      "temperature": temperature, "max_tokens": max_tokens}
        if json_fmt:
            body["response_format"] = {"type": "json_object"}
        if seed is not None:                 # β11: OFF ではキー自体を作らない(バイト一致)
            body["seed"] = int(seed)
        return body

    def _chat_body(self, prompt: str, temperature: float,
                   max_tokens: int, think: bool, json_fmt: bool,
                   seed: int | None = None) -> dict:
        body: dict = {"model": self.model,
                      "messages": [{"role": "user", "content": prompt}],
                      "stream": False, "temperature": temperature,
                      "max_tokens": max_tokens,
                      # vLLM 拡張: qwen3 の思考モード切替(対応 chat template のみ有効)
                      "chat_template_kwargs": {"enable_thinking": bool(think)}}
        if json_fmt:
            body["response_format"] = {"type": "json_object"}
        if seed is not None:                 # β11: OFF ではキー自体を作らない(バイト一致)
            body["seed"] = int(seed)
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
        seed = self.request_seed_for(rng_key)   # β11: OFF(既定)では None=従来ボディ
        for json_fmt in attempts:            # response_format 非対応(400)なら外して1度だけ再送
            try:
                if self._mode == "completions":
                    body = self._completions_body(prompt, temperature,
                                                  max_tokens, think, json_fmt,
                                                  seed)
                    return self._extract(self._post("/v1/completions", body),
                                         chat=False)
                body = self._chat_body(prompt, temperature, max_tokens,
                                       think, json_fmt, seed)
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
