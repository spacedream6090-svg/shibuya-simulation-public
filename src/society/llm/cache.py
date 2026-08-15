"""応答キャッシュ(D13)= 実 LLM 運用時の再現性の実体。

key = sha256(model + params + prompt)。初回は実呼び出し+保存、リプレイ時は再生。
mock でも同じ経路を通す(配線を常時検証するため)。

第71バッチ 2026-07-31 の追加(いずれも既定では従来と完全同一の挙動):
  - `mode`(= conf の model.cache_mode)
      "free"  (既定) … 従来どおり。ミス時は backend へ実推論し、結果を保存する。
      "replay"       … ミス時は**即例外**。**新規推論へのフォールバックは書かない**。
        silent fallback は「再現しているつもりで再現していない」最悪の失敗モードで、
        これを禁止することがこの層の存在理由(dual-mode-requirements.md §2.2)。
        例外メッセージには rng_key(= "purpose/agent_id/step" なので誰のどの step かが判る)・
        key 先頭 16 桁・モード名を必ず入れる。
  - `journal`(= LlmJournal)… 全呼び出しの**プロンプト全文**を per-run に残す観測 sink。
      generate / generate_many の**両方**を通す(どちらも通らない経路を作らない)。
      記録は書くだけで、キャッシュ判定にも決定論にも一切関与しない(R1)。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .base import LLMBackend
from ..truth_ledger import check_prompt as _check_ledger_leak

CACHE_MODES = ("free", "replay")


class CachedLLM:
    def __init__(self, backend: LLMBackend, enabled: bool = True,
                 path: Path | None = None, *, mode: str = "free",
                 journal=None):
        self.backend = backend
        self.enabled = enabled
        self.path = path
        mode = str(mode or "free")
        if mode not in CACHE_MODES:
            raise ValueError(
                f"未知の model.cache_mode '{mode}'(有効値: {' | '.join(CACHE_MODES)})。")
        if mode == "replay" and not enabled:
            # キャッシュを無効にしたまま「キャッシュだけで走れ」は定義上ありえない。
            # 黙って free に落とすと非再現性が静かに混入するので起動時に落とす。
            raise ValueError(
                "model.cache=false と model.cache_mode=replay は矛盾している"
                "(replay はキャッシュ再生専用モード)。cache=true にするか mode=free にすること。")
        self.mode = mode
        self.journal = journal
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

    # ------------------------------------------------------------------ REPLAY
    def _replay_miss(self, key: str, rng_key: str, *, n_missing: int = 1):
        """REPLAY のキャッシュミス例外(フォールバックは**書かない**)。"""
        extra = "" if n_missing <= 1 else f" (この一括発行で {n_missing} 件が未命中)"
        return RuntimeError(
            f"cache_mode=replay: キャッシュミスにより中断{extra}。"
            f" REPLAY は新規推論へフォールバックしない(silent fallback の禁止)。"
            f" rng_key={rng_key} key={key[:16]} backend={self.backend.name}"
            f" cache={self.path}")

    # ------------------------------------------------------------------ 記録
    def _request_seed(self, rng_key: str) -> int | None:
        """β11(第117): backend が送る request-level seed。持たない backend では None。

        値の導出は backend 側の 1 箇所(vllm.stable_request_seed)にしかなく、ここは
        **同じ口を読むだけ**なので「記録した seed と送った seed が違う」が原理的に起きない。
        既定 OFF / mock / ollama / API 系では None = ジャーナルに欄が生えない。
        """
        fn = getattr(self.backend, "request_seed_for", None)
        return fn(rng_key) if fn is not None else None

    def _journal_write(self, *, key: str, rng_key: str, prompt: str,
                       response: str, temperature: float, max_tokens: int,
                       think: bool, cached: bool) -> None:
        if self.journal is None:
            return
        self.journal.record(key=key, rng_key=rng_key, prompt=prompt,
                            response=response, temperature=temperature,
                            max_tokens=max_tokens, think=think, cached=cached,
                            backend=self.backend.name,
                            seed=self._request_seed(rng_key))

    def generate(self, prompt: str, *, rng_key: str, temperature: float,
                 max_tokens: int, think: bool = False) -> tuple[str, str, bool]:
        """(response, call_id, cached) を返す。"""
        # 第73バッチ Part B: 真偽台帳の漏洩に対する**実行時アサーション**。全 LLM 呼び出しが
        # 通る唯一の関門なので、ここに置けばプロンプト構築経路のどこで混入しても捕まる。
        # 台帳が 1 件も無い(= beliefs OFF)間は 1 命令で return する = 既定ランのコストはゼロ。
        _check_ledger_leak(prompt, where=f"generate rng_key={rng_key}")
        self.calls += 1
        key = self._key(prompt, temperature, max_tokens, think)
        if self.enabled and key in self._mem:
            self.hits += 1
            response = self._mem[key]
            self._journal_write(key=key, rng_key=rng_key, prompt=prompt,
                                response=response, temperature=temperature,
                                max_tokens=max_tokens, think=think, cached=True)
            return response, key[:16], True
        if self.mode == "replay":                 # ★フォールバック禁止(即 fail)
            raise self._replay_miss(key, rng_key)
        response = self.backend.generate(prompt, rng_key=rng_key,
                                         temperature=temperature,
                                         max_tokens=max_tokens, think=think)
        if self.enabled:
            self._mem[key] = response
            if self.path:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"key": key, "response": response},
                                       ensure_ascii=False) + "\n")
        self._journal_write(key=key, rng_key=rng_key, prompt=prompt,
                            response=response, temperature=temperature,
                            max_tokens=max_tokens, think=think, cached=False)
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

        cache_mode=replay では**フェーズ1(逐次)の時点で**未命中を検出して即例外にする
        (並行発行に入る前なので、どの要求で落ちるかが決定論的=ミス列挙の順序に依存しない)。
        ジャーナルはフェーズ3の後に**要求順**で書く(逐次 generate と同じ並びになる)。
        """
        n = len(requests)
        for r in requests:                  # 第73 Part B: 台帳漏洩の実行時アサーション(既定はゼロコスト)
            _check_ledger_leak(r["prompt"], where=f"generate_many rng_key={r['rng_key']}")
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
        if self.mode == "replay" and miss_order:   # ★フォールバック禁止(生成に入る前に fail)
            first = miss_order[0]
            raise self._replay_miss(keys[first], requests[first]["rng_key"],
                                    n_missing=len(miss_order))
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
        # --- ジャーナル(要求順=逐次 generate と同じ並び。書くだけ・判定に不参加) ---
        if self.journal is not None:
            for i, r in enumerate(requests):
                response, _call_id, cached = results[i]
                self._journal_write(key=keys[i], rng_key=r["rng_key"],
                                    prompt=r["prompt"], response=response,
                                    temperature=r["temperature"],
                                    max_tokens=r["max_tokens"],
                                    think=bool(r.get("think", False)),
                                    cached=cached)
        return results
