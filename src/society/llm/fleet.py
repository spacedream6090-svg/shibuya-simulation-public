"""FleetLLM: 複数 vLLM サーバへの艦隊ルータ(1台マルチGPU / 複数ノード 両対応の要)。

- sticky routing: rng_key(形式 "purpose/agent_id/step")から agent_id を抜き、
  agent_id → サーバを安定割当する。同一エージェントのペルソナ+履歴 prefix が
  同じサーバに当たり続けるため prefix cache のヒット率が上がる
  (docs/lit/infra__storage-routing.md: 単一機の天井 ~96% まで回復する実測)。
  agent_id が取れないキーは rng_key 全体のハッシュで割り当てる。
- 障害時再分配(D16): サーバがエラー/タイムアウト("__vllm_error__: ...")を返したら
  プール内の次候補へフェイルオーバし、そのサーバを cooldown_s 秒だけ外す。復帰は
  cooldown 経過後に自動(次の呼び出しで sticky 先に戻る)。全滅時は最後のエラー
  文字列を返す(シミュ側は fallback で続行=D16)。再分配は logging で追える。
- name はモデル名ベース(URL 非依存)= CachedLLM のキー安定性(D13)。サーバ構成が
  変わっても同一プロンプトの応答キャッシュが有効。
- tier seam: tiers={"reflect": [urls], "default": [urls]} を渡すと purpose
  (rng_key 先頭)別にサーバ群を分けられる(重い内省を大モデル1本に寄せる等)。
  既定 tiers=None = 全 URL を1つの default プールに(薄い実装、seam のみ明確化)。
"""
from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable

from .base import LLMBackend
from .deadline import DEFAULT_DEADLINE_S
from .vllm import VllmBackend

log = logging.getLogger("society.llm.fleet")


def _stable_int(value: str) -> int:
    """プロセス非依存の安定ハッシュ(rng._stable_hash と同流儀)。"""
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


class FleetLLM(LLMBackend):
    def __init__(self, servers: list[str], model: str, *,
                 timeout_s: float = 120.0,
                 tiers: dict[str, list[str]] | None = None,
                 cooldown_s: float = 30.0,
                 now: Callable[[], float] = time.monotonic,
                 format_mode: str = "json",
                 deadline_s: float = DEFAULT_DEADLINE_S):
        if not servers:
            raise ValueError("FleetLLM には少なくとも1本の server URL が必要。")
        self.name = f"fleet/{model}"          # ★URL 非依存(D13)
        self.model = model
        self.servers = [s.rstrip("/") for s in servers]
        self.cooldown_s = float(cooldown_s)
        self._now = now
        # format ノブ(第33バッチ 計画A)は子 VllmBackend へ透過。キー互換規約も子と同じ。
        self.format_mode = format_mode
        self.cache_extra = None if format_mode == "json" else {"f": format_mode}
        # URL ごとに1バックエンド(name は全て同一=キャッシュを共有できる)
        # 絶対時限(M-1)も子へ透過する。艦隊は 1 サーバの張り付きが失敗として
        # 他サーバへ再分配されるべき事象なので、子が時限で "__vllm_error__" を
        # 返せば既存のフェイルオーバ(cooldown)がそのまま働く。
        self.deadline_s = float(deadline_s)
        self._backend: dict[str, VllmBackend] = {
            u: VllmBackend(model, u, timeout_s=timeout_s, format_mode=format_mode,
                           deadline_s=deadline_s)
            for u in self.servers}
        # tier プール(purpose → URL リスト)。既定は全 URL の default 1プール。
        self._tiers: dict[str, list[str]] = {}
        if tiers:
            for purpose, urls in tiers.items():
                pool = [str(u).rstrip("/") for u in urls]
                for u in pool:                # tier だけに現れる URL も稼働対象へ登録
                    self._backend.setdefault(
                        u, VllmBackend(model, u, timeout_s=timeout_s,
                                       format_mode=format_mode,
                                       deadline_s=deadline_s))
                self._tiers[str(purpose)] = pool
        self._default = self._tiers.get("default", list(self.servers))
        self._cooldown: dict[str, float] = {}
        # 第78バッチ(認知階層アブレーション ablate.cognitive_tier): None=既定=従来どおり
        # purpose 別の振り分け。文字列を入れると **全 purpose** をそのティアのプールへ
        # 強制する(= 下位ティアへ落とす対照条件)。書き込みは society/ablate.py が
        # 構築時に 1 回だけ行い、以降は不変(ラン途中で変えない=指紋を作らない)。
        # 存在しないティア名を入れても素通り(_pool が既定へ後退)= 縮退は ablate 側が告知する。
        self.force_tier: str | None = None

    @staticmethod
    def _parse_key(rng_key: str) -> tuple[str, str | None]:
        parts = str(rng_key).split("/")
        purpose = parts[0] if parts else ""
        agent = parts[1] if len(parts) >= 2 and parts[1] != "" else None
        return purpose, agent

    def _pool(self, purpose: str) -> list[str]:
        # 認知階層アブレーション: force_tier が立っていれば purpose を無視して
        # そのプールへ寄せる(そのティアが未定義なら従来どおりの振り分けへ後退)。
        if self.force_tier is not None:
            forced = self._tiers.get(self.force_tier)
            if forced:
                return forced
        pool = self._tiers.get(purpose, self._default)
        return pool if pool else list(self.servers)

    def _ordered(self, rng_key: str) -> list[str]:
        """sticky な候補順(先頭=第一候補、以降=フェイルオーバ順)を返す。"""
        purpose, agent = self._parse_key(rng_key)
        pool = self._pool(purpose)
        anchor = agent if agent is not None else rng_key
        start = _stable_int(f"{self.name}:{anchor}") % len(pool)
        return [pool[(start + i) % len(pool)] for i in range(len(pool))]

    def generate(self, prompt: str, *, rng_key: str, temperature: float,
                 max_tokens: int, think: bool = False) -> str:
        now = self._now()
        ordered = self._ordered(rng_key)
        live = [u for u in ordered if self._cooldown.get(u, 0.0) <= now]
        candidates = live if live else ordered   # 全 cooldown 中でも一応叩く(復帰検知)
        last = "__fleet_error__: no server available"
        for i, url in enumerate(candidates):
            resp = self._backend[url].generate(
                prompt, rng_key=rng_key, temperature=temperature,
                max_tokens=max_tokens, think=think)
            if not resp.startswith("__vllm_error__") \
                    and not resp.startswith("__fleet_error__"):
                self._cooldown.pop(url, None)     # 成功 → cooldown 解除(復帰)
                if i > 0:
                    log.info("fleet recover: %s served %s after failover",
                             url, rng_key)
                return resp
            self._cooldown[url] = now + self.cooldown_s
            nxt = candidates[i + 1] if i + 1 < len(candidates) else "ALL DOWN"
            log.warning("fleet failover: %s failed for %s (%s) -> %s",
                        url, rng_key, resp[:60], nxt)
            last = resp
        return last
