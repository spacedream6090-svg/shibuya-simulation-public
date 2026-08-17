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
- 混合モデル艦隊(MIX-1。docs/plans/model-mix-plan.md): tier の値を
  `{"urls": [...], "model": "qwen3:14b"}` の dict 形式で書くと、**その tier のサーバだけ
  別のモデル名**で子 VllmBackend を建てる(本選 = 会話 8B×5 + 思考 14B×2)。
  URL リスト形式(現行 conf)は **1 バイトも挙動が変わらない**(model は艦隊既定)。
  1 サーバ = 1 モデル(vLLM プロセスは 1 モデルしか載せない)なので、同じ URL へ
  別モデルが来たら**起動時に ValueError**(黙って片方が勝つ = 「14B のつもりで 8B が
  走る」を作らない)。
"""
from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable

from .base import LLMBackend
from .deadline import DEFAULT_DEADLINE_S
from .vllm import VllmBackend, cache_extra_for

log = logging.getLogger("society.llm.fleet")


def _stable_int(value: str) -> int:
    """プロセス非依存の安定ハッシュ(rng._stable_hash と同流儀)。"""
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


# 第131: 「リクエスト自体の不良」を表す HTTP コード(vllm.py が
# "__vllm_error__: HTTP <code> <reason>" の形で返す)。他サーバーへ持ち込んでも
# 同じ結果なので failover の対象にしない。404 は VllmBackend 内で chat へ切替済み・
# 408/429/5xx はサーバー側の事象なので従来どおり hop する。
_CLIENT_REJECT_PREFIXES = ("__vllm_error__: HTTP 400",
                           "__vllm_error__: HTTP 413",
                           "__vllm_error__: HTTP 422")


def _is_client_reject(resp: str) -> bool:
    return resp.startswith(_CLIENT_REJECT_PREFIXES)


# MIX-1: tier を dict 形式で書くときに受理するキー(全数固定)。未知キーを黙って無視すると
# `models:` のような 1 文字違いが「宣言したつもりで艦隊既定モデルが走る」に化けるため落とす。
_TIER_KEYS = ("urls", "model")


def normalize_tiers(tiers) -> dict[str, tuple[list[str], str | None]]:
    """conf の `model.tiers` を `{purpose: (urls, model or None)}` へ正規化する(純関数)。

    受理する 2 形式:
      * URL リスト … `reflect: ["http://localhost:8005"]` = **現行**(model=None=艦隊既定)
      * dict       … `reflect: {urls: ["http://localhost:8005"], model: "qwen3:14b"}`(MIX-1)
                     `urls` 必須・`model` 任意(無ければ艦隊既定モデル)。
    それ以外は ValueError。**黙って既定へ後退させない**のがこの関数の存在理由。
    """
    out: dict[str, tuple[list[str], str | None]] = {}
    for purpose, value in (tiers or {}).items():
        name = str(purpose)
        model: str | None = None
        if isinstance(value, dict):
            unknown = sorted(str(k) for k in value if str(k) not in _TIER_KEYS)
            if unknown:
                raise ValueError(
                    f"model.tiers['{name}'] に未知のキー {unknown} がある"
                    f"(受理するのは {' / '.join(_TIER_KEYS)} だけ)。")
            if "urls" not in value:
                raise ValueError(
                    f"model.tiers['{name}'] を dict 形式で書くときは urls が必須"
                    ' (例: {urls: ["http://localhost:8005"], model: "qwen3:14b"})。')
            raw_urls = value["urls"]
            raw_model = value.get("model", None)
            if raw_model is not None:
                if not isinstance(raw_model, str) or not raw_model.strip():
                    raise ValueError(
                        f"model.tiers['{name}'].model はモデル名の文字列であること"
                        f"(受理不能: {raw_model!r})。")
                model = raw_model
        elif isinstance(value, (list, tuple)):
            raw_urls = value
        else:
            raise ValueError(
                f"model.tiers['{name}'] は URL リスト または "
                '{urls: [...], model: "..."} であること'
                f"(受理不能: {type(value).__name__})。")
        if isinstance(raw_urls, (str, bytes)) or not isinstance(raw_urls, (list, tuple)):
            raise ValueError(
                f"model.tiers['{name}'] の urls は URL のリストであること"
                f"(受理不能: {type(raw_urls).__name__})。")
        urls = [str(u).rstrip("/") for u in raw_urls]
        if model is not None and not urls:
            raise ValueError(
                f"model.tiers['{name}'] は model='{model}' を宣言しているのに urls が空。")
        out[name] = (urls, model)
    return out


class FleetLLM(LLMBackend):
    def __init__(self, servers: list[str], model: str, *,
                 timeout_s: float = 120.0,
                 tiers: dict[str, list[str] | dict] | None = None,
                 cooldown_s: float = 30.0,
                 now: Callable[[], float] = time.monotonic,
                 format_mode: str = "json",
                 deadline_s: float = DEFAULT_DEADLINE_S,
                 request_seed: int | None = None,
                 api_mode: str = "completions"):
        if not servers:
            raise ValueError("FleetLLM には少なくとも1本の server URL が必要。")
        self.name = f"fleet/{model}"          # ★URL 非依存(D13)
        self.model = model
        self.servers = [s.rstrip("/") for s in servers]
        self.cooldown_s = float(cooldown_s)
        self._now = now
        # format ノブ(第33バッチ 計画A)は子 VllmBackend へ透過。キー互換規約も子と同じ。
        self.format_mode = format_mode
        # 経路ノブ(A8 実測 2026-08-17。conf model.api_mode)も子へ透過。既定 "completions"
        # = 子の送出ボディ・キャッシュキーとも従来と 1 バイトも変わらない。
        self.api_mode = api_mode
        # URL ごとに1バックエンド(name は全て同一=キャッシュを共有できる)
        # 絶対時限(M-1)も子へ透過する。艦隊は 1 サーバの張り付きが失敗として
        # 他サーバへ再分配されるべき事象なので、子が時限で "__vllm_error__" を
        # 返せば既存のフェイルオーバ(cooldown)がそのまま働く。
        self.deadline_s = float(deadline_s)
        # β11 request-level stable seed(第117)。None(既定)= 子は seed を送らない
        # = 送出ボディが従来とバイト一致。どのサーバに振られても seed は同じ材料
        # (run_seed, rng_key)から出るので、**sticky 先が変わっても値は動かない**。
        self.request_seed = None if request_seed is None else int(request_seed)
        self._backend: dict[str, VllmBackend] = {
            u: VllmBackend(model, u, timeout_s=timeout_s, format_mode=format_mode,
                           deadline_s=deadline_s, request_seed=self.request_seed,
                           api_mode=api_mode)
            for u in self.servers}
        # tier プール(purpose → URL リスト)。既定は全 URL の default 1プール。
        self._tiers: dict[str, list[str]] = {}
        # MIX-1: **明示宣言された** tier 別モデル名だけを持つ(宣言ゼロ = 空 = 現行と完全同一)。
        self.tier_models: dict[str, str] = {}
        if tiers:
            # 1 サーバ = 1 モデルの検算。起点は model.servers(= 艦隊既定モデル)で、
            # 同じ URL へ別モデル名が来た時点で落とす(どちらが勝ったか判らない状態を作らない)。
            assigned: dict[str, str] = {u: model for u in self.servers}
            owner: dict[str, str] = {u: "model.servers" for u in self.servers}
            for purpose, (pool, tier_model) in normalize_tiers(tiers).items():
                eff = tier_model if tier_model is not None else model
                for u in pool:                # tier だけに現れる URL も稼働対象へ登録
                    prev = assigned.get(u)
                    if prev is not None and prev != eff:
                        raise ValueError(
                            f"model.tiers['{purpose}'] は {u} へモデル '{eff}' を割り当てて"
                            f"いるが、同じ URL は {owner[u]} で '{prev}' として宣言済み。"
                            " 1 サーバ = 1 モデル(vLLM プロセスは 1 モデルしか載せない)。"
                            " 別モデルを載せるサーバは model.servers から外し、tier 側だけで"
                            "宣言すること。")
                    assigned[u] = eff
                    owner.setdefault(u, f"model.tiers['{purpose}']")
                    if u not in self._backend:
                        self._backend[u] = VllmBackend(
                            eff, u, timeout_s=timeout_s,
                            format_mode=format_mode,
                            deadline_s=deadline_s,
                            request_seed=self.request_seed,
                            api_mode=api_mode)
                self._tiers[purpose] = pool
                if tier_model is not None:
                    self.tier_models[purpose] = tier_model
        # キャッシュキー拡張(D13)。★tier 別モデルの宣言が 1 つも無ければ第138 の値と
        # **バイト単位で同一**(= 既存ランの llm_cache がそのまま命中する)。
        # 宣言があるときだけ静的マップ {"tiers": {purpose: model}} を足す。purpose 別の
        # プロンプトヘッダがあるので「同じプロンプトが別 tier へ行く」実衝突は実質ゼロだが、
        # **経路の構成そのものをキーへ刻む**ことで、tier 割当を組み替えたランが旧モデルの
        # 応答を誤再生することを構造的に防ぐ(D13 の趣旨)。キー順は sorted で決定論。
        # ★正直な限界: ablate の force_tier(構築後に書き込まれる全 purpose の強制寄せ)は
        #   このマップに現れない。対照条件の指定であって conf の艦隊構成ではないため。
        _extra = cache_extra_for(format_mode, api_mode)
        if self.tier_models:
            _extra = dict(_extra or {})
            _extra["tiers"] = dict(sorted(self.tier_models.items()))
        self.cache_extra = _extra
        self._default = self._tiers.get("default", list(self.servers))
        self._cooldown: dict[str, float] = {}
        # 第78バッチ(認知階層アブレーション ablate.cognitive_tier): None=既定=従来どおり
        # purpose 別の振り分け。文字列を入れると **全 purpose** をそのティアのプールへ
        # 強制する(= 下位ティアへ落とす対照条件)。書き込みは society/ablate.py が
        # 構築時に 1 回だけ行い、以降は不変(ラン途中で変えない=指紋を作らない)。
        # 存在しないティア名を入れても素通り(_pool が既定へ後退)= 縮退は ablate 側が告知する。
        self.force_tier: str | None = None

    def request_seed_for(self, rng_key: str) -> int | None:
        """β11: この呼び出しへ送る seed(OFF=None)。CachedLLM の journal 記録用の口。

        値は (run_seed, rng_key) だけの純関数なので、どの子 backend に聞いても同じ。
        """
        if self.request_seed is None:
            return None
        from .vllm import stable_request_seed
        return stable_request_seed(self.request_seed, rng_key)

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
            if _is_client_reject(resp):
                # 第131: 400/413/422 は「リクエスト自体の不良」(典型=プロンプト+max_tokens が
                # max_model_len 超過)。どのサーバーへ持ち込んでも同じ結果なので、
                # (a) 他サーバーへ hop しない(2k×144 実測: 1 リクエストが 6 hop→ALL DOWN)
                # (b) 健全なサーバーを cooldown に入れない(後続の別リクエストが避けてしまう)。
                # 429(過負荷)や 5xx・接続断はサーバー側の事象なので従来どおり hop する。
                log.warning("fleet client-reject (no failover): %s for %s (%s)",
                            url, rng_key, resp[:60])
                return resp
            self._cooldown[url] = now + self.cooldown_s
            nxt = candidates[i + 1] if i + 1 < len(candidates) else "ALL DOWN"
            log.warning("fleet failover: %s failed for %s (%s) -> %s",
                        url, rng_key, resp[:60], nxt)
            last = resp
        return last
