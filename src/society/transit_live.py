"""SUMO ライブ連成タクシー配車 v-Ride-1 の連成コア(本体側・純ロジック)。

設計正典: docs/research/sumo-live-transit.md(ハイブリッド連成(b)・taxi device・dispatch=traci・
go/no-go 4条件)+ docs/research/traffic-indirect-effects.md(§5 チェックリスト・遅延の観測可能性)。

役割の分担:
  - 本モジュール(src/society/transit_live.py): **本体側の純ロジック**。エージェントの徒歩系 step
    (10分=600s)と SUMO の物理(配車待ち・乗車時間)を橋渡しし、到着 step の追加待ち(hold_steps)へ
    量子化する。SUMO への実接続は持たない(ブリッジに委譲=依存注入)。
  - ブリッジ(duck-typed): `reserve(...)` を1メソッド持つ物。
      * MockBridge(本モジュール): SUMO 不要の**純関数**。決定論・テスト/既定 mock backend 用。
      * SumoTaxiBridge(scripts/sumo_taxi_bridge.py): 実 SUMO を TraCI で回す(libsumo 不在環境=traci)。

決定論の絶対条件(sumo-live-transit.md §4 / traffic-indirect-effects.md §3.2):
  - 予約は呼び出し順(=エージェント id 昇順で _apply が反復)で逐次処理=純関数化。
  - k/traits/belief を配車・遅延処理に一切入れない(no-fingerprint / R1)。ここは物理量(位置・時刻・
    予約・fleet 状態)だけを見る。呼び出し側 routine._ride_extra の乗車判断も非LLM=呼数を1本も足さない。

観測(traffic-indirect-effects.md §5 条件(5)): 乗車は wait_s(配車待ち)/ ride_s(乗車時間)/
  delay_s(自由流に対する超過=信号・渋滞由来)を ride payload に必ず記録し、間接影響を L2 で事後に測れる
  状態にする。未配車(捕まらない)は taxi_unmatched で観測可能化する。

既定 OFF: transit_ride.live.enabled=false のとき live_cfg は None を返し、Simulation は TaxiLive を
  一切生成しない(=SUMO プロセスを起動しない=完全従来挙動=ゴールデン L1 バイト一致)。
"""
from __future__ import annotations

import hashlib
import math

STEP_SECONDS = 600.0        # 1 step = 10 分 = 600s(L1 / world.modes と同じ)


def _stable_hash(text: str) -> int:
    """プロセス非依存の安定ハッシュ(rng.py と同流儀)。MockBridge の純関数化に使う。"""
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def live_cfg(cfg) -> dict | None:
    """transit_ride.live を正規化する。既定 OFF(未設定/enabled=false)は None を返す
    (=呼び出し側は TaxiLive を生成しない=SUMO を起動しない=バイト一致)。"""
    rcfg = cfg.get("transit_ride", {}) or {}
    live = rcfg.get("live", {}) or {}
    if not bool(live.get("enabled", False)):
        return None
    speeds = cfg.world.modes.speeds
    return {
        "enabled": True,
        "backend": str(live.get("backend", "traci")),   # traci | mock
        "n_taxis": int(live.get("n_taxis", 8)),
        "seed": int(live.get("seed", 42)),
        "max_wait_s": float(live.get("max_wait_s", 900.0)),
        "net": (str(live["net"]) if live.get("net") else None),
        # 自由流の基準速度(m/s)= car の m/step ÷ 600s。delay_s(信号・渋滞超過)算出に使う。
        "car_m_per_s": float(speeds["car"]) / STEP_SECONDS,
    }


def quantize_hold(wait_s: float, delay_s: float,
                  step_seconds: float = STEP_SECONDS) -> int:
    """SUMO 由来の到着遅延(配車待ち + 自由流超過)を step 粒度の追加待ち step 数へ量子化する。

    hold_steps = ceil((wait_s + delay_s)/step_seconds)。SUMO が加える「呼ぶと車が来る待ち時間」+
    「信号・渋滞での超過」を step 境界へ丸める(sumo-live-transit.md §2.2 の到着 step 算式に対応)。
    追加遅延が無ければ 0(=従来と同じ到着 step)。純関数・k 非依存。"""
    extra = max(0.0, float(wait_s)) + max(0.0, float(delay_s))
    return int(math.ceil(extra / step_seconds)) if extra > 1e-9 else 0


class MockBridge:
    """SUMO 不要の**純関数ブリッジ**(既定 mock backend・テスト用)。

    reserve は (seed, from_node, to_node, call 秒) の安定ハッシュだけから待ち/乗車秒を決める
    =外部プロセスも wall-clock も引かない=完全決定論(同 seed 2回でバイト一致)。実 SUMO の
    配車物理の代理(近似)であって渋谷の実測ではない。go/no-go の①③④(LLM 呼数一致・OFF/ON・
    Windows 安定)と決定論の骨格検証に使い、②(実 SUMO の byte 一致)は SumoTaxiBridge で測る。"""

    def __init__(self, cfg: dict):
        self.seed = int(cfg["seed"])
        self.max_wait_s = float(cfg["max_wait_s"])
        self.car_m_per_s = float(cfg["car_m_per_s"])
        self.n_taxis = int(cfg["n_taxis"])

    def start(self) -> None:      # 純関数=起動不要(プロトコル互換のため空実装)
        return None

    def reserve(self, *, from_node, to_node, from_xy, to_xy,
                call_sim_sec: float) -> dict:
        h = _stable_hash(f"{self.seed}/{from_node}/{to_node}/{int(call_sim_sec)}")
        # 未配車(捕まらない): fleet が小さいほど起きやすい近似。1/(n_taxis*2+3) の決定論分岐。
        if (h % (self.n_taxis * 2 + 3)) == 0:
            return {"matched": False}
        wait_s = 60.0 + float(h % 540)                      # 配車待ち 60..600s
        dist_m = math.hypot(from_xy[0] - to_xy[0], from_xy[1] - to_xy[1])
        free_s = dist_m / max(1e-6, self.car_m_per_s)       # 自由流の乗車秒
        ride_s = free_s * (1.0 + ((h // 1000) % 60) / 100.0)  # +0..59% の信号・渋滞超過
        if wait_s > self.max_wait_s:
            return {"matched": False}
        return {"matched": True, "wait_s": wait_s, "ride_s": ride_s}

    def close(self) -> None:
        return None


class TaxiLive:
    """本体側の連成ドライバ。エージェントが taxi を選んだ瞬間に1件だけ呼ばれる(_apply)。

    ブリッジへ予約を投げ、返ってきた wait_s/ride_s を到着 step の追加待ち(hold_steps)へ量子化し、
    観測用の delay_s(自由流超過)とともに返す。ブリッジは依存注入(mock=純関数 / traci=実 SUMO)。
    k/belief を一切参照しない(no-fingerprint)。LLM を1本も呼ばない(呼数不変)。"""

    def __init__(self, cfg: dict, bridge):
        self.cfg = cfg
        self.bridge = bridge
        self.car_m_per_s = float(cfg["car_m_per_s"])
        self.n_requests = 0
        self.n_matched = 0
        self.n_unmatched = 0

    def request(self, *, agent_id: int, from_node, to_node,
                from_xy, to_xy, step: int, sim_min: int) -> dict:
        """1件の配車要求を解決する。返り値:
          matched=False → 捕まらない(呼び出し側は徒歩フォールバック + taxi_unmatched を記録)。
          matched=True  → {wait_s, ride_s, delay_s, hold_steps}(payload 記録 + 到着 step の追加待ち)。
        call_sim_sec は sim 内分 × 60(SUMO 秒時刻に同期)。"""
        self.n_requests += 1
        call_sim_sec = float(sim_min) * 60.0
        res = self.bridge.reserve(from_node=from_node, to_node=to_node,
                                  from_xy=from_xy, to_xy=to_xy,
                                  call_sim_sec=call_sim_sec)
        if not res or not res.get("matched"):
            self.n_unmatched += 1
            return {"matched": False}
        wait_s = float(res["wait_s"])
        ride_s = float(res["ride_s"])
        dist_m = math.hypot(from_xy[0] - to_xy[0], from_xy[1] - to_xy[1])
        free_s = dist_m / max(1e-6, self.car_m_per_s)        # 自由流(直線)の基準乗車秒
        delay_s = max(0.0, ride_s - free_s)                  # 信号・渋滞由来の超過(観測)
        hold_steps = quantize_hold(wait_s, delay_s)
        self.n_matched += 1
        return {"matched": True,
                "wait_s": round(wait_s, 1), "ride_s": round(ride_s, 1),
                "delay_s": round(delay_s, 1), "hold_steps": hold_steps}

    def stats(self) -> dict:
        return {"requests": self.n_requests, "matched": self.n_matched,
                "unmatched": self.n_unmatched}

    def close(self) -> None:
        try:
            self.bridge.close()
        except Exception:      # noqa: BLE001  — 終了処理は失敗しても本体を落とさない
            pass


def make_taxi_live(cfg, *, net_hint=None, origin=None, log=None):
    """Simulation から呼ぶ工場。live 無効なら None(=SUMO を起動しない=バイト一致)。

    backend=mock は純関数 MockBridge(SUMO 不要)。backend=traci は scripts.sumo_taxi_bridge を
    遅延 import して実 SUMO を起動する(不在/失敗時は None を返し、本体は従来挙動へ後退=SUMO 必須
    テストは環境ガードで skip)。src→scripts の import は traci backend のときだけ・遅延で行う。"""
    lc = live_cfg(cfg)
    if lc is None:
        return None
    backend = lc["backend"]
    if backend == "mock":
        return TaxiLive(lc, MockBridge(lc))
    if backend == "traci":
        try:
            import sys
            from pathlib import Path
            scripts_dir = str(Path(__file__).resolve().parents[2] / "scripts")
            if scripts_dir not in sys.path:
                sys.path.insert(0, scripts_dir)
            from sumo_taxi_bridge import SumoTaxiBridge
        except Exception as ex:      # noqa: BLE001
            if log:
                log(f"[transit_live] SUMO bridge import 不可({ex}); live 無効化")
            return None
        try:
            bridge = SumoTaxiBridge(lc, net_hint=net_hint, origin=origin)
            bridge.start()
        except Exception as ex:      # noqa: BLE001
            if log:
                log(f"[transit_live] SUMO 起動失敗({ex}); live 無効化")
            return None
        return TaxiLive(lc, bridge)
    if log:
        log(f"[transit_live] 未知の backend={backend}; live 無効化")
    return None
