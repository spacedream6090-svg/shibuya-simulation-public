"""摂動シナリオのカタログ(#8)。既定 baseline = 完全 no-op。

世界に外から与える出来事(shock)を宣言的に定義する層。実験条件は conf の
world.scenario / world.scenario_params で切替える(コードは書き換えない)。

カタログ:
- baseline: 何もしない(既定)。on_step は即 return し、イベントも RNG も一切増やさない。
- shock_closure: 指定 step から一定期間、中心から半径内の道路エッジを封鎖する。
  経路計算(routing.Router)が封鎖エッジを除外する。移動中のエージェントは既存経路を
  走り切り、新規に計算される経路だけが封鎖区間を迂回する。解除で封鎖を撤去し復帰する。
- shock_event: 定型のニュースを世界イベント列(_phase_world_events)に注入する(公式
  発表と同じ配信経路)。実装は最小(world_events への 1 件追記)。

街の語彙として shock という英単語は用いるが、内部状態を表す語は一切参照しない
(no-fingerprint 契約: このファイルは world/ 配下=静的検査の対象)。

A1 第67バッチ: 「時間軸のない静的な環境改変条件」は `world/worldmod.py` へ一般化した
(shock_closure = 半径セレクタ + 時限発動、worldmod = 明示/矩形セレクタ + 構築時 1 回)。
エッジの同一性規則(正準キー)は 2 モジュールで食い違ってはならないので worldmod へ集約し、
ここはそれを別名で使う(関数本体は移動前と同一=挙動・バイト完全不変)。
"""
from __future__ import annotations

import math

from ..observer.schema import Event
from .worldmod import edge_key as _edge_key


class Scenario:
    """1 本の摂動シナリオ。sim を保持せず、on_step で受け取る(参照循環の回避)。"""

    def __init__(self, kind: str, params: dict, city):
        self.kind = kind
        self.params = params
        self.city = city
        self.active = False            # shock_closure が現在発動中か
        self.closed: set[tuple[str, str]] = set()   # 封鎖中エッジ(検証・可視化用に公開)

    # -- 封鎖対象エッジ(中心から半径内。端点いずれかが半径内なら封鎖)-------------
    def _edges_in_radius(self, center: tuple[float, float], radius: float
                         ) -> set[tuple[str, str]]:
        cx, cy = center
        within = set()
        for u, v in self.city.graph.edges():
            ux, uy = self.city.node_xy(u)
            vx, vy = self.city.node_xy(v)
            if (math.hypot(ux - cx, uy - cy) <= radius
                    or math.hypot(vx - cx, vy - cy) <= radius):
                within.add(_edge_key(u, v))
        return within

    def _log(self, sim, step: int, sim_min: int, phase: str, extra: dict) -> None:
        payload = {"kind": self.kind, "at": step, "phase": phase}
        payload.update(extra)
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=-1,
                             kind="scenario_shock", x=0.0, y=0.0, payload=payload))

    # -- 発動・解除フック(run_step の冒頭で毎 step 呼ばれる)-----------------------
    def on_step(self, sim, step: int) -> None:
        if self.kind == "baseline":
            return                     # 既定: 何もしない(現行挙動と完全同一)
        sim_min = sim.clock.sim_min(step)
        p = self.params
        if self.kind == "shock_closure":
            at = int(p.get("at_step", 0))
            dur = int(p.get("duration_steps", 6))
            if step == at and not self.active:
                center = tuple(float(c) for c in p.get("center", [0.0, 0.0]))
                radius = float(p.get("radius_m", 150.0))
                self.closed = self._edges_in_radius(center, radius)
                for u, v in self.closed:
                    self.city.graph.edges[u, v]["closed"] = True
                sim.router.invalidate()          # 経路キャッシュを捨てて封鎖を反映
                self.active = True
                self._log(sim, step, sim_min, "start",
                          {"center": [center[0], center[1]], "radius_m": radius,
                           "n_edges": len(self.closed)})
            elif self.active and step >= at + dur:
                for u, v in self.closed:
                    self.city.graph.edges[u, v].pop("closed", None)
                sim.router.invalidate()          # 封鎖撤去も同様に無効化して復帰
                n = len(self.closed)
                self.closed = set()
                self.active = False
                self._log(sim, step, sim_min, "end", {"n_edges": n})
        elif self.kind == "shock_event":
            at = int(p.get("at_step", 0))
            if step == at and not self.active:
                self.active = True
                # 世界イベント列に定型注入 → 同 step の _phase_world_events が配信する
                sim.world_events.append({
                    "step": step,
                    "title": str(p.get("title", "")),
                    "text": str(p.get("text", "")),
                    "word": p.get("word") or None,
                })
                self._log(sim, step, sim_min, "event",
                          {"title": str(p.get("title", "")),
                           "word": p.get("word") or None})


def build_scenario(cfg, city) -> Scenario:
    """conf.world.scenario / scenario_params から Scenario を組む。既定 = baseline。"""
    world = cfg.get("world", {}) or {}
    kind = str(world.get("scenario", "baseline") or "baseline")
    raw = world.get("scenario_params", {}) or {}
    # OmegaConf の DictConfig を素の dict に(list/scalar もそのまま取り出す)
    try:
        from omegaconf import OmegaConf
        params = OmegaConf.to_container(raw, resolve=True) if raw else {}
    except Exception:
        params = dict(raw)
    if not isinstance(params, dict):
        params = {}
    if kind not in ("baseline", "shock_closure", "shock_event"):
        kind = "baseline"
    return Scenario(kind, params, city)
