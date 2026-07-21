#!/usr/bin/env python
"""SUMO ライブ連成タクシー配車 v-Ride-1 の**実 SUMO ブリッジ**(TraCI)。

正典: docs/research/sumo-live-transit.md(taxi device・dispatch=traci・go/no-go 4条件)。
本体側の純ロジックは src/society/transit_live.py。ここは「予約を SUMO に注入 → 決定論配車 →
picked_up/dropped_off の実秒を返す」物理委譲だけを担う(片方向)。libsumo が入らない Windows 環境
(検証済: このリポジトリは traci のみ)を想定し、**sumo をサブプロセス起動して TraCI 接続**する。

決定論(絶対条件):
  - `--seed` 固定・`--random` 禁止・`--threads` 既定1(並列ルーティング禁止)。
  - taxi fleet はソート済みエッジ id への決定論配置。
  - 配車は「新規予約を1件ずつ、id 昇順の空車(getTaxiFleet(0) をソート)へ dispatchTaxi」= 純関数規則。
  - 予約は呼び出し順(本体の agent id 昇順反復)で逐次解決。照会は step 境界(1s tick)で行う。
これにより同 seed 2回で (wait_s, ride_s) 列がバイト一致する(PoC 実測: pickup=321s, dropoff=581s 再現)。

簡約(v-Ride-1): 予約は**同期解決**(1件を dropoff まで進めてから次へ)。相乗り(greedyShared)・
真の並行配車・pt バスは v-Ride-2/3(sumo-live-transit.md §5 段階計画)。この簡約は決定論と Windows
安定性の go/no-go 判定に必要十分で、realism は段階的に深める。

SUMO 不在環境では import はできる(traci/sumolib の import は start() まで遅延)。start() が SUMO を
見つけられなければ RuntimeError を投げ、呼び出し側(transit_live.make_taxi_live)が live 無効化へ後退する。
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = Path(__file__).resolve().parent
for _p in (str(_ROOT), str(_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# sumo_pipeline の探索/投影ユーティリティを再利用(sumo_home 優先順位・inv_project の座標整合)。
import sumo_pipeline as sp   # noqa: E402

_TAXI_ADDITIONAL = (
    '<additional>\n'
    '  <vType id="taxi" vClass="taxi">\n'
    '    <param key="has.taxi.device" value="true"/>\n'
    '  </vType>\n'
    '</additional>\n'
)


def _resolve_net(net_hint) -> str | None:
    """net.xml を解決する。net_hint(明示/ラン推定)→ 既存の runs 配下の car net を探す。"""
    if net_hint:
        p = Path(net_hint)
        if p.is_file():
            return str(p)
    # runs 配下の *_car.net.xml を新しい順に探す(sumo_pipeline の成果物)
    cands = sorted((_ROOT / "runs").glob("*/sumo/*_car.net.xml"),
                   key=lambda q: q.stat().st_mtime, reverse=True)
    return str(cands[0]) if cands else None


class SumoTaxiBridge:
    """実 SUMO(TraCI)ブリッジ。reserve() が transit_live.TaxiLive から呼ばれる。"""

    def __init__(self, cfg: dict, *, net_hint=None, origin=None):
        self.cfg = cfg
        self.seed = int(cfg["seed"])
        self.n_taxis = int(cfg["n_taxis"])
        self.max_wait_s = float(cfg["max_wait_s"])
        self.net_hint = net_hint
        self.origin = tuple(origin) if origin else None    # (lat, lon)。node→edge 写像に要る
        self.traci = None
        self.net = None
        self._now = 0.0
        self._pid = 0
        self._add_file = None
        self._end = 24 * 3600

    # ------------------------------------------------------------------ 起動
    def start(self) -> None:
        net_path = _resolve_net(self.net_hint)
        if not net_path:
            raise RuntimeError("car net.xml が見つからない(sumo_pipeline で net を作成)")
        exe = sp.find_bin("sumo")
        if not exe:
            raise RuntimeError("sumo 実行ファイル不在")
        sp.ensure_tools_on_path()
        import sumolib  # type: ignore
        import traci     # type: ignore
        self.net = sumolib.net.readNet(net_path)
        if self.origin is None:      # net_hint のランから原点を引けなければ net 自身の geo 原点近似
            self.origin = (0.0, 0.0)
        fd, add = tempfile.mkstemp(prefix="taxi_vtype_", suffix=".xml")
        os.write(fd, _TAXI_ADDITIONAL.encode("utf-8"))
        os.close(fd)
        self._add_file = add
        cmd = [exe, "-n", net_path, "-a", add,
               "--seed", str(self.seed), "--step-length", "1.0",
               "--begin", "0", "--end", str(self._end),
               "--device.taxi.dispatch-algorithm", "traci",
               "--threads", "1",
               "--no-step-log", "--no-warnings", "true", "--ignore-route-errors"]
        # SUMO の同梱 DLL(bin 配下)を subprocess が解決できるよう SUMO_HOME/bin を PATH 先頭へ。
        # traci.start は os.environ を継承する(env 引数は無い)。sumo_pipeline._sumo_env と同流儀。
        h = sp.sumo_home()
        if h:
            os.environ["SUMO_HOME"] = h
            bindir = os.path.join(h, "bin")
            if bindir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")
        traci.start(cmd)
        self.traci = traci
        self._place_fleet()

    def _passenger_edges(self) -> list:
        edges = [e for e in self.net.getEdges()
                 if e.getFunction() != "internal" and e.allows("passenger")
                 and e.getLength() > 20.0]
        edges.sort(key=lambda e: e.getID())      # 決定論
        return edges

    def _place_fleet(self) -> None:
        """taxi fleet を決定論配置(ソート済みエッジへ等間隔)。idle-algorithm は既定 stop。"""
        traci = self.traci
        edges = self._passenger_edges()
        if not edges:
            raise RuntimeError("passenger エッジが net に無い")
        n = max(1, self.n_taxis)
        for i in range(n):
            eid = edges[(i * max(1, len(edges) // n)) % len(edges)].getID()
            rid = f"r_taxi{i}"
            traci.route.add(rid, [eid])
            traci.vehicle.add(f"taxi{i}", rid, typeID="taxi",
                              depart="now", line="taxi")
        for _ in range(2):           # fleet を挿入させる(空タンク回避)
            traci.simulationStep()
            self._now = traci.simulation.getTime()

    # ------------------------------------------------------------------ 写像
    def _xy_to_edge(self, xy) -> str | None:
        """本体ローカル座標(m)→ 経緯度 → net XY → 最寄り passenger エッジ id。"""
        try:
            lat, lon = sp.inv_project(float(xy[0]), float(xy[1]), self.origin)
            nx, ny = self.net.convertLonLat2XY(lon, lat)
            cands = self.net.getNeighboringEdges(nx, ny, 200.0)
            good = [(e, d) for (e, d) in cands
                    if e.getFunction() != "internal" and e.allows("passenger")]
            good.sort(key=lambda ed: (round(ed[1], 3), ed[0].getID()))  # 決定論
            return good[0][0].getID() if good else None
        except Exception:      # noqa: BLE001
            return None

    def _advance(self, until_s: float) -> None:
        while self._now < until_s:
            self.traci.simulationStep()
            self._now = self.traci.simulation.getTime()

    # ------------------------------------------------------------------ 予約解決
    def reserve(self, *, from_node, to_node, from_xy, to_xy,
                call_sim_sec: float) -> dict:
        """1件を同期解決。picked_up/dropped_off の実秒から wait_s/ride_s を返す。
        未配車(max_wait_s 内に拾えない)/ 経路不成立 は matched=False。決定論。"""
        traci = self.traci
        if traci is None:
            return {"matched": False}
        e_from = self._xy_to_edge(from_xy)
        e_to = self._xy_to_edge(to_xy)
        if not e_from or not e_to or e_from == e_to:
            return {"matched": False}
        # 呼んだ時刻へ同期(SUMO 秒時刻。過去なら現在時刻で解決=非減少)。
        self._advance(call_sim_sec)
        call_t = self._now
        pid = f"px{self._pid}"
        self._pid += 1
        try:
            traci.person.add(pid, e_from, pos=5.0, depart=self._now)
            traci.person.appendDrivingStage(pid, e_to, lines="taxi")
        except Exception:      # noqa: BLE001  — 経路/位置不成立=捕まらない扱い
            return {"matched": False}
        picked = None
        deadline_pickup = call_t + self.max_wait_s
        # --- 配車 → pickup 待ち ---
        while self._now < self._end:
            traci.simulationStep()
            self._now = traci.simulation.getTime()
            res = traci.person.getTaxiReservations(1)      # 1=new(未割当)
            if res:
                empty = traci.vehicle.getTaxiFleet(0)      # 0=空車
                if empty:
                    traci.vehicle.dispatchTaxi(sorted(empty)[0], [res[0].id])
            if pid in traci.person.getIDList():
                if traci.person.getVehicle(pid):           # 乗車済(=pickup)
                    picked = self._now
                    break
            else:                                          # 予約消滅=解決不能
                return {"matched": False}
            if self._now > deadline_pickup:                # 捕まらない
                try:
                    traci.person.remove(pid)
                except Exception:      # noqa: BLE001
                    pass
                return {"matched": False}
        if picked is None:
            return {"matched": False}
        # --- dropoff 待ち ---
        ride_deadline = picked + 3600.0
        while self._now < self._end:
            traci.simulationStep()
            self._now = traci.simulation.getTime()
            if pid not in traci.person.getIDList():        # 降車=到着
                dropped = self._now
                return {"matched": True, "wait_s": picked - call_t,
                        "ride_s": dropped - picked}
            if self._now > ride_deadline:                  # 乗車が長すぎる(異常)=打ち切り
                return {"matched": True, "wait_s": picked - call_t,
                        "ride_s": self._now - picked}
        return {"matched": True, "wait_s": picked - call_t,
                "ride_s": self._now - picked}

    # ------------------------------------------------------------------ 終了
    def close(self) -> None:
        if self.traci is not None:
            try:
                self.traci.close()
            except Exception:      # noqa: BLE001
                pass
            self.traci = None
        if self._add_file and os.path.isfile(self._add_file):
            try:
                os.remove(self._add_file)
            except OSError:
                pass
