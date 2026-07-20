"""エージェント視覚 v1: 顕著性駆動の POV 画像フェーズ(休眠骨格・既定 OFF)。

設計: docs/research/agent-vision.md §4 v1 / §5.2(R1)/ §2e(決定論の隔離)。

役割(ON のときだけ):
  1. 顕著性ゲート(**客観量のみ**): 初訪問ノード / 近傍人数しきい / 世界イベント現場 の
     いずれかで発火候補にし、専用 stream "pov_salience" の確率上限で撮る個体を絞る。
     **k 由来量(grievance/efficacy 等)・traits はゲートに一切入れない**(R1・§5.2)。
  2. CPU 決定論 POV レンダ(viz/render_pov.py)で画像を作り、サイドカー(out_dir/pov/)に PNG 保存。
  3. pov_image イベントに画像参照キーを載せる(実データは画像ストア=観測パイプラインは軽いまま)。
  4. VLM 配線は stub: cfg.pov.vlm.backend が "none" 以外かつ sim.vlm があれば mock 経路を通す
     (**text LLM の呼数には一切影響しない**=別 tier)。

配置は society 直下(engine/cognition/world/factors の静的検査対象外)。座標・訪問履歴・
世界イベントの客観量しか見ない(構成概念名・地名リテラルを持たない)。

R1(既定 OFF=バイト一致): pov.enabled=false のとき run_phase は即 return し、stream
"pov_salience" を一度も引かず・pov_image を 0 件・画像を 0 枚・agent に新フィールドを一切
生やさない。ON でも **追加 LLM 呼はゼロ**(画像は決定論の骨格に入力しない=観測ログのみ)。
画像レンダは画素決定(CPU)=同一入力→同一バイト(§2e)。
"""
from __future__ import annotations

import math
from pathlib import Path

from .world import scene_desc as scene_desc_mod
from .world.perception import hearers_of
from .observer.schema import Event

POV_DEFAULTS = {
    "enabled": False,
    "salience": {
        "first_visit": True,
        "crowd_min": 12,
        "world_event": True,
        "prob_cap": 0.05,
    },
    "render": {"width": 96, "height": 72, "fov_deg": 90.0, "max_dist_m": 200.0},
    "store": {"enabled": True, "dir": "pov"},
    "vlm": {"backend": "none"},
}

_renderer = None            # viz/render_pov.py の遅延ロード先(ON で撮る瞬間に初回ロード)


def build_cfg(raw) -> dict:
    """conf(pov)ブロックを正準化(既定 OFF=現行挙動と完全同一)。型強制つき。"""
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(raw):
            raw = OmegaConf.to_container(raw, resolve=True)
    except Exception:
        pass
    raw = dict(raw or {})
    cfg = {"enabled": bool(raw.get("enabled", False))}
    sal = dict(POV_DEFAULTS["salience"])
    for k, v in dict(raw.get("salience", {}) or {}).items():
        if k not in sal:
            continue
        sal[k] = int(v) if k == "crowd_min" else \
            (bool(v) if k in ("first_visit", "world_event") else float(v))
    cfg["salience"] = sal
    rnd = dict(POV_DEFAULTS["render"])
    for k, v in dict(raw.get("render", {}) or {}).items():
        if k not in rnd:
            continue
        rnd[k] = int(v) if k in ("width", "height") else float(v)
    cfg["render"] = rnd
    st = dict(POV_DEFAULTS["store"])
    for k, v in dict(raw.get("store", {}) or {}).items():
        if k in st:
            st[k] = bool(v) if k == "enabled" else str(v)
    cfg["store"] = st
    vlm = dict(POV_DEFAULTS["vlm"])
    for k, v in dict(raw.get("vlm", {}) or {}).items():
        if k in vlm:
            vlm[k] = str(v)
    cfg["vlm"] = vlm
    return cfg


def _cfg(sim) -> dict:
    """pov 設定を遅延構築して sim にキャッシュ(simulation.py を編集せずに済ませる)。"""
    cfg = getattr(sim, "_povcfg", None)
    if cfg is None:
        cfg = build_cfg(sim.cfg.get("pov", None))
        sim._povcfg = cfg
    return cfg


def enabled(sim) -> bool:
    return bool(_cfg(sim)["enabled"])


def _load_renderer():
    """viz/render_pov.py をファイルパスから遅延ロード(viz は sys.path 外=src のみ)。"""
    global _renderer
    if _renderer is None:
        import importlib.util
        p = Path(__file__).resolve().parents[2] / "viz" / "render_pov.py"
        spec = importlib.util.spec_from_file_location("render_pov", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _renderer = mod
    return _renderer


def _world_event_now(sim) -> bool:
    """当 step に「世界イベントの現場」条件が立っているか(客観な世界状態のみ)。

    災害の進行中(today_disaster_line が立つ)or 摂動シナリオ発動中。k・trait は見ない。"""
    if getattr(sim, "today_disaster_line", None):
        return True
    sc = getattr(sim, "scenario", None)
    return bool(getattr(sc, "active", False))


def run_phase(sim, step: int, sim_min: int) -> None:
    """POV フェーズ。既定 OFF=即 return(stream を引かず・イベント 0・画像 0・新フィールド無し)。

    位置確定後(percept_index 構築済み)に呼ばれる。撮影の抽選は専用 stream "pov_salience" のみ。"""
    cfg = _cfg(sim)
    if not cfg["enabled"]:
        return
    sal = cfg["salience"]
    idx = getattr(sim, "percept_index", None)
    radius = float(sim.cfg.world.perception_radius_m)
    world_now = bool(sal["world_event"] and _world_event_now(sim))
    prob_cap = float(sal["prob_cap"])
    crowd_min = int(sal["crowd_min"])
    want_first = bool(sal["first_visit"])

    for agent in sim.agents:                       # id 昇順=決定論
        if agent.loc == "outside" or agent.sleeping:
            continue
        if step < getattr(agent, "detained_until", 0):
            continue
        node = getattr(agent, "node", "") or ""
        seen = getattr(agent, "_pov_seen", None)
        if seen is None:
            seen = set()
            agent._pov_seen = seen
        first = bool(want_first and node and node not in seen)
        if node:
            seen.add(node)                          # 初訪問は「初めて居た step」で 1 度だけ立つ
        # --- 顕著性ゲート(客観量のみ)---
        reason = None
        if first:
            reason = "first_visit"
        elif crowd_min > 0 and idx is not None:
            if len(hearers_of(agent, idx, radius)) >= crowd_min:
                reason = "crowd"
        if reason is None and world_now:
            reason = "world_event"
        if reason is None:
            continue
        # --- 撮影確率(唯一の乱数消費=専用 stream。既存 draw 順に挿入しない)---
        rng = sim.hub.stream("pov_salience", agent.id, step)
        if float(rng.random()) >= prob_cap:
            continue
        _capture(sim, agent, step, sim_min, reason, cfg)


def _capture(sim, agent, step: int, sim_min: int, reason: str, cfg: dict) -> None:
    """1 個体の POV を CPU 決定論レンダ→サイドカー保存→pov_image イベント。追加 LLM 呼ゼロ。"""
    rnd = cfg["render"]
    heading_vec = scene_desc_mod.heading_of(agent, sim.city)
    ang = math.atan2(heading_vec[1], heading_vec[0]) if heading_vec else 0.0
    renderer = _load_renderer()
    png = renderer.render_pov(
        cam_x=float(agent.x), cam_y=float(agent.y), heading=float(ang),
        buildings=getattr(sim.city, "buildings", []) or [],
        terrain=getattr(sim, "elevation", None),
        width=int(rnd["width"]), height=int(rnd["height"]),
        fov_deg=float(rnd["fov_deg"]), max_dist_m=float(rnd["max_dist_m"]))

    ref = f"{int(step)}_{int(agent.id)}"
    store = cfg["store"]
    if store["enabled"]:
        pov_dir = Path(sim.out_dir) / str(store["dir"])
        pov_dir.mkdir(parents=True, exist_ok=True)
        (pov_dir / f"{ref}.png").write_bytes(png)

    payload = {"ref": ref, "w": int(rnd["width"]), "h": int(rnd["height"]),
               "trigger": reason}
    # VLM stub 経路(mock のみ・実 VLM は本選 Day-0)。text LLM の呼数には影響しない(別 tier)。
    vlm = getattr(sim, "vlm", None)
    if cfg["vlm"]["backend"] != "none" and vlm is not None:
        try:
            cap = vlm.describe(png)
            if cap:
                payload["vlm"] = True
        except Exception:
            pass

    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                         kind="pov_image", x=agent.x, y=agent.y, payload=payload))
