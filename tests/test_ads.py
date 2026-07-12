"""街路の環境情報(第18バッチ 2026-07-11): 街頭広告(OOH)+群衆の視覚情報のテスト。

設計(docs/research/commercial-analytics.md・ユーザー決定 2026-07-11):
- ads.enabled=true のときだけ掲出地点の視認判定(新 stream "ads")→ ad_exposure を L1 に
  記録し、発火時プロンプトに中立1行(想起窓内のみ)。広告主=シミュ内 POI(ファネルが閉じる)。
- crowd_visual.enabled=true のときだけ同席者の決定論要約1行(乱数なし・実在の集計のみ)。
- 既定 OFF=draw・イベント・プロンプトともバイト一致。R1: 呼数不変(内容差のみ)。
"""
from __future__ import annotations

import json

from society import street
from society.cognition.deliberate import build_prompt
from society.config import load_config
from society.engine.simulation import Simulation


def _cfg(name, n=20, steps=144, **ov):
    dot = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
           "observer.snapshot_every=144"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def _sim(tmp_path, name, n=20, steps=144, **ov):
    return Simulation(_cfg(name, n, steps, **ov), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _kind(sim, kind):
    return [e for e in sim.logger.events if e.kind == kind]


def _ads_on_cfg(tmp_path, name, steps=288, crowd=False):
    """実在 POI 名で掲出枠を張った ON 設定(視認率1.0・全域半径=必ず接触が出る)。"""
    probe = Simulation(_cfg(name + "_probe", steps=1),
                       out_dir=tmp_path / (name + "_probe"))
    pois = [p for p in probe.city.poi_list if str(p.get("name") or "").strip()]
    slot_name = sorted(str(p["name"]) for p in pois)[0]
    cats = sorted({str(p.get("cat") or "") for p in pois if p.get("cat")})
    cfg = _cfg(name, n=20, steps=steps, **{
        "ads.enabled": "true", "ads.p_see_large": "1.0",
        "ads.radius_m": "99999.0", "ads.cooldown_steps": "6",
        "ads.daily_cap": "4"})
    cfg.ads.large = [slot_name]
    cfg.ads.target_cats = cats
    if crowd:
        cfg.crowd_visual.enabled = True
    return cfg


# --------------------------------------------------------------------- OFF 既定
def test_off_matches_pure_default(tmp_path):
    """明示 OFF と純粋既定が L1 完全一致。広告状態なし・イベント 0 件。"""
    pure = _sim(tmp_path, "pure")
    pure.run()
    off = _sim(tmp_path, "off", **{"ads.enabled": "false",
                                   "crowd_visual.enabled": "false"})
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(第18バッチ seam が no-op でない)"
    assert not _kind(pure, "ad_exposure") and not _kind(pure, "ad_campaign")
    assert getattr(pure, "_ads_slots", None) is None
    assert all(getattr(a, "ad_recent", None) is None for a in pure.agents)


# --------------------------------------------------------------------- 単体
def test_unit_lines():
    """ads_line / crowd_line の純関数(想起窓・有効フリークエンシー・年齢構成)。"""
    class _A:
        pass
    cfg = street.build_ads_cfg({"enabled": True})
    a = _A()
    assert street.ads_line(a, cfg, step=10) is None            # 接触なし
    a.ad_recent = (10, "枠#p0", "渋谷の喫茶店")
    a.ad_seen = {"枠#p0": 1}
    line = street.ads_line(a, cfg, step=20)
    assert line and "渋谷の喫茶店" in line and "何度も" not in line
    a.ad_seen = {"枠#p0": 3}                                   # 有効フリークエンシー
    assert "何度も" in street.ads_line(a, cfg, step=20)
    assert street.ads_line(a, cfg, step=10 + 145) is None      # 想起窓(144)超え
    assert street.ads_line(a, None, step=20) is None           # OFF

    ccfg = street.build_crowd_cfg({"enabled": True})
    assert street.crowd_line([], None) is None                 # OFF
    assert "人影は少ない" in street.crowd_line([], ccfg)
    young = [type("P", (), {"age": 22})() for _ in range(6)]
    old = [type("P", (), {"age": 70})() for _ in range(2)]
    line = street.crowd_line(young + old, ccfg)
    assert "賑わっている" in line and "若い人が目立つ" in line
    assert "まばら" in street.crowd_line(old[:1], ccfg)


def test_slot_resolution(tmp_path):
    """掲出地点の決定論解決(完全一致・部分一致・未解決は黙って落とす)。"""
    sim = _sim(tmp_path, "res", steps=1)
    pois = [p for p in sim.city.poi_list if str(p.get("name") or "").strip()]
    name = sorted(str(p["name"]) for p in pois)[0]
    cfg = street.build_ads_cfg({"enabled": True, "large": [name],
                                "slots": [name[:2], "存在しない場所XYZ"]})
    slots = street.resolve_slots(sim.city, cfg)
    assert any(s["name"] == name and s["large"] for s in slots), "完全一致が解決されない"
    assert all(s["name"] != "存在しない場所XYZ" for s in slots), "未解決名が残っている"
    assert slots == sorted(slots, key=lambda s: s["name"]), "名前順でない(決定論)"


# --------------------------------------------------------------------- ON 挙動
def test_on_exposures_and_prompt(tmp_path):
    """ON: キャンペーン改定+接触が記録され、cooldown/daily_cap を守り、プロンプトに載る。"""
    cfg = _ads_on_cfg(tmp_path, "on", steps=288)
    sim = Simulation(cfg, out_dir=tmp_path / "on")
    sim.run()
    camps = _kind(sim, "ad_campaign")
    evs = _kind(sim, "ad_exposure")
    assert camps, "キャンペーン改定イベントが出ない"
    assert evs, "ON なのに ad_exposure が 1 件も出ない"
    for e in evs[:20]:
        assert set(e.payload) == {"campaign", "target", "slot", "cat", "n_seen"}
    # cooldown: 同一 agent×枠の接触は 6step 以上あける
    seen: dict = {}
    for e in evs:
        key = (e.agent_id, e.payload["slot"])
        if key in seen:
            assert e.step - seen[key] >= 6, f"cooldown 違反: {key}"
        seen[key] = e.step
    # daily_cap: agent×日で 4 件以下
    per_day: dict = {}
    for e in evs:
        key = (e.agent_id, e.sim_min // 1440)
        per_day[key] = per_day.get(key, 0) + 1
    assert max(per_day.values()) <= 4, "daily_cap 違反"
    # 反復接触で n_seen が育つ(288step=2日・視認率1.0なら必ず)
    assert any(e.payload["n_seen"] >= 2 for e in evs), "n_seen が増えない"
    # 接触済みエージェントの発火プロンプトに広告1行が載る
    agent = next(a for a in sim.agents if getattr(a, "ad_recent", None))
    line = street.ads_line(agent, sim.adscfg, step=agent.ad_recent[0])
    p = build_prompt(agent, place_name="路上", surprise=None, nearby_names=[],
                     step=agent.ad_recent[0], ads_line=line)
    assert "広告" in p and agent.ad_recent[2] in p, "プロンプトに広告行が載らない"


def test_on_deterministic(tmp_path):
    """ON(広告+群衆視覚)同士 2 回で L1 完全一致(mock・決定論)。"""
    a = Simulation(_ads_on_cfg(tmp_path, "det_a", crowd=True),
                   out_dir=tmp_path / "det_a")
    a.run()
    b = Simulation(_ads_on_cfg(tmp_path, "det_b", crowd=True),
                   out_dir=tmp_path / "det_b")
    b.run()
    assert _l1(a) == _l1(b), "ON の決定論が崩れている"


# --------------------------------------------------------------------- R1 呼数不変
class _FixedLLM:
    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.hits = 0

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        self.calls += 1
        return self.response, str(self.calls), False


def test_r1_call_count_invariant(tmp_path):
    """応答固定 backend: ads+crowd ON/OFF で generate 呼数が完全一致(288 step)。"""
    def run(name, on):
        if on:
            cfg = _ads_on_cfg(tmp_path, name, crowd=True)
        else:
            cfg = _cfg(name, n=20, steps=288, **{"ads.enabled": "false"})
        sim = Simulation(cfg, out_dir=tmp_path / name)
        sim.llm = _FixedLLM(json.dumps({"action": "speak", "text": "x"},
                                       ensure_ascii=False))
        sim.run()
        return sim
    on = run("r1_on", True)
    off = run("r1_off", False)
    assert on.llm.calls == off.llm.calls and on.llm.calls > 0, \
        f"呼数が一致しない: ON={on.llm.calls} OFF={off.llm.calls}"
