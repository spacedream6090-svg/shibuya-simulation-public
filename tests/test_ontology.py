"""群のオントロジー(文化圏×経験の世界観共有群。src/society/ontology.py)のテスト。

設計(ユーザー承認済み 2026-07-21・R1 doctrine 継承):
- S-a 割当: 安定ハッシュ(ontology.seed, persona id)=run.seed 非依存・k/trait 非参照・乱数ゼロ。
      OFF=属性なし=バイト一致。ON=経験1行がプロンプトに載る・呼数 k 非依存・composition 比率に収束。
- S-b worldview: 群ごとの初期オフセットを構築時に1回だけ適用。以後の更新則は不変(worldview.py 無改変)。
- S-c 計測: agents.json に群 id(ON 時のみキー追加)。
検証は mock / 固定 LLM のみ(実LLM 禁止・≤288 step)。ontology は純関数=乱数不使用・追加 LLM 呼ゼロ。
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "src"))

from society import ontology                              # noqa: E402
from society import worldview                             # noqa: E402
from society.cognition import deliberate                  # noqa: E402
from society.config import load_config                    # noqa: E402
from society.engine.simulation import Simulation          # noqa: E402
from society.world import pool as pool_mod                # noqa: E402
from society.world.presence import present_for_day        # noqa: E402
from society.rng import RngHub                            # noqa: E402

_ON = {"ontology.enabled": "true"}
_OFF = {"ontology.enabled": "false"}


def _sim(tmp_path, name, n=20, steps=24, **ov):
    dot = ["run.seed=42", f"run.n_agents={n}", f"run.n_steps={steps}",
           f"run.name={name}", "model.backend=mock"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _oc(enabled=True, **raw):
    return ontology.build_cfg({"enabled": enabled, "seed": 20260721,
                               "groups": {
                                   "jp_metro": {"label": "a", "line": "L_metro"},
                                   "jp_other": {"label": "b", "line": "L_other"},
                                   "asia_visit": {"label": "c", "line": "L_asia",
                                                  "wv_offsets": {"controllability": -0.05}},
                                   "west_visit": {"label": "d", "line": "L_west",
                                                  "wv_offsets": {"controllability": -0.1}}},
                               "composition": {
                                   "resident": {"jp_metro": 1.0},
                                   "stochastic": {"jp_metro": 0.45, "jp_other": 0.40,
                                                  "asia_visit": 0.10, "west_visit": 0.05},
                                   "default": {"jp_metro": 0.8, "jp_other": 0.2}},
                               **raw})


# =============================================================== 純関数(単体)
def test_stable_uniform_deterministic_and_process_stable():
    """同 (seed, pid) は常に同一値・[0,1)・pid/seed で変わる(hashlib=プロセス跨ぎ安定)。"""
    u = ontology._stable_uniform(20260721, "L4_00000001")
    assert 0.0 <= u < 1.0
    assert u == ontology._stable_uniform(20260721, "L4_00000001")   # 決定論
    assert u != ontology._stable_uniform(20260721, "L4_00000002")   # pid で変わる
    assert u != ontology._stable_uniform(1, "L4_00000001")          # seed で変わる


def test_assign_deterministic_pure():
    """assign_group は persona id のみの純関数(同 cfg・同 pid=同一群)。run.seed を引数に取らない。"""
    oc = _oc()
    for pid in ("L4_1", "abc", "42", "L2_00099"):
        g = ontology.assign_group(oc, pid, "stochastic")
        assert g == ontology.assign_group(oc, pid, "stochastic")
        assert g in oc["groups"]


def test_composition_converges_large_n():
    """大数で composition 比率に収束(許容誤差 ±0.02)。stochastic は4群に分割される。"""
    oc = _oc()
    N = 40000
    c = Counter(ontology.assign_group(oc, f"P_{i}", "stochastic") for i in range(N))
    frac = {k: v / N for k, v in c.items()}
    assert abs(frac["jp_metro"] - 0.45) < 0.02
    assert abs(frac["jp_other"] - 0.40) < 0.02
    assert abs(frac["asia_visit"] - 0.10) < 0.02
    assert abs(frac["west_visit"] - 0.05) < 0.02


def test_unknown_tier_falls_back_to_default():
    """未知の tier は default composition に後退(default は jp_metro/jp_other のみ)。"""
    oc = _oc()
    gs = {ontology.assign_group(oc, f"P_{i}", "__nope__") for i in range(500)}
    assert gs <= {"jp_metro", "jp_other"}, "default 後退が効いていない"


def test_disabled_returns_none():
    oc = _oc(enabled=False)
    assert ontology.assign_group(oc, "x", "default") is None


def test_group_line_and_offset():
    oc = _oc()
    assert ontology.group_line(oc, "west_visit") == "L_west"
    assert ontology.group_line(oc, None) is None
    # S-b: 全員 0.5 起点+群オフセット(clip)。無指定群は None(=既定 0.5 のまま)。
    assert ontology.initial_controllability(oc, "west_visit") == pytest.approx(0.4)
    assert ontology.initial_controllability(oc, "asia_visit") == pytest.approx(0.45)
    assert ontology.initial_controllability(oc, "jp_metro") is None


# =============================================================== OFF=バイト一致
def test_off_matches_pure_default(tmp_path):
    """明示 OFF と純粋既定が L1 完全一致。群属性も作られない(seam が no-op)。"""
    pure = _sim(tmp_path, "pure")
    pure.run()
    off = _sim(tmp_path, "off", **_OFF)
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(ontology seam が no-op でない)"
    assert all(getattr(a, "ontology_group", None) is None for a in pure.agents)
    assert all(getattr(a, "ontology_line", None) is None for a in pure.agents)


def test_off_matches_pure_default_long(tmp_path):
    """ゴールデンと同じ 144step 尺でも OFF は純粋既定とバイト一致。"""
    pure = _sim(tmp_path, "purel", n=15, steps=144)
    pure.run()
    off = _sim(tmp_path, "offl", n=15, steps=144, **_OFF)
    off.run()
    assert _l1(pure) == _l1(off)


# =============================================================== build_prompt バイト同一
def test_build_prompt_seam_byte_identical(tmp_path):
    """ontology_line 属性なし(OFF)は既定 build_prompt とバイト同一。ON は persona 直後に1行増える。"""
    sim = _sim(tmp_path, "seam", steps=1)
    sim.run()
    a = sim.agents[0]
    base = deliberate.build_prompt(a, place_name="路上", surprise="solo",
                                   nearby_names=[], sim_min=600, step=3)
    a.ontology_line = "海外からの旅行者で、地震をほとんど経験したことがない。"
    on = deliberate.build_prompt(a, place_name="路上", surprise="solo",
                                 nearby_names=[], sim_min=600, step=3)
    assert a.ontology_line in on and on != base
    # persona の直後(自己モデル等より前)に載る
    assert on.index(a.ontology_line) > on.index(a.persona)


# =============================================================== ON: 割当・注入・観測
def test_on_prompt_has_group_line(tmp_path):
    """ON の mock 24step で、実発火プロンプトに群の経験1行が実際に載る。"""
    sim = _sim(tmp_path, "appear", n=40, **_ON)

    class _Rec:
        def __init__(self, inner):
            self._inner = inner
            self.prompts = []

        def generate(self, prompt, **kw):
            self.prompts.append(prompt)
            return self._inner.generate(prompt, **kw)

        @property
        def calls(self):
            return self._inner.calls

        @property
        def hits(self):
            return self._inner.hits
    sim.llm = _Rec(sim.llm)
    sim.run()
    lines = {ontology.group_line(sim.ontologycfg, g)
             for g in sim.ontologycfg["groups"]}
    assert any(any(l in p for l in lines) for p in sim.llm.prompts), \
        "発火プロンプトに群の経験1行が一度も載っていない"


def test_on_agents_json_records_group(tmp_path):
    """S-c: ON は agents.json に群 id を記録。OFF はキーを持たない。"""
    on = _sim(tmp_path, "aj_on", **_ON)
    on.run()
    meta = json.loads((on.out_dir / "agents.json").read_text(encoding="utf-8"))
    assert all("ontology_group" in m for m in meta)
    assert all(m["ontology_group"] in on.ontologycfg["groups"] for m in meta)
    off = _sim(tmp_path, "aj_off", **_OFF)
    off.run()
    meta_off = json.loads((off.out_dir / "agents.json").read_text(encoding="utf-8"))
    assert all("ontology_group" not in m for m in meta_off)


def test_same_cfg_same_assignment(tmp_path):
    """同一 cfg=同一割当(決定論)。2回構築で {agent.id: group} が完全一致。"""
    a = _sim(tmp_path, "sc_a", **_ON)
    b = _sim(tmp_path, "sc_b", **_ON)
    ga = {x.id: x.ontology_group for x in a.agents}
    gb = {x.id: x.ontology_group for x in b.agents}
    assert ga == gb and ga


def test_run_seed_invariant_assignment(tmp_path):
    """run.seed を変えても割当不変(割当は run.seed を引かない=比較実験の要)。"""
    a = _sim(tmp_path, "rs_a", **{**_ON, "run.seed": "1"})
    b = _sim(tmp_path, "rs_b", **{**_ON, "run.seed": "999"})
    ga = {x.id: x.ontology_group for x in a.agents}
    gb = {x.id: x.ontology_group for x in b.agents}
    assert ga == gb and ga


def test_on_deterministic(tmp_path):
    """ON 同士 2 回で L1 完全一致(mock・乱数ゼロの決定論)。"""
    a = _sim(tmp_path, "det_a", **_ON)
    a.run()
    b = _sim(tmp_path, "det_b", **_ON)
    b.run()
    assert _l1(a) == _l1(b), "ON の決定論が崩れている(ontology が乱数を引いている疑い)"


# =============================================================== R1: 呼数不変
class _FixedLLM:
    def __init__(self):
        self.response = json.dumps({"action": "speak", "text": "やあ"},
                                   ensure_ascii=False)
        self.calls = 0
        self.hits = 0

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        self.calls += 1
        return self.response, str(self.calls), False


def test_call_count_invariant_on_off(tmp_path):
    """固定 LLM で ontology ON/OFF の generate 呼数が完全一致(追加 LLM 呼ゼロ=R1)。"""
    def run(name, on):
        sim = _sim(tmp_path, name, **{"ontology.enabled": str(on).lower()})
        sim.llm = _FixedLLM()
        sim.run()
        return sim
    on = run("cc_on", True)
    off = run("cc_off", False)
    assert on.llm.calls == off.llm.calls and on.llm.calls > 0, \
        f"ontology で呼数が変化(R1 違反): on={on.llm.calls} off={off.llm.calls}"


def test_call_count_k_invariant(tmp_path):
    """真の R1: ON のまま k(writeback)を free/off に振っても LLM 呼数が不変。"""
    free = _sim(tmp_path, "k_free", **{**_ON, "k.writeback": "free"})
    free.run()
    off = _sim(tmp_path, "k_off", **{**_ON, "k.writeback": "off"})
    off.run()
    assert len(free.logger.llm_calls) == len(off.logger.llm_calls), \
        f"ON の呼数が k で乖離(R1 違反): free={len(free.logger.llm_calls)} off={len(off.logger.llm_calls)}"


# =============================================================== S-b worldview オフセット
def test_wv_offset_applied_at_construction(tmp_path):
    """S-b: 群オフセットを構築時に1回だけ適用(offset 群=0.5+offset・無指定群=属性なし)。

    default 構成に offset 群(jp_metro に -0.1 を注入)を載せ、その群の在住者だけ 0.4 起点になることを
    構築直後(run 前)に確認。offset 無指定群は属性が作られない(worldview 既定 0.5 のまま)。"""
    sim = _sim(tmp_path, "wv", n=60,
               **{**_ON, "worldview.enabled": "true",
                  "ontology.groups.jp_metro.wv_offsets.controllability": "-0.1"})
    off_expect = ontology.initial_controllability(sim.ontologycfg, "jp_metro")
    assert off_expect == pytest.approx(0.4)
    seen_metro = seen_other = False
    for a in sim.agents:
        if a.ontology_group == "jp_metro":
            assert a.controllability == pytest.approx(0.4)
            seen_metro = True
        elif a.ontology_group == "jp_other":            # offset 無指定=属性を作らない
            assert getattr(a, "controllability", None) is None
            seen_other = True
    assert seen_metro and seen_other, "両群のサンプルが揃わなかった(N を増やす)"


def test_wv_update_rule_unchanged(tmp_path):
    """S-b: 以後の更新則は worldview 従来どおり(worldview.py 無改変)。オフセット起点から正しく動く。"""
    sim = _sim(tmp_path, "wvd", n=40, steps=288,
               **{**_ON, "worldview.enabled": "true",
                  "ontology.groups.jp_metro.wv_offsets.controllability": "-0.1"})
    sim.run()
    # 応答走査(_ctrl_apply)はオフセット起点でも同一式で動く: 0.4 起点に正の応答→式どおり増える。
    cfg = sim.worldviewcfg
    from types import SimpleNamespace
    ref = SimpleNamespace(controllability=0.4)
    worldview._ctrl_apply(ref, True, 1.0, cfg)
    base = SimpleNamespace(controllability=0.4)
    worldview._ctrl_apply(base, True, 1.0, cfg)
    assert ref.controllability == base.controllability   # 更新式は起点値のみの関数=不変
    # ON でも worldview 日次スナップが出る(観測機構は生きている)
    assert any(e.kind == "worldview" for e in sim.logger.events)


# =============================================================== pool 経路(P3 hydrate)
@pytest.fixture(scope="module")
def small_pool(tmp_path_factory):
    """P5 生成器で小プールを tmp に生成(実プール 736MB は触らない)。test_pool_rotation と同流儀。"""
    import build_persona_pool as bpp
    out = tmp_path_factory.mktemp("pool_onto")
    orgs = json.loads((_ROOT / "data" / "organizations_shibuya_wide11k.json")
                      .read_text(encoding="utf-8"))
    pop = json.loads((_ROOT / "data" / "shibuya_population.json")
                     .read_text(encoding="utf-8"))
    bpp.build_pool(out, seed=42, fraction=0.001, orgs=orgs, pop=pop,
                   total_target=1_000_000)
    return out


def _pool_cfg(name, pool_dir, n_steps, cap=400, **ov):
    dot = ["run.seed=42", "run.n_agents=10", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock",
           "pool.enabled=true", f"pool.dir={pool_dir}", f"pool.present_cap={cap}"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def test_pool_group_matches_pure_function(small_pool, tmp_path):
    """pool ON: 全 present 個体の群=assign_group(pid, presence 層)の純関数値(tier=P3 presence キー)。"""
    sim = Simulation(_pool_cfg("op", small_pool, n_steps=1, **_ON), out_dir=tmp_path / "op")
    oc = sim.ontologycfg
    assert sim.agents
    for a in sim.agents:
        rec = sim._pool.get(a.pool_pid)
        tier = str(rec.get("presence", "resident"))
        assert a.ontology_group == ontology.assign_group(oc, a.pool_pid, tier)


def test_pool_hydrate_reentry_same_group(small_pool, tmp_path):
    """P3 再来街(dormant→hydrate)でも同一群。退場→再入場を跨いで群が不変(pid 純関数)。"""
    ps = pool_mod.PoolStore(small_pool)
    recs = ps.presence_records()
    hub = RngHub(42)
    s0 = set(present_for_day(recs, 0, 400, hub, 0))
    s1 = set(present_for_day(recs, 1, 400, hub, 1))
    entrants = sorted(s1 - s0)
    assert entrants, "day0 不在→day1 present の再来街候補が見つからない"
    x = entrants[0]

    sim = Simulation(_pool_cfg("hy", small_pool, n_steps=210, **_ON), out_dir=tmp_path / "hy")
    oc = sim.ontologycfg
    tier = str(ps.get(x).get("presence", "resident"))
    expect = ontology.assign_group(oc, x, tier)
    # dormant にマーカー付きで仕込む → hydrate 経路を確実に通す
    sim._dormant.save(x, {"beliefs": ["__onto_mark__"], "money": 555.0})
    sim.run()
    live = {a.pool_pid: a for a in sim.agents}
    assert x in live, "再来街者 x が day1 に present になっていない"
    assert "__onto_mark__" in live[x].beliefs                   # hydrate を通った証拠
    assert live[x].ontology_group == expect, "hydrate 経由で群が変わった(pid 純関数が崩れている)"


def test_pool_determinism_same_seed(small_pool, tmp_path):
    """同 seed 2 回で present 個体の群割当が完全一致(pool ON の決定論)。"""
    a = Simulation(_pool_cfg("pd1", small_pool, n_steps=1, **_ON), out_dir=tmp_path / "pd1")
    b = Simulation(_pool_cfg("pd2", small_pool, n_steps=1, **_ON), out_dir=tmp_path / "pd2")
    ga = {x.pool_pid: x.ontology_group for x in a.agents}
    gb = {x.pool_pid: x.ontology_group for x in b.agents}
    assert ga == gb and ga
