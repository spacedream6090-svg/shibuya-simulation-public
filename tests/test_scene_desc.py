"""エージェント視覚 v0(構造化シーン記述 world/scene_desc)のテスト。

設計: docs/research/agent-vision.md §4 v0 / §5.4(行間レイヤ整合)。方針(R1 の鉄則を継承):
- OFF(既定): build_prompt がバイト同一・L1 が純粋既定と完全一致(ゴールデンは test_scenario が担保)。
- ON(mock ≤24step): (a)発火プロンプトに視界/注視/垂直の行が実際に載る。(b)同 seed 2回で L1 一致
  (決定論・追加乱数ゼロ)。(c)LLM 呼数は OFF と同一(追加呼ゼロ=k 非依存・R1)。(d)LOS 遮蔽が効く。
  (e)input_res 水準で注視件数が変わる(呼数・乱数・発火は不変)。(f)heading 導出・方向分類。
  (g)行間レイヤ配線(S2 ダイジェスト視覚1行・S3 会話 payload メタ)。
検証は mock / 固定 LLM のみ(実LLM 禁止・≤24 step)。scene_desc は純関数=乱数不使用・追加 LLM 呼ゼロ。
"""
from __future__ import annotations

import json

from society.cognition import deliberate, lod
from society.config import load_config
from society.engine.simulation import Simulation
from society.world import scene_desc as sd

_ON = {"world.scene_desc.enabled": "true"}
_OFF = {"world.scene_desc.enabled": "false"}


def _sim(tmp_path, name, n=25, steps=24, **ov):
    dot = ["run.seed=42", f"run.n_agents={n}", f"run.n_steps={steps}",
           f"run.name={name}"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


class _RecordingLLM:
    def __init__(self, inner):
        self._inner = inner
        self.prompts: list[str] = []

    def generate(self, prompt, **kw):
        self.prompts.append(prompt)
        return self._inner.generate(prompt, **kw)

    @property
    def calls(self):
        return self._inner.calls

    @property
    def hits(self):
        return self._inner.hits


class _FixedLLM:
    """内容非依存の固定応答(補間/シーン注入が generate() を1本も足さないことを純粋に測る)。"""

    def __init__(self):
        self.response = json.dumps({"action": "speak", "text": "やあ"},
                                   ensure_ascii=False)
        self.calls = 0
        self.hits = 0

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        self.calls += 1
        return self.response, str(self.calls), False


# ------------------------------------------------------ build_prompt バイト同一
def test_build_prompt_seam_byte_identical_off(tmp_path):
    """scene_lines=None(既定)は引数なしの build_prompt とバイト同一。ON は行が増える。"""
    sim = _sim(tmp_path, "seam", steps=1)
    sim.run()
    agent = sim.agents[0]
    base = deliberate.build_prompt(agent, place_name="路上", surprise="solo",
                                   nearby_names=[], sim_min=600, step=3)
    none = deliberate.build_prompt(agent, place_name="路上", surprise="solo",
                                   nearby_names=[], sim_min=600, step=3,
                                   scene_lines=None)
    assert none == base, "scene_lines=None が既定と非同一(バイト一致が崩れている)"
    on = deliberate.build_prompt(agent, place_name="路上", surprise="solo",
                                 nearby_names=[], sim_min=600, step=3,
                                 scene_lines=["見えるもの: 正面にビル。"])
    assert "見えるもの: 正面にビル。" in on and on != base


# ------------------------------------------------------ OFF: L1 が純粋既定と一致
def test_off_matches_pure_default(tmp_path):
    """明示 OFF と純粋既定が L1 完全一致(24step)。scene seam が no-op であることを固定。"""
    pure = _sim(tmp_path, "pure")
    pure.run()
    off = _sim(tmp_path, "expl_off", **_OFF)
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(scene seam が no-op でない)"


def test_off_matches_pure_default_long(tmp_path):
    """ゴールデンと同じ 144step 尺でも OFF は純粋既定とバイト一致。"""
    pure = _sim(tmp_path, "purel", n=15, steps=144)
    pure.run()
    off = _sim(tmp_path, "offl", n=15, steps=144, **_OFF)
    off.run()
    assert _l1(pure) == _l1(off)


# ------------------------------------------------------ (a) シーン行がプロンプトに載る
def test_scene_appears_in_prompt_on(tmp_path):
    """ON の mock 24step で、実発火プロンプトに視界/注視/垂直のいずれかの行が載る。"""
    sim = _sim(tmp_path, "appear", n=40, **_ON)
    sim.llm = _RecordingLLM(sim.llm)
    sim.run()
    assert any(("見えるもの:" in p or "目に留まるもの:" in p or "位置関係:" in p)
               for p in sim.llm.prompts), "発火プロンプトにシーン行が一度も載っていない"


# ------------------------------------------------------ (b) 決定論
def test_on_deterministic(tmp_path):
    a = _sim(tmp_path, "det_a", **_ON)
    a.run()
    b = _sim(tmp_path, "det_b", **_ON)
    b.run()
    assert _l1(a) == _l1(b), "ON の決定論が崩れている(scene が乱数を引いている疑い)"


# ------------------------------------------------------ (c) LLM 呼数不変(追加呼ゼロ)
def _run_fixed(tmp_path, name, enabled):
    sim = _sim(tmp_path, name, **{"world.scene_desc.enabled": enabled})
    sim.llm = _FixedLLM()
    sim.run()
    return sim


def test_call_count_invariant(tmp_path):
    """scene ON/OFF で LLM 呼数が完全一致(追加 LLM 呼ゼロ=R1)。固定 LLM=内容非依存。"""
    on = _run_fixed(tmp_path, "cc_on", "true")
    off = _run_fixed(tmp_path, "cc_off", "false")
    assert on.llm.calls == off.llm.calls and on.llm.calls > 0, \
        f"scene で呼数が変化(R1 違反): on={on.llm.calls} off={off.llm.calls}"


def test_call_count_k_invariant(tmp_path):
    """真の R1: ON のまま k(writeback)を free/off に振っても LLM 呼数が不変。"""
    free = _sim(tmp_path, "k_free", **{**_ON, "k.writeback": "free"})
    free.run()
    off = _sim(tmp_path, "k_off", **{**_ON, "k.writeback": "off"})
    off.run()
    assert len(free.logger.llm_calls) == len(off.logger.llm_calls), \
        f"ON の呼数が k で乖離(R1 違反): free={len(free.logger.llm_calls)} off={len(off.logger.llm_calls)}"


# ------------------------------------------------------ (d) LOS 遮蔽が効く
def test_los_occlusion_drops_person(tmp_path):
    """注視対象リストは occluder.blocks() を尊重する(壁で隔てられた人が消える)。"""
    sim = _sim(tmp_path, "los", steps=1, **_ON)
    a = sim.agents[0]

    class _P:
        def __init__(self, i, x, y, n):
            self.id, self.x, self.y, self.name = i, x, y, n
            self.building, self.floor, self.loc = None, 0, "street"

    class _Occ:
        def blocks(self, ag, p):
            return p.name == "遮蔽側"

    company = [_P(1, a.x + 5, a.y, "可視側"), _P(2, a.x + 6, a.y, "遮蔽側")]
    noocc = sd.scene_lines(a, city=sim.city, company=company,
                           cfg=sim.scenecfg, elevation=None)
    withocc = sd.scene_lines(a, city=sim.city, company=company,
                             cfg=sim.scenecfg, occluder=_Occ(), elevation=None)
    assert any("遮蔽側" in l for l in noocc), "遮蔽なしで対象が見えていない"
    assert not any("遮蔽側" in l for l in withocc), "遮蔽したのに注視に残っている"
    assert any("可視側" in l for l in withocc), "可視の相手まで消えている"


# ------------------------------------------------------ (e) input_res 水準で件数が変わる
def test_lod_scene_n_changes_count(tmp_path):
    """narrow(scene_n=2)/wide(scene_n=4)で注視件数が変わる(候補が十分あるとき)。"""
    sim = _sim(tmp_path, "lodn", steps=1, **_ON)
    a = sim.agents[0]

    class _P:
        def __init__(self, i, x, y, n):
            self.id, self.x, self.y, self.name = i, x, y, n
            self.building, self.floor, self.loc = None, 0, "street"

    company = [_P(i, a.x + i, a.y, f"人{i}") for i in range(1, 7)]

    def _gaze_count(level):
        a.input_res = dict(lod.INPUT_RES_DEFAULTS["levels"][level], level=level)
        lines = sd.scene_lines(a, city=sim.city, company=company,
                               cfg=sim.scenecfg, elevation=None)
        gaze = next((l for l in lines if l.startswith("目に留まるもの:")), "")
        return gaze.count("(")            # 各項目は距離帯 "(…)" を1個持つ

    n_narrow = _gaze_count("narrow")
    n_wide = _gaze_count("wide")
    assert n_narrow == 2 and n_wide == 4, \
        f"scene_n の LOD が効いていない: narrow={n_narrow} wide={n_wide}"


def test_lod_scene_n_present_in_defaults():
    """LOD 既定に scene_n が入り、mid=3(現行相当)。既存の6キー契約は別テストが固定。"""
    lv = lod.INPUT_RES_DEFAULTS["levels"]
    assert lv["narrow"]["scene_n"] == 2
    assert lv["mid"]["scene_n"] == 3
    assert lv["wide"]["scene_n"] == 4


# ------------------------------------------------------ (f) heading 導出・方向分類
def test_heading_derivation(tmp_path):
    """route の次ノードへの向きから heading を導出。静止(route/dest なし)は None。"""
    sim = _sim(tmp_path, "head", steps=1, **_ON)
    a = sim.agents[0]
    a.route, a.dest = [], None
    assert sd.heading_of(a, sim.city) is None, "静止なのに heading が出ている"
    nbrs = sorted(sim.city.graph.neighbors(a.node))
    a.route = [nbrs[0]]
    h = sd.heading_of(a, sim.city)
    assert h is not None and abs((h[0] ** 2 + h[1] ** 2) - 1.0) < 1e-6, "heading が単位ベクトルでない"


def test_directional_buckets(tmp_path):
    """heading があるとき視界の要素が 正面/右手/左手 に分類される(視界外は除外)。"""
    sim = _sim(tmp_path, "dir", steps=1, **_ON)
    a = sim.agents[0]
    nbrs = sorted(sim.city.graph.neighbors(a.node))
    a.route = [nbrs[0]]
    h = sd.heading_of(a, sim.city)
    items = sd._directional(a, sim.city, h, sim.scenecfg, 5)
    assert items, "方向つき視界の要素が空"
    assert all(d in ("front", "right", "left") for (d, _n, _dist) in items)
    # 決定論: 同入力2回で完全一致
    assert items == sd._directional(a, sim.city, h, sim.scenecfg, 5)


def test_vertical_line_layer(tmp_path):
    """レイヤ(高い通路/地下)から垂直関係の1行が出る。elevation=None でも安全(None ガード)。"""
    sim = _sim(tmp_path, "vert", steps=1, **_ON)
    a = sim.agents[0]
    graph = sim.city.graph
    deck = next((n for n, d in graph.nodes(data=True) if d.get("layer", 0) > 0), None)
    under = next((n for n, d in graph.nodes(data=True) if d.get("layer", 0) < 0), None)
    if deck is not None:
        a.node, a.building = deck, None
        line = sd._vertical_line(a, sim.city, sim.scenecfg, None)
        assert line and "見下ろ" in line, f"デッキで垂直行が出ない: {line}"
    if under is not None:
        a.node, a.building = under, None
        line = sd._vertical_line(a, sim.city, sim.scenecfg, None)
        assert line and "地下" in line, f"地下で垂直行が出ない: {line}"


def test_elevation_none_guard(tmp_path):
    """elevation=None のとき標高を一切参照しない(None ガード)。全個体で例外なく処理できる。"""
    sim = _sim(tmp_path, "elev", steps=1, **_ON)
    for a in sim.agents:
        # elevation=None: 標高経路に入らず(レイヤのみ)。例外なくシーン行を組めること。
        sd._vertical_line(a, sim.city, sim.scenecfg, None)
        sd.scene_lines(a, city=sim.city, company=[], cfg=sim.scenecfg,
                       elevation=None)


# ------------------------------------------------------ (g) 行間レイヤ配線
def test_s2_digest_visual_line(tmp_path):
    """interstitial + scene 両 ON で、ダイジェストに「見えたもの」(客観記述)が載る。"""
    sim = _sim(tmp_path, "s2", n=40,
               **{"prompts.interstitial.enabled": "true", **_ON})
    sim.llm = _RecordingLLM(sim.llm)
    sim.run()
    assert any("見えたもの" in p for p in sim.llm.prompts), \
        "S2 ダイジェストに視覚1行が載っていない"


def test_s2_no_visual_without_scene(tmp_path):
    """scene OFF なら interstitial ON でも「見えたもの」は出ない(配線の分離)。"""
    sim = _sim(tmp_path, "s2b", n=40, **{"prompts.interstitial.enabled": "true"})
    sim.llm = _RecordingLLM(sim.llm)
    sim.run()
    assert not any("見えたもの" in p for p in sim.llm.prompts)


def test_s3_conversation_scene_meta(tmp_path):
    """conversation + scene 両 ON で会話 payload に scene メタが載る。scene OFF は載らない。"""
    on = _sim(tmp_path, "s3", n=200,
              **{"conversation.enabled": "true", **_ON})
    on.run()
    convs = [e for e in on.logger.events if e.kind == "conversation"]
    assert convs, "会話が成立していない(密度不足)"
    assert all("scene" in e.payload for e in convs), "会話 payload に scene メタが無い"
    off = _sim(tmp_path, "s3b", n=200, **{"conversation.enabled": "true"})
    off.run()
    convs_off = [e for e in off.logger.events if e.kind == "conversation"]
    assert convs_off and all("scene" not in e.payload for e in convs_off), \
        "scene OFF なのに conversation payload に scene キーがある"
    # scene OFF の payload キーは従来どおり(会話3層 C2 の契約を壊さない)
    assert set(convs_off[0].payload) == {"with", "topic", "tone", "outcome", "acts"}
