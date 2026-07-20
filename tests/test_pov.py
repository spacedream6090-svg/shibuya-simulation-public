"""エージェント視覚 v1(顕著性POV: src/society/pov.py + viz/render_pov.py)のテスト。

設計: docs/research/agent-vision.md §4 v1 / §5.2(R1)/ §2e(決定論の隔離)。全プロファイル OFF の
休眠骨格。方針(R1 の鉄則):
- OFF(既定): L1 が純粋既定と完全一致・pov_image 0 件・画像 0 枚・agent に _pov_seen を生やさない・
  専用 stream "pov_salience" を一度も引かない。
- ON(mock ≤12step): (a)pov_image が出てサイドカーに PNG が保存される。(b)同 seed 2回で L1 一致。
  (c)顕著性ゲートは客観量のみ=k(writeback)を振っても pov も LLM 呼数も不変(R1)。
  (d)POV レンダは同一入力→同一バイト PNG(§2e)。(e)VLM は mock 経路を通すが text LLM 呼数に影響しない。
検証は mock / 固定 LLM のみ(実LLM 禁止・≤12 step)。乱数は "pov_salience" 1本のみ・追加 LLM 呼ゼロ。
"""
from __future__ import annotations

import json

from society import pov
from society.config import load_config
from society.engine.simulation import Simulation
from society.observer.schema import EVENT_KINDS

_ON = {"pov.enabled": "true"}
_OFF = {"pov.enabled": "false"}
_ALWAYS = {"pov.enabled": "true", "pov.salience.prob_cap": "1.0"}


def _sim(tmp_path, name, n=40, steps=12, **ov):
    dot = ["run.seed=42", f"run.n_agents={n}", f"run.n_steps={steps}",
           f"run.name={name}"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _imgs(sim):
    return [e for e in sim.logger.events if e.kind == "pov_image"]


# ------------------------------------------------------ schema 登録
def test_pov_image_kind_registered():
    assert "pov_image" in EVENT_KINDS


# ------------------------------------------------------ OFF: 純粋既定と一致 + 無副作用
def test_off_matches_pure_default(tmp_path):
    pure = _sim(tmp_path, "pure")
    pure.run()
    off = _sim(tmp_path, "expl_off", **_OFF)
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(pov seam が no-op でない)"
    assert not any(e.kind == "pov_image" for e in off.logger.events)
    for a in pure.agents:
        assert not hasattr(a, "_pov_seen"), "OFF なのに _pov_seen が生えている"


def test_off_matches_pure_default_long(tmp_path):
    pure = _sim(tmp_path, "purel", n=15, steps=144)
    pure.run()
    off = _sim(tmp_path, "offl", n=15, steps=144, **_OFF)
    off.run()
    assert _l1(pure) == _l1(off)


# ------------------------------------------------------ レンダの決定論(同一入力→同一バイト)
def _renderer():
    return pov._load_renderer()


_BLDS = [
    {"id": "b1", "footprint": [[10, 10], [30, 10], [30, 30], [10, 30]]},
    {"id": "b2", "footprint": [[-40, 5], [-20, 5], [-20, 25], [-40, 25]]},
]


def test_render_same_input_same_bytes():
    rp = _renderer()
    a = rp.render_pov(cam_x=0.0, cam_y=0.0, heading=0.0, buildings=_BLDS)
    b = rp.render_pov(cam_x=0.0, cam_y=0.0, heading=0.0, buildings=_BLDS)
    assert a == b, "同一入力なのに PNG バイトが違う(決定論の破れ)"
    assert a[:8] == b"\x89PNG\r\n\x1a\n", "PNG シグネチャが不正"


def test_render_varies_with_view():
    rp = _renderer()
    base = rp.render_pov(cam_x=0.0, cam_y=0.0, heading=0.0, buildings=_BLDS)
    turned = rp.render_pov(cam_x=0.0, cam_y=0.0, heading=1.57, buildings=_BLDS)
    moved = rp.render_pov(cam_x=15.0, cam_y=0.0, heading=0.0, buildings=_BLDS)
    assert base != turned and base != moved, "視点を変えても画像が変わらない"


def test_render_empty_scene_is_valid_png():
    rp = _renderer()
    png = rp.render_pov(cam_x=1000.0, cam_y=1000.0, heading=0.0, buildings=_BLDS)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


# ------------------------------------------------------ ON: pov_image + サイドカー保存
def test_on_emits_and_saves(tmp_path):
    sim = _sim(tmp_path, "on", **_ALWAYS)
    sim.run()
    imgs = _imgs(sim)
    assert imgs, "ON なのに pov_image が1件も出ていない"
    p = imgs[0].payload
    assert set(p) >= {"ref", "w", "h", "trigger"}
    assert p["trigger"] in ("first_visit", "crowd", "world_event")
    pov_dir = (tmp_path / "on" / "pov")
    saved = sorted(pov_dir.glob("*.png"))
    assert len(saved) == len(imgs), "保存 PNG 数と pov_image 数が不一致"
    assert all(f.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" for f in saved)


def test_store_off_no_files_but_events(tmp_path):
    """store.enabled=false: pov_image は出るが PNG は書かない(参照キーのみ)。"""
    sim = _sim(tmp_path, "nostore", **{**_ALWAYS, "pov.store.enabled": "false"})
    sim.run()
    assert _imgs(sim), "store OFF でも pov_image イベントは出るべき"
    pov_dir = (tmp_path / "nostore" / "pov")
    assert not pov_dir.exists() or not list(pov_dir.glob("*.png"))


# ------------------------------------------------------ ON: 決定論
def test_on_deterministic(tmp_path):
    a = _sim(tmp_path, "det_a", **{**_ON, "pov.salience.prob_cap": "0.3"})
    a.run()
    b = _sim(tmp_path, "det_b", **{**_ON, "pov.salience.prob_cap": "0.3"})
    b.run()
    assert _l1(a) == _l1(b), "ON の決定論が崩れている(pov_salience 以外の乱数漏れ?)"


# ------------------------------------------------------ 顕著性ゲートは客観量のみ(k 非依存)
def test_gate_objective_only_k_invariant(tmp_path):
    """k(writeback)を free/off に振っても pov_image の系列も LLM 呼数も完全一致。

    ゲートは物理位置・訪問履歴・世界イベント・人数のみを入力にし、k 由来量・traits を一切
    見ない(R1・agent-vision.md §5.2)。ゆえに pov 系列・呼数は k 条件間で不変になる。"""
    free = _sim(tmp_path, "k_free", **{**_ALWAYS, "k.writeback": "free"})
    free.run()
    off = _sim(tmp_path, "k_off", **{**_ALWAYS, "k.writeback": "off"})
    off.run()
    # (a) 呼数 k 乖離テスト: v1 ON(mock)でも LLM 呼数が k で一致(policy_cache 関門と同作法)
    assert len(free.logger.llm_calls) == len(off.logger.llm_calls), \
        f"pov ON の呼数が k で乖離(R1 違反): {len(free.logger.llm_calls)} vs {len(off.logger.llm_calls)}"
    # (b) pov_image の参照キー系列も k 間で一致(ゲートが k を見ていない直接証明)
    ref_free = [(e.step, e.agent_id, e.payload["trigger"]) for e in _imgs(free)]
    ref_off = [(e.step, e.agent_id, e.payload["trigger"]) for e in _imgs(off)]
    assert ref_free == ref_off and ref_free, \
        "pov 系列が k で異なる(ゲートに k 由来量が漏れている疑い)"


# ------------------------------------------------------ VLM stub(mock 経路・text LLM 非依存)
class _MockVLM:
    def __init__(self):
        self.n = 0

    def describe(self, png: bytes) -> str:
        self.n += 1
        return "a street scene"


def test_mock_vlm_path_does_not_touch_text_llm(tmp_path):
    """vlm.backend=mock + sim.vlm で mock VLM 経路が通り payload に vlm フラグ。text LLM 呼数は
    VLM の有無で不変(VLM は別 tier=text LLM 呼数に影響しない)。"""
    with_vlm = _sim(tmp_path, "vlm_on",
                    **{**_ALWAYS, "pov.vlm.backend": "mock"})
    with_vlm.vlm = _MockVLM()
    with_vlm.run()
    no_vlm = _sim(tmp_path, "vlm_off", **_ALWAYS)
    no_vlm.run()
    imgs = _imgs(with_vlm)
    assert imgs, "画像が出ていない"
    assert with_vlm.vlm.n == len(imgs), "mock VLM が全画像で呼ばれていない"
    assert all(e.payload.get("vlm") for e in imgs), "payload に vlm フラグが無い"
    # text LLM 呼数は VLM の有無で完全一致(VLM は text LLM を1本も足さない)
    assert len(with_vlm.logger.llm_calls) == len(no_vlm.logger.llm_calls), \
        "VLM 経路が text LLM 呼数を動かしている(別 tier 分離が破れている)"
