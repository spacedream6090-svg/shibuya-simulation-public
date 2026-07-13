"""入力解像度LOD(第30バッチ・lod.input_res)のテスト。

設計(docs/plans/input-resolution-lod.md): 既定OFF=属性なし=純粋既定とL1バイト一致。
ON で変わるのは注入の「件数」だけ(呼数・乱数構造・発火は不変=R1)。mock はプロンプト内容に
反応しないため ON(mock) も L1 一致=配線が流れを変えないことの機械的証明になる。
割当は cognition/lod.py の共通軸機構(trait非依存・軸専用 stream・seed再現)。
"""
from __future__ import annotations

import json
from collections import Counter

from society.cognition import lod
from society.cognition.deliberate import build_prompt
from society.config import load_config
from society.engine.simulation import Simulation


def _cfg(name, n=20, steps=24, **ov):
    dot = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
           "observer.snapshot_every=144"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def _sim(tmp_path, name, n=20, steps=24, **ov):
    return Simulation(_cfg(name, n, steps, **ov), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


# --------------------------------------------------------------------- OFF 既定
def test_off_matches_pure_default(tmp_path):
    """明示 OFF と純粋既定が L1 完全一致。属性も付かない。"""
    pure = _sim(tmp_path, "pure")
    pure.run()
    off = _sim(tmp_path, "off", **{"lod.input_res.enabled": "false"})
    off.run()
    assert _l1(pure) == _l1(off), "input_res OFF が純粋既定と不一致(seam が no-op でない)"
    assert all(getattr(a, "input_res", None) is None for a in pure.agents)


def test_on_is_deterministic(tmp_path):
    """ON の決定論: 同seedで2回構築・実行して L1 完全一致(新規乱数は軸専用 stream のみ)。

    注: mock はプロンプト内容に反応する(行き先候補・do マーカーをプロンプトから拾う)ため、
    ON と OFF の L1 は一致しなくてよい=それは「内容差→行動差」という意図した経路。
    守るべき不変量は (a) OFF=純既定バイト一致(上のテスト)と (b) ON の seed 再現性。"""
    on1 = _sim(tmp_path, "on1", **{"lod.input_res.enabled": "true"})
    on1.run()
    on2 = _sim(tmp_path, "on2", **{"lod.input_res.enabled": "true"})
    on2.run()
    assert _l1(on1) == _l1(on2), "ON が同seedで再現しない(決定論の破れ)"
    assert on1.llm.calls == on2.llm.calls


# --------------------------------------------------------------------- 割当機構
def test_assignment_deterministic_and_recorded(tmp_path):
    """同seed同割当(決定論)・3水準が出る・agents.json に共変量として記録。"""
    a = _sim(tmp_path, "asg1", n=60, steps=1, **{"lod.input_res.enabled": "true"})
    b = _sim(tmp_path, "asg2", n=60, steps=1, **{"lod.input_res.enabled": "true"})
    la = {ag.id: ag.input_res["level"] for ag in a.agents}
    lb = {ag.id: ag.input_res["level"] for ag in b.agents}
    assert la == lb
    counts = Counter(la.values())
    assert set(counts) <= {"narrow", "mid", "wide"} and len(counts) == 3
    meta = json.loads((tmp_path / "asg1" / "agents.json").read_text(encoding="utf-8"))
    assert all("input_res" in m for m in meta)


def test_mid_level_equals_current_constants():
    """mid=現行定数(poi3・人全列挙・recent4・retrieve3・feed3・salience現行)の契約を固定。"""
    mid = lod.INPUT_RES_DEFAULTS["levels"]["mid"]
    assert (mid["poi_n"], mid["people_n"], mid["recent_n"],
            mid["retrieve_n"], mid["feed_n"], mid["salience_k"]) == (3, 0, 4, 3, 3, 0)


# --------------------------------------------------------------------- ノブの実効
def test_prompt_knobs_change_injected_counts(tmp_path):
    """narrow/wide でプロンプトの注入件数が実際に変わる(build_prompt 単体)。"""
    sim = _sim(tmp_path, "knob", n=5, steps=1)
    ag = sim.agents[0]
    pois = [f"店{i}" for i in range(6)]
    names = [f"人{i}" for i in range(6)]
    feed = [f"投稿{i}" for i in range(6)]

    def prompt():
        return build_prompt(ag, place_name="ハチ公前広場", surprise="social",
                            nearby_names=names, nearby_pois=pois,
                            feed_texts=feed, step=0, sim_min=0)

    base = prompt()                                        # OFF(属性なし)=現行既定
    assert "店2" in base and "店3" not in base             # poi[:3]
    assert "投稿2" in base and "投稿3" not in base         # feed[:3]
    assert "人5" in base                                    # 人は全列挙

    ag.input_res = dict(lod.INPUT_RES_DEFAULTS["levels"]["narrow"], level="narrow")
    narrow = prompt()
    assert "店0" in narrow and "店1" not in narrow          # poi_n=1
    assert "人1" in narrow and "人2" not in narrow          # people_n=2
    assert "投稿0" in narrow and "投稿1" not in narrow      # feed_n=1

    ag.input_res = dict(lod.INPUT_RES_DEFAULTS["levels"]["wide"], level="wide")
    wide = prompt()
    assert "店4" in wide and "人5" in wide and "投稿4" in wide
    assert len(narrow) < len(base) <= len(wide)
