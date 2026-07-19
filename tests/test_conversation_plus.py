"""会話品質の構造改善2点(会話強化 S3)のテスト。

対象(docs/research/conversation-pipeline.md の乖離1・2):
  1. 対話履歴の注入(prompts.dialog_history、既定 OFF): 相手別リングバッファ(相手あたり
     最大4発話=2往復・全体最大8相手 LRU)を持ち、social/reply の build_prompt へ直近2往復を注入。
  2. 関係加重の返答宛先(prompts.reply_partner、既定 "nearest"): "closeness" で
     score = closeness·10.0 − dist_m·0.1 の argmax(同点 id 昇順)に宛先を変更。

鉄則の検証:
  - OFF はバイト同一(状態も増やさない=_dialog_hist を一度も作らない)。
  - 乱数を一切引かない(全決定論)。hearer 集合は不変(聞く人は変わらない=返す人だけ変わる)。
既定=現行 nearest が min() と同一選択であることも固定する。mock ベース。
"""
from __future__ import annotations

import json
import math

from society.cognition import deliberate
from society.config import load_config
from society.engine import scheduler
from society.engine.simulation import Simulation


# --------------------------------------------------------------------------- utils
def _sim(tmp_path, name: str, n: int = 6, steps: int = 1, **ov) -> Simulation:
    dotlist = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
               "observer.snapshot_every=1"]
    dotlist += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dotlist), out_dir=tmp_path / name)


def _colocate(*agents) -> None:
    node = agents[0].node
    for ag in agents:
        ag.loc = "street"
        ag.building = None
        ag.floor = 0
        ag.sleeping = False
        ag.node = node
        ag.x, ag.y = 50.0, 50.0
        ag.conv_turns_left = 3
        ag.conv_cooldown_until = 0


def _isolate(sim, keep) -> None:
    """keep 以外を就寝させて hearers から外す(会話を2者に閉じる=決定論)。"""
    for other in sim.agents:
        if other not in keep:
            other.sleeping = True


def _speak(text: str) -> dict:
    return {"type": "speak", "text": text, "use_items": []}


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


# --------------------------------------------------------------------------- OFF
def test_off_scripted_conversation_l1_identical(tmp_path):
    """既定(OFF)の会話 L1 が完全再現し、対話バッファを一切作らない(状態変化なし)。"""
    def run(name):
        sim = _sim(tmp_path, name)
        a, b = sim.agents[0], sim.agents[1]
        _isolate(sim, (a, b))
        _colocate(a, b)
        scheduler._apply(sim, a, _speak("こんにちは、いい天気ですね"), 0, 0)
        scheduler._apply(sim, b, _speak("やあ、今日は暇なの?"), 1, 10)
        scheduler._apply(sim, a, _speak("ぼちぼちだよ、そっちは?"), 2, 20)
        return sim, a, b

    s1, a1, b1 = run("off1")
    s2, a2, b2 = run("off2")
    assert _l1(s1) == _l1(s2), "OFF の会話 L1 が非決定的(状態漏れの疑い)"
    for ag in (a1, b1, a2, b2):
        assert not hasattr(ag, "_dialog_hist"), "OFF で対話バッファが生成された(状態変化)"


def test_off_reply_path_creates_no_state(tmp_path):
    """OFF: 返答保証(nearest で最寄りへ)は現行どおり働き、対話バッファは作られない。"""
    sim = _sim(tmp_path, "offreply")
    a, b = sim.agents[0], sim.agents[1]
    _isolate(sim, (a, b))
    _colocate(a, b)
    scheduler._apply(sim, a, _speak("こんにちは"), 0, 0)
    assert b._reply_to is not None and b._reply_to[0] == a.id, "最寄りへ返答権が渡らない"
    assert not hasattr(a, "_dialog_hist") and not hasattr(b, "_dialog_hist")
    scheduler._decide(sim, b, step=1, sim_min=10)      # 返答(reply)LLM 発火
    reply_ev = [e for e in sim.logger.events
                if e.kind == "llm_deliberate" and e.agent_id == b.id
                and e.payload.get("trigger") == "reply"]
    assert reply_ev, "OFF で返答 llm_deliberate(reply)が出ていない"
    assert not hasattr(a, "_dialog_hist") and not hasattr(b, "_dialog_hist"), \
        "OFF の reply 経路で対話バッファが作られた(状態変化)"


def test_off_build_prompt_pure_additive(tmp_path):
    """dialog_history=None は省略時の build_prompt と一字一句一致(純追記の後方互換)。"""
    sim = _sim(tmp_path, "offbp")
    a = sim.agents[0]
    kw = dict(place_name="路上", surprise="reply", nearby_names=["花子"],
              reply_to=("花子", "やあ"))
    assert deliberate.build_prompt(a, **kw) \
        == deliberate.build_prompt(a, dialog_history=None, **kw)


# --------------------------------------------------------------------------- 宛先選択
def test_default_nearest_matches_min(tmp_path):
    """既定 "nearest" は現行の min(距離, id) と完全同一の宛先を選ぶ。"""
    sim = _sim(tmp_path, "near")
    a = sim.agents[0]
    a.x, a.y = 0.0, 0.0
    near, far = sim.agents[1], sim.agents[2]
    near.x, near.y = 5.0, 0.0
    far.x, far.y = 30.0, 0.0
    hearers = [far, near]                               # 順不同でも結果は同じ
    picked = scheduler._select_partner(sim, a, hearers)
    expected = min(hearers, key=lambda h: (math.hypot(h.x - a.x, h.y - a.y), h.id))
    assert picked is expected is near, "既定 nearest が現行 min() と一致しない"


def test_closeness_prefers_intimate_over_near(tmp_path):
    """closeness 宛先: 近いが疎遠な人より、遠いが親密な人を選ぶ(数値例)。

    score = closeness·10.0 − dist_m·0.1:
      near_aloof   (dist 5,  closeness 0.0) = 0·10 − 5·0.1  = −0.5
      far_intimate (dist 30, closeness 1.0) = 1·10 − 30·0.1 = +7.0  → argmax
    現行 nearest なら距離のみで near_aloof を選ぶ(=宛先が実際に変わる)。"""
    sim = _sim(tmp_path, "close", **{"prompts.reply_partner": "closeness"})
    a = sim.agents[0]
    a.x, a.y = 0.0, 0.0
    near_aloof, far_intimate = sim.agents[1], sim.agents[2]
    near_aloof.x, near_aloof.y = 5.0, 0.0
    far_intimate.x, far_intimate.y = 30.0, 0.0
    a.mem.relations[far_intimate.id] = {"name": far_intimate.name, "count": 5,
                                        "last_step": 0, "last": "", "closeness": 1.0}
    hearers = [near_aloof, far_intimate]
    # 現行(距離のみ)は近い方
    assert min(hearers, key=lambda h: (math.hypot(h.x - a.x, h.y - a.y), h.id)) \
        is near_aloof
    # closeness は親密な遠い方(宛先が変わる。hearer 集合は不変)
    assert scheduler._select_partner(sim, a, hearers) is far_intimate


def test_closeness_tie_breaks_by_id(tmp_path):
    """closeness: スコア同点は id 昇順(argmax 決定論)。"""
    sim = _sim(tmp_path, "tie", **{"prompts.reply_partner": "closeness"})
    a = sim.agents[0]
    a.x, a.y = 0.0, 0.0
    h1, h2 = sim.agents[1], sim.agents[2]
    h1.x, h1.y = 10.0, 0.0
    h2.x, h2.y = 10.0, 0.0                              # 距離・closeness とも同じ=同点
    low, high = (h1, h2) if h1.id < h2.id else (h2, h1)
    assert scheduler._select_partner(sim, a, [high, low]) is low, "同点で id 昇順でない"


# --------------------------------------------------------------------------- 対話履歴
def test_dialog_history_accumulates_and_injects(tmp_path):
    """ON: 往復が相手別バッファに蓄積し、build_prompt に相手発話2往復が載る。"""
    sim = _sim(tmp_path, "dlg", **{"prompts.dialog_history": "true"})
    a, b = sim.agents[0], sim.agents[1]
    _isolate(sim, (a, b))
    _colocate(a, b)
    scheduler._apply(sim, a, _speak("Aやあ"), 0, 0)
    scheduler._apply(sim, b, _speak("Bどうも"), 1, 10)
    scheduler._apply(sim, a, _speak("A元気?"), 2, 20)
    scheduler._apply(sim, b, _speak("Bまあね"), 3, 30)

    hist = scheduler._dialog_get(a, b.id)               # a から見た b との対話
    assert hist is not None and len(hist) == 4, "2往復=4発話が蓄積していない"
    assert hist[0] == (a.name, "Aやあ")
    assert hist[-1] == (b.name, "Bまあね")
    # 相手(b)の buffer にも同じ会話が入る(相互に相手別で積む)
    assert scheduler._dialog_get(b, a.id) == hist

    prompt = deliberate.build_prompt(a, place_name="路上", surprise="reply",
                                     nearby_names=[b.name], nearby_ids=[b.id],
                                     reply_to=(b.name, "Bまあね"),
                                     dialog_history=hist)
    assert "直前のやりとり" in prompt, "対話履歴行が注入されていない"
    assert "Bどうも" in prompt and "Bまあね" in prompt, "相手の2発話が載っていない"
    assert "A元気?" in prompt, "自分側の直近発話も文脈に載るべき"


def test_reply_prompt_injects_dialog_when_on(tmp_path):
    """ON の end-to-end: _decide の返答経路が対話履歴を実プロンプトへ注入する。"""
    sim = _sim(tmp_path, "e2e", **{"prompts.dialog_history": "true"})
    a, b = sim.agents[0], sim.agents[1]
    _isolate(sim, (a, b))
    _colocate(a, b)
    scheduler._apply(sim, a, _speak("最近どう?"), 0, 0)
    scheduler._apply(sim, b, _speak("ぼちぼちかな"), 1, 10)
    scheduler._apply(sim, a, _speak("いい感じだね"), 2, 20)

    captured = {}
    orig = sim.llm.generate

    def _spy(prompt, **kw):
        captured["prompt"] = prompt
        return orig(prompt, **kw)

    sim.llm.generate = _spy
    b._reply_to = (a.id, "いい感じだね")
    scheduler._decide(sim, b, step=3, sim_min=30)
    assert "prompt" in captured, "返答 LLM が発火していない(予算切れ?)"
    p = captured["prompt"]
    assert "直前のやりとり" in p
    assert "最近どう?" in p and "ぼちぼちかな" in p, "相手との直近往復が注入されていない"


def test_dialog_history_off_never_builds_state(tmp_path):
    """OFF(既定)では _apply が対話バッファを一切作らない(構造的バイト一致)。"""
    sim = _sim(tmp_path, "dlgoff")
    a, b = sim.agents[0], sim.agents[1]
    _isolate(sim, (a, b))
    _colocate(a, b)
    scheduler._apply(sim, a, _speak("Aやあ"), 0, 0)
    scheduler._apply(sim, b, _speak("Bどうも"), 1, 10)
    assert not hasattr(a, "_dialog_hist") and not hasattr(b, "_dialog_hist")
    assert scheduler._dialog_get(a, b.id) is None       # 取得も安全に None


# --------------------------------------------------------------------------- LRU
def test_dialog_per_partner_cap(tmp_path):
    """相手あたり最大4発話(=2往復)。古い発話から落ちる。"""
    sim = _sim(tmp_path, "cap")
    a = sim.agents[0]
    for i in range(6):
        scheduler._dialog_push(a, 99, "太郎", f"発話{i}")
    turns = scheduler._dialog_get(a, 99)
    assert len(turns) == 4
    assert turns[0] == ("太郎", "発話2"), "相手あたり上限で古い発話が落ちていない"
    assert turns[-1] == ("太郎", "発話5")


def test_dialog_partner_lru(tmp_path):
    """全体最大8相手 LRU: 9人目で最古の相手が退避される。"""
    sim = _sim(tmp_path, "lru")
    a = sim.agents[0]
    for pid in range(1, 10):                            # 相手 1..9 を順に
        scheduler._dialog_push(a, pid, "X", "hi")
    hist = a._dialog_hist
    assert len(hist) == 8, "相手数の上限(8)が効いていない"
    assert 1 not in hist, "最古の相手が退避されていない"
    assert 9 in hist, "最新の相手が保持されていない"


def test_dialog_lru_reaccess_moves_to_end(tmp_path):
    """LRU: 既存相手への追記は最新利用へ更新され、退避対象から外れる。"""
    sim = _sim(tmp_path, "lru2")
    a = sim.agents[0]
    for pid in range(1, 9):                             # 相手 1..8 で満杯
        scheduler._dialog_push(a, pid, "X", "hi")
    scheduler._dialog_push(a, 1, "X", "again")          # 相手1 を再利用=最新へ
    scheduler._dialog_push(a, 9, "X", "hi")             # 9人目 → 退避は今や最古の相手2
    hist = a._dialog_hist
    assert 1 in hist and 9 in hist, "再利用した相手/最新の相手が落ちた"
    assert 2 not in hist, "再利用で最古が繰り上がった相手2が退避されていない"
