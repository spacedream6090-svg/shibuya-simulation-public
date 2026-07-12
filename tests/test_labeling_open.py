"""ラベリング粒度スイッチ(#9 / D9)+ 自営の日銭(#11)のテスト。

(a) open: 自由語形(フレーズ・句読点可)がそのまま item 化(40 文字切詰のみ)。
(b) constrained: 句読点・改行を含む「文」は棄却=沈黙(item にしない)。
(c) 既定(constrained)は現行の受理挙動を維持(短く句読点のない呼び名は通る)。
(d) 自営(WAGE_CAT=自営)の日銭が日次で wage(source=gig)として出る+決定論。来街者には出ない。
"""
from __future__ import annotations

from society.config import load_config
from society.engine import scheduler
from society.engine.simulation import Simulation


def _sim(tmp_path, name, n=10, steps=1, **ov):
    dot = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
           "observer.snapshot_every=1"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    cfg = load_config(dot)
    return Simulation(cfg, out_dir=tmp_path / name)


# --------------------------------------------------- labeling: open / constrained
def test_open_mode_accepts_phrase(tmp_path):
    sim = _sim(tmp_path, "open", **{"labeling.mode": "open"})
    assert sim.labels.mode == "open"
    a = sim.agents[0]
    phrase = "渋谷の、ざわつく感じ、これ"           # 句読点入りフレーズ
    item = sim.labels.coin(a, phrase, step=0, sim_min=0, logger=sim.logger)
    assert item is not None, "open でフレーズが棄却された"
    assert item.text == phrase                       # そのまま item 化
    assert phrase in a.adopted


def test_open_mode_truncates_to_40(tmp_path):
    sim = _sim(tmp_path, "opentrunc", **{"labeling.mode": "open"})
    a = sim.agents[0]
    long = "あ" * 60
    item = sim.labels.coin(a, long, step=0, sim_min=0, logger=sim.logger)
    assert item is not None and len(item.text) == 40    # 40 文字へ切詰のみ


def test_constrained_rejects_sentence(tmp_path):
    """既定(constrained): 句読点・改行を含む文は棄却(None=沈黙)。"""
    sim = _sim(tmp_path, "constr")
    assert sim.labels.mode == "constrained"             # 既定
    a = sim.agents[0]
    assert sim.labels.coin(a, "これは文です。", step=0, sim_min=0,
                           logger=sim.logger) is None
    assert sim.labels.coin(a, "改行\n入り", step=0, sim_min=0,
                           logger=sim.logger) is None
    assert sim.labels.coin(a, "とても長すぎる呼び名だなこれは", step=0, sim_min=0,
                           logger=sim.logger) is None    # 12 文字超


def test_constrained_accepts_short_word_unchanged(tmp_path):
    """既存挙動維持: 短く句読点のない呼び名はそのまま受理(mock が作る語形)。"""
    sim = _sim(tmp_path, "cshort")
    a = sim.agents[0]
    item = sim.labels.coin(a, "モヤり現象", step=0, sim_min=0, logger=sim.logger)
    assert item is not None and item.text == "モヤり現象"
    coins = [e for e in sim.logger.events if e.kind == "vocab_coin"]
    assert coins, "constrained で正当な造語が記録されていない"


def test_coin_label_silence_on_reject(tmp_path):
    """棄却時、engine の coin_label 経路は沈黙(speak を出さない)。"""
    sim = _sim(tmp_path, "silence")
    a = sim.agents[0]
    before = len([e for e in sim.logger.events if e.kind == "speak"])
    scheduler._apply(sim, a, {"type": "coin_label", "word": "これは棄却される文。",
                              "text": "使った一言"}, step=0, sim_min=0)
    after = len([e for e in sim.logger.events if e.kind == "speak"])
    assert after == before, "棄却された造語が発話されている(沈黙になっていない)"


# --------------------------------------------------- economy: 自営の日銭(gig)
def _gig_events(sim, agent_id):
    return [e for e in sim.logger.events if e.kind == "wage"
            and e.agent_id == agent_id and e.payload.get("source") == "gig"]


def test_self_employed_gets_daily_gig(tmp_path):
    sim = _sim(tmp_path, "gig")
    a = sim.agents[0]
    a.occupation = "フリーランス"                    # WAGE_CAT=自営(固定職場なし)
    a.visitor = False
    a.money = 50000.0
    sim._econ_day = -1
    scheduler._phase_daily(sim, step=0, sim_min=0)
    gigs = _gig_events(sim, a.id)
    assert gigs, "自営の gig 支給(source=gig)が出ていない"
    base = sim.economy["wages"]["自営"]
    amt = gigs[-1].payload["amount"]
    assert base * 0.2 <= amt <= base * 1.4           # 出来高 U(0.2, 1.4)
    assert a.money > 50000.0                          # 残高に反映
    assert gigs[-1].payload["balance"] == round(a.money, 1)


def test_visitor_self_employed_gets_no_gig(tmp_path):
    sim = _sim(tmp_path, "gigvis")
    a = sim.agents[0]
    a.occupation = "フリーランス"
    a.visitor = True                                 # 来街者には支給しない
    sim._econ_day = -1
    scheduler._phase_daily(sim, step=0, sim_min=0)
    assert not _gig_events(sim, a.id)


def test_non_self_employed_gets_no_gig(tmp_path):
    sim = _sim(tmp_path, "gigemp")
    a = sim.agents[0]
    a.occupation = "会社員"                           # 自営でない = gig なし
    a.visitor = False
    sim._econ_day = -1
    scheduler._phase_daily(sim, step=0, sim_min=0)
    assert not _gig_events(sim, a.id)


def test_gig_is_deterministic(tmp_path):
    def once(name):
        sim = _sim(tmp_path, name)
        a = sim.agents[0]
        a.occupation, a.visitor, a.money = "配達員", False, 30000.0
        sim._econ_day = -1
        scheduler._phase_daily(sim, step=0, sim_min=0)
        return _gig_events(sim, a.id)[-1].payload["amount"]
    assert once("gigdet1") == once("gigdet2"), "同 seed で gig 支給額が非決定論"


def test_gig_appears_in_full_run(tmp_path):
    """既定の丸1日ランで、自然に選ばれた自営エージェントに gig 支給が出る。"""
    sim = _sim(tmp_path, "gigrun", n=15, steps=144, **{"run.seed": 7})
    sim.run()
    gigs = [e for e in sim.logger.events if e.kind == "wage"
            and e.payload.get("source") == "gig"]
    assert gigs, "丸1日ランで gig 支給が一件も出ていない"
