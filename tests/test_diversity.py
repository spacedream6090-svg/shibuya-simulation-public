"""観光・多言語・犯罪・治安(現実ギャップ 後続波 H5 2026-07-07)のテスト。

方針(既存の鉄則を継承):
- OFF(既定): crime/nuisance/tourist_visit が 0 件・新 stream "tourist"/"crime" も引かない・
  イベント列は純粋既定と L1 完全一致(144 step。ゴールデン golden_baseline_l1.json を守る)。
  visitor/警察官 入り名簿の世界でも diversity OFF は完全 no-op。
- 観光 ON: 観光客(visitor の一部)がランドマークへ回遊し tourist_visit が出る(単体・決定論)。
- 多言語 ON: 異言語間の語採用が起きにくい(採用閾値が上がる。単体)。
- 犯罪 ON: crime/nuisance が出て被害者 money−・grievance+(単体・決定論)。近傍警察官で抑止。
- 治安 ON: 犯罪履歴ノードが行き先から避けられる(単体)。
- 決定論: ON 同士2回で L1 完全一致。
- R1: 犯罪(被害者 money−)・観光回遊は物理位置=対面 co-location を変えうる(career G5 と同型)。よって
  呼数不変は compute_matched 下の k 不変性(k=free と k=off の generate 呼数完全一致)で担保する。
検証は mock のみ(実LLM 禁止)。
"""
from __future__ import annotations

import json
from collections import Counter

from society import diversity
from society.config import load_config
from society.engine.simulation import Simulation
from society.labeling.labels import LabelSystem
from society.observer.provenance import ItemStore

_ON = {"society_diversity.enabled": "true"}


def _sim(tmp_path, name, n=30, steps=1, roster=None, **ov):
    dot = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
           "observer.snapshot_every=1"]
    if roster:
        dot += [f"agents.personas_file=data/{roster}"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _kind(sim, kind):
    return [e for e in sim.logger.events if e.kind == kind]


# --------------------------------------------------------------------- OFF 既定
def test_off_matches_pure_default(tmp_path):
    """明示 OFF と純粋既定が L1 完全一致(144 step)。H5 の新イベントは 1 件も出ない(seam が no-op)。"""
    pure = _sim(tmp_path, "pure", steps=144)
    pure.run()
    off = _sim(tmp_path, "expl_off", steps=144, **{"society_diversity.enabled": "false"})
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(H5 seam が no-op でない)"
    for k in ("crime", "nuisance", "tourist_visit"):
        assert not _kind(pure, k), f"OFF で {k} が出ている"


def test_off_noop_under_visitors_and_police(tmp_path):
    """visitor+警察官 入り名簿の世界でも diversity OFF は完全 no-op(名簿基準ランと L1 一致)。"""
    base = _sim(tmp_path, "rbase", steps=144, roster="personas_100_civic.json")
    base.run()
    off = _sim(tmp_path, "roff", steps=144, roster="personas_100_civic.json",
               **{"society_diversity.enabled": "false"})
    off.run()
    assert _l1(base) == _l1(off), "visitor+警察官 名簿で diversity seam が no-op でない"


# --------------------------------------------------------------------- 観光
def test_assign_marks_visitors_as_tourists(tmp_path):
    """assign_attributes: visitor の tourist_ratio 割を決定論で観光客にする(id 昇順)。"""
    sim = _sim(tmp_path, "tour_assign", n=6, **{**_ON, "society_diversity.tourist_ratio": "1.0"})
    for a in sim.agents[:3]:
        a.visitor = True
    diversity.assign_attributes(sim)
    tourists = [a for a in sim.agents if getattr(a, "tourist", False)]
    assert len(tourists) == 3 and all(a.visitor for a in tourists), \
        "visitor 全員(ratio=1.0)が観光客になっていない"
    assert all(not getattr(a, "tourist", False) for a in sim.agents if not a.visitor), \
        "非 visitor が観光客になっている"


def test_tourist_visits_landmark(tmp_path):
    """観光 ON: 観光客がランドマークへ回遊し tourist_visit が出る(決定論)。"""
    sim = _sim(tmp_path, "tour", n=6, **{**_ON, "society_diversity.tourist_bias": "1.0"})
    a = sim.agents[0]
    a.visitor = True
    a.tourist = True
    # 現在地をランドマーク上位から外す(候補が必ず残るよう dests の末尾に置く)
    a.node = sim.dests[-1]
    dest = diversity.tourist_dest(a, sim, step=5, sim_min=300)
    assert dest is not None and dest != a.node, "観光客がランドマークへ回遊していない"
    tv = _kind(sim, "tourist_visit")
    assert tv and tv[-1].payload["node"] == dest, "tourist_visit が記録されていない"
    # 決定論: 同 step・同 agent で再現
    dest2 = diversity.tourist_dest(a, sim, step=5, sim_min=300)
    assert dest2 == dest, "観光回遊が決定論でない(同 step で不一致)"


# --------------------------------------------------------------------- 多言語(伝播障壁)
def test_cross_barrier_value(tmp_path):
    """cross_barrier: 異言語ペアで追加閾値、同言語ペアで 0。"""
    sim = _sim(tmp_path, "lang", n=6, **{**_ON, "society_diversity.foreign_ratio": "0.5",
                                         "society_diversity.cross_lang_barrier": "2"})
    diversity.assign_attributes(sim)
    foreign = next(a for a in sim.agents if getattr(a, "language", ""))
    native = next(a for a in sim.agents if not getattr(a, "language", ""))
    assert diversity.cross_barrier(sim, foreign.id, native) == 2, "異言語間の障壁が立たない"
    assert diversity.cross_barrier(sim, native.id, native) == 0, "同言語間で障壁が立っている"


def test_language_barrier_delays_adoption(tmp_path):
    """伝播障壁: 異言語間(extra_threshold=2)は同言語(0)より採用に多くの聴取が要る(labeling 単体)。"""
    sim = _sim(tmp_path, "adopt", n=4, **_ON)
    ls = LabelSystem(ItemStore(), adopt_threshold=2)
    speaker, near, far = sim.agents[0], sim.agents[1], sim.agents[2]
    ls.coin(speaker, "ヤバみ", step=0, sim_min=0, logger=sim.logger)
    # 同言語: 2 回で採用
    for s in (1, 2):
        ls.on_hear(near, ["ヤバみ"], speaker.id, step=s, sim_min=s, channel="face",
                   logger=sim.logger, extra_threshold=0)
    assert "ヤバみ" in near.adopted, "同言語で 2 回聴取しても採用されない"
    # 異言語: 2 回では未採用、+2 回で採用(閾値 2+2=4)
    for s in (1, 2):
        ls.on_hear(far, ["ヤバみ"], speaker.id, step=s, sim_min=s, channel="face",
                   logger=sim.logger, extra_threshold=2)
    assert "ヤバみ" not in far.adopted, "異言語なのに 2 回で採用された(障壁が効いていない)"
    for s in (3, 4):
        ls.on_hear(far, ["ヤバみ"], speaker.id, step=s, sim_min=s, channel="face",
                   logger=sim.logger, extra_threshold=2)
    assert "ヤバみ" in far.adopted, "異言語で 4 回聴取しても採用されない"


# --------------------------------------------------------------------- 犯罪・迷惑
def _colocate(sim, ids, node):
    for i in ids:
        a = sim.agent_by_id[i]
        a.node = node
        a.loc = "street"
        a.sleeping = False


def test_theft_takes_money_and_raises_grievance(tmp_path):
    """犯罪 ON: 同ノードの被害者から窃盗=money−(現金の範囲)+ grievance+(factors 経由・決定論)。"""
    sim = _sim(tmp_path, "theft", n=4, **{**_ON, "society_diversity.crime_prob": "1.0",
                                          "society_diversity.nuisance_prob": "0.0",
                                          "society_diversity.theft_amount": "3000",
                                          "society_diversity.crime_grievance": "0.05"})
    _colocate(sim, [0, 1], sim.dests[0])       # 0,1 を同ノードに(相互に被害者)
    _colocate(sim, [2], sim.dests[1])
    _colocate(sim, [3], sim.dests[2])
    for i in (0, 1):
        sim.agent_by_id[i].money = 5000.0
        sim.agent_by_id[i].states["grievance"] = 0.1
    diversity.tick_crime(sim, step=7, sim_min=420)
    crimes = [e for e in _kind(sim, "crime") if e.payload["kind"] == "theft"]
    victims = {e.payload["victim"] for e in crimes}
    assert {0, 1} <= victims, f"同ノードの相互窃盗が起きていない: {victims}"
    for i in (0, 1):
        assert sim.agent_by_id[i].money == 2000.0, "被害額(theft_amount)が引かれていない"
    su = [e for e in _kind(sim, "state_update") if e.payload["cause"] == "crime"]
    assert su and all(e.payload["new"] > e.payload["old"] for e in su), \
        "窃盗→grievance(crime)が factors 経由で入っていない"


def test_police_deters_crime(tmp_path):
    """近傍(同ノード)に警察官が居れば窃盗が抑止される(G3 執行との接続)。"""
    sim = _sim(tmp_path, "deter", n=4, **{**_ON, "society_diversity.crime_prob": "1.0"})
    _colocate(sim, [0, 1, 2], sim.dests[0])
    _colocate(sim, [3], sim.dests[1])
    sim.agent_by_id[2].occupation = "警察官"     # 同ノードに警察官
    diversity.tick_crime(sim, step=3, sim_min=180)
    assert not _kind(sim, "crime"), "警察官が同ノードに居るのに犯罪が発生した(抑止されていない)"


def test_nuisance_raises_bystander_grievance(tmp_path):
    """迷惑行為 ON: 同ノードの周囲(bystander)に grievance+(factors 経由)+ nuisance 記録。"""
    sim = _sim(tmp_path, "nuis", n=4, **{**_ON, "society_diversity.crime_prob": "0.0",
                                         "society_diversity.nuisance_prob": "1.0",
                                         "society_diversity.nuisance_grievance": "0.03"})
    _colocate(sim, [0, 1], sim.dests[0])
    _colocate(sim, [2], sim.dests[1])
    _colocate(sim, [3], sim.dests[2])
    for i in (0, 1):
        sim.agent_by_id[i].states["grievance"] = 0.1
    diversity.tick_crime(sim, step=9, sim_min=540)
    nus = _kind(sim, "nuisance")
    assert nus and nus[0].payload["kind"], "迷惑行為(nuisance)が記録されていない"
    su = [e for e in _kind(sim, "state_update") if e.payload["cause"] == "nuisance"]
    assert su and all(e.payload["new"] > e.payload["old"] for e in su), \
        "迷惑行為→周囲の grievance(nuisance)が入っていない"


# --------------------------------------------------------------------- 治安回避
def test_danger_zone_avoided(tmp_path):
    """治安 ON: 犯罪履歴が閾値超のノード=危険地帯を自由時間の行き先から避ける。"""
    sim = _sim(tmp_path, "danger", n=4, **{**_ON, "society_diversity.danger_threshold": "2"})
    a = sim.agent_by_id[0]
    a.node = sim.dests[0]
    danger = sim.dests[1]
    sim._diversity_crime_nodes = Counter({danger: 5})
    assert diversity.is_danger(sim, danger), "犯罪履歴ノードが危険地帯と判定されない"
    safe = diversity.safe_dests(sim, a, sim.dests)
    assert danger not in safe, "危険地帯が行き先候補から除かれていない"
    assert a.node in safe or a.node == danger, "現在地が候補から消えている"


# --------------------------------------------------------------------- 決定論
def test_all_on_deterministic(tmp_path):
    """H5 ON(観光+多言語+犯罪+治安)同士 2 回で L1 完全一致(決定論・mock 144 step・civic 名簿)。"""
    ov = {**_ON, "society_diversity.tourist_ratio": "0.5",
          "society_diversity.foreign_ratio": "0.3",
          "society_diversity.crime_prob": "0.05",
          "society_diversity.nuisance_prob": "0.05",
          "society_diversity.crime_grievance": "0.08"}
    a = _sim(tmp_path, "det_a", n=30, steps=144, roster="personas_100_civic.json", **ov)
    a.run()
    b = _sim(tmp_path, "det_b", n=30, steps=144, roster="personas_100_civic.json", **ov)
    b.run()
    assert _l1(a) == _l1(b), "ON の決定論が崩れている"


# --------------------------------------------------------------------- R1 k 不変性
class _FixedLLM:
    """挙動を固定する backend(応答をプロンプトに依存させない)。呼数だけ数える。"""

    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.hits = 0

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        self.calls += 1
        return self.response, str(self.calls), False


def _run_k(tmp_path, name, *, writeback):
    """H5 ON を compute_matched(k 掃引で使う対照)下で回し generate 呼数を数える。"""
    sim = _sim(tmp_path, name, n=30, steps=144, roster="personas_100_civic.json",
               **{**_ON, "society_diversity.tourist_ratio": "0.5",
                  "society_diversity.foreign_ratio": "0.3",
                  "society_diversity.crime_prob": "0.05",
                  "society_diversity.crime_grievance": "0.08",
                  "controls.mode": "compute_matched", "k.writeback": writeback})
    sim.llm = _FixedLLM(json.dumps({"action": "speak", "text": "x"}, ensure_ascii=False))
    sim.run()
    return sim


def test_diversity_call_count_k_invariant(tmp_path):
    """犯罪(被害者 money−)・観光回遊は対面 co-location を変え ON!=OFF になりうる(実在する多様性の必然)。
    だが R1 の本旨=「呼数が k(writeback)に依存しない」ことは compute_matched 下で厳密に保たれる。H5 機構は
    名簿・config・物理位置・新 stream・犯罪履歴のみ参照し k・内面状態を一切読まないため、k=free と k=off で
    generate 呼数が完全一致する(career G5 / 群集 G4 と同型)。"""
    free = _run_k(tmp_path, "dk_free", writeback="free")
    off = _run_k(tmp_path, "dk_off", writeback="off")
    assert free.llm.calls == off.llm.calls and free.llm.calls > 0, \
        f"diversity の呼数が k に依存(R1 違反): free={free.llm.calls} off={off.llm.calls}"
