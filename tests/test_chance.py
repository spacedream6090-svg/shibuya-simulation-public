"""生活の偶発イベント層(第54バッチ 2026-07-23。純観察=不確実性許容モード)のテスト。

方針(既存の鉄則 R1 を継承):
- OFF(既定): chance_event が 0 件・新 stream "chance" も引かない・イベント列は純粋既定と L1 完全一致
  (144 step。ゴールデン golden_baseline_l1.json を守る)。
- ON 決定論: 同 seed 2 ラン → L1 完全一致(決定論エンジンは不変=不確実性を足しても再現性は seed で担保)。
- windfall/loss: money の収支が chance_event の記録(amount/balance)と一致(外生の授受の正直な会計)。
- encounter: 近傍 or 既知相手と closeness+(双方)+ remember(双方)。候補不在は不発。
- (agent, day) 個体別キー: 他機構の順序(編成順・n_agents)に非依存=同一 id は同一 seed で同一の money 抽選。
- R1 k 不変性: compute_matched 下で k=free と k=off の generate 呼数が完全一致(chance は k を読まない)。
- LUCK_KINDS 接続: audit_uncertainty が chance_event を運側変数として拾う(1 行接続の担保)。
- seed 自動採取(scripts/run.py): 2 回で異なる seed が採れ、採れた seed で固定再実行するとバイト再現。
検証は mock のみ(実LLM 禁止)。
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from society import chance
from society.config import load_config
from society.engine.simulation import Simulation

_ROOT = Path(__file__).resolve().parents[1]

# ON カタログの共通設定(daily_rate/weight を dotlist で分岐させる)。
_ON = {"chance.enabled": "true"}


def _sim(tmp_path, name, n=30, steps=1, **ov):
    dot = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
           "observer.snapshot_every=1"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _chance_events(sim, etype=None):
    out = [e for e in sim.logger.events if e.kind == "chance_event"]
    return [e for e in out if etype is None or e.payload.get("type") == etype]


# --------------------------------------------------------------------- OFF 既定
def test_off_matches_pure_default(tmp_path):
    """明示 OFF と純粋既定が L1 完全一致(144 step)。chance_event は 1 件も出ない(seam が no-op)。"""
    pure = _sim(tmp_path, "pure", steps=144)
    pure.run()
    off = _sim(tmp_path, "expl_off", steps=144, **{"chance.enabled": "false"})
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(chance seam が no-op でない)"
    assert not _chance_events(pure), "OFF で chance_event が出ている"


# --------------------------------------------------------------------- ON 決定論
def test_on_deterministic(tmp_path):
    """chance ON 同士 2 回で L1 完全一致(決定論・mock 144 step)。不確実性を足しても再現性は seed で担保。"""
    a = _sim(tmp_path, "det_a", n=30, steps=144, **{**_ON, "chance.daily_rate": "0.2"})
    a.run()
    b = _sim(tmp_path, "det_b", n=30, steps=144, **{**_ON, "chance.daily_rate": "0.2"})
    b.run()
    assert _l1(a) == _l1(b), "chance ON の決定論が崩れている"
    assert _chance_events(a), "daily_rate=0.2 で chance_event が 1 件も出ていない"


# --------------------------------------------------------------------- windfall/loss 収支
def test_windfall_money_bookkeeping(tmp_path):
    """windfall のみ ON(economy OFF で他の money 変動を排除)→ 最終 money = 初期 + Σwindfall.amount。"""
    sim = _sim(tmp_path, "wf", n=20, steps=1,
               **{**_ON, "chance.daily_rate": "1.0", "economy.enabled": "false",
                  "chance.events.windfall.weight": "1", "chance.events.loss.weight": "0",
                  "chance.events.encounter.weight": "0"})
    initial = {a.id: a.money for a in sim.agents}
    sim.run()
    gained: dict[int, float] = {}
    for e in _chance_events(sim, "windfall"):
        gained[e.agent_id] = gained.get(e.agent_id, 0.0) + e.payload["amount"]
        assert e.payload["amount"] > 0, "windfall の amount が非正"
    assert gained, "windfall が 1 件も出ていない"
    for a in sim.agents:
        exp = initial[a.id] + gained.get(a.id, 0.0)
        assert abs(a.money - exp) < 1e-6, \
            f"windfall 収支不一致 id={a.id}: money={a.money} 期待={exp}"


def test_loss_money_bookkeeping(tmp_path):
    """loss のみ ON(economy OFF)→ 記録 amount = 実際の減少分(床でも収支一致)。money は 0 未満にならない。"""
    sim = _sim(tmp_path, "ls", n=20, steps=1,
               **{**_ON, "chance.daily_rate": "1.0", "economy.enabled": "false",
                  "chance.events.windfall.weight": "0", "chance.events.loss.weight": "1",
                  "chance.events.encounter.weight": "0"})
    for a in sim.agents:                         # economy OFF は初期 money=0 なので手持ちを与える
        a.money = 1_000_000.0                    # loss(<=20000)が床に当たらない=収支検証が clean
    initial = {a.id: a.money for a in sim.agents}
    sim.run()
    lost: dict[int, float] = {}
    for e in _chance_events(sim, "loss"):
        lost[e.agent_id] = lost.get(e.agent_id, 0.0) + e.payload["amount"]
    assert lost, "loss が 1 件も出ていない"
    for a in sim.agents:
        assert a.money >= 0.0, f"loss で money が負 id={a.id}"
        exp = initial[a.id] - lost.get(a.id, 0.0)
        assert abs(a.money - exp) < 1e-6, \
            f"loss 収支不一致 id={a.id}: money={a.money} 期待={exp}"


# --------------------------------------------------------------------- encounter
def test_encounter_closeness_and_remember(tmp_path):
    """encounter のみ ON: 決定論選定した相手と closeness+(双方)+ 「偶然○○に会った」remember(双方)。"""
    sim = _sim(tmp_path, "enc", n=4, steps=1,
               **{**_ON, "chance.daily_rate": "1.0",
                  "chance.events.windfall.weight": "0", "chance.events.loss.weight": "0",
                  "chance.events.encounter.weight": "1", "chance.events.encounter.closeness": "1.5"})
    a, b = sim.agents[0], sim.agents[1]
    a.mem.record_contact(b.id, b.name, 0, "顔なじみ")   # a の既知相手に b を据える(再会候補)
    b.mem.record_contact(a.id, a.name, 0, "顔なじみ")
    chance.tick_day(sim, 0, 0)
    a_ev = [e for e in _chance_events(sim, "encounter") if e.agent_id == a.id]
    assert a_ev, "候補(既知相手 b)が居るのに a の encounter が不発"
    other_id = a_ev[0].payload["other"]
    other = sim.agent_by_id[other_id]
    assert a.mem.relations[other_id].get("closeness", 0.0) > 0.0, "a→相手の closeness+ が無い"
    assert other.mem.relations[a.id].get("closeness", 0.0) > 0.0, "相手→a の closeness+ が無い(双方向でない)"
    assert any("偶然" in ep.text for ep in a.mem.buffer), "a に『偶然○○に会った』の記憶が無い"
    assert any("偶然" in ep.text for ep in other.mem.buffer), "相手に『偶然○○に会った』の記憶が無い"


def test_encounter_unfired_when_no_candidate(tmp_path):
    """相手不在(既知相手なし・同ノードに他者なし)は不発=chance_event を出さない。"""
    sim = _sim(tmp_path, "encnone", n=3, steps=1,
               **{**_ON, "chance.daily_rate": "1.0",
                  "chance.events.windfall.weight": "0", "chance.events.loss.weight": "0",
                  "chance.events.encounter.weight": "1"})
    a = sim.agents[0]
    a.mem.relations.clear()                      # 既知相手を消す
    for other in sim.agents[1:]:
        other.loc = "outside"                    # 同ノードの近傍から外す(街の外へ)
    chance.tick_day(sim, 0, 0)
    assert not [e for e in _chance_events(sim, "encounter") if e.agent_id == a.id], \
        "候補不在なのに encounter が発火した"


# --------------------------------------------------------------------- (agent, day) キー=編成順非依存
def test_agent_day_key_order_independent(tmp_path):
    """専用 stream "chance"((agent.id, day))は n_agents(編成順)に非依存=同一 id は同一 money 抽選。"""
    ov = {**_ON, "chance.daily_rate": "1.0", "economy.enabled": "false",
          "chance.events.windfall.weight": "1", "chance.events.loss.weight": "0",
          "chance.events.encounter.weight": "0", "run.seed": "42"}
    small = _sim(tmp_path, "small", n=6, steps=1, **ov)
    large = _sim(tmp_path, "large", n=14, steps=1, **ov)
    small.run()
    large.run()
    amt_s = {e.agent_id: e.payload["amount"] for e in _chance_events(small, "windfall")}
    amt_l = {e.agent_id: e.payload["amount"] for e in _chance_events(large, "windfall")}
    common = set(amt_s) & set(amt_l)
    assert len(common) >= 6, "共通の agent id が少なすぎる(検証にならない)"
    for aid in common:
        assert amt_s[aid] == amt_l[aid], \
            f"id={aid} の windfall 抽選が n_agents に依存(編成順非依存でない): {amt_s[aid]} != {amt_l[aid]}"


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
    sim = _sim(tmp_path, name, n=30, steps=144,
               **{**_ON, "chance.daily_rate": "0.3",
                  "controls.mode": "compute_matched", "k.writeback": writeback})
    sim.llm = _FixedLLM(json.dumps({"action": "speak", "text": "x"}, ensure_ascii=False))
    sim.run()
    return sim


def test_chance_call_count_k_invariant(tmp_path):
    """chance の効果(money/closeness/記憶)は物理位置=co-location を変えうるが、機構は k を読まない=
    compute_matched 下で k=free と k=off の generate 呼数が完全一致(career G5 / 健康 H1 と同型)。"""
    free = _run_k(tmp_path, "ck_free", writeback="free")
    off = _run_k(tmp_path, "ck_off", writeback="off")
    assert free.llm.calls == off.llm.calls and free.llm.calls > 0, \
        f"chance の呼数が k に依存(R1 違反): free={free.llm.calls} off={off.llm.calls}"
    assert _chance_events(free), "daily_rate=0.3 で chance_event が不発(機構が動いていない)"


# --------------------------------------------------------------------- LUCK_KINDS 接続(運/実力分解)
def test_luck_kinds_connection():
    """audit_uncertainty.LUCK_KINDS が chance_event を運側変数として拾う(第51の 1 行接続の担保)。"""
    sys.path.insert(0, str(_ROOT / "scripts"))
    sys.path.insert(0, str(_ROOT / "src"))
    import audit_uncertainty as au
    from society.observer.schema import EVENT_KINDS
    assert "chance_event" in au.LUCK_KINDS, "LUCK_KINDS に chance_event が接続されていない"
    assert "chance_event" in EVENT_KINDS, "schema に chance_event が登録されていない"
    events = [
        {"step": 0, "sim_min": 0, "agent": 1, "kind": "chance_event",
         "payload": {"type": "windfall", "amount": 5000, "balance": 55000}},
        {"step": 3, "sim_min": 30, "agent": 2, "kind": "chance_event",
         "payload": {"type": "encounter", "other": 1}},
    ]
    attr = au.attribute_chances(events, {})
    assert attr["per_agent"][1]["chance"] == 1, "windfall が運側に帰属していない"
    assert attr["per_agent"][2]["chance"] == 1, "encounter が運側に帰属していない"
    assert attr["by_kind"]["chance_event"] == 2


# --------------------------------------------------------------------- seed 自動採取(scripts/run.py)
def _load_run_module():
    spec = importlib.util.spec_from_file_location("run_entry", _ROOT / "scripts" / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_seed_auto_samples_and_resolves():
    """_sample_seed は呼ぶたび(ほぼ)異なる値・_resolve_seed_auto は 'auto' センチネルを除去し bool 版は残す。"""
    run_mod = _load_run_module()
    seeds = {run_mod._sample_seed() for _ in range(24)}
    assert len(seeds) >= 23, "seed 採取の分散が低すぎる(自動採取が実質固定)"
    assert all(isinstance(s, int) and s >= 0 for s in seeds)
    auto, kept = run_mod._resolve_seed_auto(["run.seed=auto", "run.n_agents=5"])
    assert auto is True and "run.seed=auto" not in kept and "run.n_agents=5" in kept
    auto2, kept2 = run_mod._resolve_seed_auto(["run.seed_auto=true"])
    assert auto2 is True and "run.seed_auto=true" in kept2   # 記録のため残す
    auto3, kept3 = run_mod._resolve_seed_auto(["run.seed=42"])
    assert auto3 is False and "run.seed=42" in kept3         # 数値 seed は不変


def _cli_main(run_mod, monkeypatch, tmp_path, name, *extra):
    argv = ["run.py", "model.backend=mock", "run.n_agents=6", "run.n_steps=2",
            f"run.name={name}", f"run.out_dir={tmp_path.as_posix()}", *extra]
    monkeypatch.setattr(sys, "argv", argv)
    run_mod.main()
    return tmp_path / name


def _load_l1_parquet(run_dir):
    import pyarrow.parquet as pq
    t = pq.read_table(str(run_dir / "l1_events.parquet"),
                      columns=["step", "agent_id", "kind", "payload"])
    d = t.to_pydict()
    return list(zip(d["step"], d["agent_id"], d["kind"], d["payload"]))


def test_seed_auto_end_to_end_reproduces(tmp_path, monkeypatch):
    """run.py: seed_auto=2回で異なる seed が採れ config/summary に記録・採れた seed で固定再実行=バイト再現。"""
    from omegaconf import OmegaConf
    run_mod = _load_run_module()
    d1 = _cli_main(run_mod, monkeypatch, tmp_path, "cli_a", "run.seed_auto=true")
    d2 = _cli_main(run_mod, monkeypatch, tmp_path, "cli_b", "run.seed_auto=true")
    s1 = int(OmegaConf.load(d1 / "config.yaml").run.seed)
    s2 = int(OmegaConf.load(d2 / "config.yaml").run.seed)
    assert s1 != s2, "seed_auto の 2 ランで同じ seed が採れた(自動採取が固定化)"
    summ = json.loads((d1 / "summary.json").read_text(encoding="utf-8"))
    assert summ.get("seed") == s1 and summ.get("seed_source") == "auto", \
        "採取 seed が summary.json に記録されていない(=失っている)"
    # 採れた seed を数値で固定再実行 → l1_events が d1 とバイト再現(決定論エンジンは不変)。
    d3 = _cli_main(run_mod, monkeypatch, tmp_path, "cli_c", f"run.seed={s1}")
    assert _load_l1_parquet(d1) == _load_l1_parquet(d3), \
        "採れた seed で固定再実行しても L1 が再現しない(決定論が壊れている)"
