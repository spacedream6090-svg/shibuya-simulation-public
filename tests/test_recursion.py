"""再帰性(第9バッチ 2026-07-07)= 規範・現状の監視→知覚→フィードバック→改変 のテスト。

方針(既存の鉄則を継承):
- OFF(既定): rule_repealed/norm_digest 0 件・プロンプト注入なし・イベント列は純粋既定と
  L1 完全一致(144 step。ゴールデン golden_baseline_l1.json も別途守られる)。
- ON: (1)実効ルールの知覚(norm_line)と昨日の街の動き(digest_line)がプロンプトに載る。
  (2)repeal 提案の成立で既存ルールが廃止される(rule_repealed + active から除去)。
  (3)取り締まりが多発したルールが日次でニュース化される(impact_news)。
- 決定論: ON 同士2回で L1 完全一致。R1: 呼数は k 非依存(compute_matched 下で free==off)。
検証は mock / FixedLLM のみ(実LLM 禁止)。
"""
from __future__ import annotations

import json

from society.config import load_config
from society.engine import scheduler
from society.engine.simulation import Simulation

_ON = {"recursion.enabled": "true"}


def _sim(tmp_path, name, n=30, steps=1, **ov):
    dot = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
           "observer.snapshot_every=1"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _kind(sim, kind):
    return [e for e in sim.logger.events if e.kind == kind]


def _enact_fee(sim, name="カフェ割引の取り決め", proposer=0):
    """テスト用: fee ルールを直接制定して record を返す。"""
    rec = sim.rulebook.enact({"type": "fee", "target_cat": "cafe", "delta": -200},
                             name=name, proposer=proposer, step=0, day=0)
    assert rec is not None, "テスト前提の fee ルールが制定できない"
    return rec


# --------------------------------------------------------------------- OFF 既定
def test_off_matches_pure_default(tmp_path):
    """明示 OFF と純粋既定が L1 完全一致(144 step)。再帰性の新イベントは 1 件も出ない。"""
    pure = _sim(tmp_path, "pure", steps=144)
    pure.run()
    off = _sim(tmp_path, "expl_off", steps=144, **{"recursion.enabled": "false"})
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(recursion seam が no-op でない)"
    for k in ("rule_repealed", "norm_digest"):
        assert not _kind(pure, k), f"OFF で {k} が出ている"
    assert pure.recursion.norm_line(pure) is None
    assert pure.recursion.digest_line() is None


# --------------------------------------------------------------------- 知覚(norm/digest)
def test_norm_and_digest_lines(tmp_path):
    """実効ルールの知覚行と「昨日の街の動き」行が生成され、プロンプトに載る。"""
    from society.cognition import deliberate
    sim = _sim(tmp_path, "lines", steps=1, **_ON)
    sim.run()
    rec = sim.recursion
    assert rec.norm_line(sim) is None, "ルールが無いのに知覚行が出ている"

    _enact_fee(sim)
    line = rec.norm_line(sim)
    assert line is not None and "カフェ" in line and "-200円" in line, \
        f"知覚行がルールを描画していない: {line}"

    # 日次境界: 初日(確定のみ)→ 翌日にダイジェスト確定 + norm_digest イベント
    rec.on_day(sim, 0, 0, 0)
    rec.note("proposals")
    rec.note("enforced", rule_id=0, rule_name="カフェ割引の取り決め")
    assert rec.digest_line() is None, "前日が無いのにダイジェストが出ている"
    n0 = len(_kind(sim, "norm_digest"))
    rec.on_day(sim, 1, 144, 1440)
    dline = rec.digest_line()
    assert dline is not None and "提案1件" in dline and "取り締まり1件" in dline, \
        f"ダイジェスト行が不正: {dline}"
    dg = _kind(sim, "norm_digest")
    assert len(dg) == n0 + 1 and dg[-1].agent_id == -1 \
        and dg[-1].payload["proposals"] == 1 and dg[-1].payload["enforced"] == 1 \
        and dg[-1].payload["active_rules"] == 1, "norm_digest の payload が不正"

    agent = sim.agents[0]
    prompt = deliberate.build_prompt(agent, place_name="路上", surprise="solo",
                                     nearby_names=[], sim_min=600, step=3,
                                     norm_line=line, digest_line=dline)
    assert line in prompt and dline in prompt, "知覚行がプロンプトに載っていない"
    plain = deliberate.build_prompt(agent, place_name="路上", surprise="solo",
                                    nearby_names=[], sim_min=600, step=3)
    assert "いまの街の取り決め" not in plain and "昨日の街の動き" not in plain


# --------------------------------------------------------------------- 改変(repeal)
def test_repeal_flow(tmp_path):
    """成立した repeal 提案が既存ルールを廃止する(rule_repealed + active から除去 + ニュース)。"""
    sim = _sim(tmp_path, "repeal", steps=1, **_ON)
    sim.run()
    _enact_fee(sim)
    assert len(sim.rulebook.active) == 1
    author = sim.agents[1]
    pr = {"id": 99, "author": author.id, "text": "カフェ割引をやめるべきだ",
          "supporters": {author.id}, "passed": True,
          "rule": {"type": "repeal", "target": "カフェ割引"}}
    sim.tools._enact_rule(sim, pr, author, step=10, sim_min=100)
    assert sim.rulebook.active == [], "repeal 成立後もルールが残っている"
    assert sim.rulebook.n_repealed == 1
    ev = _kind(sim, "rule_repealed")
    assert ev and ev[-1].payload["name"] == "カフェ割引の取り決め" \
        and ev[-1].payload["repealed_by"] == author.id, "rule_repealed が不正"
    assert sim.net.news and sim.net.news[-1]["title"] == "取り決めの廃止"

    # 対象が見つからない repeal は何もしない(文言制度に降格=壊れない)
    _enact_fee(sim, name="別の取り決め")
    pr2 = {**pr, "id": 100, "rule": {"type": "repeal", "target": "存在しない名前"}}
    sim.tools._enact_rule(sim, pr2, author, step=11, sim_min=110)
    assert len(sim.rulebook.active) == 1, "対象不明の repeal がルールを消した"


def test_repeal_disabled_keeps_rule(tmp_path):
    """recursion OFF(既定)では repeal 型は従来どおり無効(文言制度へ降格)=ルールは残る。"""
    sim = _sim(tmp_path, "norepeal", steps=1)
    sim.run()
    _enact_fee(sim)
    author = sim.agents[1]
    pr = {"id": 99, "author": author.id, "text": "カフェ割引をやめるべきだ",
          "supporters": {author.id}, "passed": True,
          "rule": {"type": "repeal", "target": "カフェ割引"}}
    sim.tools._enact_rule(sim, pr, author, step=10, sim_min=100)
    assert len(sim.rulebook.active) == 1, "OFF なのに repeal が効いた"
    assert not _kind(sim, "rule_repealed")


# --------------------------------------------------------------------- 自己観測(impact_news)
def test_impact_news_on_heavy_enforcement(tmp_path):
    """前日の執行が閾値以上のルールが日次でニュース化される(社会の自己観測)。"""
    sim = _sim(tmp_path, "impact", steps=1, **_ON,
               **{"recursion.impact_news_threshold": "3"})
    sim.run()
    rec = sim.recursion
    rec.on_day(sim, 0, 0, 0)
    for _ in range(3):
        rec.note("enforced", rule_id=7, rule_name="夜間の見回り強化")
    n_news = len(sim.net.news)
    rec.on_day(sim, 1, 144, 1440)
    assert len(sim.net.news) == n_news + 1, "執行多発なのにニュースが出ていない"
    art = sim.net.news[-1]
    assert art["title"] == "取り締まり多発" and "夜間の見回り強化" in art["text"] \
        and "3件" in art["text"]

    # 閾値未満なら出ない
    rec.note("enforced", rule_id=7, rule_name="夜間の見回り強化")
    n_news = len(sim.net.news)
    rec.on_day(sim, 2, 288, 2880)
    assert len(sim.net.news) == n_news, "閾値未満なのにニュースが出た"


# --------------------------------------------------------------------- 決定論
def test_all_on_deterministic(tmp_path):
    """recursion ON 同士 2 回で L1 完全一致(決定論・mock 144 step)。"""
    a = _sim(tmp_path, "det_a", steps=144, **_ON)
    a.run()
    b = _sim(tmp_path, "det_b", steps=144, **_ON)
    b.run()
    assert _l1(a) == _l1(b), "ON の決定論が崩れている"


# --------------------------------------------------------------------- R1 呼数の k 不変
class _FixedLLM:
    """常に同じ propose(+rule)を返す backend。呼数だけ数える(プロンプト内容に非依存)。"""

    def __init__(self):
        self.response = json.dumps(
            {"action": "propose", "text": "カフェをもっと使いやすくする取り組み",
             "rule": {"type": "fee", "target_cat": "cafe", "delta": -200}},
            ensure_ascii=False)
        self.calls = 0
        self.hits = 0

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        self.calls += 1
        return self.response, str(self.calls), False


def _run_k(tmp_path, name, *, writeback):
    sim = _sim(tmp_path, name, steps=144,
               **{**_ON, "controls.mode": "compute_matched",
                  "k.writeback": writeback})
    sim.llm = _FixedLLM()
    sim.run()
    return sim


def test_recursion_call_count_k_invariant(tmp_path):
    """再帰性 ON(知覚注入+提案の連鎖)でも呼数は k(writeback)に依存しない(R1)。

    知覚行の注入は内容のみ(呼数不変)。impact_news は聞く語を変え発火が動きうるが、機構は
    k・内面状態を読まないため compute_matched 下で k=free と k=off の呼数が完全一致する。"""
    free = _run_k(tmp_path, "rk_free", writeback="free")
    off = _run_k(tmp_path, "rk_off", writeback="off")
    assert free.llm.calls == off.llm.calls and free.llm.calls > 0, \
        f"呼数が k に依存(R1 違反): free={free.llm.calls} off={off.llm.calls}"
    assert _kind(free, "proposal"), "FixedLLM の propose が一度も適用されていない"
