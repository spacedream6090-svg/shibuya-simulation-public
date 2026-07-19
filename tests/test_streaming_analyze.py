"""P4: 観測ストリーミング化の新旧同一性 + スケール完走スモーク。

- 小さな合成ラン(多彩な kind・複数 row-group)で measure(全件 list)と
  stream(row-group 逐次)の出力がバイト同一であることを確認する。
- 合成データの完走スモークは @pytest.mark.slow(`-m "not slow"` でスキップ可)。

pandas/duckdb は使わない(pyarrow+標準ライブラリのみ)。
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "src"))

from society.observer import measure as m   # noqa: E402
from society.observer import stream as st   # noqa: E402
import analyze as az                        # noqa: E402  (scripts/analyze.py の純関数)


# --------------------------------------------------------------------------- #
# 合成ランの生成
# --------------------------------------------------------------------------- #
def _write_run(rd: Path, events, agents, n_steps, row_group_size=None):
    """events(dict 列)を l1_events.parquet + agents.json + summary.json に書く。

    row_group_size を小さく取ると複数 row-group になり、逐次読みの batch 境界を跨いだ
    集計の正しさ(agent_features の2パス・network 窓の確定・cascade の step 遷移など)を
    検証できる。rng_stream/llm_call_id 列は敢えて書かず、両経路の None 補完を突く。
    """
    rd.mkdir(parents=True, exist_ok=True)
    cols = {
        "step": [int(e["step"]) for e in events],
        "sim_min": [int(e["sim_min"]) for e in events],
        "agent_id": [int(e["agent_id"]) for e in events],
        "kind": [str(e["kind"]) for e in events],
        "x": [float(e.get("x", 0.0)) for e in events],
        "y": [float(e.get("y", 0.0)) for e in events],
        "payload": [json.dumps(e["payload"], ensure_ascii=False, sort_keys=True)
                    for e in events],
    }
    tbl = pa.table(cols)
    kw = {"row_group_size": row_group_size} if row_group_size else {}
    pq.write_table(tbl, rd / "l1_events.parquet", **kw)
    (rd / "agents.json").write_text(json.dumps(agents, ensure_ascii=False),
                                    encoding="utf-8")
    (rd / "summary.json").write_text(json.dumps({"n_steps": n_steps}),
                                     encoding="utf-8")


def _ev(step, aid, kind, **payload):
    return {"step": step, "sim_min": 420 + step * 10, "agent_id": aid,
            "kind": kind, "x": 0.0, "y": 0.0, "payload": payload}


def _rich_events():
    """多彩な kind を含む step 昇順の合成 L1(伝播・会話・SNS・ツール・制度・内省…)。"""
    ev = []
    # 造語(通常 + media 発)
    ev.append(_ev(0, 0, "vocab_coin", item_id="w1", text="fizz", fire_reason="idle",
                  drive="curiosity", place="p1", company_ids=[1], saw_feed=True,
                  adopted_n=0, recent_mem=["a", "b"]))
    ev.append(_ev(0, 1, "vocab_coin", item_id="w2", text="fizzy"))
    ev.append(_ev(0, -1, "vocab_coin", item_id="w3", text="buzz", media=True))
    ev.append(_ev(0, 2, "label_coin", item_id="w4", text="buzzed"))
    # 会話(speak/hear/dm)
    ev.append(_ev(1, 0, "speak", hearers=[1, 2], items=["w1"], text="fizz time"))
    ev.append(_ev(1, 3, "hear", speaker=0, item_ids=["w1"]))
    ev.append(_ev(1, 1, "dm", to=2, text="hi fizzy"))
    ev.append(_ev(2, 4, "sns_post", items=["w1", "w2"], text="fizz and fizzy"))
    # 伝播 + 採用
    ev.append(_ev(2, 1, "transmission", item_id="w1", channel="face", **{"from": 0}))
    ev.append(_ev(2, 2, "transmission", item_id="w1", channel="sns", **{"from": 4}))
    ev.append(_ev(3, 2, "transmission", item_id="w2", channel="face", **{"from": 1}))
    ev.append(_ev(3, 2, "label_adopt", item_id="w1", text="fizz"))
    ev.append(_ev(3, 1, "label_adopt", item_id="w2", text="fizzy"))
    ev.append(_ev(3, 5, "vocab_use", item_id="w1"))
    ev.append(_ev(3, 6, "vocab_use", item_id="w2"))
    # 反応・意見・内省・欲求
    ev.append(_ev(4, 0, "sns_like", author=4, post_id="pp"))
    ev.append(_ev(4, 1, "sns_reshare", author=4, post_id="pp"))
    ev.append(_ev(4, 3, "opinion_shift", source=0, old=0.1, new=0.4))
    ev.append(_ev(4, 0, "reflect", mode="free", written_back=True, belief="deep"))
    ev.append(_ev(4, 1, "reflect", mode="free", written_back=False, belief=""))
    ev.append(_ev(4, 2, "llm_deliberate", trigger="salience"))
    ev.append(_ev(4, 0, "drive_request", granted=True, drive="d"))
    ev.append(_ev(4, 5, "drive_request", granted=False, drive="d"))
    # 到着(landmark 用)
    ev.append(_ev(5, 0, "arrive", node="n1"))
    ev.append(_ev(5, 0, "arrive", node="n2"))
    ev.append(_ev(5, 1, "arrive", node="n1"))
    # ツール(世界に働きかける affordance)
    ev.append(_ev(6, 0, "event_host", event_id="e1", title="party", node="n1",
                  start_step=6))
    ev.append(_ev(6, 1, "event_attend", event_id="e1", host=0, title="party"))
    ev.append(_ev(6, 2, "event_attend", event_id="e1", host=0, title="party"))
    # 実行者 aid<0 のツールイベント(measure は丸ごと無視=host 0 に加点しない)
    ev.append(_ev(6, -1, "event_attend", event_id="e1", host=0, title="party"))
    ev.append(_ev(6, 0, "flyer_post", author=0, node="n1", text="come", items=[]))
    ev.append(_ev(6, 3, "flyer_view", author=0, node="n1"))
    ev.append(_ev(7, 1, "group_found", group_id="g1", name="club", founder=1))
    ev.append(_ev(7, 2, "group_join", group_id="g1", name="club", founder=1))
    ev.append(_ev(7, 0, "proposal", proposal_id="pr1", text="clean the park"))
    ev.append(_ev(7, 1, "proposal_support", proposal_id="pr1", author=0))
    ev.append(_ev(7, 2, "proposal_support", proposal_id="pr1", author=0))
    ev.append(_ev(8, 0, "proposal_passed", proposal_id="pr1", text="clean the park",
                  supporters=5))
    ev.append(_ev(8, 3, "venture_open", name="stall", offer="food", node="n1",
                  cost=10, balance=90))
    ev.append(_ev(8, 3, "venture_sale", amount=5.0, balance=95, buyer=1))
    ev.append(_ev(8, 3, "venture_sale", amount=7.0, balance=102, buyer=2))
    ev.append(_ev(9, 3, "venture_close", name="stall", node="n1"))
    # 制度化(既定 OFF)・制度DSL(既定 ON)
    ev.append(_ev(9, 0, "institution", name="norm1", norm_text="no litter",
                  created_by=0))
    ev.append(_ev(9, 1, "institution_rule", rule_id="r1", type="bonus", name="park",
                  proposer=1, rule={}))
    # 世界イベント(agent_id=-1)は agent 集合に入らない
    ev.append(_ev(9, -1, "weather", date="2026-07-20", weekday="Mon", cond="晴",
                  temp_hi=30))
    return ev


def _agents():
    return [{"id": i, "name": f"A{i}", "occupation": "x",
             "nrg": round(0.1 * i, 3)} for i in range(7)]


def _traits():
    return {i: {"tA": round(0.05 * i, 3), "tB": round(0.03 * (7 - i), 3)}
            for i in range(7)}


# --------------------------------------------------------------------------- #
# 同一性(measure list 版 == stream 逐次版)
# --------------------------------------------------------------------------- #
def _canon(o):
    return json.dumps(o, ensure_ascii=False, sort_keys=True, default=str)


def _assert_same(a, b, label):
    assert _canon(a) == _canon(b), f"{label} が新旧で不一致"


@pytest.fixture(params=[None, 3, 7, 10000], ids=["rg=all", "rg=3", "rg=7", "rg=big"])
def rich_run(request, tmp_path):
    """複数 row-group サイズで同じ合成ランを用意(batch 境界の網羅)。"""
    rd = tmp_path / f"run_rg_{request.param}"
    _write_run(rd, _rich_events(), _agents(), n_steps=10,
               row_group_size=request.param)
    return rd


def test_stream_events_equals_load_events(rich_run):
    rd = str(rich_run)
    assert list(m.stream_events(rd)) == m.load_events(rd)
    # 列射影は要求キーだけを返す
    for e in m.stream_events(rd, columns=["step", "kind"]):
        assert set(e) == {"step", "kind"}
        break


def test_agent_features_identity(rich_run):
    rd = str(rich_run)
    events = m.load_events(rd)
    agents, traits = _agents(), _traits()
    _assert_same(m.agent_features(events), st.agent_features(rd),
                 "agent_features(no-meta)")
    _assert_same(m.agent_features(events, agents, traits),
                 st.agent_features(rd, agents, traits),
                 "agent_features(meta,traits)")
    yw = {"venture_revenue": 0.01, "groups_founded": 1.0, "events_hosted": 0.5}
    _assert_same(m.agent_features(events, agents, traits, y_weights=yw),
                 st.agent_features(rd, agents, traits, y_weights=yw),
                 "agent_features(y_weights)")
    ls = {"n1": 1.0, "n2": 0.25}
    _assert_same(m.agent_features(events, agents, traits, landmark_score=ls),
                 st.agent_features(rd, agents, traits, landmark_score=ls),
                 "agent_features(landmark)")


def test_cascades_network_collective_identity(rich_run):
    rd = str(rich_run)
    events = m.load_events(rd)
    l2 = m.load_l2(rd)
    n_agents = len(_agents())
    _assert_same(m.item_cascades(events), st.item_cascades(rd), "item_cascades")
    _assert_same(m.network_windows(events), st.network_windows(rd),
                 "network_windows")
    _assert_same(m.network_windows(events, window=3),
                 st.network_windows(rd, window=3), "network_windows(w=3)")
    _assert_same(m.collective_series(events, l2, n_agents),
                 st.collective_series(rd, l2, n_agents), "collective_series")


def test_communities_and_drift_identity(rich_run):
    rd = str(rich_run)
    events = m.load_events(rd)
    n_agents = len(_agents())
    _assert_same({str(k): v for k, v in m.communities(events, n_agents).items()},
                 {str(k): v for k, v in st.communities(rd, n_agents).items()},
                 "communities")
    _assert_same(m.drift_metrics(events, n_agents),
                 st.drift_metrics(rd, n_agents), "drift_metrics")


def test_coinage_and_tools_identity(rich_run):
    rd = str(rich_run)
    events = m.load_events(rd)
    _assert_same(az.coinage_contexts(events), st.coinage_contexts(rd),
                 "coinage_contexts")
    _assert_same(az.tool_usage(events), st.tool_usage(rd), "tool_usage")


def test_analyze_cli_end_to_end(tmp_path):
    """scripts/analyze.py の analyze() が逐次経路で最後まで走り、成果物を出す。"""
    rd = tmp_path / "run_cli"
    _write_run(rd, _rich_events(), _agents(), n_steps=10, row_group_size=5)
    (rd / "traits.json").write_text(json.dumps(_traits()), encoding="utf-8")
    outdir, n_events, n_items = az.analyze(str(rd))
    assert n_events == len(_rich_events())
    for name in ("agent_features.json", "items.json", "collective.json",
                 "drift.json", "tools.json", "r2.json", "report.md"):
        assert (Path(outdir) / name).exists(), name


def test_count_events(rich_run):
    assert m.count_events(str(rich_run)) == len(_rich_events())


# --------------------------------------------------------------------------- #
# 完走スモーク(スケール): 複数 row-group の合成データで逐次集計が完走する
# --------------------------------------------------------------------------- #
def _synth_scale_events(n_agents, n_steps, per_step):
    """決定論の合成イベント生成器(step 昇順)。会話・伝播・採用・欲求を織り交ぜる。"""
    words = [f"w{i}" for i in range(64)]
    for s in range(n_steps):
        for j in range(per_step):
            a = (s * per_step + j) % n_agents
            r = (s + j) % 7
            if r == 0:
                yield _ev(s, a, "speak", hearers=[(a + 1) % n_agents,
                                                  (a + 2) % n_agents],
                          items=[words[(a) % len(words)]], text="t")
            elif r == 1:
                yield _ev(s, a, "transmission", item_id=words[a % len(words)],
                          channel="face", **{"from": (a + 1) % n_agents})
            elif r == 2:
                yield _ev(s, a, "label_adopt", item_id=words[a % len(words)],
                          text="t")
            elif r == 3:
                yield _ev(s, a, "vocab_use", item_id=words[a % len(words)])
            elif r == 4:
                yield _ev(s, a, "drive_request", granted=(j % 2 == 0), drive="d")
            elif r == 5:
                yield _ev(s, a, "dm", to=(a + 3) % n_agents, text="m")
            else:
                yield _ev(s, a, "reflect", mode="free",
                          written_back=(j % 3 == 0), belief="b" * (j % 4))
        if s % 50 == 0:                    # ときどき造語
            yield _ev(s, s % n_agents, "vocab_coin",
                      item_id=words[s % len(words)], text=f"c{s % len(words)}")


@pytest.mark.slow
def test_scale_smoke_streaming_completes(tmp_path):
    """多 row-group の合成データで全ストリーミング集計が完走し、measure と一致する。"""
    n_agents, n_steps, per_step = 500, 1500, 100     # 約 15 万件
    rd = tmp_path / "scale"
    rd.mkdir(parents=True, exist_ok=True)
    events = list(_synth_scale_events(n_agents, n_steps, per_step))
    _write_run(rd, events, [{"id": i} for i in range(n_agents)],
               n_steps=n_steps, row_group_size=20000)   # 複数 row-group
    rds = str(rd)
    assert pq.ParquetFile(rd / "l1_events.parquet").metadata.num_row_groups > 1
    # 逐次版が完走し、全件 list 版と一致(スケールでも結果不変)
    _assert_same(m.item_cascades(events), st.item_cascades(rds), "cascades(scale)")
    _assert_same(m.network_windows(events), st.network_windows(rds), "net(scale)")
    _assert_same(m.collective_series(events, None, n_agents),
                 st.collective_series(rds, None, n_agents), "collective(scale)")
    _assert_same(m.agent_features(events), st.agent_features(rds), "feats(scale)")
    # 逐次版は count も O(1)
    assert m.count_events(rds) == len(events)
