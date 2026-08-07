"""小粒C タスク2: 噂の混線の切り分けオーバーレイ(scripts/analyze_rumor_contamination.py)。

検定するのは 3 点(タスク定義 (a)(b)(c)):
  (a) 噂由来を除外した c_transmission 系の**再計算値**が正しい
  (b) **混入率**(噂由来件数/全件数)が正しい
  (c) **凍結値との差分**が正しく、凍結値そのものは 1 つも書き換わっていない
加えて、噂 OFF のランでは **混入ゼロ・オーバーレイ = 凍結値**であることを機械固定する。
"""
from __future__ import annotations

import json
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import analyze_rumor_contamination as C  # noqa: E402
from society.observer import measure as m  # noqa: E402


# --------------------------------------------------------------------------- #
# 合成 L1
#
# 語彙(vocab-00001): 造語 0 → 伝播 0→1 → 伝播 1→2 → 2 が採用
#   frozen c_transmission: from への +1 と creator への +1(from!=creator のときだけ)
#     伝播#1 from=0(=creator)→ 0:+1 / 伝播#2 from=1 → 1:+1 と creator 0:+1
#   = {0: 2, 1: 1}
# 噂(rumor-00001):  誕生(知者 0,1)→ 伝播 0→2 → 伝播 0→3 → 0 が stifler 化
#   噂 Item は vocab_coin/label_coin を出さないので **creator 表に載らない** =
#   混入は from への +1 だけ = {0: +2}
# --------------------------------------------------------------------------- #
def _rows(with_rumor: bool):
    rows = []

    def add(step, aid, kind, **payload):
        rows.append({"step": step, "sim_min": step * 10, "agent_id": aid,
                     "kind": kind, "x": 0.0, "y": 0.0, "rng_stream": "",
                     "llm_call_id": None, "payload": json.dumps(payload)})

    add(0, 0, "vocab_coin", item_id="vocab-00001", text="foo")
    add(1, 0, "speak", text="hello", hearers=[1], items=[])
    add(1, 1, "transmission", item_id="vocab-00001", **{"from": 0}, channel="face")
    add(2, 2, "transmission", item_id="vocab-00001", **{"from": 1}, channel="face")
    add(2, 2, "label_adopt", item_id="vocab-00001", text="foo")
    if with_rumor:
        add(3, 0, "rumor_born", item_id="rumor-00001", src_kind="venture_open",
            node="n1", knowers=[0, 1])
        add(4, 2, "transmission", item_id="rumor-00001", **{"from": 0},
            channel="face")
        add(5, 3, "transmission", item_id="rumor-00001", **{"from": 0},
            channel="face")
        add(6, 0, "rumor_stifle", item_id="rumor-00001")
    return rows


def _events(with_rumor: bool):
    """measure.load_events と同形(payload は dict)の list。"""
    out = []
    for r in _rows(with_rumor):
        e = dict(r)
        e["payload"] = json.loads(r["payload"])
        out.append(e)
    return out


def _write_run(base, name, with_rumor: bool, n_agents=4):
    d = os.path.join(base, name)
    os.makedirs(d, exist_ok=True)
    rows = _rows(with_rumor)
    pq.write_table(pa.table({
        "step": pa.array([r["step"] for r in rows], pa.int32()),
        "sim_min": pa.array([r["sim_min"] for r in rows], pa.int32()),
        "agent_id": pa.array([r["agent_id"] for r in rows], pa.int32()),
        "kind": pa.array([r["kind"] for r in rows]),
        "x": pa.array([r["x"] for r in rows], pa.float64()),
        "y": pa.array([r["y"] for r in rows], pa.float64()),
        "rng_stream": pa.array([r["rng_stream"] for r in rows]),
        "llm_call_id": pa.array([r["llm_call_id"] for r in rows], pa.int32()),
        "payload": pa.array([r["payload"] for r in rows]),
    }), os.path.join(d, "l1_events.parquet"))
    with open(os.path.join(d, "agents.json"), "w", encoding="utf-8") as fh:
        json.dump([{"id": a, "name": f"a{a}"} for a in range(n_agents)], fh)
    return d


# --------------------------------------------------------------------------- #
# 識別(実態 = item_id の接頭辞。ラベルでも専用 L1 種でもない)
# --------------------------------------------------------------------------- #
def test_rumor_identified_by_item_id_prefix():
    assert C.RUMOR_PREFIX == "rumor-"
    assert C.is_rumor_item("rumor-00001") is True
    assert C.is_rumor_item("vocab-00001") is False
    assert C.is_rumor_item("label-00007") is False
    assert C.is_rumor_item(None) is False           # 欠測を噂と決めつけない
    assert C.is_rumor_item(123) is False


def test_item_id_prefix_matches_the_runtime_source():
    """接頭辞の源が society/rumors.KIND + ItemStore.new_item であることを固定。"""
    from society import rumors
    from society.observer.provenance import ItemStore
    item = ItemStore().new_item(rumors.KIND, "text", 0, 0)
    assert item.item_id.startswith(C.RUMOR_PREFIX)


def test_strip_removes_only_rumor_transmissions():
    ev = _events(with_rumor=True)
    clean = C.strip_rumor_transmissions(ev)
    assert len(ev) - len(clean) == 2                # 噂の伝播 2 件だけ
    # rumor_born / rumor_stifle は残す(凍結指標が読まないので値に影響しない)
    kinds = [e["kind"] for e in clean]
    assert "rumor_born" in kinds and "rumor_stifle" in kinds
    assert all(not C.is_rumor_transmission(e) for e in clean)


# --------------------------------------------------------------------------- #
# 噂 OFF ラン: 混入ゼロ・オーバーレイ = 凍結値
# --------------------------------------------------------------------------- #
def test_rumor_off_run_overlay_equals_frozen():
    res = C.compare(_events(with_rumor=False))
    assert res["rumors_detected"] is False
    assert res["events"]["transmission_rumor"] == 0
    assert res["events"]["contamination_rate"] == 0.0
    for name, d in res["metrics"].items():
        assert d["delta"] == 0, name
        assert d["frozen"] == d["overlay"], name
        assert d["contaminated_share"] in (0.0, None), name
    assert res["per_agent_contaminated"] == []
    assert all(res["checks"].values())


def test_rumor_off_run_frozen_equals_plain_measure_call():
    """凍結値は measure の素の呼び出しと**同一**(オーバーレイは指標を再定義しない)。"""
    ev = _events(with_rumor=False)
    meta = [{"id": a, "name": f"a{a}"} for a in range(4)]
    feats = m.agent_features(ev, meta)
    res = C.compare(ev, meta)
    assert res["metrics"]["c_transmission_total"]["frozen"] == \
        float(sum(f["c_transmission"] for f in feats))
    assert res["metrics"]["y_external_total"]["frozen"] == \
        float(sum(f["Y_external"] for f in feats))


# --------------------------------------------------------------------------- #
# 噂 ON ラン: (a) 再計算値 / (b) 混入率 / (c) 差分
# --------------------------------------------------------------------------- #
def test_on_run_counts_and_contamination_rate():
    res = C.compare(_events(with_rumor=True))
    ev = res["events"]
    assert res["rumors_detected"] is True
    assert ev["transmission_total"] == 4
    assert ev["transmission_rumor"] == 2
    assert ev["transmission_vocab"] == 2
    assert ev["rumor_born"] == 1 and ev["rumor_stifle"] == 1
    assert ev["contamination_rate"] == 0.5          # (b) 2/4


def test_on_run_c_transmission_frozen_overlay_delta():
    """(a)(c): 凍結 5 = 語彙 3 + 噂 2 → オーバーレイ 3。"""
    d = C.compare(_events(with_rumor=True))["metrics"]["c_transmission_total"]
    assert d["frozen"] == 5.0                       # {0:2,1:1} + 噂 {0:+2}
    assert d["overlay"] == 3.0                      # (a) 噂除外後
    assert d["delta"] == 2.0                        # (c)
    assert d["contaminated_share"] == round(2 / 5, 6)


def test_on_run_only_c_transmission_is_contaminated():
    """Y_external の汚染は c_transmission 1 経路に閉じる(他 3 成分は不変)。"""
    mt = C.compare(_events(with_rumor=True))["metrics"]
    for key in ("c_label_adopt_total", "c_new_relations_total",
                "c_sns_reach_total"):
        assert mt[key]["delta"] == 0.0, key
    assert mt["y_external_total"]["delta"] == mt["c_transmission_total"]["delta"]


def test_on_run_echo_transmission_metrics():
    """echo 側 3 量。★噂は novel_rate を**下げる**(同一話者の窓内再送出になるため)。"""
    mt = C.compare(_events(with_rumor=True))["metrics"]
    assert mt["n_transmission"]["frozen"] == 4.0
    assert mt["n_transmission"]["overlay"] == 2.0
    assert mt["n_transmission"]["delta"] == 2.0
    # 噂 2 件は同じ話者 0 の窓内再送出なので 1 件目だけが novel
    assert mt["n_transmission_novel"]["frozen"] == 3.0
    assert mt["n_transmission_novel"]["overlay"] == 2.0
    # 0.75(混線あり)vs 1.0(噂除外)= 差は**負**(汚染が率を押し下げていた)
    assert mt["transmission_novel_rate"]["frozen"] == 0.75
    assert mt["transmission_novel_rate"]["overlay"] == 1.0
    assert mt["transmission_novel_rate"]["delta"] == -0.25


def test_on_run_per_agent_and_checks():
    res = C.compare(_events(with_rumor=True))
    pa_rows = res["per_agent_contaminated"]
    assert [r["id"] for r in pa_rows] == [0]        # 噂を語ったのは 0 だけ
    r0 = pa_rows[0]
    assert (r0["c_transmission_frozen"], r0["c_transmission_overlay"],
            r0["delta"]) == (4, 2, 2)
    assert r0["Y_external_frozen"] - r0["Y_external_overlay"] == 2.0
    assert all(res["checks"].values())              # 自己検査 3 本とも OK


def test_frozen_values_are_untouched_by_the_overlay():
    """(c) の前提: 凍結値は全件 L1 に対する凍結関数の値そのもの。"""
    ev = _events(with_rumor=True)
    meta = [{"id": a, "name": f"a{a}"} for a in range(4)]
    feats = m.agent_features(ev, meta)
    echo = m.echo_novelty(ev)
    mt = C.compare(ev, meta)["metrics"]
    assert mt["c_transmission_total"]["frozen"] == \
        float(sum(f["c_transmission"] for f in feats))
    assert mt["n_transmission"]["frozen"] == float(echo["n_transmission"])
    assert mt["transmission_novel_rate"]["frozen"] == \
        float(echo["transmission_novel_rate"])


# --------------------------------------------------------------------------- #
# エンドツーエンド(parquet を読んで JSON + md を書く)
# --------------------------------------------------------------------------- #
def test_end_to_end_on_run(tmp_path):
    d = _write_run(str(tmp_path / "runs"), "rc_on", with_rumor=True)
    res = C.analyze(d)
    assert res["events"]["contamination_rate"] == 0.5
    assert res["metrics"]["c_transmission_total"]["overlay"] == 3.0
    md = C.render(res)
    assert "凍結値" in md and "オーバーレイ" in md
    assert "混入率" in md


def test_end_to_end_off_run_reports_zero(tmp_path):
    d = _write_run(str(tmp_path / "runs"), "rc_off", with_rumor=False)
    res = C.analyze(d)
    assert res["rumors_detected"] is False
    md = C.render(res)
    assert "噂 OFF のラン" in md


def test_cli_writes_json_and_markdown(tmp_path, monkeypatch):
    d = _write_run(str(tmp_path / "runs"), "rc_cli", with_rumor=True)
    out = str(tmp_path / "out")
    monkeypatch.setattr(sys, "argv",
                        ["analyze_rumor_contamination.py", d, "--out", out])
    assert C.main() == 0
    jpath = os.path.join(out, "rumor_contamination.json")
    assert os.path.isfile(jpath)
    assert os.path.isfile(os.path.join(out, "rumor_contamination_report.md"))
    with open(jpath, encoding="utf-8") as fh:
        res = json.load(fh)
    assert res["schema"] == C.SCHEMA
    assert set(res["metrics"]["c_transmission_total"]) == \
        {"frozen", "overlay", "delta", "contaminated_share"}
    assert res["run_dir"] == os.path.abspath(d)


def test_missing_l1_raises_systemexit(tmp_path):
    import pytest
    with pytest.raises(SystemExit):
        C.analyze(str(tmp_path / "nope"))
