"""未定義行動レジスタ + 沈黙の第一級化(第70バッチ IDEA②)のテスト。

方針(R1 の鉄則を継承):
- 既定 OFF(freedom.undefined_register=false / freedom.explicit_nothing=false)では
  **ヘッダ 1 バイト不変・L1 バイト一致・L2 列なし・LLM 呼数不変**(ゴールデン
  tests/data/golden_baseline_l1.json は tests/test_scenario.py が別途担保)。
- ON: enum 外だが行動を主張している出力が `undefined_action` として残り、`fallback` は
  同数だけ減る(**振り分けの保存則**)。壊れた JSON は従来どおり `fallback`。
- 沈黙: "nothing" は ON/OFF に関わらず寛容受理(P2 の move_home と同じ流儀)。
  L1 `stay{reason:"chosen_nothing"}` を出すのは ON のときだけ。
- 既存の開放行動 "do"/"free"/"activity" が未定義側へ落ちない(回帰)。

検証は mock / 固定応答のみ(実 LLM 禁止)。
"""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from society.cognition import deliberate
from society.config import load_config
from society.engine.simulation import Simulation
from society.observer import silence as silence_mod
from society.observer.schema import EVENT_KINDS

REPO_ROOT = Path(__file__).resolve().parents[1]

# enum 外だが「構文としては行動を主張している」出力(kibo_crew_sim の interact 発明が実例)
_UNKNOWN = json.dumps({"action": "interact", "with": "隣の人", "how": "手伝う"},
                      ensure_ascii=False)


class _FixedLLM:
    """応答をプロンプトに依存させない backend(呼数だけ数える)。"""

    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.hits = 0

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        self.calls += 1
        return self.response, str(self.calls), False


def _run(tmp_path, name, response=None, n=10, steps=16, **ov):
    dot = [f"run.n_agents={n}", f"run.n_steps={steps}", f"run.name={name}",
           "run.seed=42", "model.backend=mock", "observer.snapshot_every=144",
           f"run.out_dir={(tmp_path / 'runs').as_posix()}"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    sim = Simulation(load_config(dot), out_dir=tmp_path / name)
    if response is not None:
        sim.llm = _FixedLLM(response)
    sim.run()
    return sim


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _kinds(sim):
    out: dict = {}
    for e in sim.logger.events:
        out[e.kind] = out.get(e.kind, 0) + 1
    return out


def _l2_cols(sim):
    return pq.read_table(sim.out_dir / "l2_metrics.parquet").column_names


# --------------------------------------------------------------------------- #
# 1) classify_reject(純関数・観測専用)
# --------------------------------------------------------------------------- #
def test_classify_reject_three_reasons():
    reason, info = deliberate.classify_reject(_UNKNOWN)
    assert reason == "unknown_action"
    assert info["action"] == "interact"
    assert info["keys"] == ["how", "with"]            # トップレベルのキー名(action を除く・ソート)
    assert info["text"] == "手伝う"                    # 最初の非空文字列値の要約
    # 壊れた JSON は従来カテゴリ
    assert deliberate.classify_reject("{壊れて")[0] == "broken_json"
    assert deliberate.classify_reject("")[0] == "broken_json"
    # dict だが action が無い / 既知動詞なのに必須欄が空 → missing_field
    assert deliberate.classify_reject('{"text":"あ"}')[0] == "missing_field"
    assert deliberate.classify_reject('{"action":"speak"}')[0] == "missing_field"


def test_classify_reject_truncates_payload():
    """自由テキストは要約(action 40字 / キー12個・各24字 / text 120字)に切り詰める。"""
    long_text = "あ" * 500
    raw = json.dumps({"action": "x" * 100, "note": long_text,
                      **{f"k{i}" * 20: i for i in range(30)}}, ensure_ascii=False)
    _reason, info = deliberate.classify_reject(raw)
    assert len(info["action"]) == 40
    assert len(info["keys"]) == 12 and all(len(k) <= 24 for k in info["keys"])
    assert len(info["text"]) == 120


def test_known_actions_never_classified_unknown():
    """`KNOWN_ACTIONS` と parse_action の分岐の同期ズレを機械的に検知する。"""
    for verb in sorted(deliberate.KNOWN_ACTIONS):
        raw = json.dumps({"action": verb}, ensure_ascii=False)
        reason, _info = deliberate.classify_reject(raw)
        assert reason != "unknown_action", f"既知動詞が未定義扱い: {verb}"
    # 逆に、parse_action が受理する動詞は KNOWN_ACTIONS に入っていること
    for verb in ("speak", "coin_label", "post", "dm", "wander", "reflect",
                 "plan", "recall", "do", "free", "activity", "job_search",
                 "move_home", "buy", "study", "propose_partnership",
                 "break_up", "nothing"):
        assert verb in deliberate.KNOWN_ACTIONS, verb


# --------------------------------------------------------------------------- #
# 2) parse_action の受理(既存の受理挙動を壊さない)
# --------------------------------------------------------------------------- #
def test_free_action_still_accepted_not_undefined():
    """開放行動 "do"/"free"/"activity" は従来どおり受理され undefined に落ちない(回帰)。"""
    for verb in ("do", "free", "activity"):
        act = deliberate.parse_action(
            json.dumps({"action": verb, "what": "本を読む"}, ensure_ascii=False))
        assert act is not None and act["type"] == "free_action"
        assert deliberate.classify_reject(
            json.dumps({"action": verb}, ensure_ascii=False))[0] == "missing_field"


def test_nothing_is_leniently_accepted():
    """「関わらない/何もしない」は OFF でも常に寛容受理(提示されないだけ)。"""
    for verb in ("nothing", "none", "do_nothing", "idle"):
        act = deliberate.parse_action(json.dumps({"action": verb}))
        assert act == {"type": "stay", "reason": "chosen_nothing"}, verb
    # wander は従来どおり reason 無しの stay(payload バイト一致の保護)
    assert deliberate.parse_action('{"action":"wander"}') == {"type": "stay"}


def test_header_line_is_added_only_when_enabled():
    base = deliberate._header("constrained", False, "X")
    on = deliberate._header("constrained", False, "X", True)
    assert on == base + deliberate._NOTHING_LINE
    assert on.count("\n") == base.count("\n") + 1     # ちょうど 1 行だけ増える
    # open_actions との併用でも既存の順序(do → nothing)を保つ
    both = deliberate._header("constrained", True, "X", True)
    assert both.endswith(deliberate._DO_LINE + deliberate._NOTHING_LINE)


# --------------------------------------------------------------------------- #
# 3) 既定 OFF = 完全な no-op
# --------------------------------------------------------------------------- #
def test_off_is_byte_identical_and_has_no_columns(tmp_path):
    """既定と明示 false が L1 バイト一致・undefined_action 0 件・L2 列なし。"""
    a = _run(tmp_path, "u_def", response=_UNKNOWN)
    b = _run(tmp_path, "u_off", response=_UNKNOWN,
             **{"freedom.undefined_register": "false",
                "freedom.explicit_nothing": "false"})
    assert _l1(a) == _l1(b)
    assert _kinds(a).get("undefined_action", 0) == 0
    assert _kinds(a).get("fallback", 0) > 0, "検証にならない(そもそも棄却が起きていない)"
    assert [c for c in _l2_cols(a) if c in silence_mod.COLUMNS] == []


def test_nothing_off_emits_no_stay_event(tmp_path):
    """explicit_nothing OFF では "nothing" を受理しても L1 イベントは 1 件も増えない。"""
    sim = _run(tmp_path, "n_off", response=json.dumps({"action": "nothing"}))
    assert _kinds(sim).get("stay", 0) == 0
    assert _kinds(sim).get("fallback", 0) == 0        # 受理されている(棄却ではない)


# --------------------------------------------------------------------------- #
# 4) ON の振る舞い(記録される・保存則・呼数不変)
# --------------------------------------------------------------------------- #
def test_on_records_undefined_action_and_conserves_fallback(tmp_path):
    """ON は fallback を undefined_action へ **振り分ける**(分子が排他=保存則)。"""
    off = _run(tmp_path, "u2_off", response=_UNKNOWN)
    on = _run(tmp_path, "u2_on", response=_UNKNOWN,
              **{"freedom.undefined_register": "true"})
    k_off, k_on = _kinds(off), _kinds(on)
    assert k_on.get("undefined_action", 0) == k_off.get("fallback", 0) > 0
    assert k_on.get("fallback", 0) == 0
    # 分子の保存則: fallback + undefined = 旧 fallback
    assert (k_on.get("fallback", 0) + k_on.get("undefined_action", 0)
            == k_off.get("fallback", 0))
    # LLM 呼数は完全一致(パース後の分岐だけ=R1)
    assert on.llm.calls == off.llm.calls > 0
    # payload は action 名 + キー名 + 要約テキスト + trigger のみ
    ev = next(e for e in on.logger.events if e.kind == "undefined_action")
    assert set(ev.payload) == {"action", "keys", "text", "trigger"}
    assert ev.payload["action"] == "interact"


def test_on_emits_l2_columns_with_conserved_rate(tmp_path):
    """L2: undefined_action_total/rate が出て、llm_fallback_rate と分子が排他になる。"""
    on = _run(tmp_path, "u3_on", response=_UNKNOWN,
              **{"freedom.undefined_register": "true",
                 "observer.llm_health.enabled": "true"})
    tbl = pq.read_table(on.out_dir / "l2_metrics.parquet").to_pydict()
    assert "undefined_action_total" in tbl and "undefined_action_rate" in tbl
    assert tbl["undefined_action_total"][-1] == on.logger.n_undefined_action_events > 0
    assert tbl["llm_fallback_rate"][-1] == 0.0        # 全部 undefined 側へ回った
    calls = tbl["llm_calls_total"][-1]
    assert tbl["undefined_action_rate"][-1] == round(
        tbl["undefined_action_total"][-1] / calls, 6)


def test_broken_json_still_falls_back_when_on(tmp_path):
    """壊れた JSON は ON でも従来どおり fallback(undefined_action にしない)。"""
    sim = _run(tmp_path, "u4", response="{これは JSON では",
               **{"freedom.undefined_register": "true"})
    k = _kinds(sim)
    assert k.get("fallback", 0) > 0
    assert k.get("undefined_action", 0) == 0


def test_explicit_nothing_on_records_stay_and_keeps_call_count(tmp_path):
    """ON で `stay{reason:"chosen_nothing"}` が立ち、LLM 呼数は OFF と完全一致。"""
    off = _run(tmp_path, "n2_off", response=json.dumps({"action": "nothing"}))
    on = _run(tmp_path, "n2_on", response=json.dumps({"action": "nothing"}),
              **{"freedom.explicit_nothing": "true"})
    stays = [e for e in on.logger.events if e.kind == "stay"]
    assert stays and all(e.payload["reason"] == "chosen_nothing" for e in stays)
    assert _kinds(off).get("stay", 0) == 0
    assert on.llm.calls == off.llm.calls > 0
    cols = _l2_cols(on)
    assert "silent_agent_rate" in cols and "chosen_nothing_rate" in cols
    tbl = pq.read_table(on.out_dir / "l2_metrics.parquet").to_pydict()
    assert 0.0 <= tbl["silent_agent_rate"][-1] <= 1.0
    assert tbl["chosen_nothing_rate"][-1] > 0.0


def test_silent_agent_rate_counts_non_speakers(tmp_path):
    """全員が沈黙(nothing のみ)のランでは silent_agent_rate = 1.0。"""
    sim = _run(tmp_path, "n3", response=json.dumps({"action": "nothing"}),
               **{"freedom.explicit_nothing": "true"})
    tbl = pq.read_table(sim.out_dir / "l2_metrics.parquet").to_pydict()
    assert tbl["silent_agent_rate"][-1] == 1.0


# --------------------------------------------------------------------------- #
# 5) 契約
# --------------------------------------------------------------------------- #
def test_event_kind_registered():
    assert "undefined_action" in EVENT_KINDS
    assert "stay" in EVENT_KINDS          # 沈黙は既存 kind を再利用(schema 追加なし)
