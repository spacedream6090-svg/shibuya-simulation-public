"""バッチH: 「所持ツール」標準装備(tools.equip_all)の検証。

方針:
- equip_all=true → 全発火プロンプト(deliberate)に固定順・固定書式の「所持ツール」節。
  中立記述(勧誘語なし)・行動名5種・客観条件(所持金/場所)のみ = k 非依存(R1)。
- equip_all=false(既定)→ 現行と文字列レベルで完全一致(節は純追記なので前方バイト不変)。
- LLM 呼数不変: 装備は generate 呼び出しを一切増やさず「プロンプト内容だけ」を変える。
  mock は全プロンプト文字列で乱数を引くため装備で挙動が変わり総 llm_calls は正当に変動する
  (返答連鎖の再編)。よって「呼数不変」は挙動を固定した backend で厳密検証する。
"""
from __future__ import annotations

import json

from society.config import load_config
from society.engine.simulation import Simulation
from society.cognition import deliberate

TOOL_ORDER = ["propose", "host_event", "post_flyer", "found_group", "open_venture"]


def _agent(tmp_path, name="equip", n=4):
    cfg = load_config([f"run.n_agents={n}", "run.n_steps=1", f"run.name={name}"])
    return Simulation(cfg, out_dir=tmp_path / name).agents[0]


# --------------------------------------------------------------- 節の中身
def test_equip_lists_five_tools_in_fixed_order(tmp_path):
    ag = _agent(tmp_path, "order")
    out = deliberate.build_prompt(ag, place_name="渋谷", surprise="solo",
                                  nearby_names=[], sim_min=0, step=0,
                                  equip_all=True, venture_cost=30000.0)
    assert "所持ツール" in out
    positions = [out.index(name) for name in TOOL_ORDER]
    assert positions == sorted(positions), positions          # 固定順
    assert all(name in out for name in TOOL_ORDER)             # 5種すべて


def test_equip_condition_display_money(tmp_path):
    """open_venture の客観条件表示(所持金30000円)が現在額で正しく可否を出す。"""
    ag = _agent(tmp_path, "money")
    ag.money = 5000.0
    low = deliberate.build_prompt(ag, place_name="渋谷", surprise="solo",
                                  nearby_names=[], sim_min=0, step=0,
                                  equip_all=True, venture_cost=30000.0)
    assert "30000円以上" in low
    ov_low = next(l for l in low.splitlines() if l.startswith("- open_venture"))
    assert "不足" in ov_low and "5000円" in ov_low            # 不足表示 + 現在額

    ag.money = 50000.0
    hi = deliberate.build_prompt(ag, place_name="渋谷", surprise="solo",
                                 nearby_names=[], sim_min=0, step=0,
                                 equip_all=True, venture_cost=30000.0)
    ov_hi = next(l for l in hi.splitlines() if l.startswith("- open_venture"))
    assert "利用可" in ov_hi and "50000円" in ov_hi


def test_equip_off_is_byte_identical_and_append_only(tmp_path):
    """equip_all=False(既定)は現行と完全一致。on 出力は off 出力への純追記。"""
    ag = _agent(tmp_path, "off")
    kw = dict(place_name="渋谷", surprise="social", nearby_names=["A"],
              sim_min=0, step=0)
    off_default = deliberate.build_prompt(ag, **kw)           # 既定
    off_explicit = deliberate.build_prompt(ag, equip_all=False, **kw)
    assert off_default == off_explicit                        # 既定 == 明示 False
    for marker in ["所持ツール", "post_flyer", "found_group", "open_venture"]:
        assert marker not in off_default                      # 節の痕跡なし

    on = deliberate.build_prompt(ag, equip_all=True, venture_cost=30000.0, **kw)
    assert on.startswith(off_default + "\n")                  # 前方バイト不変=純追記


def test_equip_section_has_no_fingerprint_or_mock_markers(tmp_path):
    """節に因子語(指紋)や world_change 系を書かない + mock の分岐マーカーを誤爆させない。"""
    ag = _agent(tmp_path, "safe")
    section = deliberate._equip_section(ag, 30000.0)
    for tok in ["nfc", "efficacy", "ownership", "world_change",
                "risk_tolerance", "internal_locus"]:          # tests/test_contracts の禁止語
        assert tok not in section
    for marker in ["他にできること", "驚き:", "内省してください",
                   "SNSに短く投稿", "メッセージを送る", "今日一日の計画"]:
        assert marker not in section                          # mock の本文分岐に衝突しない
    for line in section.splitlines():                         # mock の行頭抽出に衝突しない
        for pre in ("場所:", "知っている言葉:", "周りにある店・場所:"):
            assert not line.startswith(pre)


# --------------------------------------------------------------- 呼数不変
class _FixedLLM:
    """挙動を固定する backend(応答をプロンプトに依存させない)。呼数だけ数える。"""

    def __init__(self, response: str):
        self.response = response
        self.calls = 0
        self.hits = 0

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        self.calls += 1
        return self.response, str(self.calls), False


_ORIG_BP = deliberate.build_prompt


def _equip_firing(agent, **kw):
    """提案するスケジューラ配線の再現: 発火プロンプト(surprise!=None)だけを装備。
    planning/reflection は deliberate から build_prompt を直接 import しており、この
    module 属性差し替えの影響を受けない(=装備は発火プロンプトに限定される)。"""
    if kw.get("surprise") is not None:
        kw["equip_all"] = True
        kw["venture_cost"] = 30000.0
    return _ORIG_BP(agent, **kw)


# 並列バッチが scheduler に導入中の government 機能は、config に government キーが無いと
# _gov() が OmegaConf.to_container({}) で必ずクラッシュする(既存 test_e2e_mock も現状 red)。
# 本バッチ H は equip の検証が目的なので、その未完機能を明示 OFF にして sim を素通しさせる。
# government 既定は OFF なので挙動は不変。修正後もこの明示 OFF はそのまま無害。
_NEUTRALIZE = ["government.enabled=false"]


def _run_fixed(tmp_path, name, patch, response, steps=24, n=20, seed=42):
    cfg = load_config([f"run.seed={seed}", f"run.n_agents={n}",
                       f"run.n_steps={steps}", f"run.name={name}"] + _NEUTRALIZE)
    if patch:
        deliberate.build_prompt = _equip_firing
    try:
        sim = Simulation(cfg, out_dir=tmp_path / name)
        sim.llm = _FixedLLM(response)
        sim.run()
        return sim.llm.calls
    finally:
        deliberate.build_prompt = _ORIG_BP


def test_equip_does_not_change_llm_call_count(tmp_path):
    """挙動固定(fixed backend)なら equip on/off で LLM 呼数が完全一致。
    返答連鎖(speak→reply)も込みで一致 = 装備は呼び出しを一切増やさない。"""
    resp = json.dumps({"action": "speak", "text": "x"}, ensure_ascii=False)
    base = _run_fixed(tmp_path, "cc_base", False, resp)
    equi = _run_fixed(tmp_path, "cc_equip", True, resp)
    assert base == equi and base > 0


def test_equip_mock_24step_completes(tmp_path):
    """mock 24step を equip_all 強制 ON で完走(クラッシュなし・発火あり)。"""
    cfg = load_config(["run.seed=7", "run.n_agents=20", "run.n_steps=24",
                       "run.name=eq_mock"] + _NEUTRALIZE)
    deliberate.build_prompt = _equip_firing
    try:
        sim = Simulation(cfg, out_dir=tmp_path / "eq_mock")
        summary = sim.run()
    finally:
        deliberate.build_prompt = _ORIG_BP
    assert summary["llm_calls"] > 0
