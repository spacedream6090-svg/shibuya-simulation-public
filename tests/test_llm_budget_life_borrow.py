"""`lod.budget.tiers.life_borrow` = 自発枠の保護(2026-08-25)。

正典: docs/plans/llm-budget-respec.md §4-3。実装 src/society/cognition/lod.py。

塞ぐ穴(本番 24c 実測): 二層予算(DPH-B)の life レーン(朝の計画 plan / 夜の内省 reflect)は
general の余りを借りられるが general は借りられない。plan の需要は**在場全員ぶん毎朝**立つので、
cap が飽和するランでは life が general の余りを毎 step 全量借り切り、**昼帯の自発発火が
n_fires=0** になっていた(需要は 18.7 万件/step まで積み上がって全拒否)。

受入基準(この順で固定する)
  (1) **既定 true = 旧実装と 1 バイト同一**: 旧 `_take` の参照実装と take 列が完全一致・
      conf 既定も true・実ラン(mock)の L1 も純粋既定と完全一致。
  (2) false = life は自枠まで / general は自枠を**全量**使える(= 保護の実体)。
  (3) reply の借用は **両設定で不変**(返答保証はレーン設計の第一目的)。
  (4) 総量の硬い上限(used <= cap)は false でも不変。
  (5) レジストリ宣言(strict / affects_k=True / risk=none / off_value=True)。
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from omegaconf import OmegaConf

from society import registry as R
from society.cognition import lod as LOD
from society.config import load_config
from society.engine.simulation import Simulation
from society.observer import starvation as SV

_REPO = Path(__file__).resolve().parents[1]

TIERS_ON = {"lod.budget.tiers.enabled": "true"}


# --------------------------------------------------------------------------- #
# 共通ヘルパ
# --------------------------------------------------------------------------- #
def _tiers(**ov) -> dict:
    raw = {"enabled": True}
    raw.update(ov)
    return LOD.build_budget_cfg(raw)


class _LegacyBudget(LOD.LodBudget):
    """本キーが入る**前**の `_take`(2026-08-25 以前)。同値ピンの参照実装。

    ★ここを書き換えてはならない: 「既定 true は旧実装と 1 ビットも違わない」という
      主張は、旧コードそのものと突き合わせて初めて主張になる。
    """

    def _take(self, lane: str) -> bool:                     # noqa: D102 (参照実装)
        if self.used >= self.max_per_step:
            return False
        if not self.tiers:
            self.used += 1
            return True
        if self.lane_used[lane] < self.caps[lane]:
            self.lane_used[lane] += 1
            self.used += 1
            return True
        if lane != "general" and self.lane_used["general"] < self.caps["general"]:
            self.lane_used["general"] += 1
            self.used += 1
            return True
        return False


def _sequence(seed: int = 42, n: int = 600) -> list[str]:
    """決定論の purpose 列(life を厚めに = 実機の plan 需要過多を模す)。"""
    rng = random.Random(seed)
    pool = (["plan"] * 5 + ["reflect"] * 2 + ["reply"] * 2
            + ["face", "interrupt", "media"])
    return [rng.choice(pool) for _ in range(n)]


def _drive(budget, seq: list[str], step_len: int = 25) -> list[bool]:
    """`seq` を step_len 件ごとに step 境界(reset)を挟んで流し、許可列を返す。"""
    out: list[bool] = []
    for i, purpose in enumerate(seq):
        if i and i % step_len == 0:
            budget.reset()
        out.append(budget.take(purpose))
    return out


def _cfg(name, n_steps=48, n_agents=12, **ov):
    dot = ["run.seed=42", f"run.n_agents={n_agents}", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock", "observer.snapshot_every=144"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def _run(tmp_path, name, n_steps=48, n_agents=12, **ov):
    sim = Simulation(_cfg(name, n_steps, n_agents, **ov), out_dir=tmp_path / name)
    sim.run()
    return sim


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _lane_granted(sim) -> dict[str, int]:
    """purpose 別 granted を 3 レーンへ畳む(観測 ON のランでのみ呼ぶ)。"""
    rows = SV.provenance(sim)["llm_budget_by_purpose"]
    out = {lane: 0 for lane in LOD.BUDGET_LANES}
    for purpose, row in rows.items():
        lane = LOD.PURPOSE_LANE.get(str(purpose), "general")
        out[lane] += int(row.get("granted", 0))
    return out


# =========================================================================== #
# (1) 既定 true = 旧実装と同一挙動
# =========================================================================== #
def test_default_is_true_everywhere():
    """既定値の在り処 3 箇所(module 既定 / build_budget_cfg / conf)が全て true。"""
    assert LOD.BUDGET_TIER_DEFAULTS["life_borrow"] is True
    assert _tiers()["life_borrow"] is True
    assert LOD.LodBudget(10, tiers=_tiers()).life_borrow is True


def test_shipped_conf_default_is_true():
    """conf/config.yaml の既定が true(= 出荷状態は現行挙動そのまま)。"""
    raw = OmegaConf.load(_REPO / "conf" / "config.yaml")
    assert raw.lod.budget.tiers.life_borrow is True
    assert load_config().lod.budget.tiers.life_borrow is True


def test_default_true_matches_the_legacy_take_exactly():
    """★同値ピン: 既定 true の take 列が旧実装と**完全一致**(1 ビットも違わない)。"""
    seq = _sequence()
    for cap in (4, 7, 10, 33, 100):
        new = LOD.LodBudget(cap, tiers=_tiers())
        old = _LegacyBudget(cap, tiers=_tiers())
        assert _drive(new, seq) == _drive(old, seq), f"cap={cap} で旧実装と挙動が違う"
        assert new.lane_used == old.lane_used and new.used == old.used


def test_default_true_still_lets_life_borrow_general_slack():
    """既定 true では life が general の余りを借りられる(旧テストと同じ主張の life 版)。"""
    b = LOD.LodBudget(10, tiers=_tiers(reply_share=0.1, life_share=0.1))
    n = sum(1 for _ in range(100) if b.take("plan"))
    assert n == 9, f"life が general の余りを借りられていない(取れたのは {n})"
    assert b.take("media") is False, "借り切った後に general が取れている"


def test_old_cfg_dict_without_the_key_falls_back_to_true():
    """キーを持たない旧 cfg dict(古い呼び出し側)から組んでも既定 true へ落ちる。"""
    legacy = {"enabled": True, "reply_share": 0.2, "life_share": 0.3,
              "max_defer_steps": 18}
    b = LOD.LodBudget(10, tiers=legacy)
    assert b.life_borrow is True
    assert sum(1 for _ in range(100) if b.take("plan")) == 3 + 5


# =========================================================================== #
# (2) false = 自発枠の保護
# =========================================================================== #
def test_false_confines_life_to_its_own_lane():
    """false では life は自枠まで(general の余りに 1 件も手を出さない)。"""
    b = LOD.LodBudget(10, tiers=_tiers(life_borrow=False))
    assert b.caps == {"reply": 2, "life": 3, "general": 5}
    assert sum(1 for _ in range(100) if b.take("plan")) == 3, "life が自枠を越えた"
    assert b.lane_used["general"] == 0, "life が general 枠を食っている"
    assert b.used == 3


def test_false_keeps_the_general_lane_whole_for_spontaneous_fires():
    """★本丸: life が全力で取りに行った後でも general が自枠を**全量**使える。"""
    b = LOD.LodBudget(10, tiers=_tiers(life_borrow=False))
    for _ in range(1000):                          # plan 需要が毎 step 積み上がる実機の形
        b.take("plan")
    assert sum(1 for _ in range(100) if b.take("media")) == 5 == b.caps["general"]
    assert b.used == 8


def test_true_lets_life_starve_the_general_lane_which_false_prevents():
    """同じ需要列で true / false を比べる = 本キーが解く問題そのものの機械固定。"""
    seq = ["plan"] * 40 + ["media"] * 40           # 先に走る plan が需要過多(実機の形)
    on = _drive(LOD.LodBudget(10, tiers=_tiers()), seq, step_len=len(seq))
    off = _drive(LOD.LodBudget(10, tiers=_tiers(life_borrow=False)),
                 seq, step_len=len(seq))
    assert (sum(on[:40]), sum(on[40:])) == (8, 0), \
        "true: life が自枠 3 + general 5 を借り切り、自発発火が 0 になる(実機の症状)"
    assert (sum(off[:40]), sum(off[40:])) == (3, 5), \
        "false: life は自枠 3 まで・general は自枠 5 を全量使える(保護の実体)"


def test_false_never_exceeds_the_hard_cap():
    """総量の硬い上限は false でも不変(どのレーンから取っても used <= cap)。"""
    seq = _sequence(seed=7, n=400)
    b = LOD.LodBudget(9, tiers=_tiers(life_borrow=False))
    granted = _drive(b, seq, step_len=50)
    assert b.used <= b.max_per_step == 9
    assert sum(granted) <= 9 * (len(seq) // 50 + 1)
    assert all(v <= b.caps[k] for k, v in b.lane_used.items()), "レーン枠を越えた"


# =========================================================================== #
# (3) reply の借用は両設定で不変
# =========================================================================== #
def test_reply_borrowing_is_identical_under_both_settings():
    """reply レーンの挙動は本キーの影響を 1 件も受けない。"""
    for cap in (10, 17, 40):
        seq = ["reply"] * 200
        on = LOD.LodBudget(cap, tiers=_tiers(reply_share=0.1, life_share=0.1))
        off = LOD.LodBudget(cap, tiers=_tiers(reply_share=0.1, life_share=0.1,
                                              life_borrow=False))
        assert _drive(on, seq, step_len=len(seq)) == _drive(off, seq, step_len=len(seq))
        assert on.lane_used == off.lane_used


def test_reply_still_borrows_after_life_is_confined():
    """life を止めても reply は general の余りを借り続ける(返答保証は最優先)。"""
    b = LOD.LodBudget(10, tiers=_tiers(reply_share=0.1, life_share=0.1,
                                       life_borrow=False))
    assert b.caps == {"reply": 1, "life": 1, "general": 8}
    assert sum(1 for _ in range(100) if b.take("plan")) == 1, "life が借りている"
    n = sum(1 for _ in range(100) if b.take("reply"))
    assert n == 9, f"reply が general の余りを借りられていない(取れたのは {n})"


# =========================================================================== #
# (4) tiers OFF では 1 バイトも効かない子トグル
# =========================================================================== #
def test_key_is_inert_while_the_parent_toggle_is_off():
    """tiers.enabled=false では build_budget_cfg が None = 単一カウンタのまま。"""
    assert LOD.build_budget_cfg({"enabled": False, "life_borrow": False}) is None
    b = LOD.LodBudget(3)                           # tiers なし = 第30 の単一カウンタ
    assert b.tiers is None and b.life_borrow is True
    assert [b.take("plan") for _ in range(5)] == [True, True, True, False, False]


# =========================================================================== #
# (5) 実ラン(mock)= 既定は L1 バイト一致 / false は general へ配分が移る
# =========================================================================== #
def test_explicit_true_is_byte_identical_to_the_pure_default(tmp_path):
    """★R1: 明示 true を置いても、キーを書かない純粋既定と L1 完全一致。"""
    base = _run(tmp_path, "lb_base", **TIERS_ON)
    same = _run(tmp_path, "lb_same", **TIERS_ON,
                **{"lod.budget.tiers.life_borrow": "true"})
    assert _l1(base) == _l1(same)
    assert len(base.logger.llm_calls) == len(same.logger.llm_calls)


def test_false_moves_the_budget_from_life_to_general(tmp_path):
    """★実ラン: cap 拘束下で false にすると general の granted が増え life は減る。

    実測(60 体 × 144 step・cap 4・mock・seed 42):
        true  … general 124 / life 71(plan が general の余りを借りている)
        false … general 127 / life 70(借用が止まり自発発火側へ戻る)
    ★per-step の硬い上限は両方とも不変(used <= cap)。総 granted が同じにならないのは
      general の需要が常時あるため = 「遊んでいた枠が使われる」向きの差(cap は超えない)。
    """
    ov = {"lod.max_llm_per_step": 4, "planning.day_plan.enabled": "true",
          "observer.starvation.enabled": "true", **TIERS_ON}
    on = _run(tmp_path, "lb_on", n_steps=144, n_agents=60, **ov)
    off = _run(tmp_path, "lb_off", n_steps=144, n_agents=60, **ov,
               **{"lod.budget.tiers.life_borrow": "false"})
    a, b = _lane_granted(on), _lane_granted(off)
    assert b["general"] > a["general"], f"general が増えていない: {a} → {b}"
    assert b["life"] < a["life"], f"life が自枠へ収まっていない: {a} → {b}"
    assert b["reply"] == a["reply"], f"reply の取り分が動いた: {a} → {b}"
    for sim in (on, off):                          # cap は両方で硬い上限のまま
        assert SV.provenance(sim)["llm_budget"]["used_per_step"]["p95"] <= 4


# =========================================================================== #
# (6) レジストリ宣言
# =========================================================================== #
def test_new_toggle_is_declared_like_its_parent():
    by_id = {f.id: f for f in R.FEATURES}
    fid = "lod.budget.tiers.life_borrow"
    assert fid in by_id, f"{fid} がレジストリに無い"
    f, parent = by_id[fid], by_id["lod.budget.tiers.enabled"]
    assert f.repro_tier == parent.repro_tier == "strict"
    assert f.affects_k is True, "呼数の配分が動くので affects_k=True(親と同じ)"
    assert f.fingerprint_risk == "none"
    # 既定 true のキーは自動 OFF で**既定値へ**戻す(incidents_env.crowd.enabled と同流儀)
    assert f.off_value is True
    assert "life_borrow" in f.description or "general" in f.description
    assert f.description.strip()


def test_no_undeclared_toggles_after_this_key():
    assert R.undeclared_toggles(load_config()) == []


def test_finals_profile_value_if_present_is_a_bool():
    """本選 conf は**書いても書かなくてもよい**(値の適用は合流時の判断)。

    ★このレーンは finals へ 1 行も書かない(基底 conf の既定 true のまま = 現行挙動)。
      合流時に `life_borrow: false` を足す運用を妨げないよう、ここでは「在るなら bool」
      だけを固定する(未宣言トグル検出は test_no_undeclared_toggles_after_this_key が担う)。
    """
    fin = OmegaConf.load(_REPO / "conf" / "finals_observe.yaml")
    tiers = ((fin.get("lod", {}) or {}).get("budget", {}) or {}).get("tiers", {}) or {}
    if "life_borrow" in tiers:
        assert isinstance(bool(tiers["life_borrow"]), bool)
