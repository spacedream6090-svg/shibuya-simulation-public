"""第152 小修正: `_decide_g` の同席判定を列挙から**人数**へ落としたことのピン。

背景: `scheduler._decide_g` は毎 decide で
    company = hearers_of(agent, _percept(sim), radius)
と 40m 圏を**全列挙して id 昇順に整列**していたが、その company は
  ・`if company:` → drive「company」加算
  ・`routine.decide(..., has_company=bool(company))`
の**ブール 2 箇所でしか使われていなかった**(リストを読む LLM 発火経路は別関数
`_fire_llm_g`:2375 で、こちらは不触)。250k step0 の py-spy で step 時間の 62.6%。

守るもの(検収基準の順)
  (1) ★意味論の同値: `bool(hearers_of(...)) == (count_hearers(...) > 0)` が
      全入力で成り立つ(密/疎/空/境界ちょうど/遮蔽あり/屋内外/就寝)。
      count == len は第150 で機械照合済み(tests/test_count_hearers.py)なので
      ここはその「> 0 とブールの一致」を独立に釘打ちする。
  (2) ★実ランの L1 完全一致: `scheduler.count_hearers` を**修正前の式**
      (`len(hearers_of(...))`)へ差し替えた B ランと、現行 A ランの L1 が
      1 バイトも違わない = drive 加算も has_company も分岐が同一。
      乱数 stream の消費本数も同一(rng 取得は :3065 で既に済んでおり不触)。
  (3) 回帰ガード: `_decide_g` の本文が `hearers_of(` を 1 度も呼ばない
      (= リスト構築が戻っていない)。

検証は mock のみ(実 LLM 禁止・≤144step)。
"""
from __future__ import annotations

import inspect
import json

from society.config import load_config
from society.engine import scheduler
from society.engine.simulation import Simulation
from society.world import perception as P


# --------------------------------------------------------------------------- #
# 共通ヘルパ
# --------------------------------------------------------------------------- #
class _A:
    """最小 duck-type(test_count_hearers.py / test_hearer_cap.py と同型)。"""

    __slots__ = ("id", "x", "y", "loc", "building", "floor", "sleeping")

    def __init__(self, i, x, y, *, building=None, floor=0, loc="street",
                 sleeping=False):
        self.id = int(i)
        self.x = float(x)
        self.y = float(y)
        self.loc = loc
        self.building = building
        self.floor = floor
        self.sleeping = sleeping


class _BlocksOdd:
    def blocks(self, a, b) -> bool:
        return bool((a.id + b.id) % 2)


def _cfg(name, n_steps=144, n_agents=15, **ov):
    dot = ["run.seed=42", f"run.n_agents={n_agents}", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock", "observer.snapshot_every=144"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


# --------------------------------------------------------------------------- #
# (1) 意味論の同値: bool(list) == (count > 0)
# --------------------------------------------------------------------------- #
def _cases() -> list[tuple[str, list, float]]:
    """同席あり/なしの両方を確実に含む入力集合。"""
    out: list[tuple[str, list, float]] = []
    # 密集(全員が互いの 40m 圏)
    out.append(("dense", [_A(i, i * 1.0, 0.0) for i in range(60)], 40.0))
    # 疎(誰も圏内にいない = 全員 company なし)
    out.append(("sparse", [_A(i, i * 500.0, 0.0) for i in range(30)], 40.0))
    # 単独(そもそも他個体ゼロ)
    out.append(("alone", [_A(0, 0.0, 0.0)], 40.0))
    # 境界ちょうど(距離 == 半径 → 圏内)と 1 つ外
    out.append(("edge", [_A(0, 0.0, 0.0), _A(1, 40.0, 0.0), _A(2, 40.001, 0.0)],
                40.0))
    # 就寝は聞き手にならない(相手が全員就寝 → company なし)
    out.append(("asleep", [_A(0, 0.0, 0.0)] +
                [_A(i, i * 1.0, 0.0, sleeping=True) for i in range(1, 20)], 40.0))
    # 屋内外の文脈分離(同じ座標でも建物/階が違えば聞こえない)
    out.append(("context",
                [_A(0, 0.0, 0.0, building="B1", floor=1, loc="indoor"),
                 _A(1, 1.0, 0.0, building="B1", floor=2, loc="indoor"),
                 _A(2, 1.0, 0.0, building="B2", floor=1, loc="indoor"),
                 _A(3, 1.0, 0.0)], 40.0))
    return out


def test_bool_of_hearers_equals_count_gt_zero():
    """★ `if company:` と `count > 0` は全入力で同じ真偽値(索引経路・legacy 経路)。"""
    seen_true = seen_false = 0
    for label, agents, radius in _cases():
        idx = P.build_index(agents, radius)
        for occ in (None, _BlocksOdd()):
            for speaker in agents:
                want = bool(P.hearers_of(speaker, agents, radius, occluder=occ))
                for src in (agents, idx):
                    got = P.count_hearers(speaker, src, radius, occluder=occ) > 0
                    assert got is want, (label, speaker.id, occ, src is idx)
                    # len との同値も同じ入力でついでに釘打ち(第150 の再確認)
                    assert P.count_hearers(speaker, src, radius, occluder=occ) == \
                        len(P.hearers_of(speaker, src, radius, occluder=occ))
                seen_true += int(want)
                seen_false += int(not want)
    assert seen_true > 0 and seen_false > 0, "片側しか出ていない入力集合では検算にならない"


# --------------------------------------------------------------------------- #
# (2) 実ラン(mock)の L1 完全一致: 修正前の式へ差し替えた B ランと比べる
# --------------------------------------------------------------------------- #
def test_l1_matches_pre_fix_expression(tmp_path, monkeypatch):
    """★A(現行 count_hearers)と B(修正前 len(hearers_of)) の L1 がバイト一致。

    `scheduler` 名前空間の `count_hearers` は **`_decide_g` の 1 箇所からしか**
    呼ばれていないので、ここを差し替えることは「修正前のコードを走らせる」ことと
    厳密に同じ(引数も半径も知覚ソースも呼び順もそのまま)。
    """
    sim_a = Simulation(_cfg("cmp_a"), out_dir=tmp_path / "cmp_a")
    sim_a.run()
    l1_a = _l1(sim_a)
    assert l1_a, "L1 が空のランで比べても意味がない"

    calls = {"n": 0, "with": 0, "without": 0}

    def _legacy(speaker, src, radius, occluder=None):
        calls["n"] += 1
        n = len(P.hearers_of(speaker, src, radius, occluder=occluder))
        calls["with" if n > 0 else "without"] += 1
        return n

    monkeypatch.setattr(scheduler, "count_hearers", _legacy)
    sim_b = Simulation(_cfg("cmp_b"), out_dir=tmp_path / "cmp_b")
    sim_b.run()

    assert calls["n"] > 0, "差し替えた経路が 1 度も通っていない(前提が崩れた)"
    # 片側しか出ていないと「同席あり」側の分岐(drive 加算)を 1 度も通らない = 空検算。
    assert calls["with"] > 0 and calls["without"] > 0, calls
    assert _l1(sim_b) == l1_a
    # LLM 呼数(L1b)も同一
    assert len(getattr(sim_b.logger, "llm_calls", [])) == \
        len(getattr(sim_a.logger, "llm_calls", []))


def test_decide_g_uses_count_not_list():
    """(3) 回帰ガード: `_decide_g` は同席判定でリストを作らない。"""
    src = inspect.getsource(scheduler._decide_g)
    body = "\n".join(ln for ln in src.splitlines()
                     if not ln.lstrip().startswith("#"))
    assert "count_hearers(" in body, "人数経路が消えている"
    assert "hearers_of(" not in body, "リスト構築が戻っている(40m 圏の全列挙)"
