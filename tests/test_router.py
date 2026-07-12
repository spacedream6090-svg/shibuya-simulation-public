"""合成ルータ RouterLLM(M2)の GPU・キー不要な単体検証。

mock の子(呼び出しを記録する薄いオブジェクト)を使い、実 LLM/実 API は一切呼ばない。確認項目:
  ① purpose 別 dispatch の決定論(rng_key 先頭で子を選ぶ)
  ② 未知 purpose は default へフォールバック
  ③ "default" 欠落 → ValueError
  ④ "agent_tier" キー → ValueError(M3 未実装ガード)
  ⑤ 委譲が呼数・引数・戻り値を変えない(子の呼び出し記録で検証)
  ⑥ calls / hits 集計(同一子を複数 purpose に載せても二重計上しない=id() 重複排除)
"""
from __future__ import annotations

import pytest

from society.llm.router import RouterLLM


class _RecChild:
    """呼び出しを記録する子。calls/hits も持ち、集計テストの対象になる。"""
    def __init__(self, tag: str, calls: int = 0, hits: int = 0):
        self.name = f"child/{tag}"
        self.tag = tag
        self.log: list[dict] = []
        self.calls = calls
        self.hits = hits

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        self.log.append({"prompt": prompt, "rng_key": rng_key,
                         "temperature": temperature, "max_tokens": max_tokens,
                         "think": think})
        return f"resp:{self.tag}"


# ---- ① purpose 別 dispatch の決定論 ----
def test_dispatch_by_purpose_is_deterministic():
    d = _RecChild("default")
    r = _RecChild("reflect")
    router = RouterLLM({"default": d, "reflect": r})

    for _ in range(5):                       # 何度呼んでも同じ purpose は同じ子へ
        assert router.generate("p", rng_key="reflect/7/2",
                               temperature=0.0, max_tokens=8) == "resp:reflect"
        assert router.generate("p", rng_key="deliberate/3/0",
                               temperature=0.0, max_tokens=8) == "resp:default"
    assert len(r.log) == 5
    assert len(d.log) == 5


# ---- ② 未知 purpose は default へ ----
def test_unknown_purpose_falls_back_to_default():
    d = _RecChild("default")
    r = _RecChild("reflect")
    router = RouterLLM({"default": d, "reflect": r})

    router.generate("p", rng_key="plan/1/0", temperature=0.0, max_tokens=8)
    assert len(d.log) == 1
    assert len(r.log) == 0


# ---- ③ "default" 欠落 → ValueError ----
def test_missing_default_raises():
    with pytest.raises(ValueError):
        RouterLLM({"reflect": _RecChild("r")})


# ---- ④ "agent_tier" キー → ValueError ----
def test_agent_tier_key_raises():
    with pytest.raises(ValueError):
        RouterLLM({"default": _RecChild("d"), "agent_tier": _RecChild("t")})


# ---- ⑤ 委譲が呼数・引数・戻り値を変えない ----
def test_delegation_preserves_call_and_args():
    d = _RecChild("default")
    router = RouterLLM({"default": d})

    out = router.generate("プロンプト", rng_key="deliberate/9/1",
                          temperature=0.7, max_tokens=42, think=True)
    assert out == "resp:default"             # 戻り値をそのまま透過
    assert d.log == [{"prompt": "プロンプト", "rng_key": "deliberate/9/1",
                      "temperature": 0.7, "max_tokens": 42, "think": True}]


def test_delegation_passes_tuple_through():
    # 子が CachedLLM のように (response, call_id, cached) を返しても router は型に関知しない
    class _TupleChild:
        name = "child/tuple"
        def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
            return ("応答", "abc123", True)

    router = RouterLLM({"default": _TupleChild()})
    assert router.generate("p", rng_key="deliberate/1/0",
                           temperature=0.0, max_tokens=8) == ("応答", "abc123", True)


# ---- ⑥ calls / hits 集計(id() 重複排除)----
def test_calls_hits_sum_over_unique_children():
    d = _RecChild("default", calls=10, hits=4)
    r = _RecChild("reflect", calls=3, hits=1)
    router = RouterLLM({"default": d, "reflect": r})
    assert router.calls == 13
    assert router.hits == 5


def test_calls_hits_no_double_count_for_shared_child():
    # 同一子オブジェクトを複数 purpose に載せても 1回だけ数える
    shared = _RecChild("shared", calls=7, hits=2)
    other = _RecChild("other", calls=5, hits=1)
    router = RouterLLM({"default": shared, "reflect": shared, "plan": other})
    assert router.calls == 12                # 7(shared・1回)+ 5(other)
    assert router.hits == 3                   # 2 + 1


def test_calls_hits_default_zero_for_children_without_counters():
    class _Bare:
        name = "child/bare"
        def generate(self, *a, **k):
            return "x"

    router = RouterLLM({"default": _Bare()})
    assert router.calls == 0
    assert router.hits == 0
