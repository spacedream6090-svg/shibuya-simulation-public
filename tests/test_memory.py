"""記憶 v2(Phase B)のユニットテスト。"""
from __future__ import annotations

from society.agents.memory import MemoryStore


def test_retrieve_prefers_relevance_and_importance():
    m = MemoryStore()
    m.observe(0, "駅前でコーヒーを飲んだ")
    m.consolidate(10, "初日", [("宮下公園で花音と長話した", 8.0),
                               ("道玄坂の混雑にうんざりした", 6.0)])
    # 文脈=宮下公園 → relevance が効いて公園の記憶が先頭に来る
    got = m.retrieve(20, ["宮下公園", "花音"], n=2)
    assert got and "宮下公園" in got[0]


def test_consolidate_caps_and_diary():
    m = MemoryStore(store_cap=4)
    for day in range(3):
        for i in range(4):
            m.observe(day * 144 + i, f"d{day} 出来事{i}")
        m.consolidate(day * 144 + 100, f"day{day} の要約",
                      [(f"d{day} 顕著{i}", 5.0 + i) for i in range(3)])
    assert len(m.episodes) <= 4                    # 上限で忘却
    assert m.day_summaries[-1] == "day2 の要約"    # 日記は最新が末尾
    assert len(m.day_summaries) <= 7


def test_relation_line():
    m = MemoryStore()
    for s in range(3):
        m.record_contact(7, "佐藤蓮", s, "やあ")
    assert "佐藤蓮とは3回" in (m.relation_line([7]) or "")
    assert m.relation_line([99]) is None
