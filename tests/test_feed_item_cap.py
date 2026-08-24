# -*- coding: utf-8 -*-
"""第157補: タイムライン1件の本文截断(prompts.feed_item_max_chars)。

背景(finals 実機 2026-08-24): RT 連鎖(「RT @A: RT @B: …」)は深さ無上限で、
1 件が 16,189 字に達して vLLM のコンテキスト上限(8,192tok)を超過 → HTTP 400 →
発話呼が欠落した。件数は lod.input_res の feed_n が絞るが、1 件の**長さ**は
`scheduler._feed_texts` が唯一の口。既定 0=無制限=現行バイト一致。
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from society.engine import scheduler as sched


class _Cfg(dict):
    """OmegaConf 風に属性アクセスも通す最小スタブ。"""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            return None


def _sim(cap=None, posts=()):
    prompts = {} if cap is None else {"feed_item_max_chars": cap}
    author = SimpleNamespace(name="山田太郎")
    sim = SimpleNamespace(
        cfg=_Cfg(prompts=_Cfg(prompts)),
        agent_by_id={7: author},
    )
    return sim, list(posts)


def _run(monkeypatch, sim, posts):
    monkeypatch.setattr(sched.infoenv_mod, "timeline",
                        lambda s, a, st, sm: posts)
    monkeypatch.setattr(sched.ablate_mod, "propagation_off", lambda s: False)
    return sched._feed_texts(sim, SimpleNamespace(id=1), 0, 0)


def test_default_zero_is_byte_identical(monkeypatch):
    long = "RT @a: " * 3000                      # 21,000 字級の RT 連鎖
    sim, posts = _sim(cap=None, posts=[{"author": 7, "text": long}])
    out = _run(monkeypatch, sim, posts)
    assert out == [f"@山田太郎: {long}"], "既定 0 で 1 バイトでも変えてはならない"


def test_cap_truncates_with_ellipsis(monkeypatch):
    long = "あ" * 500
    sim, posts = _sim(cap=140, posts=[{"author": 7, "text": long}])
    out = _run(monkeypatch, sim, posts)
    assert out == [f"@山田太郎: {'あ' * 140}…"]
    assert len(out[0]) == len("@山田太郎: ") + 141


def test_cap_boundary_exact_length_untouched(monkeypatch):
    text = "い" * 140
    sim, posts = _sim(cap=140, posts=[{"author": 7, "text": text}])
    out = _run(monkeypatch, sim, posts)
    assert out == [f"@山田太郎: {text}"], "ちょうど cap の長さは切らない(…も付けない)"


def test_short_text_and_official_author_untouched(monkeypatch):
    sim, posts = _sim(cap=140, posts=[{"author": -1, "text": "短い告知"}])
    out = _run(monkeypatch, sim, posts)
    assert out == ["@公式: 短い告知"]


def test_registry_declares_the_key():
    from society import registry as R
    f = R.BY_ID.get("prompts.feed_item_max_chars")
    assert f is not None, "prompts.feed_item_max_chars がレジストリ未宣言"
    assert f.repro_tier == "strict"
    assert f.fingerprint_risk == "possible"


def test_base_conf_default_is_zero():
    from society.config import load_config
    cfg = load_config([])
    assert int(cfg.prompts.feed_item_max_chars) == 0, \
        "基底 conf の既定は 0(無制限=現行バイト一致)でなければならない"
