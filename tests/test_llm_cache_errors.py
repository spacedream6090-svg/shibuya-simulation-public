"""B1(第122バッチ 2026-08-16): **LLM 障害応答をキャッシュへ書かない**。

バックエンドは D16 の流儀で例外を投げず `"__vllm_error__: ..."` / `"__fleet_error__: ..."`
を**通常の戻り値として**返す。素朴に保存すると「vLLM が落ちていた 1 分間の障害文字列」が
以後そのプロンプトの正解として `cached=True` で永久再生され、サーバが復旧しても直らない。

固定する契約は 3 つ:
  ① 障害応答は `_mem` にもファイルにも入らない(= 次回は普通に再試行される)。
  ② 正常応答は従来どおり保存・再生される(この修正で 1 ビットも変わらない)。
  ③ 旧ランの jsonl に**既に混入している**障害行はロード時に読み飛ばす(後方互換)。
返却値そのものは不変(上位の fallback 経路を壊さない)= それも本テストで固定する。
"""
from __future__ import annotations

import json

from society.llm.cache import CachedLLM, ERROR_PREFIXES, is_error_response


class _FlakyBackend:
    """1 回目だけ障害センチネル、2 回目以降は正常応答を返すスタブ。"""
    name = "flaky"

    def __init__(self, error: str = "__vllm_error__: HTTP 503 Service Unavailable"):
        self.error = error
        self.calls: list[str] = []

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        self.calls.append(rng_key)
        return self.error if len(self.calls) == 1 else f"ok:{prompt}"


class _AlwaysErrorBackend:
    name = "down"

    def __init__(self, error: str):
        self.error = error
        self.calls: list[str] = []

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        self.calls.append(rng_key)
        return self.error


def _gen(llm, prompt="p0", rng_key="plan/0/1"):
    return llm.generate(prompt, rng_key=rng_key, temperature=0.7, max_tokens=64)


# --------------------------------------------------------------------- ① 保存しない
def test_error_response_is_returned_but_not_cached(tmp_path):
    """障害応答は**そのまま返る**が、_mem にもファイルにも残らない。"""
    path = tmp_path / "llm_cache.jsonl"
    be = _FlakyBackend()
    llm = CachedLLM(be, enabled=True, path=path)

    resp, _cid, cached = _gen(llm)
    assert resp.startswith("__vllm_error__")   # 返却は不変(上位 fallback がそのまま働く)
    assert cached is False
    assert llm._mem == {}                      # メモリにも残らない
    assert not path.exists() or path.read_text(encoding="utf-8") == ""

    # 2 回目は「キャッシュ命中」せず backend を叩き直す = 復旧が効く。
    resp2, _cid2, cached2 = _gen(llm)
    assert (resp2, cached2) == ("ok:p0", False)
    assert len(be.calls) == 2
    # 正常応答は保存された(= ②の裏づけ)。
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    assert [r["response"] for r in rows] == ["ok:p0"]


def test_all_backend_error_prefixes_are_excluded(tmp_path):
    """vllm / fleet を含む全センチネル接頭辞で保存されない(判定の一点集約を固定)。"""
    assert "__vllm_error__" in ERROR_PREFIXES and "__fleet_error__" in ERROR_PREFIXES
    for i, prefix in enumerate(ERROR_PREFIXES):
        path = tmp_path / f"c{i}.jsonl"
        llm = CachedLLM(_AlwaysErrorBackend(f"{prefix}: boom"), enabled=True,
                        path=path)
        resp, _cid, _cached = _gen(llm, prompt=f"p{i}")
        assert is_error_response(resp)
        assert llm._mem == {}
        assert not path.exists() or path.read_text(encoding="utf-8") == ""


def test_generate_many_does_not_cache_errors(tmp_path):
    """generate_many(フェーズ3)も同じ判定を通る。重複キーでも KeyError にならない。"""
    path = tmp_path / "batch.jsonl"
    be = _AlwaysErrorBackend("__fleet_error__: no server available")
    llm = CachedLLM(be, enabled=True, path=path)
    reqs = [{"prompt": "same", "rng_key": f"k{i}", "temperature": 0.7,
             "max_tokens": 64} for i in range(3)]
    out = llm.generate_many(reqs, workers=1)

    assert all(r[0].startswith("__fleet_error__") for r in out)   # 返却は不変
    assert llm._mem == {}                                         # 1 件も載らない
    assert not path.exists() or path.read_text(encoding="utf-8") == ""
    assert len(be.calls) == 1        # 束内は初出を共有(落ちたサーバへ同 step 内で再送しない)

    # 次の一括発行では**再試行される**(障害が焼き付いていない)。
    llm.generate_many(reqs, workers=1)
    assert len(be.calls) == 2


# --------------------------------------------------------------------- ② 正常は不変
def test_normal_responses_cache_exactly_as_before(tmp_path):
    """正常応答の保存・再生・カウンタは従来と同一(この修正の非侵襲性)。"""
    path = tmp_path / "ok.jsonl"
    be = _AlwaysErrorBackend("resp")             # "常に同じ応答" のスタブとして流用
    llm = CachedLLM(be, enabled=True, path=path)

    first = _gen(llm)                            # 1 回目 = miss → 保存
    assert first[0] == "resp" and first[2] is False
    second = _gen(llm, rng_key="plan/0/2")       # 2 回目 = 同一 key で命中
    assert (second[0], second[2]) == ("resp", True)
    assert (len(be.calls), llm.calls, llm.hits) == (1, 2, 1)
    assert llm.skipped_error_rows == 0
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1 and rows[0]["response"] == "resp"

    # 別プロセス相当の読み直しで再生されること(backend を叩かない)。
    fresh = CachedLLM(_AlwaysErrorBackend("NEVER"), enabled=True, path=path)
    assert fresh._mem == {rows[0]["key"]: "resp"}


# --------------------------------------------------------------------- ③ 旧ファイル
def test_load_skips_error_rows_already_in_file(tmp_path):
    """旧ラン(修正前)の jsonl に混入した障害行はロード時に無視される(後方互換)。"""
    path = tmp_path / "legacy.jsonl"
    good_key = "a" * 64
    path.write_text(
        json.dumps({"key": good_key, "response": "good"}, ensure_ascii=False) + "\n"
        + json.dumps({"key": "b" * 64,
                      "response": "__vllm_error__: HTTP 500 Internal Server Error"},
                     ensure_ascii=False) + "\n"
        + json.dumps({"key": "c" * 64,
                      "response": "__fleet_error__: no server available"},
                     ensure_ascii=False) + "\n",
        encoding="utf-8")

    llm = CachedLLM(_AlwaysErrorBackend("NEVER"), enabled=True, path=path)
    assert llm._mem == {good_key: "good"}        # 正常行だけ載る
    assert llm.skipped_error_rows == 2           # 読み飛ばした件数が観測できる
