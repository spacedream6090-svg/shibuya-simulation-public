"""MIX-1 混合モデル艦隊 = FleetLLM の tier 別モデル名(docs/plans/model-mix-plan.md)。

本選 GPU 構成: 会話(default)= Qwen3-8B ×5 / 思考(reflect・plan)= Qwen3-14B ×2。
FleetLLM は従来「艦隊全体で 1 モデル名」だったので、tier の値に
`{urls: [...], model: "qwen3:14b"}` の dict 形式を追加受理する。

受入基準:
  - URL リスト形式(現行 conf)= **1 バイトも変わらない**(子の model 名・cache_extra とも)
  - dict 形式 = その tier の子だけ別モデルで建ち、**実送信ボディの model** がそのモデルになる
  - default へ後退する purpose は艦隊既定モデル
  - 1 サーバ = 1 モデル違反(同一 URL × 別モデル)・urls 欠落・型違いは ValueError
  - D13: tier 別モデルを宣言したときだけ cache_extra に静的マップが乗る(決定論・キー順)
  - 本選 conf(finals_observe.yaml)と、そこへ貼るプロファイル(finals-vllm7.yaml)の
    **両方**が reflect / plan = 14B を宣言している(第138 の「貼り忘れ検出」と同じ流儀)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from society.llm.cache import CachedLLM
from society.llm.fleet import FleetLLM, normalize_tiers
from society.llm.vllm import VllmBackend

REPO_ROOT = Path(__file__).resolve().parents[1]

_8B = "qwen3:8b"
_14B = "qwen3:14b"
_CONV = ["http://localhost:8000", "http://localhost:8001"]
_THINK = ["http://localhost:8005", "http://localhost:8006"]


def _capture(monkeypatch):
    """urlopen を横取りして (URL, 生ボディ) を記録する(test_vllm_api_mode と同じ流儀)。"""
    sent: list[tuple[str, bytes]] = []

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({
                "choices": [{"text": '{"action":"wander"}',
                             "message": {"content": '{"action":"wander"}'}}]
            }).encode()

    def fake_urlopen(req, timeout=None):
        sent.append((req.full_url, req.data))
        return _Resp()

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return sent


def _mix_fleet(**kw) -> FleetLLM:
    """本選 5+2 の縮小版(会話 8B×2 / 思考 14B×2)。"""
    return FleetLLM(_CONV, _8B,
                    tiers={"reflect": {"urls": _THINK, "model": _14B},
                           "plan": {"urls": _THINK, "model": _14B},
                           "default": _CONV}, **kw)


# --------------------------------------------------------------------------- #
# (1) 現行形式(URL リスト)= バイト不変
# --------------------------------------------------------------------------- #
def test_list_form_is_unchanged():
    fleet = FleetLLM(_CONV, _8B, tiers={"reflect": [_THINK[0]], "default": _CONV})
    assert fleet.model == _8B and fleet.name == f"fleet/{_8B}"
    assert fleet.tier_models == {}                     # 宣言ゼロ
    assert fleet.cache_extra is None                   # 第138 と同一(= 従来キー)
    # tier だけに現れる URL も**艦隊既定モデル**で建つ(現行どおり)
    assert {b.model for b in fleet._backend.values()} == {_8B}
    assert fleet._backend[_THINK[0]].name == f"vllm/{_8B}"


def test_list_form_cache_extra_matches_batch138_with_chat():
    """api_mode / format のノブと合成しても、tier 宣言が無ければ第138 の値そのもの。"""
    assert FleetLLM(_CONV, _8B, api_mode="chat",
                    tiers={"default": _CONV}).cache_extra == {"api": "chat"}
    assert FleetLLM(_CONV, _8B, format_mode="none",
                    tiers={"default": _CONV}).cache_extra == {"f": "none"}
    assert FleetLLM(_CONV, _8B, api_mode="chat", format_mode="none",
                    tiers={"default": _CONV}).cache_extra == {"f": "none",
                                                              "api": "chat"}


def test_dict_form_without_model_equals_list_form():
    """dict 形式でも model を書かなければ艦隊既定モデル = リスト形式と完全同値。"""
    a = FleetLLM(_CONV, _8B, tiers={"reflect": [_THINK[0]], "default": _CONV})
    b = FleetLLM(_CONV, _8B, tiers={"reflect": {"urls": [_THINK[0]]},
                                    "default": {"urls": _CONV}})
    assert a._tiers == b._tiers
    assert b.tier_models == {} and b.cache_extra is a.cache_extra is None
    assert {u: be.model for u, be in a._backend.items()} == \
           {u: be.model for u, be in b._backend.items()}


# --------------------------------------------------------------------------- #
# (2) dict 形式 = tier 専用モデル
# --------------------------------------------------------------------------- #
def test_tier_children_are_built_with_the_tier_model():
    fleet = _mix_fleet()
    assert fleet.model == _8B and fleet.name == f"fleet/{_8B}"   # 艦隊既定は会話側
    assert fleet.tier_models == {"reflect": _14B, "plan": _14B}
    for u in _THINK:
        assert fleet._backend[u].model == _14B
        assert fleet._backend[u].name == f"vllm/{_14B}"
    for u in _CONV:
        assert fleet._backend[u].model == _8B


@pytest.mark.parametrize("purpose,want_model,want_pool", [
    ("reflect", _14B, _THINK),
    ("plan", _14B, _THINK),
    ("deliberate", _8B, _CONV),      # tier 未定義 = default へ後退 = 艦隊既定モデル
    ("reply", _8B, _CONV),
])
def test_sent_body_model_follows_the_tier(monkeypatch, purpose, want_model,
                                          want_pool):
    sent = _capture(monkeypatch)
    fleet = _mix_fleet()
    fleet.generate("p", rng_key=f"{purpose}/7/12", temperature=0.7, max_tokens=32)
    url, raw = sent[0]
    assert any(url.startswith(u) for u in want_pool), url
    assert json.loads(raw.decode("utf-8"))["model"] == want_model


def test_tier_model_propagates_through_api_mode_and_seed(monkeypatch):
    """第138 の api_mode / β11 の request_seed は tier 専用の子にも従来どおり透過する。"""
    fleet = _mix_fleet(api_mode="chat", request_seed=42)
    assert all(b._mode == "chat" for b in fleet._backend.values())
    assert all(b.request_seed == 42 for b in fleet._backend.values())
    sent = _capture(monkeypatch)
    fleet.generate("p", rng_key="reflect/7/12", temperature=0.7, max_tokens=32,
                   think=True)
    url, raw = sent[0]
    assert url.endswith("/v1/chat/completions")
    body = json.loads(raw.decode("utf-8"))
    assert body["model"] == _14B and body["seed"] == fleet.request_seed_for(
        "reflect/7/12")


def test_sticky_routing_inside_the_think_tier(monkeypatch):
    """14B が 2 本でも sticky(同じ agent は同じサーバ)= prefix cache を壊さない。"""
    _capture(monkeypatch)
    fleet = _mix_fleet()
    first = {a: fleet._ordered(f"reflect/{a}/0")[0] for a in range(40)}
    for a in range(40):
        assert fleet._ordered(f"reflect/{a}/7")[0] == first[a]
    assert set(first.values()) == set(_THINK)      # 2 本に分散している


# --------------------------------------------------------------------------- #
# (3) 1 サーバ = 1 モデル / 形式検証
# --------------------------------------------------------------------------- #
def test_same_url_two_models_across_tiers_raises():
    with pytest.raises(ValueError, match="1 サーバ = 1 モデル"):
        FleetLLM(_CONV, _8B,
                 tiers={"reflect": {"urls": [_THINK[0]], "model": _14B},
                        "plan": {"urls": [_THINK[0]], "model": "qwen3:32b"}})


def test_url_in_servers_with_other_tier_model_raises():
    """servers(艦隊既定モデル)に書いたポートへ 14B を宣言したら起動時に落とす。"""
    with pytest.raises(ValueError, match="model.servers"):
        FleetLLM(_CONV + _THINK, _8B,
                 tiers={"reflect": {"urls": _THINK, "model": _14B},
                        "default": _CONV})


def test_same_url_same_model_is_fine():
    """同じモデル名なら重複宣言は許す(reflect と plan が 14B×2 を共有する本選構成)。"""
    fleet = _mix_fleet()
    assert fleet._tiers["reflect"] == fleet._tiers["plan"] == _THINK


@pytest.mark.parametrize("bad", [
    {"reflect": {"model": _14B}},                        # urls 欠落
    {"reflect": {"url": _THINK, "model": _14B}},         # 誤記(黙って既定へ落とさない)
    {"reflect": {"urls": _THINK, "models": _14B}},       # 誤記
    {"reflect": {"urls": "http://localhost:8005"}},      # urls が文字列
    {"reflect": {"urls": 8005}},                         # urls が数値
    {"reflect": {"urls": _THINK, "model": ""}},          # 空モデル名
    {"reflect": {"urls": _THINK, "model": 14}},          # モデル名が非文字列
    {"reflect": {"urls": [], "model": _14B}},            # model はあるが行き先が無い
    {"reflect": "http://localhost:8005"},                # tier 値が文字列
    {"reflect": 8005},                                   # tier 値が数値
])
def test_malformed_tiers_raise_value_error(bad):
    with pytest.raises(ValueError):
        FleetLLM(_CONV, _8B, tiers=bad)


def test_normalize_tiers_is_pure():
    assert normalize_tiers(None) == {}
    assert normalize_tiers({}) == {}
    assert normalize_tiers({"reflect": ["http://a:1/"]}) == {"reflect": (["http://a:1"], None)}
    assert normalize_tiers({"reflect": {"urls": ["http://a:1/"], "model": _14B}}) == \
        {"reflect": (["http://a:1"], _14B)}
    assert normalize_tiers({"reflect": {"urls": ["http://a:1"]}}) == \
        {"reflect": (["http://a:1"], None)}


# --------------------------------------------------------------------------- #
# (4) キャッシュキー(D13)
# --------------------------------------------------------------------------- #
def test_cache_extra_carries_the_static_tier_map():
    fleet = _mix_fleet()
    assert fleet.cache_extra == {"tiers": {"plan": _14B, "reflect": _14B}}
    # キー順は sorted で決定論(宣言順が違っても同じマップ・同じ JSON バイト列)
    other = FleetLLM(_CONV, _8B,
                     tiers={"default": _CONV,
                            "plan": {"urls": _THINK, "model": _14B},
                            "reflect": {"urls": _THINK, "model": _14B}})
    assert other.cache_extra == fleet.cache_extra
    assert list(other.cache_extra["tiers"]) == list(fleet.cache_extra["tiers"]) \
        == ["plan", "reflect"]


def test_cache_extra_composes_with_api_mode():
    fleet = _mix_fleet(api_mode="chat")
    assert fleet.cache_extra == {"api": "chat",
                                 "tiers": {"plan": _14B, "reflect": _14B}}


def test_cache_key_differs_only_when_a_tier_model_is_declared():
    plain = FleetLLM(_CONV, _8B, tiers={"reflect": _THINK, "default": _CONV})
    mixed = _mix_fleet()
    k_plain = CachedLLM(plain)._key("p", 0.7, 32, False)
    k_mixed = CachedLLM(mixed)._key("p", 0.7, 32, False)
    assert k_plain != k_mixed, "tier 別モデルのランが 8B 時代の応答を誤再生しうる"
    # 現行形式のキーは MIX-1 導入前と 1 バイトも変わらない(過去ランの llm_cache 再生互換)
    blob = json.dumps({"model": f"fleet/{_8B}", "t": 0.7, "m": 32, "think": False,
                       "p": "p"}, ensure_ascii=False, sort_keys=True)
    assert k_plain == hashlib.sha256(blob.encode("utf-8")).hexdigest()
    # 単一 URL 直結(VllmBackend)とも従来どおり別キー(name が違うため)
    assert k_plain != CachedLLM(VllmBackend(_8B))._key("p", 0.7, 32, False)


# --------------------------------------------------------------------------- #
# (5) 本選 conf(貼り忘れ検出)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rel", ["conf/finals_observe.yaml",
                                 "conf/profiles/finals-vllm7.yaml"])
def test_finals_confs_declare_the_14b_think_tiers(rel):
    """本選 conf と、そこへ貼るプロファイルの**両方**が reflect=14B / plan=艦隊既定(8B)を宣言。

    片方だけだと「貼った側の model: ブロックに無い」= 黙って全部 8B へ戻る
    (plan_max_tokens / api_mode: chat と同じ落とし穴)。
    第159(意図的変更への追随): plan ティアは撤去= plan は default(8B×5)へ後退するのが正。
    根拠=A8 実測で 8B/14B parse 同点・14B は plan_max_tokens 打ち切りで 6.56% 全損・
    8B plan 型 9.9 呼/s(壁時計 4.4x)。誤って plan=14B を書き戻すとここが落ちる(双方向ピン)。
    """
    model = OmegaConf.to_container(OmegaConf.load(REPO_ROOT / rel).model,
                                   resolve=True)
    tiers = model.get("tiers")
    assert tiers, f"{rel} に model.tiers が無い"
    assert isinstance(tiers.get("reflect"), dict), f"{rel}: reflect が dict 形式でない"
    assert tiers["reflect"]["model"] == _14B, f"{rel}: reflect が 14B でない"
    assert len(tiers["reflect"]["urls"]) == 2, f"{rel}: reflect が 2 本でない"
    assert "plan" not in tiers, \
        f"{rel}: plan ティアが復活している(第159 で撤去済み=plan は艦隊既定 8B。戻すなら理由を再記録)"
    assert len(tiers["default"]) == 5, f"{rel}: 会話プールが 5 本でない"
    assert not set(tiers["default"]) & set(tiers["reflect"]["urls"]), \
        f"{rel}: 会話と思考が同じポートを共有している(1 サーバ = 1 モデル違反)"


def test_finals_profile_builds_a_valid_mixed_fleet():
    """finals-vllm7.yaml の model: ブロックが**実際に** FleetLLM を建てられる(配線の実地確認)。

    VllmBackend の構築は HTTP を張らないので GPU も vLLM も不要。conf とコードの規約が
    ずれたら(例: 14B ポートを servers に書き戻す)ここで ValueError になる。
    """
    model = OmegaConf.to_container(
        OmegaConf.load(REPO_ROOT / "conf/profiles/finals-vllm7.yaml").model,
        resolve=True)
    fleet = FleetLLM([str(s) for s in model["servers"]], str(model["name"]),
                     tiers=model["tiers"], api_mode=str(model["api_mode"]))
    assert len(model["servers"]) == 5 and fleet.model == _8B
    # 第159: plan ティア撤去(plan は艦隊既定 8B へ)に追随。reflect のみ 14B。
    assert fleet.tier_models == {"reflect": _14B}
    assert len(fleet._backend) == 7                     # 5 + 2 = 7 GPU 分の子
    assert sum(1 for b in fleet._backend.values() if b.model == _14B) == 2
    assert all(fleet._backend[u].model == _14B
               for u in fleet._tiers["reflect"])
    assert all(fleet._backend[u].model == _8B for u in fleet._default)
