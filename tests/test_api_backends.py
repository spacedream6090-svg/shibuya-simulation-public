"""API バックエンド(OpenAI 互換 / Anthropic)の GPU・キー不要な単体検証。

urllib.request.urlopen を unittest.mock で差し替え、**実 API は絶対に呼ばない**
(課金なし・キー不要でテストが回る)。確認項目:
  ① リクエストボディの整形(model / messages / max_tokens、anthropic は temperature 無し)
  ② キー未設定時の挙動(openai_compat=Authorization ヘッダなしで送る / anthropic=呼ばずにエラー文字列)
  ③ HTTP エラー → "__api_error__" / "__anthropic_error__" 文字列(例外を投げない)
  ④ HTTP 400 → response_format を外して1度だけ再送(openai_compat)
  ⑤ name の安定性(URL 非依存)
  ⑥ エラー文字列にキーの値が絶対に含まれない(環境変数にダミーキーを入れて走査)
"""
from __future__ import annotations

import json
import urllib.error
from unittest.mock import patch

from society.llm.anthropic import AnthropicBackend
from society.llm.openai_compat import OpenAICompatBackend


# ---- urlopen の戻り(context manager + .read())スタブ ----
class _FakeResp:
    def __init__(self, payload: dict):
        self._data = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _openai_ok(content: str = '{"action": "wander"}') -> _FakeResp:
    return _FakeResp({"choices": [{"message": {"content": content}}]})


def _anthropic_ok(text: str = '{"action": "wander"}') -> _FakeResp:
    return _FakeResp({"content": [{"type": "text", "text": text}]})


def _http_error(url: str, code: int, reason: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url, code, reason, None, None)


# ============================ OpenAI 互換 ============================

def test_openai_body_and_headers(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-DUMMYKEY-openai")
    captured: dict = {}

    def fake_urlopen(req, timeout=None):
        captured["req"] = req
        return _openai_ok()

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        be = OpenAICompatBackend("qwen3:4b", base_url="http://localhost:11434/v1")
        out = be.generate("こんにちは", rng_key="deliberate/1/0",
                          temperature=0.3, max_tokens=64)

    req = captured["req"]
    body = json.loads(req.data)
    assert body["model"] == "qwen3:4b"
    assert body["messages"][0]["role"] == "user"
    assert body["max_tokens"] == 64
    assert body["response_format"] == {"type": "json_object"}
    assert req.get_header("Authorization") == "Bearer sk-DUMMYKEY-openai"
    assert req.full_url == "http://localhost:11434/v1/chat/completions"
    assert json.loads(out)["action"] == "wander"


def test_openai_empty_content_is_error(monkeypatch):
    """content が空(思考が max_tokens を消費 等)は成功扱いにせずエラー文字列で fallback へ。"""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with patch("urllib.request.urlopen", return_value=_openai_ok(content="")):
        be = OpenAICompatBackend("qwen3:4b", base_url="http://localhost:11434/v1")
        out = be.generate("こんにちは", rng_key="deliberate/1/0",
                          temperature=0.3, max_tokens=64)
    assert out.startswith("__api_error__")


def test_openai_no_key_sends_without_auth_header(monkeypatch):
    # ローカル互換サーバ(ollama 等)向け: キーが無ければヘッダを付けずに送る(呼びはする)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    captured: dict = {}

    def fake_urlopen(req, timeout=None):
        captured["req"] = req
        return _openai_ok()

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        be = OpenAICompatBackend("qwen3:4b", base_url="http://localhost:11434/v1")
        out = be.generate("x", rng_key="deliberate/1/0",
                          temperature=0.0, max_tokens=8)

    assert captured["req"].get_header("Authorization") is None
    assert json.loads(out)["action"] == "wander"


def test_openai_no_think_soft_switch(monkeypatch):
    # think=False のとき qwen 向けソフトスイッチ /no_think が末尾に付く(非 qwen では無害)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    captured: dict = {}

    def fake_urlopen(req, timeout=None):
        captured["req"] = req
        return _openai_ok()

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        be = OpenAICompatBackend("qwen3:4b", base_url="http://x/v1")
        be.generate("本文", rng_key="deliberate/1/0", temperature=0.0,
                    max_tokens=8, think=False)
        content_off = json.loads(captured["req"].data)["messages"][0]["content"]
        be.generate("本文", rng_key="deliberate/1/0", temperature=0.0,
                    max_tokens=8, think=True)
        content_on = json.loads(captured["req"].data)["messages"][0]["content"]

    assert content_off.endswith("/no_think")
    assert content_on == "本文"          # think=True では素通し


def test_openai_400_retries_without_response_format(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    seen: list[dict] = []

    def fake_urlopen(req, timeout=None):
        body = json.loads(req.data)
        seen.append(body)
        if "response_format" in body:
            raise _http_error(req.full_url, 400, "Bad Request")
        return _openai_ok()

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        be = OpenAICompatBackend("m", base_url="http://x/v1")
        out = be.generate("x", rng_key="deliberate/1/0",
                          temperature=0.0, max_tokens=8)

    assert len(seen) == 2                    # 1回目=あり(400)→ 2回目=外して再送
    assert "response_format" in seen[0]
    assert "response_format" not in seen[1]
    assert json.loads(out)["action"] == "wander"


def test_openai_http_error_returns_marker(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def fake_urlopen(req, timeout=None):
        raise _http_error(req.full_url, 500, "Server Error")

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        be = OpenAICompatBackend("m", base_url="http://x/v1")
        out = be.generate("x", rng_key="deliberate/1/0",
                          temperature=0.0, max_tokens=8)

    assert out.startswith("__api_error__")   # 例外ではなく文字列で返る


def test_openai_error_never_leaks_key(monkeypatch):
    secret = "sk-SUPERSECRET-leak-check-openai"
    monkeypatch.setenv("OPENAI_API_KEY", secret)

    def fake_urlopen(req, timeout=None):
        raise _http_error(req.full_url, 401, "Unauthorized")

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        be = OpenAICompatBackend("m", base_url="http://x/v1")
        out = be.generate("x", rng_key="deliberate/1/0",
                          temperature=0.0, max_tokens=8)

    assert out.startswith("__api_error__")
    assert secret not in out


def test_openai_name_url_independent():
    a = OpenAICompatBackend("qwen3:4b", base_url="http://a/v1")
    b = OpenAICompatBackend("qwen3:4b", base_url="http://b:8000/v1")
    assert a.name == b.name == "api/qwen3:4b"


# ============================ Anthropic ============================

def test_anthropic_no_key_returns_error_without_http(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    called = {"n": 0}

    def fake_urlopen(req, timeout=None):
        called["n"] += 1
        return _anthropic_ok()

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        be = AnthropicBackend("claude-haiku-4-5")
        out = be.generate("x", rng_key="reflect/1/0",
                          temperature=0.0, max_tokens=8)

    assert out.startswith("__anthropic_error__")
    assert "no api key" in out
    assert called["n"] == 0                  # ★HTTP を一切叩いていない


def test_anthropic_body_and_headers_no_temperature(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-DUMMY")
    captured: dict = {}

    def fake_urlopen(req, timeout=None):
        captured["req"] = req
        return _anthropic_ok()

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        be = AnthropicBackend("claude-haiku-4-5")
        out = be.generate("こんにちは", rng_key="reflect/1/0",
                          temperature=0.9, max_tokens=256)

    req = captured["req"]
    body = json.loads(req.data)
    assert body["model"] == "claude-haiku-4-5"
    assert body["max_tokens"] == 256         # ★必須パラメータ
    assert body["messages"][0]["content"] == "こんにちは"
    assert "temperature" not in body         # ★撤廃済み=送らない
    assert "seed" not in body
    assert req.get_header("X-api-key") == "sk-ant-DUMMY"
    assert req.get_header("Anthropic-version") == "2023-06-01"
    assert req.full_url == "https://api.anthropic.com/v1/messages"
    assert json.loads(out)["action"] == "wander"


def test_anthropic_http_error_returns_marker(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-DUMMY")

    def fake_urlopen(req, timeout=None):
        raise _http_error(req.full_url, 400, "Bad Request")

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        be = AnthropicBackend("claude-haiku-4-5")
        out = be.generate("x", rng_key="reflect/1/0",
                          temperature=0.0, max_tokens=8)

    assert out.startswith("__anthropic_error__")


def test_anthropic_error_never_leaks_key(monkeypatch):
    secret = "sk-ant-SUPERSECRET-leak-check"
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)

    def fake_urlopen(req, timeout=None):
        raise _http_error(req.full_url, 401, "Unauthorized")

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        be = AnthropicBackend("claude-haiku-4-5")
        out = be.generate("x", rng_key="reflect/1/0",
                          temperature=0.0, max_tokens=8)

    assert out.startswith("__anthropic_error__")
    assert secret not in out


def test_anthropic_name_url_independent():
    a = AnthropicBackend("claude-opus-4-8", base_url="https://api.anthropic.com")
    b = AnthropicBackend("claude-opus-4-8", base_url="http://proxy.local")
    assert a.name == b.name == "anthropic/claude-opus-4-8"
