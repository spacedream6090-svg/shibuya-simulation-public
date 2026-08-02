"""モデル呼び出しアダプタ(ollama / openai互換 / プラセボ)。

src/society/llm/ との関係(★src は読むだけ・書き換えない):
  - OpenAI 互換は `society.llm.openai_compat.OpenAICompatBackend` を **継承して再利用**
    する。API キーを self に載せず環境変数から都度読む規律をそのまま引き継ぐため。
  - Ollama は **継承せず自前 HTTP** で叩く。理由は `OllamaBackend.generate` が
    `options.seed` を送らないから(バッテリーの前提「同一シード」が成立しない)。
    src を編集できない本バッチでは、ここに seed 付きの呼び出しを置くのが唯一の解。
    送信内容は src/society/llm/ollama.py と同型(model/prompt/stream/think/options)で、
    options に seed を足しただけ。

エラー規約(D16 と同じ): 例外を投げず "__<backend>_error__: ..." を返す。
上位(harness)はそれを error として記録し、指標の母数から外す。
★API キーの値はコード・ログ・エラー文字列に一切出さない。
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from society.llm.openai_compat import OpenAICompatBackend   # noqa: E402

_SLUG_BAD = re.compile(r"[^0-9A-Za-z._-]+")


def slugify(name: str) -> str:
    """モデル名 → ディレクトリ名に使える安全な文字列。"""
    return _SLUG_BAD.sub("_", name).strip("_") or "model"


class BatteryClient:
    """バッテリー用の最小インタフェース。"""

    backend: str = "base"
    model: str = ""
    name: str = "base"

    @property
    def slug(self) -> str:
        return slugify(self.name)

    def describe(self) -> dict:
        """来歴(manifest)に載せるモデル情報。"""
        return {"backend": self.backend, "model": self.model, "name": self.name}

    def generate(self, prompt: str, *, temperature: float, max_tokens: int,
                 seed: int, json_mode: bool) -> str:
        raise NotImplementedError

    def unload(self) -> bool:
        """モデルを常駐から降ろす(対応しないバックエンドは False)。"""
        return False


# ---------------------------------------------------------------- Ollama
class OllamaClient(BatteryClient):
    backend = "ollama"

    def __init__(self, model: str, host: str = "http://localhost:11434",
                 timeout_s: float = 300.0):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout_s = float(timeout_s)
        self.name = f"ollama/{model}"

    def describe(self) -> dict:
        d = super().describe()
        d["host"] = self.host
        return d

    def _payload(self, prompt: str, temperature: float, max_tokens: int,
                 seed: int, json_mode: bool) -> dict:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "think": False,     # 思考モードは予算を食い空応答を招く(src の実測注記)
            "options": {
                "temperature": float(temperature),
                "num_predict": int(max_tokens),
                "seed": int(seed),          # ★ここが src の OllamaBackend との差分
            },
        }
        if json_mode:
            payload["format"] = "json"
        return payload

    def generate(self, prompt: str, *, temperature: float, max_tokens: int,
                 seed: int, json_mode: bool) -> str:
        body = json.dumps(self._payload(prompt, temperature, max_tokens, seed,
                                        json_mode)).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/generate", data=body,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError,
                json.JSONDecodeError) as exc:
            return f"__ollama_error__: {exc}"
        text = data.get("response", "")
        if not str(text).strip():
            return "__ollama_error__: empty response"
        return text

    def unload(self) -> bool:
        """keep_alive=0 で VRAM から降ろす。

        ★これが無いと次のモデルが VRAM に収まらず部分 CPU オフロードになる。
        実測(RTX 5070 12GB): qwen3:4b が常駐したまま qwen3:8b を呼ぶと
        48%/52% CPU/GPU 分割になり、1呼 7s → 100s(約14倍)まで落ちた。
        バッテリーは必ずモデルの切れ目でこれを呼ぶ。
        """
        body = json.dumps({"model": self.model, "keep_alive": 0}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/generate", data=body,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60.0) as resp:
                resp.read()
            return True
        except (urllib.error.URLError, TimeoutError, OSError):
            return False


# ---------------------------------------------------------------- OpenAI 互換
class OpenAICompatClient(OpenAICompatBackend, BatteryClient):
    """src のバックエンドを継承。seed と per-call の JSON 強制だけ足す。"""

    backend = "openai"

    def __init__(self, model: str, base_url: str = "http://localhost:11434/v1",
                 api_key_env: str = "OPENAI_API_KEY", timeout_s: float = 300.0):
        OpenAICompatBackend.__init__(self, model, base_url=base_url,
                                     api_key_env=api_key_env, timeout_s=timeout_s)
        # name は親が f"api/{model}" を設定済み(URL 非依存=キャッシュキー安定の流儀)

    def describe(self) -> dict:
        return {"backend": self.backend, "model": self.model, "name": self.name,
                "base_url": self.base_url, "api_key_env": self.api_key_env}

    def generate(self, prompt: str, *, temperature: float, max_tokens: int,
                 seed: int, json_mode: bool) -> str:
        body = self._body(prompt, temperature, max_tokens, think=False,
                          json_fmt=json_mode)
        body["seed"] = int(seed)
        try:
            text = self._extract(self._post("/chat/completions", body))
        except urllib.error.HTTPError as exc:
            return f"__api_error__: HTTP {exc.code} {exc.reason}"
        except (urllib.error.URLError, TimeoutError, OSError,
                json.JSONDecodeError, KeyError, IndexError) as exc:
            return f"__api_error__: {exc}"
        if not str(text).strip():
            return "__api_error__: empty content"
        return text


# ---------------------------------------------------------------- プラセボ
_PLACEBO_SCHEDULE = [
    {"start": "00:00", "end": "07:00", "activity": "睡眠"},
    {"start": "07:00", "end": "07:30", "activity": "身の回りの用事"},
    {"start": "07:30", "end": "08:00", "activity": "食事"},
    {"start": "08:00", "end": "09:00", "activity": "通勤・通学"},
    {"start": "09:00", "end": "12:00", "activity": "仕事"},
    {"start": "12:00", "end": "13:00", "activity": "食事"},
    {"start": "13:00", "end": "18:00", "activity": "仕事"},
    {"start": "18:00", "end": "19:00", "activity": "通勤・通学"},
    {"start": "19:00", "end": "20:00", "activity": "食事"},
    {"start": "20:00", "end": "23:00", "activity": "テレビ・ラジオ・新聞・雑誌"},
    {"start": "23:00", "end": "24:00", "activity": "睡眠"},
]
_PLACEBO_UTTERANCES = (
    "そうですね、たしかにそう思います。",
    "なるほど、それはいいですね。",
    "そうですね、私も同じように感じます。",
)
_PLACEBO_PLAN = [
    {"when": "09:00", "what": "出かける"},
    {"when": "12:00", "what": "食事をする"},
    {"when": "18:00", "what": "帰る"},
]


class PlaceboClient(BatteryClient):
    """テンプレート応答のルールベース対照。

    役割は**テスト自体の健全性確認**(正典 §4)。スキーマは必ず満たすが、
    ペルソナ・日種別・摂動・シードのいずれにも反応しない。したがって
    「分散」「ペルソナ感度」「多様性」を測る指標では必ず底に沈むはずである。
    沈まなければ、その指標が何も測れていないということになる。

    藁人形にしないための配慮: JSON は壊さない/日本語として自然/会話は3種を巡回
    (単一文字列固定ではない)。それでも沈むことに意味がある。
    """

    backend = "placebo"

    def __init__(self, variant: str = "template"):
        self.model = variant
        self.name = f"placebo/{variant}"

    def generate(self, prompt: str, *, temperature: float, max_tokens: int,
                 seed: int, json_mode: bool) -> str:
        if '"blocks"' in prompt:
            return json.dumps({"blocks": _PLACEBO_SCHEDULE}, ensure_ascii=False)
        if '"change"' in prompt:
            return json.dumps({"change": False, "new_activity": "仕事",
                               "delta_minutes": 0,
                               "reason": "予定どおりにします。"},
                              ensure_ascii=False)
        if '"choice"' in prompt:
            return json.dumps({"choice": "カフェに入る",
                               "why": "落ち着けるからです。"}, ensure_ascii=False)
        if '"plan"' in prompt:
            return json.dumps({"day": 1, "plan": _PLACEBO_PLAN,
                               "mood": "ふつうです"}, ensure_ascii=False)
        # 発話(C 層・E 層の独り言): 3種を巡回する定型を JSON 封筒に入れる
        idx = int(hashlib.sha256(str(seed).encode()).hexdigest()[:8], 16)
        text = _PLACEBO_UTTERANCES[idx % len(_PLACEBO_UTTERANCES)]
        if '"say"' in prompt:
            return json.dumps({"say": text}, ensure_ascii=False)
        return text


# ---------------------------------------------------------------- spec 解析
def parse_model_spec(spec: str) -> BatteryClient:
    """'<backend>:<model>[|k=v|k=v]' を解析してクライアントを作る。

    例:
      ollama:qwen3:4b
      ollama:qwen3:8b|host=http://localhost:11434
      openai:gpt-x|base_url=https://api.openai.com/v1|api_key_env=OPENAI_API_KEY
      placebo:template
    """
    parts = spec.split("|")
    head = parts[0]
    opts: dict[str, str] = {}
    for kv in parts[1:]:
        if "=" not in kv:
            raise ValueError(f"オプションは k=v 形式: {kv!r}")
        k, v = kv.split("=", 1)
        opts[k.strip()] = v.strip()
    if ":" not in head:
        raise ValueError(f"モデル指定は '<backend>:<model>': {spec!r}")
    backend, model = head.split(":", 1)
    backend = backend.strip().lower()
    model = model.strip()
    if not model:
        raise ValueError(f"モデル名が空: {spec!r}")
    if backend == "ollama":
        return OllamaClient(model, host=opts.get("host", "http://localhost:11434"),
                            timeout_s=float(opts.get("timeout_s", 300.0)))
    if backend in ("openai", "openai_compat"):
        return OpenAICompatClient(
            model, base_url=opts.get("base_url", "http://localhost:11434/v1"),
            api_key_env=opts.get("api_key_env", "OPENAI_API_KEY"),
            timeout_s=float(opts.get("timeout_s", 300.0)))
    if backend == "placebo":
        return PlaceboClient(model)
    raise ValueError(f"未知のバックエンド: {backend!r}"
                     "(ollama / openai / placebo)")


def is_error(text: str) -> bool:
    """バックエンドのエラー規約文字列か。"""
    return bool(text) and text.startswith("__") and "_error__" in text[:32]
