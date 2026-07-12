"""アイスブレイク(実験前の初期関係形成)のオフライン生成。

主旨: 本番シミュレーションを回す前に、エージェント同士に「初対面の立ち話」を
少しだけさせておく。その結果(誰と知り合いか・どんな会話だったか)を JSON に
落とし、simulation.py が起動時に全 k 条件で**同一ファイル**を読み込む。
→ 初期関係が条件間で完全に同一になり、k(内省の書き戻し)以外の交絡を排除する。

使い方:
    python scripts/build_icebreak.py --personas data/personas_40.json \
        --out data/icebreak_40.json --similar 2 --random 1 --turns 3 \
        --model qwen3:8b --seed 42
    python scripts/build_icebreak.py --personas data/personas_40.json --mock  # Ollama不要

パイプライン:
  1. ペアリング: 骨格属性(年齢・性別・職業・来街者)+心理尺度を標準化し
     cosine 類似度 → 各人に類似上位 --similar 人 + ランダム --random 人。
     ペアは無向で dedupe(重複するランダム picks は張り直し)。seed 固定で決定論。
  2. 会話生成: 各ペアで qwen3(Ollama, format=json, think:false)が交互に
     --turns 往復(=2×turns 発話)の初対面会話。出力 JSON は parse_action で解釈、
     失敗時は定型文にフォールバック。
  3. 感情価: 各ペアの全発話の valence() 平均を記録。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from society.cognition.deliberate import parse_action  # noqa: E402
from society.lang.sentiment import valence             # noqa: E402


# ---------------------------------------------------------------- ペアリング
def _feature_matrix(personas: list[dict]) -> np.ndarray:
    """骨格属性 + 心理尺度を結合し、列ごとに標準化した特徴行列を返す。"""
    genders = sorted({p.get("gender", "?") for p in personas})
    occs = sorted({p.get("occupation", "") for p in personas})
    trait_keys = sorted({k for p in personas for k in p.get("traits", {})})
    rows = []
    for p in personas:
        row = [float(p.get("age", 0)) / 100.0]          # 年齢正規化
        row += [1.0 if p.get("gender") == g else 0.0 for g in genders]
        row += [1.0 if p.get("occupation") == o else 0.0 for o in occs]
        row.append(1.0 if p.get("visitor") else 0.0)
        traits = p.get("traits", {})
        row += [float(traits.get(k, 0.0)) for k in trait_keys]
        rows.append(row)
    mat = np.asarray(rows, dtype=float)
    mean = mat.mean(axis=0)
    std = mat.std(axis=0)
    std[std == 0.0] = 1.0                                # 定数列は標準化しない
    return (mat - mean) / std


def _cosine(mat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(mat, axis=1, keepdims=True)
    norm[norm == 0.0] = 1.0
    unit = mat / norm
    return unit @ unit.T


def compute_pairs(personas: list[dict], n_similar: int, n_random: int,
                  seed: int) -> list[dict]:
    """各人に類似上位 n_similar 人 + ランダム n_random 人。無向 dedupe・決定論。"""
    n = len(personas)
    sim = _cosine(_feature_matrix(personas))
    rng = np.random.default_rng(seed)
    seen: set[tuple[int, int]] = set()
    pairs: list[dict] = []

    def _key(i: int, j: int) -> tuple[int, int]:
        return (i, j) if i < j else (j, i)

    for i in range(n):
        # 類似上位(自分を除く)
        order = sorted((j for j in range(n) if j != i),
                       key=lambda j: (-sim[i, j], j))
        for j in order[:n_similar]:
            key = _key(i, j)
            if key not in seen:
                seen.add(key)
                pairs.append({"a": key[0], "b": key[1], "kind": "similar"})
        # ランダム(重複ペアは張り直し)
        added = 0
        guard = 0
        while added < n_random and guard < 50 * max(1, n):
            guard += 1
            j = int(rng.integers(n))
            if j == i:
                continue
            key = _key(i, j)
            if key in seen:
                continue
            seen.add(key)
            pairs.append({"a": key[0], "b": key[1], "kind": "random"})
            added += 1
    return pairs


# ---------------------------------------------------------------- 会話生成
_SITUATION = "知人の集まりで初めて会った相手と立ち話をしている。"

_MOCK_LINES = [
    "はじめまして。ここ、よく来るんですか？",
    "そうなんですね。私はこのあたりは久しぶりで。",
    "この集まり、なかなか面白い人が多いですね。",
    "確かに。またどこかで話せたらうれしいです。",
    "お仕事は何をされてるんですか？",
    "へえ、いいですね。今日は来てよかったです。",
]


def _persona_line(p: dict) -> str:
    return p.get("persona") or f"あなたは{p.get('name', '')}。"


def _build_prompt(speaker: dict, listener: dict, history: list[str]) -> str:
    lines = [
        _persona_line(speaker),
        f"状況: {_SITUATION}相手は{listener.get('name', '')}さん。",
    ]
    if history:
        lines.append("これまでの会話:\n" + "\n".join(history))
    lines.append('相手に自然に一言話す。出力は JSON 1個のみ: '
                 '{"action": "speak", "text": "話す内容"}')
    return "\n".join(lines)


def _ollama_generate(prompt: str, model: str, host: str) -> str:
    import urllib.error
    import urllib.request
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "format": "json", "think": False,
                       "options": {"temperature": 0.8, "num_predict": 200}}
                      ).encode("utf-8")
    req = urllib.request.Request(f"{host.rstrip('/')}/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("response", "")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return f"__ollama_error__: {exc}"


def _utterance(speaker: dict, listener: dict, history: list[str], turn_idx: int,
               *, mock: bool, model: str, host: str, rng) -> str:
    """1 発話を生成。JSON は parse_action で解釈、失敗時は定型文にフォールバック。"""
    if mock:
        line = _MOCK_LINES[turn_idx % len(_MOCK_LINES)]
        response = json.dumps({"action": "speak", "text": line},
                              ensure_ascii=False)
    else:
        response = _ollama_generate(_build_prompt(speaker, listener, history),
                                    model, host)
    action = parse_action(response)
    if action is not None and action.get("type") == "speak":
        return action["text"]
    return f"はじめまして、{speaker.get('name', '')}です。"   # 定型文フォールバック


def generate_conversation(a: dict, b: dict, turns: int, *, mock: bool,
                          model: str, host: str, rng) -> list[dict]:
    """a→b→a→… と交互に turns 往復(2×turns 発話)。turns の並びは (話者, 本文)。"""
    history: list[str] = []
    out: list[dict] = []
    speakers = [a, b]
    for t in range(turns * 2):
        speaker = speakers[t % 2]
        listener = speakers[(t + 1) % 2]
        text = _utterance(speaker, listener, history, t, mock=mock,
                          model=model, host=host, rng=rng)
        history.append(f"{speaker.get('name', '')}:「{text}」")
        out.append({"speaker": speaker["_id"], "text": text})
    return out


# ---------------------------------------------------------------- main
def build(personas: list[dict], *, n_similar: int, n_random: int, turns: int,
          seed: int, mock: bool, model: str, host: str) -> dict:
    for i, p in enumerate(personas):
        p["_id"] = i                                    # ペルソナ順 = agent_id
    pair_defs = compute_pairs(personas, n_similar, n_random, seed)
    rng = np.random.default_rng(seed + 1)
    pairs = []
    for pd in pair_defs:
        a, b = personas[pd["a"]], personas[pd["b"]]
        conv = generate_conversation(a, b, turns, mock=mock, model=model,
                                     host=host, rng=rng)
        vals = [valence(t["text"]) for t in conv]
        pairs.append({"a": pd["a"], "b": pd["b"], "kind": pd["kind"],
                      "turns": conv,
                      "valence": round(sum(vals) / len(vals), 4) if vals else 0.0})
    return {"meta": {"seed": seed, "n_personas": len(personas),
                     "similar": n_similar, "random": n_random, "turns": turns,
                     "model": "mock" if mock else model,
                     "n_pairs": len(pairs)},
            "pairs": pairs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--personas", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--similar", type=int, default=2)
    ap.add_argument("--random", type=int, default=1)
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    personas_path = Path(args.personas)
    if not personas_path.is_absolute():
        personas_path = REPO_ROOT / personas_path
    personas = json.loads(personas_path.read_text(encoding="utf-8"))["personas"]

    result = build(personas, n_similar=args.similar, n_random=args.random,
                   turns=args.turns, seed=args.seed, mock=args.mock,
                   model=args.model, host=args.host)

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = REPO_ROOT / "data" / f"icebreak_{len(personas)}.json"
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(f"written: {out_path}  (pairs={result['meta']['n_pairs']})")


if __name__ == "__main__":
    main()
