#!/usr/bin/env python
"""A8 ミニ行動トーナメント — 同一シナリオ束を 2 モデルへ撃って**行動品質**を比べる。

判断 A8(docs/plans/external-audit-triage.md)= 本選のモデル構成
  S0 = 7×Qwen3-8B 維持   vs   S1 = 6×8B + 1×14B(reflect/deep を 14B へ・RouterLLM は実装済み)
の材料を作るためのハーネス。**このスクリプトは判定しない**(数字を並べるだけ。
どこで線を引くかは実験者の判断)。

三部構成
--------
1. シナリオ束(決定論)
   mock バックエンドの小さなシムを 1 本回し、その **LLM ジャーナル**
   (`llm_journal.jsonl.gz` = プロンプト全文 + 送出パラメータ)から実プロンプトを採る。
   ★合成プロンプトを書き起こさない: シムの実プロンプトビルダー
   (`cognition/deliberate.build_prompt` → planning / reflection)が組んだ**そのもの**を撃つ。
   ハーネスは engine/cognition を 1 行も変更せず、import すらしない(読むのはジャーナル)。
   purpose(deliberate / plan / reflect)別に、時刻帯 6 区分 × エージェントで
   ラウンドロビンに間引く(乱数なし = 同じ seed・同じ conf なら同じ束)。

2. 発行
   既存の `VllmBackend` + `CachedLLM.generate_many(workers=N)` をそのまま使う
   (**新しい並列機構は書かない**)。request-level seed は既存の
   `vllm.stable_request_seed`(blake2b)= A 側と B 側で同じ seed が同じシナリオに乗る。
   レイテンシは薄いデコレータ(`TimedBackend`)で 1 呼ごとに測る(生成は委譲するだけ)。

3. 採点(機械指標のみ・LLM 審判は使わない)
   JSON パース成功率 / `parse_action` 成功率 / `classify_reject` 内訳 / 空応答率 /
   応答長分布 / 行動分布(A vs B の Jensen–Shannon 距離)/ レイテンシ p50・p95。
   すべて purpose 別内訳つき。採点器は cognition の**本物**(parse_action / classify_reject)。

出力(`runs/tournament_<name>/`)
    report.md        比較表 + 判定材料の要約
    results.parquet  1 行 = 1 シナリオ × 1 モデル(応答本文つき)
    scenarios.jsonl  シナリオ束(プロンプト全文 + 送出パラメータ)
    samples.md       人手確認用: purpose 別に先頭 10 件の prompt / response 対(A と B を並記)
    manifest.json    引数・束ハッシュ・所要秒・件数

使い方
------
    # 束だけ作る(GPU 不要・mock のみ)
    python scripts/behavior_tournament.py --name a8_smoke --build-only

    # 本番: 8B(port 8000)vs 14B(port 8006)へ同じ束を撃つ
    python scripts/behavior_tournament.py --name a8 \\
        --endpoint-a http://localhost:8000 --model-a Qwen/Qwen3-8B \\
        --endpoint-b http://localhost:8006 --model-b Qwen/Qwen3-14B \\
        --per-purpose 100 --workers 16

★実 LLM を叩くのは 2 と 3 だけで、シム本体には一切書き戻さない(scripts/ 配下の下流ツール)。
★エンジン(src/society/engine・cognition)は読むだけ・conf 既定値は触らない。
★シナリオ文はこのハーネスが 1 文字も**書き起こさない**(シムが組んだ架空世界のプロンプトを
  そのまま運ぶだけ)。人物名も店名もシム側の名簿・地図に由来する。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import statistics
import sys
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

for _s in (sys.stdout, sys.stderr):        # Windows コンソール(cp932)対策(house style)
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                      # noqa: BLE001
        pass

from society.cognition.deliberate import classify_reject, parse_action   # noqa: E402
from society.llm.cache import CachedLLM                                  # noqa: E402
from society.llm.journal import iter_records                             # noqa: E402

SCHEMA = "behavior_tournament/1"
#: 既定で採るシナリオの purpose(rng_key の先頭セグメント)。
DEFAULT_PURPOSES = ("deliberate", "plan", "reflect")
#: 時刻帯の層(1 日を何等分して層別サンプルするか)。
N_TOD_BUCKETS = 6
#: HTTP バックエンドの失敗は例外でなく本文で返る(llm/vllm.py D16 の流儀)。
ERROR_PREFIXES = ("__vllm_error__", "__ollama_error__", "__api_error__")
#: parse_action が None のときの行動分布ラベル(棄却も分布の一部として比べる)。
REJECT_LABEL = "__reject__"


# =========================================================================== #
# 0. 小道具(純関数)
# =========================================================================== #
def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def parse_hhmm(value: str) -> int:
    """"HH:MM" → 分 of day。壊れていたら 0(捏造せず素直に落とさない)。"""
    try:
        hh, mm = str(value).strip().split(":")
        return (int(hh) * 60 + int(mm)) % 1440
    except (ValueError, AttributeError):
        return 0


def split_rng_key(rng_key: str) -> tuple[str, str, str]:
    """"purpose/agent_id/step[/ordinal]" → (purpose, agent_id, step)。欠けた欄は空文字。"""
    parts = str(rng_key).split("/")
    return (parts[0] if parts else "",
            parts[1] if len(parts) > 1 else "",
            parts[2] if len(parts) > 2 else "")


def percentile(values: list[float], q: float) -> float:
    """nearest-rank の分位点(numpy 不要・空列は 0.0)。q は 0..1。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1,
                     int(math.ceil(q * len(ordered))) - 1))
    return float(ordered[idx])


def jsd(p: dict[str, float], q: dict[str, float]) -> float:
    """Jensen–Shannon divergence(底 2・0..1)。空側は 0.0 を返す。

    行動分布の比較用。値が 0 に近いほど「同じような行動の出方」。
    """
    keys = sorted(set(p) | set(q))
    tot_p = float(sum(p.get(k, 0.0) for k in keys))
    tot_q = float(sum(q.get(k, 0.0) for k in keys))
    if tot_p <= 0 or tot_q <= 0:
        return 0.0
    out = 0.0
    for k in keys:
        pi = p.get(k, 0.0) / tot_p
        qi = q.get(k, 0.0) / tot_q
        mi = 0.5 * (pi + qi)
        if pi > 0:
            out += 0.5 * pi * math.log2(pi / mi)
        if qi > 0:
            out += 0.5 * qi * math.log2(qi / mi)
    return max(0.0, min(1.0, out))


def is_error_response(text: str) -> bool:
    return isinstance(text, str) and text.startswith(ERROR_PREFIXES)


# =========================================================================== #
# 1. シナリオ束(決定論)
# =========================================================================== #
def _request_fingerprint(rec: dict) -> str:
    """CachedLLM のキーと同じ材料(プロンプト + 送出パラメータ)の指紋。

    これで重複を落としておくと、同じ束を 2 度撃たない = キャッシュ有無で
    レイテンシ分布が歪まない。
    """
    blob = json.dumps({"p": rec.get("prompt", ""),
                       "t": float(rec.get("temperature", 0.0)),
                       "m": int(rec.get("max_tokens", 0)),
                       "think": bool(rec.get("think", False))},
                      ensure_ascii=False, sort_keys=True)
    return sha256_text(blob)


def select_scenarios(records, *, purposes=DEFAULT_PURPOSES, per_purpose: int = 100,
                     start_min: int = 0, dt_min: int = 10,
                     n_buckets: int = N_TOD_BUCKETS) -> list[dict]:
    """ジャーナルのレコード列 → シナリオ束(決定論・乱数ゼロ)。

    層別のとり方:
      1. purpose(deliberate / plan / reflect)ごとに独立に per_purpose 件まで採る。
      2. 各 purpose を「時刻帯 n_buckets 区分 × エージェント」のセルに割り、
         セルを (セル内順位, 時刻帯, エージェント) 順に並べてラウンドロビンで 1 件ずつ引く。
         → 先頭から採っても時刻帯とエージェントが散る(= 場所・drive・関係文脈も散る)。
      3. 完全に同じ要求(プロンプト + パラメータ)は最初の 1 件だけ残す。

    入力レコードは llm/journal.py の行(seq / rng_key / prompt / temperature /
    max_tokens / think)。順序は入力順(ジャーナル順 = シムの発行順)に依存し、
    それ自体が seed と conf の決定論的な関数なので、束も決定論になる。
    """
    wanted = tuple(purposes)
    per_bucket_min = 1440 // max(1, n_buckets)
    seen: set[str] = set()
    # purpose -> (bucket, agent) -> [item, ...]
    cells: dict[str, dict[tuple[int, str], list[dict]]] = {p: {} for p in wanted}
    for rec in records:
        purpose, agent_id, step_s = split_rng_key(rec.get("rng_key", ""))
        if purpose not in cells:
            continue
        fp = _request_fingerprint(rec)
        if fp in seen:
            continue
        seen.add(fp)
        try:
            step = int(step_s)
        except (TypeError, ValueError):
            step = 0
        tod_min = (int(start_min) + step * int(dt_min)) % 1440
        bucket = min(n_buckets - 1, tod_min // per_bucket_min)
        item = {
            "purpose": purpose,
            "agent_id": agent_id,
            "step": step,
            "tod_min": int(tod_min),
            "bucket": int(bucket),
            "rng_key": str(rec.get("rng_key", "")),
            "prompt": rec.get("prompt", ""),
            "temperature": float(rec.get("temperature", 0.0)),
            "max_tokens": int(rec.get("max_tokens", 0)),
            "think": bool(rec.get("think", False)),
            "prompt_sha256": sha256_text(rec.get("prompt", "")),
            "prompt_chars": len(rec.get("prompt", "") or ""),
        }
        cells[purpose].setdefault((bucket, agent_id), []).append(item)

    out: list[dict] = []
    for purpose in wanted:
        by_cell = cells[purpose]
        if not by_cell:
            continue
        # セル順: 時刻帯内での「その時刻帯の何人目か」→ 時刻帯 → エージェント。
        # これで最初の一巡が全時刻帯をまたぐ(先頭だけ採っても層が偏らない)。
        # ★時刻帯ごとに開始位置を名簿の 1/n_buckets ずつ回す: 回さないと「どの時刻帯でも
        #   同じ先頭エージェント」が並び、束が少数の個体に偏る(回す前の実測=20 件が 8 人に集中)。
        rank: dict[tuple[int, str], int] = {}
        by_bucket: dict[int, list[str]] = defaultdict(list)
        for bucket, agent_id in by_cell:
            by_bucket[bucket].append(agent_id)
        for bucket, agents in by_bucket.items():
            ordered = sorted(agents)
            off = (bucket * len(ordered)) // max(1, n_buckets)
            for i, agent_id in enumerate(ordered[off:] + ordered[:off]):
                rank[(bucket, agent_id)] = i
        order = sorted(by_cell, key=lambda c: (rank[c], c[0], c[1]))
        picked: list[dict] = []
        depth = 0
        deepest = max(len(v) for v in by_cell.values())
        while depth < deepest and len(picked) < per_purpose:
            for cell in order:
                queue = by_cell[cell]
                if depth < len(queue):
                    picked.append(queue[depth])
                    if len(picked) >= per_purpose:
                        break
            depth += 1
        for i, item in enumerate(picked):
            item = dict(item)
            item["scenario_id"] = f"{purpose}-{i:04d}"
            out.append(item)
    return out


def bundle_hash(scenarios: list[dict]) -> str:
    """束の指紋(順序込み)。A 側と B 側が同じ束を撃ったことの機械的な証拠。"""
    h = hashlib.sha256()
    for s in scenarios:
        h.update(s["scenario_id"].encode("utf-8"))
        h.update(b"\x1f")
        h.update(s["prompt_sha256"].encode("utf-8"))
        h.update(b"\x1f")
        h.update(f"{s['temperature']}/{s['max_tokens']}/{int(s['think'])}"
                 .encode("utf-8"))
        h.update(b"\x1e")
    return h.hexdigest()


def build_bundle(src_dir: Path, *, seed: int, n_agents: int, n_steps: int,
                 personas: str | None, profile: str | None, env: str | None,
                 sets: list[str], purposes, per_purpose: int,
                 rebuild: bool = False, quiet: bool = True) -> tuple[list[dict], dict]:
    """mock シムを 1 本回して束を作る(GPU 不要・ネットワーク不要)。

    シムは `runs/tournament_<name>/scenario_src/` に普通のランとして出力される
    (= 束の来歴が丸ごと残る)。既に journal があればそれを再利用する(--rebuild-src で作り直し)。
    """
    from society.config import load_config
    from society.engine.simulation import Simulation

    journals = sorted(src_dir.glob("llm_journal*.jsonl.gz"))
    if rebuild and src_dir.exists():
        shutil.rmtree(src_dir)
        journals = []
    overrides = [f"run.seed={seed}", "run.seed_auto=false",
                 f"run.n_agents={n_agents}", f"run.n_steps={n_steps}",
                 "run.name=scenario_src", "model.backend=mock",
                 "model.journal=true"]
    if personas:
        overrides.append(f"agents.personas_file={personas}")
    overrides += list(sets or [])
    cfg = load_config(overrides=overrides,
                      profile=Path(profile) if profile else None,
                      env=Path(env) if env else None)
    dt_min = int(cfg.run.get("dt_min", 10) or 10)
    start_min = parse_hhmm(cfg.run.get("start_tod", "07:00"))
    t0 = time.perf_counter()
    if not journals:
        sim = Simulation(cfg, out_dir=src_dir)
        summary = sim.run()
        journals = sorted(src_dir.glob("llm_journal*.jsonl.gz"))
        if not quiet:
            print(f"[bundle] mock sim: {summary.get('llm_calls', 0)} 呼 / "
                  f"{time.perf_counter() - t0:.1f}s")
    elif not quiet:
        print(f"[bundle] 既存のジャーナルを再利用: {[j.name for j in journals]}")
    if not journals:
        raise RuntimeError(
            f"{src_dir} に llm_journal*.jsonl.gz が無い(model.journal=true を確認)。")

    def _records():
        for path in journals:
            yield from iter_records(path)

    scenarios = select_scenarios(_records(), purposes=purposes,
                                 per_purpose=per_purpose,
                                 start_min=start_min, dt_min=dt_min)
    meta = {"src_dir": str(src_dir), "seed": seed, "n_agents": n_agents,
            "n_steps": n_steps, "dt_min": dt_min, "start_tod_min": start_min,
            "personas": personas, "profile": profile, "env": env,
            "sets": list(sets or []),
            "build_s": round(time.perf_counter() - t0, 2)}
    return scenarios, meta


def write_scenarios(path: Path, scenarios: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for s in scenarios:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def read_scenarios(path: Path) -> list[dict]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


# =========================================================================== #
# 2. 発行(既存の CachedLLM.generate_many を流用。並列機構は書かない)
# =========================================================================== #
class TimedBackend:
    """レイテンシ計測だけを足す薄いデコレータ(生成そのものは内側へ委譲)。

    CachedLLM が読む属性(name / cache_extra / request_seed_for)を素通しするので、
    キャッシュキーも request seed も**素の backend を使ったときと同一**になる。
    """

    def __init__(self, inner):
        self.inner = inner
        self.name = getattr(inner, "name", "unknown")
        self.cache_extra = getattr(inner, "cache_extra", None)
        self._lock = threading.Lock()
        self.samples: dict[tuple[str, str], float] = {}

    def request_seed_for(self, rng_key: str):
        fn = getattr(self.inner, "request_seed_for", None)
        return fn(rng_key) if fn is not None else None

    def generate(self, prompt: str, *, rng_key: str, temperature: float,
                 max_tokens: int, think: bool = False) -> str:
        t0 = time.perf_counter()
        try:
            return self.inner.generate(prompt, rng_key=rng_key,
                                       temperature=temperature,
                                       max_tokens=max_tokens, think=think)
        finally:
            dt = time.perf_counter() - t0
            with self._lock:
                self.samples[(rng_key, sha256_text(prompt)[:16])] = dt

    def latency_of(self, rng_key: str, prompt_sha256: str) -> float | None:
        return self.samples.get((rng_key, (prompt_sha256 or "")[:16]))


def build_vllm_backend(*, model: str, endpoint: str, timeout_s: float,
                       deadline_s: float, format_mode: str, seed: int | None):
    """本選と同じ `VllmBackend`(OpenAI 互換 HTTP)。seed は既存の blake2b 導出に委ねる。"""
    from society.llm.vllm import VllmBackend
    return VllmBackend(model, base_url=endpoint, timeout_s=timeout_s,
                       format_mode=format_mode, deadline_s=deadline_s,
                       request_seed=seed)


def score_response(response: str) -> dict:
    """1 応答の機械採点(副作用ゼロ・決定論)。採点器は cognition の本物を使う。"""
    text = response if isinstance(response, str) else ""
    error = is_error_response(text)
    stripped = text.strip()
    action = parse_action(text)
    if action is not None:
        return {"resp_chars": len(text), "error": error, "empty": False,
                "json_ok": True, "parse_ok": True,
                "action_type": str(action.get("type", "")),
                "reject_reason": "", "unknown_action": ""}
    reason, info = classify_reject(text)
    return {"resp_chars": len(text), "error": error,
            "empty": (not stripped) and not error,
            "json_ok": reason != "broken_json", "parse_ok": False,
            "action_type": REJECT_LABEL, "reject_reason": reason,
            "unknown_action": str(info.get("action", ""))}


def issue(scenarios: list[dict], backend, *, side: str, model: str,
          endpoint: str, workers: int = 1,
          cache_path: Path | None = None) -> tuple[list[dict], float]:
    """束を 1 モデルへ撃って採点行を返す。(rows, wall_s)。

    backend は `LLMBackend` 互換なら何でもよい(テストは缶詰応答の fake を渡す)。
    並列は `CachedLLM.generate_many(workers=N)` に完全に任せる。
    """
    timed = TimedBackend(backend)
    llm = CachedLLM(timed, enabled=bool(cache_path),
                    path=Path(cache_path) if cache_path else None)
    requests = [{"prompt": s["prompt"], "rng_key": s["rng_key"],
                 "temperature": float(s["temperature"]),
                 "max_tokens": int(s["max_tokens"]),
                 "think": bool(s["think"])} for s in scenarios]
    t0 = time.perf_counter()
    results = llm.generate_many(requests, workers=int(workers))
    wall = time.perf_counter() - t0
    rows: list[dict] = []
    for s, (response, call_id, cached) in zip(scenarios, results):
        score = score_response(response)
        latency = timed.latency_of(s["rng_key"], s["prompt_sha256"])
        rows.append({
            "side": side, "model": model, "endpoint": endpoint,
            "scenario_id": s["scenario_id"], "purpose": s["purpose"],
            "agent_id": str(s["agent_id"]), "step": int(s["step"]),
            "tod_min": int(s.get("tod_min", 0)), "bucket": int(s.get("bucket", 0)),
            "rng_key": s["rng_key"], "prompt_sha256": s["prompt_sha256"],
            "prompt_chars": int(s["prompt_chars"]),
            "temperature": float(s["temperature"]),
            "max_tokens": int(s["max_tokens"]), "think": bool(s["think"]),
            "call_id": str(call_id), "cached": bool(cached),
            "latency_s": float(latency) if latency is not None else float("nan"),
            "response": response if isinstance(response, str) else "",
            **score,
        })
    return rows, wall


# =========================================================================== #
# 3. 採点の集計
# =========================================================================== #
def aggregate(rows: list[dict]) -> dict:
    """(side, purpose) と (side, "ALL") の指標。rate は 0..1。"""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["side"], row["purpose"])].append(row)
        groups[(row["side"], "ALL")].append(row)
    out: dict[str, dict] = {}
    for (side, purpose), items in sorted(groups.items()):
        n = len(items)
        lat = [r["latency_s"] for r in items
               if isinstance(r["latency_s"], float) and not math.isnan(r["latency_s"])]
        lens = [int(r["resp_chars"]) for r in items]
        rejects = Counter(r["reject_reason"] for r in items if r["reject_reason"])
        actions = Counter(r["action_type"] for r in items)
        out[f"{side}|{purpose}"] = {
            "side": side, "purpose": purpose, "n": n,
            "error_rate": _rate(sum(1 for r in items if r["error"]), n),
            "empty_rate": _rate(sum(1 for r in items if r["empty"]), n),
            "json_rate": _rate(sum(1 for r in items if r["json_ok"]), n),
            "parse_rate": _rate(sum(1 for r in items if r["parse_ok"]), n),
            "cached_rate": _rate(sum(1 for r in items if r["cached"]), n),
            "reject": dict(sorted(rejects.items())),
            "actions": dict(sorted(actions.items())),
            "len_mean": round(statistics.fmean(lens), 1) if lens else 0.0,
            "len_p50": percentile([float(x) for x in lens], 0.50),
            "len_p95": percentile([float(x) for x in lens], 0.95),
            "lat_n": len(lat),
            "lat_mean": round(statistics.fmean(lat), 3) if lat else 0.0,
            "lat_p50": round(percentile(lat, 0.50), 3),
            "lat_p95": round(percentile(lat, 0.95), 3),
        }
    return out


def _rate(num: int, den: int) -> float:
    return round(num / den, 4) if den else 0.0


def action_jsd(stats: dict, purpose: str) -> float:
    a = stats.get(f"A|{purpose}", {}).get("actions", {})
    b = stats.get(f"B|{purpose}", {}).get("actions", {})
    return round(jsd({k: float(v) for k, v in a.items()},
                     {k: float(v) for k, v in b.items()}), 4)


# =========================================================================== #
# 4. 出力
# =========================================================================== #
def _pct(x: float) -> str:
    return f"{100.0 * float(x):.1f}%"


def _delta(a: float, b: float, *, pct: bool = True) -> str:
    d = float(b) - float(a)
    sign = "+" if d >= 0 else "−"
    return f"{sign}{abs(100.0 * d):.1f}pt" if pct else f"{sign}{abs(d):.2f}"


def render_report(*, stats: dict, sides: dict, purposes: list[str],
                  scenarios: list[dict], meta: dict, walls: dict) -> str:
    """report.md の本文(比較表 + 判定材料の要約)。**判定はしない**。"""
    lines: list[str] = []
    has_b = "B" in sides
    lines.append(f"# A8 ミニ行動トーナメント — {meta.get('name', '')}")
    lines.append("")
    lines.append(f"- 生成: {meta.get('created', '')}(schema `{SCHEMA}`)")
    lines.append(f"- シナリオ束: **{len(scenarios)} 件** / bundle_sha256 "
                 f"`{meta.get('bundle_sha256', '')[:16]}`")
    for side, info in sorted(sides.items()):
        lines.append(f"- {side}: model=`{info['model']}` endpoint=`{info['endpoint']}` "
                     f"wall={walls.get(side, 0.0):.1f}s")
    lines.append(f"- workers={meta.get('workers')} / request_seed={meta.get('seed')} "
                 f"/ format={meta.get('format')}")
    lines.append("")
    lines.append("> このレポートは **数字を並べるだけ**で、S0/S1 の採否は判定しない。"
                 "LLM 審判は使っていない(全指標が機械採点)。")
    lines.append("")

    # ---- 総括 ----
    lines.append("## 1. 総括(全 purpose)")
    lines.append("")
    header = ("| 指標 | A | B | 差(B−A) |" if has_b else "| 指標 | A |")
    sep = ("|---|---:|---:|---:|" if has_b else "|---|---:|")
    lines += [header, sep]
    a = stats.get("A|ALL", {})
    b = stats.get("B|ALL", {})
    rate_rows = [("parse_action 成功率", "parse_rate"),
                 ("JSON パース成功率", "json_rate"),
                 ("空応答率", "empty_rate"),
                 ("通信エラー率", "error_rate")]
    for label, key in rate_rows:
        if has_b:
            lines.append(f"| {label} | {_pct(a.get(key, 0))} | {_pct(b.get(key, 0))} "
                         f"| {_delta(a.get(key, 0), b.get(key, 0))} |")
        else:
            lines.append(f"| {label} | {_pct(a.get(key, 0))} |")
    num_rows = [("応答長 p50(字)", "len_p50"), ("応答長 p95(字)", "len_p95"),
                ("レイテンシ p50(秒)", "lat_p50"), ("レイテンシ p95(秒)", "lat_p95"),
                ("レイテンシ平均(秒)", "lat_mean")]
    for label, key in num_rows:
        if has_b:
            lines.append(f"| {label} | {a.get(key, 0)} | {b.get(key, 0)} "
                         f"| {_delta(a.get(key, 0), b.get(key, 0), pct=False)} |")
        else:
            lines.append(f"| {label} | {a.get(key, 0)} |")
    lines.append("")

    # ---- purpose 別 ----
    lines.append("## 2. purpose 別内訳")
    lines.append("")
    lines.append("| purpose | side | n | parse | json | 空 | err | len p50 | len p95 "
                 "| lat p50 | lat p95 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for purpose in purposes + ["ALL"]:
        for side in sorted(sides):
            st = stats.get(f"{side}|{purpose}")
            if not st:
                continue
            lines.append(
                f"| {purpose} | {side} | {st['n']} | {_pct(st['parse_rate'])} "
                f"| {_pct(st['json_rate'])} | {_pct(st['empty_rate'])} "
                f"| {_pct(st['error_rate'])} | {st['len_p50']:.0f} | {st['len_p95']:.0f} "
                f"| {st['lat_p50']} | {st['lat_p95']} |")
    lines.append("")

    # ---- 失敗の内訳 ----
    lines.append("## 3. 失敗の内訳(classify_reject)")
    lines.append("")
    lines.append("| purpose | side | broken_json | missing_field | unknown_action |")
    lines.append("|---|---|---:|---:|---:|")
    for purpose in purposes + ["ALL"]:
        for side in sorted(sides):
            st = stats.get(f"{side}|{purpose}")
            if not st:
                continue
            rj = st["reject"]
            lines.append(f"| {purpose} | {side} | {rj.get('broken_json', 0)} "
                         f"| {rj.get('missing_field', 0)} "
                         f"| {rj.get('unknown_action', 0)} |")
    lines.append("")

    # ---- 行動分布 ----
    lines.append("## 4. 行動分布(parse_action の type。棄却は "
                 f"`{REJECT_LABEL}`)")
    lines.append("")
    if has_b:
        lines.append("| purpose | JSD(底2・0=同一) |")
        lines.append("|---|---:|")
        for purpose in purposes + ["ALL"]:
            if f"A|{purpose}" in stats and f"B|{purpose}" in stats:
                lines.append(f"| {purpose} | {action_jsd(stats, purpose):.4f} |")
        lines.append("")
    types = sorted({t for side in sides
                    for t in stats.get(f"{side}|ALL", {}).get("actions", {})})
    lines.append("| action type | " + " | ".join(f"{s} 件" for s in sorted(sides))
                 + " |")
    lines.append("|---|" + "---:|" * len(sides))
    for t in types:
        cells = []
        for side in sorted(sides):
            st = stats.get(f"{side}|ALL", {})
            cnt = st.get("actions", {}).get(t, 0)
            cells.append(f"{cnt}({_pct(_rate(cnt, st.get('n', 0)))})")
        lines.append(f"| `{t}` | " + " | ".join(cells) + " |")
    lines.append("")

    # ---- 判定材料 ----
    lines.append("## 5. 判定材料の要約")
    lines.append("")
    if has_b:
        pa, pb = a.get("parse_rate", 0), b.get("parse_rate", 0)
        ja, jb = a.get("json_rate", 0), b.get("json_rate", 0)
        ratio = (f"{b.get('lat_p50', 0) / a.get('lat_p50', 0):.2f} 倍"
                 if a.get("lat_p50", 0) else "測定不能(A 側の p50 が 0 秒)")
        lines.append(f"- **行動の成立(parse_action)**: A {_pct(pa)} → B {_pct(pb)}"
                     f"({_delta(pa, pb)})")
        lines.append(f"- **形式の成立(JSON)**: A {_pct(ja)} → B {_pct(jb)}"
                     f"({_delta(ja, jb)})")
        lines.append(f"- **速度**: p50 は B/A = {ratio}"
                     f"(A {a.get('lat_p50', 0)}s / B {b.get('lat_p50', 0)}s)。"
                     " 本選のスループット余裕と突き合わせること。")
        lines.append(f"- **行動の出方**: 全体 JSD = {action_jsd(stats, 'ALL'):.4f}"
                     "(0 に近いほど『同じような行動の出方』= 置き換えても世界の動きは似る)。")
        lines.append(f"- 内訳で差が最も大きい purpose: "
                     f"{_max_gap_purpose(stats, purposes)}")
        lines.append("")
        half = _diff_halfwidth(pa, a.get("n", 1), pb, b.get("n", 1))
        lines.append(f"  参考: 束 {len(scenarios)} 件では、parse 成功率の差の 95% 目安幅は "
                     f"±{half:.1f}pt(二項の Wald 近似。両側とも失敗 0 件のときは"
                     " rule of three で下限を置く)。**これより小さい差はこの束では"
                     "区別できない**ので、線を引くならまず件数を増やすこと。")
    else:
        lines.append("- B 側が無い(単独計測)。差分の判定材料は次回 --endpoint-b を付けて取る。")
    lines.append("")
    lines.append("## 6. 限界(正直な宣言)")
    lines.append("")
    lines.append("- 機械指標のみ。「返答が気が利いているか」「内省が深いか」は測っていない"
                 "(人手確認は `samples.md`)。")
    lines.append("- シナリオ束は mock ラン由来。プロンプトの**形**は実物と同一だが、"
                 "本選の 25 万体ランで出る文脈分布そのものではない。")
    lines.append("- 層別したのは **purpose × 時刻帯 × 個体** まで。場所・drive・関係文脈は"
                 "その帰結として散っているだけで、直接は制御していない。")
    lines.append("- レイテンシは 1 モデルずつ順に計測した値。両モデルを同時に載せた"
                 "ときの相互干渉は含まない。")
    lines.append("- 応答は vLLM のサンプリング(temperature>0)を含むので、"
                 "同じ束でも再走で数字は多少動く(request seed は固定してある)。")
    lines.append("")
    return "\n".join(lines)


def _max_gap_purpose(stats: dict, purposes: list[str]) -> str:
    best, best_gap = "—", -1.0
    for purpose in purposes:
        a = stats.get(f"A|{purpose}")
        b = stats.get(f"B|{purpose}")
        if not a or not b:
            continue
        gap = abs(b["parse_rate"] - a["parse_rate"])
        if gap > best_gap:
            best, best_gap = purpose, gap
    return "—" if best_gap < 0 else f"`{best}`(parse 差 {best_gap * 100:.1f}pt)"


def _diff_halfwidth(p_a: float, n_a: int, p_b: float, n_b: int) -> float:
    """2 比率の差の 95% 目安幅(pt)。

    両側とも 0/1 に張り付くと Wald 幅は 0 になり「差が無い=有意」と読めてしまうので、
    観測ゼロ件の上限(rule of three: 3/n)を下限として置く。
    """
    n_a, n_b = max(1, int(n_a)), max(1, int(n_b))
    p_a = min(max(float(p_a), 0.0), 1.0)
    p_b = min(max(float(p_b), 0.0), 1.0)
    wald = 100.0 * 1.96 * math.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    return max(wald, 100.0 * 3.0 / min(n_a, n_b))


def render_samples(scenarios: list[dict], rows: list[dict], *, per_purpose: int = 10,
                   sides: list[str] | None = None) -> str:
    """人手確認用: purpose 別に先頭 N 件の prompt / response 対(A・B 並記)。"""
    by_id: dict[tuple[str, str], dict] = {(r["side"], r["scenario_id"]): r for r in rows}
    sides = sides or sorted({r["side"] for r in rows})
    out = ["# 人手確認サンプル", "",
           f"purpose 別に先頭 {per_purpose} 件。プロンプトは**シムの実プロンプト全文**。", ""]
    seen: Counter = Counter()
    for s in scenarios:
        if seen[s["purpose"]] >= per_purpose:
            continue
        seen[s["purpose"]] += 1
        out.append(f"## {s['scenario_id']} "
                   f"(agent={s['agent_id']} step={s['step']} "
                   f"tod={s['tod_min'] // 60:02d}:{s['tod_min'] % 60:02d} "
                   f"think={s['think']} max_tokens={s['max_tokens']})")
        out.append("")
        out.append("### prompt")
        out.append("")
        out.append("```")
        out.append(s["prompt"])
        out.append("```")
        out.append("")
        for side in sides:
            row = by_id.get((side, s["scenario_id"]))
            if row is None:
                continue
            tag = ("parse_ok" if row["parse_ok"]
                   else f"REJECT:{row['reject_reason']}")
            out.append(f"### {side} — `{row['model']}` [{tag} / "
                       f"{row['resp_chars']}字 / {row['latency_s']:.2f}s]")
            out.append("")
            out.append("```")
            out.append(row["response"])
            out.append("```")
            out.append("")
    return "\n".join(out)


_PARQUET_COLS = (
    ("side", "string"), ("model", "string"), ("endpoint", "string"),
    ("scenario_id", "string"), ("purpose", "string"), ("agent_id", "string"),
    ("step", "int64"), ("tod_min", "int64"), ("bucket", "int64"),
    ("rng_key", "string"), ("prompt_sha256", "string"), ("prompt_chars", "int64"),
    ("temperature", "double"), ("max_tokens", "int64"), ("think", "bool"),
    ("call_id", "string"), ("cached", "bool"), ("latency_s", "double"),
    ("resp_chars", "int64"), ("error", "bool"), ("empty", "bool"),
    ("json_ok", "bool"), ("parse_ok", "bool"), ("action_type", "string"),
    ("reject_reason", "string"), ("unknown_action", "string"),
    ("response", "string"),
)


def write_parquet(path: Path, rows: list[dict]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    types = {"string": pa.string, "int64": pa.int64,
             "double": pa.float64, "bool": pa.bool_}
    schema = pa.schema([(name, types[typ]()) for name, typ in _PARQUET_COLS])
    cols = {name: [r.get(name) for r in rows] for name, _ in _PARQUET_COLS}
    pq.write_table(pa.table(cols, schema=schema), path)


# =========================================================================== #
# 5. CLI
# =========================================================================== #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="A8 ミニ行動トーナメント(8B vs 14B の行動品質比較ハーネス)")
    ap.add_argument("--name", default="a8", help="出力先 runs/tournament_<name>/")
    ap.add_argument("--out-dir", default="runs", help="出力の親ディレクトリ(既定 runs)")
    # ---- シナリオ束 ----
    ap.add_argument("--scenarios", default=None,
                    help="既存の scenarios.jsonl を使う(束を作り直さない)")
    ap.add_argument("--build-only", action="store_true",
                    help="束だけ作って終了(GPU 不要)")
    ap.add_argument("--rebuild-src", action="store_true",
                    help="scenario_src/ を消してから mock シムを回し直す")
    ap.add_argument("--scenario-seed", type=int, default=20260817)
    ap.add_argument("--n-agents", type=int, default=60)
    ap.add_argument("--n-steps", type=int, default=432, help="144 step = シミュ内1日")
    ap.add_argument("--personas", default="data/personas_60.json",
                    help="agents.personas_file(空文字で手続き生成)")
    ap.add_argument("--profile", default=None, help="conf/*.yaml(用途プロファイル)")
    ap.add_argument("--env", default=None, help="EnvPack(env/<place>)")
    ap.add_argument("--set", dest="sets", action="append", default=[],
                    help="conf の dotlist 上書き(繰り返し可)")
    ap.add_argument("--purposes", default=",".join(DEFAULT_PURPOSES))
    ap.add_argument("--per-purpose", type=int, default=100,
                    help="purpose ごとのシナリオ数(既定 100 × 3 = 300 件)")
    ap.add_argument("--limit", type=int, default=0,
                    help=">0 なら束の先頭 N 件だけ撃つ(スモーク用)")
    # ---- 発行 ----
    ap.add_argument("--endpoint-a", default=None, help="例 http://localhost:8000")
    ap.add_argument("--model-a", default=None, help="例 Qwen/Qwen3-8B")
    ap.add_argument("--endpoint-b", default=None)
    ap.add_argument("--model-b", default=None)
    ap.add_argument("--workers", type=int, default=8, help="同時発行数(継続バッチング充填)")
    ap.add_argument("--seed", type=int, default=20260817,
                    help="request-level seed の run_seed(blake2b 安定ハッシュの材料)")
    ap.add_argument("--timeout-s", type=float, default=120.0)
    ap.add_argument("--deadline-s", type=float, default=300.0)
    ap.add_argument("--format", default="json", choices=["json", "none"],
                    help="model.format 相当(json=response_format を送る)")
    ap.add_argument("--cache", action="store_true",
                    help="応答キャッシュを使う(中断からの再開用。レイテンシは歪む)")
    ap.add_argument("--samples", type=int, default=10,
                    help="samples.md に載せる purpose あたりの件数")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    purposes = [p.strip() for p in str(args.purposes).split(",") if p.strip()]
    out_dir = Path(args.out_dir) / f"tournament_{args.name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = out_dir / "scenarios.jsonl"

    # ---- 1. 束 ----
    src_meta: dict = {}
    if args.scenarios:
        scenarios = read_scenarios(Path(args.scenarios))
        src_meta = {"reused_from": str(args.scenarios)}
        print(f"[bundle] 既存の束を読み込み: {args.scenarios}({len(scenarios)} 件)")
    elif bundle_path.exists() and not args.rebuild_src:
        scenarios = read_scenarios(bundle_path)
        src_meta = {"reused_from": str(bundle_path)}
        print(f"[bundle] 既存の束を再利用: {bundle_path}({len(scenarios)} 件)")
    else:
        scenarios, src_meta = build_bundle(
            out_dir / "scenario_src", seed=args.scenario_seed,
            n_agents=args.n_agents, n_steps=args.n_steps,
            personas=args.personas or None, profile=args.profile, env=args.env,
            sets=args.sets, purposes=purposes, per_purpose=args.per_purpose,
            rebuild=args.rebuild_src, quiet=False)
        write_scenarios(bundle_path, scenarios)
    if args.limit and args.limit > 0:
        scenarios = scenarios[:args.limit]
    if not scenarios:
        print("[error] シナリオが 0 件(purpose 名か束の生成条件を見直すこと)")
        return 2
    counts = Counter(s["purpose"] for s in scenarios)
    digest = bundle_hash(scenarios)
    print(f"[bundle] {len(scenarios)} 件 {dict(sorted(counts.items()))} "
          f"sha256={digest[:16]}")
    if args.build_only:
        (out_dir / "manifest.json").write_text(json.dumps(
            {"schema": SCHEMA, "name": args.name, "build_only": True,
             "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "bundle_sha256": digest, "counts": dict(sorted(counts.items())),
             "src": src_meta}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[done] 束のみ: {bundle_path}")
        return 0

    # ---- 2. 発行 ----
    if not (args.endpoint_a and args.model_a):
        print("[error] --endpoint-a と --model-a は必須(束だけなら --build-only)")
        return 2
    sides: dict[str, dict] = {"A": {"model": args.model_a, "endpoint": args.endpoint_a}}
    if args.endpoint_b and args.model_b:
        sides["B"] = {"model": args.model_b, "endpoint": args.endpoint_b}
    rows: list[dict] = []
    walls: dict[str, float] = {}
    for side in sorted(sides):
        info = sides[side]
        backend = build_vllm_backend(
            model=info["model"], endpoint=info["endpoint"],
            timeout_s=args.timeout_s, deadline_s=args.deadline_s,
            format_mode=args.format, seed=args.seed)
        cache_path = (out_dir / f"llm_cache.{side}.jsonl") if args.cache else None
        print(f"[issue] {side}: {info['model']} @ {info['endpoint']} "
              f"({len(scenarios)} 件 / workers={args.workers})")
        side_rows, wall = issue(scenarios, backend, side=side, model=info["model"],
                                endpoint=info["endpoint"], workers=args.workers,
                                cache_path=cache_path)
        rows += side_rows
        walls[side] = wall
        ok = sum(1 for r in side_rows if r["parse_ok"])
        print(f"[issue] {side}: wall={wall:.1f}s parse_ok={ok}/{len(side_rows)}")

    # ---- 3. 採点と出力 ----
    stats = aggregate(rows)
    meta = {"schema": SCHEMA, "name": args.name,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "bundle_sha256": digest, "counts": dict(sorted(counts.items())),
            "workers": args.workers, "seed": args.seed, "format": args.format,
            "cache": bool(args.cache), "src": src_meta,
            "walls_s": {k: round(v, 2) for k, v in walls.items()}}
    report = render_report(stats=stats, sides=sides, purposes=purposes,
                           scenarios=scenarios, meta=meta, walls=walls)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    (out_dir / "samples.md").write_text(
        render_samples(scenarios, rows, per_purpose=args.samples,
                       sides=sorted(sides)), encoding="utf-8")
    write_parquet(out_dir / "results.parquet", rows)
    (out_dir / "manifest.json").write_text(
        json.dumps({**meta, "stats": stats}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"[done] {out_dir}/report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
