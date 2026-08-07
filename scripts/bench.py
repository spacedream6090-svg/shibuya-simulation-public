#!/usr/bin/env python
"""スケーリング・ベンチ + マルチモデル LOD 計測(M4)。

3つのモードを持つ(既定は従来のスケーリング・ベンチ=後方互換):

  # 1) スケーリング・ベンチ(従来どおり。Mock ランで step 時間・メモリを測る)
  python scripts/bench.py                               # --agents 10 40 80 --steps 60 --seed 42
  python scripts/bench.py --agents 10 50 --steps 30

  # 1') N 掃引 + 既存ランの再集計(P0バッチ 2026-07-29)
  python scripts/bench.py --agents 10 40 80 100 300 1000 --steps 144 \
      --out-name bench_scaling --run-out runs/_bench/p0 \
      --include-runs "runs/_bench/n10" "runs/_bench/n40" runs/rehearsal_pool10k
    * --out-name  出力 stem を引数化(既定 bench)。.json と .md を同時に出す。
    * ★既存の <stem>.json は**上書きしない**(タイムスタンプ付きの別名に退避して警告)。
      --overwrite で従来どおり上書き。2026-07-15 に 3体×3step のスモークが bench.json を潰して
      N スケーリング実測を失った事故の再発防止。
    * --run-out   ベンチランの出力先を --out と分ける(過去のベンチラン dir を守る)。
    * --include-runs  既存 run dir の summary.json を再集計して表に混ぜる(読むだけ・新規ランなし)。
      wall time を持たない世代のランは wall_s=null のまま残す(欠測を捏造しない)。

  # 2) LOD スループット計測(M4・新規): purpose(tier)別に実 LLM を数十呼して
  #    tok/s・呼数・fallback率・平均応答長を1表に。ollama は eval_count で真の tok/s を測る。
  python scripts/bench.py --lod --model qwen3:4b --calls 4
  python scripts/bench.py --lod --backend ollama --model qwen3:4b --reflect-model qwen3:8b

  # 3) reflect 出力上限の右サイズ化(E1・新規): 既存ランの応答長分布を集計し
  #    reflect_max_tokens / plan_max_tokens / max_tokens の妥当性を実測から提案。
  python scripts/bench.py --analyze-runs "runs/modelk_*"

各モードとも runs/_bench/ に JSON を残しコンソールに markdown 表を出す。
src/ には一切変更を加えない — Simulation / バックエンドを import して config 上書きで回すだけ。

計測の注意:
  * スケーリング: wall = time.perf_counter; peak_mem = tracemalloc の Python ヒープ峰値
    (pyarrow の C++ バッファは含まない=相対比較・オーダー把握用)。Windows なので resource 不使用。
  * LOD: ollama backend は /api/generate を直接叩き eval_count(生成トークン真値)/
    eval_duration(ns)から decode tok/s を出す。他 backend は応答文字数から
    --chars-per-token でトークン概算(概算と明記)。fallback = 応答が空 or "__..._error__" 始まり。
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import statistics
import sys
import time
import tracemalloc
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts") not in sys.path:      # 同ディレクトリの run_dt を import
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Windows コンソール(cp932)で日本語パスや記号を print しても落ちないように
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import run_dt                                       # noqa: E402  (W2-3: Δt の単一の源)
from society.config import load_config              # noqa: E402
from society.engine.simulation import Simulation    # noqa: E402


# =============================================================================
# 1) スケーリング・ベンチ(従来。挙動・出力は不変)
# =============================================================================
def bench_one(n_agents: int, steps: int, seed: int, out_root: Path,
              backend: str = "mock", servers: list[str] | None = None) -> dict:
    """1 規模を計測。既定は mock backend・キャッシュ無効(純計算時間を測る)。

    backend='vllm' + servers=[...] を渡すと本選バックエンドへの疎通確認にも使える
    (例: --backend vllm --servers http://localhost:8000 --agents 2 --steps 2)。
    """
    overrides = [
        f"run.n_agents={n_agents}",
        f"run.n_steps={steps}",
        f"run.seed={seed}",
        f"run.name=n{n_agents}",
        f"run.out_dir={out_root.as_posix()}",
        f"model.backend={backend}",
        "model.cache=false",          # 純粋な生成時間を測る(リプレイ加速を除外)
    ]
    if servers:
        overrides.append(f"model.servers={json.dumps(list(servers))}")
    cfg = load_config(overrides=overrides)
    # W2-3: このベンチランの 1 日あたり step 数(Δt=10 なら 144 = 従来と同値)。
    spd = run_dt.steps_per_day(dt_min=run_dt.valid_dt_min(
        (cfg.get("run") or {}).get("dt_min")) or run_dt.CANON_DT_MIN)
    sim = Simulation(cfg)

    tracemalloc.start()
    t0 = time.perf_counter()
    summary = sim.run()
    wall = time.perf_counter() - t0
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    llm_calls = int(summary.get("llm_calls", 0))
    n_events = int(summary.get("n_events", 0))
    return {
        "agents": n_agents,
        "steps": steps,
        "seed": seed,
        "wall_s": round(wall, 4),
        "ms_per_step": round(1000.0 * wall / steps, 3) if steps else None,
        "ms_per_agent_step": (round(1000.0 * wall / (steps * n_agents), 4)
                              if steps and n_agents else None),
        "llm_calls": llm_calls,
        "llm_per_step": round(llm_calls / steps, 2) if steps else None,
        "llm_per_agent_day": _per_agent_day(llm_calls, n_agents, steps, spd),
        "peak_mem_mb": round(peak / (1024 * 1024), 2),
        # P0バッチ 2026-07-29: summary.json の新キー(プロセス実メモリ/シム側 wall)。
        # peak_mem_mb(tracemalloc=Python ヒープ)と違い pyarrow の C++ バッファを含む。
        "peak_rss_mb": summary.get("peak_rss_mb"),
        "summary_elapsed_s": summary.get("elapsed_sec"),
        "n_events": n_events,
        "events_per_step": round(n_events / steps, 1) if steps else None,
        "events_per_agent_day": _per_agent_day(n_events, n_agents, steps, spd),
        "source": "fresh",
    }


def _per_agent_day(total: int, n_agents: int, steps: int,
                   steps_per_day: int = run_dt.CANON_STEPS_PER_DAY):
    """総数 → 1エージェント1シミュ日あたり(steps_per_day step=1日)。分母 0 なら None。

    W2-3: `steps_per_day` の既定は正準 Δt=10 の 144 = 従来と完全同値。呼び出し側は
    そのランの run.dt_min から導いた値を渡す。"""
    denom = n_agents * (steps / float(steps_per_day))
    return round(total / denom, 2) if denom > 0 else None


def rows_from_existing_runs(run_dirs: list[Path]) -> list[dict]:
    """既存 run dir の summary.json を再集計して行にする(**新規ランを走らせない**)。

    P0バッチ 2026-07-29。runs/_bench/n10 等の過去ランや runs/rehearsal_pool10k のような
    大規模ランの実測点を、スケーリング表に**出典つきで**取り込むための読み取り専用モード。
    古いランには wall time が無い(summary に elapsed_sec が入ったのは P0 バッチ以降)ので
    wall_s=None のまま残し、**欠測を捏造しない**。
    """
    out: list[dict] = []
    for d in run_dirs:
        sp = d / "summary.json"
        if not sp.exists():
            print(f"[bench] summary.json が無いのでスキップ: {d}", file=sys.stderr)
            continue
        s = json.loads(sp.read_text(encoding="utf-8"))
        n_agents = int(s.get("n_agents", 0))
        steps = int(s.get("n_steps", 0))
        llm_calls = int(s.get("llm_calls", 0))
        n_events = int(s.get("n_events", 0))
        wall = s.get("elapsed_sec")
        out.append({
            "agents": n_agents, "steps": steps, "seed": None,
            "wall_s": wall,
            "ms_per_step": (round(1000.0 * wall / steps, 3)
                            if wall and steps else None),
            "ms_per_agent_step": (round(1000.0 * wall / (steps * n_agents), 4)
                                  if wall and steps and n_agents else None),
            "llm_calls": llm_calls,
            "llm_per_step": round(llm_calls / steps, 2) if steps else None,
            "llm_per_agent_day": _per_agent_day(llm_calls, n_agents, steps,
                                               run_dt.steps_per_day(d)),
            "peak_mem_mb": None,
            "peak_rss_mb": s.get("peak_rss_mb"),
            "summary_elapsed_s": wall,
            "n_events": n_events,
            "events_per_step": round(n_events / steps, 1) if steps else None,
            "events_per_agent_day": _per_agent_day(n_events, n_agents, steps,
                                                  run_dt.steps_per_day(d)),
            "source": f"existing:{d.name}",
        })
    return out


def resolve_out_paths(out_root: Path, stem: str, overwrite: bool) -> tuple[Path, Path]:
    """(json, md) の出力先を決める。**既存ファイルは既定で上書きしない**。

    P0バッチ 2026-07-29 の再発防止: 2026-07-15 に 3体×3step のスモークが
    runs/_bench/bench.json を上書きし、N スケーリング曲線の実測が失われた
    (実査レポート §2.6)。以後、同名が既に在るときは自動でタイムスタンプ付きの
    別名に退避し、標準エラーに警告を出す(--overwrite で従来どおり上書き)。
    """
    js, md = out_root / f"{stem}.json", out_root / f"{stem}.md"
    if js.exists() and not overwrite:
        ts = time.strftime("%Y%m%d-%H%M%S")
        js, md = out_root / f"{stem}-{ts}.json", out_root / f"{stem}-{ts}.md"
        print(f"[bench] {stem}.json は既存のため上書きせず {js.name} に書く"
              f"(上書きしたいときは --overwrite)", file=sys.stderr)
    return js, md


def _markdown_table(rows: list[dict]) -> str:
    cols = [
        ("agents", "agents"), ("steps", "steps"), ("wall_s", "wall(s)"),
        ("ms_per_step", "ms/step"), ("ms_per_agent_step", "ms/agent-step"),
        ("llm_calls", "llm_calls"), ("llm_per_step", "llm/step"),
        ("llm_per_agent_day", "llm/agent-day"),
        ("peak_mem_mb", "peak_heap(MB)"), ("peak_rss_mb", "peak_rss(MB)"),
        ("n_events", "events"), ("events_per_step", "events/step"),
        ("events_per_agent_day", "events/agent-day"), ("source", "source"),
    ]
    head = "| " + " | ".join(h for _, h in cols) + " |"
    sep = "|" + "|".join("---:" for _ in cols) + "|"
    body = ["| " + " | ".join(str(r.get(k)) for k, _ in cols) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def _run_scaling(args, out_root: Path) -> None:
    rows: list[dict] = []
    # 既存ランの再集計(読み取り専用。--include-runs "runs/_bench/n10" 等)を先に置く
    include: list[Path] = []
    for pat in (args.include_runs or []):
        for p in sorted(_glob.glob(pat)):
            pp = Path(p)
            if pp.is_dir():
                include.append(pp)
    if include:
        rows += rows_from_existing_runs(include)

    # ラン出力先は --run-out で分けられる(既定は --out と同じ=従来どおり)。
    # 過去のベンチラン(runs/_bench/n10 等)を上書きせずに新しい掃引を回すための seam。
    run_root = Path(args.run_out) if args.run_out else out_root
    if not run_root.is_absolute():
        run_root = REPO_ROOT / run_root
    run_root.mkdir(parents=True, exist_ok=True)
    for n in args.agents:
        print(f"[bench] running n_agents={n}, steps={args.steps}, "
              f"backend={args.backend} ...", file=sys.stderr, flush=True)
        rows.append(bench_one(n, args.steps, args.seed, run_root,
                              backend=args.backend, servers=args.servers))

    table = _markdown_table(rows)
    payload = {
        "meta": {
            "backend": args.backend, "cache": False, "steps": args.steps,
            "seed": args.seed, "agents": args.agents,
            "included_runs": [str(p) for p in include],
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "note": ("wall = time.perf_counter; peak_heap = tracemalloc の Python ヒープ峰値"
                     "(pyarrow の C++ バッファを含まない); peak_rss = プロセスのピーク常駐"
                     "メモリ(summary.json の peak_rss_mb・P0バッチで追加。取得不能なら null); "
                     "source=existing:<dir> の行は既存ランの summary.json の再集計で、"
                     "wall time が記録されていない世代のランは wall_s=null(欠測を捏造しない)。"),
        },
        "rows": rows,
    }
    js, md = resolve_out_paths(out_root, args.out_name, args.overwrite)
    js.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md.write_text(f"# {args.out_name}\n\n生成: {payload['meta']['generated']}\n\n"
                  f"{table}\n\n注記: {payload['meta']['note']}\n", encoding="utf-8")

    print("\n" + table + "\n")
    print(f"[bench] wrote {js} / {md}")


# =============================================================================
# 共通・純関数(テスト対象)
# =============================================================================
def _percentile(xs: list[float], p: float):
    """最近傍順位法のパーセンタイル(pandas 無し・純 Python)。xs 非空前提。"""
    if not xs:
        return None
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round((p / 100.0) * (len(s) - 1)))))
    return s[i]


def _summarize(xs: list[float]) -> dict:
    """長さ列を p50/p90/p95/max/mean/n に要約(空なら n=0)。"""
    if not xs:
        return {"n": 0, "p50": None, "p90": None, "p95": None,
                "max": None, "mean": None}
    return {
        "n": len(xs),
        "p50": _percentile(xs, 50),
        "p90": _percentile(xs, 90),
        "p95": _percentile(xs, 95),
        "max": max(xs),
        "mean": round(statistics.mean(xs), 1),
    }


def _estimate_tokens(text: str, chars_per_token: float) -> float:
    """文字数からトークン概算(概算と明記して使う)。日本語混じり JSON は
    qwen 系トークナイザで概ね 1.5–2.2 字/tok(--chars-per-token で指定)。"""
    return len(text) / max(chars_per_token, 1e-9)


def _recommend_max_tokens(max_chars: int, chars_per_token: float,
                          safety: float = 1.5, floor: int = 256) -> int:
    """観測最大応答文字数から出力上限(tokens)を右サイズ提案。

    tokens = ceil(max_chars / chars_per_token) * safety。切り上げは 64 刻み。
    safety=1.5 は「未観測の外れ値+JSON 途中切れ回避」の余裕。floor で下限を確保。
    """
    est = (max_chars / max(chars_per_token, 1e-9)) * safety
    val = max(floor, int(est))
    return ((val + 63) // 64) * 64          # 64 刻みに切り上げ(実装上の見栄え)


# =============================================================================
# 2) LOD スループット計測(M4)
# =============================================================================
# purpose ごとの (出力上限を引く config キー, think, 代表プロンプト)。
# max_tokens は num_predict(生成上限=ceiling)として渡すだけ。think=false では
# モデルが自然停止するので実応答長は上限より短い(=ceiling は wall を支配しない)。
_P_REFLECT = (
    "あなたは渋谷で暮らす30代の会社員です。今日は満員電車の遅延で苛立ち、昼に旧友と再会して"
    "励まされました。一日を静かに振り返ってください。\n"
    '次のJSONのみを出力: {"action":"reflect","summary":"今日一日の要約を一文",'
    '"belief":"今日得た小さな確信を一文","salient":[{"text":"心に残った出来事","importance":7}]}'
)
_P_PLAN = (
    "あなたは渋谷で暮らす30代の会社員です。今日は休日で、午前は買い物、午後は友人と会う予定です。"
    "今日の予定を最大5件、時間の流れに沿って立ててください。\n"
    '次のJSONのみを出力: {"action":"plan","plan":[{"place":"場所名","what":"すること","when":"朝/昼/夜"}]}'
)
_P_DELIB = (
    "あなたは渋谷のカフェにいる30代の会社員です。少し疲れていますが、まだ時間に余裕があります。"
    "次に取る行動を1つ決めてください。\n"
    '次のJSONのみを出力: {"action":"move|stay|speak","target":"行き先や相手","reason":"一言理由"}'
)
_P_SPEAK = (
    "あなたは渋谷でカフェにいる30代の会社員です。隣り合わせた顔見知りに、天気の話題から"
    "自然に一言、話しかけてください。\n"
    '次のJSONのみを出力: {"action":"speak","text":"実際に発する短いセリフ"}'
)

# purpose -> (config のトークン上限キー, think キー, プロンプト)
_LOD_PURPOSES = {
    "reflect":    ("reflect_max_tokens", "reflect_think", _P_REFLECT),
    "plan":       ("plan_max_tokens",    None,            _P_PLAN),
    "deliberate": ("max_tokens",         None,            _P_DELIB),
    "speak":      ("max_tokens",         None,            _P_SPEAK),
}


def _ollama_raw(host: str, model: str, prompt: str, *, max_tokens: int,
                temperature: float, think: bool, format_json: bool,
                timeout_s: float) -> dict:
    """ollama /api/generate を直接叩き、応答本文と計測メタを返す。

    OllamaBackend.generate と同じリクエスト整形(think 境界ガード: think=True には
    format を送らない)。真の tok/s のため eval_count / eval_duration を読む点だけが違う。
    エラーは backend と同様 "__ollama_error__: ..." を text に入れて返す(fallback 検出用)。
    """
    payload = {
        "model": model, "prompt": prompt, "stream": False, "think": bool(think),
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    if (not think) and format_json:
        payload["format"] = "json"
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{host.rstrip('/')}/api/generate", data=body,
        headers={"Content-Type": "application/json"})
    try:
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        wall = time.perf_counter() - t0
        return {
            "text": data.get("response", ""),
            "wall_s": wall,
            "eval_count": data.get("eval_count"),
            "eval_duration_ns": data.get("eval_duration"),
            "prompt_eval_count": data.get("prompt_eval_count"),
            "prompt_eval_duration_ns": data.get("prompt_eval_duration"),
            "total_duration_ns": data.get("total_duration"),
            "error": None,
        }
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError,
            OSError) as exc:
        return {"text": f"__ollama_error__: {exc}", "wall_s": None,
                "eval_count": None, "eval_duration_ns": None,
                "prompt_eval_count": None, "prompt_eval_duration_ns": None,
                "total_duration_ns": None, "error": str(exc)}


def _is_fallback(text: str) -> bool:
    """応答が fallback 扱い(空 or エラーマーカ始まり)か。実運用の D16 と同基準。"""
    t = (text or "").strip()
    if not t:
        return True
    return t.startswith("__") and "_error__" in t[:40]


def _aggregate_lod(records: list[dict], chars_per_token: float) -> list[dict]:
    """呼び出しレコード列(1呼=1 dict)を (model, purpose) 別に集計。

    レコードのキー: model, purpose, chars, fallback(bool),
                    eval_count(optional 真のトークン), eval_duration_ns(optional), wall_s(optional)。
    tok/s は eval_count/eval_duration が揃う呼のみで decode 実測、
    無い呼は chars/chars_per_token ÷ wall の概算(est_only=True でフラグ)。
    """
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        groups.setdefault((r["model"], r["purpose"]), []).append(r)

    out: list[dict] = []
    for (model, purpose), rs in groups.items():
        n = len(rs)
        n_fb = sum(1 for r in rs if r.get("fallback"))
        ok = [r for r in rs if not r.get("fallback")]
        chars = [r["chars"] for r in ok]
        # 真の decode tok/s(eval_count & eval_duration が揃う呼のみ)
        real = [r for r in ok if r.get("eval_count") and r.get("eval_duration_ns")]
        est_only = not real
        if real:
            tot_tok = sum(int(r["eval_count"]) for r in real)
            tot_s = sum(float(r["eval_duration_ns"]) for r in real) / 1e9
            tok_s = tot_tok / tot_s if tot_s else None
            mean_gen_tok = tot_tok / len(real)
            # 実測 字/tok(=生成本文の文字数 ÷ 生成トークン数)
            real_chars = sum(r["chars"] for r in real)
            chars_per_tok_meas = (real_chars / tot_tok) if tot_tok else None
            prompt_toks = [int(r["prompt_eval_count"]) for r in real
                           if r.get("prompt_eval_count")]
            mean_prompt_tok = (statistics.mean(prompt_toks)
                               if prompt_toks else None)
        else:
            walls = [r["wall_s"] for r in ok if r.get("wall_s")]
            est_tok = [_estimate_tokens(r["text"], chars_per_token) for r in ok]
            tok_s = (sum(est_tok) / sum(walls)) if walls and sum(walls) else None
            mean_gen_tok = statistics.mean(est_tok) if est_tok else None
            chars_per_tok_meas = None
            mean_prompt_tok = None
        out.append({
            "model": model,
            "purpose": purpose,
            "calls": n,
            "fallback": n_fb,
            "fallback_rate": round(n_fb / n, 3) if n else None,
            "mean_resp_chars": round(statistics.mean(chars), 1) if chars else None,
            "p95_resp_chars": _percentile(chars, 95) if chars else None,
            "max_resp_chars": max(chars) if chars else None,
            "mean_gen_tok": round(mean_gen_tok, 1) if mean_gen_tok else None,
            "mean_prompt_tok": round(mean_prompt_tok, 1) if mean_prompt_tok else None,
            "decode_tok_s": round(tok_s, 1) if tok_s else None,
            "chars_per_tok_meas": (round(chars_per_tok_meas, 3)
                                   if chars_per_tok_meas else None),
            "est_only": est_only,
        })
    out.sort(key=lambda d: (d["model"], d["purpose"]))
    return out


def _lod_markdown(rows: list[dict]) -> str:
    cols = [
        ("model", "model"), ("purpose", "purpose"), ("calls", "calls"),
        ("fallback_rate", "fallback"), ("decode_tok_s", "tok/s"),
        ("mean_gen_tok", "gen_tok"), ("mean_prompt_tok", "prompt_tok"),
        ("mean_resp_chars", "resp_chars"), ("p95_resp_chars", "p95_chars"),
        ("chars_per_tok_meas", "chars/tok"), ("est_only", "est?"),
    ]
    head = "| " + " | ".join(h for _, h in cols) + " |"
    sep = "|" + "|".join("---:" for _ in cols) + "|"
    body = ["| " + " | ".join(str(r.get(k)) for k, _ in cols) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def _run_lod(args, out_root: Path) -> None:
    """purpose(tier)別に実 LLM を数十呼して LOD スループット表を出す。"""
    # 出力上限は config(必要なら --profile)から取る=実運用の値を反映。
    cfg = load_config(profile=args.profile) if args.profile else load_config()
    budgets = {
        "reflect_max_tokens": int(cfg.model.get("reflect_max_tokens", 1200)),
        "plan_max_tokens": int(cfg.model.get("plan_max_tokens", 0)) or
        int(cfg.model.max_tokens),
        "max_tokens": int(cfg.model.max_tokens),
    }
    reflect_think = bool(cfg.model.get("reflect_think", True))
    fmt_json = str(cfg.model.get("format", "json")) == "json"
    temperature = float(cfg.model.get("temperature", 0.7))

    purposes = args.purposes or list(_LOD_PURPOSES.keys())
    records: list[dict] = []
    for purpose in purposes:
        if purpose not in _LOD_PURPOSES:
            print(f"[lod] 未知 purpose '{purpose}' をスキップ", file=sys.stderr)
            continue
        budget_key, think_key, prompt = _LOD_PURPOSES[purpose]
        max_tokens = budgets[budget_key]
        think = reflect_think if think_key == "reflect_think" else False
        # tier: reflect は --reflect-model があればそちらへ(purpose-LOD の実演)
        model = args.model
        if purpose == "reflect" and args.reflect_model:
            model = args.reflect_model
        n_calls = args.reflect_calls if purpose == "reflect" else args.calls
        for i in range(n_calls):
            print(f"[lod] {model} {purpose} call {i + 1}/{n_calls} "
                  f"(num_predict={max_tokens}, think={think}) ...",
                  file=sys.stderr, flush=True)
            res = _ollama_raw(args.host, model, prompt, max_tokens=max_tokens,
                              temperature=temperature, think=think,
                              format_json=fmt_json, timeout_s=args.timeout)
            text = res["text"]
            records.append({
                "model": model, "purpose": purpose, "text": text,
                "chars": len(text), "fallback": _is_fallback(text),
                "eval_count": res["eval_count"],
                "eval_duration_ns": res["eval_duration_ns"],
                "prompt_eval_count": res["prompt_eval_count"],
                "wall_s": res["wall_s"],
            })

    rows = _aggregate_lod(records, args.chars_per_token)
    table = _lod_markdown(rows)
    payload = {
        "meta": {
            "mode": "lod", "host": args.host, "model": args.model,
            "reflect_model": args.reflect_model, "profile": args.profile,
            "budgets": budgets, "reflect_think": reflect_think,
            "format_json": fmt_json, "temperature": temperature,
            "chars_per_token_assumption": args.chars_per_token,
            "note": ("ollama は eval_count/eval_duration で decode tok/s を実測。"
                     "chars/tok は生成本文字数÷生成トークンの実測比。est?=True は概算のみ。"),
        },
        "rows": rows,
    }
    (out_root / "bench_lod.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + table + "\n")
    print(f"[lod] wrote {out_root / 'bench_lod.json'}")


# =============================================================================
# 3) reflect 出力上限の右サイズ化(既存ランの応答長分布を集計)
# =============================================================================
def _load_cache_index(run_dir: Path) -> dict[str, str]:
    """run 内の llm_cache*.jsonl を key16(=llm_call_id) -> response に索引化。

    CachedLLM は call_id = sha256(...)[:16] を発行し(cache.py)、l1b_llm.parquet の
    llm_call_id がそれと一致する。full key の先頭16桁で突き合わせる。
    """
    idx: dict[str, str] = {}
    for cf in run_dir.glob("llm_cache*.jsonl"):
        try:
            lines = cf.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            key = str(row.get("key", ""))
            if key:
                idx[key[:16]] = row.get("response", "")
    return idx


def _analyze_one_run(run_dir: Path) -> dict:
    """1 run から purpose別 raw 応答長・reflect belief/summary・speak/plan 実長を集計。"""
    import pyarrow.parquet as pq

    result = {
        "raw_by_purpose": {},        # purpose -> [chars]
        "belief": [], "summary": [], "speak_text": [], "plan_n": [],
        "n_salient": [], "reflect_written": 0, "reflect_total": 0,
    }
    l1b = run_dir / "l1b_llm.parquet"
    if l1b.exists():
        cache = _load_cache_index(run_dir)
        tbl = pq.read_table(l1b, columns=["llm_call_id", "purpose"]).to_pylist()
        for r in tbl:
            resp = cache.get(r.get("llm_call_id"))
            if resp is not None:
                result["raw_by_purpose"].setdefault(
                    r.get("purpose"), []).append(len(resp))
    l1 = run_dir / "l1_events.parquet"
    if l1.exists():
        tbl = pq.read_table(l1, columns=["kind", "payload"]).to_pylist()
        for r in tbl:
            kind = r.get("kind")
            try:
                p = json.loads(r["payload"]) if r.get("payload") else {}
            except (json.JSONDecodeError, ValueError, TypeError):
                p = {}
            if kind == "reflect":
                result["reflect_total"] += 1
                if p.get("written_back"):
                    result["reflect_written"] += 1
                b, s = p.get("belief"), p.get("summary")
                if isinstance(b, str) and b:
                    result["belief"].append(len(b))
                if isinstance(s, str) and s:
                    result["summary"].append(len(s))
                if "n_salient" in p:
                    result["n_salient"].append(int(p.get("n_salient") or 0))
            elif kind == "speak":
                t = p.get("text")
                if isinstance(t, str):
                    result["speak_text"].append(len(t))
            elif kind == "day_plan":
                result["plan_n"].append(int(p.get("n") or len(p.get("plan") or [])))
    return result


def _merge_run_stats(per_run: list[dict]) -> dict:
    """複数 run の集計を統合(purpose別 raw 応答長 + フィールド長)。"""
    raw: dict[str, list[int]] = {}
    belief: list[int] = []
    summary: list[int] = []
    speak: list[int] = []
    plan_n: list[int] = []
    n_salient: list[int] = []
    r_total = r_written = 0
    for pr in per_run:
        for pu, xs in pr["raw_by_purpose"].items():
            raw.setdefault(pu, []).extend(xs)
        belief += pr["belief"]
        summary += pr["summary"]
        speak += pr["speak_text"]
        plan_n += pr["plan_n"]
        n_salient += pr["n_salient"]
        r_total += pr["reflect_total"]
        r_written += pr["reflect_written"]
    return {"raw": raw, "belief": belief, "summary": summary, "speak": speak,
            "plan_n": plan_n, "n_salient": n_salient,
            "reflect_total": r_total, "reflect_written": r_written}


def _run_analyze(args, out_root: Path) -> None:
    cpt = args.chars_per_token
    run_dirs: list[Path] = []
    for pat in args.analyze_runs:
        for p in sorted(_glob.glob(pat)):
            pp = Path(p)
            if pp.is_dir():
                run_dirs.append(pp)
    if not run_dirs:
        print(f"[analyze] 対象 run が見つからない: {args.analyze_runs}",
              file=sys.stderr)
        return
    print(f"[analyze] {len(run_dirs)} runs を集計 ...", file=sys.stderr, flush=True)
    per_run = [_analyze_one_run(d) for d in run_dirs]
    m = _merge_run_stats(per_run)

    # --- purpose別 raw 応答長分布(トークン概算つき) ---
    raw_summ = {}
    for pu, xs in m["raw"].items():
        s = _summarize(xs)
        if s["max"] is not None:
            s["max_tok_est"] = round(_estimate_tokens("x" * s["max"], cpt), 0)
            s["p95_tok_est"] = round(_estimate_tokens("x" * s["p95"], cpt), 0)
        raw_summ[pu] = s

    # speak 系(発話ファミリ)= 会話系 purpose の raw をまとめたもの(max_tokens=320 検証)
    speak_family = {"reply", "social", "solo", "post", "dm"}
    speak_raw = [x for pu in speak_family for x in m["raw"].get(pu, [])]

    reflect_raw = m["raw"].get("reflect", [])
    plan_raw = m["raw"].get("plan", [])

    fields = {
        "reflect_raw_response": _summarize(reflect_raw),
        "plan_raw_response": _summarize(plan_raw),
        "speak_family_raw_response": _summarize(speak_raw),
        "belief_field": _summarize(m["belief"]),
        "summary_field": _summarize(m["summary"]),
        "speak_text_field": _summarize(m["speak"]),
        "plan_n_items": _summarize(m["plan_n"]),
        "n_salient": _summarize(m["n_salient"]),
    }

    # --- 推奨値(観測 max × 安全率 ÷ 字/tok) ---
    recos = {}
    if reflect_raw:
        recos["reflect_max_tokens"] = _recommend_max_tokens(max(reflect_raw), cpt)
    if plan_raw:
        recos["plan_max_tokens"] = _recommend_max_tokens(max(plan_raw), cpt)
    if speak_raw:
        recos["max_tokens"] = _recommend_max_tokens(max(speak_raw), cpt)

    payload = {
        "meta": {
            "mode": "analyze-runs", "runs": [d.name for d in run_dirs],
            "n_runs": len(run_dirs), "chars_per_token_assumption": cpt,
            "note": ("raw 応答長は l1b_llm.llm_call_id → llm_cache*.jsonl(key[:16])の突合で復元。"
                     "tok_est は文字数÷chars_per_token の概算。推奨=観測max×1.5(安全率)÷字tok を"
                     "64刻み切上げ。実測トークンは scripts/bench.py --lod の chars/tok を用いて更新可。"),
        },
        "reflect_writeback": {"total": m["reflect_total"],
                              "written_back": m["reflect_written"]},
        "raw_by_purpose": raw_summ,
        "fields": fields,
        "recommendations": recos,
    }
    (out_root / "bench_reflect_sizing.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- コンソール表 ---
    def _row(name, s):
        return (f"| {name} | {s['n']} | {s['p50']} | {s['p95']} | "
                f"{s['max']} | {s['mean']} |")
    print("\n### purpose別 raw 応答長(文字)")
    print("| purpose | n | p50 | p95 | max | mean |")
    print("|---|---:|---:|---:|---:|---:|")
    for pu, s in sorted(raw_summ.items(), key=lambda kv: -(kv[1]["n"] or 0)):
        print(_row(pu, s))
    print("\n### フィールド実長(文字)")
    print("| field | n | p50 | p95 | max | mean |")
    print("|---|---:|---:|---:|---:|---:|")
    for name, s in fields.items():
        print(_row(name, s))
    print(f"\n### 推奨(字/tok={cpt} 仮定・観測max×1.5・64刻み)")
    for k, v in recos.items():
        print(f"  {k}: {v}")
    print(f"\n[analyze] wrote {out_root / 'bench_reflect_sizing.json'}")


# =============================================================================
# エントリポイント
# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Scaling bench + LOD throughput + reflect 出力上限 分析")
    # --- スケーリング(従来。既定モード)---
    ap.add_argument("--agents", type=int, nargs="+", default=[10, 40, 80])
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs/_bench")
    ap.add_argument("--out-name", default="bench",
                    help="スケーリング出力の stem(既定 bench → bench.json/bench.md)。"
                         "例: --out-name bench_scaling")
    ap.add_argument("--overwrite", action="store_true",
                    help="既存の <out-name>.json を上書きする(既定は上書きせず "
                         "タイムスタンプ付きの別名に退避=実測の喪失防止)")
    ap.add_argument("--run-out", default=None,
                    help="ベンチランの出力先(既定は --out と同じ)。過去のベンチラン"
                         "(runs/_bench/n10 等)を上書きしたくないときに分ける")
    ap.add_argument("--include-runs", nargs="*", default=None,
                    help="既存 run dir の summary.json を再集計して表に含める"
                         '(例: --include-runs "runs/_bench/n[0-9]*" runs/rehearsal_pool10k)')
    ap.add_argument("--backend", default="mock",
                    help="mock | ollama | vllm(本選)。vllm は --servers 必須。")
    ap.add_argument("--servers", nargs="*", default=None,
                    help="vLLM の base_url(複数可)。疎通確認: --backend vllm --servers http://localhost:8000")
    # --- LOD スループット計測(M4)---
    ap.add_argument("--lod", action="store_true",
                    help="purpose(tier)別に実 LLM を数十呼して tok/s・fallback率・応答長を測る。")
    ap.add_argument("--model", default="qwen3:4b", help="[--lod] ollama モデル名(既定 qwen3:4b)")
    ap.add_argument("--reflect-model", default=None,
                    help="[--lod] reflect だけ別モデル(tier 実演。例 qwen3:8b)")
    ap.add_argument("--host", default="http://localhost:11434",
                    help="[--lod] ollama ホスト")
    ap.add_argument("--calls", type=int, default=5, help="[--lod] purpose あたりの呼数")
    ap.add_argument("--reflect-calls", type=int, default=3,
                    help="[--lod] reflect の呼数(長め=時間短縮のため少なめ既定)")
    ap.add_argument("--purposes", nargs="*", default=None,
                    help="[--lod] 対象 purpose(既定: reflect plan deliberate speak)")
    ap.add_argument("--profile", default=None,
                    help="[--lod] トークン上限を取る config プロファイル(例 conf/daily.yaml)")
    ap.add_argument("--timeout", type=float, default=120.0, help="[--lod] 1呼のタイムアウト秒")
    # --- reflect 出力上限 分析 ---
    ap.add_argument("--analyze-runs", nargs="+", default=None,
                    help='既存ランの応答長分布を集計(例: --analyze-runs "runs/modelk_*")')
    ap.add_argument("--chars-per-token", type=float, default=1.8,
                    help="トークン概算の字/tok 仮定(既定1.8。--lod の実測 chars/tok で更新推奨)")
    args = ap.parse_args()

    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = REPO_ROOT / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    if args.analyze_runs:
        _run_analyze(args, out_root)
    elif args.lod:
        _run_lod(args, out_root)
    else:
        _run_scaling(args, out_root)


if __name__ == "__main__":
    main()
