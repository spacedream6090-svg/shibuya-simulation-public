#!/usr/bin/env python
"""共通ハーネス — 候補モデルに同一プロンプト・同一シード・同一温度を流す(第90バッチ)。

正典: docs/plans/source/design-discussion-20260802.md §4。

使い方(縮小版・ローカル Ollama + プラセボ):
    python scripts/model_battery/harness.py \\
        --model ollama:qwen3:4b --model ollama:qwen3:8b --model placebo:template \\
        --layers all --scale 0.2 --seed 20260803 --temperature 0.8

出力:
    data/battery/raw/<model_slug>/<test>.jsonl   … 1呼び出し1行(来歴つき)
    data/battery/raw/<model_slug>/manifest.json  … モデル/温度/シード/プロンプト鎖ハッシュ

決定論について(正直な宣言):
  - **ハーネス層は完全に決定論**である。刺激(プロンプト)・シード・温度・呼び出し順は
    seed と scale だけの関数であり、モデルにも時刻にも環境にも依存しない。
  - **モデル層の決定論は保証しない**。ollama には options.seed を渡しているが、
    再現するかはサーバ/ドライバ/バッチ状況に依存する。ハーネスは毎回
    `determinism_probe`(先頭 K 呼を同一シードで再送し応答の一致率を測る)を実行し、
    その実測値を manifest に残す。「再現した/しなかった」は宣言でなく計測で答える。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:              # scripts/ を import 経路に
    sys.path.insert(0, str(_HERE.parent))

for _s in (sys.stdout, sys.stderr):                # Windows cp932 対策(house style)
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                              # noqa: BLE001
        pass

from model_battery import HARNESS_VERSION                       # noqa: E402
from model_battery import clients as C                          # noqa: E402
from model_battery import stimuli as S                          # noqa: E402

RAW_SCHEMA = "model_battery.raw/1"
MANIFEST_SCHEMA = "model_battery.manifest/1"
REPO_ROOT = _HERE.parents[1]
DEFAULT_OUT = REPO_ROOT / "data" / "battery" / "raw"

# 記録のうち「決定論であるべき」フィールド(時刻・所要秒は含めない)
DET_FIELDS = ("layer", "test", "case_id", "turn", "params", "prompt_sha256",
              "response_sha256", "error")


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def det_digest(record: dict) -> str:
    """1記録の決定論部分の正準ハッシュ(observer.state_hash と同じ流儀)。"""
    sub = {k: record.get(k) for k in DET_FIELDS}
    blob = json.dumps(sub, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def chain(digests) -> str:
    """順序つきハッシュ鎖(1つでも違えば違う値になる)。"""
    h = hashlib.sha256()
    for d in digests:
        h.update(d.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


class Runner:
    """1モデル分のバッテリー実行。"""

    def __init__(self, client: C.BatteryClient, cfg: S.BatteryConfig,
                 out_dir: Path, *, verbose: bool = True):
        self.client = client
        self.cfg = cfg
        self.out_dir = out_dir
        self.verbose = verbose
        self.records: list[dict] = []
        self.calls: list[S.Call] = []
        self._t0 = time.perf_counter()

    def ask(self, call: S.Call) -> str:
        t0 = time.perf_counter()
        text = self.client.generate(
            call.prompt, temperature=self.cfg.temperature,
            max_tokens=call.max_tokens, seed=call.seed,
            json_mode=call.json_mode)
        elapsed = time.perf_counter() - t0
        err = C.is_error(text)
        rec = {
            "schema": RAW_SCHEMA,
            "harness_version": HARNESS_VERSION,
            "layer": call.layer,
            "test": call.test,
            "case_id": call.case_id,
            "turn": call.turn,
            "model": self.client.describe(),
            "params": {"temperature": self.cfg.temperature,
                       "max_tokens": call.max_tokens,
                       "seed": call.seed,
                       "json_mode": call.json_mode},
            "prompt": call.prompt,
            "prompt_sha256": call.prompt_sha256,
            "prompt_chars": len(call.prompt),
            "response": text,
            "response_sha256": sha256_text(text),
            "response_chars": len(text or ""),
            "error": err,
            "elapsed_s": round(elapsed, 3),
            "meta": dict(call.meta),
        }
        self.records.append(rec)
        self.calls.append(call)
        if self.verbose:
            tag = "ERR" if err else "ok "
            print(f"  [{tag}] {call.layer} {call.test} {call.case_id} "
                  f"t{call.turn} {elapsed:6.2f}s {len(text or ''):5d}ch",
                  file=sys.stderr, flush=True)
        # エラー時は空文字を返す(C/E の履歴に "__ollama_error__" を混ぜない)
        return "" if err else text

    def run(self, layers: tuple[str, ...]) -> None:
        for layer in layers:
            if self.verbose:
                print(f"[{self.client.name}] 層 {layer}", file=sys.stderr,
                      flush=True)
            S.DRIVERS[layer](self.cfg, self.ask)

    # ---- 決定論プローブ ----
    def determinism_probe(self, k: int) -> dict:
        """先頭 K 呼を**同一シード・同一プロンプト**で再送し、応答一致率を測る。

        ハーネスの決定論(プロンプトとシードが同じか)は構造上保証済みなので、
        ここで測っているのは **モデル/サーバ側の再現性**である。
        """
        k = min(int(k), len(self.records))
        if k <= 0:
            return {"n": 0, "identical": 0, "rate": None,
                    "note": "プローブ無効(--determinism-probe 0)"}
        same = 0
        details = []
        for rec in self.records[:k]:
            text = self.client.generate(
                rec["prompt"], temperature=rec["params"]["temperature"],
                max_tokens=rec["params"]["max_tokens"],
                seed=rec["params"]["seed"],
                json_mode=rec["params"]["json_mode"])
            hit = sha256_text(text) == rec["response_sha256"]
            same += int(hit)
            details.append({"test": rec["test"], "case_id": rec["case_id"],
                            "turn": rec["turn"], "identical": hit})
        return {"n": k, "identical": same, "rate": same / k, "details": details}

    # ---- 書き出し ----
    def write(self, probe: dict) -> dict:
        d = self.out_dir / self.client.slug
        d.mkdir(parents=True, exist_ok=True)
        by_test: dict[str, list[dict]] = {}
        for rec in self.records:
            by_test.setdefault(rec["test"], []).append(rec)

        files = {}
        for test, recs in by_test.items():
            path = d / f"{test}.jsonl"
            with path.open("w", encoding="utf-8", newline="\n") as fh:
                for rec in recs:
                    fh.write(json.dumps(rec, ensure_ascii=False,
                                        sort_keys=True) + "\n")
            files[test] = {
                "path": path.name,
                "n": len(recs),
                "errors": sum(1 for r in recs if r["error"]),
                "prompt_chain": chain(r["prompt_sha256"] for r in recs),
                "response_chain": chain(r["response_sha256"] for r in recs),
                "record_chain": chain(det_digest(r) for r in recs),
            }

        manifest = {
            "schema": MANIFEST_SCHEMA,
            "harness_version": HARNESS_VERSION,
            "model": self.client.describe(),
            "config": {"seed": self.cfg.seed, "temperature": self.cfg.temperature,
                       "scale": self.cfg.scale,
                       "n_personas": self.cfg.n_personas,
                       "n_topics": self.cfg.n_topics,
                       "n_dialog_turns": self.cfg.n_dialog_turns,
                       "n_d_samples": self.cfg.n_d_samples,
                       "n_e_turns": self.cfg.n_e_turns,
                       "n_e_days": self.cfg.n_e_days},
            "calls": len(self.records),
            "errors": sum(1 for r in self.records if r["error"]),
            "elapsed_s": round(time.perf_counter() - self._t0, 2),
            "files": files,
            "digests": {
                "prompt_chain": chain(r["prompt_sha256"] for r in self.records),
                "response_chain": chain(r["response_sha256"] for r in self.records),
                "record_chain": chain(det_digest(r) for r in self.records),
            },
            "determinism_probe": probe,
            "determinism_note": (
                "prompt_chain/record_chain はハーネス層の決定論、"
                "response_chain と determinism_probe はモデル層の再現性を表す。"
                "C/E 層のプロンプトは前ターンの応答に依存するため、モデルが揺れると "
                "prompt_chain も揺れる(構造上の帰結であってバグではない)。"),
        }
        (d / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n", encoding="utf-8")
        return manifest


def estimate_plan(cfg: S.BatteryConfig, layers: tuple[str, ...]) -> dict:
    calls = S.collect_calls(cfg, layers)
    per_test: dict[str, int] = {}
    for c in calls:
        per_test[c.test] = per_test.get(c.test, 0) + 1
    return {"total_calls": len(calls), "per_test": per_test,
            "prompt_chain": chain(c.prompt_sha256 for c in calls)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="モデル人間らしさテストバッテリー 共通ハーネス")
    ap.add_argument("--model", action="append", required=True,
                    help="'<backend>:<model>[|k=v]'。複数指定可。"
                         "例 ollama:qwen3:4b / placebo:template")
    ap.add_argument("--layers", default="all", help="A,B,C,D,E / all(既定)")
    ap.add_argument("--seed", type=int, default=20260803)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="各層のサンプル数の縮尺(0<scale<=1)。0.2 = 縮小版")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--determinism-probe", type=int, default=3,
                    help="先頭 K 呼を同一シードで再送し一致率を測る(0 で無効)")
    ap.add_argument("--no-unload", action="store_true",
                    help="モデル終了時に VRAM から降ろさない(既定は降ろす)")
    ap.add_argument("--dry-run", action="store_true",
                    help="LLM を呼ばず刺激の件数と鎖ハッシュだけ出す")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if not 0.0 < args.scale <= 1.0:
        print("[harness] --scale は (0,1] の範囲", file=sys.stderr)
        return 2
    cfg = S.BatteryConfig(seed=args.seed, temperature=args.temperature,
                          scale=args.scale)
    layers = S.iter_layers(args.layers)

    plan = estimate_plan(cfg, layers)
    print(f"[harness] 層={','.join(layers)} scale={cfg.scale} "
          f"seed={cfg.seed} temp={cfg.temperature}", file=sys.stderr)
    print(f"[harness] 1モデルあたり {plan['total_calls']} 呼 "
          f"{plan['per_test']}", file=sys.stderr)
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    out_dir = Path(args.out)
    summary = []
    for spec in args.model:
        client = C.parse_model_spec(spec)
        print(f"[harness] === {client.name} ===", file=sys.stderr, flush=True)
        runner = Runner(client, cfg, out_dir, verbose=not args.quiet)
        runner.run(layers)
        probe = runner.determinism_probe(args.determinism_probe)
        man = runner.write(probe)
        # ★モデルの切れ目で VRAM を空ける。常駐したままだと次のモデルが部分 CPU
        # オフロードになり十数倍遅くなる(clients.OllamaClient.unload の注記)。
        if not args.no_unload and client.unload():
            print(f"[harness] {client.name} を VRAM から降ろした",
                  file=sys.stderr, flush=True)
        summary.append({"model": client.name, "slug": client.slug,
                        "calls": man["calls"], "errors": man["errors"],
                        "elapsed_s": man["elapsed_s"],
                        "determinism_rate": probe.get("rate")})
        print(f"[harness] {client.name}: {man['calls']}呼 / "
              f"err={man['errors']} / {man['elapsed_s']}s / "
              f"det={probe.get('rate')}", file=sys.stderr, flush=True)

    print(json.dumps({"out": str(out_dir), "models": summary},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
