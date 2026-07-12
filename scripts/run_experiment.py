"""対照実験ランナー(第9バッチ 2026-07-07。ユーザー質問「条件差はコードで書くか・パラメータで
書くか」への回答=**パラメータ宣言方式**)。

決定の理由:
- 全条件を conf/experiments/*.yaml のマニフェストに**宣言**し、コードは一切書き換えない
  (config.yaml 冒頭の D11 原則の実験行列への拡張)。
- 条件間で変わるのは OmegaConf の上書きパラメータだけ=条件差が run dir の config スナップ
  ショットに完全に記録され、再現・監査・横断分析(analyze_sweep)が機械的にできる。
- コードを条件ごとに書き換える方式は、差分がコミット履歴に散らばり「どのランがどのコードか」
  の対応が壊れる+ゴールデン/決定論の保証が失われるため不採用。
- 新しい機構が必要になったら「まず既定 OFF の config ノブとして実装 → マニフェストで ON」
  という既存の波の作法に従う(条件=ノブの組合せ、という不変則を保つ)。

使い方:
    python scripts/run_experiment.py conf/experiments/example_contrast.yaml
    python scripts/run_experiment.py conf/experiments/example_contrast.yaml --dry-run

マニフェスト形式(conf/experiments/example_contrast.yaml 参照):
    name: 実験名(ラン名の接頭辞)
    profile: conf/production.yaml   # 省略可。基底 config に重ねるプロファイル
    seeds: [1, 2, 3]                # 各条件をこのシード群で反復
    common: { run.n_agents: 60 }    # 全条件共通の上書き
    conditions:                     # 条件 = 上書きパラメータの組(名前つき)
      - name: base
        overrides: {}
      - name: no_recursion
        overrides: { recursion.enabled: false }

ラン名 = <name>_<condition>_s<seed>。runs/<実験名>_index.json に行列の索引を書き出す
(analyze 横断用)。実行は直列(k 掃引 run_sweep.py と同じ作法。LLM 資源の奪い合いを避ける)。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from omegaconf import OmegaConf  # noqa: E402

from society.config import load_config  # noqa: E402


def _dot(value) -> str:
    """マニフェストの値 → dotlist の右辺文字列(bool は YAML の true/false に合わせる)。"""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def load_manifest(path: Path) -> dict:
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict):
        raise SystemExit(f"マニフェストが辞書でない: {path}")
    name = str(raw.get("name") or path.stem)
    conditions = raw.get("conditions") or []
    if not conditions:
        raise SystemExit("conditions が空(最低1条件が必要)")
    for c in conditions:
        if not isinstance(c, dict) or not c.get("name"):
            raise SystemExit(f"条件に name がない: {c}")
    return {
        "name": name,
        "profile": raw.get("profile"),
        "seeds": [int(s) for s in (raw.get("seeds") or [42])],
        "common": dict(raw.get("common") or {}),
        "conditions": [{"name": str(c["name"]),
                        "overrides": dict(c.get("overrides") or {})}
                       for c in conditions],
    }


def expand_matrix(manifest: dict) -> list[dict]:
    """条件 × シード の実行行列を組む(決定論: 宣言順 × シード昇順)。"""
    jobs: list[dict] = []
    for cond in manifest["conditions"]:
        for seed in manifest["seeds"]:
            run_name = f"{manifest['name']}_{cond['name']}_s{seed}"
            overrides = [f"{k}={_dot(v)}" for k, v in manifest["common"].items()]
            overrides += [f"{k}={_dot(v)}" for k, v in cond["overrides"].items()]
            overrides += [f"run.seed={seed}", f"run.name={run_name}"]
            jobs.append({"run": run_name, "condition": cond["name"],
                         "seed": seed, "overrides": overrides})
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", help="conf/experiments/*.yaml")
    ap.add_argument("--dry-run", action="store_true",
                    help="実行せず行列と上書きを表示する")
    args = ap.parse_args()

    mpath = Path(args.manifest)
    if not mpath.is_absolute():
        mpath = REPO_ROOT / mpath
    manifest = load_manifest(mpath)
    jobs = expand_matrix(manifest)

    print(f"実験: {manifest['name']}  条件={len(manifest['conditions'])} × "
          f"シード={len(manifest['seeds'])} = {len(jobs)} ラン"
          + (f"  (profile: {manifest['profile']})" if manifest["profile"] else ""))
    if args.dry_run:
        for j in jobs:
            print(f"  {j['run']}: {' '.join(j['overrides'])}")
        return

    from society.engine.simulation import Simulation  # 遅延 import(dry-run を軽く)
    profile = manifest["profile"]
    if profile is not None:
        p = Path(str(profile))
        profile = str(p if p.is_absolute() else REPO_ROOT / p)
    index = {"name": manifest["name"], "profile": manifest["profile"],
             "runs": []}
    for i, j in enumerate(jobs, 1):
        # 掃引と同じくクリーン実行(旧 llm_cache が条件差を消す事故を防ぐ)
        shutil.rmtree(REPO_ROOT / "runs" / j["run"], ignore_errors=True)
        t0 = time.time()
        cfg = load_config(overrides=j["overrides"], profile=profile)
        summary = Simulation(cfg).run()
        index["runs"].append({**{k: j[k] for k in ("run", "condition", "seed")},
                              "n_events": summary.get("n_events"),
                              "llm_calls": summary.get("llm_calls")})
        print(f"[{i}/{len(jobs)}] {j['run']}: {summary['n_events']}ev "
              f"llm={summary['llm_calls']} ({time.time() - t0:.0f}s)", flush=True)
    out = REPO_ROOT / "runs" / f"{manifest['name']}_index.json"
    out.write_text(json.dumps(index, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"索引: {out}")


if __name__ == "__main__":
    main()
