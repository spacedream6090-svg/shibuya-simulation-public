"""実行エントリポイント。

使い方:
    python scripts/run.py                       # 既定(10体×144step、Mock LLM)
    python scripts/run.py run.seed=7 run.n_agents=30
    python scripts/run.py k.writeback=sham      # R1 対照(偽内省)
    python scripts/run.py --profile conf/production.yaml   # 本番: 全リアリズム機能ON+初期条件
        # プロファイルは基底 config.yaml の上に差分を重ねる。dotlist で更に上書き可(例: run.seed=7)。
    python scripts/run.py --env env/shibuya      # EnvPack(場所)を束ねて読む(D1-W4)
        # 優先順位: 基底 < env < profile < dotlist。--profile と併用可(env=場所・profile=用途)。
    python scripts/run.py --resume runs/<name>  # D16 途中再開(最新 checkpoint から続行)
        # resume は run dir の config.yaml を読む。さらに先まで回すなら run.n_steps=… を併記。
        # 例: python scripts/run.py --resume runs/day80 run.n_steps=288
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from society.config import load_config          # noqa: E402
from society.engine.simulation import Simulation  # noqa: E402


def main() -> None:
    args = list(sys.argv[1:])
    resume_dir: str | None = None
    profile: str | None = None
    env: str | None = None
    overrides: list[str] = []
    i = 0
    while i < len(args):
        if args[i] == "--resume":
            resume_dir = args[i + 1]
            i += 2
        elif args[i] == "--profile":
            profile = args[i + 1]
            i += 2
        elif args[i] == "--env":
            env = args[i + 1]
            i += 2
        else:
            overrides.append(args[i])
            i += 1

    if resume_dir is not None:
        run_dir = Path(resume_dir)
        # run dir に保存された config を土台に、追加 override(例: 新しい run.n_steps)を適用。
        cfg = load_config(overrides=overrides, path=run_dir / "config.yaml")
        sim = Simulation(cfg, out_dir=run_dir)     # 既存 run dir は消さずに続行する
        summary = sim.run(resume_from=run_dir)
    else:
        cfg = load_config(overrides=overrides, profile=profile, env=env)
        sim = Simulation(cfg)
        summary = sim.run()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
