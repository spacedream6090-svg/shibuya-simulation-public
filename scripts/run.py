"""実行エントリポイント。

使い方:
    python scripts/run.py                       # 既定(10体×144step、Mock LLM)
    python scripts/run.py run.seed=7 run.n_agents=30
    python scripts/run.py run.seed=auto         # seed を OS エントロピーから自動採取(config/summary に記録)
    python scripts/run.py k.writeback=sham      # R1 対照(偽内省)
    python scripts/run.py --profile conf/production.yaml   # 本番: 全リアリズム機能ON+初期条件
        # プロファイルは基底 config.yaml の上に差分を重ねる。dotlist で更に上書き可(例: run.seed=7)。
    python scripts/run.py --profile conf/observe.yaml      # 純観察モード(不確実性許容・seed=auto 推奨)
    python scripts/run.py --env env/shibuya      # EnvPack(場所)を束ねて読む(D1-W4)
        # 優先順位: 基底 < env < profile < dotlist。--profile と併用可(env=場所・profile=用途)。
    python scripts/run.py --resume runs/<name>  # D16 途中再開(最新 checkpoint から続行)
        # resume は run dir の config.yaml を読む。さらに先まで回すなら run.n_steps=… を併記。
        # 例: python scripts/run.py --resume runs/day80 run.n_steps=288

seed の自動採取(第54バッチ 2026-07-23。純観察=不確実性許容モード):
    run.seed=auto(または run.seed_auto=true)で、起動時に OS エントロピー(secrets=os.urandom 由来)から
    seed を採取し、**通常の run.seed として注入**する(以降の機構は完全に従来どおり=決定論エンジンは不変)。
    採取した seed は (a) 標準出力 (b) run dir の config.yaml(save_config が数値 seed をそのまま保存)+
    summary.json(seed/seed_source を追記)へ必ず記録=「選ばない」だけで「失わない」(事後に同 seed で完全
    再現できる)。**数値 seed を明示した既定の挙動は完全に不変**(auto を要求しない限り 1 バイトも変えない)。
"""
from __future__ import annotations

import json
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omegaconf import OmegaConf                  # noqa: E402

from society.config import load_config          # noqa: E402
from society.engine.simulation import Simulation  # noqa: E402


def _sample_seed() -> int:
    """OS エントロピー(secrets=os.urandom 由来)から seed を採取する。

    63bit 非負整数=numpy SeedSequence/PCG64・OmegaConf(int64)・YAML/JSON いずれでも安全に往復でき、
    標準出力/ファイル記録が読みやすい範囲に収める(採取のたびに実質異なる値=実験者が seed を選ばない)。"""
    return int(secrets.randbits(63))


def _resolve_seed_auto(overrides: list[str]) -> tuple[bool, list[str]]:
    """dotlist から seed 自動採取の要求を検出し、'auto' センチネルを取り除く。

    `run.seed=auto` は load_config の int 強制(config._INT_KEYS に run.seed が含まれる)が int('auto') で
    壊れるため、ここで dotlist から除去してから load_config に渡す(OmegaConf の型都合)。`run.seed_auto=true`
    は正規の bool キーなので**残す**(config.yaml/summary に「auto を使った」記録として保存される)。両形式とも
    採取フラグに合流する。"""
    seed_auto = False
    kept: list[str] = []
    for ov in overrides:
        key, sep, val = ov.partition("=")
        k = key.strip()
        if k == "run.seed" and sep and val.strip().lower() == "auto":
            seed_auto = True
            continue                                # 除去: int('auto') で load_config を壊さない
        if k == "run.seed_auto" and sep and val.strip().lower() in ("true", "1", "yes", "on"):
            seed_auto = True                        # 残す(下で kept に積む)=記録に残す
        kept.append(ov)
    return seed_auto, kept


def _record_seed_in_summary(out_dir: Path, seed: int) -> None:
    """採取した seed を run dir の summary.json に追記する(config.yaml は save_config が数値 seed を保存済み)。

    finalize が書いた summary.json に seed/seed_source を足し戻すだけ(auto を要求した時のみ呼ぶ=既定の
    数値 seed ランでは summary.json も従来と完全同一)。"""
    path = out_dir / "summary.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}
    data["seed"] = int(seed)
    data["seed_source"] = "auto"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


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
        # resume は保存済み seed で続行する(auto は無効=再現性を壊さない)。
        _seed_auto_ignored, overrides = _resolve_seed_auto(overrides)
        # ★apply_dt=False(第94バッチ OBS-U2): run dir の config.yaml は save_config が
        # **apply_dt 済みの姿**で書いたスナップショットなのに run.dt_min を保持している。
        # 既定どおり再変換すると Δt が二重適用され(walk 80→8 / refractory 30→300 …)、
        # checkpoint の config_hash 照合で必ず ValueError になる(Δt≠10 固有。Δt=10 は
        # apply_dt が恒等パスなので従来から無風=この行の追加でも 1 バイトも変わらない)。
        cfg = load_config(overrides=overrides, path=run_dir / "config.yaml",
                          apply_dt=False)
        sim = Simulation(cfg, out_dir=run_dir)     # 既存 run dir は消さずに続行する
        summary = sim.run(resume_from=run_dir)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    # seed の自動採取(第54バッチ): auto 要求を検出 → load_config → 数値 seed を注入(以降は従来どおり)。
    seed_auto, overrides = _resolve_seed_auto(overrides)
    cfg = load_config(overrides=overrides, profile=profile, env=env)
    seed_auto = seed_auto or bool(cfg.run.get("seed_auto", False))
    if seed_auto:
        sampled = _sample_seed()
        OmegaConf.update(cfg, "run.seed", int(sampled))
        # 実験者が seed を選ばない代わりに「失わない」= 標準出力 + config.yaml(save_config) + summary.json。
        print(f"[seed] auto-sampled run.seed={sampled} "
              f"(OS エントロピー採取。config.yaml/summary.json に記録=同 seed で完全再現可)")
    sim = Simulation(cfg)
    summary = sim.run()
    if seed_auto:
        _record_seed_in_summary(Path(summary["out_dir"]), int(cfg.run.seed))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
