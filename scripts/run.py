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

起動ガードと起動バナー(β6。第117バッチ 2026-08-16。監査 E0-4 = external-audit-triage.md F5):
    基底 conf の `model.backend` は **mock**(縦煙・テストのための意図的な既定)なので、本選の起動で
    `model.backend=vllm` の dotlist を 1 つ打ち間違えると「LLM が 1 本も走っていない 25 万体ラン」が
    数十時間ぶん静かに完走してしまう(L1 は完全に正常な形をしている)。ここで 2 つの安い保険を張る:
      - `check_mock_production()` … n_agents>=10,000 かつ backend=="mock" かつ
        `run.allow_mock_production` が false なら **起動時に RuntimeError**。
      - `print_banner()` … backend / モデル名 / servers 数 / pool dir / present_cap /
        lod.max_llm_per_step を stdout へ 1 回だけ出す(目視 1 秒で条件を確かめられる)。
    どちらも **世界へは 1 バイトも触らない**(Simulation はこの 2 関数の存在を知らない)。

使用済み run dir への fresh 起動を拒否(A7。第118 レーンC 2026-08-16):
    `--resume` **なし**の起動で、出力先に前回ランの痕跡(`*.part-*.parquet` /
    `l1_events.parquet` / `checkpoint/*.pkl.gz` / `llm_cache*.jsonl`)が在れば
    `check_dirty_outdir()` が起動時に RuntimeError を出す。理由は 2 つとも「静かに
    壊れる」型である:
      - finalize は `glob("<stem>.part-*.parquet")` を無条件に結合するので、前回の
        part が残っていると**別の世界のイベントが今回の canonical に混ざる**。
      - `llm_cache.jsonl` は起動時に丸ごと読み戻されるので、新しいランのつもりが
        前回の応答の再生になり得る。
    逃し弁は `run.allow_dirty_outdir=true`(既定 false)。これも起動口だけの制御フラグで
    世界は 1 度も読まない(`run.seed_auto` / `run.allow_mock_production` と同じ族)。
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

#: β6: 「本番規模」とみなす名目エージェント数の下限(監査 E0-4 の推奨線)。
MOCK_PRODUCTION_MIN_AGENTS = 10_000

#: A7(第118 レーンC): 「この run dir は既に使われている」ことを示す痕跡の glob。
#: どれも **fresh 起動で踏むと出力が壊れる** ものだけを挙げる:
#:   - `*.part-*.parquet` … finalize は `glob(f"{stem}.part-*.parquet")` を**無条件に**
#:     結合する(observer/finalize.py)。前回ランの part が残っていれば、別の世界の
#:     イベントが今回の canonical へ**そのまま混入**する(検出手段は無い)。
#:   - `l1_events.parquet` … 完走済みの成果物。fresh 起動は黙って上書きする。
#:   - `checkpoint/*.pkl.gz` … 前回の世代。--resume の意図だった可能性が高い。
#:   - `llm_cache*.jsonl` … 起動時に丸ごと読み戻される(llm/cache.py)。前回ランの
#:     応答が今回の別プロンプトに当たることは無いが、**同じプロンプトには必ず当たる**
#:     ので「新しいランのつもりが前回の応答の再生」になり得る。
DIRTY_OUTDIR_GLOBS = (
    "*.part-*.parquet",
    "l1_events.parquet",
    "checkpoint/ckpt-*.pkl.gz",
    "checkpoint/dormant-*.pkl.gz",
    "llm_cache*.jsonl",
)


def resolve_out_dir(cfg) -> Path:
    """cfg から run dir を求める(Simulation.__init__ と**同じ式**。読むだけ)。"""
    from society.config import REPO_ROOT
    name = cfg.run.get("name", None) or f"seed{cfg.run.get('seed', '')}"
    return REPO_ROOT / str(cfg.run.get("out_dir", "runs")) / str(name)


def dirty_outdir_hits(out_dir: Path | str, limit: int = 6) -> list[str]:
    """`out_dir` にある「使用済みの痕跡」(相対パス)。無ければ空 list(純粋な読み取り)。"""
    root = Path(out_dir)
    if not root.is_dir():
        return []
    hits: list[str] = []
    for pat in DIRTY_OUTDIR_GLOBS:
        for p in sorted(root.glob(pat)):
            hits.append(p.relative_to(root).as_posix())
            if len(hits) >= limit:
                return hits
    return hits


def check_dirty_outdir(cfg) -> None:
    """非 resume 起動が**既に使われている run dir** を踏むのを拒否する(A7)。

    同名 (`run.name`) の再実行は、前回の `*.part-*.parquet` を finalize が黙って
    結合し、前回の `llm_cache.jsonl` を再生する = **2 つのランが混ざった成果物**を
    作る。しかも L1 は完全に正常な形をしているので事後に気づけない。ここで落とす。

    逃し弁は `run.allow_dirty_outdir=true`(「同じ dir へ重ねて書くのが目的」の明示)。
    ★シム本体はこのキーを 1 度も読まない(`run.seed_auto` /
      `run.allow_mock_production` と同じ **起動口だけの制御フラグ**)= ON/OFF で
      世界も L1 も 1 バイトも変わらない。conf 未搭載なので既定は false。
    """
    if bool(cfg.run.get("allow_dirty_outdir", False)):
        return
    out_dir = resolve_out_dir(cfg)
    hits = dirty_outdir_hits(out_dir)
    if hits:
        raise RuntimeError(
            f"出力先 {out_dir} は既に使われています(前回ランの痕跡: "
            f"{', '.join(hits)})。このまま新規起動すると、前回の part parquet が"
            " finalize で今回の成果物へ混入し、前回の llm_cache が再生されます。\n"
            "  - 続きを回すつもりなら: python scripts/run.py --resume "
            f"{out_dir}\n"
            "  - 別のランなら: run.name=<別の名前> を指定してください。\n"
            "  - 承知の上で重ねて書くなら: run.allow_dirty_outdir=true")


def check_mock_production(cfg) -> None:
    """本番規模の mock ラン(= LLM が 1 本も走らないラン)を起動時に拒否する(β6)。

    逃し弁は `run.allow_mock_production=true`(スケールスモーク・性能計測など「mock で回すこと
    自体が目的」のラン)。判定材料は conf の 3 キーだけで、世界の状態も乱数も 1 ビットも触らない。
    """
    n_agents = int(cfg.run.get("n_agents", 0) or 0)
    backend = str((cfg.get("model", {}) or {}).get("backend", "") or "")
    if (n_agents >= MOCK_PRODUCTION_MIN_AGENTS and backend == "mock"
            and not bool(cfg.run.get("allow_mock_production", False))):
        raise RuntimeError(
            f"本番規模({n_agents:,} 体 >= {MOCK_PRODUCTION_MIN_AGENTS:,})なのに "
            "model.backend=mock です(LLM が 1 本も走らないランになります)。"
            " 実 LLM で回すなら model.backend=vllm と model.servers=[...] を指定し"
            "(例: --profile conf/profiles/finals-vllm7.yaml)、"
            " mock で回すことが目的なら run.allow_mock_production=true を明示してください。")


def banner_lines(cfg) -> list[str]:
    """起動バナー(β6)。**読むだけ**の純関数=世界にも乱数にも触らない。"""
    model = cfg.get("model", {}) or {}
    pool = cfg.get("pool", {}) or {}
    lod = cfg.get("lod", {}) or {}
    servers = list(model.get("servers", []) or [])
    return [
        f"[launch] backend={model.get('backend', '')}"
        f" model={model.get('name', '')}"
        f" servers={len(servers)}"
        f" n_agents={int(cfg.run.get('n_agents', 0) or 0)}"
        f" n_steps={int(cfg.run.get('n_steps', 0) or 0)}"
        f" seed={cfg.run.get('seed', '')}",
        f"[launch] pool={'on' if bool(pool.get('enabled', False)) else 'off'}"
        f" pool_dir={pool.get('dir', '')}"
        f" present_cap={int(pool.get('present_cap', 0) or 0)}"
        f" max_llm_per_step={int(lod.get('max_llm_per_step', 0) or 0)}",
    ]


def print_banner(cfg) -> None:
    """起動バナーを stdout へ 1 回だけ出す(既存の `[seed]` 行と同じ様式)。"""
    for line in banner_lines(cfg):
        print(line)


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
        check_mock_production(cfg)                 # β6: resume でも同じ条件で守る
        print_banner(cfg)
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
    check_mock_production(cfg)                     # β6: 起動ガード(構築の前に落とす)
    check_dirty_outdir(cfg)                        # A7: 使用済み run dir への fresh 起動を拒否
    print_banner(cfg)
    sim = Simulation(cfg)
    summary = sim.run()
    if seed_auto:
        _record_seed_in_summary(Path(summary["out_dir"]), int(cfg.run.seed))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
