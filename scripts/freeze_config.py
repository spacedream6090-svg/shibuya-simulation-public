#!/usr/bin/env python
"""β6: 本番 conf を **完全解決済みの 1 枚**へ凍結する(第117バッチ 2026-08-16)。

なぜ要るか(監査 E0-4 / docs/plans/external-audit-triage.md F5)
--------------------------------------------------------------
本選の起動は「基底 conf/config.yaml < env < --profile < CLI dotlist」の 4 段重ねで決まる。
つまり **実際に何で回したか**は、起動コマンドの 1 行を人間が正確に覚えていない限り事後に
再構成できない(profile 単独実行だと基底の `model.backend: mock` が生き残る、という事故は
まさにこの重ね書きの帰結)。本スクリプトは起動前にその 4 段を**解決し切って** 1 枚の YAML
へ落とし、sha256 を添える。以降は

    python scripts/run.py --profile conf/finals_YYYYMMDD_frozen.yaml

の 1 行で「その sha256 の条件」を再現できる(dotlist を 1 つも打たない = 打ち間違いようがない)。

使い方
------
    python scripts/freeze_config.py --profile conf/finals_observe.yaml \
        model.backend=vllm run.n_agents=250000
    python scripts/freeze_config.py --profile conf/finals_observe.yaml --out /tmp/x.yaml
    python scripts/freeze_config.py --profile conf/finals_observe.yaml --print   # 標準出力のみ

    既定の出力先は `conf/finals_<YYYYMMDD>_frozen.yaml`(+ 同名 `.sha256`)。
    `--date` で日付部分を差し替えられる。

生成物の扱い(★重要)
--------------------
**生成された `conf/finals_*_frozen.yaml` と `.sha256` はコミットしない。** 本番直前に生成し、
sha256 を run_manifest / 進捗報告と突き合わせるための**運用成果物**であって、リポジトリの
正典(基底 conf + profile)ではない。二重管理すると「どちらが正か」が失われる。
(公開ミラーの除外域でもないので、コミットするとそのまま公開される点にも注意。)

Δt の扱い(第94バッチ OBS-U2 と同じ落とし穴)
--------------------------------------------
`load_config` は末尾で `timeconv.apply_dt` を通す = 出力される YAML は **apply_dt 済み**なのに
`run.dt_min` の値をそのまま保持している。したがって凍結ファイルを読み直すときは:

  - `python scripts/run.py --profile <frozen>` … 基底へ重ねてから **もう一度** apply_dt が走る。
    `run.dt_min=10`(正準)では apply_dt が恒等パスなので **1 バイトも変わらない**が、
    Δt≠10 のランを凍結した場合は二重変換になる。**Δt≠10 の凍結ファイルは
    `--profile` で使わないこと**(そのときは run dir の config.yaml と同じ扱い =
    `load_config(path=..., apply_dt=False)` 相当の経路が要る)。
  - 検証(再解決一致)は `load_config(path=<frozen>, apply_dt=False)` で読む。
    tests/test_freeze_config.py がこの同値性を機械固定している。

本スクリプトは **読むだけ**で、シムも世界も 1 ビットも触らない(R1 構造安全)。
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omegaconf import OmegaConf                  # noqa: E402

from society.config import REPO_ROOT, load_config  # noqa: E402


def resolve_yaml(profile: str | Path | None = None,
                 env: str | Path | None = None,
                 overrides: list[str] | None = None) -> str:
    """基底 + env + profile + dotlist を解決し切った YAML テキストを返す(副作用なし)。"""
    cfg = load_config(overrides=list(overrides or []), profile=profile, env=env)
    return OmegaConf.to_yaml(cfg, resolve=True)


def sha256_text(text: str) -> str:
    """凍結 YAML の内容ハッシュ(utf-8 のバイト列に対して採る)。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def default_out_path(date: str | None = None) -> Path:
    """既定の出力先 `conf/finals_<YYYYMMDD>_frozen.yaml`。"""
    tag = date or datetime.date.today().strftime("%Y%m%d")
    return Path(REPO_ROOT) / "conf" / f"finals_{tag}_frozen.yaml"


def freeze(profile: str | Path | None = None,
           env: str | Path | None = None,
           overrides: list[str] | None = None,
           out: str | Path | None = None,
           date: str | None = None) -> dict:
    """解決 → 書き出し(+ .sha256)。戻り値は {"path", "sha256_path", "sha256", "bytes"}。"""
    text = resolve_yaml(profile=profile, env=env, overrides=overrides)
    digest = sha256_text(text)
    path = Path(out) if out else default_out_path(date)
    path.parent.mkdir(parents=True, exist_ok=True)
    # ★`write_text` は使わない: Windows では "\n" が "\r\n" へ翻訳され、ディスク上の
    #   バイト列が sha256 を採った文字列と食い違う(= 同じ conf なのに開発機と本選機で
    #   ハッシュが変わる)。バイト列を直接書いて LF 固定にする。
    path.write_bytes(text.encode("utf-8"))
    sha_path = path.with_name(path.name + ".sha256")
    # `sha256sum -c` と同じ 1 行形式(ファイル名は basename = 置き場所に依存しない)。
    sha_path.write_bytes(f"{digest}  {path.name}\n".encode("utf-8"))
    return {"path": path, "sha256_path": sha_path, "sha256": digest,
            "bytes": len(text.encode("utf-8"))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="本番 conf を完全解決済みの 1 枚へ凍結する(β6)。生成物はコミットしない。")
    ap.add_argument("--profile", default=None, help="重ねる profile YAML(例 conf/finals_observe.yaml)")
    ap.add_argument("--env", default=None, help="EnvPack(env/<place> か env.yaml)")
    ap.add_argument("--out", default=None, help="出力先(既定 conf/finals_<YYYYMMDD>_frozen.yaml)")
    ap.add_argument("--date", default=None, help="既定出力先の日付部分(YYYYMMDD)")
    ap.add_argument("--print", dest="to_stdout", action="store_true",
                    help="ファイルを書かず、解決済み YAML と sha256 を標準出力へ出すだけ")
    ap.add_argument("overrides", nargs="*", help="dotlist(例 model.backend=vllm run.n_agents=250000)")
    ns = ap.parse_args(list(argv) if argv is not None else None)

    if ns.to_stdout:
        text = resolve_yaml(profile=ns.profile, env=ns.env, overrides=ns.overrides)
        sys.stdout.write(text)
        print(f"[freeze_config] sha256={sha256_text(text)}")
        return 0

    rep = freeze(profile=ns.profile, env=ns.env, overrides=ns.overrides,
                 out=ns.out, date=ns.date)
    print(f"[freeze_config] wrote {rep['path']} ({rep['bytes']:,} bytes)")
    print(f"[freeze_config] sha256={rep['sha256']}  -> {rep['sha256_path']}")
    print("[freeze_config] ★この生成物はコミットしない(本番直前に生成する運用成果物)。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
