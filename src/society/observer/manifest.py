"""run_manifest.json(第71バッチ 2026-07-31)= ラン1本の来歴を1ファイルに固定する。

目的(docs/plans/source/dual-mode-requirements.md §4「run manifest に必ず含める項目」):
  コード git SHA / 設定ハッシュ / run_seed / モデル ID / 実行モード(cache_mode)/
  全スイッチ状態 / 開始時刻 / 開始日。
これが無いと「この runs/ ディレクトリは、どのコードのどの設定で回したのか」が事後に
確定できない。後続バッチ(第72 機能レジストリ・第78 metrics_spec_hash)はここへ足していく。

原則
----
- **書くだけ・読む経路なし**: シム本体は run_manifest.json を読まない。読まない限り
  決定論にも LLM 呼数にも影響しえない(R1)。config 自体は従来どおり
  `config.yaml`(save_config)が正本で、manifest はその**ハッシュ**と要約を持つ。
- **壁時計の扱い**: `started_at` は観測メタデータであって世界状態ではない。
  シム時刻は従来どおり Clock(sim_min)だけが供給する。ここで `datetime.now()` という
  綴りを避けて `time.time()` + `fromtimestamp` を使っているのは、
  「コアコードに実時刻取得を置かない」T8 の静的検査(第78 予定)が
  observer 層のこの 1 箇所を機械的に区別できるようにするため。
- **git 取得の失敗は "unknown"**: リポジトリ外への配布・git 未導入でもランは落とさない
  (欠測を偽の値で埋めない=欠測と判る値にする)。プロセス内で 1 回だけ実行(lru_cache)。
"""
from __future__ import annotations

import datetime
import hashlib
import json
import platform
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path

from omegaconf import OmegaConf

SCHEMA = 1
FILENAME = "run_manifest.json"
_HISTORY_CAP = 20            # 同一 out_dir を再利用/resume したときに残す過去エントリ数


# --------------------------------------------------------------------- git
@lru_cache(maxsize=8)
def _git_raw(repo_root: str) -> tuple[str, str, str]:
    """(sha, branch, dirty) を返す。取得できない要素は "unknown"。"""
    def _run(args: list[str]) -> str | None:
        try:
            p = subprocess.run(["git", *args], cwd=repo_root, capture_output=True,
                               text=True, timeout=20, encoding="utf-8",
                               errors="replace")
        except (OSError, subprocess.SubprocessError):
            return None
        if p.returncode != 0:
            return None
        return p.stdout.strip()

    sha = _run(["rev-parse", "HEAD"]) or "unknown"
    branch = _run(["rev-parse", "--abbrev-ref", "HEAD"]) or "unknown"
    status = _run(["status", "--porcelain"])
    dirty = "unknown" if status is None else ("true" if status else "false")
    return sha, branch, dirty


def git_info(repo_root: str | Path) -> dict:
    sha, branch, dirty = _git_raw(str(repo_root))
    return {"sha": sha, "branch": branch, "dirty": dirty}


# ------------------------------------------------------------------ ハッシュ
def canonical_config_json(cfg) -> str:
    """resolved config の正準 JSON(キー順固定・非 JSON 型は str 化)。"""
    data = cfg if isinstance(cfg, dict) else OmegaConf.to_container(cfg, resolve=True)
    return json.dumps(data, sort_keys=True, ensure_ascii=False, default=str)


def config_sha256(cfg) -> str:
    return hashlib.sha256(canonical_config_json(cfg).encode("utf-8")).hexdigest()


def event_schema_sha256() -> str:
    """L1 イベント種別レジストリ(observer/schema.py)の正準ハッシュ。

    ログのスキーマが変わったランを機械的に区別するための既存値。
    """
    from .schema import EVENT_KINDS
    blob = json.dumps(sorted(EVENT_KINDS.keys()), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def collect_toggles(cfg) -> dict:
    """resolved config の**真偽値リーフを全部**ドット記法で平坦化する。

    「全スイッチ状態」を人手の列挙ではなく機械的に採る(登録漏れが原理的に起きない)。
    第72 の機能レジストリ(repro_tier 宣言)はこの一覧を突き合わせ相手にする
    (定義を二重に持たないよう registry.flatten_bools へ委譲した)。
    """
    from ..registry import flatten_bools
    return flatten_bools(cfg)


# --------------------------------------------------------------------- 構築
def build(sim) -> dict:
    """sim から manifest dict を組む(副作用なし・純関数的)。"""
    from ..config import REPO_ROOT
    from ..engine import checkpoint

    from ..registry import describe as _describe_features

    cfg = sim.cfg
    # resolve は 1 回だけ(config は 1,300 行超あり、to_container を 3 回回すと
    # Simulation 構築 1 回あたり 10ms 級の無駄になる)。以降は resolved dict を使い回す。
    resolved = OmegaConf.to_container(cfg, resolve=True)
    # 第72バッチ: ランモードと機能レジストリの報告。Simulation が構築時に作った報告
    # (自動 OFF の明細を含む)を使い、無い場合(素の sim スタブ等)は config から作り直す。
    features = getattr(sim, "run_mode_report", None) or _describe_features(resolved)
    model = cfg.get("model", {}) or {}
    started = time.time()
    calendar = (cfg.get("world", {}) or {}).get("calendar", {}) or {}

    man = {
        "schema": SCHEMA,
        # 観測メタデータ(世界状態ではない)。ローカルタイムゾーン付き ISO8601。
        "started_at": datetime.datetime.fromtimestamp(started)
                              .astimezone().isoformat(timespec="seconds"),
        "started_at_epoch": round(started, 3),
        "git": git_info(REPO_ROOT),
        "config_sha256": config_sha256(resolved),
        # resume 可否判定に使われる方(揮発キー除外)。checkpoint と同じ定義を再利用する。
        "config_determinism_sha256": checkpoint.config_hash_from_container(resolved),
        "event_schema_sha256": event_schema_sha256(),
        # 第72バッチ: ランモード(none/observe/journal/verify)。比較ガードが最初に見る場所。
        "run_mode": features["run_mode"],
        "run": {
            "mode": features["run_mode"],
            "seed": int(cfg.run.seed),
            "seed_auto": bool(cfg.run.get("seed_auto", False)),
            "n_agents": int(cfg.run.n_agents),
            "n_steps": int(cfg.run.n_steps),
            "name": (str(cfg.run.name) if cfg.run.get("name") else None),
            "start_tod": str(cfg.run.get("start_tod", "07:00")),
            "start_date": (str(calendar.get("start_date"))
                           if calendar.get("start_date") else None),
            "out_dir": str(sim.out_dir),
        },
        "model": {
            "backend": str(model.get("backend", "")),
            "name": str(model.get("name", "")),
            "cache": bool(model.get("cache", True)),
            "cache_mode": str(model.get("cache_mode", "free")),
            "journal": bool(model.get("journal", True)),
            "temperature": float(model.get("temperature", 0.0)),
            "max_tokens": int(model.get("max_tokens", 0)),
            "reflect_max_tokens": int(model.get("reflect_max_tokens", 0)),
            "reflect_think": bool(model.get("reflect_think", False)),
            "format": str(model.get("format", "json")),
            "servers": [str(s) for s in (model.get("servers", []) or [])],
        },
        "world": {
            "map": str((cfg.get("world", {}) or {}).get("map", "")),
            "scenario": str((cfg.get("world", {}) or {}).get("scenario", "")),
            "mod": (sim.worldmod.summary() if getattr(sim, "worldmod", None) else None),
        },
        "k": {"writeback": str(cfg.get("k", {}).get("writeback", "free"))},
        "controls": {"mode": str(cfg.get("controls", {}).get("mode", "none"))},
        "code": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "files": {
            "config": "config.yaml",
            "llm_journal": sorted(j.path.name for j in getattr(sim, "_journals", [])),
        },
        # 「全スイッチ状態」= resolved config の真偽値リーフ全部(機械採取)。
        "toggles": collect_toggles(resolved),
        # 第72バッチ 機能レジストリ: 有効な機能とその再現性等級 / モードで自動 OFF にした一覧 /
        # まだ宣言されていない bool キー(通常は空。CI が空を固定する)。
        #   enabled      … [{id, repro_tier, affects_k, fingerprint_risk}]
        #   auto_disabled… [{id, repro_tier, was, now, explicit, reason}]
        # これが「observe ランと verify ランを無自覚に並べる」事故を事後に検出する material。
        "features": features,
    }
    return man


def write(sim) -> Path | None:
    """out_dir/run_manifest.json を書く(observer.run_manifest=false なら何もしない)。

    同じ out_dir で作り直された場合(resume / 追い回し)は、前回の要点を `history` に
    畳んでから上書きする(来歴を失わない)。
    """
    cfg = sim.cfg
    if not bool((cfg.get("observer", {}) or {}).get("run_manifest", True)):
        return None
    out_dir = Path(sim.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / FILENAME
    man = build(sim)
    prev = None
    try:
        prev = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        prev = None
    if isinstance(prev, dict):
        hist = list(prev.get("history") or [])
        hist.append({"started_at": prev.get("started_at"),
                     "git_sha": (prev.get("git") or {}).get("sha"),
                     "config_sha256": prev.get("config_sha256"),
                     "n_steps": (prev.get("run") or {}).get("n_steps")})
        man["history"] = hist[-_HISTORY_CAP:]
    path.write_text(json.dumps(man, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return path
