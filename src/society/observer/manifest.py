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


@lru_cache(maxsize=1)
def metrics_spec() -> dict:
    """指標定義コードの凍結ハッシュ(第78バッチ・Part G)。プロセス内で 1 回だけ計算する。

    「指標コードが 1 文字でも変われば hash が変わる」ことが要件(受入基準 T7)。
    対象ファイルの列挙は observer/metrics_spec.py の SPEC_FILES が唯一の源。
    """
    from . import metrics_spec as _spec
    return _spec.compute()


# ------------------------------------------------------- G1 入力データの来歴(sha256)
# 塞ぐ穴(docs/plans/metaverse-projection-plan.md §1「運用の穴」): weather / cognition の
# 凍結入力には sha256 が残るのに、**ラン最大の入力**である
#   ① ペルソナプール(data/persona_pool。100 万人の素性・agent_id → ペルソナの逆引きの源)
#   ② 市街地図(world.map)
#   ③ 組織台帳(organizations.file)
# の来歴は 1 バイトも残っていなかった。事後(復元実験・再現)に「どのプールで回したのか」を
# 確定できないと、agent_id からペルソナへ戻る決定論の逆引きが**原理的に検証不能**になる。
#
# 設計:
#   - **ストリーミング**(64 KB チャンク)で読む。プールは 733 MB あるので全読み込みは不可。
#   - ディレクトリはシャード構成なので、**ファイル 1 枚ごとの sha256** と、
#     `<相対パス>:<sha256>\n` を連ねた正準文字列の sha256(= 結合ハッシュ)の両方を出す。
#     並びは相対パスの昇順で固定(OS の走査順が結果に漏れない)。相対パスは posix 表記に
#     揃える(Windows の `\` が結合ハッシュに出ないようにする)。
#   - **読むだけ・書く経路なし**。値は run_manifest.json にしか現れない。
#   - 既定 OFF(`observer.input_provenance.enabled`)。ON でも起動が数秒延びるだけだが、
#     「既存ランの manifest と同形」を守るためキー自体を出さない側に倒す。
_HASH_CHUNK = 1 << 16
#: ディレクトリを走査するとき無視するもの(生成物・キャッシュ = 入力ではない)。
_SKIP_DIR_PARTS = ("__pycache__", ".git")


def file_sha256(path: str | Path) -> str:
    """ファイルの生バイト列の SHA-256(ストリーミング。巨大ファイルでも定数メモリ)。"""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_HASH_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def path_sha256(path: str | Path) -> dict | None:
    """ファイル 1 枚 / ディレクトリ 1 本の来歴 dict(存在しなければ None)。

    ディレクトリは ``{"kind": "dir", "sha256": 結合ハッシュ, "n_files", "bytes", "files": {...}}``、
    ファイルは ``{"kind": "file", "sha256", "bytes"}``。**欠測は None**(偽の値で埋めない)。
    """
    p = Path(path)
    if p.is_file():
        return {"kind": "file", "sha256": file_sha256(p), "bytes": p.stat().st_size}
    if not p.is_dir():
        return None
    files: dict[str, str] = {}
    total = 0
    for child in sorted(p.rglob("*"), key=lambda q: q.as_posix()):
        if not child.is_file():
            continue
        rel = child.relative_to(p).as_posix()
        if any(part in _SKIP_DIR_PARTS for part in child.relative_to(p).parts):
            continue
        files[rel] = file_sha256(child)
        total += child.stat().st_size
    blob = "".join(f"{rel}:{sha}\n" for rel, sha in sorted(files.items()))
    return {"kind": "dir",
            "sha256": hashlib.sha256(blob.encode("utf-8")).hexdigest(),
            "n_files": len(files), "bytes": total, "files": files}


def input_provenance(sim) -> dict | None:
    """G1: プール / 地図 / 組織台帳の sha256(既定 OFF では **None** = キー自体を出さない)。"""
    cfg = sim.cfg
    obs = (cfg.get("observer", {}) or {})
    raw = obs.get("input_provenance", None)
    if not (raw is not None and hasattr(raw, "get") and bool(raw.get("enabled", False))):
        return None
    from ..config import REPO_ROOT

    def _resolve(value) -> Path | None:
        text = str(value or "").strip()
        if not text:
            return None
        p = Path(text)
        return p if p.is_absolute() else Path(REPO_ROOT) / p

    world = cfg.get("world", {}) or {}
    pool = cfg.get("pool", {}) or {}
    orgs = cfg.get("organizations", {}) or {}
    sources = {
        # ★プールは pool.enabled のときだけハッシュする(実プールは 733MB あり、1 バイトも
        #   使われないランで数秒を払う理由が無い)。OFF のランは path だけを "unused" として
        #   残す = 「採らなかった」と「実体が無かった」を事後に区別できる。
        "persona_pool": (_resolve(pool.get("dir", ""))
                         if bool(pool.get("enabled", False)) else None),
        "map": _resolve(world.get("map", "")),
        "org_ledger": _resolve(orgs.get("book", "")),
    }
    out: dict = {"schema": 1}
    if sources["persona_pool"] is None and str(pool.get("dir", "") or ""):
        out["persona_pool"] = {"kind": "unused", "path": str(pool.get("dir"))}
    for name, path in sources.items():
        if path is None:
            continue
        got = path_sha256(path)
        # 欠測(config が指すのに実体が無い)は "absent" と判る形で残す。
        out[name] = got if got is not None else {"kind": "absent"}
        out[name]["path"] = str(path)
    return out if len(out) > 1 else None


# ------------------------------------------------- β10 起動側申告(モデル・サンプリング凍結)
# 塞ぐ穴(docs/plans/external-audit-triage.md F15 / beta-implementation-plan.md §1 β10):
# manifest の `model` ブロックは conf の写しであって「実際にどの重みへ、どのサンプリングで
# 投げたか」を主張していない。特に vLLM は `--generation-config auto`(既定)のとき
# **モデル同梱の generation_config.json** を未指定パラメータの既定に使うので、
# 「conf を 1 文字も変えていないのに分布が変わる」経路が起動側に開いている。
#
# 設計(★捏造しない):
#   - 本ブロックは **起動側の申告**であって、稼働中のサーバから取得した値ではない。
#     そのことを `declared_by: "launcher"` と `verified: false` で明示する。
#   - client が **送っていない** サンプリングパラメータ(top_p / top_k)は **null**。
#     「送っていない」と「0 だった」を混同させない(欠測は欠測と判る値にする)。
#   - vLLM の版は **このプロセスから見える vllm パッケージ**の版だけを報告する。
#     取れなければ null(リモートのサーバ版を推測で書かない)。
_SAMPLING_NOT_SENT = ("top_p", "top_k")


@lru_cache(maxsize=1)
def local_vllm_version() -> str | None:
    """このプロセスから import できる vllm パッケージの版(取れなければ None)。

    ★これは「起動スクリプトと同じ機で動いている vLLM の版」に一致する**かもしれない**値で
    あって、リモートのサーバ版の保証ではない。取れない環境(ローカル Windows・別ノード)では
    None を返し、run_manifest には null が入る(推測で埋めない)。
    """
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:                                   # pragma: no cover (py<3.8)
        return None
    try:
        return str(version("vllm"))
    except (PackageNotFoundError, ValueError, OSError):   # 未インストール等
        return None


def launch_declaration(sim) -> dict:
    """β10: model / backend / sampling / vLLM 版の「起動側申告」欄。"""
    cfg = sim.cfg
    model = cfg.get("model", {}) or {}
    backend = str(model.get("backend", ""))
    servers = [str(s) for s in (model.get("servers", []) or [])]
    rs = ((cfg.get("llm", {}) or {}).get("request_seed", {}) or {})
    seed_on = bool(rs.get("enabled", False))
    sampling: dict = {
        # client が送出ボディへ必ず載せる値(src/society/llm/vllm.py の _completions_body)
        "temperature": float(model.get("temperature", 0.0)),
        "max_tokens": int(model.get("max_tokens", 0)),
        "reflect_max_tokens": int(model.get("reflect_max_tokens", 0)),
        "plan_max_tokens": int(model.get("plan_max_tokens", 0) or 0),
        "response_format": ("json_object" if str(model.get("format", "json")) == "json"
                            else None),
        # β11: 送るなら方式を明記する(値は呼ごとに変わるので式だけを残す)
        "seed": ("blake2b(run_seed, agent_id, step, purpose, ordinal) & 0x7fffffff"
                 if seed_on else None),
        "seed_enabled": seed_on,
        "seed_run_seed": (int(cfg.run.seed) if seed_on else None),
    }
    for key in _SAMPLING_NOT_SENT:      # ★client は送らない = サーバ既定に従う(null で残す)
        sampling[key] = None
    return {
        "schema": 1,
        # ★この欄は「起動側がそう申告した」という記録であって、稼働サーバからの取得値ではない。
        "declared_by": "launcher",
        "verified": False,
        "backend": backend,
        "model": str(model.get("name", "")),
        "n_servers": len(servers),
        "servers": servers,
        # vLLM 経路以外(mock / ollama / router / API)では版の概念が無いので null。
        "vllm_version": (local_vllm_version() if backend == "vllm" else None),
        # 未送出のパラメータがサーバ側の何に従うか(起動フラグ)は取得できないので申告しない。
        # ops/launch-vllm-finals.ps1 が `--generation-config vllm` を付ける規約になっている。
        "sampling": sampling,
    }


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
    from .. import ablate as _ablate_mod
    from .. import mind as _mind_mod
    from .. import weather as _weather_mod
    from ..cognition import calib as _calib_mod
    from ..config import REPO_ROOT
    from ..engine import checkpoint
    from . import regression as _regression_mod

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
    _spec = metrics_spec()

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
        # 第78バッチ Part G: 指標定義コードの凍結ハッシュ。事後に指標をいじると値が変わる。
        #   metrics_spec_hash … 対象ファイル群 + **ファイルリスト自体**の正規化ハッシュ
        #   metrics_spec      … 内訳(どのファイルのどの hash を採ったか)+ 欠測一覧
        "metrics_spec_hash": _spec["metrics_spec_hash"],
        "metrics_spec": {"schema": _spec["schema"], "n_files": _spec["n_files"],
                         "files": _spec["files"], "missing": _spec["missing"]},
        # 第72バッチ: ランモード(none/observe/journal/verify)。比較ガードが最初に見る場所。
        "run_mode": features["run_mode"],
        "run": {
            "mode": features["run_mode"],
            "seed": int(cfg.run.seed),
            "seed_auto": bool(cfg.run.get("seed_auto", False)),
            "n_agents": int(cfg.run.n_agents),
            "n_steps": int(cfg.run.n_steps),
            # 第101: Δt の第2の源(scripts/run_dt.py が config.yaml の次に読む)。
            "dt_min": int(cfg.run.get("dt_min", 10)),
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
        # β10(第117): モデル・サンプリング凍結の「起動側申告」欄。conf の写しである
        # `model` とは別に、**client が実際に送るサンプリング**と、送っていない
        # パラメータ(null)、vLLM の版(取れなければ null)を 1 箇所に集めて残す。
        "launch": launch_declaration(sim),
        "world": {
            "map": str((cfg.get("world", {}) or {}).get("map", "")),
            "scenario": str((cfg.get("world", {}) or {}).get("scenario", "")),
            "mod": (sim.worldmod.summary() if getattr(sim, "worldmod", None) else None),
        },
        "k": {"writeback": str(cfg.get("k", {}).get("writeback", "free"))},
        "controls": {"mode": str(cfg.get("controls", {}).get("mode", "none"))},
        # 第78バッチ: アブレーション 4 種。**全 OFF のランではキー自体を出さない**
        # (既存ランの manifest と同形を保つ)。ON のときは cognitive_tier が実際に
        # 効いたか(fleet 非使用ランでの縮退)も tier_effective に正直に残す。
        **({"ablate": _ablate} if (_ablate := _ablate_mod.describe(sim)) else {}),
        # 第80バッチ W2: 天候の入力データ来歴(weather.mode=generated / table のときだけ)。
        # weather_params_sha256 / weather_source_sha256 + 実ファイルの sha256 と出典表示。
        # 既定 synthetic ではキー自体を出さない(既存 manifest と同形)。
        **({"weather": _wea} if (_wea := _weather_mod.provenance(sim)) else {}),
        # 第80バッチ: 認知の凍結入力の来歴(観測チャンネル ON のときだけ)。
        # チャンネル定義 hash + 較正テーブル sha256 + σ_c 凍結 sha256(未生成なら absent)。
        # 既定 OFF ではキー自体を出さない(既存 manifest と同形=天候来歴と同じ流儀)。
        **({"cognition": _cog} if (_cog := _calib_mod.provenance(sim)) else {}),
        # 第88バッチ: 心モデル固定(1 体 1 モデル)と三層知能配置。既定 OFF ではキー自体を
        # 出さない(既存 manifest と同形=天候・認知来歴と同じ流儀)。原文書 §5 が要求する
        # 「agent_id と model_id の対応を必ずログに残す」の**来歴側**(個別対応は
        # agents.json と L1 mind_assign にある)。
        **({"mind": _mind} if (_mind := _mind_mod.provenance(sim)) else {}),
        # 第91バッチ: 退行シグナル監視の仕様(窓幅・n-gram・除外の有無・実際に出た列名)。
        # 既定 OFF ではキー自体を出さない(既存 manifest と同形)。列名を manifest に残すのは
        # 「fire OFF のランには発火率 5 列が無い」を事後に判別できるようにするため。
        **({"regression": _reg}
           if (_reg := _regression_mod.provenance(sim)) else {}),
        # G1(第114 GT ロガー): ラン最大の入力 3 件(ペルソナプール / 地図 / 組織台帳)の
        # sha256。既定 OFF ではキー自体を出さない(天候・認知来歴と同じ流儀)。
        **({"inputs": _inp} if (_inp := input_provenance(sim)) else {}),
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
