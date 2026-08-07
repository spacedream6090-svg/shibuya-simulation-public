#!/usr/bin/env python
"""W2-3: 解析スクリプトが **ランの Δt(1 step の分数)** を読むための単一の源。

なぜ要るか
----------
`docs/research/obs-u2-dt1min-design.md` §1.3(C 級)の指摘そのもの:
`STEPS_PER_DAY = 144` / `MIN_PER_STEP = 10` をモジュール定数として持つ解析が 30 本ある。
OBS-U2 の並行小ラン(`conf/smoke_dt1.yaml`・`run.dt_min=1`)は **Δt=10 ランとの比較**が
目的なので、解析側が 144 を直書きしている限り「1 日」の意味がランごとに 10 倍ずれる。

シム側の Δt はすでに 1 箇所(`conf/config.yaml: run.dt_min` → `src/society/timeconv.py`)に
集約されており、派生量の単一源は `src/society/world/clock.py`。**本モジュールはその解析側の
対**であって、変換の流儀(1440 の約数・逆比例・正準 Δt=10)は timeconv/clock に従う。

読む順序(= 黙って仮定しないための規約)
----------------------------------------
1. `<run_dir>/config.yaml` の `run.dt_min` … `society.config.save_config` が必ず書く
   スナップショット。第79バッチ以降のランはここに入っている(**正**)。
2. `<run_dir>/run_manifest.json` の `run.dt_min` … 現行の manifest は `run` ブロックに
   dt_min を持たない(`src/society/observer/manifest.py:155-166`)ので通常は空振りする。
   将来 manifest 側に載ったときに自動で拾えるよう順序だけ用意してある(観測専用・読むだけ)。
3. どちらにも無ければ **正準 Δt=10 と仮定し、その旨を stderr に 1 行出す**。
   第79バッチ以前のランは config.yaml に dt_min を持たない(実測 178 本中 173 本)ので
   この経路が普通に踏まれる。**黙って 144 を使わない**ことが本モジュールの存在理由。

出力バイト同値の約束(移行の絶対条件)
--------------------------------------
- Δt=10 では `steps_per_day()` は **144**、`min_per_step()` は **10** を返す。既存ランに
  対する解析結果は 1 ビットも変わらない(`tests/test_run_dt.py` が固定)。
- 仮定の告知は **stderr のみ**。JSON / Markdown / CSV などの成果物には 1 バイトも足さない
  (成果物にキーを足すと「Δt=10 で同値」が壊れるため。告知先を分けたのは意図的な設計)。

依存は標準ライブラリ + PyYAML(omegaconf の依存として必ず入っている)のみ。副作用なし。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# 正準値(= golden の世界)。src/society/timeconv.py の CANON_DT_MIN / MINUTES_PER_DAY と
# 同じ値でなければならない。src は読むだけなので写経になるが、二重定義の腐敗は
# tests/test_run_dt.py::test_canonical_constants_match_src が実際に突き合わせて検知する。
# --------------------------------------------------------------------------- #
CANON_DT_MIN = 10
MINUTES_PER_DAY = 1440
CANON_STEPS_PER_DAY = 144          # = 1440 // 10
CANON_STEPS_PER_HOUR = 6           # = 60 // 10

SRC_CONFIG = "config.yaml"
SRC_MANIFEST = "run_manifest.json"
SRC_ASSUMED = "assumed"

# 同じランについて何度も同じ警告を出さない(解析 1 本が run_dir を何度も引くため)。
_NOTIFIED: set = set()


# --------------------------------------------------------------------------- #
# 妥当性(timeconv.dt_of と同じ規則)
# --------------------------------------------------------------------------- #
def valid_dt_min(v) -> int | None:
    """正の整数かつ 1440 の約数なら int を返す。それ以外は None(= 採用しない)。

    timeconv.dt_of と同じ規則(日境界が step 境界に落ちない Δt は世界の日次機構を壊す)。
    """
    if isinstance(v, bool) or v is None:
        return None
    try:
        dt = int(v)
    except (TypeError, ValueError):
        return None
    if dt <= 0 or MINUTES_PER_DAY % dt != 0:
        return None
    return dt


# --------------------------------------------------------------------------- #
# 読み取り(config.yaml → run_manifest.json → 仮定)
# --------------------------------------------------------------------------- #
def _dt_from_yaml_text(text: str) -> int | None:
    """`run:` ブロック直下の `dt_min:` を拾う。PyYAML があればそちらを優先する。

    フォールバックの素朴な行走査は detect_regression.run_window_steps と同じ流儀
    (保存済み config.yaml は OmegaConf.save が書くブロック形式・2 スペース字下げ)。
    """
    try:
        import yaml                                     # noqa: PLC0415
        doc = yaml.safe_load(text)
        if isinstance(doc, dict):
            run = doc.get("run")
            if isinstance(run, dict):
                return valid_dt_min(run.get("dt_min"))
            return None
    except Exception:                                   # noqa: BLE001
        pass
    in_run = False
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if not line.startswith((" ", "\t")):            # トップレベルのキー
            in_run = s.startswith("run:")
            continue
        if in_run and s.startswith("dt_min:"):
            return valid_dt_min(s.split(":", 1)[1].split("#", 1)[0].strip())
    return None


def dt_min_from_yaml(path) -> int | None:
    """conf の YAML ファイル 1 本から `run.dt_min` を拾う(無ければ None)。

    プロファイル(`conf/smoke_dt1.yaml` 等)や基底 `conf/config.yaml` を直接指すとき用。
    run dir を指すなら `read_dt_min` を使うこと(こちらはファイル 1 本だけを見る)。
    """
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return _dt_from_yaml_text(p.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError):
        return None


def _dt_from_manifest(doc) -> int | None:
    """manifest の中の dt_min を探す(現行は非搭載。将来の搭載に備えた読み取りのみ)。"""
    if not isinstance(doc, dict):
        return None
    for path in (("run", "dt_min"), ("dt_min",), ("config", "run", "dt_min")):
        node = doc
        for seg in path:
            node = node.get(seg) if isinstance(node, dict) else None
            if node is None:
                break
        dt = valid_dt_min(node)
        if dt is not None:
            return dt
    return None


def read_dt_min(run_dir) -> tuple[int, str]:
    """`(dt_min, source)` を返す。source は config.yaml / run_manifest.json / assumed。

    **読むだけ**。存在しない run_dir・壊れたファイルは静かに次の源へ落ちる
    (最後は必ず正準 10 + source="assumed" になるので例外は投げない)。
    """
    if run_dir is None:
        return CANON_DT_MIN, SRC_ASSUMED
    rd = Path(run_dir)
    cfg = rd / SRC_CONFIG
    if cfg.is_file():
        try:
            dt = _dt_from_yaml_text(cfg.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError):
            dt = None
        if dt is not None:
            return dt, SRC_CONFIG
    man = rd / SRC_MANIFEST
    if man.is_file():
        try:
            dt = _dt_from_manifest(json.loads(man.read_text(encoding="utf-8")))
        except (OSError, UnicodeDecodeError, ValueError):
            dt = None
        if dt is not None:
            return dt, SRC_MANIFEST
    return CANON_DT_MIN, SRC_ASSUMED


def dt_min_from_dotlist(overrides) -> int | None:
    """conf の dotlist(`["run.dt_min=1", ...]`)から Δt を拾う。無ければ None。

    ラン**を作る**側のツール(measure_sigma / calibrate_theta 等)は run dir がまだ無いので、
    「これから回すランの Δt」はここと基底 conf から決まる。後勝ち(最後の指定が有効)。
    """
    dt = None
    for item in (overrides or []):
        s = str(item).strip()
        if s.startswith("run.dt_min="):
            v = valid_dt_min(s.split("=", 1)[1].strip())
            if v is not None:
                dt = v
    return dt


def _notify(run_dir, source: str, stream=None) -> None:
    """仮定したことを stderr に 1 行だけ出す(同じランについては 1 回だけ)。

    ★成果物(JSON/Markdown)には**書かない**。書くと Δt=10 の既存出力とバイト同値でなくなる。
    """
    if source != SRC_ASSUMED:
        return
    key = str(Path(run_dir).resolve()) if run_dir is not None else "<none>"
    if key in _NOTIFIED:
        return
    _NOTIFIED.add(key)
    out = stream if stream is not None else sys.stderr
    where = key if run_dir is not None else "(run_dir 指定なし)"
    msg = (f"[run_dt] {where}: run.dt_min が config.yaml / run_manifest.json の"
           f"いずれにも無いため **Δt={CANON_DT_MIN} 分(正準)と仮定**しました"
           f"(1 日 = {CANON_STEPS_PER_DAY} step)。第79バッチ以前のランでは正常です。")
    try:
        print(msg, file=out)
    except UnicodeEncodeError:                          # Windows cp932 コンソール
        try:
            print(f"[run_dt] {where}: run.dt_min not found in config.yaml / "
                  f"run_manifest.json -- ASSUMING dt={CANON_DT_MIN} min "
                  f"({CANON_STEPS_PER_DAY} steps/day).", file=out)
        except Exception:                               # noqa: BLE001
            pass
    except Exception:                                   # noqa: BLE001 (閉じた stream 等)
        pass


def dt_min_of(run_dir, *, notify: bool = True, stream=None) -> int:
    """ランの Δt [分]。読めなければ正準 10 を返し、その旨を stderr に告知する。"""
    dt, source = read_dt_min(run_dir)
    if notify:
        _notify(run_dir, source, stream)
    return dt


def provenance(run_dir, *, notify: bool = True, stream=None) -> dict:
    """`{"dt_min", "source", "assumed", "steps_per_day", "steps_per_hour", "min_per_step"}`。

    「どこから読んだか」を呼び出し側が明示したいときに使う。**既存の成果物に足すと
    バイト同値が壊れる**ので、足すのは新しい出力だけにすること。
    """
    dt, source = read_dt_min(run_dir)
    if notify:
        _notify(run_dir, source, stream)
    return {"dt_min": dt, "source": source, "assumed": source == SRC_ASSUMED,
            "steps_per_day": steps_per_day(dt_min=dt),
            "steps_per_hour": steps_per_hour(dt_min=dt),
            "min_per_step": dt}


# --------------------------------------------------------------------------- #
# 派生量(clock.py の steps_per_day / steps_per_hour / step_seconds と同一定義)
# --------------------------------------------------------------------------- #
def _resolve(run_dir, dt_min, notify, stream) -> int:
    if dt_min is not None:
        dt = valid_dt_min(dt_min)
        if dt is None:
            raise ValueError(f"dt_min は 1440 の約数の正整数でなければならない: {dt_min!r}")
        return dt
    return dt_min_of(run_dir, notify=notify, stream=stream)


def steps_per_day(run_dir=None, *, dt_min: int | None = None,
                  notify: bool = True, stream=None) -> int:
    """1 日の step 数(Δt=10 で **144**)。`dt_min` を渡せばラン参照なしで換算だけする。"""
    return max(1, int(round(MINUTES_PER_DAY / float(_resolve(run_dir, dt_min, notify, stream)))))


def steps_per_hour(run_dir=None, *, dt_min: int | None = None,
                   notify: bool = True, stream=None) -> int:
    """1 時間の step 数(Δt=10 で **6**)。"""
    return max(1, int(round(60.0 / float(_resolve(run_dir, dt_min, notify, stream)))))


def min_per_step(run_dir=None, *, dt_min: int | None = None,
                 notify: bool = True, stream=None) -> int:
    """1 step の分数(Δt=10 で **10**)。`dt_min` と同義だが呼び出し側の意図が読める名前。"""
    return _resolve(run_dir, dt_min, notify, stream)


def step_seconds(run_dir=None, *, dt_min: int | None = None,
                 notify: bool = True, stream=None) -> float:
    """1 step の秒数(Δt=10 で **600.0**)。"""
    return float(_resolve(run_dir, dt_min, notify, stream)) * 60.0


def days_of(n_steps, run_dir=None, *, dt_min: int | None = None,
            notify: bool = True, stream=None) -> int:
    """step 数 → 日数(切り捨て)。`n_steps // steps_per_day`。"""
    spd = steps_per_day(run_dir, dt_min=dt_min, notify=notify, stream=stream)
    return int(n_steps) // spd


# --------------------------------------------------------------------------- #
# 日の切り方(analyze_structure.py:85-88 の前例を正準化)
# --------------------------------------------------------------------------- #
def day_of(ev, spd: int = CANON_STEPS_PER_DAY, *, prefer_sim_min: bool = True) -> int:
    """イベント 1 件の暦日。**`sim_min // 1440` を第一手段**、欠損時のみ `step // spd`。

    L1 は固定列に `sim_min` を持つ(`docs/research/data-pipeline-lit.md:61`)ので、
    第一手段は **Δt に一切依存しない**。`spd` は後退経路のためだけに要る
    (既定は正準 144 = 従来挙動)。

    ★`step // spd` と `sim_min // 1440` は **同じ日を指すとは限らない**。
    `run.start_tod="07:00"`(既定)では前者が「起床 7:00 起点の活動日」、後者が
    「暦日」になる(`docs/research/obs-u2-dt1min-design.md` §2(c) の注意)。
    どちらを使うかは解析の意味で決まるので、**流儀を変えたい場合は本関数を使わない**。
    """
    if prefer_sim_min:
        sm = ev.get("sim_min") if hasattr(ev, "get") else None
        if sm is not None:
            return int(sm) // MINUTES_PER_DAY
    step = ev.get("step") if hasattr(ev, "get") else ev
    return int(step) // int(spd)


def activity_day_of(step, spd: int = CANON_STEPS_PER_DAY) -> int:
    """`step // steps_per_day` = **開始時刻起点の活動日**(build_panel / calibrate_report の流儀)。

    暦日ではない。既定 `start_tod="07:00"` では 07:00〜翌 07:00 が 1 日になり、夜間睡眠が
    同一日に収まる(`n_days = n_steps // steps_per_day` と一致する)。**日の切り方を
    変えないまま Δt だけ吸収する**のがこの関数の役目。
    """
    return int(step) // int(spd)


def sim_min_of(ev, dt_min: int = CANON_DT_MIN) -> int:
    """イベントの `sim_min`。欠損時のみ `step * dt_min` で復元する(Δt=10 で従来と同値)。"""
    sm = ev.get("sim_min") if hasattr(ev, "get") else None
    if sm is not None:
        return int(sm)
    step = ev.get("step") if hasattr(ev, "get") else ev
    return int(step) * int(dt_min)


__all__ = [
    "CANON_DT_MIN", "CANON_STEPS_PER_DAY", "CANON_STEPS_PER_HOUR", "MINUTES_PER_DAY",
    "SRC_ASSUMED", "SRC_CONFIG", "SRC_MANIFEST",
    "read_dt_min", "dt_min_of", "dt_min_from_dotlist", "dt_min_from_yaml",
    "provenance", "valid_dt_min",
    "steps_per_day", "steps_per_hour", "min_per_step", "step_seconds", "days_of",
    "day_of", "activity_day_of", "sim_min_of",
]


def _main(argv=None) -> int:
    """CLI: ランの Δt と派生量を出す(どこから読んだかも出す)。"""
    import argparse
    ap = argparse.ArgumentParser(description="ランの Δt(run.dt_min)と派生量を出す")
    ap.add_argument("run_dir", nargs="?", default=None)
    a = ap.parse_args(argv)
    print(json.dumps(provenance(a.run_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":                              # pragma: no cover
    raise SystemExit(_main())
