"""対照統計(人間側の参照値)の読み込みと**来歴の強制**。

掟(これを破る参照は読み込まない):
  1. 出典(publisher / title / url / accessed)が全部あること。
  2. ライセンス(name / url / redistribution)が明記されていること。
  3. redistribution が "permitted*" 以外(= 再配布不可/不明)なら、**数値本体を
     持たない記述統計だけ**を許す。生データ相当の値を持っていたら拒否する。
  4. 値が未取得なら values を null にして status="未取得" と書く。
     **推測値で埋めない**(捏造禁止)。未取得の参照を使う指標は絶対評価しない。

この強制は「参照統計の来歴」テストで検査される。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCHEMA = "model_battery.reference/1"

REQUIRED_SOURCE = ("publisher", "title", "url", "accessed")
REQUIRED_LICENSE = ("name", "url", "redistribution")

# 再配布の可否。値そのものをリポジトリに置いてよいのは permitted 系だけ。
REDISTRIBUTION_VALUES = (
    "permitted_with_attribution",   # 出典明記で複製・公衆送信可(政府標準利用規約 2.0 等)
    "permitted",                    # 無条件に可
    "derived_stats_only",           # 生データ不可・記述統計のみ可
    "prohibited",                   # 不可(数値を置いてはいけない)
    "unknown",                      # 未確認(prohibited と同じ扱いにする)
)
_VALUE_BEARING_OK = ("permitted_with_attribution", "permitted", "derived_stats_only")


class ReferenceError(ValueError):
    """参照統計の来歴・スキーマ違反。"""


def _require(d: dict, keys, where: str) -> None:
    missing = [k for k in keys if not str(d.get(k, "")).strip()]
    if missing:
        raise ReferenceError(f"{where} に必須項目が無い: {missing}")


def validate(doc: dict) -> dict:
    """辞書 1 件を検証して返す(不正なら ReferenceError)。"""
    if not isinstance(doc, dict):
        raise ReferenceError("参照統計は JSON オブジェクトであること")
    if doc.get("schema") != SCHEMA:
        raise ReferenceError(f"schema は {SCHEMA!r} であること: {doc.get('schema')!r}")
    for k in ("id", "title"):
        if not str(doc.get(k, "")).strip():
            raise ReferenceError(f"必須項目が無い: {k}")

    src = doc.get("source")
    if not isinstance(src, dict):
        raise ReferenceError("source オブジェクトが無い")
    _require(src, REQUIRED_SOURCE, "source")

    lic = doc.get("license")
    if not isinstance(lic, dict):
        raise ReferenceError("license オブジェクトが無い")
    _require(lic, REQUIRED_LICENSE, "license")
    redis = str(lic.get("redistribution"))
    if redis not in REDISTRIBUTION_VALUES:
        raise ReferenceError(
            f"license.redistribution は {REDISTRIBUTION_VALUES} のいずれか: {redis!r}")

    status = str(doc.get("status", "")).strip()
    if status not in ("取得済", "未取得"):
        raise ReferenceError("status は '取得済' か '未取得'")

    series = doc.get("series")
    has_values = bool(series) and any(
        v.get("values") is not None for v in series.values()
        if isinstance(v, dict))
    if has_values and redis not in _VALUE_BEARING_OK:
        raise ReferenceError(
            f"redistribution={redis!r} なのに数値本体を持っている。"
            "再配布不可/不明の統計は値を置かないこと。")
    if has_values and status != "取得済":
        raise ReferenceError("値があるのに status が '取得済' でない")
    if not has_values and status == "取得済":
        raise ReferenceError("status='取得済' なのに値が無い")
    if redis in _VALUE_BEARING_OK and not str(doc.get("attribution", "")).strip():
        raise ReferenceError("再配布可の参照には attribution(出典表記)が必須")
    return doc


def load(path: str | Path) -> dict:
    p = Path(path)
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ReferenceError(f"参照統計が無い: {p}")
    except json.JSONDecodeError as exc:
        raise ReferenceError(f"参照統計が JSON でない({p}): {exc}")
    doc = validate(doc)
    doc["_path"] = str(p)
    return doc


def load_dir(dirpath: str | Path) -> dict[str, dict]:
    """ディレクトリ内の *.json を全部読んで id -> doc の辞書にする。"""
    out: dict[str, dict] = {}
    d = Path(dirpath)
    if not d.is_dir():
        return out
    for p in sorted(d.glob("*.json")):
        doc = load(p)
        out[doc["id"]] = doc
    return out


def series_values(doc: dict, key: str) -> Any | None:
    """series[key].values を取り出す。未取得なら None(呼び手が絶対評価を諦める)。"""
    s = (doc.get("series") or {}).get(key)
    if not isinstance(s, dict):
        return None
    return s.get("values")


def attribution_lines(docs) -> list[str]:
    """レポート末尾に載せる出典表記の行(再配布可のものだけ)。"""
    out = []
    for doc in docs:
        att = str(doc.get("attribution", "")).strip()
        if att:
            out.append(f"- {att}")
        else:
            src = doc.get("source", {})
            out.append(f"- {doc.get('title')} — {src.get('publisher')} "
                       f"({src.get('url')}) ※値は未収載(再配布不可/未取得)")
    return out
