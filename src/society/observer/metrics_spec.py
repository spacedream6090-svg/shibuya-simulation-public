"""metrics_spec_hash(第78バッチ 2026-07-31)= 指標定義コードの凍結ハッシュ。

正典: docs/plans/source/dual-mode-instructions.md Phase 0(6)+ 受入基準 T7 /
      docs/plans/source/dual-mode-instruments.md Part G。

何を解く問題か
--------------
指標を設定ファイルではなく**コードとして**持つと、事後に「都合のいい定義」へ書き換えても
出力からは判らない。そこで **指標の定義を含むファイル群の正規化ハッシュ**を run_manifest.json
に埋め、指標コードが 1 文字でも変われば hash が変わるようにする(= 事後の指標いじりが露見する)。

原指示は「指標を `metrics/` ディレクトリへ集約してそのハッシュを採れ」だったが、本リポジトリでは
指標定義が既に L2 集計(observer/)と解析スクリプト(scripts/)に分散して**動いている**。
これを移設するのは「ついでのリファクタリング」であり原指示自身が禁じている(全体注意)。
そこで **移設せず、対象ファイルをコード内定数として明示列挙**する方式を採った。
リストそのものもハッシュ対象に含めるので、**リストからファイルを外す改竄も hash に出る**。

正規化(あえて単純さを優先)
----------------------------
- BOM(UTF-8 BOM)を除去
- 改行を LF に統一(CRLF/CR → LF。Windows/Unix のチェックアウト差を吸収)
それだけ。**コメント除去や空白正規化はしない**。「コメントを直したら hash が変わる」のは
偽陽性ではなく仕様である(コメントには定義の但し書きが書いてあり、それも定義の一部)。

R1
--
**読むだけ**。シム本体は metrics_spec_hash を読まない(manifest に書くだけ)。
L1/L2/L3・乱数・LLM 呼数のいずれにも影響しない。

正直な限界
----------
- 「指標の定義を含むファイル」の線引きは人手の判断であり、網羅の保証はない。
  リストは本 module の `SPEC_FILES` が唯一の源で、変更は必ず hash に現れる。
- ファイルが存在しない(配布形態で scripts/ を含めない等)場合は `missing` に記録し、
  hash にも "missing" という値として入る(欠測を偽の値で埋めない)。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA = 1

# --------------------------------------------------------------------------- #
# 凍結対象(リポジトリルートからの相対パス。**この並びもハッシュ対象**)
#
#   前半 = シム側で L2 列・台帳を定義する層(observer/ と真偽台帳)
#   後半 = 事後解析側で判定式を定義する層(scripts/。Part B・C・E の凍結指標)
# --------------------------------------------------------------------------- #
SPEC_FILES: tuple[str, ...] = (
    # ---- L2 集計・事後計測の定義(observer 層)----
    "src/society/observer/aggregate.py",     # L2 の全列(register_aggregator)
    "src/society/observer/measure.py",       # 事後計測の本体(k*/伝播/エコー/norm ほか)
    "src/society/observer/stream.py",        # measure の streaming 版(定義は同一でなければならない)
    "src/society/observer/echo.py",          # エコー/自己反復 5 列(第70)
    "src/society/observer/norms.py",         # 規範化ステージ 4 段(第74)
    "src/society/observer/silence.py",       # 沈黙・undefined_action_rate(第70)
    "src/society/observer/deviation.py",     # ペルソナ逸脱率
    "src/society/observer/structure.py",     # 社会構造の内生変動
    "src/society/observer/initial_frame.py", # 初期フレーム共変量(第74)
    # ---- 真偽台帳(信念距離の分母になる ground truth の定義。第73 Part B)----
    "src/society/truth_ledger.py",
    # ---- 凍結指標の判定式(解析側)----
    "scripts/analyze_beliefs.py",            # 信念距離・検証率・訂正の伝播効率(Part B)
    "scripts/analyze_norms.py",              # 規範成立・コホート・下方因果(Part E)
    "scripts/analyze_specialization.py",     # 語彙の専門化スコア(Part C。第78で新設)
    "scripts/diagnose_stationarity.py",      # 非定常性診断(事前登録 U-10 の本体)
)


def normalize(raw: bytes) -> bytes:
    """BOM 除去 + 改行 LF 統一だけを行う(コメント除去はしない=単純さ優先)。"""
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    return raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def file_sha256(path: Path) -> str:
    """正規化後の sha256。読めなければ "missing"(欠測を偽の値で埋めない)。"""
    try:
        raw = path.read_bytes()
    except OSError:
        return "missing"
    return hashlib.sha256(normalize(raw)).hexdigest()


def compute(repo_root=None) -> dict:
    """{"metrics_spec_hash", "schema", "n_files", "files": {...}, "missing": [...]}"""
    if repo_root is None:
        from ..config import REPO_ROOT
        repo_root = REPO_ROOT
    root = Path(repo_root)
    files = {rel: file_sha256(root / rel) for rel in SPEC_FILES}
    missing = sorted(rel for rel, sha in files.items() if sha == "missing")
    # リスト自体もハッシュ対象(= リストからファイルを外す改竄も hash に出る)
    blob = json.dumps({"schema": SCHEMA, "list": list(SPEC_FILES), "files": files},
                      sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return {
        "schema": SCHEMA,
        "metrics_spec_hash": hashlib.sha256(blob.encode("utf-8")).hexdigest(),
        "n_files": len(SPEC_FILES),
        "files": files,
        "missing": missing,
    }


def spec_hash(repo_root=None) -> str:
    return compute(repo_root)["metrics_spec_hash"]
