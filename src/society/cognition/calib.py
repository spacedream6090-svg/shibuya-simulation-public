"""較正テーブルと σ_c の**外部化**(第80バッチ 2026-08-01)。

正典: docs/plans/source/physics-instructions.md Part P0(3)「現実の人間の時間データへの
      較正フック」/ docs/plans/source/cognition-design-record.md §2.3(σ_c で割ることが必須)。

何を解く問題か
--------------
1. **σ_c**(発火判定式の分母)。設計 §2.3:

       S_i(t) = Σ_c g_ic·|o_c(t) − ô_ic| / σ_c + Σ_j w_ij·[trigger_j]

   「σ_c はそのチャンネルの実測標準偏差。**これで割ることが必須**」。割らないと LLM が出す
   期待値に単位がなく、エージェント間・チャンネル間で g が比較できない。そして
   「σ_c は事前のパイロットランで実測して固定する(ラン中に更新すると決定論が壊れる)」。
   → 実測は `scripts/measure_sigma.py`、**凍結ファイルの読み手**が本 module。

2. **思考頻度・反応時間の較正テーブル**。P0(3):「思考頻度・反応時間の分布をハードコード
   せず、較正テーブルとして外部化してください。文脈カテゴリ(歩行中/会話中/作業中/休息中)
   × 刺激種別ごとに、思考着手までの遅延分布と基本周期のパラメータを持たせます。初期値は
   仮置きでよいですが、私が後から人間の実測データで差し替えられる構造に」。
   → `conf/cognition/calib_default.yaml` が実体。本 module は読む・検証する・ハッシュする。

両者とも **ハッシュを run_manifest.json に載せる**(P0(3)「テーブルの内容は metrics 同様
ハッシュ化して manifest に載せます」)。流儀は第80バッチ W2 の天候来歴と同じ:
**既定 OFF のランでは manifest にキー自体を出さない**(既存ランの成果物と同形を保つ)。

R1
--
- **読むだけ**。凍結ファイルの値はまだ世界に一切作用しない(第81バッチで S を組むときに
  初めて消費される)。本バッチでの作用は「読めるか・壊れていないか・どのハッシュか」だけ。
- σ_c ファイルは **無くてもランは落とさない**。σ を測るパイロットラン自体が σ ファイルの
  無い状態で回る(鶏卵)。欠測は `status: "absent"` として来歴に正直に残す。
- 較正テーブルは **壊れていたら即エラー**(黙って既定値へ落ちない)。凍結値で世界の時定数が
  決まる層なので、欠損を暗黙のフォールバックで埋めるのが最も危険。

σ = 0(定数チャンネル)の扱い — **除外**(床を敷かない)
--------------------------------------------------------
測定側(measure_sigma)が `usable: false` を立て、消費側(第81)はそのチャンネルを S から
**外す**。最小床(σ←ε)を敷く案は採らない: 定数チャンネルに床を敷くと 1/ε が巨大な利得に
なり、浮動小数の最下位ビットの揺れが発火を駆動する装置になってしまう(「測定器が現象に
反応する」の典型)。床の値そのものはファイル内 `policy.sigma_floor` に残すので、後から
方針を変えたい人が根拠つきで切り替えられる。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA = 1

# 既定パス(リポジトリルート相対)。conf 側で上書きできる。
SIGMA_DEFAULT_REL = "data/calib/sigma_c.json"
CALIB_DEFAULT_REL = "conf/cognition/calib_default.yaml"

# 較正テーブルの軸(P0(3) が名指しした 4 文脈)。テーブル側の欠けを機械検出するための正典。
CONTEXTS: tuple[str, ...] = ("walking", "talking", "working", "resting")
# 刺激種別(発火源の分類)。設計 §6-3「発火源は驚きだけでなく内省・会話も第一級」を反映し、
# 外界(social / environmental / signage)だけでなく internal(自己状態・内省)を第一級で持つ。
STIMULI: tuple[str, ...] = ("social", "environmental", "internal", "signage")
# 遅延分布の形(ホワイトリスト。自由文は不可=設計 §2.2 の DSL 検証と同じ思想)。
DISTS: tuple[str, ...] = ("lognormal", "exponential", "gamma")


def repo_root() -> Path:
    from ..config import REPO_ROOT
    return Path(REPO_ROOT)


def resolve_path(path_str: str | None, default_rel: str) -> Path:
    """空/None なら既定パス。相対はリポジトリルート基準。"""
    raw = str(path_str or "").strip() or default_rel
    p = Path(raw)
    return p if p.is_absolute() else (repo_root() / p)


def rel_path(path) -> str:
    """来歴に載せる表示パス。リポジトリ内なら**ルート相対**にする。

    絶対パスをそのまま残すと実行環境のユーザー名が成果物に混入する(公開ミラーへ出る)。
    リポジトリ外のファイルを指したときだけ絶対パスのまま残す(所在を偽らない)。
    """
    p = Path(path)
    try:
        return p.resolve().relative_to(repo_root().resolve()).as_posix()
    except (ValueError, OSError):
        return str(p)


def normalize(raw: bytes) -> bytes:
    """BOM 除去 + 改行 LF 統一(observer/metrics_spec.py と同一の正規化)。"""
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    return raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def file_sha256(path: Path) -> str:
    """正規化後の sha256。読めなければ "missing"(欠測を偽の値で埋めない)。"""
    try:
        raw = Path(path).read_bytes()
    except OSError:
        return "missing"
    return hashlib.sha256(normalize(raw)).hexdigest()


def payload_sha256(doc: dict) -> str:
    """meta.payload_sha256 を除いた本文の正準ハッシュ(weather_gen と同じ流儀)。"""
    body = {k: v for k, v in dict(doc).items() if k != "meta"}
    meta = {k: v for k, v in dict(doc.get("meta") or {}).items()
            if k != "payload_sha256"}
    blob = json.dumps({"meta": meta, **body}, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# 較正テーブル(conf/cognition/calib_default.yaml)
# --------------------------------------------------------------------------- #
def _positive(value, where: str) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"較正テーブル {where} が数値でない: {value!r}") from exc
    if not (v > 0.0):
        raise ValueError(f"較正テーブル {where} は正でなければならない: {v}")
    return v


def validate_calib(doc: dict, path: str = "") -> dict:
    """テーブルの構造検証(欠け・未知の分布形・非正値を**その場で**落とす)。

    返り値は検証済みの正準 dict(float 化済み)。
    """
    where = f"({path})" if path else ""
    if not isinstance(doc, dict):
        raise ValueError(f"較正テーブルが mapping でない{where}")
    if int(doc.get("schema", 0)) != SCHEMA:
        raise ValueError(
            f"較正テーブルの schema が非対応{where}: {doc.get('schema')!r} != {SCHEMA}")
    contexts = tuple(doc.get("contexts") or ())
    stimuli = tuple(doc.get("stimuli") or ())
    if tuple(contexts) != CONTEXTS:
        raise ValueError(
            f"較正テーブルの contexts が正典と違う{where}: {contexts} != {CONTEXTS}")
    if tuple(stimuli) != STIMULI:
        raise ValueError(
            f"較正テーブルの stimuli が正典と違う{where}: {stimuli} != {STIMULI}")

    base = doc.get("base_period") or {}
    period: dict[str, dict[str, float]] = {}
    for ctx in CONTEXTS:
        blk = base.get(ctx)
        if not isinstance(blk, dict):
            raise ValueError(f"較正テーブル base_period.{ctx} が無い{where}")
        period[ctx] = {
            "mean_min": _positive(blk.get("mean_min"), f"base_period.{ctx}.mean_min"),
            "cv": _positive(blk.get("cv"), f"base_period.{ctx}.cv"),
        }

    onset = doc.get("onset_delay") or {}
    delay: dict[str, dict[str, dict]] = {}
    for ctx in CONTEXTS:
        row = onset.get(ctx)
        if not isinstance(row, dict):
            raise ValueError(f"較正テーブル onset_delay.{ctx} が無い{where}")
        delay[ctx] = {}
        for sti in STIMULI:
            cell = row.get(sti)
            if not isinstance(cell, dict):
                raise ValueError(f"較正テーブル onset_delay.{ctx}.{sti} が無い{where}")
            dist = str(cell.get("dist", ""))
            if dist not in DISTS:
                raise ValueError(
                    f"較正テーブル onset_delay.{ctx}.{sti}.dist が未知{where}: "
                    f"{dist!r}(有効値: {', '.join(DISTS)})")
            delay[ctx][sti] = {
                "dist": dist,
                "median_s": _positive(cell.get("median_s"),
                                      f"onset_delay.{ctx}.{sti}.median_s"),
                "sigma": _positive(cell.get("sigma"),
                                   f"onset_delay.{ctx}.{sti}.sigma"),
            }
    return {"schema": SCHEMA, "version": str(doc.get("version", "")),
            "status": str((doc.get("provenance") or {}).get("status", "unknown")),
            "contexts": list(CONTEXTS), "stimuli": list(STIMULI),
            "base_period": period, "onset_delay": delay}


def load_calib(path_str: str | None = None) -> dict:
    """較正テーブルを読み・検証し・ハッシュする。壊れていれば例外(黙って落ちない)。"""
    from omegaconf import OmegaConf
    path = resolve_path(path_str, CALIB_DEFAULT_REL)
    if not path.exists():
        raise FileNotFoundError(f"認知の較正テーブルが実在しない: {path}")
    doc = OmegaConf.to_container(OmegaConf.load(path), resolve=True) or {}
    table = validate_calib(doc, str(path))
    return {"path": str(path), "sha256": file_sha256(path), "table": table}


# --------------------------------------------------------------------------- #
# σ_c(data/calib/sigma_c.json)
# --------------------------------------------------------------------------- #
def validate_sigma(doc: dict, path: str = "") -> dict:
    """σ_c 凍結ファイルの構造検証。"""
    where = f"({path})" if path else ""
    meta = doc.get("meta") or {}
    if int(meta.get("schema", 0)) != SCHEMA:
        raise ValueError(
            f"σ_c ファイルの schema が非対応{where}: {meta.get('schema')!r} != {SCHEMA}")
    declared = meta.get("payload_sha256")
    got = payload_sha256(doc)
    if declared != got:
        raise ValueError(
            f"σ_c ファイルの payload_sha256 不一致(改竄またはファイル破損){where}\n"
            f"  記載 {declared} / 実算 {got}")
    channels = doc.get("channels")
    if not isinstance(channels, dict) or not channels:
        raise ValueError(f"σ_c ファイルに channels が無い{where}")
    return doc


def load_sigma(path_str: str | None = None, *, required: bool = False) -> dict:
    """σ_c 凍結ファイルを読む。

    **無くても落とさない**(required=False): σ を測るパイロットラン自体が σ ファイルの
    無い状態で回るため。欠測は status="absent" として返し、来歴に正直に残す。
    """
    path = resolve_path(path_str, SIGMA_DEFAULT_REL)
    if not path.exists():
        if required:
            raise FileNotFoundError(f"σ_c 凍結ファイルが実在しない: {path}")
        return {"path": str(path), "status": "absent", "sha256": "missing",
                "doc": None}
    doc = json.loads(path.read_text(encoding="utf-8"))
    validate_sigma(doc, str(path))
    return {"path": str(path), "status": "loaded", "sha256": file_sha256(path),
            "doc": doc}


def sigma_of(loaded: dict) -> dict[str, float]:
    """{channel_id: σ}。**usable=false(σ=0 等)のチャンネルは含めない**(=第81 は S から外す)。"""
    doc = (loaded or {}).get("doc")
    if not doc:
        return {}
    out: dict[str, float] = {}
    for cid, row in (doc.get("channels") or {}).items():
        if not isinstance(row, dict) or not row.get("usable"):
            continue
        try:
            out[str(cid)] = float(row["sigma"])
        except (KeyError, TypeError, ValueError):
            continue
    return out


# --------------------------------------------------------------------------- #
# 来歴(run_manifest.json 用)
# --------------------------------------------------------------------------- #
def provenance(sim) -> dict | None:
    """認知の凍結入力の来歴。チャンネル観測 OFF(既定)では **None**(=既存 manifest と同形)。"""
    cfg = getattr(sim, "channelscfg", None) or {}
    if not cfg.get("enabled"):
        return None
    from . import channels as _channels
    out: dict = {
        "schema": SCHEMA,
        "channel_spec_sha256": _channels.spec_sha256(),
        "n_channels": len(_channels.CHANNELS),
        "every_steps": int(cfg.get("every_steps", 1)),
    }
    calib = getattr(sim, "cognition_calib", None)
    if calib:
        out["calib"] = {"file": rel_path(calib["path"]), "sha256": calib["sha256"],
                        "version": calib["table"]["version"],
                        "status": calib["table"]["status"]}
    sigma = getattr(sim, "cognition_sigma", None)
    if sigma:
        entry = {"file": rel_path(sigma["path"]), "status": sigma["status"],
                 "sha256": sigma["sha256"]}
        doc = sigma.get("doc")
        if doc:
            meta = doc.get("meta") or {}
            entry["payload_sha256"] = meta.get("payload_sha256")
            entry["channel_spec_sha256"] = meta.get("channel_spec_sha256")
            entry["n_usable"] = len(sigma_of(sigma))
            entry["spec_match"] = bool(
                meta.get("channel_spec_sha256") == _channels.spec_sha256())
        out["sigma_c"] = entry
    return out
