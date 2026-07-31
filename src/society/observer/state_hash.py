"""状態ハッシュチェーン(第78バッチ 2026-07-31)= 検証ラン用の「世界が同じか」の指紋。

正典: docs/plans/source/dual-mode-instructions.md Phase 0(4)+ Phase 2 受入基準 T1/T6 /
      docs/plans/dual-mode-observe-verify-plan.md §2 第78行。

何を解く問題か
--------------
「同じ seed で回して同じ世界が再生されたか」を判定する装置は既に L1 バイト比較(golden)が
あるが、あれは **ラン終了後に parquet を突き合わせる**しかなく、どの step で分岐したかが
すぐには判らない。step ごとに world state の指紋を採って **前 step のハッシュを混ぜて
チェーン**にすれば、(a) 分岐 step が二分探索なしで判る (b) 記録を 1 行だけ改竄しても
以降が全部ずれるので検知できる。

正準シリアライズの掟(= ハッシュが環境やプロセスに依存しないための条件)
------------------------------------------------------------------------
1. **dict / set はキーでソート**してから直列化する(json.dumps(sort_keys=True))。
   エージェントは `id` 昇順、語彙集合は `sorted()`。
2. **浮動小数は固定桁の十進表記**に正規化する(`f"{v:.6f}"`)。repr の桁揺れ・
   -0.0/0.0 の差・プラットフォーム差を構造的に排除する(NaN/Inf は "nan"/"inf" 文字列)。
3. **集約の順序を固定**する(合計や連結を集合の反復順に依存させない)。
4. 非 JSON 型は `default=str` で文字列化(engine/checkpoint.config_hash と同じ流儀)。

R1(既定 OFF)
--------------
`observer.state_hash.enabled: false`(既定)では本 module は **一度も呼ばれない**
(Simulation が chain を作らない)= L1/L2/L3・summary・乱数・LLM 呼数のいずれも不変。
ON でも **書くだけ・読む経路なし**(シム本体は state_hash.jsonl を読まない)。

出力(runs/<name>/state_hash.jsonl・1 行 1 記録)
------------------------------------------------
    {"step": 0, "state": "<world state の sha256>", "hash": "<チェーン値>"}

  state … その step 終了時の world state 正準 JSON の sha256(step 単独の指紋)
  hash  … sha256(前 step の hash + "\\n" + state)。最初の step は前 hash="" で開始。

`verify_chain()` は state から hash を再計算し直して突き合わせるので、
**どの 1 行を書き換えても**(state だけ / hash だけ / 両方)必ず検出できる。

resume との関係
---------------
同じ out_dir を作り直した場合は **追記**(llm_journal / llm_cache と同じ append-only の
意味論)。resume したランのチェーンは「そのチャンクの先頭 step」で prev="" から
繋ぎ直す(前チャンクの末尾ハッシュはプロセスを跨いで持ち回らない)。したがって
チェーンの連続性が保証されるのは **1 プロセスで通した区間**であることを明記しておく。

正直な限界
----------
- ハッシュ対象は下の `_agent_row` / `_world_row` が列挙した**観測可能な状態の抜粋**であって、
  world state の全体ではない(全体を毎 step 直列化すると 10 日ランで実用外)。
  したがって「チェーン一致 = 完全同一」ではなく「**チェーン不一致 = 確実に分岐**」の
  片側判定である。完全同一の判定は従来どおり L1 バイト比較が担う。
- 抜粋の列を変えるとハッシュが変わるので、**異なるバージョンのランどうしは比較できない**。
  run_manifest.json の git SHA / metrics_spec_hash と併読すること。
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

FILENAME = "state_hash.jsonl"
SCHEMA = 1

_DEFAULTS = {
    "enabled": False,
    "interval": 1,          # 何 step ごとに採るか(1=毎 step)
    "float_digits": 6,      # 浮動小数の固定桁十進
}


# --------------------------------------------------------------------------- #
# cfg 正準化(observer/echo.cfg_of と同型: 取得不能でも落ちない)
# --------------------------------------------------------------------------- #
def cfg_from_block(block) -> dict:
    out = dict(_DEFAULTS)
    if block is None:
        return out
    try:
        get = block.get
    except AttributeError:
        return out
    try:
        out["enabled"] = bool(get("enabled", out["enabled"]))
    except (TypeError, ValueError):
        pass
    for key in ("interval", "float_digits"):
        try:
            out[key] = max(1, int(get(key, out[key])))
        except (TypeError, ValueError):
            pass
    return out


def cfg_of_config(cfg) -> dict:
    try:
        return cfg_from_block((cfg.get("observer", {}) or {}).get("state_hash", None))
    except Exception:                                  # noqa: BLE001
        return dict(_DEFAULTS)


# --------------------------------------------------------------------------- #
# 正準化ヘルパ
# --------------------------------------------------------------------------- #
def fixed(value, digits: int) -> str:
    """浮動小数 → 固定桁十進の文字列(NaN/Inf は名前で表す。-0.0 は 0.0 に潰す)。"""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "null"
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    if v == 0.0:                                       # -0.0 と 0.0 を同一視
        v = 0.0
    return f"{v:.{digits}f}"


def _agent_row(agent, digits: int) -> list:
    """エージェント 1 体の観測可能な状態(**列の並びは固定**)。

    位置・滞在・生理/心理ゲージ・所持金・語彙・記憶の量を採る。ここに列を足すと
    過去ランのハッシュと比較できなくなるので、足すときは SCHEMA を上げること。
    """
    mem = getattr(agent, "mem", None)
    return [
        int(agent.id),
        str(getattr(agent, "node", "") or ""),
        fixed(getattr(agent, "x", 0.0), digits),
        fixed(getattr(agent, "y", 0.0), digits),
        str(getattr(agent, "loc", "") or ""),
        str(getattr(agent, "building", "") or ""),
        int(getattr(agent, "floor", 0) or 0),
        bool(getattr(agent, "sleeping", False)),
        fixed(getattr(agent, "drive", 0.0), digits),
        fixed(getattr(agent, "money", 0.0), digits),
        sorted(str(w) for w in (getattr(agent, "adopted", None) or ())),
        len(getattr(mem, "buffer", ()) or ()),
        len(getattr(mem, "episodes", ()) or ()),
        len(getattr(mem, "day_summaries", ()) or ()),
        len(getattr(mem, "relations", ()) or ()),
        len(getattr(agent, "said", ()) or ()),
    ]


def _world_row(sim, step: int) -> dict:
    """世界側の集計(順序に依存しない量だけ)。"""
    logger = getattr(sim, "logger", None)
    labels = getattr(sim, "labels", None)
    net = getattr(sim, "net", None)
    return {
        "step": int(step),
        "sim_min": int(sim.clock.sim_min(step)) if getattr(sim, "clock", None) else -1,
        "n_events": int(logger.total_n_events()) if logger is not None else -1,
        "n_items": len(getattr(labels, "items", None).items) if labels is not None else -1,
        "n_posts": len(getattr(net, "posts", ()) or ()) if net is not None else -1,
        "n_news": len(getattr(net, "news", ()) or ()) if net is not None else -1,
    }


def canonical_state(sim, step: int, digits: int = 6) -> str:
    """その step 終了時の world state の**正準 JSON**(この文字列が指紋の素)。"""
    agents = sorted(getattr(sim, "agents", ()) or (), key=lambda a: int(a.id))
    blob = {
        "schema": SCHEMA,
        "world": _world_row(sim, step),
        "agents": [_agent_row(a, digits) for a in agents],
    }
    return json.dumps(blob, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), default=str)


def sha256_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chain_next(prev_hash: str, state_hash: str) -> str:
    """チェーン値 = sha256(前 step の hash + "\\n" + その step の state hash)。"""
    return sha256_of(f"{prev_hash}\n{state_hash}")


# --------------------------------------------------------------------------- #
# 記録側(sim が 1 本だけ持つ)
# --------------------------------------------------------------------------- #
class StateHashChain:
    """step 終了時に呼ばれてチェーンを伸ばし、jsonl へ 1 行追記する。

    **書くだけ**。シム本体はこのオブジェクトから値を読まない(read 経路なし=R1)。
    """

    def __init__(self, out_dir, interval: int = 1, float_digits: int = 6):
        self.path = Path(out_dir) / FILENAME
        self.interval = max(1, int(interval))
        self.float_digits = max(1, int(float_digits))
        self.prev = ""
        self.n = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def update(self, sim, step: int) -> str | None:
        """step 終了時のフック。採らない step では None を返す(何も書かない)。"""
        if (int(step) + 1) % self.interval != 0:
            return None
        state = sha256_of(canonical_state(sim, step, self.float_digits))
        cur = chain_next(self.prev, state)
        self.prev = cur
        self.n += 1
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"step": int(step), "state": state, "hash": cur},
                                ensure_ascii=False) + "\n")
        return cur


# --------------------------------------------------------------------------- #
# 検証側(解析・テストから使う。読み取り専用)
# --------------------------------------------------------------------------- #
def read_chain(path) -> list[dict]:
    """state_hash.jsonl を読む(壊れた行は無視せず ValueError にする=欠測を隠さない)。"""
    out: list[dict] = []
    with Path(path).open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"{path}:{i + 1} が JSON として読めない") from exc
            out.append(rec)
    return out


def verify_chain(records) -> dict:
    """チェーンを先頭から再計算して整合を確かめる。

    返り値 {"ok": bool, "n": int, "first_bad": int|None, "reason": str}。
    1 行でも(state / hash のどちらを書き換えても)再計算が合わなくなるので検出できる。
    """
    prev = ""
    for i, rec in enumerate(records):
        try:
            state = str(rec["state"])
            got = str(rec["hash"])
        except (KeyError, TypeError):
            return {"ok": False, "n": len(records), "first_bad": i,
                    "reason": f"レコード {i} に state/hash が無い"}
        want = chain_next(prev, state)
        if got != want:
            return {"ok": False, "n": len(records), "first_bad": i,
                    "reason": (f"レコード {i}(step={rec.get('step')})でチェーンが切れた"
                               f": hash={got[:12]}… 期待={want[:12]}…")}
        prev = got
    return {"ok": True, "n": len(records), "first_bad": None, "reason": "整合"}


def compare(path_a, path_b) -> dict:
    """2 本のチェーンを突き合わせ、**最初に分岐した step** を返す。

    返り値 {"identical": bool, "n_a", "n_b", "diverged_at": step|None}。
    """
    a, b = read_chain(path_a), read_chain(path_b)
    for ra, rb in zip(a, b):
        if ra.get("hash") != rb.get("hash"):
            return {"identical": False, "n_a": len(a), "n_b": len(b),
                    "diverged_at": ra.get("step")}
    if len(a) != len(b):
        return {"identical": False, "n_a": len(a), "n_b": len(b),
                "diverged_at": (a[min(len(a), len(b))]["step"] if len(a) > len(b)
                                else b[min(len(a), len(b))]["step"])}
    return {"identical": True, "n_a": len(a), "n_b": len(b), "diverged_at": None}
