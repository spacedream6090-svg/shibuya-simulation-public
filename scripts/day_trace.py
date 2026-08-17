#!/usr/bin/env python
"""F2: 任意エージェントの「1 日トレース」抽出 — 物語の骨格は**計画 vs 実行**(第136 承認案②)。

    python scripts/day_trace.py runs/<name> --agent 2147                 # 全日ぶん md
    python scripts/day_trace.py runs/<name> --agent 2147 --day 1 --html  # 1 日ぶん md+html
    python scripts/day_trace.py runs/<name> --agent 2147 --day 1 --with-journal
    python scripts/day_trace.py runs/<name> --day 1 --top 20             # 人選補助のみ
    python scripts/day_trace.py runs/<name> --agent 2147 --day 1 --narrate mock

正典
----
- `docs/research/emergent-events-and-narrative-ui.md` §5(素材 §5-1 / 抽出 §5-2 /
  骨格=計画 vs 実行 §5-3 / 因果リンク §5-4 / 出力の 3 層 §5-5)。
- 前例(§1-3): AgentSociety の day/t + 種別タブ、LangSmith の run 木のドリルダウン。

**完全に事後のツール**(R1 の外側)
----------------------------------
本スクリプトは `runs/<name>/` の成果物を**読むだけ**である。シム本体(`src/`)・`conf/` ・
`h.txt` ・凍結 14 本(`observer/metrics_spec.py: SPEC_FILES`)には 1 バイトも触れない。
ラン中に読まれる経路は存在しない(= LLM 呼数・乱数・決定論に影響しえない)。
唯一 LLM を呼ぶのは `--narrate` を明示したとき**だけ**で、それは「ランの外の事後 1 呼」
(§5-5 の L2 層)であり、既定は `off` = 呼数ゼロ・完全決定論である。

出力の 3 層(§5-5)
--------------------
| 層 | 生成 | 本スクリプトでの実体 |
|---|---|---|
| L0 生タイムライン | 決定論 | `trace["timeline"]`(整列済み全行。json と md 末尾) |
| L1 シーンカード   | 決定論テンプレ | `trace["scenes"]`(時刻・場所・同席者・行為・逸脱・関係) |
| L2 物語文         | ラン外の事後 LLM 1 呼 | `trace["narrative"]`(**既定 OFF**。`--narrate`) |

素材と縮退(§5-1。**無い素材は「無い」と出す**=当てにいかない)
--------------------------------------------------------------
| 素材 | ファイル | 無いとき |
|---|---|---|
| 行為 | `l1_events.parquet`(part 群も可) | **必須**(無ければエラー) |
| 思考メタ | `l1b_llm.parquet` | 因果リンクの role が出ない(行為だけ出す) |
| 思考全文 | `llm_journal*.jsonl.gz` | `--with-journal` が空振り(その旨を明記) |
| 来歴 | L1 の `llm_call_id` 列 + `payload.llm_role`(`observer.llm_link`) | (step, agent) 近似に落ちる。**多義は多義と書く** |
| 記憶 | `memory.parquet` | 記憶節を出さない |
| 関係 | `relations.parquet` | closeness 差分を出さない(L1 の relation_tier だけ) |
| 名前 | `roster.parquet` / `agents.json` | id をそのまま出す |
| 計画 | L1 `plan_created` ほか(`planning.day_plan` ON のランのみ) | 骨格節を「計画なし」で出す |

日の切り方
----------
**暦日 = `sim_min // 1440`**(`world/clock.py: Clock.day` と同一の定義)。サイドカー
(memory / relations / roster)の `day` 列と同じ意味なので突合が素直になる。
`sim_min = start_min + step * dt_min` は clock の実装上**厳密な線形**なので、
day → step 範囲は 1 行読むだけで解析的に出る(全走査しない)。既定 `start_tod="07:00"`
のランでは **day0 は 07:00〜24:00 の短い初日**になる(その旨を出力に明記する)。

同 step 内の並び(§5-2-3)
--------------------------
「計画 → 欲求/熟考 → 移動 → 行為 → 会話 → 内省」の 6 相で並べ替える。**記録順そのもの**は
`seq`(窓内の行番号)で常に復元できるので、並べ替えは情報を落とさない。

依存は pyarrow + 標準ライブラリのみ(pandas / duckdb 不使用)。副作用は `--out` 配下のみ。
"""
from __future__ import annotations

import argparse
import glob
import gzip
import html
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Windows コンソール(cp932)対策。ファイル出力は常に UTF-8。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, os.path.join(_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import l1_stream as ls                                   # noqa: E402
import run_dt                                            # noqa: E402

SCHEMA = "day_trace/1"
MINUTES_PER_DAY = run_dt.MINUTES_PER_DAY


# =========================================================================== #
# 0. 対応表(**1 箇所にまとめる**。§5-2-2)
# =========================================================================== #
# 受動側の逆引き: kind → 「相手の agent_id が入る payload キー」。
# `observer/causality.py: PATIENT_ACTOR_OVERRIDES`(4 件)と同じ流儀の表だが、
# あちらは「行為者を戻す」ための最小集合、こちらは「対象が**登場する**行を拾う」ための
# 全集合なので上位集合になる(tests/test_day_trace.py が包含を機械固定する)。
#
# ★ここに **agent_id ではないキーを書いてはならない**。`ride.from/to`(ノード)・
#   `wage.to`(口座)・`move_home.from/to`(住戸)・`boredom_explore.from/to`(ノード)は
#   数値/文字列が id と衝突しうるので**意図的に載せていない**。
PATIENT_KEYS: dict[str, tuple[str, ...]] = {
    # ---- 発話・会話 ----
    "speak":                ("hearers",),
    "hear":                 ("speaker",),
    "dm":                   ("to",),
    "conversation":         ("with",),
    "episode_start":        ("partner",),
    "episode_end":          ("partner",),
    "episode_closing":      ("partner",),
    # ---- 情報の伝播 ----
    "transmission":         ("from",),
    "belief_transmit":      ("to",),
    "belief_update":        ("from",),
    "opinion_shift":        ("source",),
    "gossip_spread":        ("target", "sources"),
    "rumor_born":           ("knowers",),
    # ---- SNS ----
    "sns_read":             ("authors",),
    "sns_like":             ("author",),
    "sns_reshare":          ("author",),
    "viral_cascade":        ("author",),
    "feed_rank":            ("agent",),
    # ---- 掲示 ----
    "flyer_post":           ("author",),
    "flyer_view":           ("author",),
    "flyer_expire":         ("author",),
    # ---- 関係 ----
    "relation_tier":        ("other",),
    "relation_break":       ("other",),
    "partner_formed":       ("other",),
    "partnership_declined": ("to",),
    "life_event":           ("other",),
    "move_in":              ("pair",),
    "move_out":             ("pair",),
    "joint_invite":         ("invitee",),
    "joint_activity":       ("with",),
    "train_copresence":     ("other_id",),
    # ---- 組織・イベント・制度 ----
    "event_attend":         ("host",),
    "group_found":          ("founder",),
    "group_join":           ("founder",),
    "proposal_support":     ("author",),
    "proposal_passed":      ("supporters",),
    "council_elected":      ("members",),
    # ---- 商い・事件 ----
    "serve":                ("customer",),
    "venture_sale":         ("buyer",),
    "crime":                ("victim", "offender"),
    "deliver":              ("courier",),
    "lost_pickup":          ("owner",),
    "lost_turnin":          ("owner",),
}

# 同 step 内の相(§5-2-3)。小さいほど先。既定 = 2(その他の行為)。
_PHASE: dict[str, int] = {}


def _phase_fill(rank: int, kinds) -> None:
    for k in kinds:
        _PHASE[k] = rank


_phase_fill(0, ("day_plan", "plan_created", "plan_repair", "plan_fallback",
                "plan_replan", "plan_slide", "plan_block_drop", "plan_cont_fire",
                "plan_block_start"))
_phase_fill(1, ("drive_request", "llm_deliberate", "reflection_trigger",
                "boredom_explore", "action_reject", "fallback"))
_phase_fill(2, ("route_start", "move_segment", "arrive", "detour", "zone_gate",
                "enter_building", "exit_building", "floor_move",
                "enter_area", "exit_area", "ride", "train_ride", "taxi_unmatched",
                "gate_pass", "move_home", "relocate", "move_in", "move_out",
                "lodging_checkin", "lodging_checkout"))
# 3 = その他の行為(既定)
_phase_fill(4, ("speak", "hear", "dm", "conversation", "episode_start",
                "episode_end", "episode_closing", "engaged_template",
                "sns_post", "sns_read", "sns_like", "sns_reshare", "news_read",
                "transmission", "opinion_shift", "belief_transmit",
                "belief_update", "gossip_spread"))
_phase_fill(5, ("reflect", "memory_recall", "memory_fail", "emotion_label",
                "affect_update", "state_update", "worldview", "sign_memory",
                "health_update", "sleep_start"))
_DEFAULT_PHASE = 3

# シーンの切れ目(場所・意識状態が変わる瞬間)。
SCENE_BREAK_KINDS = frozenset({
    "arrive", "enter_building", "exit_building", "enter_area", "exit_area",
    "sleep_start", "wake_up", "episode_start", "episode_end",
    "lodging_checkin", "lodging_checkout", "ride", "train_ride", "move_home",
})

# シーンカードに 1 行ずつは載せない嵩張る行(件数と距離だけ集計する)。L0 には残る。
BULK_KINDS = frozenset({"move_segment", "traffic_flow"})

# payload の中で表示に載せない嵩張るキー(座標列など)。
BULK_PAYLOAD_KEYS = frozenset({"pts", "segs", "cars", "path"})

# 統計で「会話」に数える kind。
TALK_KINDS = frozenset({"speak", "hear", "dm", "conversation"})


# =========================================================================== #
# 1. 時刻・素材の下ごしらえ
# =========================================================================== #
def hhmm(sim_min: int) -> str:
    """絶対分 → `HH:MM`(暦日内の時刻)。"""
    m = int(sim_min) % MINUTES_PER_DAY
    return f"{m // 60:02d}:{m % 60:02d}"


def start_min_of(run_dir) -> int | None:
    """`sim_min = start_min + step * dt_min` の `start_min` を L1 の 1 行から求める。

    `world/clock.py: Clock.sim_min` は厳密な線形なので、1 行あれば全域が決まる
    (全走査しない)。L1 が空なら None。
    """
    dt = run_dt.dt_min_of(run_dir, notify=False)
    for d in ls.iter_columns(run_dir, ["step", "sim_min"], batch_rows=1):
        if d.get("step") and d.get("sim_min"):
            return int(d["sim_min"][0]) - int(d["step"][0]) * dt
        break
    return None


def day_window(run_dir, day: int) -> tuple[int, int]:
    """暦日 `day` に属する step の閉区間 `[step_min, step_max]`。

    `sim_min ∈ [day*1440, (day+1)*1440)` を step に写す。ランの終端は max_step で切る。
    """
    dt = run_dt.dt_min_of(run_dir, notify=False)
    off = start_min_of(run_dir)
    if off is None:
        return (0, -1)
    lo_min = day * MINUTES_PER_DAY
    hi_min = (day + 1) * MINUTES_PER_DAY - 1
    lo = -(-(lo_min - off) // dt)                     # ceil 割り
    hi = (hi_min - off) // dt
    lo = max(0, int(lo))
    hi = min(int(hi), ls.max_step(run_dir))
    return (lo, hi)


def observed_days(run_dir) -> list[int]:
    """ランに存在する暦日の昇順リスト(step の下限・上限から解析的に)。"""
    off = start_min_of(run_dir)
    if off is None:
        return []
    dt = run_dt.dt_min_of(run_dir, notify=False)
    d0 = off // MINUTES_PER_DAY
    d1 = (off + ls.max_step(run_dir) * dt) // MINUTES_PER_DAY
    return list(range(int(d0), int(d1) + 1))


def probe_materials(run_dir) -> dict:
    """どの素材が在るかを正直に列挙する(縮退表示の単一の源)。"""
    rd = Path(run_dir)
    journals = sorted(Path(p).name for p in glob.glob(str(rd / "llm_journal*.jsonl.gz")))
    out = {
        "l1_events": [p.name for p in ls.l1_paths(run_dir)],
        "l1b_llm": [p.name for p in ls.l1_paths(run_dir, "l1b_llm")],
        "llm_journal": journals,
        "memory": (rd / "memory.parquet").is_file(),
        "relations": (rd / "relations.parquet").is_file(),
        "roster": (rd / "roster.parquet").is_file(),
        "agents_json": (rd / "agents.json").is_file(),
    }
    out["notes"] = []
    if not out["l1b_llm"]:
        out["notes"].append("l1b_llm.parquet が無い → 因果リンクの role(purpose)は出ない。")
    if not journals:
        out["notes"].append("llm_journal*.jsonl.gz が無い → --with-journal は空振りする"
                            "(プロンプト/応答の全文は復元できない)。")
    if not out["memory"]:
        out["notes"].append("memory.parquet が無い → 記憶の節は出ない"
                            "(observer.gt_extras.memory OFF のラン)。")
    if not out["relations"]:
        out["notes"].append("relations.parquet が無い → closeness 差分は出ない"
                            "(L1 の relation_tier / relation_break だけで語る)。")
    if not out["roster"] and not out["agents_json"]:
        out["notes"].append("roster.parquet / agents.json が無い → 名前は出ず id のまま。")
    return out


# =========================================================================== #
# 2. 収集(§5-2)
# =========================================================================== #
_L1_COLUMNS = ("step", "sim_min", "agent_id", "kind", "payload",
               "llm_call_id", "cause_type", "actor_id")


def _ids_in(payload: dict, keys) -> list[int]:
    """payload の指定キーから agent_id を取り出す(list も単値も許す)。"""
    out: list[int] = []
    for k in keys:
        v = payload.get(k)
        for x in (v if isinstance(v, list) else [v]):
            if isinstance(x, bool) or x is None:
                continue
            try:
                out.append(int(x))
            except (TypeError, ValueError):
                continue
    return out


def collect_rows(run_dir, agent_id: int, step_min: int, step_max: int) -> list[dict]:
    """窓 `[step_min, step_max]` の中で **対象が関与する行**を全部集める。

    ① 主イベント = `agent_id == 対象`
    ② 受動側の逆引き = `PATIENT_KEYS` の指すキーに対象 id が入っている行

    ②の前段に **raw payload 文字列への部分一致**を掛ける(json.loads を全行に掛けない
    ための足切り)。10 進表記の int は必ずその桁列を含むので**偽陰性は出ない**。
    偽陽性(金額や別 id の部分文字列)は後段の `PATIENT_KEYS` + 整数一致で必ず落ちる。
    """
    if step_max < step_min:
        return []
    needle = str(int(agent_id))
    rows: list[dict] = []
    seq = 0
    for d in ls.iter_columns(run_dir, list(_L1_COLUMNS),
                             step_min=step_min, step_max=step_max):
        n = len(d["step"])
        for i in range(n):
            seq += 1
            aid = int(d["agent_id"][i])
            kind = d["kind"][i]
            raw = d["payload"][i]
            own = (aid == agent_id)
            role = None
            if not own:
                keys = PATIENT_KEYS.get(kind)
                if not keys or not raw or needle not in raw:
                    continue
                try:
                    pay = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue
                if not isinstance(pay, dict) or agent_id not in _ids_in(pay, keys):
                    continue
                role = "patient"
            else:
                try:
                    pay = json.loads(raw) if raw else {}
                except (json.JSONDecodeError, TypeError):
                    pay = {}
                if not isinstance(pay, dict):
                    pay = {}
                role = "actor"
            rows.append({
                "seq": seq,
                "step": int(d["step"][i]),
                "sim_min": int(d["sim_min"][i]),
                "agent_id": aid,
                "kind": kind,
                "payload": pay,
                "llm_call_id": (d.get("llm_call_id") or [None] * n)[i],
                "cause_type": (d.get("cause_type") or [None] * n)[i],
                "actor_id": (d.get("actor_id") or [None] * n)[i],
                "side": role,
            })
    rows.sort(key=lambda r: (r["step"], _PHASE.get(r["kind"], _DEFAULT_PHASE), r["seq"]))
    return rows


def load_llm_calls(run_dir, agent_id: int, step_min: int, step_max: int) -> dict:
    """`l1b_llm.parquet` から対象 × 窓の呼を `{llm_call_id: {...}}` で返す。"""
    out: dict[str, dict] = {}
    for path in ls.l1_paths(run_dir, "l1b_llm"):
        for d in ls.iter_table_columns(path,
                                       ["llm_call_id", "agent_id", "purpose",
                                        "step", "cached"]):
            n = len(d.get("llm_call_id", ()))
            for i in range(n):
                if int(d["agent_id"][i]) != agent_id:
                    continue
                st = int(d["step"][i])
                if not (step_min <= st <= step_max):
                    continue
                cid = d["llm_call_id"][i]
                if cid:
                    out[cid] = {"purpose": d["purpose"][i], "step": st,
                                "cached": bool(d["cached"][i])}
    return out


def load_journal(run_dir, want_ids: set) -> dict:
    """`llm_journal*.jsonl.gz` から欲しい呼だけ全文を拾う。

    L1/l1b の `llm_call_id` は **ジャーナルの `key` の先頭 16 文字**
    (`llm/cache.py` が `sha256(...)[:16]` を発行する。`scripts/bench.py:541` と同じ突合)。
    走査は全レコードに及ぶ(ジャーナルは content-addressed で索引を持たない)ので、
    本関数は `--with-journal` のときだけ呼ぶ。
    """
    if not want_ids:
        return {}
    out: dict[str, dict] = {}
    for path in sorted(glob.glob(str(Path(run_dir) / "llm_journal*.jsonl.gz"))):
        try:
            fh = gzip.open(path, "rt", encoding="utf-8")
        except OSError:
            continue
        with fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue                       # 追記中の途中メンバは黙って飛ばす
                key = rec.get("key") or ""
                cid = key[:16]
                if cid in want_ids and cid not in out:
                    out[cid] = {"prompt": rec.get("prompt", ""),
                                "response": rec.get("response", ""),
                                "backend": rec.get("backend", ""),
                                "cached": bool(rec.get("cached")),
                                "seq": rec.get("seq")}
    return out


def load_memory(run_dir, agent_id: int, step_min: int, step_max: int) -> list[dict]:
    """その日に**形成された**記憶(memory.parquet の `step` が窓内)。

    memory.parquet は日次スナップショットなので**同じ記憶が複数日に現れる**。
    `(src, step, kind, text)` で重複を畳み、最初に現れた日を `first_day`、
    最後に見えた日の importance を `importance_last` として残す(減衰が見える)。
    """
    path = Path(run_dir) / "memory.parquet"
    if not path.is_file():
        return []
    agg: dict[tuple, dict] = {}
    for d in ls.iter_table_columns(path, ["day", "agent_id", "src", "idx", "step",
                                          "kind", "importance", "text"]):
        n = len(d.get("day", ()))
        for i in range(n):
            if int(d["agent_id"][i]) != agent_id:
                continue
            st = int(d["step"][i])
            if not (step_min <= st <= step_max):
                continue
            key = (d["src"][i], st, d["kind"][i], d["text"][i])
            imp = float(d["importance"][i]) if d["importance"][i] is not None else None
            day = int(d["day"][i])
            cur = agg.get(key)
            if cur is None:
                agg[key] = {"src": key[0], "step": st, "kind": key[2], "text": key[3],
                            "first_day": day, "last_day": day,
                            "importance_first": imp, "importance_last": imp}
            elif day > cur["last_day"]:
                cur["last_day"] = day
                cur["importance_last"] = imp
    return sorted(agg.values(),
                  key=lambda r: (r["step"], r["src"], r["kind"], r["text"]))


def load_relations(run_dir, agent_id: int, day: int) -> dict:
    """`relations.parquet` の day / day+1 スナップから当日の closeness 差分を作る。

    ★`day` 列は「その日の**始まり**に見た値」(= 前日の終わり)なので、
    **当日中の変化 = relations[day+1] - relations[day]** である。day+1 のスナップが
    無い(= ランの最終日)ときは差分を測らず、その旨を返す。
    """
    path = Path(run_dir) / "relations.parquet"
    if not path.is_file():
        return {"present": False, "reason": "relations.parquet が無い"}
    before: dict[int, tuple] = {}
    after: dict[int, tuple] = {}
    seen_days: set[int] = set()
    for d in ls.iter_table_columns(path, ["day", "agent_id", "other_id",
                                          "closeness", "tier", "count", "dormant"]):
        n = len(d.get("day", ()))
        for i in range(n):
            dd = int(d["day"][i])
            seen_days.add(dd)
            if int(d["agent_id"][i]) != agent_id or dd not in (day, day + 1):
                continue
            oid = int(d["other_id"][i])
            if oid < 0:                       # すれ違い行(other_id = -1)は相手が居ない
                continue
            rec = (d["closeness"][i], d["tier"][i], d["count"][i], d["dormant"][i])
            (before if dd == day else after)[oid] = rec
    if day + 1 not in seen_days:
        return {"present": False,
                "reason": f"day{day + 1} のスナップショットが無い"
                          f"(当日の差分は測れない。観測日 {sorted(seen_days)})",
                "snapshot_days": sorted(seen_days)}
    deltas = []
    for oid in sorted(set(before) | set(after)):
        b, a = before.get(oid), after.get(oid)
        c0 = float(b[0]) if b and b[0] is not None else None
        c1 = float(a[0]) if a and a[0] is not None else None
        if c0 is None and c1 is None:
            continue
        d_close = None if (c0 is None or c1 is None) else round(c1 - c0, 6)
        deltas.append({"other_id": oid, "closeness_before": c0, "closeness_after": c1,
                       "delta": d_close,
                       "tier_before": (int(b[1]) if b and b[1] is not None else None),
                       "tier_after": (int(a[1]) if a and a[1] is not None else None),
                       "new": b is None})
    deltas.sort(key=lambda r: (-(abs(r["delta"]) if r["delta"] is not None else 0.0),
                               r["other_id"]))
    return {"present": True, "n": len(deltas), "deltas": deltas}


def load_names(run_dir, want: set) -> dict:
    """欲しい id だけの `{id: {name, ...}}`。roster.parquet → agents.json の順。"""
    want = {int(i) for i in want if i is not None and int(i) >= 0}
    if not want:
        return {}
    out: dict[int, dict] = {}
    rpath = Path(run_dir) / "roster.parquet"
    if rpath.is_file():
        cols = ["agent_id", "name", "age", "gender", "occupation", "visitor",
                "home_node", "work_name", "org_role", "household_id", "day"]
        for d in ls.iter_table_columns(rpath, cols):
            n = len(d.get("agent_id", ()))
            for i in range(n):
                aid = int(d["agent_id"][i])
                if aid in want and aid not in out:
                    out[aid] = {k: d[k][i] for k in d if k != "agent_id"}
                    # roster の `day` は「名簿に載った日(入場日)」。トレースの day と
                    # 紛れるので改名して出す(値は 1 ビットも変えない)。
                    if "day" in out[aid]:
                        out[aid]["entry_day"] = out[aid].pop("day")
                    out[aid]["_src"] = "roster.parquet"
    missing = want - set(out)
    apath = Path(run_dir) / "agents.json"
    if missing and apath.is_file():
        try:
            data = json.loads(apath.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = []
        for a in data if isinstance(data, list) else []:
            try:
                aid = int(a["id"])
            except (KeyError, TypeError, ValueError):
                continue
            if aid in missing:
                rec = {k: a.get(k) for k in ("name", "age", "gender", "occupation",
                                             "visitor", "work_name", "home",
                                             "household_id")}
                rec["home_node"] = rec.pop("home", None)
                rec["_src"] = "agents.json"
                out[aid] = rec
    return out


def who(names: dict, aid) -> str:
    """id → 「名前(id)」。名簿に無ければ id だけ(捏造しない)。"""
    if aid is None:
        return "?"
    try:
        aid = int(aid)
    except (TypeError, ValueError):
        return str(aid)
    if aid < 0:
        return "世界(-1)"
    nm = (names.get(aid) or {}).get("name")
    return f"{nm}({aid})" if nm else f"#{aid}"


# =========================================================================== #
# 3. 骨格 = 計画 vs 実行(§5-3)
# =========================================================================== #
_PLAN_EVENT_KINDS = ("plan_created", "plan_block_start", "plan_block_drop",
                     "plan_slide", "plan_cont_fire", "plan_replan",
                     "plan_repair", "plan_fallback")


def build_plan(rows: list[dict]) -> dict:
    """朝の計画と、その各ブロックがどうなったかの突合表。

    帰属は 2 経路(`analyze_plan_execution.py` と同じ線引き):
      (a) **添字経路**(W4-F 以降のラン): `plan_block_*` / `plan_slide` /
          `plan_cont_fire` の `payload.block` が `plan_created.blocks[]` の添字。
          多義が構造的にゼロ。
      (b) **組経路**(旧ラン): `(act, place, start)` で照合する。同値ブロックが
          2 件以上あると**多義**なので、当てにいかず `ambiguous` に数える。
    """
    own = [r for r in rows if r["side"] == "actor"]
    created = [r for r in own if r["kind"] == "plan_created"]
    if not created:
        return {"present": False,
                "reason": "この日に plan_created が無い"
                          "(planning.day_plan OFF のラン、または個体が計画を持たない日)",
                "blocks": [], "events": [], "attribution": "none"}
    head = created[0]                                  # 1 暦日 1 計画(第115 DPH-C)
    pay = head["payload"]
    blocks = []
    for i, b in enumerate(pay.get("blocks") or []):
        if not isinstance(b, dict):
            continue
        blocks.append({"i": i, "act": b.get("act"), "place": b.get("place"),
                       "aim": b.get("aim"), "priority": b.get("priority"),
                       "flex": b.get("flex"), "start": b.get("start"),
                       "end": b.get("end"),
                       "status": "untouched", "events": []})
    have_index = False
    ambiguous = 0
    unmatched = 0
    by_tuple: dict[tuple, list[int]] = defaultdict(list)
    for b in blocks:
        by_tuple[(b["act"], b["place"], b["start"])].append(b["i"])

    events = []
    for r in own:
        if r["kind"] not in _PLAN_EVENT_KINDS:
            continue
        p = r["payload"]
        ev = {"seq": r["seq"], "step": r["step"], "sim_min": r["sim_min"],
              "kind": r["kind"], "payload": p, "block": None, "block_src": None}
        if r["kind"] in ("plan_block_start", "plan_block_drop", "plan_slide",
                         "plan_cont_fire"):
            bi = p.get("block")
            if isinstance(bi, int) and 0 <= bi < len(blocks):
                ev["block"], ev["block_src"] = bi, "index"
                have_index = True
            else:
                cands = by_tuple.get((p.get("act"), p.get("place"), p.get("start")), [])
                if len(cands) == 1:
                    ev["block"], ev["block_src"] = cands[0], "tuple"
                elif len(cands) > 1:
                    ev["block_src"] = "ambiguous"
                    ambiguous += 1
                else:
                    ev["block_src"] = "unmatched"
                    unmatched += 1
        events.append(ev)

    for ev in events:
        bi = ev["block"]
        if bi is None:
            continue
        b = blocks[bi]
        b["events"].append({"kind": ev["kind"], "sim_min": ev["sim_min"],
                            "payload": ev["payload"]})
        k = ev["kind"]
        if k == "plan_block_start":
            b["status"] = "executed"
            b["executed_at"] = ev["sim_min"]
        elif k == "plan_block_drop" and b["status"] != "executed":
            b["status"] = "dropped"
            b["drop_reason"] = ev["payload"].get("reason")
        elif k == "plan_slide":
            b["slid"] = ev["payload"].get("slid")
            if b["status"] == "untouched":
                b["status"] = "slid"
        elif k == "plan_cont_fire":
            b["cont"] = {"cond": ev["payload"].get("cond"),
                         "then": ev["payload"].get("then"),
                         "applied": bool(ev["payload"].get("applied"))}

    counts = Counter(b["status"] for b in blocks)
    return {
        "present": True,
        "src": pay.get("src"), "model": pay.get("model"),
        "version": pay.get("version"), "n": pay.get("n", len(blocks)),
        "n_cont": pay.get("n_cont"),
        "created_at": head["sim_min"], "llm_call_id": head["llm_call_id"],
        "attribution": "block_index" if have_index else
                       ("tuple_match" if blocks else "none"),
        "ambiguous": ambiguous, "unmatched": unmatched,
        "blocks": blocks, "events": events,
        "counts": {k: counts.get(k, 0) for k in
                   ("executed", "dropped", "slid", "untouched")},
        "n_replan": sum(1 for e in events if e["kind"] == "plan_replan"),
        "n_repair": sum(1 for e in events if e["kind"] == "plan_repair"),
        "n_fallback": sum(1 for e in events if e["kind"] == "plan_fallback"),
    }


# =========================================================================== #
# 4. シーンカード(§5-5 の L1 層)
# =========================================================================== #
def _place_of(row: dict) -> tuple[str, str] | None:
    """その行が「今どこか」を更新するなら `(場所ラベル, 場所id)` を返す。"""
    k, p = row["kind"], row["payload"]
    if k == "arrive":
        return (p.get("name") or p.get("node") or "?", p.get("node") or "")
    if k in ("enter_building", "floor_move"):
        nm = p.get("name") or p.get("building") or "?"
        fl = p.get("floor")
        return (f"{nm}" + (f" {fl}F" if fl is not None else ""), p.get("building") or "")
    if k == "exit_building":
        return ("路上", "")
    if k == "exit_area":
        return (f"圏外(via {p.get('via') or '?'})", p.get("gateway") or "")
    if k == "enter_area":
        return (f"圏内へ帰還(via {p.get('via') or '?'})", p.get("gateway") or "")
    if k in ("lodging_checkin",):
        return (p.get("poi") or "宿", p.get("node") or "")
    # ★`plan_block_start.node` は**行き先ではなく実行時の現在ノード**(schema.py の明記)。
    #   ここで場所として採ると「到達先」と誤読させるので採らない。
    if k == "ride":
        return (f"{p.get('mode') or '乗り物'}で移動中", str(p.get("to") or ""))
    if k == "train_ride":
        return (f"{p.get('line') or '列車'}(乗車中)", "")
    return None


def build_scenes(rows: list[dict], agent_id: int, *, gap_min: int = 60) -> list[dict]:
    """時系列を「場所・意識状態・時間の断絶」で切ってシーンカードにする。"""
    scenes: list[dict] = []
    cur: dict | None = None
    place, place_id = "?", ""
    last_min = None

    def _open(row):
        return {"t0": row["sim_min"], "t1": row["sim_min"],
                "step0": row["step"], "step1": row["step"],
                "place": place, "place_id": place_id,
                "rows": [], "partners": set(), "utterances": [],
                "plan_notes": [], "relation_notes": [], "thought_refs": [],
                "money": [], "bulk": Counter(), "dist_m": 0.0}

    for r in rows:
        upd = _place_of(r)
        brk = (cur is None or r["kind"] in SCENE_BREAK_KINDS
               or (last_min is not None and r["sim_min"] - last_min > gap_min))
        if upd is not None:
            place, place_id = upd
        if brk:
            if cur is not None:
                scenes.append(cur)
            cur = _open(r)
        cur["t1"], cur["step1"] = r["sim_min"], r["step"]
        if upd is not None and not cur["rows"]:
            cur["place"], cur["place_id"] = place, place_id
        if r["kind"] in BULK_KINDS:
            cur["bulk"][r["kind"]] += 1
            try:
                cur["dist_m"] += float(r["payload"].get("dist_m") or 0.0)
            except (TypeError, ValueError):
                pass
        else:
            cur["rows"].append(r)
        _harvest(cur, r, agent_id)
        last_min = r["sim_min"]
    if cur is not None:
        scenes.append(cur)
    for s in scenes:
        s["partners"] = sorted(s["partners"])
        s["bulk"] = dict(sorted(s["bulk"].items()))
        s["dist_m"] = round(s["dist_m"], 1)
    return scenes


def _harvest(scene: dict, r: dict, agent_id: int) -> None:
    """1 行からシーンカードの各欄(同席者・実発話・逸脱・関係・思考・金)を抜く。

    「相手」からは**本人を除く**(knowers / hearers / with には本人も入るため)。
    """
    k, p = r["kind"], r["payload"]
    keys = PATIENT_KEYS.get(k)
    if keys:
        for oid in _ids_in(p, keys):
            if oid >= 0 and oid != agent_id:
                scene["partners"].add(oid)
    if r["side"] == "patient" and 0 <= r["agent_id"] != agent_id:
        scene["partners"].add(r["agent_id"])
    if k in ("speak", "dm", "sns_post") and p.get("text"):
        scene["utterances"].append({"sim_min": r["sim_min"], "kind": k,
                                    "speaker": r["agent_id"], "text": p["text"],
                                    "to": p.get("to"), "hearers": p.get("hearers")})
    if k == "hear" and p.get("text"):
        scene["utterances"].append({"sim_min": r["sim_min"], "kind": k,
                                    "speaker": p.get("speaker"), "text": p["text"]})
    if k.startswith("plan_"):
        scene["plan_notes"].append(r)
    if k in ("relation_tier", "relation_break", "partner_formed",
             "partnership_declined", "life_event"):
        scene["relation_notes"].append(r)
    if r["llm_call_id"]:
        scene["thought_refs"].append({"sim_min": r["sim_min"], "kind": k,
                                      "llm_call_id": r["llm_call_id"],
                                      "llm_role": p.get("llm_role")})
    if k in ("spend", "wage", "tax", "rent", "withdraw", "deposit", "venture_sale",
             "rule_bonus", "interest_paid"):
        scene["money"].append({"sim_min": r["sim_min"], "kind": k,
                               "amount": p.get("amount"), "cat": p.get("cat"),
                               "payee": p.get("payee") or p.get("to")})


# =========================================================================== #
# 5. 日次サマリ統計
# =========================================================================== #
def build_stats(rows: list[dict], plan: dict, agent_id: int) -> dict:
    own = [r for r in rows if r["side"] == "actor"]
    kinds = Counter(r["kind"] for r in rows)
    own_kinds = Counter(r["kind"] for r in own)
    dist = 0.0
    routes = 0
    spend_by_cat: Counter = Counter()
    income = 0.0
    tax = 0.0
    partners: set = set()
    nodes: list[str] = []
    for r in rows:
        k, p = r["kind"], r["payload"]
        if k == "move_segment" and r["side"] == "actor":
            try:
                dist += float(p.get("dist_m") or 0.0)
            except (TypeError, ValueError):
                pass
        elif k == "route_start" and r["side"] == "actor":
            routes += 1
        elif k == "spend" and r["side"] == "actor":
            try:
                spend_by_cat[str(p.get("cat") or "?")] += float(p.get("amount") or 0.0)
            except (TypeError, ValueError):
                pass
        elif k == "wage" and r["side"] == "actor":
            try:
                income += float(p.get("amount") or 0.0)
            except (TypeError, ValueError):
                pass
        elif k == "tax" and r["side"] == "actor":
            try:
                tax += float(p.get("amount") or 0.0)
            except (TypeError, ValueError):
                pass
        elif k == "arrive" and r["side"] == "actor":
            nodes.append(str(p.get("node") or ""))
        pk = PATIENT_KEYS.get(k)
        if pk:
            partners.update(i for i in _ids_in(p, pk) if i >= 0 and i != agent_id)
        if r["side"] == "patient" and 0 <= r["agent_id"] != agent_id:
            partners.add(r["agent_id"])
    return {
        "n_rows": len(rows), "n_own": len(own), "n_patient": len(rows) - len(own),
        "n_kinds": len(kinds),
        "kinds": dict(sorted(kinds.items())),
        "own_kinds": dict(sorted(own_kinds.items())),
        "distance_m": round(dist, 1), "n_routes": routes,
        "n_talk": sum(own_kinds.get(k, 0) for k in TALK_KINDS),
        "n_speak": own_kinds.get("speak", 0), "n_hear": own_kinds.get("hear", 0),
        "n_dm": own_kinds.get("dm", 0),
        "n_conversation": own_kinds.get("conversation", 0),
        "partners": sorted(partners),
        "n_partners": len(partners),
        "spend_total": round(sum(spend_by_cat.values()), 1),
        "spend_by_cat": {k: round(v, 1) for k, v in sorted(spend_by_cat.items())},
        "income": round(income, 1), "tax": round(tax, 1),
        "n_nodes_visited": len(set(n for n in nodes if n)),
        "plan_counts": plan.get("counts") if plan.get("present") else None,
    }


# =========================================================================== #
# 6. 語り手の声(§5-4)= 夜の内省
# =========================================================================== #
def build_reflections(rows: list[dict], calls: dict, journal: dict) -> list[dict]:
    out = []
    for r in rows:
        if r["side"] != "actor" or r["kind"] != "reflect":
            continue
        p = r["payload"]
        cid = r["llm_call_id"]
        rec = {"sim_min": r["sim_min"], "step": r["step"],
               "mode": p.get("mode"), "context": p.get("context") or p.get("when"),
               "written_back": bool(p.get("written_back")),
               "summary": p.get("summary"), "belief": p.get("belief"),
               "n_salient": p.get("n_salient"), "llm_call_id": cid,
               "purpose": (calls.get(cid) or {}).get("purpose")}
        j = journal.get(cid)
        if j:
            rec["response_full"] = j["response"]
            rec["prompt_full"] = j["prompt"]
        out.append(rec)
    return out


# =========================================================================== #
# 7. トレース 1 本の組み立て(決定論・LLM 呼ゼロ)
# =========================================================================== #
def build_trace(run_dir, agent_id: int, day: int, *, with_journal: bool = False,
                gap_min: int = 60) -> dict:
    dt = run_dt.dt_min_of(run_dir, notify=False)
    prov = run_dt.provenance(run_dir, notify=False)
    off = start_min_of(run_dir) or 0
    lo, hi = day_window(run_dir, day)
    rows = collect_rows(run_dir, agent_id, lo, hi)
    calls = load_llm_calls(run_dir, agent_id, lo, hi)
    want_ids = {r["llm_call_id"] for r in rows if r["llm_call_id"]} | set(calls)
    journal = load_journal(run_dir, want_ids) if with_journal else {}
    plan = build_plan(rows)
    scenes = build_scenes(rows, agent_id, gap_min=gap_min)
    stats = build_stats(rows, plan, agent_id)
    memory = load_memory(run_dir, agent_id, lo, hi)
    relations = load_relations(run_dir, agent_id, day)
    reflections = build_reflections(rows, calls, journal)

    want_names = set(stats["partners"]) | {agent_id}
    for d in relations.get("deltas", []) or []:
        want_names.add(d["other_id"])
    names = load_names(run_dir, want_names)

    mats = probe_materials(run_dir)
    # provlink は **3 値**。行が 0 件の日に「OFF」と書くのは事実の捏造なので unknown。
    if not rows:
        mats["provlink"] = "unknown"
    elif any(r["payload"].get("llm_role") for r in rows):
        mats["provlink"] = True
    else:
        mats["provlink"] = False
        mats["notes"].append(
            "この個体日の payload に llm_role が 1 件も無い"
            "(observer.llm_link OFF のラン、またはこの日の行為が全てルール由来)→ "
            "行為→思考のリンクは (step, agent) 近似。同 step に複数呼があると多義。")
    if with_journal and want_ids and not journal:
        mats["notes"].append(
            f"--with-journal を指定したが、この個体日の呼 {len(want_ids)} 件に対して"
            f" ジャーナル上の一致は 0 件だった。")
    mats["notes"] = sorted(set(mats["notes"]))

    return {
        "schema": SCHEMA,
        "run": {"dir": str(run_dir), "name": Path(run_dir).name,
                "dt_min": dt, "dt_src": prov.get("source"),
                "steps_per_day": prov.get("steps_per_day")},
        "agent": {"id": agent_id, **(names.get(agent_id) or {})},
        "day": day,
        "window": {"step_min": lo, "step_max": hi,
                   "sim_min_min": lo * dt + off, "sim_min_max": hi * dt + off},
        "materials": mats,
        "plan": plan,
        "scenes": scenes,
        "reflections": reflections,
        "memory": memory,
        "relations": relations,
        "stats": stats,
        "thoughts": {cid: {**calls.get(cid, {}), **journal.get(cid, {})}
                     for cid in sorted(want_ids)},
        "names": {str(k): v for k, v in sorted(names.items())},
        "timeline": [{k: r[k] for k in ("seq", "step", "sim_min", "agent_id", "kind",
                                        "side", "llm_call_id", "cause_type",
                                        "actor_id", "payload")} for r in rows],
        "narrative": None,
    }


# =========================================================================== #
# 8. 人選補助 `--top N`(その日いちばん物語的な個体を提案する)
# =========================================================================== #
def rank_agents(run_dir, day: int, top: int = 20) -> dict:
    """kind の**自己較正した希少度**(surprisal)でその日の個体を並べる。

    スコア = Σ_{個体が出したことのある kind k} -log2( n_k / N )
      n_k = その日の全個体を通した kind k の件数 / N = その日の総件数。
    「珍しいことをした個体ほど高い」= 希少度は事前の重み表ではなくその日の分布から
    自動的に決まる(§4 の「希少度(自己較正)」)。多様度(distinct kind 数)も併記する。

    ★**主イベント(agent_id 列)だけ**を数える(受動側の逆引きは全個体ぶんの payload
    走査になり O(N×行) で現実的でないため)。人選の**提案**であって指標ではない。
    """
    lo, hi = day_window(run_dir, day)
    if hi < lo:
        return {"day": day, "n_agents": 0, "n_events": 0, "top": []}
    by_agent: dict[int, Counter] = defaultdict(Counter)
    kind_tot: Counter = Counter()
    total = 0
    for d in ls.iter_columns(run_dir, ["agent_id", "kind"], step_min=lo, step_max=hi):
        n = len(d["agent_id"])
        for i in range(n):
            aid = int(d["agent_id"][i])
            if aid < 0:
                continue
            k = d["kind"][i]
            by_agent[aid][k] += 1
            kind_tot[k] += 1
            total += 1
    if not total:
        return {"day": day, "n_agents": 0, "n_events": 0, "top": []}
    import math
    surp = {k: -math.log2(n / total) for k, n in kind_tot.items()}
    # 「希少 kind」= その日の件数が下位 1/4 に入る kind(件数の 25 パーセンタイル以下)。
    _counts = sorted(kind_tot.values())
    rare_cut = _counts[max(0, len(_counts) // 4 - 1)] if _counts else 0
    ranked = []
    for aid in sorted(by_agent):
        c = by_agent[aid]
        score = sum(surp[k] for k in c)
        rare = sorted(k for k in c if kind_tot[k] <= rare_cut)
        ranked.append({"agent_id": aid, "score": round(score, 4),
                       "n_events": sum(c.values()), "diversity": len(c),
                       "rare_kinds": rare, "n_rare": len(rare)})
    ranked.sort(key=lambda r: (-r["score"], r["agent_id"]))
    ranked = ranked[:max(1, int(top))]
    names = load_names(run_dir, {r["agent_id"] for r in ranked})
    for r in ranked:
        r["name"] = (names.get(r["agent_id"]) or {}).get("name")
    return {"day": day, "n_agents": len(by_agent), "n_events": total,
            "rare_threshold": rare_cut,
            "note": "主イベント(agent_id 列)のみ。受動側の逆引きは含まない。",
            "top": ranked}


# =========================================================================== #
# 9. 描画(md / html)
# =========================================================================== #
def _short(v, limit: int = 160) -> str:
    if isinstance(v, str):
        return v if len(v) <= limit else v[:limit] + "…"
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, (list, tuple)):
        s = json.dumps(v, ensure_ascii=False)
        return s if len(s) <= limit else s[:limit] + "…"
    if isinstance(v, dict):
        s = json.dumps(v, ensure_ascii=False, sort_keys=True)
        return s if len(s) <= limit else s[:limit] + "…"
    return str(v)


def _payload_brief(p: dict, limit: int = 160) -> str:
    items = [(k, v) for k, v in sorted(p.items())
             if k not in BULK_PAYLOAD_KEYS and v not in (None, "", [], {})]
    return ", ".join(f"{k}={_short(v, limit)}" for k, v in items)


def line_of(r: dict, names: dict) -> str:
    """1 行 → 人間が読む 1 文。表に無い kind は payload をそのまま短く出す。

    受動側の行(対象が payload に現れただけの行)は **「誰の行為か」を頭に付ける**。
    付けないと `relation_break{other: 対象}` が「対象が切った」ように読めてしまう。
    """
    if r["side"] == "patient":
        return f"[{who(names, r['agent_id'])} の行] " + _line_body(r, names)
    return _line_body(r, names)


def _line_body(r: dict, names: dict) -> str:
    k, p = r["kind"], r["payload"]
    w = lambda i: who(names, i)                                    # noqa: E731
    if k == "speak":
        hs = p.get("hearers") or []
        to = "、".join(w(i) for i in hs) if hs else "誰にともなく"
        return f'{w(r["agent_id"])} が {to} に「{p.get("text", "")}」'
    if k == "hear":
        return f'{w(p.get("speaker"))} の話を聞いた' + (
            f'「{p["text"]}」' if p.get("text") else "")
    if k == "dm":
        return f'{w(r["agent_id"])} → {w(p.get("to"))} へ DM「{p.get("text", "")}」'
    if k == "conversation":
        return (f'{w(r["agent_id"])} と {w(p.get("with"))} の会話'
                f'(話題={p.get("topic")} / 調子={p.get("tone")} / '
                f'{p.get("acts")}手 / 結果={p.get("outcome")})')
    if k == "reflect":
        return (f'内省({p.get("context") or p.get("when")}/{p.get("mode")}): '
                f'{p.get("summary") or ""}'
                + (f' — 信念「{p["belief"]}」' if p.get("belief") else ""))
    if k == "free_action":
        return (f'自由行動「{p.get("what")}」({p.get("category")}・'
                f'{p.get("minutes")}分・{p.get("cost")}円)')
    if k == "plan_created":
        return f'朝の計画を立てた(src={p.get("src")} / {p.get("n")} ブロック)'
    if k == "plan_block_start":
        return (f'計画 #{p.get("block")}「{p.get("act")}@{p.get("place")}」を'
                f'開始({p.get("priority")}/{p.get("flex")})')
    if k == "plan_block_drop":
        return (f'計画 #{p.get("block")}「{p.get("act")}@{p.get("place")}」を'
                f'破棄(理由={p.get("reason")})')
    if k == "plan_slide":
        return (f'計画 #{p.get("block")}「{p.get("act")}@{p.get("place")}」を'
                f'{p.get("slid")}分うしろへ')
    if k == "plan_cont_fire":
        return (f'計画 #{p.get("block")} の「もし{p.get("cond")}なら'
                f'{p.get("then")}」が発動(適用={p.get("applied")})')
    if k == "plan_replan":
        return f'再計画(version={p.get("version")} / must={p.get("n_must")})'
    if k == "arrive":
        return f'{p.get("name") or p.get("node")} に到着' + (
            "(初訪問)" if p.get("first_visit") else "")
    if k == "route_start":
        return (f'{p.get("dest_name") or p.get("dest")} へ向かう'
                f'({p.get("dist_m")}m / {p.get("mode")})')
    if k == "enter_building":
        return f'{p.get("name") or p.get("building")} に入る' + (
            "(自宅)" if p.get("home") else "")
    if k == "exit_building":
        return f'{p.get("building")} を出る'
    if k == "spend":
        return f'支出 {p.get("amount")}円({p.get("cat")} / {p.get("payee") or ""})'
    if k == "wage":
        return f'収入 {p.get("amount")}円({p.get("source") or p.get("payer") or ""})'
    if k == "sleep_start":
        return "就寝"
    if k == "wake_up":
        return f'起床({p.get("slept_steps")} step 睡眠)'
    if k == "event_host":
        return f'イベント「{p.get("title")}」を開催宣言(@{p.get("node")})'
    if k == "event_attend":
        return f'{w(p.get("host"))} のイベント「{p.get("title")}」に参加'
    if k == "group_found":
        return f'コミュニティ「{p.get("name")}」を結成'
    if k == "group_join":
        return f'{w(p.get("founder"))} の「{p.get("name")}」に加入'
    if k == "relation_tier":
        return f'{w(p.get("other"))} との関係が tier {p.get("tier")} へ'
    if k == "relation_break":
        return (f'{w(p.get("other"))} との関係が切れた'
                f'({p.get("from_tier")}→{p.get("to_tier")} / {p.get("cause")})')
    if k == "search":
        return f'検索「{p.get("query")}」({len(p.get("results") or [])} 件)'
    if k == "sns_post":
        return f'SNS 投稿「{p.get("text", "")}」'
    if k == "sns_read":
        return f'SNS を閲覧({p.get("n_posts")} 件)'
    if k == "study":
        return f'学習({p.get("subject")} / {p.get("role")})'
    if k == "production":
        return f'職場での産出「{p.get("output")}」({p.get("org")})'
    if k == "media_use":
        return f'{p.get("medium")}「{p.get("title")}」を見る({p.get("at")})'
    if k == "home_activity":
        return f'在宅「{p.get("act")}」({p.get("steps")} step)'
    if k == "emotion_label":
        return f'気分「{p.get("phrase") or p.get("label")}」'
    if k == "joint_activity":
        return (f'共同行動 {p.get("type")}(' +
                "、".join(w(i) for i in (p.get("with") or [])) + ")")
    if k == "train_ride":
        return (f'{p.get("line")} に乗車(混雑率 {p.get("load_factor")} / '
                f'{p.get("steps")} step)')
    if k == "ride":
        return f'{p.get("mode")} に乗車({p.get("fare")}円)'
    if k == "exit_area":
        return f'圏外へ退出(via {p.get("via")})'
    if k == "enter_area":
        return f'圏内へ帰還(via {p.get("via")})'
    if k == "transmission":
        return f'{w(p.get("from"))} から「{p.get("item_id")}」が伝わった'
    if k == "long_goal":
        return f'長期目標「{p.get("goal")}」'
    brief = _payload_brief(p)
    return f"{k}" + (f"({brief})" if brief else "")


def _thought_note(r: dict, thoughts: dict, journal_on: bool) -> str | None:
    """行為→思考のリンク 1 行(§5-4)。"""
    cid = r["llm_call_id"]
    if not cid:
        return None
    t = thoughts.get(cid) or {}
    role = r["payload"].get("llm_role") or t.get("purpose")
    src = "llm_role" if r["payload"].get("llm_role") else (
        "l1b.purpose" if t.get("purpose") else "不明")
    note = f"↳ 思考 `{cid}`(role={role or '?'} ・出所={src})"
    if journal_on and t.get("response"):
        note += f" 応答=「{_short(t['response'], 300)}」"
    return note


_PROVLINK_JA = {True: "ON", False: "OFF", "unknown": "判定不能(この日は行 0 件)"}


def render_md(trace: dict, *, journal_on: bool = False, max_lines: int = 60) -> str:
    names = {int(k): v for k, v in (trace.get("names") or {}).items()}
    ag, run = trace["agent"], trace["run"]
    L: list[str] = []
    title = who(names, ag["id"])
    L += [f"# 1日トレース — {title} / day {trace['day']} / `{run['name']}`", ""]
    prof = ", ".join(f"{k}={v}" for k, v in sorted(ag.items())
                     if k not in ("id", "_src") and v not in (None, "", -1))
    L += [f"- 個体: **{title}**" + (f"({prof})" if prof else ""),
          f"- ラン: `{run['dir']}` / Δt={run['dt_min']}分"
          f"(出所 {run['dt_src']}・1日={run['steps_per_day']}step)",
          f"- 窓: step {trace['window']['step_min']}〜{trace['window']['step_max']}"
          f" / sim_min {trace['window']['sim_min_min']}〜{trace['window']['sim_min_max']}"
          f"({hhmm(trace['window']['sim_min_min'])}〜"
          f"{hhmm(trace['window']['sim_min_max'])})",
          "- 日の定義: **暦日 = sim_min // 1440**(`world/clock.py: Clock.day` と同一。"
          "start_tod=07:00 のランでは day0 だけ 07:00 始まりの短い初日)", ""]

    # ---- 素材の在り無し(縮退の正直な告知)----
    m = trace["materials"]
    L += ["## 0. 素材(在るものだけで語る)", "",
          "| 素材 | 状態 |", "|---|---|",
          f"| L1 行為 | {', '.join(m['l1_events']) or '**無し**'} |",
          f"| l1b_llm(思考メタ) | {', '.join(m['l1b_llm']) or '無し'} |",
          f"| llm_journal(思考全文) | "
          f"{', '.join(m['llm_journal']) or '無し'}"
          f"{'(--with-journal で読込済)' if journal_on else '(未読込)'} |",
          f"| provlink(llm_role) | {_PROVLINK_JA.get(m.get('provlink'), '?')} |",
          f"| memory.parquet | {'有り' if m['memory'] else '無し'} |",
          f"| relations.parquet | {'有り' if m['relations'] else '無し'} |",
          f"| 名簿 | roster={'有' if m['roster'] else '無'} / "
          f"agents.json={'有' if m['agents_json'] else '無'} |", ""]
    for note in m["notes"]:
        L.append(f"> - {note}")
    if m["notes"]:
        L.append("")

    # ---- ① その日の計画 ----
    plan = trace["plan"]
    L += ["## 1. その日の計画(朝)", ""]
    if not plan["present"]:
        L += [f"- **{plan['reason']}**", ""]
    else:
        L += [f"- 確定: {hhmm(plan['created_at'])} / src=`{plan['src']}` / "
              f"model=`{plan['model']}` / version={plan['version']} / "
              f"{plan['n']} ブロック"
              + (f" / 呼 `{plan['llm_call_id']}`" if plan["llm_call_id"] else ""), ""]
        L += ["| # | 時間 | 行動 | 場所 | ねらい | 優先 | 柔軟 |",
              "|---|---|---|---|---|---|---|"]
        for b in plan["blocks"]:
            L.append(f"| {b['i']} | {hhmm(b['start'] or 0)}–{hhmm(b['end'] or 0)} |"
                     f" {b['act']} | {b['place']} | {b['aim']} |"
                     f" {b['priority']} | {b['flex']} |")
        L.append("")

    # ---- ② 時系列シーンカード ----
    L += ["## 2. 時系列シーンカード", ""]
    if not trace["scenes"]:
        L += ["- この日は 1 件もイベントが無い(在場していないか、"
              "退場後の暦日)。", ""]
    thoughts = trace["thoughts"]
    for i, s in enumerate(trace["scenes"], start=1):
        L += [f"### シーン {i}: {hhmm(s['t0'])}–{hhmm(s['t1'])} @ {s['place']}", ""]
        if s["partners"]:
            L.append("- 同席・相手: " + "、".join(who(names, p) for p in s["partners"]))
        if s["dist_m"]:
            L.append(f"- 移動: {s['dist_m']} m({s['bulk'].get('move_segment', 0)} 区間)")
        shown = 0
        for r in s["rows"]:
            if shown >= max_lines:
                L.append(f"- … 他 {len(s['rows']) - shown} 件(全行は §6 と json を参照)")
                break
            L.append(f"- `{hhmm(r['sim_min'])}` " + line_of(r, names))
            tn = _thought_note(r, thoughts, journal_on)
            if tn:
                L.append(f"    - {tn}")
            shown += 1
        L.append("")

    # ---- ③ 計画 vs 実行の突合 ----
    L += ["## 3. 計画 vs 実行の突合(物語の緊張は逸脱に宿る)", ""]
    if not plan["present"]:
        L += [f"- **{plan['reason']}** → 突合できない(0 件ではなく**未測定**)。", ""]
    else:
        c = plan["counts"]
        L += [f"- 帰属経路: **{plan['attribution']}**"
              + ("(payload の block 添字 = 多義ゼロ)" if plan["attribution"] == "block_index"
                 else "((act, place, start) の組で照合 = 同値ブロックがあると多義)"),
              f"- 多義 {plan['ambiguous']} 件 / 不照合 {plan['unmatched']} 件"
              f" / 再計画 {plan['n_replan']} 回 / 修復 {plan['n_repair']} 回"
              f" / 後退 {plan['n_fallback']} 回",
              f"- 実行 {c['executed']} / 破棄 {c['dropped']} / スライドのみ {c['slid']}"
              f" / 未着手 {c['untouched']}(= 計画 {len(plan['blocks'])} ブロック)", "",
              "| # | 計画 | 予定 | 結末 | 注記 |", "|---|---|---|---|---|"]
        _JA = {"executed": "実行", "dropped": "破棄", "slid": "スライド",
               "untouched": "未着手"}
        for b in plan["blocks"]:
            note = []
            if b.get("slid"):
                note.append(f"{b['slid']}分ずれ")
            if b.get("drop_reason"):
                note.append(f"理由={b['drop_reason']}")
            if b.get("cont"):
                note.append(f"もし{b['cont']['cond']}なら{b['cont']['then']}"
                            f"(適用={b['cont']['applied']})")
            if b.get("executed_at") is not None:
                note.append(f"開始 {hhmm(b['executed_at'])}")
            L.append(f"| {b['i']} | {b['act']}@{b['place']} |"
                     f" {hhmm(b['start'] or 0)} | **{_JA.get(b['status'], b['status'])}** |"
                     f" {'・'.join(note)} |")
        L.append("")

    # ---- ④ 夜の内省(本人の言葉)----
    L += ["## 4. 内省(本人の言葉)", ""]
    if not trace["reflections"]:
        L += ["- この日の `reflect` は 0 件。", ""]
    for rf in trace["reflections"]:
        L += [f"### {hhmm(rf['sim_min'])}({rf['context']} / {rf['mode']} / "
              f"書き戻し={rf['written_back']})", ""]
        if rf.get("summary"):
            L.append(f"> {rf['summary']}")
        if rf.get("belief"):
            L.append(">")
            L.append(f"> **信念**: {rf['belief']}")
        if rf.get("response_full"):
            L += ["", "<details><summary>LLM 応答全文(llm_journal)</summary>", "",
                  "```", rf["response_full"], "```", "", "</details>"]
        L.append("")

    # ---- 記憶・関係 ----
    if trace["memory"]:
        L += ["## 4b. その日に形成された記憶(memory.parquet)", "",
              "| 時刻 | 種 | 重み(初→最終) | 本文 |", "|---|---|---|---|"]
        for mm in trace["memory"]:
            imp0 = "?" if mm["importance_first"] is None else f"{mm['importance_first']:.2f}"
            imp1 = "?" if mm["importance_last"] is None else f"{mm['importance_last']:.2f}"
            L.append(f"| step {mm['step']} | {mm['kind']}/{mm['src']} |"
                     f" {imp0}→{imp1} |"
                     f" {_short(mm['text'], 120).replace('|', '/')} |")
        L.append("")
    rel = trace["relations"]
    if rel.get("present"):
        L += ["## 4c. 関係の変化(relations.parquet の day→day+1 差分)", "",
              "| 相手 | closeness | Δ | tier |", "|---|---|---|---|"]
        for d in rel["deltas"][:30]:
            L.append(f"| {who(names, d['other_id'])} |"
                     f" {d['closeness_before']}→{d['closeness_after']} |"
                     f" {d['delta']} | {d['tier_before']}→{d['tier_after']}"
                     f"{' **新規**' if d['new'] else ''} |")
        L.append("")
    elif rel.get("reason"):
        L += ["## 4c. 関係の変化", "", f"- 測れない: {rel['reason']}", ""]

    # ---- ⑤ 日次サマリ ----
    s = trace["stats"]
    L += ["## 5. 日次サマリ", "",
          f"- イベント {s['n_rows']} 行(主 {s['n_own']} / 受動 {s['n_patient']}"
          f" ・{s['n_kinds']} 種)",
          f"- 移動 {s['distance_m']} m / 経路 {s['n_routes']} 本 /"
          f" 到着ノード {s['n_nodes_visited']} 箇所",
          f"- 会話 {s['n_talk']} 件(speak {s['n_speak']} / hear {s['n_hear']} /"
          f" dm {s['n_dm']} / conversation {s['n_conversation']})"
          f" ・相手 {s['n_partners']} 人",
          f"- 支出 {s['spend_total']} 円 {json.dumps(s['spend_by_cat'], ensure_ascii=False)}"
          f" / 収入 {s['income']} 円 / 税 {s['tax']} 円", ""]
    if s["kinds"]:
        L += ["<details><summary>kind 別内訳</summary>", "",
              "```json", json.dumps(s["kinds"], ensure_ascii=False, indent=1,
                                    sort_keys=True), "```", "", "</details>", ""]

    # ---- L2 物語文 ----
    if trace.get("narrative"):
        nv = trace["narrative"]
        L += ["## 6. 物語文(L2・**ランの外の事後 LLM 生成**)", "",
              f"> backend=`{nv['backend']}` / model=`{nv['model']}`。"
              "以下は**生成物**であり一次事実ではない。数値・固有名は上の L0/L1 が正。", "",
              nv["text"].rstrip(), ""]

    # ---- L0 ----
    L += ["## 7. L0 生タイムライン(整列済み全行)", "",
          "並びは step → 相(計画→欲求/熟考→移動→行為→会話→内省)→ 記録順。"
          "`seq` が窓内の記録順なので**元の順序は常に復元できる**。", "",
          "| seq | step | 時刻 | kind | 側 | 内容 |", "|---|---|---|---|---|---|"]
    for r in trace["timeline"]:
        brief = _payload_brief(r["payload"], 100).replace("|", "/")
        L.append(f"| {r['seq']} | {r['step']} | {hhmm(r['sim_min'])} | {r['kind']} |"
                 f" {'主' if r['side'] == 'actor' else '受'} | {brief} |")
    L.append("")
    return "\n".join(L) + "\n"


_HTML_CSS = """
:root{color-scheme:light dark}
body{font-family:system-ui,"Segoe UI","Hiragino Sans","Noto Sans JP",sans-serif;
 max-width:1000px;margin:0 auto;padding:1.5rem;line-height:1.7}
h1{font-size:1.5rem;border-bottom:3px solid #888;padding-bottom:.3rem}
h2{font-size:1.2rem;margin-top:2rem;border-left:6px solid #888;padding-left:.5rem}
h3{font-size:1rem;margin-top:1.2rem}
table{border-collapse:collapse;width:100%;font-size:.85rem;margin:.5rem 0;
 display:block;overflow-x:auto}
th,td{border:1px solid #8886;padding:.25rem .5rem;text-align:left;vertical-align:top}
th{background:#8882}
.scene{border:1px solid #8886;border-radius:8px;padding:.6rem 1rem;margin:.8rem 0}
.scene .head{font-weight:700}
.meta{font-size:.85rem;opacity:.8}
.utter{border-left:4px solid #7a7;padding-left:.6rem;margin:.2rem 0}
.thought{font-size:.8rem;opacity:.75;margin-left:1.2rem}
.passive{opacity:.7}
blockquote{border-left:4px solid #77a;margin:.4rem 0;padding:.2rem .8rem;
 background:#7778;border-radius:4px}
code{background:#8882;padding:0 .2rem;border-radius:3px}
ul{margin:.2rem 0}
.warn{background:#c8802022;border-left:4px solid #c88020;padding:.4rem .8rem;
 border-radius:4px;font-size:.85rem}
"""


def render_html(trace: dict, *, journal_on: bool = False) -> str:
    e = html.escape
    names = {int(k): v for k, v in (trace.get("names") or {}).items()}
    ag, run, plan = trace["agent"], trace["run"], trace["plan"]
    title = who(names, ag["id"])
    P: list[str] = ["<!doctype html>", '<html lang="ja"><head><meta charset="utf-8">',
                    '<meta name="viewport" content="width=device-width,initial-scale=1">',
                    f"<title>1日トレース {e(title)} day{trace['day']}</title>",
                    f"<style>{_HTML_CSS}</style></head><body>"]
    P.append(f"<h1>1日トレース — {e(title)} / day {trace['day']} "
             f"/ <code>{e(run['name'])}</code></h1>")
    P.append(f"<p class='meta'>ラン <code>{e(run['dir'])}</code> ・ Δt={run['dt_min']}分"
             f"(出所 {e(str(run['dt_src']))})・ 窓 step "
             f"{trace['window']['step_min']}–{trace['window']['step_max']}"
             f"({e(hhmm(trace['window']['sim_min_min']))}–"
             f"{e(hhmm(trace['window']['sim_min_max']))})</p>")
    for note in trace["materials"]["notes"]:
        P.append(f"<p class='warn'>{e(note)}</p>")

    P.append("<h2>1. その日の計画</h2>")
    if not plan["present"]:
        P.append(f"<p class='warn'>{e(plan['reason'])}</p>")
    else:
        P.append("<table><tr><th>#</th><th>時間</th><th>行動</th><th>場所</th>"
                 "<th>ねらい</th><th>優先</th><th>柔軟</th></tr>")
        for b in plan["blocks"]:
            P.append(f"<tr><td>{b['i']}</td><td>{e(hhmm(b['start'] or 0))}–"
                     f"{e(hhmm(b['end'] or 0))}</td><td>{e(str(b['act']))}</td>"
                     f"<td>{e(str(b['place']))}</td><td>{e(str(b['aim']))}</td>"
                     f"<td>{e(str(b['priority']))}</td>"
                     f"<td>{e(str(b['flex']))}</td></tr>")
        P.append("</table>")

    P.append("<h2>2. 時系列シーンカード</h2>")
    thoughts = trace["thoughts"]
    for i, s in enumerate(trace["scenes"], start=1):
        P.append("<div class='scene'>")
        P.append(f"<div class='head'>シーン {i}: {e(hhmm(s['t0']))}–{e(hhmm(s['t1']))}"
                 f" @ {e(str(s['place']))}</div>")
        if s["partners"]:
            P.append("<div class='meta'>同席・相手: "
                     + e("、".join(who(names, p) for p in s["partners"])) + "</div>")
        if s["dist_m"]:
            P.append(f"<div class='meta'>移動 {s['dist_m']} m</div>")
        P.append("<ul>")
        for r in s["rows"]:
            cls = "" if r["side"] == "actor" else " class='passive'"
            P.append(f"<li{cls}><code>{e(hhmm(r['sim_min']))}</code> "
                     f"{e(line_of(r, names))}")
            tn = _thought_note(r, thoughts, journal_on)
            if tn:
                P.append(f"<div class='thought'>{e(tn)}</div>")
            P.append("</li>")
        P.append("</ul></div>")

    P.append("<h2>3. 計画 vs 実行</h2>")
    if not plan["present"]:
        P.append("<p class='warn'>計画が無いので突合できない(0 件ではなく未測定)。</p>")
    else:
        _JA = {"executed": "実行", "dropped": "破棄", "slid": "スライド",
               "untouched": "未着手"}
        P.append(f"<p class='meta'>帰属経路 <code>{e(plan['attribution'])}</code> ・"
                 f"多義 {plan['ambiguous']} / 不照合 {plan['unmatched']}</p>")
        P.append("<table><tr><th>#</th><th>計画</th><th>予定</th><th>結末</th>"
                 "<th>注記</th></tr>")
        for b in plan["blocks"]:
            note = []
            if b.get("slid"):
                note.append(f"{b['slid']}分ずれ")
            if b.get("drop_reason"):
                note.append(f"理由={b['drop_reason']}")
            if b.get("cont"):
                note.append(f"もし{b['cont']['cond']}なら{b['cont']['then']}")
            if b.get("executed_at") is not None:
                note.append(f"開始 {hhmm(b['executed_at'])}")
            P.append(f"<tr><td>{b['i']}</td><td>{e(str(b['act']))}@"
                     f"{e(str(b['place']))}</td><td>{e(hhmm(b['start'] or 0))}</td>"
                     f"<td><b>{e(_JA.get(b['status'], b['status']))}</b></td>"
                     f"<td>{e('・'.join(note))}</td></tr>")
        P.append("</table>")

    P.append("<h2>4. 内省(本人の言葉)</h2>")
    if not trace["reflections"]:
        P.append("<p class='meta'>この日の reflect は 0 件。</p>")
    for rf in trace["reflections"]:
        P.append(f"<h3>{e(hhmm(rf['sim_min']))} "
                 f"({e(str(rf['context']))} / {e(str(rf['mode']))})</h3>")
        if rf.get("summary"):
            P.append(f"<blockquote>{e(str(rf['summary']))}</blockquote>")
        if rf.get("belief"):
            P.append(f"<blockquote><b>信念</b>: {e(str(rf['belief']))}</blockquote>")
        if rf.get("response_full"):
            P.append("<details><summary>LLM 応答全文</summary><pre>"
                     f"{e(rf['response_full'])}</pre></details>")

    st = trace["stats"]
    P.append("<h2>5. 日次サマリ</h2><ul>")
    P.append(f"<li>イベント {st['n_rows']} 行(主 {st['n_own']} / 受動 "
             f"{st['n_patient']}・{st['n_kinds']} 種)</li>")
    P.append(f"<li>移動 {st['distance_m']} m / 経路 {st['n_routes']} 本</li>")
    P.append(f"<li>会話 {st['n_talk']} 件・相手 {st['n_partners']} 人</li>")
    P.append(f"<li>支出 {st['spend_total']} 円 / 収入 {st['income']} 円</li></ul>")

    if trace.get("narrative"):
        nv = trace["narrative"]
        P.append("<h2>6. 物語文(L2・ラン外の事後 LLM 生成)</h2>")
        P.append(f"<p class='warn'>backend=<code>{e(nv['backend'])}</code> "
                 f"model=<code>{e(str(nv['model']))}</code>。"
                 "これは生成物であり一次事実ではない。</p>")
        P.append("<blockquote>"
                 + "".join(f"<p>{e(par)}</p>" for par in nv["text"].split("\n") if par)
                 + "</blockquote>")

    P.append("<h2>7. L0 生タイムライン</h2>")
    P.append("<table><tr><th>seq</th><th>step</th><th>時刻</th><th>kind</th>"
             "<th>側</th><th>payload</th></tr>")
    for r in trace["timeline"]:
        P.append(f"<tr><td>{r['seq']}</td><td>{r['step']}</td>"
                 f"<td>{e(hhmm(r['sim_min']))}</td><td>{e(r['kind'])}</td>"
                 f"<td>{'主' if r['side'] == 'actor' else '受'}</td>"
                 f"<td>{e(_payload_brief(r['payload'], 100))}</td></tr>")
    P.append("</table></body></html>")
    return "\n".join(P) + "\n"


def render_top_md(res: dict, run_dir) -> str:
    L = [f"# その日いちばん物語的な個体の提案 — day {res['day']} / "
         f"`{Path(run_dir).name}`", "",
         f"- 母数: 個体 {res['n_agents']} / イベント {res['n_events']} 行",
         "- スコア = Σ_{その個体が出した kind} −log2(その日の件数比) "
         "= **希少度の自己較正和**(bit)。多様度 = distinct kind 数。",
         f"- 希少 kind の閾値: その日の件数 ≤ {res.get('rare_threshold')}",
         f"- {res.get('note', '')}", "",
         "| 順 | agent | 名前 | スコア(bit) | 多様度 | 件数 | 希少 kind |",
         "|---|---|---|---|---|---|---|"]
    for i, r in enumerate(res["top"], start=1):
        L.append(f"| {i} | {r['agent_id']} | {r['name'] or ''} | {r['score']} |"
                 f" {r['diversity']} | {r['n_events']} |"
                 f" {', '.join(r['rare_kinds'][:8])} |")
    L.append("")
    return "\n".join(L)


# =========================================================================== #
# 10. L2 物語文(§5-5。**ランの外**・既定 OFF・モック注入可)
# =========================================================================== #
NARRATE_SYSTEM = (
    "以下は渋谷の社会シミュレーションから機械的に抽出した、ある人物の 1 日の"
    "「シーンカード」と本人の内省です。\n"
    "これらに**書かれている事実だけ**を使って、その人物の一人称の日記を"
    "800 字程度の日本語で書いてください。\n"
    "禁止: カードに無い出来事・固有名・数値を足すこと。推測を断定で書くこと。\n"
    "計画どおりに行かなかった箇所があれば、そこを話の中心に置いてください。\n"
    "---\n")


def narrate_prompt(trace: dict) -> str:
    """L2 生成に渡すプロンプト(= シーンカード + 内省引用。決定論)。"""
    names = {int(k): v for k, v in (trace.get("names") or {}).items()}
    L = [NARRATE_SYSTEM,
         f"# 人物: {who(names, trace['agent']['id'])} / day {trace['day']}"]
    plan = trace["plan"]
    if plan["present"]:
        L.append("## 朝の計画")
        for b in plan["blocks"]:
            L.append(f"- {hhmm(b['start'] or 0)} {b['act']}@{b['place']}"
                     f"({b['priority']}) → 結末: {b['status']}"
                     + (f"(理由 {b['drop_reason']})" if b.get("drop_reason") else ""))
    L.append("## シーン")
    for i, s in enumerate(trace["scenes"], start=1):
        L.append(f"### {i}. {hhmm(s['t0'])}–{hhmm(s['t1'])} @ {s['place']}")
        if s["partners"]:
            L.append("同席: " + "、".join(who(names, p) for p in s["partners"]))
        for r in s["rows"][:40]:
            L.append(f"- {hhmm(r['sim_min'])} {line_of(r, names)}")
    if trace["memory"]:
        L.append("## その日に残った記憶")
        for mm in trace["memory"][:20]:
            L.append(f"- {mm['text']}")
    if trace["reflections"]:
        L.append("## 本人の内省(そのままの言葉)")
        for rf in trace["reflections"]:
            if rf.get("summary"):
                L.append(f"- {rf['summary']}")
            if rf.get("belief"):
                L.append(f"- 信念: {rf['belief']}")
    L.append("---\n上の事実だけで日記を書いてください。")
    return "\n".join(L)


def narrate(trace: dict, generate, *, backend: str = "injected",
            model: str | None = None) -> dict:
    """L2 物語文を 1 呼で作る。`generate(prompt) -> str` は**注入**(テストはモック)。

    ここが本スクリプト唯一の LLM 経路であり、**ラン後の成果物しか読まない**ので
    シムの決定論・呼数 k には触れられない(§5-5・R1 の外側)。
    """
    prompt = narrate_prompt(trace)
    text = generate(prompt)
    return {"backend": backend, "model": model, "text": str(text),
            "prompt_chars": len(prompt)}


def _make_generate(args):
    """CLI の `--narrate` からジェネレータを作る(mock は決定論・ネットワーク無し)。"""
    import summarize_run as SR                            # 遅延 import(重い依存を避ける)
    be = SR.make_backend(args.narrate, args.model, base_url=args.base_url,
                         api_key_env=args.api_key_env, host=args.host,
                         timeout_s=args.timeout)

    def _gen(prompt: str) -> str:
        return be.generate(prompt, rng_key="day_trace/narrate",
                           temperature=args.temperature,
                           max_tokens=args.max_tokens, think=False)
    return _gen


# =========================================================================== #
# 11. CLI
# =========================================================================== #
def _fail(msg: str) -> "NoReturn":                        # noqa: F821
    raise SystemExit(f"[day_trace] {msg}")


def _resolve_run(run_dir: str) -> str:
    if not os.path.isdir(run_dir):
        alt = os.path.join(_ROOT, "runs", run_dir)
        if os.path.isdir(alt):
            return alt
        _fail(f"ラン dir が無い: {run_dir}")
    return run_dir


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="任意エージェントの 1 日トレース(L0 生 / L1 シーンカード / "
                    "L2 物語文)。完全に事後・読み取り専用。")
    ap.add_argument("run_dir", help="ラン出力ディレクトリ(l1_events.parquet を含む)")
    ap.add_argument("--agent", type=int, default=None, help="対象 agent_id")
    ap.add_argument("--day", default="all",
                    help="暦日(sim_min//1440)。既定 all = 全日ぶん 1 ファイルずつ")
    ap.add_argument("--out", default=None,
                    help="出力先(既定 <run_dir>/analysis/day_trace)")
    ap.add_argument("--html", action="store_true", help="html も書く")
    ap.add_argument("--with-journal", action="store_true",
                    help="llm_journal からプロンプト/応答の全文を引く(全走査するので重い)")
    ap.add_argument("--top", type=int, default=None,
                    help="人選補助: その日いちばん物語的な個体を N 件提案して終わる")
    ap.add_argument("--gap-min", type=int, default=60,
                    help="シーンを切る無イベント間隔(分。既定 60)")
    ap.add_argument("--max-lines", type=int, default=60,
                    help="1 シーンカードに載せる行の上限(既定 60)")
    ap.add_argument("--narrate", default="off",
                    choices=["off", "mock", "ollama", "openai_compat", "anthropic"],
                    help="L2 物語文(ラン外の事後 LLM 1 呼)。**既定 off**")
    ap.add_argument("--model", default=None, help="--narrate 用モデル名")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=900)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key-env", default=None, help="API キーを入れる環境変数名")
    ap.add_argument("--host", default=None, help="ollama のホスト")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--print", dest="do_print", action="store_true",
                    help="md を標準出力にも出す")
    args = ap.parse_args(argv)

    run_dir = _resolve_run(args.run_dir)
    if not ls.l1_paths(run_dir):
        _fail(f"l1_events.parquet(および part 群)が無い: {run_dir}")
    days = observed_days(run_dir)
    if not days:
        _fail(f"L1 が空(1 行も無い): {run_dir}")

    out_dir = Path(args.out or (Path(run_dir) / "analysis" / "day_trace"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 人選補助モード ----
    if args.top is not None:
        if args.day == "all":
            _fail("--top は 1 日ぶんの提案なので --day N を指定すること"
                  f"(観測日 {days[0]}..{days[-1]})")
        day = _parse_day(args.day, days)
        res = rank_agents(run_dir, day, args.top)
        md = render_top_md(res, run_dir)
        (out_dir / f"top_day{day}.md").write_text(md, encoding="utf-8")
        (out_dir / f"top_day{day}.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1, sort_keys=True),
            encoding="utf-8")
        print(md)
        print(f"[written] {out_dir / f'top_day{day}.md'}")
        print(f"[written] {out_dir / f'top_day{day}.json'}")
        return 0

    if args.agent is None:
        _fail("--agent を指定すること(候補は --day N --top 20 で提案できる)")
    known = ls.distinct_agent_ids(run_dir)
    if args.agent not in known:
        near = sorted(known)[:10]
        _fail(f"agent {args.agent} は本ランの L1 の agent_id 列に 1 件も無い"
              f"(在籍 {len(known)} 体。先頭 10 件 = {near}。"
              f"受動側にしか現れない個体は本ツールの対象外)")

    targets = days if args.day == "all" else [_parse_day(args.day, days)]
    written: list[str] = []
    for day in targets:
        trace = build_trace(run_dir, args.agent, day,
                            with_journal=args.with_journal, gap_min=args.gap_min)
        if args.narrate != "off":
            trace["narrative"] = narrate(trace, _make_generate(args),
                                         backend=args.narrate, model=args.model)
        stem = f"agent{args.agent}_day{day}"
        md = render_md(trace, journal_on=args.with_journal,
                       max_lines=args.max_lines)
        (out_dir / f"{stem}.md").write_text(md, encoding="utf-8")
        (out_dir / f"{stem}.json").write_text(
            json.dumps(trace, ensure_ascii=False, indent=1, sort_keys=True,
                       default=str), encoding="utf-8")
        written += [str(out_dir / f"{stem}.md"), str(out_dir / f"{stem}.json")]
        if args.html:
            (out_dir / f"{stem}.html").write_text(
                render_html(trace, journal_on=args.with_journal), encoding="utf-8")
            written.append(str(out_dir / f"{stem}.html"))
        if args.do_print:
            print(md)
        else:
            s = trace["stats"]
            print(f"day {day}: {s['n_rows']} 行 / {s['n_kinds']} 種 / "
                  f"シーン {len(trace['scenes'])} 枚 / "
                  f"計画 {'あり' if trace['plan']['present'] else 'なし'} / "
                  f"内省 {len(trace['reflections'])} 件")
    for p in written:
        print(f"[written] {p}")
    return 0


def _parse_day(raw, days: list[int]) -> int:
    try:
        day = int(raw)
    except (TypeError, ValueError):
        _fail(f"--day は整数か all: {raw!r}")
    if day not in days:
        _fail(f"day {day} は本ランに存在しない(観測された day = "
              f"{days[0]}..{days[-1]})")
    return day


if __name__ == "__main__":
    raise SystemExit(main())
