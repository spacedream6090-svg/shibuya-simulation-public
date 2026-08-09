"""因果の帰属率 — 「その出来事は何が起こしたのか」を L1 から数える(第100バッチ IF-F)。

正典: docs/research/llm-world-interface-audit.md §2 / 分類表 src/society/observer/causality.py。

何を測るのか
------------
L1 の ``agent_id`` は**行為者ではない**。「この行が誰の視点で記録されたか」であって、
``speak`` なら話した本人、``hear`` なら**聞いた側**、``wage`` なら受け取った本人、
``weather`` なら -1(誰でもない)である。この 4 つが同じ 1 列に同居しているため、
素朴に agent_id で数えると「賃金を受け取った」が「その人が起こした変化」に化ける。

本スクリプトは ``observer/causality.py`` の**静的分類表**を唯一の源として、

  1. **kind × cause_type 行列**(どの種の出来事が何によって起きているか)
  2. **kind ごとの行為者帰属率**(= 行為者 id を復元できたイベントの割合)
  3. データに現れたのに**分類表に無い kind**(= 表の穴。警告として大きく出す)
  4. **agent_id=-1 の質量ランキング**(どの種が最も多くの「行為者不明」を作っているか)
  5. **device_id 別の内訳**(W2。どの装置がどれだけの出来事を作っているか。
     ``actor_id`` は int なので装置は名乗れない = 装置起因の行はこの列でしか所在が分からない)。
     W3 で **装置種(接頭辞)へのロールアップ**と、**刻めるのに 1 件も名乗っていない種**
     の一覧を足した(前者は ``pos:<ノード>`` のような動的 id が何百も並ぶのを畳むため、
     後者は「次にどこへ配線すべきか」を実測から出すため)

を出す。②を**単一のスカラーで代表させない**のは意図的である。全 kind を混ぜた
「帰属率 87%」のような数字は、``move_segment`` と ``health_update`` が毎 step 出る
tick 系であるためほぼ tick の比率しか表しておらず、指標として死んでいる。
だから **kind ごとの表が主**で、集計は「physics / schedule を明示的に除いた」ものだけを
1 行だけ併記する(除外した語を必ず印字する)。

既存ランでも動く(列が無くても分類できる)
------------------------------------------
分類は kind 単位の静的表なので、``observer.causality.enabled=false`` で回した
**過去のランでも同じ分類ができる**。ON のランには L1 に ``cause_type`` / ``actor_id``
の 2 列があるので、そのときは**表とエンジンの刻印を突き合わせ**、食い違いを報告する
(= 表が現実とずれていることの検出器。突き合わせられないランでは黙って表を信じる)。

読み取り専用
------------
runs/<name>/ を読むだけで、シム本体(src/society)の状態を 1 つも変えない。乱数を引かず
LLM も呼ばない。依存は 標準ライブラリ + pyarrow + 同ディレクトリの `l1_stream.py`
(W2-2 の共有ストリーミング読み)+ `society.observer.causality`(分類表)のみ。

メモリ
------
保持量は **O(distinct kind)**(実測 100 前後)。L1 の行数にも part 数にも比例しない。
読む列は step / agent_id / kind / payload の 4 本(+ ON のランでは 2 本)だけで、
``x`` / ``y`` は 1 バイトも読まない。payload の JSON パースは
``PATIENT_ACTOR_OVERRIDES`` に載る種(hear / opinion_shift / transmission / deliver)
だけに限る = 全行の JSON を作らない。

使い方
------
    python scripts/analyze_causality.py runs/night_llm_100a3d
    python scripts/analyze_causality.py runs/foo --json runs/foo/causality.json
    python scripts/analyze_causality.py runs/foo --out runs/foo/causality.md

終了コード: 分類表に無い kind がデータに現れた場合、または突き合わせで食い違いが
出た場合に 1(どちらも「表が現実とずれている」= 直すべき状態)。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:      # `python scripts/...` 以外の経路から import
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

# l1_stream が src/ も sys.path へ入れる(society.* を import できるようにする)。
from l1_stream import iter_record_batches, l1_paths, table_schema   # noqa: E402

from society import devices as DEV                                   # noqa: E402
from society.observer import causality as C                          # noqa: E402

#: L1 から**実際に読む列**。因果の集計は座標を 1 度も使わないので x / y は読まない。
BASE_COLUMNS: tuple[str, ...] = ("step", "agent_id", "kind", "payload")

#: ON のランにだけ存在するエンジン側の刻印列。
ENGINE_COLUMNS: tuple[str, ...] = ("cause_type", "actor_id")

#: W2 で足した 3 列目(装置の同一性)。**ENGINE_COLUMNS に入れない**のが要点:
#: W1 の ON ランには 2 列しか無いので、ここを必須にすると過去の ON ランが
#: 「刻印なし」に落ちて突き合わせが消える。あれば読む・無ければ黙る。
DEVICE_COLUMN = "device_id"

SCAN_BATCH_ROWS = 131_072


# --------------------------------------------------------------------------- #
# 1. 集計器(保持量は O(distinct kind))
# --------------------------------------------------------------------------- #
class KindTally:
    """kind 1 種ぶんの累積。行数に比例するものを 1 つも持たない。"""

    __slots__ = ("n", "n_actor", "n_neg_agent", "by_cause", "by_device",
                 "n_cause_mismatch", "n_actor_mismatch", "mismatch_example")

    def __init__(self) -> None:
        self.n = 0                       # 総件数
        self.n_actor = 0                 # 行為者 id を復元できた件数
        self.n_neg_agent = 0             # agent_id < 0(世界イベント)の件数
        self.by_cause: dict[str, int] = {}
        self.by_device: dict[str, int] = {}   # device_id 別(刻印のあるランだけ)
        self.n_cause_mismatch = 0        # エンジンの刻印と表が食い違った件数
        self.n_actor_mismatch = 0
        self.mismatch_example: dict | None = None

    def add_cause(self, cause: str) -> None:
        self.by_cause[cause] = self.by_cause.get(cause, 0) + 1

    def add_device(self, device_id: str) -> None:
        self.by_device[device_id] = self.by_device.get(device_id, 0) + 1


def scan(run_dir, *, batch_rows: int = SCAN_BATCH_ROWS) -> dict:
    """L1 を 1 パス走査して kind 別の累積を作る。

    戻り値 {"tallies": {kind: KindTally}, "engine_columns": bool,
            "unclassified": {kind: 件数}, "step_min": int, "step_max": int, "n": int}
    """
    paths = l1_paths(run_dir)
    if not paths:
        raise FileNotFoundError(
            f"L1 が見つからない: {Path(run_dir) / 'l1_events.parquet'}"
            "(canonical も完結 part も無い)")
    # 列の有無は footer だけで判る(1 行も読まない)。part 群では**全 part にある**ときだけ
    # 使う(途中で ON にしたランで列が半分しか無いのに「あるつもり」で読まないため)。
    names = [set(table_schema(p).names) for p in paths]
    engine_cols = all(set(ENGINE_COLUMNS) <= n for n in names)
    device_col = all(DEVICE_COLUMN in n for n in names)

    columns = list(BASE_COLUMNS) + (list(ENGINE_COLUMNS) if engine_cols else [])
    if device_col:
        columns.append(DEVICE_COLUMN)
    tallies: dict[str, KindTally] = {}
    unclassified: dict[str, int] = {}
    step_min, step_max, total = None, None, 0
    override_kinds = set(C.PATIENT_ACTOR_OVERRIDES)

    for batch in iter_record_batches(run_dir, columns, batch_rows=batch_rows):
        d = batch.to_pydict()
        kinds = d["kind"]
        agents = d["agent_id"]
        payloads = d.get("payload")
        steps = d.get("step")
        e_cause = d.get("cause_type") if engine_cols else None
        e_actor = d.get("actor_id") if engine_cols else None
        e_device = d.get(DEVICE_COLUMN) if device_col else None
        n = batch.num_rows
        total += n
        if steps:
            lo, hi = min(steps), max(steps)
            step_min = lo if step_min is None else min(step_min, lo)
            step_max = hi if step_max is None else max(step_max, hi)
        for i in range(n):
            kind = kinds[i]
            aid = agents[i]
            t = tallies.get(kind)
            if t is None:
                t = tallies[kind] = KindTally()
            t.n += 1
            if aid is not None and int(aid) < 0:
                t.n_neg_agent += 1
            # payload の JSON パースは**行為者を payload から戻す種だけ**(全行は作らない)
            payload = {}
            if kind in override_kinds and payloads:
                raw = payloads[i]
                if raw:
                    try:
                        payload = json.loads(raw)
                    except (TypeError, ValueError):
                        payload = {}
            table_cause = C.CAUSE_OF_KIND.get(kind)
            if table_cause is None:
                unclassified[kind] = unclassified.get(kind, 0) + 1
            # actor_of は分類表を引かないので、未分類 kind でも行為者は復元できる
            table_actor = C.actor_of(kind, aid, payload)
            # **刻印済みの行だけ**をエンジン側として扱う。列はあるのに値が null の行は
            # 「途中で ON にしたラン」(OFF の part と ON の part を unify_schemas で
            # 結合したもの)で必ず出る。ここを刻印扱いにすると突き合わせが偽陽性になる。
            stamped = engine_cols and e_cause[i] is not None
            # 行列は「エンジンの刻印があればそれ・無ければ表」で組む(= 実測の分布)
            t.add_cause((e_cause[i] if stamped else table_cause) or "?")
            if e_device is not None and e_device[i]:
                t.add_device(str(e_device[i]))
            actor = C.as_actor_id(e_actor[i]) if stamped else table_actor
            if actor is not None:
                t.n_actor += 1
            if stamped and table_cause is not None:
                if e_cause[i] != table_cause:
                    t.n_cause_mismatch += 1
                    if t.mismatch_example is None:
                        t.mismatch_example = {"field": "cause_type",
                                              "engine": e_cause[i], "table": table_cause}
                if actor != table_actor:
                    t.n_actor_mismatch += 1
                    if t.mismatch_example is None:
                        t.mismatch_example = {"field": "actor_id",
                                              "engine": e_actor[i], "table": table_actor}
    return {"tallies": tallies, "engine_columns": engine_cols,
            "device_column": device_col,
            "unclassified": unclassified, "n": total,
            # 0 行の L1(finalize は空でも canonical を書く)では -1 を返す = 欠測を
            # None のまま印字して「step None〜None」という読めない行を作らない。
            "step_min": -1 if step_min is None else int(step_min),
            "step_max": -1 if step_max is None else int(step_max),
            "paths": [p.name for p in paths]}


# --------------------------------------------------------------------------- #
# 2. レポート辞書
# --------------------------------------------------------------------------- #
def analyze(run_dir) -> dict:
    """ラン 1 本 → レポート辞書(Markdown / JSON の共通材料)。"""
    scanned = scan(run_dir)
    tallies: dict[str, KindTally] = scanned["tallies"]

    kinds: dict[str, dict] = {}
    matrix: dict[str, dict[str, int]] = {}
    per_cause: dict[str, int] = {c: 0 for c in C.CAUSE_TYPES}
    per_device: dict[str, int] = {}
    device_by_kind: dict[str, dict[str, int]] = {}
    per_cause_other = 0
    for kind in sorted(tallies):
        t = tallies[kind]
        table_cause = C.CAUSE_OF_KIND.get(kind)
        matrix[kind] = dict(sorted(t.by_cause.items()))
        if t.by_device:
            device_by_kind[kind] = dict(sorted(t.by_device.items()))
            for did, v in t.by_device.items():
                per_device[did] = per_device.get(did, 0) + int(v)
        for c, v in t.by_cause.items():
            if c in per_cause:
                per_cause[c] += v
            else:
                per_cause_other += v
        kinds[kind] = {
            "n": t.n,
            "cause_type": table_cause,          # None = 表に無い(警告セクションへ)
            "n_actor": t.n_actor,
            "actor_rate": (t.n_actor / t.n) if t.n else 0.0,
            "n_unattributed": t.n - t.n_actor,
            "n_neg_agent": t.n_neg_agent,
            "patient_override": C.PATIENT_ACTOR_OVERRIDES.get(kind),
            "n_cause_mismatch": t.n_cause_mismatch,
            "n_actor_mismatch": t.n_actor_mismatch,
            "mismatch_example": t.mismatch_example,
            "n_device": sum(t.by_device.values()),
        }

    # ---- 集計は「tick 系を明示的に除いた」ものだけ(単一スカラーの見出しを作らない)---- #
    excluded = sorted(C.TICK_CAUSES)
    inc_n = inc_actor = 0
    for kind, row in kinds.items():
        if row["cause_type"] in C.TICK_CAUSES:
            continue
        inc_n += row["n"]
        inc_actor += row["n_actor"]
    aggregate = {
        "label": f"tick 系({'/'.join(excluded)})を除く全 kind",
        "excluded_causes": excluded,
        "n_events": inc_n,
        "n_with_actor": inc_actor,
        "actor_rate": (inc_actor / inc_n) if inc_n else 0.0,
        "n_events_all": scanned["n"],
    }

    mismatches = {k: {"cause": v["n_cause_mismatch"], "actor": v["n_actor_mismatch"],
                      "example": v["mismatch_example"]}
                  for k, v in kinds.items()
                  if v["n_cause_mismatch"] or v["n_actor_mismatch"]}

    unattributed = sorted(
        ({"kind": k, "n": v["n"], "n_neg_agent": v["n_neg_agent"],
          "n_unattributed": v["n_unattributed"], "cause_type": v["cause_type"],
          "n_device": v["n_device"]}
         for k, v in kinds.items() if v["n_unattributed"]),
        key=lambda r: (-r["n_unattributed"], r["kind"]))

    # 装置 id の名簿照合(society/devices.py が唯一の源)。**捏造検出であって禁止ではない**:
    # 知らない id が出たら「名簿に足すべきか / 刻む側の誤りか」を人が決める材料として出す。
    unknown_devices = {d: n for d, n in sorted(per_device.items())
                       if not DEV.device_id_is_known(d)}
    # 装置**種**(接頭辞)へのロールアップ(W3)。``pos:<ノード>`` / ``org:<id>`` のような
    # 動的 id は個体が何百も出るので、id 別の表だけでは「どの層がどれだけ動いたか」が読めない。
    per_prefix: dict[str, int] = {}
    for did, n in per_device.items():
        head = str(did).partition(":")[0] or "?"
        per_prefix[head] = per_prefix.get(head, 0) + int(n)
    # **刻めるのに 1 件も名乗っていない種**(= 次に配線すべき対象の一覧)。
    # cause_type が DEVICE_STAMPABLE の中なのに device_id が 0 件の種だけを出す
    # (agent / physics / natural はそもそも装置の出来事ではないので出さない)。
    stampable_without_device = {
        k: v["n"] for k, v in sorted(kinds.items(), key=lambda kv: (-kv[1]["n"], kv[0]))
        if v["cause_type"] in C.DEVICE_STAMPABLE and not v["n_device"]}
    device_report = {
        "applicable": scanned["device_column"],
        "n_stamped": sum(per_device.values()),
        "per_device": dict(sorted(per_device.items())),
        "per_prefix": dict(sorted(per_prefix.items())),
        "by_kind": device_by_kind,
        "known_prefixes": sorted(DEV.DEVICE_ID_PREFIXES),
        "process_ids": list(DEV.PROCESS_DEVICE_IDS),
        "dynamic_prefixes": sorted(DEV.DYNAMIC_DEVICE_PREFIXES),
        "unknown_device_ids": unknown_devices,
        "stampable_without_device": stampable_without_device,
    }

    return {
        "run_dir": str(run_dir),
        "paths": scanned["paths"],
        "engine_columns": scanned["engine_columns"],
        "device_column": scanned["device_column"],
        "devices": device_report,
        "n_events": scanned["n"],
        "step_range": [scanned["step_min"], scanned["step_max"]],
        "cause_types": list(C.CAUSE_TYPES),
        "projection_4way": dict(C.PROJECTION_4WAY),
        "matrix": matrix,
        "per_cause": {**per_cause, **({"?": per_cause_other} if per_cause_other else {})},
        "per_cause_4way": _fold4(per_cause),
        "kinds": kinds,
        "aggregate": aggregate,
        "unclassified_kinds": dict(sorted(scanned["unclassified"].items())),
        "cross_validation": {
            "applicable": scanned["engine_columns"],
            "n_mismatch_kinds": len(mismatches),
            "n_cause_mismatch": sum(v["cause"] for v in mismatches.values()),
            "n_actor_mismatch": sum(v["actor"] for v in mismatches.values()),
            "by_kind": mismatches,
        },
        "unattributed_ranking": unattributed,
    }


def _fold4(per_cause: dict[str, int]) -> dict[str, int]:
    """6 語の件数 → 設計文書の 4 分類(schedule / physics は device の下位)。"""
    out: dict[str, int] = {}
    for c, n in per_cause.items():
        key = C.PROJECTION_4WAY.get(c, c)
        out[key] = out.get(key, 0) + n
    return dict(sorted(out.items()))


# --------------------------------------------------------------------------- #
# 3. Markdown 出力
# --------------------------------------------------------------------------- #
def _pct(x: float) -> str:
    return f"{x * 100:.1f} %"


def render_markdown(rep: dict) -> str:
    L: list[str] = []
    L.append("# 因果の帰属(cause_type × actor_id)")
    L.append("")
    L.append(f"- ラン: `{rep['run_dir']}`(L1 = {', '.join(rep['paths'])})")
    L.append(f"- イベント総数: {rep['n_events']:,} / step {rep['step_range'][0]}"
             f"〜{rep['step_range'][1]}")
    L.append(f"- L1 のエンジン刻印列(cause_type / actor_id): "
             f"{'あり(表と突き合わせる)' if rep['engine_columns'] else 'なし(静的表だけで分類する)'}")
    L.append("- 分類の単一の源: `src/society/observer/causality.py`")
    L.append("")

    # ---- 0. 未分類(あれば一番上で大きく警告する)------------------------------- #
    unc = rep["unclassified_kinds"]
    L.append("## 0. 分類表に無い kind")
    L.append("")
    if unc:
        L.append("> **警告: 表の穴。** データに現れたのに `CAUSE_OF_KIND` に無い kind がある。")
        L.append("> 下の行列・帰属率はこれらを `?` として数えており、その分だけ**過小報告**である。")
        L.append("")
        L.append("| kind | 件数 |")
        L.append("|---|---:|")
        for k, n in unc.items():
            L.append(f"| `{k}` | {n:,} |")
    else:
        L.append("なし(データに現れた全 kind が分類済み)。")
    L.append("")

    # ---- 1. cause_type 別の総量 ------------------------------------------------ #
    L.append("## 1. cause_type 別の総量")
    L.append("")
    L.append("> `schedule`(暦だけで決まる制度)と `physics`(状態で進む身体・空間の力学)は")
    L.append("> 設計文書の 4 分類では `device` の下位(DEVS の δ_int)。射影は下表のとおり。")
    L.append("")
    L.append("| cause_type | 件数 | 割合 | 4 分類 |")
    L.append("|---|---:|---:|---|")
    total = rep["n_events"] or 1
    for c, n in rep["per_cause"].items():
        L.append(f"| `{c}` | {n:,} | {_pct(n / total)} | "
                 f"`{rep['projection_4way'].get(c, '?')}` |")
    L.append("")
    L.append("4 分類へ畳んだもの: "
             + " / ".join(f"`{k}` {v:,}" for k, v in rep["per_cause_4way"].items()))
    L.append("")

    # ---- 2. kind × cause_type 行列 --------------------------------------------- #
    L.append("## 2. kind × cause_type 行列(件数)")
    L.append("")
    L.append("> 1 kind が複数の cause_type に散る場合は、**エンジンの刻印が静的表と食い違っている**")
    L.append("> ことを意味する(§4 の突き合わせ表を見ること)。列が無いランでは必ず 1 列に集まる。")
    L.append("")
    cols = [c for c in rep["cause_types"]
            if any(c in row for row in rep["matrix"].values())]
    extra = sorted({c for row in rep["matrix"].values() for c in row} - set(cols))
    cols = cols + extra
    L.append("| kind | " + " | ".join(f"`{c}`" for c in cols) + " | 計 |")
    L.append("|---|" + "---:|" * (len(cols) + 1))
    for kind in sorted(rep["matrix"], key=lambda k: (-rep["kinds"][k]["n"], k)):
        row = rep["matrix"][kind]
        cells = [f"{row[c]:,}" if row.get(c) else "·" for c in cols]
        L.append(f"| `{kind}` | " + " | ".join(cells) + f" | {rep['kinds'][kind]['n']:,} |")
    L.append("")

    # ---- 3. kind ごとの行為者帰属率 -------------------------------------------- #
    L.append("## 3. kind ごとの行為者帰属率")
    L.append("")
    L.append("> **単一のスカラーは出さない。** 全 kind を混ぜた帰属率は tick 系の比率しか")
    L.append("> 表さないため指標として死んでいる。集計は下の 1 行だけ(除外語を明記する)。")
    L.append("")
    ag = rep["aggregate"]
    L.append(f"**{ag['label']}**: {ag['n_with_actor']:,} / {ag['n_events']:,} = "
             f"**{_pct(ag['actor_rate'])}**"
             f"(全 kind の総数は {ag['n_events_all']:,} 件)")
    L.append("")
    L.append("| kind | cause_type | 件数 | 行為者あり | 帰属率 | agent_id=-1 | patient 補正 |")
    L.append("|---|---|---:|---:|---:|---:|---|")
    for kind in sorted(rep["kinds"], key=lambda k: (-rep["kinds"][k]["n"], k)):
        v = rep["kinds"][kind]
        ov = v["patient_override"]
        L.append(f"| `{kind}` | `{v['cause_type'] or '?'}` | {v['n']:,} | {v['n_actor']:,} | "
                 f"{_pct(v['actor_rate'])} | {v['n_neg_agent']:,} | "
                 f"{('payload[' + repr(ov) + ']') if isinstance(ov, str) else ('関数' if ov else '—')} |")
    L.append("")

    # ---- 4. 突き合わせ ---------------------------------------------------------- #
    L.append("## 4. 静的表 vs エンジンの刻印")
    L.append("")
    cv = rep["cross_validation"]
    if not cv["applicable"]:
        L.append("このランには刻印列が無い(`observer.causality.enabled=false` で回したラン)。")
        L.append("突き合わせは行わず、静的表だけで分類した。")
    elif not cv["by_kind"]:
        L.append(f"食い違い **0 件**(cause_type / actor_id とも全 {rep['n_events']:,} 件で一致)。")
        L.append("= 事後解析だけで既存ランを同じ精度で分類できる、ということの実測。")
    else:
        L.append(f"> **食い違い {cv['n_cause_mismatch']:,} 件(cause_type)/ "
                 f"{cv['n_actor_mismatch']:,} 件(actor_id)。**")
        L.append("> エンジンはその出来事を直接見ているので、**食い違いは表の側の誤り**である。")
        L.append("")
        L.append("| kind | cause 不一致 | actor 不一致 | 例 |")
        L.append("|---|---:|---:|---|")
        for k, v in sorted(cv["by_kind"].items(),
                           key=lambda kv: -(kv[1]["cause"] + kv[1]["actor"])):
            L.append(f"| `{k}` | {v['cause']:,} | {v['actor']:,} | `{v['example']}` |")
    L.append("")

    # ---- 4-2. 装置の同一性(device_id)------------------------------------------- #
    L.append("## 4-2. 装置の同一性(device_id)")
    L.append("")
    dev = rep["devices"]
    if not dev["applicable"]:
        L.append("このランには `device_id` 列が無い(W1 以前の ON ラン、または OFF ラン)。")
        L.append("装置起因の行が**どの装置か**は復元できないため、この節は空である。")
    elif not dev["per_device"]:
        L.append("列はあるが刻印が 1 件も無い(装置スコープを通る機能が全て OFF のラン)。")
    else:
        L.append("> `actor_id` は int(個体 id)なので装置は名乗れない。装置起因の行が"
                 "「行為者不明」に落ちるのを止めるための別列である。")
        L.append(f"> 刻印のある行: **{dev['n_stamped']:,}** 件 / 名簿の源: "
                 "`src/society/devices.py`")
        L.append("")
        # ---- 装置**種**へのロールアップを先に出す(動的 id は個体が多いため)---- #
        L.append("**装置種(接頭辞)別**"
                 f"(動的 id の族 = {', '.join('`' + p + '`' for p in dev['dynamic_prefixes'])}):")
        L.append("")
        L.append("| 装置種 | 件数 | 個体数 |")
        L.append("|---|---:|---:|")
        for head, n in sorted(dev["per_prefix"].items(), key=lambda kv: (-kv[1], kv[0])):
            n_ind = sum(1 for d in dev["per_device"]
                        if str(d).partition(":")[0] == head)
            L.append(f"| `{head}` | {n:,} | {n_ind:,} |")
        L.append("")
        L.append("| device_id | 件数 | 内訳(kind) |")
        L.append("|---|---:|---|")
        for did, n in sorted(dev["per_device"].items(), key=lambda kv: (-kv[1], kv[0])):
            parts = sorted(((k, v[did]) for k, v in dev["by_kind"].items()
                            if did in v), key=lambda kv: (-kv[1], kv[0]))
            L.append(f"| `{did}` | {n:,} | "
                     + " / ".join(f"`{k}` {v:,}" for k, v in parts[:6])
                     + (" / …" if len(parts) > 6 else "") + " |")
        L.append("")
        # ---- 刻めるのに名乗っていない種 = 次に配線すべき対象 ---- #
        gap = dev["stampable_without_device"]
        L.append("**刻めるのに装置 id が 1 件も無い種**(cause_type が device/schedule/"
                 "boundary なのに名乗っていない = 次に配線する候補):")
        L.append("")
        if gap:
            L.append("| kind | 件数 |")
            L.append("|---|---:|")
            for k, n in list(gap.items())[:20]:
                L.append(f"| `{k}` | {n:,} |")
            if len(gap) > 20:
                L.append(f"| … | 残り {len(gap) - 20} 種は JSON 側 |")
        else:
            L.append("なし(刻める種はすべてどこかの装置が名乗っている)。")
        L.append("")
        if dev["unknown_device_ids"]:
            L.append("> **警告: 名簿に無い装置 id。** `society/devices.py` の"
                     " `DEVICE_ID_PREFIXES` に無い接頭辞が L1 に出ている"
                     "(名簿に足すか、刻む側を直すかの判断が要る)。")
            L.append("")
            L.append("| device_id | 件数 |")
            L.append("|---|---:|")
            for did, n in dev["unknown_device_ids"].items():
                L.append(f"| `{did}` | {n:,} |")
            L.append("")
    L.append("")

    # ---- 5. 行為者不明の質量ランキング ------------------------------------------ #
    L.append("## 5. 行為者が復元できないイベントの質量")
    L.append("")
    L.append("> 「どの種が最も多くの『誰が起こしたか分からない出来事』を作っているか」。")
    L.append("> 上から順に潰すのが、帰属率を上げる費用対効果の高い順序である。")
    L.append("")
    L.append("> `device_id` 列があるランでは「装置 id で名乗れた件数」も併記する。")
    L.append("> **装置 id で名乗れていない行 = 次に装置スコープを通すべき対象**である")
    L.append("> (行為者 id が復元できない出来事は、装置が名乗れば所在が分かる)。")
    L.append("")
    if rep["unattributed_ranking"]:
        L.append("| kind | cause_type | 帰属不能 | うち agent_id=-1 | device_id あり | 総件数 |")
        L.append("|---|---|---:|---:|---:|---:|")
        for r in rep["unattributed_ranking"][:40]:
            L.append(f"| `{r['kind']}` | `{r['cause_type'] or '?'}` | {r['n_unattributed']:,} | "
                     f"{r['n_neg_agent']:,} | {r.get('n_device', 0):,} | {r['n']:,} |")
        if len(rep["unattributed_ranking"]) > 40:
            L.append(f"| … | | | | 残り {len(rep['unattributed_ranking']) - 40} 種は JSON 側 |")
    else:
        L.append("なし(全イベントで行為者 id が復元できている)。")
    L.append("")
    L.append("---")
    L.append("")
    L.append("*本スクリプトは読み取り専用(シム本体ゼロタッチ・乱数0・LLM呼0)。*")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# 4. CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir", help="runs/<name>")
    ap.add_argument("--out", help="Markdown の出力先(省略時は標準出力)")
    ap.add_argument("--json", dest="json_out", help="JSON の出力先")
    args = ap.parse_args(argv)

    rep = analyze(Path(args.run_dir))
    md = render_markdown(rep)
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        print(md)
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(rep, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8")
        print(f"wrote {args.json_out}")
    bad = (bool(rep["unclassified_kinds"])
           or bool(rep["cross_validation"]["by_kind"])
           or bool(rep["devices"]["unknown_device_ids"]))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
