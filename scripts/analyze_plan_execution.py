#!/usr/bin/env python
"""朝の計画(day_plan.place)と実行動の突合 — W2-4(小粒 wave2)。

    python scripts/analyze_plan_execution.py runs/<name> [--out DIR] [--map PATH]

**読み出し専用**: `runs/<name>/{l1_events.parquet, config.yaml, agents.json}` と
地図 JSON を読むだけ。`src/` も `conf/` も 1 バイトも触らない
(`observer/metrics_spec.py` の**凍結 14 ファイル**にも本スクリプトは含まれない
= `metrics_spec_hash` は無風)。

何を解く問題か
--------------
台帳 PENDING の実測: 「`day_plan.place` の語彙と地図ノード名の重なりが 266 中 5」。
これは **`place` を地図ノード名(店名・地名)と文字列比較していた**ことによる測り損ねで、
`day_plan.place` はそもそも**具体店名ではない**(`day_plan.py` の PLACE_CATS の
docstring が明言している)。`place` は**カテゴリ**であり、具体ノードへの写像は
`day_plan.resolve_place` が実行時に決める。よって「計画どおり行動したか」は
**語彙の文字列一致では測れず**、次の 2 つを繋いで初めて測れる:

  ① 計画ブロック → 実行イベントの帰属(どの行為がどのブロックの実行か)
  ② 実行到達ノード → 場所カテゴリの**逆引き**(そのノードは計画の種類の場所か)

**対応表を自前で発明しない**(本スクリプトの中心的な設計判断)
-------------------------------------------------------------
正典は **`day_plan.py` の実行側コードそのもの**である。本スクリプトは写しを持たず、
次を**そのまま import** する(= 規則の二重実装をしない):

  `DP.PLACE_CATS`   場所語彙 15 種の唯一の定義
  `DP.PLACE_ALIAS`  列挙外語の 1 段吸収表(被覆診断に使う)
  `DP.ACT_PLACE` / `DP.ACTS` / `DP.PURPOSES` / `DP.PRIORITIES` / `DP.FLEXES`
  `DP.OUTDOOR_PLACES` / `DP.INDOOR_SWAP`   contingency の差し替え表
  `DP.build_cfg`    conf の正準化(round_min / grace_min / day_end_min 等)
  `DP._round_to`    時刻の丸め(再計画のずらし先の再現に使う)

`resolve_place` 自体は **import できない**(生きた `sim` = city / commerce 設定 /
agent の所持金・訪問履歴を要求し、ラン後の成果物からは再構成できない)。そこで
本スクリプトは resolve_place を再実行せず、その**必要条件だけ**を検査する:

  resolve_place(place) の候補集合は必ず `city.pois_by_cat(place)` の部分集合である
  (営業・残高フィルタは pool を**狭めるだけ**で、外の要素を足さない)。
  ⇒ 到達ノードが `pois_by_cat(place)` に属さないなら、それは**確実に**計画の
    場所種別ではない。属するなら計画と整合する。

これは「resolve_place が選んだのと同じ 1 ノードか」ではなく「**計画した種類の場所へ
行ったか**」の検査である(その旨を出力にも明記する)。3 つの特別分岐
(`home` = agent.home_node / `work` = work_node・part_time / `street` = None を返して
習慣ポリシーへ委譲)も resolve_place の分岐と 1 対 1 に対応させ、
`tests/test_plan_execution.py` の **AST テスト**が分岐の増減を検知する。

2 つの帰属経路(どちらも②の逆引きは同一・違うのは①の帰属だけ)
--------------------------------------------------------------
(a) **PROV join 経路**(`observer.llm_link: true` のラン = 正確)
    IF-A(第93)が行為イベントへ `(llm_call_id, role)` を刻む。朝の計画から出た移動は
    `llm_call_id == plan_created.llm_call_id` かつ `payload.llm_role == "plan"` を持つので、
    「その移動が朝の計画由来である」ことが**イベント自身で自明**になる。
(b) **同 step 帰属経路**(llm_link OFF ラン = 自動フォールバック)
    `plan_block_start` と同じ (agent_id, step) の `route_start` を採る。1 step 1 行為
    なので実務上ほぼ一意だが、**証拠は無い**(role 刻印が無いため、その移動が習慣
    ポリシー由来である可能性を排除できない)。出力の `node_src` で区別できる。
(c) **留まった経路**(W4-F 以降のラン。移動イベントが 1 件も無い step のみ)
    `plan_block_start.node`(実行時の現在ノード)を採る。`move_to` は経路が張れなければ
    `route_start` を出さずに return するので、「同 step に移動イベントが無い」
    ⇔「個体はそのノードから動かなかった」が成り立つ。よってこれは推測ではない。

どの経路でも取れないときは **`unknown` と正直に出す**(当てにいかない):
  street_delegated … `place="street"` = 計画が行き先の特定を habit へ委譲した
  no_move_event    … 同 step に移動イベントが無く、`node` も記録されていない旧ラン
  node_off_map     … 到達ノードが地図に無い(地図違い)
  home_unknown / work_unknown … agents.json から自宅/職場ノードが判らない
  category_absent_on_map      … その場所カテゴリの POI が地図に 1 件も無い

W4-F(2026-08-08)で **src 側が 3 つの記録を足した**ため、下の限界 1・2 と未解決理由
`no_move_event` は**新しいランでは消える**。旧ランは 1 行も変えずに読めるよう、
本スクリプトは **新フィールドがあればそれを使い、無ければ従来経路へ自動で落ちる**:

  `agents.json.work_node` / `part_time_node`
      … `day_plan._work_node(agent)` の入力そのもの。あれば職場判定は実ノード
        (`exact_node`)、無ければ従来の建物近似(`approx_building`)。
  `plan_block_start.node`
      … 実行を決めた瞬間の**現在ノード(移動前)**。行き先ではないので、同 step に
        移動イベントが**無いときに限り**「個体はそこに留まった = そこがブロックの
        場所」として採る(`node_src="block_node"`)。移動が起きたランでは使わない。
  `plan_block_start` / `plan_block_drop` / `plan_slide` の `block`
      … `plan_created.blocks[]` の添字。あれば台帳再生の照合を一切行わず添字で直接
        帰属する(多義・不照合が構造的にゼロになる)。

正直な限界(5 件)
------------------
1. **`work` は旧ランでのみ近似**である。`agents.json` に `work_node` が無いランでは
   職場ノード候補 = その建物の入口ノード ∪ その建物内 POI のノード、とした
   (`approx_building`)。`work_building` も空の個体は `work_unknown`。
   新ランでは `{work_node, part_time_node}` の実ノード集合で判定する(`exact_node`)。
   ★ただし agents.json は**書き出した時点のスナップショット**であり、転職
     (organizations)で run 中に職場が変わった個体は取りこぼしうる。だから受理集合は
     本業とバイト先の**和**にしてある(`_work_node` は本業を優先するので和は上位集合
     = 必要条件の検査という本スクリプトの立場と整合する)。
2. **ブロックの同定は、添字が無いランでは L1 の再生である**。`plan_created.blocks[]`
   から台帳を起こし、`plan_slide` / `plan_block_drop` / `plan_cont_fire` /
   `plan_replan` を順に当てて `(act, place, aim, priority, flex, start)` で照合する。
   多義・不照合は `matching.ambiguous` / `matching.unmatched` に**数えて出す**
   (黙って捨てない)。`plan_cont_fire` は第93 から添字を持つので**そちらを正典**とし、
   直前の drop/slide の帰属が食い違えば差し戻す。W4-F 以降のランは 3 種すべてが
   添字を持つので、この再生経路は 1 度も走らない(`ledger.by_index` で確認できる)。
3. **時間帯一致は「日中にずらされなかったこと」**として測る。実行判定
   (`day_plan.current_block`)は常に**現在の**窓の中でしか起きないので、「窓の外で
   実行された」は原理的に観測できない。朝の窓と実行時の窓が同一 = 遵守、とする
   (朝の**修復**で入ったずらしは `plan_created` の時点で朝の窓に織り込み済み)。
4. **寄り道(P2 S4 `detour`)は逸脱として数える**。`route_start.dest` は寄り道先なので、
   計画が解決したノード(`detour.from_dest`)と分けて両方を出す。
5. **day_plan OFF のランでは測る対象が存在しない**。`plan_created` が 0 件のとき、
   遵守率は 0.0 ではなく **null**(測っていない)で出す。語彙の被覆診断だけは
   地図と `day_plan.py` だけで決まるので OFF ランでも出る。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))          # 正典 = 実装側 day_plan.py

from society.cognition import day_plan as DP        # noqa: E402

SCHEMA = 1

# --------------------------------------------------------------------------- #
# 語彙の 3 分類(**day_plan.resolve_place の分岐と 1 対 1**。新語は 1 つも作らない)
#   resolve_place: place=="home"  → agent.home_node
#                  place=="work"  → _work_node(agent)(本業 → バイト先)
#                  place=="street"→ None(= 習慣ポリシーへ委譲)
#                  それ以外       → city.pois_by_cat(place) を絞って 1 つ選ぶ
# --------------------------------------------------------------------------- #
AGENT_RELATIVE_PLACES: tuple[str, ...] = ("home", "work")
DELEGATED_PLACES: tuple[str, ...] = ("street",)
MAP_PLACES: tuple[str, ...] = tuple(
    p for p in DP.PLACE_CATS if p not in AGENT_RELATIVE_PLACES + DELEGATED_PLACES)
# 計画語彙 → 地図カテゴリの追加照会先(day_plan.MAP_FALLBACK_CATS と同一の表を import。
# 2026-08-07 追随: resolve_place はこの表の分だけ pois_by_cat を追加照会するようになった
# ので、必要条件の検査も同じ受理集合で行う。表を発明せず getattr で存在に追随する)。
_FALLBACK: dict[str, tuple[str, ...]] = dict(getattr(DP, "MAP_FALLBACK_CATS", {}) or {})


def _accept_cats(place: str) -> tuple[str, ...]:
    """place の到達判定で受理する地図カテゴリ(自カテゴリ+fallback・順序固定)。"""
    return (place,) + tuple(_FALLBACK.get(place, ()))

PLAN_KINDS: tuple[str, ...] = (
    "plan_created", "plan_block_start", "plan_block_drop", "plan_slide",
    "plan_replan", "plan_cont_fire", "plan_repair", "plan_fallback")
MOVE_KINDS: tuple[str, ...] = ("route_start", "detour")
WANT_KINDS: frozenset = frozenset(PLAN_KINDS + MOVE_KINDS)

# 帰属できなかった理由の有限語彙(出力キーの唯一の源)
UNRESOLVED_REASONS: tuple[str, ...] = (
    "street_delegated", "no_move_event", "node_off_map",
    "home_unknown", "work_unknown", "category_absent_on_map", "no_map")


# --------------------------------------------------------------------------- #
# 読み込み(すべて読み取り専用)
# --------------------------------------------------------------------------- #
def load_events(run_dir: str) -> list[dict]:
    """l1_events.parquet から本解析に要る種だけを **記録順のまま** 取り出す。

    ★並べ替えない: 同 step 内の並び(sweep → contingency → block_start)がそのまま
      台帳再生の順序契約になっている。
    """
    import pyarrow.parquet as pq
    path = os.path.join(run_dir, "l1_events.parquet")
    if not os.path.isfile(path):
        raise SystemExit(f"[plan_execution] l1_events.parquet が無い: {run_dir}")
    cols = ["step", "sim_min", "agent_id", "kind", "payload", "llm_call_id"]
    have = set(pq.read_schema(path).names)
    table = pq.read_table(path, columns=[c for c in cols if c in have])
    d = table.to_pydict()
    n = len(d["kind"])
    out: list[dict] = []
    for i in range(n):
        kind = d["kind"][i]
        if kind not in WANT_KINDS:
            continue
        raw = d["payload"][i]
        payload = json.loads(raw) if isinstance(raw, str) else (raw or {})
        out.append({
            "seq": i,
            "step": int(d["step"][i] or 0),
            "sim_min": int(d["sim_min"][i] or 0),
            "agent_id": int(d["agent_id"][i] if d["agent_id"][i] is not None else -1),
            "kind": str(kind),
            "payload": payload if isinstance(payload, dict) else {},
            "llm_call_id": (str(d["llm_call_id"][i])
                            if d.get("llm_call_id") and d["llm_call_id"][i] else None),
        })
    return out


def load_cfg(run_dir: str) -> dict:
    """ランの `planning.day_plan` を **DP.build_cfg で正準化**して返す(写しを作らない)。"""
    raw = None
    path = os.path.join(run_dir, "config.yaml")
    if os.path.exists(path):
        try:
            import yaml
            cfg = yaml.safe_load(open(path, encoding="utf-8")) or {}
            raw = (cfg.get("planning", {}) or {}).get("day_plan")
        except Exception:                            # noqa: BLE001 (旧 config 互換)
            raw = None
    return DP.build_cfg(raw)


def map_path_of(run_dir: str, explicit: str | None = None) -> str | None:
    """使う地図 JSON の実パス(明示 > ランの config.yaml world.map > 既定)。"""
    if explicit:
        return explicit if os.path.exists(explicit) else None
    mp = None
    path = os.path.join(run_dir, "config.yaml")
    if os.path.exists(path):
        try:
            import yaml
            cfg = yaml.safe_load(open(path, encoding="utf-8")) or {}
            mp = (cfg.get("world", {}) or {}).get("map")
        except Exception:                            # noqa: BLE001
            mp = None
    cands = []
    if mp:
        cands.append(mp if os.path.isabs(mp) else os.path.join(REPO_ROOT, mp))
    cands.append(os.path.join(REPO_ROOT, "data", "shibuya_osm.json"))
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def load_map(path: str | None) -> dict | None:
    """地図 JSON → 逆引き索引(node → 場所カテゴリ集合)。

    ★これが「place の語彙 → 具体ノード」の**唯一の対応表**であり、シム側の
      `city.pois_by_cat(cat)` が引くのと**同じ `pois[].cat` 欄**を引いている。
    """
    if not path or not os.path.exists(path):
        return None
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:                                # noqa: BLE001
        return None
    node_cats: dict[str, set] = defaultdict(set)
    cat_counts: dict[str, int] = defaultdict(int)
    for p in raw.get("pois", []) or []:
        node, cat = p.get("node"), p.get("cat")
        if node is None or cat is None:
            continue
        node_cats[str(node)].add(str(cat))
        cat_counts[str(cat)] += 1
    building_nodes: dict[str, set] = defaultdict(set)
    for b in raw.get("buildings", []) or []:
        bid, ent = b.get("id"), b.get("entrance")
        if bid is not None and ent:
            building_nodes[str(bid)].add(str(ent))
    for p in raw.get("pois", []) or []:
        bid, node = p.get("building"), p.get("node")
        if bid and node:
            building_nodes[str(bid)].add(str(node))
    return {
        "path": str(path),
        "nodes": {str(n["id"]) for n in (raw.get("nodes", []) or []) if "id" in n},
        "node_cats": {k: set(v) for k, v in node_cats.items()},
        "cat_counts": dict(cat_counts),
        "building_nodes": {k: set(v) for k, v in building_nodes.items()},
        "n_pois": len(raw.get("pois", []) or []),
    }


def load_agents(run_dir: str, mapidx: dict | None) -> dict:
    """agents.json → {id: {home, work_building, work_nodes, work_src}}。

    ★W4-F 以降のランは `work_node` / `part_time_node`(= `day_plan._work_node` の入力
      そのもの)を持つので、職場ノードは**実ノード**で判る(`work_src="exact_node"`)。
      持たない旧ランだけ、建物 id から候補ノード集合を起こす近似へ落ちる
      (`work_src="approx_building"`。限界 1)。
    """
    out: dict[int, dict] = {}
    path = os.path.join(run_dir, "agents.json")
    if not os.path.isfile(path):
        return out
    try:
        rows = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:                                # noqa: BLE001
        return out
    bnodes = (mapidx or {}).get("building_nodes", {})
    for r in rows if isinstance(rows, list) else []:
        if not isinstance(r, dict) or "id" not in r:
            continue
        wb = str(r.get("work_building") or "")
        exact = {n for n in (str(r.get("work_node") or ""),
                             str(r.get("part_time_node") or "")) if n}
        out[int(r["id"])] = {
            "home": str(r.get("home") or ""),
            "work_building": wb,
            "work_nodes": exact if exact
                          else (set(bnodes.get(wb, ())) if wb else set()),
            "work_src": "exact_node" if exact else "approx_building"}
    return out


# --------------------------------------------------------------------------- #
# ① 語彙の被覆(**ランが無くても測れる**。地図 × day_plan.py だけで決まる)
# --------------------------------------------------------------------------- #
def place_coverage(mapidx: dict | None) -> dict:
    """15 種の場所語彙のうち、この地図で**実際に解決できるもの**はどれか。

    台帳の「語彙の重なり 266 中 5」を置き換える正しい測り方。比較すべきは
    `place`(カテゴリ)と**ノード名**ではなく、`place` と **POI カテゴリ**である。
    """
    base = {
        "plan_places": list(DP.PLACE_CATS),
        "agent_relative": list(AGENT_RELATIVE_PLACES),
        "delegated": list(DELEGATED_PLACES),
        "map_backed": list(MAP_PLACES),
    }
    if mapidx is None:
        return {**base, "map": None,
                "note": "地図 JSON が見つからない = 被覆は測れない(捏造しない)"}
    cc = mapidx["cat_counts"]
    per = {p: int(cc.get(p, 0)) for p in MAP_PLACES}
    # 実効件数 = fallback 照会先込み(resolve_place が実際に引ける候補の総数)
    eff = {p: sum(int(cc.get(c, 0)) for c in _accept_cats(p)) for p in MAP_PLACES}
    resolvable = sorted(p for p, n in eff.items() if n > 0)
    unresolvable = sorted(p for p, n in eff.items() if n == 0)
    # 地図にはあるが計画語彙に無いカテゴリ(= LLM がその語を書いても届かない)
    extra = sorted(c for c in cc if c not in DP.PLACE_CATS)
    # ★別名表が地図の実カテゴリを**踏み潰している**組(最重要の不整合)。
    #   例: 地図の cat は "school" なのに PLACE_ALIAS["school"] = "education" へ
    #       書き換えられる組。MAP_FALLBACK_CATS が受け皿になっていれば実効件数で
    #       解消済みと判定される(2026-08-07 の education→school 是正)。
    shadowed = []
    for alias, target in sorted(DP.PLACE_ALIAS.items()):
        if alias in cc and eff.get(target, 0) < cc[alias]:
            shadowed.append({"alias": alias, "rewritten_to": target,
                             "n_pois_alias": int(cc[alias]),
                             "n_pois_target": int(eff.get(target, 0))})
    return {
        **base,
        "map": mapidx["path"],
        "n_pois": mapidx["n_pois"],
        "n_pois_by_place": {k: per[k] for k in sorted(per)},
        "n_pois_effective": {k: eff[k] for k in sorted(eff)},
        "fallback_cats": {k: list(v) for k, v in sorted(_FALLBACK.items())},
        "resolvable": resolvable,
        "unresolvable": unresolvable,
        "coverage_rate": (round(len(resolvable) / len(MAP_PLACES), 4)
                          if MAP_PLACES else None),
        "map_cats_not_in_plan_vocab": extra,
        "alias_shadowed": shadowed,
    }


# --------------------------------------------------------------------------- #
# ② 台帳の再生(plan_created を起点に L1 を順に当てる)
# --------------------------------------------------------------------------- #
def _new_day(aid: int, day: int, e: dict, p: dict) -> dict:
    blocks = []
    for i, b in enumerate(p.get("blocks") or []):
        plan = {"start": int(b.get("start") or 0), "end": int(b.get("end") or 0),
                "place": str(b.get("place") or ""), "act": str(b.get("act") or ""),
                "aim": str(b.get("aim") or ""),
                "priority": str(b.get("priority") or ""),
                "flex": str(b.get("flex") or "")}
        blocks.append({"i": i, "plan": plan, "cur": dict(plan), "state": "todo",
                       "n_slide": 0, "slid_seen": None, "cont": None,
                       "exec": None, "drop": None})
    return {"agent": aid, "day": day, "step": e["step"], "call_id": e["llm_call_id"],
            "src": str(p.get("src") or ""), "model": str(p.get("model") or ""),
            "n_cont": int(p.get("n_cont") or 0), "blocks": blocks, "replans": 0,
            "_pend": None}


def _snap(b: dict) -> dict:
    return {"state": b["state"], "cur": dict(b["cur"]), "slid_seen": b["slid_seen"],
            "drop": b["drop"], "n_slide": b["n_slide"]}


def _restore(b: dict, s: dict) -> None:
    b["state"], b["cur"] = s["state"], dict(s["cur"])
    b["slid_seen"], b["drop"], b["n_slide"] = s["slid_seen"], s["drop"], s["n_slide"]


def _cands(rec: dict, want: dict, need_start: bool) -> list:
    out = []
    for b in rec["blocks"]:
        if b["state"] != "todo":
            continue
        c = b["cur"]
        if any(want.get(k) is not None and c[k] != want[k]
               for k in ("act", "place", "aim", "priority", "flex")):
            continue
        if need_start and int(c["start"]) != int(want.get("start", -1)):
            continue
        out.append(b)
    return out


def _pick(rec: dict, want: dict, stats: dict, tag: str):
    """(block, exact_start) を返す。多義は**添字最小**を採り、件数を数えて出す。"""
    cands = _cands(rec, want, need_start=True)
    exact = True
    if not cands:
        cands = _cands(rec, want, need_start=False)
        exact = False
    if not cands:
        stats["unmatched"][tag] += 1
        return None, False
    if len(cands) > 1:
        stats["ambiguous"][tag] += 1
    stats["matched"][tag] += 1
    return cands[0], exact


def _apply_drop(b: dict, e: dict, p: dict) -> None:
    b["state"] = "dropped"
    b["drop"] = {"step": e["step"], "sim_min": e["sim_min"],
                 "reason": str(p.get("reason") or "")}


def _apply_slide(b: dict, p: dict) -> None:
    new_start = int(p.get("start") or 0)
    dur = int(b["cur"]["end"]) - int(b["cur"]["start"])
    b["cur"]["start"] = new_start
    b["cur"]["end"] = new_start + dur
    b["slid_seen"] = int(p.get("slid") or 0)
    b["n_slide"] += 1


def _by_index(rec: dict, p: dict, stats: dict, tag: str):
    """payload の `block` 添字で直接帰属する(W4-F 以降のラン)。

    返り値 = ブロック or None(添字が無い旧ラン = 台帳再生へ落ちる合図)。
    キーはあるのに範囲外(壊れた/`-1`)なら **数えてから** None を返す
    = 黙って当てにいかない(`plan_cont_fire` の cont_bad_index と同じ作法)。
    """
    v = p.get("block")
    if v is None:
        return None
    try:
        i = int(v)
    except (TypeError, ValueError):
        stats["block_index_bad"] += 1
        return None
    if not (0 <= i < len(rec["blocks"])):
        stats["block_index_bad"] += 1
        return None
    stats["matched"][tag] += 1
    stats["by_index"][tag] += 1
    return rec["blocks"][i]


def _pick_slide(rec: dict, p: dict, stats: dict):
    """`plan_slide` は **ずらした後の start** しか持たないので専用の照合を使う。

    候補 = todo かつ (act, place, priority) 一致かつ 現在の開始が新しい開始以下。
    2 回目以降のずらしは累積 `slid` の増分が一致するはず = それを優先根拠にする。
    """
    new_start, new_slid = int(p.get("start") or 0), int(p.get("slid") or 0)
    cands = [b for b in rec["blocks"]
             if b["state"] == "todo"
             and b["cur"]["act"] == str(p.get("act") or "")
             and b["cur"]["place"] == str(p.get("place") or "")
             and b["cur"]["priority"] == str(p.get("priority") or "")
             and int(b["cur"]["start"]) <= new_start]
    if not cands:
        stats["unmatched"]["slide"] += 1
        return None
    pref = [b for b in cands
            if b["slid_seen"] is not None
            and b["slid_seen"] + (new_start - int(b["cur"]["start"])) == new_slid]
    if len(cands) > 1 and len(pref) != 1:
        stats["ambiguous"]["slide"] += 1
    stats["matched"]["slide"] += 1
    return pref[0] if len(pref) == 1 else cands[0]


def build_ledger(events: list[dict], cfg: dict) -> tuple[list, dict]:
    """`plan_created` を起点に (agent, day) ごとのブロック台帳を再生する。"""
    stats = {"matched": defaultdict(int), "ambiguous": defaultdict(int),
             "unmatched": defaultdict(int), "by_index": defaultdict(int),
             "orphan_events": 0, "block_index_bad": 0,
             "cont_reattributed": 0, "cont_bad_index": 0, "replan_mismatch": 0,
             "n_repair": 0, "n_fallback": 0, "fallback_kinds": defaultdict(int)}
    days: dict = {}
    order: list = []
    for e in events:
        kind = e["kind"]
        if kind not in PLAN_KINDS:
            continue
        aid, p = e["agent_id"], e["payload"]
        key = (aid, e["sim_min"] // 1440)
        if kind == "plan_created":
            days[key] = _new_day(aid, key[1], e, p)
            order.append(key)
            continue
        if kind == "plan_repair":
            stats["n_repair"] += 1
            continue
        if kind == "plan_fallback":
            stats["n_fallback"] += 1
            stats["fallback_kinds"][str(p.get("kind") or "")] += 1
            continue
        rec = days.get(key)
        if rec is None:
            stats["orphan_events"] += 1          # 計画より先に来た/計画の無い実行
            continue
        if kind == "plan_block_start":
            want = {"act": str(p.get("act") or ""), "place": str(p.get("place") or ""),
                    "aim": str(p.get("aim") or ""),
                    "priority": str(p.get("priority") or ""),
                    "flex": str(p.get("flex") or ""), "start": int(p.get("start") or 0)}
            b = _by_index(rec, p, stats, "start")            # W4-F: 添字が正典
            exact, src = True, "index"
            if b is None:                                    # 旧ラン = 台帳再生へ
                b, exact = _pick(rec, want, stats, "start")
                src = "replay"
            if b is not None:
                b["state"] = "done"
                b["exec"] = {"step": e["step"], "sim_min": e["sim_min"],
                             "start": int(p.get("start") or 0),
                             "slid": int(p.get("slid") or 0),
                             "place": want["place"], "act": want["act"],
                             "aim": want["aim"],
                             "version": int(p.get("version") or 1),
                             "exact_match": bool(exact), "attrib": src,
                             # W4-F: 実行時の**現在ノード**(移動前)。移動イベントが
                             # 1 件も無い step でだけ到達ノードとして採る。
                             "node_at_start": str(p.get("node") or ""),
                             "node": "", "node_src": "", "node_planned": "",
                             "detour": None}
            rec["_pend"] = None
        elif kind == "plan_block_drop":
            want = {"act": str(p.get("act") or ""), "place": str(p.get("place") or ""),
                    "priority": str(p.get("priority") or ""),
                    "flex": str(p.get("flex") or ""), "start": int(p.get("start") or 0)}
            b = _by_index(rec, p, stats, "drop")
            if b is None:
                b, _exact = _pick(rec, want, stats, "drop")
            if b is not None:
                rec["_pend"] = {"step": e["step"], "kind": kind, "b": b,
                                "before": _snap(b), "payload": p, "event": e}
                _apply_drop(b, e, p)
        elif kind == "plan_slide":
            b = _by_index(rec, p, stats, "slide")
            if b is None:
                b = _pick_slide(rec, p, stats)
            if b is not None:
                rec["_pend"] = {"step": e["step"], "kind": kind, "b": b,
                                "before": _snap(b), "payload": p, "event": e}
                _apply_slide(b, p)
        elif kind == "plan_cont_fire":
            _handle_cont(rec, e, p, cfg, stats)
        elif kind == "plan_replan":
            _handle_replan(rec, e, p, cfg, stats)
    out = [days[k] for k in order]
    for rec in out:
        rec.pop("_pend", None)
    for k in ("matched", "ambiguous", "unmatched", "by_index", "fallback_kinds"):
        stats[k] = {kk: int(vv) for kk, vv in sorted(stats[k].items())}
    return out, stats


def _handle_cont(rec: dict, e: dict, p: dict, cfg: dict, stats: dict) -> None:
    """`plan_cont_fire` は**ブロック添字を payload に持つ唯一のイベント** = 正典。

    直前の drop/slide(= then=skip / postpone が同 step に先に出す)の帰属が
    食い違っていたら差し戻して張り直す。
    """
    idx = int(p.get("block", -1) if p.get("block") is not None else -1)
    if not (0 <= idx < len(rec["blocks"])):
        stats["cont_bad_index"] += 1
        rec["_pend"] = None
        return
    b = rec["blocks"][idx]
    then = str(p.get("then") or "")
    expect = {"skip": "plan_block_drop", "postpone": "plan_slide"}.get(then)
    pend = rec["_pend"]
    if (expect and pend and pend["step"] == e["step"] and pend["kind"] == expect
            and pend["b"] is not b):
        _restore(pend["b"], pend["before"])       # 誤帰属を差し戻す
        if expect == "plan_block_drop":
            _apply_drop(b, pend["event"], pend["payload"])
        else:
            _apply_slide(b, pend["payload"])
        stats["cont_reattributed"] += 1
    rec["_pend"] = None
    b["cont"] = {"cond": str(p.get("cond") or ""), "then": then,
                 "applied": bool(p.get("applied")), "step": e["step"]}
    if b["state"] != "dropped":
        # payload の act/place/start は **対処を適用した後**の値(apply_contingency の _log)
        new_act = str(p.get("act") or b["cur"]["act"])
        if new_act != b["cur"]["act"]:
            # go_home / swap_indoor は act を変えたときだけ purpose も既定表から引き直す。
            # ★その既定表 DP.ACT_PURPOSE を**そのまま使う**(写しを持たない)。
            b["cur"]["aim"] = DP.ACT_PURPOSE.get(new_act, b["cur"]["aim"])
        b["cur"]["act"] = new_act
        b["cur"]["place"] = str(p.get("place") or b["cur"]["place"])
        if then == "shorten":
            lo = int(cfg["min_dur_min"])
            b["cur"]["end"] = max(int(b["cur"]["start"]) + lo,
                                  int(b["cur"]["end"]) - int(cfg["shorten_min"]))
        b["cur"]["start"] = int(p.get("start") or b["cur"]["start"])


def _handle_replan(rec: dict, e: dict, p: dict, cfg: dict, stats: dict) -> None:
    """`plan_replan` は **ブロック単位のイベントを出さずに** must をずらす。

    ずらし先は `_round_to(now, round_min)` の決定論なので、`at`(= now)と
    正準化済み cfg から再現できる。件数は payload の `n_must` / `retired` で検算する。
    """
    rec["replans"] += 1
    rec["_pend"] = None
    at = int(p.get("at", e["sim_min"] % 1440) or 0)
    grace, day_end = int(cfg["grace_min"]), int(cfg["day_end_min"])
    musts = [b for b in rec["blocks"]
             if b["state"] == "todo" and b["cur"]["priority"] == "must"
             and at > int(b["cur"]["start"]) + grace]
    expect = max(0, int(p.get("n_must") or 0) - int(p.get("retired") or 0))
    if len(musts) != expect:
        stats["replan_mismatch"] += 1
    for m in musts:
        if m["cur"]["flex"] == "fixed":
            continue                              # 動かせない = 時刻は変わらない
        dur = int(m["cur"]["end"]) - int(m["cur"]["start"])
        ns = DP._round_to(at, cfg["round_min"])
        m["cur"]["start"] = ns
        m["cur"]["end"] = min(day_end, ns + dur)
        m["n_slide"] += 1


# --------------------------------------------------------------------------- #
# ③ 到達ノードの帰属((a) PROV join → (b) 同 step join の順に落ちる)
# --------------------------------------------------------------------------- #
def attach_nodes(days: list, events: list[dict]) -> dict:
    moves: dict = defaultdict(list)
    for e in events:
        if e["kind"] in MOVE_KINDS:
            moves[(e["agent_id"], e["step"])].append(e)
    counts = {"prov": 0, "step_join": 0, "block_node": 0, "none": 0, "detour": 0}
    n_role_plan = sum(1 for e in events if e["kind"] == "route_start"
                      and str(e["payload"].get("llm_role") or "") == "plan")
    for rec in days:
        cid = rec["call_id"]
        for b in rec["blocks"]:
            ex = b["exec"]
            if ex is None:
                continue
            evs = moves.get((rec["agent"], ex["step"]), [])
            routes = [x for x in evs if x["kind"] == "route_start"]
            detours = [x for x in evs if x["kind"] == "detour"]
            prov = [x for x in routes
                    if cid and x["llm_call_id"] == cid
                    and str(x["payload"].get("llm_role") or "") == "plan"]
            pick, src = (prov[0], "prov") if prov else \
                        ((routes[0], "step_join") if routes else (None, "none"))
            if pick is None:
                # (c) 経路: 移動イベントが 1 件も無い = 個体は動かなかった。W4-F 以降の
                #     ランはその「動かなかった場所」を plan_block_start.node に持つ。
                stayed = str(ex.get("node_at_start") or "")
                if stayed:
                    counts["block_node"] += 1
                    ex["node"] = stayed
                    ex["node_src"] = "block_node"
                    ex["node_planned"] = stayed
                else:
                    counts["none"] += 1
                continue
            counts[src] += 1
            ex["node"] = str(pick["payload"].get("dest") or "")
            ex["node_src"] = src
            ex["node_planned"] = ex["node"]
            if detours:
                dp = detours[0]["payload"]
                ex["detour"] = {"to": str(dp.get("to") or ""),
                                "from_dest": str(dp.get("from_dest") or "")}
                ex["node_planned"] = str(dp.get("from_dest") or ex["node"])
                counts["detour"] += 1
    return {"by_source": counts, "n_route_start_role_plan": n_role_plan,
            "path": ("prov" if counts["prov"] else
                     ("step_join" if counts["step_join"] else
                      ("block_node" if counts["block_node"] else "none")))}


# --------------------------------------------------------------------------- #
# ④ 場所の逆引き判定(resolve_place の**必要条件**を検査する)
# --------------------------------------------------------------------------- #
def place_status(place: str, node: str, aid: int,
                 mapidx: dict | None, agents: dict) -> tuple[str, str]:
    """(status, detail) を返す。status ∈ match / mismatch / delegated / unknown。"""
    if place in DELEGATED_PLACES:
        return "delegated", "street_delegated"
    if not node:
        return "unknown", "no_move_event"
    if place == "home":
        home = (agents.get(aid) or {}).get("home") or ""
        if not home:
            return "unknown", "home_unknown"
        return ("match" if node == home else "mismatch"), ""
    if place == "work":
        rec = agents.get(aid) or {}
        wn = rec.get("work_nodes") or set()
        if not wn:
            return "unknown", "work_unknown"
        # detail = その判定が実ノード由来か建物近似か(W4-F 以降は exact_node)
        return (("match" if node in wn else "mismatch"),
                str(rec.get("work_src") or "approx_building"))
    if mapidx is None:
        return "unknown", "no_map"
    accept = _accept_cats(place)
    if all(int(mapidx["cat_counts"].get(c, 0)) == 0 for c in accept):
        # その種類の場所(fallback 込み)が地図に 1 件も無い = 計画は原理的に満たされえない
        return "unknown", "category_absent_on_map"
    if node not in mapidx["nodes"]:
        return "unknown", "node_off_map"
    ncats = mapidx["node_cats"].get(node, ())
    return ("match" if any(c in ncats for c in accept) else "mismatch"), ""


# --------------------------------------------------------------------------- #
# ⑤ 3 段の遵守率
# --------------------------------------------------------------------------- #
def _rate(ok: int, n: int):
    return round(ok / n, 4) if n else None


def _blank_place_row() -> dict:
    return {"planned": 0, "executed": 0, "dropped": 0, "untouched": 0,
            "time_ok": 0, "act_ok": 0, "place_decided": 0, "place_ok": 0,
            "unresolved": 0}


def score(days: list, mapidx: dict | None, agents: dict) -> dict:
    n_planned = n_exec = n_drop = n_untouched = 0
    time_ok = act_ok = intent_ok = 0
    place_dec = place_ok = 0
    all3_dec = all3_ok = 0
    unresolved = {r: 0 for r in UNRESOLVED_REASONS}
    by_place: dict = defaultdict(_blank_place_row)
    by_drop_reason: dict = defaultdict(int)
    samples: list = []
    n_detour = n_cont = 0
    for rec in days:
        aid = rec["agent"]
        for b in rec["blocks"]:
            n_planned += 1
            planned_place = b["plan"]["place"]
            row = by_place[planned_place]
            row["planned"] += 1
            if b["cont"]:
                n_cont += 1
            if b["state"] == "dropped":
                n_drop += 1
                row["dropped"] += 1
                by_drop_reason[(b["drop"] or {}).get("reason", "")] += 1
                continue
            ex = b["exec"]
            if ex is None:
                n_untouched += 1
                row["untouched"] += 1
                continue
            n_exec += 1
            row["executed"] += 1
            t_ok = int(ex["start"]) == int(b["plan"]["start"])
            a_ok = ex["act"] == b["plan"]["act"]
            i_ok = ex["place"] == planned_place
            time_ok += int(t_ok)
            act_ok += int(a_ok)
            intent_ok += int(i_ok)
            row["time_ok"] += int(t_ok)
            row["act_ok"] += int(a_ok)
            if ex["detour"]:
                n_detour += 1
            status, detail = place_status(planned_place, ex["node"], aid,
                                          mapidx, agents)
            ex["place_status"] = status
            ex["place_detail"] = detail
            if status in ("match", "mismatch"):
                place_dec += 1
                row["place_decided"] += 1
                p_ok = status == "match"
                place_ok += int(p_ok)
                row["place_ok"] += int(p_ok)
                all3_dec += 1
                all3_ok += int(p_ok and a_ok and t_ok)
            else:
                row["unresolved"] += 1
                if detail in unresolved:
                    unresolved[detail] += 1
            if status == "mismatch" and len(samples) < 50:
                samples.append({
                    "agent": aid, "day": rec["day"], "block": b["i"],
                    "planned_place": planned_place, "planned_act": b["plan"]["act"],
                    "planned_start": b["plan"]["start"],
                    "exec_place": ex["place"], "exec_act": ex["act"],
                    "exec_start": ex["start"], "node": ex["node"],
                    "node_src": ex["node_src"],
                    "node_cats": sorted((mapidx or {}).get("node_cats", {})
                                        .get(ex["node"], ())) if mapidx else [],
                    "detour": bool(ex["detour"])})
    n_unres = sum(unresolved.values())
    return {
        "denominator": {"plans": len(days), "blocks_planned": n_planned,
                        "blocks_executed": n_exec, "blocks_dropped": n_drop,
                        "blocks_untouched": n_untouched,
                        "blocks_with_contingency": n_cont, "detours": n_detour},
        "stage": {
            # 3 段のはしご(ゆるい → きつい)
            "time":  {"n": n_exec, "ok": time_ok, "rate": _rate(time_ok, n_exec),
                      "what": "朝の窓のまま実行された(日中にずらされていない)"},
            "act":   {"n": n_exec, "ok": act_ok, "rate": _rate(act_ok, n_exec),
                      "what": "実行時の活動種別が朝の宣言と同じ"},
            "place": {"n": place_dec, "ok": place_ok, "rate": _rate(place_ok, place_dec),
                      "what": "到達ノードが朝に宣言した場所カテゴリの POI である",
                      "unresolved": n_unres,
                      "unresolved_rate": _rate(n_unres, n_exec),
                      "by_reason": {k: v for k, v in sorted(unresolved.items()) if v}},
            "place_intent": {"n": n_exec, "ok": intent_ok,
                             "rate": _rate(intent_ok, n_exec),
                             "what": "実行時の場所カテゴリが朝の宣言と同じ"
                                     "(地図を見ない計画内部の一致)"},
            "all_three": {"n": all3_dec, "ok": all3_ok, "rate": _rate(all3_ok, all3_dec),
                          "what": "時間帯・活動種別・場所カテゴリの 3 段すべて一致"},
        },
        "execution_rate": {"executed_of_planned": _rate(n_exec, n_planned),
                           "dropped_of_planned": _rate(n_drop, n_planned),
                           "untouched_of_planned": _rate(n_untouched, n_planned)},
        "by_place": {k: dict(by_place[k]) for k in sorted(by_place)},
        "by_drop_reason": {k: int(v) for k, v in sorted(by_drop_reason.items())},
        "mismatch_samples": samples,
    }


# --------------------------------------------------------------------------- #
# まとめ
# --------------------------------------------------------------------------- #
BLOCK_KINDS: tuple[str, ...] = ("plan_block_start", "plan_block_drop", "plan_slide")


def field_support(events: list[dict], agents: dict) -> dict:
    """W4-F の 3 記録がこのランに**在るか**(無ければ従来経路へ落ちている印)。

    率ではなく生の件数で出す。「新フィールドがあるのに使われていない」「一部の
    イベントだけ持っている」といった中途半端な状態を隠さないため。
    """
    n_bev = sum(1 for e in events if e["kind"] in BLOCK_KINDS)
    n_bidx = sum(1 for e in events if e["kind"] in BLOCK_KINDS
                 and e["payload"].get("block") is not None)
    n_start = sum(1 for e in events if e["kind"] == "plan_block_start")
    n_node = sum(1 for e in events if e["kind"] == "plan_block_start"
                 and e["payload"].get("node"))
    n_wexact = sum(1 for r in agents.values()
                   if r.get("work_src") == "exact_node")
    return {
        "block_index": {"n": n_bidx, "of": n_bev},
        "block_node": {"n": n_node, "of": n_start},
        "work_node": {"n": n_wexact, "of": len(agents)},
        "note": "W4-F(2026-08-08)以降のランだけが持つ記録。"
                "0 なら旧ラン = 台帳再生 + 建物近似へ自動で落ちている(測定値は出る)",
    }


def summarize(run_dir: str, events: list[dict], cfg: dict,
              mapidx: dict | None, agents: dict) -> dict:
    cov = place_coverage(mapidx)
    n_created = sum(1 for e in events if e["kind"] == "plan_created")
    base = {
        "schema": SCHEMA,
        "run_dir": str(run_dir),
        "day_plan_cfg": {k: cfg[k] for k in sorted(
            ("enabled", "use_contingency", "round_min", "grace_min",
             "day_end_min", "min_dur_min", "shorten_min", "postpone_min"))},
        "vocabulary": cov,
    }
    if n_created == 0:
        # ★測る対象が存在しない。**0.0 ではなく null** で出す(ゼロと偽らない)。
        return {**base,
                "day_plan": {"present": False, "n_plan_created": 0,
                             "reason": "plan_created が 0 件 = planning.day_plan が"
                                       " OFF のラン(遵守率は測っていない = null)"},
                "attribution": None, "ledger": None, "compliance": None}
    days, lstats = build_ledger(events, cfg)
    attrib = attach_nodes(days, events)
    comp = score(days, mapidx, agents)
    n_amb = sum(lstats["ambiguous"].values())
    n_unm = sum(lstats["unmatched"].values())
    n_mat = sum(lstats["matched"].values())
    return {
        **base,
        "day_plan": {"present": True, "n_plan_created": n_created,
                     "n_days": len(days),
                     "agents_json": bool(agents),
                     "map": (mapidx or {}).get("path"),
                     "fields": field_support(events, agents)},
        "attribution": attrib,
        "ledger": {
            "matched": lstats["matched"], "ambiguous": lstats["ambiguous"],
            "unmatched": lstats["unmatched"], "by_index": lstats["by_index"],
            "ambiguous_rate": _rate(n_amb, n_mat),
            "unmatched_rate": _rate(n_unm, n_mat + n_unm),
            "orphan_events": lstats["orphan_events"],
            "block_index_bad": lstats["block_index_bad"],
            "cont_reattributed": lstats["cont_reattributed"],
            "cont_bad_index": lstats["cont_bad_index"],
            "replan_mismatch": lstats["replan_mismatch"],
            "n_repair": lstats["n_repair"], "n_fallback": lstats["n_fallback"],
            "fallback_kinds": lstats["fallback_kinds"]},
        "compliance": comp,
    }


def render(res: dict) -> str:
    cov = res["vocabulary"]
    L = ["# 朝の計画と実行動の突合(W2-4)", "",
         "## 0. 場所語彙の被覆(地図 × day_plan.PLACE_CATS)", ""]
    if cov.get("map") is None:
        L.append(f"- {cov.get('note', '地図なし')}")
    else:
        L += [f"- 地図: `{cov['map']}`(POI {cov['n_pois']} 件)",
              f"- 計画語彙 15 種 = 地図カテゴリ {len(cov['map_backed'])} + "
              f"個体相対 {len(cov['agent_relative'])}(home/work)+ 委譲 1(street)",
              f"- **この地図で解決できる場所カテゴリ: "
              f"{len(cov['resolvable'])}/{len(cov['map_backed'])}"
              f"(被覆率 {cov['coverage_rate']})**",
              f"- 解決できない: {cov['unresolvable'] or 'なし'}",
              f"- 地図にあるが計画語彙に無い: {cov['map_cats_not_in_plan_vocab'] or 'なし'}",
              ""]
        if cov["alias_shadowed"]:
            L += ["> ★**別名表が地図の実カテゴリを踏み潰している**"
                  "(LLM が正しい語を書いても届かない):", ""]
            for s in cov["alias_shadowed"]:
                L.append(f"> - `{s['alias']}`(地図に {s['n_pois_alias']} 件)→ "
                         f"`{s['rewritten_to']}` へ書き換え(地図に "
                         f"{s['n_pois_target']} 件)")
            L.append("")
        L += ["| 場所 | 地図の POI 件数 |", "|---|---|"]
        for k, v in cov["n_pois_by_place"].items():
            L.append(f"| {k} | {v} |")
        L.append("")
    dp = res["day_plan"]
    if not dp["present"]:
        L += ["## 1. 計画遵守率", "",
              f"- **{dp['reason']}**",
              "- 遵守率・未解決率はいずれも `null`(0 件ではなく**未測定**)。", ""]
        return "\n".join(L) + "\n"
    at, lg, c = res["attribution"], res["ledger"], res["compliance"]
    d, s = c["denominator"], c["stage"]
    fl = dp.get("fields") or {}
    n_idx = sum(lg.get("by_index", {}).values())
    L += ["## 1. 帰属経路", "",
          f"- 採用経路: **{at['path']}**"
          f"(PROV {at['by_source']['prov']} 件 / 同 step {at['by_source']['step_join']} 件"
          f" / 留まった {at['by_source']['block_node']} 件"
          f" / 帰属不能 {at['by_source']['none']} 件)",
          f"- `llm_role=\"plan\"` の route_start: {at['n_route_start_role_plan']} 件"
          + ("(= observer.llm_link ON)" if at["n_route_start_role_plan"]
             else "(= observer.llm_link OFF → 同 step join へ自動フォールバック)"),
          f"- ブロック同定: 一致 {sum(lg['matched'].values())} /"
          f" 多義 {sum(lg['ambiguous'].values())}(率 {lg['ambiguous_rate']})/"
          f" 不照合 {sum(lg['unmatched'].values())}(率 {lg['unmatched_rate']})",
          f"- うち **payload の block 添字で直接帰属 {n_idx} 件**"
          f"(記録の有無: block 添字 {fl.get('block_index', {}).get('n', 0)}/"
          f"{fl.get('block_index', {}).get('of', 0)} ・ "
          f"現在ノード {fl.get('block_node', {}).get('n', 0)}/"
          f"{fl.get('block_node', {}).get('of', 0)} ・ "
          f"実職場ノードを持つ個体 {fl.get('work_node', {}).get('n', 0)}/"
          f"{fl.get('work_node', {}).get('of', 0)})", "",
          "## 2. 計画遵守率(3 段のはしご)", "",
          f"- 計画 {d['plans']} 件 / ブロック {d['blocks_planned']} 個 "
          f"→ 実行 {d['blocks_executed']} / 自動削除 {d['blocks_dropped']} /"
          f" 未着手 {d['blocks_untouched']}", "",
          "| 段 | 分母 | 一致 | 率 | 意味 |", "|---|---|---|---|---|"]
    for key, label in (("time", "① 時間帯"), ("act", "② 活動種別"),
                       ("place_intent", "②' 場所カテゴリ(計画内部)"),
                       ("place", "③ 場所解決(地図で検証)"),
                       ("all_three", "①∧②∧③")):
        r = s[key]
        L.append(f"| {label} | {r['n']} | {r['ok']} | {r['rate']} | {r['what']} |")
    L += ["",
          f"- **未解決(unknown/委譲)= {s['place']['unresolved']} 件"
          f"(実行ブロックの {s['place']['unresolved_rate']})**"
          f" 内訳: {json.dumps(s['place']['by_reason'], ensure_ascii=False) or '{}'}",
          "", "## 3. place 別の内訳", "",
          "| place | 計画 | 実行 | 削除 | 未着手 | 時間帯一致 | 活動一致 |"
          " 場所判定可 | 場所一致 | 未解決 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in c["by_place"].items():
        L.append(f"| {k} | {r['planned']} | {r['executed']} | {r['dropped']} |"
                 f" {r['untouched']} | {r['time_ok']} | {r['act_ok']} |"
                 f" {r['place_decided']} | {r['place_ok']} | {r['unresolved']} |")
    w_exact = int(fl.get("work_node", {}).get("n", 0)) > 0
    L += ["", "## 4. 自動削除の理由", "",
          json.dumps(c["by_drop_reason"], ensure_ascii=False) or "{}", "",
          "> 場所の判定は `resolve_place` の**必要条件**(到達ノードが",
          "> `pois_by_cat(place)` に属するか)であって、「同じ 1 ノードを選んだか」では",
          "> ない。`work` は " + ("agents.json の **work_node / part_time_node による実"
                                  "ノード判定**(W4-F。転職前のスナップショットである点は"
                                  "残る)。"
                                  if w_exact else
                                  "`work_building` からの**近似**(この旧ランの "
                                  "agents.json に work_node が無いため)。"),
          "> 帰属できなかったものは当てにいかず `unknown` に置いてある。"]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="朝の計画(day_plan.place)と実行動の突合(読み取り専用)")
    ap.add_argument("run_dir", help="ラン出力ディレクトリ(l1_events.parquet を含む)")
    ap.add_argument("--map", default=None,
                    help="地図 JSON(既定: <run_dir>/config.yaml の world.map)")
    ap.add_argument("--out", default=None, help="出力先(既定: <run_dir>/analysis)")
    args = ap.parse_args()

    events = load_events(args.run_dir)
    cfg = load_cfg(args.run_dir)
    mapidx = load_map(map_path_of(args.run_dir, args.map))
    agents = load_agents(args.run_dir, mapidx)
    res = summarize(args.run_dir, events, cfg, mapidx, agents)

    out_dir = args.out or os.path.join(args.run_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    jpath = os.path.join(out_dir, "plan_execution.json")
    mpath = os.path.join(out_dir, "plan_execution_report.md")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2, sort_keys=True, default=str)
    md = render(res)
    with open(mpath, "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    print(f"[written] {jpath}")
    print(f"[written] {mpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
