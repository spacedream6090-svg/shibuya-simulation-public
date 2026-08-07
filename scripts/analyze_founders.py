#!/usr/bin/env python
"""組織の自然形成 + ファウンダー成立条件の観測装置(第37バッチ Track A・読み出し専用)。

    python scripts/analyze_founders.py --run demo_event_200a3d
    python scripts/analyze_founders.py --run wv_llm_7d --out panel/ --pre-days 2

研究目標「LLMエージェント社会で **組織は自然と形成されるのか**。**ファウンダーはどのような
影響・条件でファウンダーになるのか**」の観測装置。設計哲学 = 自然界的ボトムアップ
(ファウンダーを定義で強制せず、**行動から検出する**)。断定はしない=すべて「候補」。

`runs/<name>` の l1_events.parquet(payload は JSON 文字列)・agents.json・traits.json・
config.yaml(world.map の建物名)だけを読み、sim 本体 / schema は一切変更しない(完全な後処理)。
依存は標準ライブラリ + pyarrow のみ(pandas/duckdb 不使用。numpy は任意)。

観測する4つのこと:
  1. ファウンダー行動の検出(行動ベース・複数タイプ):
       起業型 = venture_permit / venture_open / venture_fulltime / deviance(無許可出店)
       政治型 = candidacy(立候補・第37バッチ新設)/ proposal(住民提案)の高頻度者
       ハブ型 = relation_tier 由来の関係ネットワークの 次数 / 媒介中心性(純Python BFS=Brandes)
                上位者 + speak/dm 発信 > hear 受信 の「発信超過」者
  2. 形成前履歴パネル: 各ファウンダー候補の「初回ファウンダー行動」の前 N 日(既定2日)の
       日次時系列(資金 wage/spend/tax・関係数・opinion 累積移動・grievance/efficacy・
       発話/被聴取/DM・k=traits の fire_weight 等)→ panel/founders.parquet
  3. 対照群: ファウンダー行動なしの全員から 同職業×同年齢帯 で決定論マッチした対照を
       同じ列で出力(founder=0/1 列)→ 後の回帰・条件比較の土台
  4. 組織形成の検出(自然発生候補): 同一建物 × 反復共在(同じ建物に同じメンバーが
       3日以上連続共在)× 既存 organizations 台帳(=各人の勤務先/自宅建物)に無い集団を
       クラスタとして列挙 → panel/org_candidates.json(「組織」とは断定せず候補として)

イベント欠損への縮退: candidacy / venture_* が1件も無い旧ランでも例外なく動く(検出0件で続行)。
1step = 10分・1日 = 144step。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict, deque

# Windows コンソール(cp932)対策。ファイル出力は常に UTF-8。en-dash 等での print 死を防ぐ。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
RUNS_ROOT = os.path.join(_ROOT, "runs")
if _HERE not in sys.path:                 # 同ディレクトリの run_dt を import
    sys.path.insert(0, _HERE)

import run_dt                             # noqa: E402  (W2-3: ランの Δt の単一の源)

# W2-3: **ラン依存**(run.dt_min)。ここの値は正準 Δt=10 の既定で、実際の値は analyze() が
# run dir から読んで各段へ引数で配る(Δt=10 なら 144 = 従来と 1 ビットも変わらない)。
STEPS_PER_DAY = run_dt.CANON_STEPS_PER_DAY

# ---- 検出の既定パラメータ(すべて CLI で上書き可) ----
DEF_PRE_DAYS = 2          # 形成前履歴の窓(前 N 日)
DEF_PROPOSAL_MIN = 2      # 政治型: proposal をこの件数以上出したら「高頻度提案者」
DEF_HUB_DEGREE = 5        # ハブ型: 次数の下限フロア(これ未満はハブ候補にしない)
DEF_TOP_K = 10            # ハブ型: 次数 / 媒介中心性 / 発信超過の上位 K 名をハブとする
DEF_COPRESENCE_DAYS = 3   # 組織候補: 連続共在の最小日数
DEF_MIN_CORE = 2          # 組織候補: コア人数の下限(1人は集団でない)
DEF_MAX_CORE = 12         # 組織候補: これを超えるコアは「群集」扱い(公共空間の通過)


# --------------------------------------------------------------------------- #
# ローダ(detect_emergence.py と同じ流儀。numpy を引き込まない最小実装)
# --------------------------------------------------------------------------- #
# W4-D: 本解析が実際に読む kind(5 つの検出器の if/elif を全数列挙したもの)。
#   起業型 = _VENTURE_KINDS / 政治型 = candidacy・proposal / ハブ型 = relation_tier
#   + 発受信 3 種 / 日次指標 = 金銭 4 種・opinion_shift・state_update
#   / 組織候補 = enter_building。
# ここに無い kind は全ての検出器が必ず読み飛ばすので、絞っても出力は変わらない。
# 本 module に **全 kind 依存の量は無い**(n_agents は agents.json 由来・
# 日軸は各検出器が自分の kind の step から作る)ことを確認済み。
WANT_KINDS = frozenset({
    "venture_open", "venture_permit", "venture_fulltime",   # 起業型
    "candidacy", "proposal",                                # 政治型
    "relation_tier",                                        # 関係グラフ / 日次
    "speak", "dm", "hear",                                  # 発受信の非対称
    "wage", "spend", "withdraw", "tax",                     # 家計の日次
    "opinion_shift", "state_update",                        # 内面の日次
    "enter_building",                                       # 反復共在(組織候補)
})


def load_events(run_dir: str, dt_min: int = run_dt.CANON_DT_MIN,
                kinds=WANT_KINDS) -> list[dict]:
    """l1_events.parquet を列射影 + kind 絞り込み + 逐次読みし、payload を dict 展開して返す。

    返り値の各要素: {step, sim_min, agent, kind, payload(dict)}。

    W4-D: 読みは `l1_stream.iter_columns` へ(Arrow レベルで kind を落としてから
    Python 化する)。`kinds=None` を渡すと全 kind(旧挙動)。返り値・行順・キー順は不変。

    残る比例項: 返り値は list なので **WANT_KINDS の行数**にはメモリが比例する
    (この解析は「形成前後の窓」を全期間から引くので、対象行は畳めない)。"""
    import l1_stream as ls
    if not ls.l1_paths(run_dir):
        raise FileNotFoundError(os.path.join(run_dir, "l1_events.parquet"))
    want = ["step", "sim_min", "agent_id", "kind", "payload"]
    out: list[dict] = []
    for d in ls.iter_columns(run_dir, want, kinds=kinds):
        n = len(d["step"])
        step = d["step"]
        smin = d.get("sim_min")
        agent = d["agent_id"]
        kind = d["kind"]
        pays = d["payload"]
        for i in range(n):
            raw = pays[i]
            try:
                payload = json.loads(raw) if raw else {}
            except (json.JSONDecodeError, TypeError):
                payload = {}
            out.append({
                "step": int(step[i]),
                # W2-3: sim_min 欠損時の復元は step × Δt(既定 10 = 従来と完全同値)。
                "sim_min": (int(smin[i]) if smin is not None and smin[i] is not None
                            else int(step[i]) * int(dt_min)),
                "agent": int(agent[i]),
                "kind": kind[i],
                "payload": payload,
            })
    return out


def load_agents(run_dir: str) -> dict[int, dict]:
    """agents.json を {agent_id: {name, age, occupation, home_building, work_building, ...}} で返す。"""
    path = os.path.join(run_dir, "agents.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    out: dict[int, dict] = {}
    for a in data:
        try:
            out[int(a["id"])] = a
        except (KeyError, TypeError, ValueError):
            continue
    return out


def load_traits(run_dir: str) -> dict[int, dict]:
    """traits.json を {agent_id: {fire_weight, drive_threshold, internal_locus, nfc, risk_tolerance,...}}。"""
    path = os.path.join(run_dir, "traits.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    out: dict[int, dict] = {}
    if isinstance(data, dict):
        for k, v in data.items():
            try:
                out[int(k)] = v
            except (TypeError, ValueError):
                continue
    elif isinstance(data, list):
        for i, v in enumerate(data):
            aid = v.get("id", i) if isinstance(v, dict) else i
            out[int(aid)] = v
    return out


def load_building_names(run_dir: str) -> dict[str, str]:
    """config.yaml の world.map(建物一覧)から {building_id: name} を返す。無ければ空。"""
    cfg_path = os.path.join(run_dir, "config.yaml")
    map_rel = None
    if os.path.exists(cfg_path):
        try:
            from omegaconf import OmegaConf
            cfg = OmegaConf.load(cfg_path)
            map_rel = cfg.get("world", {}).get("map")
        except Exception:
            pat = re.compile(r"^\s*map:\s*([^\s#]+)")
            with open(cfg_path, encoding="utf-8") as fh:
                for line in fh:
                    mm = pat.match(line)
                    if mm:
                        map_rel = mm.group(1).strip().strip("\"'")
                        break
    if not map_rel:
        return {}
    map_path = map_rel if os.path.isabs(map_rel) else os.path.join(_ROOT, map_rel)
    if not os.path.exists(map_path):
        return {}
    try:
        with open(map_path, encoding="utf-8") as fh:
            md = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}
    names: dict[str, str] = {}
    for b in md.get("buildings", []) or []:
        if not isinstance(b, dict):
            continue
        bid = b.get("id")
        nm = (b.get("name") or "").strip()
        if bid and nm:
            names[bid] = nm
    return names


# --------------------------------------------------------------------------- #
# 1. ファウンダー行動の検出(行動ベース・複数タイプ)
# --------------------------------------------------------------------------- #
# 起業型を構成するイベント種(venture_permit の outcome=="denied" は「起業意思」だが未開業なので
# それも申請行為=前駆行動として拾う。deviance は venture_open の payload.permitted==False)。
_VENTURE_KINDS = ("venture_open", "venture_permit", "venture_fulltime")


def detect_venture_founders(events: list[dict]) -> dict[int, dict]:
    """起業型ファウンダー候補を検出。{agent: {onset_step, subtypes:set, n_actions, first_kind}}。

    subtypes: open / permit / fulltime / deviance(無許可出店)。onset_step=初回起業行動 step。"""
    out: dict[int, dict] = {}
    for e in events:
        k = e["kind"]
        if k not in _VENTURE_KINDS:
            continue
        aid = e["agent"]
        if aid is None or aid < 0:
            continue
        p = e["payload"]
        if k == "venture_open":
            sub = "deviance" if p.get("permitted") is False else "open"
        elif k == "venture_permit":
            sub = "permit"
        else:
            sub = "fulltime"
        rec = out.get(aid)
        if rec is None:
            rec = out[aid] = {"onset_step": e["step"], "subtypes": set(),
                              "n_actions": 0, "first_kind": k}
        rec["subtypes"].add(sub)
        rec["n_actions"] += 1
        if e["step"] < rec["onset_step"]:
            rec["onset_step"] = e["step"]
            rec["first_kind"] = k
    return out


def detect_political_founders(events: list[dict],
                              proposal_min: int = DEF_PROPOSAL_MIN) -> dict[int, dict]:
    """政治型ファウンダー候補を検出。candidacy(立候補)または proposal を proposal_min 件以上。

    {agent: {onset_step, subtypes:set, n_candidacy, n_proposal}}。onset は初回の立候補、無ければ
    proposal_min 件目の proposal の step(=高頻度提案者に「なった」瞬間)。"""
    cand_steps: dict[int, list[int]] = defaultdict(list)
    prop_steps: dict[int, list[int]] = defaultdict(list)
    for e in events:
        aid = e["agent"]
        if aid is None or aid < 0:
            continue
        if e["kind"] == "candidacy":
            cand_steps[aid].append(e["step"])
        elif e["kind"] == "proposal":
            prop_steps[aid].append(e["step"])

    out: dict[int, dict] = {}
    for aid in set(cand_steps) | set(prop_steps):
        cs = sorted(cand_steps.get(aid, []))
        ps = sorted(prop_steps.get(aid, []))
        subtypes: set[str] = set()
        onset = None
        if cs:
            subtypes.add("candidacy")
            onset = cs[0]
        if len(ps) >= proposal_min:
            subtypes.add("proposer")
            kth = ps[proposal_min - 1]           # 高頻度提案者になった step
            onset = kth if onset is None else min(onset, kth)
        if not subtypes:
            continue
        out[aid] = {"onset_step": onset, "subtypes": subtypes,
                    "n_candidacy": len(cs), "n_proposal": len(ps)}
    return out


# --------------------------------------------------------------------------- #
# ハブ型: 関係ネットワーク(relation_tier)+ 発信/受信の非対称
# --------------------------------------------------------------------------- #
def build_relation_graph(events: list[dict]) -> tuple[dict[int, set], dict[int, dict]]:
    """relation_tier から無向グラフを組む。返り値 (adj, reach)。

    adj: {agent: set(neighbors)}(双方向に追加)。
    reach: {agent: {"degree_step": {degree: step}, "first_step": step}}
           =各エージェントが「相異なる相手 d 人目」に達した step(次数到達の時系列)。"""
    adj: dict[int, set] = defaultdict(set)
    # 各エージェント視点で、相手ごとの初出 step(次数の成長を追う)。
    seen: dict[int, dict[int, int]] = defaultdict(dict)
    order: dict[int, list[tuple[int, int]]] = defaultdict(list)  # agent -> [(step, other)]
    for e in events:
        if e["kind"] != "relation_tier":
            continue
        aid = e["agent"]
        other = e["payload"].get("other")
        if aid is None or other is None or aid < 0:
            continue
        try:
            other = int(other)
        except (TypeError, ValueError):
            continue
        adj[aid].add(other)
        adj[other].add(aid)
        if other not in seen[aid]:
            seen[aid][other] = e["step"]
            order[aid].append((e["step"], other))

    reach: dict[int, dict] = {}
    for aid, lst in order.items():
        lst.sort()
        degree_step: dict[int, int] = {}
        for i, (st, _) in enumerate(lst, start=1):
            degree_step[i] = st
        reach[aid] = {"degree_step": degree_step,
                      "first_step": lst[0][0] if lst else None}
    # adj を通常 dict 化(孤立ノードも含め全ノードがキーになるよう保証)
    plain: dict[int, set] = {}
    for n, nb in adj.items():
        plain.setdefault(n, set()).update(nb)
        for m in nb:
            plain.setdefault(m, set())
    return plain, reach


def betweenness_bfs(adj: dict[int, set]) -> dict[int, float]:
    """媒介中心性(Brandes・非加重・無向)。純 Python の BFS のみ。孤立ノードは 0。"""
    cb: dict[int, float] = {n: 0.0 for n in adj}
    for s in adj:
        stack: list[int] = []
        pred: dict[int, list[int]] = {w: [] for w in adj}
        sigma: dict[int, float] = {w: 0.0 for w in adj}
        sigma[s] = 1.0
        dist: dict[int, int] = {w: -1 for w in adj}
        dist[s] = 0
        q = deque([s])
        while q:
            v = q.popleft()
            stack.append(v)
            for w in adj[v]:
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    q.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)
        delta: dict[int, float] = {w: 0.0 for w in adj}
        while stack:
            w = stack.pop()
            for v in pred[w]:
                if sigma[w] > 0:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                cb[w] += delta[w]
    for n in cb:                                  # 無向は 2 重計上を割る
        cb[n] /= 2.0
    return cb


def communication_stats(events: list[dict]) -> dict[int, dict]:
    """speak/dm(発信)と hear(受信)の集計。{agent: {speak_out, dm_out, listen_in, heard_by,
    first_comm_step}}。heard_by = 他者に聴取された回数(hear の speaker が当人)。"""
    st: dict[int, dict] = defaultdict(
        lambda: {"speak_out": 0, "dm_out": 0, "listen_in": 0,
                 "heard_by": 0, "first_comm_step": None})
    for e in events:
        aid = e["agent"]
        if aid is None or aid < 0:
            continue
        k = e["kind"]
        if k == "speak":
            st[aid]["speak_out"] += 1
            _bump_first(st[aid], e["step"])
        elif k == "dm":
            st[aid]["dm_out"] += 1
            _bump_first(st[aid], e["step"])
        elif k == "hear":
            st[aid]["listen_in"] += 1
            sp = e["payload"].get("speaker")
            if sp is not None:
                try:
                    st[int(sp)]["heard_by"] += 1
                except (TypeError, ValueError):
                    pass
    return dict(st)


def _bump_first(rec: dict, step: int) -> None:
    if rec["first_comm_step"] is None or step < rec["first_comm_step"]:
        rec["first_comm_step"] = step


def detect_hub_founders(adj: dict[int, set], reach: dict[int, dict],
                        comm: dict[int, dict], *, hub_degree: int = DEF_HUB_DEGREE,
                        top_k: int = DEF_TOP_K) -> dict[int, dict]:
    """ハブ型ファウンダー候補 = 例外的な「上位者」。次数 / 媒介中心性 / 発信超過の各 top_k の和集合。

    密な関係網では大多数が高次数になる(=固定閾値は無意味)ため、ファウンダー観察では分布の
    **上位**を採る。次数は hub_degree を下限フロアにしつつ、その中の上位 top_k(=最も繋がる者)。
    媒介は次数>=2 の中の上位 top_k(=橋渡し役=ブローカー)。発信超過は正の surplus の上位 top_k。

    {agent: {onset_step, signals:set, degree, betweenness, broadcast_surplus}}。
    onset_step: 次数ハブ=hub_degree 到達 step / 媒介のみ=次数2到達 step / 発信のみ=初回発話 step。"""
    degree = {n: len(nb) for n, nb in adj.items()}
    betw = betweenness_bfs(adj) if adj else {}

    surplus: dict[int, int] = {}
    for aid, c in comm.items():
        surplus[aid] = (c["speak_out"] + c["dm_out"]) - c["listen_in"]

    # 上位 K(決定論: 値降順 → agent_id 昇順)
    def top(scores: dict[int, float], k: int, positive: bool) -> set[int]:
        items = [(a, v) for a, v in scores.items() if (v > 0 if positive else True)]
        items.sort(key=lambda x: (-x[1], x[0]))
        return {a for a, _ in items[:k]}

    # 次数: フロア(hub_degree)以上の中の上位 top_k のみ(=分布の上端。全員ハブ化を防ぐ)
    elig_deg = {a: d for a, d in degree.items() if d >= hub_degree}
    by_degree = top(elig_deg, top_k, False)
    by_betw = top({a: v for a, v in betw.items() if degree.get(a, 0) >= 2}, top_k, True)
    by_cast = top(surplus, top_k, True)

    def onset(aid: int) -> int | None:
        ds = reach.get(aid, {}).get("degree_step", {})
        deg = degree.get(aid, 0)
        if aid in by_degree and hub_degree in ds:
            return ds[hub_degree]
        if deg >= 2 and 2 in ds:                 # 媒介ハブ: 「橋」になった=2人目に達した step
            return ds[2]
        if ds:                                    # 次数1: 初めて関係を持った step
            return min(ds.values())
        return comm.get(aid, {}).get("first_comm_step")

    out: dict[int, dict] = {}
    for aid in by_degree | by_betw | by_cast:
        signals: set[str] = set()
        if aid in by_degree:
            signals.add("degree")
        if aid in by_betw:
            signals.add("betweenness")
        if aid in by_cast:
            signals.add("broadcast")
        out[aid] = {"onset_step": onset(aid), "signals": signals,
                    "degree": degree.get(aid, 0),
                    "betweenness": round(betw.get(aid, 0.0), 4),
                    "broadcast_surplus": surplus.get(aid, 0)}
    return out


# --------------------------------------------------------------------------- #
# ファウンダー候補の統合
# --------------------------------------------------------------------------- #
def merge_founders(venture: dict, political: dict, hub: dict) -> dict[int, dict]:
    """3タイプを統合。{agent: {types:set, ref_step, detail:{...}}}。ref_step=初回ファウンダー行動。"""
    out: dict[int, dict] = {}
    for aid, r in venture.items():
        out.setdefault(aid, {"types": set(), "ref_step": None, "detail": {}})
        out[aid]["types"].add("venture")
        out[aid]["detail"]["venture"] = {
            "subtypes": sorted(r["subtypes"]), "n_actions": r["n_actions"],
            "onset_step": r["onset_step"]}
        out[aid]["ref_step"] = _min_opt(out[aid]["ref_step"], r["onset_step"])
    for aid, r in political.items():
        out.setdefault(aid, {"types": set(), "ref_step": None, "detail": {}})
        out[aid]["types"].add("political")
        out[aid]["detail"]["political"] = {
            "subtypes": sorted(r["subtypes"]), "n_candidacy": r["n_candidacy"],
            "n_proposal": r["n_proposal"], "onset_step": r["onset_step"]}
        out[aid]["ref_step"] = _min_opt(out[aid]["ref_step"], r["onset_step"])
    for aid, r in hub.items():
        out.setdefault(aid, {"types": set(), "ref_step": None, "detail": {}})
        out[aid]["types"].add("hub")
        out[aid]["detail"]["hub"] = {
            "signals": sorted(r["signals"]), "degree": r["degree"],
            "betweenness": r["betweenness"], "broadcast_surplus": r["broadcast_surplus"],
            "onset_step": r["onset_step"]}
        out[aid]["ref_step"] = _min_opt(out[aid]["ref_step"], r["onset_step"])
    return out


def _min_opt(a: int | None, b: int | None) -> int | None:
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


# --------------------------------------------------------------------------- #
# 2/3. 形成前履歴パネル + 対照群マッチング
# --------------------------------------------------------------------------- #
def _age_band(age) -> int:
    try:
        return (int(age) // 10) * 10
    except (TypeError, ValueError):
        return -1


def match_controls(founders: dict[int, dict], agents: dict[int, dict],
                   per_founder: int = 1) -> dict[int, int]:
    """ファウンダー行動なしの全員から 同職業×同年齢帯 で決定論マッチ。{control_id: founder_id}。

    決定論: 候補プールを agent_id 昇順、ファウンダーも agent_id 昇順で処理し、未使用の対照から
    先頭を採る(1人の対照は1人のファウンダーにのみ割当)。該当なしのファウンダーは対照0でも可。"""
    founder_ids = set(founders)
    # (occupation, age_band) -> [agent_id...](ファウンダー除外・agent_id 昇順)
    pool: dict[tuple, list[int]] = defaultdict(list)
    for aid in sorted(agents):
        if aid in founder_ids:
            continue
        a = agents[aid]
        key = (a.get("occupation"), _age_band(a.get("age")))
        pool[key].append(aid)

    used: set[int] = set()
    matches: dict[int, int] = {}
    for fid in sorted(founder_ids):
        a = agents.get(fid, {})
        key = (a.get("occupation"), _age_band(a.get("age")))
        taken = 0
        for cid in pool.get(key, []):
            if cid in used:
                continue
            matches[cid] = fid
            used.add(cid)
            taken += 1
            if taken >= per_founder:
                break
    return matches


def _empty_daily() -> dict:
    return {"income": 0.0, "spend": 0.0, "tax": 0.0, "op_move": 0.0,
            "relacts": 0, "speak": 0, "dm_out": 0, "listen": 0}


def build_daily_index(events: list[dict],
                      spd: int = STEPS_PER_DAY) -> tuple[dict, dict]:
    """全イベントを1パスで (agent, day) 集計。返り値 (G, heard_by_day)。

    G[agent] = {
        'daily': {day: {income,spend,tax,op_move,relacts,speak,dm_out,listen}},
        'balance_by_day': {day: last_balance}, 'op_last_by_day': {day: last_new},
        'griev_by_day': {day: last_new}, 'eff_by_day': {day: last_new},
        'neighbor_day': {other: first_day},  # 次数の成長(累積次数の算定に使う)
    }
    heard_by_day[speaker][day] = 他者に聴取された回数。"""
    G: dict[int, dict] = {}
    heard_by_day: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    def slot(aid: int) -> dict:
        g = G.get(aid)
        if g is None:
            g = G[aid] = {"daily": defaultdict(_empty_daily), "balance_by_day": {},
                          "op_last_by_day": {}, "griev_by_day": {}, "eff_by_day": {},
                          "neighbor_day": {}}
        return g

    for e in events:
        aid = e["agent"]
        if aid is None or aid < 0:
            continue
        k = e["kind"]
        day = e["step"] // int(spd)
        p = e["payload"]
        if k == "wage":
            g = slot(aid); g["daily"][day]["income"] += _f(p.get("amount"))
            if p.get("balance") is not None:
                g["balance_by_day"][day] = _f(p.get("balance"))
        elif k == "spend":
            g = slot(aid); g["daily"][day]["spend"] += _f(p.get("amount"))
            if p.get("balance") is not None:
                g["balance_by_day"][day] = _f(p.get("balance"))
        elif k == "withdraw":
            g = slot(aid)
            if p.get("cash") is not None:
                g["balance_by_day"][day] = _f(p.get("cash"))
        elif k == "tax":
            slot(aid)["daily"][day]["tax"] += _f(p.get("amount"))
        elif k == "opinion_shift":
            g = slot(aid)
            old, new = p.get("old"), p.get("new")
            if old is not None and new is not None:
                g["daily"][day]["op_move"] += abs(_f(new) - _f(old))
                g["op_last_by_day"][day] = _f(new)
        elif k == "state_update":
            nm = p.get("name")
            if nm == "grievance" and p.get("new") is not None:
                slot(aid)["griev_by_day"][day] = _f(p.get("new"))
            elif nm == "efficacy" and p.get("new") is not None:
                slot(aid)["eff_by_day"][day] = _f(p.get("new"))
        elif k == "relation_tier":
            g = slot(aid); g["daily"][day]["relacts"] += 1
            other = p.get("other")
            if other is not None:
                try:
                    other = int(other)
                    if other not in g["neighbor_day"]:
                        g["neighbor_day"][other] = day
                except (TypeError, ValueError):
                    pass
        elif k == "speak":
            slot(aid)["daily"][day]["speak"] += 1
        elif k == "dm":
            slot(aid)["daily"][day]["dm_out"] += 1
        elif k == "hear":
            slot(aid)["daily"][day]["listen"] += 1
            sp = p.get("speaker")
            if sp is not None:
                try:
                    heard_by_day[int(sp)][day] += 1
                except (TypeError, ValueError):
                    pass
    return G, heard_by_day


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _carry(by_day: dict[int, float], day: int) -> float | None:
    """day 以下で最も新しい記録値を返す(carry-forward)。無ければ None。"""
    best_d, best_v = None, None
    for d, v in by_day.items():
        if d <= day and (best_d is None or d > best_d):
            best_d, best_v = d, v
    return best_v


PANEL_COLS = [
    ("run", "string"), ("agent_id", "int64"), ("founder", "int64"),
    ("ftype", "string"), ("matched_to", "int64"), ("occupation", "string"),
    ("age", "int64"), ("age_band", "int64"), ("ref_step", "int64"),
    ("ref_day", "int64"), ("rel_day", "int64"), ("day", "int64"),
    ("money_income", "float64"), ("money_spend", "float64"), ("money_tax", "float64"),
    ("money_net", "float64"), ("balance_end", "float64"),
    ("n_relacts", "int64"), ("degree_cum", "int64"),
    ("opinion_move", "float64"), ("opinion_last", "float64"),
    ("grievance_last", "float64"), ("efficacy_last", "float64"),
    ("speak_count", "int64"), ("heard_by_others", "int64"), ("dm_out", "int64"),
    ("k_fire_weight", "float64"), ("drive_threshold", "float64"),
    ("internal_locus", "float64"), ("nfc", "float64"), ("risk_tolerance", "float64"),
]


def build_panel_rows(run_name: str, founders: dict[int, dict], controls: dict[int, int],
                     agents: dict[int, dict], traits: dict[int, dict],
                     G: dict, heard_by_day: dict, *, pre_days: int,
                     spd: int = STEPS_PER_DAY) -> list[dict]:
    """ファウンダー + 対照の日次パネル行を生成(founder=0/1)。窓 = [ref_day-pre_days, ref_day]。"""
    rows: list[dict] = []

    def emit(aid: int, is_founder: int, ftype: str, matched_to: int, ref_step: int):
        if ref_step is None:
            return
        ref_day = ref_step // int(spd)
        a = agents.get(aid, {})
        tr = traits.get(aid, {})
        g = G.get(aid)
        for rel in range(-pre_days, 1):              # -pre_days .. 0(=当日)
            day = ref_day + rel
            if day < 0:
                continue
            daily = (g["daily"].get(day) if g else None) or _empty_daily()
            income, spend, tax = daily["income"], daily["spend"], daily["tax"]
            balance = _carry(g["balance_by_day"], day) if g else None
            degree_cum = 0
            if g:
                degree_cum = sum(1 for fd in g["neighbor_day"].values() if fd <= day)
            op_last = _carry(g["op_last_by_day"], day) if g else None
            griev = _carry(g["griev_by_day"], day) if g else None
            eff = _carry(g["eff_by_day"], day) if g else None
            heard = heard_by_day.get(aid, {}).get(day, 0)
            rows.append({
                "run": run_name, "agent_id": aid, "founder": is_founder,
                "ftype": ftype, "matched_to": matched_to,
                "occupation": _s(a.get("occupation")), "age": _i(a.get("age")),
                "age_band": _age_band(a.get("age")), "ref_step": ref_step,
                "ref_day": ref_day, "rel_day": rel, "day": day,
                "money_income": round(income, 2), "money_spend": round(spend, 2),
                "money_tax": round(tax, 2), "money_net": round(income - spend - tax, 2),
                "balance_end": balance,
                "n_relacts": daily["relacts"], "degree_cum": degree_cum,
                "opinion_move": round(daily["op_move"], 4), "opinion_last": op_last,
                "grievance_last": griev, "efficacy_last": eff,
                "speak_count": daily["speak"], "heard_by_others": heard,
                "dm_out": daily["dm_out"],
                "k_fire_weight": _opt_f(tr.get("fire_weight")),
                "drive_threshold": _opt_f(tr.get("drive_threshold")),
                "internal_locus": _opt_f(tr.get("internal_locus")),
                "nfc": _opt_f(tr.get("nfc")),
                "risk_tolerance": _opt_f(tr.get("risk_tolerance")),
            })

    for fid in sorted(founders):
        rec = founders[fid]
        ftype = "|".join(sorted(rec["types"]))
        emit(fid, 1, ftype, fid, rec["ref_step"])
    for cid in sorted(controls):
        fid = controls[cid]
        emit(cid, 0, "control", fid, founders[fid]["ref_step"])
    return rows


def _s(v):
    return None if v is None else str(v)


def _i(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _opt_f(v):
    if v is None:
        return None
    try:
        return round(float(v), 6)
    except (TypeError, ValueError):
        return None


def write_panel_parquet(rows: list[dict], out_dir: str) -> str:
    import pyarrow as pa
    import pyarrow.parquet as pq

    os.makedirs(out_dir, exist_ok=True)
    tmap = {"string": pa.string(), "int64": pa.int64(), "float64": pa.float64()}
    schema = pa.schema([pa.field(n, tmap[t]) for n, t in PANEL_COLS])
    cols: dict[str, list] = {n: [] for n, _ in PANEL_COLS}
    for r in rows:
        for n, _ in PANEL_COLS:
            cols[n].append(r.get(n))
    path = os.path.join(out_dir, "founders.parquet")
    pq.write_table(pa.Table.from_pydict(cols, schema=schema), path)
    return path


# --------------------------------------------------------------------------- #
# 4. 組織形成の検出(自然発生候補): 建物 × 反復共在クラスタ
# --------------------------------------------------------------------------- #
def detect_org_candidates(events: list[dict], agents: dict[int, dict],
                          building_names: dict[str, str], *,
                          min_days: int = DEF_COPRESENCE_DAYS,
                          min_core: int = DEF_MIN_CORE,
                          max_core: int = DEF_MAX_CORE,
                          spd: int = STEPS_PER_DAY) -> dict:
    """同一建物 × 反復共在クラスタを列挙。既存台帳(勤務先/自宅建物)は known として除外。

    共在 = 同じ建物に同じ日に enter_building した集合。連続 min_days 日以上、その共在の
    交わり(=毎日いた核)が min_core 人以上のとき「候補」。核が max_core を超えるものは
    公共空間の通過とみなし category='crowd'。核の建物が過半の勤務先=workplace、自宅=home。"""
    # building -> {day -> set(agents)}
    bd: dict[str, dict[int, set]] = defaultdict(lambda: defaultdict(set))
    for e in events:
        if e["kind"] != "enter_building":
            continue
        aid = e["agent"]
        b = e["payload"].get("building")
        if aid is None or aid < 0 or not b:
            continue
        bd[b][e["step"] // int(spd)].add(aid)

    clusters: list[dict] = []
    seen_keys: set[tuple] = set()
    for b in sorted(bd):
        day2set = bd[b]
        days = sorted(day2set)
        n = len(days)
        for start in range(n):
            core = set(day2set[days[start]])
            run = [days[start]]
            for j in range(start + 1, n):
                if days[j] != days[j - 1] + 1:       # 連続日でない=打ち切り
                    break
                newcore = core & day2set[days[j]]
                if len(newcore) < min_core:
                    break
                core = newcore
                run.append(days[j])
            if len(run) >= min_days and len(core) >= min_core:
                key = (b, frozenset(core))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                clusters.append({"building": b, "core": frozenset(core),
                                 "first_day": run[0], "last_day": run[-1],
                                 "n_days": len(run)})

    def category(cl: dict) -> str:
        core = cl["core"]
        need = math.ceil(len(core) / 2)
        n_work = sum(1 for m in core
                     if agents.get(m, {}).get("work_building") == cl["building"])
        n_home = sum(1 for m in core
                     if agents.get(m, {}).get("home_building") == cl["building"])
        if n_work >= need:
            return "known_workplace"
        if n_home >= need:
            return "known_home"
        if len(core) > max_core:
            return "crowd"
        return "emergent"

    emergent, known, crowd = [], [], []
    for cl in clusters:
        cat = category(cl)
        members = sorted(cl["core"])
        rec = {"building": cl["building"],
               "building_name": building_names.get(cl["building"], ""),
               "members": members, "n_members": len(members),
               "first_day": cl["first_day"], "last_day": cl["last_day"],
               "copresent_days": cl["n_days"], "category": cat}
        if cat == "emergent":
            emergent.append(rec)
        elif cat == "crowd":
            crowd.append(rec)
        else:
            known.append(rec)

    def _sort(lst):
        lst.sort(key=lambda r: (-r["copresent_days"], -r["n_members"],
                                r["building"], r["first_day"]))
        return lst

    return {
        "params": {"min_days": min_days, "min_core": min_core, "max_core": max_core},
        "n_clusters_total": len(clusters),
        "n_emergent": len(emergent), "n_known": len(known), "n_crowd": len(crowd),
        "candidates": _sort(emergent),
        "known_workplace_home": _sort(known),
        "crowd": _sort(crowd),
    }


# --------------------------------------------------------------------------- #
# 実行・統合
# --------------------------------------------------------------------------- #
def analyze(run_dir: str, out_dir: str, *, pre_days: int = DEF_PRE_DAYS,
            proposal_min: int = DEF_PROPOSAL_MIN, hub_degree: int = DEF_HUB_DEGREE,
            top_k: int = DEF_TOP_K, controls_per_founder: int = 1,
            copresence_days: int = DEF_COPRESENCE_DAYS,
            min_core: int = DEF_MIN_CORE, max_core: int = DEF_MAX_CORE) -> dict:
    """ラン1本を解析し、panel/founders.parquet と panel/org_candidates.json を書く。"""
    run_name = os.path.basename(os.path.normpath(run_dir))
    # W2-3: 「1 日」はこのランの run.dt_min で決まる(Δt=10 なら 144 = 従来と同値)。
    spd = run_dt.steps_per_day(run_dir)
    dt_min = run_dt.min_per_step(run_dir)
    events = load_events(run_dir, dt_min)
    agents = load_agents(run_dir)
    traits = load_traits(run_dir)
    building_names = load_building_names(run_dir)

    venture = detect_venture_founders(events)
    political = detect_political_founders(events, proposal_min=proposal_min)
    adj, reach = build_relation_graph(events)
    comm = communication_stats(events)
    hub = detect_hub_founders(adj, reach, comm, hub_degree=hub_degree, top_k=top_k)
    founders = merge_founders(venture, political, hub)

    controls = match_controls(founders, agents, per_founder=controls_per_founder)
    G, heard_by_day = build_daily_index(events, spd)
    rows = build_panel_rows(run_name, founders, controls, agents, traits, G, heard_by_day,
                            pre_days=pre_days, spd=spd)
    orgs = detect_org_candidates(events, agents, building_names,
                                 min_days=copresence_days, min_core=min_core,
                                 max_core=max_core, spd=spd)

    os.makedirs(out_dir, exist_ok=True)
    panel_path = write_panel_parquet(rows, out_dir)
    org_path = os.path.join(out_dir, "org_candidates.json")
    org_out = {"run": run_name, **orgs}
    with open(org_path, "w", encoding="utf-8") as fh:
        json.dump(org_out, fh, ensure_ascii=False, indent=2)

    # ファウンダー明細(JSON。set は list 化)
    founders_json = {}
    for aid, r in founders.items():
        founders_json[aid] = {"types": sorted(r["types"]), "ref_step": r["ref_step"],
                              "ref_day": (r["ref_step"] // spd
                                          if r["ref_step"] is not None else None),
                              "detail": r["detail"]}
    fdet_path = os.path.join(out_dir, "founders_detail.json")
    with open(fdet_path, "w", encoding="utf-8") as fh:
        json.dump({"run": run_name, "founders": {str(k): v for k, v in founders_json.items()}},
                  fh, ensure_ascii=False, indent=2)

    return {
        "run": run_name,
        "n_venture": len(venture), "n_political": len(political), "n_hub": len(hub),
        "n_founders_total": len(founders), "n_controls": len(controls),
        "panel_rows": len(rows),
        "org": orgs,
        "founders": founders,
        "paths": {"panel": panel_path, "org": org_path, "detail": fdet_path},
    }


def _pick_run(arg_run: str | None) -> str:
    """--run 指定 or l1_events.parquet を持つ最新ランを返す(detect_emergence と同流儀)。"""
    if arg_run:
        d = arg_run if os.path.isabs(arg_run) else os.path.join(RUNS_ROOT, arg_run)
        if not os.path.isfile(os.path.join(d, "l1_events.parquet")):
            raise SystemExit(f"[founders] l1_events.parquet が無い: {d}")
        return d
    if not os.path.isdir(RUNS_ROOT):
        raise SystemExit(f"[founders] runs が無い: {RUNS_ROOT}")
    cands = []
    for name in os.listdir(RUNS_ROOT):
        pq_path = os.path.join(RUNS_ROOT, name, "l1_events.parquet")
        if os.path.isfile(pq_path):
            cands.append((os.path.getmtime(pq_path), os.path.join(RUNS_ROOT, name)))
    if not cands:
        raise SystemExit("[founders] l1_events.parquet を持つランが無い")
    cands.sort(reverse=True)
    return cands[0][1]


def _resolve_out(run_dir: str, arg_out: str | None) -> str:
    """--out。既定=run_dir/panel。相対指定は run_dir 起点、絶対はそのまま。"""
    if not arg_out:
        return os.path.join(run_dir, "panel")
    return arg_out if os.path.isabs(arg_out) else os.path.join(run_dir, arg_out)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="組織の自然形成 + ファウンダー成立条件の観測装置(読み出し専用)")
    ap.add_argument("--run", default=None,
                    help="ラン名(既定: l1_events.parquet を持つ最新ラン)")
    ap.add_argument("--out", default=None,
                    help="出力ディレクトリ(既定: <run>/panel)。相対はラン基準")
    ap.add_argument("--pre-days", type=int, default=DEF_PRE_DAYS,
                    help=f"形成前履歴の窓(前N日。既定 {DEF_PRE_DAYS})")
    ap.add_argument("--proposal-min", type=int, default=DEF_PROPOSAL_MIN,
                    help=f"政治型: 高頻度提案の閾値(既定 {DEF_PROPOSAL_MIN})")
    ap.add_argument("--hub-degree", type=int, default=DEF_HUB_DEGREE,
                    help=f"ハブ型: 次数の閾値(既定 {DEF_HUB_DEGREE})")
    ap.add_argument("--top-k", type=int, default=DEF_TOP_K,
                    help=f"ハブ型: 媒介中心性/発信超過の上位K(既定 {DEF_TOP_K})")
    ap.add_argument("--controls-per-founder", type=int, default=1,
                    help="ファウンダー1名あたりの対照数(既定 1)")
    ap.add_argument("--copresence-days", type=int, default=DEF_COPRESENCE_DAYS,
                    help=f"組織候補: 連続共在の最小日数(既定 {DEF_COPRESENCE_DAYS})")
    args = ap.parse_args()

    run_dir = _pick_run(args.run)
    out_dir = _resolve_out(run_dir, args.out)
    res = analyze(run_dir, out_dir, pre_days=args.pre_days,
                  proposal_min=args.proposal_min, hub_degree=args.hub_degree,
                  top_k=args.top_k, controls_per_founder=args.controls_per_founder,
                  copresence_days=args.copresence_days)

    org = res["org"]
    print(f"[founders] run={res['run']}")
    print(f"[founders] ファウンダー候補: 起業型 {res['n_venture']} / 政治型 "
          f"{res['n_political']} / ハブ型 {res['n_hub']} / 実数(重複除く) "
          f"{res['n_founders_total']}")
    print(f"[founders] 対照群 {res['n_controls']} 名 / パネル行数 {res['panel_rows']}")
    print(f"[founders] 組織候補(自然発生): {org['n_emergent']} 件 "
          f"(既存台帳一致 {org['n_known']} / 群集 {org['n_crowd']} / "
          f"総クラスタ {org['n_clusters_total']})")
    # 上位の見どころ
    top_hub = sorted(
        ((aid, r["detail"].get("hub", {})) for aid, r in res["founders"].items()
         if "hub" in r["types"]),
        key=lambda x: (-x[1].get("degree", 0), x[0]))[:5]
    if top_hub:
        print("\n[hub top]")
        for aid, h in top_hub:
            print(f"  agent {aid}: degree={h.get('degree')} "
                  f"betw={h.get('betweenness')} surplus={h.get('broadcast_surplus')} "
                  f"signals={h.get('signals')}")
    if org["candidates"]:
        print("\n[org candidates top]")
        for c in org["candidates"][:5]:
            nm = c["building_name"] or c["building"]
            print(f"  {nm}: members={c['members']} days={c['copresent_days']} "
                  f"(day {c['first_day']}-{c['last_day']})")
    print(f"\n[founders] -> {res['paths']['panel']}")
    print(f"[founders] -> {res['paths']['org']}")
    print(f"[founders] -> {res['paths']['detail']}")


if __name__ == "__main__":
    main()
