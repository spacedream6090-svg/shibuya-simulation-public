"""W4-D: 研究解析スクリプトの `l1_stream` 一括移行の**出力同値**を機械固定する。

何を守るのか
------------
移行の中身は 2 種類しかない。

  (1) **kind 絞り読み** — `measure.load_events` / 自前の全件ローダを
      `l1_stream.iter_events(kinds=WANT_KINDS)` に置き換えた。
      これが安全なのは「WANT_KINDS 以外の行を下流が必ず読み飛ばす」からで、
      その主張を本 file が **フィルタ有り/無しの出力比較**で実証する。

  (2) **全 kind 依存の量の付け替え** — 絞り読みにすると壊れる量
      (`max(agent_id)+1` の母数・日軸の右端・総件数・在街表・stage0 検証)を
      `l1_stream` の**列走査/フッタ読み**へ移した。こちらは
      「列走査の値 == 全件走査の値」を直接突き合わせて固定する。

さらに、将来 kind を 1 つ足したときに WANT_KINDS の更新を忘れる事故を防ぐため、
**モジュールの文字列リテラル ∩ EVENT_KINDS ⊆ WANT_KINDS** を AST で機械検査する
(kind として使っていないリテラルだけを、理由付きの明示例外にする)。
凍結側 `observer/measure.py` の kind 集合の写し(`l1_stream.*_KINDS`)も、
measure.py の AST から抽出して突き合わせる(人手の同期に頼らない)。

実 LLM は呼ばない。pandas / duckdb は使わない。
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "src"))

import l1_stream as ls                                  # noqa: E402
from society.observer import measure as m               # noqa: E402
from society.observer.schema import EVENT_KINDS         # noqa: E402

N_AGENTS = 6
N_STEPS = 300                     # 3 日ぶん(Δt=10 なら 144 step/日 = 0/1/2 日)
SPD = 144
_WORLD_KINDS = {"weather", "disaster", "annual_event", "presence_change",
                "council_elected", "election_result", "scenario_shock",
                "world_event", "norm_digest", "public_budget", "spark_roster",
                "traffic_flow", "workplace_bound", "infra_outage",
                "transit_delay", "crowd_surge"}


def _payload(kind: str, step: int, aid: int) -> dict:
    """どの分岐にも刺さるように「ありそうなキー」を全部入れた汎用 payload。

    ある kind を WANT_KINDS から取りこぼしたとき、その kind の分岐が **0 でない値**
    を作らないと同値比較で差が出ない。だから
      - 金額・相手 id・テキストを常に入れ、
      - **相手は kind ごとに変える**(speak と chance_event が同じ相手を指すと、
        片方を落としても接触グラフが変わらず取りこぼしを検出できない)、
      - 分岐が特定の値を要求する kind(`crime{kind:"theft"}` /
        `state_update{name:"grievance"}` など)は **その値**を入れる。
    これらは下の `test_synthetic_l1_is_sensitive_to_every_want_kind` が
    「WANT_KINDS から 1 種抜くと必ず出力が動くか」を機械確認している。
    """
    # kind ごとに違う相手を指す(接触/伝播グラフで kind を区別できるように)
    other = (aid + 1 + (sum(map(ord, kind)) % (N_AGENTS - 2))) % N_AGENTS
    if other == aid:
        other = (aid + 1) % N_AGENTS
    # balance / account は **kind を見ずに拾われる**(build_panel の資産追跡)。
    # 値を kind × step で変えないと「最後に書いた kind」が入れ替わっても値が動かず、
    # 取りこぼしを検出できない。
    bal = 100.0 + step + (sum(map(ord, kind)) % 97)
    p = {
        "amount": 12.5, "balance": bal, "account": bal / 2.0,
        "cash": 50.0, "fare": 3.0,
        "cost": 5.0, "price": 7.0, "deposit": 300.0, "n": 1, "steps": 2,
        "cat": "food", "category": "food", "medium": "TV", "service": "gym",
        "subject": "math", "output": "x", "org": "org1", "role": "staff",
        "other": other, "to": other, "from": other, "speaker": other,
        "hearers": [other], "author": other, "host": other, "founder": other,
        "proposer": other, "created_by": aid, "victim": other,
        "offender": other, "source": other, "courier": other,
        "with": [other], "knowers": [aid, other], "members": [aid, other],
        "item_id": "w1", "items": ["w1"], "text": f"はなし{step % 7}",
        "belief": f"しんねん{step % 5}", "written_back": True,
        "word": "ことば", "name": f"みせ{aid}", "title": "たいとる",
        "purpose": "もくてき", "offer": "coffee", "label": "よろこび",
        "goal": "もくひょう", "query": "しぶや", "reason": "r", "cause": "c",
        "node": f"n{aid % 3}", "building": f"b{aid % 3}", "poi": f"poi{aid}",
        "floor": 1, "dest": f"n{aid % 3}", "channel": "face",
        "tier": 1, "to_tier": 0, "from_tier": 1, "count": 1,
        "new": 0.6, "old": 0.4, "granted": True, "permitted": True,
        "outcome": "ok", "state": "onset", "phase": "onset", "type": "encounter",
        "kind": "typhoon", "cond": "雨", "weekday": "Mon", "holiday": False,
        "date": "2026-08-01", "temp_hi": 30,
        "event_id": f"e{aid}", "group_id": f"g{aid}", "proposal_id": f"p{aid}",
        "rule_id": f"r{aid}", "post_id": f"po{aid}", "campaign": "c1",
        "slot": "s1", "target": "t1", "day": step // SPD,
        "slept_steps": 3, "until_step": step + 3, "nights": 1,
        "ctrl": 0.5, "expect_n": 2, "err_mean": 0.1, "err_n": 1,
        "norm_rate": 0.3, "pioneer_1d": 1, "plan": [], "what": "さんぽ",
        "mode": "walk", "trigger": "t", "n_seen": 1, "reach": 2,
        "ids": [0, 1], "params": {"k": 1}, "menus": {"a": 1},
        "activation": 0.1, "tau": 0.2, "gauge": 0.3, "level": 1,
    }
    # ---- 分岐が特定の値を要求する kind(ここを外すと「読んでも 0」になる)----
    # item_id は **kind をまたいで共有**する(transmission → label_adopt の
    # 系譜が繋がらないと「採用の出所」系の分岐が全部 0 になる)。
    p["item_id"] = f"w{aid % 2}"
    if kind == "spark_roster":
        p["ids"] = list(range(N_AGENTS // 2))
    elif kind == "crime":
        p["kind"] = "theft"                            # build_panel の n_crime 条件
    elif kind == "state_update":
        p["name"] = "grievance" if aid % 2 else "efficacy"   # founders の日次
    elif kind == "health_update":
        p["name"] = "fatigue"
    elif kind == "day_plan":
        p["plan"] = [{"what": f"よてい{aid}", "place": f"ばしょ{aid % 3}"}]
    elif kind == "lodging_checkin":
        p["building"] = f"b{aid % 3}"                  # 在館カテゴリの lodging 判定
    elif kind in ("enter_building", "exit_building"):
        p["activity"] = "work" if aid % 2 else "leisure"
    elif kind == "life_event":
        p["kind"] = "partner"
    elif kind == "chance_event":
        p["type"] = "encounter"
    elif kind == "illness":
        p["state"] = "onset"
    elif kind == "transmission":
        p["channel"] = "sns" if aid % 2 else "face"
    return p


def _events() -> list[dict]:
    """登録済み **全 kind** を、複数 agent × 複数 step にわたって 1 本の L1 に流す。"""
    kinds = sorted(EVENT_KINDS)
    ev: list[dict] = []
    step = 0
    for i, kind in enumerate(kinds):
        for rep in range(5):        # 同じ kind を 5 回・全期間に散らす(窓・順序を突く)
            step = (i * 7 + rep * 61) % N_STEPS
            if kind in _WORLD_KINDS:
                ev.append({"step": step, "agent_id": -1, "kind": kind,
                           "payload": _payload(kind, step, 0)})
            for aid in range(N_AGENTS):
                ev.append({"step": step, "agent_id": aid, "kind": kind,
                           "payload": _payload(kind, step, aid)})
    # worldview は「街の行(-1)」と「個体の行」の両方が要る(split_worldview)
    for d in (0, 1):
        s = d * SPD + 10
        ev.append({"step": s, "agent_id": -1, "kind": "worldview",
                   "payload": _payload("worldview", s, 0)})
        for aid in range(N_AGENTS):
            ev.append({"step": s, "agent_id": aid, "kind": "worldview",
                       "payload": {"ctrl": 0.4 + 0.05 * aid, "expect_n": 2,
                                   "err_mean": 0.1 * aid, "err_n": 1}})
    ev.sort(key=lambda e: e["step"])
    return ev


def _write_run(rd: Path, events: list[dict]) -> None:
    rd.mkdir(parents=True, exist_ok=True)
    cols = {
        "step": pa.array([e["step"] for e in events], pa.int64()),
        "sim_min": pa.array([e["step"] * 10 + 420 for e in events], pa.int64()),
        "agent_id": pa.array([e["agent_id"] for e in events], pa.int64()),
        "kind": pa.array([e["kind"] for e in events], pa.string()),
        "x": pa.array([float(e["agent_id"]) for e in events], pa.float64()),
        "y": pa.array([float(e["step"] % 5) for e in events], pa.float64()),
        "payload": pa.array([json.dumps(e["payload"], ensure_ascii=False)
                             for e in events], pa.string()),
        "rng_stream": pa.array([None] * len(events), pa.string()),
        "llm_call_id": pa.array([None] * len(events), pa.string()),
    }
    pq.write_table(pa.table(cols), rd / "l1_events.parquet", row_group_size=97)
    agents = [{"id": i, "name": f"あ{i}", "age": 20 + i, "occupation": "会社員",
               "visitor": False, "money": 1000.0, "home_building": f"b{i % 3}",
               "work_building": f"b{(i + 1) % 3}"} for i in range(N_AGENTS)]
    (rd / "agents.json").write_text(json.dumps(agents, ensure_ascii=False),
                                    encoding="utf-8")
    (rd / "summary.json").write_text(
        json.dumps({"n_steps": N_STEPS, "n_agents": N_AGENTS}), encoding="utf-8")
    (rd / "traits.json").write_text(json.dumps(
        {str(i): {"openness": 0.1 * i, "extraversion": 0.2 + 0.1 * i}
         for i in range(N_AGENTS)}, ensure_ascii=False), encoding="utf-8")


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory) -> Path:
    rd = tmp_path_factory.mktemp("w4d") / "run"
    _write_run(rd, _events())
    return rd


@pytest.fixture(scope="module")
def full_events(run_dir) -> list[dict]:
    """全 kind の events(= 移行前のローダが返していたもの)。"""
    return m.load_events(str(run_dir))


# --------------------------------------------------------------------------- #
# A. WANT_KINDS の網羅ガード(AST。kind リテラルの取りこぼしを機械検出)
# --------------------------------------------------------------------------- #
def _module_kind_literals(name: str) -> set:
    src = (_ROOT / "scripts" / f"{name}.py").read_text(encoding="utf-8")
    lits = {n.value for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    return lits & set(EVENT_KINDS)


def _want_of(name: str) -> set:
    mod = __import__(name)
    if name == "audit_uncertainty":
        return set(mod.LUCK_KINDS)
    if name in ("analyze_communities", "analyze_rumor_contamination"):
        return set(mod._want_kinds())
    if name == "judge":
        return set(mod._ACTION_LABELS) | set(ls.AGENT_FEATURES_KINDS)
    if name == "analyze_firing":
        return set(mod.WANTED)
    if name == "panel_stats":
        return set(ls.AGENT_FEATURES_KINDS)
    return set(mod.WANT_KINDS)


# kind として使っていない文字列リテラル(= 絞り読みに関係しない)の明示例外。
_NOT_A_KIND = {
    # observe.py は建物の滞在統計を JSON キー "stay" で出す(EVENT_KIND の
    # "stay"(その場に留まる)とは無関係。observe は stay イベントを読まない)。
    "observe": {"stay"},
    # analyze_communities は共同体ライフサイクルの内部語彙 birth/merge/split/death を
    # 持つ(第64バッチ以来)。第107で L1 kind "death"(個人の死)が誕生して名前が
    # 衝突したが、共同体の「消滅」は L1 の death イベントを 1 行も読まない。
    "analyze_communities": {"death"},
}

_MIGRATED = ["analyze_founders", "audit_uncertainty", "detect_emergence",
             "trace_spark", "analyze_firing", "analyze_communities",
             "analyze_imitation", "analyze_weak_ties", "analyze_worldview",
             "analyze_structure", "build_panel", "judge", "observe",
             "observe_flows", "commercial_report", "panel_stats",
             "analyze_endo_treatment", "analyze_luck",
             "analyze_rumor_contamination"]


@pytest.mark.parametrize("name", [n for n in _MIGRATED
                                  if n != "analyze_endo_treatment"])
def test_want_kinds_covers_every_kind_literal(name):
    """モジュール内の kind リテラルは全部 WANT_KINDS に入っていること。"""
    missing = _module_kind_literals(name) - _want_of(name) - _NOT_A_KIND.get(name, set())
    assert not missing, f"{name}: WANT_KINDS に無い kind リテラル {sorted(missing)}"


def test_endo_treatment_reuses_structure_kind_set():
    """analyze_endo_treatment は自前の kind 表を持たず analyze_structure を使うこと。"""
    import analyze_endo_treatment as endo
    import analyze_structure as ast_mod
    assert not hasattr(endo, "WANT_KINDS")
    assert _module_kind_literals("analyze_endo_treatment") <= set(ast_mod.WANT_KINDS)


# --------------------------------------------------------------------------- #
# B. 凍結側(measure.py)の kind 集合の写しが実装と一致していること
# --------------------------------------------------------------------------- #
def _measure_func_kinds(*func_names: str) -> set:
    """measure.py の指定関数の中の文字列リテラル ∩ EVENT_KINDS を AST で抽出。"""
    src = (_ROOT / "src" / "society" / "observer" / "measure.py").read_text(
        encoding="utf-8")
    tree = ast.parse(src)
    consts: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in func_names:
            consts |= {n.value for n in ast.walk(node)
                       if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    return consts & set(EVENT_KINDS)


def _measure_const(name: str) -> set:
    return set(getattr(m, name))


def test_agent_features_kinds_matches_measure_source():
    got = (_measure_func_kinds("agent_features", "_tool_features", "_item_creators"))
    assert got == set(ls.AGENT_FEATURES_KINDS)


def test_echo_novelty_kinds_matches_measure_source():
    got = _measure_func_kinds("echo_feed") | _measure_const("_UTTERANCE_KINDS")
    assert got == set(ls.ECHO_NOVELTY_KINDS)


def test_item_cascades_kinds_matches_measure_source():
    got = _measure_func_kinds("item_cascades", "_item_creators")
    assert got == set(ls.ITEM_CASCADES_KINDS)


def test_communities_kinds_matches_measure_source():
    got = _measure_func_kinds("_conversation_adj", "communities")
    assert got == set(ls.COMMUNITIES_KINDS)


# --------------------------------------------------------------------------- #
# C. 新しい列走査ヘルパ == 全件走査(全 kind 依存の量の付け替えの証明)
# --------------------------------------------------------------------------- #
def test_row_count_and_max_step(run_dir, full_events):
    assert ls.row_count(run_dir) == len(full_events)
    assert ls.max_step(run_dir) == max(e["step"] for e in full_events)


def test_max_agent_id_matches_full_scan(run_dir, full_events):
    assert ls.max_agent_id(run_dir) == max(
        (e["agent_id"] for e in full_events if e["agent_id"] >= 0), default=-1)
    assert ls.max_agent_id(run_dir, nonneg=False) == max(
        (e["agent_id"] for e in full_events), default=-1)


def test_distinct_agent_ids_matches_full_scan(run_dir, full_events):
    assert ls.distinct_agent_ids(run_dir) == {
        e["agent_id"] for e in full_events if e["agent_id"] >= 0}


def test_max_day_matches_full_scan(run_dir, full_events):
    import analyze_structure as ast_mod
    assert ls.max_day(run_dir, SPD) == max(
        (ast_mod._day_of(e, SPD) for e in full_events), default=-1)
    # sim_min のみ(observe_flows._day 規則)
    assert ls.max_day(run_dir, 1, min_per_day=1440, step_fallback=False) == max(
        (e["sim_min"] // 1440 for e in full_events), default=-1)


def test_max_day_step_fallback_when_sim_min_null(tmp_path):
    """sim_min が null の行は step//spd へ後退する(`_day_of` と同じ規則)。"""
    rd = tmp_path / "r"
    rd.mkdir()
    pq.write_table(pa.table({
        "step": pa.array([0, 500], pa.int64()),
        "sim_min": pa.array([420, None], pa.int64()),
        "agent_id": pa.array([0, 1], pa.int64()),
        "kind": pa.array(["speak", "speak"], pa.string()),
        "payload": pa.array(["{}", "{}"], pa.string()),
    }), rd / "l1_events.parquet")
    assert ls.max_day(rd, SPD) == 500 // SPD          # 3(step 後退が効く)
    assert ls.max_day(rd, SPD, step_fallback=False) == 0


def test_empty_run_helpers(tmp_path):
    rd = tmp_path / "empty"
    rd.mkdir()
    assert ls.max_agent_id(rd) == -1
    assert ls.distinct_agent_ids(rd) == set()
    assert ls.max_day(rd, SPD) == -1


# --------------------------------------------------------------------------- #
# D. スクリプトごとの出力同値(絞り読み == 全件読み)
#
#    各スクリプトのローダを「全 kind を返す版」に差し替えて同じ入口を 2 回叩き、
#    結果 dict が完全一致することを見る。= 「WANT_KINDS 以外の行は下流に効かない」
#    の直接証明。差し替えないヘルパ(列走査)は C で別に固定してある。
# --------------------------------------------------------------------------- #
def _all_events(run_dir):
    return list(ls.iter_events(run_dir))


def _artifacts(d: Path) -> dict:
    """ディレクトリ配下の成果物を「相対パス -> sha256」で返す。

    返り値 dict だけを比べると **書き出した parquet/CSV/md の中身**が比較から
    抜け落ちる(build_panel が典型: 返すのは行数だけで、実体はファイル側)。
    ここで中身そのものをバイト比較に載せる。
    """
    import hashlib
    out = {}
    for p in sorted(Path(d).rglob("*")):
        if p.is_file():
            out[str(p.relative_to(d)).replace("\\", "/")] = hashlib.sha256(
                p.read_bytes()).hexdigest()
    return out


def _equal_out(monkeypatch, mod, loader_attr, call, wide=None):
    """`call()` の結果が「絞り読み」と「全件読み」で一致すること。

    `wide` は「全 kind を返すローダ」。既定は `l1_stream.iter_events`(= measure と
    同一形)だが、自前ローダを持つスクリプト(キー名が `agent` など)は
    **そのローダ自身に kind 絞りを外して呼ばせる**(形を変えずに広げる)。
    """
    narrow = call()
    orig = getattr(mod, loader_attr)
    wide = wide or (lambda run_dir, *a, **kw: _all_events(run_dir))
    monkeypatch.setattr(mod, loader_attr, wide)
    try:
        assert narrow == call()
    finally:
        monkeypatch.setattr(mod, loader_attr, orig)
    return narrow


def test_analyze_structure_equivalence(run_dir, monkeypatch):
    import analyze_structure as mod
    out = _equal_out(monkeypatch, mod, "load_events",
                     lambda: mod.analyze(str(run_dir)))
    assert out["n_days"] > 0


def test_analyze_endo_treatment_equivalence(run_dir, monkeypatch):
    import analyze_endo_treatment as mod
    import analyze_structure as ast_mod
    narrow = mod.run_metrics(str(run_dir))
    monkeypatch.setattr(ast_mod, "load_events",
                        lambda run_dir, *a, **kw: _all_events(run_dir))
    assert narrow == mod.run_metrics(str(run_dir))


def test_analyze_weak_ties_equivalence(run_dir, monkeypatch):
    import analyze_weak_ties as mod
    out = _equal_out(monkeypatch, mod, "_load_events",
                     lambda: mod.analyze(str(run_dir)))
    assert out["n_agents"] == N_AGENTS


def test_analyze_communities_equivalence(run_dir, monkeypatch):
    import analyze_communities as mod
    _equal_out(monkeypatch, mod, "load_events",
               lambda: mod.analyze(str(run_dir), window_days=1))


def test_analyze_imitation_equivalence(run_dir, monkeypatch):
    import analyze_imitation as mod
    _equal_out(monkeypatch, mod, "_load_events",
               lambda: mod.analyze(str(run_dir)))


def test_analyze_worldview_equivalence(run_dir, monkeypatch, tmp_path):
    import analyze_worldview as mod
    seq = {"i": 0}

    def _call():
        seq["i"] += 1
        d = tmp_path / f"wv{seq['i']}"
        d.mkdir()
        out = mod.analyze(str(run_dir), str(d / "wv.md"))
        out.pop("report", None)
        return out, _artifacts(d)
    _equal_out(monkeypatch, mod, "load_events", _call)


def test_detect_emergence_equivalence(run_dir, monkeypatch):
    import detect_emergence as mod
    orig = mod.load_events
    _equal_out(monkeypatch, mod, "load_events",
               lambda: mod.analyze(str(run_dir)),
               wide=lambda rd, *a, **kw: orig(rd, kinds=()))


def test_analyze_founders_equivalence(run_dir, monkeypatch, tmp_path):
    import analyze_founders as mod
    orig = mod.load_events
    seq = {"i": 0}

    def _call():
        seq["i"] += 1
        d = tmp_path / f"f{seq['i']}"
        out = mod.analyze(str(run_dir), str(d))
        out.pop("paths", None)
        return out, _artifacts(d)
    _equal_out(monkeypatch, mod, "load_events", _call,
               wide=lambda rd, dt=10, **kw: orig(rd, dt, kinds=None))


def test_trace_spark_equivalence(run_dir, monkeypatch, tmp_path):
    import trace_spark as mod
    orig = mod.load_events
    seq = {"i": 0}

    def _call():
        seq["i"] += 1
        d = tmp_path / f"s{seq['i']}"
        out = mod.trace(str(run_dir), str(d))
        out.pop("_path", None)
        return out, _artifacts(d)
    out, _ = _equal_out(monkeypatch, mod, "load_events", _call,
                        wide=lambda rd, *a, **kw: orig(rd, kinds=None))
    assert out["spark_enabled"] is True          # spark_roster を読めている


def test_observe_equivalence(run_dir, monkeypatch, tmp_path):
    import observe as mod
    seq = {"i": 0}

    def _call():
        seq["i"] += 1
        d = tmp_path / f"o{seq['i']}"
        return (mod.observe_visits(mod.load_events(str(run_dir)),
                                   mod.load_map(str(run_dir))),
                mod.observe_interests(mod.load_events(str(run_dir)),
                                      m.load_agents(str(run_dir))),
                _artifacts(Path(mod.observe(str(run_dir), str(d)))))
    _equal_out(monkeypatch, mod, "load_events", _call)


def test_observe_flows_equivalence(run_dir, monkeypatch, tmp_path):
    import observe_flows as mod
    name_of = {i: f"あ{i}" for i in range(N_AGENTS)}
    max_day = max(ls.max_day(run_dir, 1, min_per_day=mod.MIN_PER_DAY,
                             step_fallback=False), 0)
    seq = {"i": 0}

    def _call():
        seq["i"] += 1
        d = tmp_path / f"of{seq['i']}"
        ev = mod.load_events(str(run_dir))
        return (mod.observe_money(ev, name_of),
                mod.observe_attention(ev, name_of, max_day=max_day),
                _artifacts(Path(mod.observe(str(run_dir), str(d)))))
    _equal_out(monkeypatch, mod, "load_events", _call)


def test_commercial_report_equivalence(run_dir, monkeypatch, tmp_path):
    import commercial_report as mod
    seq = {"i": 0}

    def _call():
        seq["i"] += 1
        d = tmp_path / f"c{seq['i']}"
        d.mkdir()
        out = mod.report(str(run_dir), str(d / "c.md"))
        out.pop("out_md", None)
        return out, _artifacts(d)
    _equal_out(monkeypatch, mod, "load_events", _call)


def test_build_panel_equivalence(run_dir, monkeypatch, tmp_path):
    """返り値だけでなく **書き出した 4 parquet のバイト列**まで一致すること。"""
    import build_panel as mod
    seq = {"i": 0}

    def _call():
        seq["i"] += 1
        out_dir = tmp_path / f"p{seq['i']}"
        out = mod.build(str(run_dir), out_dir=str(out_dir))
        out.pop("out_dir", None)
        return out, _artifacts(out_dir)
    (info, _art) = _equal_out(monkeypatch, mod, "load_events", _call)
    assert info["n_agent_day_rows"] > 0


def test_build_panel_stage0_stream_matches_full_list(run_dir, full_events):
    """stage0 は **絞れない**段。逐次版が全件 list 版と 1 バイトも違わないこと。"""
    import build_panel as mod
    assert (mod.validate_stage0_stream(str(run_dir), N_STEPS)
            == mod.validate_stage0(full_events, N_STEPS))
    # n_steps=0(summary 欠損相当)でも一致
    assert (mod.validate_stage0_stream(str(run_dir), 0)
            == mod.validate_stage0(full_events, 0))


def test_build_panel_wealth_scan_matches_in_loop_tracking(run_dir, full_events):
    """資産追跡は **kind を見ない**段。逐次版が全件 list 版と完全一致すること。

    ★これは実ラン(runs/calib_base_30d)の HEAD 突合で `agent_day.wealth_end` の
    差として実際に検出された回帰の再発防止。`payload.balance` を持つ kind は
    静的に決まらないので、ここだけは全 kind を読む以外に正しい答えが無い。
    """
    import build_panel as mod
    agents = m.load_agents(str(run_dir))
    scan = mod.scan_all_kinds(str(run_dir), N_STEPS, agents, SPD)
    assert scan["ids_seen"] == {e["agent_id"] for e in full_events
                                if e["agent_id"] >= 0}
    # 参照実装 = build_agent_day の該当ブロックを全件 list で回したもの
    meta_by_id = {int(a["id"]): a for a in agents if "id" in a}
    ids = sorted(meta_by_id)
    last_cash = {a: float(meta_by_id.get(a, {}).get("money") or 0.0) for a in ids}
    last_acct = {a: 0.0 for a in ids}
    ref: dict = {}
    for e in sorted(full_events, key=lambda x: x["step"]):
        aid, p = e["agent_id"], e["payload"]
        if aid is None or aid < 0:
            continue
        if "balance" in p and isinstance(p.get("balance"), (int, float)):
            last_cash[aid] = float(p["balance"])
        if "account" in p and isinstance(p.get("account"), (int, float)):
            last_acct[aid] = float(p["account"])
        if aid in last_cash:
            ref.setdefault(aid, {})[mod._day(e["step"], SPD)] = (
                last_cash[aid] + last_acct.get(aid, 0.0))
    assert scan["wealth_day"] == ref


def test_analyze_rumor_contamination_equivalence(run_dir, monkeypatch):
    import analyze_rumor_contamination as mod
    narrow = mod.analyze(str(run_dir))
    narrow.pop("run_dir", None)
    monkeypatch.setattr(mod, "_want_kinds", lambda: None)
    wide = mod.analyze(str(run_dir))
    wide.pop("run_dir", None)
    assert narrow == wide


def test_audit_uncertainty_equivalence(run_dir, monkeypatch, tmp_path):
    """LUCK_KINDS 絞り読み + scan_presence == 全 kind 読み(逐語温存経路)。"""
    import audit_uncertainty as mod
    narrow = mod.analyze(str(run_dir), str(tmp_path / "a"))
    narrow.pop("paths", None)
    events_all = mod.load_events(str(run_dir), kinds=())
    wide = mod.audit(events_all, mod.load_agents(str(run_dir)), SPD)
    wide["run"] = narrow["run"]
    assert narrow == wide


def test_audit_scan_presence_matches_full_scan(run_dir, full_events):
    import audit_uncertainty as mod
    pres = mod.scan_presence(str(run_dir), SPD)
    ev = mod.load_events(str(run_dir), kinds=())
    assert pres["n_events"] == len(ev)
    assert pres["n_agent_events"] == sum(1 for e in ev if e["agent"] >= 0)
    assert pres["days"] == sorted({e["step"] // SPD for e in ev})
    assert (mod._active_filtered(pres["active_by_day"], None)
            == dict(mod._active_residents_by_day(ev, None, SPD)))
    residents = {0, 2, 4}
    assert (mod._active_filtered(pres["active_by_day"], residents)
            == dict(mod._active_residents_by_day(ev, residents, SPD)))


def test_analyze_luck_equivalence(run_dir, monkeypatch, tmp_path):
    import analyze_luck as mod
    import audit_uncertainty as au
    from society.observer import measure as meas
    narrow = mod.analyze(str(run_dir), str(tmp_path / "l"))
    narrow.pop("paths", None)
    events_all = au.load_events(str(run_dir), 10, kinds=())
    agents = au.load_agents(str(run_dir))
    traits, names, tsource = meas.load_traits(str(run_dir))
    wide = mod.decompose(events_all, agents, traits, SPD)
    wide["run"] = narrow["run"]
    wide["traits_source"] = tsource
    wide["traits_available"] = names
    assert narrow == wide


def test_analyze_firing_load_events_equivalence(run_dir):
    """絞り読みローダ == 「全列読み → Python で捨てる」旧実装(逐語の参照実装)。"""
    import analyze_firing as mod

    def _legacy(rd):
        table = pq.read_table(Path(rd) / "l1_events.parquet",
                              columns=["step", "sim_min", "agent_id", "kind",
                                       "x", "y", "payload"])
        cols = {n: table.column(n).to_pylist() for n in table.column_names}
        cog, causes, watch = [], {}, []
        for i, kind in enumerate(cols["kind"]):
            if kind not in mod.WANTED:
                continue
            blob = cols["payload"][i]
            rec = json.loads(blob) if blob else {}
            row = {"step": int(cols["step"][i]),
                   "sim_min": int(cols["sim_min"][i]),
                   "agent_id": int(cols["agent_id"][i]), "kind": kind,
                   "x": float(cols["x"][i]), "y": float(cols["y"][i]), "p": rec}
            if kind in mod.COG_KINDS:
                cog.append(row)
            elif kind == "watch_spec":
                watch.append(row)
            else:
                causes.setdefault((row["step"], kind), []).append(row)
        cog.sort(key=lambda r: (r["sim_min"], r["agent_id"], r["kind"]))
        return {"cog": cog, "causes": causes, "watch": watch}

    assert mod.load_events(run_dir) == _legacy(run_dir)


def test_judge_equivalence(run_dir):
    """ドシエ + 主指標(agent_features)が絞り読みでも全件読みでも同じ。"""
    import judge as mod
    agents = m.load_agents(str(run_dir))
    narrow = mod._load_events(str(run_dir), agents)
    wide = _all_events(run_dir)
    assert (mod.build_dossiers(narrow, agents)
            == mod.build_dossiers(wide, agents))
    assert m.agent_features(narrow, agents, None) == m.agent_features(wide, agents, None)


def test_judge_does_not_narrow_without_roster(run_dir):
    """名簿が空のときは絞らない(agent_features の行集合が変わる罠を踏まない)。"""
    import judge as mod
    assert mod._load_events(str(run_dir), []) == _all_events(run_dir)
    assert mod._load_events(str(run_dir), None) == _all_events(run_dir)


def test_panel_stats_r2_equivalence(run_dir):
    import panel_stats as mod
    agents = m.load_agents(str(run_dir))
    traits, names, _s = m.load_traits(str(run_dir), agents)
    wide = m.r2_traits(m.agent_features(_all_events(run_dir), agents, traits),
                       names).get("external", {}).get("r2")
    assert mod._r2_yext(str(run_dir)) == wide


# --------------------------------------------------------------------------- #
# E'. 同値テストの**感度**そのものを固定する(盲点を増やさないための番人)
#
# D の同値テストは「WANT_KINDS を間違えたら落ちる」ときにだけ意味がある。
# そこで WANT_KINDS から **1 種ずつ抜いて出力が動くか**を全数で確かめ、
# 動かない kind(= その解析ではこの合成ランで観測できない)を明示的に台帳化する。
# 台帳より **増えたら失敗**(= 新しい盲点ができたら気づく)。減るぶんは歓迎。
#
# 台帳に載る kind も、A の AST ガード(リテラル ⊆ WANT_KINDS)では守られている。
# --------------------------------------------------------------------------- #
_BLIND = {
    # 造語の窓帰属(coin_step)は、同じ item が伝播にも現れる限り
    # `item_cascades` の窓割り当てを変えない。
    "analyze_communities": {"label_coin", "vocab_coin"},
    # 接触グラフは 7 種の和集合。合成ランでは全ペアが他の種でも繋がるので、
    # 1 種抜いても接触集合が変わらない(実ランでは疎なので効く)。
    "analyze_imitation": {"chance_event", "dm", "hear", "joint_activity", "speak"},
    "analyze_structure": {"dm"},
    # 在館カテゴリは activity 優先。合成ランの dwell は activity 付きなので
    # 宿泊集合まで到達しない。
    "build_panel": {"lodging_checkin"},
    # 創発名台帳(sim_index)は、その名が発話本文の固有名詞候補として現れて
    # 初めて分類に効く。合成ランの本文には店名/制度名を埋めていない。
    "detect_emergence": {"group_found", "group_join", "institution",
                         "institution_rule", "venture_close", "venture_fulltime",
                         "venture_open", "venture_permit"},
}


def _sensitivity(monkeypatch, mod, loader_attr, want, call, adapt=None):
    """WANT_KINDS から 1 種ずつ抜いて出力が変わらなかった kind の集合を返す。"""
    def _sub(kinds):
        rows = [e for e in _ALL_CACHE if e["kind"] in kinds]
        return adapt(rows) if adapt else rows

    monkeypatch.setattr(mod, loader_attr, lambda rd, *a, _k=set(want), **kw: _sub(_k))
    base = key_of(call())
    dead = set()
    for k in sorted(want):
        sub = set(want) - {k}
        monkeypatch.setattr(mod, loader_attr,
                            lambda rd, *a, _k=sub, **kw: _sub(_k))
        try:
            if key_of(call()) == base:
                dead.add(k)
        except Exception:                      # 落ちる = 差が出た(=感度あり)
            pass
    return dead


def key_of(x) -> str:
    return json.dumps(x, ensure_ascii=False, sort_keys=True, default=repr)


_ALL_CACHE: list = []


def _agent_shape(rows):
    """自前ローダ({..., "agent": ...})を使うスクリプト向けにキー名を合わせる。"""
    return [{"step": e["step"], "sim_min": e["sim_min"], "agent": e["agent_id"],
             "kind": e["kind"], "payload": e["payload"]} for e in rows]


@pytest.mark.parametrize("name", sorted([
    "analyze_structure", "analyze_weak_ties", "analyze_communities",
    "analyze_imitation", "analyze_worldview", "observe", "observe_flows",
    "commercial_report", "build_panel", "analyze_founders", "trace_spark",
    "detect_emergence"]))
def test_equivalence_tests_are_sensitive_to_each_want_kind(
        run_dir, full_events, monkeypatch, tmp_path, name):
    _ALL_CACHE[:] = full_events
    mod = __import__(name)
    loader = "load_events" if hasattr(mod, "load_events") else "_load_events"
    want = _want_of(name)
    adapt = _agent_shape if name in ("analyze_founders", "trace_spark",
                                     "detect_emergence") else None
    seq = {"i": 0}

    def _out(pre):
        seq["i"] += 1
        return str(tmp_path / f"{pre}{seq['i']}")

    calls = {
        "analyze_structure": lambda: mod.analyze(str(run_dir)),
        "analyze_weak_ties": lambda: mod.analyze(str(run_dir)),
        "analyze_communities": lambda: mod.analyze(str(run_dir), window_days=1),
        "analyze_imitation": lambda: mod.analyze(str(run_dir)),
        "analyze_worldview": lambda: mod.analyze(str(run_dir), _out("wv") + ".md"),
        "observe": lambda: _artifacts(Path(mod.observe(str(run_dir), _out("ob")))),
        "observe_flows": lambda: _artifacts(Path(mod.observe(str(run_dir),
                                                             _out("of")))),
        "commercial_report": lambda: mod.report(str(run_dir), _out("cr") + ".md"),
        "build_panel": lambda: _artifacts(Path(
            mod.build(str(run_dir), out_dir=_out("bp"))["out_dir"])),
        "analyze_founders": lambda: mod.analyze(str(run_dir), _out("af")),
        "trace_spark": lambda: mod.trace(str(run_dir), _out("ts")),
        "detect_emergence": lambda: mod.analyze(str(run_dir)),
    }
    dead = _sensitivity(monkeypatch, mod, loader, want, calls[name], adapt)
    extra = dead - _BLIND.get(name, set())
    assert not extra, (
        f"{name}: 同値テストの新しい盲点 {sorted(extra)}"
        f"(この kind を落としても出力が変わらない = 取りこぼしを検出できない)")


# --------------------------------------------------------------------------- #
# E. 絞り読みが「load_events の部分列」であることの再確認(l1_stream の約束)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["analyze_structure", "analyze_weak_ties",
                                  "analyze_imitation", "analyze_worldview",
                                  "observe", "observe_flows",
                                  "commercial_report", "build_panel"])
def test_narrow_loader_is_exact_subsequence(run_dir, full_events, name):
    mod = __import__(name)
    loader = getattr(mod, "load_events", None) or mod._load_events
    want = _want_of(name)
    assert loader(str(run_dir)) == [e for e in full_events if e["kind"] in want]
