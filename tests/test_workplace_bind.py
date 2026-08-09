"""職場束ね直し(work.bind_workplace。既定 OFF)のテスト(第49バッチ)。

設計: docs/research/l2-work-reality.md §4「残課題(coverage)」。R1 の鉄則を継承:
- OFF(既定): 直接ラン L1 が純粋既定と完全一致(バイト一致)・pool 経路も work_node 付与状況/
  イベント列とも既定と一致・workplace_bound 0 件。
- ON: (a) work_node の付かない L2/L3(occupation が _WORK_CAT 非該当)が台帳 workplace_poi へ束なる。
      (b) coverage が実際に上がる(n_unbound_after < n_unbound_before・workplace_bound 1 件)。
      (c) 決定論=同 seed 2 ラン + hydrate 再入(build_pool_agent 再実行)で同一 work_node。
      (d) run.seed を変えても束ね(新規束ね分)は不変(hashlib 純関数=RngHub 非依存)。
      (e) serve 帰属が増える(束ねで生まれた勤務中スタッフに接客が帰属)。
検証は mock のみ(実LLM 禁止・≤24 step)。乱数不使用・追加 LLM 呼ゼロ。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "src"))

import build_persona_pool as bpp                       # noqa: E402
from society import work as work_mod                   # noqa: E402
from society.agents.persona import _WORK_CAT           # noqa: E402
from society.config import load_config                 # noqa: E402
from society.engine import scheduler                    # noqa: E402
from society.engine.simulation import Simulation       # noqa: E402
from society.observer.schema import EVENT_KINDS, Event  # noqa: E402


@pytest.fixture(scope="module")
def small_pool(tmp_path_factory):
    """P5 生成器で小プールを tmp に生成(実プール 736MB は触らない)。"""
    out = tmp_path_factory.mktemp("pool")
    orgs = json.loads((_ROOT / "data" / "organizations_shibuya_wide11k.json")
                      .read_text(encoding="utf-8"))
    pop = json.loads((_ROOT / "data" / "shibuya_population.json")
                     .read_text(encoding="utf-8"))
    bpp.build_pool(out, seed=42, fraction=0.001, orgs=orgs, pop=pop,
                   total_target=1_000_000)
    return out


def _pool_sim(name, pool_dir, out_dir, n_steps=1, cap=400, **ov):
    dot = ["run.seed=42", "run.n_agents=10", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock",
           "pool.enabled=true", f"pool.dir={pool_dir}", f"pool.present_cap={cap}"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=out_dir / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


# ------------------------------------------------------ (schema) 種別登録
def test_schema_registered():
    assert "workplace_bound" in EVENT_KINDS


# ------------------------------------------------------ OFF: 直接ラン L1 が純粋既定と一致
def test_off_direct_run_matches_pure_default(tmp_path):
    """bind_workplace は pool 専用=直接ランでは明示 OFF が純粋既定と L1 完全一致(バイト一致)。"""
    def mk(name, **ov):
        dot = ["run.seed=42", "run.n_agents=25", "run.n_steps=24", f"run.name={name}"]
        dot += [f"{k}={v}" for k, v in ov.items()]
        return Simulation(load_config(dot), out_dir=tmp_path / name)

    pure = mk("pure"); pure.run()
    off = mk("off", **{"work.bind_workplace.enabled": "false"}); off.run()
    assert _l1(pure) == _l1(off)
    assert not [e for e in off.logger.events if e.kind == "workplace_bound"]


# ------------------------------------------------------ OFF: pool 経路も既定と完全一致
def test_off_pool_matches_default(small_pool, tmp_path):
    """pool ON + bind 明示 OFF が pool ON(bind キー無し)と L1 完全一致=work_node 付与状況も不変。"""
    base = _pool_sim("base", small_pool, tmp_path, n_steps=6)
    base.run()
    off = _pool_sim("off", small_pool, tmp_path, n_steps=6,
                    **{"work.bind_workplace.enabled": "false"})
    off.run()
    assert _l1(base) == _l1(off)
    assert off._pool is not None
    assert not [e for e in off.logger.events if e.kind == "workplace_bound"]
    # work_node 付与状況(pid→bool)も完全一致
    wa = {a.pool_pid: bool(a.work_node) for a in base.agents}
    wb = {a.pool_pid: bool(a.work_node) for a in off.agents}
    assert wa == wb


# ------------------------------------------------------ ON: coverage が実際に上がる
def test_on_coverage_rises(small_pool, tmp_path):
    """ON で work_node の付かない L2/L3 が束なり coverage が上がる(after < before・event 1 件)。"""
    off = _pool_sim("cov_off", small_pool, tmp_path)
    on = _pool_sim("cov_on", small_pool, tmp_path,
                   **{"work.bind_workplace.enabled": "true"})
    on.run()
    stat = on._workbind_stat
    assert stat is not None
    assert stat["n_unbound_after"] < stat["n_unbound_before"], stat
    assert stat["n_bound"] > 0
    assert stat["n_unbound_before"] - stat["n_unbound_after"] == stat["n_bound"]
    # 起動時 1 件の世界イベント(agent_id=-1)
    wb = [e for e in on.logger.events if e.kind == "workplace_bound"]
    assert len(wb) == 1 and wb[0].agent_id == -1 and wb[0].step == 0
    assert wb[0].payload == stat
    # 束ねで present L2 の work_node 保有が実数として増える
    def bound_l2(sim):
        return sum(1 for a in sim.agents
                   if str(getattr(a, "pool_pid", "")).startswith("L2") and a.work_node)
    assert bound_l2(on) > bound_l2(off)
    # 束ね先は必ず現行地図のノード(経路探索の安全)
    for a in on.agents:
        if a.work_node:
            assert a.work_node in on.city.graph


# ------------------------------------------------------ ON: 決定論(同 seed 2 ラン一致)
def test_on_deterministic_same_seed(small_pool, tmp_path):
    """同 seed 2 ランで L1 完全一致 + 全 present の work_node 一致(決定論・乱数不使用)。"""
    a = _pool_sim("det_a", small_pool, tmp_path, n_steps=150,
                  **{"work.bind_workplace.enabled": "true"})
    a.run()
    b = _pool_sim("det_b", small_pool, tmp_path, n_steps=150,
                  **{"work.bind_workplace.enabled": "true"})
    b.run()
    assert _l1(a) == _l1(b)
    assert {x.pool_pid: x.work_node for x in a.agents} \
        == {x.pool_pid: x.work_node for x in b.agents}


# ------------------------------------------------------ ON: hydrate 再入で同一 work_node
def test_on_rebuild_same_work_node(small_pool, tmp_path):
    """build_pool_agent を同一 pid で再実行(=日境界ローテーション/hydrate 再入)しても同一 work_node。"""
    sim = _pool_sim("rebuild", small_pool, tmp_path,
                    **{"work.bind_workplace.enabled": "true"})
    bound = [a for a in sim.agents if a.work_node][:20]
    assert bound
    for a in bound:
        rec = sim._pool.get(a.pool_pid)
        again = sim.build_pool_agent(a.pool_pid, rec)
        assert again.work_node == a.work_node
        assert again.work_start_min == a.work_start_min


# ------------------------------------------------------ ON: run.seed を変えても束ね不変
def test_on_bind_seed_independent(small_pool, tmp_path):
    """新規に束ねた個体(occupation が _WORK_CAT 非該当)の work_node は run.seed に依らず不変。"""
    a = _pool_sim("si_a", small_pool, tmp_path,
                  **{"work.bind_workplace.enabled": "true"})
    b = _pool_sim("si_b", small_pool, tmp_path,
                  **{"work.bind_workplace.enabled": "true", "run.seed": "999"})
    da = {x.pool_pid: (x.work_node, x.occupation) for x in a.agents}
    db = {x.pool_pid: x.work_node for x in b.agents}
    common = [p for p in (set(da) & set(db)) if da[p][1] not in _WORK_CAT and da[p][0]]
    assert common, "新規束ねの共通 present pid が見つからない"
    for p in common:
        assert da[p][0] == db[p], f"束ねが run.seed 依存: {p}"


# ------------------------------------------------------ ON: serve 帰属が増える
def test_on_bind_increases_serve(small_pool, tmp_path):
    """束ねで生まれた勤務中スタッフに接客(serve)が帰属する(束ね無しなら不在=unstaffed)。"""
    sim = _pool_sim("serve", small_pool, tmp_path,
                    **{"work.bind_workplace.enabled": "true",
                       "work.service.enabled": "true"})
    # 束ねで food カテゴリの職場に付いた勤務中スタッフを1人選ぶ(取り違え排除に他は非スタッフ化)
    staff = None
    for a in sim.agents:
        if not a.work_node:
            continue
        if any(p.get("cat") == "food" for p in sim.city.pois_at_node(a.work_node)):
            staff = a
            break
    assert staff is not None, "food 職場に束ねたスタッフが居ない"
    for a in sim.agents:
        a.work_start_min = -1
    node = staff.work_node
    x, y = sim.city.node_xy(node)
    staff.node = node
    staff.x, staff.y = x, y
    staff.work_start_min, staff.work_end_min = 0, 1440   # 常時勤務窓
    customer = next(a for a in sim.agents if a.id != staff.id)
    customer.node = node
    customer.x, customer.y = x, y

    def fire():
        since = len(sim.logger.events)
        sim.logger.log(Event(step=5, sim_min=700, agent_id=customer.id, kind="spend",
                             x=x, y=y, payload={"cat": "food"}))
        scheduler._phase_work_service(sim, 5, 700, since)
        return [e for e in sim.logger.events[since:] if e.kind == "serve"]

    staffed = fire()
    assert len(staffed) == 1 and staffed[0].agent_id == staff.id
    assert staffed[0].payload["customer"] == customer.id
    # 束ね無し(work_node を外す)なら同じ消費が unstaffed=帰属が消える(=束ねが serve を増やした)
    staff.work_node = ""
    unst = fire()
    assert len(unst) == 1 and unst[0].agent_id == -1
    assert unst[0].payload.get("unstaffed") is True


# ------------------------------------------------------ 束ね関数の単体(POI 決定論マッチ)
def test_bind_fallback_deterministic_pure_function():
    """台帳ノードが地図に無い時の POI 決定論マッチが (seed, key) の純関数=乱数ゼロ。"""
    u1 = work_mod._stable_uniform(20260722, "food\x1fL2_00000001")
    u2 = work_mod._stable_uniform(20260722, "food\x1fL2_00000001")
    assert u1 == u2 and 0.0 <= u1 < 1.0
    assert work_mod._stable_uniform(20260722, "food\x1fL2_00000002") != u1


# =========================================================================== #
# 第100バッチ P3b 前提: 被覆の穴を実際に塞ぐ(coverage を正直に測る)
# =========================================================================== #
_BOUND = "work.bind_workplace.enabled"
_REBIND = "work.bind_workplace.rebind_bound"


def _is_bound(a) -> bool:
    """スタッフ資格の必要条件(work_node があり勤務窓が開いている)。共在は別条件。"""
    return bool(getattr(a, "work_node", "")) and int(getattr(a, "work_start_min", -1)) >= 0


# ------------------------------------------------------ ON: 母数が「勤務地を持つべき層」を覆う
def test_on_eligibility_covers_role_bearing_layers(small_pool, tmp_path):
    """束ね母数(bind_eligible)が org_id 保有者だけでなく role を持つ L5・就業 occupation も覆う。

    旧実装は org_id 保有者だけを数えていたので、地図に対応 POI カテゴリの無い L5(駅員/警察官/
    議員 等)が統計の**外**に落ち、n_unbound_after=0 が実態より良く見えていた。"""
    on = _pool_sim("elig", small_pool, tmp_path, **{_BOUND: "true"})
    stat, det = on._workbind_stat, on._workbind_detail
    # 母数 = 束ね手段/不能理由タグの総数(全 eligible が必ず 1 つのタグを受ける)
    assert sum(det["how"].values()) == stat["n_total"] > 0
    # L5(role を持つ duty/議員)が母数に入っている
    assert det["by_layer"].get("L5", {}).get("n", 0) > 0
    # 明示 org_id を持つ層(L2)は 1 体も未束ねで残らない
    n_org = sum(1 for a in on.agents
                if (on._pool.get(a.pool_pid) or {}).get("org_id"))
    n_org_unbound = sum(1 for a in on.agents
                        if (on._pool.get(a.pool_pid) or {}).get("org_id") and not _is_bound(a))
    assert n_org > 0 and n_org_unbound == 0, f"org 所属なのに未束ね: {n_org_unbound}/{n_org}"


# ------------------------------------------------------ ON: 束ねられないものは「理由つきで数える」
def test_on_unresolvable_is_counted_not_hidden(small_pool, tmp_path):
    """地図に対応 POI カテゴリが存在しない層は束ねず、n_unbound_after + 理由タグで開示する。

    推測で職業→カテゴリ写像を作らない(_WORK_CAT / occ_cat / 台帳 cat のどれにも無い職業は
    no_category)。開示の在り処は _workbind_detail(L1 の workplace_bound は 4 列固定の契約)。"""
    on = _pool_sim("unres", small_pool, tmp_path, **{_BOUND: "true"})
    stat, det = on._workbind_stat, on._workbind_detail
    reasons = {k: v for k, v in det["how"].items()
               if k in ("no_category", "no_poi_in_map", "no_ledger_node")}
    assert sum(reasons.values()) == stat["n_unbound_after"], (reasons, stat)
    assert stat["n_unbound_after"] > 0, "不能ゼロは母数が狭すぎる疑い(L5 が抜けている)"
    # 不能は「地図に職場カテゴリが無い役割」に限られる(L2/L3 の org 所属は 0)
    assert det["by_layer"].get("L2", {}).get("unbound_after", 0) == 0
    assert det["unresolved_occ"], "不能理由の職業内訳が空"
    # 統計の恒等式(束ね前後の差 = 新規束ね数)は母数を広げても壊れない
    assert stat["n_unbound_before"] - stat["n_unbound_after"] == stat["n_bound"]


# ------------------------------------------------------ ON: 束ね先は「客が実際に来る場所」
def test_on_bound_node_actually_carries_that_workplace(small_pool, tmp_path):
    """束ね先ノードは現行地図でその業種の POI を実際に持つ(=客の消費と共在しうる)。

    台帳 organizations_shibuya_wide11k は「産業×規模帯→建物」の分布で置いた合成値なので、
    workplace_poi.node は地図の POI 実体と食い違うものが多い(実測 wide_v7: food は 1,650社中
    401社しか同カテゴリ POI のあるノードに居ない)。食い違ったまま束ねると『客が絶対に来ない
    場所に店員が立つ』= serve が構造的に永久不在になる。"""
    on = _pool_sim("realpoi", small_pool, tmp_path, **{_BOUND: "true", _REBIND: "true"})
    city, bcfg, book = on.city, on._workbind_cfg, on._workbind_book
    checked = 0
    for a in sorted(on.agents, key=lambda x: x.id):
        rec = on._pool.get(a.pool_pid) or {}
        if not work_mod.bind_eligible(rec, bcfg) or not _is_bound(a):
            continue
        cat = work_mod._cat_for(rec, book.get(str(rec.get("org_id") or "")), bcfg)
        if not cat:
            continue
        assert work_mod._node_is_workplace_of(city, a.work_node, cat), \
            f"{a.pool_pid}: work_node={a.work_node} に {cat} の POI が無い"
        checked += 1
    assert checked > 0, "検査対象の束ね個体が居ない"


# ------------------------------------------------------ ON: 地図語彙の食い違いは既存表で吸収
def test_on_category_fallback_reuses_day_plan_table(small_pool, tmp_path):
    """POI カテゴリの解決は day_plan.MAP_FALLBACK_CATS(既存表)を再利用する=新表を作らない。"""
    from society.cognition.day_plan import MAP_FALLBACK_CATS
    assert MAP_FALLBACK_CATS.get("education") == ("school",)
    on = _pool_sim("catfb", small_pool, tmp_path, **{_BOUND: "true"})
    city = on.city
    # education が空の地図でも school 側から拾える(逆に education がある地図では素通り)
    got = work_mod._pois_for_cat(city, "education")
    assert got or not (city.pois_by_cat("education") or city.pois_by_cat("school"))
    # occupation → カテゴリは persona._WORK_CAT(既存表)の再利用
    assert work_mod._occ_cat_table() is _WORK_CAT


# ------------------------------------------------------ ON: 冪等(2 回束ねても 1 回と同じ)
def test_on_bind_is_idempotent(small_pool, tmp_path):
    """rebind_bound=true で同じ個体を 2 回束ねても work_node/勤務窓が動かない(resume 再入の安全)。

    work_node は agents pickle に載って resume を跨ぐので、再入で二重束ねのドリフトが起きると
    resume != straight になる。束ねが (pool_pid, 固定属性) の純関数であることの機械固定。"""
    sim = _pool_sim("idem", small_pool, tmp_path, **{_BOUND: "true", _REBIND: "true"})
    n = 0
    for a in sorted(sim.agents, key=lambda x: x.id):
        rec = sim._pool.get(a.pool_pid) or {}
        if not work_mod.bind_eligible(rec, sim._workbind_cfg):
            continue
        before = (a.work_node, a.work_building, a.work_floor,
                  a.work_start_min, a.work_end_min)
        for _ in range(2):
            work_mod.bind_workplace(a, rec, sim.city, sim._workbind_book,
                                    sim._workbind_cfg)
        after = (a.work_node, a.work_building, a.work_floor,
                 a.work_start_min, a.work_end_min)
        assert before == after, f"{a.pool_pid}: 再束ねでドリフト {before} -> {after}"
        n += 1
    assert n > 0


# ------------------------------------------------------ ON: 束ねは乱数を 1 draw も引かない
def test_on_bind_draws_no_randomness(small_pool, tmp_path):
    """束ね関数は RngHub を触らない(決定論=hashlib 純関数)。識別子を AST で機械固定する。

    文字列/コメントは除いて **実行される識別子だけ**を見る(docstring に「乱数ゼロ」と書いてある
    のを誤検知しない)。traces.py の propagation 不在固定と同型の AST 固定。"""
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(work_mod))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names |= {a.name.split(".")[0] for n in ast.walk(tree)
              if isinstance(n, ast.Import) for a in n.names}
    names |= {str(n.module or "").split(".")[0] for n in ast.walk(tree)
              if isinstance(n, ast.ImportFrom)}
    for word in ("rng", "random", "RngHub", "hub", "stream", "numpy", "np"):
        assert word not in names, f"work.py が乱数側の識別子 {word!r} を使っている"
    sim = _pool_sim("norng", small_pool, tmp_path, **{_BOUND: "true", _REBIND: "true"})
    a = next(x for x in sorted(sim.agents, key=lambda x: x.id) if x.work_node)
    rec = sim._pool.get(a.pool_pid) or {}
    before = a.work_node
    work_mod.bind_workplace(a, rec, sim.city, sim._workbind_book, sim._workbind_cfg)
    assert a.work_node == before


# ------------------------------------------------------ ON: coverage が実数として上がる
def test_on_raises_staff_capable_population(small_pool, tmp_path):
    """ON でスタッフ資格(work_node + 勤務窓)を満たす在場者が実数として増える。

    ★共在(勤務窓に work_node へ在場)は別条件で、本テストは緩めない。実測の unstaffed 率は
    在場密度に支配される(docs 参照)ので、ここでは『資格者が増える』ことだけを固定する。"""
    off = _pool_sim("cap_off", small_pool, tmp_path)
    on = _pool_sim("cap_on", small_pool, tmp_path, **{_BOUND: "true"})
    n_off = sum(1 for a in off.agents if _is_bound(a))
    n_on = sum(1 for a in on.agents if _is_bound(a))
    assert n_on > n_off, (n_off, n_on)
