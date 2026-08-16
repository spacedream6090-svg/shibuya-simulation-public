"""金銭・債権の恒久台帳(vital)と回転境界の保存則(レーン R1 A2)。

正典: src/society/world/pool.py の ``DormantStore`` / ``vital_of`` / ``overlay_vital``。

背景(裏取り済みの破れ):
  ``DormantStore.save`` は ``dormant_cap`` 超過で LRU により退避状態を**丸ごと**捨てていた。
  捨てられた個体が再来街すると ``pop`` が None を返し、``build_agent`` が**初期所持金で
  新規鋳造**する = 退場時の残高・預金・家賃債権・未清算の勤務実績が**真に消え、同時に
  無から現金が湧く**。しかも L1 / finance / provenance のどの保存チャネルにも痕跡が
  残らないので、総マネー保存の検査でも検出できなかった。

検証:
  (1) vital の欄が「金銭・債権・人口会計」の全数列挙とちょうど一致する。
  (2) LRU は rich だけを捨て、vital は**絶対に捨てない**。
  (3) rich が捨てられた個体の再来街で**所持金・口座・家賃債権・未清算勤務が連続**する
      —— 旧挙動(vital を当てない)では初期残高へリセットされることを**反証**で固定。
  (4) 回転境界で Σ(在場 money+account) + Σ(dormant vital) が**1 円も動かない**
      (新規鋳造 = 一度も退避されたことのない初来街者ぶんを除く)。
  (5) finance サイドカーに ``hh_dormant`` 列が出る(既存 15 列は不変)。
  (6) 旧 pool サイドカー(vital 無し)は rich から作り直して受ける。
  (7) cap=0(出荷時の既定)では vital 経路を 1 度も通らない = 完全にバイト一致。
"""
from __future__ import annotations

import gzip
import json
import pickle
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "src"))

import build_persona_pool as bpp                       # noqa: E402
from society.agents.agent import Agent                 # noqa: E402
from society.agents.memory import MemoryStore          # noqa: E402
from society.config import load_config                 # noqa: E402
from society import economy_sfc as sfc_mod             # noqa: E402
from society.engine.simulation import Simulation       # noqa: E402
from society.rng import RngHub                         # noqa: E402
from society.world import pool as pool_mod             # noqa: E402
from society.world.presence import present_for_day     # noqa: E402


@pytest.fixture(scope="module")
def small_pool(tmp_path_factory):
    out = tmp_path_factory.mktemp("pool")
    orgs = json.loads((_ROOT / "data" / "organizations_shibuya_wide11k.json")
                      .read_text(encoding="utf-8"))
    pop = json.loads((_ROOT / "data" / "shibuya_population.json")
                     .read_text(encoding="utf-8"))
    bpp.build_pool(out, seed=42, fraction=0.001, orgs=orgs, pop=pop,
                   total_target=1_000_000)
    return out


# ===================================================== (1) vital の全数列挙
def _rich_agent(aid=1, money=1234.0, account=99.0):
    a = Agent(id=aid, name=f"甲{aid}", age=30, occupation="会社員", persona="p",
              traits={}, states={}, mem=MemoryStore())
    a.money = money
    a.account = account
    a.rent_due = 4000.0            # 家賃債権(_ECON_FIELDS)
    a.arrears_days = 2
    a.work_days = 5                # 未清算の勤務日数(月給の元)
    a.wp_days = 3                  # WAGE の未清算勤務実績
    a.pop_stress = 0.42            # 人口会計(POP)
    a.pop_days = 17
    a.beliefs = ["リッチ層にだけ載る信念"]
    a.mem.record_contact(9, "乙", 1, "やあ")
    return a


def test_vital_of_is_exactly_the_money_claim_and_population_family():
    """vital = money / account / econ(_ECON_FIELDS)/ pop(POP)。それ以外は入らない。"""
    state = pool_mod.dehydrate(_rich_agent())
    vital = pool_mod.vital_of(state)
    assert set(vital) == {"money", "account", "econ", "pop"}
    assert vital["money"] == 1234.0 and vital["account"] == 99.0
    assert vital["econ"]["rent_due"] == 4000.0          # 家賃債権
    assert vital["econ"]["arrears_days"] == 2           # 滞納日数
    assert vital["econ"]["work_days"] == 5              # 未清算の勤務日数
    assert vital["econ"]["wp_days"] == 3                # 未払賃金の元
    assert vital["pop"] == {"pop_stress": 0.42, "pop_days": 17}
    # リッチ層の欄(記憶・信念・関係)は 1 つも混ざらない
    assert "beliefs" not in vital and "relations" not in vital and "episodes" not in vital


def test_vital_pop_names_are_a_subset_of_the_misc_field_table():
    """POP の欄は ``_MISC_FIELDS`` から**切り出す**(二重の真実源を作らない)。"""
    names = {f[0] for f in pool_mod._MISC_FIELDS}
    assert set(pool_mod._VITAL_POP_NAMES) <= names
    assert tuple(f[0] for f in pool_mod._POP_FIELDS) == pool_mod._VITAL_POP_NAMES


def test_vital_is_empty_when_every_mechanism_is_off():
    """口座 OFF / WAGE OFF / POP OFF の素の個体では econ / pop のキーが生えない。"""
    a = Agent(id=1, name="甲", age=30, occupation="学生", persona="p",
              traits={}, states={}, mem=MemoryStore())
    vital = pool_mod.vital_of(pool_mod.dehydrate(a))
    assert set(vital) == {"money", "account"}           # 0 円でもキーは作る(下のテスト)


def test_zero_money_still_gets_a_key_so_it_differs_from_no_ledger():
    """「0 円で街を出た」と「台帳が無い」を区別する(区別できないと再鋳造と同じになる)。"""
    a = Agent(id=1, name="甲", age=30, occupation="学生", persona="p",
              traits={}, states={}, mem=MemoryStore())
    a.money = 0.0
    vital = pool_mod.vital_of(pool_mod.dehydrate(a))
    assert vital["money"] == 0.0


# ===================================================== (2) LRU は rich だけを捨てる
def test_lru_evicts_rich_but_never_vital():
    store = pool_mod.DormantStore(cap=1)
    for i in range(3):
        store.save(f"p{i}", pool_mod.dehydrate(_rich_agent(i, money=100.0 * i)))
    assert len(store) == 1                              # rich は 1 件だけ
    assert len(store._vital) == 3                       # vital は全員残る
    for i in range(3):
        assert f"p{i}" in store                         # 「退避されている」は vital でも真
    assert store.peek_vital("p0")["money"] == 0.0
    assert store.peek_vital("p1")["money"] == 100.0


def test_pop_returns_none_for_evicted_rich_but_pop_vital_survives():
    """★反証つき: 旧挙動(rich だけ)では None = 再鋳造。vital 層が唯一の受け皿。"""
    store = pool_mod.DormantStore(cap=1)
    store.save("gone", pool_mod.dehydrate(_rich_agent(1, money=5555.0)))
    store.save("stay", pool_mod.dehydrate(_rich_agent(2, money=10.0)))
    assert store.pop("gone") is None                    # ← 旧挙動そのもの(記憶は失われる)
    vital = store.pop_vital("gone")
    assert vital is not None and vital["money"] == 5555.0
    assert store.pop_vital("gone") is None              # 消費は 1 回きり


def test_pop_merges_vital_onto_rich_without_changing_anything_when_healthy():
    """健全なら vital の上書きは恒等 = 退避辞書のオブジェクト同一性まで保つ。"""
    store = pool_mod.DormantStore(cap=0)
    state = pool_mod.dehydrate(_rich_agent(1, money=42.0))
    before = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
    store.save("p", state)
    got = store.pop("p")
    assert got is state                                  # 同じ dict がそのまま返る
    assert pickle.dumps(got, protocol=pickle.HIGHEST_PROTOCOL) == before


def test_pop_does_not_consume_vital_when_rich_is_missing():
    store = pool_mod.DormantStore(cap=1)
    store.save("a", pool_mod.dehydrate(_rich_agent(1, money=7.0)))
    store.save("b", pool_mod.dehydrate(_rich_agent(2, money=8.0)))
    assert store.pop("a") is None
    assert store.peek_vital("a")["money"] == 7.0         # ★pop が道連れにしていない


def test_overlay_vital_touches_only_the_ledger_family():
    """overlay は記憶・信念・関係・状態に 1 バイトも触らない(hydrate との違い)。"""
    fresh = Agent(id=3, name="丙", age=25, occupation="学生", persona="p",
                  traits={}, states={}, mem=MemoryStore())
    fresh.money = 1.0
    fresh.beliefs = ["再構築された信念"]
    fresh.mem.record_contact(5, "戊", 0, "顔なじみ")
    fresh.status = 0.3
    vital = pool_mod.vital_of(pool_mod.dehydrate(_rich_agent(1, money=8888.0)))
    pool_mod.overlay_vital(fresh, vital)
    assert fresh.money == 8888.0 and fresh.account == 99.0
    assert fresh.rent_due == 4000.0 and fresh.work_days == 5 and fresh.wp_days == 3
    assert fresh.pop_stress == 0.42 and fresh.pop_days == 17
    assert fresh.beliefs == ["再構築された信念"]          # 記憶側は無傷
    assert 5 in fresh.mem.relations
    assert fresh.status == 0.3


def test_vital_money_total_and_claims_total():
    store = pool_mod.DormantStore(cap=1)
    store.save("a", pool_mod.dehydrate(_rich_agent(1, money=100.0, account=10.0)))
    store.save("b", pool_mod.dehydrate(_rich_agent(2, money=200.0, account=20.0)))
    assert store.vital_money_total() == pytest.approx(330.0)
    assert store.vital_claims_total() == pytest.approx(8000.0)   # 家賃債権 4000 × 2


def test_rebuild_vital_reconstructs_from_rich_for_old_sidecars():
    store = pool_mod.DormantStore(cap=0)
    store._d["x"] = pool_mod.dehydrate(_rich_agent(1, money=321.0))   # vital 層を経ない
    assert store.peek_vital("x") is None
    store.rebuild_vital()
    assert store.peek_vital("x")["money"] == 321.0


# ===================================================== ON mock 統合
def _pool_cfg(name, pool_dir, n_steps, cap=400, **ov):
    dot = ["run.seed=42", "run.n_agents=10", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock", "pool.enabled=true",
           f"pool.dir={pool_dir}", f"pool.present_cap={cap}"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def _presence_sets(small_pool, cap=400):
    ps = pool_mod.PoolStore(small_pool)
    recs = ps.presence_records()
    hub = RngHub(42)
    return (set(present_for_day(recs, 0, cap, hub, 0 % 7)),
            set(present_for_day(recs, 1, cap, hub, 1 % 7)))


_MARK = {"money": 777.0, "account": 55.0,
         "econ": {"rent_due": 4000.0, "arrears_days": 2, "work_days": 5,
                  "wp_days": 3},
         "misc": {"pop_days": 17},
         "beliefs": ["__richonly__"]}


def _run_with_evicted_ledger(small_pool, tmp_path, name, *, apply_vital: bool):
    """全再来街候補に印つきの退避状態を仕込み、cap=1 で**ほぼ全員の rich を捨てさせる**。

    apply_vital=False は**旧挙動の反証** —— vital を当てない(= build_agent の初期残高)。
    """
    s0, s1 = _presence_sets(small_pool, cap=150)
    entrants = sorted(s1 - s0)
    assert entrants, "day0 不在 → day1 present の再来街候補が居ない(空回りの検査)"
    sim = Simulation(_pool_cfg(name, small_pool, 210, cap=150,
                               **{"pool.dormant_cap": 1}), out_dir=tmp_path / name)
    for pid in entrants:
        sim._dormant.save(pid, dict(_MARK, misc=dict(_MARK["misc"]),
                                    econ=dict(_MARK["econ"])))
    assert len(sim._dormant) == 1, "cap=1 の LRU が効いていない(前提が崩れた)"
    assert len(sim._dormant._vital) == len(entrants), "vital が LRU に巻き込まれている"
    if not apply_vital:                       # ★反証: 恒久台帳を当てない旧挙動
        sim._dormant.pop_vital = lambda pid: None
    sim.run()
    live = {a.pool_pid: a for a in sim.agents}
    moved = {e.agent_id for e in sim.logger.events
             if e.kind in ("spend", "wage", "chance_event", "windfall")}
    back = [live[pid] for pid in entrants
            if pid in live and live[pid].id not in moved]
    assert back, "取引をしなかった再来街者が 1 人も居ない(空回りの検査)"
    return sim, back


def test_money_and_claims_survive_the_lru_eviction(small_pool, tmp_path):
    """★A2 の検収本体: rich が LRU で捨てられても、財布と債権は連続する。"""
    sim, back = _run_with_evicted_ledger(small_pool, tmp_path, "vital_on",
                                         apply_vital=True)
    for a in back:
        assert a.money == 777.0, "所持金が連続していない(再鋳造されている)"
        assert a.account == 55.0
        assert a.rent_due == 4000.0            # 家賃債権
        assert a.arrears_days == 2             # 滞納日数(立退きの左辺)
        assert a.work_days == 5                # 未清算の勤務日数(月給の元)
        assert a.wp_days == 3                  # 未払賃金の元(WAGE)
        assert a.pop_days == 17                # 人口会計
        # ★リッチ層(記憶・信念)は失われている = 「記憶を失って街へ戻る」は設計どおり
        assert "__richonly__" not in a.beliefs


def test_old_behaviour_resets_the_wallet_on_reentry(small_pool, tmp_path):
    """★反証: 恒久台帳を当てない(= 修正前)と、再来街者の所持金は初期残高へ戻る。"""
    sim, back = _run_with_evicted_ledger(small_pool, tmp_path, "vital_off",
                                         apply_vital=False)
    reset = [a for a in back if a.money != 777.0]
    assert reset, "旧挙動でも所持金が保たれてしまう(反証が空回り)"
    assert all(a.rent_due == 0.0 for a in reset), "旧挙動では家賃債権も消えるはず"


def test_rotation_conserves_money_inside_a_full_run(small_pool, tmp_path):
    """★回転保存則(丸ごとのランの中で): 回転フェーズの**前後**で
    Σ(在場 money+account) + Σ(dormant vital) が 1 円も動かない。

    その step の給与・消費は回転の**後**に走るので、回転フェーズだけを内側から挟む
    (フェーズを差し替えて計測器を被せる = ランの進み方は 1 バイトも変えない)。
    新規鋳造(**一度も台帳を持ったことがない初来街者**)は正当な流入なので差し引く。
    dormant_cap を退場者数より小さくして、LRU 破棄を必ず踏ませる。
    """
    from society.engine import scheduler

    sim = Simulation(_pool_cfg("cons", small_pool, 300, cap=150,
                               **{"pool.dormant_cap": 3}), out_dir=tmp_path / "cons")

    def wallet(agents):
        return sum(float(a.money) + float(getattr(a, "account", 0.0) or 0.0)
                   for a in agents)

    drifts: list[float] = []
    rotations = 0
    orig = scheduler._phase_pool_rotation

    def measured(s, step, sim_min):
        nonlocal rotations
        day = sim_min // 1440
        if s._pool is None or day == int(getattr(s, "_pool_day", 0)):
            return orig(s, step, sim_min)                # 日境界でない = 素通り
        before = wallet(s.agents) + s._dormant.vital_money_total()
        had = set(s._dormant._vital) | set(s._dormant._d)
        ids = {a.id for a in s.agents}
        out = orig(s, step, sim_min)
        minted = wallet([a for a in s.agents
                         if a.id not in ids
                         and getattr(a, "pool_pid", None) not in had])
        after = wallet(s.agents) + s._dormant.vital_money_total()
        drifts.append(after - before - minted)
        rotations += 1
        return out

    scheduler._phase_pool_rotation = measured
    try:
        sim.run()
    finally:
        scheduler._phase_pool_rotation = orig
    assert rotations >= 2, "日境界を 2 回跨いでいない(空回りの検査)"
    assert max(abs(d) for d in drifts) < 1e-6, f"回転で金が湧いた/消えた: {drifts}"


def test_conservation_is_exact_at_the_rotation_phase(small_pool, tmp_path):
    """回転フェーズ**そのもの**(``_phase_pool_rotation`` の前後)で 1 円も動かない。

    その step の給与・消費を巻き込まないよう、回転フェーズだけを直接呼んで挟む。
    ★反証: 恒久台帳を当てないと、LRU 破棄された個体の再来街で差が立つ。
    """
    from society.engine import scheduler

    def measure(apply_vital: bool) -> float:
        sim = Simulation(_pool_cfg(f"exact{int(apply_vital)}", small_pool, 1, cap=150,
                                   **{"pool.dormant_cap": 2}),
                         out_dir=tmp_path / f"exact{int(apply_vital)}")
        if not apply_vital:
            sim._dormant.pop_vital = lambda pid: None

        def wallet():
            return sum(float(a.money) + float(getattr(a, "account", 0.0) or 0.0)
                       for a in sim.agents)

        worst = 0.0
        for day in (1, 2, 3):
            before = wallet() + sim._dormant.vital_money_total()
            had = set(sim._dormant._vital) | set(sim._dormant._d)
            ids = {a.id for a in sim.agents}
            scheduler._phase_pool_rotation(sim, day * 144, day * 1440)
            minted = sum(float(a.money) + float(getattr(a, "account", 0.0) or 0.0)
                         for a in sim.agents
                         if a.id not in ids
                         and getattr(a, "pool_pid", None) not in had)
            after = wallet() + sim._dormant.vital_money_total()
            worst = max(worst, abs(after - before - minted))
        return worst

    assert measure(True) < 1e-6, "回転そのもので金が湧いた/消えた"
    assert measure(False) > 1.0, "反証が空回り(旧挙動でも保存してしまう)"


def test_default_cap_never_overlays_anything(small_pool, tmp_path):
    """cap=0(出荷時の既定)では rich が 1 件も捨てられない = overlay を 1 度も通らない。

    = 出荷時の既定で走るランは A2 の前後で完全にバイト一致する、の機械的な根拠。
    """
    from society.engine import scheduler

    sim = Simulation(_pool_cfg("cap0", small_pool, 300, cap=150), out_dir=tmp_path / "cap0")
    assert sim._dormant.cap == 0
    overlays: list[str] = []
    orig = pool_mod.overlay_vital
    scheduler.pool_mod.overlay_vital = lambda a, v: (overlays.append(a.pool_pid),
                                                     orig(a, v))[1]
    try:
        sim.run()
    finally:
        scheduler.pool_mod.overlay_vital = orig
    assert sim._dormant._d, "退避が 1 件も無い(空回りの検査)"
    assert [e for e in sim.logger.events if e.kind == "presence_change"], \
        "日境界の回転が起きていない(空回りの検査)"
    assert overlays == [], f"cap=0 なのに恒久台帳のオーバレイが走った: {overlays}"


# ===================================================== finance サイドカー
def test_finance_sidecar_has_hh_dormant_after_the_existing_columns(small_pool, tmp_path):
    """``hh_dormant`` は**末尾に**足す(既存 15 列は順序含め不変)。値 = 退避中の現金+口座。"""
    import pyarrow.parquet as pq

    assert sfc_mod.COLUMNS[-1] == "hh_dormant"
    assert sfc_mod.COLUMNS[:-1][-1] == "k5_other"          # 既存の末尾は k5_other のまま
    sim = Simulation(_pool_cfg("fin", small_pool, 300, cap=150,
                               **{"economy.org_accounting.enabled": "true",
                                  "economy.org_accounting.sidecar": "true"}),
                     out_dir=tmp_path / "fin")
    sim.run()
    t = pq.read_table(Path(sim.out_dir) / "finance.parquet")
    assert tuple(t.column_names) == sfc_mod.COLUMNS
    vals = t.column("hh_dormant").to_pylist()
    assert any(v > 0.0 for v in vals), "退避中の家計残高が 1 円も出ていない(空回りの検査)"
    assert vals[-1] == pytest.approx(sim._dormant.vital_money_total(), abs=1e-3)


def test_dormant_total_is_zero_without_a_pool(tmp_path):
    """pool OFF のランでは常に 0.0(= 既存の finance 列は 1 バイトも変わらない)。"""
    sim = Simulation(load_config(["run.seed=42", "run.n_agents=6", "run.n_steps=2",
                                  "run.name=nopool", "model.backend=mock"]),
                     out_dir=tmp_path / "nopool")
    assert sim._dormant is None
    assert sfc_mod.dormant_total(sim) == 0.0
    assert sfc_mod.dormant_claims(sim) == 0.0


# ===================================================== サイドカーの往復
def test_pool_sidecar_carries_the_vital_ledger(small_pool, tmp_path):
    sim = Simulation(_pool_cfg("sc", small_pool, 220, cap=150,
                               **{"pool.dormant_cap": 2}), out_dir=tmp_path / "sc")
    sim.run()
    sim._save_pool_sidecar(220)
    blob = sim._read_pool_sidecar(220)
    assert "dormant_vital" in blob and blob["dormant_vital"], "vital が保存されていない"
    assert len(blob["dormant_vital"]) > len(blob["dormant"]), \
        "LRU で捨てた rich より vital のほうが多い、が成立していない(前提が崩れた)"


def test_resume_matches_straight_with_a_bounded_dormant_store(small_pool, tmp_path):
    """★分割走行(pool ON + dormant_cap>0)の L1 が一気通しと byte 一致する。

    恒久台帳(vital)と軽量参照(A1)を pool サイドカーで運べていないと、resume 側だけ
    「LRU で捨てられた個体が初期残高で再鋳造される」= L1 が食い違う。
    """
    import pyarrow.parquet as pq
    from society.engine import checkpoint, scheduler

    def rows(d):
        return pq.read_table(Path(d) / "l1_events.parquet").to_pylist()

    ov = {"pool.dormant_cap": 2}
    st = tmp_path / "st"
    Simulation(_pool_cfg("rst", small_pool, 300, cap=150, **ov), out_dir=st).run()

    rs = tmp_path / "rs"
    s1 = Simulation(_pool_cfg("rrs", small_pool, 150, cap=150,
                              **{"observer.checkpoint_every": 150}, **ov), out_dir=rs)
    for step in range(150):
        scheduler.run_step(s1, step)
    checkpoint.save(s1, 150, rs / "checkpoint" / "ckpt-000150.pkl.gz")
    s1._save_pool_sidecar(150)
    s1.logger.flush_segment()
    s2 = Simulation(_pool_cfg("rrs", small_pool, 300, cap=150,
                              **{"observer.checkpoint_every": 150}, **ov), out_dir=rs)
    s2.run(resume_from=rs)
    assert rows(st) == rows(rs), "有界ドーマントの resume が straight と byte 不一致"


def test_old_sidecar_without_vital_is_rebuilt_on_restore(small_pool, tmp_path):
    """★旧サイドカー互換: vital キーが無ければ rich から作り直す。"""
    sim = Simulation(_pool_cfg("oldv", small_pool, 220, cap=150), out_dir=tmp_path / "oldv")
    sim.run()
    blob = {"pool_day": int(sim._pool_day), "dormant": sim._dormant._d,
            "dormant_cap": sim._dormant.cap, "departed": {}}     # ← vital 無し
    path = sim._pool_sidecar_path(220)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)

    fresh = Simulation(_pool_cfg("oldv2", small_pool, 220, cap=150),
                       out_dir=tmp_path / "oldv2")
    fresh.out_dir = sim.out_dir
    fresh.__dict__.pop("_pool_sidecar_blob", None)
    fresh._restore_pool_resume(220)
    assert fresh._dormant._vital, "旧サイドカーから vital が作り直されていない"
    assert set(fresh._dormant._vital) == set(fresh._dormant._d)
