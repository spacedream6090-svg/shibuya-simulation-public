"""存在の内生化 POP(恒久転出・転入=L4 定着昇格・出生)のテスト。

正典: docs/plans/population-endogenization-plan.md(§1 全て「域内状態への応答」・§3 POP-1/2/3)/
      docs/research/population-endogenization.md(§3.2 Wolpert 閾値・§4.2-b 名簿と実体の二層・
      §4.2-d 転入は原理的に閉じない・§4.3 初期条件 R-c)/ src/society/population.py の docstring。
ユーザー決定(2026-08-14): 転入は**案A(L4 定着昇格)**。「イベントが発生することが重要で、
エージェントの増減はあまり気にしていない」= 3 つの人生イベントが実際に観測されることが主眼。

検収:
  (A) 既定 OFF が完全 no-op(台帳も属性も L1 も 0・退避辞書のバイト列不変・レジストリ宣言済み)
  (B) config 正準化・純関数(安定ハッシュ・Bresenham の分布保存)
  (C) 転出 = Wolpert 閾値(蓄積が個体固定閾値を跨いで発火・L1 に理由と閾値の痕跡・
      後始末=住戸解放/退職/世帯離脱/関係休眠/金の境界フロー・縁が引き止める)
  (D) 転入 = 定着昇格(来街日数 × 縁 × 空き住戸 × 求人・名簿不変・翌日以降 resident 資格で在場・
      roster 記載・搬送 round-trip)
  (E) 出生 = 夫婦固定の位相(世帯サイズ +1・新生児 id が名簿 id と衝突しない・LLM 呼数不変)
  (F) 人口会計(台帳の行数 == イベント件数 == counts)
  (G) resume == straight
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

import build_persona_pool as bpp                        # noqa: E402
from society import household as HH                     # noqa: E402
from society import population as P                     # noqa: E402
from society import registry as R                       # noqa: E402
from society.config import load_config                  # noqa: E402
from society.engine import checkpoint, scheduler        # noqa: E402
from society.engine.simulation import Simulation        # noqa: E402
from society.observer.schema import EVENT_KINDS         # noqa: E402
from society.world import pool as pool_mod              # noqa: E402

POP_KINDS = (P.KIND_EMIGRATE, P.KIND_SETTLE, P.KIND_BIRTH)


# --------------------------------------------------------------------------- #
# 共通ヘルパ(test_assets.py と同型)
# --------------------------------------------------------------------------- #
def _cfg(name, n_steps=24, n_agents=12, **ov):
    dot = ["run.seed=42", f"run.n_agents={n_agents}", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock", "observer.snapshot_every=144"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def _sim(tmp_path, name, n_steps=24, n_agents=12, **ov):
    return Simulation(_cfg(name, n_steps, n_agents, **ov), out_dir=tmp_path / name)


def _pop_events(sim, kind=None):
    out = []
    for e in sim.logger.events:
        if e.kind != "life_event":
            continue
        k = e.payload.get("kind")
        if k in POP_KINDS and (kind is None or k == kind):
            out.append(e)
    return out


def _run_days(sim, n_days: int, start_day: int = 0) -> None:
    """日境界だけを回す(``phase`` の直接駆動 = 速い・LLM を 1 度も呼ばない)。"""
    for d in range(start_day, start_day + n_days):
        P.phase(sim, step=d * 144, sim_min=d * 1440)


class _FixedLLM:
    """**プロンプト非依存**の巡回応答スタブ(test_assets.py と同型)。"""

    name = "fixed"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.hits = 0
        self.prompts: list[str] = []

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        out = self.responses[self.calls % len(self.responses)]
        self.calls += 1
        self.prompts.append(prompt)
        return out, str(self.calls), False


ON_EMI = {"population.emigration.enabled": "true", "relations.enabled": "true",
          "household.enabled": "true", "organizations.enabled": "true"}
ON_BIRTH = {"population.births.enabled": "true", "relations.enabled": "true",
            "household.enabled": "true"}


# =========================================================================== #
# (A) 出荷既定 = 完全 no-op
# =========================================================================== #
def test_shipped_default_is_off():
    cfg = load_config([])
    # ★``cfg.population`` を attribute で引く: conf の最上位キーを ``pop`` にすると
    #   OmegaConf の ``DictConfig.pop`` メソッドに解決されてしまう(実装中に踏んだ罠)。
    assert cfg.population.emigration.enabled is False
    assert cfg.population.immigration.enabled is False
    assert cfg.population.births.enabled is False


def test_off_creates_no_state_no_events_no_summary_key(tmp_path):
    """既定 OFF: 台帳オブジェクトすら作らず・L1 に 1 件も出ず・summary にキーが無い。"""
    sim = _sim(tmp_path, "pop_off", n_steps=24)
    sim.run()
    assert P.state_of(sim) is None, "OFF なのに sim._pop_state が生えている"
    assert not hasattr(sim, "_pop_state")
    assert _pop_events(sim) == [], "OFF なのに POP の life_event が出ている"
    summary = json.loads((sim.out_dir / "summary.json").read_text(encoding="utf-8"))
    assert "population" not in summary
    for a in sim.agents:                            # agent に属性を 1 つも生やさない
        for name in ("pop_stress", "pop_days", "pop_settled", "pop_emigrated",
                     "pop_layer", "pop_pair_since"):
            assert not hasattr(a, name), f"OFF なのに {name} が生えている"


def test_off_dehydrate_dict_is_byte_identical(tmp_path):
    """POP OFF の個体では退避辞書に pop_* のキーが 1 つも生えない(pool 回転のバイト一致)。"""
    sim = _sim(tmp_path, "pop_off_dehy", n_steps=1)
    st = pool_mod.dehydrate(sim.agents[0])
    assert "misc" not in st or not any(k.startswith("pop_") for k in st["misc"])


def test_registry_declares_all_toggles():
    """新トグルは 3 つとも registry._f() 宣言済み(未宣言検出テストの前段)。"""
    for key in ("population.emigration.enabled", "population.immigration.enabled",
                "population.births.enabled"):
        assert key in R.BY_ID, f"{key} がレジストリ未宣言"
        f = R.BY_ID[key]
        assert f.repro_tier == "strict"
        assert f.fingerprint_risk == "none"
    assert "population.immigration.require_job" in R.ALLOWLIST, "子トグルが ALLOWLIST に無い"
    assert R.undeclared_toggles(load_config([])) == [], "未宣言の bool 設定キーがある"


def test_no_new_event_kinds_registered():
    """★新しい L1 kind を 1 つも足していない(3 イベントは既存 life_event の下位 kind)。"""
    for k in POP_KINDS:
        assert k not in EVENT_KINDS, f"{k} を新しい event kind として登録している"
    assert "life_event" in EVENT_KINDS


# =========================================================================== #
# (B) config 正準化 + 純関数
# =========================================================================== #
def test_build_cfg_defaults_and_coercion():
    cfg = P.build_cfg(None)
    assert cfg["emigration"] == P.EMIGRATION_DEFAULTS
    assert cfg["immigration"] == P.IMMIGRATION_DEFAULTS
    assert cfg["births"] == P.BIRTHS_DEFAULTS
    got = P.build_cfg({"emigration": {"enabled": 1, "threshold": "3", "unknown": 9},
                       "births": {"age_min": "20"}})
    assert got["emigration"]["enabled"] is True
    assert got["emigration"]["threshold"] == 3.0
    assert "unknown" not in got["emigration"]
    assert got["births"]["age_min"] == 20 and isinstance(got["births"]["age_min"], int)


def test_stable_hash_is_deterministic_and_seed_independent():
    """閾値・位相の源は安定ハッシュ(run.seed 非依存・プロセス非依存)= 乱数 stream ゼロ。"""
    a = P._u01(20260814, "emig", "L4_00000001")
    b = P._u01(20260814, "emig", "L4_00000001")
    c = P._u01(20260814, "emig", "L4_00000002")
    assert a == b and a != c
    assert 0.0 <= a < 1.0
    # spread=0 なら全員同じ閾値・spread>0 なら base を中心に散る
    assert P._spread(6.0, 0.0, a) == 6.0
    vals = [P._spread(6.0, 0.6, P._u01(1, "x", i)) for i in range(400)]
    assert min(vals) >= 6.0 * 0.7 - 1e-9 and max(vals) <= 6.0 * 1.3 + 1e-9
    assert abs(sum(vals) / len(vals) - 6.0) < 0.2


def test_due_is_bresenham_and_preserves_rate():
    """出生の位相構成は PRES-A1 と同じ Beatty 列 = 1 日あたり確率が厳密に 1/I。"""
    I, D = 37.0, 37 * 20
    for theta in (0.0, 0.13, 0.5, 0.99):
        n = sum(1 for d in range(D) if P._due(theta, I, d))
        assert n in (int(D / I), int(D / I) + 1), f"θ={theta} で長期レートが崩れた"
    # 母集団平均は 1/I に一致する(位相が一様に散っている)
    hits = sum(1 for i in range(2000) if P._due(P._u01(7, "p", i), I, 5))
    assert abs(hits / 2000 - 1.0 / I) < 0.012
    assert P._due(0.5, 0.0, 3) is False              # 間隔 0 は発火しない(ゼロ除算なし)


# =========================================================================== #
# (C) 転出 = Wolpert 閾値
# =========================================================================== #
def _stressed(sim, n=4, **kw):
    res = [a for a in sim.agents if not a.visitor][:n]
    for a in res:
        for k, v in kw.items():
            setattr(a, k, v)
    return res


def test_emigration_accumulates_and_fires_at_individual_threshold(tmp_path):
    """蓄積が**個体固定の閾値**を跨いだ日に発火し、L1 に理由と閾値の痕跡が残る。"""
    sim = _sim(tmp_path, "pop_emi", n_steps=1,
               **dict(ON_EMI, **{"population.emigration.threshold": "3.0",
                                 "population.emigration.spread": "0.0"}))
    res = _stressed(sim, 3, rent_due=5000.0)         # w_rent 0.6 + w_alone 0.4 = 1.0/日
    _run_days(sim, 1)
    assert _pop_events(sim, P.KIND_EMIGRATE) == [], "1 日で閾値 3.0 を跨ぐのはおかしい"
    assert abs(float(res[0].pop_stress) - 1.0) < 1e-9
    _run_days(sim, 5, start_day=1)                   # 1.0 → 1.85 → 2.57 → 3.18(decay 0.85)
    ev = _pop_events(sim, P.KIND_EMIGRATE)
    assert len(ev) == 3, f"3 人とも転出するはず: {len(ev)}"
    pay = ev[0].payload
    assert pay["stress"] >= pay["threshold"], "閾値未満で発火している"
    assert set(pay["causes"]) == {"rent", "alone"}, pay["causes"]
    assert pay["threshold"] == 3.0                   # spread=0 = 全員同じ
    assert pay["day"] == 3, f"蓄積の日数計算が違う: {pay['day']}"


def test_emigration_threshold_is_individual(tmp_path):
    """spread>0 では**同じストレスでも人によって出る日が違う**(個人差 = Wolpert の閾値)。"""
    sim = _sim(tmp_path, "pop_emi_ind", n_steps=1,
               **dict(ON_EMI, **{"population.emigration.threshold": "3.0",
                                 "population.emigration.spread": "0.8"}))
    _stressed(sim, 12, rent_due=5000.0)
    _run_days(sim, 8)
    days = {e.payload["day"] for e in _pop_events(sim, P.KIND_EMIGRATE)}
    assert len(days) >= 2, f"全員が同じ日に出ている(閾値が個体固定でない): {days}"


def test_ties_hold_people_back(tmp_path):
    """縁(closeness の高い関係)が蓄積を押し戻す(Speare の媒介変数を構造化状態で書く)。"""
    sim = _sim(tmp_path, "pop_emi_tie", n_steps=1,
               **dict(ON_EMI, **{"population.emigration.threshold": "3.0",
                                 "population.emigration.spread": "0.0"}))
    res = [a for a in sim.agents if not a.visitor]
    lonely, tied = res[0], res[1]
    for a in (lonely, tied):
        a.rent_due = 5000.0
    for oid in (res[2].id, res[3].id, res[4].id, res[5].id, res[6].id):
        tied.mem.relations[oid] = {"closeness": 9.0, "count": 3, "tier": 2}
    _run_days(sim, 6)
    gone = {e.agent_id for e in _pop_events(sim, P.KIND_EMIGRATE)}
    assert lonely.id in gone, "孤立した滞納者が出ていかない"
    assert tied.id not in gone, "縁のある滞納者まで出ていっている(アンカーが効いていない)"
    assert float(tied.pop_stress) == 0.0, "縁が押し戻し切れていない(蓄積が残っている)"


def test_emigration_aftermath_is_closed(tmp_path):
    """後始末: 死と同型の永続退場・住戸解放・退職(job_change)・世帯離脱・台帳の 1 行。"""
    from society.health import NEVER_RETURN
    sim = _sim(tmp_path, "pop_emi_after", n_steps=1,
               **dict(ON_EMI, **{"population.emigration.threshold": "0.5",
                                 "population.emigration.spread": "0.0",
                                 "household.realistic": "true"}))
    scheduler._ensure_orgs(sim)
    victim = [a for a in sim.agents if not a.visitor][0]
    victim.rent_due = 5000.0
    home_was = victim.home_building
    mates = list(getattr(victim, "housemates", None) or [])
    _run_days(sim, 1)
    ev = _pop_events(sim, P.KIND_EMIGRATE)
    assert ev, "閾値 0.5 で誰も出ていかない"
    assert victim.loc == "outside" and victim.return_at == NEVER_RETURN
    assert getattr(victim, "dead", False) is False, "転出者は**生きている**(死と区別する)"
    assert victim.home_building == "", "住戸が解放されていない"
    assert getattr(victim, "_home_moved", False) is True
    assert victim.household_id is None and victim.housemates == []
    assert getattr(victim, "_hh_detached", False) is True
    for oid in mates:                               # 残るメンバの housemates からも外れる
        other = sim.agent_by_id.get(oid)
        if other is not None:
            assert victim.id not in (other.housemates or [])
    st = P.state_of(sim)
    key = P._key_of(victim)
    assert key in st["gone"] and st["counts"]["emigrate"] == len(st["gone"])
    assert st["gone"][key]["home"] == home_was


def test_emigrant_money_is_frozen_without_org_accounting(tmp_path):
    """org 会計 OFF: 第107 の死と同じく残高は**凍結**(勘定の無い所へ金を捨てない)。"""
    sim = _sim(tmp_path, "pop_emi_freeze", n_steps=1,
               **dict(ON_EMI, **{"population.emigration.threshold": "0.5",
                                 "population.emigration.spread": "0.0"}))
    victim = [a for a in sim.agents if not a.visitor][0]
    victim.rent_due = 5000.0
    had = float(victim.money) + float(getattr(victim, "account", 0.0))
    _run_days(sim, 1)
    st = P.state_of(sim)
    assert st["money_exported"] == 0.0
    assert abs(st["money_frozen"] - had) < 1e-6
    assert float(victim.money) == pytest.approx(
        had - float(getattr(victim, "account", 0.0)))


def test_emigrant_money_leaves_via_row_when_accounting_on(tmp_path):
    """org 会計 ON: 現金・口座は IF-E の境界フロー(RoW)で域外へ抜け、本人の残高は 0。"""
    from society import economy_sfc as SFC
    sim = _sim(tmp_path, "pop_emi_row", n_steps=1,
               **dict(ON_EMI, **{"population.emigration.threshold": "0.5",
                                 "population.emigration.spread": "0.0",
                                 "economy.org_accounting.enabled": "true"}))
    scheduler._ensure_orgs(sim)
    victim = [a for a in sim.agents if not a.visitor][0]
    victim.rent_due = 5000.0
    had = float(victim.money) + float(getattr(victim, "account", 0.0))
    before = SFC.row_net(sim)
    _run_days(sim, 1)
    assert float(victim.money) == 0.0
    assert float(getattr(victim, "account", 0.0)) == 0.0
    assert SFC.row_net(sim) == pytest.approx(before + had)
    assert SFC.state_of(sim)["row"][P.ROW_CHANNEL]["out"] == pytest.approx(had)


def test_emigration_fires_once_per_person(tmp_path):
    """同じ人が 2 度転出しない(冪等 = 台帳が唯一の真実源)。"""
    sim = _sim(tmp_path, "pop_emi_once", n_steps=1,
               **dict(ON_EMI, **{"population.emigration.threshold": "0.5",
                                 "population.emigration.spread": "0.0",
                                 "population.emigration.w_alone": "0.0"}))
    _stressed(sim, 3, rent_due=5000.0)
    _run_days(sim, 6)
    ids = [e.agent_id for e in _pop_events(sim, P.KIND_EMIGRATE)]
    assert len(ids) == len(set(ids)) == 3


def test_phase_is_idempotent_within_a_day(tmp_path):
    """同じ日に 2 度呼んでも 1 度しか処理しない(mid-day resume の二重発火を防ぐ)。"""
    sim = _sim(tmp_path, "pop_day_guard", n_steps=1,
               **dict(ON_EMI, **{"population.emigration.threshold": "0.5",
                                 "population.emigration.spread": "0.0"}))
    _stressed(sim, 2, rent_due=5000.0)
    P.phase(sim, step=0, sim_min=0)
    n1 = len(_pop_events(sim))
    P.phase(sim, step=5, sim_min=50)
    assert len(_pop_events(sim)) == n1


# =========================================================================== #
# (D) 転入 = 案A「L4 定着昇格」
# =========================================================================== #
@pytest.fixture(scope="module")
def small_pool(tmp_path_factory):
    """P5 生成器で 2,000 体の小プールを tmp に生成(実プール 736MB は触らない)。"""
    out = tmp_path_factory.mktemp("pop_pool")
    orgs = json.loads((_ROOT / "data" / "organizations_shibuya_wide11k.json")
                      .read_text(encoding="utf-8"))
    pop = json.loads((_ROOT / "data" / "shibuya_population.json")
                     .read_text(encoding="utf-8"))
    bpp.build_pool(out, seed=42, fraction=0.002, orgs=orgs, pop=pop,
                   total_target=1_000_000)
    return out


def _pool_cfg(pool_dir, name, n_steps, **ov):
    dot = ["run.seed=42", "run.n_agents=0", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock", "observer.snapshot_every=144",
           "run.start_tod=00:00", "run.natural_start=true", "pool.enabled=true",
           f"pool.dir={Path(pool_dir).as_posix()}", "pool.present_cap=2000"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


ON_SETTLE = {"population.immigration.enabled": "true", "population.immigration.days_min": "1",
             "population.immigration.spread": "0.0", "population.immigration.ties_min": "0",
             "population.immigration.require_job": "false",
             "relations.enabled": "true", "household.enabled": "true"}


@pytest.fixture(scope="module")
def settle_run(small_pool, tmp_path_factory):
    d = tmp_path_factory.mktemp("pop_settle") / "run"
    sim = Simulation(_pool_cfg(small_pool, "pop_settle", 3 * 144,
                               **dict(ON_SETTLE,
                                      **{"observer.roster_daily.enabled": "true"})),
                     out_dir=d)
    sim.run()
    return sim, d


def test_settle_fires_and_promotes_to_resident(settle_run):
    """L4 定着昇格が実発火し、昇格者が resident 資格(visitor=False + 住居)で在場し続ける。"""
    sim, _d = settle_run
    st = P.state_of(sim)
    ev = _pop_events(sim, P.KIND_SETTLE)
    assert ev, "L4 の定着昇格が 1 件も起きていない"
    assert len(ev) == st["counts"]["settle"] == len(st["settled"])
    for _pid, info in list(st["settled"].items())[:5]:
        a = sim.agent_by_id[info["agent_id"]]
        assert a.visitor is False, "昇格者が来街者のまま"
        assert a.home_building == info["building"] and a.home_node == info["node"]
        assert getattr(a, "pop_settled", False) is True
    # 初日に定着した人は**最終日も在場**している(resident 資格を毎日満たす)
    day0 = [i for i in st["settled"].values() if i["day"] == 0]
    assert day0, "初日の定着が無い"
    assert all(sim.present_agent(i["agent_id"]) is not None for i in day0[:10])


def test_settler_is_recorded_in_roster(settle_run):
    """昇格者は入場者名簿(roster.parquet)に行を持つ(素性が観測から落ちない)。"""
    sim, d = settle_run
    ids = {r["agent_id"] for r in pq.read_table(d / "roster.parquet").to_pylist()}
    missing = [i["agent_id"] for i in P.state_of(sim)["settled"].values()
               if i["agent_id"] not in ids]
    assert missing == [], f"roster に無い定着者 {missing[:5]}"


def test_settle_leaves_the_pool_record_untouched(settle_run, small_pool):
    """★名簿(pool record)は 1 バイトも書き換えない = 決定論の礎(昇格は台帳が持つ)。"""
    sim, _d = settle_run
    for pid in list(P.state_of(sim)["settled"])[:5]:
        rec = sim._pool.get(pid)
        assert rec["presence"] == P.LAYER_STOCHASTIC, "record の層が書き換わっている"
        assert rec["id"] == pid


def test_settle_requires_ties(small_pool, tmp_path):
    """縁(域内で築いた関係)が足りなければ定着しない = 条件は域内状態への応答である。"""
    sim = Simulation(_pool_cfg(small_pool, "pop_settle_ties", 144,
                               **dict(ON_SETTLE, **{"population.immigration.ties_min": "99"})),
                     out_dir=tmp_path / "pop_settle_ties")
    sim.run()
    assert _pop_events(sim, P.KIND_SETTLE) == []
    assert P.state_of(sim)["considered"]["settle"] > 0, "候補の走査自体が起きていない"


def test_settle_requires_vacancy(small_pool, tmp_path):
    """空き住戸が無ければ定着しない(物理的に住めない)。"""
    sim = Simulation(_pool_cfg(small_pool, "pop_settle_vac", 1, **ON_SETTLE),
                     out_dir=tmp_path / "pop_settle_vac")
    sim.city.residential_buildings = []              # 住宅を 1 棟も持たない街
    P.phase(sim, 0, 0)
    assert _pop_events(sim, P.KIND_SETTLE) == []


def test_settle_requires_job_opening_when_gated(small_pool, tmp_path):
    """求人の空き定員(受け皿)をゲートにすると、台帳が無い街では定着が起きない。"""
    sim = Simulation(_pool_cfg(small_pool, "pop_settle_job", 1,
                               **dict(ON_SETTLE,
                                      **{"population.immigration.require_job": "true",
                                         "organizations.enabled": "false"})),
                     out_dir=tmp_path / "pop_settle_job")
    P.phase(sim, 0, 0)
    assert _pop_events(sim, P.KIND_SETTLE) == []


def test_settlers_never_share_one_dwelling(settle_run):
    """1 戸に 2 人が同日入居しない(空き住戸は 1 人が取ったら消える)。"""
    blds = [i["building"] for i in P.state_of(settle_run[0])["settled"].values()]
    assert len(blds) == len(set(blds))


def test_pop_fields_survive_pool_rotation(tmp_path):
    """搬送 round-trip: pop_* の多日スケール状態が dehydrate/hydrate を往復する。"""
    sim = _sim(tmp_path, "pop_carry", n_steps=1)
    a = sim.agents[0]
    a.pop_stress, a.pop_days = 4.25, 7
    a.pop_settled, a.pop_pair_since = True, 3
    st = pool_mod.dehydrate(a)
    assert st["misc"]["pop_stress"] == 4.25 and st["misc"]["pop_days"] == 7
    assert st["misc"]["pop_settled"] is True and st["misc"]["pop_pair_since"] == 3
    b = sim.agents[1]
    pool_mod.hydrate(b, st)
    assert (b.pop_stress, b.pop_days, b.pop_settled, b.pop_pair_since) == (4.25, 7, True, 3)


def test_apply_on_entry_reseats_settler_and_reexits_emigrant(settle_run):
    """入場フックが台帳の事実を冪等に着せ直す(名簿は来街者のままでも住民に戻る)。"""
    from society.health import NEVER_RETURN
    sim, _d = settle_run
    pid, info = next(iter(P.state_of(sim)["settled"].items()))
    fresh = sim.build_pool_agent(pid, sim._pool.get(pid))
    assert fresh.visitor is False and fresh.home_building == info["building"]
    # 転出済みとして台帳へ入れると、入場しても即座に永続退場の物理状態へ戻る
    P.state_of(sim)["gone"][pid] = {"day": 0, "agent_id": info["agent_id"],
                                    "stress": 0.0, "threshold": 0.0,
                                    "causes": [], "home": "", "carried": 0.0}
    try:
        again = sim.build_pool_agent(pid, sim._pool.get(pid))
        assert again.loc == "outside" and again.return_at == NEVER_RETURN
        assert getattr(again, "pop_emigrated", False) is True
        # 在場資格からも落ちる(presence オーバレイ)
        assert pid not in P.apply_presence(sim, {pid})
    finally:
        P.state_of(sim)["gone"].pop(pid, None)


def test_apply_presence_is_identity_when_ledger_empty(tmp_path):
    """台帳が空(= 何も起きていない)ときは在場集合をそのまま返す = バイト一致。"""
    sim = _sim(tmp_path, "pop_overlay_off", n_steps=1)
    ids = {"a", "b"}
    assert P.apply_presence(sim, ids) is ids
    assert P.presence_overlay(sim) is None


# =========================================================================== #
# (E) 出生 = 夫婦固定の位相
# =========================================================================== #
def _paired(tmp_path, name, **ov):
    base = dict(ON_BIRTH)
    base.update({"population.births.interval_days": "2.0",
                 "population.births.spread": "0.0",
                 "population.births.couple_share": "1.0",
                 "population.births.age_min": "0",
                 "population.births.age_max": "99"})
    base.update(ov)
    sim = _sim(tmp_path, name, n_steps=1, **base)
    res = [a for a in sim.agents if not a.visitor]
    HH.bond(sim, res[0], res[1], 0, 0)
    return sim, res


def test_birth_fires_and_household_grows(tmp_path):
    """出生イベントが発火し、世帯サイズが +1 になる(新生児が housemates に載る)。"""
    sim, res = _paired(tmp_path, "pop_birth")
    before = len(getattr(res[0], "housemates", None) or [])
    _run_days(sim, 6)
    ev = _pop_events(sim, P.KIND_BIRTH)
    assert ev, "出生が 1 件も起きていない"
    child = ev[0].payload["child"]
    assert child in res[0].housemates and child in res[1].housemates
    assert len(res[0].housemates) >= before + 1, "世帯サイズが増えていない"
    st = P.state_of(sim)
    assert len(st["births"]) == st["counts"]["birth"] == len(ev)
    row = st["births"][0]
    assert row["parents"] == sorted([int(res[0].id), int(res[1].id)])
    assert row["child_id"] == child


def test_newborn_id_never_collides_with_the_roster(tmp_path):
    """新生児 id は名簿 id 空間の**末尾の外**から採る(既存 agent と衝突しない)。"""
    sim, _res = _paired(tmp_path, "pop_birth_id")
    _run_days(sim, 6)
    ids = {int(k) for k in sim.agent_by_id}
    rows = P.state_of(sim)["births"]
    assert rows
    for row in rows:
        assert row["child_id"] not in ids, "新生児 id が既存 agent と衝突している"
        assert row["child_id"] >= max(ids), "新生児 id が名簿 id 空間の内側から採られている"
    assert len({r["child_id"] for r in rows}) == len(rows), "新生児 id が重複している"


def test_birth_respects_age_window_and_child_cap(tmp_path):
    """年齢帯の外の夫婦は産まない / 子の上限で止まる(世帯状態の関数である)。"""
    sim, _res = _paired(tmp_path, "pop_birth_age",
                        **{"population.births.age_min": "80", "population.births.age_max": "99"})
    _run_days(sim, 8)
    assert _pop_events(sim, P.KIND_BIRTH) == [], "年齢帯の外で出産している"

    sim2, _r2 = _paired(tmp_path, "pop_birth_cap",
                        **{"population.births.max_children": "1",
                           "population.births.min_interval_days": "0"})
    _run_days(sim2, 20)
    assert len(_pop_events(sim2, P.KIND_BIRTH)) == 1


def test_birth_also_reads_household_spouses(tmp_path):
    """夫婦の源は ``partner_id`` だけでなく**世帯の続柄**(初期条件の夫婦)も含む。

    ★これが無いと 10 日ランでは夫婦が原理的にほぼ存在しない(``form_partners`` は
      closeness 閾値超でしか張らない)= 出生が構造的に 0 件になる。
    """
    base = {"population.births.enabled": "true", "relations.enabled": "true",
            "household.enabled": "true", "household.realistic": "true",
            "population.births.interval_days": "2.0",
            "population.births.spread": "0.0",
            "population.births.couple_share": "1.0",
            "population.births.age_min": "0", "population.births.age_max": "99"}
    got = {}
    for label, ov in (("on", {}),
                      ("off", {"population.births.household_spouses": "false"})):
        sim = _sim(tmp_path, f"pop_spouse_{label}", n_steps=1, **dict(base, **ov))
        # ★誰も bond しない = partner_id は 1 件も無い。夫婦は続柄からしか引けない。
        assert all(getattr(a, "partner_id", None) is None for a in sim.agents)
        spouses = sum(1 for a in sim.agents if HH.spouse_of(sim, a) is not None)
        _run_days(sim, 8)
        got[label] = (spouses, len(_pop_events(sim, P.KIND_BIRTH)))
    assert got["on"][0] >= 2, f"realistic 束ねが夫婦を 1 組も作っていない: {got}"
    assert got["on"][1] > 0, f"続柄由来の夫婦から出生が起きない: {got}"
    assert got["off"][1] == 0, f"OFF なのに続柄から出生している: {got}"


def test_newborn_is_not_an_agent_and_costs_no_llm_calls(tmp_path):
    """★新生児は sim.agents に入らない = 在場数も LLM 呼数も 1 つも変わらない。"""
    acts = [json.dumps({"action": "stay"}, ensure_ascii=False),
            json.dumps({"action": "speak", "text": "こんにちは"}, ensure_ascii=False)]
    got = []
    for name, ov in (("pop_llm_off", {"relations.enabled": "true",
                                      "household.enabled": "true"}),
                     ("pop_llm_on", dict(ON_BIRTH,
                                         **{"population.births.interval_days": "3.0",
                                            "population.births.spread": "0.0",
                                            "population.births.couple_share": "1.0",
                                            "population.births.age_min": "0",
                                            "population.births.age_max": "99"}))):
        sim = Simulation(_cfg(name, 48, 12, **ov), out_dir=tmp_path / name)
        sim.llm = _FixedLLM(acts)
        res = [a for a in sim.agents if not a.visitor]
        HH.bond(sim, res[0], res[1], 0, 0)
        sim.run()
        got.append((sim.llm.calls, len(sim.agents)))
    assert got[0][0] == got[1][0], f"LLM 呼数が ON/OFF で違う: {got}"
    assert got[0][1] == got[1][1], "新生児が在場者を増やしている"


# =========================================================================== #
# (F) 人口会計
# =========================================================================== #
def test_population_accounting_closes(tmp_path, small_pool):
    """人口会計 Σ 整合: 台帳の行数 == counts == L1 イベント件数、net の恒等式。"""
    ov = dict(ON_SETTLE, **{"population.emigration.enabled": "true",
                            "population.emigration.threshold": "1.0",
                            "population.emigration.spread": "0.0",
                            "population.births.enabled": "true",
                            "population.births.interval_days": "2.0",
                            "population.births.spread": "0.0",
                            "population.births.couple_share": "1.0",
                            "population.births.age_min": "0", "population.births.age_max": "99"})
    sim = Simulation(_pool_cfg(small_pool, "pop_account", 3 * 144, **ov),
                     out_dir=tmp_path / "pop_account")
    sim.run()
    prov = P.provenance(sim)
    st = P.state_of(sim)
    assert prov is not None
    assert prov["ledger"] == {"gone": len(st["gone"]), "settled": len(st["settled"]),
                              "births": len(st["births"])}
    for k, ledger_key in (("emigrate", "gone"), ("settle", "settled"),
                          ("birth", "births")):
        n_ev = len(_pop_events(sim, getattr(P, f"KIND_{k.upper()}")))
        assert prov["counts"][k] == prov["ledger"][ledger_key] == n_ev, k
    assert prov["net"] == (prov["counts"]["settle"] + prov["counts"]["birth"]
                           - prov["counts"]["emigrate"] - prov["deaths"])
    summary = json.loads((sim.out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["population"]["counts"] == prov["counts"]
    assert summary["population"]["target_per_day"]["emigrate"] == 7.8


def test_emigrated_never_returns(tmp_path, small_pool):
    """転出者は在場資格を**永久に**失う(日次ローテーションで戻ってこない)。"""
    ov = dict(ON_SETTLE, **{"population.emigration.enabled": "true",
                            "population.emigration.threshold": "0.1",
                            "population.emigration.spread": "0.0"})
    sim = Simulation(_pool_cfg(small_pool, "pop_never_back", 3 * 144, **ov),
                     out_dir=tmp_path / "pop_never_back")
    sim.run()
    st = P.state_of(sim)
    gone = set(st["gone"])
    assert gone, "転出が 1 件も起きていない"
    # ① 在場資格から永久に落ちる(presence オーバレイ)
    assert not (gone & P.apply_presence(sim, set(gone)))
    # ② 前日までに転出した人は日次ローテーションで sim.agents から抜けている
    last_day = int(st["day"])
    old = {pid for pid, info in st["gone"].items() if int(info["day"]) < last_day}
    still = {getattr(a, "pool_pid", None) for a in sim.agents} & old
    assert not still, f"前日までに転出した人が在場している: {sorted(still)[:5]}"
    # ③ 当日の転出者は死と同型で sim.agents に残るが、物理的には圏外(抜き取りを行わない規約)
    for a in sim.agents:
        if getattr(a, "pool_pid", None) in gone:
            assert a.loc == "outside" and getattr(a, "pop_emigrated", False) is True


# =========================================================================== #
# (G) resume == straight
# =========================================================================== #
def test_resume_matches_straight(tmp_path, small_pool):
    """POP ON で resume==straight(台帳が checkpoint に載る・日境界が二重に走らない)。"""
    ov = dict(ON_SETTLE, **{"population.emigration.enabled": "true",
                            "population.emigration.threshold": "1.0",
                            "population.emigration.spread": "0.0",
                            "population.births.enabled": "true",
                            "population.births.interval_days": "2.0",
                            "population.births.spread": "0.0",
                            "population.births.couple_share": "1.0",
                            "population.births.age_min": "0", "population.births.age_max": "99"})
    split, total = 200, 300
    straight_dir = tmp_path / "pop_straight"
    straight = Simulation(_pool_cfg(small_pool, "pop_straight", total, **ov),
                          out_dir=straight_dir)
    straight.run()

    d = tmp_path / "pop_resumed"
    sim1 = Simulation(_pool_cfg(small_pool, "pop_resumed", split, **ov,
                                **{"observer.checkpoint_every": split}), out_dir=d)
    for step in range(split):
        scheduler.run_step(sim1, step)
    checkpoint.save(sim1, split, d / "checkpoint" / f"ckpt-{split:06d}.pkl.gz")
    sim1._save_pool_sidecar(split)
    sim1.logger.flush_segment()
    sim2 = Simulation(_pool_cfg(small_pool, "pop_resumed", total, **ov,
                                **{"observer.checkpoint_every": split}), out_dir=d)
    sim2.run(resume_from=d)
    for stem in ("l1_events", "l2_metrics", "l3_snapshots"):
        sa = pq.read_table(straight_dir / f"{stem}.parquet").to_pylist()
        sb = pq.read_table(d / f"{stem}.parquet").to_pylist()
        assert sa == sb, f"{stem} 不一致(POP resume)"
    ja = json.loads((straight_dir / "summary.json").read_text(encoding="utf-8"))
    jb = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    assert ja["population"] == jb["population"], "人口会計が resume で straight と食い違う"
    assert P.state_of(sim2)["gone"].keys() == P.state_of(straight)["gone"].keys()
    assert P.state_of(sim2)["settled"].keys() == P.state_of(straight)["settled"].keys()


def test_same_seed_twice_is_identical(tmp_path, small_pool):
    """同 seed 2 ラン一致(閾値・位相が安定ハッシュ = 乱数 stream を 1 本も引かない)。"""
    ov = dict(ON_SETTLE, **{"population.emigration.enabled": "true",
                            "population.emigration.threshold": "1.0",
                            "population.emigration.spread": "0.4"})
    got = []
    for name in ("pop_det_a", "pop_det_b"):
        sim = Simulation(_pool_cfg(small_pool, name, 2 * 144, **ov),
                         out_dir=tmp_path / name)
        sim.run()
        got.append([[e.step, e.agent_id,
                     json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
                    for e in _pop_events(sim)])
    assert got[0] == got[1] and got[0], "同 seed 2 ランが一致しない / イベントが 0 件"
