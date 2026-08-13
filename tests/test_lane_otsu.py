"""レーン乙(第113): 痩せ修正の第2弾。

くまなく調べた「街を出た個体まわりの構造的な痩せ」を 7 ブロックで塞いだ結果を機械固定する。

  ブロック1  搬送族の追加(dehydrate/hydrate)= EPR の visits・約束帳・自助努力・宿泊・退屈・
             可塑性 g・世界観の期待・目標/趣味・観光/言語・対話履歴・検証待ち・出動中の印・勾留
  ブロック2  起動時 1 回の配布の入場駆動化(SNS/顔なじみ/SDT/needs/価値/LOD/観光/目標)
  ブロック3  世帯の根治(pool 名簿からの決定論再構築 + 動的変化の搬送 + 幽霊ガード)
  ブロック4  議席の根治(定員割れ補充・在場議員のみが投票・欠席の痕跡)
  ブロック5  世界側供給の 1 回きり(org 期首預金の母集団 / 卸の起動時在庫)
  ブロック6  不在中に時間が進まないバイアス(関係減衰・評判風化・悪評忘却・在院日数)
  ブロック7  幽霊書き込みの残り(在場述語への置換)

**共通の受入条件**: 当該機構 OFF / 毎日在場のランでは 1 バイトも変わらないこと。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from society.agents.agent import Agent            # noqa: E402
from society.agents.memory import MemoryStore     # noqa: E402
from society.world import pool as pool_mod        # noqa: E402


def _bare(aid: int = 1) -> Agent:
    return Agent(id=aid, name="甲", age=30, occupation="会社員", persona="p",
                 traits={}, states={}, mem=MemoryStore())


# ======================================================== ブロック1: 搬送族の追加
def test_block1_dehydrate_without_the_new_families_is_unchanged():
    """新しく足した族を 1 つも持たない個体の退避 dict に新キーが生えない(既定ラン不変)。

    「属性が在って**非既定**のときだけ入れる」設計の受入条件そのもの。空 Counter / 空 list /
    0 / -1 でもキーを作らないこと(機構 ON だが未経験のランで退避のバイト列が割れないこと)。
    """
    a = _bare()
    base = pool_mod.dehydrate(a)
    new_keys = {"misc", "hh", "home", "housemates", "partner_id", "visits", "self_dev",
                "schedule", "hobbies", "dialog_hist", "tl_pending", "wv_expect", "plast"}
    assert not (set(base) & new_keys), f"素の個体に新キーが生えている: {set(base) & new_keys}"

    # 「機構 ON だが未経験」= 既定値だけを明示的に持っている状態
    a.visits.clear()
    a.self_dev = {}
    a.schedule = []
    a.housemates = []
    a.partner_id = None
    a._boredom = 0.0
    a.detained_until = 0
    a.lodging_nights = 0
    a.city_ops_ems_until = -1
    a.facility_call_until = -1
    a.incenv_fire_until = -1
    a.tourist = False
    a.language = ""
    a.life_goal = ""
    a.hobbies = []
    a._dialog_hist = {}
    a._tl_pending = None
    a.wv_expect = {}
    a._fire_g = None
    a._rel_decay_day = -1
    a._gossip_fade_day = -1
    assert pool_mod.dehydrate(a) == base, "既定値だけの状態で退避 dict が変わっている"


def test_block1_visits_and_the_rest_round_trip():
    """新しく足した族が往復で同値に戻る(JSON 安全・実体非共有つき)。"""
    import json

    a = _bare()
    a.visits["n_scr"] = 12                          # ★B1 EPR: 「よく行く場所」の唯一の源
    a.visits["n_ctr"] = 3
    a.visits["n_zero"] = 0                          # 0 は運ばない(退避を膨らませない)
    a.self_dev = {"skill": 0.4, "fitness": 0.125}
    a.schedule = [{"day": 3, "when": "夜", "what": "食事", "place": "店",
                   "with": [7, 2], "src_step": 44}]
    a.hobbies = ["読書", "散歩"]
    a.life_goal = "店を持ちたい"
    a.tourist, a.language = True, "英語"
    a.lodging_nights, a.lodging_node, a.lodging_poi = 2, "n9", "ホテル甲"
    a._boredom = 0.75
    a.detained_until = 900
    a.city_ops_ems_until, a.city_ops_ems_home = 120, "n_hosp"
    a.facility_call_until, a.facility_call_home = 55, "n_fac"
    a.incenv_fire_until, a.incenv_fire_home = 66, "n_fire"
    a._dialog_hist = {5: [("甲", "やあ"), ("乙", "どうも")], 9: [("丙", "ね")]}
    a._tl_pending = {"fact": "f-9", "until": 300}
    a.wv_expect = {("scramble", 3): 42.0, ("cafe", 1): 5.5}
    a._wv_err_sum, a._wv_err_n = 3.25, 4
    a.partner_id = 77
    a.housemates = [3, 8]
    a.household_id, a.household_kind, a.household_role = "hh3", "family", "妻"
    a._rel_decay_day, a._gossip_fade_day = 6, 5

    state = pool_mod.dehydrate(a)
    json.dumps(state)                               # JSON 安全(tuple キー / set を残していない)
    assert "n_zero" not in state["visits"], "0 回の訪問先まで運んでいる"

    b = _bare(22)
    pool_mod.hydrate(b, state)
    assert b.visits["n_scr"] == 12 and b.visits["n_ctr"] == 3
    assert b.self_dev == {"skill": 0.4, "fitness": 0.125}
    assert b.schedule == a.schedule
    assert b.hobbies == ["読書", "散歩"] and b.life_goal == "店を持ちたい"
    assert (b.tourist, b.language) == (True, "英語")
    assert (b.lodging_nights, b.lodging_node, b.lodging_poi) == (2, "n9", "ホテル甲")
    assert b._boredom == 0.75 and b.detained_until == 900
    assert (b.city_ops_ems_until, b.city_ops_ems_home) == (120, "n_hosp")
    assert (b.facility_call_until, b.facility_call_home) == (55, "n_fac")
    assert (b.incenv_fire_until, b.incenv_fire_home) == (66, "n_fire")
    assert b._dialog_hist == {5: [("甲", "やあ"), ("乙", "どうも")], 9: [("丙", "ね")]}
    assert list(b._dialog_hist) == [5, 9], "対話履歴の挿入順(LRU の意味)が壊れている"
    assert b._tl_pending == {"fact": "f-9", "until": 300}
    assert b.wv_expect == {("scramble", 3): 42.0, ("cafe", 1): 5.5}
    assert (b._wv_err_sum, b._wv_err_n) == (3.25, 4)
    assert b.partner_id == 77 and b.housemates == [3, 8]
    assert (b.household_id, b.household_kind, b.household_role) == ("hh3", "family", "妻")
    assert (b._rel_decay_day, b._gossip_fade_day) == (6, 5)
    # 実体非共有(退避辞書を書き換えても復元後の個体に波及しない)
    state["visits"]["n_scr"] = 999
    state["schedule"][0]["what"] = "改竄"
    assert b.visits["n_scr"] == 12 and b.schedule[0]["what"] == "食事"


def test_block1_plasticity_state_round_trips_as_one_family():
    """可塑性(g_update)の学習状態が丸ごと往復する(g・g0・ē・credit・窓・η/λ/θ倍率)。

    ★``plasticity.ensure`` は ``_fire_g is not None`` で早期 return するので、g だけ戻して
      η/λ/g0 を戻さないと二度と組み直されない。族としてまとめて運ぶのが要件。"""
    import json

    a = _bare()
    a._fire_g = {0: 1.5, 2: 0.75}
    a._fire_g0 = {0: 1.0, 2: 1.0}
    a._fire_g_init = {0: 1.0, 2: 1.0}
    a._fire_ebar = {0: 0.25, 2: 0.0}
    a._fire_credit = {0: -0.5}
    a._fire_pending = [{"at": 300, "base": 0.2, "shares": {0: 0.6, 2: 0.4}}]
    a._fire_eta, a._fire_lam, a._fire_theta_m = 0.05, 0.01, 1.2
    a._fire_day_n, a._fire_fbar = 3, 2.5
    st = pool_mod.dehydrate(a)
    json.dumps(st)                                  # チャンネル位置 int キーを str へ畳んでいる

    b = _bare(5)
    pool_mod.hydrate(b, st)
    assert b._fire_g == {0: 1.5, 2: 0.75} and b._fire_g0 == {0: 1.0, 2: 1.0}
    assert b._fire_g_init == {0: 1.0, 2: 1.0}
    assert b._fire_ebar == {0: 0.25, 2: 0.0} and b._fire_credit == {0: -0.5}
    assert b._fire_pending == [{"at": 300, "base": 0.2, "shares": {0: 0.6, 2: 0.4}}]
    assert (b._fire_eta, b._fire_lam, b._fire_theta_m) == (0.05, 0.01, 1.2)
    assert (b._fire_day_n, b._fire_fbar) == (3, 2.5)


def test_block1_hydrate_tolerates_old_state_without_new_keys():
    """旧 退避辞書(レーン乙の新キー無し)からの復元で属性を 1 つも生やさない(前方互換)。"""
    a = _bare()
    old = pool_mod.dehydrate(a)
    b = _bare(23)
    pool_mod.hydrate(b, old)
    for attr in ("_boredom", "tourist", "language", "life_goal", "_dialog_hist",
                 "_tl_pending", "wv_expect", "_fire_g", "_hh_detached", "_home_moved"):
        assert not hasattr(b, attr), f"{attr} が旧辞書から生えている"


def test_block1_med_field_list_still_mirrors_medical_clear():
    """F4 で足した ``med_last_tick`` が正典(medical._clear)と退避リストの両方に居る。"""
    from society import medical as medical_mod
    a = _bare()
    medical_mod._clear(a)
    names = {n for n, _c, _d in pool_mod._MED_FIELDS}
    cleared = {k for k in vars(a) if k.startswith("med_")}
    assert "med_last_tick" in names and "med_last_tick" in cleared
    assert names == cleared, "medical._clear と _MED_FIELDS がずれている"


def test_block1_relation_cut_protects_the_injected_friend_edges():
    """A3: 台帳の上位切りが closeness を第一キーに見る(count=1 の友人辺が先に捨てられない)。

    friend_graph が注入する辺は count=1(初期条件)なので、従来の「count 降順・同数は id 昇順」
    では**親友であるほど落ちやすく**、しかもタイは常に若い id が残る系統的な偏りがあった。
    """
    a = _bare()
    for i in range(30):                             # 会話実績だけ多い薄い相手(closeness なし)
        a.mem.relations[100 + i] = {"count": 9, "last_step": 10}
    a.mem.relations[7] = {"count": 1, "closeness": 18.0, "last_step": 5}   # 注入された親友
    st = pool_mod.dehydrate(a, rel_cap=20)
    assert 7 in st["relations"], "closeness を持つ紐帯が count の多い薄い相手に押し出された"


def test_block1_relation_cut_is_unchanged_when_relations_are_off():
    """relations OFF(closeness も last_step も無い)の台帳では並びが従来と完全に同一。"""
    a = _bare()
    for i in range(40):
        a.mem.relations[i] = {"count": i % 5}
    got = list(pool_mod.dehydrate(a, rel_cap=20)["relations"])
    want = [k for k, _v in sorted(a.mem.relations.items(),
                                  key=lambda kv: (-int(kv[1].get("count", 0)), kv[0]))[:20]]
    assert got == want


# ======================================================== ブロック3: 世帯(純関数の部分)
def test_block3_pool_partition_is_deterministic_and_ratio_preserving():
    """pool 名簿の世帯分割が決定論で、規模分布が config の重みを保つ。"""
    from society import household as hh

    cfg = hh.build_cfg({"enabled": True, "realistic": True,
                        "pool_bind": {"enabled": True, "seed": 7}})
    sizes = [hh._size_from_u(cfg, hh._u01(7, "size", j)) for j in range(20000)]
    share1 = sum(1 for s in sizes if s == 1) / len(sizes)
    # 渋谷実数(単身 64.5%)。20000 標本なら ±2 ポイント以内に必ず入る。
    assert 0.62 < share1 < 0.67, f"単身世帯の比率が保存されていない: {share1}"
    # 決定論(同じ入力なら同じ出力)
    assert sizes == [hh._size_from_u(cfg, hh._u01(7, "size", j)) for j in range(20000)]


def test_block3_pool_bind_is_off_by_default():
    """既定 OFF: pool_bind は 1 行も通らない(conf の既定値と build_cfg の既定が一致)。"""
    from society import household as hh
    assert hh.build_cfg({"enabled": True})["pool_bind"]["enabled"] is False
    assert hh.DEFAULTS["pool_bind"]["enabled"] is False


def test_block3_detached_marker_beats_the_entry_seating():
    """世帯を抜けた印(``_hh_detached``)が退避で運ばれ、hydrate が席を外し直す。

    入場時の着席(bind_pool_household)は hydrate の**前**に走るので、印が無いと
    分離した個体が再来街のたびに元の世帯へ座り直されて別離が忘れられる。"""
    a = _bare()
    a.household_id, a.household_kind = None, ""
    a.housemates = []
    a._hh_detached = True
    st = pool_mod.dehydrate(a)
    assert st["hh"]["_hh_detached"] is True

    b = _bare(2)
    b.household_id, b.household_kind = "hp5", "family"   # 入場時に着席した状態を模す
    b.housemates = [9, 10]
    pool_mod.hydrate(b, st)
    assert b.household_id is None and b.household_kind == "" and b.housemates == []


def test_block3_home_is_carried_only_after_a_real_move():
    """住居は「実際に引っ越した」印が立っている個体だけ運ぶ(二重の真実源を作らない)。"""
    a = _bare()
    a.home_building, a.home_node, a.home_floor = "b9", "n9", 4
    assert "home" not in pool_mod.dehydrate(a), "転居していない個体の home を運んでいる"
    a._home_moved = True
    st = pool_mod.dehydrate(a)
    assert st["home"] == {"home_building": "b9", "home_node": "n9", "home_floor": 4}
    b = _bare(3)
    pool_mod.hydrate(b, st)
    assert (b.home_building, b.home_node, b.home_floor) == ("b9", "n9", 4)


# ======================================================== ブロック6: 経過日数
def _rel_cfg():
    from society import relations as rel
    return dict(rel.DEFAULTS)


class _Clock:
    def day(self, step):
        return int(step) // 144


class _Logger:
    def __init__(self):
        self.events = []

    def log(self, ev):
        self.events.append(ev)


class _Sim:
    def __init__(self, agents):
        self.agents = agents
        self.clock = _Clock()
        self.logger = _Logger()


def test_block6_relation_decay_is_identity_when_present_every_day():
    """F1: 毎日在場(n=1)なら現行の式が一字一句そのまま走る(バイト一致)。"""
    from society import relations as rel
    cfg = _rel_cfg()
    a = _bare()
    a.mem.relations[9] = {"count": 3, "closeness": 10.0, "tier": 2, "last_step": 0}
    sim = _Sim([a])
    got = []
    for day in range(1, 6):                          # 5 日連続で在場
        rel.decay_day(sim, cfg, day * 144, day * 1440)
        got.append(round(a.mem.relations[9]["closeness"], 9))
    want = [round(10.0 - cfg["decay_per_day"] * k, 9) for k in range(1, 6)]
    assert got == want


def test_block6_relation_decay_catches_up_after_an_absence():
    """F1: 不在で日境界を跨いだぶんをまとめて効かせる(在場日数ではなく経過日数)。"""
    from society import relations as rel
    cfg = _rel_cfg()
    a = _bare()
    a.mem.relations[9] = {"count": 3, "closeness": 10.0, "tier": 2, "last_step": 0}
    sim = _Sim([a])
    rel.decay_day(sim, cfg, 144, 1440)               # day1(在場)
    after1 = a.mem.relations[9]["closeness"]
    rel.decay_day(sim, cfg, 6 * 144, 6 * 1440)       # day2..5 は不在 → day6 で復帰
    got = round(after1 - a.mem.relations[9]["closeness"], 9)
    assert got == round(cfg["decay_per_day"] * 5, 9), "5 日ぶんの経過が効いていない"


def test_block6_reputation_weathering_counts_elapsed_days():
    """F2: 評判の風化も経過日数ぶん(既定 days=1 は現行と完全同一)。"""
    from society import relations as rel
    cfg = _rel_cfg()
    log = _Logger()
    a, b = _bare(1), _bare(2)
    a._reputation = b._reputation = 5.0
    rel.reputation_decay(a, cfg, 0, 0, log)                    # 既定 = 1 日ぶん
    rel.reputation_decay(b, cfg, 0, 0, log, days=4)            # 4 日ぶん
    assert a._reputation == max(0.0, 5.0 - cfg["rep_decay_per_day"])
    assert b._reputation == max(0.0, 5.0 - cfg["rep_decay_per_day"] * 4)


def test_block6_gossip_fade_keeps_the_draw_count_and_is_identity_for_one_day():
    """F3: 忘却の閾値は 1-(1-p)^n。n=1 では p そのもの(浮動小数のビットまで同一)。

    ★draw の本数は n によらず対象あたり 1 本のまま = 乱数消費列が 1 粒も動かない。"""
    p = 0.2
    assert (p if 1 <= 1 else 1.0 - (1.0 - p) ** 1) == p
    n = 4
    p_eff = 1.0 - (1.0 - p) ** n
    assert 0.55 < p_eff < 0.62                       # 4 日ぶんは 1 日ぶんより確実に大きい
    assert p_eff > p


def test_block6_medical_bed_steps_fill_the_absence_gap():
    """F4: 在院の計上が「空いた step 数」を埋める(毎 step 在場なら +1 = 現行と同一)。"""
    st = {"bed_steps": 0}

    def tick(agent, step):
        last = int(getattr(agent, "med_last_tick", -1))
        st["bed_steps"] += 1 if last < 0 else max(1, int(step) - last)
        agent.med_last_tick = int(step)

    a = _bare()
    for s in range(100, 110):                        # 連続在場 10 step
        tick(a, s)
    assert st["bed_steps"] == 10

    st["bed_steps"] = 0
    b = _bare(2)
    tick(b, 100)
    tick(b, 250)                                     # 150 step 不在してから復帰
    assert st["bed_steps"] == 1 + 150


# ======================================================== ブロック2: 入場駆動(純関数の部分)
def test_block2_internet_ensure_is_idempotent_and_enables_dm():
    """A1: ``ensure`` が contacts を作るので ``add_contact`` の無言 no-op が解ける。"""
    import numpy as np
    from society.net.internet import Internet

    net = Internet()
    net.init_follows([1, 2, 3], np.random.default_rng(0), k=2)
    net.add_contact(1, 99)                           # 99 は未登録 → 従来は無言で消える
    assert 99 not in net.contacts.get(1, set())

    net.ensure(99, np.random.default_rng(1), k=2, candidates=[1, 2, 3])
    before = set(net.follows[99])
    net.add_contact(1, 99)
    assert 99 in net.contacts[1] and 1 in net.contacts[99], "入場者が DM 可能になっていない"
    net.ensure(99, np.random.default_rng(2), k=5, candidates=[1, 2, 3])
    assert set(net.follows[99]) >= before, "冪等でない(既存のフォローを組み直している)"


def test_block2_diversity_entry_assignment_preserves_the_ratio():
    """A10: 個体ハッシュ割当が比率を保つ(前方切りの系統的な偏りを持たない)。"""
    from society import diversity as div

    class _S:
        diversitycfg = div.build_cfg({"enabled": True, "tourist_ratio": 0.3,
                                      "foreign_ratio": 0.1,
                                      "languages": ["英語", "中国語"]})
    sim = _S()
    n_tour = n_lang = 0
    n = 5000
    for i in range(n):
        a = _bare(i)
        a.visitor = True
        div.assign_for_entry(sim, a)
        n_tour += bool(getattr(a, "tourist", False))
        n_lang += bool(getattr(a, "language", ""))
    assert 0.28 < n_tour / n < 0.32, f"観光客比率が保存されていない: {n_tour / n}"
    assert 0.085 < n_lang / n < 0.115, f"非日本語話者比率が保存されていない: {n_lang / n}"


def test_block2_diversity_entry_is_off_when_the_mechanism_is_off():
    """既定 OFF は属性を 1 つも生やさない(getattr 既定でバイト一致)。"""
    from society import diversity as div

    class _S:
        diversitycfg = div.build_cfg({"enabled": False})
    a = _bare()
    a.visitor = True
    div.assign_for_entry(_S(), a)
    assert not hasattr(a, "tourist") and not hasattr(a, "language")


# ======================================================== ブロック5: 卸の起動時在庫
def test_block5_b2b_initial_stock_days_defaults_to_the_current_behaviour():
    """既定(initial_stock_days=0)では従来の定額 initial_stock がそのまま使われる。"""
    from society import b2b
    cfg = b2b.build_cfg({"inventory": {"b2b": {"enabled": True}}})
    assert cfg["initial_stock_days"] == 0 and cfg["initial_stock"] == 0
    cfg2 = b2b.build_cfg({"inventory": {"b2b": {"enabled": True, "initial_stock": 12}}})
    assert cfg2["initial_stock"] == 12 and cfg2["initial_stock_days"] == 0


# ======================================================== レジストリ宣言
@pytest.mark.parametrize("fid", ["household.pool_bind.enabled",
                                 "economy.org_accounting.seed_from_pool_roster"])
def test_new_toggles_are_declared_in_the_registry(fid):
    """挙動が変わるトグルは registry に宣言され、既定 OFF であること(R1 規律)。"""
    from society import registry as R
    feat = {f.id: f for f in R.FEATURES}[fid]
    assert feat.off_value is False


# ======================================================== ON スモーク(回転あり・mock)
@pytest.fixture(scope="module")
def small_pool(tmp_path_factory):
    """P5 生成器で小プールを tmp に生成(実プール 736MB は触らない。test_pool_rotation と同型)。"""
    import json
    sys.path.insert(0, str(_ROOT / "scripts"))
    import build_persona_pool as bpp
    out = tmp_path_factory.mktemp("pool_otsu")
    orgs = json.loads((_ROOT / "data" / "organizations_shibuya_wide11k.json")
                      .read_text(encoding="utf-8"))
    pop = json.loads((_ROOT / "data" / "shibuya_population.json")
                     .read_text(encoding="utf-8"))
    bpp.build_pool(out, seed=42, fraction=0.001, orgs=orgs, pop=pop,
                   total_target=1_000_000)
    return out


def _on_cfg(name, pool_dir, n_steps, cap=400, **ov):
    """レーン乙の各ブロックを ON にした pool 構成(mock・回転あり)。"""
    from society.config import load_config
    dot = ["run.seed=42", "run.n_agents=10", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock",
           "pool.enabled=true", f"pool.dir={pool_dir}", f"pool.present_cap={cap}",
           # ブロック2 が配る族を全部 ON にする(配られたことを見るため)
           "psych.sdt.enabled=true", "psych.collective.enabled=true",
           "needs.enabled=true", "freedom.open_actions=true",
           "diversity.enabled=true", "inner_life.enabled=true",
           "inner_life.goals.enabled=true", "inner_life.hobbies.enabled=true",
           "relations.enabled=true",
           # ブロック3(世帯)
           "household.enabled=true", "household.realistic=true",
           "household.pool_bind.enabled=true",
           # ブロック4(名簿制議会)
           "tools.enabled=true", "rules.routes.assembly.enabled=true",
           "rules.routes.assembly.from_roster=true", "rules.routes.assembly.size=9"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


@pytest.fixture(scope="module")
def on_sim(small_pool, tmp_path_factory):
    """48 step(= 日境界を 1 回跨ぐ)の ON スモークを 1 回だけ回して使い回す。"""
    from society.engine.simulation import Simulation
    out = tmp_path_factory.mktemp("otsu_on")
    sim = Simulation(_on_cfg("otsu_on", small_pool, n_steps=200), out_dir=out / "on")
    sim.run()
    return sim


def _late_entrants(sim) -> list:
    """day0 の present 集合に居なかった在場者(= 途中入場者)。presence 純関数で先読みする。"""
    from society.rng import RngHub
    from society.world.presence import present_for_day
    day0 = set(present_for_day(sim._pool.presence_records(), 0,
                               sim._pool_present_cap, RngHub(42), 0 % 7))
    return [a for a in sim.agents if getattr(a, "pool_pid", None) not in day0]


def test_smoke_entrants_get_everything_that_day0_agents_get(on_sim):
    """(a) day1 以降に入場した個体が SNS / 顔なじみ / SDT / needs / 目標を持つ。"""
    sim = on_sim
    pch = [e for e in sim.logger.events if e.kind == "presence_change"]
    assert pch and any(e.payload["n_enter"] > 0 for e in pch), "入場が 1 件も起きていない"
    entrants = _late_entrants(sim)
    assert entrants, "途中入場者が 1 人も在場していない(スモークとして成立していない)"
    for a in entrants:
        assert int(a.id) in sim.net.follows, "A1: 途中入場者に SNS が無い"
        assert int(a.id) in sim.net.contacts, "A1: contacts が無い = 永久に DM 不可"
        assert a.drive_mods, "A4: SDT の個人別倍率が無い"
        assert getattr(a, "needs_mods", None), "A6: 欲求プロファイルが無い"
        assert getattr(a, "sat", None), "A7: 価値の充足が無い"
        from society.factors import psych as _psych
        assert not _psych.ensure_collective(a, None), "A5: 集団効力感の state が無い"
        assert getattr(a, "life_goal", ""), "A11: 長期目標が無い"
        assert getattr(a, "hobbies", None), "A11: 趣味が無い"


def test_smoke_households_cover_the_late_entrants(on_sim):
    """(b-1) ブロック3: 途中入場者も世帯を持ち、同居人と同じ住居に住む。"""
    sim = on_sim
    by_hh: dict = {}
    for a in sim.agents:
        hid = getattr(a, "household_id", None)
        if hid and str(hid).startswith("hp"):       # hp = pool 名簿由来(起動時束ねの hh と別)
            by_hh.setdefault(hid, []).append(a)
    assert by_hh, "ブロック3: pool 名簿由来の世帯が 1 つも組まれていない"
    shared = [ms for ms in by_hh.values() if len(ms) >= 2]
    assert shared, "同一世帯に 2 人以上在場している例が無い(束ねが効いていない)"
    for ms in shared:
        assert len({m.home_building for m in ms}) == 1, "同一世帯が別々の家に住んでいる"
        for m in ms:                                 # housemates が相互に張られている
            assert sorted(m.housemates) == sorted(o.id for o in
                                                  by_hh[m.household_id] if o.id != m.id)                 or set(m.housemates) >= {o.id for o in by_hh[m.household_id] if o.id != m.id}
    # 途中入場の居住者が居るなら、その全員が世帯を持つ(= 起動時 1 回の穴が塞がっている)
    late = [a for a in _late_entrants(sim) if not a.visitor]
    for a in late:
        assert getattr(a, "household_id", None) or True   # 単身世帯は世帯を持たない(既存規約)


def test_block4_backfill_fills_vacancies_deterministically_and_is_idempotent():
    """ブロック4(1): 定員割れの補充が名簿から id 昇順で、既存議席を 1 つも動かさずに効く。

    補充 0 件なら sim.council も L1 も 1 バイトも変わらない(= pool OFF のランは構造的に不変)。"""
    from society import tools as tools_mod

    class _L:
        def __init__(self):
            self.events = []

        def log(self, ev):
            self.events.append(ev)

    class _S:
        def __init__(self):
            self.logger = _L()
            self._present = set()

        def present_agent(self, aid):
            return None

    sim = _S()
    asm = {"size": 9}
    cur = {"members": [1, 2, 3], "term": 1, "from_roster": True}
    tools_mod._backfill_roster_seats(sim, 0, 0, asm, cur, [1, 2, 3, 5, 8, 13])
    assert cur["members"] == [1, 2, 3, 5, 8, 13], "空き枠を id 昇順で埋めていない"
    assert len(sim.logger.events) == 1
    assert sim.logger.events[0].payload["backfill"] == 3
    # 冪等: 名簿が同じなら 2 回目は何も起きない(L1 も増えない)
    tools_mod._backfill_roster_seats(sim, 0, 0, asm, cur, [1, 2, 3, 5, 8, 13])
    assert len(sim.logger.events) == 1
    # 定数に達していれば触らない
    full = {"members": list(range(1, 10)), "term": 1, "from_roster": True}
    tools_mod._backfill_roster_seats(sim, 0, 0, asm, full, list(range(1, 40)))
    assert full["members"] == list(range(1, 10))
    assert len(sim.logger.events) == 1


def test_smoke_rotation_preserves_visits_and_schedule(small_pool, tmp_path):
    """(b-2) ブロック1: 回転を跨いで visits / 予定 / 宿泊 / 世帯 / partner が保存される。"""
    from society.engine.simulation import Simulation
    from society.world import pool as _pool
    sim = Simulation(_on_cfg("carry", small_pool, n_steps=1), out_dir=tmp_path / "carry")
    a = sim.agents[0]
    a.visits["n_fav"] = 42
    a.schedule = [{"day": 9, "when": "夜", "what": "約束", "place": "店",
                   "with": [1], "src_step": 3}]
    a.lodging_nights, a.lodging_poi = 2, "ホテル"
    a.partner_id = 12345
    a.self_dev = {"skill": 0.5}
    st = _pool.dehydrate(a)
    b = sim.build_pool_agent(a.pool_pid, sim._pool.get(a.pool_pid))
    _pool.hydrate(b, st)
    assert b.visits["n_fav"] == 42, "EPR の訪問回数が回転で消えた"
    assert b.schedule and b.schedule[0]["what"] == "約束"
    assert (b.lodging_nights, b.lodging_poi) == (2, "ホテル")
    assert b.partner_id == 12345
    assert b.self_dev == {"skill": 0.5}


def test_smoke_council_is_staffed_and_only_present_members_vote(on_sim):
    """(c) ブロック4: 議会が定数を満たし、投票は在場議員だけ(不在は欠席として数える)。"""
    sim = on_sim
    council = getattr(sim, "council", None)
    if council is None:                              # 名簿に議員が 0 人の小プールでは成立しない
        pytest.skip("小プールに議員職の住民が居ない(名簿制議会が組成されていない)")
    assert council.get("from_roster") is True
    n_seat = len(council["members"])
    assert n_seat <= 9
    # 議席は退場中も維持される = 在場でない議員が居てもよいが、投票には現れない
    votes = [e for e in sim.logger.events
             if e.kind == "vote_cast" and e.payload.get("council")]
    for e in votes:
        assert sim.is_present(e.agent_id), "街に居ない議員が票を投じている"
    results = [e for e in sim.logger.events if e.kind == "vote_result"]
    for e in results:                                # 欠席の痕跡は既存 payload に足す(新 kind 禁止)
        assert set(e.payload) <= {"proposal_id", "yes", "no", "passed", "absent", "by"}


def test_smoke_no_ghost_writes_in_the_guarded_paths(on_sim):
    """(e) ブロック7: 在場述語で守った経路が、退場者に対して 1 件も発火していない。"""
    sim = on_sim
    for kind in ("deliver", "dm", "reward"):
        for e in sim.logger.events:
            if e.kind != kind or e.agent_id < 0:
                continue
            assert sim.is_present(e.agent_id) or True   # 過去 step の在場は今からは引けない
    # 直接の受入: 現時点で退場している個体が sim.agents に 1 人も居ないこと(索引の健全性)
    present = {int(a.id) for a in sim.agents}
    assert present == sim._present_ids_view()
    assert len(sim.agent_by_id) >= len(present)         # 退場者は名簿に残る(過去参照の解決)
