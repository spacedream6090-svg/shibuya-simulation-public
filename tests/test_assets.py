"""所有権レイヤー O1(登記簿)+ O3(相続)(``world.assets``)のテスト。

正典
  - docs/plans/ownership-layer-plan.md **§1**(AssetLedger = 権利行 + 双方向索引 /
    資産保存則 = 貨幣保存則 IF-E の完全な双対 / 階層 LoD)・**§2 の O1・O3 行**・
    **§5 ユーザー決定 3 件**(①住戸の初期所有者に域内の不動産会社も加える /
    ②相続 = 承認 / ③本線前に実装)
  - docs/research/ownership-asset-models.md §4-2(行を「所有」ではなく**権利**にする)・
    §4-3(移転語彙と資産保存則)・§5-2(推奨 = タグ付きレコードを登記簿に置く)

守るもの(検収基準の順)
  ① OFF(既定)= ゴールデン L1 バイト一致・キー不発生・state も属性も生えない・
     **死者の財布は凍結のまま**(第107 と完全同値)
  ② ON: 初期配賦(住戸 = 持ち家 / 域内不動産 org / 域外 RoW・車両 = has_car の昇格)が立つ
  ③ ★**資産保存則**: Σ(所有者別保有数) + K5 − RoW 生成 = 初期ストック(カテゴリ別)
  ④ ★**相続(O3)**: 現金 + 資産行が世帯へ移り、**貨幣保存が破れない**。相続人不存在は
     国庫(RoW)へ出して IF-E の閉じた不変量を保つ
  ⑤ 既存イベントの写像(持ち家者の転居 = 売却)が**金を 1 円も動かさない**
  ⑥ 未分類の移転種の監視装置が武装している
  ⑦ ON 同 seed 2 ラン一致 / resume == straight(登記簿が checkpoint に載る)
  ⑧ LLM 呼数 ON/OFF 完全一致 + **プロンプトが 1 バイトも変わらない**
  ⑨ ★静的検査: 新 stream は "asset_alloc" 1 本だけ・全対全スキャンを足していない
  ⑩ 凍結 14 ファイルを 1 バイトも触っていない
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from society import assets as A
from society import economy_sfc as SFC
from society import registry as R
from society.config import load_config
from society.engine import checkpoint, scheduler
from society.engine.simulation import Simulation
from society.observer import causality as C
from society.observer.schema import EVENT_KINDS, Event

REPO = Path(__file__).resolve().parents[1]
GOLDEN = Path(__file__).resolve().parent / "data" / "golden_baseline_l1.json"

# test_lost_property.py:45 / test_traces.py:45 と同じ「意図的な既定挙動追加」の中立化
_GOLDEN_NEUTRAL = {"economy.wages.自営": 0, "planning.enabled": "false",
                   "transit_ride.taxi.enabled": "false",
                   "transit_ride.bus.enabled": "false", "rules.enabled": "false"}

OFF = {"world.assets.enabled": "false"}
ON = {"world.assets.enabled": "true", "organizations.enabled": "true"}
ON_HH = dict(ON, **{"household.enabled": "true"})

ASSET_KINDS = ("inheritance", "asset_transfer")


# --------------------------------------------------------------------------- #
# 共通ヘルパ(test_lost_property.py と同型)
# --------------------------------------------------------------------------- #
def _cfg(name, n_steps=24, n_agents=15, **ov):
    dot = ["run.seed=42", f"run.n_agents={n_agents}", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock", "observer.snapshot_every=144"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def _sim(tmp_path, name, n_steps=24, n_agents=15, **ov):
    return Simulation(_cfg(name, n_steps, n_agents, **ov), out_dir=tmp_path / name)


def _armed(tmp_path, name, n_agents=20, **ov):
    """``arm`` だけを走らせた登記簿(ラン全体を回さない = 速い)。"""
    sim = _sim(tmp_path, name, n_steps=1, n_agents=n_agents, **dict(ON_HH, **ov))
    scheduler._ensure_orgs(sim)
    A.arm(sim, 0, 0)
    return sim


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _l1x(sim):
    return [[e.step, e.agent_id, e.kind, e.llm_call_id,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _kind(sim, kind):
    return [e for e in sim.logger.events if e.kind == kind]


def _cash(sim):
    return sum(float(a.money) + float(getattr(a, "account", 0.0) or 0.0)
               for a in sim.agents)


def _die(sim, agent, step=1, sim_min=10):
    """故人を作る(**health を通さない**= 相続の写像だけを隔離して見る)。"""
    agent.dead = True
    sim.logger.log(Event(step=int(step), sim_min=int(sim_min), agent_id=int(agent.id),
                         kind="death", x=float(agent.x), y=float(agent.y),
                         payload={"cause": "cardiac"}))


class _FixedLLM:
    """**プロンプト非依存**の巡回応答スタブ(test_lost_property と同型)。"""

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


# =========================================================================== #
# (A) 出荷既定・宣言(検収基準 ①の前段)
# =========================================================================== #
def test_shipped_default_is_off():
    cfg = load_config()
    assert bool(cfg.world.assets.enabled) is False
    # 較正値が config と実装で食い違っていない(較正の単一の源)
    assert float(cfg.world.assets.owner_occupancy_rate) == A.OWNER_OCCUPANCY_RATE
    assert float(cfg.world.assets.org_landlord_share) == A.ORG_LANDLORD_SHARE
    assert str(cfg.world.assets.landlord_industry_key) == A.LANDLORD_INDUSTRY_KEY
    assert tuple(cfg.world.assets.landlord_sectors) == A.LANDLORD_SECTORS
    assert list(cfg.world.assets.right_kinds) == ["own"]   # O1 の最小核は own 1 種


def test_calibration_anchors_are_in_the_right_direction():
    """★渋谷は借家の街(持ち家は 3 割台)・家主の法人比率は少数派、という較正の主張。"""
    assert 0.25 <= A.OWNER_OCCUPANCY_RATE <= 0.40, "持ち家率が都平均(≈45%)側に寄っている"
    assert 0.05 <= A.ORG_LANDLORD_SHARE <= 0.35, "法人家主が多数派になっている(実測は個人 8 割)"


def test_registry_and_schema_declared():
    feat = {f.id: f for f in R.FEATURES}["world.assets.enabled"]
    assert feat.repro_tier == "strict"
    assert feat.affects_k is False       # generate() の呼び出しサイトを 1 つも足さない
    # ★観測層に完全に閉じている(プロンプトの行も節も 1 つも増えない)
    assert feat.fingerprint_risk == "none"
    assert feat.off_value is False
    assert R.undeclared_toggles(load_config()) == [], "未宣言の bool トグルがある"
    for kind in ASSET_KINDS:
        assert kind in EVENT_KINDS, f"{kind} が L1 スキーマに未登録"
        assert kind in C.CAUSE_OF_KIND, f"{kind} が因果台帳に未分類"
        # 相続も登記の書換も「制度が世界状態に反応して」発火する(分類の原則 2)
        assert C.CAUSE_OF_KIND[kind] == C.DEVICE, kind


def test_timeconv_declares_the_only_per_step_key():
    """Δt 変換表に「件/step」の安全弁だけが載っている(比率・個数は Δt 非依存)。"""
    from society import timeconv as T
    keys = {pat for pat, _kind, _desc in T.TABLE if pat.startswith("world.assets.")}
    assert keys == {"world.assets.max_events_per_step"}, keys


def test_vocabularies_are_finite_and_disjoint():
    """権利種・カテゴリ・移転語彙が有限表で、名前空間が衝突していない。"""
    assert A.OWN in A.RIGHT_KINDS and len(set(A.RIGHT_KINDS)) == len(A.RIGHT_KINDS)
    assert set(A.CATEGORIES) == {A.DWELLING, A.VEHICLE}
    assert len(set(A.TRANSFER_KINDS)) == len(A.TRANSFER_KINDS)
    # 研究文書 §4-3 の 9 種 + 相続人不存在(民法 959 条)= 10 語
    for word in ("sale", "gift", "inheritance", "escheat", "lease", "lien",
                 "lost", "theft", "born", "scrap"):
        assert word in A.TRANSFER_KINDS, word
    # party 型は 3 つで、綴りは前置詞で分かれている
    assert A.party_kind(A.agent_party(7)) == "agent"
    assert A.party_kind(A.org_party("co_re_1")) == "org"
    assert A.party_kind(A.row_party()) == "row"
    assert A.party_agent_id(A.agent_party(7)) == 7
    assert A.party_agent_id(A.org_party("co_re_1")) is None


def test_unknown_config_values_degrade_to_defaults():
    cfg = A.build_cfg({"enabled": True, "owner_occupancy_rate": 3.0,
                       "org_landlord_share": -1.0, "max_landlord_orgs": -5,
                       "right_kinds": ["lease", "存在しない権利"]})
    assert cfg["enabled"] is True
    assert cfg["owner_occupancy_rate"] == 1.0     # [0,1] へクリップ
    assert cfg["org_landlord_share"] == 0.0
    assert cfg["max_landlord_orgs"] == 0
    # own は消せない(登記簿の最小核)/ 表に無い権利は黙って捨てる
    assert cfg["right_kinds"][0] == A.OWN
    assert "存在しない権利" not in cfg["right_kinds"]
    assert A.LEASE in cfg["right_kinds"]


# --------------------------------------------------------------------------- #
# (A-2) ★静的検査(検収基準 ⑧⑨⑩)
# --------------------------------------------------------------------------- #
def test_module_adds_exactly_one_stream_and_no_llm():
    """新乱数は "asset_alloc" 1 本だけ・generate() の呼び出しサイトはゼロ。"""
    src = Path(A.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert "generate" not in (names | attrs), "LLM を撃っている"
    assert "llm" not in attrs
    used = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "stream" and node.args
                and isinstance(node.args[0], ast.Constant)):
            used.add(node.args[0].value)
    assert used == {"asset_alloc"}, f"新 stream が 1 本ではない: {sorted(used)}"


def test_module_never_scans_all_pairs():
    """agent × agent の入れ子走査を 1 つも足していない(25 万スケールの規律)。"""
    tree = ast.parse(Path(A.__file__).read_text(encoding="utf-8"))
    for outer in ast.walk(tree):
        if not isinstance(outer, ast.For) or "agents" not in ast.dump(outer.iter):
            continue
        for inner in ast.walk(outer):
            if inner is not outer and isinstance(inner, ast.For):
                assert "agents" not in ast.dump(inner.iter), "全対全スキャンを足している"


def test_module_never_writes_to_agents_except_cash():
    """★世界状態のうち本 module が書くのは**現金だけ**(位置・関係・記憶を触らない)。

    agent への代入(``x.attr = …`` / ``setattr(x, "attr", …)``)を AST で全数走査する。
    """
    tree = ast.parse(Path(A.__file__).read_text(encoding="utf-8"))
    allowed = {"money", "account", "dead"}
    written: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if (isinstance(tgt, ast.Attribute)
                        and isinstance(tgt.value, ast.Name)
                        and tgt.value.id in ("agent", "h", "a", "owner", "finder")):
                    written.add(tgt.attr)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "setattr" and len(node.args) == 3
                and isinstance(node.args[1], ast.Constant)):
            written.add(str(node.args[1].value))
    assert written <= allowed, f"現金以外の agent 属性を書いている: {sorted(written - allowed)}"


def test_frozen_metric_spec_files_are_untouched():
    """凍結 14 ファイル(metrics_spec.SPEC_FILES)に本レーンの痕跡が 1 つも無い。"""
    from society.observer import metrics_spec as MS
    assert len(MS.SPEC_FILES) == 14
    for rel in MS.SPEC_FILES:
        text = (REPO / rel).read_text(encoding="utf-8")
        for word in ("world.assets", "asset_alloc", "inheritance", "asset_transfer"):
            assert word not in text, f"凍結ファイル {rel} に O1 の痕跡がある"


# =========================================================================== #
# (B) OFF = バイト一致(検収基準 ①)
# =========================================================================== #
def test_off_matches_golden(tmp_path):
    golden = json.load(open(GOLDEN, encoding="utf-8"))
    sim = _sim(tmp_path, "as_golden", n_steps=144, n_agents=15, **_GOLDEN_NEUTRAL)
    sim.run()
    assert _l1(sim) == golden, "O1 の seam がゴールデンを動かしている"


def test_off_matches_pure_default(tmp_path):
    pure = _sim(tmp_path, "as_pure", n_steps=48, n_agents=12)
    pure.run()
    off = _sim(tmp_path, "as_off", n_steps=48, n_agents=12, **OFF)
    off.run()
    assert _l1x(pure) == _l1x(off)


def test_off_emits_nothing_and_grows_no_state(tmp_path):
    """OFF では arm / phase を直接叩いても L1 も state も所持金も 1 も動かない。"""
    sim = _sim(tmp_path, "as_off_noop", n_steps=1)
    money = [float(a.money) for a in sim.agents]
    n_events = len(sim.logger.events)
    for step in range(10):
        A.arm(sim, step, 420 + step * 10)
        A.phase(sim, step, 420 + step * 10)
    new = sim.logger.events[n_events:]
    assert not [e for e in new if e.kind in ASSET_KINDS]
    assert A.state_of(sim) is None
    assert getattr(sim, "_ledger_state", None) is None
    assert A.provenance(sim) is None
    assert [float(a.money) for a in sim.agents] == money


def test_off_keeps_the_dead_wallet_frozen(tmp_path):
    """★OFF = 第107 と完全同値(死者の財布は凍結されたまま = 相続が起きない)。"""
    sim = _sim(tmp_path, "as_off_dead", n_steps=1, n_agents=8, **OFF)
    victim = sim.agents[0]
    victim.money = 50_000.0
    _die(sim, victim)
    A.phase(sim, 1, 10)
    assert float(victim.money) == 50_000.0, "OFF なのに遺産が動いた"
    assert not _kind(sim, "inheritance")
    assert A.state_of(sim) is None


def test_off_summary_and_sidecar_have_no_key(tmp_path):
    sim = _sim(tmp_path, "as_sum_off", n_steps=24, n_agents=10)
    sim.run()
    js = json.loads((tmp_path / "as_sum_off" / "summary.json").read_text(encoding="utf-8"))
    assert "assets" not in js
    assert not (tmp_path / "as_sum_off" / "assets_ledger.json").exists()


# =========================================================================== #
# (C) ON: 初期配賦(検収基準 ②)
# =========================================================================== #
@pytest.fixture(scope="module")
def armed(tmp_path_factory):
    """``arm`` だけを走らせた登記簿(初期配賦の検収材料)。"""
    d = tmp_path_factory.mktemp("as_arm")
    sim = Simulation(_cfg("as_arm", 1, 20, **ON_HH), out_dir=d / "as_arm")
    scheduler._ensure_orgs(sim)
    A.arm(sim, 0, 0)
    return sim


def test_arm_registers_every_dwelling_and_vehicle(armed):
    """住戸 =(住宅系建物, 階)の全個体・車両 = has_car の個体昇格。"""
    sim = armed
    st = A.state_of(sim)
    expect_dw = sum(max(1, int(b.get("levels", 1) or 1))
                    for b in sim.city.residential_buildings)
    expect_vh = sum(1 for a in sim.agents if a.has_car)
    assert st["stock0"][A.DWELLING] == expect_dw, "住戸の列挙が (建物, 階) と食い違う"
    assert st["stock0"][A.VEHICLE] == expect_vh, "has_car の昇格が漏れている"
    assert len(st["assets"]) == expect_dw + expect_vh
    # 資産 id は決定論の綴り(逆引きできる)
    b0 = sorted(sim.city.residential_buildings, key=lambda b: str(b["id"]))[0]
    assert A.dwelling_id(b0["id"], 1) in st["assets"]


def test_every_asset_has_exactly_one_owner_row(armed):
    """★行は「所有」ではなく権利。O1 の資産はどれも ``own`` 行をちょうど 1 本持つ。"""
    st = A.state_of(armed)
    for aid, row in st["rows"].items():
        assert set(row) == {A.OWN}, (aid, sorted(row))
        assert A.party_kind(row[A.OWN]["party"]) in ("agent", "org", "row")


def test_bidirectional_index_is_consistent(armed):
    """双方向索引(by_owner / by_asset)が食い違わない = 逆引きも順引きも O(1)。"""
    sim = armed
    st = A.state_of(sim)
    n = 0
    for party, holds in st["by_owner"].items():
        for aid in holds:
            assert A.owner_of(sim, aid) == party, (party, aid)
            n += 1
    assert n == len(st["rows"]), "索引の件数が権利行の数と合わない"
    party0 = sorted(st["by_owner"])[0]
    assert A.assets_of(sim, party0) == sorted(st["by_owner"][party0])


def test_vehicles_are_owned_by_their_driver(armed):
    sim = armed
    st = A.state_of(sim)
    for a in sim.agents:
        aid = A.vehicle_id(a.id)
        if a.has_car:
            assert st["assets"][aid]["cat"] == A.VEHICLE
            assert A.owner_of(sim, aid) == A.agent_party(a.id)
        else:
            assert aid not in st["assets"], "has_car でない個体に車が生えている"


def test_rental_owner_mix_matches_the_calibration(armed):
    """★賃貸住戸の所有者 = **域内不動産 org + 域外 RoW のミックス**(ユーザー決定 §5-1)。

    法人家主 ≈ 2 割(国交省調査の「個人 8 割」の裏返し)の帯に実現値が入る。
    分母は 1,500 戸超あるので、この帯は乱数の運では通らない。
    """
    st = A.state_of(armed)
    n_org = int(st["alloc"].get(A.ALLOC_ORG, 0))
    n_row = int(st["alloc"].get(A.ALLOC_ROW, 0))
    assert n_org > 0 and n_row > 0, "ミックスになっていない(片側に全振り)"
    share = n_org / (n_org + n_row)
    assert 0.15 <= share <= 0.25, f"域内 org の家主比率 {share:.3f} が較正帯の外"


def test_owner_occupancy_is_machine_fixed_at_both_extremes(tmp_path):
    """持ち家率は**居住者が居る住戸にだけ**効く(1.0 で全員持ち家・0.0 で誰も持ち家でない)。"""
    for rate, want_resident in ((1.0, True), (0.0, False)):
        sim = _armed(tmp_path, f"as_occ_{rate}",
                     **{"world.assets.owner_occupancy_rate": str(rate)})
        st = A.state_of(sim)
        occupied = {(str(a.home_building), int(a.home_floor)) for a in sim.agents
                    if not a.visitor and a.home_building}
        assert occupied, "テスト前提が崩れた(誰も住戸に住んでいない)"
        got = {A.party_kind(A.owner_of(sim, A.dwelling_id(b, f)))
               for b, f in occupied if A.dwelling_id(b, f) in st["assets"]}
        if want_resident:
            assert got == {"agent"}, got
        else:
            assert "agent" not in got, got


def test_landlord_orgs_come_from_the_census_industry_key(armed):
    """★域内不動産 org の特定は**台帳の実在フィールドだけ**で決まる(仮定が要らない)。"""
    sim = armed
    st = A.state_of(sim)
    assert st["org_source"] == "census_industry_key", st["org_source"]
    assert st["landlords"], "家主 org が 1 社も見つかっていない"
    for oid in st["landlords"]:
        org = sim.orgs[oid]
        assert org["industry_key"] == A.LANDLORD_INDUSTRY_KEY
        assert org["sector_detail"] in A.LANDLORD_SECTORS


def test_census_book_actually_contains_the_landlord_sectors():
    """本番台帳(9,872 社)に家主の担い手が実在する(綴り違いで永久に 0 社を防ぐ)。"""
    path = REPO / "data" / "organizations_shibuya_census.json"
    if not path.exists():                          # 台帳未生成の環境ではスキップ
        pytest.skip("census 台帳が無い")
    book = json.loads(path.read_text(encoding="utf-8"))["companies"]
    re_orgs = [c for c in book if c.get("industry_key") == A.LANDLORD_INDUSTRY_KEY]
    landlords = [c for c in re_orgs if c.get("sector_detail") in A.LANDLORD_SECTORS]
    assert len(re_orgs) > 500, len(re_orgs)
    assert 300 <= len(landlords) <= len(re_orgs), len(landlords)
    # オフィス仲介・物品賃貸は**住戸の家主ではない**ので外れている
    excluded = {c.get("sector_detail") for c in re_orgs} - set(A.LANDLORD_SECTORS)
    assert excluded, "RE の全区分を家主にしてしまっている(除外が効いていない)"


def test_landlord_fallback_is_declared_honestly(tmp_path):
    """sector_detail を持たない台帳では office へ後退し、**仮定であることを残す**。"""
    sim = _sim(tmp_path, "as_fallback", n_steps=1, n_agents=6, **ON_HH)
    sim.orgs = {"co_x_1": {"id": "co_x_1", "industry_key": "IT",
                           "size": {"employees": 5},
                           "workplace_poi": {"cat": "office"}}}
    cfg = A.cfg_of(sim)
    landlords, source = A._landlord_orgs(sim, cfg)
    assert source == "office_fallback" and [o for o, _ in landlords] == ["co_x_1"]
    # org 台帳そのものが無い世界では家主は空 = 賃貸は全部 RoW(現行と同じ絵)
    sim.orgs = {}
    assert A._landlord_orgs(sim, cfg) == ([], "none")


def test_landlord_weight_is_by_employees(tmp_path):
    """家主 org の中の配り方は**従業者数の重み付き**(均等割りは所有ネットワークを殺す)。"""
    sim = _sim(tmp_path, "as_weight", n_steps=1, n_agents=6, **ON_HH)
    sim.orgs = {
        "co_re_big": {"id": "co_re_big", "industry_key": "RE",
                      "sector_detail": "賃貸仲介/管理", "size": {"employees": 300}},
        "co_re_small": {"id": "co_re_small", "industry_key": "RE",
                        "sector_detail": "開発/PM", "size": {"employees": 1}},
    }
    landlords, cum, _src = A._landlord_index(sim, A.cfg_of(sim))
    assert dict(landlords) == {"co_re_big": 300.0, "co_re_small": 1.0}
    picks = [A._pick_landlord(landlords, cum, u / 1000.0) for u in range(1000)]
    assert picks.count("co_re_big") > picks.count("co_re_small") * 20


def test_max_assets_cap_is_declared(tmp_path):
    """安全弁で打ち切ったことを黙らない(capped=true が provenance に出る)。"""
    sim = _armed(tmp_path, "as_cap", **{"world.assets.max_assets": "50"})
    st = A.state_of(sim)
    assert len(st["assets"]) == 50 and st["capped"] is True
    assert A.provenance(sim)["capped"] is True


# =========================================================================== #
# (D) ★資産保存則(検収基準 ③ = 貨幣保存則 IF-E の双対)
# =========================================================================== #
def test_conservation_holds_at_allocation(armed):
    """Σ(所有者別保有数) + K5 − RoW 生成 = 初期ストック(カテゴリ別・残差ゼロ)。"""
    cons = A.conservation(armed)
    assert cons, "台帳が空(テスト前提が崩れた)"
    for cat, c in cons.items():
        assert c["residual"] == 0, (cat, c)
        # O1 では製造も廃棄も起こさない = 両項は常に 0(O2 で初めて動く)
        assert c["born"] == 0 and c["k5"] == 0, (cat, c)


def test_transfers_never_create_or_destroy_rows(tmp_path):
    """★移転は**付け替えだけ**(行は生まれも消えもしない)= 保存則の意味そのもの。"""
    sim = _armed(tmp_path, "as_cons_tr")
    st = A.state_of(sim)
    before = A.conservation(sim)
    aids = sorted(st["assets"])[:20]
    for k, aid in enumerate(aids):                 # 個人 → 組織 → RoW と順に回す
        A.transfer(sim, aid, A.agent_party(k % 5), A.GIFT, 1)
        A.transfer(sim, aid, A.row_party(), A.SALE, 2)
    after = A.conservation(sim)
    assert after == before, "移転で保存則が動いた"
    for cat, c in after.items():
        assert c["residual"] == 0, (cat, c)


def test_unclassified_transfer_kinds_are_monitored(tmp_path):
    """★未分類の移転種の監視装置(IF-E の双対)が武装している。"""
    sim = _armed(tmp_path, "as_unc")
    st = A.state_of(sim)
    aid = sorted(st["assets"])[0]
    owner0 = A.owner_of(sim, aid)
    # 同じ party への「移転」は**何も起きない**(数えない = 動いていない物を動いたと言わない)
    assert A.transfer(sim, aid, owner0, "宇宙からの授かり物", 1) == owner0
    assert st["unclassified"] == {}
    A.transfer(sim, aid, A.agent_party(999), "宇宙からの授かり物", 1)
    assert st["unclassified"] == {"宇宙からの授かり物": 1}
    assert A.provenance(sim)["unclassified"] == {"宇宙からの授かり物": 1}
    # 正規の語彙は transfers 側へ入る
    A.transfer(sim, aid, A.agent_party(1), A.SALE, 2)
    assert st["transfers"][A.SALE] == 1


def test_transfer_of_unknown_asset_is_a_no_op(tmp_path):
    sim = _armed(tmp_path, "as_unknown")
    assert A.transfer(sim, "dw:存在しない:1", A.row_party(), A.SALE, 1) is None
    assert A.owner_of(sim, "dw:存在しない:1") is None


# =========================================================================== #
# (E) ★相続 O3(検収基準 ④)
# =========================================================================== #
def _household_victim(sim):
    for a in sim.agents:
        if getattr(a, "housemates", None):
            return a
    return None


def test_inheritance_moves_money_and_assets_to_the_household(tmp_path):
    """★死亡時に故人の現金 + 資産行が世帯構成員へ移る(第107 の観測盲点の解消)。"""
    sim = _armed(tmp_path, "as_inherit", n_agents=24)
    st = A.state_of(sim)
    victim = _household_victim(sim)
    assert victim is not None, "テスト前提が崩れた(世帯が 1 つも組まれていない)"
    heirs = [sim.agent_by_id[i] for i in victim.housemates]
    aid = sorted(st["assets"])[0]
    A.transfer(sim, aid, A.agent_party(victim.id), A.GIFT, 0)
    victim.money = 30_000.0
    heir_before = sum(float(h.money) for h in heirs)
    _die(sim, victim)
    A.phase(sim, 1, 10)
    assert float(victim.money) == 0.0, "死者の財布が凍結されたまま"
    assert sum(float(h.money) for h in heirs) == pytest.approx(
        heir_before + 30_000.0, abs=1e-6)
    assert A.party_agent_id(A.owner_of(sim, aid)) in {h.id for h in heirs}
    ev = _kind(sim, "inheritance")
    assert len(ev) == 1 and ev[0].agent_id == victim.id
    assert ev[0].payload["to"] == "household"
    assert ev[0].payload["heirs"] == len(heirs)
    assert ev[0].payload["assets"] >= 1
    assert ev[0].payload["absent"] == 0          # 全員が街に居た


def test_absent_household_members_are_not_silently_lost(tmp_path):
    """★街に居ない世帯員(プール退場中)は相続人になれないが、**人数を必ず残す**。

    ★退場の再現は**本物と同じ形**にしてある(レーン甲 2026-08-13): ``_phase_pool_rotation`` は
      ``sim.agents`` を present だけに差し替え、``agent_by_id`` からは**消さない**。以前この
      テストは ``agent_by_id.pop`` で退場を模しており、名簿が退場者を保持するという実装の
      事実(= 幽霊書き込みの温床)を回避してしまっていた。
    """
    sim = _armed(tmp_path, "as_absent", n_agents=24)
    victim = _household_victim(sim)
    gone = sim.agent_by_id[victim.housemates[0]]
    sim.agents = [a for a in sim.agents if a.id != gone.id]   # プール退場を再現
    sim.invalidate_present_index()
    assert sim.agent_by_id.get(gone.id) is gone, "名簿は退場者を保持する(実装の事実)"
    victim.money = 4_000.0
    _die(sim, victim)
    A.phase(sim, 1, 10)
    ev = _kind(sim, "inheritance")[-1]
    assert ev.payload["absent"] == 1, ev.payload
    assert A.provenance(sim)["heirs_absent"] == 1
    if ev.payload["heirs"] == 0:                      # 受け皿が全員居ない = 国庫へ
        assert ev.payload["to"] == "row"
        assert A.cash_escheated(sim) == pytest.approx(4_000.0, abs=1e-6)
    assert float(gone.money) >= 0.0                   # 退場者には 1 円も書かない


def test_money_is_conserved_through_inheritance(tmp_path):
    """★街の総現金は 1 円も増減しない(家計 → 家計の内部移転)。"""
    sim = _armed(tmp_path, "as_inh_cons", n_agents=24)
    victim = _household_victim(sim)
    victim.money = 77_777.0
    total0 = _cash(sim)
    _die(sim, victim)
    A.phase(sim, 1, 10)
    assert _cash(sim) == pytest.approx(total0, abs=1e-6)
    assert A.cash_escheated(sim) == 0.0, "世帯が居るのに国庫へ出ている"


def test_heirless_estate_goes_to_the_state_and_closes_the_invariant(tmp_path):
    """★相続人不存在 = **国庫へ帰属**(民法 959 条)。IF-E の閉じた不変量が破れない。"""
    sim = _armed(tmp_path, "as_escheat", n_agents=16,
                 **{"economy.org_accounting.enabled": "true"})
    st = A.state_of(sim)
    lone = next(a for a in sim.agents if not getattr(a, "housemates", None))
    aid = sorted(st["assets"])[3]
    A.transfer(sim, aid, A.agent_party(lone.id), A.GIFT, 0)
    lone.money = 9_000.0
    before = SFC.total_money(sim)
    _die(sim, lone)
    A.phase(sim, 1, 10)
    assert float(lone.money) == 0.0
    assert A.cash_escheated(sim) == pytest.approx(9_000.0, abs=1e-6)
    assert A.owner_of(sim, aid) == A.row_party(), "資産が国庫(RoW)へ行っていない"
    ev = _kind(sim, "inheritance")[-1]
    assert ev.payload["to"] == "row" and ev.payload["heirs"] == 0
    # 閉じた不変量 city + RoW + K5 が保たれる(名前のある RoW チャネルで開示)
    assert SFC.total_money(sim) == pytest.approx(before, abs=1e-6)
    assert "inheritance_escheat" in SFC.CHANNELS_OUT
    assert SFC.provenance(sim)["row_channels"]["inheritance_escheat"]["out"] == \
        pytest.approx(9_000.0, abs=1e-6)


def test_escheat_channel_is_declared_and_disjoint():
    """RoW の窓口は宣言済みの有限表に閉じている(K5 と名前空間が衝突しない)。"""
    assert "inheritance_escheat" in SFC.CHANNELS
    assert not (set(SFC.CHANNELS) & set(SFC.K5_KINDS))
    assert "inheritance_escheat" not in SFC.CHANNELS_IN


def test_inheritance_is_idempotent(tmp_path):
    """同じ死を 2 度相続しない(watermark が resume で 0 に戻っても二重にならない)。"""
    sim = _armed(tmp_path, "as_idem", n_agents=24)
    victim = _household_victim(sim)
    victim.money = 5_000.0
    total0 = _cash(sim)
    _die(sim, victim)
    A.phase(sim, 1, 10)
    sim._ledger_watermark = 0                      # resume 直後の状態を再現
    A.phase(sim, 2, 20)
    assert len(_kind(sim, "inheritance")) == 1, "同じ死で 2 度相続している"
    assert _cash(sim) == pytest.approx(total0, abs=1e-6)


def test_split_never_loses_a_yen():
    """等分は端数を最後の 1 人が引き取る(合計が 1 円もずれない)。"""
    for total, n in ((10.0, 3), (12345.0, 2), (1.0, 7), (0.0, 4)):
        parts = A._split(total, n)
        assert sum(parts) == pytest.approx(total, abs=1e-9), (total, n, parts)


# =========================================================================== #
# (F) 既存イベントの写像(検収基準 ⑤)
# =========================================================================== #
def test_owner_occupier_move_sells_to_a_landlord(tmp_path):
    """持ち家者の転居 = own 移転(買い手 = 域内 org / RoW)。★金は 1 円も動かない。"""
    sim = _armed(tmp_path, "as_sale", n_agents=16)
    st = A.state_of(sim)
    mover = sim.agents[2]
    aid = A.dwelling_id(mover.home_building, mover.home_floor)
    if aid not in st["assets"]:                    # 住戸が地図に無い個体は使わない
        aid = sorted(st["assets"])[7]
        A.transfer(sim, aid, A.agent_party(mover.id), A.GIFT, 0)
    else:
        A.transfer(sim, aid, A.agent_party(mover.id), A.GIFT, 0)
    cash0 = _cash(sim)
    mover.home_building = "b_new_home"             # 転居した(もうそこには住んでいない)
    mover.home_floor = 1
    sim.logger.log(Event(step=2, sim_min=20, agent_id=int(mover.id),
                         kind="move_home", x=0.0, y=0.0, payload={}))
    A.phase(sim, 2, 20)
    ev = _kind(sim, "asset_transfer")
    assert ev and ev[-1].payload["kind"] == A.SALE
    assert ev[-1].payload["from"] == A.agent_party(mover.id)
    assert A.party_kind(ev[-1].payload["to"]) in ("org", "row")
    assert A.owner_of(sim, aid) == ev[-1].payload["to"]
    assert _cash(sim) == pytest.approx(cash0, abs=1e-6), "売買代金を動かしている"
    assert A.conservation(sim)[A.DWELLING]["residual"] == 0


def test_tenant_move_touches_the_ledger_zero_times(tmp_path):
    """賃借人の転居は台帳を 1 行も動かさない(own は家主のままだから)。"""
    sim = _armed(tmp_path, "as_tenant", n_agents=16)
    st = A.state_of(sim)
    tenant = next(a for a in sim.agents
                  if not A.assets_of(sim, A.agent_party(a.id)))
    rows0 = {aid: dict(r[A.OWN]) for aid, r in st["rows"].items()}
    tenant.home_building = "b_new_home"
    sim.logger.log(Event(step=2, sim_min=20, agent_id=int(tenant.id),
                         kind="move_home", x=0.0, y=0.0, payload={}))
    A.phase(sim, 2, 20)
    assert not _kind(sim, "asset_transfer")
    assert {aid: dict(r[A.OWN]) for aid, r in st["rows"].items()} == rows0


def test_owner_who_stays_home_does_not_sell(tmp_path):
    """まだそこに住んでいる持ち家者は、転居イベントが来ても売らない。"""
    sim = _armed(tmp_path, "as_stay", n_agents=16)
    st = A.state_of(sim)
    owner = sim.agents[1]
    aid = A.dwelling_id(owner.home_building, owner.home_floor)
    if aid not in st["assets"]:
        pytest.skip("この個体の住戸が地図に無い")
    A.transfer(sim, aid, A.agent_party(owner.id), A.GIFT, 0)
    sim.logger.log(Event(step=2, sim_min=20, agent_id=int(owner.id),
                         kind="move_home", x=0.0, y=0.0, payload={}))
    A.phase(sim, 2, 20)
    assert not _kind(sim, "asset_transfer")
    assert A.owner_of(sim, aid) == A.agent_party(owner.id)


# =========================================================================== #
# (G) 決定論・resume・LLM 呼数(検収基準 ⑦⑧)
# =========================================================================== #
def test_on_is_deterministic_across_two_runs(tmp_path):
    a = Simulation(_cfg("as_det_a", 48, 20, **ON_HH), out_dir=tmp_path / "a")
    a.run()
    b = Simulation(_cfg("as_det_b", 48, 20, **ON_HH), out_dir=tmp_path / "b")
    b.run()
    sa, sb = A.state_of(a), A.state_of(b)
    assert sa["rows"] == sb["rows"]
    assert sa["alloc"] == sb["alloc"]
    assert [round(float(x.money), 6) for x in a.agents] == \
        [round(float(x.money), 6) for x in b.agents]


def test_llm_call_count_and_prompts_are_identical_on_and_off(tmp_path):
    """★呼数だけでなく**プロンプトのバイト列**まで一致する(観測層に閉じている)。"""
    acts = [json.dumps({"action": "stay"}, ensure_ascii=False),
            json.dumps({"action": "speak", "text": "こんにちは"}, ensure_ascii=False)]
    got = []
    for name, ov in (("as_k_off", OFF), ("as_k_on", ON_HH)):
        sim = Simulation(_cfg(name, 48, 12, **ov), out_dir=tmp_path / name)
        sim.llm = _FixedLLM(acts)
        sim.run()
        got.append((sim.llm.calls, sim.llm.prompts))
    assert got[0][0] == got[1][0], f"LLM 呼数が ON/OFF で違う: {got[0][0]} vs {got[1][0]}"
    assert got[0][1] == got[1][1], "プロンプトが 1 バイト変わっている(観測層を出ている)"


def test_resume_matches_straight(tmp_path):
    """登記簿 ON で resume==straight(台帳が checkpoint に載る・初期配賦が二重に走らない)。"""
    ov = {**ON_HH, "run.start_tod": "00:00", "run.natural_start": "true"}
    split, total = 24, 48
    straight_dir = tmp_path / "as_straight"
    straight = Simulation(_cfg("as_straight", total, 16, **ov), out_dir=straight_dir)
    straight.run()

    d = tmp_path / "as_resumed"
    sim1 = Simulation(_cfg("as_resumed", split, 16, **ov,
                           **{"observer.checkpoint_every": split}), out_dir=d)
    for step in range(split):
        scheduler.run_step(sim1, step)
    checkpoint.save(sim1, split, d / "checkpoint" / f"ckpt-{split:06d}.pkl.gz")
    sim1.logger.flush_segment()
    sim2 = Simulation(_cfg("as_resumed", total, 16, **ov,
                           **{"observer.checkpoint_every": split}), out_dir=d)
    sim2.run(resume_from=d)
    for stem in ("l1_events", "l2_metrics", "l3_snapshots"):
        sa = pq.read_table(straight_dir / f"{stem}.parquet").to_pylist()
        sb = pq.read_table(d / f"{stem}.parquet").to_pylist()
        assert sa == sb, f"{stem} 不一致(world.assets resume)"
    ja = json.loads((straight_dir / "summary.json").read_text(encoding="utf-8"))
    jb = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    assert ja["assets"] == jb["assets"], "登記簿のタリーが resume で straight と食い違う"
    assert A.state_of(straight)["rows"] == A.state_of(sim2)["rows"]
    assert A.state_of(sim2)["init"] is True, "resume で初期配賦がもう一度走っている"


# =========================================================================== #
# (H) 観測(summary / サイドカー / 解析スクリプト)
# =========================================================================== #
@pytest.fixture(scope="module")
def run_on(tmp_path_factory):
    d = tmp_path_factory.mktemp("as_run")
    sim = Simulation(_cfg("as_on", 48, 20, **ON_HH), out_dir=d / "as_on")
    sim.run()
    return sim, d / "as_on"


def test_summary_publishes_conservation_and_allocation(run_on):
    sim, out = run_on
    js = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    prov = js["assets"]
    assert prov["schema"] == A.SCHEMA
    for key in ("categories", "right_kinds", "n_assets", "stock0", "alloc",
                "conservation", "transfers", "unclassified", "org_source",
                "n_landlord_orgs", "by_party_kind", "holder_gini", "deaths",
                "heirs_absent", "inherited_assets", "inherited_money",
                "escheat_assets", "escheat_money", "sales", "capped"):
        assert key in prov, key
    assert prov["right_kinds"] == ["own"], "O1 の最小核が own 1 種でない"
    assert prov["unclassified"] == {}, "未分類の移転種が出た"
    for cat, c in prov["conservation"].items():
        assert c["residual"] == 0, (cat, c)
    # 所有の集中(agent / org / RoW を同列に数えた Gini)が退化していない
    assert 0.0 < prov["holder_gini"] < 1.0


def test_ledger_sidecar_is_written_and_self_consistent(run_on):
    sim, out = run_on
    led = json.loads((out / "assets_ledger.json").read_text(encoding="utf-8"))
    assert led["schema"] == A.SCHEMA
    assert led["n_rows"] == len(led["rows"]) == len(A.state_of(sim)["rows"])
    for r in led["rows"][:50]:
        assert r["right"] == A.OWN and r["cat"] in A.CATEGORIES
        assert A.party_kind(r["party"]) in ("agent", "org", "row")
    counts = {}
    for r in led["rows"]:
        counts[r["cat"]] = counts.get(r["cat"], 0) + 1
    assert counts == led["stock0"], "サイドカーの行数が初期ストックと食い違う"


def test_analyze_script_reproduces_the_conservation_check(run_on, tmp_path):
    """解析スクリプトが台帳から**独立に**数え直して同じ結論に達する。"""
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    import analyze_assets as AA
    _sim_obj, out = run_on
    ledger = AA.load_ledger(out)
    res = AA.summarize(ledger, AA.load_transfers(out), AA.load_last_cash(out),
                       last_step=47)
    for cat, c in res["conservation"].items():
        assert c["residual"] == 0, (cat, c)
    assert res["conservation_agrees"] is True
    assert res["network"]["n_assets"] == ledger["n_rows"]
    assert 0.0 <= res["inequality"]["gini_cash_only"] <= 1.0
    # ★住宅資産を入れると資産格差の見え方が変わる(この層が解く問題そのもの)
    assert res["inequality"]["gini_cash_plus_assets"] != \
        res["inequality"]["gini_cash_only"]
    assert AA.render(res).strip()


def test_analyze_script_reports_missing_ledger(tmp_path):
    """OFF のランには登記簿が無い = 解析対象なしと**正直に**言う(0 と偽らない)。"""
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    import analyze_assets as AA
    assert AA.load_ledger(tmp_path) is None


# =========================================================================== #
# (K) O4 権利行 = 家賃の受け手を own 行の家主へ(第114 レーン 優先2)
#
# 何が壊れていたか: 家賃は**誰が家主でも一律に域外(RoW)へ**出ていた。O1 で住戸 11,948 戸と
# 域内不動産 org 487 社を登記したのに、賃料は 1 円も域内で循環していなかった。
# O4 は登記簿に lease 行(住戸 × 借主)を並べ、引落の受け手を own 行の家主へ解決する。
# =========================================================================== #
LEASE_ON = dict(ON_HH, **{"world.assets.lease.enabled": "true",
                          "economy.accounts.enabled": "true",
                          "economy.org_accounting.enabled": "true",
                          "economy.accounts.payday_dom": 1})
#: 家賃日 = payday(1) の翌日 = dom 2 = block_day 1(step 144)。test_accounts と同じ形。
_RENT_STEP = 144
_RENT_MIN = 420 + 144 * 10


def _leased(sim):
    """(借主 agent, 住戸 id, 家主 party) の一覧(id 昇順)。"""
    out = []
    for party, aids in sorted(A.leases(sim).items()):
        tenant = sim.present_agent(A.party_agent_id(party))
        for aid in aids:
            if tenant is not None:
                out.append((tenant, aid, A.owner_of(sim, aid, A.OWN)))
    return out


def _charge_rent(sim, tenant, income=100000.0, balance=100000.0):
    """その 1 人だけに家賃を発生させて日境界を 1 回回す(他は period_income=0 = 家賃 0)。"""
    tenant.period_income, tenant.account, tenant.rent_due = income, balance, 0.0
    sim._acct_day = 0
    scheduler._phase_accounts_day(sim, step=_RENT_STEP, sim_min=_RENT_MIN)
    return [e for e in sim.logger.events if e.kind == "rent"][-1]


def test_lease_is_off_by_default(tmp_path):
    """既定 OFF: lease 行が 1 本も立たず、right_kinds も own だけ(O1 と完全同一)。"""
    sim = _armed(tmp_path, "o4_off", n_agents=30)
    assert A.cfg_of(sim)["lease"]["enabled"] is False
    assert A.lease_on(sim) is False
    assert tuple(A.cfg_of(sim)["right_kinds"]) == (A.OWN,)
    assert A.leases(sim) == {}
    st = A.state_of(sim)
    assert all(set(row) == {A.OWN} for row in st["rows"].values())
    assert "lease" not in st["alloc"]
    # 家主解決も走らない(呼んでも None = 呼び出し側は従来の RoW 経路へ落ちる)
    assert all(A.landlord_of(sim, a) is None for a in sim.agents)


def test_lease_rows_stand_beside_own_without_inflating_the_stock(tmp_path):
    """ON: 賃貸中の住戸に lease 行が own と**並ぶ**。戸数は 1 も増えない(保存則が生きる)。"""
    sim = _armed(tmp_path, "o4_rows", n_agents=40,
                 **{"world.assets.lease.enabled": "true"})
    st = A.state_of(sim)
    lz = A.leases(sim)
    assert lz, "賃貸中の住戸が 1 戸も無い(検査の前提が崩れている)"
    assert A.LEASE in A.cfg_of(sim)["right_kinds"], "行を書くのに宣言に無い"
    for party, aids in lz.items():
        assert A.party_kind(party) == "agent"            # 借主は必ず個人
        for aid in aids:
            row = st["rows"][aid]
            assert set(row) == {A.OWN, A.LEASE}
            assert row[A.OWN]["party"] != party, "自分に貸している"
            assert st["assets"][aid]["cat"] == A.DWELLING and \
                st["assets"][aid]["occupied"] is True
    # ★資産保存則: lease 行は**戸数に数えない**ので残差 0 のまま
    for cat, c in A.conservation(sim).items():
        assert c["residual"] == 0, (cat, c)
    # ★保有数(Gini の材料)も own 行だけ = 借主の部屋は借主の資産ではない
    hold = A.holdings(sim)
    for party in lz:
        assert party not in hold or all(
            A.owner_of(sim, aid, A.OWN) == party for aid in A.assets_of(sim, party)
            if st["rows"][aid].get(A.OWN, {}).get("party") == party)
    assert sum(hold.values()) == sum(c["live"] for c in A.conservation(sim).values())


def test_rent_reaches_an_org_landlord(tmp_path):
    """★塞いだ穴そのもの: 域内不動産 org が家主なら家賃はその org の預金へ入る。"""
    sim = _armed(tmp_path, "o4_org", n_agents=40, **LEASE_ON)
    cands = [(t, aid, own) for t, aid, own in _leased(sim)
             if A.party_kind(own) == "org"]
    assert cands, "域内 org が家主の賃貸住戸が 1 戸も無い(前提崩れ)"
    tenant, _aid, owner = cands[0]
    oid = owner.split(":", 1)[1]
    before = SFC.org_balance(sim, oid)
    ev = _charge_rent(sim, tenant)
    paid = float(ev.payload["paid"])
    assert paid > 0.0
    assert ev.payload["payee"] == oid, ev.payload
    assert abs(SFC.org_balance(sim, oid) - (before + paid)) < 1e-6
    prov = A.provenance(sim)
    assert prov["rent_to"]["org"] == 1 and prov["rent_to"]["org_yen"] == round(paid, 1)


def test_rent_reaches_an_individual_landlord(tmp_path):
    """相続などで個人が家主になった住戸の家賃は、その人の口座へ入る。"""
    sim = _armed(tmp_path, "o4_person", n_agents=40, **LEASE_ON)
    tenant, aid, _own = _leased(sim)[0]
    landlord = next(a for a in sim.agents
                    if not a.visitor and a.id != tenant.id
                    and sim.present_agent(a.id) is not None)
    # 相続で own 行が個人へ移った状態を作る(移転の唯一の口を通す)
    A.transfer(sim, aid, A.agent_party(landlord.id), A.INHERIT, 0)
    landlord.account = 0.0
    ev = _charge_rent(sim, tenant)
    paid = float(ev.payload["paid"])
    assert paid > 0.0
    assert ev.payload["payee"] == f"household:{landlord.id}"
    assert abs(landlord.account - paid) < 1e-6
    assert A.provenance(sim)["rent_to"]["agent"] == 1


def test_owner_occupier_keeps_the_old_route(tmp_path):
    """持ち家(own 行が本人)は従来どおり = 既存の動力学を 1 バイトも変えない。"""
    sim = _armed(tmp_path, "o4_owner", n_agents=40, **LEASE_ON)
    st = A.state_of(sim)
    owners = [a for a in sim.agents
              if not a.visitor and A.dwelling_of(a) in st["rows"]
              and A.owner_of(sim, A.dwelling_of(a), A.OWN) == A.agent_party(a.id)]
    assert owners, "持ち家の居住者が 1 人も居ない(前提崩れ)"
    a = owners[0]
    assert A.landlord_of(sim, a) is None
    ev = _charge_rent(sim, a)
    assert ev.payload["payee"] == "row:rent_landlord"     # IF-E2 の従来経路のまま
    assert "agent" not in (A.provenance(sim)["rent_to"] or {})


def test_absent_landlord_falls_back_to_row(tmp_path):
    """家主が街に居ないなら幽霊へは書かず RoW へ落とす(レーン甲の事故を繰り返さない)。"""
    sim = _armed(tmp_path, "o4_absent", n_agents=40, **LEASE_ON)
    tenant, aid, _own = _leased(sim)[0]
    ghost = 10 ** 7                                      # 名簿に居ない id = 常に不在
    A.transfer(sim, aid, A.agent_party(ghost), A.INHERIT, 0)
    ev = _charge_rent(sim, tenant)
    assert ev.payload["payee"] == "row:rent_landlord"
    assert A.provenance(sim)["rent_to"]["absent"] == 1


def test_rent_conserves_the_total_money(tmp_path):
    """★お金は 1 円も増えない: 受け手が変わっても閉じた不変量は動かない。"""
    sim = _armed(tmp_path, "o4_cons", n_agents=40, **LEASE_ON)
    cands = [(t, aid, own) for t, aid, own in _leased(sim)
             if A.party_kind(own) == "org"]
    tenant = cands[0][0]
    tenant.period_income, tenant.account, tenant.rent_due = 100000.0, 100000.0, 0.0
    before = SFC.total_money(sim)
    sim._acct_day = 0
    scheduler._phase_accounts_day(sim, step=_RENT_STEP, sim_min=_RENT_MIN)
    assert abs(SFC.total_money(sim) - before) < 1e-6


def test_analyze_accounting_classifies_the_new_rent_destinations():
    """部門行列: 家賃が org / 家計 / 域外 のどれに着いたかで分類される(族は rent のまま)。"""
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    import analyze_accounting as AA
    ctx = {"government_on": False, "tax_src": {}}

    def one(payee):
        got = AA.flows_for("rent", {"amount": 300.0, "paid": 300.0, "payee": payee}, ctx)
        assert len(got) == 1
        return got[0]

    assert AA.party_sector("household:7") == AA.HOUSEHOLD
    org = one("co_re_00001")
    assert (org.src, org.dst, org.tag) == (AA.HOUSEHOLD, AA.ORG, "rent:org_landlord")
    hh = one("household:7")
    assert (hh.src, hh.dst, hh.tag) == (AA.HOUSEHOLD, AA.HOUSEHOLD,
                                        "rent:household_landlord")
    row = one("row:rent_landlord")
    assert (row.src, row.dst, row.tag) == (AA.HOUSEHOLD, AA.EXTERNAL,
                                           "rent:row_landlord")
    none = one(None)
    assert (none.src, none.dst, none.tag) == (AA.HOUSEHOLD, AA.VOID, "rent:no_landlord")
    # ★漏れの**族**は 4 つとも "rent" のまま = 監視テストが見ている族集合は増えない
    assert {AA.leak_family(f.tag) for f in
            (org, hh, row, none)} == {"rent"}


def test_finals_conf_turns_the_lease_layer_on():
    """本選 conf が O4 を宣言している(第114 レーン 優先2 の決定を conf で固定)。"""
    from omegaconf import OmegaConf
    fin = OmegaConf.load(REPO / "conf" / "finals_observe.yaml")
    assert bool(fin.world.assets.lease.enabled) is True
