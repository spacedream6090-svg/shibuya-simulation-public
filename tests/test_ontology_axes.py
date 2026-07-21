"""群のオントロジー拡張(Wave2・第41バッチ 2026-07-21)のテスト。

追加機構(既存単軸=文化圏は後方互換のまま=test_ontology.py 21本 green を維持):
- 直交する第2軸(方式B): ontology.axes.<name> 配下の groups/composition/bands/seed_offset。
      軸ごとに独立ハッシュ(seed + seed_offset)で割当=文化圏軸と真に直交(軸間の割当相関=偶然)。
- 情報行動軸(info_behavior): agent.age(人口統計 P0)がある時は年代条件付き構成比、age 不明は全体分布。
- 防災訓練経験(方式A): 文化圏群の drill_line を別行として付加(既定 line は不変・drill.enabled で入/切)。
- プロンプト注入: 文化圏の経験行に続けて軸行+訓練行を append(軸ごとに1行)。
- 計測: agents.json に ontology_axes(軸別群 id)。analyze_groups.py --axis で群別集計の軸を選ぶ。

R1 doctrine 継承: 割当は persona id + age(P0)のみ=traits/k 非参照・追加 LLM 呼ゼロ・乱数ゼロ。
既定 OFF=属性なし=バイト一致。検証は mock のみ(実LLM 禁止・≤288 step)。
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "src"))

from society import ontology                              # noqa: E402
from society.cognition import deliberate                  # noqa: E402
from society.config import load_config                    # noqa: E402
from society.engine.simulation import Simulation          # noqa: E402
from society.world import pool as pool_mod                # noqa: E402
from society.world.presence import present_for_day        # noqa: E402
from society.rng import RngHub                            # noqa: E402

_ON = {"ontology.enabled": "true"}
_OFF = {"ontology.enabled": "false"}


def _sim(tmp_path, name, n=20, steps=24, **ov):
    dot = ["run.seed=42", f"run.n_agents={n}", f"run.n_steps={steps}",
           f"run.name={name}", "model.backend=mock"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _cfg(enabled=True, drill=True):
    """axes(info_behavior)+ bands + drill を持つ cfg。文化圏軸は _oc 相当の4群。"""
    return ontology.build_cfg({
        "enabled": enabled, "seed": 20260721,
        "groups": {
            "jp_metro":   {"label": "a", "line": "L_metro", "drill_line": "D_metro"},
            "jp_other":   {"label": "b", "line": "L_other", "drill_line": "D_other"},
            "asia_visit": {"label": "c", "line": "L_asia", "drill_line": "D_asia"},
            "west_visit": {"label": "d", "line": "L_west", "drill_line": "D_west"},
        },
        "composition": {
            "stochastic": {"jp_metro": 0.45, "jp_other": 0.40,
                           "asia_visit": 0.10, "west_visit": 0.05},
            "default": {"jp_metro": 0.8, "jp_other": 0.2},
        },
        "drill": {"enabled": drill},
        "axes": {
            "info_behavior": {
                "seed_offset": 101,
                "groups": {
                    "info_sns":      {"label": "s", "line": "AX_sns"},
                    "info_mixed":    {"label": "m", "line": "AX_mixed"},
                    "info_official": {"label": "o", "line": "AX_official"},
                },
                "bands": {
                    0:  {"info_sns": 0.60, "info_mixed": 0.30, "info_official": 0.10},
                    30: {"info_sns": 0.45, "info_mixed": 0.35, "info_official": 0.20},
                    45: {"info_sns": 0.30, "info_mixed": 0.35, "info_official": 0.35},
                    60: {"info_sns": 0.15, "info_mixed": 0.20, "info_official": 0.65},
                },
                "composition": {
                    "default": {"info_sns": 0.45, "info_mixed": 0.30, "info_official": 0.25}},
            },
        },
    })


# =============================================================== 純関数(単体)
def test_build_cfg_backcompat_unchanged():
    """axes/drill を足しても文化圏軸の cfg(groups/composition)は現行と同一構造(後方互換)。"""
    oc = _cfg()
    assert set(oc["groups"]) == {"jp_metro", "jp_other", "asia_visit", "west_visit"}
    assert oc["composition"]["default"] == {"jp_metro": 0.8, "jp_other": 0.2}
    # 単軸のみ(axes 無し)cfg は axes 空・drill OFF=現行と割当同一
    single = ontology.build_cfg({"enabled": True, "seed": 20260721,
                                 "groups": {"jp_metro": {"label": "a", "line": "L"}},
                                 "composition": {"default": {"jp_metro": 1.0}}})
    assert single["axes"] == {} and single["drill_enabled"] is False
    assert ontology.assign_group(single, "P_1", "default") == "jp_metro"


def test_assign_axis_deterministic_pure():
    """assign_axis は (pid, tier, age) のみの純関数(同入力=同一群・run.seed を引数に取らない)。"""
    oc = _cfg()
    for pid in ("L4_1", "abc", "42", "L2_00099"):
        g = ontology.assign_axis(oc, "info_behavior", pid, "stochastic", 25)
        assert g == ontology.assign_axis(oc, "info_behavior", pid, "stochastic", 25)
        assert g in oc["axes"]["info_behavior"]["groups"]


def test_assign_axis_disabled_or_unknown_returns_none():
    assert ontology.assign_axis(_cfg(enabled=False), "info_behavior", "x", "default", 25) is None
    assert ontology.assign_axis(_cfg(), "__nope__", "x", "default", 25) is None


def test_axis_age_band_composition_converges():
    """年代条件付き構成比に大数で収束(±0.02)。若年=SNS 優位・高齢=公式優位(内閣府勾配)。"""
    oc = _cfg()
    N = 40000
    for age, exp in ((25, {"info_sns": 0.60, "info_mixed": 0.30, "info_official": 0.10}),
                     (65, {"info_sns": 0.15, "info_mixed": 0.20, "info_official": 0.65})):
        c = Counter(ontology.assign_axis(oc, "info_behavior", f"P_{i}", "stochastic", age)
                    for i in range(N))
        frac = {k: v / N for k, v in c.items()}
        for g, p in exp.items():
            assert abs(frac.get(g, 0.0) - p) < 0.02, f"age={age} {g}: {frac.get(g)} vs {p}"


def test_axis_age_unknown_uses_overall():
    """age 不明(None)は composition の全体分布に収束(bands を使わない)。"""
    oc = _cfg()
    N = 40000
    c = Counter(ontology.assign_axis(oc, "info_behavior", f"P_{i}", "stochastic", None)
                for i in range(N))
    frac = {k: v / N for k, v in c.items()}
    assert abs(frac["info_sns"] - 0.45) < 0.02
    assert abs(frac["info_mixed"] - 0.30) < 0.02
    assert abs(frac["info_official"] - 0.25) < 0.02


def test_band_for_age_boundaries():
    """_band_for_age は下限キー最大の ≤ age を選ぶ(境界=下限は当該 band)。"""
    bands = _cfg()["axes"]["info_behavior"]["bands"]
    assert ontology._band_for_age(bands, 29)["info_sns"] == 0.60
    assert ontology._band_for_age(bands, 30)["info_sns"] == 0.45   # 境界=当該 band
    assert ontology._band_for_age(bands, 59)["info_sns"] == 0.30
    assert ontology._band_for_age(bands, 60)["info_official"] == 0.65
    assert ontology._band_for_age(bands, 15)["info_sns"] == 0.60   # 下端 0 が全若年を捕捉


def test_axes_orthogonal_independence():
    """直交性: 同一 pid で文化圏軸と情報行動軸の割当が独立(同時分布≈周辺積・偏差<0.01)。

    文化圏=_stable_uniform(seed, pid)、情報=_stable_uniform(seed+101, pid)。独立ハッシュ=無相関。"""
    oc = _cfg()
    N = 60000
    cult, info = [], []
    for i in range(N):
        pid = f"P_{i}"
        cult.append(ontology.assign_group(oc, pid, "stochastic"))
        info.append(ontology.assign_axis(oc, "info_behavior", pid, "stochastic", 25))
    cm, im = Counter(cult), Counter(info)
    jm = Counter(zip(cult, info))
    max_dev = 0.0
    for c in cm:
        for j in im:
            joint = jm.get((c, j), 0) / N
            prod = (cm[c] / N) * (im[j] / N)
            max_dev = max(max_dev, abs(joint - prod))
    assert max_dev < 0.01, f"軸間の割当が独立でない(max_dev={max_dev:.4f})"


def test_axis_line_and_drill_line():
    oc = _cfg()
    assert ontology.axis_line(oc, "info_behavior", "info_sns") == "AX_sns"
    assert ontology.axis_line(oc, "info_behavior", None) is None
    assert ontology.axis_line(oc, "__nope__", "info_sns") is None
    assert ontology.group_drill_line(oc, "jp_metro") == "D_metro"
    # drill.enabled=false は訓練行を出さない(割当=文化圏群は不変)
    assert ontology.group_drill_line(_cfg(drill=False), "jp_metro") is None


# =============================================================== OFF=属性なし・バイト一致
def test_off_no_axis_attributes(tmp_path):
    """OFF は文化圏軸だけでなく axes/drill 属性も1つも作らない(既定 OFF=完全 no-op)。"""
    off = _sim(tmp_path, "offax", **_OFF)
    off.run()
    assert all(getattr(a, "ontology_axes", None) is None for a in off.agents)
    assert all(getattr(a, "ontology_axis_lines", None) is None for a in off.agents)


def test_off_matches_pure_default_with_axes(tmp_path):
    """axes/drill を含む現行 config でも OFF は純粋既定と L1 バイト一致(ゴールデン維持)。"""
    pure = _sim(tmp_path, "pureax", n=15, steps=144)
    pure.run()
    off = _sim(tmp_path, "offax2", n=15, steps=144, **_OFF)
    off.run()
    assert _l1(pure) == _l1(off)


# =============================================================== プロンプト注入(2行以上)
def test_two_line_injection_order(tmp_path):
    """文化圏の経験行に続けて軸行+訓練行を注入(persona 直後の連続ブロック・順序保存)。"""
    sim = _sim(tmp_path, "twoline", steps=1)
    sim.run()
    a = sim.agents[0]
    a.ontology_line = "文化圏の経験行。"
    a.ontology_axis_lines = ["情報行動の行。", "訓練経験の行。"]
    p = deliberate.build_prompt(a, place_name="路上", surprise="solo",
                                nearby_names=[], sim_min=600, step=3)
    assert (p.index(a.persona) < p.index("文化圏の経験行。")
            < p.index("情報行動の行。") < p.index("訓練経験の行。"))


def test_on_prompt_has_axis_and_drill_lines(tmp_path):
    """ON の mock 24step で、実発火プロンプトに情報行動軸の行と訓練経験行が実際に載る。"""
    sim = _sim(tmp_path, "axappear", n=40, **_ON)

    class _Rec:
        def __init__(self, inner):
            self._inner = inner
            self.prompts = []

        def generate(self, prompt, **kw):
            self.prompts.append(prompt)
            return self._inner.generate(prompt, **kw)

        @property
        def calls(self):
            return self._inner.calls

        @property
        def hits(self):
            return self._inner.hits
    sim.llm = _Rec(sim.llm)
    sim.run()
    oc = sim.ontologycfg
    axlines = {ontology.axis_line(oc, "info_behavior", g)
               for g in oc["axes"]["info_behavior"]["groups"]}
    drills = {ontology.group_drill_line(oc, g) for g in oc["groups"]}
    drills.discard(None)
    assert any(any(l in p for l in axlines) for p in sim.llm.prompts), \
        "発火プロンプトに情報行動軸の行が一度も載っていない"
    assert any(any(d in p for d in drills) for p in sim.llm.prompts), \
        "発火プロンプトに訓練経験行が一度も載っていない"


# =============================================================== 計測(agents.json / analyze)
def test_on_agents_json_records_axes(tmp_path):
    """ON は agents.json に ontology_axes(軸別群 id)を記録。OFF はキーを持たない。"""
    on = _sim(tmp_path, "axjson", n=40, **_ON)
    on.run()
    meta = json.loads((on.out_dir / "agents.json").read_text(encoding="utf-8"))
    got = [m for m in meta if "ontology_axes" in m]
    assert got, "ontology_axes が1体も記録されていない"
    groups = on.ontologycfg["axes"]["info_behavior"]["groups"]
    for m in got:
        assert m["ontology_axes"]["info_behavior"] in groups
    off = _sim(tmp_path, "axjson_off", n=40, **_OFF)
    off.run()
    meta_off = json.loads((off.out_dir / "agents.json").read_text(encoding="utf-8"))
    assert all("ontology_axes" not in m for m in meta_off)


def test_analyze_groups_axis_selection(tmp_path):
    """analyze_groups.load_group_map(--axis): 既定=文化圏 / 軸指定=ontology_axes[軸](未設定体は除外)。"""
    import analyze_groups
    rows = [
        {"id": 0, "ontology_group": "jp_metro",
         "ontology_axes": {"info_behavior": "info_sns"}},
        {"id": 1, "ontology_group": "west_visit",
         "ontology_axes": {"info_behavior": "info_official"}},
        {"id": 2, "ontology_group": "jp_metro"},          # 軸未設定の個体
    ]
    (tmp_path / "agents.json").write_text(json.dumps(rows, ensure_ascii=False),
                                          encoding="utf-8")
    culture, _ = analyze_groups.load_group_map(str(tmp_path), None)
    assert culture == {0: "jp_metro", 1: "west_visit", 2: "jp_metro"}
    info, _ = analyze_groups.load_group_map(str(tmp_path), "info_behavior")
    assert info == {0: "info_sns", 1: "info_official"}    # id2 は軸未設定=除外


def test_analyze_groups_axis_defs(tmp_path):
    """load_group_defs(--axis): 文化圏は ontology.groups、軸指定は axes.<axis>.groups からラベル解決。"""
    import yaml

    import analyze_groups
    doc = {"ontology": {
        "groups": {"jp_metro": {"label": "都市圏在住の日本人"}},
        "axes": {"info_behavior": {"groups": {"info_sns": {"label": "SNS一次情報を頼る人"}}}}}}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(doc, allow_unicode=True),
                                          encoding="utf-8")
    dc = analyze_groups.load_group_defs(str(tmp_path), None)
    assert dc.get("jp_metro", {}).get("label") == "都市圏在住の日本人"
    da = analyze_groups.load_group_defs(str(tmp_path), "info_behavior")
    assert da.get("info_sns", {}).get("label") == "SNS一次情報を頼る人"


# =============================================================== R1: 呼数不変
def test_axis_call_count_k_invariant(tmp_path):
    """真の R1: axes ON のまま k(writeback)を free/off に振っても LLM 呼数が不変。"""
    free = _sim(tmp_path, "axk_free", **{**_ON, "k.writeback": "free"})
    free.run()
    off = _sim(tmp_path, "axk_off", **{**_ON, "k.writeback": "off"})
    off.run()
    assert len(free.logger.llm_calls) == len(off.logger.llm_calls), \
        f"axes ON の呼数が k で乖離(R1 違反): free={len(free.logger.llm_calls)} off={len(off.logger.llm_calls)}"


def test_axis_on_deterministic(tmp_path):
    """axes ON 同士 2 回で L1 完全一致(mock・乱数ゼロの決定論=軸割当も hashlib 純関数)。"""
    a = _sim(tmp_path, "axdet_a", **_ON)
    a.run()
    b = _sim(tmp_path, "axdet_b", **_ON)
    b.run()
    assert _l1(a) == _l1(b), "axes ON の決定論が崩れている(軸が乱数を引いている疑い)"


# =============================================================== pool 経路(P3 hydrate)
@pytest.fixture(scope="module")
def small_pool(tmp_path_factory):
    """P5 生成器で小プールを tmp に生成(実プール 736MB は触らない)。test_ontology と同流儀。"""
    import build_persona_pool as bpp
    out = tmp_path_factory.mktemp("pool_ax")
    orgs = json.loads((_ROOT / "data" / "organizations_shibuya_wide11k.json")
                      .read_text(encoding="utf-8"))
    pop = json.loads((_ROOT / "data" / "shibuya_population.json")
                     .read_text(encoding="utf-8"))
    bpp.build_pool(out, seed=42, fraction=0.001, orgs=orgs, pop=pop,
                   total_target=1_000_000)
    return out


def _pool_cfg(name, pool_dir, n_steps, cap=400, **ov):
    dot = ["run.seed=42", "run.n_agents=10", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock",
           "pool.enabled=true", f"pool.dir={pool_dir}", f"pool.present_cap={cap}"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def test_pool_axis_matches_pure_function(small_pool, tmp_path):
    """pool ON: 全 present 個体の軸群=assign_axis(pid, tier, age)の純関数値(age=record の年齢)。"""
    sim = Simulation(_pool_cfg("apf", small_pool, n_steps=1, **_ON), out_dir=tmp_path / "apf")
    oc = sim.ontologycfg
    assert sim.agents
    seen = 0
    for a in sim.agents:
        rec = sim._pool.get(a.pool_pid)
        tier = str(rec.get("presence", "resident"))
        expect = ontology.assign_axis(oc, "info_behavior", a.pool_pid, tier, a.age)
        assert a.ontology_axes.get("info_behavior") == expect
        seen += 1
    assert seen > 0


def test_pool_hydrate_reentry_same_axis(small_pool, tmp_path):
    """P3 再来街(dormant→hydrate)でも情報行動軸の群が不変(pid+age 純関数=age は record 安定)。"""
    ps = pool_mod.PoolStore(small_pool)
    recs = ps.presence_records()
    hub = RngHub(42)
    s0 = set(present_for_day(recs, 0, 400, hub, 0))
    s1 = set(present_for_day(recs, 1, 400, hub, 1))
    entrants = sorted(s1 - s0)
    assert entrants, "day0 不在→day1 present の再来街候補が見つからない"
    x = entrants[0]

    sim = Simulation(_pool_cfg("ahy", small_pool, n_steps=210, **_ON), out_dir=tmp_path / "ahy")
    oc = sim.ontologycfg
    rec = ps.get(x)
    tier = str(rec.get("presence", "resident"))
    expect = ontology.assign_axis(oc, "info_behavior", x, tier, rec.get("age"))
    sim._dormant.save(x, {"beliefs": ["__ax_mark__"], "money": 555.0})
    sim.run()
    live = {a.pool_pid: a for a in sim.agents}
    assert x in live, "再来街者 x が day1 に present になっていない"
    assert "__ax_mark__" in live[x].beliefs                     # hydrate を通った証拠
    assert live[x].ontology_axes.get("info_behavior") == expect, \
        "hydrate 経由で軸群が変わった(pid+age 純関数が崩れている)"


# ======================================================= 第3軸=同行者構成(companions・第42バッチ)
# 機構は多軸にもう1本足すだけ(Wave2 構造の再利用)。差分は(i)party_size 由来のデータ駆動割当を最優先、
# (ii)day_varying=true で (pid, day) 純関数(準静的=日ごとに変わるが決定論・run.seed 非依存)。
def _cfg_comp(enabled=True):
    """companions(同行者構成)+ info_behavior を持つ cfg(直交性テスト用に2軸)。文言はダミー。"""
    return ontology.build_cfg({
        "enabled": enabled, "seed": 20260721,
        "groups": {"jp_metro": {"label": "a", "line": "L"},
                   "jp_other": {"label": "b", "line": "L2"}},
        "composition": {"stochastic": {"jp_metro": 0.5, "jp_other": 0.5},
                        "default": {"jp_metro": 1.0}},
        "axes": {
            "info_behavior": {
                "seed_offset": 101,
                "groups": {"info_sns": {"label": "s", "line": "AX_sns"},
                           "info_official": {"label": "o", "line": "AX_official"}},
                "composition": {"default": {"info_sns": 0.5, "info_official": 0.5}}},
            "companions": {
                "seed_offset": 202, "day_varying": True,
                "party_map": {1: "solo", 2: "pair", "default": "group"},
                "groups": {"solo":  {"label": "一人", "line": "C_solo"},
                           "pair":  {"label": "二人", "line": "C_pair"},
                           "group": {"label": "三人以上", "line": "C_group"}},
                "composition": {
                    "resident":   {"solo": 0.82, "pair": 0.15, "group": 0.03},
                    "stochastic": {"solo": 0.30, "pair": 0.46, "group": 0.24},
                    "default":    {"solo": 0.55, "pair": 0.30, "group": 0.15}}},
        }})


def test_build_cfg_parses_party_map_and_day_varying():
    """build_cfg: party_map(int キー+ default)と day_varying を正準化。既定は空 dict / False(後方互換)。"""
    oc = _cfg_comp()
    ax = oc["axes"]["companions"]
    assert ax["party_map"] == {"1": "solo", "2": "pair", "default": "group"}  # キーは文字列に統一
    assert ax["day_varying"] is True
    # party_map/day_varying を持たない軸(info_behavior)は空 dict / False=従来と同一構造(後方互換)
    assert oc["axes"]["info_behavior"]["party_map"] == {}
    assert oc["axes"]["info_behavior"]["day_varying"] is False


def test_companions_party_size_priority():
    """party_size(実人数)がある個体はデータ駆動で確定: 1→solo/2→pair/3+→group(ハッシュ・composition 非依存)。"""
    oc = _cfg_comp()
    for pid in ("L4_1", "abc", "42", "L2_00099"):
        assert ontology.assign_axis(oc, "companions", pid, "stochastic", 30,
                                    day=0, party_size=1) == "solo"
        assert ontology.assign_axis(oc, "companions", pid, "stochastic", 30,
                                    day=0, party_size=2) == "pair"
        for ps in (3, 4, 5):
            assert ontology.assign_axis(oc, "companions", pid, "stochastic", 30,
                                        day=0, party_size=ps) == "group"
    # party_size 由来は day を変えても固定(準静的だが実人数個体は day 非依存で不変)
    a0 = ontology.assign_axis(oc, "companions", "L4_9", "stochastic", 30, day=0, party_size=2)
    a1 = ontology.assign_axis(oc, "companions", "L4_9", "stochastic", 30, day=7, party_size=2)
    assert a0 == a1 == "pair"
    # party_size 不明(None)や 0 は party_map を使わず composition 後退(群のどれか)
    g = ontology.assign_axis(oc, "companions", "L4_9", "stochastic", 30, day=0, party_size=None)
    assert g in oc["axes"]["companions"]["groups"]
    assert ontology._party_bucket({1: "solo", "default": "group"}, 0) is None


def test_companions_tier_composition_converges():
    """party_size 無し個体は tier 別 composition に収束(±0.02): 住民=solo 支配・stochastic=連れ厚め。"""
    oc = _cfg_comp()
    N = 40000
    res = Counter(ontology.assign_axis(oc, "companions", f"R_{i}", "resident", None, day=0)
                  for i in range(N))
    rf = {k: v / N for k, v in res.items()}
    assert abs(rf["solo"] - 0.82) < 0.02 and abs(rf.get("group", 0.0) - 0.03) < 0.02
    sto = Counter(ontology.assign_axis(oc, "companions", f"S_{i}", "stochastic", None, day=0)
                  for i in range(N))
    sf = {k: v / N for k, v in sto.items()}
    assert abs(sf["solo"] - 0.30) < 0.02
    assert abs(sf["pair"] - 0.46) < 0.02
    assert abs(sf["group"] - 0.24) < 0.02
    assert (sf["pair"] + sf["group"]) > sf["solo"], "stochastic は連れ厚め(渋谷実態)であるべき"


def test_companions_day_varying_and_deterministic():
    """day_varying: day0 と day1 で割当が異なる個体が存在(日ごとに変わる)+ 同 day 同一(決定論=resume 安定)。"""
    oc = _cfg_comp()
    N = 3000
    changed = 0
    for i in range(N):
        pid = f"D_{i}"
        g0 = ontology.assign_axis(oc, "companions", pid, "stochastic", None, day=0)
        g0b = ontology.assign_axis(oc, "companions", pid, "stochastic", None, day=0)
        g1 = ontology.assign_axis(oc, "companions", pid, "stochastic", None, day=1)
        assert g0 == g0b                              # 同 day・同 pid=同一(決定論・resume/hydrate 安定)
        if g0 != g1:
            changed += 1
    assert changed > 0, "day0 と day1 で割当が変わる個体が1体も無い(day が効いていない)"
    # 分布は day に依らず composition と一致(day は割当を混ぜるだけ=比率不変)
    c1 = Counter(ontology.assign_axis(oc, "companions", f"D_{i}", "stochastic", None, day=1)
                 for i in range(40000))
    assert abs(c1["pair"] / 40000 - 0.46) < 0.02


def test_companions_orthogonal_to_culture_and_info():
    """直交: companions が文化圏軸・情報行動軸のどちらとも独立(同時分布≈周辺積・偏差<0.01)。"""
    oc = _cfg_comp()
    N = 60000
    cult, comp, info = [], [], []
    for i in range(N):
        pid = f"P_{i}"
        cult.append(ontology.assign_group(oc, pid, "stochastic"))
        comp.append(ontology.assign_axis(oc, "companions", pid, "stochastic", None, day=0))
        info.append(ontology.assign_axis(oc, "info_behavior", pid, "stochastic", 30))
    for other in (cult, info):
        cm, om = Counter(comp), Counter(other)
        jm = Counter(zip(comp, other))
        max_dev = max(abs(jm.get((c, o), 0) / N - (cm[c] / N) * (om[o] / N))
                      for c in cm for o in om)
        assert max_dev < 0.01, f"companions が他軸と独立でない(max_dev={max_dev:.4f})"


def test_companions_run_seed_invariant():
    """割当は run.seed を引数に取らない=種を変えても不変(companions も hashlib 純関数=比較実験の要)。"""
    oc = _cfg_comp()
    for pid in ("D_1", "D_2", "D_3"):
        g = ontology.assign_axis(oc, "companions", pid, "stochastic", None, day=2)
        assert g == ontology.assign_axis(oc, "companions", pid, "stochastic", None, day=2)


# ------------------------------------------------------------------ プロンプト注入(最大4行)
def test_four_line_injection_order(tmp_path):
    """文化圏行→軸行(companions, info_behavior のソート順)→訓練行を persona 直後に順に注入(最大4行)。"""
    sim = _sim(tmp_path, "fourline", steps=1)
    sim.run()
    a = sim.agents[0]
    a.ontology_line = "文化圏の経験行。"
    a.ontology_axis_lines = ["同行者の行。", "情報行動の行。", "訓練経験の行。"]
    p = deliberate.build_prompt(a, place_name="路上", surprise="solo",
                                nearby_names=[], sim_min=600, step=3)
    assert (p.index(a.persona) < p.index("文化圏の経験行。") < p.index("同行者の行。")
            < p.index("情報行動の行。") < p.index("訓練経験の行。"))


def test_on_records_companions_axis(tmp_path):
    """ON: 直接ランで companions 軸が agents.json/agent.ontology_axes に載り、行が注入行に含まれる。"""
    on = _sim(tmp_path, "compjson", n=40, **_ON)
    on.run()
    comp_groups = on.ontologycfg["axes"]["companions"]["groups"]
    meta = json.loads((on.out_dir / "agents.json").read_text(encoding="utf-8"))
    got = [m for m in meta
           if "ontology_axes" in m and "companions" in m["ontology_axes"]]
    assert got, "companions 軸が1体も記録されていない"
    for m in got:
        assert m["ontology_axes"]["companions"] in comp_groups
    lines = {ontology.axis_line(on.ontologycfg, "companions", g) for g in comp_groups}
    seen = any(al in lines
               for a in on.agents
               for al in (getattr(a, "ontology_axis_lines", None) or []))
    assert seen, "companions の行が注入行(ontology_axis_lines)に一度も載っていない"


def test_apply_ontology_companions_party_size_wiring(tmp_path):
    """_apply_ontology(party_size=…) が companions をデータ駆動で据える(3+→group / 1→solo)。配線検証。"""
    sim = _sim(tmp_path, "applyps", n=5, **_ON)
    a = sim.agents[0]
    a.ontology_axes = None
    sim._apply_ontology(a, "stochastic", party_size=3)
    assert a.ontology_axes.get("companions") == "group"
    a.ontology_axes = None
    sim._apply_ontology(a, "stochastic", party_size=1)
    assert a.ontology_axes.get("companions") == "solo"


def test_off_no_companions_attributes(tmp_path):
    """OFF は companions 軸も1つも作らない(既定 OFF=完全 no-op・現行 config に companions を足しても不変)。"""
    off = _sim(tmp_path, "offcomp", n=15, steps=144, **_OFF)
    off.run()
    assert all(getattr(a, "ontology_axes", None) is None for a in off.agents)
    pure = _sim(tmp_path, "purecomp", n=15, steps=144)
    pure.run()
    assert _l1(pure) == _l1(off)                       # companions 追加後も OFF=純粋既定と L1 バイト一致


def test_companions_call_count_k_invariant(tmp_path):
    """R1: companions ON のまま k(writeback)を free/off に振っても LLM 呼数が不変(追加呼ゼロ)。"""
    free = _sim(tmp_path, "compk_free", **{**_ON, "k.writeback": "free"})
    free.run()
    off = _sim(tmp_path, "compk_off", **{**_ON, "k.writeback": "off"})
    off.run()
    assert len(free.logger.llm_calls) == len(off.logger.llm_calls)


# ------------------------------------------------------------------ analyze_groups --axis companions
def test_analyze_groups_companions_axis(tmp_path):
    """analyze_groups --axis companions: ontology_axes[companions] で群別集計・axes.companions.groups からラベル解決。"""
    import yaml

    import analyze_groups
    rows = [
        {"id": 0, "ontology_group": "jp_metro",
         "ontology_axes": {"info_behavior": "info_sns", "companions": "pair"}},
        {"id": 1, "ontology_group": "west_visit",
         "ontology_axes": {"info_behavior": "info_official", "companions": "solo"}},
        {"id": 2, "ontology_group": "jp_metro",
         "ontology_axes": {"info_behavior": "info_sns"}},      # companions 未設定の個体
    ]
    (tmp_path / "agents.json").write_text(json.dumps(rows, ensure_ascii=False),
                                          encoding="utf-8")
    doc = {"ontology": {"axes": {"companions": {"groups": {
        "solo": {"label": "単独で動いている人"},
        "pair": {"label": "連れと2人で動いている人"}}}}}}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(doc, allow_unicode=True),
                                          encoding="utf-8")
    m, _ = analyze_groups.load_group_map(str(tmp_path), "companions")
    assert m == {0: "pair", 1: "solo"}                 # id2 は companions 未設定=除外
    defs = analyze_groups.load_group_defs(str(tmp_path), "companions")
    assert defs.get("pair", {}).get("label") == "連れと2人で動いている人"


# ------------------------------------------------------------------ pool 経路(party_size 優先 + hydrate)
def test_pool_companions_party_size_priority(small_pool, tmp_path):
    """pool ON day0: 全 present 個体の companions=assign_axis(pid,tier,age,day=0,party_size)。party_size 個体は party_map 値。"""
    sim = Simulation(_pool_cfg("cpp", small_pool, n_steps=1, **_ON), out_dir=tmp_path / "cpp")
    oc = sim.ontologycfg
    assert sim.agents
    seen_party = 0
    for a in sim.agents:
        rec = sim._pool.get(a.pool_pid)
        tier = str(rec.get("presence", "resident"))
        ps = rec.get("party_size")
        expect = ontology.assign_axis(oc, "companions", a.pool_pid, tier, a.age,
                                      day=0, party_size=ps)
        assert a.ontology_axes.get("companions") == expect
        if ps is not None:                             # party_size 個体はデータ駆動の直接写像
            direct = {1: "solo", 2: "pair"}.get(int(ps), "group")
            assert a.ontology_axes.get("companions") == direct
            seen_party += 1
    assert seen_party > 0, "party_size を持つ来街者(L4)が day0 present に1体も居ない"


def test_pool_companions_hydrate_stable(small_pool, tmp_path):
    """P3 再来街(hydrate): party_size 個体の companions は day 非依存で固定=hydrate を跨いで不変。"""
    ps = pool_mod.PoolStore(small_pool)
    recs = ps.presence_records()
    hub = RngHub(42)
    s0 = set(present_for_day(recs, 0, 400, hub, 0))
    s1 = set(present_for_day(recs, 1, 400, hub, 1))
    entrants = [x for x in sorted(s1 - s0)
                if ps.get(x).get("party_size") is not None]
    assert entrants, "party_size を持つ day1 再来街候補が見つからない"
    x = entrants[0]
    ps_val = int(ps.get(x)["party_size"])
    expect = {1: "solo", 2: "pair"}.get(ps_val, "group")

    sim = Simulation(_pool_cfg("chy", small_pool, n_steps=210, **_ON), out_dir=tmp_path / "chy")
    sim._dormant.save(x, {"beliefs": ["__comp_mark__"], "money": 555.0})
    sim.run()
    live = {a.pool_pid: a for a in sim.agents}
    assert x in live, "再来街者 x が day1 に present になっていない"
    assert "__comp_mark__" in live[x].beliefs                   # hydrate を通った証拠
    assert live[x].ontology_axes.get("companions") == expect, \
        "hydrate 経由で同行者構成(party_size 固定)が変わった"
