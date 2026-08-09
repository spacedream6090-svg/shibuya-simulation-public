"""街の顔 = 路上の生業と条例(Wave 4 III-3 ``street_life``)のテスト。

守るもの(検収基準の順)
  ① OFF(既定)= 純粋既定と L1 バイト一致・新 12 種 0 件・agent に属性が生えない・
     sim に state が生えない・summary キーなし
  ② 束ね: 役割ごとに正しい種類の持ち場が付く / 勤務窓が帯になる /
     路上生活者は建物のない夜の居場所を得る(既存の路上就寝経路)
  ③ 各役割が**自分の時間帯**でだけ立つ(帯の外では 1 件も出ない)
  ④ セッション単位の L1(同じ持ち場に何 step 立っても 1 セッション 1 件)
  ⑤ 金の流れ: キッチンカー/占いは買い手→売り手で**保存**、募金は街の外へ出る、
     過料は所持金の範囲で徴収され行政 ON なら区の歳入
  ⑥ 条例パトロール: 警告 → 再犯で過料 5 万円 → クールダウン(決定論)
  ⑦ 警察官の接近で持ち場を移す(いたちごっこ)
  ⑧ ★尊厳規約: 路上生活者の id が crime / nuisance に**1 度も現れない** /
     module の文字列に蔑称が 1 語も無い / 権利(記憶・発火)を剥奪していない
  ⑨ 迷惑行為「客引き」の重複解消(ON では nuisance 側が供給しない)
  ⑩ L1 の上限が効く / 新 12 種が全部 causality に分類されている
  ⑪ ON 同 seed 2 回で完全一致 / **乱数 stream を 1 本も引かない**(AST 検査)
  ⑫ LLM 呼数 ON/OFF 完全一致(FixedLLM)
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from society import diversity
from society import registry as R
from society import street_life as SL
from society.config import load_config
from society.engine.simulation import Simulation
from society.observer import causality as C
from society.observer.schema import EVENT_KINDS

MODULE = Path(SL.__file__)

# 本 module が出す L1 種(すべて材料側 registration)
NEW_KINDS = ("street_performance", "street_speech", "tissue_offer", "stall_sale",
             "donation", "fortune_reading", "touting", "touting_warning",
             "touting_fine", "street_disperse", "outreach_contact", "shelter_move")

OFF = {"street_life.enabled": "false"}
ON = {"street_life.enabled": "true"}


# --------------------------------------------------------------------------- #
# 共通ヘルパ(test_traces.py / test_diversity.py と同型)
# --------------------------------------------------------------------------- #
def _cfg(name, n_steps=48, n_agents=15, **ov):
    dot = ["run.seed=42", f"run.n_agents={n_agents}", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock", "observer.snapshot_every=144"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def _sim(tmp_path, name, n_steps=48, n_agents=15, **ov):
    return Simulation(_cfg(name, n_steps, n_agents, **ov), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _kind(sim, kind):
    return [e for e in sim.logger.events if e.kind == kind]


def _place(sim, a, node):
    """個体をノードの路上に立たせる(位置確定後の世界を手で作る)。"""
    a.node = str(node)
    a.x, a.y = sim.city.node_xy(node)
    a.loc = "street"
    a.building = ""
    a.sleeping = False
    return a


def _stage(tmp_path, name, occ, band_min, n_crowd=6, n_agents=20, **ov):
    """役割 1 人 + 通行人 n 人を持ち場に立たせた舞台を作る(束ね済み)。"""
    sim = _sim(tmp_path, name, n_steps=1, n_agents=n_agents, **{**ON, **ov})
    ags = sorted(sim.agents, key=lambda a: int(a.id))
    actor = ags[0]
    actor.occupation = occ
    SL.bind(sim)
    node = actor.street_post
    _place(sim, actor, node)
    for o in ags[1:1 + n_crowd]:
        o.occupation = "会社員"
        _place(sim, o, node)
    for o in ags[1 + n_crowd:]:                    # 残りは遠くへ(混入させない)
        o.loc = "outside"
    return sim, actor, ags[1:1 + n_crowd], band_min


class _FixedLLM:
    """**プロンプト非依存**の巡回応答スタブ(test_traces.py と同型)。"""

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
# ① OFF(既定)
# =========================================================================== #
def test_off_matches_pure_default(tmp_path):
    """明示 OFF と純粋既定が L1 完全一致。新 12 種は 1 件も出ない(seam が no-op)。"""
    pure = _sim(tmp_path, "pure")
    pure.run()
    off = _sim(tmp_path, "expl_off", **OFF)
    off.run()
    assert _l1(pure) == _l1(off), "OFF が純粋既定と不一致(street_life seam が no-op でない)"
    for k in NEW_KINDS:
        assert not _kind(pure, k), f"OFF で {k} が出ている"


def test_off_grows_no_state_and_no_attributes(tmp_path):
    """OFF は sim に state を生やさず・agent に属性を生やさない(第96 traces と同流儀)。"""
    sim = _sim(tmp_path, "off_state", n_steps=24, **OFF)
    sim.run()
    assert getattr(sim, "_sl_state", None) is None, "OFF で state が生えた"
    assert getattr(sim, "_sl_posts", None) is None, "OFF で持ち場キャッシュが生えた"
    for a in sim.agents:
        for attr in ("street_post", "street_tout", "street_session",
                     "street_sleep_node", "outreach_count"):
            assert not hasattr(a, attr), f"OFF で agent に {attr} が生えた"


def test_off_provenance_is_none(tmp_path):
    """OFF は summary の street_life キー自体を作らない。"""
    sim = _sim(tmp_path, "off_prov", n_steps=1, **OFF)
    assert SL.provenance(sim) is None


def test_off_bind_and_phase_are_noop(tmp_path):
    """OFF では bind も phase も何も返さず何も書かない(直接呼んでも安全)。"""
    sim = _sim(tmp_path, "off_call", n_steps=1, **OFF)
    a = sorted(sim.agents, key=lambda z: int(z.id))[0]
    a.occupation = SL.MUSICIAN
    out = SL.bind(sim)
    SL.phase(sim, 0, 19 * 60)
    assert out["n_role"] == 0 and not hasattr(a, "street_post")
    assert SL.rough_sleeper_ids(sim) == frozenset()
    assert not [e for e in sim.logger.events if e.kind in NEW_KINDS]


def test_off_is_unchanged_even_with_street_roles_in_the_roster(tmp_path):
    """名簿に路上の役割が居ても OFF は完全 no-op(純粋既定と L1 一致)。"""
    base = _sim(tmp_path, "rl_base", n_steps=24)
    for i, a in enumerate(sorted(base.agents, key=lambda z: int(z.id))[:6]):
        a.occupation = SL.STREET_OCCS[i % len(SL.STREET_OCCS)]
    base.run()
    off = _sim(tmp_path, "rl_off", n_steps=24, **OFF)
    for i, a in enumerate(sorted(off.agents, key=lambda z: int(z.id))[:6]):
        a.occupation = SL.STREET_OCCS[i % len(SL.STREET_OCCS)]
    off.run()
    assert _l1(base) == _l1(off), "路上の役割入り名簿で seam が no-op でない"


# =========================================================================== #
# ② 束ね(持ち場・勤務窓・夜の居場所)
# =========================================================================== #
def test_posts_are_derived_from_the_map(tmp_path):
    """持ち場は**地図から決定論導出**され、どの種類も同じ 2 回で同一(乱数ゼロ)。"""
    sim = _sim(tmp_path, "posts", n_steps=1, **ON)
    for kind in (SL.P_STATION, SL.P_PLAZA, SL.P_NIGHT, SL.P_SIDE,
                 SL.P_REST, SL.P_COVERED):
        got = SL.posts(sim, kind)
        assert got == SL.posts(sim, kind), f"{kind}: 2 回で不一致(決定論でない)"
        assert all(n in sim.city.graph.nodes for n in got), f"{kind}: 地図に無いノード"
    assert SL.posts(sim, SL.P_STATION)[0] == sim.city.station_node


@pytest.mark.parametrize("occ", [o for o in SL.STREET_OCCS if o != SL.ROUGH])
def test_bind_gives_a_post_and_the_role_band_as_the_work_window(tmp_path, occ):
    """働く役割には持ち場(路面)と**役割の帯**の勤務窓が付く。"""
    sim = _sim(tmp_path, f"bind_{abs(hash(occ)) % 9999}", n_steps=1, **ON)
    a = sorted(sim.agents, key=lambda z: int(z.id))[0]
    a.occupation = occ
    SL.bind(sim)
    spec = SL.ROLE_SPECS[occ]
    assert a.street_post in SL.posts(sim, spec["post"]), f"{occ}: 持ち場の種類が違う"
    assert a.work_node == a.street_post and a.work_building == ""
    # 2 帯以上の役割は union の範囲(勤務窓は 1 本しか持てないため。実装の注記を参照)
    assert (a.work_start_min, a.work_end_min) == (
        spec["bands"][0][0], min(1440, max(b[1] for b in spec["bands"])))


def test_multi_band_roles_span_all_their_bands(tmp_path):
    """★2 帯以上の役割(朝夕の配布・朝夕の演説・昼夜の巡回)は夕方の帯も立つ。

    bands[0] だけを勤務窓にした初版では夕方の帯が一度も立たなかった(実測)。
    """
    for occ, evening in ((SL.TISSUE, 18 * 60), (SL.SPEECH, 18 * 60)):
        sim = _sim(tmp_path, f"mb_{abs(hash(occ)) % 9999}", n_steps=1, **ON)
        a = sorted(sim.agents, key=lambda z: int(z.id))[0]
        a.occupation = occ
        SL.bind(sim)
        assert a.work_start_min <= evening < a.work_end_min, \
            f"{occ}: 夕方の帯が勤務窓の外(持ち場に立てない)"
        assert SL._in_band(SL.ROLE_SPECS[occ]["bands"], evening) >= 0


def test_bind_is_idempotent(tmp_path):
    """bind は冪等(2 回呼んでも持ち場も勤務窓も 1 バイト変わらない)= resume 安全。"""
    sim = _sim(tmp_path, "bind_idem", n_steps=1, **ON)
    ags = sorted(sim.agents, key=lambda z: int(z.id))
    for i, a in enumerate(ags[:8]):
        a.occupation = SL.STREET_OCCS[i % len(SL.STREET_OCCS)]
    SL.bind(sim)
    snap = [(a.id, getattr(a, "street_post", ""), a.work_start_min,
             getattr(a, "home_node", "")) for a in ags]
    SL.bind(sim)
    assert snap == [(a.id, getattr(a, "street_post", ""), a.work_start_min,
                     getattr(a, "home_node", "")) for a in ags]


def test_rough_sleeper_gets_a_street_night_place_and_no_work_window(tmp_path):
    """路上生活者は**勤務窓を持たず**、夜の居場所(建物なし)を得る = 既存の路上就寝経路。"""
    sim = _sim(tmp_path, "rough_bind", n_steps=1, **ON)
    a = sorted(sim.agents, key=lambda z: int(z.id))[0]
    a.occupation = SL.ROUGH
    before = (a.work_start_min, a.work_end_min)
    SL.bind(sim)
    assert a.home_building == "" and a.home_floor == 0, "建物つきの家が残っている"
    assert a.home_node == a.street_sleep_node
    assert a.street_sleep_node in (SL.posts(sim, SL.P_REST) + SL.posts(sim, SL.P_COVERED))
    assert (a.work_start_min, a.work_end_min) == before, "路上生活者に勤務窓を与えた"
    assert not hasattr(a, "street_post"), "路上生活者に『持ち場』を与えた"
    assert float(a.money) <= sim.streetlifecfg["rough_money_cap"]


def test_rough_sleeper_keeps_full_agency(tmp_path):
    """★尊厳規約 3: 記憶・発火の能力を 1 つも削っていない(他の個体と同じ属性を持つ)。"""
    sim = _sim(tmp_path, "rough_agency", n_steps=1, **ON)
    ags = sorted(sim.agents, key=lambda z: int(z.id))
    rough, other = ags[0], ags[1]
    rough.occupation = SL.ROUGH
    SL.bind(sim)
    for attr in ("mem", "drive_threshold", "fire_weight", "traits", "states",
                 "opinion", "adopted"):
        assert hasattr(rough, attr), f"路上生活者から {attr} が失われている"
    rough.remember("何かがあった")
    assert type(rough.mem) is type(other.mem)


# =========================================================================== #
# ③④ 時間帯とセッション
# =========================================================================== #
_BAND_CASES = [
    (SL.MUSICIAN, 19 * 60, "street_performance"),
    (SL.SPEECH, 8 * 60 + 30, "street_speech"),
    (SL.TISSUE, 8 * 60, "tissue_offer"),
    (SL.KITCHEN, 12 * 60, "stall_sale"),
    (SL.FORTUNE, 21 * 60, "fortune_reading"),
    (SL.FUNDRAISER, 12 * 60, "donation"),
]


@pytest.mark.parametrize("occ,band_min,kind", _BAND_CASES)
def test_each_role_fires_only_inside_its_band(tmp_path, occ, band_min, kind):
    """役割は**自分の時間帯**でだけ立つ(帯の中で 1 件・帯の外で 0 件)。"""
    sim, actor, crowd, _ = _stage(tmp_path, f"band_{kind}", occ, band_min)
    for o in crowd:
        o.money = 100000.0
    SL.phase(sim, 0, band_min)
    assert len(_kind(sim, kind)) == 1, f"{occ}: 帯の中で {kind} が出ていない"
    out_min = (band_min + 12 * 60) % 1440           # 12 時間ずらす = 全役割で帯の外
    n_before = len(_kind(sim, kind))
    SL.phase(sim, 1, out_min)
    assert len(_kind(sim, kind)) == n_before, f"{occ}: 帯の外で {kind} が出た"


@pytest.mark.parametrize("occ,band_min,kind", _BAND_CASES)
def test_one_event_per_session_not_per_step(tmp_path, occ, band_min, kind):
    """★L1 の上限: 同じ持ち場に何 step 立っても **1 セッション 1 件**。"""
    sim, actor, crowd, _ = _stage(tmp_path, f"sess_{kind}", occ, band_min)
    for o in crowd:
        o.money = 100000.0
    for s in range(6):
        SL.phase(sim, s, band_min + s * 10)
    assert len(_kind(sim, kind)) == 1, f"{occ}: {kind} が step ごとに出ている"


def test_notice_cap_bounds_the_memory_hooks(tmp_path):
    """1 セッションで記憶 1 行が入る人数は notice_cap で頭打ちになる。"""
    sim, actor, crowd, _ = _stage(tmp_path, "cap", SL.MUSICIAN, 19 * 60,
                                  n_crowd=12, n_agents=24,
                                  **{"street_life.notice_cap": "3"})
    for s in range(5):
        SL.phase(sim, s, 19 * 60 + s * 10)
    assert sim._sl_state["notices"] == 3, "notice_cap を超えて記憶が入っている"


def test_performance_puts_a_neutral_line_into_passerby_memory(tmp_path):
    """演奏はその場に居合わせた人の**記憶**に定型 1 行を入れる(プロンプトの欄は増やさない)。"""
    sim, actor, crowd, _ = _stage(tmp_path, "notice", SL.MUSICIAN, 19 * 60)
    SL.phase(sim, 0, 19 * 60)
    texts = []
    for o in crowd:
        texts += [e.text for e in o.mem.buffer]
    assert SL.NOTICE_TEXT[SL.MUSICIAN] in texts, \
        "通行人の記憶に演奏の 1 行が入っていない"
    assert all(t == SL.NOTICE_TEXT[SL.MUSICIAN] for t in texts), \
        "定型文以外の行が混ざっている(no-fingerprint の逸脱)"


def test_notice_text_is_a_pure_function_of_the_role():
    """★no-fingerprint: 定型文に数字・金額・config・実験条件が 1 文字も入らない。"""
    import re
    for role, text in SL.NOTICE_TEXT.items():
        assert not re.search(r"[0-90-9]", text), f"{role}: 定型文に数字がある"
        assert "{" not in text and "}" not in text, f"{role}: 定型文に埋め込み欄がある"
    for text in (SL.TOUT_TEXT, SL.TOUT_WARN_TEXT, SL.TOUT_FINE_TEXT,
                 SL.OUTREACH_TEXT, SL.OUTREACH_WORKER_TEXT, SL.SHELTER_TEXT):
        assert not re.search(r"[0-90-9]", text)


# =========================================================================== #
# ⑤ 金の流れ
# =========================================================================== #
def test_stall_sale_moves_money_from_buyer_to_vendor(tmp_path):
    """キッチンカーの売上は買い手→売り手の移転 = 街の中で**保存**する(湧かない/消えない)。"""
    sim, actor, crowd, _ = _stage(tmp_path, "stall", SL.KITCHEN, 12 * 60)
    for o in crowd:
        o.money = 100000.0
    before = float(actor.money) + sum(float(o.money) for o in crowd)
    SL.phase(sim, 0, 12 * 60)
    after = float(actor.money) + sum(float(o.money) for o in crowd)
    ev = _kind(sim, "stall_sale")[0]
    assert ev.payload["sold"] >= 1, "誰も買っていない(刻みの設定を見直す)"
    assert abs(after - before) < 1e-6, "売上で金が湧いた/消えた"
    assert float(actor.money) > 0.0


def test_donation_leaves_the_city(tmp_path):
    """街頭募金は**街の外の団体**へ出る = 募金スタッフの所持金は 1 円も増えない。"""
    sim, actor, crowd, _ = _stage(tmp_path, "dona", SL.FUNDRAISER, 12 * 60,
                                  **{"street_life.donation_every": "3"})
    for o in crowd:
        o.money = 100000.0
    before = float(actor.money)
    SL.phase(sim, 0, 12 * 60)
    ev = _kind(sim, "donation")[0]
    assert ev.payload["donors"] >= 1 and ev.payload["amount"] > 0
    assert float(actor.money) == before, "募金スタッフの懐に入っている"


def test_donation_is_classified_as_row_when_sfc_is_on(tmp_path):
    """SFC ON では募金に payee(RoW トークン)が付く / OFF ではキー自体が無い。"""
    sim, actor, crowd, _ = _stage(tmp_path, "dona_sfc", SL.FUNDRAISER, 12 * 60,
                                  **{"economy.org_accounting.enabled": "true",
                                     "street_life.donation_every": "3"})
    for o in crowd:
        o.money = 100000.0
    SL.phase(sim, 0, 12 * 60)
    assert _kind(sim, "donation")[0].payload["payee"].startswith("row:")
    off, actor2, crowd2, _ = _stage(tmp_path, "dona_nosfc", SL.FUNDRAISER, 12 * 60,
                                    **{"street_life.donation_every": "3"})
    for o in crowd2:
        o.money = 100000.0
    SL.phase(off, 0, 12 * 60)
    assert "payee" not in _kind(off, "donation")[0].payload


def test_nobody_pays_what_they_do_not_have(tmp_path):
    """所持金が足りない通行人は買わない(マイナス残高を作らない)。"""
    sim, actor, crowd, _ = _stage(tmp_path, "poor", SL.KITCHEN, 12 * 60)
    for o in crowd:
        o.money = 1.0
    SL.phase(sim, 0, 12 * 60)
    assert _kind(sim, "stall_sale")[0].payload["sold"] == 0
    assert all(float(o.money) == 1.0 for o in crowd)


# =========================================================================== #
# ⑥ 客引き × 条例パトロール
# =========================================================================== #
def _tout_stage(tmp_path, name, **ov):
    """夜間店舗ノードに客引き 1 人 + 通行人を立たせ、警察官を 1 人用意する。"""
    sim = _sim(tmp_path, name, n_steps=1, n_agents=20, **{**ON, **ov})
    ags = sorted(sim.agents, key=lambda a: int(a.id))
    night = SL.posts(sim, SL.P_NIGHT)[0]
    tout = ags[0]
    tout.occupation = "会社員"
    tout.work_node = night
    officer = ags[1]
    officer.occupation = "警察官"
    SL.bind(sim)
    assert getattr(tout, "street_tout", False), "客引きの決定論部分集合に入っていない"
    _place(sim, tout, night)
    for o in ags[2:8]:
        o.occupation = "会社員"
        _place(sim, o, night)
    for o in ags[8:]:
        o.loc = "outside"
    officer.loc = "outside"                        # 既定は不在(呼ぶまで来ない)
    return sim, tout, officer, night


def test_touting_happens_in_the_zone_at_night(tmp_path):
    """客引きは夜の時間帯に**店の前**(職場ノード)で立つ。啓発区域の内外を payload に残す。"""
    sim, tout, officer, night = _tout_stage(tmp_path, "tout")
    SL.phase(sim, 0, 20 * 60)
    ev = _kind(sim, "touting")
    assert len(ev) == 1 and ev[0].agent_id == tout.id
    assert ev[0].payload["node"] == night and ev[0].payload["in_zone"] in (True, False)
    assert ev[0].payload["targets"] >= 1


def test_touting_does_not_happen_away_from_the_venue(tmp_path):
    """職場ノード以外では客引きしない(街のどこでも湧く迷惑行為にしない)。"""
    sim, tout, officer, night = _tout_stage(tmp_path, "tout_away")
    other = [n for n in SL.posts(sim, SL.P_SIDE) if n != night][0]
    _place(sim, tout, other)
    SL.phase(sim, 0, 20 * 60)
    assert not _kind(sim, "touting")


def test_touting_does_not_happen_by_day(tmp_path):
    sim, tout, officer, night = _tout_stage(tmp_path, "tout_day")
    SL.phase(sim, 0, 12 * 60)
    assert not _kind(sim, "touting")


def test_warning_then_fine_then_cooldown(tmp_path):
    """★条例パトロール: 1 回目=警告 / 2 回目=過料 5 万円 / どちらもクールダウンで止まる。"""
    sim, tout, officer, night = _tout_stage(tmp_path, "patrol")
    tout.money = 200000.0
    _place(sim, officer, night)                    # 警察官が現場に居る
    SL.phase(sim, 0, 20 * 60)
    assert len(_kind(sim, "touting_warning")) == 1
    assert not _kind(sim, "touting"), "警告された step に客引きが成立している"
    assert tout.street_cooldown_until == 0 + sim.streetlifecfg["cooldown_steps"]
    # クールダウン中は何も起きない
    SL.phase(sim, 1, 20 * 60 + 10)
    assert len(_kind(sim, "touting_warning")) == 1 and not _kind(sim, "touting")
    # クールダウン明け = 再犯 → 過料
    s = int(tout.street_cooldown_until) + 1
    SL.phase(sim, s, 21 * 60)
    fines = _kind(sim, "touting_fine")
    assert len(fines) == 1, "再犯で過料が出ていない"
    assert fines[0].payload["amount"] == 50000.0
    assert fines[0].payload["officer"] == officer.id
    assert fines[0].payload["target"] == tout.id
    assert float(tout.money) == 150000.0


def test_fine_is_capped_by_what_the_offender_has(tmp_path):
    """過料は所持金の範囲で徴収する(既存 enforcement と同じ作法・負にしない)。"""
    sim, tout, officer, night = _tout_stage(tmp_path, "fine_cap",
                                            **{"street_life.warn_before_fine": "0"})
    tout.money = 1200.0
    _place(sim, officer, night)
    SL.phase(sim, 0, 20 * 60)
    assert _kind(sim, "touting_fine")[0].payload["amount"] == 1200.0
    assert float(tout.money) == 0.0


def test_fine_becomes_ward_revenue_when_government_is_on(tmp_path):
    """行政 ON なら過料は区の歳入(既存 _phase_enforcement と同じ経路)。"""
    sim, tout, officer, night = _tout_stage(
        tmp_path, "fine_gov",
        **{"street_life.warn_before_fine": "0", "government.enabled": "true"})
    tout.money = 200000.0
    _place(sim, officer, night)
    from society.engine import scheduler
    gov = scheduler._gov(sim)                      # 行政を実体化(scheduler と同じ口)
    before = float(gov.balance["ward"])
    SL.phase(sim, 0, 20 * 60)
    assert float(gov.balance["ward"]) == before + 50000.0


def test_patrol_is_deterministic(tmp_path):
    """同じ舞台 2 回で警告/過料の列が完全一致(乱数ゼロ)。"""
    outs = []
    for i in (1, 2):
        sim, tout, officer, night = _tout_stage(tmp_path, f"patrol_det{i}")
        tout.money = 200000.0
        _place(sim, officer, night)
        for s in range(20):
            SL.phase(sim, s, 20 * 60 + s * 10)
        outs.append([[e.step, e.kind, json.dumps(e.payload, sort_keys=True,
                                                 ensure_ascii=False)]
                     for e in sim.logger.events if e.kind in NEW_KINDS])
    assert outs[0] == outs[1] and outs[0], "パトロールが非決定"


# =========================================================================== #
# ⑦ 警察官の接近で持ち場を移す
# =========================================================================== #
def test_police_proximity_relocates_the_musician(tmp_path):
    """★いたちごっこ: 同ノードに警察官が来たら次の持ち場へ移り、クールダウンに入る。"""
    sim, actor, crowd, _ = _stage(tmp_path, "disperse", SL.MUSICIAN, 19 * 60)
    officer = [a for a in sim.agents if a not in crowd and a is not actor][0]
    officer.occupation = "警察官"
    _place(sim, officer, actor.node)
    old_post = actor.street_post
    SL.phase(sim, 0, 19 * 60)
    ev = _kind(sim, "street_disperse")
    assert len(ev) == 1 and ev[0].payload["from"] == old_post
    assert actor.street_post != old_post and actor.work_node == actor.street_post
    assert actor.street_cooldown_until == sim.streetlifecfg["cooldown_steps"]
    assert not _kind(sim, "street_performance"), "退去した step に演奏が成立している"


def test_no_police_no_dispersal(tmp_path):
    sim, actor, crowd, _ = _stage(tmp_path, "no_disperse", SL.MUSICIAN, 19 * 60)
    SL.phase(sim, 0, 19 * 60)
    assert not _kind(sim, "street_disperse")


# =========================================================================== #
# ⑧ 尊厳規約(路上生活者)
# =========================================================================== #
# ★禁止語(**この一覧はテスト側にだけ置く**。実装の文字列に 1 語も現れてはならない)
_SLURS = ("ホームレス", "浮浪", "乞食", "物乞い", "こじき", "不審者", "たかり",
          "汚い", "臭い", "厄介者", "怠け", "落伍", "社会不適合", "浮浪者")


def _module_strings(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def test_no_derogatory_vocabulary_anywhere_in_the_module():
    """★尊厳規約 1: 実装(docstring・定型文・コメント込みの全文)に蔑称が 1 語も無い。"""
    text = MODULE.read_text(encoding="utf-8")
    hit = [w for w in _SLURS if w in text]
    assert hit == [], f"street_life.py に不適切な語がある: {hit}"
    for s in _module_strings(MODULE):
        assert not any(w in s for w in _SLURS)


def test_the_only_word_used_is_the_neutral_one():
    """路上生活者を指す語は中立語ただ 1 つ(役割名・定型文の両方)。"""
    assert SL.ROUGH == "路上生活者"
    assert "路上生活" in SL.ROUGH
    for text in (SL.OUTREACH_TEXT, SL.OUTREACH_WORKER_TEXT, SL.SHELTER_TEXT):
        assert not any(w in text for w in _SLURS)


def test_persona_pool_roles_use_neutral_vocabulary():
    """名簿ビルダ側(scripts/build_persona_pool.py)の役割行にも蔑称が無い。"""
    src = (MODULE.parents[2] / "scripts" / "build_persona_pool.py")
    text = src.read_text(encoding="utf-8")
    assert [w for w in _SLURS if w in text] == []


def test_rough_sleepers_are_excluded_from_crime_and_nuisance(tmp_path):
    """★尊厳規約 2: crime / nuisance の payload と agent_id に彼らの id が**1 度も**出ない。

    犯罪確率を極端に上げた対照スモークで、路上生活者が加害者・被害者・周囲の
    いずれにもならないことを確認する(street_life OFF ではこの除外は掛からない)。
    """
    sim = _sim(tmp_path, "dignity", n_steps=24, n_agents=24,
               **{**ON, "society_diversity.enabled": "true",
                  "society_diversity.crime_prob": "0.9",
                  "society_diversity.nuisance_prob": "0.05",
                  "society_diversity.nuisance_grievance": "0.02"})
    ags = sorted(sim.agents, key=lambda a: int(a.id))
    rough_ids = set()
    for a in ags[:8]:                              # 3 分の 1 を路上生活者にする
        a.occupation = SL.ROUGH
        rough_ids.add(int(a.id))
    sim.run()
    events = [e for e in sim.logger.events if e.kind in ("crime", "nuisance")]
    assert events, "対照スモークで crime/nuisance が 1 件も出ていない(検査が空回り)"
    for e in events:
        assert int(e.agent_id) not in rough_ids, f"{e.kind} の主語が路上生活者"
        for key in ("victim", "offender"):
            v = (e.payload or {}).get(key)
            if isinstance(v, int):
                assert int(v) not in rough_ids, f"{e.kind}.{key} が路上生活者"


def test_dignity_exclusion_is_empty_when_off(tmp_path):
    """除外集合は OFF では**必ず空**(= diversity 側のフィルタが完全 no-op)。"""
    sim = _sim(tmp_path, "excl_off", n_steps=1, **OFF)
    sorted(sim.agents, key=lambda a: int(a.id))[0].occupation = SL.ROUGH
    assert SL.rough_sleeper_ids(sim) == frozenset()
    assert diversity._street_life_excluded(sim) == frozenset()


def test_outreach_contact_and_shelter_move(tmp_path):
    """★尊厳規約 4: 区の巡回相談があり、積み重なれば既存の住居機構で住居へ移行する。"""
    sim = _sim(tmp_path, "outreach", n_steps=1, n_agents=16,
               **{**ON, "street_life.outreach_shelter_after": "2",
                  "street_life.outreach_period_days": "1"})
    ags = sorted(sim.agents, key=lambda a: int(a.id))
    worker, rough = ags[0], ags[1]
    worker.occupation = SL.OUTREACH
    rough.occupation = SL.ROUGH
    SL.bind(sim)
    node = worker.street_post
    _place(sim, worker, node)
    _place(sim, rough, node)
    for o in ags[2:]:
        o.loc = "outside"
    SL.phase(sim, 0, 15 * 60)                      # day 0
    assert len(_kind(sim, "outreach_contact")) == 1
    assert rough.outreach_count == 1
    SL.phase(sim, 144, 1440 + 15 * 60)             # day 1 = 2 回目 → 住居へ
    assert len(_kind(sim, "outreach_contact")) == 2
    mv = _kind(sim, "shelter_move")
    assert len(mv) == 1 and mv[0].payload["target"] == rough.id
    assert rough.home_building and sim.city.has_building(rough.home_building)
    assert rough.street_sleep_node == ""


def test_outreach_reaches_a_nearby_but_not_a_distant_person(tmp_path):
    """★巡回相談は**一帯を回る**行為: 近く(別ノードでも)には届き、遠くには届かない。

    初版はノード完全一致を条件にしたため接触が実測 0 件だった(= 支援事業が世界に
    存在しないのと同じ状態)。ここでその設計判断を機械固定する。
    """
    sim = _sim(tmp_path, "outreach_radius", n_steps=1, n_agents=16,
               **{**ON, "street_life.outreach_period_days": "1",
                  "street_life.outreach_shelter_after": "0",
                  "street_life.outreach_radius_m": "150.0"})
    ags = sorted(sim.agents, key=lambda a: int(a.id))
    worker, near, far = ags[0], ags[1], ags[2]
    worker.occupation = SL.OUTREACH
    near.occupation = SL.ROUGH
    far.occupation = SL.ROUGH
    SL.bind(sim)
    _place(sim, worker, worker.street_post)
    for o in ags[3:]:
        o.loc = "outside"
    # 近くの人: 別ノード扱いで 100 m 先に立たせる / 遠くの人: 1000 m 先
    for who, d in ((near, 100.0), (far, 1000.0)):
        who.node = "elsewhere-" + str(who.id)
        who.x, who.y = worker.x + d, worker.y
        who.loc = "street"
        who.building = ""
        who.sleeping = False
    SL.phase(sim, 0, 15 * 60)
    got = {e.payload["target"] for e in _kind(sim, "outreach_contact")}
    assert near.id in got, "150m 圏内の人に相談が届いていない"
    assert far.id not in got, "圏外の人にまで相談が届いている"


def test_only_the_outreach_role_roams(tmp_path):
    """持ち場を持つ役割は**持ち場でだけ**働く(巡回相談だけが例外)。"""
    assert SL.ROLE_SPECS[SL.OUTREACH].get("roam") is True
    for occ, spec in SL.ROLE_SPECS.items():
        if occ != SL.OUTREACH:
            assert not spec.get("roam", False), f"{occ} が roam になっている"
    sim, actor, crowd, _ = _stage(tmp_path, "roam_no", SL.MUSICIAN, 19 * 60)
    away = [n for n in SL.posts(sim, SL.P_SIDE) if n != actor.street_post][0]
    _place(sim, actor, away)
    for o in crowd:
        _place(sim, o, away)
    SL.phase(sim, 0, 19 * 60)
    assert not _kind(sim, "street_performance"), "持ち場の外で演奏が成立している"


def test_outreach_is_capped_to_once_per_agent_per_day(tmp_path):
    """巡回相談は 1 人 1 日 1 回まで(毎 step 出さない = L1 の上限)。"""
    sim = _sim(tmp_path, "outreach_cap", n_steps=1, n_agents=16,
               **{**ON, "street_life.outreach_period_days": "1",
                  "street_life.outreach_shelter_after": "0"})
    ags = sorted(sim.agents, key=lambda a: int(a.id))
    worker, rough = ags[0], ags[1]
    worker.occupation = SL.OUTREACH
    rough.occupation = SL.ROUGH
    SL.bind(sim)
    _place(sim, worker, worker.street_post)
    _place(sim, rough, worker.street_post)
    for o in ags[2:]:
        o.loc = "outside"
    for s in range(6):
        SL.phase(sim, s, 15 * 60 - 60 + s * 10)
    assert len(_kind(sim, "outreach_contact")) == 1


def test_rain_moves_the_night_place_under_cover(tmp_path):
    """★尊厳規約 5: 雨/雪の日は屋根のある場所へ、晴れの日は元の場所へ(天気は読むだけ)。"""
    sim = _sim(tmp_path, "rain", n_steps=1, n_agents=12, **ON)
    a = sorted(sim.agents, key=lambda z: int(z.id))[0]
    a.occupation = SL.ROUGH
    for o in sorted(sim.agents, key=lambda z: int(z.id))[1:]:
        o.loc = "outside"
    SL.bind(sim)
    base = a.street_sleep_node
    sim.today_weather = {"cond": "雨", "temp_hi": 14, "temp_lo": 9}
    SL.phase(sim, 0, 3 * 60)
    covered = SL.posts(sim, SL.P_COVERED)
    assert a.home_node in covered, "雨の日に屋根のある場所へ移っていない"
    sim.today_weather = {"cond": "晴", "temp_hi": 22, "temp_lo": 12}
    SL.phase(sim, 1, 3 * 60 + 10)
    assert a.home_node == base, "晴れの日に元の居場所へ戻っていない"
    assert a.home_building == "", "建物つきの家が付いた"


# =========================================================================== #
# ⑨ 迷惑行為「客引き」の重複解消
# =========================================================================== #
def test_nuisance_touting_is_superseded_when_street_life_is_on(tmp_path):
    """ON では nuisance の種類から「客引き」を供給しない(併存ではなく置き換え)。"""
    on = _sim(tmp_path, "sup_on", n_steps=1, **{**ON, "society_diversity.enabled": "true"})
    off = _sim(tmp_path, "sup_off", n_steps=1,
               **{**OFF, "society_diversity.enabled": "true"})
    kinds_on = diversity.nuisance_kinds_for(on, on.diversitycfg)
    kinds_off = diversity.nuisance_kinds_for(off, off.diversitycfg)
    assert "客引き" in kinds_off, "OFF で既定の nuisance 種が変わっている"
    assert "客引き" not in kinds_on, "ON でも無主体の客引きが供給されている"
    assert set(kinds_on) == set(kinds_off) - {"客引き"}, "他の種まで消えている"


def test_nuisance_touting_never_appears_in_l1_when_on(tmp_path):
    """ON の実ランで nuisance payload の kind に「客引き」が 1 件も出ない。"""
    sim = _sim(tmp_path, "sup_run", n_steps=24, n_agents=20,
               **{**ON, "society_diversity.enabled": "true",
                  "society_diversity.nuisance_prob": "0.9"})
    sim.run()
    ev = _kind(sim, "nuisance")
    assert ev, "対照スモークで nuisance が 0 件(検査が空回り)"
    assert all(e.payload["kind"] != "客引き" for e in ev)


# =========================================================================== #
# ⑩ 台帳・分類・安全弁
# =========================================================================== #
def test_all_new_kinds_are_registered_and_classified():
    """新 12 種が EVENT_KINDS(材料側 registration)と causality の両方に載っている。"""
    for k in NEW_KINDS:
        assert k in EVENT_KINDS, f"{k} が schema に未登録"
        assert k in C.CAUSE_OF_KIND, f"{k} が因果台帳に未分類"
        assert C.CAUSE_OF_KIND[k] in C.CAUSE_TYPES


def test_causality_table_has_no_stale_street_entries():
    """因果台帳に載せた街路の種は全部 EVENT_KINDS にも居る(片方だけ増えたら落ちる)。"""
    assert set(NEW_KINDS) <= set(C.CAUSE_OF_KIND) & set(EVENT_KINDS)


def test_registry_declares_the_toggle():
    ids = {f.id for f in R.FEATURES}
    assert "street_life.enabled" in ids
    f = next(f for f in R.FEATURES if f.id == "street_life.enabled")
    assert f.repro_tier == "strict" and f.affects_k is False


def test_shipped_default_is_off():
    cfg = load_config()
    assert bool(cfg.street_life.enabled) is False


def test_l1_budget_caps_events_per_step(tmp_path):
    """1 step の L1 件数は max_events_per_step で頭打ちになり、超過は捨てて数える。"""
    sim = _sim(tmp_path, "budget", n_steps=1, n_agents=24,
               **{**ON, "street_life.max_events_per_step": "2"})
    ags = sorted(sim.agents, key=lambda a: int(a.id))
    for a in ags[:6]:
        a.occupation = SL.MUSICIAN
    SL.bind(sim)
    for a in ags[:6]:
        _place(sim, a, a.street_post)
    for o in ags[6:]:
        o.loc = "outside"
    SL.phase(sim, 0, 19 * 60)
    assert len(_kind(sim, "street_performance")) == 2
    assert sim._sl_state["dropped"] >= 1


def test_provenance_reports_measured_volume(tmp_path):
    """summary の street_life キーに実測の件数が出る(ON のときだけ)。"""
    sim, actor, crowd, _ = _stage(tmp_path, "prov", SL.MUSICIAN, 19 * 60)
    SL.phase(sim, 0, 19 * 60)
    prov = SL.provenance(sim)
    assert prov and prov["sessions"] == 1 and prov["by_kind"]["street_performance"] == 1
    assert prov["fine_amount"] == 50000.0 and prov["zone_radius_m"] == 700.0


# =========================================================================== #
# ⑪ 決定論・乱数ゼロ
# =========================================================================== #
def test_on_is_deterministic_across_two_runs(tmp_path):
    """ON 同 seed 2 ランで L1 完全一致。"""
    outs = []
    for i in (1, 2):
        sim = _sim(tmp_path, f"det{i}", n_steps=48, n_agents=20, **ON)
        for j, a in enumerate(sorted(sim.agents, key=lambda z: int(z.id))[:8]):
            a.occupation = SL.STREET_OCCS[j % len(SL.STREET_OCCS)]
        sim.run()
        outs.append(_l1(sim))
    assert outs[0] == outs[1]


def test_module_draws_no_random_stream():
    """★AST 静的検査: 本 module に乱数の呼び出しが**識別子として存在しない**。"""
    src = MODULE.read_text(encoding="utf-8")
    tree = ast.parse(src)
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    for banned in ("stream", "random", "integers", "uniform", "shuffle", "choice",
                   "default_rng", "hub"):
        assert banned not in names, f"乱数の識別子 {banned} が street_life.py にある"


def test_module_makes_no_llm_call():
    """★AST 静的検査: generate() の呼び出しサイトが 1 つも無い(k 非依存)。"""
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert "generate" not in names
    assert "llm" not in {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}


def test_the_phase_hook_itself_adds_no_llm_call(tmp_path):
    """★R1: 名簿に路上の役割が 1 人も居なければ ON/OFF で generate() の呼数が完全一致。

    ★正直な限界(diversity / career G5 / 商業 H3 と同型): 名簿に路上の役割が**居る**ラン
      では、持ち場への束ね(work_node / 勤務窓 / 夜の居場所)が物理位置を変えるので
      co-location 経由で発火数が動きうる。それは「generate() の呼び出しサイトを足したか」
      という R1 の問いとは別の話で、本 module は呼び出しサイトを 1 つも持たない
      (``test_module_makes_no_llm_call`` の AST 検査が機械固定)。ここで固定するのは
      **フック自体が呼数を動かさない**ことである。
    """
    calls = []
    for name, ov in (("k_off", OFF), ("k_on", ON)):
        sim = _sim(tmp_path, name, n_steps=48, n_agents=20, **ov)
        sim.llm = _FixedLLM([json.dumps({"action": "stay"}, ensure_ascii=False)])
        sim.run()
        calls.append(sim.llm.calls)
    assert calls[0] == calls[1], f"フック自体が LLM 呼数を動かした: {calls}"
