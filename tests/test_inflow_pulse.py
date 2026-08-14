"""駅到着のパルス量子化(world.inflow_pulse。actor model Wave 3)。

境界の研究(Hänseler & Bierlaire の駅需要モデル / SUMO の運用)の要点=駅からの流入は
**平滑ではなく列車ごとの塊(platoon)**である。平滑な流入では改札の待ち行列もホームの
滞留も原理的に立たず、改札装置(world.devices.faregate)や駅員層が観測すべき現象その
ものが消える。本テストが機械固定するのは:

  (1) Transit.arrivals_between / arrivals_of_day = **読み取り専用の純関数**(再構成)
  (2) _snap_to_arrival の 3 規則(通常 / 始発前クランプ / 終電後クランプ)と同着の順序
  (3) 既定 OFF = 1 体も触らない(arrival_min 不変・属性が生えない・L1 の種が増えない)
  (4) ON = 決定論(同 seed 2 ラン完全一致)・縁ゲートウェイ不変・到着人数の保存

ネットワーク不使用・mock backend・合成ダイヤ(tmp_path)+ 既存の流入名簿/v6 地図。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from society.config import load_config
from society.engine.simulation import Simulation, _snap_to_arrival
from society.world.transit import Transit

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
ROSTER = DATA / "personas_100_inflow.json"
V6 = DATA / "shibuya_osm_v6.json"
REAL_TT = DATA / "transit_shibuya.json"

_needs_roster = pytest.mark.skipif(
    not (ROSTER.exists() and V6.exists()), reason="流入名簿 or v6 地図が未生成")


# --------------------------------------------------------------------------- #
# 合成ダイヤ / ラン生成
# --------------------------------------------------------------------------- #
def _timetable(tmp_path: Path, lines: list[dict], name: str = "tt.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps({"meta": {"note": "test"}, "lines": lines},
                            ensure_ascii=False), encoding="utf-8")
    return p


def _transit(tmp_path: Path, lines: list[dict], name: str = "tt.json") -> Transit:
    return Transit(_timetable(tmp_path, lines, name))


def _sim(tmp_path: Path, *, pulse: bool, tt: Path | None = None,
         n_steps: int = 24, seed: int = 42, n_agents: int = 40,
         name: str = "pulse", extra: list[str] | None = None) -> Simulation:
    ov = [f"run.seed={seed}", f"run.n_agents={n_agents}",
          f"run.n_steps={n_steps}", f"run.name={name}",
          f"agents.personas_file={ROSTER.as_posix()}",
          f"world.map={V6.as_posix()}",
          "world.inflow_pulse.enabled=%s" % ("true" if pulse else "false")]
    if tt is not None:
        ov.append(f"transit.file={tt.as_posix()}")
    ov += list(extra or [])
    return Simulation(load_config(overrides=ov), out_dir=tmp_path / name)


def _station_commuters(sim: Simulation) -> list:
    return [a for a in sim.agents
            if a.commute and a.commute_gateway == "station"
            and sim.city.station_node]


def _events(sim: Simulation) -> list[tuple]:
    return [(e.step, e.sim_min, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True))
            for e in sim.logger.events]


# --------------------------------------------------------------------------- #
# 1. Transit.arrivals_between = 読み取り専用の純関数(等間隔の再構成)
# --------------------------------------------------------------------------- #
def test_arrivals_reconstructed_from_first_last_headway(tmp_path):
    tr = _transit(tmp_path, [{"name": "A線", "first": "07:00", "last": "08:00",
                              "headway_min": 15}])
    assert tr.arrivals_of_day() == [(420, "A線"), (435, "A線"), (450, "A線"),
                                    (465, "A線"), (480, "A線")]


def test_arrivals_between_is_inclusive_on_both_ends(tmp_path):
    tr = _transit(tmp_path, [{"name": "A線", "first": "07:00", "last": "08:00",
                              "headway_min": 15}])
    assert tr.arrivals_between(435, 465) == [(435, "A線"), (450, "A線"),
                                             (465, "A線")]
    assert tr.arrivals_between(436, 449) == []
    assert tr.arrivals_between(500, 400) == []          # 逆順は空


def test_ties_are_ordered_by_line_name(tmp_path):
    """同一分に複数路線が着くとき、並びは**路線名の昇順**(スナップ先の一意性の根拠)。"""
    tr = _transit(tmp_path, [
        {"name": "Z線", "first": "07:00", "last": "07:00", "headway_min": 5},
        {"name": "A線", "first": "07:00", "last": "07:00", "headway_min": 5},
        {"name": "M線", "first": "07:00", "last": "07:00", "headway_min": 5}])
    assert tr.arrivals_of_day() == [(420, "A線"), (420, "M線"), (420, "Z線")]


def test_past_midnight_last_train_wraps_into_minute_of_day(tmp_path):
    """終電 24:36 はサービス日の 1476 分。分 of day の領域では翌 0:36 に現れる。"""
    tr = _transit(tmp_path, [{"name": "A線", "first": "24:30", "last": "24:36",
                              "headway_min": 3}])
    assert tr.arrivals_of_day() == [(30, "A線"), (33, "A線"), (36, "A線")]
    # 絶対分の領域では 1470/1473/1476 のまま(日跨ぎの表現を壊していない)
    assert tr.arrivals_between(1440, 1500) == [(1470, "A線"), (1473, "A線"),
                                               (1476, "A線")]


def test_arrivals_are_read_only_and_repeatable(tmp_path):
    """helper は状態を 1 バイトも変えない(2 回呼んで同値・lines も不変)。"""
    tr = _transit(tmp_path, [{"name": "A線", "first": "05:00", "last": "23:00",
                              "headway_min": 7}])
    before = json.dumps(tr.lines, ensure_ascii=False, sort_keys=True)
    first = tr.arrivals_of_day()
    second = tr.arrivals_of_day()
    assert first == second
    assert json.dumps(tr.lines, ensure_ascii=False, sort_keys=True) == before
    assert first is not second, "同じ list を返している(呼び出し側の変更が漏れる)"


def test_zero_or_missing_headway_does_not_hang(tmp_path):
    """headway 欠損/0 は 1 分刻みに丸める(無限ループを構造的に封じる)。"""
    tr = _transit(tmp_path, [{"name": "A線", "first": "07:00", "last": "07:03",
                              "headway_min": 0}])
    assert tr.arrivals_of_day() == [(420, "A線"), (421, "A線"), (422, "A線"),
                                    (423, "A線")]


def test_real_timetable_union_is_dense_but_each_line_is_sparse(tmp_path):
    """★沿線別が本質であることの根拠を数字で固定する。

    合併集合は 9 路線ぶんが重なってほぼ毎分に立つ(= 合併へスナップしても塊は
    できない)。一方**1 路線だけ**なら間隔 3〜6 分なので朝 3 時間で 30〜60 本しか
    無い = ここに寄せて初めて塊になる。"""
    arrivals = Transit(REAL_TT).arrivals_of_day()
    union = {m for m, _ln in arrivals if 420 <= m <= 600}
    assert len(union) >= 160, f"合併集合が密でない: {len(union)}/181"
    by_line: dict[str, set[int]] = {}
    for m, ln in arrivals:
        if 420 <= m <= 600:
            by_line.setdefault(ln, set()).add(m)
    assert by_line, "路線別に割れていない"
    assert max(len(v) for v in by_line.values()) <= 65, \
        f"1 路線でも朝の到着が多すぎる: {sorted((len(v), k) for k, v in by_line.items())}"


# --------------------------------------------------------------------------- #
# 1b. 居住沿線ラベル → 路線名 の照合(exact match は 0 件という実データの事実)
# --------------------------------------------------------------------------- #
def test_residence_labels_never_match_line_names_exactly():
    """★名簿の residence_line は**居住エリアのラベル**で路線名ではない(完全一致 0 件)。

    ここが 0 件だという事実が「部分文字列照合が要る」という設計判断の根拠なので、
    データが変わったら気づけるように機械固定する。"""
    roster = json.loads(ROSTER.read_text(encoding="utf-8"))
    labels = {str(r.get("residence_line", "")) for r in roster["personas"]
              if r.get("commute")} - {""}
    names = {str(ln["name"]) for ln in Transit(REAL_TT).lines}
    assert labels, "名簿に residence_line が無い(テスト前提)"
    assert labels & names == set(), f"完全一致が発生している: {labels & names}"


def test_line_for_residence_matches_labels_that_name_a_line():
    tr = Transit(REAL_TT)
    assert tr.line_for_residence("二子玉川・田園都市線沿線") == "東急田園都市線"
    assert tr.line_for_residence("横浜・東横線沿線") == "東急東横線"
    assert tr.line_for_residence("下北沢・井の頭線沿線") == "京王井の頭線"
    # 内回り/外回りの両方に 3 文字一致 → 路線名の昇順で内回り(決定論)
    assert tr.line_for_residence("新宿・山手線沿線") == "JR山手線(内回り)"


@pytest.mark.parametrize("label", ["世田谷方面", "杉並方面", "目黒方面",
                                   "川崎・多摩方面", "", "　"])
def test_line_for_residence_falls_back_when_no_line_is_named(label):
    """路線を名指さない居住ラベルは None = 呼び出し側は合併集合へ落ちる(安全側)。"""
    assert Transit(REAL_TT).line_for_residence(label) is None


def test_line_match_threshold_rejects_short_accidental_overlap(tmp_path):
    """2 文字以下の偶然一致(事業者接頭辞・地名の重なり)では結ばない。"""
    tr = _transit(tmp_path, [{"name": "東急○○線", "first": "05:00",
                              "last": "23:00", "headway_min": 5}])
    assert tr.line_for_residence("東急沿線ではない地域") is None   # 一致は「東急」= 2 文字
    assert tr.line_for_residence("○○線沿線") == "東急○○線"       # 一致は「○○線」= 3 文字


def test_line_for_residence_is_read_only(tmp_path):
    tr = _transit(tmp_path, [{"name": "A線", "first": "05:00", "last": "23:00",
                              "headway_min": 5}])
    before = json.dumps(tr.lines, ensure_ascii=False, sort_keys=True)
    tr.line_for_residence("A線沿線")
    assert json.dumps(tr.lines, ensure_ascii=False, sort_keys=True) == before


# --------------------------------------------------------------------------- #
# 2. _snap_to_arrival の 3 規則(純関数・乱数ゼロ)
# --------------------------------------------------------------------------- #
_A = [(420, "A線"), (420, "B線"), (435, "A線"), (480, "A線")]


@pytest.mark.parametrize("minute,expect", [
    (400, (420, "A線")),      # 始発前 → 始発へ繰り下げ(規則2)
    (420, (420, "A線")),      # ちょうど → その到着(同着は路線名昇順)
    (421, (435, "A線")),      # 通常 → 次の到着(規則1)
    (435, (435, "A線")),
    (436, (480, "A線")),
    (480, (480, "A線")),
    (481, (480, "A線")),      # 終電後 → 最終到着へ繰り上げ(規則3)
    (1439, (480, "A線")),
])
def test_snap_rules(minute, expect):
    assert _snap_to_arrival(_A, minute) == expect


def test_snap_on_empty_timetable_is_none():
    assert _snap_to_arrival([], 500) is None


def test_snap_is_pure_and_order_independent():
    """同じ入力なら何度呼んでも同じ・入力 list を書き換えない(個体の処理順に非依存)。"""
    src = list(_A)
    got = [_snap_to_arrival(_A, m) for m in (500, 400, 421, 400)]
    assert got == [(480, "A線"), (420, "A線"), (435, "A線"), (420, "A線")]
    assert _A == src


# --------------------------------------------------------------------------- #
# 3. 既定 OFF = 1 体も触らない
# --------------------------------------------------------------------------- #
@_needs_roster
def test_off_grows_no_attribute_and_keeps_arrival_min(tmp_path):
    off = _sim(tmp_path, pulse=False, n_steps=1, name="off")
    on = _sim(tmp_path, pulse=True, n_steps=1, name="on",
              tt=_timetable(tmp_path, [{"name": "A線", "first": "09:00",
                                        "last": "09:00", "headway_min": 5}]))
    coms = [a for a in off.agents if a.commute]
    assert coms, "commuter が名簿から生成されていない(テスト前提)"
    for a in off.agents:
        assert not hasattr(a, "pulse_train_min"), \
            f"OFF なのに agent {a.id} に属性が生えている"
        assert not hasattr(a, "pulse_line")
        assert not hasattr(a, "pulse_line_src")
    # OFF の arrival_min は persona 由来のまま = ON(全員 09:00 へ集約)と別物
    off_arr = {a.id: a.arrival_min for a in off.agents if a.commute}
    on_arr = {a.id: a.arrival_min for a in on.agents if a.commute}
    assert off_arr != on_arr, "合成ダイヤでも到着が動いていない(ゲートが効いていない)"


@_needs_roster
def test_off_run_is_identical_to_two_off_runs(tmp_path):
    """OFF の決定論(golden バイト一致の局所版。golden 本体は test_scenario 側)。"""
    a = _sim(tmp_path, pulse=False, n_steps=24, name="off_a")
    b = _sim(tmp_path, pulse=False, n_steps=24, name="off_b")
    a.run()
    b.run()
    assert _events(a) == _events(b)


@_needs_roster
def test_on_adds_no_new_event_kinds(tmp_path):
    off = _sim(tmp_path, pulse=False, n_steps=24, name="k_off")
    on = _sim(tmp_path, pulse=True, n_steps=24, name="k_on")
    off.run()
    on.run()
    assert {e.kind for e in on.logger.events} <= {e.kind for e in off.logger.events}, \
        "ON で新しいイベント種が増えている(L1 契約違反)"


# --------------------------------------------------------------------------- #
# 4. ON の正しさ(スナップ・クランプ・縁ゲートウェイ・人数保存・決定論)
# --------------------------------------------------------------------------- #
@_needs_roster
def test_on_snaps_station_commuters_to_expected_minutes(tmp_path):
    """★スナップの正しさ: 合成ダイヤ(15 分間隔)で **1 体ずつ**期待値と厳密一致。"""
    tt = _timetable(tmp_path, [{"name": "A線", "first": "05:00", "last": "23:00",
                                "headway_min": 15}])
    off = _sim(tmp_path, pulse=False, n_steps=1, name="s_off")
    on = _sim(tmp_path, pulse=True, tt=tt, n_steps=1, name="s_on")
    arrivals = Transit(tt).arrivals_of_day()
    on_by_id = {a.id: a for a in on.agents}
    checked = 0
    for a in _station_commuters(off):
        b = on_by_id[a.id]
        expect = _snap_to_arrival(arrivals, a.arrival_min)
        assert (b.arrival_min, b.pulse_line) == expect, \
            f"agent {a.id}: {a.arrival_min} -> {b.arrival_min} (期待 {expect})"
        assert b.pulse_train_min == expect[0]
        assert b.arrival_min % 15 == 0, "15 分刻みのダイヤに載っていない"
        checked += 1
    assert checked >= 5, f"駅ゲートウェイの commuter が少なすぎる: {checked}"


@_needs_roster
def test_on_snaps_to_the_agents_own_line_on_the_real_timetable(tmp_path):
    """★沿線別スナップ: 路線を名指す居住ラベルの個体は**その路線のダイヤ**に乗る。

    合併集合(ほぼ毎分)ではなく路線別(3〜6 分間隔)へ寄せるのが本質なので、
    照合が当たった個体は「その路線の到着時刻の集合」に属していなければならない。"""
    tr = Transit(REAL_TT)
    by_line: dict[str, set[int]] = {}
    for m, ln in tr.arrivals_of_day():
        by_line.setdefault(ln, set()).add(m)
    on = _sim(tmp_path, pulse=True, n_steps=1, name="pl_on")
    matched = [a for a in _station_commuters(on)
               if getattr(a, "pulse_line_src", None) == "residence"]
    assert matched, "沿線照合が 1 件も当たっていない(テスト前提)"
    for a in matched:
        assert a.pulse_line == tr.line_for_residence(a.residence_line)
        assert a.arrival_min in by_line[a.pulse_line], (a.id, a.arrival_min)
        assert a.pulse_train_min == a.arrival_min


@_needs_roster
def test_union_fallback_is_used_when_the_label_names_no_line(tmp_path):
    """路線を名指さない居住ラベル(「○○方面」)は合併集合へ落ちる=出自を src で区別する。"""
    tr = Transit(REAL_TT)
    on = _sim(tmp_path, pulse=True, n_steps=1, name="uf_on")
    srcs = {a.id: a.pulse_line_src for a in _station_commuters(on)}
    assert set(srcs.values()) == {"residence", "union"}, \
        f"両方の出自が出ていない: {set(srcs.values())}"
    for a in _station_commuters(on):
        named = tr.line_for_residence(a.residence_line) is not None
        assert a.pulse_line_src == ("residence" if named else "union"), a.id


@_needs_roster
def test_snap_is_idempotent(tmp_path):
    """★冪等: 2 回スナップしても値が動かない(pool の再入場で二重に呼ばれても安全)。"""
    on = _sim(tmp_path, pulse=True, n_steps=1, name="idem")
    before = [(a.id, a.arrival_min, a.pulse_line, a.pulse_line_src)
              for a in _station_commuters(on)]
    assert before, "駅ゲートウェイの commuter が居ない(テスト前提)"
    for a in on.agents:
        on._snap_agent_arrival(a)
        on._snap_agent_arrival(a)
    after = [(a.id, a.arrival_min, a.pulse_line, a.pulse_line_src)
             for a in _station_commuters(on)]
    assert after == before


@_needs_roster
def test_clamp_before_first_train(tmp_path):
    """始発 09:00 のダイヤ = 朝の到着は全員 09:00 以降へ**繰り下がる**(規則2)。"""
    tt = _timetable(tmp_path, [{"name": "A線", "first": "09:00", "last": "23:00",
                                "headway_min": 30}])
    off = _sim(tmp_path, pulse=False, n_steps=1, name="c1_off")
    on = _sim(tmp_path, pulse=True, tt=tt, n_steps=1, name="c1_on")
    on_by_id = {a.id: a for a in on.agents}
    early = [a for a in _station_commuters(off) if a.arrival_min < 540]
    assert early, "始発前に着く commuter が居ない(テスト前提)"
    for a in early:
        assert on_by_id[a.id].arrival_min == 540, a.id
        assert on_by_id[a.id].pulse_train_min == 540


@_needs_roster
def test_clamp_after_last_train(tmp_path):
    """終電 06:00 のダイヤ = 朝の到着は全員 06:00 へ**繰り上がる**(規則3。唯一早まる規則)。"""
    tt = _timetable(tmp_path, [{"name": "A線", "first": "05:00", "last": "06:00",
                                "headway_min": 30}])
    off = _sim(tmp_path, pulse=False, n_steps=1, name="c2_off")
    on = _sim(tmp_path, pulse=True, tt=tt, n_steps=1, name="c2_on")
    on_by_id = {a.id: a for a in on.agents}
    late = [a for a in _station_commuters(off) if a.arrival_min > 360]
    assert late, "終電後に着く commuter が居ない(テスト前提)"
    for a in late:
        assert on_by_id[a.id].arrival_min == 360, a.id


@_needs_roster
def test_edge_gateway_commuters_are_untouched(tmp_path):
    """縁ゲートウェイ(徒歩流入)は 1 体も触らない = 属性も生えない。"""
    tt = _timetable(tmp_path, [{"name": "A線", "first": "09:00", "last": "09:00",
                                "headway_min": 5}])
    off = _sim(tmp_path, pulse=False, n_steps=1, name="e_off")
    on = _sim(tmp_path, pulse=True, tt=tt, n_steps=1, name="e_on")
    on_by_id = {a.id: a for a in on.agents}
    edges = [a for a in off.agents if a.commute and a.commute_gateway == "edge"]
    assert edges, "縁ゲートウェイの commuter が居ない(テスト前提)"
    for a in edges:
        b = on_by_id[a.id]
        assert b.arrival_min == a.arrival_min, a.id
        assert not hasattr(b, "pulse_train_min")
        assert not hasattr(b, "pulse_line")


@_needs_roster
def test_non_commuters_are_untouched(tmp_path):
    tt = _timetable(tmp_path, [{"name": "A線", "first": "09:00", "last": "09:00",
                                "headway_min": 5}])
    off = _sim(tmp_path, pulse=False, n_steps=1, name="n_off")
    on = _sim(tmp_path, pulse=True, tt=tt, n_steps=1, name="n_on")
    on_by_id = {a.id: a for a in on.agents}
    for a in off.agents:
        if a.commute:
            continue
        assert on_by_id[a.id].arrival_min == a.arrival_min
        assert not hasattr(on_by_id[a.id], "pulse_train_min")


@_needs_roster
def test_on_is_deterministic_across_two_runs(tmp_path):
    a = _sim(tmp_path, pulse=True, n_steps=24, name="d_a")
    b = _sim(tmp_path, pulse=True, n_steps=24, name="d_b")
    a.run()
    b.run()
    assert _events(a) == _events(b)
    assert ([(x.id, x.arrival_min, getattr(x, "pulse_line", None)) for x in a.agents]
            == [(x.id, x.arrival_min, getattr(x, "pulse_line", None)) for x in b.agents])


@_needs_roster
def test_commuter_arrival_count_is_conserved(tmp_path):
    """★人数保存: 1 日に駅から流入する commuter の**人数**は ON/OFF で変わらない。

    量子化は到着の**時刻**を動かすだけで、誰が来るかを増減させない。実ダイヤ(始発
    04:34 / 終電 24:40)ではクランプが起きないので恒等的に一致するべき。"""
    off = _sim(tmp_path, pulse=False, n_steps=48, name="cc_off")
    on = _sim(tmp_path, pulse=True, n_steps=48, name="cc_on")
    off.run()
    on.run()

    def first_enters(sim):
        ids = {a.id for a in sim.agents if a.commute}
        seen = {}
        for e in sim.logger.events:
            if e.kind == "enter_area" and e.agent_id in ids \
                    and (e.payload or {}).get("via") == "train":
                seen.setdefault(e.agent_id, e.sim_min % 1440)
        return seen

    off_e, on_e = first_enters(off), first_enters(on)
    assert off_e, "OFF で駅からの流入が 1 件も出ていない(テスト前提)"
    assert set(off_e) == set(on_e), "流入した commuter の顔ぶれが変わっている"
    assert len(off_e) == len(on_e)


# --------------------------------------------------------------------------- #
# 5. L1: enter_area payload の追記(新 kind ゼロ・OFF は 1 バイトも増えない)
# --------------------------------------------------------------------------- #
@_needs_roster
def test_enter_area_payload_gains_train_keys_only_when_on(tmp_path):
    off = _sim(tmp_path, pulse=False, n_steps=24, name="pl_off_ev")
    on = _sim(tmp_path, pulse=True, n_steps=24, name="pl_on_ev")
    off.run()
    on.run()

    def enters(sim):
        ids = {a.id for a in sim.agents if a.commute
               and a.commute_gateway == "station"}
        return [e for e in sim.logger.events
                if e.kind == "enter_area" and e.agent_id in ids
                and (e.payload or {}).get("via") == "train"]

    off_e, on_e = enters(off), enters(on)
    assert off_e and on_e, "駅からの流入が出ていない(テスト前提)"
    for e in off_e:
        assert "train_min" not in e.payload and "line" not in e.payload, \
            f"OFF の payload に欄が増えている: {e.payload}"
    by_id = {a.id: a for a in on.agents}
    for e in on_e:
        assert e.payload["train_min"] == by_id[e.agent_id].pulse_train_min
        assert e.payload["line"] == by_id[e.agent_id].pulse_line
    # 縁ゲートウェイ(徒歩)の enter_area には ON でも欄が増えない
    edge_ids = {a.id for a in on.agents if a.commute
                and a.commute_gateway == "edge"}
    for e in on.logger.events:
        if e.kind == "enter_area" and e.agent_id in edge_ids:
            assert "train_min" not in (e.payload or {}), e.payload


# --------------------------------------------------------------------------- #
# 6. pool の日次ローテーションで途中入場する個体も同じ規則で量子化される
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def small_pool(tmp_path_factory):
    """P5 生成器で小プールを tmp に生成(実プールは触らない。test_pool_rotation と同型)。"""
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    import build_persona_pool as bpp

    out = tmp_path_factory.mktemp("pulse_pool")
    orgs = json.loads((REPO / "data" / "organizations_shibuya_wide11k.json")
                      .read_text(encoding="utf-8"))
    pop = json.loads((REPO / "data" / "shibuya_population.json")
                     .read_text(encoding="utf-8"))
    bpp.build_pool(out, seed=42, fraction=0.001, orgs=orgs, pop=pop,
                   total_target=1_000_000)
    return out


def test_pool_agents_are_snapped_too(tmp_path, small_pool):
    """★pool ON: 日境界のローテーションで途中入場する commuter も量子化される。

    起動時 1 回の _init_inflow_commuters を通らない経路(build_pool_agent)を
    塞いでおかないと、pool ON のランだけ流入が平滑に戻る。"""
    cfg = load_config(["run.seed=42", "run.n_agents=10", "run.n_steps=110",
                       "run.name=pulse_pool", "model.backend=mock",
                       "pool.enabled=true", f"pool.dir={Path(small_pool).as_posix()}",
                       "pool.present_cap=400",
                       "world.inflow_pulse.enabled=true"])
    sim = Simulation(cfg, out_dir=tmp_path / "pulse_pool")
    assert sim.city.station_node, "既定地図に駅ノードが無い(テスト前提)"
    sim.run()
    tr = sim.transit
    by_line: dict[str, set[int]] = {}
    for m, ln in tr.arrivals_of_day():
        by_line.setdefault(ln, set()).add(m)
    snapped = [a for a in sim.agents
               if getattr(a, "pulse_train_min", None) is not None]
    assert snapped, "pool 経路の commuter が 1 体も量子化されていない"
    for a in snapped:
        assert a.commute and a.commute_gateway == "station"
        assert a.arrival_min == a.pulse_train_min
        assert a.arrival_min in by_line[a.pulse_line], (a.id, a.arrival_min)


@_needs_roster
def test_pulse_survives_to_the_next_day(tmp_path):
    """arrival_min を書き換えているので **2 日目以降**(scheduler の _steps_until_tod
    経路)もパルス化される = 初日だけの細工ではない。"""
    tt = _timetable(tmp_path, [{"name": "A線", "first": "05:00", "last": "23:00",
                                "headway_min": 30}])
    on = _sim(tmp_path, pulse=True, tt=tt, n_steps=1, name="p_next")
    for a in _station_commuters(on):
        assert a.arrival_min % 30 == 0, (a.id, a.arrival_min)


# --------------------------------------------------------------------------- #
# 1c. 1 本ごとの実発車時刻(第114 レーン 3b。transit.real_departures)
#
# 何が壊れていたか: ダイヤは路線ごとの (始発, 終電, 中央値間隔) の 3 値しか持たず、
# ODPT 駅時刻表や静的 GTFS から**実発車時刻を読んだ後で**その 3 値へ畳んでいた。つまり
# 「実ダイヤに差し替え済み」と言いながら、シムが見ていたのは等間隔の再構成だった。
# --------------------------------------------------------------------------- #
ODPT_TT = DATA / "transit_odpt.json"

_needs_odpt = pytest.mark.skipif(not ODPT_TT.exists(), reason="ODPT 実ダイヤ未生成")

_DEP_LINE = [{"name": "A線", "first": "07:00", "last": "07:30", "headway_min": 10,
              "departures": [420, 421, 422, 450, 451, 452]}]


def test_departures_are_ignored_unless_the_toggle_is_on(tmp_path):
    """既定 OFF: departures 列が在っても等間隔の再構成のまま(バイト一致)。"""
    tr = Transit(_timetable(tmp_path, _DEP_LINE, "dep_off.json"))
    assert tr.real_departures is False
    got = [m for m, _ in tr.arrivals_between(0, 1439)]
    assert got == [420, 430, 440, 450]            # 07:00 から 10 分刻み


def test_departures_replace_the_reconstruction_when_on(tmp_path):
    """ON: 実発車時刻がそのまま到着表になる(再構成は 1 本も混ざらない)。"""
    tr = Transit(_timetable(tmp_path, _DEP_LINE, "dep_on.json"), real_departures=True)
    got = [m for m, _ in tr.arrivals_between(0, 1439)]
    assert got == [420, 421, 422, 450, 451, 452]
    # 日跨ぎも従来と同じ規約(サービス日の分にそのまま base を足す)
    assert [m for m, _ in tr.arrivals_between(1440, 2879)] == \
        [1440 + t for t in (420, 421, 422, 450, 451, 452)]


def test_lines_without_departures_fall_back_to_the_reconstruction(tmp_path):
    """列を持たない路線は ON でも従来どおり = 混在が正しく成立する。"""
    lines = list(_DEP_LINE) + [{"name": "B線", "first": "07:00", "last": "07:20",
                                "headway_min": 10}]
    tr = Transit(_timetable(tmp_path, lines, "dep_mix.json"), real_departures=True)
    got = tr.arrivals_between(0, 1439)
    assert [m for m, ln in got if ln == "A線"] == [420, 421, 422, 450, 451, 452]
    assert [m for m, ln in got if ln == "B線"] == [420, 430, 440]
    assert got == sorted(got)                      # 並びは (時刻, 路線名) の辞書順のまま


def test_departure_list_is_canonicalised(tmp_path):
    """重複・逆順・型違い・範囲外を落とす純関数(壊れたデータで世界が歪まない)。"""
    from society.world.transit import _canon_departures
    assert _canon_departures(None) == () and _canon_departures([]) == ()
    assert _canon_departures([430, 420, 430, "425", -5, 9999, None]) == (420, 425, 430)


def test_arrivals_stay_read_only_and_repeatable(tmp_path):
    """ON でも到着表は読み取り専用の純関数(2 度呼んで同一・ファイルを書き換えない)。"""
    tr = Transit(_timetable(tmp_path, _DEP_LINE, "dep_pure.json"), real_departures=True)
    before = json.loads((tmp_path / "dep_pure.json").read_text(encoding="utf-8"))
    a = tr.arrivals_between(0, 1439)
    b = tr.arrivals_between(0, 1439)
    assert a == b
    assert json.loads((tmp_path / "dep_pure.json").read_text(encoding="utf-8")) == before


@_needs_odpt
def test_shipped_odpt_timetable_carries_the_real_departures():
    """同梱の実ダイヤが 1 本ごとの時刻を持つ(畳んだままの退行を止める)。"""
    doc = json.loads(ODPT_TT.read_text(encoding="utf-8"))
    real = [ln for ln in doc["lines"] if "実ダイヤ" in str(ln.get("source", ""))]
    approx = [ln for ln in doc["lines"] if "実ダイヤ" not in str(ln.get("source", ""))]
    assert len(real) >= 6, f"実ダイヤ路線が減っている: {len(real)}"
    for ln in real:
        deps = ln.get("departures")
        assert deps and len(deps) > 100, (ln["name"], deps and len(deps))
        assert deps == sorted(deps) and len(set(deps)) == len(deps)
    # ★近似路線には列を**作らない**(空列を置くと「実時刻を持っている」と読めてしまう)
    for ln in approx:
        assert "departures" not in ln, ln["name"]


@_needs_odpt
def test_real_departures_restore_the_morning_peak():
    """★塞いだ穴そのもの: 単一の中央値間隔が朝を過少・日中を過大に代表していた。

    実測(本テストが数字ごと固定する): 渋谷駅の到着は朝 7:30-9:00 で **増え**、
    1 日の総数は **減る**。前者はラッシュの厚みが戻ったこと、後者は日中/深夜の
    水増しが消えたことを意味する。"""
    approx = Transit(ODPT_TT)
    real = Transit(ODPT_TT, real_departures=True)
    a_day = approx.arrivals_between(0, 1439)
    r_day = real.arrivals_between(0, 1439)
    rush = (450, 540)                              # 7:30-9:00

    def n_rush(rows):
        return sum(1 for m, _ in rows if rush[0] <= m <= rush[1])

    assert n_rush(r_day) > n_rush(a_day) * 1.10, \
        f"朝ラッシュが厚くなっていない: {n_rush(a_day)} → {n_rush(r_day)}"
    assert len(r_day) < len(a_day), \
        f"1 日の総本数が減っていない: {len(a_day)} → {len(r_day)}"
    # 実ダイヤ路線は朝の実測間隔が中央値間隔より**短い**(過少代表の直接の証拠)
    doc = json.loads(ODPT_TT.read_text(encoding="utf-8"))
    tighter = 0
    for ln in doc["lines"]:
        deps = ln.get("departures")
        if not deps:
            continue
        peak = [t for t in deps if rush[0] <= t <= rush[1]]
        gaps = [b - a for a, b in zip(peak, peak[1:])]
        if gaps and sorted(gaps)[len(gaps) // 2] < int(ln["headway_min"]):
            tighter += 1
    assert tighter >= 5, f"朝の間隔が中央値より短い路線が {tighter} 本しかない"


@_needs_odpt
def test_off_run_is_unchanged_by_the_new_field(tmp_path):
    """既定 OFF のランは departures 入りのダイヤでも L1 が完全一致(バイト一致)。"""
    if not (ROSTER.exists() and V6.exists()):
        pytest.skip("流入名簿 or v6 地図が未生成")
    a = _sim(tmp_path, pulse=True, tt=ODPT_TT, n_steps=48, name="dep_run_def")
    a.run()
    b = _sim(tmp_path, pulse=True, tt=ODPT_TT, n_steps=48, name="dep_run_off",
             extra=["transit.real_departures=false"])
    b.run()
    assert _events(a) == _events(b)
