"""Wave 4 II-1(ラッシュ時の車内 = シミュレートされた空間 ``transit_interior``)のテスト。

守るもの(検収基準の順)
  ① OFF(既定)= ゴールデン L1 バイト一致・属性も state も記憶も生えない
     + **パルスが OFF なら本層 ON でも動かない**(2 つの ON の論理積)
  ② 車両: 編成両数の中に収まる / 混雑アンカーに対して均される / ★銀座線だけ 16 m 3 扉
  ③ 区画: 充填順(ドア脇 → 座席前 → 通路は最後)と、押し込み帯のシェア抽選
     / **着席者は立位密度に数えない**
  ④ 記憶は 1 乗車 1 行まで・文面に数字ゼロ・(路線, 混雑段, 行動)の純関数
  ⑤ ★**会話を 1 件も増やさない**(speak/hear の件数と generate 呼数が ON/OFF 完全一致・
     module に会話の識別子が存在しない = AST 固定)
  ⑥ 同席: 1 日 1 対 1 件の重複排除 + 1 日総量の上限(超過は件数として正直に残す)
  ⑦ 車掌: 巡回が決定論 / 停車時間への作用は既定 0(観測のみ)= transit_staff バイト一致
  ⑧ ON 同 seed 2 ラン一致
  ⑨ 宣言: registry / timeconv / causality / L1 スキーマ
  ⑩ 名簿生成器が**車掌**を作る(死んでいた conf キーの修復)

ネットワーク不使用・mock backend・既存の流入名簿 + v6 地図。
"""
from __future__ import annotations

import ast
import collections
import json
import sys
from pathlib import Path

import pytest

from society import registry as R
from society import timeconv as TC
from society import transit_interior as TI
from society import transit_staff as TS
from society.config import load_config
from society.engine.simulation import Simulation
from society.observer import causality as CA
from society.observer.schema import EVENT_KINDS
from society.world import transit as WT

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
ROSTER = DATA / "personas_100_inflow.json"
V6 = DATA / "shibuya_osm_v6.json"
GOLDEN = Path(__file__).resolve().parent / "data" / "golden_baseline_l1.json"

KINDS = ("train_ride", "train_copresence", "train_patrol")

# test_traces.py:45 と同じ「意図的な既定挙動追加」の中立化(ゴールデン比較用)
_GOLDEN_NEUTRAL = {"economy.wages.自営": 0, "planning.enabled": "false",
                   "transit_ride.taxi.enabled": "false",
                   "transit_ride.bus.enabled": "false", "rules.enabled": "false"}

_needs_roster = pytest.mark.skipif(
    not (ROSTER.exists() and V6.exists()), reason="流入名簿 or v6 地図が未生成")


# --------------------------------------------------------------------------- #
# 共通ヘルパ
# --------------------------------------------------------------------------- #
def _cfg(name, n_steps=144, n_agents=15, **ov):
    dot = ["run.seed=42", f"run.n_agents={n_agents}", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock",
           f"observer.snapshot_every={n_steps}"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def _sim(tmp_path, name, n_steps=144, n_agents=15, **ov):
    return Simulation(_cfg(name, n_steps, n_agents, **ov), out_dir=tmp_path / name)


def _inflow(tmp_path, name, *, on: bool, pulse: bool = True, n_steps=144,
            n_agents=120, seed=42, **ov):
    """流入名簿 + v6 地図 + パルス量子化の実ラン(車内層の唯一の素材)。"""
    dot = ["run.seed=%d" % seed, f"run.n_agents={n_agents}",
           f"run.n_steps={n_steps}", f"run.name={name}", "model.backend=mock",
           f"observer.snapshot_every={n_steps}",
           f"agents.personas_file={ROSTER.as_posix()}",
           f"world.map={V6.as_posix()}",
           "world.inflow_pulse.enabled=%s" % ("true" if pulse else "false"),
           "transit_interior.enabled=%s" % ("true" if on else "false")]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _kind(sim, kind):
    return [e for e in sim.logger.events if e.kind == kind]


class _FixedLLM:
    """**プロンプト非依存**の巡回応答スタブ(test_traces.py と同型)。

    mock は prompt 全文で乱数を引くので、記憶が 1 行増えるだけで世界が分岐して
    呼数の比較にならない。応答を**呼び出し番号の関数**にすると「1 行増えたこと」が
    行動列を変えない世界になり、呼数と発話件数を素で比べられる。
    """

    name = "fixed"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.hits = 0                              # summary.json が読む欄(キャッシュなし)
        self.prompts: list[str] = []

    def generate(self, prompt, *, rng_key, temperature, max_tokens, think=False):
        out = self.responses[self.calls % len(self.responses)]
        self.calls += 1
        self.prompts.append(prompt)
        return out, str(self.calls), False


_TALKY = [json.dumps(o, ensure_ascii=False) for o in (
    {"action": "speak", "text": "おはようございます"},
    {"action": "speak", "text": "そうですね"},
    {"action": "move_to", "dest": "", "stay_steps": 1},
    {"action": "speak", "text": "また今度"},
)]


def _live_strings(module) -> list[str]:
    """module の中で**実際に評価される**文字列定数(docstring は除く)。

    地名禁止ガードの類は「散文ではなく識別子・値だけを見る」のが本 repo の作法
    (tests/test_traces.py の AST 静的検査)。docstring は世界に 1 バイトも出ないので
    対象から外し、代入・引数・辞書の値として残る文字列だけを検査する。
    """
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    docs = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            body = getattr(node, "body", None) or []
            if body and isinstance(body[0], ast.Expr) \
                    and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                docs.add(id(body[0].value))
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and id(n) not in docs]


def _live_cfg(**ov) -> dict:
    """conf をロードして車内層の cfg を正準化(純関数テスト用の実 conf 値)。"""
    return TI.build_cfg(load_config([f"{k}={v}" for k, v in ov.items()]
                                    ).get("transit_interior"))


# =========================================================================== #
# (A) 出荷既定・宣言(検収基準 ①の前段 / ⑨)
# =========================================================================== #
def test_shipped_default_is_off():
    cfg = load_config()
    assert bool(cfg.transit_interior.enabled) is False
    assert int(cfg.transit_interior.ride_minutes) == 15
    assert float(cfg.transit_interior.conductor.dwell_load_weight) == 0.0


def test_registry_declares_every_bool_toggle():
    ids = {f.id: f for f in R.FEATURES}
    parent = ids["transit_interior.enabled"]
    assert parent.repro_tier == "strict"
    assert parent.affects_k is False     # generate() の呼び出しサイトを 1 つも足さない
    assert parent.fingerprint_risk == "possible"   # 降車後の記憶に定型 1 行
    assert parent.off_value is False
    for sub in ("transit_interior.include_plan_returnees",
                "transit_interior.memory.enabled",
                "transit_interior.copresence.enabled",
                "transit_interior.copresence.adjacent_only",
                "transit_interior.conductor.enabled"):
        assert sub in ids, f"{sub} が未宣言(bool リーフの検出テストが落ちる)"
        assert ids[sub].repro_tier == "strict"
        assert ids[sub].affects_k is False


def test_event_kinds_and_causality_are_declared():
    for k in KINDS:
        assert k in EVENT_KINDS, f"{k} が L1 スキーマに無い"
        assert k in CA.CAUSE_OF_KIND, f"{k} が因果台帳に無い"
    # 高頻度の受動観測は帰属率の分母から外れる語に置く(traffic_flow と同じ配慮)
    assert CA.CAUSE_OF_KIND["train_ride"] == CA.PHYSICS
    assert CA.CAUSE_OF_KIND["train_copresence"] == CA.PHYSICS
    assert CA.PHYSICS in CA.TICK_CAUSES
    assert CA.CAUSE_OF_KIND["train_patrol"] == CA.DEVICE      # dwell_decision と同じ扱い


def test_timeconv_classifies_new_conf_keys():
    by = {k: (c, w) for k, c, w in TC.TABLE}
    assert by["transit_interior.conductor.patrol_interval_steps"][0] == TC.STEPS
    assert by["transit_interior.comfort.fatigue_per_step"][0] == TC.RATE
    # 分・分 of day・件数・per-day は Δt 非依存(理由つきで宣言してある)
    for k in ("transit_interior.ride_minutes",
              "transit_interior.congestion_scale.windows",
              "transit_interior.copresence.max_pairs_per_day",
              "transit_interior.copresence.max_pairs_per_car",
              "transit_interior.comfort.fatigue_max"):
        assert by[k][0] == TC.INVARIANT, k
        assert by[k][1].strip(), f"{k} の不変理由が空"


def test_module_has_no_llm_no_rng_and_no_conversation():
    """★**会話経路を 1 本も作らない**ことと**乱数を 1 本も引かない**ことの静的検査。

    散文(docstring / コメント)は対象外 — 実際に評価される識別子だけを見る。
    """
    tree = ast.parse(Path(TI.__file__).read_text(encoding="utf-8"))
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    funcs = {n.name for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    idents = attrs | names | funcs
    # LLM も乱数 stream も引かない
    assert "generate" not in idents and "llm" not in attrs
    assert "stream" not in attrs and "hub" not in attrs and "random" not in idents
    # 会話・発話・関係の書き込み経路をコードとして持たない(静かな同席)
    for bad in ("speak", "talk", "convers", "utter", "hearer", "dialog"):
        hit = sorted(i for i in idents if bad in i.lower())
        assert hit == [], f"車内に会話経路を作ってしまった: {hit}"
    # 欲求ゲージ(= LLM 発火数)には触らない = affects_k=False の機械的根拠
    assert "drive" not in attrs and "opinion" not in attrs


def test_frozen_files_are_untouched():
    from society.observer import metrics_spec as MS
    for rel in MS.SPEC_FILES:
        text = (REPO / rel).read_text(encoding="utf-8")
        assert "transit_interior" not in text and "train_ride" not in text, rel


# =========================================================================== #
# (B) 純関数(sim を作らずに検査できる層)
# =========================================================================== #
def test_line_table_is_data_and_ginza_keeps_its_own_constants():
    """★路線ごとの諸元は **conf のデータ**で、銀座線だけ 16 m 3 扉 6 両。

    20 m 4 扉の定数を流用していないことを機械固定する(流用すると定員も
    区画数も座席数も全部ずれる)。
    """
    cfg = _live_cfg()
    ginza = TI.spec_for(cfg, "東京メトロ銀座線")
    assert (ginza["cars"], ginza["car_len_m"], ginza["doors"]) == (6, 16.0, 3)
    assert ginza["capacity"] == 104 and ginza["seats"] == 40
    default = TI.spec_for(cfg, "この路線は表に無い")
    assert (default["cars"], default["car_len_m"], default["doors"]) == (10, 20.0, 4)
    # 20 m 4 扉の路線は銀座線と**別の**定数を持つ
    saikyo = TI.spec_for(cfg, "JR埼京線")
    assert saikyo["doors"] == 4 and saikyo["car_len_m"] == 20.0
    assert saikyo["congestion"] > ginza["congestion"]      # 167% > 153%
    yamanote = TI.spec_for(cfg, "JR山手線(内回り)")
    assert yamanote["cars"] == 11                          # E235 は 11 両
    # ★src に路線名リテラルを持たない(表はデータ側 = envpack ドクトリン)。
    #   散文(docstring)は対象外 — **実際に評価される文字列**だけを見る
    #   (tests/test_traces.py の AST 静的検査と同じ線引き)。
    for text in _live_strings(TI):
        for name in ("山手", "銀座", "埼京", "井の頭", "田園都市", "東横", "渋谷"):
            assert name not in text, f"基盤に路線名/地名リテラル {name}: {text!r}"


def test_longest_match_wins_and_unknown_falls_back():
    cfg = TI.build_cfg({"enabled": True, "lines": [
        {"match": "線", "cars": 3}, {"match": "特別線", "cars": 7},
        {"match": "", "cars": 99}]})
    assert len(cfg["lines"]) == 2                 # match 空の行は捨てる
    assert TI.spec_for(cfg, "特別線")["cars"] == 7   # 最長一致
    assert TI.spec_for(cfg, "普通線")["cars"] == 3
    assert TI.spec_for(cfg, "バス")["cars"] == TI.DEFAULT_SPEC["cars"]


def test_ride_key_has_no_day_component():
    """★乗車の同一性キーに日を混ぜない(終電の日跨ぎで割り直し + 記憶 2 行になる)。"""
    assert TI._ride_key("A線", 1435) == TI._ride_key("A線", 1435)
    assert TI._ride_key("A線", 1435) != TI._ride_key("A線", 5)
    assert TI._ride_key("A線", 5) != TI._ride_key("B線", 5)


def test_zone_kind_vocabulary_is_complete():
    assert set(TI.ZONE_KINDS) == {TI.SEAT, TI.DOOR, TI.FRONT, TI.AISLE}
    spec = TI.spec_for(_live_cfg(), "JR埼京線")
    kinds = {TI.zone_kind(z) for z in TI.zone_ids(spec)} | {TI.zone_kind(TI.SEAT)}
    assert kinds == set(TI.ZONE_KINDS)


def test_zone_ids_follow_door_count_and_fill_order():
    cfg = _live_cfg()
    z20 = TI.zone_ids(TI.spec_for(cfg, "JR埼京線"))
    assert z20 == ("door0", "door1", "door2", "door3",
                   "front0", "front1", "front2", "front3", "aisle")
    z16 = TI.zone_ids(TI.spec_for(cfg, "東京メトロ銀座線"))
    assert z16 == ("door0", "door1", "door2",
                   "front0", "front1", "front2", "aisle")
    assert z20[-1] == TI.AISLE and z16[-1] == TI.AISLE      # 通路は最後


def test_zone_caps_split_the_standing_room_by_the_observed_shares():
    cfg = _live_cfg()
    spec = TI.spec_for(cfg, "JR埼京線")
    caps = TI.zone_caps(cfg, spec)
    stand = spec["capacity"] - spec["seats"]
    door = sum(v for k, v in caps.items() if k.startswith(TI.DOOR))
    front = sum(v for k, v in caps.items() if k.startswith(TI.FRONT))
    assert door / stand == pytest.approx(0.49, abs=1e-9)
    assert front / stand == pytest.approx(0.31, abs=1e-9)
    assert caps[TI.AISLE] / stand == pytest.approx(0.20, abs=1e-9)


def test_zone_adjacency_matches_the_car_layout():
    assert TI.zone_adjacent("door1", "door1")
    assert TI.zone_adjacent("door1", "front1")          # ドアの正面が座席前
    assert not TI.zone_adjacent("door0", "door3")       # 車両の反対端
    assert TI.zone_adjacent("door0", "aisle")           # 通路は立位の背骨
    assert TI.zone_adjacent(TI.SEAT, "front2")          # 座っている人の前に立つ人
    assert not TI.zone_adjacent(TI.SEAT, TI.AISLE)
    assert not TI.zone_adjacent(TI.SEAT, "door0")


def test_fill_order_is_door_then_front_and_aisle_is_last():
    """空いた車内では**位置は選べる**: ドア脇 → 座席前 → 通路の順に埋まる。"""
    cfg = _live_cfg()
    spec = TI.spec_for(cfg, "JR埼京線")
    caps, order = TI.zone_caps(cfg, spec), TI.zone_ids(spec)
    counts: dict = {}
    got = []
    for _ in range(int(sum(caps[z] for z in order if z != TI.AISLE))):
        z, _d = TI._place(order, caps, {}, counts, 0, 0.5)
        counts[(0, z)] = counts.get((0, z), 0) + 1
        got.append(z)
    assert got[0] == "door0"
    assert TI.AISLE not in got, "通路が最後より前に埋まった"
    kinds = [TI.zone_kind(z) for z in got]
    assert kinds.index(TI.FRONT) > max(i for i, k in enumerate(kinds)
                                       if k == TI.DOOR) - len(order)


def test_crush_load_places_by_share_lottery_not_by_the_biggest_zone():
    """★押し込み帯では**シェア抽選**(初版の『占有率が最小の区画』は通路へ吸い寄せる)。"""
    cfg = _live_cfg()
    spec = TI.spec_for(cfg, "JR埼京線")
    caps, order = TI.zone_caps(cfg, spec), TI.zone_ids(spec)
    virt = {z: caps[z] * 3.0 for z in order}       # 全区画が公称の 3 倍 = 押し込み帯
    got = collections.Counter(
        TI._place(order, caps, virt, {}, 0, u / 400.0)[0] for u in range(400))
    stand = sum(caps.values())
    door = sum(v for k, v in got.items() if k.startswith(TI.DOOR)) / 400.0
    front = sum(v for k, v in got.items() if k.startswith(TI.FRONT)) / 400.0
    assert door == pytest.approx(0.49, abs=0.02)
    assert front == pytest.approx(0.31, abs=0.02)
    assert got[TI.AISLE] / 400.0 == pytest.approx(0.20, abs=0.02)
    assert stand > 0


def test_congestion_scale_windows_and_car_load():
    cfg = _live_cfg()
    assert TI.scale_at(cfg, 8 * 60) == 1.00        # 朝ラッシュ = アンカーそのもの
    assert TI.scale_at(cfg, 13 * 60) == 0.70       # 日中 = ピークの 70%
    assert TI.scale_at(cfg, 18 * 60) == 0.95
    assert TI.scale_at(cfg, 3 * 60) == 0.45        # 深夜 = 既定
    spec = TI.spec_for(cfg, "JR埼京線")
    peak = TI.car_load(cfg, spec, 8 * 60)
    noon = TI.car_load(cfg, spec, 13 * 60)
    assert peak == round(160 * 1.67)               # 定員 × 混雑率(クリップしない)
    assert noon < peak and noon == round(160 * 1.67 * 0.70)


def test_seat_ratio_and_crowd_band():
    cfg = _live_cfg()
    spec = TI.spec_for(cfg, "JR埼京線")
    assert TI.seat_ratio(spec, 267) == pytest.approx(51 / 267)
    assert TI.seat_ratio(spec, 20) == 1.0          # 乗車人数 < 座席数 = 全員座れる
    assert TI.seat_ratio({"seats": 0}, 100) == 0.0
    assert TI.crowd_band(cfg, 0.5) == 0
    assert TI.crowd_band(cfg, 1.2) == 1
    assert TI.crowd_band(cfg, 1.67) == 2
    assert TI.crowd_band(cfg, 9.9) == len(TI.CROWD_JA) - 1   # 語を捏造しない


def test_memo_line_is_a_pure_function_with_no_digits():
    """記憶 1 行は (路線, 混雑段, 行動) の純関数。★数字・実験条件を 1 文字も含まない。"""
    line = TI.memo_line("A線", 2, TI.ACT_SMARTPHONE)
    assert line == TI.memo_line("A線", 2, TI.ACT_SMARTPHONE)   # 冪等
    for band in range(len(TI.CROWD_JA)):
        for act in TI.ACTS:
            text = TI.memo_line("A線", band, act)
            assert not any(ch.isdigit() for ch in text), text
            assert "%" not in text and "混雑率" not in text
    assert TI.memo_line("", 0, TI.ACT_READ).startswith(TI.LINE_FALLBACK)
    # テンプレート側にも定型語側にも数字が無い(2 段検査。rumors/traces と同型)
    assert not any(ch.isdigit() for ch in TI.MEMO_TEXT)
    for w in list(TI.CROWD_JA) + list(TI.ACT_JA.values()):
        assert not any(ch.isdigit() for ch in w), w


def test_doze_only_when_seated():
    """うたた寝は**着席時のみ**(立ったまま寝る、を作らない)。立位では 2 種へ再配分。"""
    cfg = _live_cfg()
    stand = collections.Counter(
        TI.activity_of(cfg, 42, i, 0, 480, False) for i in range(600))
    seat = collections.Counter(
        TI.activity_of(cfg, 42, i, 0, 480, True) for i in range(600))
    assert stand[TI.ACT_DOZE] == 0
    assert seat[TI.ACT_DOZE] > 0
    assert set(stand) <= {TI.ACT_SMARTPHONE, TI.ACT_READ}
    # スマホが支配的(70〜84% の帯)
    assert 0.60 < seat[TI.ACT_SMARTPHONE] / 600.0 < 0.90


def _fill(cfg, spec, n, seed=42):
    """n 人を順に車両へ割り当てた結果 (選択列, 車両別人数)。実装と同じ逐次手続き。"""
    counts: dict = {}
    picks = []
    target = TI.car_load(cfg, spec, 480)
    for aid in range(n):
        c = TI.choose_car(cfg, spec, seed, aid, counts, target)
        assert 0 <= c < spec["cars"], "編成両数の外の車両を選んだ"
        counts[c] = counts.get(c, 0) + 1
        picks.append(c)
    return picks, counts


def test_choose_car_stays_in_the_formation_and_is_deterministic():
    cfg = _live_cfg()
    spec = TI.spec_for(cfg, "東京メトロ銀座線")          # 6 両
    picks, counts = _fill(cfg, spec, 60)
    assert len(counts) == spec["cars"], "1 両に固まった(編成にばらけていない)"
    assert picks == _fill(cfg, spec, 60)[0]            # 同じ入力 → 同じ答え


def test_choose_car_balances_more_as_the_load_weight_rises():
    """★混雑項が効いていることの機械固定(重みを上げるほど偏りが縮む)。

    既定の重みが小さいのは意図どおり: 1 両の目標乗車人数は**背景乗客込みの数百人**
    なので、地図内の乗客が数人増えても混雑の効用はほとんど動かない(= 現実でも
    「先に来た数人がどの車両に居るか」で選好は覆らない)。
    """
    spec = TI.spec_for(_live_cfg(), "東京メトロ銀座線")

    def _spread(w):
        cfg = _live_cfg(**{"transit_interior.car_choice.load_weight": w})
        _p, counts = _fill(cfg, spec, 60)
        return max(counts.values()) - min(counts.values())

    assert _spread(0.0) > _spread(50.0)
    assert _spread(50.0) <= 2, "重みを上げても均されない = 混雑項が効いていない"


def test_snap_helper_is_shared_with_world_transit(tmp_path):
    """到着表のスナップは実装を **1 つだけ**持つ(world/transit の純関数)。"""
    p = tmp_path / "tt.json"
    p.write_text(json.dumps({"meta": {}, "lines": [
        {"name": "A線", "first": "07:00", "last": "08:00", "headway_min": 15}]}),
        encoding="utf-8")
    tr = WT.Transit(p)
    table = tr.arrivals_of_day()
    assert WT.snap_to_arrival_of_day(table, 421) == (435, "A線")
    assert WT.snap_to_arrival_of_day(table, 420) == (420, "A線")   # ちょうど一致
    assert WT.snap_to_arrival_of_day(table, 100) == (420, "A線")   # 始発前
    assert WT.snap_to_arrival_of_day(table, 1400) == (480, "A線")  # 終電後クランプ
    assert WT.snap_to_arrival_of_day([], 100) is None
    assert tr.arrival_at_or_after(421) == (435, "A線")
    before = json.dumps(tr.lines, ensure_ascii=False, sort_keys=True)
    tr.arrival_at_or_after(421)
    assert json.dumps(tr.lines, ensure_ascii=False, sort_keys=True) == before


def test_broken_conf_degrades_to_defaults():
    cfg = TI.build_cfg({"enabled": True, "ride_minutes": -5,
                        "zone_share": {"door": -1, "front": -1, "aisle": -1},
                        "activity": {"smartphone": 0, "doze": 0, "read": 0},
                        "memory": {"crowd_bands": ["x", 3.0, 1.0, 2.0, 0.5]},
                        "congestion_scale": {"windows": [[9, 8, 1.0], ["a", 2, 3]],
                                             "default": -2},
                        "conductor": {"patrol_interval_steps": 0,
                                      "dwell_load_weight": -1.0},
                        "lines": [{"match": "x", "cars": 0, "seats": 999,
                                   "capacity": 100}]})
    assert cfg["ride_minutes"] == 1
    assert cfg["zone_share"] == TI.DEFAULTS["zone_share"]     # 総和 0 → 既定へ
    assert cfg["activity"] == TI.DEFAULTS["activity"]
    assert cfg["memory"]["crowd_bands"] == [0.5, 1.0]         # 昇順・語数−1 で頭打ち
    assert cfg["congestion_scale"]["windows"] == []           # 壊れた窓は捨てる
    assert cfg["congestion_scale"]["default"] == 0.0
    assert cfg["conductor"]["patrol_interval_steps"] == 1
    assert cfg["conductor"]["dwell_load_weight"] == 0.0
    assert cfg["lines"][0]["cars"] == TI.DEFAULT_SPEC["cars"]  # 非正は既定へ
    assert cfg["lines"][0]["seats"] == 100                    # 座席 ≤ 定員


# =========================================================================== #
# (C) OFF = 現行と 1 バイトも変わらない(検収基準 ①)
# =========================================================================== #
def test_off_matches_golden(tmp_path):
    golden = json.load(open(GOLDEN, encoding="utf-8"))
    sim = _sim(tmp_path, "ti_golden", **_GOLDEN_NEUTRAL)
    sim.run()
    assert _l1(sim) == golden, "II-1 の seam がゴールデンを動かしている"


@_needs_roster
def test_off_matches_pure_default_on_the_inflow_roster(tmp_path):
    pure = _inflow(tmp_path, "ti_pure", on=False, n_steps=72, n_agents=60)
    pure.run()
    off = _inflow(tmp_path, "ti_off", on=False, n_steps=72, n_agents=60,
                  **{"transit_interior.enabled": "false"})
    off.run()
    assert _l1(pure) == _l1(off)
    assert not [e for e in pure.logger.events if e.kind in KINDS]


@_needs_roster
def test_off_grows_no_state_and_no_attributes(tmp_path):
    sim = _inflow(tmp_path, "ti_off_noop", on=False, n_steps=48, n_agents=60)
    sim.run()
    assert getattr(sim, "_train_state", None) is None
    assert TI.provenance(sim) is None
    for a in sim.agents:
        for name in ("train_car", "train_zone", "train_seated", "_train_ride"):
            assert not hasattr(a, name), f"OFF なのに {name} が生えている"


@_needs_roster
def test_enabled_requires_the_inflow_pulse(tmp_path):
    """★2 つの ON の論理積: パルスが OFF なら本層 ON でも 1 件も出ない。"""
    sim = _inflow(tmp_path, "ti_nopulse", on=True, pulse=False,
                  n_steps=48, n_agents=60)
    assert TI.enabled(sim) is False
    sim.run()
    assert not [e for e in sim.logger.events if e.kind in KINDS]
    for a in sim.agents:
        assert not hasattr(a, "train_car")
    prov = TI.provenance(sim)                      # ★黙って None にしない(設定ミスを隠さない)
    assert prov is not None and prov["pulse_on"] is False and prov["rides"] == 0


# =========================================================================== #
# (D) ON = 車両と区画が付く(検収基準 ② / ③)
# =========================================================================== #
@pytest.fixture(scope="module")
def on_run(tmp_path_factory):
    if not (ROSTER.exists() and V6.exists()):
        pytest.skip("流入名簿 or v6 地図が未生成")
    tmp = tmp_path_factory.mktemp("ti_on")
    sim = _inflow(tmp, "ti_on", on=True, n_steps=288, n_agents=200)
    sim.run()
    return sim


def test_on_produces_rides_within_the_formation(on_run):
    rides = _kind(on_run, "train_ride")
    assert rides, "ON なのに 1 乗車も起きていない"
    cfg = TI.cfg_of(on_run)
    for e in rides:
        spec = TI.spec_for(cfg, e.payload["line"])
        assert e.payload["cars"] == spec["cars"]
        assert 0 <= e.payload["car"] < spec["cars"], e.payload
        assert e.payload["capacity"] == spec["capacity"]
        assert e.payload["zone"] in TI.zone_ids(spec) or e.payload["zone"] == TI.SEAT
        assert e.payload["steps"] >= 1
        assert e.payload["act"] in TI.ACTS


def test_on_spreads_riders_across_the_formation(on_run):
    cars = collections.Counter(e.payload["car"] for e in _kind(on_run, "train_ride"))
    assert len(cars) >= 5, f"編成にばらけていない: {dict(cars)}"


def test_standing_zone_shares_match_the_observed_split(on_run):
    stand = [e.payload["zone"] for e in _kind(on_run, "train_ride")
             if not e.payload["seated"]]
    assert len(stand) >= 30
    got = collections.Counter(TI.zone_kind(z) for z in stand)
    frac = {k: got[k] / len(stand) for k in (TI.DOOR, TI.FRONT, TI.AISLE)}
    assert frac[TI.DOOR] > frac[TI.FRONT] > 0.0
    assert frac[TI.DOOR] == pytest.approx(0.49, abs=0.15), frac
    assert frac[TI.AISLE] == pytest.approx(0.20, abs=0.15), frac


def test_seated_riders_are_excluded_from_the_standing_density(on_run):
    """★着席者は立位密度に 1 人も数えない(座席は立位定員の外側の量)。"""
    rides = _kind(on_run, "train_ride")
    seated = [e for e in rides if e.payload["seated"]]
    standing = [e for e in rides if not e.payload["seated"]]
    assert seated and standing
    for e in seated:
        assert e.payload["zone"] == TI.SEAT
        assert e.payload["density"] == 0.0, "着席者に立位密度が付いた"
    for e in standing:
        assert e.payload["density"] > 0.0


def test_ginza_rides_use_the_16m_constants(tmp_path):
    """★16 m 3 扉の路線に乗ると、L1 も区画も 16 m の定数だけで出来ている。

    ラン任せにせず**その路線へ直接乗せて**確かめる(短ランでは誰も乗らないことがあり、
    そのとき skip すると『定数の流用が無い』という一番大事な性質が無検査になる)。
    """
    sim = _train_sim(tmp_path, "ti_ginza")
    crowd = sim.agents[:6]
    _board(sim, crowd, train_min=480, line="東京メトロ銀座線", return_at=2)
    TI.phase(sim, 0, 480)
    TI.phase(sim, 2, 500)
    rides = _kind(sim, "train_ride")
    assert len(rides) == len(crowd)
    ok = ("door0", "door1", "door2", "front0", "front1", "front2",
          TI.AISLE, TI.SEAT)
    for e in rides:
        assert e.payload["cars"] == 6 and e.payload["capacity"] == 104
        assert 0 <= e.payload["car"] < 6
        assert e.payload["zone"] in ok, e.payload["zone"]
        # 20 m 4 扉の区画(door3 / front3)は**存在しない**
        assert e.payload["zone"] not in ("door3", "front3")
    # 定員 104 × 混雑率 1.53 × 朝ピーク 1.00 = 159(20 m 車の 150/160 とは別の数)
    assert {e.payload["load"] for e in rides} == {round(104 * 1.53)}


def test_memory_line_is_written_at_most_once_per_ride(on_run):
    prov = TI.provenance(on_run)
    assert prov["mem_lines"] == prov["rides"], "1 乗車 1 行を超えて記憶が入った"
    # 実際に記憶へ 1 行入っている(降車後に読める = プロンプトの唯一の口)
    texts = [ep.text for a in on_run.agents
             for ep in (list(a.mem.buffer) + list(a.mem.episodes))
             if any(w in ep.text for w in TI.CROWD_JA)]
    assert texts
    for t in texts:
        assert not any(ch.isdigit() for ch in t), t


def test_attributes_vanish_after_the_ride(on_run):
    """降車で属性が**消える**(属性不在が『乗っていない』の表明)。"""
    riding = [a for a in on_run.agents if hasattr(a, "train_car")]
    for a in on_run.agents:
        assert hasattr(a, "train_car") == hasattr(a, "_train_ride")
        if hasattr(a, "train_car"):
            assert a.loc == "outside"              # 乗っている = まだ圏外
            assert a.train_zone and isinstance(a.train_seated, bool)
    assert len(riding) < len(on_run.agents)


def test_l1_volume_per_ride_is_bounded(on_run):
    """L1 の量: 1 乗車あたりの行数を測って上限の効きを確認する。"""
    prov = TI.provenance(on_run)
    rows = len([e for e in on_run.logger.events if e.kind in KINDS])
    per_ride = rows / max(1, prov["rides"])
    assert per_ride < 8.0, f"1 乗車あたり {per_ride:.2f} 行(多すぎる)"
    assert prov["copresence_dropped"] == 0 or prov["copresence"] > 0


# =========================================================================== #
# (E) 決定論(検収基準 ⑧)
# =========================================================================== #
@_needs_roster
def test_two_runs_with_the_same_seed_are_identical(tmp_path):
    a = _inflow(tmp_path, "ti_d1", on=True, n_steps=72, n_agents=100)
    a.run()
    b = _inflow(tmp_path, "ti_d2", on=True, n_steps=72, n_agents=100)
    b.run()
    assert _kind(a, "train_ride"), "テスト前提が崩れた(乗車ゼロ = 空同士の比較)"
    assert _l1(a) == _l1(b)
    assert [(x.id, getattr(x, "train_car", None), getattr(x, "train_zone", None))
            for x in a.agents] == \
           [(x.id, getattr(x, "train_car", None), getattr(x, "train_zone", None))
            for x in b.agents]


@_needs_roster
def test_resume_matches_straight(tmp_path):
    """★``engine/checkpoint.py`` に 1 行も足さずに resume == straight。

    根拠は「世界に効く状態を sim 側に 1 つも置かない」設計: 乗車 1 回ぶんも
    同席の重複判定も 1 日の記録予算も **agent の属性**にあり、agents は既存の
    checkpoint が丸ごと pickle する。sim 側の ``_train_state`` は観測タリーだけで
    L1/L2/L3 に現れないので、数え直しになっても出力は 1 バイトも動かない。
    """
    import pyarrow.parquet as pq
    from society.engine import checkpoint, scheduler

    split, total = 60, 120
    # ★snapshot_every / checkpoint_every は split 側と total 側で**同じ値**にする
    #   (config_hash に入るので食い違うと resume 自体が拒否される)
    fixed = {"observer.snapshot_every": total, "observer.checkpoint_every": split}
    straight_dir = tmp_path / "ti_straight"
    straight = _inflow(tmp_path, "ti_straight", on=True, n_steps=total,
                       n_agents=100, **fixed)
    straight.run()
    # ★checkpoint_every を立てると logger が定期 flush するので、前提の確認は
    #   in-memory バッファではなく parquet 側で行う
    rows_a = pq.read_table(straight_dir / "l1_events.parquet").to_pylist()
    assert [r for r in rows_a if r["kind"] == "train_ride"], \
        "テスト前提が崩れた(乗車ゼロ)"

    d = tmp_path / "ti_resumed"
    sim1 = _inflow(tmp_path, "ti_resumed", on=True, n_steps=split, n_agents=100,
                   **fixed)
    for step in range(split):
        scheduler.run_step(sim1, step)
    checkpoint.save(sim1, split, d / "checkpoint" / f"ckpt-{split:06d}.pkl.gz")
    sim1.logger.flush_segment()
    sim2 = _inflow(tmp_path, "ti_resumed", on=True, n_steps=total, n_agents=100,
                   **fixed)
    sim2.run(resume_from=d)
    for stem in ("l1_events", "l2_metrics", "l3_snapshots"):
        assert pq.read_table(straight_dir / f"{stem}.parquet").to_pylist() == \
            pq.read_table(d / f"{stem}.parquet").to_pylist(), f"{stem} 不一致"
    assert [(a.id, getattr(a, "train_car", None), getattr(a, "train_zone", None),
             sorted(getattr(a, "_train_seen", ()) or ())) for a in straight.agents] == \
           [(a.id, getattr(a, "train_car", None), getattr(a, "train_zone", None),
             sorted(getattr(a, "_train_seen", ()) or ())) for a in sim2.agents]


# =========================================================================== #
# (F) ★会話を 1 件も増やさない(検収基準 ⑤)
# =========================================================================== #
@_needs_roster
def test_no_extra_conversation_and_identical_llm_calls(tmp_path):
    """車内は静かな同席: ON/OFF で **generate 呼数も speak/hear 件数も完全一致**。

    プロンプト非依存スタブ(_FixedLLM)を使うのは、記憶が 1 行増えるだけで
    mock の応答が変わって世界が分岐し、呼数の比較にならないため(test_traces と同型)。
    """
    def _run(name, on):
        sim = _inflow(tmp_path, name, on=on, n_steps=72, n_agents=100)
        sim.llm = _FixedLLM(_TALKY)
        sim.run()
        c = collections.Counter(e.kind for e in sim.logger.events)
        return sim.llm.calls, c["speak"], c["hear"], c["llm_deliberate"]

    on, off = _run("ti_talk_on", True), _run("ti_talk_off", False)
    assert on[1] > 0, "テスト前提が崩れた(発話ゼロ = 空同士の比較)"
    assert on == off


@_needs_roster
def test_prompts_are_byte_identical_when_the_memory_line_is_off(tmp_path):
    """★プロンプトへの口は**記憶 1 行だけ**: それを切れば ON でもバイト一致。"""
    def _prompts(name, **ov):
        sim = _inflow(tmp_path, name, n_steps=72, n_agents=60, **ov)
        sim.llm = _FixedLLM(_TALKY)
        sim.run()
        return list(sim.llm.prompts)

    off = _prompts("ti_pr_off", on=False)
    on = _prompts("ti_pr_on", on=True,
                  **{"transit_interior.memory.enabled": "false"})
    assert off and off == on


# =========================================================================== #
# (G) 同席: 重複排除 + 上限(検収基準 ⑥)
# =========================================================================== #
def _train_sim(tmp_path, name, **ov):
    """車内を**直接組む**単体用の sim(実ランを待たずに同席・巡回を検査する)。"""
    sim = _sim(tmp_path, name, n_steps=1, n_agents=12,
               **{"transit_interior.enabled": "true",
                  "world.inflow_pulse.enabled": "true", **ov})
    return sim


def _board(sim, agents, train_min=480, line="A線", return_at=2):
    """agents を「この step に車内に居る」状態へ置く(実ランの窓と同じ条件)。"""
    station = sim.city.station_node
    for a in agents:
        a.loc, a.sleeping = "outside", False
        a.return_gateway = station
        a.return_at = int(return_at)
        a.pulse_train_min, a.pulse_line = int(train_min), line


def test_copresence_is_deduped_per_pair_per_day(tmp_path):
    sim = _train_sim(tmp_path, "ti_cop")
    crowd = sim.agents[:4]
    _board(sim, crowd, return_at=2)
    # 全員を同じ車両へ寄せるため 1 両編成の路線にする
    TI.cfg_of(sim)["lines"] = [{"match": "A線", "cars": 1, "car_len_m": 20.0,
                                "capacity": 150, "seats": 51, "doors": 4,
                                "congestion": 1.5}]
    TI.phase(sim, 0, 480)
    TI.phase(sim, 1, 490)
    assert all(a.train_car == 0 for a in crowd)
    TI.phase(sim, 2, 500)                          # return_at 到達 → 全員降車
    rows = _kind(sim, "train_copresence")
    pairs = {(e.agent_id, e.payload["other_id"]) for e in rows}
    assert len(rows) == len(pairs) == 6, rows      # 4 人 = 6 対、重複なし
    for e in rows:
        assert e.agent_id < e.payload["other_id"]  # 小さい方が主語
        assert e.payload["crew"] is False
    # ---- 同じ日にもう一度乗っても対は増えない(1 日 1 対 1 件)----
    n = len(rows)
    _board(sim, crowd, train_min=1020, return_at=5)
    TI.phase(sim, 3, 1010)
    TI.phase(sim, 5, 1030)
    assert len(_kind(sim, "train_copresence")) == n


def test_copresence_respects_the_per_agent_daily_budget(tmp_path):
    """★上限は**1 日 1 人あたり**(= 個体側の予算)。総量は n_agents × 予算で抑える。

    グローバルな日次カウンタにしないのは resume のため(sim 側の日次カウンタは
    checkpoint に載せられず、上限が binding なランで resume ≠ straight になる)。
    """
    sim = _train_sim(tmp_path, "ti_cap",
                     **{"transit_interior.copresence.max_pairs_per_day": 1})
    crowd = sim.agents[:4]
    _board(sim, crowd, return_at=2)
    TI.cfg_of(sim)["lines"] = [{"match": "A線", "cars": 1, "car_len_m": 20.0,
                                "capacity": 150, "seats": 51, "doors": 4,
                                "congestion": 1.5}]
    TI.phase(sim, 0, 480)
    TI.phase(sim, 1, 490)
    TI.phase(sim, 2, 500)
    rows = _kind(sim, "train_copresence")
    prov = TI.provenance(sim)
    assert prov["copresence"] == len(rows)
    assert prov["copresence"] + prov["copresence_dropped"] == 6      # 4 人 = 6 対
    # 予算 1 = 誰も 2 件以上の対に載らない
    n = collections.Counter()
    for e in rows:
        n[e.agent_id] += 1
        n[e.payload["other_id"]] += 1
    assert rows and max(n.values()) == 1
    # 予算は個体側(agents pickle 同梱)に積まれている = resume 安全の機械的根拠
    assert all(len(a._train_seen) <= 3 for a in crowd)


def test_pairs_per_car_cap_bounds_the_roster(tmp_path):
    sim = _train_sim(tmp_path, "ti_pc",
                     **{"transit_interior.copresence.max_pairs_per_car": 1})
    crowd = sim.agents[:4]
    _board(sim, crowd, return_at=2)
    TI.cfg_of(sim)["lines"] = [{"match": "A線", "cars": 1, "car_len_m": 20.0,
                                "capacity": 150, "seats": 51, "doors": 4,
                                "congestion": 1.5}]
    TI.phase(sim, 0, 480)
    TI.phase(sim, 1, 490)
    for a in crowd:
        assert len(a._train_ride["mates"]) <= 1
    TI.phase(sim, 2, 500)
    assert len(_kind(sim, "train_copresence")) <= 4


def test_adjacent_only_filters_by_zone(tmp_path):
    sim = _train_sim(tmp_path, "ti_adj",
                     **{"transit_interior.copresence.adjacent_only": "true"})
    crowd = sim.agents[:4]
    _board(sim, crowd, return_at=2)
    TI.cfg_of(sim)["lines"] = [{"match": "A線", "cars": 1, "car_len_m": 20.0,
                                "capacity": 150, "seats": 51, "doors": 4,
                                "congestion": 1.5}]
    TI.phase(sim, 0, 480)
    TI.phase(sim, 2, 500)
    for e in _kind(sim, "train_copresence"):
        assert e.payload["zone_adjacent"] is True


# =========================================================================== #
# (H) 車掌(検収基準 ⑦)
# =========================================================================== #
def _make_crew(sim, agent, sim_min=480):
    station = sim.city.station_node
    agent.occupation = "車掌"
    agent.node, agent.loc, agent.sleeping = station, "street", False
    agent.work_node = station
    agent.work_start_min, agent.work_end_min = 0, 1439
    agent.sick = False
    return agent


def test_on_duty_crews_is_sorted_and_agrees_with_the_singular_version(tmp_path):
    sim = _sim(tmp_path, "ti_crew", n_steps=1, n_agents=12,
               **{"transit_staff.enabled": "true"})
    a, b = _make_crew(sim, sim.agents[3]), _make_crew(sim, sim.agents[1])
    crews = TS.on_duty_crews(sim, 480)
    assert [c.id for c in crews] == sorted([a.id, b.id])
    assert TS.on_duty_crew(sim, 480).id == crews[0].id


def test_conductor_patrol_is_deterministic(tmp_path):
    def _run(name):
        sim = _train_sim(tmp_path, name, **{"transit_staff.enabled": "true",
                                            "transit_interior.ride_minutes": 40})
        _make_crew(sim, sim.agents[0])
        crowd = sim.agents[1:5]
        _board(sim, crowd, return_at=4)
        TI.cfg_of(sim)["lines"] = [{"match": "A線", "cars": 3, "car_len_m": 20.0,
                                    "capacity": 150, "seats": 51, "doors": 4,
                                    "congestion": 1.5}]
        for s in range(4):
            TI.phase(sim, s, 480 + 10 * s)
        return [(e.step, e.agent_id, json.dumps(e.payload, sort_keys=True,
                                                ensure_ascii=False))
                for e in _kind(sim, "train_patrol")]

    first = _run("ti_pat1")
    assert first, "車掌の巡回が 1 件も出ていない"
    assert first == _run("ti_pat2")
    # 車両 index は step // interval % 両数 = 0,1,2,0 と進む
    assert [json.loads(p)["car"] for _s, _a, p in first] == [0, 1, 2, 0]
    assert all(json.loads(p)["cars"] == 3 for _s, _a, p in first)


def test_conductor_copresence_is_marked_as_crew(tmp_path):
    sim = _train_sim(tmp_path, "ti_crewcop", **{"transit_staff.enabled": "true"})
    _make_crew(sim, sim.agents[0])
    _board(sim, sim.agents[1:5], return_at=2)
    TI.cfg_of(sim)["lines"] = [{"match": "A線", "cars": 1, "car_len_m": 20.0,
                                "capacity": 150, "seats": 0, "doors": 4,
                                "congestion": 1.5}]        # 座席ゼロ = 全員立位
    TI.phase(sim, 0, 480)
    crew_rows = [e for e in _kind(sim, "train_copresence") if e.payload["crew"]]
    assert len(crew_rows) == 4                     # 車掌 × 同じ車両の 4 人
    # 通路は立位の背骨 = どの立位区画とも隣接(座っている客なら False になるのが正しい)
    assert all(e.payload["zone_adjacent"] for e in crew_rows)


def test_patrol_needs_transit_staff_to_be_on(tmp_path):
    sim = _train_sim(tmp_path, "ti_nostaff")       # transit_staff は既定 OFF
    _make_crew(sim, sim.agents[0])
    _board(sim, sim.agents[1:5], return_at=2)
    TI.phase(sim, 0, 480)
    assert not _kind(sim, "train_patrol")


def test_dwell_load_is_observation_only_by_default(tmp_path):
    """★車内負荷は既定で停車時間へ**足さない**(ホーム密度との二重計上の防止)。"""
    sim = _train_sim(tmp_path, "ti_dwell", **{"transit_staff.enabled": "true"})
    _board(sim, sim.agents[1:5], return_at=2)
    TI.phase(sim, 0, 480)
    assert TI.dwell_extra_load(sim) == 0
    pay = TI.dwell_payload(sim)
    assert set(pay) == {"interior_standing", "interior_cars", "interior_weight"}
    assert pay["interior_weight"] == 0.0 and pay["interior_cars"] >= 1
    # 重みを上げると足される(実験用のレバーとして生きている)
    TI.cfg_of(sim)["conductor"]["dwell_load_weight"] = 2.0
    assert TI.dwell_extra_load(sim) == 2 * pay["interior_standing"]


def test_dwell_seam_is_a_noop_when_the_interior_is_off(tmp_path):
    sim = _sim(tmp_path, "ti_dwell_off", n_steps=1, n_agents=10)
    assert TI.dwell_extra_load(sim) == 0 and TI.dwell_payload(sim) == {}


# =========================================================================== #
# (I) 疲労(局所密度)
# =========================================================================== #
def test_local_density_adds_a_small_capped_fatigue(tmp_path):
    sim = _train_sim(tmp_path, "ti_fat")
    a = sim.agents[0]
    _board(sim, [a], return_at=6)
    TI.cfg_of(sim)["lines"] = [{"match": "A線", "cars": 1, "car_len_m": 20.0,
                                "capacity": 150, "seats": 0, "doors": 4,
                                "congestion": 3.0}]        # 座席ゼロ = 必ず立つ
    a.fatigue = 0.0
    for s in range(6):
        TI.phase(sim, s, 480 + 10 * s)
    assert 0.0 < a.fatigue <= TI.DEFAULTS["comfort"]["fatigue_max"] + 1e-9
    # 上限が効く(乗り続けても max を超えない)
    assert a.fatigue == pytest.approx(
        min(TI.DEFAULTS["comfort"]["fatigue_max"], a.fatigue))


def test_seated_riders_take_no_crowding_fatigue(tmp_path):
    sim = _train_sim(tmp_path, "ti_fat_seat")
    a = sim.agents[0]
    _board(sim, [a], return_at=4)
    TI.cfg_of(sim)["lines"] = [{"match": "A線", "cars": 1, "car_len_m": 20.0,
                                "capacity": 150, "seats": 150, "doors": 4,
                                "congestion": 0.2}]        # 全員着席
    a.fatigue = 0.0
    for s in range(4):
        TI.phase(sim, s, 480 + 10 * s)
    assert a.train_seated is True
    assert a.fatigue == 0.0


# =========================================================================== #
# (J) 名簿生成器: **車掌**(死んでいた conf キーの修復。検収基準 ⑩)
# =========================================================================== #
def test_persona_pool_generator_declares_conductors():
    """conf の crew_occupations が指す「車掌」が名簿生成器に実在する。"""
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        import build_persona_pool as B
    finally:
        sys.path.pop(0)
    roles = {r[1]: r for r in B._L5_ROLES}
    assert "車掌" in roles, "conf が指す職業『車掌』が名簿に無い(死んだ設定キー)"
    role, occ, n, posts, duty, visitor = roles["車掌"]
    assert role == occ == "車掌"
    assert n == 40 and posts and visitor is True
    assert duty["rotates"] is True and duty["shift_hours"] == 8
    # ★**追記**であって挿入ではない(途中に挿すと既存 L5 の id が全部ずれる)。
    #   ★「最後の要素であること」では固定しない: 以後の波も同じ規約で**さらに末尾へ**
    #     足すので、それを禁じてしまうと規約と矛盾する。固定すべきは
    #     「車掌が、車掌より前に居た全役割よりも後ろに居る」= 挿入されていないこと。
    names = [r[1] for r in B._L5_ROLES]
    earlier = ["駅員", "電車運転士", "バス運転士", "タクシー運転手", "警察官", "配信者"]
    assert names.index("車掌") > max(names.index(x) for x in earlier), \
        "車掌が既存役割の途中に挿入された(既存 L5 の id がずれる)"
    # 乗務員 2 職の両方が名簿に実在し、運転士 ≥ 車掌(ワンマン系統の分)
    assert roles["電車運転士"][2] >= n
    # conf 側の既定と綴りが一致している(綴り違いで永久に 0 人になるのを防ぐ)
    assert set(TS.DEFAULTS["bind"]["crew_occupations"]) <= set(roles)


# =========================================================================== #
# (K) 同席の日次上限は**ログの量だけ**を動かす(第114 レーン 1c)
# =========================================================================== #
# 本選 conf(conf/finals_observe.yaml)で max_pairs_per_day を 8 → 24 へ引き上げた。
# 引き上げてよい機械的な根拠は「この上限が動力学に触れない」ことなので、それを
# **同 seed の 2 ラン(8 と 24)**で固定する。実装上の根拠は 3 つ:
#   ① `_copresence` は Event を 1 件出すか出さないかを分けるだけで、乱数を 1 粒も引かない
#   ② 重複判定の印(`_train_seen`)は**出しても捨てても同じように**付く(予算切れでも付く)
#   ③ 他の誰の state も触らない(疲労・記憶・位置はこの関数の外)
@_needs_roster
def test_daily_cap_only_changes_the_copresence_rows(tmp_path):
    """上限 8 と 24 で、同席以外の L1 行集合と乱数 stream の要求列が完全一致する。"""
    def run(name, cap):
        sim = _inflow(tmp_path, name, on=True, n_steps=288, n_agents=200,
                      **{"transit_interior.copresence.max_pairs_per_day": cap})
        seen: list[tuple] = []
        original = sim.hub.stream

        def spy(*key):
            seen.append(tuple(str(k) for k in key))
            return original(*key)

        sim.hub.stream = spy
        sim.run()
        return sim, seen

    a, keys_a = run("ti_cap8", 8)
    b, keys_b = run("ti_cap24", 24)
    # ① 乱数の要求列(用途・キー・順序)が完全一致 = 乱数消費が 1 バイトも動かない
    assert keys_a == keys_b, "上限の変更で乱数 stream の要求列が変わった"
    # ② 同席以外の L1 行が完全一致
    other_a = [r for r in _l1(a) if r[2] != "train_copresence"]
    other_b = [r for r in _l1(b) if r[2] != "train_copresence"]
    assert other_a == other_b, "上限の変更で同席以外の L1 が変わった"
    # ③ 対の総数(記録 + 打ち切り)は保存する = 動くのは「載せた割合」だけ
    pa, pb = TI.provenance(a), TI.provenance(b)
    assert pa["copresence"] + pa["copresence_dropped"] == \
        pb["copresence"] + pb["copresence_dropped"]

    # ④ 上限を上げても**対の集合は増えるだけ**(8 で載った対は 24 でも必ず載る)
    def pairs(sim):
        return {(e.sim_min // 1440, e.agent_id, e.payload["other_id"])
                for e in _kind(sim, "train_copresence")}
    assert pairs(a) <= pairs(b)


@_needs_roster
def test_daily_cap_is_binding_and_conserves_the_pair_total(tmp_path):
    """上限が実際に効く密度(1 両 12 人 = 1 人あたり 11 対)で 8 と 24 を比べる。

    ★上のテストの実ランは 288 step / 200 体では 1 人 8 対に届かない(上限が
      binding でない)。ここは「上限が効いている状態で何が起きるか」を分けて固定する:
      予算 8 では打ち切りが出て、24 では 1 対も落ちず、**対の総数は両者で等しい**。
    """
    def run(name, cap):
        sim = _train_sim(tmp_path, name,
                         **{"transit_interior.copresence.max_pairs_per_day": cap})
        crowd = sim.agents[:12]                    # 12 人 = 66 対・1 人あたり 11 対
        _board(sim, crowd, return_at=2)
        TI.cfg_of(sim)["lines"] = [{"match": "A線", "cars": 1, "car_len_m": 20.0,
                                    "capacity": 150, "seats": 51, "doors": 4,
                                    "congestion": 1.5}]
        TI.phase(sim, 0, 480)
        TI.phase(sim, 1, 490)
        TI.phase(sim, 2, 500)                      # return_at 到達 → 全員降車
        return sim

    a, b = run("ti_bind8", 8), run("ti_bind24", 24)
    pa, pb = TI.provenance(a), TI.provenance(b)
    assert pa["copresence_dropped"] > 0, "上限 8 が binding でない(素材が足りない)"
    assert pb["copresence_dropped"] == 0, "1 人 11 対なら上限 24 では 1 対も落ちない"
    assert pb["copresence"] > pa["copresence"], "24 に上げても記録が増えていない"
    # 対の総数(= 世界で実際に起きた同席)は上限に依らない = 上限はログの量だけを切る
    assert pa["copresence"] + pa["copresence_dropped"] == \
        pb["copresence"] + pb["copresence_dropped"] == 66
    # 予算 8 でも「今日この相手は見た」の印は全対に付く(= 打ち切り数が試行回数に化けない)
    assert all(len(x._train_seen) == 11 for x in a.agents[:12])


def test_finals_conf_raises_the_daily_cap(tmp_path):
    """本選 conf が上限 24 を宣言している(第114 レーン 1c の決定を conf で固定)。"""
    from omegaconf import OmegaConf
    fin = OmegaConf.load(REPO / "conf" / "finals_observe.yaml")
    assert int(fin.transit_interior.copresence.max_pairs_per_day) == 24
