"""第149 レーン1: 声の段階(作用点0)+ 最寄り K cap(作用点A/B)+ 列挙のベクトル化。

正典: docs/plans/hearer-cap-plan.md §2(作用点 0/A/B)/ §4(値表)/ §5(テスト・検収)。

守るもの(検収基準の順)
  (1) **既定 OFF = 現行と完全一致**: 集合・順序・**オブジェクト同一性**まで一致し、
      索引経路も legacy 経路も従来コードがそのまま走る(cap=0 かつ radius_eff=None)。
  (2) 声の段階: 屋内/屋外 × 3 段の実効半径・perception_radius_m へのクランプ・
      発話文脈 → 段階の決定論写像(重症度 / イベント主催 / 路上の生業の持ち場)。
  (3) 最寄り K: 期待値ピン・同距離の tie は id 昇順・返りは id 昇順。
  (4) speak 経路: **列挙段 cap の結果 == _attention_limited(フル列挙後の選抜)の結果**
      (同一規則の機械証明。二重適用しても冪等)。
  (5) ベクトル化 == ループ(遮蔽器を据えるとループ経路へ後退する)。
  (6) 決定論(同 seed 2 ラン一致)・契約列挙ピン(conf 新キー・既定値・finals 値・registry)。
検証は mock のみ(実 LLM 禁止・≤24step)。乱数は 1 本も新設しない。
"""
from __future__ import annotations

import json
from pathlib import Path

from omegaconf import OmegaConf

from society import conversation as CONV
from society import registry as R
from society.config import load_config
from society.engine import scheduler
from society.engine.simulation import Simulation
from society.observer.schema import EVENT_KINDS
from society.world import perception as P

_REPO_FINALS = Path(__file__).resolve().parents[1] / "conf" / "finals_observe.yaml"


# --------------------------------------------------------------------------- #
# 共通ヘルパ(hearers_of が要求する duck-type だけを持つ最小の個体)
# --------------------------------------------------------------------------- #
class _A:
    __slots__ = ("id", "x", "y", "loc", "building", "floor", "sleeping")

    def __init__(self, i, x, y, *, building=None, floor=0, loc="street",
                 sleeping=False):
        self.id = int(i)
        self.x = float(x)
        self.y = float(y)
        self.loc = loc
        self.building = building
        self.floor = floor
        self.sleeping = sleeping

    def __repr__(self):                              # 失敗時の可読性
        return f"<A{self.id}@({self.x},{self.y})>"


class _NeverBlocks:
    """遮蔽器の duck-type。**一度も遮らない**が「据わっている」のでループ経路へ落ちる。"""

    def __init__(self):
        self.calls = 0

    def blocks(self, a, b) -> bool:
        self.calls += 1
        return False


class _BlocksOdd:
    """id が奇数の相手だけ遮る(遮蔽あり + cap の合成を見る)。"""

    def blocks(self, a, b) -> bool:
        return bool(b.id % 2)


def _grid(n_side: int = 12, pitch: float = 3.0) -> list:
    """格子状に並べた個体(id は 0..n²-1・座標は決定論)。"""
    out = []
    i = 0
    for gx in range(n_side):
        for gy in range(n_side):
            out.append(_A(i, gx * pitch, gy * pitch))
            i += 1
    return out


def _ids(agents) -> list:
    return [a.id for a in agents]


def _sim(tmp_path, name, n=24, steps=1, seed=42, **ov):
    dot = [f"run.seed={seed}", f"run.n_agents={n}", f"run.n_steps={steps}",
           f"run.name={name}", "model.backend=mock"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


# --------------------------------------------------------------------------- #
# (1) 既定 OFF = 現行と完全一致(集合・順序・オブジェクト同一性)
# --------------------------------------------------------------------------- #
def test_default_index_path_is_byte_identical():
    """cap=0 かつ radius_eff=None は従来コードがそのまま走る(同一オブジェクト列)。"""
    agents = _grid()
    idx = P.build_index(agents, 40.0)
    for speaker in (agents[0], agents[57], agents[-1]):
        base = P.hearers_of(speaker, idx, 40.0)
        same = P.hearers_of(speaker, idx, 40.0, cap=0, radius_eff=None)
        assert base == same                      # list の要素は同一オブジェクト
        assert all(a is b for a, b in zip(base, same))
        assert _ids(base) == sorted(_ids(base))  # 返りは id 昇順(下流契約)


def test_default_legacy_path_is_byte_identical():
    """legacy(全対全 live 走査)経路も既定では 1 バイトも変わらない。"""
    agents = _grid()
    speaker = agents[57]
    base = P.hearers_of(speaker, agents, 40.0)
    same = P.hearers_of(speaker, agents, 40.0, cap=0, radius_eff=None)
    assert base == same and all(a is b for a, b in zip(base, same))


def test_index_and_legacy_agree_under_cap():
    """索引経路と legacy 経路は cap を掛けても同じ集合・同じ順序を返す。"""
    agents = _grid()
    idx = P.build_index(agents, 40.0)
    for speaker in (agents[0], agents[57], agents[100]):
        for cap in (1, 3, 7, 15, 10_000):
            a = P.hearers_of(speaker, idx, 40.0, cap=cap)
            b = P.hearers_of(speaker, agents, 40.0, cap=cap)
            assert _ids(a) == _ids(b), (speaker.id, cap)


def test_cap_larger_than_candidates_is_the_same_set():
    """K ≥ 候補数なら絞らないときと**同一集合・同一順序**。"""
    agents = _grid()
    idx = P.build_index(agents, 40.0)
    speaker = agents[57]
    base = P.hearers_of(speaker, idx, 40.0)
    big = P.hearers_of(speaker, idx, 40.0, cap=len(agents) + 1)
    assert _ids(base) == _ids(big)
    assert all(a is b for a, b in zip(base, big))


# --------------------------------------------------------------------------- #
# (2) 最寄り K の期待値ピン(距離二乗 昇順 → id 昇順 / 返りは id 昇順)
# --------------------------------------------------------------------------- #
def test_nearest_k_expected_values():
    """手組み座標で最寄り K を固定する。"""
    speaker = _A(0, 0.0, 0.0)
    others = [_A(1, 1.0, 0.0), _A(2, 2.0, 0.0), _A(3, 3.0, 0.0),
              _A(4, 30.0, 0.0)]
    idx = P.build_index([speaker] + others, 40.0)
    assert _ids(P.hearers_of(speaker, idx, 40.0, cap=2)) == [1, 2]
    assert _ids(P.hearers_of(speaker, idx, 40.0, cap=3)) == [1, 2, 3]
    assert _ids(P.hearers_of(speaker, idx, 40.0, cap=1)) == [1]


def test_distance_ties_break_by_id_ascending():
    """同距離は **id 昇順**で選ぶ(乱数ゼロ)。返り値も id 昇順。"""
    speaker = _A(100, 0.0, 0.0)
    # 4 人とも距離 5.0(同値)。cap=2 なら id の小さい 2 人。
    others = [_A(7, 5.0, 0.0), _A(3, 0.0, 5.0), _A(9, -5.0, 0.0),
              _A(5, 0.0, -5.0)]
    idx = P.build_index([speaker] + others, 40.0)
    assert _ids(P.hearers_of(speaker, idx, 40.0, cap=2)) == [3, 5]
    assert _ids(P.hearers_of(speaker, idx, 40.0, cap=3)) == [3, 5, 7]
    # legacy 経路も同じ規則
    assert _ids(P.hearers_of(speaker, [speaker] + others, 40.0, cap=2)) == [3, 5]


def test_nearest_k_pure_function_matches_the_s15_rule():
    """`perception.nearest_k` は S15 `_attention_limited` と同一規則(純関数の照合)。"""
    speaker = _A(0, 0.0, 0.0)
    cands = [_A(i, float(i % 7), float(i % 5)) for i in range(1, 40)]
    for cap in (1, 2, 5, 13, 39, 100):
        got = P.nearest_k(speaker, list(cands), cap)
        # S15 の実装(フル列挙後の選抜)と突き合わせる
        class _S:                                   # _attention_limited は sim.cfg しか見ない
            pass
        sim = _S()
        sim._attn_hearers_max = cap
        want = scheduler._attention_limited(sim, speaker, sorted(cands,
                                                                key=lambda a: a.id))
        assert _ids(got) == _ids(want), cap


# --------------------------------------------------------------------------- #
# (3) ベクトル化 == ループ(遮蔽器が据わるとループ経路へ後退する)
# --------------------------------------------------------------------------- #
def test_vectorized_equals_loop_fallback():
    """遮らない遮蔽器を据えるとループ経路になり、ベクトル化と同じ答えを返す。"""
    agents = _grid()
    speaker = agents[57]
    vec = P.hearers_of(speaker, P.build_index(agents, 40.0), 40.0, cap=9)
    occ = _NeverBlocks()
    loop = P.hearers_of(speaker, P.build_index(agents, 40.0), 40.0,
                        occluder=occ, cap=9)
    assert occ.calls > 0, "遮蔽器がループ経路で使われていない"
    assert _ids(vec) == _ids(loop)
    # radius_eff だけのときも一致する
    vec_r = P.hearers_of(speaker, P.build_index(agents, 40.0), 40.0, radius_eff=7.0)
    loop_r = P.hearers_of(speaker, P.build_index(agents, 40.0), 40.0,
                          occluder=_NeverBlocks(), radius_eff=7.0)
    assert _ids(vec_r) == _ids(loop_r)


def test_occluder_filters_before_the_cap():
    """遮蔽で外れた相手は cap の母集団に入らない(遮蔽除外後の最寄り K)。"""
    agents = _grid(6, 2.0)
    speaker = agents[14]
    got = P.hearers_of(speaker, P.build_index(agents, 40.0), 40.0,
                       occluder=_BlocksOdd(), cap=4)
    assert all(a.id % 2 == 0 for a in got), _ids(got)
    assert len(got) == 4


def test_vectorized_equals_loop_at_scale():
    """数千体の高密度セルでもベクトル化とループが**完全一致**する(選抜の機械証明)。"""
    agents = []
    for i in range(3000):                            # 決定論の擬似ランダム配置
        agents.append(_A(i, (i * 37) % 79 * 0.5, (i * 53) % 83 * 0.5))
    speaker = agents[1234]
    for cap in (0, 5, 20, 200):
        vec = P.hearers_of(speaker, P.build_index(agents, 40.0), 40.0, cap=cap,
                           radius_eff=20.0)
        loop = P.hearers_of(speaker, P.build_index(agents, 40.0), 40.0,
                            occluder=_NeverBlocks(), cap=cap, radius_eff=20.0)
        assert _ids(vec) == _ids(loop), cap


# --------------------------------------------------------------------------- #
# (4) 声の段階: 物理表(段階 × 屋内外 → 半径)とクランプ
# --------------------------------------------------------------------------- #
def test_speech_cfg_defaults_are_off_and_table_matches_the_plan():
    cfg = P.build_speech_cfg(None, 40.0)
    assert cfg["enabled"] is False
    assert cfg["normal"] == {"outdoor_m": 5.0, "indoor_m": 10.0}
    assert cfg["raised"] == {"outdoor_m": 12.0, "indoor_m": 20.0}
    assert cfg["shout"] == {"outdoor_m": 30.0, "indoor_m": 40.0}


def test_speech_cfg_clamps_to_perception_radius():
    """perception_radius_m を超える値は**クランプ**される(9 セル走査の不変条件)。"""
    cfg = P.build_speech_cfg({"enabled": True,
                              "shout": {"outdoor_m": 500.0, "indoor_m": 999.0},
                              "normal": {"outdoor_m": -3.0, "indoor_m": 10.0}},
                             40.0)
    assert cfg["shout"] == {"outdoor_m": 40.0, "indoor_m": 40.0}
    assert cfg["normal"]["outdoor_m"] == 0.0        # 負値は 0 で床止め
    assert cfg["enabled"] is True


def test_speech_radius_indoor_outdoor_by_level():
    """屋内(建物内)は同じ段階でも遠くまで届く(騒音が低いため)。"""
    cfg = P.build_speech_cfg({"enabled": True}, 40.0)
    out = _A(1, 0.0, 0.0)
    ind = _A(2, 0.0, 0.0, building="b1", floor=3)
    table = {"normal": (5.0, 10.0), "raised": (12.0, 20.0), "shout": (30.0, 40.0)}
    for level, (o, i) in table.items():
        assert P.speech_radius(cfg, level, out) == o
        assert P.speech_radius(cfg, level, ind) == i
    # 未知の段階は保守側(normal)へ倒す
    assert P.speech_radius(cfg, "whisper", out) == 5.0


def test_speech_radius_actually_shrinks_the_hearer_set():
    """半径 5m では 40m の聴衆が激減する(物理層の効き方)。"""
    agents = _grid(12, 3.0)
    speaker = agents[57]
    idx = P.build_index(agents, 40.0)
    wide = P.hearers_of(speaker, idx, 40.0)
    near = P.hearers_of(speaker, idx, 40.0, radius_eff=5.0)
    assert set(_ids(near)) < set(_ids(wide))
    assert len(near) < len(wide) / 4                 # 面積比でおおよそ 1/64 帯
    for a in near:
        assert (a.x - speaker.x) ** 2 + (a.y - speaker.y) ** 2 <= 25.0 + 1e-9


# --------------------------------------------------------------------------- #
# (5) 発話文脈 → 段階の決定論写像(LLM の自由文を読まない)
# --------------------------------------------------------------------------- #
def test_speech_level_mapping_is_deterministic(tmp_path):
    sim = _sim(tmp_path, "lvl", n=8)
    a = sim.agents[0]
    a.occupation = "会社員"
    a.severity = 0
    assert scheduler._speech_level(sim, a, 0) == "normal"

    # ① 重症(severe 以上)= 悲鳴・助けを求める声
    a.severity = 3
    assert scheduler._speech_level(sim, a, 0) == "shout"
    a.severity = 0

    # ② 開催中イベントの主催者が会場ノードに立っている = 集会・演説
    sim.tools.events[0] = {"id": 0, "host": a.id, "node": a.node,
                           "started": True, "ended": False, "attendees": set(),
                           "title": "t"}
    assert scheduler._speech_level(sim, a, 1) == "raised"
    sim.tools.events[0]["ended"] = True              # 終了したら普通の声へ戻る
    assert scheduler._speech_level(sim, a, 2) == "normal"
    sim.tools.events.pop(0)

    # ③ 声を張る路上の生業が**自分の持ち場に立っている**
    from society import street_life as SL
    for occ in (SL.SPEECH, SL.MUSICIAN, SL.FUNDRAISER, SL.KITCHEN):
        a.occupation = occ
        a.street_post = a.node
        assert scheduler._speech_level(sim, a, 3) == "raised", occ
        a.street_post = "somewhere-else"             # 持ち場を離れたら通常の声
        assert scheduler._speech_level(sim, a, 3) == "normal", occ
    # 一対一の近接発話は張り上げない(保守側)
    for occ in (SL.TISSUE, SL.FORTUNE, SL.OUTREACH, SL.ROUGH):
        a.occupation = occ
        a.street_post = a.node
        assert scheduler._speech_level(sim, a, 3) == "normal", occ


def test_speak_bounds_are_inert_by_default(tmp_path):
    """既定 conf では (cap, radius_eff) = (0, None) = 現行経路を通る。"""
    sim = _sim(tmp_path, "bounds_off", n=8)
    assert scheduler._speak_bounds(sim, sim.agents[0], 0) == (0, None)


def test_speak_bounds_wire_both_knobs(tmp_path):
    sim = _sim(tmp_path, "bounds_on", n=8,
               **{"world.speech_levels.enabled": "true",
                  "world.attention_hearers_max": 20})
    a = sim.agents[0]
    a.occupation = "会社員"
    a.severity = 0
    a.loc, a.building = "street", None
    cap, rad = scheduler._speak_bounds(sim, a, 0)
    assert cap == 20 and rad == 5.0


# --------------------------------------------------------------------------- #
# (6) speak 経路: 列挙段 cap == _attention_limited(同一規則の機械証明)
# --------------------------------------------------------------------------- #
def test_enumeration_cap_equals_attention_limited(tmp_path):
    """列挙の中で絞った結果が、フル列挙 → S15 選抜と**同一**(二重適用も冪等)。"""
    sim = _sim(tmp_path, "s15_eq", n=40,
               **{"world.attention_hearers_max": 6})
    for a in sim.agents:                             # 全員を路上の狭い範囲へ寄せる
        a.loc, a.building, a.sleeping = "street", None, False
        a.x = float((a.id * 7) % 11)
        a.y = float((a.id * 5) % 13)
    radius = float(sim.cfg.world.perception_radius_m)
    for speaker in sim.agents[:5]:
        full = P.hearers_of(speaker, sim.agents, radius)
        want = scheduler._attention_limited(sim, speaker, full)
        got = P.hearers_of(speaker, sim.agents, radius, cap=6)
        assert _ids(got) == _ids(want), speaker.id
        # 保険(_attention_limited)を二重に掛けても変わらない = 冪等
        assert _ids(scheduler._attention_limited(sim, speaker, got)) == _ids(want)


# --------------------------------------------------------------------------- #
# (7) C2 近傍 cap の配線(conversation.run_phase)
# --------------------------------------------------------------------------- #
def test_c2_neighbors_max_default_is_unbounded(tmp_path):
    sim = _sim(tmp_path, "c2cap_off", n=8)
    assert CONV.neighbors_max(sim) == 0


def test_c2_neighbors_max_reads_conf_and_caches(tmp_path):
    sim = _sim(tmp_path, "c2cap_on", n=8, **{"world.c2_neighbors_max": 15})
    assert CONV.neighbors_max(sim) == 15
    assert CONV.neighbors_max(sim) == 15             # キャッシュ経路
    neg = _sim(tmp_path, "c2cap_neg", n=8, **{"world.c2_neighbors_max": -3})
    assert CONV.neighbors_max(neg) == 0              # 負値は無制限へ倒す


def test_c2_cap_bounds_the_pass_by_counter(tmp_path):
    """cap を掛けると C3 のすれ違い集合(その日の相異なる相手)が縮む。

    会話成立(meet_prob=0)を止めて**列挙そのもの**を観測する(会話が成立すると
    `1 step 1 会話` の break で走査が途中で切れるため)。"""
    def _run(name, **ov):
        sim = _sim(tmp_path, name, n=30, steps=1,
                   **{"conversation.enabled": "true",
                      "conversation.c2.meet_prob": 0.0, **ov})
        for a in sim.agents:
            a.loc, a.building, a.floor, a.sleeping = "street", None, 0, False
            a.x = float(a.id % 6)
            a.y = float(a.id // 6)
        sim.percept_index = P.build_index(
            sim.agents, float(sim.cfg.world.perception_radius_m))
        from society import conversation
        conversation.run_phase(sim, 0, 0)
        assert not [e for e in sim.logger.events if e.kind == "conversation"]
        return max(len(getattr(a, "_c3_pass", ())) for a in sim.agents)

    wide = _run("c2_wide")
    tight = _run("c2_tight", **{"world.c2_neighbors_max": 3})
    assert wide == 29, wide          # 既定は 40m 圏の全員(自分以外)
    assert tight < wide, (wide, tight)


def test_c2_speech_level_shrinks_the_pass_by_set(tmp_path):
    """声の段階(normal=屋外 5m)だけでも C2 の母集団が物理層で縮む。"""
    def _run(name, **ov):
        sim = _sim(tmp_path, name, n=30, steps=1,
                   **{"conversation.enabled": "true",
                      "conversation.c2.meet_prob": 0.0, **ov})
        for a in sim.agents:
            a.loc, a.building, a.floor, a.sleeping = "street", None, 0, False
            a.x = float(a.id % 6) * 2.0
            a.y = float(a.id // 6) * 2.0
        sim.percept_index = P.build_index(
            sim.agents, float(sim.cfg.world.perception_radius_m))
        from society import conversation
        conversation.run_phase(sim, 0, 0)
        return max(len(getattr(a, "_c3_pass", ())) for a in sim.agents)

    assert _run("sp_off") == 29
    assert _run("sp_on", **{"world.speech_levels.enabled": "true"}) < 29


# --------------------------------------------------------------------------- #
# (8) 決定論(同 seed 2 ラン一致)
# --------------------------------------------------------------------------- #
def test_deterministic_with_both_knobs_on(tmp_path):
    ov = {"world.speech_levels.enabled": "true", "world.c2_neighbors_max": 15,
          "world.attention_hearers_max": 20, "conversation.enabled": "true"}
    a = _sim(tmp_path, "det_a", n=24, steps=12, **ov)
    a.run()
    b = _sim(tmp_path, "det_b", n=24, steps=12, **ov)
    b.run()
    assert _l1(a) == _l1(b)


def test_default_off_run_matches_pure_default(tmp_path):
    """新キーを既定値で明示指定しても L1 が完全一致(1 バイトも動かない)。"""
    base = _sim(tmp_path, "off_base", n=24, steps=12)
    base.run()
    same = _sim(tmp_path, "off_same", n=24, steps=12,
                **{"world.speech_levels.enabled": "false",
                   "world.c2_neighbors_max": 0,
                   "world.attention_hearers_max": 0})
    same.run()
    assert _l1(base) == _l1(same)


# --------------------------------------------------------------------------- #
# (9) 契約列挙ピン(conf 既定・finals 値・registry 宣言)
# --------------------------------------------------------------------------- #
def test_conf_defaults_are_all_off():
    cfg = load_config()
    assert int(cfg.world.c2_neighbors_max) == 0
    assert bool(cfg.world.speech_levels.enabled) is False
    assert int(cfg.world.attention_hearers_max) == 0
    lv = cfg.world.speech_levels
    assert [float(lv.normal.outdoor_m), float(lv.normal.indoor_m)] == [5.0, 10.0]
    assert [float(lv.raised.outdoor_m), float(lv.raised.indoor_m)] == [12.0, 20.0]
    assert [float(lv.shout.outdoor_m), float(lv.shout.indoor_m)] == [30.0, 40.0]
    assert set(lv.keys()) == {"enabled", "normal", "raised", "shout"}


def test_registry_declares_the_new_keys():
    for dotted in ("world.speech_levels.enabled", "world.c2_neighbors_max"):
        f = R.BY_ID.get(dotted)
        assert f is not None, f"{dotted} がレジストリ未宣言"
        assert f.repro_tier == "strict", dotted      # LLM の自由文を読まない
        assert f.fingerprint_risk == "none", dotted
        # 層3(聞き手側の drive 加算が動く)= S15 と同じ理由で affects_k を正直に名乗る
        assert f.affects_k is True, dotted
    assert R.BY_ID["world.c2_neighbors_max"].off_value == 0
    assert R.undeclared_toggles(load_config()) == []


def test_finals_profile_declares_the_registered_values():
    """★層3 の事前登録値をピンする(黙って別の値へ差し替わることを止める)。"""
    fin = OmegaConf.load(_REPO_FINALS)
    assert int(fin.world.c2_neighbors_max) == 15
    assert int(fin.world.attention_hearers_max) == 20
    assert bool(fin.world.speech_levels.enabled) is True
    assert float(fin.world.speech_levels.normal.outdoor_m) == 5.0
    assert float(fin.world.speech_levels.normal.indoor_m) == 10.0
    assert float(fin.world.speech_levels.raised.outdoor_m) == 12.0
    assert float(fin.world.speech_levels.shout.outdoor_m) == 30.0


def test_no_new_event_kinds_in_lane_one():
    """レーン1 は L1 の kind を 1 つも足さない(絞るだけ)。"""
    assert "speech_level" not in EVENT_KINDS
    assert "hearer_cap" not in EVENT_KINDS


def test_touched_files_are_not_frozen():
    """本レーンが触ったファイルが凍結 SPEC_FILES に 1 つも入っていない。"""
    from society.observer import metrics_spec as MS
    touched = (
        "src/society/world/perception.py",
        "src/society/conversation.py",
        "src/society/engine/scheduler.py",
        "src/society/registry.py",
        "src/society/timeconv.py",
    )
    for rel in touched:
        assert rel not in MS.SPEC_FILES, f"凍結ファイルを触っている: {rel}"
