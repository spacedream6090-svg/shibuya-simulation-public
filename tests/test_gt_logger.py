"""第114 GT ロガー G1-G7 + 認知スタックの搬送棚卸し(OBS)。

正典: docs/plans/metaverse-projection-plan.md §4。

  G1  入力 3 件(プール/地図/組織台帳)の sha256 を run_manifest.json へ
  G2  checkpoint/dormant の剪定禁止(運用文書。実装ゼロ)= 本 file の対象外
  G3  roster.parquet が finals conf で ON(第112 実装済み)
  G4  memory.parquet(記憶ストリームの日次スナップショット)
  G5  relations.parquet(関係台帳の日次差分 + C3 すれ違い集計)
  G6  channels.parquet に価値 4 軸の充足 sat 4 列
  G7  小物束(plan/reflect/emotion/traits/worldview の payload 追記)

**共通の受入条件**: 当該トグル OFF では 1 バイトも変わらないこと(観測がシムを変えない)。
全経路 mock(実 LLM 不使用)。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from society import registry as R                       # noqa: E402
from society.agents.agent import Agent                  # noqa: E402
from society.agents.memory import Episode, MemoryStore  # noqa: E402
from society.config import load_config                  # noqa: E402
from society.engine.simulation import Simulation        # noqa: E402
from society.observer import channels as OC             # noqa: E402
from society.observer import gt_extras as GT            # noqa: E402
from society.observer import manifest as MAN            # noqa: E402
from society.observer import memory as OM               # noqa: E402
from society.observer import relations as ORL           # noqa: E402

GT_ON = {"observer.memory_daily.enabled": "true",
         "observer.relations_daily.enabled": "true",
         "observer.gt_extras.enabled": "true",
         "observer.input_provenance.enabled": "true"}


def _cfg(name, n_steps=24, n_agents=12, **ov):
    dot = ["run.seed=42", f"run.n_agents={n_agents}", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock", "observer.snapshot_every=144"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def _sim(tmp_path, name, n_steps=24, n_agents=12, **ov):
    return Simulation(_cfg(name, n_steps, n_agents, **ov), out_dir=tmp_path / name)


def _fake_agent(aid, **ov):
    a = Agent(id=aid, name=f"人{aid}", age=20 + aid, occupation="会社員",
              persona="p", traits={"openness": 0.25}, states={}, mem=MemoryStore())
    for k, v in ov.items():
        setattr(a, k, v)
    return a


class _Stub:
    """sim の最小スタブ(観測層は sim.agents しか読まない)。"""

    def __init__(self, agents):
        self.agents = list(agents)


# =========================================================================== #
# 共通: 既定 OFF は 1 ファイルも 1 キーも足さない
# =========================================================================== #
def test_all_gt_loggers_are_off_by_default(tmp_path):
    sim = _sim(tmp_path, "gt_off", n_steps=2, n_agents=6)
    assert sim.memory_sc is None and sim.relations_sc is None
    assert sim.channels_sat is False
    assert sim.gt_extras["enabled"] is False
    sim.run()
    assert not (sim.out_dir / "memory.parquet").exists()
    assert not (sim.out_dir / "relations.parquet").exists()
    man = json.loads((sim.out_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert "inputs" not in man, "既定 OFF なのに manifest に inputs 節が生えている"


def test_toggles_are_declared_in_the_registry():
    feats = {f.id: f for f in R.FEATURES}
    for fid in ("observer.input_provenance.enabled", "observer.memory_daily.enabled",
                "observer.relations_daily.enabled", "observer.gt_extras.enabled",
                "cognition.channels.sat_columns"):
        assert fid in feats, f"{fid} が registry に宣言されていない"
        f = feats[fid]
        assert f.off_value is False and f.affects_k is False
        assert f.repro_tier == "strict" and f.fingerprint_risk == "none"


def test_no_undeclared_toggles_after_the_new_keys():
    assert R.undeclared_toggles(load_config()) == []


def test_new_modules_are_not_in_the_frozen_spec():
    """凍結 14 ファイル(metrics_spec)に 1 本も触れていない = spec hash 無風。"""
    from society.observer import metrics_spec as MS
    for rel in ("src/society/observer/memory.py", "src/society/observer/relations.py",
                "src/society/observer/gt_extras.py"):
        assert rel not in MS.SPEC_FILES


# =========================================================================== #
# G1  入力データの来歴 sha256
# =========================================================================== #
def test_g1_file_sha256_is_streaming_and_correct(tmp_path):
    import hashlib
    blob = b"x" * (1 << 17) + b"tail"
    p = tmp_path / "big.bin"
    p.write_bytes(blob)
    assert MAN.file_sha256(p) == hashlib.sha256(blob).hexdigest()


def test_g1_dir_hash_is_deterministic_and_order_free(tmp_path):
    """ディレクトリは「シャード毎 sha256 + 結合ハッシュ」。走査順は結果に漏れない。"""
    d = tmp_path / "pool"
    (d / "L2").mkdir(parents=True)
    (d / "meta.json").write_text('{"a":1}', encoding="utf-8")
    (d / "L2" / "part-0001.jsonl").write_text("b\n", encoding="utf-8")
    (d / "L2" / "part-0000.jsonl").write_text("a\n", encoding="utf-8")
    got = MAN.path_sha256(d)
    assert got["kind"] == "dir" and got["n_files"] == 3
    assert set(got["files"]) == {"meta.json", "L2/part-0000.jsonl", "L2/part-0001.jsonl"}
    assert all("\\" not in k for k in got["files"]), "相対パスが posix 表記でない"
    assert MAN.path_sha256(d)["sha256"] == got["sha256"], "同じ木で hash が揺れる"
    # 1 バイト変えたら結合ハッシュが変わる(改竄が露見する)
    (d / "L2" / "part-0000.jsonl").write_text("A\n", encoding="utf-8")
    assert MAN.path_sha256(d)["sha256"] != got["sha256"]


def test_g1_missing_path_is_absent_not_a_fake_value(tmp_path):
    assert MAN.path_sha256(tmp_path / "nope") is None


def test_g1_manifest_carries_map_and_ledger(tmp_path):
    sim = _sim(tmp_path, "g1_on", n_steps=2, n_agents=6, **GT_ON)
    sim.run()
    man = json.loads((sim.out_dir / "run_manifest.json").read_text(encoding="utf-8"))
    inp = man["inputs"]
    assert inp["schema"] == 1
    assert len(inp["map"]["sha256"]) == 64 and inp["map"]["kind"] == "file"
    assert len(inp["org_ledger"]["sha256"]) == 64
    # プールを使わないランは "unused"(= 採らなかった / 実体が無かった を区別する)
    assert inp["persona_pool"]["kind"] == "unused"
    assert "sha256" not in inp["persona_pool"]


# =========================================================================== #
# G4  memory.parquet
# =========================================================================== #
def test_g4_capture_writes_one_row_per_memory(tmp_path):
    import pyarrow.parquet as pq

    sc = OM.MemoryDaily(tmp_path / "m1")
    a = _fake_agent(1)
    a.mem.episodes = [Episode(step=3, text="むかしの話", kind="event", importance=7.0)]
    a.mem.buffer = [Episode(step=9, text="さっきの話", kind="heard", importance=3.0)]
    assert sc.on_step(_Stub([a]), 0, 0) == 2
    assert sc.on_step(_Stub([a]), 1, 10) == 0, "同じ日に 2 度撮った"
    assert sc.on_step(_Stub([a]), 144, 1440) == 2, "翌日のスナップショットが撮れていない"
    sc.finalize()
    rows = pq.read_table(tmp_path / "m1" / "memory.parquet").to_pylist()
    assert len(rows) == 4
    d0 = {r["src"]: r for r in rows if r["day"] == 0}
    assert d0[OM.SRC_EPISODE]["text"] == "むかしの話"
    assert d0[OM.SRC_EPISODE]["importance"] == pytest.approx(7.0)
    assert d0[OM.SRC_EPISODE]["step"] == 3 and d0[OM.SRC_EPISODE]["kind"] == "event"
    assert d0[OM.SRC_BUFFER]["text"] == "さっきの話"


def test_g4_capture_does_not_touch_the_world(tmp_path):
    """観測側だけで閉じる: 記憶を 1 件も足さない/消さない。"""
    sc = OM.MemoryDaily(tmp_path / "m2")
    a = _fake_agent(1)
    a.mem.observe(1, "できごと")
    before = ([(e.step, e.text, e.kind, e.importance) for e in a.mem.buffer],
              [(e.step, e.text) for e in a.mem.episodes])
    sc.on_step(_Stub([a]), 0, 0)
    after = ([(e.step, e.text, e.kind, e.importance) for e in a.mem.buffer],
             [(e.step, e.text) for e in a.mem.episodes])
    assert before == after


def test_g4_text_chars_truncates_only_when_asked(tmp_path):
    a = _fake_agent(1)
    a.mem.buffer = [Episode(step=1, text="あいうえおかきくけこ")]
    full = OM.MemoryDaily(tmp_path / "m3")
    full.capture(_Stub([a]), 0)
    assert full.rows[0][7] == "あいうえおかきくけこ"
    cut = OM.MemoryDaily(tmp_path / "m4", text_chars=3)
    cut.capture(_Stub([a]), 0)
    assert cut.rows[0][7] == "あいう"


def test_g4_cfg_accepts_omegaconf_nodes():
    """★DictConfig は dict の部分型ではない(素の dict しか受けないと ON が OFF に落ちる)。"""
    cfg = load_config(["observer.memory_daily.enabled=true",
                       "observer.memory_daily.text_chars=40"])
    got = OM.cfg_of_config(cfg)
    assert got["enabled"] is True and got["text_chars"] == 40
    assert OM.cfg_of_config({})["enabled"] is False


# =========================================================================== #
# G5  relations.parquet(差分 + C3)
# =========================================================================== #
def test_g5_only_changed_pairs_are_written(tmp_path):
    sc = ORL.RelationsDaily(tmp_path / "r1")
    a = _fake_agent(1)
    a.mem.relations = {2: {"name": "b", "count": 1, "last_step": 0, "closeness": 1.5,
                           "tier": 1}}
    sim = _Stub([a])
    assert sc.on_step(sim, 0, 0) == 1, "初出の対が出ていない"
    assert sc.on_step(sim, 144, 1440) == 0, "動いていない対を書いている(差分でない)"
    a.mem.relations[2]["closeness"] = 6.0
    a.mem.relations[2]["tier"] = 2
    assert sc.on_step(sim, 288, 2880) == 1
    a.mem.relations[3] = {"name": "c", "count": 1, "last_step": 0}
    assert sc.on_step(sim, 432, 4320) == 1, "新しい対が出ていない"
    rows = [r for r in sc.rows if r[2] >= 0]
    assert [(r[0], r[2]) for r in rows] == [(0, 2), (2, 2), (3, 3)]
    assert rows[1][3] == pytest.approx(6.0) and rows[1][4] == 2


def test_g5_missing_closeness_is_null_not_zero(tmp_path):
    """relations 機構 OFF(closeness キー自体が無い)の台帳を 0 と偽らない。"""
    sc = ORL.RelationsDaily(tmp_path / "r2")
    a = _fake_agent(1)
    a.mem.relations = {2: {"name": "b", "count": 3, "last_step": 5}}
    sc.on_step(_Stub([a]), 0, 0)
    row = sc.rows[0]
    assert row[3] is None and row[4] is None
    assert row[5] == 3 and row[6] is False


def test_g5_c3_rows_carry_the_daily_passing_counts(tmp_path):
    sc = ORL.RelationsDaily(tmp_path / "r3")
    a = _fake_agent(1)
    a._c3_pass = {2, 3, 4}
    a._c3_greet = {2}
    sc.on_step(_Stub([a]), 0, 0)
    self_rows = [r for r in sc.rows if r[2] == ORL.SELF_ID]
    assert len(self_rows) == 1
    assert self_rows[0][7] == 3 and self_rows[0][8] == 1
    assert self_rows[0][3] is None, "すれ違い行に関係の値が入っている"


def test_g5_zero_passing_agents_get_no_row(tmp_path):
    """欠行 = 0(25 万 × 10 日を 0 で埋めない)。"""
    sc = ORL.RelationsDaily(tmp_path / "r4")
    a = _fake_agent(1)
    sc.on_step(_Stub([a]), 0, 0)
    assert sc.rows == []


def test_g5_passing_toggle_suppresses_only_the_c3_rows(tmp_path):
    sc = ORL.RelationsDaily(tmp_path / "r5", passing=False)
    a = _fake_agent(1)
    a._c3_pass = {2}
    a.mem.relations = {2: {"name": "b", "count": 1, "last_step": 0}}
    sc.on_step(_Stub([a]), 0, 0)
    assert [r[2] for r in sc.rows] == [2]


def test_g5_reload_last_restores_the_diff_baseline(tmp_path):
    """★resume==straight の要: 既存 part から「最後に出した値」を読み戻す。"""
    sc = ORL.RelationsDaily(tmp_path / "r6")
    a = _fake_agent(1)
    a.mem.relations = {2: {"name": "b", "count": 1, "last_step": 0, "closeness": 1.2345,
                           "tier": 1}}
    sim = _Stub([a])
    sc.on_step(sim, 0, 0)
    sc.flush_segment()                                  # checkpoint 相当
    again = ORL.RelationsDaily(tmp_path / "r6")         # resume 相当(新プロセス)
    again._resumed = True
    again._day = 0                                      # Simulation.run が据える印
    assert again.on_step(sim, 144, 1440) == 0, "resume 後に全対が初出として再出力された"
    a.mem.relations[2]["count"] = 2
    assert again.on_step(sim, 288, 2880) == 1


def test_g5_closeness_roundtrips_bit_exactly(tmp_path):
    """float32 だと round(x,4) の丸めが往復で動く → 差分基準が崩れる(float64 で固定)。"""
    import pyarrow.parquet as pq

    sc = ORL.RelationsDaily(tmp_path / "r7")
    a = _fake_agent(1)
    a.mem.relations = {2: {"name": "b", "count": 1, "last_step": 0,
                           "closeness": 12.3456, "tier": 2}}
    sc.on_step(_Stub([a]), 0, 0)
    sc.finalize()
    got = pq.read_table(tmp_path / "r7" / "relations.parquet").to_pylist()[0]
    assert got["closeness"] == round(12.3456, 4)


# =========================================================================== #
# G6  channels の sat 4 列
# =========================================================================== #
def test_g6_sat_columns_come_from_the_values_layer():
    from society.values import TAGS
    cols = OC.sat_columns()
    assert len(cols) == len(TAGS) == 4
    assert all(c.startswith("sat_") for c in cols)


def test_g6_sat_values_are_null_when_the_mechanism_is_off():
    a = _fake_agent(1)
    assert OC.sat_values(a) == [None] * 4, "sat 機構 OFF を 0 で埋めている"


def test_g6_append_sat_extends_rows_in_agent_order():
    from society.values import TAGS
    a, b = _fake_agent(1), _fake_agent(2)
    a.sat = {t: 0.25 for t in TAGS}
    rows = [(0, 0, 1, 9.0), (0, 0, 2, 8.0)]
    got = OC.append_sat(_Stub([a, b]), rows)
    assert got[0][:4] == (0, 0, 1, 9.0)
    assert got[0][4:] == (0.25, 0.25, 0.25, 0.25)
    assert got[1][4:] == (None, None, None, None)


def test_g6_append_sat_survives_a_row_order_mismatch():
    """並びが崩れても id で引き直す(正しさは検算の側が持つ)。"""
    from society.values import TAGS
    a, b = _fake_agent(1), _fake_agent(2)
    b.sat = {t: 0.75 for t in TAGS}
    got = OC.append_sat(_Stub([a, b]), [(0, 0, 2, 1.0)])
    assert got[0][4:] == (0.75, 0.75, 0.75, 0.75)


def test_g6_does_not_move_the_frozen_channel_spec():
    """★sat を CHANNELS に足していない = σ_c 凍結ファイルが無効化されない。"""
    from society.cognition import channels as CC
    assert not any(c.id.startswith("sat") for c in CC.CHANNELS)
    assert all(not col.startswith("sat_") for col in CC.COLUMNS)


def test_g6_columns_appear_only_when_the_toggle_is_on(tmp_path):
    import pyarrow.parquet as pq

    off = _sim(tmp_path, "g6_off", n_steps=3, n_agents=6,
               **{"cognition.channels.enabled": "true"})
    off.run()
    on = _sim(tmp_path, "g6_on", n_steps=3, n_agents=6,
              **{"cognition.channels.enabled": "true",
                 "cognition.channels.sat_columns": "true",
                 "freedom.open_actions": "true"})
    on.run()
    c_off = pq.read_table(tmp_path / "g6_off" / "channels.parquet").column_names
    c_on = pq.read_table(tmp_path / "g6_on" / "channels.parquet").column_names
    assert not any(c.startswith("sat_") for c in c_off)
    assert c_on[:len(c_off)] == c_off, "既存列の並びが変わっている"
    assert [c for c in c_on if c.startswith("sat_")] == list(OC.sat_columns())


# =========================================================================== #
# G7  小物束
# =========================================================================== #
def test_g7_plan_extras_projects_mood_carry_and_withs():
    plan = {"mood": "すこし気が重い", "carry": "昨日の続き"}
    blocks = [{"with": []}, {"with": [7, 9]}]
    got = GT.plan_extras(plan, blocks)
    assert got["mood"] == "すこし気が重い" and got["carry"] == "昨日の続き"
    assert got["withs"] == [[], ["7", "9"]], "blocks と同じ添字で並んでいない"
    assert "withs" not in GT.plan_extras({}, [{"with": []}])
    assert GT.plan_extras({}, []) == {}


def test_g7_reflect_extras_keeps_the_generation_time_truncation():
    got = GT.reflect_extras({"self": "自分は", "ties": "人とは"})
    assert got == {"self": "自分は", "ties": "人とは"}
    assert GT.reflect_extras({"self": "x", "ties": ""}) == {"self": "x"}
    assert GT.reflect_extras(None) == {}


def test_g7_emotion_extras_carries_the_phrase():
    assert GT.emotion_extras("すこし気持ちが沈んでいる")["phrase"]
    assert GT.emotion_extras("") == {}


def test_g7_needs_extras_names_no_dimension_in_code():
    a = _fake_agent(1)
    assert GT.needs_extras(a) == {}, "needs OFF の個体にキーが生えている"
    a._needs_profile = {"zeta": 0.25, "alpha": 0.75}
    a.needs_mods = {"sns": 1.2}
    got = GT.needs_extras(a)
    assert got["needs"] == {"alpha": 0.75, "zeta": 0.25}
    assert got["needs_mods"] == {"sns": 1.2}
    src = (_ROOT / "src" / "society" / "observer" / "gt_extras.py").read_text(encoding="utf-8")
    for dim in ("stimulation", "security", "relatedness", "competence", "autonomy"):
        assert dim not in src, f"gt_extras.py が価値次元 {dim} を名指ししている"


def test_g7_expect_extras_is_deterministic_top_k():
    a = _fake_agent(1)
    a.wv_expect = {("p1", 0): 3.0, ("p2", 1): 9.0, ("p3", 2): 9.0, ("p4", 3): 1.0}
    got = GT.expect_extras(a, 2)["expect"]
    assert got == [["p2", 1, 9.0], ["p3", 2, 9.0]], "期待人数の降順+決定論のタイ解消でない"
    assert GT.expect_extras(a, 0) == {}
    assert GT.expect_extras(_fake_agent(2), 8) == {}


def test_g7_traits_json_gains_the_needs_block_only_when_on(tmp_path):
    off = _sim(tmp_path, "g7_off", n_steps=1, n_agents=6, **{"needs.enabled": "true"})
    on = _sim(tmp_path, "g7_on", n_steps=1, n_agents=6,
              **{"needs.enabled": "true", "observer.gt_extras.enabled": "true"})
    t_off = json.loads((off.out_dir / "traits.json").read_text(encoding="utf-8"))
    t_on = json.loads((on.out_dir / "traits.json").read_text(encoding="utf-8"))
    rec_off, rec_on = t_off["0"], t_on["0"]
    assert "needs" not in rec_off and "needs_mods" not in rec_off
    assert rec_on["needs"] and rec_on["needs_mods"]
    assert set(rec_on) - set(rec_off) == {"needs", "needs_mods"}


def test_g7_worldview_payload_gains_expect_only_when_on(tmp_path):
    dot = {"worldview.enabled": "true", "ontology.enabled": "true"}
    off = _sim(tmp_path, "g7wv_off", n_steps=150, n_agents=8, **dot)
    off.run()
    on = _sim(tmp_path, "g7wv_on", n_steps=150, n_agents=8,
              **dict(dot, **{"observer.gt_extras.enabled": "true"}))
    on.run()

    def _wv(sim):
        return [e for e in sim.logger.events
                if e.kind == "worldview" and e.agent_id >= 0]
    assert _wv(off), "worldview イベントが 1 件も出ていない(スモークとして不成立)"
    assert all("expect" not in (e.payload or {}) for e in _wv(off))
    assert any("expect" in (e.payload or {}) for e in _wv(on)), \
        "ON なのに期待表が 1 件も載っていない"


# =========================================================================== #
# 観測不変性: サイドカーの ON/OFF で L1 がバイト不変
# =========================================================================== #
def test_sidecars_do_not_change_l1_bytes(tmp_path):
    """★G4/G5/G6 を全部 ON にしても l1_events.parquet が**バイト一致**。

    G7 は payload にキーを足す層なので**ここには含めない**(足りることが仕様)。
    """
    base = _sim(tmp_path, "inv_off", n_steps=150, n_agents=10,
                **{"cognition.channels.enabled": "true", "freedom.open_actions": "true"})
    base.run()
    on = _sim(tmp_path, "inv_on", n_steps=150, n_agents=10,
              **{"cognition.channels.enabled": "true", "freedom.open_actions": "true",
                 "cognition.channels.sat_columns": "true",
                 "observer.memory_daily.enabled": "true",
                 "observer.relations_daily.enabled": "true",
                 "observer.input_provenance.enabled": "true"})
    on.run()
    a = (tmp_path / "inv_off" / "l1_events.parquet").read_bytes()
    b = (tmp_path / "inv_on" / "l1_events.parquet").read_bytes()
    assert a == b, "観測サイドカーの ON/OFF で L1 parquet がバイト不一致(記録が動力学へ漏れた)"
    assert (tmp_path / "inv_on" / "memory.parquet").exists()
    assert (tmp_path / "inv_on" / "relations.parquet").exists()


def test_gt_extras_off_keeps_l1_bytes(tmp_path):
    """G7 OFF が既存 L1 とバイト一致(= キーを 1 つも積んでいない)ことの裏取り。"""
    a = _sim(tmp_path, "x_a", n_steps=48, n_agents=8)
    a.run()
    b = _sim(tmp_path, "x_b", n_steps=48, n_agents=8,
             **{"observer.gt_extras.enabled": "false"})
    b.run()
    assert (tmp_path / "x_a" / "l1_events.parquet").read_bytes() == \
        (tmp_path / "x_b" / "l1_events.parquet").read_bytes()


# =========================================================================== #
# resume == straight(新サイドカー全 ON)
# =========================================================================== #
def _dot(name, n_steps, every):
    return {"observer.memory_daily.enabled": "true",
            "observer.relations_daily.enabled": "true",
            "observer.roster_daily.enabled": "true",
            "cognition.channels.enabled": "true",
            "cognition.channels.sat_columns": "true",
            "freedom.open_actions": "true",
            "observer.checkpoint_every": every}


@pytest.mark.parametrize("stem", ["memory", "relations", "roster", "channels"])
def test_split_execution_matches_the_straight_run(tmp_path, stem):
    """★resume==straight: 分割実行(80 + resume)の新サイドカーが一気通しと完全一致する。"""
    import pyarrow.parquet as pq

    straight = tmp_path / "rs_straight"
    if not straight.exists():
        Simulation(_cfg("rs_straight", 200, 10, **_dot("rs_straight", 200, 0)),
                   out_dir=straight).run()
        split = tmp_path / "rs_split"
        Simulation(_cfg("rs_split", 80, 10, **_dot("rs_split", 80, 80)),
                   out_dir=split).run()                     # 前チャンク(clean finalize)
        Simulation(_cfg("rs_split", 200, 10, **_dot("rs_split", 200, 80)),
                   out_dir=split).run(resume_from=split)    # 後チャンク(途中再開)
    split = tmp_path / "rs_split"
    a = pq.read_table(straight / f"{stem}.parquet").to_pylist()
    b = pq.read_table(split / f"{stem}.parquet").to_pylist()
    assert a, f"{stem}.parquet が空(検査が空振り)"
    assert len(a) == len(b), f"{stem}: 分割実行で行数が変わった({len(a)} vs {len(b)})"
    assert a == b, f"{stem}: 分割実行で中身が変わった"


# =========================================================================== #
# OBS  認知スタックの回転搬送棚卸し(pool.dehydrate/hydrate)
# =========================================================================== #
def test_obs_watch_spec_survives_a_rotation():
    """監視仕様(ô + トリガ)= 設計 §2.2「有効期限を持たせない」→ 街を出ても消えない。"""
    from society.world import pool as P
    a = _fake_agent(1)
    a._fire_watch = {"expect": {3: 2.5}, "triggers": [("こみあい", 0, ">", 12.0)]}
    st = P.dehydrate(a)
    assert st["watch"]["expect"] == {"3": 2.5}
    b = _fake_agent(1)
    P.hydrate(b, st)
    assert b._fire_watch["expect"] == {3: 2.5}
    assert b._fire_watch["triggers"] == [("こみあい", 0, ">", 12.0)]


def test_obs_behav_ema_survives_but_implicit_self_does_not():
    """★意味論の判断: 多日で育つ**ベースライン**は運び、日次上書きの派生テキストは運ばない。"""
    from society.world import pool as P
    a = _fake_agent(1)
    a.behav_ema = {"sns": 2.5, "_neg": 0.3, "_pos": 0.1}
    a.implicit_self = "SNSを見る時間がいつもより増えている"
    st = P.dehydrate(a)
    assert st["behav_ema"] == {"sns": 2.5, "_neg": 0.3, "_pos": 0.1}
    assert "implicit_self" not in st
    b = _fake_agent(1)
    P.hydrate(b, st)
    assert b.behav_ema == {"sns": 2.5, "_neg": 0.3, "_pos": 0.1}


def test_obs_deep_reflection_reservation_and_cooldown_survive():
    from society.world import pool as P
    a = _fake_agent(1)
    a.deep_due_day = 4
    a.deep_cooldown_until_day = 7
    st = P.dehydrate(a)
    assert st["misc"]["deep_due_day"] == 4
    assert st["misc"]["deep_cooldown_until_day"] == 7
    b = _fake_agent(1)
    P.hydrate(b, st)
    assert b.deep_due_day == 4 and b.deep_cooldown_until_day == 7


def test_obs_satiation_survives_a_rotation():
    """価値 4 軸の充足は多日スケール(日次 15% しか中立へ戻らない)。"""
    from society.values import TAGS
    from society.world import pool as P
    a = _fake_agent(1)
    a.sat = {t: 0.2 for t in TAGS}
    st = P.dehydrate(a)
    b = _fake_agent(1)
    b.sat = {t: 0.5 for t in TAGS}          # build_pool_agent が据え直す中立値
    P.hydrate(b, st)
    assert b.sat == {t: 0.2 for t in TAGS}, "再来街で渇きが中立へ戻っている"


def test_obs_carry_set_is_byte_identical_when_the_mechanisms_are_off():
    """★当該機構 OFF の個体では退避 dict が 1 バイトも変わらない(既存の dict 等値を守る)。"""
    from society.world import pool as P
    a = _fake_agent(1)
    st = P.dehydrate(a)
    for key in ("watch", "behav_ema", "sat"):
        assert key not in st, f"機構 OFF なのに退避 dict に {key} が生えている"
    assert "misc" not in st, "既定値の個体に misc が生えている"


def test_obs_volatile_cognitive_fields_are_deliberately_not_carried():
    """棚卸しの結論を機械で固定する(判断を後から読める形で残す)。

    運ばない 3 族と理由:
      _fire_obs / _fire_pred / _fire_prev_drive … 毎 step 末に凍結し直す(step スケール)
      _engaged / _engaged_refr ………………………… 不応期 30 分 << 回転間隔 1440 分
      _bore_node / _bore_cooldown ………………… 場所に紐づく(再来街は別ノード=どのみち再設定)
    """
    from society.world import pool as P
    a = _fake_agent(1)
    a._fire_obs = (1.0, 2.0)
    a._fire_pred = (1.0, 2.0)
    a._fire_prev_drive = 0.4
    a._engaged = {"kind": "talk", "start": 10}
    a._engaged_refr = {"until": 40, "trigger": "salience"}
    a._bore_node = "n1"
    a._bore_cooldown = 12
    st = P.dehydrate(a)
    for key in ("fire_obs", "fire_pred", "fire_prev_drive", "engaged",
                "engaged_refr", "bore_node", "bore_cooldown"):
        assert key not in st and key not in (st.get("misc") or {})
