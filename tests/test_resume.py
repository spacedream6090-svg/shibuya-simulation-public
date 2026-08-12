"""D16: チェックポイント / 途中再開の一致テスト(合格条件そのもの)。

合格条件:
- 「40step 一気」==「20step で ckpt → 新プロセス相当で load → 40step まで」の l1_events が
  完全一致(全カラム・全行。少なくとも kind/step/agent_id/payload)。l2/l3 も一致。
- checkpoint 無効(既定)時に挙動・出力が従来と完全一致(part を一切作らない)。
- straight run 中に checkpoint を挟んでも(flush_segment→finalize 結合)出力が不変。
- scenario shock_closure を跨いだ resume(封鎖中に save→load して復元後も封鎖が効く)。
"""
from __future__ import annotations

from collections import deque
from pathlib import Path

import pyarrow.parquet as pq

from society import worldview as _wv
from society.config import load_config
from society.engine import checkpoint, scheduler
from society.engine.simulation import Simulation
from society.observer.schema import Event


def _cfg(name: str, n_steps: int, **ov):
    dot = ["run.seed=42", "run.n_agents=20", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def _rows(run_dir: Path, stem: str = "l1_events") -> list[dict]:
    return pq.read_table(Path(run_dir) / f"{stem}.parquet").to_pylist()


def _run_straight(tmp_path, name: str, n_steps: int, **ov) -> Path:
    d = tmp_path / name
    Simulation(_cfg(name, n_steps, **ov), out_dir=d).run()
    return d


def _run_resume(tmp_path, name: str, split: int, total: int, **ov) -> Path:
    """phase1: split step 走らせて ckpt を書き finalize せず中断(クラッシュ相当)。
       phase2: 新 Simulation で load → total まで走らせ finalize(part 結合)。"""
    d = tmp_path / name
    every = {"observer.checkpoint_every": split}
    # --- phase 1(中断) ---
    sim1 = Simulation(_cfg(name, split, **every, **ov), out_dir=d)
    for step in range(split):
        scheduler.run_step(sim1, step)
    checkpoint.save(sim1, split, d / "checkpoint" / f"ckpt-{split:06d}.pkl.gz")
    sim1.logger.flush_segment()
    # --- phase 2(途中再開) ---
    sim2 = Simulation(_cfg(name, total, **every, **ov), out_dir=d)
    sim2.run(resume_from=d)
    return d


# --------------------------------------------------------------------------- #
def test_resume_matches_straight_all_layers(tmp_path):
    """一気 40step と 20+resume の l1/l2/l3 が全行・全カラム一致。"""
    straight = _run_straight(tmp_path, "straight", 40)
    resumed = _run_resume(tmp_path, "resumed", 20, 40)
    a = _rows(straight, "l1_events")
    b = _rows(resumed, "l1_events")
    assert len(a) == len(b), f"l1 行数不一致: {len(a)} vs {len(b)}"
    assert a == b, "l1_events が byte 級で不一致"
    for stem in ("l2_metrics", "l3_snapshots"):
        assert _rows(straight, stem) == _rows(resumed, stem), f"{stem} 不一致"


def test_checkpoint_disabled_is_byte_identical(tmp_path):
    """既定(checkpoint_every=0)は part を作らず、明示 0 とも完全一致(従来挙動の保存)。"""
    a = _run_straight(tmp_path, "def", 30)                       # 既定 0
    b = _run_straight(tmp_path, "zero", 30, **{"observer.checkpoint_every": 0})
    assert _rows(a) == _rows(b)
    assert not list(a.glob("l1_events.part-*.parquet")), "既定で part が作られている"


def test_segmented_straight_matches_plain(tmp_path):
    """straight run 中に checkpoint を挟んでも(flush_segment→結合)出力が不変。"""
    plain = _run_straight(tmp_path, "plain", 40)                 # every=0
    seg = _run_straight(tmp_path, "seg", 40,
                        **{"observer.checkpoint_every": 10})     # 4 セグメントを結合
    assert _rows(plain, "l1_events") == _rows(seg, "l1_events")
    for stem in ("l2_metrics", "l3_snapshots"):
        assert _rows(plain, stem) == _rows(seg, stem)
    # part は finalize で結合・削除され、canonical だけが残る
    assert not list(seg.glob("l1_events.part-*.parquet"))
    assert (seg / "l1_events.parquet").exists()


def test_resume_across_shock_closure(tmp_path):
    """封鎖中(step 5..35)に step 20 で save→load。復元後も封鎖が効き、l1 が一致する。"""
    ov = {"world.scenario": "shock_closure",
          "world.scenario_params":
              "{at_step: 5, duration_steps: 30, center: [0,0], radius_m: 150}"}
    straight = _run_straight(tmp_path, "sc_straight", 40, **ov)
    resumed = _run_resume(tmp_path, "sc_resume", 20, 40, **ov)
    assert _rows(straight, "l1_events") == _rows(resumed, "l1_events"), \
        "shock_closure を跨いだ resume の l1 が不一致"

    # step 20 の checkpoint を素の sim へ load → 封鎖が復元されていることを検査
    sim = Simulation(_cfg("sc_inspect", 40, **ov), out_dir=tmp_path / "sc_inspect")
    step = checkpoint.load(sim, tmp_path / "sc_resume" / "checkpoint" / "ckpt-000020.pkl.gz")
    assert step == 20
    assert sim.scenario.active and sim.scenario.closed, "封鎖状態が復元されていない"
    n_flags = sum(1 for _u, _v, d in sim.city.graph.edges(data=True) if d.get("closed"))
    assert n_flags == len(sim.scenario.closed) > 0, "closed フラグが city へ再適用されていない"


def test_resume_relations_day_no_double_decay(tmp_path):
    """relations ON の resume==straight(第61検収補修=_rel_day の checkpoint 中央管理の固定)。

    _rel_day が checkpoint に無いと、日境界(step102=start_tod 07:00)を過ぎた後の resume 初 step で
    _phase_relations_day が同じ日を再処理(closeness 減衰/評判風化の二重発火)し straight と食い違う。
    split=105(境界処理済み)→110 の全層一致に加え、round-trip で _rel_day 復元を直接固定して
    (mock で関係イベントが偶然 0 件でも)検証が空回りしないことを保証する。"""
    ov = {"relations.enabled": "true"}
    straight = _run_straight(tmp_path, "rel_straight", 110, **ov)
    resumed = _run_resume(tmp_path, "rel_resumed", 105, 110, **ov)
    for stem in ("l1_events", "l2_metrics", "l3_snapshots"):
        assert _rows(straight, stem) == _rows(resumed, stem), f"{stem} 不一致(relations resume)"
    # 直接検証: 境界処理済みの _rel_day が checkpoint round-trip で復元される(空回り防止)
    d = tmp_path / "rel_ck"
    sim1 = Simulation(_cfg("rel_ck", 105, **{"observer.checkpoint_every": 105}, **ov), out_dir=d)
    for step in range(105):
        scheduler.run_step(sim1, step)
    assert getattr(sim1, "_rel_day", -1) >= 1, "日境界が未処理(テスト前提が崩れた=要再調整)"
    p = checkpoint.save(sim1, 105, d / "checkpoint" / "ckpt-000105.pkl.gz")
    sim2 = Simulation(_cfg("rel_ck2", 105, **{"observer.checkpoint_every": 105}, **ov),
                      out_dir=tmp_path / "rel_ck2")
    checkpoint.load(sim2, p)
    assert getattr(sim2, "_rel_day", None) == getattr(sim1, "_rel_day", None), \
        "_rel_day が checkpoint で復元されていない"


# =========================================================== 第98バッチ 小粒A(resume 整合の全数監査)
# 「mid-day resume が同じ暦日をもう一度処理する」型のギャップ(第80 W2 と同型)を src 全体から
# 洗い出して checkpoint の中央管理へ入れたことの機械固定。
#
#   (1) 実ランでの resume==straight … 価値充足の減衰(第17)/ 偶発イベント(同じ日を引き直すと
#       **同じ当たりが二重に効く**ので最も感度が高い)
#   (2) 往復(round-trip)での**全数**復元 … 実ランで動かない機構も含めて 1 件も落とさない
#   (3) 旧 checkpoint(新キー無し)からの復元が落ちない
_GUARD_ROUNDTRIP = {                       # sim 属性 → 往復させる値(既定値と紛れない値を使う)
    "_bank_day": 3, "_reflect_day": 4, "_freedom_day": 5, "_sched_day": 6,
    "_partner_day": 7, "_health_day": 8, "_chance_day": 9, "_goods_review_day": 10,
    "_council_budget_day": 11, "_vc_review_day": 12, "_crowd_surge_day": 13,
    "_annual_day": 14, "_status_day": 15, "_wv_day": 16,
    "_status_prev_rank": {1: 2, 3: 4},
    "_status_rank_mobility": 0.375,
    "_b2b_total": 17, "_delivery_total": 18, "_service_total": 19,
    "_inner_life_init": True,
    "_b2b": {"stock": {"o1": 5}, "revenue": {"o1": 900.0}, "procurement": 900.0,
             "sold_qty": {"o1": 5}, "trades": 17},
    "_delivery_pending": [{"orderer": 1, "arrive": 42, "cat": "食事", "dispatched": False}],
    "_ads_period": 2,
    "_ads_campaigns": {"枠A": {"campaign": "新装開店", "cat": "食事"}},
    # ---- 第98バッチ W4-A(観測レンズの暦日タリー + 火種介入の起動時1回フラグ)----
    # どれも「logger を増分走査して当日ぶんを積む」観測の state(echo/norm/regression と同型)。
    # watermark(_*_processed)は logger 由来なので**保存しない**=別途 0 復帰を固定する。
    "_lens_state": {"day": 3, "value": {"utility": 2, "emotion": 1, "social": 0,
                                        "epistemic": 4},
                    "motive": {"earn": 5, "love": 0, "recognition": 2}},
    "_silence_state": {"day": 3, "speakers": {1, 2, 5}, "n_deliberate": 7,
                       "n_nothing": 2},
    "_struct_state": {"day": 3, "formed": 2, "broken": 1, "decayed": 3, "active": 11},
    "_dev_state": {"day": 3,
                   "per": {1: {"disc_dev": 2, "disc_total": 3,
                               "full_dev": 2, "full_total": 5}},
                   "occ": {1: "研究員", 2: "会社員"}},
    "_spark_reported": True,
}


def _fresh(tmp_path, name: str, **ov):
    return Simulation(_cfg(name, 1, **ov), out_dir=tmp_path / name)


def test_checkpoint_roundtrips_every_day_guard(tmp_path):
    """日/期ガードと付随状態が **1 件残らず** checkpoint 往復で戻る(全数監査の固定)。

    実ランで発火しにくい機構(VC 審査・議会予算・災害…)も含めて表で押さえる。将来
    「日カウンタを足したのに checkpoint へ入れ忘れる」再発を、この表が空回りせず捕まえる。
    """
    src = _fresh(tmp_path, "guard_src")
    for attr, val in _GUARD_ROUNDTRIP.items():
        setattr(src, attr, val)
    src.today_event_line = "今日は祭りの日だ。"
    src.today_crowd_event = "祭り"
    src._wv_pioneer = deque([1, 0, 2], maxlen=7)
    src.council = {"members": [1, 2, 3], "elected_day": 2, "term": 4}
    src.council_campaign = {"term_end": 9, "open": True, "candidates": {1: 5000.0}}
    src._disaster_day = 20
    src._disaster_active, src._disaster_kind, src._disaster_until_day = True, "地震", 23
    src._infra_active, src._infra_kind, src._infra_until_day = True, "停電", 22
    src.today_disaster_line = "今日は地震で交通が麻痺している。"
    src.transit.suspended = True

    p = checkpoint.save(src, 7, tmp_path / "guard_src" / "checkpoint" / "ckpt-000007.pkl.gz")
    dst = _fresh(tmp_path, "guard_dst")
    assert checkpoint.load(dst, p) == 7

    for attr, val in _GUARD_ROUNDTRIP.items():
        assert getattr(dst, attr, None) == val, f"{attr} が checkpoint で復元されていない"
    assert dst.today_event_line == "今日は祭りの日だ。"
    assert dst.today_crowd_event == "祭り"
    assert list(dst._wv_pioneer) == [1, 0, 2], "規範窓(C6)が復元されていない"
    assert dst.council == {"members": [1, 2, 3], "elected_day": 2, "term": 4}
    assert dst.council_campaign["candidates"] == {1: 5000.0}
    assert (dst._disaster_day, dst._disaster_active, dst._disaster_kind,
            dst._disaster_until_day) == (20, True, "地震", 23)
    assert (dst._infra_active, dst._infra_kind, dst._infra_until_day) == (True, "停電", 22)
    assert dst.today_disaster_line == "今日は地震で交通が麻痺している。"
    assert dst.transit.suspended is True, "運休フラグが復元されていない(電車が勝手に復旧する)"
    # 第98 W4-A: 観測レンズの watermark は 4 本とも 0 に戻る(fresh logger から再走査)。
    for attr in ("_lens_processed", "_silence_processed", "_struct_processed",
                 "_dev_processed"):
        assert getattr(dst, attr, None) == 0, f"{attr} が 0 に戻っていない"


def test_old_checkpoint_without_new_keys_still_loads(tmp_path):
    """新キーを 1 つも持たない旧 checkpoint から復元しても落ちず、既定(=従来挙動)に落ちる。"""
    import gzip
    import pickle

    src = _fresh(tmp_path, "old_src")
    src._freedom_day, src._chance_day = 5, 6
    src.council = {"members": [1], "elected_day": 0, "term": 1}
    p = checkpoint.save(src, 3, tmp_path / "old_src" / "checkpoint" / "ckpt-000003.pkl.gz")

    with gzip.open(p, "rb") as f:                     # 新キーを剥がして「旧 checkpoint」を作る
        blob = pickle.loads(f.read())
    new_keys = ["bank_day", "reflect_day", "freedom_day", "sched_day", "partner_day",
                "health_day", "chance_day", "goods_review_day", "council_budget_day",
                "vc_review_day", "crowd_surge_day", "annual_day", "today_event_line",
                "today_crowd_event", "status_day", "status_prev_rank",
                "status_rank_mobility", "wv_day", "wv_pioneer", "disaster_state",
                "council", "council_campaign", "b2b", "b2b_total", "delivery_pending",
                "delivery_total", "service_total", "ads_period", "ads_campaigns",
                "inner_life_init",
                # 第98 W4-A
                "lens_state", "silence_state", "struct_state", "dev_state",
                "wv_carry", "spark_reported"]
    for k in new_keys:
        blob["runtime"].pop(k, None)
    with gzip.open(p, "wb") as f:
        f.write(pickle.dumps(blob, protocol=pickle.HIGHEST_PROTOCOL))

    dst = _fresh(tmp_path, "old_dst")
    assert checkpoint.load(dst, p) == 3
    assert dst._freedom_day == -1 and dst._chance_day == -1   # 旧 ckpt = 従来どおりの既定
    assert getattr(dst, "council", None) is None
    assert not getattr(dst, "_inner_life_init", False)


def test_resume_freedom_day_no_double_decay(tmp_path):
    """価値充足の減衰(第17)ON の resume==straight(_freedom_day の中央管理の固定)。

    _freedom_day が無いと、日境界(step102)を越えた後の resume 初 step で _phase_freedom_day が
    同じ日を再処理し **中立回帰が二重に適用**される。split=105(境界処理済み)→110 の全層一致に
    加え、round-trip での復元も直接固定して(mock で sat が偶然動かなくても)空回りを防ぐ。
    """
    ov = {"freedom.open_actions": "true"}
    straight = _run_straight(tmp_path, "free_straight", 110, **ov)
    resumed = _run_resume(tmp_path, "free_resumed", 105, 110, **ov)
    for stem in ("l1_events", "l2_metrics", "l3_snapshots"):
        assert _rows(straight, stem) == _rows(resumed, stem), f"{stem} 不一致(freedom resume)"

    d = tmp_path / "free_ck"
    every = {"observer.checkpoint_every": 105}
    sim1 = Simulation(_cfg("free_ck", 105, **every, **ov), out_dir=d)
    for step in range(105):
        scheduler.run_step(sim1, step)
    assert getattr(sim1, "_freedom_day", -1) >= 1, "日境界が未処理(テスト前提が崩れた=要再調整)"
    p = checkpoint.save(sim1, 105, d / "checkpoint" / "ckpt-000105.pkl.gz")
    sim2 = Simulation(_cfg("free_ck2", 105, **every, **ov), out_dir=tmp_path / "free_ck2")
    checkpoint.load(sim2, p)
    assert getattr(sim2, "_freedom_day", None) == getattr(sim1, "_freedom_day", None)


def test_resume_chance_day_no_double_windfall(tmp_path):
    """偶発イベント ON の resume==straight(_chance_day の中央管理の固定)。

    chance の stream キーは (agent.id, **day**) なので、同じ日をもう一度回すと **同じ当たりが
    そのまま二重に効く**(臨時収入の二重入金)。daily_rate=1.0 で全員に必ず 1 件出す条件に
    しているので、二重発火があれば chance_event の件数が倍になって即座に落ちる。
    """
    ov = {"chance.enabled": "true", "chance.daily_rate": "1.0"}
    straight = _run_straight(tmp_path, "ch_straight", 110, **ov)
    resumed = _run_resume(tmp_path, "ch_resumed", 105, 110, **ov)
    n_chance = sum(1 for r in _rows(straight, "l1_events") if r["kind"] == "chance_event")
    assert n_chance > 0, "chance_event が 1 件も出ていない(テスト前提が崩れた=要再調整)"
    for stem in ("l1_events", "l2_metrics", "l3_snapshots"):
        assert _rows(straight, stem) == _rows(resumed, stem), f"{stem} 不一致(chance resume)"


# ============================================================ 第98バッチ W4-A(resume 整合の完遂)
# 小粒A の申し送り 3 件を塞いだことの機械固定。
#   (1) spark_roster の resume 二重記録(起動時 1 回のイベントを 2 つ目のプロセスが出し直す)
#   (2) 観測レンズ 4 本(lens / silence / structure / deviation)の暦日タリー未保存
#   (3) worldview C2/C6 の「直前の日境界 〜 checkpoint」区間の取りこぼし
def test_resume_spark_roster_recorded_once(tmp_path):
    """spark ON の mid-day resume で spark_roster が **1 件のまま**(全層一致つき)。

    spark.apply は Simulation.__init__ から呼ばれるので resume した 2 つ目のプロセスでも走り、
    従来は同じ t=0 名簿イベントが L1 に 2 件並んでいた(scheduler の workplace_bound が
    `step != 0` ガードで防いでいるのと同型のギャップ)。件数 1 だけでなく全層一致も固定する。
    """
    ov = {"spark.enabled": "true"}
    straight = _run_straight(tmp_path, "spk_straight", 40, **ov)
    resumed = _run_resume(tmp_path, "spk_resumed", 20, 40, **ov)
    n_st = sum(1 for r in _rows(straight, "l1_events") if r["kind"] == "spark_roster")
    n_rs = sum(1 for r in _rows(resumed, "l1_events") if r["kind"] == "spark_roster")
    assert n_st == 1, "straight で spark_roster が 1 件でない(テスト前提が崩れた)"
    assert n_rs == 1, f"resume で spark_roster が二重記録されている: {n_rs} 件"
    for stem in ("l1_events", "l2_metrics", "l3_snapshots"):
        assert _rows(straight, stem) == _rows(resumed, stem), f"{stem} 不一致(spark resume)"


# 観測レンズ 4 本を 1 ランで同時に ON にする(どれも「logger 増分走査 + 暦日タリー」で同型)。
_LENS_ON = {"lens.enabled": "true", "lens.deviation.enabled": "true",
            "lens.structure.enabled": "true", "freedom.explicit_nothing": "true",
            "relations.enabled": "true"}
_LENS_COLS = ("value4_utility", "value4_social", "value4_epistemic", "value4_emotion",
              "motive_earn", "motive_love", "motive_recognition",
              "deviation_mean", "deviation_var", "deviation_top_share",
              "deviation_fulltime_mean", "edges_formed", "edges_broken",
              "edges_decayed", "edge_churn_rate",
              "silent_agent_rate", "chosen_nothing_rate")


def test_resume_observer_lens_day_tallies(tmp_path):
    """観測レンズ 4 本 ON の mid-day resume==straight(第98 W4-A の本体)。

    _lens_state / _silence_state / _struct_state / _dev_state は「当日(暦日)のタリー」で、
    保存しないと mid-day resume が**当日分を途中から数え直す**(silence.py の docstring が
    「resume の限界(正直註記)」として明記していたもの)。split=20→40 は日境界(step102)より
    手前=完全に mid-day なので、当該 L2 列は保存が無ければ必ず食い違う。
    """
    straight = _run_straight(tmp_path, "lens_straight", 40, **_LENS_ON)
    resumed = _run_resume(tmp_path, "lens_resumed", 20, 40, **_LENS_ON)
    a = _rows(straight, "l2_metrics")
    b = _rows(resumed, "l2_metrics")
    for col in _LENS_COLS:                    # 空回り防止: 17 列すべてが実在すること
        assert col in a[-1], f"L2 列 {col} が出ていない(テスト前提が崩れた=要再調整)"
    # 当日タリー由来の列が実際に動いている(全部 0 なら一致しても意味がない)
    assert any(a[-1][c] for c in ("value4_social", "value4_epistemic", "motive_earn",
                                  "deviation_mean", "silent_agent_rate")), \
        "レンズ列が全て 0(テスト前提が崩れた=要再調整)"
    for stem in ("l1_events", "l2_metrics", "l3_snapshots"):
        assert _rows(straight, stem) == _rows(resumed, stem), f"{stem} 不一致(lens resume)"


# --------------------------------------------------------------- worldview C2/C6 の未走査区間
# 日境界は start_tod=22:00 で step 12 / 156(既定 07:00 だと 102 / 246 でテストが長くなる)。
# split=100 は day1 のど真ん中=「直前の日境界(12)〜100」の区間が resume で失われていた。
_WV_ON = {"worldview.enabled": "true", "run.start_tod": "22:00"}


def test_resume_worldview_scan_carry(tmp_path):
    """worldview ON の mid-day resume==straight(C2 応答 / C6 開拓行動の取りこぼし解消)。

    worldview の走査は**日境界にしか走らない**ので、日カウンタ _wv_day を戻すだけでは
    「直前の日境界〜checkpoint」の区間がまるごと落ちる(_wv_pos は logger 由来で 0 に戻る)。
    checkpoint が有界射影(応答 3 つ組列 + 開拓行動数)を運ぶことで塞いだ。
    """
    d = tmp_path / "wv_resumed"
    every = {"observer.checkpoint_every": 100}
    sim1 = Simulation(_cfg("wv_resumed", 100, **every, **_WV_ON), out_dir=d)
    for step in range(100):
        scheduler.run_step(sim1, step)
    carry = _wv.checkpoint_state(sim1)
    assert carry and (carry["pending"] or carry["pioneer"]), \
        "未走査区間が空(テスト前提が崩れた=split/step 数の再調整が要る)"
    checkpoint.save(sim1, 100, d / "checkpoint" / "ckpt-000100.pkl.gz")
    sim1.logger.flush_segment()
    sim2 = Simulation(_cfg("wv_resumed", 160, **every, **_WV_ON), out_dir=d)
    sim2.run(resume_from=d)

    straight = _run_straight(tmp_path, "wv_straight", 160, **_WV_ON)
    for stem in ("l1_events", "l2_metrics", "l3_snapshots"):
        assert _rows(straight, stem) == _rows(d, stem), f"{stem} 不一致(worldview resume)"
    # 日境界(step156)の worldview 世界イベントが実在=検査が空回りしていない
    world = [r for r in _rows(straight, "l1_events")
             if r["kind"] == "worldview" and r["agent_id"] == -1]
    assert world, "worldview の日次スナップショットが出ていない(テスト前提が崩れた)"


def test_worldview_segmented_and_flushed_runs_match_plain(tmp_path):
    """worldview ON でも「分割 straight == plain」が成立する(第98 W4-A で塞いだ**既存欠陥**)。

    走査は日境界にしか走らないので、日の途中で flush_segment が入ると [_wv_pos, flush 点) の
    イベントがバッファから消えて二度と読めなかった(module 冒頭が「退避された分は走査済み扱い」
    と註記していたもの。実測: checkpoint_every=40 で plain と L1 が 3 行食い違っていた)。
    flush の直前に未走査区間を carry へ吸い上げることで、checkpoint 由来の flush でも
    flush_every_steps 由来の flush でも plain と一致する(= resume も同じ土俵に乗る)。
    """
    plain = _run_straight(tmp_path, "wvseg_plain", 160, **_WV_ON)
    seg = _run_straight(tmp_path, "wvseg_ckpt", 160,
                        **{"observer.checkpoint_every": 40}, **_WV_ON)
    flushed = _run_straight(tmp_path, "wvseg_flush", 160,
                            **{"observer.flush_every_steps": 25}, **_WV_ON)
    world = [r for r in _rows(plain, "l1_events")
             if r["kind"] == "worldview" and r["agent_id"] == -1]
    assert world, "worldview の日次スナップショットが出ていない(テスト前提が崩れた)"
    for stem in ("l1_events", "l2_metrics", "l3_snapshots"):
        assert _rows(plain, stem) == _rows(seg, stem), f"{stem} 不一致(checkpoint 分割)"
        assert _rows(plain, stem) == _rows(flushed, stem), f"{stem} 不一致(定期 flush)"


def test_resume_worldview_across_many_flushes(tmp_path):
    """flush が何度も挟まる現実的な設定(every=40)で 120→160 resume == plain。

    resume 元プロセスが 40/80/120 で flush 済み = 走査位置より手前のイベントは**もう存在しない**
    という、本選ラン(checkpoint_every>0 は必ず flush を伴う)そのものの条件で固定する。
    """
    plain = _run_straight(tmp_path, "wvmf_plain", 160, **_WV_ON)
    d = tmp_path / "wvmf_resumed"
    every = {"observer.checkpoint_every": 40}
    sim1 = Simulation(_cfg("wvmf_resumed", 120, **every, **_WV_ON), out_dir=d)
    sim1.run()                                    # 120 step 走り切って中断(3 回 flush 済み)
    sim2 = Simulation(_cfg("wvmf_resumed", 160, **every, **_WV_ON), out_dir=d)
    sim2.run(resume_from=d)
    for stem in ("l1_events", "l2_metrics", "l3_snapshots"):
        assert _rows(plain, stem) == _rows(d, stem), f"{stem} 不一致(flush 多段 resume)"


def _wv_events(sim):
    """C2 応答(event_attend=弱・venture_close=強)+ C6 開拓行動(venture_open)の合成列。"""
    ids = [a.id for a in sim.agents[:3]]
    return [
        Event(step=1, sim_min=10, agent_id=-1, kind="event_attend", x=0.0, y=0.0,
              payload={"host": ids[0]}),                     # 弱+ → ids[0]
        Event(step=2, sim_min=20, agent_id=ids[1], kind="venture_close", x=0.0, y=0.0,
              payload={}),                                    # 強- → ids[1]
        Event(step=3, sim_min=30, agent_id=ids[2], kind="venture_open", x=0.0, y=0.0,
              payload={}),                                    # 開拓行動(C6 の分子)
        Event(step=4, sim_min=40, agent_id=-1, kind="event_attend", x=0.0, y=0.0,
              payload={"host": ids[0]}),                     # 弱+ → ids[0](2 件目)
    ]


def _wv_prime(sim):
    """日境界を「初日ではない」状態に整える(_on_day が snapshot 経路へ入るように)。"""
    sim._wv_day = 0
    sim._wv_pioneer = deque([0], maxlen=7)
    sim._wv_pos = 0


def _wv_fingerprint(sim):
    return [round(getattr(a, "controllability", 0.5), 9) for a in sim.agents]


def test_worldview_carry_equals_uninterrupted_scan(tmp_path):
    """「4 件を一気に走査」==「2 件で checkpoint → 復元 → 残り 2 件」(C2/C6 とも同値)。

    実ランを待たずに、未走査区間の射影が **順序・日次キャップ・開拓行動数まで含めて**
    一気走査と同値であることを直接固定する(mock ランで応答が偶然 0 件でも空回りしない)。
    """
    cfgd = {"worldview.enabled": "true"}
    straight = Simulation(_cfg("wvu_straight", 1, **cfgd), out_dir=tmp_path / "wvu_straight")
    _wv_prime(straight)
    for e in _wv_events(straight):
        straight.logger.log(e)
    _wv._on_day(straight, 1, 5, 1440, straight.worldviewcfg)

    part1 = Simulation(_cfg("wvu_a", 1, **cfgd), out_dir=tmp_path / "wvu_a")
    _wv_prime(part1)
    evs = _wv_events(part1)
    for e in evs[:2]:
        part1.logger.log(e)
    state = _wv.checkpoint_state(part1)
    assert state["pending"] and state["pioneer"] == 0, "前半の射影が想定と違う"

    part2 = Simulation(_cfg("wvu_b", 1, **cfgd), out_dir=tmp_path / "wvu_b")
    _wv_prime(part2)
    _wv.restore_checkpoint(part2, state)
    for e in evs[2:]:
        part2.logger.log(e)
    _wv._on_day(part2, 1, 5, 1440, part2.worldviewcfg)

    assert _wv_fingerprint(part2) == _wv_fingerprint(straight), \
        "carry 経由の C2 が一気走査と食い違う"
    assert list(part2._wv_pioneer) == list(straight._wv_pioneer), \
        "carry 経由の C6 開拓行動数が一気走査と食い違う"

    # 空回り防止: carry を捨てると必ず食い違う(= この検査は実際に何かを守っている)
    naive = Simulation(_cfg("wvu_c", 1, **cfgd), out_dir=tmp_path / "wvu_c")
    _wv_prime(naive)
    for e in evs[2:]:
        naive.logger.log(e)
    _wv._on_day(naive, 1, 5, 1440, naive.worldviewcfg)
    assert _wv_fingerprint(naive) != _wv_fingerprint(straight), \
        "carry 無しでも一致してしまう(テストが空回りしている)"


def test_worldview_daily_cap_survives_carry(tmp_path):
    """日次キャップ(ctrl_daily_weak)が carry を跨いで**通算**で効く(有界化が同値な根拠)。

    弱応答を上限+2 件ぶん流し、前半で checkpoint → 復元 → 後半、としても
    一気走査と同じ controllability に落ちる(= 射影の cap 先取りが結果を変えない)。
    """
    cfgd = {"worldview.enabled": "true", "worldview.ctrl_daily_weak": "2"}

    def _mk(sim):
        host = sim.agents[0].id
        return [Event(step=i, sim_min=10 * i, agent_id=-1, kind="event_attend",
                      x=0.0, y=0.0, payload={"host": host}) for i in range(1, 5)]

    straight = Simulation(_cfg("wvc_straight", 1, **cfgd), out_dir=tmp_path / "wvc_straight")
    _wv_prime(straight)
    for e in _mk(straight):
        straight.logger.log(e)
    _wv._on_day(straight, 1, 5, 1440, straight.worldviewcfg)

    part1 = Simulation(_cfg("wvc_a", 1, **cfgd), out_dir=tmp_path / "wvc_a")
    _wv_prime(part1)
    evs = _mk(part1)
    for e in evs[:3]:                          # 上限 2 を**超えて**から checkpoint する
        part1.logger.log(e)
    state = _wv.checkpoint_state(part1)
    assert len(state["pending"]) == 2, f"射影が上限で切られていない: {state['pending']}"

    part2 = Simulation(_cfg("wvc_b", 1, **cfgd), out_dir=tmp_path / "wvc_b")
    _wv_prime(part2)
    _wv.restore_checkpoint(part2, state)
    for e in evs[3:]:
        part2.logger.log(e)
    _wv._on_day(part2, 1, 5, 1440, part2.worldviewcfg)
    assert _wv_fingerprint(part2) == _wv_fingerprint(straight), \
        "日次キャップが carry を跨いで通算になっていない"


def test_worldview_checkpoint_state_is_pure_read(tmp_path):
    """checkpoint_state は sim も logger も 1 バイトも書き換えない(straight ラン不変の根拠)。"""
    sim = Simulation(_cfg("wvp", 1, **{"worldview.enabled": "true"}),
                     out_dir=tmp_path / "wvp")
    _wv_prime(sim)
    for e in _wv_events(sim):
        sim.logger.log(e)
    before = (sim._wv_pos, len(sim.logger.events),
              list(getattr(sim, "_wv_carry", []) or []),
              int(getattr(sim, "_wv_pioneer_carry", 0)),
              _wv_fingerprint(sim))
    _wv.checkpoint_state(sim)
    _wv.checkpoint_state(sim)                  # 2 回呼んでも同じ(冪等・副作用なし)
    after = (sim._wv_pos, len(sim.logger.events),
             list(getattr(sim, "_wv_carry", []) or []),
             int(getattr(sim, "_wv_pioneer_carry", 0)),
             _wv_fingerprint(sim))
    assert before == after, "checkpoint_state が状態を書き換えている"
    # OFF のランでは None(既定 OFF=挙動不変)
    off = Simulation(_cfg("wvp_off", 1), out_dir=tmp_path / "wvp_off")
    assert _wv.checkpoint_state(off) is None


# ======================================================= レーン D1(第109: 縦煙 +106 行の根治)
# 第108 の全ON縦煙(relations + friend_graph + joint + party + pool を同時 ON)で L1 が
# +106 行(+0.89%)食い違った。原因は 2 系統あり、どちらも第98 の監査の**網の外**だった:
#
#   (系統1)**起動時 1 回の発火体**が resume した 2 つ目のプロセスの __init__ でもう一度
#           走り、step=0 のイベントを同一 payload で二重に積む。第98 W4-A は spark_roster
#           **1 種だけ**を名指しで塞いでいたので、friend_graph_built(+1)と
#           party の joint_activity(+97)は素通りしていた。
#           → checkpoint が「__init__ が出し終えた時点のバッファ長」を運び、load 側が
#             その本数を先頭から捨てる(種を名指ししない = 将来の発火体も自動で守る)。
#
#   (系統2)**「1 日 1 件」/「走査済み」ガードの未保存**(第80 W2 と同型)。
#           signal_summary(+1)・viral_cascade(+3)・misinfo(+2)+その副作用の
#           state_update(+1)。特に情報環境の watermark は **sim.net の投稿 id 空間**由来
#           なので「load で 0 に戻す」流儀が通用せず、既に処理済みの投稿を全部走査し直して
#           reshares を二重加算していた(観測だけでなく世界状態が壊れる)。
def test_resume_startup_segment_recorded_once(tmp_path):
    """★系統1: friend_graph ON の resume で friend_graph_built が **1 件のまま**。

    friends.build_friend_graph は Simulation.__init__ から呼ばれ step=0 のイベントを 1 件
    出す(spark.apply と同型)。従来は resume した 2 つ目のプロセスが同じ 1 件を出し直し、
    結合後の L1 に 2 件並んでいた。全層一致も併せて固定する。
    """
    ov = {"relations.enabled": "true", "friend_graph.enabled": "true"}
    straight = _run_straight(tmp_path, "fg_straight", 40, **ov)
    resumed = _run_resume(tmp_path, "fg_resumed", 20, 40, **ov)
    n_st = sum(1 for r in _rows(straight, "l1_events") if r["kind"] == "friend_graph_built")
    n_rs = sum(1 for r in _rows(resumed, "l1_events") if r["kind"] == "friend_graph_built")
    assert n_st == 1, "straight で friend_graph_built が 1 件でない(テスト前提が崩れた)"
    assert n_rs == 1, f"resume で friend_graph_built が二重記録されている: {n_rs} 件"
    for stem in ("l1_events", "l2_metrics", "l3_snapshots"):
        assert _rows(straight, stem) == _rows(resumed, stem), f"{stem} 不一致(friend_graph)"


def test_init_event_mark_counts_the_startup_segment(tmp_path):
    """`_init_event_mark` が「__init__ が出したイベント本数」そのものである(空回り防止)。

    起動時セグメントが**実在する**構成(friend_graph ON)で > 0 になり、バッファの中身と
    本数が一致すること・OFF 相当の素の構成では 0 になることを直接押さえる。ここが空だと
    上のテストは「たまたま一致しているだけ」になりうる。
    """
    on = Simulation(_cfg("mark_on", 1, **{"relations.enabled": "true",
                                          "friend_graph.enabled": "true"}),
                    out_dir=tmp_path / "mark_on")
    assert on._init_event_mark == len(on.logger.events) > 0
    assert any(e.kind == "friend_graph_built" for e in on.logger.events)
    off = Simulation(_cfg("mark_off", 1), out_dir=tmp_path / "mark_off")
    assert off._init_event_mark == len(off.logger.events) == 0


def test_resume_infoenv_watermark_no_rescan(tmp_path):
    """★系統2: info_env ON の resume で走査済み投稿を**引き直さない**(全層一致)。

    `_infoenv_watermark` は他の watermark と違い **sim.net の絶対投稿 id** 由来なので、
    「load で 0 に戻す」流儀では既処理の投稿を全部走査し直す。`_viral_done`(1 投稿 1 回の
    加重)も同時に空へ戻るため、viral_cascade を再発火して reshares を二重加算し、
    misinfo は別 step キーで引き直されて炎上が捏造されていた。

    第109 縦煙の viral_cascade +3 / misinfo +2 / state_update +1 はこの 1 本の穴で説明できる
    (state_update は `flame_grievance>0` のときの炎上 → 発信者の不満更新 = misinfo の副作用。
    既定 0.0 なので finals_observe が 0.02 を置いている構成でしか出ない → ここでも置く)。
    """
    ov = {"info_env.enabled": "true", "info_env.influence.enabled": "true",
          "info_env.misinfo.enabled": "true", "info_env.misinfo.rate": "0.9",
          "info_env.misinfo.flame_grievance": "0.02",     # finals_observe と同値
          "net.reshare_prob": "0.9"}
    straight = _run_straight(tmp_path, "ie_straight", 40, **ov)
    resumed = _run_resume(tmp_path, "ie_resumed", 20, 40, **ov)
    rows_st = _rows(straight, "l1_events")
    assert sum(1 for r in rows_st if r["kind"] == "misinfo") > 0, \
        "misinfo が 1 件も出ていない(テスト前提が崩れた=rate の再調整が要る)"
    assert sum(1 for r in rows_st if r["kind"] == "viral_cascade") > 0, \
        "viral_cascade が 1 件も出ていない(テスト前提が崩れた=reshare_prob の再調整)"
    for stem in ("l1_events", "l2_metrics", "l3_snapshots"):
        assert _rows(straight, stem) == _rows(resumed, stem), f"{stem} 不一致(info_env)"


_SIGNAL_SPEC = {"crossing_id": 291758776, "cycle_s": 140.0, "green_s": 37.0,
                "flash_s": 10.0, "offset_s": 0.0}


def _sim_with_signal(tmp_path, name: str):
    """装置層 ON + 信号 1 台を名簿へ直に据えた sim(zones テーブル無しで機構だけ試す)。

    実ランで信号装置が生えるのは `physics.zones` の signal 宣言経由(= v8 地図 + zones
    テーブルが要る)。ここで検査したいのは**日ガードが checkpoint を跨ぐか**だけなので、
    名簿へ 1 台足して機構を直接呼ぶ(第109 縦煙の実測は finals 相当の統合テスト側で押さえる)。
    """
    from society import devices as D
    sim = Simulation(_cfg(name, 1, **{"world.devices.enabled": "true",
                                      "world.devices.signal_events": "true"}),
                     out_dir=tmp_path / name)
    reg = D.registry_of(sim)
    if reg.get(D.signal_device_id(_SIGNAL_SPEC["crossing_id"])) is None:
        reg.add(D.SignalDevice(D.signal_device_id(_SIGNAL_SPEC["crossing_id"]),
                               _SIGNAL_SPEC))
    return sim, reg


def _n_signal(sim) -> int:
    return sum(1 for e in sim.logger.events if e.kind == "signal_summary")


def test_resume_signal_summary_day_guard_survives_checkpoint(tmp_path):
    """★系統2: `_dev_signal_day`(1 交差点 1 日 1 件)が checkpoint を跨いで効く。

    このガードが checkpoint に無かったため、resume 直後にその日の信号要約をもう一度
    出していた(第109 縦煙の signal_summary +1)。負の対照(ガードを捨てた sim)が
    必ず二重に出すことも併せて固定する(= この検査が実際に何かを守っている証拠)。
    """
    from society import devices as D

    src, reg_src = _sim_with_signal(tmp_path, "sig_src")
    D._signal_summaries(src, reg_src, 3, 30)
    assert _n_signal(src) == 1, "信号要約が 1 件出ていない(テスト前提が崩れた)"
    assert src._dev_signal_day, "_dev_signal_day が立っていない"
    D._signal_summaries(src, reg_src, 4, 40)                 # 同じ日 = もう出ない
    assert _n_signal(src) == 1

    p = checkpoint.save(src, 5, tmp_path / "sig_src" / "checkpoint" / "ckpt-000005.pkl.gz")

    dst, reg_dst = _sim_with_signal(tmp_path, "sig_dst")
    checkpoint.load(dst, p)
    D._signal_summaries(dst, reg_dst, 5, 50)                 # 同じ暦日 = 1 件も出さない
    assert _n_signal(dst) == 0, "resume 後に signal_summary が二重記録されている"
    D._signal_summaries(dst, reg_dst, 200, 1500)             # 翌日 = ちゃんと出る
    assert _n_signal(dst) == 1, "翌日の信号要約まで抑止されている(過剰抑止)"

    # 負の対照: ガードを捨てると必ず二重に出る(検査が空回りしていない)
    naive, reg_naive = _sim_with_signal(tmp_path, "sig_naive")
    checkpoint.load(naive, p)
    naive._dev_signal_day = {}
    D._signal_summaries(naive, reg_naive, 5, 50)
    assert _n_signal(naive) == 1


def test_resume_family_dinner_logged_once(tmp_path):
    """★系統2: 夕食帯(18:00-21:00)の途中で resume しても family_dinner が二重に出ない。

    `_dinner_logged`((household_id, day) の記録済み印)が checkpoint に無く、帯の途中で
    resume すると同じ世帯の joint_activity{family_dinner} がもう 1 件出た。start_tod を
    17:00 に置き、split=12(=18:00 ちょうど。帯に入った直後)→ 40 で跨がせる。
    """
    ov = {"household.enabled": "true", "household.family_dinner.enabled": "true",
          "household.family_dinner.prob": "1.0", "run.start_tod": "17:00"}
    straight = _run_straight(tmp_path, "fd_straight", 40, **ov)
    resumed = _run_resume(tmp_path, "fd_resumed", 12, 40, **ov)
    dinners = [r for r in _rows(straight, "l1_events")
               if r["kind"] == "joint_activity" and "family_dinner" in str(r["payload"])]
    assert dinners, "family_dinner が 1 件も出ていない(テスト前提が崩れた=帯/split の再調整)"
    for stem in ("l1_events", "l2_metrics", "l3_snapshots"):
        assert _rows(straight, stem) == _rows(resumed, stem), f"{stem} 不一致(family_dinner)"


_D1_ROUNDTRIP = {                          # レーン D1 で checkpoint に足した状態の往復表
    "_init_event_mark": 5,
    "_dev_signal_day": {"signal:1": 3, "signal:2": 4},
    "_infoenv_watermark": 41,
    "_viral_done": {7, 11, 13},
    "_dinner_logged": {("hh-1", 2), ("hh-2", 3)},
}


def test_checkpoint_roundtrips_the_d1_guards(tmp_path):
    """レーン D1 で足した 5 つの状態が checkpoint 往復で **1 件残らず**戻る。"""
    src = _fresh(tmp_path, "d1_src")
    for attr, val in _D1_ROUNDTRIP.items():
        setattr(src, attr, val)
    p = checkpoint.save(src, 9, tmp_path / "d1_src" / "checkpoint" / "ckpt-000009.pkl.gz")

    dst = _fresh(tmp_path, "d1_dst")
    # 起動時セグメントの相当分をバッファへ積んでおく(load が先頭 5 件を捨てるはず)
    for i in range(5):
        dst.logger.log(Event(step=0, sim_min=0, agent_id=-1, kind="friend_graph_built",
                             x=0.0, y=0.0, payload={"n_edges": i, "mean_degree": 0.0}))
    assert checkpoint.load(dst, p) == 9
    assert dst.logger.events == [], "起動時セグメントが捨てられていない(二重記録が残る)"
    for attr, val in _D1_ROUNDTRIP.items():
        if attr == "_init_event_mark":     # これは「捨てる本数」で sim へは戻さない
            continue
        assert getattr(dst, attr, None) == val, f"{attr} が checkpoint で復元されていない"


def test_d1_keys_absent_in_old_checkpoint_keep_legacy_behaviour(tmp_path):
    """D1 の新キーを 1 つも持たない旧 checkpoint でも落ちず、従来の既定へ落ちる。"""
    import gzip
    import pickle

    src = _fresh(tmp_path, "d1_old_src")
    src._infoenv_watermark = 41
    src._dev_signal_day = {"signal:1": 3}
    p = checkpoint.save(src, 4, tmp_path / "d1_old_src" / "checkpoint" / "ckpt-000004.pkl.gz")
    with gzip.open(p, "rb") as f:
        blob = pickle.loads(f.read())
    for k in ("init_events", "dev_signal_day", "infoenv_watermark", "viral_done",
              "dinner_logged"):
        blob["runtime"].pop(k, None)
    with gzip.open(p, "wb") as f:
        f.write(pickle.dumps(blob, protocol=pickle.HIGHEST_PROTOCOL))

    dst = _fresh(tmp_path, "d1_old_dst")
    dst.logger.log(Event(step=0, sim_min=0, agent_id=-1, kind="friend_graph_built",
                         x=0.0, y=0.0, payload={"n_edges": 1, "mean_degree": 0.0}))
    assert checkpoint.load(dst, p) == 4
    assert len(dst.logger.events) == 1          # 旧 ckpt = 捨てない(従来挙動そのまま)
    assert dst._infoenv_watermark == 0
    assert getattr(dst, "_dev_signal_day", None) is None
