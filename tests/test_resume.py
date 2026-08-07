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

from society.config import load_config
from society.engine import checkpoint, scheduler
from society.engine.simulation import Simulation


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
                "inner_life_init"]
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
