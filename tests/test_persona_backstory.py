"""ペルソナ過去情報(サイドカー)のエンジン統合 = レーンB のテスト。

正典: src/society/world/backstory.py の docstring。

検証(受入基準):
  - OFF(既定 pool.backstory_dir="" / prompts.backstory_enabled=false)は
    **1 バイトも読まず属性を 1 つも生やさない** = L1 バイト一致(600 体 24 step mock A/B)。
  - ON(dir だけ): agent.backstory が付くが**プロンプトは 1 バイトも変わらない**
    (載せる/見せるが別トグル = A/B の切り分けが立つ)。
  - ON(dir + prompts): プロンプトの**自己紹介行の直後**に 1 節が入る。
    発話・朝の計画・夜の内省・recall の全 purpose に同一に効く(節は 1 つ = 重複なし)。
  - サイドカーに無い pid は**無音で骨格のみ**(例外にしない)。
  - プール record 本体・friends.cache_key(roster digest)に影響しない。
  - 日次ローテーションで入れ替わった個体にも正しく付く。
  - resume == straight(pool ON + backstory ON)。
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "src"))

import build_persona_pool as bpp                        # noqa: E402
from society import registry as R                       # noqa: E402
from society.config import load_config                  # noqa: E402
from society.cognition import deliberate                # noqa: E402
from society.cognition import prompt_p1 as P1           # noqa: E402
from society.engine.simulation import Simulation        # noqa: E402
from society.world import backstory as bs_mod           # noqa: E402
from society.world import pool as pool_mod              # noqa: E402


# ============================================================ 小さなテスト用プール
@pytest.fixture(scope="module")
def small_pool(tmp_path_factory):
    """P5 生成器で千体級の小プールを tmp に生成(実プール 736MB は触らない)。"""
    out = tmp_path_factory.mktemp("bs_pool")
    orgs = json.loads((_ROOT / "data" / "organizations_shibuya_wide11k.json")
                      .read_text(encoding="utf-8"))
    pop = json.loads((_ROOT / "data" / "shibuya_population.json")
                     .read_text(encoding="utf-8"))
    bpp.build_pool(out, seed=42, fraction=0.001, orgs=orgs, pop=pop,
                   total_target=1_000_000)
    return out


def _pool_pids(pool_dir) -> list[tuple[str, str]]:
    """プールの (pid, layer) を全部読む(生成レーンの出力形を模すため)。"""
    meta = json.loads((Path(pool_dir) / "meta.json").read_text(encoding="utf-8"))
    out: list[tuple[str, str]] = []
    for sh in meta["shards"]:
        for raw in (Path(pool_dir) / sh["file"]).read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if raw:
                rec = json.loads(raw)
                out.append((str(rec["id"]), str(rec.get("layer", ""))))
    return out


def _text_for(pid: str) -> str:
    return f"__bs__{pid}__ 子どものころから同じ街に住んでいる。"


@pytest.fixture(scope="module")
def sidecar(small_pool, tmp_path_factory):
    """層別 jsonl.gz のサイドカー(生成レーンが吐く形。1 行 = {pid, backstory})。"""
    out = tmp_path_factory.mktemp("bs_side")
    per_layer: dict[str, list[str]] = {}
    for pid, layer in _pool_pids(small_pool):
        per_layer.setdefault(layer or "L0", []).append(pid)
    for layer, pids in per_layer.items():
        with gzip.open(out / f"{layer}.jsonl.gz", "wt", encoding="utf-8") as f:
            for pid in pids:
                f.write(json.dumps({"pid": pid, "backstory": _text_for(pid),
                                    "model": "test", "n_tokens": 12},
                                   ensure_ascii=False) + "\n")
    return out


# =========================================================================== #
# (1) BackstoryStore 単体(遅延読み・欠損許容・壊れた行)
# =========================================================================== #
def _write(path: Path, rows, gz=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    if gz:
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(body)
    else:
        path.write_text(body, encoding="utf-8")


def test_store_of_is_none_when_unset(tmp_path):
    """既定("" / キー無し / None)は None = 完全 no-op。"""
    assert bs_mod.store_of({}) is None
    assert bs_mod.store_of({"backstory_dir": ""}) is None
    assert bs_mod.store_of({"backstory_dir": "   "}) is None
    assert bs_mod.store_of(None) is None


def test_store_reads_layered_gzip(tmp_path):
    _write(tmp_path / "L1.jsonl.gz", [{"pid": "L1_0", "backstory": "あ"}], gz=True)
    st = bs_mod.BackstoryStore(tmp_path)
    assert st.get("L1_0", "L1") == "あ"
    assert st.stats()["n_hit"] == 1


def test_store_reads_plain_jsonl_and_subdir(tmp_path):
    """非圧縮・サブディレクトリ(<root>/L2/part-0000.jsonl)も受ける。"""
    _write(tmp_path / "L2" / "part-0000.jsonl", [{"pid": "L2_0", "backstory": "い"}])
    st = bs_mod.BackstoryStore(tmp_path)
    assert st.get("L2_0", "L2") == "い"


def test_store_flat_root_fallback(tmp_path):
    """層別に割られていないサイドカーはルート一括を 1 度だけ読む。"""
    _write(tmp_path / "all.jsonl.gz", [{"pid": "L4_9", "backstory": "う"}], gz=True)
    st = bs_mod.BackstoryStore(tmp_path)
    assert st.get("L4_9", "L4") == "う"
    assert "" in st.stats()["layers"]             # ルート一括を走査した印


def test_store_is_lazy_per_layer(tmp_path):
    """要求された層だけ読む(他の層はメモリに載らない)。"""
    _write(tmp_path / "L1.jsonl.gz", [{"pid": "L1_0", "backstory": "あ"}], gz=True)
    _write(tmp_path / "L4.jsonl.gz", [{"pid": f"L4_{i}", "backstory": "え"}
                                      for i in range(5)], gz=True)
    st = bs_mod.BackstoryStore(tmp_path)
    st.get("L1_0", "L1")
    assert len(st) == 1                            # L4 の 5 件はまだ読んでいない
    st.get("L4_3", "L4")
    assert len(st) == 6


def test_store_missing_pid_is_silent(tmp_path):
    """サイドカーに無い pid は "" を返す(例外にしない)。件数は stats に出る。"""
    _write(tmp_path / "L1.jsonl.gz", [{"pid": "L1_0", "backstory": "あ"}], gz=True)
    st = bs_mod.BackstoryStore(tmp_path)
    assert st.get("L1_404", "L1") == ""
    assert st.get("", "L1") == ""
    assert st.stats()["n_miss"] == 1               # pid 空欄は数えない


def test_store_missing_dir_is_silent(tmp_path):
    """存在しないディレクトリでも落ちない(全員が骨格のみ)。"""
    st = bs_mod.store_of({"backstory_dir": str(tmp_path / "nope")})
    assert st is not None and len(st) == 0
    assert st.get("L1_0", "L1") == ""


def test_store_skips_broken_lines(tmp_path):
    """壊れた行・空行・欄欠けは黙って飛ばす(生成途中でも落とさない)。"""
    path = tmp_path / "L1.jsonl"
    path.write_text('{"pid": "a", "backstory": "A"}\n'
                    "\n"
                    "{ this is not json\n"
                    '{"pid": "b"}\n'
                    '{"backstory": "no pid"}\n'
                    '{"pid": "c", "backstory": "   "}\n'
                    '{"pid": "d", "backstory": "D"}\n', encoding="utf-8")
    st = bs_mod.BackstoryStore(tmp_path)
    assert st.get("a", "L1") == "A"
    assert st.get("d", "L1") == "D"
    assert st.get("b", "L1") == "" and st.get("c", "L1") == ""
    assert len(st) == 2


def test_store_missing_layer_does_not_reread(tmp_path):
    """生成途中(L4 がまだ無い)でも落ちず、既読ファイルを二度読みしない。"""
    _write(tmp_path / "L1.jsonl.gz", [{"pid": "L1_0", "backstory": "あ"}], gz=True)
    st = bs_mod.BackstoryStore(tmp_path)
    assert st.get("L1_0", "L1") == "あ"
    assert st.get("L4_0", "L4") == ""           # 層ファイルが無い = 無音で骨格のみ
    assert len(st) == 1                          # L1 を読み直して重複していない
    assert st.get("L1_0", "L1") == "あ"          # 既読の層はそのまま引ける
    # ★退路(ルート全体)を 1 度通ったあとでも、あとから来る**実在する層**は読める
    _write(tmp_path / "L3.jsonl.gz", [{"pid": "L3_0", "backstory": "お"}], gz=True)
    assert st.get("L3_0", "L3") == "お"


def test_store_ignores_failed_sidecar_rows(tmp_path):
    """生成レーンが並べて書く <layer>.failed.jsonl(pid + reason)は本文を持たない。"""
    _write(tmp_path / "L1.jsonl.gz", [{"pid": "L1_0", "backstory": "あ"}], gz=True)
    _write(tmp_path / "L1.failed.jsonl", [{"pid": "L1_1", "reason": "too_short"}])
    st = bs_mod.BackstoryStore(tmp_path)
    assert st.get("L1_0", "L1") == "あ"
    assert st.get("L1_1", "L1") == ""
    assert len(st) == 1


def test_store_relative_dir_resolves_under_repo_root(tmp_path):
    """相対パスはリポジトリルート基準(pool.dir と同じ規約)。"""
    st = bs_mod.store_of({"backstory_dir": "data/persona_backstory_v2"},
                         repo_root=tmp_path)
    assert st is not None
    assert st.root == tmp_path / "data" / "persona_backstory_v2"


# =========================================================================== #
# (2) プロンプト挿入(build_prompt の実出力を文字列検査)
# =========================================================================== #
class _Mem:
    day_summaries: list = []

    def recent(self, n):
        return []

    def retrieve(self, *a, **k):
        return []

    def query_ex(self, *a, **k):
        class _R:
            hits: list = []
            failed = False
            cue = "手掛かり"
        return _R()

    def relation_line(self, ids):
        return None


class _Agent:
    id = 0
    name = "見本"
    persona = "あなたは見本、40歳の店長(男性)。渋谷の街で暮らしている。"
    activity = ""
    states: dict = {}
    beliefs: list = []
    adopted: set = set()
    said: list = []
    money = 1000
    mem = _Mem()


def _agent(backstory=None, prompt=False):
    a = _Agent()
    if backstory is not None:
        a.backstory = backstory
    if prompt:
        a.backstory_prompt = True
    return a


def _deliberate(agent, p1=None):
    return deliberate.build_prompt(agent, place_name="見本通り", surprise="solo",
                                   nearby_names=[], sim_min=600, step=3,
                                   city_name="見本町", p1_purpose=p1)


def test_prompt_off_has_no_section():
    """属性が無ければ 1 行も足さない(既定 = 現行とバイト一致)。"""
    base = _deliberate(_agent())
    assert deliberate.BACKSTORY_PREFIX not in base
    # 「載せた」だけ(prompts.backstory_enabled=false)でも 1 バイトも変わらない
    loaded = _deliberate(_agent(backstory="過去のはなし"))
    assert loaded == base


def test_prompt_on_inserts_section():
    on = _deliberate(_agent(backstory="過去のはなし", prompt=True))
    assert deliberate.BACKSTORY_PREFIX + "過去のはなし" in on


def test_prompt_section_sits_right_after_persona():
    """挿入位置 = **自己紹介行の直後**(1 節・重複なし)。"""
    lines = _deliberate(_agent(backstory="過去のはなし", prompt=True)).split("\n")
    idx = lines.index(_Agent.persona)
    assert lines[idx + 1] == deliberate.BACKSTORY_PREFIX + "過去のはなし"
    assert sum(1 for ln in lines
               if ln.startswith(deliberate.BACKSTORY_PREFIX)) == 1


def test_prompt_section_precedes_p1_discipline():
    """V-P1 ON でも「ペルソナ → 過去情報 → 規律 3 行」の順で 1 度だけ入る。"""
    lines = _deliberate(_agent(backstory="過去のはなし", prompt=True),
                        p1="deliberate").split("\n")
    idx = lines.index(_Agent.persona)
    assert lines[idx + 1] == deliberate.BACKSTORY_PREFIX + "過去のはなし"
    assert lines[idx + 2:idx + 2 + len(P1.DISCIPLINE)] == list(P1.DISCIPLINE)


def test_prompt_empty_backstory_adds_nothing():
    """本文が空なら足さない(合図だけ立っていても無風)。"""
    assert _deliberate(_agent(backstory="", prompt=True)) == _deliberate(_agent())


def test_prompt_section_applies_to_all_purposes():
    """発話・朝の計画・夜の内省・recall の**全 purpose**に同一に効く(共有 build_prompt)。"""
    from society.cognition import planning as _planning
    from society.cognition import reflection as _reflection

    agent = _agent(backstory="過去のはなし", prompt=True)
    dp_cfg = {"min_blocks": 4, "max_blocks": 8, "max_conting": 3}
    prompts = {
        "deliberate": _deliberate(agent),
        "plan": _planning.build_plan_prompt(agent, place_name="見本通り",
                                            sim_min=510, step=3,
                                            city_name="見本町", day_plan=dp_cfg),
        "reflect": _reflection.build_reflect_request(
            agent, step=3, sim_min=1380, place_name="自宅", date_line=None,
            weather_line=None, reflect_cfg=None, reflect_variety=False,
            interstitial_digest=None, interstitial=False, city_name="見本町",
            max_tokens=896, think=False, recalled=[], recall_fail=None)["prompt"],
        "recall": _reflection.build_recall_request(
            agent, step=3, place_name="自宅", city_name="見本町")["prompt"],
    }
    for name, text in prompts.items():
        assert deliberate.BACKSTORY_PREFIX + "過去のはなし" in text, name
        assert text.count(deliberate.BACKSTORY_PREFIX) == 1, name


def test_prompt_prefix_is_not_a_placebo_section():
    """接頭辞は ablate.SECTIONS のどれとも衝突しない(**自分由来** = ペルソナと同じ扱い)。"""
    from society import ablate as _ablate

    prefixes = {s.prefix for s in _ablate.SECTIONS}
    assert deliberate.BACKSTORY_PREFIX not in prefixes
    for pre in prefixes:                            # 前方一致の取り違えも起きない
        assert not deliberate.BACKSTORY_PREFIX.startswith(pre)
        assert not pre.startswith(deliberate.BACKSTORY_PREFIX)


# =========================================================================== #
# (3) エンジン統合(pool ローダ)
# =========================================================================== #
def _cfg(name, pool_dir, n_steps=1, cap=400, **ov):
    dot = ["run.seed=42", "run.n_agents=10", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock",
           "pool.enabled=true", f"pool.dir={pool_dir}", f"pool.present_cap={cap}"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def test_engine_off_grows_no_attribute(small_pool, tmp_path):
    """既定 OFF: store を作らず、属性を 1 つも生やさない。"""
    sim = Simulation(_cfg("bs_off", small_pool), out_dir=tmp_path / "off")
    assert sim._backstory is None
    assert sim._backstory_prompt is False
    assert sim.agents
    for a in sim.agents:                    # __dict__ のキー集合が現行と完全一致
        assert "backstory" not in a.__dict__
        assert "backstory_prompt" not in a.__dict__


def test_engine_on_attaches_backstory(small_pool, sidecar, tmp_path):
    """dir を設定すると day0 の在場者全員に本文が付く(prompt トグルは別)。"""
    sim = Simulation(_cfg("bs_on", small_pool, **{"pool.backstory_dir": sidecar}),
                     out_dir=tmp_path / "on")
    assert sim._backstory is not None
    assert sim.agents
    for a in sim.agents:
        assert a.backstory == _text_for(a.pool_pid)
        assert "backstory_prompt" not in a.__dict__        # 見せるトグルは OFF
    assert sim._backstory.stats()["n_hit"] == len(sim.agents)
    assert sim._backstory.stats()["n_miss"] == 0


def test_engine_prompt_toggle_sets_flag(small_pool, sidecar, tmp_path):
    sim = Simulation(_cfg("bs_pon", small_pool,
                          **{"pool.backstory_dir": sidecar,
                             "prompts.backstory_enabled": "true"}),
                     out_dir=tmp_path / "pon")
    assert sim._backstory_prompt is True
    assert all(getattr(a, "backstory_prompt", False) for a in sim.agents)
    # 実プロンプトにも節が入る(scheduler を通さない直接検査)
    a = sim.agents[0]
    text = deliberate.build_prompt(a, place_name="X", surprise="solo",
                                   nearby_names=[], sim_min=600, step=0)
    assert deliberate.BACKSTORY_PREFIX + a.backstory in text


def test_engine_partial_sidecar_is_silent(small_pool, tmp_path):
    """欠損 pid は無音で骨格のみ(エラーにしない)。件数は store のカウンタに出る。"""
    side = tmp_path / "partial"
    side.mkdir()
    pids = [pid for pid, _ in _pool_pids(small_pool)]
    covered = set(pids[:5])
    _write(side / "all.jsonl.gz",
           [{"pid": p, "backstory": _text_for(p)} for p in sorted(covered)], gz=True)
    sim = Simulation(_cfg("bs_part", small_pool, **{"pool.backstory_dir": side}),
                     out_dir=tmp_path / "part")
    got = {a.pool_pid for a in sim.agents if hasattr(a, "backstory")}
    assert got == covered & {a.pool_pid for a in sim.agents}
    st = sim._backstory.stats()
    assert st["n_hit"] + st["n_miss"] == len(sim.agents)
    assert st["n_miss"] > 0


def test_engine_does_not_mutate_pool_record(small_pool, sidecar, tmp_path):
    """プール record 本体には 1 バイトも書かない(サイドカーである所以)。"""
    sim = Simulation(_cfg("bs_rec", small_pool, **{"pool.backstory_dir": sidecar}),
                     out_dir=tmp_path / "rec")
    pid = sim.agents[0].pool_pid
    rec = sim._pool.get(pid)
    assert "backstory" not in rec
    assert set(rec) == set(pool_mod.PoolStore(small_pool).get(pid))


def test_friend_cache_key_unchanged_by_backstory(small_pool, sidecar, tmp_path):
    """friends.cache_key(roster digest)に backstory を混ぜない = キャッシュ不変。"""
    from society import friends as _friends
    from society import relations as _relations

    sim = Simulation(_cfg("bs_fc", small_pool, **{"pool.backstory_dir": sidecar}),
                     out_dir=tmp_path / "fc")
    residents = sorted((a for a in sim.agents if not a.visitor), key=lambda a: a.id)
    assert residents
    cfg = {"enabled": True, "margin": 0.05}
    rc = _relations.DEFAULTS
    thr = {1: float(rc["tier_acquaintance"]), 2: float(rc["tier_friend"]),
           3: float(rc["tier_close"])}
    with_bs = _friends.cache_key(sim, residents, cfg, thr)
    for a in residents:                              # 過去情報を剥がしても同じキー
        if hasattr(a, "backstory"):
            del a.backstory
    without = _friends.cache_key(sim, residents, cfg, thr)
    assert with_bs == without


def test_engine_rotation_attaches_to_new_entrants(small_pool, sidecar, tmp_path):
    """日境界のローテーションで入場した個体にも正しく付く(再実体化の全経路を通る)。"""
    sim = Simulation(_cfg("bs_rot", small_pool, n_steps=210,
                          **{"pool.backstory_dir": sidecar,
                             "prompts.backstory_enabled": "true"}),
                     out_dir=tmp_path / "rot")
    day0 = {a.pool_pid for a in sim.agents}
    sim.run()
    entrants = [a for a in sim.agents if a.pool_pid not in day0]
    assert entrants, "日境界の入場者が 1 人も居ない(検査が空回り)"
    for a in sim.agents:
        assert a.backstory == _text_for(a.pool_pid)
        assert getattr(a, "backstory_prompt", False)


def test_engine_no_pool_no_backstory(tmp_path):
    """pool OFF のランでは prompts.backstory_enabled だけ立てても完全 no-op。"""
    dot = ["run.seed=42", "run.n_agents=8", "run.n_steps=4", "run.name=bs_nopool",
           "model.backend=mock", "prompts.backstory_enabled=true"]
    sim = Simulation(load_config(dot), out_dir=tmp_path / "nopool")
    sim.run()
    assert sim._pool is None and sim._backstory is None
    assert not [a for a in sim.agents if hasattr(a, "backstory")]


# =========================================================================== #
# (4) R1 検収: OFF バイト一致 / resume == straight
# =========================================================================== #
def _rows(out_dir):
    import pyarrow.parquet as pq
    return pq.read_table(Path(out_dir) / "l1_events.parquet").to_pylist()


def test_off_l1_byte_identical_600_agents(small_pool, sidecar, tmp_path):
    """600 体 24 step mock: **サイドカーを読んでも**プロンプト OFF なら L1 バイト一致。

    ゴールデン同型の A/B。「載せる」だけでは世界が 1 ビットも動かないことの機械固定で、
    ここが割れたら backstory がどこかで乱数・発火・呼数に触れている。
    """
    a_dir = tmp_path / "ab_off"
    Simulation(_cfg("bs_ab", small_pool, n_steps=24, cap=600),
               out_dir=a_dir).run()
    b_dir = tmp_path / "ab_load"
    Simulation(_cfg("bs_ab", small_pool, n_steps=24, cap=600,
                    **{"pool.backstory_dir": sidecar}), out_dir=b_dir).run()
    assert _rows(a_dir) == _rows(b_dir)


def test_resume_byte_matches_straight_with_backstory(small_pool, sidecar, tmp_path):
    """pool ON + backstory ON でも「一気 vs 中断→resume」の l1_events が完全一致。"""
    from society.engine import checkpoint, scheduler

    ov = {"pool.backstory_dir": sidecar, "prompts.backstory_enabled": "true"}
    st = tmp_path / "bs_st"
    Simulation(_cfg("bs_rst", small_pool, n_steps=300, **ov), out_dir=st).run()

    rs = tmp_path / "bs_rs"
    s1 = Simulation(_cfg("bs_rrs", small_pool, n_steps=150,
                         **{**ov, "observer.checkpoint_every": 150}), out_dir=rs)
    for step in range(150):
        scheduler.run_step(s1, step)
    checkpoint.save(s1, 150, rs / "checkpoint" / "ckpt-000150.pkl.gz")
    s1._save_pool_sidecar(150)
    s1.logger.flush_segment()
    s2 = Simulation(_cfg("bs_rrs", small_pool, n_steps=300,
                         **{**ov, "observer.checkpoint_every": 150}), out_dir=rs)
    s2.run(resume_from=rs)
    assert _rows(st) == _rows(rs), "backstory ON の resume が straight と byte 不一致"
    # resume 後も全員が本文を持っている(checkpoint の agents pickle + 再入場の付与)
    for a in s2.agents:
        assert a.backstory == _text_for(a.pool_pid)


# =========================================================================== #
# (5) レジストリ宣言
# =========================================================================== #
def test_registry_declares_both_keys():
    ids = {f.id for f in R.FEATURES}
    assert "pool.backstory_dir" in ids
    assert "prompts.backstory_enabled" in ids
    by_id = {f.id: f for f in R.FEATURES}
    assert by_id["pool.backstory_dir"].off_value == ""
    for fid in ("pool.backstory_dir", "prompts.backstory_enabled"):
        assert by_id[fid].affects_k is False        # LLM 呼数は 1 つも動かない
        assert by_id[fid].repro_tier == "strict"


def test_shipped_config_has_the_two_defaults():
    cfg = load_config()
    assert cfg.pool.backstory_dir == ""
    assert cfg.prompts.backstory_enabled is False
    assert R.undeclared_toggles(cfg) == []
