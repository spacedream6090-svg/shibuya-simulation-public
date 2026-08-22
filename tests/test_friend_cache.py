"""第151 レーン2: 友人グラフのディスクキャッシュ(`world.friends_cache_dir`)。

正典: docs/plans/index-cell-and-friend-cache-plan.md §1(リサーチ)/ §2-2(設計)/ §4(検収)。
実装: src/society/friends.py。

何を直したか: 250k の py-spy で **init の 68.4% が friends.build_friend_graph**(40-60 分/
起動)。生成は blake2b の安定ハッシュだけで決まる**完全な決定論**(乱数 stream ゼロ・
run.seed 非依存)なのに、毎回同じグラフを作り直していた。疫学・社会 ABM の標準構成
(FRED / Epihiper / synthpops)に倣い、決定結果を成果物としてキャッシュする。

守るもの(検収基準の順)
  (1) ★**既定 "" = OFF = 現行と完全同一**(L1 バイト一致・ファイルを 1 つも作らない)。
  (2) ★初回 build + 保存 → 2 回目ロードで **relations 台帳・tier・closeness・挿入順・
      friend_graph_built が完全一致**(in-memory の結果がバイト同一)。
  (3) キー不一致(設定 / 名簿 / tier 閾値 / friends.py の内容)で再構築し、結果は
      「その設定でフル構築したもの」と一致する。
  (4) 破損・切り詰め・キー衝突・読み書き不能で**例外を上げず**黙って再構築する。
  (5) 生成規則そのものは決定論(乱数 stream を 1 本も引かない)= キャッシュの正当性の前提。
  (6) 契約列挙ピン(conf 既定・finals 値・registry 宣言・凍結ファイル不触)。
検証は mock のみ(実 LLM 禁止・≤24step)。
"""
from __future__ import annotations

import json
from pathlib import Path

from omegaconf import OmegaConf

from society import friends as F
from society import registry as R
from society.config import load_config
from society.engine.simulation import Simulation

_REPO_FINALS = Path(__file__).resolve().parents[1] / "conf" / "finals_observe.yaml"
_ON = {"friend_graph.enabled": "true", "relations.enabled": "true"}


# --------------------------------------------------------------------------- #
# 共通ヘルパ
# --------------------------------------------------------------------------- #
def _sim(tmp_path, name, n=40, steps=1, **ov):
    dot = ["run.seed=42", f"run.n_agents={n}", f"run.n_steps={steps}",
           f"run.name={name}", "model.backend=mock"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return Simulation(load_config(dot), out_dir=tmp_path / name)


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _relations(sim) -> dict:
    """全個体の関係台帳を**挿入順も含めて**写し取る(再生の一致判定の実体)。"""
    return {a.id: [(oid, dict(rel)) for oid, rel in a.mem.relations.items()]
            for a in sim.agents}


def _built(sim):
    return [e.payload for e in sim.logger.events if e.kind == "friend_graph_built"]


def _cache_ov(path) -> dict:
    return {"world.friends_cache_dir": str(path).replace("\\", "/")}


def _files(path) -> list:
    return sorted(p.name for p in Path(path).iterdir()) if Path(path).exists() else []


# --------------------------------------------------------------------------- #
# (1) 既定 "" = OFF = 現行と完全同一
# --------------------------------------------------------------------------- #
def test_default_is_off(tmp_path):
    cfg = load_config()
    assert str(cfg.world.friends_cache_dir) == ""
    sim = _sim(tmp_path, "off_probe", **_ON)
    assert F.cache_dir_of(sim) is None


def test_off_writes_nothing_and_matches_pure_default(tmp_path):
    """OFF は 1 ファイルも作らず、明示 "" と純粋既定の L1 が完全一致。"""
    base = _sim(tmp_path, "fc_base", steps=12, **_ON)
    base.run()
    same = _sim(tmp_path, "fc_same", steps=12,
                **{**_ON, "world.friends_cache_dir": ""})
    same.run()
    assert _l1(base) == _l1(same)
    assert not (tmp_path / "cache").exists()


def test_cache_dir_resolution_is_repo_relative(tmp_path):
    """相対パスはリポジトリルート基準。★friend_graph は OFF にする(テストがリポジトリ内へ
    キャッシュを書き出さないため = 副作用ゼロ)。"""
    from society.config import REPO_ROOT
    sim = _sim(tmp_path, "fc_rel",
               **{"friend_graph.enabled": "false",
                  "world.friends_cache_dir": "data/cache/friend_graph"})
    assert F.cache_dir_of(sim) == REPO_ROOT / "data" / "cache" / "friend_graph"
    abs_dir = (tmp_path / "abs").as_posix()
    sim2 = _sim(tmp_path, "fc_abs", **_ON,
                **{"world.friends_cache_dir": abs_dir})
    assert Path(F.cache_dir_of(sim2)) == Path(abs_dir)


# --------------------------------------------------------------------------- #
# (2) 初回 build + 保存 → 2 回目ロードで完全一致
# --------------------------------------------------------------------------- #
def test_second_run_loads_and_replays_identically(tmp_path):
    """★本レーンの中核: 2 回目はロード再生で relations・イベント・挿入順まで一致。"""
    cache = tmp_path / "cache"
    ov = {**_ON, **_cache_ov(cache)}
    assert _files(cache) == []                 # 初回の前は空
    first = _sim(tmp_path, "fc_first", **ov)   # 構築 → 保存(Simulation.__init__ の中で走る)
    rel_first, built_first = _relations(first), _built(first)
    saved = _files(cache)
    assert len(saved) == 1 and saved[0].startswith("friend_graph_")
    assert saved[0].endswith(".bin")

    second = _sim(tmp_path, "fc_second", **ov)
    assert _relations(second) == rel_first
    assert _built(second) == built_first
    assert _files(cache) == saved              # 2 回目は書き直さない


def test_cached_replay_equals_the_uncached_build(tmp_path):
    """キャッシュ再生 == キャッシュ OFF のフル構築(台帳も L1 も 1 バイト違わない)。"""
    cache = tmp_path / "cache"
    warm = _sim(tmp_path, "fc_warm", **_ON, **_cache_ov(cache))
    del warm
    plain = _sim(tmp_path, "fc_plain", steps=12, **_ON)
    plain.run()
    cached = _sim(tmp_path, "fc_cached", steps=12, **_ON, **_cache_ov(cache))
    cached.run()
    assert _relations(cached) == _relations(plain)
    assert _l1(cached) == _l1(plain)


def test_downstream_structures_match(tmp_path):
    """下流(tier 分布・closeness・平均次数)がロード経路でも同一。"""
    cache = tmp_path / "cache"
    _sim(tmp_path, "fc_ds_warm", **_ON, **_cache_ov(cache))
    plain = _sim(tmp_path, "fc_ds_plain", **_ON)
    cached = _sim(tmp_path, "fc_ds_cached", **_ON, **_cache_ov(cache))

    def _prof(sim):
        out = {}
        for a in sim.agents:
            for oid, rel in a.mem.relations.items():
                if "closeness" in rel:
                    out[(a.id, oid)] = (int(rel["tier"]), float(rel["closeness"]))
        return out

    assert _prof(cached) == _prof(plain)
    assert _built(cached) == _built(plain)
    assert _prof(plain), "友人辺が 1 本も張られていない(前提が崩れている)"


# --------------------------------------------------------------------------- #
# (3) キー不一致 = 再構築(黙って別物を再生しない)
# --------------------------------------------------------------------------- #
def test_key_changes_with_config_and_roster(tmp_path):
    """設定・人数・tier 閾値が変わればキーが変わる(= 別ファイル = 誤再生しない)。"""
    cache = tmp_path / "cache"
    base = {**_ON, **_cache_ov(cache)}
    _sim(tmp_path, "fc_k0", **base)
    n0 = len(_files(cache))
    _sim(tmp_path, "fc_k1", **base, **{"friend_graph.seed": 999})
    _sim(tmp_path, "fc_k2", **base, **{"friend_graph.acq_extra": 3})
    _sim(tmp_path, "fc_k3", n=41, **base)
    _sim(tmp_path, "fc_k4", **base, **{"relations.tier_friend": 5.5})
    assert len(_files(cache)) == n0 + 4, _files(cache)


def test_rebuild_after_key_change_matches_full_build(tmp_path):
    """キーが変わった構成でも、再構築の結果は OFF のフル構築と一致する。"""
    cache = tmp_path / "cache"
    _sim(tmp_path, "fc_r_warm", **_ON, **_cache_ov(cache))
    ov = {"friend_graph.acq_extra": 3}
    plain = _sim(tmp_path, "fc_r_plain", **_ON, **ov)
    cached = _sim(tmp_path, "fc_r_cached", **_ON, **_cache_ov(cache), **ov)
    assert _relations(cached) == _relations(plain)


def test_source_hash_is_part_of_the_key(tmp_path, monkeypatch):
    """friends.py の内容 hash がキーに入っている(生成規則が変われば自動で無効化)。"""
    sim = _sim(tmp_path, "fc_src", **_ON)
    residents = sorted((a for a in sim.agents if not a.visitor), key=lambda a: a.id)
    thr = {1: 2.0, 2: 5.0, 3: 12.0}
    k0 = F.cache_key(sim, residents, sim.friendcfg, thr)
    real = F._self_hash()
    assert real != "src-unavailable" and len(real) == 32
    monkeypatch.setattr(F, "_self_hash", lambda: "0" * 32)
    assert F.cache_key(sim, residents, sim.friendcfg, thr) != k0


def test_tier_thresholds_are_part_of_the_key(tmp_path):
    sim = _sim(tmp_path, "fc_thr", **_ON)
    residents = sorted((a for a in sim.agents if not a.visitor), key=lambda a: a.id)
    k0 = F.cache_key(sim, residents, sim.friendcfg, {1: 2.0, 2: 5.0, 3: 12.0})
    k1 = F.cache_key(sim, residents, sim.friendcfg, {1: 2.0, 2: 5.5, 3: 12.0})
    assert k0 != k1 and len(k0) == 32


def test_key_changes_when_the_roster_attributes_change(tmp_path):
    """名簿(年齢・職業・所属・住まい)が違えばキーが違う = 誤再生しない。"""
    sim = _sim(tmp_path, "fc_roster", **_ON)
    residents = sorted((a for a in sim.agents if not a.visitor), key=lambda a: a.id)
    thr = {1: 2.0, 2: 5.0, 3: 12.0}
    k0 = F.cache_key(sim, residents, sim.friendcfg, thr)
    for attr, value in (("age", 99), ("occupation", "zzz"), ("org_id", "zzz"),
                        ("org_role", "zzz"), ("home_building", "zzz")):
        old = getattr(residents[0], attr, None)
        setattr(residents[0], attr, value)
        assert F.cache_key(sim, residents, sim.friendcfg, thr) != k0, attr
        setattr(residents[0], attr, old)
    assert F.cache_key(sim, residents, sim.friendcfg, thr) == k0


# --------------------------------------------------------------------------- #
# (4) 破損・失敗は黙って再構築(例外を上げない)
# --------------------------------------------------------------------------- #
def test_corrupt_cache_falls_back_to_rebuild(tmp_path):
    cache = tmp_path / "cache"
    ov = {**_ON, **_cache_ov(cache)}
    good = _sim(tmp_path, "fc_c_good", **ov)
    rel = _relations(good)
    path = cache / _files(cache)[0]
    for blob in (b"", b"garbage", b"SBYFG1\x00\x00" + b"\x00" * 8,
                 path.read_bytes()[:-3], path.read_bytes() + b"xx"):
        path.write_bytes(blob)
        again = _sim(tmp_path, f"fc_c_{len(blob)}", **ov)
        assert _relations(again) == rel, len(blob)
        assert _built(again) == _built(good)


def test_checksum_detects_a_flipped_byte(tmp_path):
    cache = tmp_path / "cache"
    ov = {**_ON, **_cache_ov(cache)}
    good = _sim(tmp_path, "fc_bit_good", **ov)
    rel = _relations(good)
    path = cache / _files(cache)[0]
    blob = bytearray(path.read_bytes())
    blob[-40] ^= 0xFF                     # 本体の 1 バイトを壊す(末尾の digest は無傷)
    path.write_bytes(bytes(blob))
    assert F.load_edges(cache, b"\x00" * 32) is None        # 別キーは当たらない
    again = _sim(tmp_path, "fc_bit_again", **ov)
    assert _relations(again) == rel


def test_unwritable_cache_dir_does_not_raise(tmp_path):
    """保存先が作れなくてもランは止まらず、結果は OFF と同じ。"""
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    plain = _sim(tmp_path, "fc_w_plain", **_ON)
    weird = _sim(tmp_path, "fc_w_weird", **_ON,
                 **_cache_ov(blocker / "sub"))
    assert _relations(weird) == _relations(plain)


def test_load_edges_returns_none_for_a_missing_file(tmp_path):
    assert F.load_edges(tmp_path / "nope", b"\x01" * 32) is None


def test_save_and_load_round_trip(tmp_path):
    import array
    key = b"\x07" * 32
    edges = (array.array("q", [1, 2, 300000]), array.array("q", [4, 5, 700000]),
             array.array("b", [3, 2, 1]))
    F.save_edges(tmp_path, key, edges)
    got = F.load_edges(tmp_path, key)
    assert got is not None
    assert list(got[0]) == [1, 2, 300000]
    assert list(got[1]) == [4, 5, 700000]
    assert list(got[2]) == [3, 2, 1]
    assert F.load_edges(tmp_path, b"\x08" * 32) is None      # 別キーは当たらない
    assert not list(tmp_path.glob("*.tmp"))                  # tmp が残らない


def test_empty_edge_list_round_trips(tmp_path):
    import array
    key = b"\x09" * 32
    F.save_edges(tmp_path, key, (array.array("q"), array.array("q"),
                                 array.array("b")))
    got = F.load_edges(tmp_path, key)
    assert got is not None and len(got[2]) == 0


# --------------------------------------------------------------------------- #
# (5) 生成規則の決定論(キャッシュの正当性の前提)
# --------------------------------------------------------------------------- #
def test_build_draws_no_random_stream(tmp_path):
    """build_friend_graph が乱数 stream を 1 本も引かない(= 同一入力 → 同一グラフ)。"""
    class _Hub:
        def __init__(self, inner):
            self._inner = inner
            self.names = []

        def stream(self, *key):
            self.names.append(str(key[0]) if key else "")
            return self._inner.stream(*key)

        def __getattr__(self, item):
            return getattr(self._inner, item)

    sim = _sim(tmp_path, "fc_det_probe", **_ON)
    residents = sorted((a for a in sim.agents if not a.visitor), key=lambda a: a.id)
    for a in residents:                       # 台帳を空にして張り直す
        a.mem.relations.clear()
    hub = _Hub(sim.hub)
    sim.hub = hub
    F.build_friend_graph(sim)
    assert hub.names == [], hub.names


def test_two_uncached_builds_agree(tmp_path):
    """同一入力を 2 回構築すると同じグラフになる(= キャッシュしてよいことの前提)。"""
    a = _sim(tmp_path, "fc_s1", **_ON)
    b = _sim(tmp_path, "fc_s2", **_ON)
    assert _relations(a) == _relations(b)
    assert _built(a) == _built(b)


def test_cache_key_is_stable_across_processes(tmp_path):
    """キーは blake2b の純関数 = 同一入力なら 2 度計算しても同じ(hash 乱択の混入なし)。"""
    sim = _sim(tmp_path, "fc_stable", **_ON)
    residents = sorted((a for a in sim.agents if not a.visitor), key=lambda a: a.id)
    thr = {1: 2.0, 2: 5.0, 3: 12.0}
    assert F.cache_key(sim, residents, sim.friendcfg, thr) == \
        F.cache_key(sim, residents, sim.friendcfg, thr)


# --------------------------------------------------------------------------- #
# (6) 契約列挙ピン
# --------------------------------------------------------------------------- #
def test_finals_profile_declares_the_registered_value():
    fin = OmegaConf.load(_REPO_FINALS)
    assert str(fin.world.friends_cache_dir) == "data/cache/friend_graph"
    assert bool(fin.friend_graph.enabled) is True


def test_registry_declares_the_new_key():
    f = R.BY_ID.get("world.friends_cache_dir")
    assert f is not None, "world.friends_cache_dir がレジストリ未宣言"
    assert f.repro_tier == "strict"
    assert f.fingerprint_risk == "none"
    assert f.affects_k is False            # 層1 = LLM 呼の発生点も本数も変わらない
    assert f.off_value == ""
    assert R.undeclared_toggles(load_config()) == []


def test_no_new_event_kinds():
    from society.observer.schema import EVENT_KINDS
    assert "friend_graph_cached" not in EVENT_KINDS
    assert "friend_graph_built" in EVENT_KINDS


def test_touched_files_are_not_frozen():
    from society.observer import metrics_spec as MS
    for rel in ("src/society/friends.py", "src/society/registry.py"):
        assert rel not in MS.SPEC_FILES, f"凍結ファイルを触っている: {rel}"
