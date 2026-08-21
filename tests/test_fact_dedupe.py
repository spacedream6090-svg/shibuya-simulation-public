"""FACT-D: fact のエピソード単位重複抑制の検収(2026-08-20 ユーザー追認)。

なぜこのテストが要るか
----------------------
`src/society/truth_ledger.py` は `observer/metrics_spec.py` の凍結 `SPEC_FILES` 14 本の
1 つで、通常は 1 バイトも触らない。FACT-D は **2026-08-20 にユーザーが追認した凍結解除**
(正典 docs/plans/inventory-two-tier-plan.md §FACT-D)で、W3-1(2026-08-07)/ WIT-1・2
(tests/test_witness_channels.py)と同じプロトコルを踏む:

  (1) **逐語温存との同値**: 変更前の `_record_fact` を本ファイル内に `_legacy_record_fact`
      として逐語で持ち、既定 OFF で台帳・重複鍵・カウンタが**完全一致**することを機械証明する。
  (2) **OFF = 現行挙動**: `beliefs.fact_dedupe` 既定 false で実ランの L1 と台帳が一致。
  (3) **ON = 決定論 + resume 安全**: 同 seed 2 ラン一致・resume == straight。
      索引は永続状態ではないので、**索引を捨てて台帳から組み直しても判定列が 1 件も
      変わらない**ことを別途機械証明する(FACT-D の resume 安全はここに懸かっている)。

何を解く問題か: 現行の重複鍵は (種, step, 当事者, 座標) なので、同じ店の同じ品切れでも
客が違えば別 fact になる(実測 stock_out 13,003 件/12h = fact/目撃/信念の洪水)。
ON では「種 × 解決済みの場所 × 話題」= エピソードを単位に、窓が開いている間は 1 件に畳む。
実 LLM は使わない(mock / 合成配置だけで完結する)。
"""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from society import registry as R
from society import truth_ledger as TL
from society.config import load_config
from society.engine.simulation import Simulation
from society.observer import metrics_spec as MS
from society.observer.schema import Event

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src" / "society" / "truth_ledger.py"
_CKPT = _ROOT / "src" / "society" / "engine" / "checkpoint.py"


# =========================================================================== #
# 合成ハーネス(実 city を使わずに `_record_fact` だけを取り出す)
# =========================================================================== #
class _Graph:
    def __init__(self, nodes):
        self._nodes = set(nodes)

    def has_node(self, n):
        return n in self._nodes


class _City:
    """`_resolve_place` / `node_xy` が要求する最小の街(POI 名 → node の索引つき)。"""

    def __init__(self, pois):
        self.poi_list = [dict(p) for p in pois]
        self._xy = {}
        for i, p in enumerate(self.poi_list):
            self._xy.setdefault(str(p["node"]), (100.0 + 10.0 * i, 200.0 + 5.0 * i))
        self.graph = _Graph(self._xy)

    def place_label(self, node):
        return f"L:{node}"

    def node_xy(self, node):
        return self._xy[str(node)]


class _Sim:
    def __init__(self, city):
        self.city = city


_POIS = [{"name": "Aマート", "node": "n_a", "cat": "food"},
         {"name": "Bマート", "node": "n_b", "cat": "food"},
         {"name": "Cホール", "node": "n_c", "cat": "shop"}]


def _sim():
    TL.reset_canaries()
    return _Sim(_City(_POIS))


def _ev(kind, step, agent_id, payload, *, x=0.0, y=0.0):
    return Event(step=int(step), sim_min=int(step) * 10, agent_id=int(agent_id),
                 kind=str(kind), x=float(x), y=float(y), payload=dict(payload))


def _cfg_plain(**over):
    cfg = TL.build_cfg(None)
    cfg.update(over)
    return cfg


_STOCK = {"topic_key": "poi"}                       # 出荷 conf の stock_out と同じ仕様
_PRICE = {"topic_key": "cat", "value_key": "ratio", "value_scale": 2.0}
_CRIME = {"topic_key": "kind"}                      # 場所が payload に無い種(座標だけ)


def _feed(sim, cfg, events, *, drop_index=False):
    """イベント列を `_record_fact` へ流し、判定列(fact_id or None)を返す。

    drop_index=True なら **1 件ごとに索引を捨てる**(= 毎回 台帳から組み直させる)。"""
    out = []
    for e, spec in events:
        if drop_index and getattr(sim, "_tl_epi", None) is not None:
            sim._tl_epi = None
        got = TL._record_fact(sim, cfg, e, spec, int(e.step), int(e.sim_min))
        out.append(None if got is None else got["id"])
    return out


def _ledger(sim):
    return {fid: dict(f) for fid, f in TL.facts_of(sim).items()}


# --- 変更前の実装の逐語温存(ここが「判定式に触っていない」の証拠)------------- #
def _legacy_record_fact(sim, cfg, e, spec, step, sim_min):
    facts = TL.facts_of(sim)
    keys = getattr(sim, "_tl_keys", None)
    if keys is None:
        keys = {}
        sim._tl_keys = keys
    key = f"{e.kind}|{e.step}|{e.agent_id}|{round(float(e.x), 2)}|{round(float(e.y), 2)}"
    if key in keys:
        return None
    node, place_name = TL._resolve_place(sim, e.payload or {})
    if node is not None:
        x, y = sim.city.node_xy(node)
    else:
        x, y = float(e.x), float(e.y)
    radius = float(spec.get("radius_m", cfg["witness_radius_m"]))
    fid = f"f{len(facts):05d}"
    fact = {
        "id": fid,
        "kind": e.kind,
        "step": int(e.step),
        "sim_min": int(e.sim_min),
        "node": node,
        "place": place_name,
        "x": round(float(x), 2),
        "y": round(float(y), 2),
        "value": round(TL._true_value(spec, e.payload or {}), 6),
        "actor": int(e.agent_id),
        "topics": TL._topics(place_name, spec, e.payload or {},
                             int(cfg["min_topic_len"])),
        "radius_m": round(radius, 2),
        "window": int(spec.get("window", cfg["witness_window"])),
        "canary": TL.register_canary(fid),
    }
    facts[fid] = fact
    keys[key] = fid
    TL._stats(sim)["facts"] += 1
    return fact


def _mixed_stream(n_steps=14):
    """種・場所・話題・当事者が入り混じったイベント列(合成の材料)。"""
    events = []
    for st in range(n_steps):
        events.append((_ev("stock_out", st, 100 + st,
                           {"poi": "Aマート", "cat": "food"}, x=float(st)), _STOCK))
        events.append((_ev("stock_out", st, 200 + st,
                           {"poi": "Bマート", "cat": "food"}, x=float(st) + 0.5),
                       _STOCK))
        events.append((_ev("price_change", st, 300 + st,
                           {"poi": "Aマート", "cat": "food", "ratio": 1.4}), _PRICE))
        events.append((_ev("price_change", st, 301 + st,
                           {"poi": "Aマート", "cat": "shop", "ratio": 0.8}), _PRICE))
        events.append((_ev("crime", st, 400 + st, {"kind": "ひったくり"},
                           x=7.0 + st, y=9.0), _CRIME))
        events.append((_ev("event_host", st, 500 + st,
                           {"node": "n_c", "title": "試しの会"}), {"topic_key": "title"}))
    return events


# =========================================================================== #
# 1) conf: 既定 OFF・型強制・レジストリ宣言
# =========================================================================== #
def test_default_is_off_everywhere():
    cfg = load_config()
    assert bool(cfg.beliefs.fact_dedupe) is False, "出荷 conf の既定が OFF でない"
    assert TL.DEFAULTS["fact_dedupe"] is False
    assert TL.build_cfg(None)["fact_dedupe"] is False
    assert TL.build_cfg(cfg.beliefs)["fact_dedupe"] is False


def test_build_cfg_coerces_the_new_key():
    """dotlist / dict / OmegaConf のどれでも bool へ正準化される(既存の流儀)。"""
    assert TL.build_cfg({"fact_dedupe": True})["fact_dedupe"] is True
    assert TL.build_cfg({"fact_dedupe": 1})["fact_dedupe"] is True
    assert TL.build_cfg({"fact_dedupe": 0})["fact_dedupe"] is False
    on = load_config(["beliefs.fact_dedupe=true"])
    assert TL.build_cfg(on.beliefs)["fact_dedupe"] is True
    # 他のキーを巻き込んでいない(既定 OFF のまま)
    assert TL.build_cfg(on.beliefs)["enabled"] is False


def test_registry_declares_the_new_toggle():
    assert "beliefs.fact_dedupe" in R.BY_ID, "新キーがレジストリ未宣言"
    feat = R.BY_ID["beliefs.fact_dedupe"]
    assert feat.repro_tier == "journal", "台帳本体(beliefs.enabled)と同じ等級であること"
    assert feat.affects_k is False, "LLM 呼数は 1 本も動かない"
    assert feat.fingerprint_risk == "none", "プロンプトに 1 バイトも足さない"
    assert R.undeclared_toggles(load_config()) == []


def test_verify_mode_disables_the_new_toggle():
    """verify モード(strict のみ)では journal 等級として自動 OFF になる。"""
    cfg = load_config(["run.mode=verify", "beliefs.enabled=true",
                       "beliefs.fact_dedupe=true"])
    gated, rep = R.apply_mode(cfg)
    assert bool(gated.beliefs.fact_dedupe) is False
    assert "beliefs.fact_dedupe" in {d["id"] for d in rep["auto_disabled"]}


# =========================================================================== #
# 2) OFF = 逐語温存の実装と完全一致(台帳・重複鍵・カウンタ・判定列)
# =========================================================================== #
def test_off_is_verbatim_identical_to_the_legacy_recorder():
    """★同じイベント列で 旧 `_record_fact` と 新(OFF)が 1 件も違わない。"""
    cfg = _cfg_plain()
    assert cfg["fact_dedupe"] is False
    events = _mixed_stream()

    old = _sim()
    old_ids = [None if (f := _legacy_record_fact(old, cfg, e, spec, e.step,
                                                 e.sim_min)) is None else f["id"]
               for e, spec in events]
    new = _sim()
    new_ids = _feed(new, cfg, events)

    assert old_ids == new_ids, "OFF なのに採録の判定列が動いた"
    assert _ledger(old) == _ledger(new), "OFF なのに台帳が動いた"
    assert dict(old._tl_keys) == dict(new._tl_keys), "OFF なのに重複鍵が動いた"
    assert dict(TL._stats(old)) == dict(TL._stats(new))
    assert "facts_deduped" not in TL._stats(new), "OFF で抑制カウンタが生えた"
    assert getattr(new, "_tl_epi", None) is None, "OFF でエピソード索引が生えた"
    assert len(_ledger(new)) == len(events), "重複が起きない前提が崩れた"


def test_off_explicit_false_matches_the_default():
    cfg_default = _cfg_plain()
    cfg_false = TL.build_cfg({"fact_dedupe": False})
    events = _mixed_stream()
    a, b = _sim(), _sim()
    assert _feed(a, cfg_default, events) == _feed(b, cfg_false, events)
    assert _ledger(a) == _ledger(b)


# =========================================================================== #
# 3) ON: 窓境界・場所/話題の粒度・場所不明の扱い
# =========================================================================== #
_ON = TL.build_cfg({"fact_dedupe": True})


def test_one_episode_becomes_one_fact_inside_the_window():
    """★窓の中(既定 6 step)は客が何人来ても 1 件・窓が閉じた step から次のエピソード。"""
    sim = _sim()
    events = [(_ev("stock_out", st, 100 + st, {"poi": "Aマート", "cat": "food"},
                   x=float(st)), _STOCK) for st in range(14)]
    ids = _feed(sim, _ON, events)
    # window=6: step 0 で 1 件目 → 1..5 は抑制 → 6 で 2 件目 → 7..11 抑制 → 12 で 3 件目
    assert ids == ["f00000", None, None, None, None, None,
                   "f00001", None, None, None, None, None,
                   "f00002", None], ids
    assert [f["step"] for f in TL.facts_of(sim).values()] == [0, 6, 12]
    assert TL._stats(sim)["facts"] == 3
    assert TL._stats(sim)["facts_deduped"] == 11, "抑制カウンタが合わない"


def test_same_step_different_customers_collapse_to_one():
    """同じ step に別々の客が同じ店で品切れに当たっても 1 件(洪水の主犯の形)。"""
    sim = _sim()
    events = [(_ev("stock_out", 0, 100 + i, {"poi": "Aマート", "cat": "food"},
                   x=float(i)), _STOCK) for i in range(50)]
    ids = _feed(sim, _ON, events)
    assert ids[0] == "f00000" and set(ids[1:]) == {None}
    assert len(TL.facts_of(sim)) == 1


def test_different_place_or_topic_is_a_different_episode():
    """場所が違えば別エピソード / 同じ場所でも話題(topic_key の値)が違えば別エピソード。"""
    sim = _sim()
    events = [
        (_ev("stock_out", 0, 1, {"poi": "Aマート", "cat": "food"}), _STOCK),
        (_ev("stock_out", 0, 2, {"poi": "Bマート", "cat": "food"}), _STOCK),   # 別の店
        (_ev("stock_out", 1, 3, {"poi": "Aマート", "cat": "food"}), _STOCK),   # 抑制
        (_ev("price_change", 1, 4,
             {"poi": "Aマート", "cat": "food", "ratio": 1.4}), _PRICE),        # 別の種
        (_ev("price_change", 1, 5,
             {"poi": "Aマート", "cat": "shop", "ratio": 1.4}), _PRICE),        # 別の cat
        (_ev("price_change", 2, 6,
             {"poi": "Aマート", "cat": "shop", "ratio": 0.9}), _PRICE),        # 抑制
    ]
    ids = _feed(sim, _ON, events)
    assert ids == ["f00000", "f00001", None, "f00002", "f00003", None]
    kinds = [(f["kind"], f["place"], tuple(f["topics"]))
             for f in TL.facts_of(sim).values()]
    assert len(set(kinds)) == 4, f"エピソードの粒度が潰れた: {kinds}"


def test_window_is_taken_from_the_kind_spec():
    """窓幅は種ごとの仕様(fact_kinds[kind].window)に従う=コードの定数ではない。"""
    spec = {"topic_key": "poi", "window": 2}
    sim = _sim()
    events = [(_ev("stock_out", st, 10 + st, {"poi": "Aマート", "cat": "food"}),
               spec) for st in range(6)]
    ids = _feed(sim, _ON, events)
    assert ids == ["f00000", None, "f00001", None, "f00002", None]


def test_unresolvable_place_is_never_folded():
    """★場所が解決できない fact は畳まない(同定できないものを潰さない)。"""
    sim = _sim()
    events = [(_ev("crime", st, 400 + st, {"kind": "ひったくり"},
                   x=float(st), y=1.0), _CRIME) for st in range(6)]
    ids = _feed(sim, _ON, events)
    assert None not in ids and len(ids) == len(set(ids)) == 6
    for f in TL.facts_of(sim).values():
        assert TL._episode_key_of(f) is None
    assert "facts_deduped" not in TL._stats(sim)


def test_episode_key_uses_only_fields_that_exist_on_the_fact():
    """鍵は fact レコードの既存欄だけから作る(新しい欄も payload も要らない)。"""
    sim = _sim()
    ids = _feed(sim, _ON, [(_ev("stock_out", 0, 1,
                                {"poi": "Aマート", "cat": "food"}), _STOCK)])
    fact = TL.facts_of(sim)[ids[0]]
    key = TL._episode_key_of(fact)
    assert key is not None
    # 台帳から取り出した欄だけの dict でも同じ鍵になる(= 索引の再構築が成り立つ根拠)
    slim = {"kind": fact["kind"], "node": fact["node"], "place": fact["place"],
            "topics": list(fact["topics"])}
    assert TL._episode_key_of(slim) == key
    assert set(slim) <= set(fact), "鍵が fact に無い欄を要求している"


# =========================================================================== #
# 4) 索引は**永続状態ではない**: 台帳から組み直しても判定が 1 件も変わらない
# =========================================================================== #
def _scan_decision(sim, cfg, e, spec, *, mode="last"):
    """索引を使わず台帳の走査だけで「抑制するか」を決める参照実装。

    mode="last" … 同じ鍵で**最後に積まれた** fact の窓を見る(実装と同じ規則)
    mode="any"  … 同じ鍵の**どれか**の fact の窓が開いていれば抑制(窓が種ごと定数なら同値)"""
    facts = TL.facts_of(sim)
    node, place = TL._resolve_place(sim, e.payload or {})
    topics = TL._topics(place, spec, e.payload or {}, int(cfg["min_topic_len"]))
    ek = TL._episode_key_of({"kind": e.kind, "node": node, "place": place,
                             "topics": topics})
    if ek is None:
        return False
    hits = [f for f in facts.values() if TL._episode_key_of(f) == ek]
    if not hits:
        return False
    if mode == "any":
        return any(int(e.step) < int(f["step"]) + max(1, int(f["window"]))
                   for f in hits)
    last = hits[-1]
    return int(e.step) < int(last["step"]) + max(1, int(last["window"]))


def test_index_free_scan_reproduces_every_decision():
    """★索引が無くても台帳の走査だけで同じ判定になる(resume 安全の中身)。"""
    sim = _sim()
    events = _mixed_stream(20)
    n_sup = 0
    for e, spec in events:
        want_last = _scan_decision(sim, _ON, e, spec, mode="last")
        want_any = _scan_decision(sim, _ON, e, spec, mode="any")
        assert want_last == want_any, "『最後の窓』と『どれかの窓』が食い違った"
        got = TL._record_fact(sim, _ON, e, spec, int(e.step), int(e.sim_min))
        assert (got is None) == want_last, "台帳走査の判定と実装の判定が違う"
        n_sup += 1 if got is None else 0
    assert n_sup > 0, "1 件も抑制が起きていない(前提が崩れた)"


def test_dropping_the_index_changes_nothing():
    """★索引を毎回捨てて組み直しても、判定列も台帳も 1 件も変わらない。"""
    events = _mixed_stream(20)
    kept = _sim()
    dropped = _sim()
    ids_kept = _feed(kept, _ON, events)
    ids_drop = _feed(dropped, _ON, events, drop_index=True)
    assert ids_kept == ids_drop, "索引の有無で判定列が変わった(resume で壊れる)"
    assert _ledger(kept) == _ledger(dropped)
    assert dict(TL._stats(kept)) == dict(TL._stats(dropped))
    assert None in ids_kept, "抑制が 1 件も起きていない(前提が崩れた)"


def test_rebuilt_index_equals_the_incremental_one():
    """逐次更新した索引と、台帳から組み直した索引が**同じ表**になる(後勝ち)。"""
    sim = _sim()
    _feed(sim, _ON, _mixed_stream(20))
    incremental = dict(sim._tl_epi)
    sim._tl_epi = None
    rebuilt = dict(TL._episode_index(sim, TL.facts_of(sim)))
    assert rebuilt == incremental, "組み直した索引が逐次更新の表と違う"
    assert rebuilt, "索引が空(前提が崩れた)"


def test_index_is_not_a_persisted_state():
    """索引を checkpoint へ載せない(= 新しい永続状態を 1 つも作っていない)。"""
    assert "_tl_epi" not in _CKPT.read_text(encoding="utf-8"), \
        "エピソード索引が checkpoint に載っている(導出キャッシュのはずが永続状態になった)"
    assert "_tl_epi" in _SRC.read_text(encoding="utf-8")


def test_a_pruned_ledger_does_not_suppress_forever():
    """台帳の掃除(将来の prune / ttl)で先行 fact が消えたら、次は新しく積む。"""
    sim = _sim()
    ev1 = (_ev("stock_out", 0, 1, {"poi": "Aマート", "cat": "food"}), _STOCK)
    assert _feed(sim, _ON, [ev1]) == ["f00000"]
    stale = dict(sim._tl_epi)
    TL.facts_of(sim).clear()                       # ★掃除された(索引は古いまま)
    sim._tl_epi = dict(stale)
    got = _feed(sim, _ON, [(_ev("stock_out", 1, 2,
                                {"poi": "Aマート", "cat": "food"}), _STOCK)])
    assert got == ["f00000"], "幽霊の参照で抑制が続いている"
    assert len(TL.facts_of(sim)) == 1


def test_episode_open_predicate_matches_the_witness_window():
    """抑制の窓は `_witness_pass` の判定(半開区間 [step, step+window))と同じ式。"""
    facts = {"f0": {"id": "f0", "step": 10, "window": 6}}
    for st in range(10, 16):
        assert TL._episode_open(facts, "f0", st) is True
        assert facts["f0"]["step"] <= st < facts["f0"]["step"] + 6
    assert TL._episode_open(facts, "f0", 16) is False
    assert TL._episode_open(facts, "f0", 9) is True       # 過去は窓の中(逆流しない)
    assert TL._episode_open(facts, None, 10) is False
    assert TL._episode_open(facts, "missing", 10) is False
    facts["f1"] = {"id": "f1", "step": 0, "window": 0}    # window<=0 は 1 step 扱い
    assert TL._episode_open(facts, "f1", 0) is True
    assert TL._episode_open(facts, "f1", 1) is False


# =========================================================================== #
# 5) 実ラン(mock): OFF バイト一致 / ON 決定論 / resume == straight / 洪水が減る
# =========================================================================== #
_COMMERCE = {"commerce.enabled": "true", "commerce.demand_ref": "2",
             "commerce.price_sensitivity": "0.3", "commerce.stock_threshold": "2",
             "commerce.stock_grievance": "0.03"}


def _cfg_run(name, n_steps, n_agents, **ov):
    dot = ["run.seed=42", f"run.n_agents={n_agents}", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


def _run(tmp_path, name, n_steps=48, n_agents=30, **ov):
    out = tmp_path / name
    sim = Simulation(_cfg_run(name, n_steps, n_agents, **ov), out_dir=out)
    summary = sim.run()
    return sim, out, summary


def _rows(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


def _ledger_json(out):
    return json.loads((out / TL.LEDGER_NAME).read_text(encoding="utf-8"))


def test_off_run_is_identical_to_the_current_world(tmp_path):
    """★fact_dedupe 省略 / false のどちらでも L1・台帳が現行と一致(既定 OFF=バイト一致)。"""
    base, out_b, _ = _run(tmp_path, "fd_base", n_steps=144, n_agents=40,
                          **{"beliefs.enabled": "true", **_COMMERCE})
    off, out_o, _ = _run(tmp_path, "fd_off", n_steps=144, n_agents=40,
                         **{"beliefs.enabled": "true",
                            "beliefs.fact_dedupe": "false", **_COMMERCE})
    assert _rows(off) == _rows(base), "false を明示しただけで L1 が動いた"
    assert _ledger_json(out_o) == _ledger_json(out_b), "OFF なのに台帳が動いた"
    assert getattr(base, "_tl_epi", None) is None, "OFF でエピソード索引が生えた"
    assert [r for r in _rows(base) if r[2] == "belief_update"], \
        "目撃/伝聞が 1 件も出ていない(前提が崩れた)"


def test_on_is_deterministic_across_runs(tmp_path):
    ov = {"beliefs.enabled": "true", "beliefs.fact_dedupe": "true", **_COMMERCE}
    a, out_a, _ = _run(tmp_path, "fd_on_a", n_steps=144, n_agents=40, **ov)
    b, out_b, _ = _run(tmp_path, "fd_on_b", n_steps=144, n_agents=40, **ov)
    assert _rows(a) == _rows(b), "ON が非決定(乱数を引いている疑い)"
    assert _ledger_json(out_a) == _ledger_json(out_b)


def test_on_cuts_the_fact_flood_and_leaves_the_world_alone(tmp_path):
    """★ON で fact/信念が減り、**belief_* 以外の L1 と LLM 呼数は 1 バイトも動かない**。"""
    ov = {"beliefs.enabled": "true", **_COMMERCE}
    off, _oo, s_off = _run(tmp_path, "fd_vol_off", n_steps=144, n_agents=40, **ov)
    on, _on_out, s_on = _run(tmp_path, "fd_vol_on", n_steps=144, n_agents=40,
                             **{**ov, "beliefs.fact_dedupe": "true"})
    assert s_on["llm_calls"] == s_off["llm_calls"] > 0, "ON で LLM 呼数が動いた(R1 違反)"
    strip = lambda sim: [r for r in _rows(sim) if not r[2].startswith("belief")]  # noqa: E731
    assert strip(on) == strip(off), "belief_* 以外の L1 が動いた(世界を歪めている)"
    n_off, n_on = len(TL.facts_of(off)), len(TL.facts_of(on))
    assert 0 < n_on < n_off, f"fact が減っていない: {n_on} vs {n_off}"
    ups = lambda sim: sum(1 for e in sim.logger.events if e.kind == "belief_update")  # noqa: E731
    assert ups(on) < ups(off), f"belief_update が減っていない: {ups(on)} vs {ups(off)}"
    assert TL._stats(on)["facts_deduped"] > 0
    assert "facts_deduped" not in TL._stats(off)
    # 不変条件: 同じエピソードの fact は窓が重ならない(1 エピソード 1 fact)
    last: dict[str, dict] = {}
    for f in TL.facts_of(on).values():
        ek = TL._episode_key_of(f)
        if ek is None:
            continue
        prev = last.get(ek)
        if prev is not None:
            assert int(f["step"]) >= int(prev["step"]) + max(1, int(prev["window"])), \
                f"窓が重なる同一エピソードの fact が 2 件ある: {ek}"
        last[ek] = f


def _bel(sim):
    return {a.id: dict(getattr(a, "_fact_beliefs", None) or {})
            for a in sim.agents}


def _resume_pair(tmp_path, name, n_steps, n_agents, every, **ov):
    """通しラン と 途中 checkpoint からの再開ラン を作って (straight, resumed, L1 2 本)。"""
    straight, out_s, _ = _run(tmp_path, f"{name}_straight", n_steps=n_steps,
                              n_agents=n_agents,
                              **{**ov, "observer.checkpoint_every": str(every)})
    part = tmp_path / f"{name}_part"
    Simulation(_cfg_run(f"{name}_part", every, n_agents,
                        **{**ov, "observer.checkpoint_every": str(every)}),
               out_dir=part).run()
    resumed = Simulation(_cfg_run(f"{name}_part", n_steps, n_agents,
                                  **{**ov, "observer.checkpoint_every": str(every)}),
                         out_dir=part)
    resumed.run(resume_from=part)
    return (straight, resumed,
            pq.read_table(out_s / "l1_events.parquet").to_pylist(),
            pq.read_table(part / "l1_events.parquet").to_pylist(),
            out_s, part)


def test_on_resume_matches_a_straight_run(tmp_path):
    """★ON の checkpoint 再開が通しランと完全一致(索引は台帳から組み直される)。"""
    ov = {"beliefs.enabled": "true", "beliefs.fact_dedupe": "true"}
    straight, resumed, l1s, l1p, out_s, part = _resume_pair(
        tmp_path, "fd_rs", 48, 30, 24, **ov)
    assert l1s == l1p, "resume != straight(L1)"
    assert _ledger_json(out_s) == _ledger_json(part), "resume != straight(台帳)"
    assert _bel(straight) == _bel(resumed), "resume で信念状態が食い違った"
    assert any(_bel(straight).values()), "信念が 1 件も無い(前提が崩れた)"
    assert TL.facts_of(straight) == TL.facts_of(resumed)
    # 再開側の索引は checkpoint から来たのではなく**台帳から**組み直されている
    rebuilt = {}
    for fid, f in TL.facts_of(resumed).items():
        ek = TL._episode_key_of(f)
        if ek is not None:
            rebuilt[ek] = fid
    assert dict(getattr(resumed, "_tl_epi", None) or {}) == rebuilt
    assert rebuilt, "索引が空(前提が崩れた)"


def test_on_resume_adds_no_gap_when_the_suppression_actually_fires(tmp_path):
    """★**抑制が実際に起きる**配置(commerce ON)での resume == straight。

    素の mock ランでは同じエピソードが窓の中で重ならず、抑制が 1 件も起きない
    = 上のテストだけでは「畳んだ判断が checkpoint を跨いで再現するか」を測れていない。
    ここでは stock_out / price_change が出る配置で、抑制が起きたことを確認したうえで
    L1・台帳・信念・カウンタの完全一致を要求する(false 側も同じ配置で回して、
    差が出るとしたら FACT-D の外側だと切り分けられるようにしてある)。"""
    base = {"beliefs.enabled": "true", **_COMMERCE}
    for tag, dd in (("off", "false"), ("on", "true")):
        straight, resumed, l1s, l1p, out_s, part = _resume_pair(
            tmp_path, f"fd_rc_{tag}", 144, 40, 72,
            **{**base, "beliefs.fact_dedupe": dd})
        assert l1s == l1p, f"resume != straight(L1・{tag})"
        assert _ledger_json(out_s) == _ledger_json(part), f"台帳が食い違った({tag})"
        assert _bel(straight) == _bel(resumed), f"信念が食い違った({tag})"
        assert dict(TL._stats(straight)) == dict(TL._stats(resumed)), \
            f"台帳カウンタが食い違った({tag})"
        if dd == "true":
            assert TL._stats(straight)["facts_deduped"] > 0, \
                "抑制が 1 件も起きていない配置で resume を測っている(前提が崩れた)"
            assert TL._stats(resumed)["facts_deduped"] == \
                TL._stats(straight)["facts_deduped"]
            # 再開側の索引が台帳から組み直され、通しランの索引と同じ表になっている
            assert dict(getattr(resumed, "_tl_epi", None) or {}) \
                == dict(getattr(straight, "_tl_epi", None) or {})


def test_resume_decisions_match_when_the_ledger_is_reloaded(tmp_path):
    """機械証明の直球: 同じ台帳から straight 側と resume 側が**同じ判定列**を出す。"""
    events = _mixed_stream(20)
    straight = _sim()
    ids_straight = _feed(straight, _ON, events)
    # 前半だけ走らせて「checkpoint」= 台帳と重複鍵だけを引き継ぐ(索引は引き継がない)
    half = len(events) // 2
    first = _sim()
    ids_first = _feed(first, _ON, events[:half])
    carried = _Sim(_City(_POIS))                   # 新しいプロセス相当(索引なし)
    carried._tl_facts = {k: dict(v) for k, v in TL.facts_of(first).items()}
    carried._tl_keys = dict(first._tl_keys)
    carried._tl_stats = dict(TL._stats(first))
    assert getattr(carried, "_tl_epi", None) is None
    ids_second = _feed(carried, _ON, events[half:])
    assert ids_first + ids_second == ids_straight, "resume の判定列が通しと食い違った"
    assert _ledger(carried) == _ledger(straight)


# =========================================================================== #
# 6) 凍結の状態(承認は『中身を直してよい』であって『凍結から外してよい』ではない)
# =========================================================================== #
def test_truth_ledger_is_still_in_the_frozen_list():
    assert "src/society/truth_ledger.py" in MS.SPEC_FILES
    assert len(MS.SPEC_FILES) == 14
    assert MS.compute()["missing"] == []


def test_freeze_release_is_recorded_in_the_docstring():
    """凍結解除の経緯を module docstring に残す(WIT-1/2 と同じ作法)。"""
    src = _SRC.read_text(encoding="utf-8")
    head = src[:src.index('"""', 3) + 3]
    assert "FACT-D" in head and "2026-08-20" in head
    assert "inventory-two-tier-plan.md" in head


def test_dedupe_path_draws_no_randomness():
    """FACT-D の経路に乱数も時計も入っていない(判定は台帳と step だけの純関数)。"""
    import ast
    tree = ast.parse(_SRC.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in (
                "_episode_key_of", "_episode_index", "_episode_open"):
            names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
            attrs = {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}
            assert not ({"random", "rng", "time"} & (names | attrs))


@pytest.mark.parametrize("on", [False, True])
def test_topics_are_unchanged_by_the_refactor(on):
    """話題トークンの中身は FACT-D の前後で同じ(局所変数へ括り出しただけ)。"""
    cfg = TL.build_cfg({"fact_dedupe": on})
    sim = _sim()
    e = _ev("price_change", 0, 1, {"poi": "Aマート", "cat": "food", "ratio": 1.4})
    fact = TL._record_fact(sim, cfg, e, _PRICE, 0, 0)
    node, place = TL._resolve_place(sim, e.payload)
    assert fact["topics"] == TL._topics(place, _PRICE, e.payload,
                                        int(cfg["min_topic_len"]))
    assert fact["topics"] == ["Aマート", "food"] and fact["node"] == "n_a"
    assert fact["value"] == pytest.approx(0.7)     # ratio 1.4 / value_scale 2.0
