"""第50バッチ: 観測レンズ3本(価値4軸 T1 / 3M欲望 T2 / 信用内訳 T6)のテスト。

検証項目(タスク検収 2026-07-23):
  (R1)  lens OFF は lens ON と L1 が完全一致(レンズは読むだけ=乱数/LLM 呼を増やさない)。
        OFF は L2 に value4_*/motive_*/trust_* 列を一切足さない(L2 不変)。
  (det) lens ON は決定論(同一入力で 2 回 → L2 のレンズ列がバイト一致)。
  (map) DEFAULT_VALUE4_MAP / DEFAULT_MOTIVE_MAP の全トップレベルキーが schema 登録済み kind
        (タイポ検出)。2段マッチ(dict 値)の下位キーは kind ではないので対象外。
  (sum) observe の当日タリーが 4軸+3M で手集計と一致(暦日境界でリセット)。
  (2段) resolve_axis の payload 2段マッチ(service_use.service / life_event.kind / free_action.tags)。
  (gini)信用 Gini・材料別内訳(status.material_breakdown)の単体。
  (viz) mock 合成ランで build_data がレンズを事後計算し、dashboard HTML 生成が通る。
        lens_map.json / L3 status が無ければ従来出力とバイト同一(後方互換)。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "viz"))

from society.config import load_config                       # noqa: E402
from society.engine.simulation import Simulation             # noqa: E402
from society.observer import lens as lens_mod                # noqa: E402
from society.observer.aggregate import _gini                 # noqa: E402
from society.observer.schema import EVENT_KINDS, Event       # noqa: E402
from society import status as status_mod                     # noqa: E402

import make_viewer as mv                                     # noqa: E402

VIEWER = REPO_ROOT / "viz" / "make_viewer.py"

LENS_COLS = ("value4_utility", "value4_emotion", "value4_social", "value4_epistemic",
             "motive_earn", "motive_love", "motive_recognition")


# --------------------------------------------------------------------------- #
# sim ヘルパ
# --------------------------------------------------------------------------- #
def _run(tmp_path, name, **ov):
    dot = ["run.seed=42", "run.n_agents=14", "run.n_steps=22", f"run.name={name}"]
    dot += [f"{k}={v}" for k, v in ov.items()]
    cfg = load_config(dot)
    sim = Simulation(cfg, out_dir=tmp_path / name)
    sim.run()
    return sim


def _l1(sim):
    return [[e.step, e.agent_id, e.kind,
             json.dumps(e.payload, ensure_ascii=False, sort_keys=True)]
            for e in sim.logger.events]


# --------------------------------------------------------------------------- #
# (R1) lens OFF == ON の L1・OFF は L2 に列なし
# --------------------------------------------------------------------------- #
def test_lens_is_read_only_l1_invariant(tmp_path):
    off = _run(tmp_path, "loff", **{"lens.enabled": "false"})
    on = _run(tmp_path, "lon", **{"lens.enabled": "true"})
    assert _l1(off) == _l1(on), "lens ON が L1 を変えた(レンズが読むだけでない=R1 違反)"


def test_lens_off_no_l2_columns(tmp_path):
    off = _run(tmp_path, "l2off", **{"lens.enabled": "false"})
    for row in off.logger.metrics:
        for col in LENS_COLS:
            assert col not in row, f"lens OFF なのに L2 に {col} 列がある(L2 不変違反)"


def test_lens_on_emits_l2_columns(tmp_path):
    on = _run(tmp_path, "l2on", **{"lens.enabled": "true"})
    # 全 step の各行にレンズ7列が在る(当日タリーの全体スカラー)
    assert on.logger.metrics, "L2 が空"
    for row in on.logger.metrics:
        for col in LENS_COLS:
            assert col in row, f"lens ON なのに L2 に {col} 列が無い"
            assert isinstance(row[col], int)


def test_lens_on_deterministic_l2(tmp_path):
    a = _run(tmp_path, "det_a", **{"lens.enabled": "true"})
    b = _run(tmp_path, "det_b", **{"lens.enabled": "true"})
    va = [[r["step"], *[r[c] for c in LENS_COLS]] for r in a.logger.metrics]
    vb = [[r["step"], *[r[c] for c in LENS_COLS]] for r in b.logger.metrics]
    assert va == vb, "lens ON の L2 レンズ列が決定論でない(2 回で不一致)"


def test_lens_off_sidecar_absent(tmp_path):
    off = _run(tmp_path, "sc_off", **{"lens.enabled": "false"})
    assert not (off.out_dir / "lens_map.json").exists(), "lens OFF でサイドカーが書かれた"


def test_lens_on_sidecar_written(tmp_path):
    on = _run(tmp_path, "sc_on", **{"lens.enabled": "true"})
    p = on.out_dir / "lens_map.json"
    assert p.exists(), "lens ON でサイドカーが書かれていない"
    m = json.loads(p.read_text(encoding="utf-8"))
    assert m["value_axes"] == list(lens_mod.VALUE_AXES)
    assert m["motives"] == list(lens_mod.MOTIVES)
    assert "value4" in m and "motive_map" in m


# --------------------------------------------------------------------------- #
# (map) kind_map の全トップレベルキーが schema 登録済み kind(タイポ検出)
# --------------------------------------------------------------------------- #
def test_kind_map_keys_are_registered():
    for label, m in (("value4", lens_mod.DEFAULT_VALUE4_MAP),
                     ("motive", lens_mod.DEFAULT_MOTIVE_MAP)):
        for k in m:
            assert k in EVENT_KINDS, f"{label} kind_map の未登録 kind '{k}'(タイポ?)"


def test_kind_map_axes_are_canonical():
    """value4 の値(直接 or _else)は values.TAGS の軸に閉じる。motive は 3M に閉じる。"""
    def _leaf_axes(spec):
        if isinstance(spec, str):
            return {spec}
        if isinstance(spec, dict):
            out = set()
            for kk, vv in spec.items():
                if kk in ("__payload__", "__payload_argmax__"):
                    continue
                if isinstance(vv, str):
                    out.add(vv)
            return out
        return set()

    for spec in lens_mod.DEFAULT_VALUE4_MAP.values():
        for ax in _leaf_axes(spec):
            assert ax in lens_mod.VALUE_AXES, f"value4 に非正準軸 {ax}"
    for spec in lens_mod.DEFAULT_MOTIVE_MAP.values():
        for ax in _leaf_axes(spec):
            assert ax in lens_mod.MOTIVES, f"motive に非正準軸 {ax}"


# --------------------------------------------------------------------------- #
# (2段) resolve_axis の payload 2段マッチ
# --------------------------------------------------------------------------- #
def test_resolve_axis_two_stage():
    vmap = lens_mod.DEFAULT_VALUE4_MAP
    mmap = lens_mod.DEFAULT_MOTIVE_MAP
    # 直接
    assert lens_mod.resolve_axis("spend", {}, vmap) == "utility"
    assert lens_mod.resolve_axis("speak", {}, vmap) == "social"
    # service_use.service で分岐
    assert lens_mod.resolve_axis("service_use", {"service": "grooming"}, vmap) == "emotion"
    assert lens_mod.resolve_axis("service_use", {"service": "lesson"}, vmap) == "epistemic"
    assert lens_mod.resolve_axis("service_use", {"service": "clinic"}, vmap) == "utility"
    assert lens_mod.resolve_axis("service_use", {}, vmap) == "utility"          # _else
    # motive: grooming=love / それ以外=None(_else)
    assert lens_mod.resolve_axis("service_use", {"service": "grooming"}, mmap) == "love"
    assert lens_mod.resolve_axis("service_use", {"service": "gym"}, mmap) is None
    # life_event.kind
    assert lens_mod.resolve_axis("life_event", {"kind": "partner"}, mmap) == "love"
    assert lens_mod.resolve_axis("life_event", {"kind": "breakup"}, mmap) is None
    # free_action.tags の argmax
    assert lens_mod.resolve_axis(
        "free_action", {"tags": {"utility": 0.1, "social": 0.7, "emotion": 0.2}}, vmap) == "social"
    assert lens_mod.resolve_axis("free_action", {}, vmap) == "emotion"          # _else
    # 未対応 kind は None
    assert lens_mod.resolve_axis("move_segment", {}, vmap) is None


# --------------------------------------------------------------------------- #
# (sum) observe の当日タリー整合 + 暦日リセット
# --------------------------------------------------------------------------- #
class _FakeLogger:
    def __init__(self, events):
        self.events = events
        self._n_flushed = 0


class _FakeSim:
    def __init__(self, events, enabled=True):
        self.logger = _FakeLogger(events)
        self.cfg = {"lens": {"enabled": enabled}}


def _ev(step, sim_min, kind, payload=None):
    return Event(step=step, sim_min=sim_min, agent_id=1, kind=kind,
                 x=0.0, y=0.0, payload=payload or {})


def test_observe_daily_tally_and_reset():
    day0 = [
        _ev(0, 100, "spend"),        # utility
        _ev(0, 100, "speak"),        # social
        _ev(0, 100, "search"),       # epistemic
        _ev(1, 110, "wage"),         # utility + earn
        _ev(1, 110, "event_host"),   # social + recognition
        _ev(1, 110, "move_segment"),  # 未対応
    ]
    sim = _FakeSim(day0)
    st = lens_mod.observe(sim)
    assert st["value"] == {"utility": 2, "emotion": 0, "social": 2, "epistemic": 1}
    assert st["motive"] == {"earn": 1, "love": 0, "recognition": 1}
    # 合計整合: value 列の和 = 価値分類された件数
    assert sum(st["value"].values()) == 5

    # 暦日をまたぐイベントを追加 → 当日タリーがリセットされる
    sim.logger.events.append(_ev(144, 1500, "partner_formed"))   # 翌日(sim_min//1440=1)
    st2 = lens_mod.observe(sim)
    assert st2["value"] == {"utility": 0, "emotion": 0, "social": 1, "epistemic": 0}
    assert st2["motive"] == {"earn": 0, "love": 1, "recognition": 0}


def test_observe_off_returns_none():
    sim = _FakeSim([_ev(0, 100, "spend")], enabled=False)
    assert lens_mod.observe(sim) is None


def test_observe_idempotent_within_step():
    sim = _FakeSim([_ev(0, 100, "spend"), _ev(0, 100, "spend")])
    a = lens_mod.observe(sim)
    b = lens_mod.observe(sim)          # 同 step 内の再呼出は二度数えない
    assert a["value"]["utility"] == 2 and b["value"]["utility"] == 2


def test_observe_survives_buffer_flush():
    """flush_segment 相当(events が空に戻り _n_flushed が増える)でも当日タリーを欠落なく継続する。"""
    sim = _FakeSim([_ev(0, 100, "spend")])
    lens_mod.observe(sim)
    # flush: 1 件を part へ出して buffer を空に→次 step で新規 1 件が積まれた状態を再現
    sim.logger._n_flushed = 1
    sim.logger.events = [_ev(1, 110, "speak")]
    st = lens_mod.observe(sim)
    assert st["value"] == {"utility": 1, "emotion": 0, "social": 1, "epistemic": 0}


# --------------------------------------------------------------------------- #
# (gini) 信用 Gini・材料別内訳
# --------------------------------------------------------------------------- #
def test_gini_unit():
    assert _gini([1, 1, 1, 1]) == 0.0                    # 完全平等
    assert _gini([]) == 0.0
    assert _gini([0, 0, 0, 0]) == 0.0                    # 合計0
    g = _gini([0, 0, 0, 10])                             # 1人に集中
    assert 0.7 < g < 0.8


def test_material_breakdown_matches_recompute(tmp_path):
    """status.material_breakdown の contrib 合計が sim の合成 status に一致(同じ百分位・重み)。"""
    sim = _run(tmp_path, "brk", **{"hierarchy.enabled": "true"})
    # a.status は最後の日次境界の値=現在の agent 状態とは時点がずれ得る。同一時点で比較するため
    # material_breakdown が使うのと同じ現在状態で _recompute を回し a.status を揃える。
    cfg = status_mod.cfg_of(sim)
    status_mod._recompute(sim, cfg)
    bd = status_mod.material_breakdown(sim, cfg)
    assert set(bd["materials"]) == {"rep", "followers", "wealth", "inst", "biz", "host"}
    assert abs(sum(bd["weights"].values()) - 1.0) < 1e-6
    by_id = {a["id"]: a for a in bd["agents"]}
    for a in sim.agents:
        rec = by_id[a.id]
        assert rec["status"] == round(a.status, 6), "内訳の status が合成 status と不一致"
        assert abs(sum(rec["contrib"].values()) - rec["status"]) < 1e-4  # 丸め誤差内


def test_trust_l2_columns_gated(tmp_path):
    """trust_gini/top10 は lens+trust+hierarchy が全て ON のときだけ L2 に出る。"""
    # hierarchy OFF(lens ON)→ trust 列なし
    a = _run(tmp_path, "tr_off", **{"lens.enabled": "true", "hierarchy.enabled": "false"})
    assert all("trust_gini" not in r for r in a.logger.metrics)
    # 両方 ON → trust 列あり
    b = _run(tmp_path, "tr_on", **{"lens.enabled": "true", "hierarchy.enabled": "true"})
    assert any("trust_gini" in r and "trust_top10" in r for r in b.logger.metrics)


# --------------------------------------------------------------------------- #
# (viz) 合成ランで build_data がレンズを事後計算 + HTML 生成
# --------------------------------------------------------------------------- #
def _minimal_map(path: Path) -> None:
    city = {
        "buildings": [{"id": "b1", "name": "テストビル", "kind": "generic",
                       "levels": 3, "footprint": [[0, 0], [10, 0], [10, 10], [0, 10]]}],
        "nodes": [{"id": "n1", "x": 0, "y": 0, "name": "ノードA"},
                  {"id": "n2", "x": 10, "y": 10}],
        "edges": [{"klass": "footway", "layer": 0, "geometry": [[0, 0], [10, 10]]}],
        "pois": [], "railways": [], "meta": {"origin_latlon": [35.66, 139.70]},
    }
    path.write_text(json.dumps(city, ensure_ascii=False), encoding="utf-8")


def _write_run(tmp_path: Path, name: str, events: list[dict], *,
               lens_map: dict | None = None, l3: list[dict] | None = None,
               agents: list[dict] | None = None) -> Path:
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True, exist_ok=True)
    map_path = tmp_path / f"{name}_map.json"
    _minimal_map(map_path)
    (run_dir / "config.yaml").write_text(
        "world:\n"
        f"  map: {map_path.as_posix()}\n"
        "transit:\n"
        "  file: data/__no_transit_for_test__.json\n",
        encoding="utf-8")
    if agents is None:
        agents = [{"id": i, "name": f"住民{i}", "age": 20 + i, "gender": "男",
                   "occupation": "会社員", "has_bicycle": False, "has_car": False}
                  for i in range(6)]
    (run_dir / "agents.json").write_text(json.dumps(agents, ensure_ascii=False),
                                         encoding="utf-8")
    cols = {
        "step": [int(e["step"]) for e in events],
        "sim_min": [int(e.get("sim_min", 420 + int(e["step"]) * 10)) for e in events],
        "agent_id": [int(e["agent_id"]) for e in events],
        "kind": [str(e["kind"]) for e in events],
        "x": [float(e.get("x", 0.0)) for e in events],
        "y": [float(e.get("y", 0.0)) for e in events],
        "payload": [json.dumps(e.get("payload", {}), ensure_ascii=False) for e in events],
    }
    pq.write_table(pa.table(cols), run_dir / "l1_events.parquet")
    if lens_map is not None:
        (run_dir / "lens_map.json").write_text(json.dumps(lens_map, ensure_ascii=False),
                                               encoding="utf-8")
    if l3 is not None:
        pq.write_table(pa.table({"step": [int(r["step"]) for r in l3],
                                 "state": [json.dumps(r["state"], ensure_ascii=False)
                                           for r in l3]}),
                       run_dir / "l3_snapshots.parquet")
    return run_dir


def _lens_map():
    cfg = lens_mod.build_lens_cfg({"enabled": True})
    return lens_mod.resolved_maps(cfg)


def _ev_dict(step, agent, kind, sim_min=None, **payload):
    d = {"step": step, "agent_id": agent, "kind": kind, "payload": payload}
    if sim_min is not None:
        d["sim_min"] = sim_min
    return d


def test_build_data_lens_present(tmp_path):
    ev = [
        _ev_dict(0, 0, "spend", sim_min=100, amount=900),      # utility
        _ev_dict(0, 0, "speak", sim_min=100, text="hi"),       # social
        _ev_dict(1, 1, "search", sim_min=110, query="q"),      # epistemic
        _ev_dict(1, 0, "wage", sim_min=110, amount=1000),      # utility + earn
        _ev_dict(2, 0, "event_host", sim_min=120, title="t"),  # social + recognition
        _ev_dict(2, 1, "partner_formed", sim_min=120, other=0),  # social + love
    ]
    rd = _write_run(tmp_path, "lz", ev, lens_map=_lens_map())
    data = mv.build_data(rd, include_traffic=False)
    assert "lens" in data
    L = data["lens"]["value"]
    # 全体日別構成(day0 のみ): utility=2, social=3, epistemic=1
    idx = L["days"].index(0)
    assert L["series"]["utility"][idx] == 2
    assert L["series"]["social"][idx] == 3
    assert L["series"]["epistemic"][idx] == 1
    M = data["lens"]["motives"]
    assert M["series"]["earn"][idx] == 1
    assert M["series"]["recognition"][idx] == 1
    assert M["series"]["love"][idx] == 1
    # 住民別プロファイルに agent0(価値を最も帯びた人)が含まれる
    ids = {p["id"] for p in L["profiles"]}
    assert 0 in ids


def test_build_data_no_lens_map_backward_compat(tmp_path):
    ev = [_ev_dict(0, 0, "spend", sim_min=100)]
    rd = _write_run(tmp_path, "nolz", ev)          # lens_map.json 無し
    data = mv.build_data(rd, include_traffic=False)
    assert "lens" not in data and "trust" not in data


def test_build_data_trust_present(tmp_path):
    ev = [
        _ev_dict(0, 0, "event_host", sim_min=100, title="t"),
        _ev_dict(0, 0, "venture_sale", sim_min=100, amount=500, buyer=1),
        _ev_dict(1, 1, "proposal", sim_min=110, proposal_id="p1", text="x"),
        _ev_dict(2, 1, "proposal_passed", sim_min=120, proposal_id="p1", text="x"),
    ]
    agents = [{"id": 0, "name": "A", "occupation": "会社員", "org_role": "管理職"},
              {"id": 1, "name": "B", "occupation": "会社員", "org_role": "一般"}]
    l3 = [{"step": 2, "state": {"agents": [
        {"id": 0, "money": 50000.0, "status": 0.9},
        {"id": 1, "money": 3000.0, "status": 0.3}]}}]
    rd = _write_run(tmp_path, "tz", ev, l3=l3, agents=agents)
    data = mv.build_data(rd, include_traffic=False)
    assert "trust" in data
    T = data["trust"]
    assert T["gini"] > 0.0
    assert len(T["agents"]) == 2
    assert T["agents"][0]["id"] == 0            # status 降順
    # org_role 相関が計算されている
    roles = {o["role"] for o in T["org"]}
    assert "管理職" in roles and "一般" in roles


def test_build_data_no_l3_no_trust(tmp_path):
    ev = [_ev_dict(0, 0, "event_host", sim_min=100)]
    rd = _write_run(tmp_path, "notz", ev, lens_map=_lens_map())   # L3 無し
    data = mv.build_data(rd, include_traffic=False)
    assert "lens" in data and "trust" not in data


def test_dashboard_html_lens_smoke(tmp_path):
    ev = [
        _ev_dict(0, 0, "spend", sim_min=100),
        _ev_dict(0, 0, "speak", sim_min=100, text="hi", hearers=[1]),
        _ev_dict(1, 1, "search", sim_min=110, query="q"),
        _ev_dict(1, 0, "wage", sim_min=110, amount=1000),
        _ev_dict(2, 0, "event_host", sim_min=120, title="t"),
    ]
    l3 = [{"step": 2, "state": {"agents": [
        {"id": 0, "money": 40000.0, "status": 0.8},
        {"id": 1, "money": 2000.0, "status": 0.2}]}}]
    rd = _write_run(tmp_path, "smoke", ev, lens_map=_lens_map(), l3=l3)
    r = subprocess.run([sys.executable, str(VIEWER), str(rd)], capture_output=True)
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    dash = (rd / "dashboard.html").read_text(encoding="utf-8")
    viewer = (rd / "viewer.html").read_text(encoding="utf-8")
    assert "__LENS_JS__" not in dash and "__LENS_TABS__" not in dash
    # タブと描画関数が入っている
    assert 'data-tab="value"' in dash and 'data-tab="motive"' in dash
    assert 'data-tab="trust"' in dash
    for sym in ("renderValue", "renderMotive", "renderTrust", "lensRender"):
        assert sym in dash, f"{sym} が dashboard に無い"
    assert '"lens":' in dash and '"trust":' in dash
    # viewer 側にはレンズトークンもタブも無い(map は不変)
    assert "__LENS_TABS__" not in viewer and 'data-tab="value"' not in viewer


def test_dashboard_html_no_lens_smoke(tmp_path):
    """lens_map/L3 無しのランでも HTML 生成が通り、レンズタブは非表示(後方互換)。"""
    ev = [{"step": s, "agent_id": 0, "kind": "arrive", "x": 1.0, "y": 2.0,
           "payload": {"name": "路上"}} for s in range(3)]
    rd = _write_run(tmp_path, "smoke_none", ev)
    r = subprocess.run([sys.executable, str(VIEWER), str(rd)], capture_output=True)
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    dash = (rd / "dashboard.html").read_text(encoding="utf-8")
    assert "__LENS_TABS__" not in dash and "__LENS_JS__" not in dash
    assert 'data-tab="value"' not in dash and "function lensRender" not in dash
