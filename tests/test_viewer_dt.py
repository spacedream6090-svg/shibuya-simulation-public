"""W4-C: viz/make_viewer.py の Δt(run.dt_min)対応の検証。

背景: `docs/research/obs-u2-dt1min-design.md` §1.3(C 級)。解析側の Δt 直書きは W2-3 で
scripts/ 31 本を `scripts/run_dt.py` へ寄せて解消したが、`viz/make_viewer.py` だけが
`STEP_MINUTES = 10` を 15 箇所で直に使っていた(**C 級最後の 1 本**)。

移行の型(W2-3 と同一):
- モジュール定数(`STEP_MINUTES` / `ROLLUP_STEPS_PER_DAY`)は**正準値のまま**据え置く。
- ランを読む入口(`build_data` / `build_rollup_data` / `main`)だけが `run_dt.min_per_step`
  でランの Δt を解決し、下位関数と HTML テンプレートへ配る。**渡さなければ完全同値**。
- JS 側の時間換算は `__STEP_MIN__` などのトークンにして `main` が実値を流し込む。
  正準 Δt=10 では "10"/"6"/"18"/"144" = 従来の直書きに戻るので **出力バイト同一**。

本ファイルが固定するもの:
  (1) 正準値の据え置きとトークン表(Δt=10 で従来の直書き値に一致)。
  (2) テンプレート/フラグメントに Δt 直書きが 1 つも残っていない(機械列挙)。
  (3) **旧ラン(Δt=10)の viewer/dashboard/rollup が HEAD 版とバイト同一**(e2e)。
  (4) Δt=1 のランで壁時計・日境界・軸目盛が 1 step = 1 分として正しく出る。
実 LLM は使わない(全経路 合成データ)。
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import types
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "viz"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import make_viewer as mv                                        # noqa: E402
import run_dt                                                   # noqa: E402


# --------------------------------------------------------------------------- 合成ラン
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


def _write_run(tmp_path: Path, name: str, *, dt_min: int | None = None,
               start_min: int = 420, n_steps: int = 8, n_agents: int = 4,
               rich: bool = True) -> Path:
    """合成ラン。`rich=True` で lens/deviation/structure/assets/endo/orgs/occupancy/
    communities/indoor の**注入フラグメントを全部**立ち上げる(Δt トークンの網羅用)。

    `dt_min` を渡すと config.yaml に `run.dt_min` を書き、sim_min も Δt に合わせて刻む。
    """
    dt = int(dt_min) if dt_min is not None else mv.STEP_MINUTES
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True, exist_ok=True)
    map_path = tmp_path / f"{name}_map.json"
    _minimal_map(map_path)
    cfg = ("world:\n"
           f"  map: {map_path.as_posix()}\n"
           "transit:\n"
           "  file: data/__no_transit_for_test__.json\n")
    if dt_min is not None:
        cfg += f"run:\n  dt_min: {int(dt_min)}\n"
    (run_dir / "config.yaml").write_text(cfg, encoding="utf-8")

    agents = [{"id": i, "name": f"住民{i}", "age": 20 + i, "gender": "男",
               "occupation": "会社員", "visitor": False,
               "has_bicycle": False, "has_car": False,
               **({"org_id": "co_test_1", "org_role": "staff"} if rich else {})}
              for i in range(n_agents)]
    (run_dir / "agents.json").write_text(json.dumps(agents, ensure_ascii=False),
                                         encoding="utf-8")

    ev: list[dict] = []

    def _add(step, aid, kind, **payload):
        ev.append({"step": int(step), "sim_min": start_min + int(step) * dt,
                   "agent_id": int(aid), "kind": kind, "x": float(aid), "y": 0.0,
                   "payload": json.dumps(payload, ensure_ascii=False) if payload else ""})

    for step in range(n_steps):
        for a in range(n_agents):
            _add(step, a, "arrive", name="路上", node="n1")
    _add(0, 0, "move_segment", mode="taxi", pts=[[0.0, 0.0], [1.0, 1.0]])
    _add(1, 0, "speak", text="やあ")
    if rich:
        _add(1, 0, "enter_building", building="b1", floor=2)
        _add(2, 0, "space_move", building="b1", floor=2, to_zone=1)
        _add(3, 0, "exit_building", building="b1")
        _add(2, 1, "joint_invite", accepted=True, basis="appointment")
        _add(3, 2, "joint_invite", accepted=False, basis="fallback")
        _add(4, 1, "production", org="co_test_1")
        _add(4, 2, "serve", org_id="co_test_1")

    cols = {k: [e[k] for e in ev] for k in
            ("step", "sim_min", "agent_id", "kind", "x", "y", "payload")}
    pq.write_table(pa.table(cols, schema=pa.schema([
        ("step", pa.int32()), ("sim_min", pa.int32()), ("agent_id", pa.int32()),
        ("kind", pa.string()), ("x", pa.float32()), ("y", pa.float32()),
        ("payload", pa.string())])), run_dir / "l1_events.parquet")
    if not rich:
        return run_dir

    # --- L2(承諾タブの日末値)/ L3(資産)---
    l2 = {"step": list(range(n_steps)),
          "joint_accept_rate": [0.5] * n_steps,
          "joint_endo_share": [0.25] * n_steps,
          "joint_accept_calib_gap": [0.1] * n_steps,
          "joint_fulfill_rate": [0.75] * n_steps,
          "mean_grievance": [0.2] * n_steps}
    pq.write_table(pa.table(l2), run_dir / "l2_metrics.parquet")
    snaps = []
    for step in (0, n_steps - 1):
        snaps.append((step, json.dumps({"agents": [
            {"id": i, "money": float(100 * (i + 1) + step), "account": 0.0}
            for i in range(n_agents)]}, ensure_ascii=False)))
    pq.write_table(pa.table({"step": [s for s, _ in snaps],
                             "state": [j for _, j in snaps]}),
                   run_dir / "l3_snapshots.parquet")

    # --- 事後サイドカー(どれも「有る時だけ」タブが出る)---
    from society.observer import assets as A                     # noqa: PLC0415
    from society.observer import deviation as dev                # noqa: PLC0415
    from society.observer import lens as lens_mod                # noqa: PLC0415
    (run_dir / "lens_map.json").write_text(json.dumps(
        lens_mod.resolved_maps(lens_mod.build_lens_cfg({"enabled": True})),
        ensure_ascii=False), encoding="utf-8")
    (run_dir / "deviation_map.json").write_text(json.dumps(
        dev.resolved_maps(dev.build_cfg({"enabled": True})),
        ensure_ascii=False), encoding="utf-8")
    (run_dir / "assets_map.json").write_text(json.dumps(
        A.resolved_maps(A.build_cfg({"enabled": True})),
        ensure_ascii=False), encoding="utf-8")
    (run_dir / "structure.json").write_text(json.dumps({
        "days": [0], "churn": {"churn_rate": [0.1]}, "rank": {"tau_prev_day": [None]},
        "centrality": {"turnover": [0.2]}, "community": {"change_rate": [0.3]},
        "stagnation": {"longest": None, "combined": [], "total_stagnant_days": 0,
                       "min_days": 3, "by_signal": {}},
    }, ensure_ascii=False), encoding="utf-8")
    (run_dir / "communities.json").write_text(json.dumps({
        "windows": [{"step_start": 0, "step_end": n_steps,
                     "communities": [{"community_id": 0,
                                      "members": list(range(n_agents))}]}]},
        ensure_ascii=False), encoding="utf-8")
    pq.write_table(pa.table({"building": ["b1"] * n_steps, "floor": [2] * n_steps,
                             "step": list(range(n_steps)), "n": [1] * n_steps}),
                   run_dir / "occupancy.parquet")
    return run_dir


# --------------------------------------------------------------------------- HEAD 版
def _load_head_module():
    """HEAD の make_viewer.py を **本物のパスの __file__ で** exec する。

    `__file__` を実ファイルに合わせるのが要点(`REPO_ROOT = parents[1]` が現行と一致し、
    data/ 配下の台帳や間取りの解決が両者で同じになる。tmp に置くとここがズレて偽の差分が出る)。
    """
    try:
        src = subprocess.check_output(
            ["git", "show", "HEAD:viz/make_viewer.py"],
            cwd=REPO_ROOT, stderr=subprocess.DEVNULL).decode("utf-8")
    except Exception:                                            # noqa: BLE001
        return None
    mod = types.ModuleType("mv_head_dt")
    mod.__file__ = str(REPO_ROOT / "viz" / "make_viewer.py")
    try:
        exec(compile(src, mod.__file__, "exec"), mod.__dict__)   # noqa: S102
    except Exception:                                            # noqa: BLE001
        return None
    return mod


HEAD = _load_head_module()

_OUTPUTS = ("viewer.html", "dashboard.html")


def _run_main(module, run_dir: Path, *flags: str) -> dict:
    argv = sys.argv
    try:
        sys.argv = ["make_viewer.py", str(run_dir), *flags]
        module.main()
    finally:
        sys.argv = argv
    names = ("rollup.html",) if "--daily-rollup" in flags else _OUTPUTS
    return {n: (run_dir / n).read_bytes() for n in names}


# ============================================================ 1. 正準値の据え置き
def test_module_constants_stay_canonical():
    """モジュール定数は正準 Δt=10 のまま(= 渡さなければ従来と完全同値)。"""
    assert mv.STEP_MINUTES == run_dt.CANON_DT_MIN == 10
    assert mv.ROLLUP_STEPS_PER_DAY == run_dt.CANON_STEPS_PER_DAY == 144
    assert mv._steps_per_day() == 144 and mv._steps_per_hour() == 6


def test_dt_token_values_match_legacy_literals():
    """Δt=10 のトークン値 = テンプレートに直書きされていた数値そのもの。"""
    assert mv.dt_token_values(10) == {"__STEP_MIN__": "10", "__STEPS_PER_HOUR__": "6",
                                      "__STEPS_PER_3H__": "18", "__STEPS_PER_DAY__": "144"}
    assert mv.dt_token_values(1) == {"__STEP_MIN__": "1", "__STEPS_PER_HOUR__": "60",
                                     "__STEPS_PER_3H__": "180", "__STEPS_PER_DAY__": "1440"}
    assert set(mv.dt_token_values(10)) == set(mv.DT_TOKENS)


def test_derived_values_agree_with_run_dt():
    """局所ヘルパは run_dt の派生量と同定義(二重定義の腐敗検知)。"""
    for dt in (1, 2, 5, 10, 15, 30, 60):
        assert mv._steps_per_day(dt) == run_dt.steps_per_day(dt_min=dt)
        assert mv._steps_per_hour(dt) == run_dt.steps_per_hour(dt_min=dt)


# ============================================================ 2. Δt 直書きの機械列挙
_TEMPLATE_NAMES = ("MAP_HTML", "DASH_HTML", "ROLLUP_HTML", "_TIME_JS", "_THEME_JS",
                   "_FLOOR_JS", "_INDOOR_JS", "_COMMUNITY_JS", "_MODE_LEGEND_JS",
                   "_LENS_JS", "_DEV_JS", "_STRUCT_JS", "_ASSETS_JS", "_ENDO_JS",
                   "_ORG_JS", "_OCC_JS", "_DASH_TAB_WRAP", "_ORG_MAP_JS")

# 「本当の時間換算」だけを狙う。`${i*10}`(逸脱ヒストの%ラベル)や `%1440`(分 of day)
# のような **偶然の 10 / 1440** は時間換算ではないので対象外(下のテストでも拾わない)。
_HARDCODED = (
    re.compile(r"D\.startMin\s*\+\s*\w+\s*\*\s*10\b"),   # 壁時計 = start + step*Δt
    re.compile(r"=>\s*d\s*\*\s*144\b"),                  # 日 → step 軸
    re.compile(r"\[\s*d\s*\*\s*144\b"),                  # 同上(org 系列)
    re.compile(r"stepsPerDay\s*=\s*144\b"),
)


def test_no_hardcoded_dt_left_in_templates():
    """テンプレート/注入フラグメントに Δt 直書きが 1 つも残っていない。"""
    bad = []
    for name in _TEMPLATE_NAMES:
        text = getattr(mv, name)
        for rx in _HARDCODED:
            for m in rx.finditer(text):
                bad.append(f"{name}: {m.group(0)!r}")
    assert not bad, "Δt 直書きが残っている: " + ", ".join(bad)


def test_generated_html_has_no_leftover_dt_token(tmp_path):
    """生成 HTML にトークンの取り残しが無い(置換漏れ = 画面の JS が壊れる)。"""
    rd = _write_run(tmp_path, "tok")
    out = _run_main(mv, rd)
    out.update(_run_main(mv, rd, "--daily-rollup"))
    for name, blob in out.items():
        text = blob.decode("utf-8")
        for tok in mv.DT_TOKENS:
            assert tok not in text, f"{name} に {tok} が残っている"


# ============================================================ 3. 旧ラン = HEAD とバイト同一
@pytest.mark.xdist_group("viewer_html")
def test_canonical_run_bytes_identical_vs_head(tmp_path):
    """Δt=10(dt_min 未記載の旧ラン)の viewer/dashboard/rollup が HEAD 版とバイト同一。

    注入フラグメント(lens/逸脱/構造/資産/承諾/会社/在館/コミュニティ/屋内/タクシー凡例)を
    全部立ち上げた合成ランで比較する = Δt トークンを置いた全経路を通る。"""
    if HEAD is None:
        pytest.skip("git 不在(HEAD 版を取れない)")
    rd = _write_run(tmp_path, "byteid", n_steps=10)
    head = _run_main(HEAD, rd, "--indoor-moves")
    head.update(_run_main(HEAD, rd, "--daily-rollup"))
    cur = _run_main(mv, rd, "--indoor-moves")
    cur.update(_run_main(mv, rd, "--daily-rollup"))
    assert set(head) == set(cur) == {"viewer.html", "dashboard.html", "rollup.html"}
    for name in sorted(head):
        assert cur[name] == head[name], f"{name} が旧ランでバイト不一致(後方互換違反)"
    # 注入経路を本当に通っていることの確認(空振りの偽合格を防ぐ)
    dash = cur["dashboard.html"].decode("utf-8")
    for probe in ('data-tab="org"', 'data-tab="occ"', 'data-tab="endo"',
                  'data-tab="structure"', 'data-tab="assets"', 'data-tab="deviation"'):
        assert probe in dash, f"{probe} が出ていない(注入経路を通っていない)"


def test_canonical_run_build_data_identical_vs_head(tmp_path):
    """build_data の返す dict も旧ランで HEAD と完全一致(payload バイト不変の根拠)。"""
    if HEAD is None:
        pytest.skip("git 不在")
    rd = _write_run(tmp_path, "bd_same", n_steps=10)
    cur = mv.build_data(rd, include_traffic=False, include_moves=True)
    old = HEAD.build_data(rd, include_traffic=False, include_moves=True)
    assert json.dumps(cur, ensure_ascii=False) == json.dumps(old, ensure_ascii=False)
    assert json.dumps(mv.build_rollup_data(rd), ensure_ascii=False) == \
        json.dumps(HEAD.build_rollup_data(rd), ensure_ascii=False)


# ============================================================ 4. Δt=1 で時刻が正しい
def test_run_step_minutes_reads_config(tmp_path):
    rd1 = _write_run(tmp_path, "dt1", dt_min=1, rich=False)
    rd10 = _write_run(tmp_path, "dt10", rich=False)
    assert mv._run_step_minutes(rd1) == 1
    assert mv._run_step_minutes(rd10) == 10          # dt_min 未記載 = 正準 10 を仮定


def test_derive_start_min_uses_dt():
    """sim_min = start_min + step*Δt の不変量。Δt を渡さなければ従来(10)のまま。"""
    ev = [{"step": 30, "sim_min": 570}]              # Δt=1 なら start=540 / Δt=10 なら 270
    assert mv._derive_start_min(ev, 1) == 540
    assert mv._derive_start_min(ev, 10) == 270
    assert mv._derive_start_min(ev) == 270           # 既定 = 正準 10 = 従来同値


def test_dt1_run_startmin_and_clock(tmp_path):
    """Δt=1 のラン: startMin が 1 step=1 分で復元され、JS の時刻式も 1 分刻みになる。"""
    rd = _write_run(tmp_path, "dt1clock", dt_min=1, start_min=540, n_steps=12)
    data = mv.build_data(rd, include_traffic=False)
    assert data["startMin"] == 540                   # 09:00 開始(step*1 で逆算)
    out = _run_main(mv, rd)
    viewer = out["viewer.html"].decode("utf-8")
    dash = out["dashboard.html"].decode("utf-8")
    # 壁時計(地図の日付/時刻表示)と昼夜補間が 1 step = 1 分になっている
    assert "Math.floor((D.startMin+t*1)/1440)" in viewer
    assert "const m=((D.startMin+t*1)%1440+1440)%1440" in viewer
    assert "const mm=Math.floor(D.startMin+s*1);" in viewer      # tstr(フィード時刻)
    # 分析グラフの軸(日→step / 目盛)も Δt=1 の値
    assert "const stepsPerDay=1440;" in dash
    assert "const X=d=>d*1440;" in dash
    assert "Math.max(180," in dash and "Math.max(60," in dash
    # 旧来の直書き(1 step=10 分)がどこにも残っていない
    for text in (viewer, dash):
        assert not re.search(r"D\.startMin\s*\+\s*\w+\s*\*\s*10\b", text)


def test_dt1_rollup_uses_run_dt(tmp_path):
    """ロールアップの 1 日 = Δt から解く(データ・画面文言の両方)。"""
    rd1 = _write_run(tmp_path, "dt1roll", dt_min=1, n_steps=6)
    assert mv.build_rollup_data(rd1)["stepsPerDay"] == 1440
    html = _run_main(mv, rd1, "--daily-rollup")["rollup.html"].decode("utf-8")
    assert "1日(1440step=1,440分)" in html
    rd10 = _write_run(tmp_path, "dt10roll", n_steps=6)
    assert mv.build_rollup_data(rd10)["stepsPerDay"] == 144
    html10 = _run_main(mv, rd10, "--daily-rollup")["rollup.html"].decode("utf-8")
    assert "1日(144step=1,440分)" in html10


def test_ev_day_fallback_uses_steps_per_day():
    """sim_min 欠損時の day は step//(1440/Δt)。既定は正準 144(従来同値)。"""
    e = {"step": 200, "sim_min": None}
    assert mv._ev_day(e) == 1                        # 200 // 144
    assert mv._ev_day(e, mv._steps_per_day(1)) == 0  # 200 // 1440
    assert mv._ev_day(e, mv._steps_per_day(60)) == 8  # 200 // 24


def test_indoor_contacts_step_uses_dt(tmp_path):
    """屋内 contacts の t_s(秒)→ step 換算が Δt に追従する。"""
    rd = _write_run(tmp_path, "dtcontact", dt_min=1, n_steps=4)
    pq.write_table(pa.table({"t_s": [125.0], "building": ["b1"], "floor": [2],
                             "id_a": [0], "id_b": [1]}),
                   rd / "indoor_tracks_contacts.parquet")
    ev = [{"step": 0, "agent_id": 0, "kind": "space_move", "payload":
           json.dumps({"building": "b1", "floor": 2, "to_zone": 1})}]
    idx = {0: 0, 1: 1}
    cfg = {"indoor": {}}
    d1 = mv.build_indoor_data(ev, idx, {"b1": 0}, rd, 4, cfg, False, 1)
    d10 = mv.build_indoor_data(ev, idx, {"b1": 0}, rd, 4, cfg, False, 10)
    assert d1["contacts"][0][0] == 2                 # 125s // 60s
    assert d10["contacts"][0][0] == 0                # 125s // 600s(= 従来)


def test_indoor_tracks_day_guard_uses_dt(tmp_path):
    """軌跡ガードは「7日以下」= step 数ではなく Δt で決まる日数で切る。"""
    samples = tmp_path / "indoor_tracks_samples.parquet"
    pq.write_table(pa.table({"agent_id": [0, 0], "building": ["b1", "b1"],
                             "floor": [2, 2], "t_s": [1.0, 2.0],
                             "x": [0.0, 1.0], "y": [0.0, 1.0]}), samples)
    n = 7 * 144 + 1                                  # Δt=10 では 7 日超 / Δt=1 では 1 日未満
    note, tracks = mv._build_indoor_tracks(samples, {0: 0}, {"b1": 0}, n)
    assert tracks is None and "7日以下" in note      # 既定(正準 144)= 従来と同じ判定
    note1, tracks1 = mv._build_indoor_tracks(samples, {0: 0}, {"b1": 0}, n,
                                             mv._steps_per_day(1))
    assert note1 == "" and tracks1 and len(tracks1[0]["pts"]) == 2
