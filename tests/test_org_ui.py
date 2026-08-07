"""viz/make_viewer.py の会社(組織)UI B7 の検証。

検収条件:
- 後方互換: 旧ラン(org データ無し)に対する生成 HTML がバイト同一(HEAD end-to-end 比較)。
  旧ランの build_data に orgs キーが増えない(payload バイト不変の根拠)。
- org データ有ラン(agents.json org_id / serve.org_id / org_ledger.parquet のいずれか)で
  D.orgs が生成され、会社名簿・従業員・日次系列が入る。
- org card 検算: card に出す production/serve が L1 集計(groupby org_id)と一致。
- org_ledger.parquet があればその値(revenue_est/wage_paid/attendance_min 等)を系列に使う。
- HTML 注入: org 有ランのみ 🏢会社タブ / orgRender / colorBy='org' が入り、旧ランには入らない。
  DASH_HTML/MAP_HTML の文字列は無改変(既存 viewer 系テストが無修正緑)。
- ETHICS(R17): 会社名は架空の合成台帳の名称のみ(実在企業名を生成しない)。

全経路 mock/合成データのみ(実 LLM 不使用)。B4(org データ層)未完のため serve.org_id が
無い経路(agents.json org_id で staff→org 束ね)も検証する。
"""
from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "viz"))

import make_viewer as mv  # noqa: E402


# --------------------------------------------------------------------------- 合成ラン
def _minimal_map(path: Path) -> None:
    city = {
        "buildings": [
            {"id": "b1", "name": "合成オフィスA", "kind": "office", "levels": 6,
             "footprint": [[0, 0], [120, 0], [120, 90], [0, 90]]},
            {"id": "b2", "name": "合成リテールB", "kind": "retail", "levels": 3,
             "footprint": [[200, 0], [260, 0], [260, 80], [200, 80]]}],
        "nodes": [{"id": "n1", "x": 0, "y": 0, "name": "ノードA"},
                  {"id": "n2", "x": 10, "y": 10}],
        "edges": [{"klass": "footway", "layer": 0, "geometry": [[0, 0], [10, 10]]}],
        "pois": [], "railways": [], "meta": {"origin_latlon": [35.66, 139.70]},
    }
    path.write_text(json.dumps(city, ensure_ascii=False), encoding="utf-8")


def _org_book(path: Path) -> None:
    """架空の合成台帳(R17)。id は agents.json の org_id と一致させる。"""
    book = {"meta": {"note": "synthetic test book"},
            "companies": [
                {"id": "co_test_1", "name": "架空コネクト社", "industry": "情報通信業",
                 "industry_key": "IT", "roles": ["エンジニア", "営業"],
                 "workplace_poi": {"cat": "office", "building": "b1", "node": "n1",
                                   "floor": 4, "x": 10.0, "y": 10.0}},
                {"id": "co_test_2", "name": "架空マート社", "industry": "小売業",
                 "industry_key": "RETAIL", "roles": ["販売"],
                 "workplace_poi": {"cat": "retail", "building": "b2", "node": "n2",
                                   "floor": 1, "x": 210.0, "y": 10.0}}],
            "schools": []}
    path.write_text(json.dumps(book, ensure_ascii=False), encoding="utf-8")


def _write_run(tmp_path: Path, name: str, *, org_on: bool,
               serve_org_id: bool = False, ledger: bool = False,
               n_steps: int = 24) -> Path:
    """合成ラン。org_on=True なら agents.json に org_id/org_role + production/serve を書く。"""
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True, exist_ok=True)
    map_path = tmp_path / f"{name}_map.json"
    _minimal_map(map_path)
    book_path = tmp_path / f"{name}_orgbook.json"
    _org_book(book_path)
    (run_dir / "config.yaml").write_text(
        "world:\n"
        f"  map: {map_path.as_posix()}\n"
        "transit:\n"
        "  file: data/__no_transit_for_test__.json\n"
        "indoor:\n"
        "  floor_layouts_path: data/floor_layouts.json\n"
        "organizations:\n"
        "  enabled: true\n"
        f"  book: {book_path.as_posix()}\n",
        encoding="utf-8")
    # 4人: 0,1=co_test_1 / 2=co_test_2 / 3=無所属
    org_of = {0: "co_test_1", 1: "co_test_1", 2: "co_test_2"}
    role_of = {0: "エンジニア", 1: "営業", 2: "販売"}
    agents = []
    for i in range(4):
        a = {"id": i, "name": f"住民{i}", "age": 30 + i, "gender": "男",
             "occupation": "会社員", "visitor": False,
             "has_bicycle": False, "has_car": False, "work_name": ""}
        if org_on and i in org_of:
            a["org_id"] = org_of[i]
            a["org_role"] = role_of[i]
        agents.append(a)
    (run_dir / "agents.json").write_text(json.dumps(agents, ensure_ascii=False),
                                         encoding="utf-8")

    rows = []

    def ev(step, aid, kind, payload, x=0.0, y=0.0):
        rows.append({"step": step, "agent_id": aid, "kind": kind,
                     "sim_min": 420 + step * mv.STEP_MINUTES, "x": x, "y": y,
                     "payload": json.dumps(payload, ensure_ascii=False),
                     "rng_stream": "", "llm_call_id": ""})

    for step in range(n_steps):
        for a in range(4):
            ev(step, a, "arrive", {"name": "路上", "node": "n1"}, x=float(a))
    if org_on:
        # production(payload.org=台帳 id): co_test_1 に 3件・co_test_2 に 1件
        ev(2, 0, "production", {"org": "co_test_1", "output": "app", "kind": "software"})
        ev(3, 1, "production", {"org": "co_test_1", "output": "app", "kind": "software"})
        ev(5, 0, "production", {"org": "co_test_1", "output": "app", "kind": "software"})
        ev(4, 2, "production", {"org": "co_test_2", "output": "sale", "kind": "retail"})
        # serve: co_test_1 の staff(agent 0)へ 2件 + unstaffed 1件
        if serve_org_id:
            ev(6, 0, "serve", {"cat": "food", "org_id": "co_test_1", "node": "n1"})
            ev(7, 0, "serve", {"cat": "food", "org_id": "co_test_1", "node": "n1"})
        else:
            ev(6, 0, "serve", {"cat": "food", "label": "接客", "customer": 3, "node": "n1"})
            ev(7, 0, "serve", {"cat": "food", "label": "接客", "customer": 3, "node": "n1"})
        ev(8, -1, "serve", {"cat": "food", "node": "n2", "unstaffed": True})
        # 屋内(occupancy 用の enter_building)+ space_move
        ev(1, 0, "enter_building", {"building": "b1", "floor": 4})
        ev(2, 0, "space_move", {"building": "b1", "floor": 4, "to_zone": 1})

    fields = [("step", pa.int32()), ("sim_min", pa.int32()),
              ("agent_id", pa.int32()), ("kind", pa.string()),
              ("x", pa.float32()), ("y", pa.float32()), ("payload", pa.string()),
              ("rng_stream", pa.string()), ("llm_call_id", pa.string())]
    cols = {nm: [r[nm] for r in rows] for nm, _ in fields}
    pq.write_table(pa.table(cols, schema=pa.schema(fields)),
                   run_dir / "l1_events.parquet")

    if ledger:
        led = {"day": [0, 0], "org_id": ["co_test_1", "co_test_2"],
               "production": [3, 1], "revenue_est": [4500.0, 1200.0],
               "wage_paid": [3000.0, 900.0], "serve_count": [2, 0],
               "attendance_min": [960, 480]}
        pq.write_table(pa.table(led, schema=pa.schema([
            ("day", pa.int32()), ("org_id", pa.string()), ("production", pa.int32()),
            ("revenue_est", pa.float64()), ("wage_paid", pa.float64()),
            ("serve_count", pa.int32()), ("attendance_min", pa.int32())])),
            run_dir / "org_ledger.parquet")
    return run_dir


# --------------------------------------------------------------------------- HEAD 版
def _load_head_module():
    try:
        src = subprocess.check_output(
            ["git", "show", "HEAD:viz/make_viewer.py"],
            cwd=REPO_ROOT, stderr=subprocess.DEVNULL).decode("utf-8")
    except Exception:
        return None
    mod = types.ModuleType("mv_head_org")
    mod.__file__ = str(REPO_ROOT / "viz" / "make_viewer.py")
    try:
        exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    except Exception:
        return None
    return mod


HEAD = _load_head_module()


# ============================================================ 1. 後方互換(バイト同一)
def test_old_run_html_byte_identical_end_to_end(tmp_path):
    """旧ラン(org 無し): HEAD の make_viewer と現行が生成する viewer/dashboard が byte 一致。"""
    src = subprocess.run(["git", "show", "HEAD:viz/make_viewer.py"],
                         cwd=REPO_ROOT, capture_output=True)
    if src.returncode != 0:
        pytest.skip("git 不在(HEAD 版を取れない)")
    head_py = tmp_path / "make_viewer_head.py"
    head_py.write_bytes(src.stdout)
    rd = _write_run(tmp_path, "old_e2e", org_on=False, n_steps=6)

    def _run(script: Path):
        r = subprocess.run([sys.executable, str(script), str(rd)],
                           cwd=REPO_ROOT, capture_output=True,
                           env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
        assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
        return ((rd / "viewer.html").read_bytes(),
                (rd / "dashboard.html").read_bytes())

    head_v, head_d = _run(head_py)
    cur_v, cur_d = _run(REPO_ROOT / "viz" / "make_viewer.py")
    assert cur_v == head_v, "viewer.html が旧ランでバイト不一致(後方互換違反)"
    assert cur_d == head_d, "dashboard.html が旧ランでバイト不一致(後方互換違反)"


def test_old_run_build_data_no_org_keys(tmp_path):
    """旧ランの build_data に orgs/occupancy キーが増えない(payload バイト不変の根拠)。"""
    rd = _write_run(tmp_path, "old_bd", org_on=False, n_steps=6)
    data = mv.build_data(rd, include_traffic=False)
    assert "orgs" not in data and "occupancy" not in data
    if HEAD is not None:
        head_data = HEAD.build_data(rd, include_traffic=False)
        assert json.dumps(data, ensure_ascii=False) == \
            json.dumps(head_data, ensure_ascii=False)


def _canon_dt(tpl: str) -> str:
    """Δt トークン(W4-C)を正準 Δt=10 の値へ畳んだテンプレート文字列。

    W4-C で JS の時間換算の直書き(1 step=10 分 / 1 日=144 step / 目盛 6・18 step)を
    `__STEP_MIN__` ほかのトークンへ置き換えた。**正準 Δt=10 では同じ文字列に戻る**ので、
    ここで畳んでから比較すれば「Δt トークン化以外の改変は 1 文字も無い」を従来どおり検知できる
    (トークンが入る前の HEAD にも、入った後の HEAD にも、同じ判定が効く)。"""
    for tok, val in mv.dt_token_values(10).items():
        tpl = tpl.replace(tok, val)
    return tpl


def test_templates_unmodified_vs_head():
    """DASH_HTML/MAP_HTML を(Δt トークン化を除いて)一切改変していない。

    出力そのもののバイト同一は test_old_run_html_byte_identical_end_to_end が別途固定する。"""
    if HEAD is None:
        pytest.skip("git 不在")
    assert _canon_dt(mv.DASH_HTML) == _canon_dt(HEAD.DASH_HTML), \
        "DASH_HTML を改変している(テンプレ無改変が崩れる)"
    assert _canon_dt(mv.MAP_HTML) == _canon_dt(HEAD.MAP_HTML), \
        "MAP_HTML を改変している(テンプレ無改変が崩れる)"


# ============================================================ 2. org データ生成
def test_org_run_embeds_orgs(tmp_path):
    rd = _write_run(tmp_path, "org_embed", org_on=True)
    data = mv.build_data(rd, include_traffic=False)
    assert "orgs" in data
    O = data["orgs"]
    assert O["source"] == "l1"
    ids = {o["id"] for o in O["list"]}
    assert ids == {"co_test_1", "co_test_2"}
    o1 = next(o for o in O["list"] if o["id"] == "co_test_1")
    assert o1["name"] == "架空コネクト社" and o1["cat"] == "情報通信業"
    assert o1["building"] == "b1" and o1["floor"] == 4
    assert o1["n_emp"] == 2 and sorted(o1["employees"]) == [0, 1]
    assert o1["roles"] == {"エンジニア": 1, "営業": 1}


def test_org_ethics_only_synthetic_names(tmp_path):
    """会社名は合成台帳の名称のみ(実在企業名を出さない=R17)。"""
    rd = _write_run(tmp_path, "org_ethics", org_on=True)
    O = mv.build_data(rd, include_traffic=False)["orgs"]
    names = {o["name"] for o in O["list"]}
    assert names == {"架空コネクト社", "架空マート社"}


# ============================================================ 3. org card 検算(L1 集計一致)
def _l1_groupby(rd: Path, agents_json: Path):
    """L1 を独立に groupby(org_id): production=payload.org / serve=payload.org_id or staff→org。"""
    agents = json.loads(agents_json.read_text(encoding="utf-8"))
    agent_org = {a["id"]: a["org_id"] for a in agents if a.get("org_id")}
    rows = pq.read_table(rd / "l1_events.parquet").to_pylist()
    prod, serve = {}, {}
    for e in rows:
        p = json.loads(e["payload"]) if e["payload"] else {}
        if e["kind"] == "production" and p.get("org"):
            prod[p["org"]] = prod.get(p["org"], 0) + 1
        elif e["kind"] == "serve":
            oid = p.get("org_id") or agent_org.get(e["agent_id"])
            if oid:
                serve[oid] = serve.get(oid, 0) + 1
    return prod, serve


@pytest.mark.parametrize("serve_org_id", [False, True])
def test_org_card_checksum_matches_l1(tmp_path, serve_org_id):
    """card の production/serve が L1 の groupby(org_id) と一致(ledger 無し=L1 再構成)。"""
    rd = _write_run(tmp_path, f"org_ck_{serve_org_id}", org_on=True,
                    serve_org_id=serve_org_id)
    O = mv.build_data(rd, include_traffic=False)["orgs"]
    prod, serve = _l1_groupby(rd, rd / "agents.json")
    for o in O["list"]:
        assert o["prod"] == prod.get(o["id"], 0), f"prod 不一致 {o['id']}"
        assert o["serve"] == serve.get(o["id"], 0), f"serve 不一致 {o['id']}"
        # 系列合計も一致
        assert sum(O["series"][o["id"]]["production"]) == prod.get(o["id"], 0)
        assert sum(O["series"][o["id"]]["serve"]) == serve.get(o["id"], 0)
    # 期待値(serve は staff=agent0→co_test_1 の 2件・unstaffed は org 帰属なし)
    assert prod == {"co_test_1": 3, "co_test_2": 1}
    assert serve == {"co_test_1": 2}


# ============================================================ 4. org_ledger 経路
def test_org_ledger_path_uses_ledger_values(tmp_path):
    rd = _write_run(tmp_path, "org_led", org_on=True, ledger=True)
    O = mv.build_data(rd, include_traffic=False)["orgs"]
    assert O["source"] == "ledger"
    s1 = O["series"]["co_test_1"]
    assert sum(s1["production"]) == 3 and sum(s1["serve"]) == 2
    assert "revenue_est" in s1 and sum(s1["revenue_est"]) == 4500.0
    assert "wage_paid" in s1 and "attendance_min" in s1


def test_org_ledger_triggers_ui_without_agent_org(tmp_path):
    """org_ledger.parquet だけ(agents.json に org_id 無し)でも会社 UI が出る(graceful)。"""
    rd = _write_run(tmp_path, "org_led_only", org_on=False, ledger=True)
    data = mv.build_data(rd, include_traffic=False)
    assert "orgs" in data and data["orgs"]["source"] == "ledger"
    ids = {o["id"] for o in data["orgs"]["list"]}
    assert ids == {"co_test_1", "co_test_2"}


# ============================================================ 5. HTML 注入(タブ/colorBy)
def _gen_html(rd: Path):
    r = subprocess.run([sys.executable, str(REPO_ROOT / "viz" / "make_viewer.py"), str(rd)],
                       cwd=REPO_ROOT, capture_output=True,
                       env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    return ((rd / "viewer.html").read_text(encoding="utf-8"),
            (rd / "dashboard.html").read_text(encoding="utf-8"))


def test_org_tab_and_colorby_injected_only_for_org_runs(tmp_path):
    rd = _write_run(tmp_path, "org_html", org_on=True)
    viewer, dash = _gen_html(rd)
    # dashboard: 会社タブ + orgRender + render ラッパ
    assert 'data-tab="org"' in dash and "function orgRender" in dash
    assert "__orgOccWrapped" in dash
    # viewer(地図): colorBy='org' オプション + orgMapColor + hook
    assert '<option value="org">' in viewer and "function orgMapColor" in viewer
    assert "if(mode==='org') return orgMapColor(i, s0);" in viewer
    for tok in ("__LENS_TABS__", "__LENS_JS__", "__COMMUNITY_OPTION__",
                "__COMMUNITY_HOOK__", "__COMMUNITY_JS__"):
        assert tok not in dash and tok not in viewer

    rd2 = _write_run(tmp_path, "org_html_off", org_on=False)
    viewer2, dash2 = _gen_html(rd2)
    assert 'data-tab="org"' not in dash2 and "function orgRender" not in dash2
    assert '<option value="org">' not in viewer2 and "function orgMapColor" not in viewer2
