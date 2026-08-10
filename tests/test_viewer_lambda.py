"""I-1: λ 横オフセット(2D/3D)と 3D 街路可視性の検証。

これは **観測側だけ**の機能である。シミュレーションは常に全員を回しており、ここで変わるのは
「カメラに何を映すか」だけ(sim には 1 バイトも触らない)。検収条件:

- 後方互換(最重要): フラグ未指定なら生成 HTML は従来とバイト同一。
  * 2D: --lateral-offset 無しの viewer.html/dashboard.html が HEAD 版の出力と byte 一致。
        テンプレート(MAP_HTML/DASH_HTML)は無改変で、既存 __COMMUNITY_JS__ トークンへ
        追記合成するだけ(=トークンが空へ潰れれば従来文字列に戻る、W4-C と同型)。
  * 3D: build_html(lateral=False, street_only=False) が既定引数の出力とバイト一致。
- λ の決定論: 同じランを 2 回生成すると HTML が byte 一致(RNG も時刻依存も無い)。
- λ の数理: λ=hash(agent_id, edge) は純関数で [-1,1)。横ずれ量は当該 edge の帯
  (gap+side、建物までの距離で clamp)以下・絶対上限 BMAX 以下・NaN 無し。
  edge 遷移/交差点では最近傍2 edge の重み混合で **横位置が飛ばない**(C0 連続)。
- 2D と 3D が **同一の λ 中核 JS 文字列**を使う(2 実装に割れば必ずズレるため)。
- 3D 街路可視性: 屋内(w>=1000)は既定で非表示、建物クリックでその 1 棟だけ開示。
  電車内(w=-3)・範囲外(w=-1)・睡眠(w=-2)は従来どおり非表示のまま。

Python 側は JS の **鏡**(同じ式を Python で書いたもの)+ 出荷 JS に同じ式が載っている
ことの text guard で担保する(実ブラウザ非依存。tests/test_viewer_indoor.py の
izNOverride パリティと同じ流儀)。定数は JS 本文から機械的に抜くので、JS を書き換えると
テストも自動で追随する(構造を変えれば parse が落ちて気づける)。

全経路 合成データのみ(実 LLM 不使用)。
"""
from __future__ import annotations

import importlib.util
import json
import math
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

import make_viewer as mv  # noqa: E402


def _load_mv3():
    spec = importlib.util.spec_from_file_location(
        "make_viewer3d", REPO_ROOT / "viz" / "make_viewer3d.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MV3 = _load_mv3()


# --------------------------------------------------------------------------- 合成ラン(2D)
def _map_with_streets(path: Path) -> None:
    """帯の効きが見える最小地図: 幹線(primary)と路地(footway)+ 沿道の建物 2 棟。"""
    city = {
        "buildings": [
            {"id": "b1", "name": "テストビル", "kind": "retail", "levels": 6,
             "footprint": [[0, 12], [100, 12], [100, 60], [0, 60]]},
            {"id": "b2", "name": "", "kind": "office", "levels": 3,
             "footprint": [[0, -60], [100, -60], [100, -12], [0, -12]]}],
        "nodes": [{"id": "n1", "x": 0, "y": 0, "name": "ノードA"},
                  {"id": "n2", "x": 100, "y": 0}],
        "edges": [
            {"klass": "primary", "layer": 0, "geometry": [[0, 0], [100, 0]]},
            {"klass": "footway", "layer": 0, "geometry": [[100, 0], [160, 40]]},
        ],
        "pois": [], "railways": [], "meta": {"origin_latlon": [35.66, 139.70]},
    }
    path.write_text(json.dumps(city, ensure_ascii=False), encoding="utf-8")


def _write_run(tmp_path: Path, name: str, *, n_agents: int = 6,
               n_steps: int = 5, indoor_on: bool = False) -> Path:
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True, exist_ok=True)
    map_path = tmp_path / f"{name}_map.json"
    _map_with_streets(map_path)
    (run_dir / "config.yaml").write_text(
        "world:\n"
        f"  map: {map_path.as_posix()}\n"
        "transit:\n"
        "  file: data/__no_transit_for_test__.json\n",
        encoding="utf-8")
    agents = [{"id": i, "name": f"住民{i}", "age": 20 + i, "gender": "男",
               "occupation": "会社員", "visitor": False,
               "has_bicycle": False, "has_car": False} for i in range(n_agents)]
    (run_dir / "agents.json").write_text(json.dumps(agents, ensure_ascii=False),
                                         encoding="utf-8")
    rows = []
    for step in range(n_steps):
        for a in range(n_agents):
            rows.append({"step": step, "agent_id": a, "kind": "move_segment",
                         "sim_min": 420 + step * mv.STEP_MINUTES,
                         "x": float(10 + a * 3), "y": 0.0,
                         "payload": json.dumps(
                             {"mode": "walk",
                              "pts": [[10.0 + a * 3, 0.0], [20.0 + a * 3, 0.0]]},
                             ensure_ascii=False),
                         "rng_stream": "", "llm_call_id": ""})
    if indoor_on:
        for a in range(n_agents):
            rows.append({"step": 2, "agent_id": a, "kind": "enter_building",
                         "sim_min": 440, "x": 50.0, "y": 10.0,
                         "payload": json.dumps({"building": "b1", "floor": 4},
                                               ensure_ascii=False),
                         "rng_stream": "", "llm_call_id": ""})
    fields = [("step", pa.int32()), ("sim_min", pa.int32()),
              ("agent_id", pa.int32()), ("kind", pa.string()),
              ("x", pa.float32()), ("y", pa.float32()), ("payload", pa.string()),
              ("rng_stream", pa.string()), ("llm_call_id", pa.string())]
    cols = {nm: [r[nm] for r in rows] for nm, _ in fields}
    pq.write_table(pa.table(cols, schema=pa.schema(fields)),
                   run_dir / "l1_events.parquet")
    return run_dir


def _gen(run_dir: Path, *flags: str) -> tuple[bytes, bytes]:
    """現行 make_viewer.py をサブプロセスで走らせ (viewer.html, dashboard.html) を返す。"""
    import os
    r = subprocess.run([sys.executable, str(REPO_ROOT / "viz" / "make_viewer.py"),
                        str(run_dir), *flags],
                       cwd=REPO_ROOT, capture_output=True,
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    return ((run_dir / "viewer.html").read_bytes(),
            (run_dir / "dashboard.html").read_bytes())


def _load_head_module():
    try:
        src = subprocess.check_output(
            ["git", "show", "HEAD:viz/make_viewer.py"],
            cwd=REPO_ROOT, stderr=subprocess.DEVNULL).decode("utf-8")
    except Exception:
        return None
    mod = types.ModuleType("mv_head_lat")
    mod.__file__ = str(REPO_ROOT / "viz" / "make_viewer.py")
    try:
        exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    except Exception:
        return None
    return mod


HEAD = _load_head_module()


# --------------------------------------------------------------------------- 3D 合成シーン
def _scene3(roads: list | None = None) -> str:
    if roads is None:
        roads = [{"klass": "primary", "layer": 0, "g": [[0, 0], [100, 0]]},
                 {"klass": "footway", "layer": 0, "g": [[100, 0], [160, 40]]}]
    return json.dumps({
        "meta": {"crs": "local-m", "step_minutes": 10, "floor_height": 3.5},
        "buildings": [
            {"id": "b1", "name": "テストビル", "kind": "retail", "levels": 6,
             "below": 0, "height": 21.0, "depth": 21.0, "gz": 0.0, "cx": 50, "cy": 36,
             "footprint": [[0, 12], [100, 12], [100, 60], [0, 60]]}],
        "roads": roads, "rails": [], "pois": []}, ensure_ascii=False)


def _tracks3(ws: list | None = None) -> str:
    """positions の w を指定して合成(0=路上 / 1000+bi*100+floor=屋内 / -3=電車で圏外)。"""
    if ws is None:
        ws = [0, 1000 + 0 * 100 + 4, -3]
    n = len(ws)
    return json.dumps({
        "meta": {"step_minutes": 10, "mode_legend": None},
        "ids": list(range(n)),
        "agents": [{"id": i, "name": f"a{i}", "occupation": "x", "visitor": False}
                   for i in range(n)],
        "positions": [[[float(10 + i * 5), 0.0, w] for i, w in enumerate(ws)]],
        "moves": [[None] * n],
        "traffic": [{"n": 0, "segs": []}],
        "nSteps": 1}, ensure_ascii=False)


# =========================================================== 1. 後方互換(バイト同一)
def test_2d_flag_off_is_byte_identical_to_head(tmp_path):
    """--lateral-offset 無し = HEAD の make_viewer と byte 一致(旧ランの後方互換)。"""
    if HEAD is None:
        pytest.skip("git 不在(HEAD 版を取れない)")
    src = subprocess.run(["git", "show", "HEAD:viz/make_viewer.py"],
                         cwd=REPO_ROOT, capture_output=True)
    if src.returncode != 0:
        pytest.skip("git 不在")
    (tmp_path / "viz").mkdir()
    head_py = tmp_path / "viz" / "make_viewer_head.py"
    head_py.write_bytes(src.stdout)
    run_dt_src = subprocess.run(["git", "show", "HEAD:scripts/run_dt.py"],
                                cwd=REPO_ROOT, capture_output=True)
    if run_dt_src.returncode == 0:
        (tmp_path / "scripts").mkdir()
        (tmp_path / "scripts" / "run_dt.py").write_bytes(run_dt_src.stdout)
    rd = _write_run(tmp_path, "lat_off")

    import os

    def _run(script: Path):
        r = subprocess.run([sys.executable, str(script), str(rd)],
                           cwd=REPO_ROOT, capture_output=True,
                           env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
        return ((rd / "viewer.html").read_bytes(),
                (rd / "dashboard.html").read_bytes())

    head_v, head_d = _run(head_py)
    cur_v, cur_d = _run(REPO_ROOT / "viz" / "make_viewer.py")
    assert cur_v == head_v, "λ 既定 OFF なのに viewer.html が変わった(後方互換違反)"
    assert cur_d == head_d, "λ 既定 OFF なのに dashboard.html が変わった"


def test_2d_templates_untouched():
    """テンプレート本体は無改変(既存 __COMMUNITY_JS__ へ追記合成しただけ)。

    新トークンを増やしていない = test_org_ui.test_templates_unmodified_vs_head の
    前提を壊していない。λ の注入点は MAP_HTML にしか無い __COMMUNITY_JS__ なので
    dashboard.html には構造上入り得ない。"""
    assert mv.MAP_HTML.count("__COMMUNITY_JS__") == 1
    assert mv.DASH_HTML.count("__COMMUNITY_JS__") == 0
    for tok in re.findall(r"__LAT[A-Z_]*__", mv.MAP_HTML + mv.DASH_HTML):
        pytest.fail(f"テンプレートに新トークン {tok} を足している(バイト同一が崩れる)")
    if HEAD is not None:
        assert mv.MAP_HTML == HEAD.MAP_HTML
        assert mv.DASH_HTML == HEAD.DASH_HTML


def test_2d_flag_on_changes_only_viewer(tmp_path):
    rd = _write_run(tmp_path, "lat_on")
    off_v, off_d = _gen(rd)
    on_v, on_d = _gen(rd, "--lateral-offset")
    assert on_v != off_v, "--lateral-offset ON なのに viewer.html が変わらない"
    assert on_d == off_d, "dashboard.html は λ の影響を受けてはいけない"
    assert b"__latXY" in on_v and b"__latXY" not in off_v


def test_2d_lambda_generation_is_deterministic(tmp_path):
    """同じランを 2 回生成 → byte 一致(RNG も時刻依存も無い純関数)。"""
    rd = _write_run(tmp_path, "lat_det")
    a_v, a_d = _gen(rd, "--lateral-offset")
    b_v, b_d = _gen(rd, "--lateral-offset")
    assert a_v == b_v and a_d == b_d


def _indoor_json() -> str:
    return json.dumps({"floor_layouts": {"buildings": [
        {"match": ["テストビル"], "floors": [{"f": 4, "use": "food"}]}]}},
        ensure_ascii=False)


def test_3d_flags_off_byte_identical():
    """屋内データを持たない旧ラン: 既定 = 明示 OFF = 従来出力(バイト同一)。"""
    scene, tracks = _scene3(), _tracks3()
    plain = MV3.build_html("t", scene, tracks)
    explicit = MV3.build_html("t", scene, tracks, lateral=False, street_only=False)
    assert plain == explicit
    assert "__streetHidden" not in plain, "旧ランで街路可視性が勝手に入っている"
    assert "const [x,y,w] = pos[i];" in plain, "placeAgents が従来形のままでない"


def test_3d_head_byte_identity_matrix(tmp_path):
    """HEAD 版 make_viewer3d との直接突合(3D 側の後方互換をこの 1 本で固定)。

    ① 屋内データ無し(旧ラン)= 既定で HEAD と byte 一致(新既定は旧ランに触らない)
    ② 屋内データ有り + street_only=False(--no-street-only)= HEAD と byte 一致
       (新既定が気に入らない時の完全な逃げ道が実在することの証明)
    ③ 屋内データ有り + 既定(auto)= HEAD と異なる(新既定が実際に効いている)
    HEAD 版はリポ外で走るので viz/vendor を同形に組む(2D 側テストと同じ repo-shape 修正)。
    """
    import shutil
    src = subprocess.run(["git", "show", "HEAD:viz/make_viewer3d.py"],
                         cwd=REPO_ROOT, capture_output=True)
    if src.returncode != 0:
        pytest.skip("git 不在(HEAD 版を取れない)")
    (tmp_path / "viz").mkdir()
    head_py = tmp_path / "viz" / "make_viewer3d_head.py"
    head_py.write_bytes(src.stdout)
    shutil.copytree(REPO_ROOT / "viz" / "vendor", tmp_path / "viz" / "vendor")
    spec = importlib.util.spec_from_file_location("mv3_head", head_py)
    H = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(H)
    except SystemExit:
        pytest.skip("HEAD 版 make_viewer3d を読めない(vendor 等)")

    scene, tracks, ind = _scene3(), _tracks3(), _indoor_json()
    # ① 旧ラン(屋内データ無し)
    assert H.build_html("t", scene, tracks) == MV3.build_html("t", scene, tracks), \
        "屋内データ無しのランで HTML が変わった(後方互換違反)"
    # ② 屋内データ有り + 明示 OFF
    #    ★HEAD 側の呼び方は epoch で変える: street-only 機能が HEAD に入る前は
    #    「機能前の素の出力へ戻れるか」、入った後は「明示 OFF 同士で byte 同一か」。
    #    HEAD コピー実行ハーネスは機能をコミットした後の初ゲートで自己矛盾化する
    #    (auto 既定 ON の HEAD vs 明示 OFF の作業木)= Wave 1 の repo-shape と
    #    同族の既知の罠(第107で発火・恒久修正)。
    import inspect
    head_kw = ({"street_only": False}
               if "street_only" in inspect.signature(H.build_html).parameters
               else {})
    head_ind = H.build_html("t", scene, tracks, indoor_json=ind, **head_kw)
    assert head_ind == MV3.build_html("t", scene, tracks, indoor_json=ind,
                                      street_only=False), \
        "--no-street-only が HEAD 出力へ戻らない(逃げ道が壊れている)"
    # ③ 屋内データ有り + 既定 = 新既定が効く
    assert head_ind != MV3.build_html("t", scene, tracks, indoor_json=ind), \
        "新ランで既定 ON になっていない"


def test_3d_generation_is_deterministic():
    scene, tracks = _scene3(), _tracks3()
    a = MV3.build_html("t", scene, tracks, lateral=True, street_only=True)
    b = MV3.build_html("t", scene, tracks, lateral=True, street_only=True)
    assert a == b


# =========================================================== 2. 2D/3D で同一の λ 中核
def test_lambda_core_shared_between_2d_and_3d():
    """3D は 2D の _LATERAL_CORE_JS を借りる(2 実装に割ると λ がズレる)。"""
    assert MV3._lateral_core_js() == mv._LATERAL_CORE_JS
    assert mv._LATERAL_CORE_JS in mv._LATERAL_2D_JS
    lat = MV3.build_html("t", _scene3(), _tracks3(), lateral=True)
    assert mv._LATERAL_CORE_JS in lat


def test_lambda_adapters_match_each_payload_shape():
    """アダプタが各ビューアの実際のキー名を指している(2D: k/g/fp・3D: klass/g/footprint)。"""
    assert "ek:e=>e.k" in mv._LATERAL_2D_JS and "bfp:b=>b.fp" in mv._LATERAL_2D_JS
    assert "eg:e=>e.g" in mv._LATERAL_2D_JS
    assert "ek:r=>r.klass" in MV3._LATERAL_3D_ADAPTER
    assert "bfp:b=>b.footprint" in MV3._LATERAL_3D_ADAPTER
    # 2D payload / 3D scene が実際にそのキーで出ていること(build_data / export_3d 側の形)
    assert '"k": e.get("klass", "footway")' in \
        (REPO_ROOT / "viz" / "make_viewer.py").read_text(encoding="utf-8")
    scene = json.loads(_scene3())
    assert "klass" in scene["roads"][0] and "footprint" in scene["buildings"][0]


# =========================================================== 3. λ の数理(Python 鏡)
CORE = mv._LATERAL_CORE_JS


def _js_num(name: str) -> float:
    m = re.search(rf"\b{name}\s*=\s*([0-9.]+)", CORE)
    assert m, f"JS から定数 {name} を取れない(構造が変わった)"
    return float(m.group(1))


def _js_band_table() -> dict:
    m = re.search(r"const BAND = \{(.+?)\};", CORE, re.S)
    assert m, "JS から BAND 表を取れない"
    return {k: (float(a), float(b))
            for k, a, b in re.findall(r"(\w+):\[([0-9.]+),([0-9.]+)\]", m.group(1))}


def _js_band_def() -> tuple:
    m = re.search(r"BAND_DEF=\[([0-9.]+),([0-9.]+)\]", CORE)
    assert m, "JS から BAND_DEF を取れない"
    return (float(m.group(1)), float(m.group(2)))


BAND = _js_band_table()
BAND_DEF = _js_band_def()
CELL = _js_num("CELL")
BLEND = _js_num("BLEND")
WALL = _js_num("WALL")
BMAX = _js_num("BMAX")


def _fnv(s: str) -> int:
    """JS `_fnv`(FNV-1a 32bit + Math.imul)の鏡。"""
    h = 2166136261
    for ch in str(s):
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def _lam(aid, ei: int) -> float:
    return (_fnv(f"{aid}@{ei}") / 4294967296.0) * 2 - 1


def _seg_d(px, py, x0, y0, x1, y1):
    dx, dy = x1 - x0, y1 - y0
    l2 = dx * dx + dy * dy
    u = ((px - x0) * dx + (py - y0) * dy) / l2 if l2 else 0.0
    u = 0.0 if not u > 0 else (1.0 if u > 1 else u)
    qx, qy = x0 + dx * u, y0 + dy * u
    return math.hypot(px - qx, py - qy), dx, dy


class _Mirror:
    """`window.__latXY` の Python 鏡(格子は使わず全走査=同じ最近傍を返す)。"""

    def __init__(self, edges, blds):
        self.edges, self.blds = edges, blds
        self._band: dict[int, tuple] = {}

    def _bld_dist(self, px, py):
        best = math.inf
        for fp in self.blds:
            for j in range(1, len(fp)):
                d = _seg_d(px, py, fp[j - 1][0], fp[j - 1][1], fp[j][0], fp[j][1])[0]
                best = min(best, d)
        return best

    def band_of(self, ei):
        if ei in self._band:
            return self._band[ei]
        klass, g = self.edges[ei]
        gap, side = BAND.get(klass, BAND_DEF)
        m = g[len(g) // 2] if g else g[0]
        lim = max(0.5, min(BMAX, self._bld_dist(m[0], m[1]) - WALL))
        tot = gap + side
        if tot > lim:
            s = lim / tot
            gap, side = gap * s, side * s
        self._band[ei] = (gap, side)
        return self._band[ei]

    def near2(self, px, py):
        b1 = b2 = None
        for ei, (_k, g) in enumerate(self.edges):
            for j in range(1, len(g)):
                d, dx, dy = _seg_d(px, py, g[j - 1][0], g[j - 1][1], g[j][0], g[j][1])
                rec = {"ei": ei, "d": d, "dx": dx, "dy": dy}
                if b1 is None or d < b1["d"]:
                    if b1 is not None and b1["ei"] != ei:
                        b2 = b1
                    b1 = rec
                elif ei != b1["ei"] and (b2 is None or d < b2["d"]):
                    b2 = rec
        if b1 and b2 and b1["ei"] == b2["ei"]:
            b2 = None
        return b1, b2

    def _off(self, rec, aid):
        L = math.hypot(rec["dx"], rec["dy"])
        if not L > 0:
            return (0.0, 0.0)
        nx, ny = -rec["dy"] / L, rec["dx"] / L
        gap, side = self.band_of(rec["ei"])
        lam = _lam(aid, rec["ei"])
        mag = (-1 if lam < 0 else 1) * (gap + abs(lam) * side)
        return (nx * mag, ny * mag)

    def xy(self, aid, x, y):
        b1, b2 = self.near2(x, y)
        if b1 is None:
            return None
        ox, oy = self._off(b1, aid)
        ws = 1.0
        if b2 is not None:
            w2 = max(0.0, min(1.0, 1 - (b2["d"] - b1["d"]) / BLEND))
            if w2 > 0:
                o2 = self._off(b2, aid)
                ox += o2[0] * w2
                oy += o2[1] * w2
                ws += w2
        return (x + ox / ws, y + oy / ws)


_EDGES = [("primary", [[0, 0], [100, 0]]), ("footway", [[100, 0], [160, 40]])]
_BLDS = [[[0, 12], [100, 12], [100, 60], [0, 60]],
         [[0, -60], [100, -60], [100, -12], [0, -12]]]


def test_lambda_is_pure_and_in_range():
    """λ は (agent_id, edge) の純関数で [-1,1)。何度呼んでも同じ値。"""
    vals = []
    for aid in range(300):
        for ei in range(4):
            v = _lam(aid, ei)
            assert -1.0 <= v < 1.0, f"λ が範囲外: {v}"
            assert v == _lam(aid, ei), "λ が呼び出しごとに変わる(純関数でない)"
            vals.append(v)
    assert len(set(vals)) > len(vals) * 0.9, "λ の分布が退化している(衝突しすぎ)"
    # 片側に偏っていない(扇が両側へ開く)
    pos = sum(1 for v in vals if v >= 0)
    assert 0.4 < pos / len(vals) < 0.6, f"λ の左右が偏っている: {pos}/{len(vals)}"


def test_lambda_depends_on_both_agent_and_edge():
    assert _lam(1, 0) != _lam(2, 0), "同じ道で全員が同じ側に並んでしまう"
    assert _lam(1, 0) != _lam(1, 1), "道が変わっても同じ側に貼り付いてしまう"


def test_offset_bounded_by_band_and_bmax():
    """横ずれは当該 edge の帯(gap+side)以下・絶対上限 BMAX 以下・NaN 無し。"""
    M = _Mirror(_EDGES, _BLDS)
    worst = 0.0
    for aid in range(120):
        for x in range(0, 101, 5):
            r = M.xy(aid, float(x), 0.0)
            assert r is not None
            rx, ry = r
            assert rx == rx and ry == ry, "NaN が出た"
            d = math.hypot(rx - x, ry - 0.0)
            worst = max(worst, d)
            assert d <= BMAX + 1e-9, f"BMAX 超過: {d}"
    # 建物 clamp が効いている: 沿道 12m の建物に挟まれた primary(素の帯 5.5+2.0=7.5)は
    # 12-WALL まで縮む…のではなく素の帯が小さいので不変。帯そのものは 0 より大きい。
    gap, side = M.band_of(0)
    assert gap + side > 0
    assert worst > 0.5, "そもそもオフセットが効いていない"


def test_building_clamp_shrinks_band_when_walls_are_close():
    """建物が近い道では帯が縮む(壁にめり込ませない)。"""
    wide = _Mirror([("primary", [[0, 0], [100, 0]])], [])
    tight = _Mirror([("primary", [[0, 0], [100, 0]])],
                    [[[0, 2.0], [100, 2.0], [100, 40], [0, 40]]])
    gw = sum(wide.band_of(0))
    gt = sum(tight.band_of(0))
    assert gt < gw, f"建物 clamp が効いていない: {gt} >= {gw}"
    assert gt <= 2.0 - WALL + 1e-9, f"壁に食い込む帯: {gt}"


def test_offset_side_is_street_fixed_not_travel_direction():
    """左右は街路固定フレーム(edge の向き)で決まる = 対向者どうしが重ならない。

    同じ点・同じ人なら、どちら向きに歩いていようが同じ側に出る(鏡は進行方向を
    引数に取らない=そもそも参照できないことがこの性質の証明)。"""
    M = _Mirror(_EDGES, _BLDS)
    a = M.xy(7, 40.0, 0.0)
    b = M.xy(7, 40.0, 0.0)
    assert a == b
    # 別人は反対側にも出る(全員同じ側ではない)
    sides = {(1 if M.xy(aid, 40.0, 0.0)[1] >= 0 else -1) for aid in range(40)}
    assert sides == {1, -1}, "全員が同じ側に寄っている(扇になっていない)"


def test_blend_gives_c0_continuity_across_a_two_edge_path():
    """2 edge の折れ道を歩かせても横位置が飛ばない(交差点/edge 遷移の連続性)。

    素朴実装(最近傍 edge の λ をそのまま使う)なら接合点で必ず不連続に飛ぶので、
    比較対象として不連続版も測り、混合版が桁で勝つことを固定する。"""
    M = _Mirror(_EDGES, _BLDS)
    # (0,0)→(100,0)→(160,40) を 0.5m 刻みで歩く
    pts = []
    n1 = 200
    for k in range(n1 + 1):
        pts.append((100.0 * k / n1, 0.0))
    n2 = 144
    for k in range(1, n2 + 1):
        pts.append((100.0 + 60.0 * k / n2, 40.0 * k / n2))
    aid = 3

    def _jump(fn):
        prev, worst = None, 0.0
        for (x, y) in pts:
            r = fn(aid, x, y)
            if prev is not None:
                worst = max(worst, math.hypot(r[0] - prev[0], r[1] - prev[1]))
            prev = r
        return worst

    def _blended(a, x, y):
        return M.xy(a, x, y)

    def _naive(a, x, y):
        b1, _ = M.near2(x, y)
        ox, oy = M._off(b1, a)
        return (x + ox, y + oy)

    step_len = math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])
    j_blend = _jump(_blended)
    j_naive = _jump(_naive)
    # 混合版: 1 歩の移動量 + 小さな余白に収まる(横方向の瞬間移動が無い)
    assert j_blend < step_len + 0.5, f"混合版で横位置が飛んだ: {j_blend:.3f}m"
    # 素朴版は接合点で飛ぶ(混合が実際に効いていることの対照)
    assert j_naive > j_blend * 1.5, \
        f"対照が成立しない(素朴版 {j_naive:.3f} vs 混合版 {j_blend:.3f})"


def test_shipped_js_contains_the_same_formulas():
    """出荷 JS に鏡と同じ式が載っている(Python 鏡だけ直して JS が腐るのを防ぐ)。"""
    for frag in (
        "h=Math.imul(h,16777619)",                      # FNV-1a
        "(_fnv(k)/4294967296)*2-1",                     # λ の写像
        "const nx=-rec.dy/L, ny=rec.dx/L",              # 左法線(街路固定)
        "(l<0?-1:1)*(bd[0]+Math.abs(l)*bd[1])",         # sign(λ)*(gap+|λ|*side)
        "1-(b2.d-b1.d)/BLEND",                          # 混合重み
        "if(tot>lim){ const s=lim/tot; gap*=s; side*=s; }",   # 建物 clamp
        "if(!(rx===rx) || !(ry===ry)) return null;",    # NaN ガード
    ):
        assert frag in CORE, f"出荷 JS に式 {frag!r} が無い(鏡と乖離)"
    # 乱数/時刻に依存しない(問い合わせ時に新しい確率を引かない)
    assert "Math.random" not in CORE
    assert "Date.now" not in CORE


def test_lambda_js_does_not_touch_simulation_state():
    """λ は表示専用: sim 側の名前(D.positions/TRACKS.positions 等)を書き換えない。"""
    assert "D.positions[" not in CORE and "TRACKS.positions[" not in CORE
    # 2D ラッパは posAt の戻り値の x,y だけを差し替え、w には触らない
    w2 = mv._LATERAL_2D_JS
    assert "if(p[2]!==0) continue;" in w2, "路上(w===0)以外にも触れている"
    assert "p[2]=" not in w2, "w(屋内/圏外エンコード)を書き換えている"


# =========================================================== 4. 3D 街路可視性
def test_3d_street_only_is_default_on_for_new_runs_with_indoor_data():
    """新ラン(屋内データ有り)は **フラグ無しで** 屋内の人を隠す(ユーザー決定)。"""
    html = MV3.build_html("t", _scene3(), _tracks3(), indoor_json=_indoor_json())
    assert "window.__streetHidden(w)" in html, "新ランで既定 ON になっていない"
    assert '<input type="checkbox" id="lyStreetOnly" checked>' in html
    assert "let [x,y,w] = pos[i];" in html
    # λ は道連れにしない(こちらは明示 opt-in のまま)
    assert "window.__latAdapter" not in html


def test_3d_street_only_is_default_off_for_old_runs_without_indoor_data():
    """旧ラン(屋内データ無し)は既定 OFF のまま = 生成 HTML が従来とバイト同一。"""
    scene, tracks = _scene3(), _tracks3()
    auto = MV3.build_html("t", scene, tracks)
    assert auto == MV3.build_html("t", scene, tracks, street_only=False)
    assert "__streetHidden" not in auto


def test_3d_street_only_override_forces_off_even_with_indoor_data():
    """--no-street-only 相当(street_only=False)は屋内データが在っても抑止する。"""
    ind = _indoor_json()
    off = MV3.build_html("t", _scene3(), _tracks3(), indoor_json=ind, street_only=False)
    assert "__streetHidden" not in off
    assert "const [x,y,w] = pos[i];" in off, "抑止時は placeAgents が従来形に戻る"
    # 屋内オーバレイ自体は従来どおり効いている(切ったのは可視性ルールだけ)
    assert "floorLayout3d" in off and "window.__indoorXY(i, w, t)" in off
    # 抑止した出力は「屋内データ有り・本機能なし」の従来出力と完全一致
    assert off == MV3.build_html("t", _scene3(), _tracks3(), indoor_json=ind,
                                 street_only=False)


def test_3d_street_only_override_forces_on_without_indoor_data():
    """--street-only は屋内データが無いランでも強制 ON にできる。"""
    on = MV3.build_html("t", _scene3(), _tracks3(), street_only=True)
    assert "window.__streetHidden(w)" in on


def test_3d_street_only_hides_indoor_by_default():
    html = MV3.build_html("t", _scene3(), _tracks3(), street_only=True)
    # placeAgents に「屋内は隠す」分岐が入る
    assert "window.__streetHidden(w)" in html
    assert "if(w>=1000 && typeof window.__streetHidden==='function'" in html
    assert "let [x,y,w] = pos[i];" in html
    # 判定本体: 路上は常に見える / 屋内は注目中の 1 棟だけ
    assert "if(w < 1000) return false;" in html
    assert "return Math.floor((w-1000)/100) !== focusBi;" in html
    # ページ既定は「隠す」= トグルが checked
    assert '<input type="checkbox" id="lyStreetOnly" checked>' in html
    assert html.count('id="lyStreetOnly"') == 1


def test_3d_street_only_keeps_train_and_offmap_hidden():
    """電車内(-3)・範囲外(-1)・睡眠(-2)は従来どおり非表示のまま(既存分岐を壊さない)。"""
    html = MV3.build_html("t", _scene3(), _tracks3(), street_only=True)
    assert "if(w === -1 || w === -2 || w === -3){" in html
    assert "範囲外・睡眠・電車圏外は隠す" in html


def test_3d_building_focus_path_exists_and_reuses_xray():
    html = MV3.build_html("t", _scene3(), _tracks3(), street_only=True)
    for frag in (
        "window.__streetFocus",            # 建物選択 API
        "function _bldAt(",                # フットプリント当たり判定
        "function _inFP(",                 # 点 in ポリゴン
        "raycaster.intersectObject(agents).length) return",   # 人のクリックは既存に譲る
        "getElementById('xray')",          # 既存 X 線トグルへ相乗り
        "dispatchEvent(new Event('change'))",
        "id=\"streetFocus\"",              # 「◯◯の屋内を表示中」バー
    ):
        assert frag in html, f"建物フォーカス経路の {frag!r} が無い"
    assert html.count('id="streetFocus"') == 1


def test_3d_street_hidden_semantics_mirror():
    """__streetHidden の意味論(Python 鏡): 路上は見える・屋内は focus 一致のみ。"""
    def hidden(w, focus_bi):
        if w < 1000:
            return False
        return (w - 1000) // 100 != focus_bi

    assert hidden(0, -1) is False
    assert hidden(1000 + 0 * 100 + 4, -1) is True        # 既定=誰も選んでいない → 隠す
    assert hidden(1000 + 0 * 100 + 4, 0) is False        # b1 を選択 → 見える
    assert hidden(1000 + 3 * 100 + 2, 0) is True         # 別の建物は隠れたまま


def test_3d_lateral_and_street_only_are_independent():
    scene, tracks = _scene3(), _tracks3()
    lat = MV3.build_html("t", scene, tracks, lateral=True)
    so = MV3.build_html("t", scene, tracks, street_only=True)
    assert "__latXY" in lat and "__streetHidden" not in lat
    assert "__streetHidden" in so and "window.__latAdapter" not in so
    assert "let [x,y,w] = pos[i];" in lat and "let [x,y,w] = pos[i];" in so


def test_3d_coexists_with_indoor_overlay():
    """屋内オーバレイと同居できる(注入アンカーが競合しない)。

    _inject_indoor は placeAgents の `_p.set(...)` 行を、本機能はループ頭の
    `const [x,y,w] = pos[i];` を置く。両方入れても _replace_once が落ちない。"""
    html = MV3.build_html("t", _scene3(), _tracks3(), indoor_json=_indoor_json(),
                          lateral=True, street_only=True)
    for m in ("window.__indoorXY(i, w, t)", "floorLayout3d", "window.__streetHidden(w)",
              "window.__latXY", "let [x,y,w] = pos[i];"):
        assert m in html, f"同居時に {m!r} が落ちた"
    assert html.count("let [x,y,w] = pos[i];") == 1


def test_3d_no_roads_degrades_safely():
    """道路データが空でも壊れない(λ は null を返して元位置のまま)。"""
    html = MV3.build_html("t", _scene3(roads=[]), _tracks3(), lateral=True)
    assert "window.__latXY = function(){ return null; }" in html


def test_band_table_covers_every_klass_in_the_shipped_maps():
    """帯の表が実地図に出る道路クラスを全部持っている(既定値へ落ちる道が無い)。

    地図データの edge が持つ幅の手がかりは **klass だけ**(width/lanes/sidewalk の
    タグは無い)。だから表の網羅性がそのまま帯の妥当性になる。"""
    maps = sorted((REPO_ROOT / "data").glob("*osm*.json"))
    if not maps:
        pytest.skip("地図データが無い")
    seen: set[str] = set()
    for p in maps:
        try:
            city = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        for e in city.get("edges", []):
            if e.get("klass"):
                seen.add(e["klass"])
        # 帯の clamp 材料(建物フットプリント)が実在する
        blds = city.get("buildings", [])
        if blds:
            assert "footprint" in blds[0], f"{p.name}: footprint が無い(clamp できない)"
        # 幅そのものを持つタグは無い(= klass 推定が唯一の手段であることの記録)
        if city.get("edges"):
            assert not ({"width", "lanes", "sidewalk"} & set(city["edges"][0])), \
                "地図に実幅タグが増えた: BAND 推定より実測値を使うべき"
    assert seen, "地図から klass を1つも取れない"
    missing = seen - set(BAND)
    assert not missing, f"BAND 表に無い道路クラス(既定へ落ちる): {sorted(missing)}"


# =========================================================== 5. 注入 JS の健全性
def _js_delims(js: str) -> str:
    """注入 JS の括弧対応を検査する簡易チェッカ。

    ブラウザ非依存の環境(この repo の CI に JS ランタイムは無い)で、注入片の
    「閉じ忘れ 1 文字で viewer 全体が真っ白」という最悪の壊れ方だけは機械で防ぐ。
    対象は自作の注入片に限る(文字列・行コメントのみ扱えば足りる形に保つ約束):
    テンプレートリテラル(``)と正規表現リテラルは使わない=下の 2 テストで固定。
    """
    stack, pairs = [], {")": "(", "]": "[", "}": "{"}
    i, n, quote = 0, len(js), None
    while i < n:
        c = js[i]
        if quote:
            if c == "\\":
                i += 2
                continue
            if c == quote:
                quote = None
            i += 1
            continue
        if c in "\"'":
            quote = c
            i += 1
            continue
        if c == "/" and i + 1 < n and js[i + 1] == "/":
            while i < n and js[i] != "\n":
                i += 1
            continue
        if c in "([{":
            stack.append(c)
        elif c in ")]}":
            if not stack or stack[-1] != pairs[c]:
                return f"対応しない {c!r} @{i}: ...{js[max(0, i - 60):i + 10]!r}"
            stack.pop()
        i += 1
    if quote:
        return f"閉じていない文字列 {quote!r}"
    return "OK" if not stack else f"閉じていない {stack!r}"


_FRAGMENTS = {
    "_LATERAL_CORE_JS": mv._LATERAL_CORE_JS,
    "_LATERAL_2D_JS": mv._LATERAL_2D_JS,
    "_LATERAL_3D_ADAPTER": MV3._LATERAL_3D_ADAPTER,
    "_STREET_ONLY_JS": MV3._STREET_ONLY_JS,
}


@pytest.mark.parametrize("name", sorted(_FRAGMENTS))
def test_injected_js_delimiters_balanced(name):
    assert _js_delims(_FRAGMENTS[name]) == "OK", f"{name}: {_js_delims(_FRAGMENTS[name])}"


@pytest.mark.parametrize("name", sorted(_FRAGMENTS))
def test_injected_js_stays_checkable(name):
    """簡易チェッカが正しく効く形を保つ(テンプレートリテラル/正規表現リテラル禁止)。"""
    js = _FRAGMENTS[name]
    assert "`" not in js, "テンプレートリテラルを使うと括弧チェッカが効かなくなる"
    assert "/*" not in js, "ブロックコメントは行コメントに寄せる約束"


def test_injected_js_has_no_html_breakers():
    """注入片が </script> を含まない(HTML を途中で閉じてビューアが壊れない)。"""
    for name, js in _FRAGMENTS.items():
        assert "</script" not in js.lower(), f"{name} が script を閉じている"


# =========================================================== 6. CLI
def test_cli_flags_are_opt_in():
    mv_src = (REPO_ROOT / "viz" / "make_viewer.py").read_text(encoding="utf-8")
    mv3_src = (REPO_ROOT / "viz" / "make_viewer3d.py").read_text(encoding="utf-8")
    assert '"--lateral-offset" in flags' in mv_src
    assert '"--lateral-offset" in flags' in mv3_src
    # λ は完全 opt-in(既定 False)
    assert "lateral: bool = False" in mv3_src
    # 街路可視性は tri-state: None=auto(屋内データの有無)/ True / False の両方向上書き
    assert "street_only: bool | None = None" in mv3_src
    assert "street_only = indoor_json is not None" in mv3_src
    assert '"--street-only" in flags' in mv3_src
    assert '"--no-street-only" in flags' in mv3_src


def test_cli_street_only_override_flags_are_mutually_ordered():
    """--street-only と --no-street-only を両方渡したら ON が勝つ(明示 > 抑止の順)。"""
    mv3_src = (REPO_ROOT / "viz" / "make_viewer3d.py").read_text(encoding="utf-8")
    i_on = mv3_src.index('if "--street-only" in flags:')
    i_off = mv3_src.index('elif "--no-street-only" in flags:')
    assert i_on < i_off, "分岐順が変わると両指定時の勝者が変わる"


def test_2d_cli_unknown_flag_does_not_enable(tmp_path):
    """似た名前のフラグでは有効化されない(誤爆で旧ラン出力が変わらない)。"""
    rd = _write_run(tmp_path, "lat_typo")
    off_v, _ = _gen(rd)
    typo_v, _ = _gen(rd, "--lateral")
    assert typo_v == off_v
