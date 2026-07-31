"""軌跡の量子化型付きバイナリ(P0: scripts/tracks_bin.py + export_3d/export_ue/make_viewer3d)。

検収の柱:
  A. 既定バイト同一 —— 新フラグ無しでは scene/tracks/glb・sim_ue.json・viewer3d.html が従来のまま。
  B. 決定論 —— 同一入力から 2 回符号化してバイト一致。
  C. 量子化 —— 往復誤差の上界(quant/2)と、0.1m 格子データでの誤差 0(厳密可逆)。
  D. JS が読む契約 —— セクションの 4B 整列・チャンク sidecar が tracks.bin の該当スライスと
     バイト一致・**JS デコーダと同じ手順を Python で再現**して tracks.json を復元できること
     (ブラウザ無しで「ビューアが同じ数字を見る」ことを機械検査する)。
  E. 旧ラン互換 —— tracks.bin の無い過去ランで従来どおりビューアが生成できる。
"""
from __future__ import annotations

import base64
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TB = _load("tracks_bin", "scripts/tracks_bin.py")
E3D = _load("export_3d", "scripts/export_3d.py")
EUE = _load("export_ue", "scripts/export_ue.py")


def _mv3():
    sys.path.insert(0, str(REPO_ROOT / "viz"))
    import make_viewer3d as mod          # noqa: E402
    return mod


# ============================================================ 合成データ
def _synthetic_map() -> dict:
    return {
        "meta": {"version": 3, "name": "mock", "attribution": "test",
                 "origin_latlon": [35.6595, 139.70062],
                 "bbox": [35.656, 139.695, 35.6625, 139.706], "crs": "local-m"},
        "nodes": [{"id": "n1", "name": "テスト交差点", "x": 0, "y": 0}],
        "edges": [{"u": "n1", "v": "n2", "klass": "primary", "layer": 0,
                   "geometry": [[0, 0], [50, 0], [50, 50]], "length": 100}],
        "buildings": [
            {"id": "bA", "name": "モックタワー", "kind": "office", "levels": 10,
             "below": 2, "cx": 5, "cy": 5,
             "footprint": [[0, 0], [10, 0], [10, 10], [0, 10]]},
            {"id": "bB", "name": "L字ビル", "kind": "retail", "levels": 3, "cx": 22, "cy": 22,
             "footprint": [[20, 20], [24, 20], [24, 22], [22, 22], [22, 24], [20, 24]]},
        ],
        "railways": [], "pois": [{"id": "p1", "name": "モックカフェ", "cat": "cafe",
                                  "x": 5, "y": 5, "building": "bA", "floor": 1}],
        "car_gateways": [],
    }


def _events(n_agents: int = 6, n_steps: int = 12) -> list:
    rows: list = []

    def add(step, aid, kind, x, y, payload):
        rows.append({"step": step, "sim_min": 7 * 60 + step * 10, "agent_id": aid,
                     "kind": kind, "x": float(x), "y": float(y),
                     "payload": json.dumps(payload, ensure_ascii=False),
                     "rng_stream": "", "llm_call_id": ""})

    for a in range(n_agents):
        add(0, a, "arrive", a * 5 - 7, -0.03, {"name": "路上"})     # -0.03 → round → -0.0
    for s in range(1, n_steps):
        for a in range(n_agents):
            if (a + s) % 3 == 0:
                add(s, a, "move_segment", a * 5 + s, s * 2,
                    {"mode": (a % 3), "pts": [[a * 5, 0], [a * 5 + s / 2, s], [a * 5 + s, s * 2]]})
    add(2, 0, "enter_building", 5, 5, {"building": "bA", "floor": 7})
    add(6, 0, "floor_move", 5, 5, {"building": "bA", "floor": 9})
    add(9, 0, "exit_building", 5, 5, {"building": "bA"})
    add(3, 1, "exit_area", -120.4, -80.5, {"gateway": "n1", "via": "walk"})
    add(8, 1, "enter_area", -120.4, -80.5, {"gateway": "n1", "via": "walk"})
    add(4, 2, "sleep_start", 3, 3, {"building": "bB", "until_step": 10})
    add(10, 2, "wake_up", 3, 3, {"slept_steps": 6})
    for s in range(n_steps):
        add(s, -1, "traffic_flow", 0, 0,
            {"n": 3 + s, "segs": [[[0, 0], [10.5, 5.5]], [[10.5, 5.5], [20, 10], [30, 12.5]]]})
    return rows


def _write_run(tmp_path: Path, name: str, n_agents: int = 6, n_steps: int = 12):
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True, exist_ok=True)
    map_path = tmp_path / f"{name}_map.json"
    map_path.write_text(json.dumps(_synthetic_map(), ensure_ascii=False), encoding="utf-8")
    (run_dir / "config.yaml").write_text(
        f"world:\n  map: {map_path.as_posix()}\n", encoding="utf-8")
    agents = [{"id": i, "name": f"住民{i}", "age": 20 + i, "gender": "男",
               "occupation": "会社員", "visitor": (i % 2 == 0),
               "has_bicycle": False, "has_car": (i == 1)} for i in range(n_agents)]
    (run_dir / "agents.json").write_text(json.dumps(agents, ensure_ascii=False),
                                         encoding="utf-8")
    rows = _events(n_agents, n_steps)
    schema = pa.schema([
        ("step", pa.int32()), ("sim_min", pa.int32()), ("agent_id", pa.int32()),
        ("kind", pa.string()), ("x", pa.float32()), ("y", pa.float32()),
        ("payload", pa.string()), ("rng_stream", pa.string()), ("llm_call_id", pa.string())])
    cols = {k: [r[k] for r in rows] for k in schema.names}
    pq.write_table(pa.table(cols, schema=schema), run_dir / "l1_events.parquet")
    return run_dir, map_path


# ============================================================ C. 量子化の上界と可逆性
def test_quantize_error_bound_generic():
    """任意の実数でも往復誤差は quant/2 を超えない(表示用途で不可視な 2.5cm)。"""
    rng = np.random.default_rng(20260731)
    v = rng.uniform(-1500.0, 1500.0, size=20000)
    q = TB._q(v, TB.QUANT_M)
    back = q.astype(np.float64) * TB.QUANT_M
    err = np.max(np.abs(back - v))
    assert err <= TB.QUANT_M / 2 + 1e-9, err
    assert err > TB.QUANT_M / 4          # 一様乱数なら上界近くまで出る=検定が効いている


def test_quantize_exact_on_tenth_grid():
    """export_3d は座標を 0.1m 丸めで書く。0.05 は 0.1 の 1/2 なので往復誤差は厳密に 0。"""
    v = np.round(np.random.default_rng(7).uniform(-1600, 1600, size=50000), 1)
    q = TB._q(v, TB.QUANT_M)
    back = np.round(q.astype(np.float64) * TB.QUANT_M, 1)
    assert np.array_equal(back, v)


def test_encode_decode_tracks_exact(tmp_path):
    """mock ラン: encode → decode で meta/agents/ids/sim_min 完全一致・座標誤差 0・状態一致。"""
    run_dir, map_path = _write_run(tmp_path, "rt")
    res = E3D.export_run(run_dir, map_path)
    tracks = json.loads(res["tracks"].read_text(encoding="utf-8"))
    blob, header = TB.encode_tracks(tracks)
    back = TB.decode_tracks(blob)
    for k in ("meta", "agents", "ids", "sim_min"):
        assert back[k] == tracks[k], k
    assert header["binary"]["max_abs_error_m"] == pytest.approx(TB.QUANT_M / 2)
    for r0, r1 in zip(tracks["positions"], back["positions"]):
        for a, b in zip(r0, r1):
            assert a[2] == b[2]                       # 状態 w は可逆(パレット)
            assert a[0] == pytest.approx(b[0], abs=0) and a[1] == pytest.approx(b[1], abs=0)
    for r0, r1 in zip(tracks["moves"], back["moves"]):
        for a, b in zip(r0, r1):
            assert (a is None) == (b is None)
            if a is not None:
                assert a[0] == b[0] and a[1] == b[1]
    assert tracks["traffic"] == back["traffic"]


def test_decode_differs_only_by_negative_zero(tmp_path):
    """唯一の非可逆点(正直に固定): -0.0 が 0.0 になる。それ以外は JSON 文字列も同一。"""
    run_dir, map_path = _write_run(tmp_path, "nz")
    res = E3D.export_run(run_dir, map_path)
    j0 = res["tracks"].read_text(encoding="utf-8")
    j1 = json.dumps(TB.decode_tracks(TB.encode_tracks(json.loads(j0))[0]),
                    ensure_ascii=False, separators=(",", ":"))
    assert "-0.0" in j0 and "-0.0" not in j1        # 合成データに -0.0 を仕込んである
    assert j0.replace("-0.0", "0.0") == j1


def test_state_palette_handles_large_w():
    """屋内状態 w = 1000+bIdx*100+floor は int16 に入らない(実測 max 720802)。
    素の int16 化(deep §2.2 の原案)では表現できないため、パレット方式が必要である証拠。"""
    big = 720802
    assert big > 32767
    tracks = {"meta": {"nSteps": 1}, "agents": [], "ids": [0, 1], "sim_min": [420],
              "positions": [[[1.0, 2.0, big], [3.0, 4.0, -3]]],
              "moves": [[None, None]], "traffic": [{"n": 0, "segs": []}]}
    back = TB.decode_tracks(TB.encode_tracks(tracks)[0])
    assert back["positions"][0][0][2] == big
    assert back["positions"][0][1][2] == -3


@pytest.mark.parametrize("tracks", [
    {"meta": {"nSteps": 0}, "agents": [], "ids": [], "sim_min": [],
     "positions": [], "moves": [], "traffic": []},
    {"meta": {"nSteps": 2}, "agents": [], "ids": [], "sim_min": [0, 10],
     "positions": [[], []], "moves": [[], []],
     "traffic": [{"n": 0, "segs": []}, {"n": 0, "segs": []}]},
    {"meta": {"nSteps": 1}, "agents": [{"id": 0}], "ids": [0], "sim_min": [0],
     "positions": [[[0.0, 0.0, 0]]], "moves": [[None]], "traffic": [{"n": 0, "segs": []}]},
])
def test_degenerate_inputs_roundtrip(tracks):
    """0 step / 0 エージェント / 1 点だけ、でも符号化・復号が壊れない(空セクション境界)。"""
    blob, header = TB.encode_tracks(tracks)
    assert TB.decode_tracks(blob) == tracks
    assert header["binary"]["n_chunks"] == (0 if not tracks["positions"] else 1)


def test_out_of_range_raises_loudly():
    """int16 に収まらない座標レンジは黙ってクリップせず例外(silent cap 禁止)。"""
    tracks = {"meta": {"nSteps": 1}, "agents": [], "ids": [0, 1], "sim_min": [0],
              "positions": [[[-5000.0, 0.0, 0], [5000.0, 0.0, 0]]],
              "moves": [[None, None]], "traffic": [{"n": 0, "segs": []}]}
    with pytest.raises(ValueError, match="int16"):
        TB.encode_tracks(tracks)


# ============================================================ B. 決定論
def test_encode_is_deterministic(tmp_path):
    run_dir, map_path = _write_run(tmp_path, "det")
    tracks = json.loads(E3D.export_run(run_dir, map_path)["tracks"].read_text(encoding="utf-8"))
    b1, h1 = TB.encode_tracks(tracks)
    b2, h2 = TB.encode_tracks(tracks)
    assert b1 == b2 and h1 == h2


def test_export_run_binary_two_passes_identical(tmp_path):
    """同一ランを 2 回エクスポート → tracks.bin / tracks_meta.json がバイト一致。"""
    run_dir, map_path = _write_run(tmp_path, "det2")
    a = E3D.export_run(run_dir, map_path, tracks_binary=True)
    b1 = {k: a[k].read_bytes() for k in ("scene", "tracks", "glb", "tracks_bin", "tracks_meta")}
    b = E3D.export_run(run_dir, map_path, tracks_binary=True)
    b2 = {k: b[k].read_bytes() for k in ("scene", "tracks", "glb", "tracks_bin", "tracks_meta")}
    assert b1 == b2


# ============================================================ A. 既定バイト同一
def test_default_output_byte_identical_with_and_without_binary(tmp_path):
    """--tracks-binary の有無で scene/tracks/glb が**全てバイト同一**(--low-mem と同じ流儀)。
    既定(フラグ無し)側には tracks.bin / tracks_meta.json が一切生成されないことも確認。"""
    run_a, map_a = _write_run(tmp_path, "base_a")
    run_b, map_b = _write_run(tmp_path, "base_b")
    ra = E3D.export_run(run_a, map_a)                          # 既定
    rb = E3D.export_run(run_b, map_b, tracks_binary=True)      # 新フラグ
    for k in ("scene", "tracks", "glb"):
        assert ra[k].read_bytes() == rb[k].read_bytes(), f"{k} がバイト不一致"
    assert not (run_a / "scene3d" / "tracks.bin").exists()
    assert not (run_a / "scene3d" / "tracks_meta.json").exists()
    assert "tracks_bin" not in ra and "tracks_meta" not in ra
    # --low-mem・--rich-tracks との組み合わせでも既定側は不変
    rc = E3D.export_run(run_b, map_b, tracks_binary=True, low_mem=True)
    assert ra["tracks"].read_bytes() == rc["tracks"].read_bytes()


def test_meta_json_equals_embedded_header(tmp_path):
    """tracks_meta.json は tracks.bin の JSON ヘッダと**同一文字列**(二重管理の防止)。"""
    run_dir, map_path = _write_run(tmp_path, "hdr")
    res = E3D.export_run(run_dir, map_path, tracks_binary=True)
    blob = res["tracks_bin"].read_bytes()
    header, off = TB.read_header(blob)
    assert blob[:8] == TB.MAGIC_TRACKS
    assert json.loads(res["tracks_meta"].read_text(encoding="utf-8")) == header
    assert off % 8 == 0
    # tracks.json の meta/agents/ids/sim_min がヘッダにそのまま入っている
    tracks = json.loads(res["tracks"].read_text(encoding="utf-8"))
    for k in ("meta", "agents", "ids", "sim_min"):
        assert header[k] == tracks[k], k


def test_no_tracks_json_flag(tmp_path):
    """--no-tracks-json: tracks.json を書かず、消費側は tracks.bin から従来の dict を得る。"""
    run_dir, map_path = _write_run(tmp_path, "nojson")
    res = E3D.export_run(run_dir, map_path, tracks_binary=True, write_tracks_json=False)
    assert not (run_dir / "scene3d" / "tracks.json").exists()
    assert "tracks" not in res
    tracks = TB.load_tracks(run_dir / "scene3d")
    assert tracks is not None and tracks["meta"]["nSteps"] == 12
    with pytest.raises(SystemExit):
        E3D.export_run(run_dir, map_path, write_tracks_json=False)


# ============================================================ D. JS が読む契約
def test_sections_are_4byte_aligned(tmp_path):
    """JS の `new Uint32Array(buf, off, n)` は 4B 整列が必須。全セクション/全チャンクで検査。"""
    run_dir, map_path = _write_run(tmp_path, "align")
    res = E3D.export_run(run_dir, map_path, tracks_binary=True)
    header, base = TB.read_header(res["tracks_bin"].read_bytes())
    for c in header["binary"]["chunks"]:
        assert c["off"] % 4 == 0
        for name, off in c["sec"].items():
            assert off % 4 == 0, f"{name} が 4B 非整列"
        assert c["len"] % 4 == 0


def _js_read_chunk(buf: bytes, c: dict, binary: dict) -> dict:
    """viz/make_viewer3d.py の __TRACKS_CHUNK__ と**同じ手順**でチャンクを型付き配列にする。
    (ブラウザが無いので、JS が使うオフセット/dtype 契約を Python 側で再現して検査する)"""
    S, ns = c["sec"], c["s1"] - c["s0"]
    na, pc = binary["n_agents"], binary["pos_coords"]
    st = "<u2" if binary["state_dtype"] == "u2" else "<u4"

    def rd(name, dt, n):
        return np.frombuffer(buf, dtype=dt, count=n, offset=S[name])

    return {"pos": rd("pos", "<i2", ns * na * pc),
            "st": rd("state", st, ns * na),
            "mvoff": rd("mvoff", "<u4", ns + 1), "mvag": rd("mvag", "<u4", c["n_moves"]),
            "mvmode": rd("mvmode", "u1", c["n_moves"]),
            "mvpo": rd("mvpo", "<u4", c["n_moves"] + 1),
            "mvpts": rd("mvpts", "<i2", c["n_move_pts"] * 2),
            "troff": rd("troff", "<u4", ns + 1), "trn": rd("trn", "<i4", ns),
            "trpo": rd("trpo", "<u4", c["n_segs"] + 1),
            "trpts": rd("trpts", "<i2", c["n_seg_pts"] * 2)}


def _js_rebuild(meta: dict, chunk_bytes: dict) -> dict:
    """JS の _posStep/_movesStep/_trafficStep と同じ式で tracks を復元する。"""
    b = meta["binary"]
    na, pc, q = b["n_agents"], b["pos_coords"], b["quant"]
    ox, oy = b["origin"][0], b["origin"][1]
    pal = b["state_palette"]
    P, M, T = [], [], []
    for c in b["chunks"]:
        A = _js_read_chunk(chunk_bytes[c["i"]], c, b)
        for li in range(c["s1"] - c["s0"]):
            base, sb = li * na * pc, li * na
            P.append([[A["pos"][base + i * pc] * q + ox,
                       A["pos"][base + i * pc + 1] * q + oy,
                       pal[A["st"][sb + i]]] for i in range(na)])
            row = [None] * na
            for r in range(int(A["mvoff"][li]), int(A["mvoff"][li + 1])):
                p0, p1 = int(A["mvpo"][r]), int(A["mvpo"][r + 1])
                row[int(A["mvag"][r])] = [
                    int(A["mvmode"][r]),
                    [[A["mvpts"][j * 2] * q + ox, A["mvpts"][j * 2 + 1] * q + oy]
                     for j in range(p0, p1)]]
            M.append(row)
            segs = []
            for r in range(int(A["troff"][li]), int(A["troff"][li + 1])):
                p0, p1 = int(A["trpo"][r]), int(A["trpo"][r + 1])
                segs.append([[A["trpts"][j * 2] * q + ox, A["trpts"][j * 2 + 1] * q + oy]
                             for j in range(p0, p1)])
            T.append({"n": int(A["trn"][li]), "segs": segs})
    return {"positions": P, "moves": M, "traffic": T}


def test_chunk_sidecars_match_bin_and_reproduce_tracks(tmp_path):
    """sidecar(base64 JSONP)が tracks.bin の該当スライスとバイト一致し、
    JS と同じ復号手順で tracks.json の数値を再現する(= ビューアが正しい数字を見る)。"""
    run_dir, map_path = _write_run(tmp_path, "side", n_agents=6, n_steps=12)
    res = E3D.export_run(run_dir, map_path, tracks_binary=True)
    blob = res["tracks_bin"].read_bytes()
    header, base = TB.read_header(blob)
    parts = TB.chunk_sidecars(blob)
    assert len(parts) == header["binary"]["n_chunks"]
    payloads = {}
    pat = re.compile(r'^__TRACKS_CHUNK__\((\d+),"([A-Za-z0-9+/=]*)"\);\n$')
    for i, (name, text) in enumerate(parts):
        assert name == f"chunk_{i:04d}.js"
        m = pat.match(text)
        assert m, f"{name} が JSONP 形式でない"
        assert int(m.group(1)) == i
        raw = base64.b64decode(m.group(2))
        c = header["binary"]["chunks"][i]
        assert raw == blob[base + c["off"]: base + c["off"] + c["len"]]
        payloads[i] = raw
    got = _js_rebuild(header, payloads)
    tracks = json.loads(res["tracks"].read_text(encoding="utf-8"))
    assert len(got["positions"]) == len(tracks["positions"])
    for r0, r1 in zip(tracks["positions"], got["positions"]):
        for a, b in zip(r0, r1):
            assert a[0] == pytest.approx(b[0], abs=1e-9)
            assert a[1] == pytest.approx(b[1], abs=1e-9)
            assert a[2] == b[2]
    for r0, r1 in zip(tracks["moves"], got["moves"]):
        for a, b in zip(r0, r1):
            assert (a is None) == (b is None)
            if a is not None:
                assert a[0] == b[0] and len(a[1]) == len(b[1])
                for p, s in zip(a[1], b[1]):
                    assert p[0] == pytest.approx(s[0], abs=1e-9)
                    assert p[1] == pytest.approx(s[1], abs=1e-9)
    for a, b in zip(tracks["traffic"], got["traffic"]):
        assert a["n"] == b["n"] and len(a["segs"]) == len(b["segs"])


def test_multi_chunk_split_is_exact(tmp_path):
    """チャンク境界をまたいでも復元は完全一致(1 チャンク = 1 step まで刻んで確認)。"""
    run_dir, map_path = _write_run(tmp_path, "multi", n_agents=6, n_steps=12)
    tracks = json.loads(E3D.export_run(run_dir, map_path)["tracks"].read_text(encoding="utf-8"))
    blob, header = TB.encode_tracks(tracks, chunk_steps=1)
    assert header["binary"]["n_chunks"] == 12
    assert [c["s0"] for c in header["binary"]["chunks"]] == list(range(12))
    one = TB.decode_tracks(blob)
    whole = TB.decode_tracks(TB.encode_tracks(tracks, chunk_steps=12)[0])
    assert one == whole
    # sidecar 経由(JS と同じ手順)でも一致
    base = TB.read_header(blob)[1]
    payloads = {c["i"]: blob[base + c["off"]: base + c["off"] + c["len"]]
                for c in header["binary"]["chunks"]}
    got = _js_rebuild(header, payloads)
    assert len(got["positions"]) == 12
    for r0, r1 in zip(one["positions"], got["positions"]):
        for a, b in zip(r0, r1):
            assert a[2] == b[2] and a[0] == pytest.approx(b[0]) and a[1] == pytest.approx(b[1])


# ============================================================ sim_ue(UE 経路)
def test_sim_ue_binary_roundtrip_and_bound(tmp_path):
    """sim_ue.bin: uu 座標を 0.05m 相当で量子化 → 誤差 ≤ 2.5uu(=2.5cm)。meta は完全保存。"""
    run_dir, map_path = _write_run(tmp_path, "ue")
    E3D.export_run(run_dir, map_path)
    tf = EUE.Transform()
    scene, tracks = EUE._ensure_scene(run_dir)
    sim_ue = EUE.build_sim_ue(scene, tracks, tf)
    blob, header = TB.encode_sim_ue(sim_ue)
    assert blob[:8] == TB.MAGIC_UE
    q = header["binary"]["quant"]
    assert q == pytest.approx(TB.QUANT_M * 100.0)          # 既定 scale=100 → 5uu
    back = TB.decode_sim_ue(blob)
    assert back["meta"] == sim_ue["meta"] and back["ids"] == sim_ue["ids"]
    err = 0.0
    for r0, r1 in zip(sim_ue["positions"], back["positions"]):
        for a, b in zip(r0, r1):
            assert a[3] == b[3]                            # state は可逆
            err = max(err, abs(a[0] - b[0]), abs(a[1] - b[1]), abs(a[2] - b[2]))
    assert err <= q / 2 + 1e-6, err
    assert err <= 2.5 + 1e-6                               # uu = cm → 2.5cm 以内


def test_export_ue_default_byte_identical(tmp_path):
    """--binary の有無で sim_ue.json はバイト同一(既定不変)。--binary 側だけ .bin が増える。"""
    run_a, map_a = _write_run(tmp_path, "ue_a")
    run_b, map_b = _write_run(tmp_path, "ue_b")
    E3D.export_run(run_a, map_a)
    E3D.export_run(run_b, map_b)
    ra = EUE.export_run(run_a, EUE.Transform())
    rb = EUE.export_run(run_b, EUE.Transform(), binary=True)
    assert ra["json"].read_bytes() == rb["json"].read_bytes()
    assert not (run_a / "scene3d" / "sim_ue.bin").exists()
    assert rb["bin"].exists() and rb["meta"].exists()
    assert json.loads(rb["meta"].read_text(encoding="utf-8")) == TB.read_header(
        rb["bin"].read_bytes())[0]
    # サイズ: JSON より小さい(1 agent-step あたりのバイト数で比較)
    assert rb["bin"].stat().st_size < ra["json"].stat().st_size


def test_export_ue_reads_binary_only_run(tmp_path):
    """tracks.json を持たないラン(--no-tracks-json)でも export_ue が従来どおり動く。"""
    run_dir, map_path = _write_run(tmp_path, "ue_bin_only")
    E3D.export_run(run_dir, map_path, tracks_binary=True, write_tracks_json=False)
    res = EUE.export_run(run_dir, EUE.Transform())
    sim = json.loads(res["json"].read_text(encoding="utf-8"))
    assert sim["meta"]["nSteps"] == 12 and len(sim["ids"]) == 6


# ============================================================ ビューア(make_viewer3d)
def test_viewer_default_unchanged_and_binary_mode(tmp_path):
    """既定の viewer3d.html は従来どおり(tracks を丸ごと埋め込み・サイドカー無し)。
    --tracks-binary では埋め込みが meta だけになり、チャンク sidecar が別ファイルに出る。"""
    mv3 = _mv3()
    # 注入 JS の固定費(約 7KB)より軌跡が十分大きい規模で比較する
    run_a, map_a = _write_run(tmp_path, "v_a", n_agents=40, n_steps=60)
    run_b, map_b = _write_run(tmp_path, "v_b", n_agents=40, n_steps=60)
    E3D.export_run(run_a, map_a)
    E3D.export_run(run_b, map_b)

    assert mv3.main([str(run_a)]) == 0
    html_a = (run_a / "viewer3d.html").read_text(encoding="utf-8")
    assert not (run_a / "tracks_bin").exists()
    assert "__TRACKS_CHUNK__" not in html_a
    assert "const TRACKS = JSON.parse(" in html_a
    tracks_json = (run_a / "scene3d" / "tracks.json").read_text(encoding="utf-8")
    assert tracks_json.replace("</", "<\\/") in html_a       # 従来どおり丸ごと埋め込み

    assert mv3.main([str(run_b), "--tracks-binary"]) == 0
    html_b = (run_b / "viewer3d.html").read_text(encoding="utf-8")
    meta_txt = (run_b / "scene3d" / "tracks_meta.json").read_text(encoding="utf-8")
    assert meta_txt.replace("</", "<\\/") in html_b
    assert "__TRACKS_CHUNK__" in html_b and "TRACKS_LAZY" in html_b
    assert 'id="binload"' in html_b
    assert html_b.count("const TRACKS = JSON.parse(") == 0   # 置換されている
    chunks = sorted((run_b / "tracks_bin").glob("chunk_*.js"))
    assert chunks and len(chunks) == json.loads(meta_txt)["binary"]["n_chunks"]
    # バイナリ版は「軌跡本体を持たない」ので既定版より小さい(軌跡が注入 JS 固定費を上回る規模で)
    assert tracks_json.replace("</", "<\\/") not in html_b
    assert len(html_b) < len(html_a)


def test_viewer_binary_html_structure(tmp_path):
    """ブラウザ無しの機械検査: script タグの対応・注入片の一意性・括弧収支・参照ファイル実在。"""
    mv3 = _mv3()
    run_dir, map_path = _write_run(tmp_path, "v_struct")
    E3D.export_run(run_dir, map_path)
    assert mv3.main([str(run_dir), "--tracks-binary"]) == 0
    html = (run_dir / "viewer3d.html").read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>") and html.rstrip().endswith("</html>")
    assert html.count("<script") == html.count("</script>")
    for marker in ("window.__TRACKS_CHUNK__ = function(ci, b64){",
                   "const TRACKS_LAZY = {", "function _showLoading(on){",
                   "if(!TRACKS_LAZY.ready(t)){"):
        assert html.count(marker) == 1, marker
    # 注入した JS ブロックが逐語で入っており、括弧収支が合う(ざっくり構文健全性)
    js = mv3._TRACKS_BIN_JS.replace("__CHUNK_DIR__", "tracks_bin")
    assert js in html
    code = "\n".join(ln for ln in js.splitlines() if not ln.strip().startswith("//"))
    for op, cl in (("{", "}"), ("(", ")"), ("[", "]")):
        assert code.count(op) == code.count(cl), f"{op}{cl} の収支が合わない"
    assert mv3._TRACKS_BIN_TICK in html
    # チャンクの参照名が実ファイルと一致する(_chunkFile の生成規則)
    n = json.loads((run_dir / "scene3d" / "tracks_meta.json").read_text(
        encoding="utf-8"))["binary"]["n_chunks"]
    for i in range(n):
        assert (run_dir / "tracks_bin" / f"chunk_{i:04d}.js").exists()
    assert 'const _CHUNK_DIR = "tracks_bin";' in html


def test_viewer_no_traffic_flag_in_binary_mode(tmp_path):
    """--no-traffic はバイナリ経路では meta のフラグで表現(交通セクションを読ませない)。"""
    mv3 = _mv3()
    run_dir, map_path = _write_run(tmp_path, "v_notr")
    E3D.export_run(run_dir, map_path)
    assert mv3.main([str(run_dir), "--tracks-binary", "--no-traffic"]) == 0
    html = (run_dir / "viewer3d.html").read_text(encoding="utf-8")
    assert '"no_traffic":true' in html


def test_viewer_stale_chunks_are_cleaned(tmp_path):
    """前回の生成物(より多いチャンク)が残っていると古いデータを掴む → 生成前に掃除する。"""
    mv3 = _mv3()
    run_dir, map_path = _write_run(tmp_path, "v_stale")
    E3D.export_run(run_dir, map_path)
    (run_dir / "tracks_bin").mkdir(parents=True, exist_ok=True)
    (run_dir / "tracks_bin" / "chunk_0099.js").write_text("__TRACKS_CHUNK__(99,\"\");\n",
                                                          encoding="ascii")
    assert mv3.main([str(run_dir), "--tracks-binary"]) == 0
    assert not (run_dir / "tracks_bin" / "chunk_0099.js").exists()


def test_viewer_legacy_run_without_binary(tmp_path):
    """旧ラン互換: tracks.bin を持たない過去ランで、従来どおり viewer3d.html が生成できる。"""
    mv3 = _mv3()
    run_dir, map_path = _write_run(tmp_path, "v_legacy")
    E3D.export_run(run_dir, map_path)
    assert not (run_dir / "scene3d" / "tracks.bin").exists()
    assert mv3.main([str(run_dir)]) == 0
    first = (run_dir / "viewer3d.html").read_bytes()
    assert mv3.main([str(run_dir)]) == 0
    assert (run_dir / "viewer3d.html").read_bytes() == first
    assert not (run_dir / "tracks_bin").exists()


def test_viewer_run_without_scene3d_still_exports(tmp_path):
    """scene3d 自体が無い過去ラン: 従来どおり export_3d が走って生成される(退行防止)。"""
    mv3 = _mv3()
    run_dir, map_path = _write_run(tmp_path, "v_fresh")
    (run_dir / "config.yaml").write_text(
        f"world:\n  map: {map_path.as_posix()}\n", encoding="utf-8")
    assert not (run_dir / "scene3d").exists()
    assert mv3.main([str(run_dir)]) == 0
    assert (run_dir / "scene3d" / "tracks.json").exists()
    assert (run_dir / "viewer3d.html").exists()


def test_viewer_reads_binary_only_run(tmp_path):
    """tracks.json を持たないラン(--no-tracks-json)を既定モードで開いても従来どおり動く。"""
    mv3 = _mv3()
    run_dir, map_path = _write_run(tmp_path, "v_bin_only")
    E3D.export_run(run_dir, map_path, tracks_binary=True, write_tracks_json=False)
    assert mv3.main([str(run_dir)]) == 0
    html = (run_dir / "viewer3d.html").read_text(encoding="utf-8")
    assert "const TRACKS = JSON.parse(" in html          # 復号して従来経路へ
    assert "__TRACKS_CHUNK__" not in html
