"""viz/notable_events.py の抽出 + viz/make_viewer3d.py の顕著イベント注入の検証。

- 合成イベント → 抽出結果(対象 kind のみ・ラベル/重要度・時刻順・frame 写像・has_pos・text)
- 間引き(magnitude 降順 上位 N / step 等分)の上限動作 = caps に件数明記(silent cap 禁止)
- notable データ無し = 生成 HTML が従来とバイト同一(既存 terrain/plateau と同じ検収基準)
- 注入マーカー(notable-data / パネル / 構築 JS)の存在と scene/tracks の健全性
- _ensure_notable: parquet 有(notable 有=非 None・sidecar 書き出し)/ noise のみ=None / 無し=None
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


NB = _load("notable_events", "viz/notable_events.py")
MV3 = _load("make_viewer3d", "viz/make_viewer3d.py")


# ----------------------------------------------------------------- ヘルパ
def _ev(step, aid, kind, x, y, payload):
    return {"step": step, "sim_min": 7 * 60 + step * 10, "agent_id": aid,
            "kind": kind, "x": float(x), "y": float(y),
            "payload": json.dumps(payload, ensure_ascii=False),
            "rng_stream": "", "llm_call_id": ""}


def _mixed_events():
    """顕著 kind + 大量のノイズ(move/opinion/hear 等)を混在させた合成列。"""
    rows = [
        _ev(2, 5, "group_found", -262.7, -17.5,
            {"founder": 5, "group_id": 0, "name": "つけめんの会", "purpose": "ゆるくつながる"}),
        _ev(6, 27, "label_coin", 629.6, 165.6, {"item_id": "vocab-00001", "text": "フワき"}),
        _ev(9, -1, "council_elected", 0.0, 0.0,
            {"day": 0, "members": [3, 15, 16, 32], "n_candidates": 32, "term": 1}),
        _ev(12, 33, "illness", -646.8, 228.9, {"days": 3, "state": "onset"}),
        _ev(4, 59, "partner_formed", -112.4, -54.7, {"other": 84}),
        _ev(1, -1, "disaster", 0.0, 0.0, {"kind": "台風", "phase": "onset"}),
    ]
    # ノイズ(抽出対象外): 大量にあっても events には出ない
    for s in range(20):
        rows.append(_ev(s, s % 4, "move_segment", 1, 1, {"mode": "walk", "pts": [[0, 0], [1, 1]]}))
        rows.append(_ev(s, s % 4, "opinion_shift", 2, 2, {"source": 1, "old": 0.1, "new": 0.2}))
        rows.append(_ev(s, s % 4, "hear", 3, 3, {"speaker": 1, "item_ids": []}))
    return rows


# ----------------------------------------------------------------- 抽出: 基本
def test_extract_selects_only_notable_kinds():
    data = NB.extract_notable(_mixed_events(), n_steps=20, stride=1)
    kinds = {e["kind"] for e in data["events"]}
    assert kinds == {"group_found", "label_coin", "council_elected",
                     "illness", "partner_formed", "disaster"}
    # ノイズは 1 件も混入しない
    assert not (kinds & {"move_segment", "opinion_shift", "hear"})
    assert data["n_kept"] == 6 and data["n_total"] == 6


def test_extract_time_ordered_and_labeled():
    data = NB.extract_notable(_mixed_events(), n_steps=20, stride=1)
    steps = [e["step"] for e in data["events"]]
    assert steps == sorted(steps)  # 時刻順
    by_kind = {e["kind"]: e for e in data["events"]}
    assert by_kind["disaster"]["label"] == "災害"
    assert by_kind["disaster"]["importance"] == 5
    assert by_kind["label_coin"]["label"] == "新語の創出"
    # text は payload から生成(新語の実文字列)
    assert "フワき" in by_kind["label_coin"]["text"]
    assert "台風" in by_kind["disaster"]["text"]
    # council_elected の members(list)は「N名」表記
    assert "4名" in by_kind["council_elected"]["text"]


def test_extract_has_pos_flag():
    data = NB.extract_notable(_mixed_events(), n_steps=20, stride=1)
    by_kind = {e["kind"]: e for e in data["events"]}
    # 世界イベント(agent_id=-1, xy=0)は位置なし=カメラ移動しない
    assert by_kind["disaster"]["has_pos"] is False
    assert by_kind["council_elected"]["has_pos"] is False
    # エージェント位置ありは has_pos True
    assert by_kind["group_found"]["has_pos"] is True
    assert by_kind["partner_formed"]["has_pos"] is True


def test_extract_frame_mapping_with_stride():
    # stride=3: raw step 9 → frame 3、n_steps=5 でクランプ確認
    data = NB.extract_notable(_mixed_events(), n_steps=5, stride=3)
    fr = {e["kind"]: e["frame"] for e in data["events"]}
    assert fr["disaster"] == 0          # step1 // 3
    assert fr["partner_formed"] == 1    # step4 // 3
    assert fr["council_elected"] == 3   # step9 // 3
    assert fr["illness"] == 4           # step12 // 3 = 4 だが n_steps-1=4 にクランプ
    assert all(0 <= e["frame"] <= 4 for e in data["events"])


def test_payload_accepts_dict_and_str():
    # payload は dict でも JSON 文字列でも同じ結果
    e_str = _ev(1, 3, "illness", 5, 5, {"state": "onset", "days": 2})
    e_dict = dict(e_str); e_dict["payload"] = {"state": "onset", "days": 2}
    d1 = NB.extract_notable([e_str], n_steps=2)
    d2 = NB.extract_notable([e_dict], n_steps=2)
    assert d1["events"][0]["text"] == d2["events"][0]["text"] == "onset"


# ----------------------------------------------------------------- 間引き(cap)
def test_cap_magnitude_kind_keeps_top_reach():
    # viral_cascade を cap 超で生成 → reach 上位のみ残る・caps に件数明記
    cap = 5
    rows = [_ev(s, 7, "viral_cascade", 0, 0, {"author": 7, "post_id": s, "reach": s})
            for s in range(30)]
    data = NB.extract_notable(rows, n_steps=30, per_kind_cap=cap)
    vc = [e for e in data["events"] if e["kind"] == "viral_cascade"]
    assert len(vc) == cap
    c = data["caps"]["viral_cascade"]
    assert c["total"] == 30 and c["kept"] == cap and c["dropped"] == 25  # silent cap 禁止
    # reach 上位 5(=25..29)が残っている(magnitude 降順で選抜)
    kept_reach = sorted(int(e["text"]) for e in vc)
    assert kept_reach == [25, 26, 27, 28, 29]
    # 表示自体は時刻順に整列
    assert [e["step"] for e in vc] == sorted(e["step"] for e in vc)


def test_cap_even_sample_non_magnitude_kind():
    # flyer_post(magnitude なし)は step 等分サンプル。両端を含む決定論選択。
    cap = 4
    rows = [_ev(s, 3, "flyer_post", 10, 10, {"author": 3, "text": f"告知{s}"})
            for s in range(10)]
    data = NB.extract_notable(rows, n_steps=10, per_kind_cap=cap)
    fp = [e for e in data["events"] if e["kind"] == "flyer_post"]
    assert len(fp) == cap
    steps = [e["step"] for e in fp]
    assert steps[0] == 0 and steps[-1] == 9   # 両端を含む
    assert data["caps"]["flyer_post"]["dropped"] == 6


def test_under_cap_keeps_all_no_drop():
    rows = [_ev(s, 3, "flyer_post", 10, 10, {"text": f"t{s}"}) for s in range(4)]
    data = NB.extract_notable(rows, n_steps=4, per_kind_cap=50)
    assert data["caps"]["flyer_post"]["dropped"] == 0
    assert data["caps"]["flyer_post"]["kept"] == 4


# ----------------------------------------------------------------- 注入 / バイト同一
def _tiny_scene_tracks():
    scene = json.dumps({"meta": {"floor_height": 3.5, "origin_latlon": None},
                        "buildings": [], "roads": [], "rails": [], "pois": []},
                       ensure_ascii=False)
    tracks = json.dumps({"meta": {"nSteps": 3, "step_minutes": 10, "start_min": 420,
                                  "floor_height": 3.5},
                         "agents": [{"id": 0, "name": "a", "visitor": False,
                                     "occupation": "?", "age": 20, "gender": "?"}],
                         "ids": [0],
                         "positions": [[[0, 0, 0]], [[0, 0, 0]], [[0, 0, 0]]],
                         "moves": [[None], [None], [None]],
                         "traffic": [{"n": 0, "segs": []}] * 3,
                         "sim_min": [420, 430, 440]}, ensure_ascii=False)
    return scene, tracks


def test_html_byte_identical_without_notable():
    """notable_json 無し(None)= 従来とバイト同一(注入は完全ゲート)。"""
    scene, tracks = _tiny_scene_tracks()
    plain = MV3.build_html("t", scene, tracks)
    none_arg = MV3.build_html("t", scene, tracks, notable_json=None)
    assert plain == none_arg


def test_html_injects_markers_with_notable():
    scene, tracks = _tiny_scene_tracks()
    data = NB.extract_notable(_mixed_events(), n_steps=3, stride=1)
    notable_json = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    html = MV3.build_html("t", scene, tracks, notable_json=notable_json)
    # 注入マーカー
    assert 'id="notable-data"' in html
    assert 'id="notable"' in html and 'id="ntList"' in html
    assert "setupNotable" in html
    assert "顕著イベント" in html
    # scene/tracks は従来どおり健在(壊していない)
    assert 'id="scene-data"' in html and 'id="tracks-data"' in html
    assert html.count('id="notable-data"') == 1
    # プレーン版に無い要素が入っている(=差分がある)
    plain = MV3.build_html("t", scene, tracks)
    assert len(html) > len(plain)


def test_injected_json_is_script_safe():
    """text に </script> 等が入っても script ブロックを壊さない(_json_for_script)。"""
    scene, tracks = _tiny_scene_tracks()
    rows = [_ev(1, 3, "flyer_post", 5, 5, {"text": "危険</script><b>xss"})]
    data = NB.extract_notable(rows, n_steps=3)
    notable_json = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    html = MV3.build_html("t", scene, tracks, notable_json=notable_json)
    assert "</script><b>xss" not in html      # 生の閉じタグは残らない
    assert "<\\/script>" in html               # エスケープ済み


# ----------------------------------------------------------------- _ensure_notable
def _write_run(tmp_path: Path, name: str, rows: list) -> Path:
    run_dir = tmp_path / name
    (run_dir / "scene3d").mkdir(parents=True)
    cols = {k: [r[k] for r in rows] for k in rows[0]}
    schema = pa.schema([
        ("step", pa.int32()), ("sim_min", pa.int32()), ("agent_id", pa.int32()),
        ("kind", pa.string()), ("x", pa.float32()), ("y", pa.float32()),
        ("payload", pa.string()), ("rng_stream", pa.string()), ("llm_call_id", pa.string())])
    pq.write_table(pa.table(cols, schema=schema), run_dir / "l1_events.parquet")
    return run_dir


def _tracks_meta(n=20):
    return json.dumps({"meta": {"nSteps": n, "step_stride": 1}})


def test_ensure_notable_writes_sidecar(tmp_path):
    run_dir = _write_run(tmp_path, "nb_hit", _mixed_events())
    text = MV3._ensure_notable(run_dir, _tracks_meta(20))
    assert text is not None
    data = json.loads(text)
    assert data["n_kept"] == 6
    # sidecar が scene3d に書き出されている
    side = run_dir / "scene3d" / "notable_events.json"
    assert side.exists()
    assert json.loads(side.read_text(encoding="utf-8"))["n_kept"] == 6


def test_ensure_notable_none_when_noise_only(tmp_path):
    """notable が 0 件 = None(= build_html は従来とバイト同一の経路)。"""
    noise = [_ev(s, s % 3, "opinion_shift", 1, 1, {"old": 0.1, "new": 0.2}) for s in range(15)]
    run_dir = _write_run(tmp_path, "nb_noise", noise)
    assert MV3._ensure_notable(run_dir, _tracks_meta(15)) is None
    assert not (run_dir / "scene3d" / "notable_events.json").exists()


def test_ensure_notable_none_when_no_parquet(tmp_path):
    run_dir = tmp_path / "nb_empty"
    (run_dir / "scene3d").mkdir(parents=True)
    assert MV3._ensure_notable(run_dir, _tracks_meta(10)) is None


def test_ensure_notable_full_pipeline_byte_identity(tmp_path):
    """同一 run で notable 有→注入 HTML、noise run→従来 HTML がバイト同一という双方向確認。"""
    scene, tracks = _tiny_scene_tracks()
    # noise run: _ensure_notable None → build_html は plain と一致
    noise = [_ev(s, 0, "hear", 1, 1, {"speaker": 1}) for s in range(3)]
    run_noise = _write_run(tmp_path, "byte_noise", noise)
    nj = MV3._ensure_notable(run_noise, tracks)
    assert nj is None
    assert MV3.build_html("t", scene, tracks, notable_json=nj) == MV3.build_html("t", scene, tracks)
