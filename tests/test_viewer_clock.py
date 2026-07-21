"""viz/make_viewer.py の壁時計原点(startMin)が run.start_tod に追従することの検証。

背景: ビューアの壁時計は step から「開始 07:00 固定」で逆算していた(startMin=7*60 ハードコード)。
run.start_tod で開始時刻が可変になったのに追従していなかった。build_data は l1_events.parquet の
sim_min 列(= Clock.sim_min = start_min + step*10 の権威値)から startMin を復元する。

方針(既存 R1 の鉄則を継承):
- 既定 07:00(sim_min=420+step*10)のランは startMin=420=従来値 → 出力バイト同一。
- 00:00 / 09:30 開始のランは sim_min 列からその開始分を復元し startMin に反映。
- sim_min 列が無い旧ランは既定 07:00 へ退避。CLI 相当の start_min 明示は最優先で上書き。
- 全経路 mock/合成データのみ(実 LLM 不使用)。JS 側は D.startMin を単一の原点として全ての
  時刻計算(tstr / nightAmt / trainsActive / 時計更新 / 分析軸)に用いるため、この 1 値で波及する。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "viz"))

import make_viewer as mv  # noqa: E402


# --------------------------------------------------------------------------- ヘルパ
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


def _write_run(tmp_path: Path, name: str, start_min: int | None,
               n_agents: int = 4, n_steps: int = 4) -> Path:
    """開始分 start_min の合成ラン(map/config/agents/parquet)を書く。
    start_min=None のときは sim_min 列を書かない(旧ラン=フォールバック検証用)。"""
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
    agents = [{"id": i, "name": f"住民{i}", "age": 20 + i, "gender": "男",
               "occupation": "会社員", "visitor": False,
               "has_bicycle": False, "has_car": False} for i in range(n_agents)]
    (run_dir / "agents.json").write_text(json.dumps(agents, ensure_ascii=False),
                                         encoding="utf-8")
    rows = []
    for step in range(n_steps):
        for a in range(n_agents):
            row = {"step": step, "agent_id": a, "kind": "arrive",
                   "x": float(a * 5), "y": 0.0,
                   "payload": json.dumps({"name": "路上", "node": "n1"},
                                         ensure_ascii=False),
                   "rng_stream": "", "llm_call_id": ""}
            if start_min is not None:
                row["sim_min"] = start_min + step * mv.STEP_MINUTES
            rows.append(row)
    fields = [("step", pa.int32()), ("agent_id", pa.int32()), ("kind", pa.string()),
              ("x", pa.float32()), ("y", pa.float32()), ("payload", pa.string()),
              ("rng_stream", pa.string()), ("llm_call_id", pa.string())]
    if start_min is not None:
        fields.insert(1, ("sim_min", pa.int32()))
    cols = {name_: [r[name_] for r in rows] for name_, _ in fields}
    pq.write_table(pa.table(cols, schema=pa.schema(fields)),
                   run_dir / "l1_events.parquet")
    return run_dir


# ============================================================ 単体: パーサ / 復元
def test_parse_start_tod_variants():
    assert mv._parse_start_tod("07:00") == 420
    assert mv._parse_start_tod("00:00") == 0
    assert mv._parse_start_tod("09:30") == 570
    assert mv._parse_start_tod("23:50") == 1430
    assert mv._parse_start_tod(None) == 420        # 欠落は既定 07:00
    assert mv._parse_start_tod(570) == 570         # 既に分
    assert mv._parse_start_tod("bogus") == 420     # 不正は既定へ退避


def test_derive_start_min_invariant():
    """sim_min = start_min + step*10 の不変量から、どのイベントでも同じ start_min を復元する。"""
    events = [{"step": 2, "sim_min": 590}, {"step": 0, "sim_min": 570},
              {"step": 5, "sim_min": 620}]      # すべて start_min=570(09:30)
    assert mv._derive_start_min(events) == 570
    assert mv._derive_start_min([{"step": 0}]) is None   # sim_min 列なし
    assert mv._derive_start_min([]) is None


# ============================================================ 既定 07:00 = 従来値(バイト同一)
def test_default_0700_startmin_is_420(tmp_path):
    """開始 07:00(sim_min=420+step*10)の run は startMin=420=従来ハードコード値。"""
    rd = _write_run(tmp_path, "s0700", start_min=420)
    data = mv.build_data(rd, include_traffic=False)
    assert data["startMin"] == 420


def test_missing_sim_min_falls_back_to_0700(tmp_path):
    """sim_min 列が無い旧ランは既定 07:00 へ退避(=従来出力とバイト同一)。"""
    rd = _write_run(tmp_path, "s_old", start_min=None)
    data = mv.build_data(rd, include_traffic=False)
    assert data["startMin"] == 420


# ============================================================ start_tod ≠ 07:00 に追従
def test_startmin_follows_midnight_run(tmp_path):
    """00:00 開始の run は startMin=0(壁時計が step から開始 07:00 で逆算しない)。"""
    rd = _write_run(tmp_path, "s0000", start_min=0)
    data = mv.build_data(rd, include_traffic=False)
    assert data["startMin"] == 0
    # startMin は埋め込み JSON に流れ、JS の時刻原点 D.startMin になる(壁時計の唯一の起点)。
    # main() と同じシリアライズ(既定セパレータ ": ")で payload に反映されることを確認。
    payload = json.dumps(data, ensure_ascii=False)
    assert '"startMin": 0' in payload


def test_startmin_follows_0930_run(tmp_path):
    """09:30 開始の run は startMin=570。"""
    rd = _write_run(tmp_path, "s0930", start_min=570)
    data = mv.build_data(rd, include_traffic=False)
    assert data["startMin"] == 570


# ============================================================ CLI --start-tod 明示上書き
def test_explicit_start_min_overrides_sim_min(tmp_path):
    """明示 start_min(CLI --start-tod 相当)は sim_min 復元より優先して上書きする。"""
    rd = _write_run(tmp_path, "s_override", start_min=420)   # run 自体は 07:00
    data = mv.build_data(rd, include_traffic=False, start_min=mv._parse_start_tod("13:15"))
    assert data["startMin"] == 13 * 60 + 15 == 795
