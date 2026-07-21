"""第45バッチ: 語彙の広がりの可視化(viz/make_viewer.py)のテスト。

検証項目(タスク検収 2026-07-22):
  (A) transmission イベント → 語ごとの trans:[[step,from,to,channel],...] 構築。
      vocab_coin の発生文脈 ctx・メディア発(creator=-1)の media フラグ。
  (B) 1語あたり上限の決定論間引き: 超過語で「当人宛の最初の聴取辺(=採用に効いた辺)」
      を最優先で残し、残り枠を step の早い辺で充填。caps に (kept/total) を記録
      (silent cap 禁止)。同一入力で 2 回 → trans/caps が一致(決定論)。
  (C) 後方互換: transmission ゼロのランでも vocab の既存キー(w/born/creator/adopts)
      は不変(trans/ctx/tn/media は「追加のみ」)。語彙ゼロなら vocab=[]・caps=[]。
  (D) mock ランで viewer/dashboard の HTML 生成が通り、語彙可視化の要素が埋まる。

house style: 合成 L1(parquet)を tmp_path に書いて build_data / main() を回す
(本機能は「runs/ を読むだけ」の純後処理なので合成 L1 で足りる)。
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

import make_viewer as mv  # noqa: E402

VIEWER = REPO_ROOT / "viz" / "make_viewer.py"


# --------------------------------------------------------------------------- #
# 合成ランの生成
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


def _write_run(tmp_path: Path, name: str, events: list[dict],
               n_agents: int = 8) -> Path:
    """events(dict列)を l1_events.parquet として書き、config/map/agents も用意。"""
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
               "occupation": "会社員", "has_bicycle": False, "has_car": False}
              for i in range(n_agents)]
    (run_dir / "agents.json").write_text(json.dumps(agents, ensure_ascii=False),
                                         encoding="utf-8")
    # 最低限、位置イベントを1つは入れて n_steps を決める(build_data は max step+1)。
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
    return run_dir


def _coin(step, agent, item_id, text, **ctx):
    p = {"item_id": item_id, "text": text}
    p.update(ctx)
    return {"step": step, "agent_id": agent, "kind": "vocab_coin", "payload": p}


def _trans(step, frm, to, item_id, channel):
    return {"step": step, "agent_id": to, "kind": "transmission",
            "payload": {"item_id": item_id, "from": frm, "channel": channel}}


def _adopt(step, agent, item_id, text):
    return {"step": step, "agent_id": agent, "kind": "label_adopt",
            "payload": {"item_id": item_id, "text": text}}


def _vocab_by_text(data, w):
    return next(v for v in data["vocab"] if v["w"] == w)


# --------------------------------------------------------------------------- #
# (A) transmission → trans 構築 / ctx / media
# --------------------------------------------------------------------------- #
def test_trans_and_ctx_construction(tmp_path):
    ev = [
        _coin(1, 0, "vocab-00001", "テスト語",
              place="テストビルの前", fire_reason="novel_place",
              drive=0.12, recent_mem=["長文の記憶は落とされる"]),
        _trans(2, 0, 1, "vocab-00001", "face"),
        _trans(2, 0, 2, "vocab-00001", "sns"),
        _trans(3, 0, 1, "vocab-00001", "face"),
        _adopt(3, 1, "vocab-00001", "テスト語"),
    ]
    data = mv.build_data(_write_run(tmp_path, "trans", ev), include_traffic=False)
    v = _vocab_by_text(data, "テスト語")

    # trans は [step, from, to, channel] を step 昇順で(同 step は記録順)
    assert v["trans"] == [[2, 0, 1, "face"], [2, 0, 2, "sns"], [3, 0, 1, "face"]]
    assert v["tn"] == 3
    assert v["media"] is False
    # ctx は発生文脈のキーを保持し、item_id/text/recent_mem は除く
    assert v["ctx"]["place"] == "テストビルの前"
    assert v["ctx"]["fire_reason"] == "novel_place"
    assert "item_id" not in v["ctx"] and "text" not in v["ctx"]
    assert "recent_mem" not in v["ctx"]   # 自由文は上限対策で落とす
    # 既存キーは不変
    assert v["born"] == 1 and v["creator"] == 0
    assert v["adopts"] == [[3, 1]]
    # transmission ゼロの語は無いのでこのランでは caps は空
    assert data["caps"] == []


def test_media_coin_flag(tmp_path):
    ev = [
        {"step": 1, "agent_id": -1, "kind": "vocab_coin",
         "payload": {"item_id": "vocab-00009", "text": "公式語", "media": True}},
        _trans(2, -1, 3, "vocab-00009", "news"),
        _trans(3, -1, 3, "vocab-00009", "news"),
        _adopt(3, 3, "vocab-00009", "公式語"),
    ]
    data = mv.build_data(_write_run(tmp_path, "media", ev), include_traffic=False)
    v = _vocab_by_text(data, "公式語")
    assert v["media"] is True and v["creator"] == -1
    assert v["trans"] == [[2, -1, 3, "news"], [3, -1, 3, "news"]]


# --------------------------------------------------------------------------- #
# (B) 上限間引きの決定論 + caps 記録
# --------------------------------------------------------------------------- #
def test_cap_thinning_deterministic_and_prioritizes_adopters(tmp_path, monkeypatch):
    monkeypatch.setattr(mv, "TRANS_CAP_PER_WORD", 4)
    # 採用者 5,6 の「当人宛の最初の聴取辺」を遅い step に置く(=素朴な早い順なら切られる)。
    ev = [
        _coin(1, 0, "vocab-00021", "上限語"),
        _trans(2, 0, 1, "vocab-00021", "face"),
        _trans(2, 0, 2, "vocab-00021", "face"),
        _trans(3, 0, 3, "vocab-00021", "sns"),
        _trans(3, 0, 4, "vocab-00021", "sns"),
        _trans(4, 0, 5, "vocab-00021", "face"),   # 採用者5の最初の辺(最優先で残す)
        _trans(4, 0, 5, "vocab-00021", "face"),   # 2回目
        _trans(5, 0, 6, "vocab-00021", "dm"),     # 採用者6の最初の辺(最優先で残す)
        _trans(5, 0, 1, "vocab-00021", "face"),
        _adopt(4, 5, "vocab-00021", "上限語"),
        _adopt(5, 6, "vocab-00021", "上限語"),
    ]

    def build():
        return mv.build_data(_write_run(tmp_path, "cap", ev), include_traffic=False)

    d1 = build()
    d2 = build()
    v1 = _vocab_by_text(d1, "上限語")
    v2 = _vocab_by_text(d2, "上限語")

    # 上限どおり 4 辺に間引き、tn は間引き前の総数
    assert len(v1["trans"]) == 4
    assert v1["tn"] == 8
    # 採用者 5・6 宛の(最初の)聴取辺が保存されている(採用に効いた辺を優先)
    tos = [e[2] for e in v1["trans"]]
    assert 5 in tos and 6 in tos
    assert [4, 0, 5, "face"] in v1["trans"]      # 5 の最初の辺(2回目でなく)
    assert [5, 0, 6, "dm"] in v1["trans"]
    # trans は step 昇順
    steps = [e[0] for e in v1["trans"]]
    assert steps == sorted(steps)
    # caps に (kept/total) を記録(silent cap 禁止)
    cap = next(c for c in d1["caps"] if c["w"] == "上限語")
    assert cap["kept"] == 4 and cap["total"] == 8
    assert cap["item_id"] == "vocab-00021"
    # 決定論: 2 回で trans も caps も一致
    assert v1["trans"] == v2["trans"]
    assert d1["caps"] == d2["caps"]


# --------------------------------------------------------------------------- #
# (C) 後方互換: transmission ゼロ / 語彙ゼロ
# --------------------------------------------------------------------------- #
def test_backward_compat_zero_transmission(tmp_path):
    """transmission が無いランでも既存キー不変・trans は空・追加キーのみ増える。"""
    ev = [
        _coin(1, 0, "vocab-00001", "無伝播語"),
        _adopt(2, 1, "vocab-00001", "無伝播語"),
        _adopt(3, 2, "vocab-00001", "無伝播語"),
    ]
    data = mv.build_data(_write_run(tmp_path, "notrans", ev), include_traffic=False)
    v = _vocab_by_text(data, "無伝播語")
    # 既存キーは従来どおり
    assert v["w"] == "無伝播語" and v["born"] == 1 and v["creator"] == 0
    assert v["adopts"] == [[2, 1], [3, 2]]
    # 追加キー: trans は空、tn=0、ctx/media も付く(追加のみ)
    assert v["trans"] == [] and v["tn"] == 0
    assert v["ctx"] == {} and v["media"] is False
    assert data["caps"] == []


def test_no_vocab_run(tmp_path):
    """語彙イベントが皆無でも vocab=[]・caps=[](クラッシュしない)。"""
    ev = [{"step": s, "agent_id": 0, "kind": "arrive", "x": 1.0, "y": 2.0,
           "payload": {"name": "路上"}} for s in range(3)]
    data = mv.build_data(_write_run(tmp_path, "novocab", ev), include_traffic=False)
    assert data["vocab"] == []
    assert data["caps"] == []


# --------------------------------------------------------------------------- #
# (D) mock ランで HTML 生成が通る + 可視化要素が埋まる
# --------------------------------------------------------------------------- #
def test_html_generation_smoke(tmp_path):
    ev = [
        _coin(1, 0, "vocab-00001", "語A", place="路上", fire_reason="novel_place"),
        _trans(2, 0, 1, "vocab-00001", "face"),
        _trans(2, 0, 2, "vocab-00001", "sns"),
        _trans(3, 0, 1, "vocab-00001", "face"),
        _adopt(3, 1, "vocab-00001", "語A"),
    ]
    rd = _write_run(tmp_path, "smoke", ev)
    subprocess.run([sys.executable, str(VIEWER), str(rd)],
                   check=True, capture_output=True)

    viewer = (rd / "viewer.html").read_text(encoding="utf-8")
    dash = (rd / "dashboard.html").read_text(encoding="utf-8")
    # 置換漏れトークンが無い
    assert "__DATA__" not in viewer and "__DATA__" not in dash
    # 地図側: 語彙(語別)モードと語選択パネル
    assert 'value="vocabword"' in viewer
    assert "vocabWordSel" in viewer and "function vwState" in viewer
    # ダッシュボード側: 詳細ビュー(採用曲線/ネットワーク/ログ)+ caps データ
    for sym in ("renderVocabDetail", "drawVCurve", "drawVNet", "drawVLog"):
        assert sym in dash, f"{sym} が dashboard に無い"
    assert '"caps":' in dash
    # 埋め込みデータに trans/ctx/media が入っている
    assert '"trans":' in dash and '"media":' in dash


def test_html_generation_no_vocab_smoke(tmp_path):
    """語彙ゼロのランでも HTML 生成がクラッシュしない(語選択パネルは早期 return)。"""
    ev = [{"step": s, "agent_id": 0, "kind": "arrive", "x": 1.0, "y": 2.0,
           "payload": {"name": "路上"}} for s in range(3)]
    rd = _write_run(tmp_path, "smoke_empty", ev)
    r = subprocess.run([sys.executable, str(VIEWER), str(rd)],
                       capture_output=True)
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    assert (rd / "viewer.html").exists() and (rd / "dashboard.html").exists()
