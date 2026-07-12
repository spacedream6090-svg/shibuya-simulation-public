"""Phase E: observer.measure の純関数ユニットテスト。

最低4ケース:
  1. ews が「分散が単調増加する系列」で var_tau > 0
  2. r2_traits が完全線形データで R^2 ≈ 1
  3. item_cascades が手作りイベント列で size / depth / adopters を正しく返す
  4. agent_features が Y_external を成分ごと正しく数える
"""
from __future__ import annotations

import math

from society.observer import measure


def _ev(step, agent_id, kind, **payload):
    """load_events 出力と同じ形の 1 イベント dict。"""
    return {"step": step, "sim_min": step, "agent_id": agent_id, "kind": kind,
            "x": 0.0, "y": 0.0, "rng_stream": "", "llm_call_id": None,
            "payload": payload}


# --------------------------------------------------------------------------- #
def test_ews_rising_variance_positive_tau():
    # 振幅が線形に増大する交番信号 → detrend 後の移動窓分散が単調増加。
    series = [((-1) ** i) * (1.0 + i) for i in range(60)]
    out = measure.ews(series, detrend_bw=0.1, window_frac=0.5)
    assert out["var_tau"] > 0.0, out["var_tau"]
    # 分散系列そのものも概ね増加している(端点比較)
    vs = out["var_series"]
    assert vs[-1] > vs[0]
    # 逆に一定分散なら τ は 0 近傍(±1近くにならない)
    flat = [((-1) ** i) for i in range(60)]
    assert abs(measure.ews(flat)["var_tau"]) < 0.5


def test_r2_traits_perfect_linear():
    t1 = [0.1, 0.9, 0.4, 0.7, 0.2, 0.6, 0.95, 0.33]
    t2 = [0.5, 0.2, 0.8, 0.1, 0.65, 0.9, 0.05, 0.44]
    feats = []
    for i, (a, b) in enumerate(zip(t1, t2)):
        y = 3.0 * a - 2.0 * b + 5.0  # 完全線形(誤差ゼロ)
        feats.append({"id": i, "t1": a, "t2": b,
                      "Y_external": y, "Y_internal": y})
    out = measure.r2_traits(feats, ["t1", "t2"])
    assert out["external"]["r2"] is not None
    assert abs(out["external"]["r2"] - 1.0) < 1e-9, out["external"]["r2"]
    assert abs(out["internal"]["r2"] - 1.0) < 1e-9
    assert out["external"]["n"] == 8
    # traits が無ければ None(skip)
    none_out = measure.r2_traits(feats, [])
    assert none_out["external"]["r2"] is None


def test_item_cascades_size_depth_adopters():
    events = [
        _ev(0, 1, "vocab_coin", item_id="X", text="foo"),
        _ev(1, 2, "transmission", item_id="X", **{"from": 1}, channel="face"),
        _ev(2, 3, "transmission", item_id="X", **{"from": 2}, channel="face"),
        _ev(3, 4, "transmission", item_id="X", **{"from": 1}, channel="face"),
        # 無関係な別 item(混線しないこと)
        _ev(1, 9, "vocab_coin", item_id="Y", text="bar"),
    ]
    casc = {c["item_id"]: c for c in measure.item_cascades(events)}
    x = casc["X"]
    assert x["creator"] == 1
    assert x["size"] == 3                    # 伝播イベント数
    assert x["n_adopters"] == 3              # distinct 受信者 {2,3,4}
    assert x["adopters"] == [2, 3, 4]
    assert x["depth"] == 2                   # 1->2->3 の辺数
    assert x["channel_mix"] == {"face": 3}
    # Y は誰にも伝播していない → size 0, depth 0
    assert casc["Y"]["size"] == 0
    assert casc["Y"]["depth"] == 0


def test_agent_features_counts_y_external():
    events = [
        _ev(0, 0, "vocab_coin", item_id="X", text="foo"),
        _ev(1, 0, "speak", hearers=[1], items=["X"], text="..."),
        _ev(2, 1, "transmission", item_id="X", **{"from": 0}, channel="face"),
        _ev(3, 2, "transmission", item_id="X", **{"from": 1}, channel="face"),
        _ev(4, 2, "label_adopt", item_id="X", text="foo"),
        _ev(5, 1, "label_adopt", item_id="X", text="foo"),
        _ev(6, 0, "reflect", mode="free", written_back=True, belief="abcd"),
    ]
    meta = [{"id": 0, "name": "A"}, {"id": 1, "name": "B"}, {"id": 2, "name": "C"}]
    feats = {f["id"]: f for f in measure.agent_features(events, meta)}
    a0 = feats[0]
    # c_transmission: from==0 (0->1) + creator credit on (1->2) = 2
    assert a0["c_transmission"] == 2
    # c_label_adopt: agent1 の採用は直前に 0->1 で伝播したので 0 に帰属 = 1
    assert a0["c_label_adopt"] == 1
    # c_new_relations: speak で 1 と会話 = 1
    assert a0["c_new_relations"] == 1
    assert a0["c_sns_reach"] == 0
    assert a0["Y_external"] == 4             # 2 + 1 + 1 + 0
    # Y_internal: writeback 1 回 + belief 4 文字 = 5
    assert a0["Y_internal"] == 5
    assert a0["compute"] == 1                # reflect 1 回

    a1 = feats[1]
    assert a1["c_transmission"] == 1         # from==1 (1->2)
    assert a1["c_label_adopt"] == 1          # agent2 の採用は 1 に帰属
    assert a1["c_new_relations"] == 1        # speak の hearer として 0 と会話
    assert a1["Y_external"] == 3
