"""第51バッチ: scripts/analyze_luck.py の検証(運/実力分解)。

house style は tests/test_analyze_founders.py に倣う。確かめること:
  1. 成果集計: income(wage/venture_sale/deliver)・relations(会話相手)・reputation。
  2. 実力側特徴が money/wage 等の成果リークを除外する。
  3. OLS(lstsq)が完全線形で R²≈1・過少決定(n<=p+1)で None。
  4. 運の対応表は audit_uncertainty と共有(全 kind schema 登録済み)。
  5. 分解出力スキーマ + 決定論(同一入力 → 同一 JSON バイト)。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))          # scripts/ は package ではない
sys.path.insert(0, str(_ROOT / "src"))              # society.observer(研究資産)

import analyze_luck as al                            # noqa: E402
import audit_uncertainty as au                       # noqa: E402


def _ev(step, agent, kind, payload=None):
    return {"step": step, "sim_min": step * 10, "agent": agent,
            "kind": kind, "payload": payload or {}}


# --------------------------------------------------------------------------- #
# 1. 成果集計
# --------------------------------------------------------------------------- #
def test_outcomes_income_relations_reputation():
    events = [
        _ev(0, 1, "wage", {"amount": 1000.0}),
        _ev(5, 1, "venture_sale", {"amount": 200.0, "buyer": 2}),
        _ev(6, -1, "deliver", {"courier": 1, "fare": 50.0}),      # 配達員=1 へ帰属
        _ev(1, 1, "speak", {"hearers": [2, 3]}),                  # 相手 2,3
        _ev(2, 4, "hear", {"speaker": 1}),                        # 1<->4
        _ev(3, 1, "reputation_update", {"old": 0.0, "new": 2.0}), # +2
        _ev(4, 5, "sns_like", {"author": 1}),                     # いいね+1
    ]
    out = al.agent_outcomes(events, residents={1, 2, 3, 4})
    assert out[1]["income"] == 1250.0                             # 1000+200+50
    assert out[1]["relations"] == 3                               # {2,3,4}
    assert out[1]["reputation"] == 3.0                            # +2 delta + 1 like


def test_income_ignores_negative_reputation_delta():
    events = [_ev(0, 1, "reputation_update", {"old": 5.0, "new": 3.0})]  # 減少は0扱い
    out = al.agent_outcomes(events, residents={1})
    assert out[1]["reputation"] == 0.0


# --------------------------------------------------------------------------- #
# 2. 実力側特徴のリーク除外
# --------------------------------------------------------------------------- #
def test_skill_features_exclude_income_leak():
    agents = {i: {"occupation": "A", "age": 30 + i} for i in range(6)}
    traits = {i: {"internal_locus": 0.5, "nfc": 0.3, "money": 99999.0, "wage": 12000.0}
              for i in range(6)}
    feats, cols, info = al.skill_features(agents, traits, residents=set(range(6)))
    assert "money" not in cols and "wage" not in cols            # 成果リークを除外
    assert "internal_locus" in cols and "nfc" in cols
    assert "age" in cols
    # 職業が一様 → ダミー無し(基準のみ)
    assert not any(c.startswith("occ=") for c in cols)


def test_skill_features_occupation_dummies():
    # A×4, B×3, C×1 → 頻度3以上=A,B。最頻A=基準。ダミーは occ=B のみ(C は稀=落とす)。
    agents = {}
    for i in range(4):
        agents[i] = {"occupation": "A", "age": 30}
    for i in range(4, 7):
        agents[i] = {"occupation": "B", "age": 30}
    agents[7] = {"occupation": "C", "age": 30}
    traits = {i: {"internal_locus": 0.5} for i in range(8)}
    _, cols, info = al.skill_features(agents, traits, residents=set(range(8)),
                                      occ_min_count=3)
    assert info["occ_baseline"] == "A"
    assert cols.count("occ=B") == 1
    assert "occ=C" not in cols
    assert "occ=A" not in cols                                    # 基準はダミー化しない


# --------------------------------------------------------------------------- #
# 3. OLS(lstsq)
# --------------------------------------------------------------------------- #
def test_ols_perfect_linear():
    # y = 2x + 1 の完全線形 → R²≈1。
    rows = [{"x": float(i), "y": 2.0 * i + 1.0} for i in range(8)]
    res = al.ols(rows, ["x"], "y")
    assert res["r2"] is not None and abs(res["r2"] - 1.0) < 1e-9
    assert abs(res["coef"]["x"] - 2.0) < 1e-6
    assert abs(res["coef"]["intercept"] - 1.0) < 1e-6


def test_ols_underdetermined_returns_none():
    rows = [{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 2.0}]           # n=2, p=1 → n<=p+1
    res = al.ols(rows, ["x"], "y")
    assert res["r2"] is None
    assert res["n"] == 2 and res["p"] == 1


def test_ols_constant_y_returns_none():
    rows = [{"x": float(i), "y": 5.0} for i in range(6)]          # Y 定数=分散0
    res = al.ols(rows, ["x"], "y")
    assert res["r2"] is None


def test_pearson_helper():
    assert abs(al._pearson([1, 2, 3], [2, 4, 6]) - 1.0) < 1e-9
    assert abs(al._pearson([1, 2, 3], [3, 2, 1]) + 1.0) < 1e-9


# --------------------------------------------------------------------------- #
# 4. 運の対応表は audit と共有・全 kind 登録済み
# --------------------------------------------------------------------------- #
def test_shared_luck_table_registered():
    from society.observer.schema import EVENT_KINDS
    assert al.au.LUCK_KINDS is au.LUCK_KINDS           # 同一オブジェクト=共有 import
    assert all(k in EVENT_KINDS for k in al.au.LUCK_KINDS)


# --------------------------------------------------------------------------- #
# 5. end-to-end: 合成ラン(決定論=同一 JSON バイト)
# --------------------------------------------------------------------------- #
def _write_run(run_dir: Path, events, agents_list, traits):
    import pyarrow as pa
    import pyarrow.parquet as pq

    run_dir.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([
        pa.field("step", pa.int32()), pa.field("sim_min", pa.int32()),
        pa.field("agent_id", pa.int32()), pa.field("kind", pa.string()),
        pa.field("payload", pa.string()),
    ])
    cols = {"step": [], "sim_min": [], "agent_id": [], "kind": [], "payload": []}
    for e in events:
        cols["step"].append(e["step"])
        cols["sim_min"].append(e["step"] * 10)
        cols["agent_id"].append(e["agent"])
        cols["kind"].append(e["kind"])
        cols["payload"].append(json.dumps(e["payload"], ensure_ascii=False))
    pq.write_table(pa.Table.from_pydict(cols, schema=schema),
                   str(run_dir / "l1_events.parquet"))
    (run_dir / "agents.json").write_text(
        json.dumps(agents_list, ensure_ascii=False), encoding="utf-8")
    (run_dir / "traits.json").write_text(
        json.dumps(traits, ensure_ascii=False), encoding="utf-8")


def _scenario():
    n = 10
    agents_list = [{"id": i, "occupation": "A", "age": 30 + i, "visitor": False}
                   for i in range(n)]
    # income を internal_locus に線形連動させる(実力 R²>0 を作る)。
    traits = {str(i): {"internal_locus": round(0.1 * i, 3), "nfc": 0.3,
                       "risk_tolerance": 0.4, "money": 50000.0, "wage": 12000.0}
              for i in range(n)}
    events = []
    for i in range(n):
        events.append(_ev(i, i, "arrive", {"node": f"n{i}"}))
        events.append(_ev(i, i, "wage", {"amount": 1000.0 + 500.0 * i}))    # 実力連動
        if i % 2 == 0:
            events.append(_ev(20 + i, i, "detour", {}))                     # 運(偶発)
        if i % 3 == 0:
            events.append(_ev(30 + i, i, "speak", {"hearers": [(i + 1) % n]}))
    return events, agents_list, traits


def test_decompose_schema_and_determinism(tmp_path):
    events, agents_list, traits = _scenario()
    r1 = tmp_path / "p1" / "syn"
    r2 = tmp_path / "p2" / "syn"
    _write_run(r1, events, agents_list, traits)
    _write_run(r2, events, agents_list, traits)
    res = al.analyze(str(r1), str(r1 / "luck"))
    al.analyze(str(r2), str(r2 / "luck"))

    # スキーマ
    for key in ("outcomes", "scatter", "skill_columns", "luck_columns",
                "collinearity_luck_vs_skill", "limitations", "n_modeled"):
        assert key in res, key
    for oc in ("income", "relations", "reputation"):
        assert oc in res["outcomes"]
        r = res["outcomes"][oc]
        assert "skill" in r and "full" in r and "luck_only" in r
        assert "delta_r2_luck" in r and "residual_after_full" in r
    # income は internal_locus 連動 → 実力 R² が出る(None でない)。
    assert res["outcomes"]["income"]["skill"]["r2"] is not None
    # 実力列に money/wage リーク無し
    assert "money" not in res["skill_columns"] and "wage" not in res["skill_columns"]
    # 散布データは住民数ぶん
    assert len(res["scatter"]) == res["n_modeled"] == 10
    assert set(res["luck_columns"]) == {"luck_exposure"}

    # 決定論(同一入力 → 同一 JSON)
    j1 = (r1 / "luck" / "decomposition.json").read_text(encoding="utf-8")
    j2 = (r2 / "luck" / "decomposition.json").read_text(encoding="utf-8")
    assert j1 == j2


def test_decompose_no_traits_degrades(tmp_path):
    """traits が無い旧ランでも例外なく縮退(実力 R²=None で続行)。"""
    agents_list = [{"id": i, "occupation": "A", "age": 30, "visitor": False}
                   for i in range(4)]
    events = [_ev(i, i, "wage", {"amount": 1000.0}) for i in range(4)]
    run_dir = tmp_path / "notraits"
    _write_run(run_dir, events, agents_list, {})
    res = al.analyze(str(run_dir), str(run_dir / "luck"))
    # trait 無し=実力列は age(+occ)のみ or 空 → 例外なく JSON が出る
    assert Path(res["paths"]["json"]).exists()
    assert "outcomes" in res
