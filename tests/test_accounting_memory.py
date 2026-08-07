"""W2-6 ②: analyze_accounting の 25万体対応(全読みの排除)のテスト。

`scripts/analyze_accounting.py` は
  - `l3_snapshots.parquet` を `read_table` で全部載せ、さらに `{step: {agent_id: 残高}}` を
    **全 step ぶん**保持していた(120 枚 × 25万体 = 3,000 万エントリ)
  - 未分類の金の経路を list に貯めてから数えていた
  - `org_ledger` / `finance` を `read_table().to_pylist()` で全列全行読んでいた
  - L1 から使わない座標列(x/y)を読んでいた
ので、在場 25万 × 10 日のランでは読み込みで落ちる。

**本テストの主眼は「速くなったこと」ではなく「答えが 1 ミリも変わっていないこと」**。
そこで**修正前の読み方(全読み)を参照実装として本ファイル内に持ち**、新旧の
`analyze()` レポートが **dict として完全同値**であることを固定する
(数値をハードコードしないので、分類器を触る並行レーンの変更とも衝突しない)。
"""
from __future__ import annotations

import ast
import inspect
import json
import sys
import textwrap
from pathlib import Path

import pyarrow.parquet as pq
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / "scripts", REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import analyze_accounting as aa                      # noqa: E402


# --------------------------------------------------------------------------- #
# 修正前の読み方(全読み)の参照実装 — ここだけは**わざと**旧コードのまま
# --------------------------------------------------------------------------- #
def eager_load_run(run_dir: Path) -> dict:
    """W2-6 ② **以前**の `load_run`(全 part 全列を載せる版)を逐語で再現した参照実装。

    ここが新実装と食い違えば「読み方を変えたら答えが変わった」ということなので、
    以後この参照は**触ってはいけない**(直すべきは本体側)。"""
    run_dir = Path(run_dir)
    cfg = aa._read_config(run_dir)

    events: list[dict] = []
    serve_rows: list[dict] = []
    budget_rows: list[dict] = []
    other_money: list[dict] = []
    needles = tuple(f'"{k}"' for k in aa.MONEY_KEYS)
    want = aa.SCAN_KINDS | {"serve", "public_budget"}
    pf = pq.ParquetFile(run_dir / "l1_events.parquet")
    for batch in pf.iter_batches(batch_size=200_000,
                                 columns=["step", "agent_id", "kind", "x", "y", "payload"]):
        d = batch.to_pydict()
        for st, aid, kind, x, y, pl in zip(d["step"], d["agent_id"], d["kind"],
                                           d["x"], d["y"], d["payload"]):
            if kind not in want:
                if kind not in aa.DERIVED_MONEY_KINDS and pl and any(n in pl for n in needles):
                    try:
                        other_money.append({"kind": kind, "payload": json.loads(pl)})
                    except (TypeError, ValueError):
                        pass
                continue
            try:
                p = json.loads(pl) if pl else {}
            except (TypeError, ValueError):
                p = {}
            rec = {"step": int(st), "agent_id": int(aid), "kind": kind, "payload": p,
                   "x": float(x or 0.0), "y": float(y or 0.0)}
            if kind == "serve":
                serve_rows.append(rec)
            elif kind == "public_budget":
                budget_rows.append({"step": int(st), **p})
            else:
                events.append(rec)

    stock_by_step: dict[int, dict[int, float]] = {}
    snap_path = run_dir / "l3_snapshots.parquet"
    if snap_path.exists():
        t = pq.read_table(snap_path)
        for st, state in zip(t.column("step").to_pylist(), t.column("state").to_pylist()):
            try:
                agents = json.loads(state).get("agents", [])
            except (TypeError, ValueError):
                continue
            stock_by_step[int(st)] = {
                int(a["id"]): float(a.get("money", 0.0)) + float(a.get("account", 0.0) or 0.0)
                for a in agents}

    ledger_rows: list[dict] = []
    led_path = run_dir / "org_ledger.parquet"
    if led_path.exists():
        ledger_rows = pq.read_table(led_path).to_pylist()

    fin_rows: list[dict] = []
    fin_path = run_dir / "finance.parquet"
    if fin_path.exists():
        fin_rows = pq.read_table(fin_path).to_pylist()

    return {"cfg": cfg, "events": events, "serve": serve_rows,
            "budget": budget_rows, "stock": stock_by_step, "ledger": ledger_rows,
            "finance": fin_rows, "other_money": other_money, "run_dir": str(run_dir)}


# --------------------------------------------------------------------------- #
# 代表フィクスチャ(mock 小ラン。経済まわりを一通り点火する)
# --------------------------------------------------------------------------- #
def _mk_run(out: Path, name: str, **extra) -> Path:
    from society.config import load_config
    from society.engine.simulation import Simulation

    dot = ["run.seed=20260805", "run.n_agents=30", "run.n_steps=288",
           f"run.name={name}", "model.backend=mock", "observer.snapshot_every=12",
           "agents.personas_file=data/personas_80.json",
           "organizations.enabled=true",
           "work.service.enabled=true", "work.service.ledger.enabled=true",
           "work.service.indoor_fields=true",
           "government.enabled=true",
           "commerce.inventory.enabled=true", "commerce.inventory.b2b.enabled=true",
           "economy.fixed_cost_daily=300", "economy.visitor_refresh=true"]
    dot += [f"{k}={v}" for k, v in extra.items()]
    d = out / name
    Simulation(load_config(dot), out_dir=d).run()
    return d


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory) -> Path:
    return _mk_run(tmp_path_factory.mktemp("acct_mem"), "am")


@pytest.fixture(scope="module")
def run_dir_sfc(tmp_path_factory) -> Path:
    """IF-E2 案B ON(finance.parquet が出る = 部門別検査①が走る)側の変種。"""
    return _mk_run(tmp_path_factory.mktemp("acct_mem_sfc"), "amsfc",
                   **{"economy.org_accounting.enabled": "true",
                      "economy.org_accounting.sidecar": "true"})


# --------------------------------------------------------------------------- #
# 1. 新旧で答えが 1 ミリも変わらない(本命)
# --------------------------------------------------------------------------- #
def _assert_reports_identical(run_dir: Path) -> dict:
    new = aa.analyze(aa.load_run(run_dir), rel_tol=1e-4)
    old = aa.analyze(eager_load_run(run_dir), rel_tol=1e-4)
    # 差分が出たときに読める形で落とす(dict 全体の repr は巨大なので節ごとに見る)
    for key in sorted(set(new) | set(old)):
        assert new[key] == old[key], f"レポートの '{key}' が新旧で不一致"
    assert new == old
    return new


def test_report_is_identical_before_and_after(run_dir):
    """代表フィクスチャで summary / leak_families / 行列 / 検査が **dict 同値**。"""
    rep = _assert_reports_identical(run_dir)
    # 空回り防止(材料が実際に点火していること)
    assert rep["totals"]["total_flow"] > 0
    assert rep["leak_families"], "漏れの族が 1 つも無い(フィクスチャが薄すぎる)"
    assert rep["checks"][aa.HOUSEHOLD]["observable"]
    assert rep["checks"][aa.HOUSEHOLD]["n_windows"] > 1
    assert rep["revenue_gap"]["total_revenue_est"] > 0


def test_report_is_identical_with_org_accounting_on(run_dir_sfc):
    """finance.parquet がある(= 部門別検査①が走る)ランでも新旧同値。"""
    rep = _assert_reports_identical(run_dir_sfc)
    assert rep["finance"]["n_rows"] >= 2, "finance サイドカーが出ていない"
    assert rep["checks"][aa.ORG]["observable"]


def test_summary_and_leak_families_are_identical(run_dir):
    """明示的に summary 相当(totals / sector_sums / leaks)と leak_families を固定する。"""
    new = aa.analyze(aa.load_run(run_dir), rel_tol=1e-4)
    old = aa.analyze(eager_load_run(run_dir), rel_tol=1e-4)
    assert new["totals"] == old["totals"]
    assert new["sector_sums"] == old["sector_sums"]
    assert new["leaks"] == old["leaks"]
    assert new["leak_families"] == old["leak_families"]
    assert new["unclassified_money_kinds"] == old["unclassified_money_kinds"]
    assert new["event_counts"] == old["event_counts"]


def test_markdown_is_identical(run_dir):
    """レポート本文(Markdown)もバイト同値(印字経路まで含めた同値)。"""
    new = aa.render_markdown(aa.analyze(aa.load_run(run_dir), rel_tol=1e-4))
    old = aa.render_markdown(aa.analyze(eager_load_run(run_dir), rel_tol=1e-4))
    assert new == old


# --------------------------------------------------------------------------- #
# 2. L3 のストリーム化(SnapshotStocks)
# --------------------------------------------------------------------------- #
def test_snapshot_stocks_yields_the_same_pairs_as_the_eager_dict(run_dir):
    eager = eager_load_run(run_dir)["stock"]
    streamed = dict(aa.SnapshotStocks(aa.l1_paths(run_dir, "l3_snapshots")))
    assert streamed == eager
    assert len(streamed) > 1


def test_conserve_household_accepts_dict_and_stream_identically(run_dir):
    """辞書でもストリームでも検査①の出力が完全一致(呼び出し規約の後方互換)。"""
    run = aa.load_run(run_dir)
    eager = eager_load_run(run_dir)
    moves = {}
    for e in run["events"]:
        if e["kind"] in aa.HOUSEHOLD_KINDS:
            moves.setdefault(e["step"], []).extend(
                aa.move_for(e["kind"], e["agent_id"], e["payload"],
                            run["cfg"]["accounts_on"]))
    a = aa.conserve_household(run["stock"], moves, 1e-4)       # SnapshotStocks
    b = aa.conserve_household(eager["stock"], moves, 1e-4)     # 従来の dict
    assert a == b
    assert a["n_windows"] > 1


def test_snapshot_stocks_is_truthy_only_when_readable(tmp_path):
    """空/不在は False・壊れた JSON しかないランも False(observable を偽装しない)。"""
    assert not aa.SnapshotStocks([])
    import pyarrow as pa
    broken = tmp_path / "l3_snapshots.parquet"
    pq.write_table(pa.table({"step": [0, 1], "state": ["{not json", "}}"]}), broken)
    assert not aa.SnapshotStocks([broken])
    ok = tmp_path / "ok.parquet"
    pq.write_table(pa.table({"step": [0], "state": ['{"agents": [{"id": 1, "money": 5.0}]}']}), ok)
    assert aa.SnapshotStocks([ok])
    assert list(aa.SnapshotStocks([ok])) == [(0, {1: 5.0})]


def test_snapshot_stocks_refuses_decreasing_steps(tmp_path):
    """step が減る L3(記録側の不変条件の破れ)は黙って通さない。"""
    import pyarrow as pa
    p = tmp_path / "l3_snapshots.parquet"
    pq.write_table(pa.table({"step": [5, 3],
                             "state": ['{"agents": []}', '{"agents": []}']}), p)
    with pytest.raises(ValueError, match="step"):
        list(aa.SnapshotStocks([p]))


def test_snapshot_stocks_reads_parts_when_canonical_is_absent(tmp_path):
    """canonical が無い(走行中/クラッシュ)ランでも完結 part を index 昇順で読む。"""
    import pyarrow as pa
    d = tmp_path / "run"
    d.mkdir()
    for i, (st, money) in enumerate([(0, 10.0), (12, 20.0)]):
        pq.write_table(
            pa.table({"step": [st],
                      "state": [json.dumps({"agents": [{"id": 1, "money": money}]})]}),
            d / f"l3_snapshots.part-{i:04d}.parquet")
    got = list(aa.SnapshotStocks(aa.l1_paths(d, "l3_snapshots")))
    assert got == [(0, {1: 10.0}), (12, {1: 20.0})]


# --------------------------------------------------------------------------- #
# 3. 未分類検知の畳み込み(MoneyKindTally)
# --------------------------------------------------------------------------- #
def test_tally_matches_the_list_based_definition():
    rows = [
        {"kind": "mystery_pay", "payload": {"amount": 120.0, "note": "x"}},
        {"kind": "mystery_pay", "payload": {"fee": 3.0}},
        {"kind": "spend", "payload": {"amount": 5.0}},          # MONEY_KINDS = 対象外
        {"kind": "ride", "payload": {"fare": 210.0}},           # DERIVED = 対象外
        {"kind": "chat", "payload": {"amount": 0.0}},           # ゼロは金の経路ではない
        {"kind": "flag", "payload": {"amount": True}},          # bool は数値ではない
        {"kind": "weird", "payload": "not-a-dict"},
    ]
    tally = aa.MoneyKindTally()
    for r in rows:
        tally.add(r["kind"], r["payload"])
    assert tally.result() == aa.unclassified_money_kinds(rows)
    assert tally.result() == {"mystery_pay": {"n": 2, "keys": ["amount", "fee"]}}
    # 累積器をそのまま渡しても同じ答え(load_run が返す形)
    assert aa.unclassified_money_kinds(tally) == tally.result()


def test_load_run_returns_a_tally_not_a_list(run_dir):
    """other_money は list ではなく畳んだ累積器(= 件数に比例して増えない)。"""
    run = aa.load_run(run_dir)
    assert isinstance(run["other_money"], aa.MoneyKindTally)
    assert aa.unclassified_money_kinds(run["other_money"]) == \
        aa.unclassified_money_kinds(eager_load_run(run_dir)["other_money"])


# --------------------------------------------------------------------------- #
# 4. 列射影(org_ledger / L1)
# --------------------------------------------------------------------------- #
def test_ledger_projection_is_sufficient(run_dir):
    """射影した 4 列だけで revenue_gap が全列版と一致する(= 他の列を使っていない証明)。"""
    full = eager_load_run(run_dir)
    assert full["ledger"], "org_ledger が空(フィクスチャが薄すぎる)"
    assert len(full["ledger"][0]) > len(aa.LEDGER_COLUMNS), "射影の意味が無い(全列 = 4 列)"
    trimmed = [{k: r[k] for k in aa.LEDGER_COLUMNS if k in r} for r in full["ledger"]]
    a = aa.revenue_gap(full["ledger"], full["serve"], [])
    b = aa.revenue_gap(trimmed, full["serve"], [])
    assert a == b
    assert set(aa.load_run(run_dir)["ledger"][0]) <= set(aa.LEDGER_COLUMNS)


def test_l1_scan_columns_exclude_coordinates():
    """会計は座標を使わないので L1 から読まない(40.6 億行ぶんの復号を捨てる)。"""
    assert "x" not in aa.L1_SCAN_COLUMNS and "y" not in aa.L1_SCAN_COLUMNS
    assert set(aa.L1_SCAN_COLUMNS) == {"step", "agent_id", "kind", "payload"}


def test_load_run_raises_when_l1_is_missing(tmp_path):
    """L1 が 1 本も無いランは**黙って空のレポート**を出さずに落ちる。"""
    d = tmp_path / "empty"
    d.mkdir()
    with pytest.raises(FileNotFoundError):
        aa.load_run(d)


# --------------------------------------------------------------------------- #
# 5. 有界性の構造的根拠(AST)
# --------------------------------------------------------------------------- #
def _names(func) -> set[str]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            out.add(node.attr)
        elif isinstance(node, ast.Name):
            out.add(node.id)
    return out


#: 表を丸ごと 1 枚に載せる識別子(= 25万体で落ちる原因)。
WHOLE_READ = {"read_table", "to_pylist", "to_pydict"}


@pytest.mark.parametrize("func", [
    aa.load_run, aa.SnapshotStocks.__iter__, aa._read_rows, aa.conserve_household,
])
def test_read_paths_never_materialize_a_whole_table(func):
    got = _names(func) & WHOLE_READ
    assert not got, f"{func.__qualname__} に全読み識別子がある: {sorted(got)}"


def test_module_has_no_read_table_call_left():
    """本体(参照実装はテスト側にしか無い)から全読み API が **コードとして** 消えている。

    AST で見るので、docstring の「旧: read_table で…」という説明文には反応しない。"""
    tree = ast.parse(Path(aa.__file__).read_text(encoding="utf-8"))
    used = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    used |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert not (used & WHOLE_READ), f"全読み API が残っている: {sorted(used & WHOLE_READ)}"


def test_conserve_household_holds_at_most_two_snapshots():
    """検査①が保持するスナップショットは高々 2 枚(隣接窓)であることの構造固定。"""
    src = inspect.getsource(aa.conserve_household)
    assert "prev = (s1, a1)" in src, "直前 1 枚の引き継ぎ形になっていない"
    assert "sorted(stock_by_step)" not in src, "全 step を同時に並べている"
