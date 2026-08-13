#!/usr/bin/env python
"""所有(登記簿)の事後解析 — 所有権レイヤー O1+O3。

    python scripts/analyze_assets.py runs/<name> [--out runs/<name>/analysis]

**読み出し専用**: `runs/<name>/assets_ledger.json`(登記簿のスナップショット)・
`l1_events.parquet`(相続・権利移転)・`l3_snapshots.parquet`(現金)を読むだけで、
シム本体・schema・既存の解析スクリプトを 1 バイトも触らない。
`observer/metrics_spec.py` の**凍結 14 ファイル**にも本スクリプトは含まれない
(= metrics_spec_hash は無風)。

★設計上の要点(研究文書 §6-3): **観測はシムに走査を足さない**。Gini も所有ネットワークも
  台帳スナップショットの純関数で、ここ(事後層)でしか計算しない。

何を出すか
----------
**(A) 総資産 Gini / Lorenz**(研究文書 §7-1)。
  現金のみの Gini(現行の資産観測が見ているもの)と、**現金 + 保有資産の評価額**の Gini を
  並べる。住宅資産は現実の家計資産の過半を占めるので、住戸を勘定に入れない資産格差は
  原理的に歪んでいる —— その歪みの大きさそのものが本表の主張である。
  ★資産の評価額は**外生の代表価格**(`--price-dwelling` / `--price-vehicle`)であって、
    シムの中に価格形成は無い(中古市場の内生化は O5)。**推定であることを出力に明記する**。

**(B) 保有期間分布**(研究文書 §7-3)。
  各権利行の `since`(取得 step)から観測終了までの保有 step 数。移転が起きた行は L1 の
  `inheritance` / `asset_transfer` に現れるので、**移転の回転率**(件/日)も併記する。
  ★O1 では移転が相続と持ち家者の転居しか無いので、10 日ランでは分布はほぼ「初期配賦のまま」
    = 中央値が観測長に張り付く。それが正しい(所有者はラン中静的というユーザー決定)。

**(C) 所有ネットワーク**(研究文書 §7-2 / Vitali-Glattfelder-Battiston の bow-tie 解析)。
  agent–asset–org の**二部グラフ**を party 側へ射影した集中度: 保有数の Gini・HHI・
  上位 10 主体シェア・party 型別(個人 / 組織 / 域外 RoW)の内訳。
  ★「創業者が何を所有して始めたか」(MEMORY: org-emergence-goal)の軸がここに載る。

**(D) 資産保存則の検査**(研究文書 §4-3)。
  `Σ 所有者別保有数 + K5 − RoW 生成 − 境界流入 = 初期ストック` をカテゴリ別に再計算して
  残差を出す。**境界流入**(`rot_in`)はプール回転で day1 以降に街へ入ってきた個体の車両で、
  「初期ストックの数え漏れ」ではなく域外→域内の実在フロー(SNA 2008 の境界フロー)。
  3 項とも 0 の旧ラン(サイドカーに欄が無い)では式が `live − stock0` へ退化する。
  シム側(`summary.assets.conservation`)と**独立に**台帳の行から数え直すので、
  台帳の書き手にバグがあれば必ず食い違う。

正直な限界(4 件)
------------------
1. **価格は外生**。(A) の資産評価はシムの中に存在しない代表価格を掛けたもので、地区差も
   築年も無い。順位付け(誰が資産を持っているか)には効くが、水準の主張には使えない。
2. **家賃・敷金の流れと接続していない**。台帳上の家主が域内 org でも、その org の預金は
   1 円も増えない(assets.py の正直な限界 2)。所有ネットワークと金のネットワークを
   突き合わせるには O4(lease 行)+ O5 が要る。
3. **L3 の現金は在場者だけ**。プール回転 ON のランでは、スナップショット時点で街に居ない
   個体の現金が (A) の分母から落ちる。台帳側の保有は残るので、**両者の母集団は厳密には
   一致しない**(出力に両方の人数を出す)。
4. **保有期間は打ち切られている**(right-censored)。ラン終了時点でまだ保有中の行は
   「少なくともこの長さ」でしかない。中央値は観測長で頭打ちになる。
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as stat
import sys
from collections import Counter, defaultdict
from pathlib import Path

_SCRIPTS = str(Path(__file__).resolve().parent)
if _SCRIPTS not in sys.path:                     # l1_stream(逐次読み)用
    sys.path.insert(0, _SCRIPTS)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCHEMA = 1

#: 資産の代表価格 [円]。**外生の推定値**(シムの中に価格形成は無い = O5 の仕事)。
#: 住戸 = 東京 23 区の中古マンション平均取引価格の桁 / 車両 = 中古乗用車の平均取引価格の桁。
PRICE_DWELLING = 50_000_000.0
PRICE_VEHICLE = 1_500_000.0

WANT_KINDS = ("inheritance", "asset_transfer")


# --------------------------------------------------------------------------- #
# 読み込み(読み取り専用)
# --------------------------------------------------------------------------- #
def load_ledger(run_dir) -> dict | None:
    """`assets_ledger.json`(登記簿スナップショット)。world.assets OFF のランでは None。"""
    path = Path(run_dir) / "assets_ledger.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_transfers(run_dir) -> list[dict]:
    """L1 の移転イベント(相続・権利移転)。l1 が無いランでは空。"""
    import l1_stream as ls
    if not ls.l1_paths(run_dir):
        return []
    out: list[dict] = []
    for e in ls.iter_events(run_dir, columns=["step", "agent_id", "kind", "payload"],
                            kinds=WANT_KINDS):
        payload = e["payload"]
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except ValueError:
                payload = {}
        out.append({"step": int(e["step"]), "agent_id": int(e["agent_id"]),
                    "kind": str(e["kind"]), "payload": payload or {}})
    return out


def load_last_cash(run_dir) -> dict[int, float]:
    """L3 スナップショットの**最後の 1 枚**の現金(money + account)。無ければ空。

    1 枚ずつ読んで直前の枚を捨てるので、常駐は **O(在場人数)**
    (`analyze_accounting.SnapshotStocks` と同じ流儀)。
    """
    from l1_stream import iter_table_columns, l1_paths
    paths = l1_paths(run_dir, stem="l3_snapshots")
    last: dict[int, float] = {}
    for path in paths:
        for d in iter_table_columns(path, ["step", "state"], batch_rows=1):
            raw = d["state"][0]
            try:
                agents = json.loads(raw).get("agents", []) if raw else []
            except (TypeError, ValueError):
                continue
            if agents:
                last = {int(a["id"]): float(a.get("money", 0.0))
                        + float(a.get("account", 0.0) or 0.0) for a in agents}
    return last


# --------------------------------------------------------------------------- #
# 指標
# --------------------------------------------------------------------------- #
def gini(values) -> float:
    """Gini 係数(0=平等 .. 1=集中)。負値・空は 0 を返す(捏造しない)。"""
    vals = sorted(float(v) for v in values if float(v) >= 0.0)
    n = len(vals)
    total = sum(vals)
    if n == 0 or total <= 0.0:
        return 0.0
    cum = sum(i * v for i, v in enumerate(vals, start=1))
    return round((2.0 * cum) / (n * total) - (n + 1.0) / n, 6)


def lorenz(values, points: int = 11) -> list[list[float]]:
    """Lorenz 曲線の代表点 [(人口累積比, 富の累積比), …](既定 = 10 分位 + 原点)。"""
    vals = sorted(float(v) for v in values if float(v) >= 0.0)
    n = len(vals)
    total = sum(vals)
    if n == 0 or total <= 0.0:
        return []
    out = []
    for k in range(points):
        idx = int(round(n * k / (points - 1)))
        out.append([round(k / (points - 1), 4),
                    round(sum(vals[:idx]) / total, 6)])
    return out


def hhi(counts) -> float:
    """Herfindahl-Hirschman 指数(シェアの二乗和。1=独占)。"""
    vals = [float(v) for v in counts if float(v) > 0.0]
    total = sum(vals)
    if total <= 0.0:
        return 0.0
    return round(sum((v / total) ** 2 for v in vals), 6)


def top_share(counts, k: int = 10) -> float:
    """上位 k 主体が占めるシェア。"""
    vals = sorted((float(v) for v in counts), reverse=True)
    total = sum(vals)
    if total <= 0.0:
        return 0.0
    return round(sum(vals[:k]) / total, 6)


def party_kind(party: str) -> str:
    """party id → 型。``society.assets.party_kind`` の**鏡**(解析側は src を import しない)。"""
    p = str(party)
    if p == "row":
        return "row"
    if p.startswith("a:"):
        return "agent"
    if p.startswith("o:"):
        return "org"
    return "unknown"


# --------------------------------------------------------------------------- #
# 集計
# --------------------------------------------------------------------------- #
def summarize(ledger: dict, transfers: list, cash: dict, *,
              price_dwelling: float = PRICE_DWELLING,
              price_vehicle: float = PRICE_VEHICLE,
              last_step: int | None = None) -> dict:
    """登記簿 + L1 + L3 → 解析結果 dict。"""
    rows = list(ledger.get("rows") or ())
    prices = {"dwelling": float(price_dwelling), "vehicle": float(price_vehicle)}

    # ---- (C) 所有ネットワーク: party 側への射影 ----
    by_party: dict[str, Counter] = defaultdict(Counter)
    by_cat: Counter = Counter()
    since_by_party: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        if str(r.get("right")) != "own":
            continue
        party, cat = str(r["party"]), str(r["cat"])
        by_party[party][cat] += 1
        by_cat[cat] += 1
        since_by_party[party].append(int(r.get("since", 0)))
    holdings = {p: sum(c.values()) for p, c in by_party.items()}
    kind_counts: Counter = Counter()
    for p, n in holdings.items():
        kind_counts[party_kind(p)] += n

    network = {
        "n_assets": int(sum(by_cat.values())),
        "n_holders": len(holdings),
        "by_category": {k: int(v) for k, v in sorted(by_cat.items())},
        "by_party_kind": {k: int(v) for k, v in sorted(kind_counts.items())},
        "holding_gini": gini(holdings.values()),
        "holding_hhi": hhi(holdings.values()),
        "top10_share": top_share(holdings.values(), 10),
        "max_holding": (max(holdings.values()) if holdings else 0),
        # 個人 party だけの集中(域外 RoW という 1 主体が全部を潰さない形も併記する)
        "agent_holding_gini": gini([n for p, n in holdings.items()
                                    if party_kind(p) == "agent"]),
        "org_holding_gini": gini([n for p, n in holdings.items()
                                  if party_kind(p) == "org"]),
    }

    # ---- (A) Gini: 現金のみ vs 現金 + 資産 ----
    asset_value: dict[int, float] = {}
    for p, c in by_party.items():
        if party_kind(p) != "agent":
            continue
        try:
            aid = int(p.split(":", 1)[1])
        except (IndexError, ValueError):
            continue
        asset_value[aid] = sum(prices.get(cat, 0.0) * n for cat, n in c.items())
    pop = sorted(set(cash) | set(asset_value))
    cash_only = [float(cash.get(a, 0.0)) for a in pop]
    with_assets = [float(cash.get(a, 0.0)) + float(asset_value.get(a, 0.0))
                   for a in pop]
    inequality = {
        "population": len(pop),
        "n_cash_observed": len(cash),
        "n_asset_holders": len(asset_value),
        "prices_assumed": prices,
        "gini_cash_only": gini(cash_only),
        "gini_cash_plus_assets": gini(with_assets),
        "gini_shift": round(gini(with_assets) - gini(cash_only), 6),
        "lorenz_cash_only": lorenz(cash_only),
        "lorenz_cash_plus_assets": lorenz(with_assets),
        "asset_share_of_wealth": (
            round(sum(asset_value.values())
                  / max(1e-9, sum(with_assets)), 6) if with_assets else 0.0),
    }

    # ---- (B) 保有期間(right-censored)と流通速度 ----
    end = int(last_step if last_step is not None
              else max([int(t["step"]) for t in transfers] or [0]))
    held = [max(0, end - int(r.get("since", 0))) for r in rows
            if str(r.get("right")) == "own"]
    tr_kinds: Counter = Counter()
    for t in transfers:
        if t["kind"] == "inheritance":
            tr_kinds["inheritance"] += int(t["payload"].get("assets", 0) or 0)
        else:
            tr_kinds[str(t["payload"].get("kind", "?"))] += 1
    n_moved = int(sum(tr_kinds.values()))
    tenure = {
        "observed_steps": end,
        "n_rows": len(held),
        "mean_held_steps": (round(stat.fmean(held), 2) if held else 0.0),
        "median_held_steps": (round(stat.median(held), 2) if held else 0.0),
        "censored": True,           # ラン終了時点でまだ保有中 = 「少なくともこの長さ」
        "n_transfers": n_moved,
        "transfers_by_kind": {k: int(v) for k, v in sorted(tr_kinds.items())},
        "turnover_rate": (round(n_moved / max(1, len(held)), 8) if held else 0.0),
    }

    # ---- (D) 資産保存則(台帳の行から**独立に**数え直す)----
    # 恒等式は `live + k5 − born − rot_in = stock0`(カテゴリ別)。
    # ★`born`(製造・輸入)/`k5`(廃棄・滅失)/`rot_in`(**境界流入** = プール回転で街に
    #   入ってきた既存資産 = A12)はサイドカーの累積カウンタで、**旧ランには存在しない**。
    #   欠落は 0 として扱う(= 3 項とも 0 だった第109 のランでは式が `live − stock0` に
    #   退化して従来と 1 文字も違わない結果になる)。
    stock0 = {k: int(v) for k, v in (ledger.get("stock0") or {}).items()}
    born_t = {k: int(v) for k, v in (ledger.get("born") or {}).items()}
    k5_t = {k: int(v) for k, v in (ledger.get("k5") or {}).items()}
    rot_t = {k: int(v) for k, v in (ledger.get("rot_in") or {}).items()}
    conservation = {}
    for cat in sorted(set(by_cat) | set(stock0) | set(born_t) | set(k5_t) | set(rot_t)):
        live = int(by_cat.get(cat, 0))
        s0 = int(stock0.get(cat, 0))
        born = int(born_t.get(cat, 0))
        k5 = int(k5_t.get(cat, 0))
        rin = int(rot_t.get(cat, 0))
        conservation[cat] = {"live": live, "born": born, "k5": k5, "rot_in": rin,
                             "stock0": s0, "residual": live + k5 - born - rin - s0}
    sim_side = ledger.get("conservation") or {}

    return {
        "schema": SCHEMA,
        "ledger": {"n_rows": int(ledger.get("n_rows", len(rows))),
                   "org_source": str(ledger.get("org_source", "")),
                   "n_landlord_orgs": len(ledger.get("landlords") or ())},
        "inequality": inequality,
        "tenure": tenure,
        "network": network,
        "conservation": conservation,
        "conservation_sim_side": sim_side,
        "conservation_agrees": all(
            int(sim_side.get(c, {}).get("live", -1)) == v["live"]
            for c, v in conservation.items()) if sim_side else None,
        "inheritance": {
            "n_events": sum(1 for t in transfers if t["kind"] == "inheritance"),
            "to_household": sum(1 for t in transfers if t["kind"] == "inheritance"
                                and t["payload"].get("to") == "household"),
            "to_row": sum(1 for t in transfers if t["kind"] == "inheritance"
                          and t["payload"].get("to") == "row"),
            "money_total": round(sum(float(t["payload"].get("amount", 0.0) or 0.0)
                                     for t in transfers
                                     if t["kind"] == "inheritance"), 1),
        },
    }


# --------------------------------------------------------------------------- #
# 出力
# --------------------------------------------------------------------------- #
def render(res: dict) -> str:
    iq, tn, nw = res["inequality"], res["tenure"], res["network"]
    L = ["# 所有(登記簿)の解析 — 所有権レイヤー O1+O3", "",
         f"- 台帳の権利行: **{res['ledger']['n_rows']}** 行"
         f"(域内不動産 org {res['ledger']['n_landlord_orgs']} 社 / "
         f"特定方法 `{res['ledger']['org_source']}`)",
         f"- カテゴリ別: {nw['by_category']}",
         f"- 主体型別の保有: {nw['by_party_kind']}", "",
         "## (A) 総資産 Gini — 現金だけ見ると何を見落とすか", "",
         "| 指標 | 値 |", "|---|---|",
         f"| 現金のみの Gini | {iq['gini_cash_only']} |",
         f"| 現金 + 資産の Gini | {iq['gini_cash_plus_assets']} |",
         f"| 差(資産を入れて動いた量) | {iq['gini_shift']} |",
         f"| 総資産に占める資産(非現金)の割合 | {iq['asset_share_of_wealth']} |",
         f"| 母集団 / 現金観測 / 資産保有者 | {iq['population']} / "
         f"{iq['n_cash_observed']} / {iq['n_asset_holders']} |", "",
         f"> ★資産の評価は**外生の代表価格**({iq['prices_assumed']})を掛けたもので、",
         "> シムの中に価格形成は無い(中古市場の内生化は O5)。順位付けには効くが、",
         "> 水準の主張には使えない。", "",
         "## (B) 保有期間と流通", "",
         f"- 観測長: **{tn['observed_steps']}** step(右側打ち切りあり)",
         f"- 平均保有 {tn['mean_held_steps']} step / 中央値 {tn['median_held_steps']} step",
         f"- 移転: **{tn['n_transfers']}** 件 {tn['transfers_by_kind']}"
         f"(回転率 {tn['turnover_rate']} 件/行)", "",
         "> ★O1 の移転は相続と持ち家者の転居しか無い(所有者はラン中静的というユーザー決定)。",
         "> 中央値が観測長に張り付くのが**正しい**状態である。", "",
         "## (C) 所有ネットワーク(agent–asset–org の二部グラフを主体側へ射影)", "",
         "| 指標 | 値 |", "|---|---|",
         f"| 主体数 | {nw['n_holders']} |",
         f"| 保有数の Gini | {nw['holding_gini']} |",
         f"| HHI | {nw['holding_hhi']} |",
         f"| 上位 10 主体シェア | {nw['top10_share']} |",
         f"| 最大保有 | {nw['max_holding']} |",
         f"| 個人だけの Gini | {nw['agent_holding_gini']} |",
         f"| 組織だけの Gini | {nw['org_holding_gini']} |", "",
         "## (D) 資産保存則(台帳の行から独立に数え直す)", ""]
    ok = True
    L += ["| カテゴリ | 生存行 | 製造 | 廃棄 | 境界流入 | 初期ストック | 残差 |",
          "|---|---|---|---|---|---|---|"]
    for cat, c in sorted(res["conservation"].items()):
        ok = ok and c["residual"] == 0
        L.append(f"| {cat} | {c['live']} | {c.get('born', 0)} | {c.get('k5', 0)} | "
                 f"{c.get('rot_in', 0)} | {c['stock0']} | {c['residual']} |")
    L += ["", f"- 判定: **{'PASS' if ok else 'FAIL'}**"
          f"(シム側の申告と一致: {res['conservation_agrees']})", "",
          "## 相続(O3)", "",
          f"- 相続イベント **{res['inheritance']['n_events']}** 件"
          f"(世帯へ {res['inheritance']['to_household']} / "
          f"国庫=RoW へ {res['inheritance']['to_row']})",
          f"- 移った現金の合計 **{res['inheritance']['money_total']}** 円", "",
          "> 相続人不存在の遺産が国庫へ出るのは民法 959 条どおり。0 件のランは",
          "> 「死が 1 件も起きなかった」か「world.assets が OFF だった」のどちらかで、",
          "> summary.json の `assets.deaths` と突き合わせれば区別できる。"]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="所有(登記簿)の解析(読み取り専用)")
    ap.add_argument("run_dir", help="ラン出力ディレクトリ(assets_ledger.json を含む)")
    ap.add_argument("--out", default=None, help="出力先(既定: <run_dir>/analysis)")
    ap.add_argument("--price-dwelling", type=float, default=PRICE_DWELLING,
                    help="住戸の代表価格 [円](★外生の推定値)")
    ap.add_argument("--price-vehicle", type=float, default=PRICE_VEHICLE,
                    help="車両の代表価格 [円](★外生の推定値)")
    ap.add_argument("--last-step", type=int, default=None,
                    help="保有期間の観測終了 step(既定: L1 の最終 step)")
    args = ap.parse_args()

    ledger = load_ledger(args.run_dir)
    if ledger is None:
        print(f"[assets] assets_ledger.json が無い: {args.run_dir}\n"
              "  world.assets.enabled=false のランには登記簿が存在しない(= 解析対象なし)。")
        return 1
    transfers = load_transfers(args.run_dir)
    cash = load_last_cash(args.run_dir)
    last_step = args.last_step
    if last_step is None:
        try:
            import l1_stream as ls
            last_step = ls.max_step(args.run_dir) if ls.l1_paths(args.run_dir) else 0
        except Exception:                            # noqa: BLE001(L1 が無いラン)
            last_step = 0
    res = summarize(ledger, transfers, cash,
                    price_dwelling=args.price_dwelling,
                    price_vehicle=args.price_vehicle, last_step=last_step)
    out_dir = args.out or os.path.join(args.run_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    jpath = os.path.join(out_dir, "assets.json")
    mpath = os.path.join(out_dir, "assets_report.md")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2, sort_keys=True)
    md = render(res)
    with open(mpath, "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    print(f"[written] {jpath}")
    print(f"[written] {mpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
