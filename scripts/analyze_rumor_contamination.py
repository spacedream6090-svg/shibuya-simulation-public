#!/usr/bin/env python
"""噂の混線の切り分けオーバーレイ — IF-C 残課題(台帳 PENDING §4)。

    python scripts/analyze_rumor_contamination.py runs/<name> [--out <dir>]

何を解く問題か
--------------
``information.rumors.enabled=true`` のランでは、噂の伝播が既存の ``transmission``
イベントとして L1 に出る(``rumors.on_talk`` → ``ItemStore.transmit``)。ところが
``observer/measure.py`` / ``stream.py`` / ``echo.py`` の ``c_transmission`` /
``n_transmission`` / ``transmission_novel_rate`` は **item の kind を見ずに全
transmission を数える**ので、語彙伝播の指標に噂ぶんが混ざる(``rumors.py`` の
docstring 「正直な限界」4 件目)。これらは ``observer/metrics_spec.py`` の**凍結 14
ファイル**なので 1 バイトも触れない。

そこで **解析側の読み取り専用オーバーレイ**で切り分ける。

★最重要の設計判断: **指標を再定義しない**。
    「噂を除いた c_transmission」を本スクリプトが自前で数え直すと、凍結指標と定義が
    ズレたときに気づけない(= 凍結の意味が消える)。そこで本スクリプトは
    **凍結関数そのもの**(``measure.agent_features`` / ``measure.echo_novelty``)を
      (1) L1 全件            → **凍結値**(正典。ラン出力の値と同一の計算)
      (2) 噂 transmission を除いた L1 → **オーバーレイ値**(診断)
    の 2 回呼び、その差を報告する。差分の意味は「凍結指標の定義そのままで、入力から
    噂の伝播イベントだけを抜いたらどうなるか」に閉じる。

**凍結値が正典・オーバーレイは診断**であり、本スクリプトは凍結指標を置き換えない。
本ファイルは ``SPEC_FILES`` に**含まれない**(= metrics_spec_hash は無風)。

噂の識別(実測で確定)
----------------------
``ItemStore.new_item`` は ``item_id = f"{kind}-{seq:05d}"`` を発行し、``rumors.KIND
= "rumor"`` なので噂 Item の id は必ず ``rumor-`` で始まる。したがって識別子は
**item_id の接頭辞**であり、ラベルでも専用 L1 種でもない(``analyze_rumors.py`` の
``RUMOR_PREFIX`` と同一の源)。``rumor_born`` / ``rumor_stifle`` は噂**専用**の L1 種
だが、凍結指標はこの 2 種を 1 件も読まないので混線の経路にはならない。

何が混ざり、何が混ざらないか(コード実査の結果)
------------------------------------------------
``measure.agent_features`` の Y_external 4 成分のうち、
  - ``c_transmission``  … **混ざる**。噂 transmission ごとに ``payload["from"]`` へ +1。
      ★ただし「creator への +1」は起きない: creator 表は ``vocab_coin`` / ``label_coin``
        からしか作られず、噂 Item はその 2 種を出さずに直接 ``new_item`` されるため
        ``creator.get(rumor-XXXXX)`` は None。= 噂 1 件の混入は **+1 で頭打ち**。
  - ``c_label_adopt``   … 混ざらない(``last_from`` は (item_id, agent) 鍵で名前空間が別。
      噂 item_id に対する ``label_adopt`` は発生しない)。
  - ``c_new_relations`` … 混ざらない(speak/hear/dm 由来。噂は既存の会話に相乗りするだけで
      会話そのものを増やさない)。
  - ``c_sns_reach``     … 実質混ざらない(``sns_post`` の ``items`` に噂 item_id は載らない
      = 噂は発話テキストにも投稿アイテムにも注入されない)。**が、判定はコードに任せる**
      (本スクリプトは自前で場合分けせず凍結関数を 2 回呼ぶだけなので、この見立てが外れて
      いれば差分に出る)。
したがって ``Y_external`` の汚染は ``c_transmission`` 経由に一本化されるはずで、
出力の ``y_external_total`` と ``c_transmission_total`` の Δ が一致するかどうかが
**その見立ての自己検査**になる(``checks.delta_flows_through_c_transmission``)。

出力(JSON + 人間可読 md)
--------------------------
(a) 噂由来を除いた再計算値      … ``metrics[*].overlay``
(b) 混入率                      … ``events.contamination_rate``(件数ベース)と
                                   ``metrics[*].contaminated_share``(指標ベース)
(c) 凍結値との差分              … ``metrics[*].frozen`` / ``delta``

正直な限界(4 件)
------------------
1. **L2 の runtime 値は再計算できない**。``transmission_novel`` / ``transmission_novel_rate``
   の L2 列は走行時に窓付きで積まれた値で、事後の ``echo_novelty``(ラン全体窓)とは
   分母が違う(``measure.echo_novelty`` の docstring が ``echo_max`` について同じ注意を
   書いている)。本スクリプトの frozen/overlay は**どちらも事後関数**の値どうしの比較で、
   L2 列の値そのものではない。参考として L2 の最終行を ``l2_final`` に**並記だけ**する。
2. **除去は transmission イベント単位**。噂が「会話の枠」を消費して語彙伝播を押し出した
   ぶん(機会費用)は復元できない = オーバーレイは「噂が無かった世界」ではなく
   「噂の伝播を数えなかった場合」の値である。
3. ``rumor_born`` の初期 knower(目撃者)は transmission を経ずに知るので、噂の到達
   人数と本スクリプトの除去件数は一致しない(到達人数は ``analyze_rumors.py`` の担当)。
4. 旧ラン互換: ``item_id`` を持たない transmission(あり得ないが)は語彙側に数える
   (欠測を噂と決めつけない)。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))
if _HERE not in sys.path:                  # l1_stream(W2-2 の共有逐次読み)用
    sys.path.insert(0, _HERE)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from society.observer import measure as m  # noqa: E402

SCHEMA = 1
# 噂 Item の item_id 接頭辞。単一の源は society/rumors.py の KIND="rumor" +
# ItemStore.new_item の f"{kind}-{seq:05d}"(analyze_rumors.RUMOR_PREFIX と同値)。
RUMOR_PREFIX = "rumor-"
# 混線しうる L2 列(参考並記のみ。再計算はしない=限界 1)
_L2_ECHO_COLS = ("transmission_novel", "transmission_novel_rate")


# --------------------------------------------------------------------------- #
# 識別と分離
# --------------------------------------------------------------------------- #
def is_rumor_item(item_id) -> bool:
    """item_id が噂のものか(接頭辞判定)。None/非文字列は False = 語彙側に数える。"""
    return isinstance(item_id, str) and item_id.startswith(RUMOR_PREFIX)


def is_rumor_transmission(event: dict) -> bool:
    """その L1 イベントが**噂の伝播**か。"""
    return (event.get("kind") == "transmission"
            and is_rumor_item((event.get("payload") or {}).get("item_id")))


def strip_rumor_transmissions(events: list[dict]) -> list[dict]:
    """噂の ``transmission`` だけを取り除いた L1(行順は保つ)。

    落とすのは transmission のみ。``rumor_born`` / ``rumor_stifle`` は凍結指標が
    1 件も読まないので**残しても値に影響しない**が、「凍結関数への入力から余計な
    ものを引かない」= 差分の原因を transmission 1 経路に限定するために残す。
    """
    return [e for e in events if not is_rumor_transmission(e)]


# --------------------------------------------------------------------------- #
# 差分の組み立て
# --------------------------------------------------------------------------- #
def _delta(frozen, overlay, ndigits: int = 6) -> dict:
    """{frozen, overlay, delta, contaminated_share}。frozen=0 なら share は None。"""
    f = float(frozen)
    o = float(overlay)
    d = f - o
    return {
        "frozen": round(f, ndigits),
        "overlay": round(o, ndigits),
        "delta": round(d, ndigits),
        "contaminated_share": (round(d / f, ndigits) if f else None),
    }


def _sum_col(rows: list[dict], col: str) -> float:
    return float(sum(r.get(col, 0) or 0 for r in rows))


def _l2_final(run_dir: str) -> dict:
    """混線しうる L2 列の**最終行**(参考並記。無ければ空 dict)。"""
    l2 = m.load_l2(run_dir)
    if not l2 or not l2.get("step"):
        return {}
    order = sorted(range(len(l2["step"])), key=lambda i: int(l2["step"][i]))
    last = order[-1]
    return {c: l2[c][last] for c in _L2_ECHO_COLS if c in l2}


def compare(events: list[dict], agents_meta: list[dict] | None = None) -> dict:
    """凍結指標(全件)と噂除外オーバーレイを並記した比較 dict。

    **凍結関数を 2 回呼ぶだけ**で、指標式は 1 つも再定義しない(module docstring)。

    ``agents_meta`` が空のときは**全件 L1 から**名簿を合成して両方に同じものを渡す:
    ``agent_features`` は名簿が無いと「L1 に現れた agent_id」を行にするので、噂の伝播
    でしか登場しない個体が overlay 側から消えて行集合がズレる(合計は 0 なので変わらないが
    個体別の突き合わせが崩れる)。行集合を固定して frozen/overlay を厳密に対応させる。
    """
    clean = strip_rumor_transmissions(events)
    if not agents_meta:
        agents_meta = [{"id": aid} for aid in
                       sorted({e["agent_id"] for e in events
                               if isinstance(e.get("agent_id"), int)
                               and e["agent_id"] >= 0})]

    feats_frozen = m.agent_features(events, agents_meta)
    feats_overlay = m.agent_features(clean, agents_meta)
    echo_frozen = m.echo_novelty(events)
    echo_overlay = m.echo_novelty(clean)

    n_all = sum(1 for e in events if e.get("kind") == "transmission")
    n_rumor = sum(1 for e in events if is_rumor_transmission(e))
    n_born = sum(1 for e in events if e.get("kind") == "rumor_born")
    n_stifle = sum(1 for e in events if e.get("kind") == "rumor_stifle")

    metrics = {
        # measure.agent_features(Y_external の 4 成分 + 合計)
        "c_transmission_total": _delta(_sum_col(feats_frozen, "c_transmission"),
                                       _sum_col(feats_overlay, "c_transmission")),
        "c_label_adopt_total": _delta(_sum_col(feats_frozen, "c_label_adopt"),
                                      _sum_col(feats_overlay, "c_label_adopt")),
        "c_new_relations_total": _delta(_sum_col(feats_frozen, "c_new_relations"),
                                        _sum_col(feats_overlay, "c_new_relations")),
        "c_sns_reach_total": _delta(_sum_col(feats_frozen, "c_sns_reach"),
                                    _sum_col(feats_overlay, "c_sns_reach")),
        "y_external_total": _delta(_sum_col(feats_frozen, "Y_external"),
                                   _sum_col(feats_overlay, "Y_external")),
        # measure.echo_novelty(エコー計測側の伝播 3 量)
        "n_transmission": _delta(echo_frozen["n_transmission"],
                                 echo_overlay["n_transmission"]),
        "n_transmission_novel": _delta(echo_frozen["n_transmission_novel"],
                                       echo_overlay["n_transmission_novel"]),
        "transmission_novel_rate": _delta(echo_frozen["transmission_novel_rate"],
                                          echo_overlay["transmission_novel_rate"]),
    }

    # 個体別の汚染(c_transmission が動いた個体だけ。Y の順位が入れ替わるかの材料)
    by_id = {int(f["id"]): f for f in feats_overlay if f.get("id") is not None}
    per_agent = []
    for f in feats_frozen:
        aid = f.get("id")
        o = by_id.get(int(aid)) if aid is not None else None
        if o is None:
            continue
        d = int(f["c_transmission"]) - int(o["c_transmission"])
        if d:
            per_agent.append({
                "id": int(aid),
                "c_transmission_frozen": int(f["c_transmission"]),
                "c_transmission_overlay": int(o["c_transmission"]),
                "delta": d,
                "Y_external_frozen": float(f["Y_external"]),
                "Y_external_overlay": float(o["Y_external"]),
            })
    per_agent.sort(key=lambda r: (-r["delta"], r["id"]))

    return {
        "schema": SCHEMA,
        "rumors_detected": bool(n_rumor or n_born or n_stifle),
        "events": {
            "transmission_total": n_all,
            "transmission_rumor": n_rumor,
            "transmission_vocab": n_all - n_rumor,
            "rumor_born": n_born,
            "rumor_stifle": n_stifle,
            # (b) 混入率 = 噂由来件数 / 全 transmission 件数
            "contamination_rate": (round(n_rumor / n_all, 6) if n_all else 0.0),
        },
        "metrics": metrics,
        "per_agent_contaminated": per_agent,
        "checks": {
            # 見立ての自己検査(module docstring)。False なら噂が想定外の経路で
            # Y_external に入っている = 要調査。
            "delta_flows_through_c_transmission": (
                metrics["y_external_total"]["delta"]
                == metrics["c_transmission_total"]["delta"]),
            # 噂 OFF ラン(または噂 0 件)では**必ず**オーバーレイ = 凍結値。
            "overlay_equals_frozen_when_no_rumor": (
                n_rumor > 0
                or all(v["delta"] == 0 for v in metrics.values())),
            # 除去件数と n_transmission の Δ は定義上一致する
            "removed_equals_n_transmission_delta": (
                metrics["n_transmission"]["delta"] == float(n_rumor)),
        },
    }


# --------------------------------------------------------------------------- #
# レポート
# --------------------------------------------------------------------------- #
def _fmt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def render(res: dict) -> str:
    ev, mt = res["events"], res["metrics"]
    L = ["# 噂の混線 切り分けオーバーレイ(IF-C 残課題)", "",
         "**凍結値が正典**。オーバーレイは「凍結指標の定義そのままで、入力から噂の "
         "transmission だけを抜いた場合」の診断値であり、指標の再定義ではない。", ""]
    if not res["rumors_detected"]:
        L += ["> **噂 OFF のラン**(`rumor_born` / `rumor_stifle` / `rumor-` 伝播が 1 件も無い)。",
              "> 混入ゼロ = オーバーレイは全項目で凍結値と一致する。", ""]
    L += [f"- 全 transmission: **{ev['transmission_total']}** 件",
          f"- うち噂由来: **{ev['transmission_rumor']}** 件 / 語彙由来: "
          f"{ev['transmission_vocab']} 件",
          f"- **混入率: {ev['contamination_rate']}**(件数ベース)",
          f"- 噂の誕生 {ev['rumor_born']} 件 / stifler 化 {ev['rumor_stifle']} 件"
          "(凍結指標はこの 2 種を読まない=混線経路ではない)", "",
          "## 凍結値 vs 噂除外オーバーレイ", "",
          "| 指標 | 凍結値(正典) | オーバーレイ | 差分 | 混入率(指標ベース) |",
          "|---|---|---|---|---|"]
    for key in ("c_transmission_total", "c_label_adopt_total",
                "c_new_relations_total", "c_sns_reach_total", "y_external_total",
                "n_transmission", "n_transmission_novel",
                "transmission_novel_rate"):
        d = mt[key]
        L.append(f"| {key} | {_fmt(d['frozen'])} | {_fmt(d['overlay'])} | "
                 f"{_fmt(d['delta'])} | {_fmt(d['contaminated_share'])} |")
    L += ["", "## 自己検査", ""]
    for k, v in res["checks"].items():
        L.append(f"- `{k}`: **{'OK' if v else 'NG'}**")
    if res.get("l2_final"):
        L += ["", "## 参考: L2 最終行(再計算不可・並記のみ)", ""]
        for k, v in sorted(res["l2_final"].items()):
            L.append(f"- `{k}` = {v}")
        L.append("")
        L.append("> L2 は走行時の窓付き値で、上表の事後 `echo_novelty`(ラン全体窓)とは")
        L.append("> 分母が違う。上表の混入率を一次補正として当てるための参考値。")
    pa = res["per_agent_contaminated"]
    if pa:
        L += ["", f"## 汚染された個体({len(pa)} 人・c_transmission の差が大きい順)", "",
              "| agent | c_transmission 凍結 | 同 オーバーレイ | 差 | "
              "Y_external 凍結 | 同 オーバーレイ |", "|---|---|---|---|---|---|"]
        for r in pa[:20]:
            L.append(f"| {r['id']} | {r['c_transmission_frozen']} | "
                     f"{r['c_transmission_overlay']} | {r['delta']} | "
                     f"{_fmt(r['Y_external_frozen'])} | "
                     f"{_fmt(r['Y_external_overlay'])} |")
        if len(pa) > 20:
            L.append(f"| … | 他 {len(pa) - 20} 人 | | | | |")
    L += ["", "> 限界: 除去は transmission イベント単位であり、噂が会話の枠を消費して",
          "> 語彙伝播を押し出した機会費用は復元できない(= 「噂が無かった世界」ではない)。"]
    return "\n".join(L) + "\n"


def _want_kinds() -> frozenset:
    """本解析が読む kind = 凍結 2 関数の入力 + 噂の 3 種。

    W4-D: `compare()` は自分では kind を数えるだけで、実質の計算は
    `measure.agent_features` と `measure.echo_novelty`(どちらも凍結)がする。
    したがって絞ってよい集合は **凍結関数が読む kind の和** + 自前で数える
    transmission / rumor_born / rumor_stifle。凍結側の集合は写経せず
    `l1_stream` の表を参照する(measure.py の AST と機械照合されている)。
    """
    import l1_stream as ls
    return (ls.AGENT_FEATURES_KINDS | ls.ECHO_NOVELTY_KINDS
            | {"transmission", "rumor_born", "rumor_stifle"})


def analyze(run_dir: str) -> dict:
    """ラン 1 本を解析して結果 dict を返す(読み取り専用)。"""
    import l1_stream as ls
    path = os.path.join(run_dir, "l1_events.parquet")
    if not os.path.isfile(path):
        raise SystemExit(f"[rumor-contamination] l1_events.parquet が無い: {run_dir}")
    # W4-D: 全件 dict 展開 → kind 絞り読み。名簿の合成だけは **全 kind** に現れる
    # agent_id が要る(噂でしか出てこない個体を落とさないための仕掛けなので、
    # 絞った list から作ると仕掛けそのものが壊れる)ので agent_id 列を別に走査する。
    events = list(ls.iter_events(run_dir, kinds=_want_kinds()))
    agents_meta = m.load_agents(run_dir)
    if not agents_meta:
        agents_meta = [{"id": aid} for aid in sorted(ls.distinct_agent_ids(run_dir))]
    res = compare(events, agents_meta)
    res["run_dir"] = os.path.abspath(run_dir)
    l2 = _l2_final(run_dir)
    if l2:
        res["l2_final"] = l2
    return res


def main() -> int:
    ap = argparse.ArgumentParser(
        description="噂の混線の切り分けオーバーレイ(読み取り専用・凍結指標を再定義しない)")
    ap.add_argument("run_dir", help="ラン出力ディレクトリ(l1_events.parquet を含む)")
    ap.add_argument("--out", default=None, help="出力先(既定: <run_dir>/analysis)")
    args = ap.parse_args()

    res = analyze(args.run_dir)
    out_dir = args.out or os.path.join(args.run_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    jpath = os.path.join(out_dir, "rumor_contamination.json")
    mpath = os.path.join(out_dir, "rumor_contamination_report.md")
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
