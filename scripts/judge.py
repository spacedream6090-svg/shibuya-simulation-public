#!/usr/bin/env python
"""LLM-judge ハーネス(M2 の *補助* 指標)。

    python scripts/judge.py runs/<name>                       # 既定 = mock(決定論)
    python scripts/judge.py runs/<name> --backend ollama --model qwen3:4b --judges 3

L1(runs/<name>/l1_events.parquet)からエージェントごとの「1日の行動ドシエ」
(発話 / SNS投稿 / DM / 「世界を変える」ツール系イベント / 造語 の時系列テキスト、
文字数上限 ~2000 で切詰)を構築し、別ファミリ推奨の LLM に

    「この人物の1日の行動記録から、彼/彼女は『世界を変えようとする人』か?」

を JSON {"world_changer": bool, "confidence": 0-1, "reason": "..."} で答えさせる。
judge ごとに温度を変えて --judges 回反復し、判定の一致率 = **Fleiss κ**(numpy 自作)
を計算する。出力は runs/<name>/analysis/judge.json + judge_report.md。

====================================================================
R4 遵守(measurement circularity への防壁)— この設計の憲法
--------------------------------------------------------------------
* **主指標は客観カウント**(observer/measure.py の Y_external = 伝播・採用・新規関係・
  SNS 到達、および POI/資源プール/署名などツールの効果カウント)。judge はあくまで
  **補助指標**であり、主指標を置き換えない。
* judge の採用条件は **人手 or 判定間 κ ≥ 0.7**。κ が閾値未満の判定は「参考値」に留め、
  R²(k) など本線の分析には使わない(この閾値は judge.json の kappa_adopted で明示)。
* judge は **シミュ本体へ一切逆流しない**。本ファイルは scripts/ 配下・L1 の *読み取り専用
  下流* であり、src/(エンジン)へは書き込まない(measure は読み取りのみ import)。
  よって judge の出力がエージェント状態・k・行動に影響することは構造的に不可能。
* **判定モデルは運用モデルと別ファミリを推奨**(trait ロールプレイの共鳴=自己一致
  バイアスを避ける)。既定 --model qwen3:4b は運用想定 Qwen 系と同ファミリなので、
  レポートに R4 警告を出す(別ファミリを使えるなら --model で差し替える)。
* ドシエは **外向きの行動ログのみ**で構築し、traits / persona / 内省 belief は入れない
  (R9 の trait→judge 汚染を配線レベルで遮断)。
====================================================================
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# Windows コンソール(cp932)で κ や日本語を print しても落ちないように
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from society.observer import measure as m  # noqa: E402  (読み取り専用の下流利用)

# ドシエに載せる「外向きの行動」イベント種類 → 表示ラベル(mock はこの接頭辞で信号を読む)。
# 内省(reflect)・traits・persona は R4/R9 のため意図的に除外。
_ACTION_LABELS: dict[str, str] = {
    "speak": "発話",
    "sns_post": "SNS投稿",
    "dm": "DM",
    "vocab_coin": "造語",
    "label_coin": "造語",
    "coin_label": "造語",
    "event_host": "イベント開催",
    "event_attend": "イベント参加",
    "flyer_post": "ビラ掲示",
    "group_found": "グループ結成",
    "group_join": "グループ加入",
    "proposal": "提案",
    "proposal_support": "提案賛同",
    "venture_open": "出店",
    "venture_sale": "売上",
}

# 「世界を変える」度合いの弱い信号(mock の決定論スコア + レポートの直感の材料)。
# ※ これは judge の内部ヒューリスティックであって主指標ではない(主指標=measure の客観量)。
_MARKER_WEIGHT: dict[str, float] = {
    "提案": 3.0, "提案賛同": 1.0, "グループ結成": 3.0, "グループ加入": 1.0,
    "出店": 3.0, "売上": 0.5, "イベント開催": 2.0, "イベント参加": 0.5,
    "ビラ掲示": 2.0, "造語": 2.0, "SNS投稿": 1.0, "DM": 0.5, "発話": 0.3,
}


# --------------------------------------------------------------------------- #
# ドシエ構築
# --------------------------------------------------------------------------- #
def _event_text(kind: str, p: dict) -> str | None:
    """外向き行動イベント 1 件を短い日本語行に。対象外 kind は None。"""
    label = _ACTION_LABELS.get(kind)
    if label is None:
        return None
    if kind in ("speak", "sns_post", "flyer_post", "proposal"):
        body = str(p.get("text") or "").strip()
    elif kind == "dm":
        body = f"→{p.get('to')}: {str(p.get('text') or '').strip()}"
    elif kind in ("vocab_coin", "label_coin", "coin_label"):
        body = str(p.get("text") or p.get("word") or "").strip()
    elif kind == "event_host":
        body = str(p.get("title") or "").strip()
    elif kind == "event_attend":
        body = str(p.get("title") or f"#{p.get('event_id')}").strip()
    elif kind == "group_found":
        body = f"{p.get('name') or ''}({p.get('purpose') or ''})".strip()
    elif kind == "group_join":
        body = str(p.get("name") or f"#{p.get('group_id')}").strip()
    elif kind == "proposal_support":
        body = f"#{p.get('proposal_id')}"
    elif kind == "venture_open":
        body = f"{p.get('name') or ''}({p.get('offer') or ''})".strip()
    elif kind == "venture_sale":
        body = f"{p.get('amount')}"
    else:  # pragma: no cover - 全 kind は上で網羅
        body = ""
    return f"{label}: {body}" if body else label


def build_dossiers(events: list[dict], agents: list[dict] | None,
                   max_chars: int = 2000) -> dict[int, str]:
    """agent_id -> 時系列ドシエ文字列。行動ゼロや空ランでも安全に空を返す。

    行は step 昇順。max_chars を超えたら**末尾**を切詰(1日の後半=高コスト行動が出やすい
    ので、先頭を優先して残す)。
    """
    by_agent: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for e in sorted(events, key=lambda e: e["step"]):
        aid = e.get("agent_id")
        if not isinstance(aid, int) or aid < 0:
            continue
        line = _event_text(e["kind"], e.get("payload") or {})
        if line is not None:
            by_agent[aid].append((e["step"], line))

    # 判定対象 agent の集合(agents.json があればそれ、無ければ登場 id)
    if agents:
        ids = sorted(int(a["id"]) for a in agents if "id" in a)
    else:
        ids = sorted(by_agent)

    out: dict[int, str] = {}
    for aid in ids:
        lines = [f"[step {s}] {t}" for s, t in by_agent.get(aid, [])]
        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[:max_chars].rsplit("\n", 1)[0] + "\n…(以下略)"
        out[aid] = text
    return out


# --------------------------------------------------------------------------- #
# judge バックエンド
# --------------------------------------------------------------------------- #
_PROMPT_TMPL = (
    "以下はある人物の「1日の行動記録」です(発話・SNS投稿・DM・イベント/出店/署名など)。\n"
    "この記録だけから、この人物が『世界を変えようとする人』か否かを判定してください。\n"
    "『世界を変えようとする人』= 街や周囲の状況を能動的に動かそうとする人(新しい言葉・場・\n"
    "集まり・仕組みを作る、人を巻き込む等)。単なる日常の移動や雑談だけなら該当しません。\n"
    "心理学用語や人物の内面の推測ではなく、記録された行動そのものから判断してください。\n\n"
    "=== 行動記録 ===\n{dossier}\n=== ここまで ===\n\n"
    '出力は必ず次の JSON のみ: {{"world_changer": true/false, '
    '"confidence": 0.0〜1.0, "reason": "根拠を一文で"}}')


class MockJudge:
    """決定論 judge(GPU/ネット不要・テスト用)。src は一切触らない独自実装。

    判定は (agent_id, judge_idx) のハッシュで**安定**に決まる(呼び出し順非依存)。
    ドシエ中の行動マーカーの弱い信号 signal∈[0,1] と、judge ごとに変わるハッシュ雑音を
    混ぜることで、境界例では judge 間の不一致が自然に生じ κ が退化しない(=κ テストが機能)。
    """

    name = "mock"

    def judge(self, agent_id: int, dossier: str, *, temperature: float,
              judge_idx: int) -> dict:
        signal = 0.0
        for line in dossier.splitlines():
            body = line.split("] ", 1)[-1]
            label = body.split(":", 1)[0].strip()
            signal += _MARKER_WEIGHT.get(label, 0.0)
        signal = min(signal / 12.0, 1.0)                 # 正規化 [0,1]
        h = int(hashlib.sha256(f"{agent_id}|{judge_idx}".encode()).hexdigest()[:8],
                16) / float(1 << 32)                     # [0,1) 安定雑音
        score = 0.6 * signal + 0.4 * h
        wc = score >= 0.5
        conf = round(min(1.0, 0.5 + abs(score - 0.5)), 3)
        reason = (f"行動記録に世界改変の兆候が{'多い' if wc else '乏しい'}"
                  f"(mock決定論・signal={signal:.2f})。")
        return {"world_changer": bool(wc), "confidence": conf, "reason": reason}


class OllamaJudge:
    """Ollama(ローカル LLM)judge。judge ごとに temperature を変えて反復する。"""

    def __init__(self, model: str, host: str = "http://localhost:11434",
                 timeout: int = 120):
        self.name = model
        self.model = model
        self.host = host
        self.timeout = timeout

    def judge(self, agent_id: int, dossier: str, *, temperature: float,
              judge_idx: int) -> dict:
        prompt = _PROMPT_TMPL.format(dossier=dossier or "(記録なし)")
        body = json.dumps({
            "model": self.model, "prompt": prompt, "stream": False,
            "format": "json", "think": False,
            "options": {"temperature": float(temperature), "num_predict": 300},
        }).encode("utf-8")
        req = urllib.request.Request(f"{self.host}/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            obj = json.loads(data.get("response", "{}"))
        except Exception as exc:                          # ネット/JSON 失敗は棄権
            return {"world_changer": False, "confidence": 0.0,
                    "reason": f"(判定失敗: {type(exc).__name__})", "_failed": True}
        wc = bool(obj.get("world_changer", False))
        try:
            conf = float(obj.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        conf = max(0.0, min(1.0, conf))
        return {"world_changer": wc, "confidence": conf,
                "reason": str(obj.get("reason", ""))[:200]}


def make_backend(name: str, model: str):
    if name == "mock":
        return MockJudge()
    if name == "ollama":
        return OllamaJudge(model)
    raise SystemExit(f"未対応 backend: {name}(mock | ollama)")


# --------------------------------------------------------------------------- #
# Fleiss κ(numpy 自作)
# --------------------------------------------------------------------------- #
def fleiss_kappa(counts: np.ndarray) -> float:
    """Fleiss κ。counts = (N item, K category) の評点数行列(各行の和 = 評者数 n で一定)。

    P_i = (Σ_j n_ij^2 − n) / (n(n−1)); P̄ = mean_i P_i;
    p_j = Σ_i n_ij / (N n); P_e = Σ_j p_j^2; κ = (P̄ − P_e)/(1 − P_e)。
    退化(評者<2 / 期待一致=1 → 0除算)は nan。返り値は常に [-1,1] か nan。
    """
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 2 or counts.shape[0] == 0:
        return float("nan")
    row_sums = counts.sum(axis=1)
    n = float(row_sums[0])
    if n < 2 or not np.allclose(row_sums, n):
        return float("nan")
    N = counts.shape[0]
    p_i = (np.sum(counts ** 2, axis=1) - n) / (n * (n - 1))
    p_bar = float(np.mean(p_i))
    p_j = counts.sum(axis=0) / (N * n)
    p_e = float(np.sum(p_j ** 2))
    denom = 1.0 - p_e
    if abs(denom) < 1e-12:
        return float("nan")
    return float((p_bar - p_e) / denom)


# --------------------------------------------------------------------------- #
# 校正(Y_external 客観量との一致・相関)
# --------------------------------------------------------------------------- #
def _pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _rank(a: np.ndarray) -> np.ndarray:
    """平均順位(タイ補正)。"""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return _pearson(_rank(x), _rank(y))


# --------------------------------------------------------------------------- #
# 実行
# --------------------------------------------------------------------------- #
def judge_run(run_dir: str, backend: str = "mock", model: str = "qwen3:4b",
              judges: int = 3, max_chars: int = 2000) -> dict:
    """1 run に judge を適用し analysis/judge.json + judge_report.md を書く。返り値=judge.json。"""
    events = m.load_events(run_dir)
    agents = m.load_agents(run_dir)
    dossiers = build_dossiers(events, agents, max_chars=max_chars)

    # 主指標(客観量)= 校正の基準。traits は渡さない(R9)。
    feats = m.agent_features(events, agents, None)
    y_ext = {f["id"]: f.get("Y_external", 0) for f in feats}
    name_by = {f["id"]: f.get("name") for f in feats}
    occ_by = {f["id"]: f.get("occupation") for f in feats}

    be = make_backend(backend, model)
    judges = max(1, int(judges))
    temps = (np.linspace(0.2, 0.9, judges).round(3).tolist() if judges > 1
             else [0.5])

    ids = sorted(dossiers)
    K = 2                                    # 0 = not, 1 = world_changer
    counts = np.zeros((len(ids), K), dtype=float)
    per_agent: list[dict] = []
    n_failed = 0
    for row, aid in enumerate(ids):
        votes, confs, reasons = [], [], []
        for j in range(judges):
            r = be.judge(aid, dossiers[aid], temperature=temps[j], judge_idx=j)
            if r.get("_failed"):
                n_failed += 1
            wc = bool(r["world_changer"])
            votes.append(wc)
            confs.append(float(r["confidence"]))
            reasons.append(r["reason"])
            counts[row, 1 if wc else 0] += 1
        n_yes = int(sum(votes))
        per_agent.append({
            "id": aid, "name": name_by.get(aid), "occupation": occ_by.get(aid),
            "votes": votes, "n_yes": n_yes, "n_judges": judges,
            "majority": n_yes * 2 > judges,
            "mean_confidence": round(float(np.mean(confs)), 3),
            "Y_external": y_ext.get(aid),
            "reasons": reasons,
        })

    kappa = fleiss_kappa(counts) if ids else float("nan")
    kappa_adopted = bool(kappa == kappa and kappa >= 0.7)   # nan は False

    # 校正: judge の world_changer 率 vs 客観 Y_external
    rate = [a["n_yes"] / a["n_judges"] for a in per_agent]
    yv = [a["Y_external"] if a["Y_external"] is not None else 0 for a in per_agent]
    calib = {
        "n": len(per_agent),
        "y_external_pearson": _pearson(rate, yv),
        "y_external_spearman": _spearman(rate, yv),
        "top_y_external": [
            {"id": a["id"], "name": a["name"], "Y_external": a["Y_external"],
             "judge_rate": round(a["n_yes"] / a["n_judges"], 3),
             "majority": a["majority"]}
            for a in sorted(per_agent, key=lambda a: -(a["Y_external"] or 0))[:8]
        ],
    }

    result = {
        "run_dir": os.path.abspath(run_dir),
        "backend": backend, "model": (model if backend != "mock" else "mock"),
        "n_judges": judges, "temperatures": temps, "max_chars": max_chars,
        "n_agents": len(per_agent),
        "n_world_changer_majority": sum(1 for a in per_agent if a["majority"]),
        "fleiss_kappa": (None if kappa != kappa else round(kappa, 4)),
        "kappa_adopt_threshold": 0.7,
        "kappa_adopted": kappa_adopted,
        "n_judge_failures": n_failed,
        "calibration": calib,
        "agents": per_agent,
        "r4_note": ("主指標は observer/measure の客観カウント。judge は補助・"
                    "κ≥0.7 で採用・シミュ本体へ逆流しない(read-only 下流)。"),
    }

    outdir = os.path.join(run_dir, "analysis")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "judge.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=1)
    _write_report(os.path.join(outdir, "judge_report.md"), result)
    return result


def _fmt(v, nd=3):
    if v is None:
        return "n/a"
    if isinstance(v, float) and v != v:
        return "nan"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def _write_report(path: str, r: dict) -> None:
    kappa = r["fleiss_kappa"]
    lines: list[str] = []
    lines.append(f"# LLM-judge report: {os.path.basename(os.path.dirname(os.path.dirname(path)))}\n")
    lines.append("> **R4(循環の防壁)**: 主指標は observer/measure の**客観カウント**"
                 "(Y_external = 伝播・採用・新規関係・SNS 到達 + ツール効果)。"
                 "本 judge は**補助指標**であり、**κ ≥ 0.7 のときだけ**採用に足る。"
                 "judge は L1 の読み取り専用下流で、**シミュ本体へ一切逆流しない**。"
                 "ドシエは外向き行動ログのみで traits/persona/内省を含まない(R9)。\n")
    lines.append("> **適用範囲(サーベイ §4.4)**: LLM 判定は**目標達成の次元**では人間評価とよく一致するが、"
                 "**他の次元では大きな乖離が残る**と報告されている。本 judge はその**一致度が比較的高い側**の"
                 "次元にあたる **world_changer の単一次元にのみ**用いる。κ ≥ 0.7 の採用条件・別ファミリの"
                 "判定者推奨・シム本体への逆流不能という 3 条件は、分野の慣行より厳しい。\n")
    lines.append("## Setup")
    lines.append(f"- backend: `{r['backend']}`  model: `{r['model']}`")
    lines.append(f"- judges: {r['n_judges']}  temperatures: {r['temperatures']}")
    lines.append(f"- dossier max_chars: {r['max_chars']}  agents judged: {r['n_agents']}")
    if r["n_judge_failures"]:
        lines.append(f"- ⚠ judge 失敗(棄権)回数: {r['n_judge_failures']}")
    if r["backend"] == "ollama" and str(r["model"]).lower().startswith("qwen"):
        lines.append("- ⚠ **R4 警告**: 判定モデルが運用想定(Qwen 系)と同ファミリの可能性。"
                     "trait ロールプレイの自己一致バイアスを避けるため、"
                     "別ファミリ(例: llama系/gemma系)の judge を併走させて κ を確認するのが望ましい。")
    lines.append("")
    lines.append("## Agreement (Fleiss κ)")
    lines.append(f"- **Fleiss κ = {_fmt(kappa, 4)}**  (採用閾値 κ ≥ {r['kappa_adopt_threshold']})")
    verdict = ("✅ 採用可(補助指標として利用可)" if r["kappa_adopted"]
               else "❌ 閾値未満 → 参考値に留め、本線分析(R²(k) 等)には使わない")
    lines.append(f"- 判定: {verdict}")
    lines.append(f"- majority で world_changer と判定された人数: "
                 f"{r['n_world_changer_majority']} / {r['n_agents']}\n")
    lines.append("## Calibration vs objective Y_external")
    c = r["calibration"]
    lines.append(f"- Pearson(judge率, Y_external) = {_fmt(c['y_external_pearson'])}"
                 f"  Spearman = {_fmt(c['y_external_spearman'])}  (n={c['n']})")
    lines.append("- 上位 Y_external 者と judge の一致(客観の上位者を judge が拾えているか):")
    lines.append("")
    lines.append("| id | name | Y_external | judge率 | majority |")
    lines.append("|---:|---|---:|---:|:---:|")
    for t in c["top_y_external"]:
        lines.append(f"| {t['id']} | {t['name']} | {t['Y_external']} | "
                     f"{t['judge_rate']} | {'○' if t['majority'] else '×'} |")
    lines.append("")
    lines.append("## Sample reasons (majority=world_changer の上位確信例)")
    flagged = [a for a in r["agents"] if a["majority"]]
    flagged.sort(key=lambda a: -a["mean_confidence"])
    if flagged:
        for a in flagged[:8]:
            # majority(=world_changer)と整合する評者の根拠を選ぶ(無ければ先頭)
            reason = next((rs for v, rs in zip(a["votes"], a["reasons"]) if v),
                          a["reasons"][0])
            lines.append(f"- agent {a['id']} ({a['name']}, {a['occupation']}) "
                         f"conf={a['mean_confidence']} Y_ext={a['Y_external']}: "
                         f"\"{reason}\"")
    else:
        lines.append("- (majority で world_changer と判定された者はいない)")
    lines.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM-judge harness (M2 補助指標)")
    ap.add_argument("run_dir", help="e.g. runs/conv_check")
    ap.add_argument("--backend", default="mock", choices=["mock", "ollama"])
    ap.add_argument("--model", default="qwen3:4b")
    ap.add_argument("--judges", type=int, default=3)
    ap.add_argument("--max-chars", type=int, default=2000)
    args = ap.parse_args()
    if not os.path.isdir(args.run_dir):
        raise SystemExit(f"not a directory: {args.run_dir}")
    r = judge_run(args.run_dir, backend=args.backend, model=args.model,
                  judges=args.judges, max_chars=args.max_chars)
    k = r["fleiss_kappa"]
    print(f"[judge] {args.run_dir}: {r['n_agents']} agents, backend={r['backend']}, "
          f"Fleiss κ={k}, adopted={r['kappa_adopted']}, "
          f"world_changer(majority)={r['n_world_changer_majority']} "
          f"-> {os.path.join(args.run_dir, 'analysis', 'judge.json')}")


if __name__ == "__main__":
    main()
