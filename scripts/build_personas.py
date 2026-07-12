"""ペルソナ v2 のオフライン生成(D6: IPF骨格 × 尺度分布 × LLM文章化)。

使い方:
    python scripts/build_personas.py 80                 # → data/personas_80.json
    python scripts/build_personas.py 80 --no-llm        # 文章化を手続き生成で代替
    python scripts/build_personas.py 80 --model qwen3:8b --seed 42

3段パイプライン(decision-agenda D6):
  1. IPF: data/shibuya_population.json の周辺分布(年齢×性別×職業、来街者比率)
     から結合分布を推定し骨格をサンプル。
  2. 尺度サンプリング: traits(tail 明示確保)+ 欲求発火の個体差(factors 写像)。
  3. LLM 文章化: Verbalized Sampling(5案+確率)→ 既出と最も異なる案を採用
     (mode collapse 対策)。構成概念語バリデータ(注入/評価分離)で検査。
シミュ実行時は生成済み JSON を読むだけ(本体の LLM 負荷ゼロ・決定論)。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from society.agents.persona import _WORK_CAT  # noqa: E402
from society.agents.validate import construct_violations  # noqa: E402
from society.factors.registry import drive_params, sample_traits  # noqa: E402

_GIVEN_F = ["美咲", "陽菜", "結衣", "葵", "凛", "さくら", "芽衣", "千夏", "真央", "花音",
            "美月", "詩織", "彩乃", "智子", "由美", "恵", "直美", "幸子", "京子", "和子"]
_GIVEN_M = ["翔太", "蓮", "大輝", "颯太", "拓海", "悠人", "健太", "亮", "光", "駿",
            "隆", "誠", "浩二", "健一", "修", "茂", "勝", "昇", "清", "実"]
_FAMILY = ["佐藤", "鈴木", "高橋", "田中", "伊藤", "渡辺", "山本", "中村", "小林", "加藤",
           "吉田", "山田", "佐々木", "山口", "松本", "井上", "木村", "林", "斎藤", "清水"]

# ---- 公務員(行政・税・公務員バッチ 2026-07-06。docs/research/shibuya-government.md §5,6)----
# 職業と重み。区職員(一般行政職)> 警察官 > 消防士(公安職)。国勢調査「公務」≈3.5% を基準に、
# 警察・消防を可視化するため小幅上乗せ(--civil-share 既定 0.06 で ~6/100)。
# ★勤務地: 地図に交番・消防署の POI カテゴリが無く、勤務地写像(society.agents.persona._pick_workplace)
#   は別バッチ所有で編集不可のため、公務員は空間上の勤務地を持たない(通常の勤務完遂経路に乗らない)。
#   給与は society.government の日次ペイロールが予算(区職員=ward / 警察官・消防士=metro)から直接支給し
#   源泉徴収する。=概念・コード上の存在として税・予算に接続する(勤務地は正直に妥協)。
_CIVIL_OCCS = ["区職員", "警察官", "消防士"]
_CIVIL_WEIGHTS = [0.55, 0.27, 0.18]


def inject_civil_servants(skeletons: list[dict], share: float,
                          rng: np.random.Generator) -> int:
    """手続き生成の骨格に公務員を注入(既存の visitor/commute フラグは保持=通勤ロジック踏襲)。

    候補は「区に生活基盤がある層」= 居住者 or 流入通勤者(余暇来街者は除く。渋谷で働く前提)。
    決定論(rng 経由)。share<=0 なら何もしない=既存の personas_*.json 生成と完全に同一挙動。"""
    if share <= 0:
        return 0
    n = len(skeletons)
    n_civil = int(round(n * float(share)))
    if n_civil <= 0:
        return 0
    cand = [i for i, sk in enumerate(skeletons)
            if (not sk["visitor"]) or sk.get("commute")]
    if not cand:
        return 0
    n_civil = min(n_civil, len(cand))
    pick = sorted(int(i) for i in rng.choice(cand, size=n_civil, replace=False))
    occs = rng.choice(_CIVIL_OCCS, size=n_civil, p=_CIVIL_WEIGHTS)
    for i, occ in zip(pick, occs):
        skeletons[i]["occupation"] = str(occ)
    return n_civil


def ipf_joint(pop: dict, iters: int = 60) -> np.ndarray:
    """年齢帯×性別×職業の結合分布を IPF で推定(周辺分布+弱い依存シード)。"""
    ages = pop["age_bands"]
    occs = pop["occupations"]
    genders = list(pop["gender"].keys())
    m_age = np.array([a["share"] for a in ages], dtype=float)
    m_gen = np.array([pop["gender"][g] for g in genders], dtype=float)
    m_occ = np.array([o["share"] for o in occs], dtype=float)
    m_age /= m_age.sum()
    m_gen /= m_gen.sum()
    m_occ /= m_occ.sum()

    joint = np.ones((len(ages), len(genders), len(occs)))
    hints = pop.get("age_x_occupation_hints", {})
    band_keys = [f"{a['band'][0]}-{a['band'][1]}" for a in ages]
    for oi, occ in enumerate(occs):
        hint = hints.get(occ["name"])
        if not isinstance(hint, dict):
            continue
        for ai, key in enumerate(band_keys):
            joint[ai, :, oi] *= float(hint.get(key, 1.0))

    for _ in range(iters):
        joint *= (m_age / joint.sum(axis=(1, 2)))[:, None, None]
        joint *= (m_gen / joint.sum(axis=(0, 2)))[None, :, None]
        joint *= (m_occ / joint.sum(axis=(0, 1)))[None, None, :]
    return joint / joint.sum()


def _sample_lead(is_student: bool, arr: dict, rng: np.random.Generator) -> int:
    """職場 start から逆算する到着リード(分)。二峰の一方(通学=早め/会社員=遅め)を担う。"""
    if is_student:
        mean, sd = arr.get("student_lead_mean", 30), arr.get("student_lead_sd", 15)
    else:
        mean, sd = arr.get("worker_lead_mean", 40), arr.get("worker_lead_sd", 20)
    lo, hi = arr.get("lead_min", 10), arr.get("lead_max", 120)
    return int(max(lo, min(hi, int(round(rng.normal(mean, sd))))))


def _sample_depart(dep: dict, rng: np.random.Generator) -> int:
    """夕方の退勤=帰宅トリガ時刻(分 of day、10分刻み)。夕ピーク + 夜遊びの遅い裾。"""
    if rng.random() < float(dep.get("evening_share", 0.8)):
        v = rng.normal(dep.get("evening_mean", 1110), dep.get("evening_sd", 45))
    else:
        v = rng.uniform(dep.get("late_lo", 1320), dep.get("late_hi", 1440))
    v = int(round(v / 10.0) * 10)
    return int(max(16 * 60, min(23 * 60 + 50, v)))          # 16:00〜23:50


def sample_skeletons(n: int, pop: dict, rng: np.random.Generator) -> list[dict]:
    ages = pop["age_bands"]
    occs = pop["occupations"]
    genders = list(pop["gender"].keys())
    joint = ipf_joint(pop)
    flat = joint.ravel()
    idx = rng.choice(len(flat), size=n, p=flat / flat.sum())
    # 流入通勤者(commuter)= visitor の下位型(docs/research/shibuya-inflow.md)
    inflow = pop.get("inflow", {})
    commuter_share = float(inflow.get("commuter_share", 0.0))
    arr = inflow.get("arrival", {})
    dep = inflow.get("departure", {})
    station_share = float(inflow.get("gateway_station_share", 0.87))
    student_occs = set(inflow.get("student_occupations", ["大学生"]))
    lines = list(inflow.get("residence_lines", []))
    out = []
    for k in idx:
        ai, gi, oi = np.unravel_index(int(k), joint.shape)
        lo, hi = ages[ai]["band"]
        age = int(rng.integers(lo, hi + 1))
        occ = occs[oi]["name"]
        visitor = bool(rng.random() < float(pop.get("visitor_share", 0.5)))
        sk = {"age": age, "gender": genders[gi], "occupation": occ,
              "visitor": visitor}
        # commuter: 職場のある職業の visitor から commuter_share ぶんを流入通勤者に。
        #  commuter_share==0(inflow 無し)なら rng を引かない=旧挙動を保つ。
        has_work = _WORK_CAT.get(occ) is not None
        if visitor and has_work and commuter_share > 0 \
                and rng.random() < commuter_share:
            sk["commute"] = True
            sk["arrival_lead_min"] = _sample_lead(occ in student_occs, arr, rng)
            sk["depart_min"] = _sample_depart(dep, rng)
            sk["commute_gateway"] = ("station" if rng.random() < station_share
                                     else "edge")
            sk["residence_line"] = (lines[int(rng.integers(len(lines)))]
                                    if lines else "")
        out.append(sk)
    return out


def _bigrams(text: str) -> set[str]:
    return {text[i:i + 2] for i in range(len(text) - 1)}


def _distance(a: str, b: str) -> float:
    ba, bb = _bigrams(a), _bigrams(b)
    if not ba or not bb:
        return 1.0
    return 1.0 - len(ba & bb) / len(ba | bb)


def _fallback_text(sk: dict, name: str) -> str:
    if sk.get("commute"):
        line = sk.get("residence_line") or "渋谷の外"
        where = f"{line}に住んでいて、毎日渋谷に通勤・通学している"
    elif sk["visitor"]:
        where = "渋谷によく遊びに来る"
    else:
        where = "渋谷の街に住んでいる"
    return (f"あなたは{name}、{sk['age']}歳の{sk['occupation']}({sk['gender']}性)。"
            f"{where}。自分の言葉で自然に、短く話す。")


def llm_persona(sk: dict, name: str, chosen: list[str], model: str,
                rng: np.random.Generator) -> str:
    """Verbalized Sampling: 5案+確率を出させ、既出と最も異なる有効案を採用。"""
    import urllib.request

    living = "渋谷の街の外に住んでいて、よく渋谷に来る" if sk["visitor"] \
        else "渋谷の街なかに住んでいる"
    prompt = (
        "次の人物の「自己紹介と人柄」を5案、互いに口調・趣味・性格が異なるように書く。\n"
        f"人物: {name}、{sk['age']}歳、{sk['occupation']}、{sk['gender']}性。{living}。\n"
        "各案は50〜80字の日本語。話し方の癖・好きなもの・価値観のどれかを入れる。\n"
        "心理学用語(効力感・当事者意識など)は使わない。\n"
        '出力JSON: {"candidates": [{"text": "...", "p": 0.2}, ...5案]}')
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "format": "json", "think": False,
                       "options": {"temperature": 0.9, "num_predict": 700}}
                      ).encode("utf-8")
    req = urllib.request.Request("http://localhost:11434/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        cands = json.loads(data.get("response", "{}")).get("candidates", [])
    except Exception:
        return _fallback_text(sk, name)
    valid = [c["text"].strip() for c in cands
             if isinstance(c, dict) and isinstance(c.get("text"), str)
             and 20 <= len(c["text"].strip()) <= 140
             and not construct_violations(c["text"])]
    if not valid:
        return _fallback_text(sk, name)
    if not chosen:
        pick = valid[int(rng.integers(len(valid)))]
    else:                                   # 既に選んだ文と最も異なる案(多様性)
        pick = max(valid, key=lambda t: min(_distance(t, c) for c in chosen[-30:]))
    intro = f"あなたは{name}、{sk['age']}歳の{sk['occupation']}({sk['gender']}性)。"
    if not pick.startswith("あなたは"):
        pick = intro + pick
    return pick


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--civil-share", type=float, default=0.0,
                    help="公務員(区職員/警察官/消防士)を骨格に注入する比率。0=注入なし"
                         "(既定・既存名簿と同一挙動)。行政・税バッチの civic 名簿は 0.06 で生成")
    ap.add_argument("--out", default=None,
                    help="出力先 JSON(既定=data/personas_{n}.json)。流入名簿は "
                         "data/personas_100_inflow.json 等")
    args = ap.parse_args()

    pop = json.loads((REPO_ROOT / "data" / "shibuya_population.json")
                     .read_text(encoding="utf-8"))
    rng = np.random.default_rng(args.seed)
    skeletons = sample_skeletons(args.n, pop, rng)
    # 公務員の注入(--civil-share>0 のときだけ。既存名簿は 0=完全に同一挙動)。
    n_civil = inject_civil_servants(skeletons, args.civil_share, rng)

    personas, chosen_texts = [], []
    for i, sk in enumerate(skeletons):
        given = _GIVEN_F if sk["gender"] == "女" else _GIVEN_M
        name = (f"{_FAMILY[int(rng.integers(len(_FAMILY)))]}"
                f"{given[int(rng.integers(len(given)))]}")
        traits = sample_traits(rng, tail_frac=0.1)
        thr, fw = drive_params(traits, rng)
        text = (_fallback_text(sk, name) if args.no_llm
                else llm_persona(sk, name, chosen_texts, args.model, rng))
        bad = construct_violations(text)
        if bad:
            print(f"  [{i}] 構成概念語の混入 {bad} → 手続き生成に差し替え")
            text = _fallback_text(sk, name)
        chosen_texts.append(text)
        # commuter は帰宅トリガ(bedtime_min)= 夕方の退勤時刻(depart_min)を流用する
        #  (persona.py 側で work_end より後へ持ち上げる)。非commuter は従来どおり夜就寝。
        commute = bool(sk.get("commute", False))
        if commute:
            bedtime = int(sk["depart_min"])
        else:
            bedtime = (22 * 60 + int(rng.integers(0, 24)) * 10) % 1440
        entry = {
            "name": name, "age": sk["age"], "gender": sk["gender"],
            "occupation": sk["occupation"], "visitor": sk["visitor"],
            "commute": commute,
            "persona": text, "traits": traits,
            "drive_threshold": thr, "fire_weight": fw,
            "bedtime_min": bedtime,
            "sleep_steps": int(rng.integers(39, 49)),
            "has_bicycle": bool(rng.random() < 0.15),
            "has_car": bool(rng.random() < 0.08),
        }
        if commute:                                      # 流入通勤者の追加属性
            entry["arrival_lead_min"] = int(sk["arrival_lead_min"])
            entry["commute_gateway"] = sk["commute_gateway"]
            entry["residence_line"] = sk.get("residence_line", "")
        personas.append(entry)
        if not args.no_llm and (i + 1) % 10 == 0:
            print(f"  {i + 1}/{args.n} ...")

    # 多様性の監視(distinct-2: 文字bigramの異なり率)
    all_bi = [b for t in chosen_texts for b in _bigrams(t)]
    distinct2 = len(set(all_bi)) / max(1, len(all_bi))
    # 構成比: 居住(resident)/ 通勤流入(commuter)/ 余暇来街(leisure visitor)
    n = len(personas)
    n_commute = sum(1 for p in personas if p.get("commute"))
    n_resident = sum(1 for p in personas if not p["visitor"])
    n_leisure = n - n_resident - n_commute
    civil = {occ: sum(1 for p in personas if p["occupation"] == occ)
             for occ in _CIVIL_OCCS}
    out = (Path(args.out) if args.out else REPO_ROOT / "data" / f"personas_{args.n}.json")
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.write_text(json.dumps(
        {"meta": {"seed": args.seed, "model": "procedural" if args.no_llm else args.model,
                  "distinct2": round(distinct2, 4),
                  "visitor_share_actual": round(sum(p["visitor"] for p in personas)
                                                / n, 3),
                  "composition": {"resident": n_resident, "commuter": n_commute,
                                  "leisure_visitor": n_leisure},
                  # 公務員(行政・税バッチ)。give: society.government の日次ペイロールで予算から支給。
                  "civil_servants": {"total": n_civil, **civil}},
         "personas": personas}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"written: {out}  (distinct-2={distinct2:.3f})")
    print(f"  構成比: 居住={n_resident} ({n_resident/n:.0%}) / "
          f"通勤流入={n_commute} ({n_commute/n:.0%}) / "
          f"余暇来街={n_leisure} ({n_leisure/n:.0%})")
    if n_civil:
        print(f"  公務員={n_civil} ({n_civil/n:.0%}): "
              + " / ".join(f"{k}={v}" for k, v in civil.items()))


if __name__ == "__main__":
    main()
