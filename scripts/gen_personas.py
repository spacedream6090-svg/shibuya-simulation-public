"""実験用ペルソナの自動生成パイプライン(プール生成 → 無作為抽出)。

使い方:
    python scripts/gen_personas.py --pool 3000 --sample 100 --seed 42 \
        --out data/personas_gen_100.json [--pool-out data/persona_pool.json]

方針(ユーザー要望 2026-07-07):
  - 本番では「余剰に生成したプール」から無作為抽出し、現実に近い分布にする。
    → プール(--pool 件)を作ってから --sample 件を無作為抽出する2段構造。
  - スキーマは data/personas_100_civic.json(scripts/build_personas.py 生成物)を
    厳密に踏襲する(フィールド名・型・値域を合わせる)。公務員(区職員/警察官/消防士)
    が数%含まれる点も維持する。
  - 生成は完全決定論: 乱数は random.Random(seed) のみ。シミュ本体(RngHub)や numpy、
    LLM は一切使わない=このスクリプト単体で完結する。
  - 性格・価値観・趣味は語彙リストの組合せでテンプレ文を作り、多様性を出す(LLM 不使用)。
  - 名前は日本人名リスト(姓×名)。男女で名リストを分け、姓×名を無作為抽出(重複なし)。

シミュ実行時は生成済み JSON を読むだけ(本体の LLM 負荷ゼロ・決定論)。
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# ----------------------------------------------------------------- 日本人名リスト(姓×名)
# 男女で名を分ける(=名文字列は男女で互いに素)ので、姓+名の完全文字列は全体で一意にできる。
_FAMILY = [
    "佐藤", "鈴木", "高橋", "田中", "伊藤", "渡辺", "山本", "中村", "小林", "加藤",
    "吉田", "山田", "佐々木", "山口", "松本", "井上", "木村", "林", "斎藤", "清水",
    "山崎", "森", "池田", "橋本", "阿部", "石川", "山下", "中島", "石井", "小川",
    "前田", "岡田", "長谷川", "藤田", "後藤", "近藤", "村上", "遠藤", "青木", "坂本",
    "福田", "太田", "西村", "藤井", "岡本", "三浦", "松田", "中川", "中野", "原田",
    "小野", "田村", "竹内", "金子", "和田", "中山", "石田", "上田", "森田", "宮本",
]
_GIVEN_F = [
    "美咲", "陽菜", "結衣", "葵", "凛", "さくら", "芽衣", "千夏", "真央", "花音",
    "美月", "詩織", "彩乃", "智子", "由美", "恵", "直美", "幸子", "京子", "和子",
    "愛", "あや", "莉子", "杏", "楓", "七海", "美優", "彩香", "沙織", "麻衣",
    "奈々", "遥", "萌", "香織", "成美", "有紗", "咲良", "美穂", "文香", "早紀",
    "瞳", "菜々子", "琴音", "優花", "千香", "真由", "千尋", "美嘉",
]
_GIVEN_M = [
    "翔太", "蓮", "大輝", "颯太", "拓海", "悠人", "健太", "亮", "光", "駿",
    "隆", "誠", "浩二", "健一", "修", "茂", "勝", "昇", "清", "実",
    "翔", "大和", "海斗", "陸", "湊", "樹", "悠斗", "航", "翼", "拓也",
    "圭", "涼介", "雄大", "和也", "直樹", "達也", "亮太", "祐介", "剛", "隼人",
    "颯", "遼", "純", "慎一", "宏", "克也", "秀樹", "洋平",
]

# ----------------------------------------------------------------- 職業分布(渋谷の昼間人口)
# (カテゴリ名, カテゴリ重み, [(具体職業, 副重み), ...])。カテゴリ順は固定(決定論)。
# 具体職業は既存スキーマ / society.agents.persona._WORK_CAT の語彙に合わせる
# (会社員=office, エンジニア/デザイナー=office, カフェ店員=food, アパレル店員/美容師=shop,
#  大学生=education, フリーランス/写真家/バンドマン/配達員/無職=職場なし)。
# 公務員(区職員/警察官/消防士)は space 上の職場を持たず society.government の予算から支給される。
_OCC_CATEGORIES: list[tuple[str, float, list[tuple[str, float]]]] = [
    ("会社員",             0.29, [("会社員", 1.0)]),
    ("IT・クリエイティブ", 0.13, [("エンジニア", 0.55), ("デザイナー", 0.45)]),
    ("販売・飲食サービス", 0.18, [("アパレル店員", 0.30), ("カフェ店員", 0.30),
                                  ("美容師", 0.20), ("配達員", 0.20)]),
    ("学生",               0.11, [("大学生", 1.0)]),
    ("フリーランス",       0.10, [("フリーランス", 0.5), ("写真家", 0.3), ("バンドマン", 0.2)]),
    ("公務員",             0.07, [("区職員", 0.5), ("警察官", 0.3), ("消防士", 0.2)]),
    ("経営者",             0.04, [("経営者", 1.0)]),
    ("無職・求職",         0.08, [("無職", 1.0)]),
]
_CIVIL_OCCS = ("区職員", "警察官", "消防士")
_STUDENT_OCCS = {"大学生"}

# ----------------------------------------------------------------- 年齢帯(20-30代を厚めに)
_AGE_BANDS: list[tuple[tuple[int, int], float]] = [
    ((18, 24), 0.15), ((25, 29), 0.19), ((30, 39), 0.29),
    ((40, 49), 0.16), ((50, 59), 0.11), ((60, 75), 0.10),
]

# 通勤流入者(commuter)= visitor の下位型。出入口/沿線タグ(data/shibuya_population.json 由来)。
_RESIDENCE_LINES = ["世田谷方面", "二子玉川・田園都市線沿線", "横浜・東横線沿線",
                    "下北沢・井の頭線沿線", "新宿・山手線沿線", "杉並方面",
                    "目黒方面", "川崎・多摩方面"]
_VISITOR_SHARE = 0.56       # 昼夜間人口比 由来(data/shibuya_population.json)
_COMMUTER_SHARE = 0.62      # visitor のうち通勤・通学で来る割合
_GATEWAY_STATION_SHARE = 0.87

# ----------------------------------------------------------------- 性格・価値観・趣味の語彙
_HOBBIES = ["音楽ライブ", "カフェ巡り", "写真", "ランニング", "古着屋めぐり", "読書",
            "ゲーム", "自炊", "街歩き", "アニメ", "映画", "筋トレ", "サウナ", "釣り",
            "登山", "ボードゲーム", "陶芸", "DIY", "自転車", "韓国ドラマ", "旅行", "散歩"]
_VALUES = ["家族との時間", "自分のペース", "新しい出会い", "誠実さ", "挑戦", "安定した暮らし",
           "仲間", "好奇心", "自由な時間", "地元とのつながり", "丁寧な仕事", "遊び心"]
_TONES = ["よく笑う", "淡々と話す", "早口でよくしゃべる", "口数は少なめ", "冗談が多い",
          "丁寧な物腰", "熱く語りがち", "マイペース", "人懐っこい", "皮肉っぽい",
          "おっとりしている", "さっぱりした性格"]


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _weighted(rng: random.Random, pairs: list[tuple]) -> object:
    """[(値, 重み), ...] から重み付き無作為抽出(random.Random.choices)。"""
    pop = [p[0] for p in pairs]
    wts = [p[1] for p in pairs]
    return rng.choices(pop, weights=wts, k=1)[0]


def _sample_age(rng: random.Random) -> int:
    (lo, hi) = _weighted(rng, [(b, w) for b, w in _AGE_BANDS])
    return rng.randint(lo, hi)


def _sample_occupation(rng: random.Random) -> str:
    cat = _weighted(rng, [(subs, w) for _name, w, subs in _OCC_CATEGORIES])
    return str(_weighted(rng, cat))


def _sample_traits(rng: random.Random, tail_frac: float = 0.1) -> dict[str, float]:
    """trait を N(0.5, 0.18) clip[0,1] から。tail_frac の確率で1因子を上位裾[0.9,1.0]に確保。

    値域は society.factors.registry.sample_traits を踏襲(内的統制/NFC/リスク許容の3因子)。
    """
    names = ["internal_locus", "nfc", "risk_tolerance"]     # sorted 済み(出力キー順)
    tail = names[rng.randrange(len(names))] if rng.random() < tail_frac else None
    out: dict[str, float] = {}
    for name in names:
        if name == tail:
            out[name] = rng.uniform(0.9, 1.0)
        else:
            out[name] = _clip(rng.gauss(0.5, 0.18), 0.0, 1.0)
    return out


def _drive_params(traits: dict[str, float], rng: random.Random) -> tuple[float, float]:
    """traits → (drive_threshold, fire_weight)。値域は factors.registry.drive_params を踏襲。"""
    nfc = traits["nfc"]
    locus = traits["internal_locus"]
    thr = _clip(rng.gauss(0.60 - 0.20 * (nfc - 0.5) * 2, 0.08), 0.30, 0.85)
    fw = _clip(rng.gauss(0.50 + 0.30 * (locus - 0.5) * 2, 0.12), 0.15, 0.90)
    return thr, fw


def _sample_depart(rng: random.Random) -> int:
    """夕方の退勤=帰宅トリガ時刻(分 of day、10分刻み)。夕ピーク+夜遊びの遅い裾。"""
    if rng.random() < 0.8:
        v = rng.gauss(1110, 45)
    else:
        v = rng.uniform(1320, 1440)
    v = int(round(v / 10.0) * 10)
    return int(_clip(v, 16 * 60, 23 * 60 + 50))


def _sample_lead(is_student: bool, rng: random.Random) -> int:
    """職場 start から逆算する到着リード(分)。通学=早め / 会社員=遅め の二峰の一方。"""
    mean, sd = (30, 15) if is_student else (40, 20)
    return int(_clip(round(rng.gauss(mean, sd)), 10, 120))


def _persona_text(name: str, age: int, occ: str, gender: str, living: str,
                  rng: random.Random) -> str:
    """語彙リストの組合せでテンプレ文を組む(LLM 不使用・心理学用語を避ける)。"""
    hobby = rng.choice(_HOBBIES)
    value = rng.choice(_VALUES)
    tone = rng.choice(_TONES)
    flavor = f"{hobby}が好き。{value}を大切にしていて、{tone}。"
    return (f"あなたは{name}、{age}歳の{occ}({gender}性)。{living}。"
            f"{flavor}自分の言葉で自然に、短く話す。")


def _build_persona(rng: random.Random, name: str, gender: str) -> dict:
    age = _sample_age(rng)
    occ = _sample_occupation(rng)
    visitor = rng.random() < _VISITOR_SHARE
    # commuter(通勤流入)= 職に就いている visitor の一部。無職の余暇来街者は除く。
    commute = bool(visitor and occ != "無職" and rng.random() < _COMMUTER_SHARE)

    traits = _sample_traits(rng)
    thr, fw = _drive_params(traits, rng)

    if commute:
        residence_line = rng.choice(_RESIDENCE_LINES)
        living = f"{residence_line}に住んでいて、毎日渋谷に通勤・通学している"
        depart = _sample_depart(rng)
        arrival_lead = _sample_lead(occ in _STUDENT_OCCS, rng)
        gateway = "station" if rng.random() < _GATEWAY_STATION_SHARE else "edge"
        bedtime = depart
    elif visitor:
        living = "渋谷によく遊びに来る"
        bedtime = (22 * 60 + rng.randrange(0, 24) * 10) % 1440
    else:
        living = "渋谷の街に住んでいる"
        bedtime = (22 * 60 + rng.randrange(0, 24) * 10) % 1440

    persona = _persona_text(name, age, occ, gender, living, rng)
    sleep_steps = rng.randint(39, 48)               # 6.5h〜8h(10分刻み)
    has_bicycle = rng.random() < 0.15
    has_car = rng.random() < 0.08

    entry = {
        "name": name, "age": age, "gender": gender, "occupation": occ,
        "visitor": visitor, "commute": commute,
        "persona": persona, "traits": traits,
        "drive_threshold": thr, "fire_weight": fw,
        "bedtime_min": int(bedtime), "sleep_steps": int(sleep_steps),
        "has_bicycle": has_bicycle, "has_car": has_car,
    }
    if commute:                                     # 流入通勤者の追加属性(スキーマ踏襲)
        entry["arrival_lead_min"] = int(arrival_lead)
        entry["commute_gateway"] = gateway
        entry["residence_line"] = residence_line
    return entry


def _name_stacks(rng: random.Random) -> tuple[list[str], list[str]]:
    """姓×名の全組合せを男女別に作りシャッフル(pop で重複なし抽出)。"""
    female = [fam + giv for fam in _FAMILY for giv in _GIVEN_F]
    male = [fam + giv for fam in _FAMILY for giv in _GIVEN_M]
    rng.shuffle(female)
    rng.shuffle(male)
    return female, male


def generate_pool(pool: int, rng: random.Random, female_share: float = 0.51) -> list[dict]:
    """決定論的にプールを生成(名前は全体で一意)。"""
    female_names, male_names = _name_stacks(rng)
    personas: list[dict] = []
    for _ in range(pool):
        gender = "女" if rng.random() < female_share else "男"
        stack = female_names if gender == "女" else male_names
        if not stack:                               # 片側の名が尽きた(巨大プール)
            gender = "男" if gender == "女" else "女"
            stack = female_names if gender == "女" else male_names
        if not stack:
            raise RuntimeError(
                f"名前プールが枯渇: pool={pool} が名リスト(男女計 "
                f"{len(_FAMILY) * (len(_GIVEN_F) + len(_GIVEN_M))} 通り)を超えた。"
                "名リストを拡張するか --pool を減らしてください。")
        name = stack.pop()
        personas.append(_build_persona(rng, name, gender))
    return personas


def _composition(personas: list[dict]) -> dict[str, int]:
    n_commute = sum(1 for p in personas if p.get("commute"))
    n_resident = sum(1 for p in personas if not p["visitor"])
    return {"resident": n_resident, "commuter": n_commute,
            "leisure_visitor": len(personas) - n_resident - n_commute}


def _civil_counts(personas: list[dict]) -> dict[str, int]:
    counts = {occ: sum(1 for p in personas if p["occupation"] == occ)
              for occ in _CIVIL_OCCS}
    return {"total": sum(counts.values()), **counts}


def _age_band_of(age: int) -> str:
    for (lo, hi), _w in _AGE_BANDS:
        if lo <= age <= hi:
            return f"{lo}-{hi}"
    return "その他"


def _print_summary(sample: list[dict], pool_size: int) -> None:
    n = len(sample)
    print(f"抽出: {n} 名(プール {pool_size} 名から無作為抽出)")
    comp = _composition(sample)
    print(f"  構成: 居住={comp['resident']} / 通勤流入={comp['commuter']} / "
          f"余暇来街={comp['leisure_visitor']}")
    print("  職業分布:")
    for occ, c in sorted(Counter(p["occupation"] for p in sample).items(),
                         key=lambda kv: (-kv[1], kv[0])):
        print(f"    {occ:<8} {c:>3}  ({c / n:.0%})")
    print("  年齢分布:")
    for (lo, hi), _w in _AGE_BANDS:
        band = f"{lo}-{hi}"
        c = sum(1 for p in sample if _age_band_of(p["age"]) == band)
        bar = "#" * c
        print(f"    {band:<7} {c:>3}  {bar}")
    civ = _civil_counts(sample)
    print(f"  公務員={civ['total']}: "
          + " / ".join(f"{k}={civ[k]}" for k in _CIVIL_OCCS))


def _resolve_out(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else REPO_ROOT / p


def _write(path: Path, meta: dict, personas: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"meta": meta, "personas": personas},
                               ensure_ascii=False, indent=1), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="実験用ペルソナの自動生成(プール→無作為抽出)")
    ap.add_argument("--pool", type=int, default=3000, help="生成するプール件数(余剰)")
    ap.add_argument("--sample", type=int, default=100, help="プールから無作為抽出する件数")
    ap.add_argument("--seed", type=int, default=42, help="乱数シード(完全決定論)")
    ap.add_argument("--out", default="data/personas_gen_100.json",
                    help="抽出名簿の出力先 JSON")
    ap.add_argument("--pool-out", default=None,
                    help="指定時、プール全体も保存する(本番の再抽出用)")
    args = ap.parse_args(argv)

    if args.sample > args.pool:
        ap.error(f"--sample ({args.sample}) は --pool ({args.pool}) 以下にしてください")

    rng = random.Random(args.seed)
    pool = generate_pool(args.pool, rng)
    sample = rng.sample(pool, args.sample)          # プールから無作為抽出(順序も決定論)

    base_meta = {"seed": args.seed, "model": "procedural-vocab",
                 "generator": "scripts/gen_personas.py",
                 "pool_size": args.pool}
    sample_meta = {**base_meta, "sample_size": args.sample,
                   "visitor_share_actual": round(
                       sum(p["visitor"] for p in sample) / len(sample), 3),
                   "composition": _composition(sample),
                   "civil_servants": _civil_counts(sample)}
    out_path = _resolve_out(args.out)
    _write(out_path, sample_meta, sample)
    print(f"written: {out_path}")

    if args.pool_out:
        pool_meta = {**base_meta,
                     "composition": _composition(pool),
                     "civil_servants": _civil_counts(pool)}
        pool_path = _resolve_out(args.pool_out)
        _write(pool_path, pool_meta, pool)
        print(f"written pool: {pool_path}")

    _print_summary(sample, args.pool)


if __name__ == "__main__":
    main()
