"""ペルソナ100万プールの決定論生成(W2 P5 / batch: persona-pool 1M)。

使い方:
    python scripts/build_persona_pool.py                       # フル(~100万)
    python scripts/build_persona_pool.py --fraction 0.02       # 小規模(テスト/検証用)
    python scripts/build_persona_pool.py --seed 42 --out data/persona_pool

設計(docs/plans/persona-pool.md §2/§5 + docs/plans/w2-execution-plan.md §4 P5):
  5層をすべて**決定論**(seed 固定・LLM 不使用)で生成し、シャーディングした JSONL に吐く。
    L1 住民      : shibuya_population.json の周辺分布から IPF で骨格(夜間人口 ~3万)
    L2 域内従業者: 組織台帳 organizations_shibuya_wide11k.json の employees を需要源に逆算
                   (会社 employees の総和 + 学校の教職員)。org_id/role/shift を本人に埋め込む
    L3 定期来街  : 学生(学校 capacity から逆算・org_id=学校)+ 常連(習い事等・週次)
    L4 非定期来街: 回転の主層(観光/買物/ビジネス来訪の匿名合成セグメント。数十万)
    L5 役割ペルソナ: 駅員/電車運転士/バス運転士/タクシー運転手/警察官/配信者 + 議員
  各レコードに layer タグと presence(§5 のローテーション区分)を付与する。

決定論:
  乱数はパートごとに numpy の SeedSequence([master_seed, layer_code, part_index]) から導出。
  → 同 seed・同 fraction なら**他パートの実行順に依存せず**同一出力(シャード並列可能)。

スキーマ互換:
  フィールド名は data/personas_300_civic.json に揃える(name/age/gender/occupation/visitor/
  commute/persona/traits/drive_threshold/fire_weight/bedtime_min/sleep_steps/has_bicycle/
  has_car (+commuter: arrival_lead_min/commute_gateway/residence_line))。
  追加フィールド(id/layer/presence/org_id/role/shift_pattern/visit_cadence/visit_purpose/
  visit_rate/is_foreign/party_size/revisit/post/duty_pattern/seat_id/party)はスキーマ拡張。
  拡張分は persona.build_agent が entry.get で無視するため後方互換(engine 非改変)。

出力(data/persona_pool/):
  {layer}/part-XXXX.jsonl(1行=1レコード)+ meta.json(層別件数・seed・スキーマ版・シャード一覧)
  + llm_targets.json(深いペルソナ化=LLM上塗り対象の id 一覧: L5 全員 + L1/L3常連の一部)。
  ※ data/persona_pool/ は .gitignore 済み(数百MB級はコミットしない)。

副産物(コミット対象・小):
  data/personas_councilors.json … 議員名簿(標準スキーマ・from_roster で着席可能)。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

try:                                       # cp932 コンソールでの print 死を回避
    sys.stdout.reconfigure(errors="replace")
except (AttributeError, ValueError):       # pytest capture 等では reconfigure 不可 → 無視
    pass

REPO_ROOT = Path(__file__).resolve().parents[1]

SCHEMA_VERSION = "persona-pool-1.0"
PART_SIZE = 50_000            # 1 シャードの最大レコード数

# ---- 層コード(SeedSequence の第2要素。決定論の名前空間分離)----
LC = {"L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5}

# ---- 名前(手続き生成・実在人物を想起させない一般的な姓名の組合せ)----
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
_FAMILY_SET = set(_FAMILY)
_GIVEN_SET = set(_GIVEN_F) | set(_GIVEN_M)

# ---- 語彙(persona 文の flavor。心理学用語は使わない)----
_HOBBIES = ["音楽ライブ", "カフェ巡り", "写真", "ランニング", "古着屋めぐり", "読書",
            "ゲーム", "自炊", "街歩き", "アニメ", "映画", "筋トレ", "サウナ", "釣り",
            "登山", "ボードゲーム", "陶芸", "DIY", "自転車", "韓国ドラマ", "旅行", "散歩"]
_TONES = ["よく笑う", "淡々と話す", "早口でよくしゃべる", "口数は少なめ", "冗談が多い",
          "丁寧な物腰", "熱く語りがち", "マイペース", "人懐っこい", "皮肉っぽい",
          "おっとりしている", "さっぱりした性格"]

_RESIDENCE_LINES = ["世田谷方面", "二子玉川・田園都市線沿線", "横浜・東横線沿線",
                    "下北沢・井の頭線沿線", "新宿・山手線沿線", "杉並方面",
                    "目黒方面", "川崎・多摩方面"]

# ---- L4 来訪セグメント(匿名合成: 目的×属性。実個人は入れない)----
_VISIT_PURPOSES = [
    ("観光・見物", 0.24), ("買い物", 0.26), ("飲食", 0.18), ("エンタメ・イベント", 0.12),
    ("ビジネス来訪", 0.10), ("友人と会う", 0.07), ("通院・用事", 0.03),
]
_L4_OCCS = [
    ("会社員", 0.34), ("学生", 0.16), ("販売・サービス", 0.12), ("主婦・主夫", 0.08),
    ("フリーランス", 0.07), ("経営者", 0.04), ("公務員", 0.03), ("無職・求職", 0.06),
    ("専門職", 0.10),
]

# ---- L5 役割ペルソナ(インフラ需要から人数を確定。docs/plans/persona-pool.md §4)----
# (role, occupation, フル人数, post 候補, duty_pattern, visitor)
_L5_ROLES = [
    ("駅員",        "駅員",        300, ["改札", "ホーム", "案内", "事務室"],
     {"days": "all", "rotates": True, "shift_hours": 8}, True),
    ("電車運転士",  "電車運転士",   60, ["山手線", "埼京線", "東横・副都心線", "田園都市・半蔵門線", "井の頭線"],
     {"days": "all", "rotates": True, "shift_hours": 8}, True),
    ("バス運転士",  "バス運転士",  120, ["東口ターミナル", "西口ターミナル", "南口", "マークシティ"],
     {"days": "all", "rotates": True, "shift_hours": 8}, True),
    ("タクシー運転手", "タクシー運転手", 350, ["道玄坂乗場", "宮益坂乗場", "駅前ロータリー", "流し(bbox内)"],
     {"days": "all", "rotates": True, "shift_hours": 10}, True),
    ("警察官",      "警察官",      140, ["渋谷警察署", "駅前交番", "宇田川交番", "神南交番", "巡回ユニット"],
     {"days": "all", "rotates": True, "shift_hours": 8}, True),
    ("配信者",      "配信者",       16, ["ハチ公前", "スクランブル交差点", "センター街", "宮下パーク"],
     {"days": "all", "rotates": False, "shift_hours": 6}, True),
]
# 議員(選挙なし・事前決定。occupation は tools.COUNCILOR_OCCS に合致)。
COUNCILOR_OCC = "議員"
N_COUNCILORS = 34            # 現実の渋谷区議会=34議席。config の assembly.size(=9)以上を満たす
_PARTIES = ["無所属", "区政クラブ", "みらい渋谷", "区民ネット", "生活者連合",
            "刷新の会", "共生フォーラム"]


def _weighted_choice_arr(rng: np.random.Generator, items, n: int):
    """(値, 重み) のリストから n 個を重み付き抽出(値配列で返す)。"""
    vals = [it[0] for it in items]
    w = np.array([it[1] for it in items], dtype=float)
    w /= w.sum()
    idx = rng.choice(len(vals), size=n, p=w)
    return np.array(vals, dtype=object)[idx]


def _seed_rng(master_seed: int, layer_code: int, part_idx: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(
        entropy=[int(master_seed), int(layer_code), int(part_idx)]))


# ------------------------------------------------------------------ 共通の個体差サンプラ(ベクトル化)
def _traits_block(rng: np.random.Generator, n: int):
    """traits(internal_locus/nfc/risk_tolerance)+ drive_threshold/fire_weight をベクトル化生成。

    値域は society.factors.registry と gen_personas.py を踏襲(N(0.5,0.18) clip[0,1]・裾確保)。"""
    locus = np.clip(rng.normal(0.5, 0.18, n), 0.0, 1.0)
    nfc = np.clip(rng.normal(0.5, 0.18, n), 0.0, 1.0)
    risk = np.clip(rng.normal(0.5, 0.18, n), 0.0, 1.0)
    # tail_frac=0.1 で 1 因子を上位裾[0.9,1.0]へ
    tail_hit = rng.random(n) < 0.1
    tail_which = rng.integers(0, 3, n)
    tail_val = rng.uniform(0.9, 1.0, n)
    locus = np.where(tail_hit & (tail_which == 0), tail_val, locus)
    nfc = np.where(tail_hit & (tail_which == 1), tail_val, nfc)
    risk = np.where(tail_hit & (tail_which == 2), tail_val, risk)
    thr = np.clip(rng.normal(0.60 - 0.20 * (nfc - 0.5) * 2, 0.08), 0.30, 0.85)
    fw = np.clip(rng.normal(0.50 + 0.30 * (locus - 0.5) * 2, 0.12), 0.15, 0.90)
    return locus, nfc, risk, thr, fw


def _names_block(rng: np.random.Generator, genders):
    """性別配列に応じた姓名(手続き生成)。姓∈_FAMILY・名∈_GIVEN_*。"""
    n = len(genders)
    fam = np.array(_FAMILY, dtype=object)[rng.integers(0, len(_FAMILY), n)]
    gf = np.array(_GIVEN_F, dtype=object)[rng.integers(0, len(_GIVEN_F), n)]
    gm = np.array(_GIVEN_M, dtype=object)[rng.integers(0, len(_GIVEN_M), n)]
    out = []
    for i in range(n):
        giv = gf[i] if genders[i] == "女" else gm[i]
        out.append(str(fam[i]) + str(giv))
    return out


def _bedtime_nonwork(rng: np.random.Generator, n):
    return ((22 * 60) + rng.integers(0, 24, n) * 10) % 1440


def _sleep_steps(rng: np.random.Generator, n):
    return rng.integers(39, 49, n)


def _depart_block(rng: np.random.Generator, n):
    evening = rng.random(n) < 0.8
    v = np.where(evening, rng.normal(1110, 45, n), rng.uniform(1320, 1440, n))
    v = np.round(v / 10.0) * 10
    return np.clip(v, 16 * 60, 23 * 60 + 50).astype(int)


def _lead_block(rng: np.random.Generator, n, is_student):
    mean = np.where(is_student, 30.0, 40.0)
    sd = np.where(is_student, 15.0, 20.0)
    return np.clip(np.round(rng.normal(mean, sd)), 10, 120).astype(int)


# ------------------------------------------------------------------ シャード書き出し
class ShardWriter:
    """層ごとに part-XXXX.jsonl を PART_SIZE 刻みで書き出し、件数と先頭行ハッシュを記録する。"""

    def __init__(self, root: Path, layer: str):
        self.dir = root / layer
        self.dir.mkdir(parents=True, exist_ok=True)
        self.layer = layer
        self.part_idx = 0
        self.buf: list[str] = []
        self.count = 0
        self.shards: list[dict] = []

    def add(self, rec: dict) -> None:
        self.buf.append(json.dumps(rec, ensure_ascii=False))
        self.count += 1
        if len(self.buf) >= PART_SIZE:
            self._flush()

    def _flush(self) -> None:
        if not self.buf:
            return
        name = f"part-{self.part_idx:04d}.jsonl"
        text = "\n".join(self.buf) + "\n"
        (self.dir / name).write_text(text, encoding="utf-8")
        h = hashlib.blake2b(self.buf[0].encode("utf-8"), digest_size=8).hexdigest()
        self.shards.append({"file": f"{self.layer}/{name}", "records": len(self.buf),
                            "first_row_blake2b": h})
        self.part_idx += 1
        self.buf = []

    def close(self) -> None:
        self._flush()


# ------------------------------------------------------------------ L1 住民(IPF)
def _ipf_joint(pop: dict, iters: int = 60) -> np.ndarray:
    ages = pop["age_bands"]; occs = pop["occupations"]
    genders = list(pop["gender"].keys())
    m_age = np.array([a["share"] for a in ages], float)
    m_gen = np.array([pop["gender"][g] for g in genders], float)
    m_occ = np.array([o["share"] for o in occs], float)
    m_age /= m_age.sum(); m_gen /= m_gen.sum(); m_occ /= m_occ.sum()
    joint = np.ones((len(ages), len(genders), len(occs)))
    hints = pop.get("age_x_occupation_hints", {})
    band_keys = [f"{a['band'][0]}-{a['band'][1]}" for a in ages]
    for oi, occ in enumerate(occs):
        hint = hints.get(occ["name"])
        if isinstance(hint, dict):
            for ai, key in enumerate(band_keys):
                joint[ai, :, oi] *= float(hint.get(key, 1.0))
    for _ in range(iters):
        joint *= (m_age / joint.sum(axis=(1, 2)))[:, None, None]
        joint *= (m_gen / joint.sum(axis=(0, 2)))[None, :, None]
        joint *= (m_occ / joint.sum(axis=(0, 1)))[None, None, :]
    return joint / joint.sum()


def gen_L1(writer: ShardWriter, n: int, seed: int, pop: dict) -> None:
    ages = pop["age_bands"]; occs = pop["occupations"]
    genders = list(pop["gender"].keys())
    joint = _ipf_joint(pop)
    flat = (joint.ravel() / joint.sum())
    idx_global = 0
    for part_idx in range(0, (n + PART_SIZE - 1) // PART_SIZE):
        m = min(PART_SIZE, n - part_idx * PART_SIZE)
        rng = _seed_rng(seed, LC["L1"], part_idx)
        k = rng.choice(len(flat), size=m, p=flat)
        ai, gi, oi = np.unravel_index(k, joint.shape)
        los = np.array([ages[a]["band"][0] for a in ai])
        his = np.array([ages[a]["band"][1] for a in ai])
        age = (los + (rng.random(m) * (his - los + 1)).astype(int))
        gender = [genders[g] for g in gi]
        occ = [occs[o]["name"] for o in oi]
        names = _names_block(rng, gender)
        locus, nfc, risk, thr, fw = _traits_block(rng, m)
        bedtime = _bedtime_nonwork(rng, m)
        slp = _sleep_steps(rng, m)
        bike = rng.random(m) < 0.15; car = rng.random(m) < 0.08
        hob = np.array(_HOBBIES, object)[rng.integers(0, len(_HOBBIES), m)]
        ton = np.array(_TONES, object)[rng.integers(0, len(_TONES), m)]
        for i in range(m):
            pid = f"L1_{idx_global:08d}"; idx_global += 1
            persona = (f"あなたは{names[i]}、{int(age[i])}歳の{occ[i]}({gender[i]}性)。"
                       f"渋谷の街に住んでいる。{hob[i]}が好きで、{ton[i]}。"
                       "自分の言葉で自然に、短く話す。")
            writer.add({
                "id": pid, "layer": "L1", "presence": "resident",
                "name": names[i], "age": int(age[i]), "gender": gender[i],
                "occupation": occ[i], "visitor": False, "commute": False,
                "persona": persona,
                "traits": {"internal_locus": float(locus[i]), "nfc": float(nfc[i]),
                           "risk_tolerance": float(risk[i])},
                "drive_threshold": float(thr[i]), "fire_weight": float(fw[i]),
                "bedtime_min": int(bedtime[i]), "sleep_steps": int(slp[i]),
                "has_bicycle": bool(bike[i]), "has_car": bool(car[i]),
            })


# ------------------------------------------------------------------ L2 域内従業者(需要駆動)
def _build_L2_slots(orgs: dict, fraction: float):
    """組織台帳の employees / 教職員数から従業者スロット(org_id/role/occupation/shift/days)を展開。"""
    slots = []  # (org_id, role, occupation, shift_pattern_dict, days)
    for c in orgs["companies"]:
        emp = c["size"]["employees"]
        k = emp if fraction >= 1.0 else int(round(emp * fraction))
        if k <= 0:
            continue
        roles = c.get("roles") or ["スタッフ"]
        sp = c.get("shift_pattern", {})
        days = sp.get("days", "mon-fri")
        for i in range(k):
            role = roles[i % len(roles)]
            slots.append((c["id"], role, role, sp, days))
    # 学校の教職員(capacity から逆算: おおむね生徒12人に1人)
    for s in orgs["schools"]:
        staff = max(3, int(round(s["capacity"] / 12.0)))
        k = staff if fraction >= 1.0 else int(round(staff * fraction))
        if k <= 0:
            continue
        st_roles = [r for r in (s.get("roles") or []) if r != "学生"] or ["教員", "職員"]
        tt = s.get("timetable", {})
        sp = {"open": tt.get("start", "08:30"), "close": "17:00",
              "days": "mon-fri", "rotates": False}
        for i in range(k):
            role = st_roles[i % len(st_roles)]
            slots.append((s["id"], role, role, sp, "mon-fri"))
    return slots


def gen_L2(writer: ShardWriter, slots, seed: int) -> None:
    n = len(slots)
    idx_global = 0
    for part_idx in range(0, (n + PART_SIZE - 1) // PART_SIZE):
        base = part_idx * PART_SIZE
        m = min(PART_SIZE, n - base)
        rng = _seed_rng(seed, LC["L2"], part_idx)
        female = rng.random(m) < 0.49
        gender = ["女" if f else "男" for f in female]
        # 就業層は 22-64 中心
        age = np.clip(rng.normal(38, 11, m), 18, 68).astype(int)
        names = _names_block(rng, gender)
        locus, nfc, risk, thr, fw = _traits_block(rng, m)
        depart = _depart_block(rng, m)
        lead = _lead_block(rng, m, np.zeros(m, bool))
        gate = np.where(rng.random(m) < 0.87, "station", "edge")
        line = np.array(_RESIDENCE_LINES, object)[rng.integers(0, len(_RESIDENCE_LINES), m)]
        slp = _sleep_steps(rng, m)
        bike = rng.random(m) < 0.15; car = rng.random(m) < 0.08
        ton = np.array(_TONES, object)[rng.integers(0, len(_TONES), m)]
        for i in range(m):
            org_id, role, occ, sp, days = slots[base + i]
            pid = f"L2_{idx_global:08d}"; idx_global += 1
            persona = (f"あなたは{names[i]}、{int(age[i])}歳の{occ}({gender[i]}性)。"
                       f"{line[i]}に住んでいて、渋谷({role})に通勤している。{ton[i]}。"
                       "自分の言葉で自然に、短く話す。")
            writer.add({
                "id": pid, "layer": "L2", "presence": "workday_shift",
                "name": names[i], "age": int(age[i]), "gender": gender[i],
                "occupation": occ, "visitor": True, "commute": True,
                "persona": persona,
                "traits": {"internal_locus": float(locus[i]), "nfc": float(nfc[i]),
                           "risk_tolerance": float(risk[i])},
                "drive_threshold": float(thr[i]), "fire_weight": float(fw[i]),
                "bedtime_min": int(depart[i]), "sleep_steps": int(slp[i]),
                "has_bicycle": bool(bike[i]), "has_car": bool(car[i]),
                "arrival_lead_min": int(lead[i]), "commute_gateway": str(gate[i]),
                "residence_line": str(line[i]),
                "org_id": org_id, "role": role,
                "shift_pattern": {"open": sp.get("open", "09:00"),
                                  "close": sp.get("close", "18:00"),
                                  "days": days, "rotates": bool(sp.get("rotates", False))},
                "work_days": days,
            })


# ------------------------------------------------------------------ L3 定期来街(学生+常連)
_SCHOOL_OCC = {"区立小学校": ("小学生", (6, 12)), "区立中学校": ("中学生", (12, 15)),
               "小中一貫校": ("小中学生", (6, 15)), "高校": ("高校生", (15, 18)),
               "大学": ("大学生", (18, 23)), "専門学校": ("専門学生", (18, 24))}


def _build_L3_student_slots(orgs: dict, fraction: float):
    slots = []  # (school_id, occ, age_lo, age_hi, start)
    for s in orgs["schools"]:
        cap = s["capacity"]
        k = cap if fraction >= 1.0 else int(round(cap * fraction))
        if k <= 0:
            continue
        occ, (lo, hi) = _SCHOOL_OCC.get(s["school_type"], ("学生", (7, 18)))
        start = s.get("timetable", {}).get("start", "08:30")
        for _ in range(k):
            slots.append((s["id"], occ, lo, hi, start))
    return slots


def gen_L3_students(writer: ShardWriter, slots, seed: int) -> None:
    n = len(slots)
    idx_global = 0
    for part_idx in range(0, (n + PART_SIZE - 1) // PART_SIZE):
        base = part_idx * PART_SIZE
        m = min(PART_SIZE, n - base)
        rng = _seed_rng(seed, LC["L3"], part_idx)
        female = rng.random(m) < 0.49
        gender = ["女" if f else "男" for f in female]
        names = _names_block(rng, gender)
        locus, nfc, risk, thr, fw = _traits_block(rng, m)
        lead = _lead_block(rng, m, np.ones(m, bool))
        gate = np.where(rng.random(m) < 0.87, "station", "edge")
        line = np.array(_RESIDENCE_LINES, object)[rng.integers(0, len(_RESIDENCE_LINES), m)]
        slp = _sleep_steps(rng, m)
        bedt = _bedtime_nonwork(rng, m)
        for i in range(m):
            sid, occ, lo, hi, start = slots[base + i]
            age = int(lo + rng.integers(0, hi - lo + 1))
            pid = f"L3_{idx_global:08d}"; idx_global += 1
            persona = (f"あなたは{names[i]}、{age}歳の{occ}({gender[i]}性)。"
                       f"{line[i]}から渋谷の学校に通っている。"
                       "自分の言葉で自然に、短く話す。")
            writer.add({
                "id": pid, "layer": "L3", "presence": "cadence",
                "name": names[i], "age": age, "gender": gender[i],
                "occupation": occ, "visitor": True, "commute": True,
                "persona": persona,
                "traits": {"internal_locus": float(locus[i]), "nfc": float(nfc[i]),
                           "risk_tolerance": float(risk[i])},
                "drive_threshold": float(thr[i]), "fire_weight": float(fw[i]),
                "bedtime_min": int(bedt[i]), "sleep_steps": int(slp[i]),
                "has_bicycle": bool(rng.random() < 0.2), "has_car": False,
                "arrival_lead_min": int(lead[i]), "commute_gateway": str(gate[i]),
                "residence_line": str(line[i]),
                "org_id": sid, "role": "学生",
                "visit_cadence": "school_day", "subtype": "student",
            })


def gen_L3_regulars(writer: ShardWriter, n: int, seed: int, part_offset: int) -> None:
    idx_global = 0
    for p in range(0, (n + PART_SIZE - 1) // PART_SIZE):
        m = min(PART_SIZE, n - p * PART_SIZE)
        rng = _seed_rng(seed, LC["L3"], 1000 + p)  # 学生パートと名前空間を分離
        female = rng.random(m) < 0.52
        gender = ["女" if f else "男" for f in female]
        age = np.clip(rng.normal(34, 13, m), 16, 78).astype(int)
        names = _names_block(rng, gender)
        occ = _weighted_choice_arr(rng, _L4_OCCS, m)
        locus, nfc, risk, thr, fw = _traits_block(rng, m)
        line = np.array(_RESIDENCE_LINES, object)[rng.integers(0, len(_RESIDENCE_LINES), m)]
        cad = rng.integers(2, 5, m)   # 週2-4回
        slp = _sleep_steps(rng, m)
        bedt = _bedtime_nonwork(rng, m)
        hob = np.array(_HOBBIES, object)[rng.integers(0, len(_HOBBIES), m)]
        for i in range(m):
            pid = f"L3reg_{idx_global:08d}"; idx_global += 1
            persona = (f"あなたは{names[i]}、{int(age[i])}歳の{occ[i]}({gender[i]}性)。"
                       f"{line[i]}に住み、週{int(cad[i])}回ほど渋谷に通う常連。{hob[i]}が好き。"
                       "自分の言葉で自然に、短く話す。")
            writer.add({
                "id": pid, "layer": "L3", "presence": "cadence",
                "name": names[i], "age": int(age[i]), "gender": gender[i],
                "occupation": str(occ[i]), "visitor": True, "commute": False,
                "persona": persona,
                "traits": {"internal_locus": float(locus[i]), "nfc": float(nfc[i]),
                           "risk_tolerance": float(risk[i])},
                "drive_threshold": float(thr[i]), "fire_weight": float(fw[i]),
                "bedtime_min": int(bedt[i]), "sleep_steps": int(slp[i]),
                "has_bicycle": bool(rng.random() < 0.15), "has_car": bool(rng.random() < 0.08),
                "residence_line": str(line[i]),
                "visit_cadence": f"weekly_{int(cad[i])}", "subtype": "regular",
            })


# ------------------------------------------------------------------ L4 非定期来街(回転主層)
def gen_L4(writer: ShardWriter, n: int, seed: int) -> None:
    idx_global = 0
    for part_idx in range(0, (n + PART_SIZE - 1) // PART_SIZE):
        m = min(PART_SIZE, n - part_idx * PART_SIZE)
        rng = _seed_rng(seed, LC["L4"], part_idx)
        female = rng.random(m) < 0.52
        gender = ["女" if f else "男" for f in female]
        age = np.clip(rng.normal(36, 14, m), 15, 82).astype(int)
        names = _names_block(rng, gender)
        occ = _weighted_choice_arr(rng, _L4_OCCS, m)
        purpose = _weighted_choice_arr(rng, _VISIT_PURPOSES, m)
        locus, nfc, risk, thr, fw = _traits_block(rng, m)
        is_foreign = rng.random(m) < 0.15
        party = 1 + rng.integers(0, 5, m)  # 同行人数 1-5
        # 来訪確率/日(low・log-uniform): 大多数は稀。~[0.003, 0.06]
        visit_rate = np.exp(rng.uniform(np.log(0.003), np.log(0.06), m))
        revisit = rng.random(m) < 0.10
        slp = _sleep_steps(rng, m)
        bedt = _bedtime_nonwork(rng, m)
        for i in range(m):
            pid = f"L4_{idx_global:08d}"; idx_global += 1
            tag = "訪日外国人" if is_foreign[i] else str(occ[i])
            persona = (f"あなたは{names[i]}、{int(age[i])}歳({gender[i]}性)。"
                       f"{purpose[i]}のため渋谷を訪れる{tag}。"
                       "自分の言葉で自然に、短く話す。")
            writer.add({
                "id": pid, "layer": "L4", "presence": "stochastic",
                "name": names[i], "age": int(age[i]), "gender": gender[i],
                "occupation": str(occ[i]), "visitor": True, "commute": False,
                "persona": persona,
                "traits": {"internal_locus": float(locus[i]), "nfc": float(nfc[i]),
                           "risk_tolerance": float(risk[i])},
                "drive_threshold": float(thr[i]), "fire_weight": float(fw[i]),
                "bedtime_min": int(bedt[i]), "sleep_steps": int(slp[i]),
                "has_bicycle": False, "has_car": bool(rng.random() < 0.05),
                "visit_purpose": str(purpose[i]),
                "visit_rate": round(float(visit_rate[i]), 4),
                "is_foreign": bool(is_foreign[i]), "party_size": int(party[i]),
                "revisit": bool(revisit[i]),
            })


# ------------------------------------------------------------------ L5 役割ペルソナ + 議員
def gen_L5(writer: ShardWriter, seed: int, fraction: float):
    """役割ペルソナ(駅員/運転士/タクシー/警察/配信者)+ 議員。councilors はフル 34 で固定。"""
    rng = _seed_rng(seed, LC["L5"], 0)
    councilors = []          # 標準スキーマの議員(personas_councilors.json 用に返す)
    idx_global = 0

    # --- 役割(インフラ需要)---
    for role, occ, full_n, posts, duty, is_visitor in _L5_ROLES:
        k = full_n if fraction >= 1.0 else max(1, int(round(full_n * fraction)))
        female = rng.random(k) < 0.35
        gender = ["女" if f else "男" for f in female]
        age = np.clip(rng.normal(40, 11, k), 20, 66).astype(int)
        names = _names_block(rng, gender)
        locus, nfc, risk, thr, fw = _traits_block(rng, k)
        line = np.array(_RESIDENCE_LINES, object)[rng.integers(0, len(_RESIDENCE_LINES), k)]
        slp = _sleep_steps(rng, k)
        bedt = _bedtime_nonwork(rng, k)
        for i in range(k):
            post = posts[i % len(posts)]
            pid = f"L5_{idx_global:08d}"; idx_global += 1
            persona = (f"あなたは{names[i]}、{int(age[i])}歳の{occ}({gender[i]}性)。"
                       f"渋谷の{post}で{role}として働いている。"
                       "自分の言葉で自然に、短く話す。")
            writer.add({
                "id": pid, "layer": "L5", "presence": "duty",
                "name": names[i], "age": int(age[i]), "gender": gender[i],
                "occupation": occ, "visitor": bool(is_visitor),
                "commute": bool(is_visitor),
                "persona": persona,
                "traits": {"internal_locus": float(locus[i]), "nfc": float(nfc[i]),
                           "risk_tolerance": float(risk[i])},
                "drive_threshold": float(thr[i]), "fire_weight": float(fw[i]),
                "bedtime_min": int(bedt[i]), "sleep_steps": int(slp[i]),
                "has_bicycle": False, "has_car": bool(rng.random() < 0.1),
                "residence_line": str(line[i]) if is_visitor else "",
                "role": role, "post": post, "duty_pattern": dict(duty),
            })

    # --- 議員(選挙なし・事前決定。フル 34 固定)---
    # ★専用の乱数ストリーム(part_index=1)+ 固定 id で fraction 非依存にする
    #   = 議員名簿(コミット対象)が --fraction に関わらず同一(検収の安定性)。
    crng = _seed_rng(seed, LC["L5"], 1)
    k = N_COUNCILORS
    female = crng.random(k) < 0.35
    gender = ["女" if f else "男" for f in female]
    age = np.clip(crng.normal(52, 9, k), 30, 74).astype(int)
    names = _names_block(crng, gender)
    locus, nfc, risk, thr, fw = _traits_block(crng, k)
    slp = _sleep_steps(crng, k)
    bedt = _bedtime_nonwork(crng, k)
    car = crng.random(k) < 0.3
    for i in range(k):
        party = _PARTIES[i % len(_PARTIES)]
        pid = f"L5c_{i + 1:03d}"       # 議員は固定 id(fraction 非依存)
        seat_id = f"seat_{i + 1:02d}"
        persona = (f"あなたは{names[i]}、{int(age[i])}歳の渋谷区議会議員({gender[i]}性)。"
                   f"会派は{party}。渋谷に住み、区政に取り組んでいる。"
                   "自分の言葉で自然に、短く話す。")
        rec = {
            "id": pid, "layer": "L5", "presence": "resident",
            "name": names[i], "age": int(age[i]), "gender": gender[i],
            "occupation": COUNCILOR_OCC, "visitor": False, "commute": False,
            "persona": persona,
            "traits": {"internal_locus": float(locus[i]), "nfc": float(nfc[i]),
                       "risk_tolerance": float(risk[i])},
            "drive_threshold": float(thr[i]), "fire_weight": float(fw[i]),
            "bedtime_min": int(bedt[i]), "sleep_steps": int(slp[i]),
            "has_bicycle": False, "has_car": bool(car[i]),
            "role": "議員", "seat_id": seat_id, "party": party,
        }
        writer.add(rec)
        # 議員名簿は標準スキーマ(拡張キーは残しても build_agent は無視する)
        councilors.append(rec)
    return councilors


# ------------------------------------------------------------------ メイン
def build_pool(out_dir: Path, seed: int, fraction: float,
               orgs: dict, pop: dict, total_target: int):
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 層別目標
    n_L1 = int(round(30_000 * fraction))
    L2_slots = _build_L2_slots(orgs, fraction)
    n_L2 = len(L2_slots)
    L3_student_slots = _build_L3_student_slots(orgs, fraction)
    n_L3s = len(L3_student_slots)
    n_L3r = int(round(20_000 * fraction))
    # L5 件数を先に確定(L4=残り の計算に必要)
    n_L5 = N_COUNCILORS + sum(
        (full if fraction >= 1.0 else max(1, int(round(full * fraction))))
        for _r, _o, full, _p, _d, _v in _L5_ROLES)
    n_fixed = n_L1 + n_L2 + n_L3s + n_L3r + n_L5
    n_L4 = max(0, int(round(total_target * fraction)) - n_fixed)

    writers: dict[str, ShardWriter] = {}
    timings: dict[str, float] = {}

    def _run(layer, fn):
        w = writers.setdefault(layer, ShardWriter(out_dir, layer))
        s = time.time()
        fn(w)
        timings[layer] = timings.get(layer, 0.0) + (time.time() - s)

    _run("L1", lambda w: gen_L1(w, n_L1, seed, pop))
    _run("L2", lambda w: gen_L2(w, L2_slots, seed))
    _run("L3", lambda w: gen_L3_students(w, L3_student_slots, seed))
    _run("L3", lambda w: gen_L3_regulars(w, n_L3r, seed, part_offset=0))
    _run("L4", lambda w: gen_L4(w, n_L4, seed))
    councilors = []

    def _run_L5(w):
        nonlocal councilors
        councilors = gen_L5(w, seed, fraction)
    _run("L5", _run_L5)

    for w in writers.values():
        w.close()

    layer_counts = {ly: w.count for ly, w in writers.items()}
    total = sum(layer_counts.values())
    shards = [s for w in writers.values() for s in w.shards]

    # ---- llm_targets(深いペルソナ化=LLM上塗り対象: L5 全員 + L1/L3常連の一部)----
    llm_targets = _collect_llm_targets(out_dir, seed)

    meta = {
        "schema_version": SCHEMA_VERSION,
        "generator": "scripts/build_persona_pool.py",
        "seed": seed, "fraction": fraction,
        "total_target": total_target, "total_generated": total,
        "layer_counts": layer_counts,
        "layer_targets": {"L1": n_L1, "L2": n_L2, "L3": n_L3s + n_L3r,
                          "L4": n_L4, "L5": n_L5},
        "councilors": len(councilors), "councilor_occupation": COUNCILOR_OCC,
        "presence_keys": ["resident", "workday_shift", "cadence", "stochastic", "duty"],
        "base_schema_ref": "data/personas_300_civic.json",
        "schema_extensions": [
            "id", "layer", "presence", "org_id", "role", "shift_pattern", "work_days",
            "visit_cadence", "subtype", "visit_purpose", "visit_rate", "is_foreign",
            "party_size", "revisit", "post", "duty_pattern", "seat_id", "party"],
        "sources": {
            "organizations": "data/organizations_shibuya_wide11k.json",
            "population_marginals": "data/shibuya_population.json"},
        "employees_ledger_total": sum(c["size"]["employees"] for c in orgs["companies"]),
        "llm_targets_count": len(llm_targets),
        "elapsed_sec": round(time.time() - t0, 2),
        "layer_elapsed_sec": {k: round(v, 2) for k, v in timings.items()},
        "shards": shards,
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return meta, councilors


def _collect_llm_targets(out_dir: Path, seed: int) -> list[str]:
    """LLM 上塗り対象 id を層別ファイルから収集(L5 全員 + L1 10% + L3常連 全員)。~2-3%。"""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 999]))
    targets: list[str] = []
    # L5 全員
    for f in sorted((out_dir / "L5").glob("part-*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line:
                targets.append(json.loads(line)["id"])
    # L3 常連(subtype=regular)全員
    for f in sorted((out_dir / "L3").glob("part-*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            r = json.loads(line)
            if r.get("subtype") == "regular":
                targets.append(r["id"])
    # L1 の 10%
    l1_ids = []
    for f in sorted((out_dir / "L1").glob("part-*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line:
                l1_ids.append(json.loads(line)["id"])
    if l1_ids:
        k = max(1, int(round(len(l1_ids) * 0.10)))
        sel = rng.choice(len(l1_ids), size=k, replace=False)
        targets.extend(sorted(l1_ids[i] for i in sel))
    (out_dir / "llm_targets.json").write_text(
        json.dumps({"note": "深いペルソナ化(LLM上塗り)対象の id。生成は後日(本選前)。"
                            "L5全員 + L3常連 + L1の10%。",
                    "seed": seed, "count": len(targets), "ids": targets},
                   ensure_ascii=False), encoding="utf-8")
    return targets


def _write_councilors_json(councilors: list[dict], seed: int,
                           path: Path | None = None) -> Path:
    """議員名簿(コミット対象・小)。標準スキーマに整形(from_roster で着席可能)。"""
    path = path or (REPO_ROOT / "data" / "personas_councilors.json")
    meta = {"seed": seed, "generator": "scripts/build_persona_pool.py",
            "note": "名簿制議会(選挙なし・事前決定)。occupation=議員 は society.tools.COUNCILOR_OCCS "
                    "に合致。conf の institution_routes.assembly.from_roster=true で着席する。",
            "count": len(councilors), "occupation": COUNCILOR_OCC}
    path.write_text(json.dumps({"meta": meta, "personas": councilors},
                               ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description="ペルソナ100万プールの決定論生成(W2 P5)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fraction", type=float, default=1.0,
                    help="層別目標に掛ける倍率(1.0=フル~100万。テストは小さく)")
    ap.add_argument("--total", type=int, default=1_000_000,
                    help="fraction=1.0 での目標総数(既定 100万)")
    ap.add_argument("--out", default="data/persona_pool",
                    help="出力ディレクトリ(既定 data/persona_pool・.gitignore 済み)")
    ap.add_argument("--no-councilors-json", action="store_true",
                    help="data/personas_councilors.json を書かない")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir

    orgs = json.loads((REPO_ROOT / "data" / "organizations_shibuya_wide11k.json")
                      .read_text(encoding="utf-8"))
    pop = json.loads((REPO_ROOT / "data" / "shibuya_population.json")
                     .read_text(encoding="utf-8"))

    meta, councilors = build_pool(out_dir, args.seed, args.fraction, orgs, pop, args.total)

    if not args.no_councilors_json:
        cpath = _write_councilors_json(councilors, args.seed)
        print(f"written councilors: {cpath} ({len(councilors)})")

    print(f"pool written: {out_dir}")
    print(f"  total={meta['total_generated']:,}  seed={args.seed}  "
          f"fraction={args.fraction}  elapsed={meta['elapsed_sec']}s")
    for ly in ("L1", "L2", "L3", "L4", "L5"):
        c = meta["layer_counts"].get(ly, 0)
        print(f"  {ly}: {c:,}  ({meta['layer_elapsed_sec'].get(ly, 0)}s)")
    print(f"  councilors={meta['councilors']}  llm_targets={meta['llm_targets_count']:,}")


if __name__ == "__main__":
    main()
