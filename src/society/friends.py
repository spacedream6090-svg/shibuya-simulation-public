"""友人グラフ生成(関係性の再現 第45バッチ S-R2。ユーザー要望 2026-07-21)。

現状の初期関係は「同建物の顔なじみ最大3人」(simulation.py の顔なじみブロック)のみ=地理的共在
だけが源泉で、homophily(年齢/職業の類似)・学校/職場の紐帯が無い。本モジュールは起動時1回、
居住者の現実的な友人ネットワークを決定論で張る:

  - homophily(McPherson 2001 "Birds of a Feather"): 類似が紐帯を生む。強さ順に 年齢>職業。
    → 年齢近接を最も強く、職業一致を次に重み付ける。
  - 共有所属: 同 work org_id(社会人後の新規友人は職場34.0%)・学生同士かつ同 school org_id
    (現在の友人は学生時代中心31.6%)。
  - 同地区近接(同一住宅建物=近所)。
  - 次数は Dunbar の入れ子層で較正: 親友~3-5(tier3=支援クリーク)・友人~10-15(tier2)・知人
    (tier1=弱い紐帯を薄く)。relations.py の tier 閾値(2.0/5.0/12.0)へ closeness を直接注入して
    その層に載せる(顔なじみ経路と同じ record_contact を使い、同一ペアの二重辺は closeness を
    加算せず直接代入で避ける)。

★ このファイルは src/society 直下(engine/cognition/actions/labeling/world の CHECKED_DIRS 外)。
  homophily/Dunbar の較正値・所属ロジックはここ(と conf)にのみ書く=no-fingerprint 契約に触れない。

決定論・run.seed 非依存(比較実験の要): 辺は (persona id ペア, friend_graph.seed) の安定ハッシュ
  (hashlib=RngHub 無風=乱数 stream を1本も引かない。ontology._stable_uniform と同方式)+ ペルソナ
  属性(age/occupation/org_id/home)の純関数。同一設定なら別ランでも同一人物ペアは同一の間柄。
  プールの pool_pid(build_persona_pool.py 生成物=run.seed 非依存で固定)がある個体はそれを、直接
  ランは agent.id を pid とする。来街者は対象外(家族・友人は圏外=household と整合)。既存の顔なじみ
  経路(simulation.py)はそのまま(friend_graph ON でも closeness を直接代入=二重辺を作らない)。

既定 OFF(enabled=false)= 何も張らず・relations 台帳に一切触れず・friend_graph_built も出さない
  =乱数消費・イベント列・プロンプトともバイト一致(ゴールデン golden_baseline_l1.json を守る)。

── ディスクキャッシュ(`world.friends_cache_dir`。既定 "" = OFF = 現行と完全同一)──────
250k の py-spy で **init の 68.4% が build_friend_graph**(40-60 分/起動)だった。中身は
居住者ごとの全相手ランキング(O(N² log N) の Python ソート)で、しかも **完全な決定論**
(`_stable_uniform` = blake2b・乱数 stream を 1 本も引かない・run.seed 非依存)。
= 同じ入力なら毎回同じグラフを作り直している。

疫学・社会 ABM の標準構成(FRED / Epihiper / synthpops)は「合成人口と接触ネットワークは
**前処理で一度生成して成果物として再利用**する」で、ランタイムで毎回張り直す設計は無い。
本モジュールもそれに倣い、**決定結果**(= 対称化後の (a.id, b.id, tier) を適用順に並べた列)を
キャッシュする:

  初回      … 従来どおり構築し、注入しながら決定列を集めて保存(tmp 書き → rename の原子的置換)。
  2 回目以降… キーが一致すればロードして**同じ順序で同じ注入を再生**する
              (closeness は保存せず、その場の cfg / tier 閾値から再計算 = 同じ float)。

キー = blake2b(形式版・friends.py の内容 hash・friend_graph 設定・tier 閾値・
  **居住者名簿ダイジェスト**・n_agents・present_cap・プール識別子)。名簿ダイジェストは
  build_friend_graph が読む属性そのもの(id / pid / 年齢 / 職業 / org_id / org_role /
  home_building を id 昇順で流し込む)なので、**キー一致 ⟹ 出力一致**が構成的に言える
  (プール抽選・seed・人数の違いは必ず名簿に現れる)。

不一致・破損・読み書き失敗は **ログ 1 行だけ出して黙って再構築**する(例外を上げない)=
キャッシュが壊れてもランは止まらないし、結果は 1 バイトも変わらない。
"""
from __future__ import annotations

import array
import hashlib
import json
import logging
import os
import struct
import sys
from pathlib import Path

from . import relations as _relations
from .observer.schema import Event

log = logging.getLogger("society.friends")

DEFAULTS = {
    "enabled": False,
    "seed": 20260722,          # 安定ハッシュ種(run.seed と独立=別ランで同一人物ペアは同一辺)
    # ---- homophily の重み(McPherson: 年齢>職業)----
    "w_age": 1.0,              # 年齢近接(最強)
    "w_occ": 0.5,              # 職業一致(次点)
    "w_same_work": 1.2,        # 共有所属: 同 work org_id(職場友人34.0%)
    "w_same_school": 1.0,      # 共有所属: 学生同士かつ同 school org_id(学生時代友人31.6%)
    "w_same_area": 0.4,        # 同地区近接(同一住宅建物=近所)
    "age_scale": 15.0,         # 年齢差の正規化スケール(歳。この差で年齢類似度が 0 になる)
    "noise": 0.3,              # スコアの決定論ノイズ幅(タイ破り・多様性=run.seed 非依存)
    # ---- Dunbar 入れ子層の次数較正(親友~3-5 / 友人~10-15 / 知人=弱い紐帯を薄く)----
    "close_min": 3,            # 親友層(tier3)の下限次数
    "close_max": 5,            # 親友層の上限次数
    "friend_min": 7,           # 友人層(tier2)の追加次数下限(親友+友人で~10-15)
    "friend_max": 12,          # 友人層の追加次数上限
    "acq_extra": 20,           # 知人層(tier1)の追加次数(弱い紐帯を薄く)
    "margin": 0.5,             # 注入 closeness の上乗せ(閾値+margin=その tier に確実に載る)
    # ---- β4 初期関係の減衰整合(第117 レーンB3・**既定 None = 上の margin と完全に同一**)----
    # 正典: docs/research/initial-relations-improvement.md §0 R2 / §1.3 / §4 機構4 (4-d) / §8.2。
    # 何が壊れていたか: 注入値は「tier 閾値 + 0.5」ちょうどなのに、`relations.decay_day` は
    #   closeness を持つ台帳を **毎日 1.0 減らす**(decay_per_day=1.0 / decay_after_days=0)。
    #   → 接触が無ければ **親友は翌日に tier2・知人は 3 日で関係消滅**、親友も 13 日で
    #   closeness 0 = **初期友人グラフは 2 週間で蒸発する過渡現象**だった(§1.3 の表)。
    #   しかも顔なじみ・icebreak は closeness を持たないので減衰の対象外 =
    #   **構造化された初期関係だけが真っ先に腐る**という逆転が起きていた。
    # 一次データ: 日本人の対面交際頻度の**中央値は「月1回〜2週に1回」**(§8.2)。
    #   「知人は 2 日会わないと消える」は現実と 1 桁以上ずれている。
    # 直し方(§4 R2): margin を層別に分け、減衰 1.0/日 の下で
    #   **親友 ≈ 2 週間 / 友人 ≈ 1 週間 / 知人 2〜3 日**は接触ゼロでも層に留まるようにする。
    # ★上限の制約: `relations.tier_of` は closeness から tier を引き直すので、注入値が
    #   **上の層の閾値に届くと昇格してしまう**(margin_friend は tier_close−tier_friend 未満、
    #   margin_acq は tier_friend−tier_acquaintance 未満でなければならない)。
    # None = このキーを書かない = 従来どおり全層 `margin` = ゴールデン L1 バイト一致。
    "margin_close": None,      # 親友(tier3)の上乗せ。既定 None → margin
    "margin_friend": None,     # 友人(tier2)の上乗せ。既定 None → margin
    "margin_acq": None,        # 知人(tier1)の上乗せ。既定 None → margin
    # ---- AGE-D: 次数の年齢曲線(第116バッチ 2026-08-15・**既定 OFF**)----
    # 正典: docs/plans/age-diversity-plan.md §4-6。
    # 現状の穴: `w_age` で「誰と繋がるか」は年齢に依るのに、**次数(何人と繋がるか)は
    #   年齢非依存**だった(15 歳も 75 歳も同じ 3-5 + 7-12 + 20)。
    # 較正(Bhattacharya, Ghosh, Monsivais, Dunbar & Kaski 2016 *R. Soc. Open Sci.* 3:160097。
    #   携帯 CDR **660 万ユーザー・年齢性別既知 320 万**): 月間 alter 数は **25 歳前後で最大の
    #   15-20 人** → 45 歳まで減少 → **45-55 は台地** → 55 以降また減少。
    #   Dunbar 2020 *Proc. R. Soc. A* 476:20200446 も「**年齢の逆 J 字関数・20-30 代ピーク**」。
    # ★重要な形の制約(4 つの独立ソースが一致): **加齢が削るのは外層(周辺・同僚)であって
    #   内核ではない**。親友は年齢不変(Bruine de Bruin: 周辺 r=−.13 / **親友 r=.01**)、
    #   内核はむしろ増える(English & Carstensen 2014: 内核 6.21→7.75 / 周辺 7.35→7.06)、
    #   家族ネットワークは規模が安定(Wrzus 2013 メタ分析 277 研究 177,635 人)。
    #   ⇒ 実装は「目標次数を年齢で縮める」ではなく **「弱紐帯(tier2 友人 / tier1 知人)の
    #   生成数だけを年齢で縮め、強紐帯(tier3 親友)は 1 人も触らない」**。
    "age_degree": False,       # ★これが AGE-D の唯一のトグル(既定 OFF=現行と完全同一)
    "age_degree_ref": 25.0,    # この年齢で倍率 1.0(= 現行較正値がピーク年齢に対応する)
    # 月間 alter 数 / ピーク(25 歳 ≈ 17.5 人)。45-55 の台地と 55 以降の再減少を折れ線で。
    "age_degree_knots": [[15.0, 0.75], [25.0, 1.00], [40.0, 0.66],
                         [45.0, 0.60], [55.0, 0.60], [70.0, 0.45], [85.0, 0.35]],
    "age_degree_min": 0.20,
    "age_degree_max": 1.20,
}

_BOOL_KEYS = ("enabled", "age_degree")
_INT_KEYS = ("seed", "close_min", "close_max", "friend_min", "friend_max", "acq_extra")
_FLOAT_KEYS = ("w_age", "w_occ", "w_same_work", "w_same_school", "w_same_area",
               "age_scale", "noise", "margin", "age_degree_ref",
               "age_degree_min", "age_degree_max")
# 層別 margin は「書かない = None = margin へ後退」を表せる必要があるので float 強制しない
_OPT_FLOAT_KEYS = ("margin_close", "margin_friend", "margin_acq")
_KNOT_KEYS = ("age_degree_knots",)
# 層別 margin の tier 対応(3=親友 / 2=友人 / 1=知人)。build_friend_graph が引く。
_MARGIN_KEY_OF_TIER = {3: "margin_close", 2: "margin_friend", 1: "margin_acq"}


def build_cfg(raw) -> dict:
    """conf の friend_graph ブロックを型強制つきで正準化(既定 OFF=現行挙動と完全同一)。"""
    from omegaconf import OmegaConf
    if OmegaConf.is_config(raw):
        raw = OmegaConf.to_container(raw, resolve=True)
    raw = dict(raw or {})
    cfg = {k: (list(v) if isinstance(v, list) else v) for k, v in DEFAULTS.items()}
    for k, v in raw.items():
        if k not in DEFAULTS:
            continue
        if k in _BOOL_KEYS:
            cfg[k] = bool(v)
        elif k in _INT_KEYS:
            cfg[k] = int(v)
        elif k in _FLOAT_KEYS:
            cfg[k] = float(v)
        elif k in _OPT_FLOAT_KEYS:
            cfg[k] = None if v is None else float(v)
        elif k in _KNOT_KEYS:
            cfg[k] = sorted(([float(x), float(y)] for x, y in (v or [])),
                            key=lambda p: p[0])
    return cfg


# ---------------------------------------------------------------- 純関数ヘルパ
def _stable_uniform(seed: int, key: str) -> float:
    """(seed, key) から run.seed 非依存の一様値 [0,1)(hashlib=決定論・RngHub 無風)。

    ontology._stable_uniform と同方式(blake2b はプロセス跨ぎで安定=別ラン/resume でも同一)。"""
    h = hashlib.blake2b(f"{int(seed)}\x1f{key}".encode("utf-8"),
                        digest_size=8).digest()
    return int.from_bytes(h, "big") / float(1 << 64)


def _pid(agent) -> str:
    """安定 persona id(プールは pool_pid=run.seed 非依存で固定、直接ランは agent.id)。"""
    pid = getattr(agent, "pool_pid", None)
    return str(pid) if pid is not None else str(agent.id)


def _pair_key(pa: str, pb: str) -> str:
    """順序に依らないペアキー(a,b と b,a で同一=対称なノイズ)。"""
    return f"{pa}\x1e{pb}" if pa <= pb else f"{pb}\x1e{pa}"


def _score(a, b, cfg: dict) -> float:
    """居住者ペアの親和スコア(homophily+共有所属+近接+決定論ノイズ)。決定論・乱数ゼロ。"""
    s = 0.0
    # homophily 年齢近接(McPherson で最強クラス)。
    aa = int(getattr(a, "age", 0) or 0)
    ab = int(getattr(b, "age", 0) or 0)
    s += float(cfg["w_age"]) * max(0.0, 1.0 - abs(aa - ab) / float(cfg["age_scale"]))
    # homophily 職業一致。
    occ = getattr(a, "occupation", "")
    if occ and occ == getattr(b, "occupation", ""):
        s += float(cfg["w_occ"])
    # 共有所属: 同 org_id(学生同士=学生時代友人 / それ以外=職場友人)。
    oa = getattr(a, "org_id", None)
    ob = getattr(b, "org_id", None)
    if oa and ob and oa == ob:
        if (getattr(a, "org_role", "") == "学生"
                and getattr(b, "org_role", "") == "学生"):
            s += float(cfg["w_same_school"])
        else:
            s += float(cfg["w_same_work"])
    # 同地区近接(同一住宅建物=近所)。
    hb = getattr(a, "home_building", "")
    if hb and hb == getattr(b, "home_building", ""):
        s += float(cfg["w_same_area"])
    # 決定論ノイズ(タイ破り・多様性)= run.seed 非依存。
    s += float(cfg["noise"]) * _stable_uniform(int(cfg["seed"]), _pair_key(_pid(a), _pid(b)))
    return s


def _degree(cfg: dict, pid: str, lo_key: str, hi_key: str, salt: str) -> int:
    """個体別の層内次数を [lo, hi] から安定ハッシュで決める(run.seed 非依存)。"""
    lo = int(cfg[lo_key])
    hi = int(cfg[hi_key])
    if hi <= lo:
        return max(0, lo)
    u = _stable_uniform(int(cfg["seed"]) + 1, f"{salt}\x1f{pid}")
    return lo + int(u * (hi - lo + 1))


def age_degree_mult(age, cfg: dict) -> float:
    """年齢 → **弱紐帯の次数**にかける倍率(AGE-D。決定論・乱数ゼロ・OFF は常に 1.0)。

    折れ線(`age_degree_knots`)を `age_degree_ref` で規格化する。範囲外は端の値で平ら
    (外挿しない = 6 歳や 82 歳へ曲線を伸ばして偽の精度を作らない)。"""
    if not cfg.get("age_degree", False):
        return 1.0
    knots = cfg.get("age_degree_knots") or []
    if not knots:
        return 1.0

    def at(x: float) -> float:
        if x <= knots[0][0]:
            return knots[0][1]
        for (x0, y0), (x1, y1) in zip(knots, knots[1:]):
            if x <= x1:
                span = x1 - x0
                return y1 if span <= 0.0 else y0 + (y1 - y0) * (x - x0) / span
        return knots[-1][1]

    ref = at(float(cfg["age_degree_ref"])) or 1.0
    m = at(float(int(age or 0))) / ref
    return min(float(cfg["age_degree_max"]), max(float(cfg["age_degree_min"]), m))


def _inject(a, b, tier: int, closeness: float) -> None:
    """a→b の関係を tier/closeness へ確定注入(直接代入=二重辺でも closeness を膨らませない)。

    record_contact で台帳エントリを確保(顔なじみ経路と同じ)し、closeness/tier を直接据える。
    tier は relations の消費側(social_lines は rel['tier'] を、joint は closeness を tier_of で
    読む)双方が整合するよう両方を据える。"""
    rel = a.mem.record_contact(b.id, b.name, 0, "友人")
    rel["closeness"] = float(closeness)
    rel["tier"] = int(tier)


# ---------------------------------------------------------------- ディスクキャッシュ
_CACHE_MAGIC = b"SBYFG1\x00\x00"      # 8 バイト固定(形式の取り違えを弾く)
_CACHE_FORMAT = 1                     # 形式版(上げるとキーが変わる=旧ファイルは自動で無視)
_CACHE_HEAD = struct.Struct("<8sII32sQ")   # magic, format, reserved, key, n_edges
_BIG_ENDIAN = sys.byteorder == "big"       # 保存は常にリトルエンディアン(可搬性)


def cache_dir_of(sim):
    """`world.friends_cache_dir` を解決する(既定 "" = None = キャッシュ OFF = 現行同一)。

    相対パスはリポジトリルート基準(既存の data 参照規約と同じ)。読めない config でも
    例外を上げず None(= OFF)へ倒す。"""
    try:
        world = sim.cfg.get("world", {}) or {}
        raw = str(world.get("friends_cache_dir", "") or "").strip()
    except Exception:                       # noqa: BLE001(旧 config 互換)
        return None
    if not raw:
        return None
    p = Path(raw)
    if not p.is_absolute():
        from .config import REPO_ROOT
        p = REPO_ROOT / p
    return p


def _self_hash() -> str:
    """friends.py の内容 hash(生成規則が 1 文字でも変われば別キーになる)。"""
    try:
        return hashlib.blake2b(Path(__file__).read_bytes(),
                               digest_size=16).hexdigest()
    except Exception:                       # noqa: BLE001(zip 配布などで読めない場合)
        return "src-unavailable"


def cache_key(sim, residents: list, cfg: dict, thr: dict) -> bytes:
    """キャッシュキー(32 バイト)。**キー一致 ⟹ build_friend_graph の出力一致**。

    載せるもの: 形式版 / friends.py の内容 hash / friend_graph 設定 / tier 閾値 /
    n_agents / present_cap / プール識別子 / **居住者名簿ダイジェスト**(生成規則が読む
    属性を id 昇順で全部)。名簿を入れてあるので、プール抽選・seed・人数の差は必ず現れる。
    """
    h = hashlib.blake2b(digest_size=32)

    def _put(label: str, value) -> None:
        h.update(label.encode("utf-8"))
        h.update(b"\x1f")
        h.update(str(value).encode("utf-8"))
        h.update(b"\x1e")

    _put("format", _CACHE_FORMAT)
    _put("src", _self_hash())
    _put("cfg", json.dumps(cfg, sort_keys=True, ensure_ascii=False,
                           default=str))
    _put("thr", json.dumps({str(k): v for k, v in sorted(thr.items())},
                           sort_keys=True))
    _put("n_agents", len(getattr(sim, "agents", ()) or ()))
    _put("present_cap", getattr(sim, "_pool_present_cap", None))
    pool = getattr(sim, "_pool", None)
    _put("pool", getattr(pool, "root", None) if pool is not None else None)
    _put("n_residents", len(residents))
    h.update(b"roster\x1f")
    for a in residents:                     # residents は id 昇順(呼び手が保証)
        h.update(("\x1f".join((
            str(a.id), _pid(a), str(getattr(a, "age", "") or ""),
            str(getattr(a, "occupation", "") or ""),
            str(getattr(a, "org_id", "") or ""),
            str(getattr(a, "org_role", "") or ""),
            str(getattr(a, "home_building", "") or ""),
        )) + "\x1e").encode("utf-8"))
    return h.digest()


def _cache_path(cache_dir: Path, key: bytes) -> Path:
    return cache_dir / f"friend_graph_{key[:12].hex()}.bin"


def load_edges(cache_dir, key: bytes):
    """キャッシュから (a_ids, b_ids, tiers) を読む。無い/壊れている/合わない = None。

    失敗しても**例外は上げない**(呼び手は黙って再構築する)。"""
    path = _cache_path(cache_dir, key)
    try:
        blob = path.read_bytes()
    except OSError:
        return None
    try:
        head = _CACHE_HEAD.size
        if len(blob) < head + 16:
            raise ValueError("truncated header")
        magic, fmt, _res, got_key, n = _CACHE_HEAD.unpack_from(blob, 0)
        if magic != _CACHE_MAGIC or fmt != _CACHE_FORMAT or got_key != key:
            raise ValueError("key/format mismatch")
        body = head + int(n) * 17          # int64 + int64 + int8
        if len(blob) != body + 16:
            raise ValueError("truncated body")
        if hashlib.blake2b(blob[:body], digest_size=16).digest() != blob[body:]:
            raise ValueError("checksum mismatch")
        n = int(n)
        a_ids = array.array("q")
        b_ids = array.array("q")
        tiers = array.array("b")
        a_ids.frombytes(blob[head:head + n * 8])
        b_ids.frombytes(blob[head + n * 8:head + n * 16])
        tiers.frombytes(blob[head + n * 16:body])
        if _BIG_ENDIAN:
            a_ids.byteswap()
            b_ids.byteswap()
        return a_ids, b_ids, tiers
    except Exception as exc:                # noqa: BLE001(壊れたキャッシュで止めない)
        log.info("friend_graph cache は使えないので再構築する(%s): %s", exc, path.name)
        return None


def save_edges(cache_dir, key: bytes, edges) -> None:
    """(a_ids, b_ids, tiers) を原子的に保存する(tmp 書き → rename)。失敗はログ 1 行。"""
    a_ids, b_ids, tiers = edges
    path = _cache_path(cache_dir, key)
    # tmp 名に pid を混ぜる: 同一設定のランを**同時に**起動しても書き込みが混ざらない
    # (rename は原子的なので、勝った方のファイルだけが残る = 内容はどちらも同一)。
    tmp = path.with_suffix(f".bin.{os.getpid()}.tmp")
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        if _BIG_ENDIAN:
            a_ids = array.array("q", a_ids)
            b_ids = array.array("q", b_ids)
            a_ids.byteswap()
            b_ids.byteswap()
        body = b"".join((
            _CACHE_HEAD.pack(_CACHE_MAGIC, _CACHE_FORMAT, 0, key, len(tiers)),
            a_ids.tobytes(), b_ids.tobytes(), tiers.tobytes()))
        with open(tmp, "wb") as f:
            f.write(body)
            f.write(hashlib.blake2b(body, digest_size=16).digest())
        os.replace(tmp, path)
    except Exception as exc:                # noqa: BLE001(保存できなくてもランは進む)
        log.info("friend_graph cache を保存できなかった(%s): %s", exc, path.name)
        try:
            tmp.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------- 起動時1回
def _desired_tiers(residents: list, cfg: dict) -> dict:
    """各居住者の相手を親和スコア降順に並べ、Dunbar 層で desired tier を割る(有向)。

    ★build_friend_graph 本体から**1 文字も変えずに**切り出した(キャッシュ有無で 2 経路に
      分かれても生成規則は 1 つしか無い)。決定論・乱数ゼロ。"""
    desired: dict = {}
    for a in residents:
        pid = _pid(a)
        n_close = _degree(cfg, pid, "close_min", "close_max", "close")
        n_friend = _degree(cfg, pid, "friend_min", "friend_max", "friend")
        n_acq = int(cfg["acq_extra"])
        # AGE-D: 年齢曲線は**弱紐帯(友人 tier2 / 知人 tier1)だけ**を伸縮させ、
        # 親友(tier3=内核)は 1 人も触らない(加齢が削るのは外層という 4 ソース一致の所見)。
        # 既定 OFF では mult が厳密に 1.0 = 下の 2 行は恒等 = バイト一致。
        mult = age_degree_mult(getattr(a, "age", 0), cfg)
        if mult != 1.0:
            n_friend = max(0, int(round(n_friend * mult)))
            n_acq = max(0, int(round(n_acq * mult)))
        ranked = sorted((b for b in residents if b.id != a.id),
                        key=lambda b: (-_score(a, b, cfg), b.id))
        for rank, b in enumerate(ranked):
            if rank < n_close:
                desired[(a.id, b.id)] = 3
            elif rank < n_close + n_friend:
                desired[(a.id, b.id)] = 2
            elif rank < n_close + n_friend + n_acq:
                desired[(a.id, b.id)] = 1
            else:
                break
    return desired


def _apply_cached(residents: list, edges, thr: dict, margin_of: dict) -> int:
    """保存された決定列を**同じ順序で**再生する(初回構築の注入と 1 バイトも変わらない)。

    ★注入の前に全件を検証する: 途中で不整合に気づいて中断すると台帳が半端に汚れるので、
      「1 件でも解決できなければ 1 件も注入せず -1 を返す(= 呼び手が再構築)」にする。
      キーに名簿ダイジェストが入っているのでここに落ちるのは hash 衝突級の異常だけ。"""
    by_id = {a.id: a for a in residents}
    a_ids, b_ids, tiers = edges
    n = len(tiers)
    if len(a_ids) != n or len(b_ids) != n:
        return -1
    for i in range(n):
        if (int(a_ids[i]) not in by_id or int(b_ids[i]) not in by_id
                or int(tiers[i]) not in margin_of):
            return -1
    for i in range(n):
        tier = int(tiers[i])
        clo = thr[tier] + margin_of[tier]
        a = by_id[int(a_ids[i])]
        b = by_id[int(b_ids[i])]
        _inject(a, b, tier, clo)
        _inject(b, a, tier, clo)
    return n


def build_friend_graph(sim) -> None:
    """居住者の友人ネットワークを起動時に決定論で張る(既定 OFF=no-op=バイト一致)。

    顔なじみブロックの直後に1呼び出しで呼ばれる。各居住者の相手を親和スコア降順に並べ、Dunbar 層
    (親友/友人/知人)で desired tier を割り(有向)、ペアの max tier で対称化して closeness/tier を
    注入する。乱数 stream を1本も引かない(全 hashlib)=既存 draw 順に無影響=run.seed 非依存。

    `world.friends_cache_dir`(既定 "" = OFF)を書くと、決定結果をディスクにキャッシュして
    2 回目以降の起動でロード + 同順再生する(モジュール docstring の「ディスクキャッシュ」)。
    OFF では下の従来経路がそのまま走る = 1 バイトも変わらない。"""
    cfg = getattr(sim, "friendcfg", None)
    if not cfg or not cfg["enabled"]:
        return
    residents = sorted((a for a in sim.agents if not a.visitor), key=lambda a: a.id)
    if len(residents) < 2:
        return
    rc = getattr(sim, "relationscfg", None) or _relations.DEFAULTS
    thr = {1: float(rc["tier_acquaintance"]), 2: float(rc["tier_friend"]),
           3: float(rc["tier_close"])}
    margin = float(cfg["margin"])
    # β4: 層別 margin(既定は 3 層とも None = margin = 従来と 1 バイトも変わらない)。
    # 減衰 1.0/日 の下で「その層が接触ゼロで何日もつか」を決める唯一の数値。
    margin_of = {tier: (margin if cfg.get(key) is None else float(cfg[key]))
                 for tier, key in _MARGIN_KEY_OF_TIER.items()}
    # ---- ディスクキャッシュ(既定 None = OFF = 以下 2 ブロックへ 1 度も入らない)----
    cache_dir = cache_dir_of(sim)
    key = None
    if cache_dir is not None:
        key = cache_key(sim, residents, cfg, thr)
        cached = load_edges(cache_dir, key)
        if cached is not None:
            n_edges = _apply_cached(residents, cached, thr, margin_of)
            if n_edges >= 0:
                _log_built(sim, residents, n_edges)
                return
            log.info("friend_graph cache の名簿が合わないので再構築する")
    # 各居住者の相手を親和スコア降順に並べ、Dunbar 層で desired tier を割る(有向)。
    desired = _desired_tiers(residents, cfg)
    # 対称化(ペアの max tier)+ closeness/tier を注入。
    collect = None if cache_dir is None else (array.array("q"), array.array("q"),
                                              array.array("b"))
    n_edges = 0
    for i, a in enumerate(residents):
        for b in residents[i + 1:]:
            tier = max(desired.get((a.id, b.id), 0), desired.get((b.id, a.id), 0))
            if tier <= 0:
                continue
            clo = thr[tier] + margin_of[tier]
            _inject(a, b, tier, clo)
            _inject(b, a, tier, clo)
            n_edges += 1
            if collect is not None:        # キャッシュ ON のときだけ決定列を並べて控える
                collect[0].append(a.id)
                collect[1].append(b.id)
                collect[2].append(tier)
    if collect is not None:
        save_edges(cache_dir, key, collect)
    _log_built(sim, residents, n_edges)


def _log_built(sim, residents: list, n_edges: int) -> None:
    """`friend_graph_built` を 1 件記録する(構築経路とキャッシュ再生経路で同一の値)。"""
    mean_deg = round(2.0 * n_edges / len(residents), 4) if residents else 0.0
    sim.logger.log(Event(step=0, sim_min=0, agent_id=-1, kind="friend_graph_built",
                         x=0.0, y=0.0,
                         payload={"n_edges": int(n_edges), "mean_degree": mean_deg}))
