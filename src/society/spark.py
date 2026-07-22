"""火種介入の実験条件化 = spark treatment(第53バッチ 2026-07-23。ユーザー要望)。

「立ち上げには初期介入が不可欠」という外部の助言と「介入ゼロで観測する」という本プロジェクト方針の
対立を、**ON/OFF 可能な treatment**として実験で決着させる層。介入は hardcode せず、**初期条件・環境の
差**(初期関係の束/初期資本・在庫/定期集会アンカー)としてのみ与える。**プロンプトへの台本注入は一切
しない**(行動差は環境を読んだ LLM/routine が自ら生む=ontology.py と同じ nature-like 志向)。

3メニュー(各 config で個別 ON/OFF):
  (a) 初期関係の束: sparked 相互 + 各 sparked の近傍(同建物優先→id 近傍の決定論選定)を
      friends._inject 流儀で友人 closeness/tier へ注入(起動時1回・乱数ゼロ)。
  (b) 初期資本・在庫: sparked の money 上乗せ + (商業 POI=自営/venture 保持者なら)その (node,cat) の
      在庫上乗せ(goods.inventory ON 時のみ実効。起動時1回・決定論)。
  (c) 集会アンカー: sparked 全員に共通の(poi_cat, spark.seed)純関数で1つの POI を定め、band 内の自由
      時間の行き先を bias 確率で寄せる(routine dest チェーンに中立1分岐=date/joint 流儀・新 stream "spark")。
      bias は `bias × exp(-decay × step)` で減衰=介入が初期だけ効く(CEO→Autonomous 移行のアナロジー)。

★ このファイルは src/society 直下(engine/cognition/actions/labeling/world の CHECKED_DIRS 外)。
  選抜・較正値・POI 選定ロジックはここ(と conf の spark ブロック)にのみ書く=no-fingerprint 契約に
  触れない(engine/cognition 側の配線は spark.apply 1呼び出しと spark_dest の中立な1分岐のみ)。

設計上の絶対条件(Fable 精査で確定・変更不可):
  1. 選抜は trait-blind 既定: sparked の選抜は (pid, spark.seed) の安定ハッシュ純関数=無作為。traits
     (nfc 等)を読む選抜は実装しない(k* 研究「生得か創発か」への交絡を禁止)。名簿明示指定(ids)は許可。
  2. decay の意味はメニュー別: (a)関係の束・(b)資本/在庫は t=0 の初期条件一発で減衰対象外(種が根付くか
     自体が観測対象)。spark.decay が効くのは (c)集会アンカーのバイアス強度のみ(step 経過で指数減衰)。
  3. イベント毎の spark タグは付けない(payload 汚染)。代わりに t=0 の名簿イベント1件(spark_roster
     {ids, menus, params}・agent_id=-1)を記録し、事後トレーサ(scripts/trace_spark.py)が L1 から
     「sparked 起点の現象か」を追跡する(sparked の行動+transmission 波及=語彙可視化と同じ資産の再利用)。

決定論・run.seed 非依存(比較実験の要): 選抜は (pid, spark.seed) の安定ハッシュ(hashlib=RngHub 無風=
  乱数 stream を1本も引かない。ontology._stable_uniform と同方式)。(a)(b)(c)の起動時適用も乱数ゼロ。
  在庫/資本/関係はランをまたいで同一設定なら同一(spark.seed 固定なら run.seed を変えても同一名簿)。

既定 OFF(enabled=false)= 何も選抜せず・関係台帳/資本/在庫に一切触れず・spark_anchor 属性も生やさず・
  spark_roster も出さず・"spark" stream を1本も引かない=乱数消費・イベント列・プロンプトともバイト一致
  (ゴールデン tests/data/golden_baseline_l1.json を守る)。
"""
from __future__ import annotations

import hashlib
import math

from . import friends as _friends
from . import relations as _relations
from .observer.schema import Event

# 在庫上乗せ(b)の対象となる消費カテゴリ(goods.py の在庫台帳キーと整合)。商業 POI を staff/自営で
# 持つ sparked のみが在庫を積める(office/education 等の非消費 POI は在庫を持たない=対象外)。
_CONSUMABLE = ("food", "cafe", "shop", "nightlife")


def build_cfg(raw) -> dict:
    """conf の spark ブロックを型強制つきで正準化(既定 OFF=現行挙動と完全同一)。

    dotlist / OmegaConf どちらでも受ける(friend_graph/party と同型)。menus はネストした固定キー
    (relations/capital/anchor)なので、それぞれ既定込みで正準化する(config_hash の json.dumps
    (sort_keys) 対策で全キーを文字列固定・値を素の Python 型へ)。"""
    from omegaconf import OmegaConf
    if OmegaConf.is_config(raw):
        raw = OmegaConf.to_container(raw, resolve=True)
    raw = dict(raw or {})
    menus = dict(raw.get("menus") or {})
    rel = dict(menus.get("relations") or {})
    cap = dict(menus.get("capital") or {})
    anc = dict(menus.get("anchor") or {})
    band = anc.get("band") or [1020, 1320]
    return {
        "enabled": bool(raw.get("enabled", False)),
        "seed": int(raw.get("seed", 20260723)),        # 選抜の安定ハッシュ種(run.seed 非依存)
        "n": int(raw.get("n", 5)),                      # 無作為選抜の人数(ids が空のとき)
        "ids": [str(x) for x in (raw.get("ids") or [])],  # 名簿明示指定(優先)。pid か agent.id 文字列
        "menus": {
            # (a) 初期関係の束
            "relations": {
                "enabled": bool(rel.get("enabled", True)),
                "closeness": float(rel.get("closeness", 5.0)),   # 友人 tier2 相当(閾値 5.0)
                "mutual_k": int(rel.get("mutual_k", 4)),         # 各 sparked に束ねる近傍人数
            },
            # (b) 初期資本・在庫
            "capital": {
                "enabled": bool(cap.get("enabled", True)),
                "money": float(cap.get("money", 50000.0)),       # 手持ちへの上乗せ(円)
                "stock": int(cap.get("stock", 20)),              # 商業 POI 在庫の上乗せ(単位)
            },
            # (c) 集会アンカー
            "anchor": {
                "enabled": bool(anc.get("enabled", True)),
                "poi_cat": str(anc.get("poi_cat", "leisure")),   # 集会 POI のカテゴリ
                "band": [int(band[0]), int(band[1])],            # 寄せる時間帯 [開始, 終了)(分 of day)
                "bias": float(anc.get("bias", 0.5)),             # 寄せる基礎確率/step
                "decay": float(anc.get("decay", 0.02)),          # step 経過での指数減衰係数
            },
        },
    }


# ---------------------------------------------------------------- 純関数ヘルパ
def _stable_uniform(seed: int, key: str) -> float:
    """(seed, key) から run.seed 非依存の一様値 [0,1)(hashlib=決定論・RngHub 無風)。

    ontology/friends._stable_uniform と同方式(blake2b はプロセス跨ぎで安定=別ラン/resume でも同一)。"""
    h = hashlib.blake2b(f"{int(seed)}\x1f{key}".encode("utf-8"),
                        digest_size=8).digest()
    return int.from_bytes(h, "big") / float(1 << 64)


def _pid(agent) -> str:
    """安定 persona id(プールは pool_pid=run.seed 非依存で固定、直接ランは agent.id)。friends と同一。"""
    pid = getattr(agent, "pool_pid", None)
    return str(pid) if pid is not None else str(agent.id)


def _residents(sim) -> list:
    """居住者(来街者は圏外=friend_graph/household と整合)を id 昇順で。"""
    return sorted((a for a in sim.agents if not a.visitor), key=lambda a: a.id)


def _select(sim, cfg: dict) -> list:
    """sparked 個体を決定論選抜する(trait-blind=設計絶対条件#1)。

    ids 明示があればそれを優先(pid か agent.id の文字列一致=実験者が固定名簿で条件を組む用)。無ければ
    (pid, spark.seed) の安定ハッシュ昇順で上位 n 名=無作為(traits を一切読まない=生得/創発の交絡を排除)。
    対象は居住者のみ。"""
    residents = _residents(sim)
    if cfg["ids"]:
        want = set(cfg["ids"])
        return [a for a in residents if _pid(a) in want or str(a.id) in want]
    n = int(cfg["n"])
    if n <= 0:
        return []
    ranked = sorted(residents,
                    key=lambda a: (_stable_uniform(int(cfg["seed"]), _pid(a)), a.id))
    return ranked[:n]


def _tier_for(sim, closeness: float) -> int:
    """closeness から relations tier を算定(注入する tier を closeness と整合させる)。"""
    rc = getattr(sim, "relationscfg", None) or _relations.DEFAULTS
    return _relations.tier_of(float(closeness), rc)


def _neighbors(a, residents: list, k: int) -> list:
    """各 sparked の「近傍」= 同 home_building 優先→ id 近傍で補完(決定論・乱数ゼロ)。

    束ねる相手を決定論で選ぶ(同建物=近所を優先し、足りなければ id の近い居住者で埋める)。Agent は
    dataclass(eq あり)なので集合演算は id で行う(オブジェクト比較を避ける)。"""
    if k <= 0:
        return []
    chosen: list = []
    chosen_ids: set = set()
    for b in residents:                              # 同 home_building(近所)を優先
        if b.id == a.id:
            continue
        hb = getattr(b, "home_building", "")
        if hb and hb == getattr(a, "home_building", ""):
            chosen.append(b)
            chosen_ids.add(b.id)
            if len(chosen) >= k:
                return chosen
    others = sorted((b for b in residents if b.id != a.id and b.id not in chosen_ids),
                    key=lambda b: (abs(b.id - a.id), b.id))
    for b in others:                                 # id 近傍で補完
        if len(chosen) >= k:
            break
        chosen.append(b)
    return chosen


# ---------------------------------------------------------------- (a) 初期関係の束
def _apply_relations(sim, sparked: list, menu: dict) -> int:
    """(a) sparked 相互 + 各 sparked の近傍を friends._inject 流儀で友人 closeness/tier へ注入する。

    起動時1回・乱数ゼロ・decay 対象外(種が根付くか自体が観測対象=設計絶対条件#2)。注入辺数を返す。"""
    clo = float(menu["closeness"])
    tier = _tier_for(sim, clo)
    k = int(menu["mutual_k"])
    residents = _residents(sim)
    n_edges = 0
    # sparked 相互(核=互いに友人)
    for i, a in enumerate(sparked):
        for b in sparked[i + 1:]:
            _friends._inject(a, b, tier, clo)
            _friends._inject(b, a, tier, clo)
            n_edges += 1
    # 各 sparked の近傍を友人束に(核から外へ広がる初期紐帯)
    seen_pairs: set = set()
    for a in sparked:
        for b in _neighbors(a, residents, k):
            key = (a.id, b.id) if a.id < b.id else (b.id, a.id)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            _friends._inject(a, b, tier, clo)
            _friends._inject(b, a, tier, clo)
            n_edges += 1
    return n_edges


# ---------------------------------------------------------------- (b) 初期資本・在庫
def _commercial_cat(sim, node: str) -> str | None:
    """node に消費カテゴリ(food/cafe/shop/nightlife)の POI があればその cat を返す(無ければ None)。"""
    for p in sim.city.pois_at_node(node):
        c = p.get("cat")
        if c in _CONSUMABLE:
            return c
    return None


def _seed_stock(sim, a, stock: int) -> bool:
    """(自営/venture 保持者=商業 POI を持つ sparked なら)その (node, cat) 在庫を上乗せする。

    goods.inventory ON かつ商業カテゴリの work_node を持つ sparked のみ実効(その他は no-op=money だけ効く)。
    在庫台帳 sim._goods_stock を capacity+stock(遅延初期化なら capacity 基準)へ底上げ=売る種を先に持たせる。
    決定論・乱数ゼロ。goods OFF は在庫概念自体が無いので何もしない。"""
    from . import goods as _goods
    if stock <= 0 or not _goods.enabled(sim):
        return False
    node = getattr(a, "work_node", "") or ""
    if not node:
        return False
    cat = _commercial_cat(sim, node)
    if cat is None:
        return False
    key = (str(node), str(cat))
    cur = sim._goods_stock.get(key)
    base = int(cur) if cur is not None else int(_goods._capacity(sim.goodscfg, cat))
    sim._goods_stock[key] = base + int(stock)
    return True


def _apply_capital(sim, sparked: list, menu: dict) -> int:
    """(b) sparked の money 上乗せ +(商業 POI 保持者なら)在庫上乗せ。起動時1回・決定論・decay 対象外。

    在庫を積めた sparked の人数を返す(観測用)。"""
    money = float(menu["money"])
    stock = int(menu["stock"])
    n_stocked = 0
    for a in sparked:
        if money:
            a.money = float(getattr(a, "money", 0.0)) + money
        if _seed_stock(sim, a, stock):
            n_stocked += 1
    return n_stocked


# ---------------------------------------------------------------- (c) 集会アンカー
def _anchor_poi(sim, menu: dict, seed: int) -> str | None:
    """sparked 全員に共通の集会 POI を(poi_cat, spark.seed)純関数で1つ定める(乱数ゼロ)。

    カテゴリに POI が無ければ dests から後退(joint._rendezvous_poi と同じ後退)。それも無ければ None。"""
    cat = str(menu["poi_cat"])
    u = _stable_uniform(int(seed), f"anchor\x1f{cat}")
    pool = sim.city.pois_by_cat(cat)
    if pool:
        return pool[int(u * len(pool)) % len(pool)]["node"]
    dests = sim.dests
    if dests:
        return dests[int(u * len(dests)) % len(dests)]
    return None


def _apply_anchor(sim, sparked: list, menu: dict, seed: int) -> str | None:
    """(c) 共通の集会 POI を定め、各 sparked に spark_anchor(poi/band/bias/decay)を据える。

    起動時は属性を据えるだけ=乱数ゼロ。実際の行き先バイアス抽選は spark_dest(routine 分岐・新 stream
    "spark")が run 中に行う。共通の集会 POI(全員同一)を返す(spark_roster の params 記録用)。"""
    poi = _anchor_poi(sim, menu, seed)
    band = [int(menu["band"][0]), int(menu["band"][1])]
    bias = float(menu["bias"])
    decay = float(menu["decay"])
    for a in sparked:
        a.spark_anchor = {"poi": poi, "band": band, "bias": bias, "decay": decay}
    return poi


# ---------------------------------------------------------------- 起動時1回
def apply(sim) -> None:
    """spark treatment を起動時に適用する(顔なじみ→friend_graph→party の後の中立な1呼び出し)。

    既定 OFF=完全 no-op=バイト一致。ON なら trait-blind 選抜→(a)関係/(b)資本・在庫/(c)アンカーを適用し、
    t=0 の名簿イベント spark_roster を1件記録する(イベント毎の spark タグは付けない=設計絶対条件#3)。"""
    cfg = getattr(sim, "sparkcfg", None)
    if not cfg or not cfg["enabled"]:
        return
    sparked = _select(sim, cfg)
    sim._spark_ids = {a.id for a in sparked}          # 観測(L2)・spark_dest 用の runtime 名簿
    menus = cfg["menus"]
    n_edges = 0
    n_stocked = 0
    anchor_poi = None
    if menus["relations"]["enabled"]:
        n_edges = _apply_relations(sim, sparked, menus["relations"])
    if menus["capital"]["enabled"]:
        n_stocked = _apply_capital(sim, sparked, menus["capital"])
    if menus["anchor"]["enabled"]:
        anchor_poi = _apply_anchor(sim, sparked, menus["anchor"], int(cfg["seed"]))
    # t=0 の名簿イベント1件(事後トレーサ scripts/trace_spark.py の起点)。ids=runtime agent.id・
    # pids=run.seed 非依存の安定 id。menus=各メニューの ON/OFF・params=較正値+共通アンカー POI。
    sim.logger.log(Event(
        step=0, sim_min=0, agent_id=-1, kind="spark_roster", x=0.0, y=0.0,
        payload={
            "ids": sorted(a.id for a in sparked),
            "pids": sorted(_pid(a) for a in sparked),
            "n": len(sparked),
            "menus": {k: bool(menus[k]["enabled"])
                      for k in ("relations", "capital", "anchor")},
            "params": {
                "seed": int(cfg["seed"]),
                "closeness": float(menus["relations"]["closeness"]),
                "mutual_k": int(menus["relations"]["mutual_k"]),
                "money": float(menus["capital"]["money"]),
                "stock": int(menus["capital"]["stock"]),
                "poi_cat": str(menus["anchor"]["poi_cat"]),
                "band": [int(menus["anchor"]["band"][0]),
                         int(menus["anchor"]["band"][1])],
                "bias": float(menus["anchor"]["bias"]),
                "decay": float(menus["anchor"]["decay"]),
                "anchor_poi": anchor_poi,
                "n_edges": int(n_edges),
                "n_stocked": int(n_stocked),
            },
        }))


# ---------------------------------------------------------------- 収束(自由行動の行き先)
def spark_dest(agent, sim, step: int, sim_min: int) -> str | None:
    """(c) sparked の集会 POI への行き先バイアス(band 内のみ・bias は step で指数減衰)。決定論抽選・非LLM。

    spark OFF / 非 sparked(spark_anchor 無し)/ band 外 / bias 枯渇 / 抽選外 なら乱数を一切引かず None
    (既定不変)。専用 stream "spark"(既存 draw 順を汚さない)で `bias × exp(-decay × step)` を抽選。
    現在地と同じなら None(=到着済み)。date/joint と同型の中立な1分岐(routine の elif チェーンで下位)。"""
    cfg = getattr(sim, "sparkcfg", None)
    if not cfg or not cfg["enabled"]:
        return None
    anc = getattr(agent, "spark_anchor", None)
    if not anc:
        return None
    poi = anc["poi"]
    if not poi:
        return None
    lo, hi = anc["band"]
    m = sim_min % 1440
    if not (lo <= m < hi):
        return None
    bias = float(anc["bias"]) * math.exp(-float(anc["decay"]) * float(step))
    if bias <= 0.0:
        return None
    rng = sim.hub.stream("spark", agent.id, step)
    if float(rng.random()) >= bias:
        return None
    return poi if poi != agent.node else None
