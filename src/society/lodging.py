"""宿泊・ホテル滞在(現実ギャップ 後続波 Wave L 2026-07-07。ユーザー要望)。

来街者(visitor、特に観光客)が夜に**範囲外へ退出する代わりにホテルへ宿泊**できる機構を
最小・非LLM・決定論で載せる層(実装 src/society/lodging.py)。現状 hotel カテゴリ POI は
commerce の「常時営業」素通し+視認重みでしか参照されず、宿泊・チェックイン・夜間滞在の
仕組みは皆無だった(docs/research/poi-reality-check.md §0-2)。

挙動: visitor が夜間に範囲外退出(_try_exit 系)をしようとするタイミングで、lodging ON かつ
  所持金 >= price_per_night のとき新 stream "lodging"(agent, step)で抽選。当選なら最寄りの
  hotel カテゴリ POI へ移動→到着でチェックイン(lodging_checkin + spend cat="lodging")→ホテル
  建物内で就寝(既存 sleep 機構を利用)→checkout_hour に lodging_checkout→通常活動へ。連泊は
  max_nights まで(毎晩支払い)。hotel POI が地図に無ければ機構は完全 no-op。

★ このファイルは src/society 直下(engine/cognition/actions/labeling/world の CHECKED_DIRS 外)。
  宿泊の抽選・最寄りホテル解決・連泊上限のロジックをここに閉じる。物理位置(ホテルへの移動・
  就寝)と金銭(spend cat="lodging")の適用は scheduler 配線側(_maybe_lodge / _lodging_checkin /
  _phase_lodging)が担う(health の受診=spend / commerce の on_purchase と同型の役割分担)。

R1 呼数不変: 機構は generate() を1本も足さない。抽選は新 stream "lodging"(既存 draw 順に挿入
  しない=決定論・ゴールデン保護)、最寄りホテルは POI 座標からの純関数、連泊上限は連泊数
  (物理カウンタ)からの決定論。宿泊は物理位置=対面 co-location を変え FixedLLM で ON!=OFF に
  なりうる(career G5 / crowd G4 / 健康 H1 / 商業 H3 と同型)が、機構は k・内面状態(構成概念)を
  発火判断に食わせず visitor 名簿・config・新 stream・物理位置・所持金・連泊数(k 非依存)のみ
  参照する=compute_matched 下の k 不変性(k=free==k=off の呼数一致)で担保する。

既定 OFF(enabled=false)= 抽選せず・チェックイン/アウトなし・"lodging" stream も引かない=
  イベント 0 件・乱数消費不変・プロンプト不変(ゴールデン golden_baseline_l1.json を守る)。
  新イベント種は lodging_checkin / lodging_checkout(schema.py 登録済み=編集しない)。
"""
from __future__ import annotations

DEFAULTS = {
    "enabled": False,
    "price_per_night": 12000.0,   # 1泊の宿泊費(spend cat="lodging" で支払い)
    "prob": 0.4,                  # 夜の退出前にホテル泊を選ぶ確率(所持金が足りる visitor のみ)
    "checkout_hour": 9,           # チェックアウト時刻(この時刻に活動再開)
    "max_nights": 3,              # 連泊の上限(超えたら退出/帰宅)
}

_BOOL_KEYS = ("enabled",)
_FLOAT_KEYS = ("price_per_night", "prob")
_INT_KEYS = ("checkout_hour", "max_nights")


def build_cfg(raw) -> dict:
    """conf の lodging ブロックを型強制つきで正準化(既定 OFF=現行挙動と完全同一)。

    dotlist / OmegaConf どちらでも受ける(health/commerce/diversity と同型)。すべて既定 OFF
    (enabled=false)で scheduler の _maybe_lodge / _lodging_checkin / _phase_lodging が完全 no-op
    (lodging_checkin/lodging_checkout を1件も出さず・"lodging" stream も引かない=ゴールデンを守る)。"""
    from omegaconf import OmegaConf
    if OmegaConf.is_config(raw):
        raw = OmegaConf.to_container(raw, resolve=True)
    raw = dict(raw or {})
    cfg = dict(DEFAULTS)
    for k, v in raw.items():
        if k not in DEFAULTS:
            continue
        if k in _BOOL_KEYS:
            cfg[k] = bool(v)
        elif k in _FLOAT_KEYS:
            cfg[k] = float(v)
        elif k in _INT_KEYS:
            cfg[k] = int(v)
    return cfg


def enabled(sim) -> bool:
    """宿泊・ホテル滞在(Wave L)が有効か。既定 OFF=新経路を一切通さない(バイト一致)。"""
    cfg = getattr(sim, "lodgingcfg", None)
    return bool(cfg and cfg["enabled"])


def hotels(sim) -> list:
    """hotel カテゴリ POI の一覧(決定論の安定順・sim に一度だけキャッシュ)。無ければ空=機構 no-op。"""
    cache = getattr(sim, "_lodging_hotels", None)
    if cache is None:
        cache = sorted(sim.city.pois_by_cat("hotel"),
                       key=lambda p: (str(p.get("id", "")), str(p.get("name", "")),
                                      str(p.get("node", ""))))
        sim._lodging_hotels = cache
    return cache


def nearest_hotel(sim, x: float, y: float) -> dict | None:
    """(x, y) から最も近い hotel POI(ユークリッド2乗距離。同距離は安定順=決定論)。無ければ None。"""
    best = None
    best_d = None
    for p in hotels(sim):
        hx, hy = sim.city.node_xy(p["node"])
        d = (hx - x) ** 2 + (hy - y) ** 2
        if best_d is None or d < best_d:
            best_d = d
            best = p
    return best


def want_lodge(sim, agent, step: int) -> bool:
    """夜の帰宅退出(homing)の前にホテル泊を選ぶか。前提がすべて揃ったときだけ新 stream "lodging" で抽選。

    前提: 有効・visitor・homing(=夜の帰宅退出のタイミング)・連泊が上限未満・所持金 >= price_per_night・
    地図に hotel POI がある。いずれか欠ければ乱数を一切引かず False(既定不変)。抽選は専用 stream
    "lodging"(agent, step)で prob(既存 draw 順を汚さない=決定論・ゴールデン保護)。"""
    if not enabled(sim):
        return False
    if not getattr(agent, "visitor", False):
        return False
    if not getattr(agent, "homing", False):          # 夜の帰宅退出の瞬間のみ(日中の気まぐれ退出は対象外)
        return False
    cfg = sim.lodgingcfg
    if int(getattr(agent, "lodging_nights", 0)) >= int(cfg["max_nights"]):
        return False                                 # 連泊上限に達した=退出/帰宅へ
    if float(getattr(agent, "money", 0.0)) < float(cfg["price_per_night"]):
        return False                                 # 所持金が足りない
    if not hotels(sim):
        return False                                 # 地図に hotel POI が無い=完全 no-op
    rng = sim.hub.stream("lodging", agent.id, step)  # 新 stream(既存 draw 順に影響しない)
    return bool(rng.random() < float(cfg["prob"]))
