"""商業・店舗のダイナミクス(現実ギャップ 後続波 H3 2026-07-07。ユーザー要望)。

現状 POI(店・会社)は静的で常時営業・価格固定。商業の動きを最小・非LLM・決定論で載せる層。
3機構(いずれも既存 POI/economy の最小拡張。都市再開発=地図が変わる機構は本波では作らない):

  1. 店の営業時間(is_open / tick_shop_state): POI のカテゴリから開閉時刻を決定論で決める
     (時刻からの純関数=乱数不要)。閉店中のカテゴリは routine の行き先候補から除外する
     (食事/買物/夜遊び先の選択)。開閉の遷移を shop_state で記録(カテゴリ単位=1日数件・sparse・
     世界イベント agent_id=-1)。
  2. 動的価格・セール・需給(on_purchase の価格係数): その POI/時間帯の需要(在館数=観測量)に
     応じて価格係数を決定論で掛ける(economy の消費額に乗る)。混雑した店/時間帯で高く(surge)、
     閑散でセール(<1)。price_change(cat, ratio)。既定係数 1.0=不変。★係数は既存 spend の抽選の
     外で決定論的に掛ける(economy の _poi_price/spend の draw を汚さない)。
  3. 在庫・品切れ・行列(on_purchase の stock 判定): 需要が閾値超の POI で資源希少化(品切れ/行列)=
     その場での購入抑制 + 不満(factors 経由)。stock_out(poi, cat)。

★ このファイルは src/society 直下(engine/cognition/actions/labeling/world の CHECKED_DIRS 外)。
  商業ロジック(営業時間・動的価格・在庫)をここに閉じる=engine/cognition/world に構成概念名を
  書かない(no-fingerprint 契約に触れない)。品切れ→grievance は factors 層 hook(on_scarcity)経由で
  不透明な magnitude だけを渡す。

R1 呼数不変: どの機構も generate() を1本も足さない。営業時間は時刻からの純関数(RNG不要・決定論)、
  動的価格・在庫は在館数(その場の観測量=物理位置由来=k 非依存)からの決定論。閉店による行き先除外は
  物理位置=対面 co-location を変えうる(=FixedLLM で ON!=OFF になりうる=career G5 / crowd G4 / 健康 H1 /
  世帯 H2 と同型)が、機構は k・内面状態(構成概念)を発火判断に食わせず、時刻・config・在館数(観測量)
  のみ参照する=compute_matched 下の k 不変性(k=free==k=off の呼数一致)で担保する。動的価格・在庫は
  移動を変えない(消費額・購入可否のみ)=呼数に無関係。品切れ→grievance は drive(発火系)に接続しない。

既定 OFF(enabled=false)= 営業時間ゲートなし(全 POI 常時営業扱い)・価格係数なし(消費額不変)・
  品切れなし・shop_state/price_change/stock_out とも 0 件・乱数消費不変(ゴールデン
  golden_baseline_l1.json を守る)。新イベント種は shop_state / price_change / stock_out(schema.py 登録済み)。
"""
from __future__ import annotations

import math

from . import devices as devices_mod
from . import night as night_mod          # 夜間経済(第101 III-1)。night 側は commerce を
#                                           module 直下で import しない = 循環しない
from .observer.schema import Event

# カテゴリ別の既定営業時間 [開店時, 閉店時](24h表記。閉<=開 は翌朝までの夜間営業)。
# ここに無いカテゴリ(office/service/education/leisure/attraction/hotel/landmark 等)はゲートせず常時営業扱い。
_DEFAULT_HOURS = {
    "food": [11, 23],       # 飲食店 11:00〜23:00
    "shop": [10, 21],       # 物販 10:00〜21:00
    "nightlife": [18, 5],   # 夜遊び 18:00〜翌05:00(夜間営業)
}

# ---------------------------------------------------------------- CRWD 既定表(混雑不満)
# 正典 docs/plans/inventory-two-tier-plan.md §1.6 / リサーチ
# docs/research/crowding-dissatisfaction-empirics.md §6 表・§7(予期)・§8(常連補正は採用棄却)。
#
# cap_cat = 「業態別の想定同時収容」。★これは閾値の**引用ではなく設計値**である(§9: 小売文献に
#   『人/m² の混雑知覚閾値』は存在しない)。アンカーは待ち行列・立位域 LOS B(0.9 m²/人)相当で
#   収まる人数で、L=0.6 が LOS A/B 境界・L=1.4 が LOS D 帯に対応するよう置いた。
_CRWD_DEFAULT_CAP = {"food": 20, "shop": 30, "nightlife": 40, "cafe": 15, "service": 8}

# 業態表(§6 表)。l0=不満の始まり / l1=飽和 / m=上限マグニチュード。
#   quiet_l/quiet_m … 閑散罰(L<quiet_l で +quiet_m。Tse 2002 の social proof=空きすぎも負)。
#   u_lo/u_hi/u_m   … U 字の**負帯**(その区間は grievance が下がる=社会的高揚)。nightlife のみ。
# 上限は品切れ不満(stock_grievance=0.02)の 0.3-0.5 倍・既存 congestion_grievance=0.010 と同水準。
_CRWD_DEFAULT_TABLE = {
    # 物販: 課題志向で最も負(Eroglu & Machleit 1990)。単調増加。
    "shop":      {"l0": 0.6, "l1": 1.3, "m": 0.008},
    # 飲食: 単調増加 + 閑散罰。
    "food":      {"l0": 0.7, "l1": 1.4, "m": 0.010, "quiet_l": 0.3, "quiet_m": 0.004},
    # カフェ: リサーチ表に独立行は無い(飲食の下位業態)。food 行を流用し cap だけ小さく置く
    #   (Yildirim & Akalin-Baskaya 2007 の席密度=中密度選好とも矛盾しない)= 正直な設計値。
    "cafe":      {"l0": 0.7, "l1": 1.4, "m": 0.010, "quiet_l": 0.3, "quiet_m": 0.004},
    # 対人サービス: 待ち主体で per-event は最小。
    "service":   {"l0": 0.6, "l1": 1.2, "m": 0.006},
    # 夜遊び: U 字。L∈[0.3,1.1] は **負**(−0.004=社会的高揚。Noone & Mattila 2009 / Novelli 2013)、
    #   L<0.3 は閑散罰、L>1.1 で圧迫(spatial は hedonic でこそ悪化=§2 の符号反転は human 側だけ)。
    #   ★l1=1.8 は**設計値**: 表の l1 は「正に戻る点」(=1.1)であって飽和点ではないので、
    #     他業態の l1−l0 ≈ 0.6-0.7 帯を踏襲して飽和点を置いた(段差にしない)。
    "nightlife": {"l0": 1.1, "l1": 1.8, "m": 0.010,
                  "u_lo": 0.3, "u_hi": 1.1, "u_m": -0.004,
                  "quiet_l": 0.3, "quiet_m": 0.004},
}

#: 時間帯バンドの名前(band_hours の各要素に対応。最後の帯だけ日を跨ぐ)。
_CRWD_BANDS = ("morning", "midday", "evening", "night")
#: バンドの開始時(24h)。朝 05-11 / 昼 11-16 / 夕 16-21 / 夜 21-05。
_CRWD_BAND_HOURS = (5, 11, 16, 21)

# E(cat, band) = その時間帯の**平常負荷**(= L の平常水準)。予期された混雑は効果が大きく減衰する
#   (de Oliveira Santini 2021: 予期される文脈で r=−.017 / されない文脈で −.308。ただし Machleit
#   2000 は「減衰であって消去でない」)ので、L̃ で絶対負荷と超過分を w_e で混ぜる。
# ★初期値は控えめに 0.2-0.8 帯へ収めた(根拠: 二山の生活リズム=飲食はランチ/ディナー・物販は
#   夕方が最大・夜遊びは夜が本番。渋谷の実測人流に当てた較正ではない=**設計値**である)。
_CRWD_DEFAULT_EXPECTED = {
    "food":      {"morning": 0.2, "midday": 0.8, "evening": 0.7, "night": 0.4},
    "cafe":      {"morning": 0.3, "midday": 0.6, "evening": 0.5, "night": 0.3},
    "shop":      {"morning": 0.3, "midday": 0.6, "evening": 0.7, "night": 0.3},
    "service":   {"morning": 0.2, "midday": 0.5, "evening": 0.5, "night": 0.2},
    "nightlife": {"morning": 0.2, "midday": 0.2, "evening": 0.5, "night": 0.8},
}

# ---------------------------------------------------------------- PRICE-B2 既定(閉店前見切り)
# 段階は「遠い→近い」順の 2 本の**平行リスト**で持つ:
#   stage_steps … 閉店まで何 step でその段階に入るか(Δt=10 分で 18=3h / 9=1.5h)
#   stage_coefs … その段階の価格係数(実務の段階式: まず軽く下げ、閉店 1-2 時間前に半額帯)
# ★なぜ [[step, 係数], ...] の入れ子にしないか: step 次元の量は Δt に**逆比例**で変換される
#   必要があり(timeconv.TABLE の STEPS)、その変換は「数値の平坦なリスト」にしか掛からない。
#   入れ子にすると「宣言はしたが実際は変換されない」= Δt を変えた瞬間に見切りの時刻が
#   ずれる、という静かな壊れ方をする。係数は Δt 非依存(INVARIANT)なので分けるのが正しい。
_MD_DEFAULT_STAGE_STEPS = (18, 9)
_MD_DEFAULT_STAGE_COEFS = (0.8, 0.5)
#: 見切りの対象カテゴリ(当日入荷バッチの概念があるのは生鮮=飲食)。
_MD_DEFAULT_CATS = ("food",)

DEFAULTS = {
    "enabled": False,
    # ---- 営業時間(cat → (open_min, close_min)。build_cfg が時→分へ変換)----
    "hours": _DEFAULT_HOURS,
    # ---- 動的価格(需給。在館数=需要 が demand_ref を超えると係数>1=surge、下回るとセール<1)----
    "price_sensitivity": 0.15,   # 在館数 1 人差あたりの価格係数変化(0.0=恒等=不変)
    "demand_ref": 2,             # 基準需要(在館数)。これと同数で係数 1.0
    "price_min": 0.7,            # 価格係数の下限(セール)
    "price_max": 1.6,            # 価格係数の上限(surge)
    # ---- 在庫/品切れ・行列(在館数が stock_threshold 以上で品切れ/行列=購入抑制+不満)----
    "stock_threshold": 6,        # これ以上の在館で品切れ/行列(0 以下=在庫機構 無効)
    "stock_grievance": 0.02,     # 品切れ/行列に遭遇 → 不満(factors 経由。0.0=観測のみ=grievance 不変)
    # ---- PRICE-B1 事前公表の時間帯料金表(cat → [[開始分, 終了分, 係数], ...])----
    #      既定 {} = 恒等 1.0 = 消費額不変(バイト一致)。★price_change は出さない
    #      (掲示済みの料金表は「価格の**変化**」ではない=イベント洪水を作らない)。
    "price_schedule": {},
    # ---- PRICE-B2 閉店前見切り(在店店員の行動 markdown)----
    "markdown": {"enabled": False, "cats": list(_MD_DEFAULT_CATS),
                 "stages": tuple(zip(_MD_DEFAULT_STAGE_STEPS, _MD_DEFAULT_STAGE_COEFS))},
    # ---- CRWD 混雑不満(購入/受給の成立時)----
    "crowding": {"enabled": False, "w_e": 0.6, "default_cap": 20.0,
                 "cap": dict(_CRWD_DEFAULT_CAP),
                 "table": {k: dict(v) for k, v in _CRWD_DEFAULT_TABLE.items()},
                 "expected": {k: dict(v) for k, v in _CRWD_DEFAULT_EXPECTED.items()},
                 "band_hours": list(_CRWD_BAND_HOURS)},
}

# 在館数の打ち切りが効かない(= 全走査する)ことを表す番兵。demand_cap / count_at_node で使う。
_NO_CAP = 1 << 62

_BOOL_KEYS = ("enabled",)
_FLOAT_KEYS = ("price_sensitivity", "price_min", "price_max", "stock_grievance")
_INT_KEYS = ("demand_ref", "stock_threshold")


def build_cfg(raw) -> dict:
    """conf の commerce ブロックを型強制つきで正準化(既定 OFF=現行挙動と完全同一)。

    dotlist / OmegaConf どちらでも受ける(career/health/household と同型)。すべて既定 OFF
    (enabled=false)で routine の閉店除外・scheduler の shop_state / 動的価格 / 在庫が完全 no-op
    (shop_state/price_change/stock_out を1件も出さず・価格係数 1.0=消費額不変=ゴールデンを守る)。

    hours は cat → [開店時, 閉店時](24h)で受け、内部では (open_min, close_min) 分に変換して保持する。
    """
    from omegaconf import OmegaConf
    if OmegaConf.is_config(raw):
        raw = OmegaConf.to_container(raw, resolve=True)
    raw = dict(raw or {})
    hours_raw = raw.get("hours", _DEFAULT_HOURS)
    hours: dict[str, tuple[int, int]] = {}
    for cat, hc in dict(hours_raw or {}).items():
        o, c = int(hc[0]), int(hc[1])
        hours[str(cat)] = (o * 60, c * 60)         # 時 → 分(0..1440)
    cfg = dict(DEFAULTS)
    cfg["hours"] = hours
    cfg["enabled"] = bool(raw.get("enabled", False))
    for k in _FLOAT_KEYS:
        cfg[k] = float(raw.get(k, DEFAULTS[k]))
    for k in _INT_KEYS:
        cfg[k] = int(raw.get(k, DEFAULTS[k]))
    cfg["price_schedule"] = _build_price_schedule(raw.get("price_schedule"))
    cfg["markdown"] = _build_markdown(raw.get("markdown"))
    cfg["crowding"] = _build_crowding(raw.get("crowding"))
    return cfg


def _build_price_schedule(raw) -> dict:
    """B1: cat → [(開始分, 終了分, 係数), ...] を型強制つきで正準化(既定 {} = 恒等)。

    帯は **リストの並び順に最初に当たったものが勝つ**(決定論・O(帯数))。開始>終了は
    日跨ぎ(営業窓 is_open_window と同じ式で判定する=時刻判定を 1 か所に保つ)。"""
    out: dict[str, tuple] = {}
    for cat, bands in dict(raw or {}).items():
        rows = []
        for b in (bands or []):
            rows.append((int(b[0]) % 1440, int(b[1]) % 1440, float(b[2])))
        if rows:
            out[str(cat)] = tuple(rows)
    return out


def _build_markdown(raw) -> dict:
    """B2: 閉店前見切りの正準 cfg(既定 enabled=false=1 度も評価しない)。

    conf は **平行な 2 本のリスト**(stage_steps / stage_coefs)で受け、内部では
    [(閉店まで何 step, 係数), ...] を **遠い→近い順**に持つ。段階は「残り step がその閾値
    以下」で成立し、より近い(=後ろの)段階が優先される(単調に下がる=戻らない)。
    2 本の長さが食い違う conf は短いほうに合わせる(黙って捏造しない)。"""
    src = dict(raw or {})
    steps = [max(0, int(s)) for s in (src.get("stage_steps") or _MD_DEFAULT_STAGE_STEPS)]
    coefs = [float(c) for c in (src.get("stage_coefs") or _MD_DEFAULT_STAGE_COEFS)]
    cats = [str(c) for c in (src.get("cats") or _MD_DEFAULT_CATS)]
    return {"enabled": bool(src.get("enabled", False)),
            "cats": tuple(cats), "stages": tuple(zip(steps, coefs))}


def _build_crowding(raw) -> dict:
    """CRWD: 混雑不満の正準 cfg(既定 enabled=false=1 度も評価しない=バイト一致)。

    cap / table / expected は **既定表へ上書きマージ**する(conf でカテゴリを 1 つ足す/
    1 つの係数だけ変える、が dotlist でも YAML でもできる)。"""
    src = dict(raw or {})

    def _merge_rows(key, default_map):
        out = {k: dict(v) for k, v in default_map.items()}
        for k, v in dict(src.get(key) or {}).items():
            row = dict(out.get(str(k), {}))
            row.update({str(kk): float(vv) for kk, vv in dict(v or {}).items()})
            out[str(k)] = row
        return out

    cap = {str(k): float(v) for k, v in _CRWD_DEFAULT_CAP.items()}
    cap.update({str(k): float(v) for k, v in dict(src.get("cap") or {}).items()})
    bands = [int(h) for h in (src.get("band_hours") or _CRWD_BAND_HOURS)]
    return {"enabled": bool(src.get("enabled", False)),
            "w_e": float(src.get("w_e", 0.6)),
            "default_cap": float(src.get("default_cap", 20.0)),
            "cap": cap,
            "table": _merge_rows("table", _CRWD_DEFAULT_TABLE),
            "expected": _merge_rows("expected", _CRWD_DEFAULT_EXPECTED),
            "band_hours": bands}


def enabled(sim) -> bool:
    """商業ダイナミクス(H3)が有効か。既定 OFF=新経路を一切通さない(バイト一致)。"""
    cfg = getattr(sim, "commercecfg", None)
    return bool(cfg and cfg["enabled"])


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


# ---------------------------------------------------------------- 営業時間(時刻の純関数)
def is_open_window(hc, sim_min: int) -> bool:
    """営業時間窓 (open_min, close_min) が sim_min に開いているか(時刻の純関数・決定論)。

    夜間営業(open>close)は「open 以降 or 翌朝 close 未満」で開店。open==close は 24 時間営業。
    ★第101 III-1: 窓の**引き当て**(cat / subcat / 夜間の上書き)と、窓の**判定**(この式)を
      分ける。夜間経済層(night.py)は引き当てだけを差し替え、この式は 1 か所に保つ。"""
    o, c = hc
    if o == c:
        return True                                # 24 時間営業
    m = int(sim_min) % 1440
    if o < c:
        return o <= m < c
    return m >= o or m < c                          # 夜間営業(翌朝まで)


def is_open(cfg: dict, cat: str, sim_min: int) -> bool:
    """カテゴリ cat の店が sim_min の時刻に開いているか(時刻からの純関数・RNG不要・決定論)。

    hours にゲート設定の無いカテゴリ(office/service/leisure 等)は常時営業=True。夜間営業
    (open>close)は「open 以降 or 翌朝 close 未満」で開店。open==close は 24 時間営業扱い。"""
    hc = cfg["hours"].get(cat)
    if hc is None:
        return True
    return is_open_window(hc, sim_min)


def is_open_poi(cfg: dict, poi: dict, sim_min: int, ncfg: dict | None = None) -> bool:
    """POI(dict)が営業中か(routine / day_plan の行き先候補フィルタ用)。

    ncfg(夜間経済 world.night_economy の正準 cfg)を渡すと POI 単位の営業時間
    (24h コンビニ・ネカフェ等の subcat 上書き)を見る。**既定 None / 夜間 OFF では
    従来どおり cat だけを見る**(同じ表・同じ式=バイト一致)。"""
    if ncfg is not None and ncfg.get("enabled"):
        hc = night_mod.poi_hours(cfg, ncfg, poi)
        return True if hc is None else is_open_window(hc, sim_min)
    return is_open(cfg, str(poi.get("cat", "")), sim_min)


def filter_open(sim, pois: list, sim_min: int) -> list:
    """行き先候補 pois から閉店中・混雑中の POI を除外する(該当機構が ON のときだけ)。

    OFF(または未設定)なら pois をそのまま返す=候補・以降の draw 順とも不変(バイト一致)。ON 時は
    is_open(時刻の純関数)で決定論フィルタ。R1: 除外は物理位置=co-location を変えうるが k 非依存
    (時刻・config のみ参照)=compute_matched 下の k 不変性で担保。

    第84バッチ(環境フィードバック 規則3): 占有>容量で待ち行列になっているノードもここで外す。
    **新しい選択ヒューリスティックは足さない**=既存の「候補から消えたら別の候補が選ばれる」
    経路をそのまま使う(= 他 POI へ流出)。★安全弁: 除外で候補が全滅するときは除外しない
    (行き先を失って世界が固まるのを防ぐ=発散対策の上限側)。env.feedback OFF なら空集合。
    """
    from . import envfeedback as _envfb
    blocked = _envfb.blocked_nodes(sim)
    if not enabled(sim) and not blocked:
        return pois                                 # ★同一オブジェクトを返す(既存の契約)
    out = pois
    if enabled(sim):
        cfg = sim.commercecfg
        # 第101 III-1(夜間経済。既定 OFF=None 相当=従来と完全同一): POI 単位の営業時間
        # (24h コンビニ・ネカフェ等)を見るための cfg を渡すだけ。OFF は cat 判定のまま。
        ncfg = getattr(sim, "nightcfg", None)
        out = [p for p in out if is_open_poi(cfg, p, sim_min, ncfg)]
    if blocked:
        kept = [p for p in out if p.get("node") not in blocked]
        if kept:
            out = kept
    return out


def tick_shop_state(sim, step: int, sim_min: int) -> None:
    """毎step: カテゴリ単位で開閉遷移を検知し shop_state を記録する(sparse・世界イベント agent_id=-1)。

    営業時間はカテゴリ共通の決定論スケジュールなので、遷移はカテゴリ単位で 1 日数件に収まる(全 POI
    を走査しない=軽量・sparse)。初回観測はベースライン設定のみ(ログなし)、以後は開閉が変わった step
    だけ 1 件記録する。commerce OFF なら完全 no-op(shop_state 0 件・状態も触らない=バイト一致)。決定論。"""
    if not enabled(sim):
        return
    cfg = sim.commercecfg
    # 第101 III-1: 夜間の上書き(例 nightlife の窓)を反映した**実効**カテゴリ表を使う。
    # 夜間 OFF なら cfg["hours"] と同一オブジェクト = 走査順・値とも従来どおり(バイト一致)。
    hours = night_mod.cat_hours(cfg, getattr(sim, "nightcfg", None))
    state = sim._commerce_open                       # dict cat -> bool(前回の開閉状態)
    for cat in sorted(hours):
        now = is_open_window(hours[cat], sim_min)
        was = state.get(cat)
        if was is None:
            state[cat] = now                         # 初回=ベースライン(ログしない)
        elif was != now:
            state[cat] = now
            sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=-1,
                                 kind="shop_state", x=0.0, y=0.0,
                                 payload={"poi": cat,
                                          "state": "open" if now else "close"}))


# ---------------------------------------------------------------- 需要(在館数)= 観測量
def occupancy(sim, node: str, counts: dict | None = None) -> int:
    """そのノードに居るアクティブな agent 数=需要の観測量(物理位置由来=k 非依存・決定論)。

    範囲外(loc=outside)・睡眠中は需要に数えない。

    ``counts`` を渡すと ``node_counts`` が作った「ノード→人数」表を **O(1)** で引くだけになる
    (値は全走査と完全同一)。渡さないときは従来どおりその場で全走査する。★表を渡してよいのは
    「その表を作ってから引くまでの間に誰の node/loc/sleeping も動かない」と**呼び出し側が
    証明できる**場所だけ(例: tools._vc_review の審査ループ=読むだけ)。"""
    if counts is not None:
        return int(counts.get(node, 0))
    return sum(1 for a in sim.agents
               if a.node == node and a.loc != "outside" and not a.sleeping)


def node_counts(sim) -> dict:
    """全 agent を **1 回**走査して「ノード → 在場・覚醒人数」表を作る(occupancy と同じ述語)。

    ``occupancy`` を B 回呼ぶと O(B×N)(25 万体では 1 回 25 万比較)。同一時点の在館数を複数
    ノードぶん要る場所(VC 審査=開店中の全 venture)では、この表を 1 回作って引く=O(N+B)。
    値は ``occupancy(sim, node)`` と常に一致しなければならない(tests/test_commerce_occupancy.py)。"""
    counts: dict = {}
    for a in sim.agents:
        if a.loc != "outside" and not a.sleeping:
            nd = a.node
            counts[nd] = counts.get(nd, 0) + 1
    return counts


def demand_cap(cfg: dict) -> int:
    """在館数の**打ち切り点**: これ以上数えても on_purchase の結論が 1 ビットも変わらない値。

    ★なぜ要るか(レーンP A4): 購入 1 件ごとの ``occupancy`` は 25 万体の全走査で、本選では
      購入数 × 25 万 = 数十億比較になる。ところが在館数 occ の使い道は 2 つしかない:
        (1) ``is_stock_out``  … occ >= stock_threshold か(閾値との大小のみ)
        (2) ``price_coef``    … clip(1 + sens×(occ - demand_ref), price_min, price_max)
      (1) が真なら on_purchase は **その場で return** するので (2) は評価されない。よって
      stock_threshold>0 のときは「閾値に達した時点で数えるのをやめてよい」。stock_threshold<=0
      (在庫機構 無効)のときは (2) の頭打ち点まで数えれば同じ値になる。どちらも**同値変換**
      (返す係数・品切れ判定・イベントとも完全同一)で、走査を早期に打ち切るぶんだけ速い。

    戻り値: 打ち切り点(この数に達したら数え終えてよい)。0 = occ を一切使わない(数えなくてよい)。
    """
    thr = int(cfg["stock_threshold"])
    if thr > 0:
        return thr                                   # 閾値到達 = 品切れ確定(価格は評価されない)
    sens = float(cfg["price_sensitivity"])
    if sens == 0.0:
        return 0                                     # 係数は恒等 1.0 = occ を使わない
    ref = int(cfg["demand_ref"])
    lo, hi = float(cfg["price_min"]), float(cfg["price_max"])
    if not (lo <= hi):                               # 逆さまの clip 幅(min>max)/NaN は
        return _NO_CAP                               # 単調飽和しない = 打ち切らない(全走査)
    target = hi if sens > 0 else lo                  # 単調なので片側で頭打ちになる
    try:
        c = ref + max(0, int(math.ceil((target - 1.0) / sens)))
        for _ in range(64):                          # 浮動小数の丸めを安全側へ 1〜2 段ずらす
            v = 1.0 + sens * (c - ref)
            if (sens > 0 and v >= target) or (sens < 0 and v <= target):
                return c if c > 0 else 0
            c += 1
    except (OverflowError, ValueError):              # 病的な config は打ち切らない(全走査)
        return _NO_CAP
    return _NO_CAP                                   # 収束しない config も全走査(安全側)


def count_at_node(sim, node: str, cap: int) -> int:
    """在館数を **cap で打ち切って**数える(= min(真の在館数, cap))。決定論(id 昇順)。

    cap は ``demand_cap`` が出した「これ以上は結論が変わらない」点。cap 未満で走査が終わった
    ときは真の在館数そのもの(=``occupancy`` と同値)を返す。"""
    if cap <= 0:
        return 0
    if cap >= _NO_CAP:
        return occupancy(sim, node)
    n = 0
    for a in sim.agents:
        if a.node == node and a.loc != "outside" and not a.sleeping:
            n += 1
            if n >= cap:
                return n
    return n


# ---------------------------------------------------------------- 動的価格・在庫(購入時)
def price_coef(cfg: dict, occ: int) -> float:
    """在館数(需要)に応じた価格係数(決定論)。demand_ref 超で surge(>1)、未満でセール(<1)。

    price_sensitivity=0.0 なら常に 1.0(恒等=消費額不変)。clip[price_min, price_max]。"""
    sens = float(cfg["price_sensitivity"])
    if sens == 0.0:
        return 1.0
    coef = 1.0 + sens * (int(occ) - int(cfg["demand_ref"]))
    return _clip(coef, float(cfg["price_min"]), float(cfg["price_max"]))


def is_stock_out(cfg: dict, occ: int) -> bool:
    """在館数(需要)が閾値以上=品切れ/行列か。stock_threshold<=0 で在庫機構 無効(常に False)。"""
    thr = int(cfg["stock_threshold"])
    return thr > 0 and int(occ) >= thr


def _poi_name(sim, node: str, cat: str) -> str | None:
    for p in sim.city.pois_at_node(node):
        if p.get("cat") == cat:
            return p.get("name")
    return None


def on_purchase(sim, agent, cat: str, base_amount: float, step: int,
                sim_min: int) -> float | None:
    """消費(食事/買物/夜遊び)に商業ダイナミクスを適用する(commerce ON の呼び出し側のみが呼ぶ)。

    戻り値: 実際に課金する金額(base_amount × 価格係数)。品切れ/行列で購入抑制する場合は None
    (呼び出し側は spend を出さない)。副作用: 品切れ→stock_out + 不満(factors 経由)、価格変動→
    price_change を記録する。RNG は一切引かない=決定論・既存 draw 順を汚さない。

    R9/no-fingerprint: 品切れ→grievance は factors 層 hook(on_scarcity)へ**不透明な magnitude**
    (stock_grievance)だけを渡す。magnitude=0.0 なら _bump が state を触らず記録もしない=grievance 不変
    (stock_out イベントのみ)。grievance は drive(発火系)には接続しない(R1: 呼数を1本も動かさない)。"""
    cfg = sim.commercecfg
    # 在館数は「品切れ判定」と「価格係数」にしか使わない。どちらも頭打ちがあるので、結論が
    # 変わらなくなる点(demand_cap)で走査を打ち切る=**同値のまま**購入 1 件あたりの比較を減らす
    # (既定 conf では cap=6 = stock_threshold)。cap に届かなければ真の在館数そのもの。
    occ = count_at_node(sim, agent.node, demand_cap(cfg))
    if is_stock_out(cfg, occ):                        # 品切れ/行列 → 購入抑制 + 不満
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                             kind="stock_out", x=agent.x, y=agent.y,
                             payload={"poi": _poi_name(sim, agent.node, cat),
                                      "cat": cat}))
        mag = float(cfg["stock_grievance"])
        if mag != 0.0:
            from .factors import update as factor_update
            factor_update.on_scarcity(agent, mag, step=step, sim_min=sim_min,
                                      logger=sim.logger)
        agent.remember("店が品切れ(行列)で買えなかった")
        return None
    coef = price_coef(cfg, occ)
    if coef != 1.0:                                   # 動的価格(surge/セール)を記録
        # IF-F W3(observer.causality ON のときだけ): 値付けを決めたのは**店の側**なので
        # device_id=commerce:pricing を刻む。agent_id は買おうとしている客(= 患者)である。
        # ★**窓(cause_scope)ではなく per-emit の刻印**にしてある: この関数の外側で
        #   呼び出し側が出す spend は「客が選んだ消費」= cause_type=agent であり、窓を
        #   開けると客の行為に店の id が付く。per-emit なら price_change の 1 行だけに閉じる
        #   (仮に窓を開けても devices.stamp / logger が DEVICE_STAMPABLE で弾くが、
        #    二重の防御に頼らず**刻む範囲そのもの**を 1 行にしておく)。
        devices_mod.log_device(
            sim, Event(step=step, sim_min=sim_min, agent_id=agent.id,
                       kind="price_change", x=agent.x, y=agent.y,
                       payload={"poi": _poi_name(sim, agent.node, cat),
                                "cat": cat, "ratio": round(coef, 3)}),
            devices_mod.DEV_COMMERCE_PRICING)
    return float(base_amount) * coef


# =========================================================================== #
# CRWD: 混雑不満(購入/受給の**成立時**)。第147 / 正典 §1.6
# --------------------------------------------------------------------------- #
# ユーザー仮説「混雑していたが買えた/食べられた場合にも不満は溜まる」を、リサーチ
# (docs/research/crowding-dissatisfaction-empirics.md)の修正 3 点つきで実装する層:
#   ① 線形無限伸長でなく**閾値つき飽和**(ramp)。感情尺度は端で頭打ちになる。
#   ② 業態で符号が割れる(nightlife の中密度帯は快=U 字。human 側だけで実証済み)。
#   ③ **予期**された混雑は効果が大きく減衰する(時間帯平常値 E からの超過で近似)。
# 採らなかったもの(見送りでなく**採用棄却**。§8): 常連補正・c_choice(自発来店割引)。
#   「常連は混雑に寛容」は文献的に支持されず、観測上のそれは displacement の生存者バイアスで
#   説明できる。ABM ではその交絡を**係数でなく行動の結果**として自然発生させるのが正しい
#   (不満の高い個体が来店時間/店を変える → 残存客の平均不満が下がる)。
#   待ち成分も入れない(待ち行列モデルが世界に無い=無いものを係数で騙らない)。
#
# 費用: 乱数ゼロ・LLM 呼数不変・k 非依存・drive(発火系)に接続しない(R1)。在館数は
#   **毎 step 冒頭に 1 回だけ**作る表(scheduler が sim._crowd_counts へ置く)を O(1) で引く。
#   ★購入ごとの全走査(count_at_node)は**復活させない**(250k で数十億比較の再発防止)。
# =========================================================================== #
#: 飽和した混雑の記憶(定型文。機構語・実験条件語・因子名を 1 文字も含まない)。
_CROWD_TEXT = "店がとても混んでいた"


def crowding_on(sim) -> bool:
    """混雑不満(CRWD)が有効か。既定 OFF=新経路を一切通さない(バイト一致)。

    ★commerce.enabled とは**独立**に効く(inventory と同じ規約)。混雑不満は営業時間ゲートや
    動的価格の有無と関係のない別機構なので、片方だけ ON にできるほうが対照実験しやすい。"""
    cfg = getattr(sim, "commercecfg", None)
    return bool(cfg and cfg["crowding"]["enabled"])


def step_counts(sim) -> dict | None:
    """CRWD 用に **1 step につき 1 回だけ**作る「ノード→在館・覚醒人数」表(OFF は None)。

    述語は ``occupancy`` と完全に同じ(loc!=outside かつ not sleeping)。scheduler が step の
    冒頭(位置が動く前)に 1 回だけ呼び、購入点はその表を O(1) で引く。OFF では 1 走査も
    作らない=バイト一致・追加コストゼロ。"""
    return node_counts(sim) if crowding_on(sim) else None


def ramp(x: float, l0: float, l1: float) -> float:
    """閾値つき飽和 ramp(x; L0, L1) = clamp((x−L0)/(L1−L0), 0, 1)(純関数)。

    L1<=L0(退化した幅)のときは段差(x>=L1 で 1.0)= 0 除算を作らない安全側。"""
    if l1 <= l0:
        return 1.0 if float(x) >= l1 else 0.0
    return _clip((float(x) - float(l0)) / (float(l1) - float(l0)), 0.0, 1.0)


def crowding_band(cfg: dict, sim_min: int) -> str:
    """時刻 → 平常負荷 E を引く時間帯バンド(朝/昼/夕/夜。時刻の純関数・決定論)。"""
    hs = cfg["crowding"]["band_hours"]
    if not hs:
        return _CRWD_BANDS[-1]
    h = (int(sim_min) % 1440) // 60
    if h < int(hs[0]) or h >= int(hs[-1]):
        return _CRWD_BANDS[-1]                       # 最後の帯(夜)だけ日を跨ぐ
    for i in range(len(hs) - 1, -1, -1):
        if h >= int(hs[i]):
            return _CRWD_BANDS[min(i, len(_CRWD_BANDS) - 1)]
    return _CRWD_BANDS[-1]


def crowding_load(cfg: dict, cat: str, occ: int, sim_min: int) -> float:
    """予期を織り込んだ負荷 L̃ =(1−w_e)·L + w_e·max(0, L−E(cat,band))(純関数)。

    L = 在館数 / cap_cat。w_e=0.0 なら純粋な絶対負荷 L そのもの(= 予期の減衰を切る)。
    E は「その時間帯なら普通これくらい混んでいる」水準で、ピーク帯ほど罰が軽くなる。"""
    cr = cfg["crowding"]
    cap = float(cr["cap"].get(str(cat), cr["default_cap"]))
    if cap <= 0.0:
        return 0.0
    load = float(int(occ)) / cap
    e = float(dict(cr["expected"].get(str(cat), {})).get(crowding_band(cfg, sim_min), 0.0))
    w = float(cr["w_e"])
    return (1.0 - w) * load + w * max(0.0, load - e)


def crowding_magnitude(cfg: dict, cat: str, occ: int, sim_min: int) -> float:
    """Δgrievance = m_cat · ramp(L̃; L0, L1)(業態別。純関数・決定論・乱数ゼロ)。

    分岐は 3 本で、上から順に排他:
      ① 閑散罰   L̃ < quiet_l → +quiet_m(food/nightlife のみ。Tse 2002: 空きすぎも負)
      ② U 字の負帯 u_lo ≤ L̃ ≤ u_hi → u_m(**負**=社会的高揚。nightlife のみ)
      ③ 単調増加 m · ramp(L̃; l0, l1)
    表に無いカテゴリは 0.0(= その業態には混雑不満を入れない、という宣言)。"""
    row = cfg["crowding"]["table"].get(str(cat))
    if not row:
        return 0.0
    x = crowding_load(cfg, cat, occ, sim_min)
    q_l = float(row.get("quiet_l", 0.0))
    if q_l > 0.0 and x < q_l:
        return float(row.get("quiet_m", 0.0))
    u_m = float(row.get("u_m", 0.0))
    if u_m != 0.0 and float(row.get("u_lo", 0.0)) <= x <= float(row.get("u_hi", 0.0)):
        return u_m
    return float(row.get("m", 0.0)) * ramp(x, float(row.get("l0", 0.0)),
                                           float(row.get("l1", 1.0)))


def crowding_saturated(cfg: dict, cat: str, occ: int, sim_min: int) -> bool:
    """L̃ ≥ L1(= 飽和 = 印象に残る混雑)か。記憶を 1 行残すかの判定に使う(純関数)。"""
    row = cfg["crowding"]["table"].get(str(cat))
    if not row:
        return False
    return crowding_load(cfg, cat, occ, sim_min) >= float(row.get("l1", 1.0))


def apply_crowding(sim, agent, cat: str, step: int, sim_min: int) -> float:
    """購入/受給が**成立した**その場の混み具合を不満へ写す(CRWD の単一作用点)。

    既定 OFF は即 0.0(grievance も記憶も 1 バイトも動かない=バイト一致)。ON では
    step 冒頭の在館数表を O(1) で引き、業態別の式で**不透明な magnitude** を作って
    factors 層 hook(on_store_crowding)へ渡す(R9: この関数の外へ因子名は出ない)。
    飽和帯(L̃≥L1)でだけ記憶を 1 行残す(nightlife の**負帯では残さない**=快い混雑を
    「混んでいた」と嘆かせない)。drive(発火系)には接続しない=R1: 呼数を 1 本も動かさない。

    ★在館数は「この step の**開始時点**でその店に居た人数」であって、いま到着した本人は
      含まれない(表は位置が動く前に 1 回だけ作る)。cap が 8-40 人の帯なので影響は小さく、
      「店に居た人数を見て混んでいると感じる」という意味づけとしても自然である。"""
    if not crowding_on(sim):
        return 0.0
    counts = getattr(sim, "_crowd_counts", None)
    if counts is None:                                # 表が無い step(OFF→ON の途中等)は何もしない
        return 0.0
    cfg = sim.commercecfg
    occ = int(counts.get(agent.node, 0))
    mag = crowding_magnitude(cfg, cat, occ, sim_min)
    delta = 0.0
    if mag != 0.0:
        from .factors import update as factor_update
        delta = factor_update.on_store_crowding(agent, mag, step=step,
                                                sim_min=sim_min, logger=sim.logger)
    if mag > 0.0 and crowding_saturated(cfg, cat, occ, sim_min):
        agent.remember(_CROWD_TEXT)
    return delta


# =========================================================================== #
# PRICE-B: 時間帯価格(B1)+ 閉店前見切り(B2)。第147 / 正典 §1.5
# --------------------------------------------------------------------------- #
# ユーザー指摘「実店舗の価格は仕入れ値+利益で固定・リアルタイム変動はめったにない」を受けた
# 価格の現実整合。現実に価格が動く 2 経路だけを、現実と同じ**型**で入れる:
#   B1 事前公表の固定料金表(ランチ/ディナー・ハッピーアワー)= 時刻の純関数。
#      ★price_change イベントは**出さない**。掲示済みの料金表どおりに払うのは「価格の変化」
#        ではないし、購入 1 件ごとに 1 件出せば L1 のイベント洪水が種目を替えて再発する。
#   B2 閉店前見切り(値引きシール)= **在店店員の行動**(markdown。L1 に載る当人の行為)。
# どちらも決定論・乱数ゼロ・LLM 呼数不変・k 非依存。既定(空 / OFF)は恒等 1.0=バイト一致。
# =========================================================================== #
def price_schedule_coef(cfg: dict, cat: str, sim_min: int) -> float:
    """B1: 事前公表の時間帯料金表から係数を引く(時刻の純関数・O(帯数)・決定論)。

    帯はリストの並び順に**最初に当たったものが勝つ**。既定 {} は常に 1.0(恒等)。"""
    bands = cfg["price_schedule"].get(str(cat))
    if not bands:
        return 1.0
    for (s, e, coef) in bands:
        if is_open_window((int(s), int(e)), sim_min):
            return float(coef)
    return 1.0


def markdown_on(sim) -> bool:
    """閉店前見切り(B2)が有効か。既定 OFF=新経路を一切通さない(バイト一致)。"""
    cfg = getattr(sim, "commercecfg", None)
    return bool(cfg and cfg["markdown"]["enabled"])


def markdown_window_open(sim, sim_min: int) -> bool:
    """この step に見切りが起きうるか(どれかの業態が閉店前の窓に入っているか)。時刻の純関数。

    ★安い先読み: 見切りが起きうるのは 1 日のうち数十 step だけなので、これを先に見れば
    「在勤者索引を組む O(N) の走査」を窓の外では 1 度も走らせずに済む(INV-B が同じ step で
    別の用事で組んでいるときは、そちらの都合で組まれた索引に相乗りするだけ)。"""
    cfg = getattr(sim, "commercecfg", None)
    if not (cfg and cfg["markdown"]["enabled"]):
        return False
    smin, hrs = _step_minutes(sim), _md_hours(sim)
    return any(markdown_stage_now(cfg, cat, sim_min, smin, hrs) > 0
               for cat in cfg["markdown"]["cats"])


def markdown_stage_now(cfg: dict, cat: str, sim_min: int, step_minutes: int,
                       hours: dict | None = None) -> int:
    """いまが閉店の何段階前か(0=まだ / 1=1段階目 / 2=2段階目…)。時刻の純関数・決定論。

    閉店時刻は既存の営業時間表(commerce.hours)から導出する(見切り専用の時刻表を作らない)。
    閉店中・24 時間営業・時刻表の無いカテゴリは 0(= 見切らない)。

    ``hours`` を渡すと **カテゴリ単位の実効表**(夜間経済の上書き込み)を使う。
    ★正直な限界: 判定は**カテゴリ単位**であり、POI 単位の上書き(24h コンビニ等の subcat)は
      見ない。見切りの対象は既定で飲食 1 業態なので実害は小さいが、深夜営業の個店は
      カテゴリの閉店時刻で値札を替えることになる(本選後の較正候補)。"""
    hc = (cfg["hours"] if hours is None else hours).get(str(cat))
    if hc is None:
        return 0
    o, c = hc
    if o == c:
        return 0                                     # 24 時間営業=閉店が無い=見切らない
    if not is_open_window(hc, sim_min):
        return 0                                     # 閉店中は値札を替えない
    left_min = (int(c) - (int(sim_min) % 1440)) % 1440
    left_steps = float(left_min) / float(max(1, int(step_minutes)))
    stage = 0
    for i, (n_steps, _coef) in enumerate(cfg["markdown"]["stages"]):
        if left_steps <= float(n_steps):
            stage = i + 1                            # 遠い→近い順なので後ろが勝つ
    return stage


def markdown_coef(sim, node: str, cat: str, sim_min: int) -> float:
    """その (node, cat) にいま効いている見切り係数(既定 1.0)。

    段階は当日の店員の行動(markdown)が刻んだ ``sim._md_stage`` から引く。翌朝は日鍵で
    自動的にリセットされ、閉店した時点でも 1.0 へ戻る(= 見切りは営業中の売り切り施策)。"""
    cfg = getattr(sim, "commercecfg", None)
    if not (cfg and cfg["markdown"]["enabled"]):
        return 1.0
    state = getattr(sim, "_md_stage", None)
    if not state:
        return 1.0
    rec = state.get((str(node), str(cat)))
    if rec is None or int(rec[0]) != int(sim_min) // 1440:
        return 1.0                                   # 別の日に刻んだ段階=翌朝リセット
    if markdown_stage_now(cfg, cat, sim_min, _step_minutes(sim), _md_hours(sim)) <= 0:
        return 1.0                                   # 閉店した=見切り終了
    stages = cfg["markdown"]["stages"]
    if not stages:                                   # 段階表が空の conf = 見切らない
        return 1.0
    idx = max(1, min(int(rec[1]), len(stages))) - 1
    return float(stages[idx][1])


def price_multiplier(sim, cat: str, node: str, sim_min: int) -> float:
    """B1 × B2 の合成係数(**この順序で乗算**する=テストが順序を固定している)。

    どちらも既定(price_schedule={} / markdown OFF)では 1.0 を返すので合成も 1.0=恒等。"""
    return (price_schedule_coef(sim.commercecfg, cat, sim_min)
            * markdown_coef(sim, node, cat, sim_min))


def apply_price(sim, amount: float, cat: str, node: str, sim_min: int) -> float:
    """消費額へ B1×B2 の係数を乗せる(購入の価格決定点から呼ぶ薄い seam)。

    係数が 1.0 のときは **受け取った値をそのまま返す**(浮動小数の乗算すら通さない)=
    既定 OFF/空では 1 ビットも変わらない。"""
    cfg = getattr(sim, "commercecfg", None)
    if cfg is None or amount is None:
        return amount
    coef = price_multiplier(sim, cat, node, sim_min)
    return amount if coef == 1.0 else float(amount) * coef


def _step_minutes(sim) -> int:
    clock = getattr(sim, "clock", None)
    return int(getattr(clock, "step_minutes", 10) or 10)


def _md_hours(sim) -> dict:
    """見切りが読むカテゴリ単位の**実効**営業時間表(夜間経済の上書きを含む)。

    夜間 OFF なら cfg["hours"] と同一オブジェクト=従来どおり(tick_shop_state と同じ流儀)。"""
    return night_mod.cat_hours(sim.commercecfg, getattr(sim, "nightcfg", None))


#: 見切りの記憶(定型文。機構語・実験条件語・因子名を 1 文字も含まない)。
_MARKDOWN_TEXT = "売れ残りに値引きの札を貼った"


def markdown_phase(sim, on_duty: dict, staffed=None, *, step: int,
                   sim_min: int) -> int:
    """B2 の単一作用点: 在店店員が閉店前に値札を替える(**当人の行動**。goods.staff_phase 相乗り)。

    引数 ``on_duty``(node → 在勤中の個体・id 昇順)と ``staffed``(担い手が 1 人でも
    割り当てられた職場ノード集合。None=フォールバックしない)は INV-B が既に組んだものを
    そのまま受け取る=**在勤述語を 1 つに保つ**(見切り専用の勤務概念を発明しない)。

    トリガ(すべて満たしたときだけ):
      ① いまが閉店の N step 前(段階が上がった)= 時刻の純関数
      ② その (node, cat) の棚に在庫が残っている(売り切るものが無ければ値札は替えない)
      ③ **当日の入荷バッチ**がある(厳密な消費期限は実装しない=当日バッチ概念のみ)
    1 店 1 step に markdown は **1 件**(複数カテゴリは 1 件へまとめる)= L1 の洪水を作らない。
    担い手が 1 人も居ない POI に限り agent_id=-1 + payload["unstaffed"]=true(INV-B と同一規約)。
    戻り値=この step に出した markdown の件数。乱数ゼロ・LLM 呼ゼロ・決定論。"""
    cfg = getattr(sim, "commercecfg", None)
    if not (cfg and cfg["markdown"]["enabled"]):
        return 0
    stage_by_cat = {}
    smin, hrs = _step_minutes(sim), _md_hours(sim)
    for cat in cfg["markdown"]["cats"]:
        s = markdown_stage_now(cfg, cat, sim_min, smin, hrs)
        if s > 0:
            stage_by_cat[str(cat)] = s
    if not stage_by_cat:
        return 0                                     # どの業態も窓の外=1 ノードも走査しない
    stock = getattr(sim, "_goods_stock", None) or {}
    if not stock:
        return 0
    delivered = getattr(sim, "_goods_delivered", None) or {}
    state = sim._md_stage
    day = int(sim_min) // 1440
    by_node: dict[str, list] = {}
    for key in sorted(stock):                        # 棚が実体化した (node,cat) だけ=有界
        node, cat = str(key[0]), str(key[1])
        stage = stage_by_cat.get(cat)
        if stage is None:
            continue
        if int(stock[key]) <= 0:                     # 棚が空=見切る物が無い
            continue
        if int(delivered.get(key, -1)) != day:       # 当日の入荷バッチが無い
            continue
        rec = state.get((node, cat))
        if rec is not None and int(rec[0]) == day and int(rec[1]) >= stage:
            continue                                 # もうその段階まで下げてある(単調・戻らない)
        by_node.setdefault(node, []).append((cat, stage))
    n = 0
    for node in sorted(by_node):
        crew = on_duty.get(node)
        agent = crew[0] if crew else None
        if agent is None and (staffed is None or node in staffed):
            continue                                 # 担い手は居るのに不在=誰も値札を替えない
        cats, deepest = [], 0
        for cat, stage in by_node[node]:
            state[(node, cat)] = [day, int(stage)]
            cats.append(cat)
            deepest = max(deepest, int(stage))
        stages = cfg["markdown"]["stages"]
        payload = {"poi": node, "cats": sorted(cats), "stage": int(deepest),
                   "coef": float(stages[min(deepest, len(stages)) - 1][1])}
        if agent is None:
            x, y = sim.city.node_xy(node)
            payload["unstaffed"] = True              # 黙って無人で値札を替えない(正直な標)
            sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=-1,
                                 kind="markdown", x=float(x), y=float(y),
                                 payload=payload))
        else:
            sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=int(agent.id),
                                 kind="markdown", x=float(agent.x), y=float(agent.y),
                                 payload=payload))
            agent.remember(_MARKDOWN_TEXT)
        n += 1
    return n


# ==================================================================== E-W2 VC/出資
# 第37バッチ 2026-07-19。docs/research/economy-abm-research.md §4。既存 venture(出店)への
# 出資判定。判定は観測可能な代理変数のみ = traction(累計売上)/ market(在館数=occupancy)/
# network(関係網の次数)。efficacy 等の内部構成概念は使わない(§7【要注意 B】= k 非依存・R1)。
# ここは commerce.occupancy(市場代理)と同居させる。config は economy.build_vc_cfg(economy["vc"])。
#
# ★純ロジック(ログしない・sim を触らない)。イベント記録(vc_investment)・乱数・
#   sim.vc_fund の遅延構築・出資の入金(owner.account)は tools/scheduler スチュワードが配線する。
#   出資は「起業意図(LLM の open_venture)」を評価するだけのルール主体=LLM 呼数 0 追加(R1)。


def vc_score(sales_total: float, occ: int, network_degree: int, cfg: dict) -> float:
    """出資スコア(§4、決定論・観測量のみ)。戻り値 [0,1](weights の総和=1 前提)。

    traction=venture の累計売上 sales_total / market=出店ノードの在館数 occupancy(sim, node) /
    network=owner の関係網の次数 len(agent.mem.relations)。各 ref で正規化し重み付き和。"""
    w = cfg["weights"]
    nt = _clip(float(sales_total) / (float(cfg["traction_ref"]) or 1.0), 0.0, 1.0)
    nm = _clip(float(occ) / (float(cfg["market_ref"]) or 1.0), 0.0, 1.0)
    nn = _clip(float(network_degree) / (float(cfg["network_ref"]) or 1.0), 0.0, 1.0)
    return w["traction"] * nt + w["market"] * nm + w["network"] * nn


def vc_candidates(scored: list, cfg: dict) -> list:
    """[(key, score), ...] から threshold 超を上位 n_deals_per_review 件選ぶ(決定論・同点はキー昇順)。

    key は owner_id 等の安定キー(決定論の tie-break)。閾値未満は出資しない=希少性(競争)を作る。"""
    passing = [(k, float(s)) for (k, s) in scored if float(s) >= float(cfg["threshold"])]
    passing.sort(key=lambda ks: (-ks[1], ks[0]))          # スコア降順・同点キー昇順=決定論
    return passing[:int(cfg["n_deals_per_review"])]


def vc_dividend(sale_amount: float, equity_share: float, cfg: dict) -> float:
    """出資後の売上から差し引く配当 = 売上 × 持分 × dividend_rate(§7 E-W2「以後の売上から配当」)。"""
    return float(sale_amount) * float(equity_share) * float(cfg["dividend_rate"])


class VCFund:
    """VC ファンドの会計主体(economy.Bank と同型)。scheduler/tools が遅延構築(sim.vc_fund)。

    原資 balance(枯れると出資停止=希少性)・持分 equity(owner_id→持分)・累計を持つ。ログはしない
    (vc_investment はスチュワードが記録)。中央制御でない局所の資金供給者(§4「institution/agent」)。"""

    def __init__(self, cfg: dict):
        self.cfg = dict(cfg)
        self.balance = float(cfg["fund_initial"])
        self.equity: dict[int, float] = {}                # owner_id -> 累計持分
        self.invested_total = 0.0
        self.dividends_total = 0.0
        self.equity_written_off = 0.0                     # 閉店で失効した持分の累計(Bank.write_offs と同型)

    def release(self, owner_id: int) -> float:
        """出資先が**消えた**(屋台の閉店・所有者の長期退場)ときに持分を落とす。戻り: 失効した持分。

        ★なぜ要るか(レーン甲 2026-08-13): ``equity`` には pop する経路が 1 つも無かったので、
          プール回転で街を出た所有者の持分が永久に残り (a) その owner は「出資済み」として
          再審査から外れ続け、(b) 売上ゼロ = 配当ゼロなので ``balance`` は単調減少するだけ
          になり、数日で原資が枯れて出資が止まっていた。閉店は「その持分の裏付けが世界から
          消えた」ことなので、Bank の貸倒(``write_off``)と同じ作法で損金計上して落とす。
          ★金は動かさない(出資額は既に所有者の口座へ渡っている)= 保存則に触らない。
        """
        eq = float(self.equity.pop(int(owner_id), 0.0))
        if eq > 0.0:
            # getattr 経由 = 本欄より前に pickle された checkpoint(VCFund は checkpoint に
            # 載る)から復元した実体にも安全に足せる。
            self.equity_written_off = float(
                getattr(self, "equity_written_off", 0.0)) + eq
        return eq

    def can_invest(self) -> bool:
        return self.balance >= float(self.cfg["ticket"])

    def invest(self, owner_id: int) -> float:
        """1件出資(ticket を支出・持分を加算)。戻り: 出資額。呼び出し前に can_invest を確認。"""
        ticket = float(self.cfg["ticket"])
        self.balance -= ticket
        self.invested_total += ticket
        self.equity[int(owner_id)] = (self.equity.get(int(owner_id), 0.0)
                                      + float(self.cfg["equity_share"]))
        return ticket

    def collect_dividend(self, owner_id: int, sale_amount: float) -> float:
        """出資先の売上から配当を回収(持分がある owner のみ)。戻り: 配当額(0=非出資先)。"""
        eq = self.equity.get(int(owner_id), 0.0)
        if eq <= 0.0:
            return 0.0
        d = vc_dividend(sale_amount, eq, self.cfg)
        self.balance += d
        self.dividends_total += d
        return d
