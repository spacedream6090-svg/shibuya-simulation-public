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

from .observer.schema import Event

# カテゴリ別の既定営業時間 [開店時, 閉店時](24h表記。閉<=開 は翌朝までの夜間営業)。
# ここに無いカテゴリ(office/service/education/leisure/attraction/hotel/landmark 等)はゲートせず常時営業扱い。
_DEFAULT_HOURS = {
    "food": [11, 23],       # 飲食店 11:00〜23:00
    "shop": [10, 21],       # 物販 10:00〜21:00
    "nightlife": [18, 5],   # 夜遊び 18:00〜翌05:00(夜間営業)
}

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
}

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
    return cfg


def enabled(sim) -> bool:
    """商業ダイナミクス(H3)が有効か。既定 OFF=新経路を一切通さない(バイト一致)。"""
    cfg = getattr(sim, "commercecfg", None)
    return bool(cfg and cfg["enabled"])


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


# ---------------------------------------------------------------- 営業時間(時刻の純関数)
def is_open(cfg: dict, cat: str, sim_min: int) -> bool:
    """カテゴリ cat の店が sim_min の時刻に開いているか(時刻からの純関数・RNG不要・決定論)。

    hours にゲート設定の無いカテゴリ(office/service/leisure 等)は常時営業=True。夜間営業
    (open>close)は「open 以降 or 翌朝 close 未満」で開店。open==close は 24 時間営業扱い。"""
    hc = cfg["hours"].get(cat)
    if hc is None:
        return True
    o, c = hc
    if o == c:
        return True                                # 24 時間営業
    m = int(sim_min) % 1440
    if o < c:
        return o <= m < c
    return m >= o or m < c                          # 夜間営業(翌朝まで)


def is_open_poi(cfg: dict, poi: dict, sim_min: int) -> bool:
    """POI(dict)が営業中か。cat から is_open を引く(routine の行き先候補フィルタ用)。"""
    return is_open(cfg, str(poi.get("cat", "")), sim_min)


def filter_open(sim, pois: list, sim_min: int) -> list:
    """行き先候補 pois から閉店中の POI を除外する(commerce ON 時のみ)。

    OFF(または未設定)なら pois をそのまま返す=候補・以降の draw 順とも不変(バイト一致)。ON 時は
    is_open(時刻の純関数)で決定論フィルタ。R1: 除外は物理位置=co-location を変えうるが k 非依存
    (時刻・config のみ参照)=compute_matched 下の k 不変性で担保。"""
    if not enabled(sim):
        return pois
    cfg = sim.commercecfg
    return [p for p in pois if is_open_poi(cfg, p, sim_min)]


def tick_shop_state(sim, step: int, sim_min: int) -> None:
    """毎step: カテゴリ単位で開閉遷移を検知し shop_state を記録する(sparse・世界イベント agent_id=-1)。

    営業時間はカテゴリ共通の決定論スケジュールなので、遷移はカテゴリ単位で 1 日数件に収まる(全 POI
    を走査しない=軽量・sparse)。初回観測はベースライン設定のみ(ログなし)、以後は開閉が変わった step
    だけ 1 件記録する。commerce OFF なら完全 no-op(shop_state 0 件・状態も触らない=バイト一致)。決定論。"""
    if not enabled(sim):
        return
    cfg = sim.commercecfg
    state = sim._commerce_open                       # dict cat -> bool(前回の開閉状態)
    for cat in sorted(cfg["hours"]):
        now = is_open(cfg, cat, sim_min)
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
def occupancy(sim, node: str) -> int:
    """そのノードに居るアクティブな agent 数=需要の観測量(物理位置由来=k 非依存・決定論)。

    範囲外(loc=outside)・睡眠中は需要に数えない。全対全の軽い走査(呼び出しは購入時のみ=低頻度)。"""
    return sum(1 for a in sim.agents
               if a.node == node and a.loc != "outside" and not a.sleeping)


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
    occ = occupancy(sim, agent.node)
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
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                             kind="price_change", x=agent.x, y=agent.y,
                             payload={"poi": _poi_name(sim, agent.node, cat),
                                      "cat": cat, "ratio": round(coef, 3)}))
    return float(base_amount) * coef


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
