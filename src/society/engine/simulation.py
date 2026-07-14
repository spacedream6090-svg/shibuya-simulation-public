"""シミュレーション本体: 構築(config→世界+エージェント)と実行ループ。"""
from __future__ import annotations

import json
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from ..agents.persona import build_agent
from ..cognition.lod import LodBudget
from ..config import REPO_ROOT, save_config
from ..labeling.labels import LabelSystem
from ..llm.cache import CachedLLM
from ..llm.mock import MockBackend
from ..observer.logger import ObserverLogger
from ..observer.provenance import ItemStore
from ..rng import RngHub
from ..world.clock import Clock
from ..world.map import CityMap
from ..world.routing import Router
from ..world.transit import Transit
from . import checkpoint, scheduler


class Simulation:
    def __init__(self, cfg: DictConfig, out_dir: Path | None = None):
        self.cfg = cfg
        name = cfg.run.name or f"seed{cfg.run.seed}"
        self.out_dir = Path(out_dir) if out_dir else REPO_ROOT / str(cfg.run.out_dir) / str(name)
        self.hub = RngHub(cfg.run.seed)
        map_path = Path(cfg.world.map)
        if not map_path.is_absolute():
            map_path = REPO_ROOT / map_path
        self.city = CityMap(map_path)
        self.router = Router(self.city)
        self.clock = Clock()
        self.logger = ObserverLogger(self.out_dir)
        self.items = ItemStore()
        self.labels = LabelSystem(self.items,
                                  adopt_threshold=cfg.labeling.adopt_threshold,
                                  mode=str(cfg.labeling.get("mode", "constrained")))
        self.budget = LodBudget(cfg.lod.max_llm_per_step)
        self.pois = self.city.pois()
        self.dests = self.city.destinations()
        transit_path = Path(str(cfg.transit.file))
        if not transit_path.is_absolute():
            transit_path = REPO_ROOT / transit_path
        gtfs_dir = cfg.transit.get("gtfs_dir")
        self.transit = Transit(transit_path, gtfs_dir=gtfs_dir or None)
        # 朝の一日計画(既定 ON = 新しい標準挙動)。ユーザー要望 2026-07-06。
        pcfg = cfg.get("planning", {}) or {}
        self.planningcfg = {
            "enabled": bool(pcfg.get("enabled", True)),
            "max_items": int(pcfg.get("max_items", 5)),
        }
        # プロンプト調整(改善 P3 第9バッチ。既定 OFF=注入なし=バイト一致)。
        prcfg = cfg.get("prompts", {}) or {}
        self.promptscfg = {
            "variety_hint": bool(prcfg.get("variety_hint", False)),
            # A4(第20バッチ検収): 内省 belief の雛形復唱対策。既定 OFF=内省プロンプト不変。
            "reflect_variety": bool(prcfg.get("reflect_variety", False)),
        }
        # 日付・カレンダー / 天気(第7バッチ 2026-07-07。すべて既定 OFF=現行挙動と完全同一)。
        # 当日の date_line/weather を run_step の日境界で確定し sim に保持(全エージェント共通・
        # k 非依存)。天気は新 stream "weather" のみ使用=既存 draw 順は不変。
        from ..world import calendar as _calendar_mod
        from .. import weather as _weather_mod
        from .. import schedule as _schedule_mod
        raw_cal = cfg.get("world", {}).get("calendar", None)
        raw_cal = (OmegaConf.to_container(raw_cal, resolve=True)
                   if OmegaConf.is_config(raw_cal) else raw_cal)
        self.calendarcfg = _calendar_mod.build_cfg(raw_cal)
        raw_wea = cfg.get("weather", None)
        raw_wea = (OmegaConf.to_container(raw_wea, resolve=True)
                   if OmegaConf.is_config(raw_wea) else raw_wea)
        self.weathercfg = _weather_mod.build_cfg(raw_wea)
        # 長期予定・スケジュール帳(第7バッチ 2026-07-07。既定 OFF=現行挙動と完全同一)。
        # 会話由来の決定論抽出(追加 LLM 呼び出しゼロ)。calendarcfg を相対日→絶対日の解決に使う。
        raw_sched = cfg.get("schedule", None)
        raw_sched = (OmegaConf.to_container(raw_sched, resolve=True)
                     if OmegaConf.is_config(raw_sched) else raw_sched)
        self.schedulecfg = _schedule_mod.build_cfg(raw_sched)
        self.today_date_line = None            # 当日の日付1行(calendar 有効時のみ非 None)
        self.today_weather = None              # 当日の天気 dict(weather 有効時のみ)
        self.today_weather_line = None         # 当日の天気1行(weather 有効時のみ)
        self._cal_day = -1                     # 日境界(日付・天気の確定)の進行管理
        self._sched_day = -1                   # 日境界(予定 GC)の進行管理
        # 文化カレンダー・群集(現実ギャップ Wave G4 2026-07-07。既定 OFF=現行挙動と完全同一)。
        # 命名年中行事(正月/ハロウィン/年末)を暦の特定日に発生させ、群集フラグ行事日には
        # 自由行動の行き先をスクランブル付近の集会ノードへ寄せる。要 world.calendar.enabled=true
        # (日付判定に暦を使う)。決定論(群集バイアスは新 stream "crowd" のみ=既存 draw 順は不変)。
        from .. import annual as _annual_mod
        raw_annual = cfg.get("annual_events", None)
        raw_annual = (OmegaConf.to_container(raw_annual, resolve=True)
                      if OmegaConf.is_config(raw_annual) else raw_annual)
        self.annualcfg = _annual_mod.build_cfg(raw_annual)
        self.today_event_line = None           # 当日の年中行事1行(annual 有効かつ当日のみ非 None)
        self.today_crowd_event = None          # 当日の群集イベント名(群集フラグ行事日のみ非 None)
        self._annual_day = -1                  # 日境界(年中行事の確定)の進行管理
        self._crowd_surge_day = -1             # crowd_surge を1日1回に制限(群集の記録済み日)
        self.crowd_node = None                 # 集会ノード(スクランブル交差点付近。ON 時のみ解決)
        if self.annualcfg["enabled"]:
            self.crowd_node = _annual_mod.gathering_node(self.city, self.dests)
        # 交通機関(タクシー既定 ON / 簡易バス既定 OFF=本番トグル)。ユーザー要望 2026-07-06。
        rcfg = cfg.get("transit_ride", {}) or {}
        taxi_c = rcfg.get("taxi", {}) or {}
        bus_c = rcfg.get("bus", {}) or {}
        self.ridecfg = {
            "taxi": {
                "enabled": bool(taxi_c.get("enabled", True)),
                "prob": float(taxi_c.get("prob", 0.15)),
                "min_dist_m": float(taxi_c.get("min_dist_m", 600)),
                "base_fare": float(taxi_c.get("base_fare", 500)),
                "per_km": float(taxi_c.get("per_km", 400)),
            },
            "bus": {
                "enabled": bool(bus_c.get("enabled", False)),
                "fare": float(bus_c.get("fare", 230)),
                "stop_radius_m": float(bus_c.get("stop_radius_m", 100)),
                "headway_steps": int(bus_c.get("headway_steps", 1)),
            },
        }
        self.buses = None
        if self.ridecfg["bus"]["enabled"]:
            from ..world.transit import BusNetwork
            self.buses = BusNetwork(
                getattr(self.transit, "bus_lines", []), self.city,
                stop_radius_m=self.ridecfg["bus"]["stop_radius_m"],
                headway_steps=self.ridecfg["bus"]["headway_steps"])
        # 背景交通(エージェント以外の通過車両)
        from ..world.traffic import TrafficFlow
        tcfg = cfg.world.get("traffic", {})
        self.traffic = TrafficFlow(
            self.city, self.router, self.hub,
            enabled=bool(tcfg.get("enabled", True)),
            cars_per_day=int(tcfg.get("cars_per_day", 30000)),
            max_log=int(tcfg.get("max_log", 120)))
        # インターネット層(SNS/ニュース/検索/DM)
        from ..net.internet import Internet
        ncfg = cfg.get("net", {})
        self.netcfg = {
            "enabled": bool(ncfg.get("enabled", True)),
            "browse_prob": float(ncfg.get("browse_prob", 0.12)),
            "news_prob": float(ncfg.get("news_prob", 0.06)),
            # #14 SNS 反応(既定で有効: 非LLM の追加挙動で対照性は壊さない)
            "like_prob": float(ncfg.get("like_prob", 0.15)),
            "reshare_prob": float(ncfg.get("reshare_prob", 0.03)),
        }
        self.net = Internet(feed_size=int(ncfg.get("feed_size", 6)),
                            posts_max=int(ncfg.get("posts_max", 0) or 0))
        # 情報環境の非対称(現実ギャップ Wave G6 2026-07-07。既定 OFF=現行挙動と完全同一)。
        # 推薦/エコーチェンバー・インフルエンサー非対称/バイラル・フェイク/炎上。決定論・非LLM
        # (推薦の意見参照は opinion.alignment 経由=不透明化)。誤情報判定のみ新 stream "info" を
        # 使用=既存 draw 順は不変。OFF=timeline は従来 TL・viral/misinfo フェーズは即 return(バイト一致)。
        from ..net import infoenv as _infoenv_mod
        raw_ie = cfg.get("info_env", None)
        raw_ie = (OmegaConf.to_container(raw_ie, resolve=True)
                  if OmegaConf.is_config(raw_ie) else raw_ie)
        self.infoenvcfg = _infoenv_mod.build_cfg(raw_ie)
        self._infoenv_watermark = 0            # バイラル/誤情報の走査済み投稿の絶対 id watermark
        self._viral_done = set()               # バイラル加重済みの元投稿 id(1投稿1回)
        # 意見力学(Friedkin-Johnsen #16)。チャネル別説得重み。既定で有効(y_external 不変)。
        ocfg = cfg.get("opinion", {}) or {}
        self.opinioncfg = {
            "enabled": bool(ocfg.get("enabled", True)),
            "w_face": float(ocfg.get("w_face", 0.20)),   # 対面
            "w_dm": float(ocfg.get("w_dm", 0.15)),       # DM
            "w_sns": float(ocfg.get("w_sns", 0.08)),     # SNS 閲覧
        }
        # 欲求駆動発火(Phase A)
        from ..cognition import drive as drive_mod
        self.drivecfg = drive_mod.build_cfg(cfg.get("drive", {}))
        self.drive_stats = {"requests": 0, "fires": 0}
        # 反射=自己モデル+出来事誘発の深い内省+無意識層(第11-12バッチ。既定 全OFF=不変)
        from ..cognition.reflection import build_reflection_cfg
        self.reflectcfg = build_reflection_cfg(cfg.get("reflection", None))
        # 実験プロトコル核(D7 対照 / D9 過正当化 / agentic pull)。既定は現状不変。
        ccfg = cfg.get("controls", {}) or {}
        self.controls_mode = str(ccfg.get("mode", "none") or "none")
        self.null_calls = int(ccfg.get("null_calls", 1))          # null 系列: 発火あたり本数
        rcfg = cfg.get("rewards", {}) or {}
        self.rewards_on = bool(rcfg.get("enabled", False))         # D9 過正当化 ablation
        self.reward_amount = float(rcfg.get("amount_per_adoption", 300))
        mcfg = cfg.get("memory", {}) or {}
        self.agentic_pull = bool(mcfg.get("agentic_pull", False))  # 内省の pull 想起
        self.relations_max = int(mcfg.get("relations_max", 0) or 0)  # B6: 関係台帳の上限(0=無制限)
        # 社会関係の質(Wave G2 2026-07-07。既定 OFF=現行挙動と完全同一)。関係の深化段階(tier)・
        # 断絶/悪化(decay)・評判(reputation)・派閥(軽量)。決定論・RNG不要(既存 draw 順は不変)。
        from .. import relations as _relations_mod
        raw_rel = cfg.get("relations", None)
        raw_rel = (OmegaConf.to_container(raw_rel, resolve=True)
                   if OmegaConf.is_config(raw_rel) else raw_rel)
        self.relationscfg = _relations_mod.build_cfg(raw_rel)
        self._rel_day = -1                     # 日境界(断絶/評判風化)の進行管理
        # 世帯・家族・恋愛(後続波 H2 2026-07-07。既定 OFF=現行挙動と完全同一)。世帯グループ化
        # (起動時1回・新 stream "household")+ 家族/同居人プロンプト文脈 + パートナー形成(日次・
        # G2 closeness から決定論)+ デート移動(新 stream "date")。決定論・LLM 非増(R1: 呼数不変は
        # compute_matched 下の k 不変性で担保)。OFF=世帯を組まず・イベント 0 件・新 stream も引かない
        # =乱数消費不変(ゴールデン golden_baseline_l1.json を守る)。
        from .. import household as _household_mod
        raw_hh = cfg.get("household", None)
        raw_hh = (OmegaConf.to_container(raw_hh, resolve=True)
                  if OmegaConf.is_config(raw_hh) else raw_hh)
        self.householdcfg = _household_mod.build_cfg(raw_hh)
        self._partner_day = -1                 # 日境界(パートナー形成)の進行管理
        # 閾値分布の seam(手続き生成経路のみ。名簿値優先=現状のまま)
        self.threshold_dist = str(cfg.get("factors", {}).get("threshold_dist",
                                                             "normal") or "normal")
        # state 更新則の magnitude(OPEN#2、B段調律。Phase D)
        from ..factors import update as factor_update
        self.mags = factor_update.build_magnitudes(cfg.get("factors", {}))
        # 感情・興味・注意ハブ(affect、既定 OFF。設計: docs/lit/neuroscience__emotion-interest-attention.md
        # §5.1/§6)。arousal スカラ1本 + salience 関数1つ。既定 OFF=現行挙動と完全同一(バイト一致)。
        from ..factors import affect as affect_mod
        self.affectcfg = affect_mod.build_cfg(cfg.get("affect", None))
        # 行動心理プラグイン(config 着脱式・既定 OFF。ユーザー方針 2026-07-06)。
        from ..factors import psych as psych_mod
        self.psychcfg = psych_mod.build_cfg(cfg.get("psych", {}))
        # Lynch: landmark_score(イメージアビリティの不透明スコア)を ON 時だけ構築(書き出しは後段)。
        self.landmark_score = None
        if self.psychcfg["lynch"]["enabled"]:
            self.landmark_score = self._build_landmark_score()
        # 経済 v0(賃金+消費+バイト+心理接続。ユーザー決定 2026-07-05)
        from ..economy import build_economy, build_career_cfg
        self.economy = build_economy(cfg.get("economy", {}))
        self._econ_day = -1                        # 日次境界(経済的逼迫)の進行管理
        self._acct_day = -1                        # 口座 E5: 暦日境界(給料日・家賃)の進行管理
        # キャリア転換(現実ギャップ Wave G5 2026-07-07。既定 OFF=現行挙動と完全同一)。失業/求職/転職
        # (organizations 再配属・収入変化・grievance)+ 起業転換(open_venture→本業化)。日境界で新 stream
        # "career" から確率判定(決定論)。k・内面状態を読まない(R1: 呼数は k 非依存)。OFF=バイト一致。
        raw_career = cfg.get("career", None)
        raw_career = (OmegaConf.to_container(raw_career, resolve=True)
                      if OmegaConf.is_config(raw_career) else raw_career)
        self.careercfg = build_career_cfg(raw_career)
        self._career_day = -1                      # 日境界(失業/求職/転職)の進行管理
        # 健康・疲労・病気・メンタル(現実ギャップ 後続波 H1 2026-07-07。既定 OFF=現行挙動と完全同一)。
        # 疲労ゲージ(内部 transient)+ 病気(新 stream "health": 欠勤/在宅/受診)+ メンタル(慢性高
        # grievance→引きこもり)。決定論・非LLM(R1: 呼数は compute_matched 下の k 不変性で担保)。OFF=バイト一致。
        from .. import health as _health_mod
        raw_health = cfg.get("health", None)
        raw_health = (OmegaConf.to_container(raw_health, resolve=True)
                      if OmegaConf.is_config(raw_health) else raw_health)
        self.healthcfg = _health_mod.build_cfg(raw_health)
        self._health_day = -1                      # 日境界(病気の発症/回復・メンタル)の進行管理
        # 商業・店舗のダイナミクス(現実ギャップ 後続波 H3 2026-07-07。既定 OFF=現行挙動と完全同一)。
        # 営業時間(時刻の純関数=閉店中は行き先から除外・shop_state)+ 動的価格(在館数=需要から係数を
        # 決定論で消費額に乗せる・price_change)+ 在庫/品切れ・行列(需要集中で購入抑制+不満・stock_out)。
        # 決定論・非LLM(R1: 閉店除外は物理位置=co-location を変えうるが k 非依存=compute_matched 下の
        # k 不変性で担保)。OFF=営業時間ゲートなし・価格係数なし・品切れなし・イベント 0 件=乱数消費不変
        # (ゴールデン golden_baseline_l1.json を守る)。品切れ→grievance は factors 層 hook(on_scarcity)経由。
        from .. import commerce as _commerce_mod
        raw_commerce = cfg.get("commerce", None)
        raw_commerce = (OmegaConf.to_container(raw_commerce, resolve=True)
                        if OmegaConf.is_config(raw_commerce) else raw_commerce)
        self.commercecfg = _commerce_mod.build_cfg(raw_commerce)
        self._commerce_open: dict = {}             # 営業時間の開閉遷移(shop_state)の進行管理
        # 都市・環境インフラのショック(現実ギャップ 後続波 H4 2026-07-07。既定 OFF=現行挙動と完全同一)。
        # 災害(台風/地震/大雪→交通麻痺=運休+外出抑制=在宅+強い grievance)+ 交通の遅延/運休(定刻ダイヤを
        # 新 stream "disaster" で乱す)+ インフラ障害(停電/通信断/断水→在宅娯楽・スマホ抑制+grievance)。
        # 決定論・非LLM(R1: 在宅/運休は物理位置・移動を変えうるが k 非依存=compute_matched 下の k 不変性で担保)。
        # 既存の scenario/transit の最小拡張(独立 phase)=摂動シナリオ shock_closure/shock_event の挙動は不変。
        # OFF=発生/遅延/障害なし・transit 運休なし・"disaster" stream も引かない=イベント 0 件・乱数消費不変
        # (ゴールデン golden_baseline_l1.json を守る)。grievance は factors 層 hook(on_disaster)経由(不透明 magnitude)。
        from .. import disaster as _disaster_mod
        raw_disaster = cfg.get("disaster", None)
        raw_disaster = (OmegaConf.to_container(raw_disaster, resolve=True)
                        if OmegaConf.is_config(raw_disaster) else raw_disaster)
        self.disastercfg = _disaster_mod.build_cfg(raw_disaster)
        self._disaster_day = -1                    # 日境界(災害/交通遅延/インフラ障害)の進行管理
        self._disaster_active = False              # 災害の発生中フラグ(交通麻痺=運休の要否判定)
        self._disaster_until_day = -1              # 災害の解除日(day-index。exclusive)
        self._disaster_kind = None                 # 発生中の災害種別(台風/地震/大雪)
        self._infra_active = False                 # インフラ障害の発生中フラグ(media/phone 抑制)
        self._infra_until_day = -1                 # インフラ障害の解除日(day-index。exclusive)
        self._infra_kind = None                    # 発生中の障害種別(停電/通信断/断水)
        self.today_disaster_line = None            # 当日の災害1行(disaster 有効かつ発生中のみ非 None)
        # 観光・多言語・犯罪・治安(現実ギャップ 後続波 H5 2026-07-07。既定 OFF=現行挙動と完全同一)。
        # 観光客(visitor の一部を回遊型に→ランドマーク回遊 tourist_visit)+ 多言語(非日本語話者=語の
        # 伝播障壁)+ 犯罪/迷惑(窃盗→被害者 money−+grievance / 迷惑→周囲 grievance、近傍警察官で抑止)+
        # 治安回避(犯罪履歴ノード=危険地帯を行き先から避ける)。決定論・非LLM(R1: 物理位置は変わりうるが
        # k 非依存=compute_matched 下の k 不変性で担保)。OFF=属性付与なし・観光回遊なし・伝播障壁なし・
        # 犯罪なし・"tourist"/"crime" stream も引かない=イベント 0 件・乱数消費不変(ゴールデンを守る)。
        # grievance は factors 層 hook(on_crime)経由(不透明 magnitude)。
        from .. import diversity as _diversity_mod
        raw_diversity = cfg.get("society_diversity", None)
        raw_diversity = (OmegaConf.to_container(raw_diversity, resolve=True)
                         if OmegaConf.is_config(raw_diversity) else raw_diversity)
        self.diversitycfg = _diversity_mod.build_cfg(raw_diversity)
        # 宿泊・ホテル滞在(現実ギャップ 後続波 Wave L 2026-07-07。既定 OFF=現行挙動と完全同一)。
        # 来街者(visitor)が夜に範囲外退出する代わりにホテルへ宿泊できる機構(移動→チェックイン→就寝→
        # checkout_hour に退館→通常活動)。決定論・非LLM(R1: 宿泊は物理位置=co-location を変えうるが
        # k 非依存=compute_matched 下の k 不変性で担保)。OFF=抽選せず・チェックイン/アウトなし・
        # "lodging" stream も引かない=イベント 0 件・乱数消費不変(ゴールデン golden_baseline_l1.json を守る)。
        # 新イベント種は lodging_checkin / lodging_checkout(schema.py 登録済み)。hotel POI が無ければ完全 no-op。
        from .. import lodging as _lodging_mod
        raw_lodging = cfg.get("lodging", None)
        raw_lodging = (OmegaConf.to_container(raw_lodging, resolve=True)
                       if OmegaConf.is_config(raw_lodging) else raw_lodging)
        self.lodgingcfg = _lodging_mod.build_cfg(raw_lodging)
        # 「世界を変える」ツール群(host_event/post_flyer/found_group/propose/open_venture)
        from ..tools import Tools, build_tools_cfg
        self.tools = Tools(build_tools_cfg(cfg.get("tools", {})))
        # 制度DSL(propose の成立で機械可読ルールを実効化。ユーザー構想 2026-07-06)。
        # fee delta の基準価格 = economy.prices + taxi/bus 初乗り(spec: taxi|bus も fee 対象)。
        from ..rules import RuleBook, build_rules_cfg
        ref_prices = dict(self.economy["prices"])
        ref_prices["taxi"] = float(self.ridecfg["taxi"]["base_fare"])
        ref_prices["bus"] = float(self.ridecfg["bus"]["fare"])
        self.rulebook = RuleBook(build_rules_cfg(cfg.get("rules", {})), ref_prices)
        # 再帰性(第9バッチ 2026-07-07): 規範・現状の監視→知覚→フィードバック→改変(廃止)の
        # 閉ループ。既定 OFF=注入なし・イベント 0 件・乱数不使用=現行挙動と完全同一。
        from ..recursion import Recursion, build_cfg as build_recursion_cfg
        self.recursioncfg = build_recursion_cfg(cfg.get("recursion", None))
        self.recursion = Recursion(self.recursioncfg)
        # シナリオイベント(何月何日に○○が発表される 等)
        self.world_events: list[dict] = []
        events_file = cfg.world.get("events_file")
        if events_file:
            events_path = Path(str(events_file))
            if not events_path.is_absolute():
                events_path = REPO_ROOT / events_path
            raw_ev = json.loads(events_path.read_text(encoding="utf-8"))
            for ev in raw_ev.get("events", []):
                h, m = str(ev.get("time", "12:00")).split(":")
                minute = int(ev.get("day", 0)) * 1440 + int(h) * 60 + int(m)
                step = (minute - 7 * 60) // 10      # シミュは day0 07:00 開始
                if step >= 0:
                    self.world_events.append({**ev, "step": step})
        # 摂動シナリオ(#8: 封鎖/イベント注入)。既定 baseline = 完全 no-op。
        from ..world.scenario import build_scenario
        self.scenario = build_scenario(cfg, self.city)
        # 範囲外への出口 = 縁のゲートウェイ + 駅(電車)。駅は選ばれやすく(重み3)
        self.exit_points = list(self.city.gateways)
        if self.city.station_node:
            self.exit_points += [self.city.station_node] * 3
        self.exit_points.sort()

        self.out_dir.mkdir(parents=True, exist_ok=True)
        backend = str(cfg.model.backend)
        if backend == "mock":
            raw = MockBackend(self.hub)
        elif backend == "ollama":
            from ..llm.ollama import OllamaBackend
            raw = OllamaBackend(str(cfg.model.name),
                                format_mode=str(cfg.model.get("format", "json")))
        elif backend == "vllm":
            # 本選: OpenAI 互換 vLLM。servers が 1本=単一バックエンド、複数=艦隊ルータ。
            # tiers があれば(単一 URL でも)艦隊で purpose 別振り分けの seam を通す。
            servers = [str(s) for s in (cfg.model.get("servers", []) or [])]
            if not servers:
                raise ValueError(
                    "model.backend=vllm には model.servers(URL リスト、1本以上)が必要。")
            model_name = str(cfg.model.name)
            timeout_s = float(cfg.model.get("timeout_s", 120.0))
            tiers = cfg.model.get("tiers", None)
            tiers = OmegaConf.to_container(tiers, resolve=True) if tiers else None
            fmt = str(cfg.model.get("format", "json"))
            if len(servers) == 1 and not tiers:
                from ..llm.vllm import VllmBackend
                raw = VllmBackend(model_name, servers[0], timeout_s=timeout_s,
                                  format_mode=fmt)
            else:
                from ..llm.fleet import FleetLLM
                raw = FleetLLM(servers, model_name, timeout_s=timeout_s,
                               tiers=tiers, format_mode=fmt)
        elif backend == "router":
            # 第23バッチ M2: 合成ルータ。purpose(rng_key 先頭)別に子バックエンドへ振り分ける。
            # ★子を各自 CachedLLM で包んでから router に渡す(キャッシュキーは子の name 由来=D13)。
            #   router を直接 CachedLLM で包むとモデル別キーが失われるため、この順序は変えないこと。
            #   キャッシュファイルは子の name 別(llm_cache.<name>.jsonl)。同一 spec の子は共有(重複構築しない)。
            from ..llm.router import RouterLLM
            rcfg = cfg.model.get("router", None)
            rcfg = OmegaConf.to_container(rcfg, resolve=True) if rcfg else None
            if not rcfg or not rcfg.get("default"):
                raise ValueError(
                    "model.backend=router には model.router.default(子バックエンド指定)が必要。")
            specs = {"default": rcfg["default"], **(rcfg.get("purpose") or {})}
            built: dict[str, CachedLLM] = {}
            children: dict[str, CachedLLM] = {}
            for purpose, spec in specs.items():
                skey = json.dumps(spec, ensure_ascii=False, sort_keys=True)
                if skey not in built:
                    child = self._build_router_child(spec)
                    safe = "".join(c if c.isalnum() or c in "._-" else "_"
                                   for c in child.name)
                    built[skey] = CachedLLM(
                        child, enabled=bool(cfg.model.cache),
                        path=self.out_dir / f"llm_cache.{safe}.jsonl")
                children[str(purpose)] = built[skey]
            self.llm = RouterLLM(children)
            raw = None
        else:
            raise NotImplementedError(
                f"backend '{backend}' は未実装(mock | ollama | vllm | router)。")
        if raw is not None:
            self.llm = CachedLLM(raw, enabled=bool(cfg.model.cache),
                                 path=self.out_dir / "llm_cache.jsonl")

        nodes = self.dests or sorted(self.city.graph.nodes)
        # ペルソナ名簿(scripts/build_personas.py 生成物。無ければ手続き生成)
        roster = None
        personas_file = cfg.get("agents", {}).get("personas_file")
        if personas_file:
            personas_path = Path(str(personas_file))
            if not personas_path.is_absolute():
                personas_path = REPO_ROOT / personas_path
            roster = json.loads(personas_path.read_text(encoding="utf-8"))["personas"]
        self.agents = []
        for agent_id in range(int(cfg.run.n_agents)):
            entry = roster[agent_id % len(roster)] if roster else None
            agent = build_agent(agent_id, self.hub.stream("persona", agent_id),
                                nodes, self.city, entry=entry,
                                economy=self.economy,
                                threshold_dist=self.threshold_dist,
                                drift=self.drivecfg["drift"],
                                reflection=self.reflectcfg)
            agent.x, agent.y = self.city.node_xy(agent.node)
            agent.visits[agent.node] += 1
            agent._heard_unknown = False
            agent._arrived_new = False
            agent._congestion = 1.0
            agent._pending_stay = 2
            agent._pending_activity = ""
            agent._ride_pending = None             # 交通機関の乗車(到着時に運賃を払う)
            agent._search_queue = []
            agent._drive_reasons = {}
            # 無意識層(第12バッチ): 感情価の生カウントを貯めるか(deep か implicit のどちらか
            # 有効なら True)。両OFF なら False=factors._impact_note が即 return=バイト一致。
            agent._impact_on = bool(self.reflectcfg["deep"]["enabled"]
                                    or self.reflectcfg["implicit_self"]["enabled"])
            agent._fire_reason = ""
            agent._last_fire_reason = ""           # 造語の発生きっかけの記録用
            agent._last_dm_from = None
            agent._reply_to = None
            agent._known_events = set()             # 参加告知を受けたイベント id
            agent._attending_event = None           # 会場へ向かっているイベント id
            agent.arousal = float(self.affectcfg["arousal_baseline"])  # 覚醒度の初期値=baseline
            agent.conv_turns_left = int(self.drivecfg["conv_max_turns"])
            agent.conv_cooldown_until = 0
            agent.mem.relations_max = self.relations_max   # B6: 台帳上限(既定0=無制限)
            self.agents.append(agent)
        self.agent_by_id = {a.id: a for a in self.agents}
        # 流入通勤者を朝の到着前=範囲外に置く(既存 visitor 帰還機構で朝 enter_area)。
        # commuter が居ない名簿(既定・procedural)では完全 no-op=既定挙動バイト一致。
        self._init_inflow_commuters()
        # 世帯構築(後続波 H2。起動時1回・既定 OFF=no-op)。同一世帯で home を共有=夜間の家庭内
        # co-location。この後の「顔なじみ」ブロックが home_building 共有から housemate を初期関係に
        # 入れる。OFF は home 割当・draw 順とも不変(新 stream "household" も引かない=バイト一致)。
        _household_mod.build_households(self)
        # 観光・多言語(後続波 H5。起動時1回・既定 OFF=no-op)。visitor の一部を回遊型観光客に、一部を
        # 非日本語話者に決定論(id 昇順)で仕立てる。乱数を引かない=既存 draw 順とも不変(バイト一致)。
        _diversity_mod.assign_attributes(self)
        # 行動心理プラグイン(既定 OFF)の個体別セットアップ。専用 stream で既存 draw 順を汚さない。
        if self.psychcfg["sdt"]["enabled"]:            # SDT: 欲求ゲージの個人別倍率 dict
            from ..factors.registry import sdt_params
            for a in self.agents:
                a.drive_mods = sdt_params(a.traits, self.hub.stream("sdt", a.id),
                                          self.psychcfg["sdt"])
        if self.psychcfg["collective"]["enabled"]:     # 集団: 新 state を ON 時だけ追加
            for a in self.agents:
                psych_mod.enable_collective(a, self.hub.stream("collective", a.id))
        # 欲求プロファイルの個人差(Wave2-2。既定 OFF)。専用 stream "needs" のみ=既存 draw 順不変。
        from ..needs import build_cfg as _needs_build_cfg, precompute as _needs_precompute
        self.needscfg = _needs_build_cfg(cfg.get("needs", None))
        if self.needscfg["enabled"]:
            _needs_precompute(self.agents, self.hub, self.needscfg)
        # 開放行動+価値の充足(第17バッチ・既定 OFF=sat/sat_mods 属性なし=バイト一致。
        # 乱数なし=既存 stream 不変)。
        from .. import values as _values_mod
        self.freedomcfg = _values_mod.build_cfg(cfg.get("freedom", None))
        if self.freedomcfg["open_actions"]:
            for a in self.agents:
                a.sat = {t: 0.5 for t in _values_mod.TAGS}
                _values_mod._refresh_mods(a, self.freedomcfg)
        # 主観的世界モデル(第20バッチ・既定 OFF=状態/イベント/プロンプトとも不変)。
        # 乱数ゼロ(EMA・カウントのみ)。可制御性は全員 0.5 起点=純経験の経路依存。
        from .. import worldview as _worldview_mod
        raw_wv = cfg.get("worldview", None)
        raw_wv = (OmegaConf.to_container(raw_wv, resolve=True)
                  if OmegaConf.is_config(raw_wv) else raw_wv)
        self.worldviewcfg = _worldview_mod.build_cfg(raw_wv)
        # 街路の環境情報(第18バッチ・既定 OFF=draw/イベント/プロンプトとも不変)。
        # 広告の乱数は新 stream "ads" のみ・群衆視覚は乱数なし(実在の集計のみ)。
        from .. import street as _street_mod
        raw_ads = cfg.get("ads", None)
        raw_ads = (OmegaConf.to_container(raw_ads, resolve=True)
                   if OmegaConf.is_config(raw_ads) else raw_ads)
        self.adscfg = _street_mod.build_ads_cfg(raw_ads)
        raw_crowd = cfg.get("crowd_visual", None)
        raw_crowd = (OmegaConf.to_container(raw_crowd, resolve=True)
                     if OmegaConf.is_config(raw_crowd) else raw_crowd)
        self.crowdcfg = _street_mod.build_crowd_cfg(raw_crowd)
        # SNS/DM 架橋距離の記録(第22バッチ P2・旧 shibuya-sim 由来)。bool 1個=OFF で完全 no-op。
        raw_snsgeo = cfg.get("sns_geo", None)
        raw_snsgeo = (OmegaConf.to_container(raw_snsgeo, resolve=True)
                      if OmegaConf.is_config(raw_snsgeo) else raw_snsgeo)
        self.snsgeo_on = bool((raw_snsgeo or {}).get("enabled", False))
        # 入力解像度LOD(第30バッチ・lod.input_res・既定OFF=属性なし=バイト一致)。
        # 割当は cognition/lod.py の共通軸機構(軸専用 stream "lod_input_res"・trait非依存)。
        from ..cognition import lod as _lod_mod
        raw_ir = cfg.get("lod", {}).get("input_res", None)
        raw_ir = (OmegaConf.to_container(raw_ir, resolve=True)
                  if OmegaConf.is_config(raw_ir) else raw_ir)
        self.inputrescfg = _lod_mod.build_input_res_cfg(raw_ir)
        if self.inputrescfg:
            _lv = _lod_mod.assign_axis([a.id for a in self.agents],
                                       self.hub.stream("lod_input_res"),
                                       self.inputrescfg["levels"])
            for a in self.agents:
                spec = self.inputrescfg["levels"][_lv[a.id]]
                a.input_res = {"level": _lv[a.id],
                               **{k: v for k, v in spec.items() if k != "share"}}
        self.net.init_follows([a.id for a in self.agents],
                              self.hub.stream("follows"),
                              k=int(cfg.get("net", {}).get("follow_k", 6)))
        # 初期関係: 同じ住宅・同じ職場は顔なじみ(complex contagion の wide bridges)
        from collections import defaultdict as _dd
        groups: dict = _dd(list)
        for a in self.agents:
            if a.home_building:
                groups[("home", a.home_building)].append(a)
            if a.work_building:
                groups[("work", a.work_building)].append(a)
        for _key, members in sorted(groups.items(), key=lambda t: t[0]):
            for i, a in enumerate(members):
                for b in members[i + 1:i + 4]:      # 1人あたり最大3人(過密回避)
                    self.net.add_contact(a.id, b.id)
                    a.mem.record_contact(b.id, b.name, 0, "顔なじみ")
                    b.mem.record_contact(a.id, a.name, 0, "顔なじみ")
        # 初期関係のアイスブレイク(実験前の初対面会話。build_icebreak.py 生成物)。
        # ★全 k 条件で同一ファイルを読む=初期関係が条件間で同一(交絡の排除)。
        self._load_icebreak(cfg)
        # Lynch: landmark_score を研究者frame の静的データとして書き出す(ON 時のみ)。
        if self.landmark_score is not None:
            (self.out_dir / "landmark_score.json").write_text(
                json.dumps(self.landmark_score, ensure_ascii=False),
                encoding="utf-8")
        # 研究者frame用の静的データ(R² 回帰の入力。エージェントには見せない)。
        # SDT プラグイン ON 時は倍率 dict を nested "sdt" で併記(数値でない=trait 名を汚さない)。
        (self.out_dir / "traits.json").write_text(json.dumps(
            {str(a.id): {**a.traits,
                         "drive_threshold": round(a.drive_threshold, 4),
                         "fire_weight": round(a.fire_weight, 4),
                         "visitor": a.visitor,
                         "money": round(a.money, 1),
                         "wage": round(a.wage, 1),
                         "part_time": bool(a.part_time),
                         **({"sdt": a.drive_mods} if a.drive_mods else {})}
             for a in self.agents},
            ensure_ascii=False), encoding="utf-8")
        # ビューア用の名簿(L1 には出ない静的情報)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "agents.json").write_text(json.dumps(
            [{"id": a.id, "name": a.name, "age": a.age, "gender": a.gender,
              "occupation": a.occupation, "visitor": a.visitor,
              "commute": a.commute, "arrival_min": a.arrival_min,
              "residence_line": a.residence_line,
              "has_bicycle": a.has_bicycle,
              "has_car": a.has_car, "home": a.home_node,
              "home_building": a.home_building, "home_floor": a.home_floor,
              "work_name": a.work_name, "work_building": a.work_building,
              "bedtime_min": a.bedtime_min,
              "money": round(a.money, 1), "wage": round(a.wage, 1),
              "part_time": bool(a.part_time),
              "part_time_name": (a.part_time["name"] if a.part_time else None),
              **({"sdt": a.drive_mods} if a.drive_mods else {}),
              **({"input_res": a.input_res["level"]}
                 if getattr(a, "input_res", None) else {})}
             for a in self.agents],
            ensure_ascii=False), encoding="utf-8")

    def _build_router_child(self, spec: dict):
        """router の子バックエンドを1つ構築する(第23バッチ M2)。

        spec = {backend: mock|ollama|vllm|openai_compat|anthropic, name: モデル名,
                base_url/host/api_key_env/timeout_s は任意}。
        APIキーは環境変数名(api_key_env)のみを渡す=キーの値は config にもログにも置かない。
        """
        b = str(spec.get("backend", ""))
        name = str(spec.get("name", ""))
        timeout_s = float(spec.get("timeout_s", 120.0))
        if b == "mock":
            return MockBackend(self.hub)
        if b == "ollama":
            from ..llm.ollama import OllamaBackend
            return OllamaBackend(name, host=str(spec.get("host", "http://localhost:11434")),
                                 timeout_s=timeout_s,
                                 format_mode=str(spec.get("format", "json")))
        if b == "vllm":
            from ..llm.vllm import VllmBackend
            return VllmBackend(name, str(spec.get("base_url", "http://localhost:8000")),
                               timeout_s=timeout_s,
                               format_mode=str(spec.get("format", "json")))
        if b == "openai_compat":
            from ..llm.openai_compat import OpenAICompatBackend
            return OpenAICompatBackend(
                name, base_url=str(spec.get("base_url", "https://api.openai.com/v1")),
                api_key_env=str(spec.get("api_key_env", "OPENAI_API_KEY")),
                timeout_s=timeout_s)
        if b == "anthropic":
            from ..llm.anthropic import AnthropicBackend
            return AnthropicBackend(
                name, api_key_env=str(spec.get("api_key_env", "ANTHROPIC_API_KEY")),
                base_url=str(spec.get("base_url", "https://api.anthropic.com")),
                timeout_s=timeout_s)
        raise ValueError(f"router の子 backend '{b}' は未対応"
                         "(mock | ollama | vllm | openai_compat | anthropic)。")

    def _init_inflow_commuters(self) -> None:
        """流入通勤者(commute=True)を朝の到着前=範囲外(loc=outside)に置き、朝の到着 step
        に既存の帰還機構(_phase_wake_and_returns)で enter_area させる。到着口は
        build_personas が配分した commute_gateway("station"~85-90% / "edge")で解決する。

        すべて決定論(乱数を引かない=既存 stream の draw 順に影響しない)。commuter が
        居ない名簿(既定・procedural、既存 personas_40/80)では 1件も該当せず完全 no-op
        =既定 config・既定地図での挙動バイト一致(docs/research/shibuya-inflow.md)。"""
        start_min = self.clock.start_min                 # day0 開始時刻(=07:00)
        for agent in self.agents:
            if not getattr(agent, "commute", False):
                continue
            if agent.commute_gateway == "station" and self.city.station_node:
                gw = self.city.station_node
            elif self.city.gateways:
                gw = self.city.gateways[agent.id % len(self.city.gateways)]
            else:
                gw = agent.node
            arr_min = agent.arrival_min if agent.arrival_min >= 0 else start_min
            agent.loc = "outside"
            agent.node = gw
            agent.return_gateway = gw
            agent.x, agent.y = self.city.node_xy(gw)
            agent.return_at = max(0, (arr_min - start_min) // 10)

    def _build_landmark_score(self) -> dict:
        """Lynch 都市イメージ: 目的地ノードごとの imageability 近似スコア(POI 数 + 名前つき
        地点 + 名前つき建物の近接)。engine は空間 affordance だけを見て因子名は使わない。

        Jiang(arXiv 1212.0940)の「少数の目立つ要素 vs 多数の平凡な要素」= head/tail の素材。
        ここでは決め打ちの landmark 指定をせず、POI 密度と命名の有無からスコアを組む(base 1.0
        で全ノード到達可能を保つ)。observer は事後に軌跡の集中度としてこれを使う。
        """
        score: dict[str, float] = {}
        for node in self.dests:
            n_pois = len(self.city.pois_at_node(node))
            named = 1.0 if self.city.graph.nodes[node].get("name") else 0.0
            n_named_bldg = sum(1 for b in self.city.buildings_at(node)
                               if b.get("name"))
            weight = self.psychcfg["lynch"]["landmark_weight"]
            score[node] = 1.0 + weight * (float(n_pois) + named
                                          + float(n_named_bldg))
        return score

    def _load_icebreak(self, cfg: DictConfig) -> None:
        """初期関係の注入(build_icebreak.py の生成物)。

        各ペアの初対面会話を、両者の記憶(関係台帳+エピソード)と SNS の
        知り合い関係(contacts)に反映する。step=0 の関係として与える。
        ★全 k 条件で同一ファイルを読むため、初期関係は条件間で完全に同一になる。
        """
        ib_file = cfg.get("agents", {}).get("icebreak_file")
        if not ib_file:
            return
        ib_path = Path(str(ib_file))
        if not ib_path.is_absolute():
            ib_path = REPO_ROOT / ib_path
        data = json.loads(ib_path.read_text(encoding="utf-8"))
        for pair in data.get("pairs", []):
            a = self.agent_by_id.get(int(pair["a"]))
            b = self.agent_by_id.get(int(pair["b"]))
            if a is None or b is None:
                continue
            # 会話の各発話 → 聞き手側の関係台帳に記録(往復数ぶん・最後の発話を保持)
            for turn in pair.get("turns", []):
                speaker = self.agent_by_id.get(int(turn["speaker"]))
                if speaker is None:
                    continue
                listener = b if speaker.id == a.id else a
                listener.mem.record_contact(speaker.id, speaker.name, 0,
                                            str(turn.get("text", "")))
            # 「初めて話した」エピソードを両者に1件ずつ(step 0)
            a.mem.observe(0, f"{b.name}と初めて話した", kind="event")
            b.mem.observe(0, f"{a.name}と初めて話した", kind="event")
            self.net.add_contact(a.id, b.id)       # 知り合い(DM 可・自動フォロー)

    def run(self, resume_from: Path | str | None = None) -> dict:
        # D16: checkpoint_every>0 で該当 step ごとに完全状態を保存 + ログを part 化。
        # 既定 0 = 無効(part を作らず出力・挙動は従来と完全同一)。
        every = int(self.cfg.observer.get("checkpoint_every", 0) or 0)
        # ログの定期 flush を checkpoint から分離(B2)。>0 で N step ごとに flush_segment
        # を呼び L1 バッファの RAM を解放する(大規模ランの OOM 回避)。既定 0 = 無効で
        # 従来と完全同一(part を作らない)。checkpoint と同一 step では二重 flush しない。
        flush_every = int(self.cfg.observer.get("flush_every_steps", 0) or 0)
        start = 0
        if resume_from is not None:
            ckpt = checkpoint.latest(Path(resume_from))
            if ckpt is None:                          # 誤って完走済み/空 dir を上書きしない
                raise FileNotFoundError(
                    f"resume 元に checkpoint が無い: {Path(resume_from) / 'checkpoint'}")
            start = checkpoint.load(self, ckpt)
        if every > 0:
            save_config(self.cfg, self.out_dir)   # 途中再開に備え config を先出しする
        for step in range(start, int(self.cfg.run.n_steps)):
            scheduler.run_step(self, step)
            did_flush = False
            if every > 0 and (step + 1) % every == 0:
                checkpoint.save(self, step + 1,
                                self.out_dir / "checkpoint"
                                / f"ckpt-{step + 1:06d}.pkl.gz")
                self.logger.flush_segment()
                did_flush = True
            if flush_every > 0 and not did_flush \
                    and (step + 1) % flush_every == 0:
                self.logger.flush_segment()        # checkpoint と独立の定期 flush
        return self.finalize()

    def finalize(self) -> dict:
        paths = self.logger.flush()
        save_config(self.cfg, self.out_dir)
        kinds = self.logger.total_event_kinds()
        summary = {
            "n_agents": len(self.agents),
            "n_steps": int(self.cfg.run.n_steps),
            "n_events": self.logger.total_n_events(),
            "event_kinds": dict(sorted(kinds.items())),
            "llm_calls": self.llm.calls,
            "llm_cache_hits": self.llm.hits,
            "n_items": len(self.items.items),
            "n_transmissions": sum(len(i.transmissions) for i in self.items.items.values()),
            "total_adoptions": sum(len(a.adopted) for a in self.agents),
            "out_dir": str(self.out_dir),
            "files": {k: str(v) for k, v in paths.items()},
        }
        (self.out_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary
