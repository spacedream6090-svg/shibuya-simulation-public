"""シミュレーション本体: 構築(config→世界+エージェント)と実行ループ。"""
from __future__ import annotations

import bisect
import json
import math
import time
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from ..agents.persona import build_agent
from ..agents.ref import to_ref as _to_ref
from ..cognition.lod import LodBudget
from ..config import REPO_ROOT, save_config
from ..labeling.labels import LabelSystem
from ..llm.cache import CachedLLM
from ..llm.mock import MockBackend
from ..observer import assets as assets_mod
from ..observer import deviation as deviation_mod
from ..observer import lens as lens_mod
from ..observer import manifest as manifest_mod
from ..observer.logger import ObserverLogger
from ..observer.provenance import ItemStore
from ..rng import RngHub
from .. import timeconv as _timeconv
from ..world.clock import Clock, STEP_MINUTES
from ..world.map import CityMap
from ..world.routing import Router
from ..world.transit import Transit
from ..world import presence as presence_mod
from ..world import pool as pool_mod
from ..world import floors as _floors_mod
from .. import worldview as _wv_mod
from . import checkpoint, scheduler


def _peak_rss_mb() -> float | None:
    """このプロセスのピーク常駐メモリ [MB]。取得できない環境では None(= summary に出さない)。

    P0バッチ 2026-07-29。**観測のみ・挙動に一切影響しない**。**真のピーク**を優先する取得順:
      1) Windows 素の stdlib: psapi.GetProcessMemoryInfo の PeakWorkingSetSize(真のピーク)
      2) POSIX 素の stdlib: resource.getrusage(RUSAGE_SELF).ru_maxrss(真のピーク。
         Linux は KB / macOS は byte)— **本選機が Linux ならここが効く**
      3) psutil(あれば)の rss = **現在値**であってピークではない(最後の手段・近似)
    tracemalloc(scripts/bench.py が使う Python ヒープ峰値)とは別物で、
    こちらは pyarrow 等の C++ バッファを含む **プロセス実メモリ**。
    """
    try:                                          # 1) Windows: psapi(stdlib ctypes のみ)
        import ctypes
        from ctypes import wintypes

        class _PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        k32 = ctypes.WinDLL("kernel32")
        psapi = ctypes.WinDLL("psapi")
        # ★argtypes/restype を明示しないと 64bit で HANDLE が切り詰められ常に失敗する
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE,
                                               ctypes.POINTER(_PMC), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = _PMC()
        counters.cb = ctypes.sizeof(_PMC)
        if psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(),
                                      ctypes.byref(counters), counters.cb):
            return round(float(counters.PeakWorkingSetSize) / (1024 * 1024), 1)
    except Exception:
        pass
    try:                                          # 2) POSIX(真のピーク)
        import resource
        import sys as _sys
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        unit = 1024 * 1024 if _sys.platform == "darwin" else 1024    # macOS=byte / Linux=KB
        return round(float(ru) / unit, 1)
    except Exception:
        pass
    try:                                          # 3) psutil(現在値=近似。最後の手段)
        import psutil                             # type: ignore
        rss = getattr(psutil.Process().memory_info(), "rss", None)
        return round(float(rss) / (1024 * 1024), 1) if rss else None
    except Exception:
        return None


def _parse_start_tod(v) -> int:
    """run.start_tod → day0 step0 の分 of day(0..1439)。既定 "07:00"=420 分=現行値(バイト一致)。

    "HH:MM" 文字列を第一に解釈する。YAML/OmegaConf が 60 進(07:00→420)や数値へ丸めた場合も
    そのまま分として受け付ける(どちらでも既定は 420=07:00 に落ちる)。乱数を一切引かない純関数。"""
    if v is None:
        return 7 * 60
    if isinstance(v, bool):                       # True/False の誤設定は既定へ倒す
        return 7 * 60
    if isinstance(v, (int, float)):               # 既に分(または 60 進で丸められた値)
        return int(v) % 1440
    s = str(v).strip()
    if ":" in s:
        hh, mm = s.split(":", 1)
        return (int(hh) * 60 + int(mm)) % 1440
    return int(s) % 1440


def _natural_wake_step(start_tod: int, bedtime_min: int, sleep_steps: int,
                       step_minutes: int = STEP_MINUTES) -> int | None:
    """開始時刻(分 of day)が自然な睡眠区間 [bedtime, bedtime+sleep) に入るなら、残りの睡眠を
    step 数(1 step=step_minutes 分)へ切り上げて返す。区間外(=開始時に覚醒しているべき)は
    None。1440 分の円環で判定し、乱数を一切使わない決定論導出(初日コールドスタート改善の核)。
    夜勤者(bedtime が日中側)や日中開始(将来 start 時刻をずらした場合)でも同じ式で自然に働く。

    step_minutes 既定 = 正準 10(第79バッチ以前の呼び出しと完全同一)。"""
    sm = int(step_minutes)
    dur = int(sleep_steps) * sm
    if dur <= 0:
        return None
    since = (int(start_tod) - int(bedtime_min)) % 1440    # 就寝からの経過分(円環)
    if since >= dur:
        return None                                        # もう起きているべき時刻=覚醒開始
    remain = dur - since                                   # 残りの睡眠(分)。since<dur なので >=1
    return (remain + sm - 1) // sm                         # step 数へ切り上げ(>=1)


def _snap_to_arrival(arrivals: list[tuple[int, str]],
                     minute: int) -> tuple[int, str] | None:
    """引かれた到着時刻を**時刻表の到着**へ量子化する(actor model Wave 3)。

    境界の研究(Hänseler & Bierlaire の駅の需要モデル / SUMO の運用)が言うのは
    「駅からの流入は**平滑ではなく列車ごとの塊(platoon)**である」ということ。
    平滑な流入では改札の待ち行列もホームの滞留も**原理的に立たない**ので、
    改札装置(P2)・駅員(P3a)の層が観測すべき現象そのものが消える。

    規則(すべて決定論・乱数ゼロ・呼ぶ順に依存しない純関数):
      (1) 通常 … `minute` **以上**で最も早い到着へ送る(= 次に来る電車に乗ってきた)。
      (2) 始発前 … 最も早い到着へ**繰り下げ**る(始発待ち。lines_in_service の
          「終電後は帰って来られない」と同じ発想の入場側)。(1) の自然な帰結でもある。
      (3) 終電後 … 最後の到着へ**繰り上げ**る。3 規則のうちこれだけが時刻を早める
          方向に動かす(送り先が翌サービス日になると『その日 1 日来ない』になり、
          流入通勤者の往復が壊れるため)。
      同着(同じ分に複数路線)は **路線名の昇順**で先頭を採る = arrivals_between が
      (時刻, 路線名) 順に並べているので添字の先頭がそれ。

    arrivals は arrivals_of_day() の並び(分 of day 昇順)。空なら None(= 呼び出し側は
    量子化しない)。返り値 = (到着分, 路線名)。
    """
    if not arrivals:
        return None
    m = int(minute)
    if m > arrivals[-1][0]:                 # (3) 終電後 = 最後の到着へ引き戻す
        return arrivals[-1]
    # (1)/(2) `("")` は任意の路線名より小さいので「時刻 >= m の先頭」を厳密に指す
    return arrivals[bisect.bisect_left(arrivals, (m, ""))]


class Simulation:
    def __init__(self, cfg: DictConfig, out_dir: Path | None = None):
        # P0バッチ 2026-07-29: summary.json の elapsed_sec 用の起点(構築込み)。run() が
        # ループ開始時に打ち直す。観測専用で挙動・乱数・出力イベントには一切影響しない。
        self._t_start = time.perf_counter()
        # 第72バッチ: ランモード(run.mode)による機能の自動取捨。**シム構築の最初の1箇所**で
        # resolved config のキーを off 値へ落とし切る(以降の全構築はゲート後の config だけを見る)。
        # 既定 run.mode=none では apply_mode は cfg を触らず同一オブジェクトを返す=1バイトも変わらない。
        # 落とした一覧は WARNING ログ + run_manifest.json(features.auto_disabled)に必ず残る。
        from ..registry import apply_mode as _apply_run_mode
        cfg, self.run_mode_report = _apply_run_mode(cfg)
        self.cfg = cfg
        name = cfg.run.name or f"seed{cfg.run.seed}"
        self.out_dir = Path(out_dir) if out_dir else REPO_ROOT / str(cfg.run.out_dir) / str(name)
        self.hub = RngHub(cfg.run.seed)
        # 環境パック(EnvPack)= 場所の知識(地名・ランドマーク・年中行事・気候・語彙・
        # メディア番組名)。基盤(src)は地名リテラルを持たず、ここで conf の envpack ブロックを
        # 1回だけ正準化して保持する(既存 build_cfg 流儀)。以降 city/transit/persona/prompt/
        # annual/weather/media/schedule/diversity は cfg 経由で読む(深い get を散らさない)。
        from .. import envpack as _envpack_mod
        raw_envpack = cfg.get("envpack", None)
        raw_envpack = (OmegaConf.to_container(raw_envpack, resolve=True)
                       if OmegaConf.is_config(raw_envpack) else raw_envpack)
        self.envpackcfg = _envpack_mod.build_cfg(raw_envpack)
        self.place_name = self.envpackcfg["lexicon"]["place_name"]  # 街の名前(プロンプト用)
        # 制度値(institutions)= 税率・予算・給付など「値=環境/参照」(D1-W3)。基盤(src)は
        # コード埋没の制度値を持たず、ここで conf の institutions ブロックを1回だけ正準化して保持する
        # (envpack と同じ build_cfg 流儀)。government(scheduler)は sim.institutionscfg 経由で読む。
        from .. import institutions as _institutions_mod
        raw_inst = cfg.get("institutions", None)
        raw_inst = (OmegaConf.to_container(raw_inst, resolve=True)
                    if OmegaConf.is_config(raw_inst) else raw_inst)
        self.institutionscfg = _institutions_mod.build_cfg(raw_inst)
        map_path = Path(cfg.world.map)
        if not map_path.is_absolute():
            map_path = REPO_ROOT / map_path
        self.city = CityMap(
            map_path,
            underground_label=self.envpackcfg["lexicon"]["underground_name"])
        self.router = Router(self.city)
        # 建物の実高さ(A1 第67バッチ。world.heights.enabled=false=既定なら**表を開きもしない**
        # =建物 dict にキーが増えない=バイト一致)。ON でも属性を付けるだけで知覚・移動・
        # プロンプト・LLM 呼数には一切影響しない(消費者は後続 C0/B-L1)。決定論・乱数ゼロ。
        hcfg = cfg.world.get("heights", None)
        self.heights_stat = None
        if hcfg is not None and bool(hcfg.get("enabled", False)):
            hpath = Path(str(hcfg.get("path", "data/building_heights_shibuya.json")))
            if not hpath.is_absolute():
                hpath = REPO_ROOT / hpath
            self.heights_stat = self.city.attach_heights(
                hpath, fallback_m_per_level=float(hcfg.get("fallback_m_per_level", 3.5)))
            # summary にはローカル絶対パスではなく conf に書かれた相対指定を残す(ラン間比較可能)
            self.heights_stat["path"] = str(hcfg.get("path",
                                                     "data/building_heights_shibuya.json"))
        # 環境改変条件(A1 第67バッチ world.mod)。現実幾何が原点、改変はその周りの反実仮想。
        # 既定 OFF=プロファイルを開きもしない=地図・エッジ属性とも不変=バイト一致。
        # 適用はここで一度だけ(決定論・乱数ゼロ)。適用後は経路キャッシュを捨てる
        # (scenario.shock_closure と同じ規約)。commerce への適用は commercecfg 構築後(下方)。
        from ..world import worldmod as _worldmod
        self.worldmod = _worldmod.load(cfg, REPO_ROOT)
        if self.worldmod is not None:
            self.worldmod.apply_world(self.city)
            self.router.invalidate()
        # 標高 z 列(3D Phase 0)。既定 OFF=イベント payload 不変=バイト一致。
        # ON でも読み出しは純関数(乱数なし)・記録専用=認知経路と LLM 呼数は不変(R1)。
        ecfg = cfg.world.get("elevation", None)
        self.elevation = None
        if ecfg is not None and bool(ecfg.get("enabled", False)):
            from ..world.elevation import ElevationGrid
            edir = Path(str(ecfg.get("dir", "data/plateau")))
            if not edir.is_absolute():
                edir = REPO_ROOT / edir
            self.elevation = ElevationGrid.load(edir)
        # 開始時刻(run.start_tod="HH:MM"。既定 "07:00"=420 分=現行値=バイト一致)。分 of day へ
        # 解釈して Clock へ渡す。day 境界・時刻表示・夜内省/朝計画・natural_start は全て clock 由来の
        # sim_min から決定論導出するので、この 1 点の配線だけで開始時刻の仮定が全経路へ波及する。
        # 中央 Δt(第79バッチ)。run.dt_min 既定 10 = STEP_MINUTES = 従来と完全同一。
        # 以後「1 step は何分か」は self.clock.step_minutes(および steps_per_day/
        # steps_per_hour/step_seconds)だけが供給する = 単一源。
        self.dt_min = _timeconv.dt_of(cfg)
        self.clock = Clock(start_min=_parse_start_tod(cfg.run.get("start_tod", "07:00")),
                           step_minutes=self.dt_min)
        self.logger = ObserverLogger(self.out_dir)
        # W2-6 / W4-E: finalize のメモリ有界化(observer.finalize.streaming。既定 OFF=従来経路)。
        # ON のときだけ属性を立てる = 既定ランは各クラスの初期値のまま=バイト一致
        # (apply_cfg が OFF で early return する = 属性を 1 つも触らない)。
        # ★同じ設定を **L1 とサイドカー全部**へ配る(新 conf キーは増やさない=1 つの判断で
        #   全ファイルが同じモードになる)。サイドカーへの配布は __init__ 末尾でまとめて行う
        #   (生成箇所が散っているため。配り漏れは tests/test_finalize_streaming_sidecars.py が
        #   Simulation の属性を機械走査して検出する)。
        from ..observer import finalize as _finalize_mod
        self._finalize_cfg = _finalize_mod.cfg_of_config(cfg)
        _finalize_mod.apply_cfg(self.logger, self._finalize_cfg)
        # IF-F 因果台帳(observer.causality.enabled。既定 OFF=属性を 1 つも触らない)。
        # finalize と同じ「ON のときだけ立てる」流儀 = 既定ランは初期値のまま=バイト一致。
        from ..observer import causality as _causality_mod
        _causality_mod.apply_cfg(self.logger, _causality_mod.cfg_of_config(cfg))
        self.items = ItemStore()
        # ---- 場所の意味づけ最小版 D1(labeling.place_binding。既定 OFF=完全 no-op=バイト一致)----
        # ON 時だけ LabelSystem が束縛台帳を持つ(状態は LabelSystem 内に閉じるので checkpoint の
        # "labels" 同梱でそのまま resume される=checkpoint.py への追加配線は不要)。
        # dotlist 上書きは文字列で入りうるのでここで型を確定させる(config.py の型強制表と同じ趣旨)。
        _pb_raw = cfg.labeling.get("place_binding", None)
        _place_binding = None
        if _pb_raw is not None and bool(_pb_raw.get("enabled", False)):
            _place_binding = {"enabled": True,
                              "min_adopters": int(_pb_raw.get("min_adopters", 1)),
                              "prompt_line": bool(_pb_raw.get("prompt_line", True))}
        self.labels = LabelSystem(self.items,
                                  adopt_threshold=cfg.labeling.adopt_threshold,
                                  mode=str(cfg.labeling.get("mode", "constrained")),
                                  place_binding=_place_binding)
        npro = cfg.lod.get("n_proportional", None)
        if npro is not None and bool(npro.get("enabled", False)):
            dens = float(npro.get("density", 0.15))
            # round(…,9) は 0.15×40=6.000000000000001 のような float 誤差での繰り上げ事故を防ぐ
            cap = max(1, math.ceil(round(dens * int(cfg.run.n_agents), 9)))
        else:
            cap = cfg.lod.max_llm_per_step
        # ---- DPH-B 二層予算(lod.budget.tiers・既定 OFF=tiers None=第30 と完全同一)----
        #   ON では plan / reflect が予算の**中**へ入り、返答保証に先取り枠が立つ
        #   (総量は cap のまま = 分配だけが変わる)。docs/plans/dayplan-horizon-plan.md §3.3
        from ..cognition.lod import build_budget_cfg as _build_budget_cfg
        _tiers = _build_budget_cfg((cfg.lod.get("budget", None) or {}).get("tiers", None))
        # ---- DPH-O ④ purpose 別の許可/拒否カウンタ(observer.starvation・既定 OFF=None)----
        from ..observer import starvation as _starv
        self.budget = LodBudget(cap, tiers=_tiers,
                                counters=_starv.budget_counters(self))
        self.pois = self.city.pois()
        self.dests = self.city.destinations()
        # ---- 屋内エンジン配線(B3。既定 OFF=完全 no-op=バイト一致)----
        # 屋内ミクロ状態(全建物・全階の区画=IndoorSpace)を単一の真実として据え、遷移駆動 SFM で
        # フロア内移動・遭遇を回す(scheduler._phase_indoor)。indoor.enabled=false なら self.indoor=None=
        # _phase_indoor が即 return=新 stream("indoor"/"indoor_meet")も引かない=ゴールデン L1 バイト一致。
        from ..world import indoor as _indoor_mod
        from ..world import indoor_flow as _indoor_flow_mod
        raw_indoor = cfg.get("indoor", None)
        raw_indoor = (OmegaConf.to_container(raw_indoor, resolve=True)
                      if OmegaConf.is_config(raw_indoor) else raw_indoor)
        self.indoorcfg = _indoor_mod.build_engine_cfg(raw_indoor)
        self.indoor = _indoor_mod.indoor_from_cfg(self.city, cfg)   # None=既定 OFF
        self.indoorparams = _indoor_flow_mod.IndoorParams.from_cfg(self.indoorcfg["sfm"])
        # ---- P3 境界縫合 竹-4(既定 OFF = ゾーン 0 件 = 一切の新経路を通らない)----
        # `physics.zones_enabled: false`(既定)/ `physics.zones: []`(既定)のとき
        # physcfg["zones"] は空 tuple になり、scheduler の physics.phase() が即 return する
        # = agent._phys_* が 1 つも生えない = zone_gate 0 件 = L1 バイト一致。
        from ..world import zones as _zones_mod
        raw_phys = cfg.get("physics", None)
        raw_phys = (OmegaConf.to_container(raw_phys, resolve=True)
                    if OmegaConf.is_config(raw_phys) else raw_phys)
        # step_seconds を渡すのは `max_sub_steps` 未指定のゾーンで上限を Δt から導くため
        # (第99・OBS-U2 §1.2 B5 の是正)。Δt=10 かつ dt_sub=0.05 では 12000 = 従来値。
        self.physcfg = _zones_mod.build_cfg(raw_phys, REPO_ROOT,
                                            step_seconds=self.clock.step_seconds)
        self._indoor_meet_plan = {}                # (group,day)->(occurs,meet_min) 決定論キャッシュ(非pickle)
        self._indoor_encounters_step = []          # step 内一時状態(動力学→観測)。B3b が動力学として読む
        self._indoor_samples_step = []             # step 内一時状態(tracks 観測用)
        self._indoor_n_space_move = 0              # per-step カウンタ(L2・resume 安全=累積しない)
        self._indoor_n_encounter = 0
        self._indoor_n_meeting = 0
        # 記録サイドカー(indoor.enabled かつ tracks.enabled のみ。tracks は indoor.enabled と独立サブトグル)
        self.indoor_tracks = None
        if self.indoor is not None and self.indoorcfg["tracks"]["enabled"]:
            from ..observer.indoor_tracks import IndoorTracks
            self.indoor_tracks = IndoorTracks(self.out_dir)
        transit_path = Path(str(cfg.transit.file))
        if not transit_path.is_absolute():
            transit_path = REPO_ROOT / transit_path
        gtfs_dir = cfg.transit.get("gtfs_dir")
        self.transit = Transit(
            transit_path, gtfs_dir=gtfs_dir or None,
            station_filters=self.envpackcfg["transit"]["station_filters"],
            # 第114 レーン 3b(既定 false=等間隔の再構成のまま=バイト一致)。
            real_departures=bool(cfg.transit.get("real_departures", False)))
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
        # 月別気候テーブルは envpack(場所の値)から与える(基盤に東京の気候を残さない)。
        # 第80バッチ W2: 生成器パラメータ / 実測表のパスも同じ流儀で envpack 経由。
        self.weathercfg = _weather_mod.build_cfg(
            raw_wea, monthly=self.envpackcfg["climate"]["monthly"],
            neutral=self.envpackcfg["climate"]["neutral"],
            gen_params=self.envpackcfg["climate"].get("gen_params"),
            table_file=self.envpackcfg["climate"].get("table"))
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
        # 自由行動の行き先を地図原点付近の集会ノードへ寄せる。要 world.calendar.enabled=true
        # (日付判定に暦を使う)。決定論(群集バイアスは新 stream "crowd" のみ=既存 draw 順は不変)。
        from .. import annual as _annual_mod
        raw_annual = cfg.get("annual_events", None)
        raw_annual = (OmegaConf.to_container(raw_annual, resolve=True)
                      if OmegaConf.is_config(raw_annual) else raw_annual)
        # 年中行事(名前・日付・群集フラグ)は envpack(文化=場所の値)を既定に使う。
        self.annualcfg = _annual_mod.build_cfg(
            raw_annual, default_events=self.envpackcfg["culture"]["events"])
        self.today_event_line = None           # 当日の年中行事1行(annual 有効かつ当日のみ非 None)
        self.today_crowd_event = None          # 当日の群集イベント名(群集フラグ行事日のみ非 None)
        self._annual_day = -1                  # 日境界(年中行事の確定)の進行管理
        self._crowd_surge_day = -1             # crowd_surge を1日1回に制限(群集の記録済み日)
        self.crowd_node = None                 # 集会ノード(地図原点付近。ON 時のみ解決)
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
                headway_steps=self.ridecfg["bus"]["headway_steps"],
                step_minutes=self.clock.step_minutes)   # A3: 便判定の分→step 換算(既定 10=同値)
        # 実バスダイヤの静的表 v-Ride-2(既定 OFF=None=従来の簡易バス近似のまま=バイト一致)。
        # bus.enabled かつ bus_table.enabled かつ表が読めるときだけ BusTable を生成し、_ride_extra が
        # 従来近似の代わりに実ダイヤ近似(次便待ち+区間所要)を使う。設計: src/society/bus_table.py。
        self.bus_table = None
        bt_raw = (rcfg.get("bus_table", {}) or {})
        if self.ridecfg["bus"]["enabled"] and bool(bt_raw.get("enabled", False)):
            from .. import bus_table as _bus_table_mod
            bt_path = str(bt_raw.get("path", "data/odpt/bus_table_shibuya.json"))
            if not Path(bt_path).is_absolute():
                bt_path = str(REPO_ROOT / bt_path)
            bt_origin = self.city.meta.get("origin_latlon")
            self.bus_table = _bus_table_mod.load_bus_table(
                bt_path, self.city,
                origin=(tuple(bt_origin) if bt_origin else None),
                access_radius_m=float(bt_raw.get("access_radius_m", 150.0)))
        # SUMO ライブ連成タクシー配車 v-Ride-1(既定 OFF=None=SUMO を一切起動しない=バイト一致)。
        # ON 時のみ TaxiLive を生成(backend=mock は SUMO 不要の純関数 / traci は実 SUMO を起動し、
        # 不在・失敗なら None へ後退=従来挙動)。設計: src/society/transit_live.py。
        from .. import transit_live as _transit_live_mod
        origin = self.city.meta.get("origin_latlon", None)
        _live_raw = (cfg.get("transit_ride", {}) or {}).get("live", {}) or {}
        self.taxi_live = _transit_live_mod.make_taxi_live(
            cfg, net_hint=_live_raw.get("net"),
            origin=(tuple(origin) if origin else None), log=print)
        # 背景交通(エージェント以外の通過車両)
        from ..world.traffic import TrafficFlow
        tcfg = cfg.world.get("traffic", {})
        self.traffic = TrafficFlow(
            self.city, self.router, self.hub,
            enabled=bool(tcfg.get("enabled", True)),
            cars_per_day=int(tcfg.get("cars_per_day", 30000)),
            max_log=int(tcfg.get("max_log", 120)),
            step_minutes=self.dt_min)
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
        # M-W(第37バッチ): ACT-R 活性化記憶。既定 OFF=agent.mem.actr は None のまま=バイト一致。
        self.actrcfg = dict(mcfg.get("actr", {}) or {})
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
        self._dinner_logged: set = set()       # 夕食共食の記録済み (household_id, day)(S-R1)
        # 共同行動エンジン(関係性の再現 第44バッチ S-R3。既定 OFF=現行挙動と完全同一)。日次編成
        # (日境界・新 stream "joint")+ 収束(routine 分岐)+ 同席観測(joint_activity)。決定論・
        # LLM 非増(呼数不変は compute_matched 下の k 不変性で担保)。OFF=何も編成せず・イベント 0 件・
        # 新 stream も引かない=乱数消費不変(ゴールデン golden_baseline_l1.json を守る)。
        from .. import joint as _joint_mod
        raw_joint = cfg.get("joint", None)
        raw_joint = (OmegaConf.to_container(raw_joint, resolve=True)
                     if OmegaConf.is_config(raw_joint) else raw_joint)
        self.jointcfg = _joint_mod.build_cfg(raw_joint)
        self._joint_day = -1                    # 日境界(共同行動の編成)の進行管理
        self._joint_groups: list = []           # 当日の成立グループ(observe が同席で1件記録)
        self._joint_total = 0                   # joint_activity の累積件数(L2 集計用)
        # 友人グラフ生成(関係性パッケージ完結 第45バッチ S-R2。既定 OFF=現行挙動と完全同一)。起動時1回、
        # 居住者に homophily+所属+Dunbar 次数の友人辺を張る(relations の closeness/tier へ注入)。辺は安定
        # ハッシュ(friend_graph.seed)+ペルソナ属性の純関数=run.seed 非依存・乱数 stream ゼロ(RngHub 無風)。
        # OFF=何も張らず・relations 台帳不変・friend_graph_built も出さない=バイト一致(ゴールデンを守る)。
        from .. import friends as _friends_mod
        raw_friend = cfg.get("friend_graph", None)
        raw_friend = (OmegaConf.to_container(raw_friend, resolve=True)
                      if OmegaConf.is_config(raw_friend) else raw_friend)
        self.friendcfg = _friends_mod.build_cfg(raw_friend)
        # 来街者 party の実体化(第45バッチ S-R5。既定 OFF=現行挙動と完全同一)。present な来街者を
        # party_size でグループ化=同一初期ノード・相互 relations・共有回遊 POI。presence 純関数(P3)には
        # 触れず present 実体化の後にグループ化するだけ=rotation の決定論・resume 不変を保つ。乱数は新
        # stream "party" のみ。OFF=グループ化せず・party_today も付かず・joint_activity 0 件=バイト一致。
        from .. import party as _party_mod
        raw_party = cfg.get("party", None)
        raw_party = (OmegaConf.to_container(raw_party, resolve=True)
                     if OmegaConf.is_config(raw_party) else raw_party)
        self.partycfg = _party_mod.build_cfg(raw_party)
        # 火種介入の実験条件化(spark treatment 第53バッチ 2026-07-23。既定 OFF=現行挙動と完全同一)。
        # 「立ち上げには初期介入が不可欠」という助言と「介入ゼロで観測」という方針の対立を ON/OFF treatment で
        # 決着させる。介入は hardcode せず初期条件(関係の束/資本・在庫/集会アンカー)の差としてのみ与える。
        # trait-blind 選抜(pid, spark.seed の安定ハッシュ純関数)・decay は集会アンカーのみ・spark_roster 1件。
        # OFF=何も適用せず・spark_roster も出さず・"spark" stream も引かない=バイト一致(ゴールデンを守る)。
        from .. import spark as _spark_mod
        raw_spark = cfg.get("spark", None)
        raw_spark = (OmegaConf.to_container(raw_spark, resolve=True)
                     if OmegaConf.is_config(raw_spark) else raw_spark)
        self.sparkcfg = _spark_mod.build_cfg(raw_spark)
        # 閾値分布の seam(手続き生成経路のみ。名簿値優先=現状のまま)
        self.threshold_dist = str(cfg.get("factors", {}).get("threshold_dist",
                                                             "normal") or "normal")
        # 初期個体差ゼロ対照(第74バッチ IDEA④。既定 OFF=build_agent 側で 1 行も通らない)。
        # conf 非搭載の旧 config でも既定へ落ちる(実験マニフェストだけが ON にする)。
        _flat = cfg.get("experiment", {}) or {}
        _flat = (_flat.get("flat_traits", None) if hasattr(_flat, "get") else None) or {}
        self.flatcfg = {
            "enabled": bool(_flat.get("enabled", False)),
            "value": float(_flat.get("value", 0.5)),
            "include_derived": bool(_flat.get("include_derived", True)),
        }
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
        # 内部可動性 第60バッチ b(転居。既定 OFF=現行挙動と完全同一)。職場変更後の通勤逼迫 or 家賃逼迫
        # (E5 滞納)から世帯単位で内生的に転居する(新 stream "housing")。同棲(household.cohabit)・
        # career選択由来化(career.by_choice)と合わせた3機構=火種介入とは別の内生変化。決定論・非LLM
        # (R1: 呼数は compute_matched 下の k 不変性で担保)。OFF=relocate 0 件・stream も引かない=バイト一致。
        from .. import mobility as _mobility_mod
        raw_housing = cfg.get("housing", None)
        raw_housing = (OmegaConf.to_container(raw_housing, resolve=True)
                       if OmegaConf.is_config(raw_housing) else raw_housing)
        self.housingcfg = _mobility_mod.build_relocation_cfg(raw_housing)
        self._housing_day = -1                      # 日境界(転居・同棲)の進行管理
        # 存在の内生化 POP(転出・転入・出生。既定 OFF=現行挙動と完全同一)。名簿そのものが
        # 増減する 3 イベントを、日次の抽選割当ではなく**個体状態の関数**として起こす層
        # (転出=Wolpert 閾値 / 転入=案A「L4 定着昇格」/ 出生=世帯状態からの決定論)。
        # 乱数 stream を 1 本も引かず(全て安定ハッシュ)・LLM 呼を 1 本も足さず・新しい L1
        # kind を 1 つも作らない(既存 life_event の下位 kind)。OFF=台帳すら作らない=バイト一致。
        from .. import population as _population_mod
        raw_pop = cfg.get("population", None)
        raw_pop = (OmegaConf.to_container(raw_pop, resolve=True)
                   if OmegaConf.is_config(raw_pop) else raw_pop)
        self.popcfg = _population_mod.build_cfg(raw_pop)
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
        # 環境改変条件の営業時間オーバーライド(world.mod。OFF=self.worldmod is None=何もしない)。
        # cat 単位の上書きのみ実効(POI 単位は worldmod の予約フィールド)。決定論・乱数ゼロ。
        if self.worldmod is not None:
            self.worldmod.apply_commerce(self.commercecfg)
        # 物流の実体化 スライス①+②(commerce.inventory。既定 OFF=現行挙動と完全同一)。店舗在庫((s,S) 方策)+
        # 日次補充トリップ(depot→店=delivery_trip→到着で restock)+ 商品実体(何を買ったか=spend.item)。
        # 決定論・非LLM・乱数ゼロ(R1: 品切れは spend 抑制のみ・補充は agent_id=-1 の世界イベント=個体位置を
        # 変えない=compute_matched 下の k 不変性で担保)。封鎖(災害の運休/shock_closure)で補充失敗→欠品波及。
        # OFF=在庫実体なし・delivery_trip/restock/stock_low/stock_out(実在庫版)0 件・所持も生えない=バイト一致。
        from .. import goods as _goods_mod
        self.goodscfg = _goods_mod.build_cfg(raw_commerce)
        self._goods_stock: dict = {}               # (node, cat) -> 在庫数(遅延初期化=購入が起きた POI のみ)
        self._goods_pending: dict = {}             # (node, cat) -> 到着 step(補充トリップの在庫。多重発注抑止)
        self._goods_review_day: int = -1           # 日次 (s,S) レビューの進行管理
        # B2B 卸→小売の仕入れ スライス⑤(commerce.inventory.b2b。既定 OFF=補充供給元は外生 depot のまま)。
        # 卸/製造 org(output_kinds に素材/製品)が生産で在庫を積み、小売補充が卸在庫から引かれ org 間で金+物が
        # 移転(会計保存)。卸在庫不足=補充失敗=欠品波及。決定論・乱数ゼロ・LLM 呼ゼロ。OFF=b2b_trade 0 件=
        # バイト一致(organizations OFF なら ON でも卸不在=外生 depot 扱いで graceful)。
        from .. import b2b as _b2b_mod
        self.b2bcfg = _b2b_mod.build_cfg(raw_commerce)
        self._b2b: dict = {}                       # 卸在庫・売上・仕入費の台帳(遅延構築)
        self._b2b_total: int = 0                   # b2b_trade 件数(aggregator 用)
        # サービスの実体化 スライス③(services。既定 OFF=来店せず・"service" stream も引かない=バイト一致)。
        # 理美容/クリニック/塾/ジム/クリーニング等の service/education POI で「滞在して効果を受ける」最小形。
        # 来店=自由時間の行き先バイアス(routine の dest チェーン中立1分岐)+ 受給=課金(_spend)+ 効果(factors
        # on_service へ不透明 magnitude)+ 記憶 + service_use。決定論・新 stream "service" のみ・LLM 呼ゼロ。
        from .. import services as _services_mod
        raw_services = cfg.get("services", None)
        raw_services = (OmegaConf.to_container(raw_services, resolve=True)
                        if OmegaConf.is_config(raw_services) else raw_services)
        self.servicescfg = _services_mod.build_cfg(raw_services)
        self._service_total: int = 0               # service_use 件数(aggregator 用)
        # 宅配・フードデリバリー スライス④(delivery。既定 OFF=注文せず・"delivery" stream も引かない=バイト一致)。
        # 在宅/職場滞在の agent が食事帯に外食の代替として注文(① 実在庫を1引き店で受取)→ 配達員(gig/自営「配達員」)を
        # 決定論選定して物理配車 or 抽象トリップ→ eta 到着で受給=注文者に課金(食事+配達手数料。_spend)+ 配達員に gig 収入
        # (_pay_wage source="gig"=金の保存)。決定論・新 stream "delivery" のみ・LLM 呼ゼロ(R1)。判定は物理位置・時刻・
        # 所持金・config・観測量のみ=k 非依存(注文で在宅に留まる=co-location 変化は services/goods と同型)。
        # OFF=order/deliver 0 件・注文台帳も配達員フラグも生えない=ゴールデン golden_baseline_l1.json をバイト一致で守る。
        from .. import delivery as _delivery_mod
        raw_delivery = cfg.get("delivery", None)
        raw_delivery = (OmegaConf.to_container(raw_delivery, resolve=True)
                        if OmegaConf.is_config(raw_delivery) else raw_delivery)
        self.deliverycfg = _delivery_mod.build_cfg(raw_delivery)
        self._delivery_pending: list = []          # 宅配注文の到着待ち台帳(遅延構築でも可)
        self._delivery_total: int = 0              # deliver(配送完了)件数(aggregator 用)
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
        # 環境フィードバック(エージェント → 環境。第84バッチ 2026-08-01。設計 §4)。既定 OFF=完全同一。
        # 集約物理量の閾値超過で**コード側が**環境イベントを発火する層(§4.1: エージェントは
        # 環境イベントを宣言できない)。規則は §4.2 の最小 3 本(ホーム密度→停車時間延長→遅延伝播 /
        # 改札飽和→入場規制 / POI 占有>容量→待ち行列→他 POI へ流出)。LLM 0 本・乱数 0 本=strict。
        # disaster の直後に置く=どちらも「世界側の事象」だが、災害は**外生ショック**、こちらは
        # **エージェントの行動が作った内生の事象**という対比が読めるようにしてある。
        # OFF=env_feedback 0 件・状態(_envfb)すら生えない=L1 バイト一致・draw 不変。実装 envfeedback.py。
        from .. import envfeedback as _envfb_mod
        raw_envfb = (cfg.get("env", None) or {})
        raw_envfb = (OmegaConf.to_container(raw_envfb, resolve=True)
                     if OmegaConf.is_config(raw_envfb) else dict(raw_envfb))
        self.envfbcfg = _envfb_mod.build_cfg(raw_envfb.get("feedback"))
        # 第67 の予約フィールド world.mod.gate_capacity を「消費済み」に記録し直す(正直な summary)。
        _envfb_mod.mark_gate_capacity_consumed(self)
        # 生活の偶発イベント層(第54バッチ 2026-07-23。純観察=不確実性許容モードの柱。既定 OFF=現行挙動と完全同一)。
        # 臨時収入/財布紛失(money±)+ 偶然の出会い(closeness+)を単一 chance_event に束ね、日境界に専用 stream
        # "chance"((agent, day) 個体別キー)で抽選する。効果は money/relations/記憶のみ=ドライブ/発火に流さない
        # (k 非汚染)。R1: generate() を1本も足さない。OFF=抽選せず・chance_event 0 件・"chance" stream も引かない
        # =乱数消費不変(ゴールデン golden_baseline_l1.json を守る)。実装 src/society/chance.py。
        from .. import chance as _chance_mod
        raw_chance = cfg.get("chance", None)
        raw_chance = (OmegaConf.to_container(raw_chance, resolve=True)
                      if OmegaConf.is_config(raw_chance) else raw_chance)
        self.chancecfg = _chance_mod.build_cfg(raw_chance)
        self._chance_day = -1                       # 日境界(偶発イベント抽選)の進行管理
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
        # L2 業務の実体(work.service。既定 OFF=現行挙動と完全同一)。勤務中エージェントへ
        # 接客(serve)/オフィス産出(org_output)の実体イベントを与える決定論機構(LLM/乱数ゼロ)。
        # OFF は _phase_work_service が即 return=イベント 0 件・状態も増やさない=バイト一致。
        from .. import work as _work_mod
        raw_work = cfg.get("work", None)
        raw_work = (OmegaConf.to_container(raw_work, resolve=True)
                    if OmegaConf.is_config(raw_work) else raw_work)
        self.workcfg = _work_mod.build_cfg(raw_work)
        # 夜間経済(world.night_economy。第101 III-1。既定 OFF=現行挙動と完全同一)。
        # 日跨ぎシフト(work._window の丸め)・24h/POI 単位の営業時間(commerce)・終電後の
        # 避難先(scheduler._try_exit)の 3 点をこの 1 ブロックで束ねる。OFF は
        # night_refuge 0 件・営業時間表 不変・勤務窓の丸め 従来どおり=L1 バイト一致。
        from .. import night as _night_mod
        self.nightcfg = _night_mod.build_cfg((cfg.get("world", {}) or {})
                                             .get("night_economy", None))
        _night_mod.wire_work(self.nightcfg, self.workcfg)
        self._night_refuge_pois = None              # 避難先候補 POI(ON 時のみ 1 回だけ組む)
        self._work_day = -1                        # 日次境界(オフィス産出集計)の進行管理
        # 会社観測データ層 B4(work.service.ledger / office.by_org。既定 OFF=バイト一致・サイドカー不在)。
        # 動力学(scheduler)が per-day アキュムレータへ積み、observer(OrgLedger)が日次境界で読むだけ。
        self._org_day = {}                         # per-day アキュムレータ(org_id → 集計)
        self._org_ledger_day = -1                  # 日次境界の進行管理(org_output by_org / ledger 締め)
        self.org_ledger_sc = None                  # 日次系列サイドカー(ledger.enabled ON 時のみ生成)
        if self.workcfg["enabled"] and self.workcfg["ledger"]["enabled"]:
            from ..observer.org_ledger import OrgLedger
            self.org_ledger_sc = OrgLedger(self.out_dir)
        # ---- IF-E2 案B: 部門別残高の日次サイドカー finance.parquet(既定 OFF=生成しない)----
        # 検査①(Caiani の貨幣ストック保存)を org / 銀行 / 行政 / RoW でも成立させるための唯一の
        # 観測点。記録と動力学の分離は org_ledger と同じ(動力学は add_rows/flush/finalize だけ)。
        from .. import economy_sfc as _sfc_mod
        self.sfccfg = _sfc_mod.build_cfg(
            (cfg.get("economy", None) or {}).get("org_accounting", None))
        self.finance_sc = None
        if self.sfccfg["enabled"] and self.sfccfg["sidecar"]:
            self.finance_sc = _sfc_mod.FinanceLedger(self.out_dir)
        # ---- 観測チャンネル o_c(t)(第80バッチ。既定 OFF=サイドカー不在=完全 no-op)----
        # 驚き駆動発火の**入力側**の定義(cognition/channels.py)を step ごとに読み取り専用で
        # 計算し、channels.parquet へ書くだけ。ON でも L1/L2/L3・乱数・LLM 呼数は不変。
        # 較正テーブル(壊れていたら即エラー)と σ_c 凍結(無ければ absent)の来歴もここで 1 回だけ採る。
        from ..cognition import calib as _calib_mod
        from ..cognition import channels as _channels_mod
        raw_ch = (cfg.get("cognition", {}) or {}).get("channels", None)
        raw_ch = (OmegaConf.to_container(raw_ch, resolve=True)
                  if OmegaConf.is_config(raw_ch) else raw_ch)
        self.channelscfg = _channels_mod.build_cfg(raw_ch)
        # ---- 閾値発火 + 認知イベントキュー(第81バッチ。既定 OFF=キュー不在=完全 no-op)----
        # ON のときだけ _phase_drive の発火権配布がキュー駆動に差し替わる(単一作用点)。
        from ..cognition import fire as _fire_mod
        raw_fire = (cfg.get("cognition", {}) or {}).get("fire", None)
        raw_fire = (OmegaConf.to_container(raw_fire, resolve=True)
                    if OmegaConf.is_config(raw_fire) else raw_fire)
        self.firecfg = _fire_mod.build_cfg(raw_fire)
        self.cogq = _fire_mod.CogQueue() if self.firecfg["enabled"] else None
        self._fire_plan_due = ()               # 朝計画の対象者(記録専用。OFF は常に空)
        self._fire_usable = None               # usable チャンネル索引のメモ(初回に確定)
        # ---- 監視仕様 watch(ô=LLM 出力+トリガ DSL)と g/θ 更新則(第82バッチ)----
        # どちらも **fire ON が前提**の既定 OFF。OFF では module が 1 度も呼ばれない。
        from ..cognition import plasticity as _plasticity_mod
        from ..cognition import watch as _watch_mod
        _raw_cog = (cfg.get("cognition", {}) or {})
        raw_watch = _raw_cog.get("watch", None)
        raw_watch = (OmegaConf.to_container(raw_watch, resolve=True)
                     if OmegaConf.is_config(raw_watch) else raw_watch)
        self.watchcfg = _watch_mod.build_cfg(raw_watch)
        raw_g = _raw_cog.get("g_update", None)
        raw_g = (OmegaConf.to_container(raw_g, resolve=True)
                 if OmegaConf.is_config(raw_g) else raw_g)
        self.gcfg = _plasticity_mod.build_cfg(raw_g)
        # ---- engaged モード(第87バッチ。**fire ON が前提**の既定 OFF)----
        # 発火(点)の上にエピソード(区間)を載せる層。OFF では module が 1 度も呼ばれない。
        from ..cognition import engaged as _engaged_mod
        raw_engaged = _raw_cog.get("engaged", None)
        raw_engaged = (OmegaConf.to_container(raw_engaged, resolve=True)
                       if OmegaConf.is_config(raw_engaged) else raw_engaged)
        self.engagedcfg = _engaged_mod.build_cfg(raw_engaged)
        # ---- Perception / Intent 契約(第85バッチ P1。既定 OFF=従来経路)----
        # ON では scheduler の材料収集 → build_prompt が world → Perception → prompt に
        # 一本化される(プロンプト文字列は 1 バイトも変わらない = ON/OFF で世界一致)。
        from ..cognition import perception_contract as _contract_mod
        raw_contract = _raw_cog.get("contract", None)
        raw_contract = (OmegaConf.to_container(raw_contract, resolve=True)
                        if OmegaConf.is_config(raw_contract) else raw_contract)
        self.contractcfg = _contract_mod.build_cfg(raw_contract)
        # 初期値条件 F/N/P(§2.7)。experiment ブロックの下= flat_traits と同じ置き場。
        _ginit = cfg.get("experiment", {}) or {}
        _ginit = (_ginit.get("g_init", None) if hasattr(_ginit, "get") else None)
        _ginit = (OmegaConf.to_container(_ginit, resolve=True)
                  if OmegaConf.is_config(_ginit) else _ginit)
        self.ginitcfg = _plasticity_mod.build_init_cfg(_ginit)
        self._g_day = -1                       # θ 恒常性の日境界(OFF は使われない)
        self.cognition_g_sc = None
        self.channels_sc = None
        self.channels_sat = False              # G6: sat 追加列(既定 OFF=列が生えない)
        self.cognition_calib = None
        self.cognition_sigma = None
        if self.channelscfg["enabled"]:
            from ..observer import channels as _obs_channels
            # G6(第114): 価値 4 軸の充足 sat を**観測側の追加列**として足す(既定 OFF)。
            # OFF では列自体が生えない = 既存 channels.parquet とスキーマ・バイトが同一。
            # ★cognition 側の CHANNELS には足さない(σ_c 凍結ファイルの spec hash が動くため)。
            self.channels_sat = bool(self.channelscfg["sat_columns"])
            _cols = tuple(_channels_mod.COLUMNS)
            if self.channels_sat:
                _cols += _obs_channels.sat_columns()
            self.channels_sc = _obs_channels.ChannelsSidecar(self.out_dir, _cols)
        # 較正テーブル(壊れていたら即エラー)と σ_c 凍結の来歴は、観測チャンネルか発火の
        # どちらかが ON なら 1 回だけ採る(発火は σ_c を S の分母として実際に消費する)。
        if self.channelscfg["enabled"] or self.firecfg["enabled"]:
            self.cognition_calib = _calib_mod.load_calib(self.channelscfg["calib_file"])
            self.cognition_sigma = _calib_mod.load_sigma(self.channelscfg["sigma_file"])
            # σ_c は Δt=10 で測った分散(OBS-U2 §2(d))。Δt が違うランで黙って使うと
            # 発火率に系統バイアスが乗るので、**警告 1 回**だけ出す(値は補正しない)。
            _calib_mod.check_sigma_dt(self)
        # g/θ の全軌跡サイドカー(§2.7「g_i(0) と g の全軌跡をログする」)。
        # 観測層なので動力学はこのバッファを読まない = ON でも L1/L2/L3 は変えない。
        if _plasticity_mod.enabled(self) and self.gcfg["log_every_steps"]:
            from ..observer.channels import CognitionGSidecar
            self.cognition_g_sc = CognitionGSidecar(
                self.out_dir, _plasticity_mod.columns(self))
        # 職場束ね直し(work.bind_workplace。既定 OFF=現行の work_node 付与と完全同一=バイト一致)。
        # ON 時: pool 経路の L2/L3(occupation が persona._WORK_CAT に載らず work_node を持たない個体)を
        # 台帳 workplace_poi.node へ org_id で束ね直し、接客(serve)/産出(org_output)の網羅率を上げる。
        # 束ねは build_pool_agent 内で決定論(pool_pid の純関数=run.seed 非依存=hydrate 再入不変)に行う。
        self._workbind_cfg = self.workcfg["bind_workplace"]
        self._workbind_book = None                 # org_id → 職場情報(ON 時のみ 1 回読む)
        self._workbind_stat = None                 # day0 present の coverage 統計(起動時 1 件で記録)
        self._workbind_agg = {"n_total": 0, "n_unbound_before": 0,
                              "n_bound": 0, "n_unbound_after": 0}
        # 束ね手段/不能理由の内訳(L1 には出さない実行時カウンタ。workplace_bound の payload は
        # observer/schema.py で 4 列固定の契約なので、そこは 1 列も増やさない)。
        self._workbind_detail = {"how": {}, "by_layer": {}, "unresolved_occ": {}}
        if self._workbind_cfg["enabled"]:
            self._workbind_book = _work_mod.load_bind_book(self._workbind_cfg, REPO_ROOT)
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
                # event の day/time は絶対 sim 分(day*1440+時刻)。step = (sim分 - 開始分)/STEP。
                # 開始分は clock.start_min(run.start_tod。既定 420=07:00 で従来の 7*60 とバイト一致)。
                step = (minute - self.clock.start_min) // self.clock.step_minutes
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
        # ---- 第71バッチ: LLM 入出力ジャーナル / REPLAY モードの配線 ----
        # journal: 全 LLM 呼び出しの**プロンプト全文**を per-run に残す観測 sink(書くだけ・
        #   シム本体からは読まない=決定論にも k 呼数にも不参加)。router の子キャッシュにも
        #   1本ずつ付けて漏らさない。既定 ON(新規別ファイルなので L1 バイトには触れない)。
        # cache_mode: free(既定=従来動作)| replay(キャッシュミスで即例外・フォールバック禁止)。
        #   矛盾する組(cache=false かつ replay)は CachedLLM の __init__ が起動時に落とす。
        cache_mode = str(cfg.model.get("cache_mode", "free"))
        journal_on = bool(cfg.model.get("journal", True))
        journal_flush = int(cfg.model.get("journal_flush_records", 128) or 128)
        self._journals: list = []
        _journal_by_stem: dict = {}

        def _new_journal(stem: str):
            if not journal_on:
                return None
            # 同名ファイルには必ず同一オブジェクトを返す。router で別 spec の子が同じ
            # backend.name(例: mock 2 個)になるとファイル名が衝突し、別オブジェクトだと
            # seq が二系列になって重複する(キャッシュファイルが子の name 別=同名共有なのと同じ扱い)。
            if stem in _journal_by_stem:
                return _journal_by_stem[stem]
            from ..llm.journal import LlmJournal
            j = LlmJournal(self.out_dir / stem, flush_records=journal_flush)
            _journal_by_stem[stem] = j
            self._journals.append(j)
            return j

        backend = str(cfg.model.backend)
        # 1 呼の絶対時限(M-1)。mock は HTTP を張らないので対象外。0 以下で無効=従来経路。
        deadline_s = float(cfg.model.get("call_deadline_s", 300.0))
        # β11(第117): request-level stable seed。既定 OFF では **None** を渡す
        # = vLLM 送出ボディが従来とバイト一致(seed キー自体を作らない)。
        # ON では run.seed を材料の 1 つとして子 backend へ渡すだけで、乱数 stream は
        # 1 本も引かない(値は (run_seed, rng_key) の blake2b 純関数)。
        _rs_cfg = (cfg.get("llm", {}) or {}).get("request_seed", {}) or {}
        request_seed = (int(cfg.run.seed)
                        if bool(_rs_cfg.get("enabled", False)) else None)
        self.llm_request_seed = request_seed
        if backend == "mock":
            raw = MockBackend(self.hub)
        elif backend == "ollama":
            from ..llm.ollama import OllamaBackend
            raw = OllamaBackend(str(cfg.model.name),
                                format_mode=str(cfg.model.get("format", "json")),
                                deadline_s=deadline_s)
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
                                  format_mode=fmt, deadline_s=deadline_s,
                                  request_seed=request_seed)
            else:
                from ..llm.fleet import FleetLLM
                raw = FleetLLM(servers, model_name, timeout_s=timeout_s,
                               tiers=tiers, format_mode=fmt,
                               deadline_s=deadline_s,
                               request_seed=request_seed)
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
                        path=self.out_dir / f"llm_cache.{safe}.jsonl",
                        mode=cache_mode,
                        journal=_new_journal(f"llm_journal.{safe}.jsonl.gz"))
                children[str(purpose)] = built[skey]
            self.llm = RouterLLM(children)
            raw = None
        else:
            raise NotImplementedError(
                f"backend '{backend}' は未実装(mock | ollama | vllm | router)。")
        if raw is not None:
            self.llm = CachedLLM(raw, enabled=bool(cfg.model.cache),
                                 path=self.out_dir / "llm_cache.jsonl",
                                 mode=cache_mode,
                                 journal=_new_journal("llm_journal.jsonl.gz"))
        # ---- 第88バッチ: 心モデル固定(1 体 1 モデル)の解決層 ------------------------- #
        # 原文書 §5「心(会話・思考・価値判断)は 1 エージェント 1 モデルを誕生時に固定する」。
        # ここで作るのは **dispatcher だけ**(新しいバックエンド型は 1 つも作らない):
        #   default = ここまでに組み上がった self.llm(= 機械的判断・対照系列・agent_id を
        #             持たない呼びの受け皿。原文書の「モデル不問の共有小型モデル」に対応)
        #   children= pool(+ 高解像度層の大型モデル)を **各自 CachedLLM に包んで**渡す
        #             = キャッシュキーは子の name 由来(D13)= モデル別に応答が分離される。
        # 既定 OFF(model.mind.enabled=false)では self.llm を 1 バイトも触らない。
        from .. import mind as _mind_mod
        raw_mind = cfg.model.get("mind", None)
        raw_mind = (OmegaConf.to_container(raw_mind, resolve=True)
                    if OmegaConf.is_config(raw_mind) else raw_mind)
        self.mindcfg = _mind_mod.build_cfg(raw_mind)
        self._mind_binding: dict = {}          # agent_id → model_id(誕生時固定のメモ化)
        self._mind_logged: set = set()         # L1 mind_assign を出し終えた agent_id
        if self.mindcfg["enabled"]:
            from ..llm.mind_router import MindRouter
            entries = list(self.mindcfg["pool"])
            _hi = self.mindcfg["high"]
            if _hi["name"] and _hi["name"] not in {e["name"] for e in entries}:
                # 高解像度層の大型モデル。spec 省略時は pool 先頭と同じ backend 種別で
                # 名前だけ差し替える(mock 検証で "mock:hi" を立てるための最短経路)。
                hspec = _hi["spec"]
                if hspec is None:
                    hspec = dict(entries[0]["spec"])
                    hspec["name"] = _hi["name"]
                entries.append({"name": _hi["name"], "weight": 0.0, "spec": hspec})
            _mind_built: dict = {}
            children: dict = {}
            for entry in entries:
                skey = json.dumps(entry["spec"], ensure_ascii=False, sort_keys=True)
                if skey not in _mind_built:
                    child = self._build_router_child(entry["spec"])
                    safe = "".join(c if c.isalnum() or c in "._-" else "_"
                                   for c in child.name)
                    _mind_built[skey] = CachedLLM(
                        child, enabled=bool(cfg.model.cache),
                        path=self.out_dir / f"llm_cache.{safe}.jsonl",
                        mode=cache_mode,
                        journal=_new_journal(f"llm_journal.{safe}.jsonl.gz"))
                children[entry["name"]] = _mind_built[skey]
            self.llm = MindRouter(
                self.llm, children,
                resolve=lambda aid: _mind_mod.model_of_id(self, aid),
                purposes=self.mindcfg["purposes"])
        # 第78バッチ: アブレーション設定の正準化(既定=全 OFF=述語が全て False=挙動不変)と、
        # 認知階層 ablate.cognitive_tier の FleetLLM への焼き付け(構築時 1 回だけ・以降不変)。
        # fleet 非使用ランでは small/mid は縮退して full と同一 → WARNING で告知する。
        from .. import ablate as _ablate_mod
        raw_ablate = cfg.get("ablate", None)
        raw_ablate = (OmegaConf.to_container(raw_ablate, resolve=True)
                      if OmegaConf.is_config(raw_ablate) else raw_ablate)
        self.ablatecfg = _ablate_mod.build_cfg(raw_ablate)
        _ablate_mod.apply_fleet_tier(self)
        # 第89バッチ: プラセボ L1(context_shuffle / persona_swap / context_sever)。
        # 既定 OFF は sim._placebo=None=個体に属性を 1 つも生やさない=プロンプトはバイト一致。
        # 名簿より前に据える(誕生の各経路 _init_agent_runtime が個体を結線するため)。
        _ablate_mod.make_placebo(self)
        # 第92バッチ: プロンプト言い換え(ablate.prompt_paraphrase = S-16)。既定 "" は
        # sim._paraphrase=None=個体に属性を 1 つも生やさない=プロンプトはバイト一致。
        # プラセボと同じ位置(名簿より前)に据える=誕生の全経路が attach_agent を通る。
        _ablate_mod.make_paraphrase(self)

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
        # ---- 群のオントロジー(文化圏×経験の世界観共有群。2026-07-21・既定 OFF=属性なし=バイト一致)----
        # 各エージェントに「文化圏×経験」の群を安定ハッシュ(ontology.seed, persona id)で1つ割り当てる。
        # run.seed 非依存=同一設定なら全ランで同一人物は同一群(比較実験の要)。乱数ゼロ(RngHub 無風)・
        # traits/k 非参照(因子と直交)。ON 時のみ agent.ontology_group / ontology_line を据える。
        # agents 構築(下の pool day0 着席・直接ラン loop)より前に cfg を用意する必要がある。
        from .. import ontology as _ontology_mod
        raw_onto = cfg.get("ontology", None)
        raw_onto = (OmegaConf.to_container(raw_onto, resolve=True)
                    if OmegaConf.is_config(raw_onto) else raw_onto)
        self.ontologycfg = _ontology_mod.build_cfg(raw_onto)
        # ---- 日次ローテーション/presence(W2 P3・既定 OFF=n_agents 固定=バイト一致)----
        # pool.enabled=true で 100万プールから day0 の present を決定論選択して着席させる。
        # OFF(pool is None)なら従来どおり run.n_agents 固定の名簿サイクリック割当(挙動不変)。
        self._pool = None                   # PoolStore(ON 時のみ)。OFF は None=全 seam が no-op
        self._dormant = None                # DormantStore(退場者のスリム状態退避)
        self._pool_day = 0                  # 日境界ローテーションの進行管理(day0 は init で着席済み)
        self._pool_nodes = nodes            # rotation の build_pool_agent が使う出発ノード候補
        poolcfg = cfg.get("pool", {}) or {}
        # ---- pool 経路の org 配属(第109 レーン ORG。既定 OFF=None=完全 no-op)------------
        # organizations.attach は「名簿ファイル → 配属ファイル」でしか配属を引けないので、
        # プールのラン(agents.personas_file=null)では配属が構造的に 0 件だった(第108 縦煙)。
        # プール record は台帳から逆算して作られていて org_id/role を持っているので、それを
        # 読んで attach と同じ意味論を与える(organizations.attach_record)。
        # ★台帳をここで先に読むのは **day0 の着席が build_pool_agent を通る**ため
        #   (scheduler._ensure_orgs の遅延初期化は step0 = 着席より後)。読み込むのは
        #   pool ON かつ organizations ON のときだけで、同じ dict を self.orgs に置くので
        #   _ensure_orgs は再読み込みしない(台帳を二重にメモリへ載せない)。
        self._pool_org_book = None          # org_id → org(pool ON かつ orgs ON のときだけ)
        self._pool_org_commute = False      # 配属時に workplace_poi へ通わせるか(台帳と同義)
        self._pool_org_stat = {"n_present": 0, "n_attached": 0}   # 検収用の実行時カウンタ
        if bool(poolcfg.get("enabled", False)):
            pool_dir = Path(str(poolcfg.get("dir", "data/persona_pool")))
            if not pool_dir.is_absolute():
                pool_dir = REPO_ROOT / pool_dir
            self._pool = pool_mod.PoolStore(pool_dir)
            self._pool_present_cap = int(poolcfg.get("present_cap", 300))
            # 層別クォータ(DP-U3 案A。既定 OFF=従来の層優先 break=選抜集合が現行と完全一致)。
            self._pool_tier_quota = bool(
                (poolcfg.get("tier_quota", {}) or {}).get("enabled", False))
            # 在場の内生化(PRES-A1/A2。既定 = mode:quota + habit OFF = 現行と完全一致)。
            # ★正準化は presence.build_presence_cfg の 1 箇所だけ(mode の妥当性検査もそこ)。
            self._pool_presence = presence_mod.build_presence_cfg(
                poolcfg.get("presence", None))
            self._dormant = pool_mod.DormantStore(cap=int(poolcfg.get("dormant_cap", 0)))
            from .. import organizations as _orgs_mod
            _orgscfg = _orgs_mod.build_orgs_cfg(cfg.get("organizations", None))
            if _orgscfg["enabled"]:
                self.orgscfg = _orgscfg           # scheduler._orgs_on と同じ正準化物を共有
                self.orgs = _orgs_mod.load_book(_orgscfg)
                self.org_ledger = {}              # _ensure_orgs が入れるのと同じ初期値
                self._pool_org_book = self.orgs
                self._pool_org_commute = bool(_orgscfg.get("commute_to_poi", False))
            # 退避スリム状態の台帳幅(M-3。既定 = 現行の素値 = 挙動不変)。dunbar の休眠関係が
            # 「再会する前に台帳から落ちる」相互作用を観察ランで緩められるようにする口。
            pool_mod.configure(rel_cap=int(poolcfg.get("relations_cap",
                                                       pool_mod._REL_CAP)),
                               ep_cap=int(poolcfg.get("episodes_cap",
                                                      pool_mod._EP_CAP)))
            weekday = self._pool_weekday(0)
            day0 = presence_mod.present_for_day(
                self._pool.presence_records(), 0, self._pool_present_cap, self.hub, weekday,
                self._pool_tier_quota,
                habit=self._pool_presence["habit"],
                emergent=self._pool_presence["emergent"],
                rain=self._pool_rain(0))
            for pid in day0:
                self.agents.append(self.build_pool_agent(pid, self._pool.get(pid)))
            # 職場束ね直しの day0 coverage 統計を控える(起動時 1 件で記録=run_step step0)。
            if self._workbind_book is not None:
                self._workbind_stat = dict(self._workbind_agg)
        else:
            for agent_id in range(int(cfg.run.n_agents)):
                entry = roster[agent_id % len(roster)] if roster else None
                agent = build_agent(agent_id, self.hub.stream("persona", agent_id),
                                    nodes, self.city, entry=entry,
                                    economy=self.economy,
                                    threshold_dist=self.threshold_dist,
                                    drift=self.drivecfg["drift"],
                                    reflection=self.reflectcfg,
                                    place_name=self.place_name,
                                    flat=self.flatcfg,
                                    dt_min=self.dt_min)
                self._init_agent_runtime(agent)
                # 第114 レーン 1b: 名簿 entry の is_foreign / visit_purpose を控える
                # (既定 OFF=属性を 1 つも生やさない)。pool でない名簿ランの入口。
                _diversity_mod.note_record(self, agent, entry)
                # 群のオントロジー(既定 OFF=no-op)。直接ランは tier=default(プール非使用)。
                self._apply_ontology(agent, "default")
                # 第88: 心のモデルと知能層を誕生時に固定(既定 OFF=属性を生やさない=no-op)。
                _mind_mod.assign(self, agent)
                self.agents.append(agent)
        self.agent_by_id = {a.id: a for a in self.agents}
        self._present_ids = None                  # 在場索引(present predicate)は遅延構築
        # 第89: ペルソナ対合(A↔B の相互交換)を名簿が出揃った時点で 1 回だけ組む。
        # 専用 stream からのみ引く=既存 draw 順に干渉しない。既定 OFF は no-op。
        _ablate_mod.finish_placebo(self)
        # S6a: pool ON かつ N 比例予算なら、当日の在場数を N に使う(1行の接続)。
        self._pool_update_budget()
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
        # 構造化シーン記述 v0(world.scene_desc・既定 OFF=プロンプト/イベント不変=バイト一致)。
        # 決定論・追加 LLM 呼ゼロ・乱数ゼロの純関数集計(方向つき視界/注視対象/垂直関係)。
        from ..world import scene_desc as _scene_desc_mod
        raw_scene = cfg.world.get("scene_desc", None)
        raw_scene = (OmegaConf.to_container(raw_scene, resolve=True)
                     if OmegaConf.is_config(raw_scene) else raw_scene)
        self.scenecfg = _scene_desc_mod.build_cfg(raw_scene)
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
        # 友人グラフ生成(第45バッチ S-R2。顔なじみブロックの直後に中立な1呼び出し)。既定 OFF=no-op。
        # ON=homophily+所属+Dunbar 次数で居住者の友人辺を relations へ注入(乱数 stream ゼロ=バイト一致)。
        _friends_mod.build_friend_graph(self)
        # 来街者 party の実体化(第45バッチ S-R5。起動時=day0 の present 来街者をグループ化)。既定 OFF=no-op。
        _party_mod.form_parties(self, 0, 0)
        # 火種介入の適用(spark treatment 第53バッチ。party の後の中立な1呼び出し)。既定 OFF=no-op=バイト一致。
        # ON=trait-blind 選抜→(a)関係の束/(b)資本・在庫/(c)集会アンカーを初期条件として据え・spark_roster を1件記録。
        _spark_mod.apply(self)
        # 初期関係のアイスブレイク(実験前の初対面会話。build_icebreak.py 生成物)。
        # ★全 k 条件で同一ファイルを読む=初期関係が条件間で同一(交絡の排除)。
        self._load_icebreak(cfg)
        # 初日コールドスタート改善(run.natural_start・既定 OFF=no-op=バイト一致)。home_building は
        # ここまでで確定(household 等の後段割当を含む)なので、着席の最後にまとめて自然な睡眠状態へ。
        self._apply_natural_start()
        # Lynch: landmark_score を研究者frame の静的データとして書き出す(ON 時のみ)。
        if self.landmark_score is not None:
            (self.out_dir / "landmark_score.json").write_text(
                json.dumps(self.landmark_score, ensure_ascii=False),
                encoding="utf-8")
        # 研究者frame用の静的データ(R² 回帰の入力。エージェントには見せない)。
        # SDT プラグイン ON 時は倍率 dict を nested "sdt" で併記(数値でない=trait 名を汚さない)。
        # G7-④(observer.gt_extras・既定 OFF): 5 軸の潜在プロファイルと reason 倍率を
        # 名簿へ足す(「何を欲しやすい個体か」という静的ラベルがどこにも無かった)。
        # OFF ではキーを 1 つも足さない = 既存 traits.json とバイト一致。
        from ..observer import gt_extras as _gt_mod
        self.gt_extras = _gt_mod.cfg_of_config(cfg)
        _gt_on = bool(self.gt_extras["enabled"])
        (self.out_dir / "traits.json").write_text(json.dumps(
            {str(a.id): {**a.traits,
                         "drive_threshold": round(a.drive_threshold, 4),
                         "fire_weight": round(a.fire_weight, 4),
                         "visitor": a.visitor,
                         "money": round(a.money, 1),
                         "wage": round(a.wage, 1),
                         "part_time": bool(a.part_time),
                         **({"sdt": a.drive_mods} if a.drive_mods else {}),
                         **(_gt_mod.needs_extras(a) if _gt_on else {})}
             for a in self.agents},
            ensure_ascii=False), encoding="utf-8")
        # ビューア用の名簿(L1 には出ない静的情報)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "agents.json").write_text(
            json.dumps(self._agents_json_records(), ensure_ascii=False),
            encoding="utf-8")
        # 第50バッチ: 観測レンズの kind_map サイドカー(lens ON 時のみ書く=OFF は後方互換でバイト同一)。
        # ビューアが runs/<name>/lens_map.json 経由で写像を読む(sim⇄viz 疎結合。軸語は observer/lens.py に閉じる)。
        lens_mod.write_sidecar(self, self.out_dir)
        # 第55バッチ タスクA: ペルソナ逸脱率レンズの map サイドカー(deviation ON 時のみ書く=OFF は
        # 後方互換でバイト同一)。ビューアが deviation_map.json 経由で読む(ペルソナ語は observer/deviation.py に閉じる)。
        deviation_mod.write_sidecar(self, self.out_dir)
        # 第59バッチ スライス(a): 資産分布レンズの map サイドカー(assets ON 時のみ書く=OFF は後方互換で
        # バイト同一)。ビューアが assets_map.json の有無で 💰資産タブを出す(資産語は observer/assets.py に閉じる)。
        assets_mod.write_sidecar(self, self.out_dir)
        # 第71バッチ: ラン来歴(git SHA・config hash・seed・モデル・cache_mode・全スイッチ・
        # 開始時刻)を run_manifest.json に固定する。**書くだけ・読む経路なし**(R1)。
        manifest_mod.write(self)
        # 第78バッチ: 状態ハッシュチェーン(既定 OFF=None=run() のフックが即 no-op)。
        # ON のときだけ chain を作る。**書くだけ・読む経路なし**(R1)。
        from ..observer import state_hash as _state_hash_mod
        _shcfg = _state_hash_mod.cfg_of_config(cfg)
        self.state_hash = (
            _state_hash_mod.StateHashChain(self.out_dir,
                                           interval=_shcfg["interval"],
                                           float_digits=_shcfg["float_digits"])
            if _shcfg["enabled"] else None)
        # ---- A13 日次入場者名簿サイドカー(レーン丙 2。observer.roster_daily・既定 OFF)----
        # traits.json は day0 の名簿・agents.json はその瞬間の在場者しか写さないので、プール
        # 回転で day1 以降に入場する個体(finals 構成で 20.9 万人)の素性がどこにも残らない。
        # 観測側だけで閉じる 1 枚のサイドカーで塞ぐ(動力学は本層のバッファを 1 度も読まない)。
        from ..observer import roster as _roster_mod
        _rostercfg = _roster_mod.cfg_of_config(cfg)
        self.roster_sc = (_roster_mod.RosterDaily(self.out_dir)
                          if _rostercfg["enabled"] else None)
        # ---- レーンG 案B: 意図収束サイドカー(observer.gathering_intent・既定 OFF)----
        # 塞ぐ穴: 「集まった事実」は crowd_surge / L1 の在場で測れるが、**「集まろうと
        # した事実」がどこにも残らない**。しかも意図の 3 欄は L1 から原理的に復元できない:
        #   ① 計画ブロックの**解決済みノード**(plan_created は place カテゴリしか射影せず、
        #      plan_block_start.node は「実行時の現在ノード」= 行き先ではない)
        #   ② 予定帳の**相手側の写し**(L1 に出るのは話者視点の 1 件だけ)
        #   ③ イベントの**認知集合**(_known_events = 参加率の分母)
        # 新しい L1 kind を 1 つも足さない(出口はサイドカー parquet 1 枚)= ON でも
        # L1 はバイト不変。観測側だけで閉じる(世界も乱数も LLM 呼数も 1 バイト不変)。
        from ..observer import gathering as _gathering_mod
        _gathcfg = _gathering_mod.cfg_of_config(cfg)
        self.gathering_sc = (
            _gathering_mod.GatheringIntent(
                self.out_dir, slot_min=_gathcfg["slot_min"],
                capture_min=_gathcfg["capture_min"],
                repeat_min=_gathcfg["repeat_min"],
                min_intent=_gathcfg["min_intent"],
                sample_cap=_gathcfg["sample_cap"])
            if _gathcfg["enabled"] else None)
        # ---- G4/G5(第114 GT ロガー): 記憶ストリーム / 関係台帳の日次サイドカー ----------
        # どちらも観測側だけで閉じる(世界を 1 バイトも書かず・L1 を 1 件も出さず・乱数ゼロ・
        # LLM 呼数不変)。既定 OFF ではオブジェクトを作らず 1 ファイルも書かない。
        # 塞ぐ穴: 記憶の本文と関係の連続値 closeness は **checkpoint にしか無く**、世代を
        # 剪定した瞬間に消える = 復元実験の正解ラベルが本選ランで失われる。
        from ..observer import memory as _memsc_mod
        from ..observer import relations as _relsc_mod
        _memcfg = _memsc_mod.cfg_of_config(cfg)
        self.memory_sc = (_memsc_mod.MemoryDaily(self.out_dir,
                                                 text_chars=_memcfg["text_chars"])
                          if _memcfg["enabled"] else None)
        _relcfg = _relsc_mod.cfg_of_config(cfg)
        self.relations_sc = (_relsc_mod.RelationsDaily(self.out_dir,
                                                       passing=_relcfg["passing"])
                             if _relcfg["enabled"] else None)
        # ---- W4-E: finalize のメモリ有界化をサイドカーへも配る(同一 conf キー)----
        # サイドカーは生成条件がばらばら(indoor.tracks / work.ledger / economy.sidecar /
        # cognition.channels / g_update / observer.roster_daily)で __init__ の各所で作られる
        # ので、**全部そろったここで 1 度だけ**配る。既定 OFF では apply_cfg が何も触らない
        # = バイト一致。配り漏れ(新サイドカー追加時)は tests が Simulation の属性を走査して検出する。
        for _sc in (self.indoor_tracks, self.org_ledger_sc, self.finance_sc,
                    self.channels_sc, self.cognition_g_sc, self.roster_sc,
                    self.memory_sc, self.relations_sc, self.gathering_sc):
            if _sc is not None:
                _finalize_mod.apply_cfg(_sc, self._finalize_cfg)
        # ---- レーン D1(第109): 「起動時セグメント」の終端印 ------------------------
        # ★__init__ が L1 へ出したイベントの本数。step を 1 つも回していないこの時点の
        #   バッファは**起動時 1 回の発火体の出力そのもの**である(friend_graph_built /
        #   party の joint_activity / spark_roster / icebreak …)。resume した 2 つ目の
        #   プロセスでも __init__ は丸ごと走るので、これらは全部もう一度積まれる
        #   = 前のプロセスが part へ確定済みのぶんと**二重記録**になる(第109 縦煙の実測:
        #   全ON構成で +106 行 / +0.89%)。checkpoint がこの本数を運び、load 側が
        #   先頭からその本数を捨てることで、種を 1 つずつ名指しせずに根治する
        #   (第98 W4-A は spark_roster **1 種だけ**を名指しで塞いだ = 網羅できていなかった)。
        # ★fresh ラン(load を通らない)では 1 バイトも影響しない = straight は完全不変。
        # ★★この行は **__init__ の最後の文でなければならない**(後ろに L1 を出す処理を
        #   足すとその分が印から漏れる)。漏れは
        #   tests/test_resume.py::test_init_event_mark_counts_the_startup_segment
        #   が `_init_event_mark == len(logger.events)` で検出する = トリップワイヤー。
        self._init_event_mark = len(self.logger.events)

    # ------------------------------------------------- 在場述語(present predicate)
    # ★なぜ要るか: ``agent_by_id`` は「これまで実体化した**全**個体」の索引で、プール回転で
    #   街を出た個体を**意図的に消さない**(過去の参照 = 造語の作者名・DM 送信者名・関係台帳が
    #   退場後も解決できるように)。したがって ``agent_by_id.get(aid) is None`` は
    #   **決して真にならない**。「街から居なくなった」の判定にそれを使っていた分岐は全て死んで
    #   おり、脱水済み(detached)のオブジェクトへ書き込んでいた = 書いた値は次の hydrate
    #   (退場時スナップショット)で捨てられる。金を動かす経路ではこれが**保存則の破れ**になる。
    # ★正典の参照実装は ``medical._agent_by_id``(sim.agents 走査)だが O(N) なので、
    #   ここでは id 集合を持って O(1) にする(25万体で毎 step 引かれる)。
    # ★索引の維持: ``self.agents`` を差し替える地点は 4 つしかない(構築 / プール回転 /
    #   checkpoint.load / resume 復元)。回転と resume では明示的に無効化し、それ以外は
    #   「要素数が変わったら作り直す」遅延検査で拾う(checkpoint.load は本体不触の掟がある)。
    def invalidate_present_index(self) -> None:
        """在場索引を捨てる(``self.agents`` を差し替えた直後に呼ぶ)。"""
        self._present_ids = None
        self._follow_cands = None        # A1 の初期フォロー母集団(同じ寿命)

    def _present_ids_view(self) -> set:
        ids = getattr(self, "_present_ids", None)
        if ids is None or len(ids) != len(self.agents):
            ids = {int(a.id) for a in self.agents}
            self._present_ids = ids
        return ids

    def is_present(self, agent_id) -> bool:
        """その id の個体が**いま街に居る**(= ``sim.agents`` に居る)か。O(1)。"""
        return int(agent_id) in self._present_ids_view()

    def present_agent(self, agent_id):
        """在場なら Agent、不在(退場 / 未実体化)なら None。O(1)。

        ★``agent_by_id.get`` の**正しい置き換え**。返るのは常に present な実体なので、
          ここに書いた値は次の dehydrate で退避され、再来街時に持ち越される。
        """
        aid = int(agent_id)
        if aid < 0 or aid not in self._present_ids_view():
            return None
        return self.agent_by_id.get(aid)

    # ------------------------------------------------------ LLM ジャーナル(第71)
    # ObserverLogger の flush_segment / checkpoint と同じ「checkpoint が真の境界」流儀で、
    # 確定点(records/bytes)を checkpoint に載せ、resume 時にそこまで巻き戻す。
    # これで「checkpoint 後に走ってクラッシュした分」が再走で二重記録にならない。
    def _journal_marks(self) -> dict:
        """checkpoint に載せる確定点(ジャーナルファイル名 → {records, bytes})。"""
        return {j.path.name: j.mark() for j in getattr(self, "_journals", [])}

    def _journal_rewind(self, marks: dict | None) -> None:
        if not marks:
            return
        for j in getattr(self, "_journals", []):
            j.rewind(marks.get(j.path.name))

    def _journal_close(self) -> None:
        for j in getattr(self, "_journals", []):
            j.close()

    def _build_router_child(self, spec: dict):
        """router の子バックエンドを1つ構築する(第23バッチ M2)。

        spec = {backend: mock|ollama|vllm|openai_compat|anthropic, name: モデル名,
                base_url/host/api_key_env/timeout_s は任意}。
        APIキーは環境変数名(api_key_env)のみを渡す=キーの値は config にもログにも置かない。
        """
        b = str(spec.get("backend", ""))
        name = str(spec.get("name", ""))
        timeout_s = float(spec.get("timeout_s", 120.0))
        # 1 呼の絶対時限(M-1): 子 spec で個別指定でき、無ければ model.call_deadline_s。
        deadline_s = float(spec.get("call_deadline_s",
                                    self.cfg.model.get("call_deadline_s", 300.0)))
        if b == "mock":
            # 第88: name を渡すと mock サブモデル(mock:a / mock:b …)になる。
            # 名前なし(従来の router 子指定)は "mock" のまま = 既存ランとバイト一致。
            return MockBackend(self.hub, name=name or None)
        if b == "ollama":
            from ..llm.ollama import OllamaBackend
            return OllamaBackend(name, host=str(spec.get("host", "http://localhost:11434")),
                                 timeout_s=timeout_s,
                                 format_mode=str(spec.get("format", "json")),
                                 deadline_s=deadline_s)
        if b == "vllm":
            from ..llm.vllm import VllmBackend
            return VllmBackend(name, str(spec.get("base_url", "http://localhost:8000")),
                               timeout_s=timeout_s,
                               format_mode=str(spec.get("format", "json")),
                               deadline_s=deadline_s,
                               # β11: router / mind の子も同じノブに従う(既定 None=不変)
                               request_seed=getattr(self, "llm_request_seed", None))
        if b == "openai_compat":
            from ..llm.openai_compat import OpenAICompatBackend
            return OpenAICompatBackend(
                name, base_url=str(spec.get("base_url", "https://api.openai.com/v1")),
                api_key_env=str(spec.get("api_key_env", "OPENAI_API_KEY")),
                timeout_s=timeout_s, deadline_s=deadline_s)
        if b == "anthropic":
            from ..llm.anthropic import AnthropicBackend
            return AnthropicBackend(
                name, api_key_env=str(spec.get("api_key_env", "ANTHROPIC_API_KEY")),
                base_url=str(spec.get("base_url", "https://api.anthropic.com")),
                timeout_s=timeout_s, deadline_s=deadline_s)
        raise ValueError(f"router の子 backend '{b}' は未対応"
                         "(mock | ollama | vllm | openai_compat | anthropic)。")

    def _inflow_pulse_arrivals(self) -> dict | None:
        """駅到着のパルス量子化に使う「定常日の到着表」。既定 OFF は **None**(= 完全 no-op)。

        conf: world.inflow_pulse.enabled(既定 false)。ON でも読むのは既にロード済みの
        ダイヤ(self.transit)だけで、ファイル I/O も乱数も LLM も足さない。到着表は
        **1 ラン 1 回**だけ組んで自身に保持し(pool の日次ローテーションが 1 体ごとに
        呼ぶため)、以降は二分探索で引く(O(log n_arrivals) / 体)。
        分解能の限界は Transit.arrivals_between の docstring(等間隔の再構成)に明記。

        返り値 = {"all": 全路線の合併集合, "by_line": {路線名: その路線だけの到着}}。
        **沿線別**に持つのが本質: 人は 1 本の路線に乗って来るので、量子化の相手は
        その人の路線のダイヤであって全路線の合併ではない(合併は 9 路線 × 3〜6 分間隔で
        ほぼ毎分に立つため、量子化しても塊が 1 つもできない = 実測済み)。
        """
        if hasattr(self, "_pulse_arrivals"):
            return self._pulse_arrivals
        table: dict | None = None
        pcfg = self.cfg.world.get("inflow_pulse", {}) or {}
        if bool(pcfg.get("enabled", False)):
            allarr = self.transit.arrivals_of_day()
            if allarr:
                by_line: dict[str, list[tuple[int, str]]] = {}
                for item in allarr:
                    by_line.setdefault(item[1], []).append(item)
                table = {"all": allarr, "by_line": by_line}
        self._pulse_arrivals = table
        return table

    def _snap_agent_arrival(self, agent) -> None:
        """流入通勤者 1 体の arrival_min を時刻表の到着へ量子化する(既定 OFF = 即 return)。

        **冪等**(何度呼んでも結果が同じ)なので、通常の流入初期化と pool の
        日次ローテーション(build_pool_agent)の両方から安全に呼べる: 2 回目は既に
        ダイヤ上に乗った時刻を渡すことになり、規則(1) の「ちょうど一致」で同じ値へ戻る。

        路線の決め方: 名簿の residence_line(居住エリアのラベル)を
        Transit.line_for_residence で積んでいるダイヤの路線名へ照合し、当たればその
        **路線のダイヤだけ**へスナップする。当たらない(路線を名指さない居住ラベル)なら
        全路線の合併集合へ落とす(誤った路線へ結ぶより安全側)。照合規則と実データでの
        当たり方は Transit.line_for_residence の docstring に明記。

        対象は駅ゲートウェイの流入通勤者だけ。縁ゲートウェイ(徒歩流入)・非通勤者・
        arrival_min 未設定(-1)は 1 バイトも触らない。乱数ゼロ・個体順に非依存。
        """
        pulse = self._inflow_pulse_arrivals()
        if pulse is None:
            return
        if not getattr(agent, "commute", False) or agent.arrival_min < 0:
            return
        if agent.commute_gateway != "station" or not self.city.station_node:
            return
        line = self.transit.line_for_residence(
            str(getattr(agent, "residence_line", "") or ""))
        table = pulse["by_line"].get(line) if line else None
        snapped = _snap_to_arrival(table or pulse["all"], agent.arrival_min)
        if snapped is None:
            return
        # 属性は ON のときだけ生やす(改札装置と同じ流儀)。観測・検証用で世界の物理には
        # 使わない(arrival_min が唯一の効き口)。pulse_line_src = 沿線照合の出自。
        agent.pulse_train_min, agent.pulse_line = snapped
        agent.pulse_line_src = "residence" if line else "union"
        agent.arrival_min = int(snapped[0])

    def _init_inflow_commuters(self) -> None:
        """流入通勤者(commute=True)を朝の到着前=範囲外(loc=outside)に置き、朝の到着 step
        に既存の帰還機構(_phase_wake_and_returns)で enter_area させる。到着口は
        build_personas が配分した commute_gateway("station"~85-90% / "edge")で解決する。

        すべて決定論(乱数を引かない=既存 stream の draw 順に影響しない)。commuter が
        居ない名簿(既定・procedural、既存 personas_40/80)では 1件も該当せず完全 no-op
        =既定 config・既定地図での挙動バイト一致(docs/research/shibuya-inflow.md)。

        ★駅到着のパルス量子化(world.inflow_pulse.enabled・既定 false)= actor model Wave 3。
        ON のときだけ、**駅ゲートウェイの流入通勤者の arrival_min をその人の路線の
        時刻表へスナップ**する(_snap_agent_arrival)。二峰分布の draw そのものには一切
        触らない(persona 側の値・stream・消費順は 1 ビットも動かさない)= 量子化だけが
        新しい。arrival_min は初日の return_at と、2 日目以降の
        `_phase_wake_and_returns`(`_steps_until_tod(sim_min, arrival_min)`)の**両方**が
        読むので、ここで 1 回書き換えれば全日がパルス化される。
        縁ゲートウェイ(commute_gateway=="edge")は徒歩流入なので**触らない**。
        OFF は _snap_agent_arrival が即 return = 1 体も触らない(属性すら生えない)。"""
        start_min = self.clock.start_min                 # day0 開始時刻(=07:00)
        for agent in self.agents:
            if not getattr(agent, "commute", False):
                continue
            self._snap_agent_arrival(agent)              # OFF は即 return(no-op)
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
            # A9(第94バッチ OBS-U2 で新規発見・設計調査 §1.1 の A 級表に**無かった** 9 件目):
            # 分 → step の 10 直書き。A1(_steps_until_tod)と同型で、Δt=1 だと流入通勤者の
            # **初日の到着**だけが実時間 1/10 で早まる(2 日目以降は A1 の経路)。
            # clock.min_to_steps は `int(分) // step_minutes` なので Δt=10 で厳密同値。
            agent.return_at = max(0, self.clock.min_to_steps(arr_min - start_min))

    def _apply_natural_start(self) -> None:
        """初日コールドスタート改善(run.natural_start・既定 false=バイト一致)。

        既定は全員 sleeping=False で開始するため、開始時刻が深夜でも起きたまま過ごし初日の朝の
        起床(wake_up)と朝計画(day_plan)がほぼ発火しない(2日目以降は正常)。ON 時は run 開始
        時刻が本人の自然な睡眠時間帯に入っている居住者を、就寝状態・自宅(home_node/home_building)・
        自然な起床 step で着席させる。self.agents を1回走査するので通常経路の init ループと pool
        経路の day0 着席の両方をまとめて覆う(2日目以降の途中日 enter には掛からない=day0 のみ)。

        起床は既存 _phase_wake_and_returns の wake_up→_schedule_plan にそのまま乗る(新しい起床
        経路を作らない=朝計画が初日から全員分出る)。判定・起床 step は bedtime_min / sleep_steps /
        開始時刻からの決定論導出(_natural_wake_step。乱数を一切引かない=stream 不使用=バイト一致
        の前提)。開始時に覚醒しているべき個体(夜勤者を含む)は従来どおり覚醒で着席。来街者・流入
        通勤者(自宅が圏外)は対象外(初日入場は presence/流入機構の管轄)。既定 OFF は 1体も触らない。"""
        if not bool(self.cfg.run.get("natural_start", False)):
            return
        start_tod = self.clock.sim_min(0) % 1440           # 開始時刻(分 of day。通常 07:00→420)
        for agent in self.agents:
            if agent.visitor or agent.commute:             # 自宅が圏外 = 対象外
                continue
            wake_step = _natural_wake_step(start_tod, agent.bedtime_min,
                                           agent.sleep_steps,
                                           self.clock.step_minutes)
            if wake_step is None:                          # 覚醒しているべき→従来どおり覚醒で着席
                continue
            home = agent.home_node or ""
            if not home and not agent.home_building:        # 帰る家が無い(念のため)=触らない
                continue
            agent.sleeping = True
            agent.sleep_until = wake_step                   # step0 起点なので step 番号そのもの
            agent.node = home or agent.node
            if agent.home_building and self.city.has_building(agent.home_building):
                bld = self.city.building(agent.home_building)
                agent.building = bld["id"]
                # 3D-U0(world.floor_clamp・既定 OFF=そのまま): go_to_bed と同じ丸め方をする
                agent.floor = _floors_mod.clamp(self, bld, agent.home_floor)
                agent.x, agent.y = bld["centroid"]         # 実在の住宅建物内で就寝(go_to_bed と同じ位置)
            else:
                agent.building = None
                agent.floor = 0
                agent.x, agent.y = self.city.node_xy(agent.node)

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

    # ---- エージェント生成時のランタイム属性初期化(名簿経路とプール経路で共有)----
    def _init_agent_runtime(self, agent) -> None:
        """build_agent が返した agent に step ループが要る transient 属性を据える。

        名簿経路(既定)と pool ローテーション経路の両方から呼ぶ共通初期化。既定挙動は
        従来の init ループと**完全に同一**(抽出リファクタのみ・draw/値とも不変=バイト一致)。"""
        agent.x, agent.y = self.city.node_xy(agent.node)
        agent.visits[agent.node] += 1
        agent._heard_unknown = False
        agent._arrived_new = False
        agent._congestion = 1.0
        agent._pending_stay = self.clock.dur_steps(2)
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
        # 中央 Δt(第79バッチ): recency_decay は「毎 step の残存割合」= KEEP 類。
        # 既定 Δt=10 では scale_keep が恒等(この行自体を通らない)=バイト一致。
        if self.dt_min != _timeconv.CANON_DT_MIN:
            agent.mem.recency_decay = _timeconv.scale_keep(agent.mem.recency_decay,
                                                           self.dt_min)
            # A7(第94バッチ OBS-U2): ACT-R 基礎活性化の冪乗則忘却も実時間の関数。
            # recency_decay と同じ block で配線する(= Δt=10 ではこの行を通らない)。
            agent.mem.dt_min = self.dt_min
        if self.actrcfg.get("enabled", False):
            # 個体別シードは run.seed から決定論導出(専用streamのみで消費=既存draw順不変)
            overrides = {k: v for k, v in self.actrcfg.items()
                         if k not in ("enabled", "seed")}
            base = int(self.actrcfg.get("seed", self.cfg.run.seed))
            agent.mem.actr = agent.mem.actr_config(
                seed=base * 1_000_003 + agent.id, **overrides)
        # 第89: プラセボ L1 への結線(既定 OFF は属性を 1 つも生やさない=no-op)。
        # 名簿経路と pool ローテーション経路の**両方**がここを通るので、日境界で実体化される
        # 個体(hydrate 再入場)も漏れなく結線される。
        from .. import ablate as _ablate_placebo
        _ablate_placebo.attach_agent(self, agent)

    # ---- 群のオントロジー(文化圏×経験の世界観共有群。2026-07-21)------------------
    def _apply_ontology(self, agent, tier: str, party_size=None) -> None:
        """群を安定ハッシュで割り当て、経験1行と worldview 初期オフセットを据える(既定 OFF=no-op)。

        既定 OFF(ontology.enabled=false)は属性を1つも作らない=プロンプト・状態とも不変=バイト一致。
        割当は persona id(pool は pool_pid・直接ランは agent.id)のみの関数=k/trait 非参照・run.seed
        非依存・RngHub 無風。pool の再来街(hydrate)でも build_pool_agent が同一 pid から同一群を再導出。
        party_size(プール record の同行人数。無い個体は None)は同行者構成軸のデータ駆動割当に使う。"""
        from .. import ontology as _ontology_mod
        cfg = self.ontologycfg
        if not cfg["enabled"]:
            return
        pid = getattr(agent, "pool_pid", None)
        if pid is None:
            pid = str(agent.id)
        gid = _ontology_mod.assign_group(cfg, pid, tier)
        if gid is None:
            return
        agent.ontology_group = gid
        line = _ontology_mod.group_line(cfg, gid)
        if line:                                       # 「経験の事実」1行(build_prompt が注入)
            agent.ontology_line = line
        c0 = _ontology_mod.initial_controllability(cfg, gid)
        if c0 is not None:                             # S-b: 可制御性の起点を群で1回だけずらす
            agent.controllability = c0
        # ---- 直交する第2軸以降(方式B)+ 訓練経験行(方式A)。既定は axes 空・drill OFF=完全 no-op。
        # 各軸は独立ハッシュ(seed+seed_offset)で文化圏軸と無相関=直交。age(人口統計 P0)がある時は
        # 年代条件付き構成比を使う(traits/k は読まない=R1)。同行者構成軸は準静的=(pid, day) 純関数
        # (day は日境界ローテーションで進む _pool_day。直接ラン/day0 は 0)+ party_size 由来はデータ駆動
        # で確定(ハッシュ不要)。追加 LLM 呼・乱数はゼロ(hashlib 自前=RngHub 無風)。
        age = getattr(agent, "age", None)
        day = int(getattr(self, "_pool_day", 0))
        axis_groups: dict = {}
        axis_lines: list = []
        for axis_name in sorted(cfg.get("axes") or {}):
            ag = _ontology_mod.assign_axis(cfg, axis_name, pid, tier, age,
                                           day=day, party_size=party_size)
            if ag is None:
                continue
            axis_groups[axis_name] = ag
            al = _ontology_mod.axis_line(cfg, axis_name, ag)
            if al:
                axis_lines.append(al)
        dl = _ontology_mod.group_drill_line(cfg, gid)  # 方式A: 訓練経験を別行で付加
        if dl:
            axis_lines.append(dl)
        if axis_groups:                                # 観測(agents.json)用の軸別群 id
            agent.ontology_axes = axis_groups
        if axis_lines:                                 # build_prompt が文化圏行に続けて注入
            agent.ontology_axis_lines = axis_lines

    # ---- 日次ローテーション/presence(W2 P3)------------------------------------
    def _pool_weekday(self, day: int) -> int:
        """当日の曜日(0=Mon..6=Sun)。presence を暦 config に結合させず day のみの純関数に保つ
        ため day % 7(day0=Monday)で決める(k 非依存・resume 不変)。"""
        return int(day) % 7

    def _pool_rain(self, day: int) -> bool | None:
        """在場の天候共変量(PRES-A1②)。**generated 以外は None = 不活性**。

        - habit.weather が OFF / habit そのものが OFF なら覗きにも行かない(no-op)。
        - 覗くのは `weather.peek_bad_day`(副作用ゼロ。summary の generated 件数を汚さない)。
        ★暦は使うが presence の weekday は使わない(weekday は day%7 の純関数のまま)=
          「暦の曜日」と「presence の曜日」は別物という現行の割り切りを 1 行も変えない。
        """
        habit = getattr(self, "_pool_presence", None)
        if not habit or not habit["habit"]["enabled"] or not habit["habit"]["weather"]:
            return None
        from .. import weather as _weather_mod2
        return _weather_mod2.peek_bad_day(self, int(day))

    def build_pool_agent(self, pid: str, record: dict):
        """P5 の full record から present エージェントを1体構築する(id はペルソナ id 安定)。

        agent.id = self._pool.id_of(pid)(密で安定=日を跨いで同一人物=k*/founder 分析の前提)。
        元のペルソナ id 文字列は agent.pool_pid に保持する(観測 agent_id からの逆引き・突合に使う)。"""
        aid = self._pool.id_of(pid)
        agent = build_agent(aid, self.hub.stream("persona", aid),
                            self._pool_nodes, self.city, entry=record,
                            economy=self.economy,
                            threshold_dist=self.threshold_dist,
                            drift=self.drivecfg["drift"],
                            reflection=self.reflectcfg,
                            place_name=self.place_name,
                            flat=self.flatcfg,
                            dt_min=self.dt_min)
        agent.pool_pid = pid
        # 第114 レーン 1b: 名簿の is_foreign / visit_purpose を個体へ控える(既定 OFF=
        # society_diversity.record_driven が false=属性を 1 つも生やさない)。★ここに置くのは
        # day0 の着席では下の _init_pool_agent_extras が早期 return するため(あちらだけに
        # 置くと day0 の在場者が台帳駆動から漏れる)。冪等なので再入場で二度通っても不変。
        from .. import diversity as _diversity_rec
        _diversity_rec.note_record(self, agent, record)
        # 駅到着のパルス量子化(既定 OFF=即 return=no-op)。★pool の日次ローテーションで
        # 途中入場する個体は _init_inflow_commuters(起動時 1 回)を通らないので、ここで
        # 同じ純関数を通す(通さないと pool ON のランだけ流入が平滑に戻る=素材の取りこぼし。
        # persona_pool.json は 500 record 中 145 が commute=true)。冪等なので day0 着席で
        # 二重に呼ばれても値は動かない。
        self._snap_agent_arrival(agent)
        # party_size(L4 来街者 record に実在=同行人数)を保持(S-R5 party の実体化が読む。無ければ None)。
        agent.party_size = record.get("party_size")
        self._init_agent_runtime(agent)
        # 群のオントロジー(既定 OFF=no-op)。tier=P5 record の presence 層(resident/duty/…)。
        # party_size(L4 来街者の record に実在=同行人数)は同行者構成軸のデータ駆動割当に使う(無ければ None)。
        self._apply_ontology(agent, str(record.get("presence", "resident")),
                             party_size=record.get("party_size"))
        # 第88: 心のモデルと知能層を誕生時に固定(既定 OFF=属性を生やさない=no-op)。
        # 割当は (master_seed, agent.id) の純関数なので、pool の再入場(hydrate)でも同一。
        from .. import mind as _mind_mod
        _mind_mod.assign(self, agent)
        # 組織への配属(第109 レーン ORG。既定 OFF=_pool_org_book is None=no-op)。
        # ★職場束ね直しの**前**に置く: organizations の束縛は台帳 node をそのまま採るが、
        #   work.bind_workplace は「その node が実際にその業種の POI を持つか」まで見て
        #   実在の職場へ着地させる(work.py §_resolve_node ①②)ので、後勝ちが正しい。
        self._attach_pool_org(agent, record)
        # 職場束ね直し(work.bind_workplace。既定 OFF は no-op=work_node の付与状況とも不変)。
        self._bind_pool_workplace(agent, record)
        # 賃金多様性 WAGE(第112。既定 OFF=no-op=agent.wage も属性も 1 つも動かない)。
        # ★**職場束ね直しの後**に置く: 判定条件の 1 つが「勤務窓を持つか」で、pool の L2 は
        #   occupation が persona._WORK_CAT に載らないため勤務窓を持つのは bind の後だけ
        #   (前に置くと 224,240 人が丸ごと対象外になる)。org 配属(industry/規模帯の出所)も
        #   既に済んでいるので、ここが台帳情報の揃う唯一の点になる。
        scheduler.wage_assign(self, agent, record)
        # 起動時 1 回の配布を「入場駆動」へ(レーン乙 ブロック2。既定 OFF の機構は no-op)。
        self._init_pool_agent_extras(agent, record)
        # 存在の内生化 POP(既定 OFF=即 return=属性を 1 つも生やさない)。**名簿の record は
        # 1 バイトも書き換えない**ので、「定着して住民になった」「転出して街を去った」という
        # ラン内の事実は台帳側にあり、入場のたびにここで冪等に着せ直す(household.pool_bind と同型)。
        from .. import population as _population_mod
        _population_mod.apply_on_entry(self, agent, record)
        return agent

    # ---------------------------------------------- 起動時1回の配布 → 入場駆動(ブロック2)
    # ★なぜ要るか: 下の族はどれも ``__init__`` の一本道で「その瞬間 sim.agents に居た個体」
    #   にだけ配られる。プール回転で day1 以降に入場する個体(finals 構成で 20.9 万人)は
    #   **一度も配布を受けない** = SNS を持たない・顔なじみが 0・SDT/needs/価値/入力解像度の
    #   個人差が無い・観光客にならない・長期目標を持たない、という別種の住民になっていた。
    # ★規律: (a) 冪等(同じ個体を二度通しても値が動かない) (b) 既存の draw 列を 1 粒も
    #   動かさない(新規抽選は**個体キーの新 named stream**か純関数ハッシュのみ。RngHub.stream
    #   は呼ぶたびに新しい Generator を派生するので、引いても他の列に影響しない)
    #   (c) 当該機構が OFF なら 1 行も通らない = 既定プロファイルはバイト一致。
    def _init_pool_agent_extras(self, agent, record: dict) -> None:
        from .. import diversity as _diversity_mod
        from .. import household as _household_mod
        from .. import inner_life as _inner_life_mod
        from ..factors import psych as psych_mod
        # ★day0 の着席では 1 行も通さない: ``__init__`` はこの後で**同じ配布を名簿全体へ**
        #   一括で行う(SNS init_follows / 顔なじみ / SDT / needs / 価値 / LOD / 世帯 /
        #   観光 / inner_life)。ここで先回りすると二重配布になるうえ、まだ ``agent_by_id``
        #   すら組まれていない(在場述語が引けない)。``_init_event_mark`` は __init__ の
        #   **最後の文**で据わる印なので、これが在る = 構築が終わっている、と読める。
        if getattr(self, "_init_event_mark", None) is None:
            return
        aid = int(agent.id)
        # A1 SNS: フォローと DM 可能リストを持たせる。``Internet.add_contact`` は
        #   「双方が contacts に居る」ときしか効かない無言 no-op なので、これが無いと
        #   途中入場者は**永久に DM を受け取れない**(会話しても contacts に載らない)。
        self.net.ensure(aid, self.hub.stream("follows_entry", aid),
                        k=int(self.cfg.get("net", {}).get("follow_k", 6)),
                        candidates=self._pool_follow_candidates())
        # A2 顔なじみ: 同じ住居/職場の現行グループへ接続(起動時ブロックと同じ「1人あたり最大3人」)。
        self._link_colocated(agent)
        # A4 SDT / A5 集団効力感 / A6 欲求プロファイル / A7 価値の充足 / A8 入力解像度LOD
        if self.psychcfg["sdt"]["enabled"] and not agent.drive_mods:
            from ..factors.registry import sdt_params
            agent.drive_mods = sdt_params(agent.traits, self.hub.stream("sdt", aid),
                                          self.psychcfg["sdt"])
        if self.psychcfg["collective"]["enabled"]:
            # ★冪等判定は factors 層(ensure_collective)に置く: 因子名を engine に
            #   書くのは no-fingerprint 契約(tests/test_contracts.py)違反。
            psych_mod.ensure_collective(agent, self.hub.stream("collective", aid))
        needscfg = getattr(self, "needscfg", None)
        if needscfg and needscfg["enabled"] and not getattr(agent, "needs_mods", None):
            from ..needs import build_mods as _needs_build_mods
            agent.needs_mods = _needs_build_mods(agent, self.hub.stream("needs", aid),
                                                 needscfg)
        freedomcfg = getattr(self, "freedomcfg", None)
        if freedomcfg and freedomcfg["open_actions"] and not getattr(agent, "sat", None):
            from .. import values as _values_mod
            agent.sat = {t: 0.5 for t in _values_mod.TAGS}
            _values_mod._refresh_mods(agent, freedomcfg)
        if getattr(self, "inputrescfg", None) and not getattr(agent, "input_res", None):
            from ..cognition import lod as _lod_mod
            lv = _lod_mod.assign_axis([aid], self.hub.stream("lod_input_res", aid),
                                      self.inputrescfg["levels"])[aid]
            spec = self.inputrescfg["levels"][lv]
            agent.input_res = {"level": lv,
                               **{k: v for k, v in spec.items() if k != "share"}}
        # A9 世帯: pool 名簿由来の世帯へ冪等に着席(既定 OFF=no-op。household.pool_bind)。
        _household_mod.bind_pool_household(self, agent, record)
        # A10 観光・多言語: 個体ハッシュの決定論割当(day0 列挙順の前方切りではない)。
        # 第114 レーン 1b: record を渡す(record_driven ON なら台帳の is_foreign /
        # visit_purpose が比率抽選に優先する。OFF は record を 1 バイトも読まない)。
        _diversity_mod.assign_for_entry(self, agent, record)
        # A11 内面: 長期目標・趣味(起動時 precompute と同じ純関数。L1 は増やさない)。
        _inner_life_mod.assign_for_entry(self, agent)

    def _pool_follow_candidates(self) -> list:
        """A1 の初期フォロー母集団(いま街に居る個体の id 昇順)。決定論。

        ``init_follows`` が起動時に使う母集団(= その時点の全個体)と同じ意味。回転の途中で
        呼ばれるので「前日の在場者」を見るが、それも決定論なので resume で同一に再現される。
        ★1 回の回転で 2 万人が入場するので、**1 回の回転につき 1 度だけ**組んで使い回す
          (毎回 25 万件を並べ直すと O(入場者 × 在場者)になる)。在場索引と同じ寿命なので
          ``invalidate_present_index`` で一緒に捨てる。"""
        cands = getattr(self, "_follow_cands", None)
        if cands is None:
            cands = sorted(int(a.id) for a in self.agents)
            self._follow_cands = cands
        return cands

    def _link_colocated(self, agent) -> None:
        """A2: 同じ住居建物 / 職場建物の**在場**メンバへ顔なじみを張る(1 人あたり最大 3 人)。

        起動時の「顔なじみ」ブロックと同じ規則。索引は building → 参入順の id 列で持ち、
        入場のたびに末尾へ足す(参入順 = 決定論)。相手は在場述語で絞る(退場者へ書かない)。"""
        idx = getattr(self, "_acq_index", None)
        if idx is None:
            from collections import defaultdict as _dd
            idx = _dd(list)
            for a in self.agents:
                if a.home_building:
                    idx[("home", a.home_building)].append(int(a.id))
                if a.work_building:
                    idx[("work", a.work_building)].append(int(a.id))
            self._acq_index = idx
        aid = int(agent.id)
        for kind, bld in (("home", agent.home_building), ("work", agent.work_building)):
            if not bld:
                continue
            bucket = idx[(kind, bld)]
            n = 0
            for oid in reversed(bucket):           # 直近に入った同建物の面々から 3 人まで
                if n >= 3:
                    break
                if oid == aid:
                    continue
                b = self.present_agent(oid)
                if b is None:
                    continue
                n += 1
                self.net.add_contact(aid, oid)
                agent.mem.record_contact(oid, b.name, 0, "顔なじみ")
                b.mem.record_contact(aid, agent.name, 0, "顔なじみ")
            if aid not in bucket:
                bucket.append(aid)

    def _attach_pool_org(self, agent, record: dict) -> None:
        """pool record の org_id/role で org へ配属する(ON 時のみ・決定論・乱数ゼロ)。

        既定 OFF(_pool_org_book is None = pool OFF または organizations OFF)は一切触らない
        =org_id は付かない=従来と完全同一。ON 時は day0 着席・日境界ローテーション・hydrate
        再入の各実体化で同一 pool_pid が同一 org へ配属される(run.seed 非依存=resume 不変)。
        career の解雇/転職(lay_off / switch_org)がその後に org_id を書き換えるのは
        名簿経路と同じで、**再入場のたびに元の配属へ戻る**点も hydrate の既存仕様どおり
        (dehydrate/hydrate は org_id を運ばない=名簿由来の属性は record から作り直す)。"""
        book = self._pool_org_book
        if book is None:
            return
        from .. import organizations as _orgs_mod
        self._pool_org_stat["n_present"] += 1
        if _orgs_mod.attach_record(agent, record, book, city=self.city,
                                   commute_to_poi=self._pool_org_commute):
            self._pool_org_stat["n_attached"] += 1

    def _bind_pool_workplace(self, agent, record: dict) -> None:
        """pool 個体の work_node を台帳 workplace_poi へ束ね直す(ON 時のみ・決定論・乱数ゼロ)。

        既定 OFF(_workbind_book is None)は一切触らない=work_node の付与状況・イベント列とも
        バイト一致。ON 時は build_pool_agent(day0 着席 + 日境界ローテーション + hydrate 再入)の
        各実体化で同一 pool_pid が同一 work_node へ束ねられる(run.seed 非依存=resume 不変)。"""
        if self._workbind_book is None:
            return
        from .. import work as _work_mod
        if not _work_mod.bind_eligible(record, self._workbind_cfg):
            return                                 # 勤務地を持たない層(L1 の無職・L4 来街者)は対象外
        had, has, how = _work_mod.bind_workplace(agent, record, self.city,
                                                 self._workbind_book, self._workbind_cfg)
        agg = self._workbind_agg
        agg["n_total"] += 1
        if not had:
            agg["n_unbound_before"] += 1
        if not has:
            agg["n_unbound_after"] += 1
        if has and not had:
            agg["n_bound"] += 1
        # 束ね手段/不能理由の内訳(**L1 には出さない**=observer/schema.py の workplace_bound は
        # 4 列固定の契約。検収・診断が sim から直接読むための実行時カウンタ)。層は pool record の
        # layer(L1..L5)。決定論=イベント列にも乱数にも一切触れない。
        det = self._workbind_detail
        det["how"][how] = det["how"].get(how, 0) + 1
        lay = str(record.get("layer") or "?")
        det["by_layer"].setdefault(lay, {"n": 0, "unbound_after": 0})
        det["by_layer"][lay]["n"] += 1
        if not has:
            det["by_layer"][lay]["unbound_after"] += 1
            occ = str(record.get("occupation") or record.get("role") or "?")
            det["unresolved_occ"][occ] = det["unresolved_occ"].get(occ, 0) + 1

    def _pool_update_budget(self) -> None:
        """S6a の N 比例 cap を、pool ON 時は当日の在場数 len(self.agents) で更新する(1行の接続)。

        n_proportional OFF のときは何もしない(固定 lod.max_llm_per_step のまま=バイト一致)。"""
        if self._pool is None:
            return
        npro = self.cfg.lod.get("n_proportional", None)
        if npro is not None and bool(npro.get("enabled", False)):
            dens = float(npro.get("density", 0.15))
            self.budget.max_per_step = max(1, math.ceil(round(dens * len(self.agents), 9)))

    def _pool_sidecar_path(self, step: int) -> Path:
        return self.out_dir / "checkpoint" / f"dormant-{int(step):06d}.pkl.gz"

    def _save_pool_sidecar(self, step: int) -> None:
        """checkpoint と対で、pool の状態を同 step の sidecar に保存する(pool ON 時のみ)。

        checkpoint.py 本体は触らず(掟)、pool 固有の状態だけを持たせる:
          - ドーマント退避ストア(再来街の記憶源)= リッチ層 + **恒久台帳 vital**(A2)。
          - 日境界の進行(_pool_day)。
          - **退場済み個体**(agent_by_id にあり sim.agents に無い個体)= 造語の作者名・DM 送信者名等
            の**過去参照**の解決先。checkpoint は present 個体しか保存しないので、ここで補完する。
            ★A1 以後、ここに載るのは軽量参照(``agents/ref.AgentRef``)= サイドカーも自然に縮む。
        presence 自体は純関数=保存不要(day から再計算)。"""
        if self._pool is None:
            return
        import gzip
        import os
        import pickle
        present_ids = {a.id for a in self.agents}
        departed = {aid: _to_ref(a) for aid, a in self.agent_by_id.items()
                    if aid not in present_ids}
        blob = {"pool_day": int(self._pool_day),
                "dormant": self._dormant._d,
                # ★A2: 恒久台帳(金銭・債権・人口会計)。旧サイドカーには無いので
                #   復元側は欠落を許容し、rich から作り直す(rebuild_vital)。
                "dormant_vital": self._dormant._vital,
                "dormant_cap": self._dormant.cap,
                "departed": departed}
        path = self._pool_sidecar_path(step)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        # ★β7(第117): checkpoint.save と**同じ理由・同じ書き方**にする。ここは pool ON の
        #   本選で checkpoint と対で必ず走る双子の writer で、`departed`(退場済み個体の実体)と
        #   ドーマントストアを丸ごと運ぶため、25 万規模では本体と同オーダーの一過性ピークを
        #   作っていた(`pickle.dumps` の中間バイト列 + 圧縮バッファ)。gzip ストリームへ直接
        #   dump すれば中間 bytes が消える。共有参照の保証は「1 回のシリアライズ」なので不変。
        #   gzip の trailer は close で書かれるので fsync は **gzip を閉じた後**に掛ける。
        with open(tmp, "wb") as fh:
            with gzip.GzipFile(fileobj=fh, mode="wb") as gz:
                pickle.dump(blob, gz, protocol=pickle.HIGHEST_PROTOCOL)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)

    def _read_pool_sidecar(self, step: int) -> dict:
        """pool サイドカーを読む(壊れていれば例外を投げる)。A6 の候補検証と復元の共通口。"""
        import gzip
        import pickle
        with gzip.open(self._pool_sidecar_path(step), "rb") as f:
            blob = pickle.load(f)
        if not isinstance(blob, dict):
            raise ValueError(f"pool sidecar の中身が dict でない: {type(blob).__name__}")
        return blob

    def _pool_sidecar_defect(self, step: int) -> str | None:
        """その世代の pool サイドカーの欠陥を述べる(健全なら None。読めた blob は握っておく)。

        ★A6(第118 レーンC): pool ON では **本体 + サイドカー**で 1 世代なので、
          サイドカーが欠落/破損している世代は「完成していない」= resume が掴んではいけない。
          ここで実際に読んで検証し、成功した blob を `_pool_sidecar_blob` に残す
          (= 採用世代のサイドカーはこの後 1 度も読み直さない = 読み込みは 1 回のまま)。
        """
        path = self._pool_sidecar_path(step)
        self._pool_sidecar_blob = None
        if not path.exists():
            return "pool sidecar 欠落"
        try:
            self._pool_sidecar_blob = (int(step), self._read_pool_sidecar(step))
        except Exception as e:                      # gzip 破損 / 途中終端 / pickle 不整合
            self._pool_sidecar_blob = None
            return f"pool sidecar 破損({type(e).__name__})"
        return None

    def _pick_resume_checkpoint(self, run_dir: Path) -> Path:
        """resume で掴む checkpoint 世代を選ぶ(A6。第118 レーンC)。

        **挙動変更(resume の意味論)**:
          - 従来: `checkpoint.latest()` が返した最新世代を無条件に掴み、pool サイドカーが
            無ければ**黙って素通り**していた(= ドーマント/退場者参照が消えた世界で続行し、
            L1 が straight と食い違っても誰も気づけない)。
          - 以後: pool 有効なら「本体 + サイドカー」が揃った世代だけを完全とみなし、
            欠落/破損の世代は飛ばして**一つ前の完全世代**へ戻る。1 つも無ければ
            明示エラー(黙って素通りしない)。★これは **pool ON の旧 checkpoint**
            (マーカーだけあってサイドカーが無い世代)にも同じ物差しを当てる =
            「pool 有効なら不完全扱い」で一貫させる。
          - pool OFF のランは分岐に入らない(= 従来と 1 バイトも変わらない)。
        """
        gens = checkpoint.complete_generations(run_dir)
        if not gens:
            raise FileNotFoundError(
                f"resume 元に checkpoint が無い: {run_dir / 'checkpoint'}")
        if self._pool is None:
            return gens[-1]
        rejected: list[str] = []
        for path in reversed(gens):
            why = self._pool_sidecar_defect(checkpoint.step_of(path))
            if why is None:
                if rejected:
                    import logging
                    logging.getLogger("society.engine").warning(
                        "resume: 不完全な世代を飛ばした(%s)→ %s から再開",
                        " / ".join(rejected), path.name)
                return path
            rejected.append(f"{path.name}: {why}")
        raise RuntimeError(
            "pool 有効なランで、pool サイドカー(dormant-*.pkl.gz)が揃った checkpoint 世代が"
            " 1 つもありません(= 完全な世代が無い)。黙って素通りすると、ドーマント退避と"
            " 退場者参照を失った世界で続行して L1 が一気通しと食い違います。\n  "
            + "\n  ".join(rejected)
            + f"\n  checkpoint dir: {run_dir / 'checkpoint'}\n"
            "  復旧: (a) 対の dormant-*.pkl.gz を退避先(backup_run)から戻す"
            " (b) 対の無い ckpt-*.pkl.gz を退避してもう一度 --resume する"
            "(それより古い完全世代へ戻る) (c) どの世代も対が無いなら新しい run.name で"
            " 起動し直す。")

    def _assert_no_future_segments(self, ckpt: Path) -> None:
        """掴む世代より**先**の part がディスクに在る = 再実行すると L1 が二重になる(A6)。

        世代 N の checkpoint は `log_seg`(= 保存直後の flush が使う part 番号)を運ぶので、
        「ディスク上に在ってよい part の最大番号」は `log_seg` である。それより先の part が
        在るのは (a) 完成前の世代へ戻った (b) `flush_every_steps` 由来の flush が checkpoint
        より先に走った、のどちらかで、どちらも**確定済みの区間を再実行する**= finalize が
        同じイベントを 2 回結合する。黙って二重化するより落とす方が正しい。
        ★`checkpoint.load` の直後に呼ぶ(本体を 2 度 unpickle しないため)。まだ 1 バイトも
          書いていない時点なので、ここで落ちても run dir は resume 前のまま。
        旧 checkpoint(`log_seg` 無し)では検査しない(従来どおり)。
        """
        seg = getattr(self, "_ckpt_log_seg", None)
        if seg is None or getattr(self, "logger", None) is None:
            return
        on_disk = int(getattr(self.logger, "_seg", 0)) - 1   # 既存 part の最大番号(無ければ -1)
        if on_disk > int(seg):
            raise RuntimeError(
                f"resume 元に、掴む世代({ckpt.name} = part-{int(seg):04d} まで)より先の"
                f" part セグメント(part-{on_disk:04d})が残っています。このまま再開すると"
                " 確定済みの区間を再実行して L1 が二重になります。先の part を退避(改名/移動)"
                " してから再開するか、より新しい完全世代を用意してください。")

    def _restore_pool_resume(self, start: int) -> None:
        """resume 時に pool の状態を復元する(pool ON 時のみ)。sim.agents は checkpoint.load が復元済み。

        ドーマントストア・日境界・退場者参照を復元 → straight run と同じ状態で続行
        (退場者参照が全て解決=reply/造語作者名などの reference が落ちない)。

        ★A6(第118 レーンC)で**挙動が変わった**: 従来は sidecar が無ければ黙って
          素通りし「ラン内一貫性のみ」で続行していたが、pool 有効な世代は sidecar と
          対で 1 つなので、欠落は**不完全世代**として扱う(世代選びの
          `_pick_resume_checkpoint` が既に弾いているので、ここに来る時点で健全。
          直接呼ばれた場合の保険として読み直し、それも駄目なら明示エラー)。"""
        if self._pool is None:
            return
        from collections import OrderedDict
        self.agent_by_id = {a.id: a for a in self.agents}
        self.invalidate_present_index()           # 在場 = checkpoint が復元した present 個体
        prev_min = self.clock.sim_min(max(0, int(start) - 1))
        self._pool_day = prev_min // 1440 if start > 0 else 0
        cached = self.__dict__.pop("_pool_sidecar_blob", None)  # 候補検証で読んだものを使う
        # ★同じ step のものだけを使う(世代選びと復元で step がずれたら読み直す)
        blob = cached[1] if cached and int(cached[0]) == int(start) else None
        if blob is None:
            path = self._pool_sidecar_path(start)
            if not path.exists():
                raise FileNotFoundError(
                    f"pool 有効なランの resume に pool sidecar がありません: {path}"
                    "(checkpoint 本体と対で 1 世代です。黙って素通りすると"
                    " ドーマント退避と退場者参照を失った世界で続行します)")
            blob = self._read_pool_sidecar(start)
        self._dormant = pool_mod.DormantStore(cap=int(blob.get("dormant_cap", 0)))
        self._dormant._d = OrderedDict(blob.get("dormant", {}))
        # ★A2: 恒久台帳(vital)。**旧サイドカー互換** = キーが無ければ rich から作り直す
        #   (rich まで LRU で捨てられていた個体は旧ランでは元から復元不能 = 既知の欠落)。
        self._dormant._vital = dict(blob.get("dormant_vital", {}))
        if "dormant_vital" not in blob:
            self._dormant.rebuild_vital()
        self._pool_day = int(blob.get("pool_day", self._pool_day))
        for aid, agent in blob.get("departed", {}).items():
            # ★A1: **旧サイドカー互換** = departed にフル Agent が入っていても、ここで
            #   軽量参照へ揃える(straight run と同じ軽さ・同じ読み口で続行する)。
            self.agent_by_id.setdefault(aid, _to_ref(agent))
        self._pool_update_budget()

    def run(self, resume_from: Path | str | None = None) -> dict:
        # P0バッチ: elapsed_sec の起点を「run 開始」に打ち直す(構築時間を除く)。
        # resume 時はこのチャンクの経過だけを測る(通算ではない)。
        self._t_start = time.perf_counter()
        # D16: checkpoint_every>0 で該当 step ごとに完全状態を保存 + ログを part 化。
        # 既定 0 = 無効(part を作らず出力・挙動は従来と完全同一)。
        every = int(self.cfg.observer.get("checkpoint_every", 0) or 0)
        # ログの定期 flush を checkpoint から分離(B2)。>0 で N step ごとに flush_segment
        # を呼び L1 バッファの RAM を解放する(大規模ランの OOM 回避)。既定 0 = 無効で
        # 従来と完全同一(part を作らない)。checkpoint と同一 step では二重 flush しない。
        flush_every = int(self.cfg.observer.get("flush_every_steps", 0) or 0)
        start = 0
        if resume_from is not None:
            # A6(第118 レーンC): 「本体 + pool サイドカー + 全 flush」が揃った世代だけを
            # 完全とみなして選ぶ(欠けていれば一つ前へ戻る。1 つも無ければ明示エラー)。
            ckpt = self._pick_resume_checkpoint(Path(resume_from))
            start = checkpoint.load(self, ckpt)
            self._assert_no_future_segments(ckpt)   # load 済み値を読むだけ(再 unpickle しない)
            self.invalidate_present_index()       # 在場索引は load 済みの sim.agents から引き直す
            self._restore_pool_resume(start)      # pool ON 時のみ: ドーマント/日境界の復元
            # 第57バッチ タスクC: 分割実行で clean finalize しても前チャンクの canonical を失わない
            # (logger._finalize_stream が resume 時のみ既存 canonical を先頭に結合する)。fresh ラン不変。
            self.logger._resumed = True
            if self.indoor_tracks is not None:    # B3: tracks サイドカーも分割実行の canonical を保つ
                self.indoor_tracks._resumed = True
            if self.org_ledger_sc is not None:    # B4: org_ledger サイドカーも分割実行の canonical を保つ
                self.org_ledger_sc._resumed = True
            if self.finance_sc is not None:       # IF-E2: finance サイドカーも同様
                self.finance_sc._resumed = True
            if self.channels_sc is not None:      # 第80: 観測チャンネルも二重記録しない(canonical 先頭結合)
                self.channels_sc._resumed = True
            if self.cognition_g_sc is not None:   # 第82: g/θ 軌跡サイドカーも同様
                self.cognition_g_sc._resumed = True
            if self.roster_sc is not None:        # A13: 日次入場者名簿も canonical を先頭結合
                self.roster_sc._resumed = True
            # ---- G4/G5: 日次サイドカーの「最後に撮った日」を据える(resume==straight の要)----
            # 前チャンクは step `start-1` まで実行済みなので、その step の日までは撮り終えて
            # いる。ここを据えないと再開直後の step で同じ日をもう一度撮り、一気通しのランと
            # 行集合が食い違う(名簿 A13 は `_seen` 集合で冪等なのでこの手当てが要らなかった)。
            # ★「既存 part の最大 day から復元する」方式は採らない: **行が 1 つも無い日**
            #   (day0 の朝はまだ誰も何も覚えていない)を撮り直してしまうため。
            # ★変数名に `sc` を含めない(tests/test_indoor_invariance.py の静的検査が
            #   局所変数名の部分一致でサイドカー参照を拾うため)。
            _prev_min = int(self.clock.sim_min(max(0, start - 1)))
            _prev_day = _prev_min // 1440
            for _daily_log in (self.memory_sc, self.relations_sc):
                if _daily_log is not None:
                    _daily_log._resumed = True
                    _daily_log._day = _prev_day
            # ---- レーンG 案B: 意図収束サイドカーは**日境界ではなく capture_min** で
            #      撮るので、「前チャンクがその日ぶんを撮り終えていたか」は時刻で決まる。
            #      判定は本体側(resume_at)に置く = 据え方の規則を 1 箇所に閉じる。
            if self.gathering_sc is not None:
                self.gathering_sc.resume_at(_prev_min)
        if every > 0:
            save_config(self.cfg, self.out_dir)   # 途中再開に備え config を先出しする
        for step in range(start, int(self.cfg.run.n_steps)):
            scheduler.run_step(self, step)
            # 第78バッチ: 状態ハッシュチェーン(既定 OFF=self.state_hash is None=分岐 1 回だけ)。
            # step 完了の直後・checkpoint/flush の**前**に採る(= 採る時点の世界が
            # 「step 終了時の世界」で一意に定まる)。書くだけでシムは読まない(R1)。
            if getattr(self, "state_hash", None) is not None:
                self.state_hash.update(self, step)
            did_flush = False
            if every > 0 and (step + 1) % every == 0:
                # ★A6(第118 レーンC): COMPLETE マーカーは**この世代の書き物が全部
                #   終わった後**に置く(下の write_complete_marker まで下ろした)。
                #   本体 rename の直後に置いていた旧順序では、「マーカーは在るが
                #   pool サイドカーも part も無い」世代が生まれ、latest() がそれを
                #   完成扱いで掴んで L1 が最大 checkpoint 間隔ぶん欠落していた。
                _ckpt_path = (self.out_dir / "checkpoint"
                              / f"ckpt-{step + 1:06d}.pkl.gz")
                checkpoint.save(self, step + 1, _ckpt_path, complete=False)
                self._save_pool_sidecar(step + 1)  # pool ON 時のみ: ドーマント退避の対保存
                _wv_mod.absorb_before_flush(self)  # 第98 W4-A: worldview 未走査区間の吸い上げ
                self.logger.flush_segment()
                if self.indoor_tracks is not None:  # B3: tracks サイドカーも対でセグメント化
                    self.indoor_tracks.flush_segment()
                if self.org_ledger_sc is not None:  # B4: org_ledger サイドカーも対でセグメント化
                    self.org_ledger_sc.flush_segment()
                if self.finance_sc is not None:     # IF-E2: 部門別残高サイドカーも対でセグメント化
                    self.finance_sc.flush_segment()
                if self.channels_sc is not None:    # 第80: 観測チャンネルも対でセグメント化
                    self.channels_sc.flush_segment()
                if self.cognition_g_sc is not None:  # 第82: g/θ 軌跡も対でセグメント化
                    self.cognition_g_sc.flush_segment()
                if self.roster_sc is not None:      # A13: 日次入場者名簿も対でセグメント化
                    self.roster_sc.flush_segment()
                if self.memory_sc is not None:      # G4: 記憶ストリームも対でセグメント化
                    self.memory_sc.flush_segment()
                if self.relations_sc is not None:   # G5: 関係台帳の日次差分も同様
                    self.relations_sc.flush_segment()
                if self.gathering_sc is not None:   # レーンG: 意図収束も対でセグメント化
                    self.gathering_sc.flush_segment()
                # ★世代トランザクションのコミット点(A6)。ここまで来て初めて
                #   「step {step+1} まで完全に書けた」= latest() の候補になる。
                checkpoint.write_complete_marker(_ckpt_path)
                did_flush = True
            if flush_every > 0 and not did_flush \
                    and (step + 1) % flush_every == 0:
                # 第98 W4-A: 日境界にしか走らない worldview の未走査区間を、L1 バッファが
                # 空になる前に carry へ吸い上げる(既定 flush_every_steps=0 では未到達)。
                _wv_mod.absorb_before_flush(self)
                self.logger.flush_segment()        # checkpoint と独立の定期 flush
                for _j in self._journals:          # 第71: LLM ジャーナルも同時に確定させる
                    _j.flush()
                if self.indoor_tracks is not None:
                    self.indoor_tracks.flush_segment()
                if self.org_ledger_sc is not None:
                    self.org_ledger_sc.flush_segment()
                if self.finance_sc is not None:
                    self.finance_sc.flush_segment()
                if self.channels_sc is not None:
                    self.channels_sc.flush_segment()
                if self.cognition_g_sc is not None:
                    self.cognition_g_sc.flush_segment()
                if self.roster_sc is not None:
                    self.roster_sc.flush_segment()
                if self.memory_sc is not None:
                    self.memory_sc.flush_segment()
                if self.relations_sc is not None:
                    self.relations_sc.flush_segment()
                if self.gathering_sc is not None:
                    self.gathering_sc.flush_segment()
        return self.finalize()

    def _agents_json_records(self) -> list:
        """ビューア用名簿(agents.json)のレコード列。org_id/org_role は配属時のみ(非配属ランはバイト同一)。

        __init__ の初期出力と finalize の再出力で共有(重複回避)。org 配属は run 中の遅延初期化
        (_ensure_orgs)で付くため、__init__ 時点では org_id が未付与=キー不在。会社観測データ層 B4
        (indoor_fields/ledger)ON のランのみ finalize が本レコードで agents.json を再出力し org_id/
        org_role を載せる(B7 会社 UI の org 相関材料)。OFF は再出力しない=既存 agents.json とバイト一致。

        W4-F: `work_node` / `part_time_node` は **day_plan 実効ランだけ**に足す 2 欄。
        これは `day_plan.resolve_place("work")` が引く `_work_node(agent)`(本業 → バイト先)
        の入力そのもので、事後解析(scripts/analyze_plan_execution.py)が職場ブロックの
        到達判定を「建物内の全ノード」という近似ではなく実ノードで行えるようにする。
        **無条件では足さない**: agents.json は day_plan OFF ランでも出る成果物であり、
        既定ランのバイト不変が本バッチの絶対条件だから。IF-E2 の org_id/org_role が
        「配属時のみキーを生やす」で既定バイトを守ったのと同じ流儀
        (mind_model / ontology_group / input_res も同型の条件付きキー)。
        ★スナップショットである点は org_id と同じ: 転職(organizations)で work_node は
          run 中に変わりうるが、本レコードは書き出した時点の値しか持たない。"""
        # day_plan が実効か(planning.enabled が前提)。cfg_of は遅延キャッシュを 1 つ
        # 生やすだけで L1/L2/乱数のいずれにも出ない(day_plan.cfg_of の明文どおり)。
        from ..cognition import day_plan as _day_plan
        dp_on = _day_plan.enabled(self)
        return [{"id": a.id, "name": a.name, "age": a.age, "gender": a.gender,
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
                    if getattr(a, "input_res", None) else {}),
                 **({"ontology_group": a.ontology_group}
                    if getattr(a, "ontology_group", None) else {}),
                 **({"ontology_axes": a.ontology_axes}
                    if getattr(a, "ontology_axes", None) else {}),
                 # 第88バッチ: agent_id → model_id の対応(原文書 §5「モデルと人格の交絡が
                 # 生じるため必ずログに残す」)。mind OFF ではキー自体が生えない=バイト一致。
                 **({"mind_model": a.mind["model"], "mind_tier": a.mind["tier"]}
                    if getattr(a, "mind", None) else {}),
                 # 第50バッチ T6: 信用内訳レンズの org 相関用(org 配属時のみ=非配属ランはバイト同一)
                 **({"org_id": a.org_id, "org_role": getattr(a, "org_role", "")}
                    if getattr(a, "org_id", None) else {}),
                 # W4-F: 計画遵守観測の職場ノード突合(day_plan 実効ランのみ=OFF はバイト同一)。
                 # day_plan._work_node と同じ 2 つの入力を、同じ優先順(本業 → バイト先)で残す。
                 **({"work_node": a.work_node,
                     "part_time_node": (a.part_time["node"] if a.part_time
                                        else None)} if dp_on else {})}
                for a in self.agents]

    def finalize(self) -> dict:
        scheduler.finalize_org_day(self)          # B4: 最終日の org_output(by_org)/ledger 行を締める
        from .. import economy_sfc as _sfc_fin     # IF-E2: 最終日の域外収支を締める(既定 OFF=no-op)
        _sfc_fin.finalize(self)
        self._journal_close()                     # 第71: LLM ジャーナルの残バッファを確定させる
        paths = self.logger.flush()               # ↑ logger.flush の前=最終日 org_output も L1 に載る
        if self.indoor_tracks is not None:        # B3: 屋内軌跡サイドカーを結合(part→canonical)
            self.indoor_tracks.finalize()
        if self.org_ledger_sc is not None:        # B4: 組織日次系列サイドカーを結合(part→canonical)
            self.org_ledger_sc.finalize()
        if self.finance_sc is not None:           # IF-E2: 部門別残高サイドカーを結合
            self.finance_sc.finalize()
        if self.channels_sc is not None:          # 第80: 観測チャンネルを結合(part→canonical)
            self.channels_sc.finalize()
        if self.cognition_g_sc is not None:       # 第82: g/θ 軌跡を結合(part→canonical)
            self.cognition_g_sc.finalize()
        if self.roster_sc is not None:            # A13: 日次入場者名簿を結合(part→canonical)
            self.roster_sc.finalize()
        if self.memory_sc is not None:            # G4: 記憶ストリームを結合(part→canonical)
            self.memory_sc.finalize()
        if self.relations_sc is not None:         # G5: 関係台帳の日次差分を結合
            self.relations_sc.finalize()
        if self.gathering_sc is not None:         # レーンG: 意図収束セルを結合
            self.gathering_sc.finalize()
        # B4 item#4: 会社観測(indoor_fields/ledger)ON かつ org 配属があるランは agents.json を再出力し
        # org_id/org_role を載せる(org 配属は run 中の遅延初期化=__init__ 時点では未付与のため)。
        # OFF は再出力しない=既存 agents.json とバイト一致(ゴールデン非該当)。
        wc = self.workcfg
        if (wc["enabled"] and (wc["indoor_fields"] or wc["ledger"]["enabled"])
                and any(getattr(a, "org_id", None) for a in self.agents)):
            (self.out_dir / "agents.json").write_text(
                json.dumps(self._agents_json_records(), ensure_ascii=False),
                encoding="utf-8")
        save_config(self.cfg, self.out_dir)
        # 第73バッチ Part B: 真偽台帳をサイドカー(beliefs_ledger.json)へ書き出す。
        # **L1 には真値を出さない**(L1 の消費経路へ紛れ込む余地を構造的に断つ)。
        # beliefs OFF のランでは 1 ファイルも作らない=既存ランの出力一式とバイト一致。
        from .. import truth_ledger as _truth_ledger
        _truth_ledger.dump(self)
        # 所有権レイヤー O1: 登記簿のスナップショットをサイドカー(assets_ledger.json)へ。
        # scripts/analyze_assets.py が Gini・保有期間・所有ネットワークを計算する材料で、
        # **シム本体に走査を足さない**ための唯一の出口(観測がシムに触らない = 研究文書 §6-3)。
        # world.assets.enabled=false のランでは 1 ファイルも作らない=既存ランの出力一式と同形。
        from .. import assets as _assets_mod
        _assets_mod.dump(self)
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
        # ---- P0バッチ 2026-07-29: 性能実測の追加キー(既存キーは一切不変)----
        # elapsed_sec: run() 開始から finalize 直前までの wall time [s]。run() を経ずに
        #   finalize() を直接呼ぶテスト経路では __init__ からの経過になる(その旨を承知で使う)。
        #   resume したランでは「そのチャンクの経過」であって通算ではない。
        # peak_rss_mb: プロセスのピーク常駐メモリ [MB]。取得できない環境ではキー自体を出さない
        #   (欠測を 0 と偽らない)。tracemalloc の Python ヒープ峰値とは別物(C++ バッファ込み)。
        summary["elapsed_sec"] = round(time.perf_counter() - self._t_start, 3)
        # ---- M-1(第86バッチ保守): LLM 1 呼の絶対時限に掛かった件数 ----
        # **0 件のときはキー自体を出さない**(mock/既定ランの summary.json は従来とバイト一致。
        # peak_rss_mb と同じ「欠測を 0 と偽らない/無風なら足さない」流儀)。
        # 源は society/llm/deadline.py のプロセス内カウンタ(全 HTTP backend が通る唯一の関門)。
        from ..llm import deadline as _llm_deadline
        _dl_exceeded = _llm_deadline.exceeded_count()
        if _dl_exceeded:
            summary["llm_deadline_exceeded"] = int(_dl_exceeded)
        peak_rss = _peak_rss_mb()
        if peak_rss is not None:
            summary["peak_rss_mb"] = peak_rss
        # ---- deliberate batch 第130: 一括発行の実績(batch OFF=属性が無い=キー自体を出さない)----
        # deferred_fallback は「応答適用を遅延した個体にパース不成立の後退経路が現れた回数」。
        # **0 のあいだ batch 経路は逐次経路と機械的に同値**(scheduler の設計注記⑤)なので、
        # ラン後の厳密一致判定はこの整数 1 個を見る。他キーは batch 率の観測用。
        _bd = getattr(self, "_batch_decide_stats", None)
        if _bd is not None:
            summary["batch_decide"] = {k: int(v) for k, v in _bd.items()}
        # ---- A1 第67バッチ 2026-07-29: 環境条件の追加キー(既定 OFF ではキー自体を出さない)----
        # world_mod: 適用したプロファイル名と適用実績(予約=未消費フィールドも正直に残す)。
        # building_heights: 実高さの付与実測(plateau 実測 / levels 推定 / 不明 の内訳)。
        if getattr(self, "worldmod", None) is not None:
            summary["world_mod"] = self.worldmod.summary()
        if getattr(self, "heights_stat", None) is not None:
            summary["building_heights"] = dict(self.heights_stat)
        # ---- 天候の来歴 第80バッチ W2(weather.mode=generated / table のときだけ)----
        # 入力データのハッシュがラン成果物に残らない、という指摘への最初の実例:
        # weather_params_sha256(較正パラメータ)+ 実測データの sha256 + フォールバック件数。
        # 既定 synthetic では None を返すのでキー自体を出さない(既存ランと同形)。
        from .. import weather as _weather_prov
        _wprov = _weather_prov.provenance(self)
        if _wprov is not None:
            summary["weather"] = _wprov
        # ---- day_plan v1 第86バッチ(planning.day_plan.enabled=false = 既定 OFF はキーなし)----
        # 原文書 §7 が「修復発生回数のモデル別集計はテストバッテリーの追加指標になる」と明記して
        # いるので、修復・フォールバックの件数と率をモデル別に summary へ残す。
        from ..cognition import day_plan as _day_plan_prov
        _dpprov = _day_plan_prov.provenance(self)
        if _dpprov is not None:
            summary["day_plan"] = _dpprov
        # ---- DPH-O 観測 4 点(observer.starvation.enabled=false = 既定 OFF はキーなし)----
        # 「削ったなら記録する」の受け皿。L2 へ列を足すのは observer/aggregate.py が
        # 凍結対象(metrics_spec.SPEC_FILES)なので不可 → summary の provenance に出す。
        from ..observer import starvation as _starv_prov
        _stprov = _starv_prov.provenance(self)
        if _stprov is not None:
            summary["starvation"] = _stprov
        # ---- V3 決定モード(observer.decision_mode.enabled=false = 既定 OFF はキーなし)----
        # 「その行動を決めたのは LLM / ルール / 再利用のどれか」の日別内訳。朝の計画
        # (plan_created.src)と夜の内省(l1b purpose=reflect + reflect_dropped)は既に
        # L1 だけで可視なので、ここが埋めるのは**日中熟慮レーンの分母**だけである。
        from ..observer import decision_mode as _decmode_prov
        _dmprov = _decmode_prov.provenance(self)
        if _dmprov is not None:
            summary["decision_mode"] = _dmprov
        # ---- engaged モード 第87バッチ(cognition.engaged.enabled=false = 既定 OFF はキーなし)----
        # 原文書 §8 補助規則 3 が「全エピソードのログ(トリガー種別・滞在時間・ターン数・
        # 脱出理由・モデルID)を記録し思考量の個体差の観測量とする」と明記しているので、
        # エピソード単位の内訳(種別/トリガ/脱出理由/滞在ヒストグラム)を summary へ残す。
        from ..cognition import engaged as _engaged_prov
        _egprov = _engaged_prov.provenance(self)
        if _egprov is not None:
            summary["engaged"] = _egprov
        # ---- 拒否通知 第94バッチ IF-B(cognition.rejection_notify="silent" = 既定はキーなし)----
        # 監査 §2-C の無音拒否に通知を入れた層。何がどれだけ拒否され、そのうち何件が
        # 「今すぐ考える」へ前倒しされ、何件が engaged OFF で memory へ縮退したかを残す
        # (アブレーション軸として事前登録するときの分母になる観測量)。
        from .. import reject as _reject_prov
        _rjprov = _reject_prov.provenance(self)
        if _rjprov is not None:
            summary["rejection_notify"] = _rjprov
        # ---- 噂(情報オブジェクト)第95バッチ IF-C(information.rumors.enabled=false = 既定はキーなし)----
        # if-lane-research §2 が要求した「本シムの拡散曲線が Daley-Kendall / Maki-Thompson の
        # 理論値(最終 ignorant≈20%・spreader ピーク≈0.307)とどれだけ違うか」の分母になる観測量。
        # 何件生まれ・何件伝わり・何人が黙り・1 噂あたり何人に届いたか(reach 分布)を残す。
        # 理論値との**判定**は解析側 scripts/analyze_rumors.py に置く(判定式を 1 箇所に保つ)。
        from .. import rumors as _rumors_prov
        _rmprov = _rumors_prov.provenance(self)
        if _rmprov is not None:
            summary["rumors"] = _rmprov
        # ---- 痕跡(場所のイベント履歴)第96バッチ IF-D(world.traces.enabled=false = 既定はキーなし)----
        # if-lane-research §1 が「減衰は欠陥ではなく機能」「単一 TTL を全種に使うのは文献に反する」と
        # した設計の観測側。何件が場所に刻まれ(marks)・何件が薄れて消え(evaporated)・
        # 何行が後から来た者のプロンプトへ入ったか(lines_injected)を種別内訳つきで残す。
        # ★閲覧は L1 に出さない(受動観測で毎 step 大量発生する)ので、注入総数はここが唯一の観測点。
        from .. import traces as _traces_prov
        _trprov = _traces_prov.provenance(self)
        if _trprov is not None:
            summary["traces"] = _trprov
        # ---- 対人事件の収束化 H4(incidents_interpersonal.enabled=false = 既定はキーなし)----
        # 計画書 §7「分母の罠」の観測側: 犯罪は人口比ではなく**共在機会比**で較正するので、
        # 件数だけでなく candidates / pairs_seen / weight_sum / draws を必ず並べて出す
        # (来街ピークで事件が過剰になっていないかを事後に判定できるようにする)。
        # パターン検収量(場所集中 node_hhi / 反復被害 victim_repeat_rate)と、
        # 通報層の非緊急率(実測 110 番の 16% と突き合わせる)も同じキーに残す。
        from .. import incidents_interpersonal as _inc_prov
        _icprov = _inc_prov.provenance(self)
        if _icprov is not None:
            summary["incidents_interpersonal"] = _icprov
        # ---- 遺失物ループ H3(lost_property.enabled=false = 既定はキーなし)----
        # 計画書 §3 が検証ターゲットに指名した **品目別返還率**(傘 ≈1% / 財布 60-80% /
        # 携帯 87% = 東京都遺失物センター実測)の分子・分母をそのまま残す。分母は実測の定義に
        # 合わせて「届けられた件数」(turnins)。落下・気づき・拾得・素通り・着服・時効・失効の
        # 件数を品目別に並べるので、較正のどの段が外れたのかが 1 枚で判る。
        # ★貨幣保存の第 3 項(cash_held = 世界に在るが誰の残高でもない現金)も一級市民として出す
        #   = 「拾得金は必ず誰かの drop から」(§4)を事後に機械検証するための観測量。
        from .. import lost_property as _lost_prov
        _lpprov = _lost_prov.provenance(self)
        if _lpprov is not None:
            summary["lost_property"] = _lpprov
        # ---- 所有権レイヤー O1+O3(world.assets.enabled=false = 既定はキーなし)----
        # 計画書 §1-2 の検証ターゲット = **資産保存則**(Σ 所有者別保有数 + K5 − RoW 生成 =
        # 初期ストック)の残差と、**未分類の移転種**(IF-E の監視装置の双対)。
        # 初期配賦の内訳(持ち家 / 域内不動産 org / 域外 RoW)と域内 org の特定方法も残すので、
        # 較正のどこが実測でどこが仮定なのかが 1 枚で判る(ユーザー決定 §5-1 の検収材料)。
        # 相続(O3)は deaths / inherited_* / escheat_*(相続人不存在 = 国庫)を分けて出す
        # = 第107 の観測盲点「死者の財布に凍結」が解消したことを事後に機械検証できる。
        from .. import assets as _assets_prov
        _asprov = _assets_prov.provenance(self)
        if _asprov is not None:
            summary["assets"] = _asprov
        # ---- 医療の受け皿 H2(medical.enabled=false = 既定はキーなし)----
        # 計画書 §2 の検証ターゲット: **搬送先が地図に在るか**(n_hospitals。subcat を持たない
        # 地図では 0 = 搬送が起きないことが数字で判る)・確定重症度別の入院件数(東京実測の
        # 程度別構成と突き合わせる)・**金の三本足の合計**(公費 / 自己負担 / 保険給付)。
        # ★受け手 org を特定できなかった保険給付(insurance_to_row)も一級市民として出す
        #   = 「街の残高が 1 円も動いていない」ことを事後に機械検証するための観測量。
        from .. import medical as _med_prov
        _mdcprov = _med_prov.provenance(self)
        if _mdcprov is not None:
            summary["medical"] = _mdcprov
        # ---- org の会計主体化 + rest-of-world IF-E2 案B(economy.org_accounting=false = 既定はキーなし)----
        # 案B 固有の新しい研究量 =「この街の経済が域外にどれだけ依存しているか」。RoW のチャネル別
        # 累積・org 預金の分布・当座借越の件数・受け手解決の段別件数・**接続できていない金の経路**
        # (ゼロと偽らない)を残す。閉じた不変量 Σ(全主体残高)+RoW 累積 の現在値も併記する。
        from .. import economy_sfc as _sfc_prov
        _sfprov = _sfc_prov.provenance(self)
        if _sfprov is not None:
            summary["org_accounting"] = _sfprov
        # ---- 心モデル固定+三層知能 第88バッチ(model.mind.enabled=false = 既定 OFF はキーなし)----
        # 原文書 §5「モデルと人格の交絡が生じるため必ずログに残す」の集計側。
        # モデル別の人数・呼数・キャッシュ命中に、第87 エピソード数と第86 修復率を統合する。
        from .. import mind as _mind_prov
        _mdprov = _mind_prov.provenance(self)
        if _mdprov is not None:
            summary["mind"] = _mdprov
        # ---- プラセボ L1 第89バッチ(ablate.* の 3 種。既定 OFF はキーなし)----
        # 「壊した量」(書き換えた節の件数・ペルソナ交換数・対合が成立したか)を必ず残す。
        # 件数 0 のランはプラセボとして無効=単調性の主張に使ってはいけない、が事後に判る。
        from .. import ablate as _placebo_prov
        _plprov = _placebo_prov.provenance(self)
        if _plprov is not None:
            summary["placebo"] = _plprov
        # ---- プロンプト言い換え 第92バッチ(ablate.prompt_paraphrase="" = 既定 OFF はキーなし)----
        # manifest はラン開始時に書かれるので mode と凍結表 sha256 しか載らない。「実際に何行を
        # 何回言い換えたか」(= 検査として有効だったか)は走り切ってからでないと判らないため、
        # プラセボの provenance と同じ流儀でここに残す。0 件のランは検査として無効と事後に判る。
        if _placebo_prov.prompt_paraphrase(self):
            summary["prompt_paraphrase"] = _placebo_prov.paraphrase_provenance(self)
        # ---- 退行シグナル監視 第91バッチ(observer.regression.enabled=false = 既定 OFF はキーなし)----
        # 設計 §3 の 4 群を L2 列として出した仕様と、「語彙系から外した発話の総数」を残す。
        # 除外総数を残すのは第87 の申し送り(定型応答は機構由来の定数なので語彙指標から
        # 外すが、**黙って落とさない**)への対応。
        from ..observer import regression as _reg_prov
        _rgprov = _reg_prov.provenance(self)
        if _rgprov is not None:
            summary["regression"] = _rgprov
        # ---- 初期フレーム共変量 第74バッチ IDEA④(observer.initial_frame.days: 0 = 既定 OFF)----
        # 確定済みの l1_events.parquet(直前の logger.flush)を読み直す **完全な事後処理**。
        # OFF ではこのブロックが即 None を返し summary にキーを足さない=既存ランと同形。
        from ..observer import initial_frame as _iframe
        from ..observer import norms as _norms_mod
        _ifcfg = _iframe.cfg_of_config(self.cfg)
        if _ifcfg["days"] > 0:
            _if = _iframe.summarize(self.out_dir, _ifcfg["days"],
                                    _ifcfg["top_kinds"],
                                    _norms_mod.cfg_of(self)["markers"],
                                    self.clock.step_minutes)
            if _if is not None:
                summary["initial_frame"] = _if
        # ---- Wave 4 現実被覆レーンの観測タリー(第109バッチ D2)------------------------
        # 第105〜108 で入った層は ``provenance(sim)`` を持ちながら **summary.json へ配線
        # されていなかった**(各 module が「書き出しは別レーンの所有」と正直に書いていた
        # 状態)。そのため縦煙では「機能が動いたか」を L1 の kind 経由でしか判定できず、
        # **0 件のときに『OFF だった』のか『ON だが起きなかった』のかが区別できない**。
        # ここで一括配線して、上の 15 ブロックと**同じ規約**に揃える:
        #   ・OFF(= provenance が None)では **キー自体を出さない**(既存ラン・golden 無風)
        #   ・provenance を持たない module には作らない(配線だけ・観測量を発明しない)
        # ★末尾にまとめて足すのは**キー順の保存**のため: 途中へ挿すと既に ON で回している
        #   ラン(medical / lost_property など)の summary.json のキー順が動く。
        # ★import は各ブロックと同じ関数内 import(循環 import の回避 = 既存の流儀)。
        # ★ループで畳まず 1 層 1 ブロックで書くのは上の 15 ブロックと同じ形にするため
        #   (``<module>.provenance(self)`` という呼び出しの形が、配線されているかを
        #   静的に走査する tests/test_summary_provenance.py の唯一の手がかりになる)。
        from .. import health as _health_prov          # H1 重症度 S0-S4 と死
        _hlprov = _health_prov.provenance(self)
        if _hlprov is not None:
            summary["health"] = _hlprov
        from .. import city_ops as _city_ops_prov      # 交番/ごみ/納品/夜間清掃/救急
        _coprov = _city_ops_prov.provenance(self)
        if _coprov is not None:
            summary["city_ops"] = _coprov
        from .. import incidents_env as _incenv_prov   # H5 火災 / 交通 / 群集
        _ieprov = _incenv_prov.provenance(self)
        if _ieprov is not None:
            summary["incidents_env"] = _ieprov
        # conf のキーが world.facilities なので summary のキー名もそちらへ揃える。
        from .. import facility_devices as _facil_prov  # 昇降設備の DEVS 摩耗
        _fdprov = _facil_prov.provenance(self)
        if _fdprov is not None:
            summary["facilities"] = _fdprov
        from .. import street_life as _street_prov     # 街路の生業
        _slprov = _street_prov.provenance(self)
        if _slprov is not None:
            summary["street_life"] = _slprov
        from .. import transit_interior as _train_prov  # 車内空間
        _tiprov = _train_prov.provenance(self)
        if _tiprov is not None:
            summary["transit_interior"] = _tiprov
        # ---- E5 職場束ね coverage(レーン丙 3。pool ランで org/束ねが ON のときだけ)------
        # 既存の検収量(_pool_org_stat / _workbind_agg)は**累計**で、回転のたびに +1 される
        # ため「担い手が痩せていても増えて見える」。ここでは**最終日の在場者を分母にした
        # 保持率**を同じキーに並べて置き、累計との乖離をそのまま観測量にする。
        # 該当しないラン(pool OFF / org も束ねも OFF)は None = キー自体を出さない。
        from .. import work as _work_prov
        _wkprov = _work_prov.provenance(self)
        if _wkprov is not None:
            summary["workplace_coverage"] = _wkprov
        # ---- 存在の内生化 POP(pop.* が全て OFF のランはキー自体を出さない)--------------
        # 人口会計: 転出 / 定着(転入)/ 出生の件数と**台帳の行数が一致すること**、および
        # 1 日あたりの発生数が実数アンカー(転入 8.4 / 転出 7.8 / 出生 0.59 人日)の帯に
        # 入るか。レートは**駆動ではなく検証目標**なので、両方を並べて出す。
        from .. import population as _pop_prov
        _ppprov = _pop_prov.provenance(self)
        if _ppprov is not None:
            summary["population"] = _ppprov
        (self.out_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary
