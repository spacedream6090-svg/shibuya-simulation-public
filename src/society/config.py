"""設定ローダ(D11)。conf/config.yaml + プロファイル差分 + dotlist 上書き。実験条件はすべてここを通る。"""
from __future__ import annotations

import datetime
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "conf" / "config.yaml"

_INT_KEYS = [
    "run.seed", "run.n_agents", "run.n_steps",
    "labeling.adopt_threshold", "lod.max_llm_per_step",
    "observer.snapshot_every", "world.edge_capacity",
    "k.reflect_period_days", "model.max_tokens", "model.reflect_max_tokens",
    "model.plan_max_tokens",                # 朝の計画のみ上限を分ける seam(0=max_tokens)

    "controls.null_calls",                 # D7 null 系列: 発火あたりのダミー呼び出し本数
    # 交通 od モードの数値パラメータ(dotlist 上書きを整数化。ambient では未使用)
    "world.traffic.od_cars_per_day", "world.traffic.capacity_per_lane",
    "world.traffic.default_lanes", "world.traffic.max_cars_log",
    "world.traffic.free_speed_min", "world.traffic.free_speed_max",
    # 朝の一日計画 / 交通機関(ユーザー要望 2026-07-06)の整数パラメータ
    "planning.max_items", "transit_ride.bus.headway_steps",
    # 制度DSL(ユーザー構想 2026-07-06)の整数パラメータ
    "rules.max_active", "rules.duration_days",
]
_FLOAT_KEYS = [
    "world.walk_speed_m_per_step", "world.perception_radius_m",
    "lod.congestion_surprise", "k.degraded_alpha", "model.temperature",
    "rewards.amount_per_adoption",          # D9: 採用1件あたりの報酬額
    "drive.slope",                          # logistic 発火のシャープさ(B段 seam)
    "opinion.w_face", "opinion.w_dm", "opinion.w_sns",  # FJ 意見更新の説得重み(#16)
    "net.like_prob", "net.reshare_prob",    # SNS 反応の確率(#14)
    "model.timeout_s",                      # 本選 vllm: 1 リクエストのタイムアウト秒
    "world.traffic.through_ratio",          # 交通 od: 通過交通の比率(ambient では未使用)
    "world.traffic.signal_cycle_s",         # 交通 od: 信号周期秒
    # 交通機関(ユーザー要望 2026-07-06)の小数パラメータ
    "transit_ride.taxi.prob", "transit_ride.taxi.min_dist_m",
    "transit_ride.taxi.base_fare", "transit_ride.taxi.per_km",
    "transit_ride.bus.fare", "transit_ride.bus.stop_radius_m",
    # 制度DSL(ユーザー構想 2026-07-06)の小数パラメータ
    "rules.fee_max_ratio", "rules.bonus_max", "rules.boost_prob",
]


def load_config(overrides: list[str] | None = None,
                path: str | Path | None = None,
                profile: str | Path | None = None) -> DictConfig:
    """基底 config.yaml(または path)に、profile 差分 YAML → dotlist の順で重ねる。

    profile: 本番用など「基底との差分だけ」を書いた YAML(conf/production.yaml 等)。
             基底の上に OmegaConf.merge で重ねる(未指定キーは基底のまま)。
    """
    cfg = OmegaConf.load(Path(path) if path else DEFAULT_CONFIG)
    if profile is not None:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(Path(profile)))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    # world.calendar.start_date="auto" → 実行開始日(現実の当日)へ解決し、以後は具体日付として
    # 凍結(config スナップショットに具体日付が残る=resume/再現が安定。tests は "auto" を使わない)。
    if OmegaConf.select(cfg, "world.calendar.start_date") == "auto":
        OmegaConf.update(cfg, "world.calendar.start_date",
                         datetime.date.today().isoformat())
    # dotlist は文字列で入るため既知キーを型強制(再現性のため曖昧さを残さない)
    for key in _INT_KEYS:
        if OmegaConf.select(cfg, key) is not None:
            OmegaConf.update(cfg, key, int(OmegaConf.select(cfg, key)))
    for key in _FLOAT_KEYS:
        if OmegaConf.select(cfg, key) is not None:
            OmegaConf.update(cfg, key, float(OmegaConf.select(cfg, key)))
    # YAML は 'off' を False と解釈するため文字列へ正規化(k.writeback=off 対策)
    wb = OmegaConf.select(cfg, "k.writeback")
    if isinstance(wb, bool):
        OmegaConf.update(cfg, "k.writeback", "off" if not wb else "free")
    return cfg


def save_config(cfg: DictConfig, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml")
