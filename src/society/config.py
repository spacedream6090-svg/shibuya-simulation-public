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
    "k.degraded_alpha", "model.temperature",
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
    # 建物の実高さ(A1 第67バッチ)。未照合建物の推定高さ = levels x この値
    "world.heights.fallback_m_per_level",
]


def _resolve_data(path_str: str, label: str) -> str:
    """EnvPack のデータ参照(リポジトリルート相対 or 絶対)を実在チェックしてそのまま返す。

    欠けていたら明確なエラー(どのキーのどのファイルが無いか)を出す。パスは config へは
    元の相対文字列のまま入れる(既存 config と同じ「リポジトリルート相対」規約=Simulation 側で解決)。
    """
    p = Path(path_str)
    full = p if p.is_absolute() else REPO_ROOT / p
    if not full.exists():
        raise FileNotFoundError(
            f"EnvPack のデータファイルが実在しない({label}): {path_str}")
    return str(path_str)


def build_env_overlay(env: str | Path) -> dict:
    """EnvPack manifest(env/<place>/env.yaml または env.yaml パス)を既存 config キー群へ写像する。

    3層設計(docs/plans/env-classification.md):
      ③-A データ実体 → world.map / transit.file / agents.personas_file / organizations.* /
                        world.traffic.features_file / ads.*
      ③-B 語彙・行事・気候・番組名・原点・駅名 → envpack.*
      ② 制度: institutions.pref を ref/institutions_jp.yaml から解決 → institutions.*(税・按分・
         労働日)+ economy.min_wage_hourly(都道府県スコープ)。既定=現行コード値=ref なので L1 不変。
      ③ 地点固有の制度: council → institution_routes.assembly/vote / rent_income_ratio → economy.accounts

    返す dict は profile と同じ重ね書き機構(OmegaConf.merge)で基底へ載せる。必須キー(env.name /
    data.map)の欠落・データファイルの不在は明確なエラーにする。transit 無し env は縮退(徒歩の街)。
    """
    p = Path(env)
    manifest = (p / "env.yaml") if p.is_dir() else p
    if not manifest.exists():
        raise FileNotFoundError(f"EnvPack manifest が見つからない: {manifest}")
    doc = OmegaConf.to_container(OmegaConf.load(manifest), resolve=True) or {}
    base_dir = manifest.parent

    # ---- 必須キー検証 ----
    if not (doc.get("env") or {}).get("name"):
        raise ValueError(f"EnvPack manifest に必須キー env.name が無い: {manifest}")
    data = doc.get("data") or {}
    if not data.get("map"):
        raise ValueError(f"EnvPack manifest に必須キー data.map が無い(地図は必須): {manifest}")

    overlay: dict = {}

    # ---- ③-A データ実体 → 既存 config キー ----
    world: dict = {"map": _resolve_data(data["map"], "data.map")}
    if data.get("traffic_features"):
        world["traffic"] = {"features_file":
                            _resolve_data(data["traffic_features"], "data.traffic_features")}
    overlay["world"] = world
    if data.get("transit"):                              # 無ければ縮退(徒歩の街)=基底の transit を使う
        overlay["transit"] = {"file": _resolve_data(data["transit"], "data.transit")}
    if data.get("personas"):
        overlay["agents"] = {"personas_file":
                             _resolve_data(data["personas"], "data.personas")}
    orgs: dict = {}
    if data.get("organizations"):
        orgs["book"] = _resolve_data(data["organizations"], "data.organizations")
    if data.get("assignments"):
        orgs["assignments"] = _resolve_data(data["assignments"], "data.assignments")
    if orgs:
        overlay["organizations"] = orgs

    # ---- ③-B envpack(語彙・行事・気候・番組名・原点・駅名フィルタ)→ envpack.* ----
    culture = doc.get("culture") or {}
    envpack: dict = {}
    if culture.get("lexicon"):
        envpack["lexicon"] = dict(culture["lexicon"])
    if culture.get("events") is not None:
        envpack["culture"] = {"events": list(culture["events"])}
    if culture.get("media"):
        envpack["media"] = dict(culture["media"])
    if (doc.get("origin") or {}).get("landmark"):
        envpack["origin"] = {"landmark": doc["origin"]["landmark"]}
    if doc.get("climate"):
        envpack["climate"] = dict(doc["climate"])
    if (doc.get("transit") or {}).get("station_filters"):
        envpack["transit"] = {"station_filters": list(doc["transit"]["station_filters"])}
    if envpack:
        overlay["envpack"] = envpack

    # ---- ③-A 街頭広告の掲出地点 → ads.*(ads.enabled は env が触らない=既定 OFF のまま)----
    ads = doc.get("ads") or {}
    ads_ov = {k: list(ads[k]) for k in ("large", "slots") if k in ads}
    if ads_ov:
        overlay["ads"] = ads_ov

    # ---- ② 制度: pref セレクタ + ref 解決 → institutions.* / economy.min_wage_hourly ----
    inst = doc.get("institutions") or {}
    if inst.get("pref"):
        ref_rel = inst.get("ref")
        ref_path = None
        if ref_rel:
            rp = Path(ref_rel)
            ref_path = rp if rp.is_absolute() else (base_dir / rp)
        if ref_path is None or not ref_path.exists():
            ref_path = REPO_ROOT / "ref" / "institutions_jp.yaml"
        if not ref_path.exists():
            raise FileNotFoundError(f"institutions.ref が実在しない: {ref_path}")
        ref = OmegaConf.to_container(OmegaConf.load(ref_path), resolve=True) or {}
        prefs = ref.get("prefectures") or {}
        pref = inst["pref"]
        if pref not in prefs:
            raise ValueError(f"ref に prefecture '{pref}' が無い: {ref_path}")
        pv = prefs[pref] or {}
        nat = ref.get("national") or {}
        inst_ov: dict = {}
        if nat.get("income_tax_effective"):             # ref は upto キー → institutions は up_to
            inst_ov["income_brackets"] = [
                {"up_to": b.get("upto"), "rate": b["rate"]}
                for b in nat["income_tax_effective"]]
        if nat.get("consumption"):
            cc = nat["consumption"]
            inst_ov["consumption"] = {
                "rate": cc.get("rate"), "reduced_rate": cc.get("reduced"),
                "national_share": cc.get("national_share"),
                "reduced_cats": list(cc.get("reduced_cats") or ["food"])}
        res: dict = {}
        if nat.get("resident_tax", {}).get("rate") is not None:
            res["rate"] = nat["resident_tax"]["rate"]
        if pv.get("special_ward_resident_split"):
            res["ward_share"] = pv["special_ward_resident_split"][0]
        if res:
            inst_ov["resident"] = res
        if (nat.get("labor") or {}).get("annual_workdays") is not None:
            inst_ov["labor"] = {"annual_workdays": nat["labor"]["annual_workdays"]}
        if inst_ov:
            overlay["institutions"] = inst_ov
        if pv.get("min_wage_hourly") is not None:       # 最賃=都道府県スコープ → economy
            overlay["economy"] = {"min_wage_hourly": pv["min_wage_hourly"]}

    # ---- ③ 地点固有の制度: council → institution_routes / rent_income_ratio → economy.accounts ----
    council = inst.get("council") or {}
    ir: dict = {}
    asm = {k: council[k] for k in ("size", "term_days") if council.get(k) is not None}
    if asm:
        ir["assembly"] = asm
    if council.get("deposit") is not None:
        ir["vote"] = {"deposit": council["deposit"]}
    if ir:
        overlay["institution_routes"] = ir
    if inst.get("rent_income_ratio") is not None:
        overlay.setdefault("economy", {}).setdefault("accounts", {})["rent_share"] = \
            inst["rent_income_ratio"]

    return overlay


def load_config(overrides: list[str] | None = None,
                path: str | Path | None = None,
                profile: str | Path | None = None,
                env: str | Path | None = None) -> DictConfig:
    """基底 config.yaml(または path)に、env → profile 差分 YAML → dotlist の順で重ねる。

    env:     EnvPack(env/<place> ディレクトリ or env.yaml パス)。「場所」を束ねた manifest を
             既存 config キー群へ写像して重ねる(build_env_overlay)。
    profile: 本番用など「基底との差分だけ」を書いた YAML(conf/production.yaml 等)。
    優先順位: 基底 < env < profile < dotlist(env=「場所」、profile=「用途」なので profile が勝つ)。
    """
    cfg = OmegaConf.load(Path(path) if path else DEFAULT_CONFIG)
    if env is not None:
        cfg = OmegaConf.merge(cfg, build_env_overlay(env))
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
