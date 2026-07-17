"""娯楽メディア(TV・動画・ゲーム)の在宅利用。既定 OFF(バッチD)。

現代人の生活再現に不可欠な在宅娯楽の提供層。役割分担は economy.py と同格:
- 年齢・職業 → 媒体構成比・開始確率・セッション長 の写像はここ(persona/sim 側)で
  precompute する。cognition は性格特性(traits/因子)を直接読まない(R9)。ここは
  **人口統計属性(age/occupation)だけ**を見て因子には触れないため、指紋契約
  (tests/test_contracts)にも無関係で安全。
- 文献接地は docs/lit/media__entertainment-effects.md:
    * 気分管理理論(Zillmann): 娯楽視聴はネガ気分を和らげる → grievance 修復。
    * 利用と満足(Katz): 気晴らし+話題獲得 → prompt_context で直近視聴タイトル1行。
    * 時間置換(Putnam/Moy): セッション中は在宅で外出・SNS閲覧をしない(機会費用)。
    * 回復体験(Sonnentag&Fritz)/ ゲーム気分改善(Vuorre 2024: +0.034 / 15分)。
    * NHK 国民生活時間調査 2020: 年層差(若年=動画・ゲーム、高齢=TV)・夜間/朝の帯。
- 因果構造だけ接地し、量(magnitude・確率)は conf の media / factors ブロックで可変。

タイトルは**架空プールのみ**(実在番組・作品名は使わない=R17)。乱数は新 stream
"media" だけを使う(既存 stream の draw 順には一切触れない)。
"""
from __future__ import annotations

import numpy as np
from omegaconf import OmegaConf

# 既定値(conf.media で上書き可)。既定 OFF=挙動は完全に旧来どおり。
_DEFAULTS = {
    "enabled": False,          # 娯楽メディア層全体の ON/OFF
    "prompt_context": False,   # 直近視聴タイトルを発火プロンプト文脈へ(独立フラグ)
    "start_prob": 0.5,         # 在宅×時間帯の1機会あたり基礎開始確率(×プロファイル倍率)
    "min_steps": 2,            # セッション長の下限(step。1 step=10分)
    "max_steps": 8,            # セッション長の上限(step)。既定 2〜8=20〜80分
    "morning_hours": [5, 10],  # 起床後の朝の帯(hour ∈ [lo, hi))
    "night_hours": [20, 24],   # 夜間・在宅後の帯(hour ∈ [lo, hi))
    "max_sessions_per_day": 2, # 1日あたりの最大セッション数(朝+夜)
}

_MEDIA = ("tv", "video", "game")
_MEDIUM_JP = {"tv": "テレビ", "video": "動画", "game": "ゲーム"}

# 年齢帯 → 媒体構成比 (tv, video, game)。NHK 2020 の年層差を写す
# (若年=動画・ゲーム中心/TV離れ、高齢=実時間TV中心)。
_MEDIUM_WEIGHTS = {
    "youth":  (0.20, 0.45, 0.35),   # <30
    "mid":    (0.45, 0.35, 0.20),   # 30-49
    "senior": (0.65, 0.25, 0.10),   # 50-64
    "elder":  (0.85, 0.12, 0.03),   # 65+
}
# 年齢帯 → 利用頻度(propensity)とセッション長の相対倍率。
_BAND_PROPENSITY = {"youth": 1.2, "mid": 0.9, "senior": 1.0, "elder": 1.1}
_BAND_LENGTH = {"youth": 0.9, "mid": 1.0, "senior": 1.2, "elder": 1.3}

# 職業 → 自由時間の多寡(開始確率倍率)。未知職業は 1.0(=不変)。
_OCC_FREE = {
    "大学生": 1.25, "無職": 1.3, "フリーランス": 1.15, "バンドマン": 1.2,
    "写真家": 1.1, "配達員": 1.0, "会社員": 0.9, "エンジニア": 0.9,
    "デザイナー": 0.95,
}

# 架空タイトルの generic フォールバック(実在番組・作品名は使わない=R17)。
# 地名を含む番組名は基盤に置かず envpack.media(場所の値)から受ける。実運用では
# routine が envpack のタイトルを渡すため、この generic 既定はスタンドアロン利用の保険。
_TITLES = {
    "tv": ["夜のニュース", "トークナイト", "みんなの気象台"],
    "video": ["路地裏グルメ散歩", "10分でわかる都市伝説", "猫と暮らす部屋"],
    "game": ["パズル横丁", "ネオンレーサー2"],
}


def media_cfg(cfg) -> dict:
    """conf(OmegaConf)→ 実行時の media 設定 dict。media ブロックが無ければ既定(OFF)。

    既定 OFF なので media を書いていない config では enabled=False=挙動バイト一致。"""
    raw = OmegaConf.select(cfg, "media")
    raw = OmegaConf.to_container(raw, resolve=True) if raw is not None else {}
    raw = raw or {}
    out = dict(_DEFAULTS)
    for k, v in raw.items():
        if k in out:
            out[k] = v
    out["enabled"] = bool(out["enabled"])
    out["prompt_context"] = bool(out["prompt_context"])
    out["start_prob"] = float(out["start_prob"])
    out["min_steps"] = max(1, int(out["min_steps"]))
    out["max_steps"] = max(out["min_steps"], int(out["max_steps"]))
    out["morning_hours"] = [int(x) for x in out["morning_hours"]][:2]
    out["night_hours"] = [int(x) for x in out["night_hours"]][:2]
    out["max_sessions_per_day"] = max(0, int(out["max_sessions_per_day"]))
    return out


def _age_band(age: int) -> str:
    if age < 30:
        return "youth"
    if age < 50:
        return "mid"
    if age < 65:
        return "senior"
    return "elder"


def profile_for(agent, mcfg: dict) -> dict:
    """agent の age/occupation から媒体構成・頻度・長さの個人プロファイルを precompute。

    人口統計属性のみ(traits/因子は見ない=R9)。決定論(乱数を引かない)。agent へ
    キャッシュする(_media_profile)。economy の precompute(wage 等)と同じ設計。"""
    cached = getattr(agent, "_media_profile", None)
    if cached is not None:
        return cached
    band = _age_band(int(getattr(agent, "age", 40)))
    occ_mult = _OCC_FREE.get(getattr(agent, "occupation", ""), 1.0)
    prof = {
        "band": band,
        "weights": _MEDIUM_WEIGHTS[band],
        "propensity": _BAND_PROPENSITY[band] * occ_mult,
        "length_mult": _BAND_LENGTH[band],
    }
    agent._media_profile = prof
    return prof


def start_prob(prof: dict, mcfg: dict) -> float:
    """在宅×時間帯の1機会あたり開始確率(基礎 × 年齢帯・職業の頻度倍率、[0,0.95] に clip)。"""
    p = float(mcfg["start_prob"]) * float(prof["propensity"])
    return max(0.0, min(0.95, p))


def session_length(prof: dict, rng: np.random.Generator, mcfg: dict) -> int:
    """セッション長(step)。既定 2〜8 step を年齢帯の長さ倍率でスケール(高齢ほど長め)。"""
    lo, hi = int(mcfg["min_steps"]), int(mcfg["max_steps"])
    base = int(rng.integers(lo, hi + 1))
    steps = int(round(base * float(prof["length_mult"])))
    return max(1, min(hi + 2, steps))


def pick_medium(prof: dict, rng: np.random.Generator) -> str:
    """媒体(tv/video/game)を年齢帯の構成比で重み付き抽選。"""
    w = np.array(prof["weights"], dtype=float)
    total = w.sum()
    if total <= 0.0:
        return _MEDIA[int(rng.integers(len(_MEDIA)))]
    w = w / total
    return _MEDIA[int(rng.choice(len(_MEDIA), p=w))]


def pick_title(medium: str, rng: np.random.Generator,
               titles: dict | None = None) -> str:
    """架空タイトルを1件抽選(実在作品名は含まない=R17)。

    titles(envpack.media=場所の値)があればそれを使う。無ければ generic 既定(_TITLES)。
    """
    pool_src = titles or _TITLES
    pool = pool_src.get(medium) or pool_src.get("tv") or _TITLES["tv"]
    return pool[int(rng.integers(len(pool)))]


def medium_jp(medium: str) -> str:
    return _MEDIUM_JP.get(medium, medium)
