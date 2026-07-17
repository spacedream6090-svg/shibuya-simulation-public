"""環境パック(EnvPack)設定の正準化(基盤モデル抽出 D1・第16/35バッチ)。

「機構=基盤・値=環境」の分離: src(基盤)は場所の知識(地名・ランドマーク・年中行事・
気候・語彙・メディア番組名)を **持たない**。ここで conf の ``envpack`` ブロックを1回だけ
正準化して sim に保持し、各所は cfg 経由で読む(深い get の散在をつくらない=既存 build_cfg 流儀)。

既定値(=現在の渋谷リテラル)は conf/config.yaml の ``envpack:`` ブロックに置く(conf に
地名があるのが正しい状態)。W4 の env.yaml(env/<place>/)がこのブロックを上書きする設計。

このモジュール自身は地名リテラルを持たない: コード側の fallback は generic(「この街」等)
または空にとどめ、実際の渋谷の値は必ず conf から与える(契約テストの地名禁止ガードと整合)。
"""
from __future__ import annotations

# 場所名の generic fallback(実値は conf/envpack.lexicon.place_name)。地名ではない。
_GENERIC_PLACE = "この街"
_GENERIC_UNDERGROUND = "地下通路"

# メディア番組名の generic fallback(地名を含まない=実値は conf/envpack.media)。
_GENERIC_MEDIA = {
    "tv": ["夜のニュース", "トークナイト", "みんなの気象台"],
    "video": ["路地裏グルメ散歩", "10分でわかる都市伝説", "猫と暮らす部屋"],
    "game": ["パズル横丁", "ネオンレーサー2"],
}


def _events(raw_events) -> list[dict]:
    """年中行事リストを正準化(名前=文化語彙・地名ではない。annual.build_cfg と同形)。"""
    out: list[dict] = []
    for e in (raw_events or []):
        e = dict(e)
        out.append({
            "name": str(e.get("name", "")),
            "month": int(e.get("month", 1)),
            "day": int(e.get("day", 1)),
            "crowd": bool(e.get("crowd", False)),
        })
    return out


def _monthly(raw_monthly) -> dict:
    """月別気候テーブルを {月(int): [最高, 最低, 雨重み, 雪重み]} に正準化。空なら {}。"""
    out: dict[int, list[float]] = {}
    for k, v in dict(raw_monthly or {}).items():
        vals = list(v)
        out[int(k)] = [float(vals[0]), float(vals[1]),
                       float(vals[2]), float(vals[3])]
    return out


def build_cfg(raw: dict | None) -> dict:
    """conf の envpack ブロックを正準化する。raw が None/空でも generic 既定で成立する。

    返す dict はネスト固定(lexicon / transit / culture / media / origin / climate)。
    実際の渋谷の値は conf/config.yaml の envpack ブロックが供給する(src には残さない)。
    """
    raw = dict(raw or {})
    lex = dict(raw.get("lexicon", {}) or {})
    transit = dict(raw.get("transit", {}) or {})
    culture = dict(raw.get("culture", {}) or {})
    media = dict(raw.get("media", {}) or {})
    origin = dict(raw.get("origin", {}) or {})
    climate = dict(raw.get("climate", {}) or {})

    place_name = str(lex.get("place_name", _GENERIC_PLACE))
    underground_name = str(lex.get("underground_name", _GENERIC_UNDERGROUND))
    place_hints = [str(h) for h in (lex.get("place_hints", []) or [])]

    station_filters = [str(s) for s in (transit.get("station_filters", []) or [])]

    events = _events(culture.get("events"))

    media_out = {}
    for kind in ("tv", "video", "game"):
        vals = media.get(kind)
        media_out[kind] = ([str(t) for t in vals] if vals
                           else list(_GENERIC_MEDIA[kind]))

    monthly = _monthly(climate.get("monthly"))
    neutral_raw = climate.get("neutral")
    neutral = ([float(x) for x in neutral_raw] if neutral_raw else None)

    return {
        "lexicon": {
            "place_name": place_name,
            "underground_name": underground_name,
            "place_hints": place_hints,
        },
        "transit": {"station_filters": station_filters},
        "culture": {"events": events},
        "media": media_out,
        "origin": {"landmark": str(origin.get("landmark", ""))},
        "climate": {"monthly": monthly, "neutral": neutral},
    }
