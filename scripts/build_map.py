"""OSM(Overpass API)から渋谷中心部の実道路網+全建物+POI+地下/デッキを取得し、地図を生成する。

使い方:
    python scripts/build_map.py                                   # → data/shibuya_osm.json(既定範囲・最新)
    python scripts/build_map.py --out data/shibuya_osm_wide.json \
        --bbox 35.65226 139.68956 35.66674 139.71168              # 拡張範囲(約2.0×1.6km)
    python scripts/build_map.py --osm-date 2023-04-01 --out data/shibuya_osm_2023.json
        # Overpass attic query で過去スナップショット(PLATEAU 基準日など)を取得

出典: OpenStreetMap contributors(ODbL)。ダウンロードは Overpass 公式 API のみ使用。

v8 の追加(第101バッチ Wave4 III-2「地図 v8」):
  - POI に **subcat(サブカテゴリ)** を付ける第2階層。**13 個の top cat は 1 つも動かさない**
    (day_plan.PLACE_CATS / commerce の営業時間表 / vision の salience 表が cat に依存する)。
    v7 では OSM の生タグが捨てられており、コンビニ・ネカフェ・サウナ・パチンコ・ゲーセン・
    神社仏閣・病院・ラブホ・保育が「識別不能」または「そもそも取り込まれない」状態だった。
  - 取り込み拡大は **v7 が None を返した POI だけ**に効く(下の poi_category を参照:
    先に v7 と 1 バイト同じ判定を通し、None のときだけ subcat 由来の top cat へ落とす)
    = v7 で cat が付いていた POI の cat は**構造的に変わりえない**。
  - POI の重複排除キーを (name, cat) → (name, cat, node) へ。旧キーはチェーン店を
    全店 1 件へ潰しており、コンビニは実測 93 → 9 件(9 割消失)だった。
  - --raw-out / --raw-in: Overpass 生データを保存/再利用(同じ取得を何度も投げない礼儀。
    取得日の違う版・パラメータ違いの版をネットワーク無しで組み直せる)。

v6 の追加:
  - bbox / 取得日 / 出力先をコマンドラインで指定可能(既定=現行範囲・最新)
  - entrance=main/yes/exit ノード(建物外周上)を取り込み building.entrances に格納。
    main があれば building.entrance(最寄り道路ノード)を main 入口に最も近い道路ノードへ置換。
  - 地下判定を強化: highway=footway で layer<0 / level<0 / tunnel=(yes|building_passage) を
    地下(layer=-1)として取り込む(渋谷ちかみち等の地下歩行者網が routing にそのまま載る)。

v3 の追加:
  - 建物を全件取得(住宅・マンション含む)+用途分類 kind → 家の割当に使う
  - POI(飲食店・会社・店舗など)を実名で取得し、建物・最寄りノードに紐付け
  - 道路エッジに layer(-1=地下, 0=地上, 1=デッキ)を付与 → 渋谷ちかみち等
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# 既定範囲: 渋谷駅中心 約1.0km×0.7km(south, west, north, east)
DEFAULT_BBOX = (35.6560, 139.6950, 35.6625, 139.7060)
# スクランブル交差点 ≒ ローカル座標原点(bbox を変えても不変)
ORIGIN = (35.65950, 139.70062)

HIGHWAY_CLASSES = {
    "primary", "secondary", "tertiary", "unclassified", "residential",
    "living_street", "service", "pedestrian", "footway", "path", "steps",
    "cycleway", "corridor", "elevator",
}
# 車が走れる道 / 自転車が走れる道(徒歩は全部OK)
DRIVABLE = {"primary", "secondary", "tertiary", "unclassified", "residential",
            "living_street", "service"}
CYCLABLE = HIGHWAY_CLASSES - {"steps", "corridor", "elevator"}

# entrance タグの種別(建物外周ノード)。main を最優先で建物入口に採用する。
ENTRANCE_KINDS = {"main", "yes", "exit", "service", "staircase", "home", "emergency"}

# 実在ランドマーク(名前を最寄りノードに付与し、目的地候補=POI にする)
LANDMARKS = [
    ("スクランブル交差点", 35.65950, 139.70062, "square"),
    ("ハチ公前広場",       35.65905, 139.70046, "square"),
    ("渋谷駅ハチ公口",     35.65880, 139.70100, "station"),
    ("SHIBUYA109",         35.65946, 139.69838, "landmark"),
    ("渋谷センター街",     35.66030, 139.69820, "shopping"),
    ("道玄坂",             35.65800, 139.69600, "street"),
    ("文化村通り",         35.65990, 139.69590, "street"),
    ("スペイン坂",         35.66120, 139.69850, "street"),
    ("渋谷PARCO",          35.66210, 139.69880, "landmark"),
    ("公園通り",           35.66150, 139.69970, "street"),
    ("宮下公園",           35.66185, 139.70290, "park"),
    ("宮益坂下",           35.65920, 139.70330, "street"),
    ("渋谷ストリーム",     35.65710, 139.70260, "landmark"),
    ("渋谷マークシティ",   35.65850, 139.69800, "landmark"),
]

# 明示的に住宅系とわかる building タグ
RESIDENTIAL_TYPES = {"residential", "apartments", "house", "detached",
                     "dormitory", "terrace", "semidetached_house", "hut"}

# 待ち合わせ名所の名称マッチ(ハチ公・モヤイ・忠犬 等)。名前にこれを含む POI は
# landmark カテゴリに寄せる(OSM のタグが amenity/tourism 等でも待ち合わせ地点として扱う)。
LANDMARK_NAME_KWS = ("忠犬", "ハチ公", "モヤイ", "モアイ")

# ハチ公像の明示座標フォールバック(OSM で landmark として拾えなかった時に data へ置く)。
# 実在座標(渋谷駅ハチ公口前・忠犬ハチ公像)。cat=landmark で待ち合わせ候補に載る。
HACHIKO_FALLBACK = ("忠犬ハチ公像", 35.6590, 139.7005)


def build_query(bbox: tuple[float, float, float, float],
                osm_date: str | None = None) -> str:
    """Overpass QL を bbox から生成。osm_date 指定時は attic query(過去スナップショット)。"""
    date_setting = ""
    if osm_date:
        # "YYYY-MM-DD" → ISO8601 UTC。Overpass の [date:"…"] で当時の版を取得。
        date_setting = f'[date:"{osm_date}T00:00:00Z"]'
    header = f"[out:json][timeout:180]{date_setting};"
    return header + """
(
  way["highway"](%s,%s,%s,%s);
  way["building"](%s,%s,%s,%s);
  way["railway"~"^(rail|subway)$"](%s,%s,%s,%s);
  node["amenity"](%s,%s,%s,%s);
  node["shop"](%s,%s,%s,%s);
  node["office"](%s,%s,%s,%s);
  node["tourism"](%s,%s,%s,%s);
  node["leisure"](%s,%s,%s,%s);
  way["amenity"](%s,%s,%s,%s);
  way["shop"](%s,%s,%s,%s);
  way["office"](%s,%s,%s,%s);
  way["leisure"](%s,%s,%s,%s);
);
(._;>;);
out body;
""" % (bbox * 12)


def fetch_overpass(bbox: tuple[float, float, float, float],
                   osm_date: str | None = None, retries: int = 3) -> dict:
    """Overpass 公式 API から取得。混雑・タイムアウト時はバックオフ付きでリトライ。

    複数のパブリック・ミラーを巡回する(504 Gateway Timeout=サーバ過負荷への耐性)。
    いずれも OpenStreetMap contributors(ODbL)のデータを提供する Overpass 実装。"""
    query = build_query(bbox, osm_date)
    body = ("data=" + urllib.request.quote(query)).encode("utf-8")
    endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass.private.coffee/api/interpreter",
        "https://overpass.osm.jp/api/interpreter",
    ]
    last_err: Exception | None = None
    for attempt in range(retries):
        endpoint = endpoints[attempt % len(endpoints)]
        try:
            req = urllib.request.Request(
                endpoint, data=body,
                headers={"Content-Type": "application/x-www-form-urlencoded",
                         "User-Agent": "shibuya-simulation research (contact: local)"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
            last_err = e
            wait = 15 * (attempt + 1)
            print(f"  取得失敗({type(e).__name__}: {e}) — {wait}s 待って再試行 "
                  f"[{attempt + 1}/{retries}]", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"Overpass 取得に失敗(全リトライ): {last_err}")


# 原点未指定時に build() が使う既定原点のセンチネル(None=既定=ORIGIN=スクランブル交差点)。
_DEFAULT = object()


def _make_project(origin: tuple[float, float]):
    """原点(lat, lon)を束ねた projector を返す。汎用化: 街ごとに原点を差し替える。

    既定原点=ORIGIN(スクランブル交差点)のとき、返る関数は従来 project と完全同値。"""
    o_lat, o_lon = origin[0], origin[1]
    cos_lat = math.cos(math.radians(o_lat))

    def _project(lat: float, lon: float) -> tuple[float, float]:
        x = (lon - o_lon) * 111320.0 * cos_lat
        y = (lat - o_lat) * 110540.0
        return round(x, 1), round(y, 1)

    return _project


def project(lat: float, lon: float) -> tuple[float, float]:
    """緯度経度 → 原点基準ローカル平面(m)。1km 級なら十分な近似。

    モジュール既定原点(ORIGIN=スクランブル交差点)版。街を差し替える build() 内では
    `project = _make_project(origin)` の局所束縛が優先される(=同名の局所変数で影を作る)。"""
    return _make_project(ORIGIN)(lat, lon)


def polyline_length(points: list[tuple[float, float]]) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1])
               for a, b in zip(points, points[1:]))


def _min_level(raw: str) -> int | None:
    """OSM の level タグ(例 "-1", "-2;-1", "0") から最下層を整数で返す。"""
    parts = []
    for p in str(raw).replace(";", ",").split(","):
        p = p.strip()
        if not p:
            continue
        try:
            parts.append(int(float(p)))
        except ValueError:
            continue
    return min(parts) if parts else None


def way_layer(tags: dict) -> int:
    """道路の垂直レイヤー: -1=地下(ちかみち等) / 0=地上 / 1=ペデストリアンデッキ。

    layer タグを最優先。無ければ level タグ(地下歩行者網は level=-1 等)を補助的に読む。
    tunnel=(yes|building_passage) は地下、bridge=yes はデッキ扱い。
    """
    layer: int | None = None
    raw_layer = tags.get("layer")
    if raw_layer is not None:
        try:
            layer = int(float(raw_layer))
        except ValueError:
            layer = None
    if layer is None:
        lvl = tags.get("level")
        if lvl is not None:
            layer = _min_level(lvl)
    if layer is None:
        layer = 0
    if tags.get("tunnel") in ("yes", "building_passage") and layer >= 0:
        layer = -1
    if tags.get("bridge") == "yes" and layer <= 0:
        layer = 1
    return max(-2, min(2, layer))


def point_in_poly(x: float, y: float, poly: list) -> bool:
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i][0], poly[i][1]
        xj, yj = poly[j][0], poly[j][1]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


# --------------------------------------------------------------------------- #
# サブカテゴリ(v8)= cat の中の「実態の違い」を運ぶ第2階層。
#
# 設計の掟(3つ):
#   (1) **top cat は増やさない・動かさない**。day_plan.PLACE_CATS(15語)・commerce の
#       営業時間表(food/shop/nightlife)・vision の salience 表・joint の poi_cat が
#       cat の語彙に依存しているので、cat を増やすと 4 箇所の意味論が同時にずれる。
#   (2) subcat は **OSM の生タグからの純関数**。POI 名(ブランド名)は 1 文字も見ない
#       (ETHICS §2: 実在企業名をソースへ書かない。night.py の subcat_keywords が
#        「既定で空」なのと同じ線引き)。
#   (3) 語彙は **閉じた 11 語**。消費側(src/society/night.py の DEFAULT_HOURS /
#       refuge.subcats)が知っている convenience / net_cafe / sauna / karaoke / club を
#       含む上位集合で、残り 6 語は「v7 で取りこぼしていた実在業態」。
#
# subcat → top cat の割り当て理由(1 語ずつ):
#   convenience → shop     … v7 でも shop=convenience は cat=shop。**cat は不変**のまま
#                            24h 営業(night.hours)だけが subcat で引き当てられるようにする。
#   pachinko    → shop     … v7 で shop=pachinko は既に cat=shop。amenity=gambling だけの
#                            パーラーも同じ箱へ寄せる(昼から開く遊技場=商業。nightlife に
#                            移すと 18-05 になり実態<10:00-23:00>から遠のく)。
#   karaoke     → nightlife… v7 で amenity=karaoke_box は既に nightlife。**cat は不変**。
#   club        → nightlife… v7 で amenity=nightclub は既に nightlife。**cat は不変**。
#   net_cafe    → nightlife… v7 では取り込まれず(amenity=internet_cafe はどの分岐にも
#                            当たらない)。終電後の受け皿=夜の業態なので nightlife。
#   sauna       → nightlife… 同上(leisure=sauna / amenity=public_bath は v7 で落ちる)。
#                            銭湯は昼も開くが、既定の窓は leisure(常時)より nightlife
#                            (18-05)の方が「深夜に開いている」実態に近い。night_economy
#                            ON では subcat=sauna が 24h を引き当てて上書きする。
#   arcade      → leisure  … ゲームセンターは遊興施設。leisure は commerce の営業時間表に
#                            エントリが無い=常時営業扱いで、10:00-23:30 の実態に対して
#                            shop(10-21)より害が小さい。
#   worship     → landmark … 神社仏閣・教会は待ち合わせ/目印の場所(city.dests が
#                            cat=landmark のノードを目的地に載せる)。金王八幡宮が v7 で
#                            手動パッチ(attraction)だった穴を OSM 側で埋める。
#   hospital    → service  … v7 は clinic/dentist/pharmacy だけを service にしていた。
#                            総合病院を同じ箱へ(公共サービス施設)。
#   love_hotel  → hotel    … 宿泊施設。lodging の宿カテゴリと同じ箱。
#   childcare   → school   … 保育園・幼稚園。v7 で kindergarten は既に cat=school なので
#                            **cat は不変**、childcare(認可外・保育所)を同じ箱へ足す。
SUBCAT_TOPCAT = {
    "convenience": "shop",
    "pachinko": "shop",
    "karaoke": "nightlife",
    "club": "nightlife",
    "net_cafe": "nightlife",
    "sauna": "nightlife",
    "arcade": "leisure",
    "worship": "landmark",
    "hospital": "service",
    "love_hotel": "hotel",
    "childcare": "school",
}
# 消費側(src/society/night.py)が既に知っている語(v8 で初めて実データが当たる)。
NIGHT_SUBCATS = ("convenience", "net_cafe", "sauna", "karaoke", "club")


# サブカテゴリの判定表(**キーを跨いだ値マッチ**)。上から順に評価する決定論。
#
# ★なぜキー(amenity/shop/leisure/tourism)を固定しないのか: 日本の OSM は同じ業態を
#   別のキーに載せる揺れが大きく、渋谷 bbox の実測(2025-04-01 スナップショット)では
#     ラブホテル  amenity=love_hotel 63 件 / tourism=love_hotel **0 件**
#     ゲーセン    leisure=amusement_arcade 3 件 / amenity=amusement_arcade **0 件**
#                 leisure=adult_gaming_centre 1 件
#     カラオケ    amenity=karaoke_box 16 件 / leisure=karaoke 2 件
#     パチンコ    amenity=gambling 8 件 / shop=pachinko **0 件**
#   だった(= キーで決め打つと、実在する業態の 9 割を取り逃す)。値の方が業態を一意に
#   指しているので、4 つのキーに載った値の集合で照合する。
SUBCAT_TAG_VALUES: tuple[tuple[str, frozenset], ...] = (
    ("convenience", frozenset({"convenience"})),
    ("pachinko",    frozenset({"pachinko", "gambling"})),
    ("karaoke",     frozenset({"karaoke_box", "karaoke"})),
    ("club",        frozenset({"nightclub"})),
    ("net_cafe",    frozenset({"internet_cafe", "manga_cafe"})),
    ("sauna",       frozenset({"sauna", "public_bath"})),
    ("arcade",      frozenset({"amusement_arcade", "adult_gaming_centre"})),
    ("worship",     frozenset({"place_of_worship"})),
    ("hospital",    frozenset({"hospital"})),
    ("love_hotel",  frozenset({"love_hotel"})),
    ("childcare",   frozenset({"childcare", "kindergarten"})),
)
# 業態を載せうる OSM のキー(この 4 つ以外は見ない = 建物タグ等の誤爆を避ける)。
SUBCAT_TAG_KEYS = ("amenity", "shop", "leisure", "tourism")


def poi_subcategory(tags: dict) -> str | None:
    """OSM タグ → POI サブカテゴリ(閉じた 11 語)。判らなければ None。

    **名前を見ない**(生タグだけの純関数)。判定順は SUBCAT_TAG_VALUES の並び=決定論。
    ここが返す語は必ず SUBCAT_TOPCAT のキー = 語彙は機械で閉じている。"""
    values = {tags.get(k) for k in SUBCAT_TAG_KEYS}
    values.discard(None)
    if not values:
        return None
    for sub, vals in SUBCAT_TAG_VALUES:
        if values & vals:
            return sub
    return None


def poi_category(tags: dict, landmark_name_kws: tuple = LANDMARK_NAME_KWS) -> str | None:
    """OSM タグ → POI カテゴリ(v8: v7 の判定 → 取りこぼしを subcat の top cat で救済)。

    **v7 で cat が付いていた POI の cat は絶対に変わらない**: 先に v7 と 1 バイト同じ
    `_poi_category_v7` を通し、それが None のときだけ subcat 由来の top cat へ落とす。
    よって v8 の POI 集合は v7 の**上位集合**(cat の付け替えは起きえない)。"""
    cat = _poi_category_v7(tags, landmark_name_kws)
    if cat is not None:
        return cat
    sub = poi_subcategory(tags)
    if sub is not None:
        return SUBCAT_TOPCAT[sub]
    return None


def _poi_category_v7(tags: dict, landmark_name_kws: tuple = LANDMARK_NAME_KWS) -> str | None:
    """OSM タグ → POI カテゴリ。v6 拡大(ユーザー要望 2026-07-06): 飲食だけでなく
    office(会社)/school(学校)/cinema(映画館)/hall(イベントホール・劇場)/
    landmark(待ち合わせ名所)まで対象を広げる。既存カテゴリ体系(food/nightlife/
    service/shop/office/hotel/attraction/leisure)に整合させて新値を追加する。
    ★旧地図(data/shibuya_osm.json)は再生成しない=旧 cat のまま。本関数の変更は
      新規に生成する地図(v6/wide)にのみ効く。

    landmark_name_kws: 名称マッチで landmark に寄せる待ち合わせ名所のキーワード。既定=
      渋谷(ハチ公/モヤイ/忠犬)。別の街では make_env/CLI が街固有語または空 () を渡す。"""
    name = tags.get("name:ja") or tags.get("name") or ""
    # 待ち合わせ名所(名称マッチ最優先: ハチ公/モヤイ/忠犬)。
    if landmark_name_kws and any(kw in name for kw in landmark_name_kws):
        return "landmark"
    amenity = tags.get("amenity", "")
    if amenity in ("restaurant", "cafe", "fast_food", "food_court",
                   "ice_cream"):
        return "food"
    if amenity in ("bar", "pub", "nightclub", "karaoke_box", "izakaya"):
        return "nightlife"
    if amenity == "cinema":                              # 映画館(商業施設)
        return "cinema"
    if amenity in ("theatre", "events_venue", "conference_centre") \
            or tags.get("leisure") == "stadium":         # イベントホール・劇場・スタジアム
        return "hall"
    if amenity in ("bank", "pharmacy", "clinic", "dentist", "post_office",
                   "police", "library"):
        return "service"
    if amenity in ("school", "university", "college", "kindergarten") \
            or tags.get("building") in ("school", "university"):  # 学校・大学
        return "school"
    if "shop" in tags:
        return "shop"
    if "office" in tags:                                 # 会社・事務所
        return "office"
    if tags.get("tourism") in ("hotel", "hostel", "guest_house"):
        return "hotel"
    # 観光名所・彫像・記念碑は待ち合わせ名所(landmark)として扱う。
    if tags.get("tourism") in ("attraction", "artwork") \
            or tags.get("man_made") == "statue" \
            or tags.get("historic") in ("memorial", "monument"):
        return "landmark"
    if tags.get("tourism") in ("museum", "gallery"):
        return "attraction"
    if tags.get("leisure") in ("park", "garden", "playground", "pitch",
                               "fitness_centre", "sports_centre"):
        return "leisure"
    return None


def building_kind(tags: dict, area: float, name: str | None) -> str:
    btype = tags.get("building", "yes")
    use = tags.get("building:use", "")
    if btype in RESIDENTIAL_TYPES or use == "residential":
        return "residential"
    if btype in ("office", "commercial") or use == "office":
        return "office"
    if btype == "retail" or use == "retail":
        return "retail"
    if btype in ("train_station", "transportation", "station"):
        return "station"
    if btype in ("hotel",):
        return "hotel"
    if btype in ("school", "university", "college", "civic", "public",
                 "government", "hospital"):
        return "public"
    # 無名の小規模建物は住宅の可能性が高い(渋谷でも桜丘・神南の縁は住宅地)
    if name is None and area < 260:
        return "house?"
    return "generic"


def build(raw: dict, bbox: tuple[float, float, float, float],
          osm_date: str | None = None, *,
          origin: tuple[float, float] | None = None,
          landmarks: list | None = None,
          landmark_name_kws=_DEFAULT,
          hachiko_fallback=_DEFAULT,
          map_name: str | None = None,
          description: str | None = None,
          fetched_at: str | None = None) -> dict:
    """OSM 生データ → シミュ地図 JSON。

    汎用化(D2): 街固有の定数を引数化。**すべて None/既定=現行渋谷値**なので、位置引数だけで
    呼ぶ従来の使い方(build(raw, bbox, date))は出力が完全同値。別の街は make_env/CLI が
    origin(原点 lat,lon)・landmarks(ランドマーク座標表)・landmark_name_kws(名称マッチ語)・
    hachiko_fallback(名所フォールバック|None=無効)・map_name/description を差し替える。"""
    origin_ll = tuple(origin) if origin is not None else ORIGIN
    # 局所束縛で street 全体の project(...) 呼を街の原点版へ切替(既定=ORIGIN で従来同値)。
    project = _make_project(origin_ll)
    landmarks = LANDMARKS if landmarks is None else list(landmarks)
    kws = (LANDMARK_NAME_KWS if landmark_name_kws is _DEFAULT
           else tuple(landmark_name_kws or ()))
    hachiko = (HACHIKO_FALLBACK if hachiko_fallback is _DEFAULT else hachiko_fallback)

    nodes_ll: dict[int, tuple[float, float]] = {}
    node_tags: dict[int, dict] = {}
    ways_road: list[dict] = []
    ways_building: list[dict] = []
    ways_rail: list[dict] = []
    ways_poi: list[dict] = []
    for el in raw["elements"]:
        if el["type"] == "node":
            nodes_ll[el["id"]] = (el["lat"], el["lon"])
            if el.get("tags"):
                node_tags[el["id"]] = el["tags"]
        elif el["type"] == "way":
            tags = el.get("tags", {})
            if tags.get("highway") in HIGHWAY_CLASSES:
                ways_road.append(el)
            if "building" in tags:
                ways_building.append(el)
            elif tags.get("railway") in ("rail", "subway"):
                ways_rail.append(el)
            if poi_category(tags, kws):
                ways_poi.append(el)

    # --- 道路: 交差点でウェイを分割してエッジ化(layer 付き)---
    use_count: dict[int, int] = {}
    for way in ways_road:
        ids = [i for i in way["nodes"] if i in nodes_ll]
        for i in ids:
            use_count[i] = use_count.get(i, 0) + 1
        for i in (ids[0], ids[-1]):
            use_count[i] = use_count.get(i, 0) + 1  # 端点は交差点扱い

    edges: list[dict] = []
    for way in ways_road:
        tags = way["tags"]
        klass = tags["highway"]
        layer = way_layer(tags)
        ids = [i for i in way["nodes"] if i in nodes_ll]
        if len(ids) < 2:
            continue
        seg = [ids[0]]
        for node_id in ids[1:]:
            seg.append(node_id)
            if use_count.get(node_id, 0) >= 2 or node_id == ids[-1]:
                pts = [project(*nodes_ll[i]) for i in seg]
                length = polyline_length(pts)
                if length >= 1.0 and seg[0] != seg[-1]:
                    e = {"u": f"n{seg[0]}", "v": f"n{seg[-1]}",
                         "klass": klass, "geometry": pts,
                         "length": round(length, 1)}
                    if layer != 0:
                        e["layer"] = layer
                    edges.append(e)
                seg = [node_id]

    # --- 次数2ノードの縮約(地図を軽くする。ランドマーク付与前に実施)---
    from collections import defaultdict
    adj: dict[str, list[int]] = defaultdict(list)
    for idx, e in enumerate(edges):
        adj[e["u"]].append(idx)
        adj[e["v"]].append(idx)

    def other(e: dict, n: str) -> str:
        return e["v"] if e["u"] == n else e["u"]

    removed = set()
    changed = True
    while changed:
        changed = False
        for node, idxs in list(adj.items()):
            live = [i for i in idxs if i not in removed]
            if len(live) != 2:
                continue
            e1, e2 = edges[live[0]], edges[live[1]]
            a, b = other(e1, node), other(e2, node)
            if a == b or a == node or b == node:
                continue
            if e1["klass"] != e2["klass"]:
                continue
            if e1.get("layer", 0) != e2.get("layer", 0):
                continue
            g1 = e1["geometry"] if e1["v"] == node else list(reversed(e1["geometry"]))
            g2 = e2["geometry"] if e2["u"] == node else list(reversed(e2["geometry"]))
            merged = {"u": a, "v": b, "klass": e1["klass"],
                      "geometry": g1 + g2[1:],
                      "length": round(e1["length"] + e2["length"], 1)}
            if e1.get("layer", 0) != 0:
                merged["layer"] = e1["layer"]
            removed.update(live)
            edges.append(merged)
            new_idx = len(edges) - 1
            adj[a] = [i for i in adj[a] if i not in removed] + [new_idx]
            adj[b] = [i for i in adj[b] if i not in removed] + [new_idx]
            adj[node] = []
            changed = True

    edges = [e for i, e in enumerate(edges) if i not in removed]

    # --- 最大連結成分のみ残す(ちかみち・デッキは階段経由で地上と繋がる)---
    import networkx as nx
    g = nx.Graph()
    for i, e in enumerate(edges):
        g.add_edge(e["u"], e["v"], idx=i)
    if g.number_of_nodes() == 0:
        raise RuntimeError("道路が取得できていない")
    largest = max(nx.connected_components(g), key=len)
    edges = [e for e in edges if e["u"] in largest and e["v"] in largest]

    node_xy: dict[str, tuple[float, float]] = {}
    node_layer: dict[str, int] = {}
    for e in edges:
        node_xy[e["u"]] = e["geometry"][0]
        node_xy[e["v"]] = e["geometry"][-1]
        lyr = e.get("layer", 0)
        for n in (e["u"], e["v"]):
            # 複数レイヤーのエッジが接続するノードは接続点(地上扱い優先)
            node_layer[n] = 0 if node_layer.get(n, lyr) != lyr else lyr

    # --- ゲートウェイ(範囲の縁の行き止まり=外界への出口)---
    sw = project(bbox[0], bbox[1])
    ne = project(bbox[2], bbox[3])
    deg: dict[str, int] = defaultdict(int)
    for e in edges:
        deg[e["u"]] += 1
        deg[e["v"]] += 1
    gateways = []
    for node, d in deg.items():
        x, y = node_xy[node]
        near_border = (abs(x - sw[0]) < 40 or abs(x - ne[0]) < 40
                       or abs(y - sw[1]) < 40 or abs(y - ne[1]) < 40)
        if d == 1 and near_border:
            gateways.append(node)

    # 車用ゲートウェイ(背景交通の発生/消滅点): 縁に近い DRIVABLE の端点
    car_gateways = []
    for node, d in deg.items():
        x, y = node_xy[node]
        near_border = (abs(x - sw[0]) < 60 or abs(x - ne[0]) < 60
                       or abs(y - sw[1]) < 60 or abs(y - ne[1]) < 60)
        if not near_border:
            continue
        if any((e["u"] == node or e["v"] == node) and e["klass"] in DRIVABLE
               and e.get("layer", 0) == 0 for e in edges):
            car_gateways.append(node)

    # --- ランドマーク名を最寄りノードへ ---
    names: dict[str, tuple[str, str]] = {}
    for name, lat, lon, poi in landmarks:
        lx, ly = project(lat, lon)
        best, best_d = None, 1e9
        for node, (x, y) in node_xy.items():
            d = math.hypot(x - lx, y - ly)
            if d < best_d:
                best, best_d = node, d
        if best is not None and best_d < 150:
            names[best] = (name, poi)

    def nearest_node(x: float, y: float, ground_only: bool = True) -> str:
        best, best_d = None, 1e9
        for node, (nx_, ny_) in node_xy.items():
            if ground_only and node_layer.get(node, 0) < 0:
                continue
            d = math.hypot(nx_ - x, ny_ - y)
            if d < best_d:
                best, best_d = node, d
        return best

    nodes_out = []
    for node, (x, y) in sorted(node_xy.items()):
        entry = {"id": node, "x": x, "y": y}
        if node_layer.get(node, 0) != 0:
            entry["layer"] = node_layer[node]
        if node in names:
            entry["name"], entry["poi"] = names[node]
        if node in gateways:
            entry["gateway"] = True
        nodes_out.append(entry)

    # --- 建物: 全件(住宅含む)+用途分類。家の割当・屋内活動に使う ---
    buildings = []
    n_main_override = 0
    n_with_entrances = 0
    for way in ways_building:
        tags = way.get("tags", {})
        ids = [i for i in way["nodes"] if i in nodes_ll]
        if len(ids) < 4:
            continue
        pts = [project(*nodes_ll[i]) for i in ids]
        area = abs(sum(pts[i][0] * pts[(i + 1) % len(pts)][1]
                       - pts[(i + 1) % len(pts)][0] * pts[i][1]
                       for i in range(len(pts)))) / 2
        if area < 25:
            continue
        name = tags.get("name:ja") or tags.get("name")
        levels = tags.get("building:levels")
        try:
            levels = max(1, int(float(levels))) if levels else (
                6 if area > 1500 else (4 if area > 400 else 2))
        except ValueError:
            levels = 3
        below = tags.get("building:levels:underground")
        try:
            below = max(0, int(float(below))) if below else 0
        except ValueError:
            below = 0
        kind = building_kind(tags, area, name)
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)

        # --- 入口(実データ): 建物外周ノードの entrance=* を収集 ---
        entrances: list[dict] = []
        for nid in way["nodes"]:
            etag = node_tags.get(nid, {}).get("entrance")
            if etag in ENTRANCE_KINDS and nid in nodes_ll:
                ex, ey = project(*nodes_ll[nid])
                entrances.append({"x": ex, "y": ey, "kind": etag})
        # main 入口があれば、そこに最も近い道路ノードを建物の入口に採用
        main = next((e for e in entrances if e["kind"] == "main"), None)
        if main is not None:
            entrance = nearest_node(main["x"], main["y"])
            n_main_override += 1
        else:
            entrance = nearest_node(cx, cy)

        b = {"id": f"b{way['id']}", "name": name or "",
             "levels": levels, "kind": kind, "footprint": pts,
             "entrance": entrance, "area": round(area),
             "cx": round(cx, 1), "cy": round(cy, 1)}
        if below:
            b["below"] = below
        if entrances:
            b["entrances"] = entrances
            n_with_entrances += 1
        buildings.append(b)
    buildings.sort(key=lambda b: -b["area"])

    # --- POI(実名の店・会社・施設)を建物と最寄りノードへ紐付け ---
    big_buildings = [b for b in buildings if b["area"] >= 100]
    pois = []
    seen_poi_names = set()

    def add_poi(tags: dict, x: float, y: float, src_id: str) -> None:
        cat = poi_category(tags, kws)
        name = tags.get("name:ja") or tags.get("name")
        if not cat or not name:
            return
        # v8: 重複排除キーに **道路ノード** を足す((name, cat) → (name, cat, node))。
        #   ★理由(実測): 旧キーはチェーン店を全店 1 件へ潰していた。渋谷 bbox の
        #     shop=convenience は「名前付き 93 要素 → 9 件」= コンビニの 9 割が地図から
        #     消えていた(同名ブランドが別の場所に何店もあるのが実態)。夜間経済
        #     (24h 営業)の受け皿がこれでは成立しない。
        #   ★同一ノードの同名同カテゴリだけを潰す = 「同じ店がノードとウェイの両方で
        #     マップされている」二重取りは従来どおり 1 件に畳まれる(元の目的は保つ)。
        #   ★純増(additive): 残る POI の id は要素 id 由来なので **既存 POI の id は不変**。
        node = nearest_node(x, y)
        if (name, cat, node) in seen_poi_names:
            return
        seen_poi_names.add((name, cat, node))
        host = None
        for b in big_buildings:
            if (abs(x - b["cx"]) < 120 and abs(y - b["cy"]) < 120
                    and point_in_poly(x, y, b["footprint"])):
                host = b
                break
        floor = 0
        lvl = tags.get("level")
        if lvl:
            try:
                floor = int(float(str(lvl).split(";")[0]))
            except ValueError:
                floor = 0
        p = {"id": f"p_{src_id}", "name": name, "cat": cat,
             "x": round(x, 1), "y": round(y, 1),
             "node": node}
        sub = poi_subcategory(tags)
        if sub:                                  # v8: 生タグ由来の第2階層(無ければキーごと無い)
            p["subcat"] = sub
        if host:
            p["building"] = host["id"]
            p["floor"] = floor
        pois.append(p)

    for nid, tags in node_tags.items():
        if nid in nodes_ll:
            x, y = project(*nodes_ll[nid])
            add_poi(tags, x, y, f"n{nid}")
    for way in ways_poi:
        ids = [i for i in way["nodes"] if i in nodes_ll]
        if len(ids) < 3:
            continue
        pts = [project(*nodes_ll[i]) for i in ids]
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        add_poi(way.get("tags", {}), cx, cy, f"w{way['id']}")

    # --- 建物 kind に office/school を反映(POI 由来。用途不明の generic のみ格上げ)---
    #     住宅系(residential/house?)は家の割当に使うので触らない=既存の家割当は不変。
    pois_by_bld: dict[str, set[str]] = defaultdict(set)
    for p in pois:
        if p.get("building"):
            pois_by_bld[p["building"]].add(p["cat"])
    for b in buildings:
        if b["kind"] != "generic":
            continue
        cats = pois_by_bld.get(b["id"], set())
        if "school" in cats:
            b["kind"] = "public"
        elif "office" in cats:
            b["kind"] = "office"

    # --- 名所フォールバック(OSM で landmark を拾えなかった場合に明示座標で置く)---
    #     既定=渋谷のハチ公像。別の街は hachiko_fallback=None で無効(名所が無ければ置かない)。
    has_hachiko = bool(hachiko) and any(
        p["cat"] == "landmark" and any(kw in p["name"] for kw in ("忠犬", "ハチ公"))
        for p in pois)
    hachiko_source = "osm" if has_hachiko else "none"
    if hachiko and not has_hachiko:
        hx, hy = project(hachiko[1], hachiko[2])
        hnode = nearest_node(hx, hy)
        if hnode is not None:
            pois.append({"id": "p_hachiko_fallback", "name": hachiko[0],
                         "cat": "landmark", "x": round(hx, 1), "y": round(hy, 1),
                         "node": hnode})
            hachiko_source = "fallback"

    # --- 線路(可視化レイヤー用: 地上=rail / 地下鉄=subway)---
    railways = []
    for way in ways_rail:
        tags = way.get("tags", {})
        ids = [i for i in way["nodes"] if i in nodes_ll]
        if len(ids) < 2:
            continue
        pts = [project(*nodes_ll[i]) for i in ids]
        if polyline_length(pts) < 30:
            continue
        railways.append({"name": tags.get("name:ja") or tags.get("name") or "線路",
                         "kind": tags["railway"], "geometry": pts})

    _default_desc = ("OSM 実データ(Overpass)による渋谷駅中心部。"
                     "全建物+実入口+POI+地下/デッキ層。"
                     "座標=スクランブル交差点原点ローカル平面(m)。")
    meta = {
        "version": 6, "name": map_name or "shibuya_osm",
        "description": description or _default_desc,
        "attribution": "© OpenStreetMap contributors (ODbL)",
        # 出典・ライセンスの明示(data/ は公開ミラー除外だが、由来は地図自身に埋める)。
        "license": "ODbL 1.0",
        "license_url": "https://opendatacommons.org/licenses/odbl/1-0/",
        "source": "OpenStreetMap via Overpass API",
        "origin_latlon": list(origin_ll), "bbox": list(bbox),
        "crs": "local-m",
        # v8: POI の第2階層。語彙は閉じている(消費側 src/society/night.py が知る 5 語 +6)。
        "subcat_vocab": sorted(SUBCAT_TOPCAT),
    }
    if osm_date:
        meta["osm_date"] = osm_date
    if fetched_at:
        meta["fetched_at"] = str(fetched_at)     # Overpass から取得した UTC 時刻(ISO8601)
    sub_counts: dict[str, int] = {}
    for p in pois:
        if p.get("subcat"):
            sub_counts[p["subcat"]] = sub_counts.get(p["subcat"], 0) + 1
    meta["_stats"] = {"main_entrance_override": n_main_override,
                      "buildings_with_entrances": n_with_entrances,
                      "hachiko_source": hachiko_source,
                      "subcats": dict(sorted(sub_counts.items()))}
    return {
        "meta": meta,
        "nodes": nodes_out,
        "edges": edges,
        "buildings": buildings,
        "pois": pois,
        "railways": railways,
        "car_gateways": sorted(car_gateways),
    }


def _report(data: dict, out: Path) -> None:
    from collections import Counter
    n_res = sum(1 for b in data["buildings"]
                if b["kind"] in ("residential", "house?"))
    n_under = sum(1 for e in data["edges"] if e.get("layer", 0) < 0)
    n_deck = sum(1 for e in data["edges"] if e.get("layer", 0) > 0)
    n_under_nodes = sum(1 for n in data["nodes"] if n.get("layer", 0) < 0)
    n_ent = sum(1 for b in data["buildings"] if b.get("entrances"))
    stats = data["meta"].get("_stats", {})
    cats = Counter(p["cat"] for p in data["pois"])
    print(f"written: {out}")
    print(f"  nodes={len(data['nodes'])} edges={len(data['edges'])} "
          f"(地下 edge={n_under}/node={n_under_nodes} デッキ={n_deck}) "
          f"buildings={len(data['buildings'])} (住宅系={n_res}) "
          f"pois={len(data['pois'])} "
          f"gateways={sum(1 for n in data['nodes'] if n.get('gateway'))} "
          f"car_gateways={len(data['car_gateways'])}")
    print(f"  入口実データ: entrances付き建物={n_ent} "
          f"(main で入口置換={stats.get('main_entrance_override', 0)})")
    print(f"  POI 拡大(v6): office={cats.get('office', 0)} school={cats.get('school', 0)} "
          f"cinema={cats.get('cinema', 0)} hall={cats.get('hall', 0)} "
          f"landmark={cats.get('landmark', 0)}  ハチ公={stats.get('hachiko_source', '?')}")
    subs = stats.get("subcats", {})
    detail = " ".join(f"{k}={v}" for k, v in sorted(subs.items())) or "(該当タグ無し)"
    print(f"  POI subcat(v8): {sum(subs.values())} 件  {detail}")


def bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    """bbox(S,W,N,E)の中心(lat, lon)。原点=bbox中心 モード用。"""
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def find_poi_latlon(raw: dict, name: str) -> tuple[float, float] | None:
    """OSM 生データから POI 名(name / name:ja 完全一致)の緯度経度を返す。

    ノードは自座標、ウェイは構成ノードの重心。原点=ランドマークPOI名 モード用。
    見つからなければ None。ネットワーク不使用の純関数(テスト可能)。"""
    if not name:
        return None
    nodes_ll: dict[int, tuple[float, float]] = {}
    for el in raw.get("elements", []):
        if el.get("type") == "node":
            nodes_ll[el["id"]] = (el["lat"], el["lon"])
    for el in raw.get("elements", []):
        tags = el.get("tags") or {}
        nm = tags.get("name:ja") or tags.get("name")
        if nm != name:
            continue
        if el.get("type") == "node" and el["id"] in nodes_ll:
            return nodes_ll[el["id"]]
        if el.get("type") == "way":
            pts = [nodes_ll[i] for i in el.get("nodes", []) if i in nodes_ll]
            if pts:
                return (sum(p[0] for p in pts) / len(pts),
                        sum(p[1] for p in pts) / len(pts))
    return None


def resolve_origin(bbox, raw=None, *, latlon=None, poi=None,
                   bbox_center_mode=False) -> tuple[tuple[float, float], str]:
    """原点を3択で決める: (1)指定座標 latlon / (2)ランドマークPOI名 poi / (3)bbox中心。

    どれも未指定なら既定=ORIGIN(渋谷スクランブル交差点)= 現行手順は不変。
    戻り値: ((lat, lon), モード名)。"""
    if latlon is not None:
        return (float(latlon[0]), float(latlon[1])), "latlon"
    if poi:
        found = find_poi_latlon(raw or {}, poi)
        if found is None:
            raise RuntimeError(f"原点POI '{poi}' が取得データに見つからない(名称一致せず)")
        return found, f"poi:{poi}"
    if bbox_center_mode:
        return bbox_center(bbox), "bbox-center"
    return ORIGIN, "default(shibuya)"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="OSM 地図ビルダー(bbox/原点/日付/ランドマーク 指定可・既定=渋谷)")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("S", "W", "N", "E"),
                    default=list(DEFAULT_BBOX),
                    help="south west north east(既定=現行渋谷範囲 約1.0×0.7km)")
    ap.add_argument("--osm-date", default=None,
                    help='過去スナップショット日 "YYYY-MM-DD"(Overpass attic query。既定=最新)')
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "shibuya_osm.json"),
                    help="出力先 JSON パス(既定=data/shibuya_osm.json)")
    # ---- 原点の決め方(3択。未指定=既定 ORIGIN=スクランブル交差点=現行不変)----
    og = ap.add_mutually_exclusive_group()
    og.add_argument("--origin-latlon", nargs=2, type=float, metavar=("LAT", "LON"),
                    default=None, help="原点を指定座標に(緯度 経度)")
    og.add_argument("--origin-poi", default=None,
                    help="原点を取得データ内のランドマークPOI名(完全一致)に")
    og.add_argument("--origin-bbox-center", action="store_true",
                    help="原点を bbox の中心に")
    ap.add_argument("--name", default=None, help="地図メタ name(既定=shibuya_osm)")
    ap.add_argument("--description", default=None,
                    help="地図メタ description(既定=渋谷の定型文)")
    ap.add_argument("--landmarks-file", default=None,
                    help='ランドマーク表 JSON([[name,lat,lon,cat],...])。既定=渋谷14件')
    ap.add_argument("--no-landmark-kws", action="store_true",
                    help="名称マッチのランドマーク寄せ(ハチ公/モヤイ等)を無効化")
    ap.add_argument("--no-hachiko-fallback", action="store_true",
                    help="名所フォールバック(渋谷ハチ公像)を無効化")
    # ---- 生データのキャッシュ(v8: 同じ取得を何度も Overpass へ投げない)----
    ap.add_argument("--raw-out", default=None,
                    help="Overpass 生データの保存先 JSON(取得時刻を _fetched_at に埋める)")
    ap.add_argument("--raw-in", default=None,
                    help="保存済み生データから組む(ネットワーク不使用。--raw-out の出力を渡す)")
    args = ap.parse_args()

    bbox = tuple(args.bbox)
    if args.raw_in:
        raw = json.loads(Path(args.raw_in).read_text(encoding="utf-8"))
        print(f"生データを再利用: {args.raw_in}(取得 {raw.get('_fetched_at', '?')})",
              file=sys.stderr)
    else:
        print(f"Overpass API から取得中... bbox={bbox} date={args.osm_date or '最新'}",
              file=sys.stderr)
        raw = fetch_overpass(bbox, args.osm_date)
        from datetime import datetime, timezone
        raw["_fetched_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if args.raw_out:
            Path(args.raw_out).write_text(json.dumps(raw, ensure_ascii=False),
                                          encoding="utf-8")
            print(f"  生データ保存: {args.raw_out}", file=sys.stderr)
    print(f"  elements: {len(raw['elements'])}", file=sys.stderr)

    origin, mode = resolve_origin(
        bbox, raw, latlon=args.origin_latlon, poi=args.origin_poi,
        bbox_center_mode=args.origin_bbox_center)
    print(f"  原点: {origin} (モード={mode})", file=sys.stderr)

    landmarks = None
    if args.landmarks_file:
        landmarks = json.loads(Path(args.landmarks_file).read_text(encoding="utf-8"))

    data = build(raw, bbox, args.osm_date, origin=origin, landmarks=landmarks,
                 landmark_name_kws=(() if args.no_landmark_kws else _DEFAULT),
                 hachiko_fallback=(None if args.no_hachiko_fallback else _DEFAULT),
                 map_name=args.name, description=args.description,
                 fetched_at=raw.get("_fetched_at"))
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    _report(data, out)


if __name__ == "__main__":
    main()
