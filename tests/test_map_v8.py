"""地図 v8(第101バッチ Wave4 III-2)のテスト = 「v7 の上に POI だけを足した」ことの機械固定。

v8 が何であるか(1 行):
    v7 と **同一の OSM スナップショット(Overpass attic 2025-04-01・同一 bbox)** を取り直し、
    build_map の POI 分類に **subcat(第2階層 11 語)** を足し、**重複排除キーを是正**
    ((name, cat) → (name, cat, node) = チェーン店の全店復元)して組み直した地図。

このファイルが固定する 5 つの契約:
  (A) **道路グラフは v7 とバイト一致**(nodes / edges / railways / car_gateways が JSON として
      完全同値)。建物も id・形・入口・階数まで同値で、差分は **POI 由来の用途格上げ 2 棟の
      kind だけ**(generic→office/public。build_map の既存ルールが良くなったデータで発火)。
      = ゾーン多角形・物理ベンチ座標・PLATEAU 突合(建物 7,210)・組織台帳の node/建物参照が
      1 件も壊れない。地図を替えたのに「街の形」は動かない。
  (B) **POI は上位集合**。v7 の POI は cat を変えずに全て残り(差分は subcat キーの追加のみ)、
      v7 が取りこぼしていた業態と、旧キーが潰していたチェーン店の各店だけが増える
      (id は要素 id 由来なので**既存 id は 1 つも動かない**)。唯一の例外は patch POI の
      採番ずれで、理由(金王八幡宮が OSM 側から入るようになった)ごとここで固定する。
  (C) **top cat 13 語は不変**(day_plan.PLACE_CATS / commerce の営業時間表 / vision の salience
      が cat に依存する)。件数の増減にもガードを張る。
  (D) **subcat の語彙は閉じている**。消費側(src/society/night.py の DEFAULT_HOURS /
      refuge.subcats)が名前で知っている語の上位集合であり、subcat→cat の対応も 1 対 1。
  (E) **v8 は opt-in**。既定の conf は v7 のまま(ゴールデン L1 バイト一致を守る)。

★正直な注記: net_cafe は OSM 側の実要素が **name タグを持たない 1 件だけ**で POI にならない
  (add_poi は名前の無い要素を落とす)。そこで data/poi_patch_shibuya.json に**一般名**
  「ネットカフェ」で 1 件だけ手当てしてある(座標は当該要素の実座標・ブランド名は書かない)。
  = v8 の net_cafe は 1 件で、これは OSM 由来ではなく手動パッチ由来。

ネットワークは使わない(生成済みファイル + 合成タグのみ)。実LLM は使わない(mock のみ)。
"""
from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

from society import commerce, night
from society.config import load_config
from society.engine.simulation import Simulation
from society.world.map import CityMap

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
V7 = DATA / "shibuya_osm_wide_v7.json"
V8 = DATA / "shibuya_osm_wide_v8.json"


def _load_build_map():
    spec = importlib.util.spec_from_file_location(
        "build_map_v8", REPO / "scripts" / "build_map.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BM = _load_build_map()


def _doc(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def v7() -> dict:
    return _doc(V7)


@pytest.fixture(scope="module")
def v8() -> dict:
    return _doc(V8)


# --------------------------------------------------------------------------- #
# (A) 街の形は動かない
# --------------------------------------------------------------------------- #
def test_v8_graph_is_byte_identical_to_v7(v7, v8):
    """道路グラフ・線路・車ゲートウェイが v7 と JSON として完全同値。

    v8 は v7 と同じ attic 日(2025-04-01)を取り直して組んでいるので、**再取得しても
    街の形は 1 バイトも動かない**。ここが落ちたら「最新版で取り直した」等の取り違えが
    起きている(最新版では実測で nodes +5.1% / edges +4.9% / 座標が動くノード 132 件)。"""
    for key in ("nodes", "edges", "railways", "car_gateways"):
        assert json.dumps(v7[key], ensure_ascii=False, sort_keys=True) == \
            json.dumps(v8[key], ensure_ascii=False, sort_keys=True), \
            f"v8 の {key} が v7 と違う(グラフが動いている)"


# 建物の唯一の差分: POI 由来の用途格上げ(build_map の「generic のみ格上げ」ルール)。
# 重複排除キー是正で復活した POI が中に入ったことで generic → office/public になった 2 棟。
# 形(footprint/area/entrance/levels)は 1 バイトも動かない。
BUILDING_KIND_UPGRADES = {
    "b137206183": ("generic", "office"),    # office POI(同名の別拠点)が中に復活した
    "b141016066": ("generic", "public"),    # school POI(同名の別校舎)が中に復活した
}


def test_v8_buildings_differ_only_by_documented_kind_upgrades(v7, v8):
    """建物は id・形・入口・階数まで v7 と同値。差分は既知 2 棟の kind だけ。"""
    a = {b["id"]: b for b in v7["buildings"]}
    b8 = {b["id"]: b for b in v8["buildings"]}
    assert set(a) == set(b8), "建物 id の集合が動いた"
    diffs = {}
    for bid, old in a.items():
        new = b8[bid]
        if old == new:
            continue
        assert {k for k in old if old[k] != new.get(k)} == {"kind"}, \
            f"{bid} が kind 以外で違う(形が動いた)"
        diffs[bid] = (old["kind"], new["kind"])
    assert diffs == BUILDING_KIND_UPGRADES, f"建物 kind の差分が想定と違う: {diffs}"
    for old_kind, new_kind in diffs.values():
        assert old_kind == "generic" and new_kind in ("office", "public"), \
            "用途不明(generic)以外の建物が書き換わっている"


def test_v8_loads_as_citymap_and_keeps_landmarks(v8):
    """CityMap にロードでき、原点・駅ノード・地下・入口が v7 と同じ形で載る。"""
    c8, c7 = CityMap(V8), CityMap(V7)
    assert c8.graph.number_of_nodes() == c7.graph.number_of_nodes()
    assert c8.graph.number_of_edges() == c7.graph.number_of_edges()
    assert c8.station_node == c7.station_node
    assert c8.gateways == c7.gateways
    assert c8.underground_edges() and c8.underground_nodes()
    d0 = min(abs(n["x"]) + abs(n["y"]) for n in v8["nodes"])
    assert d0 < 20, "原点(スクランブル交差点)がずれている"


# --------------------------------------------------------------------------- #
# (B) POI は上位集合(cat の付け替えが起きない)
# --------------------------------------------------------------------------- #
def test_v8_pois_are_superset_of_v7_and_only_add_subcat(v7, v8):
    """v7 の POI は「subcat キーが増えるだけ」で v8 に残る。cat の付け替えはゼロ。

    唯一の例外 = 手動パッチ POI の採番ずれ(p_patch_NN)。v8 では 金王八幡宮 が OSM 側
    (amenity=place_of_worship)から入るため patch_map が同名スキップし、以降の連番が
    1 つ前へ詰まる。**この 1 件だけ**であることを固定する。"""
    a = {p["id"]: p for p in v7["pois"]}
    b = {p["id"]: p for p in v8["pois"]}
    changed_other = []
    subcat_added = Counter()
    for pid, old in a.items():
        if pid.startswith("p_patch_"):
            continue
        assert pid in b, f"v7 の POI {pid} が v8 で消えた"
        new = dict(b[pid])
        sub = new.pop("subcat", None)
        if new != old:
            changed_other.append((pid, old, b[pid]))
        if sub:
            subcat_added[sub] += 1
    assert not changed_other, f"subcat 追加以外の差分がある: {changed_other[:3]}"
    assert sum(subcat_added.values()) > 0, "v7 由来の POI に subcat が 1 件も付いていない"
    # 追加された POI は v7 に無かったものだけ
    assert len(b) > len(a), "v8 で POI が増えていない"


def test_v8_patch_poi_renumber_is_the_documented_single_exception(v7, v8):
    """パッチ POI の採番ずれの理由を機械で示す: 金王八幡宮 が OSM 由来で先に居る。"""
    names7 = [p["name"] for p in v7["pois"] if p["id"].startswith("p_patch_")]
    names8 = [p["name"] for p in v8["pois"] if p["id"].startswith("p_patch_")]
    assert set(names7) - set(names8) == {"金王八幡宮"}
    # v8 で 1 件だけ増えるパッチ = ネットカフェ(OSM 側が name タグを持たず POI にならない
    # 実要素を、ブランド名を書かずに一般名で採録。座標は当該要素の実座標)。
    assert set(names8) - set(names7) == {"ネットカフェ"}
    osm_side = [p for p in v8["pois"]
                if p["name"] == "金王八幡宮" and not p["id"].startswith("p_patch_")]
    assert osm_side, "金王八幡宮 が OSM 由来でも居ない(=パッチを落としただけになっている)"
    assert osm_side[0]["subcat"] == "worship" and osm_side[0]["cat"] == "landmark"


def test_org_book_poi_refs_resolve_the_same_on_v7_and_v8():
    """本選の安全弁: 組織台帳(v7 の POI id 参照)を v8 で読んでも解決結果が変わらない。

    台帳が名指しする POI id は p_patch_09(実践女子学園・10 校)だけで、v8 では採番ずれで
    存在しない。work._resolve_building は **POI が building を持つときだけ** id を使い、
    それ以外は node 経由へ後退する。パッチ POI は building を持たないので、v7 でも
    id は使われていなかった = 結果は同値。node/建物 id はグラフ同値なので当然一致する。"""
    from society import work
    book = json.loads((DATA / "organizations_shibuya_wide11k.json")
                      .read_text(encoding="utf-8"))
    recs = [r for r in book["companies"] + book["schools"] if r.get("workplace_poi")]
    c7, c8 = CityMap(V7), CityMap(V8)
    diffs = []
    for r in recs[:400] + book["schools"]:          # 先頭 400 社 + 全校(p_patch 参照を含む)
        wp = r["workplace_poi"]
        got7 = work._resolve_building(c7, wp["node"], wp.get("building"), wp.get("poi_id"))
        got8 = work._resolve_building(c8, wp["node"], wp.get("building"), wp.get("poi_id"))
        if got7 != got8:
            diffs.append((r["id"], got7, got8))
    assert not diffs, f"v7→v8 で職場の建物解決が変わる組織がある: {diffs[:5]}"


# --------------------------------------------------------------------------- #
# (C) top cat 13 語は不変 + 件数ガード
# --------------------------------------------------------------------------- #
# **増えることが設計上わかっている** cat。増える理由は 2 つだけ:
#   (1) subcat の新規取り込み(v7 が None にしていた業態を救済。build_map の SUBCAT_TOPCAT)
#   (2) 重複排除キーの是正((name,cat) → (name,cat,node))= チェーン店の全店復元。
#       (2) は cat を選ばないので、チェーン展開の多い food / shop / service が伸びる。
# ここに無い cat は ±20% に収まっていなければならない(= 分類ロジックの侵食検知)。
GROWS_BY_DESIGN = {
    "hotel": "(1) love_hotel(円山町のラブホ 63 件。OSM-JA は amenity=love_hotel)",
    "landmark": "(1) worship(神社仏閣・教会 29 件)",
    "service": "(1) hospital 7 件 +(2) チェーン薬局等の複数店舗",
    "leisure": "(1) arcade(ゲームセンター 4 件)",
    "nightlife": "(1) sauna 4 +(2) チェーンのカラオケ各店(karaoke 10→17)",
    "shop": "(1) pachinko(amenity=gambling 8)+(2) コンビニ各店(convenience 9→93)",
}


def test_v8_top_cats_are_unchanged_and_magnitudes_guarded(v7, v8):
    """13 個の top cat は増えも減りもしない。件数は「減らない」+「設計外は ±20%」。"""
    ca = Counter(p["cat"] for p in v7["pois"])
    cb = Counter(p["cat"] for p in v8["pois"])
    assert set(cb) == set(ca), f"top cat の語彙が動いた: {set(cb) ^ set(ca)}"
    assert len(set(ca)) == 13, f"v7 の cat が 13 語でない: {sorted(ca)}"
    for cat, n7 in ca.items():
        n8 = cb[cat]
        assert n8 >= n7 * 0.8, f"{cat} が 2 割以上減った({n7}→{n8})"
        if cat not in GROWS_BY_DESIGN:
            assert n8 <= n7 * 1.2, \
                f"{cat} が想定外に増えた({n7}→{n8})= 分類ロジックが v7 を侵食している"
    for cat in GROWS_BY_DESIGN:
        assert cb[cat] > ca[cat], f"{cat} が増えていない(取り込みが効いていない)"


def test_v8_cats_are_the_vocabulary_day_plan_knows(v8):
    """v8 の cat は day_plan.PLACE_CATS(+ ルール側の home/work/street)に閉じている。"""
    from society.cognition.day_plan import PLACE_CATS
    cats = {p["cat"] for p in v8["pois"]}
    # education は「学校」の旧称で PLACE_CATS 側の語。school は地図側の語(persona.py が両対応)。
    unknown = cats - set(PLACE_CATS) - {"school"}
    assert not unknown, f"day_plan が知らない cat が地図に居る: {unknown}"


# --------------------------------------------------------------------------- #
# (D) subcat の語彙は閉じている / 消費側と一致
# --------------------------------------------------------------------------- #
# 実測値(2025-04-01 スナップショット)。件数そのものではなく「実在するか」を固定する。
EXPECTED_SUBCATS = {
    "convenience": 93,   # shop=convenience 名前付き 93 要素が**全店**残る(旧キーでは 9 に潰れていた)
    "karaoke": 17,       # amenity=karaoke_box 16 + leisure=karaoke 2(同一ノード重複 1 を畳む)
    "club": 17,          # amenity=nightclub 19(無名 2)
    "sauna": 4,          # leisure=sauna 3 + amenity=public_bath 1(改良湯・渋谷SAUNAS 等)
    "pachinko": 8,       # amenity=gambling 8(shop=pachinko は 0 件だった)
    "arcade": 4,         # leisure=amusement_arcade 3 + adult_gaming_centre 1
    "worship": 29,       # amenity=place_of_worship 33(無名 3 + 同一ノード重複 1)
    "hospital": 7,       # amenity=hospital 7(JA-OSM は美容クリニックもこのタグに載せる)
    "love_hotel": 63,    # amenity=love_hotel 63(円山町)
    "childcare": 12,     # amenity=kindergarten 12(amenity=childcare は 0 件)
    "net_cafe": 1,       # ★OSM 側は name タグ無しで POI にならない → 手動パッチ 1 件
}
# 語彙 11 語すべてが実データで当たる(空の語は無い)。
KNOWN_EMPTY_SUBCATS: set = set()


def test_v8_subcat_vocabulary_is_closed(v8):
    """地図に現れる subcat は build_map の閉じた語彙(= meta.subcat_vocab)だけ。"""
    seen = {p["subcat"] for p in v8["pois"] if p.get("subcat")}
    vocab = set(BM.SUBCAT_TOPCAT)
    assert seen <= vocab, f"語彙外の subcat: {seen - vocab}"
    assert set(v8["meta"]["subcat_vocab"]) == vocab, "meta の語彙表がコードとずれている"
    assert seen == vocab - KNOWN_EMPTY_SUBCATS, \
        f"実在する subcat の集合が想定と違う(増: {seen - vocab} / 減: {vocab - KNOWN_EMPTY_SUBCATS - seen})"


def test_v8_subcat_to_cat_mapping_is_consistent(v8):
    """POI の (subcat → cat) は SUBCAT_TOPCAT と 1 件残らず一致する。"""
    bad = [(p["id"], p["subcat"], p["cat"]) for p in v8["pois"]
           if p.get("subcat") and p["cat"] != BM.SUBCAT_TOPCAT[p["subcat"]]]
    assert not bad, f"subcat と cat の対応が壊れている: {bad[:5]}"


def test_v8_subcat_counts_match_measurement(v8):
    """主要 subcat が実在する(件数は実測値を記録。減ったら地図の作り直しを疑う)。"""
    got = Counter(p["subcat"] for p in v8["pois"] if p.get("subcat"))
    for sub, n in EXPECTED_SUBCATS.items():
        assert got[sub] == n, f"subcat={sub} の件数が {n} から {got[sub]} へ動いた"
    for sub in KNOWN_EMPTY_SUBCATS:
        assert got[sub] == 0, f"{sub} が実在するようになった(穴が埋まった=注記を更新すること)"
    # meta._stats は **パッチ前**(build_map)の集計なので、手動パッチ分を除いて突き合わせる。
    osm_side = Counter(p["subcat"] for p in v8["pois"]
                       if p.get("subcat") and not p["id"].startswith("p_patch_"))
    assert osm_side == Counter(v8["meta"]["_stats"]["subcats"]), "meta の集計と実体がずれている"
    assert got["net_cafe"] - osm_side["net_cafe"] == 1, "net_cafe の手動パッチが載っていない"


def test_v8_chain_stores_are_no_longer_collapsed(v8):
    """重複排除キー是正の実証: 同名コンビニが**別ノードに複数**載っている。

    旧キー (name, cat) は「同じブランドの全店」を 1 件へ潰していた(コンビニ 93→9)。
    新キー (name, cat, node) では同名でも別ノードなら残り、同一ノードの二重取りだけ畳む。"""
    conv = [p for p in v8["pois"] if p.get("subcat") == "convenience"]
    by_name = Counter(p["name"] for p in conv)
    assert max(by_name.values()) > 1, "同名コンビニが 1 件も複数店になっていない"
    # 同一 (name, cat, node) の重複はゼロ(畳む側の契約)
    keys = [(p["name"], p["cat"], p["node"]) for p in v8["pois"]]
    assert len(keys) == len(set(keys)), "同一ノードに同名同カテゴリの POI が二重に居る"


def test_v8_vocabulary_covers_the_night_layer_consumer():
    """消費側(src/society/night.py)が名前で知っている subcat は全て地図側の語彙にある。

    night.py は既定 OFF の層だが、その表(DEFAULT_HOURS / refuge.subcats)に書かれた語が
    地図側の語彙に無ければ **永久に当たらない死語**になる。それを機械で防ぐ。"""
    ncfg = night.build_cfg({"enabled": True})
    ccfg = commerce.build_cfg(None)
    vocab = set(BM.SUBCAT_TOPCAT)
    consumer = (set(night.DEFAULT_HOURS) | set(night.DEFAULT_REFUGE_SUBCATS)) - set(ccfg["hours"])
    assert consumer, "夜間層が subcat 語を 1 つも持っていない(前提が崩れた)"
    assert consumer <= vocab, f"地図が作れない subcat を夜間層が期待している: {consumer - vocab}"
    assert set(ncfg["refuge"]["subcats"]) <= vocab
    # 地図側が「夜間層向け」と宣言している語(build_map.NIGHT_SUBCATS)と実体が一致する
    assert set(BM.NIGHT_SUBCATS) == consumer, \
        f"地図側の宣言と夜間層の実体がずれている: {set(BM.NIGHT_SUBCATS) ^ consumer}"


def test_night_layer_reads_v8_subcat_end_to_end(v8):
    """v8 の実 POI で「コンビニが 03:00 に開く」= 表を書き換えずに効いたことの実証。"""
    ccfg = commerce.build_cfg(None)
    on = night.build_cfg({"enabled": True})
    conv = [p for p in v8["pois"] if p.get("subcat") == "convenience"]
    assert conv, "v8 にコンビニ POI が無い"
    for p in conv:
        assert night.poi_subcat(on, p) == "convenience"
        assert commerce.is_open_poi(ccfg, p, 3 * 60, on), f"{p['name']} が 03:00 に閉じている"
        assert not commerce.is_open_poi(ccfg, p, 3 * 60), "OFF でも 24h になっている"
    # 避難先(refuge)の候補が subcat 経路で増える(v7 は cat=nightlife だけ)
    saunas = [p for p in v8["pois"] if p.get("subcat") == "sauna"]
    assert saunas and all(night._poi_is_refuge(on, p) for p in saunas)


# --------------------------------------------------------------------------- #
# (E) 生成側の性質(v7 判定を侵食しない / 名前を見ない)
# --------------------------------------------------------------------------- #
V7_TAG_CASES = [
    ({"amenity": "restaurant"}, "food"), ({"amenity": "cafe"}, "food"),
    ({"amenity": "fast_food"}, "food"), ({"amenity": "bar"}, "nightlife"),
    ({"amenity": "pub"}, "nightlife"), ({"amenity": "nightclub"}, "nightlife"),
    ({"amenity": "karaoke_box"}, "nightlife"), ({"amenity": "cinema"}, "cinema"),
    ({"amenity": "theatre"}, "hall"), ({"leisure": "stadium"}, "hall"),
    ({"amenity": "bank"}, "service"), ({"amenity": "clinic"}, "service"),
    ({"amenity": "school"}, "school"), ({"amenity": "kindergarten"}, "school"),
    ({"shop": "clothes"}, "shop"), ({"shop": "convenience"}, "shop"),
    ({"office": "company"}, "office"), ({"tourism": "hotel"}, "hotel"),
    ({"tourism": "museum"}, "attraction"), ({"tourism": "artwork"}, "landmark"),
    ({"historic": "monument"}, "landmark"), ({"leisure": "park"}, "leisure"),
    ({"man_made": "statue"}, "landmark"), ({"name": "ハチ公前", "shop": "clothes"}, "landmark"),
]


def test_poi_category_never_changes_a_v7_decision():
    """v7 が cat を返したケースは v8 でも **同じ cat**(救済は None のときだけ)。"""
    for tags, expect in V7_TAG_CASES:
        assert BM._poi_category_v7(tags) == expect, f"v7 判定の前提が崩れた: {tags}"
        assert BM.poi_category(tags) == expect, f"v8 が v7 の判定を上書きした: {tags}"


def test_poi_category_rescues_only_what_v7_dropped():
    """v7 が None にしていた業態だけが、subcat 経由で top cat を得る。"""
    rescued = [({"amenity": "internet_cafe"}, "net_cafe", "nightlife"),
               ({"leisure": "sauna"}, "sauna", "nightlife"),
               ({"amenity": "public_bath"}, "sauna", "nightlife"),
               ({"amenity": "gambling"}, "pachinko", "shop"),
               ({"leisure": "amusement_arcade"}, "arcade", "leisure"),
               ({"leisure": "adult_gaming_centre"}, "arcade", "leisure"),
               ({"amenity": "place_of_worship"}, "worship", "landmark"),
               ({"amenity": "hospital"}, "hospital", "service"),
               ({"amenity": "love_hotel"}, "love_hotel", "hotel"),
               ({"tourism": "love_hotel"}, "love_hotel", "hotel"),
               ({"amenity": "childcare"}, "childcare", "school")]
    for tags, sub, cat in rescued:
        assert BM._poi_category_v7(tags) is None, f"v7 は既に拾えていた: {tags}"
        assert BM.poi_subcategory(tags) == sub
        assert BM.poi_category(tags) == cat
    # 語彙外のタグは何も生まない(誤爆しない)
    for tags in ({"amenity": "vending_machine"}, {"leisure": "dance"}, {}):
        assert BM.poi_subcategory(tags) is None


def test_poi_subcategory_never_reads_the_name():
    """ETHICS §2: subcat は生タグの純関数。POI 名(ブランド名)を 1 文字も見ない。"""
    for name in ["セブン-イレブン", "Apple Store", "ABCマート", "ネットカフェ",
                 "サウナ", "パチンコ", "○○神社", "△△病院"]:
        assert BM.poi_subcategory({"name": name}) is None
        assert BM.poi_subcategory({"name:ja": name}) is None
    # 名前を差し替えても結果は不変
    base = {"shop": "convenience"}
    assert BM.poi_subcategory(base) == BM.poi_subcategory({**base, "name": "全く別の名"})


def test_subcat_topcat_targets_are_existing_cats():
    """subcat の行き先は **既存の top cat** だけ(新カテゴリを作らない)。"""
    v7_cats = {p["cat"] for p in _doc(V7)["pois"]}
    assert set(BM.SUBCAT_TOPCAT.values()) <= v7_cats


def test_meta_carries_attribution_and_provenance(v8):
    """出典・ライセンス・取得日時・bbox が地図自身に埋まっている(ODbL 表示義務)。"""
    m = v8["meta"]
    assert "OpenStreetMap" in m["attribution"] and "ODbL" in m["attribution"]
    assert m["license"].startswith("ODbL") and m["license_url"].startswith("https://")
    assert m["source"] == "OpenStreetMap via Overpass API"
    assert m["osm_date"] == "2025-04-01", "v7 と同じ attic 日でなければ形が動く"
    assert m["fetched_at"].endswith("Z") and m["fetched_at"][:2] == "20"
    assert m["bbox"] == _doc(V7)["meta"]["bbox"]
    assert m["origin_latlon"] == list(BM.ORIGIN)
    assert "v8" in m["description"] and "subcat" in m["description"]


# --------------------------------------------------------------------------- #
# (F) v8 は opt-in(既定の conf は v7 のまま)+ 実際に走る
# --------------------------------------------------------------------------- #
def test_v8_is_opt_in_and_default_conf_still_points_at_v7():
    """既定の設定は v7 のまま = ゴールデン L1 バイト一致を壊さない。"""
    import re
    for conf in ("production.yaml", "daily.yaml", "longrun30.yaml"):
        text = (REPO / "conf" / conf).read_text(encoding="utf-8")
        maps = re.findall(r"^\s*map:\s*(\S+)", text, re.M)
        assert maps, f"{conf} に world.map が無い"
        for m in maps:
            assert not m.endswith("v8.json"), f"{conf} が既定で v8 を指している"
    base = (REPO / "conf" / "config.yaml").read_text(encoding="utf-8")
    assert re.search(r"^\s*map:\s*data/shibuya_osm\.json", base, re.M), \
        "config.yaml の既定地図が動いている"


def test_v8_mock_smoke_runs(tmp_path):
    """v8 を読んだ mock ラン(24 step)が緑で走り、夜間層の subcat 経路も生きる。"""
    dot = ["run.n_agents=12", "run.n_steps=24", "run.name=v8_smoke",
           "observer.snapshot_every=8", "model.backend=mock",
           f"world.map={V8.as_posix()}", "world.night_economy.enabled=true"]
    sim = Simulation(load_config(dot), out_dir=tmp_path / "v8_smoke")
    assert sim.city.poi_list and any(p.get("subcat") for p in sim.city.poi_list)
    sim.run()
    assert sim.logger.events, "イベントが 1 件も出ていない"
    # 夜間層が v8 の subcat を実際に引けている(避難先候補に subcat 由来が混ざる)
    refuges = night.refuge_pois(sim)
    assert any(p.get("subcat") in ("sauna", "karaoke") for p in refuges), \
        "避難先候補が cat=nightlife だけ(subcat 経路が効いていない)"
    # 手当てした net_cafe が終電後の受け皿として実際に候補へ入る(駅から徒歩圏)
    nets = [p for p in refuges if p.get("subcat") == "net_cafe"]
    assert nets, "net_cafe が避難先候補に居ない"
    sx, sy = sim.city.node_xy(sim.city.station_node)
    import math
    assert any(math.hypot(*(a - b for a, b in zip(sim.city.node_xy(p["node"]), (sx, sy))))
               <= sim.nightcfg["refuge"]["max_dist_m"] for p in nets), \
        "net_cafe が駅から徒歩圏(max_dist_m)の外に居る"
