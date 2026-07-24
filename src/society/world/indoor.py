"""空間レイヤ核: 全建物・全階の屋内ミクロ状態(間取り+壁+ドア+区画用途型)を
決定論・遅延キャッシュで供給する「単一の真実」。エンジン配線(scheduler)は次バッチ担当=本層では行わない。

- 間取り正典 = world.vision.building_layout(FNV+mulberry32 決定論・ビューア JS の floorLayout と
  同一系列=シム⇄ビューアが同じ間取りを再導出できる)。壁は vision._walls_from_layout、ドアは
  vision.doors_from_layout を、同一 layout から導いて三者を整合させる。
- SpaceType(区画用途型)= conf 側の静的マップ(use カテゴリ → 型優先順位表)で駆動する。
  型語彙・写像規則はすべて data/conf 側にあり、このコードには業種名・地名を一切書かない
  (floor_layouts.json / floorguide_shibuya.json / POI cat / 建物 kind を汎用に読むだけ=
  no-fingerprint 契約。tests/test_contracts.py 準拠)。
- 決定論: 乱数状態を持たない。区画→型の割当は「面積降順+区画 index 順」の純関数。
- floor_layouts.json(並行タスクが作成中)不在でも動く(spec 無し=間取り正典のフォールバック)。
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from . import vision


# ---- floor_layouts.json ローダ / 建物名マッチ(floorguide と同一規約)----
def load_floor_specs(path: str | Path) -> list[dict]:
    """floor_layouts.json を読み、buildings レコード列を返す。

    スキーマ: {"buildings": [{"match": [表記ゆれ...], "floors": [{"f", "use", "shops",
              "zone_mix": {useカテゴリ or 型: 区画数}, "anchors": ["ev","escalator",...]}]}]}。
    match は floorguide_shibuya.json と同一規約(部分一致・双方向)。ファイル不在なら []
    (このデータは並行タスクが作成中=本層は無くても間取り正典で動く)。"""
    p = Path(path)
    if not p.exists():
        return []
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    blds = raw.get("buildings")
    return list(blds) if isinstance(blds, list) else []


def _match(rec: dict, name: str) -> bool:
    """floorguide_shibuya.json の _fg_matches と同一規約(部分一致・双方向)。"""
    return any(m and (m in name or name in m) for m in rec.get("match", []))


def spec_floor(specs: list[dict], name: str, floor: int) -> dict | None:
    """建物名 name(表記ゆれ許容)・階 floor の floor spec を引く。無ければ None。"""
    if not name:
        return None
    fl = int(floor)
    for rec in specs:
        if _match(rec, name):
            for f in rec.get("floors", []):
                try:
                    if int(f.get("f", 0)) == fl:
                        return f
                except (TypeError, ValueError):
                    continue
            return None                          # 建物は一致・その階の spec 無し
    return None


def _zone_area(z) -> float:
    return abs((z[2] - z[0]) * (z[3] - z[1]))


def _as_list(v) -> list:
    """OmegaConf ListConfig / list / None を素の list に正規化。"""
    if v is None:
        return []
    return list(v)


def _as_map(v) -> dict:
    """OmegaConf DictConfig / dict / None を素の dict に正規化。"""
    if v is None:
        return {}
    try:
        return dict(v)
    except Exception:
        return {}


class IndoorSpace:
    """(building_id, floor) → {layout, walls, doors, zone_types, anchors} の決定論プロバイダ。

    全建物・全階を扱える純決定論(乱数状態を持たない)。結果は (building_id, floor) キーで
    遅延キャッシュする。space_types は「use カテゴリ → 型優先順位表 [bulk, accent1, accent2...]」
    の静的マップ(conf 由来)。city は building()/has_building()/pois_in_building()/floor_guide()
    を(あれば)使うが、どれも欠けていても縮退動作する(テストの最小 city / bare fixture 対応)。
    """

    def __init__(self, city, floor_specs: list[dict] | None = None,
                 space_types=None):
        self.city = city
        self.specs = list(floor_specs) if floor_specs else []
        # space_types: {use: [型...]}。値・キーとも素の str/list へ正規化(OmegaConf 対応)。
        self.space_types = {str(k): _as_list(v)
                            for k, v in _as_map(space_types).items()}
        self._cache: dict[tuple[str, int], dict] = {}

    # ---- 公開 ----
    def get(self, building_id: str, floor: int) -> dict:
        key = (building_id, int(floor))
        cell = self._cache.get(key)
        if cell is None:
            cell = self._build(building_id, int(floor))
            self._cache[key] = cell
        return cell

    def layout(self, building_id: str, floor: int) -> dict | None:
        return self.get(building_id, floor)["layout"]

    def zone_types(self, building_id: str, floor: int) -> list:
        return self.get(building_id, floor)["zone_types"]

    def doors(self, building_id: str, floor: int) -> list:
        return self.get(building_id, floor)["doors"]

    # ---- 構築 ----
    def _building(self, building_id: str) -> dict | None:
        city = self.city
        if city is None:
            return None
        has = getattr(city, "has_building", None)
        try:
            if has is not None and not has(building_id):
                return None
            return city.building(building_id)
        except Exception:
            return None

    _EMPTY = {"layout": None, "walls": [], "doors": [], "zone_types": [],
              "anchors": []}

    def _build(self, building_id: str, floor: int) -> dict:
        building = self._building(building_id)
        if building is None:
            return dict(self._EMPTY)
        name = building.get("name") or building.get("id") or ""
        spec = spec_floor(self.specs, name, floor)

        # 区画数 override: shops(明示区画数)> zone_mix の合計 > None(=間取り正典の既定)
        n_override = None
        if spec is not None:
            if spec.get("shops"):
                n_override = int(spec["shops"])
            elif spec.get("zone_mix"):
                n_override = sum(int(v) for v in _as_map(spec["zone_mix"]).values())

        layout = vision.building_layout(building, floor, self.city,
                                        n_override=n_override)
        if layout is None:
            return dict(self._EMPTY)
        # 壁・ドアは「今 computeした layout」から導く(building_walls は n_override を知らず
        # 別 n の間取りを再生成しうる=desync するため、共通の layout を使って三者を整合させる。
        # n_override=None のときは building_walls と完全に同一結果)。
        walls = vision._walls_from_layout(layout)
        doors = vision.doors_from_layout(layout)
        zone_types = self._zone_types(building, floor, layout, spec)
        anchors = _as_list((spec or {}).get("anchors"))
        return {"layout": layout, "walls": walls, "doors": doors,
                "zone_types": zone_types, "anchors": anchors}

    # ---- SpaceType 割当 ----
    def _zone_types(self, building: dict, floor: int, layout: dict,
                    spec: dict | None) -> list:
        zones = layout.get("zones") or []
        if not zones:
            return []
        zone_mix = _as_map((spec or {}).get("zone_mix"))
        if zone_mix:                             # 明示 mix が最優先=そのまま敷く
            return self._types_from_mix(zones, zone_mix)
        use = self._resolve_use(building, floor, spec)
        return self._assign(zones, self._type_list(use))

    def _type_list(self, use: str) -> list:
        st = self.space_types
        if use in st and st[use]:
            return list(st[use])
        gen = st.get("generic")
        if gen:
            return list(gen)
        return ["zone"]                          # 汎用フォールバック(型マップ未供給時)

    def _bulk_type(self, key: str) -> str:
        """zone_mix のキーを型へ写像: space_types にあれば bulk(先頭)型、無ければキー自体を型名とみなす。"""
        lst = self.space_types.get(key)
        if lst:
            return str(lst[0])
        return str(key)

    def _types_from_mix(self, zones: list, zone_mix: dict) -> list:
        """zone_mix({キー: 区画数})を面積降順に敷く。キーは辞書順=決定論。端数は最終型で埋める。"""
        order = sorted(range(len(zones)),
                       key=lambda i: (-_zone_area(zones[i]), i))
        seq: list = []
        for key in sorted(zone_mix.keys()):
            try:
                cnt = int(zone_mix[key])
            except (TypeError, ValueError):
                cnt = 0
            seq += [self._bulk_type(key)] * max(0, cnt)
        out: list = [None] * len(zones)
        fill = seq[-1] if seq else self._bulk_type("generic")
        for pos, zi in enumerate(order):
            out[zi] = seq[pos] if pos < len(seq) else fill
        return out

    @staticmethod
    def _assign(zones: list, type_list: list) -> list:
        """型優先順位表 [bulk, accent1, accent2...] を区画へ決定論割当。

        規則: bulk(先頭)を全区画へ敷き、accent(以降)を「最小側」区画へ1つずつ(小さいほど後方の
        accent)。面積降順+index 順で並べ、末尾=最小の n_accent 区画に accent を配る。
        例) [desk,meeting,break] → 最大群=desk・二番目に小=meeting・最小=break。
            [sales,register] → sales+最小1つ=register。[seating,kitchen] → seating+最小=kitchen。"""
        n = len(zones)
        if n == 0:
            return []
        if not type_list:
            return ["zone"] * n
        bulk = str(type_list[0])
        accents = [str(t) for t in type_list[1:]]
        out = [bulk] * n
        n_accent = min(len(accents), max(0, n - 1))  # bulk を最低1区画は残す
        if n_accent == 0:
            return out
        order = sorted(range(n), key=lambda i: (-_zone_area(zones[i]), i))
        accent_zones = order[n - n_accent:]      # 最小側 n_accent 区画(降順末尾=小さい順に近い)
        for k, zi in enumerate(accent_zones):
            out[zi] = accents[k]                 # accents[0]=大きめ accent 区画, 末尾=最小区画
        return out

    # ---- use 解決: spec.use > floorguide use > POI cat > 建物 kind ----
    def _resolve_use(self, building: dict, floor: int, spec: dict | None) -> str:
        if spec and spec.get("use"):
            return str(spec["use"])
        name = building.get("name") or building.get("id") or ""
        fg = self._floorguide_use(name, floor)
        if fg:
            return str(fg)
        poi = self._poi_use(building.get("id"), floor)
        if poi:
            return str(poi)
        return vision._kind_use(building.get("kind", "generic"))

    def _floorguide_use(self, name: str, floor: int) -> str | None:
        """floorguide_shibuya.json(読むだけ)から当該階の use を引く。city.floor_guide 経由。"""
        fn = getattr(self.city, "floor_guide", None)
        if fn is None:
            return None
        try:
            rec = fn(name)
        except Exception:
            return None
        if not rec:
            return None
        for f in rec.get("floors", []):
            try:
                if int(f.get("f", 0)) == int(floor):
                    return f.get("use")
            except (TypeError, ValueError):
                continue
        return None

    def _poi_use(self, building_id, floor: int) -> str | None:
        """当該階 POI の cat 多数決(同数は cat 名昇順=決定論)。"""
        if not building_id:
            return None
        fn = getattr(self.city, "pois_in_building", None)
        if fn is None:
            return None
        try:
            pois = fn(building_id, floor)
        except Exception:
            return None
        cats = [p.get("cat") for p in (pois or []) if p.get("cat")]
        if not cats:
            return None
        counts = Counter(cats)
        best = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return best[0][0]


def _cfg_get(cfg, dotted: str, default=None):
    """OmegaConf / dict どちらでも dotted パスで安全に読む(vision._cfg_get と同流儀)。"""
    cur = cfg
    for part in dotted.split("."):
        if cur is None:
            return default
        if hasattr(cur, "get"):
            try:
                cur = cur.get(part)
                continue
            except Exception:
                return default
        cur = getattr(cur, part, None)
    return default if cur is None else cur


def indoor_from_cfg(city, cfg) -> IndoorSpace | None:
    """cfg.indoor を読み、enabled のときだけ IndoorSpace を返す(既定 None=空間レイヤ不使用)。

    floor_layouts_path のファイルが無くても IndoorSpace は成立する(spec 無し=間取り正典)。
    本バッチではエンジン(scheduler)からは未配線=呼ばれない seam(次バッチが据え付ける)。"""
    if not bool(_cfg_get(cfg, "indoor.enabled", False)):
        return None
    path = _cfg_get(cfg, "indoor.floor_layouts_path", "data/floor_layouts.json")
    specs = load_floor_specs(path) if path else []
    space_types = _cfg_get(cfg, "indoor.space_types", None)
    return IndoorSpace(city, floor_specs=specs, space_types=space_types)
