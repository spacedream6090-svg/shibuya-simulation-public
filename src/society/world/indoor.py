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


def bulk_index(zone_types: list):
    """最頻型(=bulk)の zone index(同数は型名昇順=決定論)。空なら None。"""
    if not zone_types:
        return None
    counts = Counter(str(t) for t in zone_types)
    best_type = min(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    for i, t in enumerate(zone_types):
        if str(t) == best_type:
            return i
    return 0


def pick_zone(zone_types: list, preferred, rng=None, exclude=None):
    """活動の優先型 preferred に一致する zone index を決定論選択する(B3 の区画割当・回遊の共通核)。

    規則: preferred(型名の優先リスト)を先頭から見て、その型を持つ zone が在れば、それらから rng で
    1つ選ぶ(rng=None なら最小 index=決定論)。どの優先型も無ければ bulk(最頻型)の zone へ落とす。
    exclude(=現 zone)を渡すと候補から外す(markov 回遊=同/別型の別区画へ動く用途)。exclude を除く
    候補が皆無なら None(=この step は滞在=移動しない)。zone_types が空なら None。型名は conf/data 由来
    (このコードは型語を1つも埋め込まない=no-fingerprint)。"""
    n = len(zone_types or [])
    if n == 0:
        return None
    pref = [str(t) for t in (preferred or [])]
    for t in pref:
        cands = [i for i in range(n) if str(zone_types[i]) == t and i != exclude]
        if cands:
            return cands[0] if rng is None else int(cands[int(rng.integers(0, len(cands)))])
    bi = bulk_index(zone_types)
    if bi is not None and bi != exclude:
        return bi
    if exclude is None:
        return bi                                    # n>=1 は必ず bulk を持つ
    others = [i for i in range(n) if i != exclude]   # exclude==bulk: 別区画へ・無ければ滞在
    return others[0] if others else None


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


def build_engine_cfg(raw) -> dict:
    """conf.indoor を B3 エンジン配線用の素の dict へ正準化(既定 OFF=現行挙動と完全同一)。

    markov(区画割当/回遊)・sfm(積分パラメータ)・meeting(会議)・encounter(コスト制御)・tracks
    (記録サイドカー)を既定つきで平坦化する。型語(desk 等)・活動語(working 等)はここでは一切
    埋め込まず conf から読むだけ=no-fingerprint。sfm は IndoorParams.from_cfg がそのまま読む生 dict。"""
    r = _as_map(raw)
    markov = _as_map(r.get("markov"))
    sfm = _as_map(r.get("sfm"))
    meeting = _as_map(r.get("meeting"))
    enc = _as_map(r.get("encounter"))
    tracks = _as_map(r.get("tracks"))
    los = _as_map(r.get("los"))

    def _amap(m) -> dict:
        return {str(k): [str(x) for x in _as_list(v)] for k, v in _as_map(m).items()}

    win = [int(x) for x in _as_list(meeting.get("window_min"))]
    if len(win) < 2:
        win = [600, 900]
    mtypes = [str(x) for x in _as_list(meeting.get("meeting_types"))] or ["meeting"]
    return {
        "enabled": bool(r.get("enabled", False)),
        "markov": {
            "enabled": bool(markov.get("enabled", False)),
            "dwell_steps": max(1, int(markov.get("dwell_steps", 3) or 3)),
            "assign_types": _amap(markov.get("assign_types")),
            "roam_types": _amap(markov.get("roam_types")),
        },
        "sfm": {**{str(k): sfm[k] for k in sfm},
                "enabled": bool(sfm.get("enabled", False))},
        "meeting": {
            "enabled": bool(meeting.get("enabled", False)),
            "min_party": max(2, int(meeting.get("min_party", 2) or 2)),
            "prob": float(meeting.get("prob", 0.5) or 0.0),
            "window_min": [win[0], win[1]],
            "meeting_types": mtypes,
        },
        "encounter": {
            "bystander_cap": max(0, int(enc.get("bystander_cap", 24) or 0)),
            # B3b: 直近(前 step)の屋内遭遇ペアを対面会話の返答相手選択で優先する動力学接続。
            "pairing": bool(enc.get("pairing", False)),
        },
        "tracks": {"enabled": bool(tracks.get("enabled", False))},
        # B3b: 屋内知覚の壁 LOS ゲート(擬似視覚=同席文脈=発火プロンプトの近傍リストを区画粒度に絞る)。
        "los": {
            "enabled": bool(los.get("enabled", False)),
            "max_dist_m": max(0.0, float(los.get("max_dist_m", 0.0) or 0.0)),
        },
    }
