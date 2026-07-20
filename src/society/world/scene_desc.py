"""構造化シーン記述(v0)= 決定論の「見え」ビルダ(読み出し専用の純関数)。

発火時のプロンプトに 2〜4 行の「視覚に相当する客観記述」を組む土台。追加 LLM 呼ゼロ・
乱数消費ゼロ・発火順不変(純関数)。既定 OFF(cfg.enabled=false)では 1 行も返さない
=バイト一致(crowd_line / digest_line と同型の seam)。

組み立てる 3 系統(いずれも決定論・世界データからの集計のみ):
  ① 方向つき視界記述: 進行方向(heading)から見て正面/右手/左手に何があるか。
     heading は route の次ノードへの向きから導出(状態を増やさない純導出)。
  ② 注視対象リスト: 近傍の人・地点を距離で決定論選抜。視線遮蔽器(occluder)を渡すと
     壁で隔てられた相手を除外する(perception/vision の LOS を流用)。
  ③ 垂直関係: レイヤ(高い通路/地下)+ 標高。標高は elevation が非 None のときだけ利用
     (None ガード必須。無ければ標高由来の一句は出さない)。

配置は world 直下だが、地名リテラル・構成概念名は一切書かない(no-fingerprint 契約)。
記述に載る地名・施設名はすべて実行時に世界データ(POI 名・建物名・place_label)から取る。
件数(注視対象 N・視界の記述数)は入力解像度 LOD(agent.input_res.scene_n)の対象。
"""
from __future__ import annotations

import math

SCENE_DEFAULTS = {
    "enabled": False,
    "radius_m": 60.0,      # 視界・注視に含める近傍地点までの距離(crowd/ads と同レンジ)
    "scene_n": 3,          # 既定の記述件数(agent.input_res.scene_n 未設定時のフォールバック)
    "fov_deg": 100.0,      # 正面と判定する視野角(全幅)。半幅の cos を前方しきいに使う
    "vertical": True,      # 垂直関係の 1 行を出すか(レイヤ/標高)
}

# 方向ラベル(中立語。地名でも構成概念でもない)。
_DIR_LABEL = {"front": "正面に", "right": "右手に", "left": "左手に"}

# 距離帯のことば(注視対象の遠近感。決定論のしきい)。
_BANDS = ((15.0, "すぐ近く"), (30.0, "近く"), (60.0, "少し先"))


def build_cfg(raw) -> dict:
    """conf(world.scene_desc)ブロックを正準化(既定 OFF=現行挙動と完全同一)。

    dotlist 上書きは文字列で入り得るため型強制する(street.build_crowd_cfg と同流儀)。"""
    raw = dict(raw or {})
    cfg = dict(SCENE_DEFAULTS)
    for k in raw:
        if k in SCENE_DEFAULTS:
            cfg[k] = raw[k]
    cfg["enabled"] = bool(cfg["enabled"])
    cfg["vertical"] = bool(cfg["vertical"])
    cfg["radius_m"] = float(cfg["radius_m"])
    cfg["fov_deg"] = float(cfg["fov_deg"])
    cfg["scene_n"] = int(cfg["scene_n"])
    return cfg


def enabled(cfg) -> bool:
    return bool(cfg and cfg.get("enabled"))


def _scene_n(agent, cfg) -> int:
    """記述件数 = 入力解像度 LOD(agent.input_res.scene_n)> cfg 既定。最低 1。"""
    ir = getattr(agent, "input_res", None) or {}
    n = ir.get("scene_n", (cfg or SCENE_DEFAULTS).get("scene_n", 3))
    try:
        return max(1, int(n))
    except (TypeError, ValueError):
        return 3


# ------------------------------------------------------------------ heading 導出
def heading_of(agent, city):
    """進行方向を単位ベクトル (ux, uy) で返す(次に向かうノードへの向き)。

    route[0]=次ノード(agents.Agent の route 定義)への向き。route が無ければ dest への向き。
    どちらも無い/距離ゼロなら None(静止 → 方向づけない記述にフォールバック)。純導出=状態を
    増やさない・乱数を引かない。"""
    target = None
    route = getattr(agent, "route", None)
    if route:
        target = route[0]
    elif getattr(agent, "dest", None):
        target = agent.dest
    if not target:
        return None
    try:
        tx, ty = city.node_xy(target)
    except Exception:
        return None
    dx, dy = float(tx) - float(agent.x), float(ty) - float(agent.y)
    d = math.hypot(dx, dy)
    if d < 1e-6:
        return None
    return (dx / d, dy / d)


def _bucket(agent, heading, x: float, y: float, front_cos: float) -> str | None:
    """対象 (x,y) が heading 基準で 正面/右手/左手 のどれか。真後ろは None(視界外)。"""
    dx, dy = float(x) - float(agent.x), float(y) - float(agent.y)
    d = math.hypot(dx, dy)
    if d < 1e-6:
        return "front"
    ux, uy = dx / d, dy / d
    hx, hy = heading
    dot = hx * ux + hy * uy
    if dot < -0.35:                      # ほぼ真後ろ=視界に入らない
        return None
    if dot >= front_cos:
        return "front"
    cross = hx * uy - hy * ux            # heading→対象 の回転向き(<0=時計回り=右手)
    return "right" if cross < 0 else "left"


# ------------------------------------------------------------------ 近傍地点の収集
def _band(d: float) -> str:
    for hi, label in _BANDS:
        if d <= hi:
            return label
    return "遠く"


def _gather_places(agent, city, radius: float) -> list[tuple]:
    """現在ノード+隣接ノードの名前つき POI・建物を (距離, 名前, x, y) で集める(決定論)。

    グラフ近傍のみ走査=次数で有界(全 POI/建物の O(P+B) 走査を避ける)。名前の無い建物は
    除外(記述に載せる意味がない)。名前重複は先勝ちで畳む。距離→名前昇順=決定論。"""
    seen: set[str] = set()
    raw: list[tuple[str, float, float]] = []

    def _add(name, x, y):
        nm = str(name or "").strip()
        if nm and nm not in seen:
            seen.add(nm)
            raw.append((nm, float(x), float(y)))

    node = getattr(agent, "node", "") or ""
    nodes = [node]
    graph = getattr(city, "graph", None)
    if graph is not None and node in graph:
        nodes += sorted(graph.neighbors(node))
    for nb in nodes:
        try:
            nx_, ny_ = city.node_xy(nb)
        except Exception:
            continue
        for p in city.pois_at_node(nb):
            _add(p.get("name"), nx_, ny_)
        for b in city.buildings_at(nb):
            if b.get("name"):
                cx, cy = b.get("centroid", (nx_, ny_))
                _add(b["name"], cx, cy)
    # 屋内なら同一建物・同一階の POI も近傍に含める(足元の店名)。
    if getattr(agent, "building", None) and city.has_building(agent.building):
        for p in city.pois_in_building(agent.building, agent.floor):
            _add(p.get("name"), agent.x, agent.y)

    out = []
    for nm, x, y in raw:
        d = math.hypot(x - agent.x, y - agent.y)
        if d <= radius:
            out.append((d, nm, x, y))
    out.sort(key=lambda t: (round(t[0], 3), t[1]))
    return out


# ------------------------------------------------------------------ 各行の構築
def _directional(agent, city, heading, cfg, n: int) -> list[tuple[str, str, float]]:
    """[(方向ラベルキー, 名前, 距離)] を距離順に最大 n 件。heading=None は方向キー None。"""
    radius = float((cfg or SCENE_DEFAULTS)["radius_m"])
    places = _gather_places(agent, city, radius)
    if heading is None:
        return [(None, nm, d) for (d, nm, _x, _y) in places[:n]]
    front_cos = math.cos(math.radians(float((cfg or SCENE_DEFAULTS)["fov_deg"]) / 2.0))
    out = []
    for (d, nm, x, y) in places:
        b = _bucket(agent, heading, x, y, front_cos)
        if b is None:
            continue
        out.append((b, nm, d))
        if len(out) >= n:
            break
    return out


def _view_line(directional: list[tuple]) -> str | None:
    """方向つき視界の 1 行(見えるもの)。空なら None。"""
    if not directional:
        return None
    parts = []
    for (dir_key, nm, _d) in directional:
        if dir_key is None:
            parts.append(nm)
        else:
            parts.append(_DIR_LABEL[dir_key] + nm)
    return "見えるもの: " + "、".join(parts) + "。"


def _gaze_line(agent, company, directional, cfg, n: int, occluder) -> str | None:
    """注視対象リスト(人+地点)を距離で決定論選抜。壁 LOS で隔てられた人は除外。

    company(同席者=既に上流で LOS 済みのことが多い)に occluder を渡せば二重適用でも同結果。
    地点は _directional が集めた近傍名を距離つきで再利用(追加走査なし)。"""
    cands: list[tuple[float, str, str]] = []   # (距離, 名前, 種別)
    for p in (company or []):
        if occluder is not None:
            try:
                if occluder.blocks(agent, p):
                    continue
            except Exception:
                pass
        d = math.hypot(float(p.x) - float(agent.x), float(p.y) - float(agent.y))
        cands.append((d, getattr(p, "name", ""), "person"))
    for (_dir, nm, d) in directional:
        cands.append((d, nm, "place"))
    cands.sort(key=lambda t: (round(t[0], 3), t[2], t[1]))
    picked = []
    seen: set[str] = set()
    for (d, nm, _kind) in cands:
        if not nm or nm in seen:
            continue
        seen.add(nm)
        picked.append(f"{nm}({_band(d)})")
        if len(picked) >= n:
            break
    if not picked:
        return None
    return "目に留まるもの: " + "・".join(picked) + "。"


def _vertical_line(agent, city, cfg, elevation) -> str | None:
    """垂直関係(レイヤ+標高)の 1 行。何も特筆なければ None(トークン節約)。

    標高は elevation が非 None のときだけ参照(None ガード)。レイヤは世界データ(地図)由来。"""
    if not (cfg or SCENE_DEFAULTS).get("vertical", True):
        return None
    parts: list[str] = []
    try:
        layer = int(city.node_layer(agent.node))
    except Exception:
        layer = 0
    if layer > 0:
        parts.append("高い通路の上から地上を見下ろしている")
    elif layer < 0:
        parts.append("地下にいる")
    if elevation is not None and layer == 0 and not getattr(agent, "building", None):
        try:
            z0 = float(elevation.height_at(agent.x, agent.y))
            graph = getattr(city, "graph", None)
            hs = []
            if graph is not None and agent.node in graph:
                for nb in sorted(graph.neighbors(agent.node)):
                    nx_, ny_ = city.node_xy(nb)
                    hs.append(float(elevation.height_at(nx_, ny_)))
            if hs:
                mean = sum(hs) / len(hs)
                if z0 - mean >= 2.0:
                    parts.append("まわりより高い場所にいる")
                elif mean - z0 >= 2.0:
                    parts.append("まわりより低い場所にいる")
        except Exception:
            pass
    if not parts:
        return None
    return "位置関係: " + "、".join(parts) + "。"


# ------------------------------------------------------------------ 公開 API
def scene_lines(agent, *, city, company=None, cfg=None, elevation=None,
                occluder=None) -> list[str]:
    """発火プロンプトへ注入する 2〜4 行(視界/注視/垂直)。OFF・素材なしは []=注入なし。

    決定論・追加 LLM 呼ゼロ・乱数消費ゼロの純関数。件数は input_res LOD(scene_n)対象。"""
    if not enabled(cfg) or city is None:
        return []
    n = _scene_n(agent, cfg)
    heading = heading_of(agent, city)
    directional = _directional(agent, city, heading, cfg, n)
    lines = []
    vl = _view_line(directional)
    if vl:
        lines.append(vl)
    gl = _gaze_line(agent, company, directional, cfg, n, occluder)
    if gl:
        lines.append(gl)
    zl = _vertical_line(agent, city, cfg, elevation)
    if zl:
        lines.append(zl)
    return lines


def digest_line(agent, *, city, cfg=None, elevation=None) -> str | None:
    """行間ダイジェスト(S2)へ足す「客観の見え」1 行。意味づけしない客観記述のみ。

    現在地の近傍で目に入るものを列挙するだけ(なぜ印象に残ったか等の解釈は夜内省の LLM の
    仕事=二段分離)。OFF・素材なしは None=digest に 1 文字も足さない。"""
    if not enabled(cfg) or city is None:
        return None
    n = _scene_n(agent, cfg)
    heading = heading_of(agent, city)
    directional = _directional(agent, city, heading, cfg, n)
    if not directional:
        return None
    names = "・".join(nm for (_dir, nm, _d) in directional)
    return "見えたもの: " + names


def conversation_meta(agent, *, city, cfg=None) -> str | None:
    """会話イベント payload の scene メタ(場所×注視対象の中立要約)。OFF は None。

    「今この場所で何が見えているか」を会話の付帯情報として残す(実文は作らない)。"""
    if not enabled(cfg) or city is None:
        return None
    n = _scene_n(agent, cfg)
    heading = heading_of(agent, city)
    directional = _directional(agent, city, heading, cfg, n)
    place = ""
    try:
        place = city.place_label(agent.node)
    except Exception:
        place = ""
    names = "・".join(nm for (_dir, nm, _d) in directional[:2])
    if place and names:
        return f"{place}|{names}"
    return place or names or None
