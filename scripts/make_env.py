#!/usr/bin/env python3
"""環境自動生成 v0(半自動)CLI(D2・第35バッチ 2026-07-18)。

「渋谷で較正した基盤モデルを別の街へ即日展開する」ための EnvPack 半自動生成。
docs/plans/environment-autogen.md の v0 定義に沿う: **stage1-2 を自動化 / stage3-4 は
手動テンプレ** で、渋谷の build_map/transit スクリプト(汎用化済み)を place 指定で駆動する。

使い方:
    python scripts/make_env.py --place shimokita \\
        --bbox 139.662,35.656,139.674,35.667 --out env/shimokita
        # --bbox は w,s,e,n(経度西,緯度南,経度東,緯度北)。カンマ or 空白区切り。

    python scripts/make_env.py --place shimokita --bbox ... --out env/shimokita --stage 1
        # 特定 stage のみ再実行(各 stage 独立再実行可・失敗時は前段成果物を保持)。

    python scripts/make_env.py ... --raw-file env/shimokita/_osm_raw.json --stage 1
        # 取得済み OSM 生データから再ビルド(fetch→build の2段流儀。ネット不要)。

ステージ(docs/plans/environment-autogen.md):
    stage1 geography   : Overpass 取得 → map.json(検証: ノード連結性・POI/建物数を env_report に)
    stage2 transit     : ODPT 照会(キー無し/路線未定義なら「徒歩の街」宣言で継続)
    stage3-4 templates : personas/orgs の縮小版生成手順を TODO.md に / institutions 雛形を env.yaml に
    stage7 verify(最小): 生成物の構造検証 + env_report.md(取得統計・縮退の明示)

鉄則:
  - 渋谷の既存データ(data/*.json)は再生成も上書きもしない(本番資産)。生成物は --out 配下のみ。
  - ODPT キーが環境変数に無ければ「未取得」と正直に縮退(キーの値は要求も出力もしない)。
  - 文化・行事の LLM 生成はしない(v2 の領分)。取れないデータは「無い」と宣言して縮退。

出典: 地図=OpenStreetMap contributors(ODbL・Overpass 経由)。EnvPack は実体でなく取得レシピを指す。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import networkx as nx

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# build_map.py を(パッケージ外なので)動的ロードして再利用する(汎用化済みビルダー)。
# --------------------------------------------------------------------------- #
def _load_build_map():
    spec = importlib.util.spec_from_file_location(
        "build_map", SCRIPTS / "build_map.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BM = _load_build_map()


# --------------------------------------------------------------------------- #
# 純関数(テスト対象・ネットワーク不使用)
# --------------------------------------------------------------------------- #
def parse_bbox(text: str) -> tuple[float, float, float, float]:
    """"w,s,e,n"(経度西,緯度南,経度東,緯度北)→ build_map の (S, W, N, E)。

    カンマ/空白区切りの4数を受け、経度2つ・緯度2つを min/max で正規化する
    (西東・南北の取り違えを許容=v0 の使い勝手)。数が4でない/退化した bbox はエラー。"""
    parts = [p for p in text.replace(",", " ").split() if p]
    if len(parts) != 4:
        raise ValueError(f"--bbox は4つの数(w,s,e,n)で指定してください: {text!r}")
    try:
        w, s, e, n = (float(parts[0]), float(parts[1]),
                      float(parts[2]), float(parts[3]))
    except ValueError as ex:
        raise ValueError(f"--bbox に数でない値: {text!r}") from ex
    west, east = min(w, e), max(w, e)          # 経度(取り違え許容)
    south, north = min(s, n), max(s, n)        # 緯度
    if west == east or south == north:
        raise ValueError(f"--bbox が退化している(幅0): {text!r}")
    return (south, west, north, east)          # build_map の bbox 順


def validate_map(data: dict) -> dict:
    """生成した地図 JSON の構造検証。連結性・要素数を独立に測る(env_report 用)。"""
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    buildings = data.get("buildings", [])
    pois = data.get("pois", [])
    g = nx.Graph()
    for e in edges:
        g.add_edge(e["u"], e["v"])
    n_comp = nx.number_connected_components(g) if g.number_of_nodes() else 0
    largest = max((len(c) for c in nx.connected_components(g)), default=0)
    total_gnodes = g.number_of_nodes()
    ids = [n["id"] for n in nodes]
    from collections import Counter
    cats = dict(Counter(p.get("cat") for p in pois))
    n_res = sum(1 for b in buildings if b.get("kind") in ("residential", "house?"))
    return {
        "nodes": len(nodes),
        "edges": len(edges),
        "buildings": len(buildings),
        "residential_buildings": n_res,
        "pois": len(pois),
        "poi_cats": cats,
        "gateways": sum(1 for n in nodes if n.get("gateway")),
        "underground_edges": sum(1 for e in edges if e.get("layer", 0) < 0),
        "deck_edges": sum(1 for e in edges if e.get("layer", 0) > 0),
        "graph_nodes": total_gnodes,
        "n_components": n_comp,
        "largest_component": largest,
        "largest_frac": round(largest / total_gnodes, 4) if total_gnodes else 0.0,
        "connected": (n_comp == 1),
        "unique_node_ids": (len(ids) == len(set(ids))),
        "ok": bool(nodes and edges and (n_comp == 1) and (len(ids) == len(set(ids)))),
    }


def odpt_key_present(key_env: str = "ODPT_API_KEY") -> bool:
    """ODPT API キーの「有無」だけを返す(値は読まない・出力しない=縮退判定用)。"""
    if os.environ.get(key_env):
        return True
    if sys.platform == "win32":                # User 環境変数(HKCU\Environment)も確認
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                v, _ = winreg.QueryValueEx(k, key_env)
                return bool(v)
        except OSError:
            return False
    return False


def transit_status(key_present: bool, targets_defined: bool,
                   key_env: str = "ODPT_API_KEY") -> dict:
    """交通ダイヤの取得可否を判定。取れなければ「徒歩の街」を宣言(縮退)。

    v0 では make_env 自身は ODPT を叩かない(fetch_odpt→build_transit の2段は手動レシピ)。
    路線識別子(targets)が place 向けに未定義なら、キーがあっても徒歩の街に縮退する。"""
    if key_present and targets_defined:
        return {"available": True, "mode": "odpt",
                "note": ("ODPT キーあり・路線定義あり。fetch_odpt.py → build_transit_odpt.py "
                         "を手動実行して transit を生成できる(v0 は自動実行しない)。")}
    reasons = []
    reasons.append(f"{key_env}={'あり' if key_present else 'なし(未設定)'}")
    reasons.append("路線識別子" + ("定義あり" if targets_defined else "未定義"))
    return {"available": False, "mode": "walking",
            "note": "徒歩の街として継続(transit 未生成)。理由: " + " / ".join(reasons)}


def reduce_roster(src: dict, n: int, seed: int = 42) -> dict:
    """既存名簿(渋谷)から n 名を決定論で機械抽出した縮小版を返す(v0 手動テンプレ扱い)。

    persona 本文は渋谷のまま(語彙の街化は v2 の領分)= README/env_report に明記する縮退。"""
    import random
    personas = list(src.get("personas", []))
    rng = random.Random(seed)
    k = min(n, len(personas))
    sample = rng.sample(personas, k) if k < len(personas) else personas
    meta = dict(src.get("meta", {}))
    meta.update({"reduced_from": meta.get("generator", "unknown"),
                 "reduced_n": k, "reduced_seed": seed,
                 "note": "make_env v0: 渋谷名簿の機械的縮小流用(persona 本文は渋谷のまま)"})
    return {"meta": meta, "personas": sample}


def _rel_to_repo(p: Path) -> str:
    """REPO_ROOT 相対の POSIX パス文字列(env.yaml の data 参照規約に合わせる)。"""
    try:
        return Path(os.path.relpath(p, REPO_ROOT)).as_posix()
    except ValueError:                          # 別ドライブ等: 絶対のまま
        return str(p)


def env_yaml_text(place: str, place_label: str, map_rel: str, *,
                  origin_landmark: str = "", pref: str | None = None,
                  personas_rel: str | None = None, transit_rel: str | None = None,
                  bbox=None, origin_mode: str = "") -> str:
    """W4 ローダ(build_env_overlay)がそのまま読める env.yaml 本文を組む。

    必須キー env.name / data.map を必ず含む。institutions は pref 指定時のみ有効ブロック、
    未指定ならコメント雛形(ローダは読み飛ばす=government 全国既定で成立)。"""
    lines: list[str] = []
    lines.append("# ============================================================")
    lines.append(f"# 環境パック(EnvPack)manifest: {place}")
    lines.append("#   自動生成 v0(scripts/make_env.py・D2)。3層設計は docs/plans/env-classification.md。")
    lines.append("#   ③-A 地理は data/ 実体を参照(移さず取得レシピを指す)。②制度は pref セレクタで ref から引く。")
    lines.append("# ============================================================")
    lines.append("env:")
    lines.append(f"  name: {place}")
    lines.append("  locale: ja-JP")
    lines.append("  purpose: experiment        # demo | experiment | calibration | production")
    lines.append("")
    lines.append("# ---- ③-A 地理層(実体は --out 配下。map は必須)----")
    lines.append("data:")
    lines.append(f"  map:            {map_rel}    # OSM 実道路網+建物+POI。生成: scripts/make_env.py(build_map)")
    if personas_rel:
        lines.append(f"  personas:       {personas_rel}    # 渋谷名簿の機械的縮小流用(v0 手動テンプレ)")
    if transit_rel:
        lines.append(f"  transit:        {transit_rel}")
    lines.append("")
    lines.append("# ---- ③-A 原点: 群集・注目の中心(地図ローカル原点)----")
    lines.append("origin:")
    if origin_landmark:
        lines.append(f"  landmark: {origin_landmark}")
    else:
        lines.append(f"  landmark: \"\"                # 原点モード={origin_mode}(ランドマーク未指定=(0,0)最近傍を集会ノードに)")
    lines.append("")
    lines.append("# ---- ③-B 文化・語彙(v0: 地名のみ機械設定。行事/番組名は base 既定=generic に縮退)----")
    lines.append("culture:")
    lines.append("  lexicon:")
    lines.append(f"    place_name: {place_label}")
    lines.append("    underground_name: 地下通路")
    lines.append("")
    lines.append("# ---- ② 制度: pref セレクタ + ref/institutions_jp.yaml(W4 ローダが展開)----")
    if pref:
        lines.append("institutions:")
        lines.append(f"  pref: {pref}                    # ref.prefectures.{pref}(最低賃金・住民税配分)を引く")
        lines.append("  ref: ../../ref/institutions_jp.yaml")
        lines.append("  # 地点固有(都道府県では決まらない値)は必要に応じ council/rent_income_ratio を追記:")
        lines.append("  # council: {size: 9, term_days: 1460, deposit: 30000}")
        lines.append("  # rent_income_ratio: 0.30")
    else:
        lines.append("# institutions:                    # ← 雛形(pref を該当都道府県に。未指定なら government 全国既定)")
        lines.append("#   pref: tokyo                    #   ref/institutions_jp.yaml のキー(未整備の県は要一次確認)")
        lines.append("#   ref: ../../ref/institutions_jp.yaml")
        lines.append("#   council: {size: 9, term_days: 1460, deposit: 30000}")
        lines.append("#   rent_income_ratio: 0.30")
    lines.append("")
    lines.append("# ---- attribution(出典表示。共有時の権利両立)----")
    lines.append("attribution:")
    lines.append("  - \"地図: © OpenStreetMap contributors(ODbL)。実体は再生成レシピ(scripts/make_env.py / build_map.py)で取得する。\"")
    lines.append("")
    lines.append("notes: |")
    lines.append("  scripts/make_env.py(v0 半自動)で生成。地理は自動、personas/orgs/制度は手動テンプレ(TODO.md 参照)。")
    if bbox is not None:
        lines.append(f"  bbox(S,W,N,E)={list(bbox)}。取得統計・縮退の明示は env_report.md を参照。")
    return "\n".join(lines) + "\n"


def todo_text(place: str, place_label: str, map_rel: str, n_personas: int = 40) -> str:
    """stage3-4 の手動テンプレ手順(personas/orgs の渋谷分布流用の縮小版生成コマンド)。"""
    personas_out = f"env/{place}/personas_{place}.json"
    orgs_out = f"env/{place}/organizations_{place}.json"
    assign_out = f"env/{place}/org_assignments_{place}.json"
    return f"""# env/{place} — 手動テンプレ(stage3-4)TODO

make_env v0 は地理(stage1)と交通判定(stage2)までを自動化する。人口・組織・制度の
「値」は **渋谷の分布パラメータを流用した縮小版** を以下の手順で生成する(v0=手動テンプレ)。

## 1. 人口(personas)= 渋谷の職業/年齢分布を流用した縮小版
gen_personas.py は渋谷の昼間人口分布(職業・年齢・流入)を procedural に再現する。
街を差し替える最小手順は「その分布で N 名を生成」する:

    python scripts/gen_personas.py --pool 3000 --sample {n_personas} --seed 42 \\
        --out {personas_out}

→ 生成後、env.yaml の data: に `personas: {personas_out}` を追記する。
   ※ persona 本文の地名(「渋谷に通勤」等)は渋谷語彙のまま。真の街化(語彙差し替え)は
     v1/v2 の領分(env-classification.md ③-B)。v0 では分布のみ流用する縮退。

## 2. 組織・職場(organizations / assignments)= 地図 POI から導出
build_orgs.py は地図の POI 構成から架空組織台帳と配属を決定論生成する(ほぼ場所非依存):

    python scripts/build_orgs.py --map {map_rel} \\
        --roster {personas_out}
    # 生成物を env.yaml data: の organizations / assignments に接続する
    #   organizations: {orgs_out}
    #   assignments:   {assign_out}
    # ※ build_orgs.py の既定台帳は渋谷テーマの架空社名。街化は v1 の領分。

## 3. 制度(institutions)= ② 共有参照テーブルから pref セレクタで引く
env.yaml の institutions ブロック(雛形)を有効化し、pref をこの街の都道府県に:

    institutions:
      pref: <tokyo 等>            # ref/institutions_jp.yaml のキー(未整備の県は一次確認して追記)
      ref: ../../ref/institutions_jp.yaml
      council: {{size: 9, term_days: 1460, deposit: 30000}}   # 地点固有(議会規模・供託金)
      rent_income_ratio: 0.30

## 4. 交通(transit)= ODPT/GTFS(任意・徒歩の街なら不要)
この街に鉄道があり ODPT で取れるなら、路線識別子を定義して2段で生成する:

    python scripts/fetch_odpt.py --targets-file <lines.json> \\
        --station-title {place_label} --station-suffix <.RomanName>
    python scripts/build_transit_odpt.py --station {place_label} \\
        --keymap-file <keymap.json> --out env/{place}/transit_{place}.json

## 5. 検証
    python scripts/make_env.py --place {place} --out env/{place} --stage 7   # 構造検証+env_report
    python scripts/run.py --env env/{place} run.n_agents=12 run.n_steps=24   # mock スモーク
"""


# --------------------------------------------------------------------------- #
# ステージ
# --------------------------------------------------------------------------- #
def run_stage1(out_dir: Path, place: str, place_label: str,
               bbox: tuple[float, float, float, float], raw: dict,
               origin_latlon=None, origin_poi=None, origin_bbox_center=True,
               osm_date=None) -> dict:
    """stage1: OSM 生データ → map.json(build_map の汎用ビルダー)+ 構造検証。

    raw は CLI が fetch した(またはキャッシュ/モックの)Overpass 応答。ここではネットを触らない。"""
    origin, mode = BM.resolve_origin(
        bbox, raw, latlon=origin_latlon, poi=origin_poi,
        bbox_center_mode=origin_bbox_center)
    # 別の街=渋谷固有の名称マッチ(ハチ公等)とハチ公像フォールバックは無効化。
    data = BM.build(raw, bbox, osm_date, origin=origin,
                    landmarks=[], landmark_name_kws=(), hachiko_fallback=None,
                    map_name=f"{place}_osm",
                    description=f"OSM 実データ(Overpass)による {place_label} 周辺。"
                                f"座標=原点({mode})ローカル平面(m)。make_env v0 生成。")
    out_dir.mkdir(parents=True, exist_ok=True)
    map_path = out_dir / "map.json"
    map_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    report = validate_map(data)
    report.update({"origin_latlon": [round(origin[0], 6), round(origin[1], 6)],
                   "origin_mode": mode, "bbox": list(bbox),
                   "osm_elements": len(raw.get("elements", [])),
                   "map_path": _rel_to_repo(map_path)})
    return report


def run_stage2(key_env: str, targets_defined: bool) -> dict:
    """stage2: 交通ダイヤの取得可否判定(徒歩の街への縮退宣言)。ネット/キー値は触れない。"""
    present = odpt_key_present(key_env)
    return transit_status(present, targets_defined, key_env)


def run_stage34(out_dir: Path, place: str, place_label: str, map_rel: str,
                pref: str | None, personas_from: str | None, n_personas: int,
                seed: int = 42) -> dict:
    """stage3-4: TODO.md(手動テンプレ手順)+ 任意で渋谷名簿の縮小流用 personas を書く。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "TODO.md").write_text(
        todo_text(place, place_label, map_rel, n_personas), encoding="utf-8")
    info: dict = {"todo_path": _rel_to_repo(out_dir / "TODO.md"),
                  "pref": pref, "personas_rel": None, "personas_n": 0}
    if personas_from:
        src_path = Path(personas_from)
        if not src_path.is_absolute():
            src_path = REPO_ROOT / src_path
        src = json.loads(src_path.read_text(encoding="utf-8"))
        reduced = reduce_roster(src, n_personas, seed)
        p_out = out_dir / f"personas_{place}.json"
        p_out.write_text(json.dumps(reduced, ensure_ascii=False, indent=1),
                         encoding="utf-8")
        info["personas_rel"] = _rel_to_repo(p_out)
        info["personas_n"] = len(reduced["personas"])
        info["personas_src"] = _rel_to_repo(src_path)
    return info


def render_report(place: str, place_label: str, state: dict) -> str:
    """env_report.md を state(各 stage の出力)から描画。取得統計・縮退・未実施を明示。"""
    g = state.get("stage1")
    t = state.get("stage2")
    s34 = state.get("stage34")
    L: list[str] = [f"# env_report — {place}({place_label})", "",
                    "自動生成: scripts/make_env.py v0(D2)。取得統計・縮退・未実施を明示する。", ""]

    L.append("## stage1 geography(地図)")
    if g:
        conn = "連結(1成分)" if g["connected"] else f"**非連結**({g['n_components']}成分)"
        L += [
            f"- 原点: {g['origin_latlon']}(モード={g['origin_mode']})/ bbox(S,W,N,E)={g['bbox']}",
            f"- OSM elements 取得数: {g['osm_elements']}",
            f"- ノード {g['nodes']} / エッジ {g['edges']}"
            f"(地下 {g['underground_edges']} / デッキ {g['deck_edges']})",
            f"- 建物 {g['buildings']}(住宅系 {g['residential_buildings']})/ POI {g['pois']} / ゲートウェイ {g['gateways']}",
            f"- POIカテゴリ: {g['poi_cats']}",
            f"- **連結性: {conn}** / 最大成分 {g['largest_component']}/{g['graph_nodes']}"
            f"(={g['largest_frac']})/ ノードID一意={g['unique_node_ids']}",
            f"- 構造検証 ok = **{g['ok']}** / 出力: {g['map_path']}",
        ]
    else:
        L.append("- 未実施(stage1 未実行 または map.json 不在)。")
    L.append("")

    L.append("## stage2 transit(交通)")
    if t:
        if t["available"]:
            L.append(f"- 取得可能(mode={t['mode']}): {t['note']}")
        else:
            L.append(f"- **縮退=徒歩の街(mode={t['mode']})**: {t['note']}")
    else:
        L.append("- 未実施。")
    L.append("")

    L.append("## stage3-4 templates(人口・組織・制度)")
    if s34:
        L.append(f"- 手動テンプレ手順: {s34['todo_path']}(gen_personas / build_orgs / institutions 雛形)")
        if s34.get("personas_rel"):
            L.append(f"- personas(渋谷名簿の機械的縮小流用): {s34['personas_rel']} "
                     f"({s34['personas_n']} 名・src={s34.get('personas_src')})"
                     f" ※persona 本文の地名は渋谷のまま(v0 縮退)")
        else:
            L.append("- personas: 未生成(TODO.md の gen_personas 手順で作る)")
        L.append(f"- institutions: pref={s34.get('pref') or '未設定(雛形コメント。government 全国既定で成立)'}")
    else:
        L.append("- 未実施。")
    L.append("")

    L.append("## 縮退・未実施の明示(誠実性)")
    L.append("- 文化(地域行事・番組名)= LLM 生成せず base 既定(generic)に縮退(捏造ガード=v2 の領分)。")
    L.append("- 語彙 = 地名(place_name)のみ機械設定。persona/組織台帳の渋谷語彙は未街化(v1/v2)。")
    if t and not t["available"]:
        L.append("- 交通 = 徒歩の街に縮退(上記 stage2)。base のダイヤ設定が読まれるが place 固有ダイヤは未生成。")
    L.append("- 気候 = base 既定(東京近似)を継承。place 固有の気候は未設定(v1 以降)。")
    L.append("")
    L.append("## attribution")
    L.append("- 地図: © OpenStreetMap contributors(ODbL)。Overpass 経由取得。EnvPack は取得レシピを指す。")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _state_path(out_dir: Path) -> Path:
    return out_dir / "_make_env_state.json"


def _load_state(out_dir: Path) -> dict:
    p = _state_path(out_dir)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_state(out_dir: Path, state: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _state_path(out_dir).write_text(
        json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="環境自動生成 v0(半自動)。stage1-2 自動 + stage3-4 手動テンプレ。")
    ap.add_argument("--place", required=True, help="街の識別子(env/<place> のスラッグ。例: shimokita)")
    ap.add_argument("--place-label", default=None,
                    help="プロンプト用の街名(既定=--place。例: 下北沢)")
    ap.add_argument("--bbox", default=None,
                    help='取得範囲 "w,s,e,n"(経度西,緯度南,経度東,緯度北)。stage1 に必須')
    ap.add_argument("--out", required=True, help="出力先ディレクトリ(例: env/shimokita)")
    ap.add_argument("--stage", default="all",
                    choices=["1", "2", "34", "7", "all"],
                    help="実行 stage(既定 all。各 stage 独立再実行可)")
    # 原点(build_map の3択。既定=bbox中心=新しい街の妥当な既定)
    og = ap.add_mutually_exclusive_group()
    og.add_argument("--origin-latlon", nargs=2, type=float, metavar=("LAT", "LON"),
                    default=None, help="原点を指定座標に")
    og.add_argument("--origin-poi", default=None, help="原点を取得データ内のPOI名に")
    og.add_argument("--origin-bbox-center", action="store_true",
                    help="原点を bbox 中心に(新しい街の既定)")
    ap.add_argument("--origin-landmark", default="",
                    help="env.yaml origin.landmark に書く名称(任意)")
    ap.add_argument("--osm-date", default=None, help='過去日 "YYYY-MM-DD"(Overpass attic)')
    ap.add_argument("--raw-file", default=None,
                    help="取得済み Overpass 応答 JSON(指定時はネット取得せずこれを使う)")
    ap.add_argument("--save-raw", action="store_true",
                    help="stage1 で取得した OSM 生データを _osm_raw.json に保存(再ビルド用)")
    ap.add_argument("--overpass-retries", type=int, default=6,
                    help="Overpass 取得のリトライ数(ミラー巡回。既定6)")
    # stage2
    ap.add_argument("--key-env", default="ODPT_API_KEY", help="ODPT キーの環境変数名")
    ap.add_argument("--transit-targets-defined", action="store_true",
                    help="この街の ODPT 路線識別子を定義済みと宣言(既定=未定義=徒歩の街)")
    # stage3-4
    ap.add_argument("--pref", default=None,
                    help="institutions.pref(ref のキー。例: tokyo)。未指定=雛形コメント")
    ap.add_argument("--personas-from", default=None,
                    help="縮小流用する渋谷名簿(例: data/personas_100_civic.json)。指定時のみ生成")
    ap.add_argument("--n-personas", type=int, default=40, help="縮小版 personas の件数")
    ap.add_argument("--seed", type=int, default=42, help="縮小抽出の乱数シード")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    place = args.place
    place_label = args.place_label or place
    stages = ["1", "2", "34", "7"] if args.stage == "all" else [args.stage]
    state = _load_state(out_dir)

    # ---------- stage1 geography ----------
    if "1" in stages:
        if not args.bbox:
            print("stage1 には --bbox が必須です(w,s,e,n)", file=sys.stderr)
            return 2
        bbox = parse_bbox(args.bbox)
        if args.raw_file:
            raw_path = Path(args.raw_file)
            if not raw_path.is_absolute():
                raw_path = REPO_ROOT / raw_path
            print(f"[stage1] 取得済み OSM を読む: {raw_path}", file=sys.stderr)
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
        else:
            print(f"[stage1] Overpass から取得中... bbox(S,W,N,E)={bbox}", file=sys.stderr)
            try:
                raw = BM.fetch_overpass(bbox, args.osm_date, retries=args.overpass_retries)
            except RuntimeError as ex:
                print(f"[stage1] **Overpass 取得に失敗(未実施)**: {ex}", file=sys.stderr)
                return 3
            if args.save_raw:
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "_osm_raw.json").write_text(
                    json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        print(f"[stage1] elements: {len(raw.get('elements', []))}", file=sys.stderr)
        rep = run_stage1(
            out_dir, place, place_label, bbox, raw,
            origin_latlon=args.origin_latlon, origin_poi=args.origin_poi,
            origin_bbox_center=(args.origin_bbox_center or
                                (args.origin_latlon is None and args.origin_poi is None)),
            osm_date=args.osm_date)
        state["stage1"] = rep
        state["bbox"] = list(bbox)
        _save_state(out_dir, state)
        print(f"[stage1] map.json: nodes={rep['nodes']} edges={rep['edges']} "
              f"buildings={rep['buildings']} pois={rep['pois']} "
              f"連結={rep['connected']}(最大成分={rep['largest_frac']})", file=sys.stderr)

    # ---------- stage2 transit ----------
    if "2" in stages:
        st = run_stage2(args.key_env, args.transit_targets_defined)
        state["stage2"] = st
        _save_state(out_dir, state)
        print(f"[stage2] transit: {'取得可' if st['available'] else '徒歩の街(縮退)'} — {st['note']}",
              file=sys.stderr)

    # ---------- stage3-4 templates ----------
    if "34" in stages:
        map_rel = (state.get("stage1") or {}).get("map_path") or _rel_to_repo(out_dir / "map.json")
        info = run_stage34(out_dir, place, place_label, map_rel,
                           args.pref, args.personas_from, args.n_personas, args.seed)
        state["stage34"] = info
        _save_state(out_dir, state)
        print(f"[stage34] TODO.md 生成 / personas={info['personas_n'] or '未生成(手順のみ)'}",
              file=sys.stderr)

    # ---------- env.yaml(構造が揃い次第・stage1 or 34 のいずれか実行時に更新)----------
    if ("1" in stages) or ("34" in stages):
        map_rel = (state.get("stage1") or {}).get("map_path") or _rel_to_repo(out_dir / "map.json")
        s34 = state.get("stage34") or {}
        bbox_v = state.get("bbox")
        yaml_text = env_yaml_text(
            place, place_label, map_rel,
            origin_landmark=args.origin_landmark,
            pref=(s34.get("pref") or args.pref),
            personas_rel=s34.get("personas_rel"),
            transit_rel=None,
            bbox=bbox_v,
            origin_mode=(state.get("stage1") or {}).get("origin_mode", ""))
        (out_dir).mkdir(parents=True, exist_ok=True)
        (out_dir / "env.yaml").write_text(yaml_text, encoding="utf-8")
        print(f"[env.yaml] 書き出し: {_rel_to_repo(out_dir / 'env.yaml')}", file=sys.stderr)

    # ---------- stage7 verify + env_report ----------
    if "7" in stages:
        # map.json が state に無ければ、その場で読み直して検証(独立再実行)。
        if "stage1" not in state and (out_dir / "map.json").exists():
            data = json.loads((out_dir / "map.json").read_text(encoding="utf-8"))
            rep = validate_map(data)
            rep.update({"map_path": _rel_to_repo(out_dir / "map.json"),
                        "origin_latlon": data.get("meta", {}).get("origin_latlon"),
                        "origin_mode": "(map.json より)", "bbox": data.get("meta", {}).get("bbox"),
                        "osm_elements": "?"})
            state["stage1"] = rep
        report_md = render_report(place, place_label, state)
        (out_dir).mkdir(parents=True, exist_ok=True)
        (out_dir / "env_report.md").write_text(report_md, encoding="utf-8")
        _save_state(out_dir, state)
        ok = (state.get("stage1") or {}).get("ok")
        print(f"[stage7] env_report.md 生成 / stage1構造検証 ok={ok}", file=sys.stderr)
        if ok is False:
            return 4

    print(f"make_env: 完了(place={place}, out={_rel_to_repo(out_dir)}, stages={stages})",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
