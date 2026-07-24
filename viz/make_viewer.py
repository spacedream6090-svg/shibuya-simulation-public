"""実行結果 → HTML ビューア v5(sim⇄viz 疎結合: ログを読むだけ)。

使い方:  python viz/make_viewer.py runs/day80 [--no-traffic] [--start-tod HH:MM]
  --start-tod : 壁時計の開始時刻を明示上書き(既定は run の sim_min 列から復元。
                sim_min 列が無い旧ランのフォールバック値でもある。未指定既定 07:00)。
生成物(2ファイル分離、ユーザー要望 2026-07-04):
  viewer.html    — 地図ビューア(OSM タイル・レイヤー・再生・フォーカス・フロアビュー)
  dashboard.html — 情報ダッシュボード(出来事 / ネット[X風SNS・LINE風DM・検索] /
                   語彙 / 関係グラフ改良版 / 分析[論文風グラフ] / 施設 / 住民 / シナリオ)
背景タイル: © OpenStreetMap contributors(オンライン時のみ。オフラインでは無地)
検索エンジンはシミュ内データベース(D13 再現性+架空世界の閉性のため実 API 不使用)。
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

STEP_MINUTES = 10           # 1 step = 10 分(sim クロック Clock と同一)
DEFAULT_START_MIN = 7 * 60  # 既定開始 07:00(sim_min 列が無いランのフォールバック=従来値)

# 語彙の伝播(transmission)を1語あたりこの辺数で上限。超過は決定論規則で間引き、
# out["caps"] に (kept/total) を記録=silent cap 禁止(画面にも「表示は上位N」と出す)。
# 大ラン(daily300=130万伝播)でも dashboard が開けるサイズに抑えるための保険。
TRANS_CAP_PER_WORD = 2000


def _parse_start_tod(v) -> int:
    """"HH:MM"(または分 int)を分 of day(0..1439)へ。None/不正は既定 07:00。
    ビューア CLI --start-tod 用の局所パーサ(sim 本体 society.* には依存しない)。"""
    if v is None:
        return DEFAULT_START_MIN
    if isinstance(v, (int, float)):
        return int(v) % 1440
    s = str(v).strip()
    if ":" in s:
        try:
            h, m = s.split(":")
            return (int(h) * 60 + int(m)) % 1440
        except ValueError:
            return DEFAULT_START_MIN
    try:
        return int(s) % 1440
    except ValueError:
        return DEFAULT_START_MIN


def _derive_start_min(events: list) -> int | None:
    """events の sim_min 列から壁時計の原点(day0 step0 の分 of day)を復元する。
    sim_min = start_min + step*STEP_MINUTES の不変量から、任意のイベント 1 件で復元できる
    (全イベントで同値)。sim_min 列が無い/空なら None(呼び出し側が既定 07:00 へ退避)。
    既定 07:00 開始のランは step0 の sim_min=420 → 420 を返し従来 startMin とバイト同一。"""
    for e in events:
        sm = e.get("sim_min")
        if sm is not None:
            return int(sm) - int(e["step"]) * STEP_MINUTES
    return None


# ============================================================ 線路→運行路線の構築
# 渋谷駅の実態(docs/research/rail-shibuya.md)に忠実に、断片化した線路ポリラインを
# 「連続した1本の運行路線」に組み直す純関数群。座標=スクランブル交差点原点のローカル平面(単位=m)。
#   - 終着(渋谷が物理的終端): 銀座線・井の頭線 → mode="terminus"(渋谷端で折り返し)
#   - 相互直通(渋谷を跨いで通過): 東横⇔副都心 / 田都⇔半蔵門 → 2路線を連結し mode="through"
#   - 通過(途中駅): 山手・埼京・湘南新宿・千代田 など → mode="through"
# シミュ挙動には一切触れない(ビューワー描画のためだけのメタデータ)。

def _rail_dist(a: list, b: list) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _rail_len(pts: list) -> float:
    return sum(_rail_dist(pts[i - 1], pts[i]) for i in range(1, len(pts)))


def _rail_assemble(fragments: list, tol: float = 40.0) -> list:
    """断片ポリライン群を端点マッチング(許容 tol[m])で連結し、最長の連続ポリラインを返す。
    最長断片を起点に、両端へ最も近い未使用断片(端点距離 <= tol)を貪欲に継いでいく。
    連結できなかった残り断片は捨てる。"""
    frags = [[list(p) for p in f] for f in fragments if f and len(f) >= 2]
    if not frags:
        return []
    used = [False] * len(frags)
    si = max(range(len(frags)), key=lambda i: _rail_len(frags[i]))
    chain = [list(p) for p in frags[si]]
    used[si] = True
    changed = True
    while changed:
        changed = False
        head, tail = chain[0], chain[-1]
        best = None  # (gap, index, mode)
        for i, f in enumerate(frags):
            if used[i]:
                continue
            fs, fe = f[0], f[-1]
            for gap, mode in ((_rail_dist(tail, fs), "append"),
                              (_rail_dist(tail, fe), "append_rev"),
                              (_rail_dist(head, fe), "prepend"),
                              (_rail_dist(head, fs), "prepend_rev")):
                if gap <= tol and (best is None or gap < best[0]):
                    best = (gap, i, mode)
        if best:
            _, i, mode = best
            f = frags[i]
            used[i] = True
            changed = True
            if mode == "append":
                chain.extend(f)
            elif mode == "append_rev":
                chain.extend(reversed(f))
            elif mode == "prepend":
                chain[:0] = f
            else:  # prepend_rev
                chain[:0] = list(reversed(f))
    return chain


def _rail_densify(pts: list, maxstep: float = 60.0) -> list:
    """隣接点の距離が maxstep を超える区間を等分割し、全区間 < maxstep(< 100m)に整える。
    連続性テストの担保とアニメの滑らかさのため。全長は不変。"""
    if len(pts) < 2:
        return [list(p) for p in pts]
    out = [[pts[0][0], pts[0][1]]]
    for i in range(1, len(pts)):
        a, b = pts[i - 1], pts[i]
        d = _rail_dist(a, b)
        if d > maxstep:
            n = int(d // maxstep) + 1
            for k in range(1, n):
                g = k / n
                out.append([a[0] + (b[0] - a[0]) * g, a[1] + (b[1] - a[1]) * g])
        out.append([b[0], b[1]])
    return out


def _rail_connect(a: list, b: list) -> list:
    """2本の連続ポリライン a,b を最近接端点同士で継いで1本にする(直通ペアの連結)。"""
    dd = sorted(((_rail_dist(a[-1], b[0]), "AB"),
                 (_rail_dist(a[-1], b[-1]), "ArB"),
                 (_rail_dist(a[0], b[0]), "rAB"),
                 (_rail_dist(a[0], b[-1]), "rArB")), key=lambda x: x[0])
    mode = dd[0][1]
    if mode == "AB":
        return a + b
    if mode == "ArB":
        return a + b[::-1]
    if mode == "rAB":
        return a[::-1] + b
    return a[::-1] + b[::-1]


def _rail_to_min(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)  # 24:36 → 1476(翌0時台を跨ぐ終電は 1440 超で表現)


def _rail_norm(name: str) -> str:
    """事業者接頭辞・方向注記を落として路線コア名に正規化(部分一致の精度向上)。
    例: 京王電鉄井の頭線→井の頭線 / JR山手線(内回り)→山手線 / 東京メトロ副都心線→副都心線。"""
    for t in ("JR", "東京メトロ", "東京地下鉄", "東急電鉄", "東急", "京王電鉄", "京王", "電鉄",
              "(内回り)", "(外回り)", "（内回り）", "（外回り）", " ", "　"):
        name = name.replace(t, "")
    return name.strip()


def _rail_match_window(name: str, tlines: list):
    """transit の lines と路線名の部分一致で運行窓を対応付け。複数マッチ(山手 内/外回り等)は
    窓の和集合・headway 平均。対応が無ければ None。"""
    key = _rail_norm(name)
    hits = []
    for ln in tlines:
        tk = _rail_norm(ln.get("name", ""))
        if key and tk and (key in tk or tk in key):
            hits.append(ln)
    if not hits:
        return None
    firsts = [_rail_to_min(h["first"]) for h in hits]
    lasts = [_rail_to_min(h["last"]) for h in hits]
    heads = [h.get("headway_min", 5) for h in hits]
    return (min(firsts), max(lasts), round(sum(heads) / len(heads)))


def _rail_union_window(wa, wb):
    a = wa or (300, 1440, 5)
    b = wb or (300, 1440, 5)
    return (min(a[0], b[0]), max(a[1], b[1]), round((a[2] + b[2]) / 2))


# 渋谷が物理的終端の路線(渋谷端で折り返し)。それ以外は通過扱い。
_TERMINUS_KEYS = ("銀座線", "井の頭")
# 相互直通ペア: (相手A のキー, 相手B のキー, 連結後の路線名)
_THROUGH_PAIRS = (("東横", "副都心", "東急東横線⇔副都心線"),
                  ("田園都市", "半蔵門", "田園都市線⇔半蔵門線"))
_MIN_LINE_LEN = 500.0  # 連結後この長さ未満の路線は演出に値しないスタブとして捨てる


def build_rail_lines(city: dict, transit: dict) -> list:
    """city["railways"](断片化した線路)から、渋谷での実態に沿った運行路線を構築する。
    返り値: [{name, kind, mode, path:[[x,y],...], headway_min, first_min, last_min}, ...]
      mode: "through"(渋谷を跨いで通過)/ "terminus"(渋谷端で折り返し)
    直通ペアは2路線を1本に連結し名前を "…⇔…" に。terminus=銀座線・井の頭線のみ。"""
    tlines = (transit or {}).get("lines", []) or []
    railways = city.get("railways", []) or []

    # 1) name でグループ化(name が「線路」/空 のものは背景描画専用=対象外)
    groups: dict[str, list] = {}
    order: list[str] = []
    for r in railways:
        nm = (r.get("name") or "").strip()
        if nm in ("", "線路"):
            continue
        if nm not in groups:
            groups[nm] = []
            order.append(nm)
        groups[nm].append(r)

    # 2) 各グループを連結→等分割し、運行窓を対応付け
    assembled: dict[str, dict] = {}
    for nm in order:
        path = _rail_assemble([r["geometry"] for r in groups[nm]], tol=40.0)
        if len(path) < 2:
            continue
        path = _rail_densify(path, 60.0)
        if _rail_len(path) < _MIN_LINE_LEN:
            continue
        assembled[nm] = {"name": nm, "kind": groups[nm][0].get("kind", "rail"),
                         "path": path, "win": _rail_match_window(nm, tlines)}

    def _find(sub: str, used: set):
        for nm in order:
            if nm in assembled and nm not in used and sub in nm:
                return nm
        return None

    def _round(path):
        return [[round(x, 1), round(y, 1)] for x, y in path]

    out: list[dict] = []
    used: set = set()

    # 3) 直通ペアを連結して 1 本の through 路線に
    for sa, sb, out_name in _THROUGH_PAIRS:
        na, nb = _find(sa, used), _find(sb, used)
        if na and nb:
            A, Bb = assembled[na], assembled[nb]
            path = _rail_densify(_rail_connect(A["path"], Bb["path"]), 60.0)
            win = _rail_union_window(A["win"], Bb["win"])
            out.append({"name": out_name, "kind": "rail", "mode": "through",
                        "path": _round(path),
                        "first_min": win[0], "last_min": win[1], "headway_min": win[2]})
            used.add(na)
            used.add(nb)

    # 4) 残りの路線: 銀座線・井の頭線=terminus / それ以外=through
    for nm in order:
        if nm not in assembled or nm in used:
            continue
        a = assembled[nm]
        mode = "terminus" if any(k in nm for k in _TERMINUS_KEYS) else "through"
        win = a["win"] or (300, 1440, 5)
        out.append({"name": nm, "kind": a["kind"], "mode": mode,
                    "path": _round(a["path"]),
                    "first_min": win[0], "last_min": win[1], "headway_min": win[2]})
    return out


# ============================================================ 屋内ミクロ(B5 セマンティックズーム)
# 「space_move(L1)/ indoor_tracks サイドカーが有るラン=indoor ON」の時だけ、ビューアへ屋内ミクロ
# データを埋め込む(セマンティックズームで建物内フロア平面+実座標エージェントを描く材料)。無い
# 旧ランには一切足さない=out 不変=ビューア再生成でバイト同一(communities/lens と同型の後方互換)。
# 埋め込む3点: (1) floorSpecs=data/floor_layouts.json の match/shops/zone_mix(JS が n_override を
# シムと同一規則で再現=間取りパリティ)。(2) spaceMoves=区画遷移 [step, agent_idx, w, to_zone](JS が
# carry-forward で現在区画を復元→実座標配置)。(3) contacts=遭遇の短命ハイライト。--indoor-moves の
# 時だけ (4) tracks=秒スケール歩行軌跡ポリライン(サイズガード付き)。
_INDOOR_MOVES_MAX_DAYS = 7          # 屋内軌跡ポリラインを埋め込むランの上限日数(それ超は note のみ)
_INDOOR_MOVES_MAX_PTS = 50000       # 埋め込む軌跡点の総上限(超過は決定論間引き=silent cap 禁止)


def _load_floor_specs_for_viewer(cfg) -> list:
    """data/floor_layouts.json を JS 用に最小化(match + floors[f, use, shops, zone_mix])。

    Python indoor.spec_floor / vision.building_layout の n_override 規則(shops>0 → shops、無ければ
    zone_mix 合計、無ければ None=間取り正典)を JS が同一に再現するための素材。幾何に不要な anchors 等は
    落とす。ファイル不在なら []=JS 側は n_override なし(=従来 floorLayout と同一)。"""
    path = None
    try:
        path = (cfg.get("indoor", {}) or {}).get("floor_layouts_path")
    except Exception:
        path = None
    p = Path(path or "data/floor_layouts.json")
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        return []
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    out = []
    for rec in raw.get("buildings", []):
        floors = []
        for fl in rec.get("floors", []):
            e = {"f": fl.get("f"), "use": fl.get("use")}
            if fl.get("shops"):
                e["shops"] = int(fl["shops"])
            if fl.get("zone_mix"):
                e["zone_mix"] = {k: int(v) for k, v in fl["zone_mix"].items()}
            floors.append(e)
        out.append({"match": list(rec.get("match", [])), "floors": floors})
    return out


def _build_indoor_tracks(samples_path: Path, idx: dict, bld_idx: dict,
                         n_steps: int) -> tuple[str, list | None]:
    """indoor_tracks_samples.parquet → (agent, building, floor) 別の歩行軌跡ポリライン。

    7日超のランは埋め込まず note だけ返す(HTML 肥大ガード)。総点数が上限超なら決定論間引き
    (stride 抽出)して note に明記(silent cap 禁止)。t_s=サブステップ秒(step0 起点の秒)。"""
    if n_steps > _INDOOR_MOVES_MAX_DAYS * 144:
        return (f"屋内軌跡ポリラインは7日以下のランのみ埋込(このランは {n_steps} step="
                f"約{n_steps // 144}日)=非表示", None)
    rows = pq.read_table(samples_path).to_pylist()
    groups: dict[tuple, list] = {}
    for r in rows:
        aid = r["agent_id"]
        bi = bld_idx.get(r["building"])
        if aid not in idx or bi is None:
            continue
        groups.setdefault((idx[aid], bi, int(r["floor"])), []).append(
            (round(float(r["t_s"]), 1), round(float(r["x"]), 1),
             round(float(r["y"]), 1)))
    total = sum(len(v) for v in groups.values())
    stride, note = 1, ""
    if total > _INDOOR_MOVES_MAX_PTS:
        stride = -(-total // _INDOOR_MOVES_MAX_PTS)      # ceil
        note = f"屋内軌跡 {total} 点が上限 {_INDOOR_MOVES_MAX_PTS} 超 → 1/{stride} に間引き表示"
    tracks = []
    for (ai, bi, fl), pts in groups.items():
        pts.sort()
        if stride > 1:
            pts = pts[::stride]
        if len(pts) < 2:
            continue
        tracks.append({"a": ai, "w": 1000 + bi * 100 + fl, "pts": pts})
    return (note, tracks)


def build_indoor_data(events: list, idx: dict, bld_idx: dict, run_dir: Path,
                      n_steps: int, cfg: dict, include_moves: bool) -> dict | None:
    """屋内ミクロ埋め込み(indoor ON のランのみ非 None)。無い旧ランは None=out 不変=バイト同一。"""
    space_moves: list = []
    has_sm = False
    for e in events:
        if e["kind"] != "space_move":
            continue
        has_sm = True
        aid = e["agent_id"]
        if aid not in idx:
            continue
        p = json.loads(e["payload"]) if e["payload"] else {}
        bi = bld_idx.get(p.get("building"))
        if bi is None:
            continue
        space_moves.append([int(e["step"]), idx[aid],
                            1000 + bi * 100 + int(p.get("floor", 1)),
                            int(p.get("to_zone", -1))])
    samples_path = run_dir / "indoor_tracks_samples.parquet"
    contacts_path = run_dir / "indoor_tracks_contacts.parquet"
    if not (has_sm or samples_path.exists()):
        return None                                       # indoor OFF の旧ラン=埋め込みなし
    data: dict = {"floorSpecs": _load_floor_specs_for_viewer(cfg),
                  "spaceMoves": space_moves}
    if contacts_path.exists():
        contacts = []
        for r in pq.read_table(contacts_path).to_pylist():
            bi = bld_idx.get(r["building"])
            ia, ib = idx.get(r["id_a"]), idx.get(r["id_b"])
            if bi is None or ia is None or ib is None:
                continue
            contacts.append([int(r["t_s"] // (STEP_MINUTES * 60)),
                            1000 + bi * 100 + int(r["floor"]), ia, ib])
        if contacts:
            data["contacts"] = contacts
    if include_moves and samples_path.exists():
        note, tracks = _build_indoor_tracks(samples_path, idx, bld_idx, n_steps)
        if tracks:
            data["tracks"] = tracks
        if note:
            data["tracksNote"] = note
    return data


def build_data(run_dir: Path, include_traffic: bool = True,
               start_min: int | None = None,
               include_moves: bool = False) -> dict:
    events = pq.read_table(run_dir / "l1_events.parquet").to_pylist()
    n_steps = max(e["step"] for e in events) + 1

    # 壁時計の原点(day0 step0 の分 of day)。run.start_tod で可変。
    # 明示 start_min(CLI --start-tod)が最優先 → 無ければ events の sim_min 列から復元
    # → それも無ければ既定 07:00。既定 07:00 のランは 420 に戻り従来出力とバイト同一。
    if start_min is None:
        start_min = _derive_start_min(events)
    if start_min is None:
        start_min = DEFAULT_START_MIN

    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    map_path = Path(cfg["world"]["map"])
    if not map_path.is_absolute():
        map_path = REPO_ROOT / map_path
    city = json.loads(map_path.read_text(encoding="utf-8"))

    transit_path = REPO_ROOT / cfg.get("transit", {}).get("file", "data/transit_shibuya.json")
    windows = []
    transit_data: dict = {}
    if transit_path.exists():
        transit_data = json.loads(transit_path.read_text(encoding="utf-8"))
        for line in transit_data["lines"]:
            h, m = line["first"].split(":")
            first = int(h) * 60 + int(m)
            h, m = line["last"].split(":")
            last = int(h) * 60 + int(m)
            windows.append({"name": line["name"], "a": first, "b": last})

    agents_meta = json.loads((run_dir / "agents.json").read_text(encoding="utf-8")) \
        if (run_dir / "agents.json").exists() else []
    agent_ids = sorted({e["agent_id"] for e in events if e["agent_id"] >= 0})
    idx = {aid: i for i, aid in enumerate(agent_ids)}
    if not agents_meta:
        agents_meta = [{"id": a, "name": f"agent{a}", "age": 0, "gender": "?",
                        "occupation": "?", "has_bicycle": False, "has_car": False}
                       for a in agent_ids]

    buildings = city.get("buildings", [])
    bld_idx = {b["id"]: i for i, b in enumerate(buildings)}

    guide_path = REPO_ROOT / "data" / "floorguide_shibuya.json"
    guides = json.loads(guide_path.read_text(encoding="utf-8"))["buildings"] \
        if guide_path.exists() else []

    def guide_for(name: str) -> list | None:
        if not name:
            return None
        for gb in guides:
            for m in gb["match"]:
                if m in name or name in m:
                    return gb["floors"]
        return None

    pois = city.get("pois", [])
    pois_by_bld: dict[str, list] = {}
    for p in pois:
        if p.get("building"):
            pois_by_bld.setdefault(p["building"], []).append(p)

    by_step: dict[int, list[dict]] = {}
    for e in events:
        by_step.setdefault(e["step"], []).append(e)

    # 位置 [x,y,w](w: 0=路上 -1=範囲外 -2=睡眠 1000+bIdx*100+floor=屋内)
    # taxi=3 は rich-tracks 由来。move に taxi が無ければ .get 結果は不変=既存ランはバイト同一。
    mode_code = {"walk": 0, "bicycle": 1, "car": 2, "taxi": 3}
    positions, moves, traffic = [], [], []
    occ_per_step: list[dict[int, int]] = []
    cur = [[0.0, 0.0, 0] for _ in agent_ids]
    for step in range(n_steps):
        mv = [None] * len(agent_ids)
        tr_step = {"n": 0, "segs": []}
        for e in by_step.get(step, []):
            p = json.loads(e["payload"]) if e["payload"] else {}
            kind = e["kind"]
            if kind == "traffic_flow":
                if include_traffic:            # 長期ラン用: --no-traffic で背景交通の巨大な軌跡を除外
                    tr_step = {"n": p.get("n", 0), "segs": p.get("segs", [])}
                continue
            if e["agent_id"] not in idx:
                continue
            i = idx[e["agent_id"]]
            if kind in ("move_segment", "arrive", "speak", "reflect"):
                cur[i][0], cur[i][1] = round(e["x"], 1), round(e["y"], 1)
            if kind == "move_segment" and p.get("pts"):
                mv[i] = [mode_code.get(p.get("mode", "walk"), 0), p["pts"]]
            elif kind == "enter_building":
                bi = bld_idx.get(p.get("building"), 0)
                cur[i] = [round(e["x"], 1), round(e["y"], 1),
                          1000 + bi * 100 + int(p.get("floor", 1))]
            elif kind == "floor_move":
                bi = bld_idx.get(p.get("building"), 0)
                cur[i][2] = 1000 + bi * 100 + int(p.get("floor", 1))
            elif kind == "exit_building":
                cur[i] = [round(e["x"], 1), round(e["y"], 1), 0]
            elif kind == "exit_area":
                cur[i][2] = -1
            elif kind == "enter_area":
                cur[i] = [round(e["x"], 1), round(e["y"], 1), 0]
            elif kind == "sleep_start":
                cur[i][2] = -2
            elif kind == "wake_up":
                cur[i][2] = 0 if cur[i][2] == -2 else cur[i][2]
        positions.append([list(p) for p in cur])
        moves.append(mv)
        traffic.append(tr_step)
        occ = defaultdict(int)
        for p in cur:
            if p[2] >= 1000:
                occ[(p[2] - 1000) // 100] += 1
        occ_per_step.append(dict(occ))

    feed, pairs = [], []
    net_posts, net_dms, net_news, searches = [], [], [], []
    state_series: dict[int, list] = defaultdict(list)
    vocab: dict[str, dict] = {}
    # 語彙の来歴: transmission(聴取1辺=誰から誰へどのチャネルで)を item_id ごとに素で貯める。
    # 後段で語ごとに上限(TRANS_CAP_PER_WORD)まで決定論間引き→ vocab[item]["trans"] へ。
    trans_raw: dict[str, list] = defaultdict(list)
    for e in events:
        p = json.loads(e["payload"]) if e["payload"] else {}
        kind = e["kind"]
        if kind == "sns_post":
            net_posts.append({"s": e["step"], "a": e["agent_id"],
                              "t": p["text"], "i": p.get("items", [])})
            feed.append({"s": e["step"], "a": e["agent_id"], "k": "post",
                         "t": p["text"]})
        elif kind == "dm":
            net_dms.append({"s": e["step"], "a": e["agent_id"],
                            "to": p.get("to"), "t": p["text"]})
            if p.get("to") is not None:
                pairs.append([e["step"], e["agent_id"], p["to"]])
        elif kind == "world_event":
            net_news.append({"s": e["step"], "title": p["title"],
                             "text": p.get("text", ""), "w": p.get("word")})
            feed.append({"s": e["step"], "a": -1, "k": "news", "t": p["title"]})
        elif kind == "search":
            searches.append({"s": e["step"], "a": e["agent_id"],
                             "q": p["query"], "r": p.get("results", [])})
        elif kind == "state_update":
            name = p.get("name")
            if name:
                series = state_series[e["agent_id"]]
                val = round(float(p.get("new", 0.0)), 4)
                if series and series[-1][0] == e["step"]:
                    series[-1] = [e["step"], val]
                else:
                    series.append([e["step"], val])
        if kind == "speak":
            feed.append({"s": e["step"], "a": e["agent_id"], "k": "speak",
                         "t": p["text"], "h": len(p.get("hearers", []))})
            for hearer in p.get("hearers", []):
                pairs.append([e["step"], e["agent_id"], hearer])
        elif kind == "vocab_coin":
            feed.append({"s": e["step"], "a": e["agent_id"], "k": "coin", "t": p["text"]})
            # 発生文脈(ctx): coin payload から item_id/text を除いた残り(fire_reason/place/
            # drive/company_ids/saw_feed/adopted_n/media 等)=語の誕生のきっかけの観測。
            # recent_mem は自由文で嵩むため落とす(観測に必須でなく上限対策)。
            ctx = {k: v for k, v in p.items()
                   if k not in ("item_id", "text", "recent_mem")}
            media = bool(p.get("media")) or e["agent_id"] == -1   # メディア発(creator=-1)
            vocab[p["item_id"]] = {"w": p["text"], "born": e["step"],
                                   "creator": e["agent_id"], "adopts": [],
                                   "ctx": ctx, "media": media}
        elif kind == "transmission":
            # provenance.py: agent_id=聞き手(to)、payload={item_id, from, channel, (dist_m)}。
            iid = p.get("item_id")
            if iid is not None:
                trans_raw[iid].append((e["step"], p.get("from"), e["agent_id"],
                                       p.get("channel", "")))
        elif kind == "label_adopt":
            feed.append({"s": e["step"], "a": e["agent_id"], "k": "adopt", "t": p["text"]})
            if p.get("item_id") in vocab:
                vocab[p["item_id"]]["adopts"].append([e["step"], e["agent_id"]])
        elif kind == "exit_area":
            feed.append({"s": e["step"], "a": e["agent_id"], "k": "exit",
                         "t": "電車で外へ" if p.get("via") == "train" else "歩いて範囲外へ"})
        elif kind == "enter_area":
            feed.append({"s": e["step"], "a": e["agent_id"], "k": "enter",
                         "t": "電車で帰ってきた" if p.get("via") == "train" else "戻ってきた"})

    # ---- L2 集計(論文風グラフの素材)----
    metrics: dict[str, list] = {}
    l2_path = run_dir / "l2_metrics.parquet"
    if l2_path.exists():
        rows = sorted(pq.read_table(l2_path).to_pylist(), key=lambda r: r["step"])
        for key in ("mean_grievance", "n_sleeping", "n_working", "n_cars",
                    "n_inside_buildings", "n_outside", "n_moving",
                    "distinct_vocab_in_use", "n_sns_posts", "total_adoptions",
                    # 第50バッチ 観測レンズの L2 全体スカラー(lens ON 時のみ列が在る)
                    "value4_utility", "value4_emotion", "value4_social",
                    "value4_epistemic", "motive_earn", "motive_love",
                    "motive_recognition", "trust_gini", "trust_top10",
                    # 第55バッチ ペルソナ逸脱率の L2 全体スカラー(deviation ON 時のみ列が在る)
                    "deviation_mean", "deviation_var", "deviation_top_share",
                    "deviation_fulltime_mean"):
            if rows and key in rows[0]:
                metrics[key] = [r.get(key, 0) for r in rows]

    # ---- 集まる場所: 在館人数の多い建物トップ6の時系列 ----
    totals = defaultdict(int)
    for occ in occ_per_step:
        for bi, n in occ.items():
            totals[bi] += n
    top_blds = sorted(totals, key=lambda b: -totals[b])[:6]
    place_series = {
        "names": [buildings[bi].get("name") or f"ビル#{bi}" for bi in top_blds],
        "series": [[occ.get(bi, 0) for occ in occ_per_step] for bi in top_blds]}

    # ---- 語ごとの伝播辺(trans)を上限まで決定論間引き + caps に記録 ----
    # trans: [[step, from, to, channel], ...](to=聞き手)。step 昇順(安定)で格納。
    # 間引き規則(決定論・超過語のみ): ①各採用者の「当人宛の最初の聴取辺」を最優先で残す
    #   (=採用に効いた辺。伝播ネットワーク図のエッジ規則と一致)→ ②残り枠を step の早い辺で充填。
    #   最後に step 昇順へ整列。tn=間引き前の総辺数(常時)。caps=間引いた語の記録(silent 禁止)。
    caps: list[dict] = []
    for iid, v in vocab.items():
        raw = trans_raw.get(iid, [])
        total = len(raw)
        v["tn"] = total
        if total <= TRANS_CAP_PER_WORD:
            kept = sorted(raw, key=lambda r: r[0])   # 安定ソート=同 step は記録順を保つ
        else:
            adopters = {a for _s, a in v["adopts"]}
            first_to: dict[int, int] = {}            # 採用者 → 当人宛の最初の辺の index
            for k, (s, fr, to, ch) in enumerate(raw):
                if to in adopters and to not in first_to:
                    first_to[to] = k
            key_idx = set(first_to.values())         # 採用に効いた辺(最優先で残す)
            chosen = set(key_idx)
            for k in sorted(range(total), key=lambda k: (raw[k][0], k)):
                if len(chosen) >= TRANS_CAP_PER_WORD:
                    break
                chosen.add(k)
            picks = sorted(chosen, key=lambda k: (raw[k][0], k))[:TRANS_CAP_PER_WORD]
            kept = [raw[k] for k in picks]
            caps.append({"item_id": iid, "w": v["w"],
                         "kept": len(kept), "total": total})
        v["trans"] = [[s, fr, to, ch] for (s, fr, to, ch) in kept]

    rails = [{"k": r["kind"], "g": r["geometry"]} for r in city.get("railways", [])]
    rail_lines = build_rail_lines(city, transit_data)

    blds_out = []
    for b in buildings:
        cx = b.get("cx") or sum(p[0] for p in b["footprint"]) / len(b["footprint"])
        cy = b.get("cy") or sum(p[1] for p in b["footprint"]) / len(b["footprint"])
        blds_out.append({
            "id": b["id"],
            "name": b.get("name", ""), "kind": b.get("kind", "generic"),
            "levels": b["levels"], "below": b.get("below", 0),
            "fp": b["footprint"], "cx": round(cx, 1), "cy": round(cy, 1),
            "guide": guide_for(b.get("name", "")),
            "pois": [{"n": p["name"], "f": p.get("floor", 0), "c": p["cat"]}
                     for p in pois_by_bld.get(b["id"], [])[:30]]})

    out = {"agents": agents_meta, "ids": agent_ids, "positions": positions,
            "moves": moves, "traffic": traffic, "feed": feed, "pairs": pairs,
            "vocab": sorted(vocab.values(), key=lambda v: v["born"]),
            "caps": caps,
            "labels": [{"x": n["x"], "y": n["y"], "n": n["name"]}
                       for n in city["nodes"] if n.get("name")],
            "edges": [{"k": e.get("klass", "footway"), "l": e.get("layer", 0),
                       "g": e["geometry"]} for e in city["edges"]],
            "buildings": blds_out,
            "pois": [{"x": p["x"], "y": p["y"], "c": p["cat"], "n": p["name"]}
                     for p in pois],
            "rails": rails,
            "railLines": rail_lines,
            "transit": windows,
            "net": {"posts": net_posts, "dms": net_dms, "news": net_news,
                    "searches": searches},
            "metrics": metrics,
            "stateSeries": {str(k): v for k, v in state_series.items()},
            "places": place_series,
            "origin": city.get("meta", {}).get("origin_latlon"),
            "nSteps": n_steps, "startMin": start_min, "runName": run_dir.name}
    # 第18バッチ①: communities.json が「有る時だけ」自然コミュニティ対応を埋め込む。
    # 無ければ out は一切変わらない=既存ランのビューワー再生成でバイト同一(後方互換)。
    comm = load_communities(run_dir)
    if comm is not None:
        out["communities"] = comm
    # 移動手段(rich-tracks): taxi(mode 3)が現れる時「だけ」mode_legend を埋め込む。
    # 無ければ out は不変=既存ラン(徒歩/自転車/車のみ)はビューワー再生成でバイト同一。
    if any(m is not None and m[0] == 3 for step in moves for m in step):
        out["mode_legend"] = {"0": "徒歩", "1": "自転車", "2": "車", "3": "タクシー"}
    # 第50バッチ 観測レンズ: lens_map.json が「有る時だけ」価値/欲望を事後計算して埋め込む。
    # 無ければ out は不変=既存ランのビューア再生成でバイト同一(communities/mode_legend と同型)。
    lens_map = load_lens_map(run_dir)
    if lens_map is not None:
        out["lens"] = build_lens_data(events, agents_meta, lens_map, start_min)
    # 信用内訳(T6): L3 に status がある(hierarchy ON)時だけ埋め込む。無ければ非表示=不変。
    trust = build_trust_data(events, run_dir, agents_meta)
    if trust is not None:
        out["trust"] = trust
    # ペルソナ逸脱率(第55バッチ タスクA): deviation_map.json が「有る時だけ」住民別逸脱を事後計算。
    # 無ければ out は不変=既存ランのビューア再生成でバイト同一(lens/trust と同型の後方互換)。
    dev_map = load_deviation_map(run_dir)
    if dev_map is not None:
        out["deviation"] = build_deviation_data(events, agents_meta, dev_map, start_min)
    # 社会構造の内生変動(第56バッチ タスクB): structure.json(scripts/analyze_structure.py の事後出力)が
    # 「有る時だけ」そのまま埋め込む。事後層が L1+L3 から計算済み(churn 時系列・順位 τ・中心性 turnover・
    # コミュニティ変化・固着区間)を読むだけ=build_data は再計算しない。無ければ out 不変=後方互換。
    struct = load_structure(run_dir)
    if struct is not None:
        out["structure"] = struct
    # 屋内ミクロ(B5): space_move / indoor_tracks が有るランだけ埋め込む。無ければ out 不変=バイト同一。
    ind = build_indoor_data(events, idx, bld_idx, run_dir, n_steps, cfg, include_moves)
    if ind is not None:
        out.update(ind)
    return out


def load_communities(run_dir: Path) -> dict | None:
    """runs/<name>/communities.json を読み、窓ごとの agent→community 対応へ展開。

    analyze_communities.py の出力(窓→コミュニティ→メンバー)を、ビューワーが
    step で引ける {windows:[{start,end,map:{agent_id:community_id}}]} に畳む。
    ファイルが無い/壊れている場合は None(= 埋め込みなし=従来と同一出力)。
    """
    p = run_dir / "communities.json"
    if not p.exists():
        return None
    try:
        cj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    wins = []
    for w in cj.get("windows", []):
        mp: dict[str, int] = {}
        for c in w.get("communities", []):
            cid = c.get("community_id")
            for aid in c.get("members", []):
                mp[str(aid)] = cid
        wins.append({"start": w.get("step_start", 0), "end": w.get("step_end", 0),
                     "map": mp})
    return {"windows": wins}


# ============================================================ 観測レンズ(第50バッチ)
# 「lens_map.json が有る時だけ」build_data が価値4軸/3M欲望を事後計算する(sim⇄viz 疎結合)。
# 無ければ out に一切足さない=既存ランのビューア再生成でバイト同一(後方互換。communities と同型)。

def load_lens_map(run_dir: Path) -> dict | None:
    """runs/<name>/lens_map.json(observer/lens.py が lens ON 時に書く写像)を読む。無ければ None。"""
    p = run_dir / "lens_map.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _lens_axis(kind: str, payload: dict, kind_map: dict):
    """observer/lens.resolve_axis の Python 鏡(sim なしで L1 を分類。2段マッチ対応)。"""
    spec = kind_map.get(kind)
    if spec is None:
        return None
    if isinstance(spec, str):
        return spec or None
    if isinstance(spec, dict):
        if "__payload_argmax__" in spec:
            key = spec["__payload_argmax__"]
            w = payload.get(key) if isinstance(payload, dict) else None
            if isinstance(w, dict) and w:
                items = sorted(w.items(), key=lambda kv: str(kv[0]))
                best = max(items, key=lambda kv: float(kv[1]))
                return str(best[0]) or None
            return spec.get("_else")
        pkey = spec.get("__payload__")
        val = payload.get(pkey) if (isinstance(payload, dict) and pkey) else None
        if val is not None and str(val) in spec:
            return spec[str(val)]
        return spec.get("_else")
    return None


def build_lens_data(events: list, agents_meta: list, lens_map: dict,
                    start_min: int) -> dict:
    """L1 を価値4軸/3M欲望で事後分類する。全体日別構成 + 住民別プロファイル + 3M 軸間遷移。

    L2 が持つのは全体スカラーのみ(人数>時間の方針)。住民別・遷移はここで L1 から算出=シム実行コスト0。"""
    vaxes = lens_map.get("value_axes", [])
    motives = lens_map.get("motives", [])
    vmap = lens_map.get("value4", {}) if lens_map.get("value_enabled", True) else {}
    mmap = lens_map.get("motive_map", {}) if lens_map.get("motive_enabled", True) else {}

    name_of = {a["id"]: a.get("name", f"a{a['id']}") for a in agents_meta}
    v_daily: dict[int, dict] = defaultdict(lambda: {t: 0 for t in vaxes})
    m_daily: dict[int, dict] = defaultdict(lambda: {t: 0 for t in motives})
    v_profile: dict[int, dict] = defaultdict(lambda: {t: 0 for t in vaxes})
    # 3M 軸間遷移: 同一住民の「欲望を帯びた連続イベント」の (前→後) ペア集計
    m_prev: dict[int, str] = {}
    m_trans: dict[tuple, int] = defaultdict(int)

    def _day(e):
        sm = e.get("sim_min")
        if sm is None:
            sm = start_min + int(e["step"]) * STEP_MINUTES
        return int(sm) // 1440

    for e in events:
        aid = e["agent_id"]
        payload = json.loads(e["payload"]) if e["payload"] else {}
        kind = e["kind"]
        va = _lens_axis(kind, payload, vmap) if vmap else None
        if va in v_daily[_day(e)]:
            v_daily[_day(e)][va] += 1
            if aid >= 0:
                v_profile[aid][va] += 1
        ma = _lens_axis(kind, payload, mmap) if mmap else None
        if ma in m_daily[_day(e)]:
            m_daily[_day(e)][ma] += 1
            if aid >= 0:
                if aid in m_prev:
                    m_trans[(m_prev[aid], ma)] += 1
                m_prev[aid] = ma

    days = sorted(set(v_daily) | set(m_daily))
    value = {
        "axes": vaxes,
        "days": days,
        "series": {t: [v_daily[d][t] for d in days] for t in vaxes},
    }
    # 住民別プロファイル: 総数上位(価値の帯を持つ人)
    prof = sorted(v_profile.items(), key=lambda kv: -sum(kv[1].values()))[:24]
    value["profiles"] = [{"id": aid, "name": name_of.get(aid, f"a{aid}"),
                          "counts": counts, "total": sum(counts.values())}
                         for aid, counts in prof if sum(counts.values()) > 0]
    motive = {
        "motives": motives,
        "days": days,
        "series": {t: [m_daily[d][t] for d in days] for t in motives},
        "transitions": [{"from": f, "to": t, "n": n}
                        for (f, t), n in sorted(m_trans.items(),
                                                key=lambda kv: -kv[1])],
    }
    return {"value": value, "motives": motive}


# ============================================================ ペルソナ逸脱率(第55バッチ タスクA)
# 「deviation_map.json が有る時だけ」build_data が住民別逸脱を L1 から事後計算する(sim⇄viz 疎結合)。
# L2 が持つのは全体スカラーのみ(条件2)。住民別スコア・ドリルダウン・分布はここで L1 から算出。
# in-sim(observer/deviation.classify)の鏡: occupation は agents.json の最終スナップ(非 pool ラン=厳密・
# pool ラン=近似。lens/trust の name_of と同じ既知の割り切り)。無ければ out に一切足さない=後方互換。

def load_deviation_map(run_dir: Path) -> dict | None:
    """runs/<name>/deviation_map.json(observer/deviation.py が ON 時に書く map)を読む。無ければ None。"""
    p = run_dir / "deviation_map.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_structure(run_dir: Path) -> dict | None:
    """runs/<name>/structure.json(scripts/analyze_structure.py の事後出力)を読む。無ければ None。

    事後層が L1+L3 から計算済みの日次時系列(churn / 順位 τ / 中心性 turnover / コミュニティ変化 /
    固着区間)をそのまま返す。ダッシュボードは読むだけ(sim⇄viz 疎結合。communities/deviation と同型)。"""
    p = run_dir / "structure.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


# ============================================================ 日次ロールアップ(第57バッチ タスクC)
# 長期ラン(30日級)は viewer.html の positions(n_steps×n_agents)が支配項で数百MBになり得る。
# --daily-rollup は positions を一切読まず、L2 metrics を日次集計 + structure.json(事後層)を束ねた
# 軽量な rollup.html だけを追加生成する(既存 viewer/dashboard の生成経路には一切触れない=バイト一致)。
# ROLLUP_STEPS_PER_DAY: 1日=144step(= STEP_MINUTES 10分 × 144 = 1440分)。analyze_structure と同一。
ROLLUP_STEPS_PER_DAY = 1440 // STEP_MINUTES        # =144


def _rollup_start_min(run_dir: Path) -> int:
    """L1 の先頭 row group から sim_min 原点を安価に復元(全 L1 は読まない)。無ければ既定 07:00。"""
    path = run_dir / "l1_events.parquet"
    if not path.exists():
        return DEFAULT_START_MIN
    try:
        pf = pq.ParquetFile(path)
        t = pf.read_row_group(0, columns=["step", "sim_min"])
        if t.num_rows == 0:
            return DEFAULT_START_MIN
        sm0 = t.column("sim_min")[0].as_py()
        st0 = t.column("step")[0].as_py()
        if sm0 is None:
            return DEFAULT_START_MIN
        return int(sm0) - int(st0) * STEP_MINUTES
    except Exception:
        return DEFAULT_START_MIN


def build_rollup_data(run_dir: Path) -> dict:
    """L2 metrics を日次平均へ集計 + structure.json を束ねた軽量ロールアップ dict を返す。

    日境界は analyze_structure と同定義(sim_min//1440)。L2 は sim_min 列を持たないため、L1 先頭から
    復元した start_min で day=(start_min+step*10)//1440 を算出して構造指標と日インデックスを揃える。
    L2 が無いラン=metrics 空(structure だけ表示)。どちらも無ければ空の骨格(画面に案内を出す)。"""
    start_min = _rollup_start_min(run_dir)

    def _day(step: int) -> int:
        return (start_min + int(step) * STEP_MINUTES) // 1440

    # L2 を日次平均へ(全数値列を対象=lens ON の列も含め robust。step 列は集計対象外)
    daily: dict[str, list] = {}
    metric_keys: list[str] = []
    days: list[int] = []
    l2_path = run_dir / "l2_metrics.parquet"
    if l2_path.exists():
        rows = sorted(pq.read_table(l2_path).to_pylist(), key=lambda r: r["step"])
        if rows:
            metric_keys = [k for k in sorted(rows[0].keys())
                           if k != "step" and isinstance(rows[0].get(k), (int, float))
                           and not isinstance(rows[0].get(k), bool)]
            sums: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
            cnts: dict[int, int] = defaultdict(int)
            for r in rows:
                d = _day(r["step"])
                cnts[d] += 1
                for k in metric_keys:
                    v = r.get(k)
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        sums[d][k] += float(v)
            days = list(range(0, max(cnts) + 1)) if cnts else []
            for k in metric_keys:
                daily[k] = [round(sums[d][k] / cnts[d], 6) if cnts.get(d) else None
                            for d in days]

    structure = load_structure(run_dir)
    # structure 側の日レンジも取り込んで days を広い方に合わせる(片方だけのランでも表を張れる)
    if structure and structure.get("days"):
        days = list(range(0, max(days[-1] if days else 0,
                                 structure["days"][-1]) + 1))

    n_agents = None
    n_steps = None
    sm = run_dir / "summary.json"
    if sm.exists():
        try:
            s = json.loads(sm.read_text(encoding="utf-8"))
            n_agents = s.get("n_agents")
            n_steps = s.get("n_steps")
        except Exception:
            pass

    return {"runName": run_dir.name, "startMin": start_min,
            "stepsPerDay": ROLLUP_STEPS_PER_DAY, "days": days,
            "metricKeys": metric_keys, "metrics": daily,
            "structure": structure, "nAgents": n_agents, "nSteps": n_steps,
            "hasL2": bool(metric_keys)}


def build_deviation_data(events: list, agents_meta: list, dev_map: dict,
                         start_min: int) -> dict:
    """L1 を「ペルソナ期待 vs 実際の行動」で事後分類する。分布ヒスト + 日別推移 + 最逸脱者ドリルダウン。

    observer/deviation.classify と同一規則を _lens_axis(=resolve_axis の鏡)で再現。裁量(義務除外)を主・
    全時間を参考に。カテゴリ非対応/測定対象外職業/世界イベントは母数に数えない。"""
    occ_of = {a["id"]: a.get("occupation", "") for a in agents_meta}
    name_of = {a["id"]: a.get("name", f"a{a['id']}") for a in agents_meta}
    occ_map = dev_map.get("occupation_map", {})
    hobby_map = dev_map.get("hobby_map", {})
    beh_map = dev_map.get("behavior_map", {})
    work_cat = dev_map.get("work_category", "@work")
    threshold = float(dev_map.get("top_threshold", 0.5))

    def _mk():
        return {"disc_dev": 0, "disc_total": 0, "full_dev": 0, "full_total": 0}

    per: dict[int, dict] = defaultdict(_mk)                      # 住民ごとのラン累計
    actual: dict[int, dict] = defaultdict(lambda: defaultdict(int))  # 住民ごとの裁量カテゴリ構成
    daily: dict[int, dict] = defaultdict(lambda: defaultdict(_mk))   # 日→住民→累計(時系列用)

    def _day(e):
        sm = e.get("sim_min")
        if sm is None:
            sm = start_min + int(e["step"]) * STEP_MINUTES
        return int(sm) // 1440

    for e in events:
        aid = e["agent_id"]
        if aid < 0:
            continue
        occ = occ_of.get(aid)
        if occ is None or occ not in hobby_map:                 # 測定対象外の職業/未知個体
            continue
        payload = json.loads(e["payload"]) if e["payload"] else {}
        cat = _lens_axis(e["kind"], payload, beh_map)
        if cat is None:
            continue
        is_work = (cat == work_cat)
        # 義務(勤務)行動は常に従順(条件1=機械導出=定義上ペルソナに従順。observer/deviation.classify の鏡)。
        conforming = True if is_work else (cat in hobby_map.get(occ, []))
        miss = 0 if conforming else 1
        d = _day(e)
        pa, da = per[aid], daily[d][aid]
        pa["full_total"] += 1
        pa["full_dev"] += miss
        da["full_total"] += 1
        da["full_dev"] += miss
        if not is_work:
            pa["disc_total"] += 1
            pa["disc_dev"] += miss
            actual[aid][cat] += 1
            da["disc_total"] += 1
            da["disc_dev"] += miss

    days = sorted(daily)

    def _mean(recs, kd, kt):
        rs = [r[kd] / r[kt] for r in recs if r[kt] > 0]
        return round(sum(rs) / len(rs), 6) if rs else 0.0

    disc_series = [_mean(daily[d].values(), "disc_dev", "disc_total") for d in days]
    full_series = [_mean(daily[d].values(), "full_dev", "full_total") for d in days]

    # 住民別ラン累計の裁量逸脱率(分布ヒスト + ドリルダウン)
    measured = [(aid, r) for aid, r in per.items() if r["disc_total"] > 0]
    ratios = [r["disc_dev"] / r["disc_total"] for _aid, r in measured]
    bins = [0] * 10                                             # [0,.1)..[.9,1.0] の 10 ビン
    for x in ratios:
        bins[min(9, int(x * 10))] += 1

    top = sorted(measured, key=lambda kv: (-(kv[1]["disc_dev"] / kv[1]["disc_total"]),
                                           kv[0]))[:15]
    top_out = []
    for aid, r in top:
        dr = r["disc_dev"] / r["disc_total"]
        fr = (r["full_dev"] / r["full_total"]) if r["full_total"] > 0 else 0.0
        occ = occ_of.get(aid, "")
        top_out.append({
            "id": aid, "name": name_of.get(aid, f"a{aid}"), "occupation": occ,
            "expect_hobby": list(hobby_map.get(occ, [])),
            "expect_work": occ_map.get(occ),
            "disc_ratio": round(dr, 4), "full_ratio": round(fr, 4),
            "disc_dev": r["disc_dev"], "disc_total": r["disc_total"],
            "actual": dict(sorted(actual[aid].items(), key=lambda kv: (-kv[1], kv[0]))),
        })
    return {
        "days": days,
        "disc_series": disc_series, "full_series": full_series,
        "hist": bins, "n_measured": len(measured),
        "top_threshold": threshold, "top": top_out,
    }


# ============================================================ 信用内訳(第50バッチ T6)
# status.py の合成地位スコアの材料別内訳・分布を L1+L3 から事後再構成する(status を可視化するだけ)。
# hierarchy OFF(L3 に status 列なし)なら None=信用セクション非表示=後方互換。

_TRUST_WEIGHTS = {"rep": 0.25, "followers": 0.25, "wealth": 0.20,
                  "inst": 0.10, "biz": 0.10, "host": 0.10}
# ビューアが L1/L3 から復元できる材料(followers は net の被フォロー数=L1 に残らないため事後再構成不可)。
_TRUST_MATERIALS = ("rep", "wealth", "inst", "biz", "host")


def _gini_py(values: list) -> float:
    xs = sorted(float(v) for v in values)
    n = len(xs)
    if n == 0:
        return 0.0
    total = sum(xs)
    if total <= 0.0:
        return 0.0
    cum = sum(i * x for i, x in enumerate(xs, start=1))
    return round((2.0 * cum) / (n * total) - (n + 1.0) / n, 6)


def build_trust_data(events: list, run_dir: Path, agents_meta: list) -> dict | None:
    """信用(合成地位)の内訳・分布・org 相関を L1+L3 から再構成。hierarchy OFF なら None。"""
    l3 = run_dir / "l3_snapshots.parquet"
    if not l3.exists():
        return None
    status_by: dict[int, float] = {}
    money_by: dict[int, float] = {}
    try:
        rows = pq.read_table(l3).to_pylist()
    except Exception:
        return None
    if rows:
        last = max(rows, key=lambda r: r["step"])
        try:
            state = json.loads(last["state"])
        except Exception:
            state = {}
        for a in state.get("agents", []):
            money_by[a["id"]] = float(a.get("money", 0.0)) + float(a.get("account", 0.0))
            if "status" in a:
                status_by[a["id"]] = float(a["status"])
    if not status_by:                    # hierarchy OFF = status 列なし → 信用レンズは出さない
        return None

    prop_author: dict = {}
    passed: set = set()
    biz: dict = defaultdict(float)
    host: dict = defaultdict(int)
    rep: dict = {}
    for e in events:
        kind = e["kind"]
        p = json.loads(e["payload"]) if e["payload"] else {}
        if kind == "proposal":
            prop_author[p.get("proposal_id")] = e["agent_id"]
        elif kind == "proposal_passed":
            passed.add(p.get("proposal_id"))
        elif kind == "venture_sale":
            biz[e["agent_id"]] += float(p.get("amount", 0.0))
        elif kind == "event_host":
            host[e["agent_id"]] += 1
        elif kind == "reputation_update":
            rep[e["agent_id"]] = float(p.get("new", 0.0))
    inst: dict = defaultdict(int)
    for pid in passed:
        au = prop_author.get(pid)
        if au is not None:
            inst[au] += 1

    ids = sorted(status_by)
    raw = {
        "rep": {i: rep.get(i, 0.0) for i in ids},
        "wealth": {i: money_by.get(i, 0.0) for i in ids},
        "inst": {i: float(inst.get(i, 0)) for i in ids},
        "biz": {i: biz.get(i, 0.0) for i in ids},
        "host": {i: float(host.get(i, 0)) for i in ids},
    }

    def _pct(valmap):
        order = sorted(ids, key=lambda i: (valmap[i], i))
        n = len(order) or 1
        return {i: (pos + 1) / n for pos, i in enumerate(order)}

    pct = {m: _pct(raw[m]) for m in _TRUST_MATERIALS}
    # followers を欠く分、利用可能な 5 材料の重みを再正規化(近似=真の status とは一致しない旨を明記)
    wsum = sum(_TRUST_WEIGHTS[m] for m in _TRUST_MATERIALS) or 1.0
    wn = {m: _TRUST_WEIGHTS[m] / wsum for m in _TRUST_MATERIALS}
    name_of = {a["id"]: a.get("name", f"a{a['id']}") for a in agents_meta}
    role_of = {a["id"]: a.get("org_role", "") for a in agents_meta}

    top = sorted(ids, key=lambda i: -status_by[i])[:15]
    agents_out = [{
        "id": i, "name": name_of.get(i, f"a{i}"), "status": round(status_by[i], 4),
        "role": role_of.get(i, ""),
        "contrib": {m: round(wn[m] * pct[m][i], 4) for m in _TRUST_MATERIALS},
        "pct": {m: round(pct[m][i], 3) for m in _TRUST_MATERIALS},
    } for i in top]

    # org_role 相関: 役割ごとの平均 status(agents.json に org_role がある時のみ意味を持つ)
    by_role: dict = defaultdict(list)
    for i in ids:
        r = role_of.get(i, "")
        if r:
            by_role[r].append(status_by[i])
    org = sorted(({"role": r, "mean": round(sum(v) / len(v), 4), "n": len(v)}
                  for r, v in by_role.items()), key=lambda x: -x["mean"])

    svals = sorted(status_by.values(), reverse=True)
    total = sum(svals)
    k = max(1, int(len(svals) * 0.1))
    top10 = round(sum(svals[:k]) / total, 6) if total > 0 else 0.0
    return {
        "materials": list(_TRUST_MATERIALS),
        "weights": {m: round(wn[m], 4) for m in _TRUST_MATERIALS},
        "agents": agents_out, "org": org,
        "gini": _gini_py(list(status_by.values())), "top10": top10,
        "note": "内訳は L1/L3 からの近似再構成(followers は事後復元不可のため除外・重み再正規化)。",
    }


# ============================================================ 共通 CSS/JS
# 昼夜でテーマ変数を JS から書き換える(_THEME_JS の applyVars)。既定値=昼(Google Maps 風)。
_BASE_CSS = r"""
  :root {
    --bg:#eaedef; --panel:rgba(255,255,255,.92); --panel2:rgba(255,255,255,.96);
    --surface:rgba(255,255,255,.96); --surface2:rgba(0,0,0,.045);
    --ink:#202124; --dim:#5f6368; --accent:#1a73e8; --blue:#1a73e8;
    --line:rgba(0,0,0,.09); --btn:rgba(0,0,0,.055); --btnH:rgba(0,0,0,.11);
    --shadow:0 1px 2px rgba(60,64,67,.3),0 2px 10px rgba(60,64,67,.15);
    --on:rgba(26,115,232,.14); --onInk:#1a73e8;
    --thread:rgba(0,0,0,.035); --bubbleIn:rgba(0,0,0,.06); --bubbleOut:#1a73e8; --bubbleOutInk:#fff;
  }
  * { box-sizing:border-box; margin:0; }
  body { background:var(--bg); color:var(--ink);
         font-family:system-ui,-apple-system,"Hiragino Sans","Yu Gothic UI","Segoe UI",sans-serif;
         transition:background .8s ease; }
  .glass { background:var(--panel); backdrop-filter:blur(16px) saturate(1.15);
           -webkit-backdrop-filter:blur(16px) saturate(1.15);
           border:1px solid var(--line); border-radius:16px; box-shadow:var(--shadow);
           transition:background .8s ease, box-shadow .8s ease, border-color .8s ease, color .8s ease; }
  button { background:var(--btn); color:var(--ink); border:0; border-radius:10px; padding:7px 12px;
           font-size:13px; cursor:pointer; transition:background .15s, color .15s; line-height:1; }
  button:hover { background:var(--btnH); }
  button.on { background:var(--on); color:var(--onInk); }
  select, input[type=text], input[type=number], textarea {
    background:var(--btn); color:var(--ink); border:0; border-radius:9px; padding:6px 9px; font-size:12px; }
  input[type=range] { accent-color:var(--accent); }
  ::-webkit-scrollbar { width:9px; height:9px; }
  ::-webkit-scrollbar-thumb { background:var(--line); border-radius:6px; }
"""

_TIME_JS = r"""
function tstr(s){ const mm=Math.floor(D.startMin+s*10); const d=Math.floor(mm/1440);
  return `Day${d} ${String(Math.floor(mm/60)%24).padStart(2,'0')}:${String(mm%60).padStart(2,'0')}`; }
const hue = i => (i*137.508)%360;
const iOf = Object.fromEntries(D.ids.map((a,i)=>[a,i]));
const nameOf = a => a===-1? '公式' : ((D.agents.find(x=>x.id===a)||{}).name || 'a'+a);
const colOf = a => a===-1? '#f43f5e' : `hsl(${hue(iOf[a]??0)} 70% 60%)`;
"""

# ---- 起動時エラーの可視化(赤帯オーバーレイ)。握り潰さず「表示のみ」——
#      未捕捉例外(今回の地図真っ白のような)を即座に発見できるようにする保険。
_ERR_JS = r"""
window.addEventListener('error', function(ev){
  try {
    let bar=document.getElementById('__errbar');
    if(!bar){ bar=document.createElement('div'); bar.id='__errbar';
      bar.style.cssText='position:fixed;left:0;right:0;top:0;z-index:99999;background:#c0182b;color:#fff;'
        +'font:12px/1.5 ui-monospace,Menlo,Consolas,monospace;padding:8px 12px;white-space:pre-wrap;'
        +'max-height:40vh;overflow:auto;box-shadow:0 2px 10px rgba(0,0,0,.45)';
      (document.body||document.documentElement).appendChild(bar); }
    const m=(ev.error&&ev.error.stack)? ev.error.stack
      : (ev.message+' @'+(ev.filename||'').split('/').pop()+':'+ev.lineno+':'+ev.colno);
    bar.textContent='⚠ JSエラー(この帯が出たら要修正): '+m;
  } catch(_){}   // オーバーレイ自体の失敗で二次被害を出さないためだけの握り(元エラーは既定どおり console にも出る)
});
"""

# ---- 昼夜テーマ(シミュ内時刻に連動して線形補間・CSS変数と canvas 色を駆動)----
_THEME_JS = r"""
// 昼(6-17時)=Google Maps 風ライト / 夜(19-4時)=Google Earth 風ダーク。朝夕は線形補間。
const _DAY = {
  canvasBg:[234,237,239,1], mapBg:[241,243,244,1],
  road:[214,219,223,1], roadEdge:[189,196,203,1],
  under:[124,58,237,.55], deck:[13,148,136,.55],
  rail:[150,161,178,.75], subway:[150,120,200,.5],
  bld:[227,231,234,.92], bldStroke:[60,80,110,.30], bldOcc:[120,165,225,.55],
  name:[60,66,74,1], nameHalo:[255,255,255,.9], poi:[80,86,94,.85],
  bubbleBg:[255,255,255,.97], bubbleInk:[32,33,36,1],
  agentStroke:[255,255,255,.92], glow:[255,196,120,1], night:[6,10,16,0],
  card:[255,255,255,.92], card2:[255,255,255,.97], surface2:[0,0,0,.045],
  ink:[32,33,36,1], dim:[95,99,104,1], accent:[26,115,232,1], blue:[26,115,232,1],
  line:[0,0,0,.09], btn:[0,0,0,.055], btnH:[0,0,0,.11], on:[26,115,232,.14], onInk:[26,115,232,1],
  thread:[0,0,0,.035], bubbleIn:[0,0,0,.06], bubbleOut:[26,115,232,1],
  shadow:'0 1px 2px rgba(60,64,67,.3),0 2px 10px rgba(60,64,67,.15)'
};
const _NIGHT = {
  canvasBg:[8,11,16,1], mapBg:[9,13,20,1],
  road:[40,47,58,1], roadEdge:[24,30,40,1],
  under:[167,139,250,.78], deck:[45,212,191,.72],
  rail:[70,82,102,.85], subway:[120,98,180,.6],
  bld:[24,31,42,.92], bldStroke:[255,255,255,.10], bldOcc:[72,112,172,.72],
  name:[150,161,178,1], nameHalo:[0,0,0,.6], poi:[200,212,232,.9],
  bubbleBg:[24,30,40,.95], bubbleInk:[232,234,237,1],
  agentStroke:[8,12,18,.85], glow:[255,180,90,1], night:[4,7,13,.6],
  card:[20,26,34,.82], card2:[22,28,37,.9], surface2:[255,255,255,.05],
  ink:[232,234,237,1], dim:[154,160,166,1], accent:[138,180,248,1], blue:[122,162,255,1],
  line:[255,255,255,.09], btn:[255,255,255,.06], btnH:[255,255,255,.12], on:[122,162,255,.18], onInk:[173,200,255,1],
  thread:[255,255,255,.03], bubbleIn:[255,255,255,.07], bubbleOut:[34,120,90,1],
  shadow:'0 8px 30px rgba(0,0,0,.5)'
};
const _lerp=(a,b,k)=>a+(b-a)*k;
const _mix=(a,b,k)=>[_lerp(a[0],b[0],k),_lerp(a[1],b[1],k),_lerp(a[2],b[2],k),_lerp(a[3]??1,b[3]??1,k)];
const _css=c=>`rgba(${Math.round(c[0])},${Math.round(c[1])},${Math.round(c[2])},${(c[3]??1).toFixed(3)})`;
function nightAmt(t){ const m=((D.startMin+t*10)%1440+1440)%1440, h=m/60;
  if(h>=6&&h<17) return 0; if(h>=19||h<4) return 1;
  if(h>=17&&h<19) return (h-17)/2; return 1-(h-4)/2; }
let _lastK=-1, _TH=null;
function themeAt(t){
  const k=nightAmt(t);
  if(_TH && Math.abs(k-_lastK)<0.006) return _TH;
  _lastK=k; const T={k};
  for(const key in _DAY){ const dv=_DAY[key];
    if(Array.isArray(dv)){ T[key]=_mix(dv,_NIGHT[key],k); T[key+'C']=_css(T[key]); }
    else T[key]=k<.5?dv:_NIGHT[key]; }
  _TH=T; _applyVars(T); return T;
}
function _applyVars(T){ const r=document.documentElement.style;
  r.setProperty('--bg',_css(T.canvasBg)); r.setProperty('--panel',_css(T.card));
  r.setProperty('--panel2',_css(T.card2)); r.setProperty('--surface',_css(T.card2));
  r.setProperty('--surface2',_css(T.surface2));
  r.setProperty('--ink',_css(T.ink)); r.setProperty('--dim',_css(T.dim));
  r.setProperty('--accent',_css(T.accent)); r.setProperty('--blue',_css(T.blue));
  r.setProperty('--line',_css(T.line)); r.setProperty('--btn',_css(T.btn)); r.setProperty('--btnH',_css(T.btnH));
  r.setProperty('--on',_css(T.on)); r.setProperty('--onInk',_css(T.onInk));
  r.setProperty('--thread',_css(T.thread)); r.setProperty('--bubbleIn',_css(T.bubbleIn));
  r.setProperty('--bubbleOut',_css(T.bubbleOut)); r.setProperty('--shadow',T.shadow); }
"""

# ---- 建物の推定間取り(手続き生成・決定論)+ 屋内エージェント。viewer/dashboard 共用 ----
_FLOOR_JS = r"""
// 間取りは建物 id を seed に決定論生成した「推定(架空)」。実在建物の実内装は非公開(研究倫理R17)。
const USE_JP = {restaurant:'飲食', fashion:'ファッション', beauty:'ビューティー',
  lifestyle:'雑貨・ライフスタイル', food:'食物販', office:'オフィス',
  observation:'展望', hall:'ホール', theatre:'劇場', hotel:'ホテル',
  station:'駅', park:'公園', bus:'バスターミナル'};
const _POOL = {
  fashion:['アパレル','シューズ','バッグ','アクセサリー','セレクト','デニム'],
  beauty:['コスメ','スキンケア','フレグランス','ネイル','ヘアサロン'],
  food:['ベーカリー','スイーツ','デリカ','ワイン','コーヒー','惣菜'],
  restaurant:['カフェ','ダイニング','和食','イタリアン','バー','麺処'],
  lifestyle:['雑貨','インテリア','ブックス','文具','ガジェット','アート'],
  office:['オフィス','会議室','受付','ラウンジ','ワークスペース'],
  shop:['ショップ','ストア','専門店','セレクト'],
  hall:['ホール','ホワイエ','多目的室'], theatre:['劇場','ロビー','クローク'],
  hotel:['客室','フロント','ラウンジ'], station:['コンコース','改札','ホーム','売店'],
  park:['広場','芝生','屋上テラス'], nightlife:['バー','クラブ','ラウンジ'],
  service:['サービス','受付','ATM'], attraction:['展示','ショップ','受付'],
  education:['教室','ラボ','ライブラリ'], generic:['テナント','スペース','区画']};
const _CATMAP = {food:'food', shop:'shop', nightlife:'nightlife', service:'service', office:'office',
  education:'education', attraction:'attraction', hotel:'hotel', leisure:'lifestyle',
  restaurant:'restaurant', fashion:'fashion', beauty:'beauty', lifestyle:'lifestyle',
  observation:'attraction', hall:'hall', theatre:'theatre', station:'station', park:'park', bus:'station'};
const _ZONE_HUE = {food:35, shop:210, nightlife:295, service:160, office:220, education:265,
  attraction:48, hotel:330, lifestyle:180, restaurant:20, fashion:200, beauty:320, hall:255,
  theatre:280, station:0, park:130, generic:210};
function _hash(s){ let h=2166136261>>>0; s=''+s; for(let i=0;i<s.length;i++){ h^=s.charCodeAt(i); h=Math.imul(h,16777619); } return h>>>0; }
function _rng(seed){ let a=seed>>>0; return ()=>{ a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296; }; }
function _cols(a0,a1,n,rng){ if(n<=0) return []; const w=[]; let s=0;
  for(let i=0;i<n;i++){ const v=0.7+rng()*0.7; w.push(v); s+=v; }
  const out=[]; let p=a0; for(const v of w){ const seg=(a1-a0)*v/s; out.push([p,p+seg]); p+=seg; } return out; }
// 建物 b・階 f の推定間取り(決定論)。zones:[{r:[x0,y0,x1,y1],label,cat}], corridor, core, bbox
function floorLayout(b, f){
  const fp=b.fp, xs=fp.map(p=>p[0]), ys=fp.map(p=>p[1]);
  const bbox={x0:Math.min(...xs), x1:Math.max(...xs), y0:Math.min(...ys), y1:Math.max(...ys)};
  const w=bbox.x1-bbox.x0, h=bbox.y1-bbox.y0, cx=(bbox.x0+bbox.x1)/2, cy=(bbox.y0+bbox.y1)/2;
  const rng=_rng(_hash(b.id||b.name||'b')+(f+50)*2654435761>>>0);
  const guide=b.guide? b.guide.find(x=>x.f===f):null;
  const fPois=(b.pois||[]).filter(p=>p.f===f);
  let items=[];
  if(fPois.length) items=fPois.slice(0,10).map(p=>({label:p.n, cat:_CATMAP[p.c]||'shop'}));
  if(!items.length){
    const use=guide? (_CATMAP[guide.use]||'generic')
      : (b.kind==='office'?'office':b.kind==='station'?'station':b.kind==='retail'?'shop':'generic');
    const pool=_POOL[use]||_POOL.generic;
    const n=2+Math.floor(rng()*Math.min(4, pool.length-1));
    const used=new Set();
    for(let i=0;i<n;i++){ let nm; let g=0; do{ nm=pool[Math.floor(rng()*pool.length)]; }while(used.has(nm)&&g++<8);
      used.add(nm); items.push({label:nm, cat:use}); } }
  const n=items.length;
  const horiz = w>=h;                       // 通路の向き(長辺に沿わせる)
  const band = horiz? h*0.09 : w*0.09;
  const zones=[]; const nA=Math.ceil(n/2), nB=n-nA;
  if(horiz){
    const yC0=cy-band, yC1=cy+band;
    _cols(bbox.x0,bbox.x1,nA,rng).forEach((c,i)=>zones.push({r:[c[0],yC1,c[1],bbox.y1], ...items[i]}));
    _cols(bbox.x0,bbox.x1,nB,rng).forEach((c,i)=>zones.push({r:[c[0],bbox.y0,c[1],yC0], ...items[nA+i]}));
    var corridor=[bbox.x0,yC0,bbox.x1,yC1];
  } else {
    const xC0=cx-band, xC1=cx+band;
    _cols(bbox.y0,bbox.y1,nA,rng).forEach((c,i)=>zones.push({r:[xC1,c[0],bbox.x1,c[1]], ...items[i]}));
    _cols(bbox.y0,bbox.y1,nB,rng).forEach((c,i)=>zones.push({r:[bbox.x0,c[0],xC0,c[1]], ...items[nA+i]}));
    var corridor=[xC0,bbox.y0,xC1,bbox.y1];
  }
  const cs=Math.min(w,h)*0.13;
  const core=[cx-cs/2, cy-cs/2, cx+cs/2, cy+cs/2];
  return {bbox, corridor, core, zones, horiz};
}
// 屋内エージェントを区画内に決定論配置(agent id を seed)+ 微動
function _agentSpot(b, f, id, lay, nowT){
  const z = lay.zones.length? lay.zones[_hash('a'+id)%lay.zones.length] : {r:lay.corridor};
  const rng=_rng(_hash('p'+id+':'+(b.id||b.name)+':'+f));
  const u=0.2+rng()*0.6, v=0.2+rng()*0.6;
  let x=z.r[0]+(z.r[2]-z.r[0])*u, y=z.r[1]+(z.r[3]-z.r[1])*v;
  const sp=(z.r[2]-z.r[0]); const amp=Math.min(1.4, Math.max(0.4, sp*0.06));
  x += amp*Math.sin(nowT*0.5+id*1.3); y += amp*Math.sin(nowT*0.42+id*2.1);
  return [x,y];
}

let floorBld=null, floorNo=1, _floorRAF=0;
function floorList(b){ const fs=new Set();
  for(let f=1; f<=b.levels; f++) fs.add(f);
  for(let f=1; f<=(b.below||0); f++) fs.add(-f);
  if(b.guide) for(const g of b.guide) fs.add(g.f);
  return [...fs].sort((a,c)=>a-c); }
function openFloor(bi){ floorBld=bi; const b=D.buildings[bi];
  floorNo = b.guide? b.guide[0].f : (b.below? 1 : 1); if(floorNo===0) floorNo=1;
  document.getElementById('floorModal').style.display='flex';
  if(_floorRAF) cancelAnimationFrame(_floorRAF);
  const tick=()=>{ if(floorBld===null) return; drawFloor(); _floorRAF=requestAnimationFrame(tick); }; tick(); }
function closeFloor(){ document.getElementById('floorModal').style.display='none';
  floorBld=null; if(_floorRAF){ cancelAnimationFrame(_floorRAF); _floorRAF=0; } }
function drawFloor(){
  if(floorBld===null) return;
  const b=D.buildings[floorBld], s0=Math.floor(cur), nowT=performance.now()/1000;
  const T=themeAt(cur), dark=T.k>0.5;
  const fLabel=f=> f<0? `B${-f}`:`${f}F`;
  document.getElementById('floorTitle').textContent=`${b.name||'ビル'} — ${fLabel(floorNo)}`;
  const pos=D.positions[s0], occByFloor={};
  pos.forEach(p=>{ if(p[2]>=1000 && Math.floor((p[2]-1000)/100)===floorBld){ occByFloor[p[2]%100]=(occByFloor[p[2]%100]||0)+1; } });
  document.getElementById('floorBtns').innerHTML=floorList(b).map(f=>
    `<button class="${f===floorNo?'on':''}" onclick="floorNo=${f};drawFloor()">${fLabel(f)}${occByFloor[f]?` · ${occByFloor[f]}`:''}</button>`).join('');
  const g2=b.guide? b.guide.find(x=>x.f===floorNo):null;
  const fPois=(b.pois||[]).filter(p=>p.f===floorNo);
  let gtxt=g2? `📋 ${USE_JP[g2.use]||g2.use}: ${g2.label}`:'';
  if(fPois.length) gtxt+=(gtxt?'<br>':'')+'🏪 '+fPois.map(p=>p.n).join('、');
  document.getElementById('floorGuide').innerHTML=gtxt||'この階の用途情報なし';

  const c=document.getElementById('floorCv');
  c.width=c.clientWidth*devicePixelRatio; c.height=c.clientHeight*devicePixelRatio;
  const g=c.getContext('2d');
  const lay=floorLayout(b, floorNo), fb=lay.bbox;
  const pad=18*devicePixelRatio;
  const s=Math.min((c.width-pad*2)/((fb.x1-fb.x0)||1), (c.height-pad*2)/((fb.y1-fb.y0)||1));
  const ox=(fb.x0+fb.x1)/2, oy=(fb.y0+fb.y1)/2;
  const tfx=(x,y)=>[(x-ox)*s+c.width/2, c.height/2-(y-oy)*s];
  const rectPath=(r)=>{ const [ax,ay]=tfx(r[0],r[1]), [bx,by]=tfx(r[2],r[3]);
    g.beginPath(); g.rect(Math.min(ax,bx),Math.min(ay,by),Math.abs(bx-ax),Math.abs(by-ay)); };
  g.clearRect(0,0,c.width,c.height);
  // 外壁(footprint)に clip → 不整形でも綺麗に収まる
  g.save();
  g.beginPath(); b.fp.forEach((p,j)=>{ const [x,y]=tfx(p[0],p[1]); j?g.lineTo(x,y):g.moveTo(x,y); }); g.closePath();
  g.fillStyle=dark?'#131a23':'#f5f7f9'; g.fill(); g.clip();
  // 区画
  g.font=`${11*devicePixelRatio}px system-ui,sans-serif`; g.textAlign='center';
  for(const z of lay.zones){ const hueV=_ZONE_HUE[z.cat]??210;
    rectPath(z.r); g.fillStyle=`hsla(${hueV} ${dark?40:55}% ${dark?26:80}% / ${dark?.55:.8})`; g.fill();
    g.strokeStyle=dark?'rgba(255,255,255,.10)':'rgba(60,72,90,.28)'; g.lineWidth=1*devicePixelRatio; g.stroke();
    const [lx,ly]=tfx((z.r[0]+z.r[2])/2,(z.r[1]+z.r[3])/2);
    const label=z.label.length>10? z.label.slice(0,9)+'…':z.label;
    g.fillStyle=dark?'rgba(232,234,237,.9)':'rgba(40,48,60,.9)'; g.fillText(label, lx, ly); }
  // 共用通路
  rectPath(lay.corridor); g.fillStyle=dark?'rgba(200,210,225,.10)':'rgba(120,135,155,.16)'; g.fill();
  // コア(EV/階段)
  rectPath(lay.core); g.fillStyle=dark?'#26313f':'#d7dee6'; g.fill();
  g.strokeStyle=dark?'rgba(255,255,255,.18)':'rgba(60,72,90,.4)'; g.lineWidth=1.2*devicePixelRatio; g.stroke();
  const [cix,ciy]=tfx((lay.core[0]+lay.core[2])/2,(lay.core[1]+lay.core[3])/2);
  g.fillStyle=dark?'#9aa0a6':'#5f6368'; g.font=`${12*devicePixelRatio}px system-ui`; g.fillText('🛗', cix, ciy+4*devicePixelRatio);
  g.restore();
  // 外壁ライン
  g.beginPath(); b.fp.forEach((p,j)=>{ const [x,y]=tfx(p[0],p[1]); j?g.lineTo(x,y):g.moveTo(x,y); }); g.closePath();
  g.strokeStyle=dark?'rgba(150,161,178,.6)':'rgba(60,72,90,.55)'; g.lineWidth=2*devicePixelRatio; g.stroke();
  // 屋内エージェント(区画内配置+微動)
  g.font=`${10.5*devicePixelRatio}px system-ui,sans-serif`;
  D.ids.forEach((id,i)=>{ const p=pos[i];
    if(!(p[2]>=1000 && Math.floor((p[2]-1000)/100)===floorBld && p[2]%100===floorNo)) return;
    const [wx,wy]=_agentSpot(b, floorNo, id, lay, nowT);
    const [x,y]=tfx(wx,wy);
    g.beginPath(); g.arc(x,y,5*devicePixelRatio,0,7); g.fillStyle=colOf(id); g.fill();
    g.strokeStyle=dark?'rgba(0,0,0,.5)':'rgba(255,255,255,.9)'; g.lineWidth=1.4*devicePixelRatio; g.stroke();
    const nm=nameOf(id); g.fillStyle=dark?'rgba(232,234,237,.92)':'rgba(40,48,60,.92)'; g.textAlign='center';
    g.fillText(nm.length>6?nm.slice(0,6):nm, x, y-8*devicePixelRatio); });
}
"""


# ============================================================ 屋内セマンティックズーム(B5)
# space_move / indoor_tracks が有るラン「だけ」main() が __INDOOR_JS__ に注入する追加 JS。
#  - floorLayout を n_override 対応版へ差し替え(D.floorSpecs=floor_layouts の match/shops/zone_mix を
#    Python indoor.spec_floor / vision.building_layout と同一規則で読み、cols 前に乱数を消費しない
#    POI 経路と同型で n=min(12,shops|Σzone_mix) 分割=シム⇄ビューアの間取りパリティ)。
#  - cam.s≥2.6 で地図上に建物フロア平面(floorLayout)を直接描画+実座標エージェント(space_move の
#    carry-forward 区画)。±15% クロスフェード・建物ごとの表示階チップ(canvas 上の小 UI・クリックで切替)。
#  - --indoor-moves の時は歩行軌跡ポリライン(D.tracks)を再生、遭遇(D.contacts)は短命ハイライト。
# 旧ラン(space_move 無し)では main() が __INDOOR_JS__/__INDOOR_HOOK__/__INDOOR_CLICK__ を空文字へ
# 置換するため、生成 HTML は従来とバイト同一(communities/lens と同型の後方互換)。
_INDOOR_JS = r"""
// ---- floor_layouts spec 引き(Python indoor._match と同一規約: 部分一致・双方向)----
function izSpecFloor(b, f){
  const specs=D.floorSpecs; if(!specs) return null;
  const name=(b.name||b.id||''); if(!name) return null;           // 空名は override 対象外(Python spec_floor と同)
  for(const rec of specs){
    if((rec.match||[]).some(m=> m && (name.indexOf(m)>=0 || m.indexOf(name)>=0))){
      for(const fl of (rec.floors||[])) if((fl.f|0)===(f|0)) return fl;
      return null;                                                // 建物一致・その階の spec 無し
    }
  }
  return null;
}
// 区画数 override: shops(明示区画数)> zone_mix 合計 > null(=間取り正典の既定)。Python indoor._build と同順。
function izNOverride(b, f){
  const sp=izSpecFloor(b, f); if(!sp) return null;
  if(sp.shops) return sp.shops|0;
  if(sp.zone_mix){ let s=0; for(const k in sp.zone_mix) s+=(sp.zone_mix[k]|0); return s; }
  return null;
}
// floorLayout を n_override 対応へ差し替え(旧 floorLayout と幾何完全一致 + override 分岐を追加)。
// override 経路は POI 経路と同型に cols 前で乱数を1つも消費しない=vision.building_layout(n_override)と
// ビット一致(seed/_rng/_cols は既に vision と回帰済み)。ラベルは spec.use のプールを cols 前の乱数
// 非消費で敷く(幾何に無影響=表示専用)。
floorLayout = function(b, f){
  const fp=b.fp, xs=fp.map(p=>p[0]), ys=fp.map(p=>p[1]);
  const bbox={x0:Math.min(...xs), x1:Math.max(...xs), y0:Math.min(...ys), y1:Math.max(...ys)};
  const w=bbox.x1-bbox.x0, h=bbox.y1-bbox.y0, cx=(bbox.x0+bbox.x1)/2, cy=(bbox.y0+bbox.y1)/2;
  const rng=_rng(_hash(b.id||b.name||'b')+(f+50)*2654435761>>>0);
  const guide=b.guide? b.guide.find(x=>x.f===f):null;
  const fPois=(b.pois||[]).filter(p=>p.f===f);
  let items=[];
  const nOv=izNOverride(b, f);
  if(nOv!==null){                             // ① override(floor_layouts の shops/zone_mix)=最優先
    let n=Math.min(12, nOv); if(n<=0) n=1;
    const sp=izSpecFloor(b, f);
    const use=_CATMAP[sp&&sp.use]||'generic', pool=_POOL[use]||_POOL.generic;
    for(let i=0;i<n;i++) items.push({label: pool[i%pool.length], cat: use});   // ラベル=cols 前に乱数非消費
  } else if(fPois.length){                     // ② POI 経路(従来どおり・乱数非消費)
    items=fPois.slice(0,10).map(p=>({label:p.n, cat:_CATMAP[p.c]||'shop'}));
  }
  if(!items.length){                           // ③ 用途プール(従来どおり・乱数消費)
    const use=guide? (_CATMAP[guide.use]||'generic')
      : (b.kind==='office'?'office':b.kind==='station'?'station':b.kind==='retail'?'shop':'generic');
    const pool=_POOL[use]||_POOL.generic;
    const n=2+Math.floor(rng()*Math.min(4, pool.length-1));
    const used=new Set();
    for(let i=0;i<n;i++){ let nm; let g=0; do{ nm=pool[Math.floor(rng()*pool.length)]; }while(used.has(nm)&&g++<8);
      used.add(nm); items.push({label:nm, cat:use}); } }
  const n=items.length;
  const horiz = w>=h;
  const band = horiz? h*0.09 : w*0.09;
  const zones=[]; const nA=Math.ceil(n/2), nB=n-nA;
  if(horiz){
    const yC0=cy-band, yC1=cy+band;
    _cols(bbox.x0,bbox.x1,nA,rng).forEach((c,i)=>zones.push({r:[c[0],yC1,c[1],bbox.y1], ...items[i]}));
    _cols(bbox.x0,bbox.x1,nB,rng).forEach((c,i)=>zones.push({r:[c[0],bbox.y0,c[1],yC0], ...items[nA+i]}));
    var corridor=[bbox.x0,yC0,bbox.x1,yC1];
  } else {
    const xC0=cx-band, xC1=cx+band;
    _cols(bbox.y0,bbox.y1,nA,rng).forEach((c,i)=>zones.push({r:[xC1,c[0],bbox.x1,c[1]], ...items[i]}));
    _cols(bbox.y0,bbox.y1,nB,rng).forEach((c,i)=>zones.push({r:[bbox.x0,c[0],xC0,c[1]], ...items[nA+i]}));
    var corridor=[xC0,bbox.y0,xC1,bbox.y1];
  }
  const cs=Math.min(w,h)*0.13;
  const core=[cx-cs/2, cy-cs/2, cx+cs/2, cy+cs/2];
  return {bbox, corridor, core, zones, horiz};
};

// ---- space_move の carry-forward で現在区画を復元(実データからの実座標配置)----
let _izMoves=null;                                // agent_idx -> [[step,w,zone]...](step 昇順)
function izMovesFor(i){
  if(_izMoves===null){ _izMoves={};
    for(const m of (D.spaceMoves||[])){ (_izMoves[m[1]]||(_izMoves[m[1]]=[])).push([m[0],m[2],m[3]]); }
  }
  return _izMoves[i]||[];
}
// agent i の s0 時点の区画(現在位置 w に一致する直近の space_move の to_zone)。無ければ -1(=未確定)。
function izZoneOf(i, s0, w){
  const mv=izMovesFor(i); let z=-1;
  for(let k=0;k<mv.length;k++){ if(mv[k][0]>s0) break; if(mv[k][1]===w) z=mv[k][2]; }
  return z;
}
// 区画内の決定論点(内側20%余白・hash ジッタ)。sim の _indoor_zone_point は blake2b で JS 再現不可の
// ため mulberry32 で近似(実区画=実データ、区画内の正確な点だけは再現しない=正直な妥協)。
function izZonePoint(r, id){
  const rr=_rng(_hash('iz'+id)); const mx=(r[2]-r[0])*0.2, my=(r[3]-r[1])*0.2;
  const u=rr(), v=rr();
  return [r[0]+mx+u*Math.max(0,(r[2]-r[0]-2*mx)), r[1]+my+v*Math.max(0,(r[3]-r[1]-2*my))];
}

const IZ_FULL=2.6;                                // フロア平面が完全に立つ cam.s(±15% でクロスフェード)
let izFloorSel={};                                // bi -> ユーザ選択階(未選択は自動=最多在館階)
let izChips=[];                                   // 表示階チップの当たり矩形(screen px・毎フレーム再構築)
let _izCurF={};                                   // bi -> 現在表示中の階(▲▼ ナビの起点)
let _izScreen={};                                 // i -> [sx,sy] この階に描いたエージェントの画面位置(遭遇線用)
function izAlpha(){ return Math.max(0, Math.min(1, (cam.s - IZ_FULL*0.85)/(IZ_FULL*0.30))); }

// (bi,floor) 別の在館人数(この step)。selF 自動選択とチップの人数表示に使う。
function izOccByFloor(pos){
  const occ={};
  for(let i=0;i<pos.length;i++){ const w=pos[i][2]; if(w<1000) continue;
    const bi=Math.floor((w-1000)/100), fl=w%100;
    (occ[bi]||(occ[bi]={}))[fl]=(occ[bi][fl]||0)+1; }
  return occ;
}
// 建物 b の全階に在館人数を重ねた {floor: 人数}(未在館階は 0)。表示階チップ用。
function izFloorsFor(b, occ){
  const out={}; for(const f of floorList(b)) out[f]=0;
  if(occ) for(const fl in occ) out[+fl]=occ[fl];
  return out;
}
// 表示階の決定: ユーザ選択 > 最多在館階(同数は小階)> 1F(無ければ最小階)。
function izAutoFloor(floors, occ){
  if(occ){ let selF=null, best=-1;
    for(const fl in occ){ const c=occ[fl], fn=+fl;
      if(c>best || (c===best && (selF===null || fn<selF))){ best=c; selF=fn; } }
    if(selF!==null) return selF; }
  const fs=Object.keys(floors).map(Number).sort((a,c)=>a-c);
  return fs.indexOf(1)>=0? 1 : (fs.length? fs[0] : 1);
}
// draw() 末尾フック: cam.s≥閾値で「ビューポート内の建物」にフロア平面(floorLayout)+実座標
// エージェントを重ねる=マクロを注視するとミクロ(間取り)が現れる。小さすぎる建物は詳細を出さない
// (画面上のフットプリントが小さい=details-on-demand の粒度制御)。±15% クロスフェード。
function indoorOverlay(t, s0, pos){
  izChips=[];
  const a=izAlpha(); if(a<=0) return;
  const T=themeAt(t), dark=T.k>0.5;
  const occBF=izOccByFloor(pos);
  ctx.save(); ctx.globalAlpha=a;
  D.buildings.forEach((b,bi)=>{
    const [sx,sy]=tf(b.cx,b.cy);
    if(sx<-160||sy<-160||sx>cv.width+160||sy>cv.height+160) return;
    // 画面上フットプリントの対角がこの閾値未満の建物は詳細を描かない(ノイズ抑制)。
    const dpx=Math.hypot(b.fp.reduce((m,p)=>Math.max(m,p[0]),-1e9)-b.fp.reduce((m,p)=>Math.min(m,p[0]),1e9),
                         b.fp.reduce((m,p)=>Math.max(m,p[1]),-1e9)-b.fp.reduce((m,p)=>Math.min(m,p[1]),1e9))*cam.s;
    if(dpx<70*devicePixelRatio && !occBF[bi]) return;
    const floors=izFloorsFor(b, occBF[bi]);
    let selF=izFloorSel[bi];
    if(selF===undefined || !(selF in floors)) selF=izAutoFloor(floors, occBF[bi]);
    _izCurF[bi]=selF;
    izDrawFloorPlan(b, bi, selF, s0, t, pos, dark);
    izDrawChips(bi, floors, selF, sx, sy, dark);
  });
  ctx.restore();
}

function izDrawFloorPlan(b, bi, fl, s0, t, pos, dark){
  const lay=floorLayout(b, fl);
  const rp=(r)=>{ const [ax,ay]=tf(r[0],r[1]), [bx,by]=tf(r[2],r[3]);
    return [Math.min(ax,bx),Math.min(ay,by),Math.abs(bx-ax),Math.abs(by-ay)]; };
  // 外壁 footprint で clip
  ctx.save();
  ctx.beginPath(); b.fp.forEach((p,j)=>{ const [x,y]=tf(p[0],p[1]); j?ctx.lineTo(x,y):ctx.moveTo(x,y); }); ctx.closePath();
  ctx.fillStyle=dark?'rgba(19,26,35,.96)':'rgba(245,247,249,.96)'; ctx.fill(); ctx.clip();
  ctx.font=`${9.5*devicePixelRatio}px system-ui,sans-serif`; ctx.textAlign='center';
  for(const z of lay.zones){ const hueV=_ZONE_HUE[z.cat]??210; const [x,y,w,h]=rp(z.r);
    ctx.fillStyle=`hsla(${hueV} ${dark?42:56}% ${dark?28:80}% / ${dark?.62:.85})`; ctx.fillRect(x,y,w,h);
    ctx.strokeStyle=dark?'rgba(255,255,255,.1)':'rgba(60,72,90,.28)'; ctx.lineWidth=1; ctx.strokeRect(x,y,w,h);
    if(w>34*devicePixelRatio && h>13*devicePixelRatio){
      ctx.fillStyle=dark?'rgba(232,234,237,.85)':'rgba(40,48,60,.85)';
      const lb=z.label && z.label.length>8? z.label.slice(0,7)+'…':(z.label||'');
      ctx.fillText(lb, x+w/2, y+h/2+3*devicePixelRatio); } }
  { const [x,y,w,h]=rp(lay.corridor); ctx.fillStyle=dark?'rgba(200,210,225,.09)':'rgba(120,135,155,.15)'; ctx.fillRect(x,y,w,h); }
  { const [x,y,w,h]=rp(lay.core); ctx.fillStyle=dark?'#26313f':'#d7dee6'; ctx.fillRect(x,y,w,h);
    ctx.strokeStyle=dark?'rgba(255,255,255,.18)':'rgba(60,72,90,.4)'; ctx.strokeRect(x,y,w,h); }
  ctx.restore();
  // 外壁ライン
  ctx.beginPath(); b.fp.forEach((p,j)=>{ const [x,y]=tf(p[0],p[1]); j?ctx.lineTo(x,y):ctx.moveTo(x,y); }); ctx.closePath();
  ctx.strokeStyle=dark?'rgba(150,161,178,.65)':'rgba(60,72,90,.55)'; ctx.lineWidth=1.6*devicePixelRatio; ctx.stroke();
  // 歩行軌跡ポリライン(--indoor-moves)+ 遭遇の短命ハイライトはエージェント描画の下地に敷く
  const w=1000+bi*100+fl;
  izDrawTracks(w, t);
  // 実座標エージェント(現在区画の centroid 付近・未確定は従来 _agentSpot へフォールバック)
  _izScreen={};
  for(let i=0;i<D.ids.length;i++){ if(pos[i][2]!==w) continue;
    const zi=izZoneOf(i, s0, w); let wx, wy;
    if(zi>=0 && zi<lay.zones.length){ const sp=izZonePoint(lay.zones[zi].r, D.ids[i]); wx=sp[0]; wy=sp[1]; }
    else { const sp=_agentSpot(b, fl, D.ids[i], lay, t); wx=sp[0]; wy=sp[1]; }
    const [x,y]=tf(wx,wy); _izScreen[i]=[x,y];
    ctx.beginPath(); ctx.arc(x,y,4.5*devicePixelRatio,0,7); ctx.fillStyle=colorOf(i,s0); ctx.fill();
    ctx.strokeStyle='#ffd166'; ctx.lineWidth=1.3*devicePixelRatio; ctx.stroke(); }
  izDrawContacts(w, s0, t);
}

// 表示階チップ(建物中心の右に ▲ / 現在階+人数 / ▼ の3セル縦帯。高層でも一定サイズ)。
// クリックで表示階を切替(高さに依らず使える=up/down で floorList を辿る)。α はクロスフェード追従。
function izDrawChips(bi, floors, selF, sx, sy, dark){
  const cw=30*devicePixelRatio, ch=15*devicePixelRatio, gap=2*devicePixelRatio;
  const x0=sx+10*devicePixelRatio, y0=sy-(3*(ch+gap))/2;
  const occN=floors[selF]||0;
  const lbl=(selF<0?'B'+(-selF):selF+'F')+(occN?(' ·'+occN):'');
  const cells=[['▲','up',false],[lbl,'lbl',true],['▼','down',false]];
  ctx.font=`${8.5*devicePixelRatio}px system-ui,sans-serif`; ctx.textAlign='center';
  cells.forEach((cell,k)=>{ const y=y0+k*(ch+gap), isLbl=cell[2];
    ctx.fillStyle= isLbl? (dark?'#ffd166':'#1a73e8') : (dark?'rgba(20,26,35,.85)':'rgba(255,255,255,.9)');
    ctx.beginPath(); ctx.roundRect(x0,y,cw,ch,3*devicePixelRatio); ctx.fill();
    ctx.strokeStyle=dark?'rgba(255,255,255,.2)':'rgba(60,72,90,.35)'; ctx.lineWidth=1; ctx.stroke();
    ctx.fillStyle= isLbl? (dark?'#101418':'#fff') : (dark?'#c8ccd2':'#3a424e');
    ctx.fillText(cell[0], x0+cw/2, y+ch/2+3*devicePixelRatio);
    izChips.push({x:x0,y:y,w:cw,h:ch,bi:bi,act:cell[1]}); });
}

// 歩行軌跡: 表示階のポリラインを淡く敷き、現在 sim 秒(t*600)がその時間帯なら明点を進める。
function izDrawTracks(w, t){
  if(!D.tracks) return; const nowS=t*600;
  for(const tk of D.tracks){ if(tk.w!==w) continue; const pts=tk.pts; if(pts.length<2) continue;
    ctx.beginPath(); pts.forEach((p,j)=>{ const [x,y]=tf(p[1],p[2]); j?ctx.lineTo(x,y):ctx.moveTo(x,y); });
    ctx.strokeStyle='rgba(120,180,250,.35)'; ctx.lineWidth=1.4*devicePixelRatio; ctx.stroke();
    if(nowS>=pts[0][0] && nowS<=pts[pts.length-1][0]){
      let px=pts[0][1], py=pts[0][2];
      for(let j=1;j<pts.length;j++){ if(pts[j][0]>=nowS){ const t0=pts[j-1][0], t1=pts[j][0];
        const g=t1>t0? (nowS-t0)/(t1-t0):0;
        px=pts[j-1][1]+(pts[j][1]-pts[j-1][1])*g; py=pts[j-1][2]+(pts[j][2]-pts[j-1][2])*g; break; } }
      const [x,y]=tf(px,py); ctx.beginPath(); ctx.arc(x,y,3*devicePixelRatio,0,7);
      ctx.fillStyle='#60a5fa'; ctx.fill(); } }
}
// 遭遇(coplace/meeting): この step・この階の接触ペアを短命ハイライト(2エージェント間の脈打つ線)。
function izDrawContacts(w, s0, t){
  if(!D.contacts) return; const pulse=0.4+0.6*Math.abs(Math.sin(t*3.0));
  for(const c of D.contacts){ if(c[0]!==s0 || c[1]!==w) continue;
    const pa=_izScreen[c[2]], pb=_izScreen[c[3]]; if(!pa||!pb) continue;
    ctx.strokeStyle=`rgba(255,140,90,${(0.35+0.4*pulse).toFixed(3)})`; ctx.lineWidth=2*devicePixelRatio;
    ctx.beginPath(); ctx.moveTo(pa[0],pa[1]); ctx.lineTo(pb[0],pb[1]); ctx.stroke();
    const mx=(pa[0]+pb[0])/2, my=(pa[1]+pb[1])/2;
    ctx.beginPath(); ctx.arc(mx,my,(3+3*pulse)*devicePixelRatio,0,7);
    ctx.fillStyle=`rgba(255,160,100,${(0.5*pulse).toFixed(3)})`; ctx.fill(); }
}
// clickAt フック: フロア平面モードで表示階チップを最優先ヒットテスト(建物/エージェント選択より先)。
// ▲/▼ は floorList を1つ辿る(現在表示階 _izCurF から相対移動)。ラベルはヒット吸収のみ。
function izClickChips(px, py){
  for(const c of izChips){ if(!(px>=c.x&&px<=c.x+c.w&&py>=c.y&&py<=c.y+c.h)) continue;
    if(c.act==='lbl') return true;
    const b=D.buildings[c.bi]; if(!b) return true;
    const fs=floorList(b); const cur=(c.bi in _izCurF)? _izCurF[c.bi] : (fs[0]||1);
    let k=fs.indexOf(cur); if(k<0) k=0;
    k=Math.max(0, Math.min(fs.length-1, k+(c.act==='up'?1:-1)));
    izFloorSel[c.bi]=fs[k]; return true; }
  return false;
}
"""


# ============================================================ コミュニティ色分け
# 第18バッチ①: runs/<name>/communities.json が「有る時だけ」注入する追加 JS。
# communityColor(i, s0) は再生中の step s0 が属する窓の agent→community 対応を引き、
# コミュニティ id から決定論パレット(黄金角 hue)で色を返す。無所属はグレー。
# ファイルが無いランでは main() が空文字へ置換するため、出力は従来とバイト同一。
_COMMUNITY_JS = r"""
// コミュニティ色分け(communities.json 由来)。窓境界で所属が変わると色も切り替わる。
function communityColor(i, s0){
  const wins = (D.communities && D.communities.windows) || [];
  const aid = D.ids[i];
  for(let wi=0; wi<wins.length; wi++){
    const w = wins[wi];
    if(s0 >= w.start && s0 < w.end){
      const cid = w.map[aid];
      if(cid===undefined || cid===null) return '#8a97a5';   // 無所属=グレー
      const h = ((cid*137.508)%360+360)%360;                // 黄金角パレット(決定論)
      return `hsl(${h.toFixed(1)} 68% 58%)`;
    }
  }
  return '#8a97a5';
}"""


# ============================================================ 移動手段の凡例(mode_legend)
# rich-tracks(taxi=mode 3)が現れた時「だけ」main() が注入する凡例1行。colorBy='移動' の
# 時だけ表示。色は colorOf の mode 分岐(徒歩/自転車/車/タクシー)と一致させる。
# 無いランでは main() が空文字へ置換するため、viewer.html は従来とバイト同一。
_MODE_LEGEND_JS = r"""(function(){
  const LEG = D.mode_legend; if(!LEG) return;
  const MC = {0:'#60a5fa',1:'#6ee7b7',2:'#f59e0b',3:'#a855f7'};   // colorOf の mode 色と一致
  const el = document.createElement('div'); el.className='glass';
  el.style.cssText='position:fixed;left:12px;bottom:70px;z-index:6;display:none;'
    +'padding:5px 10px;font-size:11px;white-space:nowrap;';
  let h='<b style="margin-right:5px">移動</b>';
  for(const k of Object.keys(LEG)) h+='<span style="margin-right:9px">'
    +'<span style="display:inline-block;width:9px;height:9px;border-radius:2px;background:'
    +(MC[k]||'#8a97a5')+';margin-right:3px"></span>'+LEG[k]+'</span>';
  el.innerHTML=h; document.body.appendChild(el);
  const cb=document.getElementById('colorBy');
  const upd=()=>{ el.style.display=(cb&&cb.value==='mode')?'block':'none'; };
  if(cb) cb.addEventListener('change', upd); upd();
})();
"""


# ============================================================ 地図ビューア
MAP_HTML = r"""<!DOCTYPE html>
<html lang="ja"><head><meta charset="utf-8">
<title>shibuya map viewer</title>
<style>
__BASE_CSS__
  body { height:100vh; overflow:hidden; }
  #wrap { position:absolute; inset:0; }
  canvas#cv { position:absolute; inset:0; width:100%; height:100%; cursor:grab; }
  /* 左上: タイトル/検索カード */
  #titleCard { position:absolute; top:14px; left:14px; padding:12px 15px; min-width:214px; font-size:12.5px; z-index:4; }
  #titleCard .tRow { display:flex; align-items:center; gap:11px; }
  #titleCard .logo { font-size:21px; line-height:1; }
  #titleCard .tName { font-size:15px; font-weight:600; letter-spacing:.01em; }
  #titleCard .tClock { color:var(--dim); font-size:12px; margin-top:1px; }
  #titleCard .tClock b { color:var(--ink); font-weight:600; }
  #titleCard .tStats { display:flex; gap:13px; margin-top:10px; color:var(--dim); font-size:12px; }
  #titleCard .tStats b { color:var(--ink); font-weight:600; }
  #titleCard .tTrain { margin-top:7px; color:var(--dim); font-size:11.5px; }
  /* 右上: レイヤー */
  #layers { position:absolute; top:14px; right:14px; padding:11px 13px; font-size:12px; z-index:4; max-height:76%; overflow-y:auto; min-width:160px; }
  #layers .hdr { font-size:10.5px; letter-spacing:.13em; color:var(--dim); margin:0 0 7px; cursor:pointer; text-transform:uppercase; }
  #layers label { display:flex; align-items:center; gap:7px; padding:2.5px 0; cursor:pointer; }
  #layers input[type=checkbox]{ accent-color:var(--accent); }
  #layers .op { display:flex; align-items:center; gap:6px; padding:4px 0 2px; color:var(--dim); }
  #layers .op input{ width:88px; }
  #attr { position:absolute; bottom:7px; left:14px; font-size:10px; color:var(--dim); z-index:3; opacity:.7; }
  #focusCard { position:absolute; left:14px; bottom:70px; width:302px; padding:13px 15px; font-size:12.5px; display:none; z-index:5; }
  #focusCard h3 { font-size:15px; margin-bottom:4px; }
  #focusCard .words b { color:var(--accent); }
  #focusCard .x { float:right; padding:3px 8px; }
  /* 下部中央: 細いタイムライン */
  #timeline { position:absolute; left:50%; bottom:16px; transform:translateX(-50%); width:min(52%,540px);
    padding:8px 16px; display:flex; align-items:center; gap:11px; z-index:4; }
  #timeline input[type=range]{ flex:1; height:4px; }
  #timeline .tlt { font-size:11px; color:var(--dim); min-width:66px; text-align:center; white-space:nowrap; }
  /* 右下: 再生・ズーム・表示コントロール */
  #dock { position:absolute; right:14px; bottom:16px; display:flex; align-items:center; gap:6px;
    padding:7px 9px; z-index:4; flex-wrap:wrap; justify-content:flex-end; max-width:min(90%,540px); }
  #dock button { width:34px; height:34px; padding:0; font-size:15px; border-radius:9px; display:flex; align-items:center; justify-content:center; }
  #dock #play { width:38px; height:38px; font-size:15px; background:var(--accent); color:#fff; }
  #dock #play:hover { filter:brightness(1.07); }
  #dock select { height:34px; }
  #dock .sp { font-size:10.5px; color:var(--dim); margin-right:2px; }
  /* フロア(間取り)モーダル */
  #floorModal { position:absolute; inset:0; background:rgba(10,14,20,.5); backdrop-filter:blur(4px);
    display:none; align-items:center; justify-content:center; z-index:9; }
  #floorBox { width:660px; max-width:94%; max-height:92%; overflow-y:auto; padding:18px; }
  #floorBox h3 { font-size:16px; }
  #floorBtns { display:flex; flex-wrap:wrap; gap:5px; margin:10px 0; }
  #floorBtns button { padding:5px 10px; font-size:12px; }
  #floorGuide { font-size:12px; color:var(--dim); margin:5px 0 5px; line-height:1.6; }
  #floorDisc { font-size:11px; color:var(--dim); opacity:.85; margin:0 0 6px; }
  canvas#floorCv { width:100%; height:340px; border-radius:12px; display:block; }
  #floorBox .fx { text-align:right; margin-top:12px; }
</style></head><body>
<div id="wrap">
  <canvas id="cv"></canvas>
  <div id="titleCard" class="glass">
    <div class="tRow"><span class="logo" id="logo">☀</span>
      <div><div class="tName">__RUN__</div>
        <div class="tClock">Day <b id="day">0</b> · <b id="clock">07:00</b> <span id="stepLabel"></span></div></div>
    </div>
    <div class="tStats"><span>🚶 <b id="nStreet">0</b></span><span>🏢 <b id="nIn">0</b></span>
      <span>🌐 <b id="nOut">0</b></span><span>💤 <b id="nSleep">0</b></span><span>🚗 <b id="nCar">0</b></span></div>
    <div class="tTrain" id="trainInfo"></div>
  </div>
  <div id="layers" class="glass"><div class="hdr" id="lyHdr">レイヤー ▾</div><div id="lyBody">
    <label><input type="checkbox" id="lyBase" checked> 背景地図(OSM)</label>
    <div class="op">不透明度 <input type="range" id="baseOp" min="0.15" max="1" step="0.05" value="0.9"></div>
    <label><input type="checkbox" id="lyRoad" checked> 道路</label>
    <label><input type="checkbox" id="lyUnder" checked> 地下通路・デッキ</label>
    <label><input type="checkbox" id="lyBld" checked> 建物</label>
    <label><input type="checkbox" id="lyPoi"> POI(店・会社)</label>
    <label><input type="checkbox" id="lyRail" checked> 線路</label>
    <label><input type="checkbox" id="lyTrain" checked> 電車(ダイヤ演出)</label>
    <label><input type="checkbox" id="lyCars" checked> 自動車(背景交通)</label>
    <label><input type="checkbox" id="lyFlow" checked> 人流(移動軌跡)</label>
    <label><input type="checkbox" id="lyHeat"> 人口密度ヒートマップ</label>
    <label><input type="checkbox" id="lyName" checked> 地名</label>
    <label><input type="checkbox" id="lyAgent" checked> エージェント</label>
  </div></div>
  <div id="attr">© OpenStreetMap contributors</div>
  <div id="focusCard" class="glass"></div>
  <div id="timeline" class="glass">
    <input type="range" id="seek" min="0" value="0" step="0.01"><span class="tlt" id="tlt"></span>
  </div>
  <div id="dock" class="glass">
    <button id="play" title="再生/一時停止">⏸</button>
    <select id="speed" title="再生速度: ×1 = 1秒に1step(シム内10分)。歩行の実速度は中央値1.1m/s=現実の渋谷と同水準で、速く見えるのは時間圧縮のため"><option value="0.5">×0.5</option><option value="1" selected>×1</option><option value="2">×2</option><option value="4">×4</option><option value="8">×8</option><option value="16">×16</option><option value="30">×30</option></select><span class="sp">step/s</span>
    <button id="zoomIn" title="拡大">＋</button><button id="zoomOut" title="縮小">－</button>
    <button id="resetView" title="全体表示">⤢</button>
    <select id="colorBy">
      <option value="id">個体色</option><option value="gender">性別</option>
      <option value="age">年齢</option><option value="occ">職業</option>
      <option value="mode">移動</option><option value="vocab">語彙</option><option value="vocabword">語彙(語別)</option>__COMMUNITY_OPTION__
    </select>
    <select id="floorFilter">
      <option value="all">階:すべて</option>
      <option value="street">路上のみ</option>
      <option value="inside">屋内すべて</option>
      <option value="1">1F</option><option value="2">2F</option>
      <option value="3">3F</option><option value="4">4F</option>
      <option value="5">5F</option><option value="6">6F</option>
      <option value="7">7F</option><option value="8">8F</option>
      <option value="9">9F+</option>
    </select>
    <button id="dashBtn" title="ダッシュボード" onclick="window.open('dashboard.html')">📊</button>
  </div>
  <div id="floorModal"><div id="floorBox" class="glass">
    <h3 id="floorTitle"></h3>
    <div id="floorGuide"></div>
    <div id="floorDisc">※ 間取りは建物IDから生成した推定(架空)です。実在建物の実際の内装は非公開のため(研究倫理R17)。</div>
    <div id="floorBtns"></div>
    <canvas id="floorCv"></canvas>
    <div class="fx"><button onclick="closeFloor()">閉じる</button></div>
  </div></div>
</div>
<script>
__ERR_JS__
const D = __DATA__;
__TIME_JS__
__THEME_JS__
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
const seek = document.getElementById('seek'); seek.max = D.nSteps - 1;
let cur = 0, playing = true, lastT = 0;   // 既定=再生中(有機的に動き続ける)
let focusId = null;
const L = id => document.getElementById(id).checked;
// 静止エージェントの微動(agent index を seed に決定論・連続時刻 t の純関数。乱数不使用)
function microXY(i, t){
  const a=i*12.9898, b=i*7.233;
  const dx=2.6*(Math.sin(t*0.55+a)*0.62 + Math.sin(t*0.223+a*1.7)*0.38);
  const dy=2.6*(Math.sin(t*0.50+b)*0.62 + Math.sin(t*0.190+b*1.3)*0.38);
  return [dx,dy];
}

// ---- カメラ ----
const xs=[], ys=[];
for (const e of D.edges) for (const p of e.g) { xs.push(p[0]); ys.push(p[1]); }
const B = {x0:Math.min(...xs), x1:Math.max(...xs), y0:Math.min(...ys), y1:Math.max(...ys)};
let cam = null;
function resetCam(){ const s=Math.min(cv.width/(B.x1-B.x0+100), cv.height/(B.y1-B.y0+100)); cam={cx:(B.x0+B.x1)/2, cy:(B.y0+B.y1)/2, s}; }
function fit(){ cv.width=cv.clientWidth*devicePixelRatio; cv.height=cv.clientHeight*devicePixelRatio; if(!cam) resetCam(); }
window.addEventListener('resize', fit);
const tf = (x,y)=>[(x-cam.cx)*cam.s+cv.width/2, cv.height/2-(y-cam.cy)*cam.s];
const inv = (px,py)=>[(px-cv.width/2)/cam.s+cam.cx, cam.cy-(py-cv.height/2)/cam.s];
cv.addEventListener('wheel', e=>{ e.preventDefault();
  const m=e.deltaY<0?1.15:1/1.15; const [wx,wy]=inv(e.offsetX*devicePixelRatio,e.offsetY*devicePixelRatio);
  cam.s*=m; cam.cx=wx-(wx-cam.cx)/m; cam.cy=wy-(wy-cam.cy)/m; });
let drag=null;
cv.addEventListener('mousedown', e=>{ drag={moved:false}; cv.style.cursor='grabbing'; });
window.addEventListener('mousemove', e=>{ if(!drag) return; cam.cx-=e.movementX*devicePixelRatio/cam.s; cam.cy+=e.movementY*devicePixelRatio/cam.s; if(Math.abs(e.movementX)+Math.abs(e.movementY)>1) drag.moved=true; });
window.addEventListener('mouseup', e=>{ if(drag && !drag.moved) clickAt(e); drag=null; cv.style.cursor='grab'; });
document.getElementById('resetView').onclick=()=>{ focusId=null; resetCam(); renderFocus(); };
function zoomBy(m){ cam.s*=m; }   // 画面中心を保ったまま拡大縮小
document.getElementById('zoomIn').onclick=()=>zoomBy(1.25);
document.getElementById('zoomOut').onclick=()=>zoomBy(1/1.25);
document.getElementById('lyHdr').onclick=()=>{ const b=document.getElementById('lyBody');
  const off=b.style.display==='none'; b.style.display=off?'block':'none';
  document.getElementById('lyHdr').textContent=off?'レイヤー ▾':'レイヤー ▸'; };

function passFloorFilter(w){
  const f=document.getElementById('floorFilter').value;
  if(f==='all') return true;
  if(f==='street') return w===0;
  if(f==='inside') return w>=1000;
  if(w<1000) return false;
  const wf=w%100, fl=Number(f);
  return fl===9? wf>=9 : wf===fl;
}

// ---- OSM タイル(失敗・未完タイルは4秒後に再試行、読み込み中は親ズームで穴埋め)----
const tileCache = {};
const LAT0 = D.origin? D.origin[0]:null, LON0 = D.origin? D.origin[1]:null;
function mercPx(lat,lon,z){ const W=256*Math.pow(2,z);
  const mx=(lon+180)/360*W;
  const s=Math.sin(lat*Math.PI/180);
  const my=(0.5-Math.log((1+s)/(1-s))/(4*Math.PI))*W;
  return [mx,my]; }
function drawBasemap(){
  if(!LAT0) return;
  ctx.save(); ctx.globalAlpha = Number(document.getElementById('baseOp').value) * (1 - themeAt(cur).k*0.55);
  const mPerPx = 1/cam.s;
  let z = Math.round(Math.log2(156543.03392*Math.cos(LAT0*Math.PI/180)/mPerPx));
  z = Math.max(13, Math.min(19, z));
  const ppm = 1/(156543.03392*Math.cos(LAT0*Math.PI/180)/Math.pow(2,z));
  const [mx0,my0] = mercPx(LAT0,LON0,z);
  const [wx0,wy0]=inv(0,0), [wx1,wy1]=inv(cv.width,cv.height);
  const mxa=mx0+Math.min(wx0,wx1)*ppm, mxb=mx0+Math.max(wx0,wx1)*ppm;
  const mya=my0-Math.max(wy0,wy1)*ppm, myb=my0-Math.min(wy0,wy1)*ppm;
  const t0x=Math.floor(mxa/256), t1x=Math.floor(mxb/256);
  const t0y=Math.floor(mya/256), t1y=Math.floor(myb/256);
  if((t1x-t0x+1)*(t1y-t0y+1)>64){ ctx.restore(); return; }
  for(let tx=t0x;tx<=t1x;tx++) for(let ty=t0y;ty<=t1y;ty++){
    const key=`${z}/${tx}/${ty}`;
    let tc = tileCache[key];
    if(!tc || (!tc.ok && performance.now()-tc.t > 4000)){
      tc = tileCache[key] = {img:null, ok:false, t:performance.now()};
      const img=new Image(); img.crossOrigin='anonymous';
      img.onload=()=>{ tc.img=img; tc.ok=true; };
      img.onerror=()=>{ tc.ok=false; tc.t=performance.now(); };
      img.src=`https://tile.openstreetmap.org/${z}/${tx}/${ty}.png`;
    }
    const lx=(tx*256-mx0)/ppm, ly=-(ty*256-my0)/ppm;
    const [sx,sy]=tf(lx,ly);
    const size=256/ppm*cam.s;
    if(!tc.ok){
      const pc=tileCache[`${z-1}/${tx>>1}/${ty>>1}`];
      if(pc && pc.ok) ctx.drawImage(pc.img,(tx%2+2)%2*128,(ty%2+2)%2*128,128,128,sx,sy,size+1,size+1);
      continue;
    }
    ctx.drawImage(tc.img, sx, sy, size+1, size+1);
  }
  ctx.restore();
}

// ---- 道なり移動の補間 ----
function pathLen(pts){ let l=0; for(let i=1;i<pts.length;i++) l+=Math.hypot(pts[i][0]-pts[i-1][0],pts[i][1]-pts[i-1][1]); return l; }
function alongPath(pts, f){
  const total=pathLen(pts); if(total===0) return pts[0];
  let target=total*f;
  for(let i=1;i<pts.length;i++){ const seg=Math.hypot(pts[i][0]-pts[i-1][0],pts[i][1]-pts[i-1][1]);
    if(target<=seg){ const g=seg? target/seg:0;
      return [pts[i-1][0]+(pts[i][0]-pts[i-1][0])*g, pts[i-1][1]+(pts[i][1]-pts[i-1][1])*g]; }
    target-=seg; }
  return pts[pts.length-1];
}
function posAt(t){ t=Math.max(0, Math.min(t, D.nSteps-1));   // step 索引を必ず有効域に(負値/範囲外での undefined 参照を防ぐ)
  const s0=Math.floor(t), f=t-s0;
  return D.ids.map((_,i)=>{
    const w=D.positions[s0][i][2];
    if(w!==0) return [...D.positions[s0][i]];
    const nm=(s0+1<D.nSteps)? D.moves[s0+1][i]:null;
    if(nm){ const p=alongPath(nm[1], f); return [p[0],p[1],0]; }
    const a=D.positions[s0][i], b=D.positions[Math.min(s0+1,D.nSteps-1)][i];
    if(b[2]!==0) return [a[0],a[1],0];
    return [a[0]+(b[0]-a[0])*f, a[1]+(b[1]-a[1])*f, 0]; }); }
function modeAt(s,i){ const m=D.moves[Math.min(s+1,D.nSteps-1)][i]; return m? m[0]:-1; }

const OCC_COLORS = {};__COMMUNITY_JS__
function agentWords(id, s0){ const out=[];
  for(const v of D.vocab){ if(v.creator===id && v.born<=s0) out.push(v.w);
    else if(v.adopts.some(x=>x[0]<=s0 && x[1]===id)) out.push(v.w); } return out; }
// ---- 語彙(語別)モード: 選択した1語の知識状態を s0 時点でキャッシュ(採用済/聴取済み未採用/未接触)。
// D.vocab のインデックスで語を指す(selector option の value)。idx/s0 が変わった時だけ再構築。
let _vwIdx=-1, _vwStep=-2, _vwAdopt=null, _vwHeard=null;
function vwState(s0){
  const sel=document.getElementById('vocabWordSel');
  const idx=sel? Number(sel.value) : -1;
  if(idx===_vwIdx && s0===_vwStep && _vwAdopt) return;
  _vwIdx=idx; _vwStep=s0; _vwAdopt=new Set(); _vwHeard=new Set();
  const v=D.vocab[idx]; if(!v) return;
  if(v.creator>=0) _vwAdopt.add(v.creator);
  for(const a of v.adopts){ if(a[0]<=s0) _vwAdopt.add(a[1]); }
  for(const e of (v.trans||[])){ if(e[0]<=s0) _vwHeard.add(e[2]); }
}
function colorOf(i, s0){
  const mode = document.getElementById('colorBy').value;
  const a = D.agents[i]||{};__COMMUNITY_HOOK__
  if(mode==='gender') return a.gender==='女'? '#f472b6' : a.gender==='男'? '#60a5fa':'#999';
  if(mode==='age'){ const g=a.age||0; return g<25?'#6ee7b7':g<40?'#ffd166':g<60?'#fb923c':'#a78bfa'; }
  if(mode==='occ'){ if(!(a.occupation in OCC_COLORS)){ let h=0; for(const c of a.occupation||'') h=(h*31+c.charCodeAt(0))%360; OCC_COLORS[a.occupation]=h; } return `hsl(${OCC_COLORS[a.occupation]} 65% 62%)`; }
  if(mode==='mode'){ const m=modeAt(s0,i); __MODE_TAXI__return m===2?'#f59e0b':m===1?'#6ee7b7':m===0?'#60a5fa':'#8a97a5'; }
  if(mode==='vocab'){ const n=agentWords(D.ids[i],s0).length; return n===0?'#5b6572':n<3?'#d9c26a':'#ffd166'; }
  if(mode==='vocabword'){ vwState(s0); const id=D.ids[i];
    if(_vwAdopt && _vwAdopt.has(id)) return '#ffd166';    // 採用済=濃
    if(_vwHeard && _vwHeard.has(id)) return '#8a7a2a';    // 聴取済み未採用=中間
    return '#3c4552'; }                                   // 未接触=灰
  return `hsl(${hue(i)} 70% 60%)`;
}

function trainsActive(t){ const m=Math.floor((D.startMin+t*10))%1440;
  return D.transit.filter(w=> w.b<1440? (m>=w.a&&m<=w.b):(m>=w.a||m<=w.b-1440)).length; }
const trainPaths = D.rails.filter(r=>r.k==='rail' && pathLen(r.g)>250).slice(0,10);

// ---- 運行路線(build_rail_lines 由来): 断片ではなく連結済みの1本を端から端へ ----
function _shibuyaRef(){   // 渋谷駅側の端点判定に使う基準点(POI「渋谷駅」優先→原点近傍→地図中心)
  const cand=(D.pois||[]).filter(p=>p.n && p.n.indexOf('渋谷駅')>=0);
  if(cand.length){ cand.sort((a,b)=>(a.x*a.x+a.y*a.y)-(b.x*b.x+b.y*b.y)); return [cand[0].x,cand[0].y]; }
  return [(B.x0+B.x1)/2,(B.y0+B.y1)/2];
}
const _SHIB=_shibuyaRef();
const RAILLINES=(D.railLines||[]).map(rl=>{
  const path=rl.path.map(p=>[p[0],p[1]]);
  // terminus は path[0] を渋谷端に揃える(折り返し点が必ず渋谷側の端点になるように)
  if(rl.mode==='terminus' && path.length>1){
    const d0=Math.hypot(path[0][0]-_SHIB[0],path[0][1]-_SHIB[1]);
    const d1=Math.hypot(path[path.length-1][0]-_SHIB[0],path[path.length-1][1]-_SHIB[1]);
    if(d1<d0) path.reverse();
  }
  const cum=[0];
  for(let i=1;i<path.length;i++) cum.push(cum[i-1]+Math.hypot(path[i][0]-path[i-1][0],path[i][1]-path[i-1][1]));
  return {name:rl.name, mode:rl.mode, first_min:rl.first_min, last_min:rl.last_min,
          headway_min:rl.headway_min, path, cum, len:cum[cum.length-1]};
}).filter(rl=>rl.len>0);
function _railActive(rl,m){ const a=rl.first_min,b=rl.last_min;
  return b<1440 ? (m>=a && m<=b) : (m>=a || m<=b-1440); }
function _railPtAt(rl,dLen){ const cum=rl.cum,path=rl.path,Ln=rl.len;
  if(dLen<=0) return path[0]; if(dLen>=Ln) return path[path.length-1];
  let lo=0,hi=cum.length-1;
  while(lo+1<hi){ const mid=(lo+hi)>>1; if(cum[mid]<=dLen) lo=mid; else hi=mid; }
  const seg=cum[hi]-cum[lo], g=seg? (dLen-cum[lo])/seg:0;
  return [path[lo][0]+(path[hi][0]-path[lo][0])*g, path[lo][1]+(path[hi][1]-path[lo][1])*g]; }
// 演出パラメータ(実速度ではなく可視性優先。位相は連続時刻 t の純関数=決定論・乱数不使用)
const _V_TRAIN=95;      // 見た目の速度: 単位t(=10分)あたりのメートル
const _DOT_SPACING=620; // path 上のドット間隔の目安(m)
const _MAX_DOTS=9;      // 1方向あたりの描画上限(超長距離路線の描画数を抑制)
function _drawTrain(rl, frac, travelSign, night, tail){
  const d=frac*rl.len; const [wx,wy]=_railPtAt(rl,d); const [x,y]=tf(wx,wy);
  if(x<-24||y<-24||x>cv.width+24||y>cv.height+24) return;   // 画面外はスキップ(尾ごと不可視)
  if(tail){ for(let s=1;s<=2;s++){
      let dd=d - travelSign*s*8*devicePixelRatio/cam.s; dd=Math.max(0,Math.min(rl.len,dd));
      const [tx0,ty0]=_railPtAt(rl,dd); const [tx,ty]=tf(tx0,ty0);
      ctx.fillStyle=`rgba(180,200,235,${(0.30-s*0.09).toFixed(2)})`;
      ctx.beginPath(); ctx.arc(tx,ty,2.4*devicePixelRatio,0,7); ctx.fill(); } }
  if(night>0.3){ ctx.save(); ctx.globalCompositeOperation='lighter';
    ctx.fillStyle=`rgba(255,240,190,${(night*0.5).toFixed(2)})`;
    ctx.beginPath(); ctx.arc(x,y,9*devicePixelRatio,0,7); ctx.fill(); ctx.restore(); }
  ctx.fillStyle='#e5e7eb'; ctx.strokeStyle='#111'; ctx.lineWidth=1.5*devicePixelRatio;
  const w=10*devicePixelRatio,h=5*devicePixelRatio;
  ctx.beginPath(); ctx.roundRect(x-w/2,y-h/2,w,h,2*devicePixelRatio); ctx.fill(); ctx.stroke();
}
const POI_COLOR = {food:'#f59e0b', nightlife:'#e879f9', shop:'#60a5fa', office:'#94a3b8',
                   service:'#6ee7b7', education:'#a78bfa', hotel:'#f472b6',
                   attraction:'#ffd166', leisure:'#4ade80'};
const ROAD = {primary:7,secondary:6,tertiary:5,unclassified:4,residential:4,living_street:3.5,service:2.5,pedestrian:3,footway:1.6,path:1.4,steps:1.2,cycleway:1.8,corridor:1.5,elevator:1.5};
const MODE_COLOR = ['#60a5fa','#6ee7b7','#f59e0b'];

function draw(){
  const t=cur, s0=Math.floor(t), f=t-s0;
  const TH=themeAt(t), night=TH.k;
  const base = L('lyBase') && LAT0;
  ctx.fillStyle = base? TH.mapBgC : TH.canvasBgC;
  ctx.fillRect(0,0,cv.width,cv.height);
  if(base){ drawBasemap();
    if(night>0){ ctx.fillStyle=TH.nightC; ctx.fillRect(0,0,cv.width,cv.height); } }
  ctx.lineCap='round'; ctx.lineJoin='round';
  // 表示座標(非移動中の人は agent_id 由来の微動を加える=常時ゆるやかに漂う)
  const pos=posAt(t);
  const walking=i=> pos[i][2]===0 && D.moves[Math.min(s0+1,D.nSteps-1)][i];
  const dispXY=i=>{ let x=pos[i][0], y=pos[i][1];
    if(!walking(i)){ const d=microXY(i,t); x+=d[0]; y+=d[1]; } return [x,y]; };
  if(L('lyRoad')){
    for(const e of D.edges){ if(e.l!==0) continue;
      const w=(ROAD[e.k]||2)*devicePixelRatio*Math.min(1.4,Math.max(.5,cam.s/1.2));
      ctx.strokeStyle=TH.roadC; ctx.lineWidth=w; ctx.beginPath();
      e.g.forEach((p,j)=>{ const [x,y]=tf(p[0],p[1]); j? ctx.lineTo(x,y):ctx.moveTo(x,y); }); ctx.stroke(); } }
  if(L('lyUnder')){
    for(const e of D.edges){ if(e.l===0) continue;
      const under = e.l<0;
      ctx.strokeStyle = under? TH.underC : TH.deckC;
      ctx.lineWidth=2.2*devicePixelRatio; ctx.setLineDash(under?[5,5]:[]);
      ctx.beginPath(); e.g.forEach((p,j)=>{ const [x,y]=tf(p[0],p[1]); j? ctx.lineTo(x,y):ctx.moveTo(x,y); }); ctx.stroke(); }
    ctx.setLineDash([]); }
  if(L('lyRail')){
    for(const r of D.rails){ ctx.strokeStyle= r.k==='subway'? TH.subwayC : TH.railC;
      ctx.lineWidth=3*devicePixelRatio; ctx.setLineDash(r.k==='subway'?[6,6]:[]);
      ctx.beginPath(); r.g.forEach((p,j)=>{ const [x,y]=tf(p[0],p[1]); j? ctx.lineTo(x,y):ctx.moveTo(x,y); }); ctx.stroke(); ctx.setLineDash([]); } }
  const occ={};
  pos.forEach(p=>{ if(p[2]>=1000){ const bi=Math.floor((p[2]-1000)/100); occ[bi]=(occ[bi]||0)+1; } });
  if(L('lyBld')){
    const small = cam.s<0.45;
    D.buildings.forEach((b,bi)=>{ if(small && !b.name && !occ[bi]) return;
      const [cxp,cyp]=tf(b.cx,b.cy); if(cxp<-60||cyp<-60||cxp>cv.width+60||cyp>cv.height+60) return;
      ctx.beginPath();
      b.fp.forEach((p,j)=>{ const [x,y]=tf(p[0],p[1]); j? ctx.lineTo(x,y):ctx.moveTo(x,y); });
      ctx.closePath();
      ctx.fillStyle= occ[bi]? TH.bldOccC : TH.bldC;
      ctx.fill(); ctx.strokeStyle=TH.bldStrokeC; ctx.lineWidth=1; ctx.stroke(); });
    // 環境の生命感: 在館中の建物はゆっくり明滅する暖色グロー(街が呼吸する程度)
    ctx.save(); ctx.globalCompositeOperation='lighter';
    D.buildings.forEach((b,bi)=>{ if(!occ[bi]) return; const [x,y]=tf(b.cx,b.cy);
      if(x<-40||y<-40||x>cv.width+40||y>cv.height+40) return;
      const pulse=0.5+0.5*Math.sin(t*1.4+bi), rr=Math.max(14,24*cam.s);
      const a=(0.05+0.06*pulse)*(0.55+night*0.8);
      const gr=ctx.createRadialGradient(x,y,0,x,y,rr);
      gr.addColorStop(0,`rgba(255,190,110,${a.toFixed(3)})`); gr.addColorStop(1,'rgba(255,190,110,0)');
      ctx.fillStyle=gr; ctx.beginPath(); ctx.arc(x,y,rr,0,7); ctx.fill(); });
    // 夜の街の灯り(名前付き建物が微かに瞬く)
    if(night>0.12){ for(let bi=0;bi<D.buildings.length;bi++){ const b=D.buildings[bi]; if(!b.name) continue;
      const [x,y]=tf(b.cx,b.cy); if(x<0||y<0||x>cv.width||y>cv.height) continue;
      const tw=0.5+0.5*Math.sin(t*2.0+bi*1.7); const a=night*(0.12+0.22*tw);
      ctx.fillStyle=`rgba(255,210,140,${a.toFixed(3)})`;
      ctx.beginPath(); ctx.arc(x,y,1.5*devicePixelRatio,0,7); ctx.fill(); } }
    ctx.restore();
    ctx.font=`${10*devicePixelRatio}px system-ui,sans-serif`;
    D.buildings.forEach((b,bi)=>{ if(!occ[bi]) return; const [x,y]=tf(b.cx,b.cy);
      ctx.fillStyle=TH.glowC; ctx.textAlign='center'; ctx.fillText(`▣${occ[bi]}`,x,y); }); }
  if(L('lyPoi') && cam.s>0.6){
    for(const p of D.pois){ const [x,y]=tf(p.x,p.y);
      if(x<-20||y<-20||x>cv.width+20||y>cv.height+20) continue;
      ctx.fillStyle=POI_COLOR[p.c]||'#9ca3af';
      ctx.beginPath(); ctx.arc(x,y,2.2*devicePixelRatio,0,7); ctx.fill(); }
    if(cam.s>2.2){ ctx.font=`${9.5*devicePixelRatio}px system-ui,sans-serif`; ctx.textAlign='left';
      let shown=0;
      for(const p of D.pois){ if(shown>140) break; const [x,y]=tf(p.x,p.y);
        if(x<0||y<0||x>cv.width||y>cv.height) continue; shown++;
        ctx.fillStyle=TH.poiC; ctx.fillText(p.n, x+4*devicePixelRatio, y+3*devicePixelRatio); } } }
  if(L('lyHeat')){
    ctx.save(); ctx.globalCompositeOperation='lighter';
    const r=55*cam.s;
    for(let i=0;i<pos.length;i++){ const p=pos[i]; if(p[2]===-1) continue;
      if(!passFloorFilter(p[2]) && p[2]!==-2) continue;
      const [wx,wy]=dispXY(i); const [x,y]=tf(wx,wy);
      if(x<-r||y<-r||x>cv.width+r||y>cv.height+r) continue;
      const g=ctx.createRadialGradient(x,y,0,x,y,r);
      g.addColorStop(0,'rgba(255,120,40,.16)'); g.addColorStop(1,'rgba(255,120,40,0)');
      ctx.fillStyle=g; ctx.beginPath(); ctx.arc(x,y,r,0,7); ctx.fill(); }
    ctx.restore(); }
  if(L('lyFlow')){
    for(let ds=2; ds>=0; ds--){ const s=s0-ds+1; if(s<0||s>=D.nSteps) continue;
      const alpha=[.5,.3,.15][ds];
      for(let i=0;i<D.ids.length;i++){ const m=D.moves[s][i]; if(!m) continue;
        ctx.strokeStyle=MODE_COLOR[m[0]]+Math.round(alpha*255).toString(16).padStart(2,'0');
        ctx.lineWidth=(m[0]===2?3.5:2.2)*devicePixelRatio;
        ctx.beginPath(); m[1].forEach((p,j)=>{ const [x,y]=tf(p[0],p[1]); j? ctx.lineTo(x,y):ctx.moveTo(x,y); }); ctx.stroke(); } } }
  const tr = D.traffic[Math.min(s0+1,D.nSteps-1)] || {n:0,segs:[]};
  if(L('lyCars')){
    for(let ci=0; ci<tr.segs.length; ci++){ const seg=tr.segs[ci];
      const off=(ci*0.618)%1;
      const g=(f+off)%1;
      const p=alongPath(seg, g); const [x,y]=tf(p[0],p[1]);
      if(x<-8||y<-8||x>cv.width+8||y>cv.height+8) continue;
      ctx.fillStyle= night>0.5? '#fbbf24':'#fde68a';
      ctx.strokeStyle='rgba(0,0,0,.5)'; ctx.lineWidth=0.8*devicePixelRatio;
      ctx.beginPath(); ctx.arc(x,y,2.6*devicePixelRatio,0,7); ctx.fill(); ctx.stroke(); } }
  // ---- 電車: 連結済み運行路線を走らせる(through=渋谷を跨いで通過 / terminus=渋谷端で折返し)----
  const nowM=((D.startMin+t*10)%1440+1440)%1440;
  let nAct=0;
  if(RAILLINES.length){ for(const rl of RAILLINES) if(_railActive(rl,nowM)) nAct++; }
  else nAct=trainsActive(t);
  if(L('lyTrain') && RAILLINES.length){
    for(const rl of RAILLINES){ if(!_railActive(rl,nowM)) continue;
      const period=rl.len/_V_TRAIN;                        // 全長走破に要する t(=見た目一定速度)
      const nd=Math.max(1, Math.min(_MAX_DOTS, Math.round(rl.len/_DOT_SPACING)));
      if(rl.mode==='through'){
        for(let dir=0; dir<2; dir++){ const sgn=dir? -1:1; // 上下2方向を headway 位相で連続スポーン
          for(let k=0;k<nd;k++){
            let ph=(t/period)+k/nd+dir*0.5/nd; ph=((ph%1)+1)%1;
            const frac=sgn>0? ph : 1-ph;
            _drawTrain(rl, frac, sgn, night, true); } }
      } else {   // terminus: 渋谷端(path[0])で折り返す往復(三角波: 0→1→0)
        for(let k=0;k<nd;k++){
          let ph=(t/period)+k/nd; ph=((ph%1)+1)%1;
          const frac=ph<0.5? ph*2 : 2-ph*2;
          const sgn=ph<0.5? 1 : -1;
          _drawTrain(rl, frac, sgn, night, false); }
      }
    }
  } else if(L('lyTrain') && nAct>0){   // railways の無い地図: 従来の断片往復に静かに劣化
    trainPaths.forEach((r,ri)=>{ const cyc=5+ri%3;
      const ph=((t/cyc)+ri*0.37)%2, ff=ph<1? ph:2-ph;
      const p=alongPath(r.g,ff); const [x,y]=tf(p[0],p[1]);
      if(night>0.3){ ctx.save(); ctx.globalCompositeOperation='lighter';
        ctx.fillStyle=`rgba(255,240,190,${(night*0.5).toFixed(2)})`;
        ctx.beginPath(); ctx.arc(x,y,9*devicePixelRatio,0,7); ctx.fill(); ctx.restore(); }
      ctx.fillStyle='#e5e7eb'; ctx.strokeStyle='#111'; ctx.lineWidth=1.5*devicePixelRatio;
      const w=10*devicePixelRatio, h=5*devicePixelRatio;
      ctx.beginPath(); ctx.roundRect(x-w/2,y-h/2,w,h,2*devicePixelRatio); ctx.fill(); ctx.stroke(); });
  }
  document.getElementById('trainInfo').textContent = (nAct>0? `🚃 運行中 ${nAct}路線`:'🚃 終電後') + ` · 🚗 この10分 ${tr.n}台`;
  if(L('lyName') && cam.s>0.55){ ctx.font=`${11*devicePixelRatio}px system-ui,sans-serif`; ctx.textAlign='center';
    for(const l of D.labels){ const [x,y]=tf(l.x,l.y);
      if(x<0||y<0||x>cv.width||y>cv.height) continue;
      ctx.fillStyle=TH.nameC; ctx.strokeStyle=TH.nameHaloC; ctx.lineWidth=3*devicePixelRatio;
      ctx.strokeText(l.n,x,y-8*devicePixelRatio); ctx.fillText(l.n,x,y-8*devicePixelRatio); } }
  if(focusId!==null){ const i=iOf[focusId];
    ctx.strokeStyle=`hsla(${hue(i)} 85% 55% / .75)`; ctx.lineWidth=2.5*devicePixelRatio;
    ctx.beginPath(); let started=false;
    for(let s=Math.max(1,s0-60); s<=s0; s++){ const m=D.moves[s][i]; if(!m){ continue; }
      for(let j=0;j<m[1].length;j++){ const [x,y]=tf(m[1][j][0],m[1][j][1]);
        started? ctx.lineTo(x,y):ctx.moveTo(x,y); started=true; } }
    ctx.stroke(); }
  if(L('lyAgent')){
    const bubbles = D.feed.filter(e=>e.s===s0 && (e.k==='speak'||e.k==='coin'));
    const ffAll = document.getElementById('floorFilter').value==='all';
    const vis=i=>{ const w=pos[i][2]; if(w===-1) return false;
      if(w>=1000) return !ffAll; if(!ffAll && !passFloorFilter(w) && w!==-2) return false; return true; };
    // 発話中エージェントの上品なパルス(先に描いてドットを上に載せる)
    for(const b of bubbles){ const i=iOf[b.a]; if(i===undefined||!vis(i)) continue;
      const [wx,wy]=dispXY(i); const [x,y]=tf(wx,wy);
      const ph=(t*1.6+i*0.11)%1, rr=(6+ph*18)*devicePixelRatio;
      ctx.strokeStyle=`rgba(${b.k==='coin'?'255,209,102':'120,180,250'},${((1-ph)*0.5).toFixed(3)})`;
      ctx.lineWidth=2*devicePixelRatio; ctx.beginPath(); ctx.arc(x,y,rr,0,7); ctx.stroke(); }
    for(let i=0;i<pos.length;i++){ const w=pos[i][2];
      if(w===-1) continue;
      if(!ffAll && !passFloorFilter(w) && w!==-2) continue;
      const [wx,wy]=dispXY(i); const [x,y]=tf(wx,wy);
      if(x<-14||y<-14||x>cv.width+14||y>cv.height+14) continue;
      const focused=(D.ids[i]===focusId);
      if(w===-2){ ctx.globalAlpha=.55; ctx.font=`${11*devicePixelRatio}px system-ui`;
        ctx.fillStyle=TH.dimC; ctx.textAlign='center';
        ctx.fillText('💤', x, y); ctx.globalAlpha=1; continue; }
      if(w>=1000){
        if(ffAll) continue;
        ctx.fillStyle=colorOf(i,s0);
        ctx.beginPath(); ctx.arc(x,y,5*devicePixelRatio,0,7); ctx.fill();
        ctx.strokeStyle='#ffd166'; ctx.lineWidth=1.5*devicePixelRatio; ctx.stroke();
        continue; }
      if(focused){ ctx.beginPath(); ctx.arc(x,y,12*devicePixelRatio,0,7);
        ctx.fillStyle=`hsla(${hue(i)} 85% 60% / .16)`; ctx.fill(); }
      ctx.fillStyle=colorOf(i,s0);
      ctx.beginPath(); ctx.arc(x,y,(focused?8:5.5)*devicePixelRatio,0,7); ctx.fill();
      ctx.strokeStyle= focused? (night>0.5?'#fff':'#1a73e8') : TH.agentStrokeC;
      ctx.lineWidth=(focused?2.4:1.2)*devicePixelRatio; ctx.stroke(); }
    ctx.font=`${12*devicePixelRatio}px system-ui,sans-serif`;
    for(const b of bubbles){ const i=iOf[b.a]; if(i===undefined) continue;
      const w=pos[i][2]; if(w===-1||w===-2) continue; if(w>=1000&&ffAll) continue;
      const [wx,wy]=dispXY(i); const [x,y]=tf(wx,wy);
      const text=b.t.length>26? b.t.slice(0,26)+'…':b.t;
      const tw=ctx.measureText(text).width+16*devicePixelRatio, bh=20*devicePixelRatio;
      ctx.fillStyle= b.k==='coin'? '#ffd166':TH.bubbleBgC;
      ctx.strokeStyle='rgba(0,0,0,.18)';
      ctx.beginPath(); ctx.roundRect(x-tw/2,y-34*devicePixelRatio,tw,bh,7*devicePixelRatio); ctx.fill(); ctx.stroke();
      ctx.fillStyle= b.k==='coin'? '#101418':TH.bubbleInkC; ctx.textAlign='center'; ctx.fillText(text,x,y-20*devicePixelRatio); } }
__INDOOR_HOOK__  if(focusId!==null){ const i=iOf[focusId]; const p=pos[i];
    if(p[2]!==-1){ cam.cx+=(p[0]-cam.cx)*.08; cam.cy+=(p[1]-cam.cy)*.08; } }
  let nSt=0,nIn=0,nOut=0,nSl=0;
  pos.forEach(p=>{ const w=p[2]; w===0?nSt++:w===-1?nOut++:w===-2?nSl++:nIn++; });
  document.getElementById('nStreet').textContent=nSt;
  document.getElementById('nIn').textContent=nIn;
  document.getElementById('nOut').textContent=nOut;
  document.getElementById('nSleep').textContent=nSl;
  document.getElementById('nCar').textContent=tr.n;
  document.getElementById('day').textContent=Math.floor((D.startMin+t*10)/1440);
  const mm=Math.floor(D.startMin+t*10)%1440;
  const hhmm=String(Math.floor(mm/60)).padStart(2,'0')+':'+String(mm%60).padStart(2,'0');
  document.getElementById('clock').textContent=hhmm;
  document.getElementById('stepLabel').textContent=`· ${s0}/${D.nSteps-1}`;
  document.getElementById('tlt').textContent=hhmm;
  document.getElementById('logo').textContent = night>0.5?'🌙':'☀';
}

function clickAt(e){
  if (e.target !== cv) return;
  const rect=cv.getBoundingClientRect();
  const px=(e.clientX-rect.left)*devicePixelRatio, py=(e.clientY-rect.top)*devicePixelRatio;
__INDOOR_CLICK__  const pos=posAt(cur); let best=null, bd=18*devicePixelRatio;
  for(let i=0;i<pos.length;i++){ if(pos[i][2]!==0&&pos[i][2]!==-2) continue;
    const [sx,sy]=tf(pos[i][0],pos[i][1]); const d=Math.hypot(sx-px,sy-py);
    if(d<bd){ bd=d; best=D.ids[i]; } }
  if(best===null){
    const [wx,wy]=inv(px,py);
    for(let bi=0; bi<D.buildings.length; bi++){ const b=D.buildings[bi];
      if(Math.abs(wx-b.cx)>150||Math.abs(wy-b.cy)>150) continue;
      let ins=false; const fp=b.fp;
      for(let i2=0,j2=fp.length-1;i2<fp.length;j2=i2++){
        if((fp[i2][1]>wy)!==(fp[j2][1]>wy) && wx<(fp[j2][0]-fp[i2][0])*(wy-fp[i2][1])/(fp[j2][1]-fp[i2][1])+fp[i2][0]) ins=!ins; }
      if(ins && (b.name||b.levels>3)){ openFloor(bi); return; } }
  }
  focusId = best; renderFocus();
}

function renderFocus(){ const el=document.getElementById('focusCard');
  if(focusId===null){ el.style.display='none'; return; }
  const meta=D.agents.find(a=>a.id===focusId)||{}; const i=iOf[focusId];
  const p=D.positions[Math.floor(cur)][i];
  let where='路上';
  if(p[2]===-1) where='範囲外(計算停止中)';
  else if(p[2]===-2) where='自宅で就寝中💤';
  else if(p[2]>=1000){ const bi=Math.floor((p[2]-1000)/100); where=`${D.buildings[bi].name||'ビル'} ${p[2]%100}階`; }
  const words=agentWords(focusId, Math.floor(cur));
  const speaks=D.feed.filter(e=>e.a===focusId&&e.k==='speak'&&e.s<=cur).slice(-3).reverse();
  el.style.display='block';
  el.innerHTML=`<button onclick="focusId=null;renderFocus()">✕</button>
    <h3><span style="color:${colorOf(i,Math.floor(cur))}">●</span> ${meta.name||'agent'+focusId}</h3>
    <div>${meta.gender||''} ${meta.age}歳・${meta.occupation} ${meta.has_car?'🚗':''}${meta.has_bicycle?'🚲':''}</div>
    ${meta.work_name? `<div style="color:var(--dim)">職場: ${meta.work_name}</div>`:''}
    <div style="color:var(--dim)">現在: ${where}</div>
    <div class="words" style="margin-top:4px">語彙: ${words.length? words.map(w=>`<b>「${w}」</b>`).join(' '):'まだ無し'}</div>
    <div style="margin-top:4px;color:var(--dim)">${speaks.map(s=>`「${s.t}」`).join('<br>')}</div>`;
}

// ---- フロアビュー(間取り生成+屋内エージェント)は共用モジュールを注入 ----
__FLOOR_JS____INDOOR_JS__

// 連続時刻 t を rAF で進める(既定=再生・終端で先頭へループ=街が生き続ける)
function loop(ts){
  if(playing){ const dt=Math.min(0.1,Math.max(0,(ts-lastT)/1000)), sp=Number(document.getElementById('speed').value);
    // Math.max(0,…): 初回フレームは RAF の ts が lastT(=init 時 performance.now)より前になりうる。
    // 負の dt を許すと cur<0 → posAt が D.positions[-1] を参照して例外→ループ停止→地図真っ白、を防ぐ。
    cur+=dt*sp; if(cur>=D.nSteps-1) cur=0; if(cur<0) cur=0;   // ループ再生(下限も明示クランプ)
    seek.value=cur; }
  lastT=ts; draw(); renderFocus();
  requestAnimationFrame(loop);
}
document.getElementById('play').onclick=()=>{ playing=!playing;
  document.getElementById('play').textContent=playing?'⏸':'▶'; };
seek.oninput=()=>{ cur=Number(seek.value); };
// ---- 語彙(語別)モードの語選択パネル(colorBy='vocabword' の時だけ表示)----
// 上位語を採用者数順にセレクタへ。選ぶと「その語を知っている人だけ」が地図で濃/中間/灰に。
// 時間スライダーで伝播が地図上に広がって見えるのが狙い(既存 mode の挙動は不変)。
(function(){
  if(!D.vocab || !D.vocab.length) return;
  const el=document.createElement('div'); el.className='glass'; el.id='vocabWordPanel';
  el.style.cssText='position:fixed;left:12px;bottom:70px;z-index:6;display:none;padding:6px 10px;font-size:11px;max-width:270px;';
  const top=D.vocab.map((v,idx)=>({idx,v,n:v.adopts.length})).sort((a,b)=>b.n-a.n).slice(0,80);
  const opts=top.map(o=>`<option value="${o.idx}">${o.v.w}(${o.n}人)</option>`).join('');
  el.innerHTML='<b style="margin-right:5px">語彙(語別)</b><select id="vocabWordSel">'+opts+'</select>'
    +'<div style="margin-top:4px;color:var(--dim)"><span style="color:#ffd166">●</span>採用 '
    +'<span style="color:#8a7a2a">●</span>聴取のみ <span style="color:#3c4552">●</span>未接触</div>';
  document.body.appendChild(el);
  const cb=document.getElementById('colorBy');
  const upd=()=>{ el.style.display=(cb&&cb.value==='vocabword')?'block':'none'; };
  if(cb) cb.addEventListener('change', upd); upd();
})();
__MODE_LEGEND_JS__fit(); lastT=performance.now(); requestAnimationFrame(loop);
</script></body></html>
"""


# ============================================================ 観測レンズの描画(第50バッチ)
# lens_map.json(価値/欲望)or L3 status(信用)が有る時だけ main() が dashboard に注入する追加 JS。
# 無ければ空文字→ DASH_HTML はバイト同一(後方互換。communities/mode_legend と同型)。
_LENS_JS = r"""
// ---- T1 価値4軸(values.TAGS を正準軸に再利用)----
const VAX_LABEL={utility:'実用',emotion:'感情',social:'社会',epistemic:'認識'};
const VAX_COLOR={utility:'#60a5fa',emotion:'#f472b6',social:'#6ee7b7',epistemic:'#ffd166'};
function renderValue(s0){
  const L=D.lens.value, days=L.days;
  const tot=days.map((d,i)=>L.axes.reduce((s,ax)=>s+L.series[ax][i],0));
  body.innerHTML=`
   <div class="chartBox"><h3>① 価値4軸の構成比(日別)</h3>
     <div class="sub">当日のイベントを価値の4軸(実用/感情/社会/認識)へ振り分けた構成比。第17バッチ values.TAGS を正準軸に再利用</div><canvas id="lvc1"></canvas></div>
   <div class="chartBox"><h3>② 実用充足 → 感情/社会シフト(近似指標)</h3>
     <div class="sub">日別の「実用」比率と「感情+社会」比率の推移。実用が満たされるほど感情・社会へ重心が移る仮説の近似観測</div><canvas id="lvc2"></canvas></div>
   <div class="chartBox"><h3>③ 住民別の価値プロファイル(上位)</h3>
     <div class="sub">各住民のイベントを4軸で構成比表示(帯の長さ=総数)。誰がどの価値に厚いか</div>
     <div id="lvProf"></div></div>`;
  const X=d=>d*144;
  lineChart(document.getElementById('lvc1'),
    L.axes.map(ax=>({label:VAX_LABEL[ax]||ax,color:VAX_COLOR[ax]||'#999',
      data:days.map((d,i)=>[X(d), tot[i]? L.series[ax][i]/tot[i]:0])})), {pct:true, ymax:1});
  lineChart(document.getElementById('lvc2'), [
    {label:'実用',color:VAX_COLOR.utility, data:days.map((d,i)=>[X(d), tot[i]? L.series.utility[i]/tot[i]:0])},
    {label:'感情+社会',color:'#f472b6', data:days.map((d,i)=>[X(d), tot[i]? (L.series.emotion[i]+L.series.social[i])/tot[i]:0])},
  ], {pct:true, ymax:1});
  document.getElementById('lvProf').innerHTML = L.profiles.map(p=>{
    const bars=L.axes.map(ax=>{ const w=p.total? p.counts[ax]/p.total*100:0;
      return `<span style="display:inline-block;height:12px;width:${w.toFixed(2)}%;background:${VAX_COLOR[ax]||'#999'}" title="${VAX_LABEL[ax]||ax} ${p.counts[ax]}"></span>`;}).join('');
    return `<div style="margin:4px 0;font-size:12px"><b>${p.name}</b> <span style="color:var(--dim)">${p.total}件</span><br>
      <span style="display:inline-flex;width:100%;max-width:520px;border-radius:5px;overflow:hidden;background:var(--surface2)">${bars}</span></div>`;
  }).join('') || '<div class="ev">価値を帯びたイベントがまだありません</div>';
}

// ---- T2 3M欲望 ----
const MOT_LABEL={earn:'儲け(earn)',love:'モテ(love)',recognition:'承認(recognition)'};
const MOT_COLOR={earn:'#4ade80',love:'#f472b6',recognition:'#a78bfa'};
function renderMotive(s0){
  const M=D.lens.motives, days=M.days;
  body.innerHTML=`
   <div class="chartBox"><h3>① 3M欲望の活性度(日別)</h3>
     <div class="sub">儲けたい(earn)/モテたい(love)/認められたい(recognition)を帯びたイベント数の推移</div><canvas id="lmc1"></canvas></div>
   <div class="chartBox"><h3>② 欲望の遷移(同一住民の連続イベントの軸ペア)</h3>
     <div class="sub">ある欲望の直後に来る欲望。線の太さ=遷移回数(円弧=同じ欲望が続く自己ループ)。関係タブの canvas 流儀を再利用</div>
     <canvas id="lmNet" style="height:300px"></canvas>
     <div id="lmList" style="font-size:12px;color:var(--dim);margin-top:6px"></div></div>`;
  const X=d=>d*144;
  lineChart(document.getElementById('lmc1'),
    M.motives.map(mt=>({label:MOT_LABEL[mt]||mt,color:MOT_COLOR[mt]||'#999',
      data:days.map((d,i)=>[X(d), M.series[mt][i]])})), {});
  drawMotiveNet(document.getElementById('lmNet'), M);
  document.getElementById('lmList').innerHTML = M.transitions.slice(0,10)
    .map(t=>`${MOT_LABEL[t.from]||t.from} → ${MOT_LABEL[t.to]||t.to}: <b>${t.n}</b>`).join(' ・ ') || '遷移データなし';
}
function drawMotiveNet(c, M){
  if(!c) return; c.width=c.clientWidth*devicePixelRatio; c.height=c.clientHeight*devicePixelRatio;
  const g=c.getContext('2d'), W=c.width, H=c.height, cx=W/2, cy=H/2, R=Math.min(W,H)*0.30;
  g.clearRect(0,0,W,H);
  const n=M.motives.length, pos={};
  M.motives.forEach((mt,i)=>{ const ang=-Math.PI/2 + i*2*Math.PI/n;
    pos[mt]=[cx+Math.cos(ang)*R, cy+Math.sin(ang)*R]; });
  const maxN=Math.max(1, ...M.transitions.map(t=>t.n));
  for(const t of M.transitions){ const P=pos[t.from], Q=pos[t.to]; if(!P||!Q) continue;
    g.strokeStyle=(MOT_COLOR[t.from]||'#999')+'99'; g.lineWidth=(1+5*t.n/maxN)*devicePixelRatio;
    if(t.from===t.to){ g.beginPath(); g.arc(P[0], P[1]-18*devicePixelRatio, 12*devicePixelRatio, 0, 7); g.stroke(); }
    else { g.beginPath(); g.moveTo(P[0],P[1]); g.lineTo(Q[0],Q[1]); g.stroke(); } }
  const T=themeAt(cur), ink=_css(T.ink);
  g.font=`${12*devicePixelRatio}px system-ui`; g.textAlign='center';
  for(const mt of M.motives){ const P=pos[mt];
    g.fillStyle=MOT_COLOR[mt]||'#999'; g.beginPath(); g.arc(P[0],P[1],10*devicePixelRatio,0,7); g.fill();
    g.fillStyle=ink; g.fillText(MOT_LABEL[mt]||mt, P[0], P[1]-16*devicePixelRatio); }
}

// ---- T6 信用内訳(status.py の合成地位スコアを可視化するだけ)----
const MAT_LABEL={rep:'評判',wealth:'資産',inst:'制度実績',biz:'商い',host:'主催',followers:'フォロワー'};
const MAT_COLOR={rep:'#60a5fa',wealth:'#ffd166',inst:'#a78bfa',biz:'#fb923c',host:'#6ee7b7',followers:'#38bdf8'};
function renderTrust(s0){
  const T=D.trust, mats=T.materials;
  const rows=T.agents.map(a=>{
    const bar=mats.map(m=>{ const w=a.contrib[m]*100;
      return `<span style="display:inline-block;height:14px;width:${w.toFixed(2)}%;background:${MAT_COLOR[m]||'#999'}" title="${MAT_LABEL[m]||m} 寄与${a.contrib[m].toFixed(3)}(百分位${a.pct[m]})"></span>`;}).join('');
    return `<div style="margin:5px 0;font-size:12px"><b>${a.name}</b> ${a.role?`<span style="color:var(--dim)">${a.role}</span>`:''} <span style="color:var(--dim)">status ${a.status}</span><br>
      <span style="display:inline-flex;width:100%;max-width:520px;border-radius:5px;overflow:hidden;background:var(--surface2)">${bar}</span></div>`;
  }).join('');
  const legend=mats.map(m=>`<span style="font-size:11px;color:var(--dim);margin-right:10px"><i style="display:inline-block;width:10px;height:10px;border-radius:2px;background:${MAT_COLOR[m]||'#999'};margin-right:3px;vertical-align:middle"></i>${MAT_LABEL[m]||m}(重み${T.weights[m]})</span>`).join('');
  const org=(T.org||[]).length? `<div class="chartBox"><h3>③ 組織内地位(org_role)と信用の相関</h3>
     <div class="sub">役割ごとの平均 status(agents.json に org_role がある時のみ)</div>${
     T.org.map(o=>{ const w=(o.mean*100).toFixed(1);
       return `<div style="font-size:12px;margin:3px 0">${o.role} <span style="color:var(--dim)">(${o.n}人)</span>
         <span style="display:inline-block;height:12px;width:${w}%;max-width:380px;background:#60a5fa;vertical-align:middle;border-radius:3px"></span> ${o.mean}</div>`;}).join('')}</div>`:'';
  body.innerHTML=`
   <div class="chartBox"><h3>① 信用(合成地位)の分布</h3>
     <div class="sub">Gini=集中度(0平等〜1独占)/ 上位10%集中度=winner-take-all。status.py 第11バッチの合成スコアの分布</div>
     <div style="font-size:20px;font-weight:700">Gini ${T.gini}<span style="font-size:13px;color:var(--dim);margin-left:14px">上位10%が全体の ${(T.top10*100).toFixed(1)}% を占有</span></div></div>
   <div class="chartBox"><h3>② 上位者の信用内訳(材料別の寄与)</h3>
     <div class="sub">${legend}</div>
     <div style="font-size:11px;color:#e8a33d;margin-bottom:6px">${T.note}</div>
     ${rows||'<div class="ev">status データがありません</div>'}</div>
   ${org}`;
}

function lensRender(tab, s0){
  if(tab==='value' && D.lens){ renderValue(s0); }
  else if(tab==='motive' && D.lens){ renderMotive(s0); }
  else if(tab==='trust' && D.trust){ renderTrust(s0); }
}
"""


# 第55バッチ ペルソナ逸脱率タブ(deviation_map.json が有る時だけ main() が dashboard に注入)。
# 無ければ空文字→ DASH_HTML はバイト同一(lens/trust と同型の後方互換)。devRender は tab で自己ガード。
_DEV_JS = r"""
const DEV_CAT_LABEL={food:'飲食',nightlife:'ナイトライフ',shop:'買い物',leisure:'レジャー',education:'学び'};
const DEV_CAT_COLOR={food:'#f59e0b',nightlife:'#a78bfa',shop:'#f472b6',leisure:'#34d399',education:'#60a5fa'};
function renderDeviation(s0){
  const V=D.deviation, days=V.days;
  const lastDisc = V.disc_series.length? V.disc_series[V.disc_series.length-1]:0;
  const lastFull = V.full_series.length? V.full_series[V.full_series.length-1]:0;
  // ① 分布ヒスト(住民別ラン累計の裁量逸脱率を 10 ビン)
  const hmax=Math.max(1,...V.hist);
  const hbars=V.hist.map((c,i)=>{ const h=c/hmax*100;
    return `<div style="display:flex;flex-direction:column;align-items:center;flex:1">
      <div style="font-size:10px;color:var(--dim)">${c||''}</div>
      <div style="width:70%;height:${h.toFixed(1)}px;min-height:1px;background:#e8a33d;border-radius:2px 2px 0 0"></div>
      <div style="font-size:9px;color:var(--dim);margin-top:2px">${i*10}</div></div>`;}).join('');
  // ② 最逸脱者ドリルダウン(ペルソナ属性 vs 実際の行動構成)
  const rows=V.top.map(a=>{
    const tot=Object.values(a.actual).reduce((s,x)=>s+x,0)||1;
    const bar=Object.entries(a.actual).map(([c,n])=>{ const w=n/tot*100;
      const ok=a.expect_hobby.includes(c);
      return `<span style="display:inline-block;height:14px;width:${w.toFixed(2)}%;background:${DEV_CAT_COLOR[c]||'#999'};${ok?'':'outline:2px solid #ef4444;outline-offset:-2px'}" title="${DEV_CAT_LABEL[c]||c} ${n}件${ok?'(期待どおり)':'(逸脱)'}"></span>`;}).join('');
    const chips=a.expect_hobby.map(c=>`<span style="font-size:10px;padding:1px 6px;border-radius:8px;background:${(DEV_CAT_COLOR[c]||'#999')}33;color:${DEV_CAT_COLOR[c]||'#999'};margin-right:3px">${DEV_CAT_LABEL[c]||c}</span>`).join('');
    return `<div style="margin:7px 0;font-size:12px">
      <b>${a.name}</b> <span style="color:var(--dim)">${a.occupation}</span>
      <span style="color:#e8a33d;font-weight:700;margin-left:6px">裁量逸脱 ${(a.disc_ratio*100).toFixed(0)}%</span>
      <span style="color:var(--dim);margin-left:6px">(全時間 ${(a.full_ratio*100).toFixed(0)}% ・ ${a.disc_dev}/${a.disc_total}件)</span><br>
      <span style="color:var(--dim);font-size:11px">期待の行き先: </span>${chips||'<span style="color:var(--dim);font-size:11px">なし</span>'}
      <span style="color:var(--dim);font-size:11px;margin-left:6px">${a.expect_work?('職場カテゴリ '+a.expect_work):''}</span><br>
      <span style="color:var(--dim);font-size:11px">実際の裁量行動: </span>
      <span style="display:inline-flex;width:100%;max-width:520px;border-radius:5px;overflow:hidden;background:var(--surface2);vertical-align:middle">${bar}</span></div>`;
  }).join('') || '<div class="ev">裁量行動の記録がまだありません</div>';
  body.innerHTML=`
   <div class="chartBox"><h3>① 裁量逸脱率の分布(住民別・ラン累計)</h3>
     <div class="sub">各住民の「裁量時間の行動のうちペルソナ期待(構造化趣味)と不一致の割合」の分布(横軸=逸脱率%・縦軸=人数)。測定対象 ${V.n_measured} 人。逸脱がゼロ寄りなら従順=シミュレーションの限界を語るデータ</div>
     <div style="display:flex;align-items:flex-end;gap:2px;height:130px;max-width:560px;margin-top:8px">${hbars}</div></div>
   <div class="chartBox"><h3>② 逸脱率の日別推移(裁量 vs 全時間)</h3>
     <div class="sub">裁量限定(主指標)と全時間(参考)の日別平均逸脱率。全時間は義務ルーチン=勤務行動を含むため構造的に低く出る(従順度の水増し)。両者の差が「義務による水増し」の大きさ</div>
     <div style="font-size:13px">直近: 裁量 <b style="color:#e8a33d">${(lastDisc*100).toFixed(1)}%</b> ・ 全時間 <b style="color:#60a5fa">${(lastFull*100).toFixed(1)}%</b></div>
     <canvas id="dvc1"></canvas></div>
   <div class="chartBox"><h3>③ 最も逸脱した住民(ペルソナ属性 vs 実際の行動構成)</h3>
     <div class="sub">裁量逸脱率の高い順。帯=実際の裁量行動のカテゴリ構成(赤枠=期待外=逸脱)。閾値 ${(V.top_threshold*100).toFixed(0)}% 以上が「上位逸脱者」</div>
     ${rows}</div>`;
  const X=d=>d*144;
  lineChart(document.getElementById('dvc1'), [
    {label:'裁量逸脱率',color:'#e8a33d', data:days.map((d,i)=>[X(d), V.disc_series[i]])},
    {label:'全時間逸脱率',color:'#60a5fa', data:days.map((d,i)=>[X(d), V.full_series[i]])},
  ], {pct:true, ymax:1});
}
function devRender(tab, s0){ if(tab==='deviation' && D.deviation){ renderDeviation(s0); } }
"""


# 第56バッチ タスクB 社会構造タブ(structure.json が有る時だけ main() が dashboard に注入)。
# 無ければ空文字→ DASH_HTML はバイト同一(lens/deviation と同型の後方互換)。structRender は tab で自己ガード。
_STRUCT_JS = r"""
function _stBands(container, intervals, nDays, color, label){
  // 固着区間を日軸トラック上の帯として描く(観測記録=介入ではない旨は sub に明記)。
  if(!intervals || !intervals.length){ container.innerHTML=
    `<span style="color:var(--dim);font-size:11px">${label}: 検出なし</span>`; return; }
  const denom=Math.max(1,nDays);
  const bars=intervals.map(s=>{ const L=s.start_day/denom*100, W=Math.max(1.5,(s.len)/denom*100);
    return `<div title="Day ${s.start_day}–${s.end_day}(${s.len}日)" style="position:absolute;left:${L.toFixed(2)}%;width:${W.toFixed(2)}%;top:0;bottom:0;background:${color};opacity:.55;border-radius:3px"></div>`;}).join('');
  container.innerHTML=`<div style="font-size:11px;color:var(--dim);margin-bottom:2px">${label}</div>
    <div style="position:relative;height:16px;background:var(--surface2);border-radius:3px">${bars}</div>`;
}
function renderStructure(s0){
  const V=D.structure, days=V.days||[], nD=days.length;
  const ch=V.churn||{}, rk=V.rank||{}, ce=V.centrality||{}, co=V.community||{}, st=V.stagnation||{};
  const X=d=>d*144;
  const pack=(arr)=> days.map((d,i)=>[X(d), arr&&arr[i]!=null?arr[i]:null]).filter(p=>p[1]!=null);
  const lg = st.longest;
  const hero = lg
    ? `最長の構造固着: <b style="color:#f87171">Day ${lg.start_day}–${lg.end_day}(${lg.len}日)</b> ・ 固着延べ <b>${st.total_stagnant_days||0}</b> 日`
    : `<b style="color:#6ee7b7">構造固着区間なし</b>(上位中心性が ${st.min_days||3} 日以上入れ替わらない期間は検出されず=構造は動いている)`;
  const srcJP = {status:'合成地位(L3)', reputation:'評判(L1)', none:'なし'}[rk.source]||rk.source;
  body.innerHTML=`
   <div class="chartBox"><h3>■ 構造固着の検知(観測記録・介入しない)</h3>
     <div class="sub">「強制トリガーなしで社会構造は変化するのか、固着するのか」を介入せず観測。固着=上位中心性が入れ替わらない/紐帯が組み替わらない/順位が入れ替わらない期間(閾値以下で連続 ${st.min_days||3} 日)。アラートではなく観測記録</div>
     <div style="font-size:13px;margin:6px 0">${hero}</div>
     <div id="stBand1" style="margin:6px 0"></div>
     <div id="stBand2" style="margin:6px 0"></div>
     <div id="stBand3" style="margin:6px 0"></div></div>
   <div class="chartBox"><h3>① edge 組み替え(日次の件数)</h3>
     <div class="sub">形成=新規に紐帯 tier≥1 到達 / 断絶=能動的な負交流 / 風化=長期不在の decay。日常の小さな積み重ねで関係が組み替わるか</div>
     <canvas id="stc1"></canvas></div>
   <div class="chartBox"><h3>② 変化率(組み替え率・中心性入れ替わり・コミュニティ変化)</h3>
     <div class="sub">組み替え率=churn/活性関係数。中心性turnover=会話グラフ上位${ce.top_k||10}の入れ替わり率。コミュ変化率=所属クラスタの Jaccard 変化。いずれも低いまま推移=固着</div>
     <canvas id="stc2"></canvas></div>
   <div class="chartBox"><h3>③ 順位固着(ランキングの前日比・前週比 Kendall τ)</h3>
     <div class="sub">順位ソース: <b>${srcJP}</b>。τ が高い(1 に近い)=順位が入れ替わらない=ヒエラルキー固着。低下・負=順位の流動</div>
     <canvas id="stc3"></canvas></div>
   <div class="chartBox"><h3>④ 上位者の順位推移(レースチャート)</h3>
     <div class="sub">いずれかの日に上位${rk.top_k||10}入りした住民の日次順位(上=上位)。線が水平=順位固着 / 交差=入れ替わり</div>
     <canvas id="strace"></canvas></div>`;
  _stBands(document.getElementById('stBand1'), (st.by_signal||{}).centrality_churn, nD, '#f87171', '中心性 turnover 低(上位が入れ替わらない)');
  _stBands(document.getElementById('stBand2'), (st.by_signal||{}).edge_churn, nD, '#fbbf24', 'edge churn 低(紐帯が組み替わらない)');
  _stBands(document.getElementById('stBand3'), (st.by_signal||{}).rank_tau, nD, '#a78bfa', '前日比τ 高(順位が入れ替わらない)');
  lineChart(document.getElementById('stc1'), [
    {label:'形成',color:'#6ee7b7', data:pack(ch.edges_formed)},
    {label:'断絶',color:'#f87171', data:pack(ch.edges_broken)},
    {label:'風化',color:'#fbbf24', data:pack(ch.edges_decayed)},
  ], {});
  lineChart(document.getElementById('stc2'), [
    {label:'組み替え率',color:'#60a5fa', data:pack(ch.churn_rate)},
    {label:'中心性turnover',color:'#f472b6', data:pack(ce.turnover)},
    {label:'コミュ変化率',color:'#34d399', data:pack(co.change_rate)},
  ], {pct:true});
  lineChart(document.getElementById('stc3'), [
    {label:'前日比τ',color:'#a78bfa', data:pack(rk.tau_prev_day)},
    {label:'前週比τ',color:'#fb923c', data:pack(rk.tau_prev_week)},
  ], {ymax:1});
  // レースチャート: 順位は反転(上=上位)。ymax=K+1 で 1 位が上端付近。
  const K=rk.top_k||10;
  const race=(rk.race||[]).slice(0,K).map((r,ix)=>({
    label:r.name, color:PAL[ix%PAL.length],
    data: days.map((d,i)=>[X(d), r.ranks[i]!=null?(K+1-r.ranks[i]):null]).filter(p=>p[1]!=null),
  }));
  lineChart(document.getElementById('strace'), race, {ymax:K+1});
}
function structRender(tab, s0){ if(tab==='structure' && D.structure){ renderStructure(s0); } }
"""


# ============================================================ ダッシュボード
DASH_HTML = r"""<!DOCTYPE html>
<html lang="ja"><head><meta charset="utf-8">
<title>shibuya dashboard</title>
<style>
__BASE_CSS__
  body { display:flex; flex-direction:column; height:100vh; overflow:hidden; }
  #top { display:flex; gap:10px; align-items:center; padding:10px 16px; background:var(--panel2); border-bottom:1px solid var(--line); flex-wrap:wrap; transition:background .8s; }
  #top h1 { font-size:15px; font-weight:600; margin-right:6px; }
  #top .clock { color:var(--accent); font-weight:600; min-width:110px; }
  #top input[type=range]{ flex:1; min-width:200px; }
  #main { flex:1; display:flex; min-height:0; }
  #nav { width:150px; background:var(--panel2); border-right:1px solid var(--line); padding:10px 8px; display:flex; flex-direction:column; gap:4px; transition:background .8s; }
  #nav button { text-align:left; width:100%; font-size:13px; padding:9px 12px; border-radius:10px; background:transparent; }
  #nav button:hover { background:var(--btn); }
  #nav button.on { background:var(--on); color:var(--onInk); }
  #body { flex:1; overflow-y:auto; padding:16px 20px; min-width:0; }
  .ev { font-size:12.5px; padding:7px 10px; margin:4px 0; border-radius:8px; background:var(--surface2); }
  .ev.coin { border-left:3px solid var(--accent); }
  .ev.adopt { border-left:3px solid #34a853; }
  .ev.exit,.ev.enter { border-left:3px solid var(--blue); }
  .ev.news { border-left:3px solid #ea4335; }
  .ev.post { border-left:3px solid var(--blue); }
  .ev .who { color:var(--dim); cursor:pointer; }
  /* --- X 風タイムライン --- */
  .tw { display:flex; gap:10px; padding:12px 14px; border-bottom:1px solid var(--line); max-width:620px; }
  .tw:hover { background:var(--surface2); }
  .tw .av { width:42px; height:42px; border-radius:50%; flex-shrink:0; display:flex; align-items:center; justify-content:center; font-size:18px; font-weight:700; color:#fff; }
  .tw .bd { min-width:0; }
  .tw .hd { font-size:13.5px; }
  .tw .hd b { font-weight:700; }
  .tw .hd .id { color:var(--dim); font-weight:400; font-size:12.5px; }
  .tw .tx { font-size:14px; line-height:1.5; margin-top:2px; white-space:pre-wrap; word-break:break-word; }
  .tw .tag { color:var(--blue); }
  .tw .badge { background:#ea4335; color:#fff; font-size:10px; border-radius:5px; padding:1px 5px; margin-left:4px; vertical-align:middle; }
  .tw .ft { color:var(--dim); font-size:11.5px; margin-top:6px; display:flex; gap:18px; }
  /* --- LINE 風 DM --- */
  #dmWrap { display:flex; gap:12px; height:calc(100vh - 190px); }
  #dmList { width:230px; overflow-y:auto; border-right:1px solid var(--line); padding-right:8px; }
  .conv { display:flex; gap:8px; align-items:center; padding:8px; border-radius:10px; cursor:pointer; }
  .conv:hover, .conv.on { background:var(--btn); }
  .conv .av { width:34px; height:34px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-weight:700; color:#fff; flex-shrink:0; }
  .conv .nm { font-size:12.5px; }
  .conv .lm { font-size:11px; color:var(--dim); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:150px; }
  #dmThread { flex:1; overflow-y:auto; background:var(--surface2); border:1px solid var(--line); border-radius:14px; padding:14px; }
  .bub { max-width:70%; margin:6px 0; padding:8px 12px; border-radius:16px; font-size:13px; line-height:1.5; width:fit-content; }
  .bub.left { background:var(--bubbleIn); border-top-left-radius:4px; }
  .bub.right { background:var(--bubbleOut); color:#fff; border-top-right-radius:4px; margin-left:auto; }
  .bub .meta { display:block; font-size:10px; opacity:.65; margin-top:3px; }
  /* --- 検索(SERP 風) --- */
  .serp { max-width:640px; background:var(--surface); border:1px solid var(--line); border-radius:14px; padding:14px 16px; margin:10px 0; }
  .serp .qbox { display:flex; align-items:center; gap:8px; background:#fff; color:#202124; border-radius:22px; padding:8px 16px; font-size:14px; box-shadow:0 1px 3px rgba(0,0,0,.12); }
  .serp .meta { color:var(--dim); font-size:11.5px; margin:6px 2px; }
  .serp .res { padding:7px 2px; }
  .serp .res .tt { color:#1a73e8; font-size:14px; cursor:default; }
  .serp .res .sn { color:var(--dim); font-size:12px; }
  .vocab { font-size:13px; padding:5px 8px; display:flex; justify-content:space-between; max-width:480px; border-radius:8px; }
  .vocab.clk { cursor:pointer; }
  .vocab.clk:hover { background:var(--surface2); }
  .vocab b { color:var(--accent); }
  /* 語彙の詳細(採用曲線・伝播ネットワーク・伝播ログ) */
  .vlegend { display:flex; gap:14px; font-size:11.5px; color:var(--dim); margin:2px 2px 8px; flex-wrap:wrap; }
  .vlegend i { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:4px; vertical-align:middle; }
  .vlog { max-height:320px; overflow-y:auto; font-size:12px; border:1px solid var(--line); border-radius:12px; }
  .vlog .row { padding:5px 10px; border-bottom:1px solid var(--line); line-height:1.5; }
  .vlog .row:last-child { border-bottom:0; }
  .vlog .birth { background:var(--surface2); border-left:3px solid var(--accent); }
  .vlog .ch { font-weight:600; }
  .vcap { font-size:11px; color:#e8a33d; margin:2px 2px 6px; }
  .bld { font-size:12.5px; padding:7px 10px; margin:4px 0; background:var(--surface2); border-radius:8px; cursor:pointer; display:flex; justify-content:space-between; max-width:520px; }
  .bld:hover { background:var(--btnH); }
  .roster { font-size:12.5px; display:grid; grid-template-columns:repeat(auto-fill,minmax(190px,1fr)); gap:6px; }
  .roster div { padding:7px 9px; background:var(--surface2); border-radius:8px; cursor:pointer; }
  .roster div:hover { background:var(--btnH); }
  .chartBox { background:var(--surface); border:1px solid var(--line); border-radius:14px; padding:12px 14px; margin:0 0 16px; max-width:860px; }
  .chartBox h3 { font-size:13.5px; margin-bottom:2px; }
  .chartBox .sub { font-size:11.5px; color:var(--dim); margin-bottom:6px; }
  .chartBox canvas { width:100%; height:230px; }
  #relCv { width:100%; height:calc(100vh - 240px); background:var(--surface); border:1px solid var(--line); border-radius:14px; cursor:grab; }
  .scenForm input, .scenForm select, .scenForm textarea { width:100%; max-width:480px; display:block; margin:3px 0 8px; }
  .scenForm label { font-size:11px; color:var(--dim); }
  .subtabs { display:flex; gap:6px; margin-bottom:10px; }
  /* フロア(間取り)モーダル — viewer と同等の階層操作 */
  #floorModal { position:fixed; inset:0; background:rgba(10,14,20,.5); backdrop-filter:blur(4px); display:none; align-items:center; justify-content:center; z-index:30; }
  #floorBox { width:660px; max-width:94%; max-height:92%; overflow-y:auto; padding:18px; }
  #floorBox h3 { font-size:16px; }
  #floorBtns { display:flex; flex-wrap:wrap; gap:5px; margin:10px 0; }
  #floorBtns button { padding:5px 10px; font-size:12px; }
  #floorGuide { font-size:12px; color:var(--dim); margin:5px 0; line-height:1.6; }
  #floorDisc { font-size:11px; color:var(--dim); opacity:.85; margin:0 0 6px; }
  canvas#floorCv { width:100%; height:340px; border-radius:12px; display:block; }
  #floorBox .fx { text-align:right; margin-top:12px; }
</style></head><body>
<div id="top">
  <h1>📊 __RUN__ ダッシュボード</h1>
  <button id="play">▶</button>
  <input type="range" id="seek" min="0" value="0" step="1">
  <span class="clock" id="clock"></span>
  <button onclick="window.open('viewer.html')">🗺 地図ビューア</button>
</div>
<div id="main">
  <div id="nav">
    <button data-tab="feed" class="on">🕒 出来事</button>
    <button data-tab="sns">📱 SNS</button>
    <button data-tab="dm">✉️ メッセージ</button>
    <button data-tab="search">🔎 検索ログ</button>
    <button data-tab="ana">📈 分析</button>
    <button data-tab="rel">🕸 関係</button>
    <button data-tab="vocab">💬 語彙</button>
    <button data-tab="bld">🏢 施設</button>
    <button data-tab="people">👥 住民</button>__LENS_TABS__
    <button data-tab="scen">🎬 シナリオ</button>
  </div>
  <div id="body"></div>
</div>
<div id="floorModal"><div id="floorBox" class="glass">
  <h3 id="floorTitle"></h3>
  <div id="floorGuide"></div>
  <div id="floorDisc">※ 間取りは建物IDから生成した推定(架空)です。実在建物の実際の内装は非公開のため(研究倫理R17)。</div>
  <div id="floorBtns"></div>
  <canvas id="floorCv"></canvas>
  <div class="fx"><button onclick="closeFloor()">閉じる</button></div>
</div></div>
<script>
__ERR_JS__
const D = __DATA__;
__TIME_JS__
__THEME_JS__
__FLOOR_JS____INDOOR_JS__
const seek=document.getElementById('seek'); seek.max=D.nSteps-1;
let cur=D.nSteps-1, playing=false, lastT=0, tab='feed', dmSel=null, vocabSel=null;
seek.value=cur;
const body=document.getElementById('body');
const S0=()=>Math.floor(cur);
function initial(n){ return (n||'?')[0]; }

document.querySelectorAll('#nav button').forEach(b=>{
  b.onclick=()=>{ tab=b.dataset.tab;
    document.querySelectorAll('#nav button').forEach(x=>x.classList.toggle('on',x===b));
    render(true); };
});
document.getElementById('play').onclick=()=>{ playing=!playing; if(cur>=D.nSteps-1)cur=0;
  document.getElementById('play').textContent=playing?'⏸':'▶'; };
seek.oninput=()=>{ cur=Number(seek.value); render(true); };

let lastStep=-1;
function loop(ts){
  if(playing){ const dt=Math.max(0,(ts-lastT)/1000); cur=Math.max(0, Math.min(cur+dt*6, D.nSteps-1)); seek.value=cur;
    if(cur>=D.nSteps-1){ playing=false; document.getElementById('play').textContent='▶'; } }
  lastT=ts;
  themeAt(cur);   // シミュ内時刻に連動して昼夜テーマ(CSS変数)を更新
  document.getElementById('clock').textContent=tstr(S0())+`(step ${S0()})`;
  if(S0()!==lastStep){ lastStep=S0(); render(false); }
  requestAnimationFrame(ts2=>loop(ts2));
}

// ================= 描画 =================
function render(force){
  const s0=S0();
  if(tab==='scen'){ if(force && !document.querySelector('.scenForm')) renderScen(); return; }
  if(!force && (tab==='rel')) { drawRel(); return; }
  // 語彙の詳細を開いている間は step 変化で DOM を作り直さず canvas/ログだけ更新
  // (rel タブと同じ流儀。再生スクラブで曲線・ネットワークが「伸びる」)。
  if(!force && tab==='vocab' && vocabSel!==null && document.getElementById('vcNet')){ drawVocabDetail(); return; }
  if(tab==='feed') renderFeed(s0);
  else if(tab==='sns') renderSns(s0);
  else if(tab==='dm') renderDm(s0, force);
  else if(tab==='search') renderSearch(s0);
  else if(tab==='ana') renderAna(s0);
  else if(tab==='rel'){ body.innerHTML=`<div style="display:flex;gap:12px;align-items:center;margin-bottom:8px;flex-wrap:wrap">
      <span style="font-size:12px;color:var(--dim)">表示する最小会話回数</span>
      <input type="range" id="relMin" min="1" max="10" value="1" style="width:160px">
      <span id="relMinV" style="font-size:12px">1</span>
      <button id="relReset" title="全体をフィット表示にリセット">⤢ 全体表示</button>
      <span style="font-size:11.5px;color:var(--dim)">ノード=人(大きさ=会話量)・線=会話/DMの蓄積。ノードをドラッグで移動 / 背景ドラッグでパン / ホイール(2本指ピンチ)で拡大縮小 / ダブルクリックで全体表示</span></div>
      <canvas id="relCv"></canvas>`;
    document.getElementById('relMin').oninput=e=>{ document.getElementById('relMinV').textContent=e.target.value; drawRel(); };
    document.getElementById('relReset').onclick=()=>{ relFit(); drawRel(true); };
    relInit(); drawRel(); }
  else if(tab==='vocab') renderVocab(s0);
  else if(tab==='bld') renderBld(s0);
  else if(tab==='people') renderPeople(s0);
  else { if(typeof lensRender==='function') lensRender(tab, s0);   // 第50バッチ 観測レンズ(有る時だけ定義)
         if(typeof devRender==='function') devRender(tab, s0);     // 第55バッチ ペルソナ逸脱率(有る時だけ定義)
         if(typeof structRender==='function') structRender(tab, s0); }  // 第56バッチ 社会構造(有る時だけ定義)
}

function renderFeed(s0){
  const recent=D.feed.filter(e=>e.s<=s0).slice(-40).reverse();
  body.innerHTML='<div style="max-width:640px">'+recent.map(e=>{
    const who=nameOf(e.a);
    const cls=e.k==='coin'?'ev coin':e.k==='adopt'?'ev adopt':e.k==='news'?'ev news':e.k==='post'?'ev post':(e.k==='exit'||e.k==='enter')?'ev '+e.k:'ev';
    const label=e.k==='coin'?`が「${e.t}」を発明`:e.k==='adopt'?`が「${e.t}」を使い始めた`
      :e.k==='post'?`がSNSに投稿:「${e.t}」`:e.k==='news'?`📰 ${e.t}`
      :(e.k==='exit'||e.k==='enter')?`: ${e.t}`:`:「${e.t}」`;
    return `<div class="${cls}"><span class="who">${tstr(e.s)} <b style="color:${colOf(e.a)}">${who}</b></span> ${label}</div>`; }).join('')+'</div>';
}

// ---- X 風 SNS ----
function twText(t, items){
  let html=t.replace(/&/g,'&amp;').replace(/</g,'&lt;');
  for(const w of (items||[])) html=html.split(w).join(`<span class="tag">#${w}</span>`);
  return html;
}
function renderSns(s0){
  const posts=D.net.posts.filter(p=>p.s<=s0).slice(-40).reverse();
  body.innerHTML='<div style="border:1px solid var(--line);border-radius:14px;max-width:640px;overflow:hidden">'+
    (posts.map(p=>{
      const media=p.a===-1;
      const meta=D.agents.find(x=>x.id===p.a)||{};
      return `<div class="tw">
        <div class="av" style="background:${media?'#f43f5e':colOf(p.a)}">${media?'📢':initial(meta.name)}</div>
        <div class="bd">
          <div class="hd"><b>${nameOf(p.a)}</b>${media?'<span class="badge">公式</span>':''}
            <span class="id">@${media?'shibuya_news':'user'+p.a} · ${tstr(p.s)}</span></div>
          <div class="tx">${twText(p.t,p.i)}</div>
          <div class="ft"><span>💬</span><span>🔁</span><span>♡</span></div>
        </div></div>`; }).join('') || '<div class="ev">まだ投稿がありません</div>')+'</div>';
}

// ---- LINE 風 DM ----
function renderDm(s0, force){
  const dms=D.net.dms.filter(d=>d.s<=s0 && d.to!==null);
  const convs={};
  for(const d of dms){ const k=d.a<d.to? d.a+'-'+d.to : d.to+'-'+d.a;
    (convs[k]=convs[k]||[]).push(d); }
  const keys=Object.keys(convs).sort((a,b)=>{
    const la=convs[a][convs[a].length-1].s, lb=convs[b][convs[b].length-1].s; return lb-la; });
  if(dmSel===null && keys.length) dmSel=keys[0];
  body.innerHTML=`<div id="dmWrap"><div id="dmList">`+
    (keys.map(k=>{ const [a,b]=k.split('-').map(Number); const th=convs[k]; const last=th[th.length-1];
      return `<div class="conv ${k===dmSel?'on':''}" onclick="dmSel='${k}';render(true)">
        <div class="av" style="background:${colOf(a)}">${initial(nameOf(a))}</div>
        <div><div class="nm">${nameOf(a)} ⇄ ${nameOf(b)}</div>
        <div class="lm">${last.t}</div></div></div>`; }).join('')||'<div class="ev">まだDMなし</div>')+
    `</div><div id="dmThread">`+
    (dmSel && convs[dmSel]? (()=>{ const [a,b]=dmSel.split('-').map(Number);
      return `<div style="text-align:center;color:var(--dim);font-size:12px;margin-bottom:8px">${nameOf(a)}(左) ⇄ ${nameOf(b)}(右・緑)</div>`+
        convs[dmSel].map(d=>`<div class="bub ${d.a===a?'left':'right'}">${d.t}<span class="meta">${nameOf(d.a)} · ${tstr(d.s)}</span></div>`).join(''); })():'')+
    `</div></div>`;
}

// ---- 検索ログ(SERP 風)----
function renderSearch(s0){
  const ss=D.net.searches.filter(x=>x.s<=s0).slice(-15).reverse();
  body.innerHTML='<div style="font-size:12px;color:var(--dim);margin-bottom:6px">エージェントがスマホの検索エンジンで調べた内容と、返された検索結果(シミュ内データベース: 語彙の来歴・ニュース・実在POI・SNS投稿を索引)</div>'
    +(ss.map(x=>`<div class="serp">
      <div class="qbox">🔍 ${x.q}</div>
      <div class="meta">${tstr(x.s)} — <b style="color:${colOf(x.a)}">${nameOf(x.a)}</b> が検索</div>
      ${(x.r&&x.r.length? x.r.map(r=>{ const i=r.indexOf(':');
          const tt=i>0?r.slice(0,i):r, sn=i>0?r.slice(i+1):'';
          return `<div class="res"><div class="tt">${tt}</div><div class="sn">${sn}</div></div>`; }).join('')
        : '<div class="res"><div class="sn">それらしい情報は見つからなかった(未知の言葉)</div></div>')}
    </div>`).join('')||'<div class="ev">まだ検索なし</div>');
}

// ================= 分析(論文風グラフ) =================
function lineChart(cv2, series, opts){
  const c=cv2, g=c.getContext('2d');
  c.width=c.clientWidth*devicePixelRatio; c.height=c.clientHeight*devicePixelRatio;
  const W=c.width, H=c.height, mL=52*devicePixelRatio, mB=30*devicePixelRatio, mT=10*devicePixelRatio, mR=12*devicePixelRatio;
  g.clearRect(0,0,W,H);
  const x1=opts.x1??(D.nSteps-1);
  let ymax=0; for(const s of series) for(const p of s.data){ if(p[0]<=x1 && p[1]>ymax) ymax=p[1]; }
  ymax = opts.ymax ?? (ymax*1.15||1);
  const X=v=>mL+(W-mL-mR)*v/x1, Y=v=>H-mB-(H-mB-mT)*v/ymax;
  const _T=themeAt(cur), _ink=_css(_T.ink), _dim=_css(_T.dim),
        _grid=_T.k>.5?'rgba(255,255,255,.10)':'rgba(0,0,0,.10)',
        _tick=_T.k>.5?'rgba(255,255,255,.35)':'rgba(0,0,0,.35)';
  g.strokeStyle=_grid; g.fillStyle=_dim;
  g.font=`${10*devicePixelRatio}px system-ui,sans-serif`; g.textAlign='right';
  for(let k=0;k<=4;k++){ const v=ymax*k/4; g.beginPath(); g.moveTo(mL,Y(v)); g.lineTo(W-mR,Y(v)); g.stroke();
    g.fillText(opts.pct? (v*100).toFixed(0)+'%' : (Math.round(v*100)/100), mL-5*devicePixelRatio, Y(v)+3*devicePixelRatio); }
  g.textAlign='center';
  const stepsPerDay=144;
  for(let s=0;s<=x1;s+=Math.max(18,Math.round(x1/8/18)*18)){
    g.beginPath(); g.moveTo(X(s),H-mB); g.lineTo(X(s),H-mB+4*devicePixelRatio);
    g.strokeStyle=_tick; g.stroke();
    const mm=D.startMin+s*10; g.fillText(`${Math.floor(mm/60)%24}時`, X(s), H-mB+15*devicePixelRatio); }
  for(const s of series){
    g.strokeStyle=s.color; g.lineWidth=1.8*devicePixelRatio; g.beginPath(); let st=false;
    for(const p of s.data){ if(p[0]>x1) break;
      st? g.lineTo(X(p[0]),Y(p[1])) : g.moveTo(X(p[0]),Y(p[1])); st=true; }
    g.stroke(); }
  // 現在時刻カーソル
  g.strokeStyle='rgba(255,209,102,.7)'; g.setLineDash([4,4]);
  g.beginPath(); g.moveTo(X(Math.min(S0(),x1)),mT); g.lineTo(X(Math.min(S0(),x1)),H-mB); g.stroke(); g.setLineDash([]);
  // 凡例
  g.textAlign='left'; let lx=mL+8*devicePixelRatio;
  for(const s of series){ g.fillStyle=s.color; g.fillRect(lx, mT+2, 10*devicePixelRatio, 3*devicePixelRatio);
    g.fillStyle=_ink; g.fillText(s.label, lx+13*devicePixelRatio, mT+6*devicePixelRatio);
    lx += (16+s.label.length*11)*devicePixelRatio; if(lx>W-160*devicePixelRatio){ break; } }
}
const PAL=['#ffd166','#60a5fa','#6ee7b7','#f472b6','#a78bfa','#fb923c','#4ade80','#38bdf8'];
function renderAna(s0){
  body.innerHTML=`
   <div class="chartBox"><h3>① 語彙の広がり(使用者数の推移)</h3>
     <div class="sub">各語を「知っている(採用した)」人数の累積。S字カーブ=複雑伝染の目印。点線=現在時刻</div><canvas id="ch1"></canvas></div>
   <div class="chartBox"><h3>② 会話ネットワークの中心人物の移り変わり</h3>
     <div class="sub">直近4時間(24step)の会話・DM相手の数(次数中心性)。誰がハブになっているか</div><canvas id="ch2"></canvas></div>
   <div class="chartBox"><h3>③ 感情(不満 grievance)の推移</h3>
     <div class="sub">全体平均と、最終値が高い個人3名。混雑体験などで上昇する内部状態</div><canvas id="ch3"></canvas></div>
   <div class="chartBox"><h3>④ 人が集まる場所(在館人数トップの建物)</h3>
     <div class="sub">どの施設に人が吸い寄せられているか</div><canvas id="ch4"></canvas></div>
   <div class="chartBox"><h3>⑤ 街のリズム(活動人数)</h3>
     <div class="sub">勤務・睡眠・屋内・範囲外・通過車両(右軸なし・台数/10分)</div><canvas id="ch5"></canvas></div>`;
  // ① 語彙
  const top=D.vocab.map(v=>({...v, n:1+v.adopts.length})).sort((a,b)=>b.n-a.n).slice(0,8);
  lineChart(document.getElementById('ch1'), top.map((v,i)=>({label:v.w, color:PAL[i%8],
    data:(()=>{ const pts=[[v.born,1]]; let n=1;
      for(const [s,_a] of v.adopts.sort((x,y)=>x[0]-y[0])){ n++; pts.push([s,n]); }
      pts.push([D.nSteps-1,n]); return pts; })()})), {});
  // ② 中心性
  const W2=24, SAMP=6;
  const deg={};
  const samples=[];
  for(let s=0;s<D.nSteps;s+=SAMP){ const m={};
    for(const [ps,a,b] of D.pairs){ if(ps>s||ps<s-W2) continue;
      (m[a]=m[a]||new Set()).add(b); (m[b]=m[b]||new Set()).add(a); }
    samples.push([s,m]);
    for(const k in m) deg[k]=Math.max(deg[k]||0, m[k].size); }
  const topA=Object.keys(deg).sort((a,b)=>deg[b]-deg[a]).slice(0,5).map(Number);
  lineChart(document.getElementById('ch2'), topA.map((a,i)=>({label:nameOf(a), color:PAL[i%8],
    data:samples.map(([s,m])=>[s,(m[a]?m[a].size:0)])})), {});
  // ③ 感情
  const series3=[];
  if(D.metrics.mean_grievance) series3.push({label:'全体平均', color:'#eaf0f6',
    data:D.metrics.mean_grievance.map((v,s)=>[s,v])});
  const finals=Object.entries(D.stateSeries).map(([a,v])=>[Number(a), v.length? v[v.length-1][1]:0])
    .sort((x,y)=>y[1]-x[1]).slice(0,3);
  finals.forEach(([a],i)=>{ series3.push({label:nameOf(a), color:PAL[i%8],
    data:[[0,0.1],...D.stateSeries[String(a)]]}); });
  lineChart(document.getElementById('ch3'), series3, {});
  // ④ 場所
  lineChart(document.getElementById('ch4'), D.places.names.map((n,i)=>({label:n.slice(0,10), color:PAL[i%8],
    data:D.places.series[i].map((v,s)=>[s,v])})), {});
  // ⑤ リズム
  const rhythm=[['n_working','勤務中'],['n_sleeping','睡眠中'],['n_inside_buildings','屋内'],['n_outside','範囲外'],['n_cars','車(台/10分)']];
  lineChart(document.getElementById('ch5'), rhythm.filter(([k])=>D.metrics[k]).map(([k,l],i)=>({label:l, color:PAL[i%8],
    data:D.metrics[k].map((v,s)=>[s,v])})), {});
}

// ================= 関係グラフ(改良版) =================
// ノード座標は「世界座標」(初期=canvas device px)。view=世界→画面のカメラ(拡大縮小・パン)。
// 操作: ノードをドラッグで移動 / 背景ドラッグでパン / ホイール・2本指ピンチで拡大縮小(カーソル中心) /
//       ダブルクリック・「全体表示」ボタンで全ノードをフィット。
let relNodes=null, relDrag=null, relPan=null, relView={s:1,tx:0,ty:0}, _relPinch=null;
function relToWorld(px,py){ return [(px-relView.tx)/relView.s, (py-relView.ty)/relView.s]; }
function relHit(px,py){   // 画面座標(device px)でのノード当たり判定 → index or -1
  for(let i=0;i<D.ids.length;i++){
    const sx=relNodes[i].x*relView.s+relView.tx, sy=relNodes[i].y*relView.s+relView.ty;
    if(Math.hypot(sx-px,sy-py)<15*devicePixelRatio) return i; }
  return -1;
}
function relZoomAt(px,py,m){   // 画面点(px,py)を固定して倍率 m 拡大縮小
  const ns=Math.max(0.1, Math.min(12, relView.s*m)), k=ns/relView.s;
  relView.tx=px-(px-relView.tx)*k; relView.ty=py-(py-relView.ty)*k; relView.s=ns;
}
function relFit(){   // 全ノードの外接矩形を余白付きで画面にフィット
  const c=document.getElementById('relCv'); if(!c||!relNodes||!relNodes.length) return;
  let x0=1e9,y0=1e9,x1=-1e9,y1=-1e9;
  for(const nd of relNodes){ x0=Math.min(x0,nd.x); y0=Math.min(y0,nd.y); x1=Math.max(x1,nd.x); y1=Math.max(y1,nd.y); }
  const pad=46*devicePixelRatio, bw=(x1-x0)||1, bh=(y1-y0)||1;
  relView.s=Math.max(0.1, Math.min((c.width-pad*2)/bw, (c.height-pad*2)/bh, 8));
  relView.tx=c.width/2-(x0+x1)/2*relView.s; relView.ty=c.height/2-(y0+y1)/2*relView.s;
}
function relInit(){
  const c=document.getElementById('relCv'); if(!c) return;
  c.width=c.clientWidth*devicePixelRatio; c.height=c.clientHeight*devicePixelRatio;
  const n=D.ids.length;
  if(!relNodes) relNodes=D.ids.map((_,i)=>({
    x:Math.cos(i/n*6.283)*c.height*0.32+c.width/2,
    y:Math.sin(i/n*6.283)*c.height*0.32+c.height/2, vx:0, vy:0}));
  const evPx=e=>{ const r=c.getBoundingClientRect();
    return [(e.clientX-r.left)*devicePixelRatio, (e.clientY-r.top)*devicePixelRatio]; };
  // --- マウス: ノード上=ドラッグ移動 / 背景=パン(移動量2px超で pan と判定) ---
  c.onmousedown=e=>{ const [px,py]=evPx(e); const hit=relHit(px,py);
    if(hit>=0){ relDrag=hit; } else { relPan={px,py,moved:false}; }
    c.style.cursor='grabbing'; };
  c.onmousemove=e=>{ const [px,py]=evPx(e);
    if(relDrag!==null){ const [wx,wy]=relToWorld(px,py);
      relNodes[relDrag].x=wx; relNodes[relDrag].y=wy; relNodes[relDrag].vx=relNodes[relDrag].vy=0; drawRel(true); }
    else if(relPan){ relView.tx+=px-relPan.px; relView.ty+=py-relPan.py;
      if(Math.abs(px-relPan.px)+Math.abs(py-relPan.py)>2) relPan.moved=true;
      relPan.px=px; relPan.py=py; drawRel(true); } };
  const endMouse=()=>{ relDrag=null; relPan=null; c.style.cursor='grab'; };
  c.onmouseup=endMouse; c.onmouseleave=endMouse;
  c.onwheel=e=>{ e.preventDefault(); const [px,py]=evPx(e);
    relZoomAt(px,py, e.deltaY<0?1.15:1/1.15); drawRel(true); };
  c.ondblclick=e=>{ e.preventDefault(); relFit(); drawRel(true); };
  // --- タッチ: 1本=ドラッグ/パン, 2本=ピンチ拡大縮小+パン ---
  const touchPx=t=>{ const r=c.getBoundingClientRect();
    return [(t.clientX-r.left)*devicePixelRatio, (t.clientY-r.top)*devicePixelRatio]; };
  c.ontouchstart=e=>{ e.preventDefault();
    if(e.touches.length===1){ const [px,py]=touchPx(e.touches[0]); const hit=relHit(px,py);
      if(hit>=0){ relDrag=hit; relPan=null; } else { relDrag=null; relPan={px,py,moved:false}; } _relPinch=null; }
    else if(e.touches.length>=2){ relDrag=null; relPan=null;
      const a=touchPx(e.touches[0]), b=touchPx(e.touches[1]);
      _relPinch={d:Math.hypot(a[0]-b[0],a[1]-b[1]), mx:(a[0]+b[0])/2, my:(a[1]+b[1])/2}; } };
  c.ontouchmove=e=>{ e.preventDefault();
    if(e.touches.length===1 && relDrag!==null){ const [px,py]=touchPx(e.touches[0]);
      const [wx,wy]=relToWorld(px,py); relNodes[relDrag].x=wx; relNodes[relDrag].y=wy;
      relNodes[relDrag].vx=relNodes[relDrag].vy=0; drawRel(true); }
    else if(e.touches.length===1 && relPan){ const [px,py]=touchPx(e.touches[0]);
      relView.tx+=px-relPan.px; relView.ty+=py-relPan.py; relPan.px=px; relPan.py=py; drawRel(true); }
    else if(e.touches.length>=2 && _relPinch){ const a=touchPx(e.touches[0]), b=touchPx(e.touches[1]);
      const d=Math.hypot(a[0]-b[0],a[1]-b[1]), mx=(a[0]+b[0])/2, my=(a[1]+b[1])/2;
      if(_relPinch.d>0) relZoomAt(mx,my, d/_relPinch.d);
      relView.tx+=mx-_relPinch.mx; relView.ty+=my-_relPinch.my;
      _relPinch={d,mx,my}; drawRel(true); } };
  const endTouch=e=>{ if(!e.touches || e.touches.length===0){ relDrag=null; relPan=null; _relPinch=null; } };
  c.ontouchend=endTouch; c.ontouchcancel=endTouch;
}
function drawRel(noPhysics){
  const c=document.getElementById('relCv'); if(!c||!relNodes) return;
  const g=c.getContext('2d'); const n=D.ids.length; const s0=S0();
  const minW=Number((document.getElementById('relMin')||{}).value||1);
  const cnt={};
  for(const [s,a,b] of D.pairs){ if(s>s0) continue; const k=a<b?a+'-'+b:b+'-'+a; cnt[k]=(cnt[k]||0)+1; }
  const edges=Object.entries(cnt).filter(([,w])=>w>=minW)
    .map(([k,w])=>{ const [a,b]=k.split('-').map(Number); return [iOf[a],iOf[b],w]; })
    .filter(([a,b])=>a!==undefined&&b!==undefined);
  const degree=new Array(n).fill(0);
  for(const [i,j,w] of edges){ degree[i]+=w; degree[j]+=w; }
  // 物理(接続あり=引力、全対=斥力、中心へ弱い重力)。ドラッグ中/パン・ズームのみの再描画では停止。
  if(!noPhysics){
    for(let it=0; it<12; it++){
      for(let i=0;i<n;i++) for(let j=i+1;j<n;j++){
        const dx=relNodes[j].x-relNodes[i].x, dy=relNodes[j].y-relNodes[i].y;
        const d2=dx*dx+dy*dy+120, f=9000*devicePixelRatio/d2;
        relNodes[i].vx-=dx*f*.01; relNodes[i].vy-=dy*f*.01; relNodes[j].vx+=dx*f*.01; relNodes[j].vy+=dy*f*.01; }
      for(const [i,j,w] of edges){ const dx=relNodes[j].x-relNodes[i].x, dy=relNodes[j].y-relNodes[i].y;
        const f=.0011*Math.min(w,18);
        relNodes[i].vx+=dx*f; relNodes[i].vy+=dy*f; relNodes[j].vx-=dx*f; relNodes[j].vy-=dy*f; }
      for(let i=0;i<n;i++){ if(i===relDrag) continue; const nd=relNodes[i];
        nd.vx+=(c.width/2-nd.x)*.0006; nd.vy+=(c.height/2-nd.y)*.0006;
        nd.x+=nd.vx; nd.y+=nd.vy; nd.vx*=.82; nd.vy*=.82;
        nd.x=Math.max(20,Math.min(c.width-20,nd.x)); nd.y=Math.max(20,Math.min(c.height-20,nd.y)); }
    }
  }
  const V=relView;
  g.setTransform(1,0,0,1,0,0); g.clearRect(0,0,c.width,c.height);
  g.setTransform(V.s,0,0,V.s,V.tx,V.ty);   // 以降は世界座標で描画(カメラ変換をキャンバスに適用)
  for(const [i,j,w] of edges){ g.strokeStyle=`rgba(122,162,255,${Math.min(.85,.12+w*.05)})`;
    g.lineWidth=Math.min(7,.6+w*.4)*devicePixelRatio/1.5;
    g.beginPath(); g.moveTo(relNodes[i].x,relNodes[i].y); g.lineTo(relNodes[j].x,relNodes[j].y); g.stroke(); }
  g.font=`${10.5*devicePixelRatio}px sans-serif`; g.textAlign='center';
  for(let i=0;i<n;i++){
    const r=(4+Math.min(10,Math.sqrt(degree[i])*1.6))*devicePixelRatio;
    g.fillStyle=`hsl(${hue(i)} 70% 60%)`;
    g.beginPath(); g.arc(relNodes[i].x,relNodes[i].y,r,0,7); g.fill();
    g.strokeStyle='rgba(0,0,0,.5)'; g.lineWidth=1*devicePixelRatio; g.stroke();
    if(degree[i]>0 || n<=40){
      const nm=(D.agents[i]||{}).name||('a'+D.ids[i]);
      g.fillStyle='rgba(0,0,0,.65)';
      const tw=g.measureText(nm).width;
      g.fillRect(relNodes[i].x-tw/2-3, relNodes[i].y+r+2*devicePixelRatio, tw+6, 13*devicePixelRatio);
      g.fillStyle='#eaf0f6';
      g.fillText(nm, relNodes[i].x, relNodes[i].y+r+12*devicePixelRatio); }
  }
  g.setTransform(1,0,0,1,0,0);   // 変換を元に戻す(他描画・clearRect のため)
}

// ---- 語彙の広がりの可視化(第45バッチ)----
// チャネル分類: 対面(face)/DM(dm)/SNS(sns)/メディア(search・event・news・その他)。
const VCH_COLOR = {face:'#34d399', dm:'#f472b6', sns:'#60a5fa',
                   search:'#fbbf24', event:'#fbbf24', news:'#fbbf24'};
function vchCat(ch){ return ch==='face'?'対面':ch==='dm'?'DM':ch==='sns'?'SNS':'メディア'; }
function vchColor(ch){ return VCH_COLOR[ch]||'#94a3b8'; }
const VNET_CAP=250;   // ネットワーク図に描く採用者ノードの上限(超過は「表示は上位N」と明示)

function renderVocab(s0){
  if(vocabSel!==null && D.vocab[vocabSel]) { renderVocabDetail(); return; }
  // 一覧(クリックで詳細へ)。使用者数=創案者1 + s0 までの採用者。
  const rows=D.vocab.map((v,idx)=>({v,idx})).filter(({v})=>v.born<=s0)
    .map(({v,idx})=>({idx, w:v.w, born:v.born, by:nameOf(v.creator), media:v.media,
      n:1+v.adopts.filter(x=>x[0]<=s0).length})).sort((a,b)=>b.n-a.n).slice(0,60);
  body.innerHTML='<div style="font-size:12px;color:var(--dim);margin-bottom:8px">生まれた言葉をクリックすると、採用曲線・伝播ネットワーク・伝播ログで「どう広がったか」を追えます(再生時刻に連動)</div>'
    +(rows.map(r=>`<div class="vocab clk" onclick="vocabSel=${r.idx};render(true)"><span>「<b>${r.w}</b>」 <span style="color:var(--dim);font-size:11px">${tstr(r.born)} ${r.media?'📢メディア発':r.by+'発'}</span></span><span>${r.n}人 ›</span></div>`).join('')
    ||'<div class="ev">まだ言葉が生まれていない</div>');
}

function renderVocabDetail(){
  const v=D.vocab[vocabSel], s0=S0();
  const cap=(D.caps||[]).find(c=>c.w===v.w && c.total>c.kept);
  body.innerHTML=`
    <button class="vback" onclick="vocabSel=null;render(true)">← 語彙一覧へ</button>
    <h2 style="font-size:18px;margin:2px 0">「${v.w}」</h2>
    <div style="font-size:12px;color:var(--dim);margin-bottom:10px">
      誕生 ${tstr(v.born)} · ${v.media?'📢 メディア発(creator=-1)':'発案 '+nameOf(v.creator)}
      · 累計採用 ${v.adopts.length}人 · 記録された聴取辺 ${v.tn||0}本</div>
    ${cap?`<div class="vcap">⚠ 聴取辺が多いため上位 ${cap.kept} / 全 ${cap.total} 本のみ表示(誕生初期の採用に効いた辺を優先)</div>`:''}
    <div class="chartBox"><h3>① 採用曲線(累積採用者数)</h3>
      <div class="sub">現在時刻(点線)まで実線・以降は薄く。スクラバーを動かすと伸びます</div>
      <canvas id="vcCurve"></canvas></div>
    <div class="chartBox"><h3>② 伝播ネットワーク(誰から誰へ・チャネル色分け)</h3>
      <div class="sub">中心=発案者。採用者を採用時刻順に外へ配置。エッジ=当人の採用に効いた最初の聴取辺</div>
      <div class="vlegend">
        <span><i style="background:${VCH_COLOR.face}"></i>対面</span>
        <span><i style="background:${VCH_COLOR.dm}"></i>DM</span>
        <span><i style="background:${VCH_COLOR.sns}"></i>SNS</span>
        <span><i style="background:${VCH_COLOR.search}"></i>メディア/検索</span>
        <span id="vnetNote" style="color:var(--dim)"></span></div>
      <canvas id="vcNet" style="height:340px"></canvas></div>
    <div class="chartBox"><h3>③ 伝播ログ(時刻順・新しい順)</h3>
      <div class="sub">誰が誰からどのチャネルで聞いて採用したか。先頭は誕生と発生文脈(きっかけ)</div>
      <div class="vlog" id="vcLog"></div></div>`;
  drawVocabDetail();
}

// item の trans は step 昇順 → 各 to の最初の出現 = 「当人宛の最初の聴取辺」(採用に効いた辺)。
function firstHear(v){ const m={};
  for(const e of (v.trans||[])){ const to=e[2]; if(!(to in m)) m[to]=e; } return m; }

function drawVocabDetail(){
  const v=D.vocab[vocabSel]; if(!v) return; const s0=S0();
  drawVCurve(document.getElementById('vcCurve'), v, s0);
  drawVNet(document.getElementById('vcNet'), v, s0);
  drawVLog(document.getElementById('vcLog'), v, s0);
}

// ① 採用曲線: 累積採用者数 vs 時刻。s0 まで実線・以降は薄い破線。
function drawVCurve(c, v, s0){
  if(!c) return; c.width=c.clientWidth*devicePixelRatio; c.height=c.clientHeight*devicePixelRatio;
  const g=c.getContext('2d'), W=c.width, H=c.height;
  const mL=42*devicePixelRatio, mB=26*devicePixelRatio, mT=10*devicePixelRatio, mR=12*devicePixelRatio;
  const T=themeAt(cur), ink=_css(T.ink), dim=_css(T.dim);
  const grid=T.k>.5?'rgba(255,255,255,.10)':'rgba(0,0,0,.10)';
  g.clearRect(0,0,W,H);
  const pts=[[v.born,1]]; let n=1;
  for(const a of v.adopts.slice().sort((x,y)=>x[0]-y[0])){ n++; pts.push([a[0],n]); }
  pts.push([D.nSteps-1,n]);
  const x0=v.born, x1=D.nSteps-1, ymax=Math.max(1,n)*1.1;
  const X=s=>mL+(W-mL-mR)*(x1>x0?(s-x0)/(x1-x0):0), Y=val=>H-mB-(H-mB-mT)*val/ymax;
  g.strokeStyle=grid; g.fillStyle=dim; g.font=`${10*devicePixelRatio}px system-ui`; g.textAlign='right';
  for(let k=0;k<=4;k++){ const val=ymax*k/4; g.beginPath(); g.moveTo(mL,Y(val)); g.lineTo(W-mR,Y(val)); g.stroke();
    g.fillText(Math.round(val), mL-5*devicePixelRatio, Y(val)+3*devicePixelRatio); }
  // 折れ線(step 関数): s0 で実線/破線を分ける
  const drawSeg=(solid)=>{ g.beginPath(); let started=false, prev=null;
    for(const p of pts){
      if(solid && p[0]>s0){ if(prev){ const yv=prev[1]; g.lineTo(X(Math.min(s0,x1)),Y(yv)); } break; }
      if(!started){ g.moveTo(X(p[0]),Y(p[1])); started=true; }
      else { g.lineTo(X(p[0]),Y(prev[1])); g.lineTo(X(p[0]),Y(p[1])); }
      prev=p; } g.stroke(); };
  // 以降(薄い破線・全区間)
  g.strokeStyle=T.k>.5?'rgba(255,209,102,.35)':'rgba(180,130,20,.35)'; g.lineWidth=1.6*devicePixelRatio; g.setLineDash([4,4]);
  g.beginPath(); { let started=false, prev=null;
    for(const p of pts){ if(!started){ g.moveTo(X(p[0]),Y(p[1])); started=true; }
      else { g.lineTo(X(p[0]),Y(prev[1])); g.lineTo(X(p[0]),Y(p[1])); } prev=p; } } g.stroke();
  g.setLineDash([]);
  // s0 まで実線(上に重ねる)
  g.strokeStyle='#ffd166'; g.lineWidth=2.2*devicePixelRatio; drawSeg(true);
  // 現在時刻カーソル
  g.strokeStyle='rgba(255,209,102,.7)'; g.setLineDash([4,4]); g.lineWidth=1*devicePixelRatio;
  g.beginPath(); g.moveTo(X(Math.min(s0,x1)),mT); g.lineTo(X(Math.min(s0,x1)),H-mB); g.stroke(); g.setLineDash([]);
  // x 軸時刻
  g.fillStyle=dim; g.textAlign='center';
  for(let s=x0; s<=x1; s+=Math.max(6,Math.round((x1-x0)/6/6)*6||6)){
    const mm=D.startMin+s*10; g.fillText(`${Math.floor(mm/60)%24}時`, X(s), H-mB+15*devicePixelRatio); }
}

// ② 伝播ネットワーク: 発案者中心・採用者を採用時刻順に外へ・エッジ=最初の聴取辺(チャネル色)。
function drawVNet(c, v, s0){
  if(!c) return; c.width=c.clientWidth*devicePixelRatio; c.height=c.clientHeight*devicePixelRatio;
  const g=c.getContext('2d'), W=c.width, H=c.height, cx=W/2, cy=H/2;
  const T=themeAt(cur), dim=_css(T.dim);
  g.clearRect(0,0,W,H);
  const fh=firstHear(v);
  // s0 までの採用者を採用時刻順に(上限 VNET_CAP=早い順)
  let ad=v.adopts.filter(a=>a[0]<=s0).slice().sort((x,y)=>x[0]-y[0]);
  const total=ad.length; if(ad.length>VNET_CAP) ad=ad.slice(0,VNET_CAP);
  const note=document.getElementById('vnetNote');
  if(note) note.textContent = total>VNET_CAP? `(採用者 ${total}人中 早い ${VNET_CAP}人を表示)` : (total?`(採用者 ${total}人)`:'');
  const pos={}; pos[v.creator]=[cx,cy];
  const R=Math.min(W,H)*0.44;
  ad.forEach((a,i)=>{ const ang=i*2.399963;              // 黄金角スパイラル(重なりにくい)
    const rr=R*Math.sqrt((i+1)/(ad.length+1));
    pos[a[1]]=[cx+Math.cos(ang)*rr, cy+Math.sin(ang)*rr]; });
  // エッジ(採用者ごとに「最初の聴取辺の from」→当人。from が図に無ければ発案者へ退避)
  for(const a of ad){ const to=a[1], e=fh[to];
    let from = (e && e[1]!==undefined && e[1] in pos)? e[1] : v.creator;
    const ch = e? e[3] : 'face';
    const P=pos[from], Q=pos[to]; if(!P||!Q) continue;
    g.strokeStyle=vchColor(ch)+'aa'; g.lineWidth=1.3*devicePixelRatio;
    g.beginPath(); g.moveTo(P[0],P[1]); g.lineTo(Q[0],Q[1]); g.stroke(); }
  // ノード
  g.font=`${10*devicePixelRatio}px system-ui`; g.textAlign='center';
  for(const a of ad){ const P=pos[a[1]]; if(!P) continue;
    g.fillStyle=colOf(a[1]); g.beginPath(); g.arc(P[0],P[1],4*devicePixelRatio,0,7); g.fill(); }
  // 発案者(中心・強調)
  const cP=pos[v.creator];
  g.fillStyle=v.media?'#f43f5e':colOf(v.creator);
  g.beginPath(); g.arc(cP[0],cP[1],8*devicePixelRatio,0,7); g.fill();
  g.strokeStyle=T.k>.5?'#fff':'#111'; g.lineWidth=2*devicePixelRatio; g.stroke();
  g.fillStyle=dim; g.fillText(v.media?'📢メディア':nameOf(v.creator), cP[0], cP[1]-12*devicePixelRatio);
  if(!ad.length){ g.fillStyle=dim; g.textAlign='center'; g.font=`${12*devicePixelRatio}px system-ui`;
    g.fillText('この時刻ではまだ採用者がいません', cx, cy+30*devicePixelRatio); }
}

// ③ 伝播ログ: s0 までの採用を新しい順に。先頭(最下)は誕生+発生文脈。
function drawVLog(el, v, s0){
  if(!el) return; const fh=firstHear(v);
  const ad=v.adopts.filter(a=>a[0]<=s0).slice().sort((x,y)=>x[0]-y[0]);
  const rows=ad.map(a=>{ const to=a[1], e=fh[to];
    const from = e? nameOf(e[1]) : '(不明)';
    const ch = e? e[3] : '';
    const label = ch? `<span class="ch" style="color:${vchColor(ch)}">${vchCat(ch)}</span>で` : '';
    return `<div class="row">${tstr(a[0])} <b style="color:${colOf(to)}">${nameOf(to)}</b> が ${from} から ${label}聞いて採用</div>`;
  }).reverse().slice(0, 160);
  // 誕生行(発生文脈=きっかけの観察)
  const ctx=v.ctx||{}; const bits=[];
  if(ctx.place) bits.push(`場所: ${ctx.place}`);
  if(ctx.fire_reason) bits.push(`きっかけ: ${ctx.fire_reason}`);
  if(ctx.drive!==undefined) bits.push(`drive: ${ctx.drive}`);
  if(ctx.saw_feed) bits.push('直前にフィード視聴');
  if(Array.isArray(ctx.company_ids)&&ctx.company_ids.length) bits.push(`同席 ${ctx.company_ids.length}人`);
  const birth=`<div class="row birth">🌱 ${tstr(v.born)} <b style="color:${v.media?'#f43f5e':colOf(v.creator)}">${v.media?'📢メディア':nameOf(v.creator)}</b> が「${v.w}」を発案`
    +(bits.length?` <span style="color:var(--dim)">— ${bits.join(' / ')}</span>`:'')+'</div>';
  el.innerHTML = rows.join('') + birth;
}
let bldQuery='';
function renderBld(s0){
  const pos=D.positions[s0]; const occ={};
  pos.forEach(p=>{ if(p[2]>=1000){ const bi=Math.floor((p[2]-1000)/100); occ[bi]=(occ[bi]||0)+1; } });
  const rows=D.buildings.map((b,bi)=>({b,bi}))
    .filter(({b,bi})=> (b.name && (!bldQuery || b.name.includes(bldQuery))) || occ[bi])
    .sort((r1,r2)=> (occ[r2.bi]||0)-(occ[r1.bi]||0) || (r2.b.guide?1:0)-(r1.b.guide?1:0))
    .slice(0,100);
  body.innerHTML=`<input type="text" id="bldQ" placeholder="🔍 建物名で検索(例: ヒカリエ)" value="${bldQuery}" style="width:320px;margin-bottom:8px">
    <div style="font-size:11.5px;color:var(--dim);margin-bottom:6px">📋=実フロアガイドあり。クリックで階層(間取り)を表示・切替できます</div>`+
    rows.map(({b,bi})=>`<div class="bld" onclick="openFloor(${bi})"><span>${b.name||'(無名ビル)'}${b.guide?' 📋':''}(${b.below?'B'+b.below+'〜':''}${b.levels}F)</span><b>${occ[bi]||0}人</b></div>`).join('');
  const inp=document.getElementById('bldQ');
  inp.oninput=()=>{ bldQuery=inp.value; renderBld(S0());
    setTimeout(()=>{const i2=document.getElementById('bldQ'); i2.focus(); i2.setSelectionRange(i2.value.length,i2.value.length);},0); };
}
function renderPeople(s0){
  body.innerHTML='<div class="roster">'+D.agents.map(a=>{
    const i=iOf[a.id];
    const words=D.vocab.filter(v=>(v.creator===a.id&&v.born<=s0)||v.adopts.some(x=>x[0]<=s0&&x[1]===a.id)).length;
    return `<div><span style="color:${colOf(a.id)}">●</span> <b>${a.name}</b> ${a.gender||''} ${a.age}歳<br>
      <span style="color:var(--dim)">${a.occupation}${a.work_name? ' @'+a.work_name.slice(0,12):''} ${a.has_car?'🚗':''}${a.has_bicycle?'🚲':''} 語彙${words}</span></div>`; }).join('')+'</div>';
}

// ---- シナリオ作成 ----
window.scenEvents = window.scenEvents || [];
function renderScen(){
  body.innerHTML=`<div class="scenForm">
    <div style="font-size:12.5px;color:var(--dim);margin-bottom:8px;max-width:520px">シミュレーションの条件(世界イベント)を作成し、JSONで保存 →
    次回の実行に読み込みます。※ダッシュボードは記録の再生専用のため反映は次のランから(本選ではサーバー化してここから直接起動できるようにする予定)。</div>
    <label>day(0=初日)</label><input id="scDay" type="number" value="0" min="0">
    <label>時刻(例 11:00)</label><input id="scTime" value="11:00">
    <label>種類</label><select id="scType"><option value="announcement">公式発表</option><option value="rumor">噂</option><option value="incident">出来事</option></select>
    <label>タイトル</label><input id="scTitle" placeholder="新商品「○○」発表">
    <label>本文(ニュース・SNSに流れる文)</label><textarea id="scText" rows="2"></textarea>
    <label>追跡する語(伝播経路が item として記録される)</label><input id="scWord" placeholder="○○">
    <button onclick="scenAdd()">＋ 追加</button>
    <button onclick="scenSave()">💾 events_my.json を保存</button>
    <div id="scList" style="margin-top:10px;max-width:520px"></div>
    <div style="font-size:11.5px;color:var(--dim);margin-top:10px">保存したファイルを data/ に置いて実行:<br>
    <code>python scripts/run.py world.events_file=data/events_my.json run.n_agents=80 model.backend=ollama model.name=qwen3:8b</code></div>
  </div>`;
  scenList();
}
function scenList(){ const el=document.getElementById('scList'); if(!el) return;
  el.innerHTML=window.scenEvents.map((e,i)=>
    `<div class="ev">day${e.day} ${e.time} [${e.type}]「${e.title}」${e.word?`(語: ${e.word})`:''}
     <span class="who" style="text-decoration:underline" onclick="window.scenEvents.splice(${i},1);scenList()">削除</span></div>`).join('')
    ||'<div class="ev">イベントはまだありません</div>'; }
function scenAdd(){ const v=id=>document.getElementById(id).value;
  if(!v('scTitle')){ alert('タイトルを入れてください'); return; }
  const ev={day:Number(v('scDay')), time:v('scTime'), type:v('scType'),
            title:v('scTitle'), text:v('scText')};
  if(v('scWord')) ev.word=v('scWord');
  window.scenEvents.push(ev); scenList(); }
function scenSave(){
  const blob=new Blob([JSON.stringify({events:window.scenEvents}, null, 2)],
                      {type:'application/json'});
  const a=document.createElement('a'); a.href=URL.createObjectURL(blob);
  a.download='events_my.json'; a.click(); }

__LENS_JS__
render(true); requestAnimationFrame(loop);
</script></body></html>
"""


# ============================================================ 日次ロールアップ HTML(第57バッチ タスクC)
# --daily-rollup 専用の軽量ページ。positions を一切埋め込まない=30日×数百人でも数十KBで開ける。
# 画面に「日次ロールアップ表示」を明示(silent cap 禁止)。既存 viewer/dashboard とは別ファイル(rollup.html)。
ROLLUP_HTML = r"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__RUN__ · 日次ロールアップ</title>
<style>
:root{--bg:#0f1216;--card:#171b21;--ink:#e8ecf1;--dim:#93a1b0;--line:#28303a;--accent:#ffd166;--warn:#f59e0b;}
@media (prefers-color-scheme:light){:root{--bg:#f6f7f9;--card:#fff;--ink:#1c2430;--dim:#5b6774;--line:#e2e6ea;--accent:#b7791f;--warn:#b45309;}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,"Hiragino Sans","Noto Sans JP",sans-serif;padding:22px}
h1{font-size:19px;margin:0 0 2px} h2{font-size:15px;margin:22px 0 8px;color:var(--ink)}
.meta{color:var(--dim);font-size:12.5px;margin-bottom:14px}
.banner{background:color-mix(in srgb,var(--warn) 16%,transparent);border:1px solid var(--warn);
  border-radius:9px;padding:10px 14px;margin-bottom:16px;font-size:13px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin-bottom:14px}
.scroll{overflow-x:auto}
table{border-collapse:collapse;font-size:12.5px;white-space:nowrap}
th,td{padding:5px 9px;border-bottom:1px solid var(--line);text-align:right}
th{position:sticky;top:0;background:var(--card);color:var(--dim);font-weight:600;text-align:right}
th.d,td.d{text-align:left;position:sticky;left:0;background:var(--card);font-weight:600}
tbody tr:hover td{background:color-mix(in srgb,var(--accent) 8%,transparent)}
.stag{color:var(--warn);font-weight:600}
.sub{color:var(--dim);font-size:12px;margin:2px 0 8px}
code{background:color-mix(in srgb,var(--ink) 8%,transparent);padding:1px 5px;border-radius:4px;font-size:12px}
.pill{display:inline-block;background:color-mix(in srgb,var(--accent) 16%,transparent);
  border:1px solid var(--accent);border-radius:20px;padding:1px 10px;font-size:11.5px;margin-left:6px}
</style></head><body>
<h1>__RUN__ <span class="pill">日次ロールアップ</span></h1>
<div class="meta" id="meta"></div>
<div class="banner">📊 これは <b>日次ロールアップ表示</b>です。1日(144step=1,440分)ごとに L2 指標を平均集計し、
社会構造の事後分析(structure.json)を束ねた軽量ページ。<b>地図・位置アニメ・個票は含みません</b>
(長期ラン=30日級で viewer.html が数百MBになり開けない問題の回避用)。全量の閲覧は
<code>python viz/make_viewer.py runs/__RUN__ --no-traffic</code> を使ってください。</div>
<div id="body"></div>
<script>
const D=__DATA__;
function tstr(v){ if(v===null||v===undefined) return '—';
  if(typeof v!=='number') return String(v);
  const a=Math.abs(v);
  if(a!==0 && (a<0.001||a>=100000)) return v.toExponential(2);
  return (Math.round(v*1000)/1000).toString(); }
function pct(v){ return (v===null||v===undefined)?'—':(v*100).toFixed(1)+'%'; }

// 主要指標の表示順とラベル(存在する列だけ描く=lens OFF のランでも壊れない)
const CURATED=[
  ['mean_grievance','平均不満'],
  ['distinct_vocab_in_use','使用語彙'],
  ['n_sns_posts','SNS投稿'],
  ['total_adoptions','語彙採用'],
  ['value4_social','価値:社会'],
  ['value4_epistemic','価値:認識'],
  ['motive_recognition','欲望:承認'],
  ['trust_gini','信用Gini'],
  ['trust_top10','信用上位10%'],
  ['deviation_mean','逸脱(裁量)'],
  ['deviation_top_share','上位逸脱者率'],
  ['edge_churn_rate','edge組替率'],
  ['edges_formed','紐帯形成'],
  ['edges_broken','紐帯断絶'],
];
const meta=[];
if(D.nAgents!=null) meta.push(`エージェント ${D.nAgents}体`);
meta.push(`${D.days.length}日(${D.nSteps!=null?D.nSteps+'step':''})`);
meta.push(`1日=${D.stepsPerDay}step`);
document.getElementById('meta').textContent = meta.join(' · ');

const B=document.getElementById('body');
function h(html){ const d=document.createElement('div'); d.innerHTML=html; return d; }

// ---- 主要指標テーブル(日 × 主要L2列 + 構造指標)----
const struct=D.structure;
function sArr(path){ // structure の日次配列を安全に取り出す
  if(!struct) return null;
  const parts=path.split('.'); let o=struct;
  for(const p of parts){ if(o==null) return null; o=o[p]; }
  return Array.isArray(o)?o:null; }
const churn=sArr('churn.churn_rate'), tau=sArr('rank.tau_prev_day'),
      cturn=sArr('centrality.turnover'), cchg=sArr('community.change_rate');
const curCols=CURATED.filter(([k])=> D.metrics[k]!==undefined);
let structCols=[];
if(churn) structCols.push(['churn','churn率(構造)']);
if(tau) structCols.push(['tau','前日τ(構造)']);
if(cturn) structCols.push(['cturn','中心turnover(構造)']);
if(cchg) structCols.push(['cchg','コミュ変化(構造)']);

let html='<div class="card"><h2>日次サマリ</h2>'
 +'<div class="sub">主要 L2 指標の日次平均(step毎の平均)＋ 社会構造の事後指標(structure.json が有る時)。'
 +'空欄=その日にデータ無し / 列そのものが無い(lens OFF 等)。</div><div class="scroll"><table><thead><tr>'
 +'<th class="d">日</th>';
for(const [,lab] of curCols) html+=`<th>${lab}</th>`;
for(const [,lab] of structCols) html+=`<th>${lab}</th>`;
html+='</tr></thead><tbody>';
for(const d of D.days){
  html+=`<tr><td class="d">Day ${d}</td>`;
  for(const [k] of curCols){ const v=(D.metrics[k]||[])[d];
    const isPct=(k.indexOf('gini')>=0||k.indexOf('share')>=0||k.indexOf('top10')>=0||k.indexOf('churn_rate')>=0||k.indexOf('deviation')>=0);
    html+=`<td>${isPct?pct(v):tstr(v)}</td>`; }
  for(const [k] of structCols){
    const arr=k==='churn'?churn:k==='tau'?tau:k==='cturn'?cturn:cchg;
    const v=arr?arr[d]:null;
    html+=`<td>${(k==='tau')?tstr(v):pct(v)}</td>`; }
  html+='</tr>';
}
html+='</tbody></table></div></div>';
B.appendChild(h(html));

// ---- 構造固着の要約 ----
if(struct && struct.stagnation){
  const st=struct.stagnation; let s='<div class="card"><h2>構造固着の検知(観測記録・介入なし)</h2>';
  if(st.longest){ const lg=st.longest;
    s+=`<div>検出 <b>${st.combined.length}</b> 区間 / 固着延べ <b>${st.total_stagnant_days}</b> 日 / `
      +`最長 <span class="stag">Day ${lg.start_day}–${lg.end_day}(${lg.len}日)</span></div>`;
  } else {
    s+='<div>中心性が N 日以上入れ替わらない固着区間は<b>検出されず</b>(構造は動いている)。</div>';
  }
  const labels={centrality_churn:'中心性 turnover 低(上位が入れ替わらない)',
                edge_churn:'edge churn 低(紐帯が組み替わらない)',
                rank_tau:'前日τ 高(順位が入れ替わらない)'};
  s+='<div class="sub" style="margin-top:8px">信号別:</div><ul style="margin:0;padding-left:18px">';
  for(const sig of ['centrality_churn','edge_churn','rank_tau']){
    const segs=(st.by_signal&&st.by_signal[sig])||[];
    const txt=segs.length? segs.map(x=>`Day ${x.start_day}–${x.end_day}(${x.len}日)`).join('、') : 'なし';
    s+=`<li>${labels[sig]}: ${txt}</li>`;
  }
  s+='</ul></div>';
  B.appendChild(h(s));
} else {
  B.appendChild(h('<div class="card"><div class="sub">構造の事後分析(structure.json)は未生成です。'
    +'<code>python scripts/analyze_structure.py runs/'+D.runName+'</code> を掛けると固着検知・順位τ・'
    +'中心性turnover・コミュニティ変化が本ページに表示されます。</div></div>'));
}

// ---- 全 L2 指標の日次平均(横スクロール)----
if(D.hasL2){
  let a='<div class="card"><h2>全 L2 指標(日次平均)</h2><div class="sub">'
   +`${D.metricKeys.length} 列 × ${D.days.length} 日。lens ON の全体スカラーもここに出る。</div>`
   +'<div class="scroll"><table><thead><tr><th class="d">日</th>';
  for(const k of D.metricKeys) a+=`<th>${k}</th>`;
  a+='</tr></thead><tbody>';
  for(const d of D.days){ a+=`<tr><td class="d">Day ${d}</td>`;
    for(const k of D.metricKeys){ a+=`<td>${tstr((D.metrics[k]||[])[d])}</td>`; }
    a+='</tr>'; }
  a+='</tbody></table></div></div>';
  B.appendChild(h(a));
} else {
  B.appendChild(h('<div class="card"><div class="sub">L2 metrics(l2_metrics.parquet)が見つかりません。'
    +'observer が有効なランで再生成してください。</div></div>'));
}
</script></body></html>
"""


def main() -> None:
    argv = sys.argv[1:]
    # --start-tod "HH:MM" を明示指定した時だけ壁時計原点を上書き(既定=run から復元)。
    # フラグと値を argv から除去してから run_dir/フラグ判定へ(値が run_dir と誤認されないように)。
    start_min = None
    if "--start-tod" in argv:
        i = argv.index("--start-tod")
        if i + 1 < len(argv):
            start_min = _parse_start_tod(argv[i + 1])
            argv = argv[:i] + argv[i + 2:]
    args = [a for a in argv if not a.startswith("--")]
    flags = {a for a in argv if a.startswith("--")}
    run_dir = Path(args[0]) if args else REPO_ROOT / "runs" / "day80"
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    # 第57バッチ タスクC: --daily-rollup は positions を読まない軽量ページ rollup.html「だけ」を
    # 追加生成して早期 return する(既存 viewer.html/dashboard.html の生成経路には一切入らない=
    # 既定=--daily-rollup 未指定時の出力は完全に従来どおり・バイト同一)。長期ラン(30日級)向け。
    if "--daily-rollup" in flags:
        rollup = build_rollup_data(run_dir)
        html = (ROLLUP_HTML
                .replace("__RUN__", rollup["runName"])
                .replace("__DATA__", json.dumps(rollup, ensure_ascii=False)))
        out = run_dir / "rollup.html"
        out.write_text(html, encoding="utf-8")
        print(f"written: {out}  ({out.stat().st_size / 1e6:.2f} MB) [日次ロールアップ]")
        print(f"  days={len(rollup['days'])} metrics={len(rollup['metricKeys'])} "
              f"structure={'yes' if rollup['structure'] else 'no'}")
        return
    # 長期ラン(例: 100日=14400step)は背景交通の軌跡だけで数百MBになりブラウザで開けない。
    # --no-traffic で背景交通レイヤーを外す(エージェント・分析・ネットは全て従来どおり)。
    include_traffic = "--no-traffic" not in flags
    # 屋内セマンティックズーム(B5): --indoor-moves で歩行軌跡ポリラインも埋め込む(7日以下・サイズガード)。
    include_moves = "--indoor-moves" in flags
    data = build_data(run_dir, include_traffic=include_traffic, start_min=start_min,
                      include_moves=include_moves)
    payload = json.dumps(data, ensure_ascii=False)
    # 第18バッチ①: communities.json が有る時だけ色分け「コミュニティ」を追加。
    # 無ければ3トークンとも空文字へ→ MAP_HTML はバイト同一(後方互換の合格条件)。
    has_comm = "communities" in data
    comm_option = '<option value="community">コミュニティ</option>' if has_comm else ""
    comm_hook = ("\n  if(mode==='community') return communityColor(i, s0);"
                 if has_comm else "")
    comm_js = _COMMUNITY_JS if has_comm else ""
    # 移動手段(mode_legend)が有る時「だけ」タクシー色と凡例1行を注入。無ければ両トークン
    # とも空文字→ MAP_HTML はバイト同一(後方互換の合格条件)。
    has_ml = "mode_legend" in data
    mode_taxi = "if(m===3) return '#a855f7'; " if has_ml else ""
    mode_legend_js = _MODE_LEGEND_JS if has_ml else ""
    # 第50バッチ 観測レンズ: lens/trust データが有る時「だけ」タブと描画 JS を注入。無ければ
    # 両トークンとも空文字→ DASH_HTML はバイト同一(後方互換の合格条件。viewer 側にトークンは無い)。
    has_lens = "lens" in data
    has_trust = "trust" in data
    has_deviation = "deviation" in data
    has_structure = "structure" in data
    lens_tabs = ""
    if has_lens:
        lens_tabs += ('\n    <button data-tab="value">💠 価値</button>'
                      '\n    <button data-tab="motive">🎯 欲望</button>')
    if has_trust:
        lens_tabs += '\n    <button data-tab="trust">🏅 信用</button>'
    if has_deviation:
        lens_tabs += '\n    <button data-tab="deviation">🎭 逸脱</button>'
    if has_structure:
        lens_tabs += '\n    <button data-tab="structure">🏛 社会構造</button>'
    lens_js = _LENS_JS if (has_lens or has_trust) else ""
    if has_deviation:                      # 逸脱タブ JS は lens/trust と独立に注入(devRender を定義)
        lens_js += _DEV_JS
    if has_structure:                      # 社会構造タブ JS も独立に注入(structRender を定義)
        lens_js += _STRUCT_JS
    # 屋内セマンティックズーム(B5): floorSpecs(=indoor ラン)が有る時「だけ」JS/フック/クリックを注入。
    # 無ければ3トークンとも空文字→ MAP_HTML/DASH_HTML は従来とバイト同一(後方互換の合格条件)。
    has_indoor = "floorSpecs" in data
    indoor_js = _INDOOR_JS if has_indoor else ""
    indoor_hook = "  indoorOverlay(t, s0, pos);\n" if has_indoor else ""
    indoor_click = "  if(izClickChips(px,py)) return;\n" if has_indoor else ""
    for name, template in (("viewer.html", MAP_HTML), ("dashboard.html", DASH_HTML)):
        html = (template
                .replace("__COMMUNITY_OPTION__", comm_option)
                .replace("__COMMUNITY_HOOK__", comm_hook)
                .replace("__COMMUNITY_JS__", comm_js)
                .replace("__MODE_TAXI__", mode_taxi)
                .replace("__MODE_LEGEND_JS__", mode_legend_js)
                .replace("__LENS_TABS__", lens_tabs)
                .replace("__LENS_JS__", lens_js)
                .replace("__BASE_CSS__", _BASE_CSS)
                .replace("__ERR_JS__", _ERR_JS)
                .replace("__TIME_JS__", _TIME_JS)
                .replace("__THEME_JS__", _THEME_JS)
                .replace("__FLOOR_JS__", _FLOOR_JS)
                .replace("__INDOOR_JS__", indoor_js)
                .replace("__INDOOR_HOOK__", indoor_hook)
                .replace("__INDOOR_CLICK__", indoor_click)
                .replace("__RUN__", data["runName"])
                .replace("__DATA__", payload))
        out = run_dir / name
        out.write_text(html, encoding="utf-8")
        print(f"written: {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"  steps={data['nSteps']} agents={len(data['ids'])} "
          f"posts={len(data['net']['posts'])} dms={len(data['net']['dms'])} "
          f"searches={len(data['net']['searches'])} vocab={len(data['vocab'])}")


if __name__ == "__main__":
    main()
