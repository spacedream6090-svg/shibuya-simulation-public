"""L2 域内従業者の「業務の実体」(work.service。既定 OFF)。

ユーザー要望 2026-07-20:「L2 の人々も接客などのサービスを行っている、もしくは会社単位で
何かのサービスを作っているだろうからそれを反映したい」。実査は docs/research/l2-work-reality.md。

勤務中(work_node に在場・勤務時間帯)のエージェントへ「業務の実体」を与える決定論機構:
  - 接客系: 客の消費 spend(食事/カフェ/ナイトライフ/買い物)と同一 work_node に居る勤務中スタッフに
    serve を帰属(機械的=LLM 呼ゼロ・乱数ゼロ)。客側の既存イベントは不変(新イベントを足すだけ)。
  - オフィス系: 日次境界で職場単位に「出勤者数 × role重み」を org_output として集計(会社が
    「何かを作っている」の最小観測形)。

原則(R1 doctrine):
- **既定 OFF**(service.enabled=false)。OFF 時は本モジュールの経路を一切通さない=バイト一致。
- 乱数を一切引かない(新 stream 不要)・LLM 呼び出しを一切増やさない。判定は observables
  (work_node/在場/勤務時間帯/POI カテゴリ/occupation)のみ=k(信念書き戻し)非依存。
- 業種名・業務名テキストはここ(本モジュール既定)/config 由来に閉じる。本モジュールは
  src/society 直下(engine/cognition/actions/labeling/world/factors の禁則ディレクトリ外)。

配線: engine/simulation.py が `sim.workcfg = work.build_cfg(cfg.get("work"))` を1回だけ正準化して保持し、
engine/scheduler.py の `_phase_work_service` が run_step 末で本モジュールの純関数を使う。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

# 客の消費カテゴリ(spend.cat)→ 接客業務ラベルの既定写像(config 未指定時)。
# spend.cat は economy/scheduler 由来: 食事/ナイトライフ=food/nightlife、買い物=shop、
# P2 buy の leisure=cafe。taxi/bus(交通)は接客対象外なので写像に入れない。
DEFAULT_SERVE_BY_CAT = {
    "food": "飲食の接客",
    "cafe": "飲食の接客",
    "nightlife": "接客・サービス",
    "shop": "販売・レジ対応",
}

# ダイジェストで「業務が多かった」と言い換える件数の閾値(客観記述の粒度)。
_MANY_THRESHOLD = 8

# 職場束ね直し(bind_workplace。既定 OFF)。pool 経路の L2/L3 は occupation(role)が
# persona._pick_workplace の _WORK_CAT に載らないと work_node を持たない=接客(serve)/産出
# (org_output)の帰属から漏れる。台帳 organizations の workplace_poi.node へ org_id で直束ねし、
# 台帳ノードが現行地図に無い時は POI カテゴリ + 安定ハッシュで決定論マッチする(organizations.
# commute_to_poi の pool 版)。純関数(pool_pid・固定属性のみ)=run.seed 非依存=hydrate 再入不変。
_BIND_DEFAULT_BOOK = "data/organizations_shibuya_wide11k.json"


def build_cfg(raw: dict | None) -> dict:
    """conf の work ブロックを正準化(既定 OFF=現行挙動と完全同一)。"""
    raw = dict(raw or {})
    svc = dict(raw.get("service", {}) or {})
    serve = svc.get("serve_by_cat")
    serve = dict(serve) if serve else dict(DEFAULT_SERVE_BY_CAT)
    off = dict(svc.get("office", {}) or {})
    bind = dict(raw.get("bind_workplace", {}) or {})
    return {
        "enabled": bool(svc.get("enabled", False)),
        # 客の消費カテゴリ → 接客ラベル(業務名テキストは config 由来)。
        "serve_by_cat": {str(k): str(v) for k, v in serve.items()},
        # 1消費あたり帰属するスタッフ数の上限(id 昇順で先頭から)。
        "max_serve_per_event": max(0, int(svc.get("max_serve_per_event", 1))),
        # スタッフ不在の消費を agent_id=-1 の記録として残すか(挙動変更なし・観測のみ)。
        "record_unstaffed": bool(svc.get("record_unstaffed", True)),
        # interstitial(S2)ダイジェストへ業務要約を1行供給するか(ON 時のみ効く)。
        "digest": bool(svc.get("digest", True)),
        "office": {
            "enabled": bool(off.get("enabled", True)),
            # オフィス系職場と見なす職場 POI カテゴリ(既定 office)。
            "poi_cats": [str(c) for c in (off.get("poi_cats") or ["office"])],
            "base_weight": float(off.get("base_weight", 1.0)),
            # occupation/role → 産出重み(空=全員 base_weight=出勤者数に比例)。
            "role_weights": {str(k): float(v)
                             for k, v in (off.get("role_weights") or {}).items()},
        },
        # 職場束ね直し(bind_workplace。既定 OFF=現行の work_node 付与と完全同一=バイト一致)。
        "bind_workplace": {
            "enabled": bool(bind.get("enabled", False)),
            # org_id が参照する組織台帳(pool 生成元。既定=100万プールの母体 wide11k)。
            "book": str(bind.get("book", _BIND_DEFAULT_BOOK)),
            # 既に work_node を持つ個体も台帳の実 POI へ束ね直すか(既定=未束のみ=coverage 拡大に限定)。
            "rebind_bound": bool(bind.get("rebind_bound", False)),
            # 台帳ノードが現行地図に無い時、POI カテゴリ + 安定ハッシュで決定論マッチするか。
            "poi_match_fallback": bool(bind.get("poi_match_fallback", True)),
            # 決定論マッチの安定ハッシュ seed(run.seed 非依存=リプレイ・resume・別ランで不変)。
            "seed": int(bind.get("seed", 20260722)),
            # 勤務窓の既定(台帳/record の shift_pattern が無い時の後退値)。
            "default_open": str(bind.get("default_open", "09:00")),
            "default_close": str(bind.get("default_close", "18:00")),
            # occupation/role → 職場 POI カテゴリ(org_id が無い層=L5 等を束ねたい時の写像。既定=空)。
            "occ_cat": {str(k): str(v) for k, v in (bind.get("occ_cat") or {}).items()},
        },
    }


def enabled(cfg: dict | None) -> bool:
    return bool(cfg and cfg.get("enabled"))


# ---------------------------------------------------------------- 接客(serve)
def serve_label(cfg: dict, cat: str | None) -> str | None:
    """消費カテゴリ cat の接客ラベル(接客対象外=None)。"""
    if not cat:
        return None
    return cfg["serve_by_cat"].get(str(cat))


def note_serve(agent, label: str) -> None:
    """スタッフ本人の当日業務アキュムレータへ1件加える(ダイジェスト供給用)。

    属性は必要時にのみ生やす(OFF/非該当は属性不在=interstitial の実行時状態を汚さない)。"""
    acc = getattr(agent, "_work_serve_by_label", None)
    if acc is None:
        acc = {}
        agent._work_serve_by_label = acc
    acc[label] = acc.get(label, 0) + 1


def clear_digest(agent) -> None:
    """発火(ダイジェスト取得)時に当日業務アキュムレータを空にする(前回発火以降の仕切り直し)。"""
    if getattr(agent, "_work_serve_by_label", None):
        agent._work_serve_by_label = {}


def digest_line(agent) -> str | None:
    """スタッフの当日業務を interstitial ダイジェストの1事実に整形(客観記述・意味づけしない)。

    アキュムレータ不在/空なら None(=1行も足さない=バイト一致)。テキストは本モジュールに閉じる。"""
    acc = getattr(agent, "_work_serve_by_label", None)
    if not acc:
        return None
    total = sum(acc.values())
    top = sorted(acc.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    if total >= _MANY_THRESHOLD:
        return f"今日は{top}が多かった(業務{total}件)"
    return f"今日は{top}などの業務をこなした(業務{total}件)"


# ---------------------------------------------------------------- オフィス産出(org_output)
def is_office_node(city, node: str, ocfg: dict) -> bool:
    """work_node がオフィス系(poi_cats のいずれかの POI を持つ)か(決定論・乱数なし)。"""
    cats = ocfg["poi_cats"]
    for p in city.pois_at_node(node):
        if p.get("cat") in cats:
            return True
    return False


def role_weight(agent, cfg: dict) -> float:
    """出勤者の産出重み。occupation/role → role_weights、無ければ base_weight(=1人1.0)。"""
    ocfg = cfg["office"]
    key = getattr(agent, "org_role", "") or getattr(agent, "occupation", "")
    return ocfg["role_weights"].get(str(key), ocfg["base_weight"])


# ---------------------------------------------------------------- 職場束ね直し(bind_workplace)
def bind_eligible(record: dict, bcfg: dict) -> bool:
    """束ね対象か(=域内従業者/学生)。org_id を持つ(L2/L3学生)か occ_cat に写像がある層(L5 等)。

    L1 住民・L4 来街者・org_id 無しの L5 は対象外=coverage 統計の母数に入れない(勤務地を持たない
    層を『未束ね』に数えない=L2 の work_node coverage の指標を薄めない)。"""
    if record.get("org_id"):
        return True
    occ = str(record.get("occupation") or record.get("role") or "")
    return occ in bcfg.get("occ_cat", {})


def _hhmm_to_min(s, default: int) -> int:
    """"HH:MM" → 分 of day(不正なら default)。"""
    try:
        h, m = str(s).split(":")
        return int(h) * 60 + int(m)
    except (ValueError, AttributeError, TypeError):
        return int(default)


def _stable_uniform(seed: int, key: str) -> float:
    """(seed, 安定キー)から run.seed 非依存の一様値 [0,1)(hashlib=決定論・RngHub 無風・
    プロセス跨ぎ安定=リプレイ/resume/別ランで同一)。ontology._stable_uniform と同流儀。"""
    h = hashlib.blake2b(f"{int(seed)}\x1f{key}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") / float(1 << 64)


def load_bind_book(bcfg: dict, repo_root) -> dict:
    """org_id → 職場情報(node/building/floor/cat + shift open/close)を1回だけ読む。

    束ね先は organizations 台帳の workplace_poi(pool 生成元と同一)。会社 + 学校の両方。"""
    path = Path(bcfg["book"])
    if not path.is_absolute():
        path = Path(repo_root) / path
    data = json.loads(path.read_text(encoding="utf-8"))
    book: dict[str, dict] = {}
    for org in list(data.get("companies", [])) + list(data.get("schools", [])):
        wp = org.get("workplace_poi") or {}
        sp = org.get("shift_pattern") or {}
        tt = org.get("timetable") or {}
        book[str(org["id"])] = {
            "node": wp.get("node"), "building": wp.get("building"),
            "floor": wp.get("floor"), "poi_id": wp.get("poi_id"), "cat": wp.get("cat"),
            "open": sp.get("open") or tt.get("start"), "close": sp.get("close"),
        }
    return book


def _resolve_building(city, node: str, poi_building=None, poi_id=None) -> tuple[str | None, int]:
    """node に紐づく建物 id と階(POI 自身の building を優先、無ければ node の建物、無ければ None)。

    organizations._workplace_building と同型(路面の職場は building=None=その場勤務)。"""
    if poi_building and city.has_building(poi_building):
        return str(poi_building), 1
    for p in city.pois_at_node(node):                       # POI 自身が建物内なら its building
        if poi_id is not None and p.get("id") == poi_id and p.get("building") \
                and city.has_building(p["building"]):
            return str(p["building"]), int(p.get("floor") or 1) or 1
    blds = city.buildings_at(node)                          # 無ければ node の入口を持つ建物
    return (str(blds[0]["id"]), 1) if blds else (None, 0)


def _window(bcfg: dict, entry: dict | None, record: dict) -> tuple[int, int]:
    """勤務窓(open, close 分)。台帳 shift → record.shift_pattern → config 既定 の順に後退。"""
    sp = record.get("shift_pattern") or {}
    op = (entry or {}).get("open") or sp.get("open")
    cl = (entry or {}).get("close") or sp.get("close")
    o = _hhmm_to_min(op, _hhmm_to_min(bcfg["default_open"], 9 * 60))
    c = _hhmm_to_min(cl, _hhmm_to_min(bcfg["default_close"], 18 * 60))
    if c <= o:                                              # 逆転の保険(閉<=開)は既定8時間窓に補正
        c = o + 8 * 60
    return o, c


def _resolve_node(record: dict, city, book: dict, bcfg: dict, key: str):
    """束ね先を決める。(node, building, floor, entry) を返す(束ね不能なら None)。

    ① 台帳直束ね: record.org_id → book[org_id].node が現行地図にあれば採用(最精度)。
    ② 決定論マッチ: 台帳ノードが地図に無い/org_id 無しのとき、POI カテゴリ(台帳 cat →
       occupation 写像)で city.pois_by_cat から安定ハッシュ(pool_pid の純関数)で1つ選ぶ。"""
    org_id = record.get("org_id")
    entry = book.get(str(org_id)) if org_id else None
    if entry and entry.get("node") and entry["node"] in city.graph:      # ① 台帳直束ね
        bld, floor = _resolve_building(city, entry["node"], entry.get("building"),
                                       entry.get("poi_id"))
        fl = int(entry.get("floor") or 0) or floor
        return entry["node"], bld, fl, entry
    if not bcfg["poi_match_fallback"]:
        return None
    cat = (entry or {}).get("cat")                                       # ② 決定論マッチ
    if not cat:
        occ = str(record.get("occupation") or record.get("role") or "")
        cat = bcfg["occ_cat"].get(occ)
    if not cat:
        return None
    pois = city.pois_by_cat(str(cat))
    if not pois:
        return None
    idx = int(_stable_uniform(bcfg["seed"], f"{cat}\x1f{key}") * len(pois)) % len(pois)
    poi = pois[idx]
    bld, floor = _resolve_building(city, poi["node"], poi.get("building"), poi.get("id"))
    fl = int(poi.get("floor") or 0) or floor
    return poi["node"], bld, fl, entry


def bind_workplace(agent, record: dict, city, book: dict, bcfg: dict) -> tuple[bool, bool]:
    """勤務中に通う職場 POI を work_node へ束ねる(決定論・乱数ゼロ・LLM ゼロ)。

    戻り値: (had_before, has_after)。既に work_node を持つ個体は rebind_bound=false なら不変。
    work_start_min<0 のときだけ勤務窓を補う(既存の勤務窓・出勤 routine は壊さない)。付与規則は
    (pool_pid, 固定属性)の純関数=run.seed 非依存=hydrate 再入で同一 work_node。"""
    had = bool(getattr(agent, "work_node", "")) and int(getattr(agent, "work_start_min", -1)) >= 0
    if had and not bcfg["rebind_bound"]:
        return True, True
    key = str(getattr(agent, "pool_pid", "") or record.get("id", ""))
    res = _resolve_node(record, city, book, bcfg, key)
    if res is None:
        has = bool(getattr(agent, "work_node", "")) and int(getattr(agent, "work_start_min", -1)) >= 0
        return had, has
    node, bld, floor, entry = res
    agent.work_node = node
    if bld is not None:                                    # 建物内の職場: 入館して勤務
        agent.work_building = bld
        agent.work_floor = int(floor) if int(floor) >= 1 else 1
    if int(getattr(agent, "work_start_min", -1)) < 0:      # 勤務窓が無い個体にだけ窓を補う
        o, c = _window(bcfg, entry, record)
        agent.work_start_min = o
        agent.work_end_min = c
    return had, True
