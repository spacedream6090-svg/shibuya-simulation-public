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
    # 第106: services.py の受給 _spend は cat="service" で出る(services.py:30 が
    # 「work.serve_by_cat と対の seam」と明記)。センサス較正台帳は service 職場 1,972 社を
    # 持つため、この行が無いと構造的に無人接客になる。work.service 既定 OFF=golden 不変。
    "service": "窓口・施術の接客",
}

# ダイジェストで「業務が多かった」と言い換える件数の閾値(客観記述の粒度)。
_MANY_THRESHOLD = 8

# 職場束ね直し(bind_workplace。既定 OFF)。pool 経路の L2/L3 は occupation(role)が
# persona._pick_workplace の _WORK_CAT に載らないと work_node を持たない=接客(serve)/産出
# (org_output)の帰属から漏れる。台帳 organizations の workplace_poi.node へ org_id で直束ねし、
# 台帳ノードが現行地図に無い時は POI カテゴリ + 安定ハッシュで決定論マッチする(organizations.
# commute_to_poi の pool 版)。純関数(pool_pid・固定属性のみ)=run.seed 非依存=hydrate 再入不変。
#
# 第100バッチ(P3b 前提)の是正 3 点【実測 2026-08-09・mock ≤24step】:
#  (1) **台帳ノードの実在検査**: 台帳は「産業×規模帯 → 建物」の分布で置いた合成値なので、
#      workplace_poi.node は地図の POI 実体と食い違うものが多い(wide_v7: 同カテゴリ POI を持つ
#      node に居るのは food 401/1,650・shop 986/3,850・office 498/4,015・service 78/1,485 社)。
#      旧実装は node が graph にありさえすれば採用していたため、**客が絶対に来ない場所に店員が
#      立ち**、束ね ON でも unstaffed 96%・不在 serve の 100% が「誰の職場でもない node」だった。
#      → 同カテゴリ POI を持たない台帳ノードは決定論 POI マッチへ落として実在の店へ着地させる。
#  (2) **母数の是正**: 旧 bind_eligible は org_id 保有者だけを数えていたため、地図に対応 POI
#      カテゴリの無い L5(駅員/運転士/警察官/配信者/議員)が統計の外へ落ち、n_unbound_after=0 が
#      実態より良く見えていた。role 保有層と _WORK_CAT 就業者を母数に入れ、束ねられないものは
#      理由タグつきで数える(推測で職業→カテゴリ写像を発明しない)。
#  (3) **写像の再利用**: occupation→カテゴリは persona._WORK_CAT、地図語彙差(education→school)は
#      day_plan.MAP_FALLBACK_CATS。どちらも既存表で、この層で新しい対応表を作らない。
# ★残る制約(束ねでは閉じない): serve の成立には共在(勤務窓に work_node へ在場)が要る。実測では
#   在場密度が支配的で、cap=300 では unstaffed 90.7%→89.8% とほぼ動かず、cap=1,500 で 84.0%→65.8%。
_BIND_DEFAULT_BOOK = "data/organizations_shibuya_wide11k.json"


def build_cfg(raw: dict | None) -> dict:
    """conf の work ブロックを正準化(既定 OFF=現行挙動と完全同一)。"""
    raw = dict(raw or {})
    svc = dict(raw.get("service", {}) or {})
    serve = svc.get("serve_by_cat")
    serve = dict(serve) if serve else dict(DEFAULT_SERVE_BY_CAT)
    off = dict(svc.get("office", {}) or {})
    bind = dict(raw.get("bind_workplace", {}) or {})
    ledger = dict(svc.get("ledger", {}) or {})
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
        # 会社観測データ層 B4(既定 OFF=バイト一致)。ON 時のみ serve に org_id/floor を付ける。
        # 解決順: スタッフ経由(帰属スタッフの org_id)→ unstaffed は node→org が一意のときのみ →
        # 多義ノード null(推測せず unknown を正直開示)。(建物,階)絞り込みは indoor.enabled かつ本 ON 時のみ。
        "indoor_fields": bool(svc.get("indoor_fields", False)),
        "office": {
            "enabled": bool(off.get("enabled", True)),
            # オフィス系職場と見なす職場 POI カテゴリ(既定 office)。
            "poi_cats": [str(c) for c in (off.get("poi_cats") or ["office"])],
            "base_weight": float(off.get("base_weight", 1.0)),
            # occupation/role → 産出重み(空=全員 base_weight=出勤者数に比例)。
            "role_weights": {str(k): float(v)
                             for k, v in (off.get("role_weights") or {}).items()},
            # 会社観測データ層 B4(既定 OFF=バイト一致)。ON 時: org_output を org_id 単位で記録
            # (同居複数社の分解)。indoor.enabled のとき値は「頭数×重み」でなくミクロ在席分
            # (attendance_zones の職務区画に居た step 数×10分)。basis フィールドで式を自己記述。
            "by_org": bool(off.get("by_org", False)),
            # ミクロ在席分の対象区画型(desk 等の職務区画。break/rest は在席に数えない)。
            "attendance_zones": [str(z) for z in
                                 (off.get("attendance_zones") or ["desk", "meeting"])],
        },
        # 会社観測データ層 B4: org_ledger サイドカー(runs/<run>/org_ledger.parquet)を書くか
        # (既定 OFF=バイト一致・ファイル不在)。日次1行/社(活動があった社のみ)。
        "ledger": {
            "enabled": bool(ledger.get("enabled", False)),
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
            # 日跨ぎシフト(第101 III-1「夜間開放」)。**conf の work ブロックには無い**:
            # 値は world.night_economy から night.wire_work が 1 回だけ注入する(既定 False)。
            # False のとき _window は従来コード(閉<=開 → open+8h の丸め)を文字どおり通す。
            "midnight_shift": bool(bind.get("midnight_shift", False)),
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
# 写像表は**既存のものだけを再利用する**(この層で新しい職業→カテゴリ表を発明しない):
#   - occupation/role → 勤務 POI カテゴリ = persona._WORK_CAT(職場割当の唯一の源)
#   - POI カテゴリの地図語彙差(education→school)= day_plan.MAP_FALLBACK_CATS
# どちらも遅延 import(work.py は engine から module 直下で読まれるので循環を作らない)。
def _occ_cat_table() -> dict:
    from .agents.persona import _WORK_CAT
    return _WORK_CAT


def _cat_aliases(cat) -> tuple[str, ...]:
    """POI カテゴリ + 地図語彙の追加照会先(day_plan.MAP_FALLBACK_CATS と同一の表)。"""
    from .cognition.day_plan import MAP_FALLBACK_CATS
    c = str(cat)
    return (c,) + tuple(MAP_FALLBACK_CATS.get(c, ()))


def _pois_for_cat(city, cat) -> list[dict]:
    """カテゴリの POI 群(空なら追加照会先を順に見る)。決定論・乱数なし。"""
    for c in _cat_aliases(cat):
        pois = city.pois_by_cat(c)
        if pois:
            return list(pois)
    return []


def _node_is_workplace_of(city, node: str, cat) -> bool:
    """その node が現行地図で「その業種の職場」か(同カテゴリの POI を持つか)。

    cat 不明の台帳エントリは判定材料が無いので True(従来どおり node をそのまま信じる)。"""
    if not cat:
        return True
    alias = set(_cat_aliases(cat))
    return any(p.get("cat") in alias for p in city.pois_at_node(node))


def _cat_for(record: dict, entry: dict | None, bcfg: dict) -> str | None:
    """束ね先の POI カテゴリ。台帳 cat → occ_cat(config)→ persona._WORK_CAT の順に後退。

    どれにも当たらない層(駅員・警察官・タクシー運転手・議員 等=地図に対応 POI カテゴリが
    そもそも存在しない)は None=束ね不能として**正直に数える**(推測で写像を作らない)。"""
    cat = (entry or {}).get("cat")
    if cat:
        return str(cat)
    occ = str(record.get("occupation") or record.get("role") or "")
    cat = bcfg["occ_cat"].get(occ) or _occ_cat_table().get(occ)
    return str(cat) if cat else None


def bind_eligible(record: dict, bcfg: dict) -> bool:
    """束ね対象か(=勤務地を持つべき層)。coverage 統計の母数はここで決まる。

    対象: ① org 所属(org_id。L2 従業者/L3 学生)② occ_cat(config)に写像がある層
          ③ **role を持つ層**(L5 duty の駅員/運転士/警察官/配信者・議員。org_id は無いが
             「渋谷で働いている」ペルソナ)④ occupation が persona._WORK_CAT に載る就業者。
    対象外: L1 の無職/フリーランス/バンドマン等(_WORK_CAT が None=決まった職場を持たない)と
            L4 来街者(role も org_id も持たない)。
    ★③④ を母数に入れるのは意図的で、「束ねられなかった」ことを n_unbound_after として
      可視化するため(旧実装は org_id 保有者だけを数えていたので、地図に対応カテゴリの無い
      L5 が丸ごと統計の外に落ち、coverage が実態より良く見えていた)。"""
    if record.get("org_id"):
        return True
    occ = str(record.get("occupation") or record.get("role") or "")
    if occ in bcfg.get("occ_cat", {}):
        return True
    if str(record.get("role") or ""):
        return True
    return _occ_cat_table().get(occ) is not None


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


def _window(bcfg: dict, entry: dict | None, record: dict) -> tuple[int, int, bool]:
    """勤務窓 (open, close, 日跨ぎか)[分]。台帳 shift → record.shift_pattern → config 既定 の順に後退。

    ★第101 III-1「夜間開放」。``midnight_shift``(= world.night_economy が ON のときだけ
      night.wire_work が立てる)のとき、**close < open は「翌朝まで」の意味**として受ける
      (22:00→06:00 の夜勤が表現できるようになる)。close == open は長さゼロの退化なので
      夜勤とは読まず従来どおり丸める。
    ★OFF のときは**従来の 2 行を文字どおり通す**(閉<=開 → open+8h の丸め)= バイト一致。
      監査の指摘どおり、この丸めが「夜勤が原理的に作れない」唯一の原因だった。"""
    sp = record.get("shift_pattern") or {}
    op = (entry or {}).get("open") or sp.get("open")
    cl = (entry or {}).get("close") or sp.get("close")
    o = _hhmm_to_min(op, _hhmm_to_min(bcfg["default_open"], 9 * 60))
    c = _hhmm_to_min(cl, _hhmm_to_min(bcfg["default_close"], 18 * 60))
    if bcfg.get("midnight_shift") and c < o:                # 日跨ぎ(夜勤): 翌朝まで働く
        return o, c, True
    if c <= o:                                              # 逆転の保険(閉<=開)は既定8時間窓に補正
        c = o + 8 * 60
    return o, c, False


def _resolve_node(record: dict, city, book: dict, bcfg: dict, key: str):
    """束ね先を決める。(node, building, floor, entry, how) を返す(node=None なら how=不能理由)。

    ① 台帳直束ね: record.org_id → book[org_id].node。**現行地図でその node が実際にその業種の
       職場である**(同カテゴリの POI を持つ)ときだけ採用する。
       ★この追加条件が第100バッチの是正点。台帳(build_orgs.py --dist)は組織を「産業×規模帯 →
       建物」の分布で置いた合成値なので、workplace_poi.node は**地図の POI 実体と一致しない**
       ものが多数ある(実測 2026-08-09・wide_v7: 同カテゴリ POI を持つ node に居るのは
       food 401/1,650・shop 986/3,850・office 498/4,015・service 78/1,485 社のみ)。
       食い違ったまま束ねると「客が絶対に行かない場所に店員が立つ」= serve が構造的に永久
       不在になる(実測: 束ね ON でも unstaffed 96%・不在 serve の 100% が『誰の職場でもない
       node』で発生)。なので食い違いは②へ落として**実在の POI へ着地**させる。
       poi_match_fallback=false のときは②を使わない設定なので、従来どおり node を無条件採用する。
    ② 決定論マッチ: POI カテゴリ(台帳 cat → occ_cat → persona._WORK_CAT)で
       city.pois_by_cat から安定ハッシュ(pool_pid の純関数)で1つ選ぶ。地図語彙の食い違いは
       day_plan.MAP_FALLBACK_CATS で追加照会する(education→school)。"""
    org_id = record.get("org_id")
    entry = book.get(str(org_id)) if org_id else None
    fb = bcfg["poi_match_fallback"]
    # ① 台帳直束ね(node が現行地図に在り、かつ実際にその業種の POI を持つときだけ)
    if entry and entry.get("node") and entry["node"] in city.graph \
            and (not fb or _node_is_workplace_of(city, entry["node"], entry.get("cat"))):
        bld, floor = _resolve_building(city, entry["node"], entry.get("building"),
                                       entry.get("poi_id"))
        fl = int(entry.get("floor") or 0) or floor
        return entry["node"], bld, fl, entry, "ledger"
    if not fb:
        return None, None, 0, entry, "no_ledger_node"
    cat = _cat_for(record, entry, bcfg)                                  # ② 決定論マッチ
    if not cat:
        return None, None, 0, entry, "no_category"
    pois = _pois_for_cat(city, cat)
    if not pois:
        return None, None, 0, entry, "no_poi_in_map"
    idx = int(_stable_uniform(bcfg["seed"], f"{cat}\x1f{key}") * len(pois)) % len(pois)
    poi = pois[idx]
    bld, floor = _resolve_building(city, poi["node"], poi.get("building"), poi.get("id"))
    fl = int(poi.get("floor") or 0) or floor
    return poi["node"], bld, fl, entry, "poi_match"


def bind_workplace(agent, record: dict, city, book: dict, bcfg: dict) -> tuple[bool, bool, str]:
    """勤務中に通う職場 POI を work_node へ束ねる(決定論・乱数ゼロ・LLM ゼロ)。

    戻り値: (had_before, has_after, how)。how = kept(既束ねを触らず)/ ledger(台帳直束ね)/
    poi_match(決定論 POI マッチ)/ no_category・no_poi_in_map・no_ledger_node(**束ね不能の理由**
    =正直に数えるためのタグ)。既に work_node を持つ個体は rebind_bound=false なら不変。
    work_start_min<0 のときだけ勤務窓を補う(既存の勤務窓・出勤 routine は壊さない)。付与規則は
    (pool_pid, 固定属性)の純関数=run.seed 非依存=hydrate 再入で同一 work_node。"""
    had = bool(getattr(agent, "work_node", "")) and int(getattr(agent, "work_start_min", -1)) >= 0
    if had and not bcfg["rebind_bound"]:
        return True, True, "kept"
    key = str(getattr(agent, "pool_pid", "") or record.get("id", ""))
    node, bld, floor, entry, how = _resolve_node(record, city, book, bcfg, key)
    if node is None:
        has = bool(getattr(agent, "work_node", "")) and int(getattr(agent, "work_start_min", -1)) >= 0
        return had, has, how
    agent.work_node = node
    if bld is not None:                                    # 建物内の職場: 入館して勤務
        agent.work_building = bld
        agent.work_floor = int(floor) if int(floor) >= 1 else 1
    if int(getattr(agent, "work_start_min", -1)) < 0:      # 勤務窓が無い個体にだけ窓を補う
        o, c, wraps = _window(bcfg, entry, record)
        agent.work_start_min = o
        agent.work_end_min = c
        if wraps:                                          # 日跨ぎ(夜勤)= 第101 III-1。
            # ★既定 OFF ではこの属性が**そもそも生えない**ので routine.in_work_window の
            #   getattr(既定 False)は従来式へ落ちる = L1 バイト一致(証明可能な no-op)。
            agent.work_wraps = True
            _fix_night_commute(agent, record, o, c)
    return had, True, how


def _fix_night_commute(agent, record: dict, o: int, c: int) -> None:
    """夜勤者の**通勤の向き**を昼勤前提から夜勤前提へ直す(第101 III-1。ON 経路のみ)。

    なぜ要るか(実測の穴): pool 経路の L2 は occupation が persona._WORK_CAT に載らないので
    ``_pick_workplace`` が None を返し、``persona.build_agent`` は流入通勤者の到着時刻を
    **既定の 08:30** にする(職場不明時の後退値)。勤務窓だけ夜勤にしても、本人は朝 8:30 に
    街へ入り、就寝(=帰宅)トリガに触れて即座に帰る = 夜勤者が夜に街へ居ない。
    ここで到着を「出勤 lead 分前」に、帰宅トリガが勤務窓の中に埋まっている場合だけ
    「退勤 30 分後」に直す(どちらも o / c の純関数 = 乱数ゼロ・決定論)。

    ★bedtime は**勤務窓に埋まっているときだけ**触る: 生成器(build_persona_pool)が既に
      退勤後へずらしてある個体の個体差(10 分刻みの散らし)を潰さないため。"""
    if getattr(agent, "commute", False):
        lead = max(0, int(record.get("arrival_lead_min", 40) or 40))
        agent.arrival_min = (int(o) - lead) % 1440
    m = int(getattr(agent, "bedtime_min", 0)) % 1440
    if m >= int(o) or m < int(c):                          # 帰宅トリガが勤務窓の中
        agent.bedtime_min = (int(c) + 30) % 1440
