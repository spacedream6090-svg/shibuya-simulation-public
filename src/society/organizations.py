"""組織(職場・学校)台帳のエンジン接続(Wave2-1。既定 OFF)。

ユーザー要望 2026-07-06: 「現実の職場や学校が何を教え・何を提供しているかをリサーチし、
シミュレーションにも適用してほしい。経済活動や学術研究の仕組みの再現が目標。具体的な会社の
実装は社会シミュレーションに必要な要素」。

台帳(data/organizations_shibuya.json)は渋谷の産業構成に統計準拠した**架空**の組織
(R17: 実在の企業名・学校名は使わない)。配属(data/org_assignments_*.json)は名簿
(personas_*.json)の並び順に決定論で対応する。統合仕様は
docs/design-candidates/org-integration-spec.md(§0-§6)。

原則:
- **既定 OFF**(enabled=false)。OFF 時は attach されず、agent に org_* 属性が付かない
  = プロンプト・イベント列とも従来とバイト一致。
- 決定論: 本モジュールは乱数を一切引かない(配属は台帳で確定済み、産出・教科は日付巡回)。
- R1: LLM 呼び出しを増やさない(所属はプロンプトへの1行注入のみ)。所属は職業・名簿由来で
  k 条件と無関係。
- R9: 配属・所属文字列は職業・年齢・名簿のみ由来(性格特性を読まない)。
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def build_orgs_cfg(raw: dict | None) -> dict:
    """conf の organizations ブロックを正準化(既定 OFF=現行挙動と完全同一)。"""
    raw = dict(raw or {})
    return {
        "enabled": bool(raw.get("enabled", False)),
        "book": str(raw.get("book", "data/organizations_shibuya.json")),
        # 組織会計(org_ledger)の売上係数: 勤務完遂1回 = 日給×margin を収益の代理指標として積む
        # (市場メカニズムは B 段。仕様 §1.3「集計だけ」の粒度)。
        "revenue_margin": float(raw.get("revenue_margin", 1.5)),
        # 通勤・通学の物理 POI 束縛(第7バッチ 2026-07-07。既定 OFF)。true のとき配属者の
        # work_node/work_building を割当 org の workplace_poi へ上書きする(決定論・乱数なし)。
        # false(既定)なら上書きしない=既存 test_organizations はバイト一致で不変。
        "commute_to_poi": bool(raw.get("commute_to_poi", False)),
        # 配属ファイルの明示指定(第10バッチ: 広域地図移行)。None(既定)=名簿名からの導出
        # (personas_<tag> → org_assignments_<tag>)=従来と完全同一。広域地図用の台帳
        # (organizations_shibuya_wide.json)は workplace_poi のノードが広域地図由来なので、
        # 対になる配属ファイル(org_assignments_<tag>_wide.json)をここで指す。
        "assignments": (str(raw["assignments"]) if raw.get("assignments") else None),
    }


def _resolve(path_str: str) -> Path:
    p = Path(str(path_str))
    return p if p.is_absolute() else REPO_ROOT / p


def load_book(cfg: dict) -> dict[str, dict]:
    """台帳(会社+学校)を id → org(dict) で返す。"""
    data = json.loads(_resolve(cfg["book"]).read_text(encoding="utf-8"))
    book: dict[str, dict] = {}
    for org in list(data.get("companies", [])) + list(data.get("schools", [])):
        book[str(org["id"])] = org
    return book


def assignments_for(personas_file: str | None,
                    override: str | None = None) -> tuple[list[dict], int] | None:
    """名簿ファイル → (配属リスト, 名簿の人数)。対応する配属ファイルが無ければ None。

    対応規則: data/personas_<tag>.json → data/org_assignments_<tag>.json
    (例: personas_80 → org_assignments_80、personas_100_inflow → org_assignments_100_inflow)。
    override(organizations.assignments。第10バッチ)が与えられればそれを優先する
    (広域地図用の配属など、導出規則と別のファイルを使うため)。既定 None=従来と完全同一。
    名簿人数は agent_id % len(roster) の巡回対応(simulation.py の生成規則)に使う。
    """
    if not personas_file:
        return None
    if override:
        asn_path = _resolve(override)
        if not asn_path.exists():
            return None
    else:
        stem = Path(str(personas_file)).stem
        if not stem.startswith("personas_"):
            return None
        tag = stem[len("personas_"):]
        asn_path = _resolve(f"data/org_assignments_{tag}.json")
        if not asn_path.exists():
            return None
    assignments = json.loads(asn_path.read_text(encoding="utf-8")).get("assignments", [])
    roster = json.loads(_resolve(str(personas_file)).read_text(encoding="utf-8"))["personas"]
    return assignments, len(roster)


def org_line(org: dict, role: str) -> str:
    """プロンプトへ注入する所属1行(仕様 §2 の書式。中立・架空名のみ)。"""
    name = str(org.get("name", ""))
    if role == "学生":
        fields = (org.get("subjects") or org.get("faculties")
                  or org.get("lecture_fields") or [])
        detail = str(org.get("school_type", ""))
        if fields:
            detail = f"{detail}・{fields[0]}など" if detail else f"{fields[0]}など"
        return f"学校: {name}({detail})"
    detail = str(org.get("sector_detail") or org.get("industry", ""))
    return f"職場: {name}({detail}・{role})" if role else f"職場: {name}({detail})"


def _bind_workplace(agent, org: dict, city) -> None:
    """配属 org の workplace_poi へ実際に通わせる(work_node/work_building を上書き)。

    決定論・乱数なし(仕様: build から確定)。workplace_poi.node が地図に無い org は
    上書きしない(既存の work をそのまま=経路探索の安全)。node に建物があれば
    work_building も設定(POI の building 参照 → 無ければ node の建物)。無ければ
    work_building は従来値のまま(路面の職場として routine が扱う)。学生も同様に学校 POI へ。
    """
    wp = org.get("workplace_poi") or {}
    node = wp.get("node")
    if not node or node not in city.graph:
        return
    agent.work_node = node
    bld = _workplace_building(city, wp, node)
    if bld is not None:
        agent.work_building = bld


def _workplace_building(city, wp: dict, node: str) -> str | None:
    """workplace_poi のノードに紐づく建物 id(無ければ None)。決定論・乱数なし。"""
    poi_id = wp.get("poi_id")
    for p in city.pois_at_node(node):            # POI 自身が建物内なら its building
        if p.get("id") == poi_id and p.get("building") \
                and city.has_building(p["building"]):
            return str(p["building"])
    blds = city.buildings_at(node)               # 無ければ node の入口を持つ建物
    return str(blds[0]["id"]) if blds else None


def attach(agents, book: dict[str, dict], personas_file: str | None, *,
           city=None, commute_to_poi: bool = False,
           assignments_path: str | None = None) -> int:
    """各 agent に org_id / org_role / org_line を付与(既存フィールド不変)。戻り値=配属数。

    agent_index は名簿の並び順 = build_agent の生成順(仕様 §0)。n_agents が名簿人数を
    超えるときは名簿と同じ巡回規則(agent_id % 名簿人数)で対応づける。未配属
    (無職・配属ファイルに無い個体)は何も付けない=完全に従来挙動。

    commute_to_poi=true(かつ city 渡し)のときだけ、配属者の work_node/work_building を
    割当 org の workplace_poi へ上書きする(第7バッチ)。false(既定)なら上書きしない=
    既存 test_organizations はプロンプト注入のみで不変=バイト一致を維持。
    """
    found = assignments_for(personas_file, override=assignments_path)
    if found is None:
        return 0
    assignments, n_roster = found
    by_index = {int(a["agent_index"]): a for a in assignments}
    attached = 0
    for agent in agents:
        asn = by_index.get(agent.id % max(n_roster, 1))
        if asn is None:
            continue
        org = book.get(str(asn["org_id"]))
        if org is None:
            continue
        agent.org_id = str(asn["org_id"])
        agent.org_role = str(asn.get("role", ""))
        agent.org_line = org_line(org, agent.org_role)
        if commute_to_poi and city is not None:      # 物理 POI 束縛(既定 OFF)
            _bind_workplace(agent, org, city)
        attached += 1
    return attached


# ---------------------------------------------------------------- キャリア転換(Wave G5。既定 OFF)
# 失業/求職/転職/起業転換の台帳操作。src/society 直下=engine の no-fingerprint 契約の外(ただし本
# モジュールは因子語を扱わない=grievance は scheduler が factors hook 経由で処理する)。決定論・
# 乱数なし(再配属先の抽選は呼び出し側の新 stream "career" が担い、ここは台帳操作だけを行う)。
def is_employee(agent) -> bool:
    """被雇用者か(org 配属の非学生)。career の失業/転職の起点資格(tools._is_employee と同義)。"""
    return (bool(getattr(agent, "org_id", None))
            and getattr(agent, "org_role", "") != "学生")


def is_laid_off(agent) -> bool:
    """career で職を失った求職中の状態か(求職=rehire の対象)。学生/自営/未配属は該当しない。"""
    return bool(getattr(agent, "_laid_off", False))


def employer_ids(book: dict[str, dict] | None) -> list[str]:
    """再配属候補=会社(学校を除く)の org id を決定論順(sorted)で返す。台帳が無ければ空。"""
    if not book:
        return []
    return sorted(str(oid) for oid, org in book.items()
                  if not org.get("school_type"))


def _reassign_role(prev_role: str) -> str:
    """再配属時の役割。退避した本人の役割を引き継ぐ(職能は職場が変わっても保つ=最小近似)。"""
    return str(prev_role or "")


def lay_off(agent) -> str | None:
    """被雇用者の職を失わせる: org と本業勤務窓をクリアし収入を断つ。戻り値=失った org_id。

    復職(rehire)で勤務窓・賃金を復帰させるため、直前の雇用スナップショットを退避する。work_start_min
    を -1 にすると in_work_window が False=勤務に行かない(自由時間 EPR へ)。wage=0 で _settle_work
    が支給しない。org_line="" でプロンプトの所属行が消える(falsy=注入なし)。"""
    from_org = getattr(agent, "org_id", None)
    agent._career_prev = {
        "org_role": getattr(agent, "org_role", ""),
        "work_name": agent.work_name, "work_node": agent.work_node,
        "work_building": agent.work_building, "work_floor": agent.work_floor,
        "work_start_min": agent.work_start_min, "work_end_min": agent.work_end_min,
        "wage": agent.wage,
    }
    agent.org_id = None
    agent.org_role = ""
    agent.org_line = ""
    agent.work_name = ""
    agent.work_node = ""
    agent.work_building = ""
    agent.work_floor = 0
    agent.work_start_min = -1
    agent.work_end_min = 0
    agent.wage = 0.0
    agent._laid_off = True
    return from_org


def _attach(agent, org: dict, role: str, city, commute_to_poi: bool) -> None:
    """agent を org へ配属(org_id/role/line)。commute_to_poi=true なら work_node/building も上書き。"""
    agent.org_id = str(org["id"])
    agent.org_role = role
    agent.org_line = org_line(org, role)
    if commute_to_poi and city is not None:
        _bind_workplace(agent, org, city)


def rehire(agent, org: dict, *, city=None, commute_to_poi: bool = False) -> None:
    """求職中の agent を新 org へ再就職させる(勤務窓・収入を復帰)。

    退避スナップショット(_career_prev)から勤務窓・賃金を復帰する。commute_to_poi=true なら
    その後に新 org の workplace_poi へ work_node を束ねる(実際にそこへ通う)。"""
    prev = getattr(agent, "_career_prev", None) or {}
    agent.work_name = str(prev.get("work_name", "") or org.get("name", ""))
    agent.work_node = str(prev.get("work_node", "") or "")
    agent.work_building = str(prev.get("work_building", "") or "")
    agent.work_floor = int(prev.get("work_floor", 0) or 0)
    ws = int(prev.get("work_start_min", -1))
    agent.work_start_min = ws if ws >= 0 else 9 * 60
    we = int(prev.get("work_end_min", 0))
    agent.work_end_min = we if we > 0 else 18 * 60
    agent.wage = float(prev.get("wage", 0.0))
    _attach(agent, org, _reassign_role(prev.get("org_role", "")), city, commute_to_poi)
    agent._laid_off = False
    agent._career_prev = None


def switch_org(agent, org: dict, *, city=None, commute_to_poi: bool = False) -> str | None:
    """被雇用者を別 org へ転職させる(職場・所属が変わる)。戻り値=元の org_id。

    勤務窓・役割は保つ(職能は変わらない=最小近似)。commute_to_poi=true なら work_node を新 org の
    workplace_poi へ束ね直す(勤務地が物理的に変わる)。"""
    from_org = getattr(agent, "org_id", None)
    _attach(agent, org, getattr(agent, "org_role", ""), city, commute_to_poi)
    return from_org


def go_fulltime_venture(agent, venture: dict) -> None:
    """本業を辞め、出店(open_venture の屋台)を本業にする完全転換(Wave G5・起業転換)。

    org・本業賃金をクリアし、勤務地を venture ノードへ移す(収入は venture 売上に一本化)。屋台を日中に
    開ける勤務窓を与える=routine が work_node(=屋台ノード)へ通わせ、路面の職場として stay する。
    _settle_work は work_building="" かつ wage=0 のため本業賃金を一切支給しない(二重収入なし)。"""
    agent.org_id = None
    agent.org_role = ""
    agent.org_line = ""
    agent.wage = 0.0
    agent.work_name = str(venture.get("name", ""))
    agent.work_node = str(venture.get("node", ""))
    agent.work_building = ""
    agent.work_floor = 0
    agent.work_start_min = 9 * 60
    agent.work_end_min = 21 * 60
    agent._laid_off = False


def daily_output(org: dict, day: int) -> tuple[str, str] | None:
    """勤務完遂1回ぶんの産出(output, kind)を決定論で選ぶ(日替わり巡回=非乱数)。

    output_kinds / products_services を持たない組織(固定拠点の無い自営 umbrella 等)は
    None(産出なし。仕様 §1.3)。
    """
    kinds = org.get("output_kinds") or []
    services = org.get("products_services") or []
    if not kinds or not services:
        return None
    return str(services[day % len(services)]), str(kinds[0])


def daily_subject(org: dict, day: int) -> str | None:
    """登校完遂1回ぶんの教科・講義分野を決定論で選ぶ(曜日巡回=非乱数。仕様 §1.2)。"""
    pool = (org.get("subjects") or org.get("faculties")
            or org.get("lecture_fields") or [])
    if not pool:
        return None
    return str(pool[day % len(pool)])
