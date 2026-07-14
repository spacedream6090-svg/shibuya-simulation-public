"""生活の自己決定 P2(D3 未実装棚卸し。docs/plans/agent-freedom-plan.md #6-#10。既定 全 OFF)。

「生活の自己決定はほぼ全てルール強制(拒否権なし)」という監査結論への最小・正直な応答。
P1(quit_job/apply_job/move_to/skip_work/stay_up)に続く「選択の幅」の層。5項目を個別 bool で切る:

  #6 move_home  空き住戸への転居(敷金=現金障壁)。地価=ノード人気の内生化はまだしない。
  #7 buy        発火時の消費意思(既存 _spend で即時支出。非発火の buy 抽選はフォールバックで残す)。
  #8 study      学校/図書館で聴講(記録+滞在のみ)。★賃金/生産性への経路は Skill 討議後(ユーザー決定)。
  #9 partnership 交際の申込/別れ(既存 relations の closeness と household の結合処理を再利用)。
  #10 deviance  無許可出店(=既存制度の違反を「選べる」)→ 既存 enforcement が近傍で摘発。

★ このファイルは src/society 直下(engine/cognition/actions/labeling/world の CHECKED_DIRS 外)。
  メニュー文言・空き住戸の決定論選定をここに閉じる=no-fingerprint 契約(engine が因子名を
  名指ししない)に触れない。裁定は engine(scheduler)側だが、いずれも money/closeness/物理位置
  (k 非依存の観測量)と config・新 stream のみを参照し、k(信念書き戻し)を発火判断に食わせない。

R1 呼数不変: メニューは既存の同じ1呼の応答 JSON の選択肢を増やすだけ(呼数不変)。新たな確率が
  要る move_home の住戸選定は新 stream "move_home" のみ(既存 draw 順に挿入しない=決定論とゴールデン
  の保護)。中立提示: 客観条件つきで置くだけ(tools.py の _equip_section と同じ流儀。促進・誘導なし)。
"""
from __future__ import annotations

import math

# buy(#7)の3カテゴリ → 既存の消費価格カテゴリへの写像(economy.prices のキー。すべて非0円)。
# buy の taxonomy(food/goods/leisure)はユーザー面の簡略。内部は既存の消費経路を再利用する。
BUY_PRICE_CAT = {"food": "food", "goods": "shop", "leisure": "cafe"}

# 近傍メニューの提示条件に使う POI カテゴリ(客観・k 非依存)。
COMMERCIAL_CATS = frozenset({"food", "shop", "nightlife"})   # buy(#7)= 商業施設の近く
STUDY_CATS = frozenset({"school", "education"})               # study(#8)= 学校・図書館の近く

# メニュー節の目印(mock の分岐マーカーと共有。中立文=促進・誘導の語を含まない)。
_INTRO = "生活の選択(条件を満たすとき使える。使っても使わなくてもよい):"


def menu(agent, *, cfg: dict, venture_cost: float, near_commercial: bool,
         near_school: bool, free_time: bool, has_company: bool,
         accounts_on: bool) -> str | None:
    """P2 の中立メニューを組み立てる(決定論・乱数なし・k 非依存)。

    条件を満たす項目だけを1行ずつ置く(客観条件は括弧で明示)。1件も無ければ None
    (=build_prompt は1行も足さない=OFF/非該当はバイト一致)。"""
    lines: list[str] = []
    money = int(getattr(agent, "money", 0) or 0)
    account = int(getattr(agent, "account", 0) or 0) if accounts_on else 0
    total = money + account
    if cfg["move_home"] and free_time:                 # #6 覚醒・自由時間
        dep = int(cfg["deposit"])
        status = f"利用可(所持{total}円)" if total >= dep else f"不足(所持{total}円)"
        lines.append(
            '- 住まいを移す。空いている住戸に引っ越す。'
            '形式 {"action":"move_home","area":"移りたい地域(任意)"}'
            f'(条件: 敷金{dep}円以上。現在: {status})')
    if cfg["buy"] and near_commercial:                 # #7 商業POI近傍
        lines.append(
            '- 欲しい物を買う。'
            '形式 {"action":"buy","cat":"food/goods/leisure のどれか"}'
            '(条件: 商業施設の近く)')
    if cfg["study"] and near_school:                   # #8 学校/図書館系POI近傍
        lines.append(
            '- 学校や図書館で学ぶ。'
            '形式 {"action":"study","topic":"学ぶ内容(任意)"}'
            '(条件: 学校・図書館の近く)')
    if cfg["partnership"] and has_company:             # #9 相手が近くにいる
        if getattr(agent, "partner_id", None) is not None:
            lines.append(
                '- 恋人と別れる。形式 {"action":"break_up"}(条件: 交際中)')
        else:
            lines.append(
                '- 親しい相手に交際を申し込む。'
                '形式 {"action":"propose_partnership","to":"相手の名前"}'
                '(条件: 相手が近くにいて、十分に親しい)')
    if cfg["deviance"] and total >= int(venture_cost):  # #10 出店できる所持金があるとき
        lines.append(
            '- 許可を取らずに店を開くこともできる(取り締まりの対象になる)。'
            '形式 {"action":"open_venture","name":"…","offer":"…","permit":false}')
    if not lines:
        return None
    return "\n".join((_INTRO, *lines))


def pick_home(sim, agent, area, rng):
    """空き住戸(他エージェントの home でない residential 建物)を決定論で1つ選ぶ(#6)。

    候補=誰の home_building にもなっていない住宅系建物。area 指定があれば名前が部分一致する
    ものを優先(=近傍/地域優先の近似)、その中で現在地からの距離昇順+id。近い上位から新 stream
    "move_home" で1つ引く(=新たな確率は新 stream のみ)。候補が無ければ None。"""
    occupied = {getattr(a, "home_building", "") for a in sim.agents
                if getattr(a, "home_building", "")}
    cands = [b for b in sim.city.residential_buildings
             if b["id"] not in occupied and b["id"] != agent.home_building]
    if not cands:
        return None
    area = str(area or "").strip()
    if area:                                            # 地域名の部分一致を優先(近傍/地域優先)
        pref = [b for b in cands
                if area in (b.get("name") or "")
                or area in sim.city.node_name(b["entrance"])]
        if pref:
            cands = pref
    ax, ay = agent.x, agent.y

    def _dist(b):
        cx, cy = b["centroid"]
        return math.hypot(cx - ax, cy - ay)

    cands.sort(key=lambda b: (_dist(b), b["id"]))       # 近傍優先(決定論)
    pool = cands[:5]                                    # 近い上位から選ぶ(過度な遠距離を避ける)
    return pool[int(rng.integers(len(pool)))]


def partner_threshold(sim, cfg: dict) -> float:
    """交際成立の closeness 閾値(#9)。household 有効ならその partner_closeness、無ければ p2 の後退値。"""
    hh = getattr(sim, "householdcfg", None)
    if hh and hh.get("enabled"):
        return float(hh.get("partner_closeness", cfg["partner_closeness"]))
    return float(cfg["partner_closeness"])
