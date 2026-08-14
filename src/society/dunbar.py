"""ダンバー認知枠(関係の維持コスト)+ 忘却/再会イベント 第75バッチ IDEA⑤
(2026-07-31。既定 OFF=現行挙動と完全同一)。

正典: docs/plans/hackathon1-ideas-implementation-plan.md §2 投入順5 /
      docs/research/hackathon1-analysis/ideas-shortlist.md §2 ⑦。

何を解く問題か
--------------
現状の関係機構(relations.py の closeness 蓄積+日次減衰、friends.py の Dunbar 入れ子層による
初期次数較正、joint/relations_endo の誘い選抜)には **「関係の維持には認知資源が要る」** という
制約が無い。closeness はいくらでも並列に積め、上限は初期グラフの次数較正(friend_graph)だけで、
実行時に増えた紐帯には何の枠も無い。Granovetter 1973 の弱い紐帯理論の要点は「**維持コストが
低いから弱い紐帯は多く持てる**」であって、維持コストを実装して初めて強い/弱い紐帯の非対称は
**内生的に**出る(第64バッチ invite_weak_tie_rate と相補的な指標になる)。

本 module が足す機構(全決定論・乱数ゼロ・LLM 呼ゼロ)
------------------------------------------------------
(1) **維持コスト**: 各 agent の「活性関係」(= closeness 台帳を持ち休眠していない相手)の総数に
    上限を置く。上限超過時は **最も弱い紐帯**(closeness 昇順・同値は相手 id 昇順)から
    **休眠**(dormant)にする。**削除ではない**: 台帳エントリ・count・last・記憶(episodes)は
    そのまま残り、休眠前の closeness を `dormant_closeness` に退避して closeness を 0 にする。
    → tier は closeness の純関数(relations.tier_of)なので、休眠の瞬間に tier 0 へ落ち、
      下流の消費者(プロンプトの間柄行 / joint の同伴候補 / weak_tie 枠 / partnership 閾値)が
      **一箇所も改変せずに**「維持していない関係」として扱う(単一の作用点)。
(2) **忘却/再会**: 休眠を `relation_dormant`、休眠関係の再活性化を `relation_rekindle` として
    L1 に出す(ON 時のみ)。再会時の closeness は **休眠前 × rekindle_discount** で復元する
    (「久しぶりに会った友達」の非対称性=関係は残っているが元の濃さではない)。
(3) 接触(会話・DM=note_contact 経路 / joint の同席)で活性を維持・再活性化する。

層(Dunbar 入れ子)と縮約の扱い — **正直な設計上の限界**
--------------------------------------------------------
conf は Dunbar の入れ子層 5/15/50/150(親友/友人/知人/知り合い)を素の値で持つが、
**休眠を起こす拘束は最外層(total)のみ**である。内側の層(close/friend/acquaint)は
比率の宣言と感度分析の軸として持ち、L2/イベントの解釈に使う。理由: 内側の層を拘束にすると
「親友が多すぎるので降格させる」= closeness の書き換えが要り、蓄積した交流履歴の情報を
**不可逆に失う**。休眠は退避で可逆(再会で戻る)だが降格は戻せないので実装しない。

`scale`(既定 0.34)の意味: Dunbar の絶対値は N≳150 の共同体を前提にしており、本シム
(N=20〜100)ではそのままでは上限が**原理的に効かない**(相手の総数が上限に届かない)。
層の比率(1:3:10:30)を保ったまま実効規模へ縮約する係数で、既定 0.34 は
close 2 / friend 5 / acquaint 17 / **total 51** = friend_graph の初期次数(親友3-5+友人7-12+
知人20 ≈ 最大37)に会話由来の新規紐帯が積み上がって**初めて拘束が効く**水準。
数値そのものは較正対象(**Lindenfors et al. 2021 (Biology Letters 17:20210158) は 150 という
値自体の統計的根拠を否定している**=感度分析の軸として conf 化する)。

`rekindle_discount`(既定 0.5)の根拠と限界: Levin, Walter & Murnighan 2011
(Organization Science 22(4):923-939 "Dormant Ties: The Value of Reconnecting")は、休眠した
紐帯を再接続すると **信頼と共有視点は保たれたまま新規性は弱い紐帯並みに高い**ことを実証した
(=関係は消えていないが元のままでもない)。ただし **復元率の数値は文献に無い**ので 0.5 は
「半分は残る」の保守的既定であり較正対象(正直な注記)。

弱い紐帯・誘い選抜(第64バッチ)との整合規則 — 一貫した1つの規則を選ぶ
--------------------------------------------------------------------
  - **休眠関係は closeness 降順の同伴候補経路から自動的に外れる**(tier 0 になるため。
    joint._companions の friend 経路は `_tier >= friend_tier` で絞るので**コード変更ゼロ**)。
  - **休眠関係は弱い紐帯探索枠(weak_tie_slots)でのみ再会し得る**(dunbar ON 時のみ、
    relations_endo.weak_tie_candidates の pool に tier=1 知人と並べて入れる)。根拠は
    Levin et al. 2011: 休眠紐帯は「新規性は弱い紐帯並み・信頼は強い紐帯並み」= 弱い紐帯枠と
    同じ探索の箱に入れるのが理論的に一貫する。
  - **明示的な意向(前日計画の with / 前日発話の明示キュー)は休眠でも通す**。名指しの意向は
    それ自体が再活性化の宣言であり、認知枠の淘汰より上位に置く(_resolve_with /
    invite_cue_partners は relations 台帳を名前で引くので、これも**コード変更ゼロ**)。
  - dunbar OFF では `dormant` キーが**一度も生えない**ので、上記のどの経路も 1 バイトも動かない
    (weak_tie の pool 拡張だけが明示的な分岐で、`enabled(sim)` で厳密にゲートしてある)。

性能(毎 step の全ペア走査をしない)+ 上限適用を日次に一本化した理由
--------------------------------------------------------------------
  - 接触時(増分): 触れた **1 件**の台帳エントリだけを見る(休眠なら再会。O(1))。
  - 日次バッチ: 日境界に 1 回だけ全 agent を走査して上限を課し、L2 スカラーを焼き直す。
  - L2 collect(毎 step): 状態に焼いた値を読むだけ(O(1))。休眠/再会の増減は発生時に ±1。
  ★ 上限の適用を**接触時にも**行う実装を最初に書いたが、closeness が近接する紐帯群
    (friend_graph 由来の同値帯)で「最弱」が接触のたびに入れ替わり、休眠/再会が振動した
    (mock 実測 300体2日で relation_dormant 5739 件)。認知枠は本来ゆっくりした構造的過程なので、
    **適用は日境界の 1 回だけ**に固定した。代償は「活性関係数が日内では一時的に上限を超え得る」
    (翌日の日境界で収束する)= 正直な限界として記録する。

R1(既定 OFF=バイト一致): enabled=false のとき本 module の全関数は即 return し、台帳に新キーを
足さず・イベント 0 件・L2 に列なし・`sim._dunbar_state` も生えない=L1/L2/乱数消費とも従来と
完全同一。ON でも **乱数 stream を 1 本も引かず・generate() を 1 本も足さない**(全決定論)。
resume==straight: 当日/累積タリー(sim._dunbar_state)は checkpoint.py が中央管理し(第62 joint /
第70 echo /第74 norm_state と同型)、休眠状態と直近接触 step(rel の dormant*/last_step)は
agents pickle に自然同梱される(world/pool.py の dehydrate/hydrate も dict 丸ごとで往復する)。

★ 配置: src/society 直下(engine/cognition/actions/labeling/world の CHECKED_DIRS 外)=
  no-fingerprint 契約に触れない層。engine 側の配線は中立な 3 呼び出し(接触前/接触後/日境界)のみ。
"""
from __future__ import annotations

from . import relations as _relations
from .observer.schema import Event

DEFAULTS = {
    "enabled": False,
    # ---- Dunbar 入れ子層の素の値(親友5 / 友人15 / 知人50 / 知り合い150)----
    # 休眠を起こす拘束は total のみ(冒頭 docstring「層と縮約の扱い」)。
    "close": 5,
    "friend": 15,
    "acquaint": 50,
    "total": 150,
    # 実効規模への縮約係数(比率 1:3:10:30 は保つ)。既定 0.34 → 2/5/17/51。
    "scale": 0.34,
    # 再会時に休眠前 closeness をこの率で復元(1.0=完全復元 / 0.0=ゼロから)。
    "rekindle_discount": 0.5,
    # 直近この日数以内に接触した相手は休眠にしない(=接触が維持そのもの。0=当日接触のみ保護)。
    "keep_days": 0,
    # 同居人は日々の共在で維持される紐帯=認知枠の淘汰から保護する(枠は消費するが落ちない)。
    "protect_housemates": True,
}

_BOOL_KEYS = ("enabled", "protect_housemates")
_INT_KEYS = ("close", "friend", "acquaint", "total", "keep_days")
_FLOAT_KEYS = ("scale", "rekindle_discount")

# 入れ子層の名前(内側→外側)。caps_of が単調性(close ≤ friend ≤ acquaint ≤ total)を保証する。
LAYERS = ("close", "friend", "acquaint", "total")


# --------------------------------------------------------------------------- #
# cfg 正準化(relations_endo.build_cfg と同型: dict/OmegaConf 両対応・型強制)
# --------------------------------------------------------------------------- #
def _to_plain(raw):
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(raw):
            return OmegaConf.to_container(raw, resolve=True)
    except Exception:                              # noqa: BLE001 (omegaconf 不在でも動く)
        pass
    return raw


def build_cfg(raw) -> dict:
    """conf の relations.dunbar ブロックを正準化(既定 OFF=現行挙動と完全同一)。"""
    raw = dict(_to_plain(raw) or {})
    cfg = dict(DEFAULTS)
    for k, v in raw.items():
        if k not in DEFAULTS:
            continue
        if k in _BOOL_KEYS:
            cfg[k] = bool(v)
        elif k in _INT_KEYS:
            cfg[k] = int(v)
        elif k in _FLOAT_KEYS:
            cfg[k] = float(v)
    return cfg


def cfg_of(sim) -> dict:
    """dunbar 設定(初回のみ sim.cfg.relations.dunbar から遅延構築してキャッシュ)。

    キャッシュ属性 sim.dunbarcfg は L1/L2/L3/乱数に一切現れない=既定 OFF のバイト一致を
    壊さない(relations_endo.cfg_of と同型)。"""
    c = getattr(sim, "dunbarcfg", None)
    if c is None:
        try:
            raw = (sim.cfg.get("relations", None) or {}).get("dunbar", None)
        except Exception:                          # noqa: BLE001 (旧 config 互換)
            raw = None
        c = build_cfg(raw)
        sim.dunbarcfg = c
    return c


def enabled(sim) -> bool:
    """認知枠が実効か。**relations.enabled が前提**(closeness が動く経路の上に載る機構)。"""
    rc = getattr(sim, "relationscfg", None)
    if not (rc and rc.get("enabled")):
        return False
    return bool(cfg_of(sim)["enabled"])


# --------------------------------------------------------------------------- #
# 上限の導出(純関数・乱数ゼロ)
# --------------------------------------------------------------------------- #
def caps_of(cfg: dict) -> dict:
    """Dunbar 層 × scale → 実効上限 dict(入れ子の単調性と下限 1 を保証する純関数)。

    round-half-up(int(x+0.5))で決定論。既定 (5,15,50,150)×0.34 → close2/friend5/acquaint17/total51。
    """
    scale = float(cfg["scale"])
    caps: dict = {}
    prev = 1
    for k in LAYERS:
        v = int(float(cfg[k]) * scale + 0.5)
        v = max(1, v, prev)                        # 下限1 かつ 内側 ≤ 外側(入れ子の単調性)
        caps[k] = v
        prev = v
    return caps


def cap_total(sim) -> int:
    """この sim の活性関係の総数上限(= 最外層。休眠を起こす唯一の拘束)。"""
    return caps_of(cfg_of(sim))["total"]


# --------------------------------------------------------------------------- #
# 台帳の読み取りヘルパ(他 module から使う小さな口)
# --------------------------------------------------------------------------- #
def is_dormant(rel) -> bool:
    """関係台帳エントリ 1 件が休眠中か(dunbar OFF ではキー自体が無いので常に False)。"""
    return bool(rel) and bool(rel.get("dormant"))


def dormant_ids(agent) -> list:
    """休眠中の相手 id を昇順で返す(弱い紐帯探索枠の pool 拡張に使う=決定論)。"""
    rels = getattr(getattr(agent, "mem", None), "relations", None) or {}
    return sorted(int(oid) for oid, rel in rels.items() if rel.get("dormant"))


def active_ids(agent) -> list:
    """活性関係(closeness 台帳を持ち休眠でない)の相手 id を昇順で返す(観測・テスト用)。"""
    rels = getattr(getattr(agent, "mem", None), "relations", None) or {}
    return sorted(int(oid) for oid, rel in rels.items()
                  if "closeness" in rel and not rel.get("dormant"))


def _active_rows(agent) -> list:
    """活性関係の (closeness, 相手id) 行(未ソート。決定論の並べ替えは呼び手が行う)。"""
    rows: list = []
    for oid, rel in (getattr(agent.mem, "relations", None) or {}).items():
        if "closeness" not in rel or rel.get("dormant"):
            continue
        rows.append((float(rel["closeness"]), int(oid)))
    return rows


def _protected(sim, agent, cfg: dict, today: int) -> set:
    """休眠の対象から外す相手 id 集合(直近接触+同居人)。membership 判定のみ=反復しない。

      - 直近 keep_days 日以内に接触した相手(**接触こそが維持**。既定 0=当日接触のみ保護=
        memory.py B6 の LRU が持つ「いま触れた相手は落とさない」不変条件と同じ規律)。
      - 同居人(protect_housemates)= 日々の共在で維持される紐帯。
    保護対象だけで上限を超える場合は超過したまま(保護が優先=正直な限界)。"""
    prot: set = set()
    keep = int(cfg["keep_days"])
    for oid, rel in (getattr(agent.mem, "relations", None) or {}).items():
        if "closeness" not in rel or rel.get("dormant"):
            continue
        if (today - sim.clock.day(int(rel.get("last_step", 0)))) <= keep:
            prot.add(int(oid))
    if cfg["protect_housemates"]:
        for oid in (getattr(agent, "housemates", None) or []):
            prot.add(int(oid))
    return prot


# --------------------------------------------------------------------------- #
# 当日/累積タリー(L2 観測の材料。checkpoint.py が中央管理=resume==straight)
# --------------------------------------------------------------------------- #
def _state(sim, day: int) -> dict:
    """タリー state を返す(無ければ作る)。int/float のみ=set なし=決定論監査は自明。

      active_mean      … 日境界で焼き直す「1 体あたりの活性関係数」の平均
      dormant_total    … いま休眠中の関係の総数(発生時 ±1 で常に正確・日境界で全走査再同期)
      dormant_events   … ラン開始からの休眠発生の累積(単調非減少)
      rekindle_events  … ラン開始からの再会の累積(単調非減少)
    """
    st = getattr(sim, "_dunbar_state", None)
    if st is None:
        st = {"day": int(day), "active_mean": 0.0, "dormant_total": 0,
              "dormant_events": 0, "rekindle_events": 0}
        sim._dunbar_state = st
    else:
        st["day"] = int(day)
    return st


def scalars(sim) -> dict | None:
    """L2 全体スカラー 3 列(OFF は None=列なし=L2 不変。relations_endo.scalars と同型)。

      active_relations_mean … 1 体あたりの活性関係数の平均(**日境界で焼き直す**=日内は一定。
                              正直な限界: 日内の増減はこの列に出ない。件数の日内変化は L1 の
                              relation_dormant / relation_rekindle を数える)
      dormant_total         … いま休眠中の関係の総数(発生時 ±1 で日内も正確)
      rekindle_total        … 再会の累積件数(単調非減少=resume 安全)
    """
    if not enabled(sim):
        return None
    st = getattr(sim, "_dunbar_state", None)
    if st is None:
        return {"active_relations_mean": 0.0, "dormant_total": 0.0,
                "rekindle_total": 0.0}
    return {"active_relations_mean": float(st["active_mean"]),
            "dormant_total": float(st["dormant_total"]),
            "rekindle_total": float(st["rekindle_events"])}


# --------------------------------------------------------------------------- #
# 休眠 / 再会(状態遷移。全決定論・乱数ゼロ)
# --------------------------------------------------------------------------- #
def _gap_days(sim, sim_min: int, from_step: int) -> int:
    """今日と from_step の日との差(relations.decay_day と同じ日付の採り方)。"""
    today = int(sim_min) // 1440
    return int(today - sim.clock.day(int(from_step)))


def _make_dormant(sim, agent, other_id: int, step: int, sim_min: int) -> None:
    """1 件の関係を休眠にする(削除しない=退避)。closeness を 0 にし tier を落とす。"""
    rel = agent.mem.relations[other_id]
    clo = float(rel.get("closeness", 0.0))
    old_tier = int(rel.get("tier", 0))
    rel["dormant"] = True
    rel["dormant_closeness"] = clo                 # 休眠前の親密度を退避(記憶は消さない)
    rel["dormant_step"] = int(step)
    rel["closeness"] = 0.0
    rel["tier"] = 0
    st = _state(sim, int(sim_min) // 1440)
    st["dormant_total"] += 1
    st["dormant_events"] += 1
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                         kind="relation_dormant", x=agent.x, y=agent.y,
                         payload={"other": int(other_id),
                                  "closeness": round(clo, 4),
                                  "from_tier": old_tier,
                                  "gap_days": _gap_days(sim, sim_min,
                                                        rel.get("last_step", 0))}))


def mark_dormant(sim, agent, other_id: int, step: int, sim_min: int) -> bool:
    """**他レーンからの公開 API**: 1 件の関係を休眠にする(既に休眠 / 対象外なら False)。

    存在の内生化 POP(``population.py``)の転出が使う: 街を去った相手との関係は
    「切れた」のではなく「会えなくなった」= 既存の休眠(``relation_dormant``)そのもの
    なので、**新しい関係イベントを 1 種も作らない**でこれを流用する。

    dunbar OFF(= ``relations.dunbar.enabled=false`` or relations OFF)のときは
    **何もしない**(休眠という概念がそのランに無いのに台帳へキーを生やさない)。
    """
    if not enabled(sim):
        return False
    rels = getattr(getattr(agent, "mem", None), "relations", None)
    if not rels:
        return False
    rel = rels.get(int(other_id))
    if rel is None or rel.get("dormant"):
        return False
    _make_dormant(sim, agent, int(other_id), step, sim_min)
    return True


def _rekindle(sim, agent, other_id: int, step: int, sim_min: int) -> None:
    """休眠関係を再活性化する(休眠前 closeness × rekindle_discount で復元)。"""
    cfg = cfg_of(sim)
    rel = agent.mem.relations[other_id]
    before = float(rel.pop("dormant_closeness", 0.0))
    dstep = int(rel.pop("dormant_step", step))
    rel.pop("dormant", None)                       # キーごと落とす(台帳を膨らませない)
    clo = before * float(cfg["rekindle_discount"])
    rel["closeness"] = clo
    rc = getattr(sim, "relationscfg", None) or _relations.DEFAULTS
    rel["tier"] = _relations.tier_of(clo, rc)
    st = _state(sim, int(sim_min) // 1440)
    st["dormant_total"] = max(0, int(st["dormant_total"]) - 1)
    st["rekindle_events"] += 1
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                         kind="relation_rekindle", x=agent.x, y=agent.y,
                         payload={"other": int(other_id),
                                  "before": round(before, 4),
                                  "closeness": round(clo, 4),
                                  "tier": int(rel["tier"]),
                                  "gap_days": _gap_days(sim, sim_min, dstep)}))


def enforce(sim, agent, step: int, sim_min: int) -> int:
    """agent 1 体の活性関係を上限内へ収める(超過分を最も弱い紐帯から休眠)。休眠件数を返す。

    並べ替えは (closeness 昇順, 相手id 昇順)= 乱数ゼロの決定論。直近接触(keep_days)と同居人は
    保護する(_protected)。**呼び出しは日境界の 1 回だけ**(day_phase)= 接触のたびに上限を
    課すと closeness が近接する紐帯群で休眠/再会が毎接触で振動する(実測: 300体2日で 5739 件の
    休眠イベント)ため、認知枠の適用を**日次の構造的過程**に固定した。その代償として活性関係数は
    **日内では一時的に上限を超え得る**(翌日の日境界で収束する。正直な限界)。"""
    cfg = cfg_of(sim)
    rows = _active_rows(agent)
    over = len(rows) - caps_of(cfg)["total"]
    if over <= 0:
        return 0                                   # 上限内: sort も保護集合の構築もしない
    prot = _protected(sim, agent, cfg, int(sim_min) // 1440)
    rows = sorted(r for r in rows if r[1] not in prot)
    n = 0
    for _clo, oid in rows[:over]:
        _make_dormant(sim, agent, oid, step, sim_min)
        n += 1
    return n


# --------------------------------------------------------------------------- #
# engine から呼ばれる口(OFF は即 return=1 バイトも動かない)
# --------------------------------------------------------------------------- #
def on_contact(sim, actor, other_id: int, step: int, sim_min: int) -> None:
    """交流の**記録前**に呼ぶ: 相手が休眠中なら再会(closeness を割引復元)させる。

    note_contact より前に呼ぶ必要がある(後だと当該交流分の closeness 増分を復元値で潰す)。
    上限の再適用はここでは**行わない**(日境界に一本化=enforce の docstring)。"""
    if not enabled(sim):
        return
    rel = (getattr(actor.mem, "relations", None) or {}).get(int(other_id))
    if rel is not None and rel.get("dormant"):
        _rekindle(sim, actor, int(other_id), step, sim_min)


def touch(sim, agent, other_id: int, step: int, sim_min: int) -> None:
    """同席(joint)による維持・再活性化。会話を捏造しない(count/last は増やさない)。

    休眠中なら再会させ、活性なら last_step だけ現在に進める(= 同席は維持行為であり
    relations.decay_day の長期不在減衰と認知枠の淘汰の両方から守られる)。台帳に無い相手には
    **何もしない**(関係の新規作成は会話経路=note_contact の仕事。正直な限界: 同席だけでは
    知り合わない)。"""
    if not enabled(sim):
        return
    rel = (getattr(agent.mem, "relations", None) or {}).get(int(other_id))
    if rel is None:
        return
    if rel.get("dormant"):
        _rekindle(sim, agent, int(other_id), step, sim_min)
        return
    if "closeness" in rel:
        rel["last_step"] = int(step)


def touch_group(sim, member_ids, step: int, sim_min: int) -> None:
    """同席グループ(joint)の全順序対に touch を適用する(id 昇順=決定論)。"""
    if not enabled(sim):
        return
    ids = sorted(int(i) for i in member_ids)
    for a_id in ids:
        # ★在場者のみ。退場者(脱水済み)の台帳に last_step を書いても hydrate で捨てられる
        #   = 「維持したはずの関係が休眠する」= 認知枠の観測が汚れる。
        agent = sim.present_agent(a_id)
        if agent is None:
            continue
        for b_id in ids:
            if b_id != a_id:
                touch(sim, agent, b_id, step, sim_min)


def day_phase(sim, step: int, sim_min: int) -> None:
    """日境界の一括処理: 全 agent へ上限を課し、L2 スカラーを焼き直す(1 日 1 回だけ全走査)。

    呼び出し位置は relations の日次フェーズ(decay_day の**後**)= 当日の減衰後の closeness で
    順位付けする。agent の反復は sim.agents(id 昇順=decay_day と同じ規律)。"""
    if not enabled(sim):
        return
    st = _state(sim, int(sim_min) // 1440)
    for agent in sim.agents:
        enforce(sim, agent, step, sim_min)
    _refresh_scalars(sim, st)


def _refresh_scalars(sim, st: dict) -> None:
    """活性/休眠の総数を全走査で焼き直す(日境界のみ。日内は発生時の ±1 で維持)。"""
    n_active = 0
    n_dormant = 0
    n_agents = 0
    for agent in sim.agents:
        n_agents += 1
        for rel in (getattr(agent.mem, "relations", None) or {}).values():
            if rel.get("dormant"):
                n_dormant += 1
            elif "closeness" in rel:
                n_active += 1
    st["active_mean"] = round(n_active / n_agents, 6) if n_agents else 0.0
    st["dormant_total"] = int(n_dormant)
