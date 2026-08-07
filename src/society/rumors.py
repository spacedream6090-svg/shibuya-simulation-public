"""情報オブジェクトの一般化 IF-C(``information.rumors``・**既定 OFF**)。

正典
----
- docs/research/if-lane-research.md **§2**(Daley & Kendall 1964/1965 と Maki & Thompson 1973 の
  ignorant / spreader / **stifler** 3 クラス。理論的極限は **最終 ignorant ≈ 20%** ・
  **spreader ピーク = 1 − ln2 ≈ 0.3069**。OASIS の伝播指標 = **scale / depth / max_breadth**。
  ★Generative Agents (Park et al. 2023) の拡散測定は **2 日終了時点の自然言語インタビュー**で、
  ID つきの情報オブジェクトを機構として持っていない(関係形成の設問では 1.3% が幻覚)。
  本シムの ``provenance.Item`` + ``transmissions`` は**先行研究に追随ではなく既に先行**しており、
  IF-C は「不足の補填」ではなく「**優位の完成**」である)
- docs/research/llm-world-interface-audit.md **§3**(情報カテゴリの現状 = ``Item.kind`` は
  rumor / institution を**宣言済み**だが生成は vocab のみ・伝聞はテキスト直接注入)
- docs/plans/if-sv-p4-plan.md 「IF-C」行

何を解く問題か
--------------
監査 §3 の開閉マトリクスで「語彙(coin_label)」だけが **ID 付き Item + transmissions** を
持ち、因果ループが**完全に閉じている**。一方で「街で何が起きたか」= 出来事の伝聞は、
発話テキストという無構造の帯域を流れるだけで、**誰から誰へ何が伝わったのかを機構として
追えない**。IF-C は ``Item.kind="rumor"`` という**宣言済みの空き枠**に実体を入れて、
出来事の伝聞を語彙と同じ精度(誤差ゼロのカスケード木)で測れるようにする。

そして §2 が指摘した現行の欠落を 1 つだけ足す: **噂が止まる理由**である。
Daley-Kendall / Maki-Thompson の核心は「全員が知って**飽和**するから止まる」のではなく
「**既知の相手に出会った語り手が語るのをやめる**から止まる」ことで、だからこそ極限で
2 割が永久に知らないままになる。現行の伝播機構(labels / truth_ledger)には
stifler 化に相当する規則が 1 つも無い = **噂が止まらない世界**だった。

3 つの設計判断(と、その理由)
------------------------------
**(1) 噂の源は「構造化イベント」に限る。自由文の解釈はしない。**
    L1 に既に出ている**有限種の**行為イベント(``SRC_SPECS`` の 5 種)だけを源にする。
    発話本文からの話題抽出(truth_ledger 流の部分文字列)は、if-lane-research §2-1 が
    方式 (b) として挙げつつ「再現率と適合率の両方で妥協が要る/地名頻出の街では過剰一致」
    と限界を明記した経路であり、**同じ限界を二重に持ち込む価値が無い**。構造化イベントは
    方式 (c)(テンプレート化された行為が ID を直接運ぶ)に相当し、抽出誤差がゼロになる。
    ★源イベントの本文(``event_host`` の title・``venture_open`` の name)は
    **1 バイトも使わない**。これらは LLM の自由文であり、噂文面へ通せば
    (i) 本 module の等級が strict から journal へ落ち、(ii) 数字・記号が噂文面に混ざり、
    (iii) 自由文が新しい世界書き込み帯域を 1 本増やす。噂文面は
    **(源イベント種, 場所名 or 人名) の純関数**に閉じる。

**(2) 発話テキストには噂を注入しない(= 噂は「別チャネルの伝聞」)。**
    speak/dm の本文へ噂文面を混ぜ込む実装は、(i) LLM が書いた文を機械が改竄することになり
    会話内容の自然さを壊す、(ii) 改竄の有無が実験条件と相関すれば **no-fingerprint 違反**に
    直結する、(iii) 発話生成の**前**に噂を選ぶならプロンプトに噂を載せる必要があり
    「LLM に噂を語らせる」= 追加 LLM 呼または新しいプロンプト節が要る。
    そこで本 module は「その会話の**傍らで**伝聞が伝わった」という別チャネルとして扱い、
    ★**発話テキストは 1 バイトも触らず**、聞き手の**記憶に定型文 1 行**だけを載せる。
    Daley-Kendall のモデル自体が「接触したら伝わる」であって「何を喋ったか」を持たない
    ことと整合し、追加 LLM 呼はゼロ・プロンプトの**節**も 1 つも増えない
    (増えるのは既存「直近の出来事」欄の中身だけ = flyer / news / dm と同じ帯域)。

**(3) stifler 化は完全決定論(乱数 stream を 1 本も引かない)。**
    Maki-Thompson 流 = 「spreader が**既に知っている相手**に語った回数が閾値
    (``stifle_threshold``・既定 1)に達したら、以後その噂を語らない(stifler)」。
    誰と誰が同席するかは既存の物理・接触過程が既に決めているので、**接触順だけで**
    停止が決まる = 新しい確率は 1 つも要らない。``stifle: "none"`` で無効(語り続ける)。
    ★Daley-Kendall 版(接触した spreader の**両方**が stifler 化する)は実装していない。
      4 行で足せるが、指示された conf 値は ``mt`` / ``none`` の 2 つであり、
      勝手に第 3 の実験条件を増やさない(必要なら ``STIFLE_MODES`` に 1 語足すだけ)。

R1 ドクトリン
-------------
- 既定 ``enabled: false`` では ``phase`` / ``on_talk`` / ``provenance`` が即 return し、
  **L1 に 1 件も出ず・記憶に 1 行も入らず・agent に属性が生えず・sim に state が生えず・
  Item が 1 つも増えず・プロンプトが 1 バイトも変わらない**(= ゴールデン L1 バイト一致。
  第75 dunbar の dormant / 第94 IF-B と同じ「OFF ではキー自体を作らない」流儀)。
- **乱数 stream を 1 本も引かない**(専用 stream すら不要なほど完全決定論)。
- **generate() の呼び出しサイトを 1 つも作らない**(LLM 追加呼ゼロ = k 非依存)。
- no-fingerprint: 噂文面は ``sentence()`` = **(源イベント種, 場所名 or 人名) の純関数**で、
  config・実験条件・k・特性・時刻・数値を 1 つも読まない。同じ源からは全実験条件で
  同一バイト列になる(tests/test_rumors.py が機械固定)。
  ★``item_id`` は**絶対にプロンプトへ入れない**(観測側だけの概念に閉じる)。

**絶対に触らない範囲(明文規約)**
--------------------------------
1. ``truth_ledger.py`` は 1 バイトも変更しない(``observer/metrics_spec.py`` の**凍結 14
   ファイル**)。噂と真偽台帳は別レイヤーで、本 module は台帳の fact も belief も canary も
   1 つも読まない。
2. ``observer/provenance.py`` と ``labeling/labels.py`` も**変更していない**。
   ``ItemStore.new_item(kind="rumor", …)`` / ``ItemStore.transmit(…)`` は既存 API のままで
   足りる(= 伝播記録系の二重化を作らない = 既存のカスケード木ツール 19 本がそのまま効く)。
   噂 Item を ``LabelSystem`` を**経由せずに**直接 ``ItemStore`` から起こすのは意図的で、
   ``labels.text_to_item`` に載せると噂文面が「採用しうる語」になってしまうため。

正直な限界(4 件)
------------------
- **残る遅れは 2 つだけ**(第98 W2-5 で「1 step の遅れ」は解消した):
  誕生走査 :func:`birth_scan` は step 末だけでなく **``_apply`` の各回の直前**にも走るので、
  源イベントより**後に適用される個体**の会話には**同じ step のうちに**噂が乗る
  (既定 Δt=10 分の「10 分以内に人づてで広まる」が表現できるようになった)。
  残っているのは (i) **行為者自身**はその step の行動枠を出来事そのものに使い切っている
  ので語れない(= 物理的に正しい遅れ)、(ii) 発話文面を決める ``_decide`` は ``_apply``
  より前に終わっているので「噂を知ったこと」がプロンプトに映るのは次 step から
  (噂は本文に注入しない設計判断 (2) なので**伝播そのもの**には影響しない)。
- **目撃者は「出来事の直後の走査」時点の位置で決める**: 出来事の**発生時**そのものでは
  なく、その直後に走る誕生走査の時点の同席者を初期 knower にする
  (``hearers_of`` = 既存の「同席者」規約と同じ知覚半径。空間索引は ``_phase_move`` 後に
  1 回だけ張られた 1 枚)。第98 以前の「step 末の位置」よりは近いが、Δt 内の出入りは
  やはり捉えない。
- **プール退場で噂を失う**: ``world/pool.py`` の ``dehydrate`` は静的属性と明示列挙した
  可変状態しか運ばないため、``_rumors``(動的属性)は退場時に落ちる = 街を出て戻った者は
  噂を忘れて帰ってくる。``cognition/day_plan.py`` が同じ制約を明記しているのと同型で、
  pool.py を触らずに直せる方法が無いので**正直に限界として残す**(観察ランで
  ``pool`` と併用するときは reach が過小に出る)。
- **凍結指標との相互作用**: ``observer/measure.py`` / ``stream.py`` / ``echo.py``(凍結 14
  ファイル)の ``c_transmission`` / ``n_transmission`` / ``transmission_novel_rate`` は
  **item の kind を見ずに全 transmission を数える**。したがって本機能 ON のランでは
  これらの語彙伝播指標に噂の伝播が混ざる。凍結ファイルは 1 バイトも触れないので、
  切り分けは解析側(``scripts/analyze_rumors.py`` = item_id 接頭辞 ``rumor-`` で分離)で行う。
  **既定 OFF では 1 件も混ざらない**ので、過去ランとの比較は無風。
"""
from __future__ import annotations

from .observer.schema import Event
from .world.perception import hearers_of

SCHEMA = 1

# ---- 役割(Daley-Kendall / Maki-Thompson の 3 クラスのうち、知っている 2 つ)---------- #
# ignorant は「``_rumors`` にその item_id が無い」ことで表現する(状態を持たせない)。
SPREADER = "spreader"
STIFLER = "stifler"

# ---- 停止規則(conf: information.rumors.stifle)------------------------------------ #
MT = "mt"            # Maki & Thompson 1973: 既知の相手へ語った側だけが stifler 化
NONE = "none"        # 停止しない(語り続ける)= 現行の伝播機構と同じ「止まらない世界」
STIFLE_MODES: tuple[str, ...] = (MT, NONE)

# 一時属性名(**OFF では 1 度も生えない**)。checkpoint は agents pickle に自然同梱。
RUMOR_KEY = "_rumors"

# 噂 Item の kind(= item_id の接頭辞 "rumor-" になる。解析側の分離キー)。
KIND = "rumor"


# --------------------------------------------------------------------------- #
# 源イベントの有限表(**ここに無い L1 種からは噂が生まれない**)
#
#   form  … "place" = 場所を主語にする / "actor" = 人を主語にする
#   text  … 定型文面。**プレースホルダは 1 つだけ**(数値・時刻・自由文は入らない)
#   extra … 当事者を追加で取り出す payload キー(初期 knower に加える)
#
# ★源イベントの自由文(title / name)は使わない = 噂文面は純関数(設計判断 (1))。
# --------------------------------------------------------------------------- #
SRC_SPECS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    # 公共の出来事(街頭で誰の目にも触れる)= 既定の源
    "event_host":     ("place", "{place}で集まりがあるらしい。", ()),
    "venture_open":   ("place", "{place}に新しい店ができたらしい。", ()),
    "enforcement":    ("place", "{place}で取り締まりがあったらしい。", ("target",)),
    # 私生活の出来事(当事者しか知り得ない)= 既定では**源にしない**
    "partner_formed": ("actor", "{actor}に親しい相手ができたらしい。", ("other",)),
    "relation_break": ("actor", "{actor}が誰かと仲たがいしたらしい。", ()),
}

# 既定の源リスト(**保守的に = 公共の出来事 3 種だけ**)。
#
# なぜ private な 2 種を既定から外すか:
#   partner_formed / relation_break は「当事者しか知り得ない出来事」であり、目撃者(同席者)を
#   初期 knower にする本 module の規則が意味を成さない(関係の成立は日境界の台帳処理で起き、
#   その瞬間に同席している必然性が無い)。既存の負の評判機構(``gossip.py``・既定 OFF)が
#   relation_break を種として扱う設計と**二重に**なる点も避ける。conf で足せば使える。
# なぜ enforcement を既定に入れるか:
#   摘発は路上の公共的な出来事で、当事者(officer / target)と同席者が実際に居合わせる。
DEFAULT_SOURCES: tuple[str, ...] = ("event_host", "venture_open", "enforcement")

DEFAULTS = {
    "enabled": False,
    "sources": list(DEFAULT_SOURCES),
    "stifle": MT,               # mt(既定) | none
    "stifle_threshold": 1,      # 既知の相手へ語った回数がこれに達したら stifler(MT)
    "forget_steps": 0,          # 忘却 TTL [step]。0 = 忘れない(既定)
    "max_per_step": 8,          # 1 step に生む噂の上限(暴走の安全弁)
    "max_per_talk": 1,          # 1 回の発話/DM で伝える噂の本数(古い順)
}

_INT_KEYS = ("stifle_threshold", "forget_steps", "max_per_step", "max_per_talk")


# --------------------------------------------------------------------------- #
# cfg 正準化(gossip.build_cfg / relations.build_cfg と同型: dict / OmegaConf 両対応)
# --------------------------------------------------------------------------- #
def _to_plain(raw):
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(raw):
            return OmegaConf.to_container(raw, resolve=True)
    except Exception:                              # noqa: BLE001(旧 config 互換)
        pass
    return raw


def build_cfg(raw) -> dict:
    """conf の ``information.rumors`` ブロックを正準化(既定 OFF=現行挙動と完全同一)。

    dotlist 上書きは文字列で入り得るため型強制する(``gossip.build_cfg`` と同じ作法)。
    ``sources`` は**置き換え**(重ねない): 源を絞る/広げるのが目的の軸なので、既定リストが
    暗黙に残ると「何を源にしたラン」なのかが config から読めなくなる。
    ``SRC_SPECS`` に無い種は**黙って捨てる**(定型文を持たない種から噂は作れない=捏造しない)。
    """
    raw = dict(_to_plain(raw) or {})
    cfg = dict(DEFAULTS)
    cfg["sources"] = list(DEFAULT_SOURCES)
    for k, v in raw.items():
        if k == "enabled":
            cfg["enabled"] = bool(v)
        elif k == "sources":
            cfg["sources"] = [str(s) for s in (_to_plain(v) or []) if str(s) in SRC_SPECS]
        elif k == "stifle":
            s = str(v or MT)
            cfg["stifle"] = s if s in STIFLE_MODES else MT
        elif k in _INT_KEYS:
            cfg[k] = max(0, int(v))
    cfg["stifle_threshold"] = max(1, int(cfg["stifle_threshold"]))
    return cfg


def cfg_of(sim) -> dict:
    """噂設定(初回のみ ``sim.cfg.information.rumors`` から遅延構築してキャッシュ)。

    simulation.py は読み取り専用なので cfg は本 module が遅延構築する
    (``gossip.cfg_of`` / ``status.cfg_of`` と同型)。キャッシュ属性 ``sim.rumorscfg`` は
    L1/L2/L3/乱数に一切現れない = 既定 OFF のバイト一致を壊さない。
    """
    c = getattr(sim, "rumorscfg", None)
    if c is None:
        try:
            raw = (sim.cfg.get("information", None) or {}).get("rumors", None)
        except Exception:                          # noqa: BLE001(旧 config 互換)
            raw = None
        c = build_cfg(raw)
        sim.rumorscfg = c
    return c


def enabled(sim) -> bool:
    return bool(cfg_of(sim)["enabled"])


# --------------------------------------------------------------------------- #
# 定型文(純関数。sim も agent も config も見ない = 全実験条件で同一バイト列)
# --------------------------------------------------------------------------- #
def sentence(src_kind: str, place: str = "", actor: str = "") -> str:
    """噂の定型文。**(源イベント種, 場所名 or 人名) だけの純関数**。

    プレースホルダは 1 つだけで、埋まるのは**世界が所有する固有名**(``city.place_label``の
    場所名 / ``agent.name`` の表示名)。時刻・金額・件数・実験条件・機構語は 1 つも入らない
    (tests が「テンプレート側に数字が無い」ことと「固有名を除いた残りに数字が無い」ことを
    機械検査する。場所名そのものには "SHIBUYA 109" のように数字を含む実在名がある)。
    """
    spec = SRC_SPECS.get(str(src_kind))
    if spec is None:                               # 語彙外は噂にしない(捏造しない)
        return ""
    form, tmpl, _extra = spec
    if form == "place":
        return tmpl.format(place=place) if place else ""
    return tmpl.format(actor=actor) if actor else ""


# --------------------------------------------------------------------------- #
# 状態(ON 経路でのみ生やす。checkpoint.py が runtime で中央管理する)
# --------------------------------------------------------------------------- #
def _state(sim) -> dict:
    st = getattr(sim, "_rumor_state", None)
    if st is None:
        st = {"schema": SCHEMA, "born": 0, "transmissions": 0, "stiflers": 0,
              "forgotten": 0, "day": -1, "by_src": {}, "items": {}}
        sim._rumor_state = st
    return st


def _known(agent) -> dict:
    """agent が知っている噂 item_id → {role, redundant, step, src}。ON 経路でのみ生やす。

    **挿入順 = 知った順**(dict は pickle でも順序を保つ)= 決定論の反復順。
    """
    d = getattr(agent, RUMOR_KEY, None)
    if d is None:
        d = {}
        setattr(agent, RUMOR_KEY, d)
    return d


def knows(agent, item_id: str) -> bool:
    """agent がその噂を知っているか(ignorant 判定。**属性を生やさない**読み取り)。"""
    d = getattr(agent, RUMOR_KEY, None)
    return bool(d) and item_id in d


def _learn(sim, st: dict, agent, item_id: str, src_kind: str, step: int) -> bool:
    """agent を knower(spreader)にする。既に知っていれば False。"""
    d = _known(agent)
    if item_id in d:
        return False
    d[item_id] = {"role": SPREADER, "redundant": 0, "step": int(step),
                  "src": str(src_kind)}
    rec = st["items"].get(item_id)
    if rec is not None:
        rec["reach"] = int(rec["reach"]) + 1
    return True


# --------------------------------------------------------------------------- #
# 1. 噂の誕生(新規 L1 の走査。決定論・乱数ゼロ・LLM ゼロ)
#
# ★第98 W2-5(IF-C 残③「1 step の遅れ」の解消): 走査は step 末だけでなく
#   **``_apply`` の各回の直前**にも走る(scheduler.run_step)。詳細は birth_scan の
#   docstring と module docstring「正直な限界」の 1 件目。
# --------------------------------------------------------------------------- #
def _new_events(sim) -> list:
    """前回処理済み総数(watermark)以降の新規 L1(``gossip._scan_seeds`` と同型)。

    watermark は**プロセス内 logger カウンタ由来**なので checkpoint に保存しない
    (resume 直後は新しい logger の 0 から数え直す = 第59 assets / 第61 gossip と同流儀)。
    """
    logger = getattr(sim, "logger", None)
    if logger is None:
        return []
    events = logger.events
    total = int(getattr(logger, "_n_flushed", 0)) + len(events)
    processed = int(getattr(sim, "_rumor_watermark", 0))
    new = max(0, min(total - processed, len(events)))
    sim._rumor_watermark = total
    return events[len(events) - new:] if new else []


def _place_of(sim, node) -> str:
    """ノードの表示名(既存の表示名規約 = ``city.place_label`` に従う)。"""
    if not node:
        return ""
    try:
        return str(sim.city.place_label(node))
    except Exception:                              # noqa: BLE001(未知ノード)
        return ""


def _initial_knowers(sim, actor, payload: dict, extra_keys) -> list:
    """当事者 + 目撃者(同席者)。**agent_id 昇順の決定論**・乱数ゼロ。

    目撃者 = 既存の「同席者」規約 (``hearers_of`` = 知覚半径・睡眠中と屋内外の別を尊重)。
    step 末の位置で決めるという限界は module docstring に明記した。
    """
    seen = {int(actor.id): actor}
    for key in extra_keys:
        oid = payload.get(key)
        if isinstance(oid, int) and oid >= 0 and oid not in seen:
            other = sim.agent_by_id.get(oid)
            if other is not None:
                seen[oid] = other
    radius = float(sim.cfg.world.perception_radius_m)
    index = getattr(sim, "percept_index", None)
    for w in hearers_of(actor, index if index is not None else sim.agents, radius):
        seen.setdefault(int(w.id), w)
    return [seen[i] for i in sorted(seen)]


def _birth(sim, cfg: dict, st: dict, e, step: int, sim_min: int) -> bool:
    """L1 イベント 1 件 → 噂 Item 1 件(生まなければ False)。"""
    spec = SRC_SPECS.get(e.kind)
    if spec is None or int(e.agent_id) < 0:
        return False
    actor = sim.agent_by_id.get(int(e.agent_id))
    if actor is None:
        return False
    payload = e.payload or {}
    node = payload.get("node") or getattr(actor, "node", None)
    text = sentence(e.kind, place=_place_of(sim, node), actor=str(actor.name))
    if not text:                                   # 場所も名前も解決できない = 作らない
        return False
    item = sim.items.new_item(KIND, text, int(actor.id), int(step))
    st["items"][item.item_id] = {"src": e.kind, "node": str(node or ""),
                                 "born_step": int(step), "creator": int(actor.id),
                                 "reach": 0}
    knowers = _initial_knowers(sim, actor, payload, spec[2])
    for a in knowers:
        _learn(sim, st, a, item.item_id, e.kind, step)
    st["born"] += 1
    st["by_src"][e.kind] = int(st["by_src"].get(e.kind, 0)) + 1
    sim.logger.log(Event(step=int(step), sim_min=int(sim_min),
                         agent_id=int(actor.id), kind="rumor_born",
                         x=actor.x, y=actor.y,
                         payload={"item_id": item.item_id, "src_kind": e.kind,
                                  "node": str(node or ""),
                                  "knowers": [int(a.id) for a in knowers]}))
    return True


def _forget_day(sim, cfg: dict, st: dict, sim_min: int, step: int) -> None:
    """忘却 TTL の日境界掃引(``forget_steps`` = 0 の既定では 1 度も走らない)。

    知ってから ``forget_steps`` step を超えた噂を落とす(role を問わない = stifler も忘れる)。
    ``reach``(到達人数)は**累積の観測量**なので減らさない。
    """
    ttl = int(cfg["forget_steps"])
    day = int(sim_min) // 1440
    if day == int(st["day"]):
        return
    st["day"] = day
    if ttl <= 0:
        return
    cutoff = int(step) - ttl
    for a in sim.agents:
        d = getattr(a, RUMOR_KEY, None)
        if not d:
            continue
        stale = [iid for iid, r in d.items() if int(r["step"]) < cutoff]
        for iid in stale:
            del d[iid]
        st["forgotten"] += len(stale)


def _step_budget(sim, cfg: dict, step: int) -> int:
    """この step に**あと何件**噂を生めるか(``max_per_step`` = **step あたり**の上限)。

    第98 で誕生走査が 1 step に何度も走るようになったので、上限を「1 回の走査あたり」で
    数えると実質的に上限が消えてしまう。IF-C 初版の意味(= 暴走の安全弁)をそのまま
    保つために、予算は step をキーにして持ち回る。

    ★プロセス内カウンタなので **checkpoint に保存しない**(``_rumor_watermark`` と
      同流儀)。resume は必ず step 境界から始まるので、その step の最初の走査で
      straight と同じ満額から張り直される = resume==straight を壊さない。
    """
    b = getattr(sim, "_rumor_budget", None)
    if b is None or int(b[0]) != int(step):
        b = (int(step), int(cfg["max_per_step"]))
        sim._rumor_budget = b
    return int(b[1])


def birth_scan(sim, step: int, sim_min: int) -> int:
    """新規 L1 → 噂 Item の誕生走査。**同じ step に何度呼んでも安全**(誕生数を返す)。

    ★第98 W2-5(IF-C 残③): この関数は ``scheduler.run_step`` の **``_apply`` ループの
      各回の直前**と、step 末の :func:`phase` から呼ばれる。IF-C 初版は step 末の 1 回
      だけだったので、伝播口 :func:`on_talk` が呼ばれる ``_apply`` は**必ず誕生より前**
      = その step に起きた出来事は次 step の会話にしか乗らなかった(1 step の遅れ)。

    二重誕生が起きない理由は **watermark 1 本**に閉じている: :func:`_new_events` は
    ``logger`` の総件数を読んで**その場で** ``sim._rumor_watermark`` を進めるので、
    同じ L1 イベントが 2 回窓に入ることが構造的に無い(``_birth`` 側に重複判定は不要)。
    自分が出す ``rumor_born`` / ``transmission`` は次の走査で窓に入るが、
    ``SRC_SPECS`` に無い種なので源にならない(再帰しない)。

    並びの決定論: 窓は L1 の**追記順**そのもの、初期 knower は agent_id 昇順
    (:func:`_initial_knowers`)、走査点は ``_apply`` ループの順(第81 barrier ON なら
    agent_id 昇順に正準化済み)。乱数は 1 本も引かない。
    """
    if not enabled(sim):
        return 0
    cfg = cfg_of(sim)
    st = _state(sim)
    # ★ sources が空でも必ず drain する(watermark を進める = IF-C 初版と同じ順序)。
    events = _new_events(sim)
    sources = cfg["sources"]
    if not sources:
        return 0
    budget = _step_budget(sim, cfg, step)
    n = 0
    for e in events:
        if budget <= 0:
            break
        if e.kind in sources and _birth(sim, cfg, st, e, step, sim_min):
            budget -= 1
            n += 1
    sim._rumor_budget = (int(step), int(budget))
    return n


def phase(sim, step: int, sim_min: int) -> None:
    """step 末の作用点: **取りこぼしの誕生** + 忘却掃引。**既定 OFF は即 return**。

    誕生の本体は :func:`birth_scan` に移った(第98 W2-5)。ここに残っているのは
    「``_apply`` ループの**最後**に適用された個体が出した源イベント」と
    「``_apply`` より後のフェーズが出した源イベント」の受け皿 + 日境界の忘却掃引で、
    どちらもこの step にはもう語り手が居ないので遅れを生まない。

    ★世界状態(所持金・位置・関係・drive・opinion)は 1 つも動かさない。
      触るのは「誰が何を知っているか」だけ。
    """
    if not enabled(sim):
        return
    birth_scan(sim, step, sim_min)
    _forget_day(sim, cfg_of(sim), _state(sim), sim_min, step)


# --------------------------------------------------------------------------- #
# 2. 伝播(既存の会話に相乗り。**発話テキストは 1 バイトも変えない**)
# 3. stifler 化(Maki-Thompson・決定論)
# --------------------------------------------------------------------------- #
def on_talk(sim, speaker, listeners, channel: str, step: int, sim_min: int) -> int:
    """発話/DM の**発生後**に呼ぶ唯一の伝播口。伝えた件数を返す(OFF は 0)。

    ★**発話テキストは読まないし書き換えない**(引数に取っていない)。噂は「その会話の傍らで
      伝わった伝聞」という別チャネルで、聞き手の**記憶に定型文 1 行**だけを載せる
      (module docstring の設計判断 (2))。LLM 呼はゼロ・プロンプトの節も増えない。
    ★``ablate.propagation_off``(第78)では即 return する: 同 ablation の契約は
      「発話は生成するが**内容を他者の文脈へ注入しない**」であり、噂は他者由来の内容そのもの。
    ★決定論: 噂は ``_rumors`` の挿入順(= 知った順)、聞き手は agent_id 昇順。乱数ゼロ。
    """
    if not enabled(sim) or not listeners:
        return 0
    from . import ablate as _ablate
    if _ablate.propagation_off(sim):
        return 0
    told = getattr(speaker, RUMOR_KEY, None)
    if not told:
        return 0
    cfg = cfg_of(sim)
    st = _state(sim)
    mt = cfg["stifle"] == MT
    thr = int(cfg["stifle_threshold"])
    budget = int(cfg["max_per_talk"])
    audience = sorted((a for a in listeners if int(a.id) != int(speaker.id)),
                      key=lambda a: int(a.id))
    if not audience or budget <= 0:
        return 0
    n = 0
    for item_id in list(told):
        if budget <= 0:
            break
        rec = told.get(item_id)
        if rec is None or rec["role"] != SPREADER:
            continue
        item = sim.items.items.get(item_id)
        if item is None:                           # 旧 checkpoint 等で Item が無い = 語れない
            continue
        budget -= 1
        for listener in audience:
            if knows(listener, item_id):
                # ---- Maki-Thompson: 既知の相手へ語った = 語る意味が無くなる ----------- #
                if not mt:
                    continue
                rec["redundant"] = int(rec["redundant"]) + 1
                if rec["redundant"] >= thr:
                    rec["role"] = STIFLER
                    st["stiflers"] += 1
                    sim.logger.log(Event(step=int(step), sim_min=int(sim_min),
                                         agent_id=int(speaker.id),
                                         kind="rumor_stifle",
                                         x=speaker.x, y=speaker.y,
                                         payload={"item_id": item_id}))
                    break                          # 以後この噂は語らない
                continue
            # ---- 伝播(既存 API = provenance.transmit → L1 transmission)------------- #
            sim.items.transmit(sim.logger, item, step=int(step), sim_min=int(sim_min),
                               from_agent=int(speaker.id), to_agent=int(listener.id),
                               channel=str(channel), x=listener.x, y=listener.y)
            _learn(sim, st, listener, item_id, rec["src"], step)
            listener.remember(item.text, kind=KIND)
            st["transmissions"] += 1
            n += 1
    return n


# --------------------------------------------------------------------------- #
# 4. 観測(summary.json の "rumors" キー。**OFF ではキー自体を出さない**)
# --------------------------------------------------------------------------- #
def provenance(sim) -> dict | None:
    """``summary.json`` の ``rumors`` キー(既定 OFF は None = キー自体を出さない)。

    reach 分布 = 「その噂を知っている人が何人になったか」の分布。Daley-Kendall の
    「最終 ignorant ≈ 20%」との比較は **ラン全体の 1 噂あたり**でしか意味を持たないので、
    ここでは分布(平均・最大・ヒストグラム)だけを出し、**理論値との判定は解析側**
    (``scripts/analyze_rumors.py``)に置く(観測がシムを変えない/判定式を 1 箇所に保つ)。
    """
    if not enabled(sim):
        return None
    cfg = cfg_of(sim)
    out: dict = {"schema": SCHEMA, "sources": list(cfg["sources"]),
                 "stifle": cfg["stifle"],
                 "stifle_threshold": int(cfg["stifle_threshold"]),
                 "forget_steps": int(cfg["forget_steps"])}
    st = getattr(sim, "_rumor_state", None)
    if st is None:                                 # ON だが 1 件も源イベントが起きていない
        out.update({"born": 0, "transmissions": 0, "stiflers": 0, "forgotten": 0,
                    "by_src": {}, "reach_mean": 0.0, "reach_max": 0, "reach_hist": {}})
        return out
    reaches = [int(r["reach"]) for r in st["items"].values()]
    hist: dict[str, int] = {}
    for r in reaches:
        hist[str(r)] = hist.get(str(r), 0) + 1
    out.update({
        "born": int(st["born"]), "transmissions": int(st["transmissions"]),
        "stiflers": int(st["stiflers"]), "forgotten": int(st["forgotten"]),
        "by_src": {k: int(v) for k, v in sorted(st["by_src"].items())},
        "reach_mean": (round(sum(reaches) / len(reaches), 4) if reaches else 0.0),
        "reach_max": (max(reaches) if reaches else 0),
        "reach_hist": {k: hist[k] for k in sorted(hist, key=int)}})
    return out


def role_counts(sim) -> dict:
    """役割の頭数(spreader / stifler / 述べ knower)。**読むだけ**・属性を生やさない。"""
    out = {SPREADER: 0, STIFLER: 0, "knower": 0}
    for a in sim.agents:
        d = getattr(a, RUMOR_KEY, None)
        if not d:
            continue
        for rec in d.values():
            out["knower"] += 1
            out[rec["role"]] = out[rec["role"]] + 1
    return out
