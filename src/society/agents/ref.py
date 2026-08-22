"""退場者(プール回転で街を出た個体)の**軽量参照** ``AgentRef``(レーン R1 / A1)。

なぜ要るか
----------
``sim.agent_by_id`` は「これまで実体化した**全**個体」の id→参照で、プール回転で街を
出た個体を**意図的に消さない**(``engine/scheduler.py::_phase_pool_rotation`` の注記
「造語の作者名・DM 送信者名・関係台帳など**過去の参照**が退場後も解決できるように」)。
この要件そのものは正しい —— 造語を言い始めた人が街を出た瞬間に検索結果から名前が
消えるのは、世界の側の都合であって出来事ではない。

問題は「解決に要るのは名前などの数十バイトなのに、**フル Agent を丸ごと掴み続けている**」
こと。フル Agent は統合エピソード(``store_cap`` 既定 120)・未統合バッファ(30)・
関係台帳・信念・persona 文を抱えるので、本選(25 万在場 + 途中入場 20.9 万 =
**累計 45.9 万体**)では退場ぶんが解放されずに積み上がる。

方針
----
退場のたびに ``agent_by_id[id]`` をフル Agent から本クラスへ差し替える。運ぶのは

  ① 退場者に対して**実際に読まれる**と同定した属性(``_READ_BY`` の全数列挙表)
  ② + 値ごとにポインタ 1 個しか要らない**素のスカラー欄**。これは落としても RAM が
     減らないので、落とす動機が「読者の取りこぼしで AttributeError を出す危険」しか
     残らない。したがって**育つ容器**(``_DROPPED``)だけを落とす。

未定義属性は ``__getattr__`` が **AttributeError を明示 raise** する。
``getattr(x, "attr", default)`` で読む読者に既定値を静かに返すと「街を出た人だけ値が
違う」という**気づけない挙動変化**になるので、運ぶべき欄は必ず表に載せること
(``tests/test_pool_departed_ref.py`` が読者一覧と表の対応を機械固定する)。

★**書き込みは通す**(``__setattr__`` の ``_extra`` 退避)。退場者への書き込みは
  現行実装でも「脱水済みの実体へ書いて次の hydrate で捨てられる」= 世界に残らない。
  ref でも同じく残らない(ref ごと再入場で捨てられる)ので**挙動は同値**であり、
  ここで AttributeError を投げると現行では起きない新しいクラッシュを作ってしまう。
  読み(未書き込みの未知欄)だけを止めるのが、静かな既定値を禁じる目的に対する最小の網。
"""
from __future__ import annotations

import dataclasses

from .agent import Agent
from .memory import MemoryStore


# --------------------------------------------------------------------------- #
# ① 退場者に対して**実際に読まれる**属性の全数列挙(読者 → 読む欄)。
#    grep(``agent_by_id`` の全読者)+ 実測(pool ON mock ランで退場者を追跡プロキシに
#    差し替えて到達したアクセスを記録)の両方で作った表。テストがこの表を読む。
# --------------------------------------------------------------------------- #
_READ_BY: dict[str, tuple[str, ...]] = {
    # 読む欄                      : それを読む場所(ファイル:関数)
    "id":            ("worldview._apply", "tools.Events.tick", "truth_ledger._receivers",
                      "rumors._birth", "traces._bystanders", "transit_interior._copresence",
                      "scheduler._fire_llm_g(dm)"),
    "name":          ("scheduler._search_index(造語の作者名 / SNS の著者名)",
                      "scheduler._feed_texts", "scheduler._fire_llm_g(dm 相手名)",
                      "scheduler._reply(話者名)", "household.context_line",
                      "cognition/engaged._sign_memory", "rumors._birth"),
    "x":             ("scheduler._hear_words(sns_geo 距離)", "cognition/fire._log_due",
                      "gossip._roll_day(gossip_seed の座標)",
                      "tools.Institutions(proposal_review / passed / vote_result の座標)",
                      "factors/update._bump(state_update の座標)"),
    "y":             ("同上",),
    "node":          ("traces._deposit", "rumors._birth", "scheduler._serve_attrib"),
    "loc":           ("truth_ledger._present",),
    "sleeping":      ("truth_ledger._present",),
    "building":      ("scheduler._serve_attrib(同一 建物/階 の応対)",),
    "floor":         ("scheduler._serve_attrib",),
    "home_building": ("freedom_p2.pick_home(空き住戸の判定 = 名簿全体を走査)",
                      "population._occupied_buildings(同上)"),
    "visitor":       ("gossip._roll_day", "household.date_dest", "mobility.cohabit_day"),
    "evicted":       ("mobility.cohabit_day",),
    "dead":          ("assets 系の相続(present 述語経由だが同型の読み)",),
    "language":      ("diversity.cross_barrier(話者の言語)",),
    "tourist":       ("diversity(観光客文脈)",),
    "status":        ("status.attract_bonus / buy_multiplier / feed_exposure_weight",),
    "controllability": ("worldview._ctrl_apply(読み + 書き)",),
    "occupation":    ("work.role_weight(産出重み)",),
    "org_role":      ("work.role_weight",),
    "traits":        ("mind.model_of_id(未実体化 id の再導出)",),
    "mind":          ("mind.model_of",),
    "states":        ("factors/update._bump ← tools.Institutions(提案の可決で作者へ)",),
    "adopted":       ("tools.Events.tick(主催者の造語を参加者へ教える)",),
    "partner_id":    ("household.date_dest", "mobility.cohabit_day"),
    "housemates":    ("household.context_line",),
    "mem":           ("gossip._seed_knowers(対象の関係台帳から知る者を作る)",
                      "mobility._mutual_closeness(相互 closeness)",
                      # ★実測(pool ON mock で退場者を追跡プロキシに差し替えて発見):
                      #   入場者の実体化中は sim.agents がまだ前日の名簿なので present 述語が
                      #   退場者を「在場」と答える窓があり、顔なじみ張りがそこで退場者の
                      #   関係台帳へ record_contact する。差し替えを実体化の**後**へ置いて
                      #   その窓では従来どおりフル Agent を見せ、かつ _RefMem 側にも
                      #   MemoryStore と同一実装の record_contact を持たせて両側を塞いだ。
                      "simulation._link_colocated(顔なじみ = record_contact)"),
    "_fact_beliefs": ("truth_ledger._transmit_pass(話者の fact 信念)",),
    "_cohabit_since": ("mobility.cohabit_day(読み + 書き)",),
    "_cohabiting":   ("mobility.cohabit_day",),
    "_train_seen":   ("transit_interior._copresence(1 日 1 対の印。読み + 書き)",),
    "_isl_buf":      ("scheduler._isl_accumulate(実行時リングバッファ。読み + 書き)",),
    "money":         ("A2 の vital 台帳と突き合わせる観測点(退場者への支払いは在場述語で"
                      "塞いだが、読み自体は残る)",),
    "account":       ("同上",),
    "pool_pid":      ("回転の突合・観測(退場者の名簿 id)",),
}


# --------------------------------------------------------------------------- #
# ② 落とす欄 = **育つ容器**(ここだけが RAM の実体)。落とす理由を 1 件ずつ書く。
#    ★どれも「退場者に対して読む読者が 1 つも無い」ことを grep + 実測で確認した欄。
#    ★静的なペルソナ属性(persona 文)は再来街時に ``build_agent`` が P5 record から
#      決定論で同一に組み直す(``world/pool.py`` の dehydrate 冒頭の規約と同じ)。
# --------------------------------------------------------------------------- #
_DROPPED: dict[str, str] = {
    "persona":       "LLM へ渡す自己紹介文。P5 record から決定論で再構築される静的属性",
    "mem.buffer":    "未統合エピソード(既定 30 件・本文つき)。退場者から読む読者は無い",
    "mem.episodes":  "統合済み顕著記憶(既定 120 件・本文つき)。同上",
    "mem.day_summaries": "日次要約(日記)。同上",
    "beliefs":       "内省の書き戻し先(LLM 本文の列)。同上",
    "heard_counts":  "item_id → 聞いた回数の Counter。同上",
    "visits":        "EPR の訪問回数 Counter(ノード数ぶん育つ)。同上",
    "said":          "自分の直近発話(反復抑制用)。同上",
    "day_plan":      "その日の計画。回転をまたいで意味を持たない",
    "schedule":      "14 日先までの約束帳。退避辞書(dormant)が正典",
    "route":         "残りの経路。その旅に固有(pool.dehydrate がゾーン所有を運ばないのと同じ)",
    "self_dev":      "熟達ストック。退避辞書が正典",
    "part_time":     "バイト先の段取り。退場者から読む読者は無い",
    "drive_mods":    "SDT 動機プラグイン。同上",
    "drift_p":       "個人別ドリフト率。同上",
    "reflect_p":     "内省の個人パラメータ。同上",
    "sev_pending":   "当日の発症予約。同上",
    "joint_today":   "当日の共同行動。同上",
    "self_model":    "自己像(反射)。同上",
    "behav_today":   "当日の行動カウント。同上",
    "behav_ema":     "行動ベースライン EMA。同上",
    "_reply_to":     "話しかけられた印(その step 限り)",
}


# --------------------------------------------------------------------------- #
# 運ぶ欄の確定
#   _ALWAYS  = Agent の**宣言済み**欄のうち、落とす表に載っていないもの
#              (= 素のスカラー欄すべて + 上の表が要求する小さな容器)。
#   _OPTIONAL= 宣言に無い**動的属性**。持っている個体だけ運ぶ(持っていない個体に
#              生やすと ``getattr(x, name, default)`` の既定値が変わってしまう)。
# --------------------------------------------------------------------------- #
_DECLARED: tuple[str, ...] = tuple(f.name for f in dataclasses.fields(Agent))

#: 宣言済み欄のうち運ぶもの(``mem`` は下の ``_RefMem`` へ差し替えるので別扱い)。
_ALWAYS: tuple[str, ...] = tuple(
    n for n in _DECLARED if n != "mem" and n not in _DROPPED)

#: 宣言に無い動的属性で、退場者から読まれるもの(在るときだけ運ぶ)。
_OPTIONAL: tuple[str, ...] = (
    "pool_pid",            # プール名簿 id(回転の突合・観測)
    "party_size",          # L4 来街者 record の同行人数(S-R5)
    "org_id", "org_role", "org_line",   # 組織配属(work.role_weight / 所属行)
    "mind",                # 第88 心のモデル(mind.model_of)
    "tourist", "language",  # 多様性 H5(diversity.cross_barrier)
    "controllability",     # 世界観 C2(worldview._ctrl_apply は読んで書く)
    "_reputation",         # 評判スカラー(T6 合成地位の 1 項)
    "_cohabiting", "_cohabit_since",    # 同棲判定(mobility.cohabit_day)
    "_fact_beliefs",       # 真偽台帳の信念(truth_ledger._transmit_pass)
    "_train_seen",         # 車内共在の 1 日 1 対の印(transit_interior)
    "_isl_buf",            # 実行時リングバッファ(scheduler._isl_accumulate)
    "_impact_on",          # 日内衝撃ゲージを追うか(factors/update._impact_note)
    "_known_events",       # 既知イベント(tools.Events)
    "visit_purpose", "is_foreign",      # 名簿由来の観測欄(第114 レーン 1b)
)

_SLOTS: tuple[str, ...] = _ALWAYS + _OPTIONAL + ("mem", "_extra")


class _RefMem:
    """退場者の記憶のうち**関係台帳だけ**を保つ器(``AgentRef.mem``)。

    ``gossip._seed_knowers`` は ``target.mem.relations`` から「直近に会話した相手」を
    引き、``mobility._mutual_closeness`` は ``a.mem.relations`` を読む。どちらも
    退場者に対して到達しうるので、**同じ dict オブジェクトをそのまま共有**する
    (コピーしない = 退場のたびに台帳を二重に持たない)。フル Agent 側は解放されるので、
    生き残るのは台帳 1 つだけになる。
    ``episodes`` / ``buffer`` / ``day_summaries`` は**持たない** = 触れば AttributeError。

    ★``record_contact`` は ``MemoryStore`` の**実装をそのまま借りる**。あちらは
      ``self.relations`` と ``self.relations_max`` しか触らない純粋な台帳操作なので、
      同じ dict を共有している以上、フル Agent に対して呼んだのと 1 バイトも変わらない
      (写しを作ると「顔なじみを張った相手が退場者だと台帳が二重になる」)。
    """

    __slots__ = ("relations", "relations_max", "relations_evict",
                 "relations_tier_acq")

    #: ★同一実装を借りる(独立の写しを作らない = 台帳操作の真実源を 1 つに保つ)。
    record_contact = MemoryStore.record_contact
    #: S7 退避梯子(第149)も同じ実装を借りる。借りないと `relations_evict="tiered"` の
    #: ときだけ退場者の台帳操作が別の規則になる(= 同じ dict を 2 つの規則が触る)。
    _evict_rank = MemoryStore._evict_rank
    _evict_tiered = MemoryStore._evict_tiered
    _EVICT_FLOOD = MemoryStore._EVICT_FLOOD
    _EVICT_WEAK = MemoryStore._EVICT_WEAK
    _EVICT_DORMANT = MemoryStore._EVICT_DORMANT
    _EVICT_TIER1 = MemoryStore._EVICT_TIER1

    def __init__(self, relations=None, relations_max: int = 0,
                 relations_evict: str = "lru", relations_tier_acq: float = 2.0):
        self.relations = {} if relations is None else relations
        self.relations_max = int(relations_max)
        self.relations_evict = str(relations_evict)
        self.relations_tier_acq = float(relations_tier_acq)

    def __getattr__(self, name):        # __slots__ に無い欄(episodes 等)
        raise AttributeError(
            f"AgentRef.mem に '{name}' はありません(退場者は関係台帳だけを保つ"
            " = 記憶本文は世界から降ろした)。退場者の記憶本文が要る処理は"
            " sim._dormant(退避辞書)を見るか、sim.present_agent で在場者に限ること。")

    def __repr__(self):
        return f"<_RefMem relations={len(self.relations)}>"


class AgentRef:
    """退場者の軽量参照。``sim.agent_by_id`` にだけ座り、``sim.agents`` には入らない。"""

    __slots__ = _SLOTS

    #: テスト/検査から読む「運ぶ欄」「落とす欄」の表(実装と同じ 1 つの源)。
    CARRIED = _ALWAYS + _OPTIONAL + ("mem",)
    DROPPED = tuple(sorted(_DROPPED))
    READ_BY = _READ_BY

    def __init__(self, agent):
        set_ = object.__setattr__
        for name in _ALWAYS:
            set_(self, name, getattr(agent, name))
        for name in _OPTIONAL:
            if hasattr(agent, name):
                set_(self, name, getattr(agent, name))
        mem = getattr(agent, "mem", None)
        set_(self, "mem", _RefMem(getattr(mem, "relations", None),
                                  getattr(mem, "relations_max", 0),
                                  getattr(mem, "relations_evict", "lru"),
                                  getattr(mem, "relations_tier_acq", 2.0))
             if mem is not None else _RefMem())
        set_(self, "_extra", None)

    # ---- 未定義属性は**明示的に**落とす(静かな既定値を作らない)----------------
    def __getattr__(self, name):
        try:
            extra = object.__getattribute__(self, "_extra")
        except AttributeError:                     # unpickle 途中など
            extra = None
        if extra is not None and name in extra:
            return extra[name]
        raise AttributeError(self._why(name))

    def __setattr__(self, name, value):
        """書き込みは通す(現行と同値 = 世界には残らない)。未知の欄は ``_extra`` へ。"""
        try:
            object.__setattr__(self, name, value)
            return
        except AttributeError:                     # __slots__ に無い欄
            pass
        try:
            extra = object.__getattribute__(self, "_extra")
        except AttributeError:                     # unpickle 途中(スロット未設定)
            extra = None
        if extra is None:
            extra = {}
            object.__setattr__(self, "_extra", extra)
        extra[name] = value

    def _why(self, name: str) -> str:
        try:
            who = f"agent_id={object.__getattribute__(self, 'id')}"
        except AttributeError:                     # id すら未設定(unpickle 途中)
            who = "agent_id=?"
        if name in _DROPPED:
            return (f"AgentRef({who}) は '{name}' を運びません: {_DROPPED[name]}。"
                    " 退場者(プール回転で街を出た個体)の索引は軽量参照で、育つ容器は"
                    " 世界から降ろしてある。この欄が要る処理は sim.present_agent で"
                    " 在場者に限るか、sim._dormant(退避辞書)を読むこと。")
        return (f"AgentRef({who}) に '{name}' はありません(退場者の軽量参照)。"
                " 退場者に対してこの欄を読む必要があるなら"
                " src/society/agents/ref.py の _READ_BY / _OPTIONAL へ**明示的に**"
                " 足すこと(既定値で静かに済ませると『街を出た人だけ値が違う』という"
                " 気づけない挙動変化になる)。")

    # ---- Agent が持つ唯一のメソッド。退場者への記録は現行でも捨てられる ----------
    def remember(self, text: str, kind: str = "event",
                 importance_bonus: float = 0.0) -> None:
        """退場者への「覚えておく」は**何もしない**(現行と同値)。

        現行実装でも、退場時に ``dehydrate`` でスナップショットを採った**後**の
        フル Agent への書き込みは、再来街時に ``build_pool_agent`` が新しい実体を
        作って ``hydrate`` が退避辞書で上書きするため、世界に 1 バイトも残らない。
        ここで no-op にするのは「残らない」という**現行の帰結**をそのまま実装に
        写したもので、L1 にも退避辞書にも差は出ない。
        """
        return None

    def __repr__(self):
        try:
            return f"<AgentRef id={self.id} name={self.name!r}>"
        except AttributeError:
            return "<AgentRef(未初期化)>"


def is_ref(obj) -> bool:
    """その参照が退場者の軽量参照か(``sim.agent_by_id`` の値の型検査)。"""
    return isinstance(obj, AgentRef)


def to_ref(obj):
    """フル Agent なら軽量参照へ、既に軽量参照ならそのまま返す(冪等)。

    ★旧 checkpoint(``departed`` にフル Agent が入った pool サイドカー)の互換口でも
      あり、resume 時にここを通して同じ軽さへ揃える。
    """
    return obj if isinstance(obj, AgentRef) else AgentRef(obj)
