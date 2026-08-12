"""ペルソナプールの遅延読み + ドーマント退避ストア(W2 P3 / ローテーション機構)。

正典: docs/plans/w2-execution-plan.md §4 P3 / docs/plans/persona-pool.md §5・§9。

構成:
  - `PoolStore`      : P5 シャード(JSONL)を**遅延読み**。全 full record を RAM に展開せず、
                       presence 判定に要るスリム記述子(PresenceRec)と各 id のバイトオフセット・
                       密な安定整数(id_of=agent.id 用)を1回のストリーム走査で作る。full record は
                       present 個体分だけ get() で都度読む(id_of は日跨ぎ不変・衝突ゼロ・int32 安全)。
  - `DormantStore`   : 退場者のスリム状態を退避(コストゼロ・記憶保持)。有界(上限+LRU)。
  - `dehydrate/hydrate`: エージェント⇄スリム状態の往復(記憶・信念・所持金・関係上位を持続)。

RAM 方針(persona-pool §9): 「present = LLM コスト源」と「プール = 記憶保持の源」を分離する。
  full persona record(persona 文込み)は present 個体分のみ RAM に載る。プール全体の full record は
  一切常駐させない。presence 判定に要る最小フィールド(スリム)だけを常駐させる。
"""
from __future__ import annotations

import json
import sys
from collections import Counter, OrderedDict
from pathlib import Path

from .presence import PresenceRec


def _slim(rec: dict) -> PresenceRec:
    """P5 full record → presence 判定に要るスリム記述子(PresenceRec)。"""
    sp = rec.get("shift_pattern") or {}
    dp = rec.get("duty_pattern") or {}
    return PresenceRec(
        pid=str(rec["id"]),
        key=str(rec.get("presence", "resident")),
        work_days=str(rec.get("work_days") or sp.get("days") or ""),
        cadence=str(rec.get("visit_cadence") or ""),
        visit_rate=float(rec.get("visit_rate") or 0.0),
        revisit=bool(rec.get("revisit", False)),
        duty_days=str(dp.get("days") or "all"),
        # PRES-A1②: 来街目的(曜日・天候の共変量キー)。json.loads は行ごとに**別の str
        # オブジェクト**を作るので、100 万件では同じ 7 語が 70 万個の実体になる。
        # `sys.intern` で 7 個へ畳む(索引の常駐 RAM を増やさないための唯一の細工)。
        visit_purpose=sys.intern(str(rec.get("visit_purpose") or "")),
    )


class PoolStore:
    """P5 シャードの遅延読み。presence 用スリム索引 + full record の随時読み込み。"""

    def __init__(self, pool_dir):
        self.dir = Path(pool_dir)
        meta_path = self.dir / "meta.json"
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self._shards = self.meta.get("shards", [])
        self._records: list[PresenceRec] | None = None            # スリム索引(常駐)
        self._offsets: dict[str, tuple[str, int]] | None = None    # pid -> (shard file, byte offset)
        self._index: dict[str, int] | None = None                  # pid -> 密な安定整数(agent.id)

    # ---- 索引構築(1回だけ・シャードをストリーム走査。full record は保持しない)----
    def _build_index(self) -> None:
        records: list[PresenceRec] = []
        offsets: dict[str, tuple[str, int]] = {}
        index: dict[str, int] = {}
        i = 0
        for sh in self._shards:
            rel = sh["file"]
            with open(self.dir / rel, "rb") as f:
                while True:
                    pos = f.tell()
                    raw = f.readline()
                    if not raw:
                        break
                    line = raw.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    pid = str(rec["id"])
                    offsets[pid] = (rel, pos)
                    index[pid] = i                          # プール列挙順の密な整数(0..N-1)
                    i += 1
                    records.append(_slim(rec))
        self._records = records
        self._offsets = offsets
        self._index = index

    def id_of(self, pid: str) -> int:
        """ペルソナ id(str)→ agent.id 用の**密で安定**な整数(プール列挙順)。

        - 密(0..N-1・pool_size < 2^31)=**衝突ゼロ**で観測 agent_id の int32 列に収まる。
        - 同一プールなら pid→int は常に同一(日を跨いでも・resume でも不変=同一人物の観測が繋がる)。
        - ハッシュ衝突(100万規模で int32 は誕生日近似ほぼ確実に衝突)を避けるため密割当を採る。
        """
        if self._index is None:
            self._build_index()
        return self._index[pid]

    def presence_records(self) -> list[PresenceRec]:
        """presence 純関数へ渡すスリム記述子の全リスト(決定論=シャード順)。"""
        if self._records is None:
            self._build_index()
        return self._records

    def get(self, pid: str) -> dict:
        """1 ペルソナの full record を遅延読み(present 個体のみ hydrate 用に呼ぶ)。"""
        if self._offsets is None:
            self._build_index()
        rel, pos = self._offsets[pid]
        with open(self.dir / rel, "rb") as f:
            f.seek(pos)
            raw = f.readline()
        return json.loads(raw.decode("utf-8"))

    def __contains__(self, pid: str) -> bool:
        if self._offsets is None:
            self._build_index()
        return pid in self._offsets

    def __len__(self) -> int:
        return len(self.presence_records())


class DormantStore:
    """退場者のスリム状態を退避する有界ストア(上限 + LRU。persona-pool §3.3)。

    cap>0 のとき「リッチな記憶を持てる母集団」に上限を設け、超過分は最も古い個体から退避
    (evict = 保存状態を捨てる。再来街時はプールから素で再構築)。cap=0 は無制限。
    """

    def __init__(self, cap: int = 0):
        self.cap = int(cap)
        self._d: "OrderedDict[str, dict]" = OrderedDict()

    def save(self, pid: str, state: dict) -> None:
        self._d[pid] = state
        self._d.move_to_end(pid)
        if self.cap > 0:
            while len(self._d) > self.cap:
                self._d.popitem(last=False)     # LRU 退避

    def pop(self, pid: str) -> dict | None:
        return self._d.pop(pid, None)

    def peek(self, pid: str) -> dict | None:
        return self._d.get(pid)

    def __contains__(self, pid: str) -> bool:
        return pid in self._d

    def __len__(self) -> int:
        return len(self._d)


# ---- スリム状態の往復(dehydrate/hydrate)----------------------------------------
# 退避に含めるのは「経験で育った可変状態」のみ。静的なペルソナ属性(persona 文/traits/home/
# work/閾値)は再来街時に build_agent が P5 record から**決定論**で同一に再構築するので保存しない。
_EP_CAP = 30        # 退避する統合エピソードの上限(persona-pool §3.3: 非前景の episodes を小さく)
_REL_CAP = 20       # 退避する関係台帳の上限(接触回数の多い上位)

# 噂の知識を持つ動的属性名。**`society.rumors.RUMOR_KEY` と同一でなければならない**が、
# world/ 層から society 直下の機構 module を import すると依存の向きが逆流する(world は
# 下層)ため、ここでは文字列で持ち、一致は tests/test_pool_rotation.py が機械固定する。
_RUMOR_KEY = "_rumors"

# 身体(病気・重症度)の退避対象。**`society.health._SLIM_FIELDS` と同一でなければならない**が、
# 上の `_RUMOR_KEY` と同じ理由(world/ は下層なので society 直下の機構 module を import すると
# 依存の向きが逆流する)で写しを持つ。一致は tests/test_health_severity.py が機械固定する。
# 各要素 = (属性名, 型, 既定値)。**既定値と等しい欄はキーを作らない**(退避 dict のバイト列不変)。
# ★レーン D1(2026-08-12)の総点検で末尾 4 欄(sev_collapse_step / chronic_days /
#   withdrawn / fatigue)を追加した。正典側 `health._SLIM_FIELDS` と必ず同じ並びで持つ。
_HEALTH_FIELDS = (("sick", bool, False), ("sick_until", int, -1),
                  ("severity", int, 0), ("sev_channel", str, ""),
                  ("sev_until", int, -1), ("sev_outcome_step", int, -1),
                  ("sev_fatal", bool, False), ("sev_presentee", bool, False),
                  ("sev_cared", bool, False), ("sev_confirmed", int, -1),
                  ("sev_frailty", float, 1.0), ("dead", bool, False),
                  ("sev_collapse_step", int, -1), ("chronic_days", int, 0),
                  ("withdrawn", bool, False), ("fatigue", float, 0.0))

_HEALTH_PENDING = "sev_pending"

# ---- レーン D1(第109): 退避フィールドの**総点検**で見つかった未搭載 ----------------
# 判定基準(この 3 つを全部満たすものだけ運ぶ):
#   ① 個体に固有で ② 日を跨いで持続し ③ その値が行動判断 or L1 の発火可否を変える。
# 逆に「決定論で組み直せる静的属性」「その日 / その旅に固有の段取り」「プロセス内の
# 観測カウンタ」は運ばない(ゾーン所有を運ばないのと同じ理由。下の注記に列挙)。

#: H2 医療(搬送中 / 在院中の印)。**`society.medical._clear` の全数と同一**でなければ
#  ならない(`_RUMOR_KEY` と同じ理由で写しを持つ。一致は tests/test_pool_rotation.py の
#  `test_med_field_list_mirrors_medical_clear` が機械固定する)。
#  ★運ばないと「入院中の個体が回転で即退院して病院から消える」= 在院日数の較正が壊れる。
_MED_FIELDS = (("med_admitted", bool, False), ("med_transport_until", int, -1),
               ("med_until", int, -1), ("med_since", int, -1),
               ("med_crew", int, -1), ("med_confirmed", int, -1),
               ("med_node", str, ""), ("med_poi", str, ""),
               ("med_dest_node", str, ""), ("med_dest_poi", str, ""),
               ("med_dest_building", str, ""), ("med_from_node", str, ""))

#: H4 酩酊の印(減衰する ttl)。**`society.incidents_interpersonal.INTOX_KEY` と同一**。
#  当該 module が「正直な限界 4: プール退場で酩酊の印を失う」と自己申告していたもの。
_INTOX_KEY = "_inc_intox_until"

#: 負の評判(第61)の知識。`_rumors`(IF-C 残②)と**まったく同型**の「街を出ると忘れる」。
#  known = target_id → 学習日 / heard = target_id → 聞いた相異なる知人の集合
#  (集合は membership と len でしか使われない = sorted list で運んで良い。JSON 安全)。
_GOSSIP_KNOWN_KEY = "_gossip_known"
_GOSSIP_HEARD_KEY = "_gossip_heard"

#: 真偽台帳(第73 Part B)の信念。`agent.beliefs`(内省の文字列)は運んでいるのに
#  fact 信念だけ落ちるのは非対称で、checkpoint 側が `_tl_facts` を保存して
#  「信念の参照先が迷子にならない」ようにしている配慮とも食い違っていた。
#  ★`truth_ledger.py` は凍結 SPEC_FILES なので**あちらは 1 バイトも触らない**
#  (属性名の写しをここに置くのは `_RUMOR_KEY` と同じ作法)。
_FACT_BELIEFS_KEY = "_fact_beliefs"

#: 評判スカラー(relations.gain_reputation / reputation_decay が育てる)。
#  合成地位 T6(status.py)の 1 項でもあるので、落とすと再来街で地位が下がる。
_REPUTATION_KEY = "_reputation"

#: 所持品(goods の直近購入・conf で有界)。所持金は運んでいるのに買った物だけ消えていた。
_BELONGINGS_KEY = "_goods_belongings"

#: 世界観 C2 の可制御性(``worldview._ctrl_apply`` が日次で育てる Bandura の outcome
#  expectancy)。★``worldview.ctrl_line`` がプロンプト 1 行を出し分けるので**行動に効く**。
#  起点は ontology 群の決定論値なので、運ばないと再来街で「学習前」に戻る
#  (``theta_drift`` を運んでいるのと同じ理由。属性が在るときだけ運ぶ = OFF は無風)。
_CTRL_KEY = "controllability"


def _fields_slim(agent, fields: tuple) -> dict:
    """(属性名, 型, 既定値)表 → 退避辞書。**既定値と等しい欄はキーを作らない**。

    これが「健康な個体 / 機構 OFF のランでは退避 dict のバイト列が現行と完全同一」を
    成立させている芯である(tests/test_pool_rotation.py が dict 等値で機械固定)。
    """
    out: dict = {}
    for name, cast, default in fields:
        got = getattr(agent, name, default)
        got = str(got or "") if cast is str else cast(got)
        if got != default:
            out[name] = got
    return out


def _fields_apply(agent, state: dict, fields: tuple) -> None:
    """退避辞書 → 再来街エージェント(**キー欠落を許容** = 旧 退避辞書からは何も生やさない)。"""
    for name, cast, _default in fields:
        if name in state:
            setattr(agent, name, cast(state[name]))


def _health_slim(agent) -> dict:
    """身体状態の退避辞書(健康な個体・健康 OFF のランでは **空 dict** = 現行と完全同一)。"""
    out = _fields_slim(agent, _HEALTH_FIELDS)
    pending = getattr(agent, _HEALTH_PENDING, None)
    if pending:
        out[_HEALTH_PENDING] = dict(pending)       # プリミティブのみ(JSON 安全・実体非共有)
    return out


def _health_apply(agent, state: dict) -> None:
    """退避辞書 → 再来街エージェント(**キー欠落を許容** = 旧 退避辞書からは何も生やさない)。"""
    _fields_apply(agent, state, _HEALTH_FIELDS)
    pending = state.get(_HEALTH_PENDING)
    if pending:
        setattr(agent, _HEALTH_PENDING, dict(pending))

# 実効値(conf: pool.relations_cap / pool.episodes_cap)。既定は上の素値=挙動不変。
# ★なぜ引数ではなくモジュール状態か: dehydrate の呼び出し口は
#   `_phase_pool_rotation`(engine/scheduler.py)の 1 行だけで、そこは sim も cfg も
#   引数に持たない汎用フェーズである。Simulation が pool を据えるときに一度 configure()
#   するのが、呼び出し口を書き換えずに conf を通す最小の経路
#   (pool 無効のランでは configure すら呼ばれず、値は素値のまま)。
_caps = {"ep": _EP_CAP, "rel": _REL_CAP}


def configure(*, rel_cap: int | None = None, ep_cap: int | None = None) -> None:
    """dehydrate の既定上限を conf 値で据える(Simulation 構築時に 1 回)。

    第75バッチ実測の持ち越し(M-3): dunbar(認知枠)の**休眠**関係は closeness=0 に
    退避されるが、`dehydrate` の関係台帳は「接触回数の多い上位 20 件」で切られるため、
    休眠した弱い紐帯はプール退場のたびに**再会する前に台帳から落ちやすい**。
    観察ラン(dunbar ON + pool ON)ではここを広げて「忘却/再会」の軌跡を保てるようにする。
    既定値は現行の素値そのままなので、指定しない限り挙動は 1 バイトも変わらない。
    """
    if rel_cap is not None:
        _caps["rel"] = max(0, int(rel_cap))
    if ep_cap is not None:
        _caps["ep"] = max(0, int(ep_cap))


def caps() -> dict:
    """いま有効な退避上限(観測・テスト用)。"""
    return dict(_caps)


def dehydrate(agent, *, ep_cap: int | None = None, rel_cap: int | None = None) -> dict:
    """present エージェント → 退避スリム状態(記憶・信念・所持金・関係上位を持続)。

    ep_cap / rel_cap を省くと configure() で据えた実効値(既定 = 30 / 20)を使う。
    """
    ep_cap = _caps["ep"] if ep_cap is None else int(ep_cap)
    rel_cap = _caps["rel"] if rel_cap is None else int(rel_cap)
    mem = agent.mem
    rels = sorted(mem.relations.items(),
                  key=lambda kv: (-int(kv[1].get("count", 0)), kv[0]))[:rel_cap]
    state = {
        "beliefs": list(agent.beliefs),
        "day_summaries": list(mem.day_summaries),
        "episodes": [(int(e.step), e.text, e.kind, float(e.importance))
                     for e in mem.episodes[-ep_cap:]],
        "relations": {int(k): dict(v) for k, v in rels},
        "adopted": sorted(agent.adopted),
        "heard_counts": dict(agent.heard_counts),
        "money": float(agent.money),
        "account": float(getattr(agent, "account", 0.0)),
        "opinion": float(agent.opinion),
        "status": float(getattr(agent, "status", 0.0)),
        "self_model": getattr(agent, "self_model", None),
        "theta_drift": float(getattr(agent, "theta_drift", 0.0)),
    }
    # --- 第98バッチ 小粒A: 台帳の残課題 2 件(IF-C 残② / 竹-4 残③)を塞ぐ ---------------
    # ★**属性が在って非空のときだけ**キーを足す。rumors / physics OFF のランでは退避 dict が
    #   現行と 1 バイトも変わらない(tests/test_pool_rotation.py が dict 等値で機械固定)。
    # ★正規化して入れる(プリミティブ / dict のみ)= JSON 安全 + 往復同値。
    rumors = getattr(agent, _RUMOR_KEY, None)     # IF-C 残②: 街を出ると噂を忘れる(reach 過小)
    if rumors:
        # 挿入順 = 知った順(rumors.py の決定論反復順そのもの)を dict の順序で保つ。
        state["rumors"] = {
            str(iid): {"role": str(r["role"]), "redundant": int(r.get("redundant", 0)),
                       "step": int(r.get("step", 0)), "src": str(r.get("src", ""))}
            for iid, r in rumors.items()}
    body = getattr(agent, "_phys_body", None)     # 竹-4 残③: 直近の群集体感(Perception.body 3 欄)
    if body:
        state["phys_body"] = {"blocked": float(body["blocked"]),
                              "contact": float(body["contact"]),
                              "local_density": float(body["local_density"])}
    # --- レーン H1(2026-08-10): **監査で見つかったバグの同梱修正** -----------------------
    # ★``sick`` / ``sick_until`` はこれまで退避辞書に 1 つも載っていなかった = 住民がプール
    #   回転(退場 → 再来街)のたびに病気を忘れていた。重症度(severity 一式)と一緒に塞ぐ。
    # ★同じ「属性が在って**既定値でない**ときだけキーを足す」設計なので、健康な個体・
    #   健康 OFF のランでは退避 dict が現行と 1 バイトも変わらない
    #   (tests/test_pool_rotation.py の dict 等値がそのまま守られる)。
    health = _health_slim(agent)
    if health:
        state["health"] = health
    # --- レーン D1(2026-08-12): 退避フィールドの**総点検**(H1 の横展開)------------------
    # 以下はどれも「① 個体固有 ② 日を跨いで持続 ③ 行動 or 発火可否を変える」を満たすのに
    # 1 つも載っていなかった = 回転のたびに黙って忘れていた。上と同じ「非既定のときだけ
    # キーを足す」設計なので、当該機構 OFF のランでは退避 dict が 1 バイトも変わらない。
    med = _fields_slim(agent, _MED_FIELDS)        # H2: 搬送中 / 在院中の印(即退院を防ぐ)
    if med:
        state["med"] = med
    intox = int(getattr(agent, _INTOX_KEY, 0) or 0)   # H4: 酩酊(RAT の酒項)
    if intox:
        state["intox_until"] = intox
    known = getattr(agent, _GOSSIP_KNOWN_KEY, None)   # 第61: 悪評を知っている対象 → 学習日
    if known:
        state["gossip_known"] = {int(k): int(v) for k, v in known.items()}
    heard = getattr(agent, _GOSSIP_HEARD_KEY, None)   # 第61: 対象 → 聞いた相異なる知人
    if heard:
        # 値は集合だが membership / len でしか使われない(gossip.py 明記)= sorted list で
        # 運ぶ(JSON 安全 + 反復順の非決定を持ち込まない)。空集合の欄は落とす。
        _hd = {int(k): sorted(int(x) for x in v) for k, v in heard.items() if v}
        if _hd:
            state["gossip_heard"] = _hd
    facts = getattr(agent, _FACT_BELIEFS_KEY, None)   # 第73 Part B: fact_id → 信念レコード
    if facts:
        state["fact_beliefs"] = {str(k): dict(v) for k, v in facts.items()}
    rep = float(getattr(agent, _REPUTATION_KEY, 0.0) or 0.0)   # 評判スカラー(T6 の 1 項)
    if rep:
        state["reputation"] = rep
    belongings = getattr(agent, _BELONGINGS_KEY, None)         # 所持品(有界リスト)
    if belongings:
        state["belongings"] = [str(x) for x in belongings]
    ctrl = getattr(agent, _CTRL_KEY, None)        # C2 可制御性(ontology/worldview ON のみ)
    if ctrl is not None:
        state["controllability"] = float(ctrl)
    # ★**ゾーン所有(_phys_zone と _FIELDS の走行レコード)は意図的に運ばない**。あれは
    #   「いま歩いている経路 agent.route の途中」という**その旅に固有の**状態で、再来街時は
    #   build_pool_agent が別の node / route で個体を組み直すため、復元すると physics._run_zone が
    #   存在しない経路の続きを積分しようとする(= 破損)。ゾーン側に占有者名簿は無く
    #   (所有は agent._phys_zone の単一値のみ)、退場した個体は走査対象から外れるだけなので、
    #   運ばないことによる取り残しも発生しない。持続するのは体感(_phys_body)だけ。
    return state


def hydrate(agent, state: dict) -> None:
    """退避スリム状態 → 再来街エージェントへ復元(build_agent 後にオーバレイ)。"""
    from ..agents.memory import Episode
    agent.beliefs = list(state.get("beliefs", []))
    agent.mem.day_summaries = list(state.get("day_summaries", []))
    agent.mem.episodes = [Episode(step=s, text=t, kind=k, importance=imp)
                          for (s, t, k, imp) in state.get("episodes", [])]
    agent.mem.relations = {int(k): dict(v) for k, v in state.get("relations", {}).items()}
    agent.adopted = set(state.get("adopted", []))
    agent.heard_counts = Counter(state.get("heard_counts", {}))
    agent.money = float(state.get("money", agent.money))
    agent.account = float(state.get("account", getattr(agent, "account", 0.0)))
    agent.opinion = float(state.get("opinion", agent.opinion))
    agent.status = float(state.get("status", 0.0))
    agent.self_model = state.get("self_model", None)
    agent.theta_drift = float(state.get("theta_drift", 0.0))
    # 第98バッチ 小粒A: 噂の知識と群集体感を戻す。**キー欠落を許容**する(旧 退避辞書 /
    # rumors・physics OFF のラン)= その場合は属性を 1 つも生やさない = 現行と完全同一。
    rumors = state.get("rumors")
    if rumors:
        setattr(agent, _RUMOR_KEY, {
            str(iid): {"role": str(r["role"]), "redundant": int(r.get("redundant", 0)),
                       "step": int(r.get("step", 0)), "src": str(r.get("src", ""))}
            for iid, r in rumors.items()})
    body = state.get("phys_body")
    if body:
        agent._phys_body = {"blocked": float(body["blocked"]),
                            "contact": float(body["contact"]),
                            "local_density": float(body["local_density"])}
    health = state.get("health")                  # H1: 病気/重症度を持ち越す(キー欠落は許容)
    if health:
        _health_apply(agent, health)
    # --- レーン D1(2026-08-12): 総点検で足した族。**キー欠落を許容**する(旧 退避辞書 /
    #     機構 OFF のラン)= その場合は属性を 1 つも生やさない = 現行と完全同一。
    med = state.get("med")                        # H2: 搬送 / 在院の印
    if med:
        _fields_apply(agent, med, _MED_FIELDS)
    intox = state.get("intox_until")              # H4: 酩酊の ttl
    if intox:
        setattr(agent, _INTOX_KEY, int(intox))
    known = state.get("gossip_known")             # 第61: 悪評の知識
    if known:
        setattr(agent, _GOSSIP_KNOWN_KEY, {int(k): int(v) for k, v in known.items()})
    heard = state.get("gossip_heard")             # 第61: 聞いた相異なる知人(list → set へ戻す)
    if heard:
        setattr(agent, _GOSSIP_HEARD_KEY,
                {int(k): {int(x) for x in v} for k, v in heard.items()})
    facts = state.get("fact_beliefs")             # 第73 Part B: fact 信念
    if facts:
        setattr(agent, _FACT_BELIEFS_KEY, {str(k): dict(v) for k, v in facts.items()})
    rep = state.get("reputation")                 # 評判スカラー
    if rep:
        setattr(agent, _REPUTATION_KEY, float(rep))
    belongings = state.get("belongings")          # 所持品(有界リスト)
    if belongings:
        setattr(agent, _BELONGINGS_KEY, [str(x) for x in belongings])
    ctrl = state.get("controllability")           # C2 可制御性(キー欠落は許容)
    if ctrl is not None:
        setattr(agent, _CTRL_KEY, float(ctrl))
