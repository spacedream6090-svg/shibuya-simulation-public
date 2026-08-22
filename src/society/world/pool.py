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
    """退場者のスリム状態を退避する**二層**ストア(レーン R1 A2)。

    層 1 ``rich``  … 記憶・信念・関係・可塑性など。上限 + LRU(persona-pool §3.3)。
                     cap>0 のとき「リッチな記憶を持てる母集団」に上限を設け、超過分は
                     最も古い個体から捨てる。cap=0 は無制限(= 出荷時の既定)。
    層 2 ``vital`` … **金銭・債権・人口会計**。容量に依らず**絶対に捨てない**。

    ★なぜ分けるか(見つかった破れ): 旧実装は LRU 超過で退避状態を**丸ごと**捨てていた。
      捨てられた個体が再来街すると ``pop`` が None を返し、``build_agent`` が
      **初期所持金で新規鋳造**する。つまり退場時の残高・預金・家賃債権・未清算の勤務実績が
      **真に消え、同時に無から現金が湧いていた**。しかもどの保存チャネル(L1 / finance /
      provenance)にも痕跡が残らないので、総マネー保存の検査でも検出できなかった。
      vital は 1 体あたり数百バイト(既定値の欄はキーを作らない)なので、100 万体の
      プールでも数百 MB に収まる = 上限を掛ける動機がそもそも無い。

    ★挙動不変の範囲: cap=0(出荷時の既定)では rich が 1 件も捨てられないので、
      ``pop`` は従来どおり rich を返し、vital は 1 度も使われない = 完全にバイト一致。
      挙動が変わるのは「cap>0 で実際に LRU 破棄が起きたランの、その個体の再来街」だけ。
    """

    def __init__(self, cap: int = 0):
        self.cap = int(cap)
        self._d: "OrderedDict[str, dict]" = OrderedDict()
        #: pid -> vital 台帳(容量非依存)。挿入順は save 順(決定論)。
        self._vital: dict[str, dict] = {}

    # ---- 退避 -----------------------------------------------------------------
    def save(self, pid: str, state: dict) -> None:
        """退避(rich は LRU つき・vital は容量非依存)。vital は rich から**派生**する。"""
        vital = vital_of(state)
        if vital:
            self._vital[pid] = vital
        else:
            self._vital.pop(pid, None)      # 全欄が既定値 = 運ぶものが無い(キーを作らない)
        self._d[pid] = state
        self._d.move_to_end(pid)
        if self.cap > 0:
            while len(self._d) > self.cap:
                self._d.popitem(last=False)     # LRU 退避(★vital は道連れにしない)

    # ---- 取り出し -------------------------------------------------------------
    def pop(self, pid: str) -> dict | None:
        """rich を取り出す(vital を**上書きマージ**して返す)。rich が無ければ None。

        vital は rich から派生した写しなので、健全なら上書きは恒等(= バイト一致)。
        マージを明示するのは「金銭・債権の真実源は vital」という順序を実装で示すため。
        rich が LRU で捨てられている場合は ``pop_vital`` が拾う(呼び出し側の分岐)。
        """
        rich = self._d.pop(pid, None)
        if rich is None:
            return None                      # ★vital は消費しない(pop_vital に残す)
        vital = self._vital.pop(pid, None)
        if vital:
            _merge_vital(rich, vital)
        return rich

    def pop_vital(self, pid: str) -> dict | None:
        """恒久台帳だけを取り出す(rich が LRU 破棄済みの個体の再来街で使う)。"""
        return self._vital.pop(pid, None)

    def peek(self, pid: str) -> dict | None:
        return self._d.get(pid)

    def peek_vital(self, pid: str) -> dict | None:
        return self._vital.get(pid)

    # ---- 観測(economy_sfc の hh_dormant 列が読む)------------------------------
    def vital_money_total(self) -> float:
        """退避中(街に居ない)個体の**現金 + 口座**の総額。決定論(dict の値の和)。"""
        return float(sum(float(v.get("money", 0.0)) + float(v.get("account", 0.0))
                         for v in self._vital.values()))

    def vital_claims_total(self) -> float:
        """退避中の個体が抱える**家賃債権**(未払い残)の総額。"""
        return float(sum(float((v.get("econ") or {}).get("rent_due", 0.0))
                         for v in self._vital.values()))

    def rebuild_vital(self) -> None:
        """rich から vital を作り直す(**旧 pool サイドカー**の互換口)。

        vital 層より前に書かれたサイドカーには vital が無い。resume でそのまま続けると
        「rich が LRU で消えた個体だけ台帳も無い」状態になるので、読める分だけ復元する
        (rich まで捨てられていた個体は元から復元不能 = 旧ランの既知の欠落として残る)。
        """
        for pid, state in self._d.items():
            if pid in self._vital:
                continue
            vital = vital_of(state)
            if vital:
                self._vital[pid] = vital

    def __contains__(self, pid: str) -> bool:
        # ★vital だけが残っている個体も「退避されている」= True(rich は LRU で捨てられうる)。
        return pid in self._d or pid in self._vital

    def __len__(self) -> int:
        return len(self._d)                  # リッチ層の件数(従来の意味を変えない)

    def __repr__(self):
        return (f"<DormantStore rich={len(self._d)} vital={len(self._vital)}"
                f" cap={self.cap}>")


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
               ("med_dest_building", str, ""), ("med_from_node", str, ""),
               ("med_last_tick", int, -1))

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


#: 家計の月次会計(口座 E5)+ 賃金多様性 WAGE の未清算実績。
#  ★これらはどれも「① 個体固有 ② 日を跨いで持続 ③ 金の受け取りを変える」を満たすのに
#    1 つも退避辞書に載っていなかった = **プール回転のたびに毎回 0 へ戻っていた**。
#    その結果、月給・家賃・滞納・立退き・破産のどれも「回転する層(= 在場者の大半)」では
#    原理的に成立しなかった(月給の元になる勤務日数が翌日には消えるため)。
#  ★口座 OFF / WAGE OFF のランでは全欄が既定値なので**キーが 1 つも足されない**
#    (tests/test_pool_rotation.py の dict 等値がそのまま守られる)。
_ECON_FIELDS = (("period_income", float, 0.0), ("work_days", int, 0),
                ("rent_due", float, 0.0), ("arrears_days", int, 0),
                ("last_salary", float, 0.0), ("evicted", bool, False),
                ("bankrupt_until", int, 0),
                # WAGE(第112): 未清算の勤務実績・最後に清算した日・持ち越した賞与。
                # ★プラン本体(`_wage_plan`)は運ばない —— 再来街時に台帳と安定ハッシュから
                #   同じものが**決定論で組み直される**ので、運ぶと二重の真実源になる。
                ("wp_days", int, 0), ("wp_settled_day", int, -1),
                ("wp_bonus_pending", bool, False),
                # §ROLE(第114 1a): L5 役割職(タクシー/配信者/議員)の最終清算日。
                # ★これが無いと回転のたびに -1 へ戻り、議員報酬の給料日探索
                #   (`_wage_fires` の (last, today] 区間)が毎回 45 日ぶん遡って
                #   **同じ月の給料日を何度も払う**(歩合側は当日 1 回なので無害)。
                ("rp_settled_day", int, -1))


# ---- レーン乙(第113): 搬送族の**第2弾**(B クラス)-----------------------------------
# 判定基準は D1 と同じ 3 つ(① 個体固有 ② 日を跨いで持続 ③ 行動 or L1 の発火可否を変える)。
# 設計も同じ「**非既定値のときだけキーを足す**」= 当該機構 OFF のランでは退避 dict が
# 1 バイトも変わらない(tests/test_pool_rotation.py の dict 等値がそのまま守られる)。

#: 単純スカラーで運べる族。既定値は「その属性を持たない個体で getattr が返す値」と
#  厳密に一致させる(でないと機構 OFF のランでキーが生えて既存の dict 等値が割れる)。
_MISC_FIELDS = (
    # 退屈ゲージ(cognition/drive.py が日々育てる探索の駆動源)。
    ("_boredom", float, 0.0),
    # 勾留(制度深化2)。運ばないと回転で即釈放される。
    ("detained_until", int, 0),
    # 宿泊 Wave L。★連泊数を運ばないと「checkin だけ増えて checkout が出ない」
    #   単調乖離になり、max_nights の上限も回転のたびに 0 へ戻る。
    ("lodging_nights", int, 0), ("lodging_node", str, ""), ("lodging_poi", str, ""),
    # 出動中の印(救急 / 設備 / 消防)。運ばないと「出動したまま帰署しない」担い手が出る。
    # ★既定値は各 module の getattr 既定(-1 / "")と同一。
    ("city_ops_ems_until", int, -1), ("city_ops_ems_home", str, ""),
    ("facility_call_until", int, -1), ("facility_call_home", str, ""),
    ("incenv_fire_until", int, -1), ("incenv_fire_home", str, ""),
    # 多様性 H5(観光客型 / 非日本語話者)。起動時 1 回の配布なので運ばないと再来街で消える。
    ("tourist", bool, False), ("language", str, ""),
    # 内面 inner_life の長期目標(趣味は list なので下の専用経路)。
    ("life_goal", str, ""),
    # 世界観 C2 の予測誤差の累計(wv_expect と対で意味を持つ)。
    ("_wv_err_sum", float, 0.0), ("_wv_err_n", int, 0),
    # ★レーン乙 ブロック6(不在中に時間が進まないバイアス)の**要**: 日境界の減衰/忘却を
    #   最後に適用した日。これを運ばないと帰街のたびに「経過 0 日」へ戻り、街を出るだけで
    #   関係も評判も悪評も一切風化しない個体ができる(= 居続けた個体より有利)。
    ("_rel_decay_day", int, -1), ("_gossip_fade_day", int, -1),
    # ★第114 OBS(認知スタックの搬送棚卸し): 深い内省の**日スケールの予約と不応期**。
    #   どちらも「日内衝撃ゲージの閾値超え」が予約する量で、既定値は Agent の -1 と同一。
    #   運ばないと (a) 予約したまま街を出た個体の深い内省が黙って取り消され
    #   (b) 不応期(既定 3 日)が帰街のたびに 0 へ戻る = 街を出入りする個体だけ
    #   「侵入的 → 熟慮的の遅延」という設計が成立しない。判定基準 3 つを全て満たす。
    ("deep_due_day", int, -1), ("deep_cooldown_until_day", int, -1),
    # ★存在の内生化 POP(``society/population.py``): 名簿そのものの増減に効く**多日スケールの
    #   個体状態**。判定基準 3 つ(① 個体固有 ② 日を跨いで持続 ③ 行動 or 発火可否を変える)を
    #   全て満たす —— これらを運ばないと、
    #     - ``pop_stress`` … 居住ストレスの蓄積(Wolpert 閾値の左辺)が回転のたびに 0 へ戻り、
    #       **街を出入りする個体は原理的に転出できない**(居続けた個体だけが転出する系統偏り)。
    #     - ``pop_days``   … 「この街で過ごした日数」= 定着(転入)の唯一の来街履歴。
    #       運ばないと L4 は永遠に 1 日目のままで、定着が構造的に発火しない。
    #     - ``pop_settled`` / ``pop_emigrated`` … 台帳(``sim._pop_state``)が真実源だが、
    #       入場フックと退避の**両方**で同じ事実に着地することを機械固定するために運ぶ。
    #     - ``pop_pair_since`` … 出生のパートナー継続日数の起点。
    #   既定値は各所の getattr 既定と厳密に一致させてある = POP OFF のランではキーが 1 つも
    #   生えない(tests/test_pool_rotation.py の dict 等値がそのまま守られる)。
    ("pop_stress", float, 0.0), ("pop_days", int, 0),
    ("pop_settled", bool, False), ("pop_emigrated", bool, False),
    ("pop_pair_since", int, -1),
    # ★第144: 職場が宣言している営業曜日("mon-fri" / "mon-sat" / "all" …)。
    #   `work.bind_workplace` が台帳 shift_pattern.days から写す文字列で、
    #   `world.calendar.respect_work_days` が ON のとき勤務ゲート/賃金ゲートの入力になる
    #   (= ③「行動 or 発火可否を変える」)。①個体固有 ②日を跨いで持続 も満たす。
    #   ★束ねが OFF のラン(既定)ではこの属性が**生えない**ので退避 dict は 1 バイトも
    #     変わらない。既定値 "" は getattr の既定と厳密に一致。
    #   ※`work_days`(int・給料日までの勤務日数)とは**別物**。名前が似ているだけ。
    ("work_dow", str, ""),
)

#: 世帯(H2)。★レーン乙 ブロック3: 世帯の静的部分は pool record から決定論で組み直すが、
#  **動的に生じた変化**(同棲 merge / 別離 split / partner 形成)は record に無いので運ぶ。
#  partner_id / housemates は None・list なので下の専用経路で扱う。
_HH_FIELDS = (("household_id", str, ""), ("household_kind", str, ""),
              ("household_role", str, ""),
              # ★「世帯を抜けた」印(mobility._split_household)。これが無いと、分離した個体が
              #   再来街のたびに名簿由来の世帯へ座り直されて**別離が忘れられる**
              #   (household_id=None は「元から単身」と区別がつかないため)。
              ("_hh_detached", bool, False),
              # ★「シミュレーション中に実際に引っ越した」印(mobility._set_home)。
              #   立っている個体だけ home を運ぶ(下の _HOME_FIELDS)。
              ("_home_moved", bool, False))

#: 転居後の住居。**``_home_moved`` が立っているときだけ**運ぶ(それ以外の個体の home は
#  名簿 + 世帯から決定論で組み直されるので運ぶと二重の真実源になる = dehydrate 冒頭の
#  「静的なペルソナ属性は保存しない」規約と整合する)。
_HOME_FIELDS = (("home_building", str, ""), ("home_node", str, ""),
                ("home_floor", int, 1))

#: ATT(第143)。層B の注意ブロック(有限スロット)と層A の注意履歴(プライミング)。
#  判定基準 3 つ(① 個体固有 ② 日を跨いで持続 ③ 行動 or L1 の発火可否を変える)を満たす:
#  スロットはプロンプトの「いま気にしていること」節そのもので、層A の priority にも
#  w_slot で効く。履歴は層A の history 項(直近この話者を注意した割合)。
#  ★属性名は `society.attention.SLOTS_KEY` / `HISTORY_KEY` と同一でなければならないが、
#    `_RUMOR_KEY` と同じ理由(world/ は下層 = society 直下の機構 module を import すると
#    依存の向きが逆流する)で写しを持つ。一致は tests/test_attention.py が機械固定する。
#  ★ATT OFF のランではどちらの属性も**生えない**ので退避 dict は 1 バイトも変わらない。
_ATTN_SLOTS_KEY = "attention_slots"
_ATTN_HIST_KEY = "_attn_hist"

_VISITS_KEY = "visits"            # EPR preferential return(node -> 訪問回数)
_SELF_DEV_KEY = "self_dev"        # T4 自助努力(skill/fitness の累積ストック)
_SCHEDULE_KEY = "schedule"        # 14 日先まで持てる約束帳
_HOBBIES_KEY = "hobbies"          # inner_life の趣味(list[str])
_DIALOG_KEY = "_dialog_hist"      # 相手別リングバッファ(挿入順 = LRU の意味を持つ)
_TL_PENDING_KEY = "_tl_pending"   # 真偽台帳の検証待ち(★truth_ledger.py は凍結=不触)
_WV_EXPECT_KEY = "wv_expect"      # 世界観 C2 の混雑予測 {(place, band): 期待値}
_HOUSEMATES_KEY = "housemates"
_PARTNER_KEY = "partner_id"


def _plast_slim(agent) -> dict | None:
    """可塑性(g_update)の学習状態を丸ごと 1 つの nested dict へ(OFF は None=キー無し)。

    ★なぜ「非既定値の欄だけ」ではなく丸ごとか: ``plasticity.ensure`` は
      ``_fire_g is not None`` で早期 return するので、g を復元すると **eta/lam/g0/θ倍率が
      二度と作り直されない**。欄ごとに既定値で間引くと、たまたま既定と等しくなった欄だけが
      復元されずに素の値へ落ちる。族としては「g_update ON で ensure を通った個体」か
      「そうでないか」の 2 値しかないので、``_fire_g`` の有無を唯一の門にして丸ごと運ぶ。
    ★``ensure`` が早期 return して ``hub.stream(NOISE_STREAM, id)`` を引かなくなるが、
      ``RngHub.stream`` は呼ぶたびに**新しい Generator を派生する**(共有状態を持たない)
      ので、引かないことで他の draw 列が 1 粒も動かない。
    """
    g = getattr(agent, "_fire_g", None)
    if not g:
        return None

    def _d(name):
        return {str(k): float(v) for k, v in (getattr(agent, name, None) or {}).items()}

    return {
        "g": _d("_fire_g"), "g0": _d("_fire_g0"), "g_init": _d("_fire_g_init"),
        "ebar": _d("_fire_ebar"), "credit": _d("_fire_credit"),
        "pending": [{"at": int(r["at"]), "base": float(r["base"]),
                     "shares": {str(k): float(v) for k, v in r["shares"].items()}}
                    for r in (getattr(agent, "_fire_pending", None) or [])],
        "eta": float(getattr(agent, "_fire_eta", 0.0)),
        "lam": float(getattr(agent, "_fire_lam", 0.0)),
        "theta_m": float(getattr(agent, "_fire_theta_m", 1.0)),
        "day_n": int(getattr(agent, "_fire_day_n", 0)),
        "fbar": float(getattr(agent, "_fire_fbar", 0.0)),
    }


def _plast_apply(agent, st: dict) -> None:
    """可塑性の学習状態を戻す(チャンネル位置 idx を int へ復号)。"""
    def _d(key):
        return {int(k): float(v) for k, v in (st.get(key) or {}).items()}

    agent._fire_g = _d("g")
    agent._fire_g0 = _d("g0")
    agent._fire_g_init = _d("g_init")
    agent._fire_ebar = _d("ebar")
    agent._fire_credit = _d("credit")
    agent._fire_pending = [{"at": int(r["at"]), "base": float(r["base"]),
                            "shares": {int(k): float(v) for k, v in r["shares"].items()}}
                           for r in (st.get("pending") or [])]
    agent._fire_eta = float(st.get("eta", 0.0))
    agent._fire_lam = float(st.get("lam", 0.0))
    agent._fire_theta_m = float(st.get("theta_m", 1.0))
    agent._fire_day_n = int(st.get("day_n", 0))
    agent._fire_fbar = float(st.get("fbar", 0.0))


def _watch_slim(agent) -> dict | None:
    """監視仕様(watch spec)= 「この人がいま何を注視すると宣言したか」(OFF は None)。

    第114 の認知スタック棚卸しで見つかった B7 の残り: 可塑性 11 欄は運ぶのに、**同じ
    発火式 S に入るもう 1 つの個体状態**である監視仕様が 1 つも運ばれていなかった。
    判定基準は D1/B7 と同じ 3 つを全て満たす:
      ① 個体固有(LLM がその人の言葉で出した ô とトリガ)
      ② 日を跨いで持続 —— 設計 §2.2 は「**有効期限は持たせない**(時間で失効させると
         時間軸が裏口から入る)」と明記していて、上書きされるまで前回仕様が生きる
      ③ 発火可否を変える(ô は S の予測項、トリガは Σ_j w_ij の項そのもの)
    運ばないと、再来街した個体は「何も注視していない素の状態」に戻り、次に監視仕様を
    出す LLM 呼が来るまで**驚きの基準が persistence 予測へ後退する**。

    ★正規化(JSON 安全 + 往復同値): expect はチャンネル位置 idx → 値の dict、triggers は
      (名前, idx, 演算子, 値) の 4 つ組列。watch OFF のランでは属性が無いので None =
      退避 dict が 1 バイトも変わらない(tests/test_pool_rotation.py の dict 等値を守る)。
    """
    spec = getattr(agent, "_fire_watch", None)
    if not spec:
        return None
    return {
        "expect": {str(k): float(v) for k, v in (spec.get("expect") or {}).items()},
        "triggers": [[str(nm), int(idx), str(op), float(val)]
                     for (nm, idx, op, val) in (spec.get("triggers") or [])],
    }


def _watch_apply(agent, st: dict) -> None:
    """監視仕様を戻す(チャンネル位置 idx を int へ復号)。"""
    agent._fire_watch = {
        "expect": {int(k): float(v) for k, v in (st.get("expect") or {}).items()},
        "triggers": [(str(nm), int(idx), str(op), float(val))
                     for (nm, idx, op, val) in (st.get("triggers") or [])],
    }


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
    # ★レーン乙 A3: 上位 rel_cap 件の切り方。従来は「接触回数 count の多い順、同数は
    #   相手 id 昇順」だったが、これには 2 つの系統的な偏りがあった:
    #   (a) friend_graph が注入した辺は **count=1**(会話の実績ではなく初期条件)なので、
    #       台帳が埋まると**友人関係から先に捨てられる**(親友であるほど落ちやすい)。
    #   (b) 同数のタイを常に id 昇順で解くので、若い id の相手が系統的に生き残る。
    #   closeness(= 実際に育った親密度。relations OFF のランでは 1 件も存在しない)を
    #   第一キーに、最終接触を第三キーに足す。**relations OFF では全員 closeness も
    #   last_step も持たないので並びは従来と完全に同一**(= 既存ランはバイト一致)。
    rels = sorted(mem.relations.items(),
                  key=lambda kv: (-float(kv[1].get("closeness", 0.0) or 0.0),
                                  -int(kv[1].get("count", 0)),
                                  -int(kv[1].get("last_step", 0) or 0),
                                  kv[0]))[:rel_cap]
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
    econ = _fields_slim(agent, _ECON_FIELDS)      # 口座 E5 の月次会計 + WAGE の未清算実績
    if econ:
        state["econ"] = econ
    # --- レーン乙(第113) ブロック1: 搬送族の第2弾。上と同じ「非既定のときだけキーを足す」 ---
    misc = _fields_slim(agent, _MISC_FIELDS)
    if misc:
        state["misc"] = misc
    hh = _fields_slim(agent, _HH_FIELDS)          # 世帯(動的変化ぶん。ブロック3 と対)
    if hh:
        state["hh"] = hh
    if getattr(agent, "_home_moved", False):      # 実際に引っ越した個体だけ home を運ぶ
        state["home"] = {"home_building": str(getattr(agent, "home_building", "") or ""),
                         "home_node": str(getattr(agent, "home_node", "") or ""),
                         "home_floor": int(getattr(agent, "home_floor", 1) or 1)}
    mates = getattr(agent, _HOUSEMATES_KEY, None)
    if mates:
        state["housemates"] = [int(x) for x in mates]
    partner = getattr(agent, _PARTNER_KEY, None)
    if partner is not None:
        state["partner_id"] = int(partner)
    visits = getattr(agent, _VISITS_KEY, None)    # ★B1 EPR: 「よく行く場所」の唯一の源
    if visits:
        # node id は市街地図のノード集合で有界(episodes/relations のような切りは要らない)。
        state["visits"] = {str(k): int(v) for k, v in visits.items() if int(v)}
        if not state["visits"]:
            del state["visits"]
    self_dev = getattr(agent, _SELF_DEV_KEY, None)             # T4 熟達ストック
    if self_dev:
        state["self_dev"] = {str(k): float(v) for k, v in self_dev.items()}
    sched = getattr(agent, _SCHEDULE_KEY, None)                # 約束帳(14 日先まで)
    if sched:
        state["schedule"] = [{"day": int(e["day"]), "when": str(e.get("when", "")),
                              "what": str(e.get("what", "")),
                              "place": str(e.get("place", "")),
                              "with": [int(i) for i in e.get("with", [])],
                              "src_step": int(e.get("src_step", 0))} for e in sched]
    hobbies = getattr(agent, _HOBBIES_KEY, None)               # inner_life の趣味
    if hobbies:
        state["hobbies"] = [str(x) for x in hobbies]
    dhist = getattr(agent, _DIALOG_KEY, None)                  # 相手別の直近対話
    if dhist:
        # ★dict の**挿入順が LRU の意味**(先頭 = 最古)。list of pairs で順序ごと運ぶ。
        state["dialog_hist"] = [[int(pid), [[str(n), str(t)] for (n, t) in turns]]
                                for pid, turns in dhist.items() if turns]
        if not state["dialog_hist"]:
            del state["dialog_hist"]
    pend = getattr(agent, _TL_PENDING_KEY, None)               # 真偽台帳の検証待ち
    if pend:
        state["tl_pending"] = {"fact": str(pend["fact"]), "until": int(pend["until"])}
    wv = getattr(agent, _WV_EXPECT_KEY, None)                  # C2 混雑予測(tuple キー)
    if wv:
        # キーは (place_key, band) のタプル = JSON 非安全。3 つ組の list へ正規化する。
        state["wv_expect"] = [[str(k[0]), int(k[1]), float(v)] for k, v in wv.items()]
    plast = _plast_slim(agent)                    # 可塑性 g_update の学習状態(OFF は None)
    if plast:
        state["plast"] = plast
    # --- 第114 OBS: 認知スタックの搬送棚卸し(B7 の横展開)---------------------------------
    # 監視仕様(watch spec)= 発火式 S のもう一方の個体状態。設計 §2.2 が「有効期限を
    # 持たせない」と決めている以上、街を出ただけで消えるのは仕様との食い違いである。
    watch = _watch_slim(agent)                    # watch OFF は None(退避 dict はバイト不変)
    if watch:
        state["watch"] = watch
    # 行動ベースライン(無意識層の EMA)= 「いつもの自分」。**何日もかけて育つ量**なので、
    # 運ばないと再来街した個体は base=today に落ちて逸脱が原理的に検出されなくなる
    # (= 街を出入りする個体だけ「最近の自分」の 1 行が永久に空になる系統バイアス)。
    # ★対の ``implicit_self``(その 1 行の本文)は**運ばない**: あちらは reflection.py が
    #   「揮発的な作動自己 = 日次上書き」と設計宣言している派生テキストで、ここ(EMA)から
    #   翌日の日境界に決定論で作り直される。持続する状態だけを運ぶ、が D1/B7 の基準。
    ema = getattr(agent, "behav_ema", None)
    if ema:
        state["behav_ema"] = {str(k): float(v) for k, v in ema.items()}
    # 価値 4 軸の充足 sat。**多日スケールの量**(日次 15% しか中立へ戻らないので、
    # 満たされない状態は何日も尾を引く)なのに、再来街時は _init_pool_agent_extras が
    # 一律 0.5(完全に中立)へ据え直していた = 街を出るだけで「渇き」が消える。
    # ★sat_mods は運ばない: sat の純関数で、同じ日境界の values.decay_daily が
    #   復元後の sat から作り直す(二重の真実源を作らない)。
    sat = getattr(agent, "sat", None)
    if sat:
        state["sat"] = {str(k): float(v) for k, v in sorted(sat.items())}
    # --- ATT(第143): 注意ブロック(層B)と注意履歴(層A)。**属性が在って非空のときだけ**
    #     キーを足す = ATT OFF のランでは退避 dict が 1 バイトも変わらない。
    slots = getattr(agent, _ATTN_SLOTS_KEY, None)
    if slots:
        state["attention_slots"] = [
            {"kind": str(s.get("kind", "")), "target": str(s.get("target", "")),
             "why": str(s.get("why", "")),
             "salience": float(s.get("salience", 0.0) or 0.0),
             "since": int(s.get("since", 0) or 0)} for s in slots]
    ahist = getattr(agent, _ATTN_HIST_KEY, None)
    if ahist:
        state["attn_hist"] = [int(x) for x in ahist]
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
    econ = state.get("econ")                      # 口座 E5 の月次会計 + WAGE の未清算実績
    if econ:
        _fields_apply(agent, econ, _ECON_FIELDS)
    # --- レーン乙(第113) ブロック1: 搬送族の第2弾。**キー欠落を許容**(旧 退避辞書 /
    #     機構 OFF のラン)= その場合は属性を 1 つも生やさない = 現行と完全同一。
    misc = state.get("misc")
    if misc:
        _fields_apply(agent, misc, _MISC_FIELDS)
    hh = state.get("hh")                          # 世帯(動的変化ぶん)
    if hh:
        _fields_apply(agent, hh, _HH_FIELDS)
        if hh.get("_hh_detached"):
            # ★入場時の世帯着席(household.bind_pool_household)は hydrate の**前**に走るので、
            #   分離済みの個体はここで明示的に席を外す(順序に依存しない打ち消し)。
            agent.household_id = None
            agent.household_kind = ""
            agent.household_role = ""
            agent.housemates = []
    home = state.get("home")                      # 転居後の住居(_home_moved の個体のみ)
    if home:
        _fields_apply(agent, home, _HOME_FIELDS)
    mates = state.get("housemates")
    if mates:
        setattr(agent, _HOUSEMATES_KEY, [int(x) for x in mates])
    partner = state.get("partner_id")
    if partner is not None:
        setattr(agent, _PARTNER_KEY, int(partner))
    visits = state.get("visits")                  # ★B1 EPR:「いつもの場所」を持ち越す
    if visits:
        setattr(agent, _VISITS_KEY, Counter({str(k): int(v) for k, v in visits.items()}))
    self_dev = state.get("self_dev")
    if self_dev:
        setattr(agent, _SELF_DEV_KEY, {str(k): float(v) for k, v in self_dev.items()})
    sched = state.get("schedule")
    if sched:
        setattr(agent, _SCHEDULE_KEY,
                [{"day": int(e["day"]), "when": str(e.get("when", "")),
                  "what": str(e.get("what", "")), "place": str(e.get("place", "")),
                  "with": [int(i) for i in e.get("with", [])],
                  "src_step": int(e.get("src_step", 0))} for e in sched])
    hobbies = state.get("hobbies")
    if hobbies:
        setattr(agent, _HOBBIES_KEY, [str(x) for x in hobbies])
    dhist = state.get("dialog_hist")              # 挿入順(LRU)ごと戻す
    if dhist:
        setattr(agent, _DIALOG_KEY,
                {int(pid): [(str(n), str(t)) for (n, t) in turns] for pid, turns in dhist})
    pend = state.get("tl_pending")
    if pend:
        setattr(agent, _TL_PENDING_KEY,
                {"fact": str(pend["fact"]), "until": int(pend["until"])})
    wv = state.get("wv_expect")                   # 3 つ組の list → (place, band) キーへ戻す
    if wv:
        setattr(agent, _WV_EXPECT_KEY,
                {(str(p), int(b)): float(v) for (p, b, v) in wv})
    plast = state.get("plast")                    # 可塑性 g_update の学習状態
    if plast:
        _plast_apply(agent, plast)
    watch = state.get("watch")                    # 第114 OBS: 監視仕様(ô + トリガ)
    if watch:
        _watch_apply(agent, watch)
    ema = state.get("behav_ema")                  # 第114 OBS: 行動ベースライン(無意識層)
    if ema:
        agent.behav_ema = {str(k): float(v) for k, v in ema.items()}
    sat = state.get("sat")                        # 第114 OBS: 価値 4 軸の充足(多日スケール)
    if sat:
        agent.sat = {str(k): float(v) for k, v in sat.items()}
    slots = state.get("attention_slots")          # 第143 ATT 層B: 注意ブロック(有限スロット)
    if slots:
        setattr(agent, _ATTN_SLOTS_KEY,
                [{"kind": str(s.get("kind", "")), "target": str(s.get("target", "")),
                  "why": str(s.get("why", "")),
                  "salience": float(s.get("salience", 0.0) or 0.0),
                  "since": int(s.get("since", 0) or 0)} for s in slots])
    ahist = state.get("attn_hist")                # 第143 ATT 層A: 注意履歴(プライミング)
    if ahist:
        setattr(agent, _ATTN_HIST_KEY, [int(x) for x in ahist])


# --------------------------------------------------------------------------- #
# 恒久台帳(vital)= 金銭・債権・人口会計。**LRU の対象外**(レーン R1 A2)
#
# 何を vital に入れるかの基準(この 3 つを全部満たすものだけ):
#   ① 捨てると**総量が保存しない**(現金が消える / 無から湧く)、または名簿の増減が壊れる。
#   ② 再来街時に決定論で組み直せない(build_agent は初期値を鋳造するので**別の値**になる)。
#   ③ 1 体あたりのバイト数が小さい(既定値の欄はキーを作らないので、多くの個体で数十バイト)。
# 逆に「記憶・信念・関係・可塑性・評判」は ① を満たさない(捨てても総量は保存する)ので
# rich のまま = 上限 + LRU の対象。
#
# 実物の全数列挙(``dehydrate`` の実装から採る。推測で書かない):
#   money                       … agent.money(現金)          … dehydrate の "money"
#   account                     … agent.account(口座残高)    … dehydrate の "account"
#   econ(= _ECON_FIELDS 一式)  … dehydrate の "econ"
#     rent_due                  … 家賃債権(未払い残)
#     arrears_days              … 滞納の連続日数(立退きの左辺)
#     period_income             … 前回家賃日からの入金累計
#     work_days                 … 給料日までに完遂した本業勤務日数(= 未払賃金の元)
#     last_salary               … 直近の月給支給額
#     evicted / bankrupt_until  … 立退き / 破産の継続状態
#     wp_days                   … WAGE の未清算な勤務実績(= 未払賃金の元)
#     wp_settled_day            … 最後に清算した日(二重支給/取りこぼしの境界)
#     wp_bonus_pending          … 持ち越した賞与
#     rp_settled_day            … L5 役割職の最終清算日(議員報酬の二重払い防止)
#   pop(= _POP_FIELDS 一式)    … dehydrate の "misc" の中の人口会計
#     pop_stress / pop_days / pop_settled / pop_emigrated / pop_pair_since
# --------------------------------------------------------------------------- #

#: 人口会計(存在の内生化 POP)の欄名。``_MISC_FIELDS`` から**同じ実体**を切り出す
#  (二重の真実源を作らないため、名前だけをここに書いて型/既定値はあちらから引く)。
_VITAL_POP_NAMES: tuple[str, ...] = ("pop_stress", "pop_days", "pop_settled",
                                     "pop_emigrated", "pop_pair_since")
_POP_FIELDS = tuple(f for f in _MISC_FIELDS if f[0] in _VITAL_POP_NAMES)


def vital_of(state: dict) -> dict:
    """退避スリム状態(rich)→ 恒久台帳(vital)。**rich からの派生**(独立の真実源ではない)。

    ★``money`` / ``account`` は ``dehydrate`` が**必ず**書く欄なので、0 円でもキーを作る
      (「0 円で街を出た」と「台帳が無い」を区別できないと、再鋳造との違いが立たない)。
    ★``econ`` / ``pop`` は非既定の欄しか無いので、機構 OFF のランでは丸ごとキーが生えない。
    """
    out: dict = {}
    if "money" in state:
        out["money"] = float(state["money"])
    if "account" in state:
        out["account"] = float(state["account"])
    econ = state.get("econ")
    if econ:
        out["econ"] = dict(econ)
    misc = state.get("misc") or {}
    pop = {k: misc[k] for k in _VITAL_POP_NAMES if k in misc}
    if pop:
        out["pop"] = pop
    return out


def _merge_vital(rich: dict, vital: dict) -> None:
    """vital を rich へ**上書き**マージする(金銭・債権の真実源は vital 側)。

    健全な状態では両者は同値なので**1 バイトも動かない**(値が一致する欄は代入すらしない
    = 退避辞書のオブジェクト同一性まで保つ)。マージを明示するのは、将来 vital 側だけを
    更新する経路が生えたときに順序が決まっていることを実装で示すため。
    """
    for key in ("money", "account"):
        if key in vital and rich.get(key) != vital[key]:
            rich[key] = vital[key]
    econ = vital.get("econ")
    if econ and (rich.get("econ") or {}) != econ:
        rich["econ"] = {**(rich.get("econ") or {}), **econ}
    pop = vital.get("pop")
    if pop and not all((rich.get("misc") or {}).get(k) == v for k, v in pop.items()):
        rich["misc"] = {**(rich.get("misc") or {}), **pop}


def overlay_vital(agent, vital: dict) -> None:
    """恒久台帳だけを再来街エージェントへ当てる(``build_pool_agent`` の**後**に呼ぶ)。

    ``hydrate`` との違い: 記憶・信念・関係・状態・可塑性には**1 バイトも触らない**。
    リッチな退避状態が LRU 上限で捨てられた個体は「記憶を失って街へ戻る」= 設計どおりだが、
    **財布まで失う**のは設計ではなく破れ(無からの創出 + 真の消失)なので、ここだけを戻す。
    キー欠落は許容する(旧サイドカー / 機構 OFF のランでは属性を 1 つも生やさない)。
    """
    if "money" in vital:
        agent.money = float(vital["money"])
    if "account" in vital:
        agent.account = float(vital["account"])
    econ = vital.get("econ")
    if econ:
        _fields_apply(agent, econ, _ECON_FIELDS)
    pop = vital.get("pop")
    if pop:
        _fields_apply(agent, pop, _POP_FIELDS)
