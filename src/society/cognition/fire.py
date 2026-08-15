"""閾値発火 + 認知イベントキュー + 同期バリア(第81バッチ 2026-08-01・**既定 OFF**)。

正典
----
- docs/plans/source/cognition-design-record.md §2.3(発火判定 S)・§3.3(同期バリア/
  ダブルバッファ)・§3.4(却下: 負荷連動振り分け)
- docs/plans/source/physics-instructions.md Part P0(イベントキュー・発火源・k 非依存則の再定義)
- docs/plans/cognition-physics-plan.md §4 第81行 / §6-3(ユーザー修正: 発火源は驚きだけで
  なく**内省・会話も第一級**)

何を解く問題か
--------------
現行は「1 step = 10 分」が世界の進行と思考の頻度を兼ねている。本 module はその 2 つを
分離し、**いつ考えるか**を (発火時刻, agent_id) の全順序イベントキューで決める。

    S_i(t) = Σ_c  g_ic · |o_c(t) − ô_ic| / σ_c  +  Σ_j w_ij·[trigger_j]
    発火 ⟺ S_i(t) > θ_i(t)                                     (設計 §2.3)

**第82バッチで埋まった箇所**(この 3 つは既定 OFF の追加トグルで入る):
  - ô(期待値)= `cognition.watch.enabled` ON で **LLM の watch spec**(`watch.py`)。
    OFF のままなら第81 の persistence 予測(前回の認知イベント時点の観測値)。
  - g(感度)= `cognition.g_update.enabled` ON で **履歴が動かす**(`plasticity.py`:
    慣れ ē / 感作 r / 引き戻し λ)。OFF のままなら全 usable チャンネルで **1.0 固定**。
  - 名前付きトリガ Σ_j w_ij·[trigger_j] = watch ON で **制約付き DSL**(`watch.py`)。
    OFF のままなら恒等 0(= S は第1項のみ)。
  - θ = 較正テーブルの**仮値** × conf スケール × 個体倍率(g_update ON で恒常性が動く)。
    θ の絶対水準の較正そのものは**第83のパイロット**のまま。

決定論(離散イベントシミュレーションの標準作法)
------------------------------------------------
同時刻イベントの処理順が実行環境で変わると世界が変わる。これは DES の古典的な事故で、
「同時刻イベントには tie-breaking 規則を与えて全順序を強制し再現性を確保する」のが標準
(Peschlow & Martini 2010, *A discrete-event simulation tool for the analysis of
simultaneous events*, NSTOOLS; Jha & Bagrodia 2000, *Simultaneous events and lookahead
in simulation protocols*, ACM TOMACS 10(3):241-267 — 「シミュレータごとに tie-break が
場当たり的で結果が再現しない」)。並列 DES でも同じ処方で、タイムスタンプに tie-break
値を連ねて全順序を作る(McGlohon & Carothers 2021, arXiv:2105.00069)。SimPy はこれを
ヒープキー `(time, priority, 単調増加 id)` として実装している。

本 module は **1 エージェントにつき生きたイベントを高々 1 個**に保つことで、
キーを `(発火時刻, agent_id)` の 2 要素だけで**厳密な全順序**にしている
(第3要素の挿入順カウンタが要らない = 挿入順が結果に漏れる経路が原理的に無い)。
ヒープには世代トークン付きの項目を積み、古い項目は pop 時に捨てる(lazy deletion)。

同期バリア + ダブルバッファ(設計 §3.3)
----------------------------------------
「結果の適用は step 末に一斉。state(t) から読んで state(t+1) に書く。到着順に反映すると
完了順序(= GPU の気まぐれ)が結果に漏れる」。ABM でこれは古くから知られた問題で、
同期更新(全員が state(t) を読み state(t+1) へ書く)と非同期更新(逐次適用)は
**同じモデルでも結論が変わる**(Huberman & Glance 1993 PNAS 90(16):7716 が
Nowak & May 1992 Nature 359:826 の協調維持を非同期更新で消してみせた。
個体ベースモデルでの直接の実証は Caron-Lormier et al. 2008 Ecological Modelling
212(3-4):522-527、走査順の影響の分類は Weimer et al. 2019 JASSS 22(4)5)。
実装様式としては Mesa の `SimultaneousActivation`(compute→advance の 2 相)が同型。

本 module が担うのは 2 つ:
  1. **認知の read フェーズを 1 回の凍結観測に固定する**: その tick の S 判定は
     全エージェントぶんを **同一の state(t) スナップショット**から計算する
     (`agent._fire_obs` = 前 step 末に採った観測。誰かの発火が同 tick の他人の S を
     動かす経路が無い)。
  2. **apply フェーズの順序を agent_id 昇順に正準化する**(`barrier()`)。到着順で
     適用する実装への退行を T1 テストが固定する。

R1 ドクトリン
-------------
- 既定 `cognition.fire.enabled: false` では本 module は **1 度も呼ばれない**
  (`enabled()` が False → scheduler が既存経路のまま)= L1/L2/L3・乱数・LLM 呼数不変。
- ON は **journal 等級・affects_k=true**(LLM 呼の発生点そのものを差し替える)を正直に宣言する。
  新しい不変量は「**認知イベントのスケジュール列が同一なら世界状態が同一**」(P0-4)。
- 乱数は専用 stream `"cog_fire"` のみ。`period_cv_scale: 0` では **1 本も引かない**。
- プロンプトは 1 バイトも変えない: 発火理由(periodic/salience/internal)は L1 にしか出さず、
  LLM へ渡す「きっかけ」は従来どおり `drive.top_reason()` の既存語彙のまま
  (= エージェント側から発火機構の ON/OFF を読み取る経路を作らない)。

後方互換の要(P0(2))
--------------------
「現行の10分固定は『全員の基本周期が10分・他の発火源なし』という特殊ケースとして
表現できること」。本実装では

    cognition.fire.sources=[periodic]
    cognition.fire.period_override_min=10
    cognition.fire.period_cv_scale=0
    cognition.fire.sleep_period_mult=1.0

が厳密にその特殊ケースで、**全エージェントが毎 step 期限を迎える** = 現行の
「毎 step 全員が申請対象」と同じ候補集合になる。周期発火はこれまでどおり
欲求閾値+個人重み抽選+予算のゲートを通る(= 発火判定はバイト一致)。
驚き発火・内部発火だけが**抽選を飛ばす割込み**(対面会話の確定発火と同格)。
"""
from __future__ import annotations

import heapq
import math

from ..observer.schema import Event
from . import age_cog as _age_cog
from . import channels as _channels

SCHEMA = 1

# 発火源(P0(2) の 3 種 + §6-3 の修正で第一級化した social)
PERIODIC = "periodic"
SALIENCE = "salience"
INTERNAL = "internal"
SOCIAL = "social"
SOURCES: tuple[str, ...] = (PERIODIC, SALIENCE, INTERNAL, SOCIAL)

# 文脈カテゴリ(較正テーブル conf/cognition/calib_default.yaml の contexts と同一)
WALKING, TALKING, WORKING, RESTING = "walking", "talking", "working", "resting"

# 名前付きトリガ項 Σ_j w_ij·[trigger_j] の既定値(watch OFF では恒等 0)。
# (0 を足す分岐をここに残しておくのは、S の定義式が設計 §2.3 と 1 対 1 で読めるようにするため)
_TRIGGER_TERM = 0.0

# 感度 g の既定値。`cognition.g_update` OFF では全 usable チャンネルで 1.0 固定
# = 「S はチャンネル間で σ 正規化した偏差の単純和」(第81 の挙動)。
_G_FIXED = 1.0


# --------------------------------------------------------------------------- #
# cfg 正準化(既定 OFF)
# --------------------------------------------------------------------------- #
def build_cfg(raw: dict | None) -> dict:
    """conf の `cognition.fire` ブロックを型強制つきで正準化する(既定 OFF)。"""
    raw = dict(raw or {})
    srcs = raw.get("sources", None)
    if srcs is None:
        srcs = list(SOURCES)
    seq = [str(s) for s in (srcs if isinstance(srcs, (list, tuple)) else [srcs])]
    unknown = [s for s in seq if s not in SOURCES]
    if unknown:
        raise ValueError(
            f"cognition.fire.sources に未知の発火源: {unknown}"
            f"(有効値: {', '.join(SOURCES)})")
    return {
        "enabled": bool(raw.get("enabled", False)),
        # 並びは宣言順を保たず正典順に畳む(config の書き順が世界に漏れないため)
        "sources": tuple(s for s in SOURCES if s in set(seq)),
        # >0 = 全文脈の基本周期をこの分数で上書き(後方互換の特殊ケース用)。0 = 較正テーブル
        "period_override_min": max(0, int(raw.get("period_override_min", 0) or 0)),
        "period_scale": float(raw.get("period_scale", 1.0)),
        # 0 = ばらつきなし(乱数を 1 本も引かない)。1 = 較正テーブルの cv をそのまま使う
        "period_cv_scale": max(0.0, float(raw.get("period_cv_scale", 1.0))),
        "sleep_period_mult": max(1.0, float(raw.get("sleep_period_mult", 6.0))),
        "theta_scale": float(raw.get("theta_scale", 1.0)),
        "max_contrib": max(0, int(raw.get("max_contrib", 3) or 0)),
    }


def enabled(sim) -> bool:
    cfg = getattr(sim, "firecfg", None)
    return bool(cfg and cfg["enabled"])


# --------------------------------------------------------------------------- #
# 認知イベントキュー
# --------------------------------------------------------------------------- #
class CogQueue:
    """(発火時刻 sim_min, agent_id) の**辞書式全順序**キュー。

    - `pending[aid] = {"at": 分, "reason": 発火源, "tok": 世代}` が唯一の真実。
    - ヒープは索引にすぎず、`(at, aid, tok)` を積んで pop 時に世代照合で古い項目を捨てる
      (lazy deletion。標準的な heapq の更新イディオム)。
    - **1 エージェントに生きたイベントは高々 1 個**なので、実効的な順序は
      `(at, aid)` の 2 要素で厳密な全順序になる(挿入順が結果に漏れない)。
    - checkpoint には `pending` だけを載せ、ヒープは load 時に決定論的に再構築する
      (順序は `pending` の内容だけで決まるので再構築で並びが変わりえない)。
    """

    __slots__ = ("pending", "_heap", "_tok", "seeded")

    def __init__(self) -> None:
        self.pending: dict[int, dict] = {}
        self._heap: list[tuple[int, int, int]] = []
        self._tok = 0
        self.seeded = False

    # ---- 更新 ----
    def schedule(self, aid: int, at_min: int, reason: str) -> None:
        """エージェント aid の次の認知イベントを (at_min, reason) に置き換える。"""
        self._tok += 1
        self.pending[int(aid)] = {"at": int(at_min), "reason": str(reason),
                                  "tok": self._tok}
        heapq.heappush(self._heap, (int(at_min), int(aid), self._tok))

    def advance(self, aid: int, at_min: int, reason: str) -> None:
        """**より早い時刻へのみ**繰り上げる(割込み)。既に同時刻以前なら何もしない。"""
        cur = self.pending.get(int(aid))
        if cur is not None and int(cur["at"]) <= int(at_min):
            return
        self.schedule(aid, at_min, reason)

    def drop(self, aid: int) -> None:
        self.pending.pop(int(aid), None)

    # ---- 参照 ----
    def order(self) -> list[tuple[int, int]]:
        """生きているイベントを `(発火時刻, agent_id)` 昇順で列挙する(テスト/検証用)。"""
        return sorted((int(v["at"]), int(k)) for k, v in self.pending.items())

    def pop_due(self, now_min: int) -> list[tuple[int, int, str]]:
        """`at <= now_min` のイベントを **(at, agent_id) 昇順**で取り出す。

        返り値は `[(at_min, agent_id, reason), ...]`。取り出した分は pending から消える
        (呼び手が必ず再スケジュールする)。
        """
        out: list[tuple[int, int, str]] = []
        while self._heap and self._heap[0][0] <= int(now_min):
            at, aid, tok = heapq.heappop(self._heap)
            cur = self.pending.get(aid)
            if cur is None or int(cur["tok"]) != tok:
                continue                       # 差し替え済みの残骸(lazy deletion)
            del self.pending[aid]
            out.append((at, aid, str(cur["reason"])))
        # ヒープは (at, aid, tok) 昇順に pop するので既に (at, aid) 昇順だが、
        # 「順序は pending の内容だけで決まる」ことを実装上も明示する。
        out.sort(key=lambda t: (t[0], t[1]))
        return out

    # ---- checkpoint(中央管理: engine/checkpoint.py が呼ぶ)----
    def state(self) -> dict:
        return {"schema": SCHEMA, "seeded": bool(self.seeded), "tok": int(self._tok),
                "pending": {int(k): dict(v) for k, v in sorted(self.pending.items())}}

    @classmethod
    def restore(cls, blob: dict | None) -> "CogQueue":
        q = cls()
        if not blob:
            return q
        q.seeded = bool(blob.get("seeded", False))
        q._tok = int(blob.get("tok", 0))
        for aid, rec in sorted((int(k), v) for k, v in
                               dict(blob.get("pending") or {}).items()):
            q.pending[aid] = {"at": int(rec["at"]), "reason": str(rec["reason"]),
                              "tok": int(rec["tok"])}
            heapq.heappush(q._heap, (int(rec["at"]), aid, int(rec["tok"])))
        return q


# --------------------------------------------------------------------------- #
# 文脈カテゴリ(較正テーブルの軸)
# --------------------------------------------------------------------------- #
def context_of(agent, obs: tuple | None) -> str:
    """エージェントの現在の文脈カテゴリ(walking / talking / working / resting)。

    決定論・乱数ゼロ。判定材料は物理量(睡眠・移動中・同席・所在)だけで、
    因子名も信念も見ない(R1)。`obs` は前 step 末の観測(遭遇数=同席相手の有無)。
    """
    if getattr(agent, "sleeping", False):
        return RESTING
    if getattr(agent, "route", None):
        return WALKING
    if obs is not None:
        enc = obs[_ENC_IDX]
        if enc is not None and float(enc) > 0.0:
            return TALKING
    node = getattr(agent, "node", "")
    if node and node == getattr(agent, "work_node", ""):
        return WORKING
    if node and node == getattr(agent, "home_node", ""):
        return RESTING
    return WALKING


# 観測タプル内の位置(channels.CHANNELS の並びが唯一の源)
_ENC_IDX = [c.id for c in _channels.CHANNELS].index("ext.encounter")


# --------------------------------------------------------------------------- #
# S 判定(驚き)。**毎 tick コード側のみ・LLM 呼び出しゼロ**(設計 §2.3)
# --------------------------------------------------------------------------- #
def usable_channels(sim) -> tuple[tuple[int, str, float], ...]:
    """`(観測タプル内の位置, channel_id, σ_c)` の列。σ_c 凍結ファイルの usable のみ。

    σ=0(定数チャンネル)・未実装・標本なしは第80の方針どおり**除外**する
    (床を敷くと 1/ε が巨大な利得になり、最下位ビットの揺れが発火を駆動してしまう)。
    """
    cached = getattr(sim, "_fire_usable", None)
    if cached is not None:
        return cached
    from . import calib as _calib
    sigma_map = _calib.sigma_of(getattr(sim, "cognition_sigma", None) or {})
    idx_of = {c.id: i for i, c in enumerate(_channels.CHANNELS)}
    out = tuple((idx_of[cid], cid, float(sig))
                for cid, sig in sorted(sigma_map.items())
                if cid in idx_of and float(sig) > 0.0)
    sim._fire_usable = out
    return out


def terms_of(obs: tuple | None, pred: tuple | None,
             usable: tuple[tuple[int, str, float], ...],
             g: dict | None = None) -> tuple[dict, dict]:
    """`({位置: |o−ô|/σ}, {位置: g·|o−ô|/σ})` を返す(**S の材料の唯一の計算点**)。

    第1要素は **g を掛ける前の生の σ 正規化誤差** = 慣れ ē の入力(設計 §2.5)。
    第2要素は S への寄与そのもの = credit assignment の按分比の元。
    片側でも欠測(None)のチャンネルはどちらにも入れない(0 で埋めない)。
    """
    if obs is None or pred is None or not usable:
        return {}, {}
    errs: dict[int, float] = {}
    parts: dict[int, float] = {}
    for i, _cid, sigma in usable:
        o, p = obs[i], pred[i]
        if o is None or p is None:
            continue
        err = abs(float(o) - float(p)) / sigma
        errs[i] = err
        weight = _G_FIXED if g is None else float(g.get(i, _G_FIXED))
        contrib = weight * err
        if contrib > 0.0:
            parts[i] = contrib
    return errs, parts


def salience_of(obs: tuple | None, pred: tuple | None,
                usable: tuple[tuple[int, str, float], ...],
                max_contrib: int, g: dict | None = None,
                trigger: float = _TRIGGER_TERM) -> tuple[float, list]:
    """S = Σ_c g_ic·|o_c − ô_c|/σ_c + Σ_j w_ij·[trigger_j] と寄与内訳を返す(設計 §2.3)。

    `pred`(= ô)が未確定(まだ 1 度も認知イベントを経ていない)なら第1項は 0。
    `g=None` は「感度 1.0 固定」(= `cognition.g_update` OFF = 第81 の挙動)、
    `trigger=0.0` は「名前付きトリガなし」(= `cognition.watch` OFF)。
    """
    _errs, parts = terms_of(obs, pred, usable, g)
    by_id = {i: cid for i, cid, _s in usable}
    total = sum(parts.values()) + float(trigger)
    ranked = sorted(((v, by_id[i]) for i, v in parts.items()),
                    key=lambda t: (-t[0], t[1]))
    top = [[cid, round(v, 4)] for v, cid in ranked[:max_contrib]] if max_contrib else []
    return total, top


def theta_of(sim, ctx: str, agent=None) -> float:
    """閾値 θ(文脈別・**較正テーブルの仮値** × conf の全体スケール × 個体倍率)。

    個体倍率(設計 §2.6「θ_i もペルソナ由来 + 履歴で動かす」)は `plasticity.py` が持つ。
    `cognition.g_update` OFF・`agent=None` では 1.0 = 第81 と同一の値。
    """
    table = (getattr(sim, "cognition_calib", None) or {}).get("table") or {}
    row = (table.get("salience") or {}).get(ctx) or {}
    base = float(row.get("theta", 0.0)) * float(sim.firecfg["theta_scale"])
    if agent is None:
        return base
    from . import plasticity as _plasticity
    if not _plasticity.enabled(sim):
        return base
    return base * _plasticity.theta_mult(sim, agent)


# --------------------------------------------------------------------------- #
# 周期(較正テーブル consumption。**分**単位 = Δt 非依存)
# --------------------------------------------------------------------------- #
def _period_min(sim, agent, ctx: str, step: int) -> int:
    """次の定期発火までの間隔 [分]。

    - 基本周期は較正テーブル `base_period[ctx].mean_min`(P0(3) の外部化テーブル)。
      `period_override_min > 0` なら全文脈をその値で上書きする(後方互換の特殊ケース)。
    - 睡眠中は伸ばす(P0(2)「ペルソナと状況(睡眠中は伸びる等)で変調される」)。
    - ばらつき: `period_cv_scale == 0` なら**乱数を 1 本も引かない**(決定論の特殊ケース)。
      引くときは専用 stream `"cog_fire"` から対数正規で引く(平均が mean に一致する母数化)。
    """
    cfg = sim.firecfg
    table = (getattr(sim, "cognition_calib", None) or {}).get("table") or {}
    row = (table.get("base_period") or {}).get(ctx) or {}
    if cfg["period_override_min"] > 0:
        mean = float(cfg["period_override_min"])
        cv = 0.0
    else:
        mean = float(row.get("mean_min", 10.0)) * cfg["period_scale"]
        cv = float(row.get("cv", 0.0)) * cfg["period_cv_scale"]
    if getattr(agent, "sleeping", False):
        mean *= cfg["sleep_period_mult"]
    mean *= _age_cog.period_mult(sim, agent)   # AGE-C(既定 OFF=厳密に 1.0=恒等)
    mean = max(1.0, mean)
    if cv <= 0.0:
        return int(round(mean))
    rng = sim.hub.stream("cog_fire", int(agent.id), int(step))
    s = math.sqrt(math.log(1.0 + cv * cv))
    value = float(rng.lognormal(mean=math.log(mean) - 0.5 * s * s, sigma=s))
    return max(1, int(round(value)))


# --------------------------------------------------------------------------- #
# 観測の凍結(ダブルバッファの read 側)
# --------------------------------------------------------------------------- #
def observe_end(sim, step: int, sim_min: int, since_idx: int) -> None:
    """step 末に o_c(t) を採り、各エージェントへ**凍結**して置く(次 tick の S 判定の入力)。

    ここで採るのが肝: 次の step の `_phase_drive` は全員ぶんを**この 1 枚のスナップショット**
    から読むので、同 tick 内で誰かの発火が他人の S を動かす経路が存在しない(§3.3 の
    ダブルバッファ)。`since_idx` は当該 step 開始時の logger 長で、step 内の増分だけを
    見る(第71 行間補間・第80 チャンネルと同じ流儀 = resume 安全)。

    ★正直な限界: したがって step N の S 判定が読む o_c は **step N−1 末の世界**であって
      判定の瞬間の世界ではない(1 tick ぶん古い)。step 途中で採ると「その step 内の処理順が
      結果に混入してバリアの意味が消える」(設計 §4.5 と同じ理由)ので、遅れは意図的な対価。
      logger の増分で数えるチャンネル(受信発話・掲示視認)を丸ごと 1 step ぶん数えられるのも
      この位置だけである。
    """
    if since_idx < 0:
        return
    for row in _channels.observe(sim, step, sim_min, since_idx):
        agent = sim.present_agent(int(row[2]))     # ★レーン乙: 在場述語(脱水済みへ書かない)
        if agent is not None:
            agent._fire_obs = tuple(row[len(_channels.KEY_COLUMNS):])


# --------------------------------------------------------------------------- #
# 発火源 ④ social: 既存機構(会話返答・夜内省・朝計画)の第一級化(§6-3)
# --------------------------------------------------------------------------- #
def _at_first_reservation(agent, attr: str, step: int) -> bool:
    """DPH-B の FIFO 繰り越し中**でない**(= この step が初回予約 step)か。

    `lod.budget.tiers`(DPH-B)は予算の取れなかった計画/内省を翌 step へ繰り越すとき、
    `plan_due_step` / `reflect_due_step` に**最初に予約された step** を控えて
    `plan_step` / `reflect_step` を step+1 へ立て直す(`scheduler._defer_first` と同じ規約)。
    属性が生えていない = 繰り越しの概念が無い(tiers OFF)なので **常に True** を返す
    = 既定ランでは 1 バイトも挙動が変わらない。
    """
    first = int(getattr(agent, attr, -1))
    return first < 0 or first == int(step)


def note_plan_due(sim, step: int) -> None:
    """朝計画の対象者を `_phase_planning` が消費する**前に**控える(記録専用)。

    `_phase_planning` は処理した個体の `plan_step` を -1 に落とすので、`_phase_drive` の
    時点では観測できない。OFF は完全 no-op。

    ★D1-c #4b(意思決定ダッシュボード・案 b)= **初回予約 step だけを拾う**。
      DPH-B(`lod.budget.tiers`)ON では、予算の取れなかった計画は翌 step へ FIFO 繰り越し
      される(`plan_step = step + 1` が立ち直る)。素朴に `plan_step == step` で拾うと
      **繰り越した step 数だけ** `cog_event{via:"plan"}` が出て、そのたびに
      `q.schedule(... + _period_min(...))` で周期発火が先送りされる = 「予算が足りなかった」
      という実験者側の都合が「その個体が何度も考えた」に化ける。計画は 1 回の予約につき
      高々 1 回しか実行されないのだから、認知イベントも 1 回でなければならない。
      tiers OFF では `plan_due_step` が誰にも生えないので従来と完全に同一。
    """
    if not enabled(sim) or SOCIAL not in sim.firecfg["sources"]:
        sim._fire_plan_due = ()
        return
    sim._fire_plan_due = tuple(sorted(
        int(a.id) for a in sim.agents
        if int(getattr(a, "plan_step", -1)) == int(step)
        and _at_first_reservation(a, "plan_due_step", step)))


def _social_via(sim, agent, step: int, plan_due: frozenset) -> str:
    """この tick に既存機構が担う認知イベントの種別("" = 無し)。

    - reply   … 話しかけられた(`_reply_to`)= 抽選なしの返答保証(会話 = 第一級の発火源)
    - reflect … 夜の内省(`reflect_step == step`)
    - plan    … 朝の一日計画(`_phase_planning` が同 step で処理する)
    優先順は「今この場の相手 > 内省 > 計画」。

    ★D1-c #4b: 内省も DPH-B の FIFO 繰り越しの対象(`_reflect_due` が `reflect_step` を
      翌 step へ立て直す)なので、計画とまったく同じ理由で **初回予約 step だけ** を拾う。
      tiers OFF では `reflect_due_step` が生えない = 従来と 1 バイトも変わらない。
    """
    if getattr(agent, "_reply_to", None) is not None:
        return "reply"
    if (int(getattr(agent, "reflect_step", -1)) == int(step)
            and _at_first_reservation(agent, "reflect_due_step", step)):
        return "reflect"
    if int(agent.id) in plan_due:
        return "plan"
    return ""


# --------------------------------------------------------------------------- #
# tick の本体(scheduler._phase_drive からの**単一作用点**)
# --------------------------------------------------------------------------- #
def due_events(sim, step: int, sim_min: int, active) -> dict[int, dict]:
    """この tick に期限の来た認知イベントを (発火時刻, agent_id) 昇順で確定して返す。

    返り値 `{agent_id: {reason, at, s, theta, contrib, interrupt, ctx}}`。
    `interrupt=True`(salience / internal)は **抽選を飛ばす割込み**(対面会話の確定発火と同格)。

    副作用は「キューの更新」「ô の更新」「social イベントの L1 記録」だけで、
    **S の計算は全員ぶんを凍結観測から先に済ませてから**行う(ダブルバッファ)。
    """
    from . import plasticity as _plasticity
    from . import watch as _watch
    cfg = sim.firecfg
    sources = cfg["sources"]
    q: CogQueue = sim.cogq
    usable = usable_channels(sim)
    max_contrib = cfg["max_contrib"]
    plan_due = frozenset(getattr(sim, "_fire_plan_due", ()) or ())
    agents = list(sim.agents)
    watch_on = _watch.enabled(sim)
    plastic_on = _plasticity.enabled(sim)
    # 第82: θ の恒常性は**日境界 1 回**(設計 §2.6「μ の時定数は日オーダー」)。
    # ここに置くのは「S 判定より前」かつ「1 tick に 1 回」を同時に満たす唯一の点だから。
    _plasticity.day_boundary(sim, step, sim_min)

    # ---- 0) 予約の無い個体を「今」に予約する ----
    #  初回は全員がここを通る(= 現行の「毎 step 全員が申請対象」と同じ位相から始まる)。
    #  在場ローテーション(pool)で途中から入ってくる個体もここで拾う(取りこぼすと
    #  その個体は二度と発火しない)。退場した個体は pop 時に本人が居ないので再予約されず
    #  自然に消える。
    for agent in agents:
        if int(agent.id) not in q.pending:
            q.schedule(int(agent.id), int(sim_min), PERIODIC)
    q.seeded = True

    # ---- 1) read フェーズ(凍結観測から全員ぶんの S を先に計算する)----
    #  ô = watch ON なら LLM の監視仕様(未指定チャンネルは persistence へ後退)
    #  g = g_update ON なら履歴が動かした感度(OFF は 1.0 固定)
    #  トリガ項 = watch ON なら Σ_j w_ij·[trigger_j](OFF は恒等 0)
    ctx_of: dict[int, str] = {}
    s_of: dict[int, tuple[float, list]] = {}
    parts_of: dict[int, dict] = {}
    trig_of: dict[int, list] = {}
    for agent in agents:
        aid = int(agent.id)
        obs = getattr(agent, "_fire_obs", None)
        ctx_of[aid] = context_of(agent, obs)
        base_pred = getattr(agent, "_fire_pred", None)
        pred = _watch.expectation(sim, agent, base_pred) if watch_on else base_pred
        g = _plasticity.g_of(sim, agent) if plastic_on else None
        trig_w, trig_names = (_watch.trigger_term(sim, agent, obs) if watch_on
                              else (_TRIGGER_TERM, []))
        errs, parts = terms_of(obs, pred, usable, g)
        by_id = {i: cid for i, cid, _s in usable}
        ranked = sorted(((v, by_id[i]) for i, v in parts.items()),
                        key=lambda t: (-t[0], t[1]))
        top = ([[cid, round(v, 4)] for v, cid in ranked[:max_contrib]]
               if max_contrib else [])
        s_of[aid] = (sum(parts.values()) + float(trig_w), top)
        parts_of[aid] = parts
        trig_of[aid] = trig_names
        if plastic_on:                             # 慣れ ē の更新 + credit 窓の回収
            _plasticity.observe_tick(sim, agent, errs, sim_min)
    # 第87(engaged)がこの tick の S と文脈を読むための控え。**この 2 行は fire 自身の
    # 挙動に一切影響しない**(sim 属性への代入だけ・L1/乱数/プロンプト不変)。fire が OFF
    # なら due_events そのものが呼ばれないので既定ランでは 1 度も実行されない。
    # ★engaged が S を再計算しないための唯一の口: watch(ô)・g_update(感度)が入った
    #   ときも「発火が見た S」と「エピソードが見る S」が原理的に食い違わない。
    sim._fire_s = s_of
    sim._fire_ctx = ctx_of

    # ---- 2) 割込みの登録(驚き / 内部)。**より早い時刻へのみ**繰り上げる ----
    active_ids = {int(a.id) for a in active}
    if SALIENCE in sources:
        for agent in agents:
            aid = int(agent.id)
            if aid not in active_ids:
                continue
            s_val, _ = s_of[aid]
            theta = theta_of(sim, ctx_of[aid], agent)
            if theta > 0.0 and s_val > theta:
                q.advance(aid, int(sim_min), SALIENCE)
    if INTERNAL in sources:
        for agent in agents:
            aid = int(agent.id)
            if aid not in active_ids:
                continue
            # ニーズ閾値の**上向き横断**(水準ではなく事象)。水準で撃つと閾値超えの間ずっと
            # 割込みが立ち続け、周期発火の意味が消える。
            prev = getattr(agent, "_fire_prev_drive", None)
            thr = float(agent.drive_threshold)
            if prev is not None and float(prev) < thr <= float(agent.drive):
                q.advance(aid, int(sim_min), INTERNAL)

    # ---- 3) social(既存機構をイベントとして第一級化。実行は既存経路のまま)----
    if SOCIAL in sources:
        for agent in agents:
            via = _social_via(sim, agent, step, plan_due)
            if not via:
                continue
            aid = int(agent.id)
            sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=aid,
                                 kind="cog_event", x=agent.x, y=agent.y,
                                 payload={"reason": SOCIAL, "via": via,
                                          "ctx": ctx_of[aid]}))
            # 「考えた」ので期待値を更新し、次の定期発火を先送りする
            agent._fire_pred = getattr(agent, "_fire_obs", None)
            if plastic_on:                         # 認知イベント 1 回 = g 更新 1 回
                _plasticity.on_event(sim, agent, parts_of.get(aid, {}), sim_min, SOCIAL)
            q.schedule(aid, int(sim_min) + _period_min(sim, agent, ctx_of[aid], step),
                       PERIODIC)

    # ---- 4) 期限の来たイベントを取り出す(全順序)----
    out: dict[int, dict] = {}
    for at, aid, reason in q.pop_due(int(sim_min)):
        # ★レーン乙 ブロック7: 発火キューの id は日を跨いで残るので、街を出た個体が
        #   永久に pop され続け、_fire_pred / 可塑性の更新が捨てられる実体へ書かれ、
        #   さらに次周期へ再スケジュールされて「幽霊が考え続ける」状態になっていた。
        agent = sim.present_agent(aid)
        if agent is None:
            continue
        ctx = ctx_of.get(aid, WALKING)
        s_val, contrib = s_of.get(aid, (0.0, []))
        out[aid] = {"reason": reason, "at": int(at), "s": s_val,
                    "theta": theta_of(sim, ctx, agent), "contrib": contrib,
                    "interrupt": reason in (SALIENCE, INTERNAL), "ctx": ctx,
                    "active": aid in active_ids, "trig": trig_of.get(aid, [])}
        # 認知イベントを通ったので persistence 予測を更新する。watch ON では、この上に
        # **LLM が出した ô** が重なる(監視仕様の差し替えは推論後に watch.apply が行う)。
        # 監視仕様に**有効期限は無い**(設計 §2.2)ので、上書きされるまで前回仕様が生きる。
        agent._fire_pred = getattr(agent, "_fire_obs", None)
        if plastic_on:                             # 認知イベント 1 回 = g 更新 1 回
            _plasticity.on_event(sim, agent, parts_of.get(aid, {}), sim_min, reason)
        q.schedule(aid, int(sim_min) + _period_min(sim, agent, ctx, step), PERIODIC)

    # ---- 5) 次 tick の横断判定のために現ゲージを控える ----
    for agent in agents:
        agent._fire_prev_drive = float(getattr(agent, "drive", 0.0))
    return out


def log_events(sim, step: int, sim_min: int, due: dict[int, dict],
               granted: set) -> None:
    """スケジュール列そのものを L1 に残す(P0-4「思考頻度の分布自体が観測量」)。

    (発火時刻, agent_id) 昇順で 1 件ずつ。ON のときだけ生える kind なので OFF は無風。
    """
    for at, aid in sorted((v["at"], k) for k, v in due.items()):
        rec = due[aid]
        agent = sim.agent_by_id.get(aid)
        if agent is None:
            continue
        payload = {"reason": rec["reason"], "due": int(at), "ctx": rec["ctx"],
                   "s": round(float(rec["s"]), 4),
                   "theta": round(float(rec["theta"]), 4),
                   "contrib": rec["contrib"],
                   "interrupt": bool(rec["interrupt"]),
                   "active": bool(rec["active"]),
                   "granted": bool(aid in granted)}
        if rec.get("trig"):        # 成立した名前付きトリガ(watch OFF はキー自体が生えない)
            payload["trig"] = list(rec["trig"])
        sim.logger.log(Event(
            step=step, sim_min=sim_min, agent_id=aid, kind="cog_fire",
            x=agent.x, y=agent.y, payload=payload))


# --------------------------------------------------------------------------- #
# 同期バリア + ダブルバッファ(設計 §3.3)
# --------------------------------------------------------------------------- #
def barrier(sim, actions: list) -> list:
    """推論結果の world への適用順を **agent_id 昇順**に正準化する。

    OFF は引数をそのまま返す(**同一オブジェクト** = 恒等 = バイト一致)。

    ON では「到着順(= 推論の完了順・GPU の気まぐれ)」を一度意図的に撹拌してから
    昇順へ畳む経路を用意してある(`sim._fire_arrival_probe` = テスト専用フック)。
    これは T1(完了順序不変性)を**実際に反証可能な形**にするための口で、
    既定 None では何も起きない。

    ★正直な限界: 本 module が保証するのは「**到着順**が世界に漏れないこと」であって、
      「**発行順**を入れ替えても同じ」ではない。発行(= `_decide`)は今も agent_id 昇順の
      逐次で、そこで採番される語彙 item id・キャッシュ初出順は発行順の関数である。
      設計 §3.3 が名指しする危険(「到着順に反映すると完了順序が結果に漏れる」)は
      前者なので本バッチはそれを塞ぐ。発行順の入れ替え耐性は別の課題。
    """
    if not enabled(sim):
        return actions
    probe = getattr(sim, "_fire_arrival_probe", None)
    seq = list(actions)
    if probe is not None:
        seq = list(probe(seq))
    return sorted(seq, key=lambda pair: int(pair[0].id))


# --------------------------------------------------------------------------- #
# 来歴(run_manifest.json 用)
# --------------------------------------------------------------------------- #
def provenance(sim) -> dict | None:
    """発火機構の宣言。OFF(既定)では None = 既存 manifest と同形。"""
    if not enabled(sim):
        return None
    cfg = sim.firecfg
    usable = usable_channels(sim)
    return {
        "schema": SCHEMA,
        "sources": list(cfg["sources"]),
        "period_override_min": cfg["period_override_min"],
        "period_scale": cfg["period_scale"],
        "period_cv_scale": cfg["period_cv_scale"],
        "sleep_period_mult": cfg["sleep_period_mult"],
        "theta_scale": cfg["theta_scale"],
        "n_usable_channels": len(usable),
        "usable_channels": [cid for _i, cid, _s in usable],
        # ★正直な宣言: 第82 の 2 トグルがそれぞれ OFF なら第81 の暫定実装のまま走る
        "expectation_model": ("llm_watch_spec(未指定チャンネルは persistence へ後退)"
                              if _watch_on(sim)
                              else "persistence(前回の認知イベント時点の観測値)"),
        "g_policy": ("history_update(慣れ/感作/引き戻し=設計 §2.5)"
                     if _plastic_on(sim) else "fixed_1.0(更新則は既定 OFF)"),
        "trigger_dsl": ("whitelist_dsl(ops + 記号 + 数値クランプ)"
                        if _watch_on(sim) else "disabled(cognition.watch OFF)"),
        "theta_source": ("calib_table(provisional) × homeostasis(daily)"
                         if _plastic_on(sim) else "calib_table(provisional)"),
    }


def _watch_on(sim) -> bool:
    from . import watch as _watch
    return _watch.enabled(sim)


def _plastic_on(sim) -> bool:
    from . import plasticity as _plasticity
    return _plasticity.enabled(sim)
