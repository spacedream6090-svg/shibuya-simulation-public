"""存在の内生化 POP = 名簿そのものが増減する 3 イベント(転出・転入・出生)。

正典
----
- ``docs/plans/population-endogenization-plan.md`` §1(全て「域内状態への応答」として起こす)・
  §3 の実装スライス POP-1/POP-2/POP-3・§4 OPEN 1(**ユーザー決定 = 案A「L4 定着昇格」**)
- ``docs/research/population-endogenization.md`` §2(現状の空白 = 転入・恒久転出・出生)・
  §3.2(Wolpert 1965 の stress-threshold / Speare 1974)・§3.3(Wedding Ring → SAMP の系譜)・
  §3.4(渋谷区の実数アンカー)・§4.2-b(名簿と実体の二層を守る)・§4.2-d(転入は原理的に閉じない)

何を解く問題か
--------------
ユーザー原理(2026-08-12):「エージェントの存在(生成・消滅)は、エージェント自身の思考や
行動によって決まるべきであり、世界側のプログラムが規定するものであってはならない」。

この物差しで測ると、本リポジトリでは**消滅の片翼(死)と存在の前段(世帯形成・域内転居・
境界出入り)は既に行動由来**なのに、「名簿そのものが増減する」3 イベント —— **転入・恒久
転出・出生** —— だけが不在だった(リサーチ §2.1 の実測)。本 module はその 3 つを、

  **日次の抽選割当・クォータを 1 つも作らず、既存の個体状態の関数として**

実装する。現実のレート(区の転入 23,456 / 転出 21,898 / 出生 1,669 人年)は**駆動に使わない**。
レート表は**検証目標**であり、発火は行動側に置く(H3 遺失物の返還率・PRES の較正と同じ原理)。

3 イベントの行動由来の表現
--------------------------
**① 恒久転出(POP-1)= Wolpert 1965 の閾値型**
  居住ストレス(家賃滞納・立退き・失職・破産・引きこもり・縁の喪失)が日々**蓄積**し、
  **個体固定の閾値**を跨いだ日に「この街を出る」と決まる。閾値は安定ハッシュ由来の個人属性で、
  当日の乱数は 1 粒も引かない。縁(closeness の高い関係・パートナー・同居人)は蓄積を**押し戻す**
  (Speare の residential satisfaction の媒介変数を、単一スカラーではなく構造化状態の和で書く =
  Landale & Guest 1985 の批判に沿う形)。
  物理表現は**死と同型の永続退場**(``loc="outside"`` + ``return_at`` を実質無限)で、
  ``sim.agents`` からの抜き取りは 1 度も行わない(health.``_die`` の docstring の規約)。

**② 転入(POP-2)= 案A「L4 来街者の定着昇格」**(ユーザー決定 2026-08-14)
  ★正直な限界(リサーチ §4.2-d): 域外の決断はシミュレートしていないので、転入は**原理的に
  閉じない**。ILUTE も SVERIGE も in-migration は外生である。本実装は妥協を最小化する:
  **到着レートという世界のノブを持たず**、既にプールに居る非定期来街者(L4)が
  「通ううちに住むことにした」という**本人の来街履歴と域内状態への応答**としてのみ発生させる。
  条件は 3 つとも域内状態: (a) この街で過ごした日数が個体固定の閾値を超えた
  (b) 域内に一定数の関係(closeness)を築いた (c) **空き住戸がある**(占有判定は
  ``freedom_p2.pick_home`` と同じ述語)。任意で (d) 街に空き定員のある職場がある。
  ★**pool の record は 1 バイトも書き換えない**(名簿 = 決定論の礎)。昇格は
  ``sim._pop_state["settled"]`` という**搬送される run 状態**として持ち、入場のたびに
  :func:`apply_on_entry` が冪等に着せ直す(household.pool_bind と同型)。

  ★**空き住戸は誰かが出ていくことでしか生まれない**(本実装の重要な帰結): 名簿の常住者は
    地図の住宅系建物をほぼ埋め切っているので、転入の受け皿は実質**転出の裏面**である。
    渋谷の人口動態が「総量ほぼ静止・中身が高速入替」(リサーチ §5.1)であることと、
    機構の形が一致する —— 世界が転入枠を配るのではなく、去った人の家に次の人が入る。

**③ 出生(POP-3)= 夫婦固定の位相(PRES-A1 と同じ Bresenham 構成)**
  夫婦の源は 2 つあり、等しく扱う: ``form_partners``(Billari の Wedding Ring 系譜 =
  レート表なしの内生的な結婚)が**ラン中に**張る ``partner_id`` と、``household`` の
  ``realistic`` 束ねが配った**続柄の夫/妻**(= 初期条件として与えられた「既にこの街に居る
  夫婦」。リサーチ §4.3 の R-c 契約)。両者が年齢帯にあり、世帯の子の数が上限未満で、
  **個体固定の性向**(安定ハッシュ)を持つ夫婦について、
  ``floor(θ+(d+1)/I) > floor(θ+d/I)`` の日に子が生まれる。
  θ は夫婦固定の位相、I は夫婦固定の出産間隔。**当日の乱数は 1 粒も引かない**
  (:func:`_due` の docstring に分布保存の証明)。
  新生児は**世帯メンバの最小エージェント**である: LLM を 1 度も呼ばず、``sim.agents`` にも
  入らず、**戸籍(本 module の台帳の 1 行)+ 親の ``housemates`` の 1 要素**としてだけ存在する。
  ★これは手抜きではなく設計判断である。プールの L1 に子どもが 1 人も居ない(ペルソナの既知
  ギャップ)以上、深いペルソナ文を持つ子を発明すると**名簿に無い人格を世界が捏造する**ことに
  なる。存在イベントとしての出生(戸籍・世帯サイズ・親の記憶)だけを正直に持ち、成長は
  長期ランの別レーンへ送る。

保存則(人口会計)
------------------
本 module は「誰が名簿に居るか」の全変化を 1 箇所の台帳で数える::

    在場資格 = (名簿の presence 資格 ∪ settled) − gone
    Σ 台帳 = counts["emigrate"] = |gone| / counts["settle"] = |settled| / counts["birth"] = |births|

金は IF-E の境界フローで閉じる: 転出者は現金・口座を**域外へ持ち出す**ので
``economy_sfc.row_out(sim, "emigration", …)`` で RoW へ抜き、本人の残高を 0 にする
(``total_money`` = 街の残高 + RoW 累積 + K5 累積 が一定であることは既存テストが固定する)。
★org 会計 OFF のランでは RoW 部門そのものが無いので**残高は凍結したまま**にする
(第107 の死が「死者の財布は凍結」で通したのと同じ規約 = 勘定の無い所へ金を捨てない)。

R1 ドクトリン
-------------
- 既定 ``population.*.enabled=false`` では :func:`phase` / :func:`apply_on_entry` が即 return し、
  **台帳オブジェクトすら作らない**(``sim._pop_state`` が生えない・L1 に 1 件も出ず・
  agent に属性が 1 つも生えず・プロンプトが 1 バイトも変わらず・乱数 stream を 1 本も引かない)。
- **新しい乱数 stream を 1 本も引かない**。全ての「個体固定の閾値・性向」は
  ``blake2b`` の安定ハッシュ(run.seed 非依存 = 別ランでも同じ人が同じ閾値)から作る。
  household.pool_bind / mobility._stable_uniform と同じ作法。
- ``generate()`` の呼び出しサイトを 1 つも作らない(**LLM 呼数増分ゼロ**)。新生児は
  エージェントではないので発火の母集団を 1 人も増やさない。
- **新しい L1 の kind を 1 つも足さない**。3 イベントはいずれも既存の ``life_event``
  (「ライフイベント(結婚・同居・別れなど)」)の下位 kind として出す
  (``emigrate`` / ``settle`` / ``birth``)。付随する退職は既存の ``job_change``、
  関係の休眠は既存の ``relation_dormant`` をそのまま使う。
- 毎 step の走査はゼロ。走るのは日境界の 1 回だけ。

較正(数字の出どころ = **駆動ではなく検証目標**)
------------------------------------------------
リサーチ §3.4 の bbox 常住 3 万人按分(推計):

    転入 8.4 人/日 ・ 転出 7.8 人/日 ・ 出生 0.59 人/日   (10 日ランでは 84 / 78 / 6 件)

本 module の閾値既定値はこの帯を**狙って置いた設計値**であり、実測が帯に入るかどうかは
ランの事後集計(``summary.json`` の ``population``)で確かめる(較正のズレは閾値 =
**個人属性**へ帰す = Cadyts 流。世界側のレートノブは持たない)。
"""
from __future__ import annotations

import hashlib
import math

from .observer.schema import Event

SCHEMA = 1

#: 台帳の下位 kind(既存 ``life_event`` の payload["kind"])。新しい L1 kind は足さない。
KIND_EMIGRATE = "emigrate"
KIND_SETTLE = "settle"
KIND_BIRTH = "birth"

#: L4(非定期来街)を表す presence 層のキー。``world/presence.py`` の語彙と同一。
LAYER_STOCHASTIC = "stochastic"

#: 転出者の現金・口座を域外へ抜くときの IF-E 境界チャネル名。
ROW_CHANNEL = "emigration"


# =========================================================================== #
# config
# =========================================================================== #
EMIGRATION_DEFAULTS = {
    "enabled": False,
    # ---- Wolpert 閾値(個体固定)----------------------------------------------
    "threshold": 6.0,          # 閾値の基準値(個体ごとに spread で散らす)
    "spread": 0.6,             # 閾値の個人差の幅(0=全員同じ・1=0.5〜1.5倍)
    "decay": 0.85,             # 前日までの蓄積の持ち越し率(ストレスは癒える)
    "seed": 20260814,          # 安定ハッシュ種(run.seed と独立=別ランでも同じ人が同じ閾値)
    # ---- 日々の居住ストレス(全て既存の持続状態。新しい観測量を発明しない)----
    "w_rent": 0.6,             # 家賃の未払いが在る(E5 rent_due>0)
    "w_arrears": 1.0,          # 滞納日数(arrears_full_days で頭打ち)
    "arrears_full_days": 30,
    "w_evicted": 2.0,          # 立退き中(住居を失っている)
    "w_jobless": 0.8,          # career で職を失い求職中(is_laid_off)
    "w_bankrupt": 1.2,         # 破産の制限期間中
    "w_withdrawn": 0.5,        # 慢性の引きこもり(health chronic)
    "w_alone": 0.4,            # 縁が 1 つも無い
    # ---- 縁(蓄積を押し戻す)----------------------------------------------
    "w_tie": 0.8,              # 縁 1.0 ぶんが打ち消すストレス量
    "tie_closeness": 3.0,      # 「縁」と数える closeness の下限
    "tie_k": 5,                # この本数で縁アンカーが 1.0(飽和)
    "w_partner": 0.5,          # パートナーが居る(縁アンカーへの加算)
}

IMMIGRATION_DEFAULTS = {
    "enabled": False,
    "days_min": 4,             # この街で過ごした日数の基準値(個体固定閾値の中心)
    "spread": 0.6,             # 日数閾値の個人差の幅
    "ties_min": 2,             # 域内に築いた関係(closeness>=tie_closeness)の最小本数
    "tie_closeness": 3.0,
    "require_job": True,       # 街に空き定員のある職場が在ることを受け皿の条件にする
    "max_per_day": 0,          # 0=無制限。>0 は**縮退用の安全弁**(通常は使わない)
    "seed": 20260814,
}

BIRTHS_DEFAULTS = {
    "enabled": False,
    # ★間隔と位相(PRES-A1 の習慣カレンダーと**同じ Bresenham/Beatty 構成**)。
    #   夫婦 1 組が子を持つ平均間隔 [日]。IPSS 第16回「夫婦完結出生児数 1.90 人」÷
    #   有配偶の妊よう性期間(概ね 25 年 = 9,125 日)≒ 1 人あたり 4,800 日 を出発点に、
    #   「両親とも 18-45 歳」という本実装の資格窓の狭さ(= 対象夫婦が実際より少ない)を
    #   考慮して 3,000 日を既定に置く(**設計値**。実測が 0.59 人日 の帯に入るかは
    #   summary.json の population で事後照合する)。
    "interval_days": 3000.0,
    "spread": 0.6,             # 間隔の個人差の幅(0=全夫婦同じ)
    "age_min": 18,
    "age_max": 45,
    "max_children": 3,         # 1 世帯の子の上限
    "couple_share": 0.6,       # 子を持つ性向を持つ夫婦の割合(個体固定=当日の抽選ではない)
    "min_days": 0,             # パートナー成立からの最短日数(既定 0 = 置かない。docstring 参照)
    "min_interval_days": 300,  # 同じ夫婦の次子までの最短間隔(日)
    # 世帯の**続柄**(realistic 束ねの夫/妻)も夫婦として数えるか。既定 true。
    # ★これが false だと 10 日ランでは夫婦が原理的にほぼ存在しない(partner_id は
    #   form_partners が closeness 閾値超で張るので、短いランでは成立しない)。
    "household_spouses": True,
    "seed": 20260814,
}

DEFAULTS = {
    "emigration": dict(EMIGRATION_DEFAULTS),
    "immigration": dict(IMMIGRATION_DEFAULTS),
    "births": dict(BIRTHS_DEFAULTS),
}

_INT_KEYS = frozenset({"arrears_full_days", "tie_k", "seed", "days_min", "ties_min",
                       "max_per_day", "age_min", "age_max", "max_children",
                       "min_days", "min_interval_days"})
_BOOL_KEYS = frozenset({"enabled", "require_job", "household_spouses"})


def _block(raw, base: dict) -> dict:
    cfg = dict(base)
    for k, v in dict(raw or {}).items():
        if k not in base:
            continue
        if k in _BOOL_KEYS:
            cfg[k] = bool(v)
        elif k in _INT_KEYS:
            cfg[k] = int(v)
        else:
            cfg[k] = float(v)
    return cfg


def build_cfg(raw) -> dict:
    """conf の ``population`` ブロックを型強制つきで正準化(既定 OFF = 現行挙動と完全同一)。

    dotlist / OmegaConf のどちらでも受ける(household / housing と同型)。

    ★conf の最上位キーを ``pop`` にしてはいけない(実装中に踏んだ罠): OmegaConf の
      ``DictConfig`` は ``dict.pop`` メソッドを持つので、``cfg.pop`` が**メソッドに
      解決される** —— ``cfg.pop.emigration`` は AttributeError になり、
      ``if cfg.pop:`` のような真偽検査は**常に True** になる。だから ``population``。
    """
    from omegaconf import OmegaConf
    if OmegaConf.is_config(raw):
        raw = OmegaConf.to_container(raw, resolve=True)
    raw = dict(raw or {})
    return {
        "emigration": _block(raw.get("emigration"), EMIGRATION_DEFAULTS),
        "immigration": _block(raw.get("immigration"), IMMIGRATION_DEFAULTS),
        "births": _block(raw.get("births"), BIRTHS_DEFAULTS),
    }


def cfg_of(sim) -> dict:
    cfg = getattr(sim, "popcfg", None)
    return cfg if cfg else DEFAULTS


def emigration_on(sim) -> bool:
    return bool(cfg_of(sim)["emigration"]["enabled"])


def immigration_on(sim) -> bool:
    """転入(案A 定着昇格)が有効か。**プールが無いランでは原理的に成立しない**。"""
    return bool(cfg_of(sim)["immigration"]["enabled"]
                and getattr(sim, "_pool", None) is not None)


def births_on(sim) -> bool:
    return bool(cfg_of(sim)["births"]["enabled"])


def enabled(sim) -> bool:
    """3 レーンのいずれかが有効か(既定 OFF = 本 module は 1 バイトも世界に触らない)。"""
    return emigration_on(sim) or immigration_on(sim) or births_on(sim)


# =========================================================================== #
# 台帳(ON 経路でのみ生やす。checkpoint.py が runtime["pop_state"] で中央管理する)
# =========================================================================== #
def _state(sim) -> dict:
    st = getattr(sim, "_pop_state", None)
    if st is None:
        st = {
            "schema": SCHEMA,
            "day": -1,                  # 日境界の進行(mid-day resume の二重発火を防ぐ)
            "gone": {},                 # pid -> 転出の事実(理由・閾値・当日の蓄積)
            "settled": {},              # pid -> 定着の事実(住居・日数・縁)
            "births": [],               # 戸籍 = 新生児の行(親・世帯・日)
            "next_child": 0,            # 次の新生児 id(名簿 id 空間の**末尾**から採る)
            "counts": {"emigrate": 0, "settle": 0, "birth": 0},
            "money_exported": 0.0,      # 転出者が域外へ持ち出した現金 + 口座の累計
            "money_frozen": 0.0,        # ★org 会計 OFF のランで凍結したまま置いた額
            "considered": {"emigrate": 0, "settle": 0, "birth": 0},   # 走査した候補数
        }
        sim._pop_state = st
    return st


def state_of(sim):
    """ON のときだけ台帳を返す(OFF は None = checkpoint も summary もキーを作らない)。"""
    return getattr(sim, "_pop_state", None)


def checkpoint_state(sim):
    """checkpoint へ載せる台帳(OFF は None = 旧 checkpoint と互換)。"""
    return state_of(sim)


def restore_state(sim, st) -> None:
    """checkpoint からの復元(旧 checkpoint / OFF ランでは何もしない = 属性も生えない)。"""
    if st is None:
        return
    sim._pop_state = st


# =========================================================================== #
# 純関数ヘルパ(乱数 stream を 1 本も引かない)
# =========================================================================== #
def _u01(seed: int, *parts) -> float:
    """安定ハッシュ → [0,1) の一様値(run.seed 非依存・プロセス非依存・乱数 stream ゼロ)。"""
    key = ":".join([str(seed)] + [str(p) for p in parts]).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big") / 2.0 ** 64


def _spread(base: float, spread: float, u: float) -> float:
    """個体固定の閾値 = base × (1 ± spread/2)。spread=0 で全員同じ(=個人差なし)。"""
    return max(1e-9, float(base) * (1.0 + float(spread) * (float(u) - 0.5)))


def _ties(agent, thr: float) -> int:
    """域内に築いた「縁」の本数(closeness が閾値以上・休眠は数えない)。

    ★closeness は relations(Wave G2)が交流の符号から積む**k 非依存の観測量**である
      (relations OFF のランでは台帳が空 = 縁 0 本 = 誰も縁で引き止められない)。
    """
    rels = getattr(getattr(agent, "mem", None), "relations", None)
    if not rels:
        return 0
    n = 0
    for rel in rels.values():
        if rel.get("dormant"):
            continue
        if float(rel.get("closeness", 0.0) or 0.0) >= thr:
            n += 1
    return n


def _node_xy(sim, node: str):
    if not node:
        return None
    try:
        return sim.city.node_xy(node)
    except Exception:
        return None


def _dist(a, b) -> float:
    if a is None or b is None:
        return 1.0e12
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _pid_of(agent):
    return getattr(agent, "pool_pid", None)


def _key_of(agent) -> str:
    """台帳のキー。プールのランは pid(str)、非プールのランは agent.id(str)へ後退。"""
    pid = _pid_of(agent)
    return str(pid) if pid is not None else f"#{int(agent.id)}"


def _log(sim, agent, kind: str, payload: dict, step: int, sim_min: int) -> None:
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=int(agent.id),
                         kind=kind, x=agent.x, y=agent.y, payload=payload))


# =========================================================================== #
# 在場資格のオーバレイ(presence 純関数の**外側**で適用する)
# =========================================================================== #
def presence_overlay(sim):
    """``(gone_pids, settled_pids)`` を返す。OFF / 台帳が空なら ``None``。

    ★**presence.py の純関数には手を入れない**。あちらの入力は「暦 + 名簿のペルソナ属性」
      だけであるという契約(k 非依存・trait 非依存・resume 不変)を保ちたいのに対し、
      転出・定着は**ラン内に起きた出来事**であって名簿の属性ではないからである。
      オーバレイは日境界の 1 箇所(``scheduler._phase_pool_rotation``)でだけ適用し、
      台帳は checkpoint が中央管理する(= resume でも同じ集合になる)。
    """
    st = state_of(sim)
    if st is None:
        return None
    gone = st["gone"]
    settled = st["settled"]
    if not gone and not settled:
        return None
    return (frozenset(gone), frozenset(settled))


def apply_presence(sim, pids: set) -> set:
    """当日の在場 pid 集合へ台帳を反映する(定着者を足し、転出者を落とす)。"""
    ov = presence_overlay(sim)
    if ov is None:
        return pids
    gone, settled = ov
    if settled:
        pids = pids | {p for p in settled if p in sim._pool}
    if gone:
        pids = pids - gone
    return pids


# =========================================================================== #
# 入場フック(build_pool_agent から呼ぶ。既定 OFF は 1 行も通らない)
# =========================================================================== #
def apply_on_entry(sim, agent, record: dict | None = None) -> None:
    """入場した個体へ**台帳の事実**を冪等に着せ直す(pool record は 1 バイトも変えない)。

    - 定着済み(settled): ``visitor=False`` + 台帳が控えた住居を据える。名簿の record は
      「非定期来街者」のままなので、この一手が無いと**回転のたびに来街者へ戻る**。
    - 転出済み(gone): 万一入場経路を通っても即座に永続退場の物理状態へ戻す(冪等な安全弁)。
    - L4 判定に使う presence 層(``pop_layer``)を record から控える(**ON のときだけ**属性が
      生える = 既定ランでは agent に 1 つも属性が増えない)。
    """
    if not enabled(sim):
        return
    st = _state(sim)
    key = _key_of(agent)
    if record is not None:
        agent.pop_layer = str(record.get("presence", "") or "")
    got = st["settled"].get(key)
    if got is not None:
        _seat_settler(agent, got)
    if key in st["gone"]:
        agent.pop_emigrated = True
        _leave_world(agent)


def _seat_settler(agent, info: dict) -> None:
    """定着者の着席(冪等・乱数ゼロ)。住居は台帳が控えた 1 戸へ必ず戻る。"""
    agent.visitor = False
    agent.commute = False
    # 「帰る家が街の外にある」という前提で立っていた退出の段取りを畳む(冪等)。
    # ★``return_at`` は触らない: 既に圏外へ出ている個体はそのまま予定どおり帰ってくればよく、
    #   帰った先は(この関数が据える)街の中の自宅である。
    agent.exit_intent = False
    agent.homing = False
    agent.pop_settled = True
    agent.pop_settle_day = int(info.get("day", 0))
    if info.get("building"):
        agent.home_building = str(info["building"])
        agent.home_node = str(info.get("node", "") or "")
        agent.home_floor = int(info.get("floor", 1) or 1)
        agent._home_moved = True       # pool の退避に home を運ばせる印(mobility._set_home と同じ)


def _leave_world(agent) -> None:
    """世界から永久に退く物理状態(**health._die の境界 despawn と同じ表現**)。冪等。

    ★新しい退場機構を作らない: 「範囲外へ出た個体」の既存表現(``loc="outside"`` +
      帰還予定)をそのまま使い、帰還予定を実質無限へ置くだけで永続になる。したがって
      全フェーズの「範囲外は飛ばす」判定がそのまま効き、``sim.agents`` からの抜き取り
      (= 他レーンが持つ反復の前提を壊す操作)を 1 度も行わない。
    ★``dead`` は立てない —— 転出者は**生きている**(死者との区別は観測の一級の関心事)。
    """
    from .health import NEVER_RETURN
    agent.sleeping = False
    agent.route = []
    agent.dest = None
    agent.homing = False
    agent.exit_intent = False
    agent.building = None
    agent.loc = "outside"
    agent.return_at = NEVER_RETURN
    agent.stay_until = 0
    agent.day_plan = []
    agent.plan_step = -1
    # ★DPH-B の FIFO 繰り越し予約の後始末(第117 レーンB3)。理由と不変量は
    #   `health._exit_world` の同じ 1 行のコメントに書いた(この 2 つは意図して同型)。
    #   要点だけ: `plan_due_step` は `plan_step` とセットでしか意味を持たない印なので、
    #   落とすときは一緒に畳む。既定(tiers OFF)では属性が生えないので分岐は常に False。
    if int(getattr(agent, "plan_due_step", -1)) >= 0:
        agent.plan_due_step = -1


# =========================================================================== #
# 日境界フェーズ
# =========================================================================== #
def phase(sim, step: int, sim_min: int) -> None:
    """日境界: 転出 → 転入(定着昇格)→ 出生 を、この順で 1 日 1 回だけ処理する。

    順序の理由: 転出が住戸を空けた**その日のうちに**転入がそこへ入れる(空き住戸という
    域内状態への応答、という設計をオフバイワンで壊さないため)。出生は世帯の状態を読むだけ
    なので最後で良い。全て id 昇順の走査 = 決定論。

    既定 OFF は即 return(台帳も作らず・L1 も 0 件・属性も生えない = バイト一致)。
    """
    if not enabled(sim):
        return
    day = int(sim_min) // 1440
    st = _state(sim)
    if day == int(st["day"]):
        return                                    # mid-day resume でも二重に走らない
    st["day"] = day
    if emigration_on(sim):
        _emigration_day(sim, st, day, step, sim_min)
    if immigration_on(sim):
        _immigration_day(sim, st, day, step, sim_min)
    if births_on(sim):
        _births_day(sim, st, day, step, sim_min)


# --------------------------------------------------------------------------- #
# ① 恒久転出(POP-1)= Wolpert 1965 の stress-threshold
# --------------------------------------------------------------------------- #
def _stress_terms(agent, cfg: dict, step: int) -> list:
    """当日の居住ストレスの内訳 ``[(理由, 量), …]``(全て**既存の持続状態**の純関数)。

    ★ここに「不満」「効力感」といった構成概念(factors 層の内部変数)は 1 つも入れない
      —— 本 module は society 直下(CHECKED_DIRS 外)だが、no-fingerprint 契約の精神
      (機構が内面の構成概念を発火判断に食わせない)を守る。読むのは金・住居・職・身体・
      関係台帳という**観測可能な量**だけである。
    """
    out = []
    if float(getattr(agent, "rent_due", 0.0) or 0.0) > 0.0:
        out.append(("rent", float(cfg["w_rent"])))
    arrears = int(getattr(agent, "arrears_days", 0) or 0)
    if arrears > 0:
        full = max(1, int(cfg["arrears_full_days"]))
        out.append(("arrears", float(cfg["w_arrears"]) * min(1.0, arrears / full)))
    if bool(getattr(agent, "evicted", False)):
        out.append(("evicted", float(cfg["w_evicted"])))
    if bool(getattr(agent, "_laid_off", False)):
        out.append(("jobless", float(cfg["w_jobless"])))
    if int(getattr(agent, "bankrupt_until", 0) or 0) > int(step):
        out.append(("bankrupt", float(cfg["w_bankrupt"])))
    if bool(getattr(agent, "withdrawn", False)):
        out.append(("withdrawn", float(cfg["w_withdrawn"])))
    return out


def _anchor(agent, cfg: dict) -> float:
    """縁のアンカー(0..1+)。蓄積を押し戻す量 = ``w_tie × anchor``。"""
    ties = _ties(agent, float(cfg["tie_closeness"]))
    a = min(1.0, ties / max(1, int(cfg["tie_k"])))
    if getattr(agent, "partner_id", None) is not None:
        a += float(cfg["w_partner"])
    return a


def _emigration_day(sim, st: dict, day: int, step: int, sim_min: int) -> None:
    cfg = cfg_of(sim)["emigration"]
    seed = int(cfg["seed"])
    decay = float(cfg["decay"])
    w_tie = float(cfg["w_tie"])
    w_alone = float(cfg["w_alone"])
    leaving = []
    for agent in sim.agents:                       # id 昇順 = 決定論
        if agent.visitor or getattr(agent, "pop_emigrated", False):
            continue
        if bool(getattr(agent, "dead", False)):
            continue
        st["considered"]["emigrate"] += 1
        terms = _stress_terms(agent, cfg, step)
        anchor = _anchor(agent, cfg)
        if anchor <= 0.0:
            terms.append(("alone", w_alone))
        today = sum(v for _n, v in terms) - w_tie * anchor
        prev = float(getattr(agent, "pop_stress", 0.0) or 0.0)
        stress = max(0.0, prev * decay + today)
        agent.pop_stress = stress
        thr = _spread(cfg["threshold"], cfg["spread"],
                      _u01(seed, "emig", _key_of(agent)))
        if stress < thr:
            continue
        leaving.append((agent, stress, thr, [n for n, _v in terms]))
    for agent, stress, thr, causes in leaving:     # 走査中に sim.agents を触らない
        _emigrate(sim, st, agent, stress, thr, causes, day, step, sim_min)


def _emigrate(sim, st: dict, agent, stress: float, thr: float, causes: list,
              day: int, step: int, sim_min: int) -> None:
    """1 体を「域外へ永久退場」させ、後始末(住戸・職・世帯・関係・金)を閉じる。"""
    key = _key_of(agent)
    home_from = str(getattr(agent, "home_building", "") or "")
    # (a) 職を辞める(既存の career 機構をそのまま使う = 新しい退職経路を作らない)
    from . import organizations as _orgs
    from_org = None
    if _orgs.is_employee(agent):
        from_org = _orgs.lay_off(agent)
        agent._laid_off = False                    # 求職中ではない(街を出るのだから)
        _log(sim, agent, "job_change",
             {"from_org": from_org, "to_org": None, "cause": KIND_EMIGRATE},
             step, sim_min)
    # (b) 世帯から抜ける(残るメンバの housemates からも外れる = unbond 整合)
    from . import mobility as _mobility
    _mobility.leave_household(sim, agent)
    # (c) パートナーの片側清算(相手が在場なら相互に外す)
    pid = getattr(agent, "partner_id", None)
    if pid is not None:
        other = sim.present_agent(int(pid))
        if other is not None and getattr(other, "partner_id", None) == agent.id:
            other.partner_id = None
            other.remember(f"{agent.name}がこの街を離れた")
        agent.partner_id = None
    # (d) 相手側の関係を休眠にする(既存 relation_dormant を流用 = 新イベント種ゼロ)
    from . import dunbar as _dunbar
    for oid in sorted(int(k) for k in
                      (getattr(getattr(agent, "mem", None), "relations", None) or {})):
        other = sim.present_agent(oid)
        if other is not None:
            _dunbar.mark_dormant(sim, other, int(agent.id), step, sim_min)
    # (e) 住戸を解放する(空き住戸の占有判定は home_building の有無で決まる)
    agent.home_building = ""
    agent.home_node = ""
    agent.home_floor = 1
    agent._home_moved = True                       # 退避に「空にした住居」を運ばせる印
    # (f) 現金・口座を域外へ持ち出す(IF-E の境界フロー)。org 会計 OFF なら**凍結**する。
    from . import economy_sfc as _sfc
    carried = float(getattr(agent, "money", 0.0) or 0.0) \
        + float(getattr(agent, "account", 0.0) or 0.0)
    if _sfc.enabled(sim):
        if carried:
            _sfc.row_out(sim, ROW_CHANNEL, carried)
        agent.money = 0.0
        if hasattr(agent, "account"):
            agent.account = 0.0
        st["money_exported"] += carried
    else:
        st["money_frozen"] += carried              # 第107 の死と同じ「凍結」規約
    # (g) 永続退場 + 台帳 + L1(既存 life_event の下位 kind。新 kind ゼロ)
    agent.pop_emigrated = True
    agent.pop_emigrate_day = int(day)
    _leave_world(agent)
    st["gone"][key] = {"day": int(day), "agent_id": int(agent.id),
                       "stress": round(float(stress), 4),
                       "threshold": round(float(thr), 4),
                       "causes": list(causes), "home": home_from,
                       "carried": round(carried, 2)}
    st["counts"]["emigrate"] += 1
    _log(sim, agent, "life_event",
         {"kind": KIND_EMIGRATE, "stress": round(float(stress), 3),
          "threshold": round(float(thr), 3), "causes": list(causes),
          "from": home_from, "carried": round(carried, 1),
          "org": from_org, "day": int(day)}, step, sim_min)
    agent.remember("この街での暮らしをたたみ、よそへ移ることにした")


# --------------------------------------------------------------------------- #
# ② 転入 = 案A「L4 定着昇格」(POP-2)
# --------------------------------------------------------------------------- #
def _occupied_buildings(sim) -> set:
    """いま誰かの住居になっている建物 id の集合(``freedom_p2.pick_home`` と同じ述語)。

    ★``sim.agents``(在場)ではなく ``sim.agent_by_id``(**これまで実体化した全個体**)を
      見る: 家はプール回転で街を出ても空き家にはならない(第112 の修正と同じ理由)。
    """
    roster = getattr(sim, "agent_by_id", None)
    everyone = roster.values() if roster else sim.agents
    return {getattr(a, "home_building", "") for a in everyone
            if getattr(a, "home_building", "")}


def _vacancy_pool(sim) -> list:
    """空き住戸(誰の home でもない住宅系建物)の一覧。日境界に 1 回だけ組む。"""
    occupied = _occupied_buildings(sim)
    res = getattr(sim.city, "residential_buildings", None) or []
    return [b for b in res if b["id"] not in occupied]


def _has_job_opening(sim) -> bool:
    """街に「空き定員のある職場」が 1 つでもあるか(受け皿の有無だけを見る)。

    ★正直な限界(計画からの意図的な後退): 本 module は**雇用契約を結ばない**。
      勤務窓・賃金プランの再構成は career / WAGE レーンの持ち物で、ここで発明すると
      同じ意味論を 2 箇所が書くことになる。転入の因果としては「求人という域内の受け皿が
      現に空いている」という**ゲート**までを表現し、就職そのものは既存機構へ委ねる。
    """
    book = getattr(sim, "orgs", None)
    if not book:
        return False
    counts: dict = {}
    for a in sim.agents:
        oid = getattr(a, "org_id", None)
        if oid:
            counts[str(oid)] = counts.get(str(oid), 0) + 1
    for oid, org in book.items():
        if org.get("school_type"):
            continue
        cap = int((org.get("size") or {}).get("employees", 0) or 0)
        if cap and counts.get(str(oid), 0) < cap:
            return True
    return False


def _immigration_day(sim, st: dict, day: int, step: int, sim_min: int) -> None:
    cfg = cfg_of(sim)["immigration"]
    seed = int(cfg["seed"])
    tie_thr = float(cfg["tie_closeness"])
    ties_min = int(cfg["ties_min"])
    max_per_day = int(cfg["max_per_day"])
    # 来街履歴の更新(この街で過ごした日を 1 日 1 回だけ数える = 本人の行動の記録)
    cands = []
    for agent in sim.agents:                       # id 昇順 = 決定論
        if not getattr(agent, "visitor", False) or getattr(agent, "pop_settled", False):
            continue
        if getattr(agent, "pop_layer", "") != LAYER_STOCHASTIC:
            continue
        days = int(getattr(agent, "pop_days", 0) or 0) + 1
        agent.pop_days = days
        st["considered"]["settle"] += 1
        need = _spread(cfg["days_min"], cfg["spread"],
                       _u01(seed, "settle", _key_of(agent)))
        if days < need:
            continue
        ties = _ties(agent, tie_thr)
        if ties < ties_min:
            continue
        cands.append((agent, days, ties))
    if not cands:
        return
    if cfg["require_job"] and not _has_job_opening(sim):
        return                                     # 受け皿(求人)が無い = 誰も定着しない
    vac = _vacancy_pool(sim)
    if not vac:
        return                                     # 空き住戸が無い = 物理的に住めない
    taken: set = set()
    for agent, days, ties in cands:
        if max_per_day > 0 and len(taken) >= max_per_day:
            break
        bld = _pick_vacant(sim, agent, vac, taken)
        if bld is None:
            break                                  # 空き住戸を使い切った
        taken.add(bld["id"])
        _settle(sim, st, agent, bld, days, ties, day, step, sim_min)


def _pick_vacant(sim, agent, vac: list, taken: set):
    """空き住戸を 1 戸選ぶ(現在地に近い順・同点は安定ハッシュ)。**乱数ゼロ**。

    ``freedom_p2.pick_home`` の「近い上位から 1 つ」と同じ意味論を、当日の走査で確保済みの
    住戸(``taken``)を除きつつ乱数なしで書いたもの(選抜順は id 昇順で決まるので、
    同じ日には同じ人が同じ家に着く)。
    """
    axy = (agent.x, agent.y)
    best = None
    best_key = None
    for b in vac:
        if b["id"] in taken:
            continue
        d = _dist(b.get("centroid"), axy)
        key = (round(d, 3), _u01(0, "vac", int(agent.id), b["id"]), b["id"])
        if best_key is None or key < best_key:
            best, best_key = b, key
    return best


def _settle(sim, st: dict, agent, bld: dict, days: int, ties: int,
            day: int, step: int, sim_min: int) -> None:
    """L4 来街者を「この街の住民」へ昇格させる(名簿は 1 バイトも書き換えない)。"""
    key = _key_of(agent)
    levels = max(1, int(bld.get("levels", 1) or 1))
    info = {"day": int(day), "agent_id": int(agent.id),
            "building": str(bld["id"]), "node": str(bld.get("entrance", "") or ""),
            "floor": 1 + int(_u01(0, "flr", key) * levels) % levels,
            "days": int(days), "ties": int(ties)}
    st["settled"][key] = info
    st["counts"]["settle"] += 1
    _seat_settler(agent, info)
    _log(sim, agent, "life_event",
         {"kind": KIND_SETTLE, "days": int(days), "ties": int(ties),
          "to": info["building"], "day": int(day)}, step, sim_min)
    agent.remember("通ううちに愛着がわいて、この街に住むことにした")


# --------------------------------------------------------------------------- #
# ③ 出生(POP-3)= 世帯状態からの決定論
# --------------------------------------------------------------------------- #
def _child_base(sim) -> int:
    """新生児 id の起点 = 名簿 id 空間の**末尾の外**(既存 id と絶対に衝突しない)。

    プールのランは ``len(pool)``(``PoolStore.id_of`` が配る密な整数 0..N-1 の直後)。
    プール無しのランは実体化済みの最大 id + 1。どちらも int32 に収まる。
    """
    pool = getattr(sim, "_pool", None)
    if pool is not None:
        return len(pool)
    roster = getattr(sim, "agent_by_id", None) or {}
    return (max(int(k) for k in roster) + 1) if roster else len(sim.agents)


def _children_of(st: dict, lo: int, hi: int) -> list:
    return [r for r in st["births"]
            if int(r["parents"][0]) == int(lo) and int(r["parents"][1]) == int(hi)]


def _due(theta: float, interval: float, day: int) -> bool:
    """夫婦固定の位相 θ と間隔 I から「今日が出産の日か」を決める(**Bresenham / Beatty 列**)。

        due(d) ⟺ floor(θ + (d+1)/I) > floor(θ + d/I)

    ★これは PRES-A1 の習慣カレンダー(``world/presence.py::_habit_present``)と**同一の構成**で、
      同じ分布保存の証明が効く: θ ~ U[0,1) なので frac(θ + d/I) も U[0,1) 上一様 → 区間
      (d/I, (d+1)/I] に整数が入る確率は区間長 = **1/I**。つまり 1 日あたりの出産確率が
      Bernoulli(1/I) と厳密に一致し(丸め誤差なし)、D 日の出産回数も
      floor(θ+D/I) − floor(θ) ∈ {floor(D/I), ceil(D/I)} で長期レートが厳密保存される。
    ★**当日の乱数を 1 粒も引かない**のがこの構成の要点である。「今日産むか」を世界のコインが
      決めるのではなく、**その夫婦がいつ産む位相にあるか**という個人属性だけで決まる
      (ユーザー原理「世界のアルゴリズムがエージェント量を決めない」の出生版)。
    ★10 日ランでも出生が観測されるのはこの構成のおかげである(θ が夫婦ごとに一様に散って
      いるので、どの 10 日を切っても 10/I の割合の夫婦の位相がその窓に入る)。これは
      「この 10 日で出会って産んだ」ではなく「**この街に既に居る夫婦のもとに子が生まれた**」
      という読みであり、初期条件が t=0 の状態を与える(リサーチ §4.3 の R-c 契約)ことと
      整合する。だから ``min_days``(パートナー成立からの最短日数)の既定は 0 にしてある。
    """
    if interval <= 0.0:
        return False
    lo = float(day) / float(interval)
    hi = float(day + 1) / float(interval)
    return math.floor(theta + hi) > math.floor(theta + lo)


def _couple_of(sim, agent, spouses: bool):
    """agent の「夫婦の相手」(在場していなければ None)。決定論・乱数ゼロ。

    2 つの源を等しく扱う:
      ① ``partner_id`` … ``form_partners`` が**ラン中に**張る恋愛関係(Wedding Ring 系譜)
      ② 世帯の**続柄**(household.spouse_of)… ``realistic`` 束ねが配った夫/妻 =
         **初期条件として与えられた「既にこの街に居る夫婦」**(リサーチ §4.3 の R-c 契約)
    ★② を入れる理由: ① は 10 日ランではほとんど成立しない(closeness が閾値に届かない)。
      ② を外すと「出生の前段 = 世帯形成」という設計が**起動時に既に成立している世帯**を
      1 組も拾えなくなり、名簿が持っている家族像を機構が無視することになる。
    """
    pid = getattr(agent, "partner_id", None)
    if pid is not None:
        other = sim.present_agent(int(pid))
        if other is not None:
            return other
    if not spouses:
        return None
    from . import household as _hh
    return _hh.spouse_of(sim, agent)


def _births_day(sim, st: dict, day: int, step: int, sim_min: int) -> None:
    cfg = cfg_of(sim)["births"]
    seed = int(cfg["seed"])
    age_lo, age_hi = int(cfg["age_min"]), int(cfg["age_max"])
    min_days = int(cfg["min_days"])
    spouses = bool(cfg["household_spouses"])
    seen: set = set()
    born = []
    for a in sim.agents:                           # id 昇順 = 決定論
        if a.visitor or getattr(a, "pop_emigrated", False) or getattr(a, "dead", False):
            continue
        b = _couple_of(sim, a, spouses)
        if b is None or b.visitor or getattr(b, "pop_emigrated", False) \
                or getattr(b, "dead", False):
            continue
        lo, hi = (a, b) if a.id < b.id else (b, a)
        pair = (int(lo.id), int(hi.id))
        if pair in seen:
            continue
        seen.add(pair)
        st["considered"]["birth"] += 1
        # (i) パートナー継続日数(初観測日を起点にする = cohabit_day と同じ流儀。
        #     既定 -1 = 未観測 = pool の退避表 ``_MISC_FIELDS`` の既定値と厳密に一致)
        since = int(getattr(lo, "pop_pair_since", -1))   # ★`or -1` は day0 を潰すので書かない
        if since < 0:
            lo.pop_pair_since = hi.pop_pair_since = int(day)
            since = int(day)
        together = int(day) - since
        if together < min_days:
            continue
        # (ii) 年齢帯(名簿の属性 = k 非依存)
        if not (age_lo <= int(getattr(lo, "age", 0)) <= age_hi
                and age_lo <= int(getattr(hi, "age", 0)) <= age_hi):
            continue
        # (iii) 子を持つ性向(**夫婦固定** = 当日の抽選ではない)
        if _u01(seed, "want", *pair) >= float(cfg["couple_share"]):
            continue
        kids = _children_of(st, *pair)
        if len(kids) >= int(cfg["max_children"]):
            continue
        if kids:                                   # 同じ夫婦の次子は最短間隔を置く
            last = max(int(r["day"]) for r in kids)
            if int(day) - last < int(cfg["min_interval_days"]):
                continue
        # (iv) 位相(Bresenham)。当日の乱数はゼロ。
        theta = _u01(seed, "phase", *pair)
        interval = _spread(cfg["interval_days"], cfg["spread"],
                           _u01(seed, "span", *pair))
        if not _due(theta, interval, int(day)):
            continue
        born.append((lo, hi, together))
    for lo, hi, together in born:
        _birth(sim, st, lo, hi, together, day, step, sim_min)


def _birth(sim, st: dict, lo, hi, together: int, day: int, step: int,
           sim_min: int) -> None:
    """新生児を 1 人「戸籍に載せる」= 台帳の 1 行 + 親の世帯メンバに 1 人加える。

    ★新生児は ``sim.agents`` に入らない(= LLM 呼数・発火母集団・在場数を 1 も変えない)。
      在場は「世帯同伴」(親が街に居る日に居る)という意味であり、物理的な実体は持たない。
    """
    if not st["next_child"]:
        st["next_child"] = _child_base(sim)
    child = int(st["next_child"])
    st["next_child"] = child + 1
    hh_id = getattr(lo, "household_id", None) or getattr(hi, "household_id", None)
    if not hh_id:
        hh_id = f"pb{int(lo.id)}_{int(hi.id)}"
        for m in (lo, hi):
            m.household_id = hh_id
            m.household_kind = "family"
    for m in (lo, hi):                             # 世帯サイズ +1(同居人 id に子を足す)
        mates = set(int(x) for x in (getattr(m, "housemates", None) or []))
        other = hi if m is lo else lo
        mates.add(int(other.id))
        mates.add(child)
        m.housemates = sorted(mates)
    st["births"].append({"child_id": child, "day": int(day),
                         "parents": [int(lo.id), int(hi.id)],
                         "household_id": str(hh_id),
                         "home": str(getattr(lo, "home_building", "") or "")})
    st["counts"]["birth"] += 1
    _log(sim, lo, "life_event",
         {"kind": KIND_BIRTH, "child": child, "other": int(hi.id),
          "household": str(hh_id), "together_days": int(together),
          "day": int(day)}, step, sim_min)
    lo.remember("家族が一人増えた")
    hi.remember("家族が一人増えた")


# =========================================================================== #
# 観測(summary.json の "population")
# =========================================================================== #
def provenance(sim) -> dict | None:
    """人口会計の観測タリー(既定 OFF は None = summary にキー自体が出ない)。

    ★検証目標(**駆動ではない**)= リサーチ §3.4 の bbox 常住按分:
      転入 8.4 / 転出 7.8 / 出生 0.59 人日。ここに出す実測がその帯に入るかどうかで
      較正の当否を事後に判定する(ズレは閾値 = 個人属性へ帰す = Cadyts 流)。
    """
    st = state_of(sim)
    if st is None:
        return None
    cfg = cfg_of(sim)
    days = max(1, int(st["day"]) + 1)
    counts = dict(st["counts"])
    deaths = int((getattr(sim, "_h1_state", None) or {}).get("deaths", 0))
    out = {
        "schema": SCHEMA,
        "days": days,
        "enabled": {"emigration": bool(cfg["emigration"]["enabled"]),
                    "immigration": bool(cfg["immigration"]["enabled"]),
                    "births": bool(cfg["births"]["enabled"])},
        "counts": counts,
        "per_day": {k: round(v / days, 4) for k, v in counts.items()},
        "considered": dict(st["considered"]),
        # ★人口会計(Σ 整合): 台帳の行数とイベント件数が必ず一致する
        "ledger": {"gone": len(st["gone"]), "settled": len(st["settled"]),
                   "births": len(st["births"])},
        "deaths": deaths,
        "net": counts["settle"] + counts["birth"] - counts["emigrate"] - deaths,
        "money_exported": round(float(st["money_exported"]), 2),
        "money_frozen": round(float(st["money_frozen"]), 2),
        # 検証目標(実数アンカー。**駆動レートではない**)
        "target_per_day": {"settle": 8.4, "emigrate": 7.8, "birth": 0.59},
    }
    return out
