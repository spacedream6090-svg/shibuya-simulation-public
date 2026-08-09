"""計画駆動の圏外滞在(actor model P4・``planning.day_plan.boundary``・**既定 OFF**)。

正典
----
- docs/plans/actor-model-migration-plan.md **§4**(P4 境界。「朝の計画のセグメントに場所を
  与え、**場所が圏外のセグメントが despawn / respawn の時刻表そのもの**である。境界の
  出入りに別機構を作らない」)
- docs/research/shibuya-inflow.md §1 表B(区民のうち **区内へ通勤・通学するのは 6.9%**
  = 居住者の勤務先は圧倒的に区外。逆向きの流入 74% は既存の `commute` 個体が担う)

何を解く問題か
--------------
本シムの居住者は全員が地図の中で一日を終える。現実の都心区の住民の大半は**朝に区外へ出て
夕方に帰る**が、その姿は今まで表現されていなかった(街の外へ出る経路は `routine` の
確率的な気まぐれ退出 `exit_prob` と、来街者の帰宅 `homing` の 2 つだけで、どちらも
**予定に基づかない**)。本 module は朝の計画(day_plan v1)のブロックに
「この時間帯は圏外に居る」という**明示欄**を持たせ、その欄がそのまま退出/帰還の
時刻表になるようにする。

★**新しい境界機構は 1 つも作らない。**
  退出は既存の `_try_exit`(`agent.loc="outside"` + `exit_area`)、帰還は既存の
  `_phase_wake_and_returns`(`enter_area`)をそのまま使う。本 module が足すのは
    (a) 誰が圏外通勤者かの**決定論的な指名**、
    (b) 計画ブロックへの `boundary` 欄の**刻印**、
    (c) 退出時の `return_at` を**乱数ではなくブロックの終了時刻から**決める差し替え、
    (d) 帰還時の**圧縮記憶 1 行**、
  の 4 点だけである。

二系統の台帳を混ぜない
----------------------
| 台帳 | 誰 | 何で決まるか | 本 module |
|---|---|---|---|
| **計画駆動**(本 module) | 域内**居住者**が圏外へ通勤 | 朝の計画ブロック | ★担当 |
| 統計駆動(既存) | 域外**住民**が域内へ流入(`commute`/`arrival_min`) | 名簿の二峰分布 | **触らない** |

両者は L1 の payload で機械的に分離できる: 計画駆動の `exit_area`/`enter_area` だけが
``payload["boundary"] == "plan"`` を持つ(統計駆動側の payload は 1 バイトも変えない)。
= 境界フラックスの集計は「kind と boundary 欄の 2 条件」の**ただのクエリ**になる。

圏外では何も起きない(コスト 0)ことの根拠
-------------------------------------------
`loc == "outside"` の個体は **既存のエンジンが構造的に素通りする**。新しい抑止コードは
1 行も要らない。実際の作用点(HEAD 実測):
  - `engine/scheduler.py:5202` `active = [a for a in sim.agents if a.loc != "outside" ...]`
    → `_decide` / `_apply` を通らない = 行動も L1 も生まれない
  - `engine/scheduler.py:2346` `_phase_drive` の母集団が同条件 → 欲求申請・発火抽選に載らない
    = **LLM 呼(generate)が 1 本も生まれない**
  - `engine/scheduler.py:902 / 923` `_phase_planning` / `_phase_planning_batched`
    → 圏外の個体は朝計画を発行しない
  - `engine/scheduler.py:1088` `_phase_move` → 移動しない(占有にも数えない)
  - `cognition/channels.py:221` / `world/perception.py:47` → 知覚・チャネルの対象外
  - `conversation.py:288` / `chance.py:93` / `diversity.py:274` / `health.py:84` 他
    → 会話相手・偶発事・犯罪・疲労のいずれの母集団にも入らない
本 module 自身も圏外の個体に対して**乱数を 1 本も引かない**(退出時の `return_at` は
ブロックの終了時刻からの整数計算・帰還時の記憶は定型文の純関数)。

正直な近似(直さない。設計上そう決めたもの)
--------------------------------------------
1. **圏外は摩擦ゼロ**。街の外には空間も混雑も他者も無い。圏外に居る間の出来事は
   一切生成せず、「計画どおりに行動した」とみなす(記憶に残るのは定型 1 行だけ)。
   → 圏外での消費・賃金・出会い・疲労は**存在しない**。designated な個体は域内の
   賃金経路にも載らない(そもそも `work_node` を持たないので本 module の前後で不変)。
2. **フラックスは固定**。圏外通勤者の集合は (run.seed, agent.id) の純関数で run 中不変。
   域内の混雑・災害・運休が「今日は出社をやめる」といった**内側から外向きのフィードバック**
   は掛からない(帰還の遅れだけは既存の envfeedback / 終電判定が効く=非対称)。
3. **退出は「縁に着いた時刻」**。ブロックの `exit_min` は**予定**であって、実際の退出は
   縁(gateway)まで歩いた後に起きる。移動時間を無料にしない代わりに、予定と実測が
   ずれる(ずれ幅は L1 の step と `exit_min` の差として観測できる)。
4. `places` の既定は ``["work"]`` のみ。`education` を足せる形にはしてあるが、既定では
   足さない — 学業ブロックは地図内の学校 POI に**実際に解決できる**(圏外にする根拠が
   個体側に無い)ため。指名対象は「地図内に勤務先が解決しない居住者」に限る。
"""
from __future__ import annotations

import hashlib

# 圧縮記憶の場所ラベル(**定型**。自由文も数字も実験条件も混ぜない)。
PLACE_JA: dict[str, str] = {"work": "仕事", "education": "学業"}
# 圧縮記憶の固定句(§4「計画通りに実行、特筆事項なし」)。
MEMO_TAIL = "計画通りに実行、特筆事項なし。"

# --------------------------------------------------------------------------- #
# 設定(既定 OFF)
# --------------------------------------------------------------------------- #
DEFAULTS = {
    "enabled": False,
    # 指名率: **eligible(地図内に勤務先が解決しない居住者)に対する**割合。
    #   0.7 は shibuya-inflow §1 表B(区民の 6.9% しか区内で就業・就学しない)が示す
    #   向きに合わせた**未較正の暫定値**。較正は昼夜間人口比率との突き合わせが要る。
    "outside_work_fraction": 0.7,
    "places": ("work",),         # 圏外扱いにする場所カテゴリ(上の近似 4 を参照)
    "gateway": "station",        # "station"=駅ノード / それ以外=city.gateways から決定論選択
    "start_min": 480,            # 圏外勤務の開始(分 of day)。骨格計画のときだけ使う
    "start_spread_min": 150,     # 開始のばらつき幅(persona._pick_workplace と同じ 8:00〜10:30)
    "start_grid_min": 10,        # ばらつきの刻み[分](= 正準 Δt)
    "hours_min": 480,            # 圏外に居る長さ[分](骨格計画のときだけ使う)
    "min_outside_steps": 1,      # 退出が遅れて帰還時刻を過ぎた場合の最小滞在 step
}
_BOOL_KEYS = ("enabled",)
_INT_KEYS = ("start_min", "start_spread_min", "start_grid_min", "hours_min",
             "min_outside_steps")
_FLOAT_KEYS = ("outside_work_fraction",)


def _to_plain(raw):
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(raw):
            return OmegaConf.to_container(raw, resolve=True)
    except Exception:                              # noqa: BLE001 (omegaconf 不在でも動く)
        pass
    return raw


def build_cfg(raw) -> dict:
    """conf の ``planning.day_plan.boundary`` を正準化する(day_plan.build_cfg と同型)。"""
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
        elif k == "places":
            cfg["places"] = tuple(str(s) for s in (_to_plain(v) or ()))
        elif k == "gateway":
            cfg["gateway"] = str(v)
    cfg["outside_work_fraction"] = min(1.0, max(0.0, cfg["outside_work_fraction"]))
    cfg["start_grid_min"] = max(1, cfg["start_grid_min"])
    cfg["start_spread_min"] = max(0, cfg["start_spread_min"])
    cfg["hours_min"] = max(1, cfg["hours_min"])
    cfg["min_outside_steps"] = max(1, cfg["min_outside_steps"])
    return cfg


def cfg_of(sim) -> dict:
    """圏外滞在の設定(初回のみ遅延構築してキャッシュ)。キャッシュ属性は L1/L2/乱数に出ない。"""
    c = getattr(sim, "planboundarycfg", None)
    if c is None:
        try:
            dp = (sim.cfg.get("planning", None) or {}).get("day_plan", None) or {}
            raw = dp.get("boundary", None)
        except Exception:                          # noqa: BLE001 (旧 config 互換)
            raw = None
        c = build_cfg(raw)
        sim.planboundarycfg = c
    return c


def enabled(sim) -> bool:
    """計画駆動の圏外滞在が実効か。**day_plan v1 が前提**(計画がブロックの時刻表だから)。"""
    from . import day_plan as _day_plan
    if not _day_plan.enabled(sim):
        return False
    return bool(cfg_of(sim)["enabled"])


# --------------------------------------------------------------------------- #
# 指名(決定論。RngHub を 1 本も引かない = 既存の draw 順を汚さない)
# --------------------------------------------------------------------------- #
def stable_uniform(seed: int, key: str) -> float:
    """(seed, 安定キー)から一様値 [0,1)。work._stable_uniform / ontology と同流儀。

    hashlib=プロセス跨ぎで安定 = リプレイ・resume・別ランで同一値。乱数ストリームを
    引かないので、既存機構の draw 順は 1 本もずれない。
    """
    h = hashlib.blake2b(f"{int(seed)}\x1f{key}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") / float(1 << 64)


def _seed_of(sim) -> int:
    try:
        return int(sim.cfg.run.seed)
    except Exception:                              # noqa: BLE001
        return 0


def eligible(agent) -> bool:
    """圏外通勤の**候補**か = 「地図内に解決できる勤務先を持たない居住者」。

    ★`work.bind_workplace`(職場束ね直し・別レーン)との関係: 束ねが ON で `work_node` が
      付いた個体は **その時点で候補から外れる**(域内に実在の職場を持つのだから圏外通勤
      ではない)。判定は毎回 agent の現在状態から取り直す純関数なので、束ねが先に走れば
      束ねが勝つ = 二重指名も競合も起きない(束ねは hydrate/初期化で完了し、朝の計画は
      その後に立つ)。
    ★流入通勤者(`commute`)・来街者(`visitor`)は**統計駆動の台帳**の担当なので除外する。
    """
    if bool(getattr(agent, "visitor", False)) or bool(getattr(agent, "commute", False)):
        return False
    if str(getattr(agent, "work_node", "") or ""):
        return False
    if getattr(agent, "part_time", None):          # バイト先 = 地図内の勤務アンカー
        return False
    return True


def designated(sim, agent, cfg: dict | None = None) -> bool:
    """この個体が圏外通勤者か(**(run.seed, agent.id) の純関数**)。"""
    if not eligible(agent):
        return False
    cfg = cfg or cfg_of(sim)
    frac = float(cfg["outside_work_fraction"])
    if frac <= 0.0:
        return False
    if frac >= 1.0:
        return True
    return stable_uniform(_seed_of(sim), f"work_outside/{int(agent.id)}") < frac


def gateway_of(sim, agent, cfg: dict | None = None) -> str:
    """圏外への出入口ノード(決定論)。既定は駅、無ければ縁 gateway から安定選択。"""
    cfg = cfg or cfg_of(sim)
    station = str(getattr(sim.city, "station_node", "") or "")
    if str(cfg["gateway"]) == "station" and station:
        return station
    gates = list(getattr(sim.city, "gateways", []) or [])
    if not gates:
        return station
    idx = int(stable_uniform(_seed_of(sim), f"work_outside_gate/{int(agent.id)}")
              * len(gates))
    return str(gates[min(idx, len(gates) - 1)])


def window_of(sim, agent, cfg: dict | None = None) -> tuple[int, int]:
    """骨格計画で使う圏外勤務の時間窓(決定論)。(start_min, end_min)。"""
    cfg = cfg or cfg_of(sim)
    grid = int(cfg["start_grid_min"])
    n = int(cfg["start_spread_min"]) // grid + 1
    i = int(stable_uniform(_seed_of(sim), f"work_outside_start/{int(agent.id)}") * n)
    start = int(cfg["start_min"]) + min(i, n - 1) * grid
    end = min(1439, start + int(cfg["hours_min"]))
    return int(start), int(end)


def ensure(sim, agent) -> bool:
    """個体の圏外通勤フラグを確定する(**冪等な純関数の書き戻し**・乱数ゼロ)。

    毎朝の計画生成の先頭で呼ぶ。フィールドは観測とチェックポイント用の**鏡**であって
    真実の源ではない(真実の源は `designated` = (seed, id) の純関数)。したがって
    resume でも古い checkpoint でも同じ答えになる。
    """
    if not enabled(sim):
        return False
    if str(getattr(agent, "loc", "street")) != "outside":
        # 前日のやり残し(縁へ着けないまま日を跨いだ保留)は翌朝の計画が上書きする。
        # 圏外に居る個体の保留は帰還の記録に要るので絶対に落とさない。
        agent._boundary_pending = None
    cfg = cfg_of(sim)
    on = designated(sim, agent, cfg)
    agent.work_outside = bool(on)
    if on:
        agent.work_outside_gateway = gateway_of(sim, agent, cfg)
        s, e = window_of(sim, agent, cfg)
        agent.work_outside_start_min, agent.work_outside_end_min = s, e
    else:
        agent.work_outside_gateway = ""
        agent.work_outside_start_min = -1
        agent.work_outside_end_min = -1
    return bool(on)


# --------------------------------------------------------------------------- #
# 計画スキーマ: boundary 欄の刻印(**明示欄**=集計はただのクエリ)
# --------------------------------------------------------------------------- #
def is_boundary(b) -> bool:
    """ブロックが圏外ブロックか(**純関数**。`boundary` 欄の有無だけを見る)。"""
    info = b.get("boundary")
    return isinstance(info, dict) and bool(info.get("exit"))


def mark(sim, agent, b) -> str | None:
    """ブロックの `boundary` 欄を**確定**する(唯一の刻印点)。返り値 = 縁ノード or None。

    OFF / 非該当なら欄を消して None を返す(前日の計画を再利用した際の**古い刻印の
    持ち越し**をここで断つ)。`node` は圏外なので必ず None にする — 地図内のノードを
    入れると「そこに居る」と読めてしまい、既存欄の意味を上書きしてしまうから
    (§4「既存欄に多重定義を作らない」)。
    """
    if not enabled(sim) or not bool(getattr(agent, "work_outside", False)) \
            or str(b.get("place")) not in cfg_of(sim)["places"]:
        if "boundary" in b:
            del b["boundary"]
        return None
    gate = str(getattr(agent, "work_outside_gateway", "") or "") \
        or gateway_of(sim, agent)
    b["boundary"] = {
        "exit": True,
        "gateway": gate,
        "exit_min": int(b["start"]) if b.get("start") is not None else -1,
        "entry_min": int(b["end"]) if b.get("end") is not None else -1,
    }
    b["node"] = None
    return gate


def skeleton_rows(sim, agent, dp_cfg: dict) -> list | None:
    """圏外通勤者のペルソナ既定骨格(LLM なし・乱数ゼロ)。非該当/OFF は None。

    day_plan.skeleton は「勤務窓を持つ人/持たない人」の 2 分岐だが、圏外通勤者は
    `work_start_min < 0`(地図内に職場が無い)ため後者に落ちて**圏外へ出ない**。
    ここで第 3 の骨格「圏外勤務 → 帰ってきて食事 → 家」を返す。
    """
    if not enabled(sim) or not ensure(sim, agent):
        return None
    s = int(getattr(agent, "work_outside_start_min", -1))
    e = int(getattr(agent, "work_outside_end_min", -1))
    if s < 0 or e <= s:
        return None
    day_end = int(dp_cfg["day_end_min"])
    lo = int(dp_cfg["min_dur_min"])

    def _b(start, end, place, act, priority, flex, reason):
        from . import day_plan as _dp
        return {"start": int(start), "end": int(end), "place": place, "act": act,
                "with": [], "purpose": _dp.ACT_PURPOSE[act], "priority": priority,
                "flex": flex, "reason": reason, "note": "",
                "node": None, "state": "todo", "slid": 0}

    rows = [_b(s, e, "work", "work", "must", "fixed", "いつもの時間に働く"),
            _b(e, min(day_end, e + 60), "food", "meal", "should", "slideable",
               "働いた後に食べる"),
            _b(min(day_end - lo, e + 60), min(day_end, e + 180), "home", "home",
               "should", "slideable", "家に帰って休む")]
    return [r for r in rows if r["start"] < day_end and r["end"] > r["start"]]


# --------------------------------------------------------------------------- #
# 実行: 退出(_try_exit)と帰還(_phase_wake_and_returns)
# --------------------------------------------------------------------------- #
def begin(sim, agent, plan: dict, b: dict, index: int, rng) -> dict | None:
    """圏外ブロックの実行開始。返り値 = 行動 dict(縁へ移動)/ {} (既に縁に居る) / None。

    ★ここで世界を動かすのは `agent` 自身の意図フラグ 2 つだけ(`exit_intent` と保留)。
      実際の退出は既存の `_try_exit`(縁への到着 or `pending_exits` の再試行)が行う。
    """
    if not enabled(sim) or not is_boundary(b):
        return None
    info = b["boundary"]
    gate = str(info.get("gateway") or "") or gateway_of(sim, agent)
    agent._boundary_pending = {
        "gateway": gate, "block": int(index), "day": int(plan.get("day", -1)),
        "place": str(b.get("place") or ""),
        "exit_min": int(info.get("exit_min", -1)),
        "entry_min": int(info.get("entry_min", -1)),
    }
    agent.exit_intent = True
    if str(getattr(agent, "node", "") or "") == gate:
        return {}                                  # 既に縁 = 移動不要(pending_exits が退出させる)
    from . import routine as _routine
    return {"type": "move_to", "dest": gate, "stay_steps": 0,
            "mode": _routine._choose_mode(agent, sim, gate, rng),
            "exit": True, "activity": "commuting"}


def pending_exits(sim, step: int) -> tuple:
    """縁に居るのに退出できていない**計画退出者**(scheduler の再試行口)。OFF は空 tuple。

    envfeedback.pending_exits / devices.pending_exits と**同じ形・同じ位置**の再試行口。
    `_try_exit` は「ノードに到着した step」にしか呼ばれないので、
      (a) 計画開始時に既に縁に居た(移動が発生しない)
      (b) 終電・遅延・改札待ちで退出が保留された
    の 2 つはここでしか通る口が無い。既に退出済み(loc=="outside")の個体は
    `exit_intent` が落ちているので二重には拾わない。
    """
    if not enabled(sim):
        return ()
    out = []
    for a in sim.agents:
        p = getattr(a, "_boundary_pending", None)
        if not p or not getattr(a, "exit_intent", False):
            continue
        if a.loc != "street" or a.sleeping or a.route:
            continue
        if str(a.node or "") != str(p.get("gateway") or ""):
            continue
        out.append(a)
    return tuple(out)


def steps_until_tod(cur_sim_min: int, target_min: int, step_minutes: int) -> int:
    """現在の sim_min から次に time-of-day=target_min になるまでの step 数(正)。

    ★`engine/scheduler._steps_until_tod` と**同一の式**(流入通勤者の翌朝到着で既に
      使われている決定論変換)。cognition → engine の逆向き import を作らないために
      式だけを持つ(同値であることは tests/test_plan_boundary.py が機械固定する)。
    """
    delta = (int(target_min) - int(cur_sim_min) % 1440) % 1440
    if delta == 0:
        delta = 1440                               # 同時刻 = 次の周回(翌日)
    return delta // int(step_minutes)


def on_exit(sim, agent, step: int, sim_min: int) -> dict | None:
    """計画退出の確定。返り値 = `exit_area` payload への追記欄 / None(非該当)。

    ★`return_at` を**乱数ではなくブロックの終了時刻**から決める(これが「計画が境界の
      時刻表である」ということ)。`world.outside_steps` の乱数draw は 1 本も引かない。
    ★帰宅移動中(homing)の退出は計画退出として扱わず保留を捨てる — 退出に印が付かない
      以上、帰還にも付けない(= 保存則を壊さない)。
    """
    p = getattr(agent, "_boundary_pending", None)
    if p is None:
        return None
    # 保留の取り違えを構造で防ぐ: 帰宅移動中 / 別の日 / 別の縁からの退出は
    # 「この計画ブロックの退出」ではない → 保留を捨てて従来経路へ譲る(印を付けない
    # 以上、帰還にも付かない = 出入りの対称性=保存則を壊さない)。
    if not enabled(sim) or bool(getattr(agent, "homing", False)) \
            or int(sim_min) // 1440 != int(p.get("day", -2)) \
            or str(getattr(agent, "node", "") or "") != str(p.get("gateway") or ""):
        agent._boundary_pending = None
        return None
    cfg = cfg_of(sim)
    entry = int(p.get("entry_min", -1))
    stay = max(int(cfg["min_outside_steps"]),
               steps_until_tod(sim_min, entry, sim.clock.step_minutes))
    agent.return_at = int(step) + int(stay)
    p["return_at"] = int(agent.return_at)
    p["exit_step"] = int(step)
    return {"boundary": "plan", "block": int(p["block"]),
            "place": str(p["place"]), "exit_min": int(p["exit_min"]),
            "entry_min": entry}


def memo_line(place: str, exit_min: int, entry_min: int) -> str:
    """帰還時の圧縮記憶 1 行。**(場所カテゴリ, 予定時刻) だけの純関数**(LLM ゼロ・乱数ゼロ)。

    実験条件・k・因子名・機構語を 1 バイトも含まない(no-fingerprint)。実際の退出/帰還
    step ではなく**予定時刻**を書く: 圏外に居た間の出来事は生成していないのだから、
    本人が語れるのは「そういう予定だった」ことだけである(正直な近似 1)。
    """
    return (f"{_hhmm(exit_min)}〜{_hhmm(entry_min)}は"
            f"{PLACE_JA.get(str(place), str(place))}で街の外に居た。{MEMO_TAIL}")


def _hhmm(m: int) -> str:
    m = int(m) % 1440 if int(m) >= 0 else 0
    return f"{m // 60:02d}:{m % 60:02d}"


def on_return(sim, agent, step: int, sim_min: int) -> dict | None:
    """計画帰還の確定。返り値 = `enter_area` payload への追記欄 / None(非該当)。

    ①圧縮記憶を **1 ブロックにつき 1 行だけ**入れる ②計画ブロックを消化済みに印す
    ③保留を落とす。LLM ゼロ・乱数ゼロ・新イベント種ゼロ。
    """
    p = getattr(agent, "_boundary_pending", None)
    if p is None:
        return None
    agent._boundary_pending = None
    if not enabled(sim):
        return None
    agent.remember(memo_line(p["place"], p["exit_min"], p["entry_min"]))
    plan = getattr(agent, "_dayplan", None)
    if isinstance(plan, dict) and int(plan.get("day", -1)) == int(p.get("day", -2)):
        blocks = plan.get("blocks") or []
        i = int(p["block"])
        if 0 <= i < len(blocks) and blocks[i].get("state") == "away":
            blocks[i]["state"] = "done"            # 圏外滞在を最後まで果たした
    return {"boundary": "plan", "block": int(p["block"]),
            "place": str(p["place"]), "exit_min": int(p["exit_min"]),
            "entry_min": int(p["entry_min"])}
