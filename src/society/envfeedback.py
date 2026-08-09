"""環境フィードバック(エージェント → 環境)= 最小 3 規則(第84バッチ 2026-08-01)。

正典: docs/plans/source/cognition-design-record.md **§4 全体** /
      docs/plans/cognition-physics-plan.md §4 第84行(T5 非発散)。

何を解く問題か
--------------
これまでの因果は **環境 → エージェント の一方向**だった。何が起きても「入力に反応した」で
説明できてしまうため、創発を主張する足場が無い。本 module は逆向きの矢印を 1 本だけ通す:

    行動 → 集約物理量 → 閾値超過 → 環境イベント → 知覚 → 行動      (§4.3「環が閉じる」)

★原則(§4.1): **エージェントは環境イベントを宣言できない**。LLM に「電車が遅れた」と
  言わせた瞬間、世界は「エージェントが言ったこと」になり物理制約の前提が消える。
  ここに書く規則は **集約された物理量が閾値を超えたときにコード側が発火**する。
  したがって本 module は LLM を 1 本も呼ばず、乱数を 1 本も引かない(完全決定論)。

規則セット(§4.2 の最小構成 3 本。それ以上は足さない = 指紋対策 (1))
-------------------------------------------------------------------
| エージェント行動 | 集約物理量           | 環境イベント                     |
|------------------|----------------------|----------------------------------|
| ホームへの流入   | ホーム密度           | 停車時間延長 → 遅延伝播          |
| 改札通過         | 改札スループット飽和 | 入場規制                         |
| POI 滞在         | 占有率 > 容量        | 待ち行列 → 他 POI へ流出         |

実装上の「ホーム」= 唯一の駅ノード `city.station_node`(現行の地図はホーム/改札を
別ノードに持たない)。したがって集約量はどちらもこの 1 ノードで測る:

  ホーム負荷(規則1) = **ホームに立っている人数 + この step の乗降人数**
      (乗降 = 電車で街を出た人 `exit_area(via=train)` + 電車で来た人 `enter_area`)
  改札流入(規則2) = **この step に駅ノードへ入った件数**(`arrive` + `enter_area`)

★規則1 に乗降人数を入れているのは実測に基づく設計判断である: 現行エンジンの駅は
  **通過点**(着いた step のうちに範囲外へ抜ける)なので step 末の在ノード人数だけを見ると
  ほぼ 0 になり、規則が原理的に発火しない。停車時間の実証モデルでも主説明変数は「そこに
  立っている人」ではなく「乗り降りする人」なので、現実の機構としてもこちらが正しい。
**この粗さは正直な限界**として下の「限界」節に明記する。

タイミング(§4.5)
------------------
環境イベントの計算は **step 末に集約状態から一括**で行う(`update()`)。step 途中で
計算すると、その step 内の処理順が結果に混入して同期バリア(第81)の意味が消える。
規則が世界へ効くのは **次 step 以降**(ダブルバッファ)= 第81 の凍結観測と同じ構造。

二つの罠(§4.4)への対策
-------------------------
- **発散**: 遅延 → 混雑 → 遅延 は正のフィードバックで容易に爆発する。3 規則すべてに
  **減衰項と上限**を入れてある。
    規則1: 回復運転項 `recovery`(γ<1 を毎 step 掛ける)+ 遅延上限 `delay_cap_min`
           + 1 step あたりの注入上限 `dwell_cap_min` + 個体の待ち上限 `max_hold_steps`。
           不動点は `dwell_cap_min/(1−γ)` で上から押さえられる(T5 が実測で固定)。
    規則2: 規制の継続は `hold_steps` で有限・解除後は `cooldown_steps` の間 再発動しない
           + 個体の待ちは `max_hold_steps` を超えない(= 必ず通す安全弁)。
    規則3: 除外は `hold_steps` で自動失効・同時に除外できるノード数は `max_nodes` まで
           + 候補が全滅する除外はしない(安全弁)。
- **指紋**: 閾値と規則そのものが指紋である。対策は (1) 規則を最小限にする=3 本のみ、
  (2) パラメータを実データで較正する=下の「較正の口」、(3) **ON/OFF を実験条件にする**
  = `env.feedback.enabled`(既定 false)を registry へ宣言し、OFF ランとの差分でのみ
  「環境からの反作用があるとき何が変わるか」を言う。
  ★プロンプトには 1 バイトも足さない。遅延・規制・待ちは **世界の事実**として
  「動けない / 候補に無い / 混雑の不快感(既存 congestion 語彙)」の形でしか現れない。
  新語彙・新欄・「実験」を示唆する文言は一切導入しない。

較正の口(§4.4 対策2)
----------------------
`dwell_sec_per_pax` / `platform_threshold` / `gate.capacity_per_min` は **仮値**である。
将来 渋谷駅の実データ(遅延統計・改札通過人数)で較正する前提で conf にコメントを置いた。
文献の裏づけ(実装前リサーチ 2026-08-01):
  - 乗降人数がホーム/車内の混雑と相互作用して停車時間を延ばすこと、通勤列車の遅延の
    9 割前後が 5 分以下の小さな停車時間超過であること、乗降人数で説明できるのは
    約 40% であること(Stockholm/Tokyo の実測比較)。→ 「密度→停車時間→遅延」は
    現実の主機構として妥当だが、**単独では説明力 4 割**である(= 過信しない)。
  - 混雑時に駅の改札を閉じて入場を止めるのは実運用の常套手段(crowd control gating)。
  - 遅延は次列車へ伝播し、余裕時分(running time supplement)で回復する。回復は
    「毎区間で遅延に γ<1 を掛ける」形で近似できる。→ 規則1 の `recovery`。

R1 ドクトリン
-------------
- 既定 `env.feedback.enabled: false` では `enabled()` が False を返し、すべての作用点が
  即 return する = L1/L2/L3 バイト一致・乱数消費不変・LLM 呼数不変。
- 乱数を 1 本も引かない(RngHub に触れない)ので repro_tier は **strict**。
- LLM の発生点は 1 つも足さない。ON では物理位置(co-location)が変わるので発火数は
  間接的に動きうる = 既存の commerce / disaster / health と同じ扱い(affects_k=False)。
- 本ファイルは `src/society` 直下(engine/cognition/world/actions/labeling の静的検査対象外)。

限界(正直な宣言)
------------------
1. **駅は 1 ノード**。ホーム・改札・コンコースの区別が地図に無いので、「ホーム密度」と
   「改札流入」は同じノードの別の測り方(在ノード人数 / この step の到着件数)になる。
   入場規制は本来「改札の外で待たせる」が、ここでは「駅ノードから出られない」= 実質
   ホーム上で待つ形になる。物理層(第86〜)でノードが割れたら作用点を割り直すこと。
2. **下流列車への伝播は時間方向のみ**。世界に駅が 1 つしか無いので、空間的な下流駅への
   伝播は表現できない。代わりに「次の発車へ γ を掛けて持ち越す」= 同一駅の連続する
   発車列に沿った伝播として実装した(遅延伝播モデルの標準形の 1 駅版)。
3. **待ち行列の実体は無い**。規則3 の「行列」は L1 イベント + 行き先候補からの一時除外
   であって、並んで待つ個体の列は作らない(既存の POI 選択経路だけを使う = 新しい
   ヒューリスティックを足さない、という指示に従った)。
4. 閾値・係数はすべて **仮値**。実データ較正は未実施(上記「較正の口」)。
"""
from __future__ import annotations

import math

from . import devices as devices_mod
from .factors import update as factor_update
from .observer.schema import Event

# L1 に出すイベント種(observer/schema.py に登録済み)。1 種だけに束ねる:
# 「どの規則が・どの集約量を・どの閾値と比べて・どれだけ超えたか」を payload で辿る(§4.3)。
EVENT_KIND = "env_feedback"

# 規則 id(payload の "rule"。ここが唯一の定義)
RULE_TRANSIT = "transit_dwell"     # ホーム密度 → 停車時間延長 → 遅延
RULE_GATE = "gate_restrict"        # 改札スループット飽和 → 入場規制
RULE_POI = "poi_queue"             # POI 占有 > 容量 → 待ち行列 → 他 POI へ流出


# --------------------------------------------------------------------------- #
# cfg 正準化(既定 OFF)
# --------------------------------------------------------------------------- #
def build_cfg(raw) -> dict:
    """conf の `env.feedback` ブロックを型強制つきで正準化する(既定 OFF)。

    dotlist / OmegaConf のどちらでも受ける(disaster / commerce / dunbar と同型)。
    """
    from omegaconf import OmegaConf
    if OmegaConf.is_config(raw):
        raw = OmegaConf.to_container(raw, resolve=True)
    raw = dict(raw or {})
    tr = dict(raw.get("transit") or {})
    ga = dict(raw.get("gate") or {})
    po = dict(raw.get("poi") or {})
    return {
        "enabled": bool(raw.get("enabled", False)),
        "log_every_steps": max(1, int(raw.get("log_every_steps", 1) or 1)),
        "transit": {
            "enabled": bool(tr.get("enabled", True)),
            "platform_threshold": int(tr.get("platform_threshold", 25)),
            "dwell_sec_per_pax": float(tr.get("dwell_sec_per_pax", 0.8)),
            "dwell_cap_min": float(tr.get("dwell_cap_min", 3.0)),
            "recovery": float(tr.get("recovery", 0.7)),
            "delay_cap_min": float(tr.get("delay_cap_min", 15.0)),
            "flag_min": float(tr.get("flag_min", 3.0)),
            "max_hold_steps": max(0, int(tr.get("max_hold_steps", 3))),
        },
        "gate": {
            "enabled": bool(ga.get("enabled", True)),
            "capacity_per_min": float(ga.get("capacity_per_min", 1.5)),
            "hold_steps": max(1, int(ga.get("hold_steps", 2))),
            "cooldown_steps": max(0, int(ga.get("cooldown_steps", 3))),
        },
        "poi": {
            "enabled": bool(po.get("enabled", True)),
            "capacity": int(po.get("capacity", 12)),
            "hold_steps": max(1, int(po.get("hold_steps", 3))),
            "max_nodes": max(0, int(po.get("max_nodes", 8))),
        },
    }


def enabled(sim) -> bool:
    """環境フィードバック層が有効か。既定 OFF=新経路を一切通さない(バイト一致)。"""
    cfg = getattr(sim, "envfbcfg", None)
    return bool(cfg and cfg["enabled"])


def _new_state() -> dict:
    """中央管理する可変状態(素の int/float/dict のみ = checkpoint 直列化が自明)。"""
    return {
        # 規則1: いまの遅延[分](路線共通の 1 本。1 駅モデルなので線別に割らない)
        "delay_min": 0.0,
        # 規則2: 入場規制の解除 step / 再発動を許す step / 直前 step の規制フラグ
        "gate_until": -1,
        "gate_cool_until": -1,
        "gate_flag": False,
        # 規則3: 一時除外中の POI ノード -> 解除 step
        "poi_hold": {},
        # 累積カウンタ(観測用。単調非減少)
        "n_transit": 0,
        "n_gate": 0,
        "n_poi": 0,
        "n_wait": 0,
        "delay_max": 0.0,
    }


def state(sim) -> dict:
    """可変状態を返す(無ければ作る)。ON のときだけ生える = OFF は属性すら無い。"""
    st = getattr(sim, "_envfb", None)
    if st is None:
        st = _new_state()
        sim._envfb = st
    return st


# --------------------------------------------------------------------------- #
# 補助(すべて決定論・読み取り専用)
# --------------------------------------------------------------------------- #
def _dt_min(sim) -> float:
    clock = getattr(sim, "clock", None)
    return float(getattr(clock, "step_minutes", 10) or 10)


def _station(sim):
    city = getattr(sim, "city", None)
    return getattr(city, "station_node", None) if city is not None else None


def _poi_nodes(sim) -> tuple[str, ...]:
    """POI を持つノードの一覧(並びは固定。1 回だけ作って sim にぶら下げる)。"""
    cached = getattr(sim, "_envfb_poi_nodes", None)
    if cached is None:
        nodes = {str(p.get("node")) for p in getattr(sim.city, "poi_list", ())
                 if p.get("node") is not None}
        cached = tuple(sorted(nodes))
        sim._envfb_poi_nodes = cached
    return cached


def gate_scale(sim) -> float:
    """`world.mod.gate_capacity` の係数(第67バッチの予約フィールドを**ここで初めて消費**)。

    形は `{名前: 正の係数}`。駅ノード id のキーがあればそれを、無ければ単一エントリの値を、
    どちらでもなければ 1.0(改変なし)を返す。world.mod OFF なら常に 1.0。
    """
    wm = getattr(sim, "worldmod", None)
    if wm is None:
        return 1.0
    try:
        raw = dict(wm.doc.get("gate_capacity") or {})
    except AttributeError:                                # pragma: no cover - 形が違う場合
        return 1.0
    if not raw:
        return 1.0
    st_node = _station(sim)
    if st_node is not None and st_node in raw:
        return float(raw[st_node])
    if len(raw) == 1:
        return float(next(iter(raw.values())))
    return 1.0


def mark_gate_capacity_consumed(sim) -> None:
    """`world.mod` の予約フィールド記録を「消費済み」に書き換える(summary の正直な記録)。

    第67 は `{"n_entries": n, "consumed": False}` を summary に残していた。規則2 が
    有効なランではこの値が実際に効くので、その旨を summary で判るようにする。
    """
    wm = getattr(sim, "worldmod", None)
    if wm is None or not enabled(sim) or not sim.envfbcfg["gate"]["enabled"]:
        return
    rsv = getattr(wm, "reserved", None)
    if isinstance(rsv, dict) and isinstance(rsv.get("gate_capacity"), dict):
        rsv["gate_capacity"]["consumed"] = True
        rsv["gate_capacity"]["consumer"] = "env.feedback.gate"


def _log(sim, step: int, sim_min: int, agent_id: int, payload: dict,
         x: float = 0.0, y: float = 0.0) -> None:
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent_id,
                         kind=EVENT_KIND, x=x, y=y, payload=payload))


def _due(sim, step: int) -> bool:
    """L1 の記録間引き(観測装置の設定であって世界の量ではない)。"""
    return int(step) % int(sim.envfbcfg["log_every_steps"]) == 0


# --------------------------------------------------------------------------- #
# step 末の一括計算(§4.5)
# --------------------------------------------------------------------------- #
def update(sim, step: int, sim_min: int, since_idx: int) -> None:
    """この step 末の集約状態から 3 規則をまとめて評価し、環境状態を進める。

    `since_idx` は run_step 冒頭で控えた `len(sim.logger.events)`(第80 channels と同じ流儀)。
    **世界の位置・所持金・関係は 1 つも書き換えない**: ここで動くのは本 module の状態だけで、
    それを次 step 以降の作用点(`hold_exit` / `hold_return` / `blocked_nodes`)が読む。
    """
    if not enabled(sim):
        return
    cfg = sim.envfbcfg
    st = state(sim)
    station = _station(sim)

    # ---- 集約(この step 末の世界を 1 回だけ走査する)----
    at_node: dict[str, int] = {}
    for agent in sim.agents:
        if agent.loc == "outside" or agent.sleeping:
            continue
        at_node[agent.node] = at_node.get(agent.node, 0) + 1

    # この step の駅まわりのフロー(L1 を読むだけ)。
    #   inflow   … 改札通過 = 駅ノードへ入った件数(到着 + 範囲外からの帰還)= 規則2 の集約量
    #   exchange … 乗降人数 = 電車で街を出た人 + 電車で街へ来た人 = 規則1 の主説明変数
    #              (停車時間の実証モデルの説明変数は『そこに立っている人』ではなく『乗り降りする人』)
    inflow = 0
    exchange = 0
    if station is not None and since_idx >= 0:
        for ev in sim.logger.events[since_idx:]:
            payload = ev.payload or {}
            if ev.kind == "arrive" and payload.get("node") == station:
                inflow += 1
            elif ev.kind == "enter_area" and payload.get("gateway") == station:
                inflow += 1
                exchange += 1
            elif ev.kind == "exit_area" and payload.get("via") == "train":
                exchange += 1

    log_now = _due(sim, step)

    # ---- 規則1: ホーム密度 → 停車時間延長 → 遅延(回復運転で減衰)----
    tcfg = cfg["transit"]
    if tcfg["enabled"] and station is not None:
        # ホーム負荷 = ホームに立っている人 + この step の乗降人数。後者を入れないと、現行エンジンの
        # 駅が**通過点**(着いた step のうちに範囲外へ抜ける)なので step 末の在ノード人数がほぼ 0 に
        # なり、規則が原理的に発火しない(検収で実測して設計変更した)。実証モデルでも主説明変数は
        # 乗降人数であり、こちらの方が現実の機構に近い。
        standing = at_node.get(station, 0)
        n = standing + exchange
        thr = int(tcfg["platform_threshold"])
        excess = max(0, n - thr)
        # 1 step ぶんの停車時間超過[分](上限つき)。乗降人数に比例する実証モデルの最小形(天井は Δt でスケールする=timeconv RATE)。
        inject = min(float(tcfg["dwell_cap_min"]),
                     excess * float(tcfg["dwell_sec_per_pax"]) / 60.0)
        prev = float(st["delay_min"])
        # 回復運転項 γ<1 + 上限 = 発散しない(不動点 ≤ dwell_cap_min/(1−γ) かつ ≤ delay_cap_min)
        new = min(float(tcfg["delay_cap_min"]),
                  float(tcfg["recovery"]) * prev + inject)
        if new < 1e-9:
            new = 0.0
        st["delay_min"] = new
        if new > float(st["delay_max"]):
            st["delay_max"] = new
        if excess > 0:
            st["n_transit"] += 1
        if log_now and (excess > 0 or (prev > 0.0 and new == 0.0)):
            _log(sim, step, sim_min, -1,
                 {"rule": RULE_TRANSIT, "node": station, "metric": "platform_n",
                  "value": n, "threshold": thr, "excess": excess,
                  "standing": standing, "exchange": exchange,
                  "injected_min": round(inject, 3),
                  "delay_min": round(new, 3), "prev_min": round(prev, 3),
                  "recovery": float(tcfg["recovery"]),
                  "cap_min": float(tcfg["delay_cap_min"])})

    # ---- 規則2: 改札スループット飽和 → 入場規制 ----
    # ★装置層(actor model P2 / src/society/devices.py)との優先順位:
    #   `world.devices.faregate.enabled` が ON のとき、本規則は **1 度も評価しない**。
    #   規則2 は「1 step の駅流入件数 > 容量 なら数 step 入場規制」という**全体一括の
    #   近似**で、装置層の改札(通路数 × 処理間隔の待ち行列)と**同じ物理**(改札の
    #   処理能力の飽和)を指しているため、両方効かせると同じ現象の二重計上になる。
    #   細粒度モデルが在るときは細粒度が勝つ = `gate_until` も `n_gate` も 1 も動かない
    #   (tests/test_devices.py が「装置 ON で envfeedback 規則2 の状態が不動」を固定)。
    #   規則1(停車時間 → 遅延)と規則3(POI 占有)は別の物理なのでそのまま生きる。
    #   既定 OFF(devices.enabled=false)では常に False = 従来と完全同一 = バイト一致。
    gcfg = cfg["gate"]
    if gcfg["enabled"] and station is not None and not devices_mod.faregate_active(sim):
        cap = float(gcfg["capacity_per_min"]) * _dt_min(sim) * gate_scale(sim)
        active = int(st["gate_until"]) > step
        if not active and inflow > cap and step >= int(st["gate_cool_until"]):
            st["gate_until"] = step + int(gcfg["hold_steps"])
            st["gate_cool_until"] = st["gate_until"] + int(gcfg["cooldown_steps"])
            st["n_gate"] += 1
            active = True
            _log(sim, step, sim_min, -1,
                 {"rule": RULE_GATE, "node": station, "metric": "gate_inflow",
                  "value": inflow, "threshold": round(cap, 3),
                  "excess": round(inflow - cap, 3), "phase": "onset",
                  "until_step": int(st["gate_until"]),
                  "cooldown_until": int(st["gate_cool_until"])})
        elif bool(st["gate_flag"]) and not active:
            _log(sim, step, sim_min, -1,
                 {"rule": RULE_GATE, "node": station, "metric": "gate_inflow",
                  "value": inflow, "threshold": round(cap, 3), "phase": "clear",
                  "cooldown_until": int(st["gate_cool_until"])})
        st["gate_flag"] = bool(active)

    # ---- 規則3: POI 占有 > 容量 → 待ち行列 → 他 POI へ流出 ----
    pcfg = cfg["poi"]
    if pcfg["enabled"]:
        hold: dict = st["poi_hold"]
        for node in sorted(hold):                       # 期限切れの除外を落とす(減衰)
            if int(hold[node]) <= step:
                del hold[node]
        cap_n = int(pcfg["capacity"])
        max_nodes = int(pcfg["max_nodes"])
        for node in _poi_nodes(sim):                    # 並びは固定(決定論)
            if len(hold) >= max_nodes:
                break
            if node in hold or node == station:
                continue
            n = at_node.get(node, 0)
            if n > cap_n:
                hold[node] = step + int(pcfg["hold_steps"])
                st["n_poi"] += 1
                if log_now:
                    _log(sim, step, sim_min, -1,
                         {"rule": RULE_POI, "node": node, "metric": "occupancy",
                          "value": n, "threshold": cap_n, "excess": n - cap_n,
                          "until_step": int(hold[node])})


# --------------------------------------------------------------------------- #
# 作用点(1)(2): 駅から範囲外へ出る = 改札を通る
# --------------------------------------------------------------------------- #
def _release(agent) -> None:
    agent._env_exit_wait = -1
    agent._env_exit_held = 0


def _delay_steps(sim, st: dict) -> int:
    """いまの遅延[分] → 待たせる step 数(切り上げ)。"""
    dt = _dt_min(sim)
    return int(math.ceil(float(st["delay_min"]) / dt)) if st["delay_min"] > 0.0 else 0


def hold_exit(sim, agent, step: int, sim_min: int) -> bool:
    """駅の改札を通れるか。True = この step は通れない(駅に留まる)。

    2 つの規則がここで合流する:
      - 規則1(遅延): 電車が遅れているぶん出発できない。
      - 規則2(入場規制): 規制中は通さない。
    **上限**: 1 回の退出につき合計 `max_hold_steps` を超えて待たせない(必ず通す安全弁)。
    これが「駅に人が溜まり続けて二度と捌けない」発散を原理的に止める(T5)。
    """
    if not enabled(sim):
        return False
    prev_wait = int(getattr(agent, "_env_exit_wait", -1))
    if prev_wait > step:
        return True                                     # 待機中(記録は開始時に 1 回だけ)
    if prev_wait >= 0 and step > prev_wait:
        _release(agent)                                 # 前回の待ちと連続していない=別のトリップ
    cfg = sim.envfbcfg
    st = state(sim)
    budget = int(cfg["transit"]["max_hold_steps"]) - int(getattr(agent, "_env_exit_held", 0))
    if budget <= 0:
        _release(agent)
        return False
    wait, rule = 0, ""
    if cfg["gate"]["enabled"] and int(st["gate_until"]) > step:
        wait, rule = int(st["gate_until"]) - step, RULE_GATE
    if cfg["transit"]["enabled"]:
        d = _delay_steps(sim, st)
        if d > wait:
            wait, rule = d, RULE_TRANSIT
    wait = min(wait, budget)
    if wait <= 0:
        _release(agent)
        return False
    agent._env_exit_wait = step + wait
    agent._env_exit_held = int(getattr(agent, "_env_exit_held", 0)) + wait
    st["n_wait"] += 1
    _log(sim, step, sim_min, agent.id,
         {"rule": rule, "effect": "exit_wait", "wait_steps": wait,
          "delay_min": round(float(st["delay_min"]), 3),
          "gate_until": int(st["gate_until"])},
         x=agent.x, y=agent.y)
    _felt(sim, agent, wait, step, sim_min)
    return True


def hold_return(sim, agent, step: int, sim_min: int) -> bool:
    """駅経由の帰還を遅延ぶん待たせるか。True = この step は帰って来られない。

    `return_at` を直接書き換えず、専用の待ち時刻で判定する(既存の帰還条件を壊さない)。
    上限は退出側と同じ `max_hold_steps`(1 回の帰還につき)。
    """
    if not enabled(sim) or not sim.envfbcfg["transit"]["enabled"]:
        return False
    prev_wait = int(getattr(agent, "_env_ret_wait", -1))
    if prev_wait > step:
        return True
    if prev_wait >= 0 and step > prev_wait:
        agent._env_ret_wait = -1                        # 別の帰還トリップ=カウンタを畳む
        agent._env_ret_held = 0
    st = state(sim)
    budget = (int(sim.envfbcfg["transit"]["max_hold_steps"])
              - int(getattr(agent, "_env_ret_held", 0)))
    wait = min(_delay_steps(sim, st), max(0, budget))
    if wait <= 0:
        agent._env_ret_wait = -1
        agent._env_ret_held = 0
        return False
    agent._env_ret_wait = step + wait
    agent._env_ret_held = int(getattr(agent, "_env_ret_held", 0)) + wait
    st["n_wait"] += 1
    _log(sim, step, sim_min, agent.id,
         {"rule": RULE_TRANSIT, "effect": "return_wait", "wait_steps": wait,
          "delay_min": round(float(st["delay_min"]), 3)},
         x=agent.x, y=agent.y)
    return True


def pending_exits(sim, step: int) -> tuple:
    """待たされて駅に留まっている個体のうち、この step に再試行してよい者。

    `_try_exit` は「到着した step」にしか呼ばれないので、待たせた個体には再試行の口が要る。
    OFF では空 tuple(呼び出し側のループが 0 回 = バイト一致)。
    """
    if not enabled(sim):
        return ()
    station = _station(sim)
    if station is None:
        return ()
    out = []
    for agent in sim.agents:
        if (agent.loc == "street" and agent.exit_intent and not agent.sleeping
                and not agent.route and agent.node == station
                and int(getattr(agent, "_env_exit_held", 0)) > 0
                and int(getattr(agent, "_env_exit_wait", -1)) <= step):
            out.append(agent)
    return tuple(out)


def _felt(sim, agent, wait: int, step: int, sim_min: int) -> None:
    """待たされたことを **既存語彙**(混雑)でだけ体感させる(no-fingerprint)。

    新しい記憶文・新しいプロンプト欄・新しい理由キーを 1 つも足さない。街路の混雑減速と
    同じ hook(`factors.update.on_congestion`)へ不透明な factor を渡すだけ。
    ★drive(発火系)には接続しない = LLM 呼数を 1 本も動かさない(disaster 層と同じ規約)。
    """
    cap = max(1, int(sim.envfbcfg["transit"]["max_hold_steps"]))
    factor = max(0.3, 1.0 - float(wait) / float(cap))
    factor_update.on_congestion(agent, factor, mags=sim.mags, step=step,
                                sim_min=sim_min, logger=sim.logger)


# --------------------------------------------------------------------------- #
# 作用点(3): 混雑した POI を行き先候補から一時的に外す
# --------------------------------------------------------------------------- #
def blocked_nodes(sim) -> frozenset:
    """いま待ち行列で塞がっている POI ノード(OFF は空 = 呼び出し側が素通り)。"""
    if not enabled(sim) or not sim.envfbcfg["poi"]["enabled"]:
        return frozenset()
    st = getattr(sim, "_envfb", None)
    if not st or not st["poi_hold"]:
        return frozenset()
    return frozenset(st["poi_hold"])


# --------------------------------------------------------------------------- #
# 観測(第80 チャンネル ext.transit_delay への接続)
# --------------------------------------------------------------------------- #
def delay_flag(sim) -> float:
    """『鉄道が乱れているか』(1/0)。OFF では常に 0.0 = 第80 の既存挙動と同一。

    第80 の `ext.transit_delay` は `transit.suspended`(災害の運休)しか見ていなかったため、
    災害 OFF のランでは定数チャンネル(σ=0)だった。本規則が入って初めて非ゼロの分散を持つ。
    """
    if not enabled(sim) or not sim.envfbcfg["transit"]["enabled"]:
        return 0.0
    st = getattr(sim, "_envfb", None)
    if not st:
        return 0.0
    return 1.0 if float(st["delay_min"]) >= float(sim.envfbcfg["transit"]["flag_min"]) else 0.0


def delay_min(sim) -> float:
    """いまの遅延[分](観測・テスト用)。OFF は 0.0。"""
    if not enabled(sim):
        return 0.0
    st = getattr(sim, "_envfb", None)
    return float(st["delay_min"]) if st else 0.0


def gate_active(sim, step: int) -> bool:
    """入場規制が発動中か(観測・テスト用)。OFF は False。"""
    if not enabled(sim) or not sim.envfbcfg["gate"]["enabled"]:
        return False
    st = getattr(sim, "_envfb", None)
    return bool(st and int(st["gate_until"]) > int(step))
