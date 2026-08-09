"""ラッシュ時の**車内** = シミュレートされた空間(``transit_interior``・**既定 OFF**)。

正典
----
- Wave 4 II-1「ラッシュ時の車内を空間にする」。
- 監査事実(``transit_staff.py`` の docstring 冒頭):「**電車は実体ではない**。
  ``world/transit.py`` の ``has_service()`` は時刻表の述語で、列車オブジェクトも
  運転士も車掌も世界に存在しない」。駅からの流入は今まで**瞬間移動**だった
  (``loc="outside"`` → ``enter_area`` の 1 step)。
- Wave 3 ``world.inflow_pulse``(``engine/simulation._snap_agent_arrival``): 駅
  ゲートウェイの流入通勤者の到着時刻を**その人の路線の時刻表へスナップ**した。
  結果として ``(pulse_line, pulse_train_min)`` が一致する個体の集合が
  **「同じ 1 本の列車の乗客」**になっている。本 module はその集合に**車両と区画**を
  与えるだけで、新しい集団の作り方は 1 つも発明していない。

★**前提**: ``world.inflow_pulse.enabled`` が ON でなければ列車の集団が存在しない
  (全員が別々の分に着くので (路線, 到着分) の群が立たない)。したがって
  ``enabled(sim)`` は **2 つの ON の論理積**である(``transit_staff.dwell_loop_active`` が
  ``env.feedback`` の ON を要求するのと同じ流儀 = 誰も読まない数字を書かない)。

何を解く問題か
--------------
渋谷に着く人の大半は**十数分間、他人と身体的に密着していた**直後に街へ出てくる。
その十数分は本シムに存在せず、混雑率 167%(埼京線)の車内で立っていた人と、
空いた車内で座っていた人が、まったく同じ状態で改札を出ていた。本 module は
その空白に「どの車両の・どの区画に・座っていたか立っていたか」を与える。

★**会話は増やさない**(これが本 module の最も重要な設計判断)
--------------------------------------------------------
車内は **QUIET CO-PRESENCE(静かな同席)**である。実測の要点:

  - 車内行動はスマホ 70〜84% / うたた寝(着席時)/ 読書 がほぼ全部を占める。
  - **見知らぬ他人との会話はほぼゼロ**。それどころか「車内の迷惑行為」で
    **騒々しい会話が第 1 位(51.8%)**に挙がる = 話しかけない規範が強い。
  - 会話が起きるのは**元から知り合い/同行者**の間だけ。

したがって本 module は **会話経路を 1 本も作らない**。``conversation`` /
``rumors.on_talk`` / ``_phase_drive`` のいずれにも触れず、``generate()`` の
呼び出しサイトを 1 つも足さない(= k 非依存)。同席は L1 の
``train_copresence`` に**記録するだけ**で、そこから何かが起きるかどうかは
既存の関係機構(friends / relations)が街の中で決める。これは「familiar strangers
(顔見知りの他人)は通勤の規則性から自然に生まれる」という PNAS 2013 の枠組みを
そのまま採るということで、**こちらから出会わせない**。

正直な設計の単純化(何を捨てたかを明記する)
--------------------------------------------
1. **車両選択の logit は「決定論の argmax」に落とした**(logit-lite)。
   文献の効用は (ホーム進入階段からの距離) + (降車駅の出口位置・UCL の推定で
   重み ~0.44) + (混雑・負) の 3 項だが、本シムでは
     - **降車駅の出口位置の項は恒等的に消える**。地図の駅は **1 ノード**しかなく
       (``transit_staff`` の限界 1)、全員が同じ改札から出るので個体差にならない。
       消えるものを重み 0.44 で残すのは嘘なので、**項ごと持たない**。
     - **乗車駅の階段位置は観測不能**。個体の出発駅を世界が持っていないため、
       安定ハッシュ(seed, agent id)の一様値を「編成方向の好みの位置」の代理に置く。
       これは較正された量ではなく**個体差を持たせるための代理**である。
     - 残ったのは (好みの位置との距離) と (混雑・負) の 2 項で、Gumbel 誤差項は
       置かない(= 乱数ゼロ)。したがって本 module が再現するのは
       **「人は編成の中でばらけ、混んだ車両を避ける」という平均的な形**だけで、
       選択確率の分布そのものではない。
2. **背景乗客は人ではなく「数」である**(virtual background load)。
   渋谷の乗降の大半は乗換・通過であって地図に入って来ない。その人々を
   エージェントにすると 25 万体の外に更に数十万体が要る。代わりに
   **車両ごとの人数(int)**として持ち、区画の占有率と密度にだけ効かせる。
   背景乗客は L1 に id を持たず・会話せず・記憶を持たず・同席の相手にもならない。
   これは「居ることにする」のではなく「**混雑という物理量の出どころを正直に
   数として置く**」ということである。
3. **車両ごとの混雑差は作らない**。実際の通勤電車は階段位置の影響で先頭/最後尾に
   偏るが、渋谷駅の各路線のホーム階段位置という**データを持っていない**ので、
   目標乗車人数は全車均等に置く(捏造しない)。ばらけるのは (1) の代理選好の
   ぶんだけである。
4. **座席は乗車率からの抽選**。「長く乗っている人ほど座っている」が現実だが、
   個体の乗車駅を持たないので**原理的に知り得ない**。そこで
   「座れる確率 = 座席数 / 乗車人数(1 でクリップ)」の安定ハッシュ抽選に落とす
   (乱数 stream は 1 本も引かない)。混雑率 100% を超える帯では
   ほぼ全員が立つ = 現実と同じ向きに倒れる。
5. **1 本の列車に車掌は 1 人しか居ない、という制約を置けない**。乗務員は駅ノードに
   束ねられた個体(``transit_staff``)で、列車に紐づいていない。同 step に複数の
   列車が着くときは当直の乗務員を**巡回のように割り当てる**(名簿順の剰余)ので、
   1 人が複数の列車を受け持つ形になる。列車エンティティを作るまでこれは直らない。

R1 ドクトリン
-------------
- 既定 ``enabled: false``(かつ ``world.inflow_pulse`` OFF でも)では ``phase`` が
  即 return し、**L1 に 1 件も出ず・agent に属性が生えず・sim に state が生えず・
  プロンプトが 1 バイトも変わらない**(= ゴールデン L1 バイト一致)。
- **乱数 stream を 1 本も引かない**(車両も区画も座席も行動も安定ハッシュの純関数)。
- **``generate()`` の呼び出しサイトを 1 つも作らない**(LLM 追加呼ゼロ = k 非依存)。
  乗車中の個体は ``loc == "outside"`` なので、そもそも
  ``scheduler.run_step`` の ``active`` 母集団に入らない(構造的に思考しない)。
- プロンプトへ入るのは**記憶 1 行だけ**で、それも ``plan_boundary.memo_line`` と
  同じ「圧縮記憶」の口(``agent.remember``)を使う = 新しい欄も契約も増やさない。
  文面は (路線名, 混雑段, 行動) だけの純関数で、数字・実験条件・機構語を持たない。

resume の設計(``engine/checkpoint.py`` に 1 行も足さずに resume == straight)
----------------------------------------------------------------------------
本 module は **checkpoint に載せるべき状態を 1 つも持たない**。世界に効く状態は
すべて agent の属性に置き、agents は既存の checkpoint が丸ごと pickle するからである:

  ``_train_ride`` / ``train_car`` / ``train_zone`` / ``train_seated``
      … 乗車 1 回ぶん(車両・区画・着席・行動・記憶を出したか・同席名簿)
  ``_train_seen`` / ``_train_seen_day``
      … 同席の重複判定**と 1 日の記録予算**(★グローバルな日次カウンタを作れば
        checkpoint が要る。そこで上限を「1 日 1 人あたり」= 個体側の予算に置いた。
        総量の硬い上限は ``n_agents × max_pairs_per_day`` で、conf から読める)

sim 側の ``_train_state`` は**観測タリーと到着表キャッシュだけ**で、L1/L2/L3・
プロンプト・乱数のどれにも現れない(resume で数え直しになっても世界は 1 ビットも
動かない)。

正直な限界(5 件)
------------------
1. **乗車は「圏外に居る最後の数 step」でしかない**。車内に空間座標は無く、
   ``x, y`` は縁ゲートウェイのまま動かない(可視化には出ない)。
2. **降車後の混雑は既存のホーム負荷が担う**。車内の密度は ``envfeedback`` の
   ホーム密度とは別の量で、二重計上しないよう既定では停車時間へ**足さない**
   (``conductor.dwell_load_weight: 0.0``)。
3. **乗換・通過客は数でしかない**ので、「同じ電車に毎日乗っている顔見知り」は
   地図に入る個体どうしでしか生まれない(現実より少なく出る = 保守側)。
4. **顔見知りが決定論的に決まりすぎる**(実測で判った)。到着分は名簿の
   ``arrival_min`` から量子化された固定値、車両の好みも (seed, agent id) の固定値
   なので、**同じ人は毎日必ず同じ人と同じ車両に乗る**。現実の familiar strangers は
   確率的な規則性から生まれるので、ここは規則性が強すぎる。日ごとの揺らぎを
   入れる余地は ``car_choice`` にあるが、**入れていない**(車両選択の習慣性は
   実測でも高く、揺らぎの大きさに根拠が無いため。入れるなら較正が要る)。
5. **``summary.json`` に鍵を出さない**。観測タリーは ``provenance(sim)`` として
   読める形にしてあるが、``engine/simulation.py`` の summary 組み立ては本レーンの
   所有外なので配線していない(1 日の総量は L1 の ``train_ride`` /
   ``train_copresence`` / ``train_patrol`` を数えれば出る)。
"""
from __future__ import annotations

import hashlib

from .observer.schema import Event, register_event_kind

SCHEMA = 1

# --------------------------------------------------------------------------- #
# L1 イベント種の**材料側**登録(devices.py / traces.py / transit_staff.py と同じ流儀。
# observer/schema.py には 1 バイトも書かない)。
# --------------------------------------------------------------------------- #
register_event_kind(
    "train_ride",
    "車内の 1 乗車(降車の step に 1 件)。agent_id = 乗客"
    " {line, train_min, car, cars, zone, seated, load, capacity, load_factor,"
    " density, act, steps}")
register_event_kind(
    "train_copresence",
    "同じ車両に居合わせた 1 対(1 日 1 対 1 件・上限あり)。agent_id = 小さい方の id"
    " {line, train_min, car, other_id, zone_adjacent, crew}")
register_event_kind(
    "train_patrol",
    "車掌の車内巡回(受け持ち列車の車両を 1 つずつ進む)。agent_id = 乗務員"
    " {line, train_min, car, cars, n_agents, n_pax}")

# --------------------------------------------------------------------------- #
# 区画(zone)の語彙。**20 m 車も 16 m 車も同じ 3 種**で、数だけがドア数で決まる。
#   seat  … 着席(立位密度から**除外**される)
#   door  … ドア脇の吹き溜まり(立客の ~49%)
#   front … 座席前の帯(~31%)
#   aisle … 中央通路(最後に埋まる。**溢れ先**でもある = 混雑率 250% の押し込み)
# --------------------------------------------------------------------------- #
SEAT = "seat"
DOOR = "door"
FRONT = "front"
AISLE = "aisle"
ZONE_KINDS: tuple[str, ...] = (SEAT, DOOR, FRONT, AISLE)

# 車内行動(**静かな同席**の 3 種)。会話はここに**無い**(語彙として存在させない)。
ACT_SMARTPHONE = "smartphone"
ACT_DOZE = "doze"
ACT_READ = "read"
ACTS: tuple[str, ...] = (ACT_SMARTPHONE, ACT_DOZE, ACT_READ)

# 記憶 1 行の定型語(**generic な日本語のみ**。地名も数字も実験条件も入らない)。
ACT_JA: dict[str, str] = {
    ACT_SMARTPHONE: "スマホを見て過ごした",
    ACT_DOZE: "うとうとしていた",
    ACT_READ: "本を読んでいた",
}
CROWD_JA: tuple[str, ...] = ("空いていた", "混んでいた", "ぎゅうぎゅうだった")
LINE_FALLBACK = "電車"                     # 路線が特定できない乗車(合併集合)の generic 語
MEMO_TEXT = "{line}の車内は{crowd}。{act}。"


# --------------------------------------------------------------------------- #
# 既定の車両諸元(**路線名を 1 つも持たない**)。
#
# ★路線ごとの実諸元と混雑率アンカーは **conf の ``transit_interior.lines``** に置く。
#   基盤(src)は地名・路線名のリテラルを持たない、という envpack ドクトリンに従う
#   (tests/test_contracts.py の地名禁止ガードと同じ線引き)。ここに残すのは
#   「20 m 4 扉のありふれた通勤形式」という**generic な既定**だけである。
# --------------------------------------------------------------------------- #
DEFAULT_SPEC: dict = {
    "cars": 10,            # 編成両数
    "car_len_m": 20.0,     # 1 両の長さ [m]
    "capacity": 150,       # 1 両の定員 [人](★物理的上限ではない = 混雑率 250% まで乗る)
    "seats": 51,           # 1 両の座席数
    "doors": 4,            # 片側のドア数(= ドア脇区画の数)
    "congestion": 1.30,    # 混雑率アンカー(ピーク時・最混雑区間)
}
_SPEC_INT_KEYS = ("cars", "capacity", "seats", "doors")


DEFAULTS: dict = {
    "enabled": False,
    # 乗車時間 [分]。降車(= 帰還 step)から遡ってこの長さだけ「車内に居る」ことにする。
    # ★step へ丸めるので実効は max(1, round(ride_minutes / Δt)) step。
    "ride_minutes": 15,
    # 計画駆動の圏外帰還者(planning.day_plan.boundary)も駅から帰るなら列車に乗せるか。
    # 帰還時刻そのものは**1 ビットも動かさない**(乗る列車の名前を後付けするだけ)。
    "include_plan_returnees": True,
    "default_line": dict(DEFAULT_SPEC),
    # 路線別の諸元表(データ = conf)。match は路線名の**部分一致**トークン。
    # 最長一致が勝ち、同点は宣言順。どれにも当たらない路線は default_line。
    "lines": [],
    # 時間帯の混雑倍率。windows = [開始分, 終了分, 倍率](分 of day・半開区間)。
    "congestion_scale": {"windows": [], "default": 1.0},
    # 立客の区画別シェア(door ≈ 49% / front ≈ 31% / aisle = 残り)。
    "zone_share": {DOOR: 0.49, FRONT: 0.31, AISLE: 0.20},
    # 車両選択(logit-lite = 決定論 argmax)の 2 項の重み。
    "car_choice": {"pos_weight": 1.0, "load_weight": 0.6},
    # 車内行動の分布(静かな同席)。doze は**着席時のみ**で、立位では他 2 種へ再配分。
    "activity": {ACT_SMARTPHONE: 0.77, ACT_DOZE: 0.13, ACT_READ: 0.10},
    "memory": {
        "enabled": True,
        # 混雑段の境目(乗車率 = 乗車人数/定員)。2 個 → 3 段(CROWD_JA と同数)。
        "crowd_bands": [0.9, 1.5],
    },
    "comfort": {
        # 局所密度がこの値を超えた分だけ疲労が乗る(Evans & Wener: 効くのは
        # **座席周りの局所密度**であって車両平均ではない)。
        "density_floor": 1.0,
        "fatigue_per_step": 0.02,   # 1 step・超過密度 1.0 あたりの疲労 [/step]
        "fatigue_max": 0.12,        # 1 乗車で乗せる疲労の上限(安全弁)
    },
    "copresence": {
        "enabled": True,
        "max_pairs_per_car": 24,    # 1 乗車 1 人あたりに控える同席相手の上限
        # ★1 日 1 人あたりに記録する同席の対の上限。**総量上限は個体側に置く**
        #   (= n_agents × この値 が 1 日の L1 の硬い上限)。グローバルなカウンタに
        #   しない理由は resume: sim 側の日次カウンタは checkpoint に載せられない
        #   (checkpoint.py は本レーンの所有外)ので、上限が binding なランで
        #   resume ≠ straight になってしまう。個体側なら agents pickle に自然同梱される。
        "max_pairs_per_day": 8,
        "adjacent_only": False,     # true = 同一/隣接区画の対だけを記録する
    },
    "conductor": {
        "enabled": True,
        "patrol_interval_steps": 1,  # 何 step ごとに 1 両進むか
        # 車内の立客数を停車時間の負荷へ足す重み。**既定 0.0 = 観測のみ**。
        # 理由: ホーム密度 → 停車時間の較正(+15 人/車両 ≒ +1 秒)は既に
        # envfeedback 規則1 が**ホーム側の人数**で持っている。車内の立客を素で
        # 足すと同じ物理を二重計上する。0 でも payload には出るので、
        # 「車掌が何を見ていたか」は台帳から読める。
        "dwell_load_weight": 0.0,
    },
}


# =========================================================================== #
# cfg 正準化(traces.build_cfg / transit_staff.build_cfg と同型)
# =========================================================================== #
def _to_plain(raw):
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(raw):
            return OmegaConf.to_container(raw, resolve=True)
    except Exception:                              # noqa: BLE001(旧 config 互換)
        pass
    return raw


def _spec(raw, base: dict) -> dict:
    """車両諸元 1 件の正準化(未知キーは捨て・非正の値は既定へ戻す)。"""
    got = dict(_to_plain(raw) or {})
    out = dict(base)
    for k in _SPEC_INT_KEYS:
        if k in got:
            try:
                v = int(got[k])
            except (TypeError, ValueError):
                continue
            if v > 0 or (k == "seats" and v >= 0):
                out[k] = v
    if "car_len_m" in got:
        try:
            v = float(got["car_len_m"])
            if v > 0:
                out["car_len_m"] = v
        except (TypeError, ValueError):
            pass
    if "congestion" in got:
        try:
            out["congestion"] = max(0.0, float(got["congestion"]))
        except (TypeError, ValueError):
            pass
    out["seats"] = min(int(out["seats"]), int(out["capacity"]))
    return out


def _windows(raw) -> list:
    """時間帯倍率の窓 [[開始分, 終了分, 倍率], ...] を正準化(壊れた行は捨てる)。"""
    out = []
    for row in (_to_plain(raw) or []):
        try:
            a, b, s = int(row[0]), int(row[1]), float(row[2])
        except (TypeError, ValueError, IndexError, KeyError):
            continue
        if 0 <= a < b <= 1440 and s >= 0.0:
            out.append([a, b, s])
    return out


def _shares(raw, base: dict) -> dict:
    """立客の区画別シェア(負は 0・総和 0 なら既定へ戻す)。**正規化はしない**
    = 総和が 1 未満なら公称容量が小さくなる(= 早く溢れる)という素直な意味を持つ。"""
    got = dict(_to_plain(raw) or {})
    out = dict(base)
    for k in (DOOR, FRONT, AISLE):
        if k in got:
            try:
                out[k] = max(0.0, float(got[k]))
            except (TypeError, ValueError):
                pass
    return out if sum(out.values()) > 0.0 else dict(base)


def _acts(raw, base: dict) -> dict:
    got = dict(_to_plain(raw) or {})
    out = dict(base)
    for k in ACTS:
        if k in got:
            try:
                out[k] = max(0.0, float(got[k]))
            except (TypeError, ValueError):
                pass
    return out if sum(out.values()) > 0.0 else dict(base)


def _bands(raw) -> list:
    """混雑段の境目を正準化(昇順・段数は ``CROWD_JA`` の語数 − 1 で頭打ち)。

    語より多い境目を与えられても**語を捏造しない**(切り捨てる)。空・壊れた値は既定へ。
    """
    out = []
    for x in (_to_plain(raw) or []):
        try:
            out.append(float(x))
        except (TypeError, ValueError):
            continue
    out = sorted(out)[:len(CROWD_JA) - 1]
    return out or list(DEFAULTS["memory"]["crowd_bands"])


def _sub(raw, base: dict, bools=(), ints=(), floats=()) -> dict:
    got = dict(_to_plain(raw) or {})
    out = dict(base)
    for k in bools:
        if k in got:
            out[k] = bool(got[k])
    for k in ints:
        if k in got:
            try:
                out[k] = int(got[k])
            except (TypeError, ValueError):
                pass
    for k in floats:
        if k in got:
            try:
                out[k] = float(got[k])
            except (TypeError, ValueError):
                pass
    return out


def build_cfg(raw) -> dict:
    """conf の ``transit_interior`` ブロックを正準化(既定 OFF=現行挙動と完全同一)。

    dotlist 上書きは文字列で入り得るため型強制する(``traces.build_cfg`` と同じ作法)。
    未知のキーは黙って捨てる(捏造しない)。
    """
    raw = dict(_to_plain(raw) or {})
    cfg: dict = {
        "enabled": bool(raw.get("enabled", False)),
        "ride_minutes": DEFAULTS["ride_minutes"],
        "include_plan_returnees": bool(raw.get("include_plan_returnees",
                                               DEFAULTS["include_plan_returnees"])),
        "default_line": _spec(raw.get("default_line"), DEFAULT_SPEC),
        "lines": [],
        "congestion_scale": {"windows": [], "default": 1.0},
        "zone_share": _shares(raw.get("zone_share"), DEFAULTS["zone_share"]),
        "car_choice": _sub(raw.get("car_choice"), DEFAULTS["car_choice"],
                           floats=("pos_weight", "load_weight")),
        "activity": _acts(raw.get("activity"), DEFAULTS["activity"]),
        "memory": _sub(raw.get("memory"), DEFAULTS["memory"], bools=("enabled",)),
        "comfort": _sub(raw.get("comfort"), DEFAULTS["comfort"],
                        floats=("density_floor", "fatigue_per_step", "fatigue_max")),
        "copresence": _sub(raw.get("copresence"), DEFAULTS["copresence"],
                           bools=("enabled", "adjacent_only"),
                           ints=("max_pairs_per_car", "max_pairs_per_day")),
        "conductor": _sub(raw.get("conductor"), DEFAULTS["conductor"],
                          bools=("enabled",), ints=("patrol_interval_steps",),
                          floats=("dwell_load_weight",)),
    }
    if "ride_minutes" in raw:
        try:
            cfg["ride_minutes"] = max(1, int(raw["ride_minutes"]))
        except (TypeError, ValueError):
            pass
    # ---- 路線別諸元表(match トークンつき)。match が空の行は捨てる ---------------- #
    for row in (_to_plain(raw.get("lines")) or []):
        row = dict(_to_plain(row) or {})
        token = str(row.get("match", "") or "").strip()
        if not token:
            continue
        item = _spec(row, cfg["default_line"])
        item["match"] = token
        cfg["lines"].append(item)
    got = dict(_to_plain(raw.get("congestion_scale")) or {})
    cfg["congestion_scale"]["windows"] = _windows(got.get("windows"))
    try:
        cfg["congestion_scale"]["default"] = max(0.0, float(got.get("default", 1.0)))
    except (TypeError, ValueError):
        pass
    # 上限・境目の健全化(壊れた conf で無限ループ/負の容量が出ないようにする)
    cfg["memory"]["crowd_bands"] = _bands(
        dict(_to_plain(raw.get("memory")) or {}).get("crowd_bands"))
    cfg["copresence"]["max_pairs_per_car"] = max(
        0, int(cfg["copresence"]["max_pairs_per_car"]))
    cfg["copresence"]["max_pairs_per_day"] = max(
        0, int(cfg["copresence"]["max_pairs_per_day"]))
    cfg["conductor"]["patrol_interval_steps"] = max(
        1, int(cfg["conductor"]["patrol_interval_steps"]))
    cfg["conductor"]["dwell_load_weight"] = max(
        0.0, float(cfg["conductor"]["dwell_load_weight"]))
    for k in ("density_floor", "fatigue_per_step", "fatigue_max"):
        cfg["comfort"][k] = max(0.0, float(cfg["comfort"][k]))
    for k in ("pos_weight", "load_weight"):
        cfg["car_choice"][k] = max(0.0, float(cfg["car_choice"][k]))
    return cfg


def cfg_of(sim) -> dict:
    """車内層の設定(初回のみ ``sim.cfg.transit_interior`` から遅延構築してキャッシュ)。

    ``simulation.py`` は読み取り専用なので cfg は本 module が遅延構築する
    (``traces.cfg_of`` / ``transit_staff.cfg_of`` と同型)。キャッシュ属性
    ``sim.transitinteriorcfg`` は L1/L2/L3/乱数に一切現れない。
    """
    got = getattr(sim, "transitinteriorcfg", None)
    if got is None:
        try:
            raw = sim.cfg.get("transit_interior", None)
        except Exception:                          # noqa: BLE001(旧 config 互換)
            raw = None
        got = build_cfg(raw)
        sim.transitinteriorcfg = got
    return got


def pulse_on(sim) -> bool:
    """``world.inflow_pulse`` が ON か(= 列車ごとの塊が存在するか)。**読むだけ**。"""
    try:
        return bool((sim.cfg.world.get("inflow_pulse", {}) or {}).get("enabled", False))
    except Exception:                              # noqa: BLE001(旧 config 互換)
        return False


def enabled(sim) -> bool:
    """車内層が**実効**か = 自分の ON かつ ``world.inflow_pulse`` の ON。

    ★パルス量子化が OFF のときに本層だけ ON にしても、(路線, 到着分)の群が
      1 つも立たない(全員が別々の分に着く)= 車両に乗客が 1 人ずつしか居ない
      「列車の形をしていない列車」を作ることになる。誰も読まない数字を書かない、
      という ``transit_staff.dwell_loop_active`` の線引きに揃えて黙って走らせない。
    """
    return bool(cfg_of(sim)["enabled"]) and pulse_on(sim)


# =========================================================================== #
# 純関数(sim を見ない = 単体でテストできる層)
# =========================================================================== #
def stable_uniform(seed: int, key: str) -> float:
    """(seed, 安定キー)から一様値 [0,1)。``plan_boundary.stable_uniform`` と同流儀。

    hashlib = プロセス跨ぎで安定 = リプレイ・resume・別ランで同一値。**乱数
    ストリームを 1 本も引かない**ので、既存機構の draw 順は 1 本もずれない。
    """
    h = hashlib.blake2b(f"{int(seed)}\x1f{key}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") / float(1 << 64)


def spec_for(cfg: dict, line: str) -> dict:
    """路線名 → 車両諸元(**最長 match 勝ち**・同点は宣言順・非該当は default_line)。

    コード側は路線名のリテラルを 1 つも持たない: 比較するのは conf の match トークンと
    ダイヤ(データ)の路線名だけである(envpack ドクトリン)。
    """
    text = str(line or "")
    best = None
    best_n = 0
    for item in cfg["lines"]:
        token = str(item["match"])
        if token and token in text and len(token) > best_n:
            best, best_n = item, len(token)
    return dict(best or cfg["default_line"])


def scale_at(cfg: dict, min_of_day: int) -> float:
    """時間帯の混雑倍率(最初に当たった窓が勝ち・非該当は default)。純関数。"""
    m = int(min_of_day) % 1440
    for a, b, s in cfg["congestion_scale"]["windows"]:
        if a <= m < b:
            return float(s)
    return float(cfg["congestion_scale"]["default"])


def car_load(cfg: dict, spec: dict, min_of_day: int) -> int:
    """1 両あたりの目標乗車人数 = 定員 × 混雑率アンカー × 時間帯倍率(全車均等)。

    ★定員は**物理的上限ではない**(混雑率 250% まで乗るのが東京の通勤電車)。
      したがってここでクリップしない = 立位区画は溢れて aisle が受ける。
    """
    return int(round(float(spec["capacity"]) * float(spec["congestion"])
                     * scale_at(cfg, min_of_day)))


def zone_ids(spec: dict) -> tuple[str, ...]:
    """1 両の立位区画 id を**充填順**で返す(ドア脇 → 座席前 → 通路)。

    数はドア数で決まる: 20 m 4 扉なら door0..3 / front0..3 / aisle、
    16 m 3 扉なら door0..2 / front0..2 / aisle。
    """
    n = max(1, int(spec["doors"]))
    return tuple([f"{DOOR}{i}" for i in range(n)]
                 + [f"{FRONT}{i}" for i in range(n)] + [AISLE])


def zone_caps(cfg: dict, spec: dict) -> dict[str, float]:
    """区画 id → 公称容量 [人]。立位定員(定員 − 座席)をシェアで割る。"""
    n = max(1, int(spec["doors"]))
    stand = max(1.0, float(spec["capacity"]) - float(spec["seats"]))
    share = cfg["zone_share"]
    out: dict[str, float] = {}
    for i in range(n):
        out[f"{DOOR}{i}"] = stand * float(share[DOOR]) / n
        out[f"{FRONT}{i}"] = stand * float(share[FRONT]) / n
    out[AISLE] = max(1e-9, stand * float(share[AISLE]))
    return out


def zone_kind(zone: str) -> str:
    """区画 id → 種別(seat / door / front / aisle)。純関数。"""
    z = str(zone or "")
    if z == SEAT:
        return SEAT
    if z == AISLE:
        return AISLE
    if z.startswith(DOOR):
        return DOOR
    if z.startswith(FRONT):
        return FRONT
    return AISLE


def zone_adjacent(a: str, b: str) -> bool:
    """2 つの区画が「同一 or 隣接」か = **相互作用の可能性がある距離**か。

    幾何(20 m 車の見取り図)をそのまま述語にしたもの:
      - 同一区画は当然隣接
      - ドア脇 i ↔ 座席前 i(ドアの正面が座席前の帯)
      - 通路は立位のどの区画とも隣接(車両の背骨)
      - 座席 ↔ 座席前(座っている人の目の前に立つ人が居る)
      - ドア脇 i ↔ ドア脇 j(i≠j)は**隣接しない**(車両の反対端)
    """
    a, b = str(a or ""), str(b or "")
    if a == b:
        return True
    ka, kb = zone_kind(a), zone_kind(b)
    if AISLE in (ka, kb):
        return SEAT not in (ka, kb)
    if {ka, kb} == {SEAT, FRONT}:
        return True
    if {ka, kb} == {DOOR, FRONT}:
        return a[len(DOOR):] == b[len(FRONT):] or a[len(FRONT):] == b[len(DOOR):]
    return False


def crowd_band(cfg: dict, load_factor: float) -> int:
    """乗車率 → 混雑段の index(0..len(CROWD_JA)-1)。純関数。"""
    bands = cfg["memory"]["crowd_bands"]
    i = 0
    for b in bands:
        if float(load_factor) >= float(b):
            i += 1
    return min(i, len(CROWD_JA) - 1)


def activity_of(cfg: dict, seed: int, aid: int, day: int, train_min: int,
                seated: bool) -> str:
    """車内行動(**静かな同席**の 3 種)の決定論抽選。**乱数 stream を引かない**。

    ★``doze``(うたた寝)は**着席時のみ**。立っている人の分は残り 2 種へ
      重み比で再配分する(立ったまま寝る、を作らない)。
    """
    w = dict(cfg["activity"])
    if not seated:
        w[ACT_DOZE] = 0.0
    total = sum(w[k] for k in ACTS)
    if total <= 0.0:
        return ACT_SMARTPHONE
    u = stable_uniform(seed, f"train_act/{int(aid)}/{int(day)}/{int(train_min)}") * total
    acc = 0.0
    for k in ACTS:
        acc += w[k]
        if u < acc:
            return k
    return ACT_SMARTPHONE


def memo_line(line: str, band: int, act: str) -> str:
    """乗車の圧縮記憶 1 行。**(路線名, 混雑段, 行動) だけの純関数**(LLM ゼロ・乱数ゼロ)。

    ★数字・実験条件・k・因子名・機構語を 1 バイトも含まない(no-fingerprint)。
      混雑は**段の語**でしか出さない(「167%」とは絶対に書かない = 個体は
      混雑率という統計量を知らない)。路線名はダイヤ(データ)から来る世界の事実で、
      コード側は路線名のリテラルを持たない。
    """
    name = str(line or "").strip() or LINE_FALLBACK
    i = min(max(int(band), 0), len(CROWD_JA) - 1)
    return MEMO_TEXT.format(line=name, crowd=CROWD_JA[i],
                            act=ACT_JA.get(str(act), ACT_JA[ACT_SMARTPHONE]))


def choose_car(cfg: dict, spec: dict, seed: int, aid: int,
               car_counts: dict[int, int], target: int) -> int:
    """車両の決定論選択(**logit-lite = 誤差項なしの argmax**)。

    効用 = −pos_weight × |編成上の位置 − その人の好みの位置|
           −load_weight × (その車両に既に居る**地図内の**乗客数 / 目標乗車人数)

    - 好みの位置 = 安定ハッシュの一様値(乗車駅の階段位置は世界が持っていないため
      の代理。module docstring の単純化 1)。
    - 混雑項の分母を「目標乗車人数」にすることで、**背景乗客のアンカーに対して**
      相対的に均す(= アンカーが高い路線ほど 1 人の偏りが効かない)。
    - 同点は**車両 index の小さい方**(完全決定論)。
    """
    cars = max(1, int(spec["cars"]))
    pref = stable_uniform(seed, f"train_car/{int(aid)}")
    wp = float(cfg["car_choice"]["pos_weight"])
    wl = float(cfg["car_choice"]["load_weight"])
    denom = float(max(1, int(target)))
    best_i, best_u = 0, None
    for c in range(cars):
        pos = (c + 0.5) / cars
        u = -wp * abs(pos - pref) - wl * (float(car_counts.get(c, 0)) / denom)
        if best_u is None or u > best_u:
            best_i, best_u = c, u
    return best_i


def seat_ratio(spec: dict, total_load: int) -> float:
    """1 人が座れる確率 = 座席数 / 乗車人数(1 でクリップ)。純関数。

    ★「長く乗っている人ほど座っている」は現実だが、個体の乗車駅を世界が持って
      いないので**原理的に知り得ない**(module docstring の単純化 4)。ここで
      置いたのは「同じ列車の全乗客(背景の数を含む)から座席を等確率で配る」
      という最も情報を足さない仮定である。
    """
    seats = float(spec["seats"])
    if seats <= 0.0:
        return 0.0
    return min(1.0, seats / float(max(1, int(total_load))))


# =========================================================================== #
# 状態
#
# ★**世界に効く状態は 1 バイトも sim 側に置かない**(これが本 module の resume 戦略)。
#   乗車 1 回ぶんも、同席の重複判定も、1 日の記録予算も、すべて **agent 側**にある
#   (``_train_ride`` / ``train_car`` / ``train_zone`` / ``train_seated`` /
#    ``_train_seen`` / ``_train_seen_day``)。agents は checkpoint に pickle されるので、
#   ``engine/checkpoint.py`` に 1 行も足さずに **resume == straight** が成り立つ。
#   sim 側の ``_train_state`` は **観測タリーと到着表キャッシュだけ**で、L1/L2/L3・
#   プロンプト・乱数のどれにも現れない(= 数え直しになっても世界は 1 ビットも動かない)。
# =========================================================================== #
def _state(sim) -> dict:
    st = getattr(sim, "_train_state", None)
    if st is None:
        st = {"schema": SCHEMA, "arrivals": None,
              "rides": 0, "riders": 0, "mem_lines": 0, "patrols": 0,
              "copresence": 0, "copresence_dropped": 0, "cars_used": {}}
        sim._train_state = st
    return st


def _seed_of(sim) -> int:
    try:
        return int(sim.cfg.run.seed)
    except Exception:                              # noqa: BLE001
        return 0


def _station(sim):
    city = getattr(sim, "city", None)
    return str(getattr(city, "station_node", "") or "") if city is not None else ""


def _arrivals(sim, st: dict) -> list:
    """定常日の到着表(**読むだけ**・1 ラン 1 回だけ組む)。計画帰還者のスナップ用。"""
    if st["arrivals"] is None:
        transit = getattr(sim, "transit", None)
        st["arrivals"] = list(transit.arrivals_of_day()) if transit is not None else []
    return st["arrivals"]


# =========================================================================== #
# (1) 誰が今この step「車内に居る」か
# =========================================================================== #
def ride_steps(sim, cfg: dict) -> int:
    """乗車の長さ [step]。実時間の分から Δt で丸める(最低 1 step)。"""
    dt = max(1, int(getattr(sim.clock, "step_minutes", 10)))
    return max(1, int(round(float(cfg["ride_minutes"]) / float(dt))))


def _riders(sim, cfg: dict, st: dict, step: int) -> list:
    """この step に車内に居る個体 ``[(agent, line, train_min, earliest_step)]``。

    乗車窓は **step の領域**で定義する(分ではない): 帰還予定 ``return_at`` から
    遡って ``ride_steps`` step が車内である。理由は 2 つ:
      - 到着は ``_phase_wake_and_returns`` が ``step >= return_at`` で起こすので、
        「まだ着いていない = 車内」の境目は step でしか一致しない。
      - 終電・遅延・改札待ちで帰還が保留されると ``return_at - step`` が負になる。
        そのとき窓から外れる = **乗車は終わっていて駅で待っている**という
        正直な読みになる(車内に留め置かない)。
    """
    station = _station(sim)
    if not station:
        return []
    span = ride_steps(sim, cfg)
    plan_ok = bool(cfg["include_plan_returnees"])
    out = []
    for a in sim.agents:
        if a.loc != "outside" or a.sleeping:
            continue
        delta = int(getattr(a, "return_at", 0)) - int(step)
        if delta <= 0 or delta > span:
            continue
        if str(getattr(a, "return_gateway", "") or "") != station:
            continue
        tmin = getattr(a, "pulse_train_min", None)
        if tmin is not None:
            out.append((a, str(getattr(a, "pulse_line", "") or ""), int(tmin), 0))
            continue
        if not plan_ok:
            continue
        # 計画駆動の圏外帰還者(planning.day_plan.boundary)。帰還時刻は**1 ビットも
        # 動かさず**、その時刻に着く列車の名前だけを後付けする(読むだけ)。
        p = getattr(a, "_boundary_pending", None)
        if not isinstance(p, dict) or str(p.get("gateway") or "") != station:
            continue
        arr = _snap(_arrivals(sim, st),
                    int(sim.clock.sim_min(int(a.return_at))) % 1440)
        if arr is None:
            continue
        # 「計画で clamp」= 退出より前には遡らない(退出していない列車には乗れない)
        out.append((a, str(arr[1]), int(arr[0]), int(p.get("exit_step", -1)) + 1))
    return [(a, ln, tm, e) for (a, ln, tm, e) in out if int(step) >= int(e)]


def _snap(table: list, min_of_day: int):
    """到着表(昇順)から「この分以上で最も早い到着」= ``world/transit`` の純関数。

    実装は 1 つだけ持つ(``world.transit.snap_to_arrival_of_day``)。ここは呼ぶだけで、
    表は本 module 側が 1 ラン 1 回だけ組んでキャッシュする(``Transit.arrival_at_or_after``
    は毎回表を組み直すので毎 step の経路では使わない)。
    """
    from .world.transit import snap_to_arrival_of_day
    return snap_to_arrival_of_day(table, min_of_day)


# =========================================================================== #
# (2) 乗車の開始 = 車両と区画の割当(T1 + T2)
# =========================================================================== #
def _ride_key(line: str, train_min: int) -> str:
    """乗車 1 回の同一性キー = **(路線, 到着分) だけ**。日は入れない。

    ★日を入れると、終電のように**日を跨ぐ乗車**で key が途中で変わり、同じ乗車が
      「未割当」に見えて車両を割り直し・記憶をもう 1 行入れてしまう(1 乗車 1 行の破れ)。
      降車(``_end``)が必ず ``_train_ride`` を消すので、属性が生きている = いま乗車中
      であり、日を混ぜなくても同一性は一意に決まる。
    """
    return f"{line}|{int(train_min)}"


def _assign(sim, cfg: dict, st: dict, line: str, train_min: int, members: list,
            step: int, sim_min: int) -> None:
    """1 本の列車(= (路線, 到着分) の群)の未割当者に車両と区画を与える。

    順序: **agent id 昇順**(乱数ゼロ・到着順に依存しない)。既に割当済みの個体は
    1 バイトも触らない(resume で戻って来た個体を作り直さない = 冪等)。
    """
    day = int(sim_min) // 1440
    key = _ride_key(line, train_min)
    spec = spec_for(cfg, line)
    target = car_load(cfg, spec, train_min)
    seed = _seed_of(sim)
    cars = max(1, int(spec["cars"]))
    caps = zone_caps(cfg, spec)
    order = zone_ids(spec)
    seats = int(spec["seats"])
    # ---- 背景乗客(**人ではなく数**)の座席・立位分布 ---------------------------- #
    seated_total = min(seats, target)
    standing_total = max(0, target - seated_total)
    cap_sum = sum(caps[z] for z in order)
    virt_zone = {z: (standing_total * caps[z] / cap_sum if cap_sum > 0 else 0.0)
                 for z in order}
    # ---- 既割当の地図内乗客(この step までに乗った者)を数える -------------------- #
    car_counts: dict[int, int] = {}
    seat_counts: dict[int, int] = {}
    zone_counts: dict[tuple[int, str], int] = {}
    todo = []
    for a in members:
        r = getattr(a, "_train_ride", None)
        if isinstance(r, dict) and str(r.get("key")) == key:
            c = int(r["car"])
            car_counts[c] = car_counts.get(c, 0) + 1
            if bool(r["seated"]):
                seat_counts[c] = seat_counts.get(c, 0) + 1
            else:
                zk = (c, str(r["zone"]))
                zone_counts[zk] = zone_counts.get(zk, 0) + 1
        else:
            todo.append(a)
    ratio = seat_ratio(spec, target)
    for a in sorted(todo, key=lambda x: int(x.id)):
        aid = int(a.id)
        car = choose_car(cfg, spec, seed, aid, car_counts, target)
        car_counts[car] = car_counts.get(car, 0) + 1
        # ---- 座席の抽選(安定ハッシュ)------------------------------------------ #
        # 上限は「その車両で座っている**総人数**」= min(座席数, 乗車人数)。地図内の
        # 乗客がそれを超えて座ることはできない(背景の数は座席の**中**に居る)。
        seated = (seat_counts.get(car, 0) < seated_total
                  and stable_uniform(seed, f"train_seat/{aid}/{day}/{int(train_min)}")
                  < ratio)
        if seated:
            seat_counts[car] = seat_counts.get(car, 0) + 1
            zone, density = SEAT, 0.0
        else:
            zone, density = _place(
                order, caps, virt_zone, zone_counts, car,
                stable_uniform(seed, f"train_zone/{aid}/{day}/{int(train_min)}"))
            zone_counts[(car, zone)] = zone_counts.get((car, zone), 0) + 1
        act = activity_of(cfg, seed, aid, day, int(train_min), seated)
        band = crowd_band(cfg, float(target) / max(1.0, float(spec["capacity"])))
        a._train_ride = {"key": key, "line": str(line), "train_min": int(train_min),
                         "car": int(car), "cars": cars, "zone": str(zone),
                         "seated": bool(seated), "density": round(float(density), 4),
                         "load": int(target), "capacity": int(spec["capacity"]),
                         "act": str(act), "band": int(band), "start": int(step),
                         "fatigue": 0.0, "mates": []}
        # 公開属性(**ON のときだけ生える**= 属性不在が OFF の表明)
        a.train_car, a.train_zone, a.train_seated = int(car), str(zone), bool(seated)
        st["rides"] += 1
        st["cars_used"][str(car)] = int(st["cars_used"].get(str(car), 0)) + 1
        if cfg["memory"]["enabled"]:
            a.remember(memo_line(line, band, act))
            st["mem_lines"] += 1


def _place(order, caps, virt_zone, zone_counts, car: int, u: float):
    """立客 1 人を区画へ置く。返り値 = (区画 id, **局所**密度)。``u`` = [0,1) の安定一様値。

    規則は 2 段で、**混雑の帯によって支配する物理が違う**という 1 点だけを言っている:

      **① 立位定員に余裕があるうちは充填順**(ドア脇 → 座席前 → **通路は最後**)。
         「立客はまずドア脇の吹き溜まりに溜まり、中央通路は最後に埋まる」という
         観察をそのまま述語にしたもの。空いた車内では**位置は選べる**。

      **② どの立位区画も公称を超えたら(= 混雑率 100% 超の押し込み帯)、
         各区画の立位面積シェアに比例した抽選**(通路も候補に戻る)。
         この帯では位置はもう選べず、どこに押し込まれるかは実質くじである。
         定常配分は公称容量比 = 実測の立客シェア(ドア脇 ~49% / 座席前 ~31% /
         通路 ~20%)に一致する。
         ★ここを「占有率が最小の区画」にしてはいけない(初版の誤り): 1 人足したときの
           比の増分は**面積が大きい区画ほど小さい**ので、乗客が少ないと全員が
           中央通路へ吸い寄せられる(実測でそうなった)。抽選なら 1 人でも
           シェアどおりに散る。

    局所密度 = (背景の数 + 地図内の人数 + 自分) / 公称容量。
    ★**着席者は立位密度に 1 人も数えない**(座席は立位定員の外側の量)。
    """
    occ = {z: (float(virt_zone.get(z, 0.0))
               + float(zone_counts.get((car, z), 0))) for z in order}
    for z in order:
        if z != AISLE and occ[z] + 1.0 <= float(caps[z]):
            return z, (occ[z] + 1.0) / max(1e-9, float(caps[z]))
    total = sum(float(caps[z]) for z in order)
    acc, pick = 0.0, order[-1]
    for z in order:
        acc += float(caps[z])
        if float(u) * total < acc:
            pick = z
            break
    return pick, (occ[pick] + 1.0) / max(1e-9, float(caps[pick]))


# =========================================================================== #
# (3) 乗車中の効果(局所密度 → 疲労)。**会話は 1 本も作らない**
# =========================================================================== #
def _tick(sim, cfg: dict, agent) -> None:
    """局所密度の超過ぶんだけ疲労を乗せる(Evans & Wener: 効くのは**局所**密度)。

    ★``drive``(欲求ゲージ)には**絶対に触らない**: あそこを動かすと LLM の
      発火数が変わる = k 依存になる(registry の affects_k の定義そのもの)。
      触るのは ``fatigue`` だけで、これは既存の健康層(``health.tick_fatigue``)と
      同じ内部 transient(states 監査集合の外 = R²(k) を汚さない)。
    """
    r = getattr(agent, "_train_ride", None)
    if not isinstance(r, dict):
        return
    c = cfg["comfort"]
    over = float(r["density"]) - float(c["density_floor"])
    if over <= 0.0:
        return
    room = float(c["fatigue_max"]) - float(r["fatigue"])
    if room <= 0.0:
        return
    delta = min(room, float(c["fatigue_per_step"]) * over)
    if delta <= 0.0:
        return
    r["fatigue"] = float(r["fatigue"]) + delta
    agent.fatigue = min(1.0, float(getattr(agent, "fatigue", 0.0)) + delta)


def _mates(cfg: dict, members: list) -> None:
    """同じ車両の相手を各自の乗車記録へ控える(降車時に同席を出すための名簿)。

    ★上限 ``max_pairs_per_car`` で切る(1 両に 100 人居ても L1 は爆発しない)。
      切る順は **agent id 昇順**(決定論)。
    """
    cap = int(cfg["copresence"]["max_pairs_per_car"])
    if cap <= 0:
        return
    by_car: dict[int, list] = {}
    for a in members:
        r = getattr(a, "_train_ride", None)
        if isinstance(r, dict):
            by_car.setdefault(int(r["car"]), []).append(a)
    for car in sorted(by_car):
        group = sorted(by_car[car], key=lambda x: int(x.id))
        for a in group:
            r = a._train_ride
            seen = {int(m[0]) for m in r["mates"]}
            for b in group:
                if int(b.id) == int(a.id) or int(b.id) in seen:
                    continue
                if len(r["mates"]) >= cap:
                    break
                r["mates"].append((int(b.id), str(b._train_ride["zone"])))


# =========================================================================== #
# (4) 同席の記録(familiar strangers の**素材**。ここから何も起こさない)
# =========================================================================== #
def _seen_reset(agent, day: int) -> set:
    if int(getattr(agent, "_train_seen_day", -1)) != int(day):
        agent._train_seen_day = int(day)
        agent._train_seen = set()
    return agent._train_seen


def _copresence(sim, cfg: dict, st: dict, a, b, line: str, train_min: int,
                car: int, zone_a: str, zone_b: str, crew: bool,
                step: int, sim_min: int) -> None:
    """1 対の同席を L1 に 1 件(**1 日 1 対 1 件**・1 日 1 人あたりの上限つき)。

    ★重複判定も 1 日の予算も**個体側**(``_train_seen``)に持つ = agents pickle に
      自然同梱されるので resume を跨いで「今日はもう出した / もう予算が無い」が保たれる
      (= L1 が resume で分岐しない)。グローバルな日次カウンタは checkpoint に
      載せられないので**作らない**。
    ★総量の硬い上限は ``n_agents × max_pairs_per_day``。片側が予算切れなら出さない
      (両側の予算を見る = 対称 = 降車の順序に依存しない)。
    """
    ccfg = cfg["copresence"]
    if not ccfg["enabled"]:
        return
    adj = zone_adjacent(zone_a, zone_b)
    if ccfg["adjacent_only"] and not adj:
        return
    day = int(sim_min) // 1440
    lo, hi = (a, b) if int(a.id) <= int(b.id) else (b, a)
    seen_lo, seen_hi = _seen_reset(lo, day), _seen_reset(hi, day)
    if int(hi.id) in seen_lo:
        return
    # ★重複判定の印は**出しても捨てても付ける**: 予算切れの対を印なしで返すと、
    #   同じ対を相手側の降車でもう一度試すことになり、``copresence_dropped`` が
    #   「捨てた**対**の数」ではなく「試した回数」になって意味が崩れる(実測で 2 倍に
    #   膨らんだ)。印を付けておけば、この数はそのまま「今日 L1 に載らなかった対の数」
    #   = 打ち切りの正直な大きさになる。
    budget = int(ccfg["max_pairs_per_day"])
    over = len(seen_lo) >= budget or len(seen_hi) >= budget
    seen_lo.add(int(hi.id))
    seen_hi.add(int(lo.id))
    if over:
        st["copresence_dropped"] += 1
        return
    st["copresence"] += 1
    sim.logger.log(Event(
        step=int(step), sim_min=int(sim_min), agent_id=int(lo.id),
        kind="train_copresence", x=float(lo.x), y=float(lo.y),
        payload={"line": str(line), "train_min": int(train_min), "car": int(car),
                 "other_id": int(hi.id), "zone_adjacent": bool(adj),
                 "crew": bool(crew)}))


# =========================================================================== #
# (5) 降車 = 乗車 1 回の締め(L1 + 属性の消滅)
# =========================================================================== #
def _end(sim, cfg: dict, st: dict, agent, step: int, sim_min: int) -> None:
    """乗車の終わり: ``train_ride`` 1 件 + 同席の対 + **属性を消す**(OFF 慣用)。"""
    r = getattr(agent, "_train_ride", None)
    if not isinstance(r, dict):
        _clear(agent)
        return
    sim.logger.log(Event(
        step=int(step), sim_min=int(sim_min), agent_id=int(agent.id),
        kind="train_ride", x=float(agent.x), y=float(agent.y),
        payload={"line": str(r["line"]), "train_min": int(r["train_min"]),
                 "car": int(r["car"]), "cars": int(r["cars"]),
                 "zone": str(r["zone"]), "seated": bool(r["seated"]),
                 "load": int(r["load"]), "capacity": int(r["capacity"]),
                 "load_factor": round(float(r["load"]) / max(1.0,
                                                            float(r["capacity"])), 3),
                 "density": round(float(r["density"]), 3), "act": str(r["act"]),
                 # 車内に居た step 数(**降車 step は含めない**: この step には
                 # もう改札を出ている = _phase_wake_and_returns が先に走っている)
                 "steps": max(1, int(step) - int(r["start"]))}))
    st["riders"] += 1
    for oid, ozone in r["mates"]:
        other = sim.agent_by_id.get(int(oid))
        if other is None:
            continue
        _copresence(sim, cfg, st, agent, other, r["line"], r["train_min"],
                    r["car"], r["zone"], ozone, False, step, sim_min)
    _clear(agent)


def _clear(agent) -> None:
    """車内の属性を**消す**(存在しないことが OFF の表明 = 属性不在の慣用)。"""
    for name in ("_train_ride", "train_car", "train_zone", "train_seated"):
        if hasattr(agent, name):
            try:
                delattr(agent, name)
            except AttributeError:                 # dataclass 既定を持つ属性は無い
                pass


# =========================================================================== #
# (6) 車掌の巡回(``transit_staff`` も ON のときだけ)
# =========================================================================== #
def _patrol(sim, cfg: dict, st: dict, groups: dict, step: int, sim_min: int) -> None:
    """当直の乗務員が受け持ちの列車の車両を 1 つずつ進む(決定論)。

    - 受け持ちは (路線, 到着分) の群を**名簿順の剰余**で割り当てる。乗務員が
      群より少ないときは 1 人が複数の列車を受け持つ(正直な単純化 5)。
    - 車両 index は ``(step // patrol_interval_steps) % 両数`` = 完全決定論。
    - その車両の乗客とは同席として記録する(``crew: true`` の行)。
    """
    ccfg = cfg["conductor"]
    if not ccfg["enabled"]:
        return
    from . import transit_staff as _staff
    if not _staff.enabled(sim):
        return
    crews = _staff.on_duty_crews(sim, int(sim_min))
    if not crews:
        return
    interval = max(1, int(ccfg["patrol_interval_steps"]))
    for i, key in enumerate(sorted(groups)):
        line, train_min = key
        members = groups[key]
        spec = spec_for(cfg, line)
        cars = max(1, int(spec["cars"]))
        car = (int(step) // interval) % cars
        crew = crews[i % len(crews)]
        here = [a for a in members
                if isinstance(getattr(a, "_train_ride", None), dict)
                and int(a._train_ride["car"]) == car]
        sim.logger.log(Event(
            step=int(step), sim_min=int(sim_min), agent_id=int(crew.id),
            kind="train_patrol", x=float(crew.x), y=float(crew.y),
            payload={"line": str(line), "train_min": int(train_min),
                     "car": int(car), "cars": int(cars), "n_agents": len(here),
                     "n_pax": int(car_load(cfg, spec, int(train_min)))}))
        st["patrols"] += 1
        for a in sorted(here, key=lambda x: int(x.id)):
            # 車掌は通路を歩く = 立位区画のどことも隣接する
            _copresence(sim, cfg, st, crew, a, line, train_min, car,
                        AISLE, str(a._train_ride["zone"]), True, step, sim_min)


# =========================================================================== #
# (7) 車掌が読む車内負荷(``transit_staff`` の停車時間判断への配線)
# =========================================================================== #
def _live_load(sim) -> tuple[int, int]:
    """いま車内に居る地図内乗客のうち**立っている人数**と、埋まっている車両数。"""
    standing, cars = 0, set()
    for a in sim.agents:
        r = getattr(a, "_train_ride", None)
        if isinstance(r, dict):
            cars.add((str(r["line"]), int(r["train_min"]), int(r["car"])))
            if not bool(r["seated"]):
                standing += 1
    return standing, len(cars)


def dwell_extra_load(sim) -> int:
    """停車時間の負荷へ足す車内ぶん(**既定 0 = 観測のみ**)。OFF は必ず 0。

    ★既定を 0 にしてある理由: ホーム密度 → 停車時間の較正
      (+15 人/車両 ≒ +1 秒)は既に ``envfeedback`` 規則1 が**ホーム側の人数**で
      持っている。車内の立客を素で足すと**同じ物理を二重計上**する。
      重みは実験用のレバーとして conf に開けてあるだけである。
    """
    if not enabled(sim):
        return 0
    cfg = cfg_of(sim)
    w = float(cfg["conductor"]["dwell_load_weight"])
    if w <= 0.0 or not cfg["conductor"]["enabled"]:
        return 0
    standing, _cars = _live_load(sim)
    return int(round(w * standing))


def dwell_payload(sim) -> dict:
    """``dwell_decision`` の payload へ足す車内の観測欄。**OFF は空 dict**(バイト一致)。"""
    if not enabled(sim):
        return {}
    cfg = cfg_of(sim)
    if not cfg["conductor"]["enabled"]:
        return {}
    standing, cars = _live_load(sim)
    return {"interior_standing": int(standing), "interior_cars": int(cars),
            "interior_weight": round(float(cfg["conductor"]["dwell_load_weight"]), 4)}


# =========================================================================== #
# (8) 単一作用点(step 末)
# =========================================================================== #
def phase(sim, step: int, sim_min: int) -> None:
    """車内層の**単一作用点**。既定 OFF は即 return(バイト一致)。

    置き場所は ``envfeedback.update`` と ``transit_staff.phase`` の**あいだ**。理由:
      - 降車(``_phase_wake_and_returns``)は step の前半に済んでいるので、この時点で
        「まだ圏外に居る = 車内」が確定している。
      - ``transit_staff`` の停車時間判断が同じ step のうちに車内負荷を読めるように、
        車掌より**先**に車内を確定させる。

    ★世界状態(所持金・位置・関係・drive・opinion・発話)は 1 つも動かさない。
      動かすのは ``fatigue`` と車内の属性だけで、``generate()`` は 1 本も呼ばない。
    """
    if not enabled(sim):
        return
    cfg = cfg_of(sim)
    st = _state(sim)
    riders = _riders(sim, cfg, st, step)
    cur = {int(a.id) for a, _l, _t, _e in riders}
    # ---- (a) 降車: 前 step まで車内に居て、今 step は居ない個体 ------------------- #
    for a in sorted(sim.agents, key=lambda x: int(x.id)):
        if isinstance(getattr(a, "_train_ride", None), dict) and int(a.id) not in cur:
            _end(sim, cfg, st, a, step, sim_min)
    # ---- (b) 乗車: (路線, 到着分) の群ごとに車両と区画を割り当てる ----------------- #
    groups: dict[tuple[str, int], list] = {}
    for a, line, tmin, _e in riders:
        groups.setdefault((line, int(tmin)), []).append(a)
    for key in sorted(groups):
        _assign(sim, cfg, st, key[0], key[1], groups[key], step, sim_min)
        _mates(cfg, groups[key])
        for a in groups[key]:
            _tick(sim, cfg, a)
    # ---- (c) 車掌の巡回(transit_staff も ON のときだけ)------------------------- #
    _patrol(sim, cfg, st, groups, step, sim_min)


# =========================================================================== #
# 観測(読むだけ。``summary.json`` への配線は本レーンの所有外 = 正直な限界 5)
# =========================================================================== #
def provenance(sim) -> dict | None:
    """車内層の観測タリー(既定 OFF は None = 何も出さない)。

    ★``enabled`` は 2 つの ON の論理積なので、「本層は ON だがパルスが OFF」の
      ランでも**そう判る形**で返す(黙って None にしない = 設定ミスを隠さない)。
    """
    cfg = cfg_of(sim)
    if not cfg["enabled"]:
        return None
    out: dict = {"schema": SCHEMA, "pulse_required": True,
                 "pulse_on": bool(pulse_on(sim)),
                 "ride_minutes": int(cfg["ride_minutes"]),
                 "lines_declared": len(cfg["lines"])}
    st = getattr(sim, "_train_state", None)
    if st is None:
        out.update({"rides": 0, "riders": 0, "copresence": 0,
                    "copresence_dropped": 0, "patrols": 0, "mem_lines": 0,
                    "cars_used": {}})
        return out
    out.update({"rides": int(st["rides"]), "riders": int(st["riders"]),
                "copresence": int(st["copresence"]),
                "copresence_dropped": int(st["copresence_dropped"]),
                "patrols": int(st["patrols"]), "mem_lines": int(st["mem_lines"]),
                "cars_used": {k: int(v) for k, v in sorted(st["cars_used"].items())}})
    return out
