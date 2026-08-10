"""設備 = **摩耗する装置**(昇降設備の DEVS 遷移。``world.facilities``・**既定 OFF**)。

正典
----
- ``docs/plans/body-incident-layer-plan.md`` §3「設備」行:
  「DEVS 摩耗遷移。EV/エスカレーター: **wear clock + 月1保守 → 閉じ込め → インターホン →
   保守会社エージェント(30-80分) → 復旧**。停電・漏水 = **skip**(東京 SAIDI 年13分 =
   起きないが正解)」。較正アンカー = **EV 閉じ込め 全国 1 万件/年規模**。
- ``docs/plans/actor-model-migration.md`` P2 + ``src/society/devices.py``(DEVS 契約:
  状態を変えてよいのは δ_int(宣言済みのスケジュール/物理)と δ_ext(外部入力への応答)だけ)。

何を解く問題か
--------------
この街の建物には階がある(``levels``)。屋内層は人を階へ運ぶが、**運んでいる設備そのものは
世界に存在しない**。エレベーターは壊れないし、点検もされないし、止まったときに人が閉じ
込められることもない。本 module はその 1 点だけを埋める:

    昇降設備は **使われるほど摩耗し**(δ_ext)、**月 1 回の保守で摩耗が戻り**(δ_int)、
    摩耗しているほど**故障しやすく**、故障すれば**人が閉じ込められ**、閉じ込められた人が
    **インターホンで通報し**(行為)、**対応する人が来て**(行為)、復旧する。

DEVS 契約への準拠(``devices.Device`` の 2 遷移だけを口として持つ)
------------------------------------------------------------------
  δ_int ``on_schedule(sim_min)``
      - 月 1 回の保守(``next_maint_min`` を跨いだら wear と uses を 0 へ)
      - 待機摩耗(経過時間ぶんの劣化)
      - 修理完了(``down_until_min`` を跨いだら復旧)
      いずれも**宣言済みのスケジュール**であって自発的遷移ではない。

  δ_ext ``on_input(actor_id, input, sim_min)``
      - ``{"kind": "use"}``   … **エージェントの利用**(actor_id = 利用者)。摩耗が進む。
      - ``{"kind": "fault"}`` … **世界(物理)からの外部入力**(actor_id = -1)。故障する。

★``{"kind": "fault"}`` を δ_ext で渡す理由(**devices.py の契約を破らないため**):
  DEVS の δ_ext は「外部入力への遷移」であって、入力元がエージェントである必要はない。
  ``devices.py`` の docstring は既存の 2 用途を「エージェントの行為への応答」と説明して
  いるが、本 module は**世界(物理)からの入力**も同じ口へ渡す(actor_id=-1)。こうすると
  **装置自身は乱数を 1 本も引かない**(= 装置は決定論的な応答関数、という設計契約が
  そのまま保たれる)。故障の抽選は本 module の ``phase`` 側で、専用の named stream
  ``incident_facility`` から**全台まとめて 2 本だけ**引く。

較正(**公表値・実測ベース。推定にはそう書く**)
------------------------------------------------
- **EV 閉じ込め = 全国およそ 1 万件/年**、**全国のエレベーター設置台数 ≒ 78 万台**。
  → 1 台あたり **0.0128 件/年 = 3.5×10⁻⁵ 件/台/日**(``elevator.fault_per_day``)。
  現行地図 v7 で ``levels >= 5`` の建物は 405 棟なので、この世界の期待値は
  **0.014 件/日**(= 10 日ランで 0.14 件)。**「何も起きない日が正しい」**側に落ちる。
- **エスカレーターの故障率は公表統計が薄い**(閉じ込めが構造上起きないので「閉じ込め
  件数」という統計自体が存在しない)。既定 5.0×10⁻⁵ 件/台/日 は **推定**であり、
  アンカーではないことをここに明記する(``escalator.fault_per_day``)。
  ★エスカレーターは **閉じ込めない**(``traps: false``)= 停止するだけ。
- **保守 = 月 1 回**(法定の定期検査は年 1 回だが、保守点検契約は月次が業界慣行)。
- **復旧まで 30〜80 分**(計画 §3)。台ごとに安定ハッシュで決める決定論値。

★摩耗の効きは小規模ランではほぼゼロである(正直な限界): 本シムの利用回数は実規模の
  数百分の 1 なので、``wear`` は ``wear_full`` に遠く届かず、故障ハザードはほぼ基準値の
  ままになる。摩耗と保守が数字として効くのは実規模ラン(数万人 × 数十日)だけである。
  **効かないことを隠して係数を膨らませない**(較正破壊になる)。

★**停電・漏水は作らない**(``incidents_env`` の docstring と同じ理由。東京の SAIDI は
  年約 13 分 = 10 日ランの期待値 0.4 秒)。供給事業者の装置 id
  (``devices.DEV_OPERATOR_INFRA``)は名簿に在るので、必要になったらそこへ δ_ext を
  足せば成立する(前方互換のメモ = 実装はしない)。

対応する人(**保守会社の技術者は名簿に居ない**という正直な事実)
----------------------------------------------------------------
L5 名簿(``scripts/build_persona_pool.py``)に「設備保守員」は 1 人も居ない。存在しない
職を勝手に生やすと**名簿と世界が食い違う**(city_ops が自販機補充を作らなかったのと
同じ判断)。そこで既定の対応者は **``警備員``**(名簿に実在)にした。これは妥協ではなく
**現実の一次対応そのもの**である: 日本のビルで閉じ込めに最初に応じるのは防災センター /
管理会社の常駐者で、保守会社の技術者はそのあと 30〜80 分かけて到着する。したがって
本 module の ``facility_dispatch`` は「一次対応者が現場へ向かう」行為で、``repair_min``
が「保守会社の到着から復旧まで」を表す。``responder_occupations`` に「設備保守員」も
書いてあるので、名簿を再生成してその職を入れたランでは自動的にそちらが優先される。

R1 ドクトリン
-------------
- 既定 ``world.facilities.enabled: false`` では ``phase`` が即 return し、**装置名簿を
  組まず・L1 に 1 件も出ず・sim に state が生えず・agent に属性が生えず・乱数 stream を
  1 本も引かず・プロンプトが 1 バイトも変わらない**(= ゴールデン L1 バイト一致)。
- **generate() の呼び出しサイトを 1 つも作らない**(LLM 追加呼ゼロ = k 非依存)。
- 新しい乱数は **新しい named stream 1 本だけ**(``incident_facility``)。
  ★1 step に引くのは**最大 2 本**(発生数の Poisson + 台の選択)= 台数に依存しない。
- L1 は **1 行 + 前兆状態を payload 同梱**(``wear`` / ``uses`` / ``days_since_maint``)
  = 「故障が摩耗から内生した」ことを解析側で機械検証できる。

正直な限界(5 件)
------------------
1. **1 棟 1 台**(エレベーターもエスカレーターも)。地図に設備の台数が無いので、持って
   いないふりをしない。実際の大型商業施設は EV を十数台持つ。
2. **利用は L1 の ``floor_move`` / ``enter_building`` から数える**(1 step 遅れうる)。
   屋内層(``world.indoor``)が OFF のランでは階移動が起きないので、摩耗は待機ぶんだけに
   なる(= 正直な依存関係。黙って 0 件にはならず provenance の ``uses`` が 0 と出る)。
3. **閉じ込められた人はその場に留まらない**。身体・在席の状態は H1(身体レイヤー)の
   管轄なので、本 module は ``sleeping`` も ``route`` も**書かない**(同じ現象を 2 つの
   レーンが書かないため)。閉じ込めは L1 と記憶の 1 行として残る。
4. **復旧は時間だけで決まる**。対応者が現場に着いたかどうかは復旧時刻を変えない
   (``city_ops`` の救急が「現着率は観測に出すが治療内容は作らない」としたのと同じ線引き)。
5. **待機摩耗は resume 直後の 1 step だけ過少になりうる**… ということは無い:
   摩耗も保守も**絶対時刻(sim_min)**で計算するので、resume を跨いでも同じ値に落ち着く
   (``traces`` の蒸発が「経過 step から計算する」ことで resume 同値を得たのと同型)。
"""
from __future__ import annotations

import hashlib

from . import devices as _devices
from .observer.schema import Event, register_event_kind

SCHEMA = 1

# --------------------------------------------------------------------------- #
# L1 イベント種の**材料側**登録(devices.py / city_ops.py と同じ流儀)
# --------------------------------------------------------------------------- #
register_event_kind(
    "facility_fault",
    "昇降設備の故障(摩耗からの内生)。agent_id=-1(装置の出来事)"
    "{device_id, kind, building, node, wear, uses, days_since_maint, trapped,"
    " repair_min}")
register_event_kind(
    "facility_call",
    "インターホンでの通報(**この行為が対応の原因**)。agent_id = 通報者"
    "{device_id, building, node, trapped, self_call}")
register_event_kind(
    "facility_dispatch",
    "通報に応えた対応者の出動。agent_id = 対応者(不在なら -1 かつ unstaffed=true)"
    "{device_id, node, responder, response_min, repair_min, unstaffed}")
register_event_kind(
    "facility_restore",
    "昇降設備の復旧。agent_id = 対応者(不在で自然復旧した回は -1)"
    "{device_id, node, down_min, trapped}")
register_event_kind(
    "facility_maintenance",
    "月 1 回の保守点検(wear のリセット)。agent_id = 点検者(不在なら -1)"
    "{device_id, kind, wear_before, uses_before, staffed, day}")

# --------------------------------------------------------------------------- #
# 記憶の 1 行(**出来事の種類だけの純関数**。数字・config・実験条件は 1 つも入らない)
# --------------------------------------------------------------------------- #
TRAPPED_TEXT = "エレベーターが止まって閉じ込められ、インターホンで連絡した。"
CALL_TEXT = "止まってしまった設備のことをインターホンで知らせた。"
RESPOND_TEXT = "設備が止まったという連絡を受けて現場へ向かった。"

#: 設備の種別(payload に出る有限語彙)
ELEVATOR = "elevator"
ESCALATOR = "escalator"
FACILITY_KINDS: tuple[str, ...] = (ELEVATOR, ESCALATOR)

#: 対応者の職(**名簿に実在するものを既定にする**。docstring「対応する人」節を参照)
RESPONDER_OCCS: tuple[str, ...] = ("設備保守員", "警備員")

#: エスカレーターを置く建物の ``kind``(地図に設備データが無いための代理判定)
ESCALATOR_BUILDING_KINDS: tuple[str, ...] = ("retail", "station")

DEFAULTS: dict = {
    "enabled": False,
    "max_events_per_step": 16,      # 1 step に出す本 module の L1 件数の上限(安全弁)
    "responder_occupations": list(RESPONDER_OCCS),
    "response_speed_m_per_min": 80.0,   # response_min の換算(徒歩。モデル値)
    "maintenance_days": 30,         # 月 1 回の保守(業界慣行)
    "wear_full": 500.0,             # この摩耗で故障ハザードが (1+wear_gain) 倍になる
    "wear_gain": 1.0,
    "use_wear": 1.0,                # 1 回の利用で進む摩耗
    "idle_wear_per_day": 1.0,       # 待機だけでも進む摩耗(経年劣化の代理)
    "elevator": {
        "enabled": True,
        "min_levels": 5,            # この階数以上の建物に 1 台
        # ★アンカー: 全国 1 万件/年 ÷ 全国 78 万台 = 0.0128 件/台/年
        "fault_per_day": 3.5e-5,
        "repair_min_lo": 30.0,      # 復旧まで 30〜80 分(計画 §3)
        "repair_min_hi": 80.0,
        "traps": True,              # 閉じ込めが起きる
    },
    "escalator": {
        "enabled": True,
        "min_levels": 3,
        # ★**推定**(公表統計が薄い)。アンカーではないことを conf にも明記する
        "fault_per_day": 5.0e-5,
        "repair_min_lo": 30.0,
        "repair_min_hi": 80.0,
        "traps": False,             # エスカレーターは構造上**閉じ込めない**
    },
}

_TOP_FLOATS = ("response_speed_m_per_min", "wear_full", "wear_gain", "use_wear",
               "idle_wear_per_day")
_SUB_FLOATS = ("fault_per_day", "repair_min_lo", "repair_min_hi")


# --------------------------------------------------------------------------- #
# cfg 正準化(devices.build_cfg / city_ops.build_cfg と同型)
# --------------------------------------------------------------------------- #
def _to_plain(raw):
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(raw):
            return OmegaConf.to_container(raw, resolve=True)
    except Exception:                              # noqa: BLE001(旧 config 互換)
        pass
    return raw


def _sub(raw, defaults: dict) -> dict:
    out = dict(defaults)
    got = dict(_to_plain(raw) or {})
    for key, val in got.items():
        if key not in out:
            continue                               # 未知キーは捨てる(捏造しない)
        if key in ("enabled", "traps"):
            out[key] = bool(val)
        elif key in _SUB_FLOATS:
            try:
                out[key] = max(0.0, float(val))
            except (TypeError, ValueError):
                continue
        else:
            try:
                out[key] = max(1, int(val))
            except (TypeError, ValueError):
                continue
    out["repair_min_hi"] = max(float(out["repair_min_hi"]),
                               float(out["repair_min_lo"]))
    return out


def build_cfg(raw) -> dict:
    """conf の ``world.facilities`` ブロックを正準化(既定 OFF=現行と完全同一)。"""
    raw = dict(_to_plain(raw) or {})
    cfg: dict = {"enabled": bool(raw.get("enabled", False))}
    cfg["max_events_per_step"] = max(0, int(raw.get("max_events_per_step",
                                                    DEFAULTS["max_events_per_step"])))
    occs = _to_plain(raw.get("responder_occupations"))
    if isinstance(occs, str):
        occs = [occs]
    occs = tuple(str(x).strip() for x in (occs or ()) if str(x).strip())
    cfg["responder_occupations"] = list(occs or RESPONDER_OCCS)
    for key in _TOP_FLOATS:
        try:
            cfg[key] = max(0.0, float(raw.get(key, DEFAULTS[key])))
        except (TypeError, ValueError):
            cfg[key] = float(DEFAULTS[key])
    cfg["wear_full"] = max(1.0, cfg["wear_full"])
    try:
        cfg["maintenance_days"] = max(1, int(raw.get("maintenance_days",
                                                     DEFAULTS["maintenance_days"])))
    except (TypeError, ValueError):
        cfg["maintenance_days"] = int(DEFAULTS["maintenance_days"])
    cfg[ELEVATOR] = _sub(raw.get(ELEVATOR), DEFAULTS[ELEVATOR])
    cfg[ESCALATOR] = _sub(raw.get(ESCALATOR), DEFAULTS[ESCALATOR])
    return cfg


def cfg_of(sim) -> dict:
    """設備設定(初回のみ ``sim.cfg.world.facilities`` から遅延構築してキャッシュ)。"""
    got = getattr(sim, "facilitiescfg", None)
    if got is None:
        try:
            raw = (sim.cfg.get("world", None) or {}).get("facilities", None)
        except Exception:                          # noqa: BLE001(旧 config 互換)
            raw = None
        got = build_cfg(raw)
        sim.facilitiescfg = got
    return got


def enabled(sim) -> bool:
    """設備層が有効か。既定 OFF=新経路を一切通らない(バイト一致)。"""
    return bool(cfg_of(sim)["enabled"])


def _stable_hash(value: str) -> int:
    """プロセス非依存の安定ハッシュ(rng.py / city_ops.py と同流儀)。"""
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def _hash_frac(value: str) -> float:
    """安定ハッシュから [0,1) の決定論値(台ごとの保守日・復旧時間のばらし)。"""
    return (_stable_hash(value) % 1_000_000) / 1_000_000.0


def facility_device_id(building: str, kind: str) -> str:
    """昇降設備の装置 id(**安定 id**。``lift:<建物 id>-ev`` / ``-es``)。

    ★接頭辞は ``devices.LIFT_PREFIX``(名簿の唯一の源)。呼び出し側で f 文字列を書かない
      (``devices.pos_device_id`` / ``org_device_id`` と同じ規約)。
    """
    tag = "ev" if str(kind) == ELEVATOR else "es"
    return f"{_devices.LIFT_PREFIX}:{building}-{tag}"


# =========================================================================== #
# 装置本体(DEVS: δ_int = 保守・待機摩耗・修理完了 / δ_ext = 利用・故障入力)
# =========================================================================== #
class LiftDevice(_devices.Device):
    """昇降設備 1 台(エレベーター or エスカレーター)。**目標を持たないアクター**。

    状態(``STATE`` = δ でしか動かさない)
      ``wear``            … 摩耗(利用 + 待機で増え、保守で 0 へ)
      ``uses``            … 保守以降の利用回数(前兆状態として L1 に載る)
      ``down``            … 故障中か
      ``down_until_min``  … 復旧予定の絶対時刻[sim_min](**絶対時刻 = resume 同値の要**)
      ``next_maint_min``  … 次の保守の絶対時刻[sim_min]
      ``last_maint_min``  … 直近の保守の絶対時刻(``days_since_maint`` の分母)

    ★**摩耗は累積加算ではなく「その時刻の純関数」として毎 δ で置き直す**:

        wear = uses × use_wear + idle_per_min × (sim_min − last_maint_min)

      累積 ``+=`` にすると、resume で途中から足し直したときに浮動小数の丸め差が
      残り、**resume ≠ straight**(実測: 1e-3 の差が L1 の payload に出た)。
      絶対時刻の純関数にすれば、どこで切って再開しても同じ値に落ち着く
      (traces の蒸発が「経過 step から計算する」ことで resume 同値を得たのと同型)。

    ★**乱数を 1 本も引かない**(装置は決定論的な応答関数 = devices.py の設計契約)。
      故障の抽選は本 module の ``phase`` が named stream から引き、結果だけを
      ``{"kind": "fault"}`` という**外部入力**として δ_ext へ渡す。
    """

    kind = _devices.LIFT_PREFIX
    STATE = ("wear", "uses", "down", "down_until_min", "next_maint_min",
             "last_maint_min")

    def __init__(self, device_id: str, facility: str, building: str, node: str,
                 *, cfg: dict, sub: dict, start_min: int = 0,
                 restored: dict | None = None):
        super().__init__(device_id)
        self.facility = str(facility)
        self.building = str(building)
        self.node = str(node)
        self.traps = bool(sub["traps"])
        self.fault_per_day = float(sub["fault_per_day"])
        self.wear_full = float(cfg["wear_full"])
        self.wear_gain = float(cfg["wear_gain"])
        self.use_wear = float(cfg["use_wear"])
        self.idle_per_min = float(cfg["idle_wear_per_day"]) / 1440.0
        self.maint_min = max(1, int(cfg["maintenance_days"])) * 1440
        # 復旧までの時間 = 台ごとの決定論値(30〜80 分)
        lo, hi = float(sub["repair_min_lo"]), float(sub["repair_min_hi"])
        self.repair_min = round(lo + (hi - lo) * _hash_frac(f"repair/{device_id}"), 1)
        # ---- DEVS 状態(δ でしか動かさない)----
        rec = dict(restored or {})
        self.wear = float(rec.get("wear", 0.0))
        self.uses = int(rec.get("uses", 0))
        self.down = bool(rec.get("down", False))
        self.down_until_min = int(rec.get("down_until_min", -1))
        # 保守日は台ごとにばらす(全台が同じ日に一斉点検する退化を避ける)
        offset = int(_hash_frac(f"maint/{device_id}") * self.maint_min)
        self.next_maint_min = int(rec.get("next_maint_min",
                                          int(start_min) + offset))
        self.last_maint_min = int(rec.get("last_maint_min", int(start_min)))
        self.seal()

    # ---- 摩耗 = 絶対時刻の純関数(累積加算にしない = resume 同値の要)--------- #
    def _wear_at(self, sim_min: int) -> float:
        return (float(self.uses) * self.use_wear
                + self.idle_per_min * max(0, int(sim_min) - int(self.last_maint_min)))

    # ---- δ_int: 宣言済みのスケジュール / 物理 -------------------------------- #
    def _delta_int(self, sim_min: int):
        """待機摩耗 + 月 1 保守 + 修理完了。**返り値 = この tick で起きた遷移の名前**。"""
        now = int(sim_min)
        out: dict = {}
        fresh = self._wear_at(now)
        if fresh != float(self.wear):
            self.wear = fresh
        if now >= int(self.next_maint_min):
            out["maintained"] = {"wear": round(float(self.wear), 3),
                                 "uses": int(self.uses)}
            self.wear = 0.0
            self.uses = 0
            self.last_maint_min = now
            # 「跨いだぶんだけ」進める(長い resume で無限ループにしない)
            span = self.maint_min
            missed = (now - int(self.next_maint_min)) // span + 1
            self.next_maint_min = int(self.next_maint_min) + span * int(missed)
        if self.down and int(self.down_until_min) >= 0 \
                and now >= int(self.down_until_min):
            out["restored"] = {"down_min": now - (int(self.down_until_min)
                                                  - int(round(self.repair_min)))}
            self.down = False
            self.down_until_min = -1
        return out or False

    # ---- δ_ext: 外部入力への応答(利用 = エージェント / 故障 = 世界)----------- #
    def _delta_ext(self, actor_id: int, input: dict, sim_min: int) -> dict:  # noqa: A002
        kind = str(dict(input or {}).get("kind", ""))
        if kind == "use":
            if self.down:
                return {"ok": False, "down": True, "device_id": self.device_id}
            self.uses = int(self.uses) + 1
            self.wear = self._wear_at(sim_min)     # ★純関数で置き直す(累積しない)
            return {"ok": True, "down": False, "device_id": self.device_id}
        if kind == "fault":
            if self.down:
                return {"faulted": False, "device_id": self.device_id}
            self.down = True
            self.down_until_min = int(sim_min) + int(round(self.repair_min))
            return {"faulted": True, "traps": self.traps,
                    "device_id": self.device_id}
        return {"device_id": self.device_id}

    # ---- 観測(読むだけ。状態を 1 バイトも動かさない)------------------------ #
    def hazard_per_step(self, steps_per_day: int) -> float:
        """この step にこの台が故障する確率(**摩耗で増える**)。故障中は 0。"""
        if self.down:
            return 0.0
        ratio = min(1.0, float(self.wear) / self.wear_full)
        return (self.fault_per_day / max(1, int(steps_per_day))
                * (1.0 + self.wear_gain * ratio))

    def days_since_maint(self, sim_min: int) -> float:
        return round((int(sim_min) - int(self.last_maint_min)) / 1440.0, 3)

    def snapshot(self) -> dict:
        """checkpoint 用の素の dict(**DEVS 状態だけ**。導出値は保存しない)。

        ★``wear`` は丸めずに素の float を入れる: 丸めると resume 側の摩耗が
          straight と 1e-3 ずれ、L1 の payload まで食い違う(実測して直した)。
        """
        return {"wear": float(self.wear), "uses": int(self.uses),
                "down": bool(self.down),
                "down_until_min": int(self.down_until_min),
                "next_maint_min": int(self.next_maint_min),
                "last_maint_min": int(self.last_maint_min)}


# =========================================================================== #
# 名簿(地図から決定論で組む。**ON 経路でのみ生やす**)
# =========================================================================== #
def build_registry(sim) -> _devices.DeviceRegistry:
    """conf と地図から昇降設備の名簿を組む(**OFF では空の名簿**)。

    - エレベーター: ``levels >= elevator.min_levels`` の建物に 1 台
    - エスカレーター: ``kind`` が retail / station で ``levels >= escalator.min_levels``
    並びは建物 id 昇順(``DeviceRegistry`` の反復は device_id 昇順なので二重に決定論)。
    """
    reg = _devices.DeviceRegistry()
    cfg = cfg_of(sim)
    if not cfg["enabled"]:
        return reg
    start_min = int(getattr(getattr(sim, "clock", None), "start_min", 0) or 0)
    restored = dict(getattr(sim, "_facility_restored", None) or {})
    ev, es = cfg[ELEVATOR], cfg[ESCALATOR]
    for bld in sorted(sim.city.buildings, key=lambda b: str(b["id"])):
        node = str(bld.get("entrance") or "")
        if not node:
            continue
        bid = str(bld["id"])
        levels = max(1, int(bld.get("levels") or 1))
        kind = str(bld.get("kind") or "")
        if ev["enabled"] and levels >= int(ev["min_levels"]):
            did = facility_device_id(bid, ELEVATOR)
            reg.add(LiftDevice(did, ELEVATOR, bid, node, cfg=cfg, sub=ev,
                               start_min=start_min, restored=restored.get(did)))
        if (es["enabled"] and kind in ESCALATOR_BUILDING_KINDS
                and levels >= int(es["min_levels"])):
            did = facility_device_id(bid, ESCALATOR)
            reg.add(LiftDevice(did, ESCALATOR, bid, node, cfg=cfg, sub=es,
                               start_min=start_min, restored=restored.get(did)))
    return reg


def registry_of(sim) -> _devices.DeviceRegistry:
    """設備名簿(初回のみ構築)。**OFF では呼ばれない**(phase が先に return)。"""
    reg = getattr(sim, "facilities", None)
    if reg is None:
        reg = build_registry(sim)
        sim.facilities = reg
    return reg


def _by_building(sim) -> dict:
    """建物 id → {種別: 装置}(索引。名簿からの導出なので resume 不要)。"""
    cache = getattr(sim, "_facility_index", None)
    if cache is None:
        cache = {}
        for dev in registry_of(sim):
            cache.setdefault(dev.building, {})[dev.facility] = dev
        sim._facility_index = cache
    return cache


# --------------------------------------------------------------------------- #
# state(タリー。**OFF では 1 つも生えない**)
# --------------------------------------------------------------------------- #
def _state(sim) -> dict:
    st = getattr(sim, "_facility_state", None)
    if st is None:
        st = {"schema": SCHEMA, "by_kind": {}, "dropped": 0,
              "uses": 0, "uses_blocked": 0, "faults": 0, "faults_by_kind": {},
              "trapped": 0, "calls": 0, "dispatches": 0, "unstaffed": 0,
              "restores": 0, "maintenances": 0, "wear_max": 0.0}
        sim._facility_state = st
    return st


def _bump(table: dict, key: str, n: int = 1) -> None:
    table[key] = int(table.get(key, 0)) + int(n)


class _Budget:
    """1 step に出す L1 件数の上限(``city_ops._Budget`` と同型)。"""

    def __init__(self, sim, st: dict, cap: int):
        self.sim = sim
        self.st = st
        self.left = int(cap)

    def log(self, event, device_id: str) -> bool:
        if self.left <= 0:
            self.st["dropped"] += 1
            return False
        self.left -= 1
        # ★装置 id を列へ刻む(causality OFF では Event を 1 バイトも触らない)
        _devices.log_device(self.sim, event, device_id)
        _bump(self.st["by_kind"], event.kind)
        return True


# --------------------------------------------------------------------------- #
# 利用(δ_ext)= 既存 L1 の ``floor_move`` / ``enter_building`` を読む
# --------------------------------------------------------------------------- #
def _new_events(sim) -> list:
    """前回処理済み総数(watermark)以降の新規 L1(``traces._new_events`` と同型)。

    watermark は**プロセス内 logger カウンタ由来**なので checkpoint に保存しない
    (resume 直後は新しい logger の 0 から数え直す = 第59 assets / 第96 traces と同流儀)。
    """
    logger = getattr(sim, "logger", None)
    if logger is None:
        return []
    events = logger.events
    total = int(getattr(logger, "_n_flushed", 0)) + len(events)
    processed = int(getattr(sim, "_facility_watermark", 0))
    fresh = max(0, min(total - processed, len(events)))
    sim._facility_watermark = total
    return events[len(events) - fresh:] if fresh else []


def _device_for_floor(devs: dict, floor: int, es_max_floor: int):
    """その階へ運ぶ設備(**低層はエスカレーター・高層はエレベーター**の決定論規則)。

    地図に「どの階までエスカレーターがあるか」のデータが無いので、
    ``es_max_floor``(既定 4)までをエスカレーターの守備範囲とする**明示の仮定**である。
    """
    if int(floor) <= 1:
        return None                                # 1 階 = 昇降しない
    if int(floor) <= int(es_max_floor) and ESCALATOR in devs:
        return devs[ESCALATOR]
    return devs.get(ELEVATOR) or devs.get(ESCALATOR)


def _apply_uses(sim, st: dict, step: int, sim_min: int) -> None:
    """新規の階移動を装置の δ_ext(``use``)へ流す(**摩耗の唯一の内生源**)。"""
    index = _by_building(sim)
    if not index:
        return
    for e in _new_events(sim):
        if e.kind not in ("floor_move", "enter_building"):
            continue
        payload = e.payload or {}
        devs = index.get(str(payload.get("building") or ""))
        if not devs:
            continue
        try:
            floor = int(payload.get("floor") or 0)
        except (TypeError, ValueError):
            continue
        dev = _device_for_floor(devs, floor, 4)
        if dev is None:
            continue
        out = dev.on_input(int(e.agent_id), {"kind": "use"}, sim_min)
        if out.get("ok"):
            st["uses"] += 1
        else:
            st["uses_blocked"] += 1


# --------------------------------------------------------------------------- #
# 対応者(**読むだけの選定**。名簿に居なければ unstaffed を正直に出す)
# --------------------------------------------------------------------------- #
def _responder(sim, cfg: dict, node: str):
    """現場に最も近い対応者(同距離は id 昇順)。居なければ ``None``。"""
    occs = frozenset(cfg["responder_occupations"])
    try:
        tx, ty = sim.city.node_xy(str(node))
    except Exception:                              # noqa: BLE001(未知ノードの保険)
        return None
    best = None
    best_key = None
    for agent in sim.agents:
        if str(getattr(agent, "occupation", "")) not in occs:
            continue
        if agent.loc == "outside" or agent.sleeping:
            continue
        if int(getattr(agent, "facility_call_until", -1)) >= 0:
            continue                               # 別の現場へ出動中(二重出動しない)
        key = ((float(agent.x) - tx) ** 2 + (float(agent.y) - ty) ** 2, int(agent.id))
        if best_key is None or key < best_key:
            best_key, best = key, agent
    return best


def _occupant(sim, building: str):
    """その建物の中に居る個体のうち id 最小(**実在の在館者だけ**・居なければ None)。"""
    best = None
    for agent in sim.agents:
        if agent.loc == "outside" or agent.sleeping:
            continue
        if str(getattr(agent, "building", "") or "") != str(building):
            continue
        if best is None or int(agent.id) < int(best.id):
            best = agent
    return best


# =========================================================================== #
# 故障の抽選(**装置ではなく世界が引く**。stream は 1 本・1 step に最大 2 draw)
# =========================================================================== #
def _fault_tick(sim, cfg: dict, st: dict, bud: _Budget, reg, step: int,
                sim_min: int) -> None:
    steps_per_day = max(1, int(sim.clock.steps_per_day))
    hazards = []
    total = 0.0
    for dev in reg:                                # ★device_id 昇順(DeviceRegistry)
        h = dev.hazard_per_step(steps_per_day)
        if h > 0.0:
            hazards.append((dev, h))
            total += h
    if total <= 0.0:
        return
    rng = sim.hub.stream("incident_facility", step)
    n_fault = int(rng.poisson(total))
    if n_fault <= 0:
        return
    for _ in range(n_fault):
        goal = float(rng.random()) * total
        acc = 0.0
        chosen = hazards[-1][0]
        for dev, h in hazards:
            acc += h
            if goal < acc:
                chosen = dev
                break
        _fault_one(sim, cfg, st, bud, chosen, step, sim_min)


def _fault_one(sim, cfg: dict, st: dict, bud: _Budget, dev, step: int,
               sim_min: int) -> None:
    """故障 1 件 → 閉じ込め → インターホン通報(行為)→ 対応者の出動(行為)。"""
    wear_before = round(float(dev.wear), 3)
    uses_before = int(dev.uses)
    since = dev.days_since_maint(sim_min)
    out = dev.on_input(-1, {"kind": "fault"}, sim_min)   # ★世界からの外部入力(δ_ext)
    if not out.get("faulted"):
        return                                     # 既に故障中(二重に落とさない)
    trapped = _occupant(sim, dev.building) if dev.traps else None
    try:
        nx, ny = sim.city.node_xy(dev.node)
    except Exception:                              # noqa: BLE001
        nx, ny = 0.0, 0.0
    bud.log(Event(step=step, sim_min=sim_min, agent_id=-1, kind="facility_fault",
                  x=nx, y=ny,
                  payload={"device_id": dev.device_id, "kind": dev.facility,
                           "building": dev.building, "node": dev.node,
                           "wear": wear_before, "uses": uses_before,
                           "days_since_maint": since,
                           "trapped": int(trapped is not None),
                           "repair_min": float(dev.repair_min)}),
             dev.device_id)
    st["faults"] += 1
    _bump(st["faults_by_kind"], dev.facility)
    st["wear_max"] = max(float(st["wear_max"]), wear_before)
    # ---- 通報(**この行為が対応の原因**)-------------------------------------- #
    caller = trapped if trapped is not None else _occupant(sim, dev.building)
    if caller is None:
        return                                     # 誰も居ない = 通報されない = 対応も無い
    if trapped is not None:
        st["trapped"] += 1
    bud.log(Event(step=step, sim_min=sim_min, agent_id=int(caller.id),
                  kind="facility_call", x=caller.x, y=caller.y,
                  payload={"device_id": dev.device_id, "building": dev.building,
                           "node": dev.node, "trapped": int(trapped is not None),
                           "self_call": bool(trapped is not None)}),
             dev.device_id)
    st["calls"] += 1
    caller.remember(TRAPPED_TEXT if trapped is not None else CALL_TEXT)
    # ---- 対応(通報に応える行為)---------------------------------------------- #
    responder = _responder(sim, cfg, dev.node)
    if responder is None:
        bud.log(Event(step=step, sim_min=sim_min, agent_id=-1,
                      kind="facility_dispatch", x=nx, y=ny,
                      payload={"device_id": dev.device_id, "node": dev.node,
                               "responder": -1, "response_min": None,
                               "repair_min": float(dev.repair_min),
                               "unstaffed": True}),
                 dev.device_id)
        st["unstaffed"] += 1
        return
    dist = ((float(responder.x) - nx) ** 2 + (float(responder.y) - ny) ** 2) ** 0.5
    response_min = round(dist / max(1.0, float(cfg["response_speed_m_per_min"])), 1)
    responder.facility_call_home = str(getattr(responder, "work_node", "") or "")
    responder.facility_call_until = int(step) + max(
        1, int(round(float(dev.repair_min) / max(1, int(sim.clock.step_minutes)))))
    responder.work_node = str(dev.node)
    bud.log(Event(step=step, sim_min=sim_min, agent_id=int(responder.id),
                  kind="facility_dispatch", x=responder.x, y=responder.y,
                  payload={"device_id": dev.device_id, "node": dev.node,
                           "responder": int(responder.id),
                           "response_min": response_min,
                           "repair_min": float(dev.repair_min),
                           "unstaffed": False}),
             dev.device_id)
    st["dispatches"] += 1
    responder.remember(RESPOND_TEXT)


def _restore_responders(sim, step: int) -> None:
    """対応者を持ち場へ戻す(``city_ops._ems_restore`` と同型・本 module の印だけ触る)。"""
    for agent in sim.agents:
        until = int(getattr(agent, "facility_call_until", -1))
        if until >= 0 and int(step) >= until:
            home = str(getattr(agent, "facility_call_home", "") or "")
            if home:
                agent.work_node = home
            agent.facility_call_until = -1


# =========================================================================== #
# 単一作用点(scheduler の唯一のフック)
# =========================================================================== #
def phase(sim, step: int, sim_min: int) -> None:
    """毎 step: δ_int(保守・待機摩耗・復旧)→ 利用(δ_ext)→ 故障抽選。

    **既定 OFF は即 return**(名簿も state も乱数も作らない = ゴールデン L1 バイト一致)。
    ★世界に対して触るのは (a) L1、(b) 記憶の定型 1 行、(c) 出動した対応者の持ち場だけ。
    """
    if not enabled(sim):
        return
    cfg = cfg_of(sim)
    reg = registry_of(sim)
    if not len(reg):
        return
    st = _state(sim)
    bud = _Budget(sim, st, int(cfg["max_events_per_step"]))
    day = int(sim_min) // 1440
    # ---- (1) δ_int: 宣言済みのスケジュール(保守・待機摩耗・復旧)-------------- #
    for dev in reg:
        out = dev.on_schedule(sim_min)
        if not out:
            continue
        if "maintained" in out:
            info = out["maintained"]
            worker = _responder(sim, cfg, dev.node)
            bud.log(Event(step=step, sim_min=sim_min,
                          agent_id=(int(worker.id) if worker is not None else -1),
                          kind="facility_maintenance",
                          x=(worker.x if worker is not None else 0.0),
                          y=(worker.y if worker is not None else 0.0),
                          payload={"device_id": dev.device_id, "kind": dev.facility,
                                   "wear_before": float(info["wear"]),
                                   "uses_before": int(info["uses"]),
                                   "staffed": bool(worker is not None),
                                   "day": day}), dev.device_id)
            st["maintenances"] += 1
        if "restored" in out:
            try:
                nx, ny = sim.city.node_xy(dev.node)
            except Exception:                      # noqa: BLE001
                nx, ny = 0.0, 0.0
            bud.log(Event(step=step, sim_min=sim_min, agent_id=-1,
                          kind="facility_restore", x=nx, y=ny,
                          payload={"device_id": dev.device_id, "node": dev.node,
                                   # ★実際に止まっていた分数(step 粒度の丸めを含む実測)。
                                   #   予定の repair_min ではなく**経過**を出す。
                                   "down_min": int(out["restored"]["down_min"]),
                                   "trapped": 0}), dev.device_id)
            st["restores"] += 1
    # ---- (2) δ_ext: エージェントの利用(摩耗の内生源)------------------------- #
    _apply_uses(sim, st, step, sim_min)
    # ---- (3) 故障の抽選(**世界が引き、装置は応答するだけ**)------------------ #
    _fault_tick(sim, cfg, st, bud, reg, step, sim_min)
    _restore_responders(sim, step)


# --------------------------------------------------------------------------- #
# checkpoint(**DEVS 状態を中央管理する** = resume 跨ぎ同値の要)
# --------------------------------------------------------------------------- #
def state_of(sim):
    """checkpoint 用の状態(既定 OFF / 未構築では None = 旧 checkpoint 互換)。"""
    reg = getattr(sim, "facilities", None)
    st = getattr(sim, "_facility_state", None)
    if reg is None and st is None:
        return None
    return {"schema": SCHEMA, "tally": st,
            "devices": ({d.device_id: d.snapshot() for d in reg}
                        if reg is not None else {})}


def restore_state(sim, blob) -> None:
    """checkpoint から状態を戻す(**名簿は捨てて組み直す** = 復元値で構築し直す)。"""
    if not blob:
        return
    tally = blob.get("tally")
    if tally is not None:
        sim._facility_state = tally
    sim._facility_restored = dict(blob.get("devices") or {})
    sim.facilities = None                          # 次の phase で復元値から組み直す
    sim._facility_index = None
    sim._facility_watermark = 0                    # fresh logger = 0 から再走査


def audit_report(sim) -> dict:
    """全設備の DEVS 遷移内訳(``illegal > 0`` = 契約違反)。OFF は空 dict。"""
    reg = getattr(sim, "facilities", None)
    if reg is None:
        return {}
    return {d.device_id: d.audit() for d in reg}


def provenance(sim) -> dict | None:
    """観測タリー(既定 OFF は None)。**実測 vs アンカー**を並べて出す。"""
    if not enabled(sim):
        return None
    cfg = cfg_of(sim)
    reg = getattr(sim, "facilities", None)
    n_ev = len(reg.of_kind(_devices.LIFT_PREFIX)) if reg is not None else 0
    out: dict = {"schema": SCHEMA, "devices": n_ev,
                 "elevator_fault_per_day": float(cfg[ELEVATOR]["fault_per_day"]),
                 "escalator_fault_per_day": float(cfg[ESCALATOR]["fault_per_day"])}
    if reg is not None:
        ev = sum(1 for d in reg if d.facility == ELEVATOR)
        out["elevators"] = ev
        out["escalators"] = len(reg) - ev
        # ★参照値: この世界の台数 × 1 台あたりのアンカー(実測と並べる分母)
        out["fault_reference_per_day"] = round(
            ev * float(cfg[ELEVATOR]["fault_per_day"])
            + (len(reg) - ev) * float(cfg[ESCALATOR]["fault_per_day"]), 5)
    st = getattr(sim, "_facility_state", None)
    if st is None:
        out.update({"faults": 0, "calls": 0, "dispatches": 0, "uses": 0,
                    "by_kind": {}, "dropped": 0})
        return out
    out.update({
        "uses": int(st["uses"]), "uses_blocked": int(st["uses_blocked"]),
        "faults": int(st["faults"]),
        "faults_by_kind": {k: int(v) for k, v in sorted(st["faults_by_kind"].items())},
        "trapped": int(st["trapped"]), "calls": int(st["calls"]),
        "dispatches": int(st["dispatches"]), "unstaffed": int(st["unstaffed"]),
        "restores": int(st["restores"]), "maintenances": int(st["maintenances"]),
        "wear_max": round(float(st["wear_max"]), 3),
        "dropped": int(st["dropped"]),
        "by_kind": {k: int(v) for k, v in sorted(st["by_kind"].items())}})
    return out
