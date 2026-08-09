"""装置(device)= **目標を持たないアクター**(actor model P2・``world.devices``・**既定 OFF**)。

正典
----
- docs/plans/actor-model-migration.md「P2: 世界に作用するものは全てアクターにする」
  (ユーザー承認済みの設計文書)
- 本 module が実装するのは、その中の **装置(device)** の層である。

何を解く問題か
--------------
現行のシムでは「世界の側の設備」がコードの散らばった場所で暗黙に振る舞っている。
改札は ``env.feedback`` の**集約近似**(1 step の流入件数が閾値を超えたら入場規制)として
だけ存在し、歩行者信号は ``world/sfm_core.SignalGate`` という**名前を持たないインスタンス**
(誰がどの交差点の信号なのかを外から指せない)として存在していた。どちらも
「世界に作用するもの」なのに、**同一性(id)も遷移の契約も持たない**。

装置の定義(P2 設計文書)
------------------------
    装置は目標を持たないアクターである。装置が状態を変えてよいのは
      (a) **宣言済みのスケジュール / 物理**によるとき   … DEVS の内部遷移 δ_int
      (b) **エージェントの行為への応答**であるとき      … DEVS の外部遷移 δ_ext
    のいずれかだけで、**自発的に(spontaneously)状態を変えてはならない**。

本 module はこの契約を型として与える(``Device.on_schedule`` / ``Device.on_input``)。
**契約を守っているかどうかを機械が検査できる**ことが要点なので、``Device`` は
δ の窓の外で状態属性(``STATE``)へ代入が起きたら**監査証跡に記録する**
(tests/test_devices.py が、わざと契約を破る派生クラスで検出を固定する)。

★**乱数は 1 本も引かない**(設計契約: 装置は決定論的な応答関数である)。
  本 module には乱数に相当する識別子が 1 つも存在しない
  (tests/test_devices.py が AST で機械固定する。traces.py の前例と同型)。

較正値(**公表値・実測ベース**)
-------------------------------
- 日本の駅の自動改札: **持続的に 60 人/分/通路**。IC カードのタッチは 0.2 秒だが、
  持続スループットを決めるのは人の歩行間隔なので **1 人あたり 1.0 秒**を既定にする。
- エスカレーターの実測は約 80 人/分(理論値・公称値は実測より 35〜60% 高く出る)。
  → ★**公称値(nameplate)は絶対に使わない**。本 module の既定はすべて実測系の値で、
    conf(``service_rate_per_min``)から差し替えられる。

3 つの設計判断(と、その理由)
------------------------------
**(1) 改札は ``env.feedback`` 規則2 の「粗い近似」を置き換える(二重適用しない)。**
    既存の規則2 は「1 step の駅流入件数 > 容量 なら数 step 入場規制」という**全体一括**の
    近似で、個体ごとの待ち時間を持たない。本 module の改札は
    **通路数 × 処理間隔の待ち行列**という細粒度モデルで、同じ物理(改札の処理能力飽和)を
    指している。両方を同時に効かせると同じ現象を二重に数えることになるので、
    ``world.devices.faregate.enabled`` が ON のとき ``envfeedback.update`` は
    **規則2 を評価しない**(``devices.faregate_active`` を 1 箇所で見る)。
    → 規則2 の状態(``gate_until`` / ``n_gate``)は 1 も動かない = テストで固定する。
    ★規則1(停車時間 → 遅延)と規則3(POI 占有)は別の物理なので**そのまま生きる**。

**(2) 待ちは「秒」で測り、step を跨ぐときだけ世界の遷移を遅らせる。**
    1 step = 600 秒 に対して改札の待ちは通常**秒オーダー**なので、小規模ランでは
    ``wait_s`` が記録されるだけで誰も遅れない(= 装置 ON が現実の粗視化を壊さない)。
    実規模(整備窓の定員 = 通路数 × ⌊step 秒 / 処理間隔⌋ を超える流入)で初めて
    「この step では通れない」= 既存の退出遷移(``_try_exit``)が遅れる。
    ★待ちの合計は ``max_hold_steps`` で頭打ち(= 必ず通す安全弁)。envfeedback の
      T5 非発散対策と同じ形にしてある(駅に人が溜まり続けて二度と捌けない発散を止める)。

**(3) L1 の量を原理的に抑える。**
    - ``gate_pass`` は **待ちが発生した通過だけ**(``wait_s > 0``)。定員に余裕がある
      ランでは 1 件も出ない。
    - ``device_load`` は **1 台 1 時間に最大 1 件**の要約。
    - ``signal_summary`` は **1 交差点 1 日に最大 1 件**。信号の周期は 140 秒で
      1 step(600 秒)より短いため、周期ごとのイベントを出すと L1 が溢れる
      (= 「毎サイクルの信号イベント」は**作らない**)。

R1 ドクトリン
-------------
- 既定 ``world.devices.enabled: false`` では ``phase`` / ``hold_exit`` /
  ``pending_exits`` / ``faregate_active`` が即 return し、**L1 に 1 件も出ず・
  sim に registry が生えず・agent に属性が生えず・プロンプトが 1 バイトも変わらない**
  (= ゴールデン L1 バイト一致。第96 traces / 第95 rumors と同じ「OFF ではキー自体を
  作らない」流儀)。
- **乱数を 1 本も引かない**(装置は決定論的な応答関数 = 設計契約)。
- ``generate()`` の呼び出しサイトを 1 つも作らない(LLM 追加呼ゼロ = k 非依存)。
- no-fingerprint: **プロンプトへ 1 バイトも足さない**。待たされたことは既存の
  「動けない」という世界の事実としてしか現れない(語彙も欄も増やさない)。

正直な限界(4 件)
------------------
- **駅は 1 ノード**: 現行地図はホーム / 改札 / コンコースを別ノードに持たないので、
  改札装置は「駅ノードの境界」に 1 台だけ置く(envfeedback の限界 1 と同じ)。
  実際の渋谷駅は改札口が複数あり、口ごとの混み方の違いは表現できない。
- **入場方向は待たせない**: 範囲外からの帰還(``enter_area``)は装置の定員を**消費する**
  (= 退出側の待ちを押し上げる)が、帰還そのものは遅らせない。帰還の遅れは
  envfeedback 規則1(電車の遅延)の担当で、そちらと二重にしないため。
- **step 内の到着時刻を持たない**: 1 step 内の到着は全て整備窓の先頭に並んだものとして
  扱う(= 待ちの上振れ側に倒す保守的な近似)。到着時刻の分布は物理層(P3/P4)の担当。
- **負荷カウンタは checkpoint に載せない**: ``device_load`` の時間内訳は resume 直後の
  1 時間だけ過少になる(装置の DEVS 状態そのものは整備窓ごとに 0 へ戻るので resume 安全)。

**絶対に触らない範囲(明文規約)**
--------------------------------
1. ``observer/metrics_spec.py`` の**凍結ファイル群**は 1 バイトも変更しない。
2. ``sfm_core.SignalGate`` の**時刻計算(位相・青 / 青点滅 / 赤の窓)は 1 バイトも
   変更しない**。本 module が足すのは **同一性(device_id)と観測**だけである。
"""
from __future__ import annotations

from .observer.schema import Event, register_event_kind

SCHEMA = 1

# --------------------------------------------------------------------------- #
# L1 イベント種の**材料側**登録(traces.py / rumors.py と同じ「自分の module で
# 宣言する」流儀。observer/schema.py には 1 バイトも書かない)。
# --------------------------------------------------------------------------- #
register_event_kind(
    "gate_pass",
    "改札装置の通過。**待ちが発生した通過だけ**記録する(wait_s>0)"
    " {device_id, wait_s, queue_len, dir}")
register_event_kind(
    "device_load",
    "装置 1 台の負荷要約。**1 台 1 時間に最大 1 件**(毎 step の信号は出さない)"
    " {device_id, kind, passed, held, wait_s_max, wait_s_mean, queue_max, n_gates}")
register_event_kind(
    "signal_summary",
    "歩行者信号装置の周期パラメータ。**1 交差点 1 日に最大 1 件**"
    " {device_id, crossing_id, cycle_s, green_s, flash_s, offset_s, green_ratio}")

# 監査証跡に残す遷移レコードの上限(1 台あたり)。件数のタリーは上限なしで数える。
AUDIT_MAX = 64

# 装置 id の接頭辞(**唯一の定義**。解析側はここだけを見れば装置種が判る)
FAREGATE_PREFIX = "faregate"
SIGNAL_PREFIX = "signal"


# =========================================================================== #
# DEVS 契約: 装置の基底
# =========================================================================== #
class Device:
    """装置 1 台。**目標を持たないアクター**の基底(DEVS の 2 遷移だけを口として持つ)。

    状態を変えてよい経路は 2 つだけ:
      - ``on_schedule(sim_min)``  … δ_int(**宣言済みのスケジュール / 物理**による内部遷移)
      - ``on_input(actor_id, input, sim_min)`` … δ_ext(**エージェントの行為への応答**)

    派生クラスは ``STATE`` に「δ でしか動かしてよくない状態属性名」を宣言し、
    ``_delta_int`` / ``_delta_ext`` を実装する。``seal()`` を呼んだ後は、
    **δ の窓の外で ``STATE`` の属性へ代入が起きると監査証跡に illegal として残る**
    (= 自発的遷移の検出。tests/test_devices.py がわざと契約を破る派生で固定する)。

    ★監査は「禁止」ではなく「記録」である: 実行時に例外で落とすと、装置の契約違反が
      本番ランを止めてしまう。契約は**テストで落とす**もので、ランでは観測に残す。
    """

    kind: str = "device"
    STATE: tuple[str, ...] = ()

    def __init__(self, device_id: str):
        # ★ここでの代入は「構築」であって遷移ではない。seal() まで監査は始まらない。
        self.device_id = str(device_id)
        self.transitions: list[dict] = []
        # 観測タリー(DEVS の状態ベクトルではない = 世界の因果に効かない)
        self.audit_counts: dict[str, int] = {"int": 0, "ext": 0, "illegal": 0}
        self.load: dict[str, float] = _new_load()
        self.__dict__["_in_delta"] = False
        self.__dict__["_sealed"] = False

    # ---- 構築の終わり(ここから監査が始まる)------------------------------- #
    def seal(self) -> "Device":
        self.__dict__["_sealed"] = True
        return self

    # ---- 自発的遷移の検出(契約の機械化)---------------------------------- #
    def __setattr__(self, name: str, value) -> None:
        if (name in type(self).STATE and self.__dict__.get("_sealed")
                and not self.__dict__.get("_in_delta")):
            self._record("illegal", -1, name)
        object.__setattr__(self, name, value)

    def _record(self, path: str, sim_min: int, attr: str | None) -> None:
        """遷移 1 件を監査証跡へ(件数は無制限に数え、明細は AUDIT_MAX 件で打ち切る)。"""
        counts = self.__dict__["audit_counts"]
        counts[path] = int(counts.get(path, 0)) + 1
        trail = self.__dict__["transitions"]
        if len(trail) < AUDIT_MAX:
            trail.append({"path": path, "sim_min": int(sim_min), "attr": attr})

    # ---- δ_int: 宣言済みスケジュール / 物理による内部遷移 ------------------ #
    def on_schedule(self, sim_min: int):
        """装置の**宣言済みスケジュール**を 1 tick 進める(自発的遷移ではない)。"""
        self.__dict__["_in_delta"] = True
        try:
            out = self._delta_int(int(sim_min))
        finally:
            self.__dict__["_in_delta"] = False
        if out:
            self._record("int", sim_min, None)
        return out

    # ---- δ_ext: エージェントの行為への応答 -------------------------------- #
    def on_input(self, actor_id: int, input: dict, sim_min: int) -> dict:  # noqa: A002
        """エージェント ``actor_id`` の行為へ応答する(**決定論の応答関数**)。"""
        self.__dict__["_in_delta"] = True
        try:
            out = self._delta_ext(int(actor_id), dict(input or {}), int(sim_min))
        finally:
            self.__dict__["_in_delta"] = False
        self._record("ext", sim_min, None)
        return out

    # ---- 派生が実装する 2 つの遷移 ---------------------------------------- #
    def _delta_int(self, sim_min: int):
        return False

    def _delta_ext(self, actor_id: int, input: dict, sim_min: int) -> dict:  # noqa: A002
        return {}

    # ---- 監査(テスト・観測が読む)---------------------------------------- #
    def audit(self) -> dict:
        """遷移の内訳。``illegal > 0`` = **契約違反(自発的遷移)が起きた**。"""
        return dict(self.audit_counts)

    def __repr__(self) -> str:                       # pragma: no cover - 診断用
        return f"<{type(self).__name__} {self.device_id}>"


def _new_load() -> dict:
    """観測タリー(1 時間ぶん)。``device_load`` を出すたびに畳む。"""
    return {"passed": 0, "held": 0, "wait_sum": 0.0, "wait_max": 0.0,
            "queue_max": 0, "passed_total": 0, "held_total": 0, "hour": -1}


# =========================================================================== #
# 装置レジストリ(id → 装置)
# =========================================================================== #
class DeviceRegistry:
    """装置の名簿。**反復は device_id 昇順**(決定論・乱数ゼロ)。"""

    def __init__(self):
        self._by_id: dict[str, Device] = {}

    def add(self, device: Device) -> Device:
        if device.device_id in self._by_id:
            raise ValueError(f"device_id が重複している: {device.device_id}")
        self._by_id[device.device_id] = device.seal()
        return device

    def get(self, device_id: str) -> Device | None:
        return self._by_id.get(str(device_id))

    def of_kind(self, kind: str) -> tuple:
        return tuple(d for d in self if d.kind == str(kind))

    def ids(self) -> tuple:
        return tuple(sorted(self._by_id))

    def __iter__(self):
        for did in sorted(self._by_id):
            yield self._by_id[did]

    def __len__(self) -> int:
        return len(self._by_id)


# =========================================================================== #
# 改札(faregate)= 通路の配列 + FIFO 待ち行列
# =========================================================================== #
def faregate_device_id(node: str) -> str:
    """駅ノード → 改札装置 id(**安定 id**。ランを跨いでも同じ名前になる)。"""
    return f"{FAREGATE_PREFIX}:{node}"


class FaregateArray(Device):
    """改札の**通路配列** 1 台(1 駅ゲートウェイに 1 台)。

    モデル
    ------
    - 通路 ``n_gates`` 本。1 本あたりの処理間隔は ``headway_s = 60 / service_rate_per_min``
      (既定 60 人/分/通路 = **1.0 秒/人** = 日本の駅改札の公表持続スループット)。
    - 整備窓(service window)= **世界の 1 step**。窓の定員は
      ``n_gates × ⌊step 秒 / headway_s⌋``(既定 Δt=10 分・8 通路で 4800 人/step)。
    - 到着順 ``k``(0 起点)の乗客は ``k % n_gates`` 番の通路に付き、
      その通路で ``k // n_gates`` 人ぶん待つ → ``wait_s = (k // n_gates) × headway_s``。
      120 人 / 8 通路 / 1.0 秒 なら最後の 1 人の待ちは 14.0 秒、掃けきるのは 15.0 秒。

    決定論
    ------
    到着順は「**到着 step → agent_id 昇順**」。scheduler の退出フェーズが
    agent_id 昇順で個体を回すので、到着順はそのまま id 昇順になる(乱数ゼロ)。

    DEVS
    ----
    - δ_int(``_delta_int``)= **整備窓の更新**。窓が変わったら ``served`` を 0 に戻す。
      これは「宣言済みのスケジュール」であって自発的遷移ではない。
    - δ_ext(``_delta_ext``)= **1 人の通過要求への応答**。定員内なら 1 枠消費して
      待ち時間を返し、定員超過なら**状態を 1 つも動かさず**「この窓では通れない」を返す。
    """

    kind = FAREGATE_PREFIX
    STATE = ("window", "served")

    def __init__(self, device_id: str, n_gates: int = 8,
                 service_rate_per_min: float = 60.0, window_min: int = 10,
                 window_seconds: float = 600.0):
        super().__init__(device_id)
        self.n_gates = max(1, int(n_gates))
        self.headway_s = 60.0 / max(1e-9, float(service_rate_per_min))
        self.window_min = max(1, int(window_min))
        self.window_seconds = max(self.headway_s, float(window_seconds))
        # 1 整備窓(= 1 step)の定員 = 通路数 × その窓で 1 通路が捌ける人数
        self.capacity = self.n_gates * max(1, int(self.window_seconds
                                                  // self.headway_s))
        # ---- DEVS 状態(δ でしか動かさない)----
        self.window = -1                 # いま開いている整備窓 = sim_min // window_min
        self.served = 0                  # その窓でこれまでに捌いた人数
        self.seal()

    # ---- δ_int: 整備窓の更新(宣言済みスケジュール)------------------------ #
    def _delta_int(self, sim_min: int) -> bool:
        win = int(sim_min) // self.window_min
        if win == self.window:
            return False
        self.window = win
        self.served = 0
        return True

    # ---- δ_ext: 1 人の通過要求への応答 ------------------------------------ #
    def _delta_ext(self, actor_id: int, input: dict, sim_min: int) -> dict:  # noqa: A002
        if self.served >= self.capacity:
            # 定員超過 = この窓では捌けない。**状態は 1 つも動かさない**
            # (次の窓へ持ち越すのは装置ではなく個体側の待ち)。
            return {"wait_s": float(self.window_seconds),
                    "queue_len": int(self.served), "passed": False,
                    "device_id": self.device_id}
        k = int(self.served)
        self.served = k + 1
        return {"wait_s": (k // self.n_gates) * self.headway_s,
                "queue_len": k, "passed": True, "device_id": self.device_id}


# =========================================================================== #
# 歩行者信号(SignalGate)= 状態を持たない固定スケジュール装置
# =========================================================================== #
def signal_device_id(crossing_id) -> str:
    """交差点表の id → 信号装置 id(**安定 id**。zones テーブルの crossing_id 由来)。"""
    return f"{SIGNAL_PREFIX}:{int(crossing_id)}"


class SignalDevice(Device):
    """歩行者信号 1 基。``sfm_core.SignalGate`` に**同一性を与えるだけ**の薄い装置。

    ★時刻計算(位相・青 / 青点滅 / 赤の窓)は ``SignalGate`` のものを**1 バイトも
      変更せずに**使う(既存の tests/test_crossing_signals.py が無修正で緑であることが
      その証拠)。本クラスが足すのは device_id と DEVS の口だけである。

    DEVS 上の位置づけ: 位相は**絶対時刻の純関数** φ(t) = (t + offset) mod C なので、
    この装置は **記憶を持たない**(``STATE`` が空)。δ_int は「現示が変わったか」を
    返すだけで、状態ベクトルは 1 つも無い = 固定スケジュール装置の理想形。
    """

    kind = SIGNAL_PREFIX
    STATE = ()                       # ★状態を持たない(位相は絶対時刻の純関数)

    def __init__(self, device_id: str, params: dict):
        super().__init__(device_id)
        from .world.sfm_core import SignalGate      # 遅延 import(numpy を持ち込まない)
        self.params = {k: float(v) for k, v in dict(params or {}).items()
                       if k in ("cycle_s", "green_s", "flash_s", "offset_s")}
        self.crossing_id = int(dict(params or {}).get("crossing_id", 0))
        self.gate = SignalGate(device_id=device_id, **self.params)
        self.seal()

    # ---- δ_int: 現示(青 / 赤)は絶対時刻の純関数 -------------------------- #
    def _delta_int(self, sim_min: int) -> bool:
        return False                 # 記憶が無いので「遷移した」ことすら残らない

    # ---- δ_ext: 横断してよいかの応答(状態は動かさない)-------------------- #
    def _delta_ext(self, actor_id: int, input: dict, sim_min: int) -> dict:  # noqa: A002
        sec = float(dict(input or {}).get("sim_sec", float(sim_min) * 60.0))
        return {"can_cross": bool(self.gate.can_cross(sec)),
                "device_id": self.device_id}

    def green_ratio(self) -> float:
        """1 周期のうち横断できる割合(青 + 青点滅)。周期パラメータの純関数。"""
        cyc = float(self.params.get("cycle_s", 0.0))
        if cyc <= 0.0:
            return 0.0
        return round((float(self.params.get("green_s", 0.0))
                      + float(self.params.get("flash_s", 0.0))) / cyc, 4)


# =========================================================================== #
# cfg 正準化(traces.build_cfg / rumors.build_cfg と同型: dict / OmegaConf 両対応)
# =========================================================================== #
DEFAULTS = {
    "enabled": False,
    "signal_events": False,
    "faregate": {
        "enabled": False,
        "n_gates": 8,                    # 地図データに改札口の本数が無いときの既定
        "service_rate_per_min": 60.0,    # ★実測系の公表値(公称値は使わない)
        "max_hold_steps": 3,             # 1 回の退出で待たせる上限(必ず通す安全弁)
    },
}


def _to_plain(raw):
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(raw):
            return OmegaConf.to_container(raw, resolve=True)
    except Exception:                              # noqa: BLE001(旧 config 互換)
        pass
    return raw


def build_cfg(raw) -> dict:
    """conf の ``world.devices`` ブロックを正準化(既定 OFF=現行挙動と完全同一)。

    dotlist 上書きは文字列で入り得るため型強制する(``traces.build_cfg`` と同じ作法)。
    未知のキーは黙って捨てる(捏造しない)。
    """
    raw = dict(_to_plain(raw) or {})
    cfg = {"enabled": False, "signal_events": False,
           "faregate": dict(DEFAULTS["faregate"])}
    for key, val in raw.items():
        if key == "enabled":
            cfg["enabled"] = bool(val)
        elif key == "signal_events":
            cfg["signal_events"] = bool(val)
        elif key == "faregate":
            got = dict(_to_plain(val) or {})
            fg = cfg["faregate"]
            if "enabled" in got:
                fg["enabled"] = bool(got["enabled"])
            if "n_gates" in got:
                fg["n_gates"] = max(1, int(got["n_gates"]))
            if "service_rate_per_min" in got:
                fg["service_rate_per_min"] = max(1e-9,
                                                 float(got["service_rate_per_min"]))
            if "max_hold_steps" in got:
                fg["max_hold_steps"] = max(0, int(got["max_hold_steps"]))
    return cfg


def cfg_of(sim) -> dict:
    """装置設定(初回のみ ``sim.cfg.world.devices`` から遅延構築してキャッシュ)。

    simulation.py は読み取り専用なので cfg は本 module が遅延構築する
    (``traces.cfg_of`` / ``rumors.cfg_of`` と同型)。キャッシュ属性 ``sim.devicescfg`` は
    (``sim.devcfg`` は observer/deviation.py が既に使っている名前なので避ける)
    L1/L2/L3/乱数に一切現れない = 既定 OFF のバイト一致を壊さない。
    """
    got = getattr(sim, "devicescfg", None)
    if got is None:
        try:
            raw = (sim.cfg.get("world", None) or {}).get("devices", None)
        except Exception:                          # noqa: BLE001(旧 config 互換)
            raw = None
        got = build_cfg(raw)
        sim.devicescfg = got
    return got


def enabled(sim) -> bool:
    """装置層が有効か。既定 OFF=新経路を一切通らない(バイト一致)。"""
    return bool(cfg_of(sim)["enabled"])


def faregate_active(sim) -> bool:
    """改札装置が世界に効いているか。

    ★``envfeedback.update`` が**この 1 関数だけ**を見て規則2(入場規制)を評価するか
      決める(細粒度モデルと集約近似の二重適用の防止)。既定 OFF は常に False =
      envfeedback は従来どおり = バイト一致。
    """
    cfg = cfg_of(sim)
    return bool(cfg["enabled"] and cfg["faregate"]["enabled"])


# =========================================================================== #
# レジストリの構築(**ON 経路でのみ生やす**)
# =========================================================================== #
def _signal_zones(sim) -> tuple:
    """信号パラメータを持つ物理ゾーン(並びはゾーン宣言順 = 決定論)。"""
    try:
        zones = sim.physcfg["zones"]
    except Exception:                              # noqa: BLE001(physics OFF / 旧 cfg)
        return ()
    return tuple(z for z in zones if getattr(z, "signal", None))


def build_registry(sim) -> DeviceRegistry:
    """conf と地図から装置の名簿を組む(**OFF では空の名簿**)。

    ★遅延構築にしている理由: 既定 OFF で ``Simulation.__init__`` に 1 バイトも
      触らないため(sim に属性が生えない = ゴールデン L1 バイト一致の最も硬い形)。
      ON では最初の ``phase()`` で 1 度だけ組まれ、以後は同じ名簿を使う。
    """
    reg = DeviceRegistry()
    cfg = cfg_of(sim)
    if not cfg["enabled"]:
        return reg
    fg = cfg["faregate"]
    city = getattr(sim, "city", None)
    station = getattr(city, "station_node", None) if city is not None else None
    if fg["enabled"] and station:
        clock = getattr(sim, "clock", None)
        reg.add(FaregateArray(
            faregate_device_id(station), n_gates=fg["n_gates"],
            service_rate_per_min=fg["service_rate_per_min"],
            window_min=int(getattr(clock, "step_minutes", 10) or 10),
            window_seconds=float(getattr(clock, "step_seconds", 600.0) or 600.0)))
    for zone in _signal_zones(sim):
        sig = dict(zone.signal)
        did = signal_device_id(sig.get("crossing_id", 0))
        if reg.get(did) is None:               # 同一交差点を複数ゾーンが指しても 1 台
            reg.add(SignalDevice(did, sig))
    return reg


def registry_of(sim) -> DeviceRegistry:
    """sim の装置名簿(初回のみ構築)。**OFF では呼ばれない**(phase が先に return)。"""
    reg = getattr(sim, "devices", None)
    if reg is None:
        reg = build_registry(sim)
        sim.devices = reg
    return reg


def faregate_of(sim):
    """この世界の改札装置(無ければ None)。"""
    if not faregate_active(sim):
        return None
    found = registry_of(sim).of_kind(FAREGATE_PREFIX)
    return found[0] if found else None


# =========================================================================== #
# 作用点(1): step 冒頭の δ_int + 入場方向の δ_ext + 要約の記録
# =========================================================================== #
def _new_events(sim) -> list:
    """前回処理済み総数(watermark)以降の新規 L1(``traces._new_events`` と同型)。

    watermark は**プロセス内 logger カウンタ由来**なので checkpoint に保存しない
    (resume 直後は新しい logger の 0 から数え直す = 第59 assets / 第61 gossip と同流儀)。
    """
    logger = getattr(sim, "logger", None)
    if logger is None:
        return []
    events = logger.events
    total = int(getattr(logger, "_n_flushed", 0)) + len(events)
    processed = int(getattr(sim, "_dev_watermark", 0))
    fresh = max(0, min(total - processed, len(events)))
    sim._dev_watermark = total
    return events[len(events) - fresh:] if fresh else []


def _log_gate_pass(sim, agent_id: int, device, out: dict, direction: str,
                   step: int, sim_min: int, x: float, y: float) -> None:
    """``gate_pass`` 1 件(**待ちが出た通過だけ**。L1 の量を原理的に抑える)。"""
    sim.logger.log(Event(step=int(step), sim_min=int(sim_min),
                         agent_id=int(agent_id), kind="gate_pass", x=float(x),
                         y=float(y),
                         payload={"device_id": device.device_id,
                                  "wait_s": round(float(out["wait_s"]), 3),
                                  "queue_len": int(out["queue_len"]),
                                  "dir": direction}))


def _tally(device, out: dict) -> None:
    """観測タリーの更新(**DEVS の状態ベクトルではない** = 世界の因果に効かない)。"""
    load = device.load
    wait = float(out["wait_s"])
    if out["passed"]:
        load["passed"] = int(load["passed"]) + 1
        load["passed_total"] = int(load["passed_total"]) + 1
        load["wait_sum"] = float(load["wait_sum"]) + wait
        load["wait_max"] = max(float(load["wait_max"]), wait)
    else:
        load["held"] = int(load["held"]) + 1
        load["held_total"] = int(load["held_total"]) + 1
    load["queue_max"] = max(int(load["queue_max"]), int(out["queue_len"]))


def _flush_load(sim, device, step: int, sim_min: int) -> None:
    """``device_load`` を **1 台 1 時間に最大 1 件**出して時間内訳を畳む。"""
    hour = int(sim_min) // 60
    load = device.load
    if int(load["hour"]) < 0:                      # 初回 = 基準時刻を置くだけ
        load["hour"] = hour
        return
    if hour == int(load["hour"]):
        return
    passed, held = int(load["passed"]), int(load["held"])
    if passed or held:
        sim.logger.log(Event(
            step=int(step), sim_min=int(sim_min), agent_id=-1,
            kind="device_load", x=0.0, y=0.0,
            payload={"device_id": device.device_id, "kind": device.kind,
                     "passed": passed, "held": held,
                     "wait_s_max": round(float(load["wait_max"]), 3),
                     "wait_s_mean": (round(float(load["wait_sum"]) / passed, 3)
                                     if passed else 0.0),
                     "queue_max": int(load["queue_max"]),
                     "n_gates": int(getattr(device, "n_gates", 0)),
                     "hour": int(load["hour"])}))
    load.update({"passed": 0, "held": 0, "wait_sum": 0.0, "wait_max": 0.0,
                 "queue_max": 0, "hour": hour})


def _signal_summaries(sim, reg, step: int, sim_min: int) -> None:
    """``signal_summary`` を **1 交差点 1 日に最大 1 件**(周期ごとには絶対に出さない)。

    ★信号のサイクルは 140 秒で 1 step(600 秒)より短い。周期ごとのイベントを出すと
      L1 が信号だけで埋まるので、出すのは「その日その交差点はどんな周期だったか」の
      1 行だけにする(現示そのものは絶対時刻の純関数なので事後に再構成できる)。
    """
    day = int(sim_min) // 1440
    seen = getattr(sim, "_dev_signal_day", None)
    if seen is None:
        seen = {}
        sim._dev_signal_day = seen
    for device in reg.of_kind(SIGNAL_PREFIX):
        if int(seen.get(device.device_id, -1)) == day:
            continue
        seen[device.device_id] = day
        payload = {"device_id": device.device_id,
                   "crossing_id": int(device.crossing_id),
                   "green_ratio": device.green_ratio(), "day": day}
        payload.update({k: round(float(v), 3)
                        for k, v in sorted(device.params.items())})
        sim.logger.log(Event(step=int(step), sim_min=int(sim_min), agent_id=-1,
                             kind="signal_summary", x=0.0, y=0.0,
                             payload=payload))


def phase(sim, step: int, sim_min: int) -> None:
    """装置層の単一作用点。**既定 OFF は即 return**(module も hot path も触らない)。

    置き場所は ``_phase_move`` の**直前**(``physics.phase`` の隣)。理由:
      - この時点で ``_phase_wake_and_returns`` の帰還(``enter_area``)は済んでおり、
        入場方向が**先に**改札の定員を消費する(現実の因果順と同じ)。
      - この step の退出(``_try_exit`` / 再試行)は**この後**なので、同じ整備窓の
        残りの定員を消費する。
    ★世界状態(所持金・位置・関係・drive・opinion)は 1 つも動かさない。
    """
    if not enabled(sim):
        return
    reg = registry_of(sim)
    if not len(reg):
        return
    # ---- (1) δ_int: 宣言済みスケジュール(整備窓の更新)-------------------- #
    for device in reg:
        device.on_schedule(sim_min)
    # ---- (2) 観測: 1 台 1 時間の負荷要約 / 1 交差点 1 日の信号要約 --------- #
    for device in reg:
        _flush_load(sim, device, step, sim_min)
    if cfg_of(sim)["signal_events"]:
        _signal_summaries(sim, reg, step, sim_min)
    # ---- (3) δ_ext: 入場方向(この step の帰還)が改札の定員を消費する ----- #
    gate = faregate_of(sim)
    if gate is None:
        return
    city = getattr(sim, "city", None)
    station = getattr(city, "station_node", None) if city is not None else None
    if not station:
        return
    for e in _new_events(sim):
        if e.kind != "enter_area":
            continue
        payload = e.payload or {}
        if payload.get("via") != "train" or payload.get("gateway") != station:
            continue
        out = gate.on_input(int(e.agent_id), {"kind": "pass", "dir": "enter"},
                            sim_min)
        _tally(gate, out)
        if float(out["wait_s"]) > 0.0 and out["passed"]:
            _log_gate_pass(sim, e.agent_id, gate, out, "enter", step, sim_min,
                           e.x, e.y)


# =========================================================================== #
# 作用点(2): 駅から範囲外へ出る = 改札を通る(scheduler._try_exit の seam)
# =========================================================================== #
def _release(agent) -> None:
    agent._dev_gate_wait = -1
    agent._dev_gate_held = 0


def hold_exit(sim, agent, step: int, sim_min: int) -> bool:
    """改札を通れるか。True = この step は通れない(駅に留まる)。

    ``envfeedback.hold_exit`` と**同じ形**(前回の待ちの継続判定 → 上限 → 判定 →
    記録)にしてあり、上限 ``max_hold_steps`` を超えて待たせない(= 必ず通す安全弁。
    駅に人が溜まり続けて二度と捌けない発散を原理的に止める)。

    ★``envfeedback`` の規則2(入場規制)とは**二重に効かない**: 改札装置が ON の
      ランでは ``envfeedback.update`` が規則2 を評価しない(``faregate_active``)。
    ★OFF では 1 バイトも触らない(agent に属性すら生えない)= バイト一致。
    """
    gate = faregate_of(sim)
    if gate is None:
        return False
    prev = int(getattr(agent, "_dev_gate_wait", -1))
    if prev > step:
        return True                                # 待機中(記録は開始時に 1 回だけ)
    if prev >= 0 and step > prev:
        _release(agent)                            # 前回の待ちと連続していない別トリップ
    budget = (int(cfg_of(sim)["faregate"]["max_hold_steps"])
              - int(getattr(agent, "_dev_gate_held", 0)))
    if budget <= 0:
        _release(agent)
        return False                               # 安全弁: 必ず通す
    out = gate.on_input(int(agent.id), {"kind": "pass", "dir": "exit"}, sim_min)
    _tally(gate, out)
    if out["passed"]:
        if float(out["wait_s"]) > 0.0:
            _log_gate_pass(sim, agent.id, gate, out, "exit", step, sim_min,
                           agent.x, agent.y)
        _release(agent)
        return False
    agent._dev_gate_wait = step + 1
    agent._dev_gate_held = int(getattr(agent, "_dev_gate_held", 0)) + 1
    return True


def pending_exits(sim, step: int) -> tuple:
    """改札待ちで駅に留まっている個体のうち、この step に再試行してよい者。

    ``_try_exit`` は「到着した step」にしか呼ばれないので、待たせた個体には再試行の
    口が要る(``envfeedback.pending_exits`` と同じ理由・同じ形)。OFF では空 tuple
    (呼び出し側のループが 0 回 = バイト一致)。

    ★``envfeedback`` にも待たされている個体は**返さない**: あちらの再試行ループが
      先に ``_try_exit`` を呼び、その中で本 module の ``hold_exit`` も評価されるので、
      ここで返すと同じ step に ``_try_exit`` が 2 回走る(退出の二重適用)。
    """
    gate = faregate_of(sim)
    if gate is None:
        return ()
    city = getattr(sim, "city", None)
    station = getattr(city, "station_node", None) if city is not None else None
    if not station:
        return ()
    out = []
    for agent in sim.agents:
        if (agent.loc == "street" and agent.exit_intent and not agent.sleeping
                and not agent.route and agent.node == station
                and int(getattr(agent, "_dev_gate_held", 0)) > 0
                and int(getattr(agent, "_dev_gate_wait", -1)) <= step
                and int(getattr(agent, "_env_exit_held", 0)) == 0):
            out.append(agent)
    return tuple(out)


# =========================================================================== #
# 観測ヘルパ(読むだけ・state も属性も生やさない)
# =========================================================================== #
def audit_report(sim) -> dict:
    """全装置の遷移内訳(``illegal > 0`` = 契約違反)。OFF は空 dict。"""
    reg = getattr(sim, "devices", None)
    if reg is None:
        return {}
    return {d.device_id: d.audit() for d in reg}
