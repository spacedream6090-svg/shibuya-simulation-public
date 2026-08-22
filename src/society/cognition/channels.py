"""観測チャンネル o_c(t)(第80バッチ 2026-08-01)= 驚き駆動発火の**入力側**の定義。

正典: docs/plans/source/cognition-design-record.md §2.3-2.4 / §3.2 /
      docs/plans/cognition-physics-plan.md §4 第80行。

何を解く問題か
--------------
設計 §2.3 の発火判定式

    S_i(t) = Σ_c  g_ic · |o_c(t) − ô_ic| / σ_c  +  Σ_j  w_ij · [trigger_j 成立]

を実装するには、まず **o_c(t) が何であるか**が 1 箇所に確定していなければならない。
本 module はその `o_c` の**唯一の定義**であり、

  1. チャンネル一覧(id / 源 / 単位 / 実装済みか)を **データとして**持ち、
  2. 1 step 分の値を **決定論・乱数ゼロ・LLM ゼロ・読み取り専用**で計算し、
  3. その値を返すだけ(**世界には一切書き戻さない**)

の 3 つしかしない。σ_c による正規化(§2.3「これで割ることが必須」)は
`cognition/calib.py` が持つ凍結ファイルの担当で、本 module は生値だけを出す。

驚きの源は 3 種類(設計 §2.4)
-----------------------------
  外界     … 局所混雑度・遭遇数・受信発話数・サイネージ・環境情報(遅延/天候の当日値)
  身体     … 個体の内部ゲージ群。単調ドリフトによる自発的発火の源
             (環境が静かでも身体側の偏差が閾値を超えるので「静かな環境で凍る」を回避する)
  予測不成立 … 「誰かが来るはず」が起きないこと。**本バッチでは枠だけ**(下記)

★予測不成立チャンネル `pred.unmet` は `implemented=False` で登録してあり、**値は常に
  null**(未使用)。期待値 ô を LLM が出すのは第82バッチ、S を組むのは第81バッチなので、
  ここで仮の値を入れると「実装済みの数値」と誤読される。列だけ先に生やして後続の
  スキーマ変更をゼロにするのが本バッチの役割(= 枠の確保)。

正直な限界(値が null になる条件)
----------------------------------
チャンネルの**元になる機構が OFF のランでは値を捏造せず null にする**。
例: 天候ブロックが無効なランに気温は存在しない / 退屈ゲージ機構が無効なランに
その値は存在しない。0.0 で埋めると σ_c が「定数チャンネル」として測れてしまい、
第81 が実在しない偏差で発火しうる。**欠測は欠測と判る値にする**(本リポジトリの既定則)。

R1 ドクトリンとの関係
---------------------
- 既定 `cognition.channels.enabled: false` では本 module は **1 度も呼ばれない**
  (Simulation がサイドカーを作らず scheduler の分岐に入らない)= L1/L2/L3・乱数・
  LLM 呼数のいずれも不変。
- ON でも **観測サイドカー(channels.parquet)へ書くだけ**。シム本体はこの値を読まない
  (第77 live_viewer / 第78 state_hash と同じ「観測はシムを変えない」構造)。
- 乱数を 1 本も引かない(RngHub に触れない)。LLM を 1 本も呼ばない。
- 本 module は因子名を 1 つも綴らない: 内部状態ゲージのチャンネル id は
  `factors` 層が持つ初期値表の**キーから実行時に生成**する(no-fingerprint 契約)。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from .. import physics as _physics
from ..world.perception import build_index, count_hearers

SCHEMA = 1

# 源(source)。設計 §2.4 の 3 分類。
EXTERNAL = "external"
BODY = "body"
PREDICTION = "prediction"
SOURCES = (EXTERNAL, BODY, PREDICTION)


@dataclass(frozen=True)
class Channel:
    """観測チャンネル 1 本の宣言(= データ。コードに散らさない)。"""

    id: str
    source: str
    unit: str
    description: str
    implemented: bool = True
    # ★ プロンプトに出してよい**短い中立ラベル**(第82バッチ watch spec 用)。
    #   `id` は body.state.* が factors 層のキー(= 因子名)を含むので**プロンプトへ出せない**。
    #   ここには因子名も評価語も入れない(no-fingerprint 契約)。`spec_sha256` の対象外なので
    #   ラベルを足しても既存の σ_c 凍結ファイルは無効化されない。
    label: str = ""

    @property
    def column(self) -> str:
        """Parquet の列名(ドットは SQL/pandas で扱いにくいのでアンダースコアへ)。"""
        return self.id.replace(".", "_")


# --------------------------------------------------------------------------- #
# チャンネル一覧
#
# ★並び順が Parquet の列順・σ_c ファイルの並び順・observe() の返り値の並び順の
#   **唯一の源**。足すときは末尾に足し、SCHEMA を上げること(過去ランと σ が比較できなくなる)。
# --------------------------------------------------------------------------- #
_EXTERNAL: tuple[Channel, ...] = (
    Channel("ext.crowd_local", EXTERNAL, "count",
            "局所混雑度 = 同じ場所(路上は同一ノード / 屋内は同一建物の同一階)に居る"
            "自分以外の人数。範囲外に居るときは 0(誰とも同席していない)",
            label="まわりの人の多さ"),
    Channel("ext.encounter", EXTERNAL, "count",
            "遭遇数 = 知覚半径(world.perception_radius_m)の内側に居る**起きている**他者の数。"
            "判定は world/perception.py と同一(同一文脈+距離+視線遮蔽が据えてあれば LOS)",
            label="近くにいる人の数"),
    Channel("ext.heard", EXTERNAL, "count",
            "受信発話数 = この step に自分が聞いた対面発話(L1 kind='hear')+ 受け取った"
            "1対1メッセージ(L1 kind='dm' の payload.to が自分)の件数",
            label="耳や画面に入る話の量"),
    Channel("ext.signage", EXTERNAL, "count",
            "街頭掲示の視認件数 = この step に自分が見た掲出(L1 kind='ad_exposure')の件数",
            label="目に入る貼り紙・掲示の数"),
    Channel("ext.transit_delay", EXTERNAL, "flag",
            "環境情報(当日値): 鉄道が止まっているか(1=運休/麻痺・0=平常)。"
            "災害・遅延機構が無効なランでは常に 0 = 定数チャンネル(σ=0 として扱われる)",
            label="電車が止まっているか(1=止まっている)"),
    Channel("ext.weather_temp", EXTERNAL, "degC",
            "環境情報(当日値): その日の最高気温。天候機構が無効なランでは **null**",
            label="その日の暑さ(度)"),
)

_BODY_GAUGES: tuple[Channel, ...] = (
    Channel("body.drive", BODY, "level",
            "欲求ゲージ(0-1)。出来事で溜まり思考で減る、既存の発火機構の主ゲージ",
            label="何か言いたい・動きたい気持ちの強さ(0-1)"),
    Channel("body.boredom", BODY, "level",
            "退屈/好奇心ゲージ(0-1)。長居で溜まり移動・新奇で減る。"
            "機構が無効なランでは **null**(0 で埋めない)",
            label="退屈さ(0-1)"),
    Channel("body.fatigue", BODY, "level",
            "疲労ゲージ(0-1)。活動で溜まり睡眠で回復する。健康機構が無効なランでは常に 0",
            label="疲れ具合(0-1)"),
    Channel("body.arousal", BODY, "level",
            "覚醒度(core affect の第2軸)。感情ハブが無効なランでは常に既定値=定数",
            label="気持ちの高ぶり(0-1)"),
)

_PREDICTION: tuple[Channel, ...] = (
    Channel("pred.unmet", PREDICTION, "count",
            "予測不成立(『来るはずの誰か/何かが来ない』)。**第81バッチで期待値 ô が入るまで"
            "値は未使用**= 常に null。列だけ先に確保してスキーマ変更を避けるための枠",
            implemented=False, label="来ると思ったものが来ない回数"),
)


def _state_channels() -> tuple[Channel, ...]:
    """経験で動く内部状態ゲージのチャンネル(**名は factors 層のキーから実行時に生成**)。

    no-fingerprint 契約: 認知層は因子名を綴らない。並びは辞書順で固定する。
    """
    from ..factors.registry import STATE_INIT
    return tuple(
        Channel(f"body.state.{key}", BODY, "level",
                "経験で動く内部状態ゲージ(名称の定義は factors 層が持つ)",
                # ★ラベルも**位置**でしか名指ししない: id には factors 層のキーが入るので
                #   プロンプトに出せない(第82 watch spec はラベルの方を使う)。
                label=f"自分についての感じ方{i + 1}(0-1)")
        for i, key in enumerate(sorted(STATE_INIT))
    )


_STATE: tuple[Channel, ...] = _state_channels()
_STATE_KEYS: tuple[str, ...] = tuple(ch.id.rsplit(".", 1)[1] for ch in _STATE)

CHANNELS: tuple[Channel, ...] = (_EXTERNAL + _BODY_GAUGES + _STATE + _PREDICTION)

COLUMNS: tuple[str, ...] = tuple(ch.column for ch in CHANNELS)
BY_ID: dict[str, Channel] = {ch.id: ch for ch in CHANNELS}

# 固定列(Parquet の先頭 3 列)。値列はこの後ろに COLUMNS の順で並ぶ。
KEY_COLUMNS: tuple[str, ...] = ("step", "sim_min", "agent_id")

# ext.heard / ext.signage の元になる L1 イベント種(定義を 1 箇所に置く)。
_HEARD_KIND = "hear"
_DM_KIND = "dm"
_SIGNAGE_KIND = "ad_exposure"


def spec_sha256() -> str:
    """チャンネル定義の凍結ハッシュ(σ_c ファイルがどの定義で測られたかを縛る)。"""
    blob = json.dumps(
        {"schema": SCHEMA,
         "channels": [[c.id, c.source, c.unit, bool(c.implemented)] for c in CHANNELS]},
        sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def describe() -> list[dict]:
    """一覧の人間可読な写し(scripts / manifest 用)。"""
    return [{"id": c.id, "column": c.column, "source": c.source, "unit": c.unit,
             "implemented": bool(c.implemented), "description": c.description}
            for c in CHANNELS]


# --------------------------------------------------------------------------- #
# cfg 正準化(既定 OFF)
# --------------------------------------------------------------------------- #
def build_cfg(raw: dict | None) -> dict:
    """conf の `cognition.channels` ブロックを型強制つきで正準化する(既定 OFF)。"""
    raw = dict(raw or {})
    return {
        "enabled": bool(raw.get("enabled", False)),
        # 何 step ごとに採るか(1=毎 step)。観測装置の粒度であって世界の量ではない。
        "every_steps": max(1, int(raw.get("every_steps", 1) or 1)),
        # 凍結ファイルのパス。空文字 = 既定パス(cognition/calib.py の定数)。
        "sigma_file": str(raw.get("sigma_file", "") or ""),
        "calib_file": str(raw.get("calib_file", "") or ""),
        # G6(第114 GT ロガー・既定 OFF): 価値 4 軸の充足 sat を **observer 側の追加列**として
        # channels.parquet の末尾へ足すか。★本 module の `CHANNELS` は 1 本も動かさない
        # (動かすと `spec_sha256()` が変わり、既に測った σ_c 凍結ファイルが全部無効になる)。
        # sat は驚き S の入力ではないので、観測列としてだけ持つのが正しい置き場所である。
        "sat_columns": bool(raw.get("sat_columns", False)),
    }


def due(cfg: dict, step: int) -> bool:
    """その step で観測を採るか(every_steps の位相は step=0 を含む=L3 と揃う)。"""
    return int(step) % int(cfg["every_steps"]) == 0


# --------------------------------------------------------------------------- #
# 観測(読み取り専用・乱数ゼロ・決定論)
# --------------------------------------------------------------------------- #
def _place_key(agent):
    """同席の単位。範囲外は None(誰とも同席していない)。"""
    if agent.loc == "outside":
        return None
    if agent.building:
        return ("bld", agent.building, int(agent.floor))
    return ("node", agent.node)


def _world_values(sim) -> tuple[float, float | None]:
    """全員に共通の環境情報(遅延フラグ・当日の最高気温)。"""
    transit = getattr(sim, "transit", None)
    delay = 1.0 if (transit is not None
                    and bool(getattr(transit, "suspended", False))) else 0.0
    if delay == 0.0:
        # 第84バッチ: 環境フィードバック 規則1 の内生的な遅延(ホーム密度→停車時間延長)。
        # 本チャンネルは第80 では災害の運休しか見ておらず、災害 OFF のランでは定数(σ=0)
        # だった。ここが 1 になるのは **エージェントの行動が作った遅延**であって、外から
        # 与えたショックではない(= 環が閉じたことの観測側の証拠)。OFF は常に 0.0=不変。
        from .. import envfeedback as _envfb
        delay = _envfb.delay_flag(sim)
    weather = getattr(sim, "today_weather", None)
    temp: float | None = None
    if isinstance(weather, dict) and weather.get("temp_hi") is not None:
        try:
            temp = float(weather["temp_hi"])
        except (TypeError, ValueError):
            temp = None
    return delay, temp


def _event_counts(sim, since_idx: int) -> tuple[dict, dict]:
    """この step に新規記録された L1 から (受信発話数, 掲示視認数) を数える。

    `since_idx` は run_step 冒頭で控えた `len(sim.logger.events)`(第71 の行間補間と
    同じ流儀)。**イベントを読むだけ**で書かない。
    """
    heard: dict[int, int] = {}
    signage: dict[int, int] = {}
    for event in sim.logger.events[since_idx:]:
        kind = event.kind
        if kind == _HEARD_KIND:
            aid = int(event.agent_id)
            if aid >= 0:
                heard[aid] = heard.get(aid, 0) + 1
        elif kind == _DM_KIND:
            to = (event.payload or {}).get("to")
            if to is not None:
                try:
                    aid = int(to)
                except (TypeError, ValueError):
                    continue
                heard[aid] = heard.get(aid, 0) + 1
        elif kind == _SIGNAGE_KIND:
            aid = int(event.agent_id)
            if aid >= 0:
                signage[aid] = signage.get(aid, 0) + 1
    return heard, signage


def observe(sim, step: int, sim_min: int, since_idx: int) -> list[tuple]:
    """1 step 分の観測行 [(step, sim_min, agent_id, *値)] を返す(**副作用なし**)。

    値の並びは `CHANNELS` の並びと厳密に一致する(tests が固定する)。
    実装されていないチャンネル・元機構が無効なチャンネルの値は None(=null)。
    """
    agents = list(getattr(sim, "agents", ()) or ())
    if not agents:
        return []

    # ---- 外界(1): 同席数 ----
    counts: dict = {}
    for agent in agents:
        key = _place_key(agent)
        if key is not None:
            counts[key] = counts.get(key, 0) + 1

    # ---- 外界(2): 遭遇数 = 知覚半径内の起きている他者 ----
    #  位置が確定した step 末に索引を張り直す(_phase_drive 時点の索引は _apply で古くなる)。
    radius = float(sim.cfg.world.perception_radius_m)
    index = build_index(agents, radius)

    # ---- 外界(3)(4): この step の受信発話・掲示視認 ----
    heard, signage = _event_counts(sim, since_idx)

    # ---- 外界(5)(6): 全員共通の環境情報 ----
    delay, temp = _world_values(sim)

    rows: list[tuple] = []
    for agent in agents:
        aid = int(agent.id)
        key = _place_key(agent)
        crowd = float(counts.get(key, 1) - 1) if key is not None else 0.0
        # 竹-4(P3): 物理ゾーンの実測近傍人数への差し替え。**配線のみ・既定 OFF**
        # (physics.perception.channels=false のとき引数をそのまま返す=バイト一致)。
        crowd = _physics.crowd_override(sim, agent, crowd)
        values: list[float | None] = [
            crowd,
            # 人数しか要らないので列挙(遮蔽判定つきリスト構築 + id 昇順ソート)はしない。
            # `count_hearers` は `len(hearers_of(...))` と厳密に同値(perception 側で機械照合)。
            float(count_hearers(agent, index, radius)),
            float(heard.get(aid, 0)),
            float(signage.get(aid, 0)),
            delay,
            temp,
            # ---- 身体 ----
            float(getattr(agent, "drive", 0.0)),
            _opt_float(getattr(agent, "_boredom", None)),
            float(getattr(agent, "fatigue", 0.0)),
            float(getattr(agent, "arousal", 0.0)),
        ]
        states = getattr(agent, "states", None) or {}
        values.extend(_opt_float(states.get(key_name)) for key_name in _STATE_KEYS)
        # ---- 予測不成立(第81 で埋まる枠。値は未使用)----
        values.append(None)
        rows.append((int(step), int(sim_min), aid, *values))
    return rows


def _opt_float(value) -> float | None:
    """欠測(None)は None のまま返す(0 で埋めない)。"""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
