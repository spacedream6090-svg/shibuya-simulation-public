"""医療の受け皿 = 搬送・入院・医療費(身体と事件のレイヤー **H2**・既定 OFF)。

正典: docs/plans/body-incident-layer-plan.md §2。H1(``health.py`` の重症度状態機械)が
「倒れる」までを作り、``city_ops`` の救急連鎖が「通報 → 出動」までを作った。本 module が
足すのはその**続き**である:

    ① 搬送先   … v8 地図の ``subcat=hospital``(7 件)が受け皿。S3/S4 は EMS が病院へ運ぶ
    ② 入院     … 確定重症度で在院長が決まる(軽症=数時間 / 中等症=数日 / 重症=長期)。
                  物理は ``lodging``(非自宅建物で N 泊)の**完成済みテンプレートと同型**
    ③ 金の三本足 … 救急搬送=公費(区の歳出)/ 患者 3 割=既存 ``_spend(cat="medical")`` /
                  保険 7 割=**命名された RoW チャネル** ``insurance_reimbursement``

★**搬送先の是正**(監査で見つかった取りこぼし)
------------------------------------------------
地図 v8 の病院 7 件は ``cat=service`` に潰れており、``subcat`` を読まない限り
「汎用のサービス POI」と区別が付かない。実際 ``services.py`` の任意受診(clinic)は
名前ヒント(「クリニック」等)だけで引いていたため、**総合病院が健診の行き先として
選ばれていた**(v8 の hospital の 1 件は名称が「〜クリニック」= ヒントに当たる)。
本レーンで 2 つを分ける:

  - 病院(``subcat=hospital``)= **搬送先**。歩いて行く場所ではない
  - クリニック(名前ヒント・``subcat=hospital`` を**除く**)= 自力受診の行き先

★``subcat`` を持たない地図(v7 以前)では病院が **0 件**になる。これは取りこぼしではなく
  正直な依存関係で、名前から病院を推測する既定は置かない(``name_keywords`` の既定は空 =
  無い施設を捏造しない。``city_ops`` が自販機を作らなかったのと同じ線引き)。0 件のランでは
  搬送は起きず、provenance の ``no_hospital`` に件数が出る(黙って 0 件にはならない)。

金の三本足(SFC 規約 = ``economy_sfc`` の既存前例に従う)
--------------------------------------------------------
| 足 | 誰が誰へ | 実装 | 既定 OFF の挙動 |
|---|---|---|---|
| ① 救急搬送 | 区(ward)→ 街の外 | ``economy_sfc.on_ems_transport``(``on_rule_bonus`` と同型) | 記帳しない |
| ② 自己負担 | 家計 → 医療機関 org | 既存 ``_spend(cat="medical")`` に **受診先ノード**を渡す | 現行どおり |
| ③ 保険給付 | 街の外 → 医療機関 org | RoW チャネル ``insurance_reimbursement`` | 記帳しない |

★① の受け手が RoW である理由(**管轄の現実**。計画書 §7「管轄混同」への回答): 救急を
  運行するのは東京消防庁 = **都**であって、本シムの行政(``government`` = 区)ではない。
  区は都区財政調整を通じて消防費を負担する側なので、**支出主体は区・受け手は街の外**が
  実態に最も近い。街の中に救急を運行する org は 1 つも無いので、ここで org を捏造しない。
★② の是正点(監査発見): 受け手の解決は今まで**患者の現在地**で行われていた。中等症の受診は
  自宅や路上で発火するので ``(building, floor, POI種別)`` も ``(node, POI種別)`` も当たらず、
  医療費は**黙って RoW(``unknown_payee``)へ漏れていた**。本 module が受診先を決めて
  ``payee_node`` として渡すので、台帳に居る医療機関には正しく入金される(居なければ
  ``unknown_payee`` のままだが、それは「域外資本の医院」という**正しい**開示になる)。
★③ を新部門にしない理由: 保険者(協会けんぽ・国保)は街の中に居ない。RoW に落として
  **チャネル名で正直に開示する**のが ``tax_gap`` / ``clamp_gap`` と同じ流儀
  (計画書 §6-4 でユーザー承認済み)。

正直な限界(6 件)
------------------
1. **救急車の実体を作っていない**(``city_ops`` の限界 6 と同じ線引き)。搬送は
   ``transport.steps`` 経過後に患者の位置を病院へ**付け替える**形で表現する。したがって
   「何 step で着いたか」は距離ではなく config が決めている(正直なモデル値)。
2. **S2 の自力受診は移動を作らない**。H1 の既存挙動(その場で受診が成立する)を 1 ミリも
   変えず、**受け手の解決だけ**を是正する。したがって受診先ノードは会計上の帰属であって
   患者がそこへ歩いた記録ではない(歩かせると H1 の物理・呼数に手を入れることになる)。
3. **治療は転帰を変えない**。誰が入院しても死亡確率は 1 ミリも動かない(死は H1 が発症時に
   決める = レーン H1 の設計判断で、再開封しない)。入院は**在院という状態**と**会計**だけを作る。
4. **高額療養費制度・公費負担医療を作っていない**。自己負担は一律 ``coinsurance``(3 割)で、
   月額上限も年齢別の負担割合(未就学 2 割 / 75 歳以上 1 割)も入れていない。入れると
   「世帯の所得区分」という世界に無いデータが要る(持っていないふりをしない)。
5. **プール回転で入院の印が落ちる**。本 module の状態は agent の動的属性なので、
   ``world/pool.py`` の dehydrate(明示列挙した可変状態しか運ばない)を跨ぐと消える
   (``incidents_interpersonal`` の酩酊マーカーと同じ既知の粗さ)。checkpoint(pickle)は
   通るので resume は安全。
6. **病床数の制約が無い**。満床で受け入れ不能という事態は起きない(``hospital`` POI に
   病床数のデータが無い。持っていない数字を捏造しない)。逼迫が観測できるのは救急隊の側
   (``city_ops`` の ``dispatch_unstaffed`` = 全車出動中の通報)だけである。

R1 ドクトリン
-------------
- 既定 ``medical.enabled: false`` では ``phase`` / ``on_ems_dispatch`` / ``on_care`` が
  即 return し、**L1 に 1 件も出ず・agent に属性が生えず・sim に state が生えず・
  乱数 stream を 1 本も引かず・プロンプトが 1 バイトも変わらない**(= ゴールデン L1 バイト一致)。
- **乱数を 1 本も引かない**(搬送先は地図の純関数・在院長は確定重症度の純関数・
  金額は config の純関数)。tests/test_medical.py が AST でこれを機械固定する。
- **generate() の呼び出しサイトを 1 つも作らない**(LLM 追加呼ゼロ = k 非依存)。
  出来事は既存の記憶機構(``agent.remember``)へ定型 1 行が入るだけ。
- 新 L1 種は**材料側 registration**(``city_ops.py`` / ``incidents_env.py`` と同じ流儀 =
  ``observer/schema.py`` を 1 バイトも触らない)。
"""
from __future__ import annotations

from .observer.schema import Event, register_event_kind

SCHEMA = 1

# --------------------------------------------------------------------------- #
# L1 イベント種(材料側 registration)。payload に自由文は 1 つも入らない
# (poi は地図の施設名 = 世界の識別子。残りは分類名・件数・金額・step)。
# --------------------------------------------------------------------------- #
register_event_kind(
    "ems_transport",
    "救急搬送(現場 → 病院。**この行の cost が公費**)。agent_id = 隊員"
    "{patient, from_node, node, poi, sev, confirmed, steps, cost, payer, payee}")
register_event_kind(
    "hospital_admit",
    "入院(在院の開始。確定重症度で在院長が決まる)。agent_id = 患者"
    "{node, poi, building, confirmed, until_step, days}")
register_event_kind(
    "hospital_discharge",
    "退院(在院の終了)。agent_id = 患者"
    "{node, poi, confirmed, stayed_steps, billed_days}")
register_event_kind(
    "medical_bill",
    "医療費の内訳(自己負担は spend 側で会計済み。**amount = 実際に動いた保険給付**)。"
    "agent_id = 患者{node, poi, kind, days, gross, self_pay, amount, payer, payee}")

# --------------------------------------------------------------------------- #
# 記憶の 1 行(**出来事の種類だけの純関数**。数字・金額・config・実験条件・機構語・
# 地名は 1 つも入らない)= city_ops.COLLAPSE_TEXT / traces.sentence と同じ規約。
# --------------------------------------------------------------------------- #
TRANSPORT_TEXT = "急病人を病院へ運んだ。"
ADMIT_TEXT = "病院に運ばれ、そのまま入院することになった。"
DISCHARGE_TEXT = "退院して、いつもの暮らしに戻った。"
CLINIC_TEXT = "近くの医院で診てもらった。"

#: 地図 v8 が OSM の ``amenity=hospital`` に付けるサブカテゴリ。
HOSPITAL_SUBCAT = "hospital"

#: クリニック(自力受診の行き先)の名前ヒント。``services.py`` の任意受診(clinic)と
#: **同じ語彙**だが、あちらは「病気起点でない健康維持」でこちらは「病気起点の受診」なので
#: 表は別に持つ(どちらも ``subcat=hospital`` を除く = 総合病院へ歩いて行かせない)。
CLINIC_KEYWORDS: tuple[str, ...] = ("クリニック", "医院", "診療", "歯科", "内科", "眼科")

DEFAULTS: dict = {
    "enabled": False,
    "max_events_per_step": 24,     # 1 step に出す本 module の L1 件数の上限(安全弁)
    # ---- 搬送(S3/S4 = EMS が病院へ運ぶ)----
    "transport": {
        "enabled": True,
        "steps": 3,                # 現場 → 病院の搬送にかかる長さ [step](モデル値)
        "max_per_day": 24,         # 1 日に出す搬送の上限(L1 の安全弁)
        "hold_crew": True,         # 搬送のあいだ隊員を現場復帰させない(二重出動の防止)
    },
    # ---- 搬送先(病院)----
    "hospital": {
        "poi_cats": ("service",),  # v7/v8 とも build_map が amenity=hospital を service へ潰す
        "subcat": HOSPITAL_SUBCAT,
        # ★既定は空: subcat を持たない地図で名前から病院を推測しない(捏造の防止)。
        "name_keywords": (),
    },
    # ---- 自力受診(S2)の行き先 ----
    "clinic": {
        "enabled": True,
        "poi_cats": ("service",),
        "hints": CLINIC_KEYWORDS,
        "radius_m": 1500.0,        # 受診先を探す半径(徒歩圏 = 生活圏の医院)
    },
    # ---- 入院(確定重症度で在院長が決まる)----
    #  東京消防庁の搬送人員の程度別構成(軽症 52.8 / 中等症 38.6 / 重症 7.3 / 死亡 1.3 %)に
    #  対応させる。日数のアンカー: 一般病床の平均在院日数 ≒ 16 日(患者調査)。軽症は
    #  そもそも入院せず数時間で帰宅、中等症で数日、重症で長期(2 週間)。
    "admit": {
        "enabled": True,
        "mild_steps": 18,          # 軽症の在院の長さ [step](既定 Δt=10 分で 3 時間)
        "moderate_days": 3,
        "severe_days": 14,
        "arrest_days": 21,         # 心停止からの蘇生後(重症より長い)
        "max_days": 30,            # 在院長の上限(暴走の安全弁)
    },
    # ---- 金の三本足 ----
    "money": {
        "enabled": True,
        # 救急出動 1 件あたりの公費(消防庁の試算 ≒ 4.5 万円/件)。
        "ems_cost": 45000.0,
        # 患者の自己負担割合(健康保険法 = 3 割)。★年齢別(未就学 2 割 / 75 歳以上 1 割)と
        #   高額療養費の月額上限は作っていない(限界 4)。
        "coinsurance": 0.3,
        # 入院 1 日あたりの**自己負担**(急性期の 1 日あたり医療費 ≒ 3.5 万円 × 3 割)。
        "admit_self_pay_per_day": 10500.0,
    },
}

_TOP_INT = ("max_events_per_step",)


# --------------------------------------------------------------------------- #
# cfg 正準化(city_ops.build_cfg / incidents_env.build_cfg と同型:
#   dict / OmegaConf 両対応・dotlist の文字列を型強制・未知キーは黙って捨てる)
# --------------------------------------------------------------------------- #
def _to_plain(raw):
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(raw):
            return OmegaConf.to_container(raw, resolve=True)
    except Exception:                              # noqa: BLE001(旧 config 互換)
        pass
    return raw


def _words(raw, fallback) -> tuple:
    """語リストの正準化(空白除去・空要素落とし・**宣言順を保つ**)。空なら空のまま。"""
    got = _to_plain(raw)
    if isinstance(got, str):
        got = [got]
    if got is None:
        return tuple(fallback)
    return tuple(str(x).strip() for x in got if str(x).strip())


def _block(raw, defaults: dict, words_keys: tuple, floats: tuple,
           bools: tuple, strs: tuple = ()) -> dict:
    out = dict(defaults)
    got = dict(_to_plain(raw) or {})
    for key, value in got.items():
        if key not in out:
            continue                               # 未知キーは捨てる(捏造しない)
        if key in bools:
            out[key] = bool(value)
        elif key in words_keys:
            out[key] = _words(value, defaults[key])
        elif key in strs:
            out[key] = str(value)
        elif key in floats:
            out[key] = max(0.0, float(value))
        else:
            out[key] = int(value)
    return out


def build_cfg(raw) -> dict:
    """conf の ``medical`` ブロックを型強制つきで正準化(既定 OFF=現行挙動と完全同一)。"""
    raw = dict(_to_plain(raw) or {})
    cfg: dict = {"enabled": bool(raw.get("enabled", False))}
    for key in _TOP_INT:
        cfg[key] = max(0, int(raw.get(key, DEFAULTS[key])))
    cfg["transport"] = _block(raw.get("transport"), DEFAULTS["transport"], (), (),
                              ("enabled", "hold_crew"))
    cfg["hospital"] = _block(raw.get("hospital"), DEFAULTS["hospital"],
                             ("poi_cats", "name_keywords"), (), (), ("subcat",))
    cfg["clinic"] = _block(raw.get("clinic"), DEFAULTS["clinic"],
                           ("poi_cats", "hints"), ("radius_m",), ("enabled",))
    cfg["admit"] = _block(raw.get("admit"), DEFAULTS["admit"], (), (), ("enabled",))
    cfg["money"] = _block(raw.get("money"), DEFAULTS["money"], (),
                          ("ems_cost", "coinsurance", "admit_self_pay_per_day"),
                          ("enabled",))
    cfg["transport"]["steps"] = max(1, int(cfg["transport"]["steps"]))
    cfg["admit"]["mild_steps"] = max(1, int(cfg["admit"]["mild_steps"]))
    for key in ("moderate_days", "severe_days", "arrest_days"):
        cfg["admit"][key] = max(1, int(cfg["admit"][key]))
    cfg["admit"]["max_days"] = max(1, int(cfg["admit"]["max_days"]))
    cfg["money"]["coinsurance"] = min(1.0, max(0.0, float(cfg["money"]["coinsurance"])))
    return cfg


def cfg_of(sim) -> dict:
    """医療の設定(初回のみ ``sim.cfg.medical`` から遅延構築してキャッシュ)。

    ``traces.cfg_of`` / ``city_ops.cfg_of`` と同型。キャッシュ属性 ``sim.medicalcfg`` は
    L1/L2/L3/乱数に一切現れない = 既定 OFF のバイト一致を壊さない。"""
    got = getattr(sim, "medicalcfg", None)
    if got is None:
        try:
            raw = sim.cfg.get("medical", None)
        except Exception:                          # noqa: BLE001(旧 config 互換)
            raw = None
        got = build_cfg(raw)
        sim.medicalcfg = got
    return got


def enabled(sim) -> bool:
    """医療の受け皿(H2)が有効か。既定 OFF=新経路を一切通さない(バイト一致)。"""
    return bool(cfg_of(sim)["enabled"])


# --------------------------------------------------------------------------- #
# 遅延 import(循環の回避。city_ops._health_mod と同じ作法)
# --------------------------------------------------------------------------- #
def _health_mod():
    from . import health as _health
    return _health


def _sfc_mod():
    from . import economy_sfc as _sfc
    return _sfc


def _severity_on(sim) -> bool:
    """重症度の状態機械(H1)が有効か = **確定重症度が世界に在るか**(正直な依存関係)。"""
    return bool(_health_mod().severity_on(sim))


# --------------------------------------------------------------------------- #
# 地図の読み取り(**すべて純関数**。世界を 1 バイトも書き換えない)
# --------------------------------------------------------------------------- #
def is_hospital(cfg: dict, poi: dict) -> bool:
    """その POI が病院(搬送先)か。``subcat`` 優先・名前ヒントは既定で空 = 使わない。"""
    hcfg = cfg["hospital"]
    if str(poi.get("cat") or "") not in tuple(hcfg["poi_cats"]):
        return False
    if str(poi.get("subcat") or "") == str(hcfg["subcat"]):
        return True
    words = tuple(hcfg["name_keywords"])
    return bool(words) and any(w in str(poi.get("name") or "") for w in words)


def hospitals(sim) -> list[dict]:
    """搬送先の病院 POI(決定論の安定順・sim に一度だけキャッシュ)。

    ★``subcat`` を持たない地図では **0 件**(= 搬送が起きない)。無い施設を名前から
      推測しないための線引きで、0 件のまま黙らないよう provenance に件数を出す。"""
    cache = getattr(sim, "_med_hospitals", None)
    if cache is None:
        cfg = cfg_of(sim)
        cache = sorted((p for p in sim.city.poi_list if is_hospital(cfg, p)),
                       key=lambda p: (str(p.get("id", "")), str(p.get("node", ""))))
        sim._med_hospitals = cache
    return cache


def clinics(sim) -> list[dict]:
    """自力受診の行き先(**病院を除く**医院・クリニック)。決定論の安定順・キャッシュ。"""
    cache = getattr(sim, "_med_clinics", None)
    if cache is None:
        cfg = cfg_of(sim)
        cats = tuple(cfg["clinic"]["poi_cats"])
        hints = tuple(cfg["clinic"]["hints"])
        out = []
        for poi in sim.city.poi_list:
            if str(poi.get("cat") or "") not in cats:
                continue
            if is_hospital(cfg, poi):
                continue                           # ★総合病院へは歩いて行かせない
            name = str(poi.get("name") or "")
            if hints and not any(h in name for h in hints):
                continue
            out.append(poi)
        cache = sorted(out, key=lambda p: (str(p.get("id", "")),
                                           str(p.get("node", ""))))
        sim._med_clinics = cache
    return cache


def _nearest(sim, pois: list[dict], x: float, y: float, radius_m: float = 0.0):
    """(x, y) に最も近い POI(同距離は id 昇順 = 決定論)。半径 0 = 無制限。無ければ None。"""
    r2 = float(radius_m) * float(radius_m)
    best = None
    best_key = None
    for poi in pois:
        try:
            px, py = sim.city.node_xy(str(poi["node"]))
        except Exception:                          # noqa: BLE001(未知ノードの保険)
            continue
        d2 = (px - float(x)) ** 2 + (py - float(y)) ** 2
        if r2 > 0.0 and d2 > r2:
            continue
        key = (d2, str(poi.get("id", "")))
        if best_key is None or key < best_key:
            best_key, best = key, poi
    return best


def nearest_hospital(sim, x: float, y: float):
    """現場から最も近い病院(決定論)。地図に病院が無ければ None。"""
    return _nearest(sim, hospitals(sim), x, y)


def care_venue(sim, agent) -> str | None:
    """自力受診(S2)の**受診先ノード**。H1 の ``maybe_care`` から読まれる唯一の口。

    既定 OFF / クリニックが近くに無い / 地図にクリニックが無いランでは None = H1 は
    今までどおり受け手を解決せずに支払う(= 現行と完全同値)。"""
    if not enabled(sim) or not cfg_of(sim)["clinic"]["enabled"]:
        return None
    poi = _nearest(sim, clinics(sim), float(agent.x), float(agent.y),
                   float(cfg_of(sim)["clinic"]["radius_m"]))
    return None if poi is None else str(poi["node"])


def venue_poi(sim, node: str) -> str:
    """受診先ノードの施設名(観測用。決まらなければ空文字 = 捏造しない)。"""
    cfg = cfg_of(sim)
    hints = tuple(cfg["clinic"]["hints"])
    for poi in sim.city.pois_at_node(str(node)):
        if str(poi.get("cat") or "") not in tuple(cfg["clinic"]["poi_cats"]):
            continue
        if is_hospital(cfg, poi):
            continue
        name = str(poi.get("name") or "")
        if not hints or any(h in name for h in hints):
            return name
    return ""


# --------------------------------------------------------------------------- #
# state(sim 側。**OFF では 1 つも生えない**)
# --------------------------------------------------------------------------- #
def _state(sim) -> dict:
    st = getattr(sim, "_med_state", None)
    if st is None:
        st = {"schema": SCHEMA, "by_kind": {}, "dropped": 0,
              "transports": 0, "transports_by_day": {}, "no_hospital": 0,
              "admits": 0, "admits_by_sev": {}, "discharges": 0,
              "died_in_care": 0, "bed_steps": 0,
              "clinic_bills": 0, "hospital_bills": 0, "venue_missing": 0,
              "ems_cost_total": 0.0, "ems_funded": 0.0,
              "self_pay_total": 0.0, "insurance_total": 0.0,
              "insurance_to_row": 0.0, "payee_resolved": 0, "payee_row": 0}
        sim._med_state = st
    return st


def state_of(sim):
    """ON のときだけ state を返す(OFF は None = checkpoint も summary もキーを作らない)。"""
    return getattr(sim, "_med_state", None)


def restore_state(sim, state) -> None:
    """checkpoint からの復元(旧 ckpt / OFF ランでは None = no-op)。"""
    if state is not None:
        sim._med_state = state


def _bump(table: dict, key, n: int = 1) -> None:
    table[str(key)] = int(table.get(str(key), 0)) + int(n)


class _Budget:
    """1 step に出す L1 件数の上限(``city_ops._Budget`` と同型 = 同じ安全弁を 2 つ書かない)。"""

    __slots__ = ("sim", "st", "left")

    def __init__(self, sim, st: dict, cap: int):
        self.sim = sim
        self.st = st
        self.left = int(cap)

    def log(self, event) -> bool:
        if self.left <= 0:
            self.st["dropped"] += 1
            return False
        self.left -= 1
        self.sim.logger.log(event)
        _bump(self.st["by_kind"], event.kind)
        return True


# --------------------------------------------------------------------------- #
# 在院長(**確定重症度の純関数**。乱数ゼロ)
# --------------------------------------------------------------------------- #
def stay_steps(cfg: dict, confirmed: int, steps_per_day: int) -> int:
    """確定重症度 → 在院の長さ [step]。軽症だけが「数時間で帰宅」= 日未満。"""
    health = _health_mod()
    acfg = cfg["admit"]
    per_day = max(1, int(steps_per_day))
    cap = int(acfg["max_days"]) * per_day
    sev = int(confirmed)
    if sev <= health.S_MILD:
        return min(cap, max(1, int(acfg["mild_steps"])))
    if sev == health.S_MODERATE:
        return min(cap, int(acfg["moderate_days"]) * per_day)
    if sev == health.S_SEVERE:
        return min(cap, int(acfg["severe_days"]) * per_day)
    return min(cap, int(acfg["arrest_days"]) * per_day)


# =========================================================================== #
# ① 搬送(city_ops の救急連鎖からの唯一の口)
# =========================================================================== #
def on_ems_dispatch(sim, patient, crew, step: int, sim_min: int) -> bool:
    """出動が決まった患者を**病院へ搬送する**(``city_ops._ems_chain`` からの唯一の口)。

    ★身体状態(``sick`` / ``severity``)は 1 バイトも書かない(H1 の管轄)。本 module が
      書くのは搬送・在院という**物理と会計の状態**だけである。
    ★``severity`` OFF のランでは確定重症度が世界に無いので搬送しない(正直な依存関係)。
    戻り値 = 搬送を始めたか(False = 何も起きていない)。"""
    if not enabled(sim):
        return False
    cfg = cfg_of(sim)
    if not cfg["transport"]["enabled"] or not _severity_on(sim):
        return False
    if getattr(patient, "med_admitted", False) \
            or int(getattr(patient, "med_transport_until", -1)) >= 0:
        return False                               # 既に搬送中/在院中(二重搬送しない)
    st = _state(sim)
    day = int(sim_min) // 1440
    done = int(st["transports_by_day"].get(day, 0))
    if done >= int(cfg["transport"]["max_per_day"]):
        return False
    hosp = nearest_hospital(sim, float(patient.x), float(patient.y))
    if hosp is None:
        st["no_hospital"] += 1                     # 地図に病院が無い(黙って 0 件にしない)
        return False
    steps = max(1, int(cfg["transport"]["steps"]))
    patient.med_from_node = str(patient.node)
    patient.med_dest_node = str(hosp["node"])
    patient.med_dest_poi = str(hosp.get("name") or "")
    patient.med_dest_building = str(hosp.get("building") or "")
    patient.med_transport_until = int(step) + steps
    patient.med_crew = int(crew.id) if crew is not None else -1
    patient.route = []
    patient.dest = None
    patient.sleeping = True                        # 搬送中は動けない(既存の睡眠機構)
    patient.sleep_until = max(int(getattr(patient, "sleep_until", 0)),
                              int(step) + steps)
    if crew is not None and cfg["transport"]["hold_crew"]:
        # 搬送が終わるまで隊員を持ち場へ戻さない(印は **city_ops のもの**を使う =
        # 復帰経路を新設しない。``city_ops._ems_restore`` がそのまま戻す)。
        crew.city_ops_ems_until = max(int(getattr(crew, "city_ops_ems_until", -1)),
                                      int(step) + steps)
    st["transports"] += 1
    st["transports_by_day"][day] = done + 1
    return True


# =========================================================================== #
# ② 自力受診(H1 の maybe_care からの唯一の口)
# =========================================================================== #
def on_care(sim, agent, self_pay: float, spend, step: int, sim_min: int) -> bool:
    """中等症の受診の**支払いを受け持つ**(受け手の是正 + 保険給付)。

    戻り値 = 本 module が支払いを済ませたか。False のとき H1 は今までどおり
    ``spend(agent, cost, "medical")`` を呼ぶ(= 既定 OFF は現行と完全同値)。
    ★受診そのもの(いつ・誰が受診するか)は H1 の決定論選択で、本 module は 1 ミリも動かさない。"""
    if not enabled(sim) or spend is None:
        return False
    cfg = cfg_of(sim)
    if not cfg["money"]["enabled"]:
        return False
    node = care_venue(sim, agent)
    if node is None:
        _state(sim)["venue_missing"] += 1
        return False
    spend(agent, float(self_pay), "medical", node)  # ★受け手 = 受診先(現在地ではない)
    _bill(sim, agent, node, venue_poi(sim, node), float(self_pay), "clinic", 0,
          step, sim_min, None)
    _state(sim)["clinic_bills"] += 1
    return True


# --------------------------------------------------------------------------- #
# 金の三本足(記帳は economy_sfc に閉じる。既定 OFF ではトークンが 1 つも載らない)
# --------------------------------------------------------------------------- #
def _bill(sim, agent, node: str, poi: str, self_pay: float, kind: str, days: int,
          step: int, sim_min: int, bud) -> None:
    """医療費の内訳を 1 行にする(自己負担は ``spend`` 側で会計済み = ここでは保険給付だけ)。

    総医療費 = 自己負担 / 自己負担割合。保険給付 = 総医療費 − 自己負担。給付は
    **街の外の保険者**から医療機関 org へ入る(RoW チャネル ``insurance_reimbursement``)。
    受け手 org が台帳で解決できないときは街の残高が 1 円も動かないので ``amount=0`` を
    載せる(= 動いていない金を動いたことにしない)。"""
    st = _state(sim)
    cfg = cfg_of(sim)
    coins = float(cfg["money"]["coinsurance"])
    paid = float(self_pay)
    gross = paid / coins if coins > 0.0 else paid
    insurance = max(0.0, gross - paid)
    st["self_pay_total"] += paid
    st["insurance_total"] += insurance
    moved = 0.0
    payer = payee = None
    got = _sfc_mod().on_insurance(sim, str(node), "medical", insurance)
    if got is not None:
        payer, payee = got
        moved = insurance
        st["payee_resolved"] += 1
    else:
        st["insurance_to_row"] += insurance
        st["payee_row"] += 1
    payload = {"node": str(node), "poi": str(poi), "kind": str(kind),
               "days": int(days), "gross": round(gross, 1),
               "self_pay": round(paid, 1), "amount": round(moved, 1)}
    if payer is not None:
        payload["payer"] = payer
        payload["payee"] = payee
    event = Event(step=int(step), sim_min=int(sim_min), agent_id=int(agent.id),
                  kind="medical_bill", x=agent.x, y=agent.y, payload=payload)
    if bud is None:
        sim.logger.log(event)
        _bump(st["by_kind"], "medical_bill")
    else:
        bud.log(event)


# =========================================================================== #
# ③ 毎 step: 搬送の到着 → 入院 → 退院(**乱数を 1 本も引かない状態機械**)
# =========================================================================== #
def phase(sim, step: int, sim_min: int, spend=None) -> None:
    """毎 step: 搬送の到着(= 入院)と退院。**既定 OFF は即 return**。

    ★``_phase_lodging`` と**同じ位置**(``_phase_wake_and_returns`` の直前)に置く:
      退院で ``sleeping`` を落とすので、後段の起床フェーズが二重に起こさない。
    """
    if not enabled(sim):
        return
    cfg = cfg_of(sim)
    st = _state(sim)
    bud = _Budget(sim, st, int(cfg["max_events_per_step"]))
    health = _health_mod()
    per_day = max(1, int(sim.clock.steps_per_day))
    for agent in sim.agents:                       # id 昇順 = 決定論
        if health.is_dead(agent):
            if getattr(agent, "med_admitted", False) \
                    or int(getattr(agent, "med_transport_until", -1)) >= 0:
                st["died_in_care"] += 1            # 在院中の死は H1 が既に 1 行記録している
                _clear(agent)
            continue
        if getattr(agent, "med_admitted", False):
            st["bed_steps"] += 1
            if int(step) >= int(getattr(agent, "med_until", 0)):
                _discharge(sim, cfg, st, bud, agent, step, sim_min, spend)
            continue
        until = int(getattr(agent, "med_transport_until", -1))
        if until >= 0 and int(step) >= until:
            _arrive(sim, cfg, st, bud, agent, step, sim_min, per_day)


def _arrive(sim, cfg: dict, st: dict, bud, agent, step: int, sim_min: int,
            per_day: int) -> None:
    """搬送の到着 = 病院ノードへの位置の付け替え + 入院の開始 + 公費の記帳。"""
    health = _health_mod()
    node = str(getattr(agent, "med_dest_node", "") or "")
    if not node:
        _clear(agent)
        return
    from_node = str(getattr(agent, "med_from_node", "") or "")
    poi = str(getattr(agent, "med_dest_poi", "") or "")
    crew_id = int(getattr(agent, "med_crew", -1))
    confirmed = int(getattr(agent, "sev_confirmed", -1))
    if confirmed < 0:                              # 確定していない = 見かけを使う(正直な代替)
        confirmed = int(health.apparent_severity(agent))
    # ★これは config が決めた**モデル値**であって走行時間ではない(限界 1)。
    steps = max(1, int(cfg["transport"]["steps"]))
    # ---- 位置の付け替え(救急車の実体は世界に無い = 限界 1)----
    agent.node = node
    try:
        agent.x, agent.y = sim.city.node_xy(node)
    except Exception:                              # noqa: BLE001(未知ノードの保険)
        pass
    bld = str(getattr(agent, "med_dest_building", "") or "")
    if bld and sim.city.has_building(bld):         # 病院の建物があれば屋内滞在
        building = sim.city.building(bld)
        agent.building = building["id"]
        agent.floor = 1
        agent.x, agent.y = building["centroid"]
    else:
        agent.building = None
        agent.floor = 0
    agent.route = []
    agent.dest = None
    agent.activity = ""
    agent.med_transport_until = -1
    # ---- ① 救急搬送 = 公費(区の歳出。受け手は街の外 = 運行主体は都)----
    cost = float(cfg["money"]["ems_cost"]) if cfg["money"]["enabled"] else 0.0
    payer = payee = None
    if cost > 0.0:
        st["ems_cost_total"] += cost
        got = _sfc_mod().on_ems_transport(sim, cost)
        if got is not None:
            payer, payee = got
            st["ems_funded"] += cost
    payload = {"patient": int(agent.id), "from_node": from_node, "node": node,
               "poi": poi, "sev": int(health.severity_of(agent)),
               "confirmed": int(confirmed), "steps": int(steps),
               "cost": round(cost, 1)}
    if payer is not None:
        payload["payer"] = payer
        payload["payee"] = payee
    bud.log(Event(step=step, sim_min=sim_min,
                  agent_id=crew_id if crew_id >= 0 else int(agent.id),
                  kind="ems_transport", x=agent.x, y=agent.y, payload=payload))
    crew = _agent_by_id(sim, crew_id)
    if crew is not None:
        crew.remember(TRANSPORT_TEXT)
    # ---- ② 入院(確定重症度で在院長が決まる)----
    if not cfg["admit"]["enabled"]:
        _clear(agent)
        # 入院を作らない設定では、その場で既存の睡眠機構に起こさせる(この step の
        # ``_phase_wake_and_returns`` が ``wake_up`` を 1 行出す = 起床経路を新設しない)。
        agent.sleep_until = int(step)
        return
    length = stay_steps(cfg, confirmed, per_day)
    agent.med_admitted = True
    agent.med_node = node
    agent.med_poi = poi
    agent.med_confirmed = int(confirmed)
    agent.med_since = int(step)
    agent.med_until = int(step) + int(length)
    agent.sleeping = True                          # 在院中は既存の睡眠機構で動かない
    agent.sleep_until = int(step) + int(length)
    agent.stay_until = int(step)
    st["admits"] += 1
    _bump(st["admits_by_sev"], confirmed)
    agent.remember(ADMIT_TEXT)
    bud.log(Event(step=step, sim_min=sim_min, agent_id=int(agent.id),
                  kind="hospital_admit", x=agent.x, y=agent.y,
                  payload={"node": node, "poi": poi,
                           "building": str(agent.building or ""),
                           "confirmed": int(confirmed),
                           "until_step": int(agent.med_until),
                           "days": round(float(length) / float(per_day), 2)}))


def _discharge(sim, cfg: dict, st: dict, bud, agent, step: int, sim_min: int,
               spend) -> None:
    """退院 = 在院の終了(状態復帰 + L1 1 行)+ 入院費の会計。"""
    per_day = max(1, int(sim.clock.steps_per_day))
    node = str(getattr(agent, "med_node", "") or str(agent.node))
    poi = str(getattr(agent, "med_poi", "") or "")
    confirmed = int(getattr(agent, "med_confirmed", -1))
    stayed = max(1, int(step) - int(getattr(agent, "med_since", step)))
    days = max(1, int(round(float(stayed) / float(per_day))))
    agent.sleeping = False                         # 退院(= 起床)。後段の起床は素通りする
    if agent.building:                             # 病院の建物を出る(路面ノードへ)
        building = sim.city.building(agent.building)
        agent.building = None
        agent.floor = 0
        agent.node = building["entrance"]
        agent.x, agent.y = sim.city.node_xy(agent.node)
    agent.activity = ""
    agent.stay_until = int(step) + 1
    _clear(agent)
    st["discharges"] += 1
    agent.remember(DISCHARGE_TEXT)
    bud.log(Event(step=step, sim_min=sim_min, agent_id=int(agent.id),
                  kind="hospital_discharge", x=agent.x, y=agent.y,
                  # ★``billed_days``(請求日数 = 最低 1 日)と ``hospital_admit`` の
                  #   ``days``(予定の在院長 = 日未満もありうる)は**別の量**である。
                  payload={"node": node, "poi": poi, "confirmed": int(confirmed),
                           "stayed_steps": int(stayed), "billed_days": int(days)}))
    if not cfg["money"]["enabled"] or spend is None:
        return
    self_pay = float(cfg["money"]["admit_self_pay_per_day"]) * float(days)
    if self_pay <= 0.0:
        return
    spend(agent, self_pay, "medical", node)        # ★受け手 = 病院(現在地ではない)
    _bill(sim, agent, node, poi, self_pay, "hospital", days, step, sim_min, bud)
    st["hospital_bills"] += 1


def _agent_by_id(sim, agent_id: int):
    if int(agent_id) < 0:
        return None
    for agent in sim.agents:
        if int(agent.id) == int(agent_id):
            return agent
    return None


def _clear(agent) -> None:
    """搬送・在院の印を落とす(冪等)。**身体状態は 1 バイトも触らない**。

    ★この関数が据える欄と既定値が「在院/搬送の印の全数」の**正典**である。
      ``world/pool.py`` の ``_MED_FIELDS`` はその写しで(world/ は下層なので
      society 直下を import できない = ``_RUMOR_KEY`` と同じ作法)、一致は
      ``tests/test_pool_rotation.py::test_med_field_list_mirrors_medical_clear``
      が機械固定する。欄を増やすときは両方を同時に直すこと。
      第109 レーン D1 以前は退避に 1 欄も載っておらず、**入院中の個体がプール回転で
      即退院して病院から消えていた**(在院日数の較正が静かに壊れる)。
    """
    agent.med_admitted = False
    agent.med_transport_until = -1
    agent.med_until = -1
    agent.med_since = -1
    agent.med_crew = -1
    agent.med_confirmed = -1
    agent.med_node = ""
    agent.med_poi = ""
    agent.med_dest_node = ""
    agent.med_dest_poi = ""
    agent.med_dest_building = ""
    agent.med_from_node = ""


def in_care(agent) -> bool:
    """搬送中または在院中か(観測・テスト用の読み取り口)。"""
    return bool(getattr(agent, "med_admitted", False)) \
        or int(getattr(agent, "med_transport_until", -1)) >= 0


# --------------------------------------------------------------------------- #
# 観測タリー(**OFF では None = 何も出さない**。city_ops.provenance と同じ作法)
# --------------------------------------------------------------------------- #
def provenance(sim) -> dict | None:
    """医療の観測タリー(既定 OFF は None)。★三本足の合計をここで正直に出す。"""
    if not enabled(sim):
        return None
    cfg = cfg_of(sim)
    out: dict = {"schema": SCHEMA, "n_hospitals": len(hospitals(sim)),
                 "n_clinics": len(clinics(sim)),
                 "coinsurance": float(cfg["money"]["coinsurance"])}
    st = state_of(sim)
    if st is None:
        out.update({"transports": 0, "admits": 0, "discharges": 0,
                    "no_hospital": 0, "by_kind": {}})
        return out
    out.update({
        "transports": int(st["transports"]),
        "no_hospital": int(st["no_hospital"]),
        "admits": int(st["admits"]),
        "admits_by_sev": {k: int(v) for k, v in sorted(st["admits_by_sev"].items())},
        "discharges": int(st["discharges"]),
        "died_in_care": int(st["died_in_care"]),
        "bed_steps": int(st["bed_steps"]),
        "clinic_bills": int(st["clinic_bills"]),
        "hospital_bills": int(st["hospital_bills"]),
        "venue_missing": int(st["venue_missing"]),
        # 三本足の合計(① 公費 / ② 自己負担 / ③ 保険給付)。
        "ems_cost_total": round(float(st["ems_cost_total"]), 1),
        "ems_funded": round(float(st["ems_funded"]), 1),
        "self_pay_total": round(float(st["self_pay_total"]), 1),
        "insurance_total": round(float(st["insurance_total"]), 1),
        # ★受け手 org を特定できなかった給付(街の残高は 1 円も動いていない)。
        "insurance_to_row": round(float(st["insurance_to_row"]), 1),
        "payee_resolved": int(st["payee_resolved"]),
        "payee_row": int(st["payee_row"]),
        "dropped": int(st["dropped"]),
        "by_kind": {k: int(v) for k, v in sorted(st["by_kind"].items())}})
    return out
