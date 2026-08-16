"""行動方針キャッシュ(AGA Lifestyle Policy)= P2 スライス S7。

行動系列(朝計画・日中熟慮の出力)を **k 非依存の物理量のみ** をキーにキャッシュし、
似た状況を再訪したら LLM 呼び出しをスキップして保存済み出力を再利用する。設計:
docs/research/interstitial-life.md §2.7/§4.4(4-b)/§7・docs/plans/p2-interstitial-design.md §1 S7。

R1 の関門(interstitial-life §7・最大の要注意点 ④-b):
  ① キー = ペルソナ静的プロファイル(職業・年代)+ 状況骨格(曜日型・時間帯・場所種別・
     天気・活動 cat)のみ。**beliefs・k・Y_internal・writeback 状態・opinion・drive は完全排除**。
     → キー構築は `SituationSkeleton`(物理量だけを持つ frozen dataclass)を唯一の入力とし、
        信念系を受け取れない型で静的に担保する(test_policy_cache の関門③)。
  ② 呼数 k 非依存: 再利用の判定(キー一致・確率ゲート・ハード上限)は k と無関係な観測量
     だけで決まる → k∈{free,off} で再利用数・LLM 呼数が一致する。
  ③ 既定 OFF(cfg なし/enabled=false)は **完全 no-op**(cache も stream も作らず・イベントも
     出さない)= ゴールデン L1 バイト一致。

類似判定は埋め込みモデルを使わず(依存を増やさない)、構造化タプルの完全一致 + 段階的緩和
(完全一致 → 場所種別を落とす → 時間帯を落とす)の決定論 near-match。再利用にも確率ゲート
(専用 stream ``policy_cache`` 1 本)+ 再利用率のハード上限を掛け、行動レパートリーの下限
(多様性)を保証する。キャッシュは有界(個体毎上限 + 全体 LRU)・checkpoint 対象外(揮発でよい・
決定論は stream とキー由来なので復元不要)。

接続点(呼び出し側):
  - 朝計画: cognition/planning.py の make_plan 入口(本スライスで配線済み・稼働)。
  - 日中熟慮: cognition/deliberate.py の入口関数(reuse_action/store_action。scheduler への
    差し込みは主計画者が統合。本モジュールは cognition 層の関数入口を提供する)。
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, fields

from ..observer import decision_mode as _dec_mode
from ..observer.schema import Event
from ..world import calendar as _calendar

_UNSET = object()
_WILDCARD = "*"          # near-match 緩和で「その次元を無視する」ことを表す番兵


# --------------------------------------------------------------------------- #
# config(cognition.policy_cache。既定/未設定/enabled=false は None=完全 OFF=no-op)
# --------------------------------------------------------------------------- #
def _build_cfg(raw: dict) -> dict:
    return {
        # 再利用率の上限(確率ゲートの p 兼ハード上限)。多様性の下限を保証する。
        "reuse_rate_cap": float(raw.get("reuse_rate_cap", 0.6)),
        "per_agent_max": int(raw.get("per_agent_max", 16)),      # 個体あたりキャッシュ上限
        "global_max": int(raw.get("global_max", 4096)),          # 全体 LRU 上限
        # near-match 緩和の最大段(0=完全一致のみ / 1=+場所種別を落とす / 2=+時間帯を落とす)。
        "max_relax": max(0, min(2, int(raw.get("max_relax", 2)))),
    }


def cache_cfg(sim):
    """cognition.policy_cache を数値化して sim に一度だけキャッシュ。既定/未設定/enabled=false
    は None を返し、呼び出し側は完全 no-op(既存 _stochastic_cfg と同じ遅延キャッシュ作法)。"""
    cached = getattr(sim, "_policy_cache_cfg", _UNSET)
    if cached is not _UNSET:
        return cached
    raw = (sim.cfg.get("cognition", {}) or {}).get("policy_cache", {}) or {}
    cfg = _build_cfg(raw) if raw.get("enabled", False) else None
    sim._policy_cache_cfg = cfg
    return cfg


# --------------------------------------------------------------------------- #
# 状況骨格 → キー(k 非依存の物理量のみ)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SituationSkeleton:
    """キャッシュキーの唯一の入力。**物理量・環境量・静的プロファイルだけ**を持つ。

    ここに beliefs / k / Y_internal / writeback / opinion / drive を **決して入れない**
    (interstitial-life §7 関門①)。test_policy_cache の静的チェックが本 dataclass のフィールド
    名を検査し、信念系の語が混入していないことを固定する。
    """
    persona: str        # ペルソナ静的プロファイル(職業/年代=固定属性のみ)
    weekday_type: str   # 曜日型(workday/holiday/na)
    time_band: str      # 時間帯(深夜/朝/昼/午後/夕方/夜)
    place_kind: str     # 場所種別(home/street/POI カテゴリ)
    weather: str        # 天気(cond。無効時は na)
    activity_cat: str   # 活動 cat(mandatory/maintenance/discretionary)or 熟慮トリガ


# 信念系(k の作用点)を表す語。SituationSkeleton にこれらが混じっていないことを test が検査する。
FORBIDDEN_KEY_TERMS = frozenset({
    "belief", "beliefs", "k", "y_internal", "writeback", "written_back",
    "opinion", "drive", "theta", "arousal", "self_model", "implicit",
})


def build_key(skel: SituationSkeleton) -> tuple:
    """物理骨格 → 構造化タプルキー。入力は SituationSkeleton **のみ**(信念系を受け取れない型)。"""
    return (skel.persona, skel.weekday_type, skel.time_band,
            skel.place_kind, skel.weather, skel.activity_cat)


# タプル内の次元インデックス: 0 persona / 1 weekday / 2 time_band / 3 place_kind /
#                             4 weather / 5 activity_cat
def reduced_key(key: tuple, stage: int) -> tuple:
    """near-match 緩和後のキー。stage 0=完全一致 / 1=場所種別を落とす / 2=+時間帯を落とす。

    persona・weekday_type・weather・activity_cat は常に保持(=決定論 near-match の骨格)。"""
    k = list(key)
    if stage >= 1:
        k[3] = _WILDCARD        # 場所種別を落とす
    if stage >= 2:
        k[2] = _WILDCARD        # 時間帯を落とす
    return tuple(k)


# --------------------------------------------------------------------------- #
# 有界キャッシュ(個体毎上限 + 全体 LRU)。checkpoint 対象外(揮発)。
# --------------------------------------------------------------------------- #
class PolicyCache:
    def __init__(self) -> None:
        self.entries: "OrderedDict[tuple, object]" = OrderedDict()  # (aid,key)->output(全体 LRU 順)
        self.by_agent: dict = {}          # aid -> OrderedDict(key -> None)(個体毎順・near-match 走査用)
        self.n_opportunities = 0          # 再利用機会の総数(hit/miss 問わず)
        self.n_reuses = 0                 # 実際に再利用した数(ハード上限の分子)

    def lookup(self, agent_id: int, key: tuple, max_relax: int):
        """段階的緩和で最も新しい一致を返す ((output, relax_stage))。無ければ None。"""
        ag = self.by_agent.get(agent_id)
        if not ag:
            return None
        for stage in range(0, max_relax + 1):
            want = reduced_key(key, stage)
            for k in reversed(ag):                       # 新しい順(OrderedDict 末尾=最新)
                if reduced_key(k, stage) == want:
                    gk = (agent_id, k)
                    output = self.entries.get(gk)
                    self.entries.move_to_end(gk)         # LRU タッチ
                    ag.move_to_end(k)
                    return output, stage
        return None

    def store(self, agent_id: int, key: tuple, output, per_agent_max: int,
              global_max: int) -> None:
        gk = (agent_id, key)
        self.entries[gk] = output
        self.entries.move_to_end(gk)
        ag = self.by_agent.setdefault(agent_id, OrderedDict())
        ag[key] = None
        ag.move_to_end(key)
        while len(ag) > max(1, per_agent_max):           # 個体毎上限(古い順に追い出し=LRU)
            old_key, _ = ag.popitem(last=False)
            self.entries.pop((agent_id, old_key), None)
        while len(self.entries) > max(1, global_max):    # 全体 LRU 上限
            (oa, ok), _ = self.entries.popitem(last=False)
            oag = self.by_agent.get(oa)
            if oag is not None:
                oag.pop(ok, None)
                if not oag:
                    self.by_agent.pop(oa, None)

    def size(self) -> int:
        return len(self.entries)


def _get_cache(sim) -> PolicyCache:
    c = getattr(sim, "_policy_cache", None)
    if c is None:
        c = PolicyCache()
        sim._policy_cache = c
    return c


# --------------------------------------------------------------------------- #
# 骨格の抽出(すべて物理量・k 非依存)
# --------------------------------------------------------------------------- #
def _band(sim_min: int) -> str:
    h = (sim_min % 1440) // 60
    return ("深夜" if h < 5 else "朝" if h < 10 else "昼" if h < 16
            else "夕方" if h < 19 else "夜")


def _persona_profile(agent) -> str:
    """ペルソナ静的プロファイル=固定属性(職業・年代)。k・可変内面は一切見ない。"""
    age = int(getattr(agent, "age", 0) or 0)
    return f"{getattr(agent, 'occupation', '') or ''}/{age // 10 * 10}"


def _weekday_type(sim, sim_min: int) -> str:
    cal = getattr(sim, "calendarcfg", None)
    if not cal:
        return "na"
    return "holiday" if _calendar.is_holiday(cal, sim_min) else "workday"


def _weather_cond(sim) -> str:
    w = getattr(sim, "today_weather", None)
    if isinstance(w, dict):
        return str(w.get("cond", "na"))
    return "na"


# 実行時 activity ラベル → 3分類(routine._ACT_CAT と同一。未知/自由は discretionary)。
_ACT_CAT = {"working": "mandatory", "commuting": "mandatory",
            "eating": "maintenance", "shopping": "maintenance"}


def _place_kind(sim, agent) -> str:
    node = getattr(agent, "node", "") or ""
    if node and node == (getattr(agent, "home_node", "") or ""):
        return "home"
    try:
        pois = sim.city.pois_at_node(node)
    except Exception:
        pois = None
    if pois:
        return str(pois[0].get("cat", "poi") or "poi")
    return "street"


def _plan_skeleton(sim, agent, sim_min: int) -> SituationSkeleton:
    return SituationSkeleton(
        persona=_persona_profile(agent),
        weekday_type=_weekday_type(sim, sim_min),
        time_band=_band(sim_min),
        place_kind=_place_kind(sim, agent),
        weather=_weather_cond(sim),
        activity_cat=_ACT_CAT.get(getattr(agent, "activity", "") or "", "discretionary"),
    )


def _deliberate_skeleton(sim, agent, sim_min: int, trigger: str) -> SituationSkeleton:
    # 熟慮は「何をしようとしているか」= trigger(social/reply/post/dm/solo/…)が活動 cat の役割。
    cat = str(trigger) if trigger else _ACT_CAT.get(
        getattr(agent, "activity", "") or "", "discretionary")
    return SituationSkeleton(
        persona=_persona_profile(agent),
        weekday_type=_weekday_type(sim, sim_min),
        time_band=_band(sim_min),
        place_kind=_place_kind(sim, agent),
        weather=_weather_cond(sim),
        activity_cat=cat,
    )


# --------------------------------------------------------------------------- #
# 再利用ゲート(確率 + ハード上限)+ 観測
# --------------------------------------------------------------------------- #
def _copy(output):
    """再利用/保存で共有参照を切る(routine が day_plan[i]['done'] を書き換えても汚さない)。"""
    if isinstance(output, list):
        return [dict(it) if isinstance(it, dict) else it for it in output]
    if isinstance(output, dict):
        return dict(output)
    return output


def _gate_and_log(sim, cache: PolicyCache, cfg: dict, agent, step: int,
                  sim_min: int, kind: str, hit):
    """確率ゲート(stream ``policy_cache``)+ 再利用率のハード上限。通れば policy_reuse を
    1 件記録し出力のコピーを返す。落ちれば None(=呼び出し側は通常どおり LLM 生成)。"""
    output, stage = hit
    cap = cfg["reuse_rate_cap"]
    r = float(sim.hub.stream("policy_cache", agent.id, step, kind).random())
    if r >= cap:                                          # 確率ゲート(多様性の下限)
        return None
    if cache.n_reuses + 1 > cap * cache.n_opportunities:  # 再利用率のハード上限
        return None
    cache.n_reuses += 1
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                         kind="policy_reuse", x=agent.x, y=agent.y,
                         payload={"kind": kind, "relax": stage, "saved": 1}))
    # V3 決定モード(observer.decision_mode。既定 OFF は即 return)。`policy_reuse` は
    # L1 に既に出ているが、決定モードの内訳を summary 1 枚で閉じるためここでも数える
    # (L1 を読み直さずに「LLM / rule / reuse」のシェアが出せる)。
    _dec_mode.note_reuse(sim, sim_min, kind)
    return _copy(output)


# --------------------------------------------------------------------------- #
# 朝計画(planning.make_plan が配線)
# --------------------------------------------------------------------------- #
def reuse_plan(sim, agent, step: int, sim_min: int):
    """朝計画の再利用を試みる。ヒット+ゲート通過なら **キャッシュ済み plan items のコピー** を
    返し(LLM をスキップ)、そうでなければ None(=呼び出し側が通常生成)。既定 OFF は即 None。"""
    cfg = cache_cfg(sim)
    if cfg is None:
        return None
    cache = _get_cache(sim)
    cache.n_opportunities += 1
    key = build_key(_plan_skeleton(sim, agent, sim_min))
    hit = cache.lookup(agent.id, key, cfg["max_relax"])
    if hit is None:
        return None
    return _gate_and_log(sim, cache, cfg, agent, step, sim_min, "plan", hit)


def store_plan(sim, agent, step: int, sim_min: int, items) -> None:
    """生成した朝計画 items をキャッシュへ格納(コピーを保持)。既定 OFF は no-op。"""
    cfg = cache_cfg(sim)
    if cfg is None:
        return
    cache = _get_cache(sim)
    key = build_key(_plan_skeleton(sim, agent, sim_min))
    cache.store(agent.id, key, _copy(items), cfg["per_agent_max"], cfg["global_max"])


# --------------------------------------------------------------------------- #
# 日中熟慮(cognition 層の関数入口。scheduler への差し込みは主計画者が統合)
# --------------------------------------------------------------------------- #
def reuse_action(sim, agent, step: int, sim_min: int, trigger: str):
    """日中熟慮の再利用を試みる。ヒット+ゲート通過なら **キャッシュ済み action dict のコピー**
    を返し(LLM をスキップ)、そうでなければ None。既定 OFF は即 None。"""
    cfg = cache_cfg(sim)
    if cfg is None:
        return None
    cache = _get_cache(sim)
    cache.n_opportunities += 1
    key = build_key(_deliberate_skeleton(sim, agent, sim_min, trigger))
    hit = cache.lookup(agent.id, key, cfg["max_relax"])
    if hit is None:
        return None
    return _gate_and_log(sim, cache, cfg, agent, step, sim_min, "deliberate", hit)


def store_action(sim, agent, step: int, sim_min: int, trigger: str, action) -> None:
    """生成した熟慮 action をキャッシュへ格納。action が None(壊れた JSON)なら格納しない。
    既定 OFF は no-op。"""
    cfg = cache_cfg(sim)
    if cfg is None or action is None:
        return
    cache = _get_cache(sim)
    key = build_key(_deliberate_skeleton(sim, agent, sim_min, trigger))
    cache.store(agent.id, key, _copy(action), cfg["per_agent_max"], cfg["global_max"])


def skeleton_field_names() -> tuple:
    """SituationSkeleton のフィールド名(関門③の静的チェック用)。"""
    return tuple(f.name for f in fields(SituationSkeleton))
