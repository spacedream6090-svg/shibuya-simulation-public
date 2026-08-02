"""心モデル固定 + 三層知能配置(第88バッチ 2026-08-03)。

正典: docs/plans/source/design-discussion-20260802.md §5 /
      docs/plans/dayplan-engaged-plan.md 第88。

原文書 §5 が決めたこと(そのまま実装する)
------------------------------------------
- **心(会話・思考・価値判断)は 1 エージェント 1 モデルを誕生時に固定する**。理由は
  ①人格の一貫性 ②モデル差がそのまま人口の多様性になる(§1 の分散崩壊への最も安い対抗策)
  ③面白いエージェントが見つかったときにモデル別分析ができる。
- **機械的判断(経路選択・定型購買)はモデル不問**= 共有の小型モデルまたはルール。本リポでは
  経路選択も定型購買も**そもそも LLM を呼んでいない**(ルール層)ので、ここは実装ではなく
  **確認**である(tests/test_mind.py が「機械的判断の module が generate を呼ばない」を固定する)。
- **モデルと人格の交絡が生じるため、agent_id と model_id の対応は必ずログに残す**
  (manifest / agents.json / L1 `mind_assign`)。これが本バッチで最も落とせない要求。
- 知能は均質に配置しない。**基底層**(全員が習慣・スケジュールで動く=ほぼ無料・LLM なし)/
  **思考層**(驚き発火時のみ LLM・混成 fleet)/ **高解像度層**(1〜5% に大型モデルと高頻度思考)。

三層の実体(本リポの既存資産との対応)
--------------------------------------
| 層 | 何が動くか | 実装 |
|---|---|---|
| 基底層 | 習慣・スケジュール・経路・購買。**LLM を 1 本も呼ばない** | 既存(routine / mobility / commerce)。本 module は 1 バイトも触らない |
| 思考層 | 発火したときだけ LLM。混成 fleet = `mind.pool` の重み付き割当 | 本 module の `pool` |
| 高解像度層 | 1〜5%。大型モデル(`tiers.high.name`)+ 夜内省の対象 + 思考頻度の上限緩和 | 本 module の `tiers.high` |

高解像度層の選抜は **traits 非依存の一様抽選**を既定にする。理由は k*(因子→世界改変)の
研究設計と交絡させないためで、「あらかじめ賢い個体を選んでおく」と『誰が世界を変えるか』
という問い自体が自壊する(サーベイ S-08 の設計主張)。traits 依存の選抜は**アブレーション
専用オプション**(`tiers.high.select: traits`)として分離して宣言してある。

決定論(R1)
------------
- 割当は **専用 stream** `mind_model`(モデル)と `mind_tier`(層)からのみ引く。RngHub の
  stream は stateless(キーから独立に派生する)ので、新 stream を足しても既存 stream の
  draw 順は 1 つも動かない(第75 dunbar / 第78 shuffle_partners と同じ根拠)。
- したがって割当は (master_seed, agent_id) の純関数 = resume でも pool 再入場でも同一。
  checkpoint に割当表を積む必要がない(`_mind_binding` は単なるメモ化キャッシュ)。
- **generate() の呼び出しサイトを 1 つも足さない/減らさない**。本 module が変えるのは
  「どのモデルが答えるか」だけで、「いつ何本呼ぶか」は 1 バイトも触らない。
  ただし高解像度層が夜内省の選抜を引き継ぐ(§8 突入 5)ため、engaged ON のランでは
  ON/OFF で**エピソード集合が変わる** → registry の affects_k は **True** を正直に宣言する。

既定 OFF(`model.mind.enabled: false`)
--------------------------------------
`build_cfg` が enabled=False を返し、Simulation は解決層を組まない・`agent.mind` 属性を
生やさない・新 stream を 1 本も引かない・manifest / summary / agents.json / L2 とも
既存ランとバイト一致になる。

正直な限界
----------
- モデル別の集計は「そのモデルに割り当てられた個体の集計」であって、モデルの因果効果では
  ない。割当は乱数なので**交絡は無いが**、モデル差と個体差を分けるには同一個体を別モデルで
  走らせる対照(= 第89 プラセボ L1 / 第90 バッテリー)が要る。
- `tiers.high.select: traits` を使うと選抜が因子と相関する。これは**意図的な交絡**であり、
  k* の主張には使えない(アブレーション対照専用)。ON にすると manifest に警告が残る。
- 実モデルのショートリスト(3B〜14B×5〜6本)は **DP-U2 = ユーザー判断**。本 module は
  「名前と重みを受け取って割り当てる」までで、どのモデルを積むかは決めない。
"""
from __future__ import annotations

SCHEMA = 1

# 三層(原文書 §5)。base は「全員・常時」なので個体属性としては持たない
# (LLM を呼ばない層 = 属性を持つ必要が無い)。個体が持つのは think / high の別だけ。
TIER_BASE = "base"
TIER_THINK = "think"
TIER_HIGH = "high"

# 「心」の呼び出し = 会話・思考・価値判断。rng_key の先頭セグメント(purpose)で見分ける。
#   deliberate … 発話 / 返答 / 投稿 / DM / 独り言(会話と価値判断)
#   plan       … 朝の一日計画(思考)
#   reflect    … 夜の内省(思考)
#   recall     … 内省前の能動的想起(思考の一部・同一個体の内面)
# ここに無い purpose(現状 "null" = D7 対照系列)は共有の既定バックエンドが受ける
# (= 機械的判断と同じ扱い。内容非結合のダミーなので人格に属さない)。
MIND_PURPOSES: tuple[str, ...] = ("deliberate", "plan", "reflect", "recall")

# 高解像度層の選抜方式。
#   uniform … 一様抽選(既定)。traits を 1 つも読まない = k* 研究と直交する。
#   traits  … 指定 trait の上位 frac。**アブレーション専用**(意図的に交絡させる対照)。
SELECT_MODES = ("uniform", "traits")

_DEFAULTS: dict = {
    "enabled": False,
    "purposes": list(MIND_PURPOSES),
    "high": {
        "frac": 0.0,
        "name": "",
        "spec": None,
        "reflect": True,
        "cap_mult": 1.0,
        "select": "uniform",
        "select_trait": "",
    },
}


# --------------------------------------------------------------------------- #
# 設定
# --------------------------------------------------------------------------- #
def build_cfg(raw: dict | None) -> dict:
    """conf の `model.mind` を正準化する(既定 OFF = 現行挙動と完全同一)。

    pool の各要素は **router の子 spec そのもの**(backend / name / base_url / host /
    api_key_env / format / timeout_s …)+ `weight`。`name` が model_id になる
    (= 割当・ログ・summary の全てで使う識別子)。新しいバックエンド型は 1 つも作らない。
    """
    raw = dict(raw or {})
    cfg: dict = {"enabled": bool(raw.get("enabled", False)),
                 "pool": [], "high": dict(_DEFAULTS["high"])}
    seen: set[str] = set()
    for entry in (raw.get("pool") or []):
        e = dict(entry or {})
        name = str(e.get("name", "")).strip()
        if not name:
            raise ValueError("model.mind.pool の各要素には name(= model_id)が要る。")
        if name in seen:
            raise ValueError(f"model.mind.pool に重複した name '{name}' がある"
                             "(model_id は一意でなければ割当もキャッシュも混ざる)。")
        seen.add(name)
        weight = float(e.pop("weight", 1.0))
        if weight < 0.0:
            raise ValueError(f"model.mind.pool['{name}'].weight は 0 以上。")
        cfg["pool"].append({"name": name, "weight": weight, "spec": _spec(e)})
    purposes = raw.get("purposes")
    cfg["purposes"] = tuple(str(p) for p in purposes) if purposes else MIND_PURPOSES
    # ---- 三層。base は「全員・LLM なし」で個体属性を持たない(宣言だけ manifest に残す)。
    tiers = dict(raw.get("tiers") or {})
    high = dict(tiers.get("high") or {})
    hi = cfg["high"]
    hi["frac"] = min(1.0, max(0.0, float(high.get("frac", 0.0))))
    hi["name"] = str(high.get("name", "") or "").strip()
    hi["reflect"] = bool(high.get("reflect", True))
    hi["cap_mult"] = max(1.0, float(high.get("cap_mult", 1.0)))
    hi["select"] = str(high.get("select", "uniform"))
    if hi["select"] not in SELECT_MODES:
        raise ValueError(f"model.mind.tiers.high.select は {SELECT_MODES} のいずれか"
                         f"(受け取った値: {hi['select']})。")
    hi["select_trait"] = str(high.get("select_trait", "") or "")
    spec = dict(high.get("spec") or {})
    if hi["name"] and spec:
        spec.setdefault("name", hi["name"])
        hi["spec"] = _spec(spec)
    elif hi["name"]:
        # spec 省略 = pool と同じ backend 種別で名前だけ違うモデル(mock 検証で使う形)。
        hi["spec"] = None
    if cfg["enabled"] and not cfg["pool"]:
        raise ValueError("model.mind.enabled=true には model.mind.pool(1 件以上)が要る。")
    return cfg


def _spec(entry: dict) -> dict:
    """pool/high の要素から「router の子 spec」を切り出す(weight などの管理欄は落とす)。"""
    return {k: v for k, v in entry.items() if k not in ("weight",)}


def cfg_of(sim) -> dict:
    cfg = getattr(sim, "mindcfg", None)
    return cfg if cfg else build_cfg(None)


def enabled(sim) -> bool:
    return bool(cfg_of(sim)["enabled"])


# --------------------------------------------------------------------------- #
# 割当(誕生時 1 回・専用 stream・traits 非依存)
# --------------------------------------------------------------------------- #
def _u(hub, stream: str, agent_id: int) -> float:
    """専用 stream から一様値 [0,1) を 1 つ引く(既存 stream の draw 順に無風)。"""
    return float(hub.stream(stream, int(agent_id)).random())


def choose_model(pool: list, u: float) -> str:
    """重み付き累積分割で model_id を 1 つ選ぶ(**name 昇順**= conf の並び順に依らない)。"""
    items = sorted((str(e["name"]), max(0.0, float(e["weight"]))) for e in pool)
    total = sum(w for _, w in items)
    if total <= 0.0:
        return items[0][0] if items else ""
    r = u * total
    cum = 0.0
    for name, w in items:
        cum += w
        if r < cum:
            return name
    return items[-1][0]                    # 数値誤差の保険


def _high_pick(cfg: dict, hub, agent_id: int, traits) -> bool:
    """高解像度層に入るか。既定は traits を 1 つも読まない一様抽選。"""
    hi = cfg["high"]
    frac = float(hi["frac"])
    if frac <= 0.0:
        return False
    if frac >= 1.0:
        return True
    if hi["select"] == "traits":
        # ★アブレーション専用の**意図的な交絡**。trait 値が [0,1] 一様であることを前提に
        #   上位 frac を採る(名指しは conf の select_trait であって基盤コードではない)。
        key = hi["select_trait"]
        if key and traits:
            try:
                return float(traits.get(key, 0.0)) >= 1.0 - frac
            except (TypeError, ValueError):
                return False
        return False
    return _u(hub, "mind_tier", agent_id) < frac


def derive(cfg: dict, hub, agent_id: int, traits=None) -> tuple[str, str]:
    """(model_id, tier) を決定論で導く。**誕生時と解決層が共有する唯一の式**。"""
    tier = TIER_HIGH if _high_pick(cfg, hub, agent_id, traits) else TIER_THINK
    model = choose_model(cfg["pool"], _u(hub, "mind_model", agent_id))
    if tier == TIER_HIGH and cfg["high"]["name"]:
        model = cfg["high"]["name"]        # 高解像度層は pool とは別の大型モデル
    return model, tier


def assign(sim, agent) -> None:
    """誕生時(ペルソナ確定時)に 1 回だけ心のモデルと層を固定する(OFF は完全 no-op)。

    名簿経路(直接ラン)と pool ローテーション経路の**両方**から呼ぶ。乱数は専用 stream
    2 本のみ = 既存 stream の draw 順に無風。以後この個体の「心」の呼び出しは全部このモデルへ。
    """
    cfg = cfg_of(sim)
    if not cfg["enabled"]:
        return
    model, tier = derive(cfg, sim.hub, int(agent.id),
                         getattr(agent, "traits", None))
    hi = cfg["high"]
    # reflect_hook は**全個体に同じ値**を持たせる(engaged 側が sim を触らずに
    # 「この接続は生きているか」を判定できるようにするため。cfg の写しであって状態ではない)。
    agent.mind = {"model": model, "tier": tier,
                  "reflect_hook": bool(hi["reflect"]),
                  "cap_mult": (float(hi["cap_mult"]) if tier == TIER_HIGH else 1.0)}
    binding = getattr(sim, "_mind_binding", None)
    if binding is not None:
        binding[int(agent.id)] = model


def model_of(agent) -> str | None:
    """個体に固定された心のモデル(OFF では属性が無い = None)。"""
    m = getattr(agent, "mind", None)
    return str(m["model"]) if m else None


def tier_of(agent) -> str | None:
    m = getattr(agent, "mind", None)
    return str(m["tier"]) if m else None


def cap_mult(agent) -> float:
    """思考頻度の上限(engaged の turn_cap / replan_cap)にかける倍率。既定 1.0=無風。"""
    m = getattr(agent, "mind", None)
    return float(m["cap_mult"]) if m else 1.0


def reflect_override(agent) -> bool | None:
    """夜内省(§8 突入 5)の対象か。**高解像度層だけが対象**。

    mind OFF、または `tiers.high.reflect: false`(接続を切った設定)では None を返し、
    第87 の既存判定(conf `reflect_frac` × `_stable_hash`)にそのまま委ねる。
    """
    m = getattr(agent, "mind", None)
    if not m or not m.get("reflect_hook"):
        return None
    return m["tier"] == TIER_HIGH


def model_of_id(sim, agent_id) -> str | None:
    """agent_id → 心のモデル(**解決層が呼ぶ唯一の入口**)。OFF は None。

    ①メモ化キャッシュ ②在場個体の属性 ③純関数の再導出、の順に解決する。
    ③があるので resume でも pool の再入場でも同じモデルに戻る(割当表を checkpoint に
    積む必要が無い = 誕生時固定が (master_seed, agent_id) の純関数だから)。
    """
    cfg = cfg_of(sim)
    if not cfg["enabled"]:
        return None
    try:
        aid = int(agent_id)
    except (TypeError, ValueError):
        return None
    binding = getattr(sim, "_mind_binding", None)
    if binding is None:
        binding = {}
        sim._mind_binding = binding
    mid = binding.get(aid)
    if mid is not None:
        return mid
    agent = (getattr(sim, "agent_by_id", None) or {}).get(aid)
    mid = model_of(agent) if agent is not None else None
    if mid is None:
        mid, _tier = derive(cfg, sim.hub, aid,
                            getattr(agent, "traits", None) if agent else None)
    binding[aid] = mid
    return mid


# --------------------------------------------------------------------------- #
# エピソード / 計画のログに載せる model_id(第86 day_plan・第87 engaged との整合)
# --------------------------------------------------------------------------- #
def log_model_id(sim, agent=None) -> str:
    """L1 / summary に残す model_id。mind ON なら**その個体に固定されたモデル**、
    OFF なら従来どおりバックエンド名(第86/87 の暫定値)。"""
    if agent is not None:
        mid = model_of(agent)
        if mid:
            return mid
    llm = getattr(sim, "llm", None)
    name = getattr(llm, "name", None)
    if not name:
        name = getattr(getattr(llm, "backend", None), "name", None)
    return str(name or "unknown")


# --------------------------------------------------------------------------- #
# 観測(manifest / summary / L2)
# --------------------------------------------------------------------------- #
def _child_stats(sim) -> dict:
    """解決層の子(= モデル別 CachedLLM)の呼数・命中数。無ければ空。"""
    router = getattr(sim, "llm", None)
    children = getattr(router, "children", None)
    if not isinstance(children, dict):
        return {}
    out: dict = {}
    for mid, child in children.items():
        out[str(mid)] = {"calls": int(getattr(child, "calls", 0)),
                         "hits": int(getattr(child, "hits", 0))}
    return out


def scalars(sim) -> dict | None:
    """L2 の最小 2 列(OFF は None = 列なし = L2 不変。engaged.scalars と同型)。

    既存の by_model 集計(第86 day_plan / 第87 engaged)は summary 側にあるので、L2 には
    「いま在場している個体の層の分布」だけを最小限で置く(モデル数は conf 依存で可変なので
    **列名にモデル名を出さない**= 列構成がランごとに変わらない)。
    """
    if not enabled(sim):
        return None
    agents = getattr(sim, "agents", None) or []
    models = set()
    n_high = 0
    for a in agents:
        m = getattr(a, "mind", None)
        if not m:
            continue
        models.add(m["model"])
        if m["tier"] == TIER_HIGH:
            n_high += 1
    return {"mind_models_present": float(len(models)),
            "mind_high_agents": float(n_high)}


def provenance(sim) -> dict | None:
    """manifest / summary の `mind` キー(OFF はキー自体を出さない)。

    §5 の「モデルと人格の交絡が生じるため必ずログに残す」を満たす場所。
    agent_id → model_id の**個別対応**は agents.json と L1 `mind_assign` にあり、
    ここにはその集計(モデル別の人数・呼数・エピソード数・修復率)を置く。
    """
    if not enabled(sim):
        return None
    cfg = cfg_of(sim)
    hi = cfg["high"]
    agents = getattr(sim, "agents", None) or []
    by_model: dict = {}
    n_high = 0
    for a in agents:
        m = getattr(a, "mind", None)
        if not m:
            continue
        row = by_model.setdefault(m["model"], {"agents": 0, "high": 0})
        row["agents"] += 1
        if m["tier"] == TIER_HIGH:
            row["high"] += 1
            n_high += 1
    for mid, st in _child_stats(sim).items():          # 呼数・キャッシュ命中(モデル別)
        row = by_model.setdefault(mid, {"agents": 0, "high": 0})
        row["calls"] = st["calls"]
        row["cache_hits"] = st["hits"]
    # 第87 engaged: エピソード数・ターン・滞在(model_id は本バッチで固定値へ整合済み)。
    for mid, row in (getattr(sim, "_engaged_state", None) or {}).get("by_model", {}).items():
        tgt = by_model.setdefault(str(mid), {"agents": 0, "high": 0})
        tgt["episodes"] = int(row.get("episodes", 0))
        tgt["episode_turns"] = int(row.get("turns", 0))
        tgt["episode_stay_min"] = int(row.get("stay_min", 0))
    # 第86 day_plan: 計画数と修復率・後退率(原文書 §7「修復発生回数のモデル別集計」)。
    for mid, row in (getattr(sim, "_dayplan_state", None) or {}).get("by_model", {}).items():
        tgt = by_model.setdefault(str(mid), {"agents": 0, "high": 0})
        n = max(1, int(row.get("plans", 0)))
        tgt["plans"] = int(row.get("plans", 0))
        tgt["repair_rate"] = round(int(row.get("repair", 0)) / n, 4)
        tgt["fallback_rate"] = round(int(row.get("fallback", 0)) / n, 4)
    out: dict = {
        "schema": SCHEMA,
        "purposes": list(cfg["purposes"]),
        "pool": [{"name": e["name"], "weight": e["weight"]} for e in cfg["pool"]],
        "tiers": {
            # 基底層は「全員・常時・LLM なし」= 既存のルール層そのもの(宣言のみ)。
            "base": {"agents": len(agents), "llm": False,
                     "note": "習慣・スケジュール・経路・購買。LLM を 1 本も呼ばない既存層"},
            "think": {"agents": max(0, len(agents) - n_high),
                      "source": "model.mind.pool(重み付き・誕生時固定)"},
            "high": {"agents": n_high, "frac": hi["frac"], "name": hi["name"],
                     "reflect": hi["reflect"], "cap_mult": hi["cap_mult"],
                     "select": hi["select"]},
        },
        "n_bound": len(getattr(sim, "_mind_binding", {}) or {}),
        "by_model": {k: by_model[k] for k in sorted(by_model)},
        "confound_note": ("モデル差と個体差は本ランでは分離できない。割当は traits 非依存の"
                          "決定論なので**交絡は無い**が、モデルの因果効果を主張するには"
                          "同一個体を別モデルで走らせる対照(第89/第90)が要る"),
    }
    if hi["select"] == "traits":
        out["tiers"]["high"]["select_trait"] = hi["select_trait"]
        out["warning"] = ("tiers.high.select=traits は**意図的に因子と交絡させる**"
                          "アブレーション専用の選抜。k* の主張には使えない")
    return out
