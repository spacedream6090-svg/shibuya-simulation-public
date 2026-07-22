"""L2 集計プラグイン(D12)。新指標の追加 = 関数を1個 register するだけ。"""
from __future__ import annotations

from typing import Any, Callable

from .. import status as _status_mod
from . import lens as _lens

AGGREGATORS: dict[str, Callable[[Any], float | int | str]] = {}


# --------------------------------------------------------------------------- #
# 崩壊検知(R12: 均質化ドリフトの非LLM監視)用の小さな純ヘルパ(stdlib のみ)
# --------------------------------------------------------------------------- #
def _last(seq) -> str:
    """list の末尾要素(空なら "")。"""
    if not seq:
        return ""
    return seq[-1] or ""


def _char3_set(s: str) -> set:
    """文字3-gram の集合(短文は全体を1要素)。"""
    if not s:
        return set()
    if len(s) < 3:
        return {s}
    return {s[i:i + 3] for i in range(len(s) - 2)}


def _type_token_ratio(text: str) -> float:
    """連結文字列の文字3-gram type/token 比(0..1)。空なら 0.0。"""
    if not text:
        return 0.0
    if len(text) < 3:
        return 1.0                       # 1 token・1 type
    grams = [text[i:i + 3] for i in range(len(text) - 2)]
    return len(set(grams)) / len(grams)


def _jaccard_distance(a: set, b: set) -> float:
    """1 - Jaccard 類似度。両方空なら 0.0(同一の空 = 距離0)。"""
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return 1.0 - len(a & b) / len(union)


def _gini(values) -> float:
    """ジニ係数(0=完全平等..1=完全集中)。numpy 非依存(ソート+累積)。負値・合計0は 0.0。"""
    xs = sorted(float(v) for v in values)
    n = len(xs)
    if n == 0:
        return 0.0
    total = sum(xs)
    if total <= 0.0:
        return 0.0
    cum = sum(i * x for i, x in enumerate(xs, start=1))
    return (2.0 * cum) / (n * total) - (n + 1.0) / n


def register_aggregator(name: str):
    def deco(fn: Callable[[Any], float | int | str]):
        AGGREGATORS[name] = fn
        return fn
    return deco


@register_aggregator("n_moving")
def _n_moving(sim) -> int:
    return sum(1 for a in sim.agents if a.route)


@register_aggregator("n_items")
def _n_items(sim) -> int:
    return len(sim.items.items)


@register_aggregator("total_adoptions")
def _total_adoptions(sim) -> int:
    return sum(len(a.adopted) for a in sim.agents)


@register_aggregator("distinct_vocab_in_use")
def _distinct_vocab(sim) -> int:
    used: set[str] = set()
    for a in sim.agents:
        used |= a.adopted
    return len(used)


@register_aggregator("mean_grievance")
def _mean_grievance(sim) -> float:
    if not sim.agents:
        return 0.0
    return round(sum(a.states["grievance"] for a in sim.agents) / len(sim.agents), 4)


@register_aggregator("n_outside")
def _n_outside(sim) -> int:
    return sum(1 for a in sim.agents if a.loc == "outside")


@register_aggregator("n_inside_buildings")
def _n_inside(sim) -> int:
    return sum(1 for a in sim.agents if a.building)


@register_aggregator("n_sleeping")
def _n_sleeping(sim) -> int:
    return sum(1 for a in sim.agents if a.sleeping)


@register_aggregator("n_cars")
def _n_cars(sim) -> int:
    """この step に範囲内を走った車の数(通過車両を含む)。"""
    return getattr(getattr(sim, "traffic", None), "last_n", 0)


@register_aggregator("n_working")
def _n_working(sim) -> int:
    return sum(1 for a in sim.agents if a.activity == "working")


@register_aggregator("n_sns_posts")
def _n_sns_posts(sim) -> int:
    net = getattr(sim, "net", None)
    return len(net.posts) if net else 0


@register_aggregator("n_drive_requests")
def _n_drive_requests(sim) -> int:
    return getattr(sim, "drive_stats", {}).get("requests", 0)


# ---- 意見力学(FJ #16): 全員の意見の分散・極端さ(乱数不使用の毎step L2 列) ----
@register_aggregator("opinion_var")
def _opinion_var(sim) -> float:
    """全エージェントの意見(-1..1)の分散。分極すると増え、収束すると 0 へ。"""
    ops = [getattr(a, "opinion", 0.0) for a in sim.agents]
    if not ops:
        return 0.0
    mean = sum(ops) / len(ops)
    return round(sum((o - mean) ** 2 for o in ops) / len(ops), 6)


@register_aggregator("opinion_extremity")
def _opinion_extremity(sim) -> float:
    """|意見| の平均(0=全員中立、1=全員が両極)。"""
    if not sim.agents:
        return 0.0
    return round(sum(abs(getattr(a, "opinion", 0.0))
                     for a in sim.agents) / len(sim.agents), 6)


# ---- SNS 反応(#14): いいね・リシェアの累積カウント ----
# B8(スケール): net 側の増分カウンタを参照(毎step全 posts 走査を廃止)。カウンタは
# post のエイジアウト(B7)後も総数を保持するので、trim なしでは従来の全走査和と完全一致。
@register_aggregator("n_likes")
def _n_likes(sim) -> int:
    net = getattr(sim, "net", None)
    if net is None:
        return 0
    total = getattr(net, "n_likes_total", None)
    if total is not None:
        return total
    return sum(len(p.get("likes", ())) for p in net.posts)   # 後退(旧 net)


@register_aggregator("n_reshares")
def _n_reshares(sim) -> int:
    net = getattr(sim, "net", None)
    if net is None:
        return 0
    total = getattr(net, "n_reshares_total", None)
    if total is not None:
        return total
    return sum(int(p.get("reshares", 0)) for p in net.posts)  # 後退(旧 net)


@register_aggregator("n_fires")
def _n_fires(sim) -> int:
    """この step に欲求発火で LLM が起動した回数(R1 監査用)。"""
    return getattr(sim, "drive_stats", {}).get("fires", 0)


@register_aggregator("n_face_fires")
def _n_face_fires(sim) -> int:
    """この step の対面会話の確定発火数(L2 監査用)。"""
    return getattr(sim, "drive_stats", {}).get("face_fires", 0)


@register_aggregator("n_replies")
def _n_replies(sim) -> int:
    """この step に返答保証で起動した返答 LLM の回数(L2 監査用)。"""
    return getattr(sim, "drive_stats", {}).get("replies", 0)


@register_aggregator("mean_drive")
def _mean_drive(sim) -> float:
    if not sim.agents:
        return 0.0
    return round(sum(a.drive for a in sim.agents) / len(sim.agents), 4)


@register_aggregator("mean_money")
def _mean_money(sim) -> float:
    """全エージェントの平均手持ち(経済 v0)。"""
    if not sim.agents:
        return 0.0
    return round(sum(getattr(a, "money", 0.0) for a in sim.agents) / len(sim.agents), 1)


@register_aggregator("n_broke")
def _n_broke(sim) -> int:
    """手持ちが逼迫している人数(残高 < money_pressure_threshold)。

    口座 ON(E5)時は現金+口座の合算で判定(OFF 時は現金のみ=従来と完全一致)。"""
    econ = getattr(sim, "economy", None)
    if not econ:
        return 0
    thr = float(econ.get("money_pressure_threshold", 0.0))
    accounts_on = bool(econ.get("accounts", {}).get("enabled"))
    return sum(1 for a in sim.agents
               if getattr(a, "money", 0.0)
               + (getattr(a, "account", 0.0) if accounts_on else 0.0) < thr)


@register_aggregator("mean_account")
def _mean_account(sim):
    """全エージェントの平均口座残高(口座 E5 ON 時のみ。OFF は None=列なし=L2 不変)。"""
    econ = getattr(sim, "economy", None)
    if not econ or not econ.get("accounts", {}).get("enabled"):
        return None
    if not sim.agents:
        return 0.0
    return round(sum(getattr(a, "account", 0.0) for a in sim.agents)
                 / len(sim.agents), 1)


@register_aggregator("mean_theta_drift")
def _mean_theta_drift(sim):
    """内省ドリフト theta_drift の平均(E2 ON 時のみ。OFF は None=列なし=L2 不変)。

    R1 監査補助: k∈{free,off} で n_fires が乖離しないことと併せて、ドリフトが k 非依存に
    閉じていることの観測窓にする。"""
    dc = getattr(sim, "drivecfg", None)
    if not dc or not dc.get("drift", {}).get("enabled"):
        return None
    if not sim.agents:
        return 0.0
    return round(sum(getattr(a, "theta_drift", 0.0) for a in sim.agents)
                 / len(sim.agents), 5)


# ---- 社会的ヒエラルキー(地位の集中・移動性。第11バッチ ON 時のみ。OFF は None=列なし=L2 不変) ----
@register_aggregator("status_gini")
def _status_gini(sim):
    """合成地位 status の集中度(ジニ係数)。マタイ効果で上位集中が進むと増える。OFF は None=列なし。"""
    if not _status_mod.enabled(sim):
        return None
    if not sim.agents:
        return 0.0
    return round(_gini(getattr(a, "status", 0.0) for a in sim.agents), 6)


@register_aggregator("status_top10_share")
def _status_top10_share(sim):
    """上位10%の地位が占める割合(winner-take-all の可視化)。OFF は None=列なし=L2 不変。"""
    if not _status_mod.enabled(sim):
        return None
    vals = sorted((float(getattr(a, "status", 0.0)) for a in sim.agents), reverse=True)
    total = sum(vals)
    if not vals or total <= 0.0:
        return 0.0
    k = max(1, int(len(vals) * 0.1))
    return round(sum(vals[:k]) / total, 6)


@register_aggregator("status_rank_mobility")
def _status_rank_mobility(sim):
    """地位順位の日次移動性(前日 rank との平均絶対差)。高い=流動/低い=固定(寡頭化)。OFF は None。"""
    if not _status_mod.enabled(sim):
        return None
    return round(float(getattr(sim, "_status_rank_mobility", 0.0)), 6)


# ---- 「世界を変える」ツールの使用量(累積カウント。R4: 客観カウント) ----
@register_aggregator("n_events_hosted")
def _n_events_hosted(sim) -> int:
    t = getattr(sim, "tools", None)
    return len(t.events) if t else 0


@register_aggregator("n_event_attend")
def _n_event_attend(sim) -> int:
    t = getattr(sim, "tools", None)
    return sum(len(e["attendees"]) for e in t.events.values()) if t else 0


@register_aggregator("n_flyers")
def _n_flyers(sim) -> int:
    t = getattr(sim, "tools", None)
    return t.n_flyers_total if t else 0


@register_aggregator("n_groups")
def _n_groups(sim) -> int:
    t = getattr(sim, "tools", None)
    return len(t.groups) if t else 0


@register_aggregator("n_proposals")
def _n_proposals(sim) -> int:
    t = getattr(sim, "tools", None)
    return len(t.proposals) if t else 0


@register_aggregator("n_ventures")
def _n_ventures(sim) -> int:
    t = getattr(sim, "tools", None)
    return t.n_ventures_total if t else 0


# ---- 行動心理プラグイン(既定 OFF)。OFF 時は None を返し collect が列を出さない=L2 不変 ----
@register_aggregator("mean_collective_efficacy")
def _mean_collective_efficacy(sim):
    """集団効力感 state の平均(集団プラグイン ON 時のみ。OFF は state 不在で None=列なし)。"""
    vals = [a.states["collective_efficacy"] for a in sim.agents
            if "collective_efficacy" in a.states]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 4)


# ---- 生活の自己決定 P2(D3 棚卸し。既定 全 OFF)。OFF は None=列なし=L2 不変 ----
def _p2_any_on(sim) -> bool:
    fc = getattr(sim, "freedomcfg", None)
    p2 = fc.get("p2") if fc else None
    return bool(p2 and any(p2.get(k) for k in
                           ("move_home", "buy", "study", "partnership", "deviance")))


@register_aggregator("freedom_choice_points")
def _freedom_choice_points(sim):
    """その step に P2 メニュー(生活の選択肢)が LLM に提示された回数。OFF は None=列なし。"""
    if not _p2_any_on(sim):
        return None
    return int(getattr(sim, "freedom_stats", {}).get("choice_points", 0))


@register_aggregator("freedom_exercised")
def _freedom_exercised(sim):
    """その step に既定ルーチンと異なる P2 選択(引っ越し/購入/学び/交際/無許可出店)を行使した回数。OFF は None。"""
    if not _p2_any_on(sim):
        return None
    return int(getattr(sim, "freedom_stats", {}).get("exercised", 0))


# ---- 共同行動エンジン(関係性の再現 第44バッチ)。OFF(joint も家族夕食も無効)は None=列なし=L2 不変 ----
@register_aggregator("n_joint_activity")
def _n_joint_activity(sim):
    """共同行動(友人系 joint + 世帯の夕食共食)の累積成立件数。joint も family_dinner も
    無効なら None=列なし(L2 バイト不変)。ON 時のみ joint_activity の累積を出す。"""
    jc = getattr(sim, "jointcfg", None)
    hc = getattr(sim, "householdcfg", None)
    joint_on = bool(jc and jc.get("enabled"))
    dinner_on = bool(hc and hc.get("enabled")
                     and hc.get("family_dinner", {}).get("enabled"))
    if not (joint_on or dinner_on):
        return None
    return int(getattr(sim, "_joint_total", 0))


# ---- サービスの実体化 第46バッチ ③(services)。OFF は None=列なし=L2 不変 ----
@register_aggregator("n_service_use")
def _n_service_use(sim):
    """サービス受給(理美容/クリニック/塾/ジム/クリーニング等)の累積件数。services 無効なら
    None=列なし(L2 バイト不変)。ON 時のみ service_use の累積を出す。"""
    sc = getattr(sim, "servicescfg", None)
    if not (sc and sc.get("enabled")):
        return None
    return int(getattr(sim, "_service_total", 0))


# ---- B2B 卸→小売 第46バッチ ⑤(commerce.inventory.b2b)。OFF は None=列なし=L2 不変 ----
@register_aggregator("n_b2b_trade")
def _n_b2b_trade(sim):
    """会社間取引(卸→小売の仕入れ)の累積件数。inventory または b2b が無効なら None=列なし
    (L2 バイト不変)。ON 時のみ b2b_trade の累積を出す。"""
    gc = getattr(sim, "goodscfg", None)
    bc = getattr(sim, "b2bcfg", None)
    if not (gc and gc.get("enabled") and bc and bc.get("enabled")):
        return None
    return int(getattr(sim, "_b2b_total", 0))


# ---- 宅配・フードデリバリー 第47バッチ ④(delivery)。OFF は None=列なし=L2 不変 ----
@register_aggregator("n_delivery")
def _n_delivery(sim):
    """宅配の配送完了(deliver)の累積件数。delivery 無効なら None=列なし(L2 バイト不変)。
    ON 時のみ deliver の累積を出す。"""
    dc = getattr(sim, "deliverycfg", None)
    if not (dc and dc.get("enabled")):
        return None
    return int(getattr(sim, "_delivery_total", 0))


@register_aggregator("n_active_rules")
def _n_active_rules(sim):
    """制度DSL: 現在アクティブな実効ルール数(rules 無効時は None=列なし=L2 不変)。"""
    rb = getattr(sim, "rulebook", None)
    if rb is None or not rb.cfg["enabled"]:
        return None
    return len(rb.active)


@register_aggregator("n_institutions")
def _n_institutions(sim):
    """成立した制度の累積数(Searle プラグイン ON 時のみ。OFF は None=列なし)。"""
    pc = getattr(sim, "psychcfg", None)
    if not pc or not pc["searle"]["enabled"]:
        return None
    t = getattr(sim, "tools", None)
    return len(t.institutions) if t else 0


# ---- 崩壊検知(R12: 均質化ドリフトの非LLM監視。毎stepのL2列。乱数不使用) ----
@register_aggregator("speech_diversity")
def _speech_diversity(sim) -> float:
    """全エージェントの直近発話(said 末尾)連結の文字3-gram type/token 比。

    値が低下 = 皆が同じ言い回しに収束(均質化)。発話が無ければ 0.0。
    """
    if not sim.agents:
        return 0.0
    text = "".join(_last(a.said) for a in sim.agents)
    return round(_type_token_ratio(text), 6)


@register_aggregator("speech_pairwise_var")
def _speech_pairwise_var(sim) -> float:
    """id 順先頭 ≤50 体の発話間3-gram Jaccard 距離の分散。

    分散が 0 に潰れる = 発話の相互差異が消失(均質化)。人数不足なら 0.0。
    """
    agents = sorted(sim.agents, key=lambda a: a.id)[:50]
    sets = [_char3_set(_last(a.said)) for a in agents]
    dists = [_jaccard_distance(sets[i], sets[j])
             for i in range(len(sets)) for j in range(i + 1, len(sets))]
    if not dists:
        return 0.0
    mean = sum(dists) / len(dists)
    var = sum((d - mean) ** 2 for d in dists) / len(dists)
    return round(var, 6)


@register_aggregator("belief_diversity")
def _belief_diversity(sim) -> float:
    """beliefs 末尾の連結の文字3-gram type/token 比(k 起因のペルソナ均質化検知)。

    beliefs が空(内省の書き戻しが起きていない)なら 0.0。
    """
    if not sim.agents:
        return 0.0
    text = "".join(_last(getattr(a, "beliefs", None)) for a in sim.agents)
    return round(_type_token_ratio(text), 6)


# ---- 観測レンズ 第50バッチ(既定 OFF)。lens.enabled=false は全て None=列なし=L2 バイト不変 ----
# T1 価値4軸(value4): 当日のイベント数構成(utility/emotion/social/epistemic)。全体スカラーのみ。
#   observe() は step 末に collect が 1 回呼ぶ経路で当日タリーを積む(idempotent・読むだけ・乱数ゼロ)。
def _value_col(sim, axis):
    if not (_lens.enabled(sim) and _lens.cfg_of(sim)["value4"]["enabled"]):
        return None
    st = _lens.observe(sim)
    return int(st["value"].get(axis, 0)) if st else 0


def _motive_col(sim, motive):
    if not (_lens.enabled(sim) and _lens.cfg_of(sim)["motives"]["enabled"]):
        return None
    st = _lens.observe(sim)
    return int(st["motive"].get(motive, 0)) if st else 0


@register_aggregator("value4_utility")
def _value4_utility(sim):
    return _value_col(sim, "utility")


@register_aggregator("value4_emotion")
def _value4_emotion(sim):
    return _value_col(sim, "emotion")


@register_aggregator("value4_social")
def _value4_social(sim):
    return _value_col(sim, "social")


@register_aggregator("value4_epistemic")
def _value4_epistemic(sim):
    return _value_col(sim, "epistemic")


@register_aggregator("motive_earn")
def _motive_earn(sim):
    return _motive_col(sim, "earn")


@register_aggregator("motive_love")
def _motive_love(sim):
    return _motive_col(sim, "love")


@register_aggregator("motive_recognition")
def _motive_recognition(sim):
    return _motive_col(sim, "recognition")


# T6 信用内訳(trust): 合成地位 status の分布 2 列(Gini・上位10%集中度)。lens+trust+status が全て ON の
#   ときだけ出す。値は既存 status_gini/status_top10_share と同義(status を「信用」レンズとして束ねた列)。
#   status 無効(=status 値が全て 0)や lens/trust 無効なら None=列なし=L2 不変。
def _trust_on(sim) -> bool:
    return bool(_lens.enabled(sim) and _lens.cfg_of(sim)["trust"]["enabled"]
                and _status_mod.enabled(sim))


@register_aggregator("trust_gini")
def _trust_gini(sim):
    if not _trust_on(sim):
        return None
    if not sim.agents:
        return 0.0
    return round(_gini(getattr(a, "status", 0.0) for a in sim.agents), 6)


@register_aggregator("trust_top10")
def _trust_top10(sim):
    if not _trust_on(sim):
        return None
    vals = sorted((float(getattr(a, "status", 0.0)) for a in sim.agents), reverse=True)
    total = sum(vals)
    if not vals or total <= 0.0:
        return 0.0
    k = max(1, int(len(vals) * 0.1))
    return round(sum(vals[:k]) / total, 6)


def collect(sim) -> dict:
    """全 aggregator を回して L2 の1行を作る。**None を返した列は出さない**(既定 OFF の
    プラグイン列を OFF 時に完全に不在化=L2 不変)。既存 aggregator は None を返さないので
    従来出力はバイト一致で不変。"""
    out: dict = {}
    for name, fn in sorted(AGGREGATORS.items()):
        value = fn(sim)
        if value is not None:
            out[name] = value
    return out
