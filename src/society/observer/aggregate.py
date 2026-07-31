"""L2 集計プラグイン(D12)。新指標の追加 = 関数を1個 register するだけ。"""
from __future__ import annotations

from typing import Any, Callable

from .. import gossip as _gossip
from .. import relations_endo as _rendo
from .. import status as _status_mod
from .. import truth_ledger as _truth
from . import assets as _assets
from . import deviation as _dev
from . import echo as _echo
from . import lens as _lens
from . import norms as _norms
from . import silence as _silence
from . import structure as _struct

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


# ---- 火種介入 spark treatment(第53バッチ)。OFF/未選抜は None=列なし=L2 バイト不変 ----
def _spark_groups(sim):
    """(sparked 居住者, 非 sparked 居住者)を返す。spark OFF / 選抜 0 / 片群が空なら None。"""
    ids = getattr(sim, "_spark_ids", None)
    sc = getattr(sim, "sparkcfg", None)
    if not (sc and sc.get("enabled")) or not ids:
        return None
    sp = [a for a in sim.agents if a.id in ids]
    ot = [a for a in sim.agents if a.id not in ids and not a.visitor]
    if not sp or not ot:
        return None
    return sp, ot


@register_aggregator("spark_reach_ratio")
def _spark_reach_ratio(sim):
    """sparked 群 / 非 sparked 群の平均関係数(社会的リーチ=活動量の代理)の比。spark OFF は None=列なし。

    t=0 の関係注入で高く始まり、非 sparked が追いつけば 1 へ寄る(火種の持続性の live 信号)。厳密な
    活動量・波及分析は事後トレーサ scripts/trace_spark.py が L1 から行う。分母は 1.0 で床(0 除算回避)。"""
    g = _spark_groups(sim)
    if g is None:
        return None
    sp, ot = g
    msp = sum(len(getattr(a.mem, "relations", {})) for a in sp) / len(sp)
    mot = sum(len(getattr(a.mem, "relations", {})) for a in ot) / len(ot)
    return round(msp / max(mot, 1.0), 4)


@register_aggregator("spark_money_ratio")
def _spark_money_ratio(sim):
    """sparked 群 / 非 sparked 群の平均手持ち(資本の軌跡)の比。spark OFF は None=列なし=L2 不変。"""
    g = _spark_groups(sim)
    if g is None:
        return None
    sp, ot = g
    msp = sum(float(getattr(a, "money", 0.0)) for a in sp) / len(sp)
    mot = sum(float(getattr(a, "money", 0.0)) for a in ot) / len(ot)
    return round(msp / max(mot, 1.0), 4)


# ---- サービスの実体化 第46バッチ ③(services)。OFF は None=列なし=L2 不変 ----
@register_aggregator("n_service_use")
def _n_service_use(sim):
    """サービス受給(理美容/クリニック/塾/ジム/クリーニング等)の累積件数。services 無効なら
    None=列なし(L2 バイト不変)。ON 時のみ service_use の累積を出す。"""
    sc = getattr(sim, "servicescfg", None)
    if not (sc and sc.get("enabled")):
        return None
    return int(getattr(sim, "_service_total", 0))


# ---- 自助努力 affordance 第52バッチ(services.self_dev)。OFF は None=列なし=L2 不変 ----
def _self_dev_on(sim):
    sc = getattr(sim, "servicescfg", None)
    sd = sc.get("self_dev") if sc else None
    return sd if (sd and sd.get("enabled")) else None


@register_aggregator("n_self_dev_users")
def _n_self_dev_users(sim):
    """自助サービスの累積を1つでも持つ個体数(採用率の分子)。self_dev 無効なら None=列なし
    (L2 バイト不変)。ON 時のみ「自力で伸ばすことを選んだ人」の広がりを出す。"""
    if _self_dev_on(sim) is None:
        return None
    return sum(1 for a in sim.agents
               if any(v > 0.0 for v in (getattr(a, "self_dev", None) or {}).values()))


@register_aggregator("avg_self_dev")
def _avg_self_dev(sim):
    """累積を持つ個体の平均累積値(全軸プール。0=まだ誰も蓄積していない)。self_dev 無効なら
    None=列なし(L2 バイト不変)。ON 時のみ蓄積の平均的な厚みを出す。"""
    if _self_dev_on(sim) is None:
        return None
    vals = [v for a in sim.agents
            for v in (getattr(a, "self_dev", None) or {}).values() if v > 0.0]
    return round(sum(vals) / len(vals), 4) if vals else 0.0


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


# ---- ペルソナ逸脱率 第55バッチ タスクA(既定 OFF)。lens.deviation.enabled=false は全て None=列なし ----
#   裁量逸脱率(disc)を主・全時間版(fulltime)を参考に。住民別は L2 に出さず(条件2)ビューア事後計算へ。
#   scalars() は step 末に collect が 1 回呼ぶ経路で当日タリーから全体スカラーを作る(読むだけ・乱数ゼロ)。
def _dev_col(sim, key):
    if not _dev.enabled(sim):
        return None
    s = _dev.scalars(sim)
    return s.get(key) if s else 0.0


@register_aggregator("deviation_mean")
def _deviation_mean(sim):
    return _dev_col(sim, "deviation_mean")


@register_aggregator("deviation_var")
def _deviation_var(sim):
    return _dev_col(sim, "deviation_var")


@register_aggregator("deviation_top_share")
def _deviation_top_share(sim):
    return _dev_col(sim, "deviation_top_share")


@register_aggregator("deviation_fulltime_mean")
def _deviation_fulltime_mean(sim):
    return _dev_col(sim, "deviation_fulltime_mean")


# ---- 社会構造の内生変動 第56バッチ タスクB(既定 OFF)。lens.structure.enabled=false は全て None=列なし ----
#   edge 組み替え率(当日の新規形成/断絶/風化の件数・率)。前日状態を持たない軽量スカラー(条件2)。
#   順位相関(Kendall τ)・中心性入れ替わり・コミュニティ変化・固着検知は事後層 scripts/analyze_structure.py。
#   scalars() は step 末に collect が 1 回呼ぶ経路で当日タリー+当日母数から全体スカラーを作る(読むだけ・乱数ゼロ)。
def _struct_col(sim, key):
    if not _struct.enabled(sim):
        return None
    s = _struct.scalars(sim)
    return s.get(key) if s else 0


@register_aggregator("edges_formed")
def _edges_formed(sim):
    return _struct_col(sim, "edges_formed")


@register_aggregator("edges_broken")
def _edges_broken(sim):
    return _struct_col(sim, "edges_broken")


@register_aggregator("edges_decayed")
def _edges_decayed(sim):
    return _struct_col(sim, "edges_decayed")


@register_aggregator("edge_churn_rate")
def _edge_churn_rate(sim):
    return _struct_col(sim, "edge_churn_rate")


# ---- 資産分布 第59バッチ スライス(a)(既定 OFF)。lens.assets.enabled=false は全て None=列なし=L2 不変 ----
#   wealth=money+account の分布(Gini/上位10%集中/中央値/平均)を毎 step 現状から出す(status_gini と同型)。
#   格差の固定化=資産順位の前日比 Kendall τ(asset_rank_tau。前日 wealth を sim に 1 本だけ持つ=status の
#   rank_mobility と同流儀。初日/共通id<2 は初期値 1.0=順位不変)。住民別分布・ドリルダウンはビューア事後計算。
#   scalars() は step 末に collect が 1 回呼ぶ経路で現 wealth+当日境界 τ から全体スカラーを作る(読むだけ・乱数ゼロ)。
def _assets_col(sim, key):
    if not _assets.enabled(sim):
        return None
    s = _assets.scalars(sim)
    return s.get(key) if s else 0.0


@register_aggregator("asset_gini")
def _asset_gini(sim):
    return _assets_col(sim, "asset_gini")


@register_aggregator("asset_top10_share")
def _asset_top10_share(sim):
    return _assets_col(sim, "asset_top10_share")


@register_aggregator("asset_median")
def _asset_median(sim):
    return _assets_col(sim, "asset_median")


@register_aggregator("asset_mean")
def _asset_mean(sim):
    return _assets_col(sim, "asset_mean")


@register_aggregator("asset_rank_tau")
def _asset_rank_tau(sim):
    return _assets_col(sim, "asset_rank_tau")


# ---- 負の評判の内生伝播 第61バッチ スライス(c)(既定 OFF)。gossip.enabled=false は全て None=列なし=L2 不変 ----
#   現在悪評を知られている人数・当日伝播採用数・悪評1件あたり平均到達数(住民別・伝播経路は L1 から事後再構成)。
#   scalars() は step 末に collect が 1 回呼ぶ経路で現状の被知覚数+当日境界カウンタから全体スカラーを作る(読むだけ)。
def _gossip_col(sim, key):
    s = _gossip.scalars(sim)
    return s.get(key) if s else None


@register_aggregator("gossip_active_count")
def _gossip_active_count(sim):
    return _gossip_col(sim, "gossip_active_count")


@register_aggregator("gossip_spread_total")
def _gossip_spread_total(sim):
    return _gossip_col(sim, "gossip_spread_total")


@register_aggregator("gossip_reach_mean")
def _gossip_reach_mean(sim):
    return _gossip_col(sim, "gossip_reach_mean")


# ---- 承諾/拒否判断の内生化 第62バッチ フェーズ1(既定 OFF)。relations.endogenous_accept.enabled=false
#      (または joint OFF)は全て None=列なし=L2 不変(gossip/deviation と同型)。
#   当日の承諾率・内生判定率(=1−fallback率)・較正期待との乖離(承諾率−p_calib平均=決定論算出)・
#   履行率(承諾者が実際に同席したか)。タリーは joint.plan_day(誘い時)+ joint.observe(同席時)が積み、
#   scalars() は step 末に collect が 1 回呼ぶ経路で当日タリーから全体スカラーを作る(読むだけ・乱数ゼロ)。
def _endo_col(sim, key):
    s = _rendo.scalars(sim)
    return s.get(key) if s else None


@register_aggregator("joint_accept_rate")
def _joint_accept_rate(sim):
    return _endo_col(sim, "joint_accept_rate")


@register_aggregator("joint_endo_share")
def _joint_endo_share(sim):
    return _endo_col(sim, "joint_endo_share")


@register_aggregator("joint_accept_calib_gap")
def _joint_accept_calib_gap(sim):
    return _endo_col(sim, "joint_accept_calib_gap")


@register_aggregator("joint_fulfill_rate")
def _joint_fulfill_rate(sim):
    return _endo_col(sim, "joint_fulfill_rate")


# ---- 誘う相手の内生選抜 第64バッチ フェーズ3(既定 OFF)。relations.endogenous_invite.enabled=false
#      (または joint OFF)は全て None=列なし=L2 不変(endo accept 4列と同型)。実装のみ=実験投入は
#      phase3_go ゲート(docs/plans/endogenous-relations-plan.md §4)。
#   当日の弱い紐帯経路(source=weak_tie)の誘い率・内生経路(plan_with+dialog_cue)の誘い率。
#   タリーは joint.plan_day(誘い時)が積み、invite_scalars() は当日タリーから読むだけ(乱数ゼロ)。
def _invite_col(sim, key):
    s = _rendo.invite_scalars(sim)
    return s.get(key) if s else None


@register_aggregator("invite_weak_tie_rate")
def _invite_weak_tie_rate(sim):
    return _invite_col(sim, "invite_weak_tie_rate")


@register_aggregator("invite_endo_share")
def _invite_endo_share(sim):
    return _invite_col(sim, "invite_endo_share")


# ---- 関係の質の内生化 第65バッチ フェーズ4(既定 OFF)。relations.endogenous_quality.enabled=false
#      (または relations OFF)は None=列なし=L2 不変(accept/invite 列と同型)。
#   当日の会話由来 magnitude の平均(closeness の増減量に載った不透明係数=会話の厚みの観測)。
#   タリーは engine の交流記録が積み、日境界で初期化される(読むだけ・乱数ゼロ)。
@register_aggregator("quality_magnitude_mean")
def _quality_magnitude_mean(sim):
    s = _rendo.quality_scalars(sim)
    return s.get("quality_magnitude_mean") if s else None


# ---- 屋内エンジン配線 B3(indoor.enabled ON のみ・既定 OFF=None=列なし=L2 バイト不変)----
#   この step のフロア内区画遷移(space_move)/遭遇(encounter)/会議開催の件数。per-step カウンタ
#   (累積でない=resume 安全=分割再開でも各 step の値が state だけから再現される)。
def _indoor_col(sim, attr):
    if getattr(sim, "indoor", None) is None:
        return None
    return int(getattr(sim, attr, 0))


@register_aggregator("n_space_move")
def _n_space_move(sim):
    return _indoor_col(sim, "_indoor_n_space_move")


@register_aggregator("n_indoor_encounter")
def _n_indoor_encounter(sim):
    return _indoor_col(sim, "_indoor_n_encounter")


@register_aggregator("n_indoor_meeting")
def _n_indoor_meeting(sim):
    return _indoor_col(sim, "_indoor_n_meeting")


# ---- 場所の意味づけ最小版 D1 第69バッチ(既定 OFF)。labeling.place_binding.enabled=false は
#      None=列なし=L2 不変(gossip/endo/indoor 列と同型)。読むだけ・乱数ゼロ・LLM 呼ゼロ。
#   place_bound_labels … 発生ノードへ束縛された語の**累計数**(単調非減少。per-step 差分ではない)。
#   累計なので resume 安全(state=束縛台帳だけから各 step の値が再現される)。
@register_aggregator("place_bound_labels")
def _place_bound_labels(sim):
    labels = getattr(sim, "labels", None)
    fn = getattr(labels, "place_bound_count", None) if labels is not None else None
    return fn() if fn is not None else None


# ---- LLM 健全性 KPI(P0バッチ 2026-07-29・既定 OFF)。observer.llm_health.enabled=false は
#      全て None=列なし=L2 不変(gossip/endo/indoor 列と同型)。読むだけ・乱数ゼロ・LLM 呼ゼロ。
#
# 3 列(すべて**ラン開始からの累積**。per-step 差分ではない):
#   llm_calls_total    … CachedLLM/RouterLLM の calls(summary.json の llm_calls と同じ源)
#   llm_cache_hit_rate … hits / calls(mock リプレイや同一プロンプト再発で上がる)
#   llm_fallback_rate  … L1 `fallback` イベント数 / llm_calls_total
#
# ★llm_fallback_rate の正直な限界(実装可能な範囲の代理指標である旨):
#   - 分子は L1 kind="fallback"(payload.reason="parse_error")で、これを出すのは
#     scheduler の発話系 1 箇所だけ = **発話・熟考のパース失敗しか数えない**。
#   - 朝の計画は「決定論修復 → 再試行(purpose="plan_retry")→ 前日流用 → 既定骨格」の
#     失敗の階段を持ち、`fallback` イベントを出さない。夜の内省も空応答時に無言で退行する。
#     したがって本列は**真のパース失敗率の下限**であり、上振れは検知できるが下振れは保証しない。
#   - 分母は「発話系の呼数」ではなく総呼数(plan/reflect/recall/null を含む)なので、
#     purpose 構成が変わると同じ失敗率でも値が動く。**条件間の絶対比較には使わず、
#     同一構成のラン内の時系列監視(急増の検知)に使う**のが正しい用法。
#   - 実 LLM 側の HTTP エラーは backend が "__*_error__" 文字列を返し、パースに失敗して
#     `fallback` になるので分子に入る(mock では発生しない)。
def _llm_health_on(sim) -> bool:
    cfg = getattr(sim, "cfg", None)
    if cfg is None:
        return False
    try:
        obs = cfg.observer
        block = obs.get("llm_health", None) if hasattr(obs, "get") else None
    except Exception:
        return False
    if block is None:
        return False
    enabled = block.get("enabled", False) if hasattr(block, "get") else False
    return bool(enabled)


def _llm_totals(sim):
    """(calls, hits, fallbacks) を返す。llm_health OFF / 取得不能なら None。"""
    if not _llm_health_on(sim):
        return None
    llm = getattr(sim, "llm", None)
    logger = getattr(sim, "logger", None)
    if logger is None:
        return None
    calls = getattr(llm, "calls", None)
    hits = getattr(llm, "hits", None)
    if calls is None:                     # 呼数カウンタを持たない backend(素の LLMBackend)
        calls = getattr(logger, "n_llm_rows", 0)
        hits = getattr(logger, "n_llm_cached", 0)
    return int(calls), int(hits or 0), int(getattr(logger, "n_fallback_events", 0))


@register_aggregator("llm_calls_total")
def _llm_calls_total(sim):
    t = _llm_totals(sim)
    return None if t is None else t[0]


@register_aggregator("llm_cache_hit_rate")
def _llm_cache_hit_rate(sim):
    t = _llm_totals(sim)
    if t is None:
        return None
    calls, hits, _ = t
    return round(hits / calls, 6) if calls else 0.0


@register_aggregator("llm_fallback_rate")
def _llm_fallback_rate(sim):
    t = _llm_totals(sim)
    if t is None:
        return None
    calls, _, fb = t
    return round(fb / calls, 6) if calls else 0.0


# ---- エコー/自己反復の計測 第70バッチ IDEA①(**常設** 5 列)。observer/echo.py が単一の源。
#      読むだけ・乱数ゼロ・LLM 呼ゼロ・L1 イベント追加ゼロ・プロンプト 1 バイト不変。
#      `observer.echo.enabled: false` にした時だけ 5 列とも消える(過去ランとの L2 一致用の逃げ道)。
#      定義と正直な限界は observer/echo.py の docstring を参照(完全一致反復 vs 言い換え反復の
#      切り分け・閾値の恣意性・意味的言い換えは捉えられない旨)。
def _echo_col(sim, key):
    s = _echo.scalars(sim)
    return s.get(key) if s else None


@register_aggregator("echo_max")
def _echo_max_col(sim):
    return _echo_col(sim, "echo_max")


@register_aggregator("self_similarity_mean")
def _self_similarity_mean(sim):
    return _echo_col(sim, "self_similarity_mean")


@register_aggregator("echo_utterance_rate")
def _echo_utterance_rate(sim):
    return _echo_col(sim, "echo_utterance_rate")


@register_aggregator("transmission_novel")
def _transmission_novel(sim):
    return _echo_col(sim, "transmission_novel")


@register_aggregator("transmission_novel_rate")
def _transmission_novel_rate(sim):
    return _echo_col(sim, "transmission_novel_rate")


# ---- 規範化ステージ 第74バッチ IDEA③(既定 OFF = 2 列とも消える = L2 バイト不変)。
#      観測の源は observer/norms.py(語ごとの 4 段階台帳)。読むだけ・乱数ゼロ・LLM 呼ゼロ・
#      L1 イベント追加ゼロ・プロンプト 1 バイト不変。詳細な語別の台帳と命名者/制度化者の
#      突合は **解析側**(scripts/analyze_norms.py)が持ち、L2 は要約 2 列だけに絞る。
def _norm_col(sim, key):
    s = _norms.scalars(sim)
    return s.get(key) if s else None


@register_aggregator("norm_stage_max")
def _norm_stage_max(sim):
    return _norm_col(sim, "norm_stage_max")


@register_aggregator("norm_steps_to_agreement")
def _norm_steps_to_agreement(sim):
    return _norm_col(sim, "norm_steps_to_agreement")


# ---- 未定義行動レジスタ + 沈黙の第一級化 第70バッチ IDEA②(既定 OFF)。
#      freedom.undefined_register=false / freedom.explicit_nothing=false は None=列なし=L2 不変。
#      観測の源は observer/silence.py(累積カウンタ + 暦日タリー)。読むだけ・乱数ゼロ。
def _undef_col(sim, key):
    s = _silence.undefined_scalars(sim)
    return s.get(key) if s else None


@register_aggregator("undefined_action_total")
def _undefined_action_total(sim):
    return _undef_col(sim, "undefined_action_total")


@register_aggregator("undefined_action_rate")
def _undefined_action_rate(sim):
    return _undef_col(sim, "undefined_action_rate")


def _silence_col(sim, key):
    s = _silence.silence_scalars(sim)
    return s.get(key) if s else None


@register_aggregator("silent_agent_rate")
def _silent_agent_rate(sim):
    return _silence_col(sim, "silent_agent_rate")


@register_aggregator("chosen_nothing_rate")
def _chosen_nothing_rate(sim):
    return _silence_col(sim, "chosen_nothing_rate")


# ---- 真偽台帳ミニマル 第73バッチ Part B(既定 OFF)。beliefs.enabled=false は全て None=
#      列なし=L2 不変(gossip/echo/silence 列と同型)。読むだけ・乱数ゼロ・LLM 呼ゼロ。
#      定義と正直な限界は truth_ledger.py の docstring を参照(真値は 1 次元スカラー・
#      変形はホップ数のみ・話題一致は部分文字列)。
def _belief_col(sim, key):
    s = _truth.scalars(sim)
    return s.get(key) if s else None


@register_aggregator("belief_facts_total")
def _belief_facts_total(sim):
    return _belief_col(sim, "belief_facts_total")


@register_aggregator("belief_distance_mean")
def _belief_distance_mean(sim):
    return _belief_col(sim, "belief_distance_mean")


@register_aggregator("belief_verified_rate")
def _belief_verified_rate(sim):
    return _belief_col(sim, "belief_verified_rate")


@register_aggregator("belief_hop_mean")
def _belief_hop_mean(sim):
    return _belief_col(sim, "belief_hop_mean")


@register_aggregator("belief_holders_mean")
def _belief_holders_mean(sim):
    return _belief_col(sim, "belief_holders_mean")


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
