"""情報環境の非対称(現実ギャップ Wave G6 2026-07-07。ユーザー要望)。

世界改変者(keystone)の創発は情報環境に強く条件づけられる。既存の net/internet(SNS 投稿/TL/
いいね/RT/フォロー/ニュース)+ opinion(意見力学 FJ)の最小拡張として、意見分極と伝播の加速器を
3機構で実装する(すべて非LLM・決定論・既定 OFF):

  1. アルゴリズム推薦/エコーチェンバー(recommendation): SNS の TL 提示を閲覧者の意見整合バイアスで
     並べ替え/選別する。自分の意見に近い投稿を優先表示し、遠い投稿を間引く→分極の加速。意見参照は
     opinion.alignment 経由で不透明化(no-fingerprint)。feed_rank(agent, boosted, filtered) を記録。
  2. インフルエンサー非対称/バイラル(influence): フォロワー数の非対称を活かし、影響力の大きい発信者
     (高フォロワー)の投稿のリシェア到達を加重する(既存 reshare カスケードの reach を数として増幅=
     新規投稿・heard 語を作らない=物理不変の観測量)。viral_cascade(post_id, author, reach) を記録。
  3. フェイク/誤情報・炎上(misinfo): 一部投稿を誤情報として扱い(新 stream "info" で決定論判定)、訂正の
     拡散(非LLM 状態更新=拡散カウンタ)・炎上(ネガ反応の殺到)を記録する。misinfo(post_id, kind) を記録。

★ 配置(R9 / no-fingerprint): このモジュールは src/society/net 直下(engine/cognition/world の静的検査
  対象外)なので情報環境ロジックはここに置いてよい。意見参照は opinion.alignment(純関数)経由に集約=
  engine/net には生の意見差でなく [0,1] の整合スコアだけを見せる。炎上→grievance は factors.on_misinfo
  経由(不透明 magnitude・既定 0=観測のみ)で、発火系(drive)には一切接続しない。

R1 呼数不変: どの機構も generate() を1本も足さない。炎上の訂正・反応は既存 post/RT の再利用 or 非LLM の
  状態更新(拡散カウンタ・grievance)で表現する。推薦は「どの投稿を読むか」を変えるため FixedLLM で
  ON!=OFF になりうる(SNS=内容/ネットワーク層の必然)が、機構は k・内面状態(構成概念)を一切読まず
  暦・config・意見・フォロワー・新 stream のみ参照するため、compute_matched 下の k 不変性(k=free と
  k=off で generate 呼数完全一致)で R1 を厳密に担保する(career G5 / crowd G4 と同型)。

既定 OFF(enabled=false)= timeline は従来の time-series TL に後退、viral/misinfo フェーズは即 return。
  feed_rank/viral_cascade/misinfo とも 0 件・プロンプト/イベント列/乱数消費が不変(ゴールデン
  golden_baseline_l1.json を守る)。
"""
from __future__ import annotations

from .. import opinion as _opinion
from .. import status as _status
from ..factors import update as _factor_update
from ..lang.sentiment import valence
from ..observer.schema import Event
from .internet import MEDIA_ID

DEFAULTS = {
    "enabled": False,                 # マスターゲート。OFF=全機構 no-op(バイト一致)
    # ---- 推薦/エコーチェンバー ----
    "recommendation": {
        "enabled": False,             # ★ON 推奨(本番): true(意見整合バイアスで TL 選別)
    },
    # ---- インフルエンサー非対称/バイラル ----
    "influence": {
        "enabled": False,             # ★ON 推奨(本番): true(高フォロワー投稿の reach を加重)
        "follower_threshold": 8,      # フォロワーがこれ以上の author をインフルエンサーとみなす
        "reach_weight": 1.0,          # 追加 reach = weight ×(フォロワー数 − 閾値 + 1)
        "max_reach": 30,              # 1投稿あたり追加 reach の上限(暴走防止)
    },
    # ---- フェイク/誤情報・炎上 ----
    "misinfo": {
        "enabled": False,             # ★ON 推奨(本番): true(誤情報・訂正・炎上のダイナミクス)
        "rate": 0.15,                 # 新規投稿が誤情報として扱われる割合(新 stream "info")
        "correction_prob": 0.5,       # 誤情報が訂正(拡散)される確率
        "flame_grievance": 0.0,       # 炎上を受けた発信者の不満増分(factors 経由。0.0=観測のみ・grievance 不変)
    },
}

_INT_KEYS = {"influence": ("follower_threshold", "max_reach"),
             "misinfo": ()}
_FLOAT_KEYS = {"influence": ("reach_weight",),
               "misinfo": ("rate", "correction_prob", "flame_grievance")}


def build_cfg(raw) -> dict:
    """conf の info_env ブロックを正準化(既定 OFF=現行挙動と完全同一)。

    dotlist 上書きは文字列で入り得るため型強制する(relations.build_cfg と同じ作法)。"""
    raw = dict(raw or {})
    cfg = {
        "enabled": bool(raw.get("enabled", False)),
        "recommendation": dict(DEFAULTS["recommendation"]),
        "influence": dict(DEFAULTS["influence"]),
        "misinfo": dict(DEFAULTS["misinfo"]),
    }
    for block in ("recommendation", "influence", "misinfo"):
        sub = dict(raw.get(block, {}) or {})
        out = cfg[block]
        for k, v in sub.items():
            if k not in out:
                continue
            if k == "enabled":
                out[k] = bool(v)
            elif k in _INT_KEYS.get(block, ()):
                out[k] = int(v)
            elif k in _FLOAT_KEYS.get(block, ()):
                out[k] = float(v)
            else:
                out[k] = v
    return cfg


# ---------------------------------------------------------------- ゲート
def _cfg(sim) -> dict | None:
    return getattr(sim, "infoenvcfg", None)


def recommendation_on(sim) -> bool:
    cfg = _cfg(sim)
    return bool(cfg and cfg["enabled"] and cfg["recommendation"]["enabled"])


def _influence_on(cfg: dict) -> bool:
    return bool(cfg["influence"]["enabled"])


def _misinfo_on(cfg: dict) -> bool:
    return bool(cfg["misinfo"]["enabled"])


# ---------------------------------------------------------------- 推薦(TL 選別)
def timeline(sim, agent, step: int, sim_min: int) -> list[dict]:
    """閲覧者 agent の TL を返す。推薦 ON=意見整合で選別(feed_rank 記録)/ OFF=従来の時系列 TL。

    OFF 時は sim.net.timeline_for(agent.id) をそのまま呼ぶ=watermark 進行・返り値ともバイト一致。
    ヒエラルキー ON(推薦 OFF 時)は露出重み(地位×優先的選択)で選別=高地位・ハブ投稿を厚く見せる。
    """
    if not recommendation_on(sim):
        if _status.enabled(sim):               # 社会的ヒエラルキー: 露出=地位×被フォロー数で選別
            return sim.net.ranked_exposure_timeline_for(
                agent.id, lambda p: _status.feed_exposure_weight(sim, p["author"]))
        return sim.net.timeline_for(agent.id)
    o = float(getattr(agent, "opinion", 0.0))

    def score_of(post: dict) -> float:
        # 投稿の価(valence)と閲覧者の意見の整合度。参照は opinion.alignment 経由(不透明化)。
        return _opinion.alignment(o, valence(post["text"]))

    feed, boosted, filtered = sim.net.ranked_timeline_for(agent.id, score_of)
    if boosted or filtered:                       # 実際に並べ替え/間引きが起きたときだけ記録
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                             kind="feed_rank", x=agent.x, y=agent.y,
                             payload={"agent": agent.id, "boosted": boosted,
                                      "filtered": filtered}))
    return feed


# ---------------------------------------------------------------- バイラル / 炎上(日次でなく step 走査)
def _viral(sim, rt_post: dict, step: int, sim_min: int, icfg: dict) -> None:
    """リシェア(RT)投稿 rt_post の元投稿の author が高フォロワーなら reach を加重する(非LLM)。

    元投稿1件につき1回だけ加重(sim._viral_done で重複防止)。加重 = reshares カウンタの増分
    (=拡散の広がりを数として記録)+ viral_cascade イベント。新規 post を作らない=物理不変。"""
    net = sim.net
    orig_id = rt_post.get("rt_of")
    if orig_id is None:
        return
    idx = int(orig_id) - net._post_offset
    if idx < 0 or idx >= len(net.posts):
        return
    orig = net.posts[idx]
    author = orig["author"]
    if author == MEDIA_ID:                        # メディア(公式)はインフルエンサー非対称の対象外
        return
    done = sim._viral_done
    if orig_id in done:
        return
    followers = net.follower_count(author)
    threshold = int(icfg["follower_threshold"])
    if followers < threshold:
        return
    done.add(orig_id)
    extra = int(round(float(icfg["reach_weight"]) * (followers - threshold + 1)))
    extra = max(1, min(int(icfg["max_reach"]), extra))
    net.amplify_reshare(orig_id, extra)
    reach = followers + extra
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=author, x=0.0, y=0.0,
                         kind="viral_cascade",
                         payload={"post_id": int(orig_id), "author": int(author),
                                  "reach": int(reach)}))


def _misinfo(sim, post: dict, step: int, sim_min: int, mcfg: dict) -> None:
    """新規投稿 post を新 stream "info" で決定論的に誤情報判定し、訂正・炎上を記録する(非LLM)。

    誤情報と判定されたら misinfo(post) を記録。correction_prob で訂正(拡散カウンタ+1=非LLM 状態
    更新)を記録。炎上(flame)=ネガ反応の殺到を記録し、任意で発信者に grievance(factors 経由・
    flame_grievance>0 のときのみ・drive 非接続)。新規 generate/post を1件も作らない。"""
    pid = int(post["id"])
    rng = sim.hub.stream("info", pid, step)       # 新 stream "info"(既存 draw 順に挿入しない)
    if rng.random() >= float(mcfg["rate"]):
        return
    author = post["author"]
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=author, x=0.0, y=0.0,
                         kind="misinfo", payload={"post_id": pid, "kind": "post"}))
    if rng.random() < float(mcfg["correction_prob"]):
        sim.net.amplify_reshare(pid, 1)           # 訂正の拡散(非LLM 状態更新=観測用)
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=author, x=0.0, y=0.0,
                             kind="misinfo",
                             payload={"post_id": pid, "kind": "correction"}))
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=author, x=0.0, y=0.0,
                         kind="misinfo", payload={"post_id": pid, "kind": "flame"}))
    mag = float(mcfg["flame_grievance"])
    if mag > 0.0 and author != MEDIA_ID:          # 炎上→発信者の不満(任意・factors 経由・R9)
        who = sim.agent_by_id.get(author)
        if who is not None:
            _factor_update.on_misinfo(who, mag, step=step, sim_min=sim_min,
                                      logger=sim.logger)


def phase(sim, step: int, sim_min: int) -> None:
    """step 末に新規投稿を走査し、バイラル加重(influence)と誤情報/炎上(misinfo)を処理する。

    絶対 id watermark で「前回走査以降に増えた投稿」だけを見る(決定論・O(新規投稿数))。
    info_env OFF / influence・misinfo とも OFF なら即 return(乱数を引かず・イベント 0 件=バイト一致)。
    """
    cfg = _cfg(sim)
    if not (cfg and cfg["enabled"]):
        return
    infl = _influence_on(cfg)
    mis = _misinfo_on(cfg)
    if not (infl or mis):
        return
    net = sim.net
    end = net._post_offset + len(net.posts)
    scanned = getattr(sim, "_infoenv_watermark", 0)
    if scanned < net._post_offset:
        scanned = net._post_offset                # トリミング済みの古い投稿は走査対象外
    for pid in range(scanned, end):
        idx = pid - net._post_offset
        if idx < 0 or idx >= len(net.posts):
            continue
        post = net.posts[idx]
        if infl and post.get("rt_of") is not None:            # RT はバイラル加重の入口
            _viral(sim, post, step, sim_min, cfg["influence"])
        elif mis and post.get("rt_of") is None and post["author"] != MEDIA_ID:
            _misinfo(sim, post, step, sim_min, cfg["misinfo"])  # 元投稿(非RT・非メディア)
    sim._infoenv_watermark = end
