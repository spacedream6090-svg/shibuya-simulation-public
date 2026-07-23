"""ペルソナ逸脱率レンズ(第55バッチ タスクA 2026-07-23。既定 OFF)。

エージェントが注入されたペルソナ(職業・趣味)にどれだけ従順か/逸脱するかを測る観測レンズ。
逸脱がゼロに近いなら、それ自体が本シミュレーションの限界を語るデータになる(world-changer 検出
= k* 測定の感度に関わる補助指標)。**観測のみ**=逸脱を促進・抑制する介入は一切しない。

  設計正典: docs/closed-world-daily-observation.md タスクA / devlog Entry 47(Fable 精査で確定した
  設計条件)。lens.py(第50バッチ)の作法(build_cfg 正準化・cfg_of 遅延キャッシュ・observe 当日
  タリー・resolved_maps/write_sidecar サイドカー・DEFAULT を単一の源)をそのまま踏襲する。

--- Fable 精査で確定した設計条件(変更不可)-------------------------------------------------
(1) 逸脱は **自由裁量時間に限定して測る** のを主指標にする。義務ルーチン(出勤・通学=勤務行動)は
    ペルソナから機械的に導出されるため、全時間で測ると従順度が構造的に水増しされる。裁量空間の
    行動だけを母数にした版(deviation_mean/var/top_share)が主・全時間版(deviation_fulltime_mean)を
    参考として 1 列だけ併記する(2本立て)。
    ★裁量判定の近似方法 = **行動種別(activity kind)による分類**を採る(方法b)。あるイベントが
      義務(勤務)行動か裁量行動かを、その kind の分類(behavior_map で "@work" に落ちる kind =
      勤務)だけで決める。「勤務帯除外(方法a)」を採らない理由: observe は L1 イベント列を逐次走査
      する経路に載る(lens と同型)ため、各イベント時点の個体別勤務窓(pool ローテで日々入替わる)を
      正確に持てない。kind ベースなら sim なしのビューアが L1 だけから同じ判定を再現でき(sim⇄viz
      疎結合)、resume/pool でも安定する。正直な近似の限界: 勤務時間外に居ても勤務系イベント(残業の
      wage 等)は義務側に数える=時刻ではなく行動の性質で切る。
(2) **住民ごとのスコアは L2 に入れない**。L2 は全体スカラー列のみ(平均・分散・上位逸脱者の割合)。
    住民別スコア・ドリルダウンはビューア build_data の事後計算(第50バッチ lens の流儀)。
(3) **構造化フィールドだけで測る**。主に使えた軸: 職業(agents.json/agent.occupation=常在の構造化
    フィールド)→ **期待余暇 POI カテゴリ hobby_map**(inner_life の _HOBBY_BY_OCC→_HOBBY_CAT の合成=
    職業から決定論導出される構造化趣味)を裁量行動と突き合わせる。実際の行動側は、**カテゴリを payload に
    明示的に持つイベント**(spend.cat / service_use.service / media_use / event_attend / joint_activity)を
    behavior_map で POI 的カテゴリへ写す。
    使えた軸/使えなかった軸(正直な報告):
      ・使えた: 職業↔余暇POIカテゴリ(hobby_map=構造化趣味)。これが裁量逸脱率の主軸。
      ・部分的: 職業↔勤務POIカテゴリ(occupation_map=persona._WORK_CAT の知識)。勤務系イベント
        (wage/serve/org_output)が POI カテゴリを payload に持たないため厳密一致判定は不能=従順度判定
        には使わず(義務行動は常に従順=条件1)、ドリルダウンの「期待職場カテゴリ」注記にのみ使用。
      ・使えなかった: ①自由文の性格記述=構造比較不能=対象外。②性格タグ(traits: nfc/internal_locus/
        risk_tolerance)は数値傾性であって「内向/外向」等の離散社交ラベルとして構造化されておらず、
        社交頻度の期待閾値も構造化されていないため採らない。③ontology 群(文化圏×経験)は LLM 注入用の
        自己申告的「経験の事実」であって構造化された行動期待ではないため軸に採らない。④生の stay/arrive の
        node→POIカテゴリは 1 ノードに複数カテゴリの POI が同居し得て曖昧なので採らず、カテゴリを明示的に
        持つ消費/サービス/余暇イベントで「滞在 POI カテゴリの選択」を近似する。

--- R1 ドクトリン ---------------------------------------------------------------------------
既定 OFF(lens.deviation.enabled=false): observe() が即 None・L2 に列を足さず・サイドカーも書かない=
  L1/L2/乱数・ゴールデンがバイト一致で不変。ON でも「読むだけ」= 行動・乱数・LLM 呼を一切増やさない。
  ロジックは observer 層(本 module)+ conf(lens.deviation ブロック)に閉じ、engine/cognition/world/
  factors/actions/labeling にはペルソナ語・軸語を一切書かない(no-fingerprint)。決定論・RNG stream 不要。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .lens import resolve_axis   # 純関数の 2 段マッチ解決を再利用(kind+payload → 値)

# 勤務(義務)行動を表す特別カテゴリ。behavior_map でこの値に落ちた kind は裁量母数から除外する。
WORK_CATEGORY = "@work"

# --------------------------------------------------------------------------- #
# DEFAULT(単一の源=コード側。conf lens.deviation.* で上書き)
# --------------------------------------------------------------------------- #
# 職業 → 期待勤務 POI カテゴリ。persona._WORK_CAT(職場の割当規則)の知識を観測側へ写した鏡。
#   決まった職場を持たない職業(フリーランス/バンドマン/配達員/写真家/無職)は勤務カテゴリ期待を持たない
#   =勤務系イベントを出しても「期待どおりの職場に居る」とは判定しない(=義務従順に数えない)。
DEFAULT_OCCUPATION_MAP: dict[str, str] = {
    "会社員": "office", "エンジニア": "office", "デザイナー": "office",
    "カフェ店員": "food", "アパレル店員": "shop", "美容師": "shop",
    "大学生": "education",
}

# 職業 → 期待余暇 POI カテゴリ列(裁量時間に「その人らしい」行き先)。inner_life の
#   _HOBBY_BY_OCC(職業→趣味)→ _HOBBY_CAT(趣味→POIカテゴリ)を合成した構造化趣味の鏡。
#   全 12 職業を網羅する=この map が「逸脱を測れる住民」の母集団ゲート(未収載の職業は測定対象外)。
DEFAULT_HOBBY_MAP: dict[str, list[str]] = {
    "バンドマン":   ["nightlife"],   # 音楽・ライブ
    "大学生":       ["nightlife"],   # 音楽・ライブ
    "デザイナー":   ["leisure"],     # アート・写真
    "写真家":       ["leisure"],     # アート・写真
    "フリーランス": ["leisure"],     # アート・写真
    "アパレル店員": ["shop"],        # ファッション・買い物
    "美容師":       ["shop"],        # ファッション・買い物
    "カフェ店員":   ["food"],        # カフェ・食べ歩き
    "会社員":       ["food"],        # カフェ・食べ歩き
    "エンジニア":   ["food"],        # カフェ・食べ歩き
    "配達員":       ["leisure"],     # 散歩・公園めぐり
    "無職":         ["leisure"],     # 散歩・公園めぐり
}

# 実際の行動 → POI 的カテゴリ(str=直接 / dict=2段マッチ=lens.resolve_axis と同型)。
#   カテゴリ体系は hobby_map の値域({food,nightlife,shop,leisure} + education)に閉じる。勤務系は
#   "@work"(裁量母数から除外・従順度は occupation_map の有無で判定)。カテゴリを payload に持たない
#   純機構的な種(route/move/arrive/stay/sleep 等)は意図的に未対応(=どのカテゴリにも数えない)。
DEFAULT_BEHAVIOR_MAP: dict[str, Any] = {
    # ---- 裁量の消費・サービス・余暇(payload にカテゴリを明示的に持つ)----
    "spend": {"__payload__": "cat", "food": "food", "cafe": "food",
              "nightlife": "nightlife", "shop": "shop", "leisure": "leisure",
              "_else": None},                       # taxi/bus 等の交通は場所選択でない=除外
    "service_use": {"__payload__": "service", "grooming": "shop",
                    "lesson": "education", "gym": "leisure", "_else": None},
    "media_use": "leisure",
    "event_attend": "leisure",
    "joint_activity": "leisure",
    # ---- 義務(勤務)行動: ペルソナから機械導出される=裁量母数から除外(全時間版のみ数える)----
    "wage": WORK_CATEGORY, "serve": WORK_CATEGORY, "org_output": WORK_CATEGORY,
    "study": WORK_CATEGORY, "production": WORK_CATEGORY,
}

# 「上位逸脱者」= 裁量逸脱率がこの閾値以上の測定対象住民の割合(deviation_top_share)。
_DEFAULT_TOP_THRESHOLD = 0.5


# --------------------------------------------------------------------------- #
# cfg 正準化(lens.build_lens_cfg と同型: dict/OmegaConf 両対応・DEFAULT へ上書きを重ねる)
# --------------------------------------------------------------------------- #
def _to_plain(raw):
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(raw):
            return OmegaConf.to_container(raw, resolve=True)
    except Exception:
        pass
    return raw


def _merge_scalar_map(default: dict, override) -> dict:
    """str/2段dict を値に持つ map(occupation_map / behavior_map)へ conf 上書きを重ねる。

    上書き値が空/None のキーは削除(lens._merge_map と同流儀)。キーは全て str に統一(conf 作法)。"""
    out = dict(default)
    override = _to_plain(override) or {}
    if isinstance(override, dict):
        for k, v in override.items():
            if v in (None, "", {}):
                out.pop(str(k), None)
            else:
                out[str(k)] = _to_plain(v)
    return out


def _merge_list_map(default: dict, override) -> dict:
    """list を値に持つ map(hobby_map)へ conf 上書きを重ねる。空/None のキーは削除。"""
    out = {k: list(v) for k, v in default.items()}
    override = _to_plain(override) or {}
    if isinstance(override, dict):
        for k, v in override.items():
            v = _to_plain(v)
            if v in (None, "", [], {}):
                out.pop(str(k), None)
            elif isinstance(v, (list, tuple)):
                out[str(k)] = [str(x) for x in v]
            else:
                out[str(k)] = [str(v)]               # スカラー1件もリスト化
    return out


def build_cfg(raw) -> dict:
    """conf の lens.deviation ブロックを正準化(既定 OFF=現行挙動と完全同一)。

    enabled= 独立マスター(value4/motives/trust とは別の observable。lens.enabled には依存しない=
    逸脱レンズだけ ON にできる)。map 群は本 module の DEFAULT へ conf 上書きを重ねる(単一の源=コード側)。"""
    raw = _to_plain(raw) or {}
    raw = dict(raw)
    return {
        "enabled": bool(raw.get("enabled", False)),
        "top_threshold": float(raw.get("top_threshold", _DEFAULT_TOP_THRESHOLD)),
        "occupation_map": _merge_scalar_map(DEFAULT_OCCUPATION_MAP, raw.get("occupation_map")),
        "hobby_map": _merge_list_map(DEFAULT_HOBBY_MAP, raw.get("hobby_map")),
        "behavior_map": _merge_scalar_map(DEFAULT_BEHAVIOR_MAP, raw.get("behavior_map")),
    }


def cfg_of(sim) -> dict:
    """deviation 設定を返す(初回のみ sim.cfg.lens.deviation から遅延構築してキャッシュ)。

    simulation.py は読み取り専用のため cfg は本 module が遅延構築する(lens.cfg_of と同型)。
    キャッシュ属性 sim.devcfg は L1/L2/L3/乱数に一切現れない=既定 OFF のバイト一致を壊さない。"""
    c = getattr(sim, "devcfg", None)
    if c is None:
        raw = None
        try:
            raw = (sim.cfg.get("lens", None) or {}).get("deviation", None)
        except Exception:
            raw = None
        c = build_cfg(raw)
        sim.devcfg = c
    return c


def enabled(sim) -> bool:
    return bool(cfg_of(sim)["enabled"])


# --------------------------------------------------------------------------- #
# 逸脱判定の純関数(決定論・sim 非依存=ビューアが L1 だけから同じ結果を再現できる)
# --------------------------------------------------------------------------- #
def measurable(occupation, hobby_map: dict) -> bool:
    """その職業の逸脱を測れるか(=期待余暇カテゴリが定義されている)。未収載職業は測定対象外。"""
    return str(occupation) in hobby_map


def classify(kind: str, payload, occupation, cfg: dict):
    """1 イベントを (category, is_work, conforming) に分類する。対象外なら None。

    category = behavior_map による POI 的カテゴリ(勤務は WORK_CATEGORY)。None=カテゴリ非対応=数えない。
    is_work  = 勤務(義務)行動か(=裁量母数から除外)。
    conforming= ペルソナ期待に一致するか。
      ・勤務(@work): **常に従順**(条件1)。義務ルーチン=ペルソナから機械的に導出される行動なので定義上
        ペルソナに従順=全時間版はこの分だけ従順が水増しされる(だから主指標は裁量版でこの母数を外す)。
        ★勤務系イベント(wage/serve/org_output 等)は POI カテゴリを payload に持たないため occupation_map
          (職業→勤務POIカテゴリ)との厳密な一致判定はできない=occupation_map は使えない軸(ドリルダウンの
          期待職場カテゴリ注記にのみ利用)。正直な限界: 「勤務行動をしたか」だけで従順とみなす粗い近似。
      ・裁量: category が hobby_map[occupation](構造化趣味=期待余暇カテゴリ)に含まれれば一致。"""
    cat = resolve_axis(kind, payload, cfg["behavior_map"])
    if cat is None:
        return None
    if cat == WORK_CATEGORY:
        return (cat, True, True)
    return (cat, False, cat in cfg["hobby_map"].get(str(occupation), ()))


def _fresh_state() -> dict:
    return {"day": -1, "occ": {}, "per": {}}


def _mean_var_top(ratios: list, threshold: float) -> tuple[float, float, float]:
    """裁量逸脱率のリスト → (平均, 母分散, 上位逸脱者割合)。空なら (0,0,0)。"""
    n = len(ratios)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = sum(ratios) / n
    var = sum((r - mean) ** 2 for r in ratios) / n
    top = sum(1 for r in ratios if r >= threshold) / n
    return mean, var, top


# --------------------------------------------------------------------------- #
# observe(L2 用): 当日(暦日)の住民別逸脱を積み、全体スカラー材料を返す(OFF は None)
# --------------------------------------------------------------------------- #
def observe(sim) -> dict | None:
    """当日の住民別「裁量/全時間 逸脱」を集計して state を返す(OFF は None=列なし=L2 不変)。

    lens.observe と同型: collect(sim) が step 末に 1 回だけ呼ぶ経路に載り、前回処理済み総数
    sim._dev_processed 以降の新規イベントだけを走査(idempotent・暦日境界でタリー+職業map をリセット)。
    per[aid] = {disc_dev, disc_total, full_dev, full_total}(裁量=義務除外・全時間=義務込み)。
    決定論・乱数/LLM 呼ゼロ・読むだけ。"""
    cfg = cfg_of(sim)
    if not cfg["enabled"]:
        return None
    logger = getattr(sim, "logger", None)
    if logger is None:
        return None
    events = logger.events
    total = int(getattr(logger, "_n_flushed", 0)) + len(events)   # ランを通じ単調増加
    processed = int(getattr(sim, "_dev_processed", 0))
    new = total - processed
    if new < 0:
        new = 0
    if new > len(events):          # flush 前に処理済みのはず。安全側=buffer 内だけ数える
        new = len(events)
    st = getattr(sim, "_dev_state", None)
    if st is None:
        st = _fresh_state()
        sim._dev_state = st
    hmap = cfg["hobby_map"]
    for e in events[len(events) - new:]:
        d = int(e.sim_min) // 1440
        if d != st["day"]:         # 暦日が進んだ → 当日タリー + 職業map をリセット
            st["day"] = d
            st["per"] = {}
            # 職業 map をその日の在場ペルソナから作り直す(pool ローテ対応・1 日 1 回=軽い)
            st["occ"] = {a.id: getattr(a, "occupation", "")
                         for a in getattr(sim, "agents", [])}
        aid = e.agent_id
        if aid < 0:                                  # 世界イベント(agent_id=-1)は対象外
            continue
        occ = st["occ"].get(aid)
        if occ is None or not measurable(occ, hmap):  # 測定対象外の職業/未知個体は数えない
            continue
        res = classify(e.kind, e.payload, occ, cfg)
        if res is None:
            continue
        _cat, is_work, conforming = res
        rec = st["per"].get(aid)
        if rec is None:
            rec = {"disc_dev": 0, "disc_total": 0, "full_dev": 0, "full_total": 0}
            st["per"][aid] = rec
        miss = 0 if conforming else 1
        rec["full_total"] += 1
        rec["full_dev"] += miss
        if not is_work:
            rec["disc_total"] += 1
            rec["disc_dev"] += miss
    sim._dev_processed = total
    return st


def scalars(sim) -> dict | None:
    """observe の state → L2 全体スカラー 4 列(OFF は None)。住民別は L2 に出さない(条件2)。"""
    cfg = cfg_of(sim)
    if not cfg["enabled"]:
        return None
    st = observe(sim)
    per = (st or {}).get("per", {})
    disc = [rec["disc_dev"] / rec["disc_total"]
            for rec in per.values() if rec["disc_total"] > 0]
    full = [rec["full_dev"] / rec["full_total"]
            for rec in per.values() if rec["full_total"] > 0]
    mean, var, top = _mean_var_top(disc, cfg["top_threshold"])
    fmean = (sum(full) / len(full)) if full else 0.0
    return {
        "deviation_mean": round(mean, 6),           # 裁量逸脱率の平均(主指標)
        "deviation_var": round(var, 6),             # 裁量逸脱率の母分散
        "deviation_top_share": round(top, 6),       # 裁量逸脱率 >= 閾値 の住民割合
        "deviation_fulltime_mean": round(fmean, 6),  # 全時間版(参考・義務込み=水増し側)
    }


# --------------------------------------------------------------------------- #
# サイドカー(sim⇄viz 疎結合): 解決済み map をラン dir へ書き出す。ビューアが読む(lens と同型)。
# --------------------------------------------------------------------------- #
def resolved_maps(cfg: dict) -> dict:
    return {
        "occupation_map": cfg["occupation_map"],
        "hobby_map": cfg["hobby_map"],
        "behavior_map": cfg["behavior_map"],
        "work_category": WORK_CATEGORY,
        "top_threshold": cfg["top_threshold"],
    }


def write_sidecar(sim, out_dir) -> None:
    """deviation ON のときだけ runs/<name>/deviation_map.json を書く(OFF は何もしない=後方互換)。

    engine 側(simulation.py)は本関数を 1 行呼ぶだけ=ペルソナ語・軸語は本 module に閉じる(no-fingerprint)。"""
    cfg = cfg_of(sim)
    if not cfg["enabled"]:
        return
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "deviation_map.json").write_text(
        json.dumps(resolved_maps(cfg), ensure_ascii=False), encoding="utf-8")
