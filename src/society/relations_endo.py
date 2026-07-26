"""承諾/拒否判断の内生化 第62バッチ フェーズ1(2026-07-27。既定 OFF)。

正典: docs/plans/endogenous-relations-plan.md §1-2。共同行動(joint)の誘い→承諾は従来
「較正確率の抽選」(joint.accept_prob=accept_base+tier_bonus−hierarchy_penalty)のみで、
エージェント自身の状態・意向が承諾判断に一切効かなかった。本 module は較正値を事前分布に
残したまま、**被誘者の構造化状態からの決定論読み取り**で最終判断を内生化する二段構え
(LLMs as Calibrated Measurement Instruments, arXiv:2602.01022 と同型)。

技術的前提(計画書§1 訂正2=最重要): 承諾判定(_phase_joint)は日境界=真夜中に走り、当日の
planning(朝)・発話より**前**。被誘者の当日 LLM 出力は承諾時点で未生成なので、判定材料は
**前日までに生成済みの構造化出力のみ**(同stepのLLM出力への依存は物理的に不成立):
  1. 予定帳簿の当日予定(schedule ON時。会話から決定論抽出された当人の言質):
     誘い主との当日予定=受諾材料 / 他予定と時間帯衝突=拒否(conflict_veto=確定拒否)。
  2. 前日 day_schedule(朝生成=真夜中時点では前日分)の志向:
     with に誘い主=受諾材料 / 単独志向(with なし×solo_whats)が過半=拒否材料。
  3. 前日発話の明示キュー(prompts.dialog_history ON時・agent._dialog_hist):
     「(誘い主名)と…行きたい/行こう」型の明示的受諾方向パターンのみ。
  4. 判定不能 → 較正確率フォールバック(fallback率自体を品質指標として L2 記録)。

自由文抽出を「明示キューのみ・受諾方向のみ」に限定する文献根拠(計画書§1): 日本語の断りは
間接的意味公式(詫び・言い訳・願望=「行きたいのは山々ですが…」)が支配的で、直接的な断り
表現が少ない(Beebe, Takahashi & Uliss-Weltz 1990 の意味公式分類が代表枠組み。邦語の研究動向
レビュー: 宗甜甜 2018「日本語の『断り』に関する研究の動向」日本大学大学院総合社会情報研究科
紀要 19, 207-218; 馮晶・徐千恵 2015「日本語の婉曲表現における断りについての考察」或問 28,
85-94)。婉曲拒否を辞書法で負判定すると誤検出が多いため、**拒否方向の自由文抽出は実装しない**
(拒否は構造化材料=予定衝突・単独志向計画のみ)。曖昧例はフォールバックへ落とす。

合成式(計画書§1 訂正5): p = clamp(w·p_calib + (1−w)·p_endo) − gossip_penalty
  (w=prior_weight。gossip 減算は常に最後=第61バッチの不変則。減算は joint.py 側)。
  p_endo は p_calib 起点の単調・有界写像: 受諾材料→+positive_boost / 拒否材料→−negative_cut /
  fallback→p_calib(現在唯一のモード "calibrated")。conflict_veto は確率でなく確定拒否。

★ 配置: src/society 直下(mobility/gossip と同層=engine/cognition/actions/labeling/world の
  CHECKED_DIRS 外)。判定語彙(solo_whats/positive_cues)は本 module の DEFAULTS+conf 上書き
  (基盤パターン=コード・固有語彙=conf の schedule.py 層分割を踏襲)。

R1(既定 OFF=バイト一致): enabled=false のとき joint.py は本 module の判定・記録経路に一切
  入らず、joint_invite 0 件・sim._endo_state も生えない=L1/L2/乱数消費とも従来と完全同一。
  ON でも LLM 呼ゼロ(既生成の構造化出力の読み取りのみ)・乱数ゼロ(判定は全決定論)。
  抽選 draw は joint.py 側で **always-draw, conditionally-use**(draw は必ず実行し、veto 時は
  結果を破棄)= "joint" stream の decision 単位の消費数・順序が ON/OFF で不変(CRN ペア比較の
  共分散維持・監査容易・resume 無風)。resume==straight: 当日タリー(sim._endo_state)は
  checkpoint.py が中央管理(第59-61 前例)、判定材料は agent 属性=agents pickle に自然同梱。
"""
from __future__ import annotations

import re

from . import schedule as _schedule

DEFAULTS = {
    "enabled": False,
    # 較正確率の事前分布重み w。p = clamp(w·p_calib + (1−w)·p_endo) − gossip_penalty。
    # 1.0=完全較正(従来と同確率)/ 0.0=完全内生。計画書§3: 承諾率乖離が±15ppを超えたら
    # プロンプトでなく本キーで調整(呼数不変のまま)。
    "prior_weight": 0.5,
    "conflict_veto": True,     # 当日予定と時間帯衝突なら確率でなく確定拒否(draw は破棄)
    "positive_boost": 0.35,    # 受諾材料の p_endo 加算(p_calib 起点・上限1)
    "negative_cut": 0.35,      # 拒否材料の p_endo 減算(p_calib 起点・下限0)
    "fallback": "calibrated",  # 判定不能時の後退先(現在唯一のモード=較正確率をそのまま使う)
    # 前日計画の「単独志向」what 語彙(with 空 かつ これに該当する項目が過半=拒否材料)。
    # 既定の根拠: walk=ウォーキングは典型同伴「一人」(レジャー白書2024=docs/research/
    # relationships-activities.md §2.6 表)、home/personal/study は plan_schema の語彙定義上
    # 単独行動(帰宅・用足し・自習)。meal/shop/leisure/visit は同伴があり得るため含めない。
    "solo_whats": ["walk", "home", "personal", "study"],
    # 前日発話の明示キュー動詞(受諾方向のみ)。「(誘い主名)と…<cue>」で受諾材料。
    # 拒否方向のキューは意図的に置かない(婉曲拒否の誤検出回避=冒頭 docstring の文献根拠)。
    "positive_cues": ["行きたい", "行こう", "したい", "しよう",
                      "遊びたい", "遊ぼう", "会いたい", "楽しみ"],
    # 曖昧化マーカー: 同一文にこれが載る発話はキューに数えない=フォールバックへ(計画書§1
    # 「曖昧例はフォールバックへ」)。日本語の婉曲断りは「願望+逆接/言い訳」の意味公式
    # (「行きたいのは山々ですが…」= wish + excuse。Beebe et al. 1990 の間接的意味公式)で、
    # 願望動詞だけ拾うと受諾に誤検出するため、逆接・困難表現の共起で棄却する。
    "hedge_markers": ["山々", "ですが", "だけど", "けれど", "けど", "のに",
                      "無理", "難しい", "また今度", "行けたら"],
}

_FLOAT_KEYS = ("prior_weight", "positive_boost", "negative_cut")


# --------------------------------------------------------------------------- #
# cfg 正準化(gossip.build_cfg と同型: dict/OmegaConf 両対応・型強制)
# --------------------------------------------------------------------------- #
def _to_plain(raw):
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(raw):
            return OmegaConf.to_container(raw, resolve=True)
    except Exception:
        pass
    return raw


def build_cfg(raw) -> dict:
    """conf の relations.endogenous_accept ブロックを正準化(既定 OFF=現行挙動と完全同一)。"""
    raw = dict(_to_plain(raw) or {})
    cfg = dict(DEFAULTS)
    for lk in ("solo_whats", "positive_cues", "hedge_markers"):
        cfg[lk] = list(DEFAULTS[lk])
    for k, v in raw.items():
        if k == "enabled":
            cfg["enabled"] = bool(v)
        elif k == "conflict_veto":
            cfg["conflict_veto"] = bool(v)
        elif k in _FLOAT_KEYS:
            cfg[k] = float(v)
        elif k == "fallback":
            cfg["fallback"] = str(v)
        elif k in ("solo_whats", "positive_cues", "hedge_markers"):
            cfg[k] = [str(x) for x in (_to_plain(v) or [])]
    return cfg


def cfg_of(sim) -> dict:
    """endogenous_accept 設定を返す(初回のみ sim.cfg.relations.endogenous_accept から遅延構築
    してキャッシュ)。キャッシュ属性 sim.endocfg は L1/L2/L3/乱数に一切現れない=既定 OFF の
    バイト一致を壊さない(gossip.cfg_of と同型)。"""
    c = getattr(sim, "endocfg", None)
    if c is None:
        raw = None
        try:
            raw = (sim.cfg.get("relations", None) or {}).get("endogenous_accept", None)
        except Exception:
            raw = None
        c = build_cfg(raw)
        sim.endocfg = c
    return c


def enabled(sim) -> bool:
    return bool(cfg_of(sim)["enabled"])


# --------------------------------------------------------------------------- #
# 判定の純関数群(全決定論・乱数/LLM 呼ゼロ)
# --------------------------------------------------------------------------- #
def _when_minute(when: str):
    """予定帳簿の when("HH:MM" or 時間帯語)→ 代表分 of day。時間情報なしは None。

    schedule.py の既存語彙(_BAND_MIN/_clock_minutes)を再利用=二重実装しない。
    when が空の予定は「時間帯の重複」を正直に主張できない=衝突判定の対象外にする。"""
    if not when:
        return None
    if _schedule._is_clock(when):
        return _schedule._clock_minutes(when)
    if when in _schedule._BAND_MIN and when != "":
        return _schedule._BAND_MIN[when]
    return None


def _name_match(w, full_name: str) -> bool:
    """with 欄の名前(略称あり得る)が誘い主のフルネームに一致/包含されるか。
    joint._resolve_with の照合方向(nm in rn)と同じ向き。"""
    w = str(w).strip()
    return bool(w) and (w == full_name or w in full_name)


def _dialog_history_on(sim) -> bool:
    """prompts.dialog_history が有効か(scheduler._dialog_on の鏡=engine を import しない)。"""
    try:
        return bool((sim.cfg.get("prompts", {}) or {}).get("dialog_history", False))
    except Exception:
        return False


def _cue_regex(name: str, cues: list) -> re.Pattern:
    """「(誘い主名)(さん等)?と…<受諾キュー>」の明示パターン(名前参照必須=明示キューのみ)。

    名前は姓名連結(空白なし。例: 松本悠人)なので、会話の「松本さんと」も拾えるよう
    **2文字以上のプレフィックス**を名前参照として受ける(長い一致優先)。正直な限界:
    同姓の別人への言及と区別できない(明示キューの範囲内の近似=判定は受諾方向のみ)。"""
    alts = ([name[:i] for i in range(len(name), 1, -1)]
            if len(name) >= 2 else [name])
    name_alt = "|".join(re.escape(a) for a in alts)
    cue_alt = "|".join(re.escape(c) for c in cues) or "行きたい"
    return re.compile(r"(?:" + name_alt + r")(?:さん|くん|君|ちゃん)?と"
                      r"[^。!?！？\n]{0,20}?(?:" + cue_alt + ")")


def _has_positive_cue(sim, invitee, inviter, cfg: dict) -> bool:
    """前日発話の明示キュー: 被誘者**自身**の発話に「(誘い主名)と…行きたい/行こう」型が
    あるか(_dialog_hist=直近2往復×最大8相手のリングバッファを走査)。相手の勧誘発話は
    材料にしない(当人の意向のみ)。

    婉曲拒否対策: 同一文に hedge_markers(逆接・困難表現=「山々/ですが/けど…」)が共起する
    発話はキューに数えない=フォールバックへ(「行きたいのは山々ですが」= 願望+言い訳の間接的
    意味公式を受諾に誤検出しない。Beebe et al. 1990。曖昧例は落とす=計画書§1)。"""
    hist = getattr(invitee, "_dialog_hist", None)
    if not hist:
        return False
    inviter_name = str(getattr(inviter, "name", "") or "")
    if not inviter_name:
        return False
    my_name = str(getattr(invitee, "name", "") or "")
    rx = _cue_regex(inviter_name, cfg["positive_cues"])
    hedges = cfg["hedge_markers"]
    for turns in hist.values():
        for speaker, text in turns:
            if my_name and str(speaker) != my_name:
                continue
            for sent in re.split(r"[。!?！？\n]", str(text)):
                if rx.search(sent) and not any(h in sent for h in hedges):
                    return True
    return False


def decide_accept(sim, inviter, invitee, activity, when_min, cfg) -> tuple:
    """誘いに対する被誘者の内生判定 (verdict, basis) を返す(全決定論・優先順)。

    verdict ∈ {"accept", "reject", None(=判定不能)}。basis=機械可読な判定根拠ラベル:
      conflict(当日予定と時間帯衝突=拒否)/ appointment(誘い主との当日予定=受諾)/
      plan_with(前日計画の with に誘い主=受諾)/ solo_plan(単独志向の計画が過半=拒否)/
      dialog_cue(前日発話の明示キュー=受諾)/ fallback(材料なし=較正確率へ後退)。
    activity は現状ログ・将来拡張用(判定には未使用=正直に注記)。when_min=[開始分, 終了分]。
    day は plan_day が直前に確定する sim._joint_day(誘いは常に当日分)を読む。"""
    day = int(getattr(sim, "_joint_day", 0))
    lo, hi = int(when_min[0]), int(when_min[1])

    # --- 1) 予定帳簿(schedule ON 時のみ。会話から自動記入された当人の言質) ---
    scfg = getattr(sim, "schedulecfg", None)
    if scfg and scfg.get("enabled"):
        with_inviter = False
        conflict = False
        for e in (getattr(invitee, "schedule", None) or []):
            if int(e.get("day", -1)) != day:
                continue
            if inviter.id in (e.get("with") or []):
                with_inviter = True                # 誘い主と会う予定が既にある=積極受諾
                continue
            m = _when_minute(str(e.get("when", "")))
            if m is not None and lo <= m < hi:     # 他予定と時間帯が重なる=先約
                conflict = True
        if with_inviter:                           # 誘い主との予定は衝突より優先(整合的)
            return "accept", "appointment"
        if conflict:
            return "reject", "conflict"

    # --- 2) 前日 day_schedule(朝生成=真夜中時点は前日分)の志向 ---
    sched = getattr(invitee, "day_schedule", None) or []
    if sched:
        from . import joint as _joint              # 遅延 import(循環回避)
        inviter_name = str(getattr(inviter, "name", "") or "")
        if inviter.id in _joint._resolve_with(sim, invitee) or (
                inviter_name and any(_name_match(w, inviter_name)
                                     for it in sched for w in (it.get("with") or []))):
            return "accept", "plan_with"
        items = [it for it in sched if not it.get("anchor")]  # 勤務アンカーは志向でない=除外
        if items:
            solo = sum(1 for it in items
                       if not it.get("with")
                       and str(it.get("what", "")) in cfg["solo_whats"])
            if solo * 2 > len(items):              # 過半が単独志向=消極
                return "reject", "solo_plan"

    # --- 3) 前日発話の明示キュー(prompts.dialog_history ON 時のみ) ---
    if _dialog_history_on(sim) and _has_positive_cue(sim, invitee, inviter, cfg):
        return "accept", "dialog_cue"

    return None, "fallback"                        # 判定不能=較正確率フォールバック


def compose(p_calib: float, verdict, basis: str, cfg: dict) -> tuple:
    """合成 (p, forced_reject) を返す。p = clamp01(w·p_calib + (1−w)·p_endo)。単調・有界・決定論。

    p_endo = clamp01(p_calib + positive_boost)(受諾材料)/ clamp01(p_calib − negative_cut)
    (拒否材料)/ p_calib(fallback="calibrated"=現在唯一のモード。未知値も較正へ後退=正直な縮退)。
    conflict_veto は確率でなく確定拒否(forced=True。draw は呼び手が実行済み=結果を破棄)。
    gossip_penalty の減算は**呼び手(joint.py)が最後に行う**(第61バッチの不変則)。"""
    p_calib = float(p_calib)
    w = min(1.0, max(0.0, float(cfg["prior_weight"])))
    if verdict == "accept":
        p_endo = min(1.0, max(0.0, p_calib + float(cfg["positive_boost"])))
    elif verdict == "reject":
        if basis == "conflict" and cfg["conflict_veto"]:
            return 0.0, True
        p_endo = min(1.0, max(0.0, p_calib - float(cfg["negative_cut"])))
    else:
        p_endo = p_calib
    p = w * p_calib + (1.0 - w) * p_endo
    return min(1.0, max(0.0, p)), False


# --------------------------------------------------------------------------- #
# 当日タリー(L2 観測の材料。checkpoint.py が中央管理=resume==straight)
# --------------------------------------------------------------------------- #
def day_state(sim, day: int) -> dict:
    """日境界で当日タリーを初期化して返す(joint.plan_day が ON 時のみ呼ぶ)。

    accepted/fulfilled は set だが membership/len のみ使用(反復しない=pickle の集合反復順
    非保存の影響を受けない=checkpoint.py の決定論監査条件を満たす)。"""
    st = {"day": int(day), "invites": 0, "accepts": 0, "endo": 0,
          "p_calib_sum": 0.0, "accepted": set(), "fulfilled": set()}
    sim._endo_state = st
    return st


def tally_invite(st: dict, invitee_id: int, verdict, p_calib: float,
                 accepted: bool) -> None:
    """1 誘い分をタリーへ加算(決定論・読むだけの相方)。"""
    st["invites"] += 1
    st["p_calib_sum"] += float(p_calib)
    if verdict is not None:
        st["endo"] += 1
    if accepted:
        st["accepts"] += 1
        st["accepted"].add(int(invitee_id))


def active_state(sim):
    """ON かつ当日タリーが存在するときだけ state を返す(OFF/未編成は None=素通り)。"""
    if not enabled(sim):
        return None
    jc = getattr(sim, "jointcfg", None)
    if not (jc and jc.get("enabled")):
        return None
    return getattr(sim, "_endo_state", None)


def mark_fulfilled(sim, grp: dict, st: dict) -> None:
    """履行観測: 承諾者がランデブー POI で実際に**同席**(2人以上共在)した step で fulfilled に
    加える(joint.observe が band 内で毎 step 呼ぶ。読むだけ・乱数ゼロ)。1人で来ただけ
    (他は不履行)は同席でない=履行に数えない(最も正直な形)。"""
    poi = grp.get("poi")
    present = [gid for gid in grp["members"]
               if (o := sim.agent_by_id.get(gid)) is not None and o.node == poi]
    if len(present) < 2:
        return
    for gid in present:
        if gid in st["accepted"]:
            st["fulfilled"].add(int(gid))


def scalars(sim) -> dict | None:
    """L2 全体スカラー 4 列(OFF は None=列なし=L2 不変。gossip.scalars と同型)。

      joint_accept_rate      = 当日の承諾/誘い
      joint_endo_share       = 内生判定率(verdict が付いた誘いの割合=1−fallback率)
      joint_accept_calib_gap = 承諾率 − 当日 p_calib 平均(較正期待とのズレ。確率差=×100 で pp)
      joint_fulfill_rate     = 承諾者のうち当日実際に同席した割合(関心はあるが不履行、の弁別)
    誘い 0 件の日は全列 0.0(誘い自体が無い日を「乖離ゼロ」と区別したい場合は L1 joint_invite
    の件数を参照=正直な限界)。"""
    cfg = cfg_of(sim)
    if not cfg["enabled"]:
        return None
    jc = getattr(sim, "jointcfg", None)
    if not (jc and jc.get("enabled")):
        return None
    st = getattr(sim, "_endo_state", None)
    inv = int(st["invites"]) if st else 0
    if inv <= 0:
        return {"joint_accept_rate": 0.0, "joint_endo_share": 0.0,
                "joint_accept_calib_gap": 0.0, "joint_fulfill_rate": 0.0}
    acc_rate = st["accepts"] / inv
    gap = acc_rate - st["p_calib_sum"] / inv
    n_acc = len(st["accepted"])
    fulfill = (len(st["fulfilled"]) / n_acc) if n_acc else 0.0
    return {"joint_accept_rate": round(acc_rate, 6),
            "joint_endo_share": round(st["endo"] / inv, 6),
            "joint_accept_calib_gap": round(gap, 6),
            "joint_fulfill_rate": round(fulfill, 6)}
