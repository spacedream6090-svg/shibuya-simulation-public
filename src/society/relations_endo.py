"""承諾/拒否判断の内生化 第62バッチ フェーズ1 + 誘う相手の内生選抜 第64バッチ フェーズ3
(2026-07-27。どちらも既定 OFF)。

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

─────────────────────────────────────────────────────────────────────────────
第64バッチ フェーズ3(誘う相手の選択の内生化。conf relations.endogenous_invite.*。既定 OFF)。
正典: docs/plans/endogenous-relations-plan.md §4。**本バッチは実装まで**=実験条件として回すかは
フェーズ2評価(本選実LLM・analyze_endo_treatment.py の phase3_go 機械判定)の合否ゲート
(実装と実験実施の分離=計画書の原則)。

対象は joint._companions の friend 経路の**候補集合の並べ替え・拡張のみ**(選抜の枠組み・
min/max_group・発生判定 daily_rate は不変)。ON 時の候補順:
  1. 前日 day_schedule の with 解決相手(_resolve_with=従来どおり最優先)… source=plan_with
  2. 前日発話の明示キュー相手(誘い手**自身**の「(相手名)と…行きたい/行こう」。フェーズ1の
     _has_positive_cue を役割交換で invite 方向に再利用=語彙・hedge 棄却も accept と同一の源)
     … source=dialog_cue
  3. 従来の closeness 降順(較正分布を事前分布として維持=二段構えの invite 版)… source=closeness
  4. 弱い紐帯の探索枠(weak_tie_slots。tier=1 知人から (agent,day) 安定ハッシュの決定論選択=
     乱数ゼロ・候補**末尾**)… source=weak_tie

弱い紐帯枠の理論根拠: Granovetter 1973「The Strength of Weak Ties」(AJS 78(6):1360-1380=弱い
紐帯は新規情報の橋)+ Onnela et al. 2007(PNAS 104(18):7332=交流量の大半は強い紐帯が担い、弱い
紐帯との交流イベントは現実に少数)。**誘い先が知人である現実頻度の直接統計は文献に見つからず**、
weak_tie_slots=1(候補末尾の1枠=強い紐帯候補が先に埋まらない時だけ試行される)は保守的既定。

関係の総量上限は本フェーズで変更しない: 誘いは一時的接触であり、層編成(Dunbar 入れ子層=
friend_graph の close 3-5 / friend 7-12 / acq +20)は relations の tier 遷移(closeness 蓄積)が
担う。tier は closeness からの純関数(relations.tier_of)で、誘い・同席そのものは closeness を
書き換えない(交流イベント→note_contact 経由でのみ蓄積)ため、誘い候補の拡張が層次数上限を
新規超過させる経路は構造上ない。

R1(既定 OFF=バイト一致): enabled=false のとき候補順・"joint" stream 消費・イベント・L2 とも
従来と完全同一(sim._invite_state も生えない)。ON でも LLM 呼ゼロ・追加乱数ゼロ(弱い紐帯枠は
安定ハッシュ=新 stream "invite" は不要)。ON では候補列が変わるため "joint" stream の承諾 draw
数は OFF と差が出得るが、それは treatment そのもの(承諾側の always-draw とは役割が違う=計画書
§4)。承諾抽選の draw 列に invite 由来の**別用途 draw を混ぜない**(joint stream は承諾抽選のみ)。
resume==straight: 当日タリー(sim._invite_state)は checkpoint.py が中央管理(endo_state と同型)。

─────────────────────────────────────────────────────────────────────────────
第65バッチ フェーズ4(関係の質の内生化。conf relations.endogenous_quality.*。既定 OFF)。
正典: docs/plans/endogenous-relations-plan.md §4(最終段)。従来 relations.note_contact は交流の
**符号**(valence>=0=+pos_weight / <0=−neg_weight)だけを見て、どんな会話でも増減量は一定だった。
本ブロックは既生成の会話テキストから決定論抽出した不透明係数 magnitude を増減**量**に載せる
片方向 hook(値の生成のみ。掛け算は relations.note_contact、引き渡しは engine の1関数)。

抽出材料(全て既生成テキスト・LLM 呼ゼロ・乱数ゼロ):
  1. 発話長(自己開示の**量**の粗い代理)… len_chars で正規化し len_gain まで加算
  2. 往復数(相手別リングバッファ _dialog_hist の蓄積=応答のやり取りの厚み)… turn_gain/回・turn_max 上限
  3. 明示キューの共起(フェーズ1 positive_cues を**そのまま再利用**=語彙は単一の源)… cue_gain 加算
  4. hedge_markers の共起 → **中立 1.0 に倒す**(婉曲・逆接の混じる曖昧文は関係の質に効かせない=
     フェーズ1「曖昧例はフォールバックへ」と同じ保守側の設計)
最後に [mag_min, mag_max] へ clamp。

文献根拠(選択の理由。深追いはしない=優先度低の最終段):
  - Altman & Taylor 1973 の社会的浸透理論=自己開示の breadth/depth の累積が関係を深化させる。
    Laurenceau, Barrett & Pietromonaco 1998(JPSP 74(5):1238-1251)は日誌法で「知覚された開示の
    **深さ**が当日の親密さの最強の予測子」「事実より**感情**の開示が効く」を実証。
    → 材料3(明示キュー=感情/意向の表出)を長さと別枠で加点する根拠。
  - Reis & Shaver 1988 の対人過程モデル=親密さは「自己開示 + 相手の応答性」の相互作用で生じる
    (Laurenceau et al. 1998 が partner responsiveness の媒介を実証)。
    → 材料2(往復数=応答のやり取りが実際に成立した回数)を加点する根拠。
  **正直な限界**: 文字数は開示の「深さ」の代理にならない(文献が効くと言っているのは depth と
  responsiveness であって長さではない)。長さは「厚みの下限を粗く測る」以上の主張をしない=
  len_gain を最大の加点にしない既定値(0.5 で cue 0.4 / turn 0.45 と同程度)にとどめる。

片方向 hook(計画書§4「発火判定には流さない」の厳守): magnitude は closeness の増減量にのみ入る。
  発火判定(drive/申請/抽選)・会話ペアリング(_select_partner)・tier 閾値の式(tier_of)・誘い/承諾の
  判定には**一切**渡さない=会話数・LLM 呼数・イベント数・乱数消費は ON/OFF で不変。magnitude が
  closeness の蓄積速度を変えることで tier 遷移の**時期**が動くのは treatment そのもの(意図)。

R1(既定 OFF=バイト一致): enabled=false のとき contact_magnitude は 1.0 を返すだけ(タリーも作らず
  sim._quality_state が生えない)=L1/L2/乱数消費とも従来と完全同一。ON でも LLM 呼ゼロ・乱数ゼロ。
  resume==straight: 当日タリー(sim._quality_state)は checkpoint.py が中央管理(endo/invite と同型)。
"""
from __future__ import annotations

import re

from . import dunbar as _dunbar
from . import relations as _relations
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
    # ★``joint.observe`` と同じ規約: 同席は**在場者だけ**(``agent_by_id`` は退場者も返し、
    #   その node は脱水時点の古い値なので幽霊の同席が履行率を水増しする)。
    present = [gid for gid in grp["members"]
               if (o := sim.present_agent(gid)) is not None and o.node == poi]
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


# =========================================================================== #
# 第64バッチ フェーズ3: 誘う相手の内生選抜(relations.endogenous_invite。既定 OFF)
#   実装のみ=実験投入は phase3_go ゲート(冒頭 docstring 参照)。全決定論・LLM/乱数ゼロ。
# =========================================================================== #
INVITE_DEFAULTS = {
    "enabled": False,
    # 弱い紐帯(tier=1 知人)の探索枠数(候補**末尾**に付す。0=無効)。既定 1 の根拠:
    # Granovetter 1973(弱い紐帯=新規情報の橋)を観測するための最小の探索枠。現実の交流量は
    # 強い紐帯が大半を占める(Onnela et al. 2007 PNAS 104(18):7332)ため末尾1枠=強い紐帯候補が
    # 先に埋まらない時だけ試される控えめな形。誘い先が知人である現実頻度の**直接統計は文献に
    # 見つからなかった**ので「探索枠1=保守的既定」(正直な注記。冒頭 docstring)。
    "weak_tie_slots": 1,
}


def build_invite_cfg(raw) -> dict:
    """conf の relations.endogenous_invite ブロックを正準化(既定 OFF=現行挙動と完全同一)。"""
    raw = dict(_to_plain(raw) or {})
    cfg = dict(INVITE_DEFAULTS)
    for k, v in raw.items():
        if k == "enabled":
            cfg["enabled"] = bool(v)
        elif k == "weak_tie_slots":
            cfg["weak_tie_slots"] = int(v)
    return cfg


def invite_cfg_of(sim) -> dict:
    """endogenous_invite 設定(初回のみ遅延構築してキャッシュ)。キャッシュ属性 sim.endoinvcfg は
    L1/L2/L3/乱数に一切現れない=既定 OFF のバイト一致を壊さない(cfg_of と同型)。"""
    c = getattr(sim, "endoinvcfg", None)
    if c is None:
        try:
            raw = (sim.cfg.get("relations", None) or {}).get("endogenous_invite", None)
        except Exception:
            raw = None
        c = build_invite_cfg(raw)
        sim.endoinvcfg = c
    return c


def invite_enabled(sim) -> bool:
    return bool(invite_cfg_of(sim)["enabled"])


def invite_cue_partners(sim, inviter) -> list:
    """前日発話の明示キュー相手(invite 方向): 誘い手**自身**の発話に「(相手名)と…行きたい/
    行こう」型がある relations 既知相手 id を (-closeness, id) の決定論順で返す。

    フェーズ1 _has_positive_cue の**役割交換**による再利用(invitee:=誘い手=発話を走査される側 /
    inviter:=候補=名前照合される側)。語彙(positive_cues)・婉曲棄却(hedge_markers)も
    relations.endogenous_accept ブロックと共用=単一の源。dialog_history OFF / 履歴なしは空。
    正直な限界: tier 0(顔見知り未満)の相手も名指しの明示キューなら拾う(名指しの意向=内生選択
    そのものなので tier で足切りしない)。"""
    if not _dialog_history_on(sim):
        return []
    if not getattr(inviter, "_dialog_hist", None):
        return []
    cfg = cfg_of(sim)                   # 語彙は accept ブロックと共用(単一の源)
    rows = []
    for oid, rel in (getattr(inviter.mem, "relations", {}) or {}).items():
        if oid == inviter.id:
            continue
        o = sim.present_agent(oid)                  # 街に居ない相手は誘えない(幽霊を候補にしない)
        if o is None:
            continue
        if _has_positive_cue(sim, inviter, o, cfg):
            rows.append((-float(rel.get("closeness", 0.0)), int(oid)))
    return [oid for _c, oid in sorted(rows)]


def weak_tie_candidates(sim, agent, day: int, slots: int, exclude) -> list:
    """弱い紐帯の探索枠: tier=1(知人)の relations 相手から (agent,day) キーの安定ハッシュで
    slots 体を決定論選択(乱数ゼロ=新 stream "invite" 不要。_rendezvous_poi と同じ純関数流儀)。

    pool は id 昇順に固定し、開始位置が日替わりで回る=同じ知人ばかり誘わない探索
    (Granovetter 弱い紐帯の観測目的=計画書§4)。exclude(自分/既割当/既候補)と来街者は除外。

    第75バッチ(ダンバー認知枠。relations.dunbar 既定 OFF=pool も選択も従来と完全同一):
    dunbar ON のときだけ **休眠関係(dormant)を pool に加える**。休眠紐帯は closeness を 0 へ
    退避されて tier 0 になるため closeness 降順の同伴候補経路からは自動的に外れており、
    **再会し得るのはこの探索枠だけ**=誘い選抜における一貫した1つの規則にする。理論根拠:
    Levin, Walter & Murnighan 2011(Organization Science 22(4):923-939)= 休眠紐帯の再接続は
    「新規性は弱い紐帯並み・信頼は強い紐帯並み」= 弱い紐帯と同じ探索の箱に入れるのが整合的。
    ★明示的な意向(前日計画の with / 前日発話の明示キュー)は休眠でも通る(名指しの意向は
    認知枠の淘汰より上位。_resolve_with / invite_cue_partners は名前で台帳を引くため無改変)。"""
    if slots <= 0:
        return []
    rc = getattr(sim, "relationscfg", None) or _relations.DEFAULTS
    dunbar_on = _dunbar.enabled(sim)
    pool = []
    for oid in sorted((getattr(agent.mem, "relations", {}) or {})):
        if oid == agent.id or oid in exclude:
            continue
        o = sim.present_agent(oid)                  # 探索枠も在場者のみ(幽霊が枠を食わない)
        if o is None or o.visitor:
            continue
        rel = agent.mem.relations[oid]
        if dunbar_on and rel.get("dormant"):
            pool.append(oid)                # 休眠紐帯=再会の探索枠(dunbar ON 時のみ)
            continue
        if _relations.tier_of(float(rel.get("closeness", 0.0)), rc) == 1:
            pool.append(oid)
    if not pool:
        return []
    start = (agent.id * 1000003 + int(day) * 97) % len(pool)
    return [pool[(start + i) % len(pool)] for i in range(min(int(slots), len(pool)))]


# --------------------------------------------------------------------------- #
# 誘い経路の当日タリー(L2 観測の材料。checkpoint.py が中央管理=resume==straight)
# --------------------------------------------------------------------------- #
_ENDO_INVITE_SOURCES = ("plan_with", "dialog_cue")   # 内生経路(計画 with+明示キュー)


def invite_day_state(sim, day: int) -> dict:
    """日境界で当日の誘い経路タリーを初期化して返す(joint.plan_day が invite ON 時のみ呼ぶ)。
    int のみ(set なし)=pickle/決定論監査は自明。"""
    st = {"day": int(day), "invites": 0, "weak_tie": 0, "endo": 0}
    sim._invite_state = st
    return st


def tally_invite_source(st: dict, source: str) -> None:
    """1 誘い分の選抜経路をタリーへ加算(決定論・読むだけ)。"""
    st["invites"] += 1
    if source == "weak_tie":
        st["weak_tie"] += 1
    elif source in _ENDO_INVITE_SOURCES:
        st["endo"] += 1


def invite_scalars(sim) -> dict | None:
    """L2 全体スカラー 2 列(OFF は None=列なし=L2 不変。scalars と同型)。

      invite_weak_tie_rate = 弱い紐帯経路(source=weak_tie)の誘い率(当日)
      invite_endo_share    = 内生経路(plan_with+dialog_cue)の誘い率(当日)
    誘い 0 件の日は両列 0.0(scalars の流儀と同じ正直な限界=L1 件数と併読)。"""
    cfg = invite_cfg_of(sim)
    if not cfg["enabled"]:
        return None
    jc = getattr(sim, "jointcfg", None)
    if not (jc and jc.get("enabled")):
        return None
    st = getattr(sim, "_invite_state", None)
    inv = int(st["invites"]) if st else 0
    if inv <= 0:
        return {"invite_weak_tie_rate": 0.0, "invite_endo_share": 0.0}
    return {"invite_weak_tie_rate": round(st["weak_tie"] / inv, 6),
            "invite_endo_share": round(st["endo"] / inv, 6)}


# =========================================================================== #
# 第65バッチ フェーズ4: 関係の質の内生化(relations.endogenous_quality。既定 OFF)
#   会話由来の不透明 magnitude を closeness の増減**量**にのみ載せる片方向 hook。
#   全決定論・LLM/乱数ゼロ(冒頭 docstring のフェーズ4節が設計の正典)。
# =========================================================================== #
QUALITY_DEFAULTS = {
    "enabled": False,
    # clamp 範囲(この外へは出さない=暴走防止。既定 [0.5, 2.0]=最大でも従来の2倍・半分)。
    "mag_min": 0.5,
    "mag_max": 2.0,
    # 発話長: min(1, len(text)/len_chars) × len_gain を加算。len_chars=60 は本シムの発話
    # (max_tokens 制約下の1〜2文)の実測レンジから採った「厚い発話」の目安で、文献値ではない
    # (正直な注記: 開示の深さの直接指標は自由文からは取れない=長さは粗い代理)。
    "len_chars": 60.0,
    "len_gain": 0.5,
    # 往復: (バッファ内発話数−1) × turn_gain を turn_max まで加算(応答性 = Reis & Shaver 1988)。
    # リングバッファは相手あたり最大4発話なので既定では 3×0.15=0.45=turn_max に届く。
    "turn_gain": 0.15,
    "turn_max": 0.45,
    # 明示キュー(positive_cues の共起)の加算。感情/意向の表出=Laurenceau et al. 1998。
    "cue_gain": 0.4,
}

_QUAL_FLOAT_KEYS = ("mag_min", "mag_max", "len_chars", "len_gain",
                    "turn_gain", "turn_max", "cue_gain")


def build_quality_cfg(raw) -> dict:
    """conf の relations.endogenous_quality ブロックを正準化(既定 OFF=現行挙動と完全同一)。"""
    raw = dict(_to_plain(raw) or {})
    cfg = dict(QUALITY_DEFAULTS)
    for k, v in raw.items():
        if k == "enabled":
            cfg["enabled"] = bool(v)
        elif k in _QUAL_FLOAT_KEYS:
            cfg[k] = float(v)
    return cfg


def quality_cfg_of(sim) -> dict:
    """endogenous_quality 設定(初回のみ遅延構築してキャッシュ)。キャッシュ属性 sim.endoqualcfg は
    L1/L2/L3/乱数に一切現れない=既定 OFF のバイト一致を壊さない(cfg_of と同型)。"""
    c = getattr(sim, "endoqualcfg", None)
    if c is None:
        try:
            raw = (sim.cfg.get("relations", None) or {}).get("endogenous_quality", None)
        except Exception:
            raw = None
        c = build_quality_cfg(raw)
        sim.endoqualcfg = c
    return c


def quality_enabled(sim) -> bool:
    return bool(quality_cfg_of(sim)["enabled"])


def magnitude_of(text: str, turns: int, cfg: dict, qcfg: dict) -> float:
    """会話1件の magnitude を決定論算出する純関数(乱数・LLM・sim 状態に依存しない)。

    cfg=endogenous_accept ブロック(語彙 positive_cues/hedge_markers の**単一の源**)、
    qcfg=endogenous_quality ブロック(係数)。turns=その相手との直近バッファの発話数。

    hedge_markers が本文に共起したら他の材料を見ずに 1.0(中立=従来と同値)を返す。
    保守側に倒す選択の理由: 日本語の逆接・困難表現は婉曲の断り/留保の意味公式に頻出し
    (Beebe et al. 1990。冒頭 docstring)、その文を「厚い交流」と読むと関係の質を上振れさせる。
    **正直な限界**: 語彙は accept ブロック共用で「けど/のに」等の高頻度語を含むため、
    中立へ落ちる会話の割合は実測で高くなり得る(中立比率自体を観測して報告する)。"""
    for h in cfg["hedge_markers"]:
        if h in text:
            return 1.0
    mag = 1.0
    n = len(text)
    if n > 0 and qcfg["len_chars"] > 0:
        mag += min(1.0, n / float(qcfg["len_chars"])) * float(qcfg["len_gain"])
    if turns > 1:
        mag += min(float(qcfg["turn_max"]),
                   (int(turns) - 1) * float(qcfg["turn_gain"]))
    if any(c in text for c in cfg["positive_cues"]):
        mag += float(qcfg["cue_gain"])
    return min(float(qcfg["mag_max"]), max(float(qcfg["mag_min"]), mag))


def contact_magnitude(sim, speaker, other_id: int, text: str,
                      sim_min: int) -> float:
    """交流1件の magnitude を返し、当日タリーへ加算する(OFF は常に 1.0・タリーも作らない)。

    材料は speaker 側の既生成テキスト(text)と、その相手との直近バッファ(_dialog_hist=
    prompts.dialog_history ON 時のみ存在。OFF は turns=0=長さとキューだけで判定)。
    呼び手(engine)は「不透明な float を1つ受け取って note_contact に渡す」だけ=関係語彙・
    判定ロジックは本 module 側に閉じる(no-fingerprint の作法)。"""
    if not quality_enabled(sim):
        return 1.0
    qcfg = quality_cfg_of(sim)
    hist = getattr(speaker, "_dialog_hist", None) or {}
    turns = len(hist.get(other_id) or ())
    mag = magnitude_of(str(text or ""), turns, cfg_of(sim), qcfg)
    _quality_tally(sim, int(sim_min) // 1440, mag)
    return mag


# --------------------------------------------------------------------------- #
# 会話の質の当日タリー(L2 観測の材料。checkpoint.py が中央管理=resume==straight)
# --------------------------------------------------------------------------- #
def quality_day_state(sim, day: int) -> dict | None:
    """日境界で当日タリーを初期化して返す(scheduler の関係日次フェーズが ON 時のみ呼ぶ)。
    OFF なら None=状態を一切生やさない(バイト一致)。int/float のみ=決定論監査は自明。"""
    if not quality_enabled(sim):
        return None
    st = {"day": int(day), "n": 0, "sum": 0.0, "neutral": 0}
    sim._quality_state = st
    return st


def _quality_tally(sim, day: int, mag: float) -> None:
    """1交流分をタリーへ加算(決定論)。日が変わっていれば作り直す(日境界フェーズが先に
    走るのが通常経路。ここでの再作成は phase 順に依存しないための防御=同じ結果になる)。
    neutral は magnitude が厳密に 1.0(hedge 中立・材料ゼロ)だった件数=品質の正直な指標。"""
    st = getattr(sim, "_quality_state", None)
    if st is None or int(st.get("day", -1)) != int(day):
        st = {"day": int(day), "n": 0, "sum": 0.0, "neutral": 0}
        sim._quality_state = st
    st["n"] += 1
    st["sum"] += float(mag)
    if mag == 1.0:
        st["neutral"] += 1


def quality_scalars(sim) -> dict | None:
    """L2 全体スカラー 1 列(OFF は None=列なし=L2 不変。scalars と同型)。

      quality_magnitude_mean = 当日の会話由来 magnitude の平均(交流1件=(話者,聞き手)1組
        あたり1件。両方向の note_contact には同じ値が載るので二重計上しない)。
    会話 0 件の日は 0.0(magnitude は mag_min>0 の範囲なので「会話が無い」と区別可能=
    invite/accept scalars と同じ正直な限界)。relations 本体 OFF では closeness 自体が
    動かない=列も出さない。"""
    if not quality_cfg_of(sim)["enabled"]:
        return None
    rc = getattr(sim, "relationscfg", None)
    if not (rc and rc.get("enabled")):
        return None
    st = getattr(sim, "_quality_state", None)
    n = int(st["n"]) if st else 0
    if n <= 0:
        return {"quality_magnitude_mean": 0.0}
    return {"quality_magnitude_mean": round(float(st["sum"]) / n, 6)}
