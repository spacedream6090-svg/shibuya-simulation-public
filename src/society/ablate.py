"""アブレーション 4 種(第78バッチ)+ プラセボ L1 3 種(第89バッチ 2026-08-03)
= **対照条件のスイッチだけ**を置く層。

正典: docs/plans/dual-mode-observe-verify-plan.md §2 第78行 /
      docs/plans/source/dual-mode-instructions.md Phase 1 /
      docs/plans/source/dual-mode-instruments.md Part C(2)・Part D。

何のための層か
--------------
「その振る舞いは本当に LLM 由来か」「その専門化は本当に相互作用由来か」を切り分ける
**唯一の実用的手段はアブレーション**である。観察ランで出た現象に対し、機構を 1 つずつ
外した対照ランを回し、**差分でしか主張しない**ための装置をここに集める。

  llm_off          … LLM を 1 本も呼ばず、既存のルール層(routine.decide)だけで回す。
                     「ルール層まで落としても再現される現象は LLM の手柄ではない」の下限。
  propagation_off  … 発話は通常どおり生成する(= LLM 呼の発生箇所は 1 つも変わらない)が、
                     **その内容が他エージェントの文脈に一切入らない**。専門化スコアの帰無モデル。
  cognitive_tier   … llm.fleet の割当を強制的に下位ティアへ落とす(rule/small/mid/full)。
                     現象ごとに「それが消える知能の下限」を測るダイヤル。
  shuffle_partners … 対話の相手選択を関係グラフでなく一様乱択にする。
                     「関係構造が効いているのか、単に会話量が効いているのか」の分離。

プラセボ L1 3 種(第89バッチ)
------------------------------
正典: docs/plans/source/design-discussion-20260802.md §1 —
  「L0(ルールのみ)から L4(大型モデル)まで知能の水準を段階化し、特に **L1 として
   『呼び出し回数・フォーマット・乱数消費は同一のまま中身だけ壊したプラセボ LLM』
   (文脈シャッフル、ペルソナ入れ替え、文脈遮断)を挟む**。マクロ指標が L0 から L4 へ
   単調に変化するなら、LLM 由来の知的振る舞いが集団結果を生んでいる証拠になる。」

  context_shuffle … プロンプトの**文脈節**(他者由来の内容)を、同一ラン内の別エージェントの
                    **同種**の節と決定論的に入れ替える。ペルソナと自分の状態は保持する。
                    = 「文脈が正しいこと」だけを壊す。
  persona_swap    … プロンプトの**ペルソナ節**を別エージェントのペルソナと入れ替える。
                    割当は**対合**(A↔B の相互交換)なので全単射=人口のペルソナ分布は保存される。
                    世界状態・文脈は正しいまま = 「人格と行動の結びつき」だけを壊す。
  context_sever   … 文脈節を中立プレースホルダ("…")へ置換する。節の位置・区切り・項目数は
                    保つ = 「文脈が在ること」だけを壊す。

  ★L0〜L4 の梯子と実行レシピは docs/research/ablation-ladder.md にまとめてある。

  propagation_off(第78)との違い(必ず区別すること)
    propagation_off … **送り手側**の遮断。発話内容が「他エージェントの記憶・語彙・信念・
                      予定帳簿・TL 本文」へ入るのを止める。止まった結果として受け手の
                      プロンプトから該当節が**消える**(節ごと出ない)。世界状態(記憶)も変わる。
    context_sever   … **受け手側**の遮断。世界状態(記憶・語彙・関係)は**一切変えず**、
                      プロンプトを組む最後の一手前で文脈節の中身だけを "…" に潰す。
                      節は**消えない**(位置・区切り・項目数が残る)。
                      = 「文脈が届いていない」ではなく「文脈が読めない」条件。
    両者は併用不可(下の相互排他規約)。片方は伝播の因果、片方は文脈利用の因果を測る。

  共通規約(3 種とも)
    - **LLM 呼び出しの発生箇所を 1 つも足さない/減らさない**(プロンプト文字列を書き換える
      だけ。generate() の呼び出し点は 1 行も変わらない)= affects_k=False。
    - **プロンプト書式を保つ**: 節の**位置**(行の並び)・**接頭辞**・**区切り文字**・
      **項目数**を保ち、中身だけを差し替える。行数は増減しない。
    - **乱数は専用 stream のみ**("ablate_persona_swap" / "ablate_context_shuffle")。
      RngHub はステートレス(stream をキーから都度導出)なので、既存 stream の draw 順・値に
      1 バイトも干渉しない = 既定 OFF は golden バイト一致・draw 数も不変。
    - **相互排他**: 3 種は同じ節を書き換える**同一軸**の条件なので同時 ON は解釈不能
      → build_cfg が ValueError。propagation_off との併用も ValueError(上記のとおり
      「消える」と「潰す」が重なると節の同定ができない)。llm_off との併用はプロンプトが
      1 つも組まれないため**プラセボが無効**になる矛盾設定 → WARNING で告知する。

  ★fingerprint_risk の正直な宣言(**risk=known**)
    3 種とも「当人から観測できる差分」を**意図的に作る**条件である。中身が入れ替われば
    エージェントの応答は変わり、世界も変わる。**それが目的**(変わらなければプラセボとして
    無意味 = 第89 検収 3)。具体的には:
      context_shuffle … 見覚えのない出来事・その場に居ない人の名前・噛み合わない直前のやりとりが
                        自分の文脈として提示される(「知らない話をされている」ことに気づきうる)。
      persona_swap    … 自己紹介文と、自分の記憶・関係・持ち物・場所が食い違う
                        (「自分の経歴と自分の記憶が合わない」ことに気づきうる)。
      context_sever   … 「直近の出来事: … / … / …」のように**節はあるが中身が無い**状態が続く
                        (第78 が棄却した「合成文の注入」に近い痕跡を、ここでは**受け手自身の
                        プロンプトにだけ**意図的に作る。他者の記憶・世界状態には 1 バイトも
                        書かないので、語彙指標・専門化スコアの汚染は起きない)。
    隠す方法は存在しない(中身を壊すのが定義)ので registry へ known として登録し、
    manifest / summary にも placebo の内訳を必ず残す。

R1 との関係(**全て既定 OFF**)
------------------------------
- 既定値(llm_off=false / propagation_off=false / cognitive_tier="full" /
  shuffle_partners=false)では、本 module の述語は全て False を返し、呼び出し側は
  **1 バイトも挙動が変わらない**(golden L1 バイト一致)。
- shuffle_partners だけは専用 stream `"shuffle_partners"` から **always-draw**
  (ON/OFF に関係なく相手候補が 1 人以上いれば必ず 1 本引く)。RngHub はステートレスで
  stream をキーから都度導出するため、**新 stream の消費は既存 stream の draw 順に
  一切干渉しない**(= golden は無風のまま draw 数不変も同時に満たせる)。
- 各スイッチの repro_tier / affects_k / fingerprint_risk は src/society/registry.py に
  宣言してある(未宣言検出 CI が空を固定する)。

no-fingerprint の自己点検(propagation_off)
-------------------------------------------
発話が返ってこないことで「状況が異常だ」と当人が推論できてしまう経路があるか、を実装前に
洗い出した。採った方式は **話者側の自己連続性は保ち・聞き手側の内容摂取だけを切る**:

  保つもの  … 自分の発話(said / 自分の mem / 自分の dialog_hist)・同席者の顔ぶれ・
              `hear` の発生そのもの(L1)・接触台帳(count/closeness)・返答権の付与
              (= **返答の LLM 呼はこれまでどおり撃たれる** = 呼数の構造が不変)・
              「話しかけられた」drive・感情価/覚醒/意見のスカラー(交流の量)。
  切るもの  … 他者の記憶に積まれる発話文・他者の dialog_hist・他者へ渡る語彙(adopted)・
              他者の予定帳簿への抽出・他者の信念(伝聞)・SNS タイムライン本文の提示・
              DM 本文・告知 DM/貼り紙/提案文の本文。

  遮断点(実装の全リスト。監査のためここに列挙を保つ)
    engine/scheduler.py  _hear_words 冒頭 …… **語彙の授受**(from_id>=0 の全チャネル。
                             判定はチャネル名でなく from_id の符号=新チャネルにも効く)
    engine/scheduler.py  _apply(speak) …… 相手の _reply_to 本文 / 相手側 _dialog_push /
                             聞き手の remember / 聞き手の予定抽出 / _contact の本文
    engine/scheduler.py  _apply(dm)    …… 受信者の remember / 予定抽出 / _contact の本文
    engine/scheduler.py  _decide(reply)…… 返答プロンプトの reply_to(状況行ごと出さない)
    engine/scheduler.py  _feed_texts / _phone(SNS 閲覧)…… TL 本文の提示と記憶化
    conversation.py      _apply_effects …… C2 の語彙接触(テキストは無いが語は渡している)
    truth_ledger.py      phase …… 伝聞による信念の書き換え(_transmit_pass)
    tools.py             告知 DM / 貼り紙 / 提案への賛否 …… 他者が書いた本文の記憶化

  ★**合成文を注入しない**: 「誰かが何か言っていた」等のプレースホルダを他者の記憶へ入れる
    案は棄却した。その文字列は他のどの条件にも現れない**新種の痕跡**になり、語彙指標も
    汚染するため、痕跡としてはむしろ強くなる。

  ★残存リスク(正直に宣言する): この条件下の世界は **「周りは居るが誰も何も言わない街」**
    として当人から観測できる。返答トリガ(reply)のプロンプトからは状況行が丸ごと落ちるので、
    「話しかけられたのに内容が無い」形にはならない(状況行を出さない)。それでも
    「同席者が居るのに聞いた覚えが一度も無い」という統計的異常は原理的に残る。
    内容を伝えずに伝えたように見せる方法は存在しない(合成文=別の痕跡)ので、
    ここは **fingerprint_risk="known"** として登録し、隠さず報告する。
"""
from __future__ import annotations

import logging

log = logging.getLogger("society.ablate")

# 認知階層のダイヤル(下位 → 上位)。"rule" は LLM を使わないルール層(= llm_off と同じ層)。
TIERS = ("rule", "small", "mid", "full")

_DEFAULTS = {
    "llm_off": False,
    "propagation_off": False,
    "cognitive_tier": "full",
    "shuffle_partners": False,
    # ---- プラセボ L1 3 種(第89バッチ)。同一軸なので同時 ON 不可(build_cfg が検査)----
    "context_shuffle": False,
    "persona_swap": False,
    "context_sever": False,
}

# プラセボ 3 種のキー(同一軸=相互排他の判定に使う唯一の源)
PLACEBOS = ("context_shuffle", "persona_swap", "context_sever")


# --------------------------------------------------------------------------- #
# cfg 正準化(observer/echo.cfg_of と同型: 取得不能でも落ちない)
# --------------------------------------------------------------------------- #
def build_cfg(block) -> dict:
    """conf の `ablate` ブロック → 正準 dict。None/欠損は既定(全 OFF)。"""
    out = dict(_DEFAULTS)
    if block is None:
        return out
    try:
        get = block.get
    except AttributeError:
        return out
    for key in ("llm_off", "propagation_off", "shuffle_partners", *PLACEBOS):
        try:
            out[key] = bool(get(key, out[key]))
        except (TypeError, ValueError):
            pass
    tier = get("cognitive_tier", out["cognitive_tier"])
    tier = str(tier).strip().lower() if tier is not None else "full"
    if tier not in TIERS:
        raise ValueError(
            f"ablate.cognitive_tier='{tier}' は未知(有効値: {', '.join(TIERS)})。"
            "既定 full = 現行動作。")
    out["cognitive_tier"] = tier
    _check_placebo_exclusion(out)
    return out


def _check_placebo_exclusion(cfg: dict) -> None:
    """プラセボ L1 の相互排他規約(第89)。矛盾設定を**構築時に**落とす。

    - プラセボ 3 種は同じ「文脈節/ペルソナ節」を書き換える**同一軸**の条件。同時 ON にすると
      どちらの書き換えが効いたのか節ごとに同定できず、対照として解釈不能になる → ValueError。
    - propagation_off との併用も禁止(あちらは節を**消す**・こちらは節の中身を**潰す/入れ替える**。
      重なると「消えた節」と「潰した節」を区別できない)→ ValueError。
    - llm_off との併用はプロンプトが 1 つも組まれないため**プラセボが 1 バイトも効かない**。
      落とすほどの矛盾ではない(llm_off が上位の条件)ので WARNING で告知して素通しする。
    """
    on = [k for k in PLACEBOS if cfg.get(k)]
    if len(on) > 1:
        raise ValueError(
            f"ablate のプラセボは同時に 1 つだけ有効にできる(指定: {', '.join(on)})。"
            "3 種は同じ節を書き換える同一軸の条件なので、同時 ON は対照として解釈不能。"
            "L1 の 3 条件は**別々のラン**として回すこと(docs/research/ablation-ladder.md)。")
    if on and cfg.get("propagation_off"):
        raise ValueError(
            f"ablate.propagation_off と ablate.{on[0]} は併用できない。"
            "propagation_off は節を消し(送り手側の遮断)、プラセボは節の中身を潰す/入れ替える"
            "(受け手側)。重ねると節の同定ができず差分の帰属先が失われる。")
    if on and cfg.get("llm_off"):
        log.warning("[ablate] llm_off=true のため ablate.%s は 1 バイトも効かない"
                    "(プロンプトが 1 つも組まれない)。L0 と L1 は別ランで回すこと。", on[0])


def cfg_of(sim) -> dict:
    """sim が構築時に正準化した設定(無ければ既定=全 OFF)。"""
    cfg = getattr(sim, "ablatecfg", None)
    return cfg if cfg is not None else dict(_DEFAULTS)


# --------------------------------------------------------------------------- #
# 述語(ホットパスから呼ばれる。dict 参照 1 回で済ませる)
# --------------------------------------------------------------------------- #
def llm_off(sim) -> bool:
    """LLM を 1 本も呼ばないか。cognitive_tier="rule" も同じルール層に落とす。"""
    c = cfg_of(sim)
    return bool(c["llm_off"]) or c["cognitive_tier"] == "rule"


def propagation_off(sim) -> bool:
    """発話内容が他エージェントの文脈へ入るのを断つか(発話生成そのものは行う)。"""
    return bool(cfg_of(sim)["propagation_off"])


def shuffle_partners(sim) -> bool:
    """対話相手を関係グラフでなく一様乱択にするか。"""
    return bool(cfg_of(sim)["shuffle_partners"])


def cognitive_tier(sim) -> str:
    return cfg_of(sim)["cognitive_tier"]


def context_shuffle(sim) -> bool:
    """他者由来の文脈節を別エージェントの同種節と入れ替えるか(第89 プラセボ L1)。"""
    return bool(cfg_of(sim)["context_shuffle"])


def persona_swap(sim) -> bool:
    """ペルソナ節を別エージェントのペルソナと対合で入れ替えるか(第89 プラセボ L1)。"""
    return bool(cfg_of(sim)["persona_swap"])


def context_sever(sim) -> bool:
    """文脈節を中立プレースホルダへ潰すか(第89 プラセボ L1)。"""
    return bool(cfg_of(sim)["context_sever"])


def placebo_mode(sim) -> str | None:
    """有効なプラセボの名前(全 OFF なら None)。相互排他なので高々 1 つ。"""
    c = cfg_of(sim)
    for key in PLACEBOS:
        if c.get(key):
            return key
    return None


def any_on(sim) -> bool:
    c = cfg_of(sim)
    return bool(c["llm_off"] or c["propagation_off"] or c["shuffle_partners"]
                or c["cognitive_tier"] != "full"
                or any(c.get(k) for k in PLACEBOS))


# --------------------------------------------------------------------------- #
# propagation_off の適用ヘルパ(呼び出し側を 1 行の差分に保つための口)
# --------------------------------------------------------------------------- #
def heard_text(sim, text: str) -> str:
    """**他者**の文脈へ渡そうとしている発話文。propagation_off なら空文字にする。"""
    return "" if propagation_off(sim) else text


def heard_ids(sim, ids):
    """発話から**他者**の帳簿へ書き込む対象 id 列。propagation_off なら空にする。"""
    return [] if propagation_off(sim) else ids


# --------------------------------------------------------------------------- #
# shuffle_partners(専用 stream・always-draw)
# --------------------------------------------------------------------------- #
STREAM = "shuffle_partners"


def pick_partner(sim, agent, hearers):
    """候補から 1 人を選ぶ(always-draw)。

    **必ず 1 本引く**(ON/OFF・使う/使わないに関わらず)。返り値は
      ON  … 一様乱択した 1 人
      OFF … None(呼び出し側は従来の決定論選択を続ける)
    RngHub はステートレス(stream はキーから都度導出)なので、この新 stream の消費は
    既存 stream の draw 順・値に一切干渉しない = 既定 OFF ランは golden バイト一致。
    候補列は id 昇順(perception.hearers_of の出力順)を前提にした添字選択で決定論。
    """
    if not hearers:
        return None
    step = int(getattr(agent, "now_step", 0) or 0)
    rng = sim.hub.stream(STREAM, agent.id, step)
    idx = int(rng.integers(len(hearers)))         # ★always-draw(OFF でも引く)
    if not shuffle_partners(sim):
        return None
    return hearers[idx]


# --------------------------------------------------------------------------- #
# 報告(manifest / ログ)
# --------------------------------------------------------------------------- #
def describe(sim) -> dict | None:
    """有効なアブレーションの要約。全 OFF なら None(manifest にキーを足さない)。"""
    if not any_on(sim):
        return None
    c = cfg_of(sim)
    out = {"llm_off": bool(c["llm_off"]),
           "propagation_off": bool(c["propagation_off"]),
           "cognitive_tier": c["cognitive_tier"],
           "shuffle_partners": bool(c["shuffle_partners"]),
           "tier_effective": tier_effectiveness(sim)}
    out.update({k: bool(c.get(k)) for k in PLACEBOS})   # 第89: 3 種の状態を必ず残す
    mode = placebo_mode(sim)
    if mode is not None:                        # 第89: プラセボ L1(OFF ではキーを足さない)
        out["placebo"] = {
            "mode": mode, "level": "L1",
            "sections": [s.kind for s in SECTIONS],
            "persona_pairs": len(getattr(sim, "_placebo", None).pairing) // 2
            if getattr(sim, "_placebo", None) is not None else 0,
            "fingerprint_risk": "known",
            "note": "呼び出し点・書式・乱数消費は不変。中身だけを壊す条件なので"
                    "世界は変わる(変わらなければプラセボとして無意味)。"}
    return out


def tier_effectiveness(sim) -> dict:
    """cognitive_tier が**実際に効いているか**を正直に返す(縮退の宣言)。

    fleet(複数サーバ/ tiers 指定)を使っていないランでは small / mid は
    **1 バイトも効かない**(割り当て先のプールが 1 つしか無い)。これを黙って
    「ティアを下げた」と記録すると対照実験の解釈が壊れるので、実効性を明示する。
    """
    tier = cognitive_tier(sim)
    if tier == "full":
        return {"tier": tier, "applied": False, "reason": "既定(上書きなし)"}
    if tier == "rule":
        return {"tier": tier, "applied": True,
                "reason": "LLM を呼ばないルール層(llm_off と同一)"}
    backend = _fleet_of(getattr(sim, "llm", None))
    if backend is None:
        return {"tier": tier, "applied": False,
                "reason": "FleetLLM 不使用のランなので small/mid は縮退して full と同一"}
    pools = getattr(backend, "_tiers", {}) or {}
    if tier not in pools:
        return {"tier": tier, "applied": False,
                "reason": f"model.tiers に '{tier}' プールが無いので default へ後退(full と同一)"}
    return {"tier": tier, "applied": True,
            "reason": f"FleetLLM の全 purpose を '{tier}' プールへ強制"}


def _fleet_of(llm):
    """CachedLLM / RouterLLM の下にある FleetLLM を 1 つ見つける(無ければ None)。"""
    seen = set()
    stack = [llm]
    while stack:
        node = stack.pop()
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        if type(node).__name__ == "FleetLLM":
            return node
        child = getattr(node, "backend", None)
        if child is not None:
            stack.append(child)
        children = getattr(node, "children", None)
        if isinstance(children, dict):
            stack.extend(children.values())
    return None


def apply_fleet_tier(sim) -> None:
    """cognitive_tier を FleetLLM の割当へ焼き付ける(構築時に 1 回だけ)。

    small / mid のときだけ、fleet の全 purpose をそのティアのプールへ強制する。
    fleet を使っていない/そのプールが無いランでは **何もしない**(縮退を warning で告知)。
    """
    tier = cognitive_tier(sim)
    if tier in ("full", "rule"):
        return
    eff = tier_effectiveness(sim)
    if not eff["applied"]:
        log.warning("[ablate.cognitive_tier=%s] %s", tier, eff["reason"])
        return
    _fleet_of(sim.llm).force_tier = tier
    log.warning("[ablate.cognitive_tier=%s] FleetLLM の全 purpose を '%s' プールへ強制した",
                tier, tier)


# =========================================================================== #
# プラセボ・アブレーション L1(第89バッチ 2026-08-03)
#   正典: docs/plans/source/design-discussion-20260802.md §1 /
#         docs/plans/dayplan-engaged-plan.md 第89 / docs/research/ablation-ladder.md
#
# 唯一の作用点は `cognition/deliberate.build_prompt` の末尾 2 行(組み上がった行リストを
# 差し替えるだけ)。呼び出し元(発話・朝の計画・夜の内省・recall・一括発行)は 1 行も変えない
# ので、**generate() の呼び出し点は足しも減りもしない**(affects_k=False の構造的根拠)。
# =========================================================================== #
STREAM_SWAP = "ablate_persona_swap"
STREAM_SHUFFLE = "ablate_context_shuffle"

# 中立プレースホルダ(context_sever)。**1 文字の記号だけ**にする理由:
#   「(記録なし)」「誰かが何か言っていた」等の**文**は、他条件に現れない新種の痕跡になり
#   語彙指標(専門化スコア・n-gram・エントロピー)を汚染する。第78 が他者の記憶への合成文
#   注入を棄却したのと同じ判断を、ここでも文字列の側で守る。三点リーダは既存プロンプト
#   (memory_fail の「はっきりしない…」)に既に現れる字なので、新しい字種も持ち込まない。
SEVER_TOKEN = "…"

# 直近の同種節を貯める輪(context_shuffle のドナー源)の上限。節種ごとに FIFO。
# 大きくしても質は上がらない(同種であればどれでも「自分のものではない文脈」なので)。
# 小さく固定することで checkpoint 容量・pickle 時間を 25 万体でも一定に保つ。
RING_CAP = 32


class Section:
    """プロンプトの「文脈節」1 種の記述(接頭辞・区切り・再構成)。

    **監査のためここに列挙を保つ**(第78 の遮断点リストと同じ流儀)。`prefix` は
    `cognition/deliberate.build_prompt` が実際に組む文字列と一字一句同一でなければならず、
    tests/test_placebo.py が build_prompt の出力に対して機械的に突き合わせる。
    """

    __slots__ = ("kind", "prefix", "sep", "why")

    def __init__(self, kind: str, prefix: str, sep: str, why: str):
        self.kind, self.prefix, self.sep, self.why = kind, prefix, sep, why

    def match(self, line: str) -> str | None:
        """その行が本節なら**中身**(接頭辞を除いた部分)を返す。違えば None。"""
        if line.startswith(self.prefix):
            return line[len(self.prefix):]
        return None

    def render(self, content: str) -> str:
        return self.prefix + content

    def sever(self, content: str) -> str:
        """項目数と区切りを保ったまま中身だけを潰す(= 書式は 1 バイトも変えない)。"""
        n = content.count(self.sep) + 1 if self.sep else 1
        return self.sep.join([SEVER_TOKEN] * n) if self.sep else SEVER_TOKEN


# --------------------------------------------------------------------------- #
# 文脈節の一覧(= プラセボ 3 種が触る**全て**。ここに無い行は 1 バイトも触らない)
#
# 入れる基準 …… 「**他者・社会環境から受け取った内容**」がその行の中身であること。
# 入れない基準(意図的な除外。理由を必ず添える)
#   ヘッダ / ペルソナ / オントロジー行 …… persona_swap の対象(文脈ではない)
#   時刻・場所・いま・気分・周りにある店 …… 自分と物理世界の状態(「世界状態は正しいまま」)
#   自分の理解(内省より)/ 最近の自分 / あなたの考え / 昨日までの日記 /
#   あなたがさっき言ったこと / 馴染みの場所 …… **自分由来**(内省・自分の発話・自分の訪問履歴)。
#                                    仕様「ペルソナ・自分の状態は保持」を守るため触らない。
#   日付・天気・年中行事・災害・広告・取り決め・街の動き・群衆 …… 全員共通の環境放送(k 非依存)
#   評判行「あなたは街でそこそこ名前が知られている」…… 中身を持たない固定文(壊す先が無い)
#   所持ツール / p2_offers / watch / engaged / 状況(social,post,dm,solo)/ 驚き ……
#                                    **指示文**であって文脈ではない(書式を壊すと課題が変わる)
#   「(…のことを思い出そうとしたが、はっきりしない…)」…… 想起**失敗**の記録=中身が無い
# --------------------------------------------------------------------------- #
SECTIONS: tuple[Section, ...] = (
    Section("known_words", "知っている言葉: ", "、",
            "他者から受け取った語彙(adopted)"),
    Section("recent", "直近の出来事: ", " / ",
            "直近の記憶(他者の発話・接触の記録を含む)"),
    Section("retrieved", "記憶に残っていること: ", " / ",
            "手掛かり想起で引いた記憶"),
    Section("pull", "ふと思い出したこと: ", " / ",
            "agentic pull で引いた記憶"),
    Section("relation", "間柄: ", "、",
            "同席者との関係(他者との履歴)"),
    Section("household", "同席の身近な人: ", "、",
            "同席する同居者・恋人(他者)"),
    Section("nearby", "近くにいる人: ", "、",
            "同席者の顔ぶれ(他者)"),
    Section("feed", "タイムライン: ", " / ",
            "他者が書いた SNS 本文"),
    Section("dialog", "直前のやりとり: ", " → ",
            "相手との直近のやりとり(他者の発話)"),
)

# 返答トリガの状況行だけは「接頭辞 + 中身」の形をしていないので専用に扱う。
#   状況: {話者}に話しかけられた:「{発話}」。相手に自然に返事をする(speak で)。
# 中身 = (話者名, 発話文)。末尾の指示文(「相手に自然に返事をする…」以降)は**指示**なので
# 常に自分のものを使う(= 課題が変わらない)。
REPLY_HEAD = "状況: "
REPLY_MID = "に話しかけられた:「"
REPLY_TAIL = "」。"
REPLY_KIND = "reply"


def _reply_parts(line: str):
    """返答の状況行を (話者, 発話, 末尾の指示文) に割る。該当しなければ None。"""
    if not line.startswith(REPLY_HEAD) or REPLY_MID not in line:
        return None
    body = line[len(REPLY_HEAD):]
    who, rest = body.split(REPLY_MID, 1)
    if REPLY_TAIL not in rest:
        return None
    said, tail = rest.rsplit(REPLY_TAIL, 1)
    return who, said, tail


def _reply_render(who: str, said: str, tail: str) -> str:
    return f"{REPLY_HEAD}{who}{REPLY_MID}{said}{REPLY_TAIL}{tail}"


ALL_KINDS = tuple([s.kind for s in SECTIONS] + [REPLY_KIND])


class Placebo:
    """プラセボ L1 の本体(3 種で 1 つのクラス。触る節の定義を共有するため)。

    状態は「直近の同種節の輪(context_shuffle 用)」と「ペルソナ対合表(persona_swap 用)」の
    2 つだけ。前者は checkpoint の runtime へ中央管理で保存する(resume==straight のため)。
    後者は構築時の名簿から決定論に導出するので resume でも同一に再構成される。
    """

    def __init__(self, mode: str, hub, cap: int = RING_CAP):
        self.mode = mode
        self.hub = hub
        self.cap = int(cap)
        self.personas: dict[int, str] = {}
        self.pairing: dict[int, int] = {}
        self.rings: dict[str, list] = {k: [] for k in ALL_KINDS}
        # 観測用カウンタ(summary.placebo。世界は読まない=R1)
        self.n_prompts = 0
        self.n_persona_swapped = 0
        self.n_persona_selfpair = 0        # 対合の余り(奇数人口)= 自分自身へ写る個体の呼数
        self.n_persona_unpaired = 0        # 構築後に生えた個体(pool 再入場)= 一方向ドナー
        self.n_sections_shuffled = 0
        self.n_sections_severed = 0
        self.n_shuffle_starved = 0         # 輪にドナーが居らず sever へ後退した節数

    # ---- 名簿の登録と対合の構成 ------------------------------------------- #
    def register(self, agent) -> None:
        """個体 1 体を登録する(誕生時。pool の再入場でも呼ばれる)。"""
        self.personas[int(agent.id)] = agent.persona

    def build_pairing(self, ids) -> None:
        """ペルソナの**対合**(A↔B の相互交換)を専用 stream から 1 回だけ組む。

        対合(involution)にする理由: 写像が全単射かつ自己逆なので、**人口全体のペルソナ分布が
        完全に保存される**(誰か 1 人のペルソナが 2 回使われることが無い)。片方向の置換だと
        「人気ペルソナ」が偏って増え、分布が変わったせいなのか結びつきを壊したせいなのかを
        分離できなくなる。奇数人口では最後の 1 人だけが自分自身へ写る(= 交換なし)。
        """
        ids = sorted(int(i) for i in ids)
        rng = self.hub.stream(STREAM_SWAP, len(ids))
        order = [ids[int(i)] for i in rng.permutation(len(ids))]
        self.pairing = {}
        for i in range(0, len(order) - 1, 2):
            a, b = order[i], order[i + 1]
            self.pairing[a] = b
            self.pairing[b] = a
        if len(order) % 2:                       # 余りは自分自身(= 交換なし)
            self.pairing[order[-1]] = order[-1]

    def partner_of(self, aid: int) -> int:
        """ペルソナの相手。対合表に無い個体(構築後に生えた pool 個体)は一方向ドナー。"""
        aid = int(aid)
        if aid in self.pairing:
            return self.pairing[aid]
        keys = sorted(self.personas)
        if not keys:
            return aid
        rng = self.hub.stream(STREAM_SWAP, "unpaired", aid)
        return keys[int(rng.integers(len(keys)))]

    # ---- 適用(build_prompt の末尾から 1 回だけ呼ばれる)-------------------- #
    def apply(self, agent, lines: list[str], step: int) -> list[str]:
        """組み上がった行リストを書き換えて返す。**行数は変えない**。"""
        self.n_prompts += 1
        if self.mode == "persona_swap":
            return self._apply_persona(agent, lines)
        return self._apply_context(agent, lines, step)

    def _apply_persona(self, agent, lines: list[str]) -> list[str]:
        idx = _persona_index(lines, agent)
        if idx is None:                          # 想定外(persona 行が無い)= 触らない
            return lines
        pid = self.partner_of(agent.id)
        if pid == int(agent.id):
            self.n_persona_selfpair += 1
            return lines
        donor = self.personas.get(pid)
        if donor is None:
            return lines
        if int(agent.id) not in self.pairing:
            self.n_persona_unpaired += 1
        out = list(lines)
        out[idx] = donor
        self.n_persona_swapped += 1
        return out

    def _apply_context(self, agent, lines: list[str], step: int) -> list[str]:
        shuffle = self.mode == "context_shuffle"
        out = list(lines)
        for i, line in enumerate(out):
            hit = None
            for sec in SECTIONS:
                content = sec.match(line)
                if content:
                    hit = (sec.kind, content, sec)
                    break
            if hit is None:
                parts = _reply_parts(line)
                if parts is None:
                    continue
                who, said, tail = parts
                if shuffle:
                    donor = self._draw(REPLY_KIND, [who, said], agent.id, step)
                    if donor is None:
                        out[i] = _reply_render(SEVER_TOKEN, SEVER_TOKEN, tail)
                        self.n_shuffle_starved += 1
                    else:
                        out[i] = _reply_render(donor[0], donor[1], tail)
                        self.n_sections_shuffled += 1
                    self._push(REPLY_KIND, [who, said])
                else:
                    out[i] = _reply_render(SEVER_TOKEN, SEVER_TOKEN, tail)
                    self.n_sections_severed += 1
                continue
            kind, content, sec = hit
            if shuffle:
                donor = self._draw(kind, content, agent.id, step)
                if donor is None:
                    out[i] = sec.render(sec.sever(content))
                    self.n_shuffle_starved += 1
                else:
                    out[i] = sec.render(donor)
                    self.n_sections_shuffled += 1
                self._push(kind, content)
            else:
                out[i] = sec.render(sec.sever(content))
                self.n_sections_severed += 1
        return out

    # ---- ドナーの輪(同一ラン内の別エージェントの同種節)-------------------- #
    def _draw(self, kind: str, own, aid: int, step: int):
        """同種の輪から**自分のものでない**中身を 1 つ決定論に引く。空なら None。

        輪はプロンプト構築順(= 決定論)に詰まるので、同 seed の 2 ランは完全に一致する。
        「同一ラン内の別エージェントの同種文脈」という要件は、輪が**自分の中身を除いた
        直近 cap 件**であることで満たす(引く前に押し込まないので自分は候補に入らない)。
        """
        cands = [c for c in self.rings.get(kind, ()) if c != own]
        if not cands:
            return None
        rng = self.hub.stream(STREAM_SHUFFLE, int(aid), int(step), kind)
        return cands[int(rng.integers(len(cands)))]

    def _push(self, kind: str, content) -> None:
        ring = self.rings.setdefault(kind, [])
        ring.append(content)
        if len(ring) > self.cap:
            del ring[0:len(ring) - self.cap]

    # ---- checkpoint(中央管理。輪だけが「跨ぐ状態」)------------------------ #
    def state(self) -> dict:
        return {"rings": {k: list(v) for k, v in self.rings.items()},
                "counters": [self.n_prompts, self.n_persona_swapped,
                             self.n_persona_selfpair, self.n_persona_unpaired,
                             self.n_sections_shuffled, self.n_sections_severed,
                             self.n_shuffle_starved]}

    def restore(self, state: dict | None) -> None:
        if not state:
            return
        rings = state.get("rings") or {}
        self.rings = {k: list(rings.get(k, ())) for k in ALL_KINDS}
        c = list(state.get("counters") or [])
        if len(c) == 7:
            (self.n_prompts, self.n_persona_swapped, self.n_persona_selfpair,
             self.n_persona_unpaired, self.n_sections_shuffled,
             self.n_sections_severed, self.n_shuffle_starved) = c


def _persona_index(lines: list[str], agent) -> int | None:
    """ペルソナ行の位置。build_prompt は必ず lines[1] に置くが、突き合わせで確認する。"""
    persona = getattr(agent, "persona", None)
    if not persona:
        return None
    if len(lines) > 1 and lines[1] == persona:
        return 1
    for i, line in enumerate(lines):
        if line == persona:
            return i
    return None


# --------------------------------------------------------------------------- #
# 配線(Simulation / checkpoint から呼ばれる口。既定 OFF は全て no-op)
# --------------------------------------------------------------------------- #
def make_placebo(sim):
    """プラセボが 1 つ有効なら Placebo を作って sim._placebo に据える(OFF は None)。"""
    mode = placebo_mode(sim)
    sim._placebo = Placebo(mode, sim.hub) if mode is not None else None
    if mode is not None:
        log.warning("[ablate.%s] プラセボ L1 を有効化した。プロンプトの中身が壊れるので"
                    "**この条件の世界は変わる**(fingerprint_risk=known)。", mode)
    return sim._placebo


def attach_agent(sim, agent) -> None:
    """個体 1 体をプラセボへ結線する(誕生時。既定 OFF は属性を 1 つも生やさない)。"""
    pl = getattr(sim, "_placebo", None)
    if pl is None:
        return
    pl.register(agent)
    agent.placebo = pl


def finish_placebo(sim) -> None:
    """名簿が出揃った時点で対合を組む(構築時 1 回だけ)。既定 OFF は no-op。"""
    pl = getattr(sim, "_placebo", None)
    if pl is None:
        return
    pl.build_pairing([a.id for a in sim.agents])


def placebo_state(sim):
    """checkpoint へ保存する状態(OFF は None=旧 checkpoint と同形)。"""
    pl = getattr(sim, "_placebo", None)
    return pl.state() if pl is not None else None


def placebo_restore(sim, state) -> None:
    """resume: 輪を戻し、pickle から復元された個体を**構築時の Placebo へ**繋ぎ直す。

    checkpoint の agents pickle には(プラセボ ON のランでは)Placebo が同梱されるが、
    正本は「今のプロセスが構築時に作った sim._placebo」の 1 つだけにする。ここで繋ぎ直さないと
    輪が 2 つに割れて resume != straight になる。
    """
    pl = getattr(sim, "_placebo", None)
    if pl is None:
        for agent in getattr(sim, "agents", ()) or ():
            if hasattr(agent, "placebo"):        # OFF なのに旧 ckpt が持っていたら剥がす
                del agent.placebo
        return
    pl.restore(state)
    for agent in sim.agents:
        pl.register(agent)
        agent.placebo = pl
    pl.build_pairing([a.id for a in sim.agents])


def provenance(sim) -> dict | None:
    """summary の `placebo` キー(OFF はキー自体を出さない)。

    「壊した量」を必ず残す。壊れていないのに L0→L4 の単調性を主張するのは不正なので、
    節の書き換え件数が 0 の条件は**プラセボとして無効**と判る形にしておく。
    """
    pl = getattr(sim, "_placebo", None)
    if pl is None:
        return None
    out = {"mode": pl.mode, "level": "L1", "prompts": pl.n_prompts,
           "fingerprint_risk": "known",
           "sections_watched": list(ALL_KINDS)}
    if pl.mode == "persona_swap":
        out.update(persona_swapped=pl.n_persona_swapped,
                   persona_selfpair=pl.n_persona_selfpair,
                   persona_unpaired=pl.n_persona_unpaired,
                   pairs=len(pl.pairing) // 2,
                   involution=bool(pl.n_persona_unpaired == 0))
    else:
        out.update(sections_shuffled=pl.n_sections_shuffled,
                   sections_severed=pl.n_sections_severed,
                   shuffle_starved=pl.n_shuffle_starved,
                   ring_cap=pl.cap)
    return out
