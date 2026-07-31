"""アブレーション 4 種(第78バッチ 2026-07-31)= **対照条件のスイッチだけ**を置く層。

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
}


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
    for key in ("llm_off", "propagation_off", "shuffle_partners"):
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
    return out


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


def any_on(sim) -> bool:
    c = cfg_of(sim)
    return bool(c["llm_off"] or c["propagation_off"] or c["shuffle_partners"]
                or c["cognitive_tier"] != "full")


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
    return {"llm_off": bool(c["llm_off"]),
            "propagation_off": bool(c["propagation_off"]),
            "cognitive_tier": c["cognitive_tier"],
            "shuffle_partners": bool(c["shuffle_partners"]),
            "tier_effective": tier_effectiveness(sim)}


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
