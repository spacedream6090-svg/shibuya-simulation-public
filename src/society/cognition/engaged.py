"""engaged モード = AUTOPILOT / ENGAGED の 2 状態機械(第87バッチ 2026-08-03・**既定 OFF**)。

正典
----
- docs/plans/source/design-discussion-20260802.md **§8**(突入 5 条件 / 脱出 4 条件 /
  補助規則 3 つ / 初期パラメータ。概念決定済み・実装詳細は委任)
- docs/plans/dayplan-engaged-plan.md 第87 行 + §4(第86 day_plan の実装記録。
  `plan_exception` が fire キューに載る所まで完成済み)

何を解く問題か
--------------
第81(fire)が答えたのは「**いつ**考えるか」で、答えは **点**(発火時刻の全順序キュー)
だった。§8 が要求するのはその次の問い、「考え始めたあと**いつまで**考えるか」である。

    AUTOPILOT … スクリプト実行。LLM 呼び出しゼロ。
    ENGAGED   … 会話 / 再計画 / 内省のいずれかの**エピソード**(持続する区間)。

「討論中は長時間考え、日常ではほぼ考えない」という人間の時間構造は、発火率を上げ下げ
しても出ない(発火が独立事象のままだと考える時間が指数分布に潰れる)。**区間**として
モデル化して初めて出る。本 module は第81 の点発火の**上に**区間を載せる層で、

  - 突入 = 既存の発火(salience / internal / social / 予定思考)を**エピソード開始へ昇格**する
  - 持続 = エピソードが生きている間、認知イベントを「今」へ繰り上げ続ける(`pre_tick`)
  - 脱出 = 4 条件のいずれか

だけを行う。**新しい LLM 呼び出しサイトは 1 つも作らない**(§6 の予算整合)。呼数の増減は
「エピソード構造が既存の発火をどう束ねるか」の帰結としてのみ起きる。

文献的な接地(実装前リサーチ 2026-08-03)
----------------------------------------
**(a) 会話の終わり方**。会話は「話題が尽きた」瞬間に終わるのではなく、**終結の交渉**という
明示的な手続きで終わる(Schegloff & Sacks 1973, *Opening up closings*, Semiotica
8(4):289-327)。構造は 話題の締め → 先行終結(pre-closing: 「じゃあ」「そろそろ」など
**話題を進めない**内容空虚なトークン対)→ 終結交換(terminal exchange:「またね」-「またね」)
で、実際に会話を閉じるのは最後の隣接対である。要点は **pre-closing は決定ではなく申し出**で
あること: 相手はそこで新しい話題を持ち出して会話を**開き直せる**。終結は双方が相互に
方向づけて初めて成立する共同達成である。本実装が「**双方**が closing move を出したときだけ
解消(resolved)」としたのはこの非対称性が根拠で、ISO 24617-2 の Social Obligations
Management 次元が initialGoodbye / returnGoodbye を**対で**定義しているのも同じ構造。
SWBD-DAMSL(Jurafsky, Shriberg & Biasca 1997)は `fc`=Conventional-closing を与え、
注記で「**fc の系列が始まったら実際に閉じるまで全て fc**」と定めている = 終結は点ではなく
**区間**として扱うのが注釈の標準。SwDA 全体で `fc` は約 1%(2,486 発話)しかない。

計算機側の実務は一貫して「**モデルに構造化フラグを吐かせる + 外側で硬い上限を切る**」の
二層である。Generative Agents(Park et al. 2023, UIST '23, arXiv:2304.03442)は
`iterative_convo_v1.txt` で「この発話で会話は終わったか」を **JSON boolean** で要求し、
ドライバ `converse.py` は `for i in range(8): ... if end: break` と **8 ラウンドの無条件上限**で
囲う。AutoGen は max_turns / max_consecutive_auto_reply / is_termination_msg /
human_input_mode の 4 系統を持ち、「終了条件が立っても即座に終わるとは限らない」と明記する。
= **フラグは助言・上限が権威**。分類器ベースの終了検出は注釈研究側の話で、制御ループ側では
使われていない(端点検出 end-of-turn は別問題なので混同しない)。

既知の失敗様式も文献に名前がある。CAMEL(Li et al. 2023, NeurIPS)は
"infinite loop of messages … repeatedly thanking each other or saying goodbye without
progressing"(**丁寧さの無限ループ**)を主要な課題として挙げ、対策は外側の 40 メッセージ
上限である。"Too Polite to Disagree"(SIGDIAL 2026)は追従がターンを追うごとに**増幅**する
ことを示し、Cohesive Conversations(arXiv:2407.09897)は Generative Agents 系の
反復・幻覚が**シミュレーション時間とともに悪化**することを示す。ターン上限(§8 脱出 3)は
飾りではなくこの失敗様式に対する唯一の効く歯止めである。

    ・突入の判定 = S(驚きスコア)と閾値。**LLM 呼ゼロ**(25万人では判定コストが成立しない
      という §8 の指摘。Generative Agents は反応判定自体を LLM に問い合わせる)。
    ・終結の宣言 = 既存 reply スキーマに `end` 欄を 1 つ足す(**新設**。調査の結果、現行の
      `deliberate.parse_action` に終了意思を表す欄は存在しなかった)。**双方の宣言**が要る。
    ・上限 = `turn_cap`(既定 12。Generative Agents の 8 と同じ役割の権威側)。

**(b) ヒステリシス(二閾値)・最短滞在・不応期**。同一閾値で入切りすると入力が境界で
揺れたとき状態が発振する。二閾値(θ_in > θ_out)で塞ぐのは Schmitt トリガの標準意味論
(帯の中では**状態を保持**する)。行動アービトレーションでも同じ形が使われ、
Gaudl & Bryson 2018(*The extended ramp model*, Cognitive Systems Research 50:1-15)は
軽量認知アーキテクチャの目標調停に latching(ヒステリシス)を明示的に組み込む
(本実装に最も近い先例)。正当化は「発振を止める」より強い言い方ができて、
**切替にコストがあるとき履歴依存が合理的方策になる**(McFarland & Sibly 1975 の
behavioural final common path 以来の系譜。Houston らの cross-inhibition)。会話への
出入りはまさに切替コストが高い。

★リサーチで判明した**設計の穴と補修**: 閾値ギャップだけでは「駆動量がなめらかに動いて
正当に境界を往復する場合」のばたつきは止まらない。O'Brien & Arkin 2020
(*Adaptive Behavior* 28(5))はこれを behavior dithering と呼び("an agent might
continuously leave and return to a charger")、**最短実行時間(最短滞在)**を活性化関数に
添えることで解いている。そこで本実装は §8 の二閾値に加えて `min_stay_min`(最短滞在)を
持つ。Δt=10 分の現行では 1 tick に埋もれるが、Δt を細かくしたときに効く保険である。

比率の値については**正直に書いておく**: θ_out/θ_in の標準値は神経・動物行動・ABM の
どの文献にも無い。唯一の広く引かれる推奨は Canny 1986(*IEEE TPAMI* 8(6):679-698)の
ヒステリシス閾値で high:low = 2:1〜3:1(= θ_out/θ_in ≈ 0.33〜0.5)。§8 の初期値 0.5 は
その帯の端にあたる。**較正対象**として宣言し掃引する。

不応期(§8 補助規則 1)は積分発火モデルの絶対不応期(発火直後は入力によらず再発火しない
硬いロックアウト)に対応する。**馴化**(Stanley 1976, *Nature* 261:146-148:
τ·dy/dt = α[y₀ − y] − S。刺激特異的で指数回復)とは別物で、後者は第82 の g 更新
(慣れ ē)がすでに実装している。本 module の不応期はエピソード単位の**閾値倍率**であって
g を 1 バイトも書き換えない(階層が違う)。文献の助言に従えば「同じ相手に再突入しない」は
相手特異的な馴化で表すのが筋だが、本バッチではトリガ種別ごとの倍率までに留める
(相手特異的な馴化は g 更新の側の拡張=別バッチ)。

    θ_out = theta_out_ratio × θ_in  (既定 0.5 = §8 の初期値・Canny 帯の端)

R1 ドクトリン
-------------
- 既定 `cognition.engaged.enabled: false` では本 module は **1 度も呼ばれない**
  (`enabled()` が False → 呼び出し側の全 seam が従来経路)= L1/L2/L3・乱数・LLM 呼数不変。
- **fire ON が前提**(watch / g_update と同じ流儀)。`cognition.fire` が OFF なら
  `enabled()` は False = 完全 no-op。
- **乱数 stream を 1 本も引かない**(全段が決定論。高解像度層の選定も `_stable_hash`)。
- ON は **journal 等級・affects_k=true** を正直に宣言する(エピソードが既存発火を束ねる
  ので LLM 呼数が変わる)。**新しい generate() の呼び出しサイトは 0**。
- fingerprint_risk=**possible** を正直に宣言する: ON のとき会話ターンのプロンプトに
  「話を切り上げるなら `end` を書いてよい」の 1 節が増える(= 終結の宣言路そのもの。
  §8 が要求した設計なので隠さない)。ただし機構語(発火・驚き・閾値・エピソード・
  自動操縦・不応期)も実験条件語も因子名も 1 文字も書かない。

正直な限界
----------
1. `theta_out_ratio` / `turn_cap` / `replan_cap` / `refractory_min` は**全て較正対象の
   仮値**である(§8 が明記)。較正目標(4〜8 エピソード/人/日・engaged 滞在が起床時間の
   1〜2 割)への当て込みは縦煙テストの仕事で、本バッチは機構と観測量を用意するだけ。
   **既定値での実測**(mock・seed42・40 体・288 step): エピソード **7.38/人/日**= 目標帯の中、
   engaged 滞在 **起床時間の 30.4%** = 目標の 1.5〜3 倍。主因は会話エピソードの長さ
   (58.4 分)で、直接のレバーは `talk_idle_min`(20→10 で滞在 23.6% / エピソード 7.99)。
   両方を同時に帯へ入れる設定は mock では見つかっていない。θ_in が動けば両方が同時に
   動くので、本質的な較正は実 LLM 診断ラン(8/15-16)の仕事である。
2. mock は `end` を確率的に返すだけなので、**closing move の実際の出方は実 LLM でしか
   測れない**(文献が言う「エージェントは自分から終わらない」が本当に起きるかどうか)。
3. エピソード**間**の長期記憶統合は本バッチの範囲外(§7 の「やらないこと」)。
   プリエンプト時の兆しメモリは既存 `agent.remember` に 1 行書くだけで、新しい記憶機構は
   作らない。
4. 心モデル(1 体 1 モデル束縛)は第88。ここで L1 に出す `model` は **backend 名**まで
   (day_plan.model_id と同じ暫定)。
5. 定型応答 `TEMPLATE` は**単一の固定文**なので、ON のランでは同じ文が発話分布に
   繰り返し現れる(実測 288 step / 40 体で 104 件)。これは世界に見える痕跡であって
   プロンプトの指紋ではないが、語彙エントロピー等の指標を読むときは
   「機構由来の定数」として除外する必要がある(第91 の退行シグナル監視で効いてくる)。
   文面をペルソナ由来にすると LLM 呼が要る(= 定型で流す意味が消える)ので固定にしてある。
"""
from __future__ import annotations

from ..observer.schema import Event
from ..rng import _stable_hash

SCHEMA = 1

# ---- 2 状態(§8)------------------------------------------------------------ #
AUTOPILOT = "autopilot"
ENGAGED = "engaged"

# ---- エピソード種別(§8「会話・再計画・内省のいずれかのエピソード」)-------- #
TALK = "talk"          # 会話
REPLAN = "replan"      # 再計画(朝の計画生成・must 危機の作り直し・欲求臨界の考え直し)
REFLECT = "reflect"    # 内省(夜の振り返り)
KINDS: tuple[str, ...] = (TALK, REPLAN, REFLECT)

# ---- 突入トリガ(§8 の 5 条件。判定の優先順はこの並びではなく _entry の実装順)-- #
T_SOCIAL = "social"            # (2) 社会的直接性(話しかけられた / 同席で会話が起きる)
T_EXCEPTION = "plan_exception"  # (3) 実行不能例外(第86 の must 危機)
T_NEED = "need"                # (4) 欲求臨界(ニーズ閾値の上向き横断)
T_SCHEDULED = "scheduled"      # (5) 予定思考(朝計画=全員 / 夜内省=高解像度層)
T_SALIENCE = "salience"        # (1) S > θ_in
TRIGGERS: tuple[str, ...] = (T_SOCIAL, T_EXCEPTION, T_NEED, T_SCHEDULED, T_SALIENCE)

# ---- 脱出理由(§8 の 4 条件)------------------------------------------------ #
X_RESOLVED = "resolved"    # (1) 解消(双方の別れ挨拶 / 検証を通る計画)
X_DECAY = "decay"          # (2) 減衰(エピソード内顕著性 < θ_out)
X_TURN_CAP = "turn_cap"    # (3) ターン上限(切り上げターンを 1 回挟む)
X_PREEMPT = "preempt"      # (4) プリエンプト(must 期限 / 欲求臨界の割り込み)
EXITS: tuple[str, ...] = (X_RESOLVED, X_DECAY, X_TURN_CAP, X_PREEMPT)

# `CogQueue` へ積む継続の理由名(fire.SOURCES とは別語彙。L1 の cog_fire にだけ出る)。
CONTINUE = "engaged"

# 会話ターンのプロンプトに足す節の冒頭マーカー(mock がこれを見て end 欄を返す)。
PROMPT_MARK = "話を続けるかどうか:"


# --------------------------------------------------------------------------- #
# cfg 正準化(既定 OFF)
# --------------------------------------------------------------------------- #
DEFAULTS = {
    "enabled": False,
    # ---- §8 の初期パラメータ(**いずれも較正対象**)------------------------- #
    "theta_out_ratio": 0.5,   # θ_out = ratio × θ_in。**必ず 1 未満**(ヒステリシス)
    "turn_cap": 12,           # 会話のターン上限(退行ループの保険)
    "replan_cap": 3,          # 再計画の試行上限(= REPLAN/REFLECT の実効ターン上限)
    "refractory_min": 30,     # 不応期[シミュ内分]。同種刺激の閾値を一時的に引き上げる
    "refractory_mult": 2.0,   # 不応期中に θ_in へ掛ける倍率(>1 = 入りにくくする)
    # ---- 補助規則(§8)+ リサーチ由来の補修 -------------------------------- #
    "min_stay_min": 10,       # 最短滞在[分]。閾値ギャップでは止まらない dithering の保険
                              # (O'Brien & Arkin 2020。Δt=10 分では 1 tick に埋もれる)
    "talk_idle_min": 20,      # 会話が生きているとみなす無音の上限[分](超えたら減衰判定へ)
    "familiar_closeness": 1.0,  # これ未満の親密度からの定型接触は 1 ターンのテンプレで流す
    "familiar_contacts": 2,   # relations OFF(closeness が無い)ときの代替=接触回数
    "reflect_frac": 0.05,     # 夜内省をエピソード化する割合(第88 の高解像度層までの暫定)
    "wrapup": True,           # ターン上限で切り上げターンを 1 回挟むか
    "sign_memory": True,      # プリエンプト時に兆しメモリを 1 行書くか
}
_BOOL_KEYS = ("enabled", "wrapup", "sign_memory")
_INT_KEYS = ("turn_cap", "replan_cap", "refractory_min", "min_stay_min",
             "talk_idle_min", "familiar_contacts")
_FLOAT_KEYS = ("theta_out_ratio", "refractory_mult", "familiar_closeness",
               "reflect_frac")


def _to_plain(raw):
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(raw):
            return OmegaConf.to_container(raw, resolve=True)
    except Exception:                              # noqa: BLE001 (omegaconf 不在でも動く)
        pass
    return raw


def build_cfg(raw) -> dict:
    """conf の `cognition.engaged` ブロックを型強制つきで正準化する(既定 OFF)。"""
    raw = dict(_to_plain(raw) or {})
    cfg = dict(DEFAULTS)
    for k, v in raw.items():
        if k not in DEFAULTS:
            continue
        if k in _BOOL_KEYS:
            cfg[k] = bool(v)
        elif k in _INT_KEYS:
            cfg[k] = int(v)
        elif k in _FLOAT_KEYS:
            cfg[k] = float(v)
    # ヒステリシスの成立条件を**構造で**強制する(θ_out < θ_in。§8 の「必ず小さくする」)。
    cfg["theta_out_ratio"] = min(0.999, max(0.0, cfg["theta_out_ratio"]))
    cfg["turn_cap"] = max(1, cfg["turn_cap"])
    cfg["replan_cap"] = max(1, cfg["replan_cap"])
    cfg["refractory_min"] = max(0, cfg["refractory_min"])
    cfg["refractory_mult"] = max(1.0, cfg["refractory_mult"])
    cfg["min_stay_min"] = max(0, cfg["min_stay_min"])
    cfg["talk_idle_min"] = max(0, cfg["talk_idle_min"])
    cfg["reflect_frac"] = min(1.0, max(0.0, cfg["reflect_frac"]))
    return cfg


def enabled(sim) -> bool:
    """engaged モードが実効か。**発火機構(fire)が ON であることが前提**。"""
    cfg = getattr(sim, "engagedcfg", None)
    if not (cfg and cfg["enabled"]):
        return False
    from . import fire as _fire
    return _fire.enabled(sim)


def model_id(sim) -> str:
    """このエピソードを回したモデルの識別子(第88 の 1 体 1 モデル束縛までは backend 名)。"""
    llm = getattr(sim, "llm", None)
    name = getattr(llm, "name", None)
    if not name:
        name = getattr(getattr(llm, "backend", None), "name", None)
    return str(name or "unknown")


# --------------------------------------------------------------------------- #
# 状態(per-agent。checkpoint は agents pickle に自然同梱される)
# --------------------------------------------------------------------------- #
def episode_of(agent) -> dict | None:
    """生きているエピソード(なければ None = AUTOPILOT)。"""
    return getattr(agent, "_engaged", None)


def mode_of(agent) -> str:
    return ENGAGED if episode_of(agent) is not None else AUTOPILOT


def is_engaged(agent) -> bool:
    return episode_of(agent) is not None


def _new_episode(kind: str, trigger: str, step: int, sim_min: int,
                 partner: int | None, mid: str) -> dict:
    return {"kind": kind, "trigger": trigger, "start": int(sim_min),
            "start_step": int(step), "turns": 0, "last_turn_min": int(sim_min),
            "partner": (int(partner) if partner is not None else None),
            "closed_self": False, "closed_other": False, "wrapup": False,
            "model": str(mid)}


# --------------------------------------------------------------------------- #
# 集計(L2 / summary。checkpoint.py が中央管理する state)
# --------------------------------------------------------------------------- #
def _state(sim) -> dict:
    st = getattr(sim, "_engaged_state", None)
    if st is None:
        st = {"schema": SCHEMA, "episodes": 0, "closed": 0, "turns": 0,
              "stay_min": 0, "template": 0, "signs": 0, "handoff_blocked": 0,
              "by_kind": {}, "by_trigger": {}, "by_exit": {}, "by_model": {},
              "stay_hist": {}}
        sim._engaged_state = st
    return st


def _bump(table: dict, key: str, n: int = 1) -> None:
    table[key] = int(table.get(key, 0)) + int(n)


def _model_row(st: dict, mid: str) -> dict:
    row = st["by_model"].get(mid)
    if row is None:
        row = {"episodes": 0, "turns": 0, "stay_min": 0, "by_exit": {}}
        st["by_model"][mid] = row
    return row


def scalars(sim) -> dict | None:
    """L2 全体スカラー 5 列(OFF は None=列なし=L2 不変。day_plan.scalars と同型)。"""
    if not enabled(sim):
        return None
    st = getattr(sim, "_engaged_state", None)
    if st is None:
        return {"engaged_episodes_total": 0.0, "engaged_turns_total": 0.0,
                "engaged_stay_min_total": 0.0, "engaged_turns_mean": 0.0,
                "engaged_template_total": 0.0}
    n = int(st["episodes"])
    return {"engaged_episodes_total": float(st["episodes"]),
            "engaged_turns_total": float(st["turns"]),
            "engaged_stay_min_total": float(st["stay_min"]),
            "engaged_turns_mean": round(st["turns"] / n, 6) if n else 0.0,
            "engaged_template_total": float(st["template"])}


def provenance(sim) -> dict | None:
    """run_manifest の cognition.engaged キー(OFF は None=キー自体を出さない)。

    §8 が要求する観測量(トリガー種別・滞在時間・ターン数・脱出理由・モデル ID)を
    **エピソード単位の内訳**として残す = 思考量の個体差の観測量。
    """
    if not enabled(sim):
        return None
    cfg = sim.engagedcfg
    out: dict = {
        "schema": SCHEMA,
        "theta_out_ratio": cfg["theta_out_ratio"],
        "turn_cap": cfg["turn_cap"],
        "replan_cap": cfg["replan_cap"],
        "refractory_min": cfg["refractory_min"],
        "refractory_mult": cfg["refractory_mult"],
        "min_stay_min": cfg["min_stay_min"],
        "talk_idle_min": cfg["talk_idle_min"],
        "reflect_frac": cfg["reflect_frac"],
        # ★正直な宣言: 較正はまだ(§8 の初期値そのまま)。縦煙テストで詰める。
        "calibration": "provisional(§8 初期値。目標=4〜8 episodes/day/agent・"
                       "engaged 滞在が起床時間の 1〜2 割)",
        "high_res_source": ("conf.reflect_frac(第88 の tier が入るまでの暫定・"
                            "_stable_hash による決定論選定=乱数ゼロ)"),
    }
    st = getattr(sim, "_engaged_state", None)
    if st is None:                                 # ON だが 1 件もエピソードが立っていない
        out.update({"episodes": 0, "turns": 0, "stay_min": 0, "by_kind": {},
                    "by_trigger": {}, "by_exit": {}, "by_model": {}})
        return out
    n = max(1, int(st["episodes"]))
    out.update({
        "episodes": int(st["episodes"]),
        "turns": int(st["turns"]),
        "stay_min": int(st["stay_min"]),
        "turns_mean": round(st["turns"] / n, 4),
        "stay_min_mean": round(st["stay_min"] / n, 4),
        "closed_episodes": int(st["closed"]),
        "template_replies": int(st["template"]),
        "sign_memories": int(st["signs"]),
        "handoff_blocked": int(st["handoff_blocked"]),
        "by_kind": {k: int(v) for k, v in sorted(st["by_kind"].items())},
        "by_trigger": {k: int(v) for k, v in sorted(st["by_trigger"].items())},
        "by_exit": {k: int(v) for k, v in sorted(st["by_exit"].items())},
        "stay_hist": {k: int(v) for k, v in sorted(st["stay_hist"].items())},
        "by_model": {},
    })
    for mid in sorted(st["by_model"]):
        row = st["by_model"][mid]
        m = max(1, int(row["episodes"]))
        out["by_model"][mid] = {
            "episodes": int(row["episodes"]), "turns": int(row["turns"]),
            "stay_min": int(row["stay_min"]),
            "turns_mean": round(row["turns"] / m, 4),
            "stay_min_mean": round(row["stay_min"] / m, 4),
            "by_exit": {k: int(v) for k, v in sorted(row["by_exit"].items())}}
    return out


# --------------------------------------------------------------------------- #
# 純粋述語(状態機械の芯。sim を触らない = 単体テストで発振を直に反証できる)
# --------------------------------------------------------------------------- #
def theta_out(theta_in: float, cfg: dict) -> float:
    """θ_out = ratio × θ_in。**必ず θ_in より小さい**(ヒステリシス。§8 脱出 2)。"""
    return float(theta_in) * float(cfg["theta_out_ratio"])


def should_enter(s: float, theta_in: float, cfg: dict,
                 refractory: bool = False) -> bool:
    """突入条件 (1): S > θ_in。不応期中は θ_in を `refractory_mult` 倍して入りにくくする。"""
    thr = float(theta_in) * (float(cfg["refractory_mult"]) if refractory else 1.0)
    return thr > 0.0 and float(s) > thr


def should_exit_decay(s: float, theta_in: float, cfg: dict,
                      dwell_min: int | None = None) -> bool:
    """脱出条件 (2): エピソード内顕著性が θ_out を下回った。

    `dwell_min`(突入からの経過[分])を渡すと **最短滞在** `min_stay_min` の間は
    False を返す(O'Brien & Arkin 2020 の dithering 対策。閾値ギャップだけでは
    「駆動量がなめらかに境界を往復する」場合のばたつきが止まらない)。
    """
    if dwell_min is not None and int(dwell_min) < int(cfg["min_stay_min"]):
        return False
    return float(s) < theta_out(float(theta_in), cfg)


def turn_cap_of(kind: str, cfg: dict) -> int:
    """種別ごとの実効ターン上限。会話は `turn_cap`、思考系は `replan_cap`(§8 の試行上限)。"""
    return int(cfg["turn_cap"]) if kind == TALK else int(cfg["replan_cap"])


def simulate(s_series, theta_in: float, cfg: dict, dt_min: int = 10) -> list[str]:
    """S の系列に対する状態列を返す(**発振の有無を直に検査するための純関数**)。

    会話・返答義務・ターン・プリエンプトを一切含まない「減衰だけの世界」での挙動で、
    ヒステリシスが効いていれば θ_out < S < θ_in の帯では**状態が保持される**
    (= 入切りが交互に起きない)。`dt_min` は 1 tick の分数(最短滞在の判定に使う)。
    """
    out: list[str] = []
    state = AUTOPILOT
    dwell = 0
    for s in s_series:
        if state == AUTOPILOT:
            if should_enter(s, theta_in, cfg):
                state, dwell = ENGAGED, 0
        else:
            if should_exit_decay(s, theta_in, cfg, dwell):
                state = AUTOPILOT
            dwell += int(dt_min)
        out.append(state)
    return out


# --------------------------------------------------------------------------- #
# 高解像度層(第88 の tier が入るまでの暫定。**乱数を引かない**決定論選定)
# --------------------------------------------------------------------------- #
def high_res(agent, cfg: dict) -> bool:
    """夜の内省をエピソード化する層に属するか(§8 突入 5「夜の内省は高解像度層」)。

    第88 で `cognitive_tier` が入ったらそこへ委譲する。それまでは conf の割合指定を
    `_stable_hash` で決定論に割り当てる(専用 stream を作らない = 乱数消費ゼロ)。
    """
    frac = float(cfg["reflect_frac"])
    if frac <= 0.0:
        return False
    if frac >= 1.0:
        return True
    return (_stable_hash(f"engaged_hires/{int(agent.id)}") % 10_000) < int(frac * 10_000)


# --------------------------------------------------------------------------- #
# 親密度(定型接触の判定。relations OFF でも動く二段構え)
# --------------------------------------------------------------------------- #
def familiarity_ok(agent, other_id: int, cfg: dict) -> bool:
    """相手が「定型で流さない」相手か(§8 突入 2 の但し書き)。

    Wave G2(relations)ON なら台帳の closeness、OFF なら接触回数で測る。台帳自体が
    無ければ **初対面** = 定型で流す側。単位が違う 2 つの尺度を同じ 1 本の閾値で扱わない
    ため、conf も 2 本に分けてある。
    """
    rel = getattr(agent, "mem", None)
    rel = rel.relations.get(int(other_id)) if rel is not None else None
    if not rel:
        return False
    if "closeness" in rel:                         # relations ON = 親密度が測れる
        return float(rel["closeness"]) >= float(cfg["familiar_closeness"])
    return int(rel.get("count", 0)) >= int(cfg["familiar_contacts"])


# --------------------------------------------------------------------------- #
# L1(§8 補助規則 3: 全エピソードのログ)
# --------------------------------------------------------------------------- #
def _log(sim, agent, step: int, sim_min: int, kind: str, payload: dict) -> None:
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=int(agent.id),
                         kind=kind, x=agent.x, y=agent.y, payload=payload))


def _start(sim, agent, step: int, sim_min: int, kind: str, trigger: str,
           partner: int | None, s: float, theta_in: float, turns0: int = 0) -> dict:
    ep = _new_episode(kind, trigger, step, sim_min, partner, model_id(sim))
    ep["turns"] = int(turns0)
    agent._engaged = ep
    st = _state(sim)
    st["episodes"] += 1
    st["turns"] += int(turns0)
    _bump(st["by_kind"], kind)
    _bump(st["by_trigger"], trigger)
    row = _model_row(st, ep["model"])
    row["episodes"] += 1
    row["turns"] += int(turns0)
    payload = {"kind": kind, "trigger": trigger, "model": ep["model"],
               "theta_in": round(float(theta_in), 4),
               "theta_out": round(theta_out(theta_in, sim.engagedcfg), 4),
               "s": round(float(s), 4)}
    if partner is not None:
        payload["partner"] = int(partner)
    _log(sim, agent, step, sim_min, "episode_start", payload)
    return ep


def _end(sim, agent, step: int, sim_min: int, reason: str) -> None:
    ep = episode_of(agent)
    if ep is None:
        return
    stay = max(0, int(sim_min) - int(ep["start"]))
    steps = max(0, int(step) - int(ep["start_step"]))
    st = _state(sim)
    st["stay_min"] += stay
    _bump(st["by_exit"], reason)
    _bump(st["stay_hist"], str(stay))
    if reason == X_RESOLVED:
        st["closed"] += 1
    row = _model_row(st, ep["model"])
    row["stay_min"] += stay
    _bump(row["by_exit"], reason)
    agent._engaged = None
    # 不応期(§8 補助規則 1): 脱出後一定時間は**同種刺激**の閾値を引き上げる。
    # 第82 の g 更新(慣れ ē)とは別の階層で、g にも θ の恒常性にも一切触れない。
    cfg = sim.engagedcfg
    if cfg["refractory_min"] > 0:
        agent._engaged_refr = {"until": int(sim_min) + int(cfg["refractory_min"]),
                               "trigger": ep["trigger"]}
    _log(sim, agent, step, sim_min, "episode_end",
         {"kind": ep["kind"], "trigger": ep["trigger"], "exit": reason,
          "turns": int(ep["turns"]), "stay_min": stay, "steps": steps,
          "model": ep["model"], "closed_self": bool(ep["closed_self"]),
          "closed_other": bool(ep["closed_other"])})


def _refractory(agent, trigger: str, sim_min: int) -> bool:
    """いま `trigger` 種の刺激について不応期にあるか。"""
    r = getattr(agent, "_engaged_refr", None)
    if not r:
        return False
    if int(sim_min) >= int(r["until"]):
        return False
    return str(r["trigger"]) == str(trigger)


# --------------------------------------------------------------------------- #
# tick の本体(_phase_drive からの**単一作用点**は pre_tick + update の 2 本)
# --------------------------------------------------------------------------- #
def pre_tick(sim, step: int, sim_min: int) -> None:
    """生きているエピソードの継続を認知イベントキューの「今」へ繰り上げる。

    **`fire.due_events` の直前**に呼ぶ。これが「点の発火 → 区間の持続」の実体で、
    ここで繰り上げた個体は due_events の通常経路(S 計算・ô 更新・plasticity・
    次の周期の再スケジュール)を**そのまま**通る = fire の内部規約を 1 つも迂回しない。
    OFF は完全 no-op。
    """
    if not enabled(sim):
        return
    q = getattr(sim, "cogq", None)
    if q is None:
        return
    for agent in sim.agents:
        if episode_of(agent) is not None:
            q.advance(int(agent.id), int(sim_min), CONTINUE)


def update(sim, step: int, sim_min: int, due: dict, active,
           face_ids: frozenset = frozenset()) -> set:
    """脱出 → 突入 の順に評価し、**この tick に割込み権を持つ個体の id 集合**を返す。

    返した id は `_phase_drive` で `interrupt=True` 扱いになる(= 抽選を飛ばす確定発火。
    対面会話・驚き割込みと同格)。**新しい LLM 呼び出しサイトは作らない**: 呼ぶのは
    従来どおり `_decide`(reply)/`_fire_llm`(発話)/`maybe_reflect`(内省)だけで、
    engaged は「その発火が起きるかどうか」を束ねるだけ。

    ★評価順が脱出 → 突入 なのは、プリエンプト(must 期限・欲求臨界)が**同じ tick で**
      新しい再計画エピソードへ移れるようにするため(§8 脱出 4 の「割り込み」の意味)。
    """
    if not enabled(sim):
        return set()
    cfg = sim.engagedcfg
    from . import fire as _fire
    s_of = getattr(sim, "_fire_s", None) or {}
    ctx_of = getattr(sim, "_fire_ctx", None) or {}
    active_ids = {int(a.id) for a in active}
    plan_due = frozenset(getattr(sim, "_fire_plan_due", ()) or ())

    def _s(aid: int) -> float:
        row = s_of.get(int(aid))
        return float(row[0]) if isinstance(row, tuple) else float(row or 0.0)

    def _theta(agent) -> float:
        return _fire.theta_of(sim, ctx_of.get(int(agent.id), _fire.WALKING), agent)

    # ---- 1) 脱出(§8 の 4 条件)-------------------------------------------- #
    for agent in sim.agents:
        ep = episode_of(agent)
        if ep is None:
            continue
        aid = int(agent.id)
        rec = due.get(aid) or {}
        theta_in = _theta(agent)
        s_now = _s(aid)
        # (4) プリエンプト: must ブロックの期限 / 欲求臨界の割り込み。
        #     再計画エピソード自身は「割り込まれる側」ではない(すでに同じ用件の中にいる)。
        preempt = (ep["kind"] != REPLAN
                   and (int(getattr(agent, "_plan_exception", -1)) == int(sim_min)
                        or str(rec.get("reason", "")) == _fire.INTERNAL))
        if preempt:
            _note_sign(sim, agent, ep, step, sim_min)
            _end(sim, agent, step, sim_min, X_PREEMPT)
            continue
        # (1) 解消: 会話は**双方**の別れ挨拶、思考系は解決の明示(day_plan の検証通過)。
        if ep["closed_self"] and ep["closed_other"]:
            _end(sim, agent, step, sim_min, X_RESOLVED)
            continue
        if ep.get("resolved"):
            _end(sim, agent, step, sim_min, X_RESOLVED)
            continue
        # (3) ターン上限: 会話だけは切り上げターンを 1 回挟んでから終える(§8)。
        cap = turn_cap_of(ep["kind"], cfg)
        if int(ep["turns"]) >= cap:
            if ep["kind"] == TALK and cfg["wrapup"] and not ep["wrapup"]:
                ep["wrapup"] = True                # この 1 ターンだけ延命(節が切り上げを促す)
            else:                                   # 切り上げ済み or 思考系 = ここで終わる
                _end(sim, agent, step, sim_min, X_TURN_CAP)
                continue
        # (2) 減衰: ただし「返答義務がある間」と「生きている会話の最中」は抜けない。
        if getattr(agent, "_reply_to", None) is not None:
            continue
        if ep["kind"] == TALK and int(ep["turns"]) >= 1 \
                and (int(sim_min) - int(ep["last_turn_min"])) <= int(cfg["talk_idle_min"]):
            continue
        if should_exit_decay(s_now, theta_in, cfg,
                             int(sim_min) - int(ep["start"])):
            _end(sim, agent, step, sim_min, X_DECAY)

    # ---- 2) 突入(§8 の 5 条件。優先順=今この場の相手 > 例外 > 欲求 > 予定 > 驚き)-- #
    for agent in sim.agents:
        aid = int(agent.id)
        if episode_of(agent) is not None or aid not in active_ids:
            continue
        rec = due.get(aid) or {}
        reason = str(rec.get("reason", ""))
        theta_in = _theta(agent)
        s_now = _s(aid)
        # (2) 社会的直接性: 話しかけられた = **S 計算をバイパスして即突入**。
        #     ただし関係の薄い相手からの定型接触は突入せず 1 ターンのテンプレで流す
        #     (= reply_mode が "template" を返す = LLM 呼ゼロ)。
        rt = getattr(agent, "_reply_to", None)
        if rt is not None:
            if familiarity_ok(agent, rt[0], cfg):
                _start(sim, agent, step, sim_min, TALK, T_SOCIAL, rt[0],
                       s_now, theta_in)
            continue
        # (2') 社会的直接性の**発話する側**。同席していて会話クールダウン外の個体が
        #      この tick に認知イベントを迎える = 既存の「対面会話は抽選なしの確定発火」
        #      (_phase_drive の face)がまさに起きる場面で、それは「相手の方を向いた」
        #      = ENGAGED である。ここを突入にしないと、話者が AUTOPILOT のまま喋って
        #      補助規則 2(両者 ENGAGED が成立条件)が返答権を全部止めてしまい、
        #      **世界から会話が消える**(実測: 会話エピソードが 110 件中 7 件まで枯れた)。
        #      §8 が「片方が急いでいれば挨拶で終わる」と書くとき、急いでいるのは
        #      **話しかけられた側**であって、話しかける側は定義上 engaged である。
        if reason and aid in face_ids:
            _start(sim, agent, step, sim_min, TALK, T_SOCIAL, None, s_now, theta_in)
            continue
        # (3) 実行不能例外(第86 の must 危機)→ 再計画エピソード。
        if int(getattr(agent, "_plan_exception", -1)) == int(sim_min):
            _start(sim, agent, step, sim_min, REPLAN, T_EXCEPTION, None,
                   s_now, theta_in)
            continue
        # (4) 欲求臨界(ニーズ閾値の上向き横断 = fire の internal 割込み)。
        if reason == _fire.INTERNAL and not _refractory(agent, T_NEED, sim_min):
            _start(sim, agent, step, sim_min, REPLAN, T_NEED, None, s_now, theta_in)
            continue
        # (5) 予定思考: 朝の計画は全員、夜の内省は高解像度層のみ。
        #     どちらも**この tick に既存機構が既に 1 回考えている**ので turns=1 から始める
        #     (朝計画は _phase_planning が _phase_drive より前、夜内省は step 末)。
        if aid in plan_due:
            _start(sim, agent, step, sim_min, REPLAN, T_SCHEDULED, None,
                   s_now, theta_in, turns0=1)
            continue
        if int(getattr(agent, "reflect_step", -1)) == int(step) and high_res(agent, cfg):
            _start(sim, agent, step, sim_min, REFLECT, T_SCHEDULED, None,
                   s_now, theta_in, turns0=1)
            continue
        # (1) 驚き: S > θ_in(不応期中は θ_in を引き上げる)。同席していれば会話の形になる。
        if reason == _fire.SALIENCE and should_enter(
                s_now, theta_in, cfg, _refractory(agent, T_SALIENCE, sim_min)):
            kind = TALK if ctx_of.get(aid) == _fire.TALKING else REPLAN
            _start(sim, agent, step, sim_min, kind, T_SALIENCE, None,
                   s_now, theta_in)

    # ---- 3) 割込み権: 生きているエピソードを持つ在場個体 --------------------- #
    return {int(a.id) for a in active if episode_of(a) is not None}


def _note_sign(sim, agent, ep: dict, step: int, sim_min: int) -> None:
    """プリエンプトで中断した内容を**1 行だけ**兆しメモリに書く(§8 脱出 4)。

    「あとで考えよう」の実装。**新しい記憶機構は作らない**: 既存の `agent.remember`
    (= MemoryStore.observe)に kind="sign" で 1 行積むだけで、後日の想起は既存の
    retrieve/query がそのまま拾う(= 再燃フック)。
    """
    if not sim.engagedcfg["sign_memory"]:
        return
    if ep["kind"] == TALK and ep["partner"] is not None:
        other = sim.agent_by_id.get(int(ep["partner"]))
        who = other.name if other is not None else "さっきの人"
        text = f"{who}と話していたことは、あとで考えよう。"
    elif ep["kind"] == REFLECT:
        text = "振り返りかけたことは、あとで考えよう。"
    else:
        text = "さっき考えかけたことは、あとで考えよう。"
    agent.remember(text, kind="sign")
    _state(sim)["signs"] += 1
    _log(sim, agent, step, sim_min, "sign_memory",
         {"kind": ep["kind"], "turns": int(ep["turns"]),
          "stay_min": max(0, int(sim_min) - int(ep["start"]))})


# --------------------------------------------------------------------------- #
# 会話まわりの seam(scheduler から呼ばれる 4 本。OFF は全部 no-op / 恒等)
# --------------------------------------------------------------------------- #
TEMPLATE = "ええ、どうも。"      # 定型応答(**LLM を呼ばない**中立な 1 ターン)


def reply_mode(sim, agent) -> str:
    """話しかけられた個体の返答様式。"engage"(従来の LLM 返答)/ "template"(定型)。

    §8 突入 2 の但し書き「関係の薄い定型接触は 1 ターンのテンプレ応答で流すことを許す」。
    OFF では常に "engage" = 従来経路(バイト一致)。
    """
    if not enabled(sim):
        return "engage"
    rt = getattr(agent, "_reply_to", None)
    if rt is None:
        return "engage"
    return "engage" if familiarity_ok(agent, rt[0], sim.engagedcfg) else "template"


def template_reply(sim, agent, step: int, sim_min: int) -> dict:
    """定型応答の行動 dict(LLM 呼ゼロ・乱数ゼロ・決定論)。"""
    _state(sim)["template"] += 1
    _log(sim, agent, step, sim_min, "engaged_template", {"len": len(TEMPLATE)})
    return {"type": "speak", "text": TEMPLATE, "use_items": []}


def handoff_ok(sim, speaker, partner) -> bool:
    """返答権(`_reply_to`)を渡してよいか。**会話は両者 ENGAGED が成立条件**(§8 補助規則 2)。

    話者が AUTOPILOT(通り過ぎざまの独り言・定型応答で流した側)なら渡さない
    = 相手は返さない = 挨拶で終わる。OFF は常に True = 従来経路(バイト一致)。
    """
    if not enabled(sim):
        return True
    ep = episode_of(speaker)
    if ep is None or ep["kind"] != TALK:
        _state(sim)["handoff_blocked"] += 1
        return False
    return True


def note_turn(sim, agent, step: int, sim_min: int) -> None:
    """LLM を 1 回通した = エピソードのターンが 1 進んだ(**唯一の計上点**)。

    `_llm_speak` の generate 直後に置く(発話・返答・投稿・DM・独り言のどれでも
    「1 回の思考」なので同じ 1 ターン)。エピソードが無ければ完全 no-op。
    """
    if not enabled(sim):
        return
    ep = episode_of(agent)
    if ep is None:
        return
    ep["turns"] = int(ep["turns"]) + 1
    ep["last_turn_min"] = int(sim_min)
    st = _state(sim)
    st["turns"] += 1
    _model_row(st, ep["model"])["turns"] += 1


def note_closing(sim, speaker, partner, step: int, sim_min: int) -> None:
    """closing move(別れの挨拶)を検出した。**双方が出したときだけ**解消になる。

    Schegloff & Sacks 1973 の終結交換(terminal exchange)に対応する非対称性で、
    片方の pre-closing だけでは会話は終わらない(相手はまだ話題を出せる)。
    """
    if not enabled(sim):
        return
    ep = episode_of(speaker)
    if ep is not None:
        ep["closed_self"] = True
    if partner is not None:
        pep = episode_of(partner)
        if pep is not None:
            pep["closed_other"] = True
    _log(sim, speaker, step, sim_min, "episode_closing",
         {"partner": (int(partner.id) if partner is not None else None),
          "turns": int(ep["turns"]) if ep is not None else 0})


def note_resolved(sim, agent) -> None:
    """思考系エピソードの解消(§8 脱出 1「検証を通る計画が生成されたとき」)。

    day_plan が LLM 由来の計画を検証まで通したときに呼ばれる。次の tick の `update` が
    これを見て resolved で閉じる(その tick のうちに閉じないのは、脱出の評価点を
    `update` 1 箇所に閉じておくため)。
    """
    if not enabled(sim):
        return
    ep = episode_of(agent)
    if ep is not None and ep["kind"] in (REPLAN, REFLECT):
        ep["resolved"] = True


# --------------------------------------------------------------------------- #
# プロンプト節(会話ターンだけ・中立・no-fingerprint)
# --------------------------------------------------------------------------- #
def prompt_section(sim, agent, trigger: str) -> str | None:
    """会話ターンのプロンプトへ足す節(ON かつ会話エピソード中のみ。OFF は None)。

    ★ここが本 module で唯一プロンプトに触れる場所(fingerprint_risk=possible の実体)。
      書くのは「話を切り上げるつもりなら `end` と書いてよい」という**終結の宣言路**だけで、
      機構語(発火・驚き・閾値・エピソード・不応期・自動操縦)も実験条件語も因子名も
      1 文字も書かない。
    ★切り上げターン(§8 脱出 3)では、同じ節が「そろそろ締めくくる」1 行に変わる。
      上限に達したことを**数値で**は伝えない(何ターン目かは本人には見えない)。
    """
    if not enabled(sim):
        return None
    if trigger not in ("social", "reply"):
        return None
    ep = episode_of(agent)
    if ep is None or ep["kind"] != TALK:
        return None
    if ep["wrapup"]:
        return (f"{PROMPT_MARK} そろそろこの話は締めくくるつもりで、"
                "別れぎわのひとことにしてください。"
                '返事に "end": true を添えてください。')
    return (f"{PROMPT_MARK} まだ話を続けたいならそのまま返事をしてください。"
            "もう切り上げたいなら、別れぎわのひとことを言って、"
            '返事に "end": true を添えてください。')
