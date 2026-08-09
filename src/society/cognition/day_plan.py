"""day_plan v1 = 朝の構造化計画スキーマ + ルール実行系(第86バッチ 2026-08-03・**既定 OFF**)。

正典
----
- docs/plans/source/design-discussion-20260802.md **§6**(決定と実行の分離)・**§7**(朝の計画
  スキーマ day_plan v1 の概念)・§8(engaged 突入条件(3)= 実行不能例外)
- docs/plans/dayplan-engaged-plan.md 第86行

何を解く問題か
--------------
現行の朝計画(planning.py + plan_schema.py)は「時間帯バンド(朝/昼/…)× 活動語 × 自由文の
場所名」までしか構造を持たない。25万体ぶんをパースし、ルールエンジンが**無加工で実行**でき、
かつ割り込み処理(会話で時間が食われた)を**LLM を呼ばずに**裁ける表現になっていない。
本 module はその表現(スキーマ)と、検証→修復→フォールバックの3段、そして実行系を足す。

設計の骨格と文献的接地(2026-08-03 の事前リサーチで採った根拠)
--------------------------------------------------------------
(A) **活動ベース需要モデルの活動レコード**をそのまま踏襲する。ALBATROSS(Arentze &
    Timmermans 2004, *A learning-based transportation oriented simulation system for
    activity scheduling*, Transportation Research B 38(7):613-633)は活動ごとに〈種別・
    開始時刻・継続・場所・**同伴者**・交通手段・トリップチェーン〉を予測し、ActivitySim の
    `tours` 表も `tour_type / start / end / duration / destination / number_of_participants /
    composition` を持つ。本スキーマの `act`/`start`/`end`/`place`/`with` はこの対応そのまま。
    活動種別の列挙も ActivitySim の非義務トリップ目的
    (`escort, shopping, othmaint, othdiscr, eatout, social` + 義務の `work, school`)へ
    1 対 1 で写る(escort=escort / shop=shopping / personal=othmaint / leisure・park=othdiscr /
    meal=eatout / visit・meetup=social / work=work / study=school)。
    ★ Doherty らは「スケジューリングを説明するのは**種別ではなく属性**(最小継続・可能時間窓・
      場所候補数・共同性)だ」と明言している。本スキーマが `priority`/`flex`/`start-end` を
      種別と**別の軸**として持つのはその指摘に沿う設計である。
(B) **固定アンカー先置き → 空きを埋める**(skeleton-and-fill)。ALBATROSS は「義務活動を先に
    配置し裁量活動を後から入れる。時刻とチェーンの決定が場所の決定に優先し、場所が交通手段に
    優先する」。Doherty, Miller, Axhausen & Gärling(*A Conceptual Model of the Weekly
    Household Activity-Travel Scheduling Process*、CHASE 調査 41世帯66名)は「routine weekly
    activity skeleton」から出発し、各項目を Routinized / Pre-planned / Day-of / Impulsive へ
    分類する(骨格活動の平均継続 208 分に対し衝動的活動 72 分)。Habib & Miller 2006(TRR
    1985)の言い方では骨格は「柔軟な活動が周りに組織される時間の杭」。
    **空き時間をブロックにしない**(§7 の第三の設計判断)のはこの構造の系で、未計画帯は
    既存の習慣ポリシー(routine)が埋める。本 module の repair も「時刻 → 場所」の優先順を守る。
(C) **修復は決定論的な操作子の集合**にする。Doherty の Modify & Conflict Resolver が挙げる
    操作子は「時間内で動かす / 最小継続まで縮める / 場所か手段を変えて移動時間を減らす」で、
    それでも足りないときだけ削除へ落ちる。MATSim の中核 replanning も
    `TimeAllocationMutator`(既定 mutationRange=1800s=±30分)・`ReRoute`・`SubtourModeChoice`
    の 3 つだけで、**場所変更も挿入/削除も中核には無い**(Horni, Nagel & Axhausen 2016)。
    Roorda & Miller 2005 の実測では修正の内訳は **時刻 25.3% / 場所 2.0% / 同伴者 0.9% /
    種別 0.5%** = 時刻ずらしが 1 桁多い。よって本 module も **ずらし優先・削除は最後**にする:
      `round`(時刻の丸め=retime)→ `slide`(ずらし=retime)→ `truncate`(最小継続まで短縮)
      → `substitute`(場所カテゴリの差し替え=change location)→ `drop`(削除)。
      + `clip`(件数上限)。**新しいヒューリスティックは足さない**。
    ずらし幅は「必要な最小量」だけ動かす = Auld & Mohammadian 2016(*An optimization approach
    to resolve activity scheduling conflicts in ADAPTS*, Transportation 43(6))の
    **最小摂動**(衝突活動の時刻・継続の変更量を最小化する)目的関数と同じ向き。
    ★正直な限界: Auld, Mohammadian & Doherty 2008(TRR 2054:12-19)は「**種別ベースの優先度
      仮定は実際には成り立たない**」ことを実証しており、削除は本来「今この予定を新規に入れる
      としたら勝つか」の再評価であるべきである。本 module の `clip`/`drop` は
      **LLM が予定ごとに宣言した priority**(種別固定表ではなく個別宣言)を使うことで
      その批判に部分的にしか答えていない。比較評価版は将来の課題として残す。
(D) **LLM 構造化出力は「検証 → 決定論修復 → フォールバック」で受ける**(再試行ゼロ)。
    - 制約デコーディング(Outlines: Willard & Louf 2023, arXiv:2307.09702 / XGrammar:
      arXiv:2411.15100 / lm-format-enforcer)は**文法適合**を保証するが、移動時間・営業時間・
      場所の実在といった**意味の妥当性**は保証しない。よって物理検証と修復は必ず要る。
    - 小型モデルの非適合率は無視できない: SchemaBench(Lu et al. 2025, arXiv:2502.18878)は
      制約デコーディング無しで **Qwen-2.5 7B が validation error 18%、LLaMA-3.2 3B が
      parse error 約23%、MiniCPM-3 4B が 36%** と報告する。fleet の 3B〜14B を前提にすると
      **フォールバック経路は例外処理ではなく常用経路**である(だから件数をモデル別に数える)。
    - 再問い合わせ(reask)は Instructor / LangChain / Guardrails のいずれも既定 **1 回**だが、
      本バッチでは**採らない**: 25万体×毎朝で 1 回の reask は最悪で呼数 2 倍になり、
      「実効 4〜8 呼/人/日」の予算(§6)が崩れる。制約デコーディング自体も
      3.6〜8.2 倍のレイテンシ増が報告されている(arXiv:2605.02363)。修復+フォールバックで
      **世界を止めない**(§7 の第四の設計判断)方が本シムでは安い均衡である。
    - **自由文欄を先頭に置く**: 構造化出力は自己回帰なのでスキーマのキー順が生成順になる
      (OpenAI の公式ドキュメントが明示)。Instructor の実測では理由欄を先頭に置くだけで
      GSM8K が 33% → 92% へ動いた。厳格な構造化が推論を劣化させるという報告(Tam et al.
      2024 *Let Me Speak Freely?*, arXiv:2408.02442, EMNLP Industry)への実務的な落とし所も
      これで、dottxt の反論(条件を揃えると差は消える)や JSONSchemaBench(arXiv:2501.10868、
      むしろ改善)を踏まえても「理由を先に書かせる」は無害かつ有効。よって本スキーマは
      **各ブロックの先頭を `reason`(自由文1行)**にし、計画全体でも `mood` / `carry` を
      blocks より先に置く。自由文は **mood / carry / reason / note** に限る(§7)。
    - ActivitySim の `failed` フラグ / `drop_and_cleanup` に倣い、「組めなかった」を
      **黙って落とさず数える**(plan_block_drop / plan_repair / plan_fallback)。

R1 ドクトリン(既定 OFF=1 バイトも動かさない)
--------------------------------------------
- `planning.day_plan.enabled: false` では本 module の全関数が即 return する。プロンプトは
  1 バイトも変わらず(schema 節を足さない)、新イベント 0 件・L2 に列なし・`agent._dayplan` も
  生えない・**乱数 stream を 1 本も引かない**。
- ON でも **generate() を 1 本も足さない**(朝 1 呼のまま・再試行なし)。修復・フォールバック・
  場所解決・割り込み処理は全て決定論(乱数ゼロ・LLM ゼロ)。
- `planning.day_plan.use_contingency: false`(第93バッチ IF-A)では `plan["cont"]` を
  **1 度も読まない** = 第86 の挙動と完全同一(plan_cont_fire 0 件)。ON でも評価・適用は
  全て決定論(乱数ゼロ・LLM ゼロ)で、プロンプトは 1 バイトも変わらない
  (if_then の書式は day_plan ON なら本トグルと無関係に `schema_prompt` へ載る)。
- `must` が脅かされたときだけ「再計画」が起きる。本バッチの再計画は**ルール内の作り直し**
  (低優先の削除+ずらし+plan_version++)であり、LLM 呼を足さない。第81 の認知イベントキューが
  ON のときだけ、内部発火(INTERNAL)として **plan_exception** を前倒し登録する
  (= 設計 §8 突入条件(3)の先行実装。engaged エピソード本体は第87)。
- W4-F(計画遵守観測の強化)は **記録だけ**を足した。`plan_block_start` の `node`(実行時の
  現在ノード)/ `plan_block_*` の `block`(台帳の添字)はどちらも **1 箇所も読まれない**
  (`_bi` の戻り値で分岐する行はゼロ)。したがって ON ランでも行き先・乱数・LLM 呼数・
  プロンプトは 1 バイトも変わらず、OFF ランはイベント自体が 0 件なので当然無風。
- no-fingerprint: プロンプトは中立(機構語・実験条件語・因子名を 1 つも出さない)。本 module は
  性格特性も k も読まず、物理量(時刻・座標・訪問回数・営業時刻・所持金)だけを見る。

正直な限界
----------
- `model_id` は `sim.llm.name`(バックエンド名)であって 1 体 1 モデルの束縛ではない。
  agent→model_id の誕生時固定は**第88バッチ**の仕事で、fleet 配下では実際に応答した子モデルは
  ここでは判らない(集計は「バックエンド別」であって「子モデル別」ではない)。
- 物理検証の移動時間は**直線距離 ÷ 徒歩速度**であって経路探索の実所要ではない(下限見積り)。
  下限を満たさない並びだけを不能と判定する = 偽陽性(実際は不能なのに通す)側に倒してある。
- 場所解決は計画時に 1 回(習慣・距離・営業状態)行い、実行時に閉店していれば**その場で解き
  直す**。日中に移動した結果として「もっと近い同カテゴリの店」ができても解き直さない。
- `start`/`end` は独立した 2 欄である。ActivitySim は (start, end) を **単一の離散選択肢**
  (`tdd` = 190 通りの表)として扱っており、そちらのほうが不整合が原理的に出ない。本スキーマは
  LLM に自然文の時刻を書かせる都合で 2 欄に分け、`start < end` は修復側で担保している。
- 交通手段(mode)は計画に含めない。既存 routine の `_choose_mode` / `_augment_ride` が
  距離と所持金から決めるので、MATSim で言う `ChangeMode` 相当の修復操作子は本 module に無い。
- pool ローテーション(world/pool.py)の dehydrate は動的属性を運ばないため、退場→再入場した
  個体の計画は失われる(翌朝また立つ)。既存の `_plan_prev_day` と同じ性質。
"""
from __future__ import annotations

import math

from ..observer.schema import Event
from . import plan_boundary as _bnd
from .deliberate import _loads_lenient

SCHEMA = 1

# --------------------------------------------------------------------------- #
# 列挙型(棚卸しの結果)
# --------------------------------------------------------------------------- #
# 活動種別 act: 既存の計画語彙をそのまま採る。
#   plan_schema._DEFAULT_VOCAB(11 語)+ routine._WHAT_CAT が既に持つ "meetup" = 12 語。
#   ★新語を足さない = 既存の what→cat3 / what→POI カテゴリ写像が 1 行も改変なしで効く。
ACTS: tuple[str, ...] = (
    "work", "study", "meal", "shop", "leisure", "park", "walk", "home",
    "visit", "personal", "escort", "meetup",
)

# 場所カテゴリ place: **具体店名ではない**。解決可能なカテゴリだけを列挙する。
#   実地図(scripts/build_map.poi_category)が生む POI カテゴリ
#     food / nightlife / shop / service / office / education(school)/ hotel /
#     attraction / leisure / landmark / cinema / hall
#   + ルール側だけで解ける 3 つ: home(自宅)/ work(職場・バイト先)/ street(戸外・特定しない)
PLACE_CATS: tuple[str, ...] = (
    "home", "work", "street",
    "food", "nightlife", "shop", "service", "office", "education",
    "hotel", "attraction", "leisure", "landmark", "cinema", "hall",
)

# 目的 purpose: 「欲求/目標体系への紐づけ」。
#   ★正直な注記: 原文書 §7 の「7欲求」に相当する 7 次元は本リポジトリに存在しない
#     (factors 層に 5 次元あるが、cognition 層はその名前を知ってはならない = no-fingerprint)。
#     そこで **cognition から見える体系**へ紐づける:
#       (a) 全メンバが活動3分類(mandatory/maintenance/discretionary)へ決定論写像される
#           = 既存 plan_schema/routine の cat 体系と同一の箱に入る。
#       (b) long_goal は inner_life の長期目標(agent.life_goal)への紐づけ。
PURPOSES: tuple[str, ...] = (
    "livelihood",   # 生計(仕事・稼ぎ)
    "learning",     # 学び
    "sustenance",   # 食事
    "errand",       # 用事・買い物
    "care",         # 世話・送り迎え・家のこと
    "health",       # 体調・運動
    "rest",         # 休息
    "company",      # 人と過ごす
    "enjoyment",    # 楽しみ
    "long_goal",    # 長期の目標(inner_life の life_goal)
)

PRIORITIES: tuple[str, ...] = ("must", "should", "could")
FLEXES: tuple[str, ...] = ("fixed", "slideable", "droppable")

# contingency(条件分岐)。条件も対処も既存機構に対応する語だけを置く。
CONDS: tuple[str, ...] = (
    "rain",       # 天気(weather)
    "crowded",    # 混雑(envfeedback / crowd)
    "closed",     # 閉店(commerce.filter_open)
    "tired",      # 疲れ(health)
    "no_money",   # 所持金不足(economy)
    "late",       # 遅れ(本 module の slide)
    "invited",    # 誘われた(joint / relations)
)
THENS: tuple[str, ...] = ("skip", "postpone", "go_home", "swap_indoor", "shorten")

# then="swap_indoor" の差し替え表(第93 IF-A)。**屋外カテゴリだけ**を屋内へ寄せる。
#   street → shop        … 戸外の用足しを店の中で済ませる
#   leisure → cinema     … 屋外の遊び(公園・広場)を屋内の娯楽へ
#   landmark / attraction → hall … 屋外の見どころを屋内の会場へ
# 差し替え先が解決できないときは home(自宅)へ落とす。屋内カテゴリのブロックは何もしない
# (条件成立は記録するが applied=false)。★新しい選択ヒューリスティックは足していない:
# 差し替えた先の具体ノードは既存の resolve_place(習慣 → 距離 → node 昇順)が決める。
OUTDOOR_PLACES: tuple[str, ...] = ("street", "leisure", "landmark", "attraction")
INDOOR_SWAP: dict[str, str] = {
    "street": "shop", "leisure": "cinema",
    "landmark": "hall", "attraction": "hall",
}

# ---- 決定論写像(全て純関数・設定不要)----------------------------------------- #
# act → 既定の場所カテゴリ(LLM が place を落とした/壊した時の substitute 先)。
#   routine._PLAN_CAT(meal→food / shop→shop / leisure→leisure / park→leisure /
#   work→office / meetup→landmark / visit→landmark)と整合させ、残りを埋めた。
ACT_PLACE: dict[str, str] = {
    "work": "work", "study": "education", "meal": "food", "shop": "shop",
    "leisure": "leisure", "park": "leisure", "walk": "street", "home": "home",
    "visit": "landmark", "personal": "service", "escort": "street",
    "meetup": "landmark",
}
# purpose → 活動3分類(plan_schema._CATS と同一語彙)。
PURPOSE_CAT: dict[str, str] = {
    "livelihood": "mandatory", "learning": "mandatory",
    "sustenance": "maintenance", "errand": "maintenance",
    "care": "maintenance", "health": "maintenance", "rest": "maintenance",
    "company": "discretionary", "enjoyment": "discretionary",
    "long_goal": "discretionary",
}
# act → 既定 purpose(LLM が purpose を落とした/壊した時の substitute 先)。
ACT_PURPOSE: dict[str, str] = {
    "work": "livelihood", "study": "learning", "meal": "sustenance",
    "shop": "errand", "leisure": "enjoyment", "park": "enjoyment",
    "walk": "health", "home": "rest", "visit": "company",
    "personal": "errand", "escort": "care", "meetup": "company",
}
# 場所カテゴリの別名(substitute 操作子で吸収する近い語)。LLM は列挙外を書きうる。
PLACE_ALIAS: dict[str, str] = {
    "cafe": "food", "restaurant": "food", "bar": "nightlife",
    "park": "leisure", "gym": "leisure", "school": "education",
    "station": "street", "outdoor": "street", "office_building": "office",
    "museum": "attraction", "theater": "hall", "theatre": "hall",
    "movie": "cinema", "store": "shop", "clinic": "service",
}
# 計画語彙 → 地図カテゴリの**追加照会先**(resolve_place だけが読む)。地図側の実カテゴリ名が
# 計画語彙と食い違う「実測で確認された分」だけを置く(推測で足さない)。
# 実測(2026-08-07 W2-4・本線地図 wide_v7): education は 1 件しかなく school が 68 件
# = 学業ブロックが事実上解決不能だった。LLM 語の正規化(PLACE_ALIAS: school→education)
# とは向きが別の表であることに注意。fallback の無いカテゴリの解決は従来と完全同値。
MAP_FALLBACK_CATS: dict[str, tuple[str, ...]] = {
    "education": ("school",),
}
# 実行時 activity タグ(routine._PLAN_ACTIVITY と同一。到着時の課金経路に載せるため)。
ACT_ACTIVITY: dict[str, str] = {"meal": "eating", "shop": "shopping"}

# 修復操作子(文献の modification operators と同名)。集計キーの唯一の源。
REPAIR_OPS: tuple[str, ...] = (
    "round",       # 時刻の丸め(グリッドへ吸着・start<end の強制・日内へクランプ)
    "substitute",  # 列挙外の値を既定へ差し替え(place/purpose/priority/flex/act)
    "slide",       # 重なり/移動時間不足の解消のため後ろへずらす
    "truncate",    # 前のブロックの終了を切り詰める(fixed が後ろに居るとき)
    "drop",        # 実行不能ブロックの削除(場所が解けない・日に収まらない)
    "clip",        # 件数上限(max_blocks / max_conting)の切り詰め
)
FALLBACK_KINDS: tuple[str, ...] = ("prev_day", "skeleton")

# --------------------------------------------------------------------------- #
# 設定(既定 OFF)
# --------------------------------------------------------------------------- #
DEFAULTS = {
    "enabled": False,
    "use_contingency": False,    # 第93 IF-A: plan["cont"] を実行時に消費する(既定 OFF)
    "postpone_min": 60,          # then=postpone のずらし幅[分]
    "shorten_min": 30,           # then=shorten の短縮幅[分]
    "min_blocks": 4,             # スキーマが要求する最小ブロック数(§7: 4〜8)
    "max_blocks": 8,             # 同・最大(超過は clip=低優先から落とす)
    "max_conting": 3,            # contingency の最大件数(§7: 最大3個)
    "round_min": 10,             # 時刻の丸め幅[分](= 正準 Δt)
    "min_dur_min": 10,           # ブロックの最小継続[分](これ未満は round で伸ばす)
    "max_dur_min": 600,          # 同・最大[分](超過は truncate)
    "grace_min": 30,             # 開始予定からこれだけ過ぎても始まらなければ割り込み判定
    "walk_m_per_min": 80.0,      # 移動時間見積りの徒歩速度(world.modes.speeds.walk=800m/10分)
    "transfer_min": 5,           # 同一ノード内でも要る最小の乗り換え/準備時間[分]
    "max_slide_min": 180,        # 1 ブロックの累積ずらし上限[分](超えたら drop)
    "day_end_min": 1440,         # 計画が収まるべき日の終端[分 of day]
    "mood_chars": 60,            # 自由文欄の上限(mood)
    "text_chars": 80,            # 自由文欄の上限(reason / note / carry)
}
_BOOL_KEYS = ("enabled", "use_contingency")
_INT_KEYS = ("min_blocks", "max_blocks", "max_conting", "round_min",
             "min_dur_min", "max_dur_min", "grace_min", "transfer_min",
             "max_slide_min", "day_end_min", "mood_chars", "text_chars",
             "postpone_min", "shorten_min")
_FLOAT_KEYS = ("walk_m_per_min",)


def _to_plain(raw):
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(raw):
            return OmegaConf.to_container(raw, resolve=True)
    except Exception:                              # noqa: BLE001 (omegaconf 不在でも動く)
        pass
    return raw


def build_cfg(raw) -> dict:
    """conf の planning.day_plan ブロックを正準化する(dunbar.build_cfg と同型)。"""
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
    cfg["round_min"] = max(1, cfg["round_min"])
    cfg["min_blocks"] = max(1, cfg["min_blocks"])
    cfg["max_blocks"] = max(cfg["min_blocks"], cfg["max_blocks"])
    cfg["max_conting"] = max(0, cfg["max_conting"])
    return cfg


def cfg_of(sim) -> dict:
    """day_plan 設定(初回のみ遅延構築してキャッシュ)。キャッシュ属性は L1/L2/乱数に出ない。"""
    c = getattr(sim, "dayplancfg", None)
    if c is None:
        try:
            raw = (sim.cfg.get("planning", None) or {}).get("day_plan", None)
        except Exception:                          # noqa: BLE001 (旧 config 互換)
            raw = None
        c = build_cfg(raw)
        sim.dayplancfg = c
    return c


def enabled(sim) -> bool:
    """day_plan v1 が実効か。**planning.enabled が前提**(朝の計画呼びの上に載る機構)。"""
    pc = getattr(sim, "planningcfg", None)
    if not (pc and pc.get("enabled")):
        return False
    return bool(cfg_of(sim)["enabled"])


def model_id(sim, agent=None) -> str:
    """この計画を書いたモデルの識別子。

    第88(心モデル固定)が ON なら **その個体に誕生時固定されたモデル**(= 実際にこの計画を
    書いたモデル)。OFF のときだけ従来どおり `sim.llm` の backend 名へ後退する
    (CachedLLM のラッパなので素の backend まで 1 段降りる。fleet 配下では
    `fleet/<model>` までしか判らない = 第87 までの暫定値)。
    """
    from .. import mind as _mind
    return _mind.log_model_id(sim, agent)


# --------------------------------------------------------------------------- #
# プロンプト(中立・no-fingerprint)
# --------------------------------------------------------------------------- #
# 冒頭マーカー。mock はこれを見てスキーマ準拠の決定論応答へ分岐する。
PROMPT_MARK = "一日の予定表:"

_ACT_JP = ("work=仕事 / study=学業 / meal=食事 / shop=買い物 / leisure=遊び / "
           "park=公園 / walk=散歩 / home=家で過ごす / visit=人を訪ねる / "
           "personal=用事 / escort=送り迎え / meetup=待ち合わせ")
_PLACE_JP = ("home=自宅 / work=職場や学校 / street=戸外 / food=飲食店 / "
             "nightlife=夜の店 / shop=お店 / service=用事先 / office=事務所 / "
             "education=学びの場 / hotel=宿 / attraction=見どころ / "
             "leisure=遊び場 / landmark=目印になる場所 / cinema=映画館 / hall=会場")
_PURPOSE_JP = ("livelihood=暮らしの糧 / learning=学び / sustenance=食事 / "
               "errand=用事 / care=世話 / health=体調 / rest=休息 / "
               "company=人と過ごす / enjoyment=楽しみ / long_goal=長い目で見た望み")


def schema_prompt(cfg: dict) -> str:
    """朝の計画プロンプトへ足すスキーマ指示(ON のみ)。

    中立規約(no-fingerprint): 機構語(発火・驚き・閾値・修復・フォールバック・検証・ルール・
    モデル・シミュレーション・エージェント)も実験条件語も因子名も 1 つも書かない。書くのは
    「今日をどう過ごすか」の書式だけ。自由文は mood / carry / reason / note に限る(§7)。

    ★ キーの並びは**生成順**でもあるので、各区切りは `reason`(自由文1行)を先頭に置く
      (§D の根拠: 理由欄を先頭に置くと構造化出力の推論品質が大きく改善する実測がある)。
    """
    lo, hi = cfg["min_blocks"], cfg["max_blocks"]
    return (
        f"\n{PROMPT_MARK} 今日をどう過ごすつもりか、{lo}〜{hi} 個の区切りで書いてください。"
        "\n空いている時間は書かなくてよい(埋めなくてよい)。"
        "\n各区切りの項目(この順に書く):"
        "\n  reason: そうする理由を1行で(必ず最初に書く)"
        "\n  start / end: 目安の時刻 HH:MM"
        f"\n  place: 行き先の種類({_PLACE_JP})。店の名前は書かない"
        f"\n  act: することの種類({_ACT_JP})"
        "\n  with: 一緒に過ごす人の名前(いなければ空の配列)"
        f"\n  aim: そうしたい理由の種類({_PURPOSE_JP})"
        "\n  priority: must=どうしても / should=できれば / could=余裕があれば"
        "\n  flex: fixed=時刻を動かせない / slideable=後ろへずらせる / droppable=やめてよい"
        "\n  note: ひとこと(任意)"
        f"\nもし〜なら、という備えを最大 {cfg['max_conting']} 個まで添えてよい:"
        "\n  if: rain=雨 / crowded=混んでいる / closed=閉まっている / tired=疲れた /"
        " no_money=お金が足りない / late=遅れた / invited=誘われた"
        "\n  then: skip=やめる / postpone=後にする / go_home=帰る /"
        " swap_indoor=屋内に変える / shorten=短くする"
        "\n出力は次の JSON 1個のみ(キー名は厳守):"
        '\n{"action": "plan", "mood": "今の気分をひとこと",'
        ' "carry": "昨日から気にしていること",'
        ' "blocks": [{"reason": "…", "start": "08:30", "end": "09:00",'
        ' "place": "food", "act": "meal", "with": [], "aim": "sustenance",'
        ' "priority": "should", "flex": "slideable", "note": ""}, ...],'
        ' "if_then": [{"if": "rain", "then": "swap_indoor"}]}')


# --------------------------------------------------------------------------- #
# ① スキーマ検証(列挙・個数・型)
# --------------------------------------------------------------------------- #
def _hhmm(value) -> int | None:
    """"HH:MM"(または分の整数)を分 of day へ。壊れていれば None。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        m = int(value)
        return m if 0 <= m < 1440 else None
    if not isinstance(value, str):
        return None
    text = value.strip()
    if ":" not in text:
        return int(text) if text.isdigit() and 0 <= int(text) < 1440 else None
    hh, _sep, mm = text.partition(":")
    try:
        h, m = int(hh), int(mm)
    except (TypeError, ValueError):
        return None
    return h * 60 + m if 0 <= h < 24 and 0 <= m < 60 else None


def _names(value) -> list:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()][:4]
    if isinstance(value, str) and value.strip():
        return [value.strip()][:4]
    return []


def _text(value, cap: int) -> str:
    return str(value or "").strip().replace("\n", " ")[:cap]


def _enum(value, allowed: tuple, alias: dict | None = None) -> str | None:
    """列挙の membership 判定。alias があれば 1 段だけ吸収する(= substitute の材料)。"""
    v = str(value or "").strip().lower()
    if v in allowed:
        return v
    if alias is not None and v in alias and alias[v] in allowed:
        return alias[v]
    return None


def validate_schema(raw: dict | None, cfg: dict) -> tuple[list, list, list]:
    """生 dict → (blocks, contingency, errors)。**型と列挙だけを見る**(物理は見ない)。

    errors は `"<code>:<where>"` の文字列列(決定論・出現順)。列挙外/欠損は
    ここでは **None のまま通し**、修復(② の後)で `substitute` として埋める
    ("壊れている" と "既定へ寄せられる" を分けて数えるため)。
    """
    errors: list = []
    if not isinstance(raw, dict):
        return [], [], ["broken_json:root"]
    if str(raw.get("action") or "").strip() not in ("day_plan", "plan"):
        errors.append("bad_action:root")
    rows = raw.get("blocks")
    if not isinstance(rows, list) or not rows:
        return [], [], errors + ["no_blocks:root"]
    blocks: list = []
    for i, it in enumerate(rows):
        if not isinstance(it, dict):
            errors.append(f"not_object:b{i}")
            continue
        start, end = _hhmm(it.get("start")), _hhmm(it.get("end"))
        if start is None:
            errors.append(f"bad_start:b{i}")
        if end is None:
            errors.append(f"bad_end:b{i}")
        act = _enum(it.get("act"), ACTS)
        if act is None:
            errors.append(f"bad_act:b{i}")
        place = _enum(it.get("place"), PLACE_CATS, PLACE_ALIAS)
        if place is None:
            errors.append(f"bad_place:b{i}")
        purpose = _enum(it.get("aim") or it.get("purpose"), PURPOSES)
        if purpose is None:
            errors.append(f"bad_aim:b{i}")
        priority = _enum(it.get("priority"), PRIORITIES)
        if priority is None:
            errors.append(f"bad_priority:b{i}")
        flex = _enum(it.get("flex"), FLEXES)
        if flex is None:
            errors.append(f"bad_flex:b{i}")
        blocks.append({"start": start, "end": end, "place": place, "act": act,
                       "with": _names(it.get("with")), "purpose": purpose,
                       "priority": priority, "flex": flex,
                       "reason": _text(it.get("reason"), cfg["text_chars"]),
                       "note": _text(it.get("note"), cfg["text_chars"]),
                       "node": None, "state": "todo", "slid": 0})
    if len(blocks) < cfg["min_blocks"]:
        errors.append("too_few:root")
    if len(blocks) > cfg["max_blocks"]:
        errors.append("too_many:root")
    conts: list = []
    # キー名は if_then(プロンプトが指示する綴り)を正典とし、contingency も受ける
    # (機構語 "contingency" をプロンプトに出さないための言い換え。読む側は両方許す)。
    for j, it in enumerate(raw.get("if_then") or raw.get("contingency") or []):
        if not isinstance(it, dict):
            errors.append(f"not_object:c{j}")
            continue
        cond = _enum(it.get("if") or it.get("cond"), CONDS)
        then = _enum(it.get("then") or it.get("do"), THENS)
        if cond is None or then is None:
            errors.append(f"bad_conting:c{j}")
            continue
        conts.append({"if": cond, "then": then})
    if len(conts) > cfg["max_conting"]:
        errors.append("too_many_conting:root")
    return blocks, conts, errors


# --------------------------------------------------------------------------- #
# 場所の具体解決(**ルール側**。習慣・距離・営業状態の決定論・乱数ゼロ)
# --------------------------------------------------------------------------- #
def _dist(sim, a_node, b_node) -> float:
    ax, ay = sim.city.node_xy(a_node)
    bx, by = sim.city.node_xy(b_node)
    return math.hypot(ax - bx, ay - by)


def travel_min(sim, cfg: dict, a_node, b_node) -> int:
    """a→b の移動所要の**下限見積り**[分]=直線距離 ÷ 徒歩速度 + 乗り換え/準備。

    経路探索の実所要ではない(正直な限界)。下限なので「これすら満たさない並び」だけを
    不能と判定する = 偽陽性側に倒してある。
    """
    if not a_node or not b_node:
        return int(cfg["transfer_min"])
    if a_node == b_node:
        return int(cfg["transfer_min"])
    speed = max(1e-6, float(cfg["walk_m_per_min"]))
    return int(cfg["transfer_min"] + math.ceil(_dist(sim, a_node, b_node) / speed))


def _work_node(agent):
    """職場(本業)→ バイト先の順で「働く場所」のノードを返す(無ければ "")。"""
    node = str(getattr(agent, "work_node", "") or "")
    if node:
        return node
    pt = getattr(agent, "part_time", None)
    return str(pt["node"]) if pt else ""


def resolve_place(sim, agent, place: str, sim_min: int, from_node=None):
    """場所カテゴリ → 具体ノード(**決定論**)。解けなければ None。

    順位は「習慣(訪問回数の多い順)→ 近い順 → node id 昇順」。既存の営業状態フィルタ
    (commerce.filter_open)と残高フィルタ(routine._poi_price)をそのまま再利用し、
    **新しい選択ヒューリスティックを足さない**(§7 の第一の設計判断: いつもの店に行く)。
    """
    if place == "home":
        return str(getattr(agent, "home_node", "") or "") or None
    if place == "work":
        return _work_node(agent) or None
    if place == "street":
        return None                              # 特定しない(習慣ポリシーに委ねる)
    from .. import commerce as _commerce
    from . import routine as _routine
    origin = from_node or getattr(agent, "node", "")
    pool_src = list(sim.city.pois_by_cat(place))
    for _cat in MAP_FALLBACK_CATS.get(place, ()):   # 地図語彙の食い違い分の追加照会(上表)
        _have = {p["node"] for p in pool_src}
        pool_src += [p for p in sim.city.pois_by_cat(_cat) if p["node"] not in _have]
    pool = [p for p in pool_src
            if p["node"] != origin
            and float(getattr(agent, "money", 0.0)) >= _routine._poi_price(sim, p)]
    pool = _commerce.filter_open(sim, pool, sim_min)
    if not pool:
        return None
    visits = getattr(agent, "visits", {})
    ox, oy = sim.city.node_xy(origin) if origin else (0.0, 0.0)

    def _key(p):
        px, py = sim.city.node_xy(p["node"])
        return (-int(visits[p["node"]]) if origin else 0,
                math.hypot(ox - px, oy - py) if origin else 0.0,
                str(p["node"]))
    return min(pool, key=_key)["node"]


# --------------------------------------------------------------------------- #
# ② 物理検証(移動時間・営業時間・場所カテゴリの実在)
# --------------------------------------------------------------------------- #
def validate_physical(sim, agent, blocks: list, cfg: dict,
                      day_min: int) -> list:
    """解決済みブロック列の物理的な実現可能性を検査する(errors の文字列列)。

    (a) 場所カテゴリの実在: 解決不能(該当 POI なし/自宅・職場が無い)= `no_place`
    (b) 営業時間: 開始時刻に閉まっているカテゴリ = `closed`
    (c) 移動時間: 前ブロックの終了 + travel_min > 次の開始 = `unreachable`
    (d) 日に収まる: end > day_end_min = `overflow`
    ★ 副作用として各ブロックの `node`(解決結果)を埋める(実行時はこれを使う)。
    ★ 検証は**修復より前**に走るので、時刻や場所が欠損(None)のブロックが混じりうる。
      欠損は ① スキーマ検証が既に別コードで挙げているので、ここでは**二重に数えず**
      該当する検査だけを飛ばす(場所カテゴリの実在だけは None でも見る)。
    """
    from .. import commerce as _commerce
    errors: list = []
    prev_node, prev_end = str(getattr(agent, "node", "") or ""), None
    ccfg = getattr(sim, "commercecfg", None)
    for i, b in enumerate(blocks):
        start, end, place = b["start"], b["end"], b["place"]
        at = day_min + (int(start) if start is not None else 0)
        # ---- P4 計画駆動の圏外滞在(既定 OFF=mark は常に None=以降は従来どおり)----
        #  圏外ブロックは「地図の外」なので (a) 場所カテゴリの実在も (b) 営業時間も
        #  問わない(そこに POI は無いし、街の営業時間は外の職場を閉めない)。
        #  到達性だけは**縁(gateway)まで**を通常どおり見る = 移動時間は無料にしない。
        gate = _bnd.mark(sim, agent, b)
        if gate is not None:
            node = gate
        else:
            node = resolve_place(sim, agent, place, at, from_node=prev_node) \
                if place is not None else None
            b["node"] = node
            if node is None and place not in (None, "street"):
                errors.append(f"no_place:b{i}")
            if place is not None and ccfg is not None and _commerce.enabled(sim) \
                    and not _commerce.is_open(ccfg, place, at):
                errors.append(f"closed:b{i}")
        if prev_end is not None and start is not None:
            need = travel_min(sim, cfg, prev_node, node or prev_node)
            if int(start) < prev_end + need:
                errors.append(f"unreachable:b{i}")
        if end is not None and int(end) > int(cfg["day_end_min"]):
            errors.append(f"overflow:b{i}")
        if end is not None:
            prev_end = int(end)
        prev_node = node or prev_node
    return errors


# --------------------------------------------------------------------------- #
# ③ 決定的ルールによる修復(文献の modification operators)
# --------------------------------------------------------------------------- #
_PRIO_RANK = {"must": 0, "should": 1, "could": 2}


def _round_to(value: int, grid: int) -> int:
    return int(round(value / grid)) * grid


def repair(sim, agent, blocks: list, conts: list, cfg: dict,
           day_min: int) -> tuple[list, list, dict]:
    """検証で見つかった不整合を**決定論**で直す。返り値 (blocks, conts, ops)。

    ops は `{操作子: 件数}`(REPAIR_OPS のキーのみ)。**同じ入力からは常に同じ修復**が出る
    (乱数ゼロ・LLM ゼロ・順序も固定)。操作子の適用順は文献の順序(値の正規化 → 時刻の
    正規化 → 並べ替え → 実行可能性 → 件数上限)に従う。
    """
    ops = {k: 0 for k in REPAIR_OPS}

    # -- substitute: 列挙外/欠損を既定へ寄せる ---------------------------------
    kept: list = []
    for b in blocks:
        if b["act"] is None and b["place"] is None:
            ops["drop"] += 1                     # 種別も場所も無い = 実行できない
            continue
        if b["act"] is None:
            b["act"] = next((a for a, p in ACT_PLACE.items() if p == b["place"]),
                            "personal")
            ops["substitute"] += 1
        if b["place"] is None:
            b["place"] = ACT_PLACE[b["act"]]
            ops["substitute"] += 1
        if b["purpose"] is None:
            b["purpose"] = ACT_PURPOSE[b["act"]]
            ops["substitute"] += 1
        if b["priority"] is None:
            b["priority"] = "should"
            ops["substitute"] += 1
        if b["flex"] is None:
            b["flex"] = "fixed" if b["priority"] == "must" else "slideable"
            ops["substitute"] += 1
        kept.append(b)
    blocks = kept

    # -- round: 時刻の丸め・欠損補完・最小/最大継続・日内クランプ -----------------
    day_end = int(cfg["day_end_min"])
    grid, lo_dur, hi_dur = cfg["round_min"], cfg["min_dur_min"], cfg["max_dur_min"]
    for b in blocks:
        s, e = b["start"], b["end"]
        before = (s, e)
        if s is None and e is None:
            s, e = 12 * 60, 12 * 60 + lo_dur     # 時刻が皆無 = 正午に最小幅で置く
        elif s is None:
            s = max(0, e - lo_dur)
        elif e is None:
            e = s + lo_dur
        s = min(max(0, _round_to(s, grid)), day_end - lo_dur)
        e = _round_to(e, grid)
        if e - s < lo_dur:
            e = s + lo_dur
        if e - s > hi_dur:
            e = s + hi_dur
            ops["truncate"] += 1
        if e > day_end:
            e = day_end
            s = min(s, e - lo_dur)
            ops["truncate"] += 1
        if (s, e) != before:
            ops["round"] += 1
        b["start"], b["end"] = int(s), int(e)

    # -- 並べ替え(開始時刻昇順・同時刻は priority 強い順・さらに act 名で全順序)-----
    blocks.sort(key=lambda b: (b["start"], _PRIO_RANK[b["priority"]], b["act"]))

    # -- clip: 件数上限(低優先・遅い方から落とす)-------------------------------
    if len(blocks) > cfg["max_blocks"]:
        order = sorted(range(len(blocks)),
                       key=lambda i: (-_PRIO_RANK[blocks[i]["priority"]],
                                      -blocks[i]["start"]))
        drop_at = set(order[:len(blocks) - cfg["max_blocks"]])
        blocks = [b for i, b in enumerate(blocks) if i not in drop_at]
        ops["clip"] += len(drop_at)
    if len(conts) > cfg["max_conting"]:
        ops["clip"] += len(conts) - cfg["max_conting"]
        conts = conts[:cfg["max_conting"]]

    # -- 場所解決 → slide / truncate / drop(移動時間と重なりの解消)---------------
    out: list = []
    prev_node, prev_end = str(getattr(agent, "node", "") or ""), None
    for b in blocks:
        at = day_min + b["start"]
        # ---- P4 計画駆動の圏外滞在(既定 OFF=mark は常に None=以降は従来どおり)----
        #  刻印はここが**最終**(LLM 経路・前日計画経路・骨格経路のいずれも repair を通る)。
        #  圏外ブロックは地図内 POI へ解決しない = 従来なら drop されていた。縁ノードを
        #  「この区間の物理的な居場所」として扱い、以降の移動時間・日内クランプは
        #  通常ブロックと同じ規則で通す(新しいヒューリスティックを足さない)。
        gate = _bnd.mark(sim, agent, b)
        if gate is not None:
            node = gate                          # b["node"] は mark が None に据えてある
        else:
            node = resolve_place(sim, agent, b["place"], at, from_node=prev_node)
            if node is None and b["place"] != "street":
                sub = ACT_PLACE[b["act"]]        # substitute: act の既定カテゴリへ
                if sub != b["place"]:
                    node = resolve_place(sim, agent, sub, at, from_node=prev_node)
                    if node is not None or sub == "street":
                        b["place"] = sub
                        ops["substitute"] += 1
            if node is None and b["place"] != "street":
                ops["drop"] += 1                 # 場所が解けない = 実行不能
                continue
            b["node"] = node
        if prev_end is not None:
            need = travel_min(sim, cfg, prev_node, node or prev_node)
            if b["start"] < prev_end + need:
                if b["flex"] == "fixed" and out and out[-1]["flex"] != "fixed":
                    cut = prev_end + need - b["start"]   # 前を短縮して固定を守る
                    out[-1]["end"] = max(out[-1]["start"] + lo_dur,
                                         out[-1]["end"] - cut)
                    ops["truncate"] += 1
                else:
                    dur = b["end"] - b["start"]
                    shift = prev_end + need - b["start"]
                    b["start"] = _round_to(prev_end + need, grid)
                    b["end"] = b["start"] + dur
                    b["slid"] = int(b["slid"]) + int(shift)
                    ops["slide"] += 1
        if b["end"] > day_end or b["slid"] > cfg["max_slide_min"]:
            ops["drop"] += 1                     # 日に収まらない / ずらしすぎ
            continue
        out.append(b)
        prev_end, prev_node = b["end"], node or prev_node
    return out, conts, ops


# --------------------------------------------------------------------------- #
# ④ フォールバック(修復不能 → 世界を止めない)
# --------------------------------------------------------------------------- #
def skeleton(sim, agent, cfg: dict) -> list:
    """ペルソナ既定ルーチン = 職業別の骨格(LLM なし・乱数ゼロ・決定論)。

    勤務窓を持つ人は「出勤 → 退勤後の食事 → 帰宅」、持たない人は「昼の用事 → 夕食 → 家」。
    plan_schema.default_skeleton と同じ思想(初日/破綻時だけの粗い骨格)を day_plan v1 の
    表現で書き直したもの。恒久ルール化はしない(翌朝はまた計画が立つ)。
    """
    def _b(start, end, place, act, priority, flex, reason):
        return {"start": int(start), "end": int(end), "place": place, "act": act,
                "with": [], "purpose": ACT_PURPOSE[act], "priority": priority,
                "flex": flex, "reason": reason, "note": "",
                "node": None, "state": "todo", "slid": 0}

    # ---- P4 計画駆動の圏外滞在(既定 OFF=None=以降は従来の 2 分岐と完全同一)----
    #  圏外通勤者は work_start_min < 0(地図内に職場が無い)ので、下の 2 分岐では
    #  「勤務窓を持たない人」に落ちて圏外へ出ない。第 3 の骨格をここで返す。
    out_rows = _bnd.skeleton_rows(sim, agent, cfg)
    if out_rows is not None:
        return out_rows

    ws = int(getattr(agent, "work_start_min", -1) or -1)
    pt = getattr(agent, "part_time", None)
    if ws >= 0 or pt:
        s = ws if ws >= 0 else int(pt["start_min"])
        e = int(getattr(agent, "work_end_min", 0) or 0) if ws >= 0 \
            else int(pt["end_min"])
        e = max(s + cfg["min_dur_min"], e)
        rows = [_b(s, e, "work", "work", "must", "fixed", "いつもの時間に働く"),
                _b(e, e + 60, "food", "meal", "should", "slideable", "働いた後に食べる"),
                _b(e + 60, min(cfg["day_end_min"], e + 180), "home", "home",
                   "should", "slideable", "家に帰って休む")]
    else:
        rows = [_b(11 * 60, 12 * 60, "shop", "shop", "should", "slideable",
                   "昼のうちに用を済ませる"),
                _b(12 * 60, 13 * 60, "food", "meal", "should", "slideable",
                   "昼を食べる"),
                _b(18 * 60, 19 * 60, "food", "meal", "should", "droppable",
                   "夕飯を食べる"),
                _b(19 * 60, 21 * 60, "home", "home", "should", "slideable",
                   "家でゆっくり過ごす")]
    return [b for b in rows if b["start"] < cfg["day_end_min"]]


# --------------------------------------------------------------------------- #
# 集計(L2 / summary。checkpoint.py が中央管理する state)
# --------------------------------------------------------------------------- #
def _state(sim) -> dict:
    st = getattr(sim, "_dayplan_state", None)
    if st is None:
        st = {"schema": SCHEMA, "plans": 0, "blocks": 0, "exec": 0, "drop": 0,
              "slide": 0, "replan": 0, "repair": 0, "conflict": 0,
              "fallback": 0, "by_model": {}}
        sim._dayplan_state = st
    return st


def _model_row(st: dict, mid: str) -> dict:
    row = st["by_model"].get(mid)
    if row is None:
        row = {"plans": 0, "blocks": 0, "repair": 0, "conflict": 0, "fallback": 0,
               "repair_ops": {}, "fallback_kinds": {}, "schema_err": 0,
               "phys_err": 0}
        st["by_model"][mid] = row
    return row


def scalars(sim) -> dict | None:
    """L2 全体スカラー 7 列(OFF は None=列なし=L2 不変。dunbar.scalars と同型)。"""
    if not enabled(sim):
        return None
    st = getattr(sim, "_dayplan_state", None)
    if st is None:
        return {"plan_blocks_mean": 0.0, "plan_exec_total": 0.0,
                "plan_repair_total": 0.0, "plan_fallback_total": 0.0,
                "plan_drop_total": 0.0, "plan_slide_total": 0.0,
                "plan_replan_total": 0.0}
    n = int(st["plans"])
    return {"plan_blocks_mean": round(st["blocks"] / n, 6) if n else 0.0,
            "plan_exec_total": float(st["exec"]),
            "plan_repair_total": float(st["repair"]),
            "plan_fallback_total": float(st["fallback"]),
            "plan_drop_total": float(st["drop"]),
            "plan_slide_total": float(st["slide"]),
            "plan_replan_total": float(st["replan"])}


def provenance(sim) -> dict | None:
    """summary.json の day_plan キー(OFF は None=キー自体を出さない)。

    **修復・フォールバックの発生回数をモデル別に**返す(原文書 §7: テストバッテリーの
    追加指標)。率の分母は「その日その人ぶんの計画 1 件」= plans。
    """
    if not enabled(sim):
        return None
    st = getattr(sim, "_dayplan_state", None)
    if st is None:                               # ON だが 1 件も計画が立っていない(短いラン)
        return {"schema": SCHEMA, "plans": 0, "blocks": 0, "executed": 0,
                "dropped": 0, "slid": 0, "replans": 0, "by_model": {}}
    out = {"schema": SCHEMA, "plans": int(st["plans"]),
           "blocks": int(st["blocks"]), "executed": int(st["exec"]),
           "dropped": int(st["drop"]), "slid": int(st["slide"]),
           "replans": int(st["replan"]), "by_model": {}}
    for mid in sorted(st["by_model"]):
        row = st["by_model"][mid]
        n = max(1, int(row["plans"]))
        out["by_model"][mid] = {
            "plans": int(row["plans"]), "blocks": int(row["blocks"]),
            "repair_plans": int(row["repair"]),
            "repair_rate": round(row["repair"] / n, 4),
            # ★バッテリー指標として意味があるのはこちら: 時刻グリッドへの丸め(round)は
            #   ほぼ全ての計画で起きる正規化なので repair_rate は 1.0 に張り付きやすい。
            #   conflict_* は「ずらし/短縮/差し替え/削除/上限」= 実際に矛盾を直した計画の割合。
            "conflict_repair_plans": int(row["conflict"]),
            "conflict_repair_rate": round(row["conflict"] / n, 4),
            "fallback_plans": int(row["fallback"]),
            "fallback_rate": round(row["fallback"] / n, 4),
            "repair_ops": {k: int(v) for k, v in sorted(row["repair_ops"].items())},
            "fallback_kinds": {k: int(v)
                               for k, v in sorted(row["fallback_kinds"].items())},
            "schema_errors": int(row["schema_err"]),
            "physical_errors": int(row["phys_err"])}
    return out


# --------------------------------------------------------------------------- #
# 生成の入口(planning.apply_plan_response からの**単一作用点**)
# --------------------------------------------------------------------------- #
def apply(sim, agent, step: int, sim_min: int, response: str,
          call_id: str | None) -> None:
    """LLM 応答 → 検証 → 修復 → フォールバック → agent へ格納 + L1 記録。

    **generate() を 1 本も足さない**(再試行なし)。全段が決定論。
    """
    cfg = cfg_of(sim)
    day = int(sim_min) // 1440
    day_min = day * 1440
    # 第94 IF-B: 前日の失敗理由は**持ち越さない**(未解決のまま日を跨いだら消す)。
    # 朝の計画生成は 1 日 1 回なのでここが日境界の単一作用点。既定 silent は no-op。
    _clear_why(sim, agent)
    # P4 計画駆動の圏外滞在(既定 OFF=即 return=何も書かない): 圏外通勤フラグを
    # (run.seed, agent.id) の純関数から確定する。以降の検証・修復・骨格がこれを読む。
    _bnd.ensure(sim, agent)
    mid = model_id(sim, agent)
    st = _state(sim)
    row = _model_row(st, mid)

    raw = _loads_lenient(response)
    blocks, conts, errs = validate_schema(raw, cfg)
    row["schema_err"] += len(errs)
    perrs: list = []
    ops = {k: 0 for k in REPAIR_OPS}
    if blocks:
        perrs = validate_physical(sim, agent, blocks, cfg, day_min)
        row["phys_err"] += len(perrs)
        blocks, conts, ops = repair(sim, agent, blocks, conts, cfg, day_min)
    n_ops = sum(ops.values())
    if n_ops:
        st["repair"] += 1
        row["repair"] += 1
        if n_ops - ops["round"] > 0:             # 矛盾を実際に直した(丸めだけではない)
            st["conflict"] += 1
            row["conflict"] += 1
        for k, v in ops.items():
            if v:
                row["repair_ops"][k] = row["repair_ops"].get(k, 0) + int(v)
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                             kind="plan_repair", x=agent.x, y=agent.y,
                             llm_call_id=call_id,
                             payload={"ops": {k: int(v) for k, v in sorted(ops.items())
                                              if v},
                                      "n_schema_err": len(errs),
                                      "n_phys_err": len(perrs),
                                      "model": mid}))

    fallback = ""
    if not blocks:                               # 修復不能 → 世界を止めない
        prev = getattr(agent, "_dayplan_prev", None)
        if prev:
            reblocks = [dict(b, node=None, state="todo", slid=0) for b in prev]
            reblocks, conts, _o = repair(sim, agent, reblocks, conts, cfg, day_min)
            if reblocks:
                blocks, fallback = reblocks, "prev_day"
        if not blocks:
            blocks, conts = skeleton(sim, agent, cfg), []
            blocks, conts, _o = repair(sim, agent, blocks, conts, cfg, day_min)
            fallback = "skeleton"
        st["fallback"] += 1
        row["fallback"] += 1
        row["fallback_kinds"][fallback] = row["fallback_kinds"].get(fallback, 0) + 1
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                             kind="plan_fallback", x=agent.x, y=agent.y,
                             llm_call_id=call_id,
                             payload={"kind": fallback, "n": len(blocks),
                                      "n_schema_err": len(errs),
                                      "n_phys_err": len(perrs), "model": mid}))

    plan = {"schema": SCHEMA, "day": day, "version": 1, "model": mid,
            "mood": _text((raw or {}).get("mood"), cfg["mood_chars"]),
            "carry": _text((raw or {}).get("carry")
                           or (raw or {}).get("carryover"), cfg["text_chars"]),
            "blocks": blocks, "cont": conts, "src": fallback or "llm"}
    # IF-1(observer.llm_link。既定 OFF ではキー自体を生やさない = 状態も出力も不変)。
    # 「この計画を書いた朝の LLM 呼」を控えておき、日中にブロックを実行して生まれる
    # 行為イベントから role="plan" で参照できるようにする(PROV の wasInformedBy)。
    from .. import provlink as _provlink
    if call_id and _provlink.enabled(sim):
        plan["call_id"] = str(call_id)
    agent._dayplan = plan
    agent._dayplan_prev = [dict(b) for b in blocks]
    agent.day_plan = []          # 旧表現は使わない(routine の旧経路を確実に黙らせる)
    st["plans"] += 1
    st["blocks"] += len(blocks)
    row["plans"] += 1
    row["blocks"] += len(blocks)
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                         kind="plan_created", x=agent.x, y=agent.y,
                         llm_call_id=call_id,
                         payload={"n": len(blocks), "version": 1, "src": plan["src"],
                                  "model": mid, "n_cont": len(conts),
                                  "blocks": [{"start": b["start"], "end": b["end"],
                                              "place": b["place"], "act": b["act"],
                                              "aim": b["purpose"],
                                              "priority": b["priority"],
                                              "flex": b["flex"]} for b in blocks]}))
    # 第87(engaged)脱出条件 (1) 解消: 「**検証を通る計画が生成されたとき**」(設計 §8)。
    # 後退(fallback)した計画は解消ではない — 世界を止めないための応急処置であって、
    # 例外を解決したわけではないから。既定 OFF は完全 no-op。
    # 後退なしの値は ""(907行の初期値)であり None は入らない — `is None` だと恒偽で
    # 脱出条件(1)が死ぬ(IF-B 検収で発見した実バグ。真偽値判定が正)。
    if not fallback:
        from . import engaged as _engaged
        _engaged.note_resolved(sim, agent)


# --------------------------------------------------------------------------- #
# 実行系(routine.decide からの**単一作用点**)
# --------------------------------------------------------------------------- #
def _log(sim, agent, step, sim_min, kind, payload) -> None:
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                         kind=kind, x=agent.x, y=agent.y, payload=payload))


def _plan_of(agent, sim_min: int):
    plan = getattr(agent, "_dayplan", None)
    if not plan or int(plan.get("day", -1)) != int(sim_min) // 1440:
        return None
    return plan


def _bi(plan: dict, b: dict) -> int:
    """ブロックの添字(W4-F)。`plan["blocks"]` に無ければ -1。

    ★**同一性**で採る: `list.index` は `==` 比較なので、同じ時刻・同じ場所の 2 件
      (LLM は平気で書く)を取り違える。第93 IF-A の `apply_contingency` が既に
      この採り方をしており、それを関数へ括り出しただけ = 新しい規則を足していない。

    用途は**記録だけ**(この値を読んで分岐する箇所は 1 つも無い)。plan_block_* の
    payload へ載せることで、事後解析(scripts/analyze_plan_execution.py)が
    「どのイベントがどのブロックのものか」を台帳再生で推定せずに済む。
    """
    return next((i for i, x in enumerate(plan["blocks"]) if x is b), -1)


def _sweep(sim, agent, plan: dict, cfg: dict, step: int, sim_min: int) -> None:
    """priority × flex の割り込み処理(**LLM を呼ばない**・決定論)。

    「時間が食われた」= 予定の開始から grace_min を過ぎてもまだ始まっていない状態。
      - could + droppable … その場で削る(`plan_block_drop`)
      - slideable        … 後ろへずらす(`plan_slide`)。累積 max_slide_min 超過で削る
      - must が脅かされた … `_replan`(低優先を削って must の場所を空ける)+ plan_replan
    fixed は動かさない(時刻を動かせない宣言なので、間に合わなければ must なら再計画、
    must でなければ削る)。
    """
    now = int(sim_min) % 1440
    grace, day_end = int(cfg["grace_min"]), int(cfg["day_end_min"])
    threatened: list = []
    for i, b in enumerate(plan["blocks"]):
        if b["state"] != "todo" or now <= b["start"] + grace:
            continue
        if b["priority"] == "must":
            threatened.append(b)
            continue
        if b["priority"] == "could" and b["flex"] == "droppable":
            b["state"] = "dropped"
            _state(sim)["drop"] += 1
            _log(sim, agent, step, sim_min, "plan_block_drop",
                 {"act": b["act"], "place": b["place"], "start": b["start"],
                  "priority": b["priority"], "flex": b["flex"], "reason": "grace",
                  "block": i})
            continue
        if b["flex"] == "slideable":
            dur = b["end"] - b["start"]
            shift = _round_to(now, cfg["round_min"]) - b["start"]
            b["slid"] = int(b["slid"]) + max(0, int(shift))
            if b["slid"] > cfg["max_slide_min"] or now + dur > day_end:
                b["state"] = "dropped"
                _state(sim)["drop"] += 1
                _log(sim, agent, step, sim_min, "plan_block_drop",
                     {"act": b["act"], "place": b["place"], "start": b["start"],
                      "priority": b["priority"], "flex": b["flex"],
                      "reason": "slide_cap", "block": i})
            else:
                b["start"] = _round_to(now, cfg["round_min"])
                b["end"] = b["start"] + dur
                _state(sim)["slide"] += 1
                _log(sim, agent, step, sim_min, "plan_slide",
                     {"act": b["act"], "place": b["place"], "start": b["start"],
                      "slid": int(b["slid"]), "priority": b["priority"],
                      "block": i})
        else:                                    # fixed で must でない = 機会を逃した
            b["state"] = "dropped"
            _state(sim)["drop"] += 1
            _log(sim, agent, step, sim_min, "plan_block_drop",
                 {"act": b["act"], "place": b["place"], "start": b["start"],
                  "priority": b["priority"], "flex": b["flex"],
                  "reason": "fixed_past", "block": i})
    if threatened:                               # ★must が脅かされたときだけ再計画が鳴る
        _replan(sim, agent, plan, cfg, step, sim_min, threatened)


def _replan(sim, agent, plan: dict, cfg: dict, step: int, sim_min: int,
            musts: list) -> None:
    """must が脅かされたときの再計画(**本バッチはルール内**・LLM 呼ゼロ)。

    (1) must を「今」へずらす(fixed でなければ = 最小摂動)、
    (2) 動かせず窓も過ぎた must は**組めなかったものとして退役**させる(ActivitySim の
        `failed` フラグに倣い、黙って残さず数える)。ここで retire しないと同じ must が
        毎 step 脅威判定に引っかかって再計画が無限に鳴り続ける,
    (3) **ずらした後の** must の窓と重なる could を削る(場所と時間を空ける)。
        順序が肝: 先に could を削ると、ずらした先で新たに邪魔になる予定が残る,
    (4) plan_version を上げて `plan_replan` を出す、
    (5) 第81 の認知イベントキューが ON なら内部発火 `plan_exception` を前倒し登録する
        (設計 §8 突入条件(3)の先行実装。エピソード本体は第87)。

    ★ 1 step につき高々 1 回(`replan_at` で冪等)。
    """
    now = int(sim_min) % 1440
    if int(plan.get("replan_at", -1)) == int(sim_min):
        return
    plan["replan_at"] = int(sim_min)
    # ---- IF-B(第94・cognition.rejection_notify != "silent" のみ)------------------ #
    #  「何が失敗したか」を 1 件だけ控える。**先頭の must**(= 計画の並び順 = 決定論)を採り、
    #  下のループが m をずらす/退役させる**前**に分類する(書き換わった値で理由を作らない)。
    #  既定 silent では reject.note_why が即 return = agent に属性すら生えない。
    why = None
    if musts:
        m0 = musts[0]
        why = (str(m0["act"]),
               "missed" if (m0["flex"] == "fixed" and now >= m0["end"]) else "late")
    freed, retired = 0, 0
    for m in musts:
        if m["flex"] != "fixed":                 # 最小摂動: 必要なぶんだけ「今」へずらす
            dur = m["end"] - m["start"]
            shift = _round_to(now, cfg["round_min"]) - m["start"]
            m["slid"] = int(m["slid"]) + max(0, int(shift))
            m["start"] = _round_to(now, cfg["round_min"])
            m["end"] = min(int(cfg["day_end_min"]), m["start"] + dur)
        elif now >= m["end"]:                    # 動かせず窓も過ぎた = 組めなかった
            m["state"] = "dropped"
            retired += 1
            _state(sim)["drop"] += 1
            _log(sim, agent, step, sim_min, "plan_block_drop",
                 {"act": m["act"], "place": m["place"], "start": m["start"],
                  "priority": m["priority"], "flex": m["flex"], "reason": "missed",
                  "block": _bi(plan, m)})
            continue
        for i, b in enumerate(plan["blocks"]):   # ずらした先で邪魔になる could を空ける
            if b is m or b["state"] != "todo" or b["priority"] != "could":
                continue
            if b["start"] < m["end"] and m["start"] < b["end"]:
                b["state"] = "dropped"
                freed += 1
                _state(sim)["drop"] += 1
                _log(sim, agent, step, sim_min, "plan_block_drop",
                     {"act": b["act"], "place": b["place"], "start": b["start"],
                      "priority": b["priority"], "flex": b["flex"],
                      "reason": "replan", "block": i})
    plan["version"] = int(plan["version"]) + 1
    _state(sim)["replan"] += 1
    _log(sim, agent, step, sim_min, "plan_replan",
         {"version": int(plan["version"]), "n_must": len(musts), "freed": freed,
          "retired": retired, "at": now})
    note_plan_exception(sim, agent, sim_min, why=why)


def note_plan_exception(sim, agent, sim_min: int, why=None) -> None:
    """実行不能例外を第81 の認知イベントキューへ内部発火として前倒し登録する。

    `cognition.fire` が OFF(既定)なら**完全 no-op**(キューが存在しない)= LLM 呼数不変。
    ON のときだけ「いま考える」へ繰り上げる(= 設計 §8 突入条件(3) の先行実装。
    engaged エピソード本体は第87)。

    `why`(第94 IF-B・既定 None)は「予定していた〈act〉が〈reason〉でできなかった」の
    材料 2 語だけ。**渡されたときだけ**控え、engaged の REPLAN エピソード中の状況行に
    1 行だけ載る(`cognition.rejection_notify` が silent なら属性すら生えない)。
    ★拒否通知(reject.notify)経由の前倒しは why を渡さない — あちらは既に定型文 1 行を
      記憶へ書いており、同じ内容をプロンプトに二重に置かないため。
    """
    from . import fire as _fire
    if not _fire.enabled(sim) or _fire.INTERNAL not in sim.firecfg["sources"]:
        return
    q = getattr(sim, "cogq", None)
    if q is None:
        return
    q.advance(int(agent.id), int(sim_min), _fire.INTERNAL)
    agent._plan_exception = int(sim_min)          # 第87 が読む印(本バッチでは記録のみ)
    if why is not None:                           # 第94 IF-B(既定 silent は即 return)
        from .. import reject as _reject
        _reject.note_why(sim, agent, why[0], why[1])


# --------------------------------------------------------------------------- #
# contingency の消費(第93バッチ IF-A・use_contingency ON のみ・既定 OFF=0 件)
#
# 監査 §5-1 の穴: 第86 は if_then を最大 3 個 LLM に書かせ、検証して `plan["cont"]` へ
# 格納するが **読むコードが無かった**(デッドデータ)。ここがその消費側。
#
# 契約
# ----
# - **完全決定論**: 乱数を 1 本も引かず、LLM を 1 本も呼ばず、プロンプトを 1 バイトも
#   変えない(if_then の書式は day_plan ON なら本トグルと無関係に schema_prompt に載る)。
# - 判定材料は **既存の世界状態だけ**。前提の機構が OFF の条件は **常に不成立**
#   (欠測を「起きた」とも「起きなかった」とも捏造しないための片側倒し)。
# - 評価点は **ブロックが「いま実行すべき」になった瞬間の 1 回だけ**(plan_action)。
#   `_sweep`(割り込み処理)には置かない: あちらは priority×flex の規則が既に
#   ずらす/削るを決めており、そこへ第 2 の規則を差し込むと同じブロックに 2 つの
#   対処が重なる(振動する)。
# - 成立したのは **最初の 1 件だけ**を適用する(cont の並びは LLM が書いた順のまま)。
#   適用後は `cont_done` を立てて 1 ブロック 1 回に固定する(postpone の無限先送り防止)。
#
# 各条件が「評価できる」条件(= 前提機構)。**満たさないランでは常に False**:
#   rain      … weather.enabled(sim.today_weather の cond が悪天候語)
#   crowded   … env.feedback.enabled かつ env.feedback.poi.enabled
#                (envfeedback.blocked_nodes = 待ち行列で塞がっている POI)
#   closed    … commerce.enabled(営業時間表。_open_now と同一の判定)
#   tired     … health.enabled(fatigue が既存閾値 fatigue_high 以上)
#   no_money  … economy.enabled かつ **そのカテゴリに価格が定義されている**
#                (food/cafe/shop/nightlife/leisure。service/office/education 等は
#                 価格が無い = 判定不能なので常に False)
#   late      … 前提なし(割り込み処理でずらされた = slid>0 が「遅れた」の実体)
#   invited   … joint.enabled(当日の共同予定 band とブロックの時間帯が重なる)
#
# 評価**できない**条件は 1 つも無い(CONDS 7 種すべてに実装がある)が、上のとおり
# 前提機構が OFF のランでは発火しない。これは「実装が無い」のではなく
# 「その世界にはその現象が存在しない」ので、捏造せず False を返している。
# --------------------------------------------------------------------------- #
def _cond_rain(sim, agent, b: dict, cfg: dict, sim_min: int) -> bool:
    """当日の天気が悪天候(雨・雪)か。weather OFF では today_weather が無い=False。"""
    from .. import weather as _weather
    w = getattr(sim, "today_weather", None)
    # 悪天候語の定義は weather 側の単一源を参照する(二重定義を作らない)。
    return bool(w) and str(w.get("cond", "")) in _weather._BAD_CONDS


def _cond_crowded(sim, agent, b: dict, cfg: dict, sim_min: int) -> bool:
    """行き先が待ち行列で塞がっているか(env.feedback 規則3の poi_hold)。"""
    from .. import envfeedback as _envfb
    node = b.get("node")
    return bool(node) and node in _envfb.blocked_nodes(sim)


def _cond_closed(sim, agent, b: dict, cfg: dict, sim_min: int) -> bool:
    """行き先カテゴリがいま閉まっているか(commerce OFF では常に開店=False)。"""
    return not _open_now(sim, b, sim_min)


def _cond_tired(sim, agent, b: dict, cfg: dict, sim_min: int) -> bool:
    """疲労が既存の「高疲労」閾値(health.fatigue_high)以上か。health OFF は False。"""
    hcfg = getattr(sim, "healthcfg", None)
    if not hcfg or not hcfg.get("enabled"):
        return False
    return float(getattr(agent, "fatigue", 0.0) or 0.0) >= float(hcfg["fatigue_high"])


def _cond_no_money(sim, agent, b: dict, cfg: dict, sim_min: int) -> bool:
    """行き先カテゴリの価格に所持金が届かないか。価格が定義されていなければ False。"""
    from ..economy import price_of as _price_of
    eco = getattr(sim, "economy", None)
    if not eco or not eco.get("enabled"):
        return False
    price = float(_price_of(str(b["place"]), eco, getattr(sim, "rulebook", None)))
    return price > 0.0 and float(getattr(agent, "money", 0.0) or 0.0) < price


def _cond_late(sim, agent, b: dict, cfg: dict, sim_min: int) -> bool:
    """このブロックが後ろへずらされているか(= 割り込み処理が「遅れ」と判定した実体)。"""
    return int(b.get("slid", 0) or 0) > 0


def _cond_invited(sim, agent, b: dict, cfg: dict, sim_min: int) -> bool:
    """当日の共同予定(誘い)の時間帯がこのブロックと重なるか。joint OFF は False。"""
    jt = getattr(agent, "joint_today", None)
    if not jt:
        return False
    lo, hi = jt["band"]
    return int(b["start"]) < int(hi) and int(lo) < int(b["end"])


COND_FN = {"rain": _cond_rain, "crowded": _cond_crowded, "closed": _cond_closed,
           "tired": _cond_tired, "no_money": _cond_no_money, "late": _cond_late,
           "invited": _cond_invited}


def _drop_block(sim, agent, b: dict, step: int, sim_min: int, reason: str,
                block: int = -1) -> None:
    """ブロックを削る(既存の drop 計上・イベントと同一形 = st['drop'] の保存則を守る)。

    `block`(W4-F)= `plan["blocks"]` の添字。**呼び元が既に持っている値をそのまま
    受け取るだけ**で、ここで探し直さない(唯一の採り方は `_bi`)。既定 -1 は
    「plan を持たない文脈から呼ばれた」= 第93 の `plan_cont_fire` が添字を採れなかった
    ときと同じ値。src の実行経路では必ず `apply_contingency` 経由で真の添字が入る。
    """
    b["state"] = "dropped"
    _state(sim)["drop"] += 1
    _log(sim, agent, step, sim_min, "plan_block_drop",
         {"act": b["act"], "place": b["place"], "start": b["start"],
          "priority": b["priority"], "flex": b["flex"], "reason": reason,
          "block": int(block)})


def _then_skip(sim, agent, b: dict, cfg: dict, step: int, sim_min: int,
               block: int = -1) -> bool:
    _drop_block(sim, agent, b, step, sim_min, "cont_skip", block)
    return True


def _then_postpone(sim, agent, b: dict, cfg: dict, step: int, sim_min: int,
                   block: int = -1) -> bool:
    """postpone_min だけ後ろへずらす(既存 slide と同じ最小摂動の作法・継続は保つ)。"""
    dur = int(b["end"]) - int(b["start"])
    shift = int(cfg["postpone_min"])
    start = _round_to(int(b["start"]) + shift, cfg["round_min"])
    slid = int(b.get("slid", 0) or 0) + shift
    if slid > int(cfg["max_slide_min"]) or start + dur > int(cfg["day_end_min"]):
        _drop_block(sim, agent, b, step, sim_min, "cont_postpone", block)
        return True
    b["start"], b["end"], b["slid"] = start, start + dur, slid
    _state(sim)["slide"] += 1
    _log(sim, agent, step, sim_min, "plan_slide",
         {"act": b["act"], "place": b["place"], "start": b["start"],
          "slid": slid, "priority": b["priority"], "block": int(block)})
    return True


def _then_go_home(sim, agent, b: dict, cfg: dict, step: int, sim_min: int,
                  block: int = -1) -> bool:
    """行き先を自宅へ差し替える。自宅が無ければ削る(捏造しない)。"""
    node = str(getattr(agent, "home_node", "") or "")
    if not node:
        _drop_block(sim, agent, b, step, sim_min, "cont_no_place", block)
        return True
    if b["place"] == "home" and b.get("node") == node:
        return False                             # 既に自宅 = 空振り(世界は動かない)
    b["place"], b["act"] = "home", "home"
    b["purpose"] = ACT_PURPOSE["home"]
    b["node"] = node
    return True


def _then_swap_indoor(sim, agent, b: dict, cfg: dict, step: int,
                      sim_min: int, block: int = -1) -> bool:
    """屋外カテゴリを屋内へ差し替える。屋内ブロックなら空振り(applied=false)。"""
    place = str(b["place"])
    if place not in OUTDOOR_PLACES:
        return False                             # もう屋内 = 何もしない
    sub = INDOOR_SWAP[place]
    node = resolve_place(sim, agent, sub, sim_min,
                         from_node=getattr(agent, "node", ""))
    if node is None:                             # 屋内の受け皿が無ければ自宅へ落とす
        return _then_go_home(sim, agent, b, cfg, step, sim_min, block)
    b["place"], b["node"] = sub, node
    if b["act"] not in ACT_PLACE or ACT_PLACE[b["act"]] in OUTDOOR_PLACES:
        b["act"] = next((a for a, p in ACT_PLACE.items() if p == sub), b["act"])
        b["purpose"] = ACT_PURPOSE.get(b["act"], b["purpose"])
    return True


def _then_shorten(sim, agent, b: dict, cfg: dict, step: int, sim_min: int,
                  block: int = -1) -> bool:
    """滞在を shorten_min だけ短くする(min_dur_min は下回らない)。"""
    lo = int(cfg["min_dur_min"])
    end = max(int(b["start"]) + lo, int(b["end"]) - int(cfg["shorten_min"]))
    if end == int(b["end"]):
        return False                             # もう最小 = 空振り
    b["end"] = end
    return True


THEN_FN = {"skip": _then_skip, "postpone": _then_postpone,
           "go_home": _then_go_home, "swap_indoor": _then_swap_indoor,
           "shorten": _then_shorten}


def apply_contingency(sim, agent, plan: dict, b: dict, cfg: dict,
                      step: int, sim_min: int) -> bool:
    """`plan["cont"]` を評価し、最初に成立した 1 件だけ適用する(決定論)。

    返り値 = 発火したか。1 ブロックにつき高々 1 回(`cont_done` で固定)。
    """
    if b.get("cont_done"):
        return False
    for c in plan.get("cont") or ():
        cond, then = str(c.get("if") or ""), str(c.get("then") or "")
        fn, act_fn = COND_FN.get(cond), THEN_FN.get(then)
        if fn is None or act_fn is None or not fn(sim, agent, b, cfg, sim_min):
            continue
        b["cont_done"] = True
        # 同値の別ブロックを拾わないよう **同一性**で位置を採る(list.index は == 比較)。
        index = _bi(plan, b)
        # W4-F: 対処が出す plan_block_drop / plan_slide にも同じ添字を載せる
        # (= 同 step の 2 イベントが同じブロックを指していることが台帳再生を経ずに判る)。
        applied = bool(act_fn(sim, agent, b, cfg, step, sim_min, index))
        _log(sim, agent, step, sim_min, "plan_cont_fire",
             {"cond": cond, "then": then, "block": index, "act": b["act"],
              "place": b["place"], "start": int(b["start"]), "applied": applied})
        return True
    return False


def _clear_why(sim, agent) -> None:
    """第94 IF-B の**降格**(既定 silent では属性が無いので即 return = バイト一致)。"""
    from .. import reject as _reject
    _reject.clear_why(sim, agent)


def current_block(plan: dict, sim_min: int):
    """いま実行すべきブロック(開始済み・未消化・窓の中)。無ければ None = 空き時間。"""
    now = int(sim_min) % 1440
    for b in plan["blocks"]:
        if b["state"] == "todo" and b["start"] <= now < b["end"]:
            return b
    return None


def plan_action(agent, sim, sim_min: int, step: int, rng, scfg=None):
    """day_plan v1 の実行(routine.decide の朝計画スロットからの単一作用点)。

    返り値は routine と同じ action dict、または None(= 空き時間 → 既存の習慣ポリシーが埋める)。
    ★ 割り込み処理(_sweep)は乱数を 1 本も引かない。行き先が決まったときだけ、既存 routine と
      同じ順で `rng` を使う(滞在長 → 移動手段)= 既存の draw 作法に合わせる。
    """
    cfg = cfg_of(sim)
    plan = _plan_of(agent, sim_min)
    if plan is None:
        return None
    _sweep(sim, agent, plan, cfg, step, sim_min)
    b = current_block(plan, sim_min)
    if b is None:
        return None
    # ---- contingency の消費(第93 IF-A・既定 OFF=この 5 行は即素通り)------------ #
    #  「いま実行すべきブロックが決まった直後・行き先を確定させる前」= 唯一の作用点。
    #  適用の結果このブロックが消えた/窓から外れたら、この step は空き時間として扱う
    #  (= 既存の習慣ポリシーが埋める。世界を止めない第86 の流儀と同じ)。
    if cfg["use_contingency"] and plan.get("cont"):
        if apply_contingency(sim, agent, plan, b, cfg, step, sim_min):
            now = int(sim_min) % 1440
            if b["state"] != "todo" or not (b["start"] <= now < b["end"]):
                return None
    from . import routine as _routine
    dest = b["node"]
    # W4-F: 記録 2 欄。`block` = 台帳の添字、`node` = **この瞬間の現在ノード**(移動前)。
    #   node は「行き先」ではない。同 step に route_start が出なければ個体はここに
    #   留まったのだから、そのときに限りここがこのブロックの場所である
    #   (= 事後解析の no_move_event が「既に目的地に居た」ぶん消える)。
    #   どちらも読まれない=分岐に使わない=世界を 1 バイトも動かさない。
    at_node = str(getattr(agent, "node", "") or "")
    index = _bi(plan, b)
    # ---- P4 計画駆動の圏外滞在(既定 OFF=is_boundary が常に False=この節は即素通り)----
    #  ブロックの場所が圏外 = このブロックの実行は「縁へ行って街を出る」ことそのもの。
    #  退出の実行は既存の `_try_exit`、帰還は `_phase_wake_and_returns` が行う
    #  (**境界機構を新設しない**)。state="away" は「出かけたがまだ帰っていない」の意で、
    #  帰還時に "done" へ上がる(割り込み処理 _sweep は todo 以外を触らない)。
    if _bnd.enabled(sim) and _bnd.is_boundary(b):
        out = _bnd.begin(sim, agent, plan, b, index, rng)
        b["state"] = "away"
        _state(sim)["exec"] += 1
        _log(sim, agent, step, sim_min, "plan_block_start",
             {"act": b["act"], "place": b["place"], "aim": b["purpose"],
              "priority": b["priority"], "flex": b["flex"], "start": b["start"],
              "slid": int(b["slid"]), "version": int(plan["version"]),
              "node": at_node, "block": index, "boundary": "plan",
              "gateway": str(b["boundary"].get("gateway") or "")})
        _clear_why(sim, agent)                   # 第94 IF-B: 計画が動いた=侵入は止まる
        return out or None                       # {} = 既に縁に居る(移動不要)
    if b["place"] == "street":                   # 戸外・特定しない = 行き先は習慣ポリシーへ委ねる
        b["state"] = "done"                      # (消化そのものは起きたので黙って消さず数える)
        _state(sim)["exec"] += 1
        _log(sim, agent, step, sim_min, "plan_block_start",
             {"act": b["act"], "place": b["place"], "aim": b["purpose"],
              "priority": b["priority"], "flex": b["flex"], "start": b["start"],
              "slid": int(b["slid"]), "version": int(plan["version"]),
              "node": at_node, "block": index})
        _clear_why(sim, agent)                   # 第94 IF-B: 再計画が実った=侵入は止まる
        return None
    if dest is None or not _open_now(sim, b, sim_min):
        dest = resolve_place(sim, agent, b["place"], sim_min,
                             from_node=getattr(agent, "node", ""))
        b["node"] = dest
    if dest is None:
        b["state"] = "dropped"
        _state(sim)["drop"] += 1
        _log(sim, agent, step, sim_min, "plan_block_drop",
             {"act": b["act"], "place": b["place"], "start": b["start"],
              "priority": b["priority"], "flex": b["flex"], "reason": "no_place",
              "block": index})
        return None
    b["state"] = "done"
    _state(sim)["exec"] += 1
    _log(sim, agent, step, sim_min, "plan_block_start",
         {"act": b["act"], "place": b["place"], "aim": b["purpose"],
          "priority": b["priority"], "flex": b["flex"], "start": b["start"],
          "slid": int(b["slid"]), "version": int(plan["version"]),
          "node": at_node, "block": index})
    # 第94 IF-B(Masicampo & Baumeister 2011): **代替ブロックが動き出した時点**で
    # 失敗理由を降格する(= 計画が立てば侵入思考は止まる)。既定 silent は完全 no-op。
    _clear_why(sim, agent)
    if dest == getattr(agent, "node", ""):
        return None                              # もう居る = 移動しない(習慣ポリシーへ)
    stay = sim.clock.min_to_steps(max(cfg["min_dur_min"], b["end"] - b["start"]))
    base_stay = max(1, int(stay))
    if scfg is not None:                         # P2 S4: 寄り道・継続ジッター(専用 stream)
        detour = _routine._maybe_detour(agent, sim, dest, step, sim_min, scfg,
                                        what=b["act"])
        if detour is not None:
            dest = detour
        base_stay = _routine._jitter_stay(agent, sim, base_stay, step, scfg,
                                          what=b["act"])
    action = {"type": "move_to", "dest": dest, "stay_steps": base_stay,
              "mode": _routine._choose_mode(agent, sim, dest, rng)}
    activity = ACT_ACTIVITY.get(b["act"], "")
    if activity:
        action["activity"] = activity
    out = _routine._augment_ride(agent, sim, action, step, sim_min)
    # IF-1: この移動は「朝の計画」が決めたもの = その朝の LLM 呼へ辺を張る
    # (role は l1b_llm の purpose と同じ "plan")。OFF では plan に call_id が
    # 無いので stamp は即 return = 一時キーを積まない = バイト一致。
    from .. import provlink as _provlink
    _provlink.stamp(sim, out, plan.get("call_id"), "plan")
    return out


def _open_now(sim, b: dict, sim_min: int) -> bool:
    """ブロックの場所カテゴリが今この時刻に開いているか(commerce OFF なら常に True)。"""
    from .. import commerce as _commerce
    ccfg = getattr(sim, "commercecfg", None)
    if ccfg is None or not _commerce.enabled(sim):
        return True
    return _commerce.is_open(ccfg, b["place"], sim_min)
