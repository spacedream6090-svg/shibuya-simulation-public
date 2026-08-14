"""部門別会計行列 + 貨幣保存の検査(IF-E フェーズ1 = 検査。第95バッチ 2026-08-05)。

正典: docs/research/if-lane-research.md §5(Caiani, Godin, Caverzasi, Gallegati, Kinsella &
      Stiglitz 2016, *JEDC* 69:375-408 の検査法2本)/ docs/research/llm-world-interface-audit.md §3
      (経済会計の断絶: revenue_est = 日給×margin で客の spend と非接続)/ docs/plans/if-sv-p4-plan.md IF-E。

Caiani et al. (2016) §5.2 Validation の2検査をそのまま移植する:

  検査① **貨幣ストックの保存** — 「各部門の期首残高 + 流入 − 流出 = 期末残高」。
        本シムで残高が**観測できる**部門は 家計(L3 スナップショットの money+account)と
        行政(public_budget の balance)だけ。残りは残高そのものが世界に無いので
        「不可観測」と正直に開示する(ゼロと偽らない)。
  検査② **取引フロー行列のゼロ和** — 「あらゆる時点で行列のすべての行と列がゼロに合計される」。
        本シムのイベントは**片側記入**(客が払った金の受け取り手が世界に居ない)なので、
        本スクリプトは相手方を分類器で明示的に割り当て、相手が存在しない金を専用部門
        `void`(未接続)へ落とす。**void 行・列の絶対額 = 貨幣の創出/消滅 = 漏れ(leakage)**。

Caiani et al. 本文が「大規模で複雑な AB モデルで実装中の漏れ(leakage)が生じることは珍しくない」と
明言しているとおり、これは本シム固有の不手際ではなく分野で標準的に検査される既知の失敗モードである。

読み取り専用
------------
本スクリプトは runs/<name>/ を読むだけで、シム本体(src/society)を一切呼ばない・変更しない。
乱数を引かず LLM も呼ばない(R1 適合が自明)。依存は 標準ライブラリ + pyarrow +
**同ディレクトリの `l1_stream.py`**(W2-2 で新設した共有ストリーミング読み)のみ(pandas 禁止)。

メモリの上限(W2-6 ②。在場 25万 × 10 日への対応。第98バッチ 2026-08-07)
-------------------------------------------------------------------------
在場 25万 × 10 日のランは L1 が 42.7 GB・40.6 億イベント、L3 が 10 日 ×(1日12枚)=
120 枚 × 25万体 のスナップショット(`proposal-dp-u3-observe-250k.md` §2-4)。
**全読みしている箇所を潰した**。何が有界になり、何が残るかを正直に書く。

  有界化したもの(行数・体数に**比例しなくなった**)
    - `l3_snapshots` … 旧: `read_table` で全 step の state 文字列を載せ、さらに
      `stock_by_step = {step: {agent_id: 残高}}` を **全 step ぶん**保持していた
      (120 枚 × 25万体 = 3,000 万エントリ)。新: `SnapshotStocks` が 1 枚ずつ
      json.loads し、検査①(`conserve_household`)は**隣り合う 2 枚だけ**を保持して
      窓を進む。ピーク = **スナップショット 2 枚 = O(在場人数)**。
      (2 枚必要なのは検査式そのものの要請: 窓の d_common が「両方の枚に居る個体」の
       差の総和で、集合演算に個体別の値が要る。1 枚にはできない。)
    - 未分類の金の経路の検知 … 旧: 該当イベントを list に貯めてから数えていた。
      新: `MoneyKindTally` が走査中に畳む。**O(イベント種類数)**(実測 100 前後)。
    - `org_ledger` / `finance` … 旧: `read_table().to_pylist()` で全列全行。
      新: **列射影 + チャンク読み**(`l1_stream.iter_table_columns`)。ledger は
      検査が使う 4 列だけ(`LEDGER_COLUMNS`)。行数自体は O(org 数 × 日数) で残るが、
      これは 25万体でも 11,010 社 × 10 日 = 11 万行で、L1/L3 とは桁が違う。
    - L1 の読み出し列 … `x` / `y` を**読むのをやめた**(会計の検査は 1 度も使わない。
      serve↔spend の突合が座標を使えないことは `match_serve_to_spend` の docstring 参照)。
      40.6 億行 × float32 2 本 = 約 32 GB の復号が丸ごと消える。
    - part 群への対応 … canonical が無い(走行中・クラッシュ)ランでも
      `l1_stream.l1_paths` 経由で完結 part を読む。

  残るもの(**O(金額を運ぶイベント数)**。25万体でも消せない理由つき)
    - `run["events"]` … 金額を運ぶ L1 イベントだけ(全 L1 の一部)を dict で保持する。
      **1 パス化できない**のは `resolve_tax_sources` が tax の相手方を「同 step・同 agent の
      spend / wage との突合」で決めており、その結果を **payload dict の同一性 `id(payload)`**
      で `flows_for` へ渡す設計だから(= 突合の前にフローを確定できない)。ここを直すには
      `flows_for` の呼び出し規約を変える必要があり、それは検査式の意味の変更に触れる。
    - `moves_by_step` / `flows` … 上の events から作る派生列。行列・漏れ内訳・部門別
      検査はすべて**集約**なので原理的には畳めるが、`conserve_sector` /
      `conserve_government` / `classifier_consistency` が `list[Flow]` を受ける公開関数
      として既にテストで固定されているため、本バッチでは畳まない(二重実装を作らない)。
    - `serve` / spend の突合表 … `match_serve_to_spend` が全 spend の索引を要求する。
    ★運用上の含意: 25万体ランでこのスクリプトを回す前に、必要なら
      `--rel-tol` だけでなく**日を絞ったサブセット**で回すこと。L3 と ledger と未分類検知は
      有界になったので、残るのは「金額イベント数に比例する分」だけである。

部門(6+1)
----------
Caiani et al. の行列も**部門別**(家計・消費財企業・資本財企業・銀行・政府・中央銀行)であり、
25万体でエージェント別 N×N 行列は不可能。本シムは以下の 6 部門 + 未接続枠 1 で足りる。

  household  家計   = エージェント(住民+来街者)。現金 money + 口座 account。
                      ★第109バッチ D2: **遺失物バケツ**(H3 lost_property の hold = 落とした
                      財布の中の現金。街の中に在るが誰の残高でもない)もこの部門の内訳
                      (通過勘定)として扱う。供託金 escrow を行政の内訳に置くのと同じ流儀で、
                      **新しい部門は作らない**(SECTORS を増やすと全ランの sector_sums の
                      形が変わる)。詳細は LOST_BUCKET_SECTOR / LOST_BUCKET_ID の宣言。
  org        企業   = 組織台帳(organizations / org_ledger)。
  venture    個人事業 = 屋台(tools.ventures)。売上は店主(家計)へ抜ける通過部門。
  bank       銀行   = 預金利息・融資・VC ファンド。
  government 行政   = 区(ward)/都(metro)/国(nation)の予算。
  external   街外   = **設計上明示的に**街の外(来街者の財布補充・偶発事の窓口)。
  void       未接続 = 相手方が世界に存在しない金。**これが漏れの正体**。

使い方
------
    python scripts/analyze_accounting.py runs/night_llm_100a3d
    python scripts/analyze_accounting.py runs/foo --out runs/foo/accounting.md --json runs/foo/accounting.json
    python scripts/analyze_accounting.py runs/foo --rel-tol 1e-4      # 検査①の相対許容誤差

実測(2026-08-05 第95バッチ・フェーズ1の結論)
-----------------------------------------------
| ラン | 総フロー | 漏れ比率 | 漏れの族の内訳 |
|---|---:|---:|---|
| mock 40体2日(organizations ON) | 2,504,269 | **96.0 %** | org 56.4 / wage 21.5 / spend 12.8 / venture_setup 5.0 / tax 4.3 |
| 同(accounts+bank ON)          | 1,911,566 | **90.8 %** | org 78.2 / spend 17.0 / wage 2.5 / tax 1.8 / rent 0.4 |
| night_llm_100a3d(実LLM・org OFF)| 1,626,893 | **71.8 %** | spend 74.3 / wage 16.8 / tax 8.9 |

一方で **家計の貨幣ストックは保存している**(相対残差 4.0e-07 / 8.9e-18 / 3.3e-07 = 0.1円丸めの
ノイズのみ)。つまり「個体の財布の中では金は消えていない」が、「財布から出た金の受け取り手が
世界に存在しない」。これが本シムの漏れの正体である。

**フェーズ2(接続)は本バッチでは実装しない。** 判断の根拠は上の実測3点:
  (1) 漏れは `revenue_est` の断絶に**一意に帰着しない**。organizations OFF のランでは org 族は
      0% で、漏れの 74% は「客の払った金を受け取る店舗が会計主体でない」ことに由来する。
  (2) org ON のランでも org 族 56% に対し他4族が 44% を占める(独立した経路が5本ある)。
  (3) 決定的な点として、`revenue_actual` 列を org_ledger に足しても**残高は1円も動かない**
      (org は残高そのものを持たない)。すなわち計画の受入条件「金の保存則が接続後に改善する
      ことをテストで固定」は列の追加では原理的に満たせない。保存を閉じるには org へ残高を
      新設し、支払不能時の規則(倒産/借入)まで決める必要があり、これは設計判断を要する。

**選択肢(数値つき。採否はユーザー判断)**
  - 案A 観測のみ: `org_ledger.revenue_actual` を追加(本スクリプトの突合と同じ規則)。
    保存の改善 = 0 円。乖離(下表)を運用中に可視化できるだけ。工数=小。
  - 案B org を会計主体化: org に残高を新設し、①org 帰属店舗の spend を入金 ②賃金を残高から
    支出 ③枯渇時の規則を決める。閉じる漏れ = org 族 + wage 族 + spend 族の org 帰属分
    (mock org ON ラン換算で漏れの 56.4 + 21.5 + α %)。新状態=checkpoint/resume/golden への
    波及あり。設計判断が必要。
  - 案C 部門宣言: 家主・仕入先・非 org 店舗を **rest-of-world(街外)部門**として設計上明示し、
    「街 + 街外」で保存が閉じると宣言する(Caiani の外国部門と同型)。シム改変ゼロ・
    本スクリプトの写像を変えるだけ。ただし「街の中で経済が回っていない」事実は変わらない。

IF-E2(第97バッチ 2026-08-06)= **案B を採用して実装済み**(`economy.org_accounting`・既定 OFF)
--------------------------------------------------------------------------------------------
実装 `src/society/economy_sfc.py` / 設計 `docs/research/ifE2-org-accounting-research.md` §4-3。
org が**スカラー預金 1 本**を持ち(Lengnick 2013 の M=wN / Caiani の σ×予想賃金支払で初期化)、
賃金はそこから出て(不足は**自動当座借越**)、消費は台帳の静的索引で解決した受け手 org へ
**実支払−消費税**として入る。受け手が街に居ない金は rest-of-world(渋谷域外)へ
**チャネル別に**落ちる(SNA 2008 §26.2/26.5/26.6 = 行動方程式なし・残高制約なしの明示部門)。

本スクリプトは ON のランで payload の相手方(`spend.payee` / `wage.payer` / `rent.payee` …)と
サイドカー `finance.parquet`(部門別残高の日次 1 行)を読み、検査①を**残高を持つ全部門**で行う。

| ラン(mock 40体2日・同一 conf の ON/OFF) | 総フロー | 漏れ比率 | 漏れの族 |
|---|---:|---:|---|
| organizations ON / org_accounting **OFF** | 2,504,269 | **96.04 %** | org 56.4 / wage 21.5 / spend 12.8 / venture_setup 5.0 / tax 4.3 |
| 同 / org_accounting **ON**                 | 1,812,769 | **0.00 %**  | (なし。void 行・列とも 0) |
| accounts+bank ON / org_accounting **OFF**  | 1,845,937 | **92.01 %** | org 76.4 / spend 18.7 / wage 2.6 / tax 1.9 / rent 0.4 |
| 同 / org_accounting **ON**                 | 1,178,437 | **0.00 %**  | (なし) |

検査①は household / org / bank / government / **rest-of-world** の 5 部門すべてで PASS
(相対残差 8.98e-18 / 6.66e-08 / 0.0 / 0.0 / 3.80e-06 = payload の 0.1 円丸めノイズ)。
**venture と void は原理的に observable にならない**ので、そう表示し続ける:
venture は売上が即座に店主=家計へ抜ける**通過部門**で世界に残高そのものが無く、
void は「相手方が存在しない金」の**検出器**である(装置は残したまま値をゼロにするのが、
検査の網羅性を将来にわたって守る唯一の方法)。

**ゼロと偽らない — ON でも残る 3 つの限界**
  1. **RoW への依存はむしろ可視化された**。漏れが 0 になったのは金が閉じたからで、
     街の経済が自立したからではない。mock ラン実測では受け手 org を特定できなかった消費
     `unknown_payee` が 231,833 円、域内に客が居ない org の輸出代金 `export_production` が
     666,000 円(org が受け取った域内消費 17,318 円の **38 倍**)。これが案B 固有の主結果。
  2. **受け手の特定率**。台帳の静的索引 `(building, floor, POI種別)` → `(node, POI種別)` で
     解決し、決まらない消費は RoW へ落とす。小ラン(台帳 42 社)では 15/299 件しか org へ
     着地しない。研究文書 §3-c の全数測定では 11,010 社台帳で floor 鍵の一意率 56.8%。
  3. ~~接続できていない金の経路が 4 つ残る~~ → **第98バッチ IF-E2 UNCOVERED で閉じた**(下記)。

IF-E2 UNCOVERED(第98バッチ 2026-08-07)= 残り 4 種の接続
--------------------------------------------------------
`economy_sfc.UNCOVERED_KINDS` が**空になった**。ON のときだけ・動力学は 1 つも変えずに分類を足す:

  - `rule_bonus`  区(ward)予算からの歳出(payload `payer="government"`)。区の残高は負に振れうる。
  - `crime`(窃盗)**取引ではない**(SNA 2008 §3.98 逐語: "If thefts, or acts of violence
    (including war), involve significant redistributions, or destructions, of assets, it is
    necessary to take them into account. As explained below, **they are treated as other flows,
    not as transactions**.")。よって RoW(= 街の外の取引相手)ではなく、SNA 2008 第12章
    *The other changes in assets accounts*(K.5 = other changes in volume n.e.c.)に相当する
    新部門 `other_volume` へ落とす(payload `payee="k5:theft"`)。**加害者へは入金しない**
    (= 世界がまだ記録していない受け取り側を、この勘定が持っている)。
  - `chance_event` RoW の外生チャネル `chance_windfall` / `chance_loss`。
  - `b2b_trade`   org 預金間の**実移転**。買い手の小売 POI を台帳で特定できなければ
    `b2b_buyer_unknown`(RoW → 街)から払う(`unknown_payee` の裏返し)。

これで総マネー保存は **Σ(全主体残高) + RoW 累積 + K5 累積 = 一定** に拡張された
(`economy_sfc.total_money`)。`void` 部門は**残す**(値をゼロにするだけ)。

正直な限界(出力にも印字する)
------------------------------
- **許容誤差はゼロにできない。** L1 payload も L3 スナップショットも 0.1 円単位に丸められており、
  実際の残高更新は丸めない値で行われる(例 interest_paid は payload が小数2桁、加算は無限精度)。
  よって判定は**絶対誤差ではなく相対誤差**(総フローに対する比)で置き、閾値は引数化する
  (if-lane-research.md §5-3 の設計示唆 4)。
- 部門別残高が観測できるのは household と government のみ。bank.capital / venture 在庫 /
  org の内部留保は L1・L3 のどこにも出ないので検査①は「不可観測」と表示する。
- 相手方の割り当ては**本スクリプトの分類器**が行う(世界のイベントは片側記入)。分類器が漏らした
  経路は検査①の残差として必ず現れる。検査①=分類器の網羅性の独立検証、という関係になっている。
- pool.enabled=true(日次 presence 選抜)のランでは在場エージェント集合が入れ替わるため、
  L3 の合計残高が個体の出入りで跳ぶ。この分は `pool_rotation` として分離表示する(漏れではない)。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:      # `python scripts/...` 以外の経路から import
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

# W2-2 の共有ストリーミング読み(列射影・チャンク・part 群の透過連結・Windows の
# FILE_SHARE_DELETE)。ここを自前で書き直すと車輪の再発明になるので必ず借りる。
import run_dt                                        # noqa: E402  (W2-3: Δt の単一の源)
from l1_stream import iter_table_columns, l1_paths   # noqa: E402

# --------------------------------------------------------------------------- #
# 部門
# --------------------------------------------------------------------------- #
HOUSEHOLD = "household"
ORG = "org"
VENTURE = "venture"
BANK = "bank"
GOVERNMENT = "government"
EXTERNAL = "external"
#: 第98バッチ IF-E2 UNCOVERED: **取引ではない**資産変動の勘定(SNA 2008 第12章
#: *The other changes in assets accounts* / K.5 = other changes in volume n.e.c.)。
#: §3.98「窃盗・暴力による資産の再分配や破壊は **other flows であって transactions ではない**」に従い、
#: 窃盗を EXTERNAL(=街の外に居る取引相手)へ落とさずここへ分類する。混ぜると「窃盗は輸入だった」
#: という嘘になる。残高は economy_sfc の K5 アキュムレータ(finance.parquet の k5_other)で観測できる。
OTHER = "other_volume"
VOID = "void"

SECTORS: tuple[str, ...] = (HOUSEHOLD, ORG, VENTURE, BANK, GOVERNMENT, EXTERNAL,
                            OTHER, VOID)

SECTOR_JA = {
    HOUSEHOLD: "家計", ORG: "企業(org)", VENTURE: "個人事業", BANK: "銀行",
    GOVERNMENT: "行政", EXTERNAL: "街外", OTHER: "非取引(SNA K.5)",
    VOID: "未接続(漏れ)",
}

#: 残高が観測できない理由(正直開示用)。
UNOBSERVABLE_REASON = {
    ORG: "org_ledger は revenue_est/wage_paid の集計のみで残高列を持たない",
    VENTURE: "屋台は sales_total(累計売上)だけを持ち残高を持たない=通過部門",
    BANK: "Bank.capital / VCFund は L1・L3 のどこにも出力されない",
    EXTERNAL: "街の外=定義上の無限の源/シンク(残高の概念を置いていない)",
    OTHER: "非取引の資産変動(SNA K.5)= finance.parquet が無いと観測できない",
    VOID: "相手方が存在しない=残高の器そのものが無い(これが漏れの定義)",
}

#: 金額を運ぶ L1 イベント種(payload に金が乗る種のみ。schema.py の 189 種から抽出)。
MONEY_KINDS: frozenset[str] = frozenset({
    "spend", "wage", "withdraw", "rent", "interest_paid", "loan_grant", "loan_repay",
    "venture_sale", "venture_open", "deposit", "candidacy", "reward", "rule_bonus",
    "civic_service", "chance_event", "crime", "bankruptcy", "vc_investment",
    "enforcement", "tax", "b2b_trade", "move_home",
    # IF-E2 案B: 域内に客が居ない org(office/education)への**輸出代金**が
    # production の payload に revenue として載る(ON のときだけ。OFF はキーなし=金額ゼロ)。
    "production",
    # ---- H2 医療(medical.enabled。既定 OFF のランには 1 件も出ない)----
    # ems_transport = 救急搬送の公費(区 → RoW ems_operation)。cost に載る。
    # medical_bill  = 保険給付 7 割(RoW insurance_reimbursement → 医療機関 org)。
    #                 **amount = 実際に動いた額**で、受け手 org を特定できなかった回は 0
    #                 (保険者も医療機関も街の外 = 街の残高は 1 円も動かない)。
    #                 自己負担 3 割は spend(cat="medical")側で会計済み = ここでは数えない。
    "ems_transport", "medical_bill",
    # ---- H3 遺失物ループ(lost_property.enabled。既定 OFF のランには 1 件も出ない)----
    # lost_return = 返還。payload の cash(遺失物バケツ → 落とし主)と amount(報労金 =
    #               落とし主 → 拾得者)の**2 本**が同じ 1 行に載る。
    # lost_keep   = 着服(占有離脱物横領)。amount = 財布の中の現金が拾得者の手に渡った額。
    # lost_expire = 時効取得(to="finder")または失効(to="void")。前者は拾得者の手に渡り、
    #               後者は受け取り手が世界に居ない = **取引ではない**(SNA K.5)。
    # lost_drop   = 入口。財布の中の現金が落とした本人の所持金から分離され、遺失物バケツへ
    #               移る(payload の cash)。**部門をまたがない**ので flows_for は 1 本も
    #               立てないが、家計個体の残高は実際に減る = withdraw(現金⇄口座の家計内部
    #               振替)と同じ形。よって DERIVED(二重計上の禁止リスト)ではなくここに置く。
    # ★どれも「街の中に在るが誰の残高でもない現金」(lost_property の hold バケツ)の
    #   出入口である。
    "lost_drop", "lost_return", "lost_keep", "lost_expire",
})

#: 金額は payload に出るが**別のイベントで既に会計済み**の種(二重計上の禁止リスト)。
#: これらは _spend / _pay_wage を経由するので spend / wage 側でだけ数える。
DERIVED_MONEY_KINDS: frozenset[str] = frozenset({
    "service_use",      # services.charge_service → _spend(cat=service)
    "medical_visit",    # scheduler:4278 → _spend(cat="medical")
    "lodging_checkin",  # scheduler:4453 → _spend(cat="lodging")
    "order", "deliver",  # delivery.tick → _spend(注文者) + _pay_wage(配達員 source="gig")
    "free_action",      # scheduler:3711 → _spend(cat=f"free_{category}")
    "ride",             # _charge_ride: ride を記録した直後に _spend(cat=taxi/bus)
    "public_budget",    # 行政の残高観測(フローではない)
    "eviction",         # 立退き: arrears(滞納**残高**)を載せるが金は動かない
    # ---- IF-E2 案B(economy.org_accounting ON のランでのみ現れる)----
    "org_overdraft",    # org の預金が負に落ちた通知。金は wage 側で既に会計済み
    "row_flow",         # その日の域外収支の**日次集計**。個々のフローは spend/wage 側で会計済み
    # ---- H2 医療(medical.enabled)----
    "hospital_admit",     # 入院の開始(在院という状態。金額キーを持たない)
    "hospital_discharge",  # 退院。入院費は同じ step の spend / medical_bill 側で会計済み
})

#: 遺失物バケツ(``lost_property`` の hold)を**どの部門の内訳とみなすか**の宣言。
#:
#: 落とした財布の中の現金は「街の中に在るが誰の残高でもない」= 供託金(escrow)と同じ
#: 「主体を持たない域内残高」で、``economy_sfc.city_total`` には算入されるが
#: ``household_balance`` には入らない。部門行列には**新しい部門を作らない**:
#:   - 新部門を足すと ``SECTORS`` が変わり、全ランの ``sector_sums`` の形が変わる
#:     (既存の解析出力の互換を壊す)。
#:   - 経済的な実体としても、この金は「家計が落とし、家計へ戻る(または K5 へ消える)」
#:     だけで、街の外や企業を 1 度も経由しない。
#: したがって **escrow を行政の内訳として扱うのと同じ流儀**で、遺失物バケツを家計部門の
#: 内訳(通過勘定)として扱う。結果:
#:   - 落下(lost_drop)= 家計内部の資産振替 → **フローを立てない**(withdraw と同型)
#:   - 返還 / 着服 / 時効取得 = 家計 → 家計(部門内の移転。行と列で相殺される)
#:   - 失効(誰にも拾われず消える)= 家計 → K5(``economy_sfc.on_lost_lapse`` と同じ勘定)
LOST_BUCKET_SECTOR: str = HOUSEHOLD

#: 遺失物バケツを家計部門の**擬似メンバー**として表すときの agent_id(L1 の
#: 「世界イベント」と同じ -1 のセンチネル)。``move_for`` がこの id で受け払いを記録する。
#: L3 スナップショットには当然この id が居ないので ``conserve_household``(検査①)は
#: 自動的に無視し、全 move を足す ``classifier_consistency`` だけがこれを見る。
LOST_BUCKET_ID: int = -1


# --------------------------------------------------------------------------- #
# 1. フロー分類器 — 誰から誰へいくら(検査②の材料)
# --------------------------------------------------------------------------- #
class Flow:
    """部門間フロー1本(src → dst に amount)。tag は漏れの内訳キー。"""

    __slots__ = ("src", "dst", "amount", "tag", "step", "kind")

    def __init__(self, src: str, dst: str, amount: float, tag: str,
                 step: int = -1, kind: str = ""):
        self.src, self.dst, self.amount, self.tag = src, dst, float(amount), tag
        self.step, self.kind = int(step), str(kind)

    def __repr__(self) -> str:                                   # pragma: no cover
        return f"Flow({self.src}->{self.dst}, {self.amount}, {self.tag!r})"


class Move:
    """家計(個体)の残高移動1本。cash=現金 / acct=口座。合計 total が検査①の入力。"""

    __slots__ = ("agent_id", "cash", "acct", "kind")

    def __init__(self, agent_id: int, cash: float = 0.0, acct: float = 0.0,
                 kind: str = ""):
        self.agent_id, self.cash, self.acct, self.kind = int(agent_id), float(cash), float(acct), str(kind)

    @property
    def total(self) -> float:
        return self.cash + self.acct

    def __repr__(self) -> str:                                   # pragma: no cover
        return f"Move({self.agent_id}, cash={self.cash}, acct={self.acct}, {self.kind!r})"


def _f(payload: dict, key: str, default: float = 0.0) -> float:
    try:
        v = payload.get(key, default)
        return float(v) if v is not None else float(default)
    except (TypeError, ValueError):
        return float(default)


def party_sector(token) -> str | None:
    """IF-E2 案B の payload トークン(payer / payee)→ 部門。無ければ None(= 従来の写像へ)。

    ON のランでは spend / wage / rent / venture_open / move_home / enforcement /
    bankruptcy / deposit / candidacy / reward / rule_bonus / crime / chance_event /
    b2b_trade の payload に相手方が 1 語で載る:
      "row:<channel>" = rest-of-world(渋谷域外)の窓口 / "k5:<kind>" = 非取引の資産変動
      (SNA K.5。第98バッチ)/ "venture" = 屋台 / "government" = 行政 /
      "escrow" = 行政の預り金 / それ以外 = 台帳の org_id。
    OFF のランでは 1 つも載らないので本関数は常に None を返す(= 従来の分類のまま)。"""
    t = str(token or "")
    if not t:
        return None
    if t.startswith("row:"):
        return EXTERNAL
    if t.startswith("k5:"):
        return OTHER
    if t == "venture":
        return VENTURE
    if t in ("government", "escrow"):
        return GOVERNMENT
    # ★O4 権利行(第114): 個人家主への家賃。受け手が**街の家計**なので household へ。
    #   "household:<agent_id>" の形で、id は誰が受け取ったかの追跡用(部門は接頭辞で決まる)。
    if t == "household" or t.startswith("household:"):
        return HOUSEHOLD
    return ORG


def spend_destination(cat: str, payee=None) -> tuple[str, str]:
    """消費 spend の受け取り部門と漏れタグ。

    戻り値 (dst, tag)。dst==VOID は「客が払った金を受け取る主体が世界に居ない」= 漏れ。
    IF-E2 案B(economy.org_accounting ON)では payload の payee が受け手を 1 語で
    与えるので、それを最優先する(= spend 族の漏れが構造的に消える)。OFF は従来どおり
    venture だけが世界に受け皿を持つ。"""
    c = str(cat or "")
    sec = party_sector(payee)
    if sec is not None:
        return sec, f"spend:{c}" if c else "spend:?"
    if c == "venture":
        return VENTURE, "venture_purchase"
    return VOID, f"spend:{c}" if c else "spend:?"


#: wage の source → 支払い部門(と漏れタグ)。
#: gig/salary/severance/None は**どの主体の残高も減らさずに**家計へ入金される=貨幣の創出。
WAGE_SOURCE_SECTOR: dict[str, tuple[str, str]] = {
    "civil":       (GOVERNMENT, "wage:civil"),        # gov.expense(fund_level, gross) で実際に歳出
    "home_refill": (EXTERNAL,   "wage:home_refill"),  # 来街者が街の外の家で財布を補充=設計上の外生
    "gig":         (VOID,       "wage:gig"),          # 自営の日銭。客が居ない
    "salary":      (VOID,       "wage:salary"),       # 月給。org の残高は減らない
    "severance":   (VOID,       "wage:severance"),    # 退職金。org の残高は減らない
    # ---- 賃金多様性 WAGE(第112)。org_accounting ON では payer が上書きするので実質 org 払い ----
    # ★漏れタグの**族**は "wage" のままなので、監視テストが見ている族の集合は増えない
    #   (leak_family("wage:bonus") == "wage")。新しい金の経路ではなく既存 wage 経路の内訳。
    "daily":       (VOID,       "wage:daily"),        # 日給者の 1 日ぶん
    "bonus":       (VOID,       "wage:bonus"),        # 賞与(夏・冬)
    # ---- L5 役割職(第114 レーン 1a)----
    # 議員報酬は区の一般会計から出る(`fund_level="ward"` で実際に gov.expense 済み)ので
    # "civil" と同じ GOVERNMENT。タクシー・配信者の歩合は既存の "gig" をそのまま使う
    # (自営の日銭と同じ会計上の意味 = 客が居ない VOID)ので、この表に足す行は 1 行だけ。
    "stipend":     (GOVERNMENT, "wage:stipend"),      # 議員報酬(自治法 203 条の月額)
}
_WAGE_DEFAULT = (VOID, "wage:work")                   # source なし = 本業/バイトの勤務完遂

#: 家賃の受け手部門 → 漏れタグ(第114 O4)。族は全部 "rent"(監視の族集合は増えない)。
_RENT_TAG: dict[str, str] = {
    EXTERNAL:  "rent:row_landlord",        # 域外の不在家主(IF-E2 の従来経路)
    ORG:       "rent:org_landlord",        # 域内の不動産 org(O4)
    HOUSEHOLD: "rent:household_landlord",  # 域内の個人家主(O4)
    VOID:      "rent:no_landlord",         # ★受け手が世界に居ない = 本当の漏れ
}


def flows_for(kind: str, payload: dict, ctx: dict) -> list[Flow]:
    """1イベント → 部門間フローのリスト(0本以上)。

    ctx は run 全体の文脈: {"government_on": bool, "tax_src": {(step,agent,base_key): sector}}。
    tax の相手方(消費税=販売側 / 源泉税=雇用側)は build() の突合で先に解決しておく。"""
    p = payload or {}
    out: list[Flow] = []
    if kind == "spend":
        # IF-E2 ON で床クリップ(所持金 < 名目価格)が起きた spend だけ paid が載る。
        # 実際に動いた額は paid なので、あるときはそちらを使う(無ければ名目 amount)。
        amt = _f(p, "paid", _f(p, "amount"))
        dst, tag = spend_destination(p.get("cat"), p.get("payee"))
        if amt:
            out.append(Flow(HOUSEHOLD, dst, amt, tag))
    elif kind == "wage":
        amt = _f(p, "amount")                      # 手取り(行政 ON 時は gross > amount)
        src, tag = WAGE_SOURCE_SECTOR.get(str(p.get("source") or ""), _WAGE_DEFAULT)
        sec = party_sector(p.get("payer"))         # IF-E2: 支払側が payload に載る(OFF は None)
        if sec is not None:
            src = sec
        if amt:
            out.append(Flow(src, HOUSEHOLD, amt, tag))
    elif kind == "withdraw":
        pass                                        # 現金⇄口座=家計内部の資産振替(行列に出さない)
    elif kind == "rent":
        paid = _f(p, "paid")
        if paid:
            dst = party_sector(p.get("payee")) or VOID      # IF-E2: 不在家主 → RoW
            # ★漏れタグは**受け手の種類**で分ける(第114 O4)。族はどれも "rent" のままなので
            #   監視テストが見ている族の集合は増えない(leak_family("rent:org_landlord")=="rent")。
            #   VOID のときだけが本当の漏れ = 受け手が世界に居ない状態である。
            out.append(Flow(HOUSEHOLD, dst, paid, _RENT_TAG.get(dst, "rent:no_landlord")))
    elif kind == "interest_paid":
        amt = _f(p, "amount")
        if amt:
            out.append(Flow(BANK, HOUSEHOLD, amt, "interest"))
    elif kind == "loan_grant":
        amt = _f(p, "amount")
        if amt:
            out.append(Flow(BANK, HOUSEHOLD, amt, "loan_grant"))
    elif kind == "loan_repay":
        amt = _f(p, "amount")
        if amt:
            out.append(Flow(HOUSEHOLD, BANK, amt, "loan_repay"))
    elif kind == "venture_sale":
        amt, div = _f(p, "amount"), _f(p, "dividend")
        if amt - div:
            out.append(Flow(VENTURE, HOUSEHOLD, amt - div, "venture_income"))
        if div:
            out.append(Flow(VENTURE, BANK, div, "vc_dividend"))
    elif kind == "venture_open":
        cost = _f(p, "cost")
        if cost:
            dst = party_sector(p.get("payee")) or VOID      # IF-E2: 域外の内装業者・仕入
            out.append(Flow(HOUSEHOLD, dst, cost, "venture_setup_cost"))
    elif kind == "move_home":
        dep = _f(p, "deposit")
        if dep:
            dst = party_sector(p.get("payee")) or VOID      # IF-E2: 不在家主 → RoW
            out.append(Flow(HOUSEHOLD, dst, dep, "move_home:deposit_no_landlord"))
    elif kind == "vc_investment":
        amt = _f(p, "amount")
        if amt:
            out.append(Flow(BANK, HOUSEHOLD, amt, "vc_investment"))
    elif kind == "deposit":
        amt, phase = _f(p, "amount"), str(p.get("phase") or "")
        esc_in = party_sector(p.get("payee"))       # IF-E2: 行政の預り金(OFF は None=VOID)
        esc_out = party_sector(p.get("payer"))
        if amt and phase == "paid":                 # 供託=預かり金
            out.append(Flow(HOUSEHOLD, esc_in or VOID, amt, "deposit:escrow_in"))
        elif amt and phase == "refund":
            out.append(Flow(esc_out or VOID, HOUSEHOLD, amt, "deposit:escrow_out"))
        elif amt and phase == "forfeit":
            # IF-E2 ON: 預り金 → 区の歳入 = **行政の内部振替**(行列に出さない。civil 源泉税と同型)。
            # 行政が居ない世界だけ payee(RoW)へ出る。OFF は従来どおり VOID → 行政。
            if esc_out is None:
                out.append(Flow(VOID, GOVERNMENT, amt, "deposit:forfeit"))
            elif esc_in is not None and esc_in != esc_out:
                out.append(Flow(esc_out, esc_in, amt, "deposit:forfeit"))
        # phase == "insufficient" は金が動かない(拠出不能)
    elif kind == "candidacy":
        dep = _f(p, "deposit")
        if dep:
            dst = party_sector(p.get("payee")) or VOID      # IF-E2: 行政の預り金
            out.append(Flow(HOUSEHOLD, dst, dep, "candidacy:escrow_in"))
    elif kind == "reward":
        amt = _f(p, "amount")
        if amt:                                     # D9 ablation の外生報酬(IF-E2 ON でも RoW)
            out.append(Flow(party_sector(p.get("payer")) or EXTERNAL,
                            HOUSEHOLD, amt, "reward"))
    elif kind == "rule_bonus":
        amt = _f(p, "amount")
        if amt:
            # OFF: 区が「発行」するだけで行政予算は減らない(無からの創出)。
            # IF-E2 UNCOVERED(第98): payer=government = 区(ward)予算からの実際の歳出。
            src = party_sector(p.get("payer"))
            out.append(Flow(src or VOID, HOUSEHOLD, amt,
                            "rule_bonus:ward_expense" if src else "rule_bonus:unfunded"))
    elif kind == "civic_service":
        amt = _f(p, "amount")
        if amt:
            out.append(Flow(GOVERNMENT, HOUSEHOLD, amt, "civic_benefit"))
    elif kind == "chance_event":
        # IF-E2 UNCOVERED(第98): ON では RoW のチャネル(chance_windfall / chance_loss)が
        # payload に載る。OFF でも「設計上の外生」= EXTERNAL 扱いは従来どおり(分類は不変)。
        amt, t = _f(p, "amount"), str(p.get("type") or "")
        if amt and t == "windfall":
            out.append(Flow(party_sector(p.get("payer")) or EXTERNAL,
                            HOUSEHOLD, amt, "chance:windfall"))
        elif amt and t == "loss":
            out.append(Flow(HOUSEHOLD, party_sector(p.get("payee")) or EXTERNAL,
                            amt, "chance:loss"))
    elif kind == "crime":
        amt = _f(p, "amount")
        if amt and str(p.get("kind") or "") == "theft":
            # OFF: 加害者は受け取らない = 相手方が世界に無い(漏れ)。
            # IF-E2 UNCOVERED(第98): payee="k5:theft" = **取引ではない**資産変動(SNA §3.98)
            # として K5 勘定へ。RoW(街の外の取引相手)へは落とさない。
            dst = party_sector(p.get("payee"))
            out.append(Flow(HOUSEHOLD, dst or VOID, amt,
                            "theft:other_volume" if dst else "theft:destroyed"))
    elif kind == "bankruptcy":
        seized = _f(p, "seized")
        if seized:
            dst = party_sector(p.get("payee")) or VOID      # IF-E2: 債権者不在 → RoW
            out.append(Flow(HOUSEHOLD, dst, seized, "bankruptcy:seizure"))
    elif kind == "enforcement":
        pen = _f(p, "penalty")
        if pen:
            dst = party_sector(p.get("payee"))              # IF-E2(OFF は None)
            if dst is None:
                dst = GOVERNMENT if ctx.get("government_on") else VOID
            out.append(Flow(HOUSEHOLD, dst, pen, "enforcement:fine"))
    elif kind == "tax":
        amt = _f(p, "amount")
        src = ctx.get("tax_src_default", VOID)
        if amt:
            src = ctx.get("tax_src", {}).get(id(payload), src)
            if src != GOVERNMENT:                   # 公務員給与の源泉税は行政の内部振替=行列に出さない
                out.append(Flow(src, GOVERNMENT, amt, f"tax:{p.get('tax')}"))
    elif kind == "b2b_trade":
        amt = _f(p, "amount")
        if amt:
            # OFF: 帳簿 dict のみ = 部門内(卸→小売)の名目フロー(行/列で相殺)。
            # IF-E2 UNCOVERED(第98): ON では org 預金間の**実移転**になり、買い手を台帳で
            # 特定できなかった仕入れは RoW(域外資本の店)が払う = EXTERNAL → ORG。
            out.append(Flow(party_sector(p.get("payer")) or ORG,
                            party_sector(p.get("payee")) or ORG, amt, "b2b_trade"))
    elif kind == "ems_transport":
        # H2 医療①: 救急搬送の公費。支出主体は区(payer="government")で、受け手は
        # **街の外**(救急を運行するのは東京消防庁 = 都 = 本シムの行政ではない)。
        # 行政が未構築の世界では payer が載らない = 街の残高が 1 円も動いていないので
        # フローを立てない(動いていない金を動いたことにしない)。
        cost = _f(p, "cost")
        src = party_sector(p.get("payer"))
        if cost and src is not None:
            out.append(Flow(src, party_sector(p.get("payee")) or EXTERNAL, cost,
                            "ems:public_transport"))
    elif kind == "medical_bill":
        # H2 医療③: 保険給付(街の外の保険者 → 医療機関 org)。amount は**実際に動いた額**で、
        # 受け手 org を台帳で特定できなかった回は 0(保険者も医療機関も街の外)。
        # 自己負担(self_pay)は spend 側で会計済みなので**ここでは数えない**(二重計上の禁止)。
        amt = _f(p, "amount")
        dst = party_sector(p.get("payee"))
        if amt and dst is not None:
            out.append(Flow(party_sector(p.get("payer")) or EXTERNAL, dst, amt,
                            "medical:insurance"))
    elif kind == "lost_return":
        # H3 遺失物①: 返還。1 行に**2 本のフロー**が載る。
        #   cash   … 遺失物バケツ(家計の通過勘定)→ 落とし主の財布 = **家計内部**
        #   amount … 報労金(遺失物法 28 条)。落とし主 → 拾得者 = **家計内部**
        # どちらも部門内の移転なので行と列で相殺される(街の総マネーは 1 円も動かない)。
        # ★それでも**フローとして立てる**のは、遺失物の金が「どこからどこへ」動いたかを
        #   行列に残すため(0 本にすると『金が消えた/湧いた』のか『部門内で動いた』のかが
        #   事後に区別できない)。
        cash, paid = _f(p, "cash"), _f(p, "amount")
        if cash:
            out.append(Flow(LOST_BUCKET_SECTOR, HOUSEHOLD, cash, "lost:return_cash"))
        if paid:
            out.append(Flow(HOUSEHOLD, HOUSEHOLD, paid, "lost:reward"))
    elif kind == "lost_keep":
        # H3 遺失物②: 着服(占有離脱物横領)。財布の中の現金が拾得者の手に渡る。
        # ★``crime``(窃盗)と**同じ K5 へは落とさない**: 窃盗が K5 なのは「受け取り側が
        #   世界に記録されていない」からで、着服は**受け取った個体が特定できている**
        #   (拾得者の残高が実際に増える)。よって取引ではなく家計内部の移転として扱う。
        amt = _f(p, "amount")
        if amt:
            out.append(Flow(LOST_BUCKET_SECTOR, HOUSEHOLD, amt, "lost:keep"))
    elif kind == "lost_expire":
        # H3 遺失物③: 時効取得(to="finder")= 家計内部 /
        #             失効(to="void")= 受け取り手が世界に居ない = **取引ではない**
        #             → K5(SNA 2008 第12章。``economy_sfc.on_lost_lapse`` と同じ勘定)。
        amt = _f(p, "amount")
        if amt:
            if str(p.get("to") or "") == "finder":
                out.append(Flow(LOST_BUCKET_SECTOR, HOUSEHOLD, amt, "lost:expire_finder"))
            else:
                out.append(Flow(LOST_BUCKET_SECTOR, OTHER, amt, "lost:lapse"))
    elif kind == "production":
        # IF-E2 案B: 域内に客が居ない org(全 org の 36.5%・従業者の 51.5%)の**輸出代金**。
        # 地域会計では観光サテライト勘定(非居住者の域内消費=輸出)の逆向き適用にあたる。
        rev = _f(p, "revenue")
        if rev:
            out.append(Flow(EXTERNAL, ORG, rev, "export:production"))
    return out


# --------------------------------------------------------------------------- #
# 2. 家計側の残高移動(検査①の材料)
# --------------------------------------------------------------------------- #
def move_for(kind: str, agent_id: int, payload: dict, accounts_on: bool) -> list[Move]:
    """1イベント → 家計個体の残高移動(現金/口座)。**合計 total が検査①の唯一の入力**。

    accounts ON のとき現金/口座の内訳が payload から一意に決まらない種(venture_open /
    candidacy / bankruptcy)は cash に寄せて記録する。**total は常に正しい**ので
    検査①(=合計残高の保存)は内訳の曖昧さに影響されない。"""
    p = payload or {}
    a = int(agent_id)
    if kind == "spend":
        amt = _f(p, "paid", _f(p, "amount"))         # IF-E2: 床クリップ時は実支払(paid)が正
        if accounts_on and str(p.get("src") or "") == "card":
            return [Move(a, acct=-amt, kind=kind)]
        return [Move(a, cash=-amt, kind=kind)]
    if kind == "wage":
        amt = _f(p, "amount")                        # 手取り(税は既に控除済み)
        if str(p.get("to") or "") == "account":
            return [Move(a, acct=amt, kind=kind)]
        return [Move(a, cash=amt, kind=kind)]
    if kind == "withdraw":
        amt = _f(p, "amount")
        return [Move(a, cash=amt, acct=-amt, kind=kind)]
    if kind == "rent":
        return [Move(a, acct=-_f(p, "paid"), kind=kind)]
    if kind == "interest_paid":
        return [Move(a, acct=_f(p, "amount"), kind=kind)]
    if kind == "loan_grant":
        return [Move(a, acct=_f(p, "amount"), kind=kind)]
    if kind == "loan_repay":
        return [Move(a, acct=-_f(p, "amount"), kind=kind)]
    if kind == "vc_investment":
        return [Move(a, acct=_f(p, "amount"), kind=kind)]
    if kind == "venture_sale":
        net = _f(p, "amount") - _f(p, "dividend")    # 店主(=イベントの agent_id)の取り分
        return [Move(a, acct=net, kind=kind)] if accounts_on else [Move(a, cash=net, kind=kind)]
    if kind == "venture_open":
        return [Move(a, cash=-_f(p, "cost"), kind=kind)]
    if kind == "candidacy":
        return [Move(a, cash=-_f(p, "deposit"), kind=kind)]
    if kind == "deposit":
        amt, phase = _f(p, "amount"), str(p.get("phase") or "")
        if phase == "paid":
            return [Move(a, cash=-amt, kind=kind)]
        if phase == "refund":
            return [Move(a, cash=amt, kind=kind)]
        return []                                    # forfeit / insufficient は本人の残高を動かさない
    if kind in ("reward", "rule_bonus", "civic_service"):
        return [Move(a, cash=_f(p, "amount"), kind=kind)]
    if kind == "chance_event":
        amt, t = _f(p, "amount"), str(p.get("type") or "")
        if t == "windfall":
            return [Move(a, cash=amt, kind=kind)]
        if t == "loss":
            return [Move(a, cash=-amt, kind=kind)]
        return []
    if kind == "crime":                              # agent_id は加害者。減るのは被害者の現金
        if str(p.get("kind") or "") != "theft":
            return []
        v = p.get("victim")
        return [Move(int(v), cash=-_f(p, "amount"), kind=kind)] if v is not None else []
    if kind == "enforcement":                        # agent_id は警察官。減るのは target の現金
        t = p.get("target")
        return [Move(int(t), cash=-_f(p, "penalty"), kind=kind)] if t is not None else []
    if kind == "bankruptcy":
        return [Move(a, cash=-_f(p, "seized"), kind=kind)]
    if kind == "move_home":
        return [Move(a, cash=-_f(p, "deposit"), kind=kind)]
    # ---- H3 遺失物(lost_property。既定 OFF のランには 1 件も出ない)----
    # ★2 本立てで記録する。片方だけでは 2 つの検査が両立しない:
    #   (a) 個体側 … 落とし主/拾得者の財布が実際に動いた分。**検査①(L3 の個体残高)**の材料。
    #   (b) バケツ側 … 遺失物バケツ(``LOST_BUCKET_ID``)の受け払い。**分類器の自己整合**
    #       (classifier_consistency)の材料。バケツは家計部門の内訳(通過勘定)なので、
    #       部門行列の家計純額は落下でも返還でも動かない。個体側だけを数えると
    #       「行列は動いていないのに家計の合計は減った」という**見かけの不整合**が出る。
    #   ★(b) は L3 に残高を持たないので conserve_household からは自動的に外れる
    #     (agent_id=-1 は L3 スナップショットの個体集合に居ない)= 検査①は無風。
    if kind == "lost_drop":
        # agent_id = 落とした本人。財布の中の現金だけが所持金から分離される
        #(cash キーが無い落とし物 = 傘・携帯などは金が 1 円も動かない)。
        cash = _f(p, "cash")
        if not cash:
            return []
        return [Move(a, cash=-cash, kind=kind),
                Move(LOST_BUCKET_ID, cash=cash, kind=kind)]
    if kind == "lost_return":
        # agent_id = 落とし主。中の現金がバケツから戻り(+cash)、報労金(遺失物法 28 条)を
        # 拾得者へ払う(-amount)。**実際に動いた額**が payload に載っている
        #(払い手の手持ちを超えた分は lost_property._pay が床クリップ済み)。
        cash, paid = _f(p, "cash"), _f(p, "amount")
        mv: list[Move] = []
        if cash or paid:
            mv.append(Move(a, cash=cash - paid, kind=kind))
        f = p.get("finder")
        if f is not None and paid:
            mv.append(Move(int(f), cash=paid, kind=kind))
        if cash:
            mv.append(Move(LOST_BUCKET_ID, cash=-cash, kind=kind))
        return mv
    if kind == "lost_keep":                          # agent_id = 拾得者(着服)
        amt = _f(p, "amount")
        if not amt:
            return []
        return [Move(a, cash=amt, kind=kind),
                Move(LOST_BUCKET_ID, cash=-amt, kind=kind)]
    if kind == "lost_expire":
        # agent_id = 時効取得した拾得者。**誰にも拾われず失効した回は agent_id=-1**
        # (= 受け取り手が世界に居ない。バケツ側の払い出しだけが立ち、行列では K5 へ出る)。
        amt = _f(p, "amount")
        if not amt:
            return []
        mv = [Move(LOST_BUCKET_ID, cash=-amt, kind=kind)]
        if a != LOST_BUCKET_ID:
            mv.append(Move(a, cash=amt, kind=kind))
        return mv
    return []


#: move_for が家計の残高を動かす種(move_home は L1 種に「金額」列が無いが敷金で現金が減る)。
HOUSEHOLD_KINDS: frozenset[str] = frozenset({
    "spend", "wage", "withdraw", "rent", "interest_paid", "loan_grant", "loan_repay",
    "vc_investment", "venture_sale", "venture_open", "candidacy", "deposit", "reward",
    "rule_bonus", "civic_service", "chance_event", "crime", "enforcement", "bankruptcy",
    "move_home",
    # H3 遺失物: 落下で減り、返還 / 着服 / 時効取得で増える(lost_property。既定 OFF)。
    "lost_drop", "lost_return", "lost_keep", "lost_expire",
})

#: 検査に読み込むイベント種(フロー用 ∪ 家計残高用)。
SCAN_KINDS: frozenset[str] = MONEY_KINDS | HOUSEHOLD_KINDS

#: payload に「金らしい」数値キーがあるか判定するための鍵語(新しい金の経路の検知に使う)。
MONEY_KEYS: tuple[str, ...] = (
    "amount", "cost", "price", "fee", "fare", "deposit", "penalty", "seized",
    "paid", "balance", "salary", "wage", "debt", "revenue", "expense", "fine",
)


def leak_family(tag: str) -> str:
    """漏れタグ → 経路の族(`spend:food` → `spend`)。

    消費カテゴリの増減(`spend:cafe` の新設など)は**新しい金の経路ではない**ので族へ畳む。
    族が増えたときだけ「金の経路が増えた」と判定できる=監視テストの粒度。"""
    return str(tag).split(":", 1)[0]


class MoneyKindTally:
    """未分類の金の経路を **走査中に畳む**累積器(W2-6 ②のメモリ有界化)。

    旧実装は該当イベントを `list[dict]` に貯めてから数えていたので、40.6 億行の L1 では
    「未分類の経路が 1 種でも大量に出た瞬間に落ちる」形だった。判定式は 1 ミリも変えずに
    **保持量を O(イベント種類数)**(実測 distinct kind は 100 前後)へ落とす。

    `unclassified_money_kinds()` と**同じ規則**で数えること(規則の源はここ 1 箇所)。"""

    __slots__ = ("_acc",)

    def __init__(self) -> None:
        self._acc: dict[str, dict] = {}

    @staticmethod
    def hits(payload) -> list[str]:
        """payload が持つ「金らしい非ゼロ数値キー」(判定規則の単一の源)。"""
        if not isinstance(payload, dict):
            return []
        return sorted(k for k, v in payload.items()
                      if k in MONEY_KEYS and isinstance(v, (int, float))
                      and not isinstance(v, bool) and float(v) != 0.0)

    def add(self, kind, payload) -> None:
        if kind in MONEY_KINDS or kind in DERIVED_MONEY_KINDS:
            return
        hits = self.hits(payload)
        if not hits:
            return
        rec = self._acc.setdefault(str(kind), {"n": 0, "keys": set()})
        rec["n"] += 1
        rec["keys"].update(hits)

    def result(self) -> dict[str, dict]:
        return {k: {"n": v["n"], "keys": sorted(v["keys"])}
                for k, v in sorted(self._acc.items())}

    def __len__(self) -> int:
        return len(self._acc)


def unclassified_money_kinds(events) -> dict[str, dict]:
    """**金の経路の追加を検知する装置**。

    payload に金らしい数値キーを持つのに MONEY_KINDS にも DERIVED_MONEY_KINDS にも
    入っていないイベント種を洗い出す。新しい送金経路が実装されると必ずここに現れるので、
    テストで「空である」ことを固定すれば、会計検査の網羅性が構造的に守られる。

    events は L1 の全イベント(kind, payload)の列、または `load_run` が返す
    `MoneyKindTally`(走査中に畳んだ同じ集計)。戻り値 {kind: {"n":件数, "keys":[鍵語]}}。"""
    if hasattr(events, "result"):            # MoneyKindTally = 既に畳んである
        return events.result()
    tally = MoneyKindTally()
    for e in events:
        tally.add(e.get("kind"), e.get("payload") or {})
    return tally.result()


# --------------------------------------------------------------------------- #
# 3. 行列の組み立て
# --------------------------------------------------------------------------- #
def build_matrix(flows: list[Flow]) -> dict[tuple[str, str], float]:
    """フロー列 → 部門×部門の金額行列 {(src, dst): amount}。"""
    m: dict[tuple[str, str], float] = defaultdict(float)
    for fl in flows:
        m[(fl.src, fl.dst)] += fl.amount
    return dict(m)


def row_col_sums(matrix: dict[tuple[str, str], float]) -> dict[str, dict[str, float]]:
    """各部門の 流出(row)/ 流入(col)/ 純額(col-row)。

    Caiani の「行と列がゼロに合計される」は本スクリプトでは
    「**各部門の純額 = その部門の残高変化**」として検査①で判定する。
    行列そのものは二重記入で作っているので全部門の純額の総和は構造的に 0(自明)。"""
    out = {s: {"outflow": 0.0, "inflow": 0.0, "net": 0.0} for s in SECTORS}
    for (src, dst), amt in matrix.items():
        out[src]["outflow"] += amt
        out[dst]["inflow"] += amt
    for s in SECTORS:
        out[s]["net"] = out[s]["inflow"] - out[s]["outflow"]
    return out


def leak_breakdown(flows: list[Flow]) -> dict[str, dict[str, float]]:
    """void を相手にしたフローの内訳 {tag: {"created":…, "destroyed":…, "count":…}}。

    created  = void → 実在部門(何もないところから貨幣が湧いた)
    destroyed = 実在部門 → void(貨幣が消えた)"""
    acc: dict[str, dict[str, float]] = defaultdict(
        lambda: {"created": 0.0, "destroyed": 0.0, "count": 0.0})
    for fl in flows:
        if fl.src == VOID and fl.dst == VOID:
            continue
        if fl.src == VOID:
            acc[fl.tag]["created"] += fl.amount
            acc[fl.tag]["count"] += 1
        elif fl.dst == VOID:
            acc[fl.tag]["destroyed"] += fl.amount
            acc[fl.tag]["count"] += 1
    return {k: dict(v) for k, v in acc.items()}


# --------------------------------------------------------------------------- #
# 4. 検査① — 貨幣ストックの保存
# --------------------------------------------------------------------------- #
def _iter_stock(stock):
    """ストックを **step 昇順の (step, {agent_id: 残高}) ペア列**として供給する。

    受け付ける形は 2 つ。どちらでも**同じ窓・同じ順序**になる:
      - `{step: {agent_id: 残高}}` … 従来の辞書(合成テストと過去の呼び出し規約)。
      - `(step, {agent_id: 残高})` を yield する反復可能物 … `SnapshotStocks`
        (L3 を 1 枚ずつ読む W2-6 のストリーム)。

    ★ストリーム側は「L3 が step 非減少で追記される」という記録側の不変条件に乗っている
      (`ObserverLogger.log_snapshot` は step ループから 1 回ずつ呼ばれ、part は index 昇順で
      結合される)。破れていれば `SnapshotStocks` が例外を投げる = **黙って間違えない**。"""
    if isinstance(stock, dict):
        for s in sorted(stock):
            yield s, stock[s]
        return
    yield from stock


def conserve_household(stock_by_step,
                       moves_by_step: dict[int, list[Move]],
                       rel_tol: float) -> dict:
    """家計の検査①: L3 スナップショット間で Δ(money+account) = Σ move.total か。

    stock_by_step: `{step: {agent_id: money+account}}` または `SnapshotStocks`(L3 由来)
    moves_by_step: {step: [Move, ...]}(L1 由来。**その step の処理で動いた分**)

    L1 の step s のイベントは step s のスナップショット(step 末に取る)に**反映済み**なので、
    窓 (s0, s1] の流量は step が s0 < step <= s1 のイベント。

    ★W2-6 ②(25万体対応): **隣り合う 2 枚のスナップショットしか保持しない**。
      全 step を同時に持つと 120 枚 × 25万体 = 3,000 万エントリになる。2 枚が下限なのは
      検査式そのものの要請で、d_common が「両方の枚に居る個体」の差の総和だから
      (集合演算に個体別の値が要る)。判定式は 1 ミリも変えていない。"""
    windows = []
    tot_abs_flow = 0.0
    tot_resid = 0.0
    prev: tuple | None = None
    for s1, a1 in _iter_stock(stock_by_step):
        if prev is None:
            prev = (s1, a1)
            continue
        s0, a0 = prev
        prev = (s1, a1)                    # 直前の 1 枚だけを引き継ぐ(2 枚を超えて持たない)
        common = a0.keys() & a1.keys()
        d_common = sum(a1[i] - a0[i] for i in common)
        rot_in = sum(a1[i] for i in a1.keys() - a0.keys())     # 途中参加(pool)
        rot_out = sum(a0[i] for i in a0.keys() - a1.keys())    # 途中退出(pool)
        flow = 0.0
        by_kind: dict[str, float] = defaultdict(float)
        for st in range(s0 + 1, s1 + 1):
            for mv in moves_by_step.get(st, ()):
                if mv.agent_id in common:
                    flow += mv.total
                    by_kind[mv.kind] += mv.total
        resid = d_common - flow
        tot_abs_flow += sum(abs(v) for v in by_kind.values())
        tot_resid += resid
        windows.append({
            "from_step": s0, "to_step": s1, "n_common": len(common),
            "delta_stock": round(d_common, 3), "flow": round(flow, 3),
            "residual": round(resid, 3),
            "pool_in": round(rot_in, 3), "pool_out": round(rot_out, 3),
            "kinds": sorted(by_kind),
        })
    rel = abs(tot_resid) / tot_abs_flow if tot_abs_flow > 0 else 0.0
    return {
        "sector": HOUSEHOLD, "observable": True, "n_windows": len(windows),
        "total_abs_flow": round(tot_abs_flow, 3),
        "total_residual": round(tot_resid, 3),
        "relative_residual": rel,
        "pass": rel <= rel_tol,
        "rel_tol": rel_tol,
        "windows": windows,
    }


def conserve_government(budget_rows: list[dict], flows: list[Flow],
                        rel_tol: float) -> dict:
    """行政の検査①。2段構えで見る。

    (a) **内部整合**: public_budget の (revenue, expense, balance) が level ごとに
        「前回残高 + 歳入 − 歳出 = 今回残高」を満たすか(行政モジュールの自己申告の検算)。
    (b) **外部整合(本命)**: 行政の残高変化が、**本スクリプトが L1 から独立に組んだ
        フロー行列の行政純額**と一致するか。(a) が通っても (b) が破れていれば、
        行政が世界の他部門と繋がっていない金を動かしていることになる。

    注意(正直な限界): public_budget は日境界フェーズの先頭で「前日ぶん」を締めるので、
    締め step のうち締めより後に走るフェーズ(公務員ペイロール等)は次の記録に入る。
    (b) は締め step を排他境界(step < S)として集計するが、同一 step 内のフェーズ順の
    ずれは残差として現れうる。判定はこの粒度での近似である。"""
    per_level: dict[str, list[dict]] = defaultdict(list)
    for r in budget_rows:
        per_level[str(r.get("level"))].append(r)
    levels: dict[str, list[dict]] = {}
    tot_abs, tot_res = 0.0, 0.0
    for lv in sorted(per_level):
        rows = sorted(per_level[lv], key=lambda r: int(r.get("step", 0)))
        recs, prev = [], None
        for r in rows:
            rev, exp, bal = _f(r, "revenue"), _f(r, "expense"), _f(r, "balance")
            if prev is not None:
                resid = bal - (prev + rev - exp)
                recs.append({"step": r.get("step"), "residual": round(resid, 3),
                             "revenue": rev, "expense": exp, "balance": bal})
                tot_abs += abs(rev) + abs(exp)
                tot_res += resid
            prev = bal
        levels[lv] = recs
    rel = abs(tot_res) / tot_abs if tot_abs > 0 else 0.0

    steps = sorted({int(r.get("step", 0)) for r in budget_rows})
    ext: dict = {"applicable": False}
    if len(steps) >= 2:
        s_first, s_last = steps[0], steps[-1]
        bal0 = sum(_f(r, "balance") for r in budget_rows if int(r.get("step", 0)) == s_first)
        bal1 = sum(_f(r, "balance") for r in budget_rows if int(r.get("step", 0)) == s_last)
        inflow = sum(f.amount for f in flows
                     if f.dst == GOVERNMENT and f.src != GOVERNMENT and s_first <= f.step < s_last)
        outflow = sum(f.amount for f in flows
                      if f.src == GOVERNMENT and f.dst != GOVERNMENT and s_first <= f.step < s_last)
        resid = (bal1 - bal0) - (inflow - outflow)
        denom = inflow + outflow
        ext = {"applicable": True, "from_step": s_first, "to_step": s_last,
               "open_balance": round(bal0, 1), "close_balance": round(bal1, 1),
               "matrix_inflow": round(inflow, 1), "matrix_outflow": round(outflow, 1),
               "residual": round(resid, 1),
               "relative_residual": (abs(resid) / denom) if denom else 0.0}
        ext["pass"] = ext["relative_residual"] <= max(rel_tol, 1e-3)

    return {"sector": GOVERNMENT, "observable": True,
            "total_abs_flow": round(tot_abs, 3), "total_residual": round(tot_res, 3),
            "relative_residual": rel, "pass": rel <= rel_tol, "rel_tol": rel_tol,
            "levels": levels, "external": ext}


#: finance.parquet の列 → 部門残高の取り出し方(IF-E2 案B。ON のランでのみ存在する)。
#:   org        企業の預金(スカラー 1 本の総和)
#:   bank       Bank.capital + VCFund.balance(どちらも金融部門)
#:   external   RoW が**正味で吸収した**額 = row_out − row_in(街の外に残高は無い = SNA §26.6)
#:   other_volume  非取引の資産変動(SNA K.5)の累積 = k5_other(第98バッチ)
#:   household   家計の預金+現金 + 遺失物バケツ(``LOST_BUCKET_SECTOR`` の宣言どおり
#:               家計の通過勘定として扱う)。
#:               ★正直な限界: ``lost_property_held`` は **finance.parquet の列に無い**
#:                 (``economy_sfc.COLUMNS`` は 15 列で、この値は provenance にしか出ない)。
#:                 列を足すのは L2 サイドカーのスキーマ変更 = 経済レーンの所有なので本レーンでは
#:                 やらない。よって今日のランでは常に 0.0 が読まれ、遺失物が ON のあいだは
#:                 「まだバケツに残っている現金」のぶんだけ本検査(検査①の finance 版)に
#:                 残差が出る。**L3 由来の ``conserve_household`` は完全に閉じている**
#:                 (move_for が lost_drop / lost_return / lost_keep / lost_expire を数えるため)。
#:                 列が生えた日にこの行がそのまま効く = 先に正しい式を書いておく。
FINANCE_BALANCE = {
    ORG:        lambda r: _f(r, "org_balance"),
    BANK:       lambda r: _f(r, "bank_capital") + _f(r, "vc_balance"),
    EXTERNAL:   lambda r: _f(r, "row_out") - _f(r, "row_in"),
    OTHER:      lambda r: _f(r, "k5_other"),
    GOVERNMENT: lambda r: _f(r, "gov_balance") + _f(r, "escrow"),
    HOUSEHOLD:  lambda r: _f(r, "household_balance") + _f(r, "lost_property_held"),
}


def conserve_sector(fin_rows: list[dict], flows: list[Flow], sector: str,
                    rel_tol: float) -> dict:
    """検査①の一般形(IF-E2 案B の finance.parquet を使う)。

    「期首残高 + 流入 − 流出 = 期末残高」を、本スクリプトが L1 から**独立に**組んだ
    フロー行列の部門純額と突き合わせる。窓は [最初の行の step, 最後の行の step) の
    排他境界(finance の行は日境界フェーズの先頭 = その step の活動より前に採られるため。
    行政の外部整合 conserve_government(b) と同じ規約)。

    ★この検査が通ることが「org / 銀行 / RoW も**世界に実在する残高**として閉じている」の意味で、
      案B の受入条件(検査①が全部門で成立)そのものである。"""
    getter = FINANCE_BALANCE.get(sector)
    if getter is None or len(fin_rows) < 2:
        return {"sector": sector, "observable": False,
                "reason": "finance.parquet が無い(economy.org_accounting=false)"}
    rows = sorted(fin_rows, key=lambda r: int(r.get("step", 0)))
    s0, s1 = int(rows[0].get("step", 0)), int(rows[-1].get("step", 0))
    bal0, bal1 = getter(rows[0]), getter(rows[-1])
    inflow = sum(f.amount for f in flows
                 if f.dst == sector and f.src != sector and s0 <= f.step < s1)
    outflow = sum(f.amount for f in flows
                  if f.src == sector and f.dst != sector and s0 <= f.step < s1)
    n_flows = sum(1 for f in flows
                  if s0 <= f.step < s1 and (f.dst == sector or f.src == sector))
    resid = (bal1 - bal0) - (inflow - outflow)
    denom = inflow + outflow
    rel = (abs(resid) / denom) if denom else 0.0
    # 丸めの上限を**独立に**置く: 残高は無限精度で動くが L1 payload は 0.1 円に丸められるので、
    # 1 フローあたり最大 0.05 円の誤差が乗る(docstring「許容誤差はゼロにできない」)。
    # 相対誤差は「残高が大きく流量が小さい部門」(行政など)で過敏になるため、
    # **相対 or 丸め上限のどちらかを満たせば PASS** とする(判定を甘くするのではなく、
    # 丸めという既知の誤差源の大きさを明示的にモデル化する)。
    abs_tol = 0.05 * max(n_flows, 1) + 0.1
    return {"sector": sector, "observable": True, "from_step": s0, "to_step": s1,
            "open_balance": round(bal0, 1), "close_balance": round(bal1, 1),
            "total_abs_flow": round(denom, 3), "matrix_inflow": round(inflow, 1),
            "matrix_outflow": round(outflow, 1), "total_residual": round(resid, 3),
            "relative_residual": rel, "rel_tol": rel_tol,
            "n_flows": n_flows, "abs_tol": round(abs_tol, 3),
            "pass": rel <= max(rel_tol, 1e-6) or abs(resid) <= abs_tol}


def revenue_actual_by_payee(events: list[dict]) -> dict[str, float]:
    """IF-E2 案B の ON ランで、**受け手 org ごとの実売上**を spend の payee から直に集計する。

    IF-E フェーズ1 の serve→spend 突合(スタッフ不在 220/222 で機能しなかった)と違い、
    受け手は支払のその場で確定しているので突合そのものが要らない。"""
    out: dict[str, float] = defaultdict(float)
    for e in events:
        if e["kind"] != "spend":
            continue
        payee = str((e["payload"] or {}).get("payee") or "")
        if payee and party_sector(payee) == ORG:
            out[payee] += _f(e["payload"], "amount")
    return dict(out)


def classifier_consistency(flows: list[Flow], moves: list[Move]) -> dict:
    """分類器の自己整合: フロー行列の**家計純額**と、家計個体の残高移動の合計が一致するか。

    flows_for()(検査②側)と move_for()(検査①側)は別々に書かれた2つの写像なので、
    片方に経路の書き漏れがあれば必ずここで差が出る。**分類器のバグ検出器**であって
    シムのバグ検出器ではない(シム側の漏れは void 列に出る)。

    ★moves には家計個体だけでなく **遺失物バケツの擬似メンバー**(``LOST_BUCKET_ID``)が
      混じりうる。バケツは家計部門の内訳(通過勘定)なので、部門の純額を作るこの検査では
      家計の一員として足すのが正しい(足さないと「落としてまだ拾われていない現金」の
      ぶんだけ見かけの不整合が出る)。個体残高の検査①はこの id を持たない。"""
    net_matrix = (sum(f.amount for f in flows if f.dst == HOUSEHOLD and f.src != HOUSEHOLD)
                  - sum(f.amount for f in flows if f.src == HOUSEHOLD and f.dst != HOUSEHOLD))
    net_moves = sum(m.total for m in moves)
    diff = net_matrix - net_moves
    scale = max(abs(net_matrix), abs(net_moves), 1.0)
    return {"matrix_net": round(net_matrix, 3), "moves_net": round(net_moves, 3),
            "diff": round(diff, 3), "relative": abs(diff) / scale,
            "pass": abs(diff) <= 1e-6 * scale + 1e-6}


# --------------------------------------------------------------------------- #
# 5. revenue_est 乖離(既知の断絶を別枠で明示)
# --------------------------------------------------------------------------- #
UNKNOWN_ORG = "__unknown__"


def match_serve_to_spend(serve_rows: list[dict], spend_rows: list[dict]) -> dict:
    """serve イベントを、それを生んだ客の spend へ厳密に突き合わせる。

    serve は scheduler が「客の spend と同一 work_node の勤務中スタッフ」へ機械的に帰属させた
    イベントなので、対応する spend は必ず存在する。突き合わせ鍵は2通り:

      - **スタッフ有り**: payload に `customer` が入る → (step, customer, cat) で一意。
      - **スタッフ不在**(`unstaffed`): payload に customer が**入らない**(agent_id=-1)。
        座標も使えない — serve フェーズは `_phase_jitter`(路上の微移動)の**後**に走るので
        serve の (x, y) は spend 時点の座標と一致しない(実測で確認)。そこで
        **同 step・同 cat の spend を L1 記録順に消費する**。serve は spend を記録順に走査して
        発火するので、この対応は順序で一意に決まる。

    同一 spend に複数 serve が立ちうる(max_serve_per_event>1)ので、spend 側を使用済みに
    してから加算する(二重計上の禁止)。org_id が無い/多義 node は `__unknown__` へ落として
    正直に別立てする(第58 の掟=多義 node は unknown と開示)。"""
    by_key: dict[tuple[int, int, str], list[int]] = defaultdict(list)
    queue: dict[tuple[int, str], list[int]] = defaultdict(list)
    for i, sp in enumerate(spend_rows):
        st, cat = int(sp["step"]), str(sp["cat"])
        by_key[(st, int(sp["agent_id"]), cat)].append(i)
        queue[(st, cat)].append(i)

    used: set[int] = set()
    by_org: dict[str, float] = defaultdict(float)
    n_matched = n_unmatched = n_dup = 0
    n_unstaffed = 0
    for s in serve_rows:
        p = s.get("payload") or {}
        st, cat = int(s["step"]), str(p.get("cat") or "")
        if p.get("unstaffed"):
            n_unstaffed += 1
        cust = p.get("customer")
        cand = by_key.get((st, int(cust), cat), []) if cust is not None else queue.get((st, cat), [])
        pick = next((i for i in cand if i not in used), None)
        if pick is None:
            if cand:
                n_dup += 1            # 同一 spend への2本目以降の serve(二重計上しない)
            else:
                n_unmatched += 1
            continue
        used.add(pick)
        n_matched += 1
        oid = p.get("org_id")
        by_org[str(oid) if oid else UNKNOWN_ORG] += float(spend_rows[pick]["amount"])
    return {"by_org": dict(by_org), "n_matched": n_matched, "n_unmatched": n_unmatched,
            "n_dup_serve": n_dup, "n_serve": len(serve_rows), "n_unstaffed": n_unstaffed}


def revenue_gap(ledger_rows: list[dict], serve_rows: list[dict],
                spend_rows: list[dict]) -> dict:
    """org_ledger.revenue_est の合計 vs org 帰属店舗で実際に発生した spend の合計。

    ledger_rows: org_ledger.parquet の行 [{day, org_id, revenue_est, wage_paid, serve_count, ...}]
    serve_rows : L1 の serve イベント / spend_rows: L1 の spend(step, agent_id, x, y, cat, amount)。"""
    est_by_org: dict[str, float] = defaultdict(float)
    wage_by_org: dict[str, float] = defaultdict(float)
    for r in ledger_rows:
        est_by_org[str(r.get("org_id"))] += _f(r, "revenue_est")
        wage_by_org[str(r.get("org_id"))] += _f(r, "wage_paid")

    mt = match_serve_to_spend(serve_rows, spend_rows)
    actual_by_org = mt["by_org"]

    orgs = sorted(set(est_by_org) | set(actual_by_org))
    rows = []
    for oid in orgs:
        est, act = est_by_org.get(oid, 0.0), actual_by_org.get(oid, 0.0)
        rows.append({"org_id": oid, "revenue_est": round(est, 1),
                     "spend_actual": round(act, 1), "gap": round(est - act, 1),
                     "ratio": (est / act) if act else None,
                     "wage_paid": round(wage_by_org.get(oid, 0.0), 1)})
    tot_est = sum(est_by_org.values())
    tot_act = sum(actual_by_org.values())
    known_act = tot_act - actual_by_org.get(UNKNOWN_ORG, 0.0)
    return {
        "rows": rows,
        "total_revenue_est": round(tot_est, 1),
        "total_spend_actual": round(tot_act, 1),
        "known_org_spend": round(known_act, 1),
        "gap": round(tot_est - known_act, 1),
        "ratio": (tot_est / known_act) if known_act else None,
        "unknown_org_spend": round(actual_by_org.get(UNKNOWN_ORG, 0.0), 1),
        "n_serve": mt["n_serve"], "n_serve_matched": mt["n_matched"],
        "n_serve_unstaffed": mt["n_unstaffed"],
        "n_serve_without_spend": mt["n_unmatched"],
        "n_serve_dedup_dropped": mt["n_dup_serve"],
        "total_wage_paid": round(sum(wage_by_org.values()), 1),
    }


# --------------------------------------------------------------------------- #
# 6. ラン読み込み
# --------------------------------------------------------------------------- #
def _read_config(run_dir: Path) -> dict:
    """runs/<name>/config.yaml から検査に必要なトグルだけを取り出す(omegaconf 依存を避ける)。"""
    path = run_dir / "config.yaml"
    out = {"accounts_on": False, "government_on": False, "pool_on": False,
           "org_ledger_on": False, "indoor_fields_on": False,
           "org_accounting_on": False}
    if not path.exists():
        return out
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    # 素朴なインデント追跡(依存を増やさないための最小パーサ。キーの重複は経路で解決)。
    stack: list[tuple[int, str]] = []
    for raw in text:
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip())
        line = raw.strip()
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key, val = key.strip(), val.split("#")[0].strip()
        while stack and stack[-1][0] >= indent:
            stack.pop()
        path_key = ".".join([k for _, k in stack] + [key])
        stack.append((indent, key))
        if val in ("true", "false"):
            b = (val == "true")
            if path_key == "economy.accounts.enabled":
                out["accounts_on"] = b
            elif path_key == "government.enabled":
                out["government_on"] = b
            elif path_key == "agents.pool.enabled" or path_key == "pool.enabled":
                out["pool_on"] = b
            elif path_key == "work.service.ledger.enabled":
                out["org_ledger_on"] = b
            elif path_key == "work.service.indoor_fields":
                out["indoor_fields_on"] = b
            elif path_key == "economy.org_accounting.enabled":   # IF-E2 案B
                out["org_accounting_on"] = b
    return out


#: L1 から**実際に読む列**。会計の検査は座標を 1 度も使わない(serve↔spend の突合が
#: 座標を使えないことは match_serve_to_spend の docstring 参照)ので x / y は読まない。
#: 40.6 億行 × float32 2 本 = 約 32 GB の復号がこれだけで消える(W2-6 ②)。
L1_SCAN_COLUMNS: tuple[str, ...] = ("step", "agent_id", "kind", "payload")

#: org_ledger から読む列(検査が実際に使う 4 つだけ)。
#: revenue_gap は org_id / revenue_est / wage_paid を、analyze の擬似フローは day を使う。
#: **ここを増やすときは使う側と一緒に増やすこと**(射影は「使っていない列を読まない」宣言)。
LEDGER_COLUMNS: tuple[str, ...] = ("day", "org_id", "revenue_est", "wage_paid")

#: parquet を dict 行へ実体化するときのチャンク行数(l1_stream の既定と同じ)。
SCAN_BATCH_ROWS = 131_072


class SnapshotStocks:
    """L3 スナップショットを **1 枚ずつ** `(step, {agent_id: money+account})` で供給する。

    W2-6 ②(25万体対応)。旧実装は `read_table` で全 step の state 文字列を載せ、さらに
    `{step: {agent_id: 残高}}` を全 step ぶん保持していた(120 枚 × 25万体 = 3,000 万
    エントリ + 全 state の JSON 文字列)。本クラスは **1 枚ぶんの JSON と 1 枚ぶんの
    残高辞書**しか作らない。ピークを決めるのは呼び出し側(`conserve_household` が
    隣り合う 2 枚を保持する)。

    - canonical が無いラン(走行中 / クラッシュ)でも完結 part を index 昇順で読む
      (`l1_stream.l1_paths`)。
    - `step` は **非減少**でなければならない(記録側の不変条件)。破れていたら
      黙って間違った窓を作らずに `ValueError` を投げる。
    - JSON が壊れている枚は**飛ばす**(旧実装と同じ。欠測を偽の値で埋めない)。
    - 真偽値は「読める枚が 1 枚でもあるか」を実際に 1 枚読んで判定する
      (行数だけで判定すると全枚が壊れているランを observable と誤申告するため)。
    """

    __slots__ = ("paths",)

    def __init__(self, paths) -> None:
        self.paths = [Path(p) for p in paths]

    def __iter__(self):
        prev = None
        for path in self.paths:
            for d in iter_table_columns(path, ["step", "state"], batch_rows=1):
                st, raw = int(d["step"][0]), d["state"][0]
                try:
                    agents = json.loads(raw).get("agents", []) if raw else []
                except (TypeError, ValueError):
                    continue
                if prev is not None and st < prev:
                    raise ValueError(
                        f"l3_snapshots の step が減少した({prev} → {st})。"
                        "記録は追記専用で step 非減少のはずで、破れていると検査①の窓が"
                        "無意味になる(黙って間違えないためここで止める)。")
                prev = st
                yield st, {int(a["id"]): float(a.get("money", 0.0))
                           + float(a.get("account", 0.0) or 0.0)
                           for a in agents}

    def __bool__(self) -> bool:
        for _ in self:
            return True
        return False


def _read_rows(paths, columns=None) -> list[dict]:
    """parquet 群を **列射影 + チャンク**で読み、行 dict のリストにする。

    `read_table().to_pylist()` の置き換え。実体化した list そのものは行数に比例して
    残る(呼び出し側が全行を要る表にだけ使うこと)が、**読み込み中に表 1 枚を丸ごと
    Arrow で持つピーク**が消える。列射影で「使わない列は 1 バイトも読まない」。"""
    out: list[dict] = []
    for path in paths:
        for d in iter_table_columns(path, columns, batch_rows=SCAN_BATCH_ROWS):
            keys = list(d)
            for vals in zip(*(d[k] for k in keys)):
                out.append(dict(zip(keys, vals)))
    return out


def load_run(run_dir: Path) -> dict:
    """ラン1本を読み、検査に必要な素材だけをメモリに載せる(payload の JSON パースは対象種のみ)。

    W2-6 ②(25万体対応)で全読みを潰した。何が有界になり何が残るかは module docstring の
    「メモリの上限」節を参照(要約: L3 と未分類検知は有界・L1 の金額イベントは残る)。"""
    run_dir = Path(run_dir)
    cfg = _read_config(run_dir)

    events: list[dict] = []
    serve_rows: list[dict] = []
    budget_rows: list[dict] = []
    # 未分類の金の経路の検知は**走査中に畳む**(list に貯めない)。安価な部分文字列で事前ふるい。
    other_money = MoneyKindTally()
    needles = tuple(f'"{k}"' for k in MONEY_KEYS)
    want = SCAN_KINDS | {"serve", "public_budget"}
    l1_files = l1_paths(run_dir)
    if not l1_files:
        raise FileNotFoundError(
            f"L1 が見つからない: {run_dir / 'l1_events.parquet'}"
            "(canonical も完結 part も無い)")
    for path in l1_files:
        for d in iter_table_columns(path, L1_SCAN_COLUMNS, batch_rows=SCAN_BATCH_ROWS):
            for st, aid, kind, pl in zip(d["step"], d["agent_id"], d["kind"],
                                         d["payload"]):
                if kind not in want:
                    if kind not in DERIVED_MONEY_KINDS and pl and any(n in pl for n in needles):
                        try:
                            other_money.add(kind, json.loads(pl))
                        except (TypeError, ValueError):
                            pass
                    continue
                try:
                    p = json.loads(pl) if pl else {}
                except (TypeError, ValueError):
                    p = {}
                rec = {"step": int(st), "agent_id": int(aid), "kind": kind, "payload": p}
                if kind == "serve":
                    serve_rows.append(rec)
                elif kind == "public_budget":
                    budget_rows.append({"step": int(st), **p})
                else:
                    events.append(rec)

    snap_paths = l1_paths(run_dir, "l3_snapshots")
    stock = SnapshotStocks(snap_paths) if snap_paths else {}

    # org_ledger: 行数は O(org 数 × 日数)= 25万体でも 11 万行(L1/L3 とは桁が違う)。
    # 列射影で「検査が使う 4 列だけ」に絞る。
    ledger_rows = _read_rows(l1_paths(run_dir, "org_ledger"), LEDGER_COLUMNS)

    # IF-E2 案B: 部門別残高の日次サイドカー(org / 銀行 / VC / 行政 / 供託 / 家計 / RoW)。
    # これがあると検査①が org・銀行・行政・RoW でも成立する(= 案B の受入条件)。
    # **1 日 1 行**なので列射影はしない(レポートが期首/期末の全列を印字するため、
    # 絞ると出力が減ってしまう。行数が小さいので絞る必要も無い)。
    fin_rows = _read_rows(l1_paths(run_dir, "finance"))

    return {"cfg": cfg, "events": events, "serve": serve_rows,
            "budget": budget_rows, "stock": stock, "ledger": ledger_rows,
            "finance": fin_rows, "other_money": other_money, "run_dir": str(run_dir)}


# --------------------------------------------------------------------------- #
# 7. 解析本体
# --------------------------------------------------------------------------- #
def resolve_tax_sources(events: list[dict]) -> dict[int, str]:
    """tax イベントの支払い元部門を突き合わせで解決する。

    - consumption: 同 (step, agent) の spend で base==amount のものの**受け取り部門**が納税者
      (内税=価格に含まれる。客はもう払っており、remit するのは販売側)。
    - income/resident: 同 (step, agent) の wage で gross==base のものの**支払い部門**。
      公務員給与(source=civil)なら行政の内部振替なのでフローを出さない(GOVERNMENT を返す)。

    戻り値は id(payload) → 部門。payload dict の同一性で引く(build 内で1回だけ使う)。"""
    spend_ix: dict[tuple[int, int], list[dict]] = defaultdict(list)
    wage_ix: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for e in events:
        if e["kind"] == "spend":
            spend_ix[(e["step"], e["agent_id"])].append(e["payload"])
        elif e["kind"] == "wage":
            wage_ix[(e["step"], e["agent_id"])].append(e["payload"])
    out: dict[int, str] = {}
    for e in events:
        if e["kind"] != "tax":
            continue
        p = e["payload"]
        base = _f(p, "base")
        which = str(p.get("tax") or "")
        key = (e["step"], e["agent_id"])
        if which == "consumption":
            for sp in spend_ix.get(key, ()):
                if abs(_f(sp, "amount") - base) < 0.051:
                    dst = spend_destination(sp.get("cat"), sp.get("payee"))[0]
                    # IF-E2 の正直な限界: 屋台(venture)は既存実装が売上**全額**を店主へ渡すので
                    # 内税ぶんを売り手が留保していない。その穴は RoW の tax_gap チャネルが埋める
                    # (改名して隠さない)。ここも同じ相手方に揃えないと行列が閉じない。
                    if dst == VENTURE and sp.get("payee"):
                        dst = EXTERNAL
                    out[id(p)] = dst
                    break
        elif which in ("income", "resident"):
            for wg in wage_ix.get(key, ()):
                if abs(_f(wg, "gross") - base) < 0.051:
                    sec = party_sector(wg.get("payer"))   # IF-E2(OFF は None)
                    out[id(p)] = sec if sec is not None else WAGE_SOURCE_SECTOR.get(
                        str(wg.get("source") or ""), _WAGE_DEFAULT)[0]
                    break
    return out


def analyze(run: dict, rel_tol: float = 1e-6) -> dict:
    """ラン素材 → 検査①②+revenue_est 乖離のレポート辞書。"""
    events, cfg = run["events"], run["cfg"]
    ctx = {"government_on": cfg["government_on"],
           "tax_src": resolve_tax_sources(events),
           "tax_src_default": VOID}

    # W2-3: 日境界の step 数は **このランの run.dt_min** で決まる(Δt=10 なら 144 = 従来と同値)。
    spd = run_dt.steps_per_day(run.get("run_dir"))
    flows: list[Flow] = []
    flows_by_day: dict[int, list[Flow]] = defaultdict(list)
    moves_by_step: dict[int, list[Move]] = defaultdict(list)
    kind_counts: dict[str, int] = defaultdict(int)
    for e in events:
        kind, p, st = e["kind"], e["payload"], e["step"]
        kind_counts[kind] += 1
        for fl in flows_for(kind, p, ctx):
            fl.step, fl.kind = st, kind
            flows.append(fl)
            flows_by_day[st // spd].append(fl)
        if kind in HOUSEHOLD_KINDS:
            moves_by_step[st].extend(move_for(kind, e["agent_id"], p, cfg["accounts_on"]))

    # 企業(org)部門: org_ledger の revenue_est / wage_paid は**どの主体の残高とも接続していない**。
    # 行列に明示的に載せることで、org 行・列のどれだけが未接続かが1目で見える。
    # ★IF-E2 案B(org_accounting ON)では org が**実在の預金**を持ち、賃金は wage の payer・
    #   売上は spend の payee として既に行列に載っている。ここで revenue_est(推定値)を
    #   重ねると二重計上になるので、ON のランでは擬似フローを出さない。
    for r in (() if cfg.get("org_accounting_on") else run["ledger"]):
        day = int(r.get("day", 0) or 0)
        est, wage = _f(r, "revenue_est"), _f(r, "wage_paid")
        if est:
            fl = Flow(VOID, ORG, est, "org:revenue_est_unfunded", day * spd, "org_ledger")
            flows.append(fl)
            flows_by_day[day].append(fl)
        if wage:
            fl = Flow(ORG, VOID, wage, "org:wage_paid_unfunded", day * spd, "org_ledger")
            flows.append(fl)
            flows_by_day[day].append(fl)

    matrix = build_matrix(flows)
    sums = row_col_sums(matrix)
    leaks = leak_breakdown(flows)

    daily = []
    for day in sorted(flows_by_day):
        m = build_matrix(flows_by_day[day])
        s = row_col_sums(m)
        lk = leak_breakdown(flows_by_day[day])
        daily.append({
            "day": day,
            "total_flow": round(sum(m.values()), 1),
            "created": round(sum(v["created"] for v in lk.values()), 1),
            "destroyed": round(sum(v["destroyed"] for v in lk.values()), 1),
            "net": {k: round(v["net"], 1) for k, v in s.items()},
        })

    checks = {}
    if run["stock"]:
        checks[HOUSEHOLD] = conserve_household(run["stock"], moves_by_step, rel_tol)
    else:
        checks[HOUSEHOLD] = {"sector": HOUSEHOLD, "observable": False,
                             "reason": "l3_snapshots.parquet が無い"}
    if run["budget"]:
        checks[GOVERNMENT] = conserve_government(run["budget"], flows, rel_tol)
    else:
        checks[GOVERNMENT] = {"sector": GOVERNMENT, "observable": False,
                              "reason": "public_budget イベントが 0 件(government OFF)"}
    for s in (ORG, VENTURE, BANK, EXTERNAL, OTHER, VOID):
        checks[s] = {"sector": s, "observable": False, "reason": UNOBSERVABLE_REASON[s]}
    # IF-E2 案B: finance.parquet があれば org / 銀行 / RoW の残高が観測できる = 検査①が成立する。
    # 行政は public_budget 由来の既存検査(内部整合 + 外部整合)を主にし、供託(escrow)を
    # 含む finance 由来の判定を external_finance として**併記**する(既存の判定を壊さない)。
    fin = run.get("finance") or []
    if len(fin) >= 2:
        for s in (ORG, BANK, EXTERNAL, OTHER):
            checks[s] = conserve_sector(fin, flows, s, rel_tol)
        gv_fin = conserve_sector(fin, flows, GOVERNMENT, rel_tol)
        if checks[GOVERNMENT].get("observable"):
            checks[GOVERNMENT]["external_finance"] = gv_fin
        else:
            checks[GOVERNMENT] = gv_fin

    # 突合鍵は (step, agent_id, cat) と (step, cat) だけ。**座標は使えない**ので持たない
    # (match_serve_to_spend の docstring: serve フェーズは _phase_jitter の後に走るため
    #  serve の (x, y) は spend 時点の座標と一致しない)。W2-6 ②で L1 からも読むのをやめた。
    spend_rows = [{"step": e["step"], "agent_id": e["agent_id"],
                   "cat": str(e["payload"].get("cat") or ""), "amount": _f(e["payload"], "amount")}
                  for e in events if e["kind"] == "spend"]
    gap = revenue_gap(run["ledger"], run["serve"], spend_rows)
    # IF-E2 案B: 受け手 org ごとの**実売上**(突合不要 = 支払のその場で確定している)。
    actual_by_payee = revenue_actual_by_payee(events)
    gap["payee_revenue_total"] = round(sum(actual_by_payee.values()), 1)
    gap["n_payee_orgs"] = len(actual_by_payee)
    gap["payee_ratio"] = ((gap["total_revenue_est"] / gap["payee_revenue_total"])
                          if gap["payee_revenue_total"] else None)

    total_flow = sum(matrix.values())
    created = sum(v["created"] for v in leaks.values())
    destroyed = sum(v["destroyed"] for v in leaks.values())
    return {
        "run_dir": run["run_dir"], "config": cfg, "rel_tol": rel_tol,
        "event_counts": dict(sorted(kind_counts.items())),
        "matrix": {f"{a}->{b}": round(v, 1) for (a, b), v in sorted(matrix.items())},
        "sector_sums": {k: {kk: round(vv, 1) for kk, vv in v.items()} for k, v in sums.items()},
        "leaks": {k: {"created": round(v["created"], 1),
                      "destroyed": round(v["destroyed"], 1),
                      "count": int(v["count"])} for k, v in sorted(leaks.items())},
        "totals": {"total_flow": round(total_flow, 1),
                   "money_created": round(created, 1),
                   "money_destroyed": round(destroyed, 1),
                   "leak_share": (created + destroyed) / total_flow if total_flow else 0.0},
        "daily": daily,
        "checks": checks,
        "leak_families": sorted({leak_family(t) for t in leaks}),
        "unclassified_money_kinds": unclassified_money_kinds(run.get("other_money", [])),
        "classifier_consistency": classifier_consistency(
            flows, [m for mv in moves_by_step.values() for m in mv]),
        "revenue_gap": gap,
        # IF-E2 案B: 部門別残高の日次系列(finance.parquet)。期首/期末と RoW チャネル内訳。
        "finance": {
            "n_rows": len(run.get("finance") or []),
            "open": ((run.get("finance") or [{}])[0] if run.get("finance") else {}),
            "close": ((run.get("finance") or [{}])[-1] if run.get("finance") else {}),
        },
        "payee_revenue": {k: round(v, 1) for k, v in sorted(actual_by_payee.items())},
    }


# --------------------------------------------------------------------------- #
# 8. Markdown 出力
# --------------------------------------------------------------------------- #
def _num(v: float) -> str:
    return f"{v:,.1f}"


def render_markdown(rep: dict) -> str:
    L: list[str] = []
    cfg = rep["config"]
    L.append("# 部門別会計行列と貨幣保存の検査(IF-E フェーズ1)")
    L.append("")
    L.append(f"- ラン: `{rep['run_dir']}`")
    L.append(f"- トグル: accounts={cfg['accounts_on']} / government={cfg['government_on']} / "
             f"org_ledger={cfg['org_ledger_on']} / serve.indoor_fields={cfg['indoor_fields_on']} / "
             f"org_accounting={cfg.get('org_accounting_on', False)}")
    L.append(f"- 相対許容誤差 rel_tol = {rep['rel_tol']:g}")
    L.append("- 検査法の出典: Caiani et al. (2016) *JEDC* 69:375-408 §5.2 / Godley & Lavoie (2007)")
    L.append("")

    L.append("## 0. 総額")
    t = rep["totals"]
    L.append("")
    L.append("| 量 | 金額 |")
    L.append("|---|---:|")
    L.append(f"| 総フロー(部門間の記帳合計) | {_num(t['total_flow'])} |")
    L.append(f"| **貨幣の創出**(void → 実在部門) | {_num(t['money_created'])} |")
    L.append(f"| **貨幣の消滅**(実在部門 → void) | {_num(t['money_destroyed'])} |")
    L.append(f"| 漏れ比率 (創出+消滅)/総フロー | {t['leak_share']*100:.2f} % |")
    L.append("")

    L.append("## 1. 検査① 貨幣ストックの保存(部門別)")
    L.append("")
    L.append("> 「各部門の期首残高 + 流入 − 流出 = 期末残高」。残高が世界に実在し観測できる部門だけが判定可能。")
    L.append("")
    L.append("| 部門 | 残高観測 | 総流量 | 残差 | 相対残差 | 判定 |")
    L.append("|---|---|---:|---:|---:|---|")
    for s in SECTORS:
        c = rep["checks"][s]
        if not c.get("observable"):
            L.append(f"| {SECTOR_JA[s]} | ✗ 不可 | — | — | — | {c.get('reason','')} |")
            continue
        L.append(f"| {SECTOR_JA[s]} | ✓ | {_num(c['total_abs_flow'])} | "
                 f"{_num(c['total_residual'])} | {c['relative_residual']:.3e} | "
                 f"{'PASS' if c['pass'] else 'FAIL'} |")
    L.append("")

    hh = rep["checks"][HOUSEHOLD]
    if hh.get("observable"):
        bad = [w for w in hh["windows"] if abs(w["residual"]) > 0.5]
        L.append(f"家計の窓数 = {hh['n_windows']}(L3 スナップショット間隔)。"
                 f"残差 |r|>0.5 円の窓 = {len(bad)} 個。")
        if bad:
            L.append("")
            L.append("| 窓(step) | 対象人数 | Δ残高 | Σフロー | 残差 | pool 入 | pool 出 | 窓内のイベント種 |")
            L.append("|---|---:|---:|---:|---:|---:|---:|---|")
            for w in bad[:25]:
                L.append(f"| {w['from_step']}→{w['to_step']} | {w['n_common']} | "
                         f"{_num(w['delta_stock'])} | {_num(w['flow'])} | {_num(w['residual'])} | "
                         f"{_num(w['pool_in'])} | {_num(w['pool_out'])} | {', '.join(w['kinds'])} |")
            if len(bad) > 25:
                L.append(f"| … | | | | | | | 残り {len(bad)-25} 窓は JSON 側 |")
        L.append("")

    gv = rep["checks"][GOVERNMENT]
    if gv.get("observable"):
        n_bad = sum(1 for lv in gv["levels"].values() for r in lv if abs(r["residual"]) > 0.5)
        L.append(f"行政(a) 内部整合: public_budget の日次記録 "
                 f"{sum(len(v) for v in gv['levels'].values())} 本のうち |残差|>0.5 は {n_bad} 本。")
        ext = gv.get("external") or {}
        if ext.get("applicable"):
            L.append("")
            L.append(f"行政(b) 外部整合(step {ext['from_step']}→{ext['to_step']}): "
                     f"期首 {_num(ext['open_balance'])} + 流入 {_num(ext['matrix_inflow'])} "
                     f"− 流出 {_num(ext['matrix_outflow'])} vs 期末 {_num(ext['close_balance'])} "
                     f"→ 残差 {_num(ext['residual'])}(相対 {ext['relative_residual']:.3e}・"
                     f"{'PASS' if ext['pass'] else 'FAIL'})")
        L.append("")

    cc = rep["classifier_consistency"]
    L.append(f"分類器の自己整合(検査②の家計純額 {_num(cc['matrix_net'])} vs 検査①の残高移動合計 "
             f"{_num(cc['moves_net'])}): 差 {_num(cc['diff'])} → "
             f"{'PASS' if cc['pass'] else 'FAIL(分類器のどちらかに経路の書き漏れ)'}")
    L.append("")

    L.append("## 2. 検査② 部門別フロー行列(行=支払い側 / 列=受け取り側)")
    L.append("")
    L.append("> Caiani の「行と列がゼロに合計される」。本スクリプトは相手方を分類器で明示割り当てするので")
    L.append("> 行列自体は二重記入で構造的にゼロ和になる。**実質のゼロ和違反は `void` 行・列の絶対額**")
    L.append("> = 相手方が世界に存在しない金 = 貨幣の創出/消滅。")
    L.append("")
    hdr = "| from \\ to | " + " | ".join(SECTOR_JA[s] for s in SECTORS) + " | 流出計 |"
    L.append(hdr)
    L.append("|---|" + "---:|" * (len(SECTORS) + 1))
    m = {tuple(k.split("->")): v for k, v in rep["matrix"].items()}
    for a in SECTORS:
        cells = []
        for b in SECTORS:
            v = m.get((a, b), 0.0)
            cells.append(_num(v) if v else "·")
        L.append(f"| **{SECTOR_JA[a]}** | " + " | ".join(cells) + " | "
                 + _num(rep["sector_sums"][a]["outflow"]) + " |")
    L.append("| **流入計** | " + " | ".join(_num(rep["sector_sums"][b]["inflow"])
                                            for b in SECTORS) + " | |")
    L.append("")
    L.append("| 部門 | 流入 | 流出 | 純額(流入−流出) |")
    L.append("|---|---:|---:|---:|")
    for s in SECTORS:
        v = rep["sector_sums"][s]
        L.append(f"| {SECTOR_JA[s]} | {_num(v['inflow'])} | {_num(v['outflow'])} | {_num(v['net'])} |")
    L.append("")

    L.append("## 3. 漏れの内訳(発生イベント種別)")
    L.append("")
    L.append("| タグ | 件数 | 貨幣の創出 | 貨幣の消滅 |")
    L.append("|---|---:|---:|---:|")
    for tag, v in sorted(rep["leaks"].items(),
                         key=lambda kv: -(kv[1]["created"] + kv[1]["destroyed"])):
        L.append(f"| `{tag}` | {v['count']} | {_num(v['created'])} | {_num(v['destroyed'])} |")
    L.append("")
    L.append(f"漏れの経路の族: {', '.join('`%s`' % f for f in rep['leak_families'])}")
    L.append("")
    unc = rep["unclassified_money_kinds"]
    if unc:
        L.append("**未分類の金額キーを持つイベント種**(= 会計検査が見ていない金の経路の候補):")
        L.append("")
        L.append("| kind | 件数 | 金額らしい鍵 |")
        L.append("|---|---:|---|")
        for k, v in unc.items():
            L.append(f"| `{k}` | {v['n']} | {', '.join(v['keys'])} |")
    else:
        L.append("未分類の金額キーを持つイベント種: **なし**(既知の金の経路で閉じている)。")
    L.append("")

    fin = rep.get("finance") or {}
    if fin.get("n_rows"):
        L.append("## 3-b. IF-E2 案B: 部門別残高と rest-of-world(渋谷域外)")
        L.append("")
        L.append("> org はスカラー預金 1 本を持ち(Lengnick 2013 の M=wN / Caiani の σ×予想賃金支払)、")
        L.append("> 受け手が街に居ない金は RoW へ**チャネル別に**落ちる(SNA 2008 §26.2/26.5/26.6)。")
        L.append("> 不変量 **Σ(全主体残高) + RoW 累積 = 一定**(ABCredit.jl 流のスカラー総マネー保存)。")
        L.append("")
        op, cl = fin.get("open") or {}, fin.get("close") or {}
        L.append("| 部門 | 期首残高 | 期末残高 | 変化 |")
        L.append("|---|---:|---:|---:|")
        for label, key in (("企業(org)預金", "org_balance"),
                           ("銀行 capital", "bank_capital"),
                           ("VC ファンド", "vc_balance"),
                           ("行政 予算", "gov_balance"),
                           ("行政 預り金(供託)", "escrow"),
                           ("家計(現金+口座)", "household_balance"),
                           ("RoW 累計流入(街へ)", "row_in"),
                           ("RoW 累計流出(街から)", "row_out"),
                           ("非取引の資産変動(SNA K.5)", "k5_other")):
            a, b = _f(op, key), _f(cl, key)
            L.append(f"| {label} | {_num(a)} | {_num(b)} | {_num(b - a)} |")
        L.append("")
        L.append(f"org: {int(_f(cl, 'org_count'))} 社 / うち預金がマイナス(当座借越) "
                 f"{int(_f(cl, 'org_negative'))} 社 / 最小残高 {_num(_f(cl, 'org_min'))}")
        L.append("")
        try:
            ch = json.loads(cl.get("row_channels") or "{}")
        except (TypeError, ValueError):
            ch = {}
        if ch:
            L.append("**域外(RoW)チャネル別の累積**(= 街が域外にどう依存しているか):")
            L.append("")
            L.append("| チャネル | RoW → 街(流入) | 街 → RoW(流出) |")
            L.append("|---|---:|---:|")
            for name, v in sorted(ch.items(), key=lambda kv: -(kv[1]["in"] + kv[1]["out"])):
                L.append(f"| `{name}` | {_num(v['in'])} | {_num(v['out'])} |")
            L.append("")
        pr = rep.get("payee_revenue") or {}
        if pr:
            top = sorted(pr.items(), key=lambda kv: -kv[1])[:15]
            L.append(f"受け手 org が特定できた消費: **{len(pr)} 社 / 合計 "
                     f"{_num(sum(pr.values()))}**(上位 15 社)")
            L.append("")
            L.append("| org_id | 実売上(spend.payee 由来) |")
            L.append("|---|---:|")
            for oid, v in top:
                L.append(f"| `{oid}` | {_num(v)} |")
            L.append("")

    L.append("## 4. 既知の断絶: revenue_est vs 実際の spend")
    L.append("")
    g = rep["revenue_gap"]
    L.append("> `revenue_est = 日給 × revenue_margin`(scheduler.py `_log_org_output`)は")
    L.append("> **客が払った金と一切接続していない**。org 帰属店舗で実際に発生した spend と比較する。")
    L.append("")
    L.append("| 量 | 金額 |")
    L.append("|---|---:|")
    L.append(f"| org_ledger.revenue_est 合計 | {_num(g['total_revenue_est'])} |")
    L.append(f"| serve で spend へ突き合った金額の合計 | {_num(g['total_spend_actual'])} |")
    L.append(f"| うち **org が特定できた**分(乖離の分母) | {_num(g['known_org_spend'])} |")
    L.append(f"| うち org 不明(スタッフ不在×多義 node) | {_num(g['unknown_org_spend'])} |")
    L.append(f"| **乖離**(revenue_est − org特定 spend) | {_num(g['gap'])} |")
    L.append(f"| 比 est/actual | {('%.2f' % g['ratio']) if g['ratio'] else '—'} |")
    L.append(f"| org_ledger.wage_paid 合計 | {_num(g['total_wage_paid'])} |")
    L.append(f"| serve 件数 / うちスタッフ不在 | {g['n_serve']} / {g['n_serve_unstaffed']} |")
    L.append(f"| serve→spend 突合 成功 / 失敗 / 重複除去 | "
             f"{g['n_serve_matched']} / {g['n_serve_without_spend']} / {g['n_serve_dedup_dropped']} |")
    if g.get("n_payee_orgs"):        # IF-E2 案B: 突合不要の実売上(支払のその場で受け手が確定)
        L.append(f"| **payee 由来の実売上合計**(IF-E2) | {_num(g['payee_revenue_total'])} |")
        L.append(f"| 比 est/payee 実売上 | "
                 f"{('%.2f' % g['payee_ratio']) if g.get('payee_ratio') else '—'} |")
    L.append("")
    if g["rows"]:
        L.append("| org_id | revenue_est | spend 実測 | 乖離 | wage_paid |")
        L.append("|---|---:|---:|---:|---:|")
        for r in sorted(g["rows"], key=lambda r: -abs(r["gap"]))[:30]:
            L.append(f"| `{r['org_id']}` | {_num(r['revenue_est'])} | {_num(r['spend_actual'])} | "
                     f"{_num(r['gap'])} | {_num(r['wage_paid'])} |")
        L.append("")
    else:
        L.append("*(org_ledger.parquet が無い = work.service.ledger.enabled=false。乖離表は空。)*")
        L.append("")

    L.append("## 5. 日次推移")
    L.append("")
    L.append("| 日 | 総フロー | 創出 | 消滅 | 家計 純額 | 行政 純額 |")
    L.append("|---:|---:|---:|---:|---:|---:|")
    for d in rep["daily"]:
        L.append(f"| {d['day']} | {_num(d['total_flow'])} | {_num(d['created'])} | "
                 f"{_num(d['destroyed'])} | {_num(d['net'][HOUSEHOLD])} | "
                 f"{_num(d['net'][GOVERNMENT])} |")
    L.append("")
    L.append("---")
    L.append("")
    L.append("*本スクリプトは読み取り専用(シム本体ゼロタッチ・乱数0・LLM呼0)。*")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# 9. CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir", help="runs/<name>")
    ap.add_argument("--out", help="Markdown の出力先(省略時は標準出力)")
    ap.add_argument("--json", dest="json_out", help="JSON の出力先")
    ap.add_argument("--rel-tol", type=float, default=1e-6,
                    help="検査①の相対許容誤差(既定 1e-6。丸めがあるので絶対ゼロは不成立)")
    args = ap.parse_args(argv)

    run = load_run(Path(args.run_dir))
    rep = analyze(run, rel_tol=args.rel_tol)
    md = render_markdown(rep)
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        print(md)
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(rep, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote {args.json_out}")
    hh = rep["checks"][HOUSEHOLD]
    return 0 if (not hh.get("observable") or hh.get("pass")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
