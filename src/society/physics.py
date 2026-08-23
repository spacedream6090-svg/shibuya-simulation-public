"""P3 境界縫合(竹-4)— グラフ世界とゾーン物理の縫い合わせ + 物理→知覚の翻訳。

正典: docs/plans/source/physics-instructions.md **Part P3** /
      docs/research/physics-engine-selection.md ★P2 決定(2026-08-02)条件 1〜6 /
      docs/plans/cognition-physics-plan.md §4「選定後: P3 境界縫合」/
      docs/plans/highfidelity-3d-physics-plan.md §2 竹-4。
幾何・宣言・ゲート列挙は `world/zones.py`、エンジンは `world/sfm_core.py`(SFM)と
`world/orca_core.py`(ORCA)。本 module は**それらを世界へ縫い付ける層**である。

★ このファイルは src/society 直下(engine/cognition/actions/labeling/world の
  CHECKED_DIRS 外)。engine 側の配線は **中立な 2 箇所**だけ:
    (1) `_phase_move` の先頭で「物理が所有している個体はグラフ移動しない」1 行
        (既定 OFF では属性自体が生えない = `getattr(...) is None` = バイト一致。
         SUMO 配車待ち `_taxi_hold_until` と**同じ既存イディオム**)
    (2) `_phase_move` の**直前**に `physics.phase(sim, step, sim_min)` の 1 行

2 層タイムライン(P3(2) の時間整合)
------------------------------------
世界 tick は 10 分。物理は dt_sub(既定 0.05 s)で **その step の中を通しで積分する**。
  - 積分は **step 内で同期完了**する(次 step へ半端な状態を残さない。ゾーンに残る
    個体は「位置と速度」という完結した状態だけを持ち越す)= 第76/B3 と同じ構造。
  - 流入・流出の**確定**は世界 tick の境界のみ(= この関数の呼ばれた 1 回の中)。
  - サブステップ上限 `max_sub_steps` で必ず止まる(600 s / 0.05 s = 12000 が既定上限)。
    ゾーンが空になったら**その場で打ち切る**(誰も居ない時間帯のコストはゼロ)。

所有権(exclusive ownership)
----------------------------
`agent._phys_zone` が単一値であることが排他所有そのもの。物理が所有している間、
その個体は **グラフ移動をしない**(= 同一時刻に 2 つのモデルに属さない)。
ゾーン間の直接移籍は存在しない: いったんグラフへ返してからでないと次のゾーンへ入れない
(= P2 決定 条件 6「ゾーン非重複 + ゲート経由のみの移籍」)。

ゲート境界プロトコル(guarded・P2 決定 条件 5)
----------------------------------------------
ベンチ `reference/physics_bench` の実測(dt=0.1・幅3m 通路・200 体):

  | 方式・候補 | ゲート帯 accel p99 | 内部 accel p99 | ゲート帯 反転 | 最小体表間 |
  |---|---|---|---|---|
  | blind · SFM   | 14.25 | 5.63 | 0.749 | +0.024 |
  | guarded · SFM |  9.34 | 5.72 | 0.695 | +0.050 |
  | blind · ORCA  |  7.17 | 2.12 | 0.246 | −0.207 |
  | **guarded · ORCA** | **5.15** | 2.18 | **0.123** | −0.074 |

→ guarded(= 入口が空いているときだけ入れる + 速度を引き継ぐ)を採る。実装:
  **流入**: 経路がゾーンへ入る step の頭で所有を移す。位置は**その個体の現在座標
    そのもの**(= 瞬間移動が原理的に起きない)、速度はグラフ側の移動速度から連続に
    引き継ぐ(mode 速度 × 混雑係数 ÷ step 秒。上限は個体の v_max)。
    入口が塞がっていれば **入れない**(待たせる)。文献の標準規約と同じ:
    「置けなければ移管しない」(TransiTUM: 受け皿が見つからなければ今回は移管しない)。
    信号のあるゾーンでは、**青(+青点滅)の間にしか入場を許さない**
    (= 赤で縁石に溜まり、青で一斉に横断する見せ場が SignalGate から自然に出る)。
  **流出**: ゾーンの外へ出た(または出口ゲートノードに到達した)時点で、物理座標を
    経路の折れ線へ**射影**してグラフ状態 (node, route, edge_offset) を復元する。
    射影距離 = グラフ復帰時の位置の跳び。`gate.handover_jump_max_m` を超えたら
    記録に `far: true` を付ける(監視。落とさない)。

境界連続性の指標(P2 決定 条件 5 → P3 の受入テストへ昇格)
------------------------------------------------------------
`jump_max`(1 サブステップ最大変位)/ `accel_p99`(|Δv|/dt の 99 分位)/
`reversal_rate`(進行方向成分の符号反転 / 体·秒)を **ゲート帯と内部で分けて**測る。
分けるのは文献の標準規約に従うため: 変換で生じたアーティファクトは緩和帯で減衰するので、
**緩和帯で採った統計は本体の物理の評価に使わない**(Biedermann et al., *Towards
TransiTUM*, Transportation Research Procedia 2 (2014) 495–500)。
分位点は固定ビンのヒストグラム(0.1 m/s² 刻み)から採る = **メモリ O(1)・resume 安全**。

物理 → 知覚(P3(3))
--------------------
測った値は `Perception.body` の **既存の 3 欄**(`blocked` / `contact` / `local_density`)
に入れる。第85 の契約により `prompt_kwargs()` は body を出さないので、
**プロンプト文字列は 1 バイトも変わらない**(構造による no-fingerprint 保証)。
  blocked       … 1 − 実速度/希望速度(0=阻害なし、1=完全に進めない)
  contact       … 体表間が contact_gap_m 以下だったサブステップの割合 [0,1]
  local_density … 半径 density_radius_m の円内の人数 / その円の面積 [人/m²]
将来の発火チャンネル接続(ext.crowd 系)は `crowd_override()` として **配線だけ**用意し、
`physics.perception.channels`(既定 false)で閉じてある。

決定論(P3(5) strict 等級)
---------------------------
固定 dt_sub・処理順は **agent.id 昇順**・浮動小数の集約順序固定(np.bincount)・
乱数は用途別 named stream **"physics"**((zone_id, step) キー)からしか引かない。
RngHub はステートレスなので、新 stream を足しても既存の draw 順は 1 つも動かない
(第75 shuffle_partners と同じ規約)。ORCA の `pref_noise` を 0 にすれば乱数はゼロ本。

歩行物理の較正(P4-2 / P4-3。2026-08-05)
-----------------------------------------
`physics.sfm` の 3 ブロック(`far_field` / `v_of_s` / `wall`)を読み、SFM ゾーンの
エンジンに適用する(`_CalibratedCrowd`)。**3 つとも既定 OFF/現行値**で、そのとき
`_calib_kwargs()` は空 dict を返し `_build_engine` は素の `sfm_core.Crowd` を
従来の引数のまま構築する = 演算順も引数も 1 バイト変わらない(golden 無風)。
較正の実測は `reference/physics_bench/`(P4-1 = `calib_results.json` /
P4-3 = `calib_p43_results.json`)。本体の `_CalibratedCrowd` はベンチの
`ExtendedCrowd` と **バイト一致**する(tests/test_physics_calib.py が固定)。

物理痩身 第一段A(2026-08-16。docs/research/crowd-attention-physics.md「案A」)
------------------------------------------------------------------------------
**動力学を 1 ビットも変えない同値変換**だけを入れた(近傍 cap 値の変更・遮蔽・密度場は
次レーンの仕事で、ここでは一切やらない)。本 module で変えたのは 4 箇所:
  - `_CalibratedCrowd._far_forces` … 候補列挙をセル法へ(旧全ペア版は `_far_forces_dense`
    として残す。ベンチ `ExtendedCrowd` とのバイト一致検査がそのまま同値の証拠になる)
  - `_accumulate` … 全ペア距離行列(O(N²))と個体ごとの Python ループを撤去。
    集計はカウント・bool・min だけ(順序非依存)で、唯一の浮動小数の累積和 `speed_sum` は
    加算順序を変えていない
  - `_admit` … 入口の占有判定 O(W·M)/サブステップ をセル法の一括判定へ(bool の or =順序非依存)
    ※ ここで潰したのは「**呼び出し時点の**在場者」ぶんだけで、ループ内で増える
       「入場済み」ぶんの総当たり O(W²) は第154 まで残っていた(`_AdmitCells`)。
  - `phase` … `_by_id` の全体 sort を **1 step 1 回**に(ゾーン数×2 回 → 1 回。走査順は不変)
検収は tests/test_physics_hash.py(旧実装を参照実装として保持し全サブステップを突合)。

物理痩身 第二段B/C(2026-08-17。docs/research/crowd-attention-physics.md「案B」「案C」)
------------------------------------------------------------------------------------
第一段A が「意味論を変えない痩身」だったのに対し、第二段は**認知の二層をそのまま物理へ写す**:

  B 認知的近傍(`physics.cognitive.enabled`。既定 OFF)
    対人斥力(SFM)と ORCA の近傍選抜を「距離最近傍 k 体」から
    **視野円錐 + セクタ遮蔽 + 上限 k**(= 見えている少数近傍)へ差し替える。
    実装は `world/sfm_core.visual_neighbors` の 1 本で、SFM/ORCA はそれを呼ぶだけ。
    本 module は conf を読んでエンジンへ引数を渡す配線しか持たない。

  C 遠方場の密度場化(`physics.density_far.enabled`。既定 OFF・far_field ON が前提)
    較正 far 項 `_far_forces`(体表間 4.725 m のペア和)を**密度場由来の連続体力**へ置換する。
    遠方の群衆は個体の列挙ではなく**アンサンブル統計(密度・流れ)として知覚される**
    (Whitney & Yamanashi Leib 2018)という認知の主張の、最も文字通りの実装であり、
    工学側の前例は Hughes 2002 / Treuille 2006 / Narain 2009。
    ★接続係数は**解析的に導く**(捏造しない)。ペア版の far 項を密度 ρ(x) の連続体で書くと

        f_far(x) = −∫ ρ(x+u)·K(|u|)·w(e·û)·û du,   K(r) = m·a2·exp((rr−r)/b2)·taper(r)

    で、ρ を 1 次まで展開し û の角度積分を実行すると **2 項ちょうど**残る:

        f_drag = −(1−λ)·(π/2)·I₁·ρ(x)·e      … 一様密度でも残る「混雑の抵抗」
        f_grad = −(1+λ)·(π/2)·I₂·∇ρ(x)       … 混んでいる方から押し返される勾配力
        I₁ = ∫₀^R K(r)·r dr,  I₂ = ∫₀^R K(r)·r² dr    (R = rr + far_cutoff)

    (û の 3 次モーメントが 0 なので、異方性 λ は上の 2 つの係数へ**厳密に**畳み込まれる。)
    → 一様流ではペア版と同じ合力になる = 基本図の作業点が保存される。これが
      「現行 far 項の役割(混雑回避のバイアス)を保存する最小構成」の中身である。
    格子は物理 step より粗い周期(`update_every` サブステップ)で作り直し、
    各サブステップでは**現在位置で双線形サンプル**する(場は粗く、読み出しは細かく)。
    ただしエンジンは入場・退場のたびに作り直される(`_run_zone`)ので、素のままでは
    その周期が churn のたびに 0 へ巻き戻る。`density_far.carry_grid`(既定 false)は
    同一ゾーン・同一 step の作り直しで場を引き継いでそれを止める(`_carry_field`)。

R1: どちらのトグルも既定 OFF で、OFF のときは新しい関数が 1 度も呼ばれない
    = `_build_engine` は第一段A までと 1 バイト同じ呼び出しをする(golden 無風)。

物理 P4 格子外延の有界化(`physics.density_far.clip_margin_m`。既定 0 = 無効。2026-08-21)
--------------------------------------------------------------------------------------
第145(P3)の調査で判明した密度場の真犯人は平滑でも再構築周期でもなく**格子の外延**だった:
外延は「ゾーンを所有中の全メンバーの外接矩形」で決まるのに、所有は**経路がゾーンを通る
個体に現在地(数百 m 先)から**始まる。結果、38×29 m の広場(hachiko_square)に
**平均 117 万セル(~1 km 角・float64 1 枚 9.4 MB)**の場を毎回作っていた
(P3 の高速化後も physics.phase の 67%)。
`clip_margin_m > 0` のとき、外延を「ゾーン polygon の外接矩形 ± margin」との**共通部分**へ
落とす(`_field_clip` → `_CalibratedCrowd.field_clip`)。
  堆積   … 矩形の外のメンバーは数えない。**np.clip で縁のセルへ押し込まない**
           (押し込むと数百 m 先の一人歩きの密度が縁に山積みになり場が嘘になる)。
  読み出し… 矩形の外のメンバーは ρ=0・∇ρ=0(= 遠方場の力ゼロ)。孤立して経路を歩いている
           個体の足元の局所密度は実際ほぼ 0 なので、これが物理的に正しい値である。
  連続性 … 堆積の外側には既存の `pad`(= field_blur + 2)ぶんのゼロ詰め余白が付くので、
           外延の縁で階段状の勾配は立たない。margin はさらにその外側にある。
★クリップの厳密な意味は「矩形の外のメンバーを名簿から外した場」であって、それ以上でも
  以下でもない(格子も値も、外した名簿で作った場とビット一致する)。落とした堆積が場へ
  届く範囲は矩形の縁から (blur + 2) セル(平滑 blur + 中心差分 1 + 双線形 1)しかないので、
  margin をそれ以上に採れば**切り落としの影響はポリゴンの中まで届かない**。
★ON は世界(軌跡)を変える。理由は 2 つあり、どちらも正直に書いておく:
  (a) 格子の原点は「在場者の最小座標」で決まるので、遠くの所有メンバーが原点を決めていた
      ぶんだけ**離散化が載り替わる**(同じ密度でもセル境界の切れ方が変わる)。
  (b) 圏外の個体は**自分自身の堆積による見かけの抵抗**を受けなくなる(場版だけに在る
      自己相互作用。ペア版は i≠j なので持たない)。孤立歩行者の ρ が 0 になるのは
      こちらの方が正しい。
  既定 0 = 無効。

物理 2 レバー(第154。2026-08-23。**どちらも既定 OFF = 現行と 1 バイト同一**)
--------------------------------------------------------------------------------
py-spy 250k v7b の実測で step 時間の 46.6% が物理(`_run_zone` の `engine.step` 本体
34.7% / `orca_core.step` 22.4%)だった。内訳は「夕方ラッシュのゾーン内人数 × サブ
ステップ数(dt=0.1 で 6,000/step)」の正直な掛け算なので、掛ける側を両方から削る。

  レバーB 密度適応 dt(`physics.adaptive_dt`。既定 OFF)
    サブステップの**塊ごと**(`recheck_every` 回に 1 度)に在場者数から dt 係数を
    決定論選択し、`dt_eff = dt × 係数` で積分する。積分時間の総量は保存し
    (Σ dt_eff = n_sub·dt)、端数は最終塊で吸収する = 世界時計とずれない。
    根拠は基本図: ρ>2 人/m² で歩速 <0.5 m/s → dt=0.4 でも 1 サブステップ変位 <20 cm
    = 粗い刻みで失う精度が最小の領域。閑散時は係数 1 = コストゼロ。
    ★ゾーン宣言の `dt_sub` 上限 0.1 は「dt≥0.1 は ρ≥1.5 で SFM が後退爆発」という
      P2 実測から来ている。適応 dt は**混雑時にだけそれを意図的に超える**。
      `scripts/bench_physics_levers.py` の実測(0.71 人/m²・係数 2)では
        ORCA … 体表間 +0.066 → +0.049 m(重なりゼロを維持)
        SFM  … 体表間 +0.003 → **−0.171 m(重なり)**・壁クリアランス
               +0.102 → **−0.105 m(壁貫通)**・平均速さ 0.40 → 0.62 m/s
      = P2 の警告がそのまま再現した。よって `adaptive_dt.engines` で**適用エンジンを
      絞れる**ようにしてある(本選は `[orca]`)。ORCA が粗い刻みに強いのは、衝突回避を
      速度層(半平面)で解き、離散化で生じた重なりを位置層(決定論的 push-apart)で
      毎サブステップ潰す二層構成だから。SFM は斥力の陽的積分そのものが刻みに依存する。

  レバーD 近傍 cap の引き下げ(`physics.neighbor_cap`。既定 0 = ゾーン宣言のまま)
    >0 なら全ゾーンの `neighbor_cap` を置き換え、`physics.cognitive` が ON なら
    その k も `min(neighbors, cap)` へ絞る(= **作用点を 1 つに畳む**)。
    認知的近傍が ON のとき `neighbor_cap` は 1 度も読まれない(sfm_core の
    `_repulsion_cognitive` と orca_core.step はどちらも cog_neighbors を使う)ので、
    ここを揃えないと「cap を下げたのに何も起きない」= 二重 cap の罠になる。
    根拠: Ballerini et al. 2008(位相的近傍 6-7)。引き下げは現実整合の方向であって
    粗視化ではない(docs/research/crowd-attention-physics.md の正典)。

  レバーS 事後分離パスの反復上限(`physics.separation_iters`。既定 0 = ゾーン宣言のまま)
    >0 なら全 ORCA ゾーンの `orca.separation_iters`(既定 64)を置き換える。
    ORCA は「速度層(半平面 LP)+ 位置層(決定論的 push-apart)」の二層で、位置層は
    重なりが消えるか上限反復に達するまで (i<j) 辞書順の逐次 Gauss-Seidel を回す。
    ★実測(29×29 m・cap 7・dt 0.4): **位置層が ORCA 時間の ~84%**(3,000 体で
      上限 64 → 0 が 16.6 s → 2.6 s)= B/D を掛けたあとの最大の残り。
    削れば残留めり込みが増える(n=1,500 で 64→16 は体表間 −0.006 m = 実質無害、
    n=3,000 では −0.188 m)。上限に当たったかは `sep_iters_max` で監視できる。

R1(既定 OFF = 完全 no-op)
---------------------------
`physics.zones_enabled: false`(既定)または `physics.zones: []`(既定)のとき:
本 module の全関数は即 return し、`agent._phys_*` 属性は 1 つも生えず、
`sim._phys_state` も生えない。= 新イベント 0 件・L2 に列なし・乱数消費不変・
LLM 呼数不変・**L1 バイト一致**。
"""
from __future__ import annotations

import math

import numpy as np

from .observer.schema import Event
from .world import orca_core as _orca
from .world import sfm_core as _sfm
from .world import zones as _zones
from .world.indoor_flow import body_radius, desired_speed

# 加速度ヒストグラム(分位点を O(1) メモリで採る)
_ACC_BIN = 0.1            # ビン幅 [m/s²]
_ACC_BINS = 1000          # 0 〜 100 m/s²(超過は最終ビンへ)
STREAM = "physics"        # 用途別 named stream の名前(R1)
_DT_DUST = 1e-6           # 適応 dt の端数吸収しきい(dt に対する比。浮動小数の塵の畳み込み)


# --------------------------------------------------------------------------- #
# 有効判定(唯一のゲート)
# --------------------------------------------------------------------------- #
def enabled(sim) -> bool:
    """物理ゾーンが 1 つでも据わっているか。既定 OFF = False = 全経路が閉じる。"""
    cfg = getattr(sim, "physcfg", None)
    return bool(cfg and cfg["zones"])


def owned(agent) -> bool:
    """この個体を物理が所有しているか(= グラフ移動をさせない)。

    ★ 既定 OFF では `_phys_zone` 属性そのものが生えない → 常に False → `_phase_move` は
      1 バイトも挙動が変わらない(`_taxi_hold_until` と同じイディオム)。
    """
    return getattr(agent, "_phys_zone", None) is not None


def budget_scale(sim, agent, step: int) -> float:
    """`_phase_move` の移動予算にかける係数(既定 1.0 = 従来と完全同一)。

    ★ 実測で見つけた**二重移動**の修正: ゾーンを step の途中で抜けた個体は、その step の
      うちに `_phase_move` へ戻ってくる。何もしないとグラフ側が **10 分ぶんの予算を丸ごと**
      与えてしまい、「物理で歩いた距離 + グラフで歩いた距離」の二重計上になる
      (mock 実測: 1 step で入場→退場した個体が move_segment も出していた)。
      物理で消費した秒数ぶんを予算から差し引くのが正しい 2 層タイムラインの会計。
    OFF のランでは `_phys_used_step` 属性が生えない → 常に 1.0 → バイト一致。
    """
    if getattr(agent, "_phys_used_step", None) != step:
        return 1.0
    used = float(getattr(agent, "_phys_used_s", 0.0))
    total = float(sim.clock.step_seconds)
    return max(0.0, min(1.0, 1.0 - used / total)) if total > 0 else 1.0


# --------------------------------------------------------------------------- #
# 状態(checkpoint の中央管理対象。個体側の状態は agents pickle に自然同梱)
# --------------------------------------------------------------------------- #
def _new_state() -> dict:
    return {
        "enter_total": 0, "exit_total": 0, "forced_total": 0, "wait_total": 0,
        # 滞在時間は **累積** で持つ(per-step の平均にすると「退出ゼロの step」が欠測になり、
        # L2 parquet のセグメント間で列の型が null / double に割れて resume 時の結合が壊れる。
        # 累積なら ON の間ずっと数値 = 型が安定 = resume 安全)。
        "dwell_sum_s": 0.0, "dwell_n": 0,
        "by_zone": {},          # zone_id -> {occupancy, occupancy_mean, density, …}
        "cont": _new_cont(),    # 境界連続性(ゲート帯 / 内部)
        "min_gap_m": None,      # 全ゾーン通算の体表間最小すき間 [m]
        "sep_iters_max": 0,     # 分離パスの最大反復回数
        "handover_jump_max_m": 0.0,
        "sub_steps_total": 0,
    }


def _new_cont() -> dict:
    return {
        "gate": {"hist": [0] * _ACC_BINS, "n": 0, "flip": 0, "samples": 0},
        "interior": {"hist": [0] * _ACC_BINS, "n": 0, "flip": 0, "samples": 0},
        "jump_max_m": 0.0,
        "dt_sub": 0.0,
    }


def _state(sim) -> dict:
    st = getattr(sim, "_phys_state", None)
    if st is None:
        st = _new_state()
        sim._phys_state = st
    return st


def state_of(sim):
    """checkpoint 用の状態(既定 OFF では None = 旧 checkpoint 互換)。"""
    return getattr(sim, "_phys_state", None)


def restore_state(sim, blob) -> None:
    if blob is not None:
        sim._phys_state = blob


# --------------------------------------------------------------------------- #
# L2 スカラー(ON のときだけ値が出る。OFF は None = 列なし)
# --------------------------------------------------------------------------- #
def scalars(sim) -> dict:
    """L2 の集約列。OFF(または 1 度も回っていない)なら空 dict = 列なし。"""
    st = getattr(sim, "_phys_state", None)
    if not st:
        return {}
    occ = 0
    dens = 0.0
    n_zone = 0
    for zid in sorted(st["by_zone"]):
        z = st["by_zone"][zid]
        occ += int(z.get("occupancy", 0))
        dens += float(z.get("density", 0.0))
        n_zone += 1
    n = int(st["dwell_n"])
    return {
        "zone_occupancy": int(occ),
        "zone_density_mean": (dens / n_zone) if n_zone else 0.0,
        # 累積平均(退出がまだ 1 件も無ければ 0.0)。**型が step ごとに揺れない**ことが要件。
        "zone_dwell_mean_s": (float(st["dwell_sum_s"]) / n) if n else 0.0,
        "zone_gate_enter_total": int(st["enter_total"]),
        "zone_gate_exit_total": int(st["exit_total"]),
    }


def continuity(sim) -> dict:
    """境界連続性の実測(テストが閾値化する)。OFF なら空 dict。

    返す値:
      gate_accel_p99 / interior_accel_p99 [m/s²]  … ヒストグラム由来(分解能 0.1)
      gate_reversal_rate / interior_reversal_rate [回/体·秒]
      jump_max_m [m]                              … 1 サブステップ最大変位
      handover_jump_max_m [m]                     … グラフ復帰時の位置の跳び
      min_gap_m [m]                               … 体表間の最小すき間(負 = 重なり)
    """
    st = getattr(sim, "_phys_state", None)
    if not st:
        return {}
    c = st["cont"]
    dt = float(c["dt_sub"]) or 1.0
    out = {"jump_max_m": float(c["jump_max_m"]),
           "handover_jump_max_m": float(st["handover_jump_max_m"]),
           "min_gap_m": st["min_gap_m"],
           "sep_iters_max": int(st["sep_iters_max"]),
           "sub_steps_total": int(st["sub_steps_total"])}
    for band in ("gate", "interior"):
        b = c[band]
        out[f"{band}_accel_p99"] = _hist_quantile(b["hist"], b["n"], 0.99)
        out[f"{band}_accel_max"] = _hist_max(b["hist"])
        # 分母は「サンプル秒」。適応 dt(第154 レバーB)が ON だったランでは
        # サブステップ幅が一定でないので実 dt の積算値 `sec` を使う。
        # OFF(= `sec` が無い / 0.0)では従来どおり samples × dt_sub = 1 ビット同じ。
        secs = float(b.get("sec", 0.0)) or (float(b["samples"]) * dt)
        out[f"{band}_reversal_rate"] = (float(b["flip"]) / secs) if secs > 0 else None
        out[f"{band}_samples"] = int(b["samples"])
    return out


def _hist_quantile(hist, n, q):
    if not n:
        return None
    target = q * n
    acc = 0
    for i, c in enumerate(hist):
        acc += c
        if acc >= target:
            return (i + 1) * _ACC_BIN          # ビン上端(保守側)
    return len(hist) * _ACC_BIN


def _hist_max(hist):
    for i in range(len(hist) - 1, -1, -1):
        if hist[i]:
            return (i + 1) * _ACC_BIN
    return None


# --------------------------------------------------------------------------- #
# 物理 → 知覚(P3(3))
# --------------------------------------------------------------------------- #
def body_of(agent) -> dict | None:
    """直近のゾーン滞在で実測した身体項目(blocked / contact / local_density)。

    ゾーンに一度も入っていない個体は None(= 欠測。**0 で埋めない**)。
    """
    return getattr(agent, "_phys_body", None)


def crowd_override(sim, agent, value):
    """発火チャンネル `ext.crowd_local` への物理値の差し込み(**配線のみ・既定 OFF**)。

    `physics.perception.channels: true` のときだけ、物理で実測した局所密度から
    「半径 density_radius_m の円内の人数」を返す。既定 false では引数をそのまま返す
    = channels.observe は 1 バイトも挙動が変わらない。
    ★ ON にすると観測チャンネルの意味が「同席人数」から「実測近傍人数」へ変わる=
      σ の較正(第80)をやり直す必要がある。だから既定は OFF で据え置く。
    """
    cfg = getattr(sim, "physcfg", None)
    if not cfg or not cfg["perception"]["channels"]:
        return value
    body = getattr(agent, "_phys_body", None)
    if not body or body.get("local_density") is None:
        return value
    r = float(cfg["perception"]["density_radius_m"])
    return float(body["local_density"]) * math.pi * r * r


# --------------------------------------------------------------------------- #
# 本体: 1 世界 step のゾーン実行(単一の作用点)
# --------------------------------------------------------------------------- #
def phase(sim, step: int, sim_min: int) -> None:
    """全ゾーンを 1 世界 step ぶん回す。**`_phase_move` の直前**に呼ぶこと。

    直前に呼ぶ理由: この時点の (x, y) が「この step の開始時の位置」であり、
    2 層タイムラインの下層(dt_sub)はまさにこの step の 600 秒を刻むから。
    物理が所有した個体は `_phase_move` が飛ばす(= 二重に動かない)。
    """
    if not enabled(sim):
        return
    st = _state(sim)
    st["by_zone"] = {}                      # per-step(累積しない = resume 安全)
    # ★id 昇順の走査列は **1 step に 1 回**だけ作る(ゾーン数×2 回の全体 sort を廃す)。
    #   `_run_zone` は sim.agents を増減させないので、全ゾーンで同じ列を使える
    #   = 走査順は 1 つも変わらない(純粋な同値変換)。
    ordered = _by_id(sim.agents)
    for zone in sim.physcfg["zones"]:
        _run_zone(sim, zone, step, sim_min, st, ordered)


# --------------------------------------------------------------------------- #
def _run_zone(sim, zone, step: int, sim_min: int, st: dict, ordered=None) -> None:
    graph = sim.city.graph
    if ordered is None:
        ordered = _by_id(sim.agents)
    gates = _gate_nodes(sim, zone)
    gate_xy = tuple(sim.city.node_xy(n) for n in gates)
    step_seconds = float(sim.clock.step_seconds)
    dt = zone.dt_sub
    n_sub = min(int(zone.max_sub_steps), max(1, int(round(step_seconds / dt))))
    # ★第154 レバーB(密度適応 dt)。既定 OFF では `adapt is None` = 下の分岐を
    #   1 度も通らない = t も dt も従来と同じ式 = 1 バイト同一。
    adapt = _adaptive_of(sim, zone)
    total_s = n_sub * dt            # この step で覆う**積分時間の総量** [s](係数に依らず不変)

    members: list[dict] = []      # 積分対象(ゾーン内)
    waiting: list[dict] = []      # 入場待ち(guarded で弾かれた個体)

    # ---- (1) 既に所有している個体を回収 ------------------------------------ #
    for agent in ordered:
        if getattr(agent, "_phys_zone", None) != zone.id:
            continue
        rec = _record_of(agent)
        if agent.sleeping or agent.loc != "street":
            # 物理の外側の事情で状態が変わった個体は安全に手放す(所有を握り続けない)。
            # グラフ状態は**触らない**(他フェーズが既に据えた node/route を上書きしない)。
            _release(sim, zone, agent, step, sim_min, st, reason="detached",
                     elapsed_s=rec["elapsed_s"], rec=rec, restore=False)
            continue
        rec["step_n"] = 0            # **この step で**積分したサブステップ数(会計の基準)
        (waiting if rec["waiting"] else members).append(rec)

    # ---- (2) 新規流入の候補(guarded ゲートの待機列へ)---------------------- #
    for agent in ordered:
        if getattr(agent, "_phys_zone", None) is not None:
            continue                                  # 他ゾーン所有 or 既に本ゾーン
        if agent.loc != "street" or agent.sleeping or not agent.route:
            continue
        if getattr(agent, "_taxi_hold_until", -1) > step:
            continue                                  # 配車待ちは動かさない(既存規約)
        span = _zones.route_span(zone, graph, agent.node, agent.route)
        if span is None:
            continue                                  # ゾーンを通り抜けない経路 = 所有しない
        path, rest = span
        rec = _admit_record(sim, zone, agent, path, rest, step)
        agent._phys_zone = zone.id
        _save_record(rec)
        waiting.append(rec)

    if not members and not waiting:
        return

    signal = (_sfm.SignalGate(**zone.signal) if zone.signal else None)
    rng = (sim.hub.stream(STREAM, zone.id, int(step))
           if _needs_rng(zone) else None)
    base_sec = float(sim_min) * 60.0
    pcfg = sim.physcfg["perception"]

    # ★P4-2/P4-3 の較正(既定は空 dict = 従来の sfm_core.Crowd 経路そのまま)。
    #   SFM ゾーンにだけ効く(ORCA ゾーンでは _build_engine が無視する)。
    calib = _calib_kwargs(sim, zone)
    cog = _cog_kwargs(sim)
    carry = _carry_grid_on(sim)
    engine = _build_engine(zone, members, rng, calib, cog) if members else None
    cont = st["cont"]
    cont["dt_sub"] = dt
    occ_sum = 0
    sub_done = 0

    # ---- (3) サブステップ・ループ(2 層タイムラインの下層)------------------ #
    # 適応 dt の会計(OFF では cum=None・dt_eff=dt・t=k·dt = 従来の式そのまま)。
    #   cum … `_accumulate` を呼ぶたびに「この step で積分した累積秒」を積む列。
    #         個体の滞在秒は「参加した最後の m 回ぶん」= cum[-1] − cum[-1−m] で厳密に出る
    #         (在場者は入場から退場まで連続した回に参加するので m 回は必ず末尾の連続塊)。
    cum: list | None = [] if adapt is not None else None
    dt_eff = dt
    factor = 1.0
    t = 0.0
    int_s = 0.0                     # 積分済み秒(信号待ちの空回りは含めない)
    k = 0
    while k < n_sub:
        if adapt is None:
            t = k * dt
        else:
            if k % adapt[1] == 0:   # 塊の頭でだけ係数を引き直す(毎サブステップは高い)
                factor = _dt_factor(adapt[0], len(members))
            dt_eff, t_next = next_dt(t, total_s, dt, factor)
            if dt_eff <= 0.0:
                break
        # (3a) 入場(guarded + 信号)。id 昇順 = 決定論。
        if waiting:
            if engine is not None:
                _writeback(members, engine)   # 入場の占有判定は**最新の位置**で行う
            if _admit(sim, zone, waiting, members, signal, base_sec + t, t,
                      step, sim_min, st):
                prev_engine = engine
                engine = _build_engine(zone, members, rng, calib, cog)
                if carry:
                    _carry_field(prev_engine, engine)
        if not members:
            if not waiting:
                break                                 # ゾーンが空 = その場で打ち切り
            sub_done += 1                             # 信号待ち: 時間だけ進める
            k += 1
            if adapt is not None:
                t = t_next
            continue
        # (3b) 積分
        prev_pos = engine.pos.copy()
        prev_vel = engine.vel.copy()
        engine.step(dt_eff)
        sub_done += 1
        occ_sum += len(members)
        _accumulate(zone, members, engine, prev_pos, prev_vel, dt_eff, gate_xy,
                    cont, st, pcfg, dt_acc=(dt_eff if cum is not None else None))
        if cum is not None:
            int_s += dt_eff
            cum.append(int_s)
        # (3c) 通過点の前進 + 退場判定
        released = _advance_and_collect(sim, zone, members, engine)
        if released:
            _writeback(members, engine)
            _drop_recs(members, released)   # ← 1 件ずつ remove すると dataclass `__eq__` が走る
            for rec in released:
                # この step で実際に積分した秒数(= step 途中で入場した個体でも正しい)
                used = _used_s(rec, dt, cum)
                rec["elapsed_s"] += used
                _save_record(rec)
                _release(sim, zone, rec["agent"], step, sim_min, st,
                         reason="gate", elapsed_s=rec["elapsed_s"], rec=rec,
                         used_s=used)
            prev_engine = engine
            engine = _build_engine(zone, members, rng, calib, cog) if members else None
            if carry:
                _carry_field(prev_engine, engine)
        if engine is None and not waiting:
            break
        k += 1
        if adapt is not None:
            t = t_next

    # ---- (4) step 内で終わらなかった個体は状態を持ち越す(同期完了)--------- #
    if members and engine is not None:
        _writeback(members, engine)
    forced: list = []
    for rec in list(members):
        used = _used_s(rec, dt, cum)
        rec["elapsed_s"] += used
        _save_record(rec)
        if _maybe_force(sim, zone, rec, step, sim_min, st, used_s=used):
            forced.append(rec)              # 除去は下で 1 回(`_drop_recs` の docstring)
    _drop_recs(members, forced)
    forced_wait: list = []
    for rec in list(waiting):
        rec["wait_steps"] += 1
        _save_record(rec)
        st["wait_total"] += 1
        if _maybe_force_wait(sim, zone, rec, step, sim_min, st):
            forced_wait.append(rec)
    _drop_recs(waiting, forced_wait)

    st["sub_steps_total"] += sub_done
    area = zone.walkable_area_m2 or zone.area_m2() or 1.0
    # ★適応 dt(第154 レバーB)が ON のとき、この平均は「サブステップ重み」であって
    #   「時間重み」ではない(1 サブステップの長さが塊ごとに変わるため)。係数は塊
    #   (既定 20 サブステップ)単位でしか動かないので偏りは 2 次だが、厳密な時間重みに
    #   したければ Σ(len×dt_eff)/Σdt_eff にする必要がある。**OFF では従来と 1 ビット同じ**
    #   なので、ここは既定側のバイト一致を優先して式を変えていない。
    occ_mean = (occ_sum / sub_done) if sub_done else 0.0
    st["by_zone"][zone.id] = {
        "occupancy": len(members),
        "occupancy_mean": occ_mean,
        "density": occ_mean / area,
        "waiting": len(waiting),
        "sub_steps": sub_done,
    }


# --------------------------------------------------------------------------- #
# 密度適応 dt(混雑 LOD。第154 レバーB。**既定 OFF = dt 固定 = 1 バイト同一**)
# --------------------------------------------------------------------------- #
# 何をするか: ゾーン内の在場者数が多い塊のあいだだけサブステップ幅を dt×係数 へ粗くする。
#   **積分時間の総量は保存する**(Σ dt_eff = n_sub·dt = この step が覆うべき秒数)。
#   端数は最終塊で吸収するので「1 秒も失わない」= 世界時計と物理時計はずれない。
# なぜ粗くしてよいか: 基本図(Weidmann 1993 ほか)で密度 2 人/m² を超えると歩速は
#   0.5 m/s を下回る。そこでの 1 サブステップ変位は dt=0.4 s でも 20 cm 未満で、
#   **粗い刻みで失う精度が最小の領域**である。閑散時は係数 1 = コストゼロ。
# 決定論: 係数は「その瞬間の在場者数」という決定論量からの純関数。乱数を 1 本も引かない。
def _adaptive_of(sim, zone=None):
    """`physics.adaptive_dt` → (thresholds, recheck_every) / 既定 OFF は None。

    `engines` を書いたときは**そのエンジンのゾーンだけ**が対象(他は None = dt 固定 =
    そのゾーンは 1 バイト同一)。ベンチ実測で SFM は係数 2 で重なり・壁貫通を出すが
    ORCA は出さない(速度層 + 位置層の二層構成が粗い刻みに強い)ため、engine 別に
    切れる口が要る。
    """
    a = (getattr(sim, "physcfg", None) or {}).get("adaptive_dt") or {}
    if not a.get("enabled"):
        return None
    th = tuple(a.get("thresholds") or ())
    if not th:
        return None                       # 閾値ゼロ件 = 係数は常に 1 = OFF と同じ
    eng = tuple(a.get("engines") or ())
    if eng and zone is not None and zone.engine not in eng:
        return None
    return (th, max(1, int(a.get("recheck_every", 1))))


def next_dt(t: float, total_s: float, dt: float, factor: float):
    """次のサブステップ幅と、その後の時刻 `(dt_eff, t_next)`。**規則はここ 1 本**。

    - 端数は最終塊で吸収する(会計の基準 t は total_s へ厳密に着地する。列を補償総和で
      足し直すと 1e-14 s の丸め差が残るが、それは失った時間ではなく足し方の違い)。
    - 残りが `dt·_DT_DUST` 以下になる場合も今回で畳む。理由は**実測された事故**である:
      0.2 を 100 回足すと 20.0 に 3.6e-15 だけ届かず、そこに **dt≈0 のサブステップが
      1 回生える**。ORCA はその 1 回で「時間 dt の遮断円へ射影する」枝の半径が 1/dt で
      発散し、速度が v_max を大きく超えた(ベンチ実測: 平均速さ 0.57 → 53 m/s・
      accel の分位が最終ビンへ張り付き)。畳んで伸びるのは高々 dt·1e-6 秒。
    """
    dt_eff = dt * factor
    rest = total_s - t
    if dt_eff >= rest or (rest - dt_eff) <= dt * _DT_DUST:
        return rest, total_s
    return dt_eff, t + dt_eff


def _dt_factor(thresholds, n_members: int) -> float:
    """在場者数 → dt 係数(thresholds は N 昇順に正準化済み = 最後に該当した係数)。"""
    f = 1.0
    for lim, c in thresholds:
        if n_members > lim:
            f = c
        else:
            break
    return f


def _used_s(rec, dt: float, cum) -> float:
    """この step でその個体が実際に積分された秒数。

    `cum is None`(適応 dt OFF)では **従来と 1 ビット同じ** `step_n × dt`。
    ON では `cum`(= `_accumulate` を呼ぶたびに積んだ累積積分秒)の差分を採る:
    在場者は入場から退場まで**連続した回**に参加するので、その個体の参加回 m は
    必ず末尾の連続塊 = 使う区間は `cum[-1] − cum[-1−m]` で厳密に定まる。
    """
    m = int(rec["step_n"])
    if cum is None:
        return m * dt
    if m <= 0:
        return 0.0
    j = len(cum) - 1
    return cum[j] - (cum[j - m] if j - m >= 0 else 0.0)


# --------------------------------------------------------------------------- #
# ゲート(guarded)
# --------------------------------------------------------------------------- #
def _needs_rng(zone) -> bool:
    """このゾーンが乱数を引くか(ORCA の pref_noise / SFM の ξ が有効なときだけ)。"""
    if zone.engine == "orca":
        return float(zone.orca["pref_noise"]) > 0.0
    return float(zone.sfm["noise"]) > 0.0


def _gate_nodes(sim, zone):
    """ゾーンのゲートノード(遅延構築・sim にキャッシュ。地図だけの純関数なので resume 不要)。"""
    cache = getattr(sim, "_phys_gates", None)
    if cache is None:
        cache = {}
        sim._phys_gates = cache
    if zone.id not in cache:
        cache[zone.id] = _zones.gates_of(zone, sim.city.graph)
    return cache[zone.id]


def _admit(sim, zone, waiting, members, signal, sim_sec, t_in_step,
           step, sim_min, st) -> bool:
    """待機列から入場させる(guarded: 入口が空いているときだけ・id 昇順)。

    信号があるゾーンでは「青(+青点滅)の間」しか入場を許さない。
    Returns: 1 体でも入場したか(= エンジンの再構築が要るか)。
    """
    if signal is not None and not signal.can_cross(sim_sec):
        return False
    gap = float(zone.gate["min_gap_m"])
    any_admitted = False
    queue = list(waiting)
    # ★占有判定の候補列挙をセル法化(判定式・比較演算は 1 バイトも変えない)。
    #   判定は「近傍に 1 体でも居るか」= bool の or なので順序非依存 = 完全同値。
    #   相手集合は 2 つに割れる:
    #     (a) **呼び出し時点の在場者** … `_admit_blocked` の一括判定(第135 のセル法)。
    #         出せなかったとき(在場者が少ない/退化)だけ `base` を素で舐める。
    #     (b) **この呼び出しで入った個体**(`added`)… `_AdmitCells` の増分セル法。
    #   `base` は呼び出し時点のスナップショットなので (a) と (b) は交わらない。
    #   第153 まではここが `members`(= 入場のたびに伸びる生のリスト)で、(b) と
    #   完全に重複していた上に **待ち行列ぶんの総当たり O(W·M)** が残っていた
    #   (在場者ゼロ + 待ち 3,000 で 1.05 s。信号の赤明けごとに再発する)。
    blocked = _admit_blocked(queue, members, gap)
    base = list(members) if blocked is None else ()
    #   格子は **素の総当たりが元を取れなくなってから**組む(損益分岐は「素で舐めた
    #   ペア数 > 待ち行列長」= 格子の構築費 O(W) と釣り合う点。加えて入場済みが
    #   `_ADMIT_CELL_MIN` 体以下なら組まない)。こうすると「待ちは厚いが上流の
    #   `blocked` でほぼ全員弾かれる」呼び出し = 赤の間のサブステップでは
    #   格子を 1 度も組まない(`_admit` は毎サブステップ呼ばれるので固定費が効く)。
    #   どちらの経路を通っても**見る相手の集合は同じ**なので答えは変わらない。
    cells = None
    cells_off = False
    scan = 0                               # 素で舐めたペア数(上界)
    cell_work = len(queue) * _ADMIT_CELL_WORK
    added: list = []
    for qi, rec in enumerate(queue):
        px, py = rec["pos"]
        radius = rec["radius"]
        free = not (blocked is not None and blocked[qi])
        if free and base:
            for other in base:
                ox, oy = other["pos"]
                need = radius + other["radius"] + gap
                if (px - ox) ** 2 + (py - oy) ** 2 < need * need:
                    free = False
                    break
        if free and added:
            n_add = len(added)
            if (cells is None and not cells_off
                    and n_add > _ADMIT_CELL_MIN and scan >= cell_work):
                cells = _AdmitCells.build(queue, gap, added)
                cells_off = cells is None      # 退化入力 = 以後も素の総当たり
            if cells is not None:
                free = not cells.hit(px, py, radius)
            else:
                scan += n_add
                for other in added:
                    ox, oy = other["pos"]
                    need = radius + other["radius"] + gap
                    if (px - ox) ** 2 + (py - oy) ** 2 < need * need:
                        free = False
                        break
        if not free:
            continue                       # 置けなければ移管しない(待たせる)
        added.append(rec)
        if cells is not None:
            cells.add(px, py, radius)
        rec["waiting"] = False
        rec["seen_inside"] = zone.contains(px, py)
        # ★入場ぶんの `waiting` からの除去は**ループを抜けてから 1 回**で行う(下の
        #   `_drop_recs`)。ここで `waiting.remove(rec)` を撃つと、目当ての要素より手前に
        #   残っている「置けなかった」レコード 1 件ごとに dict の等値比較 → その値に
        #   居る `Agent`(dataclass)の `__eq__` が走っていた(第153 の主犯。詳細は
        #   `_drop_recs` の docstring)。ループ内で `waiting` を読む口は 1 つも無い
        #   (`queue` / `members` / `added` しか見ない)ので、除去を遅らせても
        #   判定材料は 1 ビットも変わらない。
        members.append(rec)
        any_admitted = True
        st["enter_total"] += 1
        agent = rec["agent"]
        sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                             kind="zone_gate", x=agent.x, y=agent.y,
                             payload={"zone": zone.id, "gate": rec["gate"],
                                      "dir": "enter", "engine": zone.engine,
                                      "v0": round(rec["v0"], 3),
                                      "speed": round(float(np.hypot(*rec["vel"])), 3),
                                      "span_m": round(float(rec["span_m"]), 1),
                                      # 入場までの待ち [s]。信号のあるゾーンでは
                                      # **赤で縁石に溜まった時間**がここに出る(step 内の
                                      # サブ時刻 + 跨いだ step 数)。dwell_s には入らない。
                                      "wait_s": round(float(t_in_step)
                                                      + float(rec["wait_steps"])
                                                      * float(sim.clock.step_seconds), 2),
                                      "waited_steps": int(rec["wait_steps"])}))
    if any_admitted:
        # 在場者は **agent.id 昇順**(従来は 1 体入るたびに sort していた)。id は個体ごとに
        # 一意なので「1 体ごとに整列」も「最後に 1 回整列」も**同じ全順序**へ落ちる =
        # 返る並びは完全に一致する。ループ内で `members` を見るのは占有判定の
        # 「近傍に 1 体でも居るか」= bool の or だけで、並び順に依らない。
        members.sort(key=lambda r: int(r["agent"].id))
        _drop_recs(waiting, added)
    return any_admitted


def _drop_recs(lst: list, drop) -> None:
    """`for rec in drop: lst.remove(rec)` と**同値**な一括除去(比較を同一性へ落とす)。

    ★第153 で潰した主犯。流入レコードは **dict** で、その値に `agent`(= `Agent` は
      `@dataclass`)が入っている。`list.remove(rec)` は先頭から `==` を掛けるので、
      目当ての要素より手前にある 1 件ごとに `dict.__eq__` → `Agent.__eq__`
      (dataclass 生成 = 全フィールドのタプルを 2 本作ってから比べる)が走っていた。
      250k の py-spy では `_admit` の `waiting.remove` 行が step 時間の 11.3%、その
      子フレーム `__eq__ (<string>:4)` が 10.9% として現れる(合わせて ~22%)。
      実測(待ち 2,000 体・半数入場): remove だけで 2.19 秒 / `Agent.__eq__` 499,500 回。

    同値である理由: 流入レコードは agent 1 体につき 1 個で、`agent` の `id` は相異なる。
    `dict.__eq__` は値まで見る(= `agent` 同士を比べる)ので、**相異なるレコードが
    `==` になることはない**。したがって `list.remove` が見つける「最初の等値要素」は
    必ず同一オブジェクトそのもの = 同一性で選んでも、取り除かれる要素も残る順序も
    完全に一致する。念のため前提が崩れた場合(同一性で全部が見つからない・`drop` に
    同じオブジェクトが 2 度入っている)は**従来の `remove` へそのまま後退**する。
    """
    if not drop:
        return
    marks = {id(r) for r in drop}
    if len(marks) == len(drop):
        kept = [r for r in lst if id(r) not in marks]
        if len(lst) - len(kept) == len(marks):
            lst[:] = kept
            return
    for rec in drop:                       # 保険(従来と 1 バイト同じ経路)
        lst.remove(rec)


_ADMIT_HASH_MIN = 48      # 在場者がこれ以下なら素の総当たりの方が安い(結果は同一)
_ADMIT_CELL_MIN = 24      # 入場済みがこれ以下なら素の総当たりの方が安い(結果は同一)
_ADMIT_CELL_WORK = 1      # 格子を組む損益分岐(素で舐めたペア数 ≥ 待ち行列長 × これ)
_ADMIT_CELL_SCALE = 1.5   # セル辺 = reach × これ(1 未満の余裕が floor 差 ≤ 1 を保証)


class _AdmitCells:
    """`_admit` の**入場済み**(この呼び出しで入れた個体)だけを載せる増分セル法。

    `_admit_blocked` は「呼び出し時点の在場者」を numpy で一括判定するが、入場は
    ループの中で 1 体ずつ増えるので、その増分は逐次リストで舐めるしかなかった。
    在場者ゼロ(= 信号の赤明け・夕方の空ゾーン)では **全部が増分**になり、
    待ち W 体に対して O(W²) の総当たりが残る(W=3,000 で 1.05 s)。ここを潰す。

    同値であること
    --------------
    判定は「近傍に 1 体でも居るか」= bool の **or** なので、相手を舐める順序にも
    「どこまで舐めたか(break の位置)」にも依らない。必要十分は
    **「答えが True になる相手を 1 体も取りこぼさない」**ことだけである:

      - ペアの式は素朴経路と 1 バイト同じ(`need = r_i + r_j + gap` の加算順序も、
        `(px−ox)² + (py−oy)² < need·need` の演算順序もそのまま)。
      - `reach = r_max + r_max + gap`(`r_max` = 待ち行列の最大半径)は、この格子に
        載る**どのペアの `need` の上界でもある**。IEEE 754 の加算は単調
        (a≤c ∧ b≤d ⟹ fl(a+b) ≤ fl(c+d))なので `need ≤ reach` は厳密に成り立つ。
      - `need ≥ 0` なので、判定が True なら |px−ox| ≤ dist < need ≤ reach。
      - セル辺 `cell = 1.5·reach` なので |px/cell − ox/cell| ≤ 1/1.5 = 0.667(丸め
        誤差は 10⁻¹⁶ 桁で、この 0.333 の余裕を食い潰せない)⟹ `floor` の差は
        **高々 1** = 3×3 のリング探索が距離 reach 以内の全点の上位集合。

    前提が崩れうる入力(半径や座標が非有限・負の半径・負の間隙)では `build` が
    `None` を返し、呼び出し側は**従来どおりの素の総当たり**へ落ちる。
    """

    __slots__ = ("cell", "gap", "bins")

    def __init__(self, cell, gap):
        self.cell = cell
        self.gap = gap
        self.bins: dict = {}

    @staticmethod
    def build(queue, gap, seed=()):
        """待ち行列から格子を作り `seed`(入場済み)を載せる(`None` = 素の総当たりへ後退)。

        `r_max` は **待ち行列全体**の最大半径。格子に載るのは `seed` 以降の入場者だが、
        問い合わせ側は待ち行列のどの個体にもなりうるので、上界は queue で取る。
        """
        if not queue or not (gap >= 0.0):
            return None
        r_max = 0.0
        a_max = 0.0
        for rec in queue:
            r = rec["radius"]
            px, py = rec["pos"]
            if not (r >= 0.0) or not (math.isfinite(px) and math.isfinite(py)):
                return None
            if r > r_max:
                r_max = r
            a = abs(px)
            if a > a_max:
                a_max = a
            a = abs(py)
            if a > a_max:
                a_max = a
        cell = (r_max + r_max + gap) * _ADMIT_CELL_SCALE
        # `a_max / cell` が有限 = `math.floor(px / cell)` が必ず int を返す
        # (退化した極小セル × 天文学的な座標でも例外を投げない)。
        if not (cell > 0.0) or not math.isfinite(cell) \
                or not math.isfinite(a_max / cell):
            return None
        out = _AdmitCells(cell, gap)
        for rec in seed:
            px, py = rec["pos"]
            out.add(px, py, rec["radius"])
        return out

    def add(self, px, py, radius) -> None:
        cell = self.cell
        key = (math.floor(px / cell), math.floor(py / cell))
        bucket = self.bins.get(key)
        if bucket is None:
            self.bins[key] = [(px, py, radius)]
        else:
            bucket.append((px, py, radius))

    def hit(self, px, py, radius) -> bool:
        """`for other in added: ...` の総当たりと**同じ bool** を 3×3 セルで返す。"""
        bins = self.bins
        if not bins:
            return False
        cell = self.cell
        gap = self.gap
        cx = math.floor(px / cell)
        cy = math.floor(py / cell)
        get = bins.get
        for ix in (cx - 1, cx, cx + 1):
            for iy in (cy - 1, cy, cy + 1):
                bucket = get((ix, iy))
                if bucket is None:
                    continue
                for ox, oy, orad in bucket:
                    need = radius + orad + gap
                    if (px - ox) ** 2 + (py - oy) ** 2 < need * need:
                        return True
        return False


def _admit_blocked(queue, members, gap):
    """入場待ち各体について「在場者と重なるか」を一括判定(None = セル法を使わない)。

    `_admit` の内側の総当たり(O(W·M)/サブステップ)を潰すためだけの前計算。
    判定式 `(px−ox)² + (py−oy)² < need²` も `need = r_i + r_j + gap` の加算順序も
    素朴経路と同一なので、返す bool は 1 つも変わらない。
    """
    if not queue or len(members) <= _ADMIT_HASH_MIN:
        return None
    mpos = np.array([m["pos"] for m in members], dtype=np.float64)
    mrad = np.array([m["radius"] for m in members], dtype=np.float64)
    wpos = np.array([r["pos"] for r in queue], dtype=np.float64)
    wrad = np.array([r["radius"] for r in queue], dtype=np.float64)
    reach = float(wrad.max()) + float(mrad.max()) + gap
    if not (reach > 0.0):
        return None
    field = _sfm.PointField(mpos, reach / 2.0)
    ring = int(math.ceil(reach / field.cell))
    qrow, qpt = field.candidates(wpos, ring)
    out = np.zeros(len(queue), dtype=bool)
    if qrow.shape[0] == 0:
        return out
    dx = wpos[qrow, 0] - mpos[qpt, 0]
    dy = wpos[qrow, 1] - mpos[qpt, 1]
    need = wrad[qrow] + mrad[qpt] + gap
    out[qrow[(dx * dx + dy * dy) < need * need]] = True
    return out


def _admit_record(sim, zone, agent, path, rest, step) -> dict:
    """流入レコードを作る(位置 = 現在座標そのもの・速度 = グラフ速度の連続引き継ぎ)。"""
    v0 = desired_speed(agent.id)
    exit_node = path[-1]
    ex, ey = sim.city.node_xy(exit_node)
    nx, ny = sim.city.node_xy(path[1]) if len(path) > 1 else (ex, ey)
    dx, dy = nx - agent.x, ny - agent.y
    norm = math.hypot(dx, dy)
    if norm < 1e-9:
        dx, dy, norm = ex - agent.x, ey - agent.y, max(math.hypot(ex - agent.x,
                                                                  ey - agent.y), 1e-9)
    speed = _graph_speed(sim, agent)
    speed = min(speed, v0 * zone.v_max_factor)
    span_m = 0.0
    for u, v in zip(path, path[1:]):
        span_m += float(sim.city.edge_length(u, v))
    return {
        "agent": agent,
        "zone": zone.id,
        "path": list(path),
        "rest": list(rest),
        "gate": _zones.gate_id(zone, agent.node),
        "exit_xy": (ex, ey),
        "span_m": span_m,          # ゾーン内区間のグラフ経路長 [m](dwell との比で実効速度)
        "wp": 1 if len(path) > 1 else 0,        # 目標にしている経路ノードの index
        "wp_xy": (nx, ny),                      # その座標(= engine.goal の初期値)
        "pos": (float(agent.x), float(agent.y)),
        "vel": (speed * dx / norm, speed * dy / norm),
        "dir0": (dx / norm, dy / norm),
        "seg_dir": (dx / norm, dy / norm),      # 反転率の基準(区間の向き。通過点ごとに更新)
        "v0": v0,
        "radius": body_radius(agent.id),
        "waiting": True,
        "seen_inside": False,
        "wait_steps": 0,
        "elapsed_s": 0.0,      # 累積滞在秒(step を跨ぐ)
        "step_n": 0,           # **この step で**積分したサブステップ数(2 層タイムラインの会計)
        "speed_sum": 0.0, "speed_n": 0, "contact_n": 0, "dens_sum": 0.0,
        "sign": 0,
    }


def _graph_speed(sim, agent) -> float:
    """グラフ側の移動速度 [m/s](= mode 速度 × 混雑係数 ÷ step 秒)。P3(2) の連続引き継ぎ。"""
    speeds = sim.cfg.world.modes.speeds
    per_step = float(speeds[agent.trip_mode])
    factor = float(getattr(agent, "_congestion", 1.0) or 1.0)
    return max(0.0, per_step * factor / float(sim.clock.step_seconds))


# --------------------------------------------------------------------------- #
# P4-2 / P4-3 歩行物理較正(**3 機能とも既定 OFF/現行値 = 完全恒等**)
# --------------------------------------------------------------------------- #
# 正典: docs/research/p4-calibration-research.md §4(Tordeux 型 V(s))/ §6.1(2 項構造)
#       reference/physics_bench/out/calib_results.json(P4-1)/ calib_p43_results.json(P4-3)
#
# なぜ `sfm_core.py` ではなくここに置くのか
# ----------------------------------------
# 較正 3 機能は **ゾーン物理(physics.zones_enabled)の中でしか使わない**。
# `sfm_core.Crowd` は屋内 SFM(indoor_flow)からも使われている共有コアなので、そこへ
# 分岐を足すと「屋内には効かせないつもりの較正」が屋内の既定経路にも 1 分岐ぶん載る。
# ベンチ(reference/physics_bench/engines.ExtendedCrowd)が採ったのと同じ
# 「コアを継承して forces() に足すだけ」の構造をそのまま昇格させ、コアは無改変に保つ。
#
# OFF が保つ不変条件(数値的恒等)
# --------------------------------
# `_calib_kwargs()` が空 dict を返す限り `_build_engine` は **従来の `sfm_core.Crowd` を
# 従来の引数のまま**構築する(派生クラスすら作らない)= 演算順も引数も 1 バイト変わらない。
# 既定 conf では far_field.enabled=false / v_of_s.enabled=false / wall.{a,b} は
# sfm_core の既定値と同値なので、空 dict になる。
_EXP_ARG_MAX = 4.0        # sfm_core と同じ安全弁(深い重なりの overflow 防止)

# ---- 第二段C(遠方場の密度場化)の既定値 ----
FIELD_CELL_M_DEFAULT = 1.0        # 密度格子の一辺 [m](far カットオフ 4.7 m の 1/5 弱)
FIELD_BLUR_DEFAULT = 2            # 3×3 箱平滑の回数(2 回 ≈ σ≈1.15 セルの Gauss 近似)
FIELD_UPDATE_EVERY_DEFAULT = 10   # 格子を作り直すサブステップ周期(10×0.05s = 0.5 s。
#                                   対人固視の実測時定数 ~0.5 s = Fotios et al. 2015 と同桁)
_FIELD_QUAD_N = 4096              # 接続係数 I₁/I₂ の数値積分の分点数(決定論・構築時 1 回)


def _box3(a, pad=None, out=None):
    """3×3 箱平滑(ゼロ詰め境界)。加算順序を式で固定 = 決定論。

    ゼロ詰めにするのは物理的に正しいから: 群衆の外側には誰も居ない。境界で密度が
    落ちるので群衆の縁には外向きの勾配が立つ = ペア版で「内側にしか相手が居ない」ときに
    外へ押される挙動と同じ向き。

    `pad` / `out` は呼び出し側が持ち回る作業領域(省略すればその場で確保 = 従来と同じ)。
    足す順序・丸め・除数は下の式のまま **一字も動かしていない**: `x + y + z` は左結合なので、
    中間結果を毎回新しい配列に置くか同じ配列へ上書きするかは 1 ビットも結果を変えない。
    置き場所だけを使い回す理由(実測): 25k finals conf の格子は平均 117 万セル =
    float64 1 枚 9.4 MB あり、1 回の平滑ごとにパッド 1 枚 + 中間 8 枚を新規確保していた。
    支配的なのは演算ではなく確保(初回書き込みのページフォルト)なので、ここを畳むだけで
    実測 3〜4 倍になる。`pad` は内側 [1:-1,1:-1] しか書かないので縁の 0 が保たれる
    = 何度でも使い回せる。

        (p[:-2,:-2] + p[:-2,1:-1] + p[:-2,2:]
         + p[1:-1,:-2] + p[1:-1,1:-1] + p[1:-1,2:]
         + p[2:,:-2] + p[2:,1:-1] + p[2:,2:]) / 9.0
    """
    if pad is None:
        pad = np.zeros((a.shape[0] + 2, a.shape[1] + 2), dtype=np.float64)
    pad[1:-1, 1:-1] = a
    s = np.add(pad[:-2, :-2], pad[:-2, 1:-1], out=out)
    np.add(s, pad[:-2, 2:], out=s)
    np.add(s, pad[1:-1, :-2], out=s)
    np.add(s, pad[1:-1, 1:-1], out=s)
    np.add(s, pad[1:-1, 2:], out=s)
    np.add(s, pad[2:, :-2], out=s)
    np.add(s, pad[2:, 1:-1], out=s)
    np.add(s, pad[2:, 2:], out=s)
    return np.divide(s, 9.0, out=s)


class _CalibratedCrowd(_sfm.Crowd):
    """`sfm_core.Crowd` + 長距離 social 項 + Tordeux 型 V(s)(較正 ON のときだけ作る)。

    (1) 長距離項 f2_ij = m · a2 · exp((r_i+r_j − d)/b2) · w(φ) · n_ij
        カットオフは**体表間隔** `cutoff_factor·b2` [m]、その手前 `taper_m` [m] を
        C¹ smoothstep で 0 へ落とす(Köster 2013 の言う「右辺の不連続」を作らない)。
        異方性 w(φ) は短距離項と同じ λ(= `lambda_aniso`。P4-1 の λ 先行実験が
        文献値 0.06–0.12 の仮説を棄却したので本体既定 0.5 を共有する)。
        近傍 cap は **掛けない**: ρ=3 /m² では最近傍 12 体が半径 ~1.1 m 以内に入るため、
        cap を掛けると「長距離項」が実質短距離項になり b2 が効かなくなる(P4-1 実測)。
    (2) V(s) = min{v0, max{0, (s − l)/T}} を **駆動項の希望速度だけ** に適用。
        v_max は __init__ が v0 から作った配列のままなので**速度上限は不変**。

    ★ベンチ側 `ExtendedCrowd.forces_naive()` と同じ加算経路(super().forces() に
      長距離項を足す)。ベンチはこの素朴経路が融合経路と **厳密に 0.0 差**であることを
      実測で固定しているので、較正で得た数値はそのままここで再現される。
    """

    def __init__(self, *args, far_a2=0.0, far_b2=1.890, far_cutoff_factor=2.5,
                 far_taper_m=1.0, v_of_s=False, vos_T=0.482, vos_l=0.297,
                 far_mode="pair", field_cell_m=FIELD_CELL_M_DEFAULT,
                 field_blur=FIELD_BLUR_DEFAULT,
                 field_update_every=FIELD_UPDATE_EVERY_DEFAULT,
                 field_clip=None, **kw):
        super().__init__(*args, **kw)
        self.far_a2 = float(far_a2)
        self.far_b2 = float(far_b2)
        self.far_cutoff = float(far_cutoff_factor) * self.far_b2
        self.far_taper = float(max(0.0, min(far_taper_m, self.far_cutoff)))
        self.v_of_s = bool(v_of_s)
        self.vos_T = float(vos_T)
        self.vos_l = float(vos_l)
        if self.v_of_s and not (self.vos_T > 0.0):
            raise ValueError("v_of_s は T > 0 が必要")
        # ── (3) 遠方場の密度場化(第二段C。far_mode="pair" が既定 = 従来のペア和)──
        if far_mode not in ("pair", "field"):
            raise ValueError("far_mode は 'pair' か 'field'")
        self.far_mode = far_mode
        self.field_cell_m = float(field_cell_m)
        self.field_blur = int(field_blur)
        self.field_update_every = max(1, int(field_update_every))
        if self.far_mode == "field" and not (self.field_cell_m > 0.0):
            raise ValueError("density_far.cell_m は > 0 が必要")
        # P4(2026-08-21): 格子外延のクリップ矩形 (x0, y0, x1, y1) [m]。
        # **既定 None = クリップなし = 現行挙動と 1 バイト同一**(新分岐を 1 度も通らない)。
        self.field_clip = None
        if field_clip is not None:
            x0, y0, x1, y1 = (float(v) for v in field_clip)
            if not (x0 <= x1 and y0 <= y1):
                raise ValueError("field_clip は (x0<=x1, y0<=y1) が必要")
            self.field_clip = (x0, y0, x1, y1)
        self._field_tick = 0
        self._field_grid = None           # (rho, gx, gy, ox, oy) 粗い周期で作り直す
        self._field_scratch = None        # 箱平滑の作業領域 (pad, acc)。格子の形が同じ間だけ持つ
        # 接続係数(class docstring の I₁ / I₂)。半径は不変なので **構築時に 1 度だけ**。
        self._c_drag, self._c_grad = (self._far_field_coeffs()
                                      if (self.far_mode == "field"
                                          and self.far_a2 > 0.0) else (0.0, 0.0))

    # -- (1) 長距離 social 項 --------------------------------------------- #
    def _far_forces(self):
        n = self.pos.shape[0]
        if n < 2 or self.far_a2 <= 0.0:
            return np.zeros_like(self.pos)
        if not self.pair_hash:
            return self._far_forces_dense()
        # 空間ハッシュ経路(**_far_forces_dense とビット一致**)。
        # 候補 reach = 2·r_max + far_cutoff。valid の条件 gap = d − rr ≤ far_cutoff は
        # d ≤ rr + far_cutoff ≤ reach と同値以下なので、候補は必ず上位集合になる。
        # ★ cap は掛けない(class docstring の P4-1 実測。ここでも掛けていない)。
        e, _ = self._desired_dir()
        reach = 2.0 * float(self.radius.max()) + self.far_cutoff
        ii, jj = _sfm.neighbor_pairs(self.pos, reach)
        out = np.zeros((n, 2), dtype=np.float64)
        for r0, r1, p0, p1 in _sfm.pair_blocks(ii, n):
            if p1 <= p0:
                continue
            si, sj = ii[p0:p1], jj[p0:p1]
            diff = self.pos[si] - self.pos[sj]                    # j → i
            d = np.linalg.norm(diff, axis=1)
            rr = self.radius[si] + self.radius[sj]
            with np.errstate(invalid="ignore", divide="ignore"):
                nij = diff / d[:, None]
            nij = np.nan_to_num(nij)
            cosphi = -(e[si, 0] * nij[:, 0] + e[si, 1] * nij[:, 1])
            gap = d - rr                                          # 体表間隔 [m]
            arg = np.clip(-gap / self.far_b2, a_min=None, a_max=_EXP_ARG_MAX)
            mag = self.mass * self.far_a2 * np.exp(arg)           # [N]
            w = self.lam + (1.0 - self.lam) * (1.0 + cosphi) / 2.0
            valid = self.active[sj] & (gap <= self.far_cutoff)
            if self.far_taper > 0.0:
                u = np.clip((self.far_cutoff - gap) / self.far_taper, 0.0, 1.0)
                mag = mag * (u * u * (3.0 - 2.0 * u))             # C¹ smoothstep
            contrib = (w * mag)[:, None] * nij
            contrib[~valid] = 0.0
            out[r0:r1, 0] = np.bincount(si - r0, weights=contrib[:, 0],
                                        minlength=r1 - r0)
            out[r0:r1, 1] = np.bincount(si - r0, weights=contrib[:, 1],
                                        minlength=r1 - r0)
        return out

    def _far_forces_dense(self):
        """全ペア (N,N) の長距離項(**参照実装**。空間ハッシュ版とビット一致)。

        ★ベンチ `reference/physics_bench/engines.ExtendedCrowd._far_contrib` と同一式・
          同一合算順序。tests/test_physics_calib.py の バイト一致検査がこの同値を機械固定する。
        """
        e, _ = self._desired_dir()
        diff = self.pos[:, None, :] - self.pos[None, :, :]        # j → i
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        rr = self.radius[:, None] + self.radius[None, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            nij = diff / d[:, :, None]
        nij = np.nan_to_num(nij)
        cosphi = -np.einsum("ik,ijk->ij", e, nij)
        gap = d - rr                                              # 体表間隔 [m]
        arg = np.clip(-gap / self.far_b2, a_min=None, a_max=_EXP_ARG_MAX)
        mag = self.mass * self.far_a2 * np.exp(arg)               # [N]
        w = self.lam + (1.0 - self.lam) * (1.0 + cosphi) / 2.0
        valid = self.active[None, :] & (gap <= self.far_cutoff)
        if self.far_taper > 0.0:
            u = np.clip((self.far_cutoff - gap) / self.far_taper, 0.0, 1.0)
            mag = mag * (u * u * (3.0 - 2.0 * u))                 # C¹ smoothstep
        contrib = (w * mag)[:, :, None] * nij
        contrib[~valid] = 0.0
        return contrib.sum(axis=1)

    # -- (3) 遠方場の密度場化(第二段C)------------------------------------ #
    def _far_field_coeffs(self):
        """接続係数 (c_drag, c_grad) を **ペア版の同じカーネルから** 解析+数値で導く。

        c_drag = (1−λ)·(π/2)·∫₀^R K(r)·r  dr
        c_grad = (1+λ)·(π/2)·∫₀^R K(r)·r² dr
        K(r) = m·a2·exp(clip((rr−r)/b2))·smoothstep_taper(r) は
        `_far_forces` が 1 ペアに与える力の大きさそのもの(同じ clip・同じ taper)。
        rr は体表間隔の基準で、ここでは全体の平均体径 2·r̄ を使う(半径のばらつきは
        0.25–0.35 m と狭く、係数への効きは 1 次で数 % = 正直な近似)。
        積分は中点則の固定分点(乱数ゼロ・同一入力 → 同一 float)。
        """
        rr = 2.0 * float(self.radius.mean())
        big_r = rr + self.far_cutoff
        q = _FIELD_QUAD_N
        dr = big_r / q
        r = (np.arange(q, dtype=np.float64) + 0.5) * dr
        gap = r - rr
        arg = np.clip(-gap / self.far_b2, a_min=None, a_max=_EXP_ARG_MAX)
        kern = self.mass * self.far_a2 * np.exp(arg)
        if self.far_taper > 0.0:
            u = np.clip((self.far_cutoff - gap) / self.far_taper, 0.0, 1.0)
            kern = kern * (u * u * (3.0 - 2.0 * u))
        kern = np.where(gap <= self.far_cutoff, kern, 0.0)
        i1 = float((kern * r).sum() * dr)
        i2 = float((kern * r * r).sum() * dr)
        half_pi = 0.5 * math.pi
        return ((1.0 - self.lam) * half_pi * i1,
                (1.0 + self.lam) * half_pi * i2)

    def _clip_mask(self):
        """クリップ矩形の中に居るメンバー(P4。`field_clip` が None のときは呼ばない)。

        境界は**閉区間**(縁ちょうどは内側)= `Zone.contains` の「境界上は内側」と同じ流儀。
        """
        x0, y0, x1, y1 = self.field_clip
        p = self.pos
        return ((p[:, 0] >= x0) & (p[:, 0] <= x1)
                & (p[:, 1] >= y0) & (p[:, 1] <= y1))

    def _density_grid(self):
        """密度場 ρ と勾配 (∂ρ/∂x, ∂ρ/∂y) の格子。`update_every` サブステップに 1 度作る。

        構築は O(N + G)(G = 格子セル数): セル index の `np.bincount`(整数カウント =
        加算順序に依らず厳密)→ 3×3 箱平滑 `field_blur` 回 → 中心差分。

        平滑の作業領域は格子の形が変わらない限り使い回す(`_box3` の docstring の実測)。
        ★**返す ρ は毎回新品の配列**である(最後の 1 回だけ `out=None` で確保する)。
          場は `_field_grid` として次の再構築まで — carry_grid ON なら次のエンジンへも —
          持ち回られるので、作業領域そのものを返すと後の再構築が過去の場を書き潰す。

        P4(2026-08-21)格子外延の有界化 —— `field_clip` があるときだけ
        ------------------------------------------------------------
        外延は既定では**在場者の外接矩形**で決まる。ところがゾーンの所有は「経路がゾーンを
        通り抜ける個体」に対して**現在地から**始まる(数百 m 先から所有する)ので、
        38×29 m の広場に対して ~1 km 角(平均 117 万セル・float64 1 枚 9.4 MB)の場を
        毎回作っていた(第145 の実測。physics.phase の 67%)。
        `field_clip`(= ゾーン polygon の外接矩形 ± `clip_margin_m`)を渡すと、外延を
        その矩形との**共通部分**へ落とす。得られる場は「矩形の外のメンバーを名簿から
        外して作った場」と**格子も値もビット一致**する(= クリップは名簿を削るだけで、
        残った側の計算には一切手を入れない)。全員が矩形の中なら外延も値も従来と同一。
        ★圏外の個体は **np.clip で縁のセルへ押し込まない**(押し込むと数百 m 先の一人歩きの
          密度が縁に山積みになり、場が嘘になる)。堆積させない = その場所の密度に数えない、
          が正しい: 圏外の局所密度は読み出し側(`_sample_field`)で 0 として扱う。
        縁の連続性は既存の `pad`(= field_blur + 2)がそのまま担う: 堆積の外側に平滑と
        中心差分のぶんだけゼロ詰めの余白が付くので、外延の縁で階段状の勾配は立たない。
        """
        cell = self.field_cell_m
        pad = self.field_blur + 2                   # 平滑がゼロ詰め境界を跨がない余白
        pos = self.pos
        sel = self.active
        if self.field_clip is None:
            ox = float(pos[:, 0].min()) - pad * cell
            oy = float(pos[:, 1].min()) - pad * cell
            nx = int(math.floor((float(pos[:, 0].max()) - ox) / cell)) + pad + 1
            ny = int(math.floor((float(pos[:, 1].max()) - oy) / cell)) + pad + 1
        else:
            keep = self._clip_mask()
            sel = sel & keep                        # 圏外は堆積させない(押し込まない)
            if keep.any():
                px, py = pos[keep, 0], pos[keep, 1]
                ox = float(px.min()) - pad * cell
                oy = float(py.min()) - pad * cell
                nx = int(math.floor((float(px.max()) - ox) / cell)) + pad + 1
                ny = int(math.floor((float(py.max()) - oy) / cell)) + pad + 1
            else:
                # 圏内が空 = 場は全面 0。最小格子を矩形の隅に置く(読み出しは全員 0 なので
                # この格子の値は 1 つも使われない。形だけ整えて下の共通経路へ渡す)。
                ox = self.field_clip[0] - pad * cell
                oy = self.field_clip[1] - pad * cell
                nx = ny = 3
        nx, ny = max(nx, 3), max(ny, 3)
        ci = np.clip(np.floor((pos[:, 0] - ox) / cell), 0, nx - 1).astype(np.int64)
        cj = np.clip(np.floor((pos[:, 1] - oy) / cell), 0, ny - 1).astype(np.int64)
        cnt = np.bincount((ci * ny + cj)[sel], minlength=nx * ny)
        rho = cnt.reshape(nx, ny).astype(np.float64) / (cell * cell)
        if self.field_blur:
            bpad, bacc = self._blur_scratch(nx, ny)     # 上の `pad`(余白セル数)とは別物
            last = self.field_blur - 1
            for i in range(self.field_blur):
                rho = _box3(rho, bpad, bacc if i < last else None)
        gx = np.zeros_like(rho)
        gy = np.zeros_like(rho)
        gx[1:-1, :] = (rho[2:, :] - rho[:-2, :]) / (2.0 * cell)
        gy[:, 1:-1] = (rho[:, 2:] - rho[:, :-2]) / (2.0 * cell)
        return rho, gx, gy, ox, oy

    def _blur_scratch(self, nx, ny):
        """箱平滑の作業領域 (pad, acc)。格子の形が変わったときだけ取り直す。

        格子の外周は原点 (ox, oy) を「在場者の最小座標 − pad セル」で採るので、
        在場者が動けば形も変わりうる。形が同じ間は使い回し、変わったら取り直す
        (取り直しても値は変わらない = `_box3` は縁の 0 と内側の代入だけに依存する)。
        """
        buf = self._field_scratch
        if buf is None or buf[1].shape != (nx, ny):
            buf = (np.zeros((nx + 2, ny + 2), dtype=np.float64),
                   np.empty((nx, ny), dtype=np.float64))
            self._field_scratch = buf
        return buf

    def _sample_field(self, grid):
        """格子 → 個体位置での (ρ_i, ∇ρ_i)。セル中心格子の双線形補間。

        P4: `field_clip` があるとき、**圏外の個体は ρ=0 / ∇ρ=0** を読む。
        `np.clip` で縁のセルの値を読ませてはいけない(数百 m 先の個体が、ゾーンの縁の
        混雑を自分の足元の混雑として受け取ってしまう)。0 が物理的に正しい:
        圏外に居るのは経路上をひとりで歩いている個体で、その足元の局所密度は実際ほぼ 0 =
        遠方場の混雑回避力は働かない。
        """
        rho, gx, gy, ox, oy = grid
        cell = self.field_cell_m
        nx, ny = rho.shape
        fx = (self.pos[:, 0] - ox) / cell - 0.5     # セル中心を格子点とする座標
        fy = (self.pos[:, 1] - oy) / cell - 0.5
        i0 = np.clip(np.floor(fx), 0, nx - 2).astype(np.int64)
        j0 = np.clip(np.floor(fy), 0, ny - 2).astype(np.int64)
        tx = np.clip(fx - i0, 0.0, 1.0)
        ty = np.clip(fy - j0, 0.0, 1.0)
        i1, j1 = i0 + 1, j0 + 1
        w00 = (1.0 - tx) * (1.0 - ty)
        w10 = tx * (1.0 - ty)
        w01 = (1.0 - tx) * ty
        w11 = tx * ty

        def _bi(a):
            return (a[i0, j0] * w00 + a[i1, j0] * w10
                    + a[i0, j1] * w01 + a[i1, j1] * w11)

        rho_i, grad_i = _bi(rho), np.stack([_bi(gx), _bi(gy)], axis=1)
        if self.field_clip is not None:
            keep = self._clip_mask()
            rho_i = np.where(keep, rho_i, 0.0)
            grad_i[~keep] = 0.0
        return rho_i, grad_i

    def _far_forces_field(self):
        """遠方場 = 密度場の連続体力(**ペア和 `_far_forces` の置換**。第二段C)。

        f = −c_drag·ρ·e_i − c_grad·∇ρ   (係数の導出は class docstring と
        `_far_field_coeffs`)。一様密度ではペア版と同じ合力になる = FD の作業点が保存される。
        計算量は O(N + G) で **密度に対して平坦**(ペア版は O(N·k̄), k̄ = ρπR²)。
        """
        n = self.pos.shape[0]
        if n < 2 or self.far_a2 <= 0.0:
            return np.zeros_like(self.pos)
        if self._field_grid is None or (self._field_tick % self.field_update_every) == 0:
            self._field_grid = self._density_grid()
        self._field_tick += 1
        rho, grad = self._sample_field(self._field_grid)
        e, _ = self._desired_dir()
        out = -(self._c_drag * rho)[:, None] * e - self._c_grad * grad
        out[~self.active] = 0.0
        return out

    # -- (2) 前方間隔 s と V(s) -------------------------------------------- #
    def front_spacing(self):
        """進行方向の最近前方者までの中心間距離 s_i [m](前方に誰も居なければ inf)。

        「前方」= 進行方向成分が正、かつ横ずれが体半径和 (r_i+r_j) 以内
        (= 実際に進路を塞いでいる相手)。横ずれが体幅より大きい相手は避けて通れるので
        数えない。乱数ゼロ・比較と min だけ = 決定論。
        """
        n = self.pos.shape[0]
        if n < 2:
            return np.full(n, np.inf)
        e, _ = self._desired_dir()
        diff = self.pos[None, :, :] - self.pos[:, None, :]        # i → j
        d = np.linalg.norm(diff, axis=2)
        along = np.einsum("ik,ijk->ij", e, diff)
        lateral2 = np.maximum(d * d - along * along, 0.0)
        rr = self.radius[:, None] + self.radius[None, :]
        ahead = self.active[None, :] & (along > 0.0) & (lateral2 <= rr * rr)
        np.fill_diagonal(ahead, False)
        return np.where(ahead, d, np.inf).min(axis=1)

    def v_of_s_speed(self):
        s = self.front_spacing()
        return np.minimum(self.v0, np.maximum(0.0, (s - self.vos_l) / self.vos_T))

    # -- 合成 -------------------------------------------------------------- #
    def forces(self):
        if not self.v_of_s:
            return self._forces_with_far()
        v0_saved = self.v0
        self.v0 = self.v_of_s_speed()
        try:
            return self._forces_with_far()
        finally:
            self.v0 = v0_saved

    def _forces_with_far(self):
        f = super().forces()
        if self.far_a2 > 0.0:
            # far_mode="pair"(既定)は第一段A までと 1 バイト同じ経路
            f = f + (self._far_forces_field() if self.far_mode == "field"
                     else self._far_forces())
            f[~self.active] = 0.0
        return f


def _calib_kwargs(sim, zone=None) -> dict:
    """`physics.sfm` の較正 3 機能(+ 第二段C の密度場)→ `_CalibratedCrowd` の追加引数。

    **既定(全部 OFF/現行値)では空 dict** を返す = 従来の `sfm_core.Crowd` 経路。
    ★`physics.density_far` は **far_field が ON のときだけ**効く(置換する当の項が
      無ければ意味がない)。OFF のまま far_field だけ立てれば第一段A までと同じペア和。
    ★`zone` は P4 の格子クリップ矩形を作るためだけに使う(`clip_margin_m` 既定 0 =
      矩形を作らない = 引数が 1 つも増えない = 現行と 1 バイト同一)。
    """
    cfg = getattr(sim, "physcfg", None) or {}
    s = cfg.get("sfm")
    if not s:
        return {}
    kw: dict = {}
    ff = s["far_field"]
    if ff["enabled"]:
        kw.update(far_a2=ff["a2"], far_b2=ff["b2"],
                  far_cutoff_factor=ff["cutoff_factor"], far_taper_m=ff["taper_m"])
        df = cfg.get("density_far") or {}
        if df.get("enabled"):
            kw.update(far_mode="field", field_cell_m=df["cell_m"],
                      field_blur=df["blur"], field_update_every=df["update_every"])
            clip = _field_clip(zone, df)
            if clip is not None:
                kw["field_clip"] = clip
    vs = s["v_of_s"]
    if vs["enabled"]:
        kw.update(v_of_s=True, vos_T=vs["T"], vos_l=vs["l"])
    wl = s["wall"]
    # 壁は sfm_core の既定値と同値なら **渡さない**(= 従来の呼び出しと 1 バイト同じ)。
    if (wl["a"] != _sfm.WALL_A_DEFAULT) or (wl["b"] != _sfm.WALL_B_DEFAULT):
        kw.update(wall_a=wl["a"], wall_b=wl["b"])
    return kw


def _field_clip(zone, df) -> tuple | None:
    """`physics.density_far.clip_margin_m` → 密度格子のクリップ矩形(P4)。

    矩形 = **ゾーン polygon の外接矩形 ± margin**。margin が担うのは 2 つで、
      (a) 場の数値的な依存半径 = (blur + 2) セル(平滑 blur セル + 中心差分 1 + 双線形 1)。
          これだけ余白があれば、**切り落とした堆積の影響がポリゴンの中まで届かない**。
      (b) 縁の外の個体が矩形へ出入りするときの力の跳びを、ゾーン本体から遠ざける。
    ★既定 0.0 では None を返す = 引数が生えない = 現行挙動と 1 バイト同一。
    """
    margin = float((df or {}).get("clip_margin_m", 0.0) or 0.0)
    if zone is None or margin <= 0.0:
        return None
    x0, y0, x1, y1 = zone.bbox
    return (x0 - margin, y0 - margin, x1 + margin, y1 + margin)


def _carry_grid_on(sim) -> bool:
    """`physics.density_far.carry_grid`(既定 false)。エンジン再構築で場を引き継ぐか。"""
    df = (getattr(sim, "physcfg", None) or {}).get("density_far") or {}
    return bool(df.get("carry_grid"))


def _carry_field(old, new) -> None:
    """密度場 (ρ, ∇ρ) と再構築カウンタを旧エンジンから新エンジンへ引き継ぐ。

    なぜ引き継いでよいのか: **密度場は空間の関数であって在場者の名簿ではない**。
    格子は「その時点でゾーンに居た全員の位置」を数え上げた ρ(x) で、far 項がそこから
    読むのは「自分の周りがどれだけ混んでいるか」だけである。エンジンの作り直しは
    入場・退場の瞬間に起きるが、そこでは 1 サブステップも積分していない(位置の集合は
    出入りした数体を除いて同一)ので、場の意味は保存される。そもそも場は
    `update_every` サブステップ分だけ古いまま読まれる設計なので、
    「メンバーが変わった瞬間に必ず作り直す」ことに物理的な根拠は無い。
    引き継がないと、入退場のたびに `_field_tick` が 0 へ巻き戻り
    (新インスタンスは `_field_grid=None` から始まる)、**混んでいる時間帯ほど
    `update_every` の周期が効かなくなる**という逆立ちした特性になる。

    ★作業領域(`_field_scratch`)は引き継がない: 場は新エンジンが持ち回るので、
      その置き場所まで共有すると次の再構築が持ち回り中の場を書き潰す。
    ★既定 OFF(carry_grid=false)ではこの関数が 1 度も呼ばれない = 現行挙動と 1 バイト同一。
    """
    if isinstance(old, _CalibratedCrowd) and isinstance(new, _CalibratedCrowd):
        new._field_grid = old._field_grid
        new._field_tick = old._field_tick


def _cog_kwargs(sim) -> dict:
    """`physics.cognitive` → SFM/ORCA 共通の認知的近傍の引数(第二段B)。

    **既定 OFF では空 dict** = `_build_engine` の呼び出しは第一段A までと 1 バイト同じ。
    ★`c["neighbors"]` は `physics.neighbor_cap`(第154 レバーD)が >0 なら構築時に
      `min(neighbors, neighbor_cap)` へ絞り込み済み(`zones._build_neighbor_cap`)=
      cap の作用点は**この 1 点**に畳まれている(二重 cap を作らない)。
    """
    c = (getattr(sim, "physcfg", None) or {}).get("cognitive")
    if not c or not c["enabled"]:
        return {}
    return {"cognitive": True, "cog_neighbors": int(c["neighbors"]),
            "cog_sectors": int(c["sectors"]), "cog_fov_deg": float(c["fov_deg"])}


def calib_describe(sim) -> dict:
    """ON の較正だけを要約する(既定 OFF なら空 dict = manifest にキーが生えない)。"""
    cfg = getattr(sim, "physcfg", None) or {}
    s = cfg.get("sfm")
    if not s:
        return {}
    out: dict = {}
    if s["far_field"]["enabled"]:
        out["far_field"] = dict(s["far_field"])
    if s["v_of_s"]["enabled"]:
        out["v_of_s"] = dict(s["v_of_s"])
    if (s["wall"]["a"] != _sfm.WALL_A_DEFAULT
            or s["wall"]["b"] != _sfm.WALL_B_DEFAULT):
        out["wall"] = dict(s["wall"])
    cg = cfg.get("cognitive") or {}
    if cg.get("enabled"):
        out["cognitive"] = dict(cg)
    df = cfg.get("density_far") or {}
    if df.get("enabled"):
        out["density_far"] = dict(df)
    ad = cfg.get("adaptive_dt") or {}
    if ad.get("enabled"):
        out["adaptive_dt"] = {"thresholds": [list(p) for p in ad["thresholds"]],
                              "recheck_every": int(ad["recheck_every"]),
                              "engines": list(ad.get("engines") or ())}
    if int(cfg.get("neighbor_cap") or 0) > 0:
        out["neighbor_cap"] = int(cfg["neighbor_cap"])
    if int(cfg.get("separation_iters") or 0) > 0:
        out["separation_iters"] = int(cfg["separation_iters"])
    return out


# --------------------------------------------------------------------------- #
# エンジン構築 / 積分 / 計測
# --------------------------------------------------------------------------- #
def _build_engine(zone, members, rng, calib=None, cog=None):
    pos = np.array([r["pos"] for r in members], dtype=np.float64)
    vel = np.array([r["vel"] for r in members], dtype=np.float64)
    goal = np.array([r["wp_xy"] for r in members], dtype=np.float64)
    v0 = np.array([r["v0"] for r in members], dtype=np.float64)
    radius = np.array([r["radius"] for r in members], dtype=np.float64)
    if zone.engine == "orca":
        o = zone.orca
        return _orca.OrcaCrowd(
            pos, vel, goal, v0, radius, walls=zone.walls,
            neighbor_cap=zone.neighbor_cap, tau=float(o["tau"]),
            tau_obst=float(o["tau_obst"]), neighbor_dist=float(o["neighbor_dist_m"]),
            wall_range=float(o["wall_range_m"]), v_max_factor=zone.v_max_factor,
            arrive_radius=zone.arrive_radius_m, pref_noise=float(o["pref_noise"]),
            rng=rng, radius_margin=float(o["radius_margin_m"]),
            separation_iters=int(o["separation_iters"]),
            **(cog or {}))
    s = zone.sfm
    # ★較正・認知的近傍が全部既定なら calib も cog も空 dict = 下の呼び出しは
    #   P4 以前(および第一段A)と 1 バイト同一。
    cls = _sfm.Crowd if not calib else _CalibratedCrowd
    return cls(
        pos, vel, goal, v0, radius=radius, rng=rng, noise=float(s["noise"]),
        arrive_radius=zone.arrive_radius_m,
        walls=(zone.walls or None), wall_range=float(s["wall_range_m"]),
        neighbor_cap=zone.neighbor_cap, v_max_factor=zone.v_max_factor,
        **(calib or {}), **(cog or {}))


def _writeback(members, engine) -> None:
    for i, rec in enumerate(members):
        rec["pos"] = (float(engine.pos[i, 0]), float(engine.pos[i, 1]))
        rec["vel"] = (float(engine.vel[i, 0]), float(engine.vel[i, 1]))
        rec["agent"].x = rec["pos"][0]
        rec["agent"].y = rec["pos"][1]


def _near_gate_mask(zone, pos, gate_xy):
    """`zone.near_gate` のベクトル版(同一式・同一比較 = 同一 bool)。"""
    n = pos.shape[0]
    out = np.zeros(n, dtype=bool)
    band = float(zone.gate["band_m"])
    b2 = band * band
    for gx, gy in gate_xy:
        dx = pos[:, 0] - gx
        dy = pos[:, 1] - gy
        out |= (dx * dx + dy * dy) <= b2
    return out


def _accumulate(zone, members, engine, prev_pos, prev_vel, dt, gate_xy, cont, st,
                pcfg, dt_acc=None) -> None:
    """境界連続性の指標と、個体別の身体観測をこのサブステップぶん積む。

    ★物理痩身(第一段A・2026-08-16): 全ペア距離行列(O(N²))と個体ごとの Python ループを
      セル法+ベクトル化に置換した。**値は 1 ビットも変えていない**:
        - ヒスト・帯・反転・接触・近傍数はすべて**カウントと bool** = 順序非依存。
        - 局所密度は「半径 density_radius_m 以内の人数」= カットオフ付きの数え上げなので
          セル法の候補(上位集合)から厳密な距離判定で数え直せば同じ整数になる。
        - min_gap は min = 順序非依存(orca_core.min_gap が全ペア版との一致を保証)。
        - speed_sum だけは浮動小数の累積和なので、**加算順序を変えていない**
          (個体ごと・サブステップ順の逐次加算のまま)。

    `dt_acc`(第154 レバーB): **適応 dt が ON のときだけ** 非 None。帯ごとの
      「サンプル秒」を実 dt で積む(反転率 [回/体·秒] の分母)。OFF では None =
      1 行も走らない = `continuity()` は従来どおり `samples × dt_sub` を使う。
    """
    pos = engine.pos
    vel = engine.vel
    n = len(members)
    dv = np.linalg.norm(vel - prev_vel, axis=1) / dt
    disp = np.linalg.norm(pos - prev_pos, axis=1)
    cont["jump_max_m"] = max(cont["jump_max_m"], float(disp.max()))
    # ---- 帯の判定(ゲート帯 = 緩和帯 / それ以外 = 内部)----
    at_gate = _near_gate_mask(zone, pos, gate_xy)
    # int(dv/_ACC_BIN) と同じ 0 方向切り捨て。先に上端で clip してから整数化するので、
    # 元の min(idx, _ACC_BINS−1) と同値かつ int64 の桁溢れも起きない(dv ≥ 0)。
    idx = np.clip(dv / _ACC_BIN, 0.0, float(_ACC_BINS - 1)).astype(np.int64)
    # 進行方向成分の符号反転。基準は **いま走っている経路区間の向き**(固定ベクトル)。
    #  - 入場時の向きを基準にすると「道なりに曲がった」だけで反転に数えてしまう。
    #  - 逆に「いまのゴールへの向き」を基準にすると、向きが個体と一緒に回るので
    #    符号がほぼ常に正になり **反転が原理的に検出できない**(実測で 0.000 になった)。
    #  区間の向きなら、ベンチの「通路の x 成分の符号反転」と同じ意味になる。
    seg = np.array([r["seg_dir"] for r in members], dtype=np.float64).reshape(n, 2)
    s = vel[:, 0] * seg[:, 0] + vel[:, 1] * seg[:, 1]
    sign = np.where(s > 0.0, 1, np.where(s < 0.0, -1, 0)).astype(np.int64)
    prev_sign = np.fromiter((r["sign"] for r in members), dtype=np.int64, count=n)
    flip = (sign != 0) & (prev_sign != 0) & (sign != prev_sign)
    for band, mask in (("gate", at_gate), ("interior", ~at_gate)):
        cnt = int(np.count_nonzero(mask))
        if not cnt:
            continue
        b = cont[band]
        counts = np.bincount(idx[mask], minlength=1)
        hist = b["hist"]
        for t in np.nonzero(counts)[0]:
            hist[int(t)] += int(counts[t])
        b["n"] += cnt
        b["samples"] += cnt
        b["flip"] += int(np.count_nonzero(flip & mask))
        if dt_acc is not None:      # 適応 dt: 秒は実 dt で積む(旧 blob 互換の get)
            b["sec"] = b.get("sec", 0.0) + cnt * dt_acc
    for i in np.nonzero((sign != 0) & (sign != prev_sign))[0]:
        members[int(i)]["sign"] = int(sign[i])
    # ---- 重なり(P2 決定 条件3 の検収値)----
    radius = np.array([r["radius"] for r in members], dtype=np.float64)
    gap = _orca.min_gap(pos, radius)
    if math.isfinite(gap):
        st["min_gap_m"] = gap if st["min_gap_m"] is None else min(st["min_gap_m"], gap)
    st["sep_iters_max"] = max(st["sep_iters_max"],
                              int(getattr(engine, "last_sep_iters", 0)))
    # ---- 個体別の身体観測(P3(3))----
    speed = np.linalg.norm(vel, axis=1)
    dens_r = float(pcfg["density_radius_m"])
    gap_m = float(pcfg["contact_gap_m"])
    if n > 1:
        # 候補 reach = max(密度半径, 2·r_max + 接触余裕)。どちらの判定も reach 以内にしか
        # 真を返さないので、セル法の候補は上位集合 = 数え上げは厳密に一致する。
        reach = max(dens_r, 2.0 * float(radius.max()) + gap_m)
        ii, jj = _sfm.neighbor_pairs(pos, reach)
        d = np.linalg.norm(pos[ii] - pos[jj], axis=1)
        dens_cnt = np.bincount(ii[d < dens_r], minlength=n)
        touch = np.zeros(n, dtype=bool)
        hit = d < (radius[ii] + radius[jj] + gap_m)
        if hit.any():
            touch[ii[hit]] = True
    else:
        dens_cnt = None
        touch = None
    for i, rec in enumerate(members):
        rec["speed_sum"] += float(speed[i])
        rec["speed_n"] += 1
        rec["step_n"] += 1
        if dens_cnt is not None:
            rec["dens_sum"] += int(dens_cnt[i])
            if touch[i]:
                rec["contact_n"] += 1


# --------------------------------------------------------------------------- #
# 退場・グラフ復帰
# --------------------------------------------------------------------------- #
def _advance_and_collect(sim, zone, members, engine) -> list:
    """経路の**次の通過点**へ目標を進め、退場した個体を返す。

    ★ なぜ「出口ゲートまで一直線」ではなく通過点追跡なのか(実測に基づく設計変更)
      最初の実装は目標をゾーンの出口ノードに固定した。すると物理は開放平面を**直線で**
      横断するのに対しグラフ経路は道なりに曲がるので、退場時の射影距離(= グラフ復帰の
      位置の跳び)が **41.4 m** に達した(mock 30体24step 実測)。これは「境界で位置の
      不連続を起こさない」という P3 の受入基準そのものを壊す。
      → **大域経路はグラフ・局所回避は物理**(歩行者/ロボット navigation の標準構成)に
        改めた。目標は常に「経路上の次のノード」で、到達したら次へ送る。物理は経路の
        近傍から離れないので、退場時の射影距離は数 m に収まる。
    退場は「経路の最後(= ゾーン外の出口ノード)へ到達した」または
    「一度ゾーンへ入った個体がゾーンの外に出た」で確定する。
    """
    out = []
    ar = zone.arrive_radius_m
    for i, rec in enumerate(members):
        x, y = float(engine.pos[i, 0]), float(engine.pos[i, 1])
        if zone.contains(x, y):
            rec["seen_inside"] = True
        path = rec["path"]
        last = len(path) - 1
        # 通過点の前進(複数の通過点を一気に跨ぐこともあるので while)
        while rec["wp"] < last:
            gx, gy = rec["wp_xy"]
            if math.hypot(x - gx, y - gy) >= ar:
                break
            reached = path[rec["wp"]]  # いま到達したグラフノード(= これから「直前ノード」になる)
            prev = rec["wp_xy"]
            rec["wp"] += 1
            rec["wp_xy"] = sim.city.node_xy(path[rec["wp"]])
            engine.goal[i, 0], engine.goal[i, 1] = rec["wp_xy"]
            rec["seg_dir"] = _unit(rec["wp_xy"][0] - prev[0], rec["wp_xy"][1] - prev[1])
            rec["sign"] = 0            # 区間が変わった瞬間を反転に数えない
            # ★竹-4 持ち越し②(第86バッチ保守 M-4): 所有中も **agent.node を進める**。
            #   これを入れる前は、ゾーンに入った瞬間の入場ゲートノードのまま所有が続くので、
            #   ノード基準の同席判定(cognition/channels._place_key の ("node", agent.node)・
            #   ext.crowd_local)が「実際には 100m 先を歩いている個体」を入口に居ることにして
            #   数えていた(横断中の群衆が丸ごと 1 ノードに溜まって見える)。
            #   意味論は `_release` の射影復元と同じ「直前に通過したノード」= path[i]。
            #   route は所有中に消費されない(_phase_move が飛ばす)ので触らない。退場時に
            #   (node, route, edge_offset) を射影で組み直す既存の手順はそのまま効く。
            #   決定論: 到達判定 (arrive_radius) の純関数で、乱数も LLM も増えない。
            #   既定 OFF(physics.zones_enabled=false)ではこの関数自体が呼ばれない。
            rec["agent"].node = reached
        gx, gy = rec["wp_xy"]
        reached_exit = (rec["wp"] >= last and math.hypot(x - gx, y - gy) < ar)
        if reached_exit or (rec["seen_inside"] and not zone.contains(x, y)):
            out.append(rec)
    return out


def _release(sim, zone, agent, step: int, sim_min: int, st: dict,
             reason: str = "gate", elapsed_s: float = 0.0, rec: dict | None = None,
             restore: bool = True, used_s: float = 0.0) -> None:
    """物理の所有を解いてグラフ状態を復元する(= 流出ゲート)。

    物理座標を経路の折れ線へ射影し、(node, route, edge_offset) を組み直す。
    射影距離が `handover_jump_max_m` を超えたら payload に far=true を残す(監視)。
    `restore=False` は「他フェーズが既にグラフ状態を据えている」場合(所有解除だけ行う)。
    """
    rec = rec if rec is not None else _record_of(agent)
    path = list(rec["path"] or ())
    rest = list(rec["rest"] or ())
    jump = 0.0
    if restore and len(path) >= 2:
        proj = _zones.project_on_path(sim.city, path, agent.x, agent.y)
        if proj is not None:
            i, off, px, py, jump = proj
            agent.node = path[i]
            agent.route = list(path[i + 1:]) + rest
            agent.edge_offset = float(off)
            agent.x, agent.y = float(px), float(py)
    # 2 層タイムラインの会計: **この step のうち**物理が使った秒数を残しておく
    # (同 step の _phase_move はその残りぶんしか進めない = 二重移動の防止)。
    # elapsed_s は step を跨いだ累積滞在(記録用)なので、ここでは used_s を使う。
    agent._phys_used_step = int(step)
    agent._phys_used_s = float(used_s)
    st["handover_jump_max_m"] = max(st["handover_jump_max_m"], float(jump))
    st["exit_total"] += 1
    st["dwell_sum_s"] = float(st["dwell_sum_s"]) + float(elapsed_s)
    st["dwell_n"] = int(st["dwell_n"]) + 1
    _finish_body(sim, agent, rec)
    _clear_record(agent)
    payload = {"zone": zone.id, "gate": rec["gate"], "dir": "exit",
               "reason": reason, "dwell_s": round(float(elapsed_s), 2),
               "jump_m": round(float(jump), 3)}
    if jump > float(zone.gate["handover_jump_max_m"]):
        payload["far"] = True
    sim.logger.log(Event(step=step, sim_min=sim_min, agent_id=agent.id,
                         kind="zone_gate", x=agent.x, y=agent.y, payload=payload))


def _maybe_force(sim, zone, rec, step, sim_min, st, used_s: float = 0.0) -> bool:
    """滞在が長すぎる個体を強制的にグラフへ返す(詰まりの安全弁)。返り値=返したか。"""
    limit = float(zone.gate["max_zone_steps"]) * float(sim.clock.step_seconds)
    if rec["elapsed_s"] < limit:
        return False
    st["forced_total"] += 1
    _release(sim, zone, rec["agent"], step, sim_min, st,
             reason="forced_zone", elapsed_s=rec["elapsed_s"], rec=rec, used_s=used_s)
    return True


def _maybe_force_wait(sim, zone, rec, step, sim_min, st) -> bool:
    """入場を待ちすぎた個体を所有解除してグラフへ返す(待ち行列の安全弁)。"""
    if rec["wait_steps"] < int(zone.gate["max_hold_steps"]):
        return False
    st["forced_total"] += 1
    # 待機中は 1 度もゾーンへ入っていない = グラフ状態は入場時のまま = 復元不要
    _release(sim, zone, rec["agent"], step, sim_min, st,
             reason="forced_wait", elapsed_s=0.0, rec=rec, restore=False)
    return True


# --------------------------------------------------------------------------- #
# 個体側の状態(agents pickle に自然同梱 = resume 安全)
# --------------------------------------------------------------------------- #
_FIELDS = ("path", "rest", "gate", "exit_xy", "span_m", "wp", "wp_xy", "pos", "vel", "dir0",
           "seg_dir", "v0", "radius", "waiting", "seen_inside", "wait_steps",
           "elapsed_s", "step_n", "speed_sum", "speed_n", "contact_n", "dens_sum",
           "sign")


def _unit(dx: float, dy: float) -> tuple[float, float]:
    n = math.hypot(dx, dy)
    return (dx / n, dy / n) if n > 1e-9 else (1.0, 0.0)


def _save_record(rec) -> None:
    agent = rec["agent"]
    for f in _FIELDS:
        setattr(agent, f"_phys_{f}", rec[f])


def _record_of(agent) -> dict:
    rec = {"agent": agent, "zone": agent._phys_zone}
    for f in _FIELDS:
        rec[f] = getattr(agent, f"_phys_{f}")
    return rec


def _clear_record(agent) -> None:
    for f in _FIELDS:
        if hasattr(agent, f"_phys_{f}"):
            delattr(agent, f"_phys_{f}")
    agent._phys_zone = None


def _finish_body(sim, agent, rec) -> None:
    """滞在中の実測を `Perception.body` の 3 欄へ焼く(P3(3))。

    1 サブステップも積分していない個体(入場待ちのまま返された等)は**何も書かない**
    = `body_of()` は None のまま = 欠測を捏造しない。
    """
    if not rec or not rec["speed_n"]:
        return
    n = float(rec["speed_n"])
    v_mean = rec["speed_sum"] / n
    v0 = float(rec["v0"]) or 1.0
    r = float(sim.physcfg["perception"]["density_radius_m"])
    area = math.pi * r * r
    agent._phys_body = {
        "blocked": max(0.0, min(1.0, 1.0 - v_mean / v0)),
        "contact": rec["contact_n"] / n,
        "local_density": (rec["dens_sum"] / n) / area,
    }


def _by_id(agents):
    return sorted(agents, key=lambda a: int(a.id))
