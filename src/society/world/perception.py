"""知覚(誰の声が聞こえるか)。同じ文脈(路上どうし/同じ建物の同じ階)のみ。

非ブロードキャスト原則: 範囲外(outside)のエージェントには何も聞こえない。

スケール: 全対全走査 O(n²) を避けるため、step ごとに1回だけ空間グリッド索引
(セル = perception_radius)を build_index() で構築し、hearers_of は近傍9セルだけを
走査する。**返り値の内容・順序は全対全走査(下の legacy 経路)とバイト一致**する
(cell = radius なので半径内の点は必ず ±1 セルに入り、距離判定は従来どおり残す)。
索引を使うのは位置が安定なフェーズ(_phase_drive/_decide)だけ。位置が動く _apply では
従来どおり agents リストを渡す(索引が古くなるため=live 走査で完全一致)。

擬似視覚(壁による遮蔽): occluder を渡すか install_occluder() で本層に据えると、距離・
同一階フィルタを通ったペアに視線(LOS)判定を追加する(vision.py の VisionOccluder)。
occluder が無いとき(既定)は挙動が従来と完全にバイト一致=遮蔽なし。occluder は幾何しか
見ないので no-fingerprint 契約を破らない。

── 有界化(声の段階 + 最寄り K)。正典 docs/plans/hearer-cap-plan.md §2 ────────────
`hearers_of` / `PerceptIndex.hearers` は 2 つの**任意**引数を受け取る:

  radius_eff … 「この発話が実際に届く距離」[m]。声の段階(作用点0)の実体で、
               **索引のセル半径(= world.perception_radius_m)以下へクランプ**する
               (r ≤ セル幅なら半径内の点は必ず ±1 セルに入る = 走査セルは 9 のまま)。
  cap        … 聞き手の最大人数(位相的近傍)。**距離二乗 昇順 → id 昇順**で選抜し、
               返り値は従来どおり **id 昇順**(下流契約不変)。規則は第141 S15
               (scheduler._attention_limited)と完全に同一で、乱数を 1 粒も引かない。

**どちらも既定(cap=0 / radius_eff=None)なら下の分岐へ 1 度も入らない**: 既存の列挙
コードが 1 行も変わらずそのまま走る = golden L1 バイト一致(R1 をコード構造で保証)。

有界経路に入ったときだけ 9 セルの座標を numpy 化(**セル単位で遅延構築・step 内で再利用**)
して距離判定と選抜をベクトル化する。半径判定は `np.hypot` = `math.hypot` と同一の丸め、
順位は `dx*dx+dy*dy` = S15 と同一の鍵なので、ループ実装と**同じ集合・同じ順序**になる
(tests が機械照合する)。遮蔽器が実際に据わっているときはベクトル化を捨ててループへ
後退する(正しさ優先。本選は遮蔽未配線 = occluder None)。

── セル寸法の細分化(`world.perception_cell_m`。既定 0 = 現行と完全同一)────────
索引のセル寸法は長らく **常に `perception_radius_m`(40m)** だった。声の段階(第149)が
入って実効半径が 5m になっても、問い合わせは 40m セル × 9 個 = **(120m)² の中の全員**を
numpy 距離計算に流していた(π(5m)² と比べて候補母集団が ~183 倍)。250k 夕方 step の
py-spy でこの `_hearers_bounded` が step 時間の ~80% を占めていた。

linked-cell / spatial hashing の定石は **セル寸法 ≈ 相互作用半径**、複数半径が混在するときは
**セル寸法を最小半径に合わせて大半径はリング走査**(ceil(r/cell) セル)。本層はこれを
**粗格子を残したまま**入れる:

  粗格子 `cells`  … セル寸法 = `radius`(**現行のまま 1 バイトも変えない**)。
                    既定の `hearers()` / `_count()`(= count_hearers)はここしか見ない。
  細格子 `_fcells`… セル寸法 = `cell_m`(> 0 のときだけ・**最初の有界クエリで遅延構築**)。
                    有界経路(cap / radius_eff)が「細格子の方が明らかに得」と判定した
                    半径のときだけ使う: リング = ceil(r/cell)・走査 (2ring+1)² セル。

  判定(`_fine_ring`。純関数・決定論): ring ≤ `_FINE_RING_MAX` かつ
  走査面積 × `_FINE_AREA_MARGIN` ≤ 粗格子の走査面積(9·radius²)。満たさなければ 0 =
  **粗格子(現行コード)へそのまま落ちる** = 大半径クエリ(叫び 30/40m・既定半径)に
  性能退行が起こりえない(構造で保証)。radius=40 / cell=5 なら r ≤ 20m が細格子。

  判定②(局所の混み具合。`_FINE_MIN_GAIN` / conf `world.perception_fine_gate`):
  細セル化が買うのは「要素あたりの仕事」だけでクエリ 1 本の固定費は減らないので、
  **疎な近傍では細格子が純損**になる。そこで粗格子 9 セルの在圏数から削減できる要素数を
  見積もり、しきい値を超えるときだけ細格子へ回す。**しきい値をどこに置いても返り値は同一**
  (下記)= 速さだけのつまみ。既定は実装定数(2,000)で、conf に書けば index ごとに差し替わる。

**どちらの格子を通っても返り値は同一**(集合が同じ + 選抜規則が入力順に依らない:
id は一意なので `lexsort((ids, d2))` も `argsort(ids)` も全順序)。テストが総当たりで機械照合する。

── 人数だけが要る呼び手(`count_hearers`)────────────────────────────────────
観測チャンネル `ext.encounter`(cognition/channels.py)は「声が届く**人数**」しか使わない
のに `len(hearers_of(...))` を呼んでいた = 全個体ぶんのリスト構築と id 昇順ソートを毎 step
捨てるために作っていた(25k 夕方の py-spy で step 時間の 15.8%)。`count_hearers` は
**同じ集合の要素数**(同じ `_context` 判定・同じ半径判定式・同じ遮蔽適用)を、列挙も整列も
せずに返す。cap / radius_eff は受け取らない = このチャンネルの現行 semantics(物理的に
声が届く人数)をそのまま保つ。索引経路は半径マスクの合計でベクトル化し、`np.hypot` と
`math.hypot` の 1 ULP 差は半径まわりの帯だけ `math.hypot` で裁定する(下記 `_radius_band`)。

── 「居るか否か」の 1 ビットしか要らない呼び手(`has_hearers`)─────────────────
第152 が `_decide_g` の `has_company` を `len(hearers_of(...))` → `count_hearers(...) > 0`
へ落としたが、**人数まで数える必要も無い**呼び手がまだ 4 か所ある(`scheduler._decide_g`
の `has_company` と `_phase_drive` の face 判定 3 か所)。`has_hearers` は
**`count_hearers(...) > 0` と厳密に同値**な bool を、**最初の 1 件が見つかった時点で
打ち切って**返す(docs/plans/step-time-audit.md A2)。

早期打ち切りが実際に効くようにするため、索引経路は `_count` のような「近傍 9 セルの
連結配列を作ってから全長を numpy で舐める」形を**唯一の経路にしない**。3 段構えで、
どの段も見る要素の集合は `_count` と同一(= 順序も分割も答えに影響しない):
  ⓪ 疎な近傍(在圏数 ≤ `_EXISTS_LOOP_MAX`)… 素の Python ループ。numpy のクエリ固定費
     (~6 µs)を払わない = 差し替え前の `hearers_of` の速さをそのまま保つ。
  ① 密な近傍 … 中心セルの先頭 `_EXISTS_PROBE_MAX` 人だけをループで先読み。ほぼ必ず
     ここで当たる(実測 ~0.5 µs = `count_hearers` の 25〜120 倍速)。
  ② 外れたら `_count` と同じ連結キャッシュを**全長 1 パス**(`_count` を超えない)。
判定式・帯の裁定(`_radius_band`)・遮蔽の扱いは `_count` と 1 行ずつ同型なので、
`has_hearers(...) == (count_hearers(...) > 0)` が全入力で成り立つ
(tests/test_has_hearers.py が密/疎/境界/±ULP 帯/しきい値差し替えの総当たりで機械照合)。
"""
from __future__ import annotations

import functools
import math

import numpy as np

# 本層に据えた既定の遮蔽器(None=遮蔽なし=従来と完全同一)。引数 occluder が優先。
# 全 hearers_of 呼び出し(索引/legacy 双方)に一括で効かせる本番トグルの据え付け先。
_OCCLUDER = None


def install_occluder(occluder) -> None:
    """モジュール全体の既定遮蔽器を据える(シミュ初期化で1回)。None で解除。"""
    global _OCCLUDER
    _OCCLUDER = occluder


def clear_occluder() -> None:
    global _OCCLUDER
    _OCCLUDER = None


def active_occluder():
    return _OCCLUDER


def _resolve(occluder):
    """引数優先・無ければ据え付けの既定遮蔽器。両方 None なら遮蔽なし。"""
    return occluder if occluder is not None else _OCCLUDER


def _context(agent) -> tuple:
    if agent.loc == "outside":
        return ("outside", agent.id)   # 誰とも重ならない
    if agent.building:
        return ("bld", agent.building, agent.floor)
    return ("street",)


# `np.hypot` と `math.hypot` は同じ入力に対して最大 1 ULP ずれる(実測: 100 万組で
# 最大 1 ULP)。人数だけを数えるベクトル化経路(`PerceptIndex._count`)が
# `hearers` と**厳密に同じ集合の要素数**を返すために、半径のまわりに余裕 4 ULP の帯を
# 取り、帯の中に落ちた要素だけを `math.hypot` で裁定する(帯の外は両実装で判定が一致
# することが 1 ULP 保証から従う)。帯に入る要素は実データでは事実上ゼロ件。
_BAND_ULP = 4

# ---- セル細分化(`world.perception_cell_m`)の 2 定数。既定 cell_m=0 では 1 度も読まれない ----
# `_FINE_RING_MAX`   … 細格子で走査してよいリング半径の上限(4 = 最大 81 セル)。
#   細セル化の代償は「走査セル数の増加」= 連結(np.concatenate)の入力本数。ここを開けすぎると
#   セル 1 個あたりの固定費が候補削減の利得を食う(r=40 なら 17×17 = 289 本になる)。
# `_FINE_AREA_MARGIN`… 細格子を選ぶのに要求する面積比の余裕(4 = 走査面積が 1/4 以下)。
#   面積比が僅差なら固定費のぶん粗格子(= 現行経路)の方が速いので、迷ったら現行へ倒す。
_FINE_RING_MAX = 4
_FINE_AREA_MARGIN = 4.0
# `_FINE_MIN_GAIN` … 細格子に回すのに要求する「候補削減の絶対数」(要素数)。
#   細セル化が買うのは**要素あたりの仕事**(実測 ~2.8 ns/要素)だけで、クエリ 1 本の
#   固定費(numpy 呼び出し ~20 本 = ~20 µs)は減らない。むしろ細格子の連結は共有相手が
#   少ないぶん固定費が ~6 µs 増える。したがって **疎な近傍では細格子が純損**になる
#   (実測: 0.013 人/m² で 0.7 倍 = 3 割遅い)。そこで粗格子 9 セルの在圏数から
#   「削減できる要素数」を見積もり、6 µs / 2.8 ns ≈ 2,000 要素を超えるときだけ細格子へ回す。
#   = 密な夕方のセルだけが細格子を通り、夜間・郊外セルは現行経路のまま(退行しえない)。
#   ★第153: この 2,000 は 25k 実測の当て推量だった。250k では**同じ密度帯の絶対数が違う**
#     ので、しきい値そのものを conf(`world.perception_fine_gate`)から差せるようにした。
#     既定はこの 2,000 のまま = 現行と 1 ビットも変わらない(門は集合を変えず、
#     どちらの格子を通っても返り値は同一)。
_FINE_MIN_GAIN = 2000.0

# 出荷時 conf(`world.perception_fine_gate`)に書いてある数。**この数がそのまま書いて
# あるときは「実装既定に従う」= 上の定数を問い合わせ時に読む**(= 定数が唯一の真実)。
# こうしておくと「既定値を明示的に書いたラン」と「キーを書かないラン」が構造的に同値になり、
# 既存の道具・テストが `_FINE_MIN_GAIN` を差し替えて門を開け閉めする作法もそのまま効く。
_FINE_GATE_CONF_DEFAULT = 2000.0

# 粗格子の走査オフセット。**現行の `for dx in (-1, 0, 1)` と同一のタプル**(順序も同一)。
_RING1: tuple[int, ...] = (-1, 0, 1)

# ---- 存在判定(`has_hearers` = `count_hearers(...) > 0`)の走査規則 --------------------
# 走査の順序・分割は**返り値に一切影響しない**(bool の or は可換・結合的で、見る要素の
# 集合は `_count` と同一)= 速さだけのつまみ。設計目標は 2 つで、micro bench の実測で決めた:
#   (a) 密な近傍で速い(= A1/A2 の狙い)
#   (b) **どんな配置でも差し替え前の 2 経路(`hearers_of` の Python ループ /
#       `count_hearers` の numpy 全長パス)より目に見えて遅くならない**
#
# `_EXISTS_ORDER` … セルを見る順。中心セル(話者自身が居るセル = いちばん近い候補が
#   集まる場所)から見る。遮蔽器あり経路と疎な近傍のループ経路が使う。
_EXISTS_ORDER: tuple[tuple[int, int], ...] = (
    (0, 0),
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)
# `_EXISTS_LOOP_MAX` … 近傍 9 セルの在圏数がこれ以下なら**素の Python ループ**で見る。
#   numpy はクエリ 1 本あたり ~6 µs の固定費(呼び出し ~7 本)を持つのに対し、
#   Python ループは 1 要素 ~0.09 µs。疎な近傍(夜間・郊外・屋内の個室)ではループの方が
#   桁で速い(実測: 近傍 1 人で 1.1 µs 対 8.7 µs)。差し替え前の `hearers_of` はこの
#   ループだったので、**そこを退行させないための下駄**でもある。しきい値は当たりの無い
#   最悪ケースの交点(6 µs ÷ 0.09 µs ≒ 65 要素)に置く。
#   在圏数は `_coarse_load`(dict の get と len だけ・セル単位でキャッシュ)で得る。
_EXISTS_LOOP_MAX = 64
# `_EXISTS_PROBE_MAX` … 密な近傍での①「中心セル先読み」で見る人数の上限。
#   中心セルに人が居れば、その先頭の数人のうち誰かが半径(= セル幅)内に居る確率はほぼ 1
#   なので、**9 セル連結(`_count_arrays`)を作らずに** ~0.2 µs で決着することが多い。
#   外れても損は高々この人数ぶんのループ(~3 µs)で、そのあと②へ落ちる。
#   ここを numpy で撃つ実装も試したが、この規模ではループの方が速かった(固定費が支配的)。
_EXISTS_PROBE_MAX = 32


@functools.lru_cache(maxsize=16)
def _ring_offsets(ring: int) -> tuple[int, ...]:
    """リング半径 → 走査オフセット列(-ring … +ring)。ring=1 は `_RING1` と同一の並び。"""
    return tuple(range(-int(ring), int(ring) + 1))


@functools.lru_cache(maxsize=32)
def _radius_band(radius: float) -> tuple[float, float]:
    """(lo, hi) = 半径の ±4 ULP。`h <= lo` は確実に圏内・`h > hi` は確実に圏外。

    半径は run 内で数種類しかないので純関数として憶えておく(呼び手は個体ごと)。"""
    lo = hi = float(radius)
    for _ in range(_BAND_ULP):
        lo = math.nextafter(lo, -math.inf)
        hi = math.nextafter(hi, math.inf)
    return lo, hi


# --------------------------------------------------------------------------- #
# 声の段階(作用点0)= 発話の種類 → 声の大きさ → 実効半径 の決定論写像
#   正典: docs/plans/hearer-cap-plan.md §2「作用点0」。
#   音響: 会話音声 ~60dB@1m・距離が倍で −6dB・街頭騒音 70-95dBA → 通常声の了解圏は数 m。
#   屋内(騒音 ~50dB)は同じ声が 2-3 倍届く。段階の**選び方**は engine 側(発話文脈からの
#   決定論写像)で、本層は「段階 → 距離」の物理表しか持たない(no-fingerprint)。
# --------------------------------------------------------------------------- #
SPEECH_LEVELS: tuple[str, ...] = ("normal", "raised", "shout")

SPEECH_DEFAULTS: dict = {
    "enabled": False,                                   # 既定 OFF = 半径一律(現行と完全同一)
    "normal": {"outdoor_m": 5.0, "indoor_m": 10.0},     # 通常の会話声
    "raised": {"outdoor_m": 12.0, "indoor_m": 20.0},    # 張り上げ(+9dB ≒ 2-3 倍)
    "shout": {"outdoor_m": 30.0, "indoor_m": 40.0},     # 叫び(+18-24dB ≒ 8-16 倍)
}


def build_speech_cfg(raw, max_radius_m: float) -> dict:
    """conf の `world.speech_levels` を正準化する(純関数・既定 OFF=現行と完全同一)。

    すべての距離は **[0, max_radius_m] へクランプ**する(= 索引のセル幅を超えない)。
    上限を超える値を書いても静かに落ちるのではなく、`world.perception_radius_m` まで
    切り下げられる(走査セル範囲を広げる実装を持たない = 9 セルの不変条件を守るため)。
    """
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(raw):
            raw = OmegaConf.to_container(raw, resolve=True)
    except Exception:                                   # noqa: BLE001 (omegaconf 不在でも動く)
        pass
    raw = dict(raw or {})
    top = float(max_radius_m)
    cfg: dict = {"enabled": bool(raw.get("enabled", SPEECH_DEFAULTS["enabled"]))}
    for level in SPEECH_LEVELS:
        base = dict(SPEECH_DEFAULTS[level])
        got = dict(raw.get(level, {}) or {})
        row = {}
        for side in ("outdoor_m", "indoor_m"):
            try:
                v = float(got.get(side, base[side]))
            except (TypeError, ValueError):
                v = float(base[side])
            row[side] = min(max(0.0, v), top)            # ★ ≤ perception_radius_m を強制
        cfg[level] = row
    return cfg


def speech_radius(speech_cfg: dict, level: str, speaker) -> float:
    """段階 × 屋内外 → 実効半径 [m](純関数)。未知の段階は保守側 normal へ倒す。"""
    row = speech_cfg.get(level) or speech_cfg["normal"]
    indoor = _context(speaker)[0] == "bld"
    return float(row["indoor_m"] if indoor else row["outdoor_m"])


def speech_cfg_of(sim) -> dict:
    """`world.speech_levels` を遅延構築して sim にキャッシュする(呼び手 2 つの共有口)。

    キャッシュ属性 `_speech_levels_cfg` は L1/L2/乱数のどこにも現れない(conversation._cfg と
    同型)。既定 OFF のとき返る dict の `enabled` は False = 呼び手は現行経路へ落ちる。"""
    cfg = getattr(sim, "_speech_levels_cfg", None)
    if cfg is None:
        try:
            world = sim.cfg.get("world", {}) or {}
            raw = world.get("speech_levels", None)
            top = float(world.get("perception_radius_m", 0.0) or 0.0)
        except Exception:                               # noqa: BLE001 (旧 config 互換)
            raw, top = None, 0.0
        cfg = build_speech_cfg(raw, top)
        sim._speech_levels_cfg = cfg
    return cfg


def nearest_k(speaker, candidates: list, cap: int) -> list:
    """候補から最寄り K 人を選び **id 昇順**で返す(第141 S15 `_attention_limited` と同一規則)。

    順序規則: **距離二乗 昇順 → id 昇順**(平方根も乱数も使わない完全な決定論)。
    cap<=0 または候補数<=cap のときは「id 昇順に並べ替えるだけ」= 絞らないときと同一集合・
    同一順序(呼び手が既に id 昇順を期待しているため並べ替えは常に行う)。
    """
    if cap <= 0 or len(candidates) <= cap:
        return sorted(candidates, key=lambda a: a.id)
    sx, sy = float(speaker.x), float(speaker.y)

    def _key(h):
        dx, dy = float(h.x) - sx, float(h.y) - sy
        return (dx * dx + dy * dy, int(h.id))
    return sorted(sorted(candidates, key=_key)[:cap], key=lambda a: int(a.id))


def _bounded_pick(nb, sx: float, sy: float, sid: int, radius: float,
                  cap: int, box_prefilter: bool = True) -> list:
    """連結済み近傍 (xs, ys, ids, flat) から「半径内 → 最寄り cap 人」を選ぶ(ベクトル化)。

    ★既定枝(`box_prefilter=True`)は第149 の `_hearers_bounded` 本体から切り出したもの
      (粗格子と細格子で同じ関数を通す = 2 経路が将来にわたって食い違えない構造)。
      第153 で**判定式を保ったまま numpy 呼び出しだけを畳んだ**(下の ① の註記。等価性は
      tests/test_hotpath_fixes.py が第152 の逐語コピーと総当たりで機械照合する)。
      半径判定は `np.hypot`(= `math.hypot` と同一の丸め)のままなので集合はバイト一致。
      **入力の並び順に依らない**: id は一意なので `lexsort((ids, d2))` も `argsort(ids)` も
      全順序 = 格子が変わっても同じ答え。

    `box_prefilter=False`(細格子だけが使う)は ① の外接正方形を省く縮退。判定式は
    「np.hypot <= radius かつ id != sid」で ① 有りと**厳密に同じ集合**(① は
    hypot >= max(|dx|,|dy|) による等価な前段でしかない)。細格子では近傍の広さが半径と
    同程度なので ① で落ちるのは半分以下で、numpy 呼び出し 5 本ぶんの固定費の方が高くつく。
    """
    xs, ys, ids, flat = nb
    dx_a = xs - sx
    dy_a = ys - sy
    if not box_prefilter:
        keep = np.flatnonzero((np.hypot(dx_a, dy_a) <= radius) & (ids != sid))
        n = int(keep.size)
        if n == 0:
            return []
        return _bounded_rank(dx_a, dy_a, ids, flat, keep, n, cap)
    # ① 外接正方形で粗く落とす(hypot(dx,dy) >= max(|dx|,|dy|) なので**厳密に等価**な
    #    前フィルタ)。声の段階(5m)のように半径がセル幅(40m)より小さいとき、
    #    高価な hypot に渡す要素数が桁で減る。
    #    ★第153 の畳み込み: 判定式は同じまま **近傍全長(N)を舐める回数を 10 → 7** に減らす。
    #      (a) `(|dx|<=r) & (|dy|<=r)` は `max(|dx|,|dy|) <= r` と**厳密に等価**
    #          (有限座標では両辺が同じ bool。-0.0 も abs で 0.0 に潰れる)ので、
    #          比較 2 本 + and 1 本 → maximum 1 本 + 比較 1 本に畳める。
    #      (b) 自分自身の除外 `ids != sid` は N 要素の比較 + and だったが、外接正方形を
    #          通った要素は(半径 5m / セル幅 40m では)全体の数 % しか無い。**通過後**に
    #          掛ければ同じ集合・同じ昇順のまま、その 2 本が小さい配列の仕事になる。
    ax = np.abs(dx_a)
    np.maximum(ax, np.abs(dy_a), out=ax)
    box = np.flatnonzero(ax <= radius)
    if box.size == 0:
        return []
    # ② 半径判定は np.hypot(= math.hypot と同一の丸め)= ループ実装と同じ集合。
    ids_b = ids[box]
    hit = (np.hypot(dx_a[box], dy_a[box]) <= radius) & (ids_b != sid)
    keep = box[hit]
    n = int(keep.size)
    if n == 0:
        return []
    # `ids[keep]` は既に手元にある(`ids_b[hit]`)ので選抜側で取り直さない。
    return _bounded_rank(dx_a, dy_a, ids, flat, keep, n, cap, ids_b[hit])


def _bounded_rank(dx_a, dy_a, ids, flat, keep, n: int, cap: int, ids_k=None) -> list:
    """半径内に残った添字 `keep` から「最寄り cap 人 → id 昇順」を返す(第149 と同一の規則)。

    ★第153 の畳み込み(**値は 1 ビットも変えない**。numpy 呼び出し 24 本 → 17 本):
      - `ids[keep]` / `dx_a[keep]` / `dy_a[keep]` を 1 度だけ取って使い回す
        (従来は cap 枝で `dx_a[keep]` 2 回・`dy_a[keep]` 2 回・`ids[keep]` 2 回)。
        `ids_k` は呼び手が既に持っていれば受け取る(`_bounded_pick` の外接正方形経路)。
      - しきい値は `argpartition` で cap 件の窓を切って `max` を取っていたが、それは
        定義上「cap 番目に小さい距離二乗」= `np.partition(d2, cap-1)[cap-1]` そのもの
        (どちらも配列中の**同じ要素の値**で、演算を 1 つも挟まない)。4 呼び → 2 呼び。
      - 最終の並べ替えは keep 空間の添字で行ってから 1 度だけ実体化する
        (`ids[chosen]` を取り直さない)。返す個体の並びは従来と同一。
      - 個体の取り出しは `.tolist()`(= Python int)経由にして、numpy スカラーごとの
        `int()` 変換を落とす。参照する `flat` の要素は同じ。
    """
    if ids_k is None:
        ids_k = ids[keep]
    if cap <= 0 or n <= cap:
        sel = keep[np.argsort(ids_k, kind="stable")]         # id 昇順(従来の並び)
        return [flat[i] for i in sel.tolist()]
    # 順位鍵は距離二乗(S15 と同一)。cap 番目の順序統計量でしきい値を切り、境界の同値だけ
    # 厳密に (距離二乗, id) で並べ直す = 全体ソートを避けた O(n) 選抜(決定論)。
    dx_k = dx_a[keep]
    dy_k = dy_a[keep]
    d2 = dx_k * dx_k + dy_k * dy_k
    thresh = float(np.partition(d2, cap - 1)[cap - 1])
    cand = np.flatnonzero(d2 <= thresh)
    order = np.lexsort((ids_k[cand], d2[cand]))
    sel = cand[order[:cap]]
    sel = sel[np.argsort(ids_k[sel], kind="stable")]         # 返りは id 昇順
    return [flat[i] for i in keep[sel].tolist()]


class PerceptIndex:
    """step 単位の空間ハッシュ索引。cell = (context, floor(x/r), floor(y/r))。

    睡眠中・範囲外は聞き手になり得ないので索引から除外する(legacy 走査の
    `other.sleeping` / context 不一致スキップと同値)。位置が確定したフェーズで
    一度だけ構築し、その間だけ hearers_of に渡す。
    """

    __slots__ = ("radius", "_inv", "cells", "_occ", "_np", "_nb", "_nc",
                 "cell_m", "_finv", "_fcells", "_fnp", "_fnb", "_fring", "_load",
                 "fine_gain")

    def __init__(self, radius: float, occluder=None, cell_m: float = 0.0,
                 fine_gain=None):
        self.radius = float(radius)
        self._inv = (1.0 / self.radius) if self.radius > 0 else 0.0
        self.cells: dict[tuple, list] = {}
        self._occ = occluder
        # 有界経路(cap / radius_eff)のときだけ育つ numpy キャッシュ。
        #   _np … セル 1 個の (x, y, id) 配列(近傍 9 個の中心セルから共有される)
        #   _nb … 中心セルごとの 9 セル連結(**同じセルに居る話者どうしが共有する**)。
        #         これが無いと連結と平坦化が話者ごとに O(近傍総数) 掛かり、
        #         ベクトル化の利得を打ち消す(密なセルでは 3 桁の差になる)。
        # 既定経路では **どちらも 1 度も触らない**(空 dict のまま)= 追加コストも状態もゼロ。
        # 索引は step ごとに作り直されるのでキャッシュの寿命も 1 step(陳腐化しない)。
        self._np: dict[tuple, tuple] = {}
        self._nb: dict[tuple, tuple] = {}
        #   _nc … 人数だけ数える経路(count_hearers)の 9 セル連結 (x, y, id)。
        #         _nb と違い**個体オブジェクトの平坦リストを作らない**(人数に要らない)。
        #         _np を土台に共有するので、同じセルの全員で連結は 1 回きり。
        self._nc: dict[tuple, tuple] = {}
        # ---- 細格子(`world.perception_cell_m`)。既定 0 = 以下は全て不活性 --------------
        # cell_m が半径以上なら細分化の意味が無い(粗格子と同じかそれより粗い)ので 0 へ倒す。
        cm = float(cell_m or 0.0)
        self.cell_m: float = cm if (0.0 < cm < self.radius) else 0.0
        self._finv = (1.0 / self.cell_m) if self.cell_m > 0.0 else 0.0
        # _fcells は **最初の細格子クエリで遅延構築**(None = まだ 1 度も要求されていない)。
        # 既定 cell_m=0 では永遠に None のまま = 追加の走査もメモリもゼロ。
        self._fcells: dict | None = None
        self._fnp: dict[tuple, tuple] = {}      # 細セル 1 個の (x, y, id)
        self._fnb: dict[tuple, tuple] = {}      # (ctx, cx, cy, ring) → 連結済み近傍
        self._fring: dict[float, tuple] = {}    # 半径 → (リング半径, 面積比)の記憶
        self._load: dict[tuple, int] = {}       # 粗セル → 近傍 9 セルの在圏数
        # 細格子へ回す門の「削減できる要素数」しきい値(`world.perception_fine_gate`)。
        # **None(既定)= 実装既定 `_FINE_MIN_GAIN` を問い合わせ時に読む** = 現行と完全同一
        # (モジュール定数が唯一の真実のまま = 既存のテスト/道具がその定数を差し替えて
        #  門を開け閉めする従来の作法をそのまま保つ)。
        self.fine_gain = None if fine_gain is None else float(fine_gain)

    def _cell_xy(self, x: float, y: float) -> tuple[int, int]:
        return (math.floor(x * self._inv), math.floor(y * self._inv))

    # ---- 細格子(cell_m > 0 のときだけ触られる)---------------------------------------
    def _fine_ring(self, radius: float) -> tuple:
        """クエリ半径 → (走査リング半径, 走査面積比)。**ring=0 = 細格子を使わない(=現行)**。

        規則(決定論・純関数・step 内で憶える):
          ring = ceil(radius / cell)  … 半径内の点が必ず ±ring セルに入る最小値。
          採用するのは ring <= `_FINE_RING_MAX` かつ 走査面積 × `_FINE_AREA_MARGIN` が
          粗格子の走査面積 (3·radius)² 以下のときだけ。それ以外は 0 を返して現行経路へ。
        面積比 = 細格子の走査面積 / 粗格子の走査面積(在圏数から削減要素数を見積もる材料)。
        """
        cell = self.cell_m
        if cell <= 0.0:
            return (0, 1.0)
        got = self._fring.get(radius)
        if got is not None:
            return got
        ring = int(math.ceil(radius / cell))
        if ring < 1:
            ring = 1
        while ring * cell < radius:          # 切り上げの浮動小数誤差に対する保険(通常 0 周)
            ring += 1
        span = (2 * ring + 1) * cell
        wide = 3.0 * self.radius
        ratio = (span * span) / (wide * wide) if wide > 0.0 else 1.0
        if ring > _FINE_RING_MAX or ratio * _FINE_AREA_MARGIN > 1.0:
            got = (0, 1.0)
        else:
            got = (ring, ratio)
        self._fring[radius] = got
        return got

    def _coarse_load(self, ctx: tuple, cx: int, cy: int) -> int:
        """粗格子 9 セルの在圏数(= 現行経路が距離計算に流す候補数)。セル単位で憶える。

        細格子へ回すかの判定材料。**dict の get と len だけ**で、numpy も配列確保も伴わない。"""
        key = (ctx, cx, cy)
        got = self._load.get(key)
        if got is None:
            cells = self.cells
            got = 0
            for dx in _RING1:
                for dy in _RING1:
                    bucket = cells.get((ctx, cx + dx, cy + dy))
                    if bucket:
                        got += len(bucket)
            self._load[key] = got
        return got

    def _fine_cells(self) -> dict:
        """粗格子の中身を細セルへ振り直した dict(**最初の要求で 1 回だけ**構築)。

        索引は step ごとに作り直されるので寿命も 1 step(陳腐化しない)。走査順は粗格子の
        挿入順 = build_index に渡した agents の順で決まるが、**返り値はどのみち入力順に
        依らない**(`_bounded_pick` / `nearest_k` の順序規則が全順序)。"""
        fc = self._fcells
        if fc is None:
            fc = {}
            inv = self._finv
            floor = math.floor
            for (ctx, _cx, _cy), bucket in self.cells.items():
                for a in bucket:
                    fc.setdefault((ctx, floor(a.x * inv), floor(a.y * inv)),
                                  []).append(a)
            self._fcells = fc
        return fc

    def _fine_cell_arrays(self, key: tuple, bucket: list) -> tuple:
        """細セル 1 個の (x, y, id) を numpy 化して step 内でキャッシュする(`_cell_arrays` の細格子版)。

        ★粗格子の `_np` と**別の dict** に持つ: キーの形((ctx, cx, cy))が同じでも指す
          セルは別物なので、同居させると衝突する。"""
        arr = self._fnp.get(key)
        if arr is None:
            n = len(bucket)
            arr = (np.fromiter((float(a.x) for a in bucket), np.float64, n),
                   np.fromiter((float(a.y) for a in bucket), np.float64, n),
                   np.fromiter((int(a.id) for a in bucket), np.int64, n))
            self._fnp[key] = arr
        return arr

    def _fine_neighborhood(self, ctx: tuple, cx: int, cy: int, ring: int):
        """細格子の (2·ring+1)² セルを連結した (xs, ys, ids, flat)(遅延構築・共有)。

        キーは **(中心セル, リング半径)**: 同じ中心セルでも半径階級が違えば別の連結になる
        (声の段階が normal/raised を混ぜても互いのキャッシュを壊さない)。近傍が空なら None。

        ★`_neighborhood`(粗格子・ring=1 固定)と統合しないのは、既定経路のコードを
          1 行も動かさないため(R1 を構造で守る)。選抜の本体は `_bounded_pick` で共有する。"""
        key = (ctx, cx, cy, ring)
        if key in self._fnb:
            return self._fnb[key]
        cells = self._fine_cells()
        offs = _ring_offsets(ring)
        xs_l: list = []
        ys_l: list = []
        ids_l: list = []
        objs: list = []
        for dx in offs:
            for dy in offs:
                ck = (ctx, cx + dx, cy + dy)
                bucket = cells.get(ck)
                if not bucket:
                    continue
                xa, ya, ia = self._fine_cell_arrays(ck, bucket)
                xs_l.append(xa)
                ys_l.append(ya)
                ids_l.append(ia)
                objs.append(bucket)
        if not objs:
            self._fnb[key] = None
            return None
        nb = (xs_l[0] if len(xs_l) == 1 else np.concatenate(xs_l),
              ys_l[0] if len(ys_l) == 1 else np.concatenate(ys_l),
              ids_l[0] if len(ids_l) == 1 else np.concatenate(ids_l),
              objs[0] if len(objs) == 1 else [a for b in objs for a in b])
        self._fnb[key] = nb
        return nb

    def add(self, agent) -> None:
        if agent.sleeping:
            return
        ctx = _context(agent)
        if ctx[0] == "outside":
            return
        cx, cy = self._cell_xy(agent.x, agent.y)
        self.cells.setdefault((ctx, cx, cy), []).append(agent)

    # ---- 有界経路(cap / radius_eff)。既定では 1 度も呼ばれない ------------------
    def _cell_arrays(self, key: tuple, bucket: list) -> tuple:
        """セル 1 個の (x, y, id) を numpy 化して step 内でキャッシュする(遅延構築)。

        索引は step ごとに作り直される(build_index)ので、キャッシュの寿命も 1 step。
        位置が動くフェーズでは索引そのものを使わない規約なので、値が古くなることはない。
        """
        arr = self._np.get(key)
        if arr is None:
            n = len(bucket)
            arr = (np.fromiter((float(a.x) for a in bucket), np.float64, n),
                   np.fromiter((float(a.y) for a in bucket), np.float64, n),
                   np.fromiter((int(a.id) for a in bucket), np.int64, n))
            self._np[key] = arr
        return arr

    def _neighborhood(self, ctx: tuple, cx: int, cy: int):
        """中心セルの近傍 9 セルを連結した (xs, ys, ids, flat) を返す(遅延構築・共有)。

        **同じセルに居る話者は同じ近傍を見る**ので、連結と平坦化はセルにつき 1 回で済む。
        近傍に誰も居なければ None。"""
        key = (ctx, cx, cy)
        if key in self._nb:
            return self._nb[key]
        cells = self.cells
        xs_l: list = []
        ys_l: list = []
        ids_l: list = []
        objs: list = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                ck = (ctx, cx + dx, cy + dy)
                bucket = cells.get(ck)
                if not bucket:
                    continue
                xa, ya, ia = self._cell_arrays(ck, bucket)
                xs_l.append(xa)
                ys_l.append(ya)
                ids_l.append(ia)
                objs.append(bucket)
        if not objs:
            self._nb[key] = None
            return None
        nb = (xs_l[0] if len(xs_l) == 1 else np.concatenate(xs_l),
              ys_l[0] if len(ys_l) == 1 else np.concatenate(ys_l),
              ids_l[0] if len(ids_l) == 1 else np.concatenate(ids_l),
              objs[0] if len(objs) == 1 else [a for b in objs for a in b])
        self._nb[key] = nb
        return nb

    def _hearers_bounded(self, speaker, cap: int, radius_eff) -> list:
        """半径 radius_eff(≤ セル幅)+ 最寄り cap 人で列挙する(ベクトル化・決定論)。"""
        ctx = _context(speaker)
        if ctx[0] == "outside":
            return []
        radius = self.radius if radius_eff is None else min(float(radius_eff),
                                                            self.radius)
        # ★セル細分化(`world.perception_cell_m`)。既定 cell_m=0 では ring が **常に 0**
        #   なので、以下は 1 行残らず現行(粗格子・9 セル)と同一の経路を走る。
        ring, _ratio = self._fine_ring(radius)
        cx, cy = self._cell_xy(speaker.x, speaker.y)
        if ring:
            # 局所の混み具合で最終判断(疎な近傍では細格子が純損 = `_FINE_MIN_GAIN` の注記)。
            gain = self.fine_gain
            if gain is None:                 # 既定 = 実装既定(= 現行と完全同一)
                gain = _FINE_MIN_GAIN
            if (self._coarse_load(ctx, cx, cy) * (1.0 - _ratio)) < gain:
                ring = 0
            else:
                cx, cy = (math.floor(speaker.x * self._finv),
                          math.floor(speaker.y * self._finv))
        sx, sy, sid = float(speaker.x), float(speaker.y), int(speaker.id)
        occ = _resolve(self._occ)
        if occ is not None:
            # 遮蔽器が実際に据わっている: ベクトル化を捨ててループで正しさを取る。
            cells = self._fine_cells() if ring else self.cells
            offs = _ring_offsets(ring) if ring else _RING1
            cands = []
            for dx in offs:
                for dy in offs:
                    bucket = cells.get((ctx, cx + dx, cy + dy))
                    if not bucket:
                        continue
                    for other in bucket:
                        if other.id == sid:
                            continue
                        if math.hypot(other.x - sx, other.y - sy) <= radius:
                            if not occ.blocks(speaker, other):
                                cands.append(other)
            return nearest_k(speaker, cands, cap)
        if ring:
            nb = self._fine_neighborhood(ctx, cx, cy, ring)
            if nb is None:
                return []
            return _bounded_pick(nb, sx, sy, sid, radius, cap, False)
        nb = self._neighborhood(ctx, cx, cy)
        if nb is None:
            return []
        return _bounded_pick(nb, sx, sy, sid, radius, cap)

    def _count(self, speaker, occ) -> int:
        """`len(self.hearers(speaker))` と**厳密に同値**な人数(列挙も整列もしない)。

        既定経路(cap=0 / radius_eff=None)専用。返すのは人数だけなので id 昇順の整列も
        結果リストの確保も行わず、半径マスクの合計だけを取る(第150 の観測チャネル用)。
        遮蔽器が据わっているときはループで数える(hearers と同じ順で blocks を呼ぶ)。
        """
        ctx = _context(speaker)
        if ctx[0] == "outside":
            return 0
        cx, cy = self._cell_xy(speaker.x, speaker.y)
        radius = self.radius
        sx, sy, sid = speaker.x, speaker.y, speaker.id
        cells = self.cells
        if occ is not None:
            total = 0
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    bucket = cells.get((ctx, cx + dx, cy + dy))
                    if not bucket:
                        continue
                    for other in bucket:
                        if other.id == sid:
                            continue
                        if math.hypot(other.x - sx, other.y - sy) <= radius:
                            if not occ.blocks(speaker, other):
                                total += 1
            return total
        nb = self._count_arrays(ctx, cx, cy)
        if nb is None:
            return 0
        xs, ys, ids = nb
        lo, hi = _radius_band(radius)
        dxa = xs - float(sx)
        dya = ys - float(sy)
        # ① 外接正方形で粗く落とす(**判定の揺らぎ幅 hi まで広げた**厳密に安全な前フィルタ:
        #    |dx| > hi なら math.hypot も必ず radius を超える)。近傍は 3 セル幅 = 半径の
        #    3 倍なので、これだけで高価な hypot に渡す要素が半分以下になる。
        #    ★第153: `(|dx|<=hi) & (|dy|<=hi)` は `max(|dx|,|dy|) <= hi` と**厳密に等価**
        #      (有限座標では両辺が同じ bool)。近傍全長を舐める回数が 1 本減る。
        #      ここでは自分自身の除外(`ids != sid`)は**外に出さない**: 話者が索引に載って
        #      いない場合(睡眠・範囲外)があるので「最後に 1 引く」が成り立たないため。
        ax = np.abs(dxa)
        np.maximum(ax, np.abs(dya), out=ax)
        box = np.flatnonzero((ax <= hi) & (ids != int(sid)))
        if box.size == 0:
            return 0
        h = np.hypot(dxa[box], dya[box])
        # ② np.hypot と math.hypot は最大 1 ULP ずれうるので、境界の帯 (lo, hi] に入った
        #    要素だけ math.hypot で**裁定**する。帯の外はどちらの実装でも判定が一致する
        #    ことが 1 ULP 保証から従う = hearers と厳密に同じ集合の要素数。
        total = int(np.count_nonzero(h <= lo))
        if total != int(np.count_nonzero(h <= hi)):
            for j in np.flatnonzero((h > lo) & (h <= hi)):
                k = int(box[int(j)])
                if math.hypot(float(dxa[k]), float(dya[k])) <= radius:
                    total += 1
        return total

    def _exists_block(self, xs, ys, ids, beg: int, end: int,
                      fsx: float, fsy: float, isid: int,
                      lo: float, hi: float, radius: float) -> bool:
        """連結配列の [beg, end) に「半径内で自分以外」が 1 つでもあるか(`_count` と同一判定)。

        `_count` の本体(:659-681)を**逐語**でスライスへ移したもの:
          ① 外接正方形 `max(|dx|,|dy|) <= hi` + 自分自身の除外(`hi` は +4 ULP の安全側)
          ② `h <= lo` は両実装(np/math)とも確実に圏内 = 即決
          ③ 帯 (lo, hi] に落ちた要素だけ `math.hypot` で裁定(`_count` と同一の裁定)
        `_count` はこの ②③ の件数を足し上げる。ここは「1 つでもあるか」なので、
        ② が 1 つでもあるか、③ の裁定を 1 つでも通れば True(= 部分和が正)。
        """
        ids_b = ids[beg:end]
        dxa = xs[beg:end] - fsx
        dya = ys[beg:end] - fsy
        ax = np.abs(dxa)
        np.maximum(ax, np.abs(dya), out=ax)
        box = np.flatnonzero((ax <= hi) & (ids_b != isid))
        if box.size == 0:
            return False
        h = np.hypot(dxa[box], dya[box])
        if bool(np.any(h <= lo)):
            return True
        # ここへ来た = `h <= lo` が 1 つも無い。よって `h <= hi` の要素は必ず帯 (lo, hi] に
        # 居る。`_count` が「lo 件数 != hi 件数」で裁定に入るのと同じ門(1 呼び)で、
        # 実データでは事実上ここを通らない = 全長を舐める numpy 呼びが 1 本増えない。
        if not bool(np.any(h <= hi)):
            return False
        for j in np.flatnonzero((h > lo) & (h <= hi)):
            k = int(box[int(j)])
            if math.hypot(float(dxa[k]), float(dya[k])) <= radius:
                return True
        return False

    def _exists(self, speaker, occ) -> bool:
        """`self._count(speaker, occ) > 0` と**厳密に同値**な存在判定(最初の 1 件で打ち切る)。

        既定経路(cap=0 / radius_eff=None)専用 = `hearers()` の返り値が空でないことと同値。
        `_count` との差は「数え上げをやめて即 return する」ことと「見る順番と分割」だけで、
        **判定式は 1 つも変えていない**(`_exists_block` の docstring に逐語の対応):
          - 文脈(`_context`)・自分自身の除外(`ids != sid`)・半径判定(`<= radius`)
          - 外接正方形の前フィルタ(`max(|dx|,|dy|) <= hi`。`_count` と同じ `hi` = +4 ULP)
          - 帯 (lo, hi] に落ちた要素の `math.hypot` 裁定(`_count` と同じ `_radius_band`)
          - 遮蔽器が据わっているときは `hearers` と同じ順で `blocks` を呼ぶループへ後退
        `_count` は「`h <= lo` の件数」+「帯の裁定で通った件数」の合計を返す。ここでは
        その合計が正であること = 「どこかのブロックに 1 つでもある」と同値なので、
        見つかった瞬間に True を返してよい(bool の or は可換・結合的 = 順序も分割も自由)。

        速さの構造(**答えには一切影響しない**):
          ⓪ 疎な近傍(在圏数 ≤ `_EXISTS_LOOP_MAX`)は**素の Python ループ**で見る。
             判定式は `hearers()` の逐語(= 差し替え前の `hearers_of` そのもの)なので、
             夜間・郊外の疎な個体で numpy の固定費を払わされる退行が起こらない。
          ① 密な近傍では中心セルの先頭 `_EXISTS_PROBE_MAX` 人だけをループで先読みする。
             ほぼ必ずここで当たるので、**9 セル連結も numpy も触らずに**決着する。
          ② 外れたら `_count` と**同じ連結キャッシュ**(`_count_arrays` = `_nc`)を
             **全長 1 パス**で見る。numpy の呼び方も配列も `_count` と同じ形で、
             合計を取らずに `np.any` で抜けるだけなので、**どんな配置でも `_count` を
             超えない**(ブロックに刻むと numpy 呼び出しの固定費が積み上がって、
             当たりの無い近傍で逆に遅くなる。micro bench で実測して 1 パスにした)。
          ①で見た要素を②が見直すのは無駄だが高々 32 人ぶんで、正しさには無関係。
        """
        ctx = _context(speaker)
        if ctx[0] == "outside":
            return False
        cx, cy = self._cell_xy(speaker.x, speaker.y)
        radius = self.radius
        sx, sy, sid = speaker.x, speaker.y, speaker.id
        cells = self.cells
        if occ is not None:
            # 遮蔽器が据わっている: `_count` と同じループで、最初の 1 件で打ち切る。
            for dx, dy in _EXISTS_ORDER:
                bucket = cells.get((ctx, cx + dx, cy + dy))
                if not bucket:
                    continue
                for other in bucket:
                    if other.id == sid:
                        continue
                    if math.hypot(other.x - sx, other.y - sy) <= radius:
                        if not occ.blocks(speaker, other):
                            return True
            return False
        # ---- ⓪ 疎な近傍は素の Python ループ(= `hearers()` と逐語同一の判定)------------
        if self._coarse_load(ctx, cx, cy) <= _EXISTS_LOOP_MAX:
            for dx, dy in _EXISTS_ORDER:
                bucket = cells.get((ctx, cx + dx, cy + dy))
                if not bucket:
                    continue
                for other in bucket:
                    if other.id == sid:
                        continue
                    if math.hypot(other.x - sx, other.y - sy) <= radius:
                        return True
            return False
        # ---- ① 中心セルの先読み(先頭 32 人だけ。密なセルはここでほぼ必ず当たる)---------
        bucket = cells.get((ctx, cx, cy))
        if bucket:
            n = len(bucket)
            for k in range(n if n < _EXISTS_PROBE_MAX else _EXISTS_PROBE_MAX):
                other = bucket[k]
                if other.id == sid:
                    continue
                if math.hypot(other.x - sx, other.y - sy) <= radius:
                    return True
        # ---- ② 近傍 9 セル連結(`_count` と同じキャッシュ)を全長 1 パス ------------------
        lo, hi = _radius_band(radius)
        fsx, fsy, isid = float(sx), float(sy), int(sid)
        nb = self._count_arrays(ctx, cx, cy)
        if nb is None:
            return False
        xs, ys, ids = nb
        return self._exists_block(xs, ys, ids, 0, int(xs.shape[0]),
                                  fsx, fsy, isid, lo, hi, radius)

    def _count_arrays(self, ctx: tuple, cx: int, cy: int):
        """近傍 9 セルを連結した (xs, ys, ids) を返す(遅延構築・セル単位で共有)。

        `_neighborhood` の軽量版。人数を数えるのに個体オブジェクトは要らないので
        平坦リストを作らない(= その分の確保と走査がまるごと消える)。"""
        key = (ctx, cx, cy)
        if key in self._nc:
            return self._nc[key]
        cells = self.cells
        xs_l: list = []
        ys_l: list = []
        ids_l: list = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                ck = (ctx, cx + dx, cy + dy)
                bucket = cells.get(ck)
                if not bucket:
                    continue
                xa, ya, ia = self._cell_arrays(ck, bucket)
                xs_l.append(xa)
                ys_l.append(ya)
                ids_l.append(ia)
        if not xs_l:
            self._nc[key] = None
            return None
        nb = (xs_l[0] if len(xs_l) == 1 else np.concatenate(xs_l),
              ys_l[0] if len(ys_l) == 1 else np.concatenate(ys_l),
              ids_l[0] if len(ids_l) == 1 else np.concatenate(ids_l))
        self._nc[key] = nb
        return nb

    def hearers(self, speaker, cap: int = 0, radius_eff=None) -> list:
        # ★既定(cap=0 かつ radius_eff=None)ではこの分岐に入らず、下の従来コードが
        #   1 行も変わらずそのまま走る(golden L1 バイト一致の構造的保証)。
        if cap > 0 or radius_eff is not None:
            return self._hearers_bounded(speaker, int(cap), radius_eff)
        ctx = _context(speaker)
        if ctx[0] == "outside":
            return []
        cx, cy = self._cell_xy(speaker.x, speaker.y)
        radius = self.radius
        sx, sy, sid = speaker.x, speaker.y, speaker.id
        cells = self.cells
        occ = _resolve(self._occ)              # None=遮蔽なし=従来と完全同一
        result = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                bucket = cells.get((ctx, cx + dx, cy + dy))
                if not bucket:
                    continue
                for other in bucket:
                    if other.id == sid:
                        continue
                    if math.hypot(other.x - sx, other.y - sy) <= radius:
                        if occ is None or not occ.blocks(speaker, other):
                            result.append(other)
        return sorted(result, key=lambda a: a.id)


def salience_gate(items: list, scores: list, k: int) -> list:
    """同時知覚 item を上位 K 件に絞る注意の容量制約ゲート(Cowan 2001: 焦点は約4チャンク)。

    items と scores は同順の並び。scores は上流(factors 層)が算出した**不透明な float**で、
    本層はその意味を知らずソートするだけ(価値名・valence 語をここに晒さない=no-fingerprint)。

    - k<=0(=∞/無効)または item 数<=K のとき: 入力を**順序そのまま**返す(既定=従来と同一集合・同一順序)。
    - k>0 のとき: score 上位 K を選び、**元の並び順を保って**返す(決定論。同点は先着=index 昇順)。
    """
    n = len(items)
    if k <= 0 or n <= k:
        return list(items)
    top = sorted(range(n), key=lambda i: (-scores[i], i))[:k]
    keep = set(top)
    return [it for i, it in enumerate(items) if i in keep]


def cell_m_of(sim) -> float:
    """`world.perception_cell_m` を遅延解決して sim にキャッシュする(既定 0.0 = 現行同一)。

    キャッシュ属性 `_percept_cell_m` は L1/L2/乱数のどこにも現れない(`speech_cfg_of` と同型)。"""
    v = getattr(sim, "_percept_cell_m", None)
    if v is None:
        try:
            world = sim.cfg.get("world", {}) or {}
            v = float(world.get("perception_cell_m", 0.0) or 0.0)
        except Exception:                               # noqa: BLE001 (旧 config 互換)
            v = 0.0
        sim._percept_cell_m = v
    return v


_UNSET = object()


def fine_gate_of(sim):
    """`world.perception_fine_gate` を遅延解決して sim にキャッシュする(既定 2000 = 現行同一)。

    返すのは **None(= 実装既定 `_FINE_MIN_GAIN` に従う)** か明示の float。
    出荷時の既定値(`_FINE_GATE_CONF_DEFAULT`)がそのまま書いてあるときは None を返す =
    「既定値を明示したラン」と「キーを書かないラン」が構造的に同値。
    キャッシュ属性 `_percept_fine_gate` は L1/L2/乱数のどこにも現れない(`cell_m_of` と同型)。
    値は「細格子へ回すのに要求する候補削減の絶対数(要素数)」。**返り値には一切影響しない**
    (粗格子/細格子のどちらを通っても同じ集合・同じ順序 = 選抜規則が入力順に依らない)。"""
    v = getattr(sim, "_percept_fine_gate", _UNSET)
    if v is _UNSET:
        try:
            world = sim.cfg.get("world", {}) or {}
            raw = world.get("perception_fine_gate", None)
            v = None if raw is None else float(raw)
            if v == _FINE_GATE_CONF_DEFAULT:
                v = None                                # = 実装既定に従う(現行と完全同一)
        except Exception:                               # noqa: BLE001 (旧 config 互換)
            v = None
        sim._percept_fine_gate = v
    return v


def build_index(agents, radius_m: float, occluder=None,
                cell_m: float = 0.0, fine_gain=None) -> PerceptIndex:
    """位置が確定した時点の全 agent から空間索引を1回だけ構築する。

    occluder(既定 None)を渡すと索引経由の hearers に視線遮蔽を効かせる。None のときは
    据え付けの既定遮蔽器(install_occluder)へ後退し、それも無ければ遮蔽なし=従来同一。

    cell_m(既定 0.0)= 細格子のセル寸法 [m]。0 のときは細格子を一切持たない
    (= 現行と完全に同一の索引)。> 0 でも構築は**最初の細格子クエリまで遅延**するので、
    有界経路を通らないランでは追加コストが 1 命令も発生しない。

    fine_gain(既定 None = 実装既定 `_FINE_MIN_GAIN` = 2000)= 細格子へ回す門のしきい値
    (`world.perception_fine_gate`)。**返り値には一切影響しない**(どちらの格子を通っても
    同じ集合・同じ順序)= 速さだけのつまみ。cell_m=0 のランでは 1 度も読まれない。
    """
    idx = PerceptIndex(radius_m, occluder=occluder, cell_m=cell_m,
                       fine_gain=fine_gain)
    for a in agents:
        idx.add(a)
    return idx


def hearers_of(speaker, agents_or_index, radius_m: float, occluder=None,
               *, cap: int = 0, radius_eff=None) -> list:
    """speaker の発話が聞こえるエージェント(agent_id 昇順=決定論)。

    第2引数は agent の反復可能列(従来=全対全 live 走査)または PerceptIndex
    (空間索引=近傍9セルのみ走査)。どちらでも返り値は一致する。

    occluder(既定 None)= 視線遮蔽器。None のときは索引に据えた/install_occluder した
    既定へ後退し、遮蔽器が一切無ければ従来と完全にバイト一致(=遮蔽なし)。

    cap(既定 0=無制限)/ radius_eff(既定 None=従来半径)= 有界化の 2 引数
    (docs/plans/hearer-cap-plan.md §2)。**両方とも既定なら下の従来コードがそのまま走る**。
    radius_eff は索引経路では索引のセル幅へクランプされる(9 セル走査の不変条件)。
    """
    bounded = cap > 0 or radius_eff is not None
    if isinstance(agents_or_index, PerceptIndex):
        if occluder is None:
            return (agents_or_index.hearers(speaker, cap, radius_eff) if bounded
                    else agents_or_index.hearers(speaker))
        # 明示の occluder が索引の据え付けより優先(呼び出し側の意図を尊重)。
        saved = agents_or_index._occ
        agents_or_index._occ = occluder
        try:
            return (agents_or_index.hearers(speaker, cap, radius_eff) if bounded
                    else agents_or_index.hearers(speaker))
        finally:
            agents_or_index._occ = saved
    ctx = _context(speaker)
    occ = _resolve(occluder)                   # None=遮蔽なし=従来と完全同一
    radius = radius_m if radius_eff is None else float(radius_eff)
    result = []
    for other in agents_or_index:
        if other.id == speaker.id or other.sleeping or _context(other) != ctx:
            continue
        if math.hypot(other.x - speaker.x, other.y - speaker.y) <= radius:
            if occ is None or not occ.blocks(speaker, other):
                result.append(other)
    if bounded:                                # 最寄り K 選抜(S15 と同一規則)
        return nearest_k(speaker, result, cap)
    return sorted(result, key=lambda a: a.id)


def count_hearers(speaker, agents_or_index, radius_m: float, occluder=None) -> int:
    """**`len(hearers_of(speaker, agents_or_index, radius_m, occluder))` と同値**な人数。

    「周囲の声が届く人数」しか要らない呼び手(第80 観測チャンネル `ext.encounter`)用。
    集合は `hearers_of` と厳密に同じ(同じ `_context` 判定・同じ半径判定式・同じ遮蔽適用)
    だが、**リスト構築も id 昇順の整列も行わない**。

    有界化の 2 引数(cap / radius_eff)は**受け取らない**: このチャンネルの意味は
    「物理的に声が届く人数」であって、聞き手 cap(第141 S15 / 第149)や声の段階
    (`speech_levels`)で絞られた実際の聞き手の数ではない。現行 semantics(= cap も
    radius_eff も掛けない `hearers_of` の長さ)をそのまま保つ = 観測値はバイト一致。

    第2引数は `PerceptIndex`(近傍 9 セル・ベクトル化)または agent の反復可能列
    (全対全 live 走査)。どちらでも同じ数を返す。
    """
    if isinstance(agents_or_index, PerceptIndex):
        occ = _resolve(occluder if occluder is not None else agents_or_index._occ)
        return agents_or_index._count(speaker, occ)
    ctx = _context(speaker)
    occ = _resolve(occluder)                   # None=遮蔽なし=従来と完全同一
    sx, sy, sid = speaker.x, speaker.y, speaker.id
    radius = radius_m
    total = 0
    for other in agents_or_index:
        if other.id == sid or other.sleeping or _context(other) != ctx:
            continue
        if math.hypot(other.x - sx, other.y - sy) <= radius:
            if occ is None or not occ.blocks(speaker, other):
                total += 1
    return total


def has_hearers(speaker, agents_or_index, radius_m: float, occluder=None) -> bool:
    """**`count_hearers(speaker, agents_or_index, radius_m, occluder) > 0` と同値**な bool。

    「同席者が居るか否か」の 1 ビットしか要らない呼び手(`scheduler._decide_g` の
    `has_company`・`_phase_drive` の face 判定)用。人数も列挙も整列もせず、
    **最初の 1 件が見つかった時点で走査を打ち切る**(docs/plans/step-time-audit.md A2)。

    引数面は `count_hearers` と同一(cap / radius_eff は**受け取らない**)。第2引数は
    `PerceptIndex`(近傍 9 セル・`_exists` の 3 段構え)または agent の反復可能列
    (全対全 live 走査)。どちらでも同じ bool を返す。
    """
    if isinstance(agents_or_index, PerceptIndex):
        occ = _resolve(occluder if occluder is not None else agents_or_index._occ)
        return agents_or_index._exists(speaker, occ)
    ctx = _context(speaker)
    occ = _resolve(occluder)                   # None=遮蔽なし=従来と完全同一
    sx, sy, sid = speaker.x, speaker.y, speaker.id
    radius = radius_m
    for other in agents_or_index:              # ★ count_hearers と同じ判定・同じ順序
        if other.id == sid or other.sleeping or _context(other) != ctx:
            continue
        if math.hypot(other.x - sx, other.y - sy) <= radius:
            if occ is None or not occ.blocks(speaker, other):
                return True                    # 1 件見つかった時点で打ち切る
    return False
