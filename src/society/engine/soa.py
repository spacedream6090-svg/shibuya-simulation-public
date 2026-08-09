"""SoA(Struct-of-Arrays)アクタ基盤 = P5「アクタモデル移行」の**土台のみ**。

本モジュールは **エンジンに一切配線されていない**(import されるだけで何も起きない・
conf ゲートも持たない=ゲートすべき挙動が存在しないため)。配線は後続の波で行う。

--------------------------------------------------------------------------------
移行契約(この土台が守る約束。将来のエンジン層はこれに乗る)
--------------------------------------------------------------------------------
1. **配列が真実 (array = truth)** — Covasim の言い方で「エージェントとは状態配列群を
   貫く 1 枚のスライスである」。`RowView` はオブジェクト風の**窓**にすぎず、属性の
   読み書きは必ず配列へ**素通し**される。窓は値を持たない(コピーを持たない)。
2. **昇格/降格は step 境界でのみ** — `hydrate` / `dehydrate` は将来のエンジン層が
   step の切れ目で呼ぶ。ホットループの途中で呼ぶことは想定しない(Reynolds 1997 /
   DAC の LOD 昇格と同じ規律)。連鎖昇格の抑止は**上位層の仕事**でここではやらない。
   ただし上位層が定量上限をかけられるよう、要求集合セマンティクスの
   `select_promotions()`(決定論的な granted/deferred 分割)を提供する。
3. **RngHub 法の遵守** — 既存 `src/society/rng.py` の掟「どのエージェントの乱数も
   他者の実行順に依存しない」を、**ベクトル化可能な形**で実装したものが
   `PhiloxDraws`。カウンタベース(Random123/Philox)なので、
   `uniform(ch, step, ids)`(配列一括)と `uniform1(ch, step, id)`(単体)は
   **ビット単位で同一**であり、どの順に・どの部分集合を引いても値は変わらない。
   種ドメインは `(master_seed, "soa", channel)` の sha256 に閉じており、
   `RngHub`(SeedSequence + PCG64)とは**アルゴリズムからして別系統**なので衝突しない。
4. **行は暗黙には詰めない (no implicit compaction)** — Bevy/ECS 流の世代付き ID。
   `free()` は世代を上げて席を「死」にするだけで、**その席は再利用されない**。
   再利用を許すのは明示的な `compact()` のみ(= 決定論に影響する操作)。
   古いハンドルでのアクセスは必ず `StaleHandleError`。
5. **書き込みは 1 点でまとめて流す (AMBER / Unity ECB)** — `PendingWrites` は
   いつでも受け付け、`flush()` が **(slot, column, seq) の昇順**という
   挿入順非依存の決定論的順序で適用する。同一 (id, column) への競合は
   明示的な連番による last-writer-wins。

--------------------------------------------------------------------------------
明示的な対象外 (out of scope)
--------------------------------------------------------------------------------
* **cold 側**(`MemoryStore` / persona テキスト / route リスト / dict・set 状態)は
  オブジェクト側に残す。本モジュールは hot な**固定幅スカラー**のみを扱う。
* エンジンへの配線・`Agent` クラスの置換・`scheduler.py` の書き換え。
* 連鎖昇格の抑止ポリシー、LOD の判定、conf ゲート。
* スレッド/プロセス並列(本モジュールは**スレッドを一切使わない**。
  `tests/test_soa.py` が AST で機械的に固定している)。

--------------------------------------------------------------------------------
NEP 19(numpy の乱数ストリーム互換ポリシー)について
--------------------------------------------------------------------------------
NEP 19 が安定を約束するのは numpy 自身の BitGenerator 系列であって、こちらの
使い方ではない。そこで本モジュールは **Philox4x64-10 の全単射を numpy 配列演算で
自前実装**している。依存するのは numpy の**整数演算(uint64 の巻き上がり)**だけで、
乱数ストリーム方針には依存しない = numpy 側の方針変更に対して原理的に免疫がある。

そのうえで二重に固定している:
  (a) `tests/test_soa.py` のゴールデン literal(固定キーの先頭 8 draw)= 自前実装の凍結。
  (b) `uniform1()` は **numpy の C 実装 `np.random.Philox` を通る別経路**で、
      `uniform()` は自前ベクトル実装。両者のビット一致テストが独立な相互検証になる。
      仮に numpy 側が変わればこのテストが**大声で**落ちるが、本番ストリーム
      ((a) で凍結された自前実装)は動かない。

注意(実測で踏んだ numpy の罠): `np.random.Philox(counter=[大きな Python int, ...])` は
**リストを float64 経由で変換して下位ビットを落とす**。必ず
`np.array(..., dtype=np.uint64)` を渡すこと。本モジュールは内部で常にそうしている。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

__all__ = [
    "SLOT_BITS", "GEN_BITS", "MAX_SLOTS", "MAX_GEN", "NULL_ID",
    "SchemaError", "StaleHandleError", "CapacityError",
    "Column", "ActorTable", "RowView", "PendingWrites", "FlushReport",
    "PhiloxDraws", "select_promotions", "pack_id", "unpack_id",
]


# =============================================================================
# 1 世代付き ID(packed int64)
# =============================================================================
# レイアウト(下位→上位):
#     bit  0..39  slot        (40bit = 最大 1,099,511,627,776 席)
#     bit 40..62  generation  (23bit = 1..8,388,607)
#     bit 63      未使用(常に 0 = 符号ビット)
#
# ★ブリーフの (slot:40, gen:24) から **gen を 1bit 削って 23bit** にしている。理由:
#   40+24=64bit は int64 の符号ビットを踏み、ID が負になり得る。負の ID は
#   (a) `-1` を NULL 番兵に使えない (b) 昇順ソートが席順と一致しない
#   (c) uint64 との混在で numpy の型昇格事故を招く、の 3 つを同時に壊す。
#   uint64 を採らないのは、numpy で uint64 と Python int/int64 を混ぜると
#   float64 へ落ちる古典的な罠があるため(int64 なら常に安全)。
#   gen 23bit = 同一席を 838 万回まで使い回せる。使い切った席は**永久退役**する
#   (再利用しない)ので、世代の巻き戻りによる ID 衝突は原理的に起きない。
SLOT_BITS = 40
GEN_BITS = 23
MAX_SLOTS = 1 << SLOT_BITS
MAX_GEN = (1 << GEN_BITS) - 1
_SLOT_MASK = np.int64(MAX_SLOTS - 1)
_SLOT_SHIFT = np.int64(SLOT_BITS)

#: 無効ハンドルの番兵。有効 ID は必ず >= (1 << 40) なので衝突しない。
NULL_ID = -1


def pack_id(slot, gen):
    """(slot, gen) → packed int64。スカラーでも配列でも同じ式で通る。"""
    s = np.asarray(slot, dtype=np.int64)
    g = np.asarray(gen, dtype=np.int64)
    if np.any(s < 0) or np.any(s >= MAX_SLOTS):
        raise CapacityError(f"slot が 40bit を溢れた: max={MAX_SLOTS}")
    if np.any(g < 1) or np.any(g > MAX_GEN):
        raise CapacityError(f"generation が 23bit を溢れた: max={MAX_GEN}")
    out = (g << _SLOT_SHIFT) | s
    return out if out.ndim else np.int64(out)


def unpack_id(ids) -> tuple[np.ndarray, np.ndarray]:
    """packed int64 → (slot, gen)。負値(NULL_ID など)はそのまま流れる。"""
    a = np.asarray(ids, dtype=np.int64)
    return a & _SLOT_MASK, a >> _SLOT_SHIFT


# =============================================================================
# 2 例外
# =============================================================================
class SchemaError(ValueError):
    """スキーマ定義そのものが不正(未知の列・プラットフォーム依存 dtype 等)。"""


class StaleHandleError(LookupError):
    """既に free された(= 世代が古い)ハンドルでのアクセス。"""


class CapacityError(RuntimeError):
    """席数/世代がビット幅を溢れた。"""


# =============================================================================
# 3 スキーマ
# =============================================================================
# 明示幅の dtype のみ許す。`int` / `"int"` / `np.int_` / `np.intp` は
# プラットフォームで幅が変わる(Windows と Linux で int32/int64 が食い違う)ため
# **決定論とチェックポイント互換の敵**であり、ここで問答無用で弾く。
_ALLOWED_DTYPE_NAMES = frozenset({
    "bool",
    "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
    "float16", "float32", "float64",
})

# 文字列で来たときに弾く曖昧エイリアス(np.dtype() を通すと黙って解決してしまう)
_AMBIGUOUS_STRINGS = frozenset({
    "int", "uint", "intp", "uintp", "long", "ulong", "longlong", "ulonglong",
    "float", "double", "single", "bool_", "l", "L", "q", "Q", "p", "P", "n", "N",
    "i", "u", "f", "d", "b", "?",
})
_AMBIGUOUS_OBJECTS = (int, float, bool, complex)


def _resolve_dtype(col: str, spec: Any) -> np.dtype:
    if isinstance(spec, str):
        if spec in _AMBIGUOUS_STRINGS:
            raise SchemaError(
                f"列 {col!r}: dtype {spec!r} は幅がプラットフォーム依存/曖昧。"
                f"明示幅で書くこと(例 'int64' / 'float32')")
    elif any(spec is o for o in _AMBIGUOUS_OBJECTS):
        raise SchemaError(
            f"列 {col!r}: Python 組み込み型 {spec!r} は dtype として受け付けない。"
            f"明示幅で書くこと(例 'int64' / 'float32')")
    elif spec in (np.int_, np.uint, np.intp, np.uintp):
        raise SchemaError(
            f"列 {col!r}: {spec!r} は幅がプラットフォーム依存。明示幅で書くこと")
    try:
        dt = np.dtype(spec)
    except TypeError as exc:                       # pragma: no cover - 異常系
        raise SchemaError(f"列 {col!r}: dtype {spec!r} を解決できない") from exc
    if dt.name not in _ALLOWED_DTYPE_NAMES:
        raise SchemaError(
            f"列 {col!r}: dtype {dt.name!r} は hot 列に許可されていない。"
            f"許可={sorted(_ALLOWED_DTYPE_NAMES)}(可変長・object は cold 側の担当)")
    return dt


@dataclass(frozen=True)
class Column:
    """1 列の宣言。`index` はスキーマ宣言順 = flush のタイブレーク順序。"""
    name: str
    dtype: np.dtype
    default: Any
    index: int


# =============================================================================
# 4 ActorTable
# =============================================================================
class ActorTable:
    """世代付き ID を持つ SoA テーブル。列は 1 本の 1 次元 ndarray。

    使い方::

        t = ActorTable([("money", "float32", 0.0), ("node", "int32", -1)])
        ids = t.alloc(3)
        t.col("money")[t.active] += 100.0        # ベクトル化更新(配列が真実)
        t.view(ids[0]).money                     # 窓ごしの読み(素通し)

    席(slot)と ID の関係:
      * `alloc()` は「再利用可能席(昇順)→ 新規席(昇順)」の順に配る = 決定論。
      * `free()` は世代を +1 して席を死なせるだけ。**席は再利用されない**。
      * `compact()` を明示的に呼んだときだけ、死んだ席が再利用プールへ戻る。
        これは「同じ手順でも compact の有無で以降の ID 割り当てが変わる」という
        意味で**決定論に影響する操作**であり、暗黙には絶対に起きない。
      * 生きている行は**決して移動しない**。したがって ID は成長・解放・再確保を
        またいで安定する。

    ndarray ビューの寿命(重要):
      `col()` が返すのはライブなビューであり、**次に容量が伸びるまで**有効。
      容量成長は配列の再確保を伴うので、古いビューは旧バッファを指したままになる。
      ホットループの前に `reserve()` で確保しておくか、`alloc()` 後に取り直すこと。
      (ID と席番号は成長しても一切動かない = ここが ECS の肝)
    """

    #: 初回確保の最小容量。成長は 2 倍ずつ(償却 O(1)・完全に決定論的)。
    MIN_CAPACITY = 8

    def __init__(self, schema: Sequence[tuple[str, Any, Any]], *, capacity: int = 0):
        if not schema:
            raise SchemaError("スキーマが空")
        cols: dict[str, Column] = {}
        for i, entry in enumerate(schema):
            if len(entry) != 3:
                raise SchemaError(f"スキーマ要素は (name, dtype, default) の 3 つ組: {entry!r}")
            name, spec, default = entry
            if not isinstance(name, str) or not name:
                raise SchemaError(f"列名が不正: {name!r}")
            if name in cols:
                raise SchemaError(f"列名が重複: {name!r}")
            if name.startswith("_"):
                raise SchemaError(f"列名の先頭 '_' は内部予約: {name!r}")
            cols[name] = Column(name, _resolve_dtype(name, spec), default, i)
        self._columns: dict[str, Column] = cols
        self._order: tuple[str, ...] = tuple(cols)

        self._capacity = 0
        self._n_slots = 0                      # 高水位(= 一度でも配った席の数)
        self._arrays: dict[str, np.ndarray] = {
            n: np.zeros(0, dtype=c.dtype) for n, c in cols.items()}
        self._gen = np.zeros(0, dtype=np.int32)
        self._alive = np.zeros(0, dtype=bool)
        self._free: list[int] = []             # compact() で戻された再利用可能席(昇順)
        self._dead: list[int] = []             # free 済みだが未 compact の席(昇順)
        self._n_live = 0
        self._retired = 0                      # 世代を使い切って永久退役した席数
        self._ensure_capacity(max(int(capacity), 0))

    # ---------------------------------------------------------------- schema
    @property
    def schema(self) -> tuple[Column, ...]:
        return tuple(self._columns[n] for n in self._order)

    @property
    def column_names(self) -> tuple[str, ...]:
        return self._order

    def has_column(self, name: str) -> bool:
        return name in self._columns

    def _column(self, name: str) -> Column:
        try:
            return self._columns[name]
        except KeyError:
            raise SchemaError(
                f"未知の列 {name!r}(既知={list(self._order)})") from None

    # ---------------------------------------------------------------- sizing
    def __len__(self) -> int:
        """生存行数。"""
        return self._n_live

    @property
    def capacity(self) -> int:
        """確保済み配列長。`col()` のビューはここが伸びると無効になる。"""
        return self._capacity

    @property
    def n_slots(self) -> int:
        """高水位(死んだ席も含む、席番号の上限)。`col()` / `active` の長さ。"""
        return self._n_slots

    @property
    def reclaimable(self) -> int:
        """`compact()` で再利用プールへ戻せる死んだ席の数。"""
        return len(self._dead)

    @property
    def retired(self) -> int:
        """世代 23bit を使い切って永久退役した席数(通常 0)。"""
        return self._retired

    def _growth_target(self, need: int) -> int:
        cap = self._capacity
        if cap >= need:
            return cap
        cap = max(cap, self.MIN_CAPACITY)
        while cap < need:
            cap *= 2                              # 2 倍成長 = 償却 O(1)・決定論
        return cap

    def _ensure_capacity(self, need: int) -> None:
        if need <= self._capacity:
            return
        if need > MAX_SLOTS:
            raise CapacityError(f"席数が 40bit を超える: need={need}")
        new_cap = self._growth_target(need)
        for name, col in self._columns.items():
            arr = np.full(new_cap, col.default, dtype=col.dtype)
            old = self._arrays.get(name)
            if old is not None:
                arr[: old.shape[0]] = old
            self._arrays[name] = arr
        gen = np.zeros(new_cap, dtype=np.int32)
        gen[: self._gen.shape[0]] = self._gen
        self._gen = gen
        alive = np.zeros(new_cap, dtype=bool)
        alive[: self._alive.shape[0]] = self._alive
        self._alive = alive
        self._capacity = new_cap

    def reserve(self, n: int) -> None:
        """少なくとも n 席ぶんの容量を先に確保する(ビュー無効化を前もって済ませる)。"""
        self._ensure_capacity(int(n))

    # ---------------------------------------------------------------- access
    def col(self, name: str) -> np.ndarray:
        """列の**ライブビュー** `[0:n_slots)` を返す。書けば配列に通る。"""
        self._column(name)
        return self._arrays[name][: self._n_slots]

    def col_full(self, name: str) -> np.ndarray:
        """列の全容量ぶんのビュー(通常は `col()` を使う)。"""
        self._column(name)
        return self._arrays[name]

    @property
    def active(self) -> np.ndarray:
        """`[0:n_slots)` の生存マスク(bool のライブビュー)。"""
        return self._alive[: self._n_slots]

    @property
    def generations(self) -> np.ndarray:
        """`[0:n_slots)` の世代(int32 のライブビュー)。"""
        return self._gen[: self._n_slots]

    @property
    def ids(self) -> np.ndarray:
        """生存行の packed ID を**席の昇順**で返す(新規配列)。"""
        slots = np.flatnonzero(self._alive[: self._n_slots])
        return pack_id(slots, self._gen[slots])

    def id_of_slot(self, slot: int) -> int:
        """席番号 → 現在の packed ID(死んでいれば `NULL_ID`)。"""
        s = int(slot)
        if s < 0 or s >= self._n_slots or not bool(self._alive[s]):
            return NULL_ID
        return int(pack_id(s, self._gen[s]))

    # ---------------------------------------------------------------- lifetime
    def alloc(self, n: int = 1) -> np.ndarray:
        """n 席を確保して packed ID の int64 配列を返す。

        配る順序は決定論的に「再利用可能席(昇順)→ 新規席(昇順)」。
        再利用席は**全列を default へ戻してから**渡す(前の住人の残骸を持ち越さない)。
        """
        n = int(n)
        if n < 0:
            raise ValueError("alloc(n): n は非負")
        if n == 0:
            return np.zeros(0, dtype=np.int64)

        reuse_n = min(n, len(self._free))
        reuse = self._free[:reuse_n]
        del self._free[:reuse_n]
        fresh_n = n - reuse_n
        first_fresh = self._n_slots
        if fresh_n:
            self._ensure_capacity(first_fresh + fresh_n)
        fresh = list(range(first_fresh, first_fresh + fresh_n))
        self._n_slots = first_fresh + fresh_n

        slots = np.asarray(reuse + fresh, dtype=np.int64)
        if reuse_n:
            ridx = np.asarray(reuse, dtype=np.int64)
            for name, col in self._columns.items():
                self._arrays[name][ridx] = col.default
        # 新規席は gen 0 → 1、再利用席は free() 時点で既に +1 済み。
        fresh_mask = np.zeros(slots.shape[0], dtype=bool)
        fresh_mask[reuse_n:] = True
        if fresh_n:
            self._gen[slots[fresh_mask]] = 1
        self._alive[slots] = True
        self._n_live += n
        return pack_id(slots, self._gen[slots])

    def free(self, ids) -> np.ndarray:
        """ID を無効化する。世代を +1 して席を死なせるだけで、**席は再利用しない**。

        戻り値 = 実際に解放された席番号(昇順)。既に死んでいる/古い世代の ID は
        `StaleHandleError`。行の値は消さない(死んだ行は `active` で除外して読める)。
        """
        arr = np.atleast_1d(np.asarray(ids, dtype=np.int64))
        if arr.size == 0:
            return np.zeros(0, dtype=np.int64)
        bad = ~self.valid(arr)
        if np.any(bad):
            raise StaleHandleError(
                f"free: 無効なハンドル {arr[bad][:8].tolist()}"
                f"(計 {int(bad.sum())} 件)")
        slots = np.unique(arr & _SLOT_MASK)        # unique = 二重 free を吸収
        self._alive[slots] = False
        self._n_live -= int(slots.shape[0])
        new_gen = self._gen[slots] + 1
        exhausted = new_gen > MAX_GEN
        self._gen[slots] = np.where(exhausted, self._gen[slots], new_gen)
        keep = slots[~exhausted]
        self._retired += int(exhausted.sum())
        self._dead.extend(int(s) for s in keep)
        self._dead.sort()
        return slots

    def compact(self) -> int:
        """★決定論に影響する明示操作: 死んだ席を再利用プールへ戻す。

        戻した席数を返す。**生きている行は 1 行も動かさない**ので、生存 ID は
        すべて有効なまま(物理的な行詰めをしないのはこのため。行を詰めると
        オブジェクト側が持っているハンドルを一斉に無効化してしまい、
        「ID は安定」という契約と両立しない)。

        これを呼ぶと以降の `alloc()` が返す席番号が変わる = **同じシナリオでも
        compact の有無で ID 列が変わる**。したがって暗黙には絶対に呼ばれない。
        """
        n = len(self._dead)
        if n:
            self._free.extend(self._dead)
            self._free.sort()
            self._dead.clear()
        return n

    # ---------------------------------------------------------------- validity
    def valid(self, ids) -> np.ndarray:
        """ハンドルの有効性 bool マスク(例外を投げない)。スカラーでも配列でも可。"""
        arr = np.asarray(ids, dtype=np.int64)
        if arr.ndim == 0:                          # スカラーは純 Python の速い経路へ
            return np.bool_(self.valid_one(int(arr)))
        a = arr
        slot = a & _SLOT_MASK
        gen = a >> _SLOT_SHIFT
        ok = (a >= 0) & (slot < self._n_slots)
        if np.any(ok):
            s = np.where(ok, slot, 0)
            ok &= self._alive[s] & (self._gen[s].astype(np.int64) == gen)
        return ok

    def slots_of(self, ids) -> np.ndarray:
        """ハンドル → 席番号(int64 配列)。1 つでも無効なら `StaleHandleError`。"""
        arr = np.atleast_1d(np.asarray(ids, dtype=np.int64))
        bad = ~self.valid(arr)
        if np.any(bad):
            raise StaleHandleError(
                f"stale/未知のハンドル {arr[bad][:8].tolist()}(計 {int(bad.sum())} 件)")
        return arr & _SLOT_MASK

    def valid_one(self, id_: int) -> bool:
        """単体版 `valid`。**純 Python 経路**(numpy のスカラー往復は 1 件あたり
        数 µs かかるため、窓/昇降格のホットパスはここを通す)。"""
        i = int(id_)
        if i < 0:
            return False
        s = i & (MAX_SLOTS - 1)
        if s >= self._n_slots or not self._alive[s]:
            return False
        return int(self._gen[s]) == (i >> SLOT_BITS)

    def slot_of(self, id_: int) -> int:
        """単体版 `slots_of`(純 Python 経路)。"""
        i = int(id_)
        s = i & (MAX_SLOTS - 1)
        if i < 0 or s >= self._n_slots or not self._alive[s] \
                or int(self._gen[s]) != (i >> SLOT_BITS):
            raise StaleHandleError(f"stale/未知のハンドル {i}(slot={s})")
        return s

    # ---------------------------------------------------------------- get/set
    def get(self, name: str, ids) -> np.ndarray:
        """列 name をハンドル配列で gather した**コピー**。"""
        self._column(name)
        return self._arrays[name][self.slots_of(ids)]

    def set(self, name: str, ids, values) -> None:
        """列 name へハンドル配列で scatter。重複 ID があると numpy 規則で最後が勝つ
        (= 順序依存)ので、**競合し得る書き込みは `PendingWrites` を通すこと**。"""
        col = self._column(name)
        slots = self.slots_of(ids)
        self._arrays[name][slots] = np.asarray(values, dtype=col.dtype)

    # ---------------------------------------------------------------- views
    def view(self, id_: int) -> "RowView":
        """1 行の窓(`RowView`)。値は持たず、毎アクセスで配列へ素通しする。"""
        rv = RowView(self, int(id_))
        rv._require()                              # 生成時に一度だけ検証
        return rv

    def rows(self) -> Iterator["RowView"]:
        """生存行の窓を席の昇順で返す(決定論的順序)。"""
        for i in self.ids.tolist():
            yield RowView(self, int(i))

    # ---------------------------------------------------------------- hydrate
    def hydrate(self, id_: int, *, native: bool = False) -> dict[str, Any]:
        """1 行 → dict(**昇格**: オブジェクト側へ持ち出す)。

        既定は numpy スカラー(ビット完全)。`native=True` は `.item()` で
        Python スカラーへ落とす(float32 ↔ Python float の往復は仮数が広がるだけの
        可逆変換なので、`dehydrate` で書き戻せばビット一致に戻る。NaN のペイロードは
        quiet NaN へ正規化され得る点だけ非可逆)。
        """
        s = self.slot_of(id_)
        if native:
            return {n: self._arrays[n][s].item() for n in self._order}
        return {n: self._arrays[n][s] for n in self._order}

    def dehydrate(self, id_: int, values: Mapping[str, Any], *,
                  require_full: bool = False) -> None:
        """dict → 1 行(**降格**: オブジェクト側から書き戻す)。

        未知のキーは `SchemaError`。既定では部分 dict を許し、欠けた列は据え置く
        (部分降格)。`require_full=True` でスキーマ全列の存在を要求する。
        """
        s = self.slot_of(id_)
        unknown = [k for k in values if k not in self._columns]
        if unknown:
            raise SchemaError(f"未知の列への dehydrate: {unknown}")
        if require_full:
            missing = [n for n in self._order if n not in values]
            if missing:
                raise SchemaError(f"require_full: 欠けている列 {missing}")
        for name, v in values.items():
            self._arrays[name][s] = v

    def hydrate_many(self, ids) -> dict[str, np.ndarray]:
        """複数行 → {列: 配列}(**一括昇格**)。行の順序は渡した ids の順。"""
        slots = self.slots_of(ids)
        return {n: self._arrays[n][slots] for n in self._order}

    def dehydrate_many(self, ids, values: Mapping[str, Any], *,
                       require_full: bool = False) -> None:
        """{列: 配列} → 複数行(**一括降格**)。ids に重複があると最後が勝つ。"""
        slots = self.slots_of(ids)
        unknown = [k for k in values if k not in self._columns]
        if unknown:
            raise SchemaError(f"未知の列への dehydrate_many: {unknown}")
        if require_full:
            missing = [n for n in self._order if n not in values]
            if missing:
                raise SchemaError(f"require_full: 欠けている列 {missing}")
        for name, v in values.items():
            self._arrays[name][slots] = np.asarray(v, dtype=self._columns[name].dtype)

    # ---------------------------------------------------------------- misc
    def snapshot(self) -> dict[str, np.ndarray]:
        """`[0:n_slots)` の全列コピー + 生存マスク + 世代(テスト/検証用)。"""
        out = {n: self._arrays[n][: self._n_slots].copy() for n in self._order}
        out["_alive"] = self._alive[: self._n_slots].copy()
        out["_gen"] = self._gen[: self._n_slots].copy()
        return out

    def __repr__(self) -> str:                     # pragma: no cover - 表示のみ
        return (f"<ActorTable live={self._n_live} slots={self._n_slots} "
                f"cap={self._capacity} cols={len(self._order)}>")


# =============================================================================
# 5 RowView(オブジェクト風の窓。値は持たない)
# =============================================================================
class RowView:
    """1 行を属性で読み書きする窓。**配列が真実**で、この窓は値を一切保持しない。

    毎アクセスでハンドルを検証するので、free 済みの行を触れば `StaleHandleError`。
    """
    __slots__ = ("_table", "_id", "_slot")

    def __init__(self, table: ActorTable, id_: int):
        object.__setattr__(self, "_table", table)
        object.__setattr__(self, "_id", int(id_))
        object.__setattr__(self, "_slot", int(id_) & int(_SLOT_MASK))

    # -- ハンドル検証 -------------------------------------------------------
    def _require(self) -> int:
        # 純 Python の検証(numpy スカラー往復を避ける = 窓アクセスの主コスト)
        t: ActorTable = object.__getattribute__(self, "_table")
        i: int = object.__getattribute__(self, "_id")
        s: int = object.__getattribute__(self, "_slot")
        if i < 0 or s >= t._n_slots or not t._alive[s] \
                or int(t._gen[s]) != (i >> SLOT_BITS):
            raise StaleHandleError(f"stale handle: id={i}(slot={s})")
        return s

    @property
    def id(self) -> int:
        return object.__getattribute__(self, "_id")

    @property
    def slot(self) -> int:
        return object.__getattribute__(self, "_slot")

    @property
    def gen(self) -> int:
        return object.__getattribute__(self, "_id") >> SLOT_BITS

    @property
    def table(self) -> ActorTable:
        return object.__getattribute__(self, "_table")

    @property
    def alive(self) -> bool:
        t: ActorTable = object.__getattribute__(self, "_table")
        return t.valid_one(object.__getattribute__(self, "_id"))

    # -- 素通しの属性アクセス ----------------------------------------------
    def __getattr__(self, name: str):
        # __slots__ / property に無い名前だけがここへ来る = スキーマ列のみ。
        t: ActorTable = object.__getattribute__(self, "_table")
        if name not in t._columns:
            raise AttributeError(
                f"{name!r} はスキーマ列でない(hot 列のみ。cold はオブジェクト側)")
        return t._arrays[name][self._require()]

    def __setattr__(self, name: str, value) -> None:
        t: ActorTable = object.__getattribute__(self, "_table")
        if name not in t._columns:
            raise AttributeError(
                f"{name!r} はスキーマ列でない(窓に勝手な属性は生やせない)")
        t._arrays[name][self._require()] = value

    def as_dict(self, *, native: bool = False) -> dict[str, Any]:
        t: ActorTable = object.__getattribute__(self, "_table")
        return t.hydrate(object.__getattribute__(self, "_id"), native=native)

    def __repr__(self) -> str:                     # pragma: no cover - 表示のみ
        i = object.__getattribute__(self, "_id")
        return f"<RowView id={i} slot={i & int(_SLOT_MASK)} gen={i >> SLOT_BITS}>"


# =============================================================================
# 6 PendingWrites(AMBER / Unity EntityCommandBuffer 型の書き込みバッファ)
# =============================================================================
@dataclass(frozen=True)
class FlushReport:
    """flush の結果。`conflicts` = 同一 (id, column) で上書きされて捨てられた件数。"""
    pushed: int
    applied: int
    dropped: int
    conflicts: int


class PendingWrites:
    """いつでも受け付け、`flush()` の 1 点でだけ world へ適用する書き込みバッファ。

    決定論の核:
      * 適用順序は **(slot 昇順, column のスキーマ宣言順, seq 昇順, 積んだ順)** に
        正規化される。`push` の呼び出し順(= 将来の並列/非同期な decide の到着順)は
        結果に一切影響しない。
      * 同一 (id, column) への複数書き込みは **seq が最大のものが勝つ**
        (last-writer-wins)。seq は push ごとに単調増加する連番だが、
        `push(..., seq=...)` で**呼び出し側が明示的に与えることもできる**。
        明示 seq を使えば「積む順序を入れ替えても競合の勝敗まで同一」という
        より強い順序独立性が得られる(自動 seq だと積む順=勝敗になるため、
        順序独立性は競合が無い場合に限られる)。
      * 実装は「勝者だけを列ごとに fancy-index 代入」する。逐次適用と結果は
        同値(負けた書き込みはどのみち上書きされるため)で、テストが素朴な
        逐次実装を独立なオラクルとして固定している。
    """

    __slots__ = ("_ids", "_cols", "_vals", "_seqs", "_next")

    def __init__(self):
        self._ids: list[int] = []
        self._cols: list[str] = []
        self._vals: list[Any] = []
        self._seqs: list[int] = []
        self._next = 0

    def __len__(self) -> int:
        return len(self._ids)

    @property
    def next_seq(self) -> int:
        return self._next

    def clear(self) -> None:
        self._ids.clear()
        self._cols.clear()
        self._vals.clear()
        self._seqs.clear()
        self._next = 0

    def push(self, id_: int, column: str, value, seq: int | None = None) -> int:
        """1 件積む。戻り値 = この書き込みの seq(競合の勝敗はこれで決まる)。

        `seq` を省略すると内部の単調増加カウンタを使う。明示的に与えると
        「積む順序に依存しない勝敗」が保証される(同 seq のタイは積んだ順)。
        """
        s = self._next if seq is None else int(seq)
        self._ids.append(int(id_))
        self._cols.append(str(column))
        self._vals.append(value)
        self._seqs.append(s)
        self._next = max(self._next + 1, s + 1)
        return s

    def push_many(self, ids, column: str, values, seq: int | None = None) -> None:
        """同一列へ一括で積む(ids と values は同じ長さ、または values はスカラー)。"""
        a = np.atleast_1d(np.asarray(ids, dtype=np.int64))
        v = np.atleast_1d(np.asarray(values))
        if v.shape[0] == 1 and a.shape[0] != 1:
            v = np.repeat(v, a.shape[0])
        if a.shape[0] != v.shape[0]:
            raise ValueError(f"push_many: 長さ不一致 ids={a.shape[0]} values={v.shape[0]}")
        col = str(column)
        n = a.shape[0]
        self._ids.extend(int(x) for x in a)
        self._cols.extend([col] * n)
        self._vals.extend(v.tolist())
        if seq is None:
            self._seqs.extend(range(self._next, self._next + n))
            self._next += n
        else:
            self._seqs.extend([int(seq)] * n)
            self._next = max(self._next + n, int(seq) + 1)

    def flush(self, table: ActorTable, *, on_stale: str = "raise") -> FlushReport:
        """バッファを table へ適用してクリアする。

        on_stale:
          * ``"raise"``(既定) — 無効ハンドルが 1 件でもあれば `StaleHandleError`。
            **何も適用せずに**バッファも残す(全か無か)。
          * ``"drop"`` — 無効ハンドル宛の書き込みを黙って捨てる(その step で死んだ
            アクタ宛の書き込みを許す運用向け)。捨てた件数は `FlushReport.dropped`。
        """
        if on_stale not in ("raise", "drop"):
            raise ValueError(f"on_stale は 'raise' か 'drop': {on_stale!r}")
        pushed = len(self._ids)
        if pushed == 0:
            return FlushReport(0, 0, 0, 0)

        ids = np.asarray(self._ids, dtype=np.int64)
        unknown = {c for c in self._cols if c not in table._columns}
        if unknown:
            raise SchemaError(f"未知の列への書き込み: {sorted(unknown)}")
        col_idx = np.asarray([table._columns[c].index for c in self._cols], dtype=np.int64)

        ok = table.valid(ids)
        dropped = int((~ok).sum())
        if dropped:
            if on_stale == "raise":
                raise StaleHandleError(
                    f"flush: 無効ハンドル宛の書き込みが {dropped} 件"
                    f"(先頭 {ids[~ok][:8].tolist()})。何も適用していない")
            keep_idx = np.flatnonzero(ok)
        else:
            keep_idx = np.arange(pushed, dtype=np.int64)

        slots = (ids & _SLOT_MASK)[keep_idx]
        cidx = col_idx[keep_idx]
        seq = np.asarray(self._seqs, dtype=np.int64)[keep_idx]
        # 主キー=slot、次いで列(スキーマ宣言順)、次いで seq、最後に積んだ順(タイ)
        order = np.lexsort((keep_idx, seq, cidx, slots))
        s_sorted = slots[order]
        c_sorted = cidx[order]
        n = order.shape[0]
        if n == 0:
            self.clear()
            return FlushReport(pushed, 0, dropped, 0)

        # 各 (slot, column) 群の**最後**(= seq 最大)だけが勝つ
        win = np.empty(n, dtype=bool)
        win[-1] = True
        if n > 1:
            win[:-1] = (s_sorted[:-1] != s_sorted[1:]) | (c_sorted[:-1] != c_sorted[1:])
        winners = order[win]
        conflicts = n - int(win.sum())

        by_col: dict[int, str] = {c.index: c.name for c in table.schema}
        w_cols = c_sorted[win]
        w_slots = s_sorted[win]
        for ci in np.unique(w_cols).tolist():
            name = by_col[int(ci)]
            sel = w_cols == ci
            idx_src = keep_idx[winners[sel]]
            vals = [self._vals[int(j)] for j in idx_src]
            table._arrays[name][w_slots[sel]] = np.asarray(
                vals, dtype=table._columns[name].dtype)

        self.clear()
        return FlushReport(pushed, int(win.sum()), dropped, conflicts)


# =============================================================================
# 7 PhiloxDraws(カウンタベース = 順序非依存の乱数)
# =============================================================================
_PHILOX_M0 = 0xD2E7470EE14C6C93      # PHILOX_M4x64_0
_PHILOX_M1 = 0xCA5A826395121157      # PHILOX_M4x64_1
_PHILOX_W0 = 0x9E3779B97F4A7C15      # golden ratio 64
_PHILOX_W1 = 0xBB67AE8584CAA73B      # sqrt(3)-1
_PHILOX_ROUNDS = 10
_U32_MASK = np.uint64(0xFFFFFFFF)
_U32_SHIFT = np.uint64(32)
#: 2^-53(uint64 の上位 53bit → [0,1) の double。numpy の Generator.random と同式)
_DOUBLE_SCALE = 1.0 / 9007199254740992.0
_DOUBLE_SHIFT = np.uint64(11)

#: 種ドメイン。RngHub(SeedSequence+PCG64)とは別アルゴリズムだが、
#: 名前空間としても "soa" で明示的に隔離する。
_SEED_DOMAIN = b"society.soa.philox.v1"


def _mulhilo(a_const: int, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """64x64→128 bit 乗算を uint64 の 32bit 分割で。numpy の巻き上がりのみに依存。"""
    a = np.uint64(a_const)
    a_lo, a_hi = a & _U32_MASK, a >> _U32_SHIFT
    b_lo, b_hi = b & _U32_MASK, b >> _U32_SHIFT
    lo = a_lo * b_lo
    m1 = a_hi * b_lo
    m2 = a_lo * b_hi
    hi = a_hi * b_hi
    t = (lo >> _U32_SHIFT) + (m1 & _U32_MASK) + (m2 & _U32_MASK)
    lo_out = (lo & _U32_MASK) | ((t & _U32_MASK) << _U32_SHIFT)
    hi_out = hi + (m1 >> _U32_SHIFT) + (m2 >> _U32_SHIFT) + (t >> _U32_SHIFT)
    return hi_out, lo_out


def philox4x64_10(c0, c1, c2, c3, k0, k1) -> tuple[np.ndarray, ...]:
    """Philox4x64-10 の全単射(ベクトル化)。numpy の C 実装とビット一致することを
    `tests/test_soa.py` がランダム化クロスチェックで固定している。"""
    for r in range(_PHILOX_ROUNDS):
        if r:
            k0 = k0 + np.uint64(_PHILOX_W0)
            k1 = k1 + np.uint64(_PHILOX_W1)
        hi0, lo0 = _mulhilo(_PHILOX_M0, c0)
        hi1, lo1 = _mulhilo(_PHILOX_M1, c2)
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
    return c0, c1, c2, c3


def _bump_counter(c0, c1, c2, c3):
    """256bit カウンタ(リトルエンディアン語順)を +1。numpy の Philox が
    「生成の**前に**カウンタを進める」ため、その 1 個ぶんを合わせるのに使う。"""
    one = np.uint64(1)
    n0 = c0 + one
    carry = n0 == np.uint64(0)
    n1 = c1 + carry.astype(np.uint64)
    carry = carry & (n1 == np.uint64(0))
    n2 = c2 + carry.astype(np.uint64)
    carry = carry & (n2 == np.uint64(0))
    n3 = c3 + carry.astype(np.uint64)
    return n0, n1, n2, n3


class PhiloxDraws:
    """`(actor_id, step, channel)` を鍵にしたカウンタベース乱数。

    掟(`src/society/rng.py` と同じもののベクトル化形):
      **どのアクタの乱数も、他アクタの実行順・実行有無に一切依存しない。**
      したがって
        ``uniform(ch, step, ids)[i] == uniform1(ch, step, ids[i])``
      が**ビット単位**で成り立つ(部分集合で引いても、順序を入れ替えても同値)。

    キー/カウンタの割り当て::

        key   = sha256(b"society.soa.philox.v1" | channel | master_seed) の上位 128bit
        ctr   = ( uint64(actor_id), uint64(step), uint64(sub), DOMAIN_TAG )

    * `actor_id` は **packed ID 全体**(席 + 世代)なので、席が再利用されても
      前の住人と同じ乱数列にはならない。
    * `sub` は同一 (channel, step, id) で複数回引くための添字。
    * `DOMAIN_TAG` は sha256(_SEED_DOMAIN) 由来の固定語で、他用途との衝突を塞ぐ。

    実装は 2 経路:
      * `uniform()` = 自前ベクトル Philox(本番経路。NEP 19 非依存)
      * `uniform1()` = numpy の C 実装 `np.random.Philox` 経由(独立な検証経路)
    """

    #: 1 ブロックあたりの id 数。Philox の 10 ラウンドは 1 回の呼び出しで
    #: 30 本前後の一時 uint64 配列を作るため、id をまとめて処理すると作業集合が
    #: L2/L3 を溢れて**帯域律速**になる(実測 250k 一括 232ms → 4096 分割 62ms
    #: = 3.7 倍)。分割しても各行は独立なので**出力はビット完全に同一**
    #: (`tests/test_soa.py` がブロック長非依存を機械固定している)。
    CHUNK = 4096

    __slots__ = ("master_seed", "_key_cache", "_domain_tag")

    def __init__(self, master_seed: int):
        self.master_seed = int(master_seed)
        self._key_cache: dict[str, tuple[int, int]] = {}
        d = hashlib.sha256(_SEED_DOMAIN).digest()
        self._domain_tag = int.from_bytes(d[:8], "little")

    # ------------------------------------------------------------------ key
    def key_words(self, channel: str) -> tuple[int, int]:
        """(channel) → Philox の 128bit 鍵 (k0, k1)。root = (master_seed, "soa", channel)。"""
        cached = self._key_cache.get(channel)
        if cached is not None:
            return cached
        h = hashlib.sha256()
        h.update(_SEED_DOMAIN)
        h.update(b"|")
        h.update(channel.encode("utf-8"))
        h.update(b"|")
        h.update(self.master_seed.to_bytes(16, "little", signed=True))
        d = h.digest()
        kw = (int.from_bytes(d[0:8], "little"), int.from_bytes(d[8:16], "little"))
        self._key_cache[channel] = kw
        return kw

    def counter_words(self, step: int, ids, sub: int = 0) -> tuple[np.ndarray, ...]:
        """(step, ids, sub) → 256bit カウンタの 4 語(uint64 配列)。"""
        c0 = np.atleast_1d(
            np.ascontiguousarray(np.asarray(ids, dtype=np.int64))).view(np.uint64)
        c1 = np.full(c0.shape, np.int64(step), dtype=np.int64).view(np.uint64)
        c2 = np.full(c0.shape, np.int64(sub), dtype=np.int64).view(np.uint64)
        c3 = np.full(c0.shape, self._domain_tag, dtype=np.uint64)
        return c0, c1, c2, c3

    # ------------------------------------------------------------------ raw
    def _blocks(self, channel: str, step: int, a: np.ndarray, sub: int):
        """`CHUNK` ごとに (lo, hi, (w0,w1,w2,w3)) を返す内部イテレータ。"""
        k0, k1 = self.key_words(channel)
        n = a.shape[0]
        chunk = max(int(self.CHUNK), 1)
        for lo in range(0, n, chunk):
            hi = min(lo + chunk, n)
            c = self.counter_words(step, a[lo:hi], sub)
            c = _bump_counter(*c)                  # numpy の Philox と同じ位置合わせ
            m = hi - lo
            kk0 = np.full(m, k0, dtype=np.uint64)
            kk1 = np.full(m, k1, dtype=np.uint64)
            yield lo, hi, philox4x64_10(c[0], c[1], c[2], c[3], kk0, kk1)

    @staticmethod
    def _ids(ids) -> np.ndarray:
        return np.atleast_1d(np.ascontiguousarray(np.asarray(ids, dtype=np.int64)))

    def raw4(self, channel: str, step: int, ids, sub: int = 0) -> np.ndarray:
        """(n, 4) の uint64。1 回の全単射で 4 語=4 draw ぶんが同時に出る
        (1 draw あたりの償却コストは `uniform4` を使えば 1/4)。"""
        a = self._ids(ids)
        out = np.empty((a.shape[0], 4), dtype=np.uint64)
        for lo, hi, w in self._blocks(channel, step, a, sub):
            for j in range(4):
                out[lo:hi, j] = w[j]
        return out

    # -------------------------------------------------------------- uniform
    def uniform(self, channel: str, step: int, ids, sub: int = 0) -> np.ndarray:
        """[0,1) の float64 を ids 1 つにつき 1 個(本番のベクトル経路)。"""
        a = self._ids(ids)
        out = np.empty(a.shape[0], dtype=np.float64)
        for lo, hi, w in self._blocks(channel, step, a, sub):
            out[lo:hi] = (w[0] >> _DOUBLE_SHIFT).astype(np.float64) * _DOUBLE_SCALE
        return out

    def uniform4(self, channel: str, step: int, ids, sub: int = 0) -> np.ndarray:
        """(n, 4) の [0,1)。1 回の全単射から 4 本まとめて取り出す(償却 1/4)。"""
        a = self._ids(ids)
        out = np.empty((a.shape[0], 4), dtype=np.float64)
        for lo, hi, w in self._blocks(channel, step, a, sub):
            for j in range(4):
                out[lo:hi, j] = (w[j] >> _DOUBLE_SHIFT).astype(np.float64) * _DOUBLE_SCALE
        return out

    def uniform1(self, channel: str, step: int, id_: int, sub: int = 0) -> float:
        """単体版。**numpy の C 実装 `np.random.Philox` を通る独立経路**で、
        `uniform()` とのビット一致がテストで固定されている。"""
        k0, k1 = self.key_words(channel)
        ctr = np.array([np.int64(id_).view(np.uint64), np.int64(step).view(np.uint64),
                        np.int64(sub).view(np.uint64), np.uint64(self._domain_tag)],
                       dtype=np.uint64)
        key = np.array([k0, k1], dtype=np.uint64)
        # ★ Python の list を渡すと numpy が float64 経由で下位ビットを落とす。
        #    必ず uint64 の ndarray を渡すこと。
        bg = np.random.Philox(counter=ctr, key=key)
        return float(np.random.Generator(bg).random())

    # ------------------------------------------------------------- helpers
    def bernoulli(self, channel: str, step: int, ids, p, sub: int = 0) -> np.ndarray:
        """u < p の bool 配列。p はスカラーでも ids と同長の配列でも可。"""
        return self.uniform(channel, step, ids, sub) < np.asarray(p, dtype=np.float64)

    def choice_weights(self, channel: str, step: int, ids, weights,
                       sub: int = 0) -> np.ndarray:
        """非負重みからの離散抽選(index の int64 配列)。

        weights: (k,) なら全 id 共通、(n, k) なら行ごと。行和が 0 の行は 0 を返す。
        逆関数法 + `searchsorted` 相当で、`uniform` と同じ 1 本の draw しか使わない
        = 単体/一括の一致がそのまま保たれる。
        """
        u = self.uniform(channel, step, ids, sub)
        w = np.asarray(weights, dtype=np.float64)
        if w.ndim == 1:
            w = np.broadcast_to(w, (u.shape[0], w.shape[0]))
        if w.shape[0] != u.shape[0]:
            raise ValueError(f"choice_weights: 行数不一致 ids={u.shape[0]} w={w.shape[0]}")
        if np.any(w < 0):
            raise ValueError("choice_weights: 負の重み")
        cdf = np.cumsum(w, axis=1)
        total = cdf[:, -1]
        target = u * total
        idx = (cdf <= target[:, None]).sum(axis=1)
        idx = np.minimum(idx, w.shape[1] - 1)
        return np.where(total > 0.0, idx, 0).astype(np.int64)


# =============================================================================
# 8 昇格の要求集合セマンティクス(上限は上位層が決める。ここは決定論的な分割のみ)
# =============================================================================
def select_promotions(requested, cap: int | None = None,
                      *, priority=None) -> tuple[np.ndarray, np.ndarray]:
    """昇格要求集合を (granted, deferred) に**決定論的に**分割する。

    * `cap=None` なら全件 granted。
    * `priority` を与えると (priority 降順, slot 昇順) で並べ替えてから上位 cap 件。
      与えなければ **slot 昇順**(= ID 昇順)そのまま。
    * 重複 ID は 1 件に畳む。並びは常に一意 = 呼び出し順に依存しない。

    連鎖昇格(昇格した個体が更に周囲を昇格させる)の抑止は**上位層の責務**で、
    ここでは扱わない。ここが提供するのは「上限を掛けても順序が揺れない」ことだけ。
    """
    ids = np.unique(np.atleast_1d(np.asarray(requested, dtype=np.int64)))
    if priority is not None:
        pr = np.asarray(priority, dtype=np.float64)
        req = np.atleast_1d(np.asarray(requested, dtype=np.int64))
        if pr.shape[0] != req.shape[0]:
            raise ValueError("select_promotions: priority の長さが requested と違う")
        # 重複 ID は最大優先度を採る(決定論)
        best: dict[int, float] = {}
        for i, p in zip(req.tolist(), pr.tolist()):
            if i not in best or p > best[i]:
                best[i] = p
        pv = np.asarray([best[int(i)] for i in ids], dtype=np.float64)
        order = np.lexsort((ids, -pv))
        ids = ids[order]
    if cap is None or cap >= ids.shape[0]:
        return ids, np.zeros(0, dtype=np.int64)
    cap = max(int(cap), 0)
    return ids[:cap], ids[cap:]
