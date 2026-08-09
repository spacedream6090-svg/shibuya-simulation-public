"""P5 土台: SoA アクタ基盤 `src/society/engine/soa.py` のテスト。

守るもの(検収基準の順)
  ① スキーマ = 明示幅 dtype のみ(プラットフォーム依存 int を弾く)
  ② 世代付き ID: packed int64 (slot:40, gen:23)・stale ハンドルは必ず例外
  ③ 席は暗黙に再利用されない(`compact()` を明示的に呼んだときだけ)
  ④ 成長しても ID / 席番号は 1 つも動かない(償却 2 倍成長)
  ⑤ 配列が真実: `RowView` は素通しの窓(値を持たない)
  ⑥ hydrate ∘ dehydrate = 恒等(列ごとにバイト一致)
  ⑦ PendingWrites: 積む順序に依存しない flush・明示 seq の last-writer-wins・
     素朴な逐次オラクルとの一致
  ⑧ ★PhiloxDraws: 一括 `uniform()` と単体 `uniform1()` が**ビット一致**
     (前者=自前ベクトル実装 / 後者=numpy の C 実装 = 独立な 2 経路の相互検証)
  ⑨ ゴールデン literal(固定キーの先頭 8 draw)= NEP 19 に対する自前の凍結
  ⑩ スレッド不在(AST の静的検査)・society への依存ゼロ(自己完結)
"""
from __future__ import annotations

import ast
import random
from pathlib import Path

import numpy as np
import pytest

from society.engine import soa
from society.engine.soa import (
    ActorTable, CapacityError, PendingWrites, PhiloxDraws, RowView,
    SchemaError, StaleHandleError, select_promotions,
)

MODULE_PATH = Path(soa.__file__)

SCHEMA = [
    ("money", "float32", 0.0),
    ("node", "int32", -1),
    ("need", "float64", 0.5),
    ("flags", "uint8", 0),
    ("sleeping", "bool", False),
    ("ptr", "int64", 0),
]


def _table(capacity: int = 0) -> ActorTable:
    return ActorTable(SCHEMA, capacity=capacity)


# =========================================================================== #
# ① スキーマ
# =========================================================================== #
def test_schema_keeps_declaration_order_and_dtypes():
    t = _table()
    assert t.column_names == ("money", "node", "need", "flags", "sleeping", "ptr")
    assert [c.dtype.name for c in t.schema] == [
        "float32", "int32", "float64", "uint8", "bool", "int64"]
    assert [c.index for c in t.schema] == [0, 1, 2, 3, 4, 5]


@pytest.mark.parametrize("spec", ["int", "uint", "intp", "float", "long", "l", "q", "d"])
def test_schema_rejects_ambiguous_dtype_strings(spec):
    with pytest.raises(SchemaError):
        ActorTable([("x", spec, 0)])


@pytest.mark.parametrize("spec", [int, float, bool, complex, np.int_, np.intp, np.uintp])
def test_schema_rejects_platform_dependent_objects(spec):
    with pytest.raises(SchemaError):
        ActorTable([("x", spec, 0)])


def test_schema_rejects_object_and_string_dtypes():
    for spec in ("U8", "O", "M8[ns]", "c8"):
        with pytest.raises(SchemaError):
            ActorTable([("x", spec, 0)])


def test_schema_rejects_bad_shapes():
    with pytest.raises(SchemaError):
        ActorTable([])
    with pytest.raises(SchemaError):
        ActorTable([("x", "int32")])                     # 3 つ組でない
    with pytest.raises(SchemaError):
        ActorTable([("x", "int32", 0), ("x", "int8", 0)])  # 重複
    with pytest.raises(SchemaError):
        ActorTable([("_x", "int32", 0)])                 # 先頭 '_' は内部予約
    with pytest.raises(SchemaError):
        ActorTable([("", "int32", 0)])


def test_unknown_column_access_raises():
    t = _table()
    t.alloc(1)
    with pytest.raises(SchemaError):
        t.col("nope")
    with pytest.raises(SchemaError):
        t.get("nope", t.ids)


# =========================================================================== #
# ② 世代付き ID のレイアウト
# =========================================================================== #
def test_packed_id_layout_is_slot40_gen23():
    assert soa.SLOT_BITS == 40
    assert soa.GEN_BITS == 23
    assert soa.MAX_SLOTS == 1 << 40
    assert soa.MAX_GEN == (1 << 23) - 1
    # 符号ビットを踏まない = 有効 ID は必ず非負で、席順とソート順が一致する
    assert soa.SLOT_BITS + soa.GEN_BITS == 63
    assert int(soa.pack_id(0, 1)) == 1 << 40
    assert int(soa.pack_id(soa.MAX_SLOTS - 1, soa.MAX_GEN)) == (1 << 63) - 1
    assert int(soa.pack_id(soa.MAX_SLOTS - 1, soa.MAX_GEN)) > 0


def test_pack_unpack_roundtrip():
    slots = np.array([0, 1, 12345, soa.MAX_SLOTS - 1], dtype=np.int64)
    gens = np.array([1, 7, soa.MAX_GEN, 3], dtype=np.int64)
    ids = soa.pack_id(slots, gens)
    s2, g2 = soa.unpack_id(ids)
    assert np.array_equal(s2, slots)
    assert np.array_equal(g2, gens)


def test_pack_id_rejects_overflow():
    with pytest.raises(CapacityError):
        soa.pack_id(soa.MAX_SLOTS, 1)
    with pytest.raises(CapacityError):
        soa.pack_id(0, soa.MAX_GEN + 1)
    with pytest.raises(CapacityError):
        soa.pack_id(0, 0)


def test_null_id_is_never_valid():
    t = _table()
    t.alloc(3)
    assert not bool(t.valid(np.int64(soa.NULL_ID)))
    assert not bool(t.valid(np.int64(0)))
    assert t.id_of_slot(99) == soa.NULL_ID


# =========================================================================== #
# ③ alloc / free / compact
# =========================================================================== #
def test_alloc_and_active_mask():
    t = _table()
    ids = t.alloc(4)
    assert len(t) == 4
    assert t.n_slots == 4
    assert np.array_equal(t.ids, ids)
    assert t.active.tolist() == [True] * 4
    assert t.active.shape == (4,)
    assert t.col("need").tolist() == [0.5] * 4        # default 充填
    assert t.col("node").tolist() == [-1] * 4
    assert t.alloc(0).shape == (0,)
    with pytest.raises(ValueError):
        t.alloc(-1)


def test_free_marks_dead_and_does_not_reuse_slot():
    t = _table()
    ids = t.alloc(4)
    t.free(ids[[1, 2]])
    assert len(t) == 2
    assert t.valid(ids).tolist() == [True, False, False, True]
    assert t.reclaimable == 2
    fresh = t.alloc(2)
    # 席 1,2 は死んだまま。新規席は 4,5(暗黙の再利用は絶対に起きない)
    assert soa.unpack_id(fresh)[0].tolist() == [4, 5]
    assert t.n_slots == 6


def test_compact_is_the_only_route_to_slot_reuse():
    t = _table()
    ids = t.alloc(4)
    t.col("money")[:] = [1.0, 2.0, 3.0, 4.0]
    t.free(ids[[1, 2]])
    assert t.compact() == 2
    assert t.reclaimable == 0
    reused = t.alloc(2)
    slots, gens = soa.unpack_id(reused)
    assert slots.tolist() == [1, 2]                   # 昇順で再利用
    assert gens.tolist() == [2, 2]                    # 世代が上がっている
    assert t.n_slots == 4                             # 高水位は伸びない
    # 再利用席は全列 default に戻る(前の住人の残骸を持ち越さない)
    assert t.get("money", reused).tolist() == [0.0, 0.0]
    assert t.get("node", reused).tolist() == [-1, -1]
    # 古いハンドルは世代違いで死んだまま
    assert t.valid(ids).tolist() == [True, False, False, True]
    assert t.compact() == 0


def test_stale_handles_raise_everywhere():
    t = _table()
    ids = t.alloc(3)
    dead = int(ids[1])
    t.free(np.int64(dead))
    for call in (
        lambda: t.view(dead),
        lambda: t.slots_of(np.int64(dead)),
        lambda: t.slot_of(dead),
        lambda: t.get("money", np.int64(dead)),
        lambda: t.set("money", np.int64(dead), 1.0),
        lambda: t.hydrate(dead),
        lambda: t.dehydrate(dead, {"money": 1.0}),
        lambda: t.free(np.int64(dead)),               # 二重 free
    ):
        with pytest.raises(StaleHandleError):
            call()
    assert not bool(t.valid(np.int64(dead)))


def test_scalar_fast_path_agrees_with_vectorized_valid():
    """`valid_one` / `slot_of` は純 Python の速い経路。ベクトル版と必ず一致すること。"""
    t = _table()
    ids = t.alloc(6)
    t.free(ids[[0, 4]])
    probes = np.concatenate([
        ids,
        np.array([soa.NULL_ID, 0, 1, soa.pack_id(2, 9), soa.pack_id(999, 1)],
                 dtype=np.int64),
    ])
    vec = t.valid(probes).tolist()
    assert [t.valid_one(int(i)) for i in probes] == vec
    for i, ok in zip(probes.tolist(), vec):
        if ok:
            assert t.slot_of(i) == int(t.slots_of(np.int64(i))[0])
        else:
            with pytest.raises(StaleHandleError):
                t.slot_of(i)


def test_free_of_never_allocated_id_raises():
    t = _table()
    t.alloc(1)
    with pytest.raises(StaleHandleError):
        t.free(np.int64(soa.pack_id(500, 1)))


def test_free_dedupes_duplicate_ids_in_one_call():
    t = _table()
    ids = t.alloc(3)
    freed = t.free(np.array([ids[0], ids[0], ids[2]], dtype=np.int64))
    assert freed.tolist() == [0, 2]
    assert len(t) == 1


def test_dead_row_values_survive_until_reuse():
    """死んだ行の値は消さない(事後の読み出し用)。除外は `active` マスクで行う。"""
    t = _table()
    ids = t.alloc(2)
    t.set("money", ids, [7.0, 8.0])
    t.free(ids[[0]])
    assert t.col("money").tolist() == [7.0, 8.0]
    assert t.col("money")[t.active].tolist() == [8.0]


def test_id_stability_across_alloc_free_realloc():
    """★ID は alloc/free/再確保をまたいで安定(生きている行は 1 行も動かない)。"""
    t = _table()
    a = t.alloc(5)
    t.set("ptr", a, [10, 11, 12, 13, 14])
    keep = a[[0, 4]]
    t.free(a[[1, 2, 3]])
    t.compact()
    b = t.alloc(3)
    t.set("ptr", b, [20, 21, 22])
    t.free(b[[0]])
    c = t.alloc(4)
    # keep の席・値は最初から一切変わっていない
    assert soa.unpack_id(keep)[0].tolist() == [0, 4]
    assert t.get("ptr", keep).tolist() == [10, 14]
    assert t.valid(keep).tolist() == [True, True]
    assert len(t) == 2 + 2 + 4
    assert np.array_equal(np.sort(t.ids), np.sort(np.concatenate([keep, b[1:], c])))


def test_generation_exhaustion_retires_the_slot():
    t = _table()
    (i0,) = t.alloc(1)
    t._gen[0] = soa.MAX_GEN                            # 世代を使い切った状態を作る
    cur = int(soa.pack_id(0, soa.MAX_GEN))
    t.free(np.int64(cur))
    assert t.retired == 1
    assert t.reclaimable == 0                          # 退役席は compact でも戻らない
    assert t.compact() == 0
    assert soa.unpack_id(t.alloc(1))[0].tolist() == [1]
    assert int(i0) > 0


# =========================================================================== #
# ④ 成長
# =========================================================================== #
def test_growth_is_deterministic_powers_of_two():
    t = _table()
    caps = []
    for _ in range(6):
        t.alloc(3)
        caps.append(t.capacity)
    assert caps == [8, 8, 16, 16, 16, 32]


def test_growth_does_not_move_ids_or_values():
    t = _table()
    first = t.alloc(3)
    t.set("ptr", first, [101, 102, 103])
    slots_before = soa.unpack_id(first)[0].tolist()
    cap_before = t.capacity
    t.alloc(5000)                                      # 何度も 2 倍成長させる
    assert t.capacity > cap_before
    assert soa.unpack_id(first)[0].tolist() == slots_before
    assert t.valid(first).tolist() == [True] * 3
    assert t.get("ptr", first).tolist() == [101, 102, 103]


def test_reserve_pre_sizes_without_allocating_rows():
    t = _table()
    t.reserve(1000)
    assert t.capacity >= 1000
    assert len(t) == 0 and t.n_slots == 0


def test_old_column_view_is_detached_after_growth():
    """★既知の危険: 容量成長は再確保を伴うので、成長前に取ったビューは旧バッファのまま。

    ID と席番号は動かないが、**ビューは取り直す必要がある**ことを機械で固定する。
    """
    t = _table(capacity=4)
    ids = t.alloc(4)
    view = t.col("money")
    view[:] = 1.0
    t.alloc(100)                                        # 成長 = 再確保
    stale_view = view
    stale_view[:] = 999.0                               # 旧バッファへの書き込み
    assert t.get("money", ids).tolist() == [1.0] * 4    # テーブルには通らない
    assert t.col("money")[:4].tolist() == [1.0] * 4


# =========================================================================== #
# ⑤ RowView = 素通しの窓(配列が真実)
# =========================================================================== #
def test_rowview_reads_and_writes_through_to_arrays():
    t = _table()
    ids = t.alloc(3)
    v = t.view(int(ids[1]))
    assert isinstance(v, RowView)
    v.money = 12.5
    v.node = 42
    v.sleeping = True
    assert t.col("money")[1] == np.float32(12.5)
    assert t.col("node")[1] == 42
    assert bool(t.col("sleeping")[1])
    # 逆向き: 配列を直接書けば窓からも見える(窓は値を持たない)
    t.col("money")[1] = -3.25
    assert v.money == np.float32(-3.25)
    assert v.id == int(ids[1]) and v.slot == 1 and v.gen == 1
    assert v.table is t
    assert v.alive


def test_rowview_rejects_non_schema_attributes():
    t = _table()
    (i0,) = t.alloc(1)
    v = t.view(int(i0))
    with pytest.raises(AttributeError):
        _ = v.persona                                   # cold はオブジェクト側の担当
    with pytest.raises(AttributeError):
        v.persona = "x"


def test_rowview_becomes_stale_after_free():
    t = _table()
    ids = t.alloc(2)
    v = t.view(int(ids[0]))
    t.free(ids[[0]])
    assert not v.alive
    with pytest.raises(StaleHandleError):
        _ = v.money
    with pytest.raises(StaleHandleError):
        v.money = 1.0


def test_rows_iterates_live_rows_in_slot_order():
    t = _table()
    ids = t.alloc(5)
    t.free(ids[[1, 3]])
    assert [v.slot for v in t.rows()] == [0, 2, 4]


def test_masked_vectorized_update_is_the_intended_pattern():
    t = _table()
    ids = t.alloc(10)
    t.col("need")[:] = 1.0
    t.free(ids[[2, 7]])
    need = t.col("need")
    need[t.active] *= 0.5
    assert need[t.active].tolist() == [0.5] * 8
    assert need[2] == 1.0 and need[7] == 1.0


# =========================================================================== #
# ⑥ hydrate / dehydrate 恒等
# =========================================================================== #
_EXTREMES = {
    "money": np.float32(np.pi),
    "node": np.int32(-2147483648),
    "need": np.float64(1e-300),
    "flags": np.uint8(255),
    "sleeping": np.True_,
    "ptr": np.int64(-9223372036854775808),
}


def _col_bytes(t: ActorTable) -> dict[str, bytes]:
    return {n: t.col(n).tobytes() for n in t.column_names}


def test_dehydrate_of_hydrate_is_identity_numpy_scalars():
    t = _table()
    ids = t.alloc(3)
    for i in ids.tolist():
        t.dehydrate(i, _EXTREMES)
    before = _col_bytes(t)
    for i in ids.tolist():
        t.dehydrate(i, t.hydrate(i))
    assert _col_bytes(t) == before


def test_dehydrate_of_hydrate_is_identity_native_python_scalars():
    """float32 ↔ Python float の往復も**バイト一致**で戻る(可逆な拡張のため)。"""
    t = _table()
    ids = t.alloc(4)
    rng = np.random.default_rng(3)
    t.col("money")[:] = rng.standard_normal(4).astype(np.float32)
    t.col("need")[:] = rng.standard_normal(4)
    t.col("node")[:] = rng.integers(-2**31, 2**31 - 1, 4)
    t.col("ptr")[:] = rng.integers(-2**62, 2**62, 4)
    t.col("flags")[:] = rng.integers(0, 256, 4)
    t.col("sleeping")[:] = rng.integers(0, 2, 4).astype(bool)
    before = _col_bytes(t)
    for i in ids.tolist():
        d = t.hydrate(i, native=True)
        assert all(not isinstance(v, np.generic) for v in d.values())
        t.dehydrate(i, d)
    assert _col_bytes(t) == before


def test_hydrate_dehydrate_preserves_nan_and_inf():
    t = _table()
    (i0,) = t.alloc(1)
    t.dehydrate(i0, {"money": np.float32("nan"), "need": float("-inf")})
    d = t.hydrate(i0, native=True)
    assert np.isnan(d["money"]) and d["need"] == float("-inf")
    t.dehydrate(i0, d)
    assert np.isnan(t.col("money")[0]) and np.isneginf(t.col("need")[0])


def test_hydrate_many_dehydrate_many_roundtrip():
    t = _table()
    ids = t.alloc(6)
    rng = np.random.default_rng(11)
    t.col("money")[:] = rng.standard_normal(6).astype(np.float32)
    t.col("ptr")[:] = rng.integers(-10**12, 10**12, 6)
    before = _col_bytes(t)
    bag = t.hydrate_many(ids)
    assert set(bag) == set(t.column_names)
    t.col("money")[:] = 0.0
    t.dehydrate_many(ids, bag)
    assert _col_bytes(t) == before


def test_dehydrate_validates_keys():
    t = _table()
    (i0,) = t.alloc(1)
    with pytest.raises(SchemaError):
        t.dehydrate(i0, {"persona": 1})
    with pytest.raises(SchemaError):
        t.dehydrate(i0, {"money": 1.0}, require_full=True)
    with pytest.raises(SchemaError):
        t.dehydrate_many(np.atleast_1d(i0), {"persona": [1]})
    t.dehydrate(i0, t.hydrate(i0), require_full=True)     # 全列そろえば通る


def test_partial_dehydrate_leaves_other_columns_alone():
    t = _table()
    (i0,) = t.alloc(1)
    t.dehydrate(i0, {"money": 5.0, "node": 9})
    t.dehydrate(i0, {"node": 3})
    assert t.hydrate(i0, native=True)["money"] == 5.0
    assert t.hydrate(i0, native=True)["node"] == 3


# =========================================================================== #
# ⑦ PendingWrites
# =========================================================================== #
def _naive_flush(t: ActorTable, records) -> None:
    """独立オラクル: (slot, 列宣言順, seq, 積んだ順)で並べて**素朴に逐次適用**。"""
    idx = {c.name: c.index for c in t.schema}
    order = sorted(range(len(records)),
                   key=lambda k: (int(records[k][0]) & (soa.MAX_SLOTS - 1),
                                  idx[records[k][1]], records[k][3], k))
    for k in order:
        id_, col, val, _seq = records[k]
        t.col_full(col)[t.slot_of(int(id_))] = val


def test_flush_matches_naive_sequential_oracle():
    rng = random.Random(5)
    t1, t2 = _table(), _table()
    ids1, ids2 = t1.alloc(60), t2.alloc(60)
    assert np.array_equal(ids1, ids2)
    cols = ["money", "node", "need", "flags", "ptr"]
    pw = PendingWrites()
    records = []
    for k in range(600):
        i = int(ids1[rng.randrange(60)])
        c = rng.choice(cols)
        v = rng.randrange(0, 100) if c in ("node", "flags", "ptr") else rng.random()
        seq = pw.push(i, c, v)
        records.append((i, c, v, seq))
    rep = pw.flush(t1)
    assert rep.pushed == 600 and rep.applied + rep.conflicts == 600
    assert rep.dropped == 0 and rep.conflicts > 0
    _naive_flush(t2, records)
    assert _col_bytes(t1) == _col_bytes(t2)


def test_flush_is_independent_of_insertion_order_without_conflicts():
    """★競合が無ければ、積む順序をどう入れ替えても結果はバイト一致。"""
    rng = random.Random(17)
    writes = []
    for slot in range(50):
        for c in ("money", "node", "ptr"):
            writes.append((slot, c, rng.random() if c == "money" else rng.randrange(1000)))
    results = []
    for trial in range(4):
        order = list(range(len(writes)))
        if trial:
            random.Random(100 + trial).shuffle(order)
        t = _table()
        ids = t.alloc(50)
        pw = PendingWrites()
        for k in order:
            slot, c, v = writes[k]
            pw.push(int(ids[slot]), c, v)
        pw.flush(t)
        results.append(_col_bytes(t))
    assert all(r == results[0] for r in results)


def test_explicit_seq_makes_even_conflicting_writes_order_independent():
    """★明示 seq を使えば、競合があっても積む順序に依存しない(より強い性質)。"""
    rng = random.Random(23)
    plan = [(rng.randrange(20), rng.choice(["money", "node"]),
             rng.randrange(1000), rng.randrange(10_000)) for _ in range(400)]
    results = []
    for trial in range(4):
        order = list(range(len(plan)))
        if trial:
            random.Random(900 + trial).shuffle(order)
        t = _table()
        ids = t.alloc(20)
        pw = PendingWrites()
        for k in order:
            slot, c, v, seq = plan[k]
            pw.push(int(ids[slot]), c, v, seq=seq)
        pw.flush(t)
        results.append(_col_bytes(t))
    assert all(r == results[0] for r in results)


def test_last_writer_wins_by_sequence_number():
    t = _table()
    ids = t.alloc(2)
    pw = PendingWrites()
    pw.push(int(ids[0]), "node", 1, seq=5)
    pw.push(int(ids[0]), "node", 2, seq=99)
    pw.push(int(ids[0]), "node", 3, seq=7)              # 途中で積んでも seq が小さければ負け
    pw.push(int(ids[1]), "node", 8)
    rep = pw.flush(t)
    assert rep.conflicts == 2 and rep.applied == 2
    assert t.get("node", ids).tolist() == [2, 8]


def test_flush_clears_buffer_and_empty_flush_is_noop():
    t = _table()
    ids = t.alloc(1)
    pw = PendingWrites()
    pw.push(int(ids[0]), "node", 4)
    assert len(pw) == 1
    pw.flush(t)
    assert len(pw) == 0 and pw.next_seq == 0
    rep = pw.flush(t)
    assert rep == soa.FlushReport(0, 0, 0, 0)
    assert t.get("node", ids).tolist() == [4]


def test_flush_stale_raise_is_all_or_nothing():
    t = _table()
    ids = t.alloc(3)
    t.free(ids[[1]])
    pw = PendingWrites()
    pw.push(int(ids[0]), "node", 7)
    pw.push(int(ids[1]), "node", 7)                     # 死んだ相手宛
    with pytest.raises(StaleHandleError):
        pw.flush(t)
    assert t.get("node", ids[[0]]).tolist() == [-1]     # 何も適用されていない
    assert len(pw) == 2                                 # バッファも残っている


def test_flush_stale_drop_applies_the_rest():
    t = _table()
    ids = t.alloc(3)
    t.free(ids[[1]])
    pw = PendingWrites()
    pw.push(int(ids[0]), "node", 7)
    pw.push(int(ids[1]), "node", 7)
    pw.push(int(ids[2]), "node", 9)
    rep = pw.flush(t, on_stale="drop")
    assert rep == soa.FlushReport(pushed=3, applied=2, dropped=1, conflicts=0)
    assert t.get("node", ids[[0, 2]]).tolist() == [7, 9]


def test_flush_rejects_unknown_column_and_bad_mode():
    t = _table()
    ids = t.alloc(1)
    pw = PendingWrites()
    pw.push(int(ids[0]), "persona", 1)
    with pytest.raises(SchemaError):
        pw.flush(t)
    with pytest.raises(ValueError):
        PendingWrites().flush(t, on_stale="whatever")


def test_push_many_and_scalar_broadcast():
    t = _table()
    ids = t.alloc(4)
    pw = PendingWrites()
    pw.push_many(ids, "node", [1, 2, 3, 4])
    pw.push_many(ids, "flags", 7)
    pw.flush(t)
    assert t.get("node", ids).tolist() == [1, 2, 3, 4]
    assert t.get("flags", ids).tolist() == [7] * 4
    with pytest.raises(ValueError):
        PendingWrites().push_many(ids, "node", [1, 2])


def test_flush_dtype_is_the_column_dtype_not_the_value_dtype():
    t = _table()
    ids = t.alloc(2)
    pw = PendingWrites()
    pw.push(int(ids[0]), "money", 1.0 / 3.0)            # Python float → float32 列
    pw.flush(t)
    assert t.col("money").dtype == np.float32
    assert t.get("money", ids[[0]])[0] == np.float32(1.0 / 3.0)


# =========================================================================== #
# ⑧⑨ PhiloxDraws
# =========================================================================== #
# ゴールデン: PhiloxDraws(20260806).uniform("golden", 0, arange(8))
# ★この literal は**自前ベクトル実装の凍結**。numpy 側の乱数方針(NEP 19)に
#   依存しないので、numpy を上げてもここは動かないのが正しい。
GOLDEN_UNIFORM8 = [
    0.9004285280069122,
    0.65119489092870864,
    0.18478463088709884,
    0.44388093355376235,
    0.33328957511418267,
    0.066820796153506223,
    0.73909786087109719,
    0.73645887232945284,
]
# philox4x64_10(ctr=(1,2,3,4), key=(5,6)) の生出力(全単射そのものの凍結)
GOLDEN_BIJECTION = [11789110016301065044, 12460072761081090454,
                    11575064416179582204, 635235873073864927]


def test_golden_philox_literals_are_frozen():
    d = PhiloxDraws(20260806)
    u = d.uniform("golden", 0, np.arange(8, dtype=np.int64))
    assert u.tolist() == GOLDEN_UNIFORM8
    raw = soa.philox4x64_10(*[np.array([v], dtype=np.uint64)
                              for v in (1, 2, 3, 4, 5, 6)])
    assert [int(x[0]) for x in raw] == GOLDEN_BIJECTION


def test_vectorized_philox_matches_numpy_c_implementation():
    """★独立検証: 自前ベクトル Philox4x64-10 == numpy の C 実装(ランダム 64 組)。

    numpy は「生成の**前に**カウンタを +1 する」ので、counter=c で作った
    BitGenerator の最初のブロックは c+1 に対応する。
    """
    rng = np.random.default_rng(2026)
    for _ in range(64):
        ctr = rng.integers(0, 2**64, size=4, dtype=np.uint64)
        key = rng.integers(0, 2**64, size=2, dtype=np.uint64)
        bg = np.random.Philox(counter=np.asarray(ctr, dtype=np.uint64),
                              key=np.asarray(key, dtype=np.uint64))
        # ★ list を渡すと numpy が float64 経由で下位ビットを落とす。ndarray 必須。
        assert bg.state["state"]["counter"].tolist() == ctr.tolist()
        want = [int(x) for x in bg.random_raw(4)]
        n = (int(ctr[0]) | int(ctr[1]) << 64 | int(ctr[2]) << 128
             | int(ctr[3]) << 192) + 1
        n %= 2 ** 256
        w = [np.array([(n >> (64 * i)) & (2**64 - 1)], dtype=np.uint64)
             for i in range(4)]
        got = soa.philox4x64_10(w[0], w[1], w[2], w[3],
                                np.array([key[0]], dtype=np.uint64),
                                np.array([key[1]], dtype=np.uint64))
        assert [int(x[0]) for x in got] == want


@pytest.mark.parametrize("id_dtype", [np.int64, np.int32])
def test_batch_equals_single_bitwise(id_dtype):
    """★中核の掟: 一括 draw と単体 draw がビット一致(2 つの独立実装で)。"""
    d = PhiloxDraws(4242)
    ids = np.array([0, 1, 2, 7, 1234, 2**31 - 1, soa.pack_id(3, 1),
                    soa.pack_id(3, 2), soa.pack_id(10**9, 5)],
                   dtype=np.int64).astype(id_dtype).astype(np.int64)
    for channel in ("decide", "move", "日本語チャネル", ""):
        for step in (0, 1, 143, -1, 10**7):
            u = d.uniform(channel, step, ids)
            single = np.array([d.uniform1(channel, step, int(i)) for i in ids])
            assert u.tobytes() == single.tobytes()


def test_draw_is_independent_of_batch_composition_and_order():
    d = PhiloxDraws(9)
    ids = np.arange(1000, 1200, dtype=np.int64)
    full = d.uniform("ch", 5, ids)
    perm = np.random.default_rng(1).permutation(ids.shape[0])
    assert np.array_equal(d.uniform("ch", 5, ids[perm]), full[perm])
    subset = np.array([3, 17, 199, 0], dtype=np.int64)
    assert np.array_equal(d.uniform("ch", 5, ids[subset]), full[subset])
    assert np.array_equal(d.uniform("ch", 5, ids[:1]), full[:1])


def test_channel_step_sub_and_seed_all_separate_streams():
    a, b = PhiloxDraws(1), PhiloxDraws(2)
    ids = np.arange(256, dtype=np.int64)
    base = a.uniform("x", 0, ids)
    assert not np.array_equal(base, a.uniform("y", 0, ids))       # channel
    assert not np.array_equal(base, a.uniform("x", 1, ids))       # step
    assert not np.array_equal(base, a.uniform("x", 0, ids, sub=1))  # sub
    assert not np.array_equal(base, b.uniform("x", 0, ids))       # master_seed
    assert a.key_words("x") != a.key_words("y")
    assert a.key_words("x") != b.key_words("x")


def test_recycled_slot_gets_a_different_stream():
    """席を使い回しても、packed ID に世代が入っているので乱数列は別になる。"""
    d = PhiloxDraws(77)
    old = np.int64(soa.pack_id(3, 1))
    new = np.int64(soa.pack_id(3, 2))
    assert d.uniform1("decide", 0, int(old)) != d.uniform1("decide", 0, int(new))


def test_uniform_range_and_distribution_smoke():
    d = PhiloxDraws(31337)
    u = d.uniform("smoke", 0, np.arange(200_000, dtype=np.int64))
    assert u.min() >= 0.0 and u.max() < 1.0
    assert abs(u.mean() - 0.5) < 0.005
    assert abs(u.var() - 1.0 / 12.0) < 0.002
    assert len(np.unique(u)) == u.shape[0]              # 衝突なし
    # 連番 id でも隣接相関が立たない(counter-based の要件)
    assert abs(np.corrcoef(u[:-1], u[1:])[0, 1]) < 0.01


@pytest.mark.parametrize("n,chunks", [
    (400, (1, 2, 3, 7, 399, 400, 401)),        # 端数・境界ぴったり・1 ブロック
    (9_000, (1000, 4096, 8192, 10**6)),        # 実運用サイズ帯
])
def test_draws_are_independent_of_the_internal_chunk_size(monkeypatch, n, chunks):
    """★ブロック分割はキャッシュ最適化にすぎず、出力はビット完全に不変。"""
    ids = np.arange(n, dtype=np.int64)
    ref_u = PhiloxDraws(6).uniform("ch", 4, ids)
    ref_r = PhiloxDraws(6).raw4("ch", 4, ids)
    ref_4 = PhiloxDraws(6).uniform4("ch", 4, ids)
    for chunk in chunks:
        monkeypatch.setattr(PhiloxDraws, "CHUNK", chunk)
        d = PhiloxDraws(6)
        assert d.uniform("ch", 4, ids).tobytes() == ref_u.tobytes()
        assert d.raw4("ch", 4, ids).tobytes() == ref_r.tobytes()
        assert d.uniform4("ch", 4, ids).tobytes() == ref_4.tobytes()


def test_draws_accept_scalars_lists_and_empty():
    d = PhiloxDraws(6)
    assert d.uniform("ch", 0, np.int64(5)).shape == (1,)
    assert d.uniform("ch", 0, [1, 2, 3]).shape == (3,)
    assert d.uniform("ch", 0, np.zeros(0, dtype=np.int64)).shape == (0,)
    assert d.raw4("ch", 0, np.zeros(0, dtype=np.int64)).shape == (0, 4)
    assert d.uniform("ch", 0, [7])[0] == d.uniform1("ch", 0, 7)


def test_uniform4_shares_the_first_word_with_uniform():
    d = PhiloxDraws(5)
    ids = np.arange(50, dtype=np.int64)
    u4 = d.uniform4("ch", 3, ids)
    assert u4.shape == (50, 4)
    assert np.array_equal(u4[:, 0], d.uniform("ch", 3, ids))
    assert not np.array_equal(u4[:, 0], u4[:, 1])
    assert (u4 >= 0.0).all() and (u4 < 1.0).all()


def test_raw4_shape_and_dtype():
    d = PhiloxDraws(5)
    r = d.raw4("ch", 0, np.arange(3, dtype=np.int64))
    assert r.shape == (3, 4) and r.dtype == np.uint64


def test_bernoulli_and_choice_weights_are_order_independent():
    d = PhiloxDraws(88)
    ids = np.arange(500, dtype=np.int64)
    p = np.linspace(0.0, 1.0, 500)
    bat = d.bernoulli("b", 2, ids, p)
    assert bat.dtype == bool
    assert np.array_equal(bat, d.uniform("b", 2, ids) < p)
    assert bat[0] == np.False_ and bat[-1] == np.True_

    w = np.array([1.0, 3.0, 0.0, 6.0])
    idx = d.choice_weights("c", 2, ids, w)
    assert idx.dtype == np.int64
    assert set(idx.tolist()) <= {0, 1, 3}                # 重み 0 は選ばれない
    assert np.array_equal(d.choice_weights("c", 2, ids[::-1], w), idx[::-1])
    # 行ごとの重みも通る / 行和 0 は 0 を返す
    rw = np.tile(w, (500, 1))
    assert np.array_equal(d.choice_weights("c", 2, ids, rw), idx)
    zero = d.choice_weights("c", 2, ids, np.zeros((500, 4)))
    assert zero.tolist() == [0] * 500
    with pytest.raises(ValueError):
        d.choice_weights("c", 2, ids, -np.ones(4))
    with pytest.raises(ValueError):
        d.choice_weights("c", 2, ids, np.ones((3, 4)))


def test_choice_weights_matches_empirical_frequency():
    d = PhiloxDraws(4)
    ids = np.arange(120_000, dtype=np.int64)
    w = np.array([1.0, 2.0, 7.0])
    idx = d.choice_weights("freq", 0, ids, w)
    counts = np.bincount(idx, minlength=3) / idx.shape[0]
    assert np.allclose(counts, w / w.sum(), atol=0.005)


def test_seed_domain_cannot_collide_with_rnghub():
    """RngHub(SeedSequence + PCG64)とは種ドメインもアルゴリズムも別系統。"""
    from society.rng import RngHub
    hub = RngHub(1234)
    d = PhiloxDraws(1234)
    hub_vals = [float(hub.stream("decide", i, 0).random()) for i in range(32)]
    soa_vals = d.uniform("decide", 0, np.arange(32, dtype=np.int64)).tolist()
    assert hub_vals != soa_vals
    assert not set(hub_vals) & set(soa_vals)


# =========================================================================== #
# 昇格の要求集合セマンティクス
# =========================================================================== #
def test_select_promotions_is_deterministic_and_capped():
    req = np.array([soa.pack_id(s, 1) for s in (5, 1, 9, 1, 3)], dtype=np.int64)
    g, dfr = select_promotions(req)
    assert soa.unpack_id(g)[0].tolist() == [1, 3, 5, 9]      # 重複畳み + 昇順
    assert dfr.shape == (0,)
    g, dfr = select_promotions(req, cap=2)
    assert soa.unpack_id(g)[0].tolist() == [1, 3]
    assert soa.unpack_id(dfr)[0].tolist() == [5, 9]
    # 呼び出し順を入れ替えても同一
    g2, d2 = select_promotions(req[::-1], cap=2)
    assert np.array_equal(g, g2) and np.array_equal(dfr, d2)
    assert select_promotions(req, cap=0)[0].shape == (0,)
    assert select_promotions(np.zeros(0, dtype=np.int64))[0].shape == (0,)


def test_select_promotions_priority_order():
    req = np.array([soa.pack_id(s, 1) for s in (1, 2, 3, 4)], dtype=np.int64)
    pr = np.array([0.1, 0.9, 0.9, 0.2])
    g, dfr = select_promotions(req, cap=2, priority=pr)
    assert soa.unpack_id(g)[0].tolist() == [2, 3]            # 同点は slot 昇順
    assert soa.unpack_id(dfr)[0].tolist() == [4, 1]
    with pytest.raises(ValueError):
        select_promotions(req, cap=2, priority=pr[:2])


# =========================================================================== #
# ⑩ 静的検査(AST)
# =========================================================================== #
def _module_ast() -> ast.Module:
    return ast.parse(MODULE_PATH.read_text(encoding="utf-8"))


def _imported_roots(tree: ast.Module) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
            elif node.level:                                  # 相対 import
                roots.add(".")
    return roots


def test_module_is_thread_free():
    """★スレッド/プロセス並列を一切持ち込まない(決定論の前提)。"""
    forbidden = {"threading", "_thread", "concurrent", "multiprocessing",
                 "asyncio", "queue", "subprocess"}
    assert not (_imported_roots(_module_ast()) & forbidden)
    src = MODULE_PATH.read_text(encoding="utf-8")
    for token in ("Thread(", "ThreadPool", "ProcessPool", "async def", "await "):
        assert token not in src


def test_module_is_self_contained():
    """society の他モジュールに依存しない = 配線前でも単体で成立する土台。"""
    roots = _imported_roots(_module_ast())
    assert roots <= {"__future__", "hashlib", "dataclasses", "typing", "numpy"}


def test_module_uses_no_global_or_implicit_rng():
    """グローバル RNG(np.random.seed / rand / default_rng / random モジュール)を使わない。"""
    src = MODULE_PATH.read_text(encoding="utf-8")
    for token in ("np.random.seed", "np.random.rand", "np.random.random",
                  "default_rng", "np.random.RandomState", "import random"):
        assert token not in src
    # 明示的に許すのは Philox(カウンタベース)と Generator のラッパのみ
    assert "np.random.Philox(" in src


def test_module_declares_no_conf_gate():
    """本モジュールはエンジン未配線 = conf ゲートを持たない(ゲート対象の挙動が無い)。"""
    src = MODULE_PATH.read_text(encoding="utf-8")
    for token in ("load_config", "OmegaConf", "cfg.", "registry"):
        assert token not in src


def test_module_has_no_platform_dependent_dtype_literals():
    """本体コードに `dtype=int` / `np.int_` / `np.intp` が紛れていないこと。

    `dtype=bool`(= np.bool_)は幅が固定なので許す。禁じるのは幅がプラットフォーム
    依存になるもの(`int` は Windows で int32 / Linux で int64)。
    """
    tree = _module_ast()
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "dtype":
            v = node.value
            if isinstance(v, ast.Name) and v.id in ("int", "float"):
                bad.append(ast.dump(node))
            if isinstance(v, ast.Attribute) and v.attr in (
                    "int_", "uint", "intp", "uintp", "float_", "longlong"):
                bad.append(ast.dump(node))
    assert not bad
    # 明示幅の禁止語が「テスト用の許可リスト」経由以外で出てこないこと
    src = MODULE_PATH.read_text(encoding="utf-8")
    assert "dtype=int)" not in src and "dtype=float)" not in src
