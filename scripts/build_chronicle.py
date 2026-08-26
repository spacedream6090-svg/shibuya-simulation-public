#!/usr/bin/env python
"""Shibuya Chronicle ビューワー **P0(縦切り最小版)** のビルドパイプライン。

位置づけ
--------
`docs/plans/viewer-chronicle-plan.md` §1「基盤 P0」の実装。計画の 4 画面のうち
**俯瞰(画面1 の A 側)だけ**を端から端まで動かす縦切りで、以下を作る:

1. **L2 指標の時系列**(全 step)+ **L1 kind 別の毎 step 件数**(会話数など L2 に無い量)
2. **位置 hexbin**(既定 200 m 格子 × 1 時間ビンの**在場者数**)= Uint16 量子化 + base64
3. **関係イベント**(画面2 の素材): `relation_tier` 遷移 / `partner_formed` /
   `acquaint` / `relation_break` を `(step, a, b, tier, ev)` の圧縮表へ
4. 自己完結 HTML `viz/chronicle/chronicle.html`(CDN 禁止・Canvas 2D・遅延なしの一体型)

設計上の約束(計画 §0 の原則をそのまま守る)
--------------------------------------------
- **既存ビューワー(viz/make_viewer*.py)を 1 バイトも触らない**。参照だけする。
  地図背景の OSM タイルの扱いは `make_viewer.py` の `drawBasemap` と同じ流儀
  (origin_latlon 基準の Web メルカトル・オフラインでは静かに無地)。
- **エンジン(src/society/)不触**。読むだけ。走行中のランに影響ゼロ。
- 読みは `scripts/l1_stream.py`(有界メモリ・part 形式の透過連結・完結 part のみ)。
  Δt は `scripts/run_dt.py` から読む(**144 を直書きしない**)。
- 集計はここで完了させ、ビューワーは描画に徹する(計画 §0-5 Kyrix 式)。
- A/B 対照の器を最初から持つ: `--runs A=<dir> B=<dir>` で 2 ラン同梱できる。
  P0 では Run A だけを渡すのが普通で、その場合 B 側は「未取得」と表示される。

使い方
------
    python scripts/build_chronicle.py <run_dir>
    python scripts/build_chronicle.py --runs A=<run_a> B=<run_b>
    python scripts/build_chronicle.py <run_dir> --hex-m 200 --bin-min 60

    # リポジトリの外(集計サーバー等)から回すとき
    SHIBUYA_REPO_ROOT=/path/to/repo python build_chronicle.py <run_dir> --out /tmp/out

出力
----
    viz/chronicle/data/basemap.json     … 街の形(道路/鉄道/ランドマーク・量子化)
    viz/chronicle/data/run_<label>.json … ラン 1 本ぶんの事前集計
    viz/chronicle/data/index.json       … 何が入っているかの索引
    viz/chronicle/chronicle.html        … 上記を全部埋め込んだ自己完結 1 枚

依存は pyarrow + numpy + 標準ライブラリのみ(pandas / duckdb は使わない)。
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path


# --------------------------------------------------------------------------- #
# リポジトリ解決(リポ外から回せるようにする。既定は自分の 2 つ上)
# --------------------------------------------------------------------------- #
def _repo_root(explicit=None) -> Path:
    for cand in (explicit, os.environ.get("SHIBUYA_REPO_ROOT")):
        if cand:
            p = Path(cand).expanduser().resolve()
            if (p / "scripts" / "l1_stream.py").is_file():
                return p
    return Path(__file__).resolve().parents[1]


def _install_paths(root: Path) -> None:
    for p in (root / "scripts", root / "src"):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


# --------------------------------------------------------------------------- #
# 定数
# --------------------------------------------------------------------------- #
SCHEMA = 1

#: 関係イベント(画面2 の素材)。ev コードは HTML 側と共有する。
REL_KINDS = ("relation_tier", "partner_formed", "acquaint", "relation_break")
REL_EV = {"relation_tier": 0, "partner_formed": 1, "acquaint": 2, "relation_break": 3}

#: 時系列に必ず載せる kind(L2 に相当列が無い/物語上の主役)。
#: 実在しない kind は静かに落ちる(run_manifest 駆動の流儀)。
CURATED_KINDS = (
    "conversation", "speak", "hear", "dm", "sns_post", "sns_like", "sns_reshare",
    "relation_tier", "partner_formed", "acquaint", "relation_break",
    "transmission", "vocab_coin", "label_coin", "label_adopt", "vocab_use",
    "arrive", "move_segment", "train_ride", "train_copresence",
    "enter_building", "zone_gate", "joint_activity", "joint_invite",
    "plan_created", "plan_replan", "plan_fallback", "plan_skipped",
    "reflect", "llm_deliberate", "life_event", "belief_update",
    "lost_drop", "lost_return", "gate_pass", "nuisance", "viral_cascade",
)

#: 右パネルのミニチャート既定(source:key)。source は "l2" か "kind"。
DEFAULT_CHARTS = (
    ("l2", "n_fires"),
    ("l2", "n_replies"),
    ("kind", "conversation"),
    ("l2", "edges_formed"),
    ("l2", "distinct_vocab_in_use"),
    ("l2", "silent_agent_rate"),
)

#: 道路クラス → 描画重み(0=最も細い)。地図の縮尺に応じて間引く順序でもある。
ROAD_CLASSES = ("primary", "secondary", "tertiary", "unclassified", "residential",
                "pedestrian", "service", "footway", "path", "steps", "cycleway",
                "corridor", "elevator")

HEX_M_DEFAULT = 200.0
BIN_MIN_DEFAULT = 60
REL_CAP_DEFAULT = 600_000
HTML_SOFT_LIMIT = 20 * 1024 * 1024          # P0 のサイズ規律


# --------------------------------------------------------------------------- #
# 小道具
# --------------------------------------------------------------------------- #
def _b64(arr) -> str:
    """numpy 配列 → base64(little-endian のまま)。"""
    import numpy as np
    a = np.ascontiguousarray(arr)
    if sys.byteorder != "little":
        a = a.byteswap().view(a.dtype.newbyteorder("<"))
    return base64.b64encode(a.tobytes()).decode("ascii")


def _round(v, nd=4):
    """JSON を太らせないための丸め。非有限は None(欠測を偽の値で埋めない)。"""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    r = round(f, nd)
    return int(r) if r == int(r) and abs(r) < 1e15 else r


def _log(msg: str) -> None:
    print(f"[chronicle] {msg}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
# hexbin(pointy-top・axial 座標)
# --------------------------------------------------------------------------- #
class HexGrid:
    """中心間隔 `hex_m` の pointy-top ヘックス格子。numpy でベクトル化する。

    `hex_m` は**隣接ヘックス中心の距離**(= √3 R)。既定 200 m は
    「200 m 格子」という要望をヘックスで読み替えたもの(面積 ≈ 0.0346 km²)。
    """

    __slots__ = ("hex_m", "R")

    def __init__(self, hex_m: float = HEX_M_DEFAULT):
        self.hex_m = float(hex_m)
        self.R = float(hex_m) / math.sqrt(3.0)

    def to_axial(self, x, y):
        """(x, y)[m] → (q, r) の int64 配列。"""
        import numpy as np
        R = self.R
        qf = (math.sqrt(3.0) / 3.0 * x - y / 3.0) / R
        rf = (2.0 / 3.0 * y) / R
        # cube round(x=q, z=r, y=-q-r)
        cz = rf
        cx = qf
        cy = -cx - cz
        rx, ry, rz = np.rint(cx), np.rint(cy), np.rint(cz)
        dx, dy, dz = np.abs(rx - cx), np.abs(ry - cy), np.abs(rz - cz)
        # 丸め誤差が最大の成分を捨てて再構成する(cube round の標準手順)。
        #   c1: q を補正 / dy>dz: y 成分だけ補正 = q, r は据え置き / それ以外: r を補正
        c1 = (dx > dy) & (dx > dz)
        c3 = (~c1) & (dy <= dz)
        qq = np.where(c1, -ry - rz, rx)
        rr = np.where(c3, -rx - ry, rz)
        return qq.astype(np.int64), rr.astype(np.int64)

    def center(self, q: int, r: int):
        R = self.R
        return (R * (math.sqrt(3.0) * q + math.sqrt(3.0) / 2.0 * r), R * (1.5 * r))


_CELL_BIAS = 4096          # q, r を 1 個の int64 に畳むときの下駄(± 4096 まで)


def _encode_cell(q, r):
    return (q + _CELL_BIAS) * (2 * _CELL_BIAS) + (r + _CELL_BIAS)


def _decode_cell(code: int):
    q, r = divmod(int(code), 2 * _CELL_BIAS)
    return q - _CELL_BIAS, r - _CELL_BIAS


# --------------------------------------------------------------------------- #
# L1 の 1 パス集計(hexbin + kind 別 step 件数 + 関係イベント)
# --------------------------------------------------------------------------- #
class _Scan:
    """L1 を 1 回だけ舐めて、俯瞰画面が要る 3 種類の集計を同時に作る。

    メモリは O(在籍体数 + distinct cell + 関係ペア)で有界。40.6 億行級でも
    Python オブジェクトを作るのは**関係イベントの payload だけ**(全体の 0.8%)。
    """

    def __init__(self, hex_m: float, steps_per_bin: int, rel_cap: int,
                 pair_cache_max: int = 8_000_000):
        import numpy as np
        self.np = np
        self.grid = HexGrid(hex_m)
        self.steps_per_bin = max(1, int(steps_per_bin))
        self.rel_cap = int(rel_cap)
        self.pair_cache_max = int(pair_cache_max)

        # hexbin: bin -> {cell_code: 人数} を **2 通り**作る。
        #   bins       … そのビンで実際に位置を出した体だけ(= 観測)
        #   bins_carry … 位置を出さなかった体は**最終既知位置に留まっている**とみなす(= 在場)
        # 夜間は眠っている体が位置イベントを出さないので、観測だけだと街が空になる。
        # どちらが正かは問いによるので、両方作って画面で切り替えられるようにする。
        self.bins: dict[int, Counter] = defaultdict(Counter)
        self.bins_carry: dict[int, Counter] = defaultdict(Counter)
        self.bin_sim_min: dict[int, int] = {}
        # 「その 1 時間ビンの中で最後に観測された位置」を体ごとに持つ
        self._a_cell = np.full(0, -1, dtype=np.int64)
        self._a_bin = np.full(0, -1, dtype=np.int32)
        self._cur_bin = -1

        # kind 別 step 件数
        self.kind_step: dict[str, Counter] = defaultdict(Counter)
        self.kind_total: Counter = Counter()

        # 関係イベント。Python タプルで持つと 1000 万件級で破綻するので、
        # batch ごとに numpy 配列へ畳んで貯める(1 件 13 バイト)。
        self._rel_chunks: list = []
        self.rel_pop = 0                     # 母集団(dedupe 前の該当行数)
        self.rel_kept = 0                    # dedupe 後に保持した件数
        self._pair_tier: dict[int, int] = {}
        self._pair_overflow = False
        self.rel_scan_max = 40_000_000       # 暴走よけの絶対上限(実質かからない)

        self.max_step = -1
        self.min_sim_min = None
        self.max_sim_min = None
        self.rows = 0
        self.pos_rows = 0
        self.agents_seen = 0

    # ---- hexbin ---------------------------------------------------------- #
    def _grow(self, n: int) -> None:
        np = self.np
        if n <= self._a_cell.size:
            return
        new = max(int(n), max(1024, self._a_cell.size * 2))
        c = np.full(new, -1, dtype=np.int64)
        b = np.full(new, -1, dtype=np.int32)
        c[: self._a_cell.size] = self._a_cell
        b[: self._a_bin.size] = self._a_bin
        self._a_cell, self._a_bin = c, b

    def _flush(self, bin_id: int) -> None:
        """`bin_id` の在場ヒストグラムを 2 通り(観測 / 最終既知位置)畳む。"""
        if bin_id < 0:
            return
        np = self.np
        known = self._a_cell >= 0
        obs = known & (self._a_bin == bin_id)
        for mask, tgt in ((obs, self.bins[bin_id]), (known, self.bins_carry[bin_id])):
            cells = self._a_cell[mask]
            if cells.size == 0:
                continue
            u, c = np.unique(cells, return_counts=True)
            for code, n in zip(u.tolist(), c.tolist()):
                tgt[code] += int(n)

    def _positions(self, step_np, sim_np, aid, x, y) -> None:
        np = self.np
        m = np.isfinite(x) & np.isfinite(y) & (aid >= 0)
        if not m.any():
            return
        st = step_np[m]
        ai = aid[m]
        sm_all = sim_np[m] if sim_np is not None else None
        q, r = self.grid.to_axial(x[m], y[m])
        code = _encode_cell(q, r)
        self.pos_rows += int(st.size)
        bins = st // self.steps_per_bin
        self._grow(int(ai.max()) + 1)
        # step 非減少で追記される L1 の性質を使い、ビンが進んだところで畳む。
        for b in np.unique(bins).tolist():
            b = int(b)
            if b > self._cur_bin:
                self._flush(self._cur_bin)
                self._cur_bin = b
            sub = bins == b
            idx = ai[sub]
            # numpy の fancy 代入は重複時「最後の書き込みが残る」= そのビンの最終位置
            self._a_cell[idx] = code[sub]
            self._a_bin[idx] = b
            if sm_all is not None:
                sm = sm_all[sub]
                sm = sm[sm >= 0]
                if sm.size:
                    v = int(sm.min())
                    cur = self.bin_sim_min.get(b)
                    self.bin_sim_min[b] = v if cur is None else min(cur, v)

    # ---- kind × step ----------------------------------------------------- #
    def _kinds(self, batch) -> None:
        import pyarrow as pa
        t = pa.table({"step": batch.column("step"), "kind": batch.column("kind")})
        g = t.group_by(["step", "kind"]).aggregate([("kind", "count")]).to_pydict()
        for s, k, n in zip(g["step"], g["kind"], g["kind_count"]):
            if k is None or s is None:
                continue
            self.kind_step[k][int(s)] += int(n)
            self.kind_total[k] += int(n)

    # ---- 関係イベント ----------------------------------------------------- #
    def _relations(self, batch, idx, kinds) -> None:
        """`idx` は関係 kind の行位置、`kinds` はその kind 名(既に絞り込み済み)。

        payload を Python 文字列にするのは**この行だけ**(全体の 1% 未満)。
        """
        self.rel_pop += int(idx.size)
        if self.rel_kept >= self.rel_scan_max:
            return
        import numpy as np
        import pyarrow as pa
        take = pa.array(idx)
        steps = batch.column("step").take(take).to_pylist()
        aids = batch.column("agent_id").take(take).to_pylist()
        pays = batch.column("payload").take(take).to_pylist()
        o_st: list[int] = []
        o_a: list[int] = []
        o_b: list[int] = []
        o_tv: list[int] = []
        for st, a, kd, raw in zip(steps, aids, kinds, pays):
            if a is None or st is None:
                continue
            try:
                p = json.loads(raw) if raw else {}
            except (json.JSONDecodeError, TypeError):
                continue
            b = p.get("other", p.get("other_id"))
            if b is None:
                continue
            try:
                a, b, st = int(a), int(b), int(st)
            except (TypeError, ValueError):
                continue
            if a < 0 or b < 0:
                continue
            ev = REL_EV[kd]
            if kd == "relation_tier":
                tier = int(p.get("tier") or 0)
                key = (min(a, b) << 32) | max(a, b)
                if not self._pair_overflow:
                    prev = self._pair_tier.get(key)
                    if prev == tier:
                        continue                    # 同じ段のままの再点火は物語ではない
                    self._pair_tier[key] = tier
                    if len(self._pair_tier) > self.pair_cache_max:
                        self._pair_overflow = True  # 以後は素通し(事実として記録する)
            elif kd == "partner_formed":
                tier = 5
            elif kd == "acquaint":
                tier = 1
            else:                                    # relation_break
                tier = int(p.get("to_tier") or 0)
            o_st.append(st)
            o_a.append(a)
            o_b.append(b)
            o_tv.append((ev << 4) | max(0, min(15, tier)))
        if not o_st:
            return
        self._rel_chunks.append((
            np.asarray(o_st, dtype=np.int64), np.asarray(o_a, dtype=np.uint32),
            np.asarray(o_b, dtype=np.uint32), np.asarray(o_tv, dtype=np.uint8)))
        self.rel_kept += len(o_st)

    # ---- 1 batch --------------------------------------------------------- #
    def feed(self, batch) -> None:
        import numpy as np
        import pyarrow.compute as pc
        n = batch.num_rows
        if not n:
            return
        self.rows += n
        names = set(batch.schema.names)

        step_np = pc.fill_null(batch.column("step"), -1).to_numpy(
            zero_copy_only=False).astype(np.int64)
        if step_np.size:
            self.max_step = max(self.max_step, int(step_np.max()))

        sim_np = None
        if "sim_min" in names:
            sim_np = pc.fill_null(batch.column("sim_min"), -1).to_numpy(
                zero_copy_only=False).astype(np.int64)
            ok = sim_np[sim_np >= 0]
            if ok.size:
                lo, hi = int(ok.min()), int(ok.max())
                self.min_sim_min = lo if self.min_sim_min is None else min(self.min_sim_min, lo)
                self.max_sim_min = hi if self.max_sim_min is None else max(self.max_sim_min, hi)

        self._kinds(batch)

        # 関係 kind の判定は **Arrow 側だけ**で済ませる(全行の Python 文字列化を避ける)。
        rel_idx, rel_kinds = None, None
        if "kind" in names and "payload" in names and "agent_id" in names:
            import pyarrow as pa
            kc = batch.column("kind")
            if hasattr(kc, "dictionary") and hasattr(kc, "dictionary_decode"):
                kc = kc.dictionary_decode()
            mask = pc.fill_null(
                pc.is_in(kc, value_set=pa.array(REL_KINDS, pa.string())), False)
            mnp = mask.to_numpy(zero_copy_only=False).astype(bool)
            if mnp.any():
                rel_idx = np.nonzero(mnp)[0]
                rel_kinds = kc.take(pa.array(rel_idx)).to_pylist()

        if "x" in names and "y" in names and "agent_id" in names:
            aid = pc.fill_null(batch.column("agent_id"), -1).to_numpy(
                zero_copy_only=False).astype(np.int64)
            if aid.size:
                self.agents_seen = max(self.agents_seen, int(aid.max()) + 1)
            x = batch.column("x").to_numpy(zero_copy_only=False).astype(np.float64)
            y = batch.column("y").to_numpy(zero_copy_only=False).astype(np.float64)
            self._positions(step_np, sim_np, aid, x, y)

        if rel_idx is not None:
            self._relations(batch, rel_idx, rel_kinds)

    def finish(self) -> None:
        self._flush(self._cur_bin)
        self._cur_bin = -1


def scan_l1(run_dir, *, hex_m: float, steps_per_bin: int, rel_cap: int,
            step_max=None, batch_rows: int = 262_144) -> _Scan:
    """L1 を 1 パスで舐める。`l1_stream` の有界読みだけを使う。"""
    import l1_stream
    sc = _Scan(hex_m, steps_per_bin, rel_cap)
    cols = ["step", "sim_min", "agent_id", "kind", "x", "y", "payload"]
    t0 = time.time()
    last = t0
    for batch in l1_stream.iter_record_batches(run_dir, cols, step_max=step_max,
                                               batch_rows=batch_rows):
        sc.feed(batch)
        if time.time() - last > 20:
            last = time.time()
            _log(f"  L1 {sc.rows:,} 行 / step {sc.max_step} "
                 f"({time.time() - t0:.0f}s)")
    sc.finish()
    _log(f"  L1 走査完了: {sc.rows:,} 行・位置つき {sc.pos_rows:,} 行・"
         f"最終 step {sc.max_step}・{time.time() - t0:.1f}s")
    return sc


# --------------------------------------------------------------------------- #
# hexbin のパック(Uint16 量子化 + base64)
# --------------------------------------------------------------------------- #
def _pack_layer(layer: dict, bins: list, cells: dict, sc: _Scan) -> dict:
    """1 レイヤー(観測 / 在場)を Uint16 量子化 + base64 の疎行列に畳む。"""
    import numpy as np
    gmax = max((max(c.values()) for c in layer.values() if c), default=0)
    scale = max(1, int(math.ceil(gmax / 65535.0)))
    idx_all: list[int] = []
    cnt_all: list[int] = []
    offsets = [0]
    meta = []
    for b in bins:
        c = layer.get(b) or Counter()
        for code in sorted(c):
            idx_all.append(cells[code])
            cnt_all.append(min(65535, int(round(c[code] / scale))))
        offsets.append(len(idx_all))
        meta.append({"bin": int(b), "cells": len(c),
                     "people": int(sum(c.values())),
                     "max": int(max(c.values())) if c else 0})
    n_cells = len(cells)
    return {
        "count_scale": scale,
        "global_max": int(gmax),
        "offsets": offsets,
        "idx": _b64(np.asarray(idx_all,
                               dtype=np.uint16 if n_cells < 65536 else np.uint32)),
        "cnt": _b64(np.asarray(cnt_all, dtype=np.uint16)),
        "bins": meta,
    }


def pack_hexbin(sc: _Scan, grid: HexGrid) -> dict:
    import numpy as np
    bins = sorted(set(sc.bins) | set(sc.bins_carry))
    # 全ビン・全レイヤーを通した cell 辞書(描画側は index だけを持つ)
    cells: dict[int, int] = {}
    order: list[int] = []
    for b in bins:
        for layer in (sc.bins, sc.bins_carry):
            for code in sorted(layer.get(b) or ()):
                if code not in cells:
                    cells[code] = len(order)
                    order.append(code)
    qs = np.empty(len(order), dtype=np.int16)
    rs = np.empty(len(order), dtype=np.int16)
    for i, code in enumerate(order):
        q, r = _decode_cell(code)
        qs[i], rs[i] = q, r

    return {
        "hex_m": grid.hex_m,
        "R": round(grid.R, 4),
        "steps_per_bin": sc.steps_per_bin,
        "n_cells": len(order),
        "n_bins": len(bins),
        "idx_bits": 16 if len(order) < 65536 else 32,
        "q": _b64(qs),
        "r": _b64(rs),
        "bin_meta": [{"bin": int(b), "step0": int(b) * sc.steps_per_bin,
                      "sim_min": sc.bin_sim_min.get(b)} for b in bins],
        "layers": {
            # 既定。位置イベントを出さなかった体は最終既知位置に留まっているとみなす。
            "carry": _pack_layer(sc.bins_carry, bins, cells, sc),
            # そのビンで実際に位置を出した体だけ(= 活動の観測密度)。
            "observed": _pack_layer(sc.bins, bins, cells, sc),
        },
        "layer_labels": {"carry": "在場(最終既知位置を持ち越し)",
                         "observed": "観測(そのビンで位置を出した体のみ)"},
    }


# --------------------------------------------------------------------------- #
# 関係イベントのパック
# --------------------------------------------------------------------------- #
def pack_relations(sc: _Scan) -> dict:
    """関係イベントを `(step, a, b, ev|tier)` の 4 列バイナリへ。

    上限に当たったときは **時系列を切り落とさない**。「昇格の物語に効く行」
    (partner_formed / relation_break / tier>=3)を先に全部残し、残枠を
    それ以外から**等間隔で**間引いて埋める。母集団と掲載数は必ず併記する
    (既存イベントフィードの「母集団 N 件 → 掲載 M 件」の流儀)。
    """
    import numpy as np
    base = {"population": sc.rel_pop,
            "dedupe": "relation_tier は段が変わったときだけ載せる",
            "pair_overflow": sc._pair_overflow,
            "ev_names": list(REL_KINDS)}
    if not sc._rel_chunks:
        return {**base, "shown": 0, "n": 0, "deduped": 0, "capped": False}
    step = np.concatenate([c[0] for c in sc._rel_chunks])
    a = np.concatenate([c[1] for c in sc._rel_chunks])
    b = np.concatenate([c[2] for c in sc._rel_chunks])
    tv = np.concatenate([c[3] for c in sc._rel_chunks])
    sc._rel_chunks.clear()
    deduped = int(step.size)

    order = np.argsort(step, kind="stable")
    step, a, b, tv = step[order], a[order], b[order], tv[order]

    capped = deduped > sc.rel_cap
    if capped:
        ev = tv >> 4
        tier = tv & 0x0F
        keep_pri = (ev != REL_EV["relation_tier"]) | (tier >= 3)
        pri = np.nonzero(keep_pri)[0]
        if pri.size > sc.rel_cap:                    # 主役だけで溢れたら等間隔で
            pri = pri[np.linspace(0, pri.size - 1, sc.rel_cap).astype(np.int64)]
            sel = pri
        else:
            rest = np.nonzero(~keep_pri)[0]
            budget = sc.rel_cap - pri.size
            if rest.size > budget and budget > 0:
                rest = rest[np.linspace(0, rest.size - 1, budget).astype(np.int64)]
            sel = np.sort(np.concatenate([pri, rest[:max(0, budget)]]))
        step, a, b, tv = step[sel], a[sel], b[sel], tv[sel]

    n = int(step.size)
    small = int(step.max()) < 65536 if n else True
    ev_counts = Counter((tv >> 4).tolist())
    return {
        **base,
        "shown": n,
        "n": n,
        "deduped": deduped,
        "capped": capped,
        "step_bits": 16 if small else 32,
        "step": _b64(step.astype(np.uint16 if small else np.uint32)),
        "a": _b64(a),
        "b": _b64(b),
        "tv": _b64(tv),                 # 上位 4bit = ev / 下位 4bit = tier
        "by_ev": {int(k): int(v) for k, v in sorted(ev_counts.items())},
    }


# --------------------------------------------------------------------------- #
# L2 指標の時系列
# --------------------------------------------------------------------------- #
def read_l2(run_dir) -> dict:
    """`l2_metrics` の全数値列を step 昇順の配列にする(part 形式を透過連結)。"""
    import l1_stream
    import pyarrow.parquet as pq
    paths = l1_stream.l1_paths(run_dir, "l2_metrics")
    if not paths:
        return {"steps": [], "cols": {}, "n": 0}
    rows: dict[int, dict] = {}
    names: list[str] = []
    for p in paths:
        with l1_stream._open_shared(p) as fh:
            t = pq.ParquetFile(fh).read()
        d = t.to_pydict()
        if "step" not in d:
            continue
        if not names:
            names = [c for c in t.schema.names if c != "step"]
        for i, s in enumerate(d["step"]):
            if s is None:
                continue
            rows.setdefault(int(s), {}).update(
                {c: d[c][i] for c in names if c in d})
    steps = sorted(rows)
    cols: dict[str, list] = {}
    for c in names:
        vals = [rows[s].get(c) for s in steps]
        if all(v is None or isinstance(v, (int, float, bool)) for v in vals):
            arr = [_round(v, 6) for v in vals]
            if any(v is not None for v in arr):
                cols[c] = arr
    return {"steps": steps, "cols": cols, "n": len(steps)}


# --------------------------------------------------------------------------- #
# 街の形(basemap)
# --------------------------------------------------------------------------- #
def build_basemap(map_path: Path, *, max_pois: int = 220) -> dict:
    import numpy as np
    city = json.loads(Path(map_path).read_text(encoding="utf-8"))
    meta = city.get("meta", {})

    cls_index = {k: i for i, k in enumerate(ROAD_CLASSES)}
    lens: list[int] = []
    cls: list[int] = []
    pts: list[int] = []
    for e in city.get("edges", []):
        g = e.get("geometry") or []
        if len(g) < 2:
            continue
        k = cls_index.get(e.get("klass", "footway"), len(ROAD_CLASSES) - 1)
        lens.append(len(g))
        cls.append(k)
        for px, py in g:
            pts.append(int(round(px)))
            pts.append(int(round(py)))

    rlens: list[int] = []
    rpts: list[int] = []
    for rw in city.get("railways", []):
        g = rw.get("geometry") or []
        if len(g) < 2:
            continue
        rlens.append(len(g))
        for px, py in g:
            rpts.append(int(round(px)))
            rpts.append(int(round(py)))

    # ランドマーク: 名前つき POI を cat 優先度で絞る(母集団 → 掲載を明示する)
    pri = {"station": 0, "landmark": 1, "park": 2, "culture": 3, "shopping": 4}
    pois = [p for p in city.get("pois", []) if (p.get("name") or "").strip()]
    pois.sort(key=lambda p: (pri.get(p.get("cat"), 9), p.get("name", "")))
    shown = pois[:max_pois]

    xs = [n["x"] for n in city.get("nodes", [])] or [0]
    ys = [n["y"] for n in city.get("nodes", [])] or [0]

    return {
        "name": meta.get("name"),
        "map_file": Path(map_path).name,
        "origin": meta.get("origin_latlon"),
        "attribution": meta.get("attribution", "© OpenStreetMap contributors"),
        "extent": [min(xs), min(ys), max(xs), max(ys)],
        "road_classes": list(ROAD_CLASSES),
        "roads": {"n": len(lens), "lens": _b64(np.asarray(lens, dtype=np.uint16)),
                  "cls": _b64(np.asarray(cls, dtype=np.uint8)),
                  "pts": _b64(np.asarray(pts, dtype=np.int16))},
        "rails": {"n": len(rlens), "lens": _b64(np.asarray(rlens, dtype=np.uint16)),
                  "pts": _b64(np.asarray(rpts, dtype=np.int16))},
        "pois": {"population": len(pois), "shown": len(shown),
                 "items": [[int(round(p["x"])), int(round(p["y"])),
                            p.get("cat", ""), p.get("name", "")] for p in shown]},
    }


# --------------------------------------------------------------------------- #
# ラン 1 本ぶんの組み立て
# --------------------------------------------------------------------------- #
def build_run(run_dir: Path, label: str, *, hex_m: float, bin_min: int,
              rel_cap: int, step_max=None) -> dict:
    import l1_stream
    import run_dt
    run_dir = Path(run_dir)
    prov = run_dt.provenance(run_dir)
    dt_min = prov["dt_min"]
    spd = prov["steps_per_day"]
    steps_per_bin = max(1, int(round(bin_min / float(dt_min))))

    manifest = {}
    mp = run_dir / "run_manifest.json"
    if mp.is_file():
        try:
            manifest = json.loads(mp.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            manifest = {}
    rm = manifest.get("run", {}) if isinstance(manifest, dict) else {}

    _log(f"[{label}] {run_dir}  Δt={dt_min}分 (source={prov['source']}) "
         f"/ 1日={spd}step / hexビン={steps_per_bin}step")

    t0 = time.time()
    sc = scan_l1(run_dir, hex_m=hex_m, steps_per_bin=steps_per_bin,
                 rel_cap=rel_cap, step_max=step_max)
    t_scan = time.time() - t0

    t0 = time.time()
    l2 = read_l2(run_dir)
    _log(f"[{label}] L2 {l2['n']} 行 × {len(l2['cols'])} 列 ({time.time() - t0:.1f}s)")

    n_steps_declared = int(rm.get("n_steps") or 0)
    last_step = max(sc.max_step, max(l2["steps"]) if l2["steps"] else -1)
    n_steps = last_step + 1

    # kind 時系列: 収載は「curated ∪ 総数上位30」。母集団は kind_total 全件で示す。
    top = [k for k, _ in sc.kind_total.most_common(30)]
    keep = [k for k in CURATED_KINDS if k in sc.kind_total]
    for k in top:
        if k not in keep:
            keep.append(k)
    kinds_series = {}
    for k in keep:
        c = sc.kind_step[k]
        kinds_series[k] = [int(c.get(s, 0)) for s in range(n_steps)]

    grid = HexGrid(hex_m)
    hexb = pack_hexbin(sc, grid)
    rels = pack_relations(sc)

    parts = [p.name for p in l1_stream.l1_paths(run_dir)]
    return {
        "label": label,
        "run_name": rm.get("name") or run_dir.name,
        "run_dir": str(run_dir),
        "run_mode": manifest.get("run_mode"),
        "scenario": (manifest.get("world") or {}).get("scenario"),
        "model": (manifest.get("model") or {}).get("name"),
        "backend": (manifest.get("model") or {}).get("backend"),
        "started_at": manifest.get("started_at"),
        "dt_min": dt_min,
        "dt_source": prov["source"],
        "steps_per_day": spd,
        "start_tod": rm.get("start_tod") or "00:00",
        "start_date": rm.get("start_date"),
        "n_agents": int(rm.get("n_agents") or sc.agents_seen),
        "n_steps": n_steps,
        "n_steps_declared": n_steps_declared,
        "complete": bool(n_steps_declared and n_steps >= n_steps_declared),
        "l1_parts": len(parts),
        "l1_rows": sc.rows,
        "l1_rows_positioned": sc.pos_rows,
        "sim_min_range": [sc.min_sim_min, sc.max_sim_min],
        "kind_totals": dict(sc.kind_total.most_common()),
        "series": {
            "l2_steps": l2["steps"],
            "l2": l2["cols"],
            "kinds": kinds_series,
            "kinds_population": len(sc.kind_total),
            "kinds_shown": len(kinds_series),
        },
        "hexbin": hexb,
        "relations": rels,
        "build": {"scan_sec": round(t_scan, 1)},
    }


# --------------------------------------------------------------------------- #
# HTML(自己完結・Canvas 2D・CDN 禁止)
# --------------------------------------------------------------------------- #
HTML_TEMPLATE = r"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Shibuya Chronicle — 俯瞰</title>
<style>
:root{--bg:#0a0e14;--panel:rgba(18,22,30,.86);--fg:#e6e9ee;--dim:#9aa4b2;
 --acc:#3b82f6;--bd:rgba(255,255,255,.09);--warn:#f59e0b;}
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);
 font:13px/1.5 -apple-system,"Segoe UI","Hiragino Kaku Gothic ProN",Meiryo,sans-serif;
 overflow:hidden}
#top{display:flex;align-items:center;gap:14px;padding:7px 12px;border-bottom:1px solid var(--bd);
 background:var(--panel);height:44px}
#top h1{font-size:14px;margin:0;font-weight:650;letter-spacing:.02em}
#top .sub{color:var(--dim);font-size:11.5px}
.runsw{display:flex;gap:5px}
.runsw button{background:#151a23;color:var(--dim);border:1px solid var(--bd);border-radius:5px;
 padding:3px 11px;cursor:pointer;font:inherit;font-size:12px}
.runsw button.on{background:var(--acc);color:#fff;border-color:var(--acc)}
.runsw button:disabled{opacity:.42;cursor:not-allowed}
#wrap{display:flex;height:calc(100% - 44px)}
#mapwrap{position:relative;flex:1;min-width:0}
#cv{display:block;width:100%;height:100%;cursor:grab}
#cv.drag{cursor:grabbing}
#side{width:328px;flex:none;border-left:1px solid var(--bd);background:var(--panel);
 overflow-y:auto;padding:10px 11px 26px}
#side h2{font-size:11.5px;color:var(--dim);margin:12px 0 6px;font-weight:600;
 text-transform:uppercase;letter-spacing:.06em}
#side h2:first-child{margin-top:0}
.card{background:#11151d;border:1px solid var(--bd);border-radius:6px;margin-bottom:7px;padding:6px 8px 3px}
.card .hd{display:flex;align-items:baseline;justify-content:space-between;gap:6px;margin-bottom:2px}
.card select{background:#0d1117;color:var(--fg);border:1px solid var(--bd);border-radius:4px;
 font:inherit;font-size:11px;max-width:184px;padding:1px 3px}
.card .val{font-variant-numeric:tabular-nums;font-weight:650;font-size:13px}
.card .mx{color:var(--dim);font-size:10.5px;font-variant-numeric:tabular-nums}
.card canvas{display:block;width:100%;height:46px}
#ctrl{position:absolute;left:12px;right:12px;bottom:12px;background:var(--panel);
 border:1px solid var(--bd);border-radius:8px;padding:8px 12px;display:flex;
 align-items:center;gap:11px;backdrop-filter:blur(6px)}
#ctrl button{background:#1b212c;color:var(--fg);border:1px solid var(--bd);border-radius:5px;
 width:32px;height:28px;cursor:pointer;font:inherit;font-size:13px}
#ctrl input[type=range]{flex:1;accent-color:var(--acc)}
#clock{font-variant-numeric:tabular-nums;font-size:12.5px;min-width:190px}
#clock b{font-size:14px}
#legend{position:absolute;right:12px;top:12px;background:var(--panel);border:1px solid var(--bd);
 border-radius:8px;padding:8px 10px;font-size:11px;min-width:150px}
#legend .bar{height:9px;border-radius:2px;margin:5px 0 3px}
#legend .lb{display:flex;justify-content:space-between;color:var(--dim);
 font-variant-numeric:tabular-nums}
#layers{position:absolute;left:12px;top:12px;background:var(--panel);border:1px solid var(--bd);
 border-radius:8px;padding:7px 10px;font-size:11.5px}
#layers label{display:block;margin:2px 0;cursor:pointer;white-space:nowrap}
#layers input{vertical-align:-1px}
#tip{position:absolute;pointer-events:none;background:#0d1117ee;border:1px solid var(--bd);
 border-radius:5px;padding:5px 8px;font-size:11.5px;display:none;white-space:nowrap;z-index:5}
#attrib{position:absolute;right:12px;bottom:64px;font-size:10px;color:var(--dim);
 background:#0a0e14aa;padding:2px 6px;border-radius:3px}
.kv{display:flex;justify-content:space-between;gap:8px;font-size:11.5px;padding:1.5px 0}
.kv span:last-child{font-variant-numeric:tabular-nums;color:var(--fg)}
.kv span:first-child{color:var(--dim)}
.note{color:var(--dim);font-size:10.8px;line-height:1.45;margin:5px 0 0}
.warn{color:var(--warn)}
</style></head><body>
<div id="top">
  <h1>Shibuya Chronicle</h1>
  <span class="sub" id="runmeta"></span>
  <span style="flex:1"></span>
  <span class="sub">対照:</span>
  <div class="runsw" id="runsw"></div>
</div>
<div id="wrap">
  <div id="mapwrap">
    <canvas id="cv"></canvas>
    <div id="layers">
      <label><input type="checkbox" id="lyBase"> 背景地図(OSM タイル・要オンライン)</label>
      <label><input type="checkbox" id="lyRoad" checked> 道路・鉄道</label>
      <label><input type="checkbox" id="lyHex" checked> 在場者 hexbin</label>
      <label style="padding-left:16px"><input type="checkbox" id="lyCarry" checked>
        最終既知位置を持ち越す</label>
      <label><input type="checkbox" id="lyPoi" checked> ランドマーク<span id="poiN"></span></label>
      <label><input type="checkbox" id="lyLog" checked> 対数スケール</label>
    </div>
    <div id="legend"></div>
    <div id="tip"></div>
    <div id="attrib"></div>
    <div id="ctrl">
      <button id="play" title="再生/一時停止">▶</button>
      <div id="clock"></div>
      <input type="range" id="scrub" min="0" max="0" value="0" step="1">
      <button id="hm" title="1時間戻る">«</button>
      <button id="hp" title="1時間進む">»</button>
    </div>
  </div>
  <div id="side">
    <h2>指標時系列</h2>
    <div id="charts"></div>
    <h2>このランについて</h2>
    <div id="about"></div>
    <h2>関係イベント(画面2の素材)</h2>
    <div id="relbox"></div>
  </div>
</div>
<script id="chronicle-data" type="application/json">__CHRONICLE_DATA__</script>
<script>
"use strict";
const D = JSON.parse(document.getElementById('chronicle-data').textContent);
const BM = D.basemap, RUNS = D.runs, ORDER = D.order;
let RK = ORDER[0];
const R = () => RUNS[RK];

/* ---------- base64 -> TypedArray ---------- */
function bin(b64, Type){
  if(!b64) return new Type(0);
  const s = atob(b64), n = s.length, u = new Uint8Array(n);
  for(let i=0;i<n;i++) u[i] = s.charCodeAt(i);
  return new Type(u.buffer, 0, n / Type.BYTES_PER_ELEMENT);
}

/* ---------- basemap 展開 ---------- */
function polylines(o){
  const lens = bin(o.lens, Uint16Array), pts = bin(o.pts, Int16Array);
  const out = []; let k = 0;
  for(let i=0;i<lens.length;i++){ const m = lens[i];
    out.push(pts.subarray(k, k + m*2)); k += m*2; }
  return out;
}
const ROADS = polylines(BM.roads), RAILS = polylines(BM.rails);
const ROADCLS = bin(BM.roads.cls, Uint8Array);
/* 太さ・色: 幹線ほど明るく太い。歩道は拡大時のみ。
   ヘックスの面(暗い青〜赤)の上に載るので、**明色 + 低不透明度**にして
   どの色の面の上でも骨格が読めるようにする。 */
const RSTYLE = {primary:[2.6,'rgba(226,232,240,.52)'],secondary:[2.1,'rgba(226,232,240,.42)'],
 tertiary:[1.7,'rgba(214,222,235,.34)'],unclassified:[1.3,'rgba(203,213,225,.26)'],
 residential:[1.2,'rgba(203,213,225,.24)'],pedestrian:[1.1,'rgba(203,213,225,.24)'],
 service:[0.9,'rgba(190,200,215,.17)'],footway:[0.7,'rgba(190,200,215,.15)'],
 path:[0.7,'rgba(190,200,215,.15)'],steps:[0.7,'rgba(190,200,215,.15)'],
 cycleway:[0.8,'rgba(190,200,215,.16)'],corridor:[0.7,'rgba(190,200,215,.15)'],
 elevator:[0.7,'rgba(190,200,215,.15)']};
const MINPPM = {primary:0,secondary:0,tertiary:0,unclassified:0.05,residential:0.06,
 pedestrian:0.06,service:0.12,footway:0.14,path:0.14,steps:0.2,cycleway:0.14,
 corridor:0.25,elevator:0.25};

/* ---------- hexbin 展開 ---------- */
function hexOf(run){
  const h = run.hexbin, o = {};
  o.q = bin(h.q, Int16Array); o.r = bin(h.r, Int16Array);
  o.R = h.R; o.hex_m = h.hex_m; o.spb = h.steps_per_bin;
  o.binMeta = h.bin_meta; o.labels = h.layer_labels;
  const IT = h.idx_bits===16 ? Uint16Array : Uint32Array;
  o.L = {};
  for(const k of Object.keys(h.layers)){ const L = h.layers[k];
    o.L[k] = {idx: bin(L.idx, IT), cnt: bin(L.cnt, Uint16Array), off: L.offsets,
              meta: L.bins, scale: L.count_scale, gmax: L.global_max}; }
  o.cx = new Float32Array(o.q.length); o.cy = new Float32Array(o.q.length);
  const S3 = Math.sqrt(3);
  for(let i=0;i<o.q.length;i++){
    o.cx[i] = h.R * (S3*o.q[i] + S3/2*o.r[i]);
    o.cy[i] = h.R * (1.5*o.r[i]);
  }
  o.binOfStep = new Map(); h.bin_meta.forEach((b,i)=>o.binOfStep.set(b.bin, i));
  return o;
}
const HEX = {}; ORDER.forEach(k => HEX[k] = hexOf(RUNS[k]));
function layerKey(){ return document.getElementById('lyCarry').checked ? 'carry' : 'observed'; }
function curLayer(){ return HEX[RK].L[layerKey()]; }

/* ---------- 時間 ---------- */
function nSteps(){ return R().n_steps; }
function startMin(){ const t=(R().start_tod||"00:00").split(':');
  return (+t[0]||0)*60 + (+t[1]||0); }
function simMinAt(s){ return startMin() + s * R().dt_min; }
function clockText(s){
  const spd = R().steps_per_day, day = Math.floor(s/spd);
  const m = ((simMinAt(s) % 1440) + 1440) % 1440;
  const hh = String(Math.floor(m/60)).padStart(2,'0'), mm = String(m%60).padStart(2,'0');
  return {day: day, hhmm: hh+':'+mm};
}
let cur = 0, playing = false, lastT = 0;

/* ---------- カメラ ---------- */
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
let cam = {x:0, y:0, s:0.22};      // s = px per meter
function resize(){
  const dpr = Math.min(2, window.devicePixelRatio||1);
  cv.width = Math.round(cv.clientWidth*dpr); cv.height = Math.round(cv.clientHeight*dpr);
  ctx.setTransform(dpr,0,0,dpr,0,0); draw();
}
function tf(wx, wy){ return [ (wx-cam.x)*cam.s + cv.clientWidth/2,
                              -(wy-cam.y)*cam.s + cv.clientHeight/2 ]; }
function inv(sx, sy){ return [ (sx - cv.clientWidth/2)/cam.s + cam.x,
                              -(sy - cv.clientHeight/2)/cam.s + cam.y ]; }
function fitAll(){
  const e = BM.extent, w = e[2]-e[0], h = e[3]-e[1];
  cam.x = (e[0]+e[2])/2; cam.y = (e[1]+e[3])/2;
  cam.s = Math.min(cv.clientWidth/(w*1.06), cv.clientHeight/(h*1.06));
}

/* ---------- OSM タイル(make_viewer.py drawBasemap と同じ流儀・オフラインでは無地) ---------- */
const tileCache = {};
const LAT0 = BM.origin ? BM.origin[0] : null, LON0 = BM.origin ? BM.origin[1] : null;
function mercPx(lat, lon, z){ const W = 256*Math.pow(2,z);
  const mx = (lon+180)/360*W, s = Math.sin(lat*Math.PI/180);
  return [mx, (0.5 - Math.log((1+s)/(1-s))/(4*Math.PI))*W]; }
function drawBasemap(){
  if(!LAT0) return;
  ctx.save(); ctx.globalAlpha = 0.42;
  const mPerPx = 1/cam.s;
  let z = Math.round(Math.log2(156543.03392*Math.cos(LAT0*Math.PI/180)/mPerPx));
  z = Math.max(13, Math.min(18, z));
  const ppm = 1/(156543.03392*Math.cos(LAT0*Math.PI/180)/Math.pow(2,z));
  const [mx0,my0] = mercPx(LAT0, LON0, z);
  const [wx0,wy0] = inv(0,0), [wx1,wy1] = inv(cv.clientWidth, cv.clientHeight);
  const t0x = Math.floor((mx0+Math.min(wx0,wx1)*ppm)/256), t1x = Math.floor((mx0+Math.max(wx0,wx1)*ppm)/256);
  const t0y = Math.floor((my0-Math.max(wy0,wy1)*ppm)/256), t1y = Math.floor((my0-Math.min(wy0,wy1)*ppm)/256);
  if((t1x-t0x+1)*(t1y-t0y+1) > 64){ ctx.restore(); return; }
  for(let tx=t0x;tx<=t1x;tx++) for(let ty=t0y;ty<=t1y;ty++){
    const key = z+'/'+tx+'/'+ty; let tc = tileCache[key];
    if(!tc || (!tc.ok && performance.now()-tc.t > 6000)){
      tc = tileCache[key] = {img:null, ok:false, t:performance.now()};
      const img = new Image(); img.crossOrigin = 'anonymous';
      img.onload = ()=>{ tc.img=img; tc.ok=true; requestDraw(); };
      img.onerror = ()=>{ tc.ok=false; tc.t=performance.now(); };
      img.src = 'https://tile.openstreetmap.org/'+z+'/'+tx+'/'+ty+'.png';
    }
    if(!tc.ok) continue;
    const [sx,sy] = tf((tx*256-mx0)/ppm, -(ty*256-my0)/ppm);
    const size = 256/ppm*cam.s;
    ctx.drawImage(tc.img, sx, sy, size+1, size+1);
  }
  ctx.restore();
}

/* ---------- 配色(逐次・暗背景で読める) ---------- */
const RAMP = [[13,22,42],[22,54,96],[24,101,140],[36,152,142],[110,196,102],
              [225,222,66],[252,166,44],[241,86,45],[204,26,60]];
function rampColor(t){
  t = Math.max(0, Math.min(1, t)) * (RAMP.length-1);
  const i = Math.min(RAMP.length-2, Math.floor(t)), f = t-i;
  const a = RAMP[i], b = RAMP[i+1];
  return 'rgb('+Math.round(a[0]+(b[0]-a[0])*f)+','+Math.round(a[1]+(b[1]-a[1])*f)+','
              +Math.round(a[2]+(b[2]-a[2])*f)+')';
}

/* ---------- 描画 ---------- */
function curBinIdx(){
  const H = HEX[RK]; if(!H.binMeta.length) return -1;
  const b = Math.floor(cur / H.spb);
  if(H.binOfStep.has(b)) return H.binOfStep.get(b);
  let best = 0;
  for(let i=0;i<H.binMeta.length;i++){ if(H.binMeta[i].bin <= b) best = i; else break; }
  return best;
}
function drawRoads(){
  const ppm = cam.s;
  ctx.lineJoin = 'round'; ctx.lineCap = 'round';
  for(let pass=0; pass<2; pass++){
    for(let i=0;i<ROADS.length;i++){
      const kls = BM.road_classes[ROADCLS[i]] || 'footway';
      const major = (kls==='primary'||kls==='secondary'||kls==='tertiary');
      if((pass===0) !== !major) continue;
      if(ppm < (MINPPM[kls]!==undefined?MINPPM[kls]:0.15)) continue;
      const st = RSTYLE[kls] || [0.7,'#2e3541'];
      ctx.strokeStyle = st[1]; ctx.lineWidth = Math.max(0.4, st[0]*Math.min(2.4, ppm*3.2));
      const p = ROADS[i]; ctx.beginPath();
      let s0 = tf(p[0], p[1]); ctx.moveTo(s0[0], s0[1]);
      for(let j=2;j<p.length;j+=2){ const s = tf(p[j], p[j+1]); ctx.lineTo(s[0], s[1]); }
      ctx.stroke();
    }
  }
  ctx.strokeStyle = '#6b5330'; ctx.setLineDash([7,5]);
  ctx.lineWidth = Math.max(0.6, 1.5*Math.min(2.2, ppm*3.2));
  for(const p of RAILS){ ctx.beginPath();
    let s0 = tf(p[0], p[1]); ctx.moveTo(s0[0], s0[1]);
    for(let j=2;j<p.length;j+=2){ const s = tf(p[j], p[j+1]); ctx.lineTo(s[0], s[1]); }
    ctx.stroke(); }
  ctx.setLineDash([]);
}
function hexPath(sx, sy, rpx){
  ctx.beginPath();
  for(let i=0;i<6;i++){ const a = Math.PI/180*(60*i - 30);
    const px = sx + rpx*Math.cos(a), py = sy + rpx*Math.sin(a);
    if(i===0) ctx.moveTo(px,py); else ctx.lineTo(px,py); }
  ctx.closePath();
}
let hoverCell = null;
function drawHex(){
  const H = HEX[RK], L = curLayer(), bi = curBinIdx(); if(bi < 0) return;
  const a = L.off[bi], b = L.off[bi+1];
  const logv = document.getElementById('lyLog').checked;
  const gmax = Math.max(1, L.gmax);
  const denom = logv ? Math.log1p(gmax) : gmax;
  const rpx = H.R * cam.s;
  ctx.save();
  for(let k=a;k<b;k++){
    const ci = L.idx[k], v = L.cnt[k] * L.scale;
    const [sx,sy] = tf(H.cx[ci], H.cy[ci]);
    if(sx < -rpx*2 || sy < -rpx*2 || sx > cv.clientWidth+rpx*2 || sy > cv.clientHeight+rpx*2) continue;
    const t = (logv ? Math.log1p(v) : v) / denom;
    ctx.fillStyle = rampColor(t);
    // 街の形が読めなくならない濃さに抑える(ヘックスは道路の**下**に敷く)
    ctx.globalAlpha = 0.20 + 0.42*Math.min(1, t*1.5);
    hexPath(sx, sy, rpx*0.985); ctx.fill();
  }
  ctx.restore();
}
function drawHover(){
  if(hoverCell === null) return;
  const H = HEX[RK], rpx = H.R * cam.s;
  const [sx,sy] = tf(H.cx[hoverCell], H.cy[hoverCell]);
  ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.8;
  hexPath(sx, sy, rpx*0.985); ctx.stroke();
}
/* ランドマーク名。重なった札は**描かない**(貼れた数を凡例に出す)。
   items は cat 優先度でソート済みなので、先に来たものが場所を取る。 */
let poiShown = 0;
function drawPois(){
  poiShown = 0;
  const ppm = cam.s; if(ppm < 0.08) return;
  ctx.font = '11px -apple-system,"Segoe UI",Meiryo,sans-serif';
  ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
  const items = BM.pois.items, placed = [];
  const budget = ppm < 0.2 ? 18 : (ppm < 0.45 ? 34 : 70);
  for(let i=0;i<items.length && placed.length<budget;i++){
    const p = items[i], s = tf(p[0], p[1]), sx = s[0], sy = s[1];
    if(sx<20||sy<24||sx>cv.clientWidth-20||sy>cv.clientHeight-70) continue;
    const w = ctx.measureText(p[3]).width + 8;
    const box = [sx-w/2, sy-17, sx+w/2, sy-1];
    let hit = false;
    for(const q of placed){
      if(box[0] < q[2] && box[2] > q[0] && box[1] < q[3] && box[3] > q[1]){ hit = true; break; } }
    if(hit) continue;
    placed.push(box);
    ctx.fillStyle = '#cfd6e0'; ctx.beginPath(); ctx.arc(sx,sy,2.1,0,7); ctx.fill();
    ctx.lineWidth = 3.4; ctx.strokeStyle = 'rgba(10,14,20,.92)';
    ctx.strokeText(p[3], sx, sy-4);
    ctx.fillStyle = '#e6e9ee'; ctx.fillText(p[3], sx, sy-4);
  }
  poiShown = placed.length;
}
function drawScale(){
  const targets = [100,200,500,1000,2000];
  let m = targets[0]; for(const t of targets) if(t*cam.s < 140) m = t;
  const px = m*cam.s, x0 = 16, y0 = cv.clientHeight - 76;
  ctx.strokeStyle = '#e6e9ee'; ctx.lineWidth = 1.4;
  ctx.beginPath(); ctx.moveTo(x0,y0); ctx.lineTo(x0+px,y0);
  ctx.moveTo(x0,y0-4); ctx.lineTo(x0,y0+4);
  ctx.moveTo(x0+px,y0-4); ctx.lineTo(x0+px,y0+4); ctx.stroke();
  ctx.fillStyle = '#e6e9ee'; ctx.font = '11px sans-serif'; ctx.textAlign = 'left';
  ctx.textBaseline = 'bottom'; ctx.fillText(m+' m', x0, y0-6);
}
let rafPending = false;
function requestDraw(){ if(rafPending) return; rafPending = true;
  requestAnimationFrame(()=>{ rafPending = false; draw(); }); }
function draw(){
  ctx.fillStyle = '#0a0e14'; ctx.fillRect(0,0,cv.clientWidth,cv.clientHeight);
  // 敷く順: OSM タイル → hexbin(密度の面)→ 道路・鉄道(街の骨格)→ ランドマーク。
  // 面を骨格の下に置くことで「どこが混んでいるか」と「そこがどこか」が同時に読める。
  if(document.getElementById('lyBase').checked) drawBasemap();
  if(document.getElementById('lyHex').checked) drawHex();
  if(document.getElementById('lyRoad').checked) drawRoads();
  if(document.getElementById('lyHex').checked) drawHover();
  if(document.getElementById('lyPoi').checked) drawPois();
  drawScale();
  document.getElementById('poiN').textContent =
    ' (' + poiShown + '/' + BM.pois.shown + ')';
  drawLegend(); drawCharts(); updateClock();
}

/* ---------- 凡例 ---------- */
function drawLegend(){
  const H = HEX[RK], L = curLayer(), bi = curBinIdx();
  const meta = bi>=0 ? L.meta[bi] : null;
  const other = H.L[layerKey()==='carry' ? 'observed' : 'carry'];
  const om = (bi>=0 && other) ? other.meta[bi] : null;
  const logv = document.getElementById('lyLog').checked;
  let bar = 'linear-gradient(90deg,';
  for(let i=0;i<=10;i++) bar += rampColor(i/10) + (i<10?',':'');
  bar += ')';
  const gmax = L.gmax;
  const mid = logv ? Math.round(Math.expm1(Math.log1p(gmax)*0.5)) : Math.round(gmax/2);
  document.getElementById('legend').innerHTML =
    '<div style="font-weight:600;margin-bottom:1px">'
      + (H.labels ? H.labels[layerKey()] : '在場者数') + ' / ヘックス</div>'
    + '<div style="color:var(--dim);font-size:10.5px">'+H.hex_m+' m 格子・1ビン='
      + (H.spb*R().dt_min) + '分</div>'
    + '<div class="bar" style="background:'+bar+'"></div>'
    + '<div class="lb"><span>0</span><span>'+mid.toLocaleString()+'</span><span>'
      + gmax.toLocaleString()+'</span></div>'
    + (logv?'<div class="note">対数スケール</div>':'')
    + (meta ? '<div class="note">この時間帯: '+meta.cells+' セル / '
        + meta.people.toLocaleString()+' 人・最大 '+meta.max.toLocaleString()
        + (om && layerKey()==='carry'
            ? '<br>うちこの時間帯に位置を出した体 '+om.people.toLocaleString()+' 人' : '')
        + '</div>' : '');
}

/* ---------- ミニチャート ---------- */
const CHARTS = [];
function seriesOf(src, key){
  const r = R();
  if(src === 'l2'){
    const v = r.series.l2[key]; if(!v) return null;
    return {steps: r.series.l2_steps, vals: v, src:'l2', key:key};
  }
  const v = r.series.kinds[key]; if(!v) return null;
  const st = []; for(let i=0;i<v.length;i++) st.push(i);
  return {steps: st, vals: v, src:'kind', key:key};
}
function optionsHTML(sel){
  const r = R(); let h = '<optgroup label="L2 指標">';
  Object.keys(r.series.l2).sort().forEach(k=>{
    h += '<option value="l2:'+k+'"'+(sel==='l2:'+k?' selected':'')+'>'+k+'</option>'; });
  h += '</optgroup><optgroup label="L1 イベント/step">';
  Object.keys(r.series.kinds).sort().forEach(k=>{
    h += '<option value="kind:'+k+'"'+(sel==='kind:'+k?' selected':'')+'>'+k+'</option>'; });
  return h + '</optgroup>';
}
function buildCharts(){
  const host = document.getElementById('charts'); host.innerHTML = '';
  CHARTS.length = 0;
  D.default_charts.forEach((dc, i)=>{
    const sel = dc[0]+':'+dc[1];
    const d = document.createElement('div'); d.className = 'card';
    d.innerHTML = '<div class="hd"><select data-i="'+i+'">'+optionsHTML(sel)+'</select>'
      + '<span class="val"></span></div><canvas></canvas><div class="mx"></div>';
    host.appendChild(d);
    CHARTS.push({sel:sel, el:d, cv:d.querySelector('canvas'),
                 val:d.querySelector('.val'), mx:d.querySelector('.mx')});
    d.querySelector('select').addEventListener('change', e=>{
      CHARTS[+e.target.dataset.i].sel = e.target.value; drawCharts(); });
  });
}
function drawCharts(){
  for(const c of CHARTS){
    const p = c.sel.split(':'), s = seriesOf(p[0], p.slice(1).join(':'));
    const cx = c.cv.getContext('2d');
    const dpr = Math.min(2, window.devicePixelRatio||1);
    const W = c.cv.clientWidth||300, Hh = 46;
    if(c.cv.width !== Math.round(W*dpr)){ c.cv.width = Math.round(W*dpr);
      c.cv.height = Math.round(Hh*dpr); }
    cx.setTransform(dpr,0,0,dpr,0,0);
    cx.clearRect(0,0,W,Hh);
    if(!s){ c.val.textContent = '—'; c.mx.textContent = 'この run には無い列';
      continue; }
    let mn = Infinity, mx = -Infinity;
    for(const v of s.vals){ if(v===null||v===undefined) continue;
      if(v<mn) mn=v; if(v>mx) mx=v; }
    if(!isFinite(mn)){ mn = 0; mx = 1; }
    if(mx === mn){ mn -= 0.5; mx += 0.5; }
    /* 0 起点は「件数」には正しいが「率」には向かない(0.978〜1.000 が
       全部てっぺんに潰れる)。振れ幅が水準に対して小さい系列だけデータ域に切り替える。 */
    const rel = (mx - mn) / Math.max(1e-12, Math.abs(mx));
    const zeroBase = (mn >= 0 && rel > 0.35);
    const pad = (mx - mn) * 0.10;
    const lo = zeroBase ? 0 : (mn - pad);
    const hi = zeroBase ? mx : (mx + pad);
    const span = Math.max(1e-12, hi - lo);
    const total = Math.max(1, nSteps()-1);
    const X = st => (st/total)*W;
    const Y = v => Hh-2 - ((v-lo)/span)*(Hh-6);
    cx.beginPath(); let started = false;
    for(let i=0;i<s.steps.length;i++){ const v = s.vals[i];
      if(v===null||v===undefined) continue;
      const x = X(s.steps[i]), y = Y(v);
      if(!started){ cx.moveTo(x,y); started = true; } else cx.lineTo(x,y); }
    cx.strokeStyle = '#60a5fa'; cx.lineWidth = 1.3; cx.stroke();
    if(started){
      cx.lineTo(X(s.steps[s.steps.length-1]), Hh); cx.lineTo(X(s.steps[0]), Hh);
      cx.closePath(); cx.fillStyle = 'rgba(96,165,250,.16)'; cx.fill();
    }
    const cxp = X(Math.min(cur, total));
    cx.strokeStyle = '#f59e0b'; cx.lineWidth = 1;
    cx.beginPath(); cx.moveTo(cxp,0); cx.lineTo(cxp,Hh); cx.stroke();
    // 現在値 = cur 以下で最も近い観測点(L2 は間引き格子なので直近を採る)
    let j = -1;
    for(let i=0;i<s.steps.length;i++){ if(s.steps[i] <= cur) j = i; else break; }
    const v = j>=0 ? s.vals[j] : null;
    c.val.textContent = (v===null||v===undefined) ? '—' : fmt(v);
    c.mx.textContent = (zeroBase ? '0 – ' : fmt(mn)+' – ') + fmt(mx)
      + (j>=0 && s.steps[j]!==cur ? ' · step '+s.steps[j] : '');
    if(j>=0 && v!==null && v!==undefined){
      cx.fillStyle = '#f59e0b'; cx.beginPath(); cx.arc(X(s.steps[j]), Y(v), 2.2, 0, 7); cx.fill();
    }
  }
}
function fmt(v){
  if(typeof v !== 'number') return String(v);
  if(Math.abs(v) >= 1000) return v.toLocaleString(undefined,{maximumFractionDigits:0});
  if(Number.isInteger(v)) return String(v);
  return v.toFixed(Math.abs(v) < 1 ? 3 : 2);
}

/* ---------- サイドの静的情報 ---------- */
function kv(k, v){ return '<div class="kv"><span>'+k+'</span><span>'+v+'</span></div>'; }
function fillAbout(){
  const r = R();
  let h = '';
  h += kv('ラン', r.run_name);
  h += kv('シナリオ / モード', (r.scenario||'—')+' / '+(r.run_mode||'—'));
  h += kv('モデル', (r.model||'—')+' ('+(r.backend||'—')+')');
  h += kv('人数', (r.n_agents||0).toLocaleString()+' 体');
  h += kv('Δt', r.dt_min+' 分 (1日='+r.steps_per_day+'step)');
  h += kv('step', r.n_steps.toLocaleString()
      + (r.n_steps_declared ? ' / '+r.n_steps_declared.toLocaleString() : ''));
  h += kv('L1', r.l1_rows.toLocaleString()+' 行 ('+r.l1_parts+' part)');
  h += kv('位置つき', r.l1_rows_positioned.toLocaleString()+' 行');
  h += kv('L2 観測', r.series.l2_steps.length+' 点 × '
      + Object.keys(r.series.l2).length+' 列');
  h += kv('kind 時系列', '母集団 '+r.series.kinds_population+' 種 → 掲載 '
      + r.series.kinds_shown+' 種');
  h += kv('ランドマーク', '母集団 '+BM.pois.population+' 件 → 同梱 '+BM.pois.shown+' 件');
  if(!r.complete && r.n_steps_declared){
    h += '<div class="note warn">⚠ このランは走行中(宣言 '+r.n_steps_declared
       + ' step のうち '+r.n_steps+' step ぶんの確定 part を集計)。</div>';
  }
  h += '<div class="note">在場者 = その 1 時間ビンで最後に観測された位置。'
     + '位置を 1 度も出さなかった体はそのビンに数えない(欠測を推定で埋めない)。'
     + 'ランドマークの札は重なったぶんを描かない(左上に「画面/同梱」で表示)。</div>';
  document.getElementById('about').innerHTML = h;
}
function fillRel(){
  const rl = R().relations; let h = '';
  h += kv('母集団', (rl.population||0).toLocaleString()+' 件');
  h += kv('遷移だけ残す', (rl.deduped||0).toLocaleString()+' 件');
  h += kv('掲載', (rl.shown||0).toLocaleString()+' 件');
  if(rl.by_ev){ const nm = rl.ev_names||[];
    Object.keys(rl.by_ev).forEach(k=>{ h += kv('· '+(nm[+k]||k), rl.by_ev[k].toLocaleString()); }); }
  h += '<div class="note">'+(rl.dedupe||'')+'</div>';
  if(rl.capped) h += '<div class="note warn">上限超過。時系列は切らず、'
    + 'partner_formed / relation_break / tier≥3 を全部残したうえで'
    + '残りを等間隔で間引いた。</div>';
  h += '<div class="note">P0 ではデータのみ同梱(描画は画面2 で使う)。</div>';
  document.getElementById('relbox').innerHTML = h;
}
function updateClock(){
  const c = clockText(cur);
  document.getElementById('clock').innerHTML =
    '<b>Day '+(c.day+1)+' '+c.hhmm+'</b> <span style="color:var(--dim)">step '
    + cur + ' / ' + (nSteps()-1) + '</span>';
}
function fillTop(){
  const r = R();
  document.getElementById('runmeta').textContent =
    r.run_name + ' · ' + (r.n_agents||0).toLocaleString() + ' 体 · '
    + BM.name + ' · ' + (r.started_at||'');
  const sw = document.getElementById('runsw'); sw.innerHTML = '';
  for(const k of ['A','B']){
    const b = document.createElement('button');
    b.textContent = 'Run ' + k + (RUNS[k] ? '' : ' (未取得)');
    b.disabled = !RUNS[k];
    if(k === RK) b.className = 'on';
    b.onclick = ()=>{ if(!RUNS[k]) return; RK = k; cur = Math.min(cur, nSteps()-1);
      document.getElementById('scrub').max = nSteps()-1;
      fillTop(); fillAbout(); fillRel(); buildCharts(); draw(); };
    sw.appendChild(b);
  }
  document.getElementById('attrib').textContent = BM.attribution
    + ' · 地図タイル © OpenStreetMap contributors';
}

/* ---------- 操作 ---------- */
const scrub = document.getElementById('scrub');
scrub.addEventListener('input', ()=>{ cur = +scrub.value; requestDraw(); });
document.getElementById('play').addEventListener('click', ()=>{
  playing = !playing; document.getElementById('play').textContent = playing ? '❚❚' : '▶';
  lastT = performance.now(); if(playing) requestAnimationFrame(tick); });
function tick(t){
  if(!playing) return;
  if(t - lastT > 90){ lastT = t; cur = (cur + 1) % nSteps(); scrub.value = cur; draw(); }
  requestAnimationFrame(tick);
}
document.getElementById('hm').onclick = ()=>{ const s = HEX[RK].spb;
  cur = Math.max(0, cur - s); scrub.value = cur; requestDraw(); };
document.getElementById('hp').onclick = ()=>{ const s = HEX[RK].spb;
  cur = Math.min(nSteps()-1, cur + s); scrub.value = cur; requestDraw(); };
['lyBase','lyRoad','lyHex','lyCarry','lyPoi','lyLog'].forEach(id=>
  document.getElementById(id).addEventListener('change', requestDraw));

let drag = null;
cv.addEventListener('mousedown', e=>{ drag = {x:e.clientX, y:e.clientY,
  cx:cam.x, cy:cam.y}; cv.classList.add('drag'); });
window.addEventListener('mouseup', ()=>{ drag = null; cv.classList.remove('drag'); });
window.addEventListener('mousemove', e=>{
  if(drag){ cam.x = drag.cx - (e.clientX-drag.x)/cam.s;
            cam.y = drag.cy + (e.clientY-drag.y)/cam.s; requestDraw(); return; }
});
cv.addEventListener('mousemove', e=>{
  const rect = cv.getBoundingClientRect();
  const [wx,wy] = inv(e.clientX-rect.left, e.clientY-rect.top);
  const H = HEX[RK], L = curLayer(), bi = curBinIdx(); hoverCell = null;
  const tip = document.getElementById('tip');
  if(bi >= 0){
    const a = L.off[bi], b = L.off[bi+1]; let best = -1, bd = H.R*H.R;
    for(let k=a;k<b;k++){ const ci = L.idx[k];
      const dx = H.cx[ci]-wx, dy = H.cy[ci]-wy, d = dx*dx+dy*dy;
      if(d < bd){ bd = d; best = k; } }
    if(best >= 0){
      hoverCell = L.idx[best];
      const v = L.cnt[best]*L.scale;
      const area = 2.598*H.R*H.R/1e6;
      tip.style.display = 'block';
      tip.style.left = (e.clientX-rect.left+14)+'px';
      tip.style.top = (e.clientY-rect.top+12)+'px';
      tip.innerHTML = '<b>'+v.toLocaleString()+' 人</b> / '+H.hex_m+'m ヘックス<br>'
        + '<span style="color:var(--dim)">'+Math.round(v/area).toLocaleString()
        + ' 人/km² · '+clockText(H.binMeta[bi].step0).hhmm+'台 · '
        + (layerKey()==='carry'?'在場':'観測')+'</span>';
    } else tip.style.display = 'none';
  } else tip.style.display = 'none';
  requestDraw();
});
cv.addEventListener('mouseleave', ()=>{ hoverCell = null;
  document.getElementById('tip').style.display = 'none'; requestDraw(); });
cv.addEventListener('wheel', e=>{
  e.preventDefault();
  const rect = cv.getBoundingClientRect();
  const [wx,wy] = inv(e.clientX-rect.left, e.clientY-rect.top);
  const k = Math.exp(-e.deltaY*0.0016);
  const ns = Math.max(0.03, Math.min(6, cam.s*k));
  cam.x = wx - (wx-cam.x)*(cam.s/ns); cam.y = wy - (wy-cam.y)*(cam.s/ns);
  cam.s = ns; requestDraw();
}, {passive:false});
cv.addEventListener('dblclick', ()=>{ fitAll(); requestDraw(); });
window.addEventListener('resize', resize);
window.addEventListener('keydown', e=>{
  if(e.key === 'ArrowRight'){ cur = Math.min(nSteps()-1, cur+1); scrub.value = cur; requestDraw(); }
  if(e.key === 'ArrowLeft'){ cur = Math.max(0, cur-1); scrub.value = cur; requestDraw(); }
  if(e.key === ' '){ e.preventDefault(); document.getElementById('play').click(); }
});

/* ---------- 起動(#run=A&step=54 で「その瞬間」を直接開ける) ---------- */
(function(){
  const h = new URLSearchParams(location.hash.replace(/^#/, ''));
  const rk = (h.get('run')||'').toUpperCase();
  if(RUNS[rk]) RK = rk;
  const s = parseInt(h.get('step'), 10);
  if(isFinite(s)) cur = Math.max(0, Math.min(nSteps()-1, s));
})();
function syncHash(){
  try { history.replaceState(null, '', '#run='+RK+'&step='+cur); } catch(_){}
}
scrub.addEventListener('change', syncHash);
scrub.max = nSteps()-1; scrub.value = cur;
fillTop(); fillAbout(); fillRel(); buildCharts();
fitAll(); resize();
</script></body></html>
"""


def write_html(payload: dict, out_html: Path) -> int:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    # </script> と行区切り記号だけ潰す(JSON 文字列としての意味は変わらない)
    data = (data.replace("</", "<\\/")
                .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))
    html = HTML_TEMPLATE.replace("__CHRONICLE_DATA__", data)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    return len(html.encode("utf-8"))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def _parse_runs(args) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    if args.runs:
        for spec in args.runs:
            if "=" in spec:
                lb, d = spec.split("=", 1)
            else:
                lb, d = ("A" if not out else "B"), spec
            out.append((lb.strip().upper(), Path(d).expanduser()))
    for d in (args.run_dir or []):
        lb = "A" if not out else "B"
        out.append((lb, Path(d).expanduser()))
    if not out:
        raise SystemExit("run dir を 1 つ以上指定してください(位置引数か --runs A=<dir>)")
    if len(out) > 2:
        raise SystemExit("P0 が受けるのは最大 2 ラン(A/B)です")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Shibuya Chronicle P0(俯瞰)のデータと自己完結 HTML を作る")
    ap.add_argument("run_dir", nargs="*", help="ランのディレクトリ(1〜2 本)")
    ap.add_argument("--runs", nargs="+", metavar="LABEL=DIR",
                    help="A/B 対照。例: --runs A=runs/x B=runs/y")
    ap.add_argument("--out", default=None, help="データ出力先(既定 viz/chronicle/data)")
    ap.add_argument("--html", default=None, help="HTML 出力先(既定 viz/chronicle/chronicle.html)")
    ap.add_argument("--map", default=None, help="街の JSON(既定 run_manifest の world.map)")
    ap.add_argument("--hex-m", type=float, default=HEX_M_DEFAULT,
                    help=f"ヘックス中心間隔[m](既定 {HEX_M_DEFAULT:g})")
    ap.add_argument("--bin-min", type=int, default=BIN_MIN_DEFAULT,
                    help=f"時間ビン[分](既定 {BIN_MIN_DEFAULT})")
    ap.add_argument("--rel-cap", type=int, default=REL_CAP_DEFAULT,
                    help=f"関係イベントの掲載上限(既定 {REL_CAP_DEFAULT})。"
                         "HTML の最大の項。A/B 2 ラン × 10 日を 1 枚に入れるなら"
                         "25 万前後まで下げること(1 件 ≒ 17 バイト)")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="この step までで打ち切る(検証用)")
    ap.add_argument("--repo-root", default=None, help="リポジトリの場所(リポ外実行用)")
    ap.add_argument("--html-only", action="store_true",
                    help="既存の data/*.json から HTML だけ作り直す(L1 を 1 行も読まない)")
    a = ap.parse_args(argv)

    root = _repo_root(a.repo_root)
    _install_paths(root)
    _log(f"repo = {root}")

    out_dir = Path(a.out).expanduser() if a.out else (root / "viz" / "chronicle" / "data")
    out_html = Path(a.html).expanduser() if a.html else (out_dir.parent / "chronicle.html")

    if a.html_only:
        idx_p = out_dir / "index.json"
        if not idx_p.is_file():
            raise SystemExit(f"--html-only には既存の {idx_p} が要ります")
        idx = json.loads(idx_p.read_text(encoding="utf-8"))
        payload = {
            "schema": SCHEMA,
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "basemap": json.loads((out_dir / idx["basemap"]).read_text(encoding="utf-8")),
            "runs": {k: json.loads((out_dir / v["file"]).read_text(encoding="utf-8"))
                     for k, v in idx["runs"].items()},
            "order": idx["order"],
            "default_charts": [list(c) for c in DEFAULT_CHARTS],
        }
        size = write_html(payload, out_html)
        _log(f"HTML(再生成のみ): {out_html}  {size/1024/1024:.2f} MB")
        print(json.dumps({"html": str(out_html), "html_bytes": size,
                          "mode": "html-only"}, ensure_ascii=False, indent=2))
        return 0

    pairs = _parse_runs(a)
    out_dir.mkdir(parents=True, exist_ok=True)

    # basemap: 明示指定 > 先頭ランの manifest > 既定
    map_path = None
    if a.map:
        map_path = Path(a.map).expanduser()
    else:
        mp = pairs[0][1] / "run_manifest.json"
        if mp.is_file():
            try:
                rel = (json.loads(mp.read_text(encoding="utf-8"))
                       .get("world", {}).get("map"))
                if rel:
                    cand = Path(rel)
                    map_path = cand if cand.is_absolute() else (root / rel)
            except (OSError, ValueError):
                pass
    if map_path is None or not map_path.is_file():
        raise SystemExit(f"街の JSON が見つかりません(--map で指定してください): {map_path}")

    t0 = time.time()
    basemap = build_basemap(map_path)
    _log(f"basemap: {map_path.name} 道路 {basemap['roads']['n']} 本 / "
         f"鉄道 {basemap['rails']['n']} 本 / POI {basemap['pois']['shown']}"
         f"({basemap['pois']['population']} 中) — {time.time() - t0:.1f}s")

    runs: dict[str, dict] = {}
    order: list[str] = []
    timings: dict[str, float] = {}
    for label, rd in pairs:
        if not rd.is_dir():
            raise SystemExit(f"run dir が見つかりません: {rd}")
        t1 = time.time()
        runs[label] = build_run(rd, label, hex_m=a.hex_m, bin_min=a.bin_min,
                                rel_cap=a.rel_cap, step_max=a.max_steps)
        timings[label] = round(time.time() - t1, 1)
        order.append(label)

    payload = {
        "schema": SCHEMA,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "basemap": basemap,
        "runs": runs,
        "order": order,
        "default_charts": [list(c) for c in DEFAULT_CHARTS],
    }

    # data/ には素材を分けて置く(画面2 以降が HTML を経由せず読めるように)
    (out_dir / "basemap.json").write_text(
        json.dumps(basemap, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    for label, r in runs.items():
        (out_dir / f"run_{label}.json").write_text(
            json.dumps(r, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    index = {
        "schema": SCHEMA, "built_at": payload["built_at"],
        "basemap": "basemap.json",
        "runs": {k: {"file": f"run_{k}.json", "run_name": r["run_name"],
                     "n_steps": r["n_steps"], "n_agents": r["n_agents"],
                     "complete": r["complete"], "build_sec": timings[k]}
                 for k, r in runs.items()},
        "order": order,
        "params": {"hex_m": a.hex_m, "bin_min": a.bin_min, "rel_cap": a.rel_cap},
    }
    (out_dir / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    size = write_html(payload, out_html)
    # どこにバイトが行ったかを毎回出す(規律は「測ってから切る」)。
    for label, r in runs.items():
        def _n(o):
            return len(json.dumps(o, ensure_ascii=False,
                                  separators=(",", ":")).encode("utf-8"))
        _log(f"[{label}] 内訳: hexbin {_n(r['hexbin'])/1024:.0f}KB / "
             f"series {_n(r['series'])/1024:.0f}KB / "
             f"relations {_n(r['relations'])/1024:.0f}KB")
    _log(f"HTML: {out_html}  {size/1024/1024:.2f} MB "
         f"(規律 {HTML_SOFT_LIMIT/1024/1024:.0f}MB)")
    if size > HTML_SOFT_LIMIT:
        _log("⚠ サイズ規律 超過。効く順に: --rel-cap を小さく(最大の項)/ "
             "--bin-min を長く / --hex-m を大きく。")

    print(json.dumps({
        "html": str(out_html), "html_bytes": size,
        "data_dir": str(out_dir),
        "files": {p.name: p.stat().st_size for p in sorted(out_dir.glob("*.json"))},
        "runs": {k: {"n_steps": r["n_steps"], "l1_rows": r["l1_rows"],
                     "hex_bins": r["hexbin"]["n_bins"], "hex_cells": r["hexbin"]["n_cells"],
                     "hex_max": {ln: lv["global_max"]
                                 for ln, lv in r["hexbin"]["layers"].items()},
                     "rel_population": r["relations"]["population"],
                     "rel_deduped": r["relations"].get("deduped", 0),
                     "rel_shown": r["relations"].get("shown", 0),
                     "build_sec": timings[k]} for k, r in runs.items()},
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":                              # pragma: no cover
    raise SystemExit(main())
