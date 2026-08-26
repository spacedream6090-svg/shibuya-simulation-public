#!/usr/bin/env python
"""Shibuya Chronicle ビューワーのビルドパイプライン(P0 俯瞰 + 画面2 関係の伝記)。

位置づけ
--------
`docs/plans/viewer-chronicle-plan.md` §1 の実装。計画の 4 画面のうち **俯瞰(画面1 の
A 側)** と **画面2「関係の伝記」** を端から端まで動かす。作るもの:

1. **L2 指標の時系列**(全 step)+ **L1 kind 別の毎 step 件数**(会話数など L2 に無い量)
2. **位置 hexbin**(既定 200 m 格子 × 1 時間ビンの**在場者数**)= Uint16 量子化 + base64
3. **関係イベント**: `relation_tier` 遷移 / `partner_formed` / `acquaint` /
   `relation_break` を `(step, a, b, tier, ev)` の圧縮表へ
4. **注目ペアの伝記**(画面2): 全ペアを母集団に、カテゴリ枠 + ドラマ性スコアで
   数百組を選抜し、1 組ぶんの「段の遷移 / 出会いの文脈 / 会話 / **実文** /
   同伴 / closeness / 2 人の軌跡」を同梱する。
5. 自己完結 HTML `viz/chronicle/chronicle.html`(CDN 禁止・Canvas 2D・遅延なしの一体型)

画面2 が読む素材(1 パス目に相乗り)
------------------------------------
`speak`(実文 + hearers)/ `dm`(実文 + to)/ `conversation`(topic・tone・outcome・
scene)/ `joint_activity` / `acquaint` / `train_copresence`。位置の軌跡だけは
**選抜が決まってからでないと誰の分を採るか決まらない**ので 2 パス目で拾う
(`payload` 列を読まないので 1 パス目の 1/20 の時間)。closeness は
`relations.parquet`(G5 日次差分)、名前は `roster.parquet` から選抜ぶんだけ引く。

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

#: 画面2「関係の伝記」の物語素材(2 人のあいだに何が起きたか)。
#: - speak  … 実文 + hearers(= 誰に向かって言ったか)。`hear` は本文を持たないので読まない。
#: - dm     … 実文 + to。Run A では to が null の行が 253/260(= 宛先不明の独り言)。
#: - conversation … 構造化会話 C2(実文なし)。{with, topic, tone, outcome, scene, acts}
#: - joint_activity … 同伴(step 0 の party は初期条件なので読まない。§build_pairs 参照)
#: - acquaint / train_copresence … 出会いの文脈(宣言的な知り合い / 車内の他人)
NARR_KINDS = ("speak", "dm", "conversation", "joint_activity",
              "acquaint", "train_copresence")

#: 会話チャネル(実文つき)。HTML 側と共有。
CH_SPEAK, CH_DM = 0, 1

#: 注目ペアの分類札(HTML の凡例と 1 対 1)。
PAIR_TAGS = ("familiar_strangers", "break", "fast_promote", "rich_dialogue",
             "outing", "partner", "acquaint")
PAIR_TAG_LABELS = {
    "familiar_strangers": "車内の他人 → 関係",
    "break": "こじれた関係",
    "fast_promote": "速い昇格",
    "rich_dialogue": "実文が多い",
    "outing": "一緒に出かけた",
    "partner": "パートナー",
    "acquaint": "宣言的な知り合い",
}
#: 掲載枠(カテゴリごとの最低保証。合計は cap を超えない)。
PAIR_QUOTA = {"familiar_strangers": 16, "break": 60, "fast_promote": 72,
              "rich_dialogue": 110, "outing": 24, "partner": 24, "acquaint": 30}

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
PAIR_CAP_DEFAULT = 360                      # 掲載するペア数(母集団は必ず併記する)
PAIR_CONV_DEFAULT = 16                      # 1 ペアあたりの会話掲載数
PAIR_TRAJ_DEFAULT = 480                     # 1 ペアあたりの軌跡サンプル数
CONV_CAP_DEFAULT = 24_000_000               # conversation を貯める上限(1 件 23B)
TEXT_CAP_DEFAULT = 400_000                  # 実文の distinct 上限(1 件 ~150B)
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


class _Intern:
    """文字列 → 連番 id(欠測は -1 のまま持ち上げる)。JSON を太らせないための辞書化。"""

    __slots__ = ("vals", "ix", "cap", "dropped")

    def __init__(self, cap: int = 0):
        self.vals: list = []
        self.ix: dict = {}
        self.cap = int(cap)             # 0 = 無制限
        self.dropped = 0

    def __call__(self, v):
        if v is None:
            return -1
        i = self.ix.get(v)
        if i is None:
            if self.cap and len(self.vals) >= self.cap:
                self.dropped += 1
                return -1
            i = len(self.vals)
            self.ix[v] = i
            self.vals.append(v)
        return i

    def __len__(self):
        return len(self.vals)


def _pkey(a: int, b: int) -> int:
    """無向ペア → int64 の鍵(小さい方を上位に置く)。"""
    return (a << 32) | b if a <= b else (b << 32) | a


def _xy16(v):
    """世界座標[m] → int16(範囲外・非有限は -32768 = 欠測)。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return -32768
    if not math.isfinite(f) or f <= -32767 or f >= 32767:
        return -32768
    return int(round(f))


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
                 pair_cache_max: int = 8_000_000, *, narrative: bool = True,
                 conv_cap: int = CONV_CAP_DEFAULT, text_cap: int = TEXT_CAP_DEFAULT):
        import numpy as np
        self.np = np
        self.grid = HexGrid(hex_m)
        self.steps_per_bin = max(1, int(steps_per_bin))
        self.rel_cap = int(rel_cap)
        self.pair_cache_max = int(pair_cache_max)

        # ---- 画面2「関係の伝記」の素材(narrative=False で 1 行も読まない) ---- #
        self.narrative = bool(narrative)
        self.conv_cap = int(conv_cap)
        self._conv_chunks: list = []       # (step,a,b,topic,tone,outcome,scene,x,y)
        self.conv_pop = 0                  # 母集団(該当行数)
        self.conv_kept = 0
        self._txt_chunks: list = []        # (step,a,b,speaker,text_id,x,y,ch)
        self.text_pop = 0                  # speak/dm の母集団行数
        self.text_pair_rows = 0            # ペア × 実文 の行数
        self.texts = _Intern(text_cap)     # 実文そのもの
        self.scenes = _Intern()            # conversation の場面文字列
        self.topics, self.tones, self.outcomes = _Intern(), _Intern(), _Intern()
        self.ja_types, self.ja_places = _Intern(), _Intern()
        self.tc_lines, self.acq_vias = _Intern(), _Intern()
        self.brk_causes = _Intern()
        self.acq: dict[int, tuple] = {}    # pair -> (step, via_code)
        self.tc: dict[int, tuple] = {}     # pair -> (step, line_code, car)
        self.ja: dict[int, list] = defaultdict(list)   # pair -> [(step,type,place)]
        self.ja_pop = 0                    # joint_activity の母集団行数(step 0 込み)
        self.ja_kept = 0                   # 読んだ行数(step>0 のみ)
        self.brk: dict[int, list] = defaultdict(list)  # pair -> [(step,from,to,cause)]

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
                key = (min(a, b) << 32) | max(a, b)
                # ★悪化も「段が動いた」= 直近段の更新。ここを更新しないと
                #   「こじれて他人に戻り → もう一度知り合いになった」の 2 回目が
                #   「同じ段のままの再点火」と誤判定されて消え、段の往復が読めなくなる。
                if not self._pair_overflow and key in self._pair_tier:
                    self._pair_tier[key] = tier
                if self.narrative:
                    # 「どこから どこへ なぜ」は tier コードに入りきらないので別台帳へ。
                    row = (st, int(p.get("from_tier") or 0), tier,
                           self.brk_causes(p.get("cause")))
                    lst = self.brk[key]
                    if not lst or lst[-1] != row:    # 双方向ログの片割れを落とす
                        lst.append(row)
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

    # ---- 物語素材(画面2) ------------------------------------------------- #
    def _narratives(self, batch, idx, kinds, xnp, ynp) -> None:
        """2 人のあいだに起きたことを、ペア鍵つきの圧縮表へ落とす。

        conversation は Run A で 158.8 万件ある(全 L1 の 3.9%)。ここで Python
        オブジェクトになるのはこの 3.9% だけで、残り 96.1% は Arrow のまま素通りする。
        場面・話題・語調・結末は**辞書化**(文字列は 1 回だけ持つ)。
        """
        import numpy as np
        import pyarrow as pa
        take = pa.array(idx)
        steps = batch.column("step").take(take).to_pylist()
        aids = batch.column("agent_id").take(take).to_pylist()
        pays = batch.column("payload").take(take).to_pylist()
        xs = xnp[idx].tolist() if xnp is not None else [None] * len(steps)
        ys = ynp[idx].tolist() if ynp is not None else [None] * len(steps)

        c_st: list[int] = []
        c_a: list[int] = []
        c_b: list[int] = []
        c_tp: list[int] = []
        c_tn: list[int] = []
        c_oc: list[int] = []
        c_sc: list[int] = []
        c_x: list[int] = []
        c_y: list[int] = []
        t_st: list[int] = []
        t_a: list[int] = []
        t_b: list[int] = []
        t_sp: list[int] = []
        t_tx: list[int] = []
        t_x: list[int] = []
        t_y: list[int] = []
        t_ch: list[int] = []

        for st, a, kd, raw, x, y in zip(steps, aids, kinds, pays, xs, ys):
            if a is None or st is None:
                continue
            try:
                p = json.loads(raw) if raw else {}
            except (json.JSONDecodeError, TypeError):
                continue
            a, st = int(a), int(st)
            xi, yi = _xy16(x), _xy16(y)

            if kd == "conversation":
                self.conv_pop += 1
                w = p.get("with")
                if w is None or self.conv_kept >= self.conv_cap:
                    continue
                try:
                    b = int(w)
                except (TypeError, ValueError):
                    continue
                if b < 0:
                    continue
                c_st.append(st)
                c_a.append(min(a, b))
                c_b.append(max(a, b))
                c_tp.append(self.topics(p.get("topic")) + 1)
                c_tn.append(self.tones(p.get("tone")) + 1)
                c_oc.append(self.outcomes(p.get("outcome")) + 1)
                c_sc.append(self.scenes(p.get("scene")))
                c_x.append(xi)
                c_y.append(yi)

            elif kd == "speak":
                self.text_pop += 1
                hs = p.get("hearers") or []
                if not hs:
                    continue
                tid = self.texts(p.get("text") or "")
                if tid < 0:
                    continue
                for h in hs:
                    try:
                        b = int(h)
                    except (TypeError, ValueError):
                        continue
                    if b < 0 or b == a:
                        continue
                    t_st.append(st)
                    t_a.append(min(a, b))
                    t_b.append(max(a, b))
                    t_sp.append(a)
                    t_tx.append(tid)
                    t_x.append(xi)
                    t_y.append(yi)
                    t_ch.append(CH_SPEAK)

            elif kd == "dm":
                self.text_pop += 1
                to = p.get("to")
                if to is None:                 # Run A では 253/260 が宛先 null
                    continue
                try:
                    b = int(to)
                except (TypeError, ValueError):
                    continue
                if b < 0 or b == a:
                    continue
                tid = self.texts(p.get("text") or "")
                if tid < 0:
                    continue
                t_st.append(st)
                t_a.append(min(a, b))
                t_b.append(max(a, b))
                t_sp.append(a)
                t_tx.append(tid)
                t_x.append(xi)
                t_y.append(yi)
                t_ch.append(CH_DM)

            elif kd == "joint_activity":
                self.ja_pop += 1
                # step 0 の party は初期条件(世帯・同伴の種)であって物語ではない。
                if st <= 0:
                    continue
                w = p.get("with") or []
                ty = self.ja_types(p.get("type"))
                pl = self.ja_places(p.get("place"))
                self.ja_kept += 1
                for i in range(len(w)):
                    for j in range(i + 1, len(w)):
                        try:
                            u, v = int(w[i]), int(w[j])
                        except (TypeError, ValueError):
                            continue
                        if u < 0 or v < 0 or u == v:
                            continue
                        lst = self.ja[_pkey(u, v)]
                        row = (st, ty, pl)
                        if not lst or lst[-1] != row:   # 同一同伴の重複行を落とす
                            lst.append(row)

            elif kd == "acquaint":
                o = p.get("other")
                if o is None:
                    continue
                k = _pkey(a, int(o))
                if k not in self.acq:
                    self.acq[k] = (st, self.acq_vias(p.get("via")))

            elif kd == "train_copresence":
                o = p.get("other_id", p.get("other"))
                if o is None:
                    continue
                k = _pkey(a, int(o))
                if k not in self.tc:
                    self.tc[k] = (st, self.tc_lines(p.get("line")),
                                  int(p.get("car") or 0))

        if c_st:
            self._conv_chunks.append((
                np.asarray(c_st, dtype=np.int32), np.asarray(c_a, dtype=np.uint32),
                np.asarray(c_b, dtype=np.uint32), np.asarray(c_tp, dtype=np.uint8),
                np.asarray(c_tn, dtype=np.uint8), np.asarray(c_oc, dtype=np.uint8),
                np.asarray(c_sc, dtype=np.int32), np.asarray(c_x, dtype=np.int16),
                np.asarray(c_y, dtype=np.int16)))
            self.conv_kept += len(c_st)
        if t_st:
            self._txt_chunks.append((
                np.asarray(t_st, dtype=np.int32), np.asarray(t_a, dtype=np.uint32),
                np.asarray(t_b, dtype=np.uint32), np.asarray(t_sp, dtype=np.uint32),
                np.asarray(t_tx, dtype=np.int32), np.asarray(t_x, dtype=np.int16),
                np.asarray(t_y, dtype=np.int16), np.asarray(t_ch, dtype=np.uint8)))
            self.text_pair_rows += len(t_st)

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

        # 関係 kind / 物語 kind の判定は **Arrow 側だけ**で済ませる
        # (全 40.6 億行の Python 文字列化を避ける)。
        rel_idx, rel_kinds = None, None
        narr_idx, narr_kinds = None, None
        if "kind" in names and "payload" in names and "agent_id" in names:
            import pyarrow as pa
            kc = batch.column("kind")
            if hasattr(kc, "dictionary") and hasattr(kc, "dictionary_decode"):
                kc = kc.dictionary_decode()

            def _pick(kinds_tuple):
                mask = pc.fill_null(
                    pc.is_in(kc, value_set=pa.array(kinds_tuple, pa.string())), False)
                mnp = mask.to_numpy(zero_copy_only=False).astype(bool)
                if not mnp.any():
                    return None, None
                ix = np.nonzero(mnp)[0]
                return ix, kc.take(pa.array(ix)).to_pylist()

            rel_idx, rel_kinds = _pick(REL_KINDS)
            if self.narrative:
                narr_idx, narr_kinds = _pick(NARR_KINDS)

        xnp = ynp = None
        if "x" in names and "y" in names and "agent_id" in names:
            aid = pc.fill_null(batch.column("agent_id"), -1).to_numpy(
                zero_copy_only=False).astype(np.int64)
            if aid.size:
                self.agents_seen = max(self.agents_seen, int(aid.max()) + 1)
            xnp = batch.column("x").to_numpy(zero_copy_only=False).astype(np.float64)
            ynp = batch.column("y").to_numpy(zero_copy_only=False).astype(np.float64)
            self._positions(step_np, sim_np, aid, xnp, ynp)

        if rel_idx is not None:
            self._relations(batch, rel_idx, rel_kinds)
        if narr_idx is not None:
            self._narratives(batch, narr_idx, narr_kinds, xnp, ynp)

    def finish(self) -> None:
        self._flush(self._cur_bin)
        self._cur_bin = -1


def scan_l1(run_dir, *, hex_m: float, steps_per_bin: int, rel_cap: int,
            step_max=None, batch_rows: int = 262_144, narrative: bool = True,
            conv_cap: int = CONV_CAP_DEFAULT,
            text_cap: int = TEXT_CAP_DEFAULT) -> _Scan:
    """L1 を 1 パスで舐める。`l1_stream` の有界読みだけを使う。"""
    import l1_stream
    sc = _Scan(hex_m, steps_per_bin, rel_cap, narrative=narrative,
               conv_cap=conv_cap, text_cap=text_cap)
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
    if sc.narrative:
        _log(f"  物語素材: conversation {sc.conv_kept:,}/{sc.conv_pop:,} 件・"
             f"実文 {len(sc.texts):,} 種 → ペア行 {sc.text_pair_rows:,}・"
             f"acquaint {len(sc.acq):,} 組・train_copresence {len(sc.tc):,} 組・"
             f"同伴(step>0) {sc.ja_kept:,}/{sc.ja_pop:,} 件")
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
def rel_arrays(sc: _Scan):
    """`_rel_chunks` を step 昇順の 4 本の配列へ畳む(**cap を掛ける前**)。

    画面2 のペア抽出は cap 前の全件を要る(掲載上限は「表示」の話であって
    「母集団」の話ではない)ので、cap を掛ける `pack_relations` から分離する。
    """
    import numpy as np
    if not sc._rel_chunks:
        return None
    step = np.concatenate([c[0] for c in sc._rel_chunks])
    a = np.concatenate([c[1] for c in sc._rel_chunks])
    b = np.concatenate([c[2] for c in sc._rel_chunks])
    tv = np.concatenate([c[3] for c in sc._rel_chunks])
    sc._rel_chunks.clear()
    order = np.argsort(step, kind="stable")
    return step[order], a[order], b[order], tv[order]


def pack_relations(sc: _Scan, arrs=None) -> dict:
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
    if arrs is None:
        return {**base, "shown": 0, "n": 0, "deduped": 0, "capped": False}
    step, a, b, tv = arrs
    deduped = int(step.size)

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
# 画面2「関係の伝記」— 注目ペアの自動抽出
#
# 方針(計画 §1 画面2)
# --------------------
# * 母集団は **全ペア**。掲載は cap 件で、両方を必ず併記する。
# * 「面白さ」を 1 本のスコアに賭けない: カテゴリ別の**枠**(PAIR_QUOTA)を先に
#   埋めてから、残りをスコア順で埋める。稀少カテゴリ(車内の他人 → 関係は
#   Run A で 7 組しかない)が強いカテゴリに押し流されないようにするため。
# * 1 ペアあたりの会話は「最初 / tier 遷移の前後 / 結末が動いた回 / 最後」に絞る。
#   実文(speak・dm)は稀少(Run A で 6,988 行)なので**全部載せる**。
# --------------------------------------------------------------------------- #
def _group_bounds(keys):
    """ソート済み鍵配列 → (開始, 終了) の配列 2 本。"""
    import numpy as np
    if keys.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    bnd = np.nonzero(np.diff(keys))[0] + 1
    starts = np.concatenate([np.zeros(1, dtype=np.int64), bnd])
    ends = np.concatenate([bnd, np.asarray([keys.size], dtype=np.int64)])
    return starts.astype(np.int64), ends.astype(np.int64)


def _concat_chunks(chunks, ncol):
    import numpy as np
    if not chunks:
        return None
    return [np.concatenate([c[i] for c in chunks]) for i in range(ncol)]


def read_roster(run_dir, want: set) -> dict:
    """`roster.parquet` から**必要な体だけ**名前を引く(全 40 万行は持たない)。

    roster は day 境界に撮られるので、途中から街に入った来街者は出ない場合がある。
    引けなかった体は呼び出し側が `#<id>` で表示する(欠測を偽名で埋めない)。
    """
    import l1_stream
    import pyarrow.parquet as pq
    out: dict[int, list] = {}
    if not want:
        return out
    for p in l1_stream.l1_paths(run_dir, "roster"):
        try:
            with l1_stream._open_shared(p) as fh:
                t = pq.ParquetFile(fh).read(
                    columns=["agent_id", "name", "age", "gender", "occupation"])
        except (OSError, ValueError, KeyError):
            continue
        d = t.to_pydict()
        for aid, nm, ag, ge, oc in zip(d["agent_id"], d["name"], d["age"],
                                       d["gender"], d["occupation"]):
            if aid is None:
                continue
            aid = int(aid)
            if aid in want and aid not in out:
                out[aid] = [nm or "", int(ag or 0), ge or "", oc or ""]
    return out


def read_closeness(run_dir, pairs: set, agents: set) -> tuple[dict, dict]:
    """`relations.parquet`(G5 日次差分)から選抜ペアの closeness 軌跡を引く。

    返すのは `{pair: [[day, clo_ab, tier_ab, clo_ba, tier_ba], ...]}` と素性メタ。
    **差分方式**なので「値が動かなかった日は行が無い」= 前日値の持ち越しは
    ビューワー側でやる(欠測を 0 で埋めない)。
    """
    import l1_stream
    import pyarrow.parquet as pq
    meta = {"available": False, "parts": 0, "rows": 0, "days": [], "pairs": 0}
    if not pairs:
        return {}, meta
    paths = l1_stream.l1_paths(run_dir, "relations")
    meta["parts"] = len(paths)
    if not paths:
        return {}, meta
    acc: dict[int, dict] = {}
    days = set()
    rows = 0
    for p in paths:
        try:
            with l1_stream._open_shared(p) as fh:
                pf = pq.ParquetFile(fh)
                for batch in pf.iter_batches(
                        batch_size=262_144,
                        columns=["day", "agent_id", "other_id", "closeness", "tier"]):
                    d = batch.to_pydict()
                    rows += batch.num_rows
                    for day, aid, oid, clo, ti in zip(
                            d["day"], d["agent_id"], d["other_id"],
                            d["closeness"], d["tier"]):
                        if aid is None or oid is None or oid < 0:
                            continue
                        aid, oid = int(aid), int(oid)
                        if aid not in agents or oid not in agents:
                            continue
                        k = _pkey(aid, oid)
                        if k not in pairs:
                            continue
                        days.add(int(day or 0))
                        slot = acc.setdefault(k, {}).setdefault(int(day or 0),
                                                                [None, None, None, None])
                        fwd = 0 if aid <= oid else 2
                        slot[fwd] = _round(clo, 4)
                        slot[fwd + 1] = None if ti is None else int(ti)
        except (OSError, ValueError, KeyError):
            continue
    out = {k: [[day] + v for day, v in sorted(m.items())] for k, m in acc.items()}
    meta.update({"available": True, "rows": rows, "days": sorted(days),
                 "pairs": len(out)})
    return out, meta


def scan_positions(run_dir, ids: list, n_steps: int, *, stride: int = 1,
                   step_max=None, batch_rows: int = 262_144) -> dict:
    """選抜ペアの体だけ、step ごとの位置を 2 パス目で拾う(storyline リボンの y)。

    `payload` 列を **1 バイトも読まない**ので 1 パス目よりずっと軽い。位置を出さなかった
    step は**最終既知位置を持ち越す**(hexbin の carry レイヤーと同じ流儀)。先頭側の
    未知は -32768 のまま(= 欠測を推定で埋めない)。
    """
    import numpy as np
    import l1_stream
    ids = np.asarray(sorted(int(i) for i in ids), dtype=np.int64)
    n = int(ids.size)
    ns = max(1, (int(n_steps) + stride - 1) // stride)
    X = np.full((n, ns), -32768, dtype=np.int16)
    Y = np.full((n, ns), -32768, dtype=np.int16)
    if n == 0:
        return {"stride": stride, "n": ns, "ids": [], "x": _b64(X), "y": _b64(Y)}
    t0 = time.time()
    for batch in l1_stream.iter_record_batches(
            run_dir, ["step", "agent_id", "x", "y"], step_max=step_max,
            batch_rows=batch_rows):
        import pyarrow.compute as pc
        aid = pc.fill_null(batch.column("agent_id"), -1).to_numpy(
            zero_copy_only=False).astype(np.int64)
        m = np.isin(aid, ids)
        if not m.any():
            continue
        st = pc.fill_null(batch.column("step"), -1).to_numpy(
            zero_copy_only=False).astype(np.int64)[m]
        xs = batch.column("x").to_numpy(zero_copy_only=False).astype(np.float64)[m]
        ys = batch.column("y").to_numpy(zero_copy_only=False).astype(np.float64)[m]
        ok = np.isfinite(xs) & np.isfinite(ys) & (st >= 0) & (st < n_steps)
        if not ok.any():
            continue
        ri = np.searchsorted(ids, aid[m][ok])
        ci = st[ok] // stride
        # fancy 代入は「最後の書き込みが残る」= その step ビンの最終位置
        X[ri, ci] = np.clip(np.rint(xs[ok]), -32767, 32767).astype(np.int16)
        Y[ri, ci] = np.clip(np.rint(ys[ok]), -32767, 32767).astype(np.int16)
    # 前方向の持ち越し(観測がある直近の列を引き写す)
    known = X != -32768
    idx = np.where(known, np.arange(ns)[None, :], 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    rows = np.arange(n)[:, None]
    X = X[rows, idx]
    Y = Y[rows, idx]
    _log(f"  位置 2 パス目: {n} 体 × {ns} 点 ({time.time() - t0:.1f}s)")
    return {"stride": int(stride), "n": ns, "ids": ids.tolist(),
            "x": _b64(X.reshape(-1)), "y": _b64(Y.reshape(-1))}


def build_pairs(run_dir, sc: _Scan, arrs, *, n_steps: int, cap: int,
                per_conv: int, traj_max: int, step_max=None) -> dict:
    """注目ペアを自動抽出し、1 組ぶんの物語(遷移・出会い・会話・実文・軌跡)を組む。"""
    import numpy as np
    t0 = time.time()

    pop = {"rel_pairs": 0, "tier3": 0, "partner": 0, "break": 0,
           "acquaint": len(sc.acq), "train_copresence": len(sc.tc),
           "outing": 0, "text": 0, "conv": 0, "candidates": 0}

    # ---- 関係イベントをペアごとに畳む ------------------------------------ #
    rec: dict[int, dict] = {}
    if arrs is not None:
        step, a, b, tv = arrs
        ai, bi = a.astype(np.int64), b.astype(np.int64)
        rk = (np.minimum(ai, bi) << 32) | np.maximum(ai, bi)
        ev = (tv >> 4).astype(np.int8)
        tier = (tv & 0x0F).astype(np.int8)
        o = np.lexsort((step, rk))
        rk, rs, rev, rti = rk[o], step[o], ev[o], tier[o]
        starts, ends = _group_bounds(rk)
        pop["rel_pairs"] = int(starts.size)
        for s, e in zip(starts.tolist(), ends.tolist()):
            k = int(rk[s])
            tiers: list[list[int]] = []
            partner = None
            for i in range(s, e):
                evi, sti, tii = int(rev[i]), int(rs[i]), int(rti[i])
                if evi == REL_EV["relation_tier"]:
                    if tiers and tiers[-1][0] == sti and tiers[-1][1] == tii:
                        continue        # 双方向ログの片割れ(同 step・同段)
                    tiers.append([sti, tii])
                elif evi == REL_EV["partner_formed"] and partner is None:
                    partner = sti
            r = {"tiers": tiers, "partner": partner}
            if tiers:
                top = max(t for _, t in tiers)
                r["top"] = top
                r["t0"] = tiers[0][0]
                r["t1"] = tiers[-1][0]
                if top >= 3:
                    pop["tier3"] += 1
            if partner is not None:
                pop["partner"] += 1
            rec[k] = r
    pop["break"] = len(sc.brk)
    pop["outing"] = len(sc.ja)

    # ---- conversation / 実文の索引(ペア鍵でソートして searchsorted で引く) -- #
    conv = _concat_chunks(sc._conv_chunks, 9)
    sc._conv_chunks.clear()
    if conv is not None:
        ck = (conv[1].astype(np.int64) << 32) | conv[2].astype(np.int64)
        co = np.lexsort((conv[0], ck))
        ck = ck[co]
        conv = [c[co] for c in conv]
        ck_u, ck_first, ck_cnt = np.unique(ck, return_index=True, return_counts=True)
        pop["conv"] = int(ck_u.size)
    else:
        ck = ck_u = ck_first = ck_cnt = np.zeros(0, dtype=np.int64)

    txt = _concat_chunks(sc._txt_chunks, 8)
    sc._txt_chunks.clear()
    if txt is not None:
        tk = (txt[1].astype(np.int64) << 32) | txt[2].astype(np.int64)
        to = np.lexsort((txt[0], tk))
        tk = tk[to]
        txt = [c[to] for c in txt]
        tk_u, tk_first, tk_cnt = np.unique(tk, return_index=True, return_counts=True)
        pop["text"] = int(tk_u.size)
    else:
        tk = tk_u = tk_first = tk_cnt = np.zeros(0, dtype=np.int64)

    def _lookup(keys_u, cnts, k):
        i = int(np.searchsorted(keys_u, k))
        if i < keys_u.size and int(keys_u[i]) == k:
            return i, int(cnts[i])
        return -1, 0

    unev = sc.outcomes.ix.get("uneventful", -2) + 1     # +1 = _narratives の下駄

    # ---- 候補集合 --------------------------------------------------------- #
    cand: set[int] = set(rec)
    cand |= set(sc.brk)
    cand |= set(sc.acq)
    cand |= set(sc.ja)
    cand |= {int(k) for k in tk_u.tolist()}
    # 車内の他人は 1 万組ある。**関係か会話が続いた組だけ**を候補に上げる
    # (「同じ車両に居合わせただけ」は物語ではない。母集団は別途明記する)。
    fs_pairs = set()
    for k in sc.tc:
        if k in rec or k in sc.brk or _lookup(ck_u, ck_cnt, k)[0] >= 0 \
                or _lookup(tk_u, tk_cnt, k)[0] >= 0:
            cand.add(k)
            fs_pairs.add(k)
    pop["familiar_strangers_followed"] = len(fs_pairs)
    pop["candidates"] = len(cand)

    # ---- スコアと札 ------------------------------------------------------- #
    scored: list[tuple] = []
    tags_of: dict[int, set] = {}
    stat_speed: list[int] = []
    stat_top = Counter()
    stat_via = Counter()
    stat_meet = Counter()
    for k in cand:
        r = rec.get(k) or {}
        tiers = r.get("tiers") or []
        top = int(r.get("top") or 0)
        ci, cn = _lookup(ck_u, ck_cnt, k)
        ti, tn = _lookup(tk_u, tk_cnt, k)
        nbrk = len(sc.brk.get(k) or ())
        ja = sc.ja.get(k) or ()
        acq = sc.acq.get(k)
        tc = sc.tc.get(k)
        partner = r.get("partner")

        span = None
        if len(tiers) >= 2 and top >= 2:
            span = int(tiers[-1][0] - tiers[0][0])

        tags = set()
        if k in fs_pairs:
            tags.add("familiar_strangers")
        if nbrk:
            tags.add("break")
        if top >= 3 and span is not None:
            tags.add("fast_promote")
        if tn >= 4:
            tags.add("rich_dialogue")
        if ja:
            tags.add("outing")
        if partner is not None and (tiers or tn or cn):
            tags.add("partner")
        if acq is not None:
            tags.add("acquaint")
        tags_of[k] = tags

        s = 0.0
        s += 3.0 * math.log1p(tn)                       # 実文があることが最大の価値
        s += 0.9 * top
        if span is not None:
            s += 2.2 / (1.0 + span / 6.0)               # 速い昇格ほど高い
        s += 2.4 * min(3, nbrk)
        s += 1.6 * min(3, len(ja))
        s += 0.55 * math.log1p(cn)
        if acq is not None:
            s += 0.8
        if k in fs_pairs:
            s += 4.0                                    # 稀少(Run A で 7 組)
        if partner is not None and (tn or nbrk):
            s += 1.2
        scored.append((s, k, top, span, tn, cn, nbrk))

        # 母集団統計(A/B 対比パネル。掲載ぶんではなく候補全体で取る)
        if span is not None:
            stat_speed.append(span)
        if top:
            stat_top[top] += 1
        if acq is not None:
            stat_via[sc.acq_vias.vals[acq[1]] if acq[1] >= 0 else "?"] += 1
        stat_meet["acquaint" if acq is not None else
                  ("train" if tc is not None else
                   ("partner" if partner is not None else
                    ("conversation" if cn else "tier")))] += 1

    scored.sort(key=lambda t: (-t[0], t[1]))
    rank_of = {t[1]: i for i, t in enumerate(scored)}

    # ---- 掲載の選抜(枠 → 残りはスコア順) -------------------------------- #
    chosen: list[int] = []
    seen: set[int] = set()
    quota_filled = {}
    for tag, q in PAIR_QUOTA.items():
        got = 0
        for t in scored:
            if got >= q or len(chosen) >= cap:
                break
            k = t[1]
            if k in seen or tag not in tags_of[k]:
                continue
            chosen.append(k)
            seen.add(k)
            got += 1
        quota_filled[tag] = got
    for t in scored:
        if len(chosen) >= cap:
            break
        if t[1] not in seen:
            chosen.append(t[1])
            seen.add(t[1])
    chosen.sort(key=lambda k: rank_of[k])

    # ---- 1 組ぶんの物語を組む -------------------------------------------- #
    agents: set[int] = set()
    for k in chosen:
        agents.add(k >> 32)
        agents.add(k & 0xFFFFFFFF)
    names = read_roster(run_dir, agents)
    close, close_meta = read_closeness(run_dir, set(chosen), agents)

    stride = max(1, int(math.ceil(n_steps / max(1, traj_max))))
    traj = scan_positions(run_dir, sorted(agents), n_steps, stride=stride,
                          step_max=step_max)
    traj_ix = {int(v): i for i, v in enumerate(traj["ids"])}

    scenes_used: dict[int, int] = {}
    texts_used: dict[int, int] = {}

    def _scene(i):
        if i < 0:
            return -1
        j = scenes_used.get(i)
        if j is None:
            j = len(scenes_used)
            scenes_used[i] = j
        return j

    def _text(i):
        j = texts_used.get(i)
        if j is None:
            j = len(texts_used)
            texts_used[i] = j
        return j

    items = []
    for k in chosen:
        a_id, b_id = k >> 32, k & 0xFFFFFFFF
        r = rec.get(k) or {}
        tiers = r.get("tiers") or []
        marks = [s for s, _ in tiers] + [x[0] for x in (sc.brk.get(k) or ())]

        # -- 会話(母集団 → 掲載) --
        ci, cn = _lookup(ck_u, ck_cnt, k)
        convs = []
        if ci >= 0 and conv is not None:
            s = int(ck_first[ci])
            e = s + cn
            rows = list(range(s, e))
            prio = []
            for j, i in enumerate(rows):
                st = int(conv[0][i])
                if j == 0 or j == len(rows) - 1:
                    p = 0                                   # 最初と最後は必ず
                elif int(conv[5][i]) != unev:
                    p = 1                                   # 結末が動いた回
                elif any(abs(st - m) <= 2 for m in marks):
                    p = 2                                   # 段が動く前後
                else:
                    p = 3
                prio.append((p, st, i))
            prio.sort()
            pick = sorted(i for _, _, i in prio[:per_conv])
            for i in pick:
                convs.append([int(conv[0][i]), int(conv[3][i]), int(conv[4][i]),
                              int(conv[5][i]), _scene(int(conv[6][i])),
                              int(conv[7][i]), int(conv[8][i])])

        # -- 実文(稀少なので全部) --
        ti, tn = _lookup(tk_u, tk_cnt, k)
        texts = []
        if ti >= 0 and txt is not None:
            s = int(tk_first[ti])
            for i in range(s, s + min(tn, 60)):
                texts.append([int(txt[0][i]), int(txt[3][i]), _text(int(txt[4][i])),
                              int(txt[5][i]), int(txt[6][i]), int(txt[7][i])])

        acq = sc.acq.get(k)
        tc = sc.tc.get(k)
        it = {
            "a": int(a_id), "b": int(b_id),
            "ia": traj_ix.get(int(a_id), -1), "ib": traj_ix.get(int(b_id), -1),
            "tiers": tiers,
            "brk": [list(x) for x in (sc.brk.get(k) or ())],
            "ja": [list(x) for x in (sc.ja.get(k) or ())][:8],
            "acq": [int(acq[0]), int(acq[1])] if acq else None,
            "tc": [int(tc[0]), int(tc[1]), int(tc[2])] if tc else None,
            "partner": r.get("partner"),
            "conv_n": cn, "conv_shown": len(convs), "conv": convs,
            "text_n": tn, "text": texts,
            "close": close.get(k) or [],
            "tags": sorted(tags_of[k]),
            "score": None,                       # 直後にまとめて入れる
        }
        if tiers:
            it["top"] = int(max(t for _, t in tiers))
            it["span"] = int(tiers[-1][0] - tiers[0][0])
            it["first"] = int(tiers[0][0])
        items.append(it)

    score_by = {t[1]: t[0] for t in scored}
    for it, k in zip(items, chosen):
        it["score"] = _round(score_by[k], 3)

    scenes = [None] * len(scenes_used)
    for src, dst in scenes_used.items():
        scenes[dst] = sc.scenes.vals[src] if 0 <= src < len(sc.scenes.vals) else None
    texts_out = [None] * len(texts_used)
    for src, dst in texts_used.items():
        texts_out[dst] = sc.texts.vals[src] if 0 <= src < len(sc.texts.vals) else ""

    speed_hist = Counter()
    for v in stat_speed:
        speed_hist[min(v, 60)] += 1

    _log(f"  ペア抽出: 候補 {len(cand):,} 組 → 掲載 {len(items)} 組 "
         f"({time.time() - t0:.1f}s)")
    return {
        "cap": int(cap),
        "shown": len(items),
        "population": pop,
        "quota": {k: {"target": PAIR_QUOTA[k], "filled": quota_filled.get(k, 0),
                      "label": PAIR_TAG_LABELS[k]} for k in PAIR_QUOTA},
        "tag_labels": dict(PAIR_TAG_LABELS),
        "items": items,
        "names": {str(k): v for k, v in names.items()},
        "names_missing": sorted(a for a in agents if a not in names)[:50],
        "names_missing_n": sum(1 for a in agents if a not in names),
        "scenes": scenes,
        "texts": texts_out,
        "topics": list(sc.topics.vals),
        "tones": list(sc.tones.vals),
        "outcomes": list(sc.outcomes.vals),
        "ja_types": list(sc.ja_types.vals),
        "ja_places": list(sc.ja_places.vals),
        "tc_lines": list(sc.tc_lines.vals),
        "vias": list(sc.acq_vias.vals),
        "causes": list(sc.brk_causes.vals),
        "traj": traj,
        "closeness": close_meta,
        "stats": {
            "speed_hist": {str(k): int(v) for k, v in sorted(speed_hist.items())},
            "speed_n": len(stat_speed),
            "speed_median": (sorted(stat_speed)[len(stat_speed) // 2]
                             if stat_speed else None),
            "top_tier": {str(k): int(v) for k, v in sorted(stat_top.items())},
            "via": dict(stat_via),
            "meet": dict(stat_meet),
        },
        "text_population": {"rows": sc.text_pop, "pair_rows": sc.text_pair_rows,
                            "distinct": len(sc.texts),
                            "dropped": sc.texts.dropped},
        "conv_population": {"rows": sc.conv_pop, "kept": sc.conv_kept},
        "notes": [
            "母集団=全ペア。掲載はカテゴリ枠(quota)を先に埋め、残りをスコア順で埋める。",
            "実文は speak(hearers)と dm(to)から。hear は本文を持たないので読まない。",
            "conversation は実文を持たない構造化会話 C2(topic/tone/outcome/scene)。",
            "joint_activity は step 0(初期世帯・party の種)を読まない。",
            "closeness は日境界の差分サイドカー。行が無い日は前日値の持ち越し。",
        ],
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
              rel_cap: int, step_max=None, pairs: bool = True,
              pair_cap: int = PAIR_CAP_DEFAULT, pair_conv: int = PAIR_CONV_DEFAULT,
              pair_traj: int = PAIR_TRAJ_DEFAULT,
              conv_cap: int = CONV_CAP_DEFAULT,
              text_cap: int = TEXT_CAP_DEFAULT) -> dict:
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

    n_parts = len(l1_stream.l1_paths(run_dir))
    if n_parts == 0:
        # 走行中のランが 1 度も flush していない段階(Run B 起動直後)。
        # 「まだデータが無い」を**そう表示できる形**で返す(落とさない)。
        _log(f"[{label}] ⚠ L1 の完結 part がまだ 1 本も無い(走行中の可能性)")
        return {
            "label": label, "run_name": rm.get("name") or run_dir.name,
            "run_dir": str(run_dir), "empty": True,
            "run_mode": manifest.get("run_mode"),
            "scenario": (manifest.get("world") or {}).get("scenario"),
            "model": (manifest.get("model") or {}).get("name"),
            "backend": (manifest.get("model") or {}).get("backend"),
            "started_at": manifest.get("started_at"),
            "dt_min": dt_min, "dt_source": prov["source"], "steps_per_day": spd,
            "start_tod": rm.get("start_tod") or "00:00",
            "start_date": rm.get("start_date"),
            "n_agents": int(rm.get("n_agents") or 0), "n_steps": 0,
            "n_steps_declared": int(rm.get("n_steps") or 0), "complete": False,
            "l1_parts": 0, "l1_rows": 0, "l1_rows_positioned": 0,
            "sim_min_range": [None, None], "kind_totals": {},
            "series": {"l2_steps": [], "l2": {}, "kinds": {},
                       "kinds_population": 0, "kinds_shown": 0},
            "hexbin": None, "relations": None, "pairs": None,
            "build": {"scan_sec": 0.0},
        }

    t0 = time.time()
    sc = scan_l1(run_dir, hex_m=hex_m, steps_per_bin=steps_per_bin,
                 rel_cap=rel_cap, step_max=step_max, narrative=pairs,
                 conv_cap=conv_cap, text_cap=text_cap)
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
    arrs = rel_arrays(sc)
    rels = pack_relations(sc, arrs)

    pairs_data = None
    if pairs:
        pairs_data = build_pairs(run_dir, sc, arrs, n_steps=n_steps, cap=pair_cap,
                                 per_conv=pair_conv, traj_max=pair_traj,
                                 step_max=step_max)

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
        "pairs": pairs_data,
        "build": {"scan_sec": round(t_scan, 1)},
    }


# --------------------------------------------------------------------------- #
# HTML(自己完結・Canvas 2D・CDN 禁止)
# --------------------------------------------------------------------------- #
HTML_TEMPLATE = r"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Shibuya Chronicle</title>
<style>
:root{--bg:#0a0e14;--panel:rgba(18,22,30,.86);--fg:#e6e9ee;--dim:#9aa4b2;
 --acc:#3b82f6;--bd:rgba(255,255,255,.09);--warn:#f59e0b;
 --a:#60a5fa;--b:#fb923c;--t1:#38bdf8;--t2:#4ade80;--t3:#fbbf24;--bad:#f87171;}
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
/* ---------- 画面タブ ---------- */
.tabs{display:flex;gap:4px}
.tabs button{background:transparent;color:var(--dim);border:1px solid transparent;
 border-radius:5px;padding:4px 12px;cursor:pointer;font:inherit;font-size:12.5px}
.tabs button.on{background:#1b2331;color:var(--fg);border-color:var(--bd)}
.pane{display:none;height:calc(100% - 44px)}
.pane.on{display:flex}
/* ---------- 画面2 関係の伝記 ---------- */
#plist{width:300px;flex:none;border-right:1px solid var(--bd);background:var(--panel);
 display:flex;flex-direction:column;min-height:0}
#plist .hd{padding:8px 9px 6px;border-bottom:1px solid var(--bd)}
#plist select,#plist input{background:#0d1117;color:var(--fg);border:1px solid var(--bd);
 border-radius:4px;font:inherit;font-size:11.5px;padding:3px 5px;width:100%}
#plist input{margin-top:5px}
#prows{overflow-y:auto;flex:1;min-height:0}
.prow{padding:6px 9px;border-bottom:1px solid rgba(255,255,255,.05);cursor:pointer}
.prow:hover{background:#161c26}
.prow.on{background:#1c2637;box-shadow:inset 3px 0 0 var(--acc)}
.prow .nm{font-size:12px;font-weight:600;white-space:nowrap;overflow:hidden;
 text-overflow:ellipsis}
.prow .mt{color:var(--dim);font-size:10.5px;font-variant-numeric:tabular-nums;
 display:flex;gap:7px;flex-wrap:wrap;margin-top:1px}
.bdg{display:inline-block;padding:0 5px;border-radius:3px;font-size:9.5px;line-height:15px;
 background:#243040;color:#b6c2d2;margin-right:3px}
.bdg.fs{background:#3c2c14;color:#fbbf24}.bdg.brk{background:#3d1d1f;color:#f87171}
.bdg.fast{background:#14342a;color:#4ade80}.bdg.rich{background:#16304a;color:#7cc4ff}
.bdg.out{background:#33254a;color:#c4b5fd}.bdg.par{background:#3b1f33;color:#f9a8d4}
#pmain{flex:1;min-width:0;display:flex;flex-direction:column}
#rwrap{position:relative;flex:0 0 46%;min-height:190px;border-bottom:1px solid var(--bd)}
#rcv{display:block;width:100%;height:100%}
#rhead{position:absolute;left:12px;top:9px;font-size:12.5px;pointer-events:none}
#rhead b{font-size:14px}
#rlegend{position:absolute;right:12px;top:9px;font-size:10.5px;color:var(--dim);
 text-align:right;pointer-events:none;line-height:1.5}
#ptl{flex:1;overflow-y:auto;padding:8px 12px 22px;min-height:0}
.ev{display:flex;gap:9px;align-items:flex-start;margin:0 0 6px}
.ev .st{flex:none;width:78px;color:var(--dim);font-size:10.5px;
 font-variant-numeric:tabular-nums;padding-top:3px}
.ev .bd{flex:1;min-width:0;border-radius:7px;padding:5px 9px;font-size:12px;
 border:1px solid var(--bd);cursor:pointer}
.ev .bd:hover{border-color:var(--acc)}
.ev.a .bd{background:#15243a;border-color:#22405f}
.ev.b .bd{background:#38240f;border-color:#5c3a15}
.ev.c .bd{background:#141a24;color:#b6c2d2;cursor:pointer}
.ev.m .bd{background:transparent;border:none;border-top:1px dashed var(--bd);
 border-radius:0;color:var(--dim);font-size:11px;padding:5px 0 2px;cursor:default}
.ev .who{font-size:10.5px;color:var(--dim);margin-bottom:1px}
#pside{width:302px;flex:none;border-left:1px solid var(--bd);background:var(--panel);
 overflow-y:auto;padding:10px 11px 26px}
#pside h2{font-size:11.5px;color:var(--dim);margin:12px 0 6px;font-weight:600;
 text-transform:uppercase;letter-spacing:.06em}
#pside h2:first-child{margin-top:0}
.abrow{display:flex;gap:8px;margin:3px 0 7px}
.abcol{flex:1;min-width:0}
.abcol .t{font-size:10.5px;color:var(--dim);margin-bottom:2px}
.abcol canvas{display:block;width:100%;height:52px}
.empty{padding:26px;color:var(--dim);font-size:12.5px;text-align:center}
</style></head><body>
<div id="top">
  <h1>Shibuya Chronicle</h1>
  <div class="tabs" id="tabs">
    <button data-p="map" class="on">俯瞰</button>
    <button data-p="pair">関係の伝記</button>
  </div>
  <span class="sub" id="runmeta"></span>
  <span style="flex:1"></span>
  <span class="sub">対照:</span>
  <div class="runsw" id="runsw"></div>
</div>
<div id="wrap" class="pane on">
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
<div id="wrap2" class="pane">
  <div id="plist">
    <div class="hd">
      <select id="psort">
        <option value="score">並べ替え: ドラマ性(総合)</option>
        <option value="speed">並べ替え: 昇格の速さ</option>
        <option value="text">並べ替え: 実文の多さ</option>
        <option value="conv">並べ替え: 会話回数</option>
        <option value="first">並べ替え: 出会いが早い順</option>
      </select>
      <input id="pfind" type="search" placeholder="名前・職業・タグで絞る">
      <div class="note" id="pcount"></div>
    </div>
    <div id="prows"></div>
  </div>
  <div id="pmain">
    <div id="rwrap">
      <canvas id="rcv"></canvas>
      <div id="rhead"></div>
      <div id="rlegend"></div>
      <div id="rtip" style="position:absolute;pointer-events:none;background:#0d1117ee;
        border:1px solid var(--bd);border-radius:5px;padding:5px 8px;font-size:11.5px;
        display:none;white-space:nowrap;z-index:5"></div>
    </div>
    <div id="ptl"></div>
  </div>
  <div id="pside">
    <h2>この 2 人について</h2>
    <div id="pfacts"></div>
    <h2>母集団 → 掲載</h2>
    <div id="ppop"></div>
    <h2>A/B ペア母集団の対比</h2>
    <div id="pstat"></div>
  </div>
</div>
<script id="chronicle-data" type="application/json">__CHRONICLE_DATA__</script>
<script>
"use strict";
const D = JSON.parse(document.getElementById('chronicle-data').textContent);
const BM = D.basemap, RUNS = D.runs, ORDER = D.order;
/* 走行中のランは「まだ 1 本も flush していない」ことがある(Run B 起動直後)。
   その run は空として持ち、切替ボタンを無効にする(落とさない・偽の 0 を描かない)。 */
const HAS = k => !!(RUNS[k] && !RUNS[k].empty && RUNS[k].hexbin);
let RK = ORDER.find(HAS) || ORDER[0];
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
const HEX = {}; ORDER.forEach(k => { if(HAS(k)) HEX[k] = hexOf(RUNS[k]); });
function layerKey(){ return document.getElementById('lyCarry').checked ? 'carry' : 'observed'; }
function curLayer(){ return HEX[RK].L[layerKey()]; }

/* ---------- 時間 ---------- */
function nSteps(){ return Math.max(1, R().n_steps || 1); }
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
  const H = HEX[RK]; if(!H || !H.binMeta.length) return -1;
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
  if(!HEX[RK]) return;
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
  if(!HEX[RK]){ document.getElementById('legend').innerHTML =
    '<div class="note warn">このランはまだ確定 part がありません。</div>'; return; }
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
  h += '<div class="note">この表は画面2「関係の伝記」の母集団。'
     + '<a href="#" style="color:var(--acc)" onclick="setTab(\'pair\');return false">'
     + '2 人の物語を見る →</a></div>';
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
    b.textContent = 'Run ' + k + (RUNS[k] ? (HAS(k) ? '' : ' (走行中・未flush)') : ' (未取得)');
    b.disabled = !HAS(k);
    if(k === RK) b.className = 'on';
    b.onclick = ()=>{ if(!HAS(k)) return; RK = k; cur = Math.min(cur, nSteps()-1);
      document.getElementById('scrub').max = nSteps()-1;
      fillTop(); fillAbout(); fillRel(); buildCharts(); pairInit(); draw(); };
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

/* =========================================================================
   画面2「関係の伝記」
   -------------------------------------------------------------------------
   storyline リボン = 2 本の線。y の開きが **その時刻の 2 人の距離**、線の太さが
   関係の段(closeness が 2 日以上あるランではその連続値)、合流点の印が会話。
   下段は同じ時間軸のログで、実文つきの発話は吹き出しにする。どの行を押しても
   俯瞰タブの「その瞬間・その場所」へ飛ぶ(P0 の hexbin 画面と同じ cur を共有)。
   ========================================================================= */
const TIER_NAME = ['他人','知人','友人','親友','親友+','親友++'];
const TIER_COL  = ['#64748b','#38bdf8','#4ade80','#fbbf24','#fbbf24','#fbbf24'];
const TONE_COL  = {neutral:'#7c8ea5', friendly:'#4ade80', cool:'#60a5fa',
                   hostile:'#f87171'};
const TAGCLS = {familiar_strangers:'fs', 'break':'brk', fast_promote:'fast',
                rich_dialogue:'rich', outing:'out', partner:'par', acquaint:''};
let TAB = 'map', PSEL = -1, PORD = [], PZ = {}, RGEO = [];

function esc(s){ return String(s===null||s===undefined?'':s)
  .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function P(){ return R() ? R().pairs : null; }
function curItem(){ const p = P(); return (p && PSEL>=0 && PSEL<p.items.length)
  ? p.items[PSEL] : null; }
function pname(id){ const p=P(); const v = p && p.names[String(id)];
  return (v && v[0]) ? v[0] : '#'+id; }
function pmeta(id){ const p=P(); const v = p && p.names[String(id)];
  if(!v) return '名簿に無い体(途中から街に入った来街者など)';
  return (v[1]?v[1]+'歳':'') + (v[2]?'・'+v[2]:'') + (v[3]?'・'+v[3]:''); }
function trajOf(){
  if(PZ[RK] === undefined){ const p = P(), T = p && p.traj;
    PZ[RK] = T ? {x: bin(T.x, Int16Array), y: bin(T.y, Int16Array),
                  n: T.n, stride: T.stride} : null; }
  return PZ[RK];
}
/* tier の履歴(昇格 relation_tier と 悪化 relation_break を 1 本の列に混ぜる) */
function tierTrack(it){
  const ev = [];
  (it.tiers||[]).forEach(t => ev.push([t[0], t[1], 'up']));
  (it.brk||[]).forEach(b => ev.push([b[0], b[2], 'down']));
  ev.sort((a,b)=> a[0]-b[0]);
  return ev;
}
function tierAt(track, step){ let t = 0;
  for(const e of track){ if(e[0] <= step) t = e[1]; else break; } return t; }

/* ---------- タブ ---------- */
function setTab(t){
  TAB = t;
  document.querySelectorAll('#tabs button').forEach(b =>
    b.classList.toggle('on', b.dataset.p === t));
  document.getElementById('wrap').classList.toggle('on', t === 'map');
  document.getElementById('wrap2').classList.toggle('on', t === 'pair');
  if(t === 'map') resize(); else pairResize();
  syncHash();
}
document.querySelectorAll('#tabs button').forEach(b =>
  b.onclick = ()=> setTab(b.dataset.p));

/* ---------- ペア一覧 ---------- */
function sortVal(it, mode){
  if(mode === 'speed'){                       // 昇格が速い順(段が上がった組だけ)
    if(!it.tiers || !it.tiers.length || (it.top||0) < 2) return 1e9;
    return it.span * 1000 - (it.top||0);
  }
  if(mode === 'text') return -(it.text_n||0);
  if(mode === 'conv') return -(it.conv_n||0);
  if(mode === 'first') return (it.first!==undefined ? it.first
    : (it.conv.length ? it.conv[0][0] : (it.text.length ? it.text[0][0] : 1e6)));
  return -(it.score||0);
}
function pairRowHTML(it, i){
  const p = P(), tk = tierTrack(it), top = it.top||0;
  let bd = '';
  (it.tags||[]).forEach(t => { if(t in TAGCLS)
    bd += '<span class="bdg '+TAGCLS[t]+'">'+esc(p.tag_labels[t]||t)+'</span>'; });
  const path = tk.length ? tk.map(e=>TIER_NAME[e[1]]||e[1]).join('→') : '—';
  return '<div class="prow'+(i===PSEL?' on':'')+'" data-i="'+i+'">'
    + '<div class="nm">'+esc(pname(it.a))+' <span style="color:var(--dim)">×</span> '
      + esc(pname(it.b))+'</div>'
    + '<div style="margin:2px 0 1px">'+bd+'</div>'
    + '<div class="mt"><span style="color:'+(TIER_COL[top]||'#64748b')+'">'+path+'</span>'
      + '<span>実文 '+(it.text_n||0)+'</span><span>会話 '+(it.conv_n||0)+'</span>'
      + (it.span!==undefined && top>=2 ? '<span>'+it.span+'step で昇格</span>' : '')
      + '</div></div>';
}
function buildPairList(){
  const p = P(), host = document.getElementById('prows');
  const cnt = document.getElementById('pcount');
  if(!p || !p.items.length){
    host.innerHTML = '<div class="empty">このランには画面2 の素材がありません。</div>';
    cnt.textContent = ''; return;
  }
  const mode = document.getElementById('psort').value;
  const q = (document.getElementById('pfind').value||'').trim().toLowerCase();
  PORD = p.items.map((it,i)=>i);
  if(q){
    PORD = PORD.filter(i => { const it = p.items[i];
      const s = (pname(it.a)+' '+pname(it.b)+' '+pmeta(it.a)+' '+pmeta(it.b)+' '
        + (it.tags||[]).map(t=>p.tag_labels[t]||t).join(' ')).toLowerCase();
      return s.indexOf(q) >= 0; });
  }
  PORD.sort((x,y)=> sortVal(p.items[x],mode) - sortVal(p.items[y],mode) || x-y);
  host.innerHTML = PORD.map(i => pairRowHTML(p.items[i], i)).join('');
  cnt.innerHTML = '母集団 ' + (p.population.candidates||0).toLocaleString()
    + ' 組 → 掲載 ' + p.shown + ' 組' + (q ? ' / 絞り込み '+PORD.length+' 組' : '');
  host.querySelectorAll('.prow').forEach(el =>
    el.onclick = ()=> selectPair(+el.dataset.i));
}
function selectPair(i){
  PSEL = i;
  document.querySelectorAll('#prows .prow').forEach(el =>
    el.classList.toggle('on', +el.dataset.i === i));
  drawRibbon(); buildTimeline(); fillFacts(); syncHash();
}

/* ---------- storyline リボン ---------- */
const rcv = document.getElementById('rcv'), rctx = rcv.getContext('2d');
function pairResize(){
  const dpr = Math.min(2, window.devicePixelRatio||1);
  rcv.width = Math.round(rcv.clientWidth*dpr);
  rcv.height = Math.round(rcv.clientHeight*dpr);
  rctx.setTransform(dpr,0,0,dpr,0,0);
  drawRibbon();
}
function ribbonSeries(it){
  const T = trajOf();
  if(!T || it.ia < 0 || it.ib < 0) return null;
  const n = T.n, oa = it.ia*n, ob = it.ib*n;
  const d = new Float32Array(n); d.fill(-1);
  for(let i=0;i<n;i++){
    const ax = T.x[oa+i], ay = T.y[oa+i], bx = T.x[ob+i], by = T.y[ob+i];
    if(ax === -32768 || bx === -32768) continue;
    d[i] = Math.hypot(ax-bx, ay-by);
  }
  return {d: d, n: n, stride: T.stride};
}
function drawRibbon(){
  const W = rcv.clientWidth, H = rcv.clientHeight;
  rctx.fillStyle = '#0c111a'; rctx.fillRect(0,0,W,H);
  RGEO = [];
  const it = curItem();
  const head = document.getElementById('rhead'), leg = document.getElementById('rlegend');
  if(!it){ head.innerHTML = '<span style="color:var(--dim)">ペアを選んでください</span>';
    leg.innerHTML = ''; return; }
  const p = P(), tk = tierTrack(it);
  const L = 12, Rp = 12, Tp = 40, Bp = 24;
  const x0 = L, x1 = Math.max(L+10, W-Rp), y0 = Tp, y1 = Math.max(Tp+30, H-Bp);
  const cy = (y0+y1)/2, half = (y1-y0)/2 - 4;
  const ns = nSteps(), X = s => x0 + (s/Math.max(1,ns-1))*(x1-x0);
  const ser = ribbonSeries(it);

  /* 縦のグリッド(1 時間ごと)+ 時刻ラベル */
  const spb = HEX[RK] ? HEX[RK].spb : 6;
  rctx.strokeStyle = 'rgba(255,255,255,.05)'; rctx.lineWidth = 1;
  rctx.fillStyle = '#6b7686'; rctx.font = '10px sans-serif'; rctx.textAlign = 'center';
  const gap = Math.max(1, Math.round(spb * Math.ceil(ns/spb/14)));
  for(let s=0;s<ns;s+=gap){
    const x = X(s); rctx.beginPath(); rctx.moveTo(x,y0-6); rctx.lineTo(x,y1+4); rctx.stroke();
    rctx.fillText(clockText(s).hhmm, x, y1+16);
  }

  /* 2 本の線 */
  let dcap = 300;
  if(ser){ let mx = 0; for(let i=0;i<ser.n;i++) if(ser.d[i] > mx) mx = ser.d[i];
    dcap = Math.max(150, Math.min(2500, mx)); }
  const sepOf = d => (d < 0) ? half*1.85
    : Math.sqrt(Math.min(d,dcap)/dcap) * half*1.85;
  const wOf = t => 1.0 + 1.35*Math.min(3, t);
  if(ser){
    /* 面(2 本のあいだ)= その時刻の段の色 */
    for(let i=0;i<ser.n-1;i++){
      if(ser.d[i] < 0 || ser.d[i+1] < 0) continue;   /* 欠測を跨いだ面は描かない */
      const s = i*ser.stride, sx = X(s), ex = X(Math.min(ns-1,(i+1)*ser.stride));
      const t = tierAt(tk, s), sep = sepOf(ser.d[i]), sep2 = sepOf(ser.d[i+1]);
      rctx.fillStyle = TIER_COL[t] || '#64748b';
      rctx.globalAlpha = 0.10 + 0.05*t;
      rctx.beginPath();
      rctx.moveTo(sx, cy-sep/2); rctx.lineTo(ex, cy-sep2/2);
      rctx.lineTo(ex, cy+sep2/2); rctx.lineTo(sx, cy+sep/2);
      rctx.closePath(); rctx.fill();
    }
    rctx.globalAlpha = 1;
    for(const side of [0,1]){
      rctx.strokeStyle = side ? getCss('--b') : getCss('--a');
      rctx.lineJoin = 'round'; rctx.lineCap = 'round';
      let last = null;
      for(let i=0;i<ser.n;i++){
        if(ser.d[i] < 0){ last = null; continue; }
        const s = i*ser.stride, x = X(s), sep = sepOf(ser.d[i]);
        const y = side ? cy + sep/2 : cy - sep/2;
        if(last){ rctx.beginPath(); rctx.lineWidth = wOf(tierAt(tk, s));
          rctx.moveTo(last[0], last[1]); rctx.lineTo(x, y); rctx.stroke(); }
        last = [x, y];
      }
    }
  } else {
    rctx.fillStyle = '#6b7686'; rctx.font = '11.5px sans-serif'; rctx.textAlign = 'center';
    rctx.fillText('この 2 人の位置は L1 に残っていない(軌跡なし)', (x0+x1)/2, cy);
  }

  /* 会話(合流点)*/
  it.conv.forEach((c,i)=>{
    const x = X(c[0]), tone = p.tones[c[2]-1], big = (p.outcomes[c[3]-1] !== 'uneventful');
    const y = cy;
    rctx.fillStyle = TONE_COL[tone] || '#7c8ea5';
    rctx.globalAlpha = big ? 1 : 0.62;
    rctx.beginPath(); rctx.arc(x, y, big ? 4.2 : 2.6, 0, 7); rctx.fill();
    rctx.globalAlpha = 1;
    RGEO.push({x:x, y:y, r:7, k:'conv', i:i, s:c[0]});
  });
  /* 実文(話した側の線の上に置く) */
  it.text.forEach((t,i)=>{
    const x = X(t[0]), side = (t[1] === it.b) ? 1 : 0;
    const d = ser ? ser.d[Math.min(ser.n-1, Math.round(t[0]/ser.stride))] : -1;
    const sep = ser ? sepOf(d) : half*1.85;
    const y = side ? cy + sep/2 : cy - sep/2;
    rctx.fillStyle = side ? getCss('--b') : getCss('--a');
    rctx.beginPath(); rctx.arc(x, y, 3.4, 0, 7); rctx.fill();
    rctx.strokeStyle = '#0c111a'; rctx.lineWidth = 1; rctx.stroke();
    RGEO.push({x:x, y:y, r:7, k:'text', i:i, s:t[0]});
  });
  /* 段の変化 */
  rctx.textAlign = 'center'; rctx.font = '10px sans-serif';
  tk.forEach((e,ti)=>{
    const x = X(e[0]), up = (e[2] === 'up');
    rctx.strokeStyle = up ? (TIER_COL[e[1]]||'#4ade80') : getCss('--bad');
    rctx.setLineDash([3,3]); rctx.lineWidth = 1.2;
    rctx.beginPath(); rctx.moveTo(x, y0-8); rctx.lineTo(x, y1); rctx.stroke();
    rctx.setLineDash([]);
    rctx.fillStyle = up ? (TIER_COL[e[1]]||'#4ade80') : getCss('--bad');
    rctx.fillText((up?'':'▼ ')+(TIER_NAME[e[1]]||e[1]), x, y0-11);
    RGEO.push({x:x, y:y0-14, r:9, k:'tier', i:ti, s:e[0], t:e[1], up:up});
  });
  /* 出会いの文脈 / 同伴 */
  if(it.tc){ mark(X(it.tc[0]), y1-4, '#a78bfa', 'sq'); }
  if(it.acq){ mark(X(it.acq[0]), y1-4, '#22d3ee', 'tri'); }
  (it.ja||[]).forEach(j => mark(X(j[0]), y1-4, '#c4b5fd', 'dia'));
  /* 俯瞰タブと共有している「いま」 */
  const xc = X(Math.min(cur, ns-1));
  rctx.strokeStyle = '#f59e0b'; rctx.lineWidth = 1;
  rctx.beginPath(); rctx.moveTo(xc, y0-8); rctx.lineTo(xc, y1+2); rctx.stroke();

  function mark(x, y, col, shape){
    rctx.fillStyle = col; rctx.beginPath();
    if(shape==='sq') rctx.rect(x-3, y-3, 6, 6);
    else if(shape==='tri'){ rctx.moveTo(x, y-4); rctx.lineTo(x+4, y+3); rctx.lineTo(x-4, y+3); }
    else { rctx.moveTo(x, y-4); rctx.lineTo(x+4, y); rctx.lineTo(x, y+4); rctx.lineTo(x-4, y); }
    rctx.closePath(); rctx.fill();
  }

  head.innerHTML = '<b style="color:'+getCss('--a')+'">'+esc(pname(it.a))+'</b>'
    + ' <span style="color:var(--dim)">×</span> '
    + '<b style="color:'+getCss('--b')+'">'+esc(pname(it.b))+'</b>'
    + ' <span class="sub" style="color:var(--dim);font-size:11px">'
    + esc(pmeta(it.a))+' / '+esc(pmeta(it.b))+'</span>';
  leg.innerHTML = '縦の開き = 2 人の距離(0〜'+Math.round(dcap)+' m・√スケール)<br>'
    + '線の太さ = 関係の段 / ●= 会話(色 = 語調)/ ◉= 実文<br>'
    + (ser ? '' : '<span class="warn">軌跡なし</span>');
}
function getCss(v){ return getComputedStyle(document.documentElement)
  .getPropertyValue(v).trim() || '#888'; }

rcv.addEventListener('mousemove', e=>{
  const rect = rcv.getBoundingClientRect(), mx = e.clientX-rect.left, my = e.clientY-rect.top;
  const tip = document.getElementById('rtip');
  let best = null, bd = 121;
  for(const g of RGEO){ const d = (g.x-mx)*(g.x-mx) + (g.y-my)*(g.y-my);
    if(d < bd){ bd = d; best = g; } }
  if(!best){ tip.style.display='none'; return; }
  const it = curItem(), p = P();
  let h = '';
  if(best.k === 'conv'){ const c = it.conv[best.i];
    h = '<b>会話</b> '+esc(p.topics[c[1]-1])+' / '+esc(p.tones[c[2]-1])+' / '
      + esc(p.outcomes[c[3]-1]) + '<br><span style="color:var(--dim)">'
      + esc(p.scenes[c[4]]||'')+'</span>'; }
  else if(best.k === 'text'){ const t = it.text[best.i];
    h = '<b>'+esc(pname(t[1]))+'</b>「'+esc(p.texts[t[2]])+'」'; }
  else h = '<b>'+esc(TIER_NAME[best.t]||'')+(best.up?' に上がった':' に下がった')+'</b>';
  tip.innerHTML = h + '<br><span style="color:var(--dim)">Day '
    + (clockText(best.s).day+1)+' '+clockText(best.s).hhmm+' · step '+best.s+'</span>';
  tip.style.display = 'block';
  tip.style.left = Math.min(rcv.clientWidth-260, mx+12)+'px';
  tip.style.top = (my+12)+'px';
});
rcv.addEventListener('mouseleave', ()=>
  document.getElementById('rtip').style.display='none');
rcv.addEventListener('click', e=>{
  const rect = rcv.getBoundingClientRect(), mx = e.clientX-rect.left, my = e.clientY-rect.top;
  let best = null, bd = 121;
  for(const g of RGEO){ const d = (g.x-mx)*(g.x-mx) + (g.y-my)*(g.y-my);
    if(d < bd){ bd = d; best = g; } }
  if(!best) return;
  const el = document.querySelector('#ptl [data-s="'+best.k+'-'+best.i+'"]');
  if(el){ el.scrollIntoView({block:'center'}); el.style.outline = '1px solid var(--acc)';
    setTimeout(()=>{ el.style.outline = ''; }, 1400); }
});

/* ---------- 下段: 会話タイムライン ---------- */
function jumpToMap(step, x, y){
  cur = Math.max(0, Math.min(nSteps()-1, step|0));
  const sc2 = document.getElementById('scrub'); sc2.value = cur;
  if(x !== undefined && x !== null && x !== -32768){
    cam.x = x; cam.y = y; cam.s = Math.max(cam.s, 1.0); }
  setTab('map'); draw();
}
function tlClock(s){ const c = clockText(s);
  return 'Day '+(c.day+1)+' '+c.hhmm+'<br><span style="opacity:.65">step '+s+'</span>'; }
function buildTimeline(){
  const host = document.getElementById('ptl'), it = curItem(), p = P();
  if(!it){ host.innerHTML = '<div class="empty">左の一覧からペアを選ぶと、'
    + '2 人のあいだに起きたことが時間順に並びます。</div>'; return; }
  const rows = [];
  it.text.forEach((t,i)=> rows.push({s:t[0], o:1, k:'text', i:i}));
  it.conv.forEach((c,i)=> rows.push({s:c[0], o:2, k:'conv', i:i}));
  tierTrack(it).forEach((e,i)=> rows.push({s:e[0], o:0, k:'tier', i:i, e:e}));
  if(it.acq) rows.push({s:it.acq[0], o:0, k:'acq'});
  if(it.tc) rows.push({s:it.tc[0], o:0, k:'tc'});
  if(it.partner !== null && it.partner !== undefined)
    rows.push({s:it.partner, o:0, k:'partner'});
  (it.ja||[]).forEach((j,i)=> rows.push({s:j[0], o:0, k:'ja', i:i, j:j}));
  rows.sort((a,b)=> a.s-b.s || a.o-b.o);
  let h = '';
  for(const r of rows){
    const id = 'data-s="'+r.k+'-'+(r.i||0)+'"';
    if(r.k === 'text'){
      const t = it.text[r.i], side = (t[1] === it.b) ? 'b' : 'a';
      h += '<div class="ev '+side+'" '+id+'><div class="st">'+tlClock(r.s)+'</div>'
        + '<div class="bd" data-x="'+t[3]+'" data-y="'+t[4]+'" data-step="'+r.s+'">'
        + '<div class="who">'+esc(pname(t[1]))
        + (t[5]===1?' <span style="color:var(--warn)">DM</span>':'')+'</div>'
        + esc(p.texts[t[2]]) + '</div></div>';
    } else if(r.k === 'conv'){
      const c = it.conv[r.i], tone = p.tones[c[2]-1], oc = p.outcomes[c[3]-1];
      h += '<div class="ev c" '+id+'><div class="st">'+tlClock(r.s)+'</div>'
        + '<div class="bd" data-x="'+c[5]+'" data-y="'+c[6]+'" data-step="'+r.s+'">'
        + '<span style="color:'+(TONE_COL[tone]||'#7c8ea5')+'">●</span> '
        + esc(p.topics[c[1]-1]) + ' · ' + esc(tone) + ' · ' + esc(oc)
        + ' <span style="color:var(--dim)">— ' + esc(p.scenes[c[4]]||'場所不明')
        + '</span></div></div>';
    } else if(r.k === 'tier'){
      const e = r.e, up = e[2]==='up';
      h += '<div class="ev m" '+id+'><div class="st">'+tlClock(r.s)+'</div><div class="bd">'
        + (up ? '<span style="color:'+(TIER_COL[e[1]]||'#4ade80')+'">▲ '
                + esc(TIER_NAME[e[1]]||e[1])+' になった</span>'
              : '<span style="color:'+getCss('--bad')+'">▼ '
                + esc(TIER_NAME[e[1]]||e[1])+' に下がった</span>')
        + '</div></div>';
    } else if(r.k === 'acq'){
      h += '<div class="ev m"><div class="st">'+tlClock(r.s)+'</div><div class="bd">'
        + '△ 知り合いとして成立(via ' + esc(p.vias[it.acq[1]]||'?') + ')</div></div>';
    } else if(r.k === 'tc'){
      h += '<div class="ev m"><div class="st">'+tlClock(r.s)+'</div><div class="bd">'
        + '■ 同じ電車に居合わせた(' + esc(p.tc_lines[it.tc[1]]||'?')
        + ' / ' + it.tc[2] + '号車)</div></div>';
    } else if(r.k === 'partner'){
      h += '<div class="ev m"><div class="st">'+tlClock(r.s)+'</div><div class="bd">'
        + '♥ パートナー関係</div></div>';
    } else if(r.k === 'ja'){
      h += '<div class="ev m"><div class="st">'+tlClock(r.s)+'</div><div class="bd">'
        + '◆ 一緒に ' + esc(p.ja_types[r.j[1]]||'?') + '</div></div>';
    }
  }
  host.innerHTML = h || '<div class="empty">この 2 人のログは関係イベントだけです。</div>';
  host.querySelectorAll('.bd[data-step]').forEach(el => el.onclick = ()=>
    jumpToMap(+el.dataset.step, +el.dataset.x, +el.dataset.y));
}

/* ---------- 右パネル ---------- */
function fillFacts(){
  const it = curItem(), p = P(), host = document.getElementById('pfacts');
  if(!it){ host.innerHTML = '<div class="note">ペア未選択。</div>'; return; }
  const tk = tierTrack(it);
  let h = '';
  h += kv('A', esc(pname(it.a))+' <span style="color:var(--dim)">'+esc(pmeta(it.a))+'</span>');
  h += kv('B', esc(pname(it.b))+' <span style="color:var(--dim)">'+esc(pmeta(it.b))+'</span>');
  h += kv('段の道すじ', tk.length ? tk.map(e=>(TIER_NAME[e[1]]||e[1])
    +'@'+e[0]).join(' → ') : '—');
  if(it.span !== undefined && (it.top||0) >= 2)
    h += kv('昇格の所要', it.span + ' step ('+(it.span*R().dt_min)+' 分)');
  h += kv('出会いの文脈', it.tc ? ('同じ電車 ('+esc(p.tc_lines[it.tc[1]]||'?')+')')
    : (it.acq ? ('acquaint via '+esc(p.vias[it.acq[1]]||'?'))
    : (it.partner!==null&&it.partner!==undefined ? 'パートナー(初期)' : '会話の積み重ね')));
  h += kv('会話', (it.conv_n||0).toLocaleString()+' 回(掲載 '+it.conv_shown+' 回)');
  h += kv('実文', (it.text_n||0)+' 行');
  h += kv('こじれ', (it.brk||[]).length + ' 回');
  h += kv('同伴', (it.ja||[]).length + ' 回');
  if(it.close && it.close.length){
    h += '<div class="note">closeness(日境界の台帳)</div>';
    it.close.forEach(c => h += kv('day '+c[0],
      (c[1]===null?'—':c[1]) + ' / ' + (c[3]===null?'—':c[3])
      + ' <span style="color:var(--dim)">(A→B / B→A)</span>'));
  } else {
    h += '<div class="note">closeness の行なし(このランでは日境界を跨いでいない'
       + 'か、値が動かなかった)。</div>';
  }
  h += kv('ドラマ性スコア', it.score);
  host.innerHTML = h;
}
function fillPop(){
  const p = P(), host = document.getElementById('ppop');
  if(!p){ host.innerHTML = '<div class="note">素材なし。</div>'; return; }
  const po = p.population;
  let h = '';
  h += kv('候補(母集団)', (po.candidates||0).toLocaleString()+' 組');
  h += kv('掲載', p.shown + ' / cap ' + p.cap + ' 組');
  h += kv('· 関係イベントのある組', (po.rel_pairs||0).toLocaleString());
  h += kv('· 親友(tier3)到達', (po.tier3||0).toLocaleString());
  h += kv('· パートナー', (po.partner||0).toLocaleString());
  h += kv('· こじれた組', (po.break||0).toLocaleString());
  h += kv('· 知り合い宣言', (po.acquaint||0).toLocaleString());
  h += kv('· 同じ電車の他人', (po.train_copresence||0).toLocaleString()
    + ' → 関係が続いた '+(po.familiar_strangers_followed||0));
  h += kv('· 実文のある組', (po.text||0).toLocaleString());
  h += kv('· 会話のある組', (po.conv||0).toLocaleString());
  h += '<div class="note">枠(カテゴリ最低保証)の埋まり方: '
    + Object.keys(p.quota).map(t=>esc(p.quota[t].label)+' '+p.quota[t].filled
      +'/'+p.quota[t].target).join(' · ') + '</div>';
  h += kv('実文の母集団', (p.text_population.rows||0).toLocaleString()
    + ' 行 → ペア行 '+(p.text_population.pair_rows||0).toLocaleString());
  if(p.names_missing_n)
    h += '<div class="note warn">名簿を引けなかった体 '+p.names_missing_n
       + ' 名(day 境界の roster に載らない来街者)。#id で表示。</div>';
  if(p.conv_population.kept < p.conv_population.rows)
    h += '<div class="note warn">conversation は RAM 上限で '
       + p.conv_population.kept.toLocaleString() + ' / '
       + p.conv_population.rows.toLocaleString() + ' 件しか読めていない'
       + '(--conv-cap を上げると全件)。会話回数は下振れする。</div>';
  if(p.text_population.dropped)
    h += '<div class="note warn">実文は distinct 上限で '
       + p.text_population.dropped.toLocaleString() + ' 行ぶん取りこぼした'
       + '(--text-cap)。</div>';
  h += '<div class="note">' + p.notes.map(esc).join('<br>') + '</div>';
  host.innerHTML = h;
}
function miniBars(cv2, obj, col, fmt2){
  const dpr = Math.min(2, window.devicePixelRatio||1);
  const W = cv2.clientWidth||120, H = 52;
  cv2.width = Math.round(W*dpr); cv2.height = Math.round(H*dpr);
  const c = cv2.getContext('2d'); c.setTransform(dpr,0,0,dpr,0,0);
  c.clearRect(0,0,W,H);
  const ks = Object.keys(obj); if(!ks.length){ return; }
  const vs = ks.map(k=>obj[k]), mx = Math.max.apply(null, vs);
  const bw = W/ks.length;
  c.font = '9px sans-serif'; c.textAlign = 'center';
  ks.forEach((k,i)=>{
    const hgt = mx ? (obj[k]/mx)*(H-14) : 0;
    c.fillStyle = col; c.fillRect(i*bw+1, H-12-hgt, Math.max(1,bw-2), hgt);
    c.fillStyle = '#6b7686'; c.fillText(fmt2 ? fmt2(k) : k, i*bw+bw/2, H-2);
  });
}
function fillPairStat(){
  const host = document.getElementById('pstat');
  const runs = ORDER.filter(k => RUNS[k] && RUNS[k].pairs);
  if(!runs.length){ host.innerHTML = '<div class="note">対比なし。</div>'; return; }
  let h = '<div class="note">同 seed でも軌道が分岐するので A と B に「同じペア」は'
    + '存在しない。比べられるのは<b>母集団の形</b>(昇格の速さの分布・出会いの文脈の'
    + '構成)。母集団は候補全体で、掲載ぶんではない。</div>';
  h += '<div class="abrow">' + runs.map(k =>
    '<div class="abcol"><div class="t">Run '+k+' 昇格 step 分布 (n='
    + RUNS[k].pairs.stats.speed_n.toLocaleString() + '·中央値 '
    + (RUNS[k].pairs.stats.speed_median===null?'—':RUNS[k].pairs.stats.speed_median)
    + ')</div><canvas data-r="'+k+'" data-w="speed"></canvas></div>').join('') + '</div>';
  h += '<div class="abrow">' + runs.map(k =>
    '<div class="abcol"><div class="t">Run '+k+' 出会いの文脈</div>'
    + '<canvas data-r="'+k+'" data-w="meet"></canvas></div>').join('') + '</div>';
  h += '<div class="abrow">' + runs.map(k =>
    '<div class="abcol"><div class="t">Run '+k+' 到達した段</div>'
    + '<canvas data-r="'+k+'" data-w="top"></canvas></div>').join('') + '</div>';
  host.innerHTML = h;
  const MEET = {acquaint:'宣言', train:'電車', partner:'配偶', conversation:'会話',
                tier:'段のみ'};
  host.querySelectorAll('canvas').forEach(cv2 => {
    const st = RUNS[cv2.dataset.r].pairs.stats;
    if(cv2.dataset.w === 'speed'){
      const o = {}; const src = st.speed_hist;
      for(let b=0;b<=9;b++) o[b] = 0;
      Object.keys(src).forEach(k=>{ const b = Math.min(9, Math.floor(+k/4));
        o[b] = (o[b]||0) + src[k]; });
      miniBars(cv2, o, '#4ade80', k => (+k*4)+(+k===9?'+':''));
    } else if(cv2.dataset.w === 'meet'){
      miniBars(cv2, st.meet, '#a78bfa', k => MEET[k]||k);
    } else {
      miniBars(cv2, st.top_tier, '#fbbf24', k => TIER_NAME[+k]||k);
    }
  });
}
function pairInit(){
  PSEL = -1;
  document.getElementById('psort').onchange = buildPairList;
  document.getElementById('pfind').oninput = buildPairList;
  buildPairList(); fillPop(); fillPairStat();
  const p = P();
  if(p && p.items.length && PORD.length) selectPair(PORD[0]);
  else { drawRibbon(); buildTimeline(); fillFacts(); }
}
window.addEventListener('resize', ()=>{ if(TAB==='pair') pairResize(); });

/* ---------- 起動(#run=A&step=54 で「その瞬間」を直接開ける) ---------- */
let bootTab = 'map', bootPair = -1;
(function(){
  const h = new URLSearchParams(location.hash.replace(/^#/, ''));
  const rk = (h.get('run')||'').toUpperCase();
  if(HAS(rk)) RK = rk;
  const s = parseInt(h.get('step'), 10);
  if(isFinite(s)) cur = Math.max(0, Math.min(nSteps()-1, s));
  if(h.get('tab') === 'pair') bootTab = 'pair';
  const pi = parseInt(h.get('pair'), 10);
  if(isFinite(pi)) bootPair = pi;
})();
function syncHash(){
  try { history.replaceState(null, '', '#run='+RK+'&step='+cur+'&tab='+TAB
    + (TAB==='pair' && PSEL>=0 ? '&pair='+PSEL : '')); } catch(_){}
}
scrub.addEventListener('change', syncHash);
scrub.max = nSteps()-1; scrub.value = cur;
fillTop(); fillAbout(); fillRel(); buildCharts();
fitAll(); resize();
pairInit();
if(bootPair >= 0 && P() && bootPair < P().items.length) selectPair(bootPair);
if(bootTab === 'pair') setTab('pair');
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
    ap.add_argument("--pairs", type=int, default=PAIR_CAP_DEFAULT,
                    help=f"画面2 に載せる注目ペア数(既定 {PAIR_CAP_DEFAULT})。"
                         "母集団は常に併記される")
    ap.add_argument("--pair-conv", type=int, default=PAIR_CONV_DEFAULT,
                    help=f"1 ペアあたりの会話掲載数(既定 {PAIR_CONV_DEFAULT})")
    ap.add_argument("--pair-traj", type=int, default=PAIR_TRAJ_DEFAULT,
                    help=f"1 ペアあたりの軌跡サンプル数(既定 {PAIR_TRAJ_DEFAULT})")
    ap.add_argument("--conv-cap", type=int, default=CONV_CAP_DEFAULT,
                    help="conversation を貯める行数上限(RAM よけ・1 件 23B)")
    ap.add_argument("--text-cap", type=int, default=TEXT_CAP_DEFAULT,
                    help="実文の distinct 上限(RAM よけ)")
    ap.add_argument("--no-pairs", action="store_true",
                    help="画面2 の素材を作らない(P0 と同じ出力になる)")
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
                                rel_cap=a.rel_cap, step_max=a.max_steps,
                                pairs=not a.no_pairs, pair_cap=a.pairs,
                                pair_conv=a.pair_conv, pair_traj=a.pair_traj,
                                conv_cap=a.conv_cap, text_cap=a.text_cap)
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
             f"relations {_n(r['relations'])/1024:.0f}KB / "
             f"pairs {_n(r.get('pairs'))/1024:.0f}KB")
    _log(f"HTML: {out_html}  {size/1024/1024:.2f} MB "
         f"(規律 {HTML_SOFT_LIMIT/1024/1024:.0f}MB)")
    if size > HTML_SOFT_LIMIT:
        _log("⚠ サイズ規律 超過。効く順に: --rel-cap を小さく(最大の項)/ "
             "--bin-min を長く / --hex-m を大きく。")

    print(json.dumps({
        "html": str(out_html), "html_bytes": size,
        "data_dir": str(out_dir),
        "files": {p.name: p.stat().st_size for p in sorted(out_dir.glob("*.json"))},
        "runs": {k: ({"empty": True, "build_sec": timings[k]} if r.get("empty") else {
                     "n_steps": r["n_steps"], "l1_rows": r["l1_rows"],
                     "hex_bins": r["hexbin"]["n_bins"], "hex_cells": r["hexbin"]["n_cells"],
                     "hex_max": {ln: lv["global_max"]
                                 for ln, lv in r["hexbin"]["layers"].items()},
                     "rel_population": r["relations"]["population"],
                     "rel_deduped": r["relations"].get("deduped", 0),
                     "rel_shown": r["relations"].get("shown", 0),
                     "pairs": (None if not r.get("pairs") else {
                         "shown": r["pairs"]["shown"],
                         "candidates": r["pairs"]["population"]["candidates"],
                         "population": r["pairs"]["population"],
                         "quota": {t: q["filled"] for t, q in r["pairs"]["quota"].items()},
                         "texts": len(r["pairs"]["texts"]),
                         "names_missing": r["pairs"]["names_missing_n"],
                     }),
                     "build_sec": timings[k]}) for k, r in runs.items()},
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":                              # pragma: no cover
    raise SystemExit(main())
