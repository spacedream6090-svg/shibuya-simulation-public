#!/usr/bin/env python
"""Shibuya Chronicle ビューワーのビルドパイプライン(P0 + 画面2/3/4)。

位置づけ
--------
`docs/plans/viewer-chronicle-plan.md` §1 の実装。**俯瞰(画面1 の A 側)** /
**画面2「関係の伝記」** / **画面3「語の一生(伝播)」** / **画面4「物語ピン +
今日のハイライト」** を端から端まで動かす。作るもの:

1. **L2 指標の時系列**(全 step)+ **L1 kind 別の毎 step 件数**(会話数など L2 に無い量)
2. **位置 hexbin**(既定 200 m 格子 × 1 時間ビンの**在場者数**)= Uint16 量子化 + base64
3. **関係イベント**: `relation_tier` 遷移 / `partner_formed` / `acquaint` /
   `relation_break` を `(step, a, b, tier, ev)` の圧縮表へ
4. **注目ペアの伝記**(画面2): 全ペアを母集団に、カテゴリ枠 + ドラマ性スコアで
   数百組を選抜し、1 組ぶんの「段の遷移 / 出会いの文脈 / 会話 / **実文** /
   同伴 / closeness / 2 人の軌跡」を同梱する。
5. **伝播カスケード**(画面3): 信念 / リシェア / 語彙を同じ器(木・等時線・S 字・チャネル)へ
6. **物語ピン**(画面4): 宣言的 sifting パターン(遺失物の連鎖・緊急の連鎖・人づて hop>=2・
   関係の破綻→再構築・集会と不発の集会・逸脱・カスケードの離陸)を surprise で格付けし、
   当事者の **思考 → 行為**(`l1b_llm` × `llm_journal`)を 1 ホップ添える
7. 自己完結 HTML `viz/chronicle/chronicle.html`(CDN 禁止・Canvas 2D・遅延なしの一体型)

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
import re
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

#: 画面3「伝播」の素材。**信念 / リシェア / 語彙を同じスキーマ**で扱う(器は 1 つ)。
#: 実測(Run A finals_observe_20260824c・L1 40,561,808 行)で判ったこと:
#: - `belief_update` {fact, fact_kind, from, hop, src, cause, verified, value, conf}
#:   344,822 行のうち **344,805 行が cause=witness / hop=0 / from=-1**(= 各自の目撃)。
#:   人づて(cause=transmit / src=agent / hop=1)は **17 行だけ**。つまり Run A の
#:   「信念の広がり」は伝播木ではなく**同じ現場を大勢が見た並列発生**である。
#:   これは欠測ではなく所見なので、木の形(fan/chain/parallel)として素直に出す。
#: - `belief_transmit` {fact, to[], channel, hop, topic} … 発話側の台帳。`topic` は
#:   場所名(= fact の実名)。7 行しかないが **fact → 実名の唯一の厳密な出典**。
#: - `sns_reshare` {post_id, author} … **親 post とその著者**。自分が作る RT post の
#:   id は L1 に出ない(internet.react が内部で採番する)ので、下の二重ポインタで同定する。
#: - `viral_cascade` {post_id, author, reach} … インフルエンサー加重(元 post 1 件に 1 回)。
#: - `sns_post` {text, items} … 実文つきの投稿(Run A で 35 件)。RT と公式は本文を出さない。
#: - `stock_out` {poi, cat, src} … fact の生成源。fact → 場所名の**推定**に使う。
#: - `transmission` {item_id, from, channel} … 語彙の系譜(辺そのもの)。Run A で 85 行。
#: - `vocab_coin`/`label_coin`/`label_adopt`/`place_label_bind`/`vocab_use` … 語の一生。
PROP_KINDS = ("belief_update", "belief_transmit", "sns_reshare", "viral_cascade",
              "sns_post", "stock_out", "transmission", "vocab_coin", "label_coin",
              "label_adopt", "place_label_bind", "vocab_use")

#: カスケードの種別(HTML の凡例と 1 対 1)。
CAS_BELIEF, CAS_RESHARE, CAS_VOCAB = 0, 1, 2
CAS_KIND_LABELS = {CAS_BELIEF: "信念", CAS_RESHARE: "リシェア", CAS_VOCAB: "語彙"}
#: 種別ごとの掲載枠(合計は cap を超えない。語彙は Run A でほぼ 0 件 = 枠が余っても静か)。
CAS_QUOTA = {CAS_BELIEF: 30, CAS_RESHARE: 30, CAS_VOCAB: 14}

CAS_CAP_DEFAULT = 72                        # 掲載するカスケード数(母集団は必ず併記)
CAS_NODES_DEFAULT = 300                     # 1 カスケードの木ノード上限
CAS_ISO_CELLS = 400                         # 等時線に載せるヘックス数の上限
CAS_CURVE_MAX = 320                         # S 字曲線の点数上限
BELIEF_CAP_DEFAULT = 8_000_000              # belief_update を貯める行数上限(1 件 26B)
RESHARE_CAP_DEFAULT = 8_000_000             # sns_reshare を貯める行数上限(1 件 20B)
STOCK_CAP_DEFAULT = 4_000_000               # stock_out を貯める行数上限(fact 命名用)

#: 画面4「物語ピン + 今日のハイライト」の素材(story sifting)。
#: 宣言的パターンが読む kind を**ここに全部並べる**。実在しない kind は静かに落ちる
#: (Run A には `crime` / `police_response` / `event_host` が 1 行も無い = 静かに 0 件)。
#:
#: 連鎖の結び方(**exact join**。近接での当て推量をしない):
#: - 遺失物: 落とし主 `owner` × 品目 `item` × **落とした step**。落とした step は
#:   `lost_pickup.lying_steps` / `lost_notice.delay_steps` / `lost_expire.age_steps` を
#:   その行の step から引けば厳密に復元でき、`lost_return.held_steps` は
#:   `lost_turnin` の step を厳密に指す(engine が差分を payload に書いているため)。
#: - 緊急: 患者 id(`collapse` は agent_id 自身、`injury` / `traffic_accident` は
#:   payload.victim、以降は payload.patient)× 発生 step からの窓。
#: - 集会: `event_host.event_id` × `event_attend.event_id`(完全一致)。
STORY_KINDS = (
    "lost_drop", "lost_notice", "lost_pickup", "lost_turnin",
    "lost_keep", "lost_return", "lost_expire",
    "collapse", "injury", "traffic_accident",
    "ems_call", "ems_dispatch", "ems_transport",
    "hospital_admit", "hospital_discharge",
    "crime", "police_response", "nuisance",
    "event_host", "event_attend",
)

#: パターン種(HTML の凡例・地図ピンと 1 対 1)。
PAT_LOST, PAT_EMS, PAT_HOP, PAT_PAIR, PAT_GATHER, PAT_CRIME, PAT_VIRAL = range(7)
PAT_LABELS = {
    PAT_LOST: "遺失物の連鎖",
    PAT_EMS: "緊急の連鎖",
    PAT_HOP: "人づての語・信念",
    PAT_PAIR: "関係のドラマ",
    PAT_GATHER: "集まり",
    PAT_CRIME: "逸脱と迷惑",
    PAT_VIRAL: "離陸した投稿",
}
#: 地図ピンの字(絵文字を使わない = フォント差で化けない)。
PAT_GLYPHS = {PAT_LOST: "落", PAT_EMS: "急", PAT_HOP: "語", PAT_PAIR: "縁",
              PAT_GATHER: "集", PAT_CRIME: "犯", PAT_VIRAL: "波"}
PAT_COLORS = {PAT_LOST: "#fbbf24", PAT_EMS: "#f87171", PAT_HOP: "#4ade80",
              PAT_PAIR: "#f9a8d4", PAT_GATHER: "#c4b5fd", PAT_CRIME: "#fb923c",
              PAT_VIRAL: "#7cc4ff"}
#: 掲載枠(種別ごとの最低保証。合計は cap を超えない)。
#: 迷惑行為は Run A で 32,718 行あり、枠が無いと稀少パターンを押し流す。
STORY_QUOTA = {PAT_LOST: 40, PAT_EMS: 24, PAT_HOP: 24, PAT_PAIR: 40,
               PAT_GATHER: 24, PAT_CRIME: 20, PAT_VIRAL: 20}

STORY_CAP_DEFAULT = 200                     # 掲載する物語数(母集団は必ず併記)
STORY_BEATS_MAX = 24                        # 1 物語の拍(beat)の上限
STORY_ROW_CAP = 1_500_000                   # story kind を貯める行数上限(RAM よけ)
NUI_BIN_STEPS = 6                           # 迷惑行為の束ね幅(6 step = 1 時間 @Δt10)
NUI_BURST_MIN = 4                           # 「騒ぎが続いた」とみなす 1 セルの下限
NUI_CELL_CAP = 400_000                      # 迷惑行為セル辞書の上限
GATHER_MIN_N = 10                           # 集合の臨界(detect_gatherings の DEF_MIN_N)
THOUGHT_CAP_DEFAULT = 60                    # 思考チェーンを付ける物語数の上限
JOURNAL_SCAN_CAP = 400_000                  # llm_journal を舐める行数上限(走行中よけ)
STORY_EMS_WINDOW = 12                       # 発生 → 通報/出動 を同一件とみなす step 窓

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
                 conv_cap: int = CONV_CAP_DEFAULT, text_cap: int = TEXT_CAP_DEFAULT,
                 propagation: bool = True, belief_cap: int = BELIEF_CAP_DEFAULT,
                 reshare_cap: int = RESHARE_CAP_DEFAULT,
                 stock_cap: int = STOCK_CAP_DEFAULT, stories: bool = True,
                 story_row_cap: int = STORY_ROW_CAP):
        import numpy as np
        self.np = np
        self.grid = HexGrid(hex_m)
        self.steps_per_bin = max(1, int(steps_per_bin))
        self.rel_cap = int(rel_cap)
        self.pair_cache_max = int(pair_cache_max)

        # ---- 画面3「伝播」の素材(propagation=False で 1 行も読まない) ---- #
        self.propagation = bool(propagation)
        self.bel_cap = int(belief_cap)
        self.rsh_cap = int(reshare_cap)
        self.so_cap = int(stock_cap)
        self._bel_chunks: list = []     # step,agent,fact,from,hop,cause,src,x,y
        self._rsh_chunks: list = []     # step,agent,post,author,x,y
        self._vir_chunks: list = []     # step,author,post,reach
        self._so_chunks: list = []      # step,x,y,poi,cat
        self._tx_chunks: list = []      # step,agent,item,from,chan,x,y
        self.bel_pop = self.bel_kept = 0
        self.rsh_pop = self.rsh_kept = 0
        self.so_pop = self.so_kept = 0
        self.vir_rows = self.tx_rows = 0
        self.facts = _Intern()          # fact id 文字列 → 連番
        self.bel_causes, self.bel_srcs = _Intern(), _Intern()
        self.prop_chans, self.prop_topics = _Intern(), _Intern()
        self.pois, self.cats = _Intern(), _Intern()
        self.items = _Intern()          # vocab item_id
        self.prop_texts = _Intern(200_000)   # 投稿・語の実文
        self.bel_tx: list = []          # (step, from, fact, chan, topic, n_to)
        self.sns_posts: list = []       # (step, agent, text_id)
        self.coins: list = []           # (step, agent, item, text, place, x, y, is_vocab)
        self.adopts: list = []          # (step, agent, item, text)
        self.binds: list = []           # (step, agent, word_text, node, x, y)
        self.vocab_use: Counter = Counter()

        # ---- 画面4「物語ピン」の素材(stories=False で 1 行も読まない) ------- #
        # 遺失物・緊急・犯罪・集会は Run A で合わせて 600 行しかないので **dict のまま**
        # 持つ(読みやすさ優先)。唯一の大物 `nuisance`(32,718 行)だけは
        # (ノード × 1 時間)へその場で畳んで有界にする。
        self.stories = bool(stories)
        self.story_row_cap = int(story_row_cap)
        self.st: dict[str, list] = defaultdict(list)
        self.st_pop: Counter = Counter()        # kind ごとの母集団(cap 前の行数)
        self.st_kept = 0
        self.st_dropped = 0
        self.st_nodes = _Intern()               # ノード id / 交番名 / 場所名
        self.st_words = _Intern()               # 品目・重度・原因・題名など短い語
        #: (node_id, step // NUI_BIN_STEPS) -> [件数, 最初の step, 最後の step,
        #:                                      x, y, [代表 agent 8 人], Counter(kind)]
        self.nui: dict[tuple, list] = {}
        self.nui_pop = 0
        self.nui_dropped = 0

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

    # ---- 伝播素材(画面3) ------------------------------------------------- #
    def _propagation(self, batch, idx, kinds, xnp, ynp) -> None:
        """信念 / リシェア / 語彙の伝播を、**同じ形**の圧縮表へ落とす。

        1 パス目に相乗りする。Python オブジェクトになるのは PROP_KINDS の行だけ
        (Run A で 953,285 行 = 全 L1 の 2.35%)。文字列(fact id・場所名・チャネル・
        実文)はすべて辞書化する。
        """
        import numpy as np
        import pyarrow as pa
        take = pa.array(idx)
        steps = batch.column("step").take(take).to_pylist()
        aids = batch.column("agent_id").take(take).to_pylist()
        pays = batch.column("payload").take(take).to_pylist()
        xs = xnp[idx].tolist() if xnp is not None else [None] * len(steps)
        ys = ynp[idx].tolist() if ynp is not None else [None] * len(steps)

        b = [[] for _ in range(9)]      # belief_update
        r = [[] for _ in range(6)]      # sns_reshare
        v = [[] for _ in range(4)]      # viral_cascade
        s = [[] for _ in range(5)]      # stock_out
        t = [[] for _ in range(7)]      # transmission

        for st, a, kd, raw, x, y in zip(steps, aids, kinds, pays, xs, ys):
            if st is None:
                continue
            try:
                p = json.loads(raw) if raw else {}
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(p, dict):
                continue
            st = int(st)
            a = -1 if a is None else int(a)
            xi, yi = _xy16(x), _xy16(y)

            if kd == "belief_update":
                self.bel_pop += 1
                if self.bel_kept >= self.bel_cap or a < 0:
                    continue
                fid = p.get("fact")
                if fid is None:
                    continue
                try:
                    frm = int(p.get("from", -1))
                except (TypeError, ValueError):
                    frm = -1
                b[0].append(st)
                b[1].append(a)
                b[2].append(self.facts(fid))
                b[3].append(frm)
                b[4].append(max(0, min(255, int(p.get("hop") or 0))))
                b[5].append(self.bel_causes(p.get("cause")) + 1)
                b[6].append(self.bel_srcs(p.get("src")) + 1)
                b[7].append(xi)
                b[8].append(yi)
                self.bel_kept += 1

            elif kd == "belief_transmit":
                if len(self.bel_tx) < 200_000:
                    to = p.get("to")
                    self.bel_tx.append((st, a, self.facts(p.get("fact")),
                                        self.prop_chans(p.get("channel")),
                                        self.prop_topics(p.get("topic")),
                                        len(to) if isinstance(to, list) else 0))

            elif kd == "sns_reshare":
                self.rsh_pop += 1
                if self.rsh_kept >= self.rsh_cap or a < 0:
                    continue
                pid = p.get("post_id")
                if pid is None:
                    continue
                try:
                    au = int(p.get("author", -1))
                except (TypeError, ValueError):
                    au = -1
                r[0].append(st)
                r[1].append(a)
                r[2].append(int(pid))
                r[3].append(au)
                r[4].append(xi)
                r[5].append(yi)
                self.rsh_kept += 1

            elif kd == "viral_cascade":
                pid = p.get("post_id")
                if pid is None:
                    continue
                v[0].append(st)
                v[1].append(int(p.get("author") if p.get("author") is not None else -1))
                v[2].append(int(pid))
                v[3].append(int(p.get("reach") or 0))

            elif kd == "sns_post":
                if len(self.sns_posts) < 200_000:
                    self.sns_posts.append((st, a, self.prop_texts(p.get("text") or "")))

            elif kd == "stock_out":
                self.so_pop += 1
                if self.so_kept >= self.so_cap:
                    continue
                s[0].append(st)
                s[1].append(xi)
                s[2].append(yi)
                s[3].append(self.pois(p.get("poi")))
                s[4].append(self.cats(p.get("cat")))
                self.so_kept += 1

            elif kd == "transmission":
                it = p.get("item_id")
                if it is None or a < 0:
                    continue
                try:
                    frm = int(p.get("from", -1))
                except (TypeError, ValueError):
                    frm = -1
                t[0].append(st)
                t[1].append(a)
                t[2].append(self.items(it))
                t[3].append(frm)
                t[4].append(self.prop_chans(p.get("channel")) + 1)
                t[5].append(xi)
                t[6].append(yi)

            elif kd in ("vocab_coin", "label_coin"):
                if len(self.coins) < 200_000:
                    self.coins.append((st, a, self.items(p.get("item_id")),
                                       self.prop_texts(p.get("text") or ""),
                                       self.prop_topics(p.get("place")), xi, yi,
                                       1 if kd == "vocab_coin" else 0))

            elif kd == "label_adopt":
                if len(self.adopts) < 500_000:
                    self.adopts.append((st, a, self.items(p.get("item_id")),
                                        self.prop_texts(p.get("text") or "")))

            elif kd == "place_label_bind":
                if len(self.binds) < 200_000:
                    self.binds.append((st, a, self.prop_texts(p.get("word") or ""),
                                       str(p.get("node") or ""), xi, yi))

            elif kd == "vocab_use":
                self.vocab_use[self.items(p.get("item_id"))] += 1

        if b[0]:
            self._bel_chunks.append((
                np.asarray(b[0], dtype=np.int32), np.asarray(b[1], dtype=np.int64),
                np.asarray(b[2], dtype=np.int32), np.asarray(b[3], dtype=np.int64),
                np.asarray(b[4], dtype=np.uint8), np.asarray(b[5], dtype=np.uint8),
                np.asarray(b[6], dtype=np.uint8), np.asarray(b[7], dtype=np.int16),
                np.asarray(b[8], dtype=np.int16)))
        if r[0]:
            self._rsh_chunks.append((
                np.asarray(r[0], dtype=np.int32), np.asarray(r[1], dtype=np.int64),
                np.asarray(r[2], dtype=np.int64), np.asarray(r[3], dtype=np.int64),
                np.asarray(r[4], dtype=np.int16), np.asarray(r[5], dtype=np.int16)))
        if v[0]:
            self._vir_chunks.append((
                np.asarray(v[0], dtype=np.int32), np.asarray(v[1], dtype=np.int64),
                np.asarray(v[2], dtype=np.int64), np.asarray(v[3], dtype=np.int32)))
            self.vir_rows += len(v[0])
        if s[0]:
            self._so_chunks.append((
                np.asarray(s[0], dtype=np.int32), np.asarray(s[1], dtype=np.int16),
                np.asarray(s[2], dtype=np.int16), np.asarray(s[3], dtype=np.int32),
                np.asarray(s[4], dtype=np.int16)))
        if t[0]:
            self._tx_chunks.append((
                np.asarray(t[0], dtype=np.int32), np.asarray(t[1], dtype=np.int64),
                np.asarray(t[2], dtype=np.int32), np.asarray(t[3], dtype=np.int64),
                np.asarray(t[4], dtype=np.uint8), np.asarray(t[5], dtype=np.int16),
                np.asarray(t[6], dtype=np.int16)))
            self.tx_rows += len(t[0])

    # ---- 物語素材(画面4) ------------------------------------------------- #
    def _stories(self, batch, idx, kinds, xnp, ynp) -> None:
        """遺失物 / 緊急 / 逸脱 / 集会 の行を、**連鎖を組めるだけの欄**で持ち上げる。

        1 パス目に相乗りする。Python オブジェクトになるのは STORY_KINDS の行だけ
        (Run A で 33,313 行 = 全 L1 の 0.082%)。ノード名・品目・重度は辞書化する。
        """
        import pyarrow as pa
        take = pa.array(idx)
        steps = batch.column("step").take(take).to_pylist()
        aids = batch.column("agent_id").take(take).to_pylist()
        pays = batch.column("payload").take(take).to_pylist()
        xs = xnp[idx].tolist() if xnp is not None else [None] * len(steps)
        ys = ynp[idx].tolist() if ynp is not None else [None] * len(steps)

        def _i(v, dflt=-1):
            try:
                return int(v)
            except (TypeError, ValueError):
                return dflt

        for st, a, kd, raw, x, y in zip(steps, aids, kinds, pays, xs, ys):
            if st is None:
                continue
            self.st_pop[kd] += 1
            try:
                p = json.loads(raw) if raw else {}
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(p, dict):
                continue
            st = int(st)
            a = -1 if a is None else int(a)
            xi, yi = _xy16(x), _xy16(y)

            if kd == "nuisance":
                # 32,718 行。1 件ずつ持たず (ノード × 1 時間) のセルへ畳む。
                self.nui_pop += 1
                key = (self.st_nodes(str(p.get("node") or "")), st // NUI_BIN_STEPS)
                cell = self.nui.get(key)
                if cell is None:
                    if len(self.nui) >= NUI_CELL_CAP:
                        self.nui_dropped += 1
                        continue
                    cell = [0, st, st, xi, yi, [], Counter()]
                    self.nui[key] = cell
                cell[0] += 1
                cell[1] = min(cell[1], st)
                cell[2] = max(cell[2], st)
                if cell[3] == -32768 and xi != -32768:
                    cell[3], cell[4] = xi, yi
                if a >= 0 and len(cell[5]) < 8 and a not in cell[5]:
                    cell[5].append(a)
                cell[6][str(p.get("kind") or "?")] += 1
                continue

            if self.st_kept >= self.story_row_cap:
                self.st_dropped += 1
                continue

            row = {"s": st, "a": a, "x": xi, "y": yi}
            if kd.startswith("lost_"):
                row["item"] = self.st_words(str(p.get("item") or ""))
                # `node` = 物が在った場所 / `post` = 最寄りの交番。**別の量**なので分ける
                # (混ぜると「交番へ届けた(拾った店の名前)」という嘘の行になる)。
                row["node"] = self.st_nodes(str(p.get("node") or ""))
                if p.get("post"):
                    row["post"] = self.st_nodes(str(p.get("post")))
                if kd == "lost_drop":
                    row["owner"] = a
                    row["cash"] = _round(p.get("cash") or 0.0, 1)
                    row["crowd"] = _i(p.get("crowd"), 0)
                    row["drinking"] = bool(p.get("drinking"))
                    row["rain"] = bool(p.get("rain"))
                    row["drop"] = st
                elif kd == "lost_notice":
                    row["owner"] = a
                    row["drop"] = st - _i(p.get("delay_steps"), 0)
                elif kd in ("lost_pickup", "lost_turnin", "lost_keep"):
                    row["owner"] = _i(p.get("owner"))
                    row["finder"] = a
                    row["guardians"] = _i(p.get("guardians"), -1)
                    if kd == "lost_pickup":
                        row["drop"] = st - _i(p.get("lying_steps"), 0)
                    if kd == "lost_keep":
                        row["amount"] = _round(p.get("amount") or 0.0, 1)
                        row["offense"] = self.st_words(str(p.get("offense") or ""))
                elif kd == "lost_return":
                    row["owner"] = a
                    row["finder"] = _i(p.get("finder"))
                    row["amount"] = _round(p.get("amount") or 0.0, 1)
                    row["cash"] = _round(p.get("cash") or 0.0, 1)
                    row["turnin"] = st - _i(p.get("held_steps"), 0)
                else:                                    # lost_expire
                    row["owner"] = _i(p.get("owner"))
                    row["finder"] = a
                    row["to"] = self.st_words(str(p.get("to") or ""))
                    row["amount"] = _round(p.get("amount") or 0.0, 1)
                    row["drop"] = st - _i(p.get("age_steps"), 0)

            elif kd in ("collapse", "injury", "traffic_accident"):
                row["patient"] = a if kd == "collapse" else _i(p.get("victim"), a)
                row["node"] = self.st_nodes(str(p.get("node") or ""))
                row["src"] = self.st_words(str(p.get("source") or p.get("severity") or ""))
                row["sev"] = _i(p.get("sev", p.get("severity")), -1)
                if kd == "traffic_accident":
                    row["ped_n"] = _i(p.get("ped_n"), 0)
                    row["signalized"] = bool(p.get("signalized"))

            elif kd in ("ems_call", "ems_dispatch", "ems_transport"):
                row["patient"] = _i(p.get("patient"), -1)
                row["node"] = self.st_nodes(str(p.get("node") or ""))
                if kd == "ems_call":
                    row["self_call"] = bool(p.get("self_call"))
                    row["dist_m"] = _round(p.get("dist_m"), 1)
                    row["bystanders"] = _i(p.get("bystanders"), -1)
                elif kd == "ems_dispatch":
                    row["unstaffed"] = bool(p.get("unstaffed"))
                    row["response_min"] = _round(p.get("response_min"), 1)
                    row["crew"] = _i(p.get("crew"), -1)
                else:
                    row["poi"] = self.st_nodes(str(p.get("poi") or ""))
                    row["confirmed"] = _i(p.get("confirmed"), -1)
                    row["cost"] = _round(p.get("cost"), 1)

            elif kd in ("hospital_admit", "hospital_discharge"):
                row["patient"] = a
                row["poi"] = self.st_nodes(str(p.get("poi") or ""))
                row["confirmed"] = _i(p.get("confirmed"), -1)
                row["days"] = _round(p.get("days", p.get("billed_days")), 2)

            elif kd == "crime":
                row["ckind"] = self.st_words(str(p.get("kind") or ""))
                row["victim"] = _i(p.get("victim"), -1)
                row["offender"] = _i(p.get("offender"), a)
                row["amount"] = _round(p.get("amount") or 0.0, 1)

            elif kd == "police_response":
                row["node"] = self.st_nodes(str(p.get("node") or ""))
                row["about"] = self.st_words(str(p.get("kind") or p.get("about") or ""))

            elif kd == "event_host":
                row["eid"] = _i(p.get("event_id"), -1)
                row["title"] = self.st_words(str(p.get("title") or ""))
                row["place"] = self.st_nodes(str(p.get("place") or p.get("node") or ""))
                row["start"] = _i(p.get("start_step"), st)

            elif kd == "event_attend":
                row["eid"] = _i(p.get("event_id"), -1)
                row["host"] = _i(p.get("host"), -1)
                row["title"] = self.st_words(str(p.get("title") or ""))

            self.st[kd].append(row)
            self.st_kept += 1

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
        prop_idx, prop_kinds = None, None
        story_idx, story_kinds = None, None
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
            if self.propagation:
                prop_idx, prop_kinds = _pick(PROP_KINDS)
            if self.stories:
                story_idx, story_kinds = _pick(STORY_KINDS)

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
        if prop_idx is not None:
            self._propagation(batch, prop_idx, prop_kinds, xnp, ynp)
        if story_idx is not None:
            self._stories(batch, story_idx, story_kinds, xnp, ynp)

    def finish(self) -> None:
        self._flush(self._cur_bin)
        self._cur_bin = -1


def scan_l1(run_dir, *, hex_m: float, steps_per_bin: int, rel_cap: int,
            step_max=None, batch_rows: int = 262_144, narrative: bool = True,
            conv_cap: int = CONV_CAP_DEFAULT,
            text_cap: int = TEXT_CAP_DEFAULT, propagation: bool = True,
            belief_cap: int = BELIEF_CAP_DEFAULT,
            reshare_cap: int = RESHARE_CAP_DEFAULT,
            stock_cap: int = STOCK_CAP_DEFAULT, stories: bool = True) -> _Scan:
    """L1 を 1 パスで舐める。`l1_stream` の有界読みだけを使う。"""
    import l1_stream
    sc = _Scan(hex_m, steps_per_bin, rel_cap, narrative=narrative,
               conv_cap=conv_cap, text_cap=text_cap, propagation=propagation,
               belief_cap=belief_cap, reshare_cap=reshare_cap, stock_cap=stock_cap,
               stories=stories)
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
    if sc.propagation:
        _log(f"  伝播素材: belief_update {sc.bel_kept:,}/{sc.bel_pop:,} 行"
             f"(fact {len(sc.facts):,} 件)・sns_reshare {sc.rsh_kept:,}/{sc.rsh_pop:,} 行・"
             f"viral_cascade {sc.vir_rows:,}・transmission {sc.tx_rows:,}・"
             f"coin {len(sc.coins):,}・stock_out {sc.so_kept:,}/{sc.so_pop:,}")
    if sc.stories:
        top = ", ".join(f"{k} {v:,}" for k, v in sc.st_pop.most_common(6))
        _log(f"  物語素材: {sc.st_kept:,} 行(母集団 {sum(sc.st_pop.values()):,})・"
             f"迷惑行為 {sc.nui_pop:,} 行 → {len(sc.nui):,} セル・上位[{top}]")
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
# 画面3「伝播」— 信念 / リシェア / 語彙のカスケード
#
# 方針(計画 §1 画面3)
# --------------------
# * **器は 1 つ**。信念・リシェア・語彙のどれも `{木・等時線・S字・チャネル分解}` の
#   同じ 4 点セットにする。Run A では語彙がほぼ 0 件だが、空でも壊れない形で出す
#   (Run B の語彙イベントが来たらそのまま点灯する)。
# * 母集団(全カスケード)と掲載(cap 件)は必ず併記する。
# * 木は「見て分かる」ことが目的なので、ノードは上限つき。間引くときは
#   **内部ノード(子を持つ節)を全部残してから**葉を等間隔で落とす
#   = 形(タンポポ / 連鎖)が保存される。
# * 位置は採用イベントの (x, y)。等時線は俯瞰と同じヘックス格子の
#   「そのセルに初めて届いた step」。
# --------------------------------------------------------------------------- #
#: `stock_out.poi` が POI 名でなく地図ノード id のことがある(名前のない店)。
_NODE_ID_RE = re.compile(r"n\d{4,}")


def _tree_pack(agents, parents, steps, xs, ys, depths, *, cap):
    """ノード列 → base64。cap 超過は内部ノード優先で間引く(形を壊さない)。

    間引いた節の子は、**残っている最も近い祖先へ付け替える**(= 木の収縮)。
    親を落として根にしてしまうと「独立に発生した」と誤読されるので、そうしない。
    付け替えた辺の本数は `contracted` として必ず併記する。
    """
    import numpy as np
    pop = int(agents.size)
    if pop == 0:
        return {"n": 0, "population": 0, "contracted": 0, "a": "", "p": "",
                "s": "", "x": "", "y": "", "d": ""}
    keep = np.arange(pop, dtype=np.int64)
    if pop > cap:
        # 各節の**子孫の数**で優先順位をつける。親の子孫数は必ず子より大きいので、
        # 「子孫数の多い順に上から cap 個」は**祖先について閉じている**
        # = 付け替え(contraction)が 1 本も起きず、幹と太い枝がそのまま残る。
        size = np.ones(pop, dtype=np.int64)
        par_i = parents.tolist()
        sz = size.tolist()
        for i in range(pop - 1, -1, -1):        # 親 index < 子 index なので 1 パス
            p = par_i[i]
            if p >= 0:
                sz[p] += sz[i]
        size = np.asarray(sz, dtype=np.int64)
        trunk = np.nonzero(size > 1)[0]         # 子を持つ節(幹と枝)
        sel = []
        budget = int(cap)
        if trunk.size:
            if trunk.size > budget:
                # 子孫数の多い順。同数は index 昇順(= 早い者順)で安定に切る。
                ordv = trunk[np.lexsort((trunk, -size[trunk]))][:budget]
                sel.append(ordv)
                budget = 0
            else:
                sel.append(trunk)
                budget -= int(trunk.size)
        if budget > 0:                           # 残枠は葉を step 昇順で等間隔に
            leaves = np.nonzero(size <= 1)[0]
            if leaves.size > budget:
                leaves = leaves[np.linspace(0, leaves.size - 1,
                                            budget).astype(np.int64)]
            if leaves.size:
                sel.append(leaves)
        keep = np.sort(np.concatenate(sel)) if sel else np.zeros(0, dtype=np.int64)
    remap = np.full(pop, -1, dtype=np.int64)
    remap[keep] = np.arange(keep.size, dtype=np.int64)
    # 残った最近祖先(親 index < 自分 index なので 1 パスで解ける)
    anc = np.full(pop, -1, dtype=np.int64)
    par_l = parents.tolist()
    rm_l = remap.tolist()
    anc_l = anc.tolist()
    for i in range(pop):
        p = par_l[i]
        if p < 0:
            anc_l[i] = -1
        else:
            anc_l[i] = rm_l[p] if rm_l[p] >= 0 else anc_l[p]
    anc = np.asarray(anc_l, dtype=np.int64)
    npar = anc[keep]
    contracted = int(np.count_nonzero((parents[keep] >= 0)
                                      & (remap[np.clip(parents[keep], 0, pop - 1)] < 0)
                                      & (npar >= 0)))
    return {
        "n": int(keep.size),
        "population": pop,
        "contracted": contracted,
        "depth_shown": int(depths[keep].max()) if keep.size else 0,
        "a": _b64(agents[keep].astype(np.int32)),      # -1 = 公式(メディア)
        "p": _b64(npar.astype(np.int32)),
        "s": _b64(steps[keep].astype(np.int32)),
        "x": _b64(xs[keep].astype(np.int16)),
        "y": _b64(ys[keep].astype(np.int16)),
        "d": _b64(np.clip(depths[keep], 0, 255).astype(np.uint8)),
    }


def _iso_grid(xs, ys, hex_m: float) -> HexGrid:
    """カスケードの広がりに合わせたヘックス格子(1 枚に 20-40 個くらい並ぶ大きさ)。

    信念は目撃半径のなかに全員が居るので 200 m 格子だと 1 個に潰れ、リシェアは
    街全体に散るので 25 m 格子だと数千個になる。**同じ 200 m を全部に当てない**。
    """
    import numpy as np
    ok = (xs != -32768) & (ys != -32768)
    if int(np.count_nonzero(ok)) < 2:
        return HexGrid(hex_m)
    w = float(xs[ok].max() - xs[ok].min())
    h = float(ys[ok].max() - ys[ok].min())
    span = max(w, h)
    if span <= 0:
        return HexGrid(25.0)
    return HexGrid(min(400.0, max(25.0, round(span / 16.0))))


def _iso_cells(grid: HexGrid, xs, ys, steps, *, cap=CAS_ISO_CELLS):
    """採用者の位置 → ヘックス別の (q, r, 初到達 step, 人数)。母集団も返す。"""
    import numpy as np
    ok = (xs != -32768) & (ys != -32768)
    if not ok.any():
        return [], 0
    q, r = grid.to_axial(xs[ok].astype(np.float64), ys[ok].astype(np.float64))
    code = _encode_cell(q, r)
    st = steps[ok].astype(np.int64)
    o = np.lexsort((st, code))
    code, st = code[o], st[o]
    starts, ends = _group_bounds(code)
    rows = []
    for a, b in zip(starts.tolist(), ends.tolist()):
        qq, rr = _decode_cell(int(code[a]))
        rows.append([int(qq), int(rr), int(st[a]), int(b - a)])
    pop = len(rows)
    if pop > cap:
        # 半分は人数の多いセル(波紋の芯)、残り半分は初到達 step 順に等間隔
        # (人数だけで切ると遅く届いた外周が丸ごと消えて「波が止まる」)。
        rows.sort(key=lambda v: (-v[3], v[2]))
        head = rows[: cap // 2]
        taken = {(v[0], v[1]) for v in head}
        rest = [v for v in rows if (v[0], v[1]) not in taken]
        rest.sort(key=lambda v: v[2])
        need = cap - len(head)
        if rest and need > 0:
            import numpy as _np
            ix = _np.linspace(0, len(rest) - 1, min(need, len(rest)))
            head += [rest[int(round(j))] for j in ix.tolist()]
        rows = head
    rows.sort(key=lambda v: v[2])
    return rows, pop


def _curve(steps, *, cap=CAS_CURVE_MAX):
    """step ごとの新規採用者数(累積はビューワー側で取る)。長すぎるときは畳む。"""
    import numpy as np
    if steps.size == 0:
        return []
    u, c = np.unique(steps.astype(np.int64), return_counts=True)
    if u.size <= cap:
        return [[int(a), int(b)] for a, b in zip(u.tolist(), c.tolist())]
    lo, hi = int(u[0]), int(u[-1])
    w = max(1, int(math.ceil((hi - lo + 1) / cap)))
    b = (u - lo) // w
    ub, ix = np.unique(b, return_inverse=True)
    agg = np.zeros(ub.size, dtype=np.int64)
    np.add.at(agg, ix, c)
    return [[int(lo + int(bb) * w), int(v)] for bb, v in zip(ub.tolist(), agg.tolist())]


def _shape_of(edges: int, depth: int, internal: int, n: int, roots: int) -> str:
    """木の形の分類。分岐数 = 辺 / 子を持つ節(= 1 人が平均何人へ渡したか)。

    parallel … 辺ゼロ(= 全員が独立に知った。Run A の信念はほぼこれ)
    seeded   … 辺はあるが 8 割超が独立発生(= ほぼ並列・一部だけ人づて)
    fan      … 深さ 1(= 1 人から一斉に広がったタンポポ)
    chain    … 分岐 1.5 未満(= 数珠つなぎ)
    tree     … 分岐 4 以上(= よく枝分かれした)
    mixed    … その中間
    """
    if edges <= 0:
        return "parallel"
    if n > 4 and roots > 0.8 * n:
        return "seeded"
    if depth <= 1:
        return "fan"
    branch = edges / float(max(1, internal))
    if branch < 1.5:
        return "chain"
    return "tree" if branch >= 4.0 else "mixed"


def _cascade_item(kind, key, *, title, title_src, text, agents, parents, steps,
                  xs, ys, depths, hex_m, nodes_cap, ch, who=None, extra=None):
    """1 カスケードぶんの 4 点セット(木 / 等時線 / S 字 / チャネル)を組む。"""
    import numpy as np
    n = int(agents.size)
    edges = int(np.count_nonzero(parents >= 0))
    roots = n - edges
    depth = int(depths.max()) if n else 0
    pp = parents[parents >= 0]
    internal = 0
    if pp.size:
        up, uc = np.unique(pp, return_counts=True)
        internal = int(up.size)
        if who is not None:                     # よく配った人は名前を引く(木の吹き出し)
            top = up[np.argsort(-uc)[:12]]
            for j in top.tolist():
                if 0 <= j < n:
                    who.add(int(agents[j]))
    grid = _iso_grid(xs, ys, hex_m)
    it = {
        "kind": int(kind),
        "key": str(key),
        "title": title,
        "title_src": title_src,
        "text": text,
        "n": n,
        "edges": edges,
        "roots": roots,
        "internal": internal,
        "branch": _round(edges / float(max(1, internal)), 2),
        "depth": depth,
        "shape": _shape_of(edges, depth, internal, n, roots),
        "s0": int(steps.min()) if n else 0,
        "s1": int(steps.max()) if n else 0,
        "ch": {k: int(v) for k, v in sorted(ch.items(), key=lambda kv: -kv[1])},
        "curve": _curve(steps),
        "tree": _tree_pack(agents, parents, steps, xs, ys, depths, cap=nodes_cap),
        "iso_hex_m": round(grid.hex_m, 1),
    }
    iso, iso_pop = _iso_cells(grid, xs, ys, steps)
    it["iso"] = iso
    it["iso_population"] = iso_pop
    if extra:
        it.update(extra)
    return it


def build_cascades(run_dir, sc: _Scan, *, hex_m: float, cap: int, nodes_cap: int,
                   n_steps: int) -> dict:
    """信念 / リシェア / 語彙のカスケードを 1 つの器に組む(L1 の追加走査なし)。"""
    import numpy as np
    t0 = time.time()
    pop = {
        "belief_rows": sc.bel_pop, "belief_kept": sc.bel_kept,
        "belief_facts": 0, "belief_facts_with_edges": 0, "belief_edges": 0,
        "reshare_rows": sc.rsh_pop, "reshare_kept": sc.rsh_kept,
        "reshare_cascades": 0, "viral_rows": sc.vir_rows,
        "vocab_rows": sc.tx_rows, "vocab_items": 0,
        "coins": len(sc.coins), "label_adopts": len(sc.adopts),
        "place_binds": len(sc.binds), "stock_out_rows": sc.so_pop,
    }
    cands: list[tuple] = []             # (kind, サイズ, 優先度, 生成関数)
    who: set[int] = set()               # 名前を引く体(根・主要な媒介者)

    # ---- 1. 信念 --------------------------------------------------------- #
    bel = _concat_chunks(sc._bel_chunks, 9)
    sc._bel_chunks.clear()
    bel_groups: dict[int, tuple] = {}
    if bel is not None:
        b_st, b_ag, b_fa, b_fr, b_hp, b_ca, b_sr, b_x, b_y = bel
        o = np.lexsort((b_st, b_fa))
        b_st, b_ag, b_fa = b_st[o], b_ag[o], b_fa[o]
        b_fr, b_hp, b_ca, b_sr = b_fr[o], b_hp[o], b_ca[o], b_sr[o]
        b_x, b_y = b_x[o], b_y[o]
        starts, ends = _group_bounds(b_fa.astype(np.int64))
        pop["belief_facts"] = int(starts.size)
        for s, e in zip(starts.tolist(), ends.tolist()):
            fid = int(b_fa[s])
            edges = int(np.count_nonzero(b_fr[s:e] >= 0))
            pop["belief_edges"] += edges
            if edges:
                pop["belief_facts_with_edges"] += 1
            bel_groups[fid] = (s, e, int(e - s), edges)
        for fid, (s, e, n, edges) in bel_groups.items():
            # 人づて(hop>=1)が 1 本でもある fact は稀少 = 必ず候補の先頭へ
            cands.append((CAS_BELIEF, n, 0 if edges else 1, fid))

    # fact → 場所名。厳密な出典は belief_transmit.topic、無ければ stock_out から推定。
    fact_topic: dict[int, int] = {}
    fact_tx_ch: dict[int, Counter] = defaultdict(Counter)
    for st, frm, fa, ch, tp, n_to in sc.bel_tx:
        if fa >= 0 and tp >= 0:
            fact_topic.setdefault(fa, tp)
        if fa >= 0:
            fact_tx_ch[fa][ch] += 1
    so = _concat_chunks(sc._so_chunks, 5)
    sc._so_chunks.clear()
    if so is not None:
        so_order = np.argsort(so[0], kind="stable")
        so = [c[so_order] for c in so]

    def _fact_name(fid, s, e):
        """fact の実名。① belief_transmit の topic(厳密)② 最寄の stock_out(推定)。"""
        tp = fact_topic.get(fid)
        if tp is not None and 0 <= tp < len(sc.prop_topics.vals):
            return str(sc.prop_topics.vals[tp]), "transmit_topic", None
        if so is None:
            return None, "none", None
        s0 = int(b_st[s])
        # fact が立った step の目撃者だけで重心を取る(半径内に散るので中央値)
        m = b_st[s:e] == s0
        xs0 = b_x[s:e][m]
        ys0 = b_y[s:e][m]
        xs0 = xs0[xs0 != -32768]
        ys0 = ys0[ys0 != -32768]
        if xs0.size == 0:
            return None, "none", None
        cx, cy = float(np.median(xs0)), float(np.median(ys0))
        lo = int(np.searchsorted(so[0], s0 - 3, side="left"))
        hi = int(np.searchsorted(so[0], s0, side="right"))
        if hi <= lo:
            return None, "none", None
        dx = so[1][lo:hi].astype(np.float64) - cx
        dy = so[2][lo:hi].astype(np.float64) - cy
        d2 = dx * dx + dy * dy
        j = int(np.argmin(d2))
        poi = so[3][lo + j]
        nm = sc.pois.vals[int(poi)] if 0 <= int(poi) < len(sc.pois.vals) else None
        if not nm:
            return None, "none", None
        nm = str(nm)
        if _NODE_ID_RE.fullmatch(nm):        # POI 名でなく地図ノード id(名前のない店)
            nm = f"名称のない店({nm})"
        return nm, "poi_near", round(float(math.sqrt(d2[j])), 1)

    # ---- 2. リシェア ------------------------------------------------------ #
    rsh = _concat_chunks(sc._rsh_chunks, 6)
    sc._rsh_chunks.clear()
    rsh_groups: dict[int, list] = {}
    par_ev = root_of = depth_ev = None
    if rsh is not None:
        o = np.argsort(rsh[0], kind="stable")
        r_st, r_ag, r_po, r_au, r_x, r_y = [c[o] for c in rsh]
        n_ev = int(r_st.size)
        # 「X が post P をリシェアすると RT post が 1 件生まれる」が、その id は L1 に
        # 出ない。post id は追記通し番号(= 生成時刻の昇順)なので、著者 A について
        #   観測された post id(昇順) × A が RT を作った step(昇順)
        # を単調に突き合わせれば A の k 番目の RT post が同定できる。
        created: dict[int, list] = defaultdict(list)
        seen: dict[int, dict] = defaultdict(dict)
        ag_l, po_l, au_l = r_ag.tolist(), r_po.tolist(), r_au.tolist()
        for i in range(n_ev):
            created[ag_l[i]].append(i)
            d = seen[au_l[i]]
            if po_l[i] not in d:
                d[po_l[i]] = i
        creator: dict[int, int] = {}
        st_l = r_st.tolist()
        for au, posts in seen.items():
            if au < 0:                       # 公式(メディア)は RT で作られない
                continue
            cre = created.get(au) or ()
            j = 0
            for pid in sorted(posts):
                if j >= len(cre):
                    break
                if st_l[cre[j]] <= st_l[posts[pid]]:
                    creator[pid] = cre[j]
                    j += 1
        par_ev = np.asarray([creator.get(p, -1) for p in po_l], dtype=np.int64)
        root_of = np.full(n_ev, -1, dtype=np.int64)
        depth_ev = np.zeros(n_ev, dtype=np.int64)
        pe = par_ev.tolist()
        for i in range(n_ev):
            if root_of[i] >= 0:
                continue
            chain = []
            j, root = i, -1
            for _ in range(512):
                if root_of[j] >= 0:
                    root = int(root_of[j])
                    break
                chain.append(j)
                nj = pe[j]
                if nj < 0:
                    root = po_l[j]
                    break
                j = nj
            if root < 0:
                root = po_l[chain[-1]] if chain else po_l[i]
            for c in chain:
                root_of[c] = root
        for i in range(n_ev):
            p = pe[i]
            depth_ev[i] = 1 if p < 0 else int(depth_ev[p]) + 1
        ro = np.argsort(root_of, kind="stable")
        rk = root_of[ro]
        starts, ends = _group_bounds(rk)
        pop["reshare_cascades"] = int(starts.size)
        for s, e in zip(starts.tolist(), ends.tolist()):
            rsh_groups[int(rk[s])] = [ro[s:e], int(e - s)]
            cands.append((CAS_RESHARE, int(e - s), 1, int(rk[s])))

    # viral_cascade の reach を「その post を含むカスケード」へ寄せる
    vir = _concat_chunks(sc._vir_chunks, 4)
    sc._vir_chunks.clear()
    vir_reach: dict[int, int] = defaultdict(int)
    vir_hits: dict[int, int] = defaultdict(int)
    if vir is not None and rsh is not None:
        for pid, rc in zip(vir[2].tolist(), vir[3].tolist()):
            ev = creator.get(pid)
            root = int(root_of[ev]) if ev is not None else pid
            vir_reach[root] += int(rc)
            vir_hits[root] += 1

    # 元 post の実文(sns_post は 著者 × step でしか引けないので「その著者の直近投稿」)
    posts_by_author: dict[int, list] = defaultdict(list)
    for st, ag, tid in sc.sns_posts:
        posts_by_author[int(ag)].append((int(st), int(tid)))
    for v in posts_by_author.values():
        v.sort()

    def _post_text(author: int, before: int):
        v = posts_by_author.get(int(author)) or ()
        best = None
        for st, tid in v:
            if st <= before:
                best = tid
            else:
                break
        if best is None:
            return None
        return sc.prop_texts.vals[best] if 0 <= best < len(sc.prop_texts.vals) else None

    # ---- 3. 語彙 ---------------------------------------------------------- #
    tx = _concat_chunks(sc._tx_chunks, 7)
    sc._tx_chunks.clear()
    tx_groups: dict[int, tuple] = {}
    if tx is not None:
        o = np.lexsort((tx[0], tx[2]))
        t_st, t_ag, t_it, t_fr, t_ch, t_x, t_y = [c[o] for c in tx]
        starts, ends = _group_bounds(t_it.astype(np.int64))
        pop["vocab_items"] = int(starts.size)
        for s, e in zip(starts.tolist(), ends.tolist()):
            tx_groups[int(t_it[s])] = (s, e)
            cands.append((CAS_VOCAB, int(e - s), 0, int(t_it[s])))
    # 語の造語イベント(伝播 0 件でも「生まれた」ことは載せる)
    coin_of: dict[int, tuple] = {}
    for st, ag, it_, tid, place, xi, yi, is_vocab in sc.coins:
        if it_ >= 0 and it_ not in coin_of:
            coin_of[it_] = (st, ag, tid, place, xi, yi)
            if it_ not in tx_groups:
                cands.append((CAS_VOCAB, 0, 2, it_))
    adopt_of: dict[int, list] = defaultdict(list)
    for st, ag, it_, tid in sc.adopts:
        adopt_of[it_].append((int(st), int(ag)))

    # ---- 4. 選抜(種別ごとの枠 → 残りは規模順) ---------------------------- #
    cands.sort(key=lambda c: (c[2], -c[1]))
    chosen: list[tuple] = []
    seen_keys: set = set()
    quota_filled = Counter()
    for kd, quota in CAS_QUOTA.items():
        got = 0
        for c in cands:
            if got >= quota or len(chosen) >= cap:
                break
            if c[0] != kd or (c[0], c[3]) in seen_keys:
                continue
            chosen.append(c)
            seen_keys.add((c[0], c[3]))
            got += 1
        quota_filled[kd] = got
    for c in cands:
        if len(chosen) >= cap:
            break
        if (c[0], c[3]) not in seen_keys:
            chosen.append(c)
            seen_keys.add((c[0], c[3]))
    chosen.sort(key=lambda c: (c[2], -c[1]))

    # ---- 5. 掲載ぶんを組む ------------------------------------------------- #
    items: list[dict] = []
    for kd, size, pri, key in chosen:
        if kd == CAS_BELIEF:
            s, e, n, edges = bel_groups[key]
            ag = b_ag[s:e].astype(np.int64)
            fr = b_fr[s:e].astype(np.int64)
            st = b_st[s:e].astype(np.int64)
            first: dict[int, int] = {}
            for i, a in enumerate(ag.tolist()):
                first.setdefault(a, i)
            par = np.asarray([first.get(f, -1) if f >= 0 else -1
                              for f in fr.tolist()], dtype=np.int64)
            par = np.where(par < np.arange(par.size), par, -1)   # 前向きの辺だけ
            ch = Counter()
            for c, cnt in zip(*np.unique(b_ca[s:e], return_counts=True)):
                nm = sc.bel_causes.vals[int(c) - 1] if 0 < int(c) <= len(sc.bel_causes.vals) else "?"
                ch[str(nm)] += int(cnt)
            for c, cnt in fact_tx_ch.get(key, {}).items():
                nm = sc.prop_chans.vals[int(c)] if 0 <= int(c) < len(sc.prop_chans.vals) else "?"
                ch[f"発話({nm})"] += int(cnt)
            nm, src, dist = _fact_name(key, s, e)
            fid_s = sc.facts.vals[key] if 0 <= key < len(sc.facts.vals) else str(key)
            title = (f"{nm} が品切れ" if nm else f"事実 {fid_s}")
            it = _cascade_item(
                CAS_BELIEF, fid_s, title=title, title_src=src, text=None,
                agents=ag, parents=par, steps=st, xs=b_x[s:e], ys=b_y[s:e],
                depths=b_hp[s:e].astype(np.int64), hex_m=hex_m, nodes_cap=nodes_cap,
                ch=ch, who=who, extra={"place": nm, "place_dist_m": dist,
                                       "fact_kind": "stock_out"})
            for f in fr.tolist():
                if f >= 0:
                    who.add(int(f))
            if ag.size:
                who.add(int(ag[0]))
            items.append(it)

        elif kd == CAS_RESHARE:
            ix, n = rsh_groups[key]
            ag = r_ag[ix].astype(np.int64)
            st = r_st[ix].astype(np.int64)
            xs_ = r_x[ix]
            ys_ = r_y[ix]
            dp = depth_ev[ix].astype(np.int64)
            loc = {int(g): i for i, g in enumerate(ix.tolist())}
            par = np.asarray([loc.get(int(p), -1) for p in par_ev[ix].tolist()],
                             dtype=np.int64)
            # 根(元 post の著者)を 0 番のノードとして先頭に足す
            root_au = int(r_au[ix[0]])
            ag = np.concatenate([[root_au], ag])
            st = np.concatenate([[int(st.min())], st])
            xs_ = np.concatenate([np.asarray([-32768], dtype=np.int16), xs_])
            ys_ = np.concatenate([np.asarray([-32768], dtype=np.int16), ys_])
            dp = np.concatenate([[0], dp])
            par = np.concatenate([[-1], np.where(par >= 0, par + 1, 0)])
            txt = _post_text(root_au, int(st[0])) if root_au >= 0 else None
            if root_au < 0:
                title = f"公式(メディア)の投稿 #{key}"
                tsrc = "media"
            elif txt:
                title = txt[:40]
                tsrc = "post_text"
            else:
                title = f"#{root_au} の投稿 #{key}"
                tsrc = "none"
            it = _cascade_item(
                CAS_RESHARE, f"post:{key}", title=title, title_src=tsrc, text=txt,
                agents=ag, parents=par, steps=st, xs=xs_, ys=ys_, depths=dp,
                hex_m=hex_m, nodes_cap=nodes_cap, ch=Counter({"sns": int(n)}),
                who=who, extra={"post_id": int(key), "author": root_au,
                       "reach": int(vir_reach.get(key, 0)),
                       "viral_events": int(vir_hits.get(key, 0))})
            who.add(root_au)
            items.append(it)

        else:                                     # CAS_VOCAB
            item_s = sc.items.vals[key] if 0 <= key < len(sc.items.vals) else str(key)
            coin = coin_of.get(key)
            g = tx_groups.get(key)
            if g is not None:
                s, e = g
                ag = t_ag[s:e].astype(np.int64)
                st = t_st[s:e].astype(np.int64)
                fr = t_fr[s:e].astype(np.int64)
                xs_ = t_x[s:e]
                ys_ = t_y[s:e]
                ch = Counter()
                for c, cnt in zip(*np.unique(t_ch[s:e], return_counts=True)):
                    nm = (sc.prop_chans.vals[int(c) - 1]
                          if 0 < int(c) <= len(sc.prop_chans.vals) else "?")
                    ch[str(nm)] += int(cnt)
            else:
                ag = np.zeros(0, dtype=np.int64)
                st = np.zeros(0, dtype=np.int64)
                fr = np.zeros(0, dtype=np.int64)
                xs_ = np.zeros(0, dtype=np.int16)
                ys_ = np.zeros(0, dtype=np.int16)
                ch = Counter()
            if coin is not None:                  # 造語者を 0 番のノードに置く
                ag = np.concatenate([[int(coin[1])], ag])
                st = np.concatenate([[int(coin[0])], st])
                fr = np.concatenate([[-1], fr])
                xs_ = np.concatenate([np.asarray([coin[4]], dtype=np.int16), xs_])
                ys_ = np.concatenate([np.asarray([coin[5]], dtype=np.int16), ys_])
            first = {}
            for i, a in enumerate(ag.tolist()):
                first.setdefault(a, i)
            par = np.asarray([first.get(f, -1) if f >= 0 else -1
                              for f in fr.tolist()], dtype=np.int64)
            par = np.where(par < np.arange(par.size), par, -1)
            dp = np.zeros(par.size, dtype=np.int64)
            for i in range(par.size):
                p = int(par[i])
                dp[i] = 0 if p < 0 else int(dp[p]) + 1
            text = None
            if coin is not None and 0 <= coin[2] < len(sc.prop_texts.vals):
                text = sc.prop_texts.vals[coin[2]]
            it = _cascade_item(
                CAS_VOCAB, item_s, title=(f"「{text}」" if text else item_s),
                title_src=("coin_text" if text else "none"), text=text,
                agents=ag, parents=par, steps=st, xs=xs_, ys=ys_, depths=dp,
                hex_m=hex_m, nodes_cap=nodes_cap, ch=ch, who=who,
                extra={"coin_step": (int(coin[0]) if coin else None),
                       "coiner": (int(coin[1]) if coin else None),
                       "coin_place": (sc.prop_topics.vals[coin[3]]
                                      if coin and 0 <= coin[3] < len(sc.prop_topics.vals)
                                      else None),
                       "adopts": [[s2, a2] for s2, a2 in
                                  sorted(adopt_of.get(key, ()))[:40]],
                       "adopts_n": len(adopt_of.get(key, ())),
                       "uses": int(sc.vocab_use.get(key, 0))})
            if coin is not None:
                who.add(int(coin[1]))
            items.append(it)

    # ---- 6. 名前(根と媒介者だけ) ---------------------------------------- #
    who = {a for a in who if a >= 0}
    names = read_roster(run_dir, who) if who else {}

    _log(f"  カスケード: 候補 {len(cands):,} 本 → 掲載 {len(items)} 本 "
         f"({time.time() - t0:.1f}s)")
    return {
        "cap": int(cap),
        "shown": len(items),
        "nodes_cap": int(nodes_cap),
        "population": pop,
        "kind_labels": {str(k): v for k, v in CAS_KIND_LABELS.items()},
        "quota": {str(k): {"target": v, "filled": int(quota_filled[k]),
                           "label": CAS_KIND_LABELS[k]} for k, v in CAS_QUOTA.items()},
        "items": items,
        "names": {str(k): v for k, v in names.items()},
        "hex_m": float(hex_m),
        "belief_causes": list(sc.bel_causes.vals),
        "belief_srcs": list(sc.bel_srcs.vals),
        "channels": list(sc.prop_chans.vals),
        "notes": [
            "母集団=全カスケード(fact / 元 post / 語)。掲載は種別枠を先に埋め、"
            "残りを規模順で埋める。",
            "木のノードは上限つき。間引くときは内部ノード(子を持つ節)を全部残して"
            "から葉を等間隔で落とすので、タンポポ型/連鎖型の別は保存される。",
            "等時線 = 俯瞰と同じヘックス格子の「そのセルに初めて届いた step」。"
            "位置は採用イベントの座標(位置を持たない採用は等時線に出ない)。",
            "belief_update の cause=witness は**人づてではなく各自の目撃**。"
            "親(from)を持つ辺だけが伝播である。",
            "fact の場所名は belief_transmit.topic があれば厳密、無ければ"
            "「fact が立った step の目撃者の中央位置に最も近い stock_out」からの推定。",
            "リシェアの木は post id が追記通し番号であることを使って親を同定した"
            "(自分の RT post の id は L1 に出ない)。元投稿が公式(メディア)のとき本文は無い。",
        ],
    }


# --------------------------------------------------------------------------- #
# 画面4「物語ピン + 今日のハイライト」— story sifting
#
# 方針(計画 §1 画面4)
# --------------------
# * **パターンは宣言的**。1 つのパターン = 「どの kind を どの鍵で 繋ぐか」の宣言 +
#   「当事者 / step 列 / 現場」を返す関数。器は 7 種で共通(`_story` が組む)。
# * **surprise は説明できる形で出す**: `-log2(そのパターン種の周辺頻度 / 全イベント)`
#   + 規模 z + 稀少構成ボーナス。3 項をそのまま payload に残し、ボーナスの根拠は
#   人が読める日本語(`why`)で併記する(スコアだけ出して黙らない)。
# * **母集団 → 掲載**を必ず併記する。掲載は種別枠(STORY_QUOTA)を先に埋めてから
#   surprise 降順で埋める(迷惑行為 32,718 行が稀少パターンを押し流さないため)。
# * 思考チェーンは **1 ホップだけ**。`l1b_llm`(agent × step → llm_call_id)で
#   当事者の呼び出しを引き、`llm_journal.jsonl.gz` を**逐次**舐めて該当行だけ拾う
#   (全展開しない・走行中の切れた gz でも読めたところまでで止める)。
# --------------------------------------------------------------------------- #
def read_llm_index(run_dir) -> dict:
    """`l1b_llm`(agent_id, step, llm_call_id, purpose)→ {(agent, step): [id, purpose]}。

    L1 本体の `llm_call_id` 列を読むと 40.6 億行ぶんの文字列を触ることになるので、
    **同じ対応表を持つ 2 万行のサイドカー**の方を読む(1 パス目に足さない理由)。
    """
    import l1_stream
    import pyarrow.parquet as pq
    out: dict[tuple, list] = {}
    rows = 0
    for p in l1_stream.l1_paths(run_dir, "l1b_llm"):
        try:
            with l1_stream._open_shared(p) as fh:
                t = pq.ParquetFile(fh).read(
                    columns=["agent_id", "step", "llm_call_id", "purpose"])
        except (OSError, ValueError, KeyError):
            continue
        d = t.to_pydict()
        rows += t.num_rows
        for aid, st, cid, pu in zip(d["agent_id"], d["step"],
                                    d["llm_call_id"], d["purpose"]):
            if aid is None or st is None or not cid:
                continue
            key = (int(aid), int(st))
            if key not in out:                     # 同 step 複数呼びは最初の 1 本
                out[key] = [str(cid), str(pu or "")]
    return {"map": out, "rows": rows}


def read_journal(run_dir, want: set, *, scan_cap: int = JOURNAL_SCAN_CAP,
                 prompt_tail: int = 260, resp_head: int = 200) -> dict:
    """`llm_journal.jsonl.gz` を逐次で舐め、**欲しい llm_call_id の行だけ**取り出す。

    - `key` は 64 桁、L1 / l1b_llm の `llm_call_id` はその**先頭 16 桁**。
    - 行の中身を JSON にするのは**一致した行だけ**(1 行 ~700B の prompt を
      2 万行ぶん展開しない)。
    - 走行中のランでは末尾が書きかけのことがある。例外は握って
      「読めたところまで」を返す(欠測を偽の値で埋めない)。
    """
    import gzip
    import zlib
    out: dict[str, dict] = {}
    meta = {"available": False, "scanned": 0, "matched": 0, "truncated": False,
            "wanted": len(want)}
    if not want:
        return {"items": out, "meta": meta}
    path = Path(run_dir) / "llm_journal.jsonl.gz"
    if not path.is_file():
        return {"items": out, "meta": meta}
    meta["available"] = True
    need = set(want)
    tag = b'"key": "'
    try:
        with gzip.open(path, "rb") as fh:
            for line in fh:
                meta["scanned"] += 1
                if meta["scanned"] > scan_cap:
                    meta["truncated"] = True
                    break
                i = line.find(tag)
                if i < 0:
                    continue
                j = i + len(tag)
                cid = line[j:j + 16].decode("ascii", "ignore")
                if cid not in need:
                    continue
                try:
                    d = json.loads(line.decode("utf-8", "replace"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                pr = str(d.get("prompt") or "")
                rs = str(d.get("response") or "")
                out[cid] = {
                    "rng_key": d.get("rng_key"),
                    "backend": d.get("backend"),
                    "think": bool(d.get("think")),
                    "cached": bool(d.get("cached")),
                    "prompt_len": len(pr),
                    # プロンプトの**末尾**= 「いま何が見えているか」。冒頭の共通指示ではない。
                    "prompt": pr[-prompt_tail:],
                    "prompt_clipped": len(pr) > prompt_tail,
                    "response": rs[:resp_head],
                    "response_len": len(rs),
                    "response_clipped": len(rs) > resp_head,
                }
                need.discard(cid)
                if not need:
                    break
    except (OSError, EOFError, zlib.error) as exc:         # 走行中の切れた gz
        meta["truncated"] = True
        meta["error"] = type(exc).__name__
    meta["matched"] = len(out)
    return {"items": out, "meta": meta}


def read_gathering_intent(run_dir) -> dict:
    """`gathering_intent`(観察サイドカー)を読む。無ければ静かに空。

    1 セル = (day, when_bin, place_kind, place)。同じセルが 1 日数回撮られるので
    **n_intent が最大の行**を採る(意図のピークが「臨界まであと何人」の分子)。
    """
    import l1_stream
    import pyarrow.parquet as pq
    meta = {"available": False, "parts": 0, "rows": 0, "cells": 0}
    best: dict[tuple, dict] = {}
    paths = l1_stream.l1_paths(run_dir, "gathering_intent")
    meta["parts"] = len(paths)
    if not paths:
        return {"cells": [], "meta": meta}
    for p in paths:
        try:
            with l1_stream._open_shared(p) as fh:
                t = pq.ParquetFile(fh).read()
        except (OSError, ValueError, KeyError):
            continue
        d = t.to_pydict()
        meta["rows"] += t.num_rows
        for i in range(t.num_rows):
            key = (int(d["day"][i] or 0), int(d["when_bin"][i] or 0),
                   str(d["place_kind"][i] or ""), str(d["place"][i] or ""))
            n = int(d["n_intent"][i] or 0)
            cur = best.get(key)
            if cur is not None and cur["n_intent"] >= n:
                continue
            try:
                sample = json.loads(d["sample_ids"][i] or "[]")
            except (json.JSONDecodeError, TypeError):
                sample = []
            try:
                eids = json.loads(d["event_ids"][i] or "[]")
            except (json.JSONDecodeError, TypeError):
                eids = []
            best[key] = {
                "day": key[0], "when_bin": key[1], "place_kind": key[2],
                "place": key[3], "n_intent": n,
                "n_appointment": int(d["n_appointment"][i] or 0),
                "n_plan": int(d["n_plan"][i] or 0),
                "n_event": int(d["n_event"][i] or 0),
                "cap_day": int(d["cap_day"][i] or 0),
                "cap_min": int(d["cap_min"][i] or 0),
                "lead_min": int(d["lead_min"][i] or 0),
                "event_ids": eids,
                "sample": [int(v) for v in sample[:12]],
            }
    meta["available"] = True
    meta["cells"] = len(best)
    return {"cells": [best[k] for k in sorted(best)], "meta": meta}


def _b64_arr(s, dtype):
    """base64(`_b64` の逆)→ numpy 配列。空文字は長さ 0。"""
    import numpy as np
    if not s:
        return np.zeros(0, dtype=dtype)
    return np.frombuffer(base64.b64decode(s), dtype=dtype)


def _zscores(vals):
    """規模 z(母標準偏差)。1 件しか無いパターンは 0(= 規模で加点しない)。"""
    n = len(vals)
    if n <= 1:
        return [0.0] * n
    m = sum(vals) / n
    var = sum((v - m) ** 2 for v in vals) / n
    sd = math.sqrt(var)
    if sd <= 1e-9:
        return [0.0] * n
    return [max(-2.0, min(6.0, (v - m) / sd)) for v in vals]


def build_stories(run_dir, sc: _Scan, *, n_steps: int, cap: int = STORY_CAP_DEFAULT,
                  pairs_data=None, cas_data=None, thoughts: bool = True,
                  thought_cap: int = THOUGHT_CAP_DEFAULT,
                  node_xy=None, steps_per_day: int = 144) -> dict:
    """宣言的 sifting パターン → surprise 格付け → 掲載(母集団を必ず併記)。"""
    t0 = time.time()
    words = sc.st_words.vals
    nodes = sc.st_nodes.vals

    def W(i):
        return str(words[i]) if isinstance(i, int) and 0 <= i < len(words) else ""

    #: ノード id → 表示名。地図の `nodes[].name` / POI の `node` 欄から引く。
    #: 引けないものは **id をそのまま出す**(偽の名前を作らない)。
    node_name = {k: v[2] for k, v in (node_xy or {}).items() if v and v[2]}
    named_hits = [0, 0]

    def N(i):
        """辞書 id → ノードの人が読む名前(引けなければ id の文字列)。"""
        raw = str(nodes[i]) if isinstance(i, int) and 0 <= i < len(nodes) else ""
        if not raw:
            return ""
        named_hits[1] += 1
        nm = node_name.get(raw)
        if nm:
            named_hits[0] += 1
            return f"{nm}({raw})"
        return raw

    ITEM_JP = {"umbrella": "傘", "wallet": "財布", "phone": "携帯", "other": "持ち物"}

    def item_jp(i):
        w = W(i)
        return ITEM_JP.get(w, w or "持ち物")

    pop: dict = {"l1_rows": int(sc.rows),
                 "story_rows": int(sc.st_kept),
                 "story_rows_population": int(sum(sc.st_pop.values())),
                 "story_rows_dropped": int(sc.st_dropped),
                 "by_kind": {k: int(v) for k, v in sc.st_pop.most_common()},
                 "nuisance_rows": int(sc.nui_pop),
                 "nuisance_cells": len(sc.nui),
                 "nuisance_cells_dropped": int(sc.nui_dropped)}
    matches: list[dict] = []           # 母集団(パターンが見つけた全マッチ)
    who_all: set = set()

    def _story(pat, key, *, title, sub, who, s0, s1, x, y, beats,
               size, why=None, link=None, extra=None):
        # 拍は必ず **step 昇順**(安定ソート = 同 step は組んだ順)。時計の列が
        # 行ったり来たりすると、連鎖の論理順と時間順の区別が読めなくなる。
        beats = sorted(beats, key=lambda b: b[0])
        m = {"pat": int(pat), "key": str(key), "title": title, "sub": sub,
             "who": [int(v) for v in who if v is not None and int(v) >= 0],
             "s0": int(s0), "s1": int(s1), "x": int(x), "y": int(y),
             "beats": beats[:STORY_BEATS_MAX], "beats_n": len(beats),
             "size": float(size), "why": list(why or []),
             "bonus": 0.0, "link": link}
        if extra:
            m.update(extra)
        matches.append(m)
        who_all.update(m["who"])
        return m

    def _pos(rows):
        """拍の列から代表座標を決める(欠測 -32768 は使わない)。"""
        for r in rows:
            if r is not None and r.get("x", -32768) != -32768:
                return int(r["x"]), int(r["y"])
        return -32768, -32768

    # ===================================================================== #
    # ① 遺失物の完全連鎖(落とす → 拾う → 交番 → 返還 / 横領)
    #    鍵 = (落とし主, 品目, **落とした step**)。落とした step は各行の
    #    lying_steps / delay_steps / age_steps / held_steps から厳密に復元できる。
    # ===================================================================== #
    chains: dict[tuple, dict] = {}

    def _ch(owner, item, drop_step):
        k = (int(owner), int(item), int(drop_step))
        c = chains.get(k)
        if c is None:
            c = chains[k] = {"owner": int(owner), "item": int(item),
                             "drop_step": int(drop_step)}
        return c

    for r in sc.st.get("lost_drop", ()):
        _ch(r["owner"], r["item"], r["s"])["drop"] = r
    for r in sc.st.get("lost_notice", ()):
        _ch(r["owner"], r["item"], r["drop"])["notice"] = r
    for r in sc.st.get("lost_pickup", ()):
        _ch(r["owner"], r["item"], r["drop"])["pickup"] = r
    # turnin / keep は拾得と**同じ step**に出る(engine の `_phase_pickup`)。
    by_pick: dict[tuple, dict] = {}
    for k, c in chains.items():
        pk = c.get("pickup")
        if pk is not None:
            by_pick[(int(pk["owner"]), int(pk["item"]), int(pk["finder"]),
                     int(pk["s"]))] = c
    for kd in ("lost_turnin", "lost_keep"):
        for r in sc.st.get(kd, ()):
            c = by_pick.get((int(r["owner"]), int(r["item"]),
                             int(r["finder"]), int(r["s"])))
            if c is None:                          # 拾得行が cap/期間の外(正直に単独扱い)
                c = _ch(r["owner"], r["item"], -1 - int(r["s"]))
            c[kd[5:]] = r
    by_turn: dict[tuple, dict] = {}
    for c in chains.values():
        tr = c.get("turnin")
        if tr is not None:
            by_turn[(int(tr["owner"]), int(tr["item"]), int(tr["s"]))] = c
    for r in sc.st.get("lost_return", ()):
        c = by_turn.get((int(r["owner"]), int(r["item"]), int(r["turnin"])))
        if c is None:
            c = _ch(r["owner"], r["item"], -1 - int(r["s"]))
        c["return"] = r
    for r in sc.st.get("lost_expire", ()):
        _ch(r["owner"], r["item"], r["drop"])["expire"] = r

    pop["lost_chains"] = len(chains)
    OUTCOME = {"return": "が持ち主に返った", "keep": "を拾った人が自分のものにした",
               "expire": "が時効・失効になった", "turnin": "が交番に届いたまま",
               "pickup": "が拾われたまま", "drop": "が落ちたまま",
               "notice": "の遺失届だけが残った"}
    for key, c in sorted(chains.items()):
        seq = [k for k in ("drop", "notice", "pickup", "turnin", "keep",
                           "return", "expire") if c.get(k)]
        if not seq:
            continue
        rows = [c[k] for k in seq]
        rows.sort(key=lambda r: r["s"])
        outcome = ("return" if c.get("return") else
                   "keep" if c.get("keep") else
                   "expire" if c.get("expire") else
                   "turnin" if c.get("turnin") else
                   "pickup" if c.get("pickup") else
                   "notice" if c.get("notice") else "drop")
        it = item_jp(c["item"])
        owner, finder = c["owner"], -1
        for k in ("pickup", "turnin", "keep", "return", "expire"):
            if c.get(k) and int(c[k].get("finder", -1)) >= 0:
                finder = int(c[k]["finder"])
                break
        beats = []
        for k in ("drop", "notice", "pickup", "turnin", "keep", "return", "expire"):
            r = c.get(k)
            if not r:
                continue
            if k == "drop":
                extra = []
                if r.get("cash"):
                    extra.append(f"中に {r['cash']:,.0f} 円")
                if r.get("drinking"):
                    extra.append("飲んでいた")
                if r.get("rain"):
                    extra.append("雨")
                if int(r.get("crowd") or 0) >= 8:
                    extra.append(f"周りに {r['crowd']} 人")
                txt = f"{it}を落とした" + (f"({'・'.join(extra)})" if extra else "")
                w = owner
            elif k == "notice":
                txt = ("失くしたことに気づいて遺失届を出した("
                       + (N(r.get("post", -1)) or "最寄りの交番") + ")")
                w = owner
            elif k == "pickup":
                g = int(r.get("guardians", -1))
                txt = (f"{it}を拾った"
                       + (f"(周りに {g} 人)" if g > 0
                          else "(周りに誰も居なかった)" if g == 0 else ""))
                w = int(r["finder"])
            elif k == "turnin":
                txt = ("拾った" + it + "を交番へ届けた("
                       + (N(r.get("post", -1)) or "最寄りの交番")
                       + (f"・落ちていたのは {N(r['node'])}" if N(r.get("node", -1))
                          else "") + ")")
                w = int(r["finder"])
            elif k == "keep":
                txt = (f"★拾った{it}をそのまま自分のものにした"
                       f"({W(r['offense']) or '占有離脱物横領'}"
                       + (f"・現金 {r['amount']:,.0f} 円" if r.get("amount") else "") + ")")
                w = int(r["finder"])
            elif k == "return":
                txt = (f"{it}が持ち主へ返された"
                       + (f"(報労金 {r['amount']:,.0f} 円)" if r.get("amount") else ""))
                w = owner
            else:
                txt = ("時効で拾得者のものになった" if W(r["to"]) == "finder"
                       else "誰にも拾われないまま失われた")
                w = int(r.get("finder", -1))
            beats.append([int(r["s"]), int(r["x"]), int(r["y"]), txt, int(w)])
        x, y = _pos(rows)
        why = []
        bonus = 0.0
        if outcome == "keep":
            why.append("届けずに自分のものにした(占有離脱物横領)")
            bonus += 3.0
            if c["keep"].get("guardians") == 0:
                why.append("拾った瞬間、周りに誰も居なかった(監視者ゼロ)")
                bonus += 1.0
            if c["keep"].get("amount"):
                bonus += 0.6
        elif outcome == "return":
            if len(seq) >= 4:
                why.append("落とす→拾う→交番→返還の 4 拍が全部そろった完全連鎖")
                bonus += 1.6
            if c["return"].get("amount"):
                bonus += 0.4
        elif outcome == "notice":
            why.append("遺失届は出たが、誰も拾わなかった")
            bonus += 0.8
        elif outcome == "turnin":
            why.append("交番には届いたが、持ち主はまだ取りに来ていない")
            bonus += 0.5
        if c.get("drop") and c["drop"].get("cash"):
            why.append(f"財布の中に現金 {c['drop']['cash']:,.0f} 円が入っていた")
        _story(PAT_LOST, f"lost:{key[0]}:{key[1]}:{key[2]}",
               title=f"{it}{OUTCOME[outcome]}",
               sub=f"落とし主 #{owner}" + (f" / 拾った人 #{finder}" if finder >= 0 else ""),
               who=[owner, finder], s0=rows[0]["s"], s1=rows[-1]["s"], x=x, y=y,
               beats=beats, size=len(seq), why=why,
               extra={"outcome": outcome, "chain": seq})
        matches[-1]["bonus"] = bonus

    # ===================================================================== #
    # ② 緊急連鎖(倒れる/事故/負傷 → 通報 → 出動 → 搬送 → 入院 → 退院)
    #    鍵 = 患者 id × 発生 step からの窓(STORY_EMS_WINDOW step)。
    # ===================================================================== #
    # ★同じ (患者, step) に複数の発生行が出る: `traffic_accident` は同じ step に
    #   `injury` も出す(`incidents_env._ems_chain`)。別々の連鎖として数えると
    #   片方が通報行を全部持って行き、もう片方が「誰も通報しなかった」と**嘘を言う**。
    #   同一 (患者, step) は 1 件に畳む。
    merged: dict[tuple, list] = {}
    for kd in ("collapse", "injury", "traffic_accident"):
        for r in sc.st.get(kd, ()):
            merged.setdefault((int(r["patient"]), int(r["s"])), []).append((kd, r))
    onsets: list[tuple] = []
    for (pid, st), lst in sorted(merged.items()):
        # 具体的な出来事を先に(交通事故 > 負傷 > 倒れた)。位置・ノードもそれから採る。
        lst.sort(key=lambda t: {"traffic_accident": 0, "collapse": 1,
                                "injury": 2}.get(t[0], 3))
        onsets.append((pid, st, lst[0][0], lst[0][1], [k for k, _ in lst]))
    onsets.sort(key=lambda t: (t[0], t[1]))
    by_pat: dict[int, list] = defaultdict(list)
    for i, (pid, st, kd, r, _all) in enumerate(onsets):
        by_pat[pid].append(i)
    follow: dict[int, list] = defaultdict(list)
    for kd in ("ems_call", "ems_dispatch", "ems_transport",
               "hospital_admit", "hospital_discharge"):
        for r in sc.st.get(kd, ()):
            pid = int(r.get("patient", -1))
            if pid >= 0:
                follow[pid].append((int(r["s"]), kd, r))
    for v in follow.values():
        v.sort(key=lambda t: (t[0], t[1]))
    ONSET_JP = {"collapse": "倒れた", "injury": "負傷した",
                "traffic_accident": "交通事故に遭った"}
    used_follow: set = set()
    pop["ems_onsets"] = len(onsets)
    for pid, ixs in sorted(by_pat.items()):
        for j, i in enumerate(ixs):
            _, st, kd, r0, all_kinds = onsets[i]
            nxt = onsets[ixs[j + 1]][1] if j + 1 < len(ixs) else 10 ** 9
            hi = min(nxt, st + STORY_EMS_WINDOW * 6)      # 入院・退院は日を跨ぐ
            got = [(s, k, rr) for s, k, rr in follow.get(pid, ())
                   if st <= s < hi and (pid, s, k) not in used_follow]
            for s, k, _rr in got:
                used_follow.add((pid, s, k))
            onset_txt = "・".join(ONSET_JP.get(k, k) for k in all_kinds)
            beats = [[int(r0["s"]), int(r0["x"]), int(r0["y"]),
                      onset_txt
                      + (f"({W(r0['src'])})" if W(r0["src"]) else "")
                      + (f"・歩行者 {r0['ped_n']} 人の横断中"
                         + ("・信号あり" if r0.get("signalized") else "・信号なし")
                         if r0.get("ped_n") else ""), pid]]
            who = [pid]
            why, bonus = [], 0.0
            kinds_got = {k for _s, k, _r in got}
            for s, k, rr in got:
                if k == "ems_call":
                    self_call = bool(rr.get("self_call"))
                    txt = ("自分で救急に通報した" if self_call
                           else f"居合わせた人が救急に通報した({rr.get('dist_m')} m 先から)")
                    w = int(rr["a"])
                    if not self_call:
                        who.append(w)
                    if self_call:
                        why.append("誰も気づかず、本人が自分で通報した")
                        bonus += 0.8
                elif k == "ems_dispatch":
                    if rr.get("unstaffed"):
                        txt = "★救急が出られなかった(当直不在)"
                        why.append("通報はあったのに出動できなかった(当直不在)")
                        bonus += 3.0
                        w = -1
                    else:
                        txt = (f"救急隊 #{rr.get('crew')} が出動した"
                               + (f"(現着 {rr.get('response_min')} 分)"
                                  if rr.get("response_min") is not None else ""))
                        w = int(rr.get("crew", -1))
                        if w >= 0:
                            who.append(w)
                elif k == "ems_transport":
                    txt = (f"病院へ搬送された({N(rr.get('poi', -1)) or '搬送先不明'}"
                           + (f"・確定重症度 {rr.get('confirmed')}"
                              if rr.get("confirmed", -1) >= 0 else "") + ")")
                    w = pid
                elif k == "hospital_admit":
                    txt = (f"入院した({N(rr.get('poi', -1)) or '病院'}"
                           + (f"・{rr.get('days')} 日の予定" if rr.get("days") else "") + ")")
                    w = pid
                    why.append("搬送のあと入院までつながった")
                    bonus += 1.5
                else:
                    txt = f"退院した({rr.get('days')} 日ぶん請求)"
                    w = pid
                beats.append([int(s), int(rr["x"]), int(rr["y"]), txt, int(w)])
            if not got:
                why.append("倒れた/負傷したのに、通報が 1 件も出なかった(誰も見ていない)")
                bonus += 2.5
            if "ems_call" in kinds_got and "ems_dispatch" not in kinds_got:
                why.append("通報だけで終わり、出動の行が無い")
                bonus += 1.2
            x, y = _pos([r0] + [rr for _s, _k, rr in got])
            _story(PAT_EMS, f"ems:{pid}:{st}",
                   title=f"#{pid} が{ONSET_JP.get(kd, kd)}",
                   sub=(N(r0.get("node", -1)) or "場所不明")
                       + f" / L1 {len(all_kinds) + len(got)} 行の連鎖",
                   who=who, s0=st, s1=(got[-1][0] if got else st), x=x, y=y,
                   beats=beats, size=len(all_kinds) + len(got), why=why,
                   extra={"onset": kd, "onsets": all_kinds,
                          "chain": list(all_kinds) + [k for _s, k, _r in got]})
            matches[-1]["bonus"] = bonus
    # 発生行の無い通報(発生が期間外・別機構)も落とさない
    orphan = 0
    for pid, rows in sorted(follow.items()):
        rest = [(s, k, rr) for s, k, rr in rows if (pid, s, k) not in used_follow]
        if not rest:
            continue
        orphan += 1
        beats = [[int(s), int(rr["x"]), int(rr["y"]),
                  f"{k}(発生行がこの期間に無い)", pid] for s, k, rr in rest]
        x, y = _pos([rr for _s, _k, rr in rest])
        _story(PAT_EMS, f"ems-orphan:{pid}:{rest[0][0]}",
               title=f"#{pid} をめぐる救急の行(発生が期間の外)",
               sub=f"L1 {len(rest)} 行", who=[pid], s0=rest[0][0], s1=rest[-1][0],
               x=x, y=y, beats=beats, size=len(rest),
               why=["倒れた瞬間はこの期間の外(または別の機構)にある"],
               extra={"onset": None, "chain": [k for _s, k, _r in rest]})
        matches[-1]["bonus"] = 0.5
    pop["ems_orphans"] = orphan

    # ===================================================================== #
    # ③「聞く → 使う → 第三者へ」(hop >= 2)。画面3 のカスケード木を再利用する
    #    (`_tree_pack` の a / p / s / x / y / d は kind 非依存の同じ形)。
    # ===================================================================== #
    import numpy as np
    deep_pop = 0
    if cas_data and cas_data.get("items"):
        for ci, it in enumerate(cas_data["items"]):
            tr = it.get("tree") or {}
            if not tr.get("n"):
                continue
            a = _b64_arr(tr.get("a"), np.int32)
            par = _b64_arr(tr.get("p"), np.int32)
            s = _b64_arr(tr.get("s"), np.int32)
            xs = _b64_arr(tr.get("x"), np.int16)
            ys = _b64_arr(tr.get("y"), np.int16)
            dp = _b64_arr(tr.get("d"), np.uint8)
            if dp.size == 0:
                continue
            # `tree.d` は uint8 に丸めてある(255 で頭打ち)ので、**真の深さ**は
            # カスケード側の `depth` を使う。木からは「見せる鎖」だけを取り出す。
            depth = int(it.get("depth") or dp.max())
            if depth < 2:
                continue
            deep_pop += 1
            leaf = int(np.argmax(dp))
            path = []
            j = leaf
            for _ in range(300):
                path.append(j)
                p = int(par[j]) if j < par.size else -1
                if p < 0 or p >= par.size:
                    break
                j = p
            path.reverse()
            beats = []
            for rank, j in enumerate(path):
                aid = int(a[j])
                lbl = ("最初に言い出した" if rank == 0 else
                       f"{rank} 人目に伝わった(hop {int(dp[j])})")
                beats.append([int(s[j]), int(xs[j]), int(ys[j]),
                              f"{lbl}: #{aid}" if aid >= 0 else f"{lbl}: 公式(メディア)",
                              aid])
            kind_lb = CAS_KIND_LABELS.get(it["kind"], "?")
            why = [f"人づての鎖が {depth} ホップ続いた"]
            bonus = 2.0 if depth >= 3 else 0.0
            if it["kind"] == CAS_VOCAB:
                why.append("語彙の伝播(Run A ではほぼ観測されない)")
                bonus += 2.5
            if it["kind"] == CAS_BELIEF:
                why.append("信念の人づて(Run A では 34.5 万件中 17 件だけ)")
                bonus += 2.5
            _story(PAT_HOP, f"hop:{it['kind']}:{it['key']}",
                   title=f"{kind_lb}「{it['title']}」が {depth} ホップ渡った",
                   sub=f"採用 {it['n']} 人 / 人づての辺 {it['edges']} 本 / 形 {it['shape']}"
                       f" / たどった鎖 {len(path)} 人",
                   who=[int(a[j]) for j in path], s0=int(s[path[0]]),
                   s1=int(s[path[-1]]), x=int(xs[path[-1]]), y=int(ys[path[-1]]),
                   beats=beats, size=depth, why=why,
                   link={"cas": ci}, extra={"cas_kind": int(it["kind"]),
                                            "depth": depth, "n": int(it["n"])})
            matches[-1]["bonus"] = bonus
    pop["deep_cascades"] = deep_pop

    # ===================================================================== #
    # ④ 関係ドラマ(画面2 の「破綻 → 再構築」ペアをそのまま再利用する)
    # ===================================================================== #
    drama_pop = 0
    if pairs_data and pairs_data.get("items"):
        TIERS = ["他人", "知人", "友人", "親友", "親友+", "親友++"]

        def tname(t):
            t = int(t)
            return TIERS[t] if 0 <= t < len(TIERS) else str(t)

        for pi, it in enumerate(pairs_data["items"]):
            brk = it.get("brk") or []
            if not brk:
                continue
            last_break = max(int(b[0]) for b in brk)
            after = [t for t in (it.get("tiers") or []) if int(t[0]) > last_break]
            rebuilt = max((int(t[1]) for t in after), default=-1)
            worst = min(int(b[2]) for b in brk)
            if rebuilt < 0 and (it.get("partner") is None
                                or int(it["partner"]) <= last_break):
                continue                                # 破綻して終わった組は画面2 の仕事
            drama_pop += 1
            beats = []
            ev = [(int(t[0]), "up", int(t[1])) for t in (it.get("tiers") or [])]
            ev += [(int(b[0]), "down", int(b[2]), int(b[1]),
                    (pairs_data.get("causes") or [])[b[3]]
                    if 0 <= b[3] < len(pairs_data.get("causes") or []) else "")
                   for b in brk]
            ev.sort(key=lambda e: e[0])
            for e in ev:
                # 段は**2 人のあいだ**の量なので、拍の主語は片方の名前にしない(-1)。
                if e[1] == "up":
                    beats.append([e[0], -32768, -32768,
                                  f"▲ {tname(e[2])} になった", -1])
                else:
                    beats.append([e[0], -32768, -32768,
                                  f"▼ {tname(e[3])} → {tname(e[2])} に下がった"
                                  + (f"({e[4]})" if e[4] else ""), -1])
            # 位置は会話・実文の現場から借りる(関係イベント自体は座標を持たない)
            x, y = -32768, -32768
            for row in (it.get("text") or []):
                if row[3] != -32768:
                    x, y = int(row[3]), int(row[4])
                    break
            if x == -32768:
                for row in (it.get("conv") or []):
                    if row[5] != -32768:
                        x, y = int(row[5]), int(row[6])
                        break
            why = [f"一度 {tname(worst)} まで落ちてから、もう一度 {tname(rebuilt)} に戻った"]
            bonus = 3.0
            if len(brk) >= 2:
                why.append(f"こじれたのは 1 度ではなく {len(brk)} 度")
                bonus += 1.0
            if it.get("partner") is not None and int(it["partner"]) > last_break:
                why.append("そのあとパートナーになった")
                bonus += 2.0
            if (it.get("text_n") or 0) > 0:
                why.append(f"2 人のあいだに実文が {it['text_n']} 行残っている")
                bonus += 0.6
            _story(PAT_PAIR, f"pair:{it['a']}:{it['b']}",
                   title="こじれて、戻った 2 人",
                   sub=f"#{it['a']} × #{it['b']} / 破綻 {len(brk)} 回 / "
                       f"会話 {it.get('conv_n', 0)} 回",
                   who=[it["a"], it["b"]], s0=int(ev[0][0]), s1=int(ev[-1][0]),
                   x=x, y=y, beats=beats, size=len(brk) + max(0, rebuilt),
                   why=why, link={"pair": pi},
                   extra={"breaks": len(brk), "rebuilt": rebuilt, "worst": worst})
            matches[-1]["bonus"] = bonus
    pop["pair_drama"] = drama_pop

    # ===================================================================== #
    # ⑤ 集会(event_host → event_attend n 人)と、**不発の集会**
    #    (`gathering_intent` = 現実では観測できない「集まろうとした量」)
    # ===================================================================== #
    hosts = {int(r["eid"]): r for r in sc.st.get("event_host", ()) if r["eid"] >= 0}
    att: dict[int, list] = defaultdict(list)
    for r in sc.st.get("event_attend", ()):
        if int(r["eid"]) >= 0:
            att[int(r["eid"])].append(r)
    pop["events_hosted"] = len(hosts)
    pop["events_attended"] = sum(len(v) for v in att.values())
    for eid, h in sorted(hosts.items()):
        rows = sorted(att.get(eid, ()), key=lambda r: r["s"])
        beats = [[int(h["s"]), int(h["x"]), int(h["y"]),
                  f"「{W(h['title'])}」を {N(h['place']) or '街のどこか'} で開くと決めた",
                  int(h["a"])]]
        for r in rows[:STORY_BEATS_MAX - 1]:
            beats.append([int(r["s"]), int(r["x"]), int(r["y"]),
                          f"#{r['a']} が参加した", int(r["a"])])
        x, y = _pos([h] + rows)
        why, bonus = [], 0.0
        if not rows:
            why.append("告知はしたのに、誰も来なかった")
            bonus += 2.5
        elif len(rows) >= 5:
            why.append(f"{len(rows)} 人が集まった")
            bonus += 1.0
        _story(PAT_GATHER, f"event:{eid}",
               title=f"「{W(h['title'])}」に {len(rows)} 人",
               sub=f"主催 #{h['a']} / {N(h['place']) or '場所不明'}",
               who=[h["a"]] + [r["a"] for r in rows[:12]],
               s0=int(h["s"]), s1=int(rows[-1]["s"]) if rows else int(h["s"]),
               x=x, y=y, beats=beats, size=len(rows), why=why,
               extra={"event_id": eid, "attendees": len(rows)})
        matches[-1]["bonus"] = bonus

    # ---- 不発の集会(observer サイドカー `gathering_intent`)------------------ #
    # 1 セル = (day, when_bin, place_kind, place)。Run A の実測で判ったこと:
    #   * `n_appointment > 0` のセルは **54 件・全部 place_kind="label"**
    #     (「渋谷」「カフェ」「ハチ公」= 場所を**語で**約束した)で、**1 件も
    #     集会イベントにならなかった**。これが「現実では観測できない量」の本体。
    #   * `place_kind="node"` のセルは 19,072 件あるが中身はほぼ `n_plan`
    #     (= 通勤・通学の予定が同じ駅に集まっただけ)。**集会ではないので
    #     物語にはせず**、日ごとの上位だけを「予定の集中」として別に出す。
    gi = read_gathering_intent(run_dir)
    cells = gi["cells"]
    spd = max(1, int(steps_per_day))
    # when_bin は「その日を slot_min で割ったビン番号」。slot_min は行に載らないので
    # 観測された最大ビンから逆算する(Run A では 47 → 48 ビン/日 = 30 分)。
    nb = max((c["when_bin"] for c in cells), default=-1) + 1
    nb = nb if nb >= 2 else 48
    slot_min = int(round(1440.0 / nb))
    gi["meta"]["bins_per_day"] = nb
    gi["meta"]["slot_min_inferred"] = slot_min
    gi["meta"]["appointment_cells"] = sum(1 for c in cells if c["n_appointment"] > 0)
    gi["meta"]["node_cells"] = sum(1 for c in cells if c["place_kind"] == "node")
    gi["meta"]["realized_cells"] = sum(1 for c in cells if c["n_event"] > 0)
    pop["gathering_intent"] = gi["meta"]

    dt_min = max(1, 1440 // spd)

    def _gi_step(c):
        """**意図が観測された step**(cap_day / cap_min = サイドカーを撮った時刻)。

        セルの (day, when_bin) は「いつ集まるつもりか」= **未来**なので、走査した
        範囲の外にあることがある。そこへ丸めると「起きてもいない時刻」に印を打つ
        ことになるので、拍の時刻には**観測した瞬間**を使い、目標時刻は文で言う。
        """
        st = int(c["cap_day"]) * spd + int(round(int(c["cap_min"]) / dt_min))
        return max(0, min(max(0, n_steps - 1), st))

    def _gi_when(c):
        """「いつ集まるつもりだったか」。when_bin は**その日の 0 時から**の枠番号。"""
        if c["when_bin"] < 0:
            return "時刻を決めずに"
        m = int(c["when_bin"]) * slot_min
        return f"Day {c['day'] + 1} の {m // 60:02d}:{m % 60:02d} ごろ"

    def _gi_xy(c):
        if node_xy and c["place_kind"] == "node":
            v = node_xy.get(c["place"])
            if v:
                return _xy16(v[0]), _xy16(v[1])
        return -32768, -32768

    def _gi_place(c):
        if c["place_kind"] == "node":
            nm = node_name.get(c["place"])
            return f"{nm}({c['place']})" if nm else (c["place"] or "場所不明")
        return c["place"] or ""

    unfulfilled = 0
    for c in cells:
        if c["n_event"] or c["n_appointment"] <= 0:
            continue                                # 実現した / 約束ではない(予定だけ)
        unfulfilled += 1
        short = max(0, GATHER_MIN_N - c["n_intent"])
        st = _gi_step(c)
        x, y = _gi_xy(c)
        raw_place = _gi_place(c)
        quoted = f"「{raw_place}」" if raw_place else "場所を決めないまま"
        when = _gi_when(c)
        beats = [[st, x, y,
                  f"{quoted}で {when} 会おうと約束した人が "
                  f"{c['n_appointment']} 人いた(この拍の時刻は"
                  f"意図を観測した瞬間であって、集まる予定の時刻ではない)", -1],
                 [st, x, y,
                  ("そのセルに集会イベントは 1 件も立たなかった"
                   + (f"(集合とみなす臨界 {GATHER_MIN_N} 人まであと {short} 人)"
                      if short else "(頭数は臨界を超えていたのに集まりにならなかった)")),
                  -1]]
        why = ["現実では観測できない量:「集まろうとしたが集まらなかった」",
               "場所は語で約束されている(地図のノードに解決されていない)"]
        bonus = 2.5 + (1.5 if 0 < short <= 2 else 0.0)
        if short == 0:
            why.append("臨界人数には届いていた。足りなかったのは同時性か場所の一致")
            bonus += 1.0
        _story(PAT_GATHER,
               f"gi:{c['day']}:{c['when_bin']}:{c['place_kind']}:{c['place']}",
               title=(f"「{raw_place}」に集まらなかった {c['n_appointment']} 人"
                      if raw_place
                      else f"場所を決めずに約束した {c['n_appointment']} 人が"
                           "集まらなかった"),
               sub=f"{when} / 約束 {c['n_appointment']} 件"
                   + (f" / 臨界まであと {short} 人" if short else " / 臨界は超えていた"),
               who=c["sample"][:12], s0=st, s1=st, x=x, y=y, beats=beats,
               size=c["n_appointment"], why=why,
               extra={"unfulfilled": True, "short_by": short,
                      "n_intent": c["n_intent"], "place_kind": c["place_kind"]})
        matches[-1]["bonus"] = bonus
    pop["gatherings_unfulfilled"] = unfulfilled

    # 予定の集中(集会ではない)。日ごとの上位 6 セルだけを別の題で出す。
    by_day: dict[int, list] = defaultdict(list)
    for c in cells:
        if c["place_kind"] == "node" and not c["n_event"] and c["n_intent"] >= GATHER_MIN_N:
            by_day[c["day"]].append(c)
    crowd = 0
    for day, lst in sorted(by_day.items()):
        lst.sort(key=lambda c: -c["n_intent"])
        for c in lst[:6]:
            crowd += 1
            st = _gi_step(c)
            x, y = _gi_xy(c)
            when, place = _gi_when(c), _gi_place(c)
            beats = [[st, x, y,
                      f"{when}({slot_min} 分枠)に「{place}」へ行く予定を立てていた人が "
                      f"{c['n_intent']:,} 人いた(予定 {c['n_plan']:,})"
                      "。この拍の時刻は意図を観測した瞬間", -1],
                     [st, x, y, "集会イベントにはならなかった(= 群れであって集まりではない)",
                      -1]]
            _story(PAT_GATHER, f"gicrowd:{c['day']}:{c['when_bin']}:{c['place']}",
                   title=f"{c['n_intent']:,} 人ぶんの予定が 1 点に集まった",
                   sub=f"{when} / {place}",
                   who=c["sample"][:12], s0=st, s1=st, x=x, y=y, beats=beats,
                   size=c["n_intent"],
                   why=["約束ではなく予定の集中(通勤・通学の流れ)。"
                        "集会と混同しないよう別の題で出す"],
                   extra={"unfulfilled": True, "crowd": True,
                          "n_intent": c["n_intent"], "place_kind": "node"})
            matches[-1]["bonus"] = 0.0
    pop["gatherings_crowd"] = crowd

    # ===================================================================== #
    # ⑥ 犯罪連鎖(crime → 通報 → police_response)と、迷惑行為の集中
    #    Run A に `crime` / `police_response` は 1 行も無い = 静かに 0 件になる。
    # ===================================================================== #
    police = sorted((int(r["s"]), r) for r in sc.st.get("police_response", ()))
    pop["crime_rows"] = len(sc.st.get("crime", ()))
    pop["police_rows"] = len(police)
    for r in sc.st.get("crime", ()):
        beats = [[int(r["s"]), int(r["x"]), int(r["y"]),
                  f"#{r['offender']} が #{r['victim']} から "
                  f"{r.get('amount') or 0:,.0f} 円を盗んだ({W(r['ckind']) or '窃盗'})",
                  int(r["offender"])]]
        # `crime` の payload はノードを持たない(engine の cr_payload 参照)ので、
        # 突合は **step の窓だけ**で行い、場所一致は主張しない。
        resp = [p for s, p in police if int(r["s"]) <= s <= int(r["s"]) + 12]
        for p in resp[:6]:
            beats.append([int(p["s"]), int(p["x"]), int(p["y"]),
                          f"警察が動いた({W(p['about'])})", int(p["a"])])
        why = ["犯罪の実行が L1 に残っている"]
        bonus = 1.5 + (1.5 if resp else 0.0)
        if not resp:
            why.append("警察の行は 1 件も出ていない(このランに police_response が無い)")
        _story(PAT_CRIME, f"crime:{r['s']}:{r['offender']}",
               title=f"{W(r['ckind']) or '窃盗'}が起きた",
               sub=f"加害 #{r['offender']} / 被害 #{r['victim']}",
               who=[r["offender"], r["victim"]], s0=int(r["s"]),
               s1=int(beats[-1][0]), x=int(r["x"]), y=int(r["y"]),
               beats=beats, size=1 + len(resp), why=why,
               extra={"amount": r.get("amount")})
        matches[-1]["bonus"] = bonus

    bursts = 0
    for (nid, b), cell in sorted(sc.nui.items()):
        if cell[0] < NUI_BURST_MIN:
            continue
        bursts += 1
        kinds_top = ", ".join(f"{k}×{v}" for k, v in cell[6].most_common(4))
        beats = [[int(cell[1]), int(cell[3]), int(cell[4]),
                  f"{N(nid) or '同じ場所'}で迷惑行為が {cell[0]} 件続いた({kinds_top})",
                  int(cell[5][0]) if cell[5] else -1]]
        if cell[2] != cell[1]:
            beats.append([int(cell[2]), int(cell[3]), int(cell[4]),
                          "その 1 時間の最後の 1 件", -1])
        _story(PAT_CRIME, f"nui:{nid}:{b}",
               title=f"騒がしかった 1 時間({cell[0]} 件)",
               sub=(N(nid) or "場所不明") + f" / {kinds_top}",
               who=cell[5], s0=int(cell[1]), s1=int(cell[2]),
               x=int(cell[3]), y=int(cell[4]), beats=beats, size=cell[0],
               why=[], extra={"nuisance": cell[0]})
        matches[-1]["bonus"] = 0.0
    pop["nuisance_bursts"] = bursts

    # ===================================================================== #
    # ⑦ viral 瞬間(採用が最も加速した step = カスケードの離陸点)
    # ===================================================================== #
    viral_pop = 0
    if cas_data and cas_data.get("items"):
        for ci, it in enumerate(cas_data["items"]):
            curve = it.get("curve") or []
            if int(it.get("n") or 0) < 8 or len(curve) < 3:
                continue
            viral_pop += 1
            # 離陸 = 新規採用の増分が最大の点(2 階差分の最大)。
            best, bstep, bd = 0.0, int(curve[0][0]), 0
            for k in range(1, len(curve)):
                d = curve[k][1] - curve[k - 1][1]
                if d > best:
                    best, bstep, bd = float(d), int(curve[k][0]), int(curve[k][1])
            peak = max(curve, key=lambda c: c[1])
            beats = [[int(curve[0][0]), -32768, -32768,
                      f"最初の採用({curve[0][1]} 人)", -1],
                     [bstep, -32768, -32768,
                      f"★ここで加速した(+{best:.0f} 人 / step ・この step で {bd} 人)", -1],
                     [int(peak[0]), -32768, -32768,
                      f"ピーク({peak[1]} 人 / step)", -1],
                     [int(it["s1"]), -32768, -32768,
                      f"最後の採用(合計 {it['n']} 人)", -1]]
            # 位置は木の代表点(等時線の芯)
            xs = _b64_arr((it.get("tree") or {}).get("x"), np.int16)
            ys = _b64_arr((it.get("tree") or {}).get("y"), np.int16)
            x, y = -32768, -32768
            ok = np.nonzero(xs != -32768)[0] if xs.size else np.zeros(0, dtype=int)
            if ok.size:
                x, y = int(np.median(xs[ok])), int(np.median(ys[ok]))
            why = [f"1 step で {best:.0f} 人ぶん増えた瞬間がある"]
            bonus = 0.0
            if it.get("reach"):
                why.append(f"インフルエンサー加重の到達 {it['reach']:,}")
                bonus += 1.0
            if int(it.get("depth") or 0) >= 5:
                why.append(f"深さ {it['depth']} の連鎖")
                bonus += 1.5
            _story(PAT_VIRAL, f"viral:{it['kind']}:{it['key']}",
                   title=f"「{it['title']}」が離陸した",
                   sub=f"{CAS_KIND_LABELS.get(it['kind'], '?')} / 採用 {it['n']} 人 / "
                       f"形 {it['shape']}",
                   who=[], s0=int(it["s0"]), s1=int(it["s1"]), x=x, y=y,
                   beats=beats, size=float(best), why=why, link={"cas": ci},
                   extra={"takeoff_step": bstep, "takeoff_delta": _round(best, 1),
                          "n": int(it["n"])})
            matches[-1]["bonus"] = bonus
    pop["viral_candidates"] = viral_pop

    # ===================================================================== #
    # surprise = -log2(そのパターン種の周辺頻度 / 全イベント) + 規模 z + 稀少構成
    # ===================================================================== #
    total_events = max(1, int(sc.rows))
    by_pat: dict[int, list] = defaultdict(list)
    for m in matches:
        by_pat[m["pat"]].append(m)
    rarity_of: dict[int, float] = {}
    for pat, ms in by_pat.items():
        p = max(1, len(ms)) / float(total_events)
        rarity = -math.log2(p)
        rarity_of[pat] = rarity
        zs = _zscores([m["size"] for m in ms])
        for m, z in zip(ms, zs):
            m["parts"] = {"rarity": _round(rarity, 3), "size_z": _round(z, 3),
                          "bonus": _round(m["bonus"], 3)}
            m["surprise"] = _round(rarity + z + m["bonus"], 3)
            m.pop("bonus", None)
            m.pop("bonus_raw", None)

    matches.sort(key=lambda m: (-(m["surprise"] or 0.0), m["pat"], m["key"]))

    # ---- 掲載の選抜(種別枠 → 残りは surprise 降順) ----------------------- #
    chosen: list[dict] = []
    seen: set = set()
    quota_filled = Counter()
    shown_by_pat = Counter()
    for pat, q in STORY_QUOTA.items():
        got = 0
        for m in matches:
            if got >= q or len(chosen) >= cap:
                break
            if m["pat"] != pat or m["key"] in seen:
                continue
            chosen.append(m)
            seen.add(m["key"])
            got += 1
        quota_filled[pat] = got
        shown_by_pat[pat] = got
    # 残枠は surprise 降順。ただし 1 種別が枠の 2 倍を超えて占めない(迷惑行為の
    # 「騒がしかった 1 時間」が 191 件あっても、同じ題で一覧を埋め尽くさないため)。
    for m in matches:
        if len(chosen) >= cap:
            break
        if m["key"] in seen:
            continue
        if shown_by_pat[m["pat"]] >= 2 * STORY_QUOTA.get(m["pat"], cap):
            continue
        chosen.append(m)
        seen.add(m["key"])
        shown_by_pat[m["pat"]] += 1
    chosen.sort(key=lambda m: (-(m["surprise"] or 0.0), m["pat"], m["key"]))

    # ---- 思考 → 行為 → 結果チェーン(1 ホップぶん) ------------------------ #
    th_meta = {"available": False, "attached": 0, "candidates": 0,
               "index_rows": 0, "scanned": 0, "truncated": False}
    if thoughts:
        idx = read_llm_index(run_dir)
        th_meta["index_rows"] = idx["rows"]
        lut = idx["map"]
        want: dict[str, list] = {}
        for si, m in enumerate(chosen):
            if len(want) >= thought_cap:
                break
            hit = None
            for w in m["who"]:
                for s in range(int(m["s0"]), min(int(m["s1"]), int(m["s0"]) + 6) + 1):
                    v = lut.get((int(w), s))
                    if v is not None:
                        hit = (int(w), s, v[0], v[1])
                        break
                if hit:
                    break
            if hit:
                want.setdefault(hit[2], []).append((si, hit))
        th_meta["candidates"] = len(want)
        jr = read_journal(run_dir, set(want))
        th_meta.update({k: jr["meta"][k] for k in ("available", "scanned",
                                                   "truncated") if k in jr["meta"]})
        for cid, uses in want.items():
            rec = jr["items"].get(cid)
            if not rec:
                continue
            for si, hit in uses:
                chosen[si]["thought"] = {
                    "agent": hit[0], "step": hit[1], "call": cid,
                    "purpose": hit[3], **rec}
                th_meta["attached"] += 1

    # ---- 名前(掲載ぶんの当事者だけ) -------------------------------------- #
    who_shown = {int(w) for m in chosen for w in m["who"] if int(w) >= 0}
    names = read_roster(run_dir, who_shown) if who_shown else {}

    pop["node_names"] = {"resolved": named_hits[0], "asked": named_hits[1],
                         "dictionary": len(node_name)}
    _log(f"  物語: 母集団 {len(matches):,} 件 → 掲載 {len(chosen)} 件 / "
         f"思考 {th_meta['attached']} 件 / ノード名 {named_hits[0]}/{named_hits[1]} 解決 "
         f"({time.time() - t0:.1f}s)")
    return {
        "cap": int(cap),
        "shown": len(chosen),
        "population": {**pop, "matches": len(matches),
                       "by_pat": {str(k): len(v) for k, v in sorted(by_pat.items())}},
        "pat_labels": {str(k): v for k, v in PAT_LABELS.items()},
        "pat_glyphs": {str(k): v for k, v in PAT_GLYPHS.items()},
        "pat_colors": {str(k): v for k, v in PAT_COLORS.items()},
        "quota": {str(k): {"target": v, "filled": int(quota_filled[k]),
                           "shown": int(shown_by_pat[k]), "cap": 2 * v,
                           "label": PAT_LABELS[k]} for k, v in STORY_QUOTA.items()},
        "items": chosen,
        "names": {str(k): v for k, v in names.items()},
        "thoughts": th_meta,
        "surprise": {
            "total_events": total_events,
            "rarity_by_pat": {str(k): _round(v, 3) for k, v in sorted(rarity_of.items())},
            "formula": "surprise = -log2(そのパターンのマッチ数 / L1 全行数)"
                       " + 規模 z(パターン内・[-2,6] で切る) + 稀少構成ボーナス",
        },
        "notes": [
            "母集団=各パターンが見つけた全マッチ。掲載は種別枠を先に埋め、"
            "残りを surprise 降順で埋める(迷惑行為 3.3 万行が稀少な連鎖を押し流さないため)。"
            "残枠でも 1 種別は枠の 2 倍を超えて占めない(同じ題で一覧を埋めない)。",
            "遺失物の連鎖は近接一致ではなく **exact join**: 落とした step を "
            "lying_steps / delay_steps / age_steps / held_steps から厳密に復元して繋ぐ。",
            "緊急の連鎖は患者 id × 発生 step からの窓。発生行がこの期間の外にある"
            "通報も落とさず「発生が期間の外」として別に出す。",
            "不発の集会は observer サイドカー gathering_intent(意図のピーク)のうち"
            "**約束**(n_appointment>0)が立ったセルだけ。臨界は detect_gatherings と"
            f"同じ既定 n>={GATHER_MIN_N} 人で測る。予定(n_plan)だけが同じ駅に集まった"
            "セルは集会ではないので、別の題「予定が 1 点に集まった」で日ごとの上位だけ出す。",
            "集まりの拍の時刻は「意図を**観測した**step」(サイドカーの cap_day/cap_min)。"
            "集まる予定の時刻は走査範囲の外にありうるので、そこへ丸めて印を打たない。",
            "ノード id は地図の nodes[].name と POI の node 欄から人が読む名前へ解決する。"
            "引けなかったものは id をそのまま出す(偽の名前を作らない)。",
            "思考チェーンは 1 ホップぶん: l1b_llm(agent × step → llm_call_id)で"
            "当事者の呼び出しを引き、llm_journal.jsonl.gz を逐次で舐めて該当行だけ拾う。"
            "プロンプトは**末尾 260 字**(= いま何が見えているか)、応答は先頭 200 字。",
            "surprise は説明できる 3 項の和。ボーナスの根拠は why に日本語で残す。",
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
              text_cap: int = TEXT_CAP_DEFAULT, cascades: bool = True,
              cas_cap: int = CAS_CAP_DEFAULT,
              cas_nodes: int = CAS_NODES_DEFAULT,
              belief_cap: int = BELIEF_CAP_DEFAULT,
              reshare_cap: int = RESHARE_CAP_DEFAULT,
              stock_cap: int = STOCK_CAP_DEFAULT, stories: bool = True,
              story_cap: int = STORY_CAP_DEFAULT, thoughts: bool = True,
              thought_cap: int = THOUGHT_CAP_DEFAULT, node_xy=None) -> dict:
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
            "hexbin": None, "relations": None, "pairs": None, "cascades": None,
            "stories": None,
            "build": {"scan_sec": 0.0},
        }

    t0 = time.time()
    sc = scan_l1(run_dir, hex_m=hex_m, steps_per_bin=steps_per_bin,
                 rel_cap=rel_cap, step_max=step_max, narrative=pairs,
                 conv_cap=conv_cap, text_cap=text_cap, propagation=cascades,
                 belief_cap=belief_cap, reshare_cap=reshare_cap,
                 stock_cap=stock_cap, stories=stories)
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

    cas_data = None
    if cascades:
        cas_data = build_cascades(run_dir, sc, hex_m=hex_m, cap=cas_cap,
                                  nodes_cap=cas_nodes, n_steps=n_steps)

    # 画面4 は画面2/3 の**成果物**(注目ペア・カスケード木)を再利用するので最後に走る。
    story_data = None
    if stories:
        story_data = build_stories(run_dir, sc, n_steps=n_steps, cap=story_cap,
                                   pairs_data=pairs_data, cas_data=cas_data,
                                   thoughts=thoughts, thought_cap=thought_cap,
                                   node_xy=node_xy, steps_per_day=spd)

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
        "cascades": cas_data,
        "stories": story_data,
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
/* ---------- 画面3 伝播 ---------- */
#clist{width:304px;flex:none;border-right:1px solid var(--bd);background:var(--panel);
 display:flex;flex-direction:column;min-height:0}
#clist .hd{padding:8px 9px 6px;border-bottom:1px solid var(--bd)}
#clist select,#clist input{background:#0d1117;color:var(--fg);border:1px solid var(--bd);
 border-radius:4px;font:inherit;font-size:11.5px;padding:3px 5px;width:100%;margin-bottom:4px}
#crows{overflow-y:auto;flex:1;min-height:0}
.crow{padding:6px 9px;border-bottom:1px solid rgba(255,255,255,.05);cursor:pointer}
.crow:hover{background:#161c26}
.crow.on{background:#1c2637;box-shadow:inset 3px 0 0 var(--acc)}
.crow .nm{font-size:12px;font-weight:600;white-space:nowrap;overflow:hidden;
 text-overflow:ellipsis}
.crow .mt{color:var(--dim);font-size:10.5px;font-variant-numeric:tabular-nums;
 display:flex;gap:7px;flex-wrap:wrap;margin-top:1px}
.kbd{display:inline-block;padding:0 5px;border-radius:3px;font-size:9.5px;line-height:15px;
 margin-right:4px}
.kbd.k0{background:#16304a;color:#7cc4ff}.kbd.k1{background:#3b1f33;color:#f9a8d4}
.kbd.k2{background:#14342a;color:#4ade80}
.shp{display:inline-block;padding:0 5px;border-radius:3px;font-size:9.5px;line-height:15px;
 background:#243040;color:#b6c2d2}
#cmain{flex:1;min-width:0;display:flex;flex-direction:column}
#isowrap{position:relative;flex:0 0 54%;min-height:210px;border-bottom:1px solid var(--bd)}
#icv{display:block;width:100%;height:100%;cursor:crosshair}
#ihead{position:absolute;left:12px;top:9px;font-size:12.5px;pointer-events:none;
 max-width:60%}
#ihead b{font-size:14px}
#ilegend{position:absolute;right:12px;top:9px;background:var(--panel);border:1px solid var(--bd);
 border-radius:7px;padding:7px 9px;font-size:10.5px;min-width:148px;pointer-events:none}
#ilegend .bar{height:9px;border-radius:2px;margin:5px 0 3px}
#ilegend .lb{display:flex;justify-content:space-between;color:var(--dim);
 font-variant-numeric:tabular-nums}
#ictrl{position:absolute;left:12px;right:12px;bottom:10px;background:var(--panel);
 border:1px solid var(--bd);border-radius:8px;padding:6px 11px;display:flex;
 align-items:center;gap:10px;backdrop-filter:blur(6px)}
#ictrl button{background:#1b212c;color:var(--fg);border:1px solid var(--bd);border-radius:5px;
 width:30px;height:26px;cursor:pointer;font:inherit;font-size:12px}
#ictrl input[type=range]{flex:1;accent-color:var(--acc)}
#iclock{font-variant-numeric:tabular-nums;font-size:11.5px;min-width:168px}
#itip,#ttip{position:absolute;pointer-events:none;background:#0d1117ee;border:1px solid var(--bd);
 border-radius:5px;padding:5px 8px;font-size:11.5px;display:none;white-space:nowrap;z-index:6}
#cbot{flex:1;min-height:0;display:flex}
#scurve{flex:0 0 44%;min-width:0;position:relative;border-right:1px solid var(--bd)}
#ctree{flex:1;min-width:0;position:relative}
#scurve canvas,#ctree canvas{display:block;width:100%;height:100%}
.cvtitle{position:absolute;left:10px;top:7px;font-size:10.5px;color:var(--dim);
 pointer-events:none}
#cside{width:306px;flex:none;border-left:1px solid var(--bd);background:var(--panel);
 overflow-y:auto;padding:10px 11px 26px}
#cside h2{font-size:11.5px;color:var(--dim);margin:12px 0 6px;font-weight:600;
 text-transform:uppercase;letter-spacing:.06em}
#cside h2:first-child{margin-top:0}
.chbar{display:flex;align-items:center;gap:6px;margin:3px 0}
.chbar .lb{width:96px;flex:none;font-size:11px;color:var(--dim);white-space:nowrap;
 overflow:hidden;text-overflow:ellipsis}
.chbar .tr{flex:1;height:9px;background:#161c22;border-radius:2px;overflow:hidden}
.chbar .fl{height:100%;background:var(--acc)}
.chbar .vv{width:58px;flex:none;text-align:right;font-size:10.5px;
 font-variant-numeric:tabular-nums}
.quote{background:#11151d;border:1px solid var(--bd);border-left:2px solid var(--acc);
 border-radius:5px;padding:6px 9px;font-size:12px;margin:5px 0}
/* ---------- 画面4 物語ピン + 今日のハイライト ---------- */
#slist{width:322px;flex:none;border-right:1px solid var(--bd);background:var(--panel);
 display:flex;flex-direction:column;min-height:0}
#slist .hd{padding:8px 9px 6px;border-bottom:1px solid var(--bd)}
#slist select,#slist input[type=search]{background:#0d1117;color:var(--fg);
 border:1px solid var(--bd);border-radius:4px;font:inherit;font-size:11.5px;
 padding:3px 5px;width:100%;margin-bottom:4px}
#srows{overflow-y:auto;flex:1;min-height:0}
.srow{padding:6px 9px;border-bottom:1px solid rgba(255,255,255,.05);cursor:pointer}
.srow:hover{background:#161c26}
.srow.on{background:#1c2637;box-shadow:inset 3px 0 0 var(--acc)}
.srow .nm{font-size:12px;font-weight:600}
.srow .mt{color:var(--dim);font-size:10.5px;font-variant-numeric:tabular-nums;
 display:flex;gap:7px;flex-wrap:wrap;margin-top:1px}
.pglyph{display:inline-block;width:17px;height:17px;line-height:17px;text-align:center;
 border-radius:4px;font-size:10.5px;font-weight:700;color:#0a0e14;margin-right:5px;
 vertical-align:-2px}
.sscore{font-variant-numeric:tabular-nums;font-weight:700;color:var(--warn)}
#smain{flex:1;min-width:0;overflow-y:auto;padding:14px 18px 30px}
#smain h3{font-size:16px;margin:0 0 3px;font-weight:650}
#smain .subt{color:var(--dim);font-size:12px;margin-bottom:10px}
.beat{display:flex;gap:10px;align-items:flex-start;margin:0 0 5px}
.beat .st{flex:none;width:82px;color:var(--dim);font-size:10.5px;
 font-variant-numeric:tabular-nums;padding-top:3px}
.beat .bd{flex:1;min-width:0;border:1px solid var(--bd);border-radius:7px;
 background:#11151d;padding:5px 9px;font-size:12.5px}
.beat .bd.jump{cursor:pointer}
.beat .bd.jump:hover{border-color:var(--acc)}
.beat .who{font-size:10.5px;color:var(--dim);margin-bottom:1px}
.why{margin:9px 0 0;padding:0 0 0 16px;color:#cbd5e1;font-size:11.8px;line-height:1.6}
.lnkrow{display:flex;gap:7px;flex-wrap:wrap;margin:11px 0 3px}
.lnk{background:#1b2331;color:var(--fg);border:1px solid var(--bd);border-radius:5px;
 padding:4px 11px;cursor:pointer;font:inherit;font-size:11.5px}
.lnk:hover{border-color:var(--acc)}
#sside{width:314px;flex:none;border-left:1px solid var(--bd);background:var(--panel);
 overflow-y:auto;padding:10px 11px 26px}
#sside h2{font-size:11.5px;color:var(--dim);margin:12px 0 6px;font-weight:600;
 text-transform:uppercase;letter-spacing:.06em}
#sside h2:first-child{margin-top:0}
.think{background:#0d1117;border:1px solid var(--bd);border-radius:6px;padding:7px 9px;
 font-size:11.5px;line-height:1.55;white-space:pre-wrap;word-break:break-word;
 max-height:230px;overflow-y:auto}
.think.resp{border-left:2px solid var(--t2)}
.think.prompt{border-left:2px solid var(--acc);color:#b6c2d2}
#pincard{position:absolute;left:12px;top:118px;width:290px;background:var(--panel);
 border:1px solid var(--bd);border-radius:8px;padding:9px 11px;font-size:11.5px;
 display:none;z-index:6;backdrop-filter:blur(6px)}
#pincard .cl{position:absolute;right:7px;top:5px;cursor:pointer;color:var(--dim)}
</style></head><body>
<div id="top">
  <h1>Shibuya Chronicle</h1>
  <div class="tabs" id="tabs">
    <button data-p="map" class="on">俯瞰</button>
    <button data-p="pair">関係の伝記</button>
    <button data-p="prop">伝播</button>
    <button data-p="story">物語</button>
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
      <label><input type="checkbox" id="lyPin" checked> 物語ピン<span id="pinN"></span></label>
      <label style="padding-left:16px">上位
        <input type="range" id="pinTop" min="5" max="200" value="60" step="5"
          style="width:92px;vertical-align:-3px;accent-color:var(--acc)">
        <span id="pinTopV" style="font-variant-numeric:tabular-nums"></span> 件</label>
    </div>
    <div id="legend"></div>
    <div id="pincard"></div>
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
<div id="wrap3" class="pane">
  <div id="clist">
    <div class="hd">
      <select id="ckind">
        <option value="">種別: すべて</option>
        <option value="0">種別: 信念(fact)</option>
        <option value="1">種別: リシェア(SNS)</option>
        <option value="2">種別: 語彙</option>
      </select>
      <select id="csort">
        <option value="n">並べ替え: 規模(採用者数)</option>
        <option value="edges">並べ替え: 人づての辺が多い</option>
        <option value="depth">並べ替え: 深さ</option>
        <option value="speed">並べ替え: 離陸が速い</option>
        <option value="s0">並べ替え: 発生が早い</option>
      </select>
      <input id="cfind" type="search" placeholder="題・場所・語で絞る">
      <div class="note" id="ccount"></div>
    </div>
    <div id="crows"></div>
  </div>
  <div id="cmain">
    <div id="isowrap">
      <canvas id="icv"></canvas>
      <div id="ihead"></div>
      <div id="ilegend"></div>
      <div id="itip"></div>
      <div id="ictrl">
        <button id="iplay" title="波紋を再生">▶</button>
        <div id="iclock"></div>
        <input type="range" id="iscrub" min="0" max="0" value="0" step="1">
        <button id="ifit" title="全体に合わせる">⤢</button>
      </div>
    </div>
    <div id="cbot">
      <div id="scurve"><canvas id="ccv"></canvas>
        <div class="cvtitle">採用 S 字曲線(累積 / 新規)</div></div>
      <div id="ctree"><canvas id="tcv"></canvas><div id="ttip"></div>
        <div class="cvtitle">カスケード木(中心=火元・外周=遠いホップ)</div></div>
    </div>
  </div>
  <div id="cside">
    <h2>このカスケードについて</h2>
    <div id="cfacts"></div>
    <h2>チャネル分解</h2>
    <div id="cch"></div>
    <h2>母集団 → 掲載</h2>
    <div id="cpop"></div>
  </div>
</div>
<div id="wrap4" class="pane">
  <div id="slist">
    <div class="hd">
      <select id="spat"><option value="">種別: すべて</option></select>
      <select id="ssort">
        <option value="surprise">並べ替え: 驚き(surprise)</option>
        <option value="time">並べ替え: 起きた順</option>
        <option value="size">並べ替え: 規模</option>
        <option value="beats">並べ替え: 拍が多い</option>
      </select>
      <input id="sfind" type="search" placeholder="題・当事者・場所で絞る">
      <label style="display:block;font-size:11px;color:var(--dim);margin:2px 0 3px">
        <input type="checkbox" id="sthonly" style="vertical-align:-1px">
        本人の思考が付いているものだけ</label>
      <div class="note" id="scount"></div>
    </div>
    <div id="srows"></div>
  </div>
  <div id="smain"><div id="scard"></div></div>
  <div id="sside">
    <h2>驚き(surprise)の内訳</h2>
    <div id="sscore"></div>
    <h2>本人の思考 → 行為(1 ホップ)</h2>
    <div id="sthink"></div>
    <h2>母集団 → 掲載</h2>
    <div id="spop"></div>
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
  drawPins();
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
     + '2 人の物語を見る →</a>'
     + ' · <a href="#" style="color:var(--acc)" onclick="setTab(\'prop\');return false">'
     + '話の広がりを見る →</a></div>';
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
      fillTop(); fillAbout(); fillRel(); buildCharts(); pairInit(); propInit();
      storyInit(); draw(); };
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
['lyBase','lyRoad','lyHex','lyCarry','lyPoi','lyLog','lyPin'].forEach(id=>
  document.getElementById(id).addEventListener('change', requestDraw));
document.getElementById('pinTop').addEventListener('input', requestDraw);

let drag = null, dragMoved = 0;
cv.addEventListener('mousedown', e=>{ drag = {x:e.clientX, y:e.clientY,
  cx:cam.x, cy:cam.y}; dragMoved = 0; cv.classList.add('drag'); });
window.addEventListener('mouseup', ()=>{ drag = null; cv.classList.remove('drag'); });
/* 物語ピンのクリック(掴んで動かしたときは拾わない = 3px 未満だけ「押した」とみなす) */
cv.addEventListener('click', e=>{
  if(dragMoved > 3) return;
  const rect = cv.getBoundingClientRect();
  const mx = e.clientX-rect.left, my = e.clientY-rect.top;
  let best = -1, bd = 1e9;
  for(const g of PINGEO){ const d = (g.sx-mx)*(g.sx-mx) + (g.sy-my)*(g.sy-my);
    if(d < g.r*g.r*1.6 && d < bd){ bd = d; best = g.i; } }
  if(best < 0) return;
  SSEL = best; showPinCard(best); requestDraw();
});
window.addEventListener('mousemove', e=>{
  if(drag){ dragMoved = Math.max(dragMoved,
              Math.abs(e.clientX-drag.x) + Math.abs(e.clientY-drag.y));
            cam.x = drag.cx - (e.clientX-drag.x)/cam.s;
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
  document.getElementById('wrap3').classList.toggle('on', t === 'prop');
  document.getElementById('wrap4').classList.toggle('on', t === 'story');
  if(t === 'map') resize();
  else if(t === 'prop') propResize();
  else if(t === 'pair') pairResize();
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
  h += storyBacklink('pair', PSEL);
  host.innerHTML = h;
  wireBacklinks(host);
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

/* =========================================================================
   画面3「伝播」— 信念 / リシェア / 語彙を同じ器で
   -------------------------------------------------------------------------
   1 本のカスケードにつき 4 点セットを描く。
     ① 等時線マップ … 俯瞰と同じヘックス格子で「そのセルに初めて届いた step」を色に。
                      再生すると波紋になる(scrub は step ではなく **このカスケードの時間**)。
     ② 採用 S 字曲線 … 累積(面)と新規(棒)。離陸 step と飽和が読める。
     ③ カスケード木 … 中心=火元・外周=遠いホップの放射レイアウト。
                      辺ゼロ(= 全員が独立に知った)のときは時間軸の帯に切り替える。
     ④ チャネル分解 … 対面 / SNS / DM / 目撃 の辺数。
   色は俯瞰の hexbin と同じ rampColor を使う(画面をまたいで色の意味を揃える)。
   ========================================================================= */
const CAS_KCLS = {0:'k0', 1:'k1', 2:'k2'};
const SHAPE_LABEL = {parallel:'並列発生', seeded:'ほぼ並列', fan:'タンポポ',
                     chain:'連鎖', tree:'枝分かれ', mixed:'混在'};
const SHAPE_HELP = {
  parallel:'親(from)を持つ採用が 1 件も無い = 人づてではなく、全員が独立に同じ現場を見た。',
  seeded:'8 割超が独立発生で、人づての辺はごく一部。ほぼ並列に起きたなかに'
    + '数本だけ伝播が混じっている。',
  fan:'深さ 1。1 人から一斉に広がった(タンポポ型)。',
  chain:'深さのわりに枝が少ない = 数珠つなぎに伝わった。',
  tree:'深さも枝もある = 何段にも枝分かれした。',
  mixed:'枝分かれと連鎖が混在。'};
let CSEL = -1, CORD = [], CT = 0, CPLAY = false, CGEO = [], CTREE = {}, TGEO = [];
let ICAM = {x:0, y:0, s:1}, ilastT = 0, IFIT_W = 0;

function CAS(){ return R() ? R().cascades : null; }
function curCas(){ const c = CAS();
  return (c && CSEL>=0 && CSEL<c.items.length) ? c.items[CSEL] : null; }
function cname(id){ const c = CAS();
  if(id === -1) return '公式(メディア)';
  const v = c && c.names[String(id)];
  return (v && v[0]) ? v[0] : '#'+id; }
function casTree(i){
  const key = RK+'/'+i;
  if(CTREE[key] === undefined){
    const t = CAS().items[i].tree;
    CTREE[key] = !t || !t.n ? null : {
      n: t.n, population: t.population, orphans: t.orphans,
      a: bin(t.a, Int32Array), p: bin(t.p, Int32Array), s: bin(t.s, Int32Array),
      x: bin(t.x, Int16Array), y: bin(t.y, Int16Array), d: bin(t.d, Uint8Array)};
  }
  return CTREE[key];
}
function casSpan(it){ return Math.max(1, (it.s1|0) - (it.s0|0)); }
/* 離陸の速さ = 採用の半分が済むまでの step 数(小さいほど速い) */
function casHalf(it){
  const cur2 = it.curve||[]; let tot = 0;
  for(const c of cur2) tot += c[1];
  let acc = 0;
  for(const c of cur2){ acc += c[1]; if(acc*2 >= tot) return c[0] - it.s0; }
  return casSpan(it);
}

/* ---------- 一覧 ---------- */
function casRowHTML(it, i){
  const kd = it.kind|0;
  return '<div class="crow'+(i===CSEL?' on':'')+'" data-i="'+i+'">'
    + '<div class="nm"><span class="kbd '+CAS_KCLS[kd]+'">'
    + esc(CAS().kind_labels[String(kd)]) + '</span>' + esc(it.title||it.key) + '</div>'
    + '<div class="mt"><span>' + it.n.toLocaleString() + ' 人</span>'
    + '<span>深さ ' + it.depth + '</span>'
    + '<span>人づて ' + it.edges.toLocaleString() + ' 辺</span>'
    + '<span class="shp">' + esc(SHAPE_LABEL[it.shape]||it.shape) + '</span>'
    + '</div></div>';
}
function buildCasList(){
  const c = CAS(), host = document.getElementById('crows');
  const cnt = document.getElementById('ccount');
  if(!c || !c.items.length){
    host.innerHTML = '<div class="empty">このランには画面3 の素材がありません'
      + '(走行中のランは flush 済みの part だけを見ます)。</div>';
    cnt.textContent = ''; CORD = []; return;
  }
  const kf = document.getElementById('ckind').value;
  const mode = document.getElementById('csort').value;
  const q = (document.getElementById('cfind').value||'').trim().toLowerCase();
  CORD = c.items.map((it,i)=>i);
  if(kf !== '') CORD = CORD.filter(i => String(c.items[i].kind) === kf);
  if(q) CORD = CORD.filter(i => { const it = c.items[i];
    return ((it.title||'')+' '+(it.text||'')+' '+it.key+' '+(it.place||''))
      .toLowerCase().indexOf(q) >= 0; });
  const val = it => mode==='edges' ? -(it.edges||0)
    : mode==='depth' ? -(it.depth||0)
    : mode==='speed' ? casHalf(it)
    : mode==='s0' ? (it.s0||0) : -(it.n||0);
  CORD.sort((x,y)=> val(c.items[x]) - val(c.items[y]) || x-y);
  host.innerHTML = CORD.map(i => casRowHTML(c.items[i], i)).join('');
  const po = c.population;
  const totPop = (po.belief_facts||0) + (po.reshare_cascades||0) + (po.vocab_items||0);
  cnt.innerHTML = '母集団 ' + totPop.toLocaleString() + ' 本 → 掲載 ' + c.shown + ' 本'
    + (CORD.length !== c.items.length ? ' / 絞り込み '+CORD.length+' 本' : '');
  host.querySelectorAll('.crow').forEach(el =>
    el.onclick = ()=> selectCas(+el.dataset.i));
}
function selectCas(i){
  CSEL = i;
  document.querySelectorAll('#crows .crow').forEach(el =>
    el.classList.toggle('on', +el.dataset.i === i));
  const it = curCas();
  const sc3 = document.getElementById('iscrub');
  if(it){ sc3.min = it.s0; sc3.max = Math.max(it.s0, it.s1); CT = it.s1;
          sc3.value = CT; isoFit(); }
  drawIso(); drawCurve(); drawTree(); fillCasFacts(); fillCasCh(); syncHash();
}

/* ---------- 等時線マップ ---------- */
const icv = document.getElementById('icv'), ictx = icv.getContext('2d');
function propResize(){
  const dpr = Math.min(2, window.devicePixelRatio||1);
  for(const el of [icv, document.getElementById('ccv'),
                   document.getElementById('tcv')]){
    el.width = Math.round(el.clientWidth*dpr);
    el.height = Math.round(el.clientHeight*dpr);
    el.getContext('2d').setTransform(dpr,0,0,dpr,0,0);
  }
  if(IFIT_W <= 0) isoFit();      /* 非表示のまま合わせた枠を実寸で取り直す */
  drawIso(); drawCurve(); drawTree();
}
function hexXY(q, r, R2){
  const S3 = Math.sqrt(3);
  return [R2*(S3*q + S3/2*r), R2*(1.5*r)];
}
function isoR(){ const it = curCas();
  return ((it && it.iso_hex_m) || (CAS() ? CAS().hex_m : 200))/Math.sqrt(3); }
function isoFit(){
  const it = curCas(); if(!it || !it.iso.length){ ICAM = {x:0,y:0,s:0.6}; return; }
  const R2 = isoR();
  let x0=1e9, y0=1e9, x1=-1e9, y1=-1e9;
  for(const c of it.iso){ const p = hexXY(c[0], c[1], R2);
    x0=Math.min(x0,p[0]); x1=Math.max(x1,p[0]);
    y0=Math.min(y0,p[1]); y1=Math.max(y1,p[1]); }
  const pad = R2*3;
  x0-=pad; x1+=pad; y0-=pad; y1+=pad;
  const W = icv.clientWidth||600, H = icv.clientHeight||300;
  ICAM.s = Math.min(W/Math.max(1,x1-x0), H/Math.max(1,y1-y0));
  ICAM.x = (x0+x1)/2; ICAM.y = (y0+y1)/2;
  IFIT_W = icv.clientWidth;      /* 非表示のまま合わせた枠は表示時に取り直す */
}
function itf(wx, wy){ return [ (wx-ICAM.x)*ICAM.s + icv.clientWidth/2,
                               -(wy-ICAM.y)*ICAM.s + icv.clientHeight/2 ]; }
function iinv(sx, sy){ return [ (sx - icv.clientWidth/2)/ICAM.s + ICAM.x,
                                -(sy - icv.clientHeight/2)/ICAM.s + ICAM.y ]; }
function drawIso(){
  const W = icv.clientWidth, H = icv.clientHeight;
  ictx.fillStyle = '#080c12'; ictx.fillRect(0,0,W,H);
  const it = curCas();
  const head = document.getElementById('ihead'), leg = document.getElementById('ilegend');
  if(!it){ head.innerHTML = '<span style="color:var(--dim)">'
    + '左の一覧からカスケードを選ぶと、どの順で街に届いたかが出ます</span>';
    leg.innerHTML = ''; document.getElementById('iclock').textContent = ''; return; }
  /* 道路(薄く) */
  ictx.lineWidth = 1; ictx.strokeStyle = 'rgba(226,232,240,.10)';
  const ppm = ICAM.s;
  for(let i=0;i<ROADS.length;i++){
    const kls = BM.road_classes[ROADCLS[i]] || 'footway';
    if(ppm < 0.25 && !(kls==='primary'||kls==='secondary'||kls==='tertiary')) continue;
    const p = ROADS[i]; ictx.beginPath();
    let s = itf(p[0], p[1]); ictx.moveTo(s[0], s[1]);
    for(let j=2;j<p.length;j+=2){ s = itf(p[j], p[j+1]); ictx.lineTo(s[0], s[1]); }
    ictx.stroke();
  }
  ictx.strokeStyle = 'rgba(148,163,184,.20)'; ictx.lineWidth = 1.6;
  for(const p of RAILS){ ictx.beginPath();
    let s = itf(p[0], p[1]); ictx.moveTo(s[0], s[1]);
    for(let j=2;j<p.length;j+=2){ s = itf(p[j], p[j+1]); ictx.lineTo(s[0], s[1]); }
    ictx.stroke(); }
  /* ヘックス(初到達 step を色に。CT より後のセルは輪郭だけ = まだ届いていない) */
  const R2 = isoR(), rpx = R2*ICAM.s;
  const s0 = it.s0, s1 = Math.max(it.s1, it.s0+1);
  CGEO = [];
  for(const c of it.iso){
    const p = hexXY(c[0], c[1], R2), sc4 = itf(p[0], p[1]);
    if(sc4[0] < -rpx || sc4[0] > W+rpx || sc4[1] < -rpx || sc4[1] > H+rpx) continue;
    CGEO.push({x:sc4[0], y:sc4[1], c:c});
    const t = (c[2]-s0)/(s1-s0);
    ictx.beginPath();
    for(let k=0;k<6;k++){ const a = Math.PI/180*(60*k - 30);
      const px = sc4[0]+rpx*Math.cos(a), py = sc4[1]+rpx*Math.sin(a);
      if(k===0) ictx.moveTo(px,py); else ictx.lineTo(px,py); }
    ictx.closePath();
    if(c[2] <= CT){
      ictx.fillStyle = rampColor(1-t*0.92);
      ictx.globalAlpha = 0.86; ictx.fill(); ictx.globalAlpha = 1;
      if(c[2] === CT){ ictx.strokeStyle = '#fff'; ictx.lineWidth = 2; ictx.stroke(); }
    } else {
      ictx.strokeStyle = 'rgba(148,163,184,.20)'; ictx.lineWidth = 1; ictx.stroke();
    }
  }
  /* 火元(木の根の位置) */
  const T = casTree(CSEL);
  /* 火元の印は「火元が数人」のときだけ。根が何百もあるカスケードで 60 個だけ
     打つと「その 60 人が特別」に見えてしまうので打たない。 */
  if(T && it.edges > 0 && it.roots <= 12){ let drawn = 0;
    for(let i=0;i<T.n && drawn<60;i++){
      if(T.p[i] >= 0 || T.x[i] === -32768) continue;
      const s5 = itf(T.x[i], T.y[i]);
      ictx.beginPath(); ictx.arc(s5[0], s5[1], 4, 0, 6.284);
      ictx.fillStyle = '#fbbf24'; ictx.fill();
      ictx.strokeStyle = '#0b0f16'; ictx.lineWidth = 1.4; ictx.stroke();
      drawn++;
    } }
  const arrived = it.iso.filter(c => c[2] <= CT).length;
  head.innerHTML = '<b>'+esc(it.title||it.key)+'</b><br>'
    + '<span style="color:var(--dim);font-size:11px">'
    + esc(CAS().kind_labels[String(it.kind)]) + ' · ' + it.n.toLocaleString() + ' 人 · '
    + esc(SHAPE_LABEL[it.shape]||it.shape) + ' · ヘックス ' + arrived + '/'
    + it.iso.length + ' 到達</span>';
  let bar = '';
  for(let i=0;i<=10;i++) bar += rampColor(1-(i/10)*0.92)+(i<10?',':'');
  leg.innerHTML = '<div>初到達 step</div><div class="bar" style="background:'
    + 'linear-gradient(90deg,'+bar+')"></div>'
    + '<div class="lb"><span>'+clockText(s0).hhmm+'</span><span>'
    + clockText(it.s1).hhmm+'</span></div>'
    + '<div style="margin-top:5px;color:var(--dim)">'
    + (it.edges<=0 ? '全員が根(並列発生)'
       : (it.roots<=12 ? '◯ = 火元(根)' : '根 '+it.roots.toLocaleString()+' 人(印は略)'))
    + '</div>'
    + '<div style="color:var(--dim)">輪郭のみ = 未到達</div>'
    + '<div style="color:var(--dim)">1 ヘックス = '
    + (it.iso_hex_m||CAS().hex_m) + ' m(広がりに合わせて可変)</div>';
  const ck = clockText(CT);
  document.getElementById('iclock').innerHTML = '<b>Day '+(ck.day+1)+' '+ck.hhmm
    + '</b> <span style="color:var(--dim)">step '+CT+'</span>';
}
document.getElementById('iscrub').addEventListener('input', e=>{
  CT = +e.target.value; drawIso(); drawCurve(); });
document.getElementById('iplay').addEventListener('click', ()=>{
  CPLAY = !CPLAY;
  document.getElementById('iplay').textContent = CPLAY ? '❚❚' : '▶';
  if(CPLAY){ const it = curCas(); if(it && CT >= it.s1) CT = it.s0;
    requestAnimationFrame(itick); } });
function itick(t){
  if(!CPLAY) return;
  const it = curCas();
  if(!it){ CPLAY = false; return; }
  if(t - ilastT > 110){ ilastT = t;
    CT = CT >= it.s1 ? it.s0 : CT+1;
    document.getElementById('iscrub').value = CT;
    drawIso(); drawCurve(); }
  requestAnimationFrame(itick);
}
document.getElementById('ifit').onclick = ()=>{ isoFit(); drawIso(); };
icv.addEventListener('wheel', e=>{
  e.preventDefault();
  const rect = icv.getBoundingClientRect();
  const w = iinv(e.clientX-rect.left, e.clientY-rect.top);
  const k = Math.exp(-e.deltaY*0.0016);
  const ns = Math.max(0.02, Math.min(8, ICAM.s*k));
  ICAM.x = w[0] - (w[0]-ICAM.x)*(ICAM.s/ns);
  ICAM.y = w[1] - (w[1]-ICAM.y)*(ICAM.s/ns);
  ICAM.s = ns; drawIso();
}, {passive:false});
let idrag = null;
icv.addEventListener('mousedown', e=>{ idrag = {x:e.clientX, y:e.clientY,
  cx:ICAM.x, cy:ICAM.y}; });
window.addEventListener('mouseup', ()=>{ idrag = null; });
icv.addEventListener('mousemove', e=>{
  const rect = icv.getBoundingClientRect();
  if(idrag){ ICAM.x = idrag.cx - (e.clientX-idrag.x)/ICAM.s;
             ICAM.y = idrag.cy + (e.clientY-idrag.y)/ICAM.s; drawIso(); return; }
  const mx = e.clientX-rect.left, my = e.clientY-rect.top;
  const tip = document.getElementById('itip');
  let best = null, bd = Math.pow(Math.max(6, isoR()*ICAM.s), 2);
  for(const g of CGEO){ const d = (g.x-mx)*(g.x-mx)+(g.y-my)*(g.y-my);
    if(d < bd){ bd = d; best = g; } }
  if(!best){ tip.style.display = 'none'; return; }
  const ck = clockText(best.c[2]);
  tip.innerHTML = '<b>'+best.c[3].toLocaleString()+' 人</b>がここで知った<br>'
    + '<span style="color:var(--dim)">初到達 Day '+(ck.day+1)+' '+ck.hhmm
    + ' · step '+best.c[2]+'</span>';
  tip.style.display = 'block';
  tip.style.left = Math.min(icv.clientWidth-210, mx+13)+'px';
  tip.style.top = (my+12)+'px';
});
icv.addEventListener('mouseleave', ()=>{
  document.getElementById('itip').style.display = 'none'; });
icv.addEventListener('dblclick', e=>{
  const rect = icv.getBoundingClientRect();
  let best = null, bd = Math.pow(Math.max(6, isoR()*ICAM.s), 2);
  const mx = e.clientX-rect.left, my = e.clientY-rect.top;
  for(const g of CGEO){ const d = (g.x-mx)*(g.x-mx)+(g.y-my)*(g.y-my);
    if(d < bd){ bd = d; best = g; } }
  if(!best) return;
  const p = hexXY(best.c[0], best.c[1], isoR());
  jumpToMap(best.c[2], p[0], p[1]);
});

/* ---------- 採用 S 字曲線 ---------- */
function drawCurve(){
  const cv3 = document.getElementById('ccv'), c = cv3.getContext('2d');
  const W = cv3.clientWidth, H = cv3.clientHeight;
  c.fillStyle = '#0c111a'; c.fillRect(0,0,W,H);
  const it = curCas(); if(!it || !it.curve.length) return;
  const L = 42, Rr = 12, Tp = 24, B = 24;
  const s0 = it.s0, s1 = Math.max(it.s1, it.s0+1);
  let tot = 0, mxn = 1;
  for(const p of it.curve){ tot += p[1]; mxn = Math.max(mxn, p[1]); }
  const X = s => L + (s-s0)/(s1-s0)*(W-L-Rr);
  const Yc = v => H-B - v/Math.max(1,tot)*(H-Tp-B);
  /* 新規(棒) */
  const bw = Math.max(1.2, (W-L-Rr)/Math.max(1,(s1-s0+1)) - 1);
  c.fillStyle = 'rgba(96,165,250,.34)';
  for(const p of it.curve){
    const h = p[1]/mxn*(H-Tp-B)*0.55;
    c.fillRect(X(p[0])-bw/2, H-B-h, bw, h);
  }
  /* 累積(面 + 線) */
  let acc = 0; const pts = [];
  for(const p of it.curve){ acc += p[1]; pts.push([X(p[0]), Yc(acc)]); }
  c.beginPath(); c.moveTo(pts[0][0], H-B);
  for(const p of pts) c.lineTo(p[0], p[1]);
  c.lineTo(pts[pts.length-1][0], H-B); c.closePath();
  c.fillStyle = 'rgba(74,222,128,.14)'; c.fill();
  c.beginPath(); c.moveTo(pts[0][0], pts[0][1]);
  for(const p of pts) c.lineTo(p[0], p[1]);
  c.strokeStyle = '#4ade80'; c.lineWidth = 1.8; c.stroke();
  /* いま(等時線の時刻)の縦線 */
  c.strokeStyle = 'rgba(255,255,255,.42)'; c.lineWidth = 1;
  c.beginPath(); c.moveTo(X(CT), Tp-6); c.lineTo(X(CT), H-B); c.stroke();
  /* 目盛り */
  c.fillStyle = '#6b7686'; c.font = '9.5px sans-serif'; c.textAlign = 'left';
  c.fillText('累積 '+tot.toLocaleString()+' 人', L, Tp-10);
  c.textAlign = 'right';
  c.fillText(clockText(s1).hhmm, W-Rr, H-8);
  c.textAlign = 'left';
  c.fillText(clockText(s0).hhmm, L, H-8);
  c.textAlign = 'right';
  c.fillText(tot.toLocaleString(), L-4, Tp+8);
  c.fillText('0', L-4, H-B);
  c.strokeStyle = 'rgba(255,255,255,.10)';
  c.beginPath(); c.moveTo(L, Tp-2); c.lineTo(L, H-B); c.lineTo(W-Rr, H-B); c.stroke();
}

/* ---------- カスケード木 ---------- */
function drawTree(){
  const cv4 = document.getElementById('tcv'), c = cv4.getContext('2d');
  const W = cv4.clientWidth, H = cv4.clientHeight;
  c.fillStyle = '#0c111a'; c.fillRect(0,0,W,H);
  TGEO = [];
  const it = curCas(); if(!it) return;
  const T = casTree(CSEL); if(!T || !T.n) return;
  const n = T.n, s0 = it.s0, s1 = Math.max(it.s1, it.s0+1);
  const col = i => rampColor(1-((T.s[i]-s0)/(s1-s0))*0.92);
  const px = new Float64Array(n), py = new Float64Array(n);
  if(it.edges <= 0){
    /* 辺ゼロ = 並列発生。step ごとの縦の房(蜂群)にして「一斉に起きた」を見せる。
       同じ step の人が重ならないよう、房の中で高さいっぱいに広げる。 */
    const cnt = new Map(), seen = new Map();
    for(let i=0;i<n;i++) cnt.set(T.s[i], (cnt.get(T.s[i])||0)+1);
    for(let i=0;i<n;i++){
      const st2 = T.s[i], m = cnt.get(st2), k = seen.get(st2)||0;
      seen.set(st2, k+1);
      px[i] = 30 + (st2-s0)/(s1-s0)*(W-60) + ((k % 7) - 3)*2.4;
      py[i] = 22 + (k+0.5)/m*(H-46);
    }
    c.strokeStyle = 'rgba(255,255,255,.10)';
    c.beginPath(); c.moveTo(24, H-12); c.lineTo(W-24, H-12); c.stroke();
  } else {
    const kids = []; for(let i=0;i<n;i++) kids.push([]);
    const roots = [];
    for(let i=0;i<n;i++){ if(T.p[i] < 0) roots.push(i); else kids[T.p[i]].push(i); }
    const leaf = new Float64Array(n);
    for(let i=n-1;i>=0;i--){
      if(!kids[i].length){ leaf[i] = 1; continue; }
      let s = 0; for(const k of kids[i]) s += leaf[k]; leaf[i] = s;
    }
    const a0 = new Float64Array(n), a1 = new Float64Array(n);
    let tot = 0; for(const r of roots) tot += leaf[r];
    let acc = 0;
    for(const r of roots){ a0[r] = acc/tot*6.28318; acc += leaf[r];
                           a1[r] = acc/tot*6.28318; }
    for(let i=0;i<n;i++){
      if(!kids[i].length) continue;
      let a = a0[i]; const span = a1[i]-a0[i];
      for(const k of kids[i]){ a0[k] = a; a += span*leaf[k]/leaf[i]; a1[k] = a; }
    }
    let maxd = 0; for(let i=0;i<n;i++) maxd = Math.max(maxd, T.d[i]);
    const cx = W/2, cy = H/2, RR = Math.min(W,H)/2 - 20;
    const base = roots.length > 1 ? 0.16 : 0;
    /* 深い連鎖(ホップが何十段もある)は同心円だと潰れるので渦巻きにする。
       角度も深さで回すと「数珠つなぎ」がそのまま螺旋に見える。 */
    const spiral = maxd > 24, turns = 3.0;
    for(let i=0;i<n;i++){
      const mid = (a0[i]+a1[i])/2;
      const f = maxd > 0 ? T.d[i]/maxd : 0;
      const rr = (base + (1-base)*f)*RR;
      const ang = spiral ? (6.28318*turns*f + (mid-Math.PI)*0.30) : mid;
      px[i] = cx + Math.cos(ang)*rr; py[i] = cy + Math.sin(ang)*rr;
    }
    c.strokeStyle = 'rgba(148,163,184,.30)'; c.lineWidth = 1;
    c.beginPath();
    for(let i=0;i<n;i++){ const p = T.p[i]; if(p < 0) continue;
      c.moveTo(px[p], py[p]); c.lineTo(px[i], py[i]); }
    c.stroke();
  }
  /* 火元(根)だけ黄色。ただし辺ゼロ = 全員が根なので、そのときは step の色に戻す
     (全部黄色だと「いつ知ったか」が読めなくなる)。 */
  const markRoot = it.edges > 0;
  for(let i=0;i<n;i++){
    const isRoot = markRoot && T.p[i] < 0;
    const r = isRoot ? 4.2 : (T.d[i] <= 1 ? 3.0 : 2.3);
    c.beginPath(); c.arc(px[i], py[i], r, 0, 6.284);
    c.fillStyle = isRoot ? '#fbbf24' : col(i); c.fill();
    if(isRoot){ c.strokeStyle = '#0b0f16'; c.lineWidth = 1.2; c.stroke(); }
    TGEO.push({x:px[i], y:py[i], i:i});
  }
  c.fillStyle = '#6b7686'; c.font = '9.5px sans-serif'; c.textAlign = 'right';
  c.fillText('ノード ' + T.n.toLocaleString()
    + (T.population > T.n ? ' / 母集団 '+T.population.toLocaleString() : '')
    + (it.edges<=0 ? ' · 横軸=時刻'
       : ' · 深さ '+(T.depth_shown!==undefined&&T.depth_shown!==it.depth
                    ? T.depth_shown+'/'+it.depth : it.depth)
         + ' · 分岐 '+(it.branch!==undefined?it.branch:'-')),
    W-10, H-8);
}
const tcv = document.getElementById('tcv');
tcv.addEventListener('mousemove', e=>{
  const rect = tcv.getBoundingClientRect(), mx = e.clientX-rect.left,
        my = e.clientY-rect.top;
  const tip = document.getElementById('ttip');
  let best = null, bd = 90;
  for(const g of TGEO){ const d = (g.x-mx)*(g.x-mx)+(g.y-my)*(g.y-my);
    if(d < bd){ bd = d; best = g; } }
  if(!best){ tip.style.display = 'none'; return; }
  const T = casTree(CSEL), i = best.i, ck = clockText(T.s[i]);
  tip.innerHTML = '<b>'+esc(cname(T.a[i]))+'</b>'
    + (T.p[i] < 0 ? ' <span style="color:#fbbf24">(火元)</span>' : '')
    + '<br><span style="color:var(--dim)">hop '+T.d[i]+' · Day '+(ck.day+1)+' '
    + ck.hhmm + (T.p[i] >= 0 ? ' · ←'+esc(cname(T.a[T.p[i]])) : '') + '</span>';
  tip.style.display = 'block';
  tip.style.left = Math.min(tcv.clientWidth-230, mx+12)+'px';
  tip.style.top = (my+12)+'px';
});
tcv.addEventListener('mouseleave', ()=>{
  document.getElementById('ttip').style.display = 'none'; });
tcv.addEventListener('click', e=>{
  const rect = tcv.getBoundingClientRect(), mx = e.clientX-rect.left,
        my = e.clientY-rect.top;
  let best = null, bd = 90;
  for(const g of TGEO){ const d = (g.x-mx)*(g.x-mx)+(g.y-my)*(g.y-my);
    if(d < bd){ bd = d; best = g; } }
  if(!best) return;
  const T = casTree(CSEL);
  if(T.x[best.i] === -32768){ CT = T.s[best.i];
    document.getElementById('iscrub').value = CT; drawIso(); drawCurve(); return; }
  jumpToMap(T.s[best.i], T.x[best.i], T.y[best.i]);
});

/* ---------- 右パネル ---------- */
function fillCasFacts(){
  const it = curCas(), c = CAS(), host = document.getElementById('cfacts');
  if(!it){ host.innerHTML = '<div class="note">カスケード未選択。</div>'; return; }
  let h = '';
  if(it.text) h += '<div class="quote">'+esc(it.text)+'</div>';
  h += kv('種別', esc(c.kind_labels[String(it.kind)]));
  h += kv('識別子', esc(it.key));
  if(it.place) h += kv('場所', esc(it.place)
    + (it.title_src === 'poi_near'
       ? ' <span class="warn">(推定 ±'+(it.place_dist_m||0)+'m)</span>'
       : ' <span style="color:var(--dim)">(belief_transmit 由来)</span>'));
  if(it.coiner !== undefined && it.coiner !== null)
    h += kv('造語者', esc(cname(it.coiner))
      + (it.coin_place ? ' <span style="color:var(--dim)">@'+esc(it.coin_place)+'</span>' : ''));
  if(it.author !== undefined) h += kv('元の著者', esc(cname(it.author)));
  h += kv('採用者', it.n.toLocaleString()+' 人');
  h += kv('人づての辺', it.edges.toLocaleString()+' 本');
  h += kv('根(親なし)', it.roots.toLocaleString()+' 人');
  h += kv('最大ホップ', it.depth);
  h += kv('形', esc(SHAPE_LABEL[it.shape]||it.shape));
  const ck0 = clockText(it.s0), ck1 = clockText(it.s1);
  h += kv('期間', 'Day '+(ck0.day+1)+' '+ck0.hhmm+' → Day '+(ck1.day+1)+' '+ck1.hhmm
    + ' <span style="color:var(--dim)">('+(it.s1-it.s0)+' step)</span>');
  h += kv('半数到達', casHalf(it)+' step');
  if(it.reach) h += kv('viral reach', it.reach.toLocaleString()
    + ' <span style="color:var(--dim)">('+(it.viral_events||0)+' 回加重)</span>');
  if(it.adopts_n !== undefined){
    h += kv('label_adopt', (it.adopts_n||0)+' 件');
    h += kv('vocab_use', (it.uses||0)+' 件');
  }
  h += kv('等時線ヘックス', it.iso.length
    + (it.iso_population > it.iso.length ? ' / 母集団 '+it.iso_population : ''));
  h += kv('分岐(辺/媒介者)', (it.branch!==undefined?it.branch:'—')
    + ' <span style="color:var(--dim)">媒介者 '+(it.internal||0)+' 人</span>');
  const T = casTree(CSEL);
  if(T && T.population > T.n)
    h += '<div class="note">木のノードは母集団 '+T.population.toLocaleString()
       + ' から '+T.n.toLocaleString()+' へ間引いた。残す基準は<b>子孫の数</b>で、'
       + 'これは祖先について閉じているので<b>幹と太い枝がそのまま残り</b>、辺の'
       + '付け替えは '+(T.contracted||0)+' 本(= 0 なら描いた辺はすべて実在の辺)。'
       + (T.depth_shown!==undefined && T.depth_shown < it.depth
          ? ' 描画された深さは '+T.depth_shown+'(全体は '+it.depth+')。' : '')
       + '</div>';
  h += '<div class="note">'+esc(SHAPE_HELP[it.shape]||'')+'</div>';
  h += storyBacklink('cas', CSEL);
  host.innerHTML = h;
  wireBacklinks(host);
}
function fillCasCh(){
  const it = curCas(), host = document.getElementById('cch');
  if(!it){ host.innerHTML = '<div class="note">—</div>'; return; }
  const ks = Object.keys(it.ch||{});
  if(!ks.length){ host.innerHTML = '<div class="note">チャネルの記録なし。</div>'; return; }
  let mx = 1; for(const k of ks) mx = Math.max(mx, it.ch[k]);
  const NAME = {witness:'目撃(自分の目)', transmit:'人づて(発話)', face:'対面',
                sns:'SNS', dm:'DM', search:'検索', direct:'直接', agent:'他人から',
                verify:'現場で確認', net:'ネット'};
  let h = '';
  for(const k of ks){
    h += '<div class="chbar"><div class="lb" title="'+esc(k)+'">'
      + esc(NAME[k]||k)+'</div><div class="tr"><div class="fl" style="width:'
      + (it.ch[k]/mx*100).toFixed(1)+'%"></div></div><div class="vv">'
      + it.ch[k].toLocaleString()+'</div></div>';
  }
  h += '<div class="note">信念は「採用の理由」の内訳(witness = 自分で見た = '
     + '伝播ではない)。語彙とリシェアは<b>辺</b>の内訳。</div>';
  host.innerHTML = h;
}
function fillCasPop(){
  const c = CAS(), host = document.getElementById('cpop');
  if(!c){ host.innerHTML = '<div class="note">素材なし。</div>'; return; }
  const p = c.population;
  let h = '';
  h += kv('掲載', c.shown + ' / cap ' + c.cap + ' 本');
  h += '<div class="note">枠の埋まり方: ' + Object.keys(c.quota).map(k =>
    esc(c.quota[k].label)+' '+c.quota[k].filled+'/'+c.quota[k].target).join(' · ')
    + '</div>';
  h += kv('信念 fact', (p.belief_facts||0).toLocaleString()
    + ' 本 <span style="color:var(--dim)">(うち人づて有 '
    + (p.belief_facts_with_edges||0)+')</span>');
  h += kv('· belief_update', (p.belief_kept||0).toLocaleString()+' / '
    + (p.belief_rows||0).toLocaleString()+' 行');
  h += kv('· うち人づての辺', (p.belief_edges||0).toLocaleString()+' 本');
  h += kv('リシェア連鎖', (p.reshare_cascades||0).toLocaleString()+' 本');
  h += kv('· sns_reshare', (p.reshare_kept||0).toLocaleString()+' / '
    + (p.reshare_rows||0).toLocaleString()+' 行');
  h += kv('· viral_cascade', (p.viral_rows||0).toLocaleString()+' 件');
  h += kv('語彙 item', (p.vocab_items||0).toLocaleString()+' 本');
  h += kv('· transmission', (p.vocab_rows||0).toLocaleString()+' 行');
  h += kv('· 造語(coin)', (p.coins||0).toLocaleString()+' 件');
  h += kv('· label_adopt', (p.label_adopts||0).toLocaleString()+' 件');
  h += kv('· place_label_bind', (p.place_binds||0).toLocaleString()+' 件');
  h += kv('stock_out(fact の源)', (p.stock_out_rows||0).toLocaleString()+' 行');
  if((p.belief_edges||0) === 0 && (p.belief_rows||0) > 0)
    h += '<div class="note warn">このランの信念は<b>人づての辺が 1 本も無い</b>。'
       + '= 全員が独立に同じ現場を見た(並列発生)。伝播木ではないことを、'
       + '形「並列発生」としてそのまま出している。</div>';
  h += '<div class="note">' + c.notes.map(esc).join('<br>') + '</div>';
  host.innerHTML = h;
}
function propInit(){
  CSEL = -1; CTREE = {};
  document.getElementById('ckind').onchange = buildCasList;
  document.getElementById('csort').onchange = buildCasList;
  document.getElementById('cfind').oninput = buildCasList;
  buildCasList(); fillCasPop();
  const c = CAS();
  if(c && c.items.length && CORD.length) selectCas(CORD[0]);
  else { drawIso(); drawCurve(); drawTree(); fillCasFacts(); fillCasCh(); }
}
window.addEventListener('resize', ()=>{ if(TAB==='prop') propResize(); });

/* =========================================================================
   画面4「物語ピン + 今日のハイライト」(story sifting)
   -------------------------------------------------------------------------
   ビルド側が宣言的パターンで拾った連鎖を、surprise 降順で並べる。地図には
   「上位 N 件」だけをピンで刺し(大きさ = surprise・字 = 種別)、押すとその
   現場・その瞬間へ飛ぶ。カードには拍(beat)の列と、当事者の**その時の思考**
   (llm_journal の 1 ホップ)を添える。
   ========================================================================= */
let SSEL = -1, SORD = [], PINGEO = [], STIDX = null;

function ST(){ return R() ? R().stories : null; }
function stItems(){ const s = ST(); return (s && s.items) ? s.items : []; }
function curStory(){ const a = stItems();
  return (SSEL >= 0 && SSEL < a.length) ? a[SSEL] : null; }
function sname(id){ const s = ST(); const v = s && s.names && s.names[String(id)];
  return (v && v[0]) ? v[0] : '#'+id; }
function smeta(id){ const s = ST(); const v = s && s.names && s.names[String(id)];
  if(!v) return ''; return (v[1]?v[1]+'歳':'') + (v[2]?'・'+v[2]:'') + (v[3]?'・'+v[3]:''); }
function patCol(p){ const s = ST();
  return (s && s.pat_colors && s.pat_colors[String(p)]) || '#94a3b8'; }
function patLb(p){ const s = ST();
  return (s && s.pat_labels && s.pat_labels[String(p)]) || ('種別'+p); }
function patGl(p){ const s = ST();
  return (s && s.pat_glyphs && s.pat_glyphs[String(p)]) || '·'; }
function glyphHTML(p){ return '<span class="pglyph" style="background:'+patCol(p)+'">'
  + esc(patGl(p)) + '</span>'; }
/* 画面2/3 → 画面4 の逆引き索引(ラン切替のたびに作り直す) */
function stIndex(){
  if(STIDX) return STIDX;
  STIDX = {pair:{}, cas:{}};
  stItems().forEach((it,i)=>{ const L = it.link;
    if(!L) return;
    if(L.pair !== undefined && STIDX.pair[L.pair] === undefined) STIDX.pair[L.pair] = i;
    if(L.cas !== undefined && STIDX.cas[L.cas] === undefined) STIDX.cas[L.cas] = i; });
  return STIDX;
}
function storyBacklink(kind, i){
  if(i === null || i === undefined || i < 0) return '';
  const j = stIndex()[kind][i];
  if(j === undefined) return '';
  const it = stItems()[j];
  return '<div class="lnkrow"><button class="lnk" data-story="'+j+'">'
    + glyphHTML(it.pat) + '物語「' + esc(it.title) + '」へ(驚き '
    + (it.surprise||0).toFixed(1) + ')</button></div>';
}
function wireBacklinks(host){
  host.querySelectorAll('[data-story]').forEach(el =>
    el.onclick = ()=> gotoStory(+el.dataset.story));
}
function gotoStory(i){
  document.getElementById('sfind').value = '';
  document.getElementById('spat').value = '';
  document.getElementById('sthonly').checked = false;
  buildStoryList(); setTab('story'); selectStory(i);
}
function gotoPair(i){
  if(!P() || i < 0 || i >= P().items.length) return;
  document.getElementById('pfind').value = '';
  buildPairList(); setTab('pair'); selectPair(i);
  const el = document.querySelector('#prows .prow[data-i="'+i+'"]');
  if(el) el.scrollIntoView({block:'center'});
}
function gotoCas(i){
  if(!CAS() || i < 0 || i >= CAS().items.length) return;
  document.getElementById('cfind').value = '';
  document.getElementById('ckind').value = '';
  buildCasList(); setTab('prop'); selectCas(i);
  const el = document.querySelector('#crows .crow[data-i="'+i+'"]');
  if(el) el.scrollIntoView({block:'center'});
}

/* ---------- 俯瞰地図の物語ピン ---------- */
function drawPins(){
  PINGEO = [];
  const badge = document.getElementById('pinN');
  const s = ST();
  document.getElementById('pinTopV').textContent =
    document.getElementById('pinTop').value;
  if(!document.getElementById('lyPin').checked || !s || !s.items.length){
    badge.textContent = s ? ' (0/'+ (s.items||[]).length +')' : ' (素材なし)';
    return;
  }
  const items = s.items, topN = +document.getElementById('pinTop').value;
  /* items はビルド側で surprise 降順。上位 N 件のうち**座標を持つもの**を刺す
     (関係イベントのように座標を持たない物語は地図に出さない = 偽の位置を作らない)。 */
  const pool = [];
  for(let i=0;i<items.length && pool.length<topN;i++)
    if(items[i].x !== -32768) pool.push(i);
  const smax = pool.length ? (items[pool[0]].surprise||0) : 1;
  const smin = pool.length ? (items[pool[pool.length-1]].surprise||0) : 0;
  ctx.save();
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  let hidden = 0;
  for(const i of pool){
    const it = items[i], p = tf(it.x, it.y), sx = p[0], sy = p[1];
    if(sx<-40||sy<-40||sx>cv.clientWidth+40||sy>cv.clientHeight+40) continue;
    const t = (smax > smin) ? (it.surprise - smin)/(smax - smin) : 1;
    const r = 8 + 9*Math.max(0, Math.min(1, t));
    const sel = (i === SSEL);
    /* 重なりの抑制: 渋谷駅前には物語が密集する。**驚きの高い順に置いて**、
       既に置いたピンとほぼ重なる位置は譲る(隠した数はバッジに出す)。
       選択中のピンだけは必ず描く(押した物語が消えないため)。 */
    let clash = false;
    if(!sel) for(const g of PINGEO){
      const dx = g.sx-sx, dy = g.sy-sy;
      if(dx*dx + dy*dy < 0.62*(g.r+r)*(g.r+r)){ clash = true; break; } }
    if(clash){ hidden++; continue; }
    const live = (cur >= it.s0 && cur <= it.s1);
    ctx.globalAlpha = live ? 1 : 0.62;
    ctx.beginPath(); ctx.arc(sx, sy, r, 0, 7);
    ctx.fillStyle = patCol(it.pat); ctx.fill();
    ctx.lineWidth = sel ? 2.8 : (live ? 1.8 : 1.0);
    ctx.strokeStyle = sel ? '#ffffff' : 'rgba(10,14,20,.85)';
    ctx.stroke();
    if(live){ ctx.globalAlpha = 0.5; ctx.beginPath(); ctx.arc(sx, sy, r+4.5, 0, 7);
      ctx.strokeStyle = patCol(it.pat); ctx.lineWidth = 1.4; ctx.stroke(); }
    ctx.globalAlpha = 1;
    ctx.fillStyle = '#0a0e14';
    ctx.font = '700 ' + Math.max(10, Math.round(r*1.02))
      + 'px "Hiragino Kaku Gothic ProN",Meiryo,sans-serif';
    ctx.fillText(patGl(it.pat), sx, sy+0.5);
    PINGEO.push({i:i, sx:sx, sy:sy, r:r});
  }
  ctx.restore();
  badge.textContent = ' ('+PINGEO.length+'/'+items.length
    + (hidden ? ' · 重なり '+hidden+' 件は拡大で' : '') + ')';
}
function showPinCard(i){
  const it = stItems()[i], host = document.getElementById('pincard');
  if(!it){ host.style.display = 'none'; return; }
  const c0 = clockText(it.s0);
  host.innerHTML = '<span class="cl" id="pcx">✕</span>'
    + '<div style="font-weight:650;font-size:12.5px;margin-bottom:2px">'
      + glyphHTML(it.pat) + esc(it.title) + '</div>'
    + '<div style="color:var(--dim);font-size:11px">' + esc(it.sub) + '</div>'
    + '<div style="margin:4px 0 2px">驚き <span class="sscore">'
      + (it.surprise||0).toFixed(2) + '</span>'
      + ' <span style="color:var(--dim)">· ' + esc(patLb(it.pat))
      + ' · Day '+(c0.day+1)+' '+c0.hhmm + '</span></div>'
    + (it.why && it.why.length
        ? '<div style="color:#cbd5e1;font-size:11px;margin-top:3px">'
          + esc(it.why[0]) + '</div>' : '')
    + '<div class="lnkrow"><button class="lnk" id="pcgo">詳しく見る</button>'
      + '<button class="lnk" id="pcnow">この瞬間へ</button></div>';
  host.style.display = 'block';
  document.getElementById('pcx').onclick = ()=>{ host.style.display = 'none'; };
  document.getElementById('pcgo').onclick = ()=> gotoStory(i);
  document.getElementById('pcnow').onclick = ()=>{
    cur = Math.max(0, Math.min(nSteps()-1, it.s0));
    document.getElementById('scrub').value = cur; requestDraw(); };
}

/* ---------- 一覧 ---------- */
function storyRowHTML(it, i){
  const c0 = clockText(it.s0);
  return '<div class="srow'+(i===SSEL?' on':'')+'" data-i="'+i+'">'
    + '<div class="nm">' + glyphHTML(it.pat) + esc(it.title) + '</div>'
    + '<div class="mt"><span class="sscore">'+(it.surprise||0).toFixed(2)+'</span>'
    + '<span>Day '+(c0.day+1)+' '+c0.hhmm+'</span>'
    + '<span>'+it.beats_n+' 拍</span>'
    + (it.thought ? '<span style="color:var(--t2)">思考あり</span>' : '')
    + (it.x === -32768 ? '<span style="color:var(--dim)">地図に出ない</span>' : '')
    + '</div>'
    + '<div class="mt" style="color:#8b98a8">'+esc(it.sub)+'</div></div>';
}
function buildStoryList(){
  const s = ST(), host = document.getElementById('srows');
  const cnt = document.getElementById('scount');
  const sel = document.getElementById('spat');
  if(sel.options.length <= 1 && s && s.pat_labels){
    Object.keys(s.pat_labels).sort((a,b)=>a-b).forEach(k=>{
      const o = document.createElement('option');
      o.value = k; o.textContent = '種別: ' + s.pat_labels[k]; sel.appendChild(o); });
  }
  if(!s || !s.items.length){
    host.innerHTML = '<div class="empty">このランには画面4 の素材がありません'
      + '(走行中のランは flush 済みの part だけを見ます)。</div>';
    cnt.textContent = ''; SORD = []; return;
  }
  const pf = sel.value, mode = document.getElementById('ssort').value;
  const q = (document.getElementById('sfind').value||'').trim().toLowerCase();
  const thonly = document.getElementById('sthonly').checked;
  SORD = s.items.map((it,i)=>i);
  if(pf !== '') SORD = SORD.filter(i => String(s.items[i].pat) === pf);
  if(thonly) SORD = SORD.filter(i => !!s.items[i].thought);
  if(q) SORD = SORD.filter(i => { const it = s.items[i];
    const who = (it.who||[]).map(w=>sname(w)+' #'+w).join(' ');
    return ((it.title||'')+' '+(it.sub||'')+' '+(it.why||[]).join(' ')+' '
      + who + ' ' + (it.beats||[]).map(b=>b[3]).join(' ')).toLowerCase().indexOf(q) >= 0; });
  const val = it => mode==='time' ? (it.s0||0)
    : mode==='size' ? -(it.size||0)
    : mode==='beats' ? -(it.beats_n||0) : -(it.surprise||0);
  SORD.sort((x,y)=> val(s.items[x]) - val(s.items[y]) || x-y);
  host.innerHTML = SORD.map(i => storyRowHTML(s.items[i], i)).join('');
  cnt.innerHTML = '母集団 ' + (s.population.matches||0).toLocaleString()
    + ' 件 → 掲載 ' + s.shown + ' 件'
    + (SORD.length !== s.items.length ? ' / 絞り込み '+SORD.length+' 件' : '');
  host.querySelectorAll('.srow').forEach(el =>
    el.onclick = ()=> selectStory(+el.dataset.i));
}
function selectStory(i){
  SSEL = i;
  document.querySelectorAll('#srows .srow').forEach(el =>
    el.classList.toggle('on', +el.dataset.i === i));
  renderStory(); fillStoryScore(); fillThink(); syncHash();
  if(TAB === 'map') requestDraw();
}

/* ---------- 中央: ハイライトカード ---------- */
function renderStory(){
  const it = curStory(), host = document.getElementById('scard');
  if(!it){ host.innerHTML = '<div class="empty">左の一覧から物語を選ぶと、'
    + '当事者・拍(beat)・現場・本人の思考が並びます。<br>'
    + '俯瞰タブの地図ピンを押しても同じカードに来られます。</div>'; return; }
  const c0 = clockText(it.s0), c1 = clockText(it.s1);
  let h = '<h3>' + glyphHTML(it.pat) + esc(it.title) + '</h3>';
  h += '<div class="subt">' + esc(patLb(it.pat)) + ' · ' + esc(it.sub)
     + ' · Day '+(c0.day+1)+' '+c0.hhmm
     + (it.s1 !== it.s0 ? ' → Day '+(c1.day+1)+' '+c1.hhmm : '')
     + ' <span class="sscore">驚き ' + (it.surprise||0).toFixed(2) + '</span></div>';
  if(it.who && it.who.length){
    h += '<div class="subt">当事者: ' + it.who.slice(0,8).map(w =>
      esc(sname(w)) + (smeta(w) ? ' <span style="opacity:.7">('+esc(smeta(w))+')</span>' : ''))
      .join(' · ') + (it.who.length > 8 ? ' ほか '+(it.who.length-8)+' 人' : '') + '</div>';
  }
  h += '<div style="margin-top:8px">';
  (it.beats||[]).forEach(b => {
    const jump = (b[1] !== -32768);
    h += '<div class="beat"><div class="st">' + tlClock(b[0]) + '</div>'
      + '<div class="bd' + (jump ? ' jump" data-step="'+b[0]+'" data-x="'+b[1]
          + '" data-y="'+b[2]+'"' : '"') + '>'
      + (b[4] >= 0 ? '<div class="who">'+esc(sname(b[4]))+'</div>' : '')
      + esc(b[3]) + '</div></div>';
  });
  h += '</div>';
  if(it.beats_n > (it.beats||[]).length)
    h += '<div class="note">拍は '+it.beats_n+' 件のうち先頭 '+it.beats.length
       + ' 件を表示(1 物語の上限)。</div>';
  if(it.why && it.why.length)
    h += '<ul class="why">' + it.why.map(w=>'<li>'+esc(w)+'</li>').join('') + '</ul>';
  h += '<div class="lnkrow">';
  if(it.x !== -32768)
    h += '<button class="lnk" id="sgomap">俯瞰のこの現場へ</button>';
  if(it.link && it.link.pair !== undefined)
    h += '<button class="lnk" id="sgopair">関係の伝記でこの 2 人を見る</button>';
  if(it.link && it.link.cas !== undefined)
    h += '<button class="lnk" id="sgocas">伝播でこの木を見る</button>';
  h += '</div>';
  host.innerHTML = h;
  host.querySelectorAll('.bd.jump').forEach(el => el.onclick = ()=>
    jumpToMap(+el.dataset.step, +el.dataset.x, +el.dataset.y));
  const gm = document.getElementById('sgomap');
  if(gm) gm.onclick = ()=> jumpToMap(it.s0, it.x, it.y);
  const gp = document.getElementById('sgopair');
  if(gp) gp.onclick = ()=> gotoPair(it.link.pair);
  const gc = document.getElementById('sgocas');
  if(gc) gc.onclick = ()=> gotoCas(it.link.cas);
}

/* ---------- 右パネル ---------- */
function fillStoryScore(){
  const it = curStory(), s = ST(), host = document.getElementById('sscore');
  if(!it){ host.innerHTML = '<div class="note">物語未選択。</div>'; return; }
  const p = it.parts || {};
  let h = '';
  h += kv('surprise', '<span class="sscore">'+(it.surprise||0).toFixed(3)+'</span>');
  h += kv('稀少度 -log2(頻度)', (p.rarity!==undefined?p.rarity:'—'));
  h += kv('規模 z(種別内)', (p.size_z!==undefined?p.size_z:'—'));
  h += kv('稀少構成ボーナス', (p.bonus!==undefined?p.bonus:'—'));
  h += kv('規模(この種別の量)', it.size);
  h += '<div class="note">'+esc((s.surprise&&s.surprise.formula)||'')+'</div>';
  if(s.surprise && s.surprise.rarity_by_pat){
    h += '<div class="note">種別ごとの稀少度(母集団 / L1 '
       + (s.surprise.total_events||0).toLocaleString() + ' 行):</div>';
    Object.keys(s.surprise.rarity_by_pat).sort((a,b)=>a-b).forEach(k =>
      h += kv(patLb(k), s.surprise.rarity_by_pat[k]
        + ' <span style="color:var(--dim)">('
        + ((s.population.by_pat||{})[k]||0).toLocaleString()+' 件)</span>'));
  }
  host.innerHTML = h;
}
function fillThink(){
  const it = curStory(), s = ST(), host = document.getElementById('sthink');
  const th = s ? s.thoughts : null;
  if(!it){ host.innerHTML = '<div class="note">物語未選択。</div>'; return; }
  if(!it.thought){
    let why = 'この物語の当事者には、その時刻の LLM 呼び出しが残っていない。';
    if(th && !th.available) why = 'このランに llm_journal.jsonl.gz が無い'
      + '(ルール支配のランでは思考そのものが起きない)。';
    host.innerHTML = '<div class="note">' + esc(why) + '</div>'
      + (th ? '<div class="note">l1b_llm 索引 '+(th.index_rows||0).toLocaleString()
        + ' 行 / 思考を付けた物語 '+(th.attached||0)+' 件(上限あり・'
        + (th.truncated?'journal は途中まで':'journal は最後まで')+'走査)</div>' : '');
    return;
  }
  const t = it.thought, c = clockText(t.step);
  let h = kv('誰の思考', esc(sname(t.agent)) + ' <span style="color:var(--dim)">'
    + esc(smeta(t.agent)) + '</span>');
  h += kv('いつ', 'Day '+(c.day+1)+' '+c.hhmm+' (step '+t.step+')');
  h += kv('用途 / モデル', esc(t.purpose||'?') + ' · ' + esc(t.backend||'?')
    + (t.cached ? ' <span style="color:var(--dim)">(cache)</span>' : ''));
  h += '<div class="note">何が見えていたか(プロンプト末尾 '
    + (t.prompt_clipped ? '260 字・全 '+(t.prompt_len||0)+' 字' : '全文') + ')</div>';
  h += '<div class="think prompt">' + esc((t.prompt_clipped?'…':'') + (t.prompt||'')) + '</div>';
  h += '<div class="note">どう考えて、何をしたか(応答'
    + (t.response_clipped ? ' 先頭 200 字・全 '+(t.response_len||0)+' 字' : '') + ')</div>';
  h += '<div class="think resp">' + esc(t.response||'')
    + (t.response_clipped ? ' …' : '') + '</div>';
  h += '<div class="note">llm_call_id ' + esc(t.call) + ' · rng '
    + esc(t.rng_key||'—') + '</div>';
  host.innerHTML = h;
}
function fillStoryPop(){
  const s = ST(), host = document.getElementById('spop');
  if(!s){ host.innerHTML = '<div class="note">素材なし。</div>'; return; }
  const po = s.population;
  let h = kv('パターンのマッチ', (po.matches||0).toLocaleString()+' 件');
  h += kv('掲載', s.shown + ' 件(上限 '+s.cap+')');
  h += '<div class="note">種別ごとの枠(先に埋める)</div>';
  Object.keys(s.quota).sort((a,b)=>a-b).forEach(k => h += kv(esc(s.quota[k].label),
    (s.quota[k].shown!==undefined ? s.quota[k].shown : s.quota[k].filled)
    + ' 件 <span style="color:var(--dim)">(枠 ' + s.quota[k].filled + '/'
    + s.quota[k].target + ' · 母集団 '
    + ((po.by_pat||{})[k]||0).toLocaleString()+')</span>'));
  h += '<div class="note">素材の母集団</div>';
  h += kv('L1 行', (po.l1_rows||0).toLocaleString());
  h += kv('物語 kind の行', (po.story_rows||0).toLocaleString()
    + ' / ' + (po.story_rows_population||0).toLocaleString());
  h += kv('遺失物の連鎖', (po.lost_chains||0).toLocaleString()+' 本');
  h += kv('倒れた/負傷/事故', (po.ems_onsets||0).toLocaleString()+' 件'
    + (po.ems_orphans ? '(発生が期間外 '+po.ems_orphans+')' : ''));
  h += kv('hop>=2 のカスケード', (po.deep_cascades||0).toLocaleString()+' 本');
  h += kv('破綻→再構築のペア', (po.pair_drama||0).toLocaleString()+' 組');
  h += kv('開かれた集会', (po.events_hosted||0).toLocaleString()+' 件'
    + '(参加 '+(po.events_attended||0)+')');
  const gi = po.gathering_intent || {};
  h += kv('gathering_intent', gi.available
    ? (gi.cells||0).toLocaleString()+' セル / '+(gi.rows||0).toLocaleString()+' 行'
    : 'サイドカーなし');
  h += kv('犯罪 / 警察の行', (po.crime_rows||0).toLocaleString()
    + ' / ' + (po.police_rows||0).toLocaleString());
  h += kv('迷惑行為', (po.nuisance_rows||0).toLocaleString()+' 行 → '
    + (po.nuisance_cells||0).toLocaleString()+' セル(騒ぎ '
    + (po.nuisance_bursts||0).toLocaleString()+')');
  (s.notes||[]).forEach(n => h += '<div class="note">· '+esc(n)+'</div>');
  host.innerHTML = h;
}
function storyInit(){
  SSEL = -1; STIDX = null;
  document.getElementById('pincard').style.display = 'none';
  const sel = document.getElementById('spat');
  while(sel.options.length > 1) sel.remove(1);
  sel.value = '';
  document.getElementById('spat').onchange = buildStoryList;
  document.getElementById('ssort').onchange = buildStoryList;
  document.getElementById('sfind').oninput = buildStoryList;
  document.getElementById('sthonly').onchange = buildStoryList;
  buildStoryList(); fillStoryPop();
  if(SORD.length) selectStory(SORD[0]);
  else { renderStory(); fillStoryScore(); fillThink(); }
}

/* ---------- 起動(#run=A&step=54 で「その瞬間」を直接開ける) ---------- */
let bootTab = 'map', bootPair = -1, bootCas = -1, bootStory = -1;
(function(){
  const h = new URLSearchParams(location.hash.replace(/^#/, ''));
  const rk = (h.get('run')||'').toUpperCase();
  if(HAS(rk)) RK = rk;
  const s = parseInt(h.get('step'), 10);
  if(isFinite(s)) cur = Math.max(0, Math.min(nSteps()-1, s));
  const tb = h.get('tab');
  if(tb === 'pair' || tb === 'prop' || tb === 'story') bootTab = tb;
  const pi = parseInt(h.get('pair'), 10);
  if(isFinite(pi)) bootPair = pi;
  const ci = parseInt(h.get('cas'), 10);
  if(isFinite(ci)) bootCas = ci;
  const si = parseInt(h.get('story'), 10);
  if(isFinite(si)) bootStory = si;
})();
function syncHash(){
  try { history.replaceState(null, '', '#run='+RK+'&step='+cur+'&tab='+TAB
    + (TAB==='pair' && PSEL>=0 ? '&pair='+PSEL : '')
    + (TAB==='prop' && CSEL>=0 ? '&cas='+CSEL : '')
    + (TAB==='story' && SSEL>=0 ? '&story='+SSEL : '')); } catch(_){}
}
scrub.addEventListener('change', syncHash);
scrub.max = nSteps()-1; scrub.value = cur;
fillTop(); fillAbout(); fillRel(); buildCharts();
pairInit();
propInit();
storyInit();
fitAll(); resize();
if(bootPair >= 0 && P() && bootPair < P().items.length) selectPair(bootPair);
if(bootCas >= 0 && CAS() && bootCas < CAS().items.length) selectCas(bootCas);
if(bootStory >= 0 && bootStory < stItems().length) selectStory(bootStory);
if(bootTab !== 'map') setTab(bootTab);
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
    ap.add_argument("--cascades", type=int, default=CAS_CAP_DEFAULT,
                    help=f"画面3 に載せるカスケード数(既定 {CAS_CAP_DEFAULT})。"
                         "母集団は常に併記される")
    ap.add_argument("--cascade-nodes", type=int, default=CAS_NODES_DEFAULT,
                    help=f"1 カスケードの木ノード上限(既定 {CAS_NODES_DEFAULT})")
    ap.add_argument("--belief-cap", type=int, default=BELIEF_CAP_DEFAULT,
                    help="belief_update を貯める行数上限(RAM よけ・1 件 26B)")
    ap.add_argument("--reshare-cap", type=int, default=RESHARE_CAP_DEFAULT,
                    help="sns_reshare を貯める行数上限(RAM よけ・1 件 20B)")
    ap.add_argument("--stock-cap", type=int, default=STOCK_CAP_DEFAULT,
                    help="stock_out を貯める行数上限(fact の場所名の推定に使う)")
    ap.add_argument("--no-cascades", action="store_true",
                    help="画面3 の素材を作らない")
    ap.add_argument("--stories", type=int, default=STORY_CAP_DEFAULT,
                    help=f"画面4 に載せる物語数(既定 {STORY_CAP_DEFAULT})。"
                         "母集団は常に併記される")
    ap.add_argument("--thoughts", type=int, default=THOUGHT_CAP_DEFAULT,
                    help=f"思考チェーンを付ける物語数の上限(既定 {THOUGHT_CAP_DEFAULT})")
    ap.add_argument("--no-stories", action="store_true",
                    help="画面4 の素材を作らない")
    ap.add_argument("--no-thoughts", action="store_true",
                    help="llm_journal を 1 行も読まない(思考チェーンなし)")
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
    # ノード → (x, y, 名前)。**payload には入れない**(3,499 件を丸ごと埋め込まない)。
    # 画面4 が「現場を地図に置く」「n7966131106 を『渋谷駅ハチ公口』と呼ぶ」ために使う。
    # 名前つきノードは 14 件しかないので、POI の `node` 欄からも名前を引き当てる。
    node_xy: dict = {}
    try:
        _city = json.loads(Path(map_path).read_text(encoding="utf-8"))
        for _n in _city.get("nodes", ()):
            nid = _n.get("id")
            if nid is not None:
                node_xy[str(nid)] = (_n.get("x"), _n.get("y"), _n.get("name") or None)
        for _p in _city.get("pois", ()):
            nid = _p.get("node")
            nm = (_p.get("name") or "").strip()
            if not nid or not nm:
                continue
            cur = node_xy.get(str(nid))
            if cur is None:
                node_xy[str(nid)] = (_p.get("x"), _p.get("y"), nm)
            elif not cur[2]:
                node_xy[str(nid)] = (cur[0], cur[1], nm)
        del _city
    except (OSError, ValueError):
        node_xy = {}
    _log(f"basemap: {map_path.name} 道路 {basemap['roads']['n']} 本 / "
         f"鉄道 {basemap['rails']['n']} 本 / POI {basemap['pois']['shown']}"
         f"({basemap['pois']['population']} 中) / ノード座標 {len(node_xy):,} 件"
         f" — {time.time() - t0:.1f}s")

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
                                conv_cap=a.conv_cap, text_cap=a.text_cap,
                                cascades=not a.no_cascades, cas_cap=a.cascades,
                                cas_nodes=a.cascade_nodes,
                                belief_cap=a.belief_cap, reshare_cap=a.reshare_cap,
                                stock_cap=a.stock_cap,
                                stories=not a.no_stories, story_cap=a.stories,
                                thoughts=not a.no_thoughts,
                                thought_cap=a.thoughts, node_xy=node_xy)
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
             f"pairs {_n(r.get('pairs'))/1024:.0f}KB / "
             f"cascades {_n(r.get('cascades'))/1024:.0f}KB / "
             f"stories {_n(r.get('stories'))/1024:.0f}KB")
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
                     "cascades": (None if not r.get("cascades") else {
                         "shown": r["cascades"]["shown"],
                         "population": r["cascades"]["population"],
                         "quota": {t: q["filled"]
                                   for t, q in r["cascades"]["quota"].items()},
                         "by_kind": dict(Counter(
                             CAS_KIND_LABELS[i["kind"]]
                             for i in r["cascades"]["items"])),
                         "top": [{"kind": CAS_KIND_LABELS[i["kind"]],
                                  "title": i["title"], "n": i["n"],
                                  "edges": i["edges"], "depth": i["depth"],
                                  "shape": i["shape"]}
                                 for i in r["cascades"]["items"][:8]],
                     }),
                     "stories": (None if not r.get("stories") else {
                         "shown": r["stories"]["shown"],
                         "matches": r["stories"]["population"]["matches"],
                         "by_pat": {PAT_LABELS[int(k)]: v for k, v in
                                    r["stories"]["population"]["by_pat"].items()},
                         "quota": {PAT_LABELS[int(t)]: q["filled"]
                                   for t, q in r["stories"]["quota"].items()},
                         "thoughts": r["stories"]["thoughts"],
                         "top": [{"pat": PAT_LABELS[i["pat"]], "title": i["title"],
                                  "sub": i["sub"], "surprise": i["surprise"],
                                  "parts": i["parts"], "why": i["why"],
                                  "step": i["s0"], "beats": i["beats_n"],
                                  "thought": bool(i.get("thought"))}
                                 for i in r["stories"]["items"][:10]],
                     }),
                     "build_sec": timings[k]}) for k, r in runs.items()},
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":                              # pragma: no cover
    raise SystemExit(main())
