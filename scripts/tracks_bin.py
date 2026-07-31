"""軌跡の量子化型付きバイナリ形式(P0 = DT 統合の前提工事)。

なぜ要るか(実測根拠 docs/research/dt-integration-deep.md §1.3):
  1万体×1日 で `scene3d/tracks.json` 65.8MB → `viewer3d.html` 90.4MB(**80MB ゲート超過**)。
  10日ランは単純外挿で tracks.json ≈ 660MB / 自己完結 HTML ≈ 900MB = ブラウザに載らない。
  UE 向け `sim_ue.json` も 684B/agent-step = 1万体×10日 ≈ 9.8GB。
  → **軌跡を JSON 埋め込みから量子化バイナリ + チャンク遅延ロードへ移す**のが全 3D 経路の前提。

本モジュールは「変換だけ」を持つ純関数群(シム本体・L1・conf に非依存。乱数ゼロ・決定論)。
  encode_tracks(tracks)   -> (blob, header)   scene3d/tracks.json 相当 → tracks.bin
  decode_tracks(blob)     -> tracks(dict)     逆変換(下記の可逆性を参照)
  encode_sim_ue(sim_ue)   -> (blob, header)   scene3d/sim_ue.json 相当 → sim_ue.bin
  decode_sim_ue(blob)     -> sim_ue(dict)

量子化仕様(D-3: PLATEAU メッシュ埋め込みと同一の 0.05 m/unit = export_3d.PLATEAU_QUANT):
  - 位置 x,y(と UE の z)は `q = round(v/quant) - origin_q` の int16。origin_q はデータ由来の
    中央値(量子単位の整数)で、header に持つ。復号は `v = (q + origin_q) * quant`。
  - **誤差上界は quant/2 = 0.025 m**(表示用途で不可視。歩行者の描画は 1px 未満)。
  - tracks.json の座標は export_3d が 0.1m 丸めで書いているため、0.05 は 0.1 のちょうど 1/2 =
    **その場合の往復誤差は厳密に 0**(tests/test_tracks_bin.py で固定)。
  - 時刻は量子化しない(sim_min は step 数ぶんしかない小配列なので header に素で持つ)。
  - 状態 w(0=路上 -1=範囲外 -2=睡眠 -3=電車 1000+bIdx*100+floor=屋内)は **値域が広く
    int16 に入らない**(実測 max 720802)ので、素の int16 化(deep §2.2 の案)からは意図的に
    逸脱し、**出現値パレット(int32)+ uint16 索引**にした。可逆・欠損なし・サイズは同じ 2B。
    パレットが 65536 を超える場合のみ uint32 索引へ自動昇格(header の state_dtype)。

ファイル形式(GLB と同じ「固定ヘッダ + JSON + バイナリ」流儀。依存は numpy+stdlib のみ):
  magic(8s) version(u4) header_json_len(u4) header_json(utf-8, 8B 境界まで空白埋め) payload
  payload は **step 範囲で分割したチャンクの連結**。各チャンクは自己完結(他チャンク参照なし)
  なので、ビューアは再生位置に対応するチャンクだけを読めばよい(= 遅延ロードの単位)。

チャンク内レイアウト(全セクション 4B 境界。オフセットは header の chunks[i].sec に明記):
  pos    int16 [ns*na*pos_coords]   位置(x,y[,z])
  state  u2/u4 [ns*na]              状態パレット索引
  mvoff  uint32[ns+1]               step→移動レコード範囲
  mvag   uint32[n_moves]            移動レコードのエージェント索引
  mvmode uint8 [n_moves]            移動手段(0=徒歩 1=自転車 2=車 3=タクシー)
  mvpo   uint32[n_moves+1]          移動レコード→ポリライン点範囲
  mvpts  int16 [n_move_pts*2]       ポリライン点
  troff  uint32[ns+1]               step→交通セグメント範囲
  trn    int32 [ns]                 交通の台数 n(sim_ue では 0 固定=UE 側は使わない)
  trpo   uint32[n_segs+1]           セグメント→点範囲
  trpts  int16 [n_seg_pts*2]        セグメント点

可逆性の正確な記述(正直な限界):
  - tracks: 数値としては完全復元(0.1m 丸め済みデータのため誤差 0)。ただし JSON 文字列としては
    **-0.0 が 0.0 に化ける**(量子 0 に符号が無いため)。実測 rehearsal_pool10k で 3,165 箇所。
    数値等価であり描画・解析に影響しないが、「tracks.json とバイト同一」は主張しない。
  - sim_ue: uu(cm)を 0.05m 相当(既定 5uu)で量子化するため **最大 2.5cm の丸め**が入る。
"""
from __future__ import annotations

import base64
import json
import struct

import numpy as np

MAGIC_TRACKS = b"SHBTRKB1"
MAGIC_UE = b"SHBUEB01"
FORMAT_VERSION = 1

QUANT_M = 0.05                       # m/unit(export_3d.PLATEAU_QUANT と同一・D-3)
DEFAULT_CHUNK_BYTES = 8 * 1024 * 1024
I16_MIN, I16_MAX = -32768, 32767

_SECTIONS = ("pos", "state", "mvoff", "mvag", "mvmode", "mvpo", "mvpts",
             "troff", "trn", "trpo", "trpts")


# ------------------------------------------------------------------ 低水準
class _Payload:
    """4B 境界を保ちながら numpy 配列を連結し、セクション先頭オフセットを記録する。"""

    def __init__(self) -> None:
        self._parts: list[bytes] = []
        self.n = 0

    def add(self, name: str, arr: np.ndarray, sec: dict) -> None:
        pad = (-self.n) % 4
        if pad:
            self._parts.append(b"\x00" * pad)
            self.n += pad
        sec[name] = self.n
        b = np.ascontiguousarray(arr).tobytes()
        self._parts.append(b)
        self.n += len(b)

    def bytes(self) -> bytes:
        pad = (-self.n) % 4
        if pad:
            self._parts.append(b"\x00" * pad)
            self.n += pad
        return b"".join(self._parts)


def _q(values, quant: float) -> np.ndarray:
    """float → 絶対量子(int64)。round-half-away は使わず numpy の rint(偶数丸め)で決定論。"""
    a = np.asarray(values, dtype=np.float64)
    return np.rint(a / quant).astype(np.int64)


def _pack_header(magic: bytes, header: dict) -> tuple[bytes, int]:
    """固定ヘッダ + JSON を作る。戻り値 (bytes, payload 開始オフセット)。"""
    js = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    pad = (-(16 + len(js))) % 8
    head = magic + struct.pack("<II", FORMAT_VERSION, len(js) + pad) + js + b" " * pad
    return head, len(head)


def read_header(blob: bytes) -> tuple[dict, int]:
    """blob の JSON ヘッダと payload 開始オフセットを返す。"""
    magic = blob[:8]
    if magic not in (MAGIC_TRACKS, MAGIC_UE):
        raise ValueError(f"tracks_bin: 未知の magic {magic!r}")
    ver, jlen = struct.unpack("<II", blob[8:16])
    if ver != FORMAT_VERSION:
        raise ValueError(f"tracks_bin: 未対応 version {ver}(期待 {FORMAT_VERSION})")
    header = json.loads(blob[16:16 + jlen].decode("utf-8"))
    return header, 16 + jlen


# ------------------------------------------------------------------ 統計(1 パス目)
def _scan(data: dict, pos_coords: int, traffic_kind: str, quant: float) -> dict:
    """座標の量子レンジ・状態値集合・件数を 1 パスで集める(チャンク分割の見積りに使う)。"""
    P = data.get("positions") or []
    M = data.get("moves") or []
    TR = data.get("traffic") or []
    lo = [None] * pos_coords
    hi = [None] * pos_coords
    states: set = set()
    n_moves = n_move_pts = n_segs = n_seg_pts = 0

    def upd(qq: np.ndarray, dims) -> None:
        for k, d in enumerate(dims):
            a = qq[:, k]
            if a.size == 0:
                continue
            mn, mx = int(a.min()), int(a.max())
            lo[d] = mn if lo[d] is None else min(lo[d], mn)
            hi[d] = mx if hi[d] is None else max(hi[d], mx)

    for row in P:
        if not row:
            continue
        a = np.asarray(row, dtype=np.float64)
        upd(_q(a[:, :pos_coords], quant), range(pos_coords))
        states.update(np.unique(np.rint(a[:, pos_coords]).astype(np.int64)).tolist())
    for row in M:
        for mv in row or []:
            if not mv:
                continue
            n_moves += 1
            pts = mv[1] or []
            n_move_pts += len(pts)
            if pts:
                upd(_q(pts, quant), (0, 1))
    for tr in TR:
        segs = (tr.get("segs") or []) if traffic_kind == "dict" else (tr or [])
        n_segs += len(segs)
        for seg in segs:
            n_seg_pts += len(seg)
            if seg:
                upd(_q(seg, quant), (0, 1))
    return {"lo": lo, "hi": hi, "states": sorted(states),
            "n_moves": n_moves, "n_move_pts": n_move_pts,
            "n_segs": n_segs, "n_seg_pts": n_seg_pts}


def _origin_and_check(st: dict, pos_coords: int, quant: float) -> list:
    """データ由来の origin(量子単位の整数)を決め、int16 に収まるか検査する。"""
    origin = []
    for d in range(pos_coords):
        lo, hi = st["lo"][d], st["hi"][d]
        if lo is None:
            origin.append(0)
            continue
        o = (lo + hi) // 2
        if (lo - o) < I16_MIN or (hi - o) > I16_MAX:
            span = (hi - lo) * quant
            raise ValueError(
                f"tracks_bin: 軸 {d} の座標レンジ {span:.1f} m が量子 {quant} m の int16 "
                f"({(I16_MAX - I16_MIN) * quant:.1f} m)に収まらない。quant を上げるか座標を分割する。")
        origin.append(int(o))
    return origin


# ------------------------------------------------------------------ 符号化
def _chunk_steps(n_steps: int, n_agents: int, pos_coords: int, st: dict,
                 state_bytes: int, chunk_bytes: int) -> int:
    """1 チャンクの step 数を chunk_bytes 目標から決める(データのみに依存=決定論)。"""
    if n_steps <= 0:
        return 1
    per_step = (n_agents * (pos_coords * 2 + state_bytes)
                + (st["n_moves"] * 9 + st["n_move_pts"] * 4
                   + st["n_segs"] * 4 + st["n_seg_pts"] * 4) / n_steps
                + 4 * 3)
    k = int(max(1, chunk_bytes // max(1, int(per_step))))
    return int(min(max(k, 1), n_steps))


def _encode(data: dict, magic: bytes, kind: str, pos_coords: int, traffic_kind: str,
            outer: dict, quant: float, chunk_bytes: int,
            chunk_steps: int | None = None) -> tuple[bytes, dict]:
    P = data.get("positions") or []
    M = data.get("moves") or []
    TR = data.get("traffic") or []
    n_steps = len(P)
    n_agents = len(data.get("ids") or (P[0] if P else []))

    st = _scan(data, pos_coords, traffic_kind, quant)
    origin = _origin_and_check(st, pos_coords, quant)
    palette = st["states"]
    state_dtype = "u2" if len(palette) <= 65536 else "u4"
    np_state = np.uint16 if state_dtype == "u2" else np.uint32
    state_bytes = 2 if state_dtype == "u2" else 4
    pal_arr = np.asarray(palette, dtype=np.int64)      # 昇順 → searchsorted で索引化

    ks = (int(chunk_steps) if chunk_steps
          else _chunk_steps(n_steps, n_agents, pos_coords, st, state_bytes, chunk_bytes))
    ks = max(1, min(ks, max(n_steps, 1)))

    payload = _Payload()
    chunks: list[dict] = []
    org = np.asarray(origin, dtype=np.int64)
    for c0 in range(0, max(n_steps, 0), ks):
        c1 = min(c0 + ks, n_steps)
        ns = c1 - c0
        sec: dict = {}
        off0 = payload.n + ((-payload.n) % 4)

        # --- 位置 + 状態
        pos = np.zeros((ns, n_agents, pos_coords), dtype=np.int16)
        state = np.zeros((ns, n_agents), dtype=np_state)
        for li in range(ns):
            row = P[c0 + li]
            if not row:
                continue
            a = np.asarray(row, dtype=np.float64)
            pos[li] = (_q(a[:, :pos_coords], quant) - org).astype(np.int16)
            state[li] = np.searchsorted(
                pal_arr, np.rint(a[:, pos_coords]).astype(np.int64)).astype(np_state)
        payload.add("pos", pos, sec)
        payload.add("state", state, sec)

        # --- 移動ポリライン
        mvoff = np.zeros(ns + 1, dtype=np.uint32)
        mvag: list[int] = []
        mvmode: list[int] = []
        mvpo: list[int] = [0]
        mvpts: list = []
        for li in range(ns):
            row = M[c0 + li] if (c0 + li) < len(M) else None
            for i, mv in enumerate(row or []):
                if not mv:
                    continue
                mvag.append(i)
                mvmode.append(int(mv[0]))
                pts = mv[1] or []
                if pts:
                    mvpts.append(_q(pts, quant)[:, :2] - org[:2])
                mvpo.append(mvpo[-1] + len(pts))
            mvoff[li + 1] = len(mvag)
        payload.add("mvoff", mvoff, sec)
        payload.add("mvag", np.asarray(mvag, dtype=np.uint32), sec)
        payload.add("mvmode", np.asarray(mvmode, dtype=np.uint8), sec)
        payload.add("mvpo", np.asarray(mvpo, dtype=np.uint32), sec)
        payload.add("mvpts",
                    (np.vstack(mvpts).astype(np.int16) if mvpts
                     else np.zeros((0, 2), dtype=np.int16)), sec)

        # --- 背景交通
        troff = np.zeros(ns + 1, dtype=np.uint32)
        trn = np.zeros(ns, dtype=np.int32)
        trpo: list[int] = [0]
        trpts: list = []
        nseg = 0
        for li in range(ns):
            tr = TR[c0 + li] if (c0 + li) < len(TR) else None
            if traffic_kind == "dict":
                segs = (tr or {}).get("segs") or []
                trn[li] = int((tr or {}).get("n", 0) or 0)
            else:
                segs = tr or []
            for seg in segs:
                if seg:
                    trpts.append(_q(seg, quant)[:, :2] - org[:2])
                trpo.append(trpo[-1] + len(seg))
                nseg += 1
            troff[li + 1] = nseg
        payload.add("troff", troff, sec)
        payload.add("trn", trn, sec)
        payload.add("trpo", np.asarray(trpo, dtype=np.uint32), sec)
        payload.add("trpts",
                    (np.vstack(trpts).astype(np.int16) if trpts
                     else np.zeros((0, 2), dtype=np.int16)), sec)

        end = payload.n
        chunks.append({
            "i": len(chunks), "s0": c0, "s1": c1,
            "off": off0, "len": end - off0,
            "n_moves": len(mvag), "n_move_pts": mvpo[-1],
            "n_segs": nseg, "n_seg_pts": trpo[-1],
            "sec": {k: sec[k] - off0 for k in _SECTIONS},
        })

    binary = {
        "format": "shibuya-tracks-bin", "kind": kind,
        "magic": magic.decode("ascii"), "version": FORMAT_VERSION,
        "quant": quant, "pos_coords": pos_coords, "traffic_kind": traffic_kind,
        "origin_q": origin, "origin": [o * quant for o in origin],
        "max_abs_error_m": quant / 2.0,
        "state_palette": palette, "state_dtype": state_dtype,
        "n_steps": n_steps, "n_agents": n_agents,
        "chunk_steps": ks, "n_chunks": len(chunks), "chunks": chunks,
        "totals": {"n_moves": st["n_moves"], "n_move_pts": st["n_move_pts"],
                   "n_segs": st["n_segs"], "n_seg_pts": st["n_seg_pts"]},
    }
    header = dict(outer)
    header["binary"] = binary
    head, _off = _pack_header(magic, header)
    return head + payload.bytes(), header


# ------------------------------------------------------------------ 復号
def _chunk_arrays(blob: bytes, base: int, c: dict, binary: dict) -> dict:
    sec = c["sec"]
    ns = c["s1"] - c["s0"]
    na = binary["n_agents"]
    pc = binary["pos_coords"]
    o = base + c["off"]
    st_np = np.uint16 if binary["state_dtype"] == "u2" else np.uint32

    def rd(name, dtype, count):
        return np.frombuffer(blob, dtype=dtype, count=count, offset=o + sec[name])

    return {
        "pos": rd("pos", "<i2", ns * na * pc).reshape(ns, na, pc),
        "state": rd("state", st_np, ns * na).reshape(ns, na),
        "mvoff": rd("mvoff", "<u4", ns + 1),
        "mvag": rd("mvag", "<u4", c["n_moves"]),
        "mvmode": rd("mvmode", np.uint8, c["n_moves"]),
        "mvpo": rd("mvpo", "<u4", c["n_moves"] + 1),
        "mvpts": rd("mvpts", "<i2", c["n_move_pts"] * 2).reshape(-1, 2),
        "troff": rd("troff", "<u4", ns + 1),
        "trn": rd("trn", "<i4", ns),
        "trpo": rd("trpo", "<u4", c["n_segs"] + 1),
        "trpts": rd("trpts", "<i2", c["n_seg_pts"] * 2).reshape(-1, 2),
    }


def _decode(blob: bytes, ndigits: int) -> tuple[dict, dict]:
    header, base = read_header(blob)
    binary = header["binary"]
    quant = binary["quant"]
    pc = binary["pos_coords"]
    org = np.asarray(binary["origin_q"], dtype=np.int64)
    palette = binary["state_palette"]
    traffic_kind = binary["traffic_kind"]

    positions: list = []
    moves: list = []
    traffic: list = []

    def deq(q: np.ndarray, dims) -> np.ndarray:
        return (q.astype(np.int64) + org[list(dims)]) * quant

    for c in binary["chunks"]:
        A = _chunk_arrays(blob, base, c, binary)
        ns = c["s1"] - c["s0"]
        xy = deq(A["pos"], range(pc))
        for li in range(ns):
            row = []
            for i in range(binary["n_agents"]):
                v = [round(float(xy[li, i, d]), ndigits) for d in range(pc)]
                v.append(int(palette[int(A["state"][li, i])]))
                row.append(v)
            positions.append(row)

            mrow: list = [None] * binary["n_agents"]
            for r in range(int(A["mvoff"][li]), int(A["mvoff"][li + 1])):
                p0, p1 = int(A["mvpo"][r]), int(A["mvpo"][r + 1])
                pts = deq(A["mvpts"][p0:p1], (0, 1))
                mrow[int(A["mvag"][r])] = [
                    int(A["mvmode"][r]),
                    [[round(float(p[0]), ndigits), round(float(p[1]), ndigits)] for p in pts]]
            moves.append(mrow)

            segs = []
            for r in range(int(A["troff"][li]), int(A["troff"][li + 1])):
                p0, p1 = int(A["trpo"][r]), int(A["trpo"][r + 1])
                pts = deq(A["trpts"][p0:p1], (0, 1))
                segs.append([[round(float(p[0]), ndigits), round(float(p[1]), ndigits)]
                             for p in pts])
            traffic.append({"n": int(A["trn"][li]), "segs": segs}
                           if traffic_kind == "dict" else segs)
    return header, {"positions": positions, "moves": moves, "traffic": traffic}


# ------------------------------------------------------------------ tracks(scene3d)
def encode_tracks(tracks: dict, quant: float = QUANT_M,
                  chunk_bytes: int = DEFAULT_CHUNK_BYTES,
                  chunk_steps: int | None = None) -> tuple[bytes, dict]:
    """scene3d/tracks.json 相当の dict → (tracks.bin のバイト列, ヘッダ dict)。"""
    outer = {
        "meta": tracks.get("meta", {}),
        "agents": tracks.get("agents", []),
        "ids": tracks.get("ids", []),
        "sim_min": tracks.get("sim_min", []),
    }
    return _encode(tracks, MAGIC_TRACKS, "tracks", 2, "dict", outer,
                   quant, chunk_bytes, chunk_steps)


def decode_tracks(blob: bytes) -> dict:
    """tracks.bin → scene3d/tracks.json 相当の dict(数値は完全復元。-0.0 は 0.0 になる)。"""
    header, arrays = _decode(blob, 1)
    return {"meta": header["meta"], "agents": header["agents"], "ids": header["ids"],
            "positions": arrays["positions"], "moves": arrays["moves"],
            "traffic": arrays["traffic"], "sim_min": header["sim_min"]}


# ------------------------------------------------------------------ sim_ue(UE 再生)
def encode_sim_ue(sim_ue: dict, quant_uu: float | None = None,
                  chunk_bytes: int = DEFAULT_CHUNK_BYTES,
                  chunk_steps: int | None = None) -> tuple[bytes, dict]:
    """scene3d/sim_ue.json 相当の dict → (sim_ue.bin のバイト列, ヘッダ dict)。

    quant_uu 既定は「0.05 m 相当の uu」= 0.05 * meta.scale_m_to_uu(既定 scale=100 → 5.0 uu)。
    """
    meta = sim_ue.get("meta", {})
    if quant_uu is None:
        quant_uu = QUANT_M * float(meta.get("scale_m_to_uu", 100.0) or 100.0)
    outer = {"meta": meta, "agents": sim_ue.get("agents", []),
             "ids": sim_ue.get("ids", []), "sim_min": meta.get("sim_min", [])}
    return _encode(sim_ue, MAGIC_UE, "sim_ue", 3, "list", outer,
                   quant_uu, chunk_bytes, chunk_steps)


def decode_sim_ue(blob: bytes) -> dict:
    """sim_ue.bin → scene3d/sim_ue.json 相当の dict(量子 quant_uu/2 の丸めが入る)。"""
    header, arrays = _decode(blob, 1)
    return {"meta": header["meta"], "agents": header["agents"], "ids": header["ids"],
            "positions": arrays["positions"], "moves": arrays["moves"],
            "traffic": arrays["traffic"]}


def load_tracks(scene_dir) -> dict | None:
    """scene3d/ から tracks dict を得る。tracks.json 優先・無ければ tracks.bin を復号。

    --no-tracks-json で書かれたランを、既存の消費側(export_ue / blender_import)が
    従来どおり読めるようにするための互換口。どちらも無ければ None。
    """
    from pathlib import Path as _P
    d = _P(scene_dir)
    jp = d / "tracks.json"
    if jp.exists():
        return json.loads(jp.read_text(encoding="utf-8"))
    bp = d / "tracks.bin"
    if bp.exists():
        return decode_tracks(bp.read_bytes())
    return None


# ------------------------------------------------------------------ ビューア用サイドカー
def chunk_sidecars(blob: bytes) -> list[tuple[str, str]]:
    """tracks.bin → [(ファイル名, JS 本文)]。base64 を JSONP 風の関数呼び出しで包む。

    `fetch` は file:// で CORS により失敗するため、既存の plateau_mesh.js と同じ
    `<script src>` 方式(make_viewer3d.py:915-920 の前例)を踏襲する。1 チャンク = 1 ファイルで、
    ビューアは再生位置に応じて script 要素を動的挿入して取り込む(= 遅延ロード)。
    """
    header, base = read_header(blob)
    out = []
    for c in header["binary"]["chunks"]:
        name = f"chunk_{c['i']:04d}.js"
        b64 = base64.b64encode(blob[base + c["off"]: base + c["off"] + c["len"]]).decode("ascii")
        out.append((name, f'__TRACKS_CHUNK__({c["i"]},"{b64}");\n'))
    return out
