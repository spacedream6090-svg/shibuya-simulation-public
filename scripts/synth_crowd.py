"""スクランブル領域の微視歩行者軌跡の【オフライン合成】(案a)。

承認済み計画「群衆物理・案a=オフライン合成」(docs/plans/engine-architecture.md §4 P1-2)。
シミュ本体(src/society)・conf・L1 を一切触らず、既存ランの L1 `move_segment` を
【読み出し専用】で後処理して、スクランブル交差点の円領域だけを Social Force Model
(viz/sfm.py・自前最小実装)で微視軌跡に合成する。R1/ゴールデンは完全無風。

────────────────────────────────────────────────────────────────────────
手順(docs/plans の指定どおり):
  ① L1 の move_segment(pts ポリライン)から、指定円領域を通過する区間を step ごとに抽出。
  ② 入口点・出口点・メゾ所要時間から  希望速度 v0 = 区間距離 / メゾ所要時間  を決める。
     → メゾの平均所要時間を保存し、SFM は「揉まれ方(減速・回避・レーン形成)」だけを担う。
     (メゾは pts を弧長一定速度で 1 step=600s かけて辿る = viz/make_viewer.alongPath と同じ
      パラメータ化。したがって v0 = 区間弧長 /(区間の弧長割合×600s)= その区間のメゾ実効速度。)
  ③ 同一 step(=1 窓)の集団を SFM でサブステップ積分(既定 dt=0.1s)。各体は自分の
     入口時刻(=pts 弧長割合 f_in×600s・実データ由来)に湧き、出口点で退場する。
  ④ 決定論: 窓ごとに seed = sha256(ラン名 + '|' + 窓番号)由来で固定(半径抽選・積分順)。

出力:
  runs/<name>/panel/crowd_tracks.parquet  … (window, t_s, agent_id, x, y)
  runs/<name>/panel/crowd_demo.html       … 自己完結・外部参照なしの canvas 2D アニメ

誠実性(限界の明示):
  - 通過者 0 の窓は「データ不足」として記録し、合成しない。
  - 壁・障害物なし(領域境界=円の外に出たら退場するだけ)。建物外壁・車道縁石・
    信号は持ち込まない。SFM は viz/sfm.py が省略する接触項(k, κ)を含まないため、
    低〜中密度の揉まれ方の再現に用途を限定する。詳細は viz/sfm.py の docstring。
  - このランの総エージェント数は現実のスクランブル(1 青 1,000 人以上)より遥かに少なく、
    瞬間密度は現実規模ではない。SFM 機構の視覚実証であって群集規模の再現ではない。
────────────────────────────────────────────────────────────────────────

使い方:
  python scripts/synth_crowd.py runs/<name> [--center-x X --center-y Y]
      [--radius-m 40] [--start-step S0 --end-step S1] [--dt 0.1]
      [--out-dt 0.5] [--out PATH]
  --center-x/--center-y 省略時はランの地図(config.yaml world.map)から
    「スクランブル交差点」ノード(poi=square/原点最近傍)を自動導出する。
    導出できなければ両引数を必須にする(既定座標は捏造しない)。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "viz"))
from sfm import Crowd, RADIUS_MIN, RADIUS_MAX  # noqa: E402

STEP_SECONDS = 600.0    # 1 step = 10 分 = 600 s(conf world.modes / L1 step 定義)
_MIN_PORTION_M = 0.5    # これ未満の区間弧長は「かすっただけ」として通過に数えない
_MIN_V0 = 0.05          # 希望速度の下限 [m/s](退化した極小区間を除外)


# ─────────────────────────────────────────────── 地図から中心座標を導出
def _load_yaml(path: Path) -> dict:
    try:
        import yaml
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except ModuleNotFoundError:
        from omegaconf import OmegaConf
        return OmegaConf.to_container(OmegaConf.load(str(path)), resolve=True)


def resolve_map_path(run_dir: Path) -> Path | None:
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        return None
    cfg = _load_yaml(cfg_path)
    mp = Path(cfg.get("world", {}).get("map", "data/shibuya_osm.json"))
    return mp if mp.is_absolute() else (_ROOT / mp)


def derive_center(map_path: Path | None) -> tuple[float, float] | None:
    """地図 JSON から「スクランブル交差点」中心を導出。

    poi=='square' か名前に「スクランブル」を含むノードのうち原点(0,0)最近傍を採る。
    見つからなければ None(=呼び出し側が --center-x/--center-y を必須にする)。
    """
    if map_path is None or not map_path.exists():
        return None
    city = json.loads(map_path.read_text(encoding="utf-8"))
    cands = []
    for n in city.get("nodes", []):
        if not isinstance(n, dict):
            continue
        name = str(n.get("name", ""))
        if n.get("poi") == "square" or "スクランブル" in name:
            try:
                cands.append((float(n["x"]), float(n["y"]), name))
            except (KeyError, TypeError, ValueError):
                continue
    if not cands:
        # POI レイヤも試す
        for p in city.get("pois", []):
            if "スクランブル交差点" in str(p.get("name", "")):
                cands.append((float(p["x"]), float(p["y"]), p.get("name", "")))
    if not cands:
        return None
    cands.sort(key=lambda c: c[0] ** 2 + c[1] ** 2)
    return cands[0][0], cands[0][1]


# ─────────────────────────────────────────────── 円領域の通過区間抽出
def _cumlen(pts: list) -> tuple[list, float]:
    cum = [0.0]
    for i in range(1, len(pts)):
        seg = math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
        cum.append(cum[-1] + seg)
    return cum, cum[-1]


def _point_at_arclen(pts: list, cum: list, s: float) -> tuple[float, float]:
    """弧長 s の位置を線形補間で返す(viz/make_viewer.alongPath と同じ幾何)。"""
    total = cum[-1]
    if total <= 0:
        return pts[0][0], pts[0][1]
    s = max(0.0, min(s, total))
    for i in range(1, len(pts)):
        if s <= cum[i]:
            seg = cum[i] - cum[i - 1]
            g = (s - cum[i - 1]) / seg if seg > 0 else 0.0
            return (pts[i - 1][0] + (pts[i][0] - pts[i - 1][0]) * g,
                    pts[i - 1][1] + (pts[i][1] - pts[i - 1][1]) * g)
    return pts[-1][0], pts[-1][1]


def circle_crossing(pts: list, cx: float, cy: float, r: float):
    """polyline pts が円(中心 (cx,cy)・半径 r)を通過する区間を返す。

    戻り値 None(通過なし)または dict:
        entry, exit  … 入口点・出口点 (x,y)   [円境界上、ただし端点が円内ならその端点]
        portion_dist … 入口→出口の弧長 [m]
        portion_time … その区間のメゾ所要時間 [s]  = (弧長割合)×600s
        f_in, f_out  … 入口/出口の弧長割合 [0,1]。make_viewer.alongPath と同じく
                       弧長=step内時刻なので、入口時刻 = f_in×600s(実データ由来)。
    複数回出入りする稀な経路は「最初の入口〜最後の出口」を1通過とみなす(簡略化・明示)。
    """
    if len(pts) < 2:
        return None
    cum, total = _cumlen(pts)
    if total <= 0:
        return None
    r2 = r * r
    inside_arc: list[float] = []

    def _inside(px, py):
        return (px - cx) ** 2 + (py - cy) ** 2 <= r2

    # 円内にある頂点の弧長
    for i, (px, py) in enumerate(pts):
        if _inside(px, py):
            inside_arc.append(cum[i])
    # 各セグメントと円周の交点(弧長)
    for i in range(1, len(pts)):
        x0, y0 = pts[i - 1]
        x1, y1 = pts[i]
        dx, dy = x1 - x0, y1 - y0
        a = dx * dx + dy * dy
        if a == 0:
            continue
        fx, fy = x0 - cx, y0 - cy
        b = 2.0 * (fx * dx + fy * dy)
        c = fx * fx + fy * fy - r2
        disc = b * b - 4.0 * a * c
        if disc < 0:
            continue
        sq = math.sqrt(disc)
        for t in ((-b - sq) / (2 * a), (-b + sq) / (2 * a)):
            if 0.0 <= t <= 1.0:
                inside_arc.append(cum[i - 1] + t * math.sqrt(a))
    if not inside_arc:
        return None
    s_in, s_out = min(inside_arc), max(inside_arc)
    portion_dist = s_out - s_in
    if portion_dist < _MIN_PORTION_M:
        return None
    portion_time = (portion_dist / total) * STEP_SECONDS
    entry = _point_at_arclen(pts, cum, s_in)
    exit_ = _point_at_arclen(pts, cum, s_out)
    return {"entry": entry, "exit": exit_,
            "portion_dist": portion_dist, "portion_time": portion_time,
            "f_in": s_in / total, "f_out": s_out / total}


# ─────────────────────────────────────────────── L1 読み出し
def load_crossings(run_dir: Path, cx: float, cy: float, r: float,
                   start_step: int, end_step: int) -> dict:
    """step ごとの通過エージェント一覧を返す:
        { step: [ {agent_id, entry, exit, v0, portion_time, portion_dist}, ... ] }
    walk モードの move_segment のみ対象(車・自転車は歩行者群集ではない)。
    """
    import pyarrow.parquet as pq
    cols = ["step", "agent_id", "kind", "payload"]
    try:
        table = pq.read_table(run_dir / "l1_events.parquet", columns=cols,
                              filters=[("step", ">=", start_step),
                                       ("step", "<=", end_step),
                                       ("kind", "==", "move_segment")])
    except Exception:
        # filters 非対応環境のフォールバック
        table = pq.read_table(run_dir / "l1_events.parquet", columns=cols)
    by_step: dict[int, list] = {}
    for row in table.to_pylist():
        if row["kind"] != "move_segment":
            continue
        step = row["step"]
        if not (start_step <= step <= end_step):
            continue
        payload = row.get("payload")
        if not payload:
            continue
        p = json.loads(payload)
        if p.get("mode") != "walk":
            continue
        pts = p.get("pts")
        if not pts or len(pts) < 2:
            continue
        cr = circle_crossing(pts, cx, cy, r)
        if cr is None:
            continue
        if cr["portion_time"] <= 0:
            continue
        v0 = cr["portion_dist"] / cr["portion_time"]
        if v0 < _MIN_V0:
            continue
        by_step.setdefault(step, []).append({
            "agent_id": int(row["agent_id"]),
            "entry": cr["entry"], "exit": cr["exit"],
            "v0": v0, "portion_time": cr["portion_time"],
            "portion_dist": cr["portion_dist"],
            # 入口時刻は実データの弧長割合から(rng で散らさない=捏造回避)
            "t_enter": cr["f_in"] * STEP_SECONDS,
        })
    return by_step


# ─────────────────────────────────────────────── SFM で 1 窓を合成
def _window_seed(run_name: str, window: int) -> int:
    h = hashlib.sha256(f"{run_name}|{window}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big")


def simulate_window(run_name: str, window: int, crossers: list,
                    dt: float, out_dt: float) -> dict:
    """1 step(=1 窓)の通過集団を SFM 積分し、軌跡サンプルと統計を返す。

    決定論: seed = sha256(run_name+'|'+window)。エージェントは agent_id 昇順に整列
    してから半径抽選・積分するため、順序も固定。
    """
    crossers = sorted(crossers, key=lambda c: c["agent_id"])
    n = len(crossers)
    seed = _window_seed(run_name, window)
    rng = np.random.default_rng(seed)

    aid = np.array([c["agent_id"] for c in crossers], dtype=np.int64)
    entry = np.array([c["entry"] for c in crossers], dtype=np.float64)     # (N,2)
    exit_ = np.array([c["exit"] for c in crossers], dtype=np.float64)      # (N,2)
    v0 = np.array([c["v0"] for c in crossers], dtype=np.float64)           # (N,)
    ptime = np.array([c["portion_time"] for c in crossers], dtype=np.float64)
    radius = rng.uniform(RADIUS_MIN, RADIUS_MAX, size=n)                   # 順序固定
    # 入口時刻: 各体の円到達時刻 = f_in×600s。pts は 1 step 全体を弧長一定で辿る
    #   (=make_viewer.alongPath の弧長パラメータ化)ため、弧長割合 f_in がそのまま
    #   step 内の相対到達時刻になる。実データ由来で決め、rng では散らさない(捏造回避)。
    #   これで同一窓内の co-presence(出会い)は実データの時間構造を反映する。
    t_enter = np.array([c["t_enter"] for c in crossers], dtype=np.float64)

    # 初期速度 = v0 × 入口→出口 単位ベクトル
    dirv = exit_ - entry
    dnorm = np.linalg.norm(dirv, axis=1)
    unit = np.zeros_like(dirv)
    nz = dnorm > 1e-9
    unit[nz] = dirv[nz] / dnorm[nz, None]
    vel0 = unit * v0[:, None]

    crowd = Crowd(pos=entry, vel=vel0, goal=exit_, v0=v0, radius=radius,
                  active=np.zeros(n, dtype=bool), rng=rng, noise=0.0)

    done = np.zeros(n, dtype=bool)
    t_exit_meso = t_enter + ptime
    t_sim = float(t_exit_meso.max()) + 5.0 if n else 0.0
    n_sub = int(round(t_sim / dt))
    out_stride = max(1, int(round(out_dt / dt)))

    samples: list[tuple] = []   # (t_s, agent_id, x, y)
    for i in range(n_sub + 1):
        t = i * dt
        active = (t >= t_enter) & (~done)
        crowd.active = active
        if i % out_stride == 0 and np.any(active):
            idx = np.nonzero(active)[0]
            for j in idx:
                samples.append((round(t, 3), int(aid[j]),
                                round(float(crowd.pos[j, 0]), 3),
                                round(float(crowd.pos[j, 1]), 3)))
        crowd.step(dt)
        arr = crowd.arrived()
        done |= (active & arr)
        done |= (t > t_enter + 3.0 * ptime)   # 安全弁(詰まった体の打ち切り)

    return {
        "window": window,
        "n_crossers": n,
        "n_points": len(samples),
        "meso_seconds": float(ptime.sum()),
        "mean_v0": float(v0.mean()) if n else 0.0,
        "t_sim": t_sim,
        "samples": samples,
        "seed": seed,
    }


# ─────────────────────────────────────────────── 出力(parquet / html)
def write_parquet(out_path: Path, windows: list) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq
    win, ts, aid, xs, ys = [], [], [], [], []
    for w in windows:
        for (t, a, x, y) in w["samples"]:
            win.append(w["window"]); ts.append(t); aid.append(a)
            xs.append(x); ys.append(y)
    table = pa.table({
        "window": pa.array(win, type=pa.int32()),
        "t_s": pa.array(ts, type=pa.float32()),
        "agent_id": pa.array(aid, type=pa.int32()),
        "x": pa.array(xs, type=pa.float32()),
        "y": pa.array(ys, type=pa.float32()),
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 決定論(byte 一致)重視: 圧縮なし・固定バージョンで書く
    pq.write_table(table, out_path, compression="none", version="2.6",
                   write_statistics=False)
    return table.num_rows


def build_html(run_name: str, cx: float, cy: float, r: float,
               windows: list, empty_windows: list, start_step: int,
               end_step: int, dt: float, out_dt: float) -> str:
    """自己完結の crowd_demo.html を生成(外部参照なし・canvas 2D・heatmap.html と同流儀)。"""
    # 窓データを軽量化して埋め込む: agent ごとに [t, x, y] の列
    win_js = []
    for w in windows:
        agents: dict[int, list] = {}
        for (t, a, x, y) in w["samples"]:
            agents.setdefault(a, []).append([round(t, 2), round(x, 2), round(y, 2)])
        win_js.append({
            "window": w["window"], "n": w["n_crossers"],
            "mean_v0": round(w["mean_v0"], 3), "t_sim": round(w["t_sim"], 1),
            "agents": [{"id": a, "p": pts} for a, pts in sorted(agents.items())],
        })
    data = {
        "run": run_name, "cx": round(cx, 2), "cy": round(cy, 2),
        "r": round(r, 2), "start_step": start_step, "end_step": end_step,
        "dt": dt, "out_dt": out_dt,
        "windows": win_js,
        "empty": empty_windows,
    }
    return _HTML_TEMPLATE.replace("__RUN__", run_name).replace(
        "__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")))


_HTML_TEMPLATE = r"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>スクランブル群衆合成(SFM) __RUN__</title>
<style>
  :root{--bg:#0d1117;--panel:#161b22;--ink:#e6edf3;--dim:#9aa4b2;--line:#30363d;--acc:#f59f00;}
  *{box-sizing:border-box;margin:0;}
  body{background:var(--bg);color:var(--ink);
    font-family:system-ui,-apple-system,"Hiragino Sans","Yu Gothic UI","Segoe UI",sans-serif;}
  header{padding:12px 16px;border-bottom:1px solid var(--line);}
  h1{font-size:15px;font-weight:600;}
  .sub{color:var(--dim);font-size:12px;margin-top:3px;}
  #wrap{display:flex;flex-wrap:wrap;gap:14px;padding:14px 16px;}
  #cv{background:#0a0e14;border:1px solid var(--line);border-radius:10px;max-width:100%;}
  .side{min-width:250px;flex:1;}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin-bottom:12px;}
  .row{display:flex;align-items:center;gap:10px;margin:8px 0;}
  input[type=range]{width:100%;accent-color:var(--acc);}
  button{background:#21262d;color:var(--ink);border:1px solid var(--line);border-radius:8px;
    padding:6px 12px;font-size:13px;cursor:pointer;}
  button:hover{background:#30363d;}
  select{background:#21262d;color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:5px 8px;}
  .big{font-size:22px;font-weight:700;}
  .note{color:var(--dim);font-size:11px;line-height:1.6;}
  table{border-collapse:collapse;font-size:12px;width:100%;}
  td,th{border-bottom:1px solid var(--line);padding:3px 6px;text-align:right;}
  th{color:var(--dim);font-weight:500;}
</style></head><body>
<header><h1>スクランブル交差点 群衆物理(SFM)オフライン合成 — __RUN__</h1>
<div class="sub" id="sub"></div></header>
<div id="wrap">
  <canvas id="cv" width="640" height="640"></canvas>
  <div class="side">
    <div class="card">
      <div class="row"><button id="play">▶ 再生</button>
        <select id="wsel"></select>
        <div class="big" id="tlab" style="margin-left:auto"></div></div>
      <div class="row"><input type="range" id="slider" min="0" value="0"></div>
      <div class="row"><label class="note">再生速度 ×<span id="spd">4</span></label>
        <input type="range" id="speed" min="1" max="20" value="4"></div>
    </div>
    <div class="card">
      <div class="note" style="margin-bottom:6px">この窓(step)の合成</div>
      <table><tbody id="stat"></tbody></table>
    </div>
    <div class="card note">
      <b>案a=オフライン合成</b>。L1 の walk move_segment を読み出し、円領域(中心
      (<span id="cxy"></span>)・半径 <span id="rr"></span> m)を通過する歩行者だけを
      Social Force Model(Helbing&amp;Molnár 1995 / Helbing2000・自前最小実装 viz/sfm.py)で
      微視軌跡に合成。希望速度 v0=区間距離/メゾ所要時間(平均所要時間を保存)。円到達
      時刻は pts の弧長割合(=step内時刻)から実データ由来で決める。
      <b>限界</b>: 壁・障害物なし(円境界のみ)/ 物理接触項(k,κ)省略 /
      総人数はランに依存し現実のスクランブル規模(1 青 1,000 人以上)ではない
      =SFM 機構の視覚実証。外部 CDN 非依存の単体 HTML。
    </div>
    <div class="card note" id="emptybox"></div>
  </div>
</div>
<script>
const D=__DATA__;
const cv=document.getElementById('cv'), g=cv.getContext('2d');
const PAD=28;
document.getElementById('sub').textContent =
  `step ${D.start_step}–${D.end_step} / 窓数 ${D.windows.length}(通過あり)/ dt=${D.dt}s・出力 ${D.out_dt}s刻み`;
document.getElementById('cxy').textContent = D.cx+', '+D.cy;
document.getElementById('rr').textContent = D.r;
const eb=document.getElementById('emptybox');
eb.textContent = D.empty.length? ('データ不足(通過0)の窓: '+D.empty.join(', ')) : '通過0の窓なし';

// world→canvas: 円を中心に、半径 r を画面に収める(Y は北=上)
const scale=(Math.min(cv.width,cv.height)-PAD*2)/(2*D.r);
function tf(x,y){ return [cv.width/2 + (x-D.cx)*scale, cv.height/2 - (y-D.cy)*scale]; }

const wsel=document.getElementById('wsel');
D.windows.forEach((w,i)=>{ const o=document.createElement('option');
  o.value=i; o.textContent=`窓 step ${w.window}(${w.n}人)`; wsel.appendChild(o); });
let wi=0, t=0, playing=false, speed=4;
// 既定は通過人数が最大の窓
if(D.windows.length){ wi=D.windows.reduce((b,w,i,a)=> w.n>a[b].n? i:b, 0); wsel.value=wi; }

const slider=document.getElementById('slider');
const tlab=document.getElementById('tlab');
const speedR=document.getElementById('speed'), spd=document.getElementById('spd');
speedR.oninput=()=>{ speed=+speedR.value; spd.textContent=speed; };

function curWin(){ return D.windows[wi]; }
function setWin(i){ wi=i; t=0; slider.max=Math.max(1,Math.round(curWin().t_sim)); statTable(); draw(); }
wsel.onchange=()=>setWin(+wsel.value);
slider.oninput=()=>{ t=+slider.value; draw(); };

function posAt(agent,tt){   // [t,x,y] サンプルを線形補間、範囲外は null
  const p=agent.p; if(!p.length) return null;
  if(tt<p[0][0]-1e-6 || tt>p[p.length-1][0]+1e-6) return null;
  for(let i=1;i<p.length;i++){ if(tt<=p[i][0]){
    const a=p[i-1], b=p[i], seg=b[0]-a[0], f=seg>0?(tt-a[0])/seg:0;
    return [a[1]+(b[1]-a[1])*f, a[2]+(b[2]-a[2])*f]; } }
  const last=p[p.length-1]; return [last[1],last[2]];
}
function draw(){
  g.clearRect(0,0,cv.width,cv.height);
  // 領域円
  g.beginPath(); const c=tf(D.cx,D.cy); g.arc(c[0],c[1],D.r*scale,0,Math.PI*2);
  g.strokeStyle='rgba(245,159,0,.5)'; g.lineWidth=1.5; g.stroke();
  g.fillStyle='rgba(245,159,0,.04)'; g.fill();
  // 中心十字
  g.strokeStyle='rgba(255,255,255,.14)'; g.beginPath();
  g.moveTo(c[0]-8,c[1]); g.lineTo(c[0]+8,c[1]); g.moveTo(c[0],c[1]-8); g.lineTo(c[0],c[1]+8); g.stroke();
  const w=curWin(); let live=0;
  for(const ag of w.agents){
    // 軌跡(薄い残像)
    g.strokeStyle='rgba(120,180,255,.20)'; g.lineWidth=1; g.beginPath(); let started=false;
    for(const s of ag.p){ if(s[0]>t) break; const q=tf(s[1],s[2]);
      if(!started){ g.moveTo(q[0],q[1]); started=true; } else g.lineTo(q[0],q[1]); }
    if(started) g.stroke();
    const pos=posAt(ag,t); if(!pos) continue; live++;
    const q=tf(pos[0],pos[1]);
    g.beginPath(); g.arc(q[0],q[1],4,0,Math.PI*2);
    g.fillStyle='#7cc4ff'; g.fill();
    g.strokeStyle='#0a0e14'; g.lineWidth=1; g.stroke();
  }
  tlab.textContent = `t=${t.toFixed(0)}s / 在圏 ${live}人`;
  slider.value=t;
}
function statTable(){
  const w=curWin();
  document.getElementById('stat').innerHTML =
    `<tr><th>step(窓)</th><td>${w.window}</td></tr>`+
    `<tr><th>通過人数</th><td>${w.n}</td></tr>`+
    `<tr><th>平均希望速度</th><td>${w.mean_v0} m/s</td></tr>`+
    `<tr><th>窓の長さ</th><td>${w.t_sim} s</td></tr>`;
}
document.getElementById('play').onclick=function(){
  playing=!playing; this.textContent=playing?'⏸ 停止':'▶ 再生';
  let last=performance.now();
  const tick=(now)=>{ if(!playing)return;
    const d=(now-last)/1000; last=now; t+=d*speed*D.out_dt/D.out_dt; // 実時間×speed
    if(t>curWin().t_sim){ t=0; }
    draw(); requestAnimationFrame(tick); };
  if(playing){ last=performance.now(); requestAnimationFrame(tick); }
};
if(D.windows.length){ setWin(wi); } else {
  document.getElementById('tlab').textContent='合成窓なし';
}
</script></body></html>"""


# ─────────────────────────────────────────────── main
def synth(run_dir: Path, cx: float, cy: float, r: float,
          start_step: int, end_step: int, dt: float, out_dt: float,
          out_path: Path) -> dict:
    run_name = run_dir.name
    by_step = load_crossings(run_dir, cx, cy, r, start_step, end_step)
    windows, empty_windows = [], []
    for step in range(start_step, end_step + 1):
        crossers = by_step.get(step, [])
        if not crossers:
            empty_windows.append(step)   # データ不足(通過0)
            continue
        windows.append(simulate_window(run_name, step, crossers, dt, out_dt))

    n_rows = write_parquet(out_path, windows)
    html = build_html(run_name, cx, cy, r, windows, empty_windows,
                      start_step, end_step, dt, out_dt)
    html_path = out_path.parent / "crowd_demo.html"
    html_path.write_text(html, encoding="utf-8")

    total_cross = sum(w["n_crossers"] for w in windows)
    total_pts = sum(w["n_points"] for w in windows)
    total_meso = sum(w["meso_seconds"] for w in windows)
    return {
        "run": run_name, "center": (cx, cy), "radius": r,
        "start_step": start_step, "end_step": end_step,
        "n_windows": len(windows), "n_empty": len(empty_windows),
        "total_crossers": total_cross, "total_points": total_pts,
        "total_meso_seconds": total_meso, "n_rows": n_rows,
        "parquet": out_path, "html": html_path,
    }


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(description="スクランブル領域の微視軌跡オフライン合成(案a)")
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--center-x", type=float, default=None)
    ap.add_argument("--center-y", type=float, default=None)
    ap.add_argument("--radius-m", type=float, default=40.0)
    ap.add_argument("--start-step", type=int, default=None)
    ap.add_argument("--end-step", type=int, default=None)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--out-dt", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    run_dir = args.run_dir if args.run_dir.is_absolute() else (_ROOT / args.run_dir)
    if not (run_dir / "l1_events.parquet").exists():
        print(f"error: {run_dir}/l1_events.parquet が見つかりません", file=sys.stderr)
        return 2

    cx, cy = args.center_x, args.center_y
    if cx is None or cy is None:
        derived = derive_center(resolve_map_path(run_dir))
        if derived is None:
            print("error: 中心座標を地図から導出できません。--center-x と --center-y を"
                  "指定してください(既定座標は捏造しません)。", file=sys.stderr)
            return 2
        cx, cy = derived
        print(f"  中心を地図から導出: center=({cx:.1f}, {cy:.1f})")

    start_step = 0 if args.start_step is None else args.start_step
    end_step = start_step + 11 if args.end_step is None else args.end_step

    out_path = args.out
    if out_path is None:
        out_path = run_dir / "panel" / "crowd_tracks.parquet"
    elif not out_path.is_absolute():
        out_path = _ROOT / out_path

    res = synth(run_dir, cx, cy, args.radius_m, start_step, end_step,
                args.dt, args.out_dt, out_path)
    print(f"  中心=({cx:.1f},{cy:.1f}) 半径={args.radius_m}m  step {start_step}–{end_step}")
    print(f"  合成窓={res['n_windows']}  データ不足窓={res['n_empty']}")
    print(f"  総通過={res['total_crossers']}人  軌跡点={res['total_points']}  "
          f"メゾ総所要={res['total_meso_seconds']:.0f}s")
    print(f"  -> {res['parquet']}  ({res['n_rows']} rows)")
    print(f"  -> {res['html']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
