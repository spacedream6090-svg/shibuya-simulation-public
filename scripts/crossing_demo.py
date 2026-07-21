"""歩行者信号ゲートの検収映像(オフライン静止画レンダ・第38バッチ P2)。

viz/sfm.py の SFM コア + SignalGate(信号ゲート)を合成人工群衆で駆動し、
  赤フレーム(縁石=curb に群衆が滞留)→ 青フレーム(一斉横断=スクランブルの見せ場)
の 2 枚を matplotlib(Agg=ヘッドレス)で描き出す。表示崩れ事件の教訓に従い、
【自分の目で確認できる静止画】を runs/_crossing_demo/ に出力する。

決定論: 位置・半径は固定、noise=0、位相は絶対時刻の純関数。乱数を一切引かない。

使い方:
    PYTHONIOENCODING=utf-8 python scripts/crossing_demo.py
        # → runs/_crossing_demo/red.png, green.png, phase.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "viz"))
from sfm import Crowd, SignalGate  # noqa: E402

# 代表周期(docs/research/pedestrian-signals.md §2.3)。スクランブル=全方向同相。
CYCLE_S, GREEN_S, FLASH_S = 140.0, 37.0, 10.0
HALF = 14.0            # スクランブル横断ボックスの半幅 [m](縁石=curb がこの外周)
V0 = 1.30             # 希望速度 [m/s](Helbing の代表値近傍)

# 位相オフセット: t=25s で青が点灯するように framing(赤=t<25、青=t>25)。
#   green 窓 = phase ∈ [0, 47)。offset = -25 mod 140 = 115 → phase(25)=0(青点灯)。
OFFSET_S = 115.0
T_RED = 10.0          # 赤フレームの時刻(全員 curb 待機)
T_GREEN = 46.0        # 青フレームの時刻(横断中=ボックス内を流れる)
T_END = 62.0
DT = 0.1


def build_scramble_agents():
    """4 直進 + 2 対角の合成群衆(全員 curb で青待ち)。決定論(乱数なし)。"""
    starts: list[tuple[float, float]] = []
    goals: list[tuple[float, float]] = []
    lat = np.linspace(-10.0, 10.0, 7)         # 各辺 7 人、横方向に整列
    for a in lat:                              # 4 直進: 南→北 / 北→南 / 西→東 / 東→西
        starts += [(float(a), -HALF), (float(a), HALF), (-HALF, float(a)), (HALF, float(a))]
        goals += [(float(a), HALF), (float(a), -HALF), (HALF, float(a)), (-HALF, float(a))]
    diag = np.linspace(-6.0, 6.0, 4)          # 2 対角: SW→NE / SE→NW(スクランブルの象徴)
    for d in diag:
        starts += [(-HALF, -HALF + float(d)), (HALF, -HALF + float(d))]
        goals += [(HALF, HALF - float(d)), (-HALF, HALF - float(d))]
    pos = np.array(starts, dtype=np.float64)
    goal = np.array(goals, dtype=np.float64)
    n = pos.shape[0]
    v0 = np.full(n, V0)
    radius = np.full(n, 0.30)                  # 固定(rng を引かない=決定論)
    vel = np.zeros((n, 2), dtype=np.float64)   # 静止で curb 待機。青で駆動項が動かす
    return pos, vel, goal, v0, radius


def run(gate: SignalGate | None):
    """gate で駆動し、T_RED / T_GREEN 時点の (pos, active) スナップショットを返す。"""
    pos, vel, goal, v0, radius = build_scramble_agents()
    n = pos.shape[0]
    crowd = Crowd(pos=pos, vel=vel, goal=goal, v0=v0, radius=radius,
                  active=np.zeros(n, dtype=bool), noise=0.0)
    released = np.zeros(n, dtype=bool)
    done = np.zeros(n, dtype=bool)
    n_sub = int(round(T_END / DT))
    snaps: dict[str, tuple] = {}
    targets = {"red": T_RED, "green": T_GREEN}
    for i in range(n_sub + 1):
        t = i * DT
        entered = ~done                        # 全員最初から curb に居る(t_enter=0)
        if gate is not None:
            if gate.can_cross(t):
                released |= entered
            active = entered & released
        else:
            active = entered
        crowd.active = active
        for key, tt in targets.items():
            if key not in snaps and t >= tt - 1e-9:
                snaps[key] = (crowd.pos.copy(), active.copy(),
                              gate.can_cross(t) if gate else True)
        crowd.step(DT)
        done |= (active & crowd.arrived())
    return crowd, snaps


def render(snaps: dict, out_dir: Path, gate: SignalGate) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")                      # ヘッドレス(表示なし=静止画のみ)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    # ラベルは ASCII のみ(和文フォント非依存=表示崩れ回避。図の意味は色と配置で伝える)
    labels = {"red": ("RED - pedestrians wait at curb", "#e5484d"),
              "green": ("GREEN - scramble (all cross at once)", "#30a46c")}
    for key in ("red", "green"):
        pos, active, go = snaps[key]
        title, col = labels[key]
        fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=110)
        ax.set_facecolor("#0a0e14")
        fig.patch.set_facecolor("#0d1117")
        # 横断ボックス(スクランブル)
        ax.add_patch(Rectangle((-HALF, -HALF), 2 * HALF, 2 * HALF,
                               fill=False, ec="#8b949e", lw=1.2, ls="--"))
        # 縁石(curb)ライン
        for s in (-HALF, HALF):
            ax.plot([-HALF, HALF], [s, s], color="#30363d", lw=2)
            ax.plot([s, s], [-HALF, HALF], color="#30363d", lw=2)
        # 信号灯(青/赤)を四隅に
        for (sx, sy) in [(-HALF, -HALF), (-HALF, HALF), (HALF, -HALF), (HALF, HALF)]:
            ax.scatter([sx], [sy], s=120, c=col, edgecolors="white",
                       linewidths=0.8, zorder=5)
        # 歩行者(在圏 active=移動中は明色、待機=くすみ色)
        mv = active
        ax.scatter(pos[mv, 0], pos[mv, 1], s=42, c="#58a6ff",
                   edgecolors="#0a0e14", linewidths=0.6, zorder=4, label="crossing")
        ax.scatter(pos[~mv, 0], pos[~mv, 1], s=42, c="#8b949e",
                   edgecolors="#0a0e14", linewidths=0.6, zorder=4, label="waiting")
        ax.set_xlim(-HALF - 6, HALF + 6)
        ax.set_ylim(-HALF - 6, HALF + 6)
        ax.set_aspect("equal")
        ax.set_title(f"{title}   (in-box={int(mv.sum())} / can_cross={go})",
                     color="#e6edf3", fontsize=12)
        ax.tick_params(colors="#8b949e")
        for spine in ax.spines.values():
            spine.set_color("#30363d")
        ax.legend(loc="upper right", facecolor="#161b22",
                  edgecolor="#30363d", labelcolor="#e6edf3", fontsize=9)
        p = out_dir / f"{key}.png"
        fig.savefig(p, facecolor=fig.patch.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

    # 位相タイムライン(赤/青/点滅の帯 + 2 フレーム位置)
    fig, ax = plt.subplots(figsize=(7.2, 2.0), dpi=110)
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0a0e14")
    ts = np.arange(0, T_END, 0.2)
    for t in ts:
        p = gate.phase(t)
        c = ("#30a46c" if p < gate.green_s else
             "#f2cc60" if p < gate.green_s + gate.flash_s else "#e5484d")
        ax.axvspan(t, t + 0.2, color=c, alpha=0.9)
    for tt, lbl in [(T_RED, "red"), (T_GREEN, "green")]:
        ax.axvline(tt, color="white", lw=1.5)
        ax.text(tt, 1.05, lbl, color="white", ha="center", fontsize=9)
    ax.set_xlim(0, T_END); ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("t [s]  (green=WALK / yellow=flashing / red=DONT WALK)",
                  color="#8b949e")
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    p = out_dir / "phase.png"
    fig.savefig(p, facecolor=fig.patch.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    paths.append(p)
    return paths


def main() -> int:
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    gate = SignalGate(CYCLE_S, GREEN_S, FLASH_S, offset_s=OFFSET_S)
    _crowd, snaps = run(gate)
    out_dir = _ROOT / "runs" / "_crossing_demo"
    paths = render(snaps, out_dir, gate)
    red_wait = int((~snaps["red"][1]).sum())
    green_move = int(snaps["green"][1].sum())
    print(f"信号ゲート demo: cycle={CYCLE_S:.0f}s 青={GREEN_S:.0f}s 点滅={FLASH_S:.0f}s")
    print(f"  赤フレーム t={T_RED:.0f}s: 待機={red_wait}人(横断可={snaps['red'][2]})")
    print(f"  青フレーム t={T_GREEN:.0f}s: 横断中={green_move}人(横断可={snaps['green'][2]})")
    for p in paths:
        print(f"  -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
