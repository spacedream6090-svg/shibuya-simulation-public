"""マクロ⇄ミクロ観察の統合検収 B9: 「観察がシムを変えない」3点の機械検証。

第58バッチ B0–B8 で入った屋内ミクロ観察(IndoorSpace/遷移駆動 SFM/indoor_tracks サイドカー・
会社台帳 org_ledger サイドカー・2D/3D ズーム・在館観測)が、動力学(ラン出力=L1)へ一切
影響しない構造であることを機械で固定する。設計原則③(記録と動力学の構造分離)の回帰ガード。

① tracks サブトグルの ON/OFF で L1 が **バイト不変**(空間軌跡サイドカーの有無は動力学に無関係)。
   ※ 既存 test_indoor_engine.test_tracks_off_l1_unchanged は L1 イベント列(in-memory)一致を担保。
     本 file はそれを L1 parquet の **バイト級**へ引き上げて独立に固定する。
② ビューア/観測サイドカー生成(make_viewer/make_viewer3d/build_occupancy)を「挟んでも」ラン出力が
   不変: ラン A → L1 バイト控え → 事後処理を A へ適用 → A の L1 が不変(事後処理は純シンク)→
   同一 conf の別ラン B の L1 が A と一致(事後処理がランへ状態を漏らさない)。
③ **動力学コードが観測サイドカーのバッファを読まない**テキスト検査: src/society/engine・world の全 .py で
   観測サイドカー(indoor_tracks/org_ledger_sc)への参照が許容メソッド(add_samples/add_contacts/
   add_rows/flush_segment/finalize/_resumed)だけであり、内部バッファ(.samples/.contacts/.rows 等)を
   読む向きが存在しないことを固定する(書き込み向き=動力学→observer のみが正)。

全経路 mock(実 LLM 不使用)。既定挙動不変(R1)。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pyarrow.parquet as pq

from society.config import load_config, save_config
from society.engine.simulation import Simulation

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src" / "society"

# 屋内エンジン全 ON(tracks も含む)。個別テストが上書きする。
_ON = ["indoor.enabled=true", "indoor.markov.enabled=true",
       "indoor.sfm.enabled=true", "indoor.meeting.enabled=true",
       "indoor.tracks.enabled=true"]


def _cfg(name: str, n_steps: int, n_agents: int, on=True, **ov):
    dot = ["run.seed=42", f"run.n_agents={n_agents}", f"run.n_steps={n_steps}",
           f"run.name={name}", "model.backend=mock"]
    if on:
        dot += list(_ON)
    dot += [f"{k}={v}" for k, v in ov.items()]
    return load_config(dot)


# --------------------------------------------------------------------------- #
# ① tracks サブトグルの ON/OFF で L1 が バイト不変
# --------------------------------------------------------------------------- #
def test_tracks_toggle_l1_byte_identical(tmp_path):
    """屋内全 ON のまま tracks を ON/OFF しても l1_events.parquet が **バイト一致**。

    space_move(L1)は動力学が記録し tracks 有無に依らない。tracks は秒スケール軌跡/遭遇の
    サイドカー(indoor_tracks_*.parquet)を「足すだけ」で L1 のバイトへ現れない=記録と動力学の分離。"""
    on = Simulation(_cfg("t_on", 24, 30), out_dir=tmp_path / "t_on")
    on.run()
    off = Simulation(_cfg("t_off", 24, 30, **{"indoor.tracks.enabled": "false"}),
                     out_dir=tmp_path / "t_off")
    off.run()
    l1_on = (tmp_path / "t_on" / "l1_events.parquet").read_bytes()
    l1_off = (tmp_path / "t_off" / "l1_events.parquet").read_bytes()
    assert l1_on == l1_off, "tracks の ON/OFF で L1 parquet がバイト不一致(記録が動力学へ漏れた)"
    # tracks ON のみ軌跡サイドカーが出る(=OFF は L1 に何も足さないことの裏返し)
    assert (tmp_path / "t_on" / "indoor_tracks_samples.parquet").exists()
    assert not (tmp_path / "t_off" / "indoor_tracks_samples.parquet").exists()


# --------------------------------------------------------------------------- #
# ② ビューア/サイドカー生成を挟んでもラン出力(L1)が不変
# --------------------------------------------------------------------------- #
def _postprocess(run_dir: Path) -> None:
    """make_viewer / make_viewer3d / build_occupancy を run_dir へ適用(事後処理=純シンク)。

    いずれも l1_events.parquet を「読むだけ」で HTML/占有サイドカーを書く。ここで L1 が
    書き換われば①の分離が崩れる=②の検証点。viz/ と scripts/ を import path へ足して呼ぶ。"""
    sys.path.insert(0, str(REPO_ROOT / "viz"))
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import make_viewer as mv        # noqa: E402
    import make_viewer3d as mv3      # noqa: E402
    import build_occupancy as bo     # noqa: E402

    # build_occupancy: l1(+屋内サイドカー)→ occupancy.parquet(L1 は不変)
    bo.write_occupancy(run_dir, run_dir / "occupancy.parquet")
    # make_viewer: viewer.html/dashboard.html を書く(l1 は読むだけ)。sys.argv 経由。
    old_argv = sys.argv
    try:
        sys.argv = ["make_viewer.py", str(run_dir), "--no-traffic", "--indoor-moves"]
        mv.main()
    finally:
        sys.argv = old_argv
    # make_viewer3d: scene3d/ + viewer3d.html を書く(l1 は読むだけ)。argv 引数で呼ぶ。
    mv3.main([str(run_dir), "--no-traffic"])


def test_postprocessing_does_not_change_run_output(tmp_path):
    """ラン→(ビューア/サイドカー生成)→同一 conf の別ラン、で L1 がバイト一致。

    (a) 事後処理は run_dir の l1_events.parquet を改変しない(純シンク)。
    (b) 事後処理を経ても、同一 conf の再ランが最初のランと L1 バイト一致(状態を漏らさない・決定論)。"""
    cfg_a = _cfg("run_a", 6, 20)
    a = Simulation(cfg_a, out_dir=tmp_path / "run_a")
    a.run()
    save_config(a.cfg, tmp_path / "run_a")     # ビューアが読む config.yaml(通常 run.py が書く)
    l1a_before = (tmp_path / "run_a" / "l1_events.parquet").read_bytes()

    _postprocess(tmp_path / "run_a")           # ← 事後処理を挟む

    l1a_after = (tmp_path / "run_a" / "l1_events.parquet").read_bytes()
    assert l1a_after == l1a_before, "事後処理(ビューア生成)がランの L1 parquet を書き換えた"

    b = Simulation(_cfg("run_b", 6, 20), out_dir=tmp_path / "run_b")
    b.run()
    l1b = (tmp_path / "run_b" / "l1_events.parquet").read_bytes()
    assert l1b == l1a_before, "事後処理を経た後の再ランが最初のランと L1 バイト不一致"
    # 事後処理が実際に走った証拠(サイドカー/HTML が出ている=空振りでない)
    assert (tmp_path / "run_a" / "occupancy.parquet").exists()
    assert (tmp_path / "run_a" / "viewer.html").exists()
    assert (tmp_path / "run_a" / "viewer3d.html").exists()


# --------------------------------------------------------------------------- #
# ③ 動力学コードが観測サイドカーのバッファを読まない(テキスト検査)
# --------------------------------------------------------------------------- #
# 観測サイドカー(sim に据わる観測専用オブジェクト)。動力学は「書き込み(add_*/flush/finalize)」の
# 向きだけで触れてよく、内部バッファ(.samples/.contacts/.rows/_n_*_flushed 等)を読む向きは禁止。
# スコープ注記: 本検査は B3/B4 で新設した**私有バッファを持つ観測サイドカー**(indoor_tracks/org_ledger_sc)
# に限定する。logger は共有 L1 ログであり、動力学が interstitial(S2)/work.service 用に len(logger.events)・
# events[idx:] を読むのは意図的な既存設計(私有観測バッファの逆流ではない)ため対象外=正しく除外している。
# IF-E2(第97バッチ)で finance_sc(部門別残高の日次サイドカー)を同じ規律の下に置く。
_SIDECARS = ("indoor_tracks", "org_ledger_sc", "finance_sc")
# 動力学が観測サイドカーへ許される操作(=書き込み配線の allowlist)。
_ALLOWED_ATTRS = frozenset({"add_samples", "add_contacts", "add_rows",
                            "flush_segment", "finalize", "_resumed"})


def _dyn_files():
    files: list[Path] = []
    for d in (SRC / "engine", SRC / "world"):
        files += sorted(d.glob("*.py"))
    return files


def _strip_comments(text: str) -> str:
    """行コメント(# 以降)を除去(素朴だが本検査には十分=コメント中の擬似アクセス誤検知を避ける)。"""
    out = []
    for line in text.splitlines():
        h = line.find("#")
        out.append(line if h < 0 else line[:h])
    return "\n".join(out)


def _sidecar_exprs(text: str) -> list[str]:
    """本文中で観測サイドカーに束縛される式(直接形 + ローカル束縛変数)を集める。

    直接形 = sim.<sidecar> / self.<sidecar>。束縛 = `x = getattr(sim|self, "<sidecar>"...)` や
    `x = sim|self.<sidecar>` の左辺 x。indoor.py の無関係な `tracks`(conf dict)は左辺束縛が
    サイドカー由来でないので**混入しない**(ファイル毎に束縛を発見する設計)。"""
    names = "|".join(_SIDECARS)
    exprs = [f"sim.{s}" for s in _SIDECARS] + [f"self.{s}" for s in _SIDECARS]
    for m in re.finditer(rf'(\w+)\s*=\s*getattr\(\s*(?:sim|self)\s*,\s*["\'](?:{names})["\']', text):
        exprs.append(m.group(1))
    for m in re.finditer(rf'(\w+)\s*=\s*(?:sim|self)\.(?:{names})\b(?!\s*=)', text):
        exprs.append(m.group(1))
    return exprs


def test_dynamics_never_reads_observer_sidecar_buffers():
    """engine/world の動力学が観測サイドカーの内部バッファを読まない(許容メソッドのみで触れる)。

    書き込み向き(動力学 → observer)だけが正。逆向き(observer バッファを動力学が read)を禁止。
    許容 allowlist(add_samples/add_contacts/add_rows/flush_segment/finalize/_resumed)以外の属性
    アクセスがサイドカー束縛式に現れたら FAIL。"""
    violations: list[str] = []
    touched = 0
    for f in _dyn_files():
        text = _strip_comments(f.read_text(encoding="utf-8"))
        exprs = _sidecar_exprs(text)
        if not exprs:
            continue
        for expr in exprs:
            for m in re.finditer(re.escape(expr) + r"\.([A-Za-z_]\w*)", text):
                touched += 1
                attr = m.group(1)
                if attr not in _ALLOWED_ATTRS:
                    line = text[:m.start()].count("\n") + 1
                    violations.append(f"{f.name}:{line}: {expr}.{attr} (禁止=バッファ読み取りの疑い)")
    assert touched > 0, "サイドカー参照が1件も見つからない(検査が空振り=配線の場所が変わった?)"
    assert not violations, "動力学が観測サイドカーの内部バッファを読んでいる:\n" + "\n".join(violations)


def test_write_wiring_is_present():
    """②の裏返し=書き込み配線(observer が動力学の一時状態を受け取る呼び出し)が現に存在する。

    _phase_indoor が tracks.add_samples/add_contacts を、org 日次締めが sc.add_rows を呼ぶ=
    正しい向き(動力学→observer)の配線が消えていない回帰ガード。"""
    sched = (SRC / "engine" / "scheduler.py").read_text(encoding="utf-8")
    assert "add_samples(" in sched and "add_contacts(" in sched
    assert "add_rows(" in sched
    sim = (SRC / "engine" / "simulation.py").read_text(encoding="utf-8")
    assert ".finalize()" in sim and ".flush_segment()" in sim
